"""Ground-state observables and the printed report.

Everything ``calc_gs`` computes *about* a ground state it has already found: single-particle density
matrices, thermal averages, occupation statistics, entanglement, spin/orbital Casimirs, correlation and
screening diagnostics, static susceptibilities, and the report that prints them. Kept apart from
:mod:`impurityModel.ed.groundstate`, which only decides *which* state is the ground state.
"""

import copy
import io
import os
import sys
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import numpy as np

from impurityModel.ed import config
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_restrictions import get_effective_restrictions
from impurityModel.ed.basis_transcription import comm_support_filter
from impurityModel.ed.block_structure import get_equivalent_blocks, print_block_structure
from impurityModel.ed.cipsi_solver import _degeneracy_tol
from impurityModel.ed.degeneracy import MANIFOLD_TOL, degenerate_groups
from impurityModel.ed.density_matrices import build_density_matrices
from impurityModel.ed.gs_report_model import OrbitalNaming, assemble_report
from impurityModel.ed.gs_report_render import render_summary
from impurityModel.ed.gs_statistics import (
    _boltzmann_weights,
    assemble_entanglement,
    assemble_gs_statistics,
    channel_indices_of,
    compute_impurity_rdm,
    entanglement_partials,
    gs_statistics_partials,
    print_gs_statistics,
    save_gs_statistics,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.mpi_root import on_root
from impurityModel.ed.observables import (
    block_group_labels,
    casimir_operator,
    casimir_to_quantum_number,
    compute_correlation_diagnostics,
    compute_energy_decomposition,
    compute_magnetic_summary,
    compute_screening_diagnostics,
    compute_state_summary,
    get_Sz_from_rho_pairs,
    is_j_sharp,
    make_impurity_casimir_operators,
    make_spin_operators,
    manifold_observable_matrices,
    manifold_square_weights,
    manifold_values_from_matrices,
    print_correlation_diagnostics,
    print_expectation_values,
    print_impurity_orbital_groups,
    print_screening_diagnostics,
    print_state_summary,
    print_thermal_expectation_values,
    spin_correlation_operator,
    spin_curie_susceptibility,
    thermal_observable_value,
)
from impurityModel.ed.spin_pairs import spin_layout as resolve_spin_layout
from impurityModel.ed.symmetries import extract_tensors
from impurityModel.ed.utils import (
    V_DETAIL,
    V_SUMMARY,
    Reporter,
    format_index_ranges,
    matrix_print,
    print_density_matrix_summary,
    report_banner,
    report_rule,
)


@dataclass(frozen=True)
class ReportFiles:
    """Where the ground-state report is persisted: a full text report and the JSON data.

    ``None`` for a path means that file is not written. :meth:`for_label` follows the ``save_Greens_function``
    convention (``<prefix>-<label>.<ext>`` in a directory, the working directory by default).
    """

    text: object = None
    json: object = None
    unit: object = None

    @classmethod
    def for_label(cls, label, directory=None):
        """``ground_state-<label>.txt`` and ``.json`` in ``directory`` (default: the working directory)."""
        base = os.path.join(os.fspath(directory) if directory is not None else "", f"ground_state-{label}")
        return cls(text=base + ".txt", json=base + ".json", unit=str(label))

    @classmethod
    def coerce(cls, files):
        """``files`` as a :class:`ReportFiles`: ``None`` -> nothing written, ``str`` -> a JSON path (legacy)."""
        if isinstance(files, cls):
            return files
        return cls(json=None if files is None else os.fspath(files))


def convergence_summary(convergence):
    """One line on how well the ground state is determined, from a ``convergence_report`` (or ``None``).

    Pure: reads only the replicated report dict, so it is safe on any rank and takes no collective.
    Names the PT2 residual against its tolerance, what limited the expansion, and -- when the sector
    walk recorded one -- the winning sector's margin over its nearest neighbour.
    """
    if not convergence:
        return "convergence: not recorded"
    res = convergence.get("residual_pt2")
    tol = convergence.get("e_pt2_tol")
    parts = ["converged" if convergence.get("converged") else "NOT converged"]
    if res is not None and tol is not None:
        parts.append(f"residual PT2 {res:.2e} (tol {tol:.0e})")
    if convergence.get("limited_by"):
        parts.append(f"limited by {limited_by_words(convergence['limited_by'])}")
    diag = convergence.get("sector_diagnostics")
    if diag and diag.get("winner_total") is not None:
        margin = diag.get("margin")
        tail = "undecided" if diag.get("undecided") else ("margin n/a" if margin is None else f"margin {margin:.2g}")
        parts.append(f"sector N_imp={diag['winner_total']} ({tail})")
    return "convergence: " + "; ".join(parts)


_LIMITED_BY_WORDS = {
    "cap": "the determinant cap (truncation_threshold)",
    "memory": "the memory guard",
    "incomplete_manifold": "missing states of the lowest level",
    "multiplet_split": "a split multiplet",
    "uncertified": "an uncertified selection",
    "de2_min": "the PT2 selection threshold",
}


def limited_by_words(token):
    """A ``limited_by`` token in words for a report line; an unknown token is shown as it is."""
    return _LIMITED_BY_WORDS.get(token, token)


def degraded_warnings(convergence):
    """The reasons a ground state deserves a stderr line, from a ``convergence_report`` (``None``: none).

    Only the events a user must not miss: an unconverged expansion (a determinant cap or the memory guard
    withholding PT2 mass) and an undecided sector choice (margin < 1). Pure and replicated-data only.
    """
    if not convergence:
        return []
    out = []
    if not convergence.get("converged", True):
        res = convergence.get("residual_pt2")
        why = limited_by_words(convergence.get("limited_by") or "unknown")
        extra = f" (residual PT2 {res:.2e})" if res is not None else ""
        out.append(f"ground-state expansion is NOT converged, limited by {why}{extra}")
    diag = convergence.get("sector_diagnostics") or {}
    if diag.get("undecided"):
        out.append(f"the N_imp={diag.get('winner_total')} sector choice is undecided (margin < 1)")
    return out


def _detached(exc):
    """A copy of ``exc`` that references no frame (no traceback, no chained exceptions)."""
    try:
        out = copy.copy(exc)
    except Exception:
        out = RuntimeError(f"{type(exc).__name__}: {exc}")
    out.__traceback__, out.__cause__, out.__context__ = None, None, None
    return out


def manifold_tol(es, slater_weight_min=None):
    """The energy tolerance at which the report groups ``es`` into degenerate manifolds.

    ``max(MANIFOLD_TOL, solver tolerance)``: the report never groups tighter than the solve did, so a manifold the
    solver could not split is never split into separate "states" here. ``slater_weight_min=None`` (the caller does
    not know the solver's cutoff) leaves just the report floor. Pure in the replicated ``es``.
    """
    if slater_weight_min is None:
        return MANIFOLD_TOL
    return max(MANIFOLD_TOL, _degeneracy_tol(es, slater_weight_min))


#: Operator eigenvalues closer than this are one quantum number when the sharpening diagonalises the next label inside
#: a cluster (Sz is a half-integer, S^2 and L^2 differ by whole units: this is far below either and far above rounding).
LABEL_TOL = 1e-3


@dataclass
class ManifoldBasis:
    """How the states of one member were arranged by :func:`sharpen_manifolds`.

    ``groups`` are the degenerate manifolds (ascending energy, index arrays into the state order); ``keys`` the
    labels that were diagonalised inside them, in order (``"Sz"``, ``"S2"``, ``"L2"`` or ``"J2"``); ``values[key]``
    the label's value per state in the arranged basis (``Sz`` itself, the others as Casimir eigenvalues).
    ``rotated`` is whether any manifold was rotated.
    """

    groups: list
    keys: tuple = ()
    values: dict = field(default_factory=dict)
    rotated: bool = False
    tol: float = MANIFOLD_TOL

    @property
    def two_sz(self):
        """``2 S_z`` per state as integers, or ``None`` when ``S_z`` was not measured."""
        sz = self.values.get("Sz")
        return None if sz is None else np.rint(2.0 * np.asarray(sz)).astype(int)


def _joint_diagonalise(mats):
    """Unitary ``U`` that diagonalises the Hermitian ``mats`` in turn: the first fully, each next one inside the
    clusters (eigenvalues within :data:`LABEL_TOL`) the previous ones left degenerate."""
    g = mats[0].shape[0]
    u = np.eye(g, dtype=complex)
    clusters = [list(range(g))]
    for b in mats:
        m = u.conj().T @ b @ u
        m = 0.5 * (m + m.conj().T)
        nxt = []
        for c in clusters:
            if len(c) == 1:
                nxt.append(c)
                continue
            w, v = np.linalg.eigh(m[np.ix_(c, c)])
            u[:, c] = u[:, c] @ v
            start = 0
            for i in range(1, len(c) + 1):
                if i == len(c) or w[i] - w[i - 1] > LABEL_TOL:
                    nxt.append(c[start:i])
                    start = i
        clusters = nxt
    return u


def sharpen_manifolds(blk, es, layout, casimirs, tol, comm=None, redistribute=None, enabled=True):
    """Rotate each degenerate manifold of ``blk`` to the basis that diagonalises its symmetry labels. Collective.

    The solver returns an arbitrary orthonormal basis of a degenerate manifold, so a single returned vector has no
    sharp ``S_z``, ``S^2``, ``L^2``. The labels are diagonalised inside each manifold in turn: ``2 S_z`` (when the
    spin layout conserves it), then ``S^2`` and ``L^2``; when ``2 S_z`` is not conserved (spin-orbit coupling), ``J^2``.
    ``casimirs`` is ``(l_ops, s_ops, j_ops, ...)`` from :func:`impurity_casimirs`. Energies are not touched and a
    sum over a whole manifold is unchanged. Returns ``(blk_rot, ManifoldBasis)``; ``blk_rot`` is ``blk`` itself when
    nothing needed rotating.

    Every input but ``blk`` is replicated, the key matrices are ``Allreduce``-d, and the rotation is broadcast from
    rank 0, so every rank applies the same unitary to its rows. Nothing here depends on rank-local state, and an
    empty rank takes the same collective calls with an empty block.
    """
    es = np.asarray(es, dtype=float)
    n = len(es)
    es_ref = comm.bcast(es, root=0) if comm is not None else es
    order = np.argsort(es_ref, kind="stable")
    groups = [order[b] for b in degenerate_groups(es_ref[order], tol, anchor="first")]
    info = ManifoldBasis(groups=groups, tol=tol)
    if not enabled or n == 0:
        return blk, info
    ops = []
    if layout is not None and layout.labelled and layout.pairs:
        ops.append(("Sz", make_spin_operators(layout.pairs)[2], True))
    l_ops, s_ops, j_ops = casimirs[:3] if casimirs is not None else (None, None, None)
    if l_ops is not None:
        if layout is None or layout.labelled:
            ops.append(("S2", casimir_operator(*s_ops), False))
            ops.append(("L2", casimir_operator(*l_ops), False))
        else:
            ops.append(("J2", casimir_operator(*j_ops), False))
    multi = any(len(grp) > 1 for grp in groups)
    ops = [(k, op, cheap) for k, op, cheap in ops if multi or cheap]
    if not ops:
        return blk, info
    keep = comm_support_filter([blk], comm)
    mats = {}
    for key, op, _ in ops:
        _grp, blocks = manifold_observable_matrices(
            blk,
            es_ref,
            lambda b, _op=op: _op.apply_block(b, 0, keep),
            degeneracy_tol=tol,
            comm=comm,
            redistribute=redistribute,
        )
        mats[key] = blocks
    keys = tuple(k for k, _, _ in ops)
    u_full = np.eye(n, dtype=complex)
    if multi:
        for gi, idx in enumerate(groups):
            if len(idx) > 1:
                u_full[np.ix_(idx, idx)] = _joint_diagonalise([mats[k][gi] for k in keys])
        if comm is not None:
            u_full = comm.bcast(u_full, root=0)
    values = {k: np.zeros(n) for k in keys}
    for gi, idx in enumerate(groups):
        ug = u_full[np.ix_(idx, idx)]
        for k in keys:
            values[k][idx] = np.real(np.diag(ug.conj().T @ mats[k][gi] @ ug))
    info = ManifoldBasis(groups=groups, keys=keys, values=values, rotated=multi, tol=tol)
    if multi and blk.width == n:
        blk = blk.combine_columns(u_full)
    return blk, info


def sharpen_for_report(
    Hop, ground_state_basis, es, blk, rot_to_spherical, spin_layout=None, shell_ls=None, slater_weight_min=None
):
    """:func:`sharpen_manifolds` for one member, with the layout and Casimirs resolved as the report does. Collective.

    Returns ``(blk, ManifoldBasis)``. Honours the ``GS_SHARPEN_MANIFOLDS`` knob; a failure while building the
    operators (replicated inputs, so it raises on every rank alike) leaves the block as the solver returned it.
    """
    comm = ground_state_basis.comm if ground_state_basis.is_distributed else None
    es = np.asarray(es, dtype=float)
    tol = manifold_tol(es, slater_weight_min)
    enabled = config.GS_SHARPEN_MANIFOLDS.get()
    try:
        layout = spin_layout
        if layout is None and enabled:
            layout = resolve_spin_layout(
                Hop,
                ground_state_basis.impurity_orbitals,
                ground_state_basis.bath_states,
                rot_to_spherical,
                ground_state_basis.num_spin_orbitals,
                comm=ground_state_basis.comm,
                shell_ls=shell_ls,
            )
        casimirs = impurity_casimirs(ground_state_basis, rot_to_spherical, shell_ls) if enabled else None
        return sharpen_manifolds(
            blk,
            es,
            layout,
            casimirs,
            tol,
            comm=comm,
            redistribute=ground_state_basis.redistribute_block,
            enabled=enabled,
        )
    except Exception:
        return blk, ManifoldBasis(groups=[], tol=tol)


class _LiveSource:
    """Where :func:`_evaluate` gets its per-state results when the states are in hand.

    With ``tape=None`` this is the one-shot path (:func:`report_ground_state`): every request is computed
    and returned. With a ``tape`` dict it *records* one charge-sector member (:func:`measure_member`): the same
    computations run, and each result -- or the exception it raised -- is stored under its key so that
    :class:`_ReplaySource` can hand it back after the states are gone. The keys are call-site names, never
    positions, so a member whose evaluation degraded part-way cannot shift the others' results.
    """

    def __init__(self, basis, es, psis, tau, tape=None, degeneracy_tol=MANIFOLD_TOL, manifold_basis=None):
        self.basis, self.es, self.psis, self.tau = basis, es, psis, tau
        self.manifold_basis = manifold_basis
        self.degeneracy_tol = degeneracy_tol
        self.tape = tape
        self.recording = tape is not None
        self.mov_comm = basis.comm if basis.is_distributed else None
        self.mov_redistribute = basis.redistribute_block if basis.is_distributed else None
        # Every <m|O|n> is read on the eigenstates' own support, so each apply drops the rest of O|n> as it is
        # generated. Built collectively and unconditionally, once for all of them.
        self.support_keep = comm_support_filter([psis], self.mov_comm)
        self.scales = [1.0]
        self.order = None

    def state_labels(self):
        """``(two_sz, sectors)`` per state, ``None`` where unknown."""
        return (None if self.manifold_basis is None else self.manifold_basis.two_sz), None

    def part(self, key, compute):
        """``[compute()]``: the one member's result, taped under ``key`` when recording."""
        if self.tape is None:
            return [compute()]
        try:
            value = compute()
        except Exception as exc:
            # The taped copy carries no traceback: that would pin this frame's `psis` (and the callee's
            # workspace) for as long as the tape lives, i.e. through every driver.
            self.tape[key] = ("err", _detached(exc))
            raise
        self.tape[key] = ("ok", value)
        return [value]

    def once(self, key, compute):
        return self.part(key, compute)[0]

    def stack(self, parts):
        return parts[0]

    def _measure(self, op):
        """``{"diag", "blocks"}`` of ``op``: the per-state values and the manifold matrices they come from."""
        groups, blocks = manifold_observable_matrices(
            self.psis,
            self.es,
            op,
            degeneracy_tol=self.degeneracy_tol,
            comm=self.mov_comm,
            redistribute=self.mov_redistribute,
        )
        diag = manifold_values_from_matrices(self.psis.width, groups, blocks)
        return {"diag": diag, "blocks": (groups, blocks)}

    def curie(self, group):
        """A ``curie(op)`` hook: ``(<n|O|n>, sum_m |<m|O|n>|^2 over n's manifold)`` per state."""
        calls = [0]

        def curie(op):
            key = (group, calls[0])
            calls[0] += 1
            rec = self.part(key, lambda: self._measure(op))[0]
            groups, blocks = rec["blocks"]
            return rec["diag"], manifold_square_weights(groups, blocks, len(rec["diag"]))

        return curie

    def values(self, group):
        """A ``values(op)`` hook for the ``compute_*`` observables: per-state ``<O>`` of ``op`` on the members."""
        calls = [0]

        def values(op):
            key = (group, calls[0])
            calls[0] += 1
            return self.part(key, lambda: self._measure(op))[0]["diag"]

        return values


class _ReplaySource:
    """:class:`_LiveSource`'s counterpart over the members' tapes: no states, no collective.

    ``es`` are the union's energies in ascending order of rank 0's copy (``order``); per-state arrays are
    concatenated in member order and permuted by ``order``. ``scales[k]`` is member ``k``'s share of the
    union's Boltzmann weight (its tape holds member-normalised weights). One member: no permutation and a
    scale of exactly 1, so the replay is bit-identical to the one-shot report.
    """

    recording = False
    tape = None
    psis = None
    support_keep = None

    def __init__(self, members, tau):
        self.members = list(members)
        first = self.members[0]
        # The member that kept its basis (the first visited; after the sort by energy it need not be first).
        self.lead = next(m for m in self.members if m.basis is not None)
        self.basis, self.tau = self.lead.basis, tau
        self.degeneracy_tol = max(m.degeneracy_tol for m in self.members)
        self.mov_comm = self.basis.comm if self.basis.is_distributed else None
        if len(self.members) == 1:
            self.es, self.order, self.scales = first.es, None, [1.0]
        else:
            es_cat = np.concatenate([np.asarray(m.es, dtype=float) for m in self.members])
            es_ref = np.concatenate([np.asarray(m.es_ref, dtype=float) for m in self.members])
            self.order = np.argsort(es_ref, kind="stable")
            self.es = es_cat[self.order]
            w = _boltzmann_weights(es_cat, tau)
            bounds = np.cumsum([0] + [len(m.es) for m in self.members])
            self.scales = [float(w[bounds[k] : bounds[k + 1]].sum()) for k in range(len(self.members))]

    def state_labels(self):
        """``(two_sz, sectors)`` per state in the union's order, ``None`` where a member did not record them."""
        mbs = [m.manifold_basis for m in self.members]
        two = None
        if all(mb is not None and mb.two_sz is not None for mb in mbs):
            two = np.concatenate([mb.two_sz for mb in mbs])
        sectors = None
        if len(self.members) > 1:
            sectors = np.concatenate([np.full(len(m.es), -1 if m.total is None else m.total) for m in self.members])
        if self.order is not None:
            two = None if two is None else two[self.order]
            sectors = None if sectors is None else sectors[self.order]
        return two, sectors

    def part(self, key, compute=None):
        out = []
        for m in self.members:
            entry = m.tape.get(key)
            if entry is None:
                raise RuntimeError(f"charge-sector member N_imp={m.total} did not record {key!r}")
            kind, value = entry
            if kind == "err":
                raise value
            out.append(value)
        return out

    def once(self, key, compute=None):
        kind, value = self.lead.tape.get(
            key, ("err", RuntimeError(f"member N_imp={self.lead.total} did not record {key!r}"))
        )
        if kind == "err":
            raise value
        return value

    def stack(self, parts):
        if len(parts) == 1:
            return parts[0]
        return np.concatenate([np.asarray(p) for p in parts], axis=0)[self.order]

    def values(self, group):
        calls = [0]

        def values(op):
            key = (group, calls[0])
            calls[0] += 1
            return self.stack([p["diag"] for p in self.part(key)])

        return values

    def curie(self, group):
        calls = [0]

        def curie(op):
            key = (group, calls[0])
            calls[0] += 1
            parts = self.part(key)
            col2 = [manifold_square_weights(*p["blocks"], len(p["diag"])) for p in parts]
            return self.stack([p["diag"] for p in parts]), self.stack(col2)

        return curie


@dataclass
class MemberMeasurement:
    """One charge-sector member's report data, with no state in it.

    ``tape`` holds the member's per-state results (member-normalised weights); ``es`` are its *absolute*
    energies and ``es_ref`` rank 0's copy of them, which fixes the union's order identically on every rank.
    ``basis`` is kept for the first member only (its orbital layout and communicator are all that is read).
    """

    es: np.ndarray
    es_ref: np.ndarray
    total: object
    tape: dict = field(repr=False)
    basis: object = None
    degeneracy_tol: float = MANIFOLD_TOL
    manifold_basis: object = None


def _bounds_records(src, bounds):
    """JSON-ready ``[{n_imp, max_holes, max_electrons, max_total, cap_bound}, ...]``, one per member.

    Plain ``int``/``bool`` only: the statistics write is collective and an unserialisable value once hung MPI.
    Members whose solve observed no bath orbitals (``None``) are skipped.
    """
    members = getattr(src, "members", None)
    if members is not None:
        totals = [m.total for m in members]
    else:
        occ = getattr(src.basis, "ground_state_occupation", None)
        totals = [sum(occ.values()) if occ else None]
    return [
        {
            "n_imp": None if total is None else int(total),
            "max_holes": int(b.max_holes),
            "max_electrons": int(b.max_electrons),
            "max_total": int(b.max_total),
            "cap_bound": bool(b.cap_bound),
        }
        for total, b in zip(totals, bounds)
        if b is not None
    ]


def impurity_casimirs(ground_state_basis, rot_to_spherical, shell_ls=None):
    """``(l_ops, s_ops, j_ops, per_shell_ops)`` of the impurity, or four ``None`` when it has no Casimirs.

    The operators are built in the spherical basis and rotated to the computational one via ``rot_to_spherical``. An
    impurity grouped into orbital-symmetry manifolds (eg / t2g) is aggregated into the whole shell; one that is
    genuinely not a spin-doubled l-shell has none. Replicated inputs, so the answer is the same on every rank.
    """
    impurity_indices = sorted(
        orb
        for impurity_orbital_blocks in ground_state_basis.impurity_orbitals.values()
        for block in impurity_orbital_blocks
        for orb in block
    )
    try:
        return make_impurity_casimir_operators(
            ground_state_basis.impurity_orbitals, rot_to_spherical, per_shell=True, shell_ls=shell_ls
        )
    except ValueError:
        # The impurity is grouped into orbital-symmetry manifolds (e.g. eg / t2g), none of
        # which is individually a full spin-doubled l-shell, so the per-partition build raised.
        # L/S/J are shell-*total* operators, so aggregate the manifolds into the whole shell
        # (the sorted impurity_indices, which match the single rot_to_spherical matrix) and
        # retry. Only meaningful for a single shared rotation; a dict rotation is per-shell
        # (get_spectra's multi-l case) and is already correct per partition.
        if not isinstance(rot_to_spherical, dict):
            try:
                return make_impurity_casimir_operators(
                    {0: [impurity_indices]}, rot_to_spherical, per_shell=True, shell_ls=shell_ls
                )
            except ValueError:
                # Genuinely not a spin-doubled l-shell: skip the Casimirs
                # (the rho-based <L.S>/<Lz>/<Sz> etc. still print).
                pass
    return None, None, None, None


def _evaluate(
    src,
    Hop,
    block_structure,
    rot_to_spherical,
    gs_truncation_report,
    gs_convergence_report,
    spin_layout=None,
    shell_ls=None,
    ensemble=None,
):
    """Every number the report prints, from ``src`` (live states, a recording, or a replay of recordings).

    The same code runs in all three, so the sequence of requests cannot drift between a member's recording and
    the union's replay. A recording suppresses the degrade-to-warning prints (the replay repeats them once).
    """
    ground_state_basis, psis_blk, es, tau = src.basis, src.psis, src.es, src.tau
    comm = ground_state_basis.comm
    rank = comm.rank if comm is not None else 0
    # One validated map of what every spin-orbital index is (replicated inputs: a violation raises on every rank).
    naming = OrbitalNaming.build(
        ground_state_basis.impurity_orbitals,
        ground_state_basis.bath_states,
        block_structure,
        layout=spin_layout,
        # The rotation and shell checks are input validation (done where the model is read): a mismatch there
        # makes the Casimir observables degrade below, it must not abort the report of a finished solve.
    )
    block_structure = naming.block_structure
    # The effective restrictions are printed in the ground-state-report overview below.
    effective_restrictions = src.once("restrictions", lambda: get_effective_restrictions(ground_state_basis))
    n_dets = len(ground_state_basis)
    # The bath-excitation bounds each member's solve observed (None: no bath orbitals), one per member.
    excitation_bounds = src.part("excitation_bounds", lambda: getattr(ground_state_basis, "excitation_bounds", None))
    excitation_bound_records = _bounds_records(src, excitation_bounds)
    # Single-particle density matrices, built distributed (each rank applies c_orb to its
    # local partition, redistributes, computes local inner products, then Allreduce); the
    # full rho is returned replicated on every rank. No full-state-vector gather needed.
    rhos = src.stack(src.part("rhos", lambda: build_density_matrices(ground_state_basis, psis_blk)))

    e_avg = thermal_average_scale_indep(es, es, tau)
    thermal_rho = thermal_average_scale_indep(es, rhos, tau)

    # Occupation-weight statistics of the thermal ground state, built distributed
    # (collective call); returns the dict on rank 0 and None elsewhere.
    partials = src.part("stats", lambda: gs_statistics_partials(ground_state_basis, psis_blk, es, tau))
    gs_stats = None
    if partials[0] is not None:
        gs_stats = assemble_gs_statistics(
            partials,
            src.scales,
            es,
            tau,
            thermal_rho,
            ground_state_basis.impurity_spin_orbital_indices,
            channel_indices_of(ground_state_basis),
            order=src.order,
        )
    if gs_stats is not None:
        # Record whether the ground-state basis was truncation-capped (None = not capped),
        # so it lands in the saved statistics JSON alongside the occupation weights.
        gs_stats["truncation"] = gs_truncation_report
        gs_stats["convergence"] = gs_convergence_report
        gs_stats["excitation_bounds"] = excitation_bound_records
        if ensemble is not None:
            gs_stats["charge_sector_ensemble"] = ensemble

    # Many-body impurity-bath entanglement from the impurity RDM (collective call; its
    # memory guard may skip it, deterministically on every rank). Degrades to None.
    try:
        # Per member the dense blocks are reduced to small partials at once, so none is kept while drivers run.
        partials = src.part(
            "rdm",
            lambda: entanglement_partials(
                compute_impurity_rdm(ground_state_basis, psis_blk), src.es, tau, degeneracy_tol=src.degeneracy_tol
            ),
        )
        entanglement = assemble_entanglement(partials, src.scales, src.order, es, tau)
    except Exception:
        entanglement = None
    if gs_stats is not None:
        gs_stats["entanglement"] = entanglement
    # The statistics JSON is saved after the remaining observable diagnostics below, so
    # they can be bundled into the same file.

    # Sorted (original computational) order, matching the convention of block_structure
    # (local indices over the sorted impurity orbitals) and rot_to_spherical. The
    # impurity_orbitals dict is grouped (e.g. eg then t2g), so iterating it would slice the
    # density matrix into a *reordered* basis that no longer matches rot_to_spherical /
    # block_structure — corrupting the spherical rotation and the Sz / N(Up) split.
    impurity_indices = sorted(
        orb
        for impurity_orbital_blocks in ground_state_basis.impurity_orbitals.values()
        for block in impurity_orbital_blocks
        for orb in block
    )
    impurity_ix = np.ix_(impurity_indices, impurity_indices)
    # Impurity S^2 / L^2 / J^2 and <S_imp.S_bath> are two-body observables, so they
    # need the actual eigenstates rather than the density matrix. They are evaluated
    # *distributed*: each rank block-applies the operator to its local partition,
    # redistribute_block realigns it, and manifold_observable_values Allreduces the
    # small <m|O|n> matrix (so this is a collective call with an identical result on
    # every rank — no state-vector gather). The L/S/J operators are built in the
    # spherical basis and rotated to the computational basis via rot_to_spherical
    # (symmetry-plan Phase 5).
    mov_comm = src.mov_comm
    # Every <m|O|n> below is read on the eigenstates' own support, so each apply drops the rest of
    # O|n> as it is generated. Built here, collectively and unconditionally, once for all of them.
    support_keep = src.support_keep
    s_values = l_values = j_values = None
    s2_thermal = l2_thermal = j2_thermal = None
    # Whether J is a sharp quantum number across the ground manifold (is_j_sharp); True
    # (ungated) is the safe default when the Casimirs are unavailable -- there is then no
    # signal to gate on, and the term/g_J/mu_eff rows already don't print in that case.
    j_sharp = True
    # Per-shell S^2/L^2/J^2 (Stage 4 consumes this for the per-shell report table):
    # partition -> {"l", "s2_thermal", "l2_thermal", "j2_thermal"}. Only populated when
    # there is more than one shell to distinguish -- a single shell's per-shell values
    # would just duplicate the totals computed below.
    shell_casimir = {}
    l_ops, s_ops, j_ops, per_shell_ops = impurity_casimirs(ground_state_basis, rot_to_spherical, shell_ls)
    if l_ops is not None:
        # Evaluation is deterministic and identical on every rank (manifold_observable_values
        # Allreduces), so a failure raises on all ranks together -> the try/except stays
        # collective-safe and the report degrades instead of crashing the ground-state solve.
        try:
            casimir = {}
            for name, ops in (("S", s_ops), ("L", l_ops), ("J", j_ops)):
                op2 = casimir_operator(*ops)
                vals = src.values("casimir-" + name)(lambda blk, _op=op2: _op.apply_block(blk, 0, support_keep))
                casimir[name] = (
                    np.array([casimir_to_quantum_number(v) for v in vals]),
                    thermal_observable_value(vals, es, tau),
                )
            s_values, s2_thermal = casimir["S"]
            l_values, l2_thermal = casimir["L"]
            j_values, j2_thermal = casimir["J"]
            # Without SOC, J is generically not a good quantum number: the ground
            # manifold can be energy-degenerate while its members' J differ (see
            # is_j_sharp's docstring). Gate the printed/persisted term/g_J/mu_eff on
            # this rather than reporting a single J that isn't actually well-defined.
            j_sharp = is_j_sharp(j_values, es, degeneracy_tol=src.degeneracy_tol)
        except Exception as exc:
            if rank == 0 and not src.recording:
                print(f"S^2/L^2/J^2 not reported: {exc}")
            s_values = l_values = j_values = None
            s2_thermal = l2_thermal = j2_thermal = None
            j_sharp = True
            shell_casimir = {}
    # Per-shell Casimirs run whenever there is a per-shell operator to evaluate, not only when the totals exist: a
    # partial shell has no L/S/J of its own (and so no total), but its complete neighbours still do.
    multi_shell = shell_ls is not None and len(shell_ls) > 1
    if per_shell_ops and (multi_shell or len(per_shell_ops) > 1):
        try:
            for partition, (shell_l, shell_l_ops, shell_s_ops, shell_j_ops) in per_shell_ops.items():
                entry = {"l": shell_l}
                for name, ops in (("s2", shell_s_ops), ("l2", shell_l_ops), ("j2", shell_j_ops)):
                    op2 = casimir_operator(*ops)
                    vals = src.values(f"shell-{partition}-{name}")(
                        lambda blk, _op=op2: _op.apply_block(blk, 0, support_keep)
                    )
                    entry[f"{name}_thermal"] = thermal_observable_value(vals, es, tau)
                shell_casimir[partition] = entry
        except Exception as exc:
            if rank == 0 and not src.recording:
                print(f"per-shell S^2/L^2/J^2 not reported: {exc}")
            shell_casimir = {}
    if multi_shell and rank == 0 and not src.recording:
        for a, l_a in enumerate(shell_ls):
            if a not in (per_shell_ops or {}):
                print(f"Shell {a} (l={l_a}): S^2/L^2/J^2 skipped (partial shell: some orbitals are not in the model)")
    # Kondo impurity-bath spin correlation <S_imp . S_bath>. The (dn, up) pairing is the run's
    # spin layout (spin_pairs.spin_layout): impurity pairs from the rotated S_+ or the index
    # convention, bath pairs from the index convention, equal-energy matching or chain position,
    # whichever makes H conserve S_z. When only S_z is verified (collinear / S_z-only grade) the
    # transverse (S_+-) parts are flagged and the exact longitudinal correlation is reported
    # alongside; with no pairing (SOC) it is skipped rather than reported wrong.
    sisb_values = None
    sisb_thermal = None
    sisb_z_values = None
    sisb_z_thermal = None
    sisb_z_connected = None
    sisb_pairing_approx = False
    pairing_verified = True
    sisb_pairing_note = None
    layout = spin_layout
    sisb_skip_reason = None
    spin_pairs = None
    try:
        n_orb = ground_state_basis.num_spin_orbitals
        layout = spin_layout
        if layout is None:
            layout = src.once(
                "layout",
                lambda: resolve_spin_layout(
                    Hop,
                    ground_state_basis.impurity_orbitals,
                    ground_state_basis.bath_states,
                    rot_to_spherical,
                    n_orb,
                    comm=mov_comm,
                    shell_ls=shell_ls,
                ),
            )
        sisb_pairing_approx = not layout.su2
        pairing_verified = layout.grade in ("su2", "collinear")
        sisb_pairing_note = layout.sisb_note()
        if layout.labelled:
            spin_pairs = (list(layout.imp_pairs), list(layout.bath_pairs))
        if spin_pairs is None:
            sisb_skip_reason = "no (dn, up) pairing of the impurity and bath makes H conserve S_z" + (
                "; " + "; ".join(layout.rejected) if layout.rejected else ""
            )
        else:
            imp_pairs, bath_pairs = spin_pairs
            ops_imp = make_spin_operators(imp_pairs)
            ops_bath = make_spin_operators(bath_pairs)
            sisb_op = spin_correlation_operator(ops_imp, ops_bath)
            sisb_raw = src.values("sisb")(lambda blk: sisb_op.apply_block(blk, 0, support_keep))
            sisb_values = np.real(sisb_raw)
            sisb_thermal = thermal_observable_value(sisb_raw, es, tau)
            if sisb_pairing_approx:
                # Longitudinal part <Sz_imp Sz_bath>: exact under the collinear check (needs only
                # the verified spin labels), reported next to the pairing-dependent full value,
                # plus its connected form against the thermal single-particle polarizations.
                sisb_z_op = spin_correlation_operator(ops_imp, ops_bath, z_only=True)
                sisb_z_raw = src.values("sisb-z")(lambda blk: sisb_z_op.apply_block(blk, 0, support_keep))
                sisb_z_values = np.real(sisb_z_raw)
                sisb_z_thermal = np.real(thermal_observable_value(sisb_z_raw, es, tau))
                sz_imp = get_Sz_from_rho_pairs(thermal_rho, imp_pairs)
                sz_bath = get_Sz_from_rho_pairs(thermal_rho, bath_pairs)
                sisb_z_connected = sisb_z_thermal - sz_imp * sz_bath
    except Exception as exc:
        # Deterministic + identical on every rank (collective Allreduce inside), so this raises
        # on all ranks together and stays collective-safe.
        sisb_values = None
        sisb_thermal = None
        sisb_z_values = None
        sisb_z_thermal = None
        sisb_z_connected = None
        sisb_pairing_approx = False
        spin_pairs = None
        sisb_skip_reason = f"spin-correlation evaluation failed: {exc}"

    # Longitudinal spin susceptibility (Curie term of the retained manifolds): only where S_z commutes with H,
    # i.e. a verified spin layout. Built on the manifold matrices of S_z, so it is the Curie law and not a
    # pure-state variance. Collective and deterministic; its own degrade guard.
    chi_spin_zz = None
    if spin_pairs is not None:
        try:
            sz_op = make_spin_operators(spin_pairs[0])[2]
            sz_diag, sz_col2 = src.curie("sz-imp")(lambda blk: sz_op.apply_block(blk, 0, support_keep))
            chi_spin_zz = spin_curie_susceptibility(es, tau, sz_diag, sz_col2)
        except Exception:
            chi_spin_zz = None

    # Correlation-strength, screening and energy-decomposition diagnostics. Collective
    # (manifold_observable_values Allreduces inside) and deterministic on identical inputs,
    # so every rank takes the same path — same degrade-to-warning pattern as above.
    corr_diagnostics = None
    screening_diagnostics = None
    energy_decomposition = None
    diagnostics_skip_reason = None
    try:
        h1 = src.once("h1", lambda: extract_tensors(Hop, n_orb=ground_state_basis.num_spin_orbitals, two_body=False)[0])
        energy_decomposition = compute_energy_decomposition(thermal_rho, h1, impurity_indices, e_avg)
        if spin_pairs is not None:
            imp_pairs, bath_pairs = spin_pairs
            corr_diagnostics = compute_correlation_diagnostics(
                psis_blk,
                es,
                tau,
                thermal_rho,
                imp_pairs,
                keep=support_keep,
                values=src.values("correlation"),
                pairing_verified=pairing_verified,
            )
            # Impurity channels grouped like the N(...) columns (equivalent-block groups),
            # using the same orbital-group labels (e.g. "group 1") as the printed report, not
            # the raw block indices (which are positions in block_structure.blocks, not
            # orbital indices, and name nothing the rest of the run refers to).
            # Block orbitals are positions into the sorted impurity list; map them back to global spin-orbital indices.
            equivalent_blocks = get_equivalent_blocks(block_structure)
            block_labels, _ = block_group_labels(
                equivalent_blocks, block_structure, ground_state_basis.impurity_orbitals
            )
            imp_groups = {}
            for blocks, label in zip(equivalent_blocks, block_labels):
                group_orbs = {impurity_indices[orb] for b in blocks for orb in block_structure.blocks[b]}
                in_group = [p for p in imp_pairs if p[0] in group_orbs and p[1] in group_orbs]
                if in_group:
                    imp_groups[label] = in_group
            if len(imp_groups) < 2:
                imp_groups = None  # a single channel equals the total <S_imp.S_bath>; skip
            screening_diagnostics = compute_screening_diagnostics(
                psis_blk,
                es,
                tau,
                thermal_rho,
                imp_pairs,
                bath_pairs,
                h1,
                imp_groups=imp_groups,
                z_only=sisb_pairing_approx,
                keep=support_keep,
                values=src.values("screening"),
            )
        else:
            diagnostics_skip_reason = "no validated spin pairing (see the <S_imp.S_bath> message)"
    except Exception as exc:
        corr_diagnostics = None
        screening_diagnostics = None
        energy_decomposition = None
        diagnostics_skip_reason = f"evaluation failed: {exc}"

    # Bundle the observable diagnostics into the statistics dict (rank 0) and save the
    # JSON only now, so ground_state_statistics.json carries the complete report.
    full_impurity_ix = np.ix_(np.arange(len(rhos)), impurity_indices, impurity_indices)
    try:
        state_summary = compute_state_summary(
            rhos[full_impurity_ix],
            es,
            rot_to_spherical,
            s_values,
            l_values,
            j_values,
            entanglement,
            impurity_orbitals=ground_state_basis.impurity_orbitals,
            shell_ls=shell_ls,
        )
    except Exception as exc:
        state_summary = None
        if rank == 0 and not src.recording:
            print(f"per-state summary not reported: {exc}")
    return SimpleNamespace(
        naming=naming,
        corr_diagnostics=corr_diagnostics,
        diagnostics_skip_reason=diagnostics_skip_reason,
        e_avg=e_avg,
        effective_restrictions=effective_restrictions,
        energy_decomposition=energy_decomposition,
        excitation_bound_records=excitation_bound_records,
        entanglement=entanglement,
        full_impurity_ix=full_impurity_ix,
        gs_stats=gs_stats,
        impurity_indices=impurity_indices,
        impurity_ix=impurity_ix,
        j_sharp=j_sharp,
        j2_thermal=j2_thermal,
        j_values=j_values,
        l2_thermal=l2_thermal,
        l_values=l_values,
        layout=layout,
        n_dets=n_dets,
        rhos=rhos,
        s2_thermal=s2_thermal,
        s_values=s_values,
        screening_diagnostics=screening_diagnostics,
        shell_casimir=shell_casimir,
        sisb_pairing_approx=sisb_pairing_approx,
        sisb_pairing_note=sisb_pairing_note,
        sisb_skip_reason=sisb_skip_reason,
        sisb_thermal=sisb_thermal,
        sisb_values=sisb_values,
        sisb_z_connected=sisb_z_connected,
        sisb_z_thermal=sisb_z_thermal,
        sisb_z_values=sisb_z_values,
        state_summary=state_summary,
        chi_spin_zz=chi_spin_zz,
        thermal_rho=thermal_rho,
    )


def _assemble(R, src, magnetic, ensemble, gs_convergence_report):
    """The :class:`~impurityModel.ed.gs_report_model.GroundStateReport` of ``R``; ``None`` if it cannot be built.

    Pure and replicated (no collective): a failure is a defect of the model, never of the solve, so it
    degrades to ``None`` and the old report still prints.
    """
    try:
        two_sz, sectors = src.state_labels()
        return assemble_report(
            src.es,
            src.tau,
            R.n_dets,
            degeneracy_tol=src.degeneracy_tol,
            naming=R.naming,
            layout=R.layout,
            two_sz=two_sz,
            sectors=sectors,
            s_values=R.s_values,
            l_values=R.l_values,
            j_values=R.j_values,
            sisb_values=R.sisb_values,
            sisb_z_values=R.sisb_z_values,
            manifold_entropy=None if R.entanglement is None else R.entanglement.get("manifold_entropy"),
            thermal={
                "e_avg": R.e_avg,
                "s2": R.s2_thermal,
                "l2": R.l2_thermal,
                "j2": R.j2_thermal,
                "sisb": None if R.sisb_thermal is None else float(np.real(R.sisb_thermal)),
                "sisb_z": R.sisb_z_thermal,
                "chi_spin_zz": None if R.chi_spin_zz is None else R.chi_spin_zz["chi_spin_zz"],
            },
            magnetic=magnetic,
            correlation=R.corr_diagnostics,
            screening=R.screening_diagnostics,
            energy_decomposition=R.energy_decomposition,
            excitation_bounds=R.excitation_bound_records,
            convergence=gs_convergence_report,
            ensemble=ensemble,
            warnings=degraded_warnings(gs_convergence_report),
        )
    except Exception:
        return None


def _present(
    R,
    src,
    rot_to_spherical,
    block_structure,
    verbose,
    stats_path,
    gs_convergence_report,
    shell_ls=None,
    ensemble=None,
    partial=None,
):
    """Bundle the observables into the statistics, save them, and print the report (rank 0).

    ``partial`` (a header line) marks a report over the members that finished before a failure: it is
    printed first and the statistics JSON is not written, so a partial file is never mistaken for a result.
    """
    ground_state_basis, es, tau = src.basis, src.es, src.tau
    files = ReportFiles.coerce(stats_path)
    comm = ground_state_basis.comm
    rank = comm.rank if comm is not None else 0
    block_structure = R.naming.block_structure
    report = Reporter(verbose, rank)
    corr_diagnostics = R.corr_diagnostics
    diagnostics_skip_reason = R.diagnostics_skip_reason
    e_avg = R.e_avg
    effective_restrictions = R.effective_restrictions
    energy_decomposition = R.energy_decomposition
    excitation_bound_records = R.excitation_bound_records
    full_impurity_ix = R.full_impurity_ix
    gs_stats = R.gs_stats
    impurity_indices = R.impurity_indices
    impurity_ix = R.impurity_ix
    j_sharp = R.j_sharp
    j2_thermal = R.j2_thermal
    j_values = R.j_values
    l2_thermal = R.l2_thermal
    l_values = R.l_values
    layout = R.layout
    n_dets = R.n_dets
    rhos = R.rhos
    s2_thermal = R.s2_thermal
    s_values = R.s_values
    screening_diagnostics = R.screening_diagnostics
    shell_casimir = R.shell_casimir
    sisb_pairing_approx = R.sisb_pairing_approx
    sisb_pairing_note = R.sisb_pairing_note
    sisb_skip_reason = R.sisb_skip_reason
    sisb_thermal = R.sisb_thermal
    sisb_values = R.sisb_values
    sisb_z_connected = R.sisb_z_connected
    sisb_z_thermal = R.sisb_z_thermal
    sisb_z_values = R.sisb_z_values
    state_summary = R.state_summary
    chi_spin_zz = R.chi_spin_zz
    thermal_rho = R.thermal_rho
    if partial and rank == 0:
        print(partial, flush=True)
    if gs_stats is not None:
        try:
            magnetic_summary = compute_magnetic_summary(
                thermal_rho[impurity_ix],
                rot_to_spherical,
                s2_thermal,
                l2_thermal,
                j2_thermal,
                impurity_orbitals=ground_state_basis.impurity_orbitals,
                j_sharp=j_sharp,
                shell_ls=shell_ls,
            )
        except Exception as exc:
            magnetic_summary = None
            if rank == 0:
                print(f"magnetic summary not reported: {exc}")
        gs_stats["observables"] = {
            "magnetic": magnetic_summary,
            "per_state": state_summary,
            "sisb": {
                "thermal": None if sisb_thermal is None else float(np.real(sisb_thermal)),
                "z_thermal": sisb_z_thermal,
                "z_connected": sisb_z_connected,
                "pairing_approx": sisb_pairing_approx,
                "skip_reason": sisb_skip_reason,
                "spin_layout": None if layout is None else asdict(layout),
            },
            "correlation": corr_diagnostics,
            "screening": screening_diagnostics,
            "energy_decomposition": energy_decomposition,
            "chi_spin_zz": chi_spin_zz,
        }
    R.report = _assemble(R, src, magnetic_summary if gs_stats is not None else None, ensemble, gs_convergence_report)
    if gs_stats is not None and R.report is not None:
        gs_stats["report"] = R.report.to_json()
    summary_only = verbose == V_SUMMARY
    want_text = files.text is not None and not partial
    summary_lines = []
    if rank == 0 and (summary_only or want_text) and R.report is not None:
        diag = np.real(np.diag(thermal_rho)) if thermal_rho is not None else None
        stats_ = gs_stats or {}
        configs = [
            (tuple(r["config"]), r["fraction"]) for r in stats_.get("thermal_config_weights", {}).get("rows", [])
        ]
        try:
            summary_lines = render_summary(
                R.report,
                diag=diag,
                configurations=configs,
                one_body_entropy=stats_.get("one_body_entropy"),
                unit=files.unit,
                files=(files.text, files.json) if (files.text or files.json) else None,
            )
        except Exception as exc:
            summary_lines = [f"[warning] ground-state summary incomplete (GS still returned): {exc}"]
    if rank == 0 and summary_only and R.report is not None:
        # -v: the compact summary stands in for the terse record and the full report (which -vv prints).
        for line in summary_lines:
            print(line)
        for reason in degraded_warnings(gs_convergence_report):
            print(f"WARNING: {reason}", file=sys.stderr, flush=True)
    elif rank == 0:
        # Terse run record: shown at V_RESULT and V_DETAIL+, independent of the full report below.
        gs_occ = getattr(ground_state_basis, "ground_state_occupation", None)
        occ_str = f"; ground-state impurity occupation: {sum(gs_occ.values())}" if gs_occ else ""
        print(f"E0 = {np.min(es):.6f}   retained states = {len(es)}   tau = {tau:.4g}")
        print(f"{n_dets:,} Slater determinants{occ_str}")
        for rec in excitation_bound_records:
            tag = "" if rec["n_imp"] is None or len(excitation_bound_records) == 1 else f"N_imp = {rec['n_imp']}: "
            print(
                f"{tag}effective bath-excitation bounds (max over the GS basis; excited bases run at +1): "
                f"holes <= {rec['max_holes']}, electrons <= {rec['max_electrons']}, total <= {rec['max_total']}"
                + (" [cap-bound]" if rec["cap_bound"] else "")
            )
        print(convergence_summary(gs_convergence_report))
        if ensemble is not None:
            print(
                "charge-sector ensemble: "
                + ", ".join(f"N_imp = {m['total']} (weight {m['weight']:.4f})" for m in ensemble["members"])
                + f"; error bar on <N_imp> {ensemble['error_bar']:.3g} e"
                + (
                    f"; <N_imp> in [{ensemble['n_imp_interval'][0]:.4f}, {ensemble['n_imp_interval'][1]:.4f}]"
                    if ensemble.get("n_imp_interval")
                    else ""
                )
            )
        for reason in degraded_warnings(gs_convergence_report):
            # stdout too: RSPt redirects only stdout to the file the user reads.
            print(f"WARNING: {reason}", flush=True)
            print(f"WARNING: {reason}", file=sys.stderr, flush=True)
        if not report.enabled(V_DETAIL):
            note = f"; saved to {files.text or files.json}" if (files.text or files.json) else ""
            print(f"(full report: -vv{note})")
    echo_full = report.enabled(V_DETAIL)
    full_buffer = io.StringIO()
    if echo_full or (rank == 0 and want_text):
        _stdout = sys.stdout
        sys.stdout = full_buffer
        # The ground state is fully computed by here; formatting/printing the observable report
        # must never crash the solve. Any failure degrades to a warning and still returns the GS.
        # Rank-0-only (report.enabled already checks rank == 0; no collectives inside), so the
        # guard cannot desync an MPI run.
        try:
            report_banner("Ground-state report")
            print(f"impurity spin-orbitals: {impurity_indices}")
            # Range-compressed: a bath set spans ~90 spin-orbitals, and spelled out in full the
            # index list is all one sees of a line whose point is the occupation window at its end.
            print("Effective GS restrictions (observed electron count per orbital set):")
            for indices, occupations in effective_restrictions.items():
                print(f"  {format_index_ranges(indices)} : {occupations[0]}-{occupations[1]}")
            if excitation_bound_records:
                print("Effective bath-excitation bounds (valence holes / conduction electrons / total):")
                for rec in excitation_bound_records:
                    n_imp = "" if rec["n_imp"] is None else f"N_imp = {rec['n_imp']} : "
                    print(
                        f"  {n_imp}valence holes {rec['max_holes']} -> {rec['max_holes'] + 1} | "
                        f"conduction electrons {rec['max_electrons']} -> {rec['max_electrons'] + 1} | "
                        f"total {rec['max_total']} -> {rec['max_total'] + 1}"
                        + ("  [cap-bound]" if rec["cap_bound"] else "")
                    )
            print("Block structure:")
            print_block_structure(block_structure)
            print_impurity_orbital_groups(
                get_equivalent_blocks(block_structure), block_structure, ground_state_basis.impurity_orbitals
            )

            report_rule("Thermal expectation values")
            extra_groups = []
            if energy_decomposition is not None:
                extra_groups.append(
                    (
                        "Energy decomposition",
                        [
                            ("<H_1body>", energy_decomposition["e_one_body"], ""),
                            ("<H_imp,1b>", energy_decomposition["e_imp_1b"], ""),
                            ("<H_bath>", energy_decomposition["e_bath"], ""),
                            ("<E_hyb>", energy_decomposition["e_hyb"], ""),
                            ("<H_Coulomb>", energy_decomposition["e_coulomb"], "(= <E> - <H_1body>)"),
                        ],
                    )
                )
            if chi_spin_zz is not None:
                extra_groups.append(
                    (
                        "Spin susceptibility",
                        [
                            ("chi_spin_zz", chi_spin_zz["chi_spin_zz"], "((<Sz^2> - <Sz>^2)/tau)"),
                            (
                                "note",
                                None,
                                f"Curie term of the retained manifolds only (no Van Vleck part); tau = {chi_spin_zz['tau']:.4g}",  # noqa: E501
                            ),
                        ],
                    )
                )
            print_thermal_expectation_values(
                thermal_rho[impurity_ix],
                e_avg,
                rot_to_spherical,
                block_structure,
                s_thermal=s2_thermal,
                l_thermal=l2_thermal,
                j_thermal=j2_thermal,
                sisb_thermal=sisb_thermal,
                sisb_z_thermal=sisb_z_thermal,
                sisb_z_connected=sisb_z_connected,
                sisb_pairing_note=sisb_pairing_note,
                spin_pairing=None if layout is None else layout.pairs_source(),
                extra_groups=extra_groups or None,
                impurity_orbitals=ground_state_basis.impurity_orbitals,
                shell_casimir=shell_casimir,
                j_sharp=j_sharp,
                shell_ls=shell_ls,
            )
            report_rule("Eigenstates")
            print_expectation_values(
                rhos[full_impurity_ix],
                es,
                rot_to_spherical,
                block_structure,
                s_values=s_values,
                l_values=l_values,
                j_values=j_values,
                sisb_values=sisb_values,
                sisb_z_values=sisb_z_values,
                impurity_orbitals=ground_state_basis.impurity_orbitals,
                shell_ls=shell_ls,
            )
            if state_summary is not None:
                print_state_summary(state_summary)
                print()
            if sisb_skip_reason is not None:
                print(f"<S_imp.S_bath> not reported: {sisb_skip_reason}")
            if corr_diagnostics is not None:
                report_rule("Correlation strength")
                print_correlation_diagnostics(corr_diagnostics)
            if screening_diagnostics is not None:
                report_rule("Screening")
                print_screening_diagnostics(screening_diagnostics)
            if diagnostics_skip_reason is not None:
                print(f"correlation/screening diagnostics not reported: {diagnostics_skip_reason}")
            if gs_stats is not None:
                report_rule("Configurations & entanglement")
                print_gs_statistics(gs_stats, verbose=2 if verbose >= V_DETAIL else 1)
                report_rule("Density matrices")
                print("Ground state impurity / bath density matrices:")
                valence_bath_states, conduction_bath_states = ground_state_basis.bath_states
                for i in ground_state_basis.impurity_orbitals.keys():
                    print(f"orbital set {i}:")
                    impurity_orbital_blocks = ground_state_basis.impurity_orbitals[i]
                    valence_bath_orbital_blocks = valence_bath_states[i]
                    conduction_bath_orbital_blocks = conduction_bath_states[i]
                    for block_i, (imp_orbs, val_orbs, con_orbs) in enumerate(
                        zip(impurity_orbital_blocks, valence_bath_orbital_blocks, conduction_bath_orbital_blocks)
                    ):
                        print(f"Block {block_i}: impurity {imp_orbs}, valence {val_orbs}, conduction {con_orbs}")
                        impurity_ix = np.ix_(imp_orbs, imp_orbs)
                        bath_ix = np.ix_(val_orbs + con_orbs, val_orbs + con_orbs)
                        matrix_print(thermal_rho[impurity_ix], "Impurity density matrix:", n_prec=5)
                        if verbose >= V_DETAIL:
                            matrix_print(thermal_rho[bath_ix], "Bath density matrix:", n_prec=5)
                        else:
                            print_density_matrix_summary(thermal_rho[bath_ix], "Bath density matrix (summary):")
                    print("", flush=verbose >= V_DETAIL)
                print()
        except Exception as exc:
            print(f"[warning] ground-state observable report incomplete (GS still returned): {exc}")
        finally:
            sys.stdout = _stdout
        if echo_full:
            sys.stdout.write(full_buffer.getvalue())
            sys.stdout.flush()
    if (files.text is not None or files.json is not None) and not partial:
        # Collective, so gated on replicated state only (the file names): a failed write on rank 0 (an
        # unserialisable value, a full disk, a missing directory) is raised on every rank; a rank-0-only raise
        # left the others in the next collective (the GS8 numpy-bool hang).
        def _write():
            if rank != 0:
                return False
            if files.text is not None:
                with open(files.text, "w") as fh:
                    fh.write("\n".join(summary_lines) + "\n\n" + full_buffer.getvalue())
            if files.json is not None and gs_stats is not None:
                save_gs_statistics(gs_stats, files.json)
            return True

        on_root(comm if ground_state_basis.is_distributed else None, _write)
    return R


def _info(R, gs_truncation_report, gs_convergence_report):
    return {
        "rhos": R.rhos,
        "statistics": R.gs_stats,
        "truncation": gs_truncation_report,
        "convergence": gs_convergence_report,
        "excitation_bounds": R.excitation_bound_records,
        "correlation_diagnostics": R.corr_diagnostics,
        "screening_diagnostics": R.screening_diagnostics,
        "energy_decomposition": R.energy_decomposition,
        "chi_spin_zz": R.chi_spin_zz,
        "report": getattr(R, "report", None),
    }


def report_ground_state(
    Hop,
    ground_state_basis,
    es,
    psis,
    tau,
    block_structure,
    rot_to_spherical,
    verbose,
    stats_path,
    gs_truncation_report,
    gs_convergence_report,
    spin_layout=None,
    shell_ls=None,
    ensemble=None,
    slater_weight_min=None,
):
    """Observables of the thermal ground manifold ``(es, psis)``, and the report that prints them.

    Collective on ``ground_state_basis.comm``. Returns ``(psis, es, ground_state_basis, thermal_rho,
    gs_info)`` -- the tuple ``calc_gs`` returns, with ``psis`` redistributed onto the basis and
    materialized as a list.

    ``spin_layout`` is the :class:`~impurityModel.ed.spin_pairs.SpinLayout` of the solver basis
    (``calc_gs`` passes the one it solved with); ``None`` resolves it here from ``Hop``.
    ``shell_ls`` (``l`` of each column block of an ndarray ``rot_to_spherical``) makes every shell-aware
    observable and the per-shell Casimirs treat a multi-shell impurity shell by shell; ``None`` is one shell.
    ``ensemble`` (the charge-sector ensemble record of ``calc_gs``) is stored in the statistics as
    ``charge_sector_ensemble``; its ``state_sector`` gives the sector of each state of ``(es, psis)``.
    """
    # Redistribute onto the basis as it stands; do not first collapse it onto the support of
    # `psis`. The basis is also the seed space for the excited/Green's-function expansion
    # downstream, and shrinking it to the thermal eigenvectors' support throws away exactly the
    # determinants a transition operator is most likely to reach.
    # `psis` is the block `get_eigenvectors` returns, so it redistributes as a block --
    # `*psis` would unpack its determinant KEYS, and at 1 rank `redistribute_psis` no-ops and
    # hands them straight back rather than raising. `prune_rows(0.0)` matches what
    # `redistribute_psis` did per column (it pruned each slice, and the caller's `from_states`
    # re-unioned), so the support is unchanged.
    psis_blk = ground_state_basis.redistribute_block(
        psis if isinstance(psis, ManyBodyState) else ManyBodyState.from_states(list(psis))
    )
    psis_blk.prune_rows(0.0)
    psis_blk, manifold_basis = sharpen_for_report(
        Hop, ground_state_basis, es, psis_blk, rot_to_spherical, spin_layout, shell_ls, slater_weight_min
    )
    # `calc_gs`'s documented return contract is `list[ManyBodyState]` (the spectra / GF / RIXS
    # seed paths consume it as one), so the list is materialized here from the block.
    psis = psis_blk.to_states()
    src = _LiveSource(
        ground_state_basis,
        es,
        psis_blk,
        tau,
        degeneracy_tol=manifold_tol(np.asarray(es), slater_weight_min),
        manifold_basis=manifold_basis,
    )
    R = _evaluate(
        src,
        Hop,
        block_structure,
        rot_to_spherical,
        gs_truncation_report,
        gs_convergence_report,
        spin_layout=spin_layout,
        shell_ls=shell_ls,
        ensemble=ensemble,
    )
    _present(
        R,
        src,
        rot_to_spherical,
        block_structure,
        verbose,
        stats_path,
        gs_convergence_report,
        shell_ls=shell_ls,
        ensemble=ensemble,
    )
    return psis, es, ground_state_basis, R.thermal_rho, _info(R, gs_truncation_report, gs_convergence_report)


def measure_member(
    Hop,
    ground_state_basis,
    es,
    psis_blk,
    tau,
    block_structure,
    rot_to_spherical,
    spin_layout=None,
    shell_ls=None,
    total=None,
    keep_basis=True,
    slater_weight_min=None,
    manifold_basis=None,
):
    """One charge-sector member's per-state report data, taken while its states are alive. Collective.

    ``psis_blk`` is the member's thermal block already redistributed onto ``ground_state_basis`` and pruned
    (the caller owns it and converts it for the driver afterwards). Nothing but small, replicated per-state
    results is kept, so the states can be freed before the next member is solved; :func:`finalize_members`
    turns the members into the report. Prints nothing: a quantity that fails here fails again, once, in the
    replay, which reports it.
    """
    tape = {}
    tol = manifold_tol(np.asarray(es, dtype=float), slater_weight_min)
    src = _LiveSource(ground_state_basis, es, psis_blk, tau, tape=tape, degeneracy_tol=tol)
    _evaluate(
        src,
        Hop,
        block_structure,
        rot_to_spherical,
        None,
        None,
        spin_layout=spin_layout,
        shell_ls=shell_ls,
    )
    es = np.asarray(es, dtype=float)
    # Rank 0's copy of the energies fixes the union's order the same way on every rank (the eigensolver's
    # energies are rank-local, a degenerate multiplet differs in its last bits). One unconditional broadcast.
    es_ref = ground_state_basis.comm.bcast(es, root=0) if ground_state_basis.is_distributed else es
    return MemberMeasurement(es, es_ref, total, tape, ground_state_basis if keep_basis else None, tol, manifold_basis)


def finalize_members(
    Hop,
    members,
    tau,
    block_structure,
    rot_to_spherical,
    verbose,
    stats_path,
    gs_truncation_report,
    gs_convergence_report,
    spin_layout=None,
    shell_ls=None,
    ensemble=None,
    partial=None,
):
    """The ground-state report over the members' :class:`MemberMeasurement`, printed and saved.

    Returns ``(None, es, basis, thermal_rho, gs_info)`` (the states are not held). Collective only through the
    statistics save, which ``partial`` (a header naming the members that finished) turns off: a partial
    report is rank-local pure combination and printing, so it cannot deadlock a run in which one rank failed.
    """
    src = _ReplaySource(members, tau)
    R = _evaluate(
        src,
        Hop,
        block_structure,
        rot_to_spherical,
        gs_truncation_report,
        gs_convergence_report,
        spin_layout=spin_layout,
        shell_ls=shell_ls,
        ensemble=ensemble,
    )
    _present(
        R,
        src,
        rot_to_spherical,
        block_structure,
        verbose,
        stats_path,
        gs_convergence_report,
        shell_ls=shell_ls,
        ensemble=ensemble,
        partial=partial,
    )
    return None, src.es, src.basis, R.thermal_rho, _info(R, gs_truncation_report, gs_convergence_report)
