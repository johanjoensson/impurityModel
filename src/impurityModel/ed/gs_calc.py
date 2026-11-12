"""``calc_gs``: the selfenergy/spectra ground-state entry point.

Builds the :class:`GroundStateSpec`, runs the HF-seed-then-walk search through
:func:`~impurityModel.ed.gs_ensemble.solve_charge_sector_ensemble`, measures each member with
:class:`_MeasureVisitor` and returns the thermal state the Green's-function code consumes.
"""

import contextlib
import io
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from impurityModel.ed import config
from impurityModel.ed.basis_restrictions import (
    build_weighted_restrictions,
)
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.groundstate import GS_DE2_MIN, GS_E_PT2_TOL
from impurityModel.ed.gs_ensemble import DrivenState, _agree_on_switch, solve_charge_sector_ensemble
from impurityModel.ed.gs_report import (
    ReportFiles,
    finalize_members,
    measure_member,
    sharpen_for_report,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.sector_ensemble import (
    resolve_mixed_valence,
)
from impurityModel.ed.spin_pairs import spin_layout
from impurityModel.ed.utils import (
    V_SUMMARY,
)


@dataclass(frozen=True)
class GroundStateSpec:
    """What decides *which state* ``solve_ground_state`` finds, apart from the model (A1).

    One per driver, built explicitly so the differences between drivers (spectra solves at a Slater
    weight of 0 and builds its restrictions at another; susceptibility never chain-restricts) are
    visible values instead of the accident of which keys a call site happened to pass. Deliberately
    absent: ``comm``, ``verbose``, ``plan_store``, ``shift`` (how, not what), ``shell_ls`` and the
    per-shell N0 (owned by ``SolverBasis``), and the impurity/bath layout (the model).

    The weighted restrictions are stored as their *inputs* (``cost_budget``,
    ``restriction_slater_weight_min``): the built list depends on ``h_op``, which differs between the
    ``dc_guess`` a DC search builds at and the final ``dc`` a self-energy builds at.

    Attributes
    ----------
    tau : float
        Temperature (thermal window width).
    chain_restrict, dense_cutoff, use_hf_seed, e_pt2_tol, de2_min :
        As in :func:`solve_ground_state`.
    truncation_threshold : int, float, CapPolicy or None
        The determinant cap, or the caller-resolved policy (P10).
    cost_budget, restriction_slater_weight_min :
        Inputs of :func:`~impurityModel.ed.basis_restrictions.build_weighted_restrictions`.
    """

    tau: float
    chain_restrict: bool = False
    dense_cutoff: int = config.GS_DENSE_CUTOFF
    use_hf_seed: bool = True
    e_pt2_tol: float = GS_E_PT2_TOL
    de2_min: float = GS_DE2_MIN
    truncation_threshold: object = None
    cost_budget: Optional[float] = None
    restriction_slater_weight_min: Optional[float] = None
    mixed_valence: object = None

    def solve_kwargs(self, *names):
        """The ``solve_ground_state`` keywords this spec decides (only ``names`` when given).

        A narrower entry (``find_ground_state_sector``, ``solve_sector``) takes a subset; naming it keeps
        the value in one place without handing a keyword to a function that does not have it.
        """
        kwargs = {
            "tau": self.tau,
            "chain_restrict": self.chain_restrict,
            "dense_cutoff": self.dense_cutoff,
            "use_hf_seed": self.use_hf_seed,
            "e_pt2_tol": self.e_pt2_tol,
            "de2_min": self.de2_min,
            "truncation_threshold": self.truncation_threshold,
        }
        if names:
            unknown = sorted(set(names) - set(kwargs))
            if unknown:
                raise KeyError(f"GroundStateSpec decides no {unknown}")
            kwargs = {name: kwargs[name] for name in names}
        return kwargs

    def weighted_restrictions(self, bath_states, h_op, impurity_orbitals):
        """The weighted restrictions for ``h_op``.

        ``None`` when no cost budget is requested; the excitation bounds are an output of the solve, not an input.
        """
        return build_weighted_restrictions(
            bath_states,
            cost_budget=self.cost_budget,
            h_op=h_op,
            impurity_orbitals=impurity_orbitals,
            slater_weight_min=self.restriction_slater_weight_min,
        )


class _HeldBack(io.StringIO):
    """A driver's stdout, held back until the ground-state report has printed.

    ``WARNING`` lines are also written to stderr at once, so they are not lost if the job is killed.
    """

    def __init__(self):
        super().__init__()
        self._line = ""

    def write(self, text):
        n = super().write(text)
        *whole, self._line = (self._line + text).split("\n")
        for line in whole:
            if line.startswith("WARNING"):
                print(line, file=sys.stderr, flush=True)
        return n


class _MeasureVisitor:
    """The ground-state report's own visitor: measures each member, then runs the driver's on the same states.

    The member's raw block is taken from its :class:`MemberState` (which then holds no reference), redistributed
    and pruned once, measured (:func:`~impurityModel.ed.gs_report.measure_member`: only small per-state results
    are kept) and converted once to the width-1 ``states`` the driver works on. No other copy of the thermal
    states is alive while the driver runs, and none is kept after it returns unless there is no driver
    (``states`` is then the caller's return value).

    ``visit`` returns ``(measurement, truncation, convergence, driver_result, states)``; only the first visited
    member keeps its basis (it carries the report), ``states`` is ``None`` when a driver consumed them.
    """

    def __init__(self, report_args, driver_visitor=None):
        self.report_args = report_args
        self.driver = driver_visitor
        self.count = 0
        # (total, text) per driven member, in visit order: what each driver printed, held back so the ground-state
        # report comes first. Appended when the driver returns or raises, so a failing member's text is here too.
        self.captured = []
        # What a failure path can still report: the members measured so far, and the one being visited (None
        # between visits, i.e. the failure came from a solve).
        self.measured = []
        self.current = None

    def visit(self, member, state):
        self.current = member.total
        basis, n = state.basis, len(state.es)
        raw = state.take_psis()
        blk = basis.redistribute_block(raw if isinstance(raw, ManyBodyState) else ManyBodyState.from_states(list(raw)))
        del raw
        blk.prune_rows(0.0)
        blk, manifold_basis = sharpen_for_report(
            self.report_args["Hop"],
            basis,
            np.array(state.es),
            blk,
            self.report_args["rot_to_spherical"],
            self.report_args["spin_layout"],
            self.report_args["shell_ls"],
            self.report_args.get("slater_weight_min"),
        )
        measurement = measure_member(
            self.report_args["Hop"],
            basis,
            np.array(state.es),
            blk,
            self.report_args["tau"],
            self.report_args["block_structure"],
            self.report_args["rot_to_spherical"],
            spin_layout=self.report_args["spin_layout"],
            shell_ls=self.report_args["shell_ls"],
            total=member.total,
            keep_basis=self.count == 0,
            slater_weight_min=self.report_args.get("slater_weight_min"),
            manifold_basis=manifold_basis,
        )
        self.count += 1
        self.measured.append(measurement)
        states = blk.to_states()
        del blk
        # A rank that received no rows holds a width-0 block whose to_states() is []: keep len(es) states on every rank.
        states = states if states else [ManyBodyState(width=1) for _ in range(n)]
        driven = None
        if self.driver is not None:
            if basis.comm is None or basis.comm.rank == 0:
                print(f"[N_imp={member.total}] ground state measured; running the driver ...", flush=True)
            # stdout of the driver is held back on every rank (a split-communicator colour root prints from any
            # rank); stderr, WARNINGs included, stays live.
            buffer = _HeldBack()
            self.captured.append((member.total, buffer))
            with contextlib.redirect_stdout(buffer):
                driven = self.driver.visit(
                    member,
                    DrivenState(basis, state.es, states, state.occupation, state.truncation, state.convergence),
                )
            states = None
        self.current = None
        return measurement, state.truncation, state.convergence, driven, states


def _replay_driver_output(captured, with_headers):
    """Print what the drivers wrote while the members ran (held back by :class:`_MeasureVisitor`), per rank."""
    for total, buffer in captured:
        text = buffer.getvalue()
        if text:
            if with_headers and len(captured) > 1:
                print(f"--- N_imp = {total} ---")
            print(text, end="" if text.endswith("\n") else "\n", flush=True)


def _report_what_exists(
    measure, exc, Hop, tau, block_structure, rot_to_spherical, verbose, layout, shell_ls, comm=None, stage="solve"
):
    """Failure path of :func:`calc_gs`: print the reports gathered so far; never raises, never communicates.

    A partial ground-state report over the members already measured (rank 0), then every rank's own held-back
    driver text (the failing member's up to the failure). No collectives, so it cannot deadlock a run in which
    only some ranks failed; the statistics JSON is not written. Anything going wrong in here is reduced to one
    line on stderr so the original exception is the one that propagates.
    """
    try:
        done = [m.total for m in measure.measured]
        if stage != "solve":
            where = stage
        else:
            where = f"N_imp={measure.current}" if measure.current is not None else "a member's ground-state solve"
        header = (
            f"PARTIAL GROUND-STATE REPORT: members N_imp={done} measured; "
            f"failed at {where} ({type(exc).__name__}: {exc})"
        )
        if stage != "solve":
            header = f"GROUND-STATE REPORT FAILED in {stage} ({type(exc).__name__}: {exc}); driver output follows"
        if measure.measured and stage == "solve":
            finalize_members(
                Hop,
                measure.measured,
                tau,
                block_structure,
                rot_to_spherical,
                verbose,
                None,
                None,
                None,
                spin_layout=layout,
                shell_ls=shell_ls,
                partial=header,
            )
        else:
            print(header, flush=True)
        _replay_driver_output(measure.captured, True)
    except BaseException as printing:
        import sys

        print(f"(partial report not printed: {type(printing).__name__}: {printing})", file=sys.stderr, flush=True)


_UNSET = object()


def calc_gs(
    Hop: ManyBodyOperator,
    basis_setup: dict,
    block_structure: BlockStructure,
    rot_to_spherical: np.ndarray,
    verbose,
    slaterWeightMin=0,
    stats_path=_UNSET,
    shell_ls=None,
    spec: Optional[GroundStateSpec] = None,
    visitor=None,
    report_files: Optional[ReportFiles] = None,
    **kwargs,
):
    """
    Calculate the ground-state wavefunction, eigen-energies, and density matrices.

    This function determines the ground-state charge sector, optimizes the
    variational many-body basis, solves the eigensystem for the low-energy
    states, and computes the thermally-averaged density matrix and expectation
    values.

    Parameters
    ----------
    Hop : ManyBodyOperator
        The Hamiltonian operator.
    basis_setup : dict
        Configuration dictionary containing parameters for the basis setup,
        such as 'impurity_orbitals', 'bath_states', 'nominal_impurity_occ', etc.
    block_structure : BlockStructure
        The block structure defining mapping and symmetry relationships.
    rot_to_spherical : ndarray
        Transformation matrix from local to spherical harmonics.
    verbose : int or bool
        Verbosity level (``impurityModel.ed.utils.V_RESULT``..``V_DEBUG``). ``V_RESULT``
        (0) prints only a terse summary; ``V_SUMMARY`` (1, ``-v``) adds the full
        ground-state observable report; ``V_DETAIL`` (2, ``-vv``) adds the full bath
        density matrices (a summary prints below that) and CIPSI-round chatter. A plain
        ``bool`` degrades gracefully (``True`` behaves like ``V_SUMMARY``, an int
        subclass in Python).
    shell_ls : tuple of int, optional
        ``l`` of each column block of an ndarray ``rot_to_spherical`` for a multi-shell impurity (reporting
        only); ``None`` is a single shell.
    report_files : ReportFiles, optional
        Where to write the full text report and the JSON data (:meth:`ReportFiles.for_label`). The default
        writes nothing.
    stats_path : str, optional
        Deprecated alias for ``ReportFiles(json=stats_path)``.
    spec : GroundStateSpec, optional
        What decides which state is found (see :class:`GroundStateSpec`). Its fields are merged into
        ``basis_setup``; a key set by both raises ``ValueError``.
    slaterWeightMin : float
        Minimum weight threshold for determinants in the basis.

    visitor : MemberVisitor, optional
        Called collectively as ``visitor.visit(member, state)`` for each charge-sector member (one under
        ``mixed_valence`` off) while its basis is alive; the returns are listed in ``gs_info["member_results"]``,
        dominant member first. ``state`` is a :class:`DrivenState`: the member's thermal ``states`` (width-1,
        redistributed onto its basis; the only live copy of them) and no raw ``psis``. What the visitor prints is
        held back and shown after the ground-state report (``WARNING`` lines also go to stderr at once).

    Returns
    -------
    psis : list of ManyBodyState or None
        The low-energy eigenstates of the dominant member; ``None`` when a ``visitor`` consumed them.
    es : ndarray
        The corresponding eigen-energies.
    ground_state_basis : Basis
        The optimized many-body basis.
    thermal_rho : ndarray
        The thermally-averaged density matrix.
    gs_info : dict
        A dictionary containing additional ground-state info (e.g. 'rhos' list).
    """

    if stats_path is not _UNSET:
        warnings.warn(
            "calc_gs(stats_path=...) is deprecated; pass report_files=ReportFiles(json=...)",
            DeprecationWarning,
            stacklevel=2,
        )
        if report_files is None:
            report_files = ReportFiles.coerce(stats_path)
    basis_setup = dict(basis_setup)
    if "impurity_orbital" in basis_setup:
        basis_setup["impurity_orbitals"] = basis_setup.pop("impurity_orbital")
    if "nominal_impurity_occ" in basis_setup:
        basis_setup["N0"] = basis_setup.pop("nominal_impurity_occ")

    if spec is not None:
        merged = spec.solve_kwargs()
        merged["weighted_restrictions"] = spec.weighted_restrictions(
            basis_setup["bath_states"], Hop, basis_setup["impurity_orbitals"]
        )
        clash = sorted(set(merged) & set(basis_setup))
        if clash:
            raise ValueError(f"calc_gs: {clash} given both in basis_setup and in the GroundStateSpec")
        basis_setup.update(merged)
    # The charge-sector switch is not a solve_ground_state keyword (J4): it picks the driver's path here.
    mv_given = basis_setup.pop("mixed_valence", None)
    if mv_given is None and spec is not None:
        mv_given = spec.mixed_valence
    mixed_valence, mv_note = resolve_mixed_valence(mv_given, driver_default="on")
    _agree_on_switch(mixed_valence, basis_setup.get("comm"))
    if mv_note is not None and (basis_setup.get("comm") is None or basis_setup["comm"].rank == 0):
        print(mv_note, flush=True)
    tau = basis_setup["tau"]
    layout = basis_setup.get("spin_layout")
    if layout is None:
        # Not resolved by a driver (get_spectra): resolve here, once, from the full operator.
        blocks = [b for part in basis_setup["impurity_orbitals"].values() for b in part]
        bath = [b for baths in basis_setup["bath_states"] for part in baths.values() for b in part]
        n_orb = 1 + max(
            max((o for term in Hop.keys() for o, _ in term), default=-1),
            max((o for block in blocks + bath for o in block), default=-1),
        )
        layout = spin_layout(
            Hop,
            basis_setup["impurity_orbitals"],
            basis_setup["bath_states"],
            rot_to_spherical,
            n_orb,
            comm=basis_setup.get("comm"),
            shell_ls=shell_ls,
        )
        basis_setup["spin_layout"] = layout
    note = layout.provenance_note()
    if note is not None and verbose >= V_SUMMARY and (basis_setup.get("comm") is None or basis_setup["comm"].rank == 0):
        print(note, flush=True)
    # One definition of "the ground state of this model", shared with the double-counting
    # criteria (see solve_charge_sector_ensemble): it owns the sector walk, its cutoff, the eigensolver choice
    # and the width of the thermal manifold. `basis_setup` is the caller's dict, so pass it through
    # rather than re-listing its keys.
    measure = _MeasureVisitor(
        {
            "Hop": Hop,
            "tau": tau,
            "block_structure": block_structure,
            "rot_to_spherical": rot_to_spherical,
            "spin_layout": layout,
            "shell_ls": shell_ls,
            "slater_weight_min": slaterWeightMin,
        },
        visitor,
    )
    try:
        result = solve_charge_sector_ensemble(
            Hop,
            verbose=verbose,
            slaterWeightMin=slaterWeightMin,
            mixed_valence=mixed_valence,
            visitor=measure,
            **basis_setup,
        )
    except BaseException as exc:
        _report_what_exists(
            measure,
            exc,
            Hop,
            tau,
            block_structure,
            rot_to_spherical,
            verbose,
            layout,
            shell_ls,
            comm=basis_setup.get("comm"),
        )
        raise
    try:
        kept = result.visited
        measurements = [k[0] for k in kept]
        converged = [(k[2] or {}).get("converged") is not False for k in kept]
        # The union's order, from rank 0's copy of the energies (identical on every rank); one member keeps its own.
        union_es = np.concatenate([m.es_ref for m in measurements])
        union = np.arange(len(union_es)) if len(measurements) == 1 else np.argsort(union_es, kind="stable")
        labels = np.concatenate([np.full(len(m.es), m.total) for m in measurements])
        info = {
            "mixed_valence": bool(mixed_valence),
            "tau": tau,
            "e0": float(min(result.realised)),
            "error_bar": result.error_bar,
            "n_imp_interval": None if result.n_imp_interval is None else [float(x) for x in result.n_imp_interval],
            "refused_reason": result.refused_reason,
            "members": [
                {
                    "total": member.total,
                    "weight": float(w),
                    "e0": float(e0),
                    "walk_energy": member.walk_energy,
                    "residual_pt2": member.residual,
                    "error_bar": member.error_bar,
                    "n_states": len(m.es),
                    "converged": ok,
                    "truncation": k[1],
                }
                for member, w, e0, m, k, ok in zip(
                    result.members, result.weights, result.realised, measurements, kept, converged
                )
            ],
            # Per state of the merged union, which is ascending in energy (rank 0's order); "es" is that
            # union, the energies `rhos` belongs to (review H-MV4).
            "state_sector": [int(labels[k]) for k in union],
            "es": [float(e) for e in union_es[union]],
        }
        _, es, basis, thermal_rho, gs_info = finalize_members(
            Hop,
            measurements,
            tau,
            block_structure,
            rot_to_spherical,
            verbose,
            report_files,
            result.truncation,
            result.convergence,
            spin_layout=layout,
            shell_ls=shell_ls,
            # The statistics and the printed overview describe an ensemble only when one was asked for.
            ensemble=info if mixed_valence else None,
        )
    except BaseException as exc:
        # Every member is done; what failed is the report itself. The drivers' text must not die with it.
        _report_what_exists(
            measure,
            exc,
            Hop,
            tau,
            block_structure,
            rot_to_spherical,
            verbose,
            layout,
            shell_ls,
            comm=basis_setup.get("comm"),
            stage="the report's assembly",
        )
        raise
    gs_info["ensemble"] = info
    gs_info["member_results"] = [k[3] for k in kept]
    _replay_driver_output(measure.captured, mixed_valence)
    dominant = [k for k, total in enumerate(info["state_sector"]) if total == result.members[0].total]
    if visitor is not None:
        # The driver consumed every member's states; none is held for the return.
        return None, np.asarray(es)[dominant], basis, thermal_rho, gs_info
    # No driver: the dominant member's states, as the single-sector contract hands them on, in ascending energy
    # (the union's order restricted to the member; one member keeps its own order).
    states = kept[0][4]
    if len(kept) > 1:
        states = [states[k] for k in np.argsort(measurements[0].es_ref, kind="stable")]
    return states, np.asarray(es)[dominant], basis, thermal_rho, gs_info
