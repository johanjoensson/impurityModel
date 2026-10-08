import os
from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed import gf_diagnostics as _gfd
from impurityModel.ed.average import ThermalEnsemble, thermal_average_scale_indep
from impurityModel.ed.basis_restrictions import build_excited_restrictions, intersect_windows
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.BlockLanczosArray import Reort

# The module was split for readability (gf_primitives/gf_convergence/gf_shift_recycling hold the
# solver-primitive, convergence-monitor and shift-recycled-resolvent layers respectively); the
# re-exports below keep every existing `greens_function.X` / `gf.X` access (tests, spectra.py)
# working, so the F401 suppressions below are deliberate (imported-for-re-export, not unused).
from impurityModel.ed.gf_convergence import (  # noqa: F401  -- re-exported for backward compat
    _GF_MONITOR_POINTS,
    _GF_REL_TOL_FLOOR,
    _gf_axis_tols,
    _gf_eval_meshes,
    _gf_rel_tol,
    _gf_sample_mesh,
    _gf_signed_axes,
    _greens_function_change,
    _lanczos_convergence_summary,
    _make_gf_convergence_monitor,
    _weighted_axis_tols,
)
from impurityModel.ed.gf_engine import (
    combine_sides,
    lanczos_unit_kernel,
    states_by_group,
)
from impurityModel.ed.gf_primitives import (  # noqa: F401  -- re-exported for backward compat
    _CappedBasisProxy,
    _distributed_seed_qr,
    _sanitize_continued_fraction,
    _trim_blocks,
    build_qr,
    calc_G,
    calc_thermally_averaged_G,
)
from impurityModel.ed.gf_shift_recycling import (  # noqa: F401  -- re-exported for backward compat
    KrylovShiftedResolvent,
    SectorResolventCache,
)
from impurityModel.ed.gf_solvers import (  # noqa: F401  -- block_Green(_sparse) re-exported for rixs/tests
    block_Green,
    block_Green_bicgstab,
    block_Green_sparse,
)
from impurityModel.ed.gf_units import (
    enumerate_gf_units,
    run_units_distributed,
    tolerance_cost_ratios,
    unit_cost_weights,
)
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, inner_multi
from impurityModel.ed.solver_trace import note as _trace_note
from impurityModel.ed.symmetries import widen_weighted_restrictions

comm = MPI.COMM_WORLD
rank = comm.rank


def build_full_greens_function(block_gf, block_structure: BlockStructure):
    """
    Assemble the full Green's function from individual blocks and block symmetries.

    Parameters
    ----------
    block_gf : list of ndarray
        Green's functions for each inequivalent block.
    block_structure : BlockStructure
        The block structure defining mapping and symmetry relationships.

    Returns
    -------
    res : ndarray
        The full Green's function matrix.
    """
    (
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_transposed_blocks,
        inequivalent_blocks,
    ) = block_structure
    if any(particle_hole_blocks) or any(particle_hole_transposed_blocks):
        # A particle-hole image is not an elementwise map of sampled values: G_B(i w) =
        # -conj(G_A(i w)) holds on the Matsubara axis, but on the retarded axis the image is
        # -conj(G_A(-w)), and moments map with order-dependent signs (Sigma_1 keeps its sign).
        # This function sees neither the axis nor the moment order, and the -conj it used to
        # apply put negative spectral weight on the real axis (review ledger C1). The block
        # structures the solver builds never carry particle-hole relations; those blocks are
        # computed directly.
        raise ValueError(
            "build_full_greens_function cannot reconstruct particle-hole related blocks; "
            "compute them directly (impurity_block_structure never emits particle-hole relations)"
        )
    n_orb = sum(len(block) for block in block_structure.blocks)
    if len(block_gf[0].shape) == 2:
        res = np.zeros((n_orb, n_orb), dtype=block_gf[0].dtype)
    elif len(block_gf[0].shape) == 3:
        res = np.zeros((block_gf[0].shape[0], n_orb, n_orb), dtype=block_gf[0].dtype)
    else:
        raise RuntimeError(
            f"Unknown data shape {block_gf[0].shape}. Should be 3 index (n_freq, n_orb,n_orb) or 2 index (n_orb,n_orb)"
        )
    if len(block_gf) == len(inequivalent_blocks):
        # block_gf contains only symmetrically inequivalent blocks
        for inequiv_i, gf_i in enumerate(block_gf):
            for block_i in identical_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                res[block_idx] = gf_i
            for block_i in transposed_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                    res[block_idx] = np.transpose(gf_i, (1, 0))
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                    res[block_idx] = np.transpose(gf_i, (0, 2, 1))
    else:
        raise RuntimeError(f"Block structure does not match block_gf.\n{block_structure=} {len(block_gf)=}")
    return res


def get_greens_function_moments(psis, es, tau, basis, hOp, impurity_indices, max_order=3):
    r"""Exact spectral moments of the interacting impurity Green's function.

    Returns ``M`` of shape ``(max_order + 1, n_corr, n_corr)`` (``n_corr = len(impurity_indices)``)
    in the *solver* basis, thermally averaged over the retained eigenstates, with ``M[0] = I``.
    These are the high-frequency coefficients of :math:`G(z) = \sum_n M_n / z^{n+1}`:

    .. math::

        M_n[a,b] = \frac{1}{Z} \sum_e e^{-(E_e-E_0)/\tau}
            \Big( \langle \psi_e| c_a (H-E_e)^n c_b^\dagger |\psi_e\rangle
                + (-1)^n \langle \psi_e| c_b^\dagger (H-E_e)^n c_a |\psi_e\rangle \Big),

    the addition (greater, poles at :math:`+(E_m-E_e)`) and removal (lesser, poles at
    :math:`-(E_m-E_e)`, hence the alternating sign) contributions -- the standard spectral
    decomposition of the anticommutator Green's function. Only two applications of ``(H - E_e)``
    per seed are needed for ``max_order <= 3``, using the Hermiticity of ``H - E_e`` (e.g.
    :math:`M_3^>[a,b] = \langle (H-E)c_a^\dagger\psi | (H-E)^2 c_b^\dagger\psi\rangle`).

    Parameters
    ----------
    psis : list of ManyBodyState
        The retained (thermal) eigenstates, distributed over ``basis``.
    es : array_like
        Their eigen-energies.
    tau : float
        Thermal energy scale ``k_B * T`` (same weighting as ``thermal_rho``).
    basis : Basis
        The ground-state basis owning ``psis`` (provides ``redistribute_psis``, ``comm``).
    hOp : ManyBodyOperator
        The interacting solver-basis Hamiltonian.
    impurity_indices : sequence of int
        Sorted impurity spin-orbital indices (the correlated block).
    max_order : int, optional
        Highest moment order to compute (default 3, i.e. ``M0..M3``).

    Returns
    -------
    np.ndarray
        ``(max_order + 1, n_corr, n_corr)`` complex moment tensor, ``M[0] = I``.

    Notes
    -----
    .. warning:: **Collective on** ``basis.comm``. Applies ``hOp`` -- which discovers
       determinants outside ``basis`` (the ``N \pm 1`` sectors); inner products stay correct
       because :meth:`Basis.redistribute_psis` routes purely by ``hash(sd) % size``, independent
       of basis membership -- and ``Allreduce``\ s the per-state moments. Call it unconditionally
       on every rank, exactly like :func:`basis_transcription.build_density_matrices`.
    """
    es = np.asarray(es, dtype=float)
    n_corr = len(impurity_indices)
    n_states = len(psis)
    # The moments are exact, so they need the full H. ``hOp`` is usually the solver Hamiltonian
    # the ground state and the Green's function just ran on, and both leave their occupation
    # windows on it (masks are sticky and cannot be read back). Clear them: with a window in
    # force, (H - E) applied to the seeds drops every out-of-window row and M_2, M_3 -- and so
    # sigma_moment_1/_2 -- are truncated (review ledger C3: up to 18%).
    hOp.set_restrictions(None)
    hOp.set_weighted_restrictions(None)
    # Per-state moments, indexed [state, order, a, b]. M[., 0] is the identity because
    # {c_a, c_b^dag} = delta_ab, so the greater/lesser contributions sum to I on every state.
    # Set it on exactly one rank (root): every other slot below is a genuine per-rank partial
    # sum that the Allreduce below combines into the whole, but this one is already the whole
    # answer on its own -- setting it unconditionally on every rank would make a distributed
    # Allreduce sum it comm.size times over, silently returning comm.size * I instead of I.
    M_per_state = np.zeros((n_states, max_order + 1, n_corr, n_corr), dtype=complex)
    if not basis.is_distributed or basis.comm.rank == 0:
        M_per_state[:, 0] = np.eye(n_corr)[np.newaxis, :, :]

    annihilation = [ManyBodyOperator({((orb, "a"),): 1.0}) for orb in impurity_indices]
    creation = [op.adjoint() for op in annihilation]

    def side_krylov_moments(seeds, e):
        """K[n][a,b] = <seed_a | (H - e)^n | seed_b> for n = 1..max_order (Hermitian H - e)."""
        s0 = seeds
        s1 = [hOp(s, 0) - complex(e) * s for s in s0]
        s2 = [hOp(s, 0) - complex(e) * s for s in s1] if max_order >= 3 else None
        if basis.is_distributed:
            s0 = basis.redistribute_psis(*s0)
            s1 = basis.redistribute_psis(*s1)
            s2 = basis.redistribute_psis(*s2) if s2 is not None else None
        k = {}
        if max_order >= 1:
            k[1] = inner_multi(s0, s1)  # <s0_a | (H-e) s0_b>
        if max_order >= 2:
            k[2] = inner_multi(s1, s1)  # <s0_a | (H-e)^2 s0_b>
        if max_order >= 3:
            k[3] = inner_multi(s1, s2)  # <s0_a | (H-e)^3 s0_b>
        return k

    for n, (psi_n, e_n) in enumerate(zip(psis, es)):
        # Greater (addition): seed c_a^dag|psi>, so greater[order][a,b] = <psi|c_a (H-E)^order c_b^dag|psi>.
        greater = side_krylov_moments([op(psi_n, 0) for op in creation], e_n)
        # Lesser (removal): seed c_a|psi> gives <psi|c_a (H-E)^order c_b|psi>; the moment needs the
        # mirrored index order <psi|c_b (H-E)^order c_a|psi>, hence the transpose, and the
        # (-1)^order sign of the removal poles at -(E_m - E).
        lesser = side_krylov_moments([op(psi_n, 0) for op in annihilation], e_n)
        for order in range(1, max_order + 1):
            M_per_state[n, order] = greater[order] + ((-1) ** order) * lesser[order].T

    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, M_per_state, op=MPI.SUM)

    return thermal_average_scale_indep(es, M_per_state, tau)


# Spectral side of a Green's-function work unit, in `get_Greens_function`'s `SIDES` order:
# 0 = addition (a creation operator on the thermal state, inverse photoemission),
# 1 = removal (an annihilation operator, photoemission).
_UNIT_SIDE_LABELS = ("create (addition)", "annihilate (removal)")


def _report_max_unit_basis(heading, rows):
    """Print the largest basis each work unit reached, one line per unit.

    Called once per completed calculation, on the rank that holds the assembled results
    (global rank 0, or every rank of a serial run). Shared by the Green's-function drivers
    here and by :func:`spectra.calc_spectra`, which differ in how they label a unit and in
    *which* basis they measure -- hence the caller-supplied ``heading`` and pre-formatted
    row labels. The sizes are global determinant counts (``Basis.size`` is the Allreduce'd
    total, not this rank's share).

    Not behind ``verbose``: sizing the next run's basis budget is exactly what someone reads
    off a finished calculation, whereas the existing per-unit lines
    (:func:`_block_green_group`, :func:`gf_solvers.block_Green_bicgstab`) are emitted mid-run
    at ``-vv``, interleaved with every other unit's output.

    Parameters
    ----------
    heading : str
        Names the measurement, e.g. ``"Maximum excited basis size per unit"``. Callers must
        distinguish measurements that are not the same construction -- the block-Lanczos
        excited basis and the per-frequency rebuilt solve basis are both "how big did this
        unit get" but are not comparable.
    rows : list of (str, int or None, bool)
        ``(label, size, cap_hit)`` per unit, in print order. ``size`` is ``None`` when the
        recurrence's support was not tracked (the sparse block-Lanczos path with no finite
        determinant cap -- only :class:`gf_primitives._CappedBasisProxy` counts the support
        the matvec discovers, and it is installed only under a finite cap). ``cap_hit`` marks
        a unit that froze at the cap, where the number reported *is* the cap and therefore
        says nothing about how large the unit wanted to be. An empty list prints nothing.
    """
    if not rows:
        return
    label_width = max(len(label) for label, _, _ in rows)
    tracked = [size for _, size, _ in rows if size is not None]
    size_width = max((len(f"{size:,}") for size in tracked), default=0)
    print(f"{heading}:", flush=True)
    for label, size, cap_hit in rows:
        if size is None:
            detail = "not tracked (no determinant cap set)"
        else:
            noun = "determinant" if size == 1 else "determinants"
            detail = f"{size:>{size_width},} {noun}" + (" -- frozen at the cap" if cap_hit else "")
        print(f"  {label:<{label_width}}  {detail}", flush=True)
    if tracked:
        # Qualify the headline number wherever it is not the plain answer to "how big did the
        # largest unit get": a frozen unit reports the cap, not its demand, and an untracked
        # unit is missing from the maximum entirely -- without which "maximum over all units:
        # 0 determinants" is what an uncapped run with two empty units prints.
        untracked = sum(1 for _, size, _ in rows if size is None)
        notes = []
        if any(cap_hit for _, size, cap_hit in rows if size is not None):
            notes.append("at least one unit froze at the cap, so this is a lower bound")
        if untracked:
            notes.append(f"{untracked} of {len(rows)} units not tracked and not counted here")
        scope = "the tracked units" if untracked else "all units"
        suffix = f" ({'; '.join(notes)})" if notes else ""
        top = max(tracked)
        print(f"  maximum over {scope}: {top:,} determinant{'' if top == 1 else 's'}{suffix}", flush=True)


def _unit_basis_rows(blocks, max_basis):
    """``(label, size, cap_hit)`` rows for a ``(block_i, side_i)``-keyed accumulator.

    The row label pads the orbital block and the spectral side into fixed columns, measured
    over the units that actually reported -- a block with no units contributes no line and no
    indentation. ``max_basis[key]`` is ``(size, cap_hit)``; see :func:`_report_max_unit_basis`
    for what a ``None`` size means.
    """
    labels = [str(block) for block in blocks]
    reported = [(bi, si) for bi in range(len(labels)) for si in range(len(_UNIT_SIDE_LABELS)) if (bi, si) in max_basis]
    if not reported:
        return []
    block_width = max(len(labels[bi]) for bi, _ in reported)
    side_width = max(len(_UNIT_SIDE_LABELS[si]) for _, si in reported)
    rows = []
    for bi, si in reported:
        size, cap_hit = max_basis[(bi, si)]
        rows.append((f"block {labels[bi]:<{block_width}}  {_UNIT_SIDE_LABELS[si]:<{side_width}}", size, cap_hit))
    return rows


def _merge_unit_basis(acc, key, size, cap_hit):
    """Fold one work unit's basis size into the per-unit maximum ``acc[key]``.

    Several units feed one ``(block, side)`` pair -- the eigenstate chunks -- and within one unit the
    basis only grows, so its final size is that unit's own maximum. Tracking is a property of
    the phase's cap, uniform across the units of a pair, so a ``None`` size never competes
    with a real one here; it is carried through so the row can say so.
    """
    prev_size, prev_cap = acc.get(key, (None, False))
    if size is None:
        merged = prev_size
    elif prev_size is None:
        merged = int(size)
    else:
        merged = max(prev_size, int(size))
    acc[key] = (merged, prev_cap or bool(cap_hit))


def get_Greens_function(
    matsubara_mesh: np.ndarray,
    omega_mesh: np.ndarray,
    psis: list[ManyBodyState],
    es: list[float],
    tau: float,
    basis: Basis,
    hOp: ManyBodyOperator,
    delta: float,
    blocks: list[list[int]],
    verbose: bool,
    verbose_extra: bool,
    reort: Optional[Reort],
    dN: Optional[int],
    occ_cutoff: float,
    slaterWeightMin: float,
    sparse: bool,
    num_wanted: int | None = None,
    gf_method: str = "lanczos",
    operator_families=None,
    gf_admission: Optional[str] = None,
    gf_admit_tol: Optional[float] = None,
    gf_tol: Optional[float] = None,
    gf_real_tol: Optional[float] = None,
    ensemble_es=None,
):
    """
    Calculate interacting Greens function.

    Returns ``(gs_matsubara, gs_realaxis, report)`` on the root rank, where ``report`` is a
    :class:`gf_diagnostics.DiagnosticReport` of per-block convergence/consistency checks
    (``(None, None, None)`` on non-root ranks). ``num_wanted`` is the number of thermal states
    the eigensolver was asked for, used by the ensemble-truncation check.

    ``gf_method`` selects the resolvent kernel: ``"lanczos"`` (default) runs one block-Lanczos
    recurrence per work unit serving the whole mesh; ``"bicgstab"`` solves one linear system
    per frequency point with a rebuilt-and-discarded basis (:func:`block_Green_bicgstab`).
    On the per-frequency path ``sparse`` is ignored (the solvers work on the ManyBodyState
    representation only).

    ``gf_admission`` is the per-frequency kernel's basis-growth policy: ``"all"`` or ``"outer"``
    (``None``, the default: ``GF_BICGSTAB_ADMISSION``, else ``"all"``). ``"outer"`` solves on a
    frozen basis, scores the residual outside it and admits only what clears
    ``gf_admit_tol`` (``None``: ``GF_BICGSTAB_ADMIT_TOL_AMP``), and repeats; see
    :mod:`impurityModel.ed.gf_admission`. ``"outer"`` also records a measured error bound in the
    diagnostics report. An explicit argument wins over the matching environment knob.

    ``gf_tol`` / ``gf_real_tol`` are the block-Lanczos convergence tolerances (relative change of
    ``G``) on every axis / on the real axis only (``None``: ``GF_TOL`` / ``GF_REAL_TOL``, else
    ``max(slaterWeightMin**2, 1e-9)`` on both axes; see :func:`gf_convergence._gf_axis_tols`).
    They govern the ``"lanczos"`` kernel only; passing either with ``"bicgstab"`` is an error.

    ``ensemble_es`` is the whole thermal ensemble the eigensolver returned, when the caller passes
    only part of it in ``es`` (``SolverOptions.gf_min_weight``); the ensemble-truncation check
    judges the eigensolver's output, not the subset. ``None``: ``es``.

    ``operator_families`` is the self-energy estimator seam
    (:mod:`impurityModel.ed.sigma_estimators`): ``operator_families(block)`` returns
    ``(addition_ops, removal_ops)``, two equally long operator lists. With ``removal_ops = X``
    and ``addition_ops = X^dag``, each returned block is the Green's function of the family X,
    ``G_ab(z) = <X_a (z - H + E)^-1 X_b^dag> + <X_b^dag (z + H - E)^-1 X_a>``. The contract: a
    family's leading ``len(block)`` operators are the block's own ``c^dag`` / ``c`` in block
    order, so the leading sub-block is the impurity Green's function and the anticommutator sum
    rule is checked on it. ``None`` is exactly that plain family, so each block is
    ``len(block)`` wide.
    """
    config.warn_retired_knobs()
    if gf_method in config.RETIRED_GF_METHODS:
        raise ValueError(f"gf_method {gf_method!r} {config.RETIRED_GF_METHODS[gf_method]}")
    if gf_method not in config.GF_METHODS:
        raise ValueError(f"Unknown gf_method {gf_method!r}; expected one of {', '.join(map(repr, config.GF_METHODS))}")
    if gf_admission is not None and gf_admission not in config.GF_ADMISSIONS:
        raise ValueError(
            f"Unknown gf_admission {gf_admission!r}; expected one of {', '.join(map(repr, config.GF_ADMISSIONS))}"
        )
    if gf_admission == "outer" and gf_method != "bicgstab":
        raise ValueError(f"gf_admission='outer' needs gf_method='bicgstab' (got {gf_method!r})")
    if gf_method != "lanczos" and (gf_tol is not None or gf_real_tol is not None):
        raise ValueError(f"gf_tol/gf_real_tol are block-Lanczos tolerances; gf_method={gf_method!r} ignores them")
    # Explicit arguments win over the knobs. Resolved (and validated) here, before any collective,
    # so a bad value fails on every rank at once.
    axis_tols = _gf_axis_tols(
        slaterWeightMin,
        config.GF_TOL.get() if gf_tol is None else gf_tol,
        config.GF_REAL_TOL.get() if gf_real_tol is None else gf_real_tol,
    )
    # GF_WEIGHTED_TOL: loosen the tolerance of low-weight eigenstates' units (their error enters G
    # multiplied by the weight). Rank 0's environment decides and is broadcast, so every rank of a
    # color derives identical tolerances -- a per-rank difference would desynchronise the monitor's
    # collectives. The bcast is unconditional on a communicator.
    weighted_tol = (config.GF_WEIGHTED_TOL.get(), config.GF_WEIGHTED_TOL_CEILING.get())
    if basis.comm is not None:
        weighted_tol = basis.comm.bcast(weighted_tol, root=0)
    use_weighted_tol = bool(weighted_tol[0]) and gf_method == "lanczos"
    if use_weighted_tol:
        thermal_weights = ThermalEnsemble(es, tau).weights
        max_thermal_weight = float(np.max(thermal_weights))
    # Excited-sector restrictions are independent of the orbital block and of the spectral side
    # (the dN occupation window is symmetric and spans all impurity orbitals), so build them once
    # on the full basis instead of per block.
    excited_restrictions, excited_weighted_restrictions = _build_excited_restrictions(
        basis, hOp, psis, es, dN, occ_cutoff, slater_weight_min=slaterWeightMin
    )
    # The per-unit kernels print the restrictions only when they exist, so an unrestricted run
    # would otherwise be silent -- indistinguishable from restrictions lost somewhere.
    if verbose and (basis.comm is None or basis.comm.rank == 0):
        if excited_restrictions is None:
            reason = " (dN unset: no occupation window)" if dN is None else ""
            print(f"Excited restrictions: none{reason}", flush=True)
        if excited_weighted_restrictions is None:
            print("Weight restrictions: none", flush=True)
    n_psis = len(psis)

    # Per-state excited windows (see _gf_per_state_restrict). Built on the full basis before the
    # split, from globally-reduced density matrices, so every rank holds the identical list and can
    # look up any unit's window locally. Cheap and identical to the ensemble window when the bath
    # classification is not state-dependent (chain_restrict off, or a directly-hybridizing shell).
    if _gf_per_state_restrict(basis.chain_restrict):
        per_state_restrictions = [
            _build_excited_restrictions(
                basis, hOp, [psis[ei]], [es[ei]], dN, occ_cutoff, slater_weight_min=slaterWeightMin
            )[0]
            for ei in range(n_psis)
        ]
    else:
        per_state_restrictions = None

    # --- Enumerate the work units = (block, addition/removal, eigenstate-group) -----------
    # One operator group per (block, spectral side); the flat unit decomposition, cost model and
    # single split are the shared engine (enumerate_gf_units / unit_cost_weights /
    # run_units_distributed), load-balanced across the full (block x side x eigenstate)
    # cross-product -- important when there are many small symmetry blocks (the typical
    # production case).
    if operator_families is None:
        operator_families = impurity_operator_family
    SIDE_DELTAS = (delta, -delta)  # 0 = addition (IPS), 1 = removal (PS)
    op_groups = []
    group_meta = []  # (block_i, side_i) per operator group
    widths = []  # G width per block: the family's operator count
    for block_i, block in enumerate(blocks):
        families = operator_families(block)
        if len(families[0]) != len(families[1]) or len(families[0]) < len(block):
            raise ValueError(
                f"operator family for block {block} has {len(families[0])} addition and {len(families[1])} "
                f"removal operators; both must be equal and at least len(block) = {len(block)}"
            )
        widths.append(len(families[0]))
        for side_i, ops in enumerate(families):
            op_groups.append((ops, SIDE_DELTAS[side_i]))
            group_meta.append((block_i, side_i))
    units, unit_seeds, unit_restrictions = enumerate_gf_units(
        op_groups,
        psis,
        [excited_restrictions] * len(op_groups),
        excited_weighted_restrictions,
        slaterWeightMin,
        per_state_restrictions,
    )
    unit_weights = unit_cost_weights(unit_seeds, basis.comm)

    def unit_axis_tols(unit):
        """This unit's per-axis tolerances: ``axis_tols``, loosened by its thermal weight if enabled."""
        if not use_weighted_tol:
            return axis_tols
        return _weighted_axis_tols(
            axis_tols,
            float(max(thermal_weights[ei] for ei in unit.chunk)),
            max_thermal_weight,
            weighted_tol[1],
        )

    if use_weighted_tol:
        # A unit converged to a looser tolerance stops after a fraction of the blocks, so it must not
        # weigh the same as a dominant one: the packer would give it a colour's worth of ranks, and the
        # queue would hold the units that set the wall behind it.
        unit_weights = unit_weights * tolerance_cost_ratios([min(unit_axis_tols(u)) for u in units], min(axis_tols))

    if gf_method == "bicgstab":
        return _get_greens_function_bicgstab(
            matsubara_mesh,
            omega_mesh,
            es,
            tau,
            basis,
            hOp,
            delta,
            blocks,
            widths,
            units,
            unit_seeds,
            unit_weights,
            unit_restrictions,
            group_meta,
            excited_weighted_restrictions,
            slaterWeightMin,
            verbose,
            verbose_extra,
            num_wanted,
            gf_admission=gf_admission,
            gf_admit_tol=gf_admit_tol,
        )

    def eval_meshes_for(unit):
        # Converge G where this unit's G will actually be evaluated: the caller's meshes, shifted
        # by each thermal energy the unit stacks and signed by its spectral side. Without this the
        # monitor resolves the real-axis resolvent at broadening `delta` even for a Matsubara-only
        # self-energy, which costs 3.6-4.1x the blocks such a run needs.
        return _gf_eval_meshes(
            matsubara_mesh,
            omega_mesh,
            group_meta[unit.group_i][1],
            delta,
            [es[ei] for ei in unit.chunk],
            axis_tols=unit_axis_tols(unit),
        )

    kernel = lanczos_unit_kernel(
        units,
        hOp,
        unit_restrictions,
        excited_weighted_restrictions,
        reort=reort,
        sparse=sparse,
        slaterWeightMin=slaterWeightMin,
        solver_verbose=verbose_extra,
        print_windows=verbose,
        print_size=verbose_extra,
        eval_meshes_for=eval_meshes_for,
    )

    # This unit-level dump belongs at -vv (verbose_extra): the roots/per-color summary is
    # detail beyond the -v per-block roll-up the caller already prints.
    results = run_units_distributed(
        basis, unit_seeds, unit_weights, kernel, verbose=verbose_extra, reort=reort, unit_windows=unit_restrictions
    )

    gs_matsubara = gs_realaxis = report = None
    if results is not None:
        assert isinstance(results, list)  # this call passes no reduce_fn, so results is the gathered list
        # Reassemble the per-unit results (global unit order) into per-(block, side)
        # eigenstate-indexed coefficient lists. acc[(block_i, side_i)] = (alphas_list, betas_list,
        # r_list) indexed by eigenstate; r_list[ei] is that eigenstate's seed-projection matrix.
        acc = dict(zip(group_meta, states_by_group(units, results, len(group_meta), n_psis)))
        # Worst-case cap state per block over all its (side, eigenstate) solves: any
        # frozen solve marks the block; retained_size is the smallest frozen size.
        cap_acc: dict[int, dict] = {}
        # Worst-case convergence verdict per block, over all its (side, eigenstate) solves --
        # the solver's own runtime verdict (see block_Green_sparse/block_green_impl's `info`),
        # not a post-hoc recompute: a block reports converged only if every unit feeding it did,
        # and d_g/n_blocks take the worst (max) over those units.
        conv_acc: dict[int, dict] = {}
        # Largest excited basis per (block, spectral side) work unit, over the eigenstate
        # chunks feeding it. Separate from `cap_acc`, which takes
        # the *smallest* frozen size and only of the units that actually hit the cap.
        max_basis: dict[tuple[int, int], tuple[Optional[int], bool]] = {}
        for unit, (_alphas, _betas, _r_slices, cap_stats, conv_stats) in zip(units, results):
            block_i, unit_side_i = group_meta[unit.group_i]
            _merge_unit_basis(max_basis, (block_i, unit_side_i), cap_stats.get("retained_size"), cap_stats["cap_hit"])
            # Root-side record of every unit (the kernel's own `gf_unit_memory` note is emitted per
            # color, so a traced rank 0 sees only its own color's units there).
            _trace_note(
                "gf_unit_basis",
                method="lanczos",
                block=int(block_i),
                side=int(unit_side_i),
                retained_size=cap_stats.get("retained_size"),
                seed_size=cap_stats.get("seed_size"),
                cap_hit=bool(cap_stats["cap_hit"]),
                n_blocks=conv_stats.get("n_blocks"),
            )
            stats = cap_acc.setdefault(block_i, {"cap_hit": False, "retained_size": None, "cap": cap_stats["cap"]})
            seed_size = cap_stats.get("seed_size")
            if seed_size is not None and np.isfinite(cap_stats["cap"]) and seed_size >= cap_stats["cap"]:
                stats["seed_frozen"] = True
            if cap_stats.get("memory_frozen"):
                stats["memory_frozen"] = True
            if cap_stats["cap_hit"]:
                stats["cap_hit"] = True
                stats["cap"] = cap_stats["cap"]
                retained = cap_stats.get("retained_size")
                if retained is not None and (stats["retained_size"] is None or retained < stats["retained_size"]):
                    stats["retained_size"] = retained
            cstats = conv_acc.setdefault(
                block_i,
                {
                    "converged": True,
                    "d_g": 0.0,
                    "n_blocks": 0,
                    # Under GF_WEIGHTED_TOL the units' own tolerances differ, so the block reports
                    # against the base (strictest) one and each unit's d_g is rescaled to it below.
                    "tol": min(axis_tols) if use_weighted_tol else conv_stats.get("tol", np.nan),
                },
            )
            cstats["converged"] = cstats["converged"] and bool(conv_stats.get("converged", True))
            # A trivially-converged (empty seed) unit reports d_g=nan -- it made no measurement,
            # so it must not poison the block's worst-case max with a NaN.
            unit_d_g = conv_stats.get("d_g")
            if unit_d_g is not None and not np.isnan(unit_d_g):
                if use_weighted_tol and conv_stats.get("tol"):
                    # d_g / (this unit's tol / base tol): 1.0 for a dominant unit, so a unit that
                    # met its loosened tolerance does not read as unconverged against the base.
                    unit_d_g = unit_d_g * (cstats["tol"] / conv_stats["tol"])
                cstats["d_g"] = max(cstats["d_g"], unit_d_g)
            cstats["n_blocks"] = max(cstats["n_blocks"], conv_stats.get("n_blocks", 0))

        thermal = ThermalEnsemble(es, tau)
        e0, Z = thermal.e0, thermal.Z
        gs_matsubara = (
            [np.empty((len(matsubara_mesh), n, n), dtype=complex) for n in widths]
            if matsubara_mesh is not None
            else None
        )
        gs_realaxis = (
            [np.empty((len(omega_mesh), n, n), dtype=complex) for n in widths] if omega_mesh is not None else None
        )
        report = _gfd.DiagnosticReport()
        for block_i, block in enumerate(blocks):
            a_add, b_add, r_add = acc[(block_i, 0)]
            a_rem, b_rem, r_rem = acc[(block_i, 1)]
            if matsubara_mesh is not None:
                G_IPS = calc_thermally_averaged_G(a_add, b_add, r_add, matsubara_mesh, es, e0, tau, 0)
                G_PS = calc_thermally_averaged_G(a_rem, b_rem, r_rem, -matsubara_mesh, es, e0, tau, 0)
                gs_matsubara[block_i][:] = combine_sides(G_IPS, G_PS, Z)
            G_IPS_real = G_PS_real = combined_real = None
            if omega_mesh is not None:
                G_IPS_real = calc_thermally_averaged_G(a_add, b_add, r_add, omega_mesh, es, e0, tau, delta)
                G_PS_real = calc_thermally_averaged_G(a_rem, b_rem, r_rem, -omega_mesh, es, e0, tau, -delta)
                combined_real = combine_sides(G_IPS_real, G_PS_real, Z)
                gs_realaxis[block_i][:] = combined_real

            # --- per-block convergence / consistency diagnostics ---------------------------
            ensemble = es if ensemble_es is None else ensemble_es
            diags = [
                _gfd.check_thermal_weight_cutoff(ensemble, e0, tau, n_returned=len(ensemble), num_wanted=num_wanted)
            ]
            block_cap = cap_acc.get(block_i)
            if block_cap is not None:
                diags.append(
                    _gfd.check_basis_truncation(
                        block_cap["cap_hit"],
                        block_cap["retained_size"],
                        block_cap["cap"],
                        seed_frozen=block_cap.get("seed_frozen", False),
                        memory_frozen=block_cap.get("memory_frozen", False),
                    )
                )
            # The sum rule holds for the plain c/c^dag part of a family: its leading len(block) columns.
            n_c = len(block)
            if widths[block_i] != n_c:
                r_add, r_rem = ([r[:, :n_c] for r in rs] for rs in (r_add, r_rem))
            diags.insert(0, _gfd.check_spectral_sum_rule(r_add, r_rem, es, e0, tau, n_c))
            lanczos_tol = min(axis_tols)
            # The solver's own runtime verdict (block_Green_sparse/block_green_impl's
            # `converged_fn`, tested on the caller's actual eval_meshes) -- not a recompute.
            conv_stats = conv_acc.get(block_i, {"converged": True, "d_g": 0.0, "n_blocks": 0, "tol": lanczos_tol})
            diags.append(
                _gfd.check_lanczos_convergence(
                    conv_stats["converged"], conv_stats["d_g"], conv_stats["n_blocks"], conv_stats["tol"]
                )
            )
            # Complementary, band-wide measure: convergence of the *whole* resolved Ritz
            # band, not just the caller's evaluation mesh -- signals spectral weight the
            # solver never had to (and didn't) resolve, e.g. outside the omega window.
            # A real-axis band measure, so it is judged at the real-axis tolerance.
            band_tol = axis_tols[1]
            if use_weighted_tol:
                # Per eigenstate, in units of that state's loosened real-axis tolerance, so the
                # light states (stopped early by design) do not WARN against the dominant state's.
                band_value = 0.0
                for ei in range(len(es)):
                    state_tol = _weighted_axis_tols(
                        axis_tols, float(thermal_weights[ei]), max_thermal_weight, weighted_tol[1]
                    )[1]
                    for a_s, b_s, sgn in ((a_add, b_add, delta), (a_rem, b_rem, -delta)):
                        d_band = _lanczos_convergence_summary([a_s[ei]], [b_s[ei]], sgn, tol=band_tol)[1]
                        band_value = max(band_value, d_band * (band_tol / state_tol))
            else:
                conv_add = _lanczos_convergence_summary(a_add, b_add, delta, tol=band_tol)
                conv_rem = _lanczos_convergence_summary(a_rem, b_rem, -delta, tol=band_tol)
                band_value = max(conv_add[1], conv_rem[1])
            diags.append(_gfd.check_lanczos_band_resolution(band_value, band_tol))
            if G_IPS_real is not None:
                diags.append(_gfd.check_mesh_density(omega_mesh, delta))
                diags.append(
                    _gfd.check_integrated_weight(G_IPS_real, r_add, es, e0, tau, omega_mesh, "add", delta=delta)
                )
                diags.append(
                    _gfd.check_integrated_weight(G_PS_real, r_rem, es, e0, tau, -omega_mesh, "rem", delta=delta)
                )
                diags.append(_gfd.check_causality(combined_real, "G"))
            report.extend(str(block), diags)

        _report_max_unit_basis("Maximum excited basis size per unit", _unit_basis_rows(blocks, max_basis))

    return (gs_matsubara, gs_realaxis, report)


def impurity_operator_family(block):
    """The impurity Green's-function family of ``block``: ``([c_i^dag], [c_i])`` in block order."""
    return tuple([ManyBodyOperator({((orb, op_char),): 1}) for orb in block] for op_char in ("c", "a"))


def _get_greens_function_bicgstab(
    matsubara_mesh,
    omega_mesh,
    es,
    tau,
    basis,
    hOp,
    delta,
    blocks,
    widths,
    units,
    unit_seeds,
    unit_weights,
    unit_restrictions,
    group_meta,
    excited_weighted_restrictions,
    slaterWeightMin,
    verbose,
    verbose_extra,
    num_wanted,
    gf_admission=None,
    gf_admit_tol=None,
):
    r"""Distribution + assembly of the per-frequency BiCGSTAB Green's function.

    The unit decomposition (and the excited windows) are exactly the Lanczos driver's --
    :func:`get_Greens_function` hands them over after :func:`enumerate_gf_units` -- only the
    per-unit kernel and the result contract differ: each unit returns ``G`` already evaluated
    on the caller's meshes (:func:`block_Green_bicgstab`); the shared assembler :func:`_run_evaluated_gf_units` does the
    rest.
    """

    def kernel(split_basis, u, seeds):
        unit = units[u]
        _block_i, side_i = group_meta[unit.group_i]
        z_axes = _gf_signed_axes(matsubara_mesh, omega_mesh, side_i, delta)
        return block_Green_bicgstab(
            hOp,
            seeds,
            split_basis,
            [es[ei] for ei in unit.chunk],
            unit.n_ops,
            z_axes,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose_extra,
            excited_restrictions=unit_restrictions[u],
            excited_weighted_restrictions=excited_weighted_restrictions,
            admission=gf_admission,
            admit_tol=gf_admit_tol,
        )

    units_meta = [(group_meta[unit.group_i][0], group_meta[unit.group_i][1], unit.chunk) for unit in units]
    return _run_evaluated_gf_units(
        matsubara_mesh,
        omega_mesh,
        es,
        tau,
        basis,
        delta,
        blocks,
        widths,
        units_meta,
        unit_seeds,
        unit_weights,
        kernel,
        verbose,
        num_wanted,
        unit_windows=unit_restrictions,
    )


def _run_evaluated_gf_units(
    matsubara_mesh,
    omega_mesh,
    es,
    tau,
    basis,
    delta,
    blocks,
    widths,
    units_meta,
    unit_seeds,
    unit_weights,
    kernel,
    verbose,
    num_wanted,
    unit_windows=None,
):
    r"""Distribute, accumulate and assemble Green's-function units that return evaluated ``G``.

    The engine behind the per-frequency drivers: ``kernel(split_basis,
    u, seeds)`` must return ``(G_axes, stats)`` in :func:`block_Green_bicgstab`'s contract,
    and ``units_meta[u] = (block_i, side_i, chunk)`` names where unit ``u``'s result belongs.
    The assembly is a streaming Boltzmann-weighted accumulation into per-``(block, side)``
    arrays (rank 0 never holds more than one color's payload) followed by the same
    :math:`(G_\mathrm{IPS} - G_\mathrm{PS}^T)/Z` combination the Lanczos path applies to its
    evaluated continued fractions.

    The diagnostics report keeps the representation-independent checks (thermal cutoff, mesh
    density, causality, basis truncation) plus the solver-residual record
    (:func:`gf_diagnostics.check_bicgstab_convergence`). The spectral sum rule
    and integrated-weight checks are expressed in seed-projection/continued-fraction terms
    these paths do not produce.
    """
    thermal = ThermalEnsemble(es, tau)
    e0, boltzmann, Z = thermal.e0, thermal.weights, thermal.Z
    axis_lens = [len(m) for m in (matsubara_mesh, omega_mesh) if m is not None]

    # Streaming accumulators, populated on global rank 0 only (reduce_fn's contract).
    is_root = basis.comm is None or basis.comm.rank == 0
    G_acc = (
        {
            (bi, si): [np.zeros((L, widths[bi], widths[bi]), dtype=complex) for L in axis_lens]
            for bi in range(len(blocks))
            for si in (0, 1)
        }
        if is_root
        else None
    )
    stats_acc = {} if is_root else None
    # Largest per-frequency solve basis per (block, spectral side) work unit. `stats_acc` keys
    # on the block alone (its diagnostics are per-block) and its `retained_size` is the
    # *smallest* frozen size, so neither answers "how big did this unit's basis get". This is a
    # different construction from the block-Lanczos path's excited basis -- a basis rebuilt and
    # discarded per frequency point, not one recurrence's support -- so it is reported under
    # its own heading below and the two numbers must not be compared.
    max_basis_acc = {} if is_root else None

    def reduce_fn(u, result):
        G_axes, stats = result
        block_i, side_i, chunk = units_meta[u]
        _merge_unit_basis(max_basis_acc, (block_i, side_i), stats["max_solve_basis"], stats["cap_hit"])
        _trace_note(
            "gf_unit_basis",
            method="bicgstab",
            block=int(block_i),
            side=int(side_i),
            retained_size=stats["max_solve_basis"],
            max_rebuild_basis=stats["max_rebuild_basis"],
            cap_hit=bool(stats["cap_hit"]),
            points=stats["points"],
        )
        for p, ei in enumerate(chunk):
            for ax in range(len(axis_lens)):
                G_acc[(block_i, side_i)][ax] += boltzmann[ei] * G_axes[ax][p]
        agg = stats_acc.setdefault(
            block_i,
            {
                "n_points": 0,
                "n_unconverged": 0,
                "max_rel_residual": 0.0,
                "iterations": 0,
                "gmres_points": 0,
                "gmres_iterations": 0,
                "atol": stats["atol"],
                "cap": stats["cap"],
                "cap_hit": False,
                "retained_size": None,
                "seed_overflow": False,
                "max_solve_basis": 0,
                "max_rebuild_basis": 0,
                "max_dG_bound": None,
                "max_boundary": None,
            },
        )
        for key in ("n_points", "n_unconverged", "iterations", "gmres_points", "gmres_iterations"):
            agg[key] += stats[key]
        for key in ("max_rel_residual", "max_solve_basis", "max_rebuild_basis"):
            agg[key] = max(agg[key], stats[key])
        for key in ("max_dG_bound", "max_boundary"):
            if stats[key] is not None:
                agg[key] = max(agg[key] or 0.0, stats[key])
        agg["cap_hit"] = agg["cap_hit"] or stats["cap_hit"]
        agg["seed_overflow"] = agg["seed_overflow"] or stats["seed_overflow"]
        if stats["retained_size"] is not None:
            agg["retained_size"] = (
                stats["retained_size"]
                if agg["retained_size"] is None
                else min(agg["retained_size"], stats["retained_size"])
            )

    got = run_units_distributed(
        basis,
        unit_seeds,
        unit_weights,
        kernel,
        verbose=verbose,
        reduce_fn=reduce_fn,
        gf_method="bicgstab",
        unit_windows=unit_windows,
    )
    if got is None:
        return None, None, None

    gs_matsubara = (
        [np.empty((len(matsubara_mesh), n, n), dtype=complex) for n in widths] if matsubara_mesh is not None else None
    )
    gs_realaxis = [np.empty((len(omega_mesh), n, n), dtype=complex) for n in widths] if omega_mesh is not None else None
    report = _gfd.DiagnosticReport()
    for block_i, block in enumerate(blocks):
        ax = 0
        combined_real = None
        if matsubara_mesh is not None:
            G_IPS = G_acc[(block_i, 0)][ax]
            G_PS = G_acc[(block_i, 1)][ax]
            gs_matsubara[block_i][:] = combine_sides(G_IPS, G_PS, Z)
            ax += 1
        if omega_mesh is not None:
            G_IPS_real = G_acc[(block_i, 0)][ax]
            G_PS_real = G_acc[(block_i, 1)][ax]
            combined_real = combine_sides(G_IPS_real, G_PS_real, Z)
            gs_realaxis[block_i][:] = combined_real

        agg = stats_acc[block_i]
        if verbose:
            print(
                f"block {block}: {agg['n_points']} bicgstab solves, {agg['iterations']} iterations "
                f"({agg['gmres_points']} GMRES-fallback points, {agg['gmres_iterations']} of the iterations), "
                f"max per-point basis {agg['max_solve_basis']:,} "
                f"(rebuild floor {agg['max_rebuild_basis']:,}), "
                f"max residual {agg['max_rel_residual']:.1e}",
                flush=True,
            )
        diags = [
            _gfd.check_thermal_weight_cutoff(es, e0, tau, n_returned=len(es), num_wanted=num_wanted),
            _gfd.check_bicgstab_convergence(
                agg["n_points"],
                agg["n_unconverged"],
                agg["max_rel_residual"],
                agg["atol"],
                seed_overflow=agg["seed_overflow"],
                n_gmres_fallbacks=agg["gmres_points"],
            ),
        ]
        if np.isfinite(agg["cap"]):
            diags.append(_gfd.check_basis_truncation(agg["cap_hit"], agg["retained_size"], agg["cap"]))
        if agg["max_dG_bound"] is not None:
            diags.append(_gfd.check_truncation_error_bound(agg["max_dG_bound"], agg["max_boundary"]))
        if combined_real is not None:
            diags.append(_gfd.check_mesh_density(omega_mesh, delta))
            diags.append(_gfd.check_causality(combined_real, "G"))
        report.extend(str(block), diags)

    _report_max_unit_basis("Maximum per-frequency solve basis size per unit", _unit_basis_rows(blocks, max_basis_acc))

    return gs_matsubara, gs_realaxis, report


def _build_excited_restrictions(
    basis, hOp, psis, es, dN, occ_cutoff, dN_imp=None, dN_val=None, dN_con=None, slater_weight_min=None
):
    """Build the excited-sector occupation restrictions for a Green's-function calculation.

    The window widens the ground-state impurity occupation by ``dN`` symmetrically (so it admits
    both the addition ``c_j^\\dagger`` and removal ``c_j`` sectors) and is therefore *independent
    of the orbital block and of the spectral side* -- it depends only on ``(hOp, psis, es, dN)``.
    Shared by :func:`calc_Greens_function_with_offdiag` (per block) and :func:`get_Greens_function`
    (computed once for all blocks).

    Returns
    -------
    tuple
        ``(excited_restrictions, excited_weighted_restrictions)``.
    """
    if dN_imp is None:
        if dN is not None:
            dN_imp = dict.fromkeys(basis.impurity_orbitals, (dN, dN))
    else:
        dN_imp = {i: dN_imp.get(i) for i in basis.impurity_orbitals}
    if dN_val is None:
        if dN is not None:
            dN_val = dict.fromkeys(basis.impurity_orbitals, (dN, 0))
    else:
        dN_val = {i: dN_val.get(i) for i in basis.impurity_orbitals}
    if dN_con is None:
        if dN is not None:
            dN_con = dict.fromkeys(basis.impurity_orbitals, (0, dN))
    else:
        dN_con = {i: dN_con.get(i) for i in basis.impurity_orbitals}
    excited_restrictions = build_excited_restrictions(
        basis,
        hOp,
        psis,
        es,
        imp_change=dN_imp,
        val_change=dN_val,
        con_change=dN_con,
        cutoff=occ_cutoff,
        slater_weight_min=slater_weight_min,
    )
    # Weighted (e.g. S_z) restriction for the excited sector: widen the ground-state bounds by one
    # orbital weight so the addition / removal sectors q_psi ± w_j are admitted while still
    # confining the basis.
    excited_weighted_restrictions = widen_weighted_restrictions(basis.weighted_restrictions)
    return excited_restrictions, excited_weighted_restrictions


def calc_Greens_function_with_offdiag(
    hOp,
    tOps,
    psis,
    es,
    block_basis,
    delta,
    reort: Optional[Reort] = None,
    dN: Optional[int] = None,
    occ_cutoff: float = 1e-6,
    slaterWeightMin: float = 0,
    verbose: bool = True,
    sparse: bool = False,
    dN_imp=None,
    dN_val=None,
    dN_con=None,
    extra_restrictions=None,
    unit_report_label=None,
):
    r"""
    Return block-Lanczos Green's-function coefficients for the given transition operators.

    For states :math:`|psi \rangle`, the coefficients represent:

    :math:`g(w+1j*delta) =
    = \langle psi| tOp^\dagger ((w+1j*delta+e)*\hat{1} - hOp)^{-1} tOp
    |psi \rangle`,

    where :math:`e = \langle psi| hOp |psi \rangle`.

    Thin wrapper over the shared distribution engine: the (tOps x eigenstate-chunk) work units
    are enumerated by :func:`enumerate_gf_units`, weighted by :func:`unit_cost_weights` and run
    through :func:`run_units_distributed` (one split over all units).

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian operator.
    tOps : list of ManyBodyOperator
        Transition operators; together they form one Green's-function block of width
        ``len(tOps)``.
    psis : list of ManyBodyState
        Thermal eigenstates.
    es : list of float
        Total energies of the eigenstates.
    block_basis : Basis
        The basis container (carries the communicator).
    delta : float
        Deviation from the real axis (broadening/resolution parameter).
    slaterWeightMin : float
        Restrict the number of product states by looking at ``|amplitudes|^2``.
    unit_report_label : str, optional
        When given, the completed calculation prints the maximum excited basis size its work
        units reached, under a heading naming this spectrum
        (:func:`_report_max_unit_basis`). One operator group means one reported row -- the
        maximum over the eigenstate chunks sharing the recurrence. ``None`` (the default)
        prints nothing.
    extra_restrictions : dict, optional
        Conserved-charge sector confinement, intersected onto the excited-sector occupation
        window (it can only tighten the excited basis, never loosen it).

    Returns
    -------
    tuple
        ``(excited_alphas, excited_betas, excited_r)`` -- per-eigenstate block-tridiagonal
        coefficients and seed projections on rank 0 of ``block_basis.comm`` (and on every rank
        in the serial path); ``(None, None, None)`` elsewhere.
    """

    # Set limits for change occupation, if any. Limits are pairs of integers (max_holes, max_el),
    # imposed on top of the (effective) ground-state limitations. The window is block- and
    # side-independent (see _build_excited_restrictions).
    excited_restrictions, excited_weighted_restrictions = _build_excited_restrictions(
        block_basis,
        hOp,
        psis,
        es,
        dN,
        occ_cutoff,
        dN_imp=dN_imp,
        dN_val=dN_val,
        dN_con=dN_con,
        slater_weight_min=slaterWeightMin,
    )
    # Optional conserved-charge sector confinement (symmetries.transition_sector_restrictions):
    # pins the seed's charge sector on top of the per-shell occupation window, pruning
    # sector-violating determinants the window alone would admit. Intersected key-by-key so it
    # can only tighten the excited basis, never loosen it.
    if extra_restrictions:
        excited_restrictions = intersect_windows(excited_restrictions, extra_restrictions)
    if verbose and excited_restrictions is not None and (block_basis.comm is None or block_basis.comm.rank == 0):
        print("Excited state restrictions:")
        for indices, occupations in excited_restrictions.items():
            print(f"---> {sorted(indices)} : {occupations}")

    # One operator group holding the whole tOps block.
    units, unit_seeds, unit_restrictions = enumerate_gf_units(
        [(tOps, delta)],
        psis,
        [excited_restrictions],
        excited_weighted_restrictions,
        slaterWeightMin,
    )
    unit_weights = unit_cost_weights(unit_seeds, block_basis.comm)

    kernel = lanczos_unit_kernel(
        units,
        hOp,
        unit_restrictions,
        excited_weighted_restrictions,
        reort=reort,
        sparse=sparse,
        slaterWeightMin=slaterWeightMin,
        solver_verbose=verbose,
        print_size=verbose,
    )
    results = run_units_distributed(
        block_basis, unit_seeds, unit_weights, kernel, verbose=verbose, reort=reort, unit_windows=unit_restrictions
    )

    excited_alphas = excited_betas = excited_r = None
    if results is not None:
        assert isinstance(results, list)  # this call passes no reduce_fn, so results is the gathered list
        ((excited_alphas, excited_betas, excited_r),) = states_by_group(units, results, 1, len(psis))
        # This function enumerates ONE operator group -- the whole `tOps` block shares a single
        # recurrence -- so the units are its eigenstate chunks and they collapse to one reported
        # row, the maximum over them.
        max_basis: dict[int, tuple[Optional[int], bool]] = {}
        for _alphas, _betas, _r_slices, cap_stats, _conv in results:
            _merge_unit_basis(max_basis, 0, cap_stats["retained_size"], cap_stats["cap_hit"])
        if unit_report_label is not None and 0 in max_basis:
            _report_max_unit_basis(
                f"Maximum excited basis size per unit -- {unit_report_label}",
                [("transition block", *max_basis[0])],
            )

    return excited_alphas, excited_betas, excited_r


def _gf_per_state_restrict(chain_restrict):
    r"""Whether to build the excited-sector occupation window *per thermal state* (per work unit)
    instead of once from the whole thermal ensemble.

    Default: **on exactly when ``chain_restrict`` is on**. Per-state windows differ from the
    ensemble window only through the state-dependent bath filled/empty classification, which is
    itself only produced under ``chain_restrict`` (and only for sites past the coupling-distance
    filter -- long chains); with ``chain_restrict`` off the two are identical, so per-state would
    be pure overhead. :data:`config.GF_PER_STATE_RESTRICT` overrides the default either way.

    The ensemble window is effectively the union over all thermal states' filled/empty bath
    classifications: a bath orbital counts as cleanly filled/empty only if the *thermal-average*
    occupation is within ``occ_cutoff`` of 1/0. A single eigenstate usually pins strictly more baths
    (its own occupations are 0/1 to machine precision where the ensemble average is merely close),
    so its own window carries more restriction subsets -> a smaller excited basis and cheaper
    Lanczos. Each work unit uses the *union* of the per-state windows over the eigenstates it stacks
    (:func:`basis_restrictions.union_windows`) so the shared block Krylov space still contains every seed's
    dynamics; at the default ``GF_EIGENSTATE_GROUP=1`` every unit is a single state, giving the full
    per-state tightening. The seed ``c_i|psi_e>`` is unchanged (an impurity operator preserves bath
    occupation, so the seed lies inside ``psi_e``'s own window), so only the excited-basis span
    tightens -- no seed is truncated.

    This differs from the ensemble window *only* when the bath filled/empty classification is
    state-dependent, i.e. ``chain_restrict=True`` with sites far enough from the impurity to clear
    the coupling-distance filter (long chains). For a directly-hybridizing single bath shell the
    per-state and ensemble windows are identical and this is a no-op.
    """
    override = config.GF_PER_STATE_RESTRICT.get()
    if override is None:
        return bool(chain_restrict)
    return override


def save_Greens_function(gs, omega_mesh, label, cluster_label, e_scale=1, tol=1e-8, directory=None):
    """
    Save Greens function to file, using RSPt .dat format. Including offdiagonal elements.

    The files go to ``directory``, or the current directory when it is ``None`` -- which is
    where RSPt's interface reads them from, so that stays the default.

    Caller contract: every in-tree caller invokes this only on rank 0 (selfenergy.py's
    unphysical-result save and scripts/selfenergy.py's ``_save_results``, both already
    inside a ``rank == 0`` guard), so the prints/file writes below are unconditional --
    this function itself takes no ``comm`` to gate on.
    """
    n_orb = gs.shape[1]
    axis_label = "-realaxis"
    if np.all(np.abs(np.imag(omega_mesh)) > 1e-6):
        omega_mesh = np.imag(omega_mesh)
        axis_label = ""

    off_diags = []
    for column in range(gs.shape[2]):
        for row in range(gs.shape[1]):
            if row == column:
                continue
            if np.any(np.abs(gs[:, row, column]) > tol):
                off_diags.append((row, column))

    print(f"Writing {label}{axis_label}-{cluster_label} to files")
    directory = "." if directory is None else directory
    with (
        open(os.path.join(directory, f"real-{label}{axis_label}-{cluster_label}.dat"), "w") as fg_real,
        open(os.path.join(directory, f"imag-{label}{axis_label}-{cluster_label}.dat"), "w") as fg_imag,
    ):
        header = "# Frequency, total, spin down, spin up\n"
        header += "# indexmap: (column index of projected elements)"
        for row in range(gs.shape[1]):
            header += "\n# "
            for column in range(gs.shape[2]):
                if row == column:
                    header += f"{5 + row:< 4d}"
                elif (row, column) in off_diags:
                    header += f"{5 + n_orb + off_diags.index((row, column)):< 4d}"
                else:
                    header += f"{0:< 4d}"
        fg_real.write(header + "\n")
        fg_imag.write(header + "\n")
        for i, w in enumerate(omega_mesh):
            fg_real.write(
                f"{w * e_scale} {np.real(np.sum(np.diag(gs[i, :, :]))) / e_scale} "
                + f"{np.real(np.sum(np.diag(gs[i, : n_orb // 2, : n_orb // 2]))) / e_scale} "
                + f"{np.real(np.sum(np.diag(gs[i, n_orb // 2 :, n_orb // 2 :]))) / e_scale} "
                + " ".join(f"{np.real(el) / e_scale}" for el in np.diag(gs[i, :, :]))
                + " "
                + " ".join(f"{np.real(gs[i, row, column]) / e_scale}" for row, column in off_diags)
                + "\n"
            )
            fg_imag.write(
                f"{w * e_scale} {np.imag(np.sum(np.diag(gs[i, :, :]))) / e_scale} "
                + f"{np.imag(np.sum(np.diag(gs[i, : n_orb // 2, : n_orb // 2]))) / e_scale} "
                + f"{np.imag(np.sum(np.diag(gs[i, n_orb // 2 :, n_orb // 2 :]))) / e_scale} "
                + " ".join(f"{np.imag(el) / e_scale}" for el in np.diag(gs[i, :, :]))
                + " "
                + " ".join(f"{np.imag(gs[i, row, column]) / e_scale}" for row, column in off_diags)
                + "\n"
            )
