from collections import defaultdict
from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.basis_restrictions import build_excited_restrictions
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.symmetries import widen_weighted_restrictions
from impurityModel.ed.chebyshev_filter import chebyshev_apply, partition_of_unity, spectral_bounds
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed import gf_diagnostics as _gfd

# The module was split for readability (gf_primitives/gf_convergence/gf_shift_recycling hold the
# solver-primitive, convergence-monitor and shift-recycled-resolvent layers respectively); these
# re-exports keep every existing `greens_function.X` / `gf.X` access (tests, spectra.py) working.
from impurityModel.ed.gf_primitives import (
    PairwiseGF,
    _CappedBasisProxy,
    _distributed_seed_qr,
    _sanitize_continued_fraction,
    _trim_blocks,
    build_qr,
    calc_continuants,
    calc_G,
    calc_G_pairwise,
    calc_thermally_averaged_G,
)
from impurityModel.ed.gf_convergence import (
    _GF_MONITOR_POINTS,
    _GF_REL_TOL_FLOOR,
    _gf_eval_meshes,
    _gf_rel_tol,
    _gf_sample_mesh,
    _gf_signed_axes,
    _greens_function_change,
    _lanczos_convergence_summary,
    _make_gf_convergence_monitor,
)
from impurityModel.ed.gf_shift_recycling import KrylovShiftedResolvent, SectorResolventCache
from impurityModel.ed.gf_units import (
    _gf_operator_split,
    enumerate_gf_units,
    run_units_distributed,
    unit_cost_weights,
)
from impurityModel.ed.gf_solvers import (
    block_Green,
    block_Green_bicgstab,
    block_Green_sparse,
)

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
            for block_i in particle_hole_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                res[block_idx] = -np.conj(gf_i)
            for block_i in particle_hole_transposed_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                    res[block_idx] = -np.transpose(np.conj(gf_i), (1, 0))
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                    res[block_idx] = -np.transpose(np.conj(gf_i), (0, 2, 1))
    elif len(block_gf) == len(blocks):
        # block_gf contains all blocks
        for block_i, gf_i in enumerate(block_gf):
            if len(gf_i.shape) == 2:
                block_idx = np.ix_(blocks[block_i], blocks[block_i])
            elif len(gf_i.shape) == 3:
                block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
            res[block_idx] = gf_i
    else:
        raise RuntimeError(f"Block structure does not match block_gf.\n{block_structure=} {len(block_gf)=}")
    return res


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
    reort: Optional,
    dN: Optional[int],
    occ_cutoff: float,
    slaterWeightMin: float,
    sparse: bool,
    num_wanted: int = None,
    gf_method: str = "lanczos",
):
    """
    Calculate interacting Greens function.

    Returns ``(gs_matsubara, gs_realaxis, report)`` on the root rank, where ``report`` is a
    :class:`gf_diagnostics.DiagnosticReport` of per-block convergence/consistency checks
    (``(None, None, None)`` on non-root ranks). ``num_wanted`` is the number of thermal states
    the eigensolver was asked for, used by the ensemble-truncation check.

    ``gf_method`` selects the resolvent kernel: ``"lanczos"`` (default) runs one block-Lanczos
    recurrence per work unit serving the whole mesh; ``"bicgstab"`` solves one linear system
    per frequency point with a rebuilt-and-discarded basis (:func:`block_Green_bicgstab`);
    ``"sliced"`` decomposes ``G`` into Chebyshev spectral-window terms with per-slice bases
    (:func:`_get_greens_function_sliced`; requires a real-axis mesh -- a Matsubara-only call
    falls back to ``bicgstab``, where slicing has nothing to offer). On the non-Lanczos paths
    ``sparse`` is ignored (the solvers work on the ManyBodyState representation only) and the
    operator-split (pairwise) decomposition is never used (the linear solve yields the full
    ``G_ij`` block directly).
    """
    if gf_method not in ("lanczos", "bicgstab", "sliced"):
        raise ValueError(f"Unknown gf_method {gf_method!r}; expected 'lanczos', 'bicgstab' or 'sliced'")
    # Excited-sector restrictions are independent of the orbital block and of the spectral side
    # (the dN occupation window is symmetric and spans all impurity orbitals), so build them once
    # on the full basis instead of per block.
    excited_restrictions, excited_weighted_restrictions = _build_excited_restrictions(
        basis, hOp, psis, es, dN, occ_cutoff
    )
    pairwise = _gf_operator_split() if gf_method == "lanczos" else False
    n_psis = len(psis)

    # Per-state excited windows (see _gf_per_state_restrict). Built on the full basis before the
    # split, from globally-reduced density matrices, so every rank holds the identical list and can
    # look up any unit's window locally. Cheap and identical to the ensemble window when the bath
    # classification is not state-dependent (chain_restrict off, or a directly-hybridizing shell).
    if _gf_per_state_restrict(basis.chain_restrict):
        per_state_restrictions = [
            _build_excited_restrictions(basis, hOp, [psis[ei]], [es[ei]], dN, occ_cutoff)[0] for ei in range(n_psis)
        ]
    else:
        per_state_restrictions = None

    # --- Enumerate the work units = (block, addition/removal, eigenstate-group) -----------
    # One operator group per (block, spectral side); the flat unit decomposition, cost model and
    # single split are the shared engine (enumerate_gf_units / unit_cost_weights /
    # run_units_distributed), load-balanced across the full (block x side x eigenstate)
    # cross-product -- important when there are many small symmetry blocks (the typical
    # production case).
    SIDES = (("c", delta), ("a", -delta))  # 0 = addition (IPS), 1 = removal (PS)
    op_groups = []
    group_meta = []  # (block_i, side_i) per operator group
    for block_i, block in enumerate(blocks):
        for side_i, (op_char, delta_signed) in enumerate(SIDES):
            op_groups.append(([ManyBodyOperator({((orb, op_char),): 1}) for orb in block], delta_signed))
            group_meta.append((block_i, side_i))
    units, unit_seeds, unit_restrictions = enumerate_gf_units(
        op_groups,
        psis,
        [excited_restrictions] * len(op_groups),
        excited_weighted_restrictions,
        slaterWeightMin,
        per_state_restrictions,
        pairwise=pairwise,
    )
    unit_weights = unit_cost_weights(unit_seeds, basis.comm)

    if gf_method in ("bicgstab", "sliced"):
        driver = (
            _get_greens_function_sliced
            if gf_method == "sliced" and omega_mesh is not None
            else _get_greens_function_bicgstab
        )
        return driver(
            matsubara_mesh,
            omega_mesh,
            es,
            tau,
            basis,
            hOp,
            delta,
            blocks,
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
        )

    def kernel(split_basis, u, seeds):
        unit = units[u]
        # Converge G where this unit's G will actually be evaluated: the caller's meshes, shifted
        # by each thermal energy the unit stacks and signed by its spectral side. Without this the
        # monitor resolves the real-axis resolvent at broadening `delta` even for a Matsubara-only
        # self-energy, which costs 3.6-4.1x the blocks such a run needs.
        eval_meshes = _gf_eval_meshes(
            matsubara_mesh,
            omega_mesh,
            group_meta[unit.group_i][1],
            delta,
            [es[ei] for ei in unit.chunk],
        )
        alphas, betas, r, cap_stats = _block_green_group(
            split_basis,
            hOp,
            seeds,
            reort,
            unit.delta,
            slaterWeightMin,
            sparse,
            verbose_extra,
            unit_restrictions[u],
            excited_weighted_restrictions,
            eval_meshes=eval_meshes,
        )
        if verbose_extra:
            print(f"Expanded excited state basis contains {cap_stats["retained_size"]} elements.")
        return (
            alphas,
            betas,
            [r[:, p * unit.n_ops : (p + 1) * unit.n_ops] for p in range(len(unit.chunk))],
            cap_stats,
        )

    results = run_units_distributed(basis, unit_seeds, unit_weights, kernel, verbose=verbose, reort=reort)

    gs_matsubara = gs_realaxis = report = None
    if results is not None:
        # Reassemble the per-unit results (global unit order) into per-(block, side)
        # eigenstate-indexed coefficient lists, then build each block's Green's function exactly
        # as before. acc[(block_i, side_i)] = (alphas_list, betas_list, r_list) indexed by
        # eigenstate. In grouped mode r_list[ei] is the seed-projection matrix; in pairwise mode
        # it is a PairwiseGF assembled from the eigenstate's scalar continued fractions
        # (a_list/b_list stay None -- each PairwiseGF carries its own scalar coefficients).
        acc = {
            (bi, si): ([None] * n_psis, [None] * n_psis, [None] * n_psis) for bi in range(len(blocks)) for si in (0, 1)
        }
        # Worst-case cap state per block over all its (side, eigenstate) solves: any
        # frozen solve marks the block; retained_size is the smallest frozen size.
        cap_acc = {}
        for unit, (_alphas, _betas, _r_slices, cap_stats) in zip(units, results):
            block_i, _ = group_meta[unit.group_i]
            stats = cap_acc.setdefault(block_i, {"cap_hit": False, "retained_size": None, "cap": cap_stats["cap"]})
            if cap_stats["cap_hit"]:
                stats["cap_hit"] = True
                stats["cap"] = cap_stats["cap"]
                retained = cap_stats.get("retained_size")
                if retained is not None and (stats["retained_size"] is None or retained < stats["retained_size"]):
                    stats["retained_size"] = retained
        if pairwise:
            # pw_cf[(block_i, side_i, ei)] = {"diag": {i: cf}, "sum": {(i,j): cf}, "imag": {(i,j): cf}}
            pw_cf = defaultdict(lambda: {"diag": {}, "sum": {}, "imag": {}})
            for unit, (alphas, betas, r_slices, _cap_stats) in zip(units, results):
                block_i, side_i = group_meta[unit.group_i]
                cf = (alphas, betas, r_slices[0])
                role, a, b = unit.pw_tag
                key = a if role == "diag" else (a, b)
                pw_cf[(block_i, side_i, unit.chunk[0])][role][key] = cf
            for (block_i, side_i, ei), roles in pw_cf.items():
                n = len(blocks[block_i])
                diag = [roles["diag"][i] for i in range(n)]
                pairs = {ij: (roles["sum"][ij], roles["imag"][ij]) for ij in roles["sum"]}
                acc[(block_i, side_i)][2][ei] = PairwiseGF(n, diag, pairs)
        else:
            for unit, (alphas, betas, r_slices, _cap_stats) in zip(units, results):
                block_i, side_i = group_meta[unit.group_i]
                a_list, b_list, r_list = acc[(block_i, side_i)]
                for p, ei in enumerate(unit.chunk):
                    a_list[ei], b_list[ei], r_list[ei] = alphas, betas, r_slices[p]

        e0 = np.min(es)
        Z = np.sum(np.exp(-(es - e0) / tau))
        gs_matsubara = (
            [np.empty((len(matsubara_mesh), len(b), len(b)), dtype=complex) for b in blocks]
            if matsubara_mesh is not None
            else None
        )
        gs_realaxis = (
            [np.empty((len(omega_mesh), len(b), len(b)), dtype=complex) for b in blocks]
            if omega_mesh is not None
            else None
        )
        report = _gfd.DiagnosticReport()
        for block_i, block in enumerate(blocks):
            a_add, b_add, r_add = acc[(block_i, 0)]
            a_rem, b_rem, r_rem = acc[(block_i, 1)]
            if matsubara_mesh is not None:
                G_IPS = calc_thermally_averaged_G(a_add, b_add, r_add, matsubara_mesh, es, e0, tau, 0)
                G_PS = calc_thermally_averaged_G(a_rem, b_rem, r_rem, -matsubara_mesh, es, e0, tau, 0)
                gs_matsubara[block_i][:] = (G_IPS - np.transpose(G_PS, (0, 2, 1))) / Z
            G_IPS_real = G_PS_real = combined_real = None
            if omega_mesh is not None:
                G_IPS_real = calc_thermally_averaged_G(a_add, b_add, r_add, omega_mesh, es, e0, tau, delta)
                G_PS_real = calc_thermally_averaged_G(a_rem, b_rem, r_rem, -omega_mesh, es, e0, tau, -delta)
                combined_real = (G_IPS_real - np.transpose(G_PS_real, (0, 2, 1))) / Z
                gs_realaxis[block_i][:] = combined_real

            # --- per-block convergence / consistency diagnostics ---------------------------
            # The pairwise path stores per-eigenstate PairwiseGF objects rather than the seed
            # projection matrices and scalar-tridiagonal coefficients the r-/(alphas,betas)-based
            # checks consume, so in that mode only the G-derived checks (thermal cutoff, mesh
            # density, causality) apply.
            diags = [_gfd.check_thermal_weight_cutoff(es, e0, tau, n_returned=len(es), num_wanted=num_wanted)]
            block_cap = cap_acc.get(block_i)
            if block_cap is not None:
                diags.append(
                    _gfd.check_basis_truncation(block_cap["cap_hit"], block_cap["retained_size"], block_cap["cap"])
                )
            if not pairwise:
                diags.insert(0, _gfd.check_spectral_sum_rule(r_add, r_rem, es, e0, tau, len(block)))
                lanczos_tol = _gf_rel_tol(slaterWeightMin)
                conv_add = _lanczos_convergence_summary(a_add, b_add, delta, tol=lanczos_tol)
                conv_rem = _lanczos_convergence_summary(a_rem, b_rem, delta, tol=lanczos_tol)
                n_blocks = max(conv_add[2], conv_rem[2])
                diags.append(
                    _gfd.check_lanczos_convergence(
                        conv_add[0] and conv_rem[0], max(conv_add[1], conv_rem[1]), n_blocks, n_blocks
                    )
                )
            if G_IPS_real is not None:
                diags.append(_gfd.check_mesh_density(omega_mesh, delta))
                if not pairwise:
                    diags.append(
                        _gfd.check_integrated_weight(G_IPS_real, r_add, es, e0, tau, omega_mesh, "add", delta=delta)
                    )
                    diags.append(
                        _gfd.check_integrated_weight(G_PS_real, r_rem, es, e0, tau, -omega_mesh, "rem", delta=delta)
                    )
                diags.append(_gfd.check_causality(combined_real, "G"))
            report.extend(str(block), diags)

    return (gs_matsubara, gs_realaxis, report)


def _get_greens_function_bicgstab(
    matsubara_mesh,
    omega_mesh,
    es,
    tau,
    basis,
    hOp,
    delta,
    blocks,
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
):
    r"""Distribution + assembly of the per-frequency BiCGSTAB Green's function.

    The unit decomposition (and the excited windows) are exactly the Lanczos driver's --
    :func:`get_Greens_function` hands them over after :func:`enumerate_gf_units` -- only the
    per-unit kernel and the result contract differ: each unit returns ``G`` already evaluated
    on the caller's meshes (:func:`block_Green_bicgstab`); the shared assembler
    :func:`_run_evaluated_gf_units` does the rest.
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
        units_meta,
        unit_seeds,
        unit_weights,
        kernel,
        verbose,
        num_wanted,
    )


def _run_evaluated_gf_units(
    matsubara_mesh,
    omega_mesh,
    es,
    tau,
    basis,
    delta,
    blocks,
    units_meta,
    unit_seeds,
    unit_weights,
    kernel,
    verbose,
    num_wanted,
    extra_diags=None,
    gf_method="bicgstab",
):
    r"""Distribute, accumulate and assemble Green's-function units that return evaluated ``G``.

    The shared engine behind the ``bicgstab`` and ``sliced`` drivers: ``kernel(split_basis,
    u, seeds)`` must return ``(G_axes, stats)`` in :func:`block_Green_bicgstab`'s contract,
    and ``units_meta[u] = (block_i, side_i, chunk)`` names where unit ``u``'s result belongs.
    The assembly is a streaming Boltzmann-weighted accumulation into per-``(block, side)``
    arrays (rank 0 never holds more than one color's payload) followed by the same
    :math:`(G_\mathrm{IPS} - G_\mathrm{PS}^T)/Z` combination the Lanczos path applies to its
    evaluated continued fractions. Because the accumulation is a plain sum, several units may
    target the same ``(block, side, eigenstate)`` -- the sliced driver's window terms sum to
    the full ``G`` exactly this way.

    The diagnostics report keeps the representation-independent checks (thermal cutoff, mesh
    density, causality, basis truncation) plus the solver-residual record
    (:func:`gf_diagnostics.check_bicgstab_convergence`); ``extra_diags(block_i)``, when given,
    appends caller-specific checks (e.g. the slice-partition record). The spectral sum rule
    and integrated-weight checks are expressed in seed-projection/continued-fraction terms
    these paths do not produce.
    """
    e0 = np.min(es)
    boltzmann = np.exp(-(np.asarray(es) - e0) / tau)
    Z = float(np.sum(boltzmann))
    axis_lens = [len(m) for m in (matsubara_mesh, omega_mesh) if m is not None]

    # Streaming accumulators, populated on global rank 0 only (reduce_fn's contract).
    is_root = basis.comm is None or basis.comm.rank == 0
    G_acc = (
        {
            (bi, si): [np.zeros((L, len(blocks[bi]), len(blocks[bi])), dtype=complex) for L in axis_lens]
            for bi in range(len(blocks))
            for si in (0, 1)
        }
        if is_root
        else None
    )
    stats_acc = {} if is_root else None

    def reduce_fn(u, result):
        G_axes, stats = result
        block_i, side_i, chunk = units_meta[u]
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
            },
        )
        for key in ("n_points", "n_unconverged", "iterations", "gmres_points", "gmres_iterations"):
            agg[key] += stats[key]
        for key in ("max_rel_residual", "max_solve_basis", "max_rebuild_basis"):
            agg[key] = max(agg[key], stats[key])
        agg["cap_hit"] = agg["cap_hit"] or stats["cap_hit"]
        agg["seed_overflow"] = agg["seed_overflow"] or stats["seed_overflow"]
        if stats["retained_size"] is not None:
            agg["retained_size"] = (
                stats["retained_size"]
                if agg["retained_size"] is None
                else min(agg["retained_size"], stats["retained_size"])
            )

    got = run_units_distributed(
        basis, unit_seeds, unit_weights, kernel, verbose=verbose, reduce_fn=reduce_fn, gf_method=gf_method
    )
    if got is None:
        return None, None, None

    gs_matsubara = (
        [np.empty((len(matsubara_mesh), len(b), len(b)), dtype=complex) for b in blocks]
        if matsubara_mesh is not None
        else None
    )
    gs_realaxis = (
        [np.empty((len(omega_mesh), len(b), len(b)), dtype=complex) for b in blocks] if omega_mesh is not None else None
    )
    report = _gfd.DiagnosticReport()
    for block_i, block in enumerate(blocks):
        ax = 0
        combined_real = None
        if matsubara_mesh is not None:
            G_IPS = G_acc[(block_i, 0)][ax]
            G_PS = G_acc[(block_i, 1)][ax]
            gs_matsubara[block_i][:] = (G_IPS - np.transpose(G_PS, (0, 2, 1))) / Z
            ax += 1
        if omega_mesh is not None:
            G_IPS_real = G_acc[(block_i, 0)][ax]
            G_PS_real = G_acc[(block_i, 1)][ax]
            combined_real = (G_IPS_real - np.transpose(G_PS_real, (0, 2, 1))) / Z
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
        if combined_real is not None:
            diags.append(_gfd.check_mesh_density(omega_mesh, delta))
            diags.append(_gfd.check_causality(combined_real, "G"))
        if extra_diags is not None:
            diags.extend(extra_diags(block_i))
        report.extend(str(block), diags)

    return gs_matsubara, gs_realaxis, report


def _get_greens_function_sliced(
    matsubara_mesh,
    omega_mesh,
    es,
    tau,
    basis,
    hOp,
    delta,
    blocks,
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
):
    r"""The spectrum-slicing Green's function: filtered work units through the shared engine.

    Implements the partition-of-unity identity (``doc/plans/spectrum_slicing.md``,
    theory in ``doc/greens_function_theory.md`` section 5)

    .. math:: G_{ij}(z) = \sum_s \langle v_i | (z - H)^{-1} \, p_s(H) v_j \rangle :

    every base unit fans out into one engine unit per Chebyshev window, whose seeds are the
    *filtered* kets ``p_s(H) v`` and whose bra block is the unfiltered ``v`` (the
    ``bra_seeds`` cross-element mode of :func:`block_Green_bicgstab`). The windows tile the
    spectral interval and telescope to 1 identically, so the streaming sum of the slice
    terms in :func:`_run_evaluated_gf_units` reconstructs the exact ``G`` -- the only
    approximations are the per-solve ``atol`` and the optional slice-seed truncation
    ``GF_SLICE_TOL`` (the Phase-0-calibrated memory knob: filtered seeds' dominant
    amplitudes are energy-local, their sub-1e-6 tails are not).

    Filtering runs *before* the split, unit by unit, collectively on the full communicator
    (one Chebyshev recurrence per unit serves all its windows); the spectral bounds are
    estimated once and shared. Knobs: ``GF_SLICES`` (windows across the evaluation band),
    ``GF_SLICE_DEGREE`` (0 = auto from bandwidth/slice width), ``GF_SLICE_TOL``.
    """
    cap = getattr(basis, "truncation_threshold", np.inf)

    def _excited_clone(u):
        return basis.clone(
            initial_basis=sorted({state for s in unit_seeds[u] for state in s.keys()}),
            restrictions=unit_restrictions[u],
            weighted_restrictions=excited_weighted_restrictions,
            verbose=False,
        )

    def _capped(b):
        return _CappedBasisProxy(b, cap) if np.isfinite(cap) else b

    w_lo, w_hi = float(np.min(omega_mesh)), float(np.max(omega_mesh))
    n_slices, degree_knob, slice_tol = _slice_count(), _slice_degree(), _slice_tol()
    sliced_meta = []  # (block_i, side_i, chunk, n_ops, unit_restrictions index)
    sliced_seeds = []  # filtered kets + unfiltered bras, flat per engine unit
    n_windows = degree_used = edge_width = None
    for u, unit in enumerate(units):
        block_i, side_i = group_meta[unit.group_i]
        sign = 1.0 if side_i == 0 else -1.0
        chunk_es = [es[ei] for ei in unit.chunk]
        # Spectral bounds PER UNIT: each unit's excited sector has its own reachable
        # spectrum, and a Chebyshev polynomial evaluated even slightly outside its
        # interval grows as cosh(n*arccosh|x|) -- a 1% bounds violation at degree ~10^3
        # is a ~1e100 blowup (measured; the norm guard below turns any recurrence of it
        # into a hard error instead of silent garbage). The bounds Lanczos and the filter
        # share one capped clone.
        unit_clone = _excited_clone(u)
        # The seeds were built by applying c/c^dagger rank-locally, so each amplitude sits on
        # the rank that *generated* it, not on the rank that *owns* that determinant (owner =
        # routing_hash % size). The three-term recurrence redistributes H*t but not t, so a
        # misplaced row leaves H*t on the owner and t on the generator: the recurrence
        # decouples across ranks and diverges. Every other solver reaches its basis through
        # the same redistribute -- the filter stage is just the one that runs before it.
        seeds_u = unit_clone.redistribute_psis(list(unit_seeds[u]))
        unit_basis = _capped(unit_clone)
        bounds = spectral_bounds(hOp, unit_basis)
        ends = [e + sign * w for e in chunk_es for w in (w_lo, w_hi)]
        band_lo = max(bounds[0], min(ends))
        band_hi = min(bounds[1], max(ends))
        slice_width = max((band_hi - band_lo) / n_slices, 1e-12)
        degree = degree_knob or int(np.clip(8.0 * (bounds[1] - bounds[0]) / slice_width, 200, 4000))
        coeff_sets, _windows, edge_width = partition_of_unity(
            bounds, np.linspace(band_lo, band_hi, n_slices + 1), degree
        )
        if verbose and (basis.comm is None or basis.comm.rank == 0):
            print(
                f"Spectrum slicing unit {u}: bounds [{bounds[0]:.3f}, {bounds[1]:.3f}], "
                f"{len(coeff_sets)} windows, degree {degree}, slice tol {slice_tol:g}.",
                flush=True,
            )
        filtered = chebyshev_apply(hOp, unit_basis, list(seeds_u), coeff_sets, slaterWeightMin, bounds)
        seed_norm2 = sum(s.norm2() for s in seeds_u)
        filt_norm2 = max(sum(k.norm2() for k in kets) for kets in filtered)
        if basis.comm is not None:
            seed_norm2 = basis.comm.allreduce(seed_norm2, op=MPI.SUM)
            filt_norm2 = basis.comm.allreduce(filt_norm2, op=MPI.SUM)
        if filt_norm2 > 4.0 * max(seed_norm2, 1e-300):
            # |p_s| <= ~1.1 on the interval (Jackson-damped windows), so a filtered norm
            # beyond ~2x the seed norm means the recurrence left the spectral interval.
            raise RuntimeError(
                f"Chebyshev filter diverged on GF unit {u} (filtered norm^2 {filt_norm2:.3e} vs "
                f"seed norm^2 {seed_norm2:.3e}): spectral bounds [{bounds[0]:.4f}, {bounds[1]:.4f}] "
                "do not contain this unit's reachable spectrum. Increase the bounds padding "
                "(spectral_bounds pad_rel) or its Lanczos depth."
            )
        for kets in filtered:
            if slice_tol > 0:
                for ket in kets:
                    ket.prune(slice_tol)
            sliced_meta.append((block_i, side_i, unit.chunk, unit.n_ops, u))
            sliced_seeds.append(list(kets) + list(seeds_u))
        n_windows, degree_used = len(coeff_sets), degree

    sliced_weights = unit_cost_weights(sliced_seeds, basis.comm)

    def kernel(split_basis, su, seeds):
        _block_i, side_i, chunk, n_ops, u = sliced_meta[su]
        n_cols = len(chunk) * n_ops
        z_axes = _gf_signed_axes(matsubara_mesh, omega_mesh, side_i, delta)
        return block_Green_bicgstab(
            hOp,
            list(seeds[:n_cols]),
            split_basis,
            [es[ei] for ei in chunk],
            n_ops,
            z_axes,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose_extra,
            excited_restrictions=unit_restrictions[u],
            excited_weighted_restrictions=excited_weighted_restrictions,
            bra_seeds=list(seeds[n_cols:]),
        )

    def extra_diags(_block_i):
        return [_gfd.check_slice_partition(n_windows, degree_used, edge_width, slice_tol)]

    units_meta = [(m[0], m[1], m[2]) for m in sliced_meta]
    return _run_evaluated_gf_units(
        matsubara_mesh,
        omega_mesh,
        es,
        tau,
        basis,
        delta,
        blocks,
        units_meta,
        sliced_seeds,
        sliced_weights,
        kernel,
        verbose,
        num_wanted,
        extra_diags=extra_diags,
        gf_method="sliced",
    )


def _build_excited_restrictions(basis, hOp, psis, es, dN, occ_cutoff, dN_imp=None, dN_val=None, dN_con=None):
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
        basis, hOp, psis, es, imp_change=dN_imp, val_change=dN_val, con_change=dN_con, cutoff=occ_cutoff
    )
    # Weighted (e.g. S_z) restriction for the excited sector: widen the ground-state bounds by one
    # orbital weight so the addition / removal sectors q_psi ± w_j are admitted while still
    # confining the basis.
    excited_weighted_restrictions = widen_weighted_restrictions(basis.weighted_restrictions)
    return excited_restrictions, excited_weighted_restrictions


def _intersect_restrictions(base, extra):
    """Conjunctively merge two ``{frozenset: (min, max)}`` restriction dicts.

    Shared keys are intersected (``max`` of the mins, ``min`` of the maxs); keys unique to
    either side are kept. Every ``Basis`` restriction entry is enforced, so the result confines
    a determinant iff it satisfies *both* inputs. ``base`` is treated as empty when ``None``.
    """
    if not base:
        return dict(extra)
    merged = dict(base)
    for key, (lo, hi) in extra.items():
        if key in merged:
            blo, bhi = merged[key]
            merged[key] = (max(blo, lo), min(bhi, hi))
        else:
            merged[key] = (lo, hi)
    return merged


def _block_green_group(
    split_basis,
    hOp,
    group_seed_states,
    reort,
    delta,
    slaterWeightMin,
    sparse,
    verbose,
    excited_restrictions,
    excited_weighted_restrictions,
    eval_meshes=None,
):
    """Run one (possibly wide) block-Lanczos Green's function for a group of stacked seeds.

    ``group_seed_states`` is the flat list of seed columns for an eigenstate group (length
    ``len(group) * n_ops`` in ``(eigenstate, operator)`` order). Builds the excited basis from
    their union, points ``hOp`` at its restrictions, and runs the sparse or dense block-Green
    kernel. Returns ``(alphas, betas, r, n_basis, cap_stats)``; the caller slices ``r``'s columns
    per eigenstate (``r[:, p*n_ops:(p+1)*n_ops]``) since ``(alphas, betas)`` are shared by the
    group. ``cap_stats`` is ``{"cap_hit", "retained_size", "cap"}`` describing whether this
    solve froze at ``truncation_threshold`` (feeds the basis_cap diagnostic).

    ``eval_meshes`` (:func:`_gf_eval_meshes`) tells the convergence monitor which frequencies this
    unit's ``G`` will be evaluated on. Passing ``None`` -- the default, and what the spectra/RIXS
    callers do -- leaves it converging the real-axis resolvent over the resolved Ritz band.
    """
    excited_basis = split_basis.clone(
        initial_basis=set(state for p in group_seed_states for state in p),
        restrictions=excited_restrictions,
        weighted_restrictions=excited_weighted_restrictions,
        verbose=False,
    )
    if excited_basis.restrictions is not None:
        hOp.set_restrictions(excited_basis.restrictions)
    if excited_basis.weighted_restrictions is not None:
        hOp.set_weighted_restrictions(excited_basis.weighted_restrictions)
    cap = getattr(excited_basis, "truncation_threshold", np.inf)
    if sparse:
        cap_info = {}
        alphas, betas, r = block_Green_sparse(
            reort=reort,
            hOp=hOp,
            psi_arr=excited_basis.redistribute_psis(group_seed_states),
            basis=excited_basis,
            delta=delta,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose,
            cap_info=cap_info,
            eval_meshes=eval_meshes,
        )
        cap_stats = {
            "cap_hit": bool(cap_info.get("cap_hit", False)),
            "retained_size": cap_info.get("retained_size"),
            "cap": cap,
        }
    else:
        alphas, betas, r = block_Green(
            reort=reort,
            hOp=hOp,
            psi_arr=excited_basis.redistribute_psis(group_seed_states),
            basis=excited_basis,
            delta=delta,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose,
            eval_meshes=eval_meshes,
        )
        # The array path stops expanding when the basis crosses the cap (never removes).
        cap_stats = {
            "cap_hit": bool(np.isfinite(cap) and excited_basis.size > cap),
            "retained_size": len(excited_basis),
            "cap": cap,
        }
    return alphas, betas, r, cap_stats


def calc_Greens_function_with_offdiag(
    hOp,
    tOps,
    psis,
    es,
    block_basis,
    delta,
    reort: Optional = None,
    dN: Optional[int] = None,
    occ_cutoff: float = 1e-6,
    slaterWeightMin: float = 0,
    verbose: bool = True,
    sparse: bool = False,
    dN_imp=None,
    dN_val=None,
    dN_con=None,
    extra_restrictions=None,
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
        block_basis, hOp, psis, es, dN, occ_cutoff, dN_imp=dN_imp, dN_val=dN_val, dN_con=dN_con
    )
    # Optional conserved-charge sector confinement (symmetries.transition_sector_restrictions):
    # pins the seed's charge sector on top of the per-shell occupation window, pruning
    # sector-violating determinants the window alone would admit. Intersected key-by-key so it
    # can only tighten the excited basis, never loosen it.
    if extra_restrictions:
        excited_restrictions = _intersect_restrictions(excited_restrictions, extra_restrictions)
    if verbose and excited_restrictions is not None:
        print("Excited state restrictions:")
        for indices, occupations in excited_restrictions.items():
            print(f"---> {sorted(indices)} : {occupations}")

    # One operator group holding the whole tOps block; pairwise=False because this function's
    # return contract (per-eigenstate r matrices) cannot represent scalar pairwise fractions.
    units, unit_seeds, unit_restrictions = enumerate_gf_units(
        [(tOps, delta)],
        psis,
        [excited_restrictions],
        excited_weighted_restrictions,
        slaterWeightMin,
        pairwise=False,
    )
    unit_weights = unit_cost_weights(unit_seeds, block_basis.comm)

    def kernel(split_basis, u, seeds):
        unit = units[u]
        alphas, betas, r, _cap_stats = _block_green_group(
            split_basis,
            hOp,
            seeds,
            reort,
            unit.delta,
            slaterWeightMin,
            sparse,
            verbose,
            unit_restrictions[u],
            excited_weighted_restrictions,
        )
        if verbose:
            print(f"Expanded excited state basis contains {_cap_stats["retained_size"]} elements.")
        return alphas, betas, [r[:, p * unit.n_ops : (p + 1) * unit.n_ops] for p in range(len(unit.chunk))]

    results = run_units_distributed(block_basis, unit_seeds, unit_weights, kernel, verbose=verbose, reort=reort)

    excited_alphas = excited_betas = excited_r = None
    if results is not None:
        excited_alphas = [None for _ in psis]
        excited_betas = [None for _ in psis]
        excited_r = [None for _ in psis]
        for unit, (alphas, betas, r_slices) in zip(units, results):
            for p, ei in enumerate(unit.chunk):
                excited_alphas[ei] = alphas
                excited_betas[ei] = betas
                excited_r[ei] = r_slices[p]
        assert not any(alpha is None for alpha in excited_alphas), f"{excited_alphas=}"
        assert not any(beta is None for beta in excited_betas), f"{excited_betas=}"
        assert not any(r is None for r in excited_r), f"{excited_r=}"

    return excited_alphas, excited_betas, excited_r


# --- Spectrum slicing (gf_method="sliced") --------------------------------------------------
# The Phase-0 calibration (doc/plans/spectrum_slicing.md): filtered seeds' dominant amplitudes
# are energy-local, their sub-1e-6 tails are not -- so the memory lever is GF_SLICE_TOL (extra
# amplitude truncation of each filtered seed), traded explicitly against accuracy (discarded
# tail ~ sqrt(n_tail) * tol) and reported by the diagnostics.


def _slice_count():
    """Chebyshev windows tiling the real-axis evaluation band (:data:`config.GF_SLICES`)."""
    return config.GF_SLICES.get()


def _slice_degree():
    """Filter degree (:data:`config.GF_SLICE_DEGREE`); 0 = auto (bandwidth / slice-width)."""
    return config.GF_SLICE_DEGREE.get()


def _slice_tol():
    """Amplitude truncation of the filtered slice seeds (:data:`config.GF_SLICE_TOL`)."""
    return config.GF_SLICE_TOL.get()


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
    (:func:`_union_restrictions`) so the shared block Krylov space still contains every seed's
    dynamics; in operator-split mode every unit is a single state, giving the full per-state
    tightening. The seed ``c_i|psi_e>`` is unchanged (an impurity operator preserves bath
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


def rotate_Greens_function(G, T):
    r"""
    Rotate the Greens function, G, using the matrix T.
    Returns G'(\omega) = T^\dagger G(\omega) T
    Parameters
    ==========
    G : NDArray - Greens function to rotate
    T : NDArray - Rotation matrix to use
    Returns
    =======
    G' : NDArray - The rotated Greens function
    """
    return np.conj(T.T)[np.newaxis, :, :] @ G @ T[np.newaxis, :, :]


def rotate_4index_U(U4, T):
    r"""
    Rotate the four index tensor, U4, using the matrix T.
    Returns U4' = T^\daggerT^\dagger U4 TT
    Parameters
    ==========
    U4 : NDArray - Tensor function to rotate
    T : NDArray - Rotation matrix to use
    Returns
    =======
    U4' : NDArray - The rotated tensor function
    """
    return np.einsum("ij,kl, jlmo, mn, op", np.conj(T.T), np.conj(T.T), U4, T, T)


def save_Greens_function(gs, omega_mesh, label, cluster_label, e_scale=1, tol=1e-8):
    """
    Save Greens function to file, using RSPt .dat format. Including offdiagonal elements.
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
    with (
        open(f"real-{label}{axis_label}-{cluster_label}.dat", "w") as fg_real,
        open(f"imag-{label}{axis_label}-{cluster_label}.dat", "w") as fg_imag,
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
