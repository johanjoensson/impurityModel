from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.basis_restrictions import build_excited_restrictions
from impurityModel.ed.basis_split import split_basis_and_redistribute_psi
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.BlockLanczosArray import (
    Reort,
    block_lanczos_array,
    resolve_reort,
)
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.symmetries import widen_weighted_restrictions
from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.chebyshev_filter import chebyshev_apply, partition_of_unity, spectral_bounds
from impurityModel.ed.gmres import block_gmres
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyBlockState,
    ManyBodyOperator,
    ManyBodyState,
    block_inner_cy,
)
from impurityModel.ed.memory_estimate import (
    estimate_gf_peak_bytes,
    format_bytes,
    max_colors_within_budget,
)
from impurityModel.ed.mpi_comm import gather_distributed_results
from impurityModel.ed import gf_diagnostics as _gfd
from impurityModel.ed.basis_transcription import build_dense_matrix, build_sparse_matrix, build_state, build_vector
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


@dataclass(frozen=True)
class GFUnit:
    """One distributable Green's-function work unit: a (possibly wide) block-Lanczos recurrence.

    A unit stacks the transition-operator seeds of ``chunk`` thermal eigenstates from one
    operator group into a single recurrence of width ``len(chunk) * n_ops`` (or a single scalar
    seed in operator-split mode, identified by ``pw_tag``). Units are the atoms of the MPI
    distribution: :func:`run_units_distributed` never splits one across colors.

    Attributes
    ----------
    group_i : int
        Index into the caller's operator-group list (e.g. block x spectral side, or a
        transition-operator index).
    chunk : tuple of int
        Thermal-eigenstate indices whose seeds this unit stacks.
    n_ops : int
        Seed columns per eigenstate (1 in operator-split mode).
    delta : float
        Signed broadening of this unit's recurrence (sign selects addition/removal).
    pw_tag : tuple, optional
        ``("diag"|"sum"|"imag", i, j)`` identifying the scalar seed in operator-split mode;
        ``None`` for grouped (wide-block) units.
    """

    group_i: int
    chunk: tuple[int, ...]
    n_ops: int
    delta: float
    pw_tag: Optional[tuple] = None


def unit_cost_weights(unit_seeds: list[list[ManyBodyState]], comm) -> np.ndarray:
    """Predicted block-Lanczos cost per work unit -- the single source of truth for split weights.

    The per-step cost is dominated by two terms that are both known at split time: the matvec
    (~ excited-basis size x block width) and the block reorthogonalization (~ width^2, firing on
    nearly every step on a near-degenerate spectrum). The total seed mass (sum of per-column nnz)
    is the cheapest per-unit correlate of the reachable excited-sector size, so

        weight = seed_mass * width + 1.0

    (the +1 floor keeps an all-empty seed set from zeroing the weight norm). Because seed mass
    already scales ~linearly with the column count this is ~ per-column mass * width^2, matching
    matvec + reort. This replaced the old ``log10(len)+1`` compression, which crushed 10-100x true
    cost spreads into a <2.3x band -- nearly equalizing units and burying the exactly-known block
    width -- so the widest block became the straggler color. Only the seed mass varies per rank
    and needs the Allreduce; the width is a structural, per-rank-identical count.
    """
    lengths = np.array([sum(len(s) for s in seeds) for seeds in unit_seeds], dtype=float)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, lengths, op=MPI.SUM)
    widths = np.array([len(seeds) for seeds in unit_seeds], dtype=float)
    return lengths * widths + 1.0


def enumerate_gf_units(
    op_groups: list[tuple[list[ManyBodyOperator], float]],
    psis: list[ManyBodyState],
    group_restrictions: list,
    weighted_restrictions,
    slaterWeightMin: float,
    per_state_restrictions: Optional[list] = None,
    pairwise: Optional[bool] = None,
) -> tuple[list[GFUnit], list[list[ManyBodyState]], list]:
    """Enumerate the flat work units of a Green's-function calculation.

    Applies each operator group's transition operators to every thermal state (collective on
    the full basis), then chunks the eigenstates into groups of ``GF_EIGENSTATE_GROUP`` -- each
    chunk seeds one (possibly wide) block-Lanczos recurrence and is one work unit. This is the
    single global decomposition that is load-balanced across the full
    (operator group x eigenstate) cross-product -- important when there are many small symmetry
    blocks (the typical production case).

    In operator-split mode (``GF_OPERATOR_SPLIT``) every unit is a width-1 scalar recurrence:
    one per diagonal seed ``v_i``, plus per off-diagonal pair (i<j) the two polarization seeds
    ``v_i + v_j`` and ``v_i + i v_j``. This is the narrow end of the granularity spectrum:
    maximal communication-free units, no shared Krylov space.

    Parameters
    ----------
    op_groups : list of (list of ManyBodyOperator, float)
        One ``(tOps, signed_delta)`` entry per operator group (e.g. per block x spectral side,
        or per transition operator).
    psis : list of ManyBodyState
        Thermal eigenstates.
    group_restrictions : list
        Excited-sector restriction dict per operator group, used both when applying the
        group's operators and as the unit fallback window.
    weighted_restrictions
        Weighted (e.g. S_z) excited-sector restrictions, shared by all groups.
    slaterWeightMin : float
        Determinant-weight cutoff for the seed application.
    per_state_restrictions : list, optional
        Per-eigenstate excited windows; when given, each unit's window is the union
        (:func:`_union_restrictions`) over the eigenstates it stacks instead of the group
        fallback.
    pairwise : bool, optional
        Override the ``GF_OPERATOR_SPLIT`` environment default. Callers whose result
        contract cannot represent scalar pairwise fractions (per-eigenstate ``r``
        matrices) pass ``False``.

    Returns
    -------
    tuple
        ``(units, unit_seeds, unit_restrictions)`` -- the :class:`GFUnit` metadata, the flat
        seed-column list per unit in (eigenstate, operator) order, and the excited window per
        unit.
    """
    if pairwise is None:
        pairwise = _gf_operator_split()
    group = 1 if pairwise else _gf_eigenstate_group()
    n_psis = len(psis)
    units: list[GFUnit] = []
    unit_seeds: list[list[ManyBodyState]] = []
    for g, (tOps, delta_signed) in enumerate(op_groups):
        block_v = _apply_transition_ops(tOps, psis, group_restrictions[g], weighted_restrictions, slaterWeightMin)
        n_ops = len(tOps)
        if pairwise:
            for ei in range(n_psis):
                for i in range(n_ops):
                    units.append(GFUnit(g, (ei,), 1, delta_signed, ("diag", i, i)))
                    unit_seeds.append([block_v[ei][i]])
                for i in range(n_ops):
                    for j in range(i + 1, n_ops):
                        units.append(GFUnit(g, (ei,), 1, delta_signed, ("sum", i, j)))
                        unit_seeds.append([block_v[ei][i] + block_v[ei][j]])
                        units.append(GFUnit(g, (ei,), 1, delta_signed, ("imag", i, j)))
                        unit_seeds.append([block_v[ei][i] + 1j * block_v[ei][j]])
        else:
            for chunk_start in range(0, n_psis, group):
                chunk = tuple(range(chunk_start, min(chunk_start + group, n_psis)))
                units.append(GFUnit(g, chunk, n_ops, delta_signed, None))
                unit_seeds.append([block_v[j][i] for j in chunk for i in range(n_ops)])

    # Per-unit excited window: the union of the per-state windows over the eigenstates the unit
    # stacks (exactly that state's window for a single-state / operator-split unit). Falls back
    # to the group window when per-state restrictions are disabled or state-independent.
    if per_state_restrictions is not None:
        unit_restrictions = [_union_restrictions([per_state_restrictions[ei] for ei in u.chunk]) for u in units]
    else:
        unit_restrictions = [group_restrictions[u.group_i] for u in units]
    return units, unit_seeds, unit_restrictions


def run_units_distributed(
    basis: Basis,
    unit_seeds: list[list[ManyBodyState]],
    unit_weights: np.ndarray,
    kernel: Callable,
    verbose: bool = False,
    reduce_fn: Optional[Callable] = None,
    reort=None,
    gf_method: str = "lanczos",
) -> Optional[list]:
    """Distribute work units over MPI colors, run ``kernel`` per unit, gather to global rank 0.

    The one distribution primitive shared by every Green's-function driver (self-energy and
    spectra): ONE :func:`basis_split.split_basis_and_redistribute_psi` over all units, each color runs
    ``kernel(split_basis, unit_index, seeds)`` for its assigned units on its sub-communicator,
    and the per-unit results are gathered to global rank 0 in global unit order.

    ``kernel`` must be collective on ``split_basis.comm`` only (every rank of a color executes
    the identical unit list, so MPI stays in lock-step) and return a picklable object.

    ``reduce_fn(unit_index, unit_result)``, when given, is called on global rank 0 (and in the
    serial path) as each color's payload arrives, and the payload is dropped afterwards --
    rank 0 then never holds more than one color's results at a time instead of all units
    simultaneously (e.g. the caller accumulates into a preallocated output tensor).

    ``reort`` is the GF reorthogonalization mode the kernel will run with, and ``gf_method``
    names the kernel family (``"lanczos"`` / ``"bicgstab"``); both only feed the memory model
    that caps the number of simultaneous colors (each color's unit basis may fill the same
    ``truncation_threshold`` on fewer ranks, so memory bounds the concurrency).

    Returns
    -------
    list or None
        On global rank 0 (and on every rank in the serial path): ``results[u]`` = kernel result
        for unit ``u``, or ``True`` when ``reduce_fn`` consumed the results. ``None`` on other
        ranks. The split communicator is freed collectively before returning.
    """
    n_units = len(unit_seeds)
    if basis.comm is None or basis.comm.size <= 1:
        if reduce_fn is not None:
            for u in range(n_units):
                reduce_fn(u, kernel(basis, u, unit_seeds[u]))
            return True
        return [kernel(basis, u, unit_seeds[u]) for u in range(n_units)]

    seed_offsets = np.concatenate(([0], np.cumsum([len(s) for s in unit_seeds]))).astype(int)
    # Every color's unit basis inherits the same truncation_threshold, so colors multiply
    # per-rank memory: each rank's share of a capped unit basis is threshold/(ranks/n_colors).
    # Cap the concurrency so a cap-filling unit basis still fits the per-rank budget. The
    # probe is collective on basis.comm; the gates (cap finiteness, unit/rank counts) are
    # replicated, so every rank computes the identical max_colors.
    cap = getattr(basis, "truncation_threshold", np.inf)
    width = max((len(s) for s in unit_seeds), default=1)
    max_colors = None
    if np.isfinite(cap) and min(basis.comm.size, n_units) > 1:
        max_colors = max_colors_within_budget(
            int(cap), basis.num_spin_orbitals, width, reort, basis.comm, min(basis.comm.size, n_units), method=gf_method
        )
        if verbose and basis.comm.rank == 0 and max_colors < min(basis.comm.size, n_units):
            print(
                f"Memory budget caps the unit split at {max_colors} simultaneous unit bases "
                f"(truncation_threshold={int(cap):,}).",
                flush=True,
            )
    (
        unit_indices,
        unit_roots,
        _unit_color,
        units_per_color,
        split_basis,
        split_seeds,
        _,  # intercomms -- freed collectively inside the split
    ) = split_basis_and_redistribute_psi(basis, unit_weights, [s for seeds in unit_seeds for s in seeds], max_colors)
    if verbose:
        print(f"New unit roots: {unit_roots}")
        print(f"Units per color: {units_per_color}")
        print("=" * 80, flush=True)
    # All inputs of the per-color prediction are replicated (no collectives), so gating the
    # print on rank 0 is safe.
    if verbose and basis.comm.rank == 0 and np.isfinite(cap):
        n_colors = len(units_per_color)
        per_rank = estimate_gf_peak_bytes(
            int(cap),
            basis.num_spin_orbitals,
            width,
            reort,
            ranks=max(1, basis.comm.size // max(1, n_colors)),
            method=gf_method,
        )
        print(
            f"{n_colors} simultaneous unit bases at truncation_threshold={int(cap):,}: "
            f"predicted per-rank GF peak {format_bytes(per_rank)} if a unit fills its cap.",
            flush=True,
        )
    sub_rank = split_basis.comm.rank if split_basis.comm is not None else 0
    unit_indices_per_color = gather_distributed_results(
        basis.comm, sub_rank, unit_roots, units_per_color, np.array(unit_indices), is_array=True
    )

    local_results = [kernel(split_basis, u, split_seeds[seed_offsets[u] : seed_offsets[u + 1]]) for u in unit_indices]

    results = None
    if reduce_fn is None:
        gathered = gather_distributed_results(
            basis.comm, sub_rank, unit_roots, units_per_color, local_results, is_array=False
        )
        if basis.comm.rank == 0:
            results = [None] * n_units
            for i, u in enumerate(unit_indices_per_color):
                results[int(u)] = gathered[i]
    elif basis.comm.rank == 0:
        # Streaming consume: receive one color's payload at a time (same color order and
        # send/recv pairing as gather_distributed_results), reduce it, drop it.
        offset = 0
        for count, root in zip(units_per_color, unit_roots):
            if count == 0:
                continue
            payload = local_results if root == 0 else basis.comm.recv(source=root)
            for i in range(count):
                reduce_fn(int(unit_indices_per_color[offset + i]), payload[i])
            payload = None
            offset += count
        local_results = None
        results = True
    elif sub_rank == 0:
        basis.comm.send(local_results, dest=0)
        local_results = None

    # Free the split communicator collectively before returning. MPI_Comm_free is collective --
    # it must be called by all ranks in the comm at the same time. Leaving it for Python gc risks
    # non-collective freeing.
    if split_basis is not None and split_basis.comm != basis.comm:
        split_basis.free_comm()
    return results


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
        alphas, betas, r, n_basis, cap_stats = _block_green_group(
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
            print(f"Expanded excited state basis contains {n_basis} elements.")
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


def _apply_transition_ops(tOps, psis, excited_restrictions, excited_weighted_restrictions, slaterWeightMin):
    """Apply each transition operator to every thermal state, returning the seed blocks.

    Returns ``block_v`` indexed ``[j_psi][i_tOp]`` -- the excited state ``tOps[i] |psi_j>`` confined
    to the excited sector. These are the columns of each eigenstate's block-Lanczos seed.
    """
    # The thermal states share their support, so each transition operator is applied to
    # the whole block at once (term/sign/accumulator work once per determinant, near-flat
    # in the number of eigenstates — Phase 2 block-state matvec).
    psi_blk = ManyBodyBlockState.from_states(list(psis))
    block_v = [[ManyBodyState({}) for _ in tOps] for _ in psis]
    for i_tOp, tOp in enumerate(tOps):
        tOp.set_restrictions(excited_restrictions)
        tOp.set_weighted_restrictions(excited_weighted_restrictions)
        res_psis = tOp.apply_block(psi_blk, slaterWeightMin).to_states()
        for j_psi, res_psi in enumerate(res_psis):
            block_v[j_psi][i_tOp] += res_psi
    return block_v


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
    return alphas, betas, r, len(excited_basis), cap_stats


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
        alphas, betas, r, n_basis, _cap_stats = _block_green_group(
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
            print(f"Expanded excited state basis contains {n_basis} elements.")
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


def block_Green(
    hOp,
    psi_arr,
    basis,
    delta,
    reort,
    slaterWeightMin=0,
    verbose=True,
    eval_meshes=None,
    info=None,
):
    """
    calculate  one block of the Greens function. This function builds the many body basis iteratively. Reducing memory requrements.

    ``eval_meshes`` is the caller's evaluation mesh per axis (see :func:`_gf_eval_meshes`); ``None``
    leaves the convergence monitor on its spectral-edge fallback.

    ``info`` (optional dict) is filled with the last :func:`block_green_impl` call's
    ``{"converged", "d_g", "n_blocks"}`` (diagnostics; e.g. the RIXS R2 solve summary
    aggregates it across every call). A caller-supplied dict is mutated in place so a
    unit's cumulative counters keep accumulating across a whole run.
    """

    n = len(psi_arr)

    # alphas/betas stay padded (k, P, P) here so the cross-expansion elementwise
    # diff below has matching shapes; they are trimmed to true block widths before
    # any continued-fraction evaluation and at the final return.
    alphas, betas, r, last_q, widths = block_green_impl(
        basis, hOp, basis.redistribute_psis(psi_arr), delta, reort, slaterWeightMin, verbose, eval_meshes, info
    )
    done = False
    while not done:
        old_size = basis.size
        # Reachability probe: repeatedly apply H to the residual block to discover new
        # determinants. The block shares its support, so the block matvec applies here too.
        # The truncation_threshold check sits INSIDE the probe loop so the basis can
        # overshoot the cap by at most one H-application batch (checking only after all
        # five rounds used to blow past it by the full five-fold fanout). basis.size is
        # replicated by add_states, so the break is collective-consistent.
        probe = ManyBodyBlockState.from_states(list(last_q))
        capped = False
        for i in range(5):
            probe = hOp.apply_block(probe, slaterWeightMin)
            basis.add_states(
                set(state for state in probe.support_keys(0.0) if state not in basis.local_basis),
            )
            if basis.size > basis.truncation_threshold:
                capped = True
                break
        if basis.size == old_size or capped:
            break
        if verbose:
            print(f"    expanded basis contains {basis.size} states")
        alphas_prev = alphas
        betas_prev = betas
        widths_prev = widths
        alphas, betas, r, last_q, widths = block_green_impl(
            basis, hOp, basis.redistribute_psis(psi_arr), delta, reort, slaterWeightMin, verbose, eval_meshes, info
        )

        n_test = min(alphas.shape[0], alphas_prev.shape[0])
        # relatively large changes in alpha and/or betas means we have not converged
        if np.any(np.abs(alphas[:n_test] - alphas_prev[:n_test]) > 1e-12) or np.any(
            np.abs(betas[:n_test] - betas_prev[:n_test]) > 1e-12
        ):
            done = False
            continue

        # alphas seem decently converged, check the Greens function to be sure
        a_t, b_t = _trim_blocks(alphas, betas, widths)
        ap_t, bp_t = _trim_blocks(alphas_prev, betas_prev, widths_prev)
        ws = np.concatenate([np.diagonal(a) for a in a_t])[: n_test * n] if a_t else np.zeros(0, dtype=complex)
        G_prev = calc_G(ap_t, bp_t, np.identity(n), ws, 0, delta)
        G = calc_G(a_t, b_t, np.identity(n), ws, 0, delta)
        done = (
            np.all(np.diagonal(G.imag, axis1=1, axis2=2) * np.sign(delta) <= 0) and np.max(np.abs(G - G_prev)) < 1e-12
        )
    return _trim_blocks(alphas, betas, widths) + (r,)


# --- Per-frequency BiCGSTAB Green's function (gf_method="bicgstab") -------------------------
# The tunable parameters (atol, iteration bound, restarts, the GMRES fallback's restart
# lengths) are declared in `ed/config.py` and read at call time -- an import-time constant
# cannot be set by a caller that has already imported this module (which silently voided a
# slicing test once).
#
# Solutions retained for the warm start: quadratic extrapolation in z through the last three
# is the measured optimum (doc/plans/bicgstab_per_frequency_gf.md Phase 3a; cubic amplifies the
# atol-level noise it extrapolates through, and each retained block costs live memory).
_GF_BICGSTAB_WARM_HISTORY = 3
# A restart must shrink the reported residual by at least this factor to earn the next one, so
# a genuinely stuck point stops early and is reported rather than looping.
_GF_BICGSTAB_RESTART_PROGRESS = 0.5


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


def _gf_eigenstate_group():
    r"""Number of thermal eigenstates stacked into one block-Lanczos recurrence (the
    "wide block" granularity knob).

    For a Green's-function block of ``n_ops`` transition operators, ``g = 1`` (the default)
    runs one width-``n_ops`` recurrence per thermal eigenstate -- the historical behavior.
    ``g > 1`` stacks ``g`` eigenstates' seeds into a single width-``g * n_ops`` block that
    shares one Krylov space: the shared block-tridiagonal coefficients ``(alphas, betas)``
    are reused for every eigenstate in the group, while each eigenstate keeps its own
    ``n_ops`` columns of the seed projection ``r`` and its own energy shift, so
    :func:`calc_G` reconstructs that eigenstate's ``n_ops x n_ops`` Green's-function block
    exactly (the block Krylov space of the stacked seed contains each eigenstate's own
    Krylov space). Stacking shares the matvec/Krylov build across eigenstates but grows the
    per-step reorthogonalization with the block width, so the optimum is workload-dependent
    (see ``doc/plans/calc_selfenergy_performance.md``). Override with
    :data:`config.GF_EIGENSTATE_GROUP`.
    """
    return config.GF_EIGENSTATE_GROUP.get()


def _gf_operator_split():
    r"""Whether to use the pairwise / scalar operator-split decomposition (the *narrow* end of
    the block-width granularity spectrum). Default off (:data:`config.GF_OPERATOR_SPLIT`).

    When on, a block of ``n`` transition operators is computed not as one width-``n`` block-Lanczos
    recurrence but as ``n`` width-1 (scalar) recurrences for the diagonal seeds ``v_i = c_i|psi>``
    plus, for each off-diagonal pair ``i < j``, two more scalar recurrences for the polarization
    seeds ``v_i + v_j`` and ``v_i + i v_j``. The off-diagonal ``G_ij`` is recovered exactly from the
    four scalar resolvents (:func:`calc_G_pairwise`). This maximizes the number of independent
    (communication-free) work units -- useful when ranks greatly outnumber the
    ``block x eigenstate`` units -- at the cost of redundant Krylov building (no shared subspace
    across columns). Mutually exclusive with eigenstate grouping; the operator split takes
    precedence when both are requested.
    """
    return config.GF_OPERATOR_SPLIT.get()


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


def _union_restrictions(rests):
    r"""Loosest single restriction dict admitting every input window's feasible set.

    A work unit that stacks several eigenstates shares one block Krylov space, which must contain
    *every* stacked seed's dynamics; so the unit window must admit a determinant that is feasible
    for **any** state in the group. Restriction dicts are conjunctions of per-subset ``(min, max)``
    occupation bounds, so the group window keeps only the subset keys **common to all** states (a
    key absent from some state imposes no bound there, hence cannot be enforced for the group) and
    loosens each shared key to ``(min of mins, max of maxs)``. The result is a superset of each
    input window, so it never truncates a stacked state's Krylov space. ``None`` means "no
    restriction"; if any input is ``None`` (unconstrained) the union is ``None``. For a single-state
    group (operator-split, or ``g = 1``) the union is exactly that state's window -- maximal
    tightening.
    """
    rests = list(rests)
    if not rests or any(r is None for r in rests):
        return None
    if len(rests) == 1:
        return rests[0]
    common = set(rests[0])
    for r in rests[1:]:
        common &= set(r)
    if not common:
        return None
    out = {}
    for key in common:
        bounds = [r[key] for r in rests]
        out[key] = (min(lo for lo, _ in bounds), max(hi for _, hi in bounds))
    return out


def block_green_impl(basis, hOp, psi_arr, delta, reort, slaterWeightMin, verbose, eval_meshes=None, info=None):
    """
    Internal block Green's function implementation.

    Parameters
    ----------
    basis : Basis
        The many-body basis.
    hOp : dict
        Hamiltonian operator.
    psi_arr : list of ManyBodyState
        Input state vectors.
    delta : float or ndarray
        Imaginary part/mesh info.
    reort : Reort
        Reorthogonalization method.
    slaterWeightMin : float
        Slater determinant cutoff weight.
    verbose : bool
        Whether to print verbose output.
    info : dict, optional
        Filled with ``{"converged", "d_g", "n_blocks"}`` from the convergence monitor
        (diagnostics/tests; e.g. the RIXS R2 solve summary).

    Returns
    -------
    gs_matsubara : ndarray
        Matsubara Green's function.
    gs_realaxis : ndarray
        Real axis Green's function.
    r : ndarray
        R matrix from QR.
    psi_arr : list
        Resulting states.
    """
    n = len(psi_arr)

    comm = basis.comm
    rank = comm.rank if comm is not None else 0

    dense = len(basis) < 500
    if dense:
        psi_dense = build_vector(basis, psi_arr, slaterWeightMin=0).T
        psi_dense_local, r = build_qr(psi_dense)
    else:
        # 0, not `slaterWeightMin`: this branch has always built its seed block unpruned
        # (unlike block_Green_sparse/KrylovShiftedResolvent, which prune it) -- preserved
        # as-is here since this is a mechanical extraction, not a behaviour change.
        psi_dense_local, r = _distributed_seed_qr(basis, psi_arr, 0)

    if psi_dense_local.shape[1] == 0:
        return np.zeros((0, n, n), dtype=complex), np.zeros((0, n, n), dtype=complex), r, psi_arr, []

    converged, converged_flag, delta_min, last_dg = _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes)

    # The continued fraction only consumes alphas/betas plus the final residual block
    # (q_last below), so with reort NONE skip the full Krylov-basis retention.
    resolved_reort = resolve_reort(reort if reort is not None else Reort.NONE)

    if dense:
        H = build_dense_matrix(basis, hOp)
        alphas, betas, Q_list, widths, status = block_lanczos_array(
            psi0=psi_dense_local,
            h_op=H,
            converged=converged,
            verbose=False and verbose,
            reort=resolved_reort,
            build_krylov_basis=resolved_reort != Reort.NONE,
            return_widths=True,
            return_status=True,
            # ceil (not floor): spanning an N-dim (possibly closed) sector with a width-w block
            # needs ceil(N/w) blocks; floor truncates the final, deflating block and leaves up to
            # w-1 dimensions of the sector unresolved -- a systematic resolvent error that grows
            # with the block width (the RIXS tensor floor). The final block simply deflates.
            max_iter=-(-H.shape[0] // psi_dense_local.shape[1]),
        )
    else:
        h_local = build_sparse_matrix(basis, hOp)[:, basis.local_indices]

        def matmat(v):
            """
            Perform matrix-matrix multiplication with the local Hamiltonian.

            Applies the local Hamiltonian to a set of state vectors and performs
            an MPI reduction across MPI processes to accumulate the results.

            Parameters
            ----------
            v : ndarray
                Input vectors to multiply.

            Returns
            -------
            res : ndarray
                The resulting matrix product after MPI reduction.
            """
            res = h_local @ v
            if comm is not None:
                comm.Reduce(MPI.IN_PLACE if rank == 0 else res, res, op=MPI.SUM, root=0)
            return res.reshape(h_local.shape[0], v.shape[1])

        H = sp.sparse.linalg.LinearOperator(
            (len(basis), len(basis.local_indices)),
            matvec=matmat,
            rmatvec=matmat,
            matmat=matmat,
            rmatmat=matmat,
            dtype=complex,
        )

        # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
        alphas, betas, Q_list, widths, status = block_lanczos_array(
            psi0=psi_dense_local,
            h_op=H,
            converged=converged,
            reort=resolved_reort,
            build_krylov_basis=resolved_reort != Reort.NONE,
            verbose=False and verbose,
            comm=comm,
            return_widths=True,
            return_status=True,
            # ceil (not floor): spanning an N-dim (possibly closed) sector with a width-w block
            # needs ceil(N/w) blocks; floor truncates the final, deflating block and leaves up to
            # w-1 dimensions of the sector unresolved -- a systematic resolvent error that grows
            # with the block width (the RIXS tensor floor). The final block simply deflates.
            max_iter=-(-H.shape[0] // psi_dense_local.shape[1]),
        )
    # An invariant subspace closes the Krylov space under H, so the continued fraction is
    # exact: treat it as converged (same semantics as the sparse path) so it does not trip
    # the non-convergence warning below.
    if status == "invariant_subspace":
        converged_flag[0] = True
    if not converged_flag[0] and verbose and rank == 0:
        print(
            f"warning: block Green's function did not reach the convergence tolerance "
            f"{delta_min:.1e} in {len(alphas)} block(s). The continued fraction uses the "
            f"subspace built so far.",
            flush=True,
        )
    if info is not None:
        info["converged"] = converged_flag[0]
        info["d_g"] = last_dg[0]
        info["n_blocks"] = len(alphas)
    # Keep alphas/betas padded (k, P, P) for the caller's elementwise cross-expansion diff;
    # only drop a corrupted trailing tail (whole blocks + widths) so it never reaches the
    # continued fraction. Norms of padded blocks equal those of the true blocks (zeros add
    # nothing), so the scan is valid on the padded arrays.
    keep = len(_sanitize_continued_fraction(list(alphas), list(betas), verbose=verbose, rank=rank)[0])
    if keep < len(alphas):
        alphas, betas, widths = alphas[:keep], betas[:keep], widths[:keep]
    q_last = Q_list[:, -1:]
    return alphas, betas, r, build_state(basis, q_last.T, slaterWeightMin=slaterWeightMin), widths


def block_Green_sparse(
    hOp,
    psi_arr,
    basis,
    delta,
    reort: Optional = None,
    slaterWeightMin=0,
    verbose=True,
    cap_info=None,
    krylov_dtype=None,
    eval_meshes=None,
):
    """
    calculate  one block of the Greens function. This function builds the many body basis iteratively. Reducing memory requrements.

    ``basis.truncation_threshold`` caps the number of Slater determinants the
    recurrence may touch (see :class:`_CappedBasisProxy`); ``np.inf`` (the ``Basis``
    default) leaves the growth bounded only by ``slaterWeightMin`` and the
    restrictions. Pass a dict as ``cap_info`` to receive ``{"cap_hit",
    "retained_size", "proxy"}`` back (diagnostics/tests).

    ``krylov_dtype`` sets the storage precision of the retained Krylov basis, which is the
    dominant allocation of a reorthogonalized run (``16 * p * n_blocks`` bytes per retained
    determinant, ~30x everything else at the FCC-Ni operating point). ``complex64`` halves
    it, at the cost of an orthogonality (and Green's function) floor at fp32 roundoff,
    ~6e-8. It is **opt-in**, not the default, for two reasons: it is rejected outright by
    ``PARTIAL``/``SELECTIVE``, whose Paige-Simon estimator steers to ``sqrt(EPS) ~ 1.5e-8``
    and cannot be fed a basis known only to ~6e-8; and it would silently break the exactness
    guarantee that a capped recurrence reproduces the dense ``P H P`` resolvent (see
    ``test_gf_truncation``). Only the *stored* basis narrows -- the recurrence, the overlaps
    and the residual stay complex128. See ``doc/plans/blocklanczos_reort_memory.md``.

    ``eval_meshes`` is the caller's evaluation mesh per axis (see :func:`_gf_eval_meshes`), which
    the convergence monitor tests ``G`` on. ``None`` leaves it on the spectral-edge fallback, which
    converges the real-axis resolvent whether or not a real-axis mesh was asked for.
    """
    mpi = basis.comm is not None
    comm = basis.comm if mpi else None
    rank = comm.rank if mpi else 0

    N = len(basis)
    n = len(psi_arr)

    if N == 0 or n == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), np.zeros((n, n), dtype=complex)
    psi_dense_local, r = _distributed_seed_qr(basis, psi_arr, slaterWeightMin)
    psi_arr = build_state(basis, psi_dense_local.T, slaterWeightMin=0)
    if len(psi_arr) == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), r

    converged, converged_flag, delta_min, _last_dg = _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes)

    # The block-Lanczos matvec (h_op.apply_multi) discovers new Slater determinants as the
    # recurrence proceeds, so the reachable Krylov dimension is *not* bounded by the initial
    # excited-basis size: convergence can require many more blocks than basis.size // n. Rather
    # than guess one large cap (which either cuts the recurrence off early or wastes work),
    # resume the recurrence in growing chunks until either the Green's function converges or
    # the recurrence terminates on its own (invariant subspace / rank-deficient residual), at
    # which point the continued fraction is already exact on the space built so far. A round
    # returns fewer than `budget` new blocks exactly when the kernel stopped early; otherwise
    # it used the whole budget and there may be more spectrum to resolve, so we extend it.
    alphas = betas = Q = W = widths = None
    budget = max(int(getattr(basis, "size", 0)) // max(n, 1), 1)
    # Enforce the determinant cap on the recurrence: the proxy persists across the
    # resume rounds below, so the retained set (and a freeze) carries over.
    cap = getattr(basis, "truncation_threshold", np.inf)
    lanczos_basis = _CappedBasisProxy(basis, cap) if np.isfinite(cap) else basis
    # With reort NONE the kernel never projects against the accumulated Krylov basis and
    # the resume protocol reads only the two-block tail, so skip the full retention.
    resolved_reort = resolve_reort(reort if reort is not None else Reort.NONE)
    while True:
        alphas, betas, Q, W, widths, status = block_lanczos_cy(
            psi_arr,
            hOp,
            lanczos_basis,
            converged,
            verbose=verbose,
            reort=resolved_reort,
            slaterWeightMin=slaterWeightMin,
            max_iter=budget,
            return_widths=True,
            return_status=True,
            alphas_init=alphas,
            betas_init=betas,
            Q_init=Q,
            W_init=W,
            block_widths_init=widths,
            store_krylov=resolved_reort != Reort.NONE,
            krylov_dtype=krylov_dtype,
        )
        # The kernel reports exactly why it stopped (see block_lanczos_cy):
        #   * "converged"          -- the GF convergence monitor was satisfied.
        #   * "invariant_subspace" -- the block-Krylov space is closed under H (within the
        #                             excited-sector restrictions), so the continued fraction
        #                             is *exact*: this is a converged result.
        #   * "diverged"           -- the divergence guard truncated a corrupted tail; not
        #                             converged, and no further blocks can be built.
        #   * "max_iter"           -- the budget was exhausted while the matvec was still
        #                             reaching new determinants; grow the budget and resume.
        if status in ("converged", "invariant_subspace"):
            converged_flag[0] = True
            break
        if status == "diverged":
            break
        budget *= 2

    if isinstance(lanczos_basis, _CappedBasisProxy):
        if lanczos_basis.cap_hit and verbose and rank == 0:
            print(lanczos_basis.freeze_message(), flush=True)
        if cap_info is not None:
            cap_info["cap_hit"] = lanczos_basis.cap_hit
            cap_info["retained_size"] = lanczos_basis.retained_size
            cap_info["proxy"] = lanczos_basis
    elif cap_info is not None:
        cap_info["cap_hit"] = False
        cap_info["retained_size"] = None
        cap_info["proxy"] = None

    if not converged_flag[0] and rank == 0:
        print(
            f"warning: block Green's function did not reach the convergence tolerance "
            f"{delta_min:.1e}; the block-Lanczos recurrence was truncated "
            f"after {len(alphas)} block(s) (divergent tail). The continued fraction uses the "
            f"subspace built so far.",
            flush=True,
        )

    alphas, betas = _trim_blocks(alphas, betas, widths)
    alphas, betas = _sanitize_continued_fraction(alphas, betas, verbose=verbose, rank=rank)
    return alphas, betas, r


def _warm_start_extrapolation(zs, sols, z_new, n_cols):
    r"""Warm-start guess at ``z_new``: Lagrange extrapolation through the retained solutions.

    ``zs``/``sols`` hold the last (up to :data:`_GF_BICGSTAB_WARM_HISTORY`) frequencies and
    solution blocks of the sweep, oldest first. Zero, one and two retained solutions give the
    cold start, the previous solution and linear extrapolation respectively; three gives the
    quadratic optimum. The coefficients sum to 1 (an extrapolation, not a fit), so a solution
    that is locally polynomial in ``z`` is reproduced exactly.
    """
    if not sols:
        return [ManyBodyState() for _ in range(n_cols)]
    coeffs = []
    for k, zk in enumerate(zs):
        c = 1.0 + 0j
        for j, zj in enumerate(zs):
            if j != k:
                c *= (z_new - zj) / (zk - zj)
        coeffs.append(c)
    return [sum((sol[col] * c for c, sol in zip(coeffs, sols)), ManyBodyState()) for col in range(n_cols)]


def _bicgstab_sweep_order(z_shifted):
    r"""Sweep indices from the easiest frequency toward the hardest.

    Distance to the spectrum is governed by ``|Im z|``: a point far from the real axis is
    nearly diagonal-dominant and converges in a couple of iterations, so sweeping from large
    ``|Im z|`` down builds the warm-start chain on cheap solves before it reaches the hard
    region. On a fixed-broadening real-axis mesh all ``|Im z|`` are equal and the stable sort
    leaves the caller's (monotone-in-``omega``) order unchanged -- exactly the contiguous
    sweep the warm start wants there.
    """
    return np.argsort(-np.abs(np.imag(z_shifted)), kind="stable")


def solve_shifted_block(A_op, x0, rhs, basis, slaterWeightMin, atol, rtol=0.0, max_iter=None, info=None):
    r"""Restart-while-progressing BiCGSTAB, escalated to ``block_gmres`` on stagnation.

    Shared by every per-frequency resolvent solve on this branch (:func:`block_Green_bicgstab`,
    the RIXS R1 fallback in ``rixs._rixs_map_flat``): runs up to ``1 + config.GF_BICGSTAB_RESTARTS``
    :func:`~impurityModel.ed.cg.block_bicgstab` attempts, restarting with the current iterate as
    long as each attempt still makes at least ``_GF_BICGSTAB_RESTART_PROGRESS`` progress over the
    previous residual (each restart re-deflates ``Y - A x0`` and picks a fresh shadow residual,
    which is what cures near-pole stagnation -- a plain re-solve from the same iterate would not).
    If still unconverged after the restarts, escalates to :func:`~impurityModel.ed.gmres.block_gmres`,
    warm-started from BiCGSTAB's last iterate, before that iterate can poison a warm-start chain
    downstream.

    Every field of ``info`` derives from allreduce'd norms (``block_bicgstab``/``block_gmres`` are
    collective), so this restart loop is collective-consistent: every rank takes the same branch.

    Parameters
    ----------
    A_op, x0, rhs, basis, slaterWeightMin, atol
        Forwarded to ``block_bicgstab``/``block_gmres`` (``x0`` is the warm start; ``rhs`` is
        the right-hand side block, ``y`` in their signature).
    rtol : float, optional
        BiCGSTAB-only relative tolerance floor (some callers pin the RIXS R1 solve to one
        additionally); 0 (default) omits it and uses ``block_bicgstab``'s own default.
    max_iter : int, optional
        BiCGSTAB-only per-attempt iteration bound; ``None`` uses ``block_bicgstab``'s default.
    info : dict, optional
        Filled (created if not supplied) with ``converged``, ``rel_residual`` (both as reported
        by whichever solver ran last), cumulative ``iterations`` across every attempt and any
        GMRES escalation, ``gmres_used`` and ``gmres_iterations``.

    Returns
    -------
    list of ManyBodyState
        The solution block.
    """
    if info is None:
        info = {}
    bicgstab_kwargs = {"atol": atol, "info": info}
    if rtol:
        bicgstab_kwargs["rtol"] = rtol
    if max_iter is not None:
        bicgstab_kwargs["max_iter"] = max_iter

    iterations = 0
    X = x0
    prev_residual = np.inf
    for _attempt in range(1 + config.GF_BICGSTAB_RESTARTS.get()):
        X = block_bicgstab(A_op, X, rhs, basis, slaterWeightMin, **bicgstab_kwargs)
        iterations += info["iterations"]
        if info["converged"] or info["rel_residual"] > _GF_BICGSTAB_RESTART_PROGRESS * prev_residual:
            break
        prev_residual = info["rel_residual"]

    gmres_used = False
    gmres_iterations = 0
    if not info["converged"]:
        X = block_gmres(
            A_op,
            X,
            rhs,
            basis,
            slaterWeightMin,
            atol=atol,
            restart=config.GF_GMRES_RESTART.get(),
            max_restarts=config.GF_GMRES_MAX_RESTARTS.get(),
            info=info,
        )
        iterations += info["iterations"]
        gmres_used = True
        gmres_iterations = info["iterations"]

    info["iterations"] = iterations
    info["gmres_used"] = gmres_used
    info["gmres_iterations"] = gmres_iterations
    return X


def block_Green_bicgstab(
    hOp,
    psi_arr,
    basis,
    es,
    n_ops,
    z_axes,
    slaterWeightMin=0,
    atol=None,
    max_iter=None,
    verbose=False,
    excited_restrictions=None,
    excited_weighted_restrictions=None,
    bra_seeds=None,
):
    r"""Per-frequency BiCGSTAB Green's function for one work unit (memory-first path).

    For every stacked eigenstate ``e`` and every frequency ``z`` of every requested axis this
    solves the resolvent linear system

    .. math:: (z + E_e - H)\, X = \text{seeds}_e, \qquad
              G_e[i, j](z) = \langle \text{seed}_i | X_j \rangle

    instead of running one block-Lanczos recurrence for the whole mesh. The memory contract is
    the point: the excited basis is **rebuilt from the current seed + warm-start support and
    discarded at every frequency point** (the RIXS resolvent's ``tmp_basis`` pattern), so the
    retained footprint is the largest *single-point* support, not the union over the mesh that
    a Lanczos recurrence accumulates -- and a finite ``basis.truncation_threshold`` caps even
    that via :class:`_CappedBasisProxy` (freeze-growth, exact on the retained subspace). No
    Krylov store exists on this path and there is no orthogonality to lose, so accuracy is set
    by ``atol`` alone.

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian. Each point solves against a fresh ``z*I - hOp`` operator (the RIXS
        identity-operator construction), whose occupation restrictions come from the rebuilt
        basis and whose weighted restrictions are set from ``excited_weighted_restrictions``.
    psi_arr : list of ManyBodyState
        Flat seed columns in ``(eigenstate, operator)`` order -- ``len(es) * n_ops`` entries,
        exactly the unit-seed convention of :func:`enumerate_gf_units`.
    basis : Basis
        The unit's (split) basis; carries the communicator, the clone template and
        ``truncation_threshold``.
    es : sequence of float
        Energies of the stacked eigenstates. Each eigenstate is solved separately: the shift
        enters the operator, so solves cannot be stacked across eigenstates the way one
        Lanczos recurrence serves them all.
    n_ops : int
        Seed columns per eigenstate (the Green's-function block width).
    z_axes : list of ndarray
        Complex frequency axes from :func:`_gf_signed_axes` -- *before* the ``E_e`` shift,
        which is applied here per eigenstate.
    atol : float, optional
        Per-solve residual tolerance relative to the seed norm; defaults to
        :data:`config.GF_BICGSTAB_ATOL`.
    max_iter : int, optional
        Per-point iteration bound; defaults to :data:`config.GF_BICGSTAB_MAX_ITER`.
    bra_seeds : list of ManyBodyState, optional
        Cross-element mode (the spectrum-slicing driver): a second flat block in the same
        ``(eigenstate, operator)`` order whose columns form the *bra* of the Gram,
        ``G_e[i, j] = <bra_i | X_j>`` -- e.g. the unfiltered seeds against a filtered
        right-hand side, computing ``<v| (z-H)^{-1} p_s(H) |v>``. ``None`` (default) uses
        the seeds themselves (the symmetric element).

    Returns
    -------
    tuple
        ``(G_axes, stats)``: ``G_axes[ax][p, k]`` is the ``n_ops x n_ops`` block of eigenstate
        ``p`` at frequency ``k`` of axis ``ax`` (caller's mesh order), and ``stats`` is the
        reliability/memory record consumed by the diagnostics -- solver convergence
        (``n_points``, ``n_unconverged``, ``max_rel_residual``, ``iterations``), the cap state
        (``cap``, ``cap_hit``, ``retained_size``, ``seed_overflow``) and the measured
        per-point support (``max_solve_basis``, ``max_rebuild_basis`` -- the numbers that
        decide whether this path's memory promise holds on a given workload).
    """
    atol = config.GF_BICGSTAB_ATOL.get() if atol is None else atol
    max_iter = config.GF_BICGSTAB_MAX_ITER.get() if max_iter is None else max_iter
    n_e = len(es)
    sub_comm = basis.comm
    cap = getattr(basis, "truncation_threshold", np.inf)
    # One clone (and one cloned communicator) per unit; the per-point rebuild is
    # clear() + add_states, never a re-clone. Freed collectively below -- every rank of the
    # color runs the identical unit list, so this stays in lock-step.
    tmp_basis = basis.clone(
        initial_basis=[],
        restrictions=excited_restrictions,
        weighted_restrictions=excited_weighted_restrictions,
        verbose=False,
        comm=sub_comm.Clone() if sub_comm is not None else None,
    )

    G_axes = [np.zeros((n_e, len(z_axis), n_ops, n_ops), dtype=complex) for z_axis in z_axes]
    stats = {
        "n_points": 0,
        "n_unconverged": 0,
        "max_rel_residual": 0.0,
        "iterations": 0,
        "gmres_points": 0,
        "gmres_iterations": 0,
        "atol": atol,
        "cap": cap,
        "cap_hit": False,
        "retained_size": None,
        "seed_overflow": False,
        "max_solve_basis": 0,
        "max_rebuild_basis": 0,
    }

    for p in range(n_e):
        seeds = list(psi_arr[p * n_ops : (p + 1) * n_ops])
        # Cross-element mode (spectrum slicing): the bra of the Gram is a separate block
        # (the unfiltered seeds) riding along through every per-point redistribution.
        bras = list(bra_seeds[p * n_ops : (p + 1) * n_ops]) if bra_seeds is not None else None
        for ax, z_axis in enumerate(z_axes):
            z_shifted = z_axis + es[p]
            # Fresh warm-start chain per (eigenstate, axis): extrapolating across axes (or
            # across eigenstates) would extrapolate through a discontinuous z-path.
            hist_z: list[complex] = []
            hist_x: list[list[ManyBodyState]] = []
            for k in _bicgstab_sweep_order(z_shifted):
                z = complex(z_shifted[k])
                x0 = _warm_start_extrapolation(hist_z, hist_x, z, n_ops)
                if slaterWeightMin > 0:
                    for x in x0:
                        x.prune(slaterWeightMin)
                # Rebuild-and-discard: the basis holds only this point's seed + warm-start
                # support; redistribute_psis aligns the amplitudes to the fresh ownership
                # layout (the solver assumes its states are distributed per `basis`).
                #
                # The bras are redistributed but deliberately NOT added to the basis. They
                # enter only the closing Gram, and block_inner_cy merge-joins the two key
                # vectors, so a determinant in supp(bra)\supp(X) contributes nothing;
                # ownership is by determinant hash, which is basis-independent, so the
                # merge-join stays MPI-consistent. Admitting them would pin every basis to
                # the *unfiltered* seed support -- exactly the quantity spectrum slicing
                # exists to avoid paying (on FCC Ni the unfiltered seeds saturate the cap,
                # so it would have silently capped every slice at the union support).
                carried = seeds + x0 + (bras if bras is not None else [])
                tmp_basis.clear()
                tmp_basis.add_states(sorted({state for psi in seeds + x0 for state in psi.keys()}))
                redistributed = tmp_basis.redistribute_psis(carried)
                seeds = list(redistributed[:n_ops])
                x0 = list(redistributed[n_ops : 2 * n_ops])
                if bras is not None:
                    bras = list(redistributed[2 * n_ops :])
                stats["max_rebuild_basis"] = max(stats["max_rebuild_basis"], int(tmp_basis.size))

                solve_basis = tmp_basis
                if np.isfinite(cap):
                    if tmp_basis.size > cap:
                        # The seed/warm-start support alone exceeds the cap. Never truncate
                        # the right-hand side silently: solve on it frozen (exact on that
                        # subspace) and flag it for the diagnostics.
                        stats["seed_overflow"] = True
                    solve_basis = _CappedBasisProxy(tmp_basis, cap)

                # z*(n_0 + h_0) = z*I -- the RIXS identity-operator construction. A fresh
                # operator per point: block_bicgstab sets its occupation restrictions from
                # the basis; the weighted restrictions are set here (unconditionally, so a
                # None clears any stale mask -- the Basis.expand convention).
                A_op = ManyBodyOperator({((0, "c"), (0, "a")): z, ((0, "a"), (0, "c")): z}) - hOp
                A_op.set_weighted_restrictions(excited_weighted_restrictions)

                # Solve, restarting while unconverged and still making progress and
                # escalating to GMRES on stagnation (block_Green_bicgstab's own warm-start
                # chain is separate from the RIXS one but shares the same solver policy).
                info = {}
                X = solve_shifted_block(
                    A_op, x0, seeds, solve_basis, slaterWeightMin, atol, max_iter=max_iter, info=info
                )

                stats["n_points"] += 1
                stats["iterations"] += info["iterations"]
                stats["max_rel_residual"] = max(stats["max_rel_residual"], info["rel_residual"])
                if info["gmres_used"]:
                    stats["gmres_points"] += 1
                    stats["gmres_iterations"] += info["gmres_iterations"]
                if not info["converged"]:
                    stats["n_unconverged"] += 1
                stats["max_solve_basis"] = max(stats["max_solve_basis"], int(tmp_basis.size))
                if isinstance(solve_basis, _CappedBasisProxy) and solve_basis.cap_hit:
                    stats["cap_hit"] = True
                    retained = solve_basis.retained_size
                    if stats["retained_size"] is None or retained < stats["retained_size"]:
                        stats["retained_size"] = retained

                # G_e[i, j] = <bra_i | X_j> (bra = seeds unless the caller supplied a
                # separate bra block); both blocks live on tmp_basis's layout, so the
                # local Gram + Allreduce is the whole inner product (no state-vector gather).
                gram = block_inner_cy(
                    ManyBodyBlockState.from_states(bras if bras is not None else seeds),
                    ManyBodyBlockState.from_states(list(X)),
                )
                if sub_comm is not None:
                    sub_comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
                G_axes[ax][p, k] = gram

                hist_z.append(z)
                hist_x.append(list(X))
                if len(hist_z) > _GF_BICGSTAB_WARM_HISTORY:
                    hist_z.pop(0)
                    hist_x.pop(0)
            if verbose:
                print(
                    f"    axis {ax}, eigenstate {p}: {len(z_shifted)} solves, "
                    f"{stats['iterations']} cumulative iterations, "
                    f"max per-point basis {stats['max_solve_basis']}",
                    flush=True,
                )

    if sub_comm is not None:
        tmp_basis.free_comm()
    return G_axes, stats


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
