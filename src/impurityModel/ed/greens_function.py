import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy as sp
from mpi4py import MPI

# from impurityModel.ed import spectra
from impurityModel.ed.basis_restrictions import build_excited_restrictions
from impurityModel.ed.basis_split import split_basis_and_redistribute_psi
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.BlockLanczosArray import (
    Reort,
    block_lanczos_array,
    resolve_reort,
    BETA_BLOWUP_FACTOR,
)
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.manybody_basis import Basis, collective_amplitude_cutoff
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
from impurityModel.ed.memory_estimate import estimate_gf_peak_bytes, format_bytes, max_colors_within_budget
from impurityModel.ed.mpi_comm import gather_distributed_results
from impurityModel.ed import gf_diagnostics as _gfd
from impurityModel.ed.basis_transcription import build_dense_matrix, build_sparse_matrix, build_state, build_vector

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
        unit_basis = _capped(_excited_clone(u))
        bounds = spectral_bounds(hOp, unit_basis)
        ends = [e + sign * w for e in chunk_es for w in (w_lo, w_hi)]
        band_lo = max(bounds[0], min(ends))
        band_hi = min(bounds[1], max(ends))
        slice_width = max((band_hi - band_lo) / max(_GF_SLICES, 1), 1e-12)
        degree = _GF_SLICE_DEGREE or int(np.clip(8.0 * (bounds[1] - bounds[0]) / slice_width, 200, 4000))
        coeff_sets, _windows, edge_width = partition_of_unity(
            bounds, np.linspace(band_lo, band_hi, _GF_SLICES + 1), degree
        )
        if verbose and (basis.comm is None or basis.comm.rank == 0):
            print(
                f"Spectrum slicing unit {u}: bounds [{bounds[0]:.3f}, {bounds[1]:.3f}], "
                f"{len(coeff_sets)} windows, degree {degree}, slice tol {_GF_SLICE_TOL:g}.",
                flush=True,
            )
        filtered = chebyshev_apply(hOp, unit_basis, list(unit_seeds[u]), coeff_sets, slaterWeightMin, bounds)
        seed_norm2 = sum(s.norm2() for s in unit_seeds[u])
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
            if _GF_SLICE_TOL > 0:
                for ket in kets:
                    ket.prune(_GF_SLICE_TOL)
            sliced_meta.append((block_i, side_i, unit.chunk, unit.n_ops, u))
            sliced_seeds.append(list(kets) + list(unit_seeds[u]))
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
        return [_gfd.check_slice_partition(n_windows, degree_used, edge_width, _GF_SLICE_TOL)]

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


def build_qr(psi):
    """
    Perform an economic QR decomposition of a state matrix.

    Parameters
    ----------
    psi : ndarray
        The input state matrix.

    Returns
    -------
    psi_orthogonal : ndarray
        The orthogonalized matrix Q.
    r : ndarray
        The upper triangular matrix R.
    """
    # Do a QR decomposition of the starting block.
    # Later on, use r to restore the psi block
    psi, r = sp.linalg.qr(psi.copy(), mode="economic", overwrite_a=True, check_finite=False, pivoting=False)
    return np.ascontiguousarray(psi), r


def calc_continuants(diagonal, offdiagonal):
    """
    Calculate continued fraction continuants.

    """

    An = np.empty_like(diagonal)
    Bn = np.empty_like(An)
    An[-1] = np.eye(diagonal.shape[1])
    Bn[-1] = 0
    An[0] = diagonal[0]
    Bn[0] = 1
    for n in range(1, diagonal.shape[0]):
        An[n] = diagonal[n] * An[n - 1] - np.conj(offdiagonal[n]) * An[n - 2] * offdiagonal[n]
        Bn[n] = diagonal[n] * Bn[n - 1] - np.conj(offdiagonal[n]) * Bn[n - 2] * offdiagonal[n]
    return An, Bn


def block_Green(
    hOp,
    psi_arr,
    basis,
    delta,
    reort,
    slaterWeightMin=0,
    verbose=True,
    eval_meshes=None,
):
    """
    calculate  one block of the Greens function. This function builds the many body basis iteratively. Reducing memory requrements.

    ``eval_meshes`` is the caller's evaluation mesh per axis (see :func:`_gf_eval_meshes`); ``None``
    leaves the convergence monitor on its spectral-edge fallback.
    """

    len(basis)
    n = len(psi_arr)

    # alphas/betas stay padded (k, P, P) here so the cross-expansion elementwise
    # diff below has matching shapes; they are trimmed to true block widths before
    # any continued-fraction evaluation and at the final return.
    alphas, betas, r, last_q, widths = block_green_impl(
        basis, hOp, basis.redistribute_psis(psi_arr), delta, reort, slaterWeightMin, verbose, eval_meshes
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
            basis, hOp, basis.redistribute_psis(psi_arr), delta, reort, slaterWeightMin, verbose, eval_meshes
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


def _scatter_qr_columns(comm, psi_dense, r, local_size):
    """Scatter the row-distributed QR factor ``Q`` (held on rank 0) across MPI ranks.

    Rank 0 holds the full ``(N, n)`` ``Q`` and the ``(n, n)`` ``R`` after :func:`build_qr`.
    Broadcast ``R`` and the column count, then ``Scatterv`` ``Q``'s rows onto each rank's
    local partition (``local_size`` rows). Shared by ``block_green_impl`` (sparse branch)
    and ``block_Green_sparse``.

    Returns
    -------
    psi_dense_local : ndarray
        This rank's ``(local_size, n)`` slice of ``Q``.
    r : ndarray
        The ``(n, n)`` ``R`` factor (replicated on every rank).
    """
    rank = comm.rank
    r = comm.bcast(r if rank == 0 else None, root=0)
    columns = comm.bcast(psi_dense.shape[1] if rank == 0 else None, root=0)
    psi_dense_local = np.empty((local_size, columns), dtype=complex, order="C")
    send_counts = np.empty((comm.size), dtype=int) if rank == 0 else None
    comm.Gather(np.array([psi_dense_local.size]), send_counts, root=0)
    offsets = np.array([np.sum(send_counts[:rr]) for rr in range(comm.size)], dtype=int) if rank == 0 else None
    comm.Scatterv(
        [psi_dense, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None,
        psi_dense_local,
        root=0,
    )
    return psi_dense_local, r


# Relative-change convergence floor for the block-Lanczos Green's function, shared by the
# runtime monitor (_make_gf_convergence_monitor) and the post-hoc diagnostic summary
# (_lanczos_convergence_summary) so the two can never disagree -- single source of truth.
#
# 1e-9, not the historical 1e-6. While the monitor sampled the whole Ritz band on the real axis it
# converged a resolvent the caller often never evaluated, and so *over*-delivered: a declared 1e-6
# returned sigma accurate to 1e-13..1e-15. Now that it tests G on the caller's own mesh the
# tolerance means what it says, and leaving the floor at 1e-6 would have quietly turned that into
# ~5e-8. Measured end to end on the 20-bath NiO self-energy, sigma against a deeply-converged
# reference, with the number of Lanczos blocks the run actually produced:
#
#     monitor        floor    Matsubara            real axis            blocks
#     band-wide      1e-6     3.2e-15              2.3e-13                 336
#     caller's mesh  1e-6     2.0e-08              5.2e-08                 158
#     caller's mesh  1e-9     3.5e-11              2.4e-12                 180 / 214
#     caller's mesh  1e-10    3.8e-14              2.4e-12                 214
#
# So 1e-9 is a deliberate trade, not a restoration: against the old *accidental* accuracy it gives
# up ~4 orders on the Matsubara axis and ~10x on the real axis, both still comfortably inside the
# tolerance it now honestly declares, and it needs 1.9x fewer blocks (and 1.9x less retained Q).
# Callers who want the old numbers back should set 1e-10, which costs 1.57x instead.
#
# CAVEAT: that table was measured on `_nio_workload`, which defaults to
# `chargeTransferCorrection=None`. Without a double counting the addition-GF poles sit ~14688 eV
# above E0 while the meshes span |z| <= 4.7, so `G` is *constant* on the frequencies it is evaluated
# at (relative variation 5.0e-08 across the whole Matsubara mesh). Its block counts mean nothing.
#
# Re-measured on the real workloads (the RSPt Hamiltonians in `impmod_tests/*/impurityModel_data.h5`,
# which have `max|Im G| = 19.9` in the window). NiO, 1 bath/orbital, 200 mesh points, reort=partial,
# blocks summed over all (block, side, eigenstate) units:
#
#     axis             band-wide      caller's mesh     sigma agreement
#     Matsubara only   3496 blocks    360 blocks        5.4e-13
#     real axis only   3496 blocks    3479 blocks       3.5e-11
#
# So ~10x on a Matsubara-only self-energy and ~nothing on the real axis -- the shape Phase 3a-bis
# predicted. On antiferromagnetic NiO (matrix-valued G, block widths to 4) the caller's-mesh monitor
# finishes all 240 units in 5.2 s; the band-wide one manages 45 of them in 300 s and is still going
# after two hours, all of it inside `_block_cf_inverse`, this monitor's own O(k^2) rebuild.
#
# The 1e-9 floor stands: it is strictly tighter than the old 1e-6, so nothing can silently degrade.
# See doc/plans/bicgstab_per_frequency_gf.md, Phases 3a-quater and 3a-quinquies.
#
# Note `_gf_rel_tol` takes max(slaterWeightMin**2, this), so this floor -- not the cutoff --
# governs every production slaterWeightMin (1e-5 gives 1e-10, far below it). Only a cutoff looser
# than sqrt(floor) ever overrides it, and then basis truncation is the accuracy limit anyway.
_GF_REL_TOL_FLOOR = 1e-9
# Minimum blocks before the convergence mesh may be frozen (let the extremal Ritz values start
# to settle before we commit to a sampling window).
_GF_MESH_FREEZE_BLOCKS = 3
# Mesh padding fraction AND the per-step edge-growth threshold below which the spectral edges
# (the alpha-diagonal range) count as "settled": we only freeze the mesh once a new block grows
# the range by less than the margin we pad with, and we re-extend the mesh if a later block's
# range escapes the padded window. One constant ties the two so they stay consistent.
_GF_MESH_MARGIN_REL = 0.05
# Consecutive sub-tolerance steps required before declaring convergence -- guards against a
# single accidental small relative change tripping convergence prematurely.
_GF_CONSEC_CONVERGED = 2
# Adaptive convergence-test sampling (see _make_gf_convergence_monitor). The resolvent-change
# test rebuilds the O(k)-level block continued fraction each call -- the single largest cost of
# the block-Lanczos Green's function (~53% of runtime for reort=NONE; measured). During the long
# approach (relative change still far above tolerance, convergence impossible) it is sampled only
# every _GF_CHECK_EVERY blocks; once a check lands within _GF_NEAR_FACTOR x tol it switches to
# every block so the exact convergence point is caught with no added Lanczos steps. The measure
# and tolerance are unchanged, so the converged Green's function is unchanged.
_GF_CHECK_EVERY = int(os.environ.get("GF_CHECK_EVERY", 8))  # set to 1 to disable (check every block)
# Switch from sparse to per-block sampling once the relative change is within this factor of the
# tolerance (i.e. convergence is imminent and must not be sampled coarsely). Kept small: the
# relative change typically sits on a long noisy plateau a decade or two above tolerance before
# the final descent, and that plateau must stay in the sparse regime for the sampling to pay off.
_GF_NEAR_FACTOR = float(os.environ.get("GF_NEAR_FACTOR", 2.0))
# Sample points per requested axis, per eigenstate, when the caller's evaluation mesh is known
# (see _gf_eval_meshes). The production real-axis mesh is ~2000 points and the Matsubara one ~375;
# feeding either to the monitor verbatim would multiply the continued-fraction cost above -- which
# is already the largest single cost of the recurrence -- by more than an order of magnitude. The
# monitor only has to decide *whether* G has stopped moving, so it subsamples. Matches the 64
# points _gf_sample_mesh uses for its spectral-edge fallback, so the cost is unchanged.
_GF_MONITOR_POINTS = 64
# --- Per-frequency BiCGSTAB Green's function (gf_method="bicgstab") -------------------------
# Residual tolerance of one per-frequency solve, relative to the seed norm (block_bicgstab's
# `atol` contract). 1e-8 measured to give |dG| ~ 7e-9 against a converged block-Lanczos
# reference (doc/plans/bicgstab_per_frequency_gf.md, Phase 3a) -- inside the 2.5e-8 spread
# PARTIAL-vs-FULL reorthogonalization itself shows on the real workloads. The reliability
# diagnostics (gf_diagnostics.check_bicgstab_convergence) derive their thresholds from the
# value actually used -- never re-hardcode it.
_GF_BICGSTAB_ATOL = float(os.environ.get("GF_BICGSTAB_ATOL", "1e-8"))
# Hard per-point iteration bound. Warm-started production solves measure ~3 iterations and a
# cold start ~6, so 500 is pathology headroom: a stagnating solve (a real-axis point within
# `delta` of a pole) ends and is *reported* by the diagnostics instead of iterating until the
# growing seen-support exhaustion bound -- which a solve that keeps discovering determinants
# may never reach.
_GF_BICGSTAB_MAX_ITER = int(os.environ.get("GF_BICGSTAB_MAX_ITER", "500"))
# Solutions retained for the warm start: quadratic extrapolation in z through the last three
# is the measured optimum (Phase 3a; cubic amplifies the atol-level noise it extrapolates
# through, and each retained block costs live memory).
_GF_BICGSTAB_WARM_HISTORY = 3
# Restarts of one per-point solve that ends unconverged (re-entering block_bicgstab with the
# current solution re-deflates the residual block and picks a fresh shadow residual r0_t --
# the standard cure for BiCGSTAB's r0-orthogonality stagnation, which real-axis points within
# ~delta of a pole do hit; the sparse path's basis-exhaustion bound also ends hard solves
# after ~N/width iterations and a restart grants the next round). Progress-gated below, so a
# genuinely stuck point stops early and is reported rather than looping.
_GF_BICGSTAB_RESTARTS = int(os.environ.get("GF_BICGSTAB_RESTARTS", "10"))
# A restart must shrink the reported residual by at least this factor to earn the next one.
_GF_BICGSTAB_RESTART_PROGRESS = 0.5
# --- Spectrum slicing (gf_method="sliced") --------------------------------------------------
# Chebyshev windows tiling the real-axis evaluation band (plus the rest-windows completing
# the partition of unity). The Phase-0 calibration (doc/plans/spectrum_slicing.md): filtered
# seeds' dominant amplitudes are energy-local, their sub-1e-6 tails are not -- so the memory
# lever is GF_SLICE_TOL (extra amplitude truncation of each filtered seed), traded explicitly
# against accuracy (discarded tail ~ sqrt(n_tail) * tol) and reported by the diagnostics.
_GF_SLICES = int(os.environ.get("GF_SLICES", "8"))
# Filter polynomial degree; 0 = auto (~8 * bandwidth / slice width, clipped to [200, 4000]).
_GF_SLICE_DEGREE = int(os.environ.get("GF_SLICE_DEGREE", "0"))
# Amplitude truncation of the filtered slice seeds; 0 = none (exactness-first default).
_GF_SLICE_TOL = float(os.environ.get("GF_SLICE_TOL", "0"))
# GMRES fallback for the points BiCGSTAB leaves unconverged (its shadow-residual
# recurrence stagnates within ~delta of a pole; GMRES minimizes the residual and has no
# such mode). The restart length bounds the fallback's live Krylov blocks -- the
# memory-model transient in estimate_gf_peak_bytes(method="bicgstab") must match it.
_GF_GMRES_RESTART = int(os.environ.get("GF_GMRES_RESTART", "40"))
_GF_GMRES_MAX_RESTARTS = int(os.environ.get("GF_GMRES_MAX_RESTARTS", "25"))


def _gf_signed_axes(matsubara_mesh, omega_mesh, side_i, delta):
    r"""The requested frequency axes in the resolvent frame, *before* the thermal-energy shift.

    One complex array per requested axis (Matsubara first): ``sign*mesh + i*sign*axis_delta``,
    where ``sign`` selects addition (+, ``side_i = 0``) vs removal (-, ``side_i = 1``) and the
    broadening applies to the real axis only (a Matsubara mesh already carries its imaginary
    part). The sign multiplies the mesh *and* the broadening so ``Im(z)`` keeps the sign of the
    unit's signed delta. Adding an eigenstate energy ``e`` to any returned axis gives exactly the
    ``omegaP`` frame :func:`calc_G` evaluates in -- the single source of that frame, shared by
    the convergence monitor (:func:`_gf_eval_meshes`, subsampled) and the per-frequency BiCGSTAB
    driver (full axes). Empty list when no mesh was requested.
    """
    sign = 1.0 if side_i == 0 else -1.0
    axes = []
    if matsubara_mesh is not None:
        axes.append((sign * np.asarray(matsubara_mesh)).astype(complex))
    if omega_mesh is not None:
        axes.append((sign * np.asarray(omega_mesh) + 1j * sign * delta).astype(complex))
    return axes


def _gf_eval_meshes(matsubara_mesh, omega_mesh, side_i, delta, es, n_points=_GF_MONITOR_POINTS):
    r"""The frequencies the caller will actually evaluate ``G`` at, in the ``alphas`` frame.

    ``calc_G`` forms :math:`\omega_P = \omega + i\delta + e`, and ``get_Greens_function`` calls it
    with ``(+mesh, +delta)`` for the addition side and ``(-mesh, -delta)`` for removal, with
    ``delta = 0`` on the Matsubara axis. Reproduce exactly that, for every thermal eigenstate the
    unit stacks.

    Returned as one array **per axis**, not one concatenated array, because the convergence measure
    is relative: :math:`\max|\Delta G| / \max|G|` over a mesh spanning both axes would divide the
    real-axis change by the Matsubara peak. At :math:`T = 0.002` and :math:`\delta = 0.2` those
    scales are :math:`1/\pi T \approx 159` and :math:`1/\delta = 5`, so the real axis would be
    declared converged more than an order of magnitude early. The monitor takes the max of the
    per-axis relative changes instead.

    Returns ``None`` when the caller asked for no mesh at all, which sends the monitor back to its
    spectral-edge fallback.
    """
    axes = _gf_signed_axes(matsubara_mesh, omega_mesh, side_i, delta)
    if not axes:
        return None

    per_axis = max(2, n_points // max(1, len(es)))

    meshes = []
    for axis_mesh in axes:
        # Subsampling by index commutes with the (already applied) affine sign/broadening map,
        # so this is exactly the old subsample-then-shift construction.
        sub = (
            axis_mesh
            if len(axis_mesh) <= per_axis
            else axis_mesh[np.linspace(0, len(axis_mesh) - 1, per_axis).astype(int)]
        )
        meshes.append(np.concatenate([sub + e for e in es]))
    return meshes


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
    (see ``doc/plans/calc_selfenergy_performance.md``). Override with ``GF_EIGENSTATE_GROUP``.
    """
    return max(1, int(os.environ.get("GF_EIGENSTATE_GROUP", 1)))


def _gf_operator_split():
    r"""Whether to use the pairwise / scalar operator-split decomposition (the *narrow* end of
    the block-width granularity spectrum). Default off (``GF_OPERATOR_SPLIT`` unset/0).

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
    return os.environ.get("GF_OPERATOR_SPLIT", "0") not in ("0", "", "false", "False")


def _gf_per_state_restrict(chain_restrict):
    r"""Whether to build the excited-sector occupation window *per thermal state* (per work unit)
    instead of once from the whole thermal ensemble.

    Default: **on exactly when ``chain_restrict`` is on**. Per-state windows differ from the
    ensemble window only through the state-dependent bath filled/empty classification, which is
    itself only produced under ``chain_restrict`` (and only for sites past the coupling-distance
    filter -- long chains); with ``chain_restrict`` off the two are identical, so per-state would
    be pure overhead. ``GF_PER_STATE_RESTRICT`` overrides the default either way (for tests/A-B).

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
    env = os.environ.get("GF_PER_STATE_RESTRICT")
    if env is None:
        return bool(chain_restrict)
    return env not in ("0", "", "false", "False")


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


class PairwiseGF:
    r"""Per-eigenstate Green's-function block assembled from scalar (width-1) continued fractions.

    Holds the scalar block-Lanczos coefficients for the operator-split decomposition of one
    ``n x n`` block (one thermal state, one spectral side): the ``n`` diagonal seeds and, per
    off-diagonal pair, the two polarization seeds. :func:`calc_G_pairwise` evaluates these on a
    frequency mesh and reassembles the full matrix via the polarization identity.

    Attributes
    ----------
    n : int
        Block dimension (number of transition operators).
    diag : list[tuple]
        Length-``n`` list of ``(alphas, betas, r)`` scalar continued fractions for ``v_i``.
    pairs : dict[tuple[int, int], tuple[tuple, tuple]]
        ``{(i, j): (cf_sum, cf_imag)}`` for ``i < j`` -- the scalar continued fractions for the
        seeds ``v_i + v_j`` and ``v_i + i v_j``.
    """

    __slots__ = ("n", "diag", "pairs")

    def __init__(self, n, diag, pairs):
        self.n = n
        self.diag = diag
        self.pairs = pairs


def calc_G_pairwise(pgf: "PairwiseGF", mesh, e, delta):
    r"""Assemble an ``n x n`` Green's-function block from its scalar continued fractions.

    Each scalar seed ``w`` gives the resolvent
    ``S(w) = w^\dagger (\omega + i\delta + e - H)^{-1} w`` via the width-1 continued fraction
    (:func:`calc_G`). The diagonal elements are ``G_ii = S(v_i)``; each off-diagonal pair is
    recovered from the polarization identity

    .. math::

        S(v_i + v_j)   &= M_{ii} + M_{jj} + M_{ij} + M_{ji}, \\
        S(v_i + i v_j) &= M_{ii} + M_{jj} + i M_{ij} - i M_{ji},

    so ``M_ij = ½[S(v_i+v_j) - i S(v_i+i v_j) - (1-i)(M_ii+M_jj)]`` and ``M_ji`` is its mirror.
    Exact (no approximation) given converged scalar continued fractions.
    """
    n = pgf.n
    G = np.zeros((len(mesh), n, n), dtype=complex)

    def S(cf):
        alphas, betas, r = cf
        return calc_G(alphas, betas, r, mesh, e, delta)[:, 0, 0]

    diag_S = [S(cf) for cf in pgf.diag]
    for i in range(n):
        G[:, i, i] = diag_S[i]
    for (i, j), (cf_sum, cf_imag) in pgf.pairs.items():
        Mii, Mjj = diag_S[i], diag_S[j]
        S_sum, S_imag = S(cf_sum), S(cf_imag)
        G[:, i, j] = 0.5 * (S_sum - 1j * S_imag - (1 - 1j) * (Mii + Mjj))
        G[:, j, i] = 0.5 * (S_sum + 1j * S_imag - (1 + 1j) * (Mii + Mjj))
    return G


def _gf_rel_tol(slaterWeightMin):
    """Relative-change convergence tolerance: the basis-truncation floor ``slaterWeightMin**2``
    but never below :data:`_GF_REL_TOL_FLOOR`. Used by both the runtime monitor and the
    diagnostic summary so they apply identical thresholds."""
    return max(slaterWeightMin**2, _GF_REL_TOL_FLOOR)


def _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes=None):
    r"""Relative-change convergence monitor for the block-Lanczos Green's function.

    Shared by both GF kernels (``block_green_impl``, ``block_Green_sparse``). Returns
    ``(converged_fn, converged_flag, delta_min)`` where ``delta_min`` is the convergence
    tolerance actually used (the single source of truth for the warning messages, so they
    never drift from this declaration): ``converged_fn(alphas, betas, verbose, block_widths)``
    estimates ``G`` and reports convergence only after :data:`_GF_CONSEC_CONVERGED` *consecutive*
    steps whose relative change (:func:`_greens_function_change`, with the cross-step ``gs_cache``)
    stays below ``max(slaterWeightMin**2, 1e-6)``.  Requiring the tolerance to hold for several
    steps in a row guards against a single fluke step.  ``converged_flag[0]`` records whether
    convergence was actually declared, for the non-convergence warning.

    ``eval_meshes`` (from :func:`_gf_eval_meshes`) is the list of frequency arrays the caller will
    actually evaluate ``G`` on -- one per requested axis, already shifted into the ``alphas`` frame.
    Given it, the monitor tests convergence *there*, and takes the **max** of the per-axis relative
    changes so neither axis can mask the other.

    Without it the monitor falls back to an *adaptively* frozen mesh spanning the resolved Ritz
    band on the line :math:`\omega + i\delta` (:func:`_gf_converged_mesh`) -- frozen once the
    spectral edges settle and re-extended if a later block escapes the window. That fallback
    converges the **real-axis** resolvent at broadening ``delta`` whether or not a real-axis mesh
    was requested, and a Matsubara point :math:`i\omega_n` sits a distance
    :math:`\sqrt{E_k^2 + \omega_n^2}` from every pole while a real-axis point can come within
    ``delta`` of one. So a Matsubara-only self-energy was being charged for a resolvent it never
    evaluates: measured 3.6-4.1x more blocks than it needs, against 1.2-1.4x when the real axis is
    also requested. It remains the right behaviour for a caller that supplies no mesh.
    """
    delta_min = _gf_rel_tol(slaterWeightMin)
    converged_flag = [False]
    mesh_cache = [None, -1]  # [mesh, frozen_block_count]; frozen_block_count detects (re)freezes
    gs_cache = [None, 0]
    consec = [0]  # consecutive sub-tolerance steps on the current (stable) mesh
    step = [0]  # block count since the mesh froze, for the adaptive sampling gate
    last_dg = [None]  # most recent relative change, to decide sparse vs dense sampling
    # One continued-fraction cache per axis: the caller's meshes never move, so the cross-step
    # reuse in _greens_function_change is always valid and never needs the freeze bookkeeping.
    axis_caches = [[None, 0] for _ in (eval_meshes or ())]

    def converged_on_eval_meshes(alphas, betas, verbose, block_widths):
        d_g = 0.0
        for mesh, cache in zip(eval_meshes, axis_caches):
            d = _greens_function_change(alphas, betas, block_widths, delta, omegaP=mesh, cache=cache)
            if d is None:  # spurious (wrong-sign) imaginary part on this axis -> not converged
                return None
            d_g = max(d_g, d)
        return d_g

    def converged(alphas, betas, verbose=False, block_widths=None, **kwargs):
        if len(alphas) <= 1:
            return False
        # B6 adaptive sampling. The resolvent-change test rebuilds an O(k)-level block continued
        # fraction every call, so running it on every block is O(k^2) per GF invocation and is the
        # single largest cost for reort=NONE (~53% measured). But the test also *terminates* the
        # recurrence, so simply running it less often delays convergence and adds (more expensive)
        # Lanczos steps that cancel the saving. Instead: during the long approach, where the change
        # is still far above tolerance and convergence is impossible, sample only every
        # _GF_CHECK_EVERY blocks; once a check lands within _GF_NEAR_FACTOR x tol, switch to every
        # block so the precise convergence point and the _GF_CONSEC_CONVERGED gate are detected with
        # no delay. Same convergence measure/tolerance -> same converged Green's function.
        step[0] += 1
        near = last_dg[0] is not None and last_dg[0] < _GF_NEAR_FACTOR * delta_min
        if not near and (step[0] % _GF_CHECK_EVERY) != 0:
            return False
        if eval_meshes is not None:
            # The caller's mesh is fixed from the start, so there is nothing to freeze or
            # re-extend: no spectral-edge warm-up, no cache resets.
            d_g = converged_on_eval_meshes(alphas, betas, verbose, block_widths)
        else:
            A_trim, _ = (
                _trim_blocks(alphas, betas, block_widths)
                if (block_widths is not None and len(block_widths) == len(alphas))
                else ([np.asarray(a) for a in alphas], None)
            )
            res = _gf_converged_mesh(A_trim, delta)
            if res is None:  # spectral edges have not settled yet -> keep building
                return False
            mesh, frozen = res
            if frozen != mesh_cache[1]:
                # First freeze or a re-extension changed the mesh: the cross-step resolvent cache
                # and the consecutive-step count are measured against the old window, so reset both
                # and re-confirm convergence on the new one.
                mesh_cache[0], mesh_cache[1] = mesh, frozen
                gs_cache[0], gs_cache[1] = None, 0
                consec[0] = 0
                last_dg[0] = None
            d_g = _greens_function_change(alphas, betas, block_widths, delta, omegaP=mesh_cache[0], cache=gs_cache)
        if d_g is None:  # spurious (wrong-sign) imaginary part -> not converged
            consec[0] = 0
            last_dg[0] = None
            return False
        last_dg[0] = d_g
        if verbose:
            print(rf"$\delta$ = {d_g}", flush=True)
        consec[0] = consec[0] + 1 if d_g < delta_min else 0
        is_conv = consec[0] >= _GF_CONSEC_CONVERGED
        converged_flag[0] = converged_flag[0] or is_conv
        return is_conv

    return converged, converged_flag, delta_min


def block_green_impl(basis, hOp, psi_arr, delta, reort, slaterWeightMin, verbose, eval_meshes=None):
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
    len(basis)
    n = len(psi_arr)

    comm = basis.comm
    rank = comm.rank if comm is not None else 0

    dense = len(basis) < 500
    if dense:
        psi_dense = build_vector(basis, psi_arr, slaterWeightMin=0).T
        psi_dense_local, r = build_qr(psi_dense)
    else:
        psi_dense = build_vector(basis, psi_arr, root=0, slaterWeightMin=0).T
        if rank == 0:
            psi_dense, r = build_qr(psi_dense)
        psi_dense_local, r = _scatter_qr_columns(
            comm, psi_dense if rank == 0 else None, r if rank == 0 else None, len(basis.local_basis)
        )

    if psi_dense_local.shape[1] == 0:
        return np.zeros((0, n, n), dtype=complex), np.zeros((0, n, n), dtype=complex), r, psi_arr, []

    converged, converged_flag, delta_min = _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes)

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
    # Keep alphas/betas padded (k, P, P) for the caller's elementwise cross-expansion diff;
    # only drop a corrupted trailing tail (whole blocks + widths) so it never reaches the
    # continued fraction. Norms of padded blocks equal those of the true blocks (zeros add
    # nothing), so the scan is valid on the padded arrays.
    keep = len(_sanitize_continued_fraction(list(alphas), list(betas), verbose=verbose, rank=rank)[0])
    if keep < len(alphas):
        alphas, betas, widths = alphas[:keep], betas[:keep], widths[:keep]
    q_last = Q_list[:, -1:]
    return alphas, betas, r, build_state(basis, q_last.T, slaterWeightMin=slaterWeightMin), widths


def calc_thermally_averaged_G(alphas, betas, r, mesh, es, e0, tau, delta):
    """
    Calculate the thermally averaged Green's function over multiple initial states.

    Parameters
    ----------
    alphas : list of list of ndarray
    betas : list of list of ndarray
    r : list of ndarray
    mesh : ndarray
    es : list of float
    e0 : float
    tau : float
    delta : float

    Returns
    -------
    G_avg : ndarray
    """
    # Operator-split (pairwise) path: r holds a per-eigenstate PairwiseGF; each carries its own
    # scalar continued fractions, so (alphas, betas) are unused and calc_G_pairwise assembles the
    # block from the polarization identity.
    if any(isinstance(r_e, PairwiseGF) for r_e in r):
        n_ops = next(r_e.n for r_e in r if isinstance(r_e, PairwiseGF))
        G_avg = np.zeros((len(mesh), n_ops, n_ops), dtype=complex)
        for e, r_e in zip(es, r):
            if r_e is None:
                continue
            G_avg += calc_G_pairwise(r_e, mesh, e, delta) * np.exp(-(e - e0) / tau)
        return G_avg

    if len(alphas) == 0:
        return np.zeros((len(mesh), 0, 0), dtype=complex)

    n_ops = r[0].shape[-1]
    G_avg = np.zeros((len(mesh), n_ops, n_ops), dtype=complex)

    for e, alphas_e, betas_e, r_e in zip(es, alphas, betas, r):
        G_avg += calc_G(alphas_e, betas_e, r_e, mesh, e, delta) * np.exp(-(e - e0) / tau)

    return G_avg


class _CappedBasisProxy:
    """Enforce ``truncation_threshold`` on the sparse-kernel GF recurrence.

    ``block_lanczos_cy``'s matvec discovers new Slater determinants every step, so the
    live block-state support (and, at reort != none, the Krylov store) grows without
    bound — the excited ``Basis`` itself stays frozen and never sees them. This proxy
    wraps that basis and caps the growth at the one point where every residual row
    sits on its hash-owner rank: the per-step ``redistribute_block`` call.

    Policy (freeze-growth + importance-ranked boundary admission):

    * while ``retained + n_new <= cap``: admit every newly discovered determinant;
    * on the single overflow step: rank that step's candidate rows by max column
      ``|amp|^2`` of the residual and admit the top ``cap - retained`` via a
      fixed-iteration distributed amplitude bisection (allreduce'd counts, so the
      cutoff is collective and deterministic), then freeze;
    * after the freeze: drop non-retained rows of every residual (rank-local
      ``keep_rows`` merge; ownership routing makes membership checks local).

    Why this is safe: every previously accepted Krylov block has support inside the
    retained set, so the diagonal projector ``P`` is invisible to inner products
    against them (``<Q_j, P wp> = <Q_j, wp>``) — orthogonality is untouched. From the
    freeze on, the recurrence is an *exact* block Lanczos of the Hermitian projected
    operator ``P H P``: the continued fraction stays causal, moments up to the freeze
    are exact w.r.t. ``H``, and the recurrence terminates as ``invariant_subspace``
    (already treated as exact-on-subspace). All reort modes remain valid, and the
    Krylov store's row set is bounded by the retained set (it never needs removal).

    MPI: one scalar allreduce per pre-freeze step; the freeze decision and the
    bisection derive only from allreduce'd data, so ranks cannot disagree, and every
    collective runs unconditionally (a rank may retain zero rows).

    The per-frequency BiCGSTAB driver (:func:`block_Green_bicgstab`) reuses this proxy
    unchanged in spirit: ``block_bicgstab``'s matvec routes through
    ``redistribute_block`` (also in serial runs, keyed on ``caps_growth``), so the same
    freeze-growth policy bounds a linear solve's live support, and post-freeze the solve
    is an exact BiCGSTAB of the projected operator ``P H P`` -- the same
    exact-on-retained-subspace contract as the capped Lanczos recurrence. The extra
    forwarders below (``add_states``, ``contains_local``, the restriction properties)
    are the attributes ``block_bicgstab`` reads off its basis.
    """

    caps_growth = True

    def __init__(self, basis, cap):
        self._basis = basis
        self.cap = int(cap)
        self.comm = basis.comm
        # Width-0 key-only mask of the retained determinants on this rank; grown by
        # in-place C++ sorted merges only (no per-row Python objects in the hot path).
        seed = ManyBodyState(dict.fromkeys(basis.local_basis, 1.0 + 0j))
        self._mask = ManyBodyBlockState.from_states([seed]).key_union(ManyBodyBlockState())
        self._global_count = int(basis.size)
        self._frozen = self._global_count >= self.cap
        self.cap_hit = self._frozen
        self._verbose_freeze_logged = False

    # --- attributes block_lanczos_cy reads off its basis ---------------------
    @property
    def local_basis(self):
        return self._basis.local_basis

    @property
    def size(self):
        return self._basis.size

    @property
    def n_bytes(self):
        return self._basis.n_bytes

    @property
    def is_distributed(self):
        return self._basis.is_distributed

    @property
    def restrictions(self):
        return self._basis.restrictions

    @property
    def weighted_restrictions(self):
        return self._basis.weighted_restrictions

    def redistribute_psis(self, psis):
        return self._basis.redistribute_psis(psis)

    def add_states(self, new_states, unique_sorted=False):
        # Growth bookkeeping only: every determinant block_bicgstab offers here came off a
        # redistribute_block-capped block, so it is already inside the retained mask and
        # counted by _global_count -- the wrapped basis can never outgrow the cap through
        # this path.
        return self._basis.add_states(new_states, unique_sorted=unique_sorted)

    def contains_local(self, state):
        return self._basis.contains_local(state)

    @property
    def retained_size(self):
        """Global number of determinants currently admitted to the recurrence."""
        return self._global_count

    def retained_keys(self):
        """Rank-local retained determinants as ``SlaterDeterminant`` wrappers (sorted).

        Builds one Python object per retained determinant — diagnostics/tests only,
        never the hot path."""
        keys, _ = self._mask.row_max_norms2()
        return keys

    def _allreduce_sum(self, value):
        if self.comm is None or self.comm.size == 1:
            return value
        return self.comm.allreduce(value, op=MPI.SUM)

    def redistribute_block(self, block):
        block = self._basis.redistribute_block(block)
        if self._frozen:
            block.keep_rows(self._mask)
            return block
        n_new = self._allreduce_sum(len(block) - block.count_rows_in(self._mask))
        if self._global_count + n_new <= self.cap:
            self._mask.merge_keys(block)
            self._global_count += n_new
            return block
        self._admit_top_and_freeze(block)
        block.keep_rows(self._mask)
        return block

    def _admit_top_and_freeze(self, block):
        """Admit the ``cap - retained`` most important candidate rows, then freeze.

        The amplitude-cutoff bisection runs a fixed iteration count on allreduce'd
        counts, so all ranks compute the identical cutoff. Ties at the cutoff are
        under-admitted (the cap is never exceeded); near-tie retained sets may differ
        across rank counts through summation-order rounding, like the CIPSI basis
        trajectory.
        """
        slots = self.cap - self._global_count
        norms2 = block.new_row_max_norms2(self._mask)
        cutoff2 = collective_amplitude_cutoff(norms2, slots, self.comm)
        admitted = block.keys_new_above(self._mask, cutoff2)
        self._global_count += self._allreduce_sum(len(admitted))
        self._mask.merge_keys(admitted)
        self._frozen = True
        self.cap_hit = True

    def freeze_message(self):
        """One-line description of the cap state (rank-0 logging)."""
        return (
            f"GF basis cap hit: froze the recurrence support at {self._global_count:,} "
            f"determinants (truncation_threshold={self.cap:,}); the Green's function is "
            f"exact on the retained subspace."
        )


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
    psi_dense = build_vector(basis, psi_arr, root=0, slaterWeightMin=slaterWeightMin).T
    if rank == 0:
        psi_dense, r = build_qr(psi_dense)
    if mpi:
        psi_dense_local, r = _scatter_qr_columns(
            comm, psi_dense if rank == 0 else None, r if rank == 0 else None, len(basis.local_basis)
        )
    else:
        psi_dense_local = psi_dense
    psi_arr = build_state(basis, psi_dense_local.T, slaterWeightMin=0)
    if len(psi_arr) == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), r

    converged, converged_flag, delta_min = _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes)

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
        :data:`_GF_BICGSTAB_ATOL`.
    max_iter : int, optional
        Per-point iteration bound; defaults to :data:`_GF_BICGSTAB_MAX_ITER`.
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
    atol = _GF_BICGSTAB_ATOL if atol is None else atol
    max_iter = _GF_BICGSTAB_MAX_ITER if max_iter is None else max_iter
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
                # (+ bra) support; redistribute_psis aligns the amplitudes to the fresh
                # ownership layout (the solver assumes its states are distributed per
                # `basis`).
                carried = seeds + x0 + (bras if bras is not None else [])
                tmp_basis.clear()
                tmp_basis.add_states(sorted({state for psi in carried for state in psi.keys()}))
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

                # Solve, restarting while unconverged and still making progress (each call
                # re-deflates Y - A x0 and picks a fresh shadow residual). Every rank sees the
                # identical info dict -- its fields derive from allreduce'd norms -- so the
                # restart loop is collective-consistent.
                info = {}
                iterations = 0
                X = x0
                prev_residual = np.inf
                for _attempt in range(1 + _GF_BICGSTAB_RESTARTS):
                    X = block_bicgstab(
                        A_op,
                        X,
                        seeds,
                        solve_basis,
                        slaterWeightMin,
                        atol=atol,
                        max_iter=max_iter,
                        info=info,
                    )
                    iterations += info["iterations"]
                    if info["converged"] or info["rel_residual"] > _GF_BICGSTAB_RESTART_PROGRESS * prev_residual:
                        break
                    prev_residual = info["rel_residual"]

                # GMRES fallback: warm-started from BiCGSTAB's partial iterate, before the
                # solution enters the extrapolation history -- so a rescued point also
                # repairs the warm-start chain its stagnated result would have poisoned.
                if not info["converged"]:
                    X = block_gmres(
                        A_op,
                        X,
                        seeds,
                        solve_basis,
                        slaterWeightMin,
                        atol=atol,
                        restart=_GF_GMRES_RESTART,
                        max_restarts=_GF_GMRES_MAX_RESTARTS,
                        info=info,
                    )
                    iterations += info["iterations"]
                    stats["gmres_points"] += 1
                    stats["gmres_iterations"] += info["iterations"]

                stats["n_points"] += 1
                stats["iterations"] += iterations
                stats["max_rel_residual"] = max(stats["max_rel_residual"], info["rel_residual"])
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


def _trim_blocks(alphas, betas, block_widths):
    r"""Strip the zero padding from block-Lanczos coefficients (shrinking blocks).

    The Lanczos kernels store every block into a fixed ``(P, P)`` pre-allocated
    buffer, zero-padding the inactive rows/columns whenever a block deflates
    (``block_widths[i] < P``).  This returns the true variable-dimension blocks:
    ``alphas[i] -> (w_i, w_i)`` and ``betas[i] -> (w_{i+1}, w_i)`` where
    ``w_i = block_widths[i]`` (the trailing ``betas[-1]`` residual block keeps its
    stored row count — it is the coupling beyond the subspace and is unused by the
    continued fraction).

    Args:
        alphas: Diagonal blocks, ``(k, P, P)`` ndarray (or length-``k`` sequence).
        betas: Off-diagonal blocks, same outer length.
        block_widths: True width ``w_i`` of every block.

    Returns:
        tuple[list, list]: ragged ``(alphas, betas)`` lists of 2D arrays.

    Raises:
        ValueError: if the width table and the coefficient arrays disagree in length.
            The kernels append a width for every stored block, so a mismatch means a
            caller trimmed one and not the other -- which would silently shorten the
            continued fraction (``k = len(widths)``) instead of failing.
    """
    widths = [int(w) for w in block_widths]
    k = len(widths)
    if k != len(alphas) or k != len(betas):
        raise ValueError(
            f"block_widths has {k} entries but alphas/betas have {len(alphas)}/{len(betas)}; "
            "the continued fraction would silently use only the first "
            f"{min(k, len(alphas))} block(s)."
        )
    a = [np.asarray(alphas[i])[: widths[i], : widths[i]] for i in range(k)]
    b = []
    for i in range(k):
        rows = widths[i + 1] if i + 1 < k else np.asarray(betas[i]).shape[0]
        b.append(np.asarray(betas[i])[:rows, : widths[i]])
    return a, b


def _sanitize_continued_fraction(alphas, betas, verbose=False, rank=0):
    r"""Drop a corrupted trailing tail from the block-Lanczos coefficients.

    Defense-in-depth before the continued fraction / self-energy: the Lanczos kernels now
    truncate a diverging recurrence at the source (CholeskyQR2 + the ``BETA_BLOWUP_FACTOR``
    guard), but should a non-finite or runaway block ever reach here it must *not* be fed
    silently into :func:`calc_G` and ``sig_static``.  Scans the (trimmed) blocks and keeps
    only the leading run whose norms stay bounded relative to the healthy part; the trailing
    ``beta`` of the kept run is the (ignored) residual coupling, so dropping the tail is
    consistent with the continued fraction's own convention.

    Returns the (possibly shortened) ``(alphas, betas)`` and warns when a tail is dropped.
    """
    norm_max = 0.0
    keep = len(alphas)
    for i in range(len(alphas)):
        a = np.asarray(alphas[i])
        b = np.asarray(betas[i])
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            keep = i
            break
        a_norm = float(np.linalg.norm(a, 2)) if a.size else 0.0
        b_norm = float(np.linalg.norm(b, 2)) if b.size else 0.0
        if i > 0 and max(a_norm, b_norm) > BETA_BLOWUP_FACTOR * max(norm_max, 1.0):
            keep = i
            break
        norm_max = max(norm_max, a_norm, b_norm)
    if keep < len(alphas):
        if verbose and rank == 0:
            print(
                f"warning: discarding {len(alphas) - keep} corrupted block(s) from the "
                f"Green's-function continued fraction before computing the self-energy.",
                flush=True,
            )
        return alphas[:keep], betas[:keep]
    return alphas, betas


def _block_cf_inverse(alphas, betas, omegaP):
    r"""Level-0 inverse resolvent of a block-tridiagonal :math:`T` by continued fraction.

    Builds, for every frequency ``omegaP`` (already shifted, i.e.
    :math:`\omega + i\delta + e`),

    .. math::

        G^{-1}_0(\omega) = \omega I - \alpha_0
            - \beta_0^\dagger \big(\omega I - \alpha_1 - \cdots\big)^{-1} \beta_0 ,

    where ``alphas[i]`` is the ``(n_i, n_i)`` diagonal block and ``betas[i]`` the
    ``(n_{i+1}, n_i)`` sub-diagonal block coupling block ``i`` to ``i+1``.  Block
    dimensions may vary from level to level (shrinking-block deflation) and the
    ``betas`` may be rectangular; the identity at each level is sized from that
    level's diagonal block, so no fixed block dimension is assumed.  The trailing
    ``betas[-1]`` (residual coupling beyond the retained subspace) is ignored.

    Args:
        alphas: Length-``k`` sequence of square diagonal blocks.
        betas: Length-``k`` sequence of sub-diagonal blocks.
        omegaP: ``(n_w,)`` complex frequency mesh (shift already applied).

    Returns:
        numpy.ndarray: ``(n_w, n_0, n_0)`` inverse resolvent at the first block.
    """
    nw = omegaP.shape[0]

    def wI(n):
        return omegaP[:, np.newaxis, np.newaxis] * np.identity(n, dtype=complex)[np.newaxis]

    a_last = np.asarray(alphas[-1])
    G_inv = wI(a_last.shape[0]) - a_last[np.newaxis]
    for alpha, beta in zip(alphas[-2::-1], betas[-2::-1]):
        alpha = np.asarray(alpha)
        beta = np.asarray(beta)
        n_i = alpha.shape[0]
        beta_b = np.broadcast_to(beta, (nw,) + beta.shape)
        G_inv = wI(n_i) - alpha[np.newaxis] - np.conj(beta.T)[np.newaxis] @ np.linalg.solve(G_inv, beta_b)
    return G_inv


def calc_G(alphas, betas, r, omega, e, delta):
    r"""Green's function from block-Lanczos continued-fraction coefficients.

    Computes :math:`G(\omega) = r^\dagger (\omega + i\delta + e - T)^{-1} r` where
    ``T`` is the block-tridiagonal matrix with diagonal blocks ``alphas`` and
    sub-diagonal blocks ``betas``.  ``alphas`` / ``betas`` may be either a uniform
    ``(k, p, p)`` ndarray (no deflation) or ragged sequences of variable-dimension
    2D blocks (after :func:`_trim_blocks`); rectangular ``betas`` from shrinking-block
    deflation are handled — no fixed block dimension is assumed.

    Parameters
    ----------
    alphas : ndarray or sequence of ndarray
        Diagonal continued-fraction blocks.
    betas : ndarray or sequence of ndarray
        Off-diagonal continued-fraction blocks (``betas[i]`` couples block ``i`` to
        ``i+1`` with shape ``(n_{i+1}, n_i)``).
    r : ndarray
        ``(n_0, n_ops)`` projection of the seed block onto the first Lanczos block.
    omega : ndarray
        Frequency mesh.
    e : float
        Energy offset.
    delta : float
        Broadening factor.

    Returns
    -------
    G : ndarray
        ``(len(omega), n_ops, n_ops)`` Green's function.
    """
    r = np.asarray(r)
    if len(alphas) == 0:
        n_ops = r.shape[-1]
        return np.zeros((len(omega), n_ops, n_ops), dtype=complex)
    omegaP = np.asarray(omega) + 1j * delta + e
    G_inv = _block_cf_inverse(alphas, betas, omegaP)
    r_b = np.broadcast_to(r, (omegaP.shape[0],) + r.shape)
    return np.conj(r.T)[np.newaxis] @ np.linalg.solve(G_inv, r_b)


def _greens_function_change(alphas, betas, block_widths, delta, omegaP=None, cache=None):
    r"""Relative change in the block resolvent when the last Lanczos block is added.

    Block-Lanczos convergence monitor for the Green's function.  It compares the
    seed-block resolvent :math:`G = (G^{-1}_0)^{-1}` built from all ``k`` blocks against
    the one from the first ``k-1`` blocks, on sample frequencies drawn from the (trimmed)
    diagonal blocks (the broadened Ritz values — where the spectral weight sits).  The
    measure is the *relative* change

    .. math::

        d_g = \frac{\max_\omega \lVert G_k(\omega) - G_{k-1}(\omega)\rVert}
                   {\max_\omega \lVert G_k(\omega)\rVert},

    so it is scale-invariant and reflects the spectral function the self-energy actually
    needs — unlike the absolute change in :math:`G^{-1}_0`, whose leading
    :math:`\omega I - \alpha_0` term (:math:`\sim\lvert\omega\rvert`) is identical between
    the two and which therefore never decays to a tight absolute tolerance even when the
    spectrum is fully resolved.  Shrinking-block deflation is handled by trimming to
    ``block_widths`` first, so no fixed block dimension is assumed.

    The optional ``cache`` (a ``[gs_value, n_blocks]`` list) lets the convergence loop reuse
    work: this step's ``G^{-1}`` over ``k-1`` blocks is *identical* to the previous step's
    over ``k-1`` blocks on the same frozen mesh, so the ``gs_prev`` continued fraction is taken
    from the cache instead of rebuilt — halving the per-step continued-fraction cost. Exact
    (no behavior change); pass the same list each step.

    Returns:
        float or None: the relative change, or ``None`` if the freshly added block yields
        a wrong-sign spectral weight (not yet stabilized).
    """
    if block_widths is not None and len(block_widths) == len(alphas):
        A, B = _trim_blocks(alphas, betas, block_widths)
    else:
        A = [np.asarray(alphas[i]) for i in range(len(alphas))]
        B = [np.asarray(betas[i]) for i in range(len(betas))]
    if omegaP is None:
        # Default (back-compat / standalone use): sample at the current Ritz values. The
        # convergence loop should instead pass a *frozen* mesh (see _gf_sample_mesh): the
        # Ritz set grows every step, so each new block adds a pole at a fresh sample point
        # and the change never decays — measuring on a fixed mesh is what converges.
        n0 = A[0].shape[0]
        ws = np.concatenate([np.diagonal(a) for a in A])[: 15 * n0]
        omegaP = ws.real + delta * 1j
    gs_new = _block_cf_inverse(A, B, omegaP)
    if cache is not None and cache[0] is not None and cache[1] == len(A) - 1:
        gs_prev = cache[0]  # == previous step's gs_new (CF over the same k-1 blocks/mesh)
    else:
        gs_prev = _block_cf_inverse(A[:-1], B[:-1], omegaP)
    if cache is not None:
        cache[0], cache[1] = gs_new, len(A)
    if np.any(np.diagonal(gs_new.imag, axis1=1, axis2=2) * np.sign(delta) < 0):
        return None
    # Compare the resolvents G = (G^{-1})^{-1}, not their inverses: the broadening
    # (Im omega = delta) keeps G^{-1} non-singular, so the inverse is well defined.
    G_new = np.linalg.inv(gs_new)
    G_prev = np.linalg.inv(gs_prev)
    scale = np.max(np.abs(G_new))
    return np.max(np.abs(G_new - G_prev)) / max(scale, np.finfo(float).tiny)


def _lanczos_convergence_summary(alphas_list, betas_list, delta, tol=_GF_REL_TOL_FLOOR):
    r"""Post-hoc block-Lanczos convergence summary over the per-thermal-state coefficients.

    Avoids threading run-time monitor state out of ``block_Green_sparse``: for each thermal
    state's trimmed ``(alphas, betas)`` it re-evaluates the final relative resolvent change via
    :func:`_greens_function_change`.  To agree with the run-time monitor's verdict (and not
    raise spurious "not converged" warnings) it mirrors that monitor exactly:

    * **Same frozen mesh.**  The monitor freezes its sample mesh after the first
      ``_GF_MESH_FREEZE_BLOCKS`` blocks; here we rebuild it from those same leading blocks
      (``A[:_GF_MESH_FREEZE_BLOCKS]``), not from the full final Ritz range -- otherwise the
      last block's spectral-edge contribution registers as a large change on a wider mesh the
      monitor never used.
    * **Invariant subspace == exact.**  When the recurrence terminated on an invariant subspace
      the trailing coupling block vanished (``betas[-1] ~ 0``); the continued fraction is then
      exact regardless of the relative-change value, so that state counts as converged.  (A run
      that stopped on the tolerance instead has a normal, nonzero trailing ``beta``.)

    Args:
        alphas_list, betas_list: Per-thermal-state lists of trimmed Lanczos blocks.
        delta: Broadening used to place the frozen sample mesh off the real axis.
        tol: Relative-change threshold below which a state counts as converged. Defaults to
            :data:`_GF_REL_TOL_FLOOR`; pass :func:`_gf_rel_tol` (slaterWeightMin) to match the
            monitor's basis-truncation floor.

    Returns:
        tuple[bool, float, int]: ``(all_converged, worst_final_change, max_blocks)``.
    """
    worst = 0.0
    max_blocks = 0
    all_converged = True
    for A, B in zip(alphas_list, betas_list):
        A = list(A)
        B = list(B)
        max_blocks = max(max_blocks, len(A))
        if len(A) < _GF_MESH_FREEZE_BLOCKS:  # invariant subspace reached almost immediately -> exact
            continue
        # Invariant subspace: trailing coupling vanished -> exact (matches the kernel's
        # "invariant_subspace" status, which the run-time monitor treats as converged).
        tail = np.asarray(B[-1]) if len(B) else np.zeros(0)
        scale = max((float(np.linalg.norm(np.asarray(a), 2)) for a in A), default=1.0)
        if tail.size == 0 or float(np.linalg.norm(tail, 2)) <= _GF_REL_TOL_FLOOR * max(scale, 1.0):
            continue
        # Use the *same* adaptively-frozen mesh the run-time monitor settled on (pure function of
        # the coefficients, so the two agree). If the edges never settled the run is genuinely
        # under-resolved.
        res = _gf_converged_mesh(A, delta if delta else 1.0)
        if res is None:
            all_converged = False
            continue
        mesh, _ = res
        d_g = _greens_function_change(A, B, None, delta if delta else 1.0, omegaP=mesh)
        if d_g is None:
            all_converged = False
            continue
        worst = max(worst, float(d_g))
        if d_g >= tol:
            all_converged = False
    return all_converged, worst, max_blocks


def _gf_sample_mesh(alphas, delta, n_points=64):
    r"""Frozen real-frequency mesh for the block-Lanczos Green's-function convergence test.

    Spans the current Ritz range (the diagonal entries of the ``alphas`` blocks — Lanczos
    resolves the spectral *edges* within a few blocks) padded by a margin, on the line
    :math:`\omega + i\,\mathrm{sign}(\delta)\,\lvert\delta\rvert`.  The caller builds this
    once and reuses it, so the convergence measure is evaluated at *fixed* frequencies and
    actually decays as the spectrum fills in.
    """
    ws = np.concatenate([np.real(np.diagonal(np.asarray(a))) for a in alphas])
    mesh, _, _ = _gf_mesh_from_range(float(np.min(ws)), float(np.max(ws)), delta, n_points)
    return mesh


def _gf_mesh_from_range(lo, hi, delta, n_points=64):
    r"""Build the convergence sample mesh covering ``[lo, hi]`` padded by the margin, on the line
    :math:`\omega + i\,\delta`.  Returns ``(mesh, ext_lo, ext_hi)`` where ``[ext_lo, ext_hi]`` is
    the padded extent actually sampled (used by :func:`_gf_converged_mesh` to detect a later
    block whose spectral edge escapes the window)."""
    span = hi - lo
    margin = _GF_MESH_MARGIN_REL * span + 10.0 * abs(delta)
    ext_lo, ext_hi = lo - margin, hi + margin
    return np.linspace(ext_lo, ext_hi, n_points) + 1j * delta, ext_lo, ext_hi


def _gf_diag_range(alpha):
    """Min/max of the real alpha-block diagonal (the per-direction Rayleigh quotients ~ the
    spectral edges resolved so far)."""
    d = np.real(np.diagonal(np.asarray(alpha)))
    return float(np.min(d)), float(np.max(d))


def _gf_converged_mesh(alphas, delta):
    r"""Adaptively-frozen convergence mesh, as a *pure* function of the (trimmed) alpha blocks so
    the runtime monitor and the post-hoc summary settle on exactly the same window.

    Block-Lanczos resolves the spectral edges from the inside out, so a mesh frozen after a fixed
    handful of blocks can be too narrow.  Instead we scan the blocks, tracking the cumulative
    alpha-diagonal range ``[lo, hi]`` (the resolved edges), and:

    * **Freeze** once at least :data:`_GF_MESH_FREEZE_BLOCKS` blocks are in *and* the range grew
      by less than :data:`_GF_MESH_MARGIN_REL` in the last block (edges settled within the margin
      we pad with).
    * **Re-extend** afterwards if a later block's range escapes the padded window, rebuilding the
      mesh around the new, wider range.

    Returns ``(mesh, frozen_blocks)`` -- ``frozen_blocks`` is the block count at the last (re)freeze,
    so a caller can detect when the mesh changed -- or ``None`` if the edges have not settled yet.
    """
    if len(alphas) < _GF_MESH_FREEZE_BLOCKS:
        return None
    lo = hi = None
    prev_span = None
    mesh = None
    ext_lo = ext_hi = None
    frozen = 0
    for k, alpha in enumerate(alphas):
        cur_lo, cur_hi = _gf_diag_range(alpha)
        lo = cur_lo if lo is None else min(lo, cur_lo)
        hi = cur_hi if hi is None else max(hi, cur_hi)
        span = hi - lo
        if mesh is None:
            if k + 1 >= _GF_MESH_FREEZE_BLOCKS and prev_span is not None:
                growth = (span - prev_span) / max(span, np.finfo(float).tiny)
                if growth < _GF_MESH_MARGIN_REL:
                    mesh, ext_lo, ext_hi = _gf_mesh_from_range(lo, hi, delta)
                    frozen = k + 1
            prev_span = span
        elif lo < ext_lo or hi > ext_hi:  # a later edge escaped the sampled window -> re-extend
            mesh, ext_lo, ext_hi = _gf_mesh_from_range(lo, hi, delta)
            frozen = k + 1
    if mesh is None:
        return None
    return mesh, frozen


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
