r"""Green's-function distribution engine: partition GF work into units and run them MPI-parallel.

This is the *distribution* half of the Green's-function machinery. It enumerates the
independent GF "units" a spectrum needs (:func:`enumerate_gf_units`, :class:`GFUnit`),
estimates their relative cost (:func:`unit_cost_weights`), and drives them across a
color-split communicator with per-unit basis rebuild + seed redistribution
(:func:`run_units_distributed`). The per-unit resolvent kernels live in
:mod:`impurityModel.ed.gf_solvers`; the top-level drivers that call this engine live in
:mod:`impurityModel.ed.greens_function`.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.basis_split import split_basis_and_redistribute_psi
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.memory_estimate import estimate_gf_peak_bytes, format_bytes, max_colors_within_budget
from impurityModel.ed.mpi_comm import gather_distributed_results

comm = MPI.COMM_WORLD
rank = comm.rank


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
) -> list | bool | None:
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

    assert split_seeds is not None  # seeds passed in are a (possibly empty) list, never None
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


def _apply_transition_ops(tOps, psis, excited_restrictions, excited_weighted_restrictions, slaterWeightMin):
    """Apply each transition operator to every thermal state, returning the seed blocks.

    Returns ``block_v`` indexed ``[j_psi][i_tOp]`` -- the excited state ``tOps[i] |psi_j>`` confined
    to the excited sector. These are the columns of each eigenstate's block-Lanczos seed.
    """
    # The thermal states share their support, so each transition operator is applied to
    # the whole block at once (term/sign/accumulator work once per determinant, near-flat
    # in the number of eigenstates — Phase 2 block-state matvec).
    psi_blk = ManyBodyState.from_states(list(psis))
    block_v = [[ManyBodyState({}) for _ in tOps] for _ in psis]
    for i_tOp, tOp in enumerate(tOps):
        tOp.set_restrictions(excited_restrictions)
        tOp.set_weighted_restrictions(excited_weighted_restrictions)
        res_psis = tOp.apply_block(psi_blk, slaterWeightMin).to_states()
        for j_psi, res_psi in enumerate(res_psis):
            block_v[j_psi][i_tOp] += res_psi
    return block_v


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
