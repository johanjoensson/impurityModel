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
from impurityModel.ed.basis_restrictions import union_windows
from impurityModel.ed.basis_split import _pack_units, split_basis_and_redistribute_psi
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.memory_estimate import (
    DEFAULT_MEMORY_SAFETY,
    _resident_adjusted_budget,
    available_bytes_per_rank,
    current_rss_bytes,
    estimate_gf_peak_bytes,
    format_bytes,
    max_unit_dets_within_budget,
    release_freed_heap,
)
from impurityModel.ed.mpi_comm import gather_distributed_results

comm = MPI.COMM_WORLD
rank = comm.rank


def gf_cap(basis):
    """The determinant cap a Green's-function unit basis built from ``basis`` runs under.

    When the ground-state solve attached its :class:`~impurityModel.ed.memory_estimate.CapPolicy`
    (``basis.cap_policy``, set by ``find_ground_state_basis``), that policy's ``gf`` cap -- the
    cap the driver resolved, *not* whatever ``basis.truncation_threshold`` the ground-state memory
    guard left behind. The guard may hold one ground-state expansion below its cap; that says
    nothing about what a Green's-function unit can afford, and inheriting it is how a 120-determinant
    ground state once froze every SrMnO3 GF unit at 120 determinants and made Sigma acausal.
    Without a policy (a basis built directly), the basis's own cap, as before.
    """
    policy = getattr(basis, "cap_policy", None)
    if policy is not None:
        # `gf is None` is an auto GF cap, sized at GF entry (`_auto_gf_cap`); until then, none.
        return np.inf if policy.gf is None else policy.gf
    return getattr(basis, "truncation_threshold", np.inf)


def _describe_gf_cap(basis, cap, layout):
    """The ``determinant cap:`` line for a GF stage, or ``None`` for a basis without a policy."""
    policy = getattr(basis, "cap_policy", None)
    if policy is None:
        return None
    if not np.isfinite(cap):
        how = "no cap; the GF memory guard may hold a unit lower"
    elif not policy.from_memory:
        how = "set by you; final"
    else:
        how = "auto, sized for the Green's-function path; the GF memory guard may hold a unit lower"
    size = "unlimited" if not np.isfinite(cap) else f"{int(cap):,}"
    return f"determinant cap: GF {size} per unit ({how}); {layout}"


def _colors_affording(basis, unit_weights, width, reort, gf_method, floor=None):
    """How many colors a GF stage may run so that every unit can afford ``floor`` determinants.

    ``floor`` is the unit cap for a cap the user set (or a basis without a policy); for an auto
    cap it defaults as described below.

    An auto GF cap is the largest unit basis the *smallest* color affords, so concurrency trades
    directly against it. The floor is the ground-state basis's actual size -- a unit's seeds are
    the union of ``c^dagger psi`` over its block, at least the ground state's support, so a GF cap
    below it would freeze units at their seeds -- or, once this kernel's cap is pinned for the
    calculation, that pinned cap. Not the ground state's *cap*: at 256 ranks an auto GS cap of
    tens of millions against a ground state of ~1M determinants would force two or three colors
    where the support needs no such restriction. The largest color count whose *real* packing
    (``_pack_units``, the same call the split makes) affords the floor on every color wins;
    ``None`` leaves the packing alone.

    Collective on ``basis.comm`` (memory probe, resident MAX); every input is replicated. The
    resident set is sampled before the split, while the cap is sized after it (the split
    replicates the ground-state basis into each color), so the floor is a target, not a guarantee.
    """
    if floor is None:
        pinned = getattr(basis, "_auto_gf_caps", {}).get(_pin_key(reort, gf_method))
        floor = pinned[0] if pinned is not None else int(basis.size)
    candidates = min(basis.comm.size, len(unit_weights))
    if candidates <= 1:
        return None
    resident = basis.comm.allreduce(current_rss_bytes(), op=MPI.MAX)
    affordable = {}

    def afford(ranks):
        if ranks not in affordable:
            affordable[ranks] = max_unit_dets_within_budget(
                basis.num_spin_orbitals,
                width,
                reort,
                ranks,
                basis.comm,
                safety=_gf_sizing_safety(),
                method=gf_method,
                resident_bytes=resident,
            )
        return affordable[ranks]

    for n_colors in range(candidates, 1, -1):
        _subgroups, procs = _pack_units(unit_weights, basis.comm.size, basis.split_threshold, n_colors)
        if procs is None:
            return 1
        if min(afford(int(p)) for p in procs) >= floor:
            return n_colors
    return 1


def _is_auto_gf(basis):
    """Whether the GF cap is auto: sized at GF entry from the GF path's own memory, not the GS cap."""
    policy = getattr(basis, "cap_policy", None)
    return policy is not None and policy.from_memory and policy.gf is None


def _pin_key(reort, gf_method):
    """What an auto GF cap is pinned per: the kernel, whose per-determinant cost it was sized for."""
    return (str(getattr(reort, "name", reort)), str(gf_method))


def _pinned_auto_gf_cap(basis, rank_counts, width, reort, gf_method, resident_bytes):
    """The calculation's auto GF cap for this kernel: sized by its first GF stage, reused after.

    A spectra or RIXS calculation runs several GF stages on one ground-state basis (IPS, PS, XAS,
    each adaptive incoming-energy round), and the resident set grows between them as caches are
    kept -- sizing each stage afresh would run parts of one spectrum at different caps. So the
    first stage of each kernel (reort, method) pins its number on the ground-state basis (which
    lives exactly as long as the calculation), and later stages of that kernel reuse it. A later
    stage with wider units may not afford the pinned number even on one color spanning every
    rank; it is clamped to what that affords rather than trusted to a guard it may not have.
    Replicated: the sizing is collective and identical on every rank.
    """
    pins = getattr(basis, "_auto_gf_caps", None)
    if pins is None:
        pins = {}
        basis._auto_gf_caps = pins
    key = _pin_key(reort, gf_method)
    if key not in pins:
        pins[key] = (_auto_gf_cap(basis, rank_counts, width, reort, gf_method, resident_bytes), resident_bytes)
    cap, pinned_resident = pins[key]
    # The clamp is for width, not for memory drift: it is evaluated against the resident set the
    # pin was taken with, so a grown resident set does not move the calculation's cap.
    all_ranks = basis.comm.size if basis.comm is not None else 1
    return min(cap, _auto_gf_cap(basis, [all_ranks], width, reort, gf_method, pinned_resident))


def _auto_gf_cap(basis, rank_counts, width, reort, gf_method, resident_bytes):
    """The largest unit basis the smallest of ``rank_counts`` can afford, with what is resident.

    Applied to every unit of a stage: equal treatment of equivalent blocks does not depend on
    which color a unit landed on, and the ground-state cap -- sized for a different path -- plays
    no part. Collective on ``basis.comm`` (the memory probe); ``rank_counts`` is replicated.
    """
    return min(
        max_unit_dets_within_budget(
            basis.num_spin_orbitals,
            width,
            reort,
            ranks,
            basis.comm,
            safety=_gf_sizing_safety(),
            method=gf_method,
            resident_bytes=resident_bytes,
        )
        for ranks in sorted(set(int(r) for r in rank_counts))
    )


def _guard_safety():
    """``GS_MEMORY_BUDGET_SAFETY`` (default :data:`DEFAULT_MEMORY_SAFETY`); ``<= 0`` disables."""
    safety = config.GS_MEMORY_BUDGET_SAFETY.get()
    return DEFAULT_MEMORY_SAFETY if safety is None else safety


def _gf_sizing_safety():
    """The safety fraction an auto GF cap is sized with: the guard's, so the cap and the guard
    that polices it agree; the default when the guard is switched off."""
    safety = _guard_safety()
    return safety if safety > 0.0 else DEFAULT_MEMORY_SAFETY


def _gf_memory_budget(available, resident):
    """Absolute per-rank RSS budget for the GF units' measured guard, or ``None`` when disabled.

    ``resident + _resident_adjusted_budget(safety, available, resident)``: the very headroom the
    auto GF cap was sized against (floor included), on top of what is resident at GF entry. A
    guard budget of ``safety * (available + resident)`` instead has no floor, so after a heavy
    ground state (``resident >= available`` at the default safety) it would sit at or below the
    resident set and freeze every auto unit at its seeds -- the round-9 failure, on the GF side.
    Pure; the caller samples ``available``/``resident`` collectively.
    """
    safety = _guard_safety()
    if safety <= 0.0:
        return None
    return int(resident + _resident_adjusted_budget(safety, available, resident))


def _set_gf_memory_guard(target, basis, budget):
    """Put the guard's budget and policy on ``target`` for the kernels; returns what to restore."""
    saved = (getattr(target, "gf_memory_budget", None), getattr(target, "gf_memory_policy", None))
    target.gf_memory_budget = budget
    target.gf_memory_policy = "tighten" if _may_lower_gf_cap(basis) else "warn"
    return saved


def _restore_gf_memory_guard(target, saved):
    target.gf_memory_budget, target.gf_memory_policy = saved


def _may_lower_gf_cap(basis):
    """Whether memory may size a GF unit below :func:`gf_cap`: not for a cap the user set."""
    policy = getattr(basis, "cap_policy", None)
    return policy is None or policy.from_memory


@dataclass(frozen=True)
class GFUnit:
    """One distributable Green's-function work unit: a (possibly wide) block-Lanczos recurrence.

    A unit stacks the transition-operator seeds of ``chunk`` thermal eigenstates from one
    operator group into a single recurrence of width ``len(chunk) * n_ops``. Units are the atoms of the MPI
    distribution: :func:`run_units_distributed` never splits one across colors.

    Attributes
    ----------
    group_i : int
        Index into the caller's operator-group list (e.g. block x spectral side, or a
        transition-operator index).
    chunk : tuple of int
        Thermal-eigenstate indices whose seeds this unit stacks.
    n_ops : int
        Seed columns per eigenstate.
    delta : float
        Signed broadening of this unit's recurrence (sign selects addition/removal).
    """

    group_i: int
    chunk: tuple[int, ...]
    n_ops: int
    delta: float


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
) -> tuple[list[GFUnit], list[list[ManyBodyState]], list]:
    """Enumerate the flat work units of a Green's-function calculation.

    Applies each operator group's transition operators to every thermal state (collective on
    the full basis), then chunks the eigenstates into groups of ``GF_EIGENSTATE_GROUP`` -- each
    chunk seeds one (possibly wide) block-Lanczos recurrence and is one work unit. This is the
    single global decomposition that is load-balanced across the full
    (operator group x eigenstate) cross-product -- important when there are many small symmetry
    blocks (the typical production case).

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
        (:func:`basis_restrictions.union_windows`) over the eigenstates it stacks instead of the group
        fallback.

    Returns
    -------
    tuple
        ``(units, unit_seeds, unit_restrictions)`` -- the :class:`GFUnit` metadata, the flat
        seed-column list per unit in (eigenstate, operator) order, and the excited window per
        unit.
    """
    group = _gf_eigenstate_group()
    n_psis = len(psis)
    units: list[GFUnit] = []
    unit_seeds: list[list[ManyBodyState]] = []
    for g, (tOps, delta_signed) in enumerate(op_groups):
        block_v = _apply_transition_ops(tOps, psis, group_restrictions[g], weighted_restrictions, slaterWeightMin)
        n_ops = len(tOps)
        for chunk_start in range(0, n_psis, group):
            chunk = tuple(range(chunk_start, min(chunk_start + group, n_psis)))
            units.append(GFUnit(g, chunk, n_ops, delta_signed))
            unit_seeds.append([block_v[j][i] for j in chunk for i in range(n_ops)])

    # Per-unit excited window: the union of the per-state windows over the eigenstates the unit
    # stacks (exactly that state's window for a single-state unit). Falls back
    # to the group window when per-state restrictions are disabled or state-independent.
    if per_state_restrictions is not None:
        unit_restrictions = [union_windows([per_state_restrictions[ei] for ei in u.chunk]) for u in units]
        # The seeds above were cut by the group window, but the unit's recurrence runs under its
        # own (per-state) window. A seed row outside the recurrence window sees P H, which has no
        # diagonal and a one-way coupling there -- the Lanczos operator is no longer Hermitian on
        # the seed (review ledger C5). Re-cut each such unit's seeds by the window it runs under.
        # Rank-local (the apply is local), so no collective is added.
        for u, unit in enumerate(units):
            if unit_restrictions[u] == group_restrictions[unit.group_i]:
                continue
            tOps, _delta = op_groups[unit.group_i]
            block_v = _apply_transition_ops(
                tOps, [psis[ei] for ei in unit.chunk], unit_restrictions[u], weighted_restrictions, slaterWeightMin
            )
            unit_seeds[u] = [block_v[p][i] for p in range(len(unit.chunk)) for i in range(unit.n_ops)]
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
        # The kernels clone `basis`, so the GF cap has to be on it for the duration; restored in
        # the `finally`, so the caller's (ground-state) cap survives this call on every path.
        caller_cap = basis.truncation_threshold
        saved_guard = (getattr(basis, "gf_memory_budget", None), getattr(basis, "gf_memory_policy", None))
        resident = current_rss_bytes()
        width = max((len(s) for s in unit_seeds), default=1)
        try:
            basis.truncation_threshold = (
                _pinned_auto_gf_cap(basis, [1], width, reort, gf_method, resident)
                if _is_auto_gf(basis)
                else gf_cap(basis)
            )
            _set_gf_memory_guard(basis, basis, _gf_memory_budget(available_bytes_per_rank(basis.comm), resident))
            line = _describe_gf_cap(basis, basis.truncation_threshold, f"{n_units} units, serial")
            if line is not None:
                print(line, flush=True)
            results = []
            for u in range(n_units):
                result = kernel(basis, u, unit_seeds[u])
                release_freed_heap()  # see release_freed_heap: the next unit starts from a clean RSS
                if reduce_fn is not None:
                    reduce_fn(u, result)
                else:
                    results.append(result)
            return True if reduce_fn is not None else results
        finally:
            basis.truncation_threshold = caller_cap
            _restore_gf_memory_guard(basis, saved_guard)

    seed_offsets = np.concatenate(([0], np.cumsum([len(s) for s in unit_seeds]))).astype(int)
    # Every color's unit basis inherits the same truncation_threshold, so colors multiply
    # per-rank memory: each rank's share of a capped unit basis is threshold/(ranks/n_colors).
    # Cap the concurrency so a cap-filling unit basis still fits the per-rank budget. The
    # probe is collective on basis.comm; the gates (cap finiteness, unit/rank counts) are
    # replicated, so every rank computes the identical max_colors.
    cap = gf_cap(basis)
    width = max((len(s) for s in unit_seeds), default=1)
    # One rule for every cap: the most colors whose real packing lets each unit afford its floor
    # (the user's cap exactly; for auto, the GS basis size or this kernel's pinned cap).
    max_colors = None
    if _is_auto_gf(basis):
        max_colors = _colors_affording(basis, unit_weights, width, reort, gf_method)
    elif np.isfinite(cap):
        max_colors = _colors_affording(basis, unit_weights, width, reort, gf_method, floor=int(cap))
    (
        unit_indices,
        unit_roots,
        _unit_color,
        units_per_color,
        split_basis,
        split_seeds,
    ) = split_basis_and_redistribute_psi(basis, unit_weights, [s for seeds in unit_seeds for s in seeds], max_colors)
    # `split_basis` inherited the job-wide `cap` verbatim (sized for basis.comm.size ranks),
    # but this color runs on only its own share of them -- the mismatch that let a unit basis
    # grow ~n_colors x too large before the previous round's fix (doc/plans/dc_smo_memory.md,
    # "GF unit memory"). Re-derive the cap this color's own rank count can actually afford and
    # tighten (never loosen) `split_basis`'s cap to it. `max_unit_dets_within_budget` is
    # collective on `basis.comm` (it probes available memory); every input up to here is
    # replicated, so every rank of every color computes the identical bound and calls it
    # unconditionally, whether or not the print below fires.
    #
    # `split_basis` IS `basis` whenever the split collapses to a single color
    # (`basis_split._pack_units` returns `None` subgroups at `n_colors <= 1`, and
    # `split_basis_and_redistribute_psi` then hands the caller's own object back). Writing the
    # unit cap onto it would therefore outlive this call and silently ratchet the caller's cap
    # down -- `spectra.simulate_spectra` reuses ONE basis across IPS/PS/XAS/NIXS/RIXS, so each
    # such call would tighten the next spectrum's basis, an order-dependent accuracy loss with
    # no opt-out. The cap is scoped to this GF phase instead: set below, restored in the
    # `finally` at the end of the function, on every path including an exception.
    n_colors = len(units_per_color)
    # `_pack_units` apportions ranks to colors proportionally to bin mass with a floor of 1
    # (basis_split.py's largest-remainder step), NOT evenly -- colors genuinely differ in rank
    # count. `split_basis.comm.size` is THIS color's real count (identical across every rank of
    # the color, since `split_basis_and_redistribute_psi` verified the packing is rank-invariant
    # before splitting); using the job-wide mean `basis.comm.size // n_colors` here (as a round-7
    # version of this function did) under-caps the larger colors and over-caps the smaller ones --
    # a round-8 SrMnO3 archive had two colors on 4 ranks while the mean said 5, a 25% miss. The
    # collective memory probe itself stays on `basis.comm` (`available_bytes_per_rank` splits a
    # shared-memory sub-communicator internally; calling it on `split_basis.comm` would blind the
    # ranks-per-node count to the OTHER colors sharing this node, which run simultaneously, and
    # over-estimate the per-rank budget) -- only the `ranks` argument fed to the pure bisection
    # after that probe varies per color.
    ranks_per_color = split_basis.comm.size if split_basis.comm is not None else 1
    caller_cap = basis.truncation_threshold
    # What this rank already holds entering the GF phase (ground-state basis, eigenvectors,
    # stored Hamiltonian, the RSPt/Python floor). MAX over the communicator, because the cap
    # has to hold on the rank that is worst off. Sampled unconditionally, outside the
    # `np.isfinite(cap)` gate: it is the only collective here, and keeping it off a
    # conditional keeps the gate's rank-invariance from mattering (CLAUDE.md -- never gate a
    # collective on state that could differ). Without this the per-unit cap below cannot bind
    # at all; see `max_unit_dets_within_budget`'s `resident_bytes`.
    resident_bytes = basis.comm.allreduce(current_rss_bytes(), op=MPI.MAX)
    # Collective on basis.comm (same function max_unit_dets_within_budget calls below); sampled
    # here, unconditionally, purely to report it -- reconstructing this crash needed inverting
    # the color-count bound to recover a number the process had in hand the whole
    # time (doc/plans/dc_smo_memory.md, "GF unit memory", item 4). Printed once, before any unit
    # runs, alongside the block width and the rank-count spread across colors -- none of which
    # the split print recorded before this round, and round 7's own per-unit reporting never
    # fires when the kill lands inside the first unit.
    available_bytes = available_bytes_per_rank(basis.comm)
    if verbose and basis.comm.rank == 0:
        # Derived from `unit_roots` (already rank-invariant, verified in
        # split_basis_and_redistribute_psi) rather than a fresh collective -- the same
        # arithmetic (`np.diff(unit_roots + [comm.size])`) this round used to reconstruct the
        # crash's rank apportionment from the log alone.
        color_rank_counts = [int(d) for d in np.diff(unit_roots + [basis.comm.size])]
        print(f"New unit roots: {unit_roots}")
        print(f"Units per color: {units_per_color}")
        print(
            f"Ranks per color: {color_rank_counts} (block width={width}, "
            f"resident={format_bytes(resident_bytes)}, available={format_bytes(available_bytes)}/rank).",
            flush=True,
        )
        print("=" * 80, flush=True)
    guard = _set_gf_memory_guard(split_basis, basis, _gf_memory_budget(available_bytes, resident_bytes))
    try:
        # Every color starts from the GF cap, not from the cap `split_basis` inherited from the
        # ground-state basis (which the memory guard may have lowered for the ground state).
        if _is_auto_gf(basis):
            # Replicated (derived from the verified-rank-invariant `unit_roots`), so every rank
            # makes the same collective memory-probe calls.
            rank_counts = [int(d) for d in np.diff(unit_roots + [basis.comm.size])]
            cap = _pinned_auto_gf_cap(basis, rank_counts, width, reort, gf_method, resident_bytes)
        split_basis.truncation_threshold = cap
        if np.isfinite(cap) and _may_lower_gf_cap(basis) and not _is_auto_gf(basis):
            unit_cap = max_unit_dets_within_budget(
                basis.num_spin_orbitals,
                width,
                reort,
                ranks_per_color,
                basis.comm,
                method=gf_method,
                resident_bytes=resident_bytes,
            )
            split_basis.truncation_threshold = min(float(cap), float(unit_cap))
            if verbose and basis.comm.rank == 0:
                per_rank = estimate_gf_peak_bytes(
                    int(split_basis.truncation_threshold),
                    basis.num_spin_orbitals,
                    width,
                    reort,
                    ranks=ranks_per_color,
                    method=gf_method,
                )
                print(
                    f"{n_colors} simultaneous unit bases (rank 0's own color has "
                    f"{ranks_per_color} rank(s) -- see 'Ranks per color' above for the rest): "
                    f"unit basis capped at {int(split_basis.truncation_threshold):,} determinants "
                    f"(job-wide truncation_threshold={int(cap):,}; predicted per-rank GF peak "
                    f"{format_bytes(per_rank)}).",
                    flush=True,
                )
        if basis.comm.rank == 0:
            color_sizes = [int(d) for d in np.diff(unit_roots + [basis.comm.size])]
            layout = f"{n_units} units on {n_colors} color(s) of {color_sizes} ranks"
            if max_colors is not None:
                _unconstrained, procs = _pack_units(unit_weights, basis.comm.size, basis.split_threshold, None)
                n_free = 1 if procs is None else len(procs)
                if n_colors < n_free:
                    layout += f" (memory cut this from {n_free} colors, so each unit can afford its cap)"
            line = _describe_gf_cap(basis, split_basis.truncation_threshold, layout)
            if line is not None:
                print(line, flush=True)
        sub_rank = split_basis.comm.rank if split_basis.comm is not None else 0
        unit_indices_per_color = gather_distributed_results(
            basis.comm, sub_rank, unit_roots, units_per_color, np.array(unit_indices), is_array=True
        )

        assert split_seeds is not None  # seeds passed in are a (possibly empty) list, never None
        local_results = []
        for u in unit_indices:
            local_results.append(kernel(split_basis, u, split_seeds[seed_offsets[u] : seed_offsets[u + 1]]))
            release_freed_heap()  # see release_freed_heap: the next unit starts from a clean RSS

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
    finally:
        basis.truncation_threshold = caller_cap
        _restore_gf_memory_guard(split_basis, guard)
        # Free the split communicator collectively, on the error path too: a kernel that raised on
        # every rank (a recoverable failure the caller may retry, e.g. the self-energy's thermal
        # retry or the double-counting search) otherwise leaks one communicator per failure
        # (review ledger M4). MPI_Comm_free is collective; an exception raised on only *some*
        # ranks already leaves the others blocked in the kernel's own collectives, so this adds no
        # new way to hang. Leaving it to Python's gc would free it non-collectively.
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
