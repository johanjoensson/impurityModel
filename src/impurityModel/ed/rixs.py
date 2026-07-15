"""RIXS (resonant inelastic x-ray scattering) map drivers.

The Kramers-Heisenberg two-step machinery: for each incoming photon energy the core-excited
intermediate resolvent is applied and then projected back through the emission operator to
give the energy-loss spectrum. This module owns the whole RIXS half of the spectroscopy
stack -- the incoming-energy work-unit sizing and greedy adaptive sampler, the per-tier
solver chain (``_R1SolverChain``: spectral sector cache -> shift-recycled Krylov ->
BiCGSTAB/GMRES), the flat-unit distribution driver, and the two public entry points
(:func:`calc_map` for per-polarization maps and :func:`calc_tensor_map` for the
Kramers-Heisenberg tensor that :mod:`polarization` contracts at plot time).

Sits on top of :mod:`greens_function` (imported as ``gf``) like the rest of the spectra
layer; :mod:`spectra` re-exports the two public drivers so ``simulate_spectra`` and existing
``spectra.getRIXSmap_*`` callers reach them unchanged.
"""

from math import ceil

import numpy as np
from mpi4py import MPI

import impurityModel.ed.greens_function as gf
from impurityModel.ed import config
from impurityModel.ed.gf_solvers import solve_shifted_block
from impurityModel.ed.basis_restrictions import build_excited_restrictions
from impurityModel.ed.rational_sampling import barycentric_eval, greedy_next_samples, set_valued_aaa
from impurityModel.ed.BlockLanczosArray import Reort
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
from impurityModel.ed.symmetries import (
    conserved_subset_charges,
    measure_conserved_charges,
    transition_sector_restrictions,
)

comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


def _rixs_win_chunk(n_eigen: int, n_win: int, comm_size: int) -> int:
    r"""Number of contiguous incoming-photon frequencies stacked into one RIXS work unit.

    A unit is (eigenstate x contiguous wIn-chunk); contiguity preserves the bicgstab
    warm-start locality (consecutive wIn points reuse the previous resolvent solution as the
    initial guess) inside a unit -- a unit is atomic, the engine never reorders within one.
    The default targets ~3 units per rank so the LPT packing has slack to balance, without
    fragmenting the warm-start chains more than needed. Serial runs get one unit per
    eigenstate (maximal warm-start locality). Override with :data:`config.GF_RIXS_WIN_CHUNK`.
    """
    override = config.GF_RIXS_WIN_CHUNK.get()
    if override is not None:
        return override
    if comm_size <= 1:
        return max(1, n_win)
    return max(1, min(n_win, ceil(n_eigen * n_win / (3 * comm_size))))


def _rixs_adaptive_tol():
    """Adaptive-wIn stop tolerance (:data:`config.GF_RIXS_ADAPTIVE_TOL`); unset/empty disables."""
    return config.GF_RIXS_ADAPTIVE_TOL.get()


def _rixs_adaptive_batch():
    """New wIn solves per adaptive round (:data:`config.GF_RIXS_ADAPTIVE_BATCH`)."""
    return config.GF_RIXS_ADAPTIVE_BATCH.get()


# Below this many requested wIn points the adaptive sampler cannot beat the dense sweep
# (its initial space-filling sample plus the two-quiet-rounds stop already cost that much).
_RIXS_ADAPTIVE_MIN_GRID = 12
# Fit-component subsample bound: the set-valued AAA weights are determined from at most this
# many components (strided across polarization pairs x wLoss); the final reconstruction uses
# the full component set with the shared weights, which is exact for shared-pole functions.
_RIXS_ADAPTIVE_MAX_FIT_COMPONENTS = 4096
# A reconstructed magnitude exceeding the solved data's envelope by this factor at an
# unsolved node is treated as a spurious barycentric pole (Froissart artifact) and forces a
# solve there; legitimate inter-sample peaks are excluded by the grid being finer than the
# physical broadening.
_RIXS_ADAPTIVE_BLOWUP_FACTOR = 2.0
# Relative residual target of the RIXS intermediate (R1) resolvent solves -- shared by
# every solver on that path (shift-recycled Krylov, BiCGSTAB, its GMRES escalation), so
# a point rescued by a different solver meets the same accuracy. It was 1e-5 while
# BiCGSTAB applied the tolerance to the *warm-start* residual instead, which on this
# sweep is ~10x smaller -- so 1e-6 is what the RIXS map was actually getting, and keeps
# the measured accuracy-vs-dense at 2.6e-7 (test_rixs_tensor_perf). Tighten to 1e-7 for
# ~6e-9 if a map ever needs it.
_RIXS_R1_ATOL = 1e-6


def _rixs_map_adaptive(map_fn, wIns, comm, tol, verbose):
    r"""Greedy adaptive-wIn evaluation of a RIXS map via set-valued AAA.

    Every component of the map (polarization pairs x energy-loss points) shares its
    ``wIn`` poles -- the intermediate core-hole resolvent's -- so the whole map is a
    vector-valued rational function of ``wIn`` and can be reconstructed from solves at a
    few support points (measured on NiO L3: 20 of 121 points at 1e-3, ``doc`` Gate B).

    Strategy: solve a small space-filling initial sample, fit a set-valued AAA
    approximant (one shared support/weight set for a strided component subsample),
    and iterate greedily -- each round solves the wIn point(s) where two consecutive
    approximants disagree most, until they agree within ``tol * max|map|`` on every
    unsolved point for two consecutive rounds (the standard guard against the
    lookahead's failure mode: two iterates agreeing prematurely). Solved points enter
    the returned map exactly; unsolved points are barycentric-evaluated with the full
    component set. Falls back to the dense sweep (all points solved) if convergence
    never sets in.

    MPI: the greedy selection runs on global rank 0 and is broadcast each round;
    ``map_fn`` (collective) is called by every rank with the identical wIn subset, so
    all collectives stay in lock-step. Returns the assembled map on rank 0, ``None``
    elsewhere (the :func:`_rixs_map_flat` contract).

    Trade-off: within one round the warm-start chain only spans that round's batch, so
    each adaptive solve pays more bicgstab iterations than a dense-sweep point; the
    win is the ~5-10x reduction in the number of solves.
    """
    wIns = np.asarray(wIns)
    n = len(wIns)
    root = comm is None or comm.rank == 0
    batch_size = _rixs_adaptive_batch()

    solved: list[int] = []
    cols = {}  # wIn index -> (n_i, n_o, n_l) map column; root only
    fit_idx = None  # strided component subsample, fixed on first solve
    prev_R = None
    quiet_rounds = 0
    min_solves = min(n, 8)
    next_batch = sorted(set(int(i) for i in np.round(np.linspace(0, n - 1, min(5, n)))))
    n_rounds = 0
    last_fit = None  # (support indices into `solved`, weights) of the last guarded fit

    while True:
        if comm is not None:
            next_batch = comm.bcast(next_batch, root=0)
        if not next_batch:
            break
        n_rounds += 1
        sub = map_fn(wIns[np.asarray(next_batch)])  # collective
        if root:
            for k, idx in enumerate(next_batch):
                cols[idx] = sub[:, :, k, :]
            solved.extend(next_batch)
            x_solved = wIns[np.asarray(solved)]
            F_full = np.array([cols[i].reshape(-1) for i in solved])  # (n_solved, K)
            if fit_idx is None:
                stride = max(1, ceil(F_full.shape[1] / _RIXS_ADAPTIVE_MAX_FIT_COMPONENTS))
                fit_idx = np.arange(0, F_full.shape[1], stride)
            F_fit = F_full[:, fit_idx]
            scale = np.max(np.abs(F_fit))
            support, weights = set_valued_aaa(x_solved, F_fit, rtol=0.1 * tol)
            last_fit = (support, weights)
            R = barycentric_eval(wIns, x_solved[support], weights, F_fit[support])
            if prev_R is None or scale == 0.0:
                surrogate = None
            else:
                surrogate = np.max(np.abs(R - prev_R), axis=1) / scale
            prev_R = R
            unsolved = [i for i in range(n) if i not in set(solved)]
            # Spurious-pole (Froissart) guard: a barycentric denominator zero between
            # support points makes the reconstruction blow up at nodes no data pins
            # down -- and BOTH consecutive iterates can share the artifact, so the
            # surrogate alone would happily converge on it (that exact failure produced
            # a 3x-of-scale spike on the NiO L3 map). With the grid finer than the
            # physical broadening, no true inter-sample feature can exceed the sampled
            # envelope by this factor: treat any such node as must-solve.
            recon_mag = np.max(np.abs(R), axis=1)
            blown = [
                i
                for i in unsolved
                if not np.isfinite(recon_mag[i]) or recon_mag[i] > _RIXS_ADAPTIVE_BLOWUP_FACTOR * scale
            ]
            if blown:
                if surrogate is None:
                    surrogate = np.zeros(n)
                finite_max = float(np.max(surrogate[np.isfinite(surrogate)], initial=tol))
                surrogate[blown] = 10.0 * finite_max
            converged = not unsolved or (
                not blown and len(solved) >= min_solves and surrogate is not None and np.max(surrogate[unsolved]) <= tol
            )
            quiet_rounds = quiet_rounds + 1 if converged else 0
            if not unsolved or quiet_rounds >= 2:
                next_batch = []
            else:
                next_batch = greedy_next_samples(wIns, solved, surrogate, batch_size)
        else:
            next_batch = None

    if not root:
        return None
    n_i, n_o, n_l = cols[solved[0]].shape
    gs = np.empty((n_i, n_o, n, n_l), dtype=complex)
    for idx in solved:
        gs[:, :, idx, :] = cols[idx]
    unsolved = [i for i in range(n) if i not in set(solved)]
    if unsolved:
        # Reuse the loop's LAST fit -- it is the one the blow-up guard vetted; a fresh
        # refit here could reintroduce an unchecked artifact.
        support, weights = last_fit
        x_solved = wIns[np.asarray(solved)]
        F_full = np.array([cols[i].reshape(-1) for i in solved])
        recon = barycentric_eval(wIns[np.asarray(unsolved)], x_solved[support], weights, F_full[support])
        for k, idx in enumerate(unsolved):
            gs[:, :, idx, :] = recon[k].reshape(n_i, n_o, n_l)
    if verbose:
        print(
            f"Adaptive RIXS wIn sampling: solved {len(solved)}/{n} points in {n_rounds} rounds "
            f"({len(last_fit[0]) if unsolved else 0} support points; tol {tol:g})."
        )
        print(f"  solved wIn: {np.array2string(np.sort(wIns[np.asarray(solved)]), precision=4, max_line_width=100)}")
    return gs


def _new_rixs_solver_stats():
    """Fresh, zeroed RIXS solver-tier counters -- see :func:`_report_rixs_solver_stats`."""
    return {
        "r1_spectral": 0,
        "r1_recycled": 0,
        "r1_bicgstab": 0,
        "r1_gmres": 0,
        "r1_unconverged": 0,
        "r2_cache": 0,
        "r2_lanczos": 0,
        "r2_unconverged": 0,
        "r2_worst_d_g": 0.0,
    }


def _report_rixs_solver_stats(stats, comm, verbose):
    """Reduce rank-local RIXS solver-tier counters to global rank 0 and print once.

    Counts every R1 intermediate-resolvent solve by the tier that served it (dense
    spectral cache / shift-recycled Krylov / per-point BiCGSTAB, further split into how
    many of those needed a GMRES escalation or still finished unconverged) and every R2
    final-resolvent evaluation by cache-hit vs. per-seed block-Lanczos fallback, plus
    that fallback's worst final relative change and unconverged count -- replacing what
    used to be one ``block Green's function did not reach the convergence tolerance``
    warning line per unconverged R2 point with a single end-of-run summary.

    Collective on ``comm``: every rank's counters (however many units it processed) are
    summed (max for the worst ``d_g``), so this must run unconditionally on every rank --
    only the print itself is gated on ``verbose``.
    """
    counts = {k: v for k, v in stats.items() if k != "r2_worst_d_g"}
    worst_d_g = stats["r2_worst_d_g"]
    if comm is not None:
        counts = {k: comm.reduce(v, op=MPI.SUM, root=0) for k, v in counts.items()}
        worst_d_g = comm.reduce(worst_d_g, op=MPI.MAX, root=0)
    if verbose and (comm is None or comm.rank == 0):
        r1_total = counts["r1_spectral"] + counts["r1_recycled"] + counts["r1_bicgstab"]
        r2_total = counts["r2_cache"] + counts["r2_lanczos"]
        print(
            f"RIXS solver summary: R1 {r1_total} solves "
            f"({counts['r1_spectral']} spectral / {counts['r1_recycled']} recycled / "
            f"{counts['r1_bicgstab']} bicgstab / {counts['r1_gmres']} gmres / "
            f"{counts['r1_unconverged']} unconverged); "
            f"R2 {r2_total} evals ({counts['r2_cache']} cache / {counts['r2_lanczos']} lanczos, "
            f"{counts['r2_unconverged']} unconverged, worst d_g {worst_d_g:.2e})",
            flush=True,
        )


class _R1SolverChain:
    r"""Per-chunk intermediate (R1) resolvent solver: dense sector cache -> shift-recycled
    Krylov -> BiCGSTAB restarts -> GMRES escalation. All tiers target ``_RIXS_R1_ATOL``.

    One instance is created per work unit (one (eigenstate, wIn-chunk) pair, see
    :func:`_rixs_map_flat`'s kernel) and walks that chunk's wIn points in order via
    repeated :meth:`solve` calls. ``r1_cache`` is the caller-owned, per-eigenstate
    :class:`greens_function.SectorResolventCache` (``None`` disables the spectral and
    recycled tiers outright, forcing every point through the per-point fallback).

    The spectral and recycled tiers return that point's solution directly and never
    touch ``psi2_all``; the fallback tier is the only one that reads and updates it,
    exactly as the original inline kernel did -- so a warm-start chain across points
    only exists once the fallback has run at least once for this chunk.

    ``counters``, when given, is one of the caller's (run-lifetime, rank-local)
    :func:`_new_rixs_solver_stats` dicts: each tier increments its own ``r1_*`` count as
    it serves a point, so :func:`_report_rixs_solver_stats` can summarize the whole run.
    """

    def __init__(self, r1_cache, eigenstate, counters=None):
        self.r1_cache = r1_cache
        self.eigenstate = eigenstate  # for the unconverged-fallback warning message only
        self.counters = counters
        self._recycled = None  # in-chunk wIn index -> shift-recycled solution
        self._recycle_declined = r1_cache is None

    def solve(self, tmp_basis, hOp, psi1_all, psi2_all, k, win, remaining_wins, delta1, E_e, slaterWeightMin, verbose):
        """Intermediate-resolvent solution at wIn index ``k`` of the chunk.

        ``remaining_wins`` is this chunk's wIn values from ``k`` onward (the shared shift
        set the recycler solves in one recurrence). ``psi1_all``/``psi2_all`` are mutated
        in place by the fallback tier (rebuild-and-redistribute onto ``tmp_basis``, then
        the warm-started solve), mirroring the original kernel's rebinding.
        """
        z = win + delta1 * 1j + E_e
        if self.r1_cache is not None:
            psi2_spectral = self.r1_cache.try_solve(
                tmp_basis, hOp, psi1_all, z, slaterWeightMin=slaterWeightMin, verbose=verbose
            )
            if psi2_spectral is not None:
                if self.counters is not None:
                    self.counters["r1_spectral"] += 1
                return psi2_spectral
            if self._recycled is None and not self._recycle_declined:
                # The dense sector cache declined (distributed basis or oversized
                # sector): recycle ONE block-Lanczos recurrence across every remaining
                # shift of the chunk -- the right-hand-side block is wIn-independent,
                # so all shifts share the same Krylov space.
                sols = gf.KrylovShiftedResolvent().solve(
                    tmp_basis,
                    hOp,
                    psi1_all,
                    remaining_wins + delta1 * 1j + E_e,
                    slaterWeightMin=slaterWeightMin,
                    atol=_RIXS_R1_ATOL,
                    verbose=verbose,
                )
                if sols is None:
                    self._recycle_declined = True
                else:
                    self._recycled = dict(zip(range(k, k + len(remaining_wins)), sols))
            if self._recycled is not None:
                if self.counters is not None:
                    self.counters["r1_recycled"] += 1
                return self._recycled.pop(k)

        for psi2 in psi2_all:
            psi2.prune(slaterWeightMin)
        tmp_basis.clear()
        tmp_basis.add_states(sorted(set(state for p in psi1_all + psi2_all for state in p.keys())))
        # Align seeds and warm starts to tmp_basis's ownership layout -- the solver assumes
        # its states are distributed per `basis`, and the layout of the freshly rebuilt
        # tmp_basis need not match where the amplitudes currently live.
        n1 = len(psi1_all)
        redistributed = tmp_basis.redistribute_psis(psi1_all + psi2_all)
        psi1_all[:] = redistributed[:n1]
        psi2_all[:] = redistributed[n1:]
        A_op = ManyBodyOperator({((0, "c"), (0, "a")): z, ((0, "a"), (0, "c")): z}) - hOp
        # Warm-started resolvent solved as one block over all in-components, sharing a
        # single Krylov space / iteration (block_bicgstab deflates a rank-deficient block).
        # atol is relative to ||psi1_all|| (see _RIXS_R1_ATOL); the extra iterations are
        # cheap now that a warm start shortens the solve rather than silently tightening
        # its target. solve_shifted_block restarts while unconverged and still making
        # progress and escalates to GMRES on stagnation -- near-pole points are exactly
        # where BiCGSTAB stagnates (measured: a cold-started solve at the NiO L3 window
        # edge silently returned relative residual 7.2), and a stagnated solve caps the
        # map's accuracy at its residual level, so a wrong column must be rescued, and
        # failing that, loud.
        solve_info = {}
        psi2_all[:] = solve_shifted_block(
            A_op, psi2_all, psi1_all, tmp_basis, slaterWeightMin, _RIXS_R1_ATOL, rtol=1e-7, info=solve_info
        )
        if self.counters is not None:
            self.counters["r1_bicgstab"] += 1
            if solve_info["gmres_used"]:
                self.counters["r1_gmres"] += 1
            if not solve_info["converged"]:
                self.counters["r1_unconverged"] += 1
        if not solve_info["converged"] and (tmp_basis.comm is None or tmp_basis.comm.rank == 0):
            print(
                f"warning: RIXS intermediate resolvent at wIn = {win:.6g} (eigenstate {self.eigenstate}) "
                f"stopped unconverged at relative residual "
                f"{solve_info.get('rel_residual', float('nan')):.2e} (after GMRES escalation).",
                flush=True,
            )
        return psi2_all


def _rixs_map_flat(
    hOp,
    in_ops,
    psis,
    Es,
    tau,
    wIns,
    wLoss,
    delta1,
    delta2,
    basis,
    verbose,
    slaterWeightMin,
    n_i,
    n_o,
    eval_out,
    r1_caches=None,
    solver_stats=None,
):
    r"""Shared flat-unit RIXS driver behind :func:`calc_map` and :func:`calc_tensor_map`.

    Work units = (eigenstate x contiguous wIn-chunk), distributed in ONE weighted split through
    the shared engine (:func:`gf_units.run_units_distributed`) -- the same scheme as the
    self-energy and spectra paths. Per-eigenstate metadata (in-component seeds
    ``Tin_a |psi_e>``, conserved-charge sector windows) is computed on the full communicator
    before the split, so every rank holds the identical unit list.

    Each unit's kernel walks its wIn chunk in order: warm-started :func:`cg.block_bicgstab`
    for the intermediate resolvent (R1 sector confinement, R2 in-component block), then
    ``eval_out(green_basis, psi2_all, E_e) -> (n_i, n_o, len(wLoss))`` evaluates the
    out-transition Green's functions (per-pair diagonal or full tensor contraction).

    ``r1_caches`` (dict, eigenstate index -> :class:`greens_function.SectorResolventCache`,
    owned by the caller so it survives repeated invocations, e.g. adaptive-sampling rounds)
    solves the intermediate resolvent spectrally on cacheable sectors -- exact and immune to
    the near-pole BiCGSTAB stagnation that silently poisoned solved columns (measured: a
    cold-started solve at the NiO L3 window edge returned relative residual 7.2 while
    targeting 1e-6). Sectors the cache declines (distributed bases, oversized sectors) go to
    :class:`greens_function.KrylovShiftedResolvent`: one distributed block-Lanczos recurrence
    per chunk serves every remaining wIn shift (the right-hand side is wIn-independent).
    Should that decline too (memory bound), the per-point ``block_bicgstab`` fallback runs,
    restarted while progressing and escalated to ``block_gmres`` when stagnated -- every tier
    targets the same ``_RIXS_R1_ATOL``.

    ``solver_stats`` (one of the caller's :func:`_new_rixs_solver_stats` dicts, likewise
    owned across repeated invocations) accumulates rank-local per-tier solve counts as the
    kernel runs; the caller reports it via :func:`_report_rixs_solver_stats` once the whole
    map (every adaptive round, if any) is done.

    Returns ``gs[i, o, wIn, wLoss] / Z`` (thermally averaged) on global rank 0 and in the
    serial path; ``None`` on other ranks.
    """
    excited_restrictions = build_excited_restrictions(
        basis,
        hOp,
        psis,
        Es,
        imp_change={1: (1, 0), 2: (1, 1)},
        val_change={1: (0, 0), 2: (1, 0)},
        con_change={1: (0, 0), 2: (0, 1)},
    )

    E0 = min(Es)
    Z = np.sum(np.exp(-(Es - E0) / tau))
    comm = basis.comm
    n_win = len(wIns)

    # Conserved-charge sector of the core-excited intermediate state (all in-components share
    # the same charge shift): confines the resolvent solve (R1). Computed per eigenstate on the
    # full communicator (collective, lock-step) before the split.
    charges = conserved_subset_charges(hOp, n_orb=basis.num_spin_orbitals)
    psi1_per_e = [[applyOp_test(tin, psi_e) for tin in in_ops] for psi_e in psis]
    tmp_restrictions_per_e = []
    for psi_e in psis:
        tmp = excited_restrictions
        if charges:
            gs_occ = measure_conserved_charges(psi_e, charges, basis.num_spin_orbitals, comm=comm)
            sector_in = transition_sector_restrictions(charges, gs_occ, in_ops[0])
            if sector_in:
                tmp = gf._intersect_restrictions(excited_restrictions, sector_in)
        tmp_restrictions_per_e.append(tmp)

    # Flat work units. Unit seeds are the eigenstate's in-component excitations (duplicated
    # across its wIn chunks -- core-excited seeds are small); the unit weight is the shared
    # cost model scaled by the chunk's wIn count (resolvent cost is linear in wIn points).
    chunk_size = _rixs_win_chunk(len(Es), n_win, 1 if comm is None else comm.size)
    unit_infos = []  # (eigenstate index, contiguous wIn indices) per unit
    unit_seeds = []
    for e in range(len(Es)):
        for start in range(0, n_win, chunk_size):
            unit_infos.append((e, list(range(start, min(start + chunk_size, n_win)))))
            unit_seeds.append(psi1_per_e[e])
    unit_weights = gf.unit_cost_weights(unit_seeds, comm) * np.array(
        [len(chunk) for _, chunk in unit_infos], dtype=float
    )

    # green_basis depends only on excited_restrictions (identical for every unit), so each
    # color creates it once on its first unit (lazily -- all ranks of a color run the same
    # unit list, so the collective Clone stays in lock-step) and clears it between units
    # instead of cloning a fresh sub-communicator + basis per unit. Freed collectively after
    # run_units_distributed returns on all ranks.
    green_basis_cache = {}

    def kernel(split_basis, u, seeds):
        e, w_chunk = unit_infos[u]
        E_e = Es[e]
        thermal_weight = np.exp(-(E_e - E0) / tau)
        sub_comm = split_basis.comm
        # green_basis hosts the out-transition block-Green solves and accumulates states over
        # the chunk; tmp_basis hosts the intermediate resolvent and is rebuilt per wIn point.
        green_basis = green_basis_cache.get(id(split_basis))
        if green_basis is None:
            green_basis = split_basis.clone(
                initial_basis=[],
                restrictions=excited_restrictions,
                verbose=False,
                comm=sub_comm.Clone() if sub_comm is not None else None,
            )
            green_basis_cache[id(split_basis)] = green_basis
        else:
            green_basis.clear()
        tmp_basis = split_basis.clone(
            initial_basis=[],
            restrictions=tmp_restrictions_per_e[e],
            verbose=False,
            comm=sub_comm.Clone() if sub_comm is not None else None,
        )
        psi1_all = list(seeds)
        psi2_all = [ManyBodyState() for _ in in_ops]
        r1_cache = r1_caches.setdefault(e, gf.SectorResolventCache()) if r1_caches is not None else None
        chain = _R1SolverChain(r1_cache, eigenstate=e, counters=solver_stats)
        out = np.zeros((len(w_chunk), n_i, n_o, len(wLoss)), dtype=complex)
        wins = wIns[w_chunk]
        for k, win in enumerate(wins):
            psi2 = chain.solve(
                tmp_basis, hOp, psi1_all, psi2_all, k, win, wins[k:], delta1, E_e, slaterWeightMin, verbose
            )
            out[k] = eval_out(green_basis, psi2, E_e) * thermal_weight
        # Free the per-unit cloned sub-communicator collectively -- every rank of this color
        # runs the same unit list in the same order. green_basis's clone outlives the unit
        # (per-color cache) and is freed after run_units_distributed.
        if sub_comm is not None:
            tmp_basis.free_comm()
        return out

    # Accumulate each unit's contribution into the preallocated output as it arrives, so
    # rank 0 never holds all unit results plus the assembled tensor simultaneously.
    gs = np.zeros((n_i, n_win, n_o, len(wLoss)), dtype=complex)

    def accumulate(u, res):
        _e, w_chunk = unit_infos[u]
        for k, w_global in enumerate(w_chunk):
            gs[:, w_global, :, :] += res[k]

    got = gf.run_units_distributed(basis, unit_seeds, unit_weights, kernel, verbose=verbose, reduce_fn=accumulate)
    # Collective on each color's clone: every rank of a color created (at most) one cached
    # green_basis and reaches this point after its unit loop.
    for cached in green_basis_cache.values():
        if cached.comm is not None:
            cached.free_comm()
    green_basis_cache.clear()
    if got is None:  # non-root rank of a distributed run
        return None
    return np.transpose(gs, (0, 2, 1, 3)).copy() / Z


def calc_map(
    hOp,
    tOpsIn,
    tOpsOut,
    psis,
    Es,
    tau,
    wIns,
    wLoss,
    delta1,
    delta2,
    basis,
    verbose,
    slaterWeightMin,
):
    r"""
    Return RIXS Green's function for states.

    The map ``gs[in, out, wIn, wLoss]`` is the Kramers-Heisenberg intensity for every
    (in-operator, out-operator) pair. Two efficiency levers are applied on top of the
    straightforward per-pair evaluation (numerically identical results):

    * **Conserved-charge sector confinement (R1):** the intermediate resolvent
      ``(wIn + i delta1 + E - H)^{-1} Tin|psi>`` stays in the core-excited charge sector of
      ``Tin|psi>``, so the resolvent basis is pinned to that sector
      (:func:`symmetries.transition_sector_restrictions`) on top of the occupation window.
    * **In-component block resolvent (R2):** for each ``wIn`` all in-operators' resolvents are
      solved as one block (:func:`cg.block_bicgstab`), sharing a single Krylov space /
      iteration; the block solver deflates a rank-deficient in-component right-hand side.
    * **Out-component block Green (R3):** for a fixed in-operator all out-operators are run
      through a single block-Lanczos (:func:`gf_solvers.block_Green`); the diagonal
      ``(j, j)`` of the resulting block reproduces the per-out-operator Green's function.

    (A full polarization tensor with arbitrary in/out polarizations would require the rank-4
    cross tensor and is not computed; the map is over the supplied operator pairs.)

    For states :math:`|psi \rangle`, calculate:

    :math:`g(w+1j*delta)
    = \langle psi| ROp^\dagger ((wLoss+1j*delta2+e)*\hat{1} - hOp)^{-1} ROp
    |psi \rangle`,

    where :math:`e = \langle psi| hOp |psi \rangle`, and

    :math:`Rop = tOpOut ((wIns+1j*delta1+e)*\hat{1} - hOp)^{-1} tOpIn`.

    Calculations are performed according to:

    1) Calculate state `|psi1> = tOpIn |psi>`.
    2) Calculate state `|psi2> = ((wIns+1j*delta1+e)*\hat{1} - hOp)^{-1}|psi1>`
        This is done by introducing operator:
        `A = (wIns+1j*delta1+e)*\hat{1} - hOp`.
        By applying A from the left on `|psi2> = A^{-1}|psi1>` gives
        the inverse problem: `A|psi2> = |psi1>`.
        This equation can be solved by guessing `|psi2>` and iteratively
        improving it.
    3) Calculate state `|psi3> = tOpOut |psi2>`
    4) Calculate `normalization = sqrt(<psi3|psi3>)`
    5) Normalize psi3 according to: `psi3 /= normalization`
    6) Now the Green's function is given by:
        :math:`g(wLoss+1j*delta2) = normalization^2
        * \langle psi3| ((wLoss+1j*delta2+e)*\hat{1} - hOp)^{-1} |psi3 \rangle`,
        which can efficiently be evaluation using Lanczos.

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian.
    tOpsIn : list of ManyBodyOperator
        Transition operators describing the core-hole excitation.
    tOpsOut : list of ManyBodyOperator
        Transition operators describing the filling of the core-hole.
    psis : list of ManyBodyState
        Thermal eigenstates.
    Es : list of float
        Total energies of the eigenstates.
    wIns : ndarray
        Real axis energy mesh for the incoming photon energy.
    wLoss : ndarray
        Real axis energy mesh for the photon energy loss, i.e. ``wLoss = wIns - wOut``.
    delta1 : float
        Deviation from the real axis for the intermediate (core-excited) resolvent.
    delta2 : float
        Deviation from the real axis for the final (energy-loss) resolvent.
    basis : Basis
        The basis container (carries the communicator).
    slaterWeightMin : float
        Restrict the number of product states by looking at ``|amplitudes|^2``.

    Returns
    -------
    ndarray or None
        ``gs[in, out, wIn, wLoss]`` (thermally averaged) on global rank 0 and in the serial
        path; ``None`` on other ranks.
    """
    n_in = len(tOpsIn)
    n_out = len(tOpsOut)
    solver_stats = _new_rixs_solver_stats()

    def eval_out(green_basis, psi2_all, E_e):
        out = np.zeros((n_in, n_out, len(wLoss)), dtype=complex)
        for i in range(n_in):
            # R3: build the final states for every out-component and run one block-Green over
            # them; the diagonal (out-component j vs itself) reproduces the per-operator result.
            psi3_all = [applyOp_test(tout, psi2_all[i]) for tout in tOpsOut]
            for psi3 in psi3_all:
                green_basis.add_states(psi3.keys())
            psi3_all = green_basis.redistribute_psis(psi3_all)
            r2_info = {}
            # verbose=False regardless of the caller's own verbose flag: this runs once per
            # (eigenstate, wIn, in-component) -- hundreds of times on a real map -- and its
            # non-convergence warning is exactly what produced 309 near-identical lines on
            # the NiO L3 validation run. solver_stats aggregates the same information once.
            alphas, betas, r = gf.block_Green(
                hOp,
                psi3_all,
                green_basis,
                delta2,
                Reort.NONE,
                slaterWeightMin=slaterWeightMin,
                verbose=False,
                info=r2_info,
            )
            solver_stats["r2_lanczos"] += 1
            if not r2_info.get("converged", True):
                solver_stats["r2_unconverged"] += 1
            if r2_info.get("d_g") is not None:
                solver_stats["r2_worst_d_g"] = max(solver_stats["r2_worst_d_g"], r2_info["d_g"])
            g_tensor = gf.calc_G(alphas, betas, r, wLoss, E_e, delta2)
            for j in range(n_out):
                out[i, j, :] = g_tensor[:, j, j]
        return out

    gs = _rixs_map_flat(
        hOp,
        tOpsIn,
        psis,
        Es,
        tau,
        wIns,
        wLoss,
        delta1,
        delta2,
        basis,
        verbose,
        slaterWeightMin,
        n_i=n_in,
        n_o=n_out,
        eval_out=eval_out,
        solver_stats=solver_stats,
    )
    _report_rixs_solver_stats(solver_stats, basis.comm, verbose)
    return gs


def calc_tensor_map(
    hOp,
    in_component_ops,
    out_component_ops,
    psis,
    Es,
    tau,
    wIns,
    wLoss,
    delta1,
    delta2,
    basis,
    verbose,
    slaterWeightMin,
    adaptive_wIn_tol=None,
):
    r"""Full rank-4 Kramers-Heisenberg tensor over Cartesian in/out transition components.

    A dipole (or NIXS) transition operator is *linear* in the polarization,
    :math:`T_\varepsilon = \sum_\alpha \varepsilon_\alpha T_\alpha`, so the Kramers-Heisenberg
    amplitude for any pair of in/out polarizations is a contraction (see
    :func:`impurityModel.ed.polarization.contract_rixs_tensor`) of a single Cartesian-component
    tensor

    .. math:: C_{\alpha\alpha'\beta\beta'}(\omega_\text{in}, \omega_\text{loss}) =
              \langle \psi^{(2)}_\alpha | T^{\text{out}\dagger}_\beta R_2 T^\text{out}_{\beta'}
              | \psi^{(2)}_{\alpha'} \rangle, \qquad
              \psi^{(2)}_\alpha = R_1 T^\text{in}_\alpha |g\rangle,

    with :math:`R_1 = (\omega_\text{in} + i\delta_1 + E - H)^{-1}` and
    :math:`R_2 = (\omega_\text{loss} + i\delta_2 + E - H)^{-1}`. Since
    :math:`C_{\alpha\alpha'\beta\beta'} = \langle s_{\alpha\beta} | R_2 | s_{\alpha'\beta'}
    \rangle` with the seeds :math:`s_{\alpha\beta} = T^\text{out}_\beta \psi^{(2)}_\alpha`, the
    tensor is exactly the resolvent matrix over the flattened seed block -- one block-Lanczos
    (:func:`gf_solvers.block_Green`) yields every polarization cross term at once.

    This function computes and returns ``C`` itself (not a polarization contraction), so any
    number of in/out polarization pairs -- including ones chosen after the fact, e.g. for
    circular dichroism -- can be evaluated as a cheap post-processing step instead of
    re-running the solve. This is the RIXS analogue of :func:`calc_spectra_tensor`.

    The same efficiency levers as :func:`calc_map` apply -- R1 conserved-charge sector
    confinement of the intermediate resolvent and the R2 block resolvent over the in-components
    (:func:`cg.block_bicgstab`, which deflates the frequently rank-deficient Cartesian
    in-component right-hand side).

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian.
    in_component_ops : list of ManyBodyOperator
        Cartesian in-transition (core-hole excitation) component operators, e.g. the three
        dipole components ``dipole_operators(nBaths, [[1,0,0],[0,1,0],[0,0,1]])``.
    out_component_ops : list of ManyBodyOperator
        Cartesian out-transition (core-hole filling) component operators, e.g. the daggered
        dipole components ``daggered_dipole_operators(nBaths, [[1,0,0],[0,1,0],[0,0,1]])``.
    adaptive_wIn_tol : float, optional
        Enable greedy adaptive sampling of the ``wIns`` grid (:func:`_rixs_map_adaptive`):
        only the AAA-selected support frequencies are actually solved, the rest are
        rational-reconstructed to this relative tolerance. ``None`` (default) reads
        ``GF_RIXS_ADAPTIVE_TOL`` from the environment; unset there too means dense.
        Grids shorter than ``_RIXS_ADAPTIVE_MIN_GRID`` are always solved densely.
    **kwargs
        The remaining parameters match :func:`calc_map`.

    Returns
    -------
    ndarray
        ``C[in, out, in', out', wIn, wLoss]`` on rank 0, thermally averaged; empty elsewhere.
    """
    n_in = len(in_component_ops)
    n_out = len(out_component_ops)
    n_pairs = n_in * n_out

    # The R1 (per eigenstate) and R2 sectors are the same for every wIn point and
    # adaptive round (the shift enters only at evaluation time), so each sector's
    # eigendecomposition is computed once: every point's intermediate solve and
    # resolvent matrix become dense contractions. Held in this closure so they outlive
    # the per-round _rixs_map_flat calls of the adaptive sampler.
    r1_caches = {}
    r2_cache = gf.SectorResolventCache()
    solver_stats = _new_rixs_solver_stats()

    def eval_out(green_basis, psi2_all, E_e):
        # Flattened out-seed block s_{a,b} = Tout_b psi2_a; index kf = a * n_out + b. One
        # block-Green over all seeds gives the full resolvent matrix (every cross term).
        seeds = [applyOp_test(out_component_ops[b], psi2_all[a]) for a in range(n_in) for b in range(n_out)]
        g_flat = r2_cache.try_eval(
            green_basis, hOp, seeds, wLoss + 1j * delta2 + E_e, slaterWeightMin=slaterWeightMin, verbose=verbose
        )
        if g_flat is None:  # distributed or over the dense-size bound: per-seed block-Lanczos
            for s in seeds:
                green_basis.add_states(s.keys())
            seeds = green_basis.redistribute_psis(seeds)
            r2_info = {}
            # verbose=False regardless of the caller's own verbose flag -- see the matching
            # comment in calc_map's eval_out: solver_stats aggregates this instead.
            alphas, betas, r = gf.block_Green(
                hOp,
                seeds,
                green_basis,
                delta2,
                Reort.NONE,
                slaterWeightMin=slaterWeightMin,
                verbose=False,
                info=r2_info,
            )
            solver_stats["r2_lanczos"] += 1
            if not r2_info.get("converged", True):
                solver_stats["r2_unconverged"] += 1
            if r2_info.get("d_g") is not None:
                solver_stats["r2_worst_d_g"] = max(solver_stats["r2_worst_d_g"], r2_info["d_g"])
            g_flat = gf.calc_G(alphas, betas, r, wLoss, E_e, delta2)
        else:
            solver_stats["r2_cache"] += 1
        # C[w, alpha, beta, alpha', beta'] = <s_{alpha,beta}| R2 |s_{alpha',beta'}>; flatten the
        # (alpha, beta) / (alpha', beta') pairs into the (n_i, n_o) work-unit axes expected by
        # _rixs_map_flat (kf = a * n_out + b matches the seed ordering above).
        C5 = g_flat.reshape(len(wLoss), n_in, n_out, n_in, n_out)
        return np.moveaxis(C5, 0, -1).reshape(n_pairs, n_pairs, len(wLoss))

    def map_fn(wIn_subset):
        return _rixs_map_flat(
            hOp,
            in_component_ops,
            psis,
            Es,
            tau,
            wIn_subset,
            wLoss,
            delta1,
            delta2,
            basis,
            verbose,
            slaterWeightMin,
            n_i=n_pairs,
            n_o=n_pairs,
            eval_out=eval_out,
            r1_caches=r1_caches,
            solver_stats=solver_stats,
        )

    tol = adaptive_wIn_tol if adaptive_wIn_tol is not None else _rixs_adaptive_tol()
    if tol is not None and len(wIns) >= _RIXS_ADAPTIVE_MIN_GRID:
        gs = _rixs_map_adaptive(map_fn, wIns, basis.comm, tol, verbose)
    else:
        gs = map_fn(np.asarray(wIns))
    _report_rixs_solver_stats(solver_stats, basis.comm, verbose)
    if gs is None:  # non-root rank of a distributed run
        return None
    n_win = gs.shape[2]
    return gs.reshape(n_in, n_out, n_in, n_out, n_win, len(wLoss))
