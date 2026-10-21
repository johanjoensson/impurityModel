from itertools import product
from typing import NamedTuple

import numpy as np

from impurityModel.ed import config, solver_trace
from impurityModel.ed.basis_restrictions import build_excited_restrictions
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.cipsi_solver import DEFAULT_E_PT2_TOL, CIPSISolver
from impurityModel.ed.eigensolvers import Lowest, Within
from impurityModel.ed.gs_report import report_ground_state
from impurityModel.ed.hartree_fock import hartree_fock_occupation
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.memory_estimate import (
    DEFAULT_MEMORY_SAFETY,
    CapPolicy,
    absolute_rss_budget,
    available_bytes_per_rank,
    cap_value,
    resident_bytes_per_rank,
    resolve_cap_policy,
)
from impurityModel.ed.sector_solver import certification_plan, solve_certified
from impurityModel.ed.seed_sectors import seed_class_plan
from impurityModel.ed.utils import (
    V_DETAIL,
    V_SUMMARY,
)

#: Residual Epstein-Nesbet PT2 energy a *converged* solve is expanded to: the production ground
#: state, and every sector energy a double-counting criterion differences.
#:
#: This is the accuracy the answer is quoted at, so it is the accuracy anything the answer is
#: derived from has to be computed at. It bounds the *sum* of the PT2 contributions left out
#: (:meth:`CIPSISolver.expand`'s ``e_pt2_tol``), which is what bounds the energy error.
GS_E_PT2_TOL = DEFAULT_E_PT2_TOL

#: Per-determinant Epstein-Nesbet PT2 floor for the same solves: off. It used to be the accuracy
#: control at ``1e-8``, and it cannot be one: it bounds each discarded candidate, not their sum.
#: The SrMnO3 production ground state stopped "PT2-converged" at this floor with 5.1e-4 of PT2
#: energy left below it (``doc/plans/dc_smo_memory.md``). Callers may still pass a floor to
#: loosen a solve deliberately; :data:`GS_E_PT2_TOL` is what converges it.
GS_DE2_MIN = 0.0


def sector_key(trial_N0, frozen_occupations=None):
    """The sector a trial impurity occupation actually selects.

    :func:`basis_generation.generate_initial_basis` pins each *frozen* group at its own value
    and, when two or more groups are left unfrozen, lets every one of them range over its whole
    ``[0, group_size]``, filtering only the **total** impurity charge to
    ``[total_nominal +/- total_slack]``. So two trial occupations that differ only in how the
    same total is split across unfrozen groups generate a **bit-identical determinant set** and
    are one sector solve, not several -- and since ``nominal_impurity_occ`` reaches the ``Basis``
    only through that generator (it is not stored on the basis, and the ``delta_*`` arguments use
    ``dict.fromkeys``, i.e. the keys alone), the whole solve is a function of this key.

    With one or zero unfrozen groups the generator instead uses the per-group window
    ``nominal +/- mixed_valence``, but then the frozen values plus the total determine the
    remaining group uniquely, so the key stays injective and merges nothing.

    Every component is an ``int``. That is not incidental: a float anywhere in the key would make
    a cache hit depend on roundoff, and one rank taking the cached value while another runs the
    collective sector solve is an immediate deadlock.
    """
    frozen_occupations = frozen_occupations or set()
    total = sum(int(occ) for occ in trial_N0.values())
    pinned = tuple(sorted((i, int(trial_N0[i])) for i in trial_N0 if i in frozen_occupations))
    return total, pinned


def expand_memory_budget(comm):
    """Per-rank RSS trip-wire for :meth:`CIPSISolver.expand`, or ``None`` when disabled.

    .. warning:: **Collective on** ``comm`` -- :func:`memory_estimate.available_bytes_per_rank`
       splits a shared-memory sub-communicator and min-reduces the result, so every rank must
       reach this. It is called unconditionally at both :meth:`CIPSISolver.expand` call sites for
       that reason; never move it behind a rank-local test.

    The fraction comes from ``GS_MEMORY_BUDGET_SAFETY``, defaulting to
    :data:`memory_estimate.DEFAULT_MEMORY_SAFETY`. A safety of ``0`` disables the guard and
    returns ``None``, which restores the behaviour in which an uncapped expansion grows until the
    kernel kills the rank.

    The budget is **absolute**, not an increment allowance: both guards in
    :meth:`CIPSISolver.expand` measure this process's *whole* RSS, so what they may be compared
    against is the rank's whole share of node RAM -- :func:`memory_estimate.absolute_rss_budget`,
    i.e. ``safety * (available + resident)``. ``safety * available_bytes_per_rank`` alone is the
    *free* memory fraction, and comparing that against absolute RSS is what pinned the SrMnO3
    cubic gap-DC search at its seed basis in every sector after the first: 2.5 GiB resident
    against a 2.45 GiB budget, negative headroom before any work was done, and a first strike
    that locked the threshold there permanently (``doc/plans/dc_smo_memory.md``, round 9).

    Re-sampled on every call, which is per expansion: a process whose RSS has grown gets a
    budget that has grown with it, and `available` falls by the same amount when the growth is
    real, so the ceiling tracks the rank's share rather than drifting with allocator history.

    Shared by both call sites on purpose: the budget is a policy, and two copies of a policy
    expressed as arithmetic is how they drift.
    """
    safety = config.GS_MEMORY_BUDGET_SAFETY.get()
    if safety is None:
        safety = DEFAULT_MEMORY_SAFETY
    if safety <= 0.0:
        return None
    # Both collectives, both unconditional and in a fixed order on every rank -- including the
    # resident one, which runs even when the knob below is about to discard its result. An env
    # var is not guaranteed uniform across ranks, and a collective skipped on some of them is
    # the deadlock class CLAUDE.md's MPI rules exist for.
    available = available_bytes_per_rank(comm)
    resident = resident_bytes_per_rank(comm)
    return int(absolute_rss_budget(safety, available, resident))


def _memory_policy(truncation_threshold):
    """``CIPSISolver.expand``'s ``memory_policy`` for a cap: ``"warn"`` only if the user set it."""
    return "tighten" if CapPolicy.coerce(truncation_threshold).from_memory else "warn"


def build_basis_and_solver(
    h_op,
    impurity_indices,
    bath_states,
    N0,
    mixed_valence,
    tau,
    chain_restrict,
    truncation_threshold,
    comm,
    verbose,
    frozen_occupations,
    weighted_restrictions,
    slaterWeightMin,
    total_charge_slack=0,
    *,
    manifold,
    dense_cutoff=1e3,
):
    basis = Basis(
        impurity_indices,
        bath_states,
        delta_impurity_occ=dict.fromkeys(N0, 0),
        delta_valence_occ=dict.fromkeys(N0, 0),
        delta_conduction_occ=dict.fromkeys(N0, 0),
        frozen_occupations=frozen_occupations,
        nominal_impurity_occ=N0,
        mixed_valence=mixed_valence,
        total_charge_slack=total_charge_slack,
        tau=tau,
        chain_restrict=chain_restrict,
        truncation_threshold=cap_value(truncation_threshold),
        verbose=verbose,
        comm=comm,
        weighted_restrictions=weighted_restrictions,
    )
    solver = CIPSISolver(basis)
    solver.truncate_initial(h_op, manifold, dense_cutoff=dense_cutoff)

    basis.restrictions = build_excited_restrictions(basis, h_op, psis=None, es=None, slater_weight_min=slaterWeightMin)
    return basis, solver


class SectorSolve(NamedTuple):
    """One charge sector, solved: ``(es, psis, basis, solver)``.

    ``es`` and ``psis`` are rank-local solver output (``es`` is replicated only to roundoff);
    ``([inf], [])`` when the sector admits no determinants. The basis is whole, not collapsed onto
    the eigenvector support: collapsing it threw away every determinant below ``slaterWeightMin``
    that the next expansion would have to rediscover by PT2 selection.
    """

    es: np.ndarray
    psis: object
    basis: Basis
    solver: CIPSISolver


def solve_sector(
    h_op,
    impurity_indices,
    bath_states,
    N0,
    mixed_valence,
    tau,
    chain_restrict,
    dense_cutoff,
    *,
    manifold,
    comm=None,
    verbose=True,
    truncation_threshold=None,
    slaterWeightMin=1e-12,
    weighted_restrictions=None,
    frozen_occupations=None,
    de2_min=GS_DE2_MIN,
    e_pt2_tol=GS_E_PT2_TOL,
    plan_store=None,
    shift=0.0,
):
    """One charge sector, solved for the requested ``manifold``: :class:`SectorSolve`.

    The one body behind the occupation walk, the double-counting sector energies and the
    production ground state: build the basis, expand it for ``manifold``, read the manifold off the
    last eigensolve.

    Parameters
    ----------
    manifold : Lowest or Within
        What the sector is solved *for*, and so what its expansion refines the basis for.
        :class:`~impurityModel.ed.eigensolvers.Lowest`: the ground level's energy and one
        eigenvector, no claim about the level's degeneracy -- what an energy comparison between
        sectors needs. :class:`~impurityModel.ed.eigensolvers.Within`: every state inside the
        window, certified complete -- what a thermal average needs (``Within.thermal(tau)``).
    tau : float
        Stored on the returned basis (the Green's-function stage reads ``basis.tau``). It does not
        choose the manifold; ``manifold`` does.
    de2_min, e_pt2_tol : float, float or None
        Convergence of the expansion. The defaults are the occupation walk's (a per-determinant
        floor, no residual criterion: only the *ordering* of sectors is decided). Callers that
        difference two sector energies, or quote the answer, pass :data:`GS_DE2_MIN` and
        :data:`GS_E_PT2_TOL`: the truncation error does not cancel in a difference.
    plan_store, shift : SectorPlanStore or None, float
        A search over a shift ``H - shift * N_imp`` (``h_op`` is already shifted) passes its store:
        a ``Lowest`` sector is then certified once, and at other shifts only the classes that can
        still be lowest are expanded (:class:`~impurityModel.ed.sector_solver.SectorPlanStore`).
    """
    # One call is one *sector solve* -- the unit the occupation walk and the double-counting search
    # are both counted in. The nested timers say which of build / expand / diagonalize the cost
    # sits in; all are no-ops outside a solver_trace.tracing() block.
    with solver_trace.timed("sector_solve", n0=tuple(sorted(N0.items()))) as sector_fields:
        with solver_trace.timed("build"):
            basis, solver = build_basis_and_solver(
                h_op,
                impurity_indices,
                bath_states,
                N0,
                mixed_valence,
                tau,
                chain_restrict,
                truncation_threshold,
                comm,
                verbose,
                frozen_occupations,
                weighted_restrictions,
                slaterWeightMin,
                manifold=manifold,
                dense_cutoff=dense_cutoff,
            )

        if len(basis) == 0:
            sector_fields["n_dets"] = 0
            return SectorSolve(np.array([np.inf]), [], basis, solver)
        expand_kwargs = dict(
            dense_cutoff=dense_cutoff,
            de2_min=de2_min,
            e_pt2_tol=e_pt2_tol,
            slaterWeightMin=slaterWeightMin,
            # Measured-RSS guards (see `expand_memory_budget`). They may hold the basis below
            # a cap derived from memory, or below no cap at all; a cap the user set is final,
            # so for it they only warn. A run that stays under budget is unaffected either way.
            memory_budget_bytes=expand_memory_budget(comm),
            memory_policy=_memory_policy(truncation_threshold),
        )
        if config.GS_SECTOR_CERTIFY.get():
            store = plan_store
            store_key = sector_key(N0, frozen_occupations)
            certification = store.plan(store_key) if store is not None else None
            window = 0.0 if isinstance(manifold, Lowest) else float(manifold.max_energy)
            only = store.jobs_at(store_key, shift, window=window) if certification is not None else None
            if certification is None:
                with solver_trace.timed("certify_plan"):
                    certification = certification_plan(h_op, basis, max_seed=config.GS_CERTIFY_MAX_SEED.get())
            if certification is not None and len(certification.jobs) > 1:
                with solver_trace.timed("certify"):
                    es, psis, sector_basis, sector_solver, verdicts = solve_certified(
                        h_op,
                        basis,
                        certification,
                        manifold,
                        expand_kwargs=expand_kwargs,
                        dense_cutoff=dense_cutoff,
                        slater_weight_min=slaterWeightMin,
                        verbose=verbose,
                        only=only,
                    )
                    if only is not None and store.violated(store_key, shift, verdicts):
                        # Some class came out above its certified tangent: the classes this shift
                        # skipped cannot be trusted either. Certify the sector afresh.
                        solver_trace.note("certify_tangent_violation", n0=store_key)
                        es, psis, sector_basis, sector_solver, verdicts = solve_certified(
                            h_op,
                            basis,
                            certification,
                            manifold,
                            expand_kwargs=expand_kwargs,
                            dense_cutoff=dense_cutoff,
                            slater_weight_min=slaterWeightMin,
                            verbose=verbose,
                        )
                if store is not None:
                    store.record(store_key, certification, shift, verdicts)
                if sector_solver.truncation_report is None:
                    # A class may have lost because the cap or the memory guard held it high: say so on
                    # the solve callers read, so a cap-decided comparison is not taken for a converged one.
                    capped = [v for v in verdicts if v.truncation_report is not None]
                    if capped:
                        sector_solver.truncation_report = dict(
                            capped[0].truncation_report,
                            losing_class=repr(capped[0].job),
                            memory_bound=any(v.memory_bound for v in capped),
                        )
                sector_fields["certified_classes"] = len(verdicts)
                sector_fields["n_dets"] = len(sector_basis)
                return SectorSolve(es, psis, sector_basis, sector_solver)
        with solver_trace.timed("seed_classes"):
            plan = seed_class_plan(h_op, basis, max_seed=int(dense_cutoff), verbose=verbose)
        if plan is not None and len(plan.expanded) > 1:
            sector_fields["seed_classes"] = len(plan.expanded)
            return _solve_sector_by_class(plan, h_op, basis, manifold, comm, verbose, expand_kwargs, sector_fields)
        with solver_trace.timed("expand"):
            solver.expand(h_op, manifold, **expand_kwargs)

        with solver_trace.timed("eigensolve"):
            es, psis = solver.get_eigenvectors(
                h_op,
                manifold,
                dense_cutoff=dense_cutoff,
                slaterWeightMin=slaterWeightMin,
                psi_refs=solver.psi_refs,
                # `expand` ended on a certified solve of this very determinant set whenever it did not
                # truncate afterwards; the generation check inside decides, and a wider request than
                # the one it solved is solved afresh.
                reuse_last=True,
            )
        sector_fields["n_dets"] = len(basis)
        return SectorSolve(es, psis, basis, solver)


def _solve_sector_by_class(plan, h_op, basis, manifold, comm, verbose, expand_kwargs, sector_fields):
    """:func:`solve_sector` with one expansion per seed symmetry class (:mod:`seed_sectors`).

    Each class starts from the seed determinants of its own span, under its exact restrictions
    (``S_z``), with :class:`~impurityModel.ed.seed_sectors.ClassPenalty` lifting every other class
    so that no eigensolve of the expansion snaps back to another class's state. The class's energy
    is then the *unpenalized* lowest eigenvalue of its final basis -- variational whatever the
    penalty did -- and the lowest class wins.

    ``Lowest`` returns the winner as it stands, with the sector's own restrictions put back on its
    basis and on ``h_op`` (the class's ``S_z`` restriction must not outlive the solve), and with the
    cap report of a *losing* class attached when the winner's own expansion never hit the cap: a
    class may have lost because its cap held it too high, and callers read the report to tell a
    cap-decided comparison from a converged one.

    ``Within`` also expands the partner classes, which ``Lowest`` skips: every row of the winning
    multiplet has to be in the thermal manifold, and a partner left at seed quality sits above the
    expanded member by its whole hybridization energy -- outside any thermal window, so the window
    solve would never take it as a reference and the manifold would come out short of copies. The
    bases of every class whose energy lies within the window (plus the largest residual PT2 of the
    expansions, their error bar) are then merged into the unrestricted seed basis and expanded for
    the window.

    Every branch decision reads a broadcast value, so all ranks take the same path through the
    collective solves.
    """
    from impurityModel.ed.manybody_basis import INHERIT
    from impurityModel.ed.seed_sectors import (
        DRIFT_TOL,
        ClassPenalty,
        class_restrictions,
        class_seed_keys,
        class_weight,
    )

    rank = comm.rank if comm is not None else 0
    dense_cutoff = expand_kwargs["dense_cutoff"]
    slater_weight_min = expand_kwargs["slaterWeightMin"]
    lowest = isinstance(manifold, Lowest)
    results = []
    wanted = ("expand",) if lowest else ("expand", "partner")
    cap_reports = []
    for k, cls in enumerate(plan.classes):
        if cls.status not in wanted:
            continue
        extra = class_restrictions(plan, cls)
        weighted = list(basis.weighted_restrictions or []) + extra if extra else INHERIT
        class_basis = basis.clone(initial_basis=class_seed_keys(plan, cls, basis), weighted_restrictions=weighted)
        class_solver = CIPSISolver(class_basis)
        class_solver.matrix_hook = ClassPenalty(plan, cls)
        with solver_trace.timed("expand", seed_class=k):
            class_solver.expand(h_op, Lowest(), **expand_kwargs)
        weight = class_weight(plan, cls, class_basis, class_solver.psi_refs)
        class_solver.matrix_hook = None
        class_solver.eig = None
        with solver_trace.timed("eigensolve", seed_class=k):
            es, psis = class_solver.get_eigenvectors(
                h_op, Lowest(), dense_cutoff=dense_cutoff, slaterWeightMin=slater_weight_min, psi_refs=None
            )
        energy = float(np.min(es)) if len(es) else np.inf
        energy = comm.bcast(energy, root=0) if comm is not None else energy
        residual = (class_solver.convergence_report or {}).get("residual_pt2") or 0.0
        residual = comm.bcast(float(residual), root=0) if comm is not None else float(residual)
        if verbose and rank == 0:
            drift = (
                "" if weight >= DRIFT_TOL else f"  warning: only {weight:.2f} of its seed weight stayed in the class"
            )
            print(
                f"  seed class {k}: E = {energy:.8f} ({len(class_basis):,} determinants){drift}",
                flush=True,
            )
        if class_solver.truncation_report is not None:
            cap_reports.append((k, class_solver.truncation_report))
        results.append((energy, residual, k, class_basis, class_solver, es, psis))
        if lowest:
            # Only the best class so far is ever needed again; drop the rest as soon as they lose.
            results = [min(results, key=lambda r: (r[0], r[2]))]

    best = min(results, key=lambda r: (r[0], r[2]))
    if lowest:
        energy, _residual, k_best, class_basis, class_solver, es, psis = best
        class_basis.weighted_restrictions = basis.weighted_restrictions
        h_op.set_weighted_restrictions(basis.weighted_restrictions)
        if class_solver.truncation_report is None and cap_reports:
            k_capped, report = cap_reports[0]
            class_solver.truncation_report = dict(report, seed_class=k_capped, winning_seed_class=k_best)
        sector_fields["n_dets"] = len(class_basis)
        return SectorSolve(es, psis, class_basis, class_solver)

    e_min = best[0]
    margin = float(manifold.max_energy) + max(r[1] for r in results)
    merged = basis.clone(initial_basis=list(basis.local_basis))
    for energy, _residual, _k, class_basis, _solver, _es, _psis in results:
        if energy - e_min <= margin:
            merged.add_states(list(class_basis.local_basis))
    results.clear()
    solver = CIPSISolver(merged)
    with solver_trace.timed("expand", seed_class="merged"):
        solver.expand(h_op, manifold, **expand_kwargs)
    with solver_trace.timed("eigensolve"):
        es, psis = solver.get_eigenvectors(
            h_op,
            manifold,
            dense_cutoff=dense_cutoff,
            slaterWeightMin=slater_weight_min,
            psi_refs=solver.psi_refs,
            reuse_last=True,
        )
    sector_fields["n_dets"] = len(merged)
    return SectorSolve(es, psis, merged, solver)


def calc_energy(
    h_op,
    impurity_indices,
    bath_states,
    N0,
    mixed_valence,
    chain_restrict,
    dense_cutoff,
    comm=None,
    verbose=True,
    truncation_threshold=None,
    slaterWeightMin=1e-12,
    weighted_restrictions=None,
    frozen_occupations=None,
    de2_min=GS_DE2_MIN,
    e_pt2_tol=GS_E_PT2_TOL,
):
    """The ground-state energy of one charge sector: ``(e0, truncation_report)``.

    :func:`solve_sector` for :class:`~impurityModel.ed.eigensolvers.Lowest`, reduced to its energy:
    one converged eigenpair, nothing about the level's degeneracy, and no basis or solver retained --
    the occupation walk holds floats and nothing else. ``truncation_report`` is the expansion's
    (``None`` when the cap never bound), so a caller can tell a cap-limited energy from a converged
    one. Parameters are those of :func:`solve_sector`; ``N0`` is the nominal impurity occupation.

    ``e0`` is **broadcast**, not merely returned. A Lanczos eigenvalue is replicated across ranks
    only to roundoff, and every caller compares it against something: the occupation walk against
    the running best, and the double-counting criteria against a tolerance *derived from it*, which
    gates every branch deciding whether the next collective solve happens. Enforcing it here stops
    the next caller inheriting the trap. One float against a whole CIPSI expansion.
    """
    es, _psis, _basis, solver = solve_sector(
        h_op,
        impurity_indices,
        bath_states,
        N0,
        mixed_valence,
        0,
        chain_restrict,
        dense_cutoff,
        manifold=Lowest(),
        comm=comm,
        verbose=verbose,
        truncation_threshold=truncation_threshold,
        slaterWeightMin=slaterWeightMin,
        weighted_restrictions=weighted_restrictions,
        frozen_occupations=frozen_occupations,
        de2_min=de2_min,
        e_pt2_tol=e_pt2_tol,
    )
    e0 = float(np.min(es))
    e0 = e0 if comm is None else comm.bcast(e0, root=0)
    return e0, solver.truncation_report


def hartree_fock_seed_occupation(
    h_op, impurity_orbitals, bath_states, N0, frozen_occupations=None, comm=None, verbose=False
):
    """Nominal impurity occupation ``N0`` from a cheap unrestricted Hartree-Fock solve.

    This is the default seed for :func:`find_ground_state_sector`. Instead of running an
    accurate solve for every candidate impurity occupation (the legacy ``O(3^k)`` ``dN``
    scan) — or a rough many-body CIPSI over a *broadened* occupation window, which for long
    bath chains can build a massive basis and exhaust memory — it solves the problem at
    mean-field level (:func:`impurityModel.ed.hartree_fock.hartree_fock_occupation`). The
    unrestricted-HF single determinant variationally minimises the mean-field energy, so
    **its impurity occupation is "the occupation corresponding to the lowest energy" at
    mean-field level**, obtained from a handful of small ``(n_orb x n_orb)`` dense
    diagonalisations. It is quick and hard-bounded in memory regardless of chain length. HF
    is deterministic and its input (``h_op``) is replicated, so every rank computes the
    identical ``N0`` with no communication.

    The mean-field seed can miss the true integer sector by ``±1`` in strongly-correlated,
    near-degenerate cases; the ``mixed_valence`` window of the subsequent accurate solve
    absorbs such a miss.

    Parameters
    ----------
    h_op, impurity_orbitals, bath_states, N0
        As for :func:`calc_energy` / :func:`find_ground_state_sector`. ``N0`` sets the total
        (conserved) particle number: ``sum_i N0[i]`` impurity electrons plus the nominally
        full valence baths.
    comm : MPI.Comm or None
        Used only to gate the verbose print to rank 0; HF itself is uncommunicated.
    verbose : bool, default False

    Returns
    -------
    winning_N0 : dict
        Nominal impurity occupation per orbital set (rounded HF occupation).
    converged : bool
        Whether the HF iteration converged. A non-converged seed is **not** an occupation: it
        is whatever the last iterate happened to be, and callers must not use it (see
        :func:`find_ground_state_sector`, which falls back to the ``dN`` scan). Measured on the
        NiO L-edge workload, a non-converged seed returned ``{1: 4, 2: 10}`` -- a 3d10 impurity
        paid for by emptying the 2p core -- which, once the frozen core is restored, closes the
        shell, collapses the basis to a single determinant and zeroes every core-level spectrum.
    """
    winning_N0, energy, converged = hartree_fock_occupation(
        h_op, impurity_orbitals, bath_states, N0, frozen_occupations=frozen_occupations
    )
    if verbose and (comm is None or comm.rank == 0):
        status = "converged" if converged else "NOT converged"
        per_group = ", ".join(f"group {i} = {winning_N0[i]}" for i in sorted(winning_N0))
        print(f"HF seed: impurity occupation {sum(winning_N0.values())} ({per_group}), E_HF = {energy:.3f}, {status}")
    if comm is not None:
        winning_N0 = comm.bcast(winning_N0, root=0)
        converged = comm.bcast(converged, root=0)

    return winning_N0, converged


class SectorSearch(NamedTuple):
    """The result of the charge-sector walk.

    ``occupation`` is the winning nominal impurity occupation (per orbital set; only its total is
    determined, see :func:`sector_key`), ``energy`` its ground-state energy, ``truncation_report``
    the winning expansion's report (``None`` if the cap never bound it), ``cap_policy`` the resolved
    cap the walk ran under. Nothing else is kept: the walk's bases and solvers are discarded.
    """

    occupation: dict
    energy: float
    truncation_report: dict | None
    cap_policy: CapPolicy


def find_ground_state_sector(
    h_op,
    impurity_orbitals,
    bath_states,
    N0,
    frozen_occupations=None,
    mixed_valence=False,
    chain_restrict=False,
    dense_cutoff=1000,
    comm=None,
    truncation_threshold=None,
    verbose=0,
    slaterWeightMin=1e-12,
    use_hf_seed=True,
    weighted_restrictions=None,
    de2_min=GS_DE2_MIN,
    e_pt2_tol=GS_E_PT2_TOL,
    sector_energy=None,
):
    """
    Find the charge sector with the lowest energy, compare N0 - 1, N0 and N0 + 1: :class:`SectorSearch`.

    Every trial sector is solved for its lowest energy alone (:func:`calc_energy`, a
    :class:`~impurityModel.ed.eigensolvers.Lowest` solve), certified, at the caller's own accuracy
    (``slaterWeightMin``, ``de2_min``, ``e_pt2_tol``). The walk used to run looser
    (``sqrt(slaterWeightMin)`` and a per-determinant ``de2`` floor) because it "only orders sectors";
    but the order *is* the answer, and a looser expansion stalls in the wrong symmetry class: on the
    NiO 10-bath archive the d6 class that holds the ground state stopped 0.17 high at the walk's
    accuracy, which reorders the sectors. The thermal ground state is then solved from scratch at the
    winner (:func:`solve_ground_state`).

    The cost is real on large sectors: on the SrMnO3 cubic archive the loose walk's sector solves
    stopped with a residual PT2 energy of 1.3e-3 to 6.1e-2 (3,464 to 144,555 determinants), so
    converging them is more expansion than the walk used to do. A search that visits a sector at
    many shifts should not pay it at every one.

    ``sector_energy``, if given, replaces :func:`calc_energy`: ``trial_N0 -> (energy,
    truncation_report)``, broadcast. The double-counting search passes its own cached sector solve,
    so the walk and the search's ``N +- 1`` solves at one shift are the same solves.

    use_hf_seed (default True): seed the nominal occupation with a cheap, memory-bounded
    unrestricted Hartree-Fock solve (the mean-field lowest-energy determinant), instead of
    the O(3^k) accurate scan over every dN combination. Set False for the legacy scan.

    verbose : int or bool
        Verbosity level (see ``impurityModel.ed.utils.V_SUMMARY``/``V_DETAIL``); the
        occupation-search table (one row per distinct charge sector solved) and its result
        line print at ``V_SUMMARY``, the per-trial basis-generation windows at ``V_DETAIL``.
        A plain ``bool`` degrades gracefully: ``True`` behaves like ``V_SUMMARY`` (int
        subclass of bool in Python), ``False`` like the terse default.

    truncation_threshold (default None): global cap on the number of Slater determinants in
    the basis; when the basis would grow past it, only the currently most important
    determinants are kept. ``None`` derives the cap from the available per-rank memory
    (:func:`impurityModel.ed.memory_estimate.resolve_cap_policy`; collective on
    ``comm``), ``np.inf`` disables capping.

    """
    # Resolve here only a cap no driver has resolved yet: a bare number or None (calc_gs callers
    # such as get_spectra, library users). A CapPolicy arrives already resolved and announced --
    # re-resolving it would reprint the cap line and restart the warning counts on every trial of
    # a double-counting search.
    if not isinstance(truncation_threshold, CapPolicy) or truncation_threshold.gs is None:
        # Same spin-orbital count formula as Basis.__init__ (blocked orbital lists).
        num_spin_orbitals = sum(
            sum(len(orbs) for orbs in impurity_orbitals[i])
            + sum(len(orbs) for orbs in bath_states[0][i])
            + sum(len(orbs) for orbs in bath_states[1][i])
            for i in bath_states[0]
        )
        truncation_threshold, _ = resolve_cap_policy(
            truncation_threshold,
            num_spin_orbitals,
            comm=comm,
            verbose=verbose,
            label="ground-state basis",
            log="derived",
        )
    if mixed_valence is None or mixed_valence is False:
        mixed_valence = dict.fromkeys(N0, 0)
    (
        _num_val_baths,
        _num_cond_baths,
    ) = bath_states
    if frozen_occupations is None:
        frozen_occupations = set()
    winning_impurity_occ = N0.copy()

    # Per sector (`sector_key`): the energy and the truncation report of its (memory-capped)
    # expansion, so a capped ground-state determination is auditable even when the final basis fits
    # under the cap. Bare floats and dicts -- no Basis is retained -- and `best_cached_key` tracks the
    # lowest energy seen, for the sanity check below: the winning occupation the walk settles on must
    # match it, or every trial was out of bounds.
    energy_cache = {}
    best_cached_key = None

    # Whole-impurity capacity: the groups redistribute charge freely at fixed total, so this --
    # not any per-group size -- is the bound a trial total has to satisfy to be attainable.
    max_total_impurity_occupation = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)

    def get_energy(trial_N0):
        """
        Helper function to calculate, cache, and return the energy for a trial N0.

        Parameters
        ----------
        trial_N0 : dict
            The trial nominal occupations for each orbital set.

        Returns
        -------
        energy : float
            The ground state energy.
        """

        nonlocal best_cached_key

        # Bound on the TOTAL impurity occupation, checked BEFORE the cache lookup because the
        # cache key *is* the total (see sector_key). An unattainable total must never be
        # answerable from the cache: generate_initial_basis builds the window
        # [total_nominal - total_slack, total_nominal + total_slack] and clamps each group to
        # [0, group_size], so with mixed_valence > 0 a request for 7 electrons in 6 orbitals
        # silently yields a basis at an attainable total and a perfectly finite energy filed
        # under the impossible one.
        total_occ = sum(trial_N0.values())
        if total_occ < 0 or total_occ > max_total_impurity_occupation:
            return np.inf
        # Per-GROUP bounds, deliberately NOT cached (recomputing costs one loop over the groups).
        # The key is the total, and generate_initial_basis filters only the total (the groups
        # redistribute freely at fixed whole-impurity charge), so on group sizes (4, 6) the
        # trials {0: 1, 1: 7} and {0: 4, 1: 4} are the same, reachable, total-8 sector and must
        # return the same energy. Only a *frozen* group is genuinely pinned to its own
        # occupation, so only those bounds can reject a trial.
        for orbital_idx, occ in trial_N0.items():
            if orbital_idx not in frozen_occupations:
                continue
            max_occ = sum(len(block) for block in impurity_orbitals[orbital_idx])
            if occ < 0 or occ > max_occ:
                return np.inf
        # Keyed on the sector, not on the split: distinct per-group splits at the same total
        # generate a bit-identical basis and are one solve (see sector_key). This is where the
        # walk's redundancy is -- the diagonal 3^k probe alone visits many splits of the same
        # total -- and a hit here skips the whole solve, not just the basis build.
        key = sector_key(trial_N0, frozen_occupations)
        if key in energy_cache:
            solver_trace.note("sector_cache_hit", n0=key)
            return energy_cache[key][0]

        if sector_energy is not None:
            e_trial, report = sector_energy(trial_N0)
        else:
            e_trial, report = calc_energy(
                h_op,
                impurity_orbitals,
                bath_states,
                trial_N0,
                mixed_valence,
                chain_restrict,
                dense_cutoff,
                comm=comm,
                # CIPSI-round / basis-generation chatter is -vv (V_DETAIL) territory, not -v: the
                # walk's own per-trial summary at V_SUMMARY is already printed above by the caller.
                verbose=verbose >= V_DETAIL,
                truncation_threshold=truncation_threshold,
                slaterWeightMin=slaterWeightMin,
                weighted_restrictions=weighted_restrictions,
                frozen_occupations=frozen_occupations,
                de2_min=de2_min,
                e_pt2_tol=e_pt2_tol,
            )
        energy_cache[key] = (e_trial, report)
        if best_cached_key is None or e_trial < energy_cache[best_cached_key][0]:
            best_cached_key = key
        return e_trial

    keys = list(N0.keys())
    # One report for every branch of the walk (HF seed, its diagonal probe, the axis walk and
    # the legacy dN scan), so the search reads as a single table however it was reached.
    frozen_keys = sorted(i for i in keys if i in frozen_occupations)
    reported_sectors: set = set()

    def report_trial(trial, energy, note=""):
        """Print one row of the occupation-search table, once per distinct sector.

        The row is keyed by the *total* impurity occupation (plus the frozen groups, which are
        genuinely pinned) rather than by the per-group split, because that is what the solve is
        a function of -- see :func:`sector_key`. The old per-group rows invited reading the
        split as orbital polarization, which it is not: with two or more unfrozen groups every
        split of one total builds the identical determinant set, so the printed split was
        merely whichever the walk happened to reach first, and an out-of-bounds *group* value
        (a group occupation of -1) still showed a perfectly finite energy. The per-group
        filling is a measurement only in the impurity density matrix, further downstream.

        Deduplicating on the sector also removes the repeats: the diagonal probe and the axis
        walk revisit the same total many times, and every repeat was an energy-cache hit that
        printed another identical line.
        """
        if not (verbose >= V_SUMMARY and (comm is None or comm.rank == 0)):
            return
        key = sector_key(trial, frozen_occupations)
        if key in reported_sectors:
            return
        if not reported_sectors:
            header = f"  {'N_imp':>5}" + "".join(f"  {f'grp {i}':>7}" for i in frozen_keys) + f"  {'E':>14}"
            print("Impurity occupation search:")
            print(header)
        reported_sectors.add(key)
        row = f"  {sum(trial.values()):>5d}" + "".join(f"  {trial[i]:>7d}" for i in frozen_keys)
        row += f"  {energy:>14.6f}" if np.isfinite(energy) else f"  {'out of range':>14}"
        print(f"{row}  {note}".rstrip(), flush=True)

    def _hf_seed():
        """The HF seed occupation, or ``None`` when it cannot be trusted.

        The frozen shells are passed *into* HF, which solves at their fixed occupation
        (:func:`hartree_fock.hartree_fock_density_matrix`), so on a core-level workload the seed
        is now converged and usable and neither guard below fires. The guards remain because they
        are the difference between a wrong answer and a loud one:

        **(1) HF did not converge** -- the returned occupation is then merely the last iterate,
        not a minimiser. **(2) Re-freezing the frozen shells changes the impurity electron
        count** -- a seed that did not respect the freeze paid for the charge it moved somewhere
        else; restoring the frozen shells keeps its answer for the *unfrozen* ones while undoing
        that compensation, so the total silently drifts.

        Both fired on the NiO L-edge workload before the constraint existed, and either alone was
        fatal: unconstrained HF returned ``{2p: 4, 3d: 10}`` (sum 14 = the correct impurity count,
        a 3d10 paid for out of the core), and re-freezing the core to 6 left ``{2p: 6, 3d: 10}``
        = 16 -- two electrons from nowhere. That closes the shell (a filled system has exactly one
        determinant), so the basis collapsed and every core-level spectrum came out identically
        zero, while PES still looked healthy. Measured against the true d8 ground state, which
        lies inside the scan's own candidate space, the accepted state was 8.7 eV too high.
        """
        seed_N0, converged = hartree_fock_seed_occupation(
            h_op,
            impurity_orbitals,
            bath_states,
            N0,
            frozen_occupations=frozen_occupations,
            comm=comm,
            verbose=verbose,
        )
        refrozen = {i: N0[i] if i in frozen_occupations else seed_N0[i] for i in N0}
        leaked = sum(refrozen.values()) != sum(seed_N0.values())
        if converged and not leaked:
            return refrozen
        if verbose >= V_SUMMARY and (comm is None or comm.rank == 0):
            why = (
                "HF did not converge"
                if not converged
                else (
                    f"re-freezing {sorted(frozen_occupations)} changed the impurity count "
                    f"{sum(seed_N0.values())} -> {sum(refrozen.values())}"
                )
            )
            print(f"HF seed rejected ({why}); falling back to the dN occupation scan.", flush=True)
        return None

    hf_seed = _hf_seed() if use_hf_seed else None
    if hf_seed is not None:
        # A cheap unrestricted Hartree-Fock solve locates the GS impurity occupation
        # (the mean-field lowest-energy determinant), replacing the O(3^k) dN scan; then a
        # single accurate solve at that occupation refines it. HF is quick and memory-bounded
        # (no broad-window CIPSI expansion), which matters for long bath chains.
        winning_impurity_occ = hf_seed
        e_gs = get_energy(winning_impurity_occ)
        report_trial(winning_impurity_occ, e_gs, note="HF seed")

        # The HF seed is a single mean-field determinant and can miss the true sector along more
        # than one group at once -- e.g. an inter-group charge transfer (dN = (-1, +1)) whose two
        # axis neighbours are BOTH higher in energy than the seed (each pays a charging energy
        # U_A or U_B in isolation, while only the combined move gains the transfer energy), so a
        # pure coordinate-descent walk finds no improving neighbour and stops short. Probe every
        # combination move once from the seed, exactly like the legacy scan's first pass, so a
        # diagonal correction is reachable; bounded to a handful of groups (3^k) so this stays
        # cheap, since impurities with more groups fall back to axis-only moves from the seed.
        if 1 < len(keys) <= 4:
            best_combo = winning_impurity_occ
            best_combo_e = e_gs
            for dN in product([0, -1, 1], repeat=len(keys)):
                if all(d == 0 for d in dN):
                    continue
                trial = {
                    keys[i]: winning_impurity_occ[keys[i]] + (0 if keys[i] in frozen_occupations else dN[i])
                    for i in range(len(keys))
                }
                e_trial = get_energy(trial)
                report_trial(trial, e_trial)
                if e_trial < best_combo_e:
                    best_combo_e = e_trial
                    best_combo = trial
            winning_impurity_occ = best_combo
            e_gs = best_combo_e

        while True:
            best_neighbor = None
            best_e = e_gs

            for i in keys:
                if i in frozen_occupations:
                    continue
                for delta in (-1, 1):
                    trial = winning_impurity_occ.copy()
                    trial[i] += delta
                    max_occ = sum(len(block) for block in impurity_orbitals[i])
                    if 0 <= trial[i] <= max_occ:
                        e_trial = get_energy(trial)
                        report_trial(trial, e_trial)
                        if e_trial < best_e:
                            best_e = e_trial
                            best_neighbor = trial

            if best_neighbor is None or best_e >= e_gs:
                break

            winning_impurity_occ = best_neighbor
            e_gs = best_e
    else:
        dN_trials = [
            {keys[i]: dN[i] if keys[i] not in frozen_occupations else 0 for i in range(len(keys))}
            for dN in product([0, -1, 1], repeat=len(keys))
        ]
        e_gs = np.inf
        dN_gs = dict.fromkeys(N0.keys(), 0)
        for dN in dN_trials:
            trial_N0 = {i: N0[i] + dN[i] for i in N0}
            e_trial = get_energy(trial_N0)
            report_trial(trial_N0, e_trial)
            if e_trial < e_gs:
                e_gs = e_trial
                dN_gs = dN
                winning_impurity_occ = trial_N0
        for i in N0:
            while (
                dN_gs[i] != 0
                and all(imp_occ + dN_gs[j] > 0 for j, imp_occ in winning_impurity_occ.items())
                and all(
                    imp_occ + dN_gs[j] <= sum(len(block) for block in impurity_orbitals[j])
                    for j, imp_occ in winning_impurity_occ.items()
                )
            ):
                trial_N0 = {j: n + dN_gs[i] if i == j else n for j, n in winning_impurity_occ.items()}
                e_trial = get_energy(trial_N0)
                report_trial(trial_N0, e_trial)
                if e_trial >= e_gs:
                    break
                winning_impurity_occ[i] += dN_gs[i]
                e_gs = e_trial

    if (
        sector_key(winning_impurity_occ, frozen_occupations) != best_cached_key
        and verbose >= V_SUMMARY
        and (comm is None or comm.rank == 0)
    ):
        print(
            "WARNING: the winning occupation does not match the cache's best entry "
            f"({winning_impurity_occ} vs key {best_cached_key}); this can happen if every trial "
            "occupation was out of bounds.",
            flush=True,
        )
    if verbose >= V_SUMMARY and (comm is None or comm.rank == 0):
        # The *total* is what this search determined. The per-group split is a basis-generation
        # parameter, not a measurement: with two or more unfrozen groups the generator lets each
        # range over its whole [0, group_size] and filters only the total (see sector_key), so
        # every split of the winning total generates the identical determinant set and the one
        # printed here is simply whichever the walk reached first. Printing it as a per-group
        # table invited reading it as orbital polarization, which it is not -- the actual per-group
        # filling is the impurity density matrix calc_gs reports downstream.
        total = sum(winning_impurity_occ.values())
        frozen_note = (
            "  (frozen: " + ", ".join(f"group {i} = {winning_impurity_occ[i]}" for i in frozen_keys) + ")"
            if frozen_keys
            else ""
        )
        print(f"Ground state: impurity occupation {total}{frozen_note}, E_GS = {e_gs:.4f}")
        print("  per-group filling is not determined here -- see the impurity density matrix below")
        print("=" * 80)
    winning_key = sector_key(winning_impurity_occ, frozen_occupations)
    if winning_key not in energy_cache:
        # The walk settled on a sector it never solved (every trial out of bounds, or a collapsed
        # search): solve the winner.
        get_energy(winning_impurity_occ)
    e_gs, report = energy_cache[winning_key]
    return SectorSearch(dict(winning_impurity_occ), e_gs, report, CapPolicy.coerce(truncation_threshold))


class GroundState(NamedTuple):
    """The thermal ground state: ``basis``, ``solver`` (``truncation_report``, ``convergence_report``,
    ``psi_refs``), the energies ``es`` and eigenstates ``psis`` within the thermal window,
    the winning ``occupation`` and ``walk_report`` (the sector walk's truncation report at the winner)."""

    basis: Basis
    solver: CIPSISolver
    es: np.ndarray
    psis: object
    occupation: dict
    walk_report: dict | None


def solve_ground_state(
    h_op,
    impurity_orbitals,
    bath_states,
    N0,
    *,
    tau,
    slaterWeightMin,
    frozen_occupations=None,
    mixed_valence=None,
    chain_restrict=False,
    dense_cutoff=1000,
    comm=None,
    rank=0,
    verbose=False,
    truncation_threshold=None,
    weighted_restrictions=None,
    use_hf_seed=True,
    de2_min=GS_DE2_MIN,
    e_pt2_tol=GS_E_PT2_TOL,
):
    """Find the thermal ground state: a sector walk, then one complete solve of the winning sector.

    **The single definition of "the ground state of this model".** ``calc_gs`` and the
    double-counting criteria in :mod:`dc_criteria` both go through here, because a ``dc``
    measured on a different state than the one ``calc_selfenergy`` later finds does not
    approximate the requested physics -- it locks the downstream calculation onto the wrong
    charge state. Anything that decides *which* state is found belongs in this signature so the
    callers cannot disagree about it.

    Two steps, with two different manifolds:

    1. **Which sector** -- :func:`find_ground_state_sector`: every trial sector is solved for its
       lowest energy only. Cheap, no completeness claim, and nothing from it is kept but the winning
       occupation (and its truncation report).
    2. **The thermal manifold** -- :func:`solve_sector` at the winner, built from the seed basis
       and expanded for ``Within.thermal(tau)``, converged to a residual PT2 energy of
       ``e_pt2_tol`` at the full ``slaterWeightMin``. Every state within ``energy_cut(tau)`` of the
       lowest is returned, certified complete (or reported not to be). The walk's own basis is *not*
       continued: it was refined for the ground level alone, so any low-lying state outside that
       level sat at seed quality in it -- never a reference, its determinants never selected. Here
       every symmetry partner and every low-lying state starts at equal quality. The NiO ground
       state that motivated this is 3-fold quasi-degenerate, so a manifold short of one copy
       misreports every thermal average taken from it.

    Returns
    -------
    GroundState
        ``basis`` carries ``ground_state_occupation`` (the winning sector) and ``cap_policy``.
    """
    search = find_ground_state_sector(
        h_op,
        impurity_orbitals,
        bath_states,
        N0,
        frozen_occupations=frozen_occupations,
        mixed_valence=mixed_valence,
        chain_restrict=chain_restrict,
        dense_cutoff=dense_cutoff,
        comm=comm,
        verbose=verbose,
        truncation_threshold=truncation_threshold,
        slaterWeightMin=slaterWeightMin,
        use_hf_seed=use_hf_seed,
        weighted_restrictions=weighted_restrictions,
        de2_min=de2_min,
        e_pt2_tol=e_pt2_tol,
    )
    es, psis, basis, solver = solve_sector(
        h_op,
        impurity_orbitals,
        bath_states,
        search.occupation,
        mixed_valence,
        tau,
        chain_restrict,
        dense_cutoff,
        manifold=Within.thermal(tau),
        comm=comm,
        verbose=verbose >= V_DETAIL,
        # The resolved policy, not what the caller passed: the walk announced it once.
        truncation_threshold=search.cap_policy,
        slaterWeightMin=slaterWeightMin,
        weighted_restrictions=weighted_restrictions,
        frozen_occupations=frozen_occupations,
        de2_min=de2_min,
        e_pt2_tol=e_pt2_tol,
    )
    # The sector this search settled on. It cannot be recovered from the basis: the CIPSI expansion
    # widened the impurity occupation window (build_excited_restrictions is called with
    # imp_change=None, i.e. unconstrained), so its determinants span several impurity occupations --
    # measured {1, 2, 3} on a split-block toy whose winning sector is 2.
    basis.ground_state_occupation = dict(search.occupation)
    # The resolved cap policy, read-only and immutable, for the Green's-function stage to size its
    # unit bases from (gf_units.gf_cap). Deliberately not `basis.truncation_threshold`: the memory
    # guard may lower that one for the ground state, and a GF unit must not inherit it.
    basis.cap_policy = search.cap_policy

    # The residual in `solver.convergence_report` covers the reference states `expand` carried: the
    # states the last selection round scored, not `solver.psi_refs` -- a capped expansion returns the
    # block of an eigensolve that ran after that round and may be wider.
    n_references = int((solver.last_selection or {}).get("n_references", 0))
    widened = len(es) > n_references
    if basis.is_distributed:
        widened = basis.comm.bcast(widened, root=0)
    report = getattr(solver, "convergence_report", None)
    if widened and report is not None and report.get("converged"):
        report["converged"] = False
        report["limited_by"] = "manifold_widened"
        if rank == 0:
            print(
                f"WARNING: the thermal manifold ({len(es)} states) is wider than the {n_references} "
                "reference states the CIPSI expansion converged; the residual PT2 energy does not "
                "cover the rest.",
                flush=True,
            )
    return GroundState(basis, solver, es, psis, search.occupation, search.truncation_report)


def refined_truncation_report(solver, walk_report):
    """The truncation report that describes the ground state.

    The production solve's own when it has one. Otherwise the occupation walk's (``walk_report``),
    which still says whether the *cap* bound the search -- but not its memory verdict: the production
    solve ran from the resolved cap, so a walk the memory guard held is no longer a memory-bound
    ground state once the solve ran unbound. Carrying ``memory_bound`` through would raise a false
    ``gs_memory`` warning in the self-energy and a spurious refusal in the double-counting search.
    The walk's verdict is kept as ``walk_memory_bound``, for the record.
    """
    report = getattr(solver, "truncation_report", None)
    if report is not None:
        return report
    if walk_report is None:
        return None
    return {**walk_report, "memory_bound": False, "walk_memory_bound": bool(walk_report.get("memory_bound", False))}


def calc_gs(
    Hop: ManyBodyOperator,
    basis_setup: dict,
    block_structure: BlockStructure,
    rot_to_spherical: np.ndarray,
    verbose,
    slaterWeightMin=0,
    stats_path: str = "ground_state_statistics.json",
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
    slaterWeightMin : float
        Minimum weight threshold for determinants in the basis.

    Returns
    -------
    psis : list of ManyBodyState
        The low-energy eigenstates.
    es : ndarray
        The corresponding eigen-energies.
    ground_state_basis : Basis
        The optimized many-body basis.
    thermal_rho : ndarray
        The thermally-averaged density matrix.
    gs_info : dict
        A dictionary containing additional ground-state info (e.g. 'rhos' list).
    """

    basis_setup = dict(basis_setup)
    if "impurity_orbital" in basis_setup:
        basis_setup["impurity_orbitals"] = basis_setup.pop("impurity_orbital")
    if "nominal_impurity_occ" in basis_setup:
        basis_setup["N0"] = basis_setup.pop("nominal_impurity_occ")

    tau = basis_setup["tau"]
    # One definition of "the ground state of this model", shared with the double-counting
    # criteria (see solve_ground_state): it owns the sector walk, its cutoff, the eigensolver choice
    # and the width of the thermal manifold. `basis_setup` is the caller's dict, so pass it through
    # rather than re-listing its keys.
    ground_state = solve_ground_state(Hop, verbose=verbose, slaterWeightMin=slaterWeightMin, **basis_setup)
    ground_state_basis, solver, es, psis = ground_state[:4]
    # Record whether the truncation_threshold bound the ground-state determination (and how
    # the fixed-budget refinement resolved it), so a capped GS is auditable downstream
    # (returned in gs_info and saved to the statistics JSON). None when the cap never bound.
    # The cap can bind either the final expansion here or the earlier occupation search
    # (whose final basis may then fit under the cap); report either.
    gs_truncation_report = refined_truncation_report(solver, ground_state.walk_report)
    # How far the refinement is from PT2 convergence: `{"residual_pt2", "e_pt2_tol", "converged",
    # "limited_by"}` (see CIPSISolver.expand). Always present, unlike the truncation report --
    # an uncapped expansion can still stop short of its tolerance (a `de2_min` floor).
    gs_convergence_report = getattr(solver, "convergence_report", None)
    return report_ground_state(
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
    )
