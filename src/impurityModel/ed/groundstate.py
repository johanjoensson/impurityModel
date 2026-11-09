import contextlib
import dataclasses
import io
import sys
from dataclasses import dataclass
from itertools import product
from typing import NamedTuple, Optional

import numpy as np

from impurityModel.ed import config, solver_trace
from impurityModel.ed.basis_generation import SeedWindow
from impurityModel.ed.basis_restrictions import (
    build_excited_restrictions,
    build_weighted_restrictions,
    excitation_bound_restrictions,
    get_effective_excitation_bounds,
)
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.cipsi_solver import DEFAULT_E_PT2_TOL, CIPSISolver
from impurityModel.ed.eigensolvers import Lowest, Within
from impurityModel.ed.gs_connectivity import measure_gs_connectivity
from impurityModel.ed.gs_report import (
    finalize_members,
    measure_member,
    sharpen_for_report,
)
from impurityModel.ed.hartree_fock import hartree_fock_occupation
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.ed.memory_estimate import (
    DEFAULT_MEMORY_SAFETY,
    CapPolicy,
    absolute_rss_budget,
    available_bytes_per_rank,
    cap_value,
    resident_bytes_per_rank,
    resolve_cap_policy,
)
from impurityModel.ed.mpi_root import on_root
from impurityModel.ed.orbits import build_orbits
from impurityModel.ed.sector_ensemble import (
    ENSEMBLE_RESIDUAL_FACTOR,
    plan_ensemble,
    resolve_mixed_valence,
    sector_weights,
    weight_interval,
    winner_plan,
)
from impurityModel.ed.sector_solver import (
    WINDOW_OVERTAKE_WARN,
    SectorPlanStore,
    certification_plan,
    describe_plan,
    solve_certified,
    window_overtake,
)
from impurityModel.ed.seed_sectors import _images as seed_images
from impurityModel.ed.seed_sectors import gather_seed
from impurityModel.ed.spin_pairs import spin_layout
from impurityModel.ed.symmetries import PARITY_TERM_TOL
from impurityModel.ed.truncation_flags import cap_ever_hit, is_cap_bound
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


def _hard_frozen(frozen_occupations):
    """The groups a freeze really pins: none for a :class:`~basis_generation.SeedWindow`.

    A seed window holds its groups out of the walk (they stay in the sector key, the bounds check and the
    move list) but is not a statement about the physics: Hartree-Fock finds their filling by energy, the
    report does not call them frozen, and the leak warning does not apply.
    """
    return set() if isinstance(frozen_occupations, SeedWindow) else (frozen_occupations or set())


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

    With one or zero unfrozen groups the frozen values plus the total determine the remaining group
    uniquely, so the key stays injective and merges nothing.

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

    The fraction comes from ``GS_MEMORY_BUDGET_SAFETY`` (rank 0's value, broadcast), defaulting to
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

    Sampled once per :func:`solve_sector` and shared by every class expansion and the union expand of
    that sector (not re-sampled per expansion): the RSS a sector's class runs add is not seen by the
    later runs of the same sector. A new sector, or a new call, samples again, and `available` falls
    by the same amount as RSS grows, so the ceiling tracks the rank's share across sectors.

    Shared by both call sites on purpose: the budget is a policy, and two copies of a policy
    expressed as arithmetic is how they drift.
    """
    safety = config.GS_MEMORY_BUDGET_SAFETY.get()
    if safety is None:
        safety = DEFAULT_MEMORY_SAFETY
    if comm is not None:
        # Root's value wins. The knob is read from each rank's own environment, and whether the
        # budget is ``None`` decides whether ``CIPSISolver.expand`` runs its measuring collectives
        # (and the early return in ``_frontier_matrix_fits``): a rank that read ``0`` while the
        # others did not left them waiting in an allreduce it never entered.
        safety = comm.bcast(safety, root=0)
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
    *,
    tau,
    chain_restrict,
    truncation_threshold,
    comm,
    verbose,
    frozen_occupations,
    weighted_restrictions,
    slaterWeightMin,
    total_charge_slack=0,
    manifold,
    dense_cutoff=config.GS_DENSE_CUTOFF,
    spin_layout=None,
):
    basis = Basis(
        impurity_indices,
        bath_states,
        delta_impurity_occ=dict.fromkeys(N0, 0),
        delta_valence_occ=dict.fromkeys(N0, 0),
        delta_conduction_occ=dict.fromkeys(N0, 0),
        frozen_occupations=frozen_occupations,
        nominal_impurity_occ=N0,
        total_charge_slack=total_charge_slack,
        tau=tau,
        chain_restrict=chain_restrict,
        truncation_threshold=cap_value(truncation_threshold),
        verbose=verbose,
        comm=comm,
        weighted_restrictions=weighted_restrictions,
    )
    solver = CIPSISolver(basis)
    if basis.size > basis.truncation_threshold:
        # Cutting the seed itself: by whole orbits too. The restrictions are not set yet, so no member is filtered.
        solver.orbits = _orbits_for(h_op, basis, manifold, spin_layout)
    solver.truncate_initial(h_op, manifold, dense_cutoff=dense_cutoff)
    solver.orbits = None

    basis.restrictions = build_excited_restrictions(basis, h_op, psis=None, es=None, slater_weight_min=slaterWeightMin)
    return basis, solver


#: ``build_basis_and_solver`` under a second name. The connectivity probe's seed is a measurement
#: instrument, not one of the walk's trial bases; going through this name keeps it out of anything that
#: hooks ``build_basis_and_solver`` to see the bases the calculation solves.
build_probe_seed = build_basis_and_solver


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


def _orbits_for(h_op, basis, manifold, spin_layout):
    """The symmetry orbits a ``Within`` solve selects and truncates by (collective), or ``None``."""
    if isinstance(manifold, Lowest) or not config.GS_ORBIT_CLOSURE.get():
        return None
    orbits = build_orbits(h_op, basis, spin_layout=spin_layout)
    if orbits is not None:
        _report_orbit_notes(orbits, basis)
    return orbits


#: Orbit degradations already reported by this process (one line per distinct note, not one per solve).
_REPORTED_ORBIT_NOTES: set[str] = set()


def _report_orbit_notes(orbits, basis):
    """Say, once per process on rank 0, what the orbit closure left out (review GS11).

    ``Orbits.notes`` records that the selection and truncation are *not* closed over some symmetry
    (unpaired orbitals, a symmetry that does not map spin pairs, a group over ``GROUP_CAP``). The
    orbit-closed guarantee then no longer holds for the solve, which the user must be able to see.
    Rank-local print only, no collective.
    """
    comm = basis.comm if basis.is_distributed else None
    if comm is not None and comm.rank != 0:
        return
    for note in orbits.notes:
        if note not in _REPORTED_ORBIT_NOTES:
            _REPORTED_ORBIT_NOTES.add(note)
            print(f"  WARNING: symmetry orbit closure is degraded: {note}", flush=True)


def widen_by_single_excitations(basis, *, max_seed):
    """``(wide basis or None, too_big)``: ``basis`` widened by every single excitation (collective).

    The seed of a closed shell (d0, d10, one determinant) spans one symmetry class of ``H``, and the
    ground state can sit in another one. ``H|seed>`` cannot help: ``H`` conserves what defines the class,
    so the whole Krylov space of the seed stays in it (review S1: the expansion closed on a 15-determinant
    span holding 2e-30 of the exact ground state, which lies on the sector's other determinant
    component). A single excitation does not conserve it: the determinants it adds span the other
    classes, and the plan made on the widened seed expands each one.

    The excitations are generated on rank 0 under the seed's own restrictions and broadcast;
    ``Basis.add_states`` routes each determinant to its owner and drops duplicates, so the replicated
    list is safe on a rank that holds none of the seed. ``(None, False)`` when there is nothing to add;
    ``(None, True)`` when the widened seed would be above ``max_seed`` (no plan can be made on it): the
    caller keeps the seed it has and must not call the sector certified.
    """
    comm = basis.comm if basis.is_distributed else None
    seed = gather_seed(basis)

    def _excitations():
        n = basis.num_spin_orbitals
        excite = tensors_to_operator(np.ones((n, n)))
        excite.set_restrictions(basis.restrictions)
        excite.set_weighted_restrictions(basis.weighted_restrictions)
        known = set(seed)
        found = set()
        for det in seed:
            found.update(key for key in seed_images(excite, det) if key not in known)
            if len(seed) + len(found) > max_seed:
                break  # decided on rank 0: the rest of the shell is never built or broadcast
        return False if len(seed) + len(found) > max_seed else sorted(found)

    # A rank-0 failure is raised on every rank before the broadcast of the result.
    shell = on_root(comm, _excitations)
    if comm is not None:
        shell = comm.bcast(shell, root=0)
    if shell is False:
        solver_trace.note("widen_skipped", n_seed=len(seed), max_seed=max_seed)
        return None, True
    if not shell:
        return None, False
    return basis.clone(initial_basis=[SlaterDeterminant(det) for det in seed + shell]), False


def solve_sector(
    h_op,
    impurity_indices,
    bath_states,
    N0,
    *,
    tau,
    chain_restrict,
    dense_cutoff,
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
    spin_layout=None,
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
        Convergence of the expansion: an optional per-determinant admission floor, and the residual
        PT2 energy the expansion converges to. The defaults are the production ground state's
        (:data:`GS_DE2_MIN`, :data:`GS_E_PT2_TOL`), which the occupation walk runs at too: the
        order of the sectors is its answer.
    plan_store, shift : SectorPlanStore or None, float
        A search over a shift ``H - shift * N_imp`` (``h_op`` is already shifted) passes its store:
        a sector is then certified once, and at other shifts only the classes that can still be
        lowest are expanded (:class:`~impurityModel.ed.sector_solver.SectorPlanStore`). A record is
        reused only at the seed size, accuracy and cap it was made with.
    """
    config.assert_uniform(comm)  # public entry (review H-GS3)
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
                tau=tau,
                chain_restrict=chain_restrict,
                truncation_threshold=truncation_threshold,
                comm=comm,
                verbose=verbose,
                frozen_occupations=frozen_occupations,
                weighted_restrictions=weighted_restrictions,
                slaterWeightMin=slaterWeightMin,
                manifold=manifold,
                dense_cutoff=dense_cutoff,
                spin_layout=spin_layout,
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
            max_seed = config.GS_CERTIFY_MAX_SEED.get()

            def acquire(seed_basis):
                # A record holds for the seed and the accuracy it was certified at, so both are in its
                # key (every component replicated). The seed is a function of the key (the sector and, with a
                # seed window, the window's fixed groups) and the plan is the same at every shift, which is
                # shell-diagonal. The shift's slope is not constant on the seed; the store measures it per
                # class over its active orbitals.
                key = (
                    sector_key(N0, frozen_occupations),
                    len(seed_basis),
                    float(de2_min or 0.0),
                    float(e_pt2_tol or 0.0),
                    float(slaterWeightMin),
                    seed_basis.truncation_threshold,
                    int(dense_cutoff),
                )
                plan = store.plan(key) if store is not None else None
                recorded = plan is not None
                jobs_only = store.jobs_at(key, shift, window=window) if recorded else None
                if plan is None:
                    with solver_trace.timed("certify_plan"):
                        plan = certification_plan(h_op, seed_basis, max_seed=max_seed, spin_layout=spin_layout)
                    if plan is None:
                        # `len(basis)` is replicated, so is this branch. Said, because the single
                        # expansion that follows can stop in an excited symmetry class.
                        solver_trace.note("certify_skipped", n0=key[0], n_seed=len(seed_basis))
                        if store is not None:
                            store.uncertified.add(key)
                        if verbose and (comm is None or comm.rank == 0):
                            print(
                                f"  sector {key[0]}: seed of {len(seed_basis):,} determinants is above "
                                f"GS_CERTIFY_MAX_SEED = {max_seed:,}; solved by one expansion, not certified",
                                flush=True,
                            )
                return key, plan, jobs_only, recorded

            window = 0.0 if isinstance(manifold, Lowest) else float(manifold.max_energy)
            store_key, certification, only, recorded = acquire(basis)
            skipped = False
            if certification is not None and len(certification.jobs) <= 1 and not recorded:
                # One class on the seed does not make one class in the sector: a closed-shell seed (d0,
                # d10, a single determinant) spans one symmetry class of H, and the ground state can
                # sit in another one that H|seed> never reaches (review S1: the expansion closed on an
                # H-invariant span that held 2e-30 of the exact ground state). So the seed is widened,
                # once, by its single excitations, and the plan is made again on it: the added
                # determinants span the other classes, and each is expanded.
                wide, skipped = widen_by_single_excitations(basis, max_seed=max_seed)
                if skipped:
                    # The trapped-class risk stays: say so, as for a seed too large to plan.
                    if store is not None:
                        store.uncertified.add(store_key)
                    if verbose and (comm is None or comm.rank == 0):
                        print(
                            f"  sector {store_key[0]}: the seed's single excitations exceed "
                            f"GS_CERTIFY_MAX_SEED = {max_seed:,}; solved by one expansion, not certified",
                            flush=True,
                        )
                if wide is not None:
                    basis, solver = wide, CIPSISolver(wide)
                    store_key, certification, only, recorded = acquire(basis)
            if certification is not None and len(certification.jobs) <= 1 and store is not None and not skipped:
                # Still one class, on the widened seed where there was a shell to widen by: certified
                # as it stands. No verdicts, so no shift ever skips its expansion.
                store.record(store_key, certification, shift, ())
            if certification is not None and len(certification.jobs) > 1:
                replace_record = False
                if verbose and (comm is None or comm.rank == 0):
                    print(f"  sector {store_key[0]}: {describe_plan(certification)}", flush=True)
                orbits = _orbits_for(h_op, basis, manifold, spin_layout)
                with solver_trace.timed("certify"):
                    es, psis, sector_basis, sector_solver, verdicts = solve_certified(
                        h_op,
                        basis,
                        certification,
                        manifold,
                        orbits=orbits,
                        expand_kwargs=expand_kwargs,
                        dense_cutoff=dense_cutoff,
                        slater_weight_min=slaterWeightMin,
                        verbose=verbose,
                        only=only,
                        occupation_orbitals=None if store is None else store.active_orbitals,
                    )
                    if only is not None and store.violated(store_key, shift, verdicts):
                        # Some class came out above its certified tangent: the classes this shift
                        # skipped cannot be trusted either. Certify the sector afresh.
                        solver_trace.note("certify_tangent_violation", n0=store_key[0])
                        store.recertified.add(store_key)
                        es, psis, sector_basis, sector_solver, verdicts = solve_certified(
                            h_op,
                            basis,
                            certification,
                            manifold,
                            orbits=orbits,
                            expand_kwargs=expand_kwargs,
                            dense_cutoff=dense_cutoff,
                            slater_weight_min=slaterWeightMin,
                            verbose=verbose,
                            occupation_orbitals=store.active_orbitals,
                        )
                        # The fresh certification replaces the stale record and is a full one.
                        only = None
                        replace_record = True
                if store is not None:
                    store.record(store_key, certification, shift, verdicts, replace=replace_record)
                if only is None:
                    # A full certification: say whether the seed window had room to spare.
                    overtake = window_overtake(certification, verdicts)
                    if overtake is not None and store is not None:
                        store.overtake = overtake if store.overtake is None else max(store.overtake, overtake)
                    if overtake is not None and overtake > WINDOW_OVERTAKE_WARN and (comm is None or comm.rank == 0):
                        print(
                            f"WARNING: sector {store_key[0]}: a symmetry class overtook a lower-lying one by "
                            f"{overtake:.2f} of the seed window: a class outside the window could win the same "
                            "way. Raise GS_CERTIFY_WINDOW_FACTOR to check.",
                            flush=True,
                        )
                # A class may have lost because the cap or the memory guard held it high: say so on the solve
                # callers read, so a cap-decided comparison is not taken for a converged one. The union's own
                # report (a cut, say) must not hide that: its memory_bound is the classes' too.
                capped = [v for v in verdicts if cap_ever_hit(v.truncation_report)]
                if capped and sector_solver.truncation_report is None:
                    sector_solver.truncation_report = dict(
                        capped[0].truncation_report,
                        cap_bound=any(v.cap_bound for v in capped),
                        losing_class=repr(capped[0].job),
                        memory_bound=any(v.memory_bound for v in capped),
                    )
                elif capped:
                    sector_solver.truncation_report = dict(
                        sector_solver.truncation_report,
                        cap_bound=is_cap_bound(sector_solver.truncation_report) or any(v.cap_bound for v in capped),
                        losing_class=repr(capped[0].job),
                        memory_bound=bool(sector_solver.truncation_report.get("memory_bound"))
                        or any(v.memory_bound for v in capped),
                    )
                sector_fields["certified_classes"] = len(verdicts)
                sector_fields["n_dets"] = len(sector_basis)
                return SectorSolve(es, psis, sector_basis, sector_solver)
        solver.orbits = _orbits_for(h_op, basis, manifold, spin_layout)
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


class SectorEnergy(NamedTuple):
    """What a ``Lowest`` solve of one sector says about its energy: ``(energy, truncation_report, residual_pt2,
    certified, cap_bound)``.

    ``residual_pt2`` is the expansion's residual PT2 energy -- the error bar of ``energy`` (``None`` when no round
    ran or the source does not know it) -- and ``certified`` is ``convergence_report["converged"]``: ``True`` when
    that residual was measured on the final basis and is within ``e_pt2_tol``, ``False`` when it is not, ``None``
    when unknown. ``cap_bound`` says the cap or the memory guard held the final basis
    (:func:`~impurityModel.ed.truncation_flags.is_cap_bound` of ``truncation_report``), which is also what a
    source that has no report to give (the double-counting search's cached solves) can still tell.
    All five are rank-replicated.
    """

    energy: float
    truncation_report: dict | None
    residual_pt2: float | None = None
    certified: bool | None = None
    cap_bound: bool = False


def calc_energy(
    h_op,
    impurity_indices,
    bath_states,
    N0,
    *,
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
    plan_store=None,
    shift=0.0,
    spin_layout=None,
):
    """The ground-state energy of one charge sector: a :class:`SectorEnergy`.

    :func:`solve_sector` for :class:`~impurityModel.ed.eigensolvers.Lowest`, reduced to its energy:
    one converged eigenpair, nothing about the level's degeneracy, and no basis or solver retained --
    the occupation walk holds numbers and nothing else. ``truncation_report`` is the expansion's
    (``None`` when the cap never bound), so a caller can tell a cap-limited energy from a converged
    one; ``residual_pt2`` and ``certified`` say how far from converged it is. Parameters are those of
    :func:`solve_sector`; ``N0`` is the nominal impurity occupation.

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
        tau=0,
        chain_restrict=chain_restrict,
        dense_cutoff=dense_cutoff,
        manifold=Lowest(),
        comm=comm,
        verbose=verbose,
        truncation_threshold=truncation_threshold,
        slaterWeightMin=slaterWeightMin,
        weighted_restrictions=weighted_restrictions,
        frozen_occupations=frozen_occupations,
        de2_min=de2_min,
        e_pt2_tol=e_pt2_tol,
        plan_store=plan_store,
        shift=shift,
        spin_layout=spin_layout,
    )
    convergence = getattr(solver, "convergence_report", None) or {}
    residual = convergence.get("residual_pt2")
    certified = convergence.get("converged")
    detail = (
        float(np.min(es)),
        None if residual is None else float(residual),
        None if certified is None else bool(certified),
    )
    # One broadcast carries all three: the residual and the verdict are replicated by construction (allreduced
    # selection statistics), but the energy is replicated only to roundoff, and the pair is read together.
    e0, residual, certified = detail if comm is None else comm.bcast(detail, root=0)
    return SectorEnergy(e0, solver.truncation_report, residual, certified, is_cap_bound(solver.truncation_report))


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
    runs on rank 0 and its result is broadcast, so every rank starts from the identical ``N0``
    (bit-identity of a replicated computation is not relied on).

    The mean-field seed can miss the true integer sector by ``±1`` in strongly-correlated,
    near-degenerate cases (by 2-3 on some random models); the +-1 descent of
    :func:`find_ground_state_sector` absorbs such a miss when ``E(N)`` is convex.

    Parameters
    ----------
    h_op, impurity_orbitals, bath_states, N0
        As for :func:`calc_energy` / :func:`find_ground_state_sector`. ``N0`` sets the total
        (conserved) particle number: ``sum_i N0[i]`` impurity electrons plus the nominally
        full valence baths.
    comm : MPI.Comm or None
        The seed is broadcast from rank 0, where the verbose line is printed.
    verbose : bool, default False

    Returns
    -------
    winning_N0 : dict
        Nominal impurity occupation per orbital set (rounded HF occupation).
    converged : bool
        Whether the HF iteration converged. A non-converged seed is **not** an occupation: it
        is whatever the last iterate happened to be, and callers must not use it (see
        :func:`find_ground_state_sector`, which then walks +-1 from the nominal ``N0``). Measured on the
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


class SectorRecord(NamedTuple):
    """One charge sector the walk solved: ``(trial_N0, energy, residual_pt2, certified, cap_bound,
    truncation_report)``.

    ``trial_N0`` is the occupation the sector was first reached by (only its total is determined, see
    :func:`sector_key`); the rest is that sector's :class:`SectorEnergy`.
    """

    trial_N0: dict
    energy: float
    residual_pt2: float | None
    certified: bool | None
    cap_bound: bool
    truncation_report: dict | None

    @property
    def total(self):
        """The whole-impurity occupation of the sector."""
        return sum(self.trial_N0.values())


class SectorDiagnostics(NamedTuple):
    """How firmly the walk's winner beats the sectors next to it (review P11). Recorded, never acted on.

    ``margin`` is the smallest, over the solved neighbours ``N +- 1``, of ``(E_nbr - E_win) / (err_win +
    err_nbr)``, the gap in units of the two energies' error bars (``residual_pt2``): ``inf`` where both are exact,
    negative where a neighbour is *lower*. Below 1 the order of the two is not established by the solves.
    ``second_difference`` is ``E(N+1) - 2 E(N) + E(N-1)`` (``None`` unless both neighbours were solved); a negative
    one means ``E(N)`` is not convex there, and the +-1 descent has no guarantee. ``unknown_error`` is set when a
    sector involved reported no residual, so ``margin`` took its error as 0.
    """

    winner_total: int
    neighbour_totals: tuple
    margin: float | None
    second_difference: float | None
    unknown_error: bool

    @property
    def undecided(self):
        """The neighbour is not separated from the winner beyond the truncation error."""
        return self.margin is not None and self.margin < 1.0


def sector_diagnostics(solved, winner_total):
    """The :class:`SectorDiagnostics` of the winner among the ``solved`` :class:`SectorRecord`s (pure).

    Only sectors of total ``winner_total +- 1`` are neighbours. Where none was solved, ``margin`` is ``None``.
    """
    by_total = {rec.total: rec for rec in solved}
    winner = by_total.get(winner_total)
    if winner is None:
        return SectorDiagnostics(winner_total, (), None, None, False)
    margins, totals, unknown = [], [], winner.residual_pt2 is None
    for total in (winner_total - 1, winner_total + 1):
        rec = by_total.get(total)
        if rec is None or not np.isfinite(rec.energy):
            continue
        totals.append(total)
        unknown = unknown or rec.residual_pt2 is None
        error = (winner.residual_pt2 or 0.0) + (rec.residual_pt2 or 0.0)
        gap = rec.energy - winner.energy
        margins.append(np.inf if error <= 0.0 and gap >= 0.0 else (gap / error if error > 0.0 else -np.inf))
    second = None
    below, above = by_total.get(winner_total - 1), by_total.get(winner_total + 1)
    if below is not None and above is not None and np.isfinite(below.energy) and np.isfinite(above.energy):
        second = float(above.energy - 2.0 * winner.energy + below.energy)
    return SectorDiagnostics(winner_total, tuple(totals), min(margins) if margins else None, second, bool(unknown))


class SectorSearch(NamedTuple):
    """The result of the charge-sector walk.

    ``occupation`` is the winning nominal impurity occupation (per orbital set; only its total is
    determined, see :func:`sector_key`), ``energy`` its ground-state energy, ``truncation_report``
    the winning expansion's report (``None`` if the cap never bound it), ``cap_policy`` the resolved
    cap the walk ran under. ``solved`` is every sector the walk solved, as :class:`SectorRecord`s in solve order
    (replicated: every walk decision comes from broadcast energies), and ``diagnostics`` how firmly the winner beats
    its ``N +- 1`` neighbours (:class:`SectorDiagnostics`). The walk's bases and solvers are discarded.
    """

    occupation: dict
    energy: float
    truncation_report: dict | None
    cap_policy: CapPolicy
    solved: tuple = ()
    diagnostics: SectorDiagnostics | None = None


def _report_margin(diagnostics, winner, verbose, always=True):
    """Print how firmly the winner beats its neighbours: one ``V_SUMMARY`` line, and at every verbosity when the
    order is not established (``margin < 1``) - unless `always` is False (a double-counting search, which walks once per
    trial shift and sits at a crossover by construction), when that too waits for `V_SUMMARY`. Rank 0 only; nothing
    collective depends on it."""
    if diagnostics.margin is None:
        return
    nbrs = "/".join(str(t) for t in diagnostics.neighbour_totals)
    error = "" if diagnostics.unknown_error else f", winner residual PT2 {winner.residual_pt2:.2e}"
    curv = "" if diagnostics.second_difference is None else f", second difference {diagnostics.second_difference:.4f}"
    if diagnostics.undecided and (always or verbose >= V_SUMMARY):
        lower = diagnostics.margin < 0
        print(
            f"WARNING: sector N_imp = {diagnostics.winner_total} is "
            + (
                "not the lowest solved: a neighbour is lower"
                if lower
                else "not separated from its neighbour beyond the truncation error"
            )
            + f" (margin {diagnostics.margin:.2f} over N_imp = {nbrs}{error}{curv}). The comparison is not "
            "established by the solves; raising the cap or tightening e_pt2_tol would decide it.",
            flush=True,
        )
    elif verbose >= V_SUMMARY and not diagnostics.undecided:
        print(f"  margin to N_imp = {nbrs}: {diagnostics.margin:.3g}{error}{curv}", flush=True)


def neighbour_gaps(solved, winner_total, tau):
    """``{N: (E_N - E_win) / tau}`` for the solved sectors ``N = winner_total +- 1`` (pure; review P9).

    The Boltzmann weight a fixed-N thermal state silently drops is ``exp(-gap)`` of the neighbour's lowest level,
    so a gap below :data:`NEIGHBOUR_GAP_WARN` is thermal weight the ground state does not hold. Empty without a
    positive ``tau`` or a solved winner.
    """
    by_total = {rec.total: rec for rec in solved}
    winner = by_total.get(winner_total)
    if winner is None or not tau > 0.0:
        return {}
    return {
        total: float((by_total[total].energy - winner.energy) / tau)
        for total in (winner_total - 1, winner_total + 1)
        if total in by_total and np.isfinite(by_total[total].energy)
    }


#: ``ln(1e4)``: a neighbour sector whose lowest level carries a Boltzmann weight above 1e-4 of the winner's.
NEIGHBOUR_GAP_WARN = float(np.log(1e4))


def canonical_split(occupation, sizes, frozen_occupations=None):
    """The per-group split of an occupation total, clipped to ``[0, size]``: a new dict.

    The walk decides the *total* (:func:`sector_key`); how it is split over the unfrozen groups does not
    change the determinant set, so a walk can end on a split such as ``{0: -1, 1: 1}`` or one
    with a group over-full. This moves such a group to the nearest bound, in key order, and puts the
    surplus or deficit on the other unfrozen groups that have room (also in key order). The total and the
    frozen groups are untouched, and an already valid split is returned unchanged.
    """
    frozen = set(frozen_occupations or ())
    out = dict(occupation)
    free = [i for i in sorted(out) if i not in frozen]
    surplus = 0
    for i in free:
        clipped = min(max(out[i], 0), sizes[i])
        surplus += out[i] - clipped
        out[i] = clipped
    for i in free:
        if surplus == 0:
            break
        step = min(surplus, sizes[i] - out[i]) if surplus > 0 else max(surplus, -out[i])
        out[i] += step
        surplus -= step
    return out


def find_ground_state_sector(
    h_op,
    impurity_orbitals,
    bath_states,
    N0,
    frozen_occupations=None,
    *,
    chain_restrict=False,
    dense_cutoff=config.GS_DENSE_CUTOFF,
    comm=None,
    truncation_threshold=None,
    verbose=0,
    slaterWeightMin=1e-12,
    use_hf_seed=True,
    weighted_restrictions=None,
    de2_min=GS_DE2_MIN,
    e_pt2_tol=GS_E_PT2_TOL,
    sector_energy=None,
    plan_store=None,
    shift=0.0,
    spin_layout=None,
    more_totals=None,
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

    ``sector_energy``, if given, replaces :func:`calc_energy`: ``trial_N0 -> SectorEnergy`` (energy,
    truncation report, residual PT2, certified verdict, cap flag), broadcast. The double-counting search passes its
    own cached sector solve,
    so the walk and the search's ``N +- 1`` solves at one shift are the same solves.

    use_hf_seed (default True): seed the nominal occupation with a cheap, memory-bounded
    unrestricted Hartree-Fock solve (the mean-field lowest-energy determinant), instead of
    the O(3^k) accurate scan over every dN combination, then descend on ``E(N)`` by +-1 steps.
    A rejected seed (HF not converged, or a leak through the frozen shells) starts the same
    descent from the nominal ``N0`` instead, and says so at every verbosity (review P8). Set
    False for the legacy scan.

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
    config.assert_uniform(comm)  # public entry (review H-GS3)
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
            connectivity=lambda: measure_gs_connectivity(
                h_op, impurity_orbitals, bath_states, N0, comm, frozen_occupations=frozen_occupations
            ),
        )
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
    max_total_impurity_occupation, _, _ = sector_bounds(impurity_orbitals, frozen_occupations)

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
        # [0, group_size], so with a non-zero charge window a request for 7 electrons in 6 orbitals
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
            return energy_cache[key].energy

        if sector_energy is not None:
            solved_sector = sector_energy(trial_N0)
        else:
            solved_sector = calc_energy(
                h_op,
                impurity_orbitals,
                bath_states,
                trial_N0,
                chain_restrict=chain_restrict,
                dense_cutoff=dense_cutoff,
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
                plan_store=plan_store,
                shift=shift,
                spin_layout=spin_layout,
            )
        e_trial = solved_sector.energy
        energy_cache[key] = SectorRecord(
            dict(trial_N0),
            e_trial,
            solved_sector.residual_pt2,
            solved_sector.certified,
            solved_sector.cap_bound,
            solved_sector.truncation_report,
        )
        if best_cached_key is None or e_trial < energy_cache[best_cached_key].energy:
            best_cached_key = key
        return e_trial

    keys = list(N0.keys())
    # One report for every branch of the walk (HF seed, the axis walk and the legacy dN scan), so
    # the search reads as a single table however it was reached.
    frozen_keys = sorted(i for i in keys if i in _hard_frozen(frozen_occupations))
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

        Deduplicating on the sector also removes the repeats: the axis walk revisits the same
        total once per unfrozen group, and every repeat was an energy-cache hit that
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
            frozen_occupations=_hard_frozen(frozen_occupations),
            comm=comm,
            verbose=verbose,
        )
        refrozen = {i: N0[i] if i in _hard_frozen(frozen_occupations) else seed_N0[i] for i in N0}
        if isinstance(frozen_occupations, SeedWindow) and (comm is None or comm.rank == 0):
            # The window is centred on HF's filling of its groups, not on the nominal: nothing else can
            # correct it, so say when HF put it elsewhere (a large DC shift can push the d levels below a
            # shallow bath-less shell). Rank 0 only; nothing collective depends on it.
            hf_window = sum(seed_N0[i] for i in frozen_occupations)
            nominal_window = sum(N0[i] for i in frozen_occupations)
            if hf_window != nominal_window:
                print(
                    f"WARNING: Hartree-Fock fills the bath-less shell(s) {sorted(frozen_occupations)} with "
                    f"{hf_window} electrons, not the nominal {nominal_window}: the ground-state seed window is "
                    "centred on the HF count. Check the nominal occupation and the double counting.",
                    flush=True,
                )
        leaked = sum(refrozen.values()) != sum(seed_N0.values())
        if converged and not leaked:
            return refrozen
        # At every verbosity (V_RESULT): a rejected seed changes where the walk starts, and the user
        # should see that (review P8). Rank 0 only, and nothing collective depends on it.
        if comm is None or comm.rank == 0:
            why = (
                "HF did not converge"
                if not converged
                else (
                    f"re-freezing {sorted(_hard_frozen(frozen_occupations))} changed the impurity count "
                    f"{sum(seed_N0.values())} -> {sum(refrozen.values())}"
                )
            )
            print(
                f"HF seed rejected ({why}); walking +-1 from the nominal occupation {sum(N0.values())} instead.",
                flush=True,
            )
        return None

    if use_hf_seed:
        # A cheap unrestricted Hartree-Fock solve locates the GS impurity occupation
        # (the mean-field lowest-energy determinant), replacing the O(3^k) dN scan; then a
        # single accurate solve at that occupation refines it. HF is quick and memory-bounded
        # (no broad-window CIPSI expansion), which matters for long bath chains.
        #
        # A rejected seed starts the same +-1 descent from the nominal N0 (review P8). It used to fall
        # back to the 3^k scan, whose cost grows with the number of groups and which every wrong-sector
        # failure of the review's oracle went through (P7). `hf_seed` is replicated (decided on rank 0
        # and broadcast in hartree_fock_seed_occupation), so every rank starts at the same sector.
        hf_seed = _hf_seed()
        winning_impurity_occ = dict(N0) if hf_seed is None else hf_seed
        e_gs = get_energy(winning_impurity_occ)
        report_trial(winning_impurity_occ, e_gs, note="nominal (HF seed rejected)" if hf_seed is None else "HF seed")

        # No diagonal (multi-group) probe: a sector is its total impurity occupation plus the frozen
        # groups (see sector_key), and frozen groups never move, so every combination move lands on
        # a total. An inter-group charge transfer at fixed total -- what the 3^k probe from the seed
        # was for -- is the seed's own sector, and the totals N0 +- 1 are the axis walk's first
        # step. What the probe added was solving N0 +- 2 (+- k on k groups) on 2-4-group impurities
        # only: on the 5-bath NiO linked-chain run (two unfrozen groups, eg and t2g) that was the
        # d6 sector at every DMFT iteration, the most expensive solve of the walk and 0.10 above
        # the winner. The walk is a descent on E(N) from the HF seed, for any number of groups.

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
            # No per-group bound here: `dN_gs` is 0 for a frozen group, and `get_energy` rejects a total
            # outside [0, max] (and a frozen value out of range), so a step that leaves the lattice
            # is not accepted. Guarding every group at `> 0` made occupation 0 unreachable and let
            # any group at 0 block the extension of all the others (review P7).
            while dN_gs[i] != 0:
                trial_N0 = {j: n + dN_gs[i] if i == j else n for j, n in winning_impurity_occ.items()}
                e_trial = get_energy(trial_N0)
                report_trial(trial_N0, e_trial)
                if e_trial >= e_gs:
                    break
                winning_impurity_occ[i] += dN_gs[i]
                e_gs = e_trial

    winning_impurity_occ = canonical_split(
        winning_impurity_occ,
        {i: sum(len(block) for block in impurity_orbitals[i]) for i in winning_impurity_occ},
        frozen_occupations,
    )
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
    winner = energy_cache[winning_key]
    e_gs, report = winner.energy, winner.truncation_report
    solved = tuple(energy_cache.values())
    if more_totals is not None:
        # Solve further totals the caller (the charge-sector ensemble) asks for. `more_totals` must be a pure
        # function of the replicated `solved`, so every rank asks for the same totals and the collective
        # `get_energy` solves stay matched. A total already tried (or unattainable: `get_energy` stores nothing
        # for it) is never asked for twice, so the loop ends.
        tried = {r.total for r in solved}
        # The same move set as the +-1 walk: a SeedWindow group is held out of it too, and a total is spread
        # over the unfrozen groups with their capacities (a bare `+= dN` could leave a group negative).
        movable = [i for i in keys if i not in (frozen_occupations or ())]
        sizes = {i: sum(len(block) for block in impurity_orbitals[i]) for i in keys}
        while movable:
            wanted = [t for t in more_totals(solved) if t not in tried]
            if not wanted:
                break
            for t in wanted:
                tried.add(t)
                trial = dict(winning_impurity_occ)
                trial[movable[0]] += t - sum(trial.values())
                get_energy(canonical_split(trial, sizes, frozen_occupations))
            solved = tuple(energy_cache.values())
    # Decided on rank 0 and broadcast: every input is replicated, but the verdict steers a print, and two ranks
    # disagreeing on it would be a bug that only shows as a missing line.
    diagnostics = sector_diagnostics(solved, winner.total)
    if comm is not None:
        diagnostics = comm.bcast(diagnostics, root=0)
    if comm is None or comm.rank == 0:
        _report_margin(diagnostics, winner, verbose, always=sector_energy is None)
    return SectorSearch(
        dict(winning_impurity_occ), e_gs, report, CapPolicy.coerce(truncation_threshold), solved, diagnostics
    )


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


def frozen_shell_leak(h_op, impurity_orbitals, frozen_occupations):
    """``(largest, total)`` ``|coefficient|`` of the terms of ``h_op`` that change a frozen shell's electron count.

    A freeze pins only the **seed** (:func:`basis_generation.generate_initial_basis` and :func:`sector_key`); the
    restrictions bound the total impurity charge, not a shell's. So a frozen shell is held exactly only when ``H``
    conserves its electron count, as the spectra Hamiltonian does for the X-ray core by construction (review P1b:
    ``slater_condon_Uop`` builds F_cc, F_cv and G_cv only). A term whose creators and annihilators on the frozen
    orbitals differ in number is ``[H, N_core]``; each is reported by its coefficient. Pure: ``h_op`` is replicated.
    """
    frozen = {orb for i in frozen_occupations or () for block in impurity_orbitals[i] for orb in block}
    largest = total = 0.0
    if not frozen:
        return largest, total
    for term, coeff in h_op.items():
        moved = sum(1 if action == "c" else -1 for orb, action in term if orb in frozen)
        if moved:
            largest = max(largest, abs(coeff))
            total += abs(coeff)
    return largest, total


def _warn_frozen_leak(h_op, impurity_orbitals, frozen_occupations, comm):
    """Say once, on rank 0, when ``H`` does not conserve a frozen shell (the freeze then pins only the seed)."""
    if _hard_frozen(frozen_occupations) and (comm is None or comm.rank == 0):
        # A freeze is only ever the user's (review O5), and it is exact only where H conserves the shell. Where it
        # does not (RSPt h_dft with a one-body core-valence element, say), say so once, at every verbosity. Rank 0
        # alone: a print, nothing collective depends on it.
        largest, total = frozen_shell_leak(h_op, impurity_orbitals, frozen_occupations)
        scale = max((abs(c) for c in h_op.values()), default=0.0)
        if largest > PARITY_TERM_TOL * scale:
            print(
                f"WARNING: H does not conserve the electron count of the frozen shell(s) {sorted(frozen_occupations)}: "
                f"terms moving charge in or out of them reach {largest:.2e} (sum {total:.2e}). The freeze pins only "
                "the seed, so the solve can leak charge out of the shell. Freeze only a shell H conserves (the X-ray "
                "core of a spectra Hamiltonian), or leave it unfrozen.",
                flush=True,
            )


def _announce_uncertified(plan_store, key, comm):
    """One rank-0 WARNING, at every verbosity, when the thermal ground state's sector was never certified.

    A seed above ``GS_CERTIFY_MAX_SEED`` is solved by a single expansion, which can close in an excited symmetry
    class. The sector walk says so only under ``verbose``. For the sector the calculation then rests on, it is
    said always, once per sector and store. The store's sets are replicated, so the condition is too; nothing
    collective depends on it.
    """
    if plan_store is None or key in plan_store.announced_uncertified:
        return
    if not any(isinstance(k, tuple) and k[0] == key for k in plan_store.uncertified):
        return
    plan_store.announced_uncertified.add(key)
    if comm is None or comm.rank == 0:
        print(
            f"WARNING: the ground-state sector {key} was solved by a single expansion, not certified: its seed is "
            "above GS_CERTIFY_MAX_SEED, so the expansion may have closed in an excited symmetry class. Raise "
            "GS_CERTIFY_MAX_SEED to certify it.",
            flush=True,
        )


def solve_member(
    search,
    h_op,
    impurity_orbitals,
    bath_states,
    *,
    tau,
    chain_restrict,
    dense_cutoff,
    comm,
    verbose,
    slaterWeightMin,
    weighted_restrictions,
    frozen_occupations,
    de2_min,
    e_pt2_tol,
    plan_store,
    shift,
    spin_layout,
    warn_neighbours,
):
    """The thermal manifold of the sector ``search.occupation`` (the second step of :func:`solve_ground_state`).

    Shared by the single-sector path and by every member of a charge-sector ensemble, which passes the walk's
    ``search`` with ``occupation`` (and ``truncation_report``) replaced by the member's.
    """
    es, psis, basis, solver = solve_sector(
        h_op,
        impurity_orbitals,
        bath_states,
        search.occupation,
        tau=tau,
        chain_restrict=chain_restrict,
        dense_cutoff=dense_cutoff,
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
        plan_store=plan_store,
        shift=shift,
        spin_layout=spin_layout,
    )
    # The sector this search settled on. It cannot be recovered from the basis: the CIPSI expansion
    # widened the impurity occupation window (build_excited_restrictions is called with
    # imp_change=None, i.e. unconstrained), so its determinants span several impurity occupations --
    # measured {1, 2, 3} on a split-block toy whose winning sector is 2.
    basis.ground_state_occupation = dict(search.occupation)
    _announce_uncertified(plan_store, sector_key(search.occupation, frozen_occupations), comm)
    # The resolved cap policy, read-only and immutable, for the Green's-function stage to size its
    # unit bases from (gf_units.gf_cap). Deliberately not `basis.truncation_threshold`: the memory
    # guard may lower that one for the ground state, and a GF unit must not inherit it.
    basis.cap_policy = search.cap_policy
    # The bath-excitation bounds this ground state needed (valence holes, conduction electrons, total): an
    # output of the solve, so the excited bases (Green's function, spectra, RIXS, susceptibility) run under them
    # through the one inheritance step, `widen(floor(basis.weighted_restrictions))` (+1 on each). Collective
    # (one Allreduce) and unconditional on every rank. The list is REBOUND, never mutated: the caller's list is
    # shared by every ensemble member and every double-counting trial shift. (The occupation search of
    # `dc_criteria` also passes through here, so it pays one basis scan per trial on a basis it then discards.)
    bounds = get_effective_excitation_bounds(basis, cap_bound=is_cap_bound(solver.truncation_report))
    basis.excitation_bounds = bounds
    if bounds is not None:
        basis.weighted_restrictions = list(basis.weighted_restrictions or []) + excitation_bound_restrictions(
            bath_states, bounds
        )

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
        if comm is None or comm.rank == 0:
            print(
                f"WARNING: the thermal manifold ({len(es)} states) is wider than the {n_references} "
                "reference states the CIPSI expansion converged; the residual PT2 energy does not "
                "cover the rest.",
                flush=True,
            )
    if report is not None and search.diagnostics is not None:
        # How firmly the winning sector beat its N +- 1 neighbours (MV 3): in the convergence record, so the
        # ground-state JSON and the self-energy's info carry it next to the WARNING. Replicated (broadcast above).
        d = search.diagnostics
        report["sector_diagnostics"] = {
            "winner_total": d.winner_total,
            "neighbour_totals": list(d.neighbour_totals),
            # null where the comparison is exact (inf) or a neighbour is lower at zero error (-inf): strict JSON.
            "margin": d.margin if d.margin is not None and np.isfinite(d.margin) else None,
            "second_difference": d.second_difference,
            "undecided": bool(d.undecided),
        }
        gaps = neighbour_gaps(search.solved, d.winner_total, tau)
        report["sector_diagnostics"]["neighbour_gaps_tau"] = {str(n): g for n, g in gaps.items()}
        if warn_neighbours:
            _warn_near_neighbours(search, tau, comm)
    return GroundState(basis, solver, es, psis, search.occupation, search.truncation_report)


_SINGLE_SECTOR_WHY = (
    "this run keeps a single charge sector (mixed_valence = off / RSPt no_mv / --mixed_valence off, "
    "or the neighbour was not admitted to the ensemble)"
)


def _warn_near_neighbours(search, tau, comm, kept_totals=None, why=None):
    """Warn (rank 0) when a neighbour of the walk's winner lies inside the thermal window but is not kept.

    ``kept_totals`` are the sectors the ground state holds: the winner alone by default, or the members an
    ensemble kept when it was cut after a member's thermal solve failed (review 7c). The gaps come from the
    walk's records, which are replicated, so this needs no state and no collective. ``why`` names the reason the
    sector is out (a refused member's reason); by default the run keeps one sector because mixed valence is off.
    """
    d = search.diagnostics
    if d is None:
        return
    kept = {d.winner_total} if kept_totals is None else set(kept_totals)
    gaps = neighbour_gaps(search.solved, d.winner_total, tau)
    near = {n: g for n, g in gaps.items() if g < NEIGHBOUR_GAP_WARN and n not in kept}
    if not near or not (comm is None or comm.rank == 0):
        return
    where = ", ".join(
        f"N_imp = {n}: {abs(g):.2f} tau {'above' if g >= 0 else 'BELOW'}" for n, g in sorted(near.items())
    )
    held = f"N_imp = {d.winner_total}" if len(kept) == 1 else f"N_imp = {sorted(kept)}"
    print(
        f"WARNING: the thermal state is taken at {held}, but a neighbouring sector lies "
        f"within the thermal window ({where} N_imp = {d.winner_total}, tau = {tau:.4g}); its Boltzmann "
        f"weight (> 1e-4) is not in the ground state: {why or _SINGLE_SECTOR_WHY}.",
        flush=True,
    )


def solve_ground_state(
    h_op,
    impurity_orbitals,
    bath_states,
    N0,
    *,
    tau,
    slaterWeightMin,
    frozen_occupations=None,
    chain_restrict=False,
    dense_cutoff=config.GS_DENSE_CUTOFF,
    comm=None,
    verbose=False,
    truncation_threshold=None,
    weighted_restrictions=None,
    use_hf_seed=True,
    de2_min=GS_DE2_MIN,
    e_pt2_tol=GS_E_PT2_TOL,
    plan_store=None,
    shift=0.0,
    spin_layout=None,
    warn_neighbours=True,
):
    """Find the thermal ground state: a sector walk, then one complete solve of the winning sector.

    **The single-sector definition of "the ground state of this model".** ``calc_gs`` and the
    double-counting criteria in :mod:`dc_criteria` go through :func:`solve_charge_sector_ensemble`,
    which with ``mixed_valence`` off is exactly this function's state (the walk's winner, one
    :func:`solve_member`); this function stays as that reference. A ``dc``
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

    ``plan_store`` (a :class:`~impurityModel.ed.sector_solver.SectorPlanStore`; a fresh one when
    ``None``) is shared by the walk and the thermal solve, so the winning sector is certified once,
    and by every ground state a caller sweeping ``H - shift * N_imp`` solves with it.

    Returns
    -------
    GroundState
        ``basis`` carries ``ground_state_occupation`` (the winning sector) and ``cap_policy``.
    """
    config.assert_uniform(comm)  # collective: a rank-divergent env knob is an error, not a deadlock
    _warn_frozen_leak(h_op, impurity_orbitals, frozen_occupations, comm)
    if plan_store is None:
        plan_store = SectorPlanStore()
    search = find_ground_state_sector(
        h_op,
        impurity_orbitals,
        bath_states,
        N0,
        frozen_occupations=frozen_occupations,
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
        plan_store=plan_store,
        shift=shift,
        spin_layout=spin_layout,
    )
    return solve_member(
        search,
        h_op,
        impurity_orbitals,
        bath_states,
        tau=tau,
        chain_restrict=chain_restrict,
        dense_cutoff=dense_cutoff,
        comm=comm,
        verbose=verbose,
        slaterWeightMin=slaterWeightMin,
        weighted_restrictions=weighted_restrictions,
        frozen_occupations=frozen_occupations,
        de2_min=de2_min,
        e_pt2_tol=e_pt2_tol,
        plan_store=plan_store,
        shift=shift,
        spin_layout=spin_layout,
        warn_neighbours=warn_neighbours,
    )


def sector_bounds(impurity_orbitals, frozen_occupations):
    """The total-N bounds of a model: ``(max_total, free_capacity, pinned_of)``.

    ``max_total`` is the whole-impurity capacity (the groups redistribute charge freely at fixed total, so this,
    not any per-group size, bounds an attainable total). ``free_capacity`` is the capacity of the groups that are
    not frozen. ``pinned_of(occupation)`` is the electron count the frozen groups hold in ``occupation``; a total
    ``t`` is attainable when ``0 <= t <= max_total`` and ``pinned <= t <= pinned + free_capacity``.
    """
    frozen = frozen_occupations or ()
    max_total = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)
    free_capacity = sum(len(block) for i, blocks in impurity_orbitals.items() if i not in frozen for block in blocks)

    def pinned_of(occupation):
        return sum(occupation[i] for i in occupation if i in frozen)

    return max_total, free_capacity, pinned_of


def _agree_on_switch(mixed_valence, comm):
    """Raise on every rank if the resolved ``mixed_valence`` differs between ranks (env knobs are checked elsewhere)."""
    if comm is not None and comm.size > 1:
        modes = comm.allgather(bool(mixed_valence))
        if len(set(modes)) != 1:
            raise RuntimeError(f"mixed_valence resolved differently on different ranks: {modes}")


def member_refusal(member, truncation, convergence, comm):
    """The reason a member's thermal solve may not enter a merge; ``None`` when it may.

    Admission is by energy (review S4): a cap-bound basis or a residual above ``e_pt2_tol`` no longer refuses a
    member, it widens the weight interval (:func:`~impurityModel.ed.sector_ensemble.weight_interval`) and is
    warned about. Only the measured-memory guard refuses: it held the basis below what any cap would, so the
    member's energy and weight are unreliable in a way the interval does not cover. ``truncation`` is the solver's
    raw ``truncation_report`` (``None`` when not capped), ``convergence`` its ``convergence_report``. The verdict
    is taken from rank 0's reports and broadcast; the broadcast is unconditional on every rank of ``comm``.
    """
    memory_bound = bool(truncation and truncation.get("memory_bound"))
    convergence = convergence or {}
    verdict = (
        memory_bound,
        convergence.get("converged") is False,
        convergence.get("residual_pt2"),
        convergence.get("limited_by"),
    )
    if comm is not None:
        verdict = comm.bcast(verdict, root=0)
    memory_bound, unconverged, residual, limited_by = verdict
    if memory_bound:
        return f"the N_imp = {member.total} member's thermal solve is held by the measured-memory guard"
    if unconverged and (comm is None or comm.rank == 0):
        detail = f"residual PT2 {residual:.2e}" if residual is not None else "no residual"
        if limited_by:
            detail += f", limited by {limited_by}"
        print(
            f"WARNING: the N_imp = {member.total} ensemble member's thermal solve is not converged ({detail}); "
            "it is merged, and the occupation weights carry the corresponding interval.",
            flush=True,
        )
    return None


def worst_truncation(reports):
    """The truncation report that describes an ensemble: a memory-bound member's, else a cap-bound one's.

    The dominant member (``reports[0]``) may be uncapped while a lighter member is cap- or memory-bound; the
    report that ``gs_info`` carries must not hide that (review MV7). One member: its own report, unchanged.
    """
    present = [r for r in reports if r is not None]
    if not present:
        return None
    memory = [r for r in present if r.get("memory_bound")]
    if memory:
        return memory[0]
    bound = [r for r in present if is_cap_bound(r)]
    return bound[0] if bound else present[0]


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


@dataclass
class MemberState:
    """One ensemble member as a visitor sees it: no solver, no Krylov store (they are dropped before the visit).

    ``truncation`` is :func:`refined_truncation_report` (the solve's own report, else the walk's) and
    ``convergence`` the solver's ``convergence_report``; both are rank-local solver state, read from the solver
    before it went. ``psis`` is the raw thermal block as the eigensolver returned it; :meth:`take_psis` hands it
    to its one consumer and clears the field, so the member does not keep a second reference to it.
    """

    basis: Basis
    es: np.ndarray
    psis: object
    occupation: dict
    truncation: dict | None
    convergence: dict | None

    def take_psis(self):
        """The raw thermal block; this member no longer references it afterwards."""
        psis, self.psis = self.psis, None
        return psis


class DrivenState(NamedTuple):
    """What a driver visitor of :func:`calc_gs` gets for one member: the thermal ``states`` it works on.

    ``states`` are the member's width-1 states redistributed onto ``basis`` and pruned -- the states the
    ground-state report was measured on, and the only copy of them that is alive (``len(es)`` of them on every
    rank). The other fields are those of :class:`MemberState`.
    """

    basis: Basis
    es: np.ndarray
    states: list
    occupation: dict
    truncation: dict | None
    convergence: dict | None


def release(ground_state):
    """``(MemberState, raw_truncation)`` of a :class:`GroundState`; the caller then drops it (and so the solver).

    ``raw_truncation`` is the solver's own ``truncation_report`` (``None`` when it was not capped), which
    :func:`member_refusal` needs unmixed with the walk's.
    """
    solver = ground_state.solver
    raw = getattr(solver, "truncation_report", None)
    state = MemberState(
        ground_state.basis,
        ground_state.es,
        ground_state.psis,
        ground_state.occupation,
        refined_truncation_report(solver, ground_state.walk_report),
        getattr(solver, "convergence_report", None),
    )
    return state, raw


class EnsembleResult(NamedTuple):
    """What :func:`solve_charge_sector_ensemble` returns. One order: the realised-lowest member first.

    ``members``, ``weights`` (``c_s``, summing to 1), ``realised`` (each member's lowest thermal energy) and
    ``visited`` (the visitor's return per member) are parallel. ``truncation`` is :func:`worst_truncation` over the
    members; ``convergence`` the dominant member's, demoted when another member is unconverged. For K >= 2 the
    members' ``basis`` objects are not kept: a visitor that needs one reads it in ``visit``.
    """

    members: tuple
    weights: np.ndarray
    realised: list
    visited: list
    refused_reason: str | None
    error_bar: float
    truncation: dict | None
    convergence: dict | None
    search: object
    #: ``(lo, hi)`` of ``<N_imp>`` over the members' energy uncertainty (``None`` for one member).
    n_imp_interval: tuple | None = None


def _ensemble_interval(members, realised, z, residuals, tau, comm):
    """The ``(lo, hi)`` interval of ``<N_imp>`` over the members' energy uncertainty; ``None`` for one member.

    Each member's true energy may lie below the realised one by ``ENSEMBLE_RESIDUAL_FACTOR`` times its residual
    PT2 (a calibrated factor, not a bound); an uncertified member is no better. The residuals are rank 0's,
    broadcast, so every rank returns the same interval.
    """
    if len(members) < 2:
        return None
    shifts = [ENSEMBLE_RESIDUAL_FACTOR * r for r in residuals]
    if comm is not None:
        shifts = comm.bcast(shifts, root=0)
    n = [m.total for m in members]
    return weight_interval(n, realised, z, tau, shifts)


def solve_charge_sector_ensemble(
    h_op,
    impurity_orbitals,
    bath_states,
    N0,
    *,
    mixed_valence,
    visitor,
    tau,
    slaterWeightMin,
    frozen_occupations=None,
    chain_restrict=False,
    dense_cutoff=config.GS_DENSE_CUTOFF,
    comm=None,
    verbose=False,
    truncation_threshold=None,
    weighted_restrictions=None,
    use_hf_seed=True,
    de2_min=GS_DE2_MIN,
    e_pt2_tol=GS_E_PT2_TOL,
    plan_store=None,
    shift=0.0,
    spin_layout=None,
    announce_refusal=True,
    warn_neighbours=True,
):
    """The thermal ground state: which charge sectors, then each member solved and visited, one at a time.

    **The single definition of "the ground state of this model".** ``calc_gs`` and the double-counting criteria
    both go through here, because a ``dc`` measured on a different state than the one ``calc_selfenergy`` later
    finds locks the downstream calculation onto the wrong charge state. Anything that decides *which* state is
    found belongs in this signature so the callers cannot disagree about it.

    Level 1 of three: this function decides the members (the sector walk, then
    :func:`~impurityModel.ed.sector_ensemble.plan_ensemble` when ``mixed_valence`` is on, else
    :func:`~impurityModel.ed.sector_ensemble.winner_plan`: the winner alone, through the same loop);
    :func:`solve_member` (level 2) solves each member's thermal manifold; ``visitor.visit(member, state)``
    (level 3) consumes it. The member's solver and Krylov store are dropped before the visit, so the peak is one
    member plus whatever the visitor keeps.

    When the plan is refused (a member is cap-bound, uncertified, or the weights are not fixed to
    ``ENSEMBLE_WEIGHT_ERROR_MAX``) the ensemble is the winner alone: a warning is printed (``announce_refusal``)
    and the run goes on. The same holds when a member's own thermal solve, checked before the visitor sees it,
    is cap-bound or unconverged, except that the ensemble is cut at that member rather than to the winner: the
    members before it, already visited, are kept. ``refused_reason`` says so.

    Returns
    -------
    EnsembleResult
    """
    config.assert_uniform(comm)  # collective: a rank-divergent env knob is an error, not a deadlock
    mixed_valence = bool(mixed_valence)
    _agree_on_switch(mixed_valence, comm)
    _warn_frozen_leak(h_op, impurity_orbitals, frozen_occupations, comm)
    if plan_store is None:
        plan_store = SectorPlanStore()
    max_total, free_capacity, pinned_of = sector_bounds(impurity_orbitals, frozen_occupations)
    pinned_box = [pinned_of(N0)]

    def attainable(total):
        pinned = pinned_box[0]
        return 0 <= total <= max_total and pinned <= total <= pinned + free_capacity

    def needs_of(solved):
        pinned_box[0] = pinned_of(solved[0].trial_N0) if solved else pinned_box[0]
        return plan_ensemble(solved, tau, attainable=attainable).needs

    search = find_ground_state_sector(
        h_op,
        impurity_orbitals,
        bath_states,
        N0,
        frozen_occupations=frozen_occupations,
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
        plan_store=plan_store,
        shift=shift,
        spin_layout=spin_layout,
        more_totals=needs_of if mixed_valence else None,
    )
    winner_total = sum(search.occupation.values())
    pinned_box[0] = pinned_of(search.occupation)
    if mixed_valence:
        plan = plan_ensemble(search.solved, tau, attainable=attainable)
        if plan.needs and plan.refused_reason is None:
            # An edge that could not be solved leaves the plan provisional, and a provisional plan has skipped the
            # cap / certification / error-bar checks: refuse rather than merge unchecked. (The winner is the lowest
            # sector, so it is always admitted; should it ever not be, keep the dominant member rather than an
            # empty plan, which would crash every rank in the loop below.)
            plan = dataclasses.replace(
                plan,
                members=tuple(m for m in plan.members if m.total == winner_total) or plan.members[:1],
                refused_reason=f"the edge sector(s) {list(plan.needs)} could not be solved",
                needs=(),
            )
        if plan.refused_reason is not None and announce_refusal and (comm is None or comm.rank == 0):
            print(
                f"WARNING: mixed_valence: keeps the single N_imp = {plan.members[0].total} "
                f"sector: {plan.refused_reason}.",
                flush=True,
            )
    else:
        plan = winner_plan(search.solved, winner_total, tau)
    records = {r.total: r for r in search.solved}
    kw = dict(
        tau=tau,
        chain_restrict=chain_restrict,
        dense_cutoff=dense_cutoff,
        comm=comm,
        verbose=verbose,
        slaterWeightMin=slaterWeightMin,
        weighted_restrictions=weighted_restrictions,
        frozen_occupations=frozen_occupations,
        de2_min=de2_min,
        e_pt2_tol=e_pt2_tol,
        plan_store=plan_store,
        shift=shift,
        spin_layout=spin_layout,
    )
    members = plan.members
    refused_reason = plan.refused_reason
    visited, realised, z, truncations, convergences = [], [], [], [], []
    single = len(plan.members) == 1
    for index, member in enumerate(plan.members):
        if member.total == winner_total:
            member_search = search
        else:
            member_search = search._replace(
                occupation=dict(member.trial_N0),
                truncation_report=records[member.total].truncation_report,
                diagnostics=None,
            )
        ground_state = solve_member(
            member_search, h_op, impurity_orbitals, bath_states, warn_neighbours=single and warn_neighbours, **kw
        )
        state, raw_truncation = release(ground_state)
        del ground_state  # the solver and its Krylov store go before the visitor's own workspace is built
        bad = None if single else member_refusal(member, raw_truncation, state.convergence, comm)
        if bad is not None:
            # The walk certified the plan; the member's own thermal solve did not hold up (review H-MV11). Decided
            # on rank 0's reports and broadcast, so every rank takes the same branch. The ensemble is cut at this
            # member: the members before it (the lower sectors, already visited) stay, so no consumer's work is
            # thrown away (review 7b, owner decision); this one and those after it go. A failing winner is still a
            # valid one-member ensemble, so it is visited alone.
            reason = bad
            kept_members = plan.members[: max(index, 1)]
            if announce_refusal and (comm is None or comm.rank == 0):
                held = (
                    f"the single N_imp = {kept_members[0].total} sector"
                    if len(kept_members) == 1
                    else f"the N_imp = {[m.total for m in kept_members]} members solved before it"
                )
                print(f"WARNING: mixed_valence: keeps {held}: {reason}.", flush=True)
            if warn_neighbours:
                # The kept members were solved as ensemble members, without the single-sector note (review 7c).
                _warn_near_neighbours(search, tau, comm, kept_totals=[m.total for m in kept_members], why=reason)
            members, refused_reason = kept_members, reason
            if visited:
                del state
                break
        es = np.asarray(state.es)
        realised.append(float(es.min()))
        z.append(float(np.exp(-(es - es.min()) / tau).sum()))
        truncations.append(state.truncation)
        convergences.append(state.convergence)
        visited.append(visitor.visit(member, state))
        del state, es
        if bad is not None:
            break  # the winner, kept alone
    # The realised energies and partition functions are the eigensolver's, rank-local: a near tie must not order
    # (or weight) the members differently on different ranks. One broadcast, unconditional on every rank.
    if comm is not None:
        realised, z = comm.bcast((realised, z), root=0)
    # The walk ranked the sectors by a lowest level solved with the walk's own (looser) basis. The thermal solves
    # may find another member lowest; the dominant member is the realised lowest (review H-MV6), so this is a
    # note, not an error. Computed before the sort.
    if int(np.argmin(realised)) != 0 and (comm is None or comm.rank == 0):
        print(
            "NOTE: the thermal solves rank the charge sectors differently from the walk (realised lowest "
            f"energies {['%.6f' % e for e in realised]} for totals {[m.total for m in members]}); "
            "the dominant member is the realised lowest.",
            flush=True,
        )
    order = [int(k) for k in np.argsort(np.asarray(realised), kind="stable")]  # stable: one order everywhere
    members = tuple(members[k] for k in order)
    realised = [realised[k] for k in order]
    z = [z[k] for k in order]
    visited = [visited[k] for k in order]
    truncations = [truncations[k] for k in order]
    convergences = [convergences[k] for k in order]
    weights = sector_weights(np.array(realised), np.array(z), tau)
    converged = [(c or {}).get("converged") is not False for c in convergences]
    truncations_residuals = [float((c or {}).get("residual_pt2") or 0.0) for c in convergences]
    convergence = dict(convergences[0]) if convergences[0] is not None else None
    if convergence is not None and len(members) > 1 and not all(converged):
        convergence["converged"] = False
        convergence.setdefault("limited_by", None)
        if convergence["limited_by"] is None:
            bad_totals = [m.total for m, ok in zip(members, converged) if not ok]
            convergence["limited_by"] = f"ensemble member N_imp = {bad_totals[0]}"
    interval = _ensemble_interval(members, realised, z, truncations_residuals, tau, comm)
    n_avg = float(weights @ np.array([m.total for m in members], dtype=float))
    wide = interval is not None and interval[1] - interval[0] > config.ENSEMBLE_WEIGHT_ERROR_MAX
    if wide and announce_refusal and (comm is None or comm.rank == 0):
        print(
            f"WARNING: the occupation <N_imp> = {n_avg:.3f} is fixed only to [{interval[0]:.3f}, "
            f"{interval[1]:.3f}] e: the members' energies carry truncation error (residual PT2 times "
            f"{ENSEMBLE_RESIDUAL_FACTOR:g}, calibrated, not a bound; tau = {tau:.3g}). Raise the determinant "
            "cap (truncation_threshold) to narrow it.",
            flush=True,
        )
    return EnsembleResult(
        members,
        weights,
        realised,
        visited,
        refused_reason,
        plan.error_bar,
        worst_truncation(truncations),
        convergence,
        search,
        interval,
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


def calc_gs(
    Hop: ManyBodyOperator,
    basis_setup: dict,
    block_structure: BlockStructure,
    rot_to_spherical: np.ndarray,
    verbose,
    slaterWeightMin=0,
    stats_path: str = "ground_state_statistics.json",
    shell_ls=None,
    spec: Optional[GroundStateSpec] = None,
    visitor=None,
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
            stats_path,
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
