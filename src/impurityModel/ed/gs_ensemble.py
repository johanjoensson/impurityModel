"""The charge-sector ensemble: one ground-state search, then every member sector the plan keeps.

:func:`solve_charge_sector_ensemble` walks the charge sectors (via
:func:`~impurityModel.ed.groundstate.find_ground_state_sector`), plans the members, solves each, and hands every finished member to a visitor. The per-member state
(:class:`MemberState`, :class:`DrivenState`), the refusal rule (:func:`member_refusal`) and the
truncation roll-up live here; :mod:`~impurityModel.ed.gs_calc` drives it for the selfenergy/spectra solve.
"""

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from impurityModel.ed import config
from impurityModel.ed.groundstate import (
    GS_DE2_MIN,
    GS_E_PT2_TOL,
    _warn_frozen_leak,
    _warn_near_neighbours,
    find_ground_state_sector,
    sector_bounds,
    solve_member,
)
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.sector_ensemble import (
    ENSEMBLE_RESIDUAL_FACTOR,
    plan_ensemble,
    sector_weights,
    weight_interval,
    winner_plan,
)
from impurityModel.ed.sector_solver import SectorPlanStore
from impurityModel.ed.truncation_flags import is_cap_bound


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
