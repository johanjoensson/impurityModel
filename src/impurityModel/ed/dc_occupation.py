"""The occupation criterion: :func:`fixed_occupation_dc` pins the impurity occupation (default: the DFT reference).

Its own context (:class:`_OccupationContext`) is built on :mod:`~impurityModel.ed.dc_sector_context`.
"""

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config, dc_record, solver_trace
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_restrictions import build_weighted_restrictions
from impurityModel.ed.dc_reference import (
    _SATURATION_ADVICE,
    _noninteracting_impurity_occupation,
    _warn_if_not_fermi_referenced,
    _warn_if_reference_far_from_nominal,
    _warn_if_reference_saturated,
)
from impurityModel.ed.dc_search import (
    DoubleCountingUnreachable,
    _dc_chi,
    _dc_search_trace,
    _solve_dc_shift,
    bracket_width_tol,
)
from impurityModel.ed.dc_sector_context import (
    _active_nominal,
    _active_shift,
    _calibrate_cap,
    _check_active_shells,
    _ctx_spec,
    _dump_dc_matrices,
    _mu_tol_effective,
    _reject_if_memory_bound,
    _resolve_dc_de2_min,
    _resolve_dc_e_pt2_tol,
)
from impurityModel.ed.density_matrices import occupation_moments
from impurityModel.ed.gs_connectivity import solver_basis_connectivity
from impurityModel.ed.lie_algebra import extract_tensors, tensors_to_operator
from impurityModel.ed.memory_estimate import CapPolicy, resolve_cap_policy
from impurityModel.ed.sector_ensemble import resolve_mixed_valence
from impurityModel.ed.sector_solver import SectorPlanStore
from impurityModel.ed.solver_basis import prepare_solver_basis
from impurityModel.ed.truncation_flags import cap_ever_hit


@dataclass
class _OccupationContext:
    """The mu-independent setup a fixed-occupation evaluation needs, shared between
    :func:`fixed_occupation_dc`'s search and the single-mu diagnostic
    :func:`occupation_and_energy_at_mu` (B2's convergence sweep,
    ``test/support/dc_diagnostics.py``): the solver-basis layout, symmetry
    generators, the derived determinant cap, and the weighted restrictions. Building this once
    and evaluating many ``mu`` against it (the search) or one ``mu`` (the diagnostic) keeps both
    on the identical code path -- the diagnostic is not a reimplementation of the search's setup.
    """

    h0_op: object
    h_op_i: object
    impurity_orbitals: dict
    bath_states: dict
    N0: dict
    nominal_active: float
    tau: float
    chain_restrict: bool
    dense_cutoff: int
    slaterWeightMin: float
    truncation_threshold: int
    #: As :class:`_SectorContext`: only a cap that defaulted may be recalibrated against the
    #: search's tolerance, and the memory-derived value is kept as the parity reference.
    memory_cap: object
    cap_from_memory: bool
    weighted_restrictions: object
    #: Orbitals whose occupation is measured and shifted (the active shells; every impurity orbital on one shell).
    impurity_indices: list
    shift_input: np.ndarray
    shift_solver: np.ndarray
    total_impurity_orbitals: int
    n0: float
    #: Residual PT2 energy the ground-state determinations converge to: the solver's own
    #: ``BasisOptions.e_pt2_tol`` (else :data:`groundstate.GS_E_PT2_TOL`) and nothing else. This
    #: criterion solves on the production ground-state path, so it takes the self-energy run's
    #: tolerance rather than an override -- the double-counting line
    #: does not reach it, for the same reason ``de2_min`` does not.
    e_pt2_tol: float
    #: The per-determinant PT2 floor of the same solves, from ``BasisOptions.de2_min`` (else
    #: :data:`groundstate.GS_DE2_MIN`) and nothing else, for the same parity reason.
    de2_min: float
    #: ``mu -> bool``: did ``truncation_threshold`` bind the ground-state determination at that
    #: shift? Filled by :func:`_evaluate_occupation_and_energy_at_mu`, and **broadcast** there, so
    #: this holds a rank-replicated fact rather than one rank's view of its own solver -- the same
    #: contract :class:`_SectorContext` states for ``sector_at``.
    #:
    #: Recorded on the context rather than returned, because the public
    #: :func:`occupation_and_energy_at_mu` has a documented ``(n, E0, sector)`` return and this is
    #: not part of the answer -- it qualifies it.
    #: The resolved ``mixed_valence``: off measures the winning charge sector alone, on the grand-canonical
    #: ensemble of the sectors inside the thermal window.
    mixed_valence: bool = False
    #: ``mu -> (intra, inter)``: the variance of the impurity occupation inside the members and between them.
    ensemble_variance_at: dict | None = None
    #: ``mu -> reason``: trial points whose ensemble was refused (replicated). Such a point carries the winning
    #: sector alone and is flagged through ``cap_bound_at``; it is reported once per search, not per trial.
    refused_at: dict | None = None
    #: ``mu -> K``: how many members the ensemble had at each trial point (replicated; 1 when off).
    members_at: dict | None = None
    #: ``mu -> (lo, hi)``: the interval of the ensemble's <N_imp> at each trial point (replicated; absent when off).
    interval_at: dict | None = None
    cap_bound_at: dict | None = None
    memory_bound_at: dict | None = None
    #: Each sector's certification, shared by every ground state this search solves
    #: (:class:`sector_solver.SectorPlanStore`).
    plan_store: object = None
    #: The (dn, up) pairing of the solver basis, as on :class:`_SectorContext`.
    spin_layout: object = None
    #: As :class:`_SectorContext`.
    seed_window: object = None
    #: As :class:`_SectorContext`: where the cap ladder records the cap it settled on (P10).
    cap_ledger: object = None

    def __post_init__(self):
        if self.cap_bound_at is None:
            self.cap_bound_at = {}
        if self.refused_at is None:
            self.refused_at = {}
        if self.members_at is None:
            self.members_at = {}
        if self.interval_at is None:
            self.interval_at = {}
        if self.memory_bound_at is None:
            self.memory_bound_at = {}
        if self.plan_store is None:
            self.plan_store = SectorPlanStore()


def _prepare_occupation_context(model, basis, solver, comm=None, verbosity=0, cap_ledger=None):
    """Build the :class:`_OccupationContext` for a fixed-occupation evaluation.

    Verbatim extraction of the setup :func:`fixed_occupation_dc` used to build inline: the same
    solver-basis derivation, symmetry-generator computation, and determinant-cap
    derivation :func:`groundstate.calc_gs` uses, so a fixed-mu diagnostic built on this context
    measures on exactly the variational space the production search would have used at that
    ``mu`` -- not an approximation of it.
    """
    h0_op = model.h0
    u = model.u4
    n_imp = len(model.impurity_indices)
    N0 = basis.nominal_occ
    mixed_valence, _mv_note = resolve_mixed_valence(basis.mixed_valence, driver_default="on")
    tau = basis.tau
    chain_restrict = basis.chain_restrict
    slaterWeightMin = basis.slater_weight_min
    dense_cutoff = solver.dense_cutoff
    verbose = verbosity > 0

    # Derive the same solver-basis layout calc_selfenergy uses (the DC search must measure its
    # observable on the same sector/basis the downstream selfenergy solve will use, or "fixed"
    # does not mean what it says). The mu shift is a uniform impurity shift (-mu * P, the projector onto the
    # hybridising shells; the identity on a single shell),
    # which commutes with the impurity block, so the layout derived once here at the guess dc is valid for every trial mu.
    sb = prepare_solver_basis(
        h0_op,
        model.dc,
        u,
        model.impurity_orbitals,
        N0,
        rot_to_spherical=model.rot_to_spherical,
        verbosity=verbosity,
        rank=comm.rank if comm is not None else 0,
        comm=comm if comm is not None else MPI.COMM_WORLD,
        shell_ls=model.shell_ls,
        interaction_merges_blocks=basis.interaction_merges_blocks,
        bath_split=model.bath_states,
    )
    _check_active_shells(
        sb,
        comm.rank if comm is not None else MPI.COMM_WORLD.rank,
        verbosity,
        comm if comm is not None else MPI.COMM_WORLD,
    )
    impurity_orbitals = sb.impurity_orbitals
    bath_states = sb.bath_states
    shift_input, shift_solver, active_indices = _active_shift(model, sb, n_imp)
    N0 = sb.nominal_occ
    h_op_i = sb.h

    total_impurity_orbitals = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)
    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]
    if active_indices is not None:
        impurity_indices = list(active_indices)
        total_impurity_orbitals = len(impurity_indices)

    # Nothing is frozen: calc_selfenergy freezes nothing, and the dc must be measured on the state it
    # solves (review J1; see the module warning).

    truncation_threshold = basis.truncation_threshold
    # As `_prepare_sector_context`: provenance decides whether the cap may be recalibrated.
    # Recalibration (the cap ladder) is only for a cap the caller left to memory; `unlimited` and
    # a set number are instructions. Broadcast, though derived from a replicated input.
    cap_from_memory = MPI.COMM_WORLD.bcast(CapPolicy.coerce(truncation_threshold).source == "auto", root=0)
    # The one resolver every driver uses: the same ground-state-only sizing, safety and
    # GS_NUM_WANTED as calc_selfenergy, so the dc is found at the determinant budget the
    # self-energy then solves with; it prints the `determinant cap:` line and starts a new
    # calculation's memory-warning count. Collective on COMM_WORLD, unconditional.
    policy, _ = resolve_cap_policy(
        truncation_threshold,
        model.n_spin_orbitals,
        comm=MPI.COMM_WORLD,
        verbose=verbose,
        label="fixed-occupation dc",
        log="derived",
        connectivity=solver_basis_connectivity(sb, MPI.COMM_WORLD),
        ledger=cap_ledger,
    )
    truncation_threshold = policy.gs
    memory_cap = policy.gs

    # DFT reference occupation: Fermi filling of the raw h0 (the KS Hamiltonian of the
    # h0 - dc + U contract; no double counting subtracted before filling), independent of
    # dc_guess and of the trial shift, and of the solver-basis rotation (h0's raw filling is
    # evaluated in the model's input basis).
    # h0_op is in the input basis, so the active orbitals are counted by their input-basis indices (the diagonal
    # of ``shift_input``): the solver-basis labels in ``impurity_indices`` can be permuted against them (review MS3).
    n0_indices = impurity_indices
    if active_indices is not None:
        n0_indices = [orb for orb, on in zip(sorted(model.impurity_indices), np.diag(shift_input)) if on]
    n0 = _noninteracting_impurity_occupation(h0_op, n0_indices, model.n_spin_orbitals, tau)

    return _OccupationContext(
        h0_op=h0_op,
        h_op_i=h_op_i,
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        N0=N0,
        nominal_active=_active_nominal(sb),
        tau=tau,
        chain_restrict=chain_restrict,
        dense_cutoff=dense_cutoff,
        slaterWeightMin=slaterWeightMin,
        truncation_threshold=truncation_threshold,
        memory_cap=memory_cap,
        cap_from_memory=cap_from_memory,
        cap_ledger=cap_ledger,
        spin_layout=sb.spin_layout,
        plan_store=SectorPlanStore(active_orbitals=sb.active_impurity_indices),
        seed_window=sb.seed_window,
        weighted_restrictions=build_weighted_restrictions(
            bath_states,
            cost_budget=basis.cost_budget,
            h_op=h_op_i,
            impurity_orbitals=impurity_orbitals,
            slater_weight_min=slaterWeightMin,
        ),
        impurity_indices=impurity_indices,
        shift_input=shift_input,
        shift_solver=shift_solver,
        total_impurity_orbitals=total_impurity_orbitals,
        n0=n0,
        e_pt2_tol=_resolve_dc_e_pt2_tol(None, basis.e_pt2_tol),
        de2_min=_resolve_dc_de2_min(None, basis.de2_min),
        mixed_valence=mixed_valence,
    )


class _OccupationVisitor:
    """Level 3 of ``solve_charge_sector_ensemble`` for the occupation search: one member's moments and verdicts."""

    def __init__(self, ctx):
        self.ctx = ctx

    def visit(self, member, state):
        ctx = self.ctx
        occ, occ2 = occupation_moments(state.basis, state.psis, ctx.impurity_indices)
        es = np.asarray(state.es)
        return (
            float(thermal_average_scale_indep(es, occ, ctx.tau)),
            float(es.min()),
            float(np.exp(-(es - es.min()) / ctx.tau).sum()),
            float(thermal_average_scale_indep(es, occ2, ctx.tau)),
            # The final refinement's report or the walk's: either says the cap bound this determination.
            cap_ever_hit(state.truncation),
            bool((state.truncation or {}).get("memory_bound", False)),
        )


def _evaluate_occupation_and_energy_at_mu(ctx, mu, verbose, rank):
    """Evaluate the true, re-expanding occupation observable at one ``mu``: the identical
    HF-seed-then-walk :func:`groundstate.find_ground_state_sector` search, refinement, and thermal
    average :func:`fixed_occupation_dc` performs per trial shift -- not a cheaper approximation.
    Returns ``(n, E0, sector)``: the achieved thermal impurity occupation, the lowest eigenvalue
    found, and the *walk's* winning sector total (``sum(mb_basis.ground_state_occupation.values())``
    -- the integer charge state the search settled on, which need not equal ``round(n)`` when the
    thermal average mixes competing sectors near a crossing).

    .. note::
       The observable's own collectives run on ``MPI.COMM_WORLD``, matching
       :func:`fixed_occupation_dc`'s pre-existing behaviour (not the caller's ``comm`` -- a latent,
       out-of-scope concern recorded in the campaign plan, not introduced here).
    """
    from impurityModel.ed.gs_ensemble import solve_charge_sector_ensemble

    h_op = ctx.h_op_i - tensors_to_operator(np.asarray(mu * ctx.shift_solver, dtype=complex))
    # One evaluation for both settings of `mixed_valence`: with it off the entry's ensemble is the walk's
    # winner alone, so the thermal average is the single sector's, exactly what `calc_selfenergy` will find.
    # N0 stays the *original* nominal occupation on every call (never a carried-forward sector from the
    # previous mu): feeding the walk its own last answer would make the result path-dependent on the
    # bracket's evaluation order.
    with solver_trace.timed("expand", stage="dc_refine"):
        result = solve_charge_sector_ensemble(
            h_op,
            ctx.impurity_orbitals,
            ctx.bath_states,
            ctx.N0,
            # A trial point whose ensemble is refused is a flagged point (a re-ranked one is only printed),
            # not a failed search. The refusal reason is replicated, so every rank flags the point alike.
            mixed_valence=ctx.mixed_valence,
            visitor=_OccupationVisitor(ctx),
            slaterWeightMin=ctx.slaterWeightMin,
            comm=MPI.COMM_WORLD,
            verbose=verbose,
            weighted_restrictions=ctx.weighted_restrictions,
            **_ctx_spec(ctx).solve_kwargs(),
            plan_store=ctx.plan_store,
            shift=mu,
            spin_layout=ctx.spin_layout,
            frozen_occupations=ctx.seed_window,
            announce_refusal=False,
            warn_neighbours=False,
        )
    n_s, _e0_s, _z_s, n2_s, bound, memory = (np.array(col) for col in zip(*result.visited))
    c = result.weights
    n_avg = float(c @ n_s)
    if ctx.mixed_valence:
        if ctx.ensemble_variance_at is None:
            ctx.ensemble_variance_at = {}
        ctx.ensemble_variance_at[mu] = (float(c @ (n2_s - n_s**2)), float(c @ (n_s - n_avg) ** 2))
    # Broadcast before it is stored: the verdicts are rank-local solver state, and the cap strategy branches on
    # them to decide whether to run a second, collective evaluation.
    ctx.cap_bound_at[mu] = MPI.COMM_WORLD.bcast(bool(bound.any()), root=0)
    ctx.members_at[mu] = len(result.members)
    if result.n_imp_interval is not None:
        ctx.interval_at[mu] = (float(result.n_imp_interval[0]), float(result.n_imp_interval[1]))
    if result.refused_reason is not None:
        ctx.refused_at[mu] = result.refused_reason
    else:
        ctx.refused_at.pop(mu, None)
    ctx.memory_bound_at[mu] = MPI.COMM_WORLD.bcast(bool(memory.any()), root=0)
    # The sector of the point is the member that realised the lowest energy (review H-MV6): the result's first.
    return n_avg, float(min(result.realised)), result.members[0].total


def _member_flip_collapse(ctx, bracket):
    """Whether a collapsed bracket of the sector-first occupation search lies between two formed ensembles.

    Then the step is a re-ranking of the members, not a charge-sector step: both endpoints carry the
    grand-canonical average over at least two members, so ``n(mu)`` is continuous between them and a
    continuous search can reach a target inside the step (review H-MV12). A refused endpoint carries the
    winning sector alone, so its step is a real one, and a retry would only repeat the miss at a finer
    resolution.
    """
    if not ctx.mixed_valence or bracket is None or not ctx.members_at:
        return False
    # The search reports the bracket's raw endpoints; the observable ran at their snapped shifts, the nearest
    # evaluated points.
    ends = [min(ctx.members_at, key=lambda trial: abs(trial - mu)) for mu in bracket]
    return all(ctx.members_at[mu] >= 2 and mu not in ctx.refused_at for mu in ends)


def occupation_and_energy_at_mu(model, basis, solver, mu, *, comm=None, verbosity=0):
    """Evaluate the true occupation observable at **one fixed** ``mu`` -- no search.

    This is the per-``mu`` core of :func:`fixed_occupation_dc`, exposed standalone for B2's
    convergence diagnostic (``test/support/dc_diagnostics.py``'s ``occupation_convergence``
    mode): does the achieved occupation and ground-state energy drift with
    ``BasisOptions.truncation_threshold``/``chain_restrict`` at a fixed
    ``mu``? If ``n`` is not converged with respect to the ED's own variational truncation, the
    search absorbs that drift into ``dc`` rather than into physics -- a fixed-``mu`` sweep across
    the cap ladder is the cheap way to check, since it needs no search at all (Haule 2015's
    truncation-as-fitting-parameter pathology).

    Parameters
    ----------
    mu : float
        The fixed trial shift; ``H(mu) = H(0) - mu * N_imp``.

    Returns
    -------
    (n, E0, sector) : tuple
        Achieved thermal impurity occupation, the lowest eigenvalue, and the walk's winning
        sector total at this ``mu`` (see :func:`_evaluate_occupation_and_energy_at_mu`).
    """
    config.assert_uniform(MPI.COMM_WORLD)  # collective: a rank-divergent env knob is an error, not a deadlock
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0
    ctx = _prepare_occupation_context(model, basis, solver, comm=comm, verbosity=verbosity)
    return _evaluate_occupation_and_energy_at_mu(ctx, mu, verbose, rank)


def fixed_occupation_dc(
    model,
    basis,
    solver,
    *,
    occupation=None,
    comm=None,
    verbosity=0,
    occ_tol=1e-2,
    initial_step=0.25,
    max_shift=20.0,
    return_sector=False,
    report=None,
    allow_memory_bound=False,
    cap_ledger=None,
):
    r"""
    Calculate the double counting correction using a fixed (or self-consistent) impurity
    occupation criterion.

    With an explicit ``occupation``, choose the double counting so that the interacting thermal
    occupation of the hybridising shells equals the requested value, :math:`\mathrm{Tr}\,\rho_{act} =
    N_{target}` (the whole impurity on a single shell; a deep bath-less shell is not counted, and a target
    outside ``[0, number of active orbitals]`` is refused).

    With ``occupation=None``, pin it to the DFT reference occupation instead: :math:`N(\mu) =
    N_0`, where :math:`N_0` is the Fermi filling of the *raw* non-interacting ``h0`` at
    :math:`\mu_{chem} = 0` (:func:`_noninteracting_impurity_occupation`), computed once before
    the search -- the DFT impurity occupation the CSC DFT+DMFT self-consistency loop targets.
    ``h0`` is the KS/DFT Hamiltonian of the ``h0 - dc + U`` contract, so no double counting is
    subtracted before filling: :math:`N_0` is independent of both ``dc_guess`` and the trial
    shift. Note that :math:`N_0` is a property of the *discretized* bath: a coarse valence-only
    fit may place no impurity spectral weight across the Fermi level at all, saturating
    :math:`N_0` at the full (or empty) shell -- a threshold effect of the discretization, not a
    smooth error. A warning is printed in that case; supply an explicit ``occupation`` target
    instead.

    The double counting is parametrized as a uniform shift of the guess, ``dc(mu) = dc_guess +
    mu * P``, coupling to the occupation of the shifted (hybridising) shells as :math:`-\mu \hat N_{act}`
    (``P`` is the identity on a single shell, so :math:`N_{act} = N_{imp}`), so the
    residual :math:`N(\mu) - N_{target}` is monotone in ``mu`` for either target and is driven
    to zero by the search :func:`_solve_dc_shift`. At low temperature and weak
    hybridization the occupation approaches a staircase in ``mu``; if the requested (fractional)
    occupation falls on a plateau -- the observable steps across the target at a charge-sector
    boundary rather than crossing it -- the search reports the jump, the boundary and the
    distance to it on rank 0, then raises :class:`dc_search.DoubleCountingUnreachable` (B3: a
    target with no solution must be signalled as such to the caller, not merely narrated and
    silently approximated).

    Unlike :func:`fixed_peak_dc`, the sector this search measures the occupation on is **not**
    fixed at the input ``model.impurity_orbitals``/``model.bath_states``: at every trial ``mu``
    it derives the same solver-basis layout :func:`calc_selfenergy` uses
    (:func:`solver_basis.prepare_solver_basis` -- the impurity/bath grouping, the bath valence/
    conduction split from ``model.bath_states`` (the sign of the bath on-site energy when it is ``None``),
    and any symmetry-adapted rotation), then determines the ground-state sector by calling
    :func:`groundstate.find_ground_state_sector` itself -- the identical HF-seed-then-walk search
    :func:`groundstate.calc_gs` uses for the selfenergy/spectra solve, not a cheaper
    approximation of it. A dc measured on a sector different from the one calc_selfenergy later
    finds would not just be imprecise: it would lock the downstream calculation onto the wrong
    charge state, so this search pays the full per-trial cost of walking rather than caching a
    single Hartree-Fock seed across mu. This is what makes the "fixed" occupation the same
    quantity :func:`groundstate.find_ground_state_sector` finds for the selfenergy/spectra solve
    at the returned ``dc`` -- the point of this whole search. The search follows wherever that
    walk leads, including across a change in total electron number (matching
    ``find_ground_state_sector``'s own grand-canonical-style sector search); the reachable
    occupations are limited by the bath, and a tight
    ``BasisOptions.chain_restrict`` can itself make a requested occupation
    unreachable, or reachable only via a sector the un-walked search could never have found.

    Parameters other than the following match :func:`fixed_peak_dc` (``model``, ``basis``,
    ``solver``, ``comm``, ``verbosity``); neither function requires ``model.bath_states`` (the
    valence/conduction split used by both is the one derived at solve time, not the hybridization
    fit). Unlike :func:`fixed_peak_dc`, ``basis.truncation_threshold=None`` here derives the cap
    without halving the safety fraction -- this search does not hold an upper and a lower sector
    basis simultaneously.

    Parameters
    ----------
    occupation : float or None
        Requested impurity occupation (may be fractional). ``None`` targets the DFT reference
        occupation :math:`N_0` instead (see above).
    occ_tol : float
        Convergence tolerance on the occupation.
    initial_step : float
        First bracketing step for ``mu``, in the energy units of the
        Hamiltonian (energies here carry no fixed unit, they follow the
        inputs -- e.g. Ry when called from RSPt). A small fraction of the
        bandwidth is a good choice.
    max_shift : float
        Bracketing gives up if ``|mu|`` exceeds this, in the energy units of
        the Hamiltonian (the requested occupation is then unreachable).

    Returns
    -------
    dc : ndarray
        The double counting matrix, ``dc_guess + mu * P``.

    Raises
    ------
    dc_search.DoubleCountingUnreachable
        If the target cannot be bracketed within ``max_shift``, or the requested occupation falls
        on a charge-sector plateau the observable steps across rather than crosses (B3: this is a
        modelling verdict -- "no solution here" -- not a tolerable miss to approximate silently).
    """
    config.assert_uniform(MPI.COMM_WORLD)  # collective: a rank-divergent env knob is an error, not a deadlock
    with dc_record.recording(
        "occupation", rank=comm.rank if comm is not None else MPI.COMM_WORLD.rank, report=report
    ) as dc_rec:
        n_imp = len(model.impurity_indices)
        dc_guess = extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0]
        rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
        verbose = verbosity > 0

        ctx = _prepare_occupation_context(model, basis, solver, comm=comm, verbosity=verbosity, cap_ledger=cap_ledger)

        self_consistent = occupation is None
        if self_consistent:
            occupation = ctx.n0
            # Only when N0 is the *target*: with an explicit target a saturated reference is merely
            # logged, not acted on.
            # The E_F = 0 convention underpins both the reference filling below and every sector
            # comparison the occupation walk makes; it is asserted throughout and was never checked.
            _warn_if_not_fermi_referenced(ctx.h0_op, model.n_spin_orbitals, rank=rank)
            # Saturation is the more specific diagnosis, so it wins when both would fire; the
            # nominal-gap check catches the reference that is grossly wrong *without* being pinned at
            # a shell edge, which the saturation test alone lets through (a runaway CSC iterate
            # reporting n0 = 1.54 against a nominal 8 converged silently to a 43 eV shift).
            if not _warn_if_reference_saturated(
                ctx.n0, ctx.total_impurity_orbitals, _SATURATION_ADVICE["search"], occ_tol=occ_tol, rank=rank
            ):
                _warn_if_reference_far_from_nominal(ctx.n0, ctx.nominal_active, _SATURATION_ADVICE["search"], rank=rank)
        # Tolerance absorbs the roundoff of a target derived elsewhere the same way
        # _noninteracting_impurity_occupation is (a sum of Fermi occupations, each in [0, 1]).
        if not -1e-9 <= occupation <= ctx.total_impurity_orbitals + 1e-9:
            raise ValueError(f"Requested impurity occupation {occupation} outside [0, {ctx.total_impurity_orbitals}].")

        occupation_at = {}
        # The walk's winning sector at each evaluated mu (M1): the sector at the RETURNED mu is the
        # number this whole branch exists to establish -- DC <-> GS parity is a statement about which
        # charge state the downstream calc_selfenergy will find. find_ground_state_sector already
        # returns it (.ground_state_occupation), previously discarded here.
        sector_at = {}

        def occupation_observable(mu):
            with solver_trace.labelled(mu=mu), solver_trace.timed("dc_evaluation") as evaluation_fields:
                n_out, _e0, sector = _evaluate_occupation_and_energy_at_mu(ctx, mu, verbose, rank)
                occupation_at[mu] = n_out
                sector_at[mu] = sector
                if verbose and rank == 0:
                    # The sector itself is reported by find_ground_state_sector's own verbose output
                    # above ("Ground state occupation"), evaluated fresh at this mu.
                    print(f"mu={mu:.6f} n={n_out:.6f} n0={ctx.n0:.6f}")
                evaluation_fields["n"] = n_out
                return n_out

        # The same bracket-collapse resolution _solve_dc_shift uses below (width_tol); computed once
        # so the trace's chi report filters against the identical value, not a second copy of it.
        continuous = False  # sector-first; the continuous search is the fallback below
        # Continuous mode: n(mu) is smooth over ~tau, so a bracket as wide as tau would read a steep but
        # continuous step as a discontinuity. Resolve mu to the accuracy that pins n to occ_tol.
        continuous_width_tol = max(bracket_width_tol(ctx.tau) * occ_tol, 1e-6)
        occ_width_tol = continuous_width_tol if continuous else bracket_width_tol(ctx.tau)

        target = occupation
        if self_consistent:
            unreachable = (
                f"Could not reach the DFT reference occupation N0 = {ctx.n0:.4f} with |mu| <= "
                f"{max_shift}: " + "the occupation reached {value:.4f} at mu = {mu:.3f}. "
                "The target may be unreachable with the available bath states."
            )
        else:
            unreachable = (
                "Could not bracket the requested impurity occupation {target} with "
                f"|mu| <= {max_shift}: " + "the occupation reached {value:.4f} at mu = {mu:.3f}. "
                "The target may be unreachable with the available bath states."
            )
        unreachable += (
            # on already forms the ensemble where the sectors are thermally mixed: "use on" would be no advice
            " mixed_valence already averages over the charge sectors it could weight; where it refused "
            "the ensemble (see the refusal note) the occupation steps with the winning sector."
            if ctx.mixed_valence
            else " mixed_valence = true measures the thermal ensemble over charge sectors, where the occupation "
            "can vary continuously with the shift."
        )
        # Held by the record so the emit in its `finally` can read the search's own verdict even on
        # the path where the raise skips past every line that would have projected it.
        search_report = dc_rec.search = {}
        dc_rec.certification = ctx.plan_store
        dc_rec["target"] = target
        dc_rec["tol"] = occ_tol
        dc_rec["n_ref"] = ctx.n0
        dc_rec["de2_min"] = ctx.de2_min
        dc_rec["e_pt2_tol"] = ctx.e_pt2_tol
        with _dc_search_trace("fixed-occupation", MPI.COMM_WORLD, rank, width_tol=occ_width_tol, report=search_report):
            # Same calibration as the gap criterion, against this criterion's own tolerance. Only
            # when the cap defaulted -- an explicit truncation_threshold is the caller's
            # instruction. Measured on nio_5peeled, this criterion's answer was identical to six
            # decimals across caps 500/2000/8000, so it should settle on the first rung -- the
            # ladder costs a couple of cheap evaluations to establish that rather than assuming it.
            if ctx.cap_from_memory:

                def _clear_occupation_caches():
                    occupation_at.clear()
                    sector_at.clear()
                    ctx.plan_store.clear()
                    ctx.refused_at.clear()
                    ctx.members_at.clear()
                    if ctx.ensemble_variance_at is not None:
                        ctx.ensemble_variance_at.clear()

                _calibrate_cap(
                    ctx,
                    lambda: _evaluate_occupation_and_energy_at_mu(ctx, 0.0, verbose, rank)[0],
                    _clear_occupation_caches,
                    occ_tol,
                    dc_rec,
                    verbose=verbose,
                    rank=rank,
                )
                # Cleared unconditionally, unlike the gap criterion's cache-reuse fix -- there is
                # nothing here to keep. `_evaluate_occupation_and_energy_at_mu` holds no cache of
                # its own (every call is a fresh `solve_ground_state`), and `occupation_at`/
                # `sector_at` are written only by `occupation_observable`, which the ladder calls
                # directly and bypasses. So the rung's ground state is already gone by the time
                # this runs regardless of what these two dicts hold; left open deliberately
                # (91109b8), since closing it would mean seeding `_solve_dc_shift`'s own cache
                # with the ladder's last rung -- an API change on the search all three criteria
                # share, not a decision local to this loop.
                _clear_occupation_caches()

            def search(width_tol, continuous):
                return _solve_dc_shift(
                    occupation_observable,
                    target,
                    tol=occ_tol,
                    width_tol=width_tol,
                    initial_step=max(10 * ctx.tau, initial_step),
                    max_shift=max_shift,
                    plateau_ok=True,
                    unreachable_message=unreachable,
                    rank=rank,
                    # Sector-first, and the nominal occupation is what "prefer the user's nominal" means:
                    # the criterion is solved inside the mu-interval where the walk lands on that charge
                    # state, so a root at another sector cannot win by sitting nearer mu = 0.
                    # Continuous fallback (mixed_valence on): n(mu) is the grand-canonical average, smooth across a
                    # sector change, so there is no sector boundary to solve inside -- a plain bracketed root.
                    sector_of=None if continuous else sector_at.get,
                    nominal_sector=None if continuous else sum(ctx.N0.values()),
                    # dn/dmu >= 0: a positive mu lowers the impurity level and raises its occupation.
                    # No analytic magnitude here -- chi is precisely what the plateau/gap question is
                    # about, and assuming a value for it is how a search talks itself past a flat region.
                    # Direction only; the walk finds the scale.
                    slope_sign=1,
                    report=search_report,
                    verbose=verbose,
                    # The observable's own collectives run on COMM_WORLD (the basis builds below
                    # hardcode it), not on the caller's `comm`, so the residual must be broadcast
                    # there too or the branch decisions can diverge between ranks.
                    comm=MPI.COMM_WORLD,
                )

            try:
                try:
                    mu = search(occ_width_tol, continuous)
                except DoubleCountingUnreachable:
                    if not _member_flip_collapse(ctx, search_report.get("collapsed_bracket")):
                        raise
                    # auto: the bracket collapsed between two formed ensembles. The step is the dominant member
                    # changing, not a charge-sector step, and the grand-canonical n(mu) is continuous across it, so
                    # the search is rerun as under 'on' (review H-MV12). The collapse check is replicated: both
                    # dicts are written from broadcast plans, so every rank takes this branch together.
                    if rank == 0:
                        lo, hi = search_report["collapsed_bracket"]
                        print(
                            f"fixed_occupation_dc: the occupation steps between mu = {lo:.6f} and {hi:.6f} where the "
                            "dominant ensemble member changes; searching the grand-canonical n(mu) continuously.",
                            flush=True,
                        )
                    continuous, occ_width_tol = True, continuous_width_tol
                    # Recorded before the rerun, so a fallback that also fails is visible in the record; the
                    # first search's collapsed bracket is not this answer's.
                    dc_rec["continuous_fallback"] = True
                    search_report.pop("collapsed_bracket", None)
                    search_report["width_tol"] = occ_width_tol
                    first_evaluations = int(search_report.get("evaluations", 0))
                    try:
                        mu = search(occ_width_tol, continuous)
                    finally:
                        search_report["evaluations"] = first_evaluations + int(search_report.get("evaluations", 0))
            except DoubleCountingUnreachable:
                # An unreachable target is the likeliest outcome of a refused ensemble (n(mu) then steps with the
                # winning sector): the record must still say how many trial points were flagged.
                if ctx.refused_at:
                    dc_rec.update(
                        ensemble_refused_points=len(ctx.refused_at),
                        ensemble_refusal_reasons=sorted(set(ctx.refused_at.values())),
                    )
                raise

        # mu is always a point _solve_dc_shift actually evaluated (the mu=0 fast path, a direct scan
        # hit, or a refined bracket point), so occupation_observable(mu) ran and cached it here.
        n = occupation_at[mu]
        sector = sector_at[mu]
        dc = dc_guess + mu * ctx.shift_input

        # M1: the sector at the RETURNED mu is the number this whole branch exists to establish -- DC
        # <-> GS parity is a statement about which charge state calc_selfenergy will find at this same
        # dc. Warn, not raise: unlike fixed_peak_dc, following the walk across sectors here is
        # deliberate (the module docstring), and the thermally averaged N need not equal the walk's
        # winning sector total near a crossing -- that disagreement is itself worth surfacing, not an
        # error. Unconditional on rank 0, matching the other reference-quality warnings in this module.
        nominal_total = sum(ctx.N0.values())
        if rank == 0 and sector != nominal_total and not continuous:
            print(
                f"WARNING: fixed_occupation_dc found mu = {mu:.6f} (dc = dc_guess + mu), but the "
                f"walk's winning sector there totals {sector}, not the nominal occupation "
                f"{nominal_total}. calc_selfenergy will find this same sector at the returned dc.",
                flush=True,
            )

        # chi is recorded unconditionally, not only on a miss: a *converged* occupation with a small
        # chi still leaves dc essentially undetermined (delta_mu = delta_residual / chi), which is
        # exactly the case a miss-only report would hide (review correction 5). occupation_at holds
        # every mu the TRUE, re-expanding observable actually visited, so this is free -- no frozen
        # space, no extra solve -- and it is strictly better evidence than a frozen-space surrogate
        # would be. width_tol matches _solve_dc_shift's own bracket resolution and the sector
        # predicate keeps the difference inside one charge state, so neither a collapsed bracket at a
        # sector boundary nor a step across one is read as a slope (see _dc_chi).
        occ_chi, occ_chi_span = _dc_chi(
            occupation_at,
            mu,
            width_tol=occ_width_tol,
            in_sector=(None if continuous else (lambda trial: sector_at.get(trial) == sector_at.get(mu))),
        )
        dc_rec.update(
            n_gs=n,
            chi=occ_chi,
            chi_span=occ_chi_span,
        )
        flagged = {m: iv for m, iv in ctx.interval_at.items() if iv[1] - iv[0] > config.ENSEMBLE_WEIGHT_ERROR_MAX}
        if flagged:
            widest = max(flagged.values(), key=lambda iv: iv[1] - iv[0])
            dc_rec.update(ensemble_weight_interval=[round(x, 4) for x in widest], ensemble_flagged_points=len(flagged))
        if ctx.refused_at:
            # One note per search (the reasons are replicated, so every rank agrees): at these trial shifts the
            # sector energies did not fix the occupation weights well enough, so the point carries the winning
            # sector alone and the result is flagged cap-bound.
            reasons = sorted(set(ctx.refused_at.values()))
            dc_rec.update(ensemble_refused_points=len(ctx.refused_at), ensemble_refusal_reasons=reasons)
            if rank == 0:
                print(
                    f"WARNING: fixed_occupation_dc: the charge-sector ensemble was refused at {len(ctx.refused_at)} of "
                    f"{len(occupation_at)} trial shifts ({'; '.join(reasons)}). Those points use the winning sector "
                    "alone, so n(mu) is not the grand-canonical average there and the result is flagged.",
                    flush=True,
                )
        variance = (ctx.ensemble_variance_at or {}).get(mu)
        if variance is not None:
            dc_rec.update(var_n_intra_sector=variance[0], var_n_inter_sector=variance[1])
            if variance[1] > variance[0] and rank == 0:
                print(
                    f"WARNING: at the double-counting shift the fluctuation of the impurity occupation is mostly "
                    f"between charge sectors (Var N = {variance[1]:.3g} between, {variance[0]:.3g} within): the bath "
                    "lacks states near E_F to screen a charge change, so the occupation is set by the sector "
                    "energies' alignment and this dc is only as good as that gap.",
                    flush=True,
                )
        # The same error bar the gap criterion reports (see its own comment on this field), so a
        # caller reading the record programmatically gets one for both. This criterion has no
        # `delta_sum` -- it is a sector-energy quantity, and this one is not -- so it comes from
        # `chi`, the secant, which is `None` exactly when the search converged in one evaluation.
        #
        # `is not None`, not truthiness, matching the gap criterion's own fix: `occ_chi == 0.0` is
        # a genuine plateau (this same function's `plateau_ok=True` treats it as a real, expected
        # case, not an error) -- a measured zero slope, not a missing measurement -- see
        # `_mu_tol_effective`.
        if occ_chi is not None:
            dc_rec["mu_tol_effective"] = _mu_tol_effective(occ_tol, occ_chi)
        dc_rec["dc_trace"], dc_rec["dc_level"] = dc_record.dc_levels(dc)
        dc_rec["dc_spread"] = dc_record.dc_spread(dc)
        # No "achieved occupation misses the target" line (B3): _solve_dc_shift only returns via a
        # path where |n - target| <= occ_tol -- a miss beyond that raises DoubleCountingUnreachable
        # instead, so the record's n_gs never has a genuine miss hidden in it.
        _dump_dc_matrices(dc_guess, dc, rank)
        _reject_if_memory_bound(ctx, dc_rec, rank=rank, allow_memory_bound=allow_memory_bound)
        return (dc, sector) if return_sector else dc
