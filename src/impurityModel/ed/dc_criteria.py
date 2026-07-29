r"""The two ED-based double-counting criteria: pin a spectral peak, or pin the occupation.

Both are special cases of :func:`dc_search._solve_dc_shift` over the uniform shift
``dc(mu) = dc_guess + mu * identity``. :func:`fixed_peak_dc` pins a chosen spectral peak --
the sector-energy difference ``E[N+1] - E[N]`` (or ``E[N] - E[N-1]`` for a removal peak);
:func:`fixed_occupation_dc` pins the impurity occupation, and with no explicit target pins it to
the DFT reference occupation from :mod:`dc_reference`, which is the natural criterion for CSC
DFT+DMFT of wide-window p-d models.

At every trial shift both determine the ground-state sector through the same *function*
:func:`groundstate.calc_gs` uses -- :func:`groundstate.find_ground_state_basis`'s
HF-seed-then-walk search, not a search pinned at the input occupation. That matters because a
``dc`` measured on a different sector than the one ``calc_selfenergy`` later finds does not
approximate the requested physics, it misdirects it: the downstream calculation is locked onto the
wrong charge state, which is worse than not fixing anything at all.

.. warning::
   Until this fix, it was the same function but **not the same call** -- an earlier version of
   this docstring claimed otherwise. Three differences, all found by review, are now closed:

   * ``calc_gs`` divides ``tau`` by 100 before the walk (``groundstate.py``, ``calc_gs``); both
     criteria now do the same for their own ``find_ground_state_basis`` call, restoring the full
     ``tau`` for everything computed afterward at the found sector (the thermal average, the
     peak's sector-energy solves).
   * ``calc_gs`` computes ``symmetry_generators`` via :func:`solver_basis.get_symmetry_generators`
     and passes it down; both criteria now compute it once (a uniform ``mu`` shift on the impurity
     diagonal commutes with any one-body symmetry, so it is valid for every trial ``mu``) and pass
     it into every ``find_ground_state_basis``/``calc_energy``/``expand`` call.
   * ``find_ground_state_basis`` accepts ``frozen_occupations``, and the CLI populates it for
     bath-less core shells (``get_spectra.py``: ``{i for i in nBaths if nBaths[i] == 0}``); both
     criteria now derive the same set from :attr:`solver_basis.SolverBasis.sum_bath_states` and
     pass it down, so a core shell can no longer drain in the double-counting search only.

Like :func:`selfenergy.calc_selfenergy` and :func:`groundstate.find_ground_state_basis`, both
derive their determinant budget from available per-rank memory
(:func:`impurityModel.ed.memory_estimate.suggest_truncation_threshold`) when
``BasisOptions.truncation_threshold`` is left at ``None``, and honor
``BasisOptions.excitation_budget``/``chain_restrict`` through the same
:func:`impurityModel.ed.basis_restrictions.build_weighted_restrictions` the other ED drivers use
-- the double counting is otherwise found on a different variational space than the solve that
will use it.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed import solver_trace
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_restrictions import build_weighted_restrictions
from impurityModel.ed.basis_transcription import build_density_matrices
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.dc_reference import (
    _SATURATION_ADVICE,
    _noninteracting_impurity_occupation,
    _warn_if_not_fermi_referenced,
    _warn_if_reference_far_from_nominal,
    _warn_if_reference_saturated,
)
from impurityModel.ed.dc_search import _dc_search_trace, _solve_dc_shift
from impurityModel.ed.lie_algebra import extract_tensors, tensors_to_operator
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.memory_estimate import DEFAULT_MEMORY_SAFETY, log_memory_budget, suggest_truncation_threshold
from impurityModel.ed.solver_basis import _per_group_occupation, get_symmetry_generators, prepare_solver_basis
from impurityModel.ed.utils import matrix_print


def _dc_operator(dc):
    """Build the double-counting one-body operator, ``dc[i, j] c^dagger_i c_j``."""
    return tensors_to_operator(np.asarray(dc, dtype=complex))


def fixed_peak_dc(model, basis, solver, *, peak_position, comm=None, verbosity=0):
    r"""
    Calculate the double counting correction using a fixed peak position criterion.

    Choose the double counting so that a peak in the impurity spectral function
    lands at the requested energy,

    .. math::
        E[N+1] - E[N] &= \omega_{peak},\quad \omega_{peak} \geq 0,\\
        E[N] - E[N-1] &= \omega_{peak},\quad \omega_{peak} < 0,

    where :math:`E[M]` is the lowest energy with M electrons on the impurity.
    A positive peak position places an electron-addition peak, a negative one an
    electron-removal peak.

    The double counting is parametrized as a uniform shift of the guess,
    ``dc(mu) = dc_guess + mu * identity``. The shift couples to the impurity
    occupation as :math:`-\mu \hat N_{imp}`, so the peak position responds
    monotonically, :math:`d(E_{upper} - E_{lower})/d\mu = -(\langle N
    \rangle_{upper} - \langle N \rangle_{lower}) \approx -1`, and ``mu`` is
    found by the shared secant/bisection search :func:`_solve_dc_shift`.

    At every trial ``mu`` the *center* sector ``N`` is not the input ``model.impurity_orbitals``
    occupation: it is found by :func:`groundstate.find_ground_state_basis`, the identical
    HF-seed-then-walk search :func:`groundstate.calc_gs` uses for the selfenergy/spectra solve.
    A peak measured relative to a different center than the one calc_selfenergy later finds would
    not approximate the requested peak position, it would misplace it -- the same reasoning that
    drives :func:`fixed_occupation_dc`'s sector determination. The ``N +- 1`` sectors are then
    each a single, fixed-occupation solve (:func:`groundstate.calc_energy`) relative to that
    found center, not a further search. ``N`` here is the *whole-impurity* occupation, never one
    orbital-symmetry group's: :func:`solver_basis.prepare_solver_basis` re-derives the grouping
    from the block structure (a cubic d-shell input arrives as one group and comes back as
    ``eg``/``t2g``), and the many-body basis is filtered on the total impurity charge alone, so
    the total is the only well-defined handle. Energies carry no fixed unit, they follow the inputs
    (e.g. Ry when called from RSPt); the convergence tolerance is ``max(tau, 1e-4)`` in those
    units. All three sectors honor ``BasisOptions.excitation_budget``/``chain_restrict``
    identically -- the bath-only excitation-budget restriction never references the occupation,
    so the same restriction list applies regardless of which sector it is evaluated on.

    Parameters
    ----------
    model : impurityModel.ed.model.ImpurityModel
        The impurity problem: ``h0`` (non-interacting Hamiltonian), ``u4`` (Coulomb tensor),
        ``dc`` (double counting correction, used as the search's starting guess),
        ``impurity_orbitals`` and ``bath_states`` -- the ``(valence, conduction)`` bath split is
        required here (build the model with it, e.g. ``from_blocks(..., bath_valence_conduction=...)``).
    basis : impurityModel.ed.model.BasisOptions
        Nominal occupation (``{group: N}``; any number of groups -- the criterion moves one
        electron on/off the impurity *as a whole*, so no group is singled out), mixed valence,
        spin-flip determinants,
        temperature and the determinant budget. ``truncation_threshold=None`` (the default)
        derives the cap from available per-rank memory (collective on ``MPI.COMM_WORLD``,
        :func:`impurityModel.ed.memory_estimate.suggest_truncation_threshold`), halved to
        account for the upper and lower sector bases held simultaneously; ``numpy.inf``
        disables capping.
    solver : impurityModel.ed.model.SolverOptions
        Provides the dense-eigensolver cutoff.
    peak_position : float
        Requested peak position; the sign selects addition/removal, see above.
        The magnitude is kept above ``4 * tau`` (thermal broadening).
    comm : MPI.Comm or None
        MPI communicator (used for rank-0 logging; the basis build uses ``MPI.COMM_WORLD``).
    verbosity : int
        Verbosity level.

    Returns
    -------
    dc : ndarray
        The double counting matrix, ``dc_guess + mu * identity``.

    Raises
    ------
    RuntimeError
        If the requested peak cannot be bracketed within the reachable range,
        e.g. because the criterion is ill conditioned (the upper and lower
        sectors have the same impurity occupation, so a uniform shift cannot
        move the peak).
    """
    from impurityModel.ed.groundstate import calc_energy, find_ground_state_basis

    # Unpack the grouped parameters into the local names used throughout the body.
    h0_op = model.h0
    u = model.u4
    n_imp = len(model.impurity_indices)
    dc_guess = extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0]
    N0 = basis.nominal_occ
    mixed_valence = basis.mixed_valence
    spin_flip_dj = basis.spin_flip_dj
    tau = basis.tau
    truncation_threshold = basis.truncation_threshold
    excitation_budget = basis.excitation_budget
    chain_restrict = basis.chain_restrict
    slaterWeightMin = basis.slater_weight_min
    dense_cutoff = solver.dense_cutoff
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0

    # Derive the same solver-basis layout calc_selfenergy uses (root cause 3 of the original
    # DC<->GS mismatch: the fit valence/conduction split model.bath_states carries is not
    # necessarily the split calc_selfenergy re-derives from the sign of the bath on-site energy).
    sb = prepare_solver_basis(
        h0_op, model.dc, u, model.impurity_orbitals, N0, mixed_valence, model.rot_to_spherical, verbosity
    )
    impurity_orbitals = sb.impurity_orbitals
    bath_states = sb.bath_states
    N0 = sb.nominal_occ
    mixed_valence = sb.mixed_valence
    h_op_i = sb.h
    # The peak criterion moves one electron on/off the impurity *as a whole*, never a named
    # group: the many-body basis is generated from the whole-impurity charge window (only the
    # total is filtered, see basis_generation.generate_initial_basis) and n_center below is
    # likewise read off as a whole-impurity count, so the total is the only well-defined handle.
    # Anything keyed on a single group is a bug here: prepare_solver_basis re-derives the
    # grouping from the block structure, so a single-group input becomes several derived groups
    # on any cubic crystal field ({0: [0..9]} -> eg/t2g), and a group key taken from the *input*
    # N0 then indexes the *derived* layout. _per_group_occupation is the same total -> groups
    # mapping prepare_solver_basis itself uses (energetic filling); since the basis depends only
    # on the total, which split it returns cannot change the answer.
    max_occ_total = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)
    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]
    # One-body matrix in the *solver* basis (h0_solve, i.e. after any symmetry rotation), so the
    # energetic filling below sees the same on-site energies that defined the derived groups.
    h_solver_matrix = extract_tensors(sb.h0_solve, n_orb=sb.n_spin_orbitals, two_body=False)[0]
    identity = np.identity(dc_guess.shape[0])

    # The three inputs that make this the same call calc_gs makes (see the module warning):
    # a bath-less group pinned exactly as get_spectra.py pins it, the same one-body symmetry
    # generators (valid at every trial mu -- a uniform impurity shift commutes with them), and
    # the walk's tau divided by 100 below, restored to the full tau for everything solved at the
    # sector the walk finds.
    frozen_occupations = {i for i in impurity_orbitals if sb.sum_bath_states[i] == 0}
    symmetry_generators = get_symmetry_generators(h_op_i, impurity_orbitals, bath_states)

    if truncation_threshold is None:
        # Two fixed-sector solves (N +- 1) are built alongside the center search's own basis,
        # so halve the safety fraction relative to a single-basis driver to keep the same
        # overall per-rank headroom.
        truncation_threshold = suggest_truncation_threshold(
            model.n_spin_orbitals, comm=MPI.COMM_WORLD, safety=DEFAULT_MEMORY_SAFETY / 2
        )
        log_memory_budget(
            truncation_threshold, model.n_spin_orbitals, comm=MPI.COMM_WORLD, verbose=verbose, label="fixed-peak dc"
        )

    # Bath-only restriction (the reference "valence filled, conduction empty" occupation never
    # references N0), so the identical restriction list is valid for every sector -- the same
    # restrictions find_ground_state_basis applies to its own occupation-scan trials.
    weighted_restrictions = build_weighted_restrictions(bath_states, excitation_budget)

    # Keep the requested peak outside the thermal broadening, preserving the sign: a negative
    # peak position places a removal peak at E[N] - E[N-1].
    if peak_position >= 0:
        peak_position = max(peak_position, 4 * tau)
        addition = True
    else:
        peak_position = min(peak_position, -4 * tau)
        addition = False

    # Scale for the out-of-bounds sector penalty below: large enough to dominate any reachable
    # E_upper - E_lower (which is O(bandwidth)), but finite -- _refine_bracket's secant step
    # divides by a residual difference, and an infinite residual there is 0/0 (both sectors
    # unreachable) or inf/inf (not a valid bracket update either way).
    h1_for_scale = extract_tensors(ManyBodyOperator(h0_op), n_orb=model.n_spin_orbitals, two_body=False)[0]
    unreachable_penalty = 1e4 * max(float(np.ptp(np.linalg.eigvalsh(h1_for_scale))), 1.0)

    def peak_observable(mu):
        with solver_trace.labelled(mu=mu), solver_trace.timed("dc_evaluation") as evaluation_fields:
            gap = _peak_gap_at_mu(mu)
            evaluation_fields["gap"] = gap
            return gap

    def _peak_gap_at_mu(mu):
        # h_op_i (sb.h, from prepare_solver_basis) already has dc_guess subtracted (H = h0 - DC +
        # U with DC = model.dc = dc_guess); only the incremental mu shift is subtracted here, or
        # dc_guess would be double-counted. Matches fixed_occupation_dc's identical pattern.
        h_op = h_op_i - _dc_operator(mu * identity)

        # The center sector, determined the same way calc_selfenergy will (see the docstring):
        # find_ground_state_basis's own HF-seed-then-walk search, not a search pinned at the
        # input N0.
        basis_center = find_ground_state_basis(
            h_op,
            impurity_orbitals,
            bath_states,
            N0,
            frozen_occupations=frozen_occupations,
            mixed_valence=mixed_valence,
            # calc_gs's own convention (groundstate.py): the sector-selection walk runs at
            # tau/100, restored to the full tau for the sector energies solved below.
            tau=tau / 100,
            chain_restrict=chain_restrict,
            dense_cutoff=dense_cutoff,
            spin_flip_dj=spin_flip_dj,
            comm=MPI.COMM_WORLD,
            verbose=verbose,
            truncation_threshold=truncation_threshold,
            slaterWeightMin=slaterWeightMin,
            weighted_restrictions=weighted_restrictions,
            symmetry_generators=symmetry_generators,
        )
        # The centre sector, read from the search that chose it. NOT from a determinant: the
        # returned basis is the eigenvector support of an expansion whose impurity occupation
        # window was widened (build_excited_restrictions with imp_change=None is unconstrained),
        # so it spans several impurity occupations -- {1, 2, 3} on a split-block toy whose
        # winning sector is 2, where the first determinant reports 1. Centring the peak on
        # whichever determinant happened to come first measured E[2] - E[1] where E[3] - E[2]
        # was meant. fixed_occupation_dc never had this problem: it reads Tr rho_imp over the
        # whole basis, which averages the mixed occupations correctly.
        n_center = sum(basis_center.ground_state_occupation.values())

        n_upper = n_center + 1 if addition else n_center
        n_lower = n_center if addition else n_center - 1

        def sector_energy(n_trial):
            if not 0 <= n_trial <= max_occ_total:
                return unreachable_penalty
            occ_trial = _per_group_occupation(n_trial, impurity_orbitals, h_solver_matrix)
            e_trial, _ = calc_energy(
                h_op,
                impurity_orbitals,
                bath_states,
                occ_trial,
                mixed_valence,
                tau,
                chain_restrict,
                spin_flip_dj,
                dense_cutoff,
                comm=MPI.COMM_WORLD,
                verbose=verbose,
                truncation_threshold=truncation_threshold,
                slaterWeightMin=slaterWeightMin,
                weighted_restrictions=weighted_restrictions,
                frozen_occupations=frozen_occupations,
                symmetry_generators=symmetry_generators,
            )
            # calc_energy itself returns inf for an empty basis (e.g. a sector the current
            # restrictions admit no determinants for); clamp to the same finite penalty for the
            # same reason as the bounds check above.
            return e_trial if np.isfinite(e_trial) else unreachable_penalty

        e_upper = sector_energy(n_upper)
        e_lower = sector_energy(n_lower)
        if verbose and rank == 0:
            print(f"mu={mu:.6f} n_center={n_center} E_upper - E_lower={e_upper - e_lower:.6f}")
        return e_upper - e_lower

    # Scale the bracketing to the non-interacting bandwidth (the spread of the one-body h0
    # eigenvalues); the peak position responds to the shift with slope ~ -1, so this comfortably
    # covers the reachable range. An observable that does not move with mu (the upper and lower
    # sectors hold the same impurity occupation -- the old delta_n ~ 0 ill-conditioning) never
    # brackets and surfaces here as the unreachable RuntimeError.
    bandwidth = unreachable_penalty / 1e4
    tol = max(tau, 1e-4)
    unreachable = (
        "The fixed-peak double counting could not place the peak at {target}: E_upper - E_lower "
        "reached {value:.4f} at mu = {mu:.3f}. The upper and lower sectors may hold equal impurity "
        "occupation (a uniform shift cannot move the peak), or the target lies beyond the "
        "reachable range."
    )
    with _dc_search_trace("fixed-peak", MPI.COMM_WORLD, rank):
        mu = _solve_dc_shift(
            peak_observable,
            peak_position,
            tol=tol,
            width_tol=tol,
            initial_step=max(10 * tau, abs(peak_position)),
            max_shift=max(bandwidth, 10 * abs(peak_position), 1.0),
            plateau_ok=False,
            unreachable_message=unreachable,
            rank=rank,
            # The observable's own collectives run on COMM_WORLD (the basis builds below
            # hardcode it), not on the caller's `comm`, so the residual must be broadcast
            # there too or the branch decisions can diverge between ranks.
            comm=MPI.COMM_WORLD,
        )

    dc = dc_guess + mu * identity
    if verbose and rank == 0:
        print(f"Fixed-peak double counting (peak position = {peak_position}, mu = {mu:.6f}):")
        matrix_print(dc_guess, label="DC guess:")
        matrix_print(dc, label="DC found:")

    return dc


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
):
    r"""
    Calculate the double counting correction using a fixed (or self-consistent) impurity
    occupation criterion.

    With an explicit ``occupation``, choose the double counting so that the interacting thermal
    impurity occupation equals the requested value, :math:`\mathrm{Tr}\,\rho_{imp} = N_{target}`.

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
    mu * identity``, coupling to the impurity occupation as :math:`-\mu \hat N_{imp}`, so the
    residual :math:`N(\mu) - N_{target}` is monotone in ``mu`` for either target and is driven
    to zero by the search :func:`_solve_dc_shift`. At low temperature and weak
    hybridization the occupation approaches a staircase in ``mu``; if the requested (fractional)
    occupation falls on a plateau, the search converges to the closest step and a warning is
    printed.

    Unlike :func:`fixed_peak_dc`, the sector this search measures the occupation on is **not**
    fixed at the input ``model.impurity_orbitals``/``model.bath_states``: at every trial ``mu``
    it derives the same solver-basis layout :func:`calc_selfenergy` uses
    (:func:`solver_basis.prepare_solver_basis` -- the impurity/bath grouping, the bath valence/
    conduction split from the sign of the bath on-site energy rather than the hybridization fit,
    and any symmetry-adapted rotation), then determines the ground-state sector by calling
    :func:`groundstate.find_ground_state_basis` itself -- the identical HF-seed-then-walk search
    :func:`groundstate.calc_gs` uses for the selfenergy/spectra solve, not a cheaper
    approximation of it. A dc measured on a sector different from the one calc_selfenergy later
    finds would not just be imprecise: it would lock the downstream calculation onto the wrong
    charge state, so this search pays the full per-trial cost of walking rather than caching a
    single Hartree-Fock seed across mu. This is what makes the "fixed" occupation the same
    quantity :func:`groundstate.find_ground_state_basis` finds for the selfenergy/spectra solve
    at the returned ``dc`` -- the point of this whole search. The search follows wherever that
    walk leads, including across a change in total electron number (matching
    ``find_ground_state_basis``'s own grand-canonical-style sector search); the reachable
    occupations are limited by the bath, and a tight
    ``BasisOptions.excitation_budget``/``chain_restrict`` can itself make a requested occupation
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
        The double counting matrix, ``dc_guess + mu * identity``.

    Raises
    ------
    RuntimeError
        If the target cannot be bracketed within ``max_shift``.
    """
    from impurityModel.ed.groundstate import find_ground_state_basis

    # Unpack the grouped parameters into the local names used throughout the body.
    h0_op = model.h0
    u = model.u4
    n_imp = len(model.impurity_indices)
    dc_guess = extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0]
    N0 = basis.nominal_occ
    mixed_valence = basis.mixed_valence
    spin_flip_dj = basis.spin_flip_dj
    tau = basis.tau
    excitation_budget = basis.excitation_budget
    chain_restrict = basis.chain_restrict
    slaterWeightMin = basis.slater_weight_min
    dense_cutoff = solver.dense_cutoff
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0

    # Derive the same solver-basis layout calc_selfenergy uses (the DC search must measure its
    # observable on the same sector/basis the downstream selfenergy solve will use, or "fixed"
    # does not mean what it says). The mu shift is a uniform impurity shift (-mu * identity),
    # which commutes with the impurity block and hence with any rotation prepare_solver_basis
    # applies, so the layout derived once here at the guess dc is valid for every trial mu.
    sb = prepare_solver_basis(
        h0_op, model.dc, u, model.impurity_orbitals, N0, mixed_valence, model.rot_to_spherical, verbosity
    )
    impurity_orbitals = sb.impurity_orbitals
    bath_states = sb.bath_states
    N0 = sb.nominal_occ
    mixed_valence = sb.mixed_valence
    h_op_i = sb.h

    total_impurity_orbitals = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)
    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]

    # The three inputs that make this the same call calc_gs makes (see the module warning):
    # a bath-less group pinned exactly as get_spectra.py pins it, the same one-body symmetry
    # generators (valid at every trial mu -- a uniform impurity shift commutes with them), and
    # the walk's tau divided by 100 below, restored to the full tau for everything solved at the
    # sector the walk finds.
    frozen_occupations = {i for i in impurity_orbitals if sb.sum_bath_states[i] == 0}
    symmetry_generators = get_symmetry_generators(h_op_i, impurity_orbitals, bath_states)

    # DFT reference occupation: Fermi filling of the raw h0 (the KS Hamiltonian of the
    # h0 - dc + U contract; no double counting subtracted before filling), independent of
    # dc_guess and of the trial shift, and of the solver-basis rotation (h0's raw filling is
    # evaluated in the model's input basis). Logged at every trial; with occupation=None it is
    # also the search target.
    n0 = _noninteracting_impurity_occupation(h0_op, impurity_indices, model.n_spin_orbitals, tau)
    self_consistent = occupation is None
    if self_consistent:
        occupation = n0
        # Only when N0 is the *target*: with an explicit target a saturated reference is merely
        # logged, not acted on.
        # The E_F = 0 convention underpins both the reference filling below and every sector
        # comparison the occupation walk makes; it is asserted throughout and was never checked.
        _warn_if_not_fermi_referenced(h0_op, model.n_spin_orbitals, rank=rank)
        # Saturation is the more specific diagnosis, so it wins when both would fire; the
        # nominal-gap check catches the reference that is grossly wrong *without* being pinned at
        # a shell edge, which the saturation test alone lets through (a runaway CSC iterate
        # reporting n0 = 1.54 against a nominal 8 converged silently to a 43 eV shift).
        if not _warn_if_reference_saturated(
            n0, total_impurity_orbitals, _SATURATION_ADVICE["search"], occ_tol=occ_tol, rank=rank
        ):
            _warn_if_reference_far_from_nominal(n0, sum(N0.values()), _SATURATION_ADVICE["search"], rank=rank)
    # Tolerance absorbs the roundoff of a target derived elsewhere the same way
    # _noninteracting_impurity_occupation is (a sum of Fermi occupations, each in [0, 1]).
    if not -1e-9 <= occupation <= total_impurity_orbitals + 1e-9:
        raise ValueError(f"Requested impurity occupation {occupation} outside [0, {total_impurity_orbitals}].")

    truncation_threshold = basis.truncation_threshold
    if truncation_threshold is None:
        truncation_threshold = suggest_truncation_threshold(model.n_spin_orbitals, comm=MPI.COMM_WORLD)
        log_memory_budget(
            truncation_threshold,
            model.n_spin_orbitals,
            comm=MPI.COMM_WORLD,
            verbose=verbose,
            label="fixed-occupation dc",
        )

    identity = np.identity(dc_guess.shape[0])
    energy_cut = -tau * np.log(1e-4)
    weighted_restrictions = build_weighted_restrictions(bath_states, excitation_budget)

    occupation_at = {}

    def occupation_observable(mu):
        with solver_trace.labelled(mu=mu), solver_trace.timed("dc_evaluation") as evaluation_fields:
            n_out = _occupation_at_mu(mu)
            evaluation_fields["n"] = n_out
            return n_out

    def _occupation_at_mu(mu):
        h_op = h_op_i - _dc_operator(mu * identity)

        # Determine the ground-state sector the SAME way calc_selfenergy will: the identical
        # HF-seed-then-walk find_ground_state_basis performs (not a cheaper approximation of it,
        # e.g. a bare HF seed with no correction). A dc value measured on a different sector than
        # the one calc_selfenergy later finds does not just add noise -- it locks the downstream
        # calculation onto the wrong charge state, which is worse than not fixing anything at
        # all. N0 stays the *original* nominal occupation on every call (not a carried-forward
        # sector from the previous mu): feeding the walk its own last answer would make the
        # result path-dependent on the bracket's evaluation order, which breaks parity by a
        # different route.
        mb_basis = find_ground_state_basis(
            h_op,
            impurity_orbitals,
            bath_states,
            N0,
            frozen_occupations=frozen_occupations,
            mixed_valence=mixed_valence,
            # calc_gs's own convention (groundstate.py): the sector-selection walk runs at
            # tau/100, restored to the full tau for the refinement and thermal average below.
            tau=tau / 100,
            chain_restrict=chain_restrict,
            dense_cutoff=dense_cutoff,
            spin_flip_dj=spin_flip_dj,
            comm=MPI.COMM_WORLD,
            verbose=verbose,
            truncation_threshold=truncation_threshold,
            slaterWeightMin=slaterWeightMin,
            weighted_restrictions=weighted_restrictions,
            symmetry_generators=symmetry_generators,
        )
        mb_solver = CIPSISolver(mb_basis)
        # Refine at the same de2_min calc_gs uses for its own final solve (find_ground_state_basis
        # itself only expands to 1e-6, tight enough to compare sectors but not to match the
        # occupation calc_selfenergy will report at this same dc). This refinement is *not* part
        # of any sector solve -- it runs once per mu, on top of the whole walk -- so it is timed
        # under the same "expand" kind to land in the aggregate, tagged to say where it came from.
        with solver_trace.timed("expand", stage="dc_refine"):
            mb_solver.expand(
                h_op,
                dense_cutoff=dense_cutoff,
                de2_min=1e-8,
                slaterWeightMin=slaterWeightMin,
                symmetry_generators=symmetry_generators,
            )

        def solve_thermal_occupation():
            # The NiO ground state that motivated this rework is 3-fold quasi-degenerate, so a
            # single lowest state (as the frozen-basis search used) misrepresents the thermal
            # occupation; widen num_wanted until the manifold within energy_cut is captured.
            # The termination test on Lanczos energies is only replicated to roundoff, so
            # broadcast rank 0's decision -- ranks disagreeing here would diverge on whether to
            # re-enter the next collective get_eigenvectors call.
            num_wanted = 10
            while True:
                with solver_trace.timed("eigensolve", stage="dc_thermal", num_wanted=num_wanted):
                    es, psis = mb_solver.get_eigenvectors(
                        h_op,
                        num_wanted=num_wanted,
                        max_energy=energy_cut,
                        dense_cutoff=dense_cutoff,
                        slaterWeightMin=slaterWeightMin,
                        solver="irlm",
                        psi_refs=mb_solver.psi_refs,
                    )
                done = len(es) < num_wanted or (len(es) >= 1 and es[-1] - es[0] >= energy_cut) or num_wanted >= 100
                if mb_basis.is_distributed:
                    done = mb_basis.comm.bcast(done, root=0)
                if done:
                    break
                num_wanted += 10
            rhos = build_density_matrices(mb_basis, psis, impurity_indices, impurity_indices)
            rho = thermal_average_scale_indep(es, rhos, tau)
            return float(np.real(np.trace(rho)))

        n_out = solve_thermal_occupation()

        occupation_at[mu] = n_out
        if verbose and rank == 0:
            # The sector itself is reported by find_ground_state_basis's own verbose output above
            # ("Ground state occupation"), evaluated fresh at this mu.
            print(f"mu={mu:.6f} n={n_out:.6f} n0={n0:.6f}")
        return n_out

    target = occupation
    if self_consistent:
        unreachable = (
            f"Could not reach the DFT reference occupation N0 = {n0:.4f} with |mu| <= "
            f"{max_shift}: " + "the occupation reached {value:.4f} at mu = {mu:.3f}. "
            "The target may be unreachable with the available bath states."
        )
    else:
        unreachable = (
            "Could not bracket the requested impurity occupation {target} with "
            f"|mu| <= {max_shift}: " + "the occupation reached {value:.4f} at mu = {mu:.3f}. "
            "The target may be unreachable with the available bath states."
        )
    with _dc_search_trace("fixed-occupation", MPI.COMM_WORLD, rank):
        mu = _solve_dc_shift(
            occupation_observable,
            target,
            tol=occ_tol,
            width_tol=max(tau, 1e-4),
            initial_step=max(10 * tau, initial_step),
            max_shift=max_shift,
            plateau_ok=True,
            unreachable_message=unreachable,
            rank=rank,
            # The observable's own collectives run on COMM_WORLD (the basis builds below
            # hardcode it), not on the caller's `comm`, so the residual must be broadcast
            # there too or the branch decisions can diverge between ranks.
            comm=MPI.COMM_WORLD,
        )

    # mu is always a point _solve_dc_shift actually evaluated (the mu=0 fast path, a direct scan
    # hit, or a refined bracket point), so occupation_observable(mu) ran and cached it here.
    n = occupation_at[mu]
    dc = dc_guess + mu * identity
    if verbose and rank == 0:
        label = "N == N0" if self_consistent else f"target = {occupation}"
        print(f"Fixed-occupation double counting ({label}, achieved N = {n:.4f}, N0 = {n0:.4f}, mu = {mu:.6f}):")
        if abs(n - target) > occ_tol:
            print(
                f"WARNING: the achieved occupation {n:.4f} misses the target {target:.4f} by "
                f"more than occ_tol={occ_tol:.4f}.",
                flush=True,
            )
        matrix_print(dc_guess, label="DC guess:")
        matrix_print(dc, label="DC found:")
    return dc
