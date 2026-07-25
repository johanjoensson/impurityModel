"""Double-counting determination for the self-energy workflow.

The double counting is a first-class part of the model (``model.dc``), always subtracted from
the non-interacting Hamiltonian: the local Hamiltonian is ``h0 - dc + U(u4)`` throughout this
module and in :func:`selfenergy._prepare_solver_basis` / :func:`susceptibility` (never added).
The impurity double-counting potential is fixed by one of two searches over a uniform shift
``dc(mu) = dc_guess + mu * identity``: :func:`fixed_peak_dc` pins a chosen spectral peak,
:func:`fixed_occupation_dc` pins the impurity occupation. Both are special cases of one
generic search, :func:`_solve_dc_shift`: at a trial shift it builds the variational ground
state and its thermal density matrix (:func:`_lowest_energy_and_thermal_rho`), reads off a
scalar observable (the peak position ``E[N+1] - E[N]`` or the impurity occupation
``Tr rho_imp``), and drives it to the requested target with a safeguarded secant/bisection
root-finder. :func:`fixed_occupation_dc` also accepts no target at all: it then derives one
from the non-interacting ``h_loc`` (the Fermi-filled occupation of ``model.h0`` at mu=0,
:func:`_noninteracting_impurity_occupation`), which is the natural target for CSC DFT+DMFT of
wide-window p-d models. The self-energy extraction proper lives in :mod:`sigma`; the
orchestration and CLI in :mod:`selfenergy`, which re-exports
``fixed_peak_dc``/``fixed_occupation_dc`` so existing callers are unchanged.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_transcription import build_density_matrices
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.lie_algebra import extract_tensors, tensors_to_operator
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.utils import matrix_print


def _normalize_dc_orbitals(impurity_orbitals, bath_states):
    """Normalize flat orbital-index lists to the ``{group: [block, ...]}`` format of ``Basis``.

    Flat lists (the RSPt interface convention) are wrapped as a single block per
    group, so ``nominal_impurity_occ`` constrains the *total* impurity
    occupation -- which is the N of E[N +- 1] in the fixed-peak criterion.
    Grouping by conserved charges instead would pin per-spin occupations and
    distort the ground-state energies. Already blocked input passes through
    unchanged.
    """

    def as_blocked(orbital_dict):
        out = {}
        for key, val_raw in orbital_dict.items():
            val = list(val_raw)
            if len(val) > 0 and not hasattr(val[0], "__iter__"):
                out[key] = [sorted(val)]
            else:
                out[key] = val
        return out

    valence_baths, conduction_baths = bath_states
    return as_blocked(impurity_orbitals), (as_blocked(valence_baths), as_blocked(conduction_baths))


def _require_bath_states(model, func_name):
    """Return ``model.bath_states`` or raise a clear error when the split is missing.

    The double-counting search builds the many-body basis directly from the explicit bath
    valence/conduction partition (unlike ``calc_selfenergy``, which re-derives it from ``h0``),
    so the model must carry it.
    """
    if model.bath_states is None:
        raise ValueError(
            f"{func_name} requires model.bath_states (the valence/conduction bath split); "
            "build the model with it, e.g. ImpurityModel.from_blocks(..., bath_valence_conduction=(val, con))."
        )
    return model.bath_states


def _dc_operator(dc):
    """Build the double-counting one-body operator, ``dc[i, j] c^dagger_i c_j``."""
    return tensors_to_operator(np.asarray(dc, dtype=complex))


def _prepare_dc_solver(
    h_op, impurity_orbitals, bath_states, nominal_occ, mixed_valence, truncation_threshold, spin_flip_dj, tau, verbose
):
    """Build a many-body basis around ``nominal_occ`` and a CIPSI solver on it."""
    basis = Basis(
        impurity_orbitals,
        bath_states,
        nominal_impurity_occ=nominal_occ,
        mixed_valence=mixed_valence,
        truncation_threshold=truncation_threshold,
        verbose=verbose,
        comm=MPI.COMM_WORLD,
        spin_flip_dj=spin_flip_dj,
        tau=tau,
    )
    solver = CIPSISolver(basis)
    solver.truncate_initial(h_op)
    return basis, solver


def _lowest_energy_and_thermal_rho(basis, solver, h_op, impurity_indices, energy_cut, dense_cutoff, slaterWeightMin):
    """Lowest eigenvalue and thermally averaged impurity density matrix of ``h_op``."""
    es, psis = solver.get_eigenvectors(
        h_op,
        num_wanted=1,
        max_energy=energy_cut,
        dense_cutoff=dense_cutoff,
        slaterWeightMin=slaterWeightMin,
        solver="irlm",
    )
    rhos = build_density_matrices(
        basis,
        psis,
        orbital_indices_left=impurity_indices,
        orbital_indices_right=impurity_indices,
    )
    rho = thermal_average_scale_indep(es, rhos, basis.tau)
    # ``rhos`` is Allreduced in ``build_density_matrices`` and so is identical on every
    # rank, but ``es`` comes from the Lanczos kernel and is only replicated to roundoff
    # (MPI SUM reductions are not order-deterministic). The DC searches branch on this
    # energy -- ``fixed_peak_dc``'s Newton convergence and update -- so a value sitting on
    # a decision boundary could make ranks disagree about looping and deadlock on the next
    # collective solve. Broadcast rank 0's energy so every rank decides identically, the
    # same guard ``get_eigenvectors`` already applies to its own re-solve decision.
    lowest_energy = basis.comm.bcast(es[0], root=0) if basis.comm is not None else es[0]
    return lowest_energy, rho


def _noninteracting_impurity_occupation(h0_op, h_dc, impurity_indices, n_spin_orbitals, tau):
    r"""Thermal impurity occupation of the non-interacting ``h_loc`` at the Fermi level.

    Diagonalise the full one-body Hamiltonian ``h0`` (impurity *and* bath), occupy the
    single-particle levels with Fermi-Dirac statistics at chemical potential ``mu = 0`` -- the
    RSPt convention places the Fermi level at zero -- and trace the resulting one-particle
    density matrix over the impurity block:

    .. math::
        \rho = \sum_n f(\epsilon_n)\, |v_n\rangle\langle v_n|,\quad
        f(\epsilon) = \frac{1}{1 + e^{\epsilon / \tau}},\qquad
        N = \mathrm{Tr}\,\rho_{imp}.

    Because it hybridises the impurity with the bath before tracing, this is the DFT impurity
    occupation of a wide-window p-d model, which is the target :func:`fixed_occupation_dc` pins
    when the caller supplies none. It is a deterministic NumPy computation on the replicated
    ``h0`` (no MPI collective), so every rank obtains an identical value.

    Parameters
    ----------
    h0_op : dict or ManyBodyOperator
        Non-interacting Hamiltonian in single-index operator form (``model.h0``).
    h_dc : dict or ManyBodyOperator
        Double counting correction to be subtracted from h0 (``model.dc``).
    impurity_indices : sequence of int
        Impurity spin-orbital indices (the block traced over).
    n_spin_orbitals : int
        Total number of spin-orbitals (impurity + bath).
    tau : float
        Fundamental temperature ``k_B T`` in the energy units of ``h0``. ``tau <= 0`` fills
        every level below the Fermi level (a zero-temperature step).

    Returns
    -------
    float
        The non-interacting impurity occupation ``Tr rho_imp``.
    """
    h = extract_tensors(ManyBodyOperator(h0_op) - ManyBodyOperator(h_dc), n_orb=n_spin_orbitals, two_body=False)[0]
    energies, vecs = np.linalg.eigh(h)
    if tau > 0:
        # 1/(1 + exp(e/tau)) without overflow warnings: exp saturates to inf/0, giving f -> 0/1.
        with np.errstate(over="ignore"):
            occupations = 1.0 / (1.0 + np.exp(energies / tau))
    else:
        occupations = (energies < 0).astype(float)
    rho = (vecs * occupations) @ vecs.conj().T
    impurity_ix = np.ix_(list(impurity_indices), list(impurity_indices))
    return float(np.real(np.trace(rho[impurity_ix])))


def _refine_bracket(residual, mu_low, g_low, mu_high, g_high, tol, width_tol):
    r"""Safeguarded secant/bisection refinement of a bracket straddling a root.

    ``[mu_low, mu_high]`` must have residuals of opposite sign (``g_low`` and ``g_high``).
    A secant estimate that leaves the bracket, or hugs an endpoint, is replaced by the
    midpoint, guaranteeing a geometric decrease of the bracket width every step. This makes
    no assumption of monotonicity beyond the bracket invariant (opposite-sign endpoints),
    so it also handles a non-monotone residual with a single sign change in the bracket.

    Returns
    -------
    (mu, g) : tuple of float
        The best point found: the root itself, with ``|g| <= tol``, once met; otherwise the
        closer of the two endpoints of the fully narrowed bracket (a plateau/step in the
        residual, collapsed below ``width_tol`` without ever meeting ``tol``).
    """
    while mu_high - mu_low > width_tol:
        mu_mid = mu_high - g_high * (mu_high - mu_low) / (g_high - g_low) if g_high != g_low else np.inf
        # Safeguard: reject a secant estimate that leaves the bracket or hugs an endpoint,
        # keeping a guaranteed geometric decrease via bisection.
        margin = 0.01 * (mu_high - mu_low)
        if not (mu_low + margin <= mu_mid <= mu_high - margin):
            mu_mid = 0.5 * (mu_low + mu_high)
        g_mid = residual(mu_mid)
        if abs(g_mid) <= tol:
            return mu_mid, g_mid
        # Keep the sub-bracket that still straddles a root (opposite-sign endpoints);
        # correct regardless of whether the residual is monotone.
        if g_mid * g_low < 0:
            mu_high, g_high = mu_mid, g_mid
        else:
            mu_low, g_low = mu_mid, g_mid
    return (mu_low, g_low) if abs(g_low) <= abs(g_high) else (mu_high, g_high)


def _solve_dc_shift(
    observable,
    target,
    *,
    tol,
    width_tol,
    initial_step,
    max_shift,
    plateau_ok,
    unreachable_message,
    rank=0,
):
    r"""Find the uniform shift ``mu`` that drives a scalar observable onto ``target``.

    Generic root-finder shared by :func:`fixed_peak_dc` and :func:`fixed_occupation_dc`. The
    double counting is parametrized as ``dc(mu) = dc_guess + mu * identity``; the caller passes an
    ``observable(mu)`` closure that builds ``dc(mu)``, solves the model and returns the scalar to
    control (the peak position or the impurity occupation). No monotonicity in ``mu`` is assumed:
    the residual ``observable(mu) - target`` can be a difference of two independently-monotone
    quantities (e.g. an interacting and a non-interacting occupation), which is not itself
    monotone.

    The search scans both directions from ``mu = 0`` in geometrically growing steps
    (``initial_step, 2*initial_step, ...``, evaluated in a fixed +/- order each level so every
    rank makes the same sequence of collective calls). At each level it checks both new points
    for a direct hit (``|residual| <= tol``, returned immediately) and for a bracket (a
    sign change against that direction's previous point); any brackets found at that level are
    refined right away, nearest-``mu=0``-first, by :func:`_refine_bracket` (a safeguarded secant
    step with bisection fallback). Only if every bracket found so far collapses without meeting
    ``tol`` (a plateau) does the scan grow to the next level -- so a well-behaved, near-``mu=0``
    root is found as cheaply as the old single-direction search, while a residual with a false
    near bracket (e.g. a non-monotone ``n(mu) - n0(mu)``) still finds a genuine root farther out.
    The scan stops once both directions have exceeded ``max_shift``.

    Parameters
    ----------
    observable : callable
        ``observable(mu) -> float``. Evaluated collectively (it runs the eigensolver); call it the
        same number of times on every rank.
    target : float
        Requested observable value.
    tol : float
        Convergence tolerance on ``|observable - target|``.
    width_tol : float
        Stop refining once a bracket in ``mu`` is narrower than this (plateau detection).
    initial_step : float
        First bracketing step for ``|mu|``.
    max_shift : float
        Scanning a direction gives up once ``|mu|`` exceeds this.
    plateau_ok : bool
        If every bracket collapses without meeting ``tol`` (the observable steps across the
        target -- a plateau) and no bracket is found at all: ``True`` returns the closest side
        seen and warns on rank 0, ``False`` raises ``RuntimeError``.
    unreachable_message : str
        ``RuntimeError`` message when the target cannot be reached.
    rank : int
        MPI rank, for rank-0-only logging.

    Returns
    -------
    float
        The shift ``mu``.

    Raises
    ------
    RuntimeError
        If the target cannot be bracketed within ``max_shift`` in either direction (or every
        bracket collapses without meeting ``tol`` and ``plateau_ok=False``).
    """
    evaluated = {}

    def residual(mu):
        if mu not in evaluated:
            evaluated[mu] = observable(mu) - target
        return evaluated[mu]

    g0 = residual(0.0)
    if abs(g0) <= tol:
        return 0.0

    # Bidirectional geometric scan, fixed +1/-1 order per level (rank-invariant collective call
    # sequence). Brackets found at a level are refined immediately, before growing further.
    prev = {1: (0.0, g0), -1: (0.0, g0)}
    active = {1, -1}
    closest_unmet = None
    level = max(width_tol, initial_step)
    while active:
        level_brackets = []
        for direction in (1, -1):
            if direction not in active:
                continue
            mu = direction * level
            if abs(mu) > max_shift:
                active.discard(direction)
                continue
            g = residual(mu)
            if abs(g) <= tol:
                return mu
            mu_prev, g_prev = prev[direction]
            if g * g_prev < 0:
                bracket = (mu_prev, g_prev, mu, g) if mu_prev < mu else (mu, g, mu_prev, g_prev)
                level_brackets.append(bracket)
            prev[direction] = (mu, g)

        # Refine this level's brackets nearest-mu=0-first -- the smallest correction to the guess
        # wins when the residual has more than one root (the non-monotone case can bracket both
        # directions at the same level). Sorted by nearest bracket *endpoint*, not nearest root:
        # two brackets tied on that endpoint distance break ties by scan order (+1 before -1),
        # which is only guaranteed optimal when each bracket holds at most one root -- true for
        # every criterion in this module (peak position, occupation).
        level_brackets.sort(key=lambda b: min(abs(b[0]), abs(b[2])))
        for mu_low, g_low, mu_high, g_high in level_brackets:
            mu_c, g_c = _refine_bracket(residual, mu_low, g_low, mu_high, g_high, tol, width_tol)
            if abs(g_c) <= tol:
                return mu_c
            # Bracket collapsed without meeting tol (a plateau/step): remember the closest point
            # reached in case every bracket ever found does this.
            if closest_unmet is None or abs(g_c) < abs(closest_unmet[1]):
                closest_unmet = (mu_c, g_c)
        level *= 2

    if closest_unmet is not None:
        mu, g = closest_unmet
        if not plateau_ok:
            raise RuntimeError(unreachable_message.format(mu=mu, value=g + target, target=target))
        if rank == 0:
            print(
                f"WARNING: the requested double-counting target {target} falls on a plateau; the "
                f"closest achievable observable is {g + target:.4f} (mu = {mu:.6f})."
            )
        return mu

    # Neither direction ever bracketed the target within max_shift; report the closer of the two
    # farthest points actually probed.
    best_mu, best_g = min(prev.values(), key=lambda mu_g: abs(mu_g[1]))
    raise RuntimeError(unreachable_message.format(mu=best_mu, value=best_g + target, target=target))


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

    Note: the many-body bases are expanded once, with the guess double
    counting; the search reuses them. Energies carry no fixed unit, they follow
    the inputs (e.g. Ry when called from RSPt); the convergence tolerance is
    ``max(tau, 1e-4)`` in those units.

    Parameters
    ----------
    model : impurityModel.ed.model.ImpurityModel
        The impurity problem: ``h0`` (non-interacting Hamiltonian), ``u4`` (Coulomb tensor),
        ``dc`` (double counting correction, used as the search's starting guess),
        ``impurity_orbitals`` and ``bath_states`` -- the ``(valence, conduction)`` bath split is
        required here (build the model with it, e.g. ``from_blocks(..., bath_valence_conduction=...)``).
    basis : impurityModel.ed.model.BasisOptions
        Nominal occupation (``{group: N}``; a single group only -- with more groups it is
        ambiguous which gains/loses the electron), mixed valence, spin-flip determinants,
        temperature and the determinant budget.
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
    # Unpack the grouped parameters into the local names used throughout the body.
    h0_op = model.h0
    u = model.u4
    n_imp = len(model.impurity_indices)
    dc_guess = extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0]
    impurity_orbitals = model.impurity_orbitals
    bath_states = _require_bath_states(model, "fixed_peak_dc")
    N0 = basis.nominal_occ
    mixed_valence = basis.mixed_valence
    spin_flip_dj = basis.spin_flip_dj
    tau = basis.tau
    truncation_threshold = basis.truncation_threshold
    slaterWeightMin = basis.slater_weight_min
    dense_cutoff = solver.dense_cutoff
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0

    if len(N0) != 1:
        raise ValueError(
            f"fixed_peak_dc supports a single impurity group, got N0 = {N0}. "
            "With multiple groups it is ambiguous which group gains/loses the electron."
        )
    h_op_i = ManyBodyOperator(h0_op) + ManyBodyOperator(u)
    impurity_orbitals, bath_states = _normalize_dc_orbitals(impurity_orbitals, bath_states)

    # Keep the requested peak outside the thermal broadening, preserving the
    # sign: a negative peak position places a removal peak at E[N] - E[N-1].
    if peak_position >= 0:
        peak_position = max(peak_position, 4 * tau)
        occ_upper = {i: N0[i] + 1 for i in N0}
        occ_lower = dict(N0)
    else:
        peak_position = min(peak_position, -4 * tau)
        occ_upper = dict(N0)
        occ_lower = {i: N0[i] - 1 for i in N0}

    basis_upper, solver_upper = _prepare_dc_solver(
        h_op_i,
        impurity_orbitals,
        bath_states,
        occ_upper,
        mixed_valence,
        truncation_threshold,
        spin_flip_dj,
        tau,
        verbose,
    )
    basis_lower, solver_lower = _prepare_dc_solver(
        h_op_i,
        impurity_orbitals,
        bath_states,
        occ_lower,
        mixed_valence,
        truncation_threshold,
        spin_flip_dj,
        tau,
        verbose,
    )

    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]
    identity = np.identity(dc_guess.shape[0])

    # Expand the many-body bases once, with the guess double counting.
    h_guess = h_op_i - ManyBodyOperator(model.dc)
    solver_upper.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-5, slaterWeightMin=slaterWeightMin)
    solver_lower.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-5, slaterWeightMin=slaterWeightMin)

    energy_cut = -tau * np.log(1e-4)

    def peak_observable(mu):
        h_op = h_op_i - _dc_operator(dc_guess + mu * identity)
        e_upper, _ = _lowest_energy_and_thermal_rho(
            basis_upper, solver_upper, h_op, impurity_indices, energy_cut, dense_cutoff, slaterWeightMin
        )
        e_lower, _ = _lowest_energy_and_thermal_rho(
            basis_lower, solver_lower, h_op, impurity_indices, energy_cut, dense_cutoff, slaterWeightMin
        )
        if verbose and rank == 0:
            print(f"mu={mu:.6f} E_upper - E_lower={e_upper - e_lower:.6f}")
        return e_upper - e_lower

    # Scale the bracketing to the non-interacting bandwidth (the spread of the one-body h0
    # eigenvalues); the peak position responds to the shift with slope ~ -1, so this comfortably
    # covers the reachable range. An observable that does not move with mu (the upper and lower
    # sectors hold the same impurity occupation -- the old delta_n ~ 0 ill-conditioning) never
    # brackets and surfaces here as the unreachable RuntimeError.
    h1 = extract_tensors(ManyBodyOperator(h0_op), n_orb=model.n_spin_orbitals, two_body=False)[0]
    bandwidth = float(np.ptp(np.linalg.eigvalsh(h1)))
    tol = max(tau, 1e-4)
    unreachable = (
        "The fixed-peak double counting could not place the peak at {target}: E_upper - E_lower "
        "reached {value:.4f} at mu = {mu:.3f}. The upper and lower sectors may hold equal impurity "
        "occupation (a uniform shift cannot move the peak), or the target lies beyond the "
        "reachable range."
    )
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
    Calculate the double counting correction using a fixed impurity occupation criterion.

    Choose the double counting so that the thermal impurity occupation equals
    the requested value, :math:`\mathrm{Tr}\,\rho_{imp} = N_{target}`.

    The double counting is parametrized as a uniform shift of the guess,
    ``dc(mu) = dc_guess + mu * identity``. The shift couples to the impurity
    occupation as :math:`-\mu \hat N_{imp}`, so
    :math:`\langle N_{imp}\rangle(\mu)` is non-decreasing and the scalar shift
    is found by the shared secant/bisection search :func:`_solve_dc_shift`. At
    low temperature and weak hybridization the occupation approaches a staircase
    in ``mu``; if the requested (fractional) occupation falls on a plateau,
    the search converges to the closest step and a warning is printed.

    When ``occupation`` is not supplied, the target is derived from the
    non-interacting ``h_loc``: the full one-body ``model.h0`` (impurity + bath)
    is diagonalised, Fermi-filled at :math:`\mu = 0` (the RSPt Fermi level) at
    temperature ``tau``, and the impurity block of the resulting density matrix
    is traced (:func:`_noninteracting_impurity_occupation`). That is the DFT
    impurity occupation, so the fixed-occupation double counting can be used for
    CSC DFT+DMFT of wide-window p-d models without the caller knowing the
    filling in advance.

    Note: the total electron number is conserved, so the impurity occupation
    changes through impurity-bath charge transfer; the reachable occupations
    are limited by the bath. The many-body basis is expanded once, with the
    guess double counting.

    Parameters other than the following match :func:`fixed_peak_dc` (``model``, ``basis``,
    ``solver``, ``comm``, ``verbosity``). ``basis.nominal_occ`` is the nominal impurity
    occupation used to build the many-body basis; use the integer occupation closest to the
    requested one.

    Parameters
    ----------
    occupation : float or None
        Requested impurity occupation (may be fractional). ``None`` derives the target from the
        non-interacting ``h_loc`` (see above).
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
        If the requested occupation cannot be bracketed within ``max_shift``.
    """
    # Unpack the grouped parameters into the local names used throughout the body.
    h0_op = model.h0
    u = model.u4
    n_imp = len(model.impurity_indices)
    dc_guess = extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0]
    impurity_orbitals = model.impurity_orbitals
    bath_states = _require_bath_states(model, "fixed_occupation_dc")
    N0 = basis.nominal_occ
    mixed_valence = basis.mixed_valence
    spin_flip_dj = basis.spin_flip_dj
    tau = basis.tau
    truncation_threshold = basis.truncation_threshold
    slaterWeightMin = basis.slater_weight_min
    dense_cutoff = solver.dense_cutoff
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0

    h_op_i = ManyBodyOperator(h0_op) + ManyBodyOperator(u)
    impurity_orbitals, bath_states = _normalize_dc_orbitals(impurity_orbitals, bath_states)

    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]
    total_impurity_orbitals = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)

    # No target supplied: pin the DFT occupation, i.e. the Fermi-filled occupation of the
    # non-interacting h_loc (at the guess dc) at mu=0. This is the target for wide-window p-d
    # CSC DFT+DMFT.
    if occupation is None:
        occupation = _noninteracting_impurity_occupation(
            model.h0, model.dc, impurity_indices, model.n_spin_orbitals, tau
        )
        if verbose and rank == 0:
            print(f"Fixed-occupation double counting: target derived from h_loc = {occupation:.4f}")
    # Tolerance absorbs the roundoff of the auto-derived target (a sum of Fermi occupations,
    # each in [0, 1], that can land a ULP outside the exact bound).
    if not -1e-9 <= occupation <= total_impurity_orbitals + 1e-9:
        raise ValueError(f"Requested impurity occupation {occupation} outside [0, {total_impurity_orbitals}].")

    # Local many-body basis / CIPSI solver (distinct from the BasisOptions/SolverOptions params).
    mb_basis, mb_solver = _prepare_dc_solver(
        h_op_i, impurity_orbitals, bath_states, N0, mixed_valence, truncation_threshold, spin_flip_dj, tau, verbose
    )
    identity = np.identity(dc_guess.shape[0])

    # Expand the many-body basis once, with the guess double counting.
    h_guess = h_op_i - ManyBodyOperator(model.dc)
    mb_solver.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-5, slaterWeightMin=slaterWeightMin)

    energy_cut = -tau * np.log(1e-4)

    # Cache each evaluated occupation so the final log reuses it instead of re-solving.
    occupation_at = {}

    def impurity_occupation(mu):
        h_op = h_op_i - _dc_operator(dc_guess + mu * identity)
        _, rho = _lowest_energy_and_thermal_rho(
            mb_basis, mb_solver, h_op, impurity_indices, energy_cut, dense_cutoff, slaterWeightMin
        )
        n = float(np.real(np.trace(rho)))
        occupation_at[mu] = n
        if verbose and rank == 0:
            print(f"mu={mu:.6f} n={n:.6f}")
        return n

    unreachable = (
        "Could not bracket the requested impurity occupation {target} with "
        f"|mu| <= {max_shift}: " + "the occupation reached {value:.4f} at mu = {mu:.3f}. "
        "The target may be unreachable with the available bath states."
    )
    mu = _solve_dc_shift(
        impurity_occupation,
        occupation,
        tol=occ_tol,
        width_tol=max(tau, 1e-4),
        initial_step=max(10 * tau, initial_step),
        max_shift=max_shift,
        plateau_ok=True,
        unreachable_message=unreachable,
        rank=rank,
    )

    n = occupation_at.get(mu, occupation)
    dc = dc_guess + mu * identity
    if verbose and rank == 0:
        print(f"Fixed-occupation double counting (target = {occupation}, achieved = {n:.4f}, mu = {mu:.6f}):")
        matrix_print(dc_guess, label="DC guess:")
        matrix_print(dc, label="DC found:")
    return dc
