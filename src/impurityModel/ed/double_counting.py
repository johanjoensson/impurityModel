r"""Double-counting determination for the self-energy workflow.

The double counting is a first-class part of the model (``model.dc``), always subtracted from
the non-interacting Hamiltonian: the local Hamiltonian is ``h0 - dc + U(u4)`` throughout this
module and in :func:`solver_basis.prepare_solver_basis` / :func:`susceptibility` (never added).
The impurity double-counting potential is fixed by one of two searches over a uniform shift
``dc(mu) = dc_guess + mu * identity``: :func:`fixed_peak_dc` pins a chosen spectral peak,
:func:`fixed_occupation_dc` pins the impurity occupation. Both are special cases of one
generic, non-monotonicity-safe search, :func:`_solve_dc_shift`. At a trial shift, both determine
the ground-state sector the identical way :func:`groundstate.calc_gs` does
(:func:`groundstate.find_ground_state_basis`'s HF-seed-then-walk search) rather than searching on
a sector fixed at the input occupation -- a dc measured on a different sector than the one
calc_selfenergy later finds would misdirect, not approximate, the requested physics. Each then
reads off a scalar residual (the peak position offset, or the occupation offset), and the search
drives it to zero with a bidirectional bracketing search safeguarded by secant/bisection
refinement.
:func:`fixed_occupation_dc` also accepts no explicit target at all: it then pins the interacting
occupation to the DFT reference occupation :math:`N_0` -- the Fermi filling of the *raw* ``h0``
(:func:`_noninteracting_impurity_occupation`), computed once before the search -- the natural
target for CSC DFT+DMFT of wide-window p-d models. ``model.h0`` is the KS/DFT Hamiltonian of the
``h0 - dc + U`` contract; its raw filling *is* the DFT occupation, and no double counting is
subtracted before filling. The self-energy extraction proper lives in :mod:`sigma`; the
orchestration and CLI in :mod:`selfenergy`, which re-exports
``fixed_peak_dc``/``fixed_occupation_dc`` so existing callers are unchanged.

Like :func:`selfenergy.calc_selfenergy` and :func:`groundstate.find_ground_state_basis`, both
searches derive their many-body basis's determinant budget from available per-rank memory
(:func:`impurityModel.ed.memory_estimate.suggest_truncation_threshold`) when
``BasisOptions.truncation_threshold`` is left at its default ``None``, and honor
``BasisOptions.excitation_budget``/``chain_restrict`` via the same
:func:`impurityModel.ed.basis_restrictions.build_weighted_restrictions` the other ED drivers use --
the double counting is otherwise found on a different variational space than the solve that will
use it.

Three static schemes complement the two searches above: :func:`fll_dc` (Fully Localized Limit),
:func:`amf_dc` (Around Mean Field) and :func:`sigma_inf_dc` (K. Held's :math:`\Sigma(\infty)`,
the full static Hartree-Fock self-energy matrix, :func:`impurityModel.ed.sigma.get_Sigma_static`).
Unlike the two searches, these are deterministic one-body NumPy computations with no ED solve and
no MPI collective, identical on every rank. Each evaluates its formula at the DFT impurity
occupation or density matrix -- the Fermi filling of the raw ``h0``
(:func:`_noninteracting_impurity_rho`), the same reference :func:`fixed_occupation_dc` targets --
unless the caller supplies the occupation/density matrix explicitly.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.atomic_physics import uj_from_u4
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_restrictions import build_weighted_restrictions
from impurityModel.ed.basis_transcription import build_density_matrices
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.lie_algebra import extract_tensors, rotate_two_body, tensors_to_operator
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.memory_estimate import DEFAULT_MEMORY_SAFETY, log_memory_budget, suggest_truncation_threshold
from impurityModel.ed.sigma import get_Sigma_static
from impurityModel.ed.solver_basis import prepare_solver_basis
from impurityModel.ed.utils import matrix_print


def _dc_operator(dc):
    """Build the double-counting one-body operator, ``dc[i, j] c^dagger_i c_j``."""
    return tensors_to_operator(np.asarray(dc, dtype=complex))


def _noninteracting_impurity_rho(h0_op, impurity_indices, n_spin_orbitals, tau):
    r"""Thermal impurity density matrix of the non-interacting ``h0`` at the Fermi level.

    Diagonalise the full one-body Hamiltonian ``h0`` (impurity *and* bath), occupy the
    single-particle levels with Fermi-Dirac statistics at chemical potential ``mu = 0`` -- the
    RSPt convention places the Fermi level at zero -- and return the impurity block of the
    resulting one-particle density matrix:

    .. math::
        \rho = \sum_n f(\epsilon_n)\, |v_n\rangle\langle v_n|,\quad
        f(\epsilon) = \frac{1}{1 + e^{\epsilon / \tau}}.

    Because it hybridises the impurity with the bath before tracing, ``Tr rho_imp`` is the DFT
    impurity occupation of a wide-window p-d model, which is the target :func:`fixed_occupation_dc`
    pins when the caller supplies none. ``model.h0`` is the KS/DFT Hamiltonian of the
    ``h0 - dc + U`` model contract (it already contains the DFT mean-field interaction -- that is
    what the double counting corrects for), so the DFT reference is the filling of the *raw*
    ``h0`` and this function deliberately takes no double counting: subtracting a realistic dc
    first sinks the impurity levels far below the Fermi level and saturates the occupation at the
    full shell (NiO: n0 pinned at 10 instead of ~8.6 for a ~4 Ry dc). It is a deterministic NumPy
    computation on the replicated ``h0`` (no MPI collective), so every rank obtains an identical
    value.

    Parameters
    ----------
    h0_op : dict or ManyBodyOperator
        Non-interacting Hamiltonian in single-index operator form (``model.h0``).
    impurity_indices : sequence of int
        Impurity spin-orbital indices (the block traced over).
    n_spin_orbitals : int
        Total number of spin-orbitals (impurity + bath).
    tau : float
        Fundamental temperature ``k_B T`` in the energy units of ``h0``. ``tau <= 0`` fills
        every level below the Fermi level (a zero-temperature step).

    Returns
    -------
    numpy.ndarray
        The impurity block of the density matrix, ``(n_imp, n_imp)`` complex.
    """
    h = extract_tensors(ManyBodyOperator(h0_op), n_orb=n_spin_orbitals, two_body=False)[0]
    energies, vecs = np.linalg.eigh(h)
    if tau > 0:
        # 1/(1 + exp(e/tau)) without overflow warnings: exp saturates to inf/0, giving f -> 0/1.
        with np.errstate(over="ignore"):
            occupations = 1.0 / (1.0 + np.exp(energies / tau))
    else:
        occupations = (energies < 0).astype(float)
    rho = (vecs * occupations) @ vecs.conj().T
    impurity_ix = np.ix_(list(impurity_indices), list(impurity_indices))
    return rho[impurity_ix]


def _noninteracting_impurity_occupation(h0_op, impurity_indices, n_spin_orbitals, tau):
    """Thermal impurity occupation ``Tr rho_imp``; see :func:`_noninteracting_impurity_rho`."""
    rho = _noninteracting_impurity_rho(h0_op, impurity_indices, n_spin_orbitals, tau)
    return float(np.real(np.trace(rho)))


def _model_u4_dense(model):
    r"""Recover the dense, impurity-local RSPt Coulomb tensor from ``model.u4``.

    ``model.u4`` is the *raw* (never canonicalized) operator dict built by
    :func:`impurityModel.ed.atomic_physics.getUop_from_rspt_u4`: one term per index quadruple,
    amplitude ``u4[i,j,k,l] / 2``, with no folding of equivalent terms. Wrapping it in a
    :class:`ManyBodyOperator` first (which canonicalizes/folds terms together) would lose the
    direct/exchange split this relies on -- do not do that before calling this function.
    :func:`extract_tensors` on the raw dict gives ``V[i,j,k,l] = u4[i,j,k,l] / 2`` entry-for-entry
    (no two raw keys ever map to the same ``V`` cell), so ``2 * V`` already recovers ``u4``
    exactly. ``V + V.transpose(1, 0, 3, 2)`` is used instead of a bare ``2 * V`` because it is
    numerically identical for a raw dict (via ``u4``'s own exchange symmetry, ``u4[i,j,k,l] =
    u4[j,i,l,k]``, which any RSPt-convention tensor satisfies by construction) while also being
    the natural, symmetric way to state "recover the tensor from the operator".

    Parameters
    ----------
    model : ImpurityModel

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp, n_imp, n_imp)
        The Coulomb tensor over the impurity spin-orbitals, in the model's input basis.

    Raises
    ------
    ValueError
        If ``model.u4`` is ``None``.
    """
    if model.u4 is None:
        raise ValueError("model.u4 is None; pass an explicit u=/j= (or n=/rho=) instead of deriving them from u4.")
    n_imp = len(model.impurity_indices)
    _, V, _ = extract_tensors(model.u4, n_orb=n_imp)
    return V + V.transpose(1, 0, 3, 2)


def _model_uj(model):
    r"""Average Coulomb repulsion and exchange (:func:`uj_from_u4`) derived from ``model.u4``.

    Rotates the dense impurity Coulomb tensor (:func:`_model_u4_dense`) into the spherical
    basis with :func:`impurityModel.ed.lie_algebra.rotate_two_body` (the same transformation as
    :func:`impurityModel.ed.greens_function.rotate_4index_U`, avoiding an import of the
    heavyweight solver module) using ``model.rot_to_spherical``, then reads off ``(U, J)``.

    Parameters
    ----------
    model : ImpurityModel

    Returns
    -------
    (U, J) : tuple of float

    Raises
    ------
    ValueError
        If ``model.u4`` is ``None``, or ``model.rot_to_spherical`` is a multi-group dict (per-
        group rotations are not supported here, matching :func:`fixed_peak_dc`'s restriction).
    """
    if isinstance(model.rot_to_spherical, dict):
        raise ValueError("_model_uj does not support a multi-group model.rot_to_spherical; pass explicit u=/j=.")
    u4_dense = _model_u4_dense(model)
    rotation = np.asarray(model.rot_to_spherical, dtype=complex)
    u4_spherical = rotate_two_body(u4_dense, rotation)
    return uj_from_u4(u4_spherical)


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

    At every trial ``mu`` the *center* sector ``N`` is not the input ``model.impurity_orbitals``
    occupation: it is found by :func:`groundstate.find_ground_state_basis`, the identical
    HF-seed-then-walk search :func:`groundstate.calc_gs` uses for the selfenergy/spectra solve.
    A peak measured relative to a different center than the one calc_selfenergy later finds would
    not approximate the requested peak position, it would misplace it -- the same reasoning that
    drives :func:`fixed_occupation_dc`'s sector determination. The ``N +- 1`` sectors are then
    each a single, fixed-occupation solve (:func:`groundstate.calc_energy`) relative to that
    found center, not a further search. Energies carry no fixed unit, they follow the inputs
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
        Nominal occupation (``{group: N}``; a single group only -- with more groups it is
        ambiguous which gains/loses the electron), mixed valence, spin-flip determinants,
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
    from impurityModel.ed import product_state_representation as psr
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

    if len(N0) != 1:
        raise ValueError(
            f"fixed_peak_dc supports a single impurity group, got N0 = {N0}. "
            "With multiple groups it is ambiguous which group gains/loses the electron."
        )
    (group_key,) = N0.keys()

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
    max_occ_group = sum(len(block) for block in impurity_orbitals[group_key])
    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]
    identity = np.identity(dc_guess.shape[0])

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
            mixed_valence=mixed_valence,
            tau=tau,
            chain_restrict=chain_restrict,
            dense_cutoff=dense_cutoff,
            spin_flip_dj=spin_flip_dj,
            comm=MPI.COMM_WORLD,
            verbose=verbose,
            truncation_threshold=truncation_threshold,
            slaterWeightMin=slaterWeightMin,
            weighted_restrictions=weighted_restrictions,
        )
        # A single group, so every determinant in the winning sector shares the same impurity
        # occupation; read it off one representative determinant.
        state = next(iter(basis_center))
        occ = psr.bytes2tuple(bytes(state.to_bytearray()), model.n_spin_orbitals)
        n_center = sum(1 for o in occ if o in impurity_indices)

        occ_upper = {group_key: n_center + 1 if addition else n_center}
        occ_lower = {group_key: n_center if addition else n_center - 1}

        def sector_energy(occ_trial):
            if not 0 <= occ_trial[group_key] <= max_occ_group:
                return unreachable_penalty
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
            )
            # calc_energy itself returns inf for an empty basis (e.g. a sector the current
            # restrictions admit no determinants for); clamp to the same finite penalty for the
            # same reason as the bounds check above.
            return e_trial if np.isfinite(e_trial) else unreachable_penalty

        e_upper = sector_energy(occ_upper)
        e_lower = sector_energy(occ_lower)
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

    # DFT reference occupation: Fermi filling of the raw h0 (the KS Hamiltonian of the
    # h0 - dc + U contract; no double counting subtracted before filling), independent of
    # dc_guess and of the trial shift, and of the solver-basis rotation (h0's raw filling is
    # evaluated in the model's input basis). Logged at every trial; with occupation=None it is
    # also the search target.
    n0 = _noninteracting_impurity_occupation(h0_op, impurity_indices, model.n_spin_orbitals, tau)
    self_consistent = occupation is None
    if self_consistent:
        occupation = n0
        if (n0 >= total_impurity_orbitals - occ_tol or n0 <= occ_tol) and rank == 0:
            # Saturation is a threshold phenomenon of coarse discretizations, not a smooth
            # error: impurity weight crosses the Fermi level only when level repulsion pushes
            # a mixed state through it (NiO: n0 = 8.63 at 15 bath states per block, exactly
            # 10.0 at 1 and 5).
            print(
                f"WARNING: the DFT reference occupation N0 = {n0:.4f} is saturated at the "
                f"{'full' if n0 > occ_tol else 'empty'} impurity shell "
                f"(of {total_impurity_orbitals} spin-orbitals): the discretized bath places no "
                "impurity spectral weight across the Fermi level (typical for coarse "
                "valence-only bath fits), so the self-consistent occupation criterion is "
                "meaningless and the search will very likely fail as unreachable. Supply an "
                "explicit occupation target instead (RSPt interface: 'occ <N>' on the "
                "double-counting line), or improve the bath discretization around the Fermi "
                "level.",
                flush=True,
            )
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
            mixed_valence=mixed_valence,
            tau=tau,
            chain_restrict=chain_restrict,
            dense_cutoff=dense_cutoff,
            spin_flip_dj=spin_flip_dj,
            comm=MPI.COMM_WORLD,
            verbose=verbose,
            truncation_threshold=truncation_threshold,
            slaterWeightMin=slaterWeightMin,
            weighted_restrictions=weighted_restrictions,
        )
        mb_solver = CIPSISolver(mb_basis)
        # Refine at the same de2_min calc_gs uses for its own final solve (find_ground_state_basis
        # itself only expands to 1e-6, tight enough to compare sectors but not to match the
        # occupation calc_selfenergy will report at this same dc).
        mb_solver.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-8, slaterWeightMin=slaterWeightMin)

        def solve_thermal_occupation():
            # The NiO ground state that motivated this rework is 3-fold quasi-degenerate, so a
            # single lowest state (as the frozen-basis search used) misrepresents the thermal
            # occupation; widen num_wanted until the manifold within energy_cut is captured.
            # The termination test on Lanczos energies is only replicated to roundoff, so
            # broadcast rank 0's decision -- ranks disagreeing here would diverge on whether to
            # re-enter the next collective get_eigenvectors call.
            num_wanted = 10
            while True:
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


def fll_dc(model, *, tau=0.002, n=None, u=None, j=None):
    r"""Fully Localized Limit double counting, ``dc = [U(N - 1/2) - (J/2)(N - 1)] I``.

    ``U``, ``J`` default to :func:`_model_uj`'s spherical average (needs ``model.u4``); either
    may be overridden explicitly (e.g. from tabulated values), independently of the other. ``N``
    defaults to the DFT impurity occupation, the Fermi filling of the raw ``h0``
    (:func:`_noninteracting_impurity_rho`); an explicit ``N`` (together with both ``u`` and
    ``j``) needs no ``model.u4`` at all.

    Parameters
    ----------
    model : ImpurityModel
    tau : float, optional
        Fundamental temperature for the DFT occupation (ignored if ``n`` is given).
    n : float, optional
        Impurity occupation ``N``. ``None`` uses the DFT impurity occupation.
    u, j : float, optional
        Average Coulomb repulsion and exchange. ``None`` derives them from ``model.u4`` via
        :func:`_model_uj`.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    n_imp = len(model.impurity_indices)
    if u is None or j is None:
        u_auto, j_auto = _model_uj(model)
        u = u_auto if u is None else u
        j = j_auto if j is None else j
    identity = np.identity(n_imp, dtype=complex)
    if n is None:
        n = _noninteracting_impurity_occupation(model.h0, model.impurity_indices, model.n_spin_orbitals, tau)
    return (u * (n - 0.5) - 0.5 * j * (n - 1.0)) * identity


def amf_dc(model, *, tau=0.002, n=None):
    r"""Around Mean Field double counting, ``dc = Σ_static(u4, (N / n_imp) I)``.

    The static Hartree-Fock self-energy (:func:`impurityModel.ed.sigma.get_Sigma_static`)
    evaluated at a *uniform* trial density matrix, ``N`` spread evenly over every impurity
    spin-orbital -- the defining assumption of AMF, that the impurity has no orbital *or spin*
    polarization -- as opposed to :func:`sigma_inf_dc`, which uses the actual (possibly
    anisotropic) density matrix. For a spin-polarized ground state this is the paramagnetic
    (spin-blind) AMF potential, not a per-spin-channel one. ``N`` defaults to the DFT impurity
    occupation, the Fermi filling of the raw ``h0`` (:func:`_noninteracting_impurity_rho`).

    Parameters
    ----------
    model : ImpurityModel
    tau : float, optional
        Fundamental temperature for the DFT occupation (ignored if ``n`` is given).
    n : float, optional
        Impurity occupation ``N``. ``None`` uses the DFT impurity occupation.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    n_imp = len(model.impurity_indices)
    u4_dense = _model_u4_dense(model)
    identity = np.identity(n_imp, dtype=complex)
    if n is None:
        n = _noninteracting_impurity_occupation(model.h0, model.impurity_indices, model.n_spin_orbitals, tau)
    return get_Sigma_static(u4_dense, (n / n_imp) * identity)


def sigma_inf_dc(model, *, tau=0.002, rho=None):
    r"""K. Held's :math:`\Sigma(\infty)` double counting: the full static Hartree-Fock
    self-energy matrix, ``dc = Σ_static(u4, rho_imp)``.

    Unlike :func:`amf_dc`, uses the actual (possibly anisotropic) non-interacting impurity
    density matrix rather than a uniform trial -- the two agree exactly when that density matrix
    happens to be uniform (e.g. a single, orbitally-degenerate shell), and differ whenever the
    impurity levels split. ``rho`` defaults to the DFT impurity density matrix, the Fermi
    filling of the raw ``h0`` (:func:`_noninteracting_impurity_rho`).

    Parameters
    ----------
    model : ImpurityModel
    tau : float, optional
        Fundamental temperature for the DFT density matrix (ignored if ``rho`` is given).
    rho : numpy.ndarray, shape (n_imp, n_imp), optional
        Impurity density matrix. ``None`` uses the DFT impurity density matrix.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    u4_dense = _model_u4_dense(model)
    if rho is None:
        rho = _noninteracting_impurity_rho(model.h0, model.impurity_indices, model.n_spin_orbitals, tau)
    return get_Sigma_static(u4_dense, np.asarray(rho, dtype=complex))
