r"""Double-counting determination for the self-energy workflow.

The double counting is a first-class part of the model (``model.dc``), always subtracted from
the non-interacting Hamiltonian: the local Hamiltonian is ``h0 - dc + U(u4)`` throughout this
module and in :func:`selfenergy._prepare_solver_basis` / :func:`susceptibility` (never added).
The impurity double-counting potential is fixed by one of two searches over a uniform shift
``dc(mu) = dc_guess + mu * identity``: :func:`fixed_peak_dc` pins a chosen spectral peak,
:func:`fixed_occupation_dc` pins the impurity occupation. Both are special cases of one
generic, non-monotonicity-safe search, :func:`_solve_dc_shift`: at a trial shift it builds the
variational ground state and its thermal density matrix (:func:`_lowest_energy_and_thermal_rho`),
reads off a scalar residual (the peak position offset, or the occupation offset), and drives it
to zero with a bidirectional bracketing search safeguarded by secant/bisection refinement.
:func:`fixed_occupation_dc` also accepts no explicit target at all: it then solves the
self-consistent condition that the interacting and non-interacting (Fermi-filled ``h0 - dc(mu)``,
:func:`_noninteracting_impurity_occupation`) impurity occupations agree, :math:`N(\mu) =
N_0(\mu)` -- the natural target for CSC DFT+DMFT of wide-window p-d models -- re-evaluating both
sides at every trial shift. The self-energy extraction proper lives in :mod:`sigma`; the
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
no MPI collective, identical on every rank. Each needs the non-interacting impurity occupation or
density matrix *at self-consistency with its own output* (the double counting shifts the impurity
level, which shifts the non-interacting occupation the scheme reads back in), so each iterates a
damped fixed point (:func:`_iterate_dc_fixed_point`) from ``model.dc`` unless the caller supplies
the occupation/density matrix explicitly, which skips the iteration entirely.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.atomic_physics import uj_from_u4
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_restrictions import build_weighted_restrictions
from impurityModel.ed.basis_transcription import build_density_matrices
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.lie_algebra import extract_tensors, rotate_two_body, tensors_to_operator
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.memory_estimate import DEFAULT_MEMORY_SAFETY, log_memory_budget, suggest_truncation_threshold
from impurityModel.ed.sigma import get_Sigma_static
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
    h_op,
    impurity_orbitals,
    bath_states,
    nominal_occ,
    mixed_valence,
    truncation_threshold,
    spin_flip_dj,
    tau,
    verbose,
    weighted_restrictions=None,
    chain_restrict=False,
):
    """Build a many-body basis around ``nominal_occ`` and a CIPSI solver on it."""
    basis = Basis(
        impurity_orbitals,
        bath_states,
        nominal_impurity_occ=nominal_occ,
        mixed_valence=mixed_valence,
        truncation_threshold=truncation_threshold,
        weighted_restrictions=weighted_restrictions,
        chain_restrict=chain_restrict,
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


def _noninteracting_impurity_rho(h0_op, h_dc, impurity_indices, n_spin_orbitals, tau):
    r"""Thermal impurity density matrix of the non-interacting ``h_loc`` at the Fermi level.

    Diagonalise the full one-body Hamiltonian ``h0`` (impurity *and* bath), occupy the
    single-particle levels with Fermi-Dirac statistics at chemical potential ``mu = 0`` -- the
    RSPt convention places the Fermi level at zero -- and return the impurity block of the
    resulting one-particle density matrix:

    .. math::
        \rho = \sum_n f(\epsilon_n)\, |v_n\rangle\langle v_n|,\quad
        f(\epsilon) = \frac{1}{1 + e^{\epsilon / \tau}}.

    Because it hybridises the impurity with the bath before tracing, ``Tr rho_imp`` is the DFT
    impurity occupation of a wide-window p-d model, which is the target :func:`fixed_occupation_dc`
    pins when the caller supplies none. It is a deterministic NumPy computation on the replicated
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
    numpy.ndarray
        The impurity block of the density matrix, ``(n_imp, n_imp)`` complex.
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
    return rho[impurity_ix]


def _noninteracting_impurity_occupation(h0_op, h_dc, impurity_indices, n_spin_orbitals, tau):
    """Thermal impurity occupation ``Tr rho_imp``; see :func:`_noninteracting_impurity_rho`."""
    rho = _noninteracting_impurity_rho(h0_op, h_dc, impurity_indices, n_spin_orbitals, tau)
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

    Note: the many-body bases are expanded once, with the guess double
    counting; the search reuses them. Energies carry no fixed unit, they follow
    the inputs (e.g. Ry when called from RSPt); the convergence tolerance is
    ``max(tau, 1e-4)`` in those units. Both sector bases honor
    ``BasisOptions.excitation_budget``/``chain_restrict`` identically -- the bath-only
    excitation-budget restriction never references the occupation, so the same restriction list
    applies to the N0 and N0 +- 1 sectors, matching :func:`groundstate.find_ground_state_basis`'s
    own occupation-scan trials.

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
    excitation_budget = basis.excitation_budget
    chain_restrict = basis.chain_restrict
    slaterWeightMin = basis.slater_weight_min
    dense_cutoff = solver.dense_cutoff
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0

    if truncation_threshold is None:
        # The upper and lower sector bases are held in memory simultaneously (each is free
        # to fill the cap independently), so halve the safety fraction relative to a
        # single-basis driver to keep the same overall per-rank headroom.
        truncation_threshold = suggest_truncation_threshold(
            model.n_spin_orbitals, comm=MPI.COMM_WORLD, safety=DEFAULT_MEMORY_SAFETY / 2
        )
        log_memory_budget(
            truncation_threshold, model.n_spin_orbitals, comm=MPI.COMM_WORLD, verbose=verbose, label="fixed-peak dc"
        )

    if len(N0) != 1:
        raise ValueError(
            f"fixed_peak_dc supports a single impurity group, got N0 = {N0}. "
            "With multiple groups it is ambiguous which group gains/loses the electron."
        )
    h_op_i = ManyBodyOperator(h0_op) + ManyBodyOperator(u)
    impurity_orbitals, bath_states = _normalize_dc_orbitals(impurity_orbitals, bath_states)
    # Bath-only restriction (the reference "valence filled, conduction empty" occupation never
    # references N0), so the identical restriction list is valid for both sectors -- the same
    # restrictions find_ground_state_basis applies to its own N0 +- 1 occupation-scan trials.
    weighted_restrictions = build_weighted_restrictions(bath_states, excitation_budget)

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
        weighted_restrictions=weighted_restrictions,
        chain_restrict=chain_restrict,
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
        weighted_restrictions=weighted_restrictions,
        chain_restrict=chain_restrict,
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
    Calculate the double counting correction using a fixed (or self-consistent) impurity
    occupation criterion.

    With an explicit ``occupation``, choose the double counting so that the interacting thermal
    impurity occupation equals the requested value, :math:`\mathrm{Tr}\,\rho_{imp} = N_{target}`.

    With ``occupation=None``, choose it so that the interacting and non-interacting impurity
    occupations agree self-consistently, :math:`N(\mu) = N_0(\mu)`, where :math:`N_0(\mu)` is
    the Fermi-filled occupation of the non-interacting ``h0 - dc(\mu)`` at :math:`\mu_{chem} = 0`
    (:func:`_noninteracting_impurity_occupation`) -- the DFT impurity occupation the CSC DFT+DMFT
    self-consistency loop targets, re-evaluated at every trial shift (unlike a single value
    derived once from the guess). Both :math:`N(\mu)` and :math:`N_0(\mu)` are computed and
    logged at every trial regardless of which criterion drives the search.

    The double counting is parametrized as a uniform shift of the guess, ``dc(mu) = dc_guess +
    mu * identity``, coupling to the impurity occupation as :math:`-\mu \hat N_{imp}`,
    :math:`\partial N/\partial\mu \geq 0` and :math:`\partial N_0/\partial\mu \geq 0`. The
    explicit-target residual :math:`N(\mu) - N_{target}` is therefore monotone; the
    self-consistent residual :math:`N(\mu) - N_0(\mu)` is a difference of two independently
    monotone quantities and need not be monotone itself, so both are driven to zero by the
    non-monotonicity-safe search :func:`_solve_dc_shift`. At low temperature and weak
    hybridization the occupation approaches a staircase in ``mu``; if the requested (fractional)
    occupation falls on a plateau, the search converges to the closest step and a warning is
    printed.

    Note: the total electron number is conserved, so the impurity occupation
    changes through impurity-bath charge transfer; the reachable occupations
    are limited by the bath. The many-body basis is expanded once, with the
    guess double counting, honoring ``BasisOptions.excitation_budget``/``chain_restrict`` (a
    tight budget can itself make a requested occupation unreachable, since it bounds how far
    the bath can be depopulated/populated).

    Parameters other than the following match :func:`fixed_peak_dc` (``model``, ``basis``,
    ``solver``, ``comm``, ``verbosity``), except that ``basis.truncation_threshold=None`` here
    derives the cap for a single basis (no halving -- only one many-body basis is built).
    ``basis.nominal_occ`` is the nominal impurity occupation used to build the many-body basis;
    use the integer occupation closest to the requested one.

    Parameters
    ----------
    occupation : float or None
        Requested impurity occupation (may be fractional). ``None`` solves the self-consistent
        :math:`N(\mu) = N_0(\mu)` criterion instead (see above).
    occ_tol : float
        Convergence tolerance on the occupation (or on :math:`N - N_0` when self-consistent).
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
    excitation_budget = basis.excitation_budget
    chain_restrict = basis.chain_restrict
    slaterWeightMin = basis.slater_weight_min
    dense_cutoff = solver.dense_cutoff
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0

    if truncation_threshold is None:
        truncation_threshold = suggest_truncation_threshold(model.n_spin_orbitals, comm=MPI.COMM_WORLD)
        log_memory_budget(
            truncation_threshold,
            model.n_spin_orbitals,
            comm=MPI.COMM_WORLD,
            verbose=verbose,
            label="fixed-occupation dc",
        )

    h_op_i = ManyBodyOperator(h0_op) + ManyBodyOperator(u)
    impurity_orbitals, bath_states = _normalize_dc_orbitals(impurity_orbitals, bath_states)
    weighted_restrictions = build_weighted_restrictions(bath_states, excitation_budget)

    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]
    total_impurity_orbitals = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)

    self_consistent = occupation is None
    # Tolerance absorbs the roundoff of a target derived elsewhere the same way
    # _noninteracting_impurity_occupation is (a sum of Fermi occupations, each in [0, 1]).
    if not self_consistent and not -1e-9 <= occupation <= total_impurity_orbitals + 1e-9:
        raise ValueError(f"Requested impurity occupation {occupation} outside [0, {total_impurity_orbitals}].")

    # Local many-body basis / CIPSI solver (distinct from the BasisOptions/SolverOptions params).
    mb_basis, mb_solver = _prepare_dc_solver(
        h_op_i,
        impurity_orbitals,
        bath_states,
        N0,
        mixed_valence,
        truncation_threshold,
        spin_flip_dj,
        tau,
        verbose,
        weighted_restrictions=weighted_restrictions,
        chain_restrict=chain_restrict,
    )
    identity = np.identity(dc_guess.shape[0])

    # Expand the many-body basis once, with the guess double counting.
    h_guess = h_op_i - ManyBodyOperator(model.dc)
    mb_solver.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-5, slaterWeightMin=slaterWeightMin)

    energy_cut = -tau * np.log(1e-4)

    # Cache each evaluated (n, n0) pair so the final log reuses it instead of re-solving.
    occupation_at = {}
    noninteracting_at = {}

    def occupation_observable(mu):
        dc_op = _dc_operator(dc_guess + mu * identity)
        h_op = h_op_i - dc_op
        _, rho = _lowest_energy_and_thermal_rho(
            mb_basis, mb_solver, h_op, impurity_indices, energy_cut, dense_cutoff, slaterWeightMin
        )
        n = float(np.real(np.trace(rho)))
        n0 = _noninteracting_impurity_occupation(h0_op, dc_op.to_dict(), impurity_indices, model.n_spin_orbitals, tau)
        occupation_at[mu] = n
        noninteracting_at[mu] = n0
        if verbose and rank == 0:
            print(f"mu={mu:.6f} n={n:.6f} n0={n0:.6f}")
        # Self-consistent: the observable already IS the residual (target=0 below), since
        # N(mu) - N0(mu) is not of the form N(mu) - const. Explicit: raw N(mu), target=occupation.
        return n - n0 if self_consistent else n

    if self_consistent:
        target = 0.0
        unreachable = (
            "Could not find a self-consistent double counting (N = N0) with |mu| <= "
            f"{max_shift}: the closest residual N - N0 reached was {{value:.4f}} at mu = "
            "{mu:.3f}. The target may be unreachable with the available bath states."
        )
    else:
        target = occupation
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
    # hit, or a refined bracket point), so occupation_observable(mu) ran and cached both here.
    n = occupation_at[mu]
    n0 = noninteracting_at[mu]
    dc = dc_guess + mu * identity
    if verbose and rank == 0:
        label = "N == N0" if self_consistent else f"target = {occupation}"
        print(f"Fixed-occupation double counting ({label}, achieved N = {n:.4f}, N0 = {n0:.4f}, mu = {mu:.6f}):")
        matrix_print(dc_guess, label="DC guess:")
        matrix_print(dc, label="DC found:")
    return dc


def _iterate_dc_fixed_point(model, rho_to_dc, *, tau, mix, fixpoint_tol, max_iter, verbosity):
    r"""Damped fixed point for a static double-counting scheme, starting from ``model.dc``.

    The static schemes (:func:`fll_dc`, :func:`amf_dc`, :func:`sigma_inf_dc`) each need the
    non-interacting impurity density matrix *at self-consistency with their own output*: the
    double counting shifts the impurity level (entering ``h0 - dc``), which shifts the
    Fermi-filled occupation the scheme reads back in to produce the next ``dc``. Iterates

    .. math:: \rho_k = \rho_0(h_0 - dc_k),\qquad dc_{k+1} = (1 - \mathrm{mix})\, dc_k +
        \mathrm{mix} \cdot \mathrm{scheme}(\rho_k)

    (:func:`_noninteracting_impurity_rho` for :math:`\rho_0`) from ``dc_0 = `` ``model.dc`` (or
    zero), converging when the *unmixed* update ``scheme(rho_k) - dc_k`` is small -- this is the
    fixed-point residual regardless of ``mix``, so convergence is checked before damping is
    applied. A deterministic NumPy computation, identical on every rank; no MPI collective.

    Parameters
    ----------
    model : ImpurityModel
    rho_to_dc : callable
        Maps the current trial density matrix (``(n_imp, n_imp)`` complex) to the scheme's next
        double-counting matrix.
    tau : float
        Fundamental temperature passed to :func:`_noninteracting_impurity_rho`.
    mix : float
        Linear mixing fraction of the new update, in ``(0, 1]``.
    fixpoint_tol : float
        Convergence tolerance on ``max(abs(scheme(rho_k) - dc_k))``.
    max_iter : int
        Maximum number of iterations before giving up.
    verbosity : int
        Prints the per-iteration residual when ``> 0``.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
        The converged double-counting matrix.

    Raises
    ------
    RuntimeError
        If ``max_iter`` iterations do not bring the residual below ``fixpoint_tol``.
    """
    h0_op = model.h0
    impurity_indices = model.impurity_indices
    n_spin_orbitals = model.n_spin_orbitals
    n_imp = len(impurity_indices)
    dc = extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0]
    delta = np.inf
    for iteration in range(max_iter):
        rho = _noninteracting_impurity_rho(h0_op, _dc_operator(dc).to_dict(), impurity_indices, n_spin_orbitals, tau)
        dc_new = rho_to_dc(rho)
        delta = float(np.max(np.abs(dc_new - dc)))
        if verbosity > 0:
            print(f"Double-counting fixed point, iteration {iteration}: max|Δdc| = {delta:.3e}")
        if delta < fixpoint_tol:
            return dc_new
        dc = (1.0 - mix) * dc + mix * dc_new
    raise RuntimeError(
        f"Double-counting fixed point did not converge in {max_iter} iterations "
        f"(max|Δdc| = {delta:.3e} > {fixpoint_tol}); try a larger tau, a smaller mix, or pass "
        "the occupation/density matrix explicitly (n=/rho=) to skip the iteration."
    )


def fll_dc(model, *, tau=0.002, n=None, u=None, j=None, mix=0.5, fixpoint_tol=1e-8, max_iter=200, verbosity=0):
    r"""Fully Localized Limit double counting, ``dc = [U(N - 1/2) - (J/2)(N - 1)] I``.

    ``U``, ``J`` default to :func:`_model_uj`'s spherical average (needs ``model.u4``); either
    may be overridden explicitly (e.g. from tabulated values), independently of the other. ``N``
    defaults to the self-consistent non-interacting impurity occupation, found by
    :func:`_iterate_dc_fixed_point`; an explicit ``N`` skips the iteration (and, if both ``u``
    and ``j`` are also given, needs no ``model.u4`` at all).

    Parameters
    ----------
    model : ImpurityModel
    tau : float, optional
        Fundamental temperature for the fixed-point occupation (ignored if ``n`` is given).
    n : float, optional
        Impurity occupation ``N``. ``None`` solves for it self-consistently.
    u, j : float, optional
        Average Coulomb repulsion and exchange. ``None`` derives them from ``model.u4`` via
        :func:`_model_uj`.
    mix, fixpoint_tol, max_iter, verbosity : see :func:`_iterate_dc_fixed_point`.

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

    def dc_from_occupation(occupation):
        return (u * (occupation - 0.5) - 0.5 * j * (occupation - 1.0)) * identity

    if n is not None:
        return dc_from_occupation(n)
    return _iterate_dc_fixed_point(
        model,
        lambda rho: dc_from_occupation(float(np.real(np.trace(rho)))),
        tau=tau,
        mix=mix,
        fixpoint_tol=fixpoint_tol,
        max_iter=max_iter,
        verbosity=verbosity,
    )


def amf_dc(model, *, tau=0.002, n=None, mix=0.5, fixpoint_tol=1e-8, max_iter=200, verbosity=0):
    r"""Around Mean Field double counting, ``dc = Σ_static(u4, (N / n_imp) I)``.

    The static Hartree-Fock self-energy (:func:`impurityModel.ed.sigma.get_Sigma_static`)
    evaluated at a *uniform* trial density matrix, ``N`` spread evenly over every impurity
    spin-orbital -- the defining assumption of AMF, that the impurity has no orbital *or spin*
    polarization -- as opposed to :func:`sigma_inf_dc`, which uses the actual (possibly
    anisotropic) density matrix. For a spin-polarized ground state this is the paramagnetic
    (spin-blind) AMF potential, not a per-spin-channel one. ``N`` defaults to the self-consistent
    non-interacting impurity occupation (:func:`_iterate_dc_fixed_point`); an explicit ``N``
    skips the iteration.

    Parameters
    ----------
    model : ImpurityModel
    tau : float, optional
        Fundamental temperature for the fixed-point occupation (ignored if ``n`` is given).
    n : float, optional
        Impurity occupation ``N``. ``None`` solves for it self-consistently.
    mix, fixpoint_tol, max_iter, verbosity : see :func:`_iterate_dc_fixed_point`.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    n_imp = len(model.impurity_indices)
    u4_dense = _model_u4_dense(model)
    identity = np.identity(n_imp, dtype=complex)

    def dc_from_occupation(occupation):
        return get_Sigma_static(u4_dense, (occupation / n_imp) * identity)

    if n is not None:
        return dc_from_occupation(n)
    return _iterate_dc_fixed_point(
        model,
        lambda rho: dc_from_occupation(float(np.real(np.trace(rho)))),
        tau=tau,
        mix=mix,
        fixpoint_tol=fixpoint_tol,
        max_iter=max_iter,
        verbosity=verbosity,
    )


def sigma_inf_dc(model, *, tau=0.002, rho=None, mix=0.5, fixpoint_tol=1e-8, max_iter=200, verbosity=0):
    r"""K. Held's :math:`\Sigma(\infty)` double counting: the full static Hartree-Fock
    self-energy matrix, ``dc = Σ_static(u4, rho_imp)``.

    Unlike :func:`amf_dc`, uses the actual (possibly anisotropic) non-interacting impurity
    density matrix rather than a uniform trial -- the two agree exactly when that density matrix
    happens to be uniform (e.g. a single, orbitally-degenerate shell), and differ whenever the
    impurity levels split. ``rho`` defaults to the self-consistent non-interacting impurity
    density matrix (:func:`_iterate_dc_fixed_point`); an explicit ``rho`` skips the iteration.

    Parameters
    ----------
    model : ImpurityModel
    tau : float, optional
        Fundamental temperature for the fixed-point density matrix (ignored if ``rho`` is given).
    rho : numpy.ndarray, shape (n_imp, n_imp), optional
        Impurity density matrix. ``None`` solves for it self-consistently.
    mix, fixpoint_tol, max_iter, verbosity : see :func:`_iterate_dc_fixed_point`.

    Returns
    -------
    numpy.ndarray, shape (n_imp, n_imp)
    """
    u4_dense = _model_u4_dense(model)

    def dc_from_rho(trial_rho):
        return get_Sigma_static(u4_dense, trial_rho)

    if rho is not None:
        return dc_from_rho(np.asarray(rho, dtype=complex))
    return _iterate_dc_fixed_point(
        model,
        dc_from_rho,
        tau=tau,
        mix=mix,
        fixpoint_tol=fixpoint_tol,
        max_iter=max_iter,
        verbosity=verbosity,
    )
