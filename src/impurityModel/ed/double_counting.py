"""Double-counting determination for the self-energy workflow.

The impurity double-counting potential is fixed by one of two searches over a chemical
potential: :func:`fixed_peak_dc` pins a chosen spectral peak, :func:`fixed_occupation_dc`
pins the impurity occupation. Both repeatedly build the variational ground state and its
thermal density matrix (:func:`_lowest_energy_and_thermal_rho`) at trial potentials and
bisect. The self-energy extraction proper lives in :mod:`sigma`; the orchestration and CLI
in :mod:`selfenergy`, which re-exports ``fixed_peak_dc``/``fixed_occupation_dc`` so existing
callers are unchanged.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed import atomic_physics
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_transcription import build_density_matrices
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.operator_algebra import addOps
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
        for key, val in orbital_dict.items():
            val = list(val)
            if len(val) > 0 and not hasattr(val[0], "__iter__"):
                out[key] = [sorted(val)]
            else:
                out[key] = val
        return out

    valence_baths, conduction_baths = bath_states
    return as_blocked(impurity_orbitals), (as_blocked(valence_baths), as_blocked(conduction_baths))


def _dc_operator(dc):
    """Build the double-counting one-body operator, ``-dc[i, j] c^dagger_i c_j``."""
    return ManyBodyOperator(
        {
            ((i, "c"), (j, "a")): -dc[i, j] + 0j
            for i in range(dc.shape[0])
            for j in range(dc.shape[1])
            if abs(dc[i, j]) > 0
        }
    )


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


def fixed_peak_dc(
    h0_op,
    N0,
    mixed_valence,
    impurity_orbitals,
    bath_states,
    u4,
    peak_position,
    dc_guess,
    spin_flip_dj,
    tau,
    rank,
    verbose,
    dense_cutoff,
    slaterWeightMin,
    truncation_threshold,
):
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
    occupation as :math:`-\mu \hat N_{imp}`, so the peak position responds as
    :math:`d(E_{upper} - E_{lower})/d\mu = -(\langle N \rangle_{upper} -
    \langle N \rangle_{lower}) \approx -1`, and ``mu`` is found with
    well-conditioned Newton iterations.

    Note: the many-body bases are expanded once, with the guess double
    counting; the Newton iterations reuse them. Energies carry no fixed unit,
    they follow the inputs (e.g. Ry when called from RSPt); the convergence
    tolerance is ``max(tau, 1e-4)`` in those units.

    Parameters
    ----------
    h0_op : ManyBodyOperator or dict
        The non-interacting Hamiltonian.
    N0 : dict
        Nominal impurity occupation, ``{group: N}``. Only a single group is
        supported (with more groups, which one receives the extra electron
        would be ambiguous).
    mixed_valence : dict or None
        Mixed valence bounds, forwarded to the ``Basis``.
    impurity_orbitals : dict
        Impurity spin-orbital indices per group; flat lists or lists of blocks.
    bath_states : tuple of dict
        (valence, conduction) bath spin-orbital indices per group; flat lists
        or lists of blocks.
    u4 : ndarray
        Coulomb interaction U tensor (RSPt convention).
    peak_position : float
        Requested peak position; the sign selects addition/removal, see above.
        The magnitude is kept above ``4 * tau`` (thermal broadening).
    dc_guess : ndarray
        Initial guess for the double counting matrix.
    spin_flip_dj : bool
        Whether to generate spin-flipped determinants.
    tau : float
        Temperature.
    rank : int
        MPI process rank.
    verbose : bool
        Verbosity flag.
    dense_cutoff : int
        Cutoff dimension for the dense eigensolver.
    slaterWeightMin : float
        Minimum Slater determinant weight.
    truncation_threshold : float or None
        Global cap on the number of Slater determinants per basis; ``None`` derives it
        from available per-rank memory (see :mod:`impurityModel.ed.memory_estimate`).

    Returns
    -------
    dc : ndarray
        The double counting matrix, ``dc_guess + mu * identity``.

    Raises
    ------
    RuntimeError
        If the iteration does not converge, or the criterion is ill
        conditioned (upper and lower sectors have the same impurity
        occupation).
    """
    if len(N0) != 1:
        raise ValueError(
            f"fixed_peak_dc supports a single impurity group, got N0 = {N0}. "
            "With multiple groups it is ambiguous which group gains/loses the electron."
        )
    u = atomic_physics.getUop_from_rspt_u4(u4)
    h_op_i = ManyBodyOperator(addOps([h0_op, u]))
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
    h_guess = h_op_i + _dc_operator(dc_guess)
    solver_upper.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-5, slaterWeightMin=slaterWeightMin)
    solver_lower.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-5, slaterWeightMin=slaterWeightMin)

    energy_cut = -tau * np.log(1e-4)

    def peak_and_occupations(mu):
        h_op = h_op_i + _dc_operator(dc_guess + mu * identity)
        e_upper, rho_upper = _lowest_energy_and_thermal_rho(
            basis_upper, solver_upper, h_op, impurity_indices, energy_cut, dense_cutoff, slaterWeightMin
        )
        e_lower, rho_lower = _lowest_energy_and_thermal_rho(
            basis_lower, solver_lower, h_op, impurity_indices, energy_cut, dense_cutoff, slaterWeightMin
        )
        return e_upper - e_lower, np.real(np.trace(rho_upper)), np.real(np.trace(rho_lower))

    mu = 0.0
    tol = max(tau, 1e-4)
    max_iterations = 20
    converged = False
    error = np.inf
    for _ in range(max_iterations):
        peak, n_upper, n_lower = peak_and_occupations(mu)
        error = peak - peak_position
        if abs(error) < tol:
            converged = True
            break
        delta_n = n_upper - n_lower
        if abs(delta_n) < 0.1:
            raise RuntimeError(
                "The fixed-peak double counting criterion is ill conditioned: the upper and "
                f"lower sectors differ by only {delta_n:.3f} impurity electrons, so a uniform "
                "shift cannot move the peak."
            )
        # The shift couples as -mu * N_imp, so d(peak)/d(mu) = -(n_upper - n_lower).
        mu += error / delta_n
    if not converged:
        raise RuntimeError(
            f"The fixed-peak double counting did not converge in {max_iterations} iterations: "
            f"E_upper - E_lower - peak_position = {error:.6f} (tolerance {tol:.6f}), mu = {mu:.6f}."
        )

    dc = dc_guess + mu * identity
    if verbose and rank == 0:
        print(f"Fixed-peak double counting (peak position = {peak_position}, mu = {mu:.6f}):")
        matrix_print(dc_guess, label="DC guess:")
        matrix_print(dc, label="DC found:")

    return dc


def fixed_occupation_dc(
    h0_op,
    N0,
    mixed_valence,
    impurity_orbitals,
    bath_states,
    u4,
    occupation,
    dc_guess,
    spin_flip_dj,
    tau,
    rank,
    verbose,
    dense_cutoff,
    slaterWeightMin,
    truncation_threshold,
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
    is found by exponential bracketing followed by bisection. At low
    temperature and weak hybridization the occupation approaches a staircase
    in ``mu``; if the requested (fractional) occupation falls on a plateau,
    the search converges to the closest step and a warning is printed.

    Note: the total electron number is conserved, so the impurity occupation
    changes through impurity-bath charge transfer; the reachable occupations
    are limited by the bath. The many-body basis is expanded once, with the
    guess double counting.

    Parameters other than the following match :func:`fixed_peak_dc`.

    Parameters
    ----------
    N0 : dict
        Nominal impurity occupation used to build the many-body basis; use
        the integer occupation closest to the requested one.
    occupation : float
        Requested impurity occupation (may be fractional).
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
    u = atomic_physics.getUop_from_rspt_u4(u4)
    h_op_i = ManyBodyOperator(addOps([h0_op, u]))
    impurity_orbitals, bath_states = _normalize_dc_orbitals(impurity_orbitals, bath_states)

    total_impurity_orbitals = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)
    if not 0 <= occupation <= total_impurity_orbitals:
        raise ValueError(f"Requested impurity occupation {occupation} outside [0, {total_impurity_orbitals}].")

    basis, solver = _prepare_dc_solver(
        h_op_i, impurity_orbitals, bath_states, N0, mixed_valence, truncation_threshold, spin_flip_dj, tau, verbose
    )
    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]
    identity = np.identity(dc_guess.shape[0])

    # Expand the many-body basis once, with the guess double counting.
    h_guess = h_op_i + _dc_operator(dc_guess)
    solver.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-5, slaterWeightMin=slaterWeightMin)

    energy_cut = -tau * np.log(1e-4)

    def impurity_occupation(mu):
        h_op = h_op_i + _dc_operator(dc_guess + mu * identity)
        _, rho = _lowest_energy_and_thermal_rho(
            basis, solver, h_op, impurity_indices, energy_cut, dense_cutoff, slaterWeightMin
        )
        return np.real(np.trace(rho))

    def found(mu, n):
        dc = dc_guess + mu * identity
        if verbose and rank == 0:
            print(f"Fixed-occupation double counting (target = {occupation}, " f"achieved = {n:.4f}, mu = {mu:.6f}):")
            matrix_print(dc_guess, label="DC guess:")
            matrix_print(dc, label="DC found:")
        return dc

    mu = 0.0
    n = impurity_occupation(mu)
    if abs(n - occupation) <= occ_tol:
        return found(mu, n)

    # Bracket the target: <N_imp>(mu) is non-decreasing in mu.
    direction = 1.0 if n < occupation else -1.0
    mu_inner, n_inner = 0.0, n
    step = max(10 * tau, initial_step)
    mu_outer = direction * step
    n_outer = impurity_occupation(mu_outer)
    while (n_outer - occupation) * direction < 0:
        if abs(n_outer - occupation) <= occ_tol:
            return found(mu_outer, n_outer)
        mu_inner, n_inner = mu_outer, n_outer
        mu_outer *= 2
        if abs(mu_outer) > max_shift:
            raise RuntimeError(
                f"Could not bracket the requested impurity occupation {occupation} with "
                f"|mu| <= {max_shift}: the occupation reached {n_outer:.4f} at mu = {mu_inner:.3f}. "
                "The target may be unreachable with the available bath states."
            )
        n_outer = impurity_occupation(mu_outer)
    if abs(n_outer - occupation) <= occ_tol:
        return found(mu_outer, n_outer)

    if direction > 0:
        mu_low, n_low, mu_high, n_high = mu_inner, n_inner, mu_outer, n_outer
    else:
        mu_low, n_low, mu_high, n_high = mu_outer, n_outer, mu_inner, n_inner

    # Bisection; stop on the occupation tolerance or when the bracket has
    # collapsed onto an occupation step (plateau).
    width_tol = max(tau, 1e-4)
    while mu_high - mu_low > width_tol:
        mu_mid = 0.5 * (mu_low + mu_high)
        n_mid = impurity_occupation(mu_mid)
        if abs(n_mid - occupation) <= occ_tol:
            return found(mu_mid, n_mid)
        if n_mid < occupation:
            mu_low, n_low = mu_mid, n_mid
        else:
            mu_high, n_high = mu_mid, n_mid

    # Plateau: the occupation steps across the target. Return the side closest
    # to the target, loudly.
    if abs(n_low - occupation) <= abs(n_high - occupation):
        mu, n = mu_low, n_low
    else:
        mu, n = mu_high, n_high
    if rank == 0:
        print(
            f"WARNING: the requested impurity occupation {occupation} falls on an occupation "
            f"plateau; the closest achievable occupation is {n:.4f} (mu = {mu:.6f})."
        )
    return found(mu, n)
