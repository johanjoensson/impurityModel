import argparse
import itertools
from collections import OrderedDict

import numpy as np
from mpi4py import MPI

from impurityModel.ed import atomic_physics
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.hamiltonian_io import get_noninteracting_hamiltonian_operator
from impurityModel.ed.symmetries import (
    classify_bath_occupation,
    extract_tensors,
    group_orbitals_by_blocks,
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)
from impurityModel.ed.operator_algebra import addOps, c2i
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.greens_function import build_full_greens_function, get_Greens_function, save_Greens_function
from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.utils import matrix_print

EV_TO_RY = 1 / 13.605693122994

# Adaptive symmetry-adapted-basis rotation (calc_selfenergy): drop rotated operator terms below
# this magnitude (eV; removes rotation round-off fill), and rotate into the symmetry-adapted
# basis only if it keeps the operator term count within this factor of the input basis.
_ROTATION_TRIM_TOL = 1e-8
_MAX_ROTATION_FILL = 2.0


def _per_group_occupation(nominal_occ, impurity_orbitals, h=None):
    """Map ``nominal_occ`` onto the derived orbital-symmetry groups.

    Accepts a dict already keyed by the group indices (used as-is), or any other dict / a
    scalar interpreted as the *total* impurity occupation. When the one-body Hamiltonian ``h``
    is supplied, the total is distributed by **energetic filling** — the lowest on-site-energy
    impurity spin-orbitals (``h[o, o]``) are occupied first — so e.g. a cubic d-shell fills the
    lower ``t2g`` manifold before ``eg`` (giving ``t2g=6``, ``eg=2`` for ``d8``) and the split
    is spin-symmetric. Without ``h`` it falls back to a size-proportional split (remainder to
    the largest groups). The prescan refines the per-group split, so this only needs to be a
    sensible starting point.
    """
    keys = list(impurity_orbitals)
    if isinstance(nominal_occ, dict) and set(nominal_occ) == set(keys):
        return {k: int(nominal_occ[k]) for k in keys}
    total = int(sum(nominal_occ.values()) if isinstance(nominal_occ, dict) else nominal_occ)

    if h is not None:
        # Energetic filling: occupy the lowest on-site-energy impurity spin-orbitals first and
        # count how many land in each group. Ties broken by orbital index for determinism.
        orb_to_group = {orb: k for k in keys for block in impurity_orbitals[k] for orb in block}
        ordered = sorted(orb_to_group, key=lambda o: (np.real(h[o, o]), o))
        alloc = {k: 0 for k in keys}
        for orb in ordered[: max(0, min(total, len(ordered)))]:
            alloc[orb_to_group[orb]] += 1
        return alloc

    sizes = {k: sum(len(block) for block in impurity_orbitals[k]) for k in keys}
    tot_size = sum(sizes.values()) or 1
    alloc = {k: int(total * sizes[k] // tot_size) for k in keys}
    remainder = total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: sizes[k], reverse=True):
        if remainder <= 0:
            break
        alloc[k] += 1
        remainder -= 1
    return alloc


def _per_group_scalar(value, impurity_orbitals, default=0):
    """Map a per-group scalar setting (e.g. mixed_valence) onto the derived group keys."""
    keys = list(impurity_orbitals)
    if isinstance(value, dict) and set(value) == set(keys):
        return dict(value)
    return {k: default for k in keys}


class UnphysicalGreensFunctionError(Exception):
    """
    Excpetion signalling an unphysical Greens function, i.e. the imaginary part is positive for some frequencies.
    """


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
        solver="trlm",
    )
    rhos = basis.build_density_matrices(
        psis,
        orbital_indices_left=impurity_indices,
        orbital_indices_right=impurity_indices,
    )
    rho = thermal_average_scale_indep(es, rhos, basis.tau)
    return es[0], rho


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
    truncation_threshold : float
        Basis truncation threshold.

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
    solver_upper.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-3, slaterWeightMin=slaterWeightMin)
    solver_lower.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-3, slaterWeightMin=slaterWeightMin)

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

    Parameters
    ----------
    Same as :func:`fixed_peak_dc`, except:
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
    solver.expand(h_guess, dense_cutoff=dense_cutoff, de2_min=1e-3, slaterWeightMin=slaterWeightMin)

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


def calc_selfenergy(
    h0,
    u4,
    iw,
    w,
    delta,
    nominal_occ,
    mixed_valence,
    impurity_orbitals,
    tau,
    verbosity,
    rot_to_spherical,
    cluster_label,
    reort,
    dense_cutoff,
    spin_flip_dj,
    comm,
    chain_restrict,
    occ_cutoff,
    truncation_threshold,
    slaterWeightMin,
    dN,
    sparse_green,
):
    """Calculate the self energy of the impurity.

    Parameters
    ----------
    h0 : dict or ManyBodyOperator
        The non-interacting Hamiltonian.
    u4 : np.ndarray
        The Coulomb interaction matrix.
    iw : np.ndarray or None
        Matsubara frequency mesh.
    w : np.ndarray or None
        Real frequency mesh.
    delta : float
        Smearing parameter for real frequencies.
    nominal_occ : dict
        Nominal occupation.
    mixed_valence : bool
        Whether to consider mixed valence.
    impurity_orbitals : dict[int, list[int]]
        Flat impurity spin-orbital index lists per group; re-grouped into conserved-charge blocks
        internally by :func:`symmetries.group_orbitals_by_charges`. The bath orbitals (everything
        else in ``h0``) and their valence/conduction (occupied/empty) split are derived from the
        Hamiltonian via :func:`symmetries.classify_bath_occupation`.
    tau : float
        Temperature parameter.
    verbosity : int
        Verbosity level.
    rot_to_spherical : np.ndarray
        Rotation matrix to spherical harmonics.
    cluster_label : str
        Label for the cluster.
    reort : float or None
        Reorthogonalization parameter.
    dense_cutoff : int
        Cutoff for dense matrix representation.
    spin_flip_dj : bool
        Whether to include spin-flip terms.
    comm : MPI.Comm or None
        MPI communicator.
    chain_restrict : bool
        Whether to restrict to chain geometry.
    occ_cutoff : float
        Cutoff for occupation numbers.
    truncation_threshold : float
        Threshold for truncating the basis.
    slaterWeightMin : float
        Minimum weight for Slater determinants.
    dN : int or None
        Particle number constraint.
    sparse_green : bool
        Whether to use sparse representation for Green's function.

    Returns
    -------
    dict
        Dictionary containing self-energy, Green's function, thermal density matrix, and ground state info.
    """
    # MPI variables
    rank = comm.rank if comm is not None else 0
    # Confine this section's logging to the master rank: silencing the others keeps the
    # output readable under MPI (the verbose flags below are only forwarded to printing,
    # never to collective operations).
    if rank != 0:
        verbosity = 0

    def log(msg="", *, level=1, **kwargs):
        if verbosity >= level:
            print(msg, **kwargs)

    def banner(title, *, level=1):
        if verbosity >= level:
            print("\n" + "=" * 80)
            print(f"  {title}")
            print("=" * 80, flush=verbosity >= 2)

    # construct local, interacting, hamiltonian (in the caller's input/correlated basis B)
    u = atomic_physics.getUop_from_rspt_u4(u4)
    h_input = ManyBodyOperator(h0) + ManyBodyOperator(u)

    # Flatten the impurity orbital dict (dict[int, list[int]]) into a plain spin-orbital index
    # list; the total orbital count is inferred from the Hamiltonian (impurity + bath). The bath
    # orbitals and their valence/conduction split are derived below, not passed in.
    impurity_indices = sorted(o for orbs in impurity_orbitals.values() for o in orbs)
    h_input_matrix = extract_tensors(h_input, two_body=False)[0]
    n_spin_orbitals = h_input_matrix.shape[0]

    # Adaptive symmetry-adapted basis: diagonalising the impurity one-body block collapses the
    # Green's-function block structure to its finest form (e.g. 1x1 eg/t2g blocks) BUT can
    # express the Coulomb interaction more densely. h0 and u4 are in the caller's "correlated"
    # input basis (NOT assumed spherical); the fill test below is measured *relative to that
    # input basis*, so we rotate only when it does not densify the operator (fill <= threshold)
    # and keep the input basis otherwise (e.g. a j,m_j eigenbasis under spin-orbit coupling
    # densifies the Coulomb tensor). Every output is rotated back to the input basis B before
    # returning; nothing here presumes a spherical-harmonic input.
    rotation_full, u_imp = impurity_symmetry_rotation(
        h_input, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_input_matrix
    )
    h_rotated = rotate_hamiltonian(h_input, rotation_full, tol=_ROTATION_TRIM_TOL)
    n_terms_input = sum(1 for v in h_input.values() if abs(v) > _ROTATION_TRIM_TOL)
    fill_ratio = len(h_rotated) / max(n_terms_input, 1)

    rotate = fill_ratio <= _MAX_ROTATION_FILL
    if rotate:
        h = h_rotated
        h0_solve = rotate_hamiltonian(ManyBodyOperator(h0), rotation_full, tol=_ROTATION_TRIM_TOL).to_dict()
        # Observable rotation for the solve (spherical -> S): compose the caller's input rotation
        # R_in (spherical -> B) with W^dag (B -> S). On the impurity block, R = u_imp^dag @ R_in.
        rot_to_spherical = u_imp.conj().T @ np.asarray(rot_to_spherical, dtype=complex)
    else:
        # Stay in the input basis; make the output rotation below a no-op.
        h = h_input
        h0_solve = h0
        rotation_full = np.eye(n_spin_orbitals, dtype=complex)
        u_imp = np.eye(len(impurity_indices), dtype=complex)

    # One-body matrix of the solver-basis Hamiltonian, extracted once and shared by the
    # classification/grouping helpers below (each would otherwise re-walk the full operator
    # and allocate its own dense n_orb x n_orb copy).
    h_matrix = extract_tensors(h, n_orb=n_spin_orbitals, two_body=False)[0] if rotate else h_input_matrix

    # Derive the bath orbitals (complement of the impurity set) and their initial occupation:
    # baths below the Fermi level (h[o, o] < 0) are valence (initially occupied), the rest are
    # conduction (initially empty). The bath one-body diagonal is unchanged by the impurity-only
    # rotation, so this is consistent whether measured in the input or solver basis.
    valence_flat, conduction_flat = classify_bath_occupation(
        h, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_matrix
    )

    # GF block structure from the hybridization-dressed impurity matrix (h[imp,imp] + V^dag V),
    # in whichever basis we solve in (fixes bath-mediated coupling; 1x1 blocks when rotated).
    # Derived from h *after* any rotation, so the blocks label the sectors of the solver basis.
    block_structure = impurity_block_structure(h, impurity_indices, h0_matrix=h_matrix)

    # Group the flat orbital lists into orbital-symmetry manifolds (the inequivalent blocks and
    # their spin-degenerate partners, e.g. eg / t2g) **in the solver basis** h. Grouping by the
    # block structure keeps both spins of a manifold in one group, so the many-body basis spans
    # all S_z sectors (spin multiplets stay degenerate); the impurity occupation window is tied
    # across groups by the restriction machinery, not pinned per group.
    impurity_orbitals, bath_states = group_orbitals_by_blocks(
        h, impurity_indices, valence_flat, conduction_flat, block_structure, n_orb=n_spin_orbitals, h0_matrix=h_matrix
    )
    nominal_occ = _per_group_occupation(nominal_occ, impurity_orbitals, h_matrix)
    mixed_valence = _per_group_scalar(mixed_valence, impurity_orbitals, default=0)

    valence_baths, conduction_baths = bath_states
    total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
    sum_bath_states = {
        i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
        for i in valence_baths
    }

    if verbosity > 0:
        basis_note = f"symmetry-adapted (fill {fill_ratio:.1f}x)" if rotate else f"input basis (fill {fill_ratio:.1f}x)"
        print(f"Block structure: {len(block_structure.blocks)} blocks, solving in {basis_note}")
    basis_information = {
        "impurity_orbitals": impurity_orbitals,
        "bath_states": bath_states,
        "N0": nominal_occ,
        "mixed_valence": mixed_valence,
        "tau": tau,
        "chain_restrict": chain_restrict,
        "dense_cutoff": dense_cutoff,
        "spin_flip_dj": spin_flip_dj,
        "rank": rank,
        "comm": comm,
        "truncation_threshold": truncation_threshold,
    }
    # Compute the thermal ground state and the interacting Green's function, with a single
    # auto-retry: the diagnostics report (gf_diagnostics) can detect that the thermal
    # ensemble was truncated (the highest retained state still carries Boltzmann weight); if
    # so we re-run the eigensolver with more requested states (num_wanted) once.
    num_wanted = 10
    max_retries = 2
    for _attempt in range(max_retries + 1):
        psis, es, ground_state_basis, thermal_rho, gs_info = calc_gs(
            h,
            basis_information,
            block_structure,
            rot_to_spherical,
            verbosity >= 1,
            slaterWeightMin=slaterWeightMin,
            num_wanted=num_wanted,
        )
        restrictions = ground_state_basis.restrictions

        if restrictions is not None:
            log("Restrictions on ground-state occupation:", level=2)
            for indices, limits in restrictions.items():
                log(f"  {sorted(indices)} : {limits}", level=2)

        banner("Interacting Green's function")
        log(f"Considering {len(es)} eigenstate(s) for the spectra.")
        log("Calculating interacting Green's function ...", flush=verbosity >= 2)

        gs_matsubara, gs_realaxis, gf_report = get_Greens_function(
            matsubara_mesh=iw,
            omega_mesh=w,
            psis=psis,
            es=es,
            tau=tau,
            basis=ground_state_basis,
            hOp=h,
            delta=delta,
            blocks=[block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
            verbose=verbosity >= 1,
            verbose_extra=verbosity >= 2,
            reort=reort,
            dN=dN,
            occ_cutoff=occ_cutoff,
            slaterWeightMin=slaterWeightMin,
            sparse=sparse_green,
            num_wanted=num_wanted,
        )

        # Root rank renders the diagnostics report and decides whether to retry; the decision
        # is broadcast so every rank re-enters calc_gs collectively (or all break).
        retry = False
        if rank == 0 and gf_report is not None:
            log(gf_report.render())
            retry = gf_report.needs_more_states and _attempt < max_retries
        if comm is not None:
            retry = comm.bcast(retry, root=0)
        if not retry:
            break
        num_wanted *= 2
        log(f"\nThermal ensemble appears truncated; retrying with num_wanted = {num_wanted}.\n", flush=True)
    if gs_matsubara is not None:
        try:
            for gs in gs_matsubara:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            raise UnphysicalGreensFunctionError("Matsubara interacting Greens function:\n" + str(err)) from None
    if gs_realaxis is not None:
        try:
            for gs in gs_realaxis:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            raise UnphysicalGreensFunctionError("Real frequency interacting Greens function:\n" + str(err)) from None

    banner("Self-energy")
    log("Calculating self-energy ...")
    if gs_realaxis is not None:
        sigma_real = get_sigma(
            omega_mesh=w,
            impurity_orbitals=total_impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gs_realaxis,
            h0op=h0_solve,
            delta=delta,
            clustername=cluster_label,
            blocks=[block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
        )
        try:
            for sig in sigma_real:
                check_greens_function(sig)
        except UnphysicalGreensFunctionError as err:
            for i, sig in enumerate(sigma_real):
                save_Greens_function(sig, w, f"sig+dc-{i}", cluster_label)
            raise UnphysicalGreensFunctionError("Real frequency self-energy:\n" + str(err)) from None
    else:
        sigma_real = None
    if gs_matsubara is not None:
        sigma = get_sigma(
            omega_mesh=iw,
            impurity_orbitals=total_impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gs_matsubara,
            h0op=h0_solve,
            delta=0,
            clustername=cluster_label,
            blocks=[block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
        )
        try:
            for sig in sigma:
                check_greens_function(sig)
        except UnphysicalGreensFunctionError as err:
            for i, sig in enumerate(sigma):
                save_Greens_function(sig, iw, f"sig+dc-{i}", cluster_label)
            raise UnphysicalGreensFunctionError("Matsubara self-energy:\n" + str(err)) from None
    else:
        sigma = None
    log("Calculating static self-energy ...")
    impurity_indices = [
        orb
        for impurity_blocks in ground_state_basis.impurity_orbitals.values()
        for block in impurity_blocks
        for orb in block
    ]
    impurity_ix = np.ix_(impurity_indices, impurity_indices)

    # Rotate every result from the solver basis S back to the caller's input basis B
    # (O_B = W O_S W^dag; impurity block u_imp). When the adaptive test kept the input basis,
    # W and u_imp are identity and these are no-ops. The density matrix is full-space (rotate
    # with W); the self-energies / Green's functions are impurity-only (rotate with u_imp).
    thermal_rho = rotation_full @ thermal_rho @ rotation_full.conj().T

    def _to_input_basis(block_list):
        """Reassemble per-inequivalent-block matrices (basis S) and rotate to input basis B."""
        if block_list is None:
            return None
        full_s = build_full_greens_function(block_list, block_structure)
        if full_s.ndim == 3:  # (n_omega, n_imp, n_imp)
            return np.einsum("ij,wjk,lk->wil", u_imp, full_s, u_imp.conj())
        return u_imp @ full_s @ u_imp.conj().T

    sigma_full = _to_input_basis(sigma)
    sigma_real_full = _to_input_basis(sigma_real)
    gs_matsubara_full = _to_input_basis(gs_matsubara)
    gs_realaxis_full = _to_input_basis(gs_realaxis)

    # Static (Hartree-Fock) self-energy from the input-basis density matrix and u4 (input basis).
    sigma_static = get_Sigma_static(u4, thermal_rho[impurity_ix])

    return {
        "sigma": sigma_full,
        "sigma_real": sigma_real_full,
        "sigma_static": sigma_static,
        "gs_matsubara": gs_matsubara_full,
        "gs_realaxis": gs_realaxis_full,
        "thermal_rho": thermal_rho,
        "rhos": gs_info["rhos"],
        "block_structure": block_structure,
    }


def check_greens_function(G):
    """Verify that the Green's function makes physical sense.

    Raises an exception if the diagonal elements of the imaginary part are positive.

    Parameters
    ----------
    G : np.ndarray
        The Green's function matrix.

    Raises
    ------
    UnphysicalGreensFunctionError
        If the diagonal term has a positive imaginary part.
    """
    if np.any(np.diagonal(G, axis1=1, axis2=2).imag > 0):
        raise UnphysicalGreensFunctionError("Diagonal term has positive imaginary part.")


def get_hcorr_v_hbath(h0op, impurity_orbitals, sum_bath_states):
    """Extract the correlation Hamiltonian, hybridization, and bath Hamiltonian.

    The matrix form of h0op can be written as:
      [  hcorr  V^+    ]
      [  V      hbath  ]

    Parameters
    ----------
    h0op : dict or ManyBodyOperator
        The non-interacting Hamiltonian operator.
    impurity_orbitals : dict
        Dictionary of impurity orbitals.
    sum_bath_states : dict
        Dictionary of total bath states.

    Returns
    -------
    hcorr : np.ndarray
        Hamiltonian for the correlated impurity orbitals.
    v : np.ndarray
        Hopping from impurity to bath orbitals.
    v_dagger : np.ndarray
        Hopping from bath to impurity orbitals.
    h_bath : np.ndarray
        Hamiltonian for the non-interacting bath orbitals.
    """

    num_spin_orbitals = sum(impurity_orbitals[i] + sum_bath_states[i] for i in impurity_orbitals)
    n_corr = sum(ni for ni in impurity_orbitals.values())
    h0Matrix = np.zeros((num_spin_orbitals, num_spin_orbitals), dtype=complex)
    for ((i, opi), (j, opj)), val in h0op.items():
        if opi == "c" and opj == "a":
            h0Matrix[i, j] = val
        elif opj == "c" and opi == "a":
            if i == j:
                h0Matrix[i, j] = 1 - val
            else:
                h0Matrix[i, j] = -val
    hcorr = h0Matrix[0:n_corr, 0:n_corr]
    v_dagger = h0Matrix[0:n_corr, n_corr:]
    v = h0Matrix[n_corr:, 0:n_corr]
    h_bath = h0Matrix[n_corr:, n_corr:]
    return hcorr, v, v_dagger, h_bath


def hyb(ws, v, hbath, delta):
    """Calculate hybridization function from hopping parameters and bath energies.

    Δ(w) = V^dag [(w + i*delta)I - hbath]^-1 V

    Parameters
    ----------
    ws : np.ndarray
        Frequency mesh.
    v : np.ndarray
        Hopping matrix V.
    hbath : np.ndarray
        Bath Hamiltonian matrix.
    delta : float
        Smearing parameter.

    Returns
    -------
    np.ndarray
        The hybridization function.
    """
    return np.conj(v.T) @ np.linalg.solve(
        (ws + 1j * delta)[:, None, None] * np.identity(hbath.shape[0], dtype=complex)[None, :, :] - hbath[None, :, :],
        v[None, :, :],
    )


def get_sigma(
    omega_mesh,
    impurity_orbitals,
    nBaths,
    gs,
    h0op,
    delta,
    blocks,
    clustername="",
):
    """Calculate self-energy from interacting Greens function and local hamiltonian.

    Parameters
    ----------
    omega_mesh : np.ndarray
        Frequency mesh.
    impurity_orbitals : dict
        Dictionary of impurity orbitals.
    nBaths : dict
        Dictionary of total bath states.
    gs : list of np.ndarray
        List of block Green's function matrices.
    h0op : dict or ManyBodyOperator
        The non-interacting Hamiltonian operator.
    delta : float
        Smearing parameter.
    blocks : list of list of int
        List of blocks.
    clustername : str, optional
        Label for the cluster.

    Returns
    -------
    list of np.ndarray
        The self-energy matrices for each block.
    """
    hcorr, v_full, _, h_bath = get_hcorr_v_hbath(h0op, impurity_orbitals, nBaths)

    res = []
    for block, g in zip(blocks, gs):
        block_ix = np.ix_(block, block)
        wIs = (omega_mesh + 1j * delta)[:, np.newaxis, np.newaxis] * np.eye(len(block))[np.newaxis, :, :]
        g0_inv = wIs - hcorr[block_ix] - hyb(omega_mesh, v_full[:, block], h_bath, delta)
        res.append(g0_inv - np.linalg.inv(g))

    return res


def get_Sigma_static(U4, rho):
    """Calculate the static (Hartree-Fock) self-energy.

    Parameters
    ----------
    U4 : np.ndarray
        Coulomb interaction tensor.
    rho : np.ndarray
        Density matrix.

    Returns
    -------
    np.ndarray
        The static self-energy.
    """
    sigma_static = np.zeros_like(rho)
    for i, j in itertools.product(range(rho.shape[0]), range(rho.shape[1])):
        sigma_static += (U4[j, :, :, i] - U4[j, :, i, :]) * rho[i, j]

    return sigma_static


def get_selfenergy(
    clustername,
    h0_filename,
    ls,
    nBaths,
    nValBaths,
    n0imps,
    dnTols,
    dnValBaths,
    dnConBaths,
    Fdd,
    xi,
    chargeTransferCorrection,
    hField,
    nPsiMax,
    nPrintSlaterWeights,
    tau,
    energy_cut,
    delta,
    verbose,
):
    """Calculate the self energy starting from a large number of arguments.

    Parameters
    ----------
    clustername : str
        Label for the cluster.
    h0_filename : str
        Filename of the non-interacting Hamiltonian.
    ls : int
        Angular momentum of correlated orbitals.
    nBaths : int
        Total number of bath states.
    nValBaths : int
        Number of valence bath states.
    n0imps : int
        Nominal impurity occupation.
    dnTols : int
        Max deviation from nominal impurity occupation.
    dnValBaths : int
        Max number of electrons to leave valence bath orbitals.
    dnConBaths : int
        Max number of electrons to enter conduction bath orbitals.
    Fdd : list of float
        Slater-Condon parameters.
    xi : float
        Spin-orbit coupling value.
    chargeTransferCorrection : float
        Double counting parameter.
    hField : tuple of float
        Magnetic field vector (hx, hy, hz).
    nPsiMax : int
        Maximum number of eigenstates to consider.
    nPrintSlaterWeights : int
        Printing parameter for Slater weights.
    tau : float
        Fundamental temperature.
    energy_cut : float
        Energy cutoff for eigenstates.
    delta : float
        Smearing parameter.
    verbose : bool
        Verbosity flag.
    """
    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # omega_mesh = np.linspace(-25, 25, 2000)
    omega_mesh = np.linspace(-1.83, 1.83, 2000)
    # omega_mesh = 1j*np.pi*tau*np.arange(start = 1, step = 2, stop = 2*375)

    # if rank == 0:
    #     t0 = time.perf_counter()
    # -- System information --

    sum_baths = OrderedDict({ls: nBaths})

    # -- Basis occupation information --
    nominal_occ = {ls: n0imps}

    # Construct u4 and rot_to_spherical, mixed_valence, etc.
    n_imp_spin_orbitals = 2 * (2 * ls + 1)
    u4 = np.zeros((n_imp_spin_orbitals, n_imp_spin_orbitals, n_imp_spin_orbitals, n_imp_spin_orbitals), dtype=complex)
    uOp = atomic_physics.getUop(l1=ls, l2=ls, l3=ls, l4=ls, R=Fdd)
    nBaths_for_c2i = OrderedDict({ls: 0})
    for process, val in uOp.items():
        i = c2i(nBaths_for_c2i, process[0][0])
        j = c2i(nBaths_for_c2i, process[1][0])
        k = c2i(nBaths_for_c2i, process[2][0])
        l = c2i(nBaths_for_c2i, process[3][0])
        u4[i, j, k, l] = 2.0 * val

    # Flat impurity spin-orbital index list (dict[int, list[int]]); calc_selfenergy re-groups the
    # orbitals into per-conserved-charge blocks, derives the bath orbitals + their valence/
    # conduction split from the Hamiltonian, and derives the block structure internally.
    impurity_orbitals = {ls: list(range(n_imp_spin_orbitals))}
    mixed_valence = {ls: 0}

    rot_to_spherical = np.eye(n_imp_spin_orbitals, dtype=complex)

    # Hamiltonian
    if rank == 0 and verbose:
        print("Constructing the Hamiltonian operator ...")
    hOp = get_noninteracting_hamiltonian_operator(
        sum_baths,
        [0, xi],
        hField,
        h0_filename,
        rank,
        verbose,
    )
    # Convert spin-orbital and bath state indices to a single index notation.
    hOp_new = {}
    for process, value in hOp.items():
        new_process = []
        for spinOrb, action in process:
            try:
                new_process.append((c2i(sum_baths, spinOrb), action))
            except Exception as e:
                print(f"FAILED on spinOrb: {spinOrb} in process {process}", flush=True)
                raise e
        hOp_new[tuple(new_process)] = value
    hOp = hOp_new

    # calc_selfenergy returns a result dict, not a tuple. Keys: "sigma"/"sigma_real" and
    # "gs_matsubara"/"gs_realaxis" (full (n_omega, n_imp, n_imp) matrices rotated back to the
    # caller's input basis, or None), "sigma_static", "thermal_rho", "rhos", "block_structure".
    result = calc_selfenergy(
        h0=hOp,
        u4=u4,
        iw=None,
        w=omega_mesh,
        delta=delta,
        nominal_occ=nominal_occ,
        mixed_valence=mixed_valence,
        impurity_orbitals=impurity_orbitals,
        tau=tau,
        verbosity=2 if verbose else 0,
        rot_to_spherical=rot_to_spherical,
        cluster_label=clustername,
        reort=None,
        dense_cutoff=500,
        spin_flip_dj=False,
        comm=comm,
        chain_restrict=False,
        occ_cutoff=1e-12,
        truncation_threshold=1000,
        slaterWeightMin=1e-12,
        dN=None,
        sparse_green=True,
    )

    if rank == 0 and verbose:
        print(f"Self-energy computed for cluster '{clustername}'.")
    # To persist the results, save the relevant entries of `result`, e.g.:
    #     if rank == 0:
    #         np.savetxt(f"real-sig_static-{clustername}.dat", np.real(result["sigma_static"]))
    #         np.savetxt(f"imag-sig_static-{clustername}.dat", np.imag(result["sigma_static"]))
    #         save_Greens_function(
    #             gs=result["sigma_real"], omega_mesh=omega_mesh, label=f"Sigma-{clustername}", e_scale=1
    #         )
    return result


if __name__ == "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser(description="Calculate selfenergy")
    parser.add_argument(
        "h0_filename",
        type=str,
        help="Filename of non-interacting Hamiltonian.",
    )
    parser.add_argument(
        "--clustername",
        type=str,
        default="cluster",
        help="Id of cluster, used for generating the filename in which to store the calculated self-energy.",
    )
    parser.add_argument(
        "--ls",
        type=int,
        default=2,
        help="Angular momenta of correlated orbitals.",
    )
    parser.add_argument(
        "--nBaths",
        type=int,
        default=10,
        help="Total number of bath states, for the correlated orbitals.",
    )
    parser.add_argument(
        "--nValBaths",
        type=int,
        default=10,
        help="Number of valence bath states for the correlated orbitals.",
    )
    parser.add_argument(
        "--n0imps",
        type=int,
        default=8,
        help="Nominal impurity occupation.",
    )
    parser.add_argument(
        "--dnTols",
        type=int,
        default=2,
        help=("Max devation from nominal impurity occupation."),
    )
    parser.add_argument(
        "--dnValBaths",
        type=int,
        default=2,
        help=("Max number of electrons to leave valence bath orbitals."),
    )
    parser.add_argument(
        "--dnConBaths",
        type=int,
        default=0,
        help=("Max number of electrons to enter conduction bath orbitals."),
    )
    parser.add_argument(
        "--Fdd",
        type=float,
        nargs="+",
        default=[7.5, 0, 9.9, 0, 6.6],
        help="Slater-Condon parameters Fdd. d-orbitals are assumed.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=0,
        help="SOC value for valence orbitals. Assumed to be d-orbitals",
    )
    parser.add_argument(
        "--chargeTransferCorrection",
        type=float,
        default=None,
        help="Double counting parameter.",
    )
    parser.add_argument(
        "--hField",
        type=float,
        nargs="+",
        default=[0, 0, 0.0001],
        help="Magnetic field. (h_x, h_y, h_z)",
    )
    parser.add_argument(
        "--nPsiMax",
        type=int,
        default=5,
        help="Maximum number of eigenstates to consider.",
    )
    parser.add_argument("--nPrintSlaterWeights", type=int, default=3, help="Printing parameter.")
    parser.add_argument("--tau", type=float, default=0.002, help="Fundamental temperature (kb*T).")
    parser.add_argument(
        "--energy_cut",
        type=float,
        default=10,
        help="How many k_B*T above lowest eigenenergy to consider.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help=("Smearing, half width half maximum (HWHM). Due to short core-hole lifetime."),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=("Set verbose output (very loud...)"),
    )
    args = parser.parse_args()

    # Sanity checks
    assert args.nBaths >= args.nValBaths
    assert args.n0imps >= 0
    assert args.n0imps <= 2 * (2 * args.ls + 1)
    assert len(args.Fdd) == 5
    assert len(args.hField) == 3

    get_selfenergy(
        clustername=args.clustername,
        h0_filename=args.h0_filename,
        ls=(args.ls),
        nBaths=(args.nBaths),
        nValBaths=(args.nValBaths),
        n0imps=(args.n0imps),
        dnTols=(args.dnTols),
        dnValBaths=(args.dnValBaths),
        dnConBaths=(args.dnConBaths),
        Fdd=(args.Fdd),
        xi=args.xi,
        chargeTransferCorrection=args.chargeTransferCorrection,
        hField=tuple(args.hField),
        nPsiMax=args.nPsiMax,
        nPrintSlaterWeights=args.nPrintSlaterWeights,
        tau=args.tau,
        energy_cut=args.energy_cut,
        delta=args.delta,
        verbose=args.verbose,
    )
