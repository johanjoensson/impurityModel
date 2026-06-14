import numpy as np
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, inner
from impurityModel.ed.finite import thermal_average_scale_indep as thermal_average


def calc_density_matrix(psi: ManyBodyState, orbital_indices: list[int]):
    r"""
    Compute the single-particle density matrix for a many-body state.

    rho[i, j] = <psi| c_{orb_j}^dagger c_{orb_i} |psi>

    Uses the identity rho[i, j] = <phi_j | phi_i> where |phi_k> = c_{orb_k} |psi>,
    reducing the number of operator applications from n_orb^2 to n_orb.
    The Hermitian symmetry rho[j, i] = conj(rho[i, j]) further halves the
    number of inner products required.

    Parameters
    ----------
    psi : ManyBodyState
        The many-body state to compute the density matrix for.
    orbital_indices : list[int]
        Orbital indices to include in the density matrix.

    Returns
    -------
    rho : np.ndarray, shape (n_orb, n_orb), dtype=complex
        The single-particle density matrix.
    """
    n_orb = len(orbital_indices)
    rho = np.zeros((n_orb, n_orb), dtype=complex)

    # Precompute |phi_k> = c_{orb_k} |psi> for each orbital k.
    annihilated = []
    for orb in orbital_indices:
        op = ManyBodyOperator({((orb, "a"),): 1.0})
        annihilated.append(op(psi, cutoff=0))

    # rho[i, j] = <psi| c_j^dag c_i |psi> = <phi_j | phi_i>
    # Exploit Hermitian symmetry: rho[j, i] = conj(rho[i, j])
    for i in range(n_orb):
        # Diagonal element (always real for a valid density matrix)
        amp = inner(annihilated[i], annihilated[i])
        if np.abs(amp) > 0:
            rho[i, i] = amp
        # Off-diagonal: compute upper triangle and mirror
        for j in range(i + 1, n_orb):
            amp = inner(annihilated[j], annihilated[i])
            if np.abs(amp) > 0:
                rho[i, j] = amp
                rho[j, i] = amp.conjugate()

    return rho


def calc_density_matrices(psis: list[ManyBodyState], orbital_indices: list[int]):
    r"""
    Compute single-particle density matrices for a list of many-body states.

    Parameters
    ----------
    psis : list[ManyBodyState]
        List of many-body states.
    orbital_indices : list[int]
        Orbital indices to include in each density matrix.

    Returns
    -------
    rhos : np.ndarray, shape (n_psi, n_orb, n_orb), dtype=complex
        Density matrices for each state.
    """
    n_psi = len(psis)
    n_orb = len(orbital_indices)
    rhos = np.empty((n_psi, n_orb, n_orb), dtype=complex)
    for i, psi in enumerate(psis):
        rhos[i] = calc_density_matrix(psi, orbital_indices)
    return rhos
