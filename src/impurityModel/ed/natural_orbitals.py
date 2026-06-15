import numpy as np


def get_huge_star(w, hyb, n_bins=1000, grid_type="linear"):
    """Discretize the hybridization function on the real axis into a huge star geometry.

    Parameters
    ----------
    w : np.ndarray
        1D array of frequencies (N_w)
    hyb : np.ndarray
        3D array of hybridization (N_w, n_imp, n_imp)
    n_bins : int
        Number of energy intervals
    grid_type : str
        Type of grid: "linear" or "logarithmic"

    Returns
    -------
    E_huge : np.ndarray
        1D array of bath energies
    V_huge : np.ndarray
        2D array of hoppings (N_bath, n_imp)
    """
    n_w = len(w)
    n_imp = hyb.shape[1]

    # Spectral function A(w) = -1/pi Im(hyb)
    A = -np.imag(hyb) / np.pi

    if grid_type == "logarithmic":
        w_core = 1e-4
        w_min, w_max = w[0], w[-1]
        if w_min < -w_core and w_max > w_core:
            n_neg = n_bins // 2
            n_pos = n_bins - n_neg
            neg_edges = -np.geomspace(abs(w_min), w_core, n_neg)
            pos_edges = np.geomspace(w_core, w_max, n_pos)
            bin_edges = np.concatenate((neg_edges, [0.0], pos_edges))
        elif w_min >= -w_core:
            bin_edges = np.geomspace(max(w_min, w_core), max(w_max, w_core * 2), n_bins + 1)
        else:
            bin_edges = -np.geomspace(abs(w_min), max(abs(w_max), w_core * 2), n_bins + 1)
        bin_edges[0] = w[0]
        bin_edges[-1] = w[-1]
    else:
        bin_edges = np.linspace(w[0], w[-1], n_bins + 1)

    bin_indices = np.searchsorted(w, bin_edges)

    E_huge = []
    V_huge = []

    for i in range(n_bins):
        start = bin_indices[i]
        end = bin_indices[i + 1]
        if end <= start:
            continue
        if start == end - 1:
            end = min(start + 2, n_w)
            if start == end - 1:
                start = max(0, end - 2)

        w_bin = w[start:end]
        A_bin = A[start:end]

        # Integrate A(w) over the bin
        M_k = np.trapz(A_bin, w_bin, axis=0)

        # Integrate w * Tr(A(w)) to find center of mass energy
        tr_A = np.trace(A_bin, axis1=1, axis2=2)
        norm = np.trapz(tr_A, w_bin)

        if norm < 1e-12:
            continue

        E_k = np.trapz(w_bin * tr_A, w_bin) / norm

        # Diagonalize M_k
        evals, evecs = np.linalg.eigh(M_k)

        for idx in range(n_imp):
            if evals[idx] > 1e-12:
                E_huge.append(E_k)
                V_huge.append(np.sqrt(evals[idx]) * evecs[:, idx])

    if len(E_huge) == 0:
        return np.array([]), np.empty((0, n_imp))

    return np.array(E_huge), np.array(V_huge)


def build_mean_field_hamiltonian(H_imp, E_huge, V_huge):
    """Build the full mean field Hamiltonian.

    Parameters
    ----------
    H_imp : np.ndarray
        (n_imp, n_imp) impurity Hamiltonian block
    E_huge : np.ndarray
        (N_bath,) bath energies
    V_huge : np.ndarray
        (N_bath, n_imp) hoppings

    Returns
    -------
    np.ndarray
        The full non-interacting Hamiltonian matrix
    """
    n_imp = H_imp.shape[0]
    N_bath = len(E_huge)
    N_tot = n_imp + N_bath

    H_MF = np.zeros((N_tot, N_tot), dtype=complex)
    H_MF[:n_imp, :n_imp] = H_imp
    H_MF[n_imp:, n_imp:] = np.diag(E_huge)
    H_MF[n_imp:, :n_imp] = V_huge
    H_MF[:n_imp, n_imp:] = np.conj(V_huge.T)

    return H_MF


def truncate_natural_orbitals(H_imp, w, hyb, n_keep, n_bins=1000, grid_type="linear"):
    """Perform the Natural Orbitals truncation.

    Parameters
    ----------
    H_imp : np.ndarray
        The local impurity Hamiltonian block
    w : np.ndarray
        Frequencies
    hyb : np.ndarray
        Hybridization function
    n_keep : int
        Number of natural orbitals to keep
    n_bins : int
        Number of bins for the initial huge star
    grid_type : str
        Grid type: "linear" or "logarithmic"

    Returns
    -------
    E_opt : np.ndarray
        Truncated bath energies
    V_opt_reshaped : np.ndarray
        Truncated hoppings reshaped to (N_keep, 1, n_imp)
    """
    n_imp = H_imp.shape[0]
    E_huge, V_huge = get_huge_star(w, hyb, n_bins, grid_type)

    if len(E_huge) == 0:
        return np.array([]), np.empty((0, 1, n_imp), dtype=complex)

    H_MF = build_mean_field_hamiltonian(H_imp, E_huge, V_huge)

    # Diagonalize H_MF
    eigvals, U = np.linalg.eigh(H_MF)

    # Occupied states (assuming mu=0)
    occ_idx = np.where(eigvals < 0.0)[0]
    U_occ = U[:, occ_idx]

    # Bath density matrix
    U_bath_occ = U_occ[n_imp:, :]
    D_bath = U_bath_occ @ np.conj(U_bath_occ.T)

    # Diagonalize bath density matrix
    occ_bath, W = np.linalg.eigh(D_bath)

    # Entanglement (distance from frozen 0 or 1)
    entanglement = np.minimum(occ_bath, 1.0 - occ_bath)

    # Select top n_keep entangled states
    sort_idx = np.argsort(entanglement)[::-1]
    actual_keep = min(n_keep, len(E_huge))
    keep_idx = sort_idx[:actual_keep]

    W_trunc = W[:, keep_idx]

    # Project bath Hamiltonian
    H_bath_huge = np.diag(E_huge)
    H_bath_trunc = np.conj(W_trunc.T) @ H_bath_huge @ W_trunc

    # Diagonalize H_bath_trunc to get independent scalar bath states
    E_opt, Z = np.linalg.eigh(H_bath_trunc)

    # Rotate hoppings
    V_opt = np.conj(Z.T) @ np.conj(W_trunc.T) @ V_huge

    # Reshape V_opt to (N_keep, 1, n_imp) to match fit_hyb format
    V_opt_reshaped = V_opt.reshape((actual_keep, 1, n_imp))

    return E_opt, V_opt_reshaped


def fit_hyb_natural_orbitals(w, hyb, H_imp_blocks, bath_states_per_orbital, block_structure, n_bins=1000, grid_type="linear"):
    """Wrapper to behave like fit_hyb but using Natural Orbitals.

    Parameters
    ----------
    w : np.ndarray
        Frequencies
    hyb : np.ndarray
        Block-diagonalized hybridization
    H_imp_blocks : list of np.ndarray
        List of H_imp for each inequivalent block.
    bath_states_per_orbital : int
        Number of bath states per impurity orbital
    block_structure : BlockStructure
        The block structure mappings.
    n_bins : int
        Number of bins for discretization
    grid_type : str
        Grid type: "linear" or "logarithmic"

    Returns
    -------
    ebs_star : list of np.ndarray
        List of bath energies for each block
    vs_star : list of np.ndarray
        List of hopping arrays for each block
    """
    ebs_star = []
    vs_star = []

    for i, block_i in enumerate(block_structure.inequivalent_blocks):
        orbs = block_structure.blocks[block_i]
        hyb_block = hyb[:, np.ix_(orbs, orbs)[0], np.ix_(orbs, orbs)[1]]
        H_imp_block = H_imp_blocks[i]

        n_keep = bath_states_per_orbital * len(orbs)

        E_opt, V_opt = truncate_natural_orbitals(H_imp_block, w, hyb_block, n_keep, n_bins, grid_type)

        ebs_star.append(E_opt)
        vs_star.append(V_opt)

    return ebs_star, vs_star
