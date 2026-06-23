import numpy as np
import scipy as sp

from impurityModel.ed.block_structure import BlockStructure, build_block_structure

# from .utils import matrix_print, matrix_connectivity_print
from impurityModel.ed.utils import matrix_connectivity_print, matrix_print


def build_imp_bath_blocks(
    H: np.ndarray, n_orb: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    """Build impurity and bath blocks based on the block structure of Hamiltonian H.

    Parameters
    ----------
    H : np.ndarray
        The Hamiltonian matrix.
    n_orb : int
        The number of impurity orbitals.

    Returns
    -------
    impurity_indices : list of list of int
        Impurity orbital indices for each block.
    occupied_indices : list of list of int
        Occupied bath orbital indices for each block.
    unoccupied_indices : list of list of int
        Unoccupied bath orbital indices for each block.
    block_structure : BlockStructure
        The block structure representation of the Hamiltonian.
    """
    block_structure = build_block_structure(H)
    impurity_indices = [None for _ in range(len(block_structure.blocks))]
    occupied_indices = [None for _ in range(len(block_structure.blocks))]
    unoccupied_indices = [None for _ in range(len(block_structure.blocks))]
    for block_i, orbs in enumerate(block_structure.blocks):
        bath_orbs = set(orb for orb in orbs if orb >= n_orb)
        impurity_orbs = set(orbs) - bath_orbs
        impurity_indices[block_i] = sorted(impurity_orbs)
        occupied_indices[block_i] = set(orb for orb in bath_orbs if H[orb, orb] < 0)
        unoccupied_indices[block_i] = sorted(bath_orbs - occupied_indices[block_i])
        occupied_indices[block_i] = sorted(occupied_indices[block_i])
        orbs[:] = impurity_indices[block_i]
    return impurity_indices, occupied_indices, unoccupied_indices, block_structure


def build_H_bath_v(H_dft, ebs_star, vs_star, bath_geometry, block_structure, verbose, extra_verbose):
    """Transform bath parameters from star to chain or Haverkort geometry.

    Parameters
    ----------
    H_dft : np.ndarray
        DFT Hamiltonian matrix.
    ebs_star : list of np.ndarray
        Bath energies for each block in star geometry.
    vs_star : list of np.ndarray
        Hopping parameters for each block in star geometry.
    bath_geometry : str
        The geometry of the bath: "chain", "haver" (Haverkort), or others (fallback to star).
    block_structure : BlockStructure
        The block structure.
    verbose : bool
        Whether to print verbose output.
    extra_verbose : bool
        Whether to print extremely verbose output.

    Returns
    -------
    H_baths : list of np.ndarray
        The bath Hamiltonians for each block.
    vs : list of np.ndarray
        The hopping terms from the impurity to the bath for each block.
    """

    H_baths = []
    vs = []
    if bath_geometry == "chain":
        for i_b, (v, ebs) in enumerate(zip(vs_star, ebs_star)):
            block_ix = block_structure.inequivalent_blocks[i_b]
            block_orbs = block_structure.blocks[block_ix]
            b_ix = np.ix_(block_orbs, block_orbs)
            if len(ebs) <= 1:
                H_baths.append(np.diag(np.repeat(ebs, v.shape[1])))
                vs.append(v.reshape((v.shape[0] * v.shape[1], v.shape[2])))
                continue
            vc, hc = double_chains(H_dft[b_ix], v, ebs, verbose)
            H_baths.append(hc)
            vs.append(vc)
        if verbose:
            for bi, (Hb, vb) in enumerate(zip(H_baths, vs)):
                print(
                    f"Block {bi} (impurity orbitals {block_structure.blocks[block_structure.inequivalent_blocks[bi]]})"
                )
                matrix_print(Hb, "Chain bath")
                matrix_print(vb, "Chain hopping")
                print("")
            print("=" * 80)
    elif bath_geometry == "single_chain":
        for i_b, (v, ebs) in enumerate(zip(vs_star, ebs_star)):
            block_ix = block_structure.inequivalent_blocks[i_b]
            block_orbs = block_structure.blocks[block_ix]
            b_ix = np.ix_(block_orbs, block_orbs)
            if len(ebs) <= 1:
                H_baths.append(np.diag(np.repeat(ebs, v.shape[1])))
                vs.append(v.reshape((v.shape[0] * v.shape[1], v.shape[2])))
                continue
            vc, hc = single_chain(H_dft[b_ix], v, ebs, verbose)
            H_baths.append(hc)
            vs.append(vc)
        if verbose:
            for bi, (Hb, vb) in enumerate(zip(H_baths, vs)):
                print(
                    f"Block {bi} (impurity orbitals {block_structure.blocks[block_structure.inequivalent_blocks[bi]]})"
                )
                matrix_print(Hb, "Single Chain bath")
                matrix_print(vb, "Single Chain hopping")
                print("")
            print("=" * 80)
    elif bath_geometry == "haver":
        for i_b, (v, ebs) in enumerate(zip(vs_star, ebs_star)):
            block_ix = block_structure.inequivalent_blocks[i_b]
            block_orbs = block_structure.blocks[block_ix]
            b_ix = np.ix_(block_orbs, block_orbs)
            # For the linked double chains to make sense we need at least 3 bath states,
            # otherwise we might as well just use a star geometry
            if len(ebs) <= 2:
                H_baths.append(np.diag(np.repeat(ebs, len(block_orbs))))
                vs.append(v.reshape(v.shape[0] * len(block_orbs), len(block_orbs)))
                continue

            vh, Hh = linked_double_chain(H_dft[b_ix], v, ebs, verbose=verbose, extremely_verbose=extra_verbose)
            H_baths.append(Hh)
            vs.append(vh)
        if verbose:
            for bi, (Hb, vb) in enumerate(zip(H_baths, vs)):
                print(
                    f"Block {bi} (impurity orbitals {block_structure.blocks[block_structure.inequivalent_blocks[bi]]})"
                )
                matrix_print(Hb, "Haverkort bath")
                matrix_print(vb, "Haverkort hopping")
                print("")
            print("=" * 80)
    # Star geometry is the fallback
    else:
        for i_b, (v, ebs) in enumerate(zip(vs_star, ebs_star)):
            H_baths.append(np.diag(np.repeat(ebs, v.shape[1])))
            vs.append(v.reshape(v.shape[0] * v.shape[1], v.shape[2]))
    return H_baths, vs


def build_full_bath(
    H_bath_inequiv: list[np.ndarray], v_inequiv: list[np.ndarray], block_structure: BlockStructure
) -> np.ndarray:
    """Build the full bath Hamiltonian and hopping matrices from block components.

    Parameters
    ----------
    H_bath_inequiv : list of np.ndarray
        List of bath Hamiltonians for inequivalent blocks.
    v_inequiv : list of np.ndarray
        List of hopping matrices for inequivalent blocks.
    block_structure : BlockStructure
        The block structure.

    Returns
    -------
    H_bath_full : np.ndarray
        The block-diagonalized full bath Hamiltonian matrix.
    v_full : np.ndarray
        The stacked full hopping matrix.
    """
    (
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transposed_blocks,
        inequivalent_blocks,
    ) = block_structure
    n_orb = sum(len(b) for b in blocks)
    H_baths = [None] * len(blocks)
    vs = [None] * len(blocks)
    for i, block_i in enumerate(inequivalent_blocks):
        H_bath = H_bath_inequiv[i]
        v = v_inequiv[i]
        for b in identical_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy()
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
        for b in transposed_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy().T
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
        for b in particle_hole_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy() @ (-np.identity(H_bath.shape[0]))
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
        for b in particle_hole_and_transposed_blocks[block_i]:
            v_tmp = np.zeros((v.shape[0], n_orb), dtype=complex)
            H_baths[b] = H_bath.copy().T @ (-np.identity(H_bath.shape[0]))
            v_tmp[:, blocks[b]] = v
            vs[b] = v_tmp
    return sp.linalg.block_diag(*H_baths), np.vstack(vs)


def householder_reflector(A):
    """Compute the Householder reflector matrix for a matrix A.

    Parameters
    ----------
    A : np.ndarray
        The input matrix.

    Returns
    -------
    np.ndarray
        The Householder reflector matrix.
    """
    r = A.shape[1]
    X, Z = np.linalg.qr(A, mode="reduced")
    W, s, V = np.linalg.svd(X[:r], full_matrices=True)
    Vh = np.conj(V.T)

    np.linalg.multi_dot((-W, V, Z))
    Y = X + np.linalg.multi_dot((np.eye(X.shape[0], r), W, V))

    U = np.linalg.multi_dot((Y, Vh, np.diag(1 / np.sqrt(2 + 2 * s))))
    return U


def householder_matrix(v):
    """Compute the Householder transformation matrix from a reflector vector.

    Parameters
    ----------
    v : np.ndarray
        Householder reflector vector/matrix.

    Returns
    -------
    np.ndarray
        The Householder transformation matrix.
    """
    return np.eye(v.shape[0], dtype=v.dtype) - 2 * v @ np.conj(v.T)


def block_qr(A, block_size=1, overwrite_A=False):
    """Perform block QR decomposition on matrix A.

    Parameters
    ----------
    A : np.ndarray
        Matrix to decompose.
    block_size : int, default 1
        Block size for Householder reflections.
    overwrite_A : bool, default False
        If True, overwrite `A` during decomposition.

    Returns
    -------
    Q : np.ndarray
        Unitary matrix Q.
    R : np.ndarray
        Upper triangular block matrix R.
    """
    m, n = A.shape
    A.copy()
    if not overwrite_A:
        A = A.copy()
    n_steps = min(m // block_size, n // block_size)
    Qd = np.eye(m, dtype=A.dtype)
    for i in range(n_steps):
        Ap = A[i * block_size :, i * block_size :].copy()
        Ai = A[i * block_size :, i * block_size : (i + 1) * block_size].copy()

        vi = householder_reflector(Ai)
        hi = householder_matrix(vi)

        Hi = np.eye(m, dtype=Qd.dtype)
        Hi[i * block_size :, i * block_size :] = hi
        A[i * block_size :, i * block_size :] = hi @ Ap
        Qd = Hi @ Qd
    return np.conj(Qd.T), A





def single_chain(H_imp: np.ndarray, vs: np.ndarray, ebs: np.ndarray, verbose: bool = True, extremely_verbose=False):
    """Transform the bath geometry from a star into a single chain.

    Parameters
    ----------
    H_imp : np.ndarray
        Impurity Hamiltonian (block_size, block_size).
    vs : np.ndarray
        Hopping parameters for star geometry (Neb, block_size).
    ebs : np.ndarray
        Bath energies for star geometry (Neb).
    verbose : bool, default True
        Whether to print verbose output.
    extremely_verbose : bool, default False
        Whether to print extremely verbose output.

    Returns
    -------
    chain_v : np.ndarray
        Hopping parameters for chain geometry (Neb_chain, block_size).
    H_bath_chain : np.ndarray
        Hamiltonian describing the bath in chain geometry (Neb_chain, Neb_chain).
    """
    verbose = verbose or extremely_verbose
    if isinstance(H_imp, (int, float, complex)):
        H_imp = np.array([[H_imp]])
    if len(vs.shape) == 1:
        vs = vs.reshape((vs.shape[0], 1))
    sort_idx = np.argsort(ebs, stable=True)
    ebs = ebs[sort_idx]
    vs = vs[sort_idx]
    n_imp = H_imp.shape[1]

    H_star = build_star_geometry_hamiltonian(H_imp, vs, ebs)
    if extremely_verbose:
        matrix_print(H_star, "Original hamiltonian for single chain")
        print("", flush=True)
    
    H_chain = transform_to_lanczos_tridagonal_matrix(H_star, n_imp)
    
    V = H_chain[n_imp:, :n_imp]
    Hb = H_chain[n_imp:, n_imp:]
    
    if verbose:
        matrix_print(H_chain)
        matrix_connectivity_print(H_chain, n_imp, "Block structure of single chain geometry Hamiltonian")
    return V, Hb


def double_chains(H_imp: np.ndarray, vs: np.ndarray, ebs: np.ndarray, verbose: bool = True, extremely_verbose=False):
    """Transform the bath geometry from a star into one or two auxiliary chains.

    The two chains correspond to the occupied and unoccupied parts of the spectra respectively.

    Parameters
    ----------
    H_imp : np.ndarray
        Impurity Hamiltonian (block_size, block_size).
    vs : np.ndarray
        Hopping parameters for star geometry (Neb, block_size).
    ebs : np.ndarray
        Bath energies for star geometry (Neb).
    verbose : bool, default True
        Whether to print verbose output.
    extremely_verbose : bool, default False
        Whether to print extremely verbose output.

    Returns
    -------
    chain_v : np.ndarray
        Hopping parameters for chain geometry (Neb_chain, block_size).
    H_bath_chain : np.ndarray
        Hamiltonian describing the bath in chain geometry (Neb_chain, Neb_chain).
    """
    verbose = verbose or extremely_verbose
    if isinstance(H_imp, (int, float, complex)):
        H_imp = np.array([[H_imp]])
    if len(vs.shape) == 1:
        vs = vs.reshape((vs.shape[0], 1))
    sort_idx = np.argsort(ebs, stable=True)
    ebs = ebs[sort_idx]
    vs = vs[sort_idx]
    n_imp = H_imp.shape[1]

    n_occ = sum(ebs < 0)
    H_occ = build_star_geometry_hamiltonian(H_imp, np.flip(vs[:n_occ], axis=0), np.flip(ebs[:n_occ], axis=0))
    if extremely_verbose:
        matrix_print(H_occ, "Original hamiltonian for occupied part")
        print("", flush=True)
    H_occ = transform_to_lanczos_tridagonal_matrix(H_occ, n_imp)

    H_unocc = build_star_geometry_hamiltonian(H_imp, vs[n_occ:], ebs[n_occ:])
    if extremely_verbose:
        matrix_print(H_unocc, "Original hamiltonian for unoccupied part")
    H_unocc = transform_to_lanczos_tridagonal_matrix(H_unocc, n_imp)
    V = np.vstack((H_occ[n_imp:, :n_imp], H_unocc[n_imp:, :n_imp]))
    Hb = sp.linalg.block_diag(H_occ[n_imp:, n_imp:], H_unocc[n_imp:, n_imp:])
    if verbose:
        H_tmp = np.block([[H_imp, V.T], [V, Hb]])
        matrix_print(H_tmp)
        matrix_connectivity_print(H_tmp, n_imp, "Block structure of double chain geometry Hamiltonian")
    return V, Hb


def build_star_geometry_hamiltonian(H_imp, vs, es):
    """Construct a Hamiltonian matrix in star geometry.

    Parameters
    ----------
    H_imp : np.ndarray or float or complex
        Impurity Hamiltonian block.
    vs : np.ndarray
        Hopping parameters.
    es : np.ndarray
        Bath energies.

    Returns
    -------
    np.ndarray
        The constructed star geometry Hamiltonian matrix.
    """
    if isinstance(H_imp, (float, complex)):
        H_imp = np.array([[H_imp]])
    if len(vs.shape) == 1:
        vs = vs.reshape((vs.shape[0], 1))
    n_imp = H_imp.shape[1]
    n_bath = es.shape[0]
    
    if len(vs.shape) == 2 and vs.shape[0] == n_bath:
        H_star = np.empty((n_imp + n_bath, n_imp + n_bath), dtype=H_imp.dtype)
        H_star[:n_imp, :n_imp] = H_imp
        H_star[n_imp:, :n_imp] = vs
        H_star[:n_imp, n_imp:] = np.conj(vs.T)
        H_star[n_imp:, n_imp:] = np.diag(es)
    elif len(vs.shape) == 3 and vs.shape[1] == 1 and vs.shape[0] == n_bath:
        # Non-degenerate bath packed as (N_bath, 1, n_imp)
        vs_2d = vs.reshape(n_bath, n_imp)
        H_star = np.empty((n_imp + n_bath, n_imp + n_bath), dtype=H_imp.dtype)
        H_star[:n_imp, :n_imp] = H_imp
        H_star[n_imp:, :n_imp] = vs_2d
        H_star[:n_imp, n_imp:] = np.conj(vs_2d.T)
        H_star[n_imp:, n_imp:] = np.diag(es)
    else:
        H_star = np.empty((n_imp + n_bath * n_imp, n_imp + n_bath * n_imp), dtype=H_imp.dtype)
        H_star[:n_imp, :n_imp] = H_imp
        H_star[n_imp:, :n_imp] = vs.reshape((n_imp * n_bath, n_imp))
        H_star[:n_imp, n_imp:] = np.conj(vs.reshape((n_imp * n_bath, n_imp)).T)
        H_star[n_imp:, n_imp:] = np.diag(np.repeat(es, n_imp))
        
    return H_star


def build_block_tridiagonal_hermitian_matrix(diagonals, offdiagonals):
    """Assemble a block tridiagonal Hermitian matrix from diagonal and off-diagonal blocks.

    Parameters
    ----------
    diagonals : np.ndarray
        Diagonal blocks.
    offdiagonals : np.ndarray
        Off-diagonal blocks.

    Returns
    -------
    np.ndarray
        The assembled block tridiagonal Hamiltonian.
    """
    num_blocks = diagonals.shape[0]
    block_size = diagonals.shape[1]
    num_orbs = num_blocks * block_size
    H = np.zeros((num_orbs, num_orbs), dtype=diagonals.dtype)
    if num_blocks == 0:
        return H

    def idx_(j):
        """Slice generator for block index j.

        Parameters
        ----------
        j : int
            Block index.

        Returns
        -------
        slice
            Slice object for index range.
        """
        return slice(j * block_size, (j + 1) * block_size)

    for i in range(num_blocks - 1):
        H[idx_(i), idx_(i)] = diagonals[i]
        H[idx_(i), idx_(i + 1)] = np.conj(offdiagonals[i].T)
        H[idx_(i + 1), idx_(i)] = offdiagonals[i]
    i = num_blocks - 1
    H[idx_(i), idx_(i)] = diagonals[i]
    return H


def transform_to_lanczos_tridagonal_matrix(H, n_imp):
    """Transform a Hamiltonian to Lanczos tridiagonal form.

    Parameters
    ----------
    H : np.ndarray
        The Hamiltonian matrix.
    n_imp : int
        The number of impurity orbitals.

    Returns
    -------
    np.ndarray
        The transformed tridiagonal Hamiltonian matrix.
    """
    matrix_print(H, "Matrix to tridiagonalize")
    if H.shape[0] <= 2:
        return H
    Hb = H[n_imp:, n_imp:]
    V0 = H[n_imp:, :n_imp]
    
    block_size = V0.shape[1]
    V0_q, V0_r = sp.linalg.qr(V0, mode="economic", overwrite_a=False, check_finite=True)
    
    if V0_q.shape[0] == 0:
        alphas = np.empty((0, block_size, block_size), dtype=H.dtype)
        betas = np.empty((0, block_size, block_size), dtype=H.dtype)
    else:
        from impurityModel.ed.BlockLanczosArray import block_lanczos_array, Reort

        def converged(alphas, betas, **kwargs):
            return len(betas) > 0 and np.linalg.norm(betas[-1]) < 1e-14

        alphas, betas, _ = block_lanczos_array(
            psi0=V0_q,
            h_op=Hb,
            converged=converged,
            reort=Reort.FULL,
            max_iter=int(np.ceil(Hb.shape[0] / block_size))
        )

    H_tridiagonal = build_block_tridiagonal_hermitian_matrix(alphas, betas)
    
    if np.isrealobj(H) and np.allclose(H_tridiagonal.imag, 0, atol=1e-12):
        H_tridiagonal = H_tridiagonal.real
        
    new_N = n_imp + H_tridiagonal.shape[0]
    res = np.zeros((new_N, new_N), dtype=np.result_type(H.dtype, H_tridiagonal.dtype))
    
    res[:n_imp, :n_imp] = H[:n_imp, :n_imp]
    
    # Only place the coupling if the tridiagonal matrix has at least one block
    if H_tridiagonal.shape[0] > 0:
        res[n_imp : n_imp + block_size, :n_imp] = V0_r
        res[:n_imp, n_imp : n_imp + block_size] = np.conj(V0_r.T)
        
    res[n_imp:, n_imp:] = H_tridiagonal

    return res


def create_decoupled_hamiltonian(H, n_imp):
    """Take any Hamiltonian, transform it to contain two separate decoupled blocks.

    Parameters
    ----------
    H : np.ndarray
        The input Hamiltonian matrix.
    n_imp : int
        The number of impurity orbitals.

    Returns
    -------
    H_decoupled : np.ndarray
        The decoupled Hamiltonian.
    pivot : int
        The pivot index separating the occupied and unoccupied blocks.
    Q_decoupled : np.ndarray
        The unitary transformation matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    sort_idx = np.argsort(eigvals)
    eigvals[:] = eigvals[sort_idx]
    eigvecs[:] = eigvecs[:, sort_idx]

    # Put the pivot point at the eigenstate with energy closest to 0
    # In order to ensure we always get two decoupled blocks, the pivot will never
    # be placed at the last eigenstate (unless there is only one eigenstate block.)
    pivot = n_imp * (min(np.argmin(np.abs(eigvals)), max(len(eigvals) - 2 * n_imp, 0)) // n_imp)

    #          [ v_0, . . ., v_pivot-1, v_pivot, ..., v_m-1 ]
    # eigvecs  |                                         |
    #          [                                         ]
    # e_0 <= e_1 <= ... <= e_pivot-1 <= e_pivot <= ... <= e_m-1
    # Put highest energy occupied state first
    Q_occ_orig = eigvecs[:, : pivot + n_imp][:, ::-1]
    # Put lowest energy unoccupied state first
    Q_unocc_orig = eigvecs[:, pivot + n_imp :]

    # Eigenstates closest to the impurity have been places first (by column) in each matrix
    _, Q_occ = block_qr(Q_occ_orig.T, n_imp)
    _, Q_unocc = block_qr(Q_unocc_orig.T, n_imp)
    Q_decoupled = np.empty_like(H)

    # We need to reverse the order of the columns for the transformation of the occupied part, because the pivot sits last in this part
    Q_decoupled[:, : pivot + n_imp] = Q_occ.T[:, ::-1]
    # The unoccupied part does not need to be reversed, since the coupling bath state already sits first in this part
    Q_decoupled[:, pivot + n_imp :] = Q_unocc.T

    return np.linalg.multi_dot((np.conj(Q_decoupled.T), H, Q_decoupled)), pivot, Q_decoupled


def separate_orbital_character(q):
    """Perform SVD decomposition to separate orbital characters.

    Parameters
    ----------
    q : np.ndarray
        The input matrix containing mixed orbital states.

    Returns
    -------
    np.ndarray
        Unitary transformation matrix to restore orbital character.
    """
    U, s, Vh = np.linalg.svd(q, full_matrices=True)
    Um = np.eye(Vh.shape[0], dtype=q.dtype)
    Um[: q.shape[0], : q.shape[0]] = U
    return np.conj(Vh.T) @ np.conj(Um.T)


def linked_double_chain(H_imp, vs, es, verbose=True, extremely_verbose=False):
    """Transform the bath geometry from star to a linked double chain geometry.

    Parameters
    ----------
    H_imp : np.ndarray or float or complex
        Impurity Hamiltonian block.
    vs : np.ndarray
        Hopping parameters.
    es : np.ndarray
        Bath energies.
    verbose : bool, default True
        Whether to print verbose output.
    extremely_verbose : bool, default False
        Whether to print extremely verbose output.

    Returns
    -------
    v_chain : np.ndarray
        Hopping terms from impurity to the linked double chain.
    H_chain : np.ndarray
        The bath Hamiltonian of the linked double chain.
    """
    verbose = verbose or extremely_verbose
    if isinstance(H_imp, int):
        H_imp = np.array([[H_imp]], dtype=float)
    elif isinstance(H_imp, (float, complex)):
        H_imp = np.array([[H_imp]])
    if len(vs.shape) == 1:
        vs = vs.reshape((vs.shape[0], 1))

    n_imp = H_imp.shape[0]
    n_bath = es.shape[0]
    if n_bath == 0:
        return np.empty((0, n_imp), dtype=H_imp.dtype), np.empty((0, 0), dtype=H_imp.dtype)
    H_star = build_star_geometry_hamiltonian(H_imp, vs, es)
    if extremely_verbose:
        matrix_print(
            H_star, "Original Hamiltonian (star geometry), the impurity sits in the top left corner", flush=True
        )

    H_decoupled, imp_index, Q_decoupled = create_decoupled_hamiltonian(H_star, n_imp)
    if extremely_verbose:
        print(
            f"After reshuffling the orbitals, the impurity sits at indices {np.arange(imp_index, imp_index + n_imp)} and the coupling bath state at indices {np.arange(imp_index + n_imp, imp_index + 2 * n_imp)}"
        )
        matrix_print(Q_decoupled, "Orbital character for states")
        matrix_print(H_decoupled, "Hamiltonian transformed into decoupled occupied and unoccupied blocks", flush=True)
    # Undo the mixing of impurity and bath states
    R_couple = np.eye(Q_decoupled.shape[0], dtype=Q_decoupled.dtype)
    R_couple[imp_index : imp_index + 2 * n_imp, imp_index : imp_index + 2 * n_imp] = separate_orbital_character(
        Q_decoupled[:n_imp, imp_index : imp_index + 2 * n_imp]
    )

    if extremely_verbose:
        matrix_print(Q_decoupled @ R_couple, "Restored impurity character")
        matrix_print(
            np.conj(R_couple.T) @ H_decoupled @ R_couple,
            "Hamiltonian transformed into coupled occupied and unoccupied blocks",
            flush=True,
        )

    occ_chain = transform_to_lanczos_tridagonal_matrix(
        H_decoupled[: imp_index + n_imp, : imp_index + n_imp][::-1, ::-1], n_imp
    )[::-1, ::-1]

    # The coupling state sits just after the impurity (i.e. imp_index + n_imp)
    # Check that we have unoccupied states, beyond just the impurity.
    if imp_index + n_imp < H_star.shape[0]:
        top_left = imp_index + n_imp
        unocc_chain = transform_to_lanczos_tridagonal_matrix(
            H_decoupled[top_left:, top_left:], n_imp
        )
        H_tridiagonal_decoupled = sp.linalg.block_diag(occ_chain, unocc_chain)
    else:
        H_tridiagonal_decoupled = occ_chain

    new_imp_index = occ_chain.shape[0] - n_imp

    R_couple_new = np.eye(H_tridiagonal_decoupled.shape[0], dtype=Q_decoupled.dtype)
    R_couple_new[new_imp_index : new_imp_index + 2 * n_imp, new_imp_index : new_imp_index + 2 * n_imp] = separate_orbital_character(
        Q_decoupled[:n_imp, imp_index : imp_index + 2 * n_imp]
    )

    H_linked_chains = np.linalg.multi_dot((np.conj(R_couple_new.T), H_tridiagonal_decoupled, R_couple_new))
    if extremely_verbose:
        matrix_print(H_tridiagonal_decoupled, "Decoupled Hamiltonian with tridiagonal blocks")
        matrix_print(H_linked_chains, "Hamiltonian with coupled tridiagonal blocks")

    indices = np.append(
        np.roll(np.arange(new_imp_index + n_imp), -new_imp_index), np.arange(new_imp_index + n_imp, H_linked_chains.shape[1])
    )
    idx = np.ix_(indices, indices)
    H_linked_chains = H_linked_chains[idx]

    def delta(m1, m2):
        """Calculate the maximum absolute difference between two matrices.

        Parameters
        ----------
        m1 : np.ndarray
            First matrix.
        m2 : np.ndarray
            Second matrix.

        Returns
        -------
        float
            The maximum absolute difference.
        """
        return np.max(np.abs(m2 - m1))

    if verbose:
        matrix_connectivity_print(H_linked_chains, n_imp, "Block structure of linked double chain Hamiltonian")
    if extremely_verbose:
        matrix_print(H_linked_chains, "Hamiltonian with linked chains (impurity sits in the top left corner)")

    return H_linked_chains[n_imp:, :n_imp], H_linked_chains[n_imp:, n_imp:]
