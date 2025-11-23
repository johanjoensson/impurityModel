import numpy as np
import scipy as sp
from impurityModel.ed.block_structure import build_block_structure, BlockStructure

# from .utils import matrix_print, matrix_connectivity_print
from impurityModel.ed.utils import matrix_print, matrix_connectivity_print
from typing import Optional


def build_imp_bath_blocks(
    H: np.ndarray, n_orb: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    block_structure = build_block_structure(H)
    impurity_indices = [None] * len(block_structure.blocks)
    occupied_indices = [None] * len(block_structure.blocks)
    unoccupied_indices = [None] * len(block_structure.blocks)
    for block_i, orbs in enumerate(block_structure.blocks):
        bath_orbs = {orb for orb in orbs if orb >= n_orb}
        impurity_orbs = set(orbs) - bath_orbs
        impurity_indices[block_i] = sorted(impurity_orbs)
        occupied_indices[block_i] = {orb for orb in bath_orbs if H[orb, orb] < 0}
        unoccupied_indices[block_i] = sorted(bath_orbs - occupied_indices[block_i])
        occupied_indices[block_i] = sorted(occupied_indices[block_i])
        orbs[:] = impurity_indices[block_i]
    return impurity_indices, occupied_indices, unoccupied_indices, block_structure


def build_H_bath_v(H_dft, ebs_star, vs_star, bath_geometry, block_structure, verbose, extra_verbose):

    H_baths = []
    vs = []
    if bath_geometry == "chain":
        for v, ebs in zip(vs_star, ebs_star):
            if len(ebs) <= 1:
                H_baths.append(np.diag(ebs))
                vs.append(v)
                continue
            vc, hc = double_chains(v, ebs, verbose)
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
    elif bath_geometry == "haver":
        for i_b, (v, ebs) in enumerate(zip(vs_star, ebs_star)):
            # For the linked double chains to make sense we need at least 3 bath states,
            # otherwise we might as well just use a star geometry
            if len(ebs) <= 2:
                H_baths.append(np.diag(ebs))
                vs.append(v)
                continue

            block_ix = block_structure.inequivalent_blocks[i_b]
            block_orbs = block_structure.blocks[block_ix]
            b_ix = np.ix_(block_orbs, block_orbs)
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
    else:  # bath_geometry == "star"
        H_baths = [np.diag(eb) for eb in ebs_star]
        vs = vs_star
    return H_baths, vs


def build_full_bath(
    H_bath_inequiv: list[np.ndarray], v_inequiv: list[np.ndarray], block_structure: BlockStructure
) -> np.ndarray:
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
    r = A.shape[1]
    X, Z = np.linalg.qr(A, mode="reduced")
    W, s, V = np.linalg.svd(X[:r], full_matrices=True)
    Vh = np.conj(V.T)

    beta = -W @ V @ Z
    Y = X + np.eye(X.shape[0], r) @ W @ V

    U = Y @ (Vh @ np.diag(1 / (np.sqrt(2 + 2 * s))))
    return U


def householder_matrix(v):
    return np.eye(v.shape[0], dtype=v.dtype) - 2 * v @ np.conj(v.T)


def test_householder():
    M = np.ones((4, 4))
    M[[1, 1, 2, 2, 3, 3, 3], [1, 3, 2, 3, 1, 2, 3]] = -1
    a1 = householder_reflector(M[:, 0:2])
    H1 = householder_matrix(a1)
    H1_exact = np.zeros_like(M)
    H1_exact[[0, 0, 1, 1, 2, 3], [0, 2, 1, 3, 0, 1]] = -1
    H1_exact[[2, 3], [2, 3]] = 1
    H1_exact *= np.sqrt(1 / 2)
    assert np.allclose(H1, H1_exact)

    Q, R = block_qr(M, 1)
    R_exact = np.eye(4)
    R_exact[[0, 1, 2, 3], [0, 1, 2, 3]] = [-2, 2, 2, -1]
    R_exact[:3, 3] = 1
    Q_exact = 1 / 2 * np.ones((4, 4))
    Q_exact[[0, 1, 2, 3, 1, 3, 2, 3, 0, 3], [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]] *= -1
    assert np.allclose(Q, Q_exact)
    assert np.allclose(R, R_exact)

    M = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
    Q, R = block_qr(M, 1)
    Q_np, R_np = np.linalg.qr(M)
    assert np.allclose(Q @ R, M)
    assert np.allclose(np.conj(Q.T) @ Q, np.conj(Q_np).T @ Q_np)


def block_qr(A, block_size=1, overwrite_A=False):
    m, n = A.shape
    R = A.copy()
    if not overwrite_A:
        A = A.copy()
    n_steps = min(m // block_size, n // block_size)
    Qd = np.eye(m, dtype=A.dtype)
    for i in range(n_steps):
        Ap = A[i * block_size :, i * block_size :].copy()
        Ai = A[i * block_size :, i * block_size : (i + 1) * block_size].copy()

        vi = householder_reflector(Ai)
        hi = householder_matrix(vi)
        assert np.allclose(np.conj(hi.T) @ hi, np.eye(hi.shape[1]))
        assert np.allclose(np.conj(hi.T), hi)

        Hi = np.eye(m, dtype=Qd.dtype)
        Hi[i * block_size :, i * block_size :] = hi
        A[i * block_size :, i * block_size :] = hi @ Ap
        assert np.allclose(np.conj(Hi.T) @ Hi, np.eye(m))
        assert np.allclose(np.conj(Hi.T), Hi)
        Qd = Hi @ Qd
    assert np.allclose(np.conj(Qd.T) @ Qd, np.eye(Qd.shape[1]))
    return np.conj(Qd.T), A


def get_lanczos_vectors(H, v0, alphas, betas):
    v0, _ = sp.linalg.qr(v0, mode="economic")
    n_imp = v0.shape[1]
    n_it = alphas.shape[0]

    Q = np.zeros((v0.shape[0], n_imp * n_it), dtype=complex)
    Q[:, :n_imp] = v0

    # betas = np.append(betas, np.zeros((1, n_imp, n_imp)), axis=0)
    qim = np.zeros_like(v0)
    for i in range(n_it - 1):
        qi = Q[:, i * n_imp : (i + 1) * n_imp]
        wp = H @ qi
        wp -= qi @ alphas[i] + qim @ np.conj(betas[i - 1].T)
        for _ in range(2):
            wp -= Q[:, : i * n_imp] @ np.conj(Q[:, : i * n_imp].T) @ wp
        qim = qi
        tmp, _ = sp.linalg.qr(wp, mode="economic")
        Q[:, (i + 1) * n_imp : (i + 2) * n_imp] = tmp

    return Q


def tridiagonalize(H, v0):
    assert H.shape[0] == v0.shape[0]
    block_size = v0.shape[1]

    v0, v0_tilde = sp.linalg.qr(v0, mode="economic", overwrite_a=True, check_finite=False)

    if v0.shape[0] == 0:
        return (
            np.empty((0, block_size, block_size), dtype=H.dtype),
            np.empty((0, block_size, block_size), dtype=H.dtype),
            v0_tilde,
        )

    N = H.shape[0]
    Q = np.zeros((N, N), dtype=complex)
    q = np.zeros((2, N, block_size), dtype=complex)
    q[1, :, :block_size] = v0
    alphas = np.empty((N // block_size, block_size, block_size), dtype=complex)
    betas = np.zeros((N // block_size, block_size, block_size), dtype=complex)

    for i in range(N // block_size):
        wp = H @ q[1]
        alphas[i] = np.conj(q[1].T) @ wp
        wp -= q[1] @ alphas[i] + q[0] @ np.conj(betas[i - 1].T)
        for _ in range(2):
            wp -= Q @ np.conj(Q.T) @ wp
        Q[:, i * block_size : (i + 1) * block_size] = q[1]
        q[0] = q[1]
        q[1], betas[i] = np.linalg.qr(wp)

    return alphas, betas, v0_tilde


def double_chains(vs: np.ndarray, ebs: np.ndarray, verbose: bool):
    """
    Transform the bath geometry from a star into one or two auxilliary chains.
    The two chains correspond to the occupied and unoccupied parts of the spectra respectively.
    Returns the Hopping term from the impurity onto the chains and the chain bath Hamiltonian.
    Parameters:
    vs: np.ndarray((Neb, block_size)) - Hopping parameters for star geometry.
    ebs: np.ndarray((Neb)) - Bath energies for star geometry
    Returns:
    chain_v, H_bath_chain
    chain_v: np.ndarray((Neb_chain, block_size)) - Hopping parameters for chain geometry.
    H_bath_chain: np.ndarray((Neb_chain, Neb_chain)) - Hamiltonian describind the bath in chain geometry.
    """
    sort_idx = np.argsort(ebs)
    ebs = ebs[sort_idx]
    vs = vs[sort_idx, :]
    n_imp = vs.shape[1]

    n_occ = sum(ebs < 0)
    ebs[:n_occ] = ebs[:n_occ][::-1]
    vs[:n_occ] = vs[:n_occ][::-1]
    H_occ = build_star_geometry_hamiltonian(np.zeros((n_imp, n_imp), dtype=vs.dtype), vs[:n_occ], ebs[:n_occ])
    if verbose:
        matrix_print(H_occ, "Original hamiltonian for occupied part")
        print("", flush=True)
    H_occ[:] = transform_to_lanczos_tridagonal_matrix(H_occ, n_imp)

    H_unocc = build_star_geometry_hamiltonian(np.zeros((n_imp, n_imp), dtype=vs.dtype), vs[n_occ:], ebs[n_occ:])
    if verbose:
        matrix_print(H_unocc, "Original hamiltonian for unoccupied part")
    H_unocc[:] = transform_to_lanczos_tridagonal_matrix(H_unocc, n_imp)
    V = np.vstack((H_occ[n_imp:, :n_imp], H_unocc[n_imp:, :n_imp]))
    Hb = sp.linalg.block_diag(H_occ[n_imp:, n_imp:], H_unocc[n_imp:, n_imp:])
    if verbose:
        H_imp = np.ones((n_imp, n_imp))
        H_tmp = np.block([[H_imp, V.T], [V, Hb]])
        matrix_print(H_tmp)
        matrix_connectivity_print(H_tmp, n_imp, "Block structure of double chain geometry Hamiltonian")
    return V, Hb


def haverkort_chain(eloc, tns, ens):
    block_size = eloc.shape[0]
    assert (
        block_size == 1
    ), f"The current implementation does not support offdiagonal elements in the hybridization!\n{block_size=}"
    hsize = len(ens) + 1
    H = np.zeros((hsize, hsize), dtype=complex)
    H[0, 0] = eloc
    for i in range(len(ens)):
        H[i + 1, i + 1] = ens[i]
        H[i, i + 1] = np.conj(tns[i].T)
        H[i + 1, i] = tns[i]

    w, v = np.linalg.eigh(H)

    n = np.argmin(np.abs(w))

    prevtocc = v[:, n - 1 :: -1].transpose()
    prevtunocc = v[:, n:].transpose()
    qocc, vtocc = sp.linalg.qr(prevtocc, check_finite=False, overwrite_a=True)
    qunocc, vtunocc = sp.linalg.qr(prevtunocc, check_finite=False, overwrite_a=True)

    vtot = np.zeros((hsize, hsize), dtype=complex)
    vtot[:, 0:n] = vtocc[::-1, :].transpose()
    vtot[:, n:hsize] = vtunocc.transpose()

    matrix_print(vtot, "Hopping before rotating impurity")
    # Get the tridiagonal terms
    for i in range(hsize - 1):
        tmp = np.conj(vtot[:, i].T) @ H @ vtot[:, i + 1]
        if np.real(tmp) < 0:  # Adjust the phase of the eigenvectors
            vtot[:, i + 1] = -vtot[:, i + 1]

    # Get the final transform to extract the impurity orbital (It goes into element n-1)
    cs = vtot[0, n - 1 : n + 1]
    r = np.linalg.norm(cs)
    R = np.empty((2, 2), dtype=complex)
    R[0, 0] = np.conj(cs[0]) / r
    R[1, 0] = -np.conj(cs[1]) / r
    R[0, 1] = cs[1] / r
    R[1, 1] = cs[0] / r
    matrix_print(R, "R")

    vtot[:, n - 1 : n + 1] = vtot[:, n - 1 : n + 1] @ np.conj(R.T)
    matrix_print(vtot, "Hopping after rotating impurity")

    indices = np.append(np.roll(np.arange(0, n), 1), np.arange(n, hsize))
    idx = np.ix_(indices, indices)
    Hnew = np.conj(vtot.T) @ H @ vtot
    matrix_print(Hnew, "linked double chain Hamiltonian")
    Hnew = Hnew[idx]

    assert np.allclose(np.linalg.eigvalsh(H), np.linalg.eigvalsh(Hnew))

    return Hnew[block_size:, :block_size].copy(), Hnew[block_size:, block_size:].copy()


def build_star_geometry_hamiltonian(H_imp, vs, es):
    n_imp = vs.shape[1]
    n_bath = len(es)
    H_star = np.empty((n_imp + n_bath, n_imp + n_bath), dtype=H_imp.dtype)
    H_star[:n_imp, :n_imp] = H_imp
    H_star[n_imp:, :n_imp] = vs
    H_star[:n_imp, n_imp:] = np.conj(vs.T)
    H_star[n_imp:, n_imp:] = np.diag(es)
    return H_star


def build_block_tridiagonal_hermitian_matrix(diagonals, offdiagonals):
    num_blocks = diagonals.shape[0]
    block_size = diagonals.shape[1]
    num_orbs = num_blocks * block_size
    H = np.zeros((num_orbs, num_orbs), dtype=diagonals.dtype)
    if num_blocks == 0:
        return H

    def idx_(j):
        return slice(j * block_size, (j + 1) * block_size)

    for i in range(num_blocks - 1):
        H[idx_(i), idx_(i)] = diagonals[i]
        H[idx_(i), idx_(i + 1)] = np.conj(offdiagonals[i].T)
        H[idx_(i + 1), idx_(i)] = offdiagonals[i]
    i = num_blocks - 1
    H[idx_(i), idx_(i)] = diagonals[i]
    return H


def transform_to_lanczos_tridagonal_matrix(H, n_imp):
    Hb = H[n_imp:, n_imp:]
    V0 = H[n_imp:, :n_imp]
    alphas, betas, V0 = tridiagonalize(Hb, V0)
    H_tridiagonal = build_block_tridiagonal_hermitian_matrix(alphas, betas)
    res = np.zeros_like(H)
    res[:n_imp, :n_imp] = H[:n_imp, :n_imp]
    res[n_imp : 2 * n_imp, :n_imp] = V0
    res[:n_imp, n_imp : 2 * n_imp] = np.conj(V0.T)
    res[n_imp:, n_imp:] = H_tridiagonal

    assert np.allclose(
        np.linalg.eigvalsh(H), np.linalg.eigvalsh(res)
    ), f"{np.linalg.eigvalsh(H)=}\n{np.linalg.eigvalsh(res)=}"
    return res


def create_decoupled_hamiltonian(H, n_imp):
    """
    Take any Hamiltonian, transform it to contain two separate decoupled blocks.
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
    Q_coupled = np.empty_like(eigvecs)
    Q_coupled[:, : pivot + n_imp] = Q_occ_orig
    Q_coupled[:, pivot + n_imp :] = Q_unocc_orig

    _, Q_occ = block_qr(Q_occ_orig.T, n_imp)
    _, Q_unocc = block_qr(Q_unocc_orig.T, n_imp)
    Q_decoupled = np.empty_like(H)

    Q_decoupled[:, : pivot + n_imp] = Q_occ.T[:, ::-1]
    Q_decoupled[:, pivot + n_imp :] = Q_unocc.T

    return np.linalg.multi_dot((np.conj(Q_decoupled.T), H, Q_decoupled)), pivot, Q_decoupled


def separate_orbital_character(q):
    U, s, Vh = np.linalg.svd(q, full_matrices=True)
    Um = np.eye(Vh.shape[0], dtype=q.dtype)
    Um[: q.shape[0], : q.shape[0]] = U
    return np.conj(Vh.T) @ np.conj(Um.T)


def linked_double_chain(H_imp, vs, es, verbose=True, extremely_verbose=False):
    verbose = verbose or extremely_verbose
    if isinstance(H_imp, (float, complex)):
        H_imp = np.array([H_imp])
    if len(vs.shape) == 1:
        vs = vs.reshape((vs.shape[0], 1))

    n_imp = H_imp.shape[0]
    n_bath = es.shape[0]
    if n_bath == 0:
        return np.empty((0, n_imp), dtype=H_imp.dtype), np.empty((0, 0), dtype=complex)
    H_star = build_star_geometry_hamiltonian(H_imp, vs, es)
    if extremely_verbose:
        matrix_print(
            H_star, "Original Hamiltonian (star geometry), the impurity sits in the top left corner", flush=True
        )

    H_decoupled, imp_index, Q_decoupled = create_decoupled_hamiltonian(H_star, n_imp)
    if extremely_verbose:
        print(
            f"After reshuffling the orbitals, the impurity sits at indices {np.arange(imp_index, imp_index+n_imp)} and the coupling bath state at indices {np.arange(imp_index + n_imp, imp_index+2*n_imp)}"
        )
        matrix_print(Q_decoupled, "Orbital character for states")
        matrix_print(H_decoupled, "Hamiltonian transformed into decoupled occupied and unoccupied blocks")
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
        )

    H_tridiagonal_decoupled = np.zeros_like(H_decoupled)
    H_tridiagonal_decoupled[: imp_index + n_imp, : imp_index + n_imp] = transform_to_lanczos_tridagonal_matrix(
        H_decoupled[: imp_index + n_imp, : imp_index + n_imp][::-1, ::-1], n_imp
    )[::-1, ::-1]
    if imp_index < H_star.shape[0]:
        # The coupling bath state sits in the top left corner of this block
        top_left = imp_index + n_imp
        H_tridiagonal_decoupled[top_left:, top_left:] = transform_to_lanczos_tridagonal_matrix(
            H_decoupled[top_left:, top_left:], n_imp
        )

    H_linked_chains = np.conj(R_couple.T) @ H_tridiagonal_decoupled @ R_couple
    if extremely_verbose:
        matrix_print(H_tridiagonal_decoupled, "Decoupled Hamiltonian with tridiagonal blocks")
        matrix_print(H_linked_chains, "Hamiltonian with coupled tridiagonal blocks")

    indices = np.append(
        np.roll(np.arange(imp_index + n_imp), -imp_index), np.arange(imp_index + n_imp, H_linked_chains.shape[1])
    )
    idx = np.ix_(indices, indices)
    H_linked_chains = H_linked_chains[idx]

    def delta(m1, m2):
        return np.max(np.abs(m2 - m1))

    if verbose:
        matrix_connectivity_print(H_linked_chains, n_imp, "Block structure of linked double chain Hamiltonian")
    if extremely_verbose:
        matrix_print(H_linked_chains, "Hamiltonian with linked chains (impurity sits in the top left corner)")

    # Make sure that we have not changed the spectrum of the Hamiltonian
    assert np.allclose(
        np.linalg.eigvalsh(H_star), np.linalg.eigvalsh(H_tridiagonal_decoupled), atol=np.finfo(float).eps
    ), f"{np.linalg.eigvalsh(H_star)}\n{np.linalg.eigvalsh(H_tridiagonal_decoupled)}"
    assert np.allclose(
        np.linalg.eigvalsh(H_star), np.linalg.eigvalsh(H_linked_chains), atol=np.finfo(float).eps
    ), f"{np.linalg.eigvalsh(H_star)}\n{np.linalg.eigvalsh(H_linked_chains)}"
    # Make sure that we have restored the impurity block exactly
    assert np.allclose(H_linked_chains[:n_imp, :n_imp], H_imp, atol=np.finfo(float).eps)
    return H_linked_chains[n_imp:, :n_imp], H_linked_chains[n_imp:, n_imp:]


if __name__ == "__main__":
    test_householder()
    n_orb = 1
    n_b = 8 * n_orb
    n = n_orb + n_b
    H_start = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    H_start = 1 / 2 * (H_start + np.conj(H_start.T))

    h_imp = H_start[:n_orb, :n_orb]
    v = H_start[n_orb:, :n_orb]
    eb = -np.linspace(1, 5, num=n_b)
    # eb = np.linalg.eigvals(H_start[n_orb:, n_orb:])

    v, hb = linked_double_chain(h_imp, v, eb, extremely_verbose=True)
    matrix_print(hb, "bath hamiltonian")
    matrix_print(v, "Hopping term")
