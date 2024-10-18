import numpy as np
import scipy as sp
from impurityModel.ed.lanczos import eigsh


def matrix_print(m, label=None):
    if label is not None:
        print(label)
    print(
        "\n".join(
            [
                " ".join([f"{np.real(el): .12f}" for el in row])
                # " ".join([f"{np.real(el): .3f} {np.imag(el):+.3f}j" for el in row])
                for row in m
            ]
        )
    )


def tridiagonalize(eb, tns):
    assert len(eb) == tns.shape[0]
    block_size = tns.shape[1]

    v0, v0_tilde = sp.linalg.qr(tns, mode="economic")

    H = np.diag(eb)
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
        wp -= Q @ np.conj(Q.T) @ wp
        Q[:, i * block_size : (i + 1) * block_size] = q[1]
        q[0] = q[1]
        q[1], betas[i] = np.linalg.qr(wp)

    # e_orig = np.linalg.eigvalsh(H)
    # if alphas.shape[0] > 1:
    #     e_tri = eigsh(alphas, betas, eigvals_only=True)
    # else:
    #     e_tri = np.linalg.eigvalsh(alphas[0])
    # e_tri = sp.linalg.eigvalsh_tridiagonal(alphas[:, 0, 0].real, betas[:-1, 0, 0].real)
    # assert np.allclose(
    #     np.sort(eb), np.sort(e_tri), atol=np.finfo(float).eps
    # ), f"{np.max(np.abs(eb-e_tri))=}\n{eb=}\n{e_tri=}\n{alphas=}\n{betas=}"
    return alphas, betas, v0_tilde


def edchains(vs, ebs):
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
    n_block_orb = vs.shape[1]
    n = sum(ebs < 0)
    sorted_indices = np.argsort(ebs)
    ebs = ebs[sorted_indices]
    vs = vs[sorted_indices]
    ebs[:n] = ebs[:n][::-1]
    vs[:n] = vs[:n][::-1]
    chain_eb, chain_v, v0_tilde = tridiagonalize(ebs[:n], vs[:n])
    chain_v_occ = np.zeros((len(chain_eb) * n_block_orb, n_block_orb), dtype=complex)
    H_bath_occ = np.zeros((len(chain_eb) * n_block_orb, len(chain_eb) * n_block_orb), dtype=complex)
    chain_v_occ[0:n_block_orb] = v0_tilde
    for i in range(0, len(chain_eb) - 1):
        H_bath_occ[i * n_block_orb : (i + 1) * n_block_orb, i * n_block_orb : (i + 1) * n_block_orb] = chain_eb[i]
        H_bath_occ[(i + 1) * n_block_orb : (i + 2) * n_block_orb, i * n_block_orb : (i + 1) * n_block_orb] = chain_v[i]
        H_bath_occ[i * n_block_orb : (i + 1) * n_block_orb, (i + 1) * n_block_orb : (i + 2) * n_block_orb] = np.conj(
            chain_v[i].T
        )
    H_bath_occ[-n_block_orb:, -n_block_orb:] = chain_eb[-1]
    if n < len(ebs):
        chain_eb, chain_v, v0_tilde = tridiagonalize(ebs[n:], vs[n:])
        chain_v_unocc = np.zeros((len(chain_eb) * n_block_orb, n_block_orb), dtype=complex)
        H_bath_unocc = np.zeros((len(chain_eb) * n_block_orb, len(chain_eb) * n_block_orb), dtype=complex)
        chain_v_unocc[0:n_block_orb] = v0_tilde
        for i in range(0, len(chain_eb) - 1):
            H_bath_unocc[i * n_block_orb : (i + 1) * n_block_orb, i * n_block_orb : (i + 1) * n_block_orb] = chain_eb[i]
            H_bath_unocc[
                (i + 1) * n_block_orb : (i + 2) * n_block_orb, i * n_block_orb : (i + 1) * n_block_orb
            ] = chain_v[i]
            H_bath_unocc[
                i * n_block_orb : (i + 1) * n_block_orb, (i + 1) * n_block_orb : (i + 2) * n_block_orb
            ] = np.conj(chain_v[i].T)
        H_bath_unocc[-n_block_orb:, -n_block_orb:] = chain_eb[-1]
    else:
        chain_v_unocc = np.zeros((0 * n_block_orb, n_block_orb), dtype=complex)
        H_bath_unocc = np.zeros((0 * n_block_orb, 0 * n_block_orb), dtype=complex)
    return (H_bath_occ[::-1, ::-1], chain_v_occ[::-1]), (H_bath_unocc, chain_v_unocc)


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
        H[i, i + 1] = tns[i]
        H[i + 1, i] = tns[i]
    matrix_print(H, "Hchain=")
    print("", flush=True)

    w, v = np.linalg.eigh(H)

    n = np.argmin(np.abs(w))
    # n = min(sum(w < 0) - 1, hsize - 1)

    prevtocc = v[:, n - 1 :: -1].transpose()
    prevtunocc = v[:, n:].transpose()
    qocc, vtocc = sp.linalg.qr(prevtocc, check_finite=False, overwrite_a=True)
    qunocc, vtunocc = sp.linalg.qr(prevtunocc, check_finite=False, overwrite_a=True)

    vtot = np.zeros((hsize, hsize), dtype=complex)
    vtot[:, 0:n] = vtocc[::-1, :].transpose()
    vtot[:, n:hsize] = vtunocc.transpose()

    # # Get the tridiagonal terms
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

    Hnew = np.conj(vtot.T) @ H @ vtot
    matrix_print(vtot, "Q before rotating impurity=")
    matrix_print(Hnew, "H before rotating impurity=")
    vtot[:, n - 1 : n + 1] = vtot[:, n - 1 : n + 1] @ np.conj(R.T)
    matrix_print(vtot, "Q after rotating impurity=")

    indices = np.append(np.roll(np.arange(0, n), 1), np.arange(n, hsize))
    idx = np.ix_(indices, indices)
    Hnew = np.conj(vtot.T) @ H @ vtot
    matrix_print(Hnew, "Hhaverkort=")
    Hnew = Hnew[idx]
    matrix_print(Hnew, "Hhaverkort 2=")
    matrix_print(Hnew[block_size:, :block_size], "T haverkort=")
    matrix_print(Hnew[block_size:, block_size:], "H haverkort=")

    assert np.allclose(np.linalg.eigvalsh(H), np.linalg.eigvalsh(Hnew))

    return Hnew[block_size:, :block_size], Hnew[block_size:, block_size:]
