import numpy as np
import scipy as sp


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
    tns_blocked = tns.reshape((tns.shape[0] // block_size, block_size, block_size))
    v0 = (tns_blocked / np.linalg.norm(tns_blocked, axis=0)).reshape(tns.shape)
    v0, _ = np.linalg.qr(v0)
    H = np.diag(eb)
    N = H.shape[0]
    Q = np.zeros((N, N), dtype=complex)
    q = np.zeros((2, N, block_size), dtype=complex)
    q[1, :, :block_size] = v0
    alphas = np.empty((N // block_size, block_size, block_size), dtype=complex)
    betas = np.empty((N // block_size, block_size, block_size), dtype=complex)

    for i in range(N // block_size):
        wp = H @ q[1]
        alphas[i] = np.conj(q[1].T) @ wp
        wp -= q[1] @ alphas[i]
        if i > 0:
            wp -= q[0] @ np.conj(betas[i - 1].T)
            wp -= Q @ np.conj(Q.T) @ wp
        Q[:, i * block_size : (i + 1) * block_size] = q[1]
        q[0] = q[1]
        q[1], betas[i] = np.linalg.qr(wp)
        if np.linalg.norm(betas[i]) < 1e-8 + np.finfo(float).eps:
            alphas = alphas[: i + 1]
            betas = betas[: i + 1]
            Q = Q[:, : (i + 1) * block_size]
            break

    e_orig = np.linalg.eigvalsh(H)
    e_tri = sp.linalg.eigvalsh_tridiagonal(alphas[:, 0, 0].real, betas[:-1, 0, 0].real)
    assert np.allclose(np.sort(e_orig), np.sort(e_tri)), f"{np.max(np.abs(e_orig-e_tri))=}\n{e_orig=}\n{e_tri=}"
    return alphas, betas, Q


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
    print(f"occupied star bath energies {ebs[:n]}")
    print(f"occupied star bath hopping {vs[:n]}", flush=True)
    chain_eb, chain_v, Q_block = tridiagonalize(ebs[:n], vs[:n])
    print(f"occupied chain bath energies {chain_eb}")
    print(f"occupied chain bath hoppings {chain_v}", flush=True)
    chain_v_occ = np.zeros((len(chain_eb), n_block_orb), dtype=complex)
    H_bath_occ = np.zeros((len(chain_eb), len(chain_eb)), dtype=complex)
    chain_v_occ[0:n_block_orb] = np.linalg.norm(vs[:n], axis=0)
    for i in range(0, len(chain_eb) - 1):
        H_bath_occ[i, i] = chain_eb[i]
        H_bath_occ[i + 1, i] = chain_v[i]
        H_bath_occ[i, i + 1] = np.conj(chain_v[i].T)
    H_bath_occ[-1, -1] = chain_eb[-1]
    if n < len(ebs):
        print(f"unoccupied star bath energies {ebs[n:]}")
        print(f"unoccupied star bath hopping {vs[n:]}", flush=True)
        chain_eb, chain_v, Q_block = tridiagonalize(ebs[n:], vs[n:])
        print(f"unoccupied chain bath energies {chain_eb}")
        print(f"unoccupied chain bath hoppings {chain_v}", flush=True)
        chain_v_unocc = np.zeros((len(chain_eb), n_block_orb), dtype=complex)
        H_bath_unocc = np.zeros((len(chain_eb), len(chain_eb)), dtype=complex)
        chain_v_unocc[0:n_block_orb] = np.linalg.norm(vs[n:], axis=0)
        for i in range(0, len(chain_eb) - 1):
            H_bath_unocc[i, i] = chain_eb[i]
            H_bath_unocc[i + 1, i] = chain_v[i]
            H_bath_unocc[i, i + 1] = np.conj(chain_v[i].T)
        H_bath_unocc[-1, -1] = chain_eb[-1]
    else:
        chain_v_unocc = np.zeros((0 * n_block_orb, n_block_orb), dtype=complex)
        H_bath_unocc = np.zeros((0 * n_block_orb, 0 * n_block_orb), dtype=complex)
    return np.append(chain_v_occ, chain_v_unocc, axis=0), sp.linalg.block_diag(*(H_bath_occ, H_bath_unocc))


def edchains_haverkort(eloc, tns, ens):
    nb = len(ens)
    assert tns.shape[0] == ens.shape[0]
    H = np.zeros((nb + 1, nb + 1), dtype=complex)
    H[0, 0] = eloc
    for i in range(len(ens)):
        H[i + 1, i + 1] = ens[i]
        H[i, i + 1] = tns[i]
        H[i + 1, i] = tns[i]
    w, v = np.linalg.eigh(H)
    n = sum(w < 0)
    # Number of nom. occupied bath states
    prevtocc = v[:, n - 1 :: -1].T
    prevtunocc = v[:, n:].T

    qocc, vtocc = sp.linalg.qr(prevtocc)
    qunocc, vtunocc = sp.linalg.qr(prevtunocc)

    vtot = np.zeros_like(H)
    vtot[:, :n] = vtocc[::-1, :].T
    vtot[:, n:] = vtunocc.T

    tns2 = np.zeros((nb, 1), dtype=complex)
    for i in range(tns2.shape[0]):
        tns2[i] = (np.conj(vtot[:, i].T) * H * vtot[:, i + 1])[0, 0]
        if np.real(tns2[i]) < 0:
            vtot[:, i + 1] = -vtot[:, i + 1]
            tns2[i] = -tns2[i]
    ens2 = np.zeros((len(ens) + 1), dtype=complex)
    for i in range(nb + 1):
        ens2[i] = (np.conj(vtot[:, i].T) * H * vtot[:, i])[0, 0]

    # Get the final transform to extract the impurity orbital (It goes into element n-1)
    cs = vtot[1, n - 1 : n + 1]
    r = np.linalg.norm(cs)
    R = np.empty((2, 2), dtype=complex)
    R[0, 0] = np.conj(cs[0]) / r
    R[1, 0] = -np.conj(cs[1]) / r
    R[0, 1] = cs[1] / r
    R[1, 1] = cs[0] / r

    # R ~ [[Hopping to occupied chain, herm_conj, hopping to unoccupied chain]
    #      [Hopping to unoccupied chain, - herm_conj, hopping to occupied chain]]

    # Write it out
    print(f"{n=}")
    matrix_print(np.round(H, 3), "H")
    vtot[:, n - 1 : n + 1] = vtot[:, n - 1 : n + 1] @ np.conj(R.T)
    Htest = np.conj(vtot.T) @ H @ vtot
    print("ens2")
    print(np.round(ens2, 3))
    matrix_print(np.round(vtot, 3), "vtot")
    matrix_print(np.round(Htest, 3), "Htest")

    # print(np.linalg.inv(H)[0, 0], "H^{-1}")
    # print(np.linalg.inv(Htest)[n, n], "Htest^{-1}")

    return ens2, tns2, R
