import numpy as np
import scipy as sp
from impurityModel.ed.lanczos import eigsh
from impurityModel.ed.block_structure import build_block_structure, BlockStructure


def matrix_print(m, label=None):
    if label is not None:
        print(label)
    m_is_complex = np.any(np.abs(m.imag) > 1e-6)
    if m_is_complex:
        print(
            "\n".join(
                [
                    " ".join([f"{np.real(el): .6f} {np.imag(el):+.6f}j" for el in row])
                    for row in m
                ]
            )
        )
    else:
        print(
            "\n".join(
                [
                    " ".join([f"{np.real(el): .6f}" for el in row])
                    for row in m
                ]
            )
        )


def build_imp_bath_blocks(H, n_orb):
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


def build_H_bath_v(
    H_dft,
    ebs_star,
    vs_star,
    bath_geometry,
    block_structure,
    verbose,
):

    H_baths = []
    vs = []
    if bath_geometry == "chain":
        for v, ebs in zip(vs_star, ebs_star):
            if len(ebs) <= 1:
                H_baths.append(np.diag(ebs))
                vs.append(v)
                continue
            (H_bath_occ, v_occ), (H_bath_unocc, v_unocc) = edchains(v, ebs)
            H_baths.append(sp.linalg.block_diag(H_bath_occ, H_bath_unocc))
            vs.append(np.vstack((v_occ, v_unocc)))
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
            if len(ebs) == 0:
                H_baths.append(np.array([], dtype=complex))
                vs.append(v)
                continue
            if len(ebs) == 1:
                H_baths.append(np.diag(ebs))
                vs.append(v)
                continue

            ebs_chain, tns_chain, v0 = tridiagonalize(ebs, v)
            block_ix = block_structure.inequivalent_blocks[i_b]
            block_orbs = block_structure.blocks[block_ix]
            b_ix = np.ix_(block_orbs, block_orbs)
            vh, Hh = haverkort_chain(H_dft[b_ix], np.append(v0, tns_chain[:-1]), ebs_chain)
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

def build_full_bath(H_bath_inequiv, v_inequiv, block_structure: BlockStructure):
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
            H_bath_unocc[(i + 1) * n_block_orb : (i + 2) * n_block_orb, i * n_block_orb : (i + 1) * n_block_orb] = (
                chain_v[i]
            )
            H_bath_unocc[i * n_block_orb : (i + 1) * n_block_orb, (i + 1) * n_block_orb : (i + 2) * n_block_orb] = (
                np.conj(chain_v[i].T)
            )
        H_bath_unocc[-n_block_orb:, -n_block_orb:] = chain_eb[-1]
    else:
        chain_v_unocc = np.zeros((0 * n_block_orb, n_block_orb), dtype=complex)
        H_bath_unocc = np.zeros((0 * n_block_orb, 0 * n_block_orb), dtype=complex)
    return (H_bath_occ[::-1, ::-1].copy(), chain_v_occ[::-1].copy()), (H_bath_unocc, chain_v_unocc)


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

    Hnew = np.conj(vtot.T) @ H @ vtot
    vtot[:, n - 1 : n + 1] = vtot[:, n - 1 : n + 1] @ np.conj(R.T)

    indices = np.append(np.roll(np.arange(0, n), 1), np.arange(n, hsize))
    idx = np.ix_(indices, indices)
    Hnew = np.conj(vtot.T) @ H @ vtot
    Hnew = Hnew[idx]

    assert np.allclose(np.linalg.eigvalsh(H), np.linalg.eigvalsh(Hnew))

    return Hnew[block_size:, :block_size].copy(), Hnew[block_size:, block_size:].copy()
