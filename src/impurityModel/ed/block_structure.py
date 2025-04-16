from collections import namedtuple
import numpy as np
import scipy as sp

BlockStructure = namedtuple(
    "BlockStructure",
    [
        "blocks",
        "identical_blocks",
        "transposed_blocks",
        "particle_hole_blocks",
        "particle_hole_transposed_blocks",
        "inequivalent_blocks",
    ],
)


def print_block_structure(block_structure):
    n_orb = sum(len(block) for block in block_structure.blocks)
    mat = np.empty((n_orb, n_orb), dtype=int)
    mat[:, :] = -1
    for block_i, orbs in enumerate(block_structure.blocks):
        idx = np.ix_(orbs, orbs)
        mat[idx] = block_i
    print("\n".join(" ".join(f"{el:^3d}" if el != -1 else " + " for el in row) for row in mat))


def get_equivalent_orbs(block_structure):
    (blocks, ident_blocks, transp_blocks, ph_blocks, phtransp_blocks, ineq_blocks) = block_structure
    eq_orbs = [[] for _ in ineq_blocks]
    for ib, i_eq_orbs in zip(ineq_blocks, eq_orbs):
        for jb in ident_blocks[ib]:
            i_eq_orbs.extend(blocks[jb])
        for jb in transp_blocks[ib]:
            i_eq_orbs.extend(blocks[jb])
        for jb in ph_blocks[ib]:
            i_eq_orbs.extend(blocks[jb])
        for jb in phtransp_blocks[ib]:
            i_eq_orbs.extend(blocks[jb])
    return eq_orbs


def get_equivalent_blocks(block_structure):
    (blocks, ident_blocks, transp_blocks, ph_blocks, phtransp_blocks, ineq_blocks) = block_structure
    eq_blocks = [[] for _ in ineq_blocks]
    for ib, i_eq_blocks in zip(ineq_blocks, eq_blocks):
        for jb in ident_blocks[ib]:
            i_eq_blocks.append(jb)
        for jb in transp_blocks[ib]:
            i_eq_blocks.append(jb)
        for jb in ph_blocks[ib]:
            i_eq_blocks.append(jb)
        for jb in phtransp_blocks[ib]:
            i_eq_blocks.append(jb)
    return eq_blocks


def build_block_structure(hyb, ham=None, tol=1e-6):
    blocks = get_block_structure(hyb, ham, tol=tol)
    identical_blocks = get_identical_blocks(blocks, hyb, ham, tol=tol)
    transposed_blocks = get_transposed_blocks(blocks, hyb, ham, tol=tol)
    particle_hole_blocks = get_particle_hole_blocks(blocks, hyb, ham, tol=tol)
    particle_hole_and_transposed_blocks = get_particle_hole_and_transpose_blocks(blocks, hyb, ham, tol=tol)
    inequivalent_blocks = get_inequivalent_blocks(
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transposed_blocks,
    )

    return BlockStructure(
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_and_transposed_blocks,
        inequivalent_blocks,
    )


def get_inequivalent_blocks(
    identical_blocks,
    transposed_blocks,
    particle_hole_blocks,
    particle_hole_and_transpose_blocks,
):
    inequivalent_blocks = []
    for blocks in identical_blocks:
        if len(blocks) == 0:
            continue
        unique = True
        for transpose in transposed_blocks:
            if blocks[0] in transpose[1:]:
                unique = False
                break
        for particle_hole in particle_hole_blocks:
            if blocks[0] in particle_hole[1:]:
                unique = False
                break
        for particle_hole_and_transpose in particle_hole_and_transpose_blocks:
            if blocks[0] in particle_hole_and_transpose[1:]:
                unique = False
                break
        if unique:
            inequivalent_blocks.append(blocks[0])
    return inequivalent_blocks


def get_block_structure(hyb: np.ndarray, hamiltonian=None, tol=1e-6):
    # Extract matrix elements with nonzero hybridization function
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    mask = np.logical_or(np.any(np.abs(hyb) > tol, axis=0), np.abs(hamiltonian) > tol)

    # Use the extracted mask to extract blocks

    n_blocks, block_idxs = sp.sparse.csgraph.connected_components(
        csgraph=sp.sparse.csr_matrix(mask), directed=False, return_labels=True
    )

    blocks = [[] for _ in range(n_blocks)]
    for orb_i, block_i in enumerate(block_idxs):
        blocks[block_i].append(orb_i)

    return blocks


def get_identical_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    identical_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in identical_blocks]):
            continue
        identical = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in identical_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if np.all(np.abs(hyb[idx_i] - hyb[idx_j]) < tol) and np.all(
                np.abs(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]]) < tol
            ):
                identical.append(j)
        identical_blocks[i] = identical
    return identical_blocks


def get_transposed_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    transposed_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if len(block_i) == 1 or np.any([i in b for b in transposed_blocks]):
            continue
        transposed = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i + 1 :]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp + 1
            if any(j in b for b in transposed_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if np.all(np.abs(hyb[idx_i] - np.transpose(hyb[idx_j], (0, 2, 1))) < tol) and np.all(
                np.abs(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]].T) < tol
            ):
                transposed.append(j)
        transposed_blocks.append(transposed)
    return transposed_blocks


def get_particle_hole_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[0], hyb.shape[1]))
    particle_hole_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in particle_hole_blocks]):
            continue
        particle_hole = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in particle_hole_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if (
                np.all(np.abs(np.real(hyb[idx_i] + hyb[idx_j])) < tol)
                and np.all(np.abs(np.imag(hyb[idx_i] - hyb[idx_j])) < tol)
                and np.all(np.abs(np.real(hamiltonian[idx_i[1:]] + hamiltonian[idx_j[1:]])) < tol)
                and np.all(np.abs(np.imag(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]])) < tol)
            ):
                particle_hole.append(j)
        particle_hole_blocks.append(particle_hole)
    return particle_hole_blocks


def get_particle_hole_and_transpose_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    patricle_hole_and_transpose_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in patricle_hole_and_transpose_blocks]):
            continue
        patricle_hole_and_transpose = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in patricle_hole_and_transpose_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if (
                np.all(np.abs(np.real(hyb[idx_i] + np.transpose(hyb[idx_j], (0, 2, 1)))) < tol)
                and np.all(np.abs(np.imag(hyb[idx_i] - np.transpose(hyb[idx_j], (0, 2, 1)))) < tol)
                and np.all(np.abs(np.real(hamiltonian[idx_i[1:]] + hamiltonian[idx_j[1:]].T)) < tol)
                and np.all(np.abs(np.imag(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]].T)) < tol)
            ):
                patricle_hole_and_transpose.append(j)
        patricle_hole_and_transpose_blocks.append(patricle_hole_and_transpose)
    return patricle_hole_and_transpose_blocks
