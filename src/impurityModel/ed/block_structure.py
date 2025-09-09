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
    orb_offset = min(orb for block in block_structure.blocks for orb in block)
    n_orb = sum(len(block) for block in block_structure.blocks)
    mat = np.empty((n_orb, n_orb), dtype=int)
    mat[:, :] = -1
    for block_i, orbs in enumerate(block_structure.blocks):
        idx = np.ix_([orb - orb_offset for orb in orbs], [orb - orb_offset for orb in orbs])
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


def build_block_structure(G, mat=None, tol=1e-6):
    assert G is not None or mat is not None, "You must supply at least one of G or mat"
    if len(G.shape) == 2:
        G = G.reshape((1, G.shape[0], G.shape[1]))
    blocks = get_blocks(G, mat, tol=tol)
    identical_blocks = get_identical_blocks(blocks, G, mat, tol=tol)
    transposed_blocks = get_transposed_blocks(blocks, G, mat, tol=tol)
    particle_hole_blocks = get_particle_hole_blocks(blocks, G, mat, tol=tol)
    particle_hole_and_transposed_blocks = get_particle_hole_and_transpose_blocks(blocks, G, mat, tol=tol)
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


def get_n_blocks_block_indices_mask_matrix(mat: np.ndarray, tol=1e-6):
    mask = np.abs(mat) > tol
    return sp.sparse.csgraph.connected_components(mask, directed=False, return_labels=True)


def get_n_blocks_block_indices_mask(G: np.ndarray = None, mat: np.ndarray = None, tol=1e-6):
    assert G is not None or mat is not None
    if G is not None:
        if len(G.shape) == 2:
            G = G.reshape((1, G.shape[0], G.shape[1]))
        mask = np.any(np.abs(G) > tol, axis=0)
        if mat is not None:
            mask = np.logical_or(mask, np.abs(mat) > tol)
    else:
        mask = np.abs(mat) > tol

    return sp.sparse.csgraph.connected_components(mask, directed=False, return_labels=True)


def get_blocks(G: np.ndarray = None, mat=None, tol=1e-6):
    assert G is not None or mat is not None, "Must supply at least on of hamiltonian or G"
    if G is not None:
        if len(G.shape) == 2:
            G = G.reshape((1, G.shape[0], G.shape[1]))
        n_blocks, block_idxs = get_n_blocks_block_indices_mask(G)
    else:
        n_blocks, block_idxs = get_n_blocks_block_indices_mask_matrix(mat)

    blocks = [[] for _ in range(n_blocks)]
    for orb_i, block_i in enumerate(block_idxs):
        blocks[block_i].append(orb_i)

    return blocks


def _identical_blocks_mat(blocks, mat, tol):
    identical_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in identical_blocks]):
            continue
        identical = []
        idx_i = np.ix_(block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in identical_blocks):
                continue
            idx_j = np.ix_(block_j, block_j)
            if np.all(np.abs(mat[idx_i] - mat[idx_j]) < tol):
                identical.append(j)
        identical_blocks[i] = identical
    return identical_blocks


def _identical_blocks(blocks, G, mat, tol):
    identical_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in identical_blocks]):
            continue
        identical = []
        idx_i = np.ix_(range(G.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in identical_blocks):
                continue
            idx_j = np.ix_(range(G.shape[0]), block_j, block_j)
            if np.all(np.abs(G[idx_i] - G[idx_j]) < tol) and np.all(np.abs(mat[idx_i[1:]] - mat[idx_j[1:]]) < tol):
                identical.append(j)
        identical_blocks[i] = identical
    return identical_blocks


def get_identical_blocks(blocks, G=None, mat=None, tol=1e-6):
    assert G is not None or mat is not None
    if G is None:
        return _identical_blocks_mat(blocks, mat, tol)
    if len(G.shape) == 2:
        G = G.reshape((1, G.shape[0], G.shape[1]))
    if mat is None:
        mat = np.zeros((G.shape[1], G.shape[1]))
    return _identical_blocks(blocks, G, mat, tol)


def _transposed_blocks_matrix(blocks, mat, tol):
    transposed_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if len(block_i) == 1 or np.any([i in b for b in transposed_blocks]):
            continue
        transposed = []
        idx_i = np.ix_(block_i, block_i)
        for jp, block_j in enumerate(blocks[i + 1 :]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp + 1
            if any(j in b for b in transposed_blocks):
                continue
            idx_j = np.ix_(block_j, block_j)
            if np.all(np.abs(mat[idx_i] - mat[idx_j].T) < tol):
                transposed.append(j)
        transposed_blocks.append(transposed)
    return transposed_blocks


def _transposed_blocks(blocks, G, mat, tol):
    transposed_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if len(block_i) == 1 or np.any([i in b for b in transposed_blocks]):
            continue
        transposed = []
        idx_i = np.ix_(range(G.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i + 1 :]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp + 1
            if any(j in b for b in transposed_blocks):
                continue
            idx_j = np.ix_(range(G.shape[0]), block_j, block_j)
            if np.all(np.abs(G[idx_i] - np.transpose(G[idx_j], (0, 2, 1))) < tol) and np.all(
                np.abs(mat[idx_i[1:]] - mat[idx_j[1:]].T) < tol
            ):
                transposed.append(j)
        transposed_blocks.append(transposed)
    return transposed_blocks


def get_transposed_blocks(blocks, G=None, mat=None, tol=1e-6):
    assert G is not None or mat is not None
    if G is None:
        return _transposed_blocks_matrix(blocks, mat, tol)
    if len(G.shape) == 2:
        G = G.reshape((1, G.shape[0], G.shape[1]))
    if mat is None:
        mat = np.zeros((G.shape[1], G.shape[2]))
    return _transposed_blocks(blocks, G, mat, tol)


def _particle_hole_blocks_matrix(blocks, mat, tol):
    particle_hole_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in particle_hole_blocks]):
            continue
        particle_hole = []
        idx_i = np.ix_(block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in particle_hole_blocks):
                continue
            idx_j = np.ix_(block_j, block_j)
            if np.all(np.abs(np.real(mat[idx_i] + mat[idx_j])) < tol) and np.all(
                np.abs(np.imag(mat[idx_i] - mat[idx_j])) < tol
            ):
                particle_hole.append(j)
        particle_hole_blocks.append(particle_hole)
    return particle_hole_blocks


def _particle_hole_blocks(blocks, G, mat, tol):
    particle_hole_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in particle_hole_blocks]):
            continue
        particle_hole = []
        idx_i = np.ix_(range(G.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in particle_hole_blocks):
                continue
            idx_j = np.ix_(range(G.shape[0]), block_j, block_j)
            if (
                np.all(np.abs(np.real(G[idx_i] + G[idx_j])) < tol)
                and np.all(np.abs(np.imag(G[idx_i] - G[idx_j])) < tol)
                and np.all(np.abs(np.real(mat[idx_i[1:]] + mat[idx_j[1:]])) < tol)
                and np.all(np.abs(np.imag(mat[idx_i[1:]] - mat[idx_j[1:]])) < tol)
            ):
                particle_hole.append(j)
        particle_hole_blocks.append(particle_hole)
    return particle_hole_blocks


def get_particle_hole_blocks(blocks, G=None, mat=None, tol=1e-6):
    assert G is not None or mat is not None
    if G is None:
        return _particle_hole_blocks_matrix(blocks, mat, tol)
    if len(G.shape) == 2:
        G = G.reshape((1, G.shape[0], G.shape[1]))
    if mat is None:
        mat = np.zeros((G.shape[0], G.shape[1]))
    return _particle_hole_blocks(blocks, G, mat, tol)


def _particle_hole_transpose_blocks_matrix(blocks, mat, tol):
    patricle_hole_and_transpose_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in patricle_hole_and_transpose_blocks]):
            continue
        patricle_hole_and_transpose = []
        idx_i = np.ix_(block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in patricle_hole_and_transpose_blocks):
                continue
            idx_j = np.ix_(block_j, block_j)
            if np.all(np.abs(np.real(mat[idx_i] + mat[idx_j].T)) < tol) and np.all(
                np.abs(np.imag(mat[idx_i] - mat[idx_j].T)) < tol
            ):
                patricle_hole_and_transpose.append(j)
        patricle_hole_and_transpose_blocks.append(patricle_hole_and_transpose)
    return patricle_hole_and_transpose_blocks


def _particle_hole_transpose_blocks(blocks, G, mat, tol):
    patricle_hole_and_transpose_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in patricle_hole_and_transpose_blocks]):
            continue
        patricle_hole_and_transpose = []
        idx_i = np.ix_(range(G.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in patricle_hole_and_transpose_blocks):
                continue
            idx_j = np.ix_(range(G.shape[0]), block_j, block_j)
            if (
                np.all(np.abs(np.real(G[idx_i] + np.transpose(G[idx_j], (0, 2, 1)))) < tol)
                and np.all(np.abs(np.imag(G[idx_i] - np.transpose(G[idx_j], (0, 2, 1)))) < tol)
                and np.all(np.abs(np.real(mat[idx_i[1:]] + mat[idx_j[1:]].T)) < tol)
                and np.all(np.abs(np.imag(mat[idx_i[1:]] - mat[idx_j[1:]].T)) < tol)
            ):
                patricle_hole_and_transpose.append(j)
        patricle_hole_and_transpose_blocks.append(patricle_hole_and_transpose)
    return patricle_hole_and_transpose_blocks


def get_particle_hole_and_transpose_blocks(blocks, G=None, mat=None, tol=1e-6):
    assert G is not None or mat is not None
    if G is None:
        return _particle_hole_transpose_blocks_matrix(blocks, mat, tol)
    if len(G.shape) == 2:
        G = G.reshape((1, G.shape[0], G.shape[1]))
    if mat is None:
        mat = np.zeros((G.shape[1], G.shape[2]))
    return _particle_hole_transpose_blocks(blocks, G, mat, tol)
