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
    """Print a text representation of the block structure of the Hamiltonian.

    Parameters
    ----------
    block_structure : BlockStructure
        The block structure representation to print.
    """
    orb_offset = min(orb for block in block_structure.blocks for orb in block)
    n_orb = sum(len(block) for block in block_structure.blocks)
    mat = np.empty((n_orb, n_orb), dtype=int)
    mat[:, :] = -1
    for block_i, orbs in enumerate(block_structure.blocks):
        idx = np.ix_([orb - orb_offset for orb in orbs], [orb - orb_offset for orb in orbs])
        mat[idx] = block_i
    print("\n".join(" ".join(f"{el:^3d}" if el != -1 else " + " for el in row) for row in mat))


def get_equivalent_orbs(block_structure):
    """Get lists of equivalent orbitals grouped by inequivalent blocks.

    Parameters
    ----------
    block_structure : BlockStructure
        The block structure representation containing blocks and their relationships.

    Returns
    -------
    list of list of int
        A list of lists of equivalent orbital indices for each inequivalent block.
    """
    blocks, ident_blocks, transp_blocks, ph_blocks, phtransp_blocks, ineq_blocks = block_structure
    eq_orbs = [[] for _ in ineq_blocks]
    print(f"{blocks=}")
    for ib, i_eq_orbs in zip(ineq_blocks, eq_orbs):
        tmp = set()
        for jb in ident_blocks[ib]:
            tmp.update(blocks[jb])
        for jb in transp_blocks[ib]:
            tmp.update(blocks[jb])
        for jb in ph_blocks[ib]:
            tmp.update(blocks[jb])
        for jb in phtransp_blocks[ib]:
            tmp.update(blocks[jb])
        i_eq_orbs.extend(sorted(tmp))
    return eq_orbs


def get_equivalent_blocks(block_structure):
    """Get lists of equivalent blocks grouped by inequivalent blocks.

    Parameters
    ----------
    block_structure : BlockStructure
        The block structure representation containing blocks and their relationships.

    Returns
    -------
    list of list of int
        A list of lists of equivalent block indices for each inequivalent block.
    """
    blocks, ident_blocks, transp_blocks, ph_blocks, phtransp_blocks, ineq_blocks = block_structure
    eq_blocks = [[] for _ in ineq_blocks]
    for ib, i_eq_blocks in zip(ineq_blocks, eq_blocks):
        tmp = set()
        for jb in ident_blocks[ib]:
            tmp.add(jb)
        for jb in transp_blocks[ib]:
            tmp.add(jb)
        for jb in ph_blocks[ib]:
            tmp.add(jb)
        for jb in phtransp_blocks[ib]:
            tmp.add(jb)
        i_eq_blocks.extend(sorted(tmp))
    return eq_blocks


def build_block_structure(G, mat=None, tol=1e-6):
    """Analyze a Green's function or a matrix to build a BlockStructure object.

    Parameters
    ----------
    G : np.ndarray, optional
        Green's function array of shape `(n_orb, n_orb)` or `(n_omega, n_orb, n_orb)`.
    mat : np.ndarray, optional
        Hamiltonian or other matrix of shape `(n_orb, n_orb)`.
    tol : float, default 1e-6
        Tolerance threshold for considering matrix elements/Green's function values
        as non-zero or identical.

    Returns
    -------
    BlockStructure
        The built block structure object.
    """
    assert G is not None or mat is not None, "You must supply at least one of G or mat"
    if G is not None and len(G.shape) == 2:
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
    """Determine the indices of inequivalent blocks based on relation mappings.

    Parameters
    ----------
    identical_blocks : list of list of int
        Mapping from a block index to indices of identical blocks.
    transposed_blocks : list of list of int
        Mapping from a block index to indices of transposed blocks.
    particle_hole_blocks : list of list of int
        Mapping from a block index to indices of particle-hole equivalent blocks.
    particle_hole_and_transpose_blocks : list of list of int
        Mapping from a block index to indices of particle-hole and transposed blocks.

    Returns
    -------
    list of int
        List of representative block indices for the inequivalent blocks.
    """
    inequivalent_blocks = []
    for blocks in identical_blocks:
        if len(blocks) == 0:
            continue
        unique = True
        for transpose in transposed_blocks:
            if blocks[0] in transpose:
                unique = False
                break
        for particle_hole in particle_hole_blocks:
            if blocks[0] in particle_hole:
                unique = False
                break
        for particle_hole_and_transpose in particle_hole_and_transpose_blocks:
            if blocks[0] in particle_hole_and_transpose:
                unique = False
                break
        if unique:
            inequivalent_blocks.append(blocks[0])
    return inequivalent_blocks


def get_n_blocks_block_indices_mask_matrix(mat: np.ndarray, tol=1e-6):
    """Determine block components of a matrix using connected components analysis.

    Parameters
    ----------
    mat : np.ndarray
        The input matrix to analyze.
    tol : float, default 1e-6
        Tolerance threshold for connectedness.

    Returns
    -------
    n_components : int
        The number of connected components (blocks) found.
    labels : np.ndarray of int
        An array of component labels for each orbital.
    """
    mask = np.abs(mat) > tol
    return sp.sparse.csgraph.connected_components(mask, directed=False, return_labels=True)


def get_n_blocks_block_indices_mask(G: np.ndarray = None, mat: np.ndarray = None, tol=1e-6):
    """Determine block components of a Green's function and/or a matrix.

    Parameters
    ----------
    G : np.ndarray, optional
        Green's function array of shape `(n_orb, n_orb)` or `(n_omega, n_orb, n_orb)`.
    mat : np.ndarray, optional
        The matrix to analyze.
    tol : float, default 1e-6
        Tolerance threshold for connection.

    Returns
    -------
    n_components : int
        The number of connected components (blocks) found.
    labels : np.ndarray of int
        An array of component labels for each orbital.
    """
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
    """Group orbital indices into blocks based on connectedness in G and/or mat.

    Parameters
    ----------
    G : np.ndarray, optional
        Green's function array.
    mat : np.ndarray, optional
        The matrix.
    tol : float, default 1e-6
        Tolerance threshold.

    Returns
    -------
    list of list of int
        A list where each element is a list of orbital indices belonging to that block.
    """
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
    """Helper function to find identical blocks based on matrix values.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    mat : np.ndarray
        The matrix to check block identity against.
    tol : float
        Tolerance threshold for block comparison.

    Returns
    -------
    list of list of int
        List of lists where the i-th list contains the indices of blocks identical to block i.
    """
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
    """Helper function to find identical blocks based on G and mat values.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    G : np.ndarray
        The Green's function array.
    mat : np.ndarray
        The matrix.
    tol : float
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists where the i-th list contains the indices of blocks identical to block i.
    """
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
    """Find all identical blocks in the block structure.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    G : np.ndarray, optional
        The Green's function array.
    mat : np.ndarray, optional
        The matrix.
    tol : float, default 1e-6
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of identical block indices.
    """
    assert G is not None or mat is not None
    if G is None:
        return _identical_blocks_mat(blocks, mat, tol)
    if len(G.shape) == 2:
        G = G.reshape((1, G.shape[0], G.shape[1]))
    if mat is None:
        mat = np.zeros((G.shape[1], G.shape[1]))
    return _identical_blocks(blocks, G, mat, tol)


def _transposed_blocks_matrix(blocks, mat, tol):
    """Helper function to find blocks that are transposes of each other based on mat.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    mat : np.ndarray
        The matrix.
    tol : float
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of transposed block indices.
    """
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
        transposed_blocks[i] = transposed
    return transposed_blocks


def _transposed_blocks(blocks, G, mat, tol):
    """Helper function to find blocks that are transposes of each other based on G and mat.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    G : np.ndarray
        The Green's function array.
    mat : np.ndarray
        The matrix.
    tol : float
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of transposed block indices.
    """
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
        transposed_blocks[i] = transposed
    return transposed_blocks


def get_transposed_blocks(blocks, G=None, mat=None, tol=1e-6):
    """Find all transposed blocks in the block structure.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    G : np.ndarray, optional
        The Green's function array.
    mat : np.ndarray, optional
        The matrix.
    tol : float, default 1e-6
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of transposed block indices.
    """
    assert G is not None or mat is not None
    if G is None:
        return _transposed_blocks_matrix(blocks, mat, tol)
    if len(G.shape) == 2:
        G = G.reshape((1, G.shape[0], G.shape[1]))
    if mat is None:
        mat = np.zeros((G.shape[1], G.shape[2]))
    return _transposed_blocks(blocks, G, mat, tol)


def _particle_hole_blocks_matrix(blocks, mat, tol):
    """Helper function to find particle-hole equivalent blocks based on mat.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    mat : np.ndarray
        The matrix.
    tol : float
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of particle-hole equivalent block indices.
    """
    particle_hole_blocks = [[] for _ in blocks]
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
        particle_hole_blocks[i] = particle_hole
    return particle_hole_blocks


def _particle_hole_blocks(blocks, G, mat, tol):
    """Helper function to find particle-hole equivalent blocks based on G and mat.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    G : np.ndarray
        The Green's function array.
    mat : np.ndarray
        The matrix.
    tol : float
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of particle-hole equivalent block indices.
    """
    particle_hole_blocks = [[] for _ in blocks]
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
        particle_hole_blocks[i] = particle_hole
    return particle_hole_blocks


def get_particle_hole_blocks(blocks, G=None, mat=None, tol=1e-6):
    """Find all particle-hole equivalent blocks in the block structure.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    G : np.ndarray, optional
        The Green's function array.
    mat : np.ndarray, optional
        The matrix.
    tol : float, default 1e-6
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of particle-hole equivalent block indices.
    """
    assert G is not None or mat is not None
    if G is None:
        return _particle_hole_blocks_matrix(blocks, mat, tol)
    if len(G.shape) == 2:
        G = G.reshape((1, G.shape[0], G.shape[1]))
    if mat is None:
        mat = np.zeros((G.shape[1], G.shape[2]))
    return _particle_hole_blocks(blocks, G, mat, tol)


def _particle_hole_transpose_blocks_matrix(blocks, mat, tol):
    """Helper function to find particle-hole and transposed equivalent blocks based on mat.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    mat : np.ndarray
        The matrix.
    tol : float
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of particle-hole and transposed equivalent block indices.
    """
    patricle_hole_and_transpose_blocks = [[] for _ in blocks]
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
        patricle_hole_and_transpose_blocks[i] = patricle_hole_and_transpose
    return patricle_hole_and_transpose_blocks


def _particle_hole_transpose_blocks(blocks, G, mat, tol):
    """Helper function to find particle-hole and transposed equivalent blocks based on G and mat.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    G : np.ndarray
        The Green's function array.
    mat : np.ndarray
        The matrix.
    tol : float
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of particle-hole and transposed equivalent block indices.
    """
    patricle_hole_and_transpose_blocks = [[] for _ in blocks]
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
        patricle_hole_and_transpose_blocks[i] = patricle_hole_and_transpose
    return patricle_hole_and_transpose_blocks


def get_particle_hole_and_transpose_blocks(blocks, G=None, mat=None, tol=1e-6):
    """Find all particle-hole and transposed equivalent blocks in the block structure.

    Parameters
    ----------
    blocks : list of list of int
        List of orbital indices for each block.
    G : np.ndarray, optional
        The Green's function array.
    mat : np.ndarray, optional
        The matrix.
    tol : float, default 1e-6
        Tolerance threshold.

    Returns
    -------
    list of list of int
        List of lists of particle-hole and transposed equivalent block indices.
    """
    assert G is not None or mat is not None
    if G is None:
        return _particle_hole_transpose_blocks_matrix(blocks, mat, tol)
    if len(G.shape) == 2:
        G = G.reshape((1, G.shape[0], G.shape[1]))
    if mat is None:
        mat = np.zeros((G.shape[1], G.shape[2]))
    return _particle_hole_transpose_blocks(blocks, G, mat, tol)


def build_matrix(inequivalent_parts: list[np.ndarray], block_structure: BlockStructure):
    """Build a full matrix from its unique inequivalent block parts.

    Parameters
    ----------
    inequivalent_parts : list of np.ndarray
        List of arrays representing the unique, inequivalent block parts.
    block_structure : BlockStructure
        The block structure describing the block mappings.

    Returns
    -------
    np.ndarray
        The assembled full matrix.
    """
    assert len(inequivalent_parts) != 0
    assert len(inequivalent_parts[0].shape) == 2
    n_orb = sum(len(block) for block in block_structure.blocks)
    M = np.zeros((n_orb, n_orb), dtype=inequivalent_parts[0].dtype)
    for i, m in enumerate(inequivalent_parts):
        i_block = block_structure.inequivalent_blocks[i]
        for block in block_structure.identical_blocks[i_block]:
            orbs = np.ix_(block_structure.blocks[block], block_structure.blocks[block])
            M[orbs] = m
        for block in block_structure.transposed_blocks[i_block]:
            orbs = np.ix_(block_structure.blocks[block], block_structure.blocks[block])
            M[orbs] = m.T
        for block in block_structure.particle_hole_blocks[i_block]:
            orbs = np.ix_(block_structure.blocks[block], block_structure.blocks[block])
            M[orbs] = m
        for block in block_structure.particle_hole_transposed_blocks[i_block]:
            orbs = np.ix_(block_structure.blocks[block], block_structure.blocks[block])
            M[orbs] = m.T
    return M


def build_greens_function(inequivalent_parts: list[np.ndarray], block_structure: BlockStructure):
    """Build a full Green's function from its unique inequivalent block parts.

    Parameters
    ----------
    inequivalent_parts : list of np.ndarray
        List of arrays representing the unique, inequivalent block parts.
    block_structure : BlockStructure
        The block structure describing the block mappings.

    Returns
    -------
    np.ndarray
        The assembled full Green's function.
    """
    assert len(inequivalent_parts) != 0
    assert len(inequivalent_parts[0].shape) > 2
    n_orb = sum(len(block) for block in block_structure.blocks)
    initial_shape = inequivalent_parts[0].shape[:-2]
    G = np.zeros(initial_shape + (n_orb, n_orb), dtype=inequivalent_parts[0].dtype)
    for i, m in enumerate(inequivalent_parts):
        i_block = block_structure.inequivalent_blocks[i]
        for block in block_structure.identical_blocks[i_block]:
            orbs = np.ix_(block_structure.blocks[block], block_structure.blocks[block])
            G[..., orbs[0], orbs[1]] = m
        for block in block_structure.transposed_blocks[i_block]:
            orbs = np.ix_(block_structure.blocks[block], block_structure.blocks[block])
            G[..., orbs[0], orbs[1]] = np.transpose(tuple(range(len(initial_shape)) + (m.ndim - 2, m.ndim - 1)))
        for block in block_structure.particle_hole_blocks[i_block]:
            orbs = np.ix_(block_structure.blocks[block], block_structure.blocks[block])
            G[..., ::-1, orbs[0], orbs[1]] = m
        for block in block_structure.particle_hole_transposed_blocks[i_block]:
            orbs = np.ix_(block_structure.blocks[block], block_structure.blocks[block])
            G[..., ::-1, orbs[0], orbs[1]] = np.transpose(tuple(range(len(initial_shape)) + (m.ndim - 2, m.ndim - 1)))

    return G
