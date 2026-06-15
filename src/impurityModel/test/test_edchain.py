import numpy as np
import pytest
from impurityModel.ed import edchain
from impurityModel.ed.block_structure import BlockStructure


def test_householder():
    M = np.ones((4, 4))
    M[[1, 1, 2, 2, 3, 3, 3], [1, 3, 2, 3, 1, 2, 3]] = -1
    a1 = edchain.householder_reflector(M[:, 0:2])
    H1 = edchain.householder_matrix(a1)
    H1_exact = np.zeros_like(M)
    H1_exact[[0, 0, 1, 1, 2, 3], [0, 2, 1, 3, 0, 1]] = -1
    H1_exact[[2, 3], [2, 3]] = 1
    H1_exact *= np.sqrt(1 / 2)
    assert np.allclose(H1, H1_exact)


def test_block_qr():
    M = np.ones((4, 4))
    M[[1, 1, 2, 2, 3, 3, 3], [1, 3, 2, 3, 1, 2, 3]] = -1
    Q, R = edchain.block_qr(M, 1)
    R_exact = np.eye(4)
    R_exact[[0, 1, 2, 3], [0, 1, 2, 3]] = [-2, 2, 2, -1]
    R_exact[:3, 3] = 1
    Q_exact = 1 / 2 * np.ones((4, 4))
    Q_exact[[0, 1, 2, 3, 1, 3, 2, 3, 0, 3], [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]] *= -1
    assert np.allclose(Q, Q_exact)
    assert np.allclose(R, R_exact)

    np.random.seed(0)
    M = np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
    Q, R = edchain.block_qr(M, 1)
    Q_np, R_np = np.linalg.qr(M)
    assert np.allclose(Q @ R, M)
    assert np.allclose(np.conj(Q.T) @ Q, np.conj(Q_np).T @ Q_np)

    M = np.ones((150, 150), dtype=float)
    Q, R = edchain.block_qr(M, 1)
    Q_np, R_np = np.linalg.qr(M)
    assert np.allclose(Q @ R, M)
    assert np.allclose(np.conj(Q.T) @ Q, np.conj(Q_np).T @ Q_np)


def test_build_star_geometry_hamiltonian():
    H_imp = np.array([[1.0, 0.5], [0.5, 1.0]])
    vs = np.array([[0.1, 0.2], [0.3, 0.4], [0.1, 0.2], [0.3, 0.4]])
    es = np.array([-1.0, 1.0])
    H_star = edchain.build_star_geometry_hamiltonian(H_imp, vs, es)
    assert H_star.shape == (6, 6)
    assert np.allclose(H_star[:2, :2], H_imp)
    assert np.allclose(H_star[2:, :2], vs.reshape(4, 2))
    assert np.allclose(H_star[:2, 2:], vs.reshape(4, 2).T)


def test_build_block_tridiagonal_hermitian_matrix():
    diagonals = np.array([[[1.0]]])
    offdiagonals = np.array([])
    H = edchain.build_block_tridiagonal_hermitian_matrix(diagonals, offdiagonals)
    assert H.shape == (1, 1)
    assert H[0, 0] == 1.0


def test_separate_orbital_character():
    q = np.eye(2)
    res = edchain.separate_orbital_character(q)
    assert res.shape == (2, 2)
    assert np.allclose(res @ np.conj(res.T), np.eye(2))


from unittest.mock import patch


@patch("impurityModel.ed.edchain.build_block_structure")
def test_build_imp_bath_blocks(mock_build_block_structure):
    # Mock the return value
    from impurityModel.ed.block_structure import BlockStructure

    mock_build_block_structure.return_value = BlockStructure(
        blocks=[[0, 2, 3, 4], [1, 5]],
        identical_blocks=[],
        transposed_blocks=[],
        particle_hole_blocks=[],
        particle_hole_transposed_blocks=[],
        inequivalent_blocks=[0, 1],
    )

    # Build a diagonal matrix
    H = np.diag([1, 2, 3, 4, -1, -2])
    # Connect 0 to 2, 3, 4
    H[0, 2] = H[2, 0] = 0.1
    H[0, 3] = H[3, 0] = 0.1
    H[0, 4] = H[4, 0] = 0.1
    # Connect 1 to 5
    H[1, 5] = H[5, 1] = 0.1

    impurity_indices, occupied_indices, unoccupied_indices, block_structure = edchain.build_imp_bath_blocks(H, n_orb=2)

    assert len(block_structure.blocks) == 2
    # block 0 has 0, 2, 3, 4
    # block 1 has 1, 5
    # sorting matters

    assert set(impurity_indices[0]) == {0}
    assert set(occupied_indices[0]) == {4}  # only -1 is negative
    assert set(unoccupied_indices[0]) == {2, 3}

    assert set(impurity_indices[1]) == {1}
    assert set(occupied_indices[1]) == {5}  # only -2 is negative
    assert set(unoccupied_indices[1]) == set()


def test_linked_double_chain():
    np.random.seed(42)
    H_imp = np.array([[0.0]])
    vs = np.random.rand(4, 1)
    ebs = np.array([-2.0, -1.0, 1.0, 2.0])
    v, hb = edchain.linked_double_chain(H_imp, vs, ebs, verbose=False, extremely_verbose=False)
    assert v.shape[1] == 1
    assert hb.shape == (4, 4)


def test_double_chains():
    np.random.seed(42)
    H_imp = np.array([[0.0]])
    vs = np.random.rand(4, 1)
    ebs = np.array([-2.0, -1.0, 1.0, 2.0])
    v, hb = edchain.double_chains(H_imp, vs, ebs, verbose=False, extremely_verbose=False)
    assert v.shape[1] == 1
    assert hb.shape == (4, 4)
