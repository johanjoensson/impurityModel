import numpy as np
import scipy.sparse
import pytest
from impurityModel.ed.finite import (
    HermitianOperator,
    get_job_tasks,
    rotate_matrix,
    daggerOp,
    assert_hermitian,
    dc_MLFT,
    get_spherical_2_cubic_matrix,
    get_Lz_from_rho_spherical,
    get_occupations_from_rho_spherical,
)

def test_hermitian_operator():
    diagonal = np.array([1.0, 2.0])
    diagonal_indices = np.array([0, 1])
    triangular_part = scipy.sparse.csr_matrix([[0.0, 0.5j], [0.0, 0.0]])
    op = HermitianOperator(diagonal, diagonal_indices, triangular_part)
    
    # matvec
    v = np.array([1.0, 1.0j])
    res = op @ v
    
    # expected:
    # H = [[1, 0.5j], [-0.5j, 2]]
    # H @ [1, 1j] = [1 - 0.5, -0.5j + 2j] = [0.5, 1.5j]
    expected = np.array([0.5, 1.5j])
    assert np.allclose(res, expected)
    
    # matmat
    m = np.array([[1.0, 0], [1.0j, 1]])
    res_mat = op @ m
    expected_mat = np.array([[0.5, 0.5j], [1.5j, 2]])
    assert np.allclose(res_mat, expected_mat)
    
    assert op._adjoint() is op

def test_get_job_tasks():
    tasks = [0, 1, 2, 3, 4]
    assert get_job_tasks(0, 2, tasks) == (0, 1, 4)
    assert get_job_tasks(1, 2, tasks) == (2, 3)

def test_rotate_matrix():
    M = np.array([[1.0, 0], [0, 2.0]])
    T = np.array([[0, 1], [1, 0]])
    M_rot = rotate_matrix(M, T)
    expected = np.array([[2.0, 0], [0, 1.0]])
    assert np.allclose(M_rot, expected)

    T_dict = {0: np.array([[0, 1], [1, 0]])}
    M_rot_dict = rotate_matrix(M, T_dict)
    assert np.allclose(M_rot_dict, expected)

def test_daggerOp():
    op = {((0, 'c'), (1, 'a')): 1.0 + 0.5j}
    dag = daggerOp(op)
    assert len(dag) == 1
    # Note: dagger reverses the tuple and swaps 'c' and 'a', conjugates value
    assert ((1, 'c'), (0, 'a')) in dag
    assert dag[((1, 'c'), (0, 'a'))] == 1.0 - 0.5j

def test_assert_hermitian():
    op_hermitian = {
        ((0, 'c'), (1, 'a')): 1.0j,
        ((1, 'c'), (0, 'a')): -1.0j
    }
    assert_hermitian(op_hermitian)
    
    op_nonhermitian = {
        ((0, 'c'), (1, 'a')): 1.0j
    }
    with pytest.raises(AssertionError):
        assert_hermitian(op_nonhermitian)

def test_dc_MLFT():
    # Only 3d
    res = dc_MLFT(5, 1.0, Fdd=[5.0, 0, 1.0, 0, 1.0])
    assert 2 in res
    assert np.isclose(res[2], (5.0 - 14.0/441 * 2.0) * 5 - 1.0)
    
    # with 2p
    res = dc_MLFT(5, 1.0, Fdd=[5.0, 0, 1.0, 0, 1.0], n2p_i=6, Fpd=[4.0, 0, 0], Gpd=[0, 1.0, 0, 1.0])
    assert 1 in res
    assert 2 in res

def test_get_spherical_2_cubic_matrix():
    u_p = get_spherical_2_cubic_matrix(l=1)
    assert u_p.shape == (3, 3)
    
    u_d = get_spherical_2_cubic_matrix(l=2)
    assert u_d.shape == (5, 5)

def test_rho_spherical_functions():
    rho = np.eye(2)
    N, Ndn, Nup = get_occupations_from_rho_spherical(rho)
    assert N == 2
    assert Ndn == 1
    assert Nup == 1
    
    Lz = get_Lz_from_rho_spherical(rho, l=0)
    assert Lz == 0.0
