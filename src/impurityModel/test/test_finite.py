import numpy as np
import scipy.sparse
import pytest
from impurityModel.ed.utils import rotate_matrix
from impurityModel.ed.operator_algebra import assert_hermitian, daggerOp
from impurityModel.ed.atomic_physics import dc_MLFT, get_spherical_2_cubic_matrix
from impurityModel.ed.eigensolvers import HermitianOperator
from impurityModel.ed.observables import (
    get_Lz_from_rho_spherical,
    get_occupations_from_rho_spherical,
)
from impurityModel.ed.mpi_comm import get_job_tasks


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
    op = {((0, "c"), (1, "a")): 1.0 + 0.5j}
    dag = daggerOp(op)
    assert len(dag) == 1
    # Note: dagger reverses the tuple and swaps 'c' and 'a', conjugates value
    assert ((1, "c"), (0, "a")) in dag
    assert dag[((1, "c"), (0, "a"))] == 1.0 - 0.5j


def test_assert_hermitian():
    op_hermitian = {((0, "c"), (1, "a")): 1.0j, ((1, "c"), (0, "a")): -1.0j}
    assert_hermitian(op_hermitian)

    op_nonhermitian = {((0, "c"), (1, "a")): 1.0j}
    with pytest.raises(AssertionError):
        assert_hermitian(op_nonhermitian)


def test_dc_MLFT():
    # Only 3d
    res = dc_MLFT(5, 1.0, Fdd=[5.0, 0, 1.0, 0, 1.0])
    assert 2 in res
    assert np.isclose(res[2], (5.0 - 14.0 / 441 * 2.0) * 5 - 1.0)

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


def test_eigensystem():
    from impurityModel.ed.eigensolvers import eigensystem

    N = 30
    # Create a 30x30 matrix
    np.random.seed(42)
    H_dense = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    H_dense = H_dense + H_dense.T.conj()

    diagonal = np.diag(H_dense).real
    diagonal_indices = np.arange(N)

    # triangular part
    H_tri = np.triu(H_dense, k=1)
    triangular_part = scipy.sparse.csr_matrix(H_tri)

    op = HermitianOperator(diagonal, diagonal_indices, triangular_part)
    op.shape = (N, N)  # Set shape if not already done by init
    op.size = N

    # Add dtype attribute if required by scipy LinearOperator
    op.dtype = np.complex128

    # Test sparse solver (should hit TRLM or CIPSI or scipy_eigensystem)
    # Using k=4. Since N=30 and N>20, it will use thick_restarted_block_lanczos (if available) or fallback
    es, vs = eigensystem(op, e_max=100.0, k=4, dense=False)
    assert len(es) >= 1
    assert vs is not None

    # Also explicitly test scipy_eigensystem which we know is missing coverage
    from impurityModel.ed.eigensolvers import scipy_eigensystem

    es_scipy, vs_scipy = scipy_eigensystem(op, e_max=100.0, k=4, return_eigvecs=True)
    assert len(es_scipy) >= 1
    assert vs_scipy is not None
