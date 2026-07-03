import numpy as np
import scipy.sparse

from impurityModel.ed.eigensolvers import HermitianOperator


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
