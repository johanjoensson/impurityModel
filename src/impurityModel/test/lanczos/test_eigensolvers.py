import numpy as np
import scipy.sparse
import scipy.sparse.linalg

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

    # Test the sparse (ARPACK) path. Using k=4. Since N=30 and N>20, eigensystem's
    # dense-or-scipy dispatch routes to scipy_eigensystem, not the dense fallback.
    es, vs = eigensystem(op, e_max=100.0, k=4, dense=False)
    assert len(es) >= 1
    assert vs is not None

    # Also explicitly test scipy_eigensystem which we know is missing coverage
    from impurityModel.ed.eigensolvers import scipy_eigensystem

    es_scipy, vs_scipy = scipy_eigensystem(op, e_max=100.0, k=4, return_eigvecs=True)
    assert len(es_scipy) >= 1
    assert vs_scipy is not None


def test_eigensystem_none_e_max_keeps_all_states():
    """e_max=None means no energy cutoff; it must not raise (max(None, ...) TypeError).

    get_eigenvectors calls eigensystem with e_max=max_energy=None on the dense path
    (basis < dense_cutoff). Regression: that used to crash in max(e_max, ...).
    """
    from impurityModel.ed.eigensolvers import eigensystem

    N = 8  # <= 20 forces the dense branch, where e_max only gates the returned mask
    np.random.seed(0)
    H = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    H = H + H.conj().T
    op = scipy.sparse.csr_matrix(H)

    es, vs = eigensystem(op, e_max=None, k=N, dense=True)
    # No cutoff -> the whole computed spectrum comes back, sorted, matching dense eigh.
    ref = np.linalg.eigvalsh(H)
    assert len(es) == N
    assert np.allclose(es, ref)
    assert vs.shape == (N, N)


# ---- ARPACK failure handling in _scipy_eigensystem_solve -------------------------------------
#
# Neither except-branch had ever run in the suite: ARPACK on a well-conditioned test matrix just
# converges. They are forced here by a stub that raises once and then delegates to the real
# eigsh, so what is checked is the recovery -- that the solve still returns the true lowest
# eigenpairs -- and that the branch actually changed what the retry asked for.

N = 60


def _gapped_hermitian():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    return np.diag(np.arange(N, dtype=float)) + 0.05 * (a + a.conj().T)


def _solve_with_one_failure(monkeypatch, make_error):
    from impurityModel.ed import eigensolvers

    h = _gapped_hermitian()
    real_eigsh = eigensolvers.eigsh
    calls = []

    def flaky_eigsh(op, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise make_error(op, kwargs, real_eigsh)
        return real_eigsh(op, **kwargs)

    monkeypatch.setattr(eigensolvers, "eigsh", flaky_eigsh)
    operator = scipy.sparse.linalg.aslinearoperator(h)
    es, vecs = eigensolvers._scipy_eigensystem_solve(operator, e_max=2.5, k=4, v0=None, eigenValueTol=0)
    return h, es, vecs, calls


def _assert_true_lowest_eigenpairs(h, es, vecs):
    exact = np.linalg.eigvalsh(h)
    assert len(es) >= 4
    np.testing.assert_allclose(es, exact[: len(es)], atol=1e-8)
    np.testing.assert_allclose(h @ vecs, vecs * es, atol=1e-6)


def test_arpack_no_convergence_with_nothing_retries_from_a_fresh_start(monkeypatch):
    from scipy.sparse.linalg import ArpackNoConvergence

    h, es, vecs, calls = _solve_with_one_failure(
        monkeypatch, lambda op, kw, _: ArpackNoConvergence("stub", np.array([]), np.empty((N, 0)))
    )
    assert len(calls) >= 2
    # The retry relaxes the requested accuracy from 0 (machine precision) to a finite tolerance.
    assert calls[0]["tol"] == 0 and calls[1]["tol"] > 0
    _assert_true_lowest_eigenpairs(h, es, vecs)


def test_arpack_no_convergence_with_partial_results_warm_starts_from_them(monkeypatch):
    from scipy.sparse.linalg import ArpackNoConvergence

    returned = {}

    def partial(op, kw, real_eigsh):
        e, v = real_eigsh(op, k=1, which="SA")
        returned["v"] = v[:, 0]
        return ArpackNoConvergence("stub", e, v)

    h, es, vecs, calls = _solve_with_one_failure(monkeypatch, partial)
    assert len(calls) >= 2 and calls[1]["tol"] > 0
    # Recovery must not throw the converged part away: the retry starts from it. (A fresh random
    # start would still converge on this matrix, so only the start vector can tell them apart.)
    np.testing.assert_allclose(calls[1]["v0"], returned["v"])
    _assert_true_lowest_eigenpairs(h, es, vecs)


def test_arpack_error_retries_with_a_larger_krylov_space(monkeypatch):
    from scipy.sparse.linalg import ArpackError

    h, es, vecs, calls = _solve_with_one_failure(monkeypatch, lambda op, kw, _: ArpackError(-9999))
    assert len(calls) >= 2
    assert calls[0]["ncv"] is None and calls[1]["ncv"] is not None and calls[1]["ncv"] >= 20
    _assert_true_lowest_eigenpairs(h, es, vecs)
