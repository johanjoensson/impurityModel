"""Regression oracle for the BLAS (zgemm) replacement of matmul_nogil.

matmul_nogil in BlockLanczosArray.pyx computes
    C (m x n, row-major) = beta*C + alpha * opA(A) (m x k) @ opB(B) (k x n)
with op = conjugate-transpose when the trans flag is b'C'. It is now backed by
scipy.linalg.cython_blas.zgemm; this test pins the row/col-major mapping against
numpy for all four trans combinations and a range of alpha/beta, plus the
zero-dimension guards used by the empty-rank MPI path.
"""

import numpy as np
import pytest

from impurityModel.ed.BlockLanczosArray import _matmul_nogil_test

N = ord("N")
C = ord("C")


def _rc(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("transA,transB", [(N, N), (C, N), (N, C), (C, C)])
@pytest.mark.parametrize("alpha", [1.0, -1.0, 2.5 + 1j])
@pytest.mark.parametrize("beta", [0.0, 1.0, 0.5])
def test_zgemm_matmul(transA, transB, alpha, beta):
    rng = np.random.default_rng(0)
    m, n, k = 3, 4, 5
    A = _rc(rng, m, k) if transA == N else _rc(rng, k, m)
    B = _rc(rng, k, n) if transB == N else _rc(rng, n, k)
    opA = A if transA == N else A.conj().T
    opB = B if transB == N else B.conj().T

    C0 = _rc(rng, m, n)
    expected = beta * C0 + alpha * (opA @ opB)
    got = _matmul_nogil_test(A, transA, B, transB, alpha, beta, C0.copy(), m, n, k)
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("transA,transB", [(N, N), (C, N), (N, C), (C, C)])
def test_zgemm_matmul_k_zero_is_beta_scale(transA, transB):
    """k == 0 (no contraction, hit on empty MPI ranks) must give C = beta*C."""
    rng = np.random.default_rng(1)
    m, n = 3, 4
    A = np.zeros((m, 0), dtype=complex) if transA == N else np.zeros((0, m), dtype=complex)
    B = np.zeros((0, n), dtype=complex) if transB == N else np.zeros((n, 0), dtype=complex)
    C0 = _rc(rng, m, n)
    got = _matmul_nogil_test(A, transA, B, transB, 1.0, 0.5, C0.copy(), m, n, 0)
    np.testing.assert_allclose(got, 0.5 * C0, atol=1e-13, rtol=0)


def test_zgemm_matmul_empty_rows():
    """m == 0 (empty local partition) must be a no-op on an empty C."""
    got = _matmul_nogil_test(
        np.zeros((0, 5), dtype=complex),
        N,
        np.zeros((5, 4), dtype=complex),
        N,
        1.0,
        0.0,
        np.zeros((0, 4), dtype=complex),
        0,
        4,
        5,
    )
    assert np.asarray(got).shape == (0, 4)
