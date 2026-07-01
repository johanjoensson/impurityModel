"""Unit tests for ``impurityModel.ed.givens_qr.implicit_qr_step_block``.

The routine applies a single implicit shifted-QR (bulge-chase) step to a block
tridiagonal matrix ``T`` as a unitary similarity ``T <- U^H T U`` and, optionally,
accumulates ``U`` into ``U_k``. The defining properties checked here are:

* the similarity preserves the eigenvalues of ``T``,
* the accumulated ``U_k`` is unitary and consistent with the transform actually
  applied to ``T`` (``T_after == U_k^H T_before U_k``),
* ``T`` stays block tridiagonal (no fill beyond the first off-diagonal block).
"""

import numpy as np
import pytest

from impurityModel.ed.givens_qr import implicit_qr_step_block


def _hermitian_block_tridiagonal(m, p, seed=0):
    """Build a random Hermitian block-tridiagonal matrix with ``m`` blocks of size ``p``."""
    rng = np.random.default_rng(seed)
    N = m * p
    T = np.zeros((N, N), dtype=complex)
    for i in range(m):
        a = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        T[i * p : (i + 1) * p, i * p : (i + 1) * p] = a + a.conj().T
        if i < m - 1:
            b = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
            T[i * p : (i + 1) * p, (i + 1) * p : (i + 2) * p] = b
            T[(i + 1) * p : (i + 2) * p, i * p : (i + 1) * p] = b.conj().T
    return T


@pytest.mark.parametrize("m,p", [(4, 2), (5, 1), (3, 3)])
def test_step_preserves_eigenvalues(m, p):
    T = _hermitian_block_tridiagonal(m, p)
    before = np.sort(np.linalg.eigvalsh(T))
    shift = float(before[0])  # a real shift, as in Lanczos restarts
    T_out, _ = implicit_qr_step_block(T.copy(), p, shift)
    after = np.sort(np.linalg.eigvalsh(T_out))
    np.testing.assert_allclose(before, after, atol=1e-10)


def test_accumulated_U_is_unitary_and_consistent():
    m, p = 4, 2
    T = _hermitian_block_tridiagonal(m, p, seed=3)
    N = m * p
    U0 = np.eye(N, dtype=complex)
    shift = float(np.linalg.eigvalsh(T)[0])

    T_out, U_k = implicit_qr_step_block(T.copy(), p, shift, U0)

    # U_k accumulates the full similarity, so it must be unitary ...
    np.testing.assert_allclose(U_k.conj().T @ U_k, np.eye(N), atol=1e-10)
    # ... and reproduce exactly the transform applied to T.
    np.testing.assert_allclose(U_k.conj().T @ T @ U_k, T_out, atol=1e-10)


def test_step_keeps_block_tridiagonal_structure():
    m, p = 5, 2
    T = _hermitian_block_tridiagonal(m, p, seed=7)
    shift = float(np.linalg.eigvalsh(T)[0])
    T_out, _ = implicit_qr_step_block(T.copy(), p, shift)
    # Blocks farther than one off the diagonal must remain (numerically) zero.
    for i in range(m):
        for j in range(m):
            if abs(i - j) > 1:
                block = T_out[i * p : (i + 1) * p, j * p : (j + 1) * p]
                assert np.max(np.abs(block)) < 1e-9


def test_none_transform_matrix_is_returned_as_none():
    T = _hermitian_block_tridiagonal(3, 2, seed=11)
    _, U_k = implicit_qr_step_block(T.copy(), 2, 0.0, None)
    assert U_k is None
