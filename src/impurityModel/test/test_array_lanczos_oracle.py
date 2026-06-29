"""Regression oracle for the array Block Lanczos kernel (BLAS-acceleration plan Item 0).

`test_array_lanczos_matches_dense` (eigenvalues to 1e-8) and
`test_array_lanczos_orthonormality` (||Q Q^dagger - I|| < sqrt(eps)) — the safety net
for any kernel change. Serial + MPI (the MPI eigenvalue path with an empty rank also
lives in test_block_lanczos_array_empty_rank.py).
"""

import numpy as np
import pytest
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import block_lanczos_array, _build_full_T, Reort

_SQRT_EPS = np.sqrt(np.finfo(float).eps)


def _hermitian(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


def _start_block(n, p, seed=1):
    rng = np.random.default_rng(seed)
    psi0 = rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))
    q, _ = np.linalg.qr(psi0)
    return np.ascontiguousarray(q[:, :p], dtype=complex)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_array_lanczos_matches_dense(p):
    """Full Krylov reproduces the dense eigenvalues to 1e-8 (serial)."""
    n = 12
    h = _hermitian(n, seed=p)
    n_blocks = -(-n // p)  # ceil: full Krylov
    alphas, betas, _q = block_lanczos_array(
        psi0=_start_block(n, p),
        h_op=sps.csr_matrix(h),
        converged=lambda a, b, **kw: False,
        max_iter=n_blocks,
        verbose=False,
        reort=Reort.FULL,
        return_W=False,
        comm=None,
    )
    eig = np.sort(np.linalg.eigvalsh(_build_full_T(alphas, betas)))
    exact = np.sort(np.linalg.eigvalsh(h))
    np.testing.assert_allclose(eig[: len(exact)], exact, atol=1e-8)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_array_lanczos_orthonormality(p):
    """The Lanczos basis is orthonormal: ||Q^dagger Q - I|| < sqrt(eps) (serial).

    Q is (local_N, n_vectors): the vectors are columns, so orthonormality is the
    column Gram Q^dagger Q (summed over rows / ranks).
    """
    n = 12
    h = _hermitian(n, seed=10 + p)
    _alphas, _betas, q = block_lanczos_array(
        psi0=_start_block(n, p),
        h_op=sps.csr_matrix(h),
        converged=lambda a, b, **kw: False,
        max_iter=-(-n // p),
        verbose=False,
        reort=Reort.FULL,
        return_W=False,
        comm=None,
    )
    q_mat = np.asarray(q)  # (n, n_vectors)
    gram = q_mat.conj().T @ q_mat
    assert np.linalg.norm(gram - np.eye(gram.shape[0])) < _SQRT_EPS


@pytest.mark.mpi
def test_array_lanczos_orthonormality_mpi():
    """Distributed Lanczos basis is orthonormal across ranks (row-block partition)."""
    comm = MPI.COMM_WORLD
    n, p = 8, 2
    h = _hermitian(n, seed=7)
    psi0_full = _start_block(n, p, seed=3)

    counts = [n // comm.size + (1 if r < n % comm.size else 0) for r in range(comm.size)]
    c0 = sum(counts[: comm.rank])
    c1 = c0 + counts[comm.rank]
    h_local = sps.csr_matrix(h[:, c0:c1])
    psi0_local = np.ascontiguousarray(psi0_full[c0:c1, :], dtype=complex)

    _alphas, _betas, q = block_lanczos_array(
        psi0=psi0_local,
        h_op=h_local,
        converged=lambda a, b, **kw: False,
        max_iter=-(-n // p),
        verbose=False,
        reort=Reort.FULL,
        return_W=False,
        comm=comm,
    )
    q_local = np.asarray(q)  # (local_N, n_vectors); each column a vector's local rows
    # Column Gram summed over the row-block partition reconstructs the full Q^dagger Q.
    gram_local = q_local.conj().T @ q_local
    total = np.zeros_like(gram_local)
    comm.Allreduce(np.ascontiguousarray(gram_local), total, op=MPI.SUM)
    assert np.linalg.norm(total - np.eye(total.shape[0])) < _SQRT_EPS
