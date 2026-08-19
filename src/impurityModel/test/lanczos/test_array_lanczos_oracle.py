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

from impurityModel.ed.BlockLanczosArray import Reort, _build_full_T, block_lanczos_array

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


def test_array_lanczos_resume_partial_reort_without_w_init():
    """Warm-start resume with ``reort='partial'`` and ``W=None`` (the documented
    Exact-Overlap-Restart path), on the ARRAY kernel -- the array-side analogue of
    ``test_block_lanczos_cy_resume_partial_reort_without_w_init`` in
    ``test_block_lanczos_cy.py``.

    Regression test for an R9 adversarial-review finding: ``seed_w_estimator``'s EOR
    seed (``_block_ops.pxi``) sliced ``Q_basis[a:b]`` unconditionally, which is correct
    for the MBS kernel's ``SparseKrylovDense``/``ManyBodyState``-list representation
    (where a bare slice selects the intended COLUMN range) but wrong for the array
    kernel's plain 2D ``ndarray`` Krylov buffer, where a 1-D slice selects ROWS instead
    -- raising a shape ``ValueError`` on the very next ``block_inner`` call. Fixed by
    branching on ``is_array(Q_basis)`` inside ``seed_w_estimator``.

    A large enough system and long enough continuation that PARTIAL's Paige-Simon
    tracking has to act for real on the EOR-seeded W (not just avoid the crash),
    mirroring the MBS test's own reasoning for using its bigger fixture.
    """
    n, p = 40, 2
    h = _hermitian(n, seed=5)

    alphas1, betas1, Q1, widths1 = block_lanczos_array(
        psi0=_start_block(n, p, seed=2),
        h_op=sps.csr_matrix(h),
        converged=lambda a, b, **kw: False,
        max_iter=3,
        verbose=False,
        reort=Reort.PARTIAL,
        return_widths=True,
        return_W=False,
        comm=None,
    )
    assert len(alphas1) == 3

    # Resume dropping W entirely -- forces the array-path EOR reseed this fix targets --
    # then keep going for many more blocks so PARTIAL's tracking has real work to do
    # against the reconstructed W.
    alphas2, betas2, Q2, widths2 = block_lanczos_array(
        psi0=None,
        h_op=sps.csr_matrix(h),
        converged=lambda a, b, **kw: False,
        max_iter=10,
        verbose=False,
        reort=Reort.PARTIAL,
        alphas=alphas1,
        betas=betas1,
        Q=Q1,
        W=None,
        block_widths_init=widths1,
        return_widths=True,
        return_W=False,
        comm=None,
    )
    assert len(alphas2) == 13

    q2 = np.asarray(Q2)
    gram = q2.conj().T @ q2
    assert np.linalg.norm(gram - np.eye(gram.shape[0])) < _SQRT_EPS

    # 13 blocks (26 vectors) of a 40-dimensional random Hermitian matrix isn't enough to
    # fully resolve the spectrum against the dense reference, so cross-check against a
    # one-shot run of the same total length instead (same approach as the MBS EOR
    # regression test): the resumed and one-shot block-tridiagonal spectra must agree
    # near machine precision, which they can only do if the EOR-reseeded W actually
    # reconstructed the true historical overlaps (a subtly wrong but right-shaped
    # reconstruction would still run, but would desync PARTIAL's trigger decisions from
    # the one-shot run's and drift the spectrum).
    eig_resumed = np.sort(np.linalg.eigvalsh(_build_full_T(alphas2, betas2, block_widths=widths2)))

    alphas_oneshot, betas_oneshot, _Q, widths_oneshot = block_lanczos_array(
        psi0=_start_block(n, p, seed=2),
        h_op=sps.csr_matrix(h),
        converged=lambda a, b, **kw: False,
        max_iter=13,
        verbose=False,
        reort=Reort.PARTIAL,
        return_widths=True,
        return_W=False,
        comm=None,
    )
    eig_oneshot = np.sort(np.linalg.eigvalsh(_build_full_T(alphas_oneshot, betas_oneshot, block_widths=widths_oneshot)))
    np.testing.assert_allclose(eig_resumed, eig_oneshot, atol=1e-8)
