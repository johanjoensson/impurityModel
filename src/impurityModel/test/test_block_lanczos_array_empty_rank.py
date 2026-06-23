"""Regression tests for the distributed array Block Lanczos with empty MPI ranks.

When more MPI ranks exist than the basis distributes onto, some rank can own
*zero* local basis states.  In that case ``H[:, local_cols]`` is an empty
``(global_N, 0)`` slice and ``scipy.tocsr()`` chooses **int32** index arrays
(it picks the minimal index dtype for small/empty matrices), whereas non-empty
ranks get **int64**.  ``block_lanczos_array_cy`` binds the CSR ``indices``/
``indptr`` to ``cdef long[:]`` (int64) memoryviews, so the empty rank used to
raise a buffer-dtype-mismatch ``ValueError`` and leave the routine -- while the
other ranks blocked forever on the in-loop collective ``Allreduce`` -> MPI
deadlock.

These tests force the empty-rank condition deterministically (independent of the
hash distribution) by partitioning a known Hermitian matrix across the
communicator with the last rank(s) deliberately empty, and assert the
distributed run completes and reproduces the dense eigenvalues.
"""

import numpy as np
import pytest

from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import block_lanczos_array, _build_full_T, Reort


def _contiguous_counts_with_empty_last(global_N, size):
    """Partition ``global_N`` contiguous indices over ``size`` ranks, last empty.

    The last rank always gets 0; ``global_N`` is spread over the first
    ``size - 1`` ranks.  If ``size - 1 > global_N`` some leading ranks are empty
    too (an even stronger test).  For ``size == 1`` the single rank owns
    everything (no empty rank, but the test still exercises the kernel).
    """
    if size == 1:
        return [global_N]
    base, rem = divmod(global_N, size - 1)
    counts = [base + (1 if r < rem else 0) for r in range(size - 1)] + [0]
    assert sum(counts) == global_N
    assert counts[-1] == 0
    return counts


def _run_distributed_array_lanczos(H_global, comm):
    """Run the distributed array Block Lanczos on a contiguous row/col partition.

    ``H_global`` is the full Hermitian matrix (identical on every rank).  Each
    rank owns a contiguous block of rows/columns; the last rank owns none.
    Returns the eigenvalues of the resulting block-tridiagonal ``T`` (global,
    identical on all ranks).
    """
    rank = comm.rank
    size = comm.size
    global_N = H_global.shape[0]

    counts = _contiguous_counts_with_empty_last(global_N, size)
    offsets = np.array([sum(counts[:r]) for r in range(size)], dtype=int)
    c0 = offsets[rank]
    c1 = c0 + counts[rank]

    # Local column slice, passed as a scipy CSR matrix.  On the empty rank this
    # is the (global_N, 0) slice that triggers scipy's int32 index dtype.
    import scipy.sparse as sps

    h_local = sps.csr_matrix(H_global[:, c0:c1])

    # Deterministic, globally-normalised starting block (single vector).
    rng = np.random.default_rng(0)
    psi0_full = rng.standard_normal((global_N, 1)) + 1j * rng.standard_normal((global_N, 1))
    psi0_full /= np.linalg.norm(psi0_full)
    psi0_local = np.ascontiguousarray(psi0_full[c0:c1, :], dtype=complex)

    # Full Krylov space (n=1 block, global_N steps) reproduces every eigenvalue.
    alphas, betas, _Q, *_ = block_lanczos_array(
        psi0=psi0_local,
        h_op=h_local,
        converged=lambda a, b, **kw: False,
        max_iter=global_N,
        verbose=False,
        reort=Reort.FULL,
        return_W=False,
        comm=comm,
    )

    T_full = _build_full_T(alphas, betas)
    return np.sort(np.linalg.eigvalsh(T_full))


@pytest.mark.mpi
def test_array_lanczos_empty_rank_tridiagonal():
    """1D tight-binding chain; last rank empty; eigenvalues match the dense ones."""
    comm = MPI.COMM_WORLD
    global_N = 6

    # Tight-binding chain: known spectrum -2 cos(pi (j+1)/(N+1)).
    H = np.zeros((global_N, global_N), dtype=complex)
    for i in range(global_N - 1):
        H[i, i + 1] = -1.0
        H[i + 1, i] = -1.0

    exact = np.sort(np.linalg.eigvalsh(H))
    got = _run_distributed_array_lanczos(H, comm)

    np.testing.assert_allclose(got, exact, atol=1e-8)


@pytest.mark.mpi
def test_array_lanczos_empty_rank_random_hermitian():
    """Dense random Hermitian; last rank empty; eigenvalues match the dense ones."""
    comm = MPI.COMM_WORLD
    global_N = 6

    rng = np.random.default_rng(42)
    A = rng.standard_normal((global_N, global_N)) + 1j * rng.standard_normal((global_N, global_N))
    H = A + A.conj().T  # Hermitian

    exact = np.sort(np.linalg.eigvalsh(H))
    got = _run_distributed_array_lanczos(H, comm)

    np.testing.assert_allclose(got, exact, atol=1e-8)
