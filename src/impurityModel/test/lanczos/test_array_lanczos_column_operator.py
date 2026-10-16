"""The array block Lanczos on a ``LocalColumnOperator`` against the same matrix as a CSR.

The frozen-basis Green's function hands the kernel this rank's columns as the build's column blocks
(``basis_transcription.LocalColumnOperator``), and the kernel multiplies by scattering columns into
the exchange buffers instead of running CSR rows. Both sum every output row in ascending column
order, so the recurrences must agree to rounding: every alpha and beta, at every rank count, with
a balanced and an empty-last partition, under the graph exchange (one column per round under a
16-byte budget too) and under ``reduce`` (which falls back to the CSR).

``H`` is complex Hermitian with complex off-diagonals and the seeds are complex, so an operator
that applied ``H^T`` or ``conj(H)`` instead of ``H`` -- a transposed or unconjugated scatter --
changes the alphas: a real symmetric fixture would not see it.
"""

import numpy as np
import pytest
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed.basis_transcription import LocalColumnOperator
from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array
from impurityModel.test.support.lanczos_fixtures import _contiguous_counts_with_empty_last

GLOBAL_N = 23


@pytest.fixture(autouse=True)
def _knobs_unset(monkeypatch):
    monkeypatch.delenv("GS_MATVEC_EXCHANGE", raising=False)
    monkeypatch.delenv("GS_MATVEC_EXCHANGE_BYTES", raising=False)


def _hermitian(seed=3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((GLOBAL_N, GLOBAL_N)) + 1j * rng.standard_normal((GLOBAL_N, GLOBAL_N))
    A[rng.random((GLOBAL_N, GLOBAL_N)) < 0.7] = 0.0  # sparse enough that some rank pairs decouple
    return A + A.conj().T


def _column_operator(h_local_csc, columns_per_segment, row_dtype=np.int32):
    """A ``LocalColumnOperator`` over the same columns, split into blocks of a few columns."""
    n_cols = h_local_csc.shape[1]
    segments = []
    for c0 in range(0, n_cols, columns_per_segment):
        c1 = min(c0 + columns_per_segment, n_cols)
        block = h_local_csc[:, c0:c1].tocsc()
        segments.append((c0, block.indptr.astype(np.int64), block.indices.astype(row_dtype), block.data.copy()))
    return LocalColumnOperator(h_local_csc.shape, segments)


def _local(comm, partition, width):
    if partition == "empty_last":
        counts = _contiguous_counts_with_empty_last(GLOBAL_N, comm.size)
    else:
        counts = [GLOBAL_N // comm.size + (1 if r < GLOBAL_N % comm.size else 0) for r in range(comm.size)]
    c0 = sum(counts[: comm.rank])
    c1 = c0 + counts[comm.rank]
    H = _hermitian()
    rng = np.random.default_rng(0)
    psi0 = rng.standard_normal((GLOBAL_N, width)) + 1j * rng.standard_normal((GLOBAL_N, width))
    psi0 = np.linalg.qr(psi0)[0]
    return sps.csc_array(H[:, c0:c1]), np.ascontiguousarray(psi0[c0:c1]), H


def _run(h_op, psi0_local, comm, max_iter):
    alphas, betas, *_ = block_lanczos_array(
        psi0=psi0_local,
        h_op=h_op,
        converged=lambda a, b, **kw: False,
        max_iter=max_iter,
        verbose=False,
        reort=Reort.FULL,
        return_W=False,
        comm=comm,
    )
    return np.asarray(alphas), np.asarray(betas)


@pytest.mark.mpi
@pytest.mark.parametrize("row_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("partition", ["balanced", "empty_last"])
@pytest.mark.parametrize("exchange", ["graph", "graph_one_column", "reduce"])
@pytest.mark.parametrize("width", [1, 3])
def test_the_column_operator_runs_the_same_recurrence_as_the_csr(monkeypatch, row_dtype, partition, exchange, width):
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("GS_MATVEC_EXCHANGE", "reduce" if exchange == "reduce" else "graph")
    if exchange == "graph_one_column":
        monkeypatch.setenv("GS_MATVEC_EXCHANGE_BYTES", "16")
    h_csc, psi0_local, _ = _local(comm, partition, width)
    max_iter = 5
    want_a, want_b = _run(h_csc.tocsr(), psi0_local, comm, max_iter)
    got_a, got_b = _run(_column_operator(h_csc, 2, row_dtype), psi0_local, comm, max_iter)
    np.testing.assert_allclose(got_a, want_a, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(got_b, want_b, rtol=1e-13, atol=1e-13)


def test_the_column_operator_is_applied_as_h_not_its_transpose():
    """Serial: the recurrence on the column operator matches ``H``'s and not ``H^T``'s. ``H^T`` is
    ``conj(H)`` for a Hermitian ``H``, the result of a scatter that swapped rows and columns."""
    h_csc, psi0, H = _local(MPI.COMM_SELF, "balanced", 2)
    got_a, _ = _run(_column_operator(h_csc, 3), psi0, None, 4)
    want_a, _ = _run(sps.csr_array(H), psi0, None, 4)
    wrong_a, _ = _run(sps.csr_array(H.T), psi0, None, 4)
    np.testing.assert_allclose(got_a, want_a, rtol=1e-13, atol=1e-13)
    assert not np.allclose(wrong_a, want_a), "premise: H and its transpose give different alphas"


@pytest.mark.mpi
def test_a_bound_operator_runs_again_with_the_same_result():
    """The GF path reruns the kernel on the same operator when it doubles its block budget; the
    second run sees slot indices, not global rows, and must still reproduce the first."""
    comm = MPI.COMM_WORLD
    h_csc, psi0_local, _ = _local(comm, "balanced", 2)
    column_op = _column_operator(h_csc, 2)
    first_a, first_b = _run(column_op, psi0_local, comm, 4)
    second_a, second_b = _run(column_op, psi0_local, comm, 4)
    assert np.array_equal(first_a, second_a) and np.array_equal(first_b, second_b)
