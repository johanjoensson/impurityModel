"""``mpi_comm.MatvecExchangePlan``: the sparse reduce-scatter behind ``GS_MATVEC_EXCHANGE=graph``.

The distributed array kernels hold ``H`` as ``(global_N, N_local)`` -- every global row, this
rank's columns -- so a block matvec is a reduce-scatter of ``H[rows of d, cols of s] @ V_s`` into
every destination ``d``. The plan reads which destinations are structurally reachable from the CSR
row pointers, learns its sources by an ``Alltoall`` of those flags (never from Hermitian symmetry),
exchanges over a cached distributed graph, and sums in ascending source order.

Pinned here: the neighbourhood matches the block structure of ``H`` and is the transpose of what
the other ranks declare; the accumulated result equals the dense product; every rank runs the same
number of column rounds; a tiny byte budget forces one column per round without changing the
result; an empty rank (``-n 3``) participates with an empty neighbourhood; and the pure chunking
arithmetic, including its C-int guard, is exercised serially.
"""

import numpy as np
import pytest
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed.mpi_comm import (
    MATVEC_EXCHANGE_MAX_BYTES,
    MatvecExchangePlan,
    dest_flags_all,
    dest_flags_from_csr_indptr,
    matvec_column_chunk_width,
)
from impurityModel.test.support.lanczos_fixtures import _contiguous_counts_with_empty_last

GLOBAL_N = 11
WIDTH = 3


# ---------------------------------------------------------------------------
# Serial: the pure arithmetic
# ---------------------------------------------------------------------------


def test_chunk_width_no_neighbours_takes_the_whole_block():
    assert matvec_column_chunk_width(0, WIDTH, 16) == WIDTH


def test_chunk_width_tiny_budget_is_one_column():
    assert matvec_column_chunk_width(1000, WIDTH, 16) == 1


def test_chunk_width_huge_budget_is_one_round():
    assert matvec_column_chunk_width(1000, WIDTH, 2**40) == WIDTH


def test_chunk_width_scales_with_budget():
    # 1000 rows x 16 B = 16 kB per column: a 40 kB budget affords two columns of a width-5 block.
    assert matvec_column_chunk_width(1000, 5, 40_000) == 2


def test_chunk_width_budget_is_clamped():
    # Above the clamp the width is what the clamp affords, not what was asked.
    rows = MATVEC_EXCHANGE_MAX_BYTES // 16
    assert matvec_column_chunk_width(rows, 4, 2**50) == 1


def test_chunk_width_c_int_guard():
    with pytest.raises(ValueError, match="C-int"):
        matvec_column_chunk_width(2**31, 1, 2**50)


def test_dest_flags_from_csr_indptr_reads_row_blocks():
    # 3 ranks owning rows [0,2), [2,5), [5,5): entries in rows 0 and 3 only.
    counts = np.array([2, 3, 0])
    offsets = np.array([0, 2, 5])
    indptr = np.array([0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(dest_flags_from_csr_indptr(indptr, counts, offsets), [True, True, False])
    indptr = np.array([0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(dest_flags_from_csr_indptr(indptr, counts, offsets), [False, True, False])


def test_dest_flags_all_skips_empty_ranks():
    np.testing.assert_array_equal(dest_flags_all([2, 0, 3]), [True, False, True])


def test_plan_on_a_single_rank_has_no_neighbours():
    comm = MPI.COMM_SELF
    plan = MatvecExchangePlan(comm, [GLOBAL_N], [0], dest_flags_all([GLOBAL_N]), WIDTH, 16)
    assert plan.destinations == [] and plan.sources == []
    assert plan.need_max == 0 and plan.w_c == WIDTH
    assert plan.column_rounds(WIDTH) == [(0, WIDTH)]
    d = plan.describe()
    assert d["n_dest"] == 0 and d["n_src"] == 0 and d["rounds"] == 1 and d["send_bytes"] == 0
    X = np.arange(GLOBAL_N * WIDTH, dtype=complex).reshape(GLOBAL_N, WIDTH)
    plan.self_slot(WIDTH)[...] = X
    plan.exchange(WIDTH)
    out = np.empty((GLOBAL_N, WIDTH), dtype=complex)
    plan.accumulate(out, WIDTH)
    np.testing.assert_array_equal(out, X)


# ---------------------------------------------------------------------------
# MPI: the exchange against a dense oracle
# ---------------------------------------------------------------------------


def _balanced_counts(global_N, size):
    return [global_N // size + (1 if r < global_N % size else 0) for r in range(size)]


def _block_tridiagonal(counts, seed=3):
    """Hermitian ``H`` whose (rank-block r, rank-block s) block is nonzero iff ``|r - s| <= 1`` --
    so at three or more non-empty ranks some rank pairs are structurally decoupled."""
    rng = np.random.default_rng(seed)
    n = sum(counts)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    H = np.zeros((n, n), dtype=complex)
    for r in range(len(counts)):
        for s in range(len(counts)):
            if abs(r - s) <= 1:
                blk = rng.standard_normal((counts[r], counts[s])) + 1j * rng.standard_normal((counts[r], counts[s]))
                H[offsets[r] : offsets[r + 1], offsets[s] : offsets[s + 1]] = blk
    return H + H.conj().T


def _run_plan(comm, counts, budget, width=WIDTH):
    """Build the plan for this rank's column slice of ``H`` and run the exchange for a random
    ``X``; return ``(plan, got, want)`` with ``want`` the rows this rank owns of ``H @ X``."""
    rank = comm.rank
    offsets = np.concatenate(([0], np.cumsum(counts)))[:-1].astype(np.int64)
    H = _block_tridiagonal(counts)
    c0, c1 = offsets[rank], offsets[rank] + counts[rank]
    h_local = sps.csr_matrix(H[:, c0:c1])
    indptr = np.ascontiguousarray(h_local.indptr, dtype=np.int64)  # the empty rank's slice is int32
    rng = np.random.default_rng(100 + rank)
    x_local = np.ascontiguousarray(
        rng.standard_normal((counts[rank], width)) + 1j * rng.standard_normal((counts[rank], width))
    )
    x_global = np.concatenate(comm.allgather(x_local), axis=0)
    want = (H @ x_global)[c0:c1]

    plan = MatvecExchangePlan(comm, counts, offsets, dest_flags_from_csr_indptr(indptr, counts, offsets), width, budget)
    got = np.empty((counts[rank], width), dtype=complex)
    for a, b in plan.column_rounds(width):
        w = b - a
        for i, d in enumerate(plan.destinations):
            rows = slice(offsets[d], offsets[d] + counts[d])
            plan.send_slot(i, w)[...] = h_local[rows, :] @ x_local[:, a:b]
        plan.self_slot(w)[...] = h_local[c0:c1, :] @ x_local[:, a:b]
        plan.exchange(w)
        plan.accumulate(got[:, a:b], w)
    return plan, got, want


@pytest.mark.mpi
@pytest.mark.parametrize("partition", ["balanced", "empty_last"])
def test_plan_neighbourhood_matches_the_block_structure(partition):
    comm = MPI.COMM_WORLD
    counts = (
        _balanced_counts(GLOBAL_N, comm.size)
        if partition == "balanced"
        else _contiguous_counts_with_empty_last(GLOBAL_N, comm.size)
    )
    plan, _got, _want = _run_plan(comm, counts, 2**30)
    r = comm.rank
    expected = [s for s in range(comm.size) if s != r and abs(s - r) <= 1 and counts[s] > 0 and counts[r] > 0]
    assert plan.destinations == expected
    assert plan.sources == expected  # H is Hermitian, so the structure is symmetric ...
    # ... but the plan must have learned that from the other ranks, not assumed it:
    all_dests = comm.allgather(plan.destinations)
    assert plan.sources == sorted(s for s in range(comm.size) if r in all_dests[s])
    assert len(set(comm.allgather(len(plan.column_rounds(WIDTH))))) == 1


@pytest.mark.mpi
@pytest.mark.parametrize("partition", ["balanced", "empty_last"])
@pytest.mark.parametrize("budget", [2**30, 16])
def test_plan_exchange_equals_the_dense_product(partition, budget):
    """A 16-byte budget affords a single column per round on any rank with a neighbour, so the
    same result must come out of ``WIDTH`` rounds as out of one."""
    comm = MPI.COMM_WORLD
    counts = (
        _balanced_counts(GLOBAL_N, comm.size)
        if partition == "balanced"
        else _contiguous_counts_with_empty_last(GLOBAL_N, comm.size)
    )
    plan, got, want = _run_plan(comm, counts, budget)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)
    rounds = plan.column_rounds(WIDTH)
    if budget == 16 and plan.need_max > 0:  # some rank has a neighbour (rank-invariant by construction)
        assert plan.w_c == 1 and len(rounds) == WIDTH
    assert len(set(comm.allgather(len(rounds)))) == 1


@pytest.mark.mpi
def test_plan_reuses_the_cached_graph_communicator():
    comm = MPI.COMM_WORLD
    counts = _balanced_counts(GLOBAL_N, comm.size)
    plan1, _g, _w = _run_plan(comm, counts, 2**30)
    plan2, _g, _w = _run_plan(comm, counts, 2**30)
    assert plan2.graph is plan1.graph


@pytest.mark.mpi
def test_plan_width_disagreement_raises_on_every_rank():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs two ranks to disagree")
    counts = _balanced_counts(GLOBAL_N, comm.size)
    offsets = np.concatenate(([0], np.cumsum(counts)))[:-1]
    width = 2 if comm.rank == 0 else 3
    with pytest.raises(RuntimeError, match="width disagrees"):
        MatvecExchangePlan(comm, counts, offsets, dest_flags_all(counts), width, 2**30)
