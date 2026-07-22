"""Unit tests for the SparseKrylovDense column store (Phase 3 commit A of
doc/plans/blocklanczos_partial_perf_memory.md): sequence protocol (materialization
round-trip), combine() vs the flat_map reference, and growth across realloc
boundaries.

Also the memory regressions of ``doc/plans/blocklanczos_reort_memory.md`` Phase 1: the
streaming ``reort`` must never materialize an ``(n_rows x n_cols)`` buffer, and a chunk
retired by row growth must not keep its unwritten columns."""

import tracemalloc

import numpy as np
import pytest

from impurityModel.ed.BlockLanczos import block_combine_sparse
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyBlockState,
    ManyBodyState,
    SlaterDeterminant,
    SparseKrylovDense,
    inner_multi,
    reorth_cgs2_dense,
)


def _det(i):
    return SlaterDeterminant.from_bytes(bytes([(i >> 8) & 0xFF, i & 0xFF]))


def _random_states(rng, n_states, n_dets, sparsity=0.7):
    """Random ManyBodyStates over a shared pool of n_dets determinants."""
    dets = [_det(i + 1) for i in range(n_dets)]
    states = []
    for _ in range(n_states):
        st = ManyBodyState()
        for d in dets:
            if rng.random() < sparsity:
                st[d] = rng.standard_normal() + 1j * rng.standard_normal()
        states.append(st)
    return states


def test_store_roundtrip_bit_exact():
    rng = np.random.default_rng(7)
    states = _random_states(rng, 6, 40)
    store = SparseKrylovDense()
    store.append(states[:2])
    store.append(states[2:5])
    store.append(states[5:])

    assert len(store) == 6
    for i, st in enumerate(states):
        assert store[i] == st  # flat_map equality: identical keys and identical coefficients
    assert store[-1] == states[-1]
    # slice and iteration materialize the same states
    assert store[1:4] == states[1:4]
    assert list(store) == states


def test_store_index_errors():
    store = SparseKrylovDense()
    assert len(store) == 0
    with pytest.raises(IndexError):
        store[0]
    store.append(_random_states(np.random.default_rng(1), 2, 5))
    with pytest.raises(IndexError):
        store[2]
    with pytest.raises(IndexError):
        store[-3]


def test_store_growth_across_realloc():
    """Cross the initial 256-row / 32-col capacities; every column must survive."""
    rng = np.random.default_rng(3)
    states = _random_states(rng, 40, 400, sparsity=0.5)
    store = SparseKrylovDense()
    for i in range(0, 40, 2):
        store.append(states[i : i + 2])
    assert len(store) == 40
    for i in (0, 1, 15, 31, 32, 39):
        assert store[i] == states[i]


def test_store_combine_matches_block_combine_sparse():
    rng = np.random.default_rng(11)
    states = _random_states(rng, 8, 60)
    store = SparseKrylovDense()
    store.append(states)

    Y = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
    ref = block_combine_sparse(states, Y)
    out = store.combine(Y)
    assert len(out) == 3
    for r, o in zip(ref, out):
        diff = r - o
        assert np.sqrt(diff.norm2()) < 1e-13 * max(np.sqrt(r.norm2()), 1.0)

    # column-range combine: rows of Y address Q[:, a:b]
    Y2 = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
    ref2 = block_combine_sparse(states[2:5], Y2)
    out2 = store.combine(Y2, 2, 5)
    for r, o in zip(ref2, out2):
        diff = r - o
        assert np.sqrt(diff.norm2()) < 1e-13 * max(np.sqrt(r.norm2()), 1.0)

    # pruning matches the reference semantics
    ref3 = block_combine_sparse(states, Y, slaterWeightMin=1e-1)
    out3 = store.combine(Y, slaterWeightMin=1e-1)
    for r, o in zip(ref3, out3):
        assert len(r) == len(o)


def test_store_combine_single_vector_and_shape_check():
    rng = np.random.default_rng(5)
    states = _random_states(rng, 4, 20)
    store = SparseKrylovDense()
    store.append(states)
    (out,) = store.combine(np.array([1.0, 0.0, 0.0, 0.0]))
    assert out == states[0]
    with pytest.raises(ValueError):
        store.combine(np.ones((3, 1)))


def test_store_gram_matches_states():
    """Materialized store columns give the same Gram matrix as the original states."""
    rng = np.random.default_rng(13)
    states = _random_states(rng, 5, 30)
    store = SparseKrylovDense()
    store.append(states)
    g_ref = inner_multi(states, states)
    g_store = inner_multi(list(store), list(store))
    np.testing.assert_allclose(g_store, g_ref, atol=1e-14)


def test_store_memory_bytes():
    store = SparseKrylovDense()
    assert store.memory_bytes() == 0  # chunked store allocates lazily on first append
    store.append(_random_states(np.random.default_rng(2), 4, 50))
    assert store.memory_bytes() > 0


def _orthonormal_columns(rng, n_cols, n_dets):
    """A Krylov-like orthonormal column set, so projecting it out is meaningful."""
    cols = _random_states(rng, n_cols, n_dets, sparsity=0.8)
    L = np.linalg.cholesky(inner_multi(cols, cols))
    return block_combine_sparse(cols, np.linalg.inv(L).conj().T)


@pytest.mark.parametrize(
    "cols",
    [
        None,  # FULL: every column
        list(range(5, 22)),  # a contiguous run (the zero-copy view path)
        [0, 1, 2, 9, 10, 11, 30, 31, 32, 33],  # gaps straddling chunk boundaries
        [17],  # a single column
    ],
    ids=["all", "contiguous", "gapped", "single"],
)
def test_store_reort_matches_reorth_cgs2_dense(cols):
    """The streaming per-chunk CGS2 must reproduce the independent dense reference.

    ``reorth_cgs2_dense`` materializes Q on the merged support and runs one zgemm pair;
    ``store.reort`` streams the same projection chunk by chunk. They may differ only by
    accumulation order.
    """
    rng = np.random.default_rng(42)
    n_dets, n_cols, p = 120, 37, 3
    qcols = _orthonormal_columns(rng, n_cols, n_dets)

    store = SparseKrylovDense()
    for i in range(0, n_cols, 5):  # uneven append widths => several chunks
        store.append(qcols[i : i + 5])
    assert len(store) == n_cols

    wp = _random_states(rng, p, n_dets, sparsity=0.9)
    selected = qcols if cols is None else [qcols[c] for c in cols]

    wp_blk = ManyBodyBlockState.from_states([ManyBodyState(dict(s.items())) for s in wp])
    out_new_blk, o_new = store.reort(wp_blk, cols, 2, None)
    out_new = out_new_blk.to_states()
    out_ref, o_ref = reorth_cgs2_dense([ManyBodyState(dict(s.items())) for s in wp], selected, 2, None)

    np.testing.assert_allclose(o_new, o_ref, atol=1e-12)
    for a, b in zip(out_new, out_ref):
        assert np.sqrt((a - b).norm2()) < 1e-12
    # and the projection actually removed the selected directions
    assert np.max(np.abs(inner_multi(selected, out_new))) < 1e-13


@pytest.mark.parametrize("cols", [None, list(range(110))], ids=["full", "partial"])
def test_store_reort_transient_is_not_store_sized(cols):
    """``reort`` must not gather ``Q[:, cols]`` (nor its conjugate transpose).

    The retired implementation held two dense ``(n_rows x len(cols))`` copies at once —
    measured at 1.85x the whole store for a FULL sweep. The streaming version's transient
    is bounded by the residual ``(n_rows x p)`` plus one chunk's columns.
    """
    rng = np.random.default_rng(5)
    n_dets, n_cols, p = 800, 160, 2
    store = SparseKrylovDense()
    store.reserve_rows(n_dets)
    qcols = _random_states(rng, n_cols, n_dets, sparsity=1.0)
    for i in range(0, n_cols, p):
        store.append(qcols[i : i + p])

    buffer_bytes = store.stats()["buffer_bytes"]
    wp = ManyBodyBlockState.from_states(_random_states(rng, p, n_dets, sparsity=1.0))

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        store.reort(wp, cols, 2, None)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    transient = peak - base
    gathered = n_dets * (n_cols if cols is None else len(cols)) * 16
    assert transient < 0.25 * buffer_bytes, f"transient {transient} vs buffer {buffer_bytes}"
    assert transient < 0.25 * gathered, f"transient {transient} vs gathered copy {gathered}"


def test_store_retired_chunk_drops_unwritten_columns():
    """A chunk retired by row growth keeps only the columns it received.

    On a growing determinant support (the Green's-function regime, where ``reserve_rows``
    only sees the seed basis) row growth — not column exhaustion — ends a chunk's life,
    leaving most of its reserved columns unwritten. Those were 95% of all chunk slack
    before the trim. Afterwards only the one still-open chunk may hold unwritten columns,
    which is what the bound below states; without the trim every retired chunk keeps its
    own reservation and this fails by ~2.4x.
    """
    rng = np.random.default_rng(9)
    store = SparseKrylovDense()
    store.reserve_rows(32)  # seed-sized hint: the support will outgrow it repeatedly
    p = 2
    for it in range(60):
        store.append(_random_states(rng, p, 32 + it * 60, sparsity=1.0))

    s = store.stats()
    assert s["n_chunks"] > 3, "test needs several retired chunks to be meaningful"
    # The invariant: a retired chunk reserves no column it did not receive.
    for rows_c, cols_c, used_c in s["chunks"][:-1]:
        assert used_c == cols_c, f"retired chunk ({rows_c}x{cols_c}) kept {cols_c - used_c} dead columns"
    # ... so only the one still-open chunk may hold unwritten columns.
    open_rows, open_cols, open_used = s["chunks"][-1]
    assert s["unused_col_bytes"] == open_rows * (open_cols - open_used) * 16
    # The staircase must beat a flat (n_rows x n_cols) buffer.
    assert s["buffer_bytes"] < s["rows"] * s["cols"] * 16
