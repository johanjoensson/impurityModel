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

from impurityModel.ed.ManyBodyUtils import (
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


def _dense_combine(states, Y, slater_weight_min=0.0):
    """Dense-matrix ``Q @ Y`` reference built directly from the union support, replacing
    the retired ``block_combine_sparse`` (which was itself only a thin wrapper around
    ``add_scaled_multi`` -- the very kernel these tests exist to validate)."""
    keys = []
    idx = {}
    for st in states:
        for k in st.to_dict():
            if k not in idx:
                idx[k] = len(keys)
                keys.append(k)
    M = np.zeros((len(keys), len(states)), dtype=complex)
    for j, st in enumerate(states):
        for k, v in st.to_dict().items():
            M[idx[k], j] = v[0]
    out = M @ np.asarray(Y, dtype=complex)
    result = [ManyBodyState({keys[i]: out[i, c] for i in range(len(keys))}) for c in range(out.shape[1])]
    if slater_weight_min > 0:
        for st in result:
            st.prune(slater_weight_min)
    return result


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


def test_store_combine_matches_dense_reference():
    rng = np.random.default_rng(11)
    states = _random_states(rng, 8, 60)
    store = SparseKrylovDense()
    store.append(states)

    Y = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
    ref = _dense_combine(states, Y)
    out = store.combine(Y)
    assert len(out) == 3
    for r, o in zip(ref, out):
        diff = r - o
        assert np.sqrt(diff.norm2()) < 1e-13 * max(np.sqrt(r.norm2()), 1.0)

    # column-range combine: rows of Y address Q[:, a:b]
    Y2 = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
    ref2 = _dense_combine(states[2:5], Y2)
    out2 = store.combine(Y2, 2, 5)
    for r, o in zip(ref2, out2):
        diff = r - o
        assert np.sqrt(diff.norm2()) < 1e-13 * max(np.sqrt(r.norm2()), 1.0)

    # pruning matches the reference semantics
    ref3 = _dense_combine(states, Y, slater_weight_min=1e-1)
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


def test_store_combine_block_matches_combine():
    """``combine_block`` must equal ``combine`` -- same dense product, scattered into ONE
    ``ManyBodyState`` over the union of nonzero rows instead of per-column states."""
    rng = np.random.default_rng(23)
    states = _random_states(rng, 8, 60)
    store = SparseKrylovDense()
    store.append(states)

    Y = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
    ref = store.combine(Y)
    out = store.combine_block(Y)
    assert isinstance(out, ManyBodyState)
    assert out.width == 3
    assert out.to_states() == ref

    # column-range combine: rows of Y address Q[:, a:b]
    Y2 = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
    ref2 = store.combine(Y2, 2, 5)
    out2 = store.combine_block(Y2, 2, 5)
    assert out2.to_states() == ref2


def test_store_combine_block_prunes_by_row_not_by_column():
    """``combine_block``'s ``slaterWeightMin`` is a ROW cutoff (:meth:`prune_rows`: a row
    survives if ANY column exceeds the cutoff) -- NOT the same as ``combine``'s per-column
    ``prune``. A "mixed" row (above cutoff in one column, below it in another) keeps its
    sub-cutoff entries in ``combine_block`` but loses them in ``combine``.

    A random Y (as in ``test_store_combine_block_matches_combine``) essentially never
    produces such a row -- every store-support row there is nonzero in every column, so
    that test cannot see this divergence. Construct the mixed row explicitly instead.
    """
    det0 = _det(1)
    det1 = _det(2)
    col0 = ManyBodyState()
    col0[det0] = 1.0
    col1 = ManyBodyState()
    col1[det1] = 1.0
    store = SparseKrylovDense()
    store.append([col0, col1])

    cutoff = 0.1
    # Q = [[1, 0], [0, 1]] (det0, det1), so C = Y itself: row det0 = Y[0] = [0.5, 0.05]
    # (above cutoff in column 0, below it in column 1); row det1 = Y[1] = [0.05, 0.5]
    # (the mirror image).
    Y = np.array([[0.5, 0.05], [0.05, 0.5]], dtype=complex)

    ref = store.combine(Y, slaterWeightMin=cutoff)
    assert ref[0].to_dict() == {det0: 0.5 + 0j}
    assert ref[1].to_dict() == {det1: 0.5 + 0j}

    out = store.combine_block(Y, slaterWeightMin=cutoff)
    out_col0, out_col1 = out.to_states()
    assert out_col0.to_dict() == {det0: 0.5 + 0j, det1: 0.05 + 0j}
    assert out_col1.to_dict() == {det0: 0.05 + 0j, det1: 0.5 + 0j}


def test_store_slice_block_matches_from_states_getitem():
    """``slice_block`` replaces ``ManyBodyState.from_states(store[a:b])`` at its two
    production call sites -- verify it produces the same block (same support, same
    coefficients) as that reference path, not just the same combine() product."""
    rng = np.random.default_rng(31)
    states = _random_states(rng, 8, 60)
    store = SparseKrylovDense()
    store.append(states)

    ref = ManyBodyState.from_states(store[2:6])
    out = store.slice_block(2, 6)
    assert isinstance(out, ManyBodyState)
    assert out.width == 4
    assert out.to_states() == ref.to_states()

    # whole range, default b
    assert store.slice_block(0).to_states() == ManyBodyState.from_states(store[0:8]).to_states()

    # empty range
    empty = store.slice_block(3, 3)
    assert empty.width == 0 or len(empty.to_states()) == 0


def test_store_slice_block_clamps_b_past_n_cols():
    """``b`` beyond the stored column count must clamp to ``n_cols``, matching Python
    slice semantics (``store[a:b]`` silently narrows) -- not zero-pad the output out to
    the requested width. This is the regime ``_lanczos_step.pxi``'s
    ``len(Q_basis) < p: q_curr = Q_basis.slice_block(0, p)`` guards against: a resumed
    store holding fewer columns than the target block width ``p``."""
    rng = np.random.default_rng(34)
    states = _random_states(rng, 3, 20)
    store = SparseKrylovDense()
    store.append(states)

    ref = ManyBodyState.from_states(store[0:5])
    out = store.slice_block(0, 5)
    assert out.width == ref.width == 3
    assert out.to_states() == ref.to_states()


def test_store_slice_block_rejects_negative_start():
    """Unlike ``self[a:b]`` (Python slice semantics wrap a negative index), ``slice_block``
    takes a plain integer start and does not wrap -- a negative ``a`` must raise rather
    than silently shift which columns are read."""
    rng = np.random.default_rng(35)
    store = SparseKrylovDense()
    store.append(_random_states(rng, 3, 10))
    with pytest.raises(ValueError):
        store.slice_block(-1, 2)


def test_store_slice_block_across_chunk_boundary():
    """Same check, but with the slice straddling a chunk boundary (the initial chunk is
    32 columns; 40 states force at least one more)."""
    rng = np.random.default_rng(32)
    states = _random_states(rng, 40, 100, sparsity=0.5)
    store = SparseKrylovDense()
    for i in range(0, 40, 7):
        store.append(states[i : i + 7])

    ref = ManyBodyState.from_states(store[28:36])
    out = store.slice_block(28, 36)
    assert out.to_states() == ref.to_states()


def test_store_slice_block_used_by_q_slice():
    """``_q_slice`` (shared by TRLM/IRLM) dispatches a ``SparseKrylovDense`` operand
    through ``slice_block`` -- exercise that dispatch directly, not just the primitive."""
    from impurityModel.ed.BlockLanczos import _q_slice

    rng = np.random.default_rng(33)
    states = _random_states(rng, 6, 25)
    store = SparseKrylovDense()
    store.append(states)

    ref = ManyBodyState.from_states(store[1:4])
    out = _q_slice(store, 1, 4)
    assert isinstance(out, ManyBodyState)
    assert out.to_states() == ref.to_states()


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
    return _dense_combine(cols, np.linalg.inv(L).conj().T)


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

    wp_blk = ManyBodyState.from_states([ManyBodyState(dict(s.items())) for s in wp])
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
    wp = ManyBodyState.from_states(_random_states(rng, p, n_dets, sparsity=1.0))

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
