"""Unit tests for the SparseKrylovDense column store (Phase 3 commit A of
doc/plans/blocklanczos_partial_perf_memory.md): sequence protocol (materialization
round-trip), combine() vs the flat_map reference, and growth across realloc
boundaries."""

import numpy as np
import pytest

from impurityModel.ed.BlockLanczos import block_combine_sparse
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant, SparseKrylovDense, inner_multi


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
