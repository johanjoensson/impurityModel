"""``KrylovColumnStore``: the TRLM restart continuation's append-only Krylov basis.

The store replaced a ``concat_cols`` of the whole basis per continuation block
(``doc/plans/trlm_krylov_store_quadratic_copies.md``). These tests pin what that change
must not lose: the stored columns are exactly the appended ones, in order; the basis is a
view (no per-block copy), and products against it are bit-identical to products against a
contiguous copy; the buffer survives a same-size restart and grows correctly past its
capacity.
"""

import numpy as np

from impurityModel.ed.block_view import KrylovColumnStore, concat_cols
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant


def _blocks(rng, n, widths):
    return [rng.standard_normal((n, w)) + 1j * rng.standard_normal((n, w)) for w in widths]


def test_array_basis_is_the_appended_columns_in_order():
    rng = np.random.default_rng(0)
    blocks = _blocks(rng, 50, [5, 3, 3, 2])
    store = KrylovColumnStore()
    store.reset(blocks[0], 13)
    for b in blocks:
        store.append(b)
    np.testing.assert_array_equal(store.basis, np.concatenate(blocks, axis=1))


def test_array_basis_is_a_view_and_its_products_are_bit_identical():
    """The point of the store: no copy per block, and no change in the arithmetic."""
    rng = np.random.default_rng(1)
    blocks = _blocks(rng, 200, [4] * 6)
    store = KrylovColumnStore()
    store.reset(blocks[0], 40)  # spare capacity: the basis is a strided prefix view
    for b in blocks:
        store.append(b)
    Q = store.basis
    assert Q.base is not None and not Q.flags.c_contiguous
    ref = np.concatenate(blocks, axis=1)
    wp = rng.standard_normal((200, 4)) + 1j * rng.standard_normal((200, 4))
    y = rng.standard_normal((24, 3)) + 1j * rng.standard_normal((24, 3))
    np.testing.assert_array_equal(np.conj(Q.T) @ wp, np.conj(ref.T) @ wp)
    np.testing.assert_array_equal(Q @ y, ref @ y)


def test_array_store_reuses_its_buffer_across_same_size_restarts():
    rng = np.random.default_rng(2)
    store = KrylovColumnStore()
    first = _blocks(rng, 30, [4, 4])
    store.reset(first[0], 8)
    for b in first:
        store.append(b)
    buf = store.basis.base
    second = _blocks(rng, 30, [3, 4])
    store.reset(second[0], 8)
    for b in second:
        store.append(b)
    assert store.basis.base is buf
    np.testing.assert_array_equal(store.basis, np.concatenate(second, axis=1))


def test_array_store_grows_past_its_capacity_without_losing_columns():
    rng = np.random.default_rng(3)
    blocks = _blocks(rng, 20, [4, 4, 4, 4])
    store = KrylovColumnStore()
    store.reset(blocks[0], 6)
    for b in blocks:
        store.append(b)
    np.testing.assert_array_equal(store.basis, np.concatenate(blocks, axis=1))
    # A restart asking for more than the current buffer holds reallocates it.
    store.reset(blocks[0], 64)
    store.append(blocks[0])
    np.testing.assert_array_equal(store.basis, blocks[0])


def test_appending_does_not_alias_the_caller_block():
    rng = np.random.default_rng(4)
    (b,) = _blocks(rng, 10, [2])
    store = KrylovColumnStore()
    store.reset(b, 2)
    store.append(b)
    b[:] = 0
    assert np.any(store.basis != 0)


def _det(i):
    raw = bytearray(1)
    raw[0] |= 1 << (7 - i)
    return SlaterDeterminant.from_bytes(bytes(raw))


def test_manybody_basis_matches_concat_cols():
    a = ManyBodyState.from_states([ManyBodyState({_det(0): 1.0, _det(1): 0.5}), ManyBodyState({_det(2): 2.0})])
    b = ManyBodyState.from_states([ManyBodyState({_det(3): 1.0j, _det(0): -1.0})])
    store = KrylovColumnStore()
    store.reset(a, 3)
    store.append(a)
    store.append(b)
    got = store.basis
    ref = concat_cols(a, b)
    assert got.width == ref.width == 3
    assert got == ref
