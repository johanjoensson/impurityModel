"""``KrylovColumnStore``: the TRLM restart continuation's append-only Krylov basis.

The store replaced a ``concat_cols`` of the whole basis per continuation block
(``doc/plans/trlm_krylov_store_quadratic_copies.md``). These tests pin what that change
must not lose: the stored columns are exactly the appended ones, in order; the basis is a
view (no per-block copy), and products against it are bit-identical to products against a
contiguous copy; the buffer survives a same-size restart and grows correctly past its
capacity; and
``project`` is one Gram-Schmidt pass that also yields the pre-projection overlaps.
"""

import numpy as np

from impurityModel.ed.trlm import KrylovColumnStore


def _blocks(rng, n, widths):
    return [rng.standard_normal((n, w)) + 1j * rng.standard_normal((n, w)) for w in widths]


def test_array_basis_is_the_appended_columns_in_order():
    rng = np.random.default_rng(0)
    blocks = _blocks(rng, 50, [5, 3, 3, 2])
    store = KrylovColumnStore()
    store.reset(blocks[0], 13)
    for b in blocks:
        store.append(b)
    np.testing.assert_array_equal(store.columns, np.concatenate(blocks, axis=1))


def test_array_basis_is_a_view_and_its_products_are_bit_identical():
    """The point of the store: no copy per block, and no change in the arithmetic."""
    rng = np.random.default_rng(1)
    blocks = _blocks(rng, 200, [4] * 6)
    store = KrylovColumnStore()
    store.reset(blocks[0], 40)  # spare capacity: the basis is a strided prefix view
    for b in blocks:
        store.append(b)
    Q = store.columns
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
    buf = store.columns.base
    second = _blocks(rng, 30, [3, 4])
    store.reset(second[0], 8)
    for b in second:
        store.append(b)
    assert store.columns.base is buf
    np.testing.assert_array_equal(store.columns, np.concatenate(second, axis=1))


def test_array_store_grows_past_its_capacity_without_losing_columns():
    rng = np.random.default_rng(3)
    blocks = _blocks(rng, 20, [4, 4, 4, 4])
    store = KrylovColumnStore()
    store.reset(blocks[0], 6)
    for b in blocks:
        store.append(b)
    np.testing.assert_array_equal(store.columns, np.concatenate(blocks, axis=1))
    # A restart asking for more than the current buffer holds reallocates it.
    store.reset(blocks[0], 64)
    store.append(blocks[0])
    np.testing.assert_array_equal(store.columns, blocks[0])


def test_appending_does_not_alias_the_caller_block():
    rng = np.random.default_rng(4)
    (b,) = _blocks(rng, 10, [2])
    store = KrylovColumnStore()
    store.reset(b, 2)
    store.append(b)
    b[:] = 0
    assert np.any(store.columns != 0)


def test_project_is_one_cgs_pass_with_the_pre_projection_overlaps():
    """``O = Q^H wp`` before the pass (its last rows are the Lanczos alpha), ``wp - Q O`` after."""
    rng = np.random.default_rng(5)
    Q = np.linalg.qr(rng.standard_normal((8, 5)) + 1j * rng.standard_normal((8, 5)))[0]
    wp = rng.standard_normal((8, 2)) + 1j * rng.standard_normal((8, 2))
    blocks = [Q[:, :3], Q[:, 3:]]
    store = KrylovColumnStore()
    store.reset(blocks[0], 5)
    for blk in blocks:
        store.append(blk)
    out, overlaps = store.project(wp.copy())
    np.testing.assert_allclose(overlaps, np.conj(Q.T) @ wp, atol=1e-14)
    np.testing.assert_allclose(out, wp - Q @ (np.conj(Q.T) @ wp), atol=1e-14)
