"""Unit tests for the row-valued mapping surface of the block state.

Phase 7 of the state unification (``doc/plans/manybodystate_block_unification.md``)
merged the flat_map ``ManyBodyState`` into the block class under one name: ``p == 1``
is an ordinary block, not a special case, and a determinant maps to a **row** (its
``width`` amplitudes) rather than to a single scalar even at width 1.
"""

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    SlaterDeterminant,
    block_inner_cy,
    block_inner_scalar,
)


def _det(i):
    return SlaterDeterminant.from_bytes(bytes([(i >> 8) & 0xFF, i & 0xFF]))


def _amps(n, seed):
    """A determinant -> amplitude dict, built in DESCENDING key order.

    The insertion order is deliberately the reverse of the row order the container
    maintains, so every test below would notice a build that forgot to sort.
    """
    rng = np.random.default_rng(seed)
    keys = sorted({int(k) for k in rng.integers(1, 400, size=n)}, reverse=True)
    return {_det(k): complex(rng.standard_normal(), rng.standard_normal()) for k in keys}


@pytest.fixture
def data():
    return _amps(30, 1001)


@pytest.fixture
def other():
    return _amps(25, 1002)


# --- construction and the mapping surface ----------------------------------


def test_dict_construction_sorts_and_matches_flat_map(data):
    """A dict in arbitrary order builds the sorted support the container maintains."""
    block = ManyBodyState(data)
    state = ManyBodyState(data)
    assert block.width == 1
    assert len(block) == len(state)
    assert list(block.keys()) == sorted(data)
    assert list(block.keys()) == list(state.keys())
    assert all(block[k][0] == state[k][0] for k in data)


def test_lookup_yields_a_row_not_a_scalar(data):
    block = ManyBodyState(data)
    row = block[next(iter(sorted(data)))]
    assert len(row) == 1
    assert np.asarray(row).shape == (1,)


def test_missing_determinant_raises_without_inserting(data):
    """Unlike flat_map's operator[], a failed lookup does not grow the support."""
    block = ManyBodyState(data)
    before = len(block)
    with pytest.raises(KeyError):
        block[_det(9999)]
    assert len(block) == before
    assert block.get(_det(9999)) is None
    assert _det(9999) not in block


def test_iteration_order_is_row_order(data):
    block = ManyBodyState(data)
    state = ManyBodyState(data)
    assert [k for k, _ in block.items()] == list(block.keys())
    assert list(iter(block)) == list(block.keys())
    assert all(row[0] == state[k][0] for k, row in block.items())


def test_to_dict_gives_detached_rows(data):
    block = ManyBodyState(data)
    dumped = block.to_dict()
    key = next(iter(sorted(data)))
    dumped[key][0] = 123.0
    assert block[key][0] == data[key]


def test_setitem_scalar_and_sequence(data):
    block = ManyBodyState(data)
    key = next(iter(sorted(data)))
    block[key] = 2.5 + 1.5j
    assert block[key][0] == 2.5 + 1.5j

    wide = ManyBodyState({k: (v, 2 * v) for k, v in data.items()})
    wide[key] = (1 + 0j, 7 + 0j)
    assert list(wide[key]) == [1 + 0j, 7 + 0j]
    wide[key] = 3 + 0j  # scalars broadcast across the row
    assert list(wide[key]) == [3 + 0j, 3 + 0j]
    with pytest.raises(ValueError):
        wide[key] = (1 + 0j, 2 + 0j, 3 + 0j)


def test_rejected_setitem_does_not_insert_a_row(data):
    """A ValueError from a bad-width assignment must not leave a spurious zero row in
    the support -- it would otherwise ship over the wire on the next redistribute."""
    wide = ManyBodyState({k: (v, 2 * v) for k, v in data.items()})
    new = _det(9999)
    before = len(wide)
    with pytest.raises(ValueError):
        wide[new] = (1 + 0j, 2 + 0j, 3 + 0j)
    assert len(wide) == before
    assert new not in wide


def test_setitem_inserts_a_new_determinant(data):
    block = ManyBodyState(data)
    new = _det(9999)
    block[new] = 4 + 0j
    assert new in block
    assert block[new][0] == 4 + 0j
    assert list(block.keys()) == sorted(list(data) + [new])


def test_erase_and_clear(data):
    block = ManyBodyState(data)
    key = next(iter(sorted(data)))
    assert block.erase(key) is True
    assert block.erase(key) is False
    assert len(block) == len(data) - 1
    block.clear()
    assert block.is_empty() and len(block) == 0


# --- rows are views ---------------------------------------------------------


def test_row_write_aliases_the_block(data):
    block = ManyBodyState(data)
    key = sorted(data)[0]
    row = block[key]
    row[0] = 7.5 + 0.25j
    del row
    assert np.asarray(block)[0, 0] == 7.5 + 0.25j


def test_row_does_not_block_mutation(data):
    """A live row does not pin the state against structural mutation -- unlike
    np.asarray(block), it re-derives its pointer on every access rather than caching
    one, so mutation is safe and a STALE read is what raises."""
    block = ManyBodyState(data)
    row = block[sorted(data)[0]]
    block.prune_rows(0.0)  # allowed even with `row` alive
    block[_det(9999)] = 1 + 0j  # ditto
    with pytest.raises(RuntimeError):
        row[0]  # but reading through the now-stale view raises


def test_row_keeps_its_state_alive(data):
    row = ManyBodyState(data)[sorted(data)[0]]
    assert row[0] == data[sorted(data)[0]]


def test_row_survives_a_reference_cycle(data):
    """A Row caught only in a garbage-collected cycle must not permanently wedge its
    state: tp_clear may null Row._owner before __dealloc__ runs, so the guard must not
    depend on a __dealloc__-time decrement (see the commit fixing this)."""
    import gc

    block = ManyBodyState(data)
    row = block[sorted(data)[0]]

    class _Cycle:
        pass

    cycle = _Cycle()
    cycle.self_ref = cycle
    cycle.row = row
    del row, cycle
    gc.collect()

    block.prune_rows(0.0)  # would stay locked forever under the old export-count design


# --- vector space vs the flat_map oracle ------------------------------------


def test_add_sub_scale_match_flat_map_bit_for_bit(data, other):
    b1, b2 = ManyBodyState(data), ManyBodyState(other)
    s1, s2 = ManyBodyState(data), ManyBodyState(other)
    z = 0.3 - 1.7j

    for block, state in ((b1 + b2, s1 + s2), (b1 - b2, s1 - s2), (b1 * z, s1 * z), (b1 / z, s1 / z), (-b1, -s1)):
        assert list(block.keys()) == list(state.keys())
        assert all(block[k][0] == state[k][0] for k in state.keys())


def test_add_scaled_matches_flat_map_bit_for_bit(data, other):
    block, state = ManyBodyState(data), ManyBodyState(data)
    block.add_scaled(ManyBodyState(other), 0.3 - 1.7j)
    state.add_scaled(ManyBodyState(other), 0.3 - 1.7j)
    assert list(block.keys()) == list(state.keys())
    assert all(block[k][0] == state[k][0] for k in state.keys())


def test_add_scaled_with_itself(data):
    """The merge reads its own rows while building the result."""
    block = ManyBodyState(data)
    block.add_scaled(block, 1.0 + 0j)
    assert all(block[k][0] == 2 * data[k] for k in data)


def test_norms_and_counts_match_flat_map(data):
    block, state = ManyBodyState(data), ManyBodyState(data)
    # norm2 is a plain in-order sum; the flat_map used std::transform_reduce, which is
    # free to reassociate, so these agree to ~1 ULP rather than exactly.
    assert block.norm2() == pytest.approx(state.norm2(), rel=4e-16)
    assert block.max_norm2() == state.max_norm2()
    assert block.count_above(0.5) == state.count_above(0.5)


def test_truncate_matches_flat_map(data):
    block, state = ManyBodyState(data), ManyBodyState(data)
    block.truncate(10)
    state.truncate(10)
    assert list(block.keys()) == list(state.keys())


# --- width semantics --------------------------------------------------------


def test_default_state_is_the_polymorphic_zero(data, other):
    """Width 0 with no rows adopts a width on first use, so sums need no width up front."""
    zero = ManyBodyState()
    assert zero.width == 0 and zero.is_empty()

    narrow = zero + ManyBodyState(data)
    assert narrow.width == 1 and len(narrow) == len(data)

    wide = ManyBodyState.from_states([ManyBodyState(data), ManyBodyState(other)])
    assert (ManyBodyState() + wide).width == 2


def test_empty_dict_is_also_the_polymorphic_zero(data):
    """An empty mapping carries no width information, so it must NOT default to width 1
    -- that would make ManyBodyState({}) + wide_block raise on the width check."""
    empty = ManyBodyState({})
    assert empty.width == 0 and empty.is_empty()

    wide = ManyBodyState({k: (v, 2 * v) for k, v in data.items()})
    grown = empty + wide
    assert grown.width == 2 and len(grown) == len(data)


def test_width_mismatch_raises_python_exception_not_abort(data, other):
    """Every arithmetic operator must translate the C++ width-mismatch exception into a
    Python one (via `except +`) instead of letting it unwind into an abort."""
    narrow = ManyBodyState(data)
    wide = ManyBodyState({k: (v, 2 * v) for k, v in other.items()})
    for op in (
        lambda: narrow + wide,
        lambda: narrow - wide,
        lambda: narrow.add_scaled(wide, 1.0 + 0j),
    ):
        with pytest.raises(ValueError):
            op()


def test_explicit_width_is_checked_not_adopted(data, other):
    wide = ManyBodyState.from_states([ManyBodyState(data), ManyBodyState(other)])
    with pytest.raises(ValueError):
        ManyBodyState(width=3).add_scaled(wide, 1.0 + 0j)


def test_mask_block_rejects_amplitude_assignment(data):
    """A width-0 state that already HAS rows is a key-only mask, not the polymorphic
    zero; assigning amplitudes into it is rejected rather than silently discarding
    them or corrupting its width-0 invariant."""
    mask = ManyBodyState.from_states([ManyBodyState(data)]).key_union(ManyBodyState())
    assert mask.width == 0 and len(mask) > 0
    with pytest.raises(ValueError):
        mask[next(iter(data))] = 5 + 0j


def test_mask_block_rejects_truncate(data):
    """Every row-max is 0 on a width-0 mask, so truncate would otherwise silently keep
    everything regardless of max_rows -- reject instead of pretending it worked."""
    mask = ManyBodyState.from_states([ManyBodyState(data)]).key_union(ManyBodyState())
    with pytest.raises(ValueError):
        mask.truncate(1)


def test_explicit_width_builds_an_empty_block():
    block = ManyBodyState(width=4)
    assert block.width == 4 and len(block) == 0
    assert np.asarray(block).shape == (0, 4)


def test_width_p_columns_are_independent(data):
    """Every per-column result equals the width-1 computation on that column."""
    wide = ManyBodyState({k: (v, 2 * v) for k, v in data.items()})
    assert wide.width == 2
    assert all(wide[k][1] == 2 * data[k] for k in data)

    narrow = ManyBodyState(data)
    assert wide.norm2() == pytest.approx(5 * narrow.norm2(), rel=1e-12)
    assert np.allclose(np.asarray(wide)[:, 0], [data[k] for k in sorted(data)])


def test_dict_width_mismatch_rejected(data):
    with pytest.raises(ValueError):
        ManyBodyState({k: (v, 2 * v) for k, v in data.items()}, width=3)


# --- Phase 1.2 additions: column/select, block_inner_scalar, insert_rows, in-place ops --


def test_column_and_select(data):
    wide = ManyBodyState({k: (v, 2 * v, 3 * v) for k, v in data.items()})
    col1 = wide.column(1)
    assert col1.width == 1
    assert all(col1[k][0] == 2 * data[k] for k in data)

    sel = wide.select([2, 0])
    assert sel.width == 2
    assert all(sel[k][0] == 3 * data[k] and sel[k][1] == data[k] for k in data)

    assert all(wide.select([-1])[k][0] == 3 * data[k] for k in data)
    with pytest.raises(IndexError):
        wide.select([5])


def test_block_inner_scalar_matches_gram_and_flat_map(data, other):
    from impurityModel.ed.ManyBodyUtils import inner as flat_inner

    a, b = ManyBodyState(data), ManyBodyState(other)
    sa, sb = ManyBodyState(data), ManyBodyState(other)
    scalar = block_inner_scalar(a, b)
    mat = block_inner_cy(a, b)
    assert scalar == pytest.approx(mat[0, 0])
    assert scalar == pytest.approx(flat_inner(sa, sb))


def test_block_inner_scalar_requires_width_one(data, other):
    wide = ManyBodyState({k: (v, 2 * v) for k, v in other.items()})
    with pytest.raises(ValueError):
        block_inner_scalar(ManyBodyState(data), wide)


def test_insert_rows_bulk_build_and_overwrite(data):
    base = ManyBodyState(data)
    new_keys = [_det(9001), _det(9002)]
    existing_key = sorted(data)[3]
    base.insert_rows(new_keys + [existing_key], [5 + 0j, 6 + 0j, 999 + 0j])
    assert base[new_keys[0]][0] == 5 + 0j
    assert base[new_keys[1]][0] == 6 + 0j
    assert base[existing_key][0] == 999 + 0j  # last-write-wins, like dict.update
    assert len(base) == len(data) + 2


def test_insert_rows_last_write_wins_within_one_call(data):
    """A determinant repeated WITHIN the same `keys` argument must resolve to the LAST
    entry, matching dict.update -- not merely new-beats-old (the from_unsorted merge
    this is built on keeps the FIRST of equal keys, so the new rows must be handed to
    it in reverse order for a true "last wins")."""
    base = ManyBodyState(data)
    new_key = _det(9001)
    base.insert_rows([new_key, new_key, new_key], [1 + 0j, 2 + 0j, 3 + 0j])
    assert base[new_key][0] == 3 + 0j


def test_insert_rows_empty_is_a_true_no_op():
    """An empty batch must not commit the polymorphic zero to a width -- there is no
    width to adopt it from."""
    zero = ManyBodyState()
    zero.insert_rows([], [])
    assert zero.width == 0 and zero.is_empty()


def test_insert_rows_adopts_width_on_the_polymorphic_zero():
    zero = ManyBodyState()
    zero.insert_rows([_det(1)], [[1 + 0j, 2 + 0j]])
    assert zero.width == 2


def test_in_place_operators_match_flat_map(data, other):
    s1, s2 = ManyBodyState(data), ManyBodyState(other)
    b1, b2 = ManyBodyState(data), ManyBodyState(other)

    b1 += b2
    s1 += s2
    assert all(b1[k][0] == pytest.approx(s1[k][0]) for k in data)

    b1 -= b2
    s1 -= s2
    assert all(b1[k][0] == pytest.approx(s1[k][0]) for k in data)

    z = 2 + 1j
    b1 *= z
    s1 *= z
    assert all(b1[k][0] == pytest.approx(s1[k][0]) for k in data)

    b1 /= z
    s1 /= z
    assert all(b1[k][0] == pytest.approx(s1[k][0]) for k in data)


def test_imul_and_itruediv_do_not_invalidate_a_row(data):
    """*= and /= scale values without moving anything, so a live Row survives them --
    unlike += and -= (add_scaled), which rebuild storage over the union support."""
    block = ManyBodyState(data)
    key = sorted(data)[0]
    row = block[key]

    block *= 2 + 0j
    assert row[0] == 2 * data[key]

    block /= 2 + 0j
    assert row[0] == pytest.approx(data[key])


def test_iadd_invalidates_a_row(data, other):
    block = ManyBodyState(data)
    row = block[sorted(data)[0]]
    block += ManyBodyState(other)
    with pytest.raises(RuntimeError):
        row[0]
