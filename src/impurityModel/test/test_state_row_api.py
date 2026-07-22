"""Unit tests for the row-valued mapping surface of the block state.

Step 1 of the state unification (``doc/plans/manybodystate_block_unification.md``):
``ManyBodyBlockState`` grows the container surface that the flat_map ``ManyBodyState``
provides -- determinant lookup, iteration, the vector space -- with a determinant now
mapping to a **row** (its ``width`` amplitudes) rather than to a single scalar.

While both containers exist, the flat_map class is the oracle: at width 1 every operation
here must agree with it, bit-for-bit where no summation order changed. Those comparisons go
away with the flat_map class; the width-1 vs width-p equivalences below outlive it.
"""

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import ManyBodyBlockState, ManyBodyState, SlaterDeterminant


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
    block = ManyBodyBlockState(data)
    state = ManyBodyState(data)
    assert block.width == 1
    assert len(block) == len(state)
    assert list(block.keys()) == sorted(data)
    assert list(block.keys()) == list(state.keys())
    assert all(block[k][0] == state[k] for k in data)


def test_lookup_yields_a_row_not_a_scalar(data):
    block = ManyBodyBlockState(data)
    row = block[next(iter(sorted(data)))]
    assert len(row) == 1
    assert np.asarray(row).shape == (1,)


def test_missing_determinant_raises_without_inserting(data):
    """Unlike flat_map's operator[], a failed lookup does not grow the support."""
    block = ManyBodyBlockState(data)
    before = len(block)
    with pytest.raises(KeyError):
        block[_det(9999)]
    assert len(block) == before
    assert block.get(_det(9999)) is None
    assert _det(9999) not in block


def test_iteration_order_is_row_order(data):
    block = ManyBodyBlockState(data)
    state = ManyBodyState(data)
    assert [k for k, _ in block.items()] == list(block.keys())
    assert list(iter(block)) == list(block.keys())
    assert all(row[0] == state[k] for k, row in block.items())


def test_to_dict_gives_detached_rows(data):
    block = ManyBodyBlockState(data)
    dumped = block.to_dict()
    key = next(iter(sorted(data)))
    dumped[key][0] = 123.0
    assert block[key][0] == data[key]


def test_setitem_scalar_and_sequence(data):
    block = ManyBodyBlockState(data)
    key = next(iter(sorted(data)))
    block[key] = 2.5 + 1.5j
    assert block[key][0] == 2.5 + 1.5j

    wide = ManyBodyBlockState({k: (v, 2 * v) for k, v in data.items()})
    wide[key] = (1 + 0j, 7 + 0j)
    assert list(wide[key]) == [1 + 0j, 7 + 0j]
    wide[key] = 3 + 0j  # scalars broadcast across the row
    assert list(wide[key]) == [3 + 0j, 3 + 0j]
    with pytest.raises(ValueError):
        wide[key] = (1 + 0j, 2 + 0j, 3 + 0j)


def test_setitem_inserts_a_new_determinant(data):
    block = ManyBodyBlockState(data)
    new = _det(9999)
    block[new] = 4 + 0j
    assert new in block
    assert block[new][0] == 4 + 0j
    assert list(block.keys()) == sorted(list(data) + [new])


def test_erase_and_clear(data):
    block = ManyBodyBlockState(data)
    key = next(iter(sorted(data)))
    assert block.erase(key) is True
    assert block.erase(key) is False
    assert len(block) == len(data) - 1
    block.clear()
    assert block.is_empty() and len(block) == 0


# --- rows are views ---------------------------------------------------------


def test_row_write_aliases_the_block(data):
    block = ManyBodyBlockState(data)
    key = sorted(data)[0]
    row = block[key]
    row[0] = 7.5 + 0.25j
    del row
    assert np.asarray(block)[0, 0] == 7.5 + 0.25j


def test_row_view_blocks_reallocation(data):
    """A live row participates in the same export guard as np.asarray(block)."""
    block = ManyBodyBlockState(data)
    row = block[sorted(data)[0]]
    with pytest.raises(RuntimeError):
        block.prune_rows(0.0)
    with pytest.raises(RuntimeError):
        block[_det(9999)] = 1 + 0j
    del row
    block.prune_rows(0.0)  # released, so the mutation is allowed again


def test_row_keeps_its_state_alive(data):
    row = ManyBodyBlockState(data)[sorted(data)[0]]
    assert row[0] == data[sorted(data)[0]]


# --- vector space vs the flat_map oracle ------------------------------------


def test_add_sub_scale_match_flat_map_bit_for_bit(data, other):
    b1, b2 = ManyBodyBlockState(data), ManyBodyBlockState(other)
    s1, s2 = ManyBodyState(data), ManyBodyState(other)
    z = 0.3 - 1.7j

    for block, state in ((b1 + b2, s1 + s2), (b1 - b2, s1 - s2), (b1 * z, s1 * z), (b1 / z, s1 / z), (-b1, -s1)):
        assert list(block.keys()) == list(state.keys())
        assert all(block[k][0] == state[k] for k in state.keys())


def test_add_scaled_matches_flat_map_bit_for_bit(data, other):
    block, state = ManyBodyBlockState(data), ManyBodyState(data)
    block.add_scaled(ManyBodyBlockState(other), 0.3 - 1.7j)
    state.add_scaled(ManyBodyState(other), 0.3 - 1.7j)
    assert list(block.keys()) == list(state.keys())
    assert all(block[k][0] == state[k] for k in state.keys())


def test_add_scaled_with_itself(data):
    """The merge reads its own rows while building the result."""
    block = ManyBodyBlockState(data)
    block.add_scaled(block, 1.0 + 0j)
    assert all(block[k][0] == 2 * data[k] for k in data)


def test_norms_and_counts_match_flat_map(data):
    block, state = ManyBodyBlockState(data), ManyBodyState(data)
    # norm2 is a plain in-order sum; the flat_map used std::transform_reduce, which is
    # free to reassociate, so these agree to ~1 ULP rather than exactly.
    assert block.norm2() == pytest.approx(state.norm2(), rel=4e-16)
    assert block.max_norm2() == state.max_norm2()
    assert block.count_above(0.5) == state.count_above(0.5)


def test_truncate_matches_flat_map(data):
    block, state = ManyBodyBlockState(data), ManyBodyState(data)
    block.truncate(10)
    state.truncate(10)
    assert list(block.keys()) == list(state.keys())


# --- width semantics --------------------------------------------------------


def test_default_state_is_the_polymorphic_zero(data, other):
    """Width 0 with no rows adopts a width on first use, so sums need no width up front."""
    zero = ManyBodyBlockState()
    assert zero.width == 0 and zero.is_empty()

    narrow = zero + ManyBodyBlockState(data)
    assert narrow.width == 1 and len(narrow) == len(data)

    wide = ManyBodyBlockState.from_states([ManyBodyState(data), ManyBodyState(other)])
    assert (ManyBodyBlockState() + wide).width == 2


def test_explicit_width_is_checked_not_adopted(data, other):
    wide = ManyBodyBlockState.from_states([ManyBodyState(data), ManyBodyState(other)])
    with pytest.raises(ValueError):
        ManyBodyBlockState(width=3).add_scaled(wide, 1.0 + 0j)


def test_explicit_width_builds_an_empty_block():
    block = ManyBodyBlockState(width=4)
    assert block.width == 4 and len(block) == 0
    assert np.asarray(block).shape == (0, 4)


def test_width_p_columns_are_independent(data):
    """Every per-column result equals the width-1 computation on that column."""
    wide = ManyBodyBlockState({k: (v, 2 * v) for k, v in data.items()})
    assert wide.width == 2
    assert all(wide[k][1] == 2 * data[k] for k in data)

    narrow = ManyBodyBlockState(data)
    assert wide.norm2() == pytest.approx(5 * narrow.norm2(), rel=1e-12)
    assert np.allclose(np.asarray(wide)[:, 0], [data[k] for k in sorted(data)])


def test_dict_width_mismatch_rejected(data):
    with pytest.raises(ValueError):
        ManyBodyBlockState({k: (v, 2 * v) for k, v in data.items()}, width=3)
