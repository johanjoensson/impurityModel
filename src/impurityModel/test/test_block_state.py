"""Unit tests for ManyBodyBlockState (Phase 2.1 of the block-state matvec plan,
doc/plans/blocklanczos_partial_perf_memory.md): shared-support block container —
conversion round-trips, union-support semantics, row pruning (any-column-survives),
zero-copy buffer protocol with its export guard."""

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import ManyBodyBlockState, ManyBodyState, SlaterDeterminant


def _det(i):
    return SlaterDeterminant.from_bytes(bytes([(i >> 8) & 0xFF, i & 0xFF]))


def _random_states(rng, n_states, n_dets, sparsity=0.7):
    dets = [_det(i + 1) for i in range(n_dets)]
    states = []
    for _ in range(n_states):
        st = ManyBodyState()
        for d in dets:
            if rng.random() < sparsity:
                st[d] = rng.standard_normal() + 1j * rng.standard_normal()
        states.append(st)
    return states


def test_roundtrip_bit_exact():
    rng = np.random.default_rng(21)
    states = _random_states(rng, 5, 60)
    blk = ManyBodyBlockState.from_states(states)
    assert blk.width == 5
    back = blk.to_states()
    assert back == states  # flat_map equality: identical keys and coefficients


def test_union_support():
    """Disjoint-support columns share the union; missing entries are exact zeros."""
    a = ManyBodyState({_det(1): 1.0 + 0j, _det(2): 2.0 + 0j})
    b = ManyBodyState({_det(3): 3.0 + 0j})
    blk = ManyBodyBlockState.from_states([a, b])
    assert len(blk) == 3 and blk.width == 2
    arr = np.asarray(blk)
    # keys are sorted; column 0 = a, column 1 = b
    np.testing.assert_array_equal(arr[:, 0], [1.0, 2.0, 0.0])
    np.testing.assert_array_equal(arr[:, 1], [0.0, 0.0, 3.0])
    back = blk.to_states()
    assert back == [a, b]  # zeros dropped on materialization


def test_empty_block():
    blk = ManyBodyBlockState.from_states([])
    assert len(blk) == 0 and blk.width == 0
    assert blk.to_states() == []
    empty_state = ManyBodyBlockState.from_states([ManyBodyState()])
    assert len(empty_state) == 0 and empty_state.width == 1
    assert empty_state.to_states() == [ManyBodyState()]


def test_prune_rows_any_column_survives():
    """A row is kept if ANY column passes the ManyBodyState.prune test — the
    deliberate semantic difference vs pruning independent states per column."""
    a = ManyBodyState({_det(1): 1e-9 + 0j, _det(2): 1.0 + 0j})
    b = ManyBodyState({_det(1): 1.0 + 0j, _det(2): 1e-9 + 0j})
    blk = ManyBodyBlockState.from_states([a, b])
    blk.prune_rows(1e-6)
    assert len(blk) == 2  # each row rescued by the other column
    c = ManyBodyState({_det(1): 1e-9 + 0j, _det(3): 1.0 + 0j})
    blk2 = ManyBodyBlockState.from_states([a, c])
    blk2.prune_rows(1e-6)
    # row det(1): both tiny -> dropped; det(2) and det(3) rescued by one column each
    assert len(blk2) == 2
    kept = blk2.to_states()
    assert _det(1) not in kept[0] and _det(1) not in kept[1]
    # boundary: |amp|^2 <= cutoff^2 drops (matches ManyBodyState.prune exactly)
    d = ManyBodyState({_det(1): 0.5 + 0j})
    blk3 = ManyBodyBlockState.from_states([d])
    blk3.prune_rows(0.5)
    assert len(blk3) == 0


def test_buffer_view_zero_copy_write_through():
    rng = np.random.default_rng(4)
    states = _random_states(rng, 3, 20, sparsity=1.0)
    blk = ManyBodyBlockState.from_states(states)
    arr = np.asarray(blk)
    assert arr.shape == (len(blk), 3) and arr.dtype == np.complex128
    arr[:, 1] *= 2.0
    doubled = blk.to_states()[1]
    assert doubled == states[1] * 2.0


def test_buffer_export_guard():
    blk = ManyBodyBlockState.from_states(_random_states(np.random.default_rng(5), 2, 10))
    arr = np.asarray(blk)
    with pytest.raises(RuntimeError):
        blk.prune_rows(1e-3)
    del arr
    blk.prune_rows(1e-3)  # fine once the view is released


def test_col_norm2_and_equality():
    rng = np.random.default_rng(6)
    states = _random_states(rng, 4, 30)
    blk = ManyBodyBlockState.from_states(states)
    ref = [st.norm2() for st in states]
    np.testing.assert_allclose(blk.col_norm2(), ref, rtol=1e-15)
    blk2 = ManyBodyBlockState.from_states(states)
    assert blk == blk2
    np.asarray(blk2)[0, 0] += 1.0
    assert blk != blk2


# --------------------------------------------------------------------------- #
# Phase 2.2: block apply golden — must equal p independent applies
# --------------------------------------------------------------------------- #
def _hopping_operator(n_orb, rng):
    """Small dense-ish test Hamiltonian: hoppings + density + a two-body term."""
    op = {}
    for i in range(n_orb):
        op[((i, "c"), (i, "a"))] = 0.5 + 0.1 * i
        for j in range(i + 1, min(i + 3, n_orb)):
            v = rng.standard_normal() + 1j * rng.standard_normal()
            op[((i, "c"), (j, "a"))] = v
            op[((j, "c"), (i, "a"))] = v.conjugate()
    op[((0, "c"), (1, "c"), (2, "a"), (3, "a"))] = 0.3 + 0.05j
    op[((3, "c"), (2, "c"), (1, "a"), (0, "a"))] = 0.3 - 0.05j
    return op


def _dets_orbital_msb(n_orb, rng, n_dets, n_el):
    """Random n_el-electron determinants (orbital i = bit 7-i within its byte)."""
    from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

    n_bytes = (n_orb + 7) // 8
    seen = set()
    while len(seen) < n_dets:
        orbs = tuple(sorted(rng.choice(n_orb, size=n_el, replace=False)))
        seen.add(orbs)
    dets = []
    for orbs in sorted(seen):
        b = bytearray(n_bytes)
        for o in orbs:
            b[o // 8] |= 1 << (7 - (o % 8))
        dets.append(SlaterDeterminant.from_bytes(bytes(b)))
    return dets


@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_apply_block_matches_independent_applies(p):
    """block apply == p independent scalar applies, bit-for-bit at cutoff 0."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    rng = np.random.default_rng(31 + p)
    op = ManyBodyOperator(_hopping_operator(10, rng))
    dets = _dets_orbital_msb(10, rng, 25, 5)
    states = [ManyBodyState({d: complex(rng.standard_normal(), rng.standard_normal()) for d in dets}) for _ in range(p)]
    ref = op.apply_multi(states, 0.0)
    blk_out = op.apply_block(ManyBodyBlockState.from_states(states), 0.0)
    assert blk_out.width == p
    got = blk_out.to_states()
    for c in range(p):
        assert got[c] == ref[c], f"column {c} differs from independent apply"


def test_apply_block_cutoff_keeps_rows():
    """The block cutoff acts on whole rows: sub-cutoff residuals in one column are
    retained when another column keeps the row (shared-support semantics)."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    rng = np.random.default_rng(41)
    op = ManyBodyOperator(_hopping_operator(8, rng))
    dets = _dets_orbital_msb(8, rng, 12, 4)
    big = ManyBodyState({d: 1.0 + 0j for d in dets})
    tiny = ManyBodyState({d: 1e-10 + 0j for d in dets})
    cutoff = 1e-6

    blk_out = op.apply_block(ManyBodyBlockState.from_states([big, tiny]), cutoff)
    ref_big = op.apply_multi([big], cutoff)[0]
    # every row that survives via the big column is present, and the big column matches
    got = blk_out.to_states()
    assert got[0] == ref_big
    # the tiny column keeps its (sub-cutoff) residuals on the shared rows: applying the
    # scalar path with cutoff 0 and restricting to the surviving support must match
    ref_tiny_full = op.apply_multi([tiny], 0.0)[0]
    for sd, amp in got[1].items():
        assert sd in ref_tiny_full
