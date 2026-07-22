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
    """block apply == p independent scalar applies at cutoff 0.

    Last-ulp tolerance, not bit-for-bit: the block and scalar loops are different
    code, so compilers may contract their multiply-accumulates into FMA differently
    (Intel icx does; the threaded merge additionally reorders duplicate accumulation).
    """
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
        diff = got[c] - ref[c]
        assert np.sqrt(diff.norm2()) < 1e-12 * max(
            np.sqrt(ref[c].norm2()), 1.0
        ), f"column {c} differs from independent apply"


def test_apply_block_cutoff_keeps_rows():
    """The block cutoff acts on whole rows: sub-cutoff residuals in one column are
    retained when another column keeps the row (shared-support semantics)."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    rng = np.random.default_rng(41)
    op = ManyBodyOperator(_hopping_operator(8, rng))
    dets = _dets_orbital_msb(8, rng, 12, 4)
    big = ManyBodyState(dict.fromkeys(dets, 1.0 + 0j))
    tiny = ManyBodyState(dict.fromkeys(dets, 1e-10 + 0j))
    cutoff = 1e-6

    blk_out = op.apply_block(ManyBodyBlockState.from_states([big, tiny]), cutoff)
    ref_big = op.apply_multi([big], cutoff)[0]
    # every row that survives via the big column is present, and the big column matches
    got = blk_out.to_states()
    assert got[0] == ref_big
    # the tiny column keeps its (sub-cutoff) residuals on the shared rows: applying the
    # scalar path with cutoff 0 and restricting to the surviving support must match
    ref_tiny_full = op.apply_multi([tiny], 0.0)[0]
    for sd, _amp in got[1].items():
        assert sd in ref_tiny_full


# --------------------------------------------------------------------------- #
# Phase 2.3: block redistribute
# --------------------------------------------------------------------------- #
def test_graph_alltoall_block_serial_copy():
    """Serial (comm=None): an independent copy comes back, like graph_alltoall_psis."""
    from impurityModel.ed.mpi_comm import graph_alltoall_block

    rng = np.random.default_rng(51)
    states = _random_states(rng, 3, 20)
    blk = ManyBodyBlockState.from_states(states)
    out = graph_alltoall_block(blk, 2, None)
    assert out == blk
    np.asarray(out)[0, 0] += 1.0
    assert out != blk  # independent storage


@pytest.mark.mpi
def test_graph_alltoall_block_matches_scalar_path():
    """Distributed: block redistribute must equal the scalar redistribute of the same
    columns bit-for-bit, including cross-rank duplicate summation."""
    from mpi4py import MPI

    from impurityModel.ed.mpi_comm import graph_alltoall_block, graph_alltoall_psis

    comm = MPI.COMM_WORLD
    p = 3
    n_bytes = 2
    rng = np.random.default_rng(400 + comm.rank)
    # Every rank holds partial amplitudes for an overlapping determinant set, so the
    # owners receive contributions from several ranks and must sum them.
    states = _random_states(rng, p, 30, sparsity=0.8)

    ref = graph_alltoall_psis(states, n_bytes, comm)
    out = graph_alltoall_block(ManyBodyBlockState.from_states(states), n_bytes, comm)

    assert out.width == p
    got = out.to_states()
    for c in range(p):
        assert got[c] == ref[c], f"rank {comm.rank}: column {c} differs from scalar redistribute"


@pytest.mark.mpi
def test_graph_alltoall_block_empty_contributor():
    """A rank contributing zero rows (empty block of width p) must not deadlock or
    corrupt the result — collectives are unconditional, dtypes fixed."""
    from mpi4py import MPI

    from impurityModel.ed.mpi_comm import graph_alltoall_block, graph_alltoall_psis

    comm = MPI.COMM_WORLD
    p = 2
    n_bytes = 2
    if comm.rank == comm.size - 1:
        states = [ManyBodyState() for _ in range(p)]
    else:
        rng = np.random.default_rng(500 + comm.rank)
        states = _random_states(rng, p, 15, sparsity=0.9)

    ref = graph_alltoall_psis(states, n_bytes, comm)
    out = graph_alltoall_block(ManyBodyBlockState.from_states(states), n_bytes, comm)
    got = out.to_states()
    for c in range(p):
        assert got[c] == ref[c]


# --------------------------------------------------------------------------- #
# Phase 2.4a: block linear-algebra primitives — bit-for-bit vs the list ops
# --------------------------------------------------------------------------- #
def test_block_inner_matches_inner_multi():
    from impurityModel.ed.ManyBodyUtils import block_inner_cy, inner_multi

    rng = np.random.default_rng(61)
    A_states = _random_states(rng, 3, 25)
    # different (overlapping) support for B
    B_states = [
        ManyBodyState({_det(i + 10): rng.standard_normal() + 1j for i in range(20) if rng.random() < 0.7})
        for _ in range(2)
    ]
    A = ManyBodyBlockState.from_states(A_states)
    B = ManyBodyBlockState.from_states(B_states)
    # Same accumulation order, but FMA contraction of conj(a)*b sums differs across
    # compilers (Intel icx) between the block and list code paths — last-ulp tolerance.
    np.testing.assert_allclose(block_inner_cy(A, B), inner_multi(A_states, B_states), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(block_inner_cy(A, A), inner_multi(A_states, A_states), rtol=1e-13, atol=1e-14)


def test_block_add_scaled_matches_add_scaled_multi():
    from impurityModel.ed.ManyBodyUtils import add_scaled_multi, block_add_scaled_cy

    rng = np.random.default_rng(62)
    A_states = _random_states(rng, 3, 25)
    B_states = [
        ManyBodyState(
            {_det(i + 15): rng.standard_normal() + 1j * rng.standard_normal() for i in range(20) if rng.random() < 0.7}
        )
        for _ in range(2)
    ]
    C = rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
    ref = [s.copy() for s in A_states]
    add_scaled_multi(ref, B_states, np.ascontiguousarray(C))
    out = block_add_scaled_cy(ManyBodyBlockState.from_states(A_states), ManyBodyBlockState.from_states(B_states), C)
    for j, col in enumerate(out.to_states()):
        # same accumulation order; last-ulp tolerance for compiler FMA differences
        diff = col - ref[j]
        assert np.sqrt(diff.norm2()) < 1e-13 * max(np.sqrt(ref[j].norm2()), 1.0)
    with pytest.raises(ValueError):
        block_add_scaled_cy(
            ManyBodyBlockState.from_states(A_states), ManyBodyBlockState.from_states(B_states), np.ones((3, 2))
        )


def test_block_inner_dispatcher_accepts_blocks():
    """``BlockLanczosArray.block_inner`` must dispatch a ``ManyBodyBlockState`` to
    ``block_inner_cy`` rather than falling through to the list-only ``inner_multi``
    (which would silently iterate determinant keys instead of rows)."""
    from impurityModel.ed.BlockLanczosArray import block_inner
    from impurityModel.ed.ManyBodyUtils import block_inner_cy

    rng = np.random.default_rng(65)
    A_states = _random_states(rng, 3, 25)
    B_states = _random_states(rng, 2, 25)
    A = ManyBodyBlockState.from_states(A_states)
    B = ManyBodyBlockState.from_states(B_states)
    np.testing.assert_array_equal(block_inner(A, B), block_inner_cy(A, B))


def test_block_add_scaled_dispatcher_accepts_blocks():
    """``BlockLanczosArray.block_add_scaled`` must rebind (not mutate in place) when fed
    a ``ManyBodyBlockState`` -- the union support can outgrow the target's own storage,
    so the caller must use the returned block, exactly as every current caller does."""
    from impurityModel.ed.BlockLanczosArray import block_add_scaled
    from impurityModel.ed.ManyBodyUtils import block_add_scaled_cy

    rng = np.random.default_rng(66)
    A_states = _random_states(rng, 3, 25)
    B_states = [
        ManyBodyState(
            {_det(i + 15): rng.standard_normal() + 1j * rng.standard_normal() for i in range(20) if rng.random() < 0.7}
        )
        for _ in range(3)
    ]
    A = ManyBodyBlockState.from_states(A_states)
    B = ManyBodyBlockState.from_states(B_states)
    C = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))

    ref = block_add_scaled_cy(A, B, C)
    out = block_add_scaled(A, B, C)
    assert out.to_states() == ref.to_states()
    # A itself is untouched -- the block was rebound, not mutated.
    assert A.to_states() == A_states

    # slaterWeightMin pruning is applied to the rebound result.
    pruned = block_add_scaled(A, B, C, slaterWeightMin=1e6)
    assert all(len(s) == 0 for s in pruned.to_states())


def test_combine_columns_matches_block_combine_sparse():
    from impurityModel.ed.BlockLanczos import block_combine_sparse

    rng = np.random.default_rng(63)
    states = _random_states(rng, 4, 30)
    blk = ManyBodyBlockState.from_states(states)
    Y = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    ref = block_combine_sparse(states, Y)
    for j, col in enumerate(blk.combine_columns(Y).to_states()):
        # same accumulation order; last-ulp tolerance for compiler FMA differences
        diff = col - ref[j]
        assert np.sqrt(diff.norm2()) < 1e-13 * max(np.sqrt(ref[j].norm2()), 1.0)
    # width can shrink (deflation) and grow
    assert blk.combine_columns(np.eye(4)[:, :1]).width == 1


def test_store_append_block_matches_append():
    from impurityModel.ed.ManyBodyUtils import SparseKrylovDense

    rng = np.random.default_rng(64)
    states = _random_states(rng, 3, 40)
    more = _random_states(rng, 2, 55)
    s1, s2 = SparseKrylovDense(), SparseKrylovDense()
    s1.append(states)
    s1.append(more)
    s2.append_block(ManyBodyBlockState.from_states(states))
    s2.append_block(ManyBodyBlockState.from_states(more))
    assert len(s1) == len(s2) == 5
    assert list(s1) == list(s2)


def test_keep_rows_intersection():
    """keep_rows keeps exactly the rows whose key is in the mask's support (the
    set-intersection complement of prune_rows), preserving row order."""
    a = ManyBodyState({_det(1): 1.0 + 0j, _det(2): 2.0 + 0j, _det(4): 4.0 + 0j})
    b = ManyBodyState({_det(2): -1.0 + 0j, _det(3): 3.0 + 0j})
    blk = ManyBodyBlockState.from_states([a, b])
    assert len(blk) == 4
    # Mask keys need not be a subset of the block's support (det(9) is absent).
    mask_state = ManyBodyState(dict.fromkeys([_det(2), _det(4), _det(9)], 1.0 + 0j))
    blk.keep_rows(ManyBodyBlockState.from_states([mask_state]))
    assert len(blk) == 2
    kept = blk.to_states()
    assert kept[0] == ManyBodyState({_det(2): 2.0 + 0j, _det(4): 4.0 + 0j})
    assert kept[1] == ManyBodyState({_det(2): -1.0 + 0j})


def test_keep_rows_superset_and_empty_mask():
    rng = np.random.default_rng(31)
    states = _random_states(rng, 3, 25)
    blk = ManyBodyBlockState.from_states(states)
    full = blk.copy()
    all_keys = ManyBodyState(dict.fromkeys(blk.support_keys(0.0), 1.0 + 0j))
    blk.keep_rows(ManyBodyBlockState.from_states([all_keys]))
    assert blk == full  # superset mask is the identity
    blk.keep_rows(ManyBodyBlockState.from_states([]))
    assert len(blk) == 0  # empty mask drops everything


def test_keep_rows_export_guard():
    blk = ManyBodyBlockState.from_states(_random_states(np.random.default_rng(32), 2, 10))
    mask = blk.copy()
    arr = np.asarray(blk)
    with pytest.raises(RuntimeError):
        blk.keep_rows(mask)
    del arr
    blk.keep_rows(mask)  # fine once the view is released


def test_row_max_norms2_values_and_alignment():
    """keys[i] is aligned with norms2[i] over the FULL support, including exact-zero
    rows (which support_keys(0.0) drops)."""
    a = ManyBodyState({_det(1): 0.0 + 0j, _det(2): 3.0 + 4.0j})
    b = ManyBodyState({_det(2): 1.0 + 0j, _det(3): -2.0 + 0j})
    blk = ManyBodyBlockState.from_states([a, b])
    keys, norms2 = blk.row_max_norms2()
    assert keys == [_det(1), _det(2), _det(3)]
    np.testing.assert_allclose(norms2, [0.0, 25.0, 4.0])
    # the zero row is invisible to support_keys(0.0) but present here
    assert _det(1) not in blk.support_keys(0.0)
    assert len(keys) == len(blk)


def test_row_max_norms2_empty():
    blk = ManyBodyBlockState.from_states([])
    keys, norms2 = blk.row_max_norms2()
    assert keys == [] and norms2.shape == (0,)


def test_count_rows_in_and_new_row_max_norms2():
    a = ManyBodyState({_det(1): 1.0 + 0j, _det(2): 2.0 + 0j, _det(4): 0.5 + 0j})
    b = ManyBodyState({_det(2): -3.0 + 0j, _det(3): 1.0 + 1.0j})
    blk = ManyBodyBlockState.from_states([a, b])  # rows 1,2,3,4
    mask = ManyBodyBlockState.from_states([ManyBodyState(dict.fromkeys([_det(2), _det(4)], 1.0 + 0j))])
    assert blk.count_rows_in(mask) == 2
    # new rows are det(1) and det(3), in row order
    np.testing.assert_allclose(blk.new_row_max_norms2(mask), [1.0, 2.0])
    assert len(blk.new_row_max_norms2(mask)) == len(blk) - blk.count_rows_in(mask)
    empty_mask = ManyBodyBlockState.from_states([])
    assert blk.count_rows_in(empty_mask) == 0
    np.testing.assert_allclose(blk.new_row_max_norms2(empty_mask), [1.0, 9.0, 2.0, 0.25])


def test_keys_new_above_and_key_union():
    a = ManyBodyState({_det(1): 1.0 + 0j, _det(2): 2.0 + 0j, _det(4): 0.5 + 0j})
    b = ManyBodyState({_det(2): -3.0 + 0j, _det(3): 1.0 + 1.0j})
    blk = ManyBodyBlockState.from_states([a, b])
    mask = ManyBodyBlockState.from_states([ManyBodyState({_det(2): 1.0 + 0j})])
    # candidates 1 (norm2 1.0), 3 (norm2 2.0), 4 (norm2 0.25); cutoff2=0.5 admits 1 and 3
    admitted = blk.keys_new_above(mask, 0.5)
    assert admitted.width == 0
    assert len(admitted) == 2
    grown = mask.key_union(admitted)
    assert len(grown) == 3 and grown.width == 0
    # the union mask keeps exactly rows 1,2,3 of the block
    kept = blk.copy()
    kept.keep_rows(grown)
    assert len(kept) == 3
    keys, _ = kept.row_max_norms2()
    assert keys == [_det(1), _det(2), _det(3)]
    # cutoff above every candidate admits nothing; union with empty is identity
    assert len(blk.keys_new_above(mask, 100.0)) == 0
    assert len(mask.key_union(ManyBodyBlockState.from_states([]))) == 1


def test_merge_keys_inplace_matches_key_union():
    a = ManyBodyState({_det(1): 1.0 + 0j, _det(4): 4.0 + 0j})
    b = ManyBodyState({_det(2): -1.0 + 0j, _det(4): 3.0 + 0j, _det(7): 1.0 + 0j})
    mask = ManyBodyBlockState.from_states([a]).key_union(ManyBodyBlockState())
    assert mask.width == 0 and len(mask) == 2
    other = ManyBodyBlockState.from_states([b])
    expected = mask.key_union(other)
    mask.merge_keys(other)
    keys, _ = mask.row_max_norms2()
    exp_keys, _ = expected.row_max_norms2()
    assert keys == exp_keys == [_det(1), _det(2), _det(4), _det(7)]
    mask.merge_keys(ManyBodyBlockState())  # empty merge is a no-op
    assert len(mask) == 4
    # width-0 requirement is enforced
    with pytest.raises(ValueError):
        other.merge_keys(mask)


def test_keys_matches_support_keys_and_works_at_width_zero():
    """``keys()`` is the accessor for width-0 mask blocks, where ``support_keys`` cannot help.

    ``support_keys`` filters on the row's column amplitudes; a key-only mask has none, so it
    returns nothing there. ``keys()`` returns the support in row order either way, and on a
    block whose rows all carry amplitude it agrees with ``support_keys(0.0)`` elementwise.
    """
    a = ManyBodyState({_det(1): 1.0 + 0j, _det(4): 4.0 + 0j})
    b = ManyBodyState({_det(2): -1.0 + 0j, _det(4): 3.0 + 0j})
    blk = ManyBodyBlockState.from_states([a, b])
    assert blk.keys() == blk.support_keys(0.0) == [_det(1), _det(2), _det(4)]

    mask = ManyBodyBlockState()
    mask.merge_keys(blk)
    assert mask.width == 0
    assert mask.support_keys(0.0) == []  # no amplitudes to filter on
    assert mask.keys() == [_det(1), _det(2), _det(4)]
    assert ManyBodyBlockState().keys() == []


def test_seen_and_offered_masks_track_sub_cutoff_determinants_separately():
    """The invariant behind BiCGSTAB's two-mask bookkeeping.

    A determinant can enter the block support below ``slaterWeightMin`` -- so it is *seen*
    (it counts toward the Krylov-exhaustion bound) but must not yet be *offered* to the
    basis -- and grow above the cutoff on a later iteration, at which point it must be
    offered. A single mask would record it on the first sighting and never offer it.
    """
    cutoff2 = 1e-12**2
    seen, offered = ManyBodyBlockState(), ManyBodyBlockState()

    # Iteration 1: det(2) is present but far below the cutoff.
    step1 = ManyBodyBlockState.from_states([ManyBodyState({_det(1): 1.0 + 0j, _det(2): 1e-20 + 0j})])
    seen.merge_keys(step1.keys_new_above(seen, 0.0))
    new_offered = step1.keys_new_above(offered, cutoff2)
    offered.merge_keys(new_offered)
    assert seen.keys() == [_det(1), _det(2)]
    assert new_offered.keys() == [_det(1)]

    # Iteration 2: det(2) has grown above the cutoff. It is already seen, but not offered,
    # so exactly one new determinant reaches the basis.
    step2 = ManyBodyBlockState.from_states([ManyBodyState({_det(1): 1.0 + 0j, _det(2): 0.5 + 0j})])
    seen.merge_keys(step2.keys_new_above(seen, 0.0))
    new_offered = step2.keys_new_above(offered, cutoff2)
    offered.merge_keys(new_offered)
    assert len(seen) == 2  # unchanged: nothing newly reachable
    assert new_offered.keys() == [_det(2)]
    assert offered.keys() == [_det(1), _det(2)]

    # Idempotent: a third sighting offers nothing.
    assert step2.keys_new_above(offered, cutoff2).keys() == []
