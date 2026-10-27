r"""``discovered_matrix.DiscoveredMatrix``: ``H`` over every discovered key, columns for the admitted ones.

The oracle shares no storage with the matrix: the admitted determinants' superposition is pushed
through ``applyOp`` in Python (replicated on every rank) and indexed by determinant bytes. Pinned:

* ``matvec(rows="all")`` is exactly ``H V`` -- every discovered key carries the oracle's amplitude and
  the discovered keys are exactly the admitted ones plus what their images reach;
* ``matvec(rows="P")`` is the same product restricted to the admitted keys;
* one operator application per admitted key, however the keys arrive (stages, repeats);
* a rank with no admitted key but discovered ones (``-n 3``) takes part in every collective;
* ``drop_columns`` then ``compact_p_slots`` leave the product of the remaining columns;
* ``to_scipy`` and ``matvec`` describe the same matrix.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.discovered_matrix import DiscoveredMatrix
from impurityModel.ed.ManyBodyUtils import ManyBodyState, applyOp
from impurityModel.test.basis.test_sparse_matrix_build import _N_BYTES, _all_dets, _operator
from impurityModel.test.support.comms import COMMS

WIDTH = 2


def _key(d):
    return bytes(d.to_bytearray()[:_N_BYTES])


def _owned(dets, comm):
    if comm is None or comm.size == 1:
        return list(dets)
    return [d for d in dets if d.routing_hash() % comm.size == comm.rank]


def _gather(comm, obj):
    return [obj] if comm is None else comm.allgather(obj)


def _admitted(n, seed=2):
    dets = _all_dets()
    order = np.random.default_rng(seed).permutation(len(dets))
    return [dets[i] for i in order[:n]]


def _x_of(dets):
    """A reproducible block on ``dets``: the same amplitudes on every rank for the same determinant."""
    out = np.empty((len(dets), WIDTH), dtype=complex)
    for i, d in enumerate(dets):
        rng = np.random.default_rng(int.from_bytes(_key(d), "little"))
        out[i] = rng.standard_normal(WIDTH) + 1j * rng.standard_normal(WIDTH)
    return out


def _oracle(op, dets):
    """``{key: (WIDTH,) amplitudes}`` of ``H sum_d x_d |d>``, applied in Python, per column."""
    x = _x_of(dets)
    total = {}
    for c in range(WIDTH):
        psi = ManyBodyState({d: complex(x[i, c]) for i, d in enumerate(dets)})
        image = applyOp(op, psi, cutoff=0)
        for k, a in zip(image.keys(), np.asarray(image)[:, 0]):
            total.setdefault(_key(k), np.zeros(WIDTH, dtype=complex))[c] = a
    return total


def _matrix(op, comm, owned):
    mat = DiscoveredMatrix(op, chunks_per_state=1, comm=comm)
    mat.grow(owned)
    return mat


def _check_matvec(mat, op, admitted, comm, exact_k=True):
    owned = [d for d in admitted if mat.size == 1 or d.routing_hash() % mat.size == mat.rank]
    p_keys = mat.p_keys()
    alive = mat.alive()
    live_dets = [k for k, a in zip(p_keys, alive) if a]
    X = np.zeros((mat.n_p_local, WIDTH), dtype=complex)
    X[alive] = _x_of(live_dets)
    all_live = [d for part in _gather(comm, live_dets) for d in part]
    want = _oracle(op, all_live)

    got_all = mat.matvec(X, rows="all")
    k_keys = mat.keys_of(np.arange(mat.n_k_local))
    all_k = [_key(k) for part in _gather(comm, k_keys) for k in part]
    assert len(set(all_k)) == len(all_k), "a key is in K twice"
    # K = what the images reach (admitted keys with no image row are in K as their own columns)
    assert set(want) <= set(all_k)
    if exact_k:
        assert set(all_k) - set(want) <= {_key(d) for d in all_live} | {_key(d) for d in admitted}
    for i, k in enumerate(k_keys):
        np.testing.assert_allclose(got_all[i], want.get(_key(k), np.zeros(WIDTH)), atol=1e-12)

    got_p = mat.matvec(X, rows="P")
    assert got_p.shape == (mat.n_p_local, WIDTH)
    for slot, (k, a) in enumerate(zip(p_keys, alive)):
        expected = want.get(_key(k), np.zeros(WIDTH)) if a else np.zeros(WIDTH)  # a dead slot is no P row
        np.testing.assert_allclose(got_p[slot], expected, atol=1e-12)
    return owned


@pytest.mark.parametrize("comm", COMMS)
def test_matvec_is_the_operator_applied_to_the_admitted_block(comm):
    op = _operator()
    admitted = _admitted(150)
    mat = _matrix(op, comm, _owned(admitted, comm))
    _check_matvec(mat, op, admitted, comm)


@pytest.mark.parametrize("comm", COMMS)
def test_one_application_per_admitted_key_in_any_arrival_order(comm):
    op = _operator()
    admitted = _admitted(120)
    mine = _owned(admitted, comm)
    mat = DiscoveredMatrix(op, 1, comm, batch=200)  # several lookup batches
    half = len(mine) // 2
    mat.grow(mine[:half])
    mat.grow(mine[:half])  # a repeat admits nothing
    assert mat.n_applied == half
    mat.grow(mine)  # the first half is already in
    assert mat.n_applied == len(mine)
    _check_matvec(mat, op, admitted, comm)
    reference = _matrix(op, comm, mine)
    assert reference.n_applied == len(mine)
    assert mat.nnz == reference.nnz


@pytest.mark.mpi
def test_a_rank_without_admitted_keys_still_takes_part():
    comm = MPI.COMM_WORLD
    op = _operator()
    admitted = _admitted(1)
    mat = _matrix(op, comm, _owned(admitted, comm))
    _check_matvec(mat, op, admitted, comm)
    if comm.size >= 3:
        empty = [mat.n_p_local == 0 and mat.n_k_local > 0 for _ in [0]]
        assert any(comm.allgather(empty[0])), "no rank has discovered keys but no column"


@pytest.mark.parametrize("comm", COMMS)
def test_dropped_columns_leave_the_remaining_product_and_compaction_keeps_it(comm):
    op = _operator()
    admitted = _admitted(100, seed=4)
    mat = _matrix(op, comm, _owned(admitted, comm))
    drop = np.arange(0, mat.n_p_local, 3)
    kept_keys = [k for i, k in enumerate(mat.p_keys()) if i % 3 != 0]
    mat.drop_columns(drop)
    assert mat.n_alive_local == len(kept_keys)
    dead_cols = mat.to_scipy("all")[:, drop]
    assert dead_cols.nnz == 0, "a dropped column still has entries in to_scipy"
    _check_matvec(mat, op, admitted, comm, exact_k=False)  # dropped columns' rows stay in K
    keep = mat.compact_p_slots()
    assert mat.n_p_local == len(kept_keys) and len(keep) == len(kept_keys)
    assert [_key(k) for k in mat.p_keys()] == [_key(k) for k in kept_keys]
    _check_matvec(mat, op, admitted, comm, exact_k=False)
    # a dropped key comes back as a new column
    dropped = [d for d in _owned(admitted, comm) if _key(d) not in {_key(k) for k in kept_keys}]
    n_before = mat.n_applied
    mat.grow(dropped)
    assert mat.n_applied == n_before + len(dropped)
    _check_matvec(mat, op, admitted, comm, exact_k=False)


@pytest.mark.parametrize("comm", COMMS)
def test_to_scipy_and_matvec_describe_the_same_matrix(comm):
    op = _operator()
    admitted = _admitted(80, seed=6)
    mat = _matrix(op, comm, _owned(admitted, comm))
    X = _x_of(mat.p_keys())
    for rows in ("all", "P"):
        local = mat.to_scipy(rows) @ X
        total = local if comm is None else comm.allreduce(local)
        got = mat.matvec(X, rows=rows)
        want = np.concatenate(_gather(comm, got), axis=0)
        np.testing.assert_allclose(total, want, atol=1e-12)


@pytest.mark.parametrize("comm", COMMS)
def test_without_the_frontier_the_matrix_is_p_h_p(comm):
    op = _operator()
    admitted = _admitted(150)
    mine = _owned(admitted, comm)
    mat = DiscoveredMatrix(op, 1, comm, keep_frontier=False)
    mat.grow(mine)
    assert mat.n_k_local == mat.n_p_local, "K must stay equal to P"
    alive_keys = mat.p_keys()
    X = _x_of(alive_keys)
    want = _oracle(op, [d for part in _gather(comm, alive_keys) for d in part])
    in_p = {_key(d) for part in _gather(comm, alive_keys) for d in part}
    got = mat.matvec(X, rows="all")
    for slot, k in enumerate(alive_keys):
        np.testing.assert_allclose(got[slot], want.get(_key(k), np.zeros(WIDTH)), atol=1e-12)
    assert {_key(k) for k in alive_keys} <= in_p


@pytest.mark.parametrize("comm", COMMS)
def test_columns_built_after_admission_equal_columns_grown_with_it(comm):
    op = _operator()
    mine = _owned(_admitted(120, seed=8), comm)
    grown = DiscoveredMatrix(op, 1, comm, keep_frontier=False)
    grown.grow(mine)
    late = DiscoveredMatrix(op, 1, comm, keep_frontier=False)
    late.admit_keys(mine)
    assert late.n_applied == 0 and late.nnz == 0
    late.build_columns()
    assert late.n_applied == len(mine)
    assert [_key(k) for k in late.p_keys()] == [_key(k) for k in grown.p_keys()]
    X = _x_of(grown.p_keys())
    np.testing.assert_allclose(late.matvec(X, rows="all"), grown.matvec(X, rows="all"), atol=1e-13)


def test_the_identity_translation_is_recognised_per_rank_block():
    """``_is_identity`` is true exactly when every K row of every rank maps to the P slot of the same number."""
    from impurityModel.ed.discovered_matrix import _is_identity

    goff = np.array([0, 3, 5], dtype=np.int64)  # ranks hold 3, 2 and 4 K rows
    identity = np.array([0, 1, 2, 0, 1, 0, 1, 2, 3], dtype=np.int64)
    assert _is_identity(goff, identity)
    assert _is_identity(goff, np.zeros(0, dtype=np.int64))
    dead = identity.copy()
    dead[4] = -1  # a K row outside P
    assert not _is_identity(goff, dead)
    shifted = identity.copy()
    shifted[7] = 1  # a P slot that is not the row's own number
    assert not _is_identity(goff, shifted)
    global_numbering = np.arange(9, dtype=np.int64)  # numbered over all ranks, not per rank
    assert not _is_identity(goff, global_numbering)


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("slot_bits", [None, 0])
def test_both_row_widths_give_the_same_matrix(comm, slot_bits):
    """Entries are int32 rows when every slot fits and int64 otherwise (forced here with ``slot_bits = 0``);
    the matvec and ``to_scipy`` are the same either way."""
    op = _operator()
    admitted = _admitted(150)
    mat = DiscoveredMatrix(op, chunks_per_state=1, comm=comm)
    if slot_bits is not None:
        mat._slot_bits = slot_bits
    mat.grow(_owned(admitted, comm))
    want = np.int32 if slot_bits is None else np.int64
    assert all(seg[2].dtype == want for seg in mat._segments)
    _check_matvec(mat, op, admitted, comm)
    X = np.ones((mat.n_p_local, WIDTH), dtype=complex)
    contribution = np.ascontiguousarray(mat.to_scipy("all") @ X)  # this rank's columns; decodes the stored rows
    total = np.empty_like(contribution)
    if comm is None or comm.size == 1:
        total[...] = contribution
    else:
        comm.Allreduce(contribution, total, op=MPI.SUM)
    gathered = np.concatenate(_gather(comm, mat.matvec(X, rows="all")))
    np.testing.assert_allclose(total, gathered, atol=1e-12)
    mat.drop_columns(np.arange(0, mat.n_p_local, 3))
    mat.compact_p_slots()
    _check_matvec(mat, op, admitted, comm, exact_k=False)


@pytest.mark.parametrize("comm", COMMS)
def test_sorting_the_p_slots_numbers_them_in_key_order_before_the_columns_are_built(comm):
    """The matrix-free modes admit keys with empty columns and build them at the freeze; sorting the slots before
    that gives the product the matrix admitted in one sorted batch has, with the block carried by ``order``."""
    op = _operator()
    mine = _owned(_admitted(150), comm)
    mat = DiscoveredMatrix(op, 1, comm, keep_frontier=False)
    mat.admit_keys(mine[: len(mine) // 3])
    mat.admit_keys(mine[len(mine) // 3 :])  # two batches: the slots are in arrival order
    old_keys = mat.p_keys()
    order = mat.sort_p_slots()
    keys = [_key(k) for k in mat.p_keys()]
    assert keys == sorted(keys)
    assert [_key(old_keys[i]) for i in order] == keys
    mat.build_columns()
    ref = DiscoveredMatrix(op, 1, comm, keep_frontier=False)
    ref.admit_keys(mine)
    ref.build_columns()
    assert [_key(k) for k in ref.p_keys()] == keys
    X = _x_of(mat.p_keys())
    np.testing.assert_allclose(mat.matvec(X, rows="P"), ref.matvec(X, rows="P"), atol=1e-12)
