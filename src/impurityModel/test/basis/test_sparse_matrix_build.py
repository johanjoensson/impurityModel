r"""``basis_transcription.build_sparse_matrix``: the matrix, its layout, and its memory.

The build assembles the CSC directly (images arrive one ket at a time, so the entries are already
grouped by column). These tests pin that against an oracle that shares no code with it -- the full
operator applied to every determinant in Python and indexed by position -- and pin the memory
the frozen-basis fit check (``gf_solvers._CSR_BYTES_PER_ELEMENT``) relies on:

* every rank's stored columns equal the oracle's, whatever the rank count, including ranks that own
  no determinant;
* ``local_columns=True`` is exactly the ``(N, N_local)`` slice of the default ``(N, N)`` result;
* the result is canonical (sorted, duplicate-free), as the old COO route made it;
* the frozen-basis column operator's build peak stays below what the fit check budgets.
"""

import itertools
import tracemalloc

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import basis_transcription, gf_solvers
from impurityModel.ed.basis_transcription import build_sparse_matrix
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp

N_ORB = 12
_N_BYTES = 2


def _det(occupied):
    """MSB-first: orbital i is bit 7 - i%8 of byte i//8."""
    raw = bytearray(_N_BYTES)
    for i in occupied:
        raw[i // 8] |= 1 << (7 - i % 8)
    return SlaterDeterminant.from_bytes(bytes(raw))


def _operator(n_orb=N_ORB, seed=1):
    rng = np.random.default_rng(seed)
    terms = {((o, "c"), (o, "a")): float(rng.normal()) for o in range(n_orb)}
    for a in range(n_orb):
        for b in range(a + 1, n_orb):
            if rng.random() < 0.35:
                v = float(rng.normal()) * 0.3 + 0.1j * float(rng.normal())
                terms[((a, "c"), (b, "a"))] = v
                terms[((b, "c"), (a, "a"))] = np.conj(v)
    for a in range(4):
        terms[((a, "c"), (a + 4, "c"), (a + 4, "a"), (a, "a"))] = 3.0
    return ManyBodyOperator(terms)


def _basis(dets, comm, n_orb=N_ORB):
    imp = {0: [list(range(0, 4))]}
    baths = ({0: [list(range(4, 8))]}, {0: [list(range(8, n_orb))]})
    return Basis(imp, baths, initial_basis=sorted(dets), comm=comm, verbose=False)


def _all_dets(n_orb=N_ORB):
    return [_det(c) for c in itertools.combinations(range(n_orb), n_orb // 2)]


def _oracle_columns(basis, op, comm):
    """Dense ``(N, N_local)`` columns of ``op`` by applying it to each local determinant in Python.

    Global indices come from the keys every rank holds (allgathered as bytes), not from the
    build's own routed lookup.
    """
    local_keys = [bytes(d.to_bytearray()[:_N_BYTES]) for d in basis.local_basis]
    keys = [k for rank_keys in (comm.allgather(local_keys) if comm is not None else [local_keys]) for k in rank_keys]
    position = {k: i for i, k in enumerate(keys)}
    dense = np.zeros((len(keys), len(local_keys)), dtype=complex)
    for col, det in enumerate(basis.local_basis):
        image = applyOp(op, ManyBodyState({det: 1.0 + 0j}), cutoff=0)
        for state, amp in zip(image.keys(), np.asarray(image)[:, 0]):
            row = position.get(bytes(state.to_bytearray()[:_N_BYTES]))
            if row is not None:
                dense[row, col] += amp
    return dense


@pytest.mark.mpi
@pytest.mark.parametrize("n_dets", [None, 2])
def test_stored_columns_equal_the_independent_oracle(n_dets):
    """The full basis, and a two-determinant one that leaves ranks empty at -n 3."""
    comm = MPI.COMM_WORLD
    dets = _all_dets() if n_dets is None else _all_dets()[:n_dets]
    basis = _basis(dets, comm)
    op = _operator()
    oracle = _oracle_columns(basis, op, comm)
    full = build_sparse_matrix(basis, op)
    local = build_sparse_matrix(basis, op, local_columns=True)

    assert full.shape == (basis.size, basis.size)
    assert local.shape == (basis.size, len(basis.local_basis))
    np.testing.assert_allclose(local.toarray(), oracle, atol=1e-13)
    np.testing.assert_allclose(full.toarray()[:, basis.local_indices], oracle, atol=1e-13)
    # No entry outside this rank's columns.
    outside = np.ones(basis.size, dtype=bool)
    outside[basis.local_indices] = False
    assert not full.toarray()[:, outside].any()


@pytest.mark.mpi
def test_local_columns_is_the_local_slice_of_the_default_result():
    comm = MPI.COMM_WORLD
    basis = _basis(_all_dets(), comm)
    op = _operator()
    sliced = build_sparse_matrix(basis, op)[:, basis.local_indices]
    local = build_sparse_matrix(basis, op, local_columns=True)
    assert local.format == sliced.format == "csc"
    assert np.array_equal(local.indptr, sliced.indptr)
    assert np.array_equal(local.indices, sliced.indices)
    assert np.array_equal(local.data, sliced.data)


@pytest.mark.mpi
def test_the_result_is_canonical():
    comm = MPI.COMM_WORLD
    basis = _basis(_all_dets(), comm)
    for local_columns in (False, True):
        m = build_sparse_matrix(basis, _operator(), local_columns=local_columns)
        assert m.has_sorted_indices
        assert m.has_canonical_format


@pytest.mark.mpi
def test_the_column_operator_peak_stays_below_its_budget(monkeypatch):
    """The build of the frozen-basis operator against what the fit check budgets for it.

    ``gf_solvers._CSR_BYTES_PER_ELEMENT`` per stored element plus the column pointers, and
    ``_CSR_BATCH_BYTES_PER_ELEMENT`` per element of one build batch: a build that exceeds them
    makes the check admit a matrix that does not fit. Batches are small, as in production, where
    a batch is a constant next to the matrix. Serial and distributed: the distributed build holds
    the batch's key objects until its collective lookup.
    """
    monkeypatch.setattr(basis_transcription, "_SPARSE_BUILD_BATCH", 5000)
    comm = MPI.COMM_WORLD
    basis = _basis(_all_dets(14), comm if comm.size > 1 else None, n_orb=14)
    op = _operator(14)
    basis_transcription.build_local_column_operator(basis, op)  # lazy lookup structures are built once
    tracemalloc.start()
    try:
        m = basis_transcription.build_local_column_operator(basis, op)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    n_local = len(basis.local_basis)
    budget = (
        m.nnz * gf_solvers._CSR_BYTES_PER_ELEMENT
        + 8 * (n_local + 1)
        + basis_transcription._SPARSE_BUILD_BATCH * gf_solvers._CSR_BATCH_BYTES_PER_ELEMENT
    )
    assert m.nnz > 20_000 / comm.size, "the fixture must be large enough for a per-element figure to mean anything"
    assert peak <= budget, f"{peak / m.nnz:.1f} B/element peak exceeds the {budget / m.nnz:.1f} the fit check budgets"


def test_a_strided_walk_visits_exactly_the_requested_local_states():
    basis = _basis(_all_dets(), None)
    op = _operator()
    full = list(basis_transcription.iter_local_operator_images(basis, op, 0))
    picked = [0, 7, 400, len(full) - 1]
    sampled = list(basis_transcription.iter_local_operator_images(basis, op, 0, indices=picked))
    assert len(sampled) == len(picked)
    for image, i in zip(sampled, picked):
        assert list(image.keys()) == list(full[i].keys())
        np.testing.assert_array_equal(np.asarray(image), np.asarray(full[i]))
    assert list(basis_transcription.iter_local_operator_images(basis, op, 0, indices=())) == []


@pytest.mark.mpi
@pytest.mark.parametrize("serial", [True, False])
def test_the_build_answers_the_gf_unit_queue_as_it_goes(monkeypatch, serial):
    """The build of a frozen-basis CSR runs for minutes with no Lanczos block in it; on the rank
    hosting the GF unit queue it must still answer the other colors' fetches. With one poke per
    ket every local ket is a poke."""
    monkeypatch.setattr(basis_transcription, "_QUEUE_POKE_KETS", 1)
    pokes = []
    monkeypatch.setattr(basis_transcription, "queue_progress", lambda: pokes.append(1))
    comm = MPI.COMM_SELF if serial else MPI.COMM_WORLD
    basis = _basis(_all_dets(), comm)
    build_sparse_matrix(basis, _operator())
    assert len(pokes) == len(basis.local_basis)


@pytest.mark.mpi
@pytest.mark.parametrize("batch", [None, 7])
@pytest.mark.parametrize("n_dets", [None, 2])
def test_the_column_operator_holds_the_same_matrix(monkeypatch, batch, n_dets):
    """``build_local_column_operator`` keeps the build's column blocks instead of one CSC: the same
    entries, int32 rows, one segment per batch. A 7-element batch splits the columns into many
    segments; two determinants leave ranks empty at -n 3."""
    if batch is not None:
        monkeypatch.setattr(basis_transcription, "_SPARSE_BUILD_BATCH", batch)
    comm = MPI.COMM_WORLD
    dets = _all_dets() if n_dets is None else _all_dets()[:n_dets]
    basis = _basis(dets, comm)
    op = _operator()
    column_op = basis_transcription.build_local_column_operator(basis, op)
    assert column_op.shape == (basis.size, len(basis.local_basis))
    assert all(rows.dtype == np.int32 for _, _, rows, _ in column_op.segments)
    if batch is not None and len(basis.local_basis) > 1:
        assert len(column_op.segments) > 1, "premise: a small batch splits the columns"
    np.testing.assert_allclose(column_op.tocsc().toarray(), _oracle_columns(basis, op, comm), atol=1e-13)


@pytest.mark.mpi
def test_bound_rows_are_the_exchange_buffer_rows():
    """``bind_slots`` maps a row owned by ``destinations[i]`` into the send buffer at
    ``send_row_offsets[i]`` and a row this rank owns into the self buffer after the send rows."""
    comm = MPI.COMM_WORLD
    basis = _basis(_all_dets(), comm)
    column_op = basis_transcription.build_local_column_operator(basis, _operator())
    counts = np.array(comm.allgather(len(basis.local_basis)), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    flags = column_op.owner_flags(counts, offsets)
    flags[comm.rank] = False
    destinations = [int(d) for d in np.flatnonzero(flags)]
    send_row_offsets = np.concatenate(([0], np.cumsum(counts[destinations]))).astype(np.int64)
    total_send = int(send_row_offsets[-1])
    before = [rows.astype(np.int64) for _, _, rows, _ in column_op.segments]
    column_op.bind_slots(comm.rank, offsets, destinations, send_row_offsets, total_send)
    for rows_before, (_, _, rows, _) in zip(before, column_op.segments):
        owner = np.searchsorted(offsets, rows_before, side="right") - 1
        expected = np.where(
            owner == comm.rank,
            total_send + rows_before - offsets[comm.rank],
            send_row_offsets[np.searchsorted(destinations, owner)] + rows_before - offsets[owner],
        )
        assert np.array_equal(rows, expected)
    column_op.bind_slots(comm.rank, offsets, destinations, send_row_offsets, total_send)  # same layout: no-op
    with pytest.raises(RuntimeError, match="different exchange layout"):
        column_op.bind_slots(comm.rank, offsets, destinations, send_row_offsets, total_send + 1)
