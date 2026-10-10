r"""``basis_transcription.build_sparse_matrix``: the matrix, its layout, and its memory.

The build assembles the CSC directly (images arrive one ket at a time, so the entries are already
grouped by column). These tests pin that against an oracle that shares no code with it -- the full
operator applied to every determinant in Python and indexed by position -- and pin the memory
the frozen-basis CSR fit check (``gf_solvers._CSR_BYTES_PER_ELEMENT``) relies on:

* every rank's stored columns equal the oracle's, whatever the rank count, including ranks that own
  no determinant;
* ``local_columns=True`` is exactly the ``(N, N_local)`` slice of the default ``(N, N)`` result;
* the result is canonical (sorted, duplicate-free), as the old COO route made it;
* the peak, ``tocsr()`` included, stays below the per-element constant the fit check budgets.
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


def test_the_peak_stays_below_the_budgeted_bytes_per_element(monkeypatch):
    """Build plus the CSR conversion the kernel needs, against ``_CSR_BYTES_PER_ELEMENT``.

    The fit check budgets this constant per stored element, so a build that exceeds it makes the
    check admit a matrix that does not fit. Batches are small, as in production, where a batch is
    a constant next to the matrix; one batch holding the whole image would measure the batch.
    """
    monkeypatch.setattr(basis_transcription, "_SPARSE_BUILD_BATCH", 5000)
    basis = _basis(_all_dets(14), None, n_orb=14)
    op = _operator(14)
    build_sparse_matrix(basis, op, local_columns=True)  # lazy lookup structures are built once
    tracemalloc.start()
    try:
        m = build_sparse_matrix(basis, op, local_columns=True).tocsr()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    per_element = peak / m.nnz
    assert m.nnz > 20_000, "the fixture must be large enough for a per-element figure to mean anything"
    assert (
        per_element <= gf_solvers._CSR_BYTES_PER_ELEMENT
    ), f"{per_element:.1f} B/element peak exceeds the {gf_solvers._CSR_BYTES_PER_ELEMENT} B the CSR fit check budgets"


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
