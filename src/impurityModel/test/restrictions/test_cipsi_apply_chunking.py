"""``GS_APPLY_ROW_CHUNKS``: the row-chunked path of ``CIPSISolver._apply_block_and_redistribute``.

The one-shot path holds the raw apply output, its packed send buffer, the receive buffer and the
merged block at once -- ~6x the owned candidate block, the selection round's peak and the step
that took the SrMnO3 double-counting job from 2.4 to 5.8 GiB in one cycle
(``doc/plans/dc_smo_memory.md``, round 6). The chunked path applies one chunk of the reference
rows at a time and accumulates the redistributed pieces.

What is pinned here: the chunked result has the *same support* as the one-shot one and agrees
to roundoff (summation order differs when a candidate is reached from rows in different
chunks); a chunk count larger than a rank's row count (empty chunks, including ranks that own
nothing) is handled without desynchronizing the collectives; and the knob unset is the one-shot
path bit for bit.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3]]}, {0: [[4, 5]]})
N_SPIN_ORBITALS = 6
N_ELECTRONS = 3
_HOP = 0.15 + 0.05j


@pytest.fixture(autouse=True)
def _knob_unset(monkeypatch):
    monkeypatch.delenv("GS_APPLY_ROW_CHUNKS", raising=False)


def _det(occupied):
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _make_solver(comm, n_basis=8):
    basis = Basis(IMPURITY_ORBITALS, BATH_STATES, nominal_impurity_occ={0: 1}, comm=comm, verbose=False)
    all_dets = [_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), N_ELECTRONS)]
    basis.add_states(all_dets[:n_basis])
    return CIPSISolver(basis), all_dets[:n_basis]


def _hamiltonian():
    terms = {((i, "c"), (i, "a")): 0.3 * (i + 1) for i in range(N_SPIN_ORBITALS)}
    for a, b in ((0, 2), (1, 3), (2, 4), (3, 5), (0, 4)):
        terms[((a, "c"), (b, "a"))] = _HOP
        terms[((b, "c"), (a, "a"))] = _HOP.conjugate()
    return ManyBodyOperator(terms)


def _psi_ref_block(solver, basis_dets, widths=(1, 2, 3, 5)):
    """Reference columns over the rank's *own* determinants (as `expand` hands them over),
    with overlapping supports of increasing size."""
    local = set(solver.basis.local_basis)
    cols = []
    for w in widths:
        amps = {d: 1.0 + 0.3j * (i + 1) for i, d in enumerate(basis_dets[: w + 1]) if d in local}
        cols.append(ManyBodyState(amps, width=1))
    return cols


def _run(comm, n_chunks, cutoff, monkeypatch):
    solver, basis_dets = _make_solver(comm)
    H = _hamiltonian()
    psi_ref = _psi_ref_block(solver, basis_dets)
    if n_chunks is None:
        monkeypatch.delenv("GS_APPLY_ROW_CHUNKS", raising=False)
    else:
        monkeypatch.setenv("GS_APPLY_ROW_CHUNKS", str(n_chunks))
    assert config.GS_APPLY_ROW_CHUNKS.get() == n_chunks
    return solver._apply_block_and_redistribute(H, psi_ref, cutoff)


def _assert_same(chunked, one_shot):
    assert chunked.width == one_shot.width
    assert chunked.keys() == one_shot.keys(), "chunking changed the candidate support"
    np.testing.assert_allclose(np.asarray(chunked), np.asarray(one_shot), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n_chunks", [1, 2, 3, 7, 50])
@pytest.mark.parametrize("cutoff", [0.0, 0.001])
def test_chunked_matches_one_shot_serial(n_chunks, cutoff, monkeypatch):
    one_shot = _run(None, None, cutoff, monkeypatch)
    chunked = _run(None, n_chunks, cutoff, monkeypatch)
    _assert_same(chunked, one_shot)
    if n_chunks == 1:
        # 1 chunk IS the one-shot path: bit for bit, not just to tolerance.
        np.testing.assert_array_equal(np.asarray(chunked), np.asarray(one_shot))


@pytest.mark.parametrize("n_chunks, expected_applies", [(None, 1), (1, 1), (3, 3), (50, 50)])
def test_apply_count_follows_the_knob(n_chunks, expected_applies, monkeypatch):
    """Unset and 1 are the one-shot path (a single operator apply); `n` chunks apply exactly `n`
    times, empty chunks included -- that count is what keeps the collective redistributions in
    step across ranks, so it must not depend on how many rows a rank happens to own."""
    calls = []
    original = CIPSISolver._apply_and_prune_columns

    def counting(H, block, cutoff):
        calls.append(len(block))
        return original(H, block, cutoff)

    monkeypatch.setattr(CIPSISolver, "_apply_and_prune_columns", staticmethod(counting))
    _run(None, n_chunks, 0.0, monkeypatch)
    assert len(calls) == expected_applies


@pytest.mark.mpi
@pytest.mark.parametrize("n_chunks", [2, 3, 50])
def test_chunked_matches_one_shot_mpi(n_chunks, monkeypatch):
    """50 chunks over a handful of rows per rank means most chunks are empty on every rank, and
    at 3 ranks some rank may own no reference rows at all -- every one of them must still make
    the same number of collective redistributions."""
    comm = MPI.COMM_WORLD
    one_shot = _run(comm, None, 0.001, monkeypatch)
    chunked = _run(comm, n_chunks, 0.001, monkeypatch)
    _assert_same(chunked, one_shot)
    # And the result is a consistent global object: every rank agrees on the total row count.
    totals = comm.allgather(len(chunked))
    assert sum(totals) == sum(comm.allgather(len(one_shot)))
