"""Walks that grow a basis from images, and the Lanczos step's global truncation, must not depend on how
the determinants are spread over ranks.

``Basis.expand`` and the CIPSI symmetry closure discover determinants wave by wave. With a rank-local
membership test (``contains_local``) a rank treated a determinant another rank owns as new: it walked
on through it, and ranks reaching the same image counted it once each toward ``truncation_threshold``,
so a capped expansion stopped early on more ranks. ``Basis.new_owned_keys`` routes each wave's images
to their owners first. The oracle is the serial run.
"""

import hashlib
import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczos import _keep_global_top
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

_multirank = pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs determinants owned by more than one rank")


def _det(occupied):
    """Two-byte determinant of the 12-orbital model (MSB-first: orbital i = bit 7 - i % 8)."""
    b = [0, 0]
    for i in occupied:
        b[i // 8] |= 1 << (7 - i % 8)
    return SlaterDeterminant.from_bytes(bytes(b))


def _model():
    """12 spin-orbitals, two impurity orbitals hybridizing with ten bath orbitals, no degeneracy."""
    terms = {((o, "c"), (o, "a")): -1.0 + 0.3 * o for o in range(12)}
    for a in (0, 1):
        for b in range(2, 12):
            terms[((a, "c"), (b, "a"))] = 0.4
            terms[((b, "c"), (a, "a"))] = 0.4
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = 3.0
    return ManyBodyOperator(terms)


_SEEDS = {
    "two": ([0, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]),
    "three": ([0, 1, 2, 3, 4, 7], [0, 2, 3, 4, 8, 9], [1, 5, 6, 7, 10, 11]),
}


def _basis(seeds, comm, cap=np.inf):
    return Basis(
        {0: [[0, 1]]},
        ({0: [list(range(2, 7))]}, {0: [list(range(7, 12))]}),
        initial_basis=[_det(o) for o in _SEEDS[seeds]],
        comm=comm,
        truncation_threshold=cap,
        verbose=False,
    )


def _content(basis):
    keys = sorted(bytes(k.to_bytearray()) for k in basis.local_basis)
    if basis.is_distributed:
        keys = sorted(k for part in basis.comm.allgather(keys) for k in part)
    return len(keys), hashlib.md5(b"".join(keys)).hexdigest()


@pytest.mark.mpi
@_multirank
@pytest.mark.parametrize(
    "seeds, cap",
    # The capped cases sit between the true number of new determinants and the over-count: the
    # rank-local walk stopped there at 62 of 362 (two seeds, cap 370) and 3 of 169 (three, cap 170).
    [("two", np.inf), ("three", np.inf), ("two", 370), ("two", 400), ("three", 170)],
)
def test_expand_does_not_depend_on_the_rank_count(seeds, cap):
    distributed = _basis(seeds, MPI.COMM_WORLD, cap)
    distributed.expand(_model(), max_it=4)
    serial = _basis(seeds, None, cap)
    serial.expand(_model(), max_it=4)
    assert _content(distributed) == _content(serial)


@pytest.mark.mpi
@_multirank
def test_new_owned_keys_reports_each_new_determinant_once_by_its_owner():
    """Every rank offers every candidate (duplicates across ranks), half of them already in the basis
    and owned anywhere: the owners report exactly the other half, each once."""
    comm = MPI.COMM_WORLD
    basis = _basis("two", comm)
    basis.expand(_model(), max_it=2)
    held = {bytes(k.to_bytearray()) for part in comm.allgather(list(basis.local_basis)) for k in part}
    candidates = [_det(o) for o in ([0, 1, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])]
    expected = {bytes(c.to_bytearray()) for c in candidates} - held
    assert expected and len(expected) < len(candidates), "the fixture must mix held and new candidates"

    mine = basis.new_owned_keys(candidates)
    reported = [bytes(k.to_bytearray()) for part in comm.allgather(list(mine)) for k in part]
    assert sorted(reported) == sorted(expected)

    excluded = basis.new_owned_keys(candidates, exclude=mine)
    assert comm.allreduce(len(excluded), op=MPI.SUM) == 0


def _tied_state():
    """30 determinants of the 12-orbital model in groups of 5 equal amplitudes."""
    dets = [_det(o) for o in itertools.islice(itertools.combinations(range(12), 6), 30)]
    return {d: complex(1.0 + i // 5) for i, d in enumerate(dets)}


def _top_by_oracle(amps, k):
    """Largest |amp| first, equal ones in determinant-key order."""
    ranked = sorted(amps, key=lambda d: (-abs(amps[d]), bytes(d.to_bytearray())))
    return {bytes(d.to_bytearray()) for d in ranked[:k]}


@pytest.mark.parametrize("k", [3, 5, 12, 29])
def test_keep_global_top_is_the_exact_top_k_with_ties_in_key_order(k):
    amps = _tied_state()
    st = ManyBodyState(dict(amps), width=1)
    _keep_global_top(st, k, None)
    assert {bytes(d.to_bytearray()) for d in st.keys()} == _top_by_oracle(amps, k)


@pytest.mark.mpi
@_multirank
@pytest.mark.parametrize("k", [3, 5, 12, 29])
def test_keep_global_top_does_not_depend_on_the_rank_count(k):
    """The Lanczos step's truncation_threshold: the same rows kept on any layout (owner-distributed)."""
    comm = MPI.COMM_WORLD
    amps = _tied_state()
    basis = _basis("two", comm)
    (st,) = basis.redistribute_psis(ManyBodyState(dict(amps) if comm.rank == 0 else {}, width=1))
    _keep_global_top(st, k, comm)
    kept = {bytes(d.to_bytearray()) for part in comm.allgather(list(st.keys())) for d in part}
    assert kept == _top_by_oracle(amps, k)
