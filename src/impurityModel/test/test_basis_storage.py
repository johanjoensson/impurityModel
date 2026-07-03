"""Unit tests for the distributed determinant storage of :class:`Basis`.

Formerly ``test_manybody_state_containers.py``: the state-container hierarchy was
dissolved into ``Basis`` (its determinant list, state -> global-index dict, and
hash-routed distributed lookups), so the same storage API is exercised through
``Basis`` directly. Serial tests use ``MPI.COMM_SELF`` so they behave identically
no matter how many ranks ``pytest`` is launched with; one MPI-marked check covers
the distributed size/lookup path.
"""

import pytest
from mpi4py import MPI

from impurityModel.ed.manybody_basis import Basis, batched
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

N_BYTES = 8
N_SPIN_ORBITALS = 8 * N_BYTES


def _sd(byte0):
    """A single-orbital SlaterDeterminant with bit ``byte0`` set in the first byte."""
    return SlaterDeterminant.from_bytes(bytes([byte0]) + b"\x00" * (N_BYTES - 1))


def _make_states():
    # Distinct single-bit determinants; order is decided by SlaterDeterminant's own <.
    return [_sd(0x80), _sd(0x40), _sd(0x20), _sd(0x10)]


def _make_basis(states, comm=MPI.COMM_SELF):
    return Basis(
        impurity_orbitals={0: [list(range(N_SPIN_ORBITALS))]},
        bath_states=({0: []}, {0: []}),
        initial_basis=states,
        comm=comm,
        verbose=False,
    )


# --------------------------------------------------------------------------- #
# batched
# --------------------------------------------------------------------------- #
def test_batched_splits_into_chunks():
    assert list(batched("ABCDEFG", 3)) == [("A", "B", "C"), ("D", "E", "F"), ("G",)]


def test_batched_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        list(batched("ABC", 0))


# --------------------------------------------------------------------------- #
# construction / len / ordering
# --------------------------------------------------------------------------- #
def test_len_counts_unique_states():
    states = _make_states()
    b = _make_basis(states)
    assert len(b) == len(states)


def test_duplicates_are_deduplicated():
    states = _make_states()
    b = _make_basis(states + states)
    assert len(b) == len(states)


def test_iteration_is_sorted():
    states = _make_states()
    b = _make_basis(states)
    assert list(b) == sorted(set(states))


# --------------------------------------------------------------------------- #
# indexing
# --------------------------------------------------------------------------- #
def test_getitem_int_and_index_roundtrip():
    states = _make_states()
    b = _make_basis(states)
    for i in range(len(b)):
        assert b.index(b[i]) == i


def test_getitem_slice():
    states = _make_states()
    b = _make_basis(states)
    ordered = sorted(set(states))
    assert list(b[1:3]) == ordered[1:3]


def test_index_missing_raises():
    b = _make_basis(_make_states())
    with pytest.raises(ValueError):
        b.index(_sd(0x01))  # never inserted


# --------------------------------------------------------------------------- #
# membership
# --------------------------------------------------------------------------- #
def test_contains_single():
    states = _make_states()
    b = _make_basis(states)
    assert states[0] in b
    assert _sd(0x01) not in b


def test_contains_sequence_returns_bools():
    states = _make_states()
    b = _make_basis(states)
    got = list(b.contains([states[0], _sd(0x01), states[2]]))
    assert got == [True, False, True]


# --------------------------------------------------------------------------- #
# mutation
# --------------------------------------------------------------------------- #
def test_add_states_extends_and_dedups():
    states = _make_states()
    b = _make_basis(states[:2])
    assert len(b) == 2
    b.add_states([states[2], states[0]])  # one new, one already present
    assert len(b) == 3
    assert states[2] in b


def test_clear_empties_basis():
    b = _make_basis(_make_states())
    b.clear()
    assert len(b) == 0
    assert list(b) == []


# --------------------------------------------------------------------------- #
# distributed path
# --------------------------------------------------------------------------- #
@pytest.mark.mpi
def test_distributed_global_size_and_lookup():
    comm = MPI.COMM_WORLD
    # Each rank contributes distinct states; the basis must agree on the global
    # size and be able to look up every state from every rank.
    base = 0x80 >> comm.rank
    states = [SlaterDeterminant.from_bytes(bytes([base]) + bytes([r]) + b"\x00" * (N_BYTES - 2)) for r in range(3)]
    b = _make_basis(states, comm=comm)

    expected_total = 3 * comm.size
    assert len(b) == expected_total
    # Every state this rank created is present in the (possibly distributed) basis.
    for s in states:
        assert s in b
