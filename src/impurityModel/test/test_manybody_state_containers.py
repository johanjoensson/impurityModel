"""Unit tests for ``impurityModel.ed.manybody_state_containers``.

Covers the module-level helpers (:func:`batched`, :func:`hash_key`) and the public
API of :class:`SimpleDistributedStateContainer` in its serial configuration
(``MPI.COMM_SELF`` => ``is_distributed`` is False), plus one MPI-marked check of the
distributed size/lookup path.

Serial tests use ``MPI.COMM_SELF`` so they behave identically no matter how many
ranks ``pytest`` is launched with.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import SlaterDeterminant
from impurityModel.ed.manybody_state_containers import (
    SimpleDistributedStateContainer,
    batched,
    hash_key,
)

N_BYTES = 8


def _sd(byte0):
    """A single-orbital SlaterDeterminant with bit ``byte0`` set in the first byte."""
    return SlaterDeterminant.from_bytes(bytes([byte0]) + b"\x00" * (N_BYTES - 1))


def _make_states():
    # Distinct single-bit determinants; order is decided by SlaterDeterminant's own <.
    return [_sd(0x80), _sd(0x40), _sd(0x20), _sd(0x10)]


def _serial_container(states):
    return SimpleDistributedStateContainer(states, bytes_per_state=N_BYTES, comm=MPI.COMM_SELF, verbose=False)


# --------------------------------------------------------------------------- #
# batched
# --------------------------------------------------------------------------- #
def test_batched_splits_into_chunks():
    assert list(batched("ABCDEFG", 3)) == [("A", "B", "C"), ("D", "E", "F"), ("G",)]


def test_batched_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        list(batched("ABC", 0))


# --------------------------------------------------------------------------- #
# hash_key
# --------------------------------------------------------------------------- #
def test_hash_key_delegates_to_get_hash():
    s = _sd(0x80)
    assert hash_key(s) == s.get_hash()


# --------------------------------------------------------------------------- #
# construction / len / ordering
# --------------------------------------------------------------------------- #
def test_len_counts_unique_states():
    states = _make_states()
    c = _serial_container(states)
    assert len(c) == len(states)


def test_duplicates_are_deduplicated():
    states = _make_states()
    c = _serial_container(states + states)
    assert len(c) == len(states)


def test_iteration_is_sorted():
    states = _make_states()
    c = _serial_container(states)
    assert list(c) == sorted(set(states))


# --------------------------------------------------------------------------- #
# indexing
# --------------------------------------------------------------------------- #
def test_getitem_int_and_index_roundtrip():
    states = _make_states()
    c = _serial_container(states)
    for i in range(len(c)):
        assert c.index(c[i]) == i


def test_getitem_slice():
    states = _make_states()
    c = _serial_container(states)
    ordered = sorted(set(states))
    assert list(c[1:3]) == ordered[1:3]


def test_index_missing_raises():
    c = _serial_container(_make_states())
    with pytest.raises(ValueError):
        c.index(_sd(0x01))  # never inserted


# --------------------------------------------------------------------------- #
# membership
# --------------------------------------------------------------------------- #
def test_contains_single():
    states = _make_states()
    c = _serial_container(states)
    assert states[0] in c
    assert _sd(0x01) not in c


def test_contains_sequence_returns_bools():
    states = _make_states()
    c = _serial_container(states)
    got = list(c.contains([states[0], _sd(0x01), states[2]]))
    assert got == [True, False, True]


# --------------------------------------------------------------------------- #
# mutation
# --------------------------------------------------------------------------- #
def test_add_states_extends_and_dedups():
    states = _make_states()
    c = _serial_container(states[:2])
    assert len(c) == 2
    c.add_states([states[2], states[0]])  # one new, one already present
    assert len(c) == 3
    assert states[2] in c


def test_clear_empties_container():
    c = _serial_container(_make_states())
    c.clear()
    assert len(c) == 0
    assert list(c) == []


# --------------------------------------------------------------------------- #
# distributed path
# --------------------------------------------------------------------------- #
@pytest.mark.mpi
def test_distributed_global_size_and_lookup():
    comm = MPI.COMM_WORLD
    # Each rank contributes distinct states; the container must agree on the global
    # size and be able to look up every state from every rank.
    base = 0x80 >> comm.rank
    states = [SlaterDeterminant.from_bytes(bytes([base]) + bytes([r]) + b"\x00" * (N_BYTES - 2)) for r in range(3)]
    c = SimpleDistributedStateContainer(states, bytes_per_state=N_BYTES, comm=comm, verbose=False)

    expected_total = 3 * comm.size
    assert len(c) == expected_total
    # Every state this rank created is present in the (possibly distributed) container.
    for s in states:
        assert s in c
