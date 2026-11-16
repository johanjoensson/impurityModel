"""``trlm._root_bcast``: a root broadcast that says where it came from, so a rank waiting in a different
collective raises an error naming both positions instead of unpickling another call's bytes."""

import pickle

import numpy as np
import pytest

from impurityModel.ed.trlm import CollectiveDesync, _checksum, _root_bcast


class _Comm:
    """A non-root rank whose ``bcast`` hands back (or raises) whatever the test supplies."""

    rank = 1
    size = 2

    def __init__(self, incoming):
        self.incoming = incoming

    def bcast(self, obj, root=0):
        assert obj is None, "a non-root rank must not pickle a payload"
        if isinstance(self.incoming, Exception):
            raise self.incoming
        return self.incoming


def _message(site, key, payload):
    return (site, key, tuple(_checksum(a) if isinstance(a, np.ndarray) else None for a in payload), payload)


def test_no_communicator_is_the_identity():
    payload = (np.arange(3.0), True)
    assert _root_bcast(None, "extract", (3,), payload) is payload


def test_a_matching_broadcast_returns_rank_0s_payload():
    theta, s = np.arange(3.0), np.eye(3, dtype=complex)
    got = _root_bcast(_Comm(_message("extract", (3,), (theta, s))), "extract", (3,), (None, None))
    np.testing.assert_array_equal(got[0], theta)
    np.testing.assert_array_equal(got[1], s)


@pytest.mark.parametrize(
    "incoming, said",
    [
        (_message("restart", (2, 3), (np.arange(3.0),)), "rank 0 broadcast from restart (2, 3)"),
        (_message("extract", (4,), (np.arange(4.0),)), "rank 0 broadcast from extract (4,)"),
        (pickle.UnpicklingError("invalid load key, 'D'."), "could not be decoded"),
        (np.arange(3.0), "received a ndarray"),
        ((True, False), "received a tuple"),
    ],
)
def test_a_broadcast_from_another_position_names_both(incoming, said):
    with pytest.raises(CollectiveDesync, match="rank 1") as err:
        _root_bcast(_Comm(incoming), "extract", (3,), (None,))
    assert said in str(err.value)
    assert "extract (3,)" in str(err.value)


def test_a_damaged_array_is_told_apart_from_a_desynchronized_rank():
    site, key, sums, payload = _message("extract", (3,), (np.arange(3.0), np.eye(3, dtype=complex)))
    damaged = (payload[0], payload[1] + 1e-3)
    with pytest.raises(CollectiveDesync, match="item 1 .* fails its checksum") as err:
        _root_bcast(_Comm((site, key, sums, damaged)), "extract", (3,), (None, None))
    assert "the ranks are in step" in str(err.value)


@pytest.mark.mpi
def test_every_rank_receives_rank_0s_arrays():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    theta = np.arange(5.0) + (0 if comm.rank == 0 else 1e-13)  # a rank-local eigh's last bits
    s = np.eye(5, dtype=complex) * (1 if comm.rank == 0 else 1 + 1e-13)
    got_theta, got_s = _root_bcast(comm, "extract", (5,), (theta, s))
    np.testing.assert_array_equal(got_theta, np.arange(5.0))
    np.testing.assert_array_equal(got_s, np.eye(5, dtype=complex))
