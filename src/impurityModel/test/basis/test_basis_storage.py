"""Unit tests for the distributed determinant storage of :class:`Basis`.

``Basis`` owns its storage directly (the rank-local sorted determinant list, the
state -> global-index dict, and hash-routed distributed lookups), so the storage
API is exercised through ``Basis``. Serial tests use ``MPI.COMM_SELF`` so they behave identically
no matter how many ranks ``pytest`` is launched with; one MPI-marked check covers
the distributed size/lookup path.
"""

import pytest
from mpi4py import MPI

from impurityModel.ed.manybody_basis import Basis
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


def test_chunk_count_is_ceil_of_n_bytes_not_a_floor_division():
    """``n_bytes`` counts BYTES, so the chunk count is ``ceil(n_bytes / 8)``.

    Pins the arithmetic behind the routed index lookup. ``n_bytes // 8`` is 0 for every basis
    under 64 spin-orbitals, which makes a buffer exchange send zero bytes while still counting
    determinants -- the receiving rank then answers nothing for determinants it was told to
    expect. That failure showed up as an MPI hang rather than an error, so it is pinned here
    as a local assertion that fails loudly instead.
    """
    basis = _make_basis(_make_states())
    chunks = (basis.n_bytes + 7) // 8
    assert chunks >= 1
    assert basis.n_bytes // 8 != chunks or basis.n_bytes % 8 == 0
    for state in basis.local_basis:
        assert len(state) == chunks, f"determinant has {len(state)} chunks, basis implies {chunks}"


def test_routed_index_lookup_rejects_a_zero_chunk_width():
    """The guard that turns the hang above into an immediate, local error.

    It has to raise *before* any collective runs: raising afterwards leaves the other ranks
    waiting in an exchange this one has already left.
    """
    from impurityModel.ed.mpi_comm import routed_index_lookup

    with pytest.raises(ValueError, match="chunks_per_state"):
        routed_index_lookup([], 0, lambda buf: None, MPI.COMM_SELF)


# --------------------------------------------------------------------------- #
# Iteration streams; it does not rebuild the determinant list
# --------------------------------------------------------------------------- #


class _NoMaterializeKeys:
    """A key block that forwards everything except ``keys()``.

    Pins the mechanism rather than the symptom: a memory assertion on iteration would be a
    threshold that drifts, but "iteration never calls ``keys()``" is exactly the property
    that makes it free, and it fails loudly the moment the view goes back to materializing.
    """

    def __init__(self, keys):
        self._inner = keys

    def __len__(self):
        return len(self._inner)

    def __contains__(self, item):
        return item in self._inner

    def key_at(self, i):
        return self._inner.key_at(i)

    def keys(self):
        raise AssertionError("iterating the local basis materialized the whole determinant list")


def test_iterating_the_local_basis_does_not_materialize_the_determinant_list():
    from impurityModel.ed.manybody_basis import _LocalBasisView

    basis = _make_basis(_make_states())
    expected = [basis.local_basis[i] for i in range(len(basis.local_basis))]

    view = _LocalBasisView(_NoMaterializeKeys(basis._keys))
    assert list(view) == expected


def test_iterating_the_local_basis_yields_sorted_order():
    """The order is the contract: ``local_indices`` is ``range(offset, offset + len)``, so the
    loop position of an iteration *is* the determinant's global index."""
    basis = _make_basis(_make_states())
    walked = list(basis.local_basis)
    assert walked == sorted(walked)
    assert walked == [basis.local_basis[i] for i in range(len(walked))]
    assert len(walked) == len(basis.local_basis)


def test_growing_the_basis_mid_iteration_raises_instead_of_skipping():
    """``add_states`` merges into the key block in place, so a streaming walk would silently
    skip or repeat determinants that move. It must raise, the way a ``dict`` does."""
    basis = _make_basis(_make_states())
    new = _sd(0x08)
    assert new not in basis.local_basis

    it = iter(basis.local_basis)
    next(it)
    basis.add_states([new])
    with pytest.raises(RuntimeError, match="changed size during iteration"):
        next(it)


def test_iteration_is_repeatable():
    """``__iter__`` hands back a fresh walk each time; a one-shot iterator would break every
    caller that reads the basis twice."""
    basis = _make_basis(_make_states())
    assert list(basis.local_basis) == list(basis.local_basis)
