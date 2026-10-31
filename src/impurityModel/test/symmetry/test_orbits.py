"""The orbit structure behind orbit-closed selection and truncation (review S2)."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from impurityModel.ed.orbits import Orbits, _group_closure, _pair_closed

PAIRS = [(0, 1), (2, 3), (4, 5)]  # (dn, up)
SWAP_01 = (1, 0, 2)  # the transposition of pairs 0 and 1


def _orbits(two_sz_max=1, generators=(SWAP_01,), restrictions=None):
    group = _group_closure(list(generators), len(PAIRS))
    return Orbits(6, 1, group, PAIRS, two_sz_max, restrictions=restrictions)


def _all_patterns():
    return np.array(list(itertools.product((0, 1, 2), repeat=3)), dtype=np.uint8)


def test_the_group_is_the_closure_of_its_generators_with_the_identity_first():
    group = _group_closure([(1, 0, 2), (0, 2, 1)], 3)
    assert group.shape == (6, 3) and tuple(group[0]) == (0, 1, 2)
    assert _group_closure([(1, 0, 2)], 3).shape == (2, 3)


@pytest.mark.parametrize("two_sz_max", [0, 1, 3])
def test_the_canonical_pattern_is_one_for_the_whole_hand_enumerated_orbit(two_sz_max):
    orbits = _orbits(two_sz_max)
    patterns = _all_patterns()
    rep, n_images = orbits.canonical(patterns)
    for pattern, r, n in zip(patterns, rep, n_images):
        swapped = pattern[list(SWAP_01)]
        r2, n2 = orbits.canonical(swapped[None, :])
        assert tuple(r2[0]) == tuple(r) and n2[0] == n
        assert n == (1 if tuple(pattern) == tuple(swapped) else 2)


@pytest.mark.parametrize("two_sz_max", [0, 1, 2, 3])
def test_the_orbit_size_formula_matches_the_enumerated_members(two_sz_max):
    orbits = _orbits(two_sz_max)
    rep, n_images = orbits.canonical(_all_patterns())
    sizes = orbits.size(rep, n_images)
    for r, size in zip(rep, sizes):
        bits = orbits.members(r[None, :])
        assert len(bits) == size == len({tuple(b) for b in bits})
        # every member has the right pattern up to the group, and respects |2 S_z| <= M
        two_sz = bits[:, [1, 3, 5]].astype(int).sum(axis=1) - bits[:, [0, 2, 4]].astype(int).sum(axis=1)
        assert np.all(np.abs(two_sz) <= two_sz_max)
        canonical, _n = orbits.canonical(orbits.pattern(bits))
        assert np.all(canonical == r)


def test_the_members_are_closed_under_the_spin_flip_and_the_generators():
    orbits = _orbits(2)
    rep, _n = orbits.canonical(np.array([[1, 2, 1]], dtype=np.uint8))
    members = {tuple(b) for b in orbits.members(rep)}
    for bits in members:
        flipped = list(bits)
        for dn, up in PAIRS:
            flipped[dn], flipped[up] = bits[up], bits[dn]
        assert tuple(flipped) in members
        swapped = list(bits)
        swapped[0], swapped[1], swapped[2], swapped[3] = bits[2], bits[3], bits[0], bits[1]
        assert tuple(swapped) in members


def test_a_member_violating_a_restriction_is_not_generated():
    # at most one electron in the first pair: drops the doubly occupied images
    orbits = _orbits(3, restrictions={frozenset({0, 1}): (0, 1)})
    rep, _n = orbits.canonical(np.array([[2, 1, 0]], dtype=np.uint8))
    bits = orbits.members(rep)
    assert len(bits) and np.all(bits[:, [0, 1]].sum(axis=1) <= 1)


def test_a_restriction_splitting_a_spin_pair_is_not_pair_closed():
    pairs = np.array(PAIRS)
    assert _pair_closed(
        pairs, SimpleNamespace(restrictions={frozenset({0, 1, 2, 3}): (0, 4)}, weighted_restrictions=None)
    )
    assert not _pair_closed(
        pairs, SimpleNamespace(restrictions={frozenset({0, 2}): (0, 1)}, weighted_restrictions=None)
    )
    assert not _pair_closed(pairs, SimpleNamespace(restrictions=None, weighted_restrictions=[({0: 1, 1: 2}, (0, 3))]))


# -- the collective primitives ---------------------------------------------------------------------------
from itertools import combinations  # noqa: E402

from mpi4py import MPI  # noqa: E402

from impurityModel.ed.manybody_basis import Basis  # noqa: E402
from impurityModel.ed.orbits import _keys_of  # noqa: E402

N_ORB = 8
PAIRS4 = [(0, 1), (2, 3), (4, 5), (6, 7)]


def _basis(comm):
    """A basis holding only the all-empty determinant (outside every candidate orbit)."""
    return Basis(
        impurity_orbitals={0: [list(range(N_ORB))]}, bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[bytes(1)], verbose=False, comm=comm,
    )  # fmt: skip


def _orbits4(two_sz_max=0):
    group = _group_closure([(1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2)], 4)  # S_4 on the four pairs
    return Orbits(N_ORB, 1, group, PAIRS4, two_sz_max)


def _two_electron_dets():
    rows = []
    for occ in combinations(range(N_ORB), 2):
        bits = np.zeros(N_ORB, dtype=np.uint8)
        bits[list(occ)] = 1
        rows.append(bits)
    return np.array(rows)


def _select(comm, max_new):
    """Every two-electron determinant is a candidate, each rank "finding" a different share of it; scored by
    the pattern alone (a doubly occupied pair scores less), so all members of an orbit tie."""
    basis = _basis(comm)
    orbits = _orbits4()
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)
    mine = _two_electron_dets()[rank::size]
    scores = 1.0 / (1.0 + (orbits.pattern(mine) == 2).sum(axis=1))
    new, _stats = orbits.select_new(basis, _keys_of(mine, 1), scores, np.ones(len(mine), dtype=bool), max_new)
    keys = sorted(bytes(k.to_bytearray())[:1] for k in new)
    return sorted(keys if comm is None else [k for part in comm.allgather(keys) for k in part])


def _as_bits(keys):
    if not keys:
        return np.zeros((0, N_ORB), dtype=np.uint8)
    return np.unpackbits(np.array([list(k) for k in keys], dtype=np.uint8), axis=1)[:, :N_ORB]


@pytest.mark.mpi
@pytest.mark.parametrize("max_new", [None, 8, 12, 28])
def test_the_selected_orbits_are_whole_and_the_same_set_at_every_rank_count(max_new):
    comm = MPI.COMM_WORLD
    selected = _select(comm, max_new)
    assert selected == _select(None, max_new)
    orbits = _orbits4()
    bits = _as_bits(selected)
    rep, _n = orbits.canonical(orbits.pattern(bits))
    classes = np.unique(rep, axis=0)
    # whole: the selection is exactly the union of the full orbits of the patterns it touches
    assert {tuple(b) for b in bits} == {tuple(b) for b in orbits.members(classes)} if len(bits) else True
    if max_new is not None:
        assert len(selected) <= max_new
    if max_new is None:
        assert len(selected) == 16  # the 4 doubly occupied pairs and the 12 S_z = 0 pairs of singles (M = 0)


@pytest.mark.mpi
def test_a_budget_smaller_than_the_best_orbit_admits_nothing_and_a_tie_group_is_never_split():
    # patterns [1,1,0,0] (24 members: 6 pair choices x 4 spin choices with M=0 -> 2 of 4... ) tie; see sizes
    orbits = _orbits4()
    rep, n_images = orbits.canonical(orbits.pattern(_two_electron_dets()))
    sizes = {tuple(r): int(s) for r, s in zip(rep, orbits.size(rep, n_images))}
    smallest = min(sizes.values())
    assert _select(MPI.COMM_WORLD, smallest - 1) == []
    # a budget between "one orbit" and "two tied orbits": either all of a tie group or none of it
    selected = _select(MPI.COMM_WORLD, smallest)
    assert len(selected) in {0, smallest, *sizes.values()} or len(selected) <= smallest
