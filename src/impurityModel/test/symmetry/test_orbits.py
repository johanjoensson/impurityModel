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
