"""window_dimension: exact determinant count of an occupation window."""

import random
from itertools import combinations
from math import comb

import pytest

from impurityModel.ed.basis_restrictions import window_dimension


def _brute(window, n_electrons, n_orbitals):
    return sum(
        1
        for occ in combinations(range(n_orbitals), n_electrons)
        if all(lo <= len(key & set(occ)) <= hi for key, (lo, hi) in (window or {}).items())
    )


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_on_overlapping_windows(seed):
    rng = random.Random(seed)
    for _ in range(60):
        n = rng.randint(1, 10)
        window = {}
        for _ in range(rng.randint(0, 4)):
            key = frozenset(rng.sample(range(n), rng.randint(1, n)))
            lo = rng.randint(0, len(key))
            window[key] = (lo, rng.randint(lo, len(key)))
        n_electrons = rng.randint(0, n)
        assert window_dimension(window, n_electrons, n) == _brute(window, n_electrons, n), (window, n_electrons)


def test_unrestricted_and_out_of_range():
    assert window_dimension(None, 3, 8) == comb(8, 3)
    assert window_dimension({}, 3, 8) == comb(8, 3)
    assert window_dimension(None, 9, 8) == 0
    assert window_dimension({frozenset(range(4)): (0, 1)}, -1, 8) == 0


def test_impurity_and_bath_window():
    # 10 impurity orbitals with 8-9 electrons, a 10-orbital bath at most 2 holes, 18 electrons.
    imp, bath = frozenset(range(10)), frozenset(range(10, 20))
    expected = sum(comb(10, k) * comb(10, 18 - k) for k in (8, 9) if 18 - k >= 8)
    assert window_dimension({imp: (8, 9), bath: (8, 10)}, 18, 20) == expected


def test_an_entangled_window_gives_up_at_the_state_bound():
    rng = random.Random(0)
    window = {}
    for _ in range(6):  # six random overlapping sets: the unbounded count ran over a minute
        key = frozenset(rng.sample(range(124), rng.randint(8, 30)))
        window[key] = (0, min(len(key), rng.randint(4, 20)))
    assert window_dimension(window, 60, 124, max_states=10_000) is None
    small = {frozenset(range(10)): (8, 9), frozenset(range(10, 20)): (8, 10)}
    assert window_dimension(small, 18, 20, max_states=10_000) == window_dimension(small, 18, 20)
