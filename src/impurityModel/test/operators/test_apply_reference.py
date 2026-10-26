"""``ManyBodyOperator.apply_block`` against the independent pure-Python reference (``apply_reference``).

Fixtures are chosen to hit the places a term-visit rewrite can go wrong: term counts around the 64-term word
boundary, orbitals either side of the chunk boundary (63/64), raw (non-normal-ordered) strings that revisit an
orbital, the constant term, terms on orbitals beyond the determinant width, and restrictions that reject some
input rows while density terms fire on them.
"""

import itertools
import random

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

from .apply_reference import assert_matches, block_from_orbital_sets, block_to_orbital_sets, reference_apply


def _random_states(rng, pool, n_states, n_lo, n_hi, p):
    states = {}
    while len(states) < n_states:
        occ = frozenset(rng.sample(pool, rng.randrange(n_lo, n_hi + 1)))
        states[occ] = np.array([complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(p)])
    return states


def _coeff(rng):
    return complex(rng.gauss(0, 1), rng.gauss(0, 1))


def _check(op, terms, states, n_chunks, restrictions=None, weighted=None):
    if restrictions:
        op.set_restrictions(restrictions)
    if weighted:
        op.set_weighted_restrictions(weighted)
    got = block_to_orbital_sets(op.apply_block(block_from_orbital_sets(states, n_chunks), 0.0))
    ref = reference_apply(terms, states, n_chunks, restrictions, weighted)
    assert ref, "the fixture produces nothing, so it tests nothing"
    assert_matches(got, ref)
    return got


def _raw_operator(terms):
    """An operator that keeps the strings exactly as written (no normal ordering)."""
    op = ManyBodyOperator()
    op.set_normal_ordering(False)
    for key, c in terms.items():
        op[key] = c
    return op


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_canonical_random_hamiltonian_matches_reference(seed):
    """Also pins the string-order and sign conventions of the reference itself against the C++ apply."""
    rng = random.Random(seed)
    terms = {}
    for _ in range(40):
        i, j, k, l = (rng.randrange(0, 70) for _ in range(4))
        terms[((i, "c"), (j, "a"))] = _coeff(rng)
        terms[((i, "c"), (j, "c"), (k, "a"), (l, "a"))] = _coeff(rng)
        terms[((i, "c"), (i, "a"))] = _coeff(rng)
    op = ManyBodyOperator(terms)
    canonical = op.to_dict()
    _check(op, canonical, _random_states(rng, list(range(70)), 60, 20, 35, 2), 2)


@pytest.mark.parametrize("n_hops", [1, 63, 64, 65, 127, 129])
def test_term_count_around_the_word_boundary(n_hops):
    """Exactly `n_hops` distinct hops (plus nothing else) on 1-chunk keys: the last (partial) word of the firing
    bitmap must neither drop nor invent a term."""
    rng = random.Random(100 + n_hops)
    pairs = [(i, j) for i in range(40) for j in range(40) if i != j]
    rng.shuffle(pairs)
    terms = {((i, "c"), (j, "a")): _coeff(rng) for i, j in pairs[:n_hops]}
    op = ManyBodyOperator(terms)
    assert op.num_flat_terms() == n_hops
    _check(op, op.to_dict(), _random_states(rng, list(range(40)), 80, 10, 30, 2), 1)


def test_terms_straddling_the_chunk_boundary():
    rng = random.Random(7)
    terms = {
        ((63, "c"), (64, "a")): 1.0 + 0.5j,
        ((64, "c"), (63, "a")): -0.7 + 0.2j,
        ((0, "c"), (127, "a")): 0.3 - 0.9j,  # between-mask spans both chunks
        ((127, "c"), (0, "a")): 0.4 + 0.1j,
        ((63, "c"), (63, "a"), (64, "c"), (64, "a")): 0.8,  # n_63 n_64
        ((63, "c"), (64, "c"), (0, "a"), (1, "a")): -0.6 + 0.6j,
        ((0, "c"), (1, "c"), (63, "a"), (64, "a")): 0.5 - 0.5j,
        ((64, "c"), (64, "a")): 1.3,
    }
    pool = [0, 1, 2, 30, 61, 62, 63, 64, 65, 66, 100, 125, 126, 127]
    op = ManyBodyOperator(terms)
    _check(op, op.to_dict(), _random_states(rng, pool, 120, 4, 10, 3), 2)


def test_raw_strings_that_revisit_an_orbital_and_the_constant_term():
    rng = random.Random(3)
    terms = {(): 0.25 + 0.0j}
    for i in range(0, 8):
        terms[((i, "c"), (i, "a"), (i, "c"))] = _coeff(rng)  # c^d_i c_i c^d_i: live only on an empty orbital
        terms[((i, "c"), (i, "c"))] = _coeff(rng)  # never fires
        terms[((i, "a"), (i, "c"))] = _coeff(rng)  # c_i c^d_i = 1 - n_i, a live general diagonal term
        terms[((i, "a"), (i, "a"))] = _coeff(rng)  # never fires
        terms[((i, "c"), ((i + 1) % 8, "a"), (i, "a"), ((i + 2) % 8, "c"))] = _coeff(rng)
    op = _raw_operator(terms)
    _check(op, terms, _random_states(rng, list(range(8)), 60, 2, 6, 2), 1)


def test_orbital_beyond_the_determinant_width_never_fires():
    rng = random.Random(5)
    terms = {((70, "c"), (70, "a")): 2.0, ((3, "c"), (4, "a")): 1.0 + 1.0j, ((80, "a"), (81, "c")): 1.0}
    op = _raw_operator(terms)  # on 1-chunk keys orbitals >= 64 do not exist
    _check(op, terms, _random_states(rng, list(range(12)), 40, 3, 8, 2), 1)


def test_restrictions_that_reject_some_input_rows_while_density_terms_fire():
    rng = random.Random(9)
    terms = {}
    for i in range(10):
        terms[((i, "c"), (i, "a"))] = _coeff(rng)
    for i, j in itertools.permutations(range(10), 2):
        terms[((i, "c"), (j, "a"))] = _coeff(rng)
    for _ in range(25):
        i, j, k, l = rng.sample(range(10), 4)
        terms[((i, "c"), (j, "c"), (k, "a"), (l, "a"))] = _coeff(rng)
    op = ManyBodyOperator(terms)
    states = _random_states(rng, list(range(10)), 150, 2, 8, 2)
    restr = {frozenset(range(0, 6)): (2, 4)}
    weighted = [({o: (1 if o % 2 else -1) for o in range(10)}, (-2, 2))]
    got = _check(op, op.to_dict(), states, 1, restr, weighted)
    # some input rows really are outside the window (otherwise this tests nothing about the row gate)
    assert any(not 2 <= len(s & frozenset(range(6))) <= 4 for s in states)
    assert all(2 <= len(s & frozenset(range(6))) <= 4 for s in got)
