"""``SeedWindow``: a bath-less shell's *seed* occupation is a window, not a pin (review J1, step 11.0b)."""

import pytest

from impurityModel.ed.basis_generation import DEEP_SEED_WIDTH, SeedWindow, generate_initial_basis
from impurityModel.test.basis.test_sector_key import N_ORB, _model

NOMINAL = {0: 2, 1: 2}


def _seed(frozen, nominal=NOMINAL):
    _h, impurity_orbitals, bath_states = _model()
    states, _n = generate_initial_basis(
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        delta_valence_occ=dict.fromkeys(nominal, 0),
        delta_conduction_occ=dict.fromkeys(nominal, 0),
        delta_impurity_occ=dict.fromkeys(nominal, 0),
        nominal_impurity_occ=nominal,
        n_bytes=(N_ORB + 7) // 8,
        verbose=False,
        frozen_occupations=frozen,
    )
    return {bytes(s.to_bytearray()) for s in states}


def _occ(seed, orbs):
    """Occupations of ``orbs`` over a seed (orbital -> bit is MSB-first within a byte)."""
    out = set()
    for det in seed:
        out.add(sum((det[o // 8] >> (7 - o % 8)) & 1 for o in orbs))
    return out


def test_window_spans_width_and_keeps_total_electrons():
    nominal = {0: 1, 1: 1}  # total impurity charge 2
    narrow = _seed(SeedWindow({0}, width=0), nominal)
    wide = _seed(SeedWindow({0}, width=1), nominal)
    assert _occ(narrow, (0, 1)) == {1}
    assert _occ(wide, (0, 1)) == {0, 1, 2}
    assert narrow < wide
    # Charge is conserved: every determinant holds the same number of electrons.
    assert len({sum(bin(b).count("1") for b in det) for det in wide}) == 1


def test_width_zero_is_a_total_pin():
    seed = _seed(SeedWindow({0}, width=0))
    assert _occ(seed, (0, 1)) == {2}


def test_window_contains_the_pinned_seed():
    pinned = _seed({0})
    window = _seed(SeedWindow({0}, width=1))
    assert pinned <= window <= _seed(None)


def test_plain_set_path_is_unchanged():
    assert _seed({0}) == _seed(set({0}))
    assert _occ(_seed({0}), (0, 1)) == {2}


def test_default_width():
    assert SeedWindow({0}).width == DEEP_SEED_WIDTH == 1
