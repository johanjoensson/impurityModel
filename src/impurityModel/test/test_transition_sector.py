"""Unit tests for the conserved-charge sector confinement of transition operators (B1).

These cover :func:`symmetries.operator_charge_shift` and
:func:`symmetries.transition_sector_restrictions`, which confine the excited-state Lanczos
of a spectrum to the conserved-charge sector that ``tOp|gs>`` occupies.
"""

from impurityModel.ed.symmetries import (
    operator_charge_shift,
    restrictions_from_charges,
    transition_sector_restrictions,
)


# Two conserved subsets: a "core" {0,1} and a "valence" {2,3}.
CHARGES = [frozenset({0, 1}), frozenset({2, 3})]


def test_charge_shift_single_creation():
    # c_2^dagger: +1 in the valence subset, 0 in the core subset.
    op = {((2, "c"),): 1.0}
    assert operator_charge_shift(op, CHARGES) == [0, 1]


def test_charge_shift_single_annihilation():
    op = {((0, "a"),): 1.0}
    assert operator_charge_shift(op, CHARGES) == [-1, 0]


def test_charge_shift_dipole_consistent():
    # Dipole-like c_valence^dagger c_core: every term shifts (core:-1, valence:+1).
    op = {
        ((2, "c"), (0, "a")): 0.5,
        ((3, "c"), (1, "a")): -0.5,
        ((2, "c"), (1, "a")): 0.3,
    }
    assert operator_charge_shift(op, CHARGES) == [-1, 1]


def test_charge_shift_inconsistent_returns_none():
    # One term raises valence, another lowers it -> no definite sector.
    op = {
        ((2, "c"), (0, "a")): 1.0,
        ((0, "c"), (2, "a")): 1.0,
    }
    assert operator_charge_shift(op, CHARGES) is None


def test_charge_shift_orbital_outside_charges_returns_none():
    op = {((9, "c"),): 1.0}
    assert operator_charge_shift(op, CHARGES) is None


def test_charge_shift_ignores_negligible_terms():
    op = {
        ((2, "c"),): 1.0,
        ((0, "c"),): 1e-15,  # below tol -> ignored, so sector stays definite
    }
    assert operator_charge_shift(op, CHARGES) == [0, 1]


def test_sector_restrictions_match_shifted_occupations():
    gs_occ = [2, 1]
    op = {((2, "c"), (0, "a")): 1.0}  # shift (core:-1, valence:+1)
    restr = transition_sector_restrictions(CHARGES, gs_occ, op)
    expected = restrictions_from_charges(CHARGES, [1, 2])
    assert restr == expected
    # Strict sector: each subset pinned to a single occupation.
    assert restr[frozenset({0, 1})] == (1, 1)
    assert restr[frozenset({2, 3})] == (2, 2)


def test_sector_restrictions_none_when_no_definite_sector():
    gs_occ = [2, 1]
    op = {((2, "c"), (0, "a")): 1.0, ((0, "c"), (2, "a")): 1.0}
    assert transition_sector_restrictions(CHARGES, gs_occ, op) is None


def test_sector_restrictions_none_when_negative_occupation():
    # Removing from an empty subset -> unreachable sector, no confinement.
    gs_occ = [0, 1]
    op = {((0, "a"),): 1.0}  # shift (core:-1) -> core occ -1 < 0
    assert transition_sector_restrictions(CHARGES, gs_occ, op) is None


def test_sector_restrictions_slack_widens():
    gs_occ = [2, 1]
    op = {((2, "c"),): 1.0}  # shift (valence:+1)
    restr = transition_sector_restrictions(CHARGES, gs_occ, op, slack=1)
    # core: 2 +/- 1 -> (1, 2) capped by subset size 2; valence: 2 +/- 1 -> (1, 2)
    assert restr[frozenset({0, 1})] == (1, 2)
    assert restr[frozenset({2, 3})] == (1, 2)
