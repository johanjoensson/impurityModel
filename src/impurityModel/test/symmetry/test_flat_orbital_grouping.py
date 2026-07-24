"""Flat-list orbital grouping for calc_selfenergy (group by full-H conserved charges)."""

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.selfenergy import _per_group_occupation, _per_group_scalar
from impurityModel.ed.symmetries import classify_bath_occupation, group_orbitals_by_charges


def _two_spin_model():
    """Impurity 0(dn),1(up); valence bath 2(dn),3(up). Spin-conserving hop + density-density U.

    Conserved charges: {0,2} (down) and {1,3} (up) — U = n0 n1 conserves each spin count.
    """
    t, U = 0.7, 3.0
    terms = {
        ((0, "c"), (2, "a")): -t,
        ((2, "c"), (0, "a")): -t,
        ((1, "c"), (3, "a")): -t,
        ((3, "c"), (1, "a")): -t,
        ((0, "c"), (1, "c"), (1, "a"), (0, "a")): U,
    }
    return ManyBodyOperator(terms)


def test_classify_bath_occupation_splits_by_onsite_energy():
    """Baths below the Fermi level (h[o,o] < 0) are valence (occupied); the rest conduction."""
    terms = {
        ((0, "c"), (0, "a")): -1.0,  # impurity
        ((1, "c"), (1, "a")): -2.5,  # bath below Fermi -> valence (occupied)
        ((2, "c"), (2, "a")): 0.0,  # bath at Fermi -> conduction (empty)
        ((3, "c"), (3, "a")): 1.5,  # bath above Fermi -> conduction (empty)
        ((0, "c"), (1, "a")): 0.3,  # some hybridization
        ((1, "c"), (0, "a")): 0.3,
    }
    op = ManyBodyOperator(terms)
    valence, conduction = classify_bath_occupation(op, [0], n_orb=4)
    assert valence == [1]
    assert conduction == [2, 3]


def test_group_orbitals_by_charges_splits_by_spin():
    op = _two_spin_model()
    imp, (val, con) = group_orbitals_by_charges(op, [0, 1], [2, 3], [], n_orb=4)
    assert imp == {0: [[0]], 1: [[1]]}
    assert val == {0: [[2]], 1: [[3]]}
    assert con == {0: [[]], 1: [[]]}


def test_group_keys_sorted_by_min_impurity_orbital():
    op = _two_spin_model()
    imp, _ = group_orbitals_by_charges(op, [0, 1], [2, 3], [], n_orb=4)
    # group 0 owns the lowest impurity orbital (0), group 1 the next (1).
    assert min(imp[0][0]) < min(imp[1][0])


def test_bath_joins_its_impurity_group():
    """A conduction bath hybridising only with impurity 1 lands in group 1."""
    op = _two_spin_model()
    terms = op.to_dict()
    terms[((1, "c"), (4, "a"))] = -0.3  # impurity 1 <-> conduction bath 4
    terms[((4, "c"), (1, "a"))] = -0.3
    op2 = ManyBodyOperator(terms)
    _imp, (_val, con) = group_orbitals_by_charges(op2, [0, 1], [2, 3], [4], n_orb=5)
    assert con[1] == [[4]]
    assert con[0] == [[]]


def test_per_group_occupation_distributes_total():
    imp = {0: [[0, 1, 2, 3, 4]], 1: [[5, 6, 7, 8, 9]]}
    assert _per_group_occupation(8, imp) == {0: 4, 1: 4}  # scalar total split evenly
    assert _per_group_occupation({0: 3, 1: 5}, imp) == {0: 3, 1: 5}  # matching dict passes through
    assert _per_group_occupation({2: 8}, imp) == {0: 4, 1: 4}  # non-matching dict -> sum, redistribute


def test_per_group_occupation_uneven_sizes_remainder_to_largest():
    imp = {0: [[0, 1, 2]], 1: [[3]]}  # sizes 3 and 1
    assert _per_group_occupation(3, imp) == {0: 3, 1: 0}  # 3*3//4=2, 3*1//4=0, remainder 1 -> largest


def test_per_group_scalar():
    imp = {0: [[0]], 1: [[1]]}
    assert _per_group_scalar(0, imp) == {0: 0, 1: 0}
    assert _per_group_scalar({0: 1, 1: 0}, imp) == {0: 1, 1: 0}
