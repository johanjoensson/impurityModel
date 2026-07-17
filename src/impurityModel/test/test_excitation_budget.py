"""Tests for the excitation-budget weighted restriction (Phase 3a).

``basis_restrictions.excitation_budget_restriction`` turns a "max total bath excitations"
budget into a single weighted restriction consumed by
``ManyBodyOperator.set_weighted_restrictions``. Reference occupation: valence baths filled,
conduction baths empty; a hole in a filled-valence orbital or an electron in an
empty-conduction orbital each costs one (default), and the budget caps their sum.
"""

from impurityModel.ed.basis_restrictions import excitation_budget_restriction
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant


def _sd(occ, n=8):
    b = bytearray((n + 7) // 8)
    for o in occ:
        b[o // 8] |= 1 << (7 - o % 8)
    return SlaterDeterminant.from_bytes(bytes(b))


def _occset(det, n=8):
    import impurityModel.ed.product_state_representation as psr

    return {k for k, bit in enumerate(psr.bytes2bitarray(bytes(det.to_bytearray()), n)) if bit}


# 8 spin-orbitals. Valence baths (nominally filled): {0,1,2,3}. Conduction (empty): {4,5,6,7}.
_BATHS = ({0: [[0, 1, 2, 3]]}, {0: [[4, 5, 6, 7]]})
_VAL = {0, 1, 2, 3}
_CON = {4, 5, 6, 7}


def _excitations(occ):
    """Number of holes in valence + electrons in conduction."""
    return len(_VAL - occ) + len(occ & _CON)


def test_budget_weights_and_bounds():
    """The construction encodes E = C + sum w n with the documented weights/bounds."""
    weights, (q_lo, q_hi) = excitation_budget_restriction(_BATHS, budget=2)
    assert {weights[o] for o in _VAL} == {-1}
    assert {weights[o] for o in _CON} == {1}
    # C = sum of valence costs = 4; bounds = (-C, budget - C) = (-4, -2).
    assert (q_lo, q_hi) == (-4, -2)


def test_budget_filters_by_excitation_order():
    """Only determinants within the excitation budget survive the mask (exact, not a superset)."""
    from itertools import combinations

    n_total = ManyBodyOperator({((i, "c"), (i, "a")): 1.0 for i in range(8)})
    # Every 4-electron determinant of the 8 orbitals spans excitation orders 0..4.
    psi = ManyBodyState({_sd(list(occ)): 1.0 for occ in combinations(range(8), 4)})

    for budget in (0, 1, 2, 3, 4):
        rest = excitation_budget_restriction(_BATHS, budget=budget)
        n_total.set_weighted_restrictions([rest])
        survivors = [_occset(d) for d, _ in n_total(psi).items()]
        assert survivors, f"budget {budget} kept nothing"
        assert all(_excitations(o) <= budget for o in survivors), budget
        # Everything within budget is kept (the mask is exact, not a superset).
        expected = {frozenset(occ) for occ in combinations(range(8), 4) if _excitations(set(occ)) <= budget}
        assert {frozenset(o) for o in survivors} == expected


def test_budget_infinite_is_noop():
    """A budget >= the largest possible cost keeps every determinant (oracle)."""
    from itertools import combinations

    n_total = ManyBodyOperator({((i, "c"), (i, "a")): 1.0 for i in range(8)})
    psi = ManyBodyState({_sd(list(occ)): 1.0 for occ in combinations(range(8), 4)})
    all_dets = {frozenset(_occset(d)) for d, _ in n_total(psi).items()}

    rest = excitation_budget_restriction(_BATHS, budget=8)  # max cost = 4 holes + 4 electrons
    n_total.set_weighted_restrictions([rest])
    survivors = {frozenset(_occset(d)) for d, _ in n_total(psi).items()}
    assert survivors == all_dets


def test_budget_graded_cost():
    """A per-orbital cost function weights the budget (graded distance form)."""
    # Cost 2 for the outermost orbitals (3 in valence, 7 in conduction), 1 otherwise.
    cost = {3: 2, 7: 2}
    weights, (q_lo, q_hi) = excitation_budget_restriction(_BATHS, budget=3, cost_fn=lambda o: cost.get(o, 1))
    assert weights[3] == -2 and weights[7] == 2
    assert weights[0] == -1 and weights[4] == 1
    # C = 1+1+1+2 = 5; bounds = (-5, 3 - 5) = (-5, -2).
    assert (q_lo, q_hi) == (-5, -2)


def test_budget_no_baths_returns_none():
    """With no bath orbitals there is nothing to bound."""
    assert excitation_budget_restriction(({0: [[]]}, {0: [[]]}), budget=1) is None


def test_budget_composes_in_basis_expand():
    """A Basis carrying the budget keeps expand within the excitation budget."""
    from impurityModel.ed.manybody_basis import Basis

    imp = {0: [[]]}  # no impurity orbitals; all 8 are baths
    # A hopping operator that moves electrons between valence and conduction (raises excitation).
    op = ManyBodyOperator(
        {
            ((4, "c"), (3, "a")): 1.0,  # val3 -> con4 (one excitation)
            ((3, "c"), (4, "a")): 1.0,
            ((5, "c"), (2, "a")): 1.0,  # val2 -> con5 (another excitation)
            ((2, "c"), (5, "a")): 1.0,
        }
    )
    seed = [bytes(_sd([0, 1, 2, 3]).to_bytearray())]  # reference: 0 excitations
    budget1 = excitation_budget_restriction(_BATHS, budget=1)
    restricted = Basis(
        impurity_orbitals=imp,
        bath_states=_BATHS,
        initial_basis=list(seed),
        weighted_restrictions=[budget1],
        verbose=False,
    )
    restricted.expand(op)
    assert all(_excitations(_occset(d)) <= 1 for d in restricted.local_basis)

    free = Basis(impurity_orbitals=imp, bath_states=_BATHS, initial_basis=list(seed), verbose=False)
    free.expand(op)
    assert any(_excitations(_occset(d)) > 1 for d in free.local_basis)  # leaks past 1 without the budget
