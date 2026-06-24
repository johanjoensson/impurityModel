"""Tests for weighted-sum occupation restrictions, e.g. S_z (symmetry plan, Phase 6)."""

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant


# 4 spin-orbitals: 0=dn0, 1=up0, 2=dn1, 3=up1.
_UP = [1, 3]
_DN = [0, 2]
# 2*S_z weights (integer): +1 on up orbitals, -1 on down orbitals.
_SZ2_WEIGHTS = {1: 1, 3: 1, 0: -1, 2: -1}


def _sd(occ, n=4):
    b = bytearray((n + 7) // 8)
    for o in occ:
        b[o // 8] |= 1 << (7 - o % 8)
    return SlaterDeterminant.from_bytes(bytes(b))


def _occset(det, n=4):
    import impurityModel.ed.product_state_representation as psr

    return {k for k, bit in enumerate(psr.bytes2bitarray(bytes(det.to_bytearray()), n)) if bit}


def _two_sz(occ):
    occ = set(occ)
    return len(occ & set(_UP)) - len(occ & set(_DN))


def test_weighted_restriction_Sz():
    """Only determinants with the target S_z survive the weighted restriction."""
    # n_total preserves every determinant (with its occupation as coefficient).
    n_total = ManyBodyOperator({((i, "c"), (i, "a")): 1.0 for i in range(4)})

    # Input spans 2*S_z = -2, 0, +2: {0,2} (both dn), {0,1} (dn+up), {1,3} (both up).
    psi = ManyBodyState({_sd([0, 2]): 1.0, _sd([0, 1]): 1.0, _sd([1, 3]): 1.0})

    # Restrict to S_z = 0 (2*S_z in [0, 0]).
    n_total.set_weighted_restrictions([(_SZ2_WEIGHTS, (0, 0))])
    survivors = [_occset(d) for d, _ in n_total(psi).items()]
    assert all(_two_sz(occ) == 0 for occ in survivors)
    assert {0, 1} in survivors  # the S_z=0 determinant is kept
    assert {0, 2} not in survivors and {1, 3} not in survivors  # other sectors removed

    # Restrict to maximal S_z (2*S_z in [2, 2]): only {1,3} (both up) survives.
    n_total.set_weighted_restrictions([(_SZ2_WEIGHTS, (2, 2))])
    survivors = [_occset(d) for d, _ in n_total(psi).items()]
    assert survivors == [{1, 3}]


def test_weighted_restriction_roundtrip():
    """A basis spanning multiple S_z sectors collapses to the target sector under the mask."""
    n_total = ManyBodyOperator({((i, "c"), (i, "a")): 1.0 for i in range(4)})
    # All 2-electron determinants of the 4 orbitals.
    from itertools import combinations

    psi = ManyBodyState({_sd(list(occ)): 1.0 for occ in combinations(range(4), 2)})

    n_total.set_weighted_restrictions([(_SZ2_WEIGHTS, (0, 0))])
    survivors = {frozenset(_occset(d)) for d, _ in n_total(psi).items()}
    # 2-electron S_z=0 determinants: one up + one down -> {dn,up} pairs.
    expected = {frozenset(occ) for occ in combinations(range(4), 2) if _two_sz(occ) == 0}
    assert survivors == expected


def test_weighted_and_subset_restrictions_combine():
    """Subset and weighted restrictions are enforced together."""
    n_total = ManyBodyOperator({((i, "c"), (i, "a")): 1.0 for i in range(4)})
    from itertools import combinations

    psi = ManyBodyState({_sd(list(occ)): 1.0 for occ in combinations(range(4), 2)})

    # Total occupation exactly 2 (subset = all orbitals) AND S_z = 0.
    n_total.set_restrictions({frozenset(range(4)): (2, 2)})
    n_total.set_weighted_restrictions([(_SZ2_WEIGHTS, (0, 0))])
    survivors = {frozenset(_occset(d)) for d, _ in n_total(psi).items()}
    expected = {frozenset(occ) for occ in combinations(range(4), 2) if _two_sz(occ) == 0}
    assert survivors == expected


def test_automatic_restrictions_Sz():
    """The auto-generated S_z restriction (from spin pairs) isolates the S_z sector."""
    from itertools import combinations
    from impurityModel.ed.finite import impurity_spin_pairs, spin_pairs_consistent_with_h
    from impurityModel.ed.symmetries import sz_weighted_restriction

    # 4-orbital impurity: 0=dn0,1=up0,2=dn1,3=up1 (impurity_orbitals down-then-up).
    imp = {0: [[0, 1, 2, 3]]}
    pairs = impurity_spin_pairs(imp)
    assert pairs == [(0, 2), (1, 3)]  # (dn, up): (0,2) and (1,3)

    # dn = {0, 1}, up = {2, 3}. Spin-diagonal hopping (within-spin: 0<->1, 2<->3)
    # conserves S_z, so the pairing is validated.
    terms = {}
    for a, b in ((0, 1), (2, 3)):
        terms[((a, "c"), (b, "a"))] = -1.0
        terms[((b, "c"), (a, "a"))] = -1.0
    h = ManyBodyOperator(terms)
    assert spin_pairs_consistent_with_h(h, pairs, 4)

    n_total = ManyBodyOperator({((i, "c"), (i, "a")): 1.0 for i in range(4)})
    psi = ManyBodyState({_sd(list(occ)): 1.0 for occ in combinations(range(4), 2)})

    # impurity_spin_pairs gives (0,2),(1,3): up = {2,3}, dn = {0,1}.
    weights, bounds = sz_weighted_restriction(pairs, two_sz_target=0)
    assert weights == {2: 1, 3: 1, 0: -1, 1: -1}
    n_total.set_weighted_restrictions([(weights, bounds)])

    def two_sz(occ):
        return len(occ & {2, 3}) - len(occ & {0, 1})

    survivors = {frozenset(_occset(d)) for d, _ in n_total(psi).items()}
    expected = {frozenset(occ) for occ in combinations(range(4), 2) if two_sz(set(occ)) == 0}
    assert survivors == expected


def test_weighted_restriction_in_basis_expand():
    """A Basis with a weighted S_z restriction keeps expand inside the S_z sector."""
    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.ed.symmetries import sz_weighted_restriction
    from impurityModel.ed.finite import impurity_spin_pairs

    imp = {0: [[0, 1, 2, 3]]}  # dn = {0,1}, up = {2,3}
    pairs = impurity_spin_pairs(imp)  # [(0,2),(1,3)]
    sz = sz_weighted_restriction(pairs, two_sz_target=0)  # weights {2:1,3:1,0:-1,1:-1}

    # Operator: a within-spin hop (dn0->dn1) and a spin-flip (dn0->up0, leaves S_z=0).
    op = ManyBodyOperator(
        {
            ((1, "c"), (0, "a")): 1.0,
            ((0, "c"), (1, "a")): 1.0,
            ((2, "c"), (0, "a")): 1.0,
            ((0, "c"), (2, "a")): 1.0,
        }
    )
    seed = [_sd([0, 3])]  # dn0 + up1, 2*S_z = 0
    kw = dict(impurity_orbitals=imp, bath_states=({0: [[]]}, {0: [[]]}), verbose=False)

    def two_sz(occ):
        return len(occ & {2, 3}) - len(occ & {0, 1})

    restricted = Basis(initial_basis=list(seed), weighted_restrictions=[sz], **kw)
    restricted.expand(op)
    assert all(two_sz(_occset(d)) == 0 for d in restricted.local_basis)

    free = Basis(initial_basis=list(seed), **kw)
    free.expand(op)
    assert any(two_sz(_occset(d)) != 0 for d in free.local_basis)  # spin-flip leaks out
