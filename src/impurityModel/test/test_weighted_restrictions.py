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


def test_calc_gs_weighted_sz_restriction():
    """calc_gs with a weighted S_z restriction keeps the whole ground state in-sector."""
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.groundstate import calc_gs
    from impurityModel.ed.finite import impurity_spin_pairs, bath_spin_pairs
    from impurityModel.ed.symmetries import sz_weighted_restriction

    # SIAM 6 orbitals: 0,1=imp(dn,up); 2,3=val bath(dn,up); 4,5=cond bath(dn,up).
    ed, U, ev, ec, V = -2.0, 6.0, -4.0, 4.0, 1.0
    terms = {((o, "c"), (o, "a")): ed for o in (0, 1)}
    terms.update({((o, "c"), (o, "a")): ev for o in (2, 3)})
    terms.update({((o, "c"), (o, "a")): ec for o in (4, 5)})
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = U
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = V
        terms[((b, "c"), (a, "a"))] = V
    Hop = ManyBodyOperator(terms)

    imp = {0: [[0, 1]]}
    bath = ({0: [[2, 3]]}, {0: [[4, 5]]})
    pairs = impurity_spin_pairs(imp) + bath_spin_pairs(bath)  # up={1,3,5}, dn={0,2,4}
    sz0 = sz_weighted_restriction(pairs, two_sz_target=0)

    bs = BlockStructure(
        blocks=[[0, 1]], identical_blocks=[[0]], transposed_blocks=[[]],
        particle_hole_blocks=[[]], particle_hole_transposed_blocks=[[]], inequivalent_blocks=[0],
    )
    basis_setup = dict(
        impurity_orbitals=imp, bath_states=bath, N0={0: 1}, mixed_valence={0: 1}, tau=0.01,
        dense_cutoff=1000, spin_flip_dj=False, comm=None, truncation_threshold=100000,
        weighted_restrictions=[sz0],
    )
    psis, es, basis, _, _ = calc_gs(Hop, basis_setup, bs, np.eye(2, dtype=complex), verbose=False, slaterWeightMin=1e-12)

    up, dn = {1, 3, 5}, {0, 2, 4}
    for det in psis[0]:
        occ = _occset(det, 6)
        assert len(occ & up) - len(occ & dn) == 0  # every determinant has 2*S_z = 0


def test_greens_function_weighted_restriction_unchanged():
    """A correctly-widened weighted S_z restriction does not change the Green's function."""
    from mpi4py import MPI
    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.ed.greens_function import calc_Greens_function_with_offdiag
    from impurityModel.ed.symmetries import sz_weighted_restriction
    from impurityModel.ed.finite import impurity_spin_pairs

    # 4 orbitals (2 spatial): impurity_spin_pairs -> dn={0,1}, up={2,3}.
    imp = {0: [[0, 1, 2, 3]]}
    pairs = impurity_spin_pairs(imp)
    # Spin-diagonal hopping: dn0<->dn1 (0<->1), up0<->up1 (2<->3).
    hop = {}
    for a, b in ((0, 1), (2, 3)):
        hop[((a, "c"), (b, "a"))] = -0.5
        hop[((b, "c"), (a, "a"))] = -0.5
    hOp = ManyBodyOperator(hop)

    # Ground state: a 2-electron S_z=0 state |0,3> (dn0 + up1).
    psi = ManyBodyState({_sd([0, 3]): 1.0})
    tOp = ManyBodyOperator({((0, "a"),): 1.0})  # remove dn0

    def run(weighted, sparse):
        basis = Basis(
            impurity_orbitals=imp, bath_states=({0: [[]]}, {0: [[]]}),
            initial_basis=[bytes(_sd([0, 3]).to_bytearray())], comm=MPI.COMM_SELF,
            weighted_restrictions=weighted,
        )
        return calc_Greens_function_with_offdiag(
            hOp=hOp, tOps=[tOp], psis=[psi], es=[0.0], block_basis=basis,
            delta=0.01, dN=2, occ_cutoff=1e-6, slaterWeightMin=0.0, verbose=False, sparse=sparse,
        )

    # GS is in 2*S_z = 0 sector; restrict to it (the GF widens by one orbital weight).
    sz0 = sz_weighted_restriction(pairs, two_sz_target=0)
    # Both the sparse and the dense (basis-expanding) paths must be unchanged by the
    # weighted restriction (the dense path also exercises the block_green_impl fix).
    for sparse in (True, False):
        a0, b0, r0 = run(None, sparse)
        a1, b1, r1 = run([sz0], sparse)
        assert len(a0) == len(a1)
        for x0, x1 in zip(a0, a1):
            np.testing.assert_allclose(x0, x1, atol=1e-10)
        for y0, y1 in zip(b0, b1):
            np.testing.assert_allclose(y0, y1, atol=1e-10)
        for z0, z1 in zip(r0, r1):
            np.testing.assert_allclose(z0, z1, atol=1e-10)
