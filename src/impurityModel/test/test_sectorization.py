"""Tests for symmetry-derived sectorization / restrictions (symmetry plan, Phase 3)."""

from itertools import combinations

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import (
    conserved_subset_charges,
    restrictions_from_charges,
    subset_occupations,
)


def _sd(occupied, n_orbs):
    n_bytes = (n_orbs + 7) // 8
    data = bytearray(n_bytes)
    for orb in occupied:
        data[orb // 8] |= 1 << (7 - (orb % 8))
    return SlaterDeterminant.from_bytes(bytes(data))


def _sector_basis(n_orb, n_elec):
    return [tuple(occ) for occ in combinations(range(n_orb), n_elec)]


def _dense_matrix(op, occs, n_orb):
    """<a|op|b> over an explicit list of occupied-orbital tuples."""
    states = [ManyBodyState({_sd(o, n_orb): 1.0}) for o in occs]
    n = len(states)
    mat = np.zeros((n, n), dtype=complex)
    for b, sb in enumerate(states):
        col = applyOp(op, sb)
        for a, sa in enumerate(states):
            mat[a, b] = inner(sa, col)
    return mat


def _hubbard_dimer():
    """0,1 = up sites; 2,3 = down sites. Spin-diagonal hopping + on-site U (density-density)."""
    t, u = 0.9, 2.3
    terms = {}
    for a, b in ((0, 1), (2, 3)):
        terms[((a, "c"), (b, "a"))] = -t
        terms[((b, "c"), (a, "a"))] = -t
    for up, dn in ((0, 2), (1, 3)):
        terms[((up, "c"), (dn, "c"), (dn, "a"), (up, "a"))] = u  # n_up n_dn
    return ManyBodyOperator(terms)


def test_conserved_charges_hubbard_separates_spin():
    """No spin mixing -> N_up and N_down are separately conserved subset charges."""
    op = _hubbard_dimer()
    charges = conserved_subset_charges(op, n_orb=4)
    assert charges == [frozenset({0, 1}), frozenset({2, 3})]


def _commutes_with_H(op, charge, n_orb):
    """True iff N_charge commutes with op on every fixed-N sector (block-diagonal)."""
    for n_elec in range(n_orb + 1):
        occs = _sector_basis(n_orb, n_elec)
        if not occs:
            continue
        h = _dense_matrix(op, occs, n_orb)
        n_s = np.diag([len(set(o) & charge) for o in occs]).astype(complex)
        if np.linalg.norm(h @ n_s - n_s @ h) > 1e-10:
            return False
    return True


def test_conserved_charges_actually_commute():
    """Each discovered subset charge commutes with the full many-body Hamiltonian."""
    op = _hubbard_dimer()
    charges = conserved_subset_charges(op, n_orb=4)
    for charge in charges:
        assert _commutes_with_H(op, charge, 4)


def test_one_body_spin_flip_merges_charges():
    """A one-body spin-flip term (SOC-like) couples up<->down -> only total N survives."""
    op = _hubbard_dimer()
    terms = op.to_dict()
    # add c†_0 c_2 + h.c. (up-site-0 <-> down-site-0): spin mixing
    terms[((0, "c"), (2, "a"))] = 0.4
    terms[((2, "c"), (0, "a"))] = 0.4
    op2 = ManyBodyOperator(terms)
    charges = conserved_subset_charges(op2, n_orb=4)
    assert charges == [frozenset({0, 1, 2, 3})]  # single charge = total N
    assert _commutes_with_H(op2, charges[0], 4)


def test_two_body_pair_hopping_merges_charges():
    """A pair-hopping term that moves 2 up <-> 2 down breaks N_up/N_down."""
    op = _hubbard_dimer()
    terms = op.to_dict()
    # c†_0 c†_1 c_2 c_3 : create two up, annihilate two down -> N_up not conserved
    terms[((0, "c"), (1, "c"), (2, "a"), (3, "a"))] = 0.3
    terms[((3, "c"), (2, "c"), (1, "a"), (0, "a"))] = 0.3  # h.c.
    op2 = ManyBodyOperator(terms)
    charges = conserved_subset_charges(op2, n_orb=4)
    assert charges == [frozenset({0, 1, 2, 3})]
    assert _commutes_with_H(op2, charges[0], 4)


def test_density_density_does_not_merge():
    """A genuine density-density cross term must NOT merge the spin charges."""
    op = _hubbard_dimer()
    terms = op.to_dict()
    # n_0 n_3 (up-site0 * down-site1) density-density: conserves both spin charges
    terms[((0, "c"), (3, "c"), (3, "a"), (0, "a"))] = 0.5
    op2 = ManyBodyOperator(terms)
    charges = conserved_subset_charges(op2, n_orb=4)
    assert charges == [frozenset({0, 1}), frozenset({2, 3})]


def test_restrictions_from_charges_and_occupations():
    """Mapping charges + a reference determinant to Basis.restrictions format."""
    op = _hubbard_dimer()
    charges = conserved_subset_charges(op, n_orb=4)
    # reference determinant: one electron per spin (orbital 0 up, orbital 2 down)
    occ = subset_occupations(charges, {0, 2})
    assert occ == [1, 1]

    restr = restrictions_from_charges(charges, occ)
    assert restr == {frozenset({0, 1}): (1, 1), frozenset({2, 3}): (1, 1)}

    restr_slack = restrictions_from_charges(charges, occ, slack=1)
    assert restr_slack == {frozenset({0, 1}): (0, 2), frozenset({2, 3}): (0, 2)}
