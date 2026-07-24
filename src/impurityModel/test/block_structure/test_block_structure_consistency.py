"""Phase 0 diagnostics: impurity-only vs interacting-GF block structure.

These tests pin down the correctness gap the symmetry-basis work fixes: the spectra /
self-energy paths derive the GF block structure from ``h[imp, imp]`` only, which can be
strictly too fine when impurity orbitals are coupled through a shared bath orbital
(bath-mediated hybridization) or by a two-body term. The full-Hamiltonian
:func:`green_function_block_structure` catches those couplings.
"""

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import (
    auto_block_structure,
    discovered_orbital_blocks,
    impurity_gf_block_consistency,
)


def _bath_mediated():
    """Impurity orbitals 0,1 with NO direct hopping, both hybridising with bath orbital 2.

    Per spin would double this; here a single spinless triangle-minus-one-edge suffices to
    expose the bath-mediated coupling of the two impurity orbitals.
    """
    v = 0.5
    e_bath = 1.3
    terms = {
        ((2, "c"), (2, "a")): e_bath,
        ((0, "c"), (2, "a")): v,
        ((2, "c"), (0, "a")): v,
        ((1, "c"), (2, "a")): v,
        ((2, "c"), (1, "a")): v,
    }
    return ManyBodyOperator(terms)


def test_impurity_only_is_too_fine_for_bath_mediated_coupling():
    """h[imp,imp] splits 0 and 1; the interacting GF keeps them coupled through bath orb 2."""
    op = _bath_mediated()
    imp = [0, 1]

    # Impurity-only structure sees no 0-1 element -> two singleton blocks.
    imp_only = auto_block_structure(op, orbitals=imp).blocks
    assert len({frozenset(b) for b in imp_only}) == 2

    report = impurity_gf_block_consistency(op, imp, n_orb=3)
    assert not report.consistent
    assert report.impurity_only_blocks == [frozenset({0}), frozenset({1})]
    assert report.gf_blocks == [frozenset({0, 1})]
    assert report.missing_pairs == [(0, 1)]


def test_direct_hopping_is_consistent():
    """With a direct 0-1 impurity hop, both partitions agree (single impurity block)."""
    op = _bath_mediated()
    terms = op.to_dict()
    terms[((0, "c"), (1, "a"))] = 0.2
    terms[((1, "c"), (0, "a"))] = 0.2
    op2 = ManyBodyOperator(terms)

    report = impurity_gf_block_consistency(op2, [0, 1], n_orb=3)
    assert report.consistent
    assert report.gf_blocks == [frozenset({0, 1})]
    assert report.missing_pairs == []


def test_decoupled_impurities_stay_split():
    """Two impurity orbitals with independent baths remain in separate GF blocks."""
    v = 0.5
    terms = {
        ((2, "c"), (2, "a")): 1.0,
        ((3, "c"), (3, "a")): 1.0,
        ((0, "c"), (2, "a")): v,
        ((2, "c"), (0, "a")): v,
        ((1, "c"), (3, "a")): v,
        ((3, "c"), (1, "a")): v,
    }
    op = ManyBodyOperator(terms)
    report = impurity_gf_block_consistency(op, [0, 1], n_orb=4)
    assert report.consistent
    assert report.gf_blocks == [frozenset({0}), frozenset({1})]
    assert report.missing_pairs == []


def test_gf_blocks_restricted_to_impurity_match_full_projection():
    """The reported gf_blocks equal the full one-body connectivity projected onto imp."""
    op = _bath_mediated()
    imp = {0, 1}
    full = discovered_orbital_blocks(op, n_orb=3)
    projected = sorted((frozenset(b & imp) for b in full if b & imp), key=min)
    report = impurity_gf_block_consistency(op, sorted(imp), n_orb=3)
    assert report.gf_blocks == projected
