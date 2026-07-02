"""P1: impurity_block_structure folds full-Hamiltonian conserved charges into the GF blocks.

Verifies the routine the spectra / self-energy paths use to partition the impurity
Green's function: it must merge impurity orbitals coupled through a shared bath (or a
two-body term) that ``auto_block_structure`` (one-body-only) would wrongly split, while
still returning the value-based equivalences when the two partitions agree.
"""

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import (
    auto_block_structure,
    discovered_orbital_blocks,
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)


def _bath_mediated(imp=(0, 1), bath=2, v=0.5, e_bath=1.3):
    terms = {((bath, "c"), (bath, "a")): e_bath}
    for o in imp:
        terms[((o, "c"), (bath, "a"))] = v
        terms[((bath, "c"), (o, "a"))] = v
    return ManyBodyOperator(terms)


def test_merges_bath_mediated_impurity_pair():
    """0 and 1 share bath orb 2 -> one impurity block, even though h[0,1] = 0."""
    op = _bath_mediated()
    # auto_block_structure (one-body only) wrongly splits them.
    assert len(auto_block_structure(op, orbitals=[0, 1]).blocks) == 2

    bs = impurity_block_structure(op, [0, 1], n_orb=3)
    assert [sorted(b) for b in bs.blocks] == [[0, 1]]
    assert bs.inequivalent_blocks == [0]
    # No spurious cross-block equivalences in the coarsened fallback.
    assert bs.identical_blocks == [[0]]


def test_local_index_convention_for_noncontiguous_impurity():
    """Blocks are returned in local (0..n_imp-1) indices over sorted impurity orbitals."""
    # Impurity orbitals 3 and 5 coupled through bath orbital 4.
    terms = {((4, "c"), (4, "a")): 1.0}
    for o in (3, 5):
        terms[((o, "c"), (4, "a"))] = 0.5
        terms[((4, "c"), (o, "a"))] = 0.5
    op = ManyBodyOperator(terms)
    bs = impurity_block_structure(op, [3, 5], n_orb=6)
    # Local indices: 3->0, 5->1, merged into one block.
    assert [sorted(b) for b in bs.blocks] == [[0, 1]]


def test_agreement_keeps_equivalences():
    """When one-body connectivity already matches the GF partition, keep the equivalences."""
    # Diagonal cubic d-shell: eg (0,1,5,6) and t2g (2,3,4,7,8,9) degenerate; bath absent so
    # one-body connectivity == conserved-charge partition.
    e_eg, e_t2g = 0.6, 0.0
    diag = [e_eg, e_eg, e_t2g, e_t2g, e_t2g] * 2
    op = ManyBodyOperator({((i, "c"), (i, "a")): diag[i] for i in range(10)})

    bs = impurity_block_structure(op, list(range(10)), n_orb=10)
    one_body = auto_block_structure(op, orbitals=list(range(10)))
    # Identical object contents -> equivalences preserved (not the coarsened fallback).
    assert {frozenset(b) for b in bs.blocks} == {frozenset(b) for b in one_body.blocks}
    identical_groups = {frozenset(g) for g in bs.identical_blocks if len(g) > 1}
    assert identical_groups  # degenerate eg / t2g detected as identical blocks


def test_rotation_splits_degenerate_shell_into_1x1_blocks():
    """A scrambled doubly-degenerate impurity shell rotates to diagonal 1x1 blocks with the
    degenerate orbitals detected as identical (the eg/t2g-style collapse)."""
    rng = np.random.default_rng(0)
    diag = np.array([1.0, 1.0, 2.0, 2.0])  # two 2-fold-degenerate "irreps"
    X = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, _ = np.linalg.qr(X)
    h_imp = Q @ np.diag(diag) @ Q.conj().T  # entangled (non-diagonal) crystal field
    op = ManyBodyOperator(
        {((i, "c"), (j, "a")): h_imp[i, j] for i in range(4) for j in range(4) if abs(h_imp[i, j]) > 1e-12}
    )

    # Before rotation the shell is one connected 4x4 block.
    assert len(impurity_block_structure(op, [0, 1, 2, 3], n_orb=4).blocks) == 1

    W, _u = impurity_symmetry_rotation(op, [0, 1, 2, 3], n_orb=4)
    rotated = rotate_hamiltonian(op, W)
    bs = impurity_block_structure(rotated, [0, 1, 2, 3], n_orb=4)

    # Ten... here four 1x1 blocks, with the two degenerate pairs flagged identical.
    assert [sorted(b) for b in bs.blocks] == [[0], [1], [2], [3]]
    degenerate = sorted(sorted(g) for g in bs.identical_blocks if len(g) > 1)
    assert len(degenerate) == 2 and all(len(g) == 2 for g in degenerate)
    assert len(bs.inequivalent_blocks) == 2  # one representative GF per irrep


def test_partition_matches_gf_selection_rule():
    """The block partition equals the full one-body connectivity restricted to impurity."""
    op = _bath_mediated(imp=(0, 1), bath=2)
    imp = {0, 1}
    gf = discovered_orbital_blocks(op, n_orb=3)
    projected = sorted((frozenset(b & imp) for b in gf if b & imp), key=min)
    bs = impurity_block_structure(op, [0, 1], n_orb=3)
    # Map local block indices back to global to compare with the GF selection rule.
    imp_sorted = [0, 1]
    got = sorted((frozenset(imp_sorted[i] for i in b) for b in bs.blocks), key=min)
    assert got == projected
