r"""``symmetries.conserved_parities`` / ``parity_labels`` / ``conserves_sz``: the hard labels.

A selected-CI expansion admits only determinants in the support of ``H|psi_ref>``. Labels diagonal on
determinants that ``H`` conserves therefore bound it exactly: an expansion seeded inside one label
never leaves it. These tests pin that the detected parities really are conserved (term by term, on
determinants), that roundoff terms do not break one and real terms do, and the ``S_z`` check.
"""

import itertools

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp
from impurityModel.ed.spin_pairs import bath_spin_pairs, impurity_spin_pairs
from impurityModel.ed.symmetries import PARITY_TERM_TOL, conserved_parities, conserves_sz, parity_labels
from impurityModel.test.support.cubic_d_shell import cubic_d_shell


def _det(occupied):
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _bits(det, n_orb):
    raw = np.frombuffer(bytes(det.to_bytearray()), dtype=np.uint8)
    return np.unpackbits(raw)[:n_orb]


def test_every_detected_parity_is_conserved_by_every_matrix_element():
    _h, h_op, _imp, _baths = cubic_d_shell(n_bath_sets=1)
    n_orb = 20
    parities = conserved_parities(h_op)
    assert len(parities) >= 2, "the cubic shell has two-fold rotations beyond the total-number parity"
    rng = np.random.default_rng(0)
    all_dets = list(itertools.combinations(range(n_orb), 12))
    for k in rng.choice(len(all_dets), size=40, replace=False):
        ket = _det(all_dets[k])
        label = parity_labels(_bits(ket, n_orb)[None, :], parities)
        image = applyOp(h_op, ManyBodyState({ket: 1.0}), cutoff=0)
        assert len(image) > 1
        for bra in image.keys():
            np.testing.assert_array_equal(parity_labels(_bits(bra, n_orb)[None, :], parities), label)


def test_a_roundoff_term_does_not_break_a_parity_and_a_real_one_does():
    hop = {((0, "c"), (1, "a")): 1.0, ((1, "c"), (0, "a")): 1.0, ((2, "c"), (3, "a")): 1.0, ((3, "c"), (2, "a")): 1.0}
    base = len(conserved_parities(ManyBodyOperator(hop)))
    # {0, 1} and {2, 3} are separately conserved; a 0-2 hopping joins them into one.
    noisy = {**hop, ((0, "c"), (2, "a")): 0.1 * PARITY_TERM_TOL, ((2, "c"), (0, "a")): 0.1 * PARITY_TERM_TOL}
    real = {**hop, ((0, "c"), (2, "a")): 1e-3, ((2, "c"), (0, "a")): 1e-3}
    assert base == 2
    assert len(conserved_parities(ManyBodyOperator(noisy))) == base
    assert len(conserved_parities(ManyBodyOperator(real))) == base - 1


def test_a_pair_hop_conserves_parity_but_not_number():
    """``c+_0 c+_1 c_3 c_2`` moves two electrons from {2, 3} to {0, 1}: N_{01} changes, its parity does not."""
    op = ManyBodyOperator(
        {((0, "c"), (1, "c"), (3, "a"), (2, "a")): 1.0, ((2, "c"), (3, "c"), (1, "a"), (0, "a")): 1.0}
    )
    masks = conserved_parities(op)
    labels = parity_labels(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), masks)
    np.testing.assert_array_equal(labels[0], labels[1])
    # ...and the parity of {0, 1} alone is one of the conserved labels: it is not the number.
    assert (
        parity_labels(np.array([[1, 0, 0, 0]]), masks).tolist()
        != parity_labels(np.array([[0, 0, 0, 0]]), masks).tolist()
    )


def test_sz_conservation_is_exact_and_spin_mixing_breaks_it():
    _h, h_op, imp, baths = cubic_d_shell()
    pairs = impurity_spin_pairs(imp) + bath_spin_pairs(baths)
    assert conserves_sz(h_op, pairs)
    # A spin-flip hopping changes 2 S_z by 2.
    _h, h_soc, _imp, _baths = cubic_d_shell(soc=0.1)
    assert not conserves_sz(h_soc, pairs)
