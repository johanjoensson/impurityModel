"""Identical-block detection must see the bath, not only the first moment (ledger row C10).

``symmetries.impurity_block_structure`` reads the GF block structure off
``M = h_imp + V^dagger V`` alone. Two impurity orbitals with the same level and the same total
hybridization strength but *different bath energies* have equal ``M`` -- so they are declared
identical and one Green's function is copied into the other -- although their hybridization
functions, and therefore their Green's functions, differ at every frequency beyond the first
moment. A spin-polarized bath (antiferromagnetic NiO) is the production shape of this.
"""

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import impurity_block_structure

C10 = "C10: identical-block detection ignores h_bath (doc/reviews/gf_review.md)"

EPS, V = -0.3, 0.4


def _two_orbitals_with_baths(e_bath_0, e_bath_1):
    """Impurity orbitals 0, 1 (equal level), each hopping (V) to its own bath orbital 2, 3."""
    terms = {((0, "c"), (0, "a")): EPS, ((1, "c"), (1, "a")): EPS}
    for imp, bath, e_b in ((0, 2, e_bath_0), (1, 3, e_bath_1)):
        terms[((bath, "c"), (bath, "a"))] = e_b
        terms[((imp, "c"), (bath, "a"))] = V
        terms[((bath, "c"), (imp, "a"))] = V
    return ManyBodyOperator(terms)


def _noninteracting_g(e_bath, z):
    return 1.0 / (z - EPS - V**2 / (z - e_bath))


def test_equal_baths_are_identical_blocks():
    """The control: same bath energy, genuinely identical Green's functions."""
    bs = impurity_block_structure(_two_orbitals_with_baths(-1.0, -1.0), [0, 1], n_orb=4)
    assert len(bs.inequivalent_blocks) == 1, bs


@pytest.mark.xfail(strict=True, reason=C10)
def test_different_bath_energies_are_not_identical_blocks():
    z = np.array([0.2 + 0.1j])
    assert not np.allclose(_noninteracting_g(-1.0, z), _noninteracting_g(1.5, z)), "premise: the Gs differ"
    bs = impurity_block_structure(_two_orbitals_with_baths(-1.0, 1.5), [0, 1], n_orb=4)
    assert len(bs.inequivalent_blocks) == 2, bs
