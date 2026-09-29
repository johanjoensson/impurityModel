"""Particle-hole block equivalences (ledger row C1, ``doc/reviews/gf_review.md``).

``block_structure`` used to mark block ``B`` as the particle-hole image of ``A`` when their first
moments satisfy ``M_B = -conj(M_A)``, and ``build_full_greens_function`` then filled ``B`` from
``A`` as ``-conj(G_A)``. Three things were wrong, all after ``_check_gf_physical`` had passed:

* the scan included the block itself, so a block with ``Re M = 0`` was its own "partner" and its
  G was overwritten with ``-conj(G)``;
* ``-conj(G_A(w))`` is the image only on the Matsubara axis -- on the retarded real axis it is
  ``-conj(G_A(-w))``, and without the reversal the block carried negative spectral weight;
* the moment matrices went through the same map, but the second self-energy moment keeps its
  sign under particle-hole, so the positive-semidefinite ``Sigma_1`` came out negative.

The detector also only compared first moments and never checked that the interaction is
particle-hole invariant. Now no block is its own partner, the solver's block structures carry no
particle-hole relations (those blocks are computed directly), and ``build_full_greens_function``
rejects them. No production archive in ``impmod_tests`` has a particle-hole pair.
"""

import numpy as np
import pytest

from impurityModel.ed.block_structure import build_block_structure
from impurityModel.ed.greens_function import build_full_greens_function
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import impurity_block_structure

EPS = 0.5
W = np.linspace(-2.0, 2.0, 41) + 0.1j


def _level_gf(eps, z):
    return (1.0 / (np.asarray(z) - eps))[:, None, None]


def test_a_block_is_not_its_own_particle_hole_partner():
    """A block whose first moment is (numerically) zero is not listed as its own image."""
    bs = build_block_structure(None, mat=np.diag([0.0, 0.7]))
    for i, partners in enumerate(bs.particle_hole_blocks):
        assert i not in partners, bs
    for i, partners in enumerate(bs.particle_hole_transposed_blocks):
        assert i not in partners, bs


def test_a_zero_moment_block_keeps_its_own_green_function():
    """The consequence the self-inclusion had: the representative overwritten by -conj(G)."""
    bs = build_block_structure(None, mat=np.diag([0.0, 0.7]))
    g_ineq = [_level_gf(0.0, W), _level_gf(0.7, W)][: len(bs.inequivalent_blocks)]
    full = build_full_greens_function(g_ineq, bs)
    np.testing.assert_allclose(full[:, 0, 0], _level_gf(0.0, W)[:, 0, 0])


def test_the_solver_block_structure_computes_particle_hole_pairs_directly():
    """Two decoupled levels at +EPS and -EPS: a first-moment particle-hole pair. The solver's
    block structure must list both as inequivalent (each gets its own Green's function)."""
    h = ManyBodyOperator({((0, "c"), (0, "a")): EPS, ((1, "c"), (1, "a")): -EPS})
    bs = impurity_block_structure(h, [0, 1], n_orb=2)
    assert not any(bs.particle_hole_blocks) and not any(bs.particle_hole_transposed_blocks), bs
    assert sorted(bs.inequivalent_blocks) == [0, 1], bs


def test_reconstruction_rejects_particle_hole_relations():
    """An elementwise -conj is not the particle-hole image on the real axis or for every moment;
    the reconstruction refuses rather than guessing which of those it was handed."""
    bs = build_block_structure(None, mat=np.diag([EPS, -EPS]))
    assert bs.particle_hole_blocks[bs.inequivalent_blocks[0]] == [1], bs
    with pytest.raises(ValueError, match="particle-hole"):
        build_full_greens_function([_level_gf(EPS, W)], bs)
