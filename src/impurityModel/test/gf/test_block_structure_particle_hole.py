"""Particle-hole block equivalences (ledger row C1, ``doc/reviews/gf_review.md``).

``block_structure`` marks block ``B`` as the particle-hole image of ``A`` when their first
moments satisfy ``M_B = -conj(M_A)``, and ``build_full_greens_function`` then fills ``B`` from
``A`` as ``-conj(G_A)``. These tests pin what that map must do, on exactly solvable inputs:

* no block is its own partner (``_particle_hole_blocks_matrix`` scans ``blocks[i:]``, so a block
  with ``Re M = 0`` currently lists itself and its own G is overwritten with ``-conj(G)``);
* on the Matsubara axis ``G_B(i w) = -conj(G_A(i w))`` holds, but on the retarded real axis the
  image is ``G_B(w) = -conj(G_A(-w))``: without the frequency reversal the reconstructed block
  carries negative spectral weight;
* the second self-energy moment (the ``1/z`` coefficient) is positive semidefinite for any
  physical system, and under particle-hole it maps with a *plus* sign, so ``-conj`` turns it
  negative semidefinite. ``selfenergy.calc_selfenergy`` routes the moment matrices through the
  same ``build_full_greens_function``.

None of the production archives in ``impmod_tests`` has a particle-hole pair (their block
structures print empty particle-hole lists), so this is latent there -- but it is silently wrong
whenever it fires, and it fires after ``_check_gf_physical`` has already passed.
"""

import numpy as np
import pytest

from impurityModel.ed.block_structure import build_block_structure
from impurityModel.ed.greens_function import build_full_greens_function

C1 = "C1: particle-hole block reconstruction (doc/reviews/gf_review.md)"

EPS = 0.5
DELTA = 0.1
W = np.linspace(-2.0, 2.0, 41)  # symmetric about 0, so a reversal would be available
IW = 1j * np.pi * 0.2 * (2 * np.arange(8) + 1)


def _level_gf(eps, z):
    """Exact G of one non-interacting level, as a ``(n_z, 1, 1)`` block."""
    return (1.0 / (np.asarray(z) - eps))[:, None, None]


def _pair_structure():
    """Two 1x1 blocks with first moments +EPS and -EPS: a particle-hole pair."""
    bs = build_block_structure(None, mat=np.diag([EPS, -EPS]))
    assert bs.particle_hole_blocks[bs.inequivalent_blocks[0]] == [1], bs
    return bs


@pytest.mark.xfail(strict=True, reason=C1)
def test_a_block_is_not_its_own_particle_hole_partner():
    """A block whose first moment is (numerically) zero must not be listed as its own image."""
    bs = build_block_structure(None, mat=np.diag([0.0, 0.7]))
    for i, partners in enumerate(bs.particle_hole_blocks):
        assert i not in partners, bs
    for i, partners in enumerate(bs.particle_hole_transposed_blocks):
        assert i not in partners, bs


@pytest.mark.xfail(strict=True, reason=C1)
def test_a_self_partnered_block_keeps_its_own_green_function():
    """The consequence of self-inclusion: the representative block is overwritten by -conj(G)."""
    bs = build_block_structure(None, mat=np.diag([0.0, 0.7]))
    z = W + 1j * DELTA
    g_ineq = [_level_gf(0.0, z), _level_gf(0.7, z)][: len(bs.inequivalent_blocks)]
    full = build_full_greens_function(g_ineq, bs)
    np.testing.assert_allclose(full[:, 0, 0], _level_gf(0.0, z)[:, 0, 0])


def test_particle_hole_image_on_the_matsubara_axis():
    """On i w the -conj map is the correct image: G_B(i w) = 1/(i w + EPS)."""
    bs = _pair_structure()
    full = build_full_greens_function([_level_gf(EPS, IW)], bs)
    np.testing.assert_allclose(full[:, 1, 1], _level_gf(-EPS, IW)[:, 0, 0], atol=1e-14)


@pytest.mark.xfail(strict=True, reason=C1)
def test_particle_hole_image_on_the_real_axis():
    """On w + i delta the image needs w -> -w; -conj(G_A(w)) is advanced and sign-flipped."""
    bs = _pair_structure()
    z = W + 1j * DELTA
    full = build_full_greens_function([_level_gf(EPS, z)], bs)
    got = full[:, 1, 1]
    assert np.all(got.imag <= 1e-14), "reconstructed block has negative spectral weight"
    np.testing.assert_allclose(got, _level_gf(-EPS, z)[:, 0, 0], atol=1e-12)


@pytest.mark.xfail(strict=True, reason=C1)
def test_the_second_self_energy_moment_stays_positive_semidefinite():
    """Sigma_1 (the 1/z coefficient) is a variance, PSD for any physical system; the moment path
    of calc_selfenergy maps it with the same -conj, which makes the image negative."""
    bs = _pair_structure()
    sigma_1_a = np.array([[0.8]])
    full = build_full_greens_function([sigma_1_a], bs)
    assert np.all(np.linalg.eigvalsh(full) >= 0), full
