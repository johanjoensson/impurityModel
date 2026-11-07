"""The DC-reference causality diagnostic is the anti-Hermitian part of Delta, not an elementwise Im."""

import numpy as np

from impurityModel.ed.dc_reference import _continuum_fermi_filling


def _violation(delta_w):
    n_orb = delta_w.shape[1]
    w = np.linspace(-2.0, 2.0, delta_w.shape[0])
    return _continuum_fermi_filling(np.zeros((n_orb, n_orb), dtype=complex), delta_w, w, 0.05, 0.0)[2]


def _const(mat, n_w=9):
    return np.repeat(np.asarray(mat, dtype=complex)[None], n_w, axis=0)


def test_causal_delta_with_positive_offdiagonal_imaginary_part_is_not_flagged():
    # (Delta - Delta^dagger)/2i = -[[1, 0.5], [0.5, 1]] is negative definite: causal,
    # although Im Delta_01 = +0.5 > 0 elementwise.
    delta = -1j * np.array([[1.0, -0.5], [-0.5, 1.0]])
    assert np.max(_const(delta).imag) > 0
    assert _violation(_const(delta)) == 0.0


def test_acausal_delta_is_flagged_by_its_largest_eigenvalue():
    assert _violation(_const(1j * np.eye(2))) > 0.99


def test_complex_hopping_is_handled():
    # Complex off-diagonal V: anti-Hermitian part [[-1, 0.2i], [-0.2i, -1]] has eigenvalues -1 +- 0.2: causal.
    delta = np.array([[-1j, 0.2j * 1j], [0.2j * 1j * -1, -1j]])
    anti = (delta - delta.conj().T) / 2j
    assert np.linalg.eigvalsh(anti).max() < 0
    assert _violation(_const(delta)) == 0.0
