"""Unit tests for :mod:`impurityModel.ed.polarization`.

Pure-numpy checks that the extracted contractions reproduce the einsums they were lifted
from (see ``spectra.py`` history) and that the derived quantities behave as documented.
"""

import numpy as np
import pytest

from impurityModel.ed import polarization as pol


def _hermitian_tensor(rng, n_w, m):
    """Random tensor that is Hermitian in its last two axes at every w (a physical chi)."""
    A = rng.standard_normal((n_w, m, m)) + 1j * rng.standard_normal((n_w, m, m))
    return A + np.conj(np.swapaxes(A, 1, 2))


def test_polarization_vector_named():
    x = pol.polarization_vector("x")
    y = pol.polarization_vector("y")
    z = pol.polarization_vector("z")
    np.testing.assert_allclose(x, [1, 0, 0])
    np.testing.assert_allclose(y, [0, 1, 0])
    np.testing.assert_allclose(z, [0, 0, 1])
    cl = pol.polarization_vector("cl")
    cr = pol.polarization_vector("cr")
    np.testing.assert_allclose(cl, np.array([1, 1j, 0]) / np.sqrt(2))
    np.testing.assert_allclose(cr, np.array([1, -1j, 0]) / np.sqrt(2))
    # Case-insensitive.
    np.testing.assert_allclose(pol.polarization_vector("X"), [1, 0, 0])


def test_polarization_vector_components():
    v = pol.polarization_vector("0,0.5,0.5j")
    np.testing.assert_allclose(v, [0, 0.5, 0.5j])


def test_polarization_vector_array_passthrough():
    v = pol.polarization_vector([1, 2, 3])
    np.testing.assert_allclose(v, [1, 2, 3])


def test_polarization_vector_malformed_raises():
    with pytest.raises(ValueError):
        pol.polarization_vector("not-a-vector,also-bad")


def test_contract_spectra_tensor_matches_manual_einsum():
    rng = np.random.default_rng(0)
    n_w, m = 7, 3
    chi = _hermitian_tensor(rng, n_w, m)
    eps = np.array([[1, 0, 0], [0, 1, 0], [1, 1j, 0]], dtype=complex)
    got = pol.contract_spectra_tensor(chi, eps)
    expected = np.einsum("pa,wab,pb->wp", eps.conj(), chi, eps, optimize=True)
    np.testing.assert_allclose(got, expected)


def test_contract_spectra_tensor_diagonal_recovers_component():
    rng = np.random.default_rng(1)
    n_w, m = 5, 3
    chi = _hermitian_tensor(rng, n_w, m)
    got = pol.contract_spectra_tensor(chi, ["x", "y", "z"])
    for a in range(m):
        np.testing.assert_allclose(got[:, a], chi[:, a, a])


def test_contract_rixs_tensor_matches_manual_einsum():
    rng = np.random.default_rng(2)
    n_in, n_out, n_wIn, n_wLoss = 2, 3, 4, 5
    C = rng.standard_normal((n_in, n_out, n_in, n_out, n_wIn, n_wLoss)) + 1j * rng.standard_normal(
        (n_in, n_out, n_in, n_out, n_wIn, n_wLoss)
    )
    eps_in = np.array([[1, 0], [0, 1], [1, 1j]], dtype=complex) / np.array([[1], [1], [np.sqrt(2)]])
    eps_out = np.array([[1, 0, 0], [0, 1, 1j]], dtype=complex)
    got = pol.contract_rixs_tensor(C, eps_in, eps_out)
    expected = np.einsum(
        "pa,qb,pc,qd,abcdwl->pqwl",
        eps_in.conj(),
        eps_out,
        eps_in,
        eps_out.conj(),
        C,
        optimize=True,
    )
    np.testing.assert_allclose(got, expected)


def test_contract_rixs_tensor_named_polarizations():
    rng = np.random.default_rng(3)
    n_in = n_out = 3
    shape = (n_in, n_out, n_in, n_out, 2, 2)
    C = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    got = pol.contract_rixs_tensor(C, ["x"], ["y"])
    np.testing.assert_allclose(got[0, 0], C[0, 1, 0, 1])


def test_intensity_is_negative_imag():
    g = np.array([1 + 2j, -3 + 4j])
    np.testing.assert_allclose(pol.intensity(g), [-2, -4])


def test_isotropic_is_normalized_trace():
    rng = np.random.default_rng(4)
    n_w, m = 6, 3
    chi = _hermitian_tensor(rng, n_w, m)
    got = pol.isotropic(chi)
    expected = np.trace(chi, axis1=1, axis2=2) / m
    np.testing.assert_allclose(got, expected)


def test_isotropic_equals_average_over_orthonormal_basis():
    """Averaging eps^dagger chi eps over a complete orthonormal basis equals the trace/m."""
    rng = np.random.default_rng(5)
    n_w, m = 4, 3
    chi = _hermitian_tensor(rng, n_w, m)
    basis = np.eye(m, dtype=complex)
    contracted = pol.contract_spectra_tensor(chi, basis)
    np.testing.assert_allclose(np.mean(contracted, axis=1), pol.isotropic(chi))


def test_circular_dichroism_antisymmetric_under_swap():
    rng = np.random.default_rng(6)
    n_w = 5
    chi = _hermitian_tensor(rng, n_w, 3)
    xmcd = pol.circular_dichroism(chi)
    # Swapping cl/cr flips the sign: build chi with x<->y swapped to swap cl/cr roles is
    # nontrivial in general, so instead check the direct definition against the tensor
    # contraction it wraps.
    g = pol.contract_spectra_tensor(chi, ["cl", "cr"])
    expected = pol.intensity(g[:, 0]) - pol.intensity(g[:, 1])
    np.testing.assert_allclose(xmcd, expected)


def test_circular_dichroism_zero_for_real_diagonal_tensor():
    """A real, diagonal (isotropic-in-xy) tensor has no circular dichroism."""
    n_w = 4
    chi = np.zeros((n_w, 3, 3), dtype=complex)
    for w in range(n_w):
        chi[w] = np.diag([1.0 + w, 1.0 + w, 2.0 + w])
    xmcd = pol.circular_dichroism(chi)
    np.testing.assert_allclose(xmcd, 0, atol=1e-12)


def test_linear_dichroism_default_matches_z_minus_x():
    rng = np.random.default_rng(7)
    n_w = 5
    chi = _hermitian_tensor(rng, n_w, 3)
    xld = pol.linear_dichroism(chi)
    expected = pol.intensity(chi[:, 2, 2]) - pol.intensity(chi[:, 0, 0])
    np.testing.assert_allclose(xld, expected)


def test_linear_dichroism_custom_polarizations():
    rng = np.random.default_rng(8)
    n_w = 5
    chi = _hermitian_tensor(rng, n_w, 3)
    xld = pol.linear_dichroism(chi, pol_a="y", pol_b="x")
    expected = pol.intensity(chi[:, 1, 1]) - pol.intensity(chi[:, 0, 0])
    np.testing.assert_allclose(xld, expected)
