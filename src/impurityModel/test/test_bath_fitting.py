import numpy as np
import pytest

from impurityModel.ed.bath_fitting import BathFitter


def test_bath_fitter_real_single_orbital():
    w = np.linspace(-5, 5, 201)
    eta = 0.1
    n_imp = 1

    z = w + 1j * eta
    # Target: two distinct poles
    # eps1 = -1.5, V1^2 = 0.3 => V1 = sqrt(0.3) ~ 0.5477
    # eps2 =  2.0, V2^2 = 0.5 => V2 = sqrt(0.5) ~ 0.7071
    delta_target = (0.5 / (z - 2.0) + 0.3 / (z + 1.5))[:, None, None]

    fitter = BathFitter(w, delta_target, n_bath=2, n_imp=n_imp, eta=eta, complex_v=False)

    # We enforce M0 = sum V^2 = 0.8
    fitter.set_moment(0, np.array([[0.8]]), weight=10.0)

    np.random.seed(42)
    eps, v = fitter.fit(n_starts=10)

    assert fitter.best_cost < 1e-5

    # Check that eps matches up to permutation
    eps_sorted = np.sort(eps)
    np.testing.assert_allclose(eps_sorted, [-1.5, 2.0], atol=1e-3)

    # Check that V squared matches
    v_sq_sorted = np.sort(v.flatten() ** 2)
    np.testing.assert_allclose(v_sq_sorted, [0.3, 0.5], atol=1e-3)


def test_bath_fitter_moments():
    w = np.linspace(-2, 2, 51)
    eta = 0.1
    n_imp = 2
    n_bath = 3

    # We just want to check if the moment targets are physically evaluated
    delta_target = np.zeros((len(w), n_imp, n_imp), dtype=complex)

    fitter = BathFitter(w, delta_target, n_bath=n_bath, n_imp=n_imp, eta=eta, complex_v=False)

    M0_target = np.array([[1.0, 0.2], [0.2, 1.0]])
    M1_target = np.array([[0.0, 0.1], [0.1, 0.0]])

    fitter.set_moment(0, M0_target, weight=1.0)
    fitter.set_moment(1, M1_target, weight=1.0)
    fitter.weight_real = 0.0
    fitter.weight_imag = 0.0

    eps, v = fitter.fit(n_starts=5)

    v_tensor = v[:, None, :] * v[None, :, :]
    fit_M0 = np.sum(v_tensor, axis=-1)
    fit_M1 = np.sum(v_tensor * eps[None, None, :], axis=-1)

    np.testing.assert_allclose(fit_M0, M0_target, atol=1e-3)
    np.testing.assert_allclose(fit_M1, M1_target, atol=1e-3)


def test_bath_fitter_matsubara():
    # Matsubara frequencies wn = (2n+1) pi T for fermions
    wn = np.array([(2 * n + 1) * np.pi * 0.01 for n in range(50)])
    n_imp = 1

    z = 1j * wn
    # Target: one pole at eps=-0.5, V=0.8 => V^2=0.64
    delta_target = (0.64 / (z - (-0.5)))[:, None, None]

    fitter = BathFitter(wn, delta_target, n_bath=1, n_imp=n_imp, complex_v=False, matsubara=True)

    np.random.seed(123)
    eps, v = fitter.fit(n_starts=5)

    assert fitter.best_cost < 1e-5
    np.testing.assert_allclose(eps, [-0.5], atol=1e-3)
    np.testing.assert_allclose(v.flatten() ** 2, [0.64], atol=1e-3)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_zero_hybridization_drives_couplings_to_zero():
    """A vanishing target hybridization (and no moment constraints) is fit best by V=0."""
    w = np.linspace(-3, 3, 61)
    delta_target = np.zeros((len(w), 1, 1), dtype=complex)
    fitter = BathFitter(w, delta_target, n_bath=2, n_imp=1, eta=0.1, complex_v=False)

    np.random.seed(0)
    _, v = fitter.fit(n_starts=5)

    assert fitter.best_cost < 1e-6
    assert np.max(np.abs(v)) < 1e-2


def test_unpack_real_convention():
    """``unpack`` splits x into eps (n_bath) then a row-major (n_imp, n_bath) V."""
    w = np.linspace(-1, 1, 5)
    fitter = BathFitter(w, np.zeros((5, 1, 2), dtype=complex), n_bath=2, n_imp=1, eta=0.1)
    x = np.array([0.5, -0.5, 1.0, 2.0])  # eps=[0.5,-0.5], v=[[1.0, 2.0]]
    eps, v = fitter.unpack(x)
    np.testing.assert_allclose(eps, [0.5, -0.5])
    np.testing.assert_allclose(v, [[1.0, 2.0]])


def test_unpack_complex_convention():
    """With ``complex_v`` the tail of x holds the imaginary parts of V."""
    w = np.linspace(-1, 1, 5)
    fitter = BathFitter(w, np.zeros((5, 1, 2), dtype=complex), n_bath=2, n_imp=1, eta=0.1, complex_v=True)
    x = np.array([0.5, -0.5, 1.0, 2.0, 3.0, 4.0])  # eps, v_real=[1,2], v_imag=[3,4]
    eps, v = fitter.unpack(x)
    np.testing.assert_allclose(eps, [0.5, -0.5])
    np.testing.assert_allclose(v, [[1.0 + 3.0j, 2.0 + 4.0j]])
