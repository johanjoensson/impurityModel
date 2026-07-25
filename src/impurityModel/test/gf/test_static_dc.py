"""Tests for the static double-counting schemes: fll_dc, amf_dc, sigma_inf_dc.

Dimer model shared with test_fixed_dc.py: two impurity spin-orbitals at energy EPS with a
Hubbard interaction U (J = 0, single spatial orbital), weakly coupled (hopping v) to two
valence bath spin-orbitals at energy EPS_B, chosen so the interacting/non-interacting ground
state has both impurity spin-orbitals occupied (N -> 2):

    fll_dc       -> U (N - 1/2) I = 4.5 I
    amf_dc       -> Sigma_static(u4, I) = 3 I           (uniform trial rho at N/n_imp = 1)
    sigma_inf_dc -> Sigma_static(u4, rho_imp) ~= 3 I     (rho_imp ~= I here, so agrees with AMF)

A split-dimer variant (impurity levels EPS1 != EPS2, deep valence baths so both still fill)
distinguishes Held's actual (anisotropic) density matrix from AMF/FLL's uniform trial: AMF and
FLL agree on a single uniform shift while Held's Sigma_static, evaluated at the true (diagonal
but unequal) occupations, produces different values for the two levels.
"""

import numpy as np
import pytest

from impurityModel.ed.double_counting import amf_dc, fll_dc, sigma_inf_dc
from impurityModel.ed.model import ImpurityModel

EPS = -1.0
U = 3.0
EPS_B = -4.0


def build_dimer_model(v, dc=None):
    h0 = np.zeros((4, 4), dtype=complex)
    for s in range(2):
        imp, bath = s, 2 + s
        h0[imp, imp] = EPS
        h0[bath, bath] = EPS_B
        h0[imp, bath] = v
        h0[bath, imp] = v
    u4 = np.zeros((4, 4, 4, 4), dtype=complex)
    u4[0, 1, 0, 1] = U
    u4[1, 0, 1, 0] = U
    return ImpurityModel.from_solver_matrix(
        h0,
        2,
        dc=dc,
        u4=u4,
        rot_to_spherical=np.eye(2, dtype=complex),
        bath_valence_conduction=([2, 3], []),
    )


def build_split_dimer_model(v, eps1, eps2, eps_b):
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 0] = eps1
    h0[1, 1] = eps2
    h0[2, 2] = eps_b
    h0[3, 3] = eps_b
    h0[0, 2] = v
    h0[2, 0] = v
    h0[1, 3] = v
    h0[3, 1] = v
    u4 = np.zeros((4, 4, 4, 4), dtype=complex)
    u4[0, 1, 0, 1] = U
    u4[1, 0, 1, 0] = U
    return ImpurityModel.from_solver_matrix(
        h0,
        2,
        dc=None,
        u4=u4,
        rot_to_spherical=np.eye(2, dtype=complex),
        bath_valence_conduction=([2, 3], []),
    )


def test_fll_dc_dimer_matches_analytic_uniform_shift():
    model = build_dimer_model(v=0.01)
    dc = fll_dc(model, tau=1e-3)
    assert np.allclose(dc, 4.5 * np.eye(2), atol=1e-4)


def test_amf_dc_dimer_matches_analytic_uniform_shift():
    model = build_dimer_model(v=0.01)
    dc = amf_dc(model, tau=1e-3)
    assert np.allclose(dc, 3.0 * np.eye(2), atol=1e-4)


def test_sigma_inf_dc_dimer_agrees_with_amf_when_occupation_is_uniform():
    model = build_dimer_model(v=0.01)
    dc = sigma_inf_dc(model, tau=1e-3)
    assert np.allclose(dc, 3.0 * np.eye(2), atol=1e-4)


def test_split_dimer_distinguishes_held_from_uniform_schemes():
    model = build_split_dimer_model(v=0.001, eps1=-1.0, eps2=5.0, eps_b=-100.0)

    fll = fll_dc(model, tau=1e-4)
    amf = amf_dc(model, tau=1e-4)
    held = sigma_inf_dc(model, tau=1e-4)

    assert np.allclose(fll, 1.5 * np.eye(2), atol=1e-3)
    assert np.allclose(amf, 1.5 * np.eye(2), atol=1e-3)
    # Held uses the actual (anisotropic) occupation: one level empty, one full.
    assert np.isclose(held[0, 0].real, 0.0, atol=1e-3)
    assert np.isclose(held[1, 1].real, 3.0, atol=1e-3)
    assert not np.allclose(held, amf, atol=0.1)


def test_fll_dc_explicit_n_u_j_needs_no_u4():
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 0] = EPS
    h0[1, 1] = EPS
    h0[2, 2] = EPS_B
    h0[3, 3] = EPS_B
    h0[0, 2] = 0.01
    h0[2, 0] = 0.01
    h0[1, 3] = 0.01
    h0[3, 1] = 0.01
    model_no_u4 = ImpurityModel.from_solver_matrix(
        h0, 2, dc=None, u4=None, rot_to_spherical=np.eye(2, dtype=complex), bath_valence_conduction=([2, 3], [])
    )
    dc = fll_dc(model_no_u4, n=2.0, u=U, j=0.0)
    assert np.allclose(dc, 4.5 * np.eye(2))


def test_sigma_inf_dc_explicit_rho_matches_amf_explicit_n():
    model = build_dimer_model(v=0.01)
    held = sigma_inf_dc(model, rho=np.eye(2, dtype=complex))
    amf = amf_dc(model, n=2.0)
    assert np.allclose(held, amf)


def test_amf_dc_explicit_n_matches_uniform_trial():
    model = build_dimer_model(v=0.01)
    dc = amf_dc(model, n=2.0)
    assert np.allclose(dc, 3.0 * np.eye(2))


def test_static_schemes_ignore_model_dc():
    # The DFT reference is the Fermi filling of the RAW h0 -- model.dc must not feed back into
    # it (the NiO n0 == 10 bug family). With dc = -3 the OLD h0 - dc reference pushed the
    # impurity level to +2 (empty, N ~ 0) and FLL settled at U(0 - 1/2) = -1.5; the raw-h0
    # occupation is N = 2 regardless of model.dc, so all three schemes must match the dc=None
    # fixture exactly.
    model = build_dimer_model(v=0.01)
    model_shifted = build_dimer_model(v=0.01, dc=-3.0 * np.eye(2, dtype=complex))
    assert np.allclose(fll_dc(model_shifted, tau=1e-3), fll_dc(model, tau=1e-3))
    assert np.allclose(amf_dc(model_shifted, tau=1e-3), amf_dc(model, tau=1e-3))
    assert np.allclose(sigma_inf_dc(model_shifted, tau=1e-3), sigma_inf_dc(model, tau=1e-3))
    assert np.allclose(fll_dc(model_shifted, tau=1e-3), 4.5 * np.eye(2), atol=1e-4)


def test_fll_dc_raises_without_u4_when_uj_not_overridden():
    h0 = np.zeros((2, 2), dtype=complex)
    h0[0, 0] = EPS
    h0[1, 1] = EPS_B
    model = ImpurityModel.from_solver_matrix(
        h0, 1, dc=None, u4=None, rot_to_spherical=np.eye(1, dtype=complex), bath_valence_conduction=([1], [])
    )
    with pytest.raises(ValueError, match="u4"):
        fll_dc(model, n=1.0)


def test_amf_dc_raises_without_u4():
    h0 = np.zeros((2, 2), dtype=complex)
    h0[0, 0] = EPS
    h0[1, 1] = EPS_B
    model = ImpurityModel.from_solver_matrix(
        h0, 1, dc=None, u4=None, rot_to_spherical=np.eye(1, dtype=complex), bath_valence_conduction=([1], [])
    )
    with pytest.raises(ValueError, match="u4"):
        amf_dc(model, n=1.0)
