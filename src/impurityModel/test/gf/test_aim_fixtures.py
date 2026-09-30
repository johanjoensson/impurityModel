"""The F-NiO fixture and its exact reference, checked at a size where everything is verifiable.

The basis-size comparison is only as good as its reference: a wrong Kanamori sign or a transposed
side in ``reference_G`` would give a clean-looking benchmark of the wrong physics. So the fixture is
pinned three ways -- against closed-form impurity energies, against its own symmetries, and against
the production driver's uncapped answer on both frequency axes.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.greens_function import get_Greens_function
from impurityModel.test.support.aim_fixtures import (
    N_IMP,
    build_nio_like,
    charge_transfer_weight,
    free_G_inverse,
    interaction_increment,
    reference_G,
    self_energy,
    spin_of,
)

NB = 2  # 12 spin-orbitals: a 15-determinant ground-state sector, removal sectors of 90 and 20


def test_the_isolated_high_spin_ground_state_has_the_closed_form_energy():
    """V = 0: two parallel spins in different orbitals cost 2 eps_d + U' - J = 2 eps_d + U - 3J, and the
    filled bath adds the sum of its levels."""
    U, J = 8.0, 1.0
    aim = build_nio_like(NB, v_eff=0.0, U=U, J=J)
    eps_d = 4.0 - interaction_increment(U, J)
    levels = 3.0 * np.linspace(-0.5, 0.5, NB)
    expected = 2 * eps_d + (U - 3 * J) + N_IMP * np.sum(levels)
    assert aim.e0 == pytest.approx(expected, abs=1e-9)
    assert aim.weights[2] == pytest.approx(1.0) and sum(w for n, w in aim.weights.items() if n != 2) < 1e-12


def test_the_charge_transfer_energy_is_what_was_asked_for():
    """At V = 0 the lowest excitation in the ground-state sector is d8 -> d9L: one electron from the top
    filled bath level to the impurity, costing ``delta_ct - width/2`` (``delta_ct`` is the cost from the
    bath *centre*). Measured from the dense spectrum, not assumed."""
    delta_ct, width = 4.0, 3.0
    aim = build_nio_like(NB, v_eff=0.0, delta_ct=delta_ct, width=width)
    n_up0 = 2 + 2 * NB
    evals = aim._sector_cache[(n_up0, aim.n_electrons - n_up0)][2]
    assert evals[1] - evals[0] == pytest.approx(delta_ct - width / 2, abs=1e-9)


def test_the_operator_is_hermitian_and_conserves_both_spin_populations():
    aim = build_nio_like(NB, v_eff=1.3)
    assert aim.hOp.is_hermitian()
    n_up0 = 2 + 2 * NB
    n_dn0 = aim.n_electrons - n_up0
    for sector in ((n_up0, n_dn0), (n_up0 - 1, n_dn0), (n_up0, n_dn0 - 1)):
        dets = aim.sector(*sector)
        keys = {bytes(d.to_bytearray()) for d in dets}
        from impurityModel.ed.ManyBodyUtils import ManyBodyState

        image = aim.hOp.apply_block(ManyBodyState.from_states([ManyBodyState({d: 1.0 + 0j}) for d in dets[:6]]), 0.0)
        assert {bytes(k.to_bytearray()) for k in image.keys()} <= keys


def test_spin_layout_pairs_each_bath_star_with_its_impurity_spin_orbital():
    assert [spin_of(i, NB) for i in range(N_IMP)] == [0, 0, 1, 1]
    assert [spin_of(N_IMP + m * NB + k, NB) for m in range(N_IMP) for k in range(NB)] == [0] * 2 * NB + [1] * 2 * NB


@pytest.mark.parametrize("target", [0.05, 0.15, 0.27])
def test_calibration_hits_the_requested_charge_transfer_weight(target):
    aim = build_nio_like(NB, target_d9L=target)
    assert charge_transfer_weight(aim) == pytest.approx(target, abs=1e-6)


def test_the_charge_transfer_weight_grows_with_the_hybridization():
    weights = [charge_transfer_weight(build_nio_like(NB, v_eff=v)) for v in (0.2, 0.5, 1.0, 2.0)]
    assert weights == sorted(weights) and weights[0] < weights[-1]


def _driver_G(aim, method, matsubara, omega, delta=0.3):
    basis = aim.basis(sorted(aim.gs.keys()), comm=MPI.COMM_SELF)
    mat, real, _report = get_Greens_function(
        matsubara_mesh=matsubara,
        omega_mesh=omega,
        psis=[aim.gs],
        es=[aim.e0],
        tau=1e-3,
        basis=basis,
        hOp=aim.hOp,
        delta=delta,
        blocks=[aim.imp],
        verbose=False,
        verbose_extra=False,
        reort="full" if method == "lanczos" else None,
        dN=None,
        occ_cutoff=0.0,
        slaterWeightMin=0.0,
        sparse=True,
        gf_method=method,
    )
    return mat, real


@pytest.mark.parametrize("method", ["bicgstab", "lanczos"])
def test_the_dense_reference_matches_the_production_driver_on_both_axes(method, monkeypatch):
    """The discriminating check of the reference: sides, transposes and signs all have to be right."""
    monkeypatch.setenv("GF_BICGSTAB_ATOL", "1e-12")
    aim = build_nio_like(NB, target_d9L=0.15)
    matsubara = 1j * np.pi * 0.4 * (2 * np.arange(6) + 1)
    omega = np.linspace(-14.0, 6.0, 11)
    delta = 0.3
    mat, real = _driver_G(aim, method, matsubara, omega, delta)
    np.testing.assert_allclose(mat[0], reference_G(aim, matsubara), atol=1e-7)
    np.testing.assert_allclose(real[0], reference_G(aim, omega + 1j * delta), atol=1e-7)


def test_the_reference_is_causal_and_obeys_the_sum_rule():
    aim = build_nio_like(NB, target_d9L=0.15)
    omega = np.linspace(-40.0, 40.0, 4001)
    delta = 0.05
    G = reference_G(aim, omega + 1j * delta)
    assert np.all(np.diagonal(G.imag, axis1=1, axis2=2) <= 1e-12)
    # integral of -Im G_ii / pi over a broad window is the spectral weight {c_i, c_i^+} = 1
    weight = -np.trapezoid(np.diagonal(G.imag, axis1=1, axis2=2), omega, axis=0) / np.pi
    np.testing.assert_allclose(weight, 1.0, atol=2e-2)


def test_the_non_interacting_limit_gives_zero_self_energy():
    """U = J = 0, eps_d matched: G equals the free G0, so Sigma vanishes -- pins free_G_inverse."""
    aim = build_nio_like(NB, v_eff=1.0, U=0.0, J=0.0)
    z = np.linspace(-6, 6, 7) + 0.4j
    sigma = self_energy(reference_G(aim, z), free_G_inverse(aim, z))
    np.testing.assert_allclose(sigma, 0.0, atol=1e-9)
