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
    block_krylov_basis,
    build_nio_like,
    build_semicircle_siam,
    charge_transfer_weight,
    free_G_inverse,
    geometry_variants,
    interaction_increment,
    is_filled,
    linked_chain_h1,
    natural_orbital_rotation,
    one_body_density,
    reference_G,
    rotate_bath,
    self_energy,
    semicircle_star,
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


# --- F-metal: the semicircular-bath SIAM -------------------------------------------------------------


def test_the_semicircle_star_has_unit_weight_symmetric_levels_and_the_continuum_tail():
    D, v = 0.5, 0.7
    levels, couplings = semicircle_star(7, D, v)
    assert np.sum(couplings**2) == pytest.approx(v**2)
    np.testing.assert_allclose(np.sort(levels), -np.sort(levels)[::-1], atol=1e-14)
    assert np.min(np.abs(levels)) < 1e-14  # odd n_b: a level at the Fermi energy
    # Delta(w) -> v^2 / w, and approaches the continuum 2 v^2/D^2 (w - sqrt(w^2 - D^2)) away from the band
    w = 6.0
    delta = np.sum(couplings**2 / (w - levels))
    assert delta == pytest.approx(2 * v**2 / D**2 * (w - np.sqrt(w**2 - D**2)), rel=2e-3)


@pytest.mark.parametrize("n_b", [3, 5])
def test_particle_hole_symmetry_pins_re_sigma_to_half_U_on_the_matsubara_axis(n_b):
    """At the symmetric point Re Sigma(i w_n) = U/2 exactly, for every n: a sign, factor or reference
    error anywhere in the Hubbard term, the impurity level or G breaks it."""
    U = 0.5
    aim = build_semicircle_siam(n_b, U=U, v=0.4)
    mats = 1j * np.pi * 0.05 * (2 * np.arange(6) + 1)
    sigma = self_energy(reference_G(aim, mats), free_G_inverse(aim, mats))
    np.testing.assert_allclose(sigma[:, 0, 0].real, U / 2, atol=1e-9)
    np.testing.assert_allclose(sigma[:, 1, 1].real, U / 2, atol=1e-9)


def test_the_metal_ground_state_is_a_half_filled_singlet_with_a_symmetric_occupation():
    aim = build_semicircle_siam(5, v=0.5)
    w = aim.weights
    assert sum(w.values()) == pytest.approx(1.0)
    assert w[0] == pytest.approx(w[2], abs=1e-9)  # particle-hole symmetric: empty and doubly occupied match
    assert w[1] > 0.3  # a metal still has a local moment, but the weight is spread over all occupations


def test_the_metal_without_interaction_has_zero_self_energy():
    aim = build_semicircle_siam(5, U=0.0, v=0.5)
    z = np.linspace(-0.6, 0.6, 7) + 0.1j
    np.testing.assert_allclose(self_energy(reference_G(aim, z), free_G_inverse(aim, z)), 0.0, atol=1e-9)


@pytest.mark.parametrize("keep", [None, 40])
@pytest.mark.parametrize("method", ["bicgstab", "lanczos"])
def test_the_metal_reference_matches_the_driver_for_the_exact_and_a_truncated_ground_state(method, keep, monkeypatch):
    """A truncated ground state is not an eigenstate of H, but G for fixed seeds and energy is a
    well-defined resolvent, so the dense reference must still match the production driver on it."""
    monkeypatch.setenv("GF_BICGSTAB_ATOL", "1e-12")
    aim = build_semicircle_siam(5, v=0.5, gs_keep=keep)
    matsubara = 1j * np.pi * 0.1 * (2 * np.arange(5) + 1)
    omega = np.linspace(-1.2, 1.2, 9)
    mat, real = _driver_G(aim, method, matsubara, omega, 0.05)
    np.testing.assert_allclose(mat[0], reference_G(aim, matsubara), atol=1e-7)
    np.testing.assert_allclose(real[0], reference_G(aim, omega + 0.05j), atol=1e-7)


def test_a_truncated_ground_state_is_variational_and_has_a_smaller_support():
    exact = build_semicircle_siam(5, v=0.5)
    cut = build_semicircle_siam(5, v=0.5, gs_keep=40)
    assert len(list(cut.gs.keys())) <= 40 < len(list(exact.gs.keys()))
    assert cut.e0 >= exact.e0 - 1e-12 and cut.exact_e0 == pytest.approx(exact.e0)
    assert sum(abs(a[0]) ** 2 for _d, a in cut.gs.items()) == pytest.approx(1.0)


# --- exact changes of the bath basis -------------------------------------------------------------------


def test_block_krylov_basis_is_orthogonal_tridiagonalizes_and_completes_a_closed_space():
    rng = np.random.default_rng(7)
    levels = np.sort(rng.normal(size=6))
    h = np.diag(levels)
    v = rng.normal(size=6)
    Q = block_krylov_basis(h, v[:, None])
    np.testing.assert_allclose(Q.T @ Q, np.eye(6), atol=1e-12)
    T = Q.T @ h @ Q
    assert np.max(np.abs(np.triu(T, 2))) < 1e-10 and np.max(np.abs(np.tril(T, -2))) < 1e-10
    np.testing.assert_allclose(Q.T @ v, [np.linalg.norm(v), 0, 0, 0, 0, 0], atol=1e-12)
    # a coupling that misses a level closes the Krylov space early; the rest is completed orthonormally
    v2 = v.copy()
    v2[3] = 0.0
    Q2 = block_krylov_basis(np.diag(np.where(np.arange(6) == 3, levels[3], levels)), v2[:, None])
    np.testing.assert_allclose(Q2.T @ Q2, np.eye(6), atol=1e-12)


@pytest.mark.parametrize(
    "build",
    [lambda: build_nio_like(2, target_d9L=0.15), lambda: build_semicircle_siam(5, v=0.5)],
    ids=["F-NiO", "F-metal"],
)
def test_every_bath_basis_gives_the_same_ground_state_energy_G_and_Sigma(build):
    """The decisive check of the rotations: a unitary change of the bath cannot move any impurity
    observable, so the dense reference must agree in all four bases to round-off."""
    aim = build()
    z = np.linspace(-1.0, 1.0, 5) + 0.15j
    mats = 1j * np.pi * 0.1 * (2 * np.arange(4) + 1)
    ref_G, ref_S = reference_G(aim, z), self_energy(reference_G(aim, mats), free_G_inverse(aim, mats))
    for name, v in geometry_variants(aim).items():
        assert v.basis_name == name and v.hOp.is_hermitian()
        assert v.e0 == pytest.approx(aim.e0, abs=1e-10)
        np.testing.assert_allclose(reference_G(v, z), ref_G, atol=1e-9, err_msg=name)
        np.testing.assert_allclose(
            self_energy(reference_G(v, mats), free_G_inverse(v, mats)), ref_S, atol=1e-9, err_msg=name
        )


def test_the_chain_basis_is_tridiagonal_with_the_impurity_at_one_end():
    aim = build_nio_like(3, target_d9L=0.15)
    chain = geometry_variants(aim)["chain"]
    n, nb = aim.n_imp, aim.n_b
    for m in aim.imp:
        sites = slice(n + m * nb, n + (m + 1) * nb)
        block = chain.h1[sites, sites]
        assert np.max(np.abs(np.triu(block, 2))) < 1e-10
        coupling = chain.h1[m, sites]
        assert abs(coupling[0]) > 1e-3 and np.max(np.abs(coupling[1:])) < 1e-10


def test_the_natural_basis_diagonalizes_the_bath_density_matrix_and_orders_it_by_occupation():
    aim = build_semicircle_siam(5, v=0.5)
    natural = geometry_variants(aim)["natural"]
    rho = one_body_density(natural)
    assert np.max(np.abs(rho - np.diag(np.diag(rho)))) < 1e-9
    n = aim.n_imp
    for positions in ([p for p in range(len(aim.bath)) if spin_of(n + p, aim.n_b, n) == s] for s in (0, 1)):
        occ = np.diag(rho)[positions]
        assert np.all(np.diff(occ) <= 1e-12)  # filled first


def test_in_the_natural_chain_basis_the_impurity_couples_only_to_the_head_of_each_chain():
    aim = build_semicircle_siam(5, v=0.5)
    nc = geometry_variants(aim)["natural-chains"]
    n = aim.n_imp
    occupations = natural_orbital_rotation(aim)[1]
    for spin in (0, 1):
        positions = [p for p in range(len(aim.bath)) if spin_of(n + p, aim.n_b, n) == spin]
        n_filled = int(np.sum(is_filled(occupations[positions])))
        imp = [m for m in aim.imp if spin_of(m, aim.n_b, n) == spin]
        coupling = nc.h1[np.ix_(imp, [n + p for p in positions])]
        heads = {0, n_filled} & set(range(len(positions)))
        away = [k for k in range(len(positions)) if k not in heads]
        assert np.max(np.abs(coupling[:, away])) < 1e-10


def test_a_bath_rotation_refuses_to_mix_spins_or_to_be_non_orthogonal():
    aim = build_semicircle_siam(3, v=0.5)
    n_bath = len(aim.bath)
    mix = np.eye(n_bath)
    mix[0, n_bath - 1] = mix[n_bath - 1, 0] = 0.5
    with pytest.raises(ValueError):
        rotate_bath(aim, mix, "bad")
    with pytest.raises(ValueError, match="orthogonal"):
        rotate_bath(aim, 2.0 * np.eye(n_bath), "bad")


# --- the linked double chain (rspt2spectra) -----------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [lambda: build_nio_like(2, target_d9L=0.15), lambda: build_semicircle_siam(5, v=0.5)],
    ids=["F-NiO", "F-metal"],
)
def test_the_linked_chain_is_an_exact_change_of_the_bath(build):
    """Same impurity block, same hybridization function, same ground-state energy, G and Sigma: the linked
    chain of ``rspt2spectra`` is a unitary rotation of the star bath, so it can be compared with the others."""
    pytest.importorskip("rspt2spectra.edchain")
    aim = build()
    linked = geometry_variants(aim, names=("linked-chain",))["linked-chain"]
    n = aim.n_imp
    np.testing.assert_array_equal(linked.h1[:n, :n], aim.h1[:n, :n])
    z = np.linspace(-1.0, 1.0, 5) + 0.15j
    mats = 1j * np.pi * 0.1 * (2 * np.arange(4) + 1)
    np.testing.assert_allclose(free_G_inverse(linked, z), free_G_inverse(aim, z), atol=1e-10)  # Delta(z)
    assert linked.e0 == pytest.approx(aim.e0, abs=1e-10) and linked.hOp.is_hermitian()
    np.testing.assert_allclose(reference_G(linked, z), reference_G(aim, z), atol=1e-9)
    np.testing.assert_allclose(
        self_energy(reference_G(linked, mats), free_G_inverse(linked, mats)),
        self_energy(reference_G(aim, mats), free_G_inverse(aim, mats)),
        atol=1e-9,
    )
    # a rotation of the bath alone keeps its spectrum
    n_b_total = len(aim.bath)
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(linked.h1[n:, n:])), np.sort(np.linalg.eigvalsh(aim.h1[n:, n:])), atol=1e-10
    )
    assert linked.h1.shape == aim.h1.shape and n_b_total > 0


def test_the_linked_chain_is_built_from_a_star_and_refuses_anything_else():
    pytest.importorskip("rspt2spectra.edchain")
    aim = build_semicircle_siam(5, v=0.5)
    chain = geometry_variants(aim, names=("chain",))["chain"]
    with pytest.raises(ValueError, match="star"):
        linked_chain_h1(chain)


def test_an_unknown_bath_basis_is_refused():
    aim = build_semicircle_siam(3, v=0.5)
    with pytest.raises(ValueError, match="unknown bath basis"):
        geometry_variants(aim, names=("star", "banana"))
