"""The model-interaction builders against an independent Fock-space oracle.

Every comparison here is between *operators*, built as dense Fock-space matrices with a
self-contained Jordan-Wigner construction -- not between raw tensors, which can differ by index
permutations and Pauli-forbidden entries while describing the same Hamiltonian -- and not
through ManyBodyOperator, whose conventions are what is being pinned.
"""

import itertools

import numpy as np
import pytest

from impurityModel.ed import atomic_physics
from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix, uj_from_u4
from impurityModel.ed.interaction_models import (
    cubic_to_spherical_u4,
    density_density_u4,
    kanamori_u4,
    mlft_uvv,
    spatial_to_spin_u4,
    terms_to_tensor,
    terms_u4,
    validate_u4,
)
from impurityModel.ed.lie_algebra import rotate_two_body
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.ed.model import atomic_u4


def _annihilators(n):
    """Jordan-Wigner c_i on 2**n states; bit i of the state index is orbital i's occupation."""
    dim = 2**n
    ops = []
    for i in range(n):
        c = np.zeros((dim, dim))
        for state in range(dim):
            if state >> i & 1:
                sign = (-1) ** bin(state & ((1 << i) - 1)).count("1")
                c[state ^ (1 << i), state] = sign
        ops.append(c)
    return ops


def fock_matrix(u4):
    """Dense ``1/2 sum u4[i,j,k,l] c+_i c+_j c_l c_k`` on the full Fock space."""
    n = u4.shape[0]
    c = _annihilators(n)
    cd = [x.T for x in c]
    h = np.zeros((2**n, 2**n), dtype=complex)
    for i, j, k, l in zip(*np.nonzero(u4)):
        h += 0.5 * u4[i, j, k, l] * cd[i] @ cd[j] @ c[l] @ c[k]
    return h


def assert_same_operator(u4_a, u4_b):
    np.testing.assert_allclose(fock_matrix(u4_a), fock_matrix(u4_b), atol=1e-12)


def _number_sector_levels(u4, n_particles):
    h = fock_matrix(u4)
    idx = [s for s in range(h.shape[0]) if bin(s).count("1") == n_particles]
    return np.linalg.eigvalsh(h[np.ix_(idx, idx)])


def _siam_hand_u4(U):
    """The hand-built tensor of test_selfenergy_end_to_end._siam_model."""
    u4 = np.zeros((2, 2, 2, 2))
    u4[0, 1, 0, 1] = u4[1, 0, 1, 0] = U
    return u4


# ---------------------------------------------------------------------------- oracles


def test_single_orbital_forms_are_one_operator():
    """Slater l=0, Kanamori n=1, a one-term spatial list, density-density, and the hand-built
    SIAM tensor are all U n_up n_dn."""
    U = 2.7
    reference = _siam_hand_u4(U)
    for u4 in (
        atomic_u4(0, [U]),
        kanamori_u4(1, U),
        terms_u4(1, spatial=[[0, 0, 0, 0, U]]),
        density_density_u4([[U]]),
    ):
        assert_same_operator(u4, reference)
    # and that operator is U on the doubly occupied state only
    np.testing.assert_allclose(np.diag(fock_matrix(reference)).real, [0, 0, 0, U])


def test_two_orbital_term_list_matches_kanamori():
    """The index-order oracle: one spatial element per Kanamori parameter, with distinct values,
    must reproduce kanamori_u4. The single-orbital oracle cannot tell <pq|V|rs> from (pq|rs);
    this one can."""
    U, Up, J, Jp = 4.0, 2.3, 0.7, 0.45
    terms = [
        [0, 0, 0, 0, U],
        [1, 1, 1, 1, U],
        [0, 1, 0, 1, Up],
        [0, 1, 1, 0, J],
        [0, 0, 1, 1, Jp],
    ]
    assert_same_operator(terms_u4(2, spatial=terms), kanamori_u4(2, U, J=J, U_prime=Up, J_pair=Jp))


def test_two_orbital_kanamori_multiplets():
    """N=2 levels of the two-orbital Kanamori atom, each parameter moving a different level:
    triplet U'-J (x3), inter-orbital singlet U'+J, intra-orbital singlets U -+ J_p."""
    U, Up, J, Jp = 4.0, 2.3, 0.7, 0.45
    levels = _number_sector_levels(kanamori_u4(2, U, J=J, U_prime=Up, J_pair=Jp), 2)
    expected = sorted([Up - J] * 3 + [Up + J, U - Jp, U + Jp])
    np.testing.assert_allclose(levels, expected, atol=1e-12)


def test_kanamori_defaults_are_the_rotationally_invariant_point():
    U, J = 4.0, 0.7
    assert_same_operator(kanamori_u4(3, U, J=J), kanamori_u4(3, U, J=J, U_prime=U - 2 * J, J_pair=J))


def test_d_shell_slater_f0_only_is_kanamori_without_hund():
    U = 3.1
    assert_same_operator(atomic_u4(2, [U, 0, 0, 0, 0]), kanamori_u4(5, U, J=0.0))


def test_density_density_is_kanamori_without_spin_flip_or_pair_hopping():
    U, Up, J = 4.0, 2.3, 0.7
    n = 3
    u_opp = np.full((n, n), Up) + (U - Up) * np.eye(n)
    u_same = np.full((n, n), Up - J) - (Up - J) * np.eye(n)
    kan = kanamori_spatial_without_flips(n, U, Up, J)
    assert_same_operator(density_density_u4(u_opp, u_same), kan)


def kanamori_spatial_without_flips(n, U, Up, J):
    """Kanamori's density part only: drop the spin-flip and pair-hopping elements."""
    u = np.zeros((n,) * 4)
    for a in range(n):
        u[a, a, a, a] = U
        for b in range(n):
            if a != b:
                u[a, b, a, b] = Up
    u4 = spatial_to_spin_u4(u)
    # same-spin exchange only: <a s, b s|V|b s, a s> = J reduces the same-spin density term
    for s in range(2):
        for a in range(n):
            for b in range(n):
                if a != b:
                    u4[s * n + a, s * n + b, s * n + b, s * n + a] = J
    return u4


# ---------------------------------------------------------------------------- term lists


def test_symmetry_completion_is_idempotent():
    """Writing an element and its images gives the same tensor as writing it once."""
    once = terms_to_tensor(3, [[0, 1, 2, 1, 0.4, 0.1]])
    with_images = terms_to_tensor(3, [[0, 1, 2, 1, 0.4, 0.1], [1, 0, 1, 2, 0.4, 0.1], [2, 1, 0, 1, 0.4, -0.1]])
    np.testing.assert_array_equal(once, with_images)
    validate_u4(spatial_to_spin_u4(once), 6)


def test_contradicting_image_is_refused():
    with pytest.raises(ValueError, match="already set"):
        terms_to_tensor(2, [[0, 1, 1, 0, 0.5], [1, 0, 0, 1, 0.6]])


def test_literal_terms_skip_completion():
    u = terms_to_tensor(2, [[0, 1, 1, 0, 0.5]], complete_symmetries=False)
    assert np.count_nonzero(u) == 1


def test_spin_orbital_terms_add_to_the_spatial_expansion():
    """A spin-orbital density term on top of a spatial U: U n_up n_dn + V n_dn(orb 0) n_dn(orb 1)."""
    U, V = 2.0, 0.3
    u4 = terms_u4(2, spatial=[[0, 0, 0, 0, U], [1, 1, 1, 1, U]], spin_orbital=[[0, 1, 0, 1, V]])
    n = 2
    reference = density_density_u4(U * np.eye(n))
    reference[0, 1, 0, 1] = reference[1, 0, 1, 0] = V
    assert_same_operator(u4, reference)


@pytest.mark.parametrize("bad", [[0, 0, 0, 5, 1.0], [0, 0, 0, 1], [0, 0.5, 0, 0, 1.0]])
def test_malformed_terms_are_refused(bad):
    with pytest.raises(ValueError):
        terms_to_tensor(2, [bad])


# ---------------------------------------------------------------------------- validation


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_existing_slater_tensors_pass_validation(l):
    rng = np.random.default_rng(l)
    validate_u4(atomic_u4(l, rng.uniform(0.5, 5.0, size=2 * l + 1)), 2 * (2 * l + 1))


def test_non_hermitian_tensor_is_refused():
    u4 = kanamori_u4(2, 3.0, J=0.5).copy()
    u4[0, 2, 1, 3] += 0.1j
    with pytest.raises(ValueError, match="not Hermitian"):
        validate_u4(u4, 4)


def test_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        validate_u4(kanamori_u4(2, 3.0), 6)


# ---------------------------------------------------------------------------- averages


def test_uj_from_u4_is_independent_of_the_orbital_basis():
    """U and U-J are pair-space traces, so any spin-independent unitary within the shell leaves
    them unchanged -- which is why a model shell needs no rot_to_spherical for its DC."""
    rng = np.random.default_rng(3)
    u4 = atomic_u4(2, [6.0, 0.0, 8.0, 0.0, 5.0])
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
    rotated = rotate_two_body(u4, np.kron(np.eye(2), q))
    np.testing.assert_allclose(uj_from_u4(rotated), uj_from_u4(u4), atol=1e-12)


def test_uj_of_kanamori_is_the_orbital_average():
    U, Up, J, n = 4.0, 2.3, 0.7, 3
    u_bar, j_bar = uj_from_u4(kanamori_u4(n, U, J=J, U_prime=Up, J_pair=0.2))
    np.testing.assert_allclose(u_bar, (U + (n - 1) * Up) / n)
    # U - J is the mean same-spin interaction over distinct orbitals, U' - J
    np.testing.assert_allclose(u_bar - j_bar, Up - J)


@pytest.mark.parametrize("l", [1, 2, 3])
def test_mlft_uvv_matches_dc_mlft(l):
    rng = np.random.default_rng(10 + l)
    fvv = rng.uniform(0.5, 8.0, size=2 * l + 1)
    fvv[1::2] = 0.0
    dc = atomic_physics.dc_MLFT(l, 1, 0.0, fvv)[l]
    np.testing.assert_allclose(mlft_uvv(atomic_u4(l, fvv)), dc, rtol=1e-12)


def test_mlft_uvv_d_shell_closed_form():
    f0, f2, f4 = 7.5, 9.9, 6.6
    np.testing.assert_allclose(mlft_uvv(atomic_u4(2, [f0, 0, f2, 0, f4])), f0 - 2 / 63 * (f2 + f4))


# ---------------------------------------------------------------------------- basis


@pytest.mark.parametrize("l", [1, 2])
def test_cubic_to_spherical_inverts_the_spherical_to_cubic_rotation(l):
    """The Slater tensor is real in the (real) cubic-harmonic basis -- a check that the forward
    rotation really lands on the real orbitals -- and cubic_to_spherical_u4 brings it back."""
    fvv = [6.0, 0, 8.0, 0, 5.0][: 2 * l + 1]
    u4_sph = atomic_u4(l, fvv)
    u = get_spherical_2_cubic_matrix(spinpol=False, l=l)
    u4_cubic = rotate_two_body(u4_sph, np.kron(np.eye(2), u))
    assert np.max(np.abs(u4_cubic.imag)) < 1e-12
    np.testing.assert_allclose(cubic_to_spherical_u4(u4_cubic, l), u4_sph, atol=1e-12)


# ---------------------------------------------------------------------------- solver convention


def test_manybody_operator_agrees_with_the_oracle():
    """The oracle above is only worth anything if the solver's own operator is the same thing:
    the spectrum of ManyBodyOperator(getUop_from_rspt_u4(u4)) on the full Fock space must equal
    fock_matrix(u4)'s. (Spectra, so the two codes' bit orderings need not agree.)"""
    u4 = kanamori_u4(2, 4.0, J=0.7, U_prime=2.3, J_pair=0.45)
    n = u4.shape[0]
    op = ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(u4))
    states = [
        ManyBodyState({SlaterDeterminant.from_bytes((s << (8 - n)).to_bytes(8, byteorder="little")): 1.0})
        for s in range(2**n)
    ]
    keys = [next(iter(b.keys())) for b in states]
    applied = op.apply_multi(states)
    h = np.zeros((2**n, 2**n), dtype=complex)
    for j in range(2**n):
        for i, key in enumerate(keys):
            amp = applied[j].get(key)
            h[i, j] = 0.0 if amp is None else amp[0]
    np.testing.assert_allclose(np.linalg.eigvalsh(h), np.linalg.eigvalsh(fock_matrix(u4)), atol=1e-12)


def test_kanamori_every_parameter_is_visible():
    """Guard against a parameter being silently ignored: changing any one changes the operator."""
    base = dict(U=4.0, J=0.7, U_prime=2.3, J_pair=0.45)
    ref = fock_matrix(kanamori_u4(2, **base))
    for key, value in base.items():
        changed = dict(base, **{key: value + 0.1})
        assert np.max(np.abs(fock_matrix(kanamori_u4(2, **changed)) - ref)) > 1e-3, key


def test_all_permutations_of_symmetric_elements_are_filled():
    u = terms_to_tensor(2, [[0, 1, 1, 0, 0.5]])
    for key in itertools.product(range(2), repeat=4):
        expected = 0.5 if key in {(0, 1, 1, 0), (1, 0, 0, 1)} else 0.0
        assert u[key] == expected
