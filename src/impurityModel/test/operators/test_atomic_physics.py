import numpy as np
import pytest

from impurityModel.ed.atomic_physics import (
    dc_MLFT,
    gauntC,
    get2p3dSlaterCondonUop,
    get_spherical_2_cubic_matrix,
    n_octahedral_splittings,
    octahedral_level_structure,
    getUop,
    getUop_from_rspt_u4,
    slater_condon_Uop,
    uj_from_u4,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, inner
from impurityModel.ed.model import atomic_u4


def test_dc_MLFT():
    # Only 3d
    res = dc_MLFT(2, 5, 1.0, [5.0, 0, 1.0, 0, 1.0])
    assert set(res) == {2}
    assert np.isclose(res[2], (5.0 - 14.0 / 441 * 2.0) * 5 - 1.0)

    # with 2p
    res = dc_MLFT(2, 5, 1.0, [5.0, 0, 1.0, 0, 1.0], lc=1, n_core_i=6, Fcv=[4.0, 0, 0], Gcv=[0, 1.0, 0, 1.0])
    assert set(res) == {1, 2}


def test_dc_MLFT_reproduces_the_hardcoded_2p3d_coefficients():
    """The shell-agnostic spherical averages must equal the historical Udd / Upd literals."""
    Fdd, Fpd, Gpd = [5.0, 0, 1.0, 0, 1.0], [4.0, 0, 0], [0, 1.0, 0, 1.0]
    Udd = Fdd[0] - 14.0 / 441 * (Fdd[2] + Fdd[4])
    Upd = Fpd[0] - (1 / 15.0) * Gpd[1] - (3 / 70.0) * Gpd[3]

    res = dc_MLFT(2, 5, 1.0, Fdd, lc=1, n_core_i=6, Fcv=Fpd, Gcv=Gpd)
    assert np.isclose(res[2], Udd * 5 + Upd * 6 - 1.0)
    assert np.isclose(res[1], Upd * (5 + 1) - 1.0)


def test_dc_MLFT_generalises_to_another_edge():
    """A K-edge (1s core, 2p valence) pair is assembled from its own 3j coefficients."""
    Fpp, Fsp, Gsp = [3.0, 0, 1.0], [2.0], [0.0, 0.5]
    res = dc_MLFT(1, 3, 0.5, Fpp, lc=0, n_core_i=2, Fcv=Fsp, Gcv=Gsp)
    assert set(res) == {0, 1}
    # intra(1) = (2/25,) on k=2; inter(1, 0) = (1/6,) on k=1.
    Upp = Fpp[0] - (2 / 25) * Fpp[2]
    Usp = Fsp[0] - (1 / 6) * Gsp[1]
    assert np.isclose(res[1], Upp * 3 + Usp * 2 - 0.5)
    assert np.isclose(res[0], Usp * (3 + 1) - 0.5)


def test_dc_MLFT_rejects_a_partially_filled_core():
    with pytest.raises(ValueError, match="filled core shell"):
        dc_MLFT(2, 5, 1.0, [5.0, 0, 1.0, 0, 1.0], lc=1, n_core_i=5, Fcv=[4.0, 0, 0], Gcv=[0, 1.0, 0, 1.0])


def test_dc_MLFT_rejects_an_incomplete_core_specification():
    with pytest.raises(ValueError, match="needs all of"):
        dc_MLFT(2, 5, 1.0, [5.0, 0, 1.0, 0, 1.0], lc=1, n_core_i=6)


def test_slater_condon_Uop_matches_the_2p3d_assembler():
    """Gate 1: the general assembler must reproduce the L2,3 operator bit-identically."""
    Fdd, Fpp, Fpd, Gpd = (7.5, 0, 9.9, 0, 6.6), (0.0, 0.0, 0.0), (8.9, 0, 6.8), (0, 5.0, 0, 2.8)
    reference = get2p3dSlaterCondonUop(Fdd=Fdd, Fpp=Fpp, Fpd=Fpd, Gpd=Gpd)
    general = slater_condon_Uop(2, 1, Fdd, Fcc=Fpp, Fcv=Fpd, Gcv=Gpd)
    assert set(general) == set(reference)
    assert max(abs(general[k] - reference[k]) for k in reference) == 0.0


def test_slater_condon_Uop_without_a_core_shell():
    """The documented Fcc=None case must build the valence-only operator, not raise."""
    Fdd = (7.5, 0, 9.9, 0, 6.6)
    assert slater_condon_Uop(2, None, Fdd) == getUop(l1=2, l2=2, l3=2, l4=2, R=Fdd)


def test_slater_condon_Uop_cross_checks_array_lengths():
    with pytest.raises(ValueError, match="l_core=1 requires 3|requires 3"):
        slater_condon_Uop(2, 1, (7.5, 0, 9.9, 0, 6.6), Fcc=(0.0,) * 5)
    with pytest.raises(ValueError, match="no core shell"):
        slater_condon_Uop(2, None, (7.5, 0, 9.9, 0, 6.6), Fcc=(0.0,) * 3)


def test_slater_condon_Uop_builds_a_k_edge():
    """l_core = 0 is exactly the case a length-derived l_core would get wrong."""
    U = slater_condon_Uop(1, 0, (3.0, 0.0, 1.0), Fcc=(2.0,), Fcv=(1.5,), Gcv=(0.0, 0.5))
    assert U
    # Every label must name one of the two declared shells.
    assert {label[0] for process in U for label, _ in process} == {0, 1}


def test_get_spherical_2_cubic_matrix():
    u_p = get_spherical_2_cubic_matrix(l=1)
    assert u_p.shape == (3, 3)

    u_d = get_spherical_2_cubic_matrix(l=2)
    assert u_d.shape == (5, 5)


def _det(occupied, n_orbs):
    """Single-determinant ManyBodyState with the given occupied orbitals (MSB-first bits)."""
    data = bytearray((n_orbs + 7) // 8)
    for orb in occupied:
        data[orb // 8] |= 1 << (7 - orb % 8)
    return ManyBodyState({SlaterDeterminant.from_bytes(bytes(data)): 1.0})


def _random_rspt_u4(n, seed=0):
    """Random tensor with RSPt's symmetries: u(i,j,k,l)=u(j,i,l,k) and u(i,j,k,l)=conj(u(k,l,i,j))."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n, n, n, n)) + 1j * rng.standard_normal((n, n, n, n))
    r = r + r.transpose((1, 0, 3, 2))  # two-electron exchange symmetry
    return r + np.conj(r.transpose((2, 3, 0, 1)))  # hermiticity


def test_getUop_from_rspt_u4_density_density():
    """u4[i,j,i,j] = <ij|V|ij> is the direct (Hartree) element: <D|U|D> = U n_i n_j."""
    n = 2
    u4 = np.zeros((n, n, n, n), dtype=complex)
    u4[0, 1, 0, 1] = 1.7
    u4[1, 0, 1, 0] = 1.7  # exchange-symmetric partner
    u_op = ManyBodyOperator(getUop_from_rspt_u4(u4))

    both = _det([0, 1], n)
    assert np.isclose(inner(both, u_op(both, 0)), 1.7)
    single = _det([0], n)
    assert np.isclose(inner(single, u_op(single, 0)), 0.0)


def test_getUop_from_rspt_u4_matches_old_convention():
    """The RSPt-order operator equals the old (moveaxis(u4, 1, 0)) operator.

    RSPt's u4[i,j,k,l] multiplies c^dag_i c^dag_j c_l c_k; the previous code
    read the tensor as c^dag_i c^dag_j c_k c_l and relied on the wrapper
    pre-swapping the first two indices. By the exchange symmetry
    u(i,j,k,l) = u(j,i,l,k) both prescriptions define the same operator.
    """
    from itertools import combinations

    n = 4
    u4 = _random_rspt_u4(n)
    new_op = ManyBodyOperator(getUop_from_rspt_u4(u4))

    u4_old = np.moveaxis(u4, 1, 0)
    old_dict = {}
    for i, j, k, l in np.ndindex(u4_old.shape):
        if abs(u4_old[i, j, k, l]) > 1e-10:
            old_dict[((i, "c"), (j, "c"), (k, "a"), (l, "a"))] = u4_old[i, j, k, l] / 2
    old_op = ManyBodyOperator(old_dict)

    dets = [_det(occ, n) for n_el in range(n + 1) for occ in combinations(range(n), n_el)]
    for bra in dets:
        for ket in dets:
            assert np.isclose(inner(bra, new_op(ket, 0)), inner(bra, old_op(ket, 0)), atol=1e-12)


def test_uj_from_u4_d_shell_matches_slater_condon_average():
    F0, F2, F4 = 7.5, 9.9, 6.6
    u4 = atomic_u4(2, [F0, 0, F2, 0, F4])
    U, J = uj_from_u4(u4)
    assert np.isclose(U, F0, atol=1e-10)
    assert np.isclose(J, (F2 + F4) / 14, atol=1e-10)


def test_uj_from_u4_p_shell_matches_slater_condon_average():
    F0, F2 = 8.0, 5.0
    u4 = atomic_u4(1, [F0, 0, F2])
    U, J = uj_from_u4(u4)
    assert np.isclose(U, F0, atol=1e-10)
    assert np.isclose(J, F2 / 5, atol=1e-10)


def test_uj_from_u4_s_shell_has_no_exchange():
    F0 = 6.3
    u4 = atomic_u4(0, [F0])
    U, J = uj_from_u4(u4)
    assert np.isclose(U, F0, atol=1e-10)
    assert J == 0.0


def test_uj_from_u4_rejects_noncollinear_basis():
    """A basis with nonzero cross-spin exchange (e.g. spin-orbit-coupled) must be rejected."""
    n = 4
    u4 = _random_rspt_u4(n)
    with pytest.raises(ValueError, match="cross-spin exchange"):
        uj_from_u4(u4)


def test_uj_from_u4_requires_even_spin_orbital_dimension():
    u4 = np.zeros((3, 3, 3, 3), dtype=complex)
    with pytest.raises(ValueError, match="even"):
        uj_from_u4(u4)


# --- Octahedral level structure ------------------------------------------------------------
#
# The table in `atomic_physics.OCTAHEDRAL_LEVELS` is a claim about symmetry, so it is checked
# against the thing it claims to summarise rather than against itself: the crystal field of
# six point charges on the +-x, +-y, +-z axes, expanded in `gauntC`. Nothing below reads the
# table's numbers to build its reference.

_OCTAHEDRAL_DIRECTIONS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _octahedral_cf_operator(l, k):
    """Rank-``k`` octahedral crystal field of six axial point charges, in the |l,m> basis."""
    from scipy.special import sph_harm_y

    def ylm(k, q, v):
        v = np.asarray(v, dtype=float)
        return sph_harm_y(k, q, np.arccos(v[2] / np.linalg.norm(v)), np.arctan2(v[1], v[0]))

    ms = range(-l, l + 1)
    return np.array(
        [
            [
                (
                    sum(np.conj(ylm(k, m - mp, d)) for d in _OCTAHEDRAL_DIRECTIONS) * gauntC(k, l, m, l, mp, prec=16)
                    if abs(m - mp) <= k
                    else 0
                )
                for mp in ms
            ]
            for m in ms
        ],
        dtype=complex,
    )


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_the_cubic_harmonics_are_unitary(l):
    u = get_spherical_2_cubic_matrix(l=l)
    assert u.shape == (2 * l + 1, 2 * l + 1)
    assert np.allclose(u.conj().T @ u, np.eye(2 * l + 1), atol=1e-14)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_the_cubic_harmonics_diagonalise_every_octahedral_invariant(l):
    """The defining property, and the one an unmixed real-harmonic f basis fails.

    Both the rank-4 and the rank-6 invariant must come out diagonal in the *same* basis. For
    l=3 they are independent operators, so this is two conditions on one matrix.
    """
    u = get_spherical_2_cubic_matrix(l=l)
    for k in range(2, 2 * l + 1, 2):
        d = u.conj().T @ _octahedral_cf_operator(l, k) @ u
        assert np.max(np.abs(d - np.diag(np.diag(d)))) < 1e-12, f"k={k} is not diagonal"


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_the_octahedral_level_table_matches_the_point_charge_field(l):
    """Degeneracies, column order and weights, all rederived.

    The weights are normalised so the highest and lowest level of one invariant differ by 1,
    which is what makes the d-shell entry the historical ``(3/5, -2/5)`` of ``10Dq``.
    """
    levels = octahedral_level_structure(l)
    u = get_spherical_2_cubic_matrix(l=l)

    assert sum(degeneracy for _, degeneracy, _ in levels) == 2 * l + 1

    invariants = [k for k in range(2, 2 * l + 1, 2) if np.max(np.abs(_octahedral_cf_operator(l, k))) > 1e-10]
    assert len(invariants) == len(levels) - 1, "one splitting parameter per level boundary"
    assert all(len(weights) == len(invariants) for _, _, weights in levels)

    for column, k in enumerate(invariants):
        diagonal = np.real(np.diag(u.conj().T @ _octahedral_cf_operator(l, k) @ u))
        expected = diagonal / (diagonal.max() - diagonal.min())
        offset = 0
        for irrep, degeneracy, weights in levels:
            block = expected[offset : offset + degeneracy]
            assert np.allclose(block, block[0], atol=1e-12), f"{irrep} is not one level"
            assert weights[column] == pytest.approx(block[0], abs=1e-12), f"{irrep}, k={k}"
            offset += degeneracy
        assert sum(degeneracy * weights[column] for _, degeneracy, weights in levels) == pytest.approx(0, abs=1e-12)


def test_the_d_shell_weights_are_the_historical_10Dq_convention():
    """`get_CF_hamiltonian` used to write these two numbers out by hand."""
    assert octahedral_level_structure(2) == (("eg", 2, (3 / 5,)), ("t2g", 3, (-2 / 5,)))


def test_an_f_shell_needs_two_splitting_parameters():
    """The reason `e_deltaO_imp` alone cannot describe an f shell, under any naming."""
    assert n_octahedral_splittings(0) == 0
    assert n_octahedral_splittings(1) == 0
    assert n_octahedral_splittings(2) == 1
    assert n_octahedral_splittings(3) == 2


@pytest.mark.parametrize("l", [4, 5, 7])
def test_an_untabulated_shell_says_why_rather_than_failing_on_an_unbound_name(l):
    """It used to fall off the end of the if/elif chain and raise UnboundLocalError."""
    with pytest.raises(ValueError, match="l=3"):
        get_spherical_2_cubic_matrix(l=l)
    with pytest.raises(ValueError, match="irrep appears more than once|Implemented"):
        octahedral_level_structure(l)


# --- The level table against the literature, by a second route --------------------------------
#
# The tests above rederive the table from a point-charge lattice sum. These check it against the
# Stevens operator equivalents instead -- the form crystal-field parameters are quoted in --
# so a shared mistake in the lattice sum cannot hide.


def _stevens_cubic_operators(L):
    """``O_4^0 + 5 O_4^4`` and ``O_6^0 - 21 O_6^4`` in the |L, m> basis."""
    ms = np.arange(-L, L + 1)
    n = len(ms)
    identity = np.eye(n)
    Lz = np.diag(ms.astype(float))
    Lp = np.zeros((n, n))
    for i, m in enumerate(ms[:-1]):
        Lp[i + 1, i] = np.sqrt(L * (L + 1) - m * (m + 1))
    Lm = Lp.T
    X = L * (L + 1)
    Z2 = Lz @ Lz
    Z4 = np.linalg.matrix_power(Lz, 4)
    Z6 = np.linalg.matrix_power(Lz, 6)
    quartic = np.linalg.matrix_power(Lp, 4) + np.linalg.matrix_power(Lm, 4)

    o40 = 35 * Z4 - 30 * X * Z2 + 25 * Z2 - 6 * X * identity + 3 * X**2 * identity
    o60 = (
        231 * Z6
        - 315 * X * Z4
        + 735 * Z4
        + 105 * X**2 * Z2
        - 525 * X * Z2
        + 294 * Z2
        - 5 * X**3 * identity
        + 40 * X**2 * identity
        - 60 * X * identity
    )
    o64_prefactor = 11 * Z2 - (X + 38) * identity
    return o40 + 5 * (0.5 * quartic), o60 - 21 * (0.25 * (o64_prefactor @ quartic + quartic @ o64_prefactor))


@pytest.mark.parametrize("l", [2, 3])
def test_the_cubic_harmonics_diagonalise_the_stevens_operators(l):
    """The same basis, checked against the operators the literature quotes."""
    u = get_spherical_2_cubic_matrix(l=l)
    for k, operator in zip((4, 6), _stevens_cubic_operators(l)):
        if k > 2 * l:
            assert np.max(np.abs(operator)) < 1e-10, f"O_{k} must vanish on an l={l} shell"
            continue
        rotated = u.conj().T @ operator.astype(complex) @ u
        assert np.max(np.abs(rotated - np.diag(np.diag(rotated)))) < 1e-10, f"k={k}"


def test_the_splitting_parameters_sign_convention_against_stevens():
    """``e_deltaO_imp`` follows Stevens' sign; ``e_delta6_imp`` is its OPPOSITE.

    Both parameters carry the sign a point-charge octahedron produces, so a real octahedral
    field has *both* positive -- which is why the convention was chosen. The standard Stevens
    parametrisation ``B4(O_4^0 + 5 O_4^4) + B6(O_6^0 - 21 O_6^4)`` does not: an octahedron
    gives ``B4 > 0`` and ``B6 < 0``. So the two agree at rank 4 and disagree at rank 6, and a
    user importing ``B6`` from a paper must flip its sign.

    That asymmetry is a trap, which is exactly why it is pinned here rather than left to be
    rediscovered from a spectrum that comes out mirrored.
    """
    l = 3
    u = get_spherical_2_cubic_matrix(l=l)
    levels = octahedral_level_structure(l)
    for index, (k, operator) in enumerate(zip((4, 6), _stevens_cubic_operators(l))):
        diagonal = np.real(np.diag(u.conj().T @ operator.astype(complex) @ u))
        stevens_weights = diagonal / (diagonal.max() - diagonal.min())
        ours = np.array([w[index] for _, degeneracy, w in levels for _ in range(degeneracy)])
        expected_sign = +1 if k == 4 else -1
        assert np.allclose(
            ours, expected_sign * stevens_weights, atol=1e-12
        ), f"k={k}: ours {np.round(ours, 6)} vs Stevens {np.round(stevens_weights, 6)}"
