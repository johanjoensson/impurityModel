"""Tests for symmetry-related ground-state observables (symmetry plan, Phase 1)."""

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, inner
from impurityModel.ed.observables import (
    casimir_operator,
    casimir_to_quantum_number,
    expect_casimir,
    get_LS_from_rho_spherical,
    make_spin_operators,
    manifold_observable_values,
)
from impurityModel.ed.operator_algebra import c2i


def _sd(occupied, n_orbs=4):
    """SlaterDeterminant with the given occupied orbitals (MSB-first within byte)."""
    n_bytes = (n_orbs + 7) // 8
    data = bytearray(n_bytes)
    for orb in occupied:
        data[orb // 8] |= 1 << (7 - (orb % 8))
    return SlaterDeterminant.from_bytes(bytes(data))


def _state(terms, n_orbs=4):
    """Normalised ManyBodyState from [(occupied_orbs, amplitude), ...]."""
    psi = ManyBodyState({_sd(occ, n_orbs): amp for occ, amp in terms})
    return psi / psi.norm()


def _index(l, ml, spin):
    """Index of orbital (ml, spin) in the spherical [down, up] x [ml] layout.

    spin = -1 -> spin-down block, spin = +1 -> spin-up block.
    """
    n = 2 * l + 1
    a = ml + l
    return a if spin < 0 else a + n


def test_spin_orbit_observable():
    """<L.S> = Tr(rho . (l.s)) matches analytic single-particle values."""
    l = 2  # d shell
    n = 2 * l + 1

    # 1) Full shell: l.s is traceless, so <L.S> = 0.
    rho_full = np.eye(2 * n)
    assert abs(get_LS_from_rho_spherical(rho_full, l=l)) < 1e-12

    # 2) Single electron in a number eigenstate |ml, ms>: only l_z s_z survives,
    #    so <L.S> = ml * ms.
    for ml, spin in [(2, +1), (2, -1), (-1, +1), (0, +1)]:
        rho = np.zeros((2 * n, 2 * n), dtype=complex)
        idx = _index(l, ml, spin)
        rho[idx, idx] = 1.0
        expected = ml * (0.5 * spin)
        assert np.isclose(get_LS_from_rho_spherical(rho, l=l), expected, atol=1e-12)

    # 3) Coherent single-particle superposition exercising the ladder term:
    #    phi = (|ml=1, up> + |ml=2, down>)/sqrt(2).  By hand <phi|L.S|phi> = 3/4.
    v = np.zeros(2 * n, dtype=complex)
    v[_index(l, 1, +1)] = 1.0 / np.sqrt(2)
    v[_index(l, 2, -1)] = 1.0 / np.sqrt(2)
    rho = np.outer(v, v.conj())
    assert np.isclose(get_LS_from_rho_spherical(rho, l=l), 0.75, atol=1e-12)


def test_spin_orbit_observable_l_inferred():
    """l defaults from rho shape; full p shell (l=1) gives <L.S> = 0."""
    l = 1
    n = 2 * l + 1
    rho_full = np.eye(2 * n)
    assert abs(get_LS_from_rho_spherical(rho_full)) < 1e-12


# Two spatial orbitals A, B; layout (dn, up): A=(0,1), B=(2,3).
SPIN_PAIRS = [(0, 1), (2, 3)]


def _S2(psi):
    s_plus, s_minus, s_z = make_spin_operators(SPIN_PAIRS)
    return expect_casimir(psi, s_plus, s_minus, s_z)


def test_S2_observable():
    """<S^2> = S(S+1) for singlet, doublet, and triplet states."""
    # Vacuum and fully-filled shell: S = 0.
    assert np.isclose(_S2(_state([([], 1.0)])), 0.0, atol=1e-12)
    assert np.isclose(_S2(_state([([0, 1, 2, 3], 1.0)])), 0.0, atol=1e-12)

    # One electron -> doublet, S = 1/2, S(S+1) = 3/4.
    for occ in ([1], [0], [3], [2]):
        s2 = _S2(_state([(occ, 1.0)]))
        assert np.isclose(s2, 0.75, atol=1e-12)
        assert np.isclose(casimir_to_quantum_number(s2), 0.5, atol=1e-9)

    # Two electrons, one per orbital.
    # Triplet S_z = +1 (both up) and S_z = -1 (both down): S(S+1) = 2.
    assert np.isclose(_S2(_state([([1, 3], 1.0)])), 2.0, atol=1e-12)
    assert np.isclose(_S2(_state([([0, 2], 1.0)])), 2.0, atol=1e-12)

    # The two S_z = 0 combinations of |up_A dn_B> and |dn_A up_B>:
    # one is the triplet (S^2 = 2), the other the singlet (S^2 = 0).
    plus = _S2(_state([([1, 2], 1.0), ([0, 3], 1.0)]))
    minus = _S2(_state([([1, 2], 1.0), ([0, 3], -1.0)]))
    assert np.isclose(min(plus, minus), 0.0, atol=1e-12)  # singlet
    assert np.isclose(max(plus, minus), 2.0, atol=1e-12)  # triplet
    assert np.isclose(casimir_to_quantum_number(max(plus, minus)), 1.0, atol=1e-9)


def test_L2_J2_observable():
    """<L^2>, <S^2>, <J^2> on a stretched single d-electron state |ml=2, up>.

    Layout (n_orbs=10): spin-down block indices 0..4 (ml=-2..2),
    spin-up block indices 5..9 (ml=-2..2).
    """
    from impurityModel.ed.observables import make_orbital_angular_momentum_operators

    n_orbs = 10
    down = [0, 1, 2, 3, 4]  # ml = -2..2, spin down
    up = [5, 6, 7, 8, 9]  # ml = -2..2, spin up
    spin_pairs = list(zip(down, up))  # (dn, up) per spatial orbital

    # Single electron in |ml=+2, up> -> orbital index 9. This is the stretched
    # state |j=5/2, mj=5/2>, an exact eigenstate of L^2, S^2 and J^2.
    psi = _state([([9], 1.0)], n_orbs=n_orbs)

    s_plus, s_minus, s_z = make_spin_operators(spin_pairs)
    l_plus, l_minus, l_z = make_orbital_angular_momentum_operators([down, up])

    S2 = expect_casimir(psi, s_plus, s_minus, s_z)
    L2 = expect_casimir(psi, l_plus, l_minus, l_z)
    # J = L + S
    j_plus = l_plus + s_plus
    j_minus = l_minus + s_minus
    j_z = l_z + s_z
    J2 = expect_casimir(psi, j_plus, j_minus, j_z)

    assert np.isclose(S2, 0.75, atol=1e-12)  # S = 1/2
    assert np.isclose(L2, 6.0, atol=1e-12)  # L = 2 -> l(l+1) = 6
    assert np.isclose(J2, 35.0 / 4, atol=1e-12)  # j = 5/2 -> 35/4
    assert np.isclose(casimir_to_quantum_number(L2), 2.0, atol=1e-9)
    assert np.isclose(casimir_to_quantum_number(J2), 2.5, atol=1e-9)


def test_degenerate_manifold_observable():
    """S^2 on an accidentally-degenerate singlet+triplet manifold.

    The 2-electron / 2-orbital space {|1,3>,|0,2>,|1,2>,|0,3>} is a singlet (S=0)
    plus a triplet (S=1). Treated as one degenerate manifold, diagonalising S^2
    recovers eigenvalues {0, 2, 2, 2}, whereas the naive per-vector <psi|S^2|psi>
    on a singlet/triplet mixture is neither 0 nor 2.
    """
    s_plus, s_minus, s_z = make_spin_operators(SPIN_PAIRS)
    s2_op = casimir_operator(s_plus, s_minus, s_z)

    manifold = ManyBodyState.from_states(
        [
            _state([([1, 3], 1.0)]),  # triplet S_z=+1
            _state([([0, 2], 1.0)]),  # triplet S_z=-1
            _state([([1, 2], 1.0)]),  # mixes singlet/triplet S_z=0
            _state([([0, 3], 1.0)]),  # mixes singlet/triplet S_z=0
        ]
    )
    energies = np.zeros(4)  # accidentally degenerate

    vals = manifold_observable_values(manifold, energies, lambda blk: s2_op.apply_block(blk, 0))
    np.testing.assert_allclose(np.sort(vals), [0.0, 2.0, 2.0, 2.0], atol=1e-10)

    # Naive per-vector value on a singlet/triplet mixture is wrong (gives 1, not 0/2).
    singlet = _state([([1, 2], 1.0), ([0, 3], -1.0)])
    triplet0 = _state([([1, 2], 1.0), ([0, 3], 1.0)])
    mixed = singlet + triplet0
    mixed = mixed / mixed.norm()
    naive = expect_casimir(mixed, s_plus, s_minus, s_z)
    assert not np.isclose(naive, 0.0, atol=1e-3)
    assert not np.isclose(naive, 2.0, atol=1e-3)
    assert np.isclose(naive, 1.0, atol=1e-10)


def test_thermal_observable():
    """Thermal average matches a brute-force Boltzmann sum and reduces to T=0."""
    from impurityModel.ed.observables import thermal_observable_value

    values = np.array([0.0, 2.0, 2.0, 2.0])
    energies = np.array([0.0, 1.0, 1.0, 3.0])

    for tau in (0.1, 0.5, 2.0):
        weights = np.exp(-(energies - energies.min()) / tau)
        expected = np.sum(weights * values) / np.sum(weights)
        assert np.isclose(thermal_observable_value(values, energies, tau), expected, atol=1e-12)

    # T -> 0 selects the lowest-energy state's value.
    assert np.isclose(thermal_observable_value(values, energies, 1e-6), values[0], atol=1e-9)


def test_kondo_correlation():
    """<S_imp . S_bath> on a two-spin model: singlet screened, triplet positive."""
    from impurityModel.ed.observables import expect_spin_correlation

    # Orbital A = "impurity" (dn=0, up=1); orbital B = "bath" (dn=2, up=3).
    ops_imp = make_spin_operators([(0, 1)])
    ops_bath = make_spin_operators([(2, 3)])

    # Singlet of the two spin-1/2: S_A.S_B = 1/2[S^2 - S_A^2 - S_B^2] = -3/4.
    singlet = _state([([1, 2], 1.0), ([0, 3], -1.0)])
    assert np.isclose(expect_spin_correlation(singlet, ops_imp, ops_bath), -0.75, atol=1e-12)

    # Triplet (S_z=0): +1/4. Stretched triplet (both up): also +1/4.
    triplet0 = _state([([1, 2], 1.0), ([0, 3], 1.0)])
    assert np.isclose(expect_spin_correlation(triplet0, ops_imp, ops_bath), 0.25, atol=1e-12)
    triplet_up = _state([([1, 3], 1.0)])
    assert np.isclose(expect_spin_correlation(triplet_up, ops_imp, ops_bath), 0.25, atol=1e-12)

    # Empty impurity (no impurity electron): correlation vanishes.
    only_bath = _state([([2], 1.0)])
    assert np.isclose(expect_spin_correlation(only_bath, ops_imp, ops_bath), 0.0, atol=1e-12)


def test_spin_z_correlation():
    """<Sz_A Sz_B> (longitudinal part) matches analytic two-spin values."""
    from impurityModel.ed.observables import apply_spin_z_correlation, get_Sz_from_rho_pairs

    ops_imp = make_spin_operators([(0, 1)])
    ops_bath = make_spin_operators([(2, 3)])

    def expect_z(psi):
        return np.real(inner(psi, apply_spin_z_correlation(psi, ops_imp, ops_bath)))

    # Both singlet and Sz=0 triplet are built from |up,dn> components: <Sz_A Sz_B> = -1/4.
    singlet = _state([([1, 2], 1.0), ([0, 3], -1.0)])
    triplet0 = _state([([1, 2], 1.0), ([0, 3], 1.0)])
    assert np.isclose(expect_z(singlet), -0.25, atol=1e-12)
    assert np.isclose(expect_z(triplet0), -0.25, atol=1e-12)
    # Stretched triplet (both up): +1/4.
    triplet_up = _state([([1, 3], 1.0)])
    assert np.isclose(expect_z(triplet_up), 0.25, atol=1e-12)

    # get_Sz_from_rho_pairs: diagonal occupations [dn=0, up=1] per pair -> Sz = +1/2 each.
    rho = np.diag([0.0, 1.0, 0.0, 1.0]).astype(complex)
    assert np.isclose(get_Sz_from_rho_pairs(rho, [(0, 1), (2, 3)]), 1.0, atol=1e-12)
    assert np.isclose(get_Sz_from_rho_pairs(rho, [(2, 3)]), 0.5, atol=1e-12)


def test_spin_correlation_operator_matches_sequential_application_everywhere():
    """The product-built S_A.S_B agrees with sequential application on every determinant.

    ``spin_correlation_operator`` composes the one-body factors with ``*``; the oracle here
    applies them to the state one after another, which never forms the two-body product.
    Sweeping the entire 2^10 Fock space of a 2-pair model (rather than a handful of
    hand-picked states) is what makes a fermionic sign slip in the normal-ordering of the
    composed four-ladder string impossible to miss -- flipping one such sign moves the
    action by 1.0 here.

    It does *not* test the operand order of ``*``: A and B address disjoint orbitals and
    every factor is an even-length string, so the two factors genuinely commute and
    ``A*B == B*A``. Operand order is pinned by
    ``test_operator_algebra_cy.test_product_is_composition_not_its_reverse``.
    """
    import itertools

    from impurityModel.ed.observables import spin_correlation_operator

    ops_a = make_spin_operators([(0, 5)])
    ops_b = make_spin_operators([(1, 6)])
    a_plus, a_minus, a_z = ops_a
    b_plus, b_minus, b_z = ops_b

    def sequential(psi, z_only):
        result = a_z(b_z(psi, 0), 0)
        if not z_only:
            result += 0.5 * a_plus(b_minus(psi, 0), 0)
            result += 0.5 * a_minus(b_plus(psi, 0), 0)
        return result

    n_orb = 10
    worst = 0.0
    nonzero = 0
    for z_only in (False, True):
        product = spin_correlation_operator(ops_a, ops_b, z_only=z_only)
        for k in range(n_orb + 1):
            for occupied in itertools.combinations(range(n_orb), k):
                psi = ManyBodyState({_sd(occupied, n_orbs=n_orb): 1.0 + 0j})
                expected = sequential(psi, z_only)
                produced = product(psi, 0)
                nonzero += len(expected)
                for key in set(expected.keys()) | set(produced.keys()):
                    e = expected.get(key)
                    p = produced.get(key)
                    e = 0j if e is None else e[0]
                    p = 0j if p is None else p[0]
                    worst = max(worst, abs(e - p))
    assert nonzero > 0, "the sweep never produced a non-trivial action"
    assert worst == 0.0


def test_magnetic_dipole_tz():
    """T_z matrix: Hermitian, traceless, matches the closed form on pure |ml,ms> states."""
    from impurityModel.ed.observables import _single_particle_tz_matrix, get_Tz_from_rho_spherical

    for l in (1, 2, 3):
        tz = _single_particle_tz_matrix(l)
        n = 2 * l + 1
        assert np.allclose(tz, tz.conj().T)  # Hermitian
        assert abs(np.trace(tz)) < 1e-12  # traceless: full shell has <T_z> = 0
        denom = (2 * l - 1) * (2 * l + 3)
        for ml in range(-l, l + 1):
            for spin, ms in ((-1, -0.5), (+1, +0.5)):
                rho = np.zeros((2 * n, 2 * n), dtype=complex)
                idx = _index(l, ml, spin)
                rho[idx, idx] = 1.0
                # <T_z> = m_s [1 - 3<z^2>_{l ml}], <z^2> = (2l^2+2l-1-2ml^2)/((2l-1)(2l+3))
                expected = ms * (1 - 3 * (2 * l * l + 2 * l - 1 - 2 * ml * ml) / denom)
                assert np.isclose(get_Tz_from_rho_spherical(rho, l=l), expected, atol=1e-10)
    assert abs(get_Tz_from_rho_spherical(np.eye(10, dtype=complex))) < 1e-12


def test_term_symbol_and_lande():
    """Term symbols and Lande g / effective moments for textbook configurations."""
    from impurityModel.ed.observables import lande_g_and_moments, term_symbol

    assert term_symbol(1.0, 3.0, 4.0) == "3F4"  # d8 Hund ground term
    assert term_symbol(0.5, 2.0, 2.5) == "2D5/2"  # d9
    assert term_symbol(0.5, 3.0, 2.5) == "2F5/2"  # f1
    assert term_symbol(0.883, 2.757, 3.003).startswith("~")  # mixed valence -> approximate

    # f1 (2F5/2): g_J = 6/7, mu_eff = g_J sqrt(J(J+1)).
    g, mu, mu_s = lande_g_and_moments(0.75, 12.0, 35.0 / 4.0)
    assert np.isclose(g, 6.0 / 7.0, atol=1e-12)
    assert np.isclose(mu, 6.0 / 7.0 * np.sqrt(35.0 / 4.0), atol=1e-12)
    assert np.isclose(mu_s, 2.0 * np.sqrt(0.75), atol=1e-12)
    # J = 0: no Lande factor, spin-only moment still defined.
    g, mu, mu_s = lande_g_and_moments(2.0, 2.0, 0.0)
    assert g is None and mu is None
    assert np.isclose(mu_s, 2.0 * np.sqrt(2.0), atol=1e-12)


def test_moments_from_rho():
    """<Lz+2Sz> and <Jz> for a single |ml=2, up> electron: 3 and 5/2."""
    from impurityModel.ed.observables import get_moments_from_rho_spherical

    rho = np.zeros((10, 10), dtype=complex)
    rho[9, 9] = 1.0  # ml=+2, spin up
    m_z, j_z = get_moments_from_rho_spherical(rho)
    assert np.isclose(m_z, 3.0, atol=1e-12)
    assert np.isclose(j_z, 2.5, atol=1e-12)


def _d_shell_block_structure():
    from impurityModel.ed.block_structure import BlockStructure

    return BlockStructure(
        blocks=[list(range(10))],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )


def test_print_expectation_values_columns(capsys):
    """Existing per-eigenstate columns are preserved and <L.S> is appended."""
    from impurityModel.ed.observables import print_expectation_values

    n = 10  # d shell, l=2
    rot = np.eye(n)
    bs = _d_shell_block_structure()
    # Two eigenstates: a full shell (L.S=0) and a single |ml=2,up> (L.S = +1).
    rho_full = np.eye(n, dtype=complex)
    rho_one = np.zeros((n, n), dtype=complex)
    rho_one[9, 9] = 1.0  # ml=+2 (a=4), spin-up block -> index 9
    es = np.array([0.0, 1.0])

    print_expectation_values(np.array([rho_full, rho_one]), es, rot, bs)
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if "E-E0" in line)
    for col in ("i", "E-E0", "N", "N(Dn)", "N(Up)", "Lz", "Sz"):
        assert col in header
    assert "L.S" in header  # new column appended
    # Single-electron row reports <L.S> = ml*ms = 2 * 0.5 = 1.0 in the last field.
    rows = [ln for ln in out.splitlines() if ln.strip().startswith(("0", "1"))]
    last_field = float(rows[1].split()[-1])
    assert np.isclose(last_field, 1.0, atol=1e-6)


def test_print_thermal_expectation_values_lines(capsys):
    """Existing thermal lines are preserved and a <L.S> line is added."""
    from impurityModel.ed.observables import print_thermal_expectation_values

    n = 10
    rot = np.eye(n)
    bs = _d_shell_block_structure()
    rho_one = np.zeros((n, n), dtype=complex)
    rho_one[9, 9] = 1.0  # |ml=2, up> -> <L.S> = 1.0

    print_thermal_expectation_values(rho_one, 0.0, rot, bs)
    out = capsys.readouterr().out
    for label in ("<E>", "<N>", "<N(Dn)>", "<N(Up)>", "<Lz>", "<Sz>", "<Lz+2Sz>", "<Jz>", "<T_z>"):
        assert label in out
    ls_line = next(line for line in out.splitlines() if line.lstrip().startswith("<L.S>"))
    assert np.isclose(float(ls_line.split("=")[1].split()[0]), 1.0, atol=1e-6)
    # |ml=2, up>: Lz+2Sz = 3, Jz = 5/2; no Casimirs passed -> no term/g_J rows.
    mz_line = next(line for line in out.splitlines() if line.lstrip().startswith("<Lz+2Sz>"))
    assert np.isclose(float(mz_line.split("=")[1].split()[0]), 3.0, atol=1e-6)
    assert "term" not in out and "g_J" not in out

    # With Casimirs: term symbol, g_J, mu_eff rows appear (d9-like: 2D5/2, g_J = 6/5).
    print_thermal_expectation_values(rho_one, 0.0, rot, bs, s_thermal=0.75, l_thermal=6.0, j_thermal=35.0 / 4.0)
    out = capsys.readouterr().out
    term_line = next(line for line in out.splitlines() if line.lstrip().startswith("term"))
    assert "2D5/2" in term_line
    g_line = next(line for line in out.splitlines() if line.lstrip().startswith("g_J"))
    assert np.isclose(float(g_line.split("=")[1].split()[0]), 1.2, atol=1e-6)
    assert "mu_eff" in out and "mu_spin_only" in out


def test_print_expectation_values_S_column(capsys):
    """Passing s_values appends an 'S' column; omitting it preserves old output."""
    from impurityModel.ed.observables import print_expectation_values

    n = 10
    rot = np.eye(n)
    bs = _d_shell_block_structure()
    rho = np.eye(n, dtype=complex)
    es = np.array([0.0, 1.0])
    s_values = np.array([1.0, 0.5])

    print_expectation_values(np.array([rho, rho]), es, rot, bs, s_values=s_values)
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if "E-E0" in line)
    assert header.split()[-1] == "S"
    rows = [ln for ln in out.splitlines() if ln.strip().startswith(("0", "1"))]
    assert np.isclose(float(rows[0].split()[-1]), 1.0, atol=1e-6)
    assert np.isclose(float(rows[1].split()[-1]), 0.5, atol=1e-6)


def test_print_thermal_S2_line(capsys):
    """Passing s_thermal adds an <S^2> line with the matching S quantum number."""
    from impurityModel.ed.observables import print_thermal_expectation_values

    n = 10
    rot = np.eye(n)
    bs = _d_shell_block_structure()
    rho = np.eye(n, dtype=complex)

    print_thermal_expectation_values(rho, 0.0, rot, bs, s_thermal=2.0)
    out = capsys.readouterr().out
    s2_line = next(line for line in out.splitlines() if line.lstrip().startswith("<S^2>"))
    assert np.isclose(float(s2_line.split("=")[1].split()[0]), 2.0, atol=1e-6)
    assert "S =  1.0000" in s2_line  # S(S+1)=2 -> S=1


def test_impurity_casimir_operators_rotated():
    """make_impurity_casimir_operators gives correct, rotation-invariant <L^2>,<S^2>,<J^2>.

    Stretched single d-electron |ml=2, up> = |j=5/2, mj=5/2>: L^2=6, S^2=3/4, J^2=35/4.
    """
    from impurityModel.ed.observables import expect_casimir, make_impurity_casimir_operators

    imp = {0: [list(range(10))]}  # one d-shell, layout [down(ml=-2..2), up(ml=-2..2)]

    # Identity rotation: the computational basis is already spherical.
    L, S, J = make_impurity_casimir_operators(imp, np.eye(10, dtype=complex))
    psi = _state([([9], 1.0)], n_orbs=10)  # |ml=2, up>
    assert np.isclose(expect_casimir(psi, *L), 6.0, atol=1e-10)
    assert np.isclose(expect_casimir(psi, *S), 0.75, atol=1e-10)
    assert np.isclose(expect_casimir(psi, *J), 35.0 / 4, atol=1e-10)

    # A non-trivial (random) spherical->computational rotation R: the same physical
    # state has computational coordinates R[:, 9]; the Casimirs are unchanged.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((10, 10)) + 1j * rng.standard_normal((10, 10))
    rot, _ = np.linalg.qr(a)
    coords = rot[:, 9]
    psi_rot = _state([([a_], complex(coords[a_])) for a_ in range(10) if abs(coords[a_]) > 1e-12], n_orbs=10)
    Lr, Sr, Jr = make_impurity_casimir_operators(imp, rot)
    assert np.isclose(expect_casimir(psi_rot, *Lr), 6.0, atol=1e-9)
    assert np.isclose(expect_casimir(psi_rot, *Sr), 0.75, atol=1e-9)
    assert np.isclose(expect_casimir(psi_rot, *Jr), 35.0 / 4, atol=1e-9)


def test_print_expectation_values_LJ_columns(capsys):
    """Passing l_values / j_values appends 'L' and 'J' columns after 'S'."""
    from impurityModel.ed.observables import print_expectation_values

    n = 10
    bs = _d_shell_block_structure()
    rho = np.eye(n, dtype=complex)
    es = np.array([0.0])
    print_expectation_values(
        np.array([rho]),
        es,
        np.eye(n),
        bs,
        s_values=np.array([1.0]),
        l_values=np.array([2.0]),
        j_values=np.array([2.5]),
    )
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if "E-E0" in line)
    assert header.split()[-3:] == ["S", "L", "J"]
    row = next(ln for ln in out.splitlines() if ln.strip().startswith("0"))
    s, l, j = (float(x) for x in row.split()[-3:])
    assert (s, l, j) == (1.0, 2.0, 2.5)


def test_print_thermal_LJ_lines(capsys):
    """Passing l_thermal / j_thermal adds <L^2> and <J^2> lines with quantum numbers."""
    from impurityModel.ed.observables import print_thermal_expectation_values

    n = 10
    bs = _d_shell_block_structure()
    print_thermal_expectation_values(
        np.eye(n, dtype=complex),
        0.0,
        np.eye(n),
        bs,
        s_thermal=2.0,
        l_thermal=6.0,
        j_thermal=35.0 / 4,
    )
    out = capsys.readouterr().out
    assert "S = " in out and "<S^2>" in out
    l_line = next(line for line in out.splitlines() if line.lstrip().startswith("<L^2>"))
    assert "L =  2.0000" in l_line  # L(L+1)=6 -> L=2
    j_line = next(line for line in out.splitlines() if line.lstrip().startswith("<J^2>"))
    assert "J =  2.5000" in j_line  # J(J+1)=35/4 -> J=5/2


def test_bath_spin_pairs_and_consistency():
    """bath_spin_pairs + spin_pairs_consistent_with_h validate/skip the spin assignment."""
    from impurityModel.ed.observables import expect_spin_correlation, make_spin_operators
    from impurityModel.ed.spin_pairs import (
        bath_spin_pairs,
        impurity_spin_pairs,
        spin_pairs_consistent_with_h,
    )

    # 4-orbital Anderson: imp 0=dn,1=up ; bath 2=dn,3=up. Spin-diagonal hopping.
    imp_orbitals = {0: [[0, 1]]}
    bath = ({0: [[2, 3]]}, {0: [[]]})
    assert impurity_spin_pairs(imp_orbitals) == [(0, 1)]
    assert bath_spin_pairs(bath) == [(2, 3)]
    pairs = impurity_spin_pairs(imp_orbitals) + bath_spin_pairs(bath)

    terms = {((o, "c"), (o, "a")): -1.0 for o in (0, 1)}
    for a, b in ((0, 2), (1, 3)):  # spin-diagonal hybridization
        terms[((a, "c"), (b, "a"))] = 0.5
        terms[((b, "c"), (a, "a"))] = 0.5
    h_conserving = ManyBodyOperator(terms)
    assert spin_pairs_consistent_with_h(h_conserving, pairs, 4)

    # Add a spin-flip (SOC-like) term -> spin no longer conserved -> not consistent.
    soc = dict(terms)
    soc[((0, "c"), (3, "a"))] = 0.3
    soc[((3, "c"), (0, "a"))] = 0.3
    assert not spin_pairs_consistent_with_h(ManyBodyOperator(soc), pairs, 4)

    # When consistent, the Kondo correlation is well-defined: singlet -> -3/4.
    ops_imp = make_spin_operators(impurity_spin_pairs(imp_orbitals))
    ops_bath = make_spin_operators(bath_spin_pairs(bath))
    singlet = _state([([1, 2], 1.0), ([0, 3], -1.0)])  # imp-up bath-dn minus imp-dn bath-up
    assert np.isclose(expect_spin_correlation(singlet, ops_imp, ops_bath), -0.75, atol=1e-12)


def test_kondo_correlation_reported(capsys):
    """calc_gs on a SIAM with baths reports <S_imp.S_bath> (thermal line + per-state column)."""
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.groundstate import calc_gs

    ed, U, ev, ec, V = -2.0, 6.0, -4.0, 4.0, 1.0
    terms = {((o, "c"), (o, "a")): ed for o in (0, 1)}
    terms.update({((o, "c"), (o, "a")): ev for o in (2, 3)})
    terms.update({((o, "c"), (o, "a")): ec for o in (4, 5)})
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = U
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = V
        terms[((b, "c"), (a, "a"))] = V
    Hop = ManyBodyOperator(terms)

    bs = BlockStructure(
        blocks=[[0, 1]],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    basis_setup = dict(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        N0={0: 1},
        mixed_valence={0: 1},
        tau=0.01,
        dense_cutoff=1000,
        spin_flip_dj=False,
        comm=None,
        truncation_threshold=100000,
    )
    calc_gs(Hop, basis_setup, bs, np.eye(2, dtype=complex), verbose=True, slaterWeightMin=1e-12)
    out = capsys.readouterr().out

    # Thermal line and per-eigenstate column both present.
    assert any(line.lstrip().startswith("<S_imp.S_bath>") for line in out.splitlines())
    header = next(line for line in out.splitlines() if "E-E0" in line and "Sz" in line)
    assert "Si.Sb" in header
    # Full SU(2) case: no pairing flag, no longitudinal-only lines/column.
    sisb_line = next(line for line in out.splitlines() if line.lstrip().startswith("<S_imp.S_bath>"))
    assert "pairing" not in sisb_line
    assert not any(line.lstrip().startswith("<Sz_imp.Sz_bath>") for line in out.splitlines())
    assert "Szi.Szb" not in header
    # New report sections all present.
    for section in (
        "Ground-state report",
        "-- Thermal expectation values ",
        "-- Eigenstates ",
        "-- Correlation strength ",
        "-- Screening ",
        "-- Configurations & entanglement ",
        "-- Density matrices ",
        "Per-state summary",
        "Impurity correlation diagnostics",
        "Screening channels",
        "Impurity-bath entanglement",
        "one-body entanglement entropy",
        "<Lz+2Sz>",
        "<T_z>",
        "<H_Coulomb>",
        "mu_spin_only",
        "Static susceptibilities (Curie)",
        "chi_spin_zz",
        "chi_charge",
    ):
        assert section in out, f"missing report section: {section}"
    # The two independent Sz implementations (pairs-based chi_zz vs Casimir-op
    # chi_spin_zz) must agree on the same thermal manifold.
    chi_zz = float(next(ln for ln in out.splitlines() if "chi_zz = (<Sz^2>" in ln).split("/tau =")[1].split("(")[0])
    chi_spin = float(
        next(ln for ln in out.splitlines() if ln.lstrip().startswith("chi_spin_zz")).split("=")[1].split("(")[0]
    )
    # chi_zz prints with 4 decimals, so compare at that resolution.
    assert chi_spin == pytest.approx(chi_zz, abs=1e-4)


def test_kondo_correlation_reported_polarized_bath(capsys):
    """calc_gs on a SIAM with a spin-polarized bath reports the flagged full value plus the
    exact longitudinal correlation (raw + connected) and the Szi.Szb column.

    RSPt-style: spin-degenerate impurity, all polarization in the hybridization (spin-split
    bath energies and hoppings). Before the collinear check this case was skipped entirely.
    """
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.groundstate import calc_gs

    ed, U = -2.0, 6.0
    ev = {0: -4.0, 1: -3.6}  # valence bath energy per spin (0 = dn, 1 = up)
    ec = {0: 4.0, 1: 4.4}  # conduction bath energy per spin
    V = {0: 1.0, 1: 0.8}  # hybridization per spin
    terms = {((o, "c"), (o, "a")): ed for o in (0, 1)}
    terms.update({((o, "c"), (o, "a")): ev[s] for s, o in ((0, 2), (1, 3))})
    terms.update({((o, "c"), (o, "a")): ec[s] for s, o in ((0, 4), (1, 5))})
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = U
    for s, a, b in ((0, 0, 2), (1, 1, 3), (0, 0, 4), (1, 1, 5)):
        terms[((a, "c"), (b, "a"))] = V[s]
        terms[((b, "c"), (a, "a"))] = V[s]
    Hop = ManyBodyOperator(terms)

    bs = BlockStructure(
        blocks=[[0, 1]],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    basis_setup = dict(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        N0={0: 1},
        mixed_valence={0: 1},
        tau=0.01,
        dense_cutoff=1000,
        spin_flip_dj=False,
        comm=None,
        truncation_threshold=100000,
    )
    calc_gs(Hop, basis_setup, bs, np.eye(2, dtype=complex), verbose=True, slaterWeightMin=1e-12)
    out = capsys.readouterr().out

    # Full value present and flagged as pairing-dependent.
    sisb_line = next(line for line in out.splitlines() if line.lstrip().startswith("<S_imp.S_bath>"))
    assert "pairing" in sisb_line
    # Exact longitudinal lines present; singlet-like screening -> negative correlation.
    z_line = next(line for line in out.splitlines() if line.lstrip().startswith("<Sz_imp.Sz_bath>"))
    z_value = float(z_line.split("=")[1].split()[0])
    assert z_value < 0
    assert any(line.lstrip().startswith("cov(Sz_imp,Sz_bath)") for line in out.splitlines())
    # Per-eigenstate column for the longitudinal part.
    header = next(line for line in out.splitlines() if "E-E0" in line and "Sz" in line)
    assert "Szi.Szb" in header


def _hop(diag, hops):
    """Hermitian one-body ManyBodyOperator from on-site energies and hopping triples."""

    terms = {((o, "c"), (o, "a")): e for o, e in diag}
    for a, b, t in hops:
        terms[((a, "c"), (b, "a"))] = t
        terms[((b, "c"), (a, "a"))] = t
    return ManyBodyOperator(terms)


def test_derive_spin_pairs_chain():
    """derive_spin_pairs recovers the (dn,up) pairing for an interleaved-index chain."""
    from impurityModel.ed.spin_pairs import derive_spin_pairs, spin_pairs_consistent_with_h

    # 6 spin-orbitals, spins interleaved (NOT down-then-up): impurity (0 dn, 1 up);
    # spin-down chain 0-2-4, spin-up chain 1-3-5; orbs 2,3 valence (e<0), 4,5 conduction.
    Hop = _hop(
        [(0, 0.3), (1, 0.3), (2, -0.4), (3, -0.4), (4, 0.7), (5, 0.7)],
        [(0, 2, 0.5), (1, 3, 0.5), (2, 4, 0.2), (3, 5, 0.2)],
    )
    derived = derive_spin_pairs(Hop, {0: [[0, 1]]}, np.eye(2, dtype=complex), 6)
    assert derived is not None
    imp_pairs, bath_pairs = derived
    assert imp_pairs == [(0, 1)]
    assert sorted(bath_pairs) == [(2, 3), (4, 5)]
    assert spin_pairs_consistent_with_h(Hop, imp_pairs + bath_pairs, 6)
    # The naive down-then-up pairing of the same bath orbitals is inconsistent with h.
    assert not spin_pairs_consistent_with_h(Hop, [(0, 1), (2, 4), (3, 5)], 6)


def test_derive_spin_pairs_returns_none_when_unresolvable():
    """derive_spin_pairs gives up on a disconnected bath orbital or spin-mixing rotation."""
    from impurityModel.ed.spin_pairs import derive_spin_pairs

    # Bath orbitals 4,5 are isolated (no hopping) -> cannot be paired to the impurity.
    disconnected = _hop(
        [(0, 0.0), (1, 0.0), (2, -0.4), (3, -0.4), (4, 0.7), (5, 0.7)],
        [(0, 2, 0.5), (1, 3, 0.5)],
    )
    assert derive_spin_pairs(disconnected, {0: [[0, 1]]}, np.eye(2, dtype=complex), 6) is None

    # A rotation that mixes the two impurity spins makes the rotated S_+ non-permutation.
    theta = 0.3
    rot_mix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=complex)
    connected = _hop(
        [(0, 0.0), (1, 0.0), (2, -0.4), (3, -0.4), (4, 0.7), (5, 0.7)],
        [(0, 2, 0.5), (1, 3, 0.5), (2, 4, 0.2), (3, 5, 0.2)],
    )
    assert derive_spin_pairs(connected, {0: [[0, 1]]}, rot_mix, 6) is None


def test_derive_spin_pairs_block_grouped_bath():
    """Bath grouped by impurity block with unequal sizes (not (k,k+n/2)) still resolves.

    Mirrors the real layout: bath orbitals are ordered [all bath coupling to impurity
    block 0, then block 1, ...] and inequivalent blocks may have different bath counts.
    derive_spin_pairs follows H's hopping graph, so the ordering is irrelevant.
    """
    from impurityModel.ed.spin_pairs import derive_spin_pairs, spin_pairs_consistent_with_h

    # Two l=0 impurity shells: imp0 = (0 dn, 1 up), imp1 = (2 dn, 3 up).
    # Bath block 0 (couples to imp0) has 2 sites/spin: dn 4,5 / up 6,7.
    # Bath block 1 (couples to imp1) has 1 site/spin: dn 8 / up 9.  Different sizes.
    Hop = _hop(
        [(0, 0.2), (1, 0.2), (2, -0.1), (3, -0.1), (4, -0.3), (6, -0.3), (5, 0.6), (7, 0.6), (8, 0.4), (9, 0.4)],
        [(0, 4, 0.5), (1, 6, 0.5), (0, 5, 0.4), (1, 7, 0.4), (2, 8, 0.3), (3, 9, 0.3)],
    )
    derived = derive_spin_pairs(Hop, {0: [[0, 1]], 1: [[2, 3]]}, np.eye(2, dtype=complex), 10)
    assert derived is not None
    imp_pairs, bath_pairs = derived
    assert sorted(imp_pairs) == [(0, 1), (2, 3)]
    assert sorted(bath_pairs) == [(4, 6), (5, 7), (8, 9)]
    assert spin_pairs_consistent_with_h(Hop, imp_pairs + bath_pairs, 10)


def test_derive_spin_pairs_crystal_field_manifolds():
    """A single l-shell split into crystal-field manifolds (eg/t2g-like) resolves whole-shell.

    The impurity is one complete l=1 shell whose partitions are crystal-field sub-manifolds
    (not individually spin-doubled l-shells), described by a single whole-impurity rotation in
    sorted-orbital order. The per-partition derivation cannot size its sub-shell S_+ to the
    whole rotation, so the pairing is read from the full-shell S_+ across all manifolds at once.
    """
    from impurityModel.ed.spin_pairs import (
        _impurity_pairs_per_partition,
        _impurity_pairs_whole_shell,
        derive_spin_pairs,
        spin_pairs_consistent_with_h,
    )

    # Impurity = l=1 spherical shell, spin-down 0,1,2 / spin-up 3,4,5, split into two manifolds
    # {orbital 1} and {orbitals 0,2}. Baths 6 dn/7 up couple to spatial orbital 0, 8 dn/9 up to 1.
    Hop = _hop(
        [(0, 0.2), (1, 0.1), (2, 0.3), (3, 0.2), (4, 0.1), (5, 0.3), (6, -0.4), (7, -0.4), (8, 0.6), (9, 0.6)],
        [(0, 6, 0.5), (3, 7, 0.5), (1, 8, 0.4), (4, 9, 0.4)],
    )
    # Symmetry-adapted (spin-blind spatial) rotation: the same 3x3 mixing on each spin block.
    theta = 0.4
    U = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rot = np.zeros((6, 6), dtype=complex)
    rot[:3, :3] = U
    rot[3:, 3:] = U
    imp_orbitals = {0: [[1, 4]], 1: [[0, 2, 3, 5]]}

    # Per-partition cannot resolve manifolds under a whole-shell rotation; whole-shell does.
    assert _impurity_pairs_per_partition(imp_orbitals, rot) is None
    assert sorted(_impurity_pairs_whole_shell(imp_orbitals, rot)) == [(0, 3), (1, 4), (2, 5)]

    derived = derive_spin_pairs(Hop, imp_orbitals, rot, 10)
    assert derived is not None
    imp_pairs, bath_pairs = derived
    assert sorted(imp_pairs) == [(0, 3), (1, 4), (2, 5)]
    assert sorted(bath_pairs) == [(6, 7), (8, 9)]
    assert spin_pairs_consistent_with_h(Hop, imp_pairs + bath_pairs, 10)


def test_correlation_diagnostics_analytic():
    """Double occupancy / local moments / Sz^2 / Hund matrix on constructed two-orbital states."""
    from impurityModel.ed.observables import compute_correlation_diagnostics

    # Two impurity spatial orbitals: pairs (0,1) and (2,3); no bath (4-orbital space).
    imp_pairs = [(0, 1), (2, 3)]
    tau = 0.01

    # State 1: both orbitals doubly occupied (a "d10-like" closed shell).
    full = _state([([0, 1, 2, 3], 1.0)])
    rho_full = np.diag([1.0, 1.0, 1.0, 1.0]).astype(complex)
    corr = compute_correlation_diagnostics(ManyBodyState.from_states([full]), np.array([0.0]), tau, rho_full, imp_pairs)
    assert np.allclose(corr["docc"], [1.0, 1.0], atol=1e-12)
    assert np.isclose(corr["docc_total"], 2.0, atol=1e-12)
    assert np.allclose(corr["local_moment_z2"], 0.0, atol=1e-12)
    assert np.isclose(corr["sz2_thermal"], 0.0, atol=1e-12)
    assert np.allclose(corr["hund"], 0.0, atol=1e-12)

    # State 2: stretched triplet, one up electron per orbital.
    triplet = _state([([1, 3], 1.0)])
    rho_t = np.diag([0.0, 1.0, 0.0, 1.0]).astype(complex)
    corr = compute_correlation_diagnostics(ManyBodyState.from_states([triplet]), np.array([0.0]), tau, rho_t, imp_pairs)
    assert np.allclose(corr["docc"], 0.0, atol=1e-12)
    assert np.allclose(corr["local_moment_z2"], 0.25, atol=1e-12)  # a full 1/2 moment per orbital
    assert np.isclose(corr["sz2_thermal"], 1.0, atol=1e-12)  # (Sz = 1)^2
    # Hund matrix: diagonal <S_a^2> = 3/4, off-diagonal <S_1.S_2> = +1/4 (aligned spins).
    assert np.allclose(np.diag(corr["hund"]), 0.75, atol=1e-12)
    assert np.isclose(corr["hund"][0, 1], 0.25, atol=1e-12)

    # State 3: inter-orbital singlet -> <S_1.S_2> = -3/4, <Sz^2> = 0.
    singlet = _state([([1, 2], 1.0), ([0, 3], -1.0)])
    rho_s = np.diag([0.5, 0.5, 0.5, 0.5]).astype(complex)
    corr = compute_correlation_diagnostics(ManyBodyState.from_states([singlet]), np.array([0.0]), tau, rho_s, imp_pairs)
    assert np.isclose(corr["hund"][0, 1], -0.75, atol=1e-12)
    assert np.isclose(corr["sz2_thermal"], 0.0, atol=1e-12)
    assert np.allclose(corr["docc"], 0.0, atol=1e-12)


def test_energy_decomposition_analytic():
    """Tr(h rho) block split on a hand-built rho/h; Coulomb = remainder."""
    from impurityModel.ed.observables import compute_energy_decomposition

    # 2 impurity orbitals (0,1), 2 bath orbitals (2,3).
    h1 = np.array(
        [
            [-1.0, 0.0, 0.5, 0.0],
            [0.0, -1.0, 0.0, 0.5],
            [0.5, 0.0, -3.0, 0.0],
            [0.0, 0.5, 0.0, -3.0],
        ],
        dtype=complex,
    )
    # rho[i,j] = <c_j^dag c_i>: diagonal occupations + imp-bath coherence.
    rho = np.array(
        [
            [0.3, 0.0, 0.1, 0.0],
            [0.0, 0.3, 0.0, 0.1],
            [0.1, 0.0, 0.9, 0.0],
            [0.0, 0.1, 0.0, 0.9],
        ],
        dtype=complex,
    )
    e_total = -6.0
    dec = compute_energy_decomposition(rho, h1, [0, 1], e_total)
    assert np.isclose(dec["e_imp_1b"], 2 * (-1.0 * 0.3), atol=1e-12)
    assert np.isclose(dec["e_bath"], 2 * (-3.0 * 0.9), atol=1e-12)
    assert np.isclose(dec["e_hyb"], 2 * (2 * 0.5 * 0.1), atol=1e-12)
    assert np.isclose(dec["e_one_body"], dec["e_imp_1b"] + dec["e_bath"] + dec["e_hyb"], atol=1e-12)
    assert np.isclose(dec["e_coulomb"], e_total - dec["e_one_body"], atol=1e-12)


def test_screening_diagnostics_analytic():
    """Bath-level table + per-level correlation on the two-spin singlet/triplet."""
    from impurityModel.ed.observables import compute_screening_diagnostics

    imp_pairs = [(0, 1)]
    bath_pairs = [(2, 3)]
    h1 = np.zeros((4, 4), dtype=complex)
    h1[2, 2] = h1[3, 3] = -2.0
    h1[0, 2] = h1[2, 0] = 0.7
    h1[1, 3] = h1[3, 1] = 0.7
    rho = np.diag([0.5, 0.5, 0.5, 0.5]).astype(complex)

    singlet = _state([([1, 2], 1.0), ([0, 3], -1.0)])
    scr = compute_screening_diagnostics(
        ManyBodyState.from_states([singlet]), np.array([0.0]), 0.01, rho, imp_pairs, bath_pairs, h1
    )
    assert len(scr["levels"]) == 1
    row = scr["levels"][0]
    assert np.isclose(row["eps_dn"], -2.0) and np.isclose(row["eps_up"], -2.0)
    assert np.isclose(row["v_dn"], 0.7) and np.isclose(row["v_up"], 0.7)
    assert np.isclose(row["sisb"], -0.75, atol=1e-12)  # singlet fully screened

    # z_only mode: singlet <Sz_imp Sz_b> = -1/4.
    scr_z = compute_screening_diagnostics(
        ManyBodyState.from_states([singlet]), np.array([0.0]), 0.01, rho, imp_pairs, bath_pairs, h1, z_only=True
    )
    assert scr_z["z_only"]
    assert np.isclose(scr_z["levels"][0]["sisb"], -0.25, atol=1e-12)

    # Channel resolution: one group -> its value equals the total.
    scr_g = compute_screening_diagnostics(
        ManyBodyState.from_states([singlet]),
        np.array([0.0]),
        0.01,
        rho,
        imp_pairs,
        bath_pairs,
        h1,
        imp_groups={"0": imp_pairs},
    )
    assert scr_g["channels"] == [("0", scr_g["levels"][0]["sisb"])]


def test_spin_pair_consistency_polarized_bath():
    """A collinear spin-polarized bath passes the Sz check but fails the S+ check.

    RSPt-style setup: all spin polarization lives in the hybridization (spin-degenerate
    impurity block, spin-split bath energies and hoppings). The index-convention pairing
    is a correct labelling ([h,Sz]=0) even though full SU(2) consistency fails.
    """
    from impurityModel.ed.spin_pairs import (
        collinear_spin_pairs_consistent_with_h,
        spin_pair_consistency,
        spin_pairs_consistent_with_h,
    )

    # imp (0 dn, 1 up); valence bath (2 dn, 4 up); conduction bath (3 dn, 5 up).
    # Spin-degenerate impurity; bath energies AND hoppings differ per spin.
    Hop = _hop(
        [(0, -0.1), (1, -0.1), (2, -1.0), (3, 0.7), (4, -0.8), (5, 0.9)],
        [(0, 2, 0.5), (0, 3, 0.3), (1, 4, 0.45), (1, 5, 0.25)],
    )
    imp_pairs = [(0, 1)]
    bath_pairs = [(2, 4), (3, 5)]

    sz_ok, splus_ok = spin_pair_consistency(Hop, imp_pairs + bath_pairs, 6)
    assert sz_ok and not splus_ok
    assert not spin_pairs_consistent_with_h(Hop, imp_pairs + bath_pairs, 6)
    # The collinear check accepts: labels verified globally, pairing verified on the impurity.
    assert collinear_spin_pairs_consistent_with_h(Hop, imp_pairs, bath_pairs, 6)


def test_collinear_check_rejects_mislabels_and_polarized_impurity():
    """The collinear check still rejects what it cannot verify.

    A relative bath mislabel breaks [h,Sz]=0; spin polarization on the *impurity* itself
    breaks the impurity-block SU(2) (the transverse impurity operators would be wrong).
    """
    from impurityModel.ed.spin_pairs import collinear_spin_pairs_consistent_with_h

    Hop = _hop(
        [(0, -0.1), (1, -0.1), (2, -1.0), (3, 0.7), (4, -0.8), (5, 0.9)],
        [(0, 2, 0.5), (0, 3, 0.3), (1, 4, 0.45), (1, 5, 0.25)],
    )
    # Relative mislabel: swapping one bath pair's (dn, up) makes h couple unlike labels.
    assert not collinear_spin_pairs_consistent_with_h(Hop, [(0, 1)], [(4, 2), (3, 5)], 6)

    # Spin-split impurity on-site energies: labels fine, impurity pairing not SU(2).
    Hop_imp_pol = _hop(
        [(0, -0.2), (1, -0.1), (2, -1.0), (3, 0.7), (4, -0.8), (5, 0.9)],
        [(0, 2, 0.5), (0, 3, 0.3), (1, 4, 0.45), (1, 5, 0.25)],
    )
    assert not collinear_spin_pairs_consistent_with_h(Hop_imp_pol, [(0, 1)], [(2, 4), (3, 5)], 6)


def test_resolve_spin_pairs_cascade():
    """resolve_spin_pairs: SU(2) case -> exact pairing; polarized bath -> pairing_approx;
    spin-polarized impurity -> None."""
    from impurityModel.ed.spin_pairs import resolve_spin_pairs

    impurity_orbitals = {0: [[0, 1]]}
    bath_states = ({0: [[2, 4]]}, {0: [[3, 5]]})
    rot = np.eye(2, dtype=complex)

    # Fully SU(2)-symmetric: spin-degenerate bath energies and hoppings.
    Hop_su2 = _hop(
        [(0, -0.1), (1, -0.1), (2, -1.0), (3, 0.7), (4, -1.0), (5, 0.7)],
        [(0, 2, 0.5), (0, 3, 0.3), (1, 4, 0.5), (1, 5, 0.3)],
    )
    resolved = resolve_spin_pairs(Hop_su2, impurity_orbitals, bath_states, rot, 6)
    assert resolved is not None
    imp_pairs, bath_pairs, approx = resolved
    assert imp_pairs == [(0, 1)] and not approx
    assert sorted(bath_pairs) == [(2, 4), (3, 5)]

    # Collinear spin-polarized bath: accepted with the pairing_approx flag.
    Hop_pol = _hop(
        [(0, -0.1), (1, -0.1), (2, -1.0), (3, 0.7), (4, -0.8), (5, 0.9)],
        [(0, 2, 0.5), (0, 3, 0.3), (1, 4, 0.45), (1, 5, 0.25)],
    )
    resolved = resolve_spin_pairs(Hop_pol, impurity_orbitals, bath_states, rot, 6)
    assert resolved is not None
    assert resolved[2] is True

    # Spin-polarized impurity: nothing to trust.
    Hop_imp_pol = _hop(
        [(0, -0.2), (1, -0.1), (2, -1.0), (3, 0.7), (4, -0.8), (5, 0.9)],
        [(0, 2, 0.5), (0, 3, 0.3), (1, 4, 0.45), (1, 5, 0.25)],
    )
    assert resolve_spin_pairs(Hop_imp_pol, impurity_orbitals, bath_states, rot, 6) is None


def _thermal_sisb(out):
    """Parse the '<S_imp.S_bath> = value' thermal line from calc_gs output."""
    line = next(ln for ln in out.splitlines() if ln.lstrip().startswith("<S_imp.S_bath>"))
    return float(line.split("=")[1])


def _cubic_dshell(n=10):
    """Build the whole-d-shell Casimir operators in cubic harmonics + the spherical->cubic rotation."""
    from impurityModel.ed import atomic_physics
    from impurityModel.ed.observables import make_impurity_casimir_operators

    Rot = atomic_physics.get_spherical_2_cubic_matrix(spinpol=True, l=2)  # spherical<->cubic (10x10)
    l_ops, s_ops, j_ops = make_impurity_casimir_operators({0: [list(range(n))]}, Rot.conj().T)
    return l_ops, s_ops, j_ops


def test_whole_shell_casimir_aggregation_dshell():
    """Aggregating a manifold-grouped d-shell into the whole l-shell builds correct L/S/J.

    Regression for the calc_gs whole-shell Casimir fix. Per-manifold (eg:4 / t2g:6) the build
    must raise (not a spin-doubled l-shell); aggregated over the whole shell it must succeed and,
    on the known high-spin d8 determinant (t2g^6 eg-up^2, S=1 Ms=1), give <S^2> = 2 exactly.
    """
    from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix
    from impurityModel.ed.observables import apply_casimir, make_impurity_casimir_operators

    Rot = get_spherical_2_cubic_matrix(spinpol=True, l=2)
    # Per-manifold build raises (the case that made calc_gs skip the Casimirs before the fix).
    import pytest

    with pytest.raises(ValueError):
        make_impurity_casimir_operators({0: [[0, 1, 5, 6]]}, Rot.conj().T)  # eg only (4 orbs)

    # Whole-shell build succeeds; <S^2> = 2 on the high-spin d8 determinant (rotation-invariant).
    _, s_ops, _ = _cubic_dshell(10)

    def _sd10(occ):
        data = bytearray(2)
        for o in occ:
            data[o // 8] |= 1 << (7 - o % 8)
        return ManyBodyState({SlaterDeterminant.from_bytes(bytes(data)): 1.0})

    # cubic order: eg dn 0,1 ; t2g dn 2,3,4 ; eg up 5,6 ; t2g up 7,8,9. d8 Ms=1: t2g^6 + eg-up^2.
    psi = _sd10([2, 3, 4, 5, 6, 7, 8, 9])
    s2 = float(np.real(inner(psi, apply_casimir(psi, *s_ops))))
    assert np.isclose(s2, 2.0, atol=1e-9)  # S=1


def test_calc_gs_reports_casimirs_for_cubic_manifold_grouped_dshell(capsys):
    """calc_gs reports S^2/L^2/J^2 (not silently skipped) for a manifold-grouped cubic d-shell.

    Integration/plumbing check for the whole-shell aggregation: group_orbitals_by_blocks splits
    the d-shell into eg/t2g manifolds, so the per-partition Casimir build raises; calc_gs must
    aggregate them and still report L/S/J without crashing. (Exact S is checked in the unit test
    above; this synthetic has no double counting, so it needn't land on a specific occupation.)
    """
    from collections import OrderedDict

    import pytest

    from impurityModel.ed import atomic_physics
    from impurityModel.ed.groundstate import calc_gs
    from impurityModel.ed.symmetries import (
        classify_bath_occupation,
        group_orbitals_by_blocks,
        impurity_block_structure,
    )

    Fdd = [7.5, 0, 9.9, 0, 6.6]
    uOp = atomic_physics.getUop(l1=2, l2=2, l3=2, l4=2, R=Fdd)
    nB = OrderedDict({2: 0})
    V4 = np.zeros((10,) * 4, dtype=complex)
    for proc, val in uOp.items():
        ix = [c2i(nB, proc[p][0]) for p in range(4)]
        # RSPt convention: V4[i,j,k,l] multiplies c^dag_i c^dag_j c_l c_k, so
        # the process operators (p2, p3) fill the tensor with swapped indices.
        V4[ix[0], ix[1], ix[3], ix[2]] = 2.0 * val
    Rot = atomic_physics.get_spherical_2_cubic_matrix(spinpol=True, l=2)
    u4 = np.einsum("ia,jb,ijkl,kc,ld->abcd", Rot.conj(), Rot.conj(), V4, Rot, Rot, optimize=True)
    u_dict = atomic_physics.getUop_from_rspt_u4(u4)

    eg, t2g = [0, 1, 5, 6], [2, 3, 4, 7, 8, 9]
    h0 = {}
    for o in t2g:
        h0[((o, "c"), (o, "a"))] = -8.6
    for o in eg:
        h0[((o, "c"), (o, "a"))] = -8.0
    for k in range(10):
        b = 10 + k
        h0[((b, "c"), (b, "a"))] = -0.5  # valence (below Fermi 0)
        h0[((k, "c"), (b, "a"))] = 0.15
        h0[((b, "c"), (k, "a"))] = 0.15
    Hop = ManyBodyOperator(h0) + ManyBodyOperator(u_dict)

    imp_flat = list(range(10))
    bs = impurity_block_structure(Hop, imp_flat)
    val_flat, con_flat = classify_bath_occupation(Hop, imp_flat)
    impurity_orbitals, bath_states = group_orbitals_by_blocks(Hop, imp_flat, val_flat, con_flat, bs)
    assert len(impurity_orbitals) >= 2  # eg / t2g -> manifold-grouped (the case that used to skip)
    N0 = {g: (6 if len(blocks[0]) == 6 else 2) for g, blocks in impurity_orbitals.items()}

    setup = dict(
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        N0=N0,
        mixed_valence=dict.fromkeys(impurity_orbitals, 1),
        tau=0.01,
        # dense_cutoff=4000 forced the O(N^3) dense eigh branch on every trial solve in the
        # eg/t2g diagonal-move probe + axis walk (~330s measured). This is a "must not raise" +
        # sane-S(S+1) plumbing check with no golden numeric target, so the dense/iterative
        # choice is free: 200 keeps every trial iterative and measured 23s, unchanged assertions.
        dense_cutoff=200,
        spin_flip_dj=True,
        comm=None,
        truncation_threshold=200000,
    )
    calc_gs(Hop, setup, bs, Rot.conj().T, verbose=True, slaterWeightMin=1e-12)  # must not raise
    out = capsys.readouterr().out
    assert "<S^2>" in out and "<L^2>" in out and "<J^2>" in out  # reported, not skipped
    s2 = float(next(ln for ln in out.splitlines() if ln.lstrip().startswith("<S^2>")).split("=")[1].split("(")[0])
    assert s2 > 0.0 and s2 == pytest.approx(round(s2 * 4) / 4, abs=0.05)  # sane S(S+1) value


def test_kondo_correlation_fallback_matches_fast_path(capsys):
    """A non-down-then-up bath ordering falls back to derive_spin_pairs and agrees."""
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.groundstate import calc_gs

    ed, U, ev, ec, V = -2.0, 6.0, -4.0, 4.0, 1.0
    diag = [(0, ed), (1, ed), (2, ev), (3, ev), (4, ec), (5, ec)]
    hops = [(0, 2, V), (1, 3, V), (0, 4, V), (1, 5, V)]
    terms = {((o, "c"), (o, "a")): e for o, e in diag}
    for a, b, t in hops:
        terms[((a, "c"), (b, "a"))] = t
        terms[((b, "c"), (a, "a"))] = t
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = U

    Hop = ManyBodyOperator(terms)
    bs = BlockStructure(
        blocks=[[0, 1]],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )

    def run(valence_block, conduction_block):
        setup = dict(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [valence_block]}, {0: [conduction_block]}),
            N0={0: 1},
            mixed_valence={0: 1},
            tau=0.01,
            dense_cutoff=1000,
            spin_flip_dj=False,
            comm=None,
            truncation_threshold=100000,
        )
        calc_gs(Hop, setup, bs, np.eye(2, dtype=complex), verbose=True, slaterWeightMin=1e-12)
        return _thermal_sisb(capsys.readouterr().out)

    fast = run([2, 3], [4, 5])  # down-then-up within each block -> fast path
    fallback = run([3, 2], [5, 4])  # spins swapped within block -> derive_spin_pairs fallback
    assert np.isclose(fast, fallback, atol=1e-9)


# --------------------------------------------------------------------------- #
# compute_static_susceptibilities (Curie terms of the retained manifold)
# --------------------------------------------------------------------------- #
def test_static_susceptibility_free_spin_half():
    """Free spin-1/2 doublet: chi_spin_zz = 1/(4 tau), frozen charge -> chi_charge = 0."""
    from impurityModel.ed.observables import compute_static_susceptibilities, make_spin_operators

    tau = 0.02
    # One spatial orbital, (dn, up) = (0, 1); degenerate |dn>, |up> manifold.
    psis = [_state([((0,), 1.0)], n_orbs=2), _state([((1,), 1.0)], n_orbs=2)]
    es = np.array([0.0, 0.0])
    s_z = make_spin_operators([(0, 1)])[2]
    chi = compute_static_susceptibilities(ManyBodyState.from_states(psis), es, tau, impurity_indices=[0, 1], s_z_op=s_z)
    assert chi["chi_spin_zz"] == pytest.approx(0.25 / tau, rel=1e-12)
    assert chi["chi_charge"] == pytest.approx(0.0, abs=1e-12)
    assert chi["chi_orb_zz"] is None


def test_static_susceptibility_orbital_and_cross():
    """p-shell single electron, manifold {|ml=1,up>, |ml=-1,dn>}:

    <Lz>_th = <Sz>_th = 0 but <Lz^2>_th = 1, <Sz Lz>_th = 1/2 ->
    chi_orb = 1/tau, chi_spin_orb = 1/(2 tau) (ferro spin-orbital locking).
    """
    from impurityModel.ed.observables import compute_static_susceptibilities, make_impurity_casimir_operators

    tau = 0.05
    n = 6  # spin-doubled p shell, spherical layout: dn ml=-1,0,1 -> 0,1,2; up -> 3,4,5
    l_ops, s_ops, _ = make_impurity_casimir_operators({0: [list(range(n))]}, np.eye(n, dtype=complex))
    psis = [
        _state([((_index(1, 1, 1),), 1.0)], n_orbs=n),  # |ml=+1, up>
        _state([((_index(1, -1, -1),), 1.0)], n_orbs=n),  # |ml=-1, dn>
    ]
    es = np.array([0.0, 0.0])
    chi = compute_static_susceptibilities(
        ManyBodyState.from_states(psis), es, tau, impurity_indices=list(range(n)), s_z_op=s_ops[2], l_z_op=l_ops[2]
    )
    assert chi["chi_orb_zz"] == pytest.approx(1.0 / tau, rel=1e-12)
    assert chi["chi_spin_zz"] == pytest.approx(0.25 / tau, rel=1e-12)
    assert chi["chi_spin_orb"] == pytest.approx(0.5 / tau, rel=1e-12)
    assert chi["chi_charge"] == pytest.approx(0.0, abs=1e-12)


def test_static_susceptibility_mixed_valence_charge():
    """Valence superposition a|N=1> + b|N=2>: chi_charge = (<N^2>-<N>^2)/tau exactly."""
    from impurityModel.ed.observables import compute_static_susceptibilities

    tau = 0.1
    a2, b2 = 0.7, 0.3
    psi = _state([((0,), np.sqrt(a2)), ((0, 1), np.sqrt(b2))], n_orbs=2)
    chi = compute_static_susceptibilities(
        ManyBodyState.from_states([psi]), np.array([0.0]), tau, impurity_indices=[0, 1]
    )
    mean = a2 * 1 + b2 * 2
    second = a2 * 1 + b2 * 4
    assert chi["chi_charge"] == pytest.approx((second - mean**2) / tau, rel=1e-12)


# --------------------------------------------------------------------------- #
# impurity_shell_rhos / compute_shell_observables (multi-shell reporting fix)
# --------------------------------------------------------------------------- #
def _two_shell_fixture():
    """Filled l=1 (2p, 6 spin-orbitals) + a single |ml=+2,up> electron in l=2 (3d).

    Global orbital layout: l=1 shell occupies 0-5 (down 0-2, up 3-5); l=2 shell occupies
    6-15 (down 6-10, up 11-15), so local d-orbital index 9 (|ml=+2,up>, matching
    ``_index(2, 2, +1)``) is global orbital 15. Both shells use an identity rotation
    (computational == spherical), so ``rho_imp`` can be built directly as a diagonal
    occupation-number matrix.
    """
    impurity_orbitals = {1: [list(range(6))], 2: [list(range(6, 16))]}
    rot_to_spherical = {1: np.eye(6, dtype=complex), 2: np.eye(10, dtype=complex)}
    occupied = [0, 1, 2, 3, 4, 5, 6 + _index(2, 2, +1)]
    rho_imp = np.diag([1.0 if orb in occupied else 0.0 for orb in range(16)]).astype(complex)
    return impurity_orbitals, rot_to_spherical, rho_imp, occupied


def test_impurity_shell_rhos_splits_by_shell():
    """Each shell's spherical rho is sliced out with its own global orbitals only."""
    from impurityModel.ed.observables import impurity_shell_rhos

    impurity_orbitals, rot_to_spherical, rho_imp, _ = _two_shell_fixture()
    shells = list(impurity_shell_rhos(rho_imp, rot_to_spherical, impurity_orbitals))
    assert [l for l, _, _ in shells] == [1, 2]
    assert [partition for _, partition, _ in shells] == [1, 2]  # partition key, joinable elsewhere
    _l1, _partition1, rho1 = shells[0]
    _l2, _partition2, rho2 = shells[1]
    assert rho1.shape == (6, 6)
    assert np.allclose(rho1, np.eye(6))
    assert rho2.shape == (10, 10)
    assert np.isclose(np.real(np.trace(rho2)), 1.0)
    assert np.isclose(np.real(rho2[_index(2, 2, +1), _index(2, 2, +1)]), 1.0)


def test_impurity_shell_rhos_non_dict_rotation_is_one_aggregate_shell():
    """A plain ndarray rotation (selfenergy path, incl. eg/t2g-grouped impurity_orbitals)
    always yields a single aggregate shell, never split by (ambiguous) orbital count.

    A t2g manifold has 6 spin-orbitals -- the same count as a genuine l=1 shell -- so
    orbital-count alone cannot tell them apart; only the rotation's type (dict vs ndarray)
    may decide whether to split. Regression guard for that ambiguity.
    """
    from impurityModel.ed.observables import impurity_shell_rhos

    # eg (4 spin-orbitals) + t2g (6 spin-orbitals): a sub-shell grouping of one d-shell,
    # as `group_orbitals_by_blocks` would produce for a selfenergy run.
    impurity_orbitals = {0: [list(range(4))], 1: [list(range(4, 10))]}
    rot_to_spherical = np.eye(10, dtype=complex)  # single matrix -> selfenergy signature
    rho_imp = np.eye(10, dtype=complex)
    shells = list(impurity_shell_rhos(rho_imp, rot_to_spherical, impurity_orbitals))
    assert len(shells) == 1
    l, partition, rho = shells[0]
    assert l is None
    assert partition is None
    assert rho.shape == (10, 10)


def test_impurity_shell_rhos_no_impurity_orbitals_is_one_aggregate_shell():
    """`impurity_orbitals=None` (or omitted) reproduces today's single-shell behaviour."""
    from impurityModel.ed.observables import impurity_shell_rhos

    rho_imp = np.eye(10, dtype=complex)
    shells = list(impurity_shell_rhos(rho_imp, np.eye(10, dtype=complex)))
    assert len(shells) == 1
    assert shells[0][0] is None


def test_impurity_shell_rhos_rejects_non_shell_dict_partition():
    """A dict rotation promises every partition is a full l-shell; a violation must raise,
    not silently fall back -- silently reinterpreting would reintroduce exactly the kind
    of misclassification this function exists to prevent."""
    from impurityModel.ed.observables import impurity_shell_rhos

    impurity_orbitals = {0: [list(range(4))]}  # 4 spin-orbitals: not 2*(2l+1) for any l
    rot_to_spherical = {0: np.eye(4, dtype=complex)}
    rho_imp = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="spin-doubled l-shell"):
        list(impurity_shell_rhos(rho_imp, rot_to_spherical, impurity_orbitals))


def test_compute_shell_observables_filled_shell_is_exactly_zero():
    """A completely filled shell is SOC-inert: every one-body angular-momentum
    observable vanishes exactly (to roundoff), regardless of any xi parameter, because
    the physical operators evaluated here don't even depend on xi -- only the *many-body*
    Hamiltonian would. This is the free, strongest structural check: a nonzero value here
    means the global-orbital -> sorted-position mapping is off by construction."""
    from impurityModel.ed.observables import compute_shell_observables

    impurity_orbitals, rot_to_spherical, rho_imp, _ = _two_shell_fixture()
    result = compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals)
    l1 = result["shells"][0]
    assert l1["l"] == 1
    assert l1["n"] == pytest.approx(6.0, abs=1e-12)
    for key in ("lz", "sz", "m_z", "j_z", "l_dot_s", "t_z"):
        assert l1[key] == pytest.approx(0.0, abs=1e-12), key


def test_compute_shell_observables_matches_known_single_electron_values():
    """The l=2 shell holds exactly the single |ml=+2,up> electron from the fixture, with
    known analytic values (matching test_spin_orbit_observable's convention)."""
    from impurityModel.ed.observables import compute_shell_observables

    impurity_orbitals, rot_to_spherical, rho_imp, _ = _two_shell_fixture()
    result = compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals)
    l2 = result["shells"][1]
    assert l2["l"] == 2
    assert l2["n"] == pytest.approx(1.0, abs=1e-12)
    assert l2["lz"] == pytest.approx(2.0, abs=1e-12)
    assert l2["sz"] == pytest.approx(0.5, abs=1e-12)
    assert l2["m_z"] == pytest.approx(3.0, abs=1e-12)  # lz + 2sz
    assert l2["j_z"] == pytest.approx(2.5, abs=1e-12)  # lz + sz
    assert l2["l_dot_s"] == pytest.approx(1.0, abs=1e-12)  # ml*ms = 2*0.5


def test_compute_shell_observables_total_is_sum_over_shells():
    """The reported total must be the sum over shells, not a re-evaluation on the
    concatenated rho (that re-evaluation is precisely the bug being fixed)."""
    from impurityModel.ed.observables import compute_shell_observables

    impurity_orbitals, rot_to_spherical, rho_imp, _ = _two_shell_fixture()
    result = compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals)
    l1, l2 = result["shells"]
    total = result["total"]
    for key in ("n", "n_dn", "n_up", "lz", "sz", "m_z", "j_z", "l_dot_s", "t_z"):
        assert total[key] == pytest.approx(l1[key] + l2[key], abs=1e-12)
    assert total["n"] == pytest.approx(7.0, abs=1e-12)
    assert total["sz"] == pytest.approx(0.5, abs=1e-12)
    assert total["lz"] == pytest.approx(2.0, abs=1e-12)


def test_compute_shell_observables_carries_partition_for_ambiguous_l_join():
    """Two distinct l=1 shells (same inferred l, different partitions) must each carry
    their own distinguishing `partition` key -- joining Stage 4's per-shell table against
    make_impurity_casimir_operators's per-shell Casimirs by `l` alone would silently merge
    them; `partition` is the only safe join key."""
    from impurityModel.ed.observables import compute_shell_observables

    # Two independent l=1 shells at global orbitals 0-5 and 6-11, distinguished only by
    # partition key ("A", "B") -- inferred l is 1 for both.
    impurity_orbitals = {"A": [list(range(6))], "B": [list(range(6, 12))]}
    rot_to_spherical = {"A": np.eye(6, dtype=complex), "B": np.eye(6, dtype=complex)}
    rho_imp = np.eye(12, dtype=complex)
    result = compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals)
    assert [shell["l"] for shell in result["shells"]] == [1, 1]
    assert [shell["partition"] for shell in result["shells"]] == ["A", "B"]


def test_old_concatenated_inference_was_wrong_on_the_two_shell_fixture():
    """Regression trap: pin down *why* the pre-fix behaviour was wrong, not just that the
    new path works. Calling the single-shell primitives directly on the concatenated
    16-orbital block (today's calc_gs behaviour before this fix) misinfers l=3 (from
    floor((16//2-1)/2)=3), silently drops orbitals 14-15, and returns wildly wrong values
    instead of the correct total (sz=0.5, lz=2.0)."""
    from impurityModel.ed.observables import get_Lz_from_rho_spherical, get_Sz_from_rho_spherical

    _, _, rho_imp, _ = _two_shell_fixture()
    sz_old = get_Sz_from_rho_spherical(rho_imp)  # l inferred as 3 (wrong)
    lz_old = get_Lz_from_rho_spherical(rho_imp)
    assert sz_old == pytest.approx(-3.0, abs=1e-12)
    assert lz_old == pytest.approx(-3.0, abs=1e-12)
    assert not np.isclose(sz_old, 0.5)
    assert not np.isclose(lz_old, 2.0)


def test_compute_shell_observables_matches_many_body_casimir_path():
    """Cross-path oracle: the density-matrix total must agree with the independent
    many-body-operator path (make_impurity_casimir_operators + compute_static_susceptibilities),
    which infers l per shell correctly by construction. This is exactly the two paths that
    disagreed in the real bug report (ground_state_statistics.json: sz_thermal ~ -1e-11 vs
    the (buggy) rho-path sz ~ -0.58)."""
    from impurityModel.ed.observables import (
        compute_shell_observables,
        compute_static_susceptibilities,
        make_impurity_casimir_operators,
    )

    impurity_orbitals, rot_to_spherical, rho_imp, occupied = _two_shell_fixture()
    result = compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals)

    psi = _sd(occupied, n_orbs=16)
    psi = ManyBodyState({psi: 1.0})
    l_ops, s_ops, _ = make_impurity_casimir_operators(impurity_orbitals, rot_to_spherical)
    impurity_indices = list(range(16))
    chi = compute_static_susceptibilities(
        ManyBodyState.from_states([psi]), np.array([0.0]), 0.05, impurity_indices, s_z_op=s_ops[2], l_z_op=l_ops[2]
    )
    assert result["total"]["sz"] == pytest.approx(chi["sz_thermal"], abs=1e-9)
    assert result["total"]["lz"] == pytest.approx(chi["lz_thermal"], abs=1e-9)


def test_impurity_shell_rhos_orders_by_orbital_position_not_by_l():
    """Shells are yielded in ascending global-orbital order (matching how ``rho_imp`` is
    built by ``groundstate.calc_gs``), not ascending ``l``. Nothing forces the lower-``l``
    shell to hold the lower orbital indices, so a fixture with the l=2 shell first must
    yield [2, 1], not [1, 2]."""
    from impurityModel.ed.observables import impurity_shell_rhos

    # l=2 (10 spin-orbitals) at global 0-9, l=1 (6 spin-orbitals) at global 10-15 --
    # deliberately reversed relative to _two_shell_fixture.
    impurity_orbitals = {2: [list(range(10))], 1: [list(range(10, 16))]}
    rot_to_spherical = {1: np.eye(6, dtype=complex), 2: np.eye(10, dtype=complex)}
    rho_imp = np.eye(16, dtype=complex)
    shells = list(impurity_shell_rhos(rho_imp, rot_to_spherical, impurity_orbitals))
    assert [l for l, _, _ in shells] == [2, 1]
    assert [partition for _, partition, _ in shells] == [2, 1]


def _random_unitary(n, seed):
    """A Haar-ish random unitary via QR of a complex Gaussian matrix (fixed seed)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(a)
    phases = np.diagonal(r) / np.abs(np.diagonal(r))
    return q * phases.conj()


def test_compute_shell_observables_matches_many_body_casimir_path_with_rotation():
    """Same cross-path oracle as test_compute_shell_observables_matches_many_body_casimir_path,
    but with a genuine non-trivial unitary rotation on the l=2 shell -- matching the real
    tutorial run, where rot_to_spherical[2] = u_imp.conj().T is not the identity. This is
    exactly the composition a wrong dict key or a wrong shell ordering would corrupt while
    leaving <N> (rotation-invariant) looking correct, so it is the case that actually
    exercises the risk the dict-key-vs-inferred-l fix (Stage 1 review item 1) addresses."""
    from impurityModel.ed.observables import (
        compute_shell_observables,
        compute_static_susceptibilities,
        make_impurity_casimir_operators,
    )

    w = _random_unitary(10, seed=1234)
    impurity_orbitals = {1: [list(range(6))], 2: [list(range(6, 16))]}
    rot_to_spherical = {1: np.eye(6, dtype=complex), 2: w}
    # Single occupied computational orbital in the l=2 shell (local index 3 -> global 9),
    # plus the filled l=1 core (global 0-5).
    occupied = [0, 1, 2, 3, 4, 5, 9]
    rho_imp = np.diag([1.0 if orb in occupied else 0.0 for orb in range(16)]).astype(complex)

    result = compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals)

    psi = ManyBodyState({_sd(occupied, n_orbs=16): 1.0})
    l_ops, s_ops, _ = make_impurity_casimir_operators(impurity_orbitals, rot_to_spherical)
    chi = compute_static_susceptibilities(
        ManyBodyState.from_states([psi]),
        np.array([0.0]),
        0.05,
        list(range(16)),
        s_z_op=s_ops[2],
        l_z_op=l_ops[2],
    )
    assert result["total"]["sz"] == pytest.approx(chi["sz_thermal"], abs=1e-9)
    assert result["total"]["lz"] == pytest.approx(chi["lz_thermal"], abs=1e-9)
    # A non-trivial rotation must actually move Lz/Sz away from the un-rotated single-
    # electron values (2.0 / 0.5 for a bare |ml=2,up>) -- otherwise this test isn't
    # exercising the rotation at all.
    assert not np.isclose(result["total"]["lz"], 2.0, atol=1e-6)


def test_compute_shell_observables_single_key_dict_matches_no_dict():
    """A dict with exactly one shell (e.g. a plain d-shell spectra run, no core hole --
    `get_spectra.py` always passes a dict, even with a single correlated l) must agree
    with the `impurity_orbitals=None` aggregate path, for both an identity and a
    non-trivial rotation. Every printer test above exercises `impurity_orbitals=None`
    (the never-actually-used-in-production case); production's *single*-shell case is
    the dict-of-one, which this pins directly instead of by argument alone."""
    from impurityModel.ed.observables import compute_shell_observables

    n = 10
    rho = np.diag([1.0] * 3 + [0.0] * 4 + [1.0] * 3).astype(complex)
    for rot in (np.eye(n, dtype=complex), _random_unitary(n, seed=7)):
        via_dict = compute_shell_observables(rho, {2: rot}, {2: [list(range(n))]})["total"]
        via_none = compute_shell_observables(rho, rot, None)["total"]
        for key in via_dict:
            assert via_dict[key] == pytest.approx(via_none[key], abs=1e-12), key


# --------------------------------------------------------------------------- #
# make_impurity_casimir_operators(per_shell=True) (Stage 3: per-shell Casimirs)
# --------------------------------------------------------------------------- #
def test_make_impurity_casimir_operators_default_return_is_unchanged():
    """per_shell=False (the default) must keep returning exactly the 3-tuple every
    existing caller (calc_gs, susceptibility.py, other tests) already unpacks."""
    from impurityModel.ed.observables import make_impurity_casimir_operators

    impurity_orbitals, rot_to_spherical, _, _ = _two_shell_fixture()
    result = make_impurity_casimir_operators(impurity_orbitals, rot_to_spherical)
    assert len(result) == 3
    l_ops, s_ops, j_ops = result
    assert len(l_ops) == 3 and len(s_ops) == 3 and len(j_ops) == 3


def test_make_impurity_casimir_operators_per_shell_sums_to_the_totals():
    """Each shell's own (L, S, J) summed together must reproduce the whole-impurity
    totals exactly -- the shells address disjoint orbitals, so this is a plain sum, and
    it is exactly the property Stage 4's per-shell table total row relies on."""
    from impurityModel.ed.observables import make_impurity_casimir_operators

    impurity_orbitals, rot_to_spherical, _, _ = _two_shell_fixture()
    l_ops, s_ops, j_ops, per_shell_ops = make_impurity_casimir_operators(
        impurity_orbitals, rot_to_spherical, per_shell=True
    )
    assert set(per_shell_ops) == {1, 2}
    assert per_shell_ops[1][0] == 1  # inferred l
    assert per_shell_ops[2][0] == 2

    # l_ops/s_ops must actually contain terms, so the equality-via-subtraction check
    # below cannot pass vacuously on an empty operator.
    assert any(len(op) > 0 for op in l_ops)
    assert any(len(op) > 0 for op in s_ops)

    summed_l = [ManyBodyOperator(), ManyBodyOperator(), ManyBodyOperator()]
    summed_s = [ManyBodyOperator(), ManyBodyOperator(), ManyBodyOperator()]
    for _l, shell_l_ops, shell_s_ops, _j in per_shell_ops.values():
        for i in range(3):
            summed_l[i] += shell_l_ops[i]
            summed_s[i] += shell_s_ops[i]
    # ManyBodyOperator equality via subtraction-to-empty (matches the algebra's own
    # normal-ordering convention rather than assuming dict equality on the raw terms).
    for i in range(3):
        diff = summed_l[i] - l_ops[i]
        assert all(abs(v) < 1e-12 for v in diff.values())
        diff = summed_s[i] - s_ops[i]
        assert all(abs(v) < 1e-12 for v in diff.values())


def test_make_impurity_casimir_operators_per_shell_matches_known_single_electron_values():
    """The l=2 shell's own S^2/L^2/J^2 (evaluated via casimir_operator + the many-body
    machinery, not just density-matrix moments) reproduce the analytic single-electron
    values for the fixture's |ml=+2,up> state: S=1/2 -> S^2=3/4, L=2 -> L^2=6,
    J=5/2 -> J^2=35/4. The l=1 shell (filled) has S=L=J=0."""
    from impurityModel.ed.observables import (
        casimir_operator,
        make_impurity_casimir_operators,
        manifold_observable_values,
    )

    impurity_orbitals, rot_to_spherical, _, occupied = _two_shell_fixture()
    _, _, _, per_shell_ops = make_impurity_casimir_operators(impurity_orbitals, rot_to_spherical, per_shell=True)
    psi = ManyBodyState({_sd(occupied, n_orbs=16): 1.0})
    manifold = ManyBodyState.from_states([psi])
    es = np.array([0.0])

    shell_l, l_ops, s_ops, j_ops = per_shell_ops[2]
    assert shell_l == 2
    for ops, expected in ((s_ops, 0.75), (l_ops, 6.0), (j_ops, 35.0 / 4.0)):
        op2 = casimir_operator(*ops)
        vals = manifold_observable_values(manifold, es, lambda blk, _op=op2: _op.apply_block(blk, 0))
        assert vals[0] == pytest.approx(expected, abs=1e-9)

    shell_l1, l_ops1, s_ops1, j_ops1 = per_shell_ops[1]
    assert shell_l1 == 1
    for ops in (s_ops1, l_ops1, j_ops1):
        op2 = casimir_operator(*ops)
        vals = manifold_observable_values(manifold, es, lambda blk, _op=op2: _op.apply_block(blk, 0))
        assert vals[0] == pytest.approx(0.0, abs=1e-9)


def test_make_impurity_casimir_operators_per_shell_raises_like_the_totals():
    """per_shell=True must not silently swallow the sub-shell ValueError the totals path
    already raises on (e.g. a grouped eg/t2g impurity) -- the caller's existing
    ValueError-catching fallback in calc_gs depends on that."""
    from impurityModel.ed.observables import make_impurity_casimir_operators

    impurity_orbitals = {0: [list(range(4))]}  # 4 spin-orbitals: not a shell for any l
    rot_to_spherical = {0: np.eye(4, dtype=complex)}
    with pytest.raises(ValueError, match="spin-doubled l-shell"):
        make_impurity_casimir_operators(impurity_orbitals, rot_to_spherical, per_shell=True)


# --------------------------------------------------------------------------- #
# block_group_labels / print_impurity_orbital_groups / per-shell table
# (Stage 4: N(...) label fix + per-shell report table)
# --------------------------------------------------------------------------- #
def _eg_t2g_block_structure():
    """A cubic d-shell auto-split into 2 equivalence classes (eg: orbitals 0,1,4,5;
    t2g: orbitals 2,3,6,7,8,9), each with 2 blocks (spin partners)."""
    from impurityModel.ed.block_structure import BlockStructure

    return BlockStructure(
        blocks=[[0, 4], [1, 5], [2, 6], [3, 7, 8, 9]],
        identical_blocks=[[0, 1], [0, 1], [2, 3], [2, 3]],
        transposed_blocks=[[], [], [], []],
        particle_hole_blocks=[[], [], [], []],
        particle_hole_transposed_blocks=[[], [], [], []],
        inequivalent_blocks=[0, 2],
    )


def test_block_group_labels_no_impurity_orbitals():
    """Without impurity_orbitals there is no group to name a class after, so labels fall
    back to plain letters."""
    from impurityModel.ed.observables import block_group_labels, get_equivalent_blocks

    bs = _eg_t2g_block_structure()
    eq = get_equivalent_blocks(bs)
    labels, legend = block_group_labels(eq, bs)
    assert labels == ["a", "b"]
    assert [orbs for _, orbs in legend] == [[0, 1, 4, 5], [2, 3, 6, 7, 8, 9]]


def test_block_group_labels_one_class_per_group():
    """One equivalence class per group -- the grouping every solver path derives from the
    block structure -- names each class after its group key, nothing else."""
    from impurityModel.ed.observables import block_group_labels, get_equivalent_blocks

    bs = _eg_t2g_block_structure()
    eq = get_equivalent_blocks(bs)
    impurity_orbitals = {0: [[0, 1, 4, 5]], 1: [[2, 3, 6, 7, 8, 9]]}
    labels, legend = block_group_labels(eq, bs, impurity_orbitals)
    assert labels == ["group 0", "group 1"]
    assert dict(legend)["group 0"] == [0, 1, 4, 5]
    assert dict(legend)["group 1"] == [2, 3, 6, 7, 8, 9]


def test_block_group_labels_group_split_into_several_classes():
    """A single group the block structure splits further (one d shell auto-split into
    eg/t2g) keeps the group key and adds a letter, so the classes stay distinguishable."""
    from impurityModel.ed.observables import block_group_labels, get_equivalent_blocks

    bs = _eg_t2g_block_structure()
    eq = get_equivalent_blocks(bs)
    impurity_orbitals = {2: [list(range(10))]}  # one shell, split into two classes
    labels, legend = block_group_labels(eq, bs, impurity_orbitals)
    assert labels == ["group 2.a", "group 2.b"]
    assert dict(legend)["group 2.a"] == [0, 1, 4, 5]
    assert dict(legend)["group 2.b"] == [2, 3, 6, 7, 8, 9]


def test_block_group_labels_multi_group_impurity():
    """A two-group impurity names each class after its own group; the keys are whatever the
    caller's impurity_orbitals uses (here the angular momenta of a 2p + 3d impurity)."""
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.observables import block_group_labels, get_equivalent_blocks

    bs = BlockStructure(
        blocks=[list(range(6)), list(range(6, 16))],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0, 1],
    )
    impurity_orbitals = {1: [list(range(6))], 2: [list(range(6, 16))]}
    eq = get_equivalent_blocks(bs)
    labels, legend = block_group_labels(eq, bs, impurity_orbitals)
    assert labels == ["group 1", "group 2"]
    assert dict(legend)["group 1"] == list(range(6))
    assert dict(legend)["group 2"] == list(range(6, 16))


def test_block_group_labels_class_spanning_groups():
    """A (synthetic, shouldn't happen in practice) class spanning two groups is named after
    both rather than silently attributed to either."""
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.observables import block_group_labels, get_equivalent_blocks

    bs = BlockStructure(
        blocks=[list(range(6)) + list(range(6, 16))],  # one block spanning both groups
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    impurity_orbitals = {1: [list(range(6))], 2: [list(range(6, 16))]}
    eq = get_equivalent_blocks(bs)
    labels, _ = block_group_labels(eq, bs, impurity_orbitals)
    assert labels == ["groups 1,2"]


def test_block_group_labels_orbital_outside_every_group():
    """An orbital in no group at all leaves nothing to name the class after, so it falls
    back to a letter instead of guessing a group."""
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.observables import block_group_labels, get_equivalent_blocks

    bs = BlockStructure(
        blocks=[[0, 1], [2, 3]],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0, 1],
    )
    eq = get_equivalent_blocks(bs)
    labels, _ = block_group_labels(eq, bs, {0: [[0, 1]]})  # orbitals 2, 3 belong to no group
    assert labels == ["group 0", "a"]


def test_print_impurity_orbital_groups_legend(capsys):
    """The legend prints one entry per label, mapping to the correct global orbitals, and
    is a no-op when there is nothing to disambiguate (a single group)."""
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.observables import get_equivalent_blocks, print_impurity_orbital_groups

    bs = _eg_t2g_block_structure()
    impurity_orbitals = {0: [[0, 1, 4, 5]], 1: [[2, 3, 6, 7, 8, 9]]}
    print_impurity_orbital_groups(get_equivalent_blocks(bs), bs, impurity_orbitals)
    out = capsys.readouterr().out
    assert "group 0: [0, 1, 4, 5]" in out
    assert "group 1: [2, 3, 6, 7, 8, 9]" in out

    single_bs = BlockStructure(
        blocks=[list(range(10))],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    print_impurity_orbital_groups(get_equivalent_blocks(single_bs), single_bs, {0: [list(range(10))]})
    assert capsys.readouterr().out == ""


def test_print_thermal_expectation_values_n_labels_name_their_group(capsys):
    """The N(...) column labels in the thermal report name the orbital group the column is
    for -- not raw block indices (the bug that produced the tutorial's meaningless
    N(4,6,9,11) label), and not a shell guessed from the group's orbital count."""
    from impurityModel.ed.observables import print_thermal_expectation_values

    bs = _eg_t2g_block_structure()
    impurity_orbitals = {0: [list(range(10))]}
    rot_to_spherical = {0: np.eye(10, dtype=complex)}
    rho = np.eye(10, dtype=complex)
    print_thermal_expectation_values(rho, 0.0, rot_to_spherical, bs, impurity_orbitals=impurity_orbitals)
    out = capsys.readouterr().out
    assert "<N(group 0.a)>" in out
    assert "<N(group 0.b)>" in out
    # The old, meaningless block-index join must not reappear.
    assert "<N(0,1)>" not in out and "<N(2,3)>" not in out


def test_print_thermal_expectation_values_per_shell_table(capsys):
    """The per-shell table appears (only) for a multi-shell impurity, with the correct
    per-shell N/Lz/Sz values and a total row, and Casimir/term columns populated from
    shell_casimir when given."""
    from impurityModel.ed.observables import print_thermal_expectation_values

    from impurityModel.ed.block_structure import BlockStructure

    bs = BlockStructure(
        blocks=[list(range(6)), list(range(6, 16))],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0, 1],
    )
    impurity_orbitals = {1: [list(range(6))], 2: [list(range(6, 16))]}
    rot_to_spherical = {1: np.eye(6, dtype=complex), 2: np.eye(10, dtype=complex)}
    rho = np.diag([1.0] * 6 + [0.8] * 10).astype(complex)
    shell_casimir = {
        1: {"l": 1, "s2_thermal": 0.0, "l2_thermal": 0.0, "j2_thermal": 0.0},
        2: {"l": 2, "s2_thermal": 2.0, "l2_thermal": 6.0, "j2_thermal": 12.0},
    }
    print_thermal_expectation_values(
        rho, 0.0, rot_to_spherical, bs, impurity_orbitals=impurity_orbitals, shell_casimir=shell_casimir
    )
    out = capsys.readouterr().out
    assert "Per-shell impurity observables:" in out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith(("l=1", "l=2", "total"))]
    assert len(lines) == 3
    l1_fields = lines[0].split()
    assert l1_fields[0] == "l=1"
    assert float(l1_fields[1]) == pytest.approx(6.0, abs=1e-9)  # N
    assert "1S0" in lines[0]  # filled p-shell -> singlet term
    l2_fields = lines[1].split()
    assert l2_fields[0] == "l=2"
    assert float(l2_fields[1]) == pytest.approx(8.0, abs=1e-9)  # N
    total_fields = lines[2].split()
    assert total_fields[0] == "total"
    assert float(total_fields[1]) == pytest.approx(14.0, abs=1e-9)  # N = 6 + 8
    # Total row has no Casimir/term columns (only 8 fields: "total" + 7 numbers).
    assert len(total_fields) == 8


def test_print_thermal_expectation_values_single_shell_has_no_per_shell_table(capsys):
    """A single-shell impurity (or impurity_orbitals=None) prints no per-shell table --
    unchanged from before this feature existed."""
    from impurityModel.ed.observables import print_thermal_expectation_values

    bs = _d_shell_block_structure()
    rho = np.eye(10, dtype=complex)
    print_thermal_expectation_values(rho, 0.0, np.eye(10, dtype=complex), bs)
    out = capsys.readouterr().out
    assert "Per-shell impurity observables" not in out


# --------------------------------------------------------------------------- #
# is_j_sharp / gated_term_symbol / term_symbol(j=None) (Stage 5: J-sharpness gate)
# --------------------------------------------------------------------------- #
def test_term_symbol_without_j():
    """term_symbol(j=None) omits the J subscript but keeps the S/L cleanliness check
    and the existing 3-positional-argument call sites unaffected."""
    from impurityModel.ed.observables import term_symbol

    assert term_symbol(1.0, 3.0, 4.0) == "3F4"  # unchanged 3-arg behaviour
    assert term_symbol(1.0, 3.0) == "3F"  # j omitted
    assert term_symbol(1.0, 3.0, None) == "3F"  # j explicitly None
    assert term_symbol(0.8, 3.0, None) == "~3F"  # unclean S -> still marked approximate


def test_is_j_sharp_identical_j_in_ground_manifold():
    """A ground manifold where every state shares the same J is sharp."""
    from impurityModel.ed.observables import is_j_sharp

    es = np.array([0.0, 0.0, 0.0, 1.0])
    j_values = np.array([2.5, 2.5, 2.5, 9.0])  # excited state's J is irrelevant
    assert is_j_sharp(j_values, es)


def test_is_j_sharp_spread_in_ground_manifold():
    """A ground manifold whose members' J genuinely differ (the user's own observation:
    J ranging ~3.09-3.35 within one degenerate manifold, expected physics without SOC
    sharpening J) is reported as not sharp."""
    from impurityModel.ed.observables import is_j_sharp

    es = np.array([0.0, 0.0, 0.0])
    j_values = np.array([3.09, 3.22, 3.35])
    assert not is_j_sharp(j_values, es)


def test_is_j_sharp_ignores_excited_states():
    """Only the ground (lowest-energy) manifold is examined -- a spread among excited
    states must not gate the ground-state term symbol."""
    from impurityModel.ed.observables import is_j_sharp

    es = np.array([0.0, 0.0, 1.0, 1.0])
    j_values = np.array([2.5, 2.5, 3.09, 3.35])  # excited manifold: spread, ground: sharp
    assert is_j_sharp(j_values, es)


def test_gated_term_symbol_suppresses_g_j_and_mu_eff_when_not_sharp():
    """When j_sharp=False, g_J/mu_eff are None and the term omits the J subscript; when
    j_sharp=True, behaviour is identical to the pre-Stage-5 unconditional computation."""
    from impurityModel.ed.observables import gated_term_symbol, lande_g_and_moments, term_symbol

    s2, l2, j2 = 2.0, 12.0, 12.0
    term_sharp, g_j_sharp, mu_eff_sharp = gated_term_symbol(s2, l2, j2, True)
    expected_g_j, expected_mu_eff, _ = lande_g_and_moments(s2, l2, j2)
    assert term_sharp == term_symbol(1.0, 3.0, 3.0)
    assert g_j_sharp == pytest.approx(expected_g_j)
    assert mu_eff_sharp == pytest.approx(expected_mu_eff)

    term_unsharp, g_j_unsharp, mu_eff_unsharp = gated_term_symbol(s2, l2, j2, False)
    assert term_unsharp == term_symbol(1.0, 3.0, None)
    assert "4" not in term_unsharp  # no J subscript
    assert g_j_unsharp is None
    assert mu_eff_unsharp is None


def test_print_thermal_expectation_values_j_sharp_gate(capsys):
    """print_thermal_expectation_values(j_sharp=False) omits g_J/mu_eff from the scalar
    magnetism block and from the per-shell table, and both use the same (ungated-J)
    term -- the desync the Stage 4 review flagged must not happen."""
    from impurityModel.ed.observables import print_thermal_expectation_values
    from impurityModel.ed.block_structure import BlockStructure

    bs = BlockStructure(
        blocks=[list(range(6)), list(range(6, 16))],
        identical_blocks=[[0], [1]],
        transposed_blocks=[[], []],
        particle_hole_blocks=[[], []],
        particle_hole_transposed_blocks=[[], []],
        inequivalent_blocks=[0, 1],
    )
    impurity_orbitals = {1: [list(range(6))], 2: [list(range(6, 16))]}
    rot_to_spherical = {1: np.eye(6, dtype=complex), 2: np.eye(10, dtype=complex)}
    rho = np.diag([1.0] * 6 + [0.8] * 10).astype(complex)
    shell_casimir = {
        1: {"l": 1, "s2_thermal": 0.0, "l2_thermal": 0.0, "j2_thermal": 0.0},
        2: {"l": 2, "s2_thermal": 2.0, "l2_thermal": 12.0, "j2_thermal": 12.0},
    }
    print_thermal_expectation_values(
        rho,
        0.0,
        rot_to_spherical,
        bs,
        s_thermal=2.0,
        l_thermal=12.0,
        j_thermal=12.0,
        impurity_orbitals=impurity_orbitals,
        shell_casimir=shell_casimir,
        j_sharp=False,
    )
    out = capsys.readouterr().out
    assert "g_J" not in out
    assert "mu_eff" not in out
    assert "mu_spin_only" in out  # unaffected by the gate
    # Scalar term (S=1,L=3 -> "3F") and the per-shell l=2 term must agree -- both ungated.
    term_lines = [ln for ln in out.splitlines() if ln.lstrip().startswith("term ")]
    assert len(term_lines) == 1
    assert "3F" in term_lines[0] and "4" not in term_lines[0].split("=")[1]
    shell_line = next(ln for ln in out.splitlines() if ln.strip().startswith("l=2"))
    assert shell_line.strip().split()[-1] == "3F"  # per-shell term, no J subscript
