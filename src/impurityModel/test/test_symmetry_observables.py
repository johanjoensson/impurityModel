"""Tests for symmetry-related ground-state observables (symmetry plan, Phase 1)."""

import numpy as np

from impurityModel.ed.finite import (
    get_LS_from_rho_spherical,
    make_spin_operators,
    apply_casimir,
    expect_casimir,
    casimir_to_quantum_number,
    manifold_observable_values,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant


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
    from impurityModel.ed.finite import make_orbital_angular_momentum_operators

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

    def apply_S2_op(psi):
        return apply_casimir(psi, s_plus, s_minus, s_z)

    manifold = [
        _state([([1, 3], 1.0)]),  # triplet S_z=+1
        _state([([0, 2], 1.0)]),  # triplet S_z=-1
        _state([([1, 2], 1.0)]),  # mixes singlet/triplet S_z=0
        _state([([0, 3], 1.0)]),  # mixes singlet/triplet S_z=0
    ]
    energies = np.zeros(4)  # accidentally degenerate

    vals = manifold_observable_values(manifold, energies, apply_S2_op)
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
    from impurityModel.ed.finite import thermal_observable_value

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
    from impurityModel.ed.finite import expect_spin_correlation

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
    from impurityModel.ed.finite import print_expectation_values

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
    from impurityModel.ed.finite import print_thermal_expectation_values

    n = 10
    rot = np.eye(n)
    bs = _d_shell_block_structure()
    rho_one = np.zeros((n, n), dtype=complex)
    rho_one[9, 9] = 1.0  # |ml=2, up> -> <L.S> = 1.0

    print_thermal_expectation_values(rho_one, 0.0, rot, bs)
    out = capsys.readouterr().out
    for label in ("<E-E0>", "<N>", "<N(Dn)>", "<N(Up)>", "<Lz>", "<Sz>"):
        assert label in out
    ls_line = next(line for line in out.splitlines() if line.startswith("<L.S>"))
    assert np.isclose(float(ls_line.split("=")[1]), 1.0, atol=1e-6)


def test_print_expectation_values_S_column(capsys):
    """Passing s_values appends an 'S' column; omitting it preserves old output."""
    from impurityModel.ed.finite import print_expectation_values

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
    from impurityModel.ed.finite import print_thermal_expectation_values

    n = 10
    rot = np.eye(n)
    bs = _d_shell_block_structure()
    rho = np.eye(n, dtype=complex)

    print_thermal_expectation_values(rho, 0.0, rot, bs, s_thermal=2.0)
    out = capsys.readouterr().out
    s2_line = next(line for line in out.splitlines() if line.startswith("<S^2>"))
    assert np.isclose(float(s2_line.split("=")[1].split()[0]), 2.0, atol=1e-6)
    assert "S =  1.0000" in s2_line  # S(S+1)=2 -> S=1


def test_impurity_casimir_operators_rotated():
    """make_impurity_casimir_operators gives correct, rotation-invariant <L^2>,<S^2>,<J^2>.

    Stretched single d-electron |ml=2, up> = |j=5/2, mj=5/2>: L^2=6, S^2=3/4, J^2=35/4.
    """
    from impurityModel.ed.finite import make_impurity_casimir_operators, expect_casimir

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
    from impurityModel.ed.finite import print_expectation_values

    n = 10
    bs = _d_shell_block_structure()
    rho = np.eye(n, dtype=complex)
    es = np.array([0.0])
    print_expectation_values(
        np.array([rho]), es, np.eye(n), bs,
        s_values=np.array([1.0]), l_values=np.array([2.0]), j_values=np.array([2.5]),
    )
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if "E-E0" in line)
    assert header.split()[-3:] == ["S", "L", "J"]
    row = next(ln for ln in out.splitlines() if ln.strip().startswith("0"))
    s, l, j = (float(x) for x in row.split()[-3:])
    assert (s, l, j) == (1.0, 2.0, 2.5)


def test_print_thermal_LJ_lines(capsys):
    """Passing l_thermal / j_thermal adds <L^2> and <J^2> lines with quantum numbers."""
    from impurityModel.ed.finite import print_thermal_expectation_values

    n = 10
    bs = _d_shell_block_structure()
    print_thermal_expectation_values(
        np.eye(n, dtype=complex), 0.0, np.eye(n), bs,
        s_thermal=2.0, l_thermal=6.0, j_thermal=35.0 / 4,
    )
    out = capsys.readouterr().out
    assert "S = " in out and "<S^2>" in out
    l_line = next(line for line in out.splitlines() if line.startswith("<L^2>"))
    assert "L =  2.0000" in l_line  # L(L+1)=6 -> L=2
    j_line = next(line for line in out.splitlines() if line.startswith("<J^2>"))
    assert "J =  2.5000" in j_line  # J(J+1)=35/4 -> J=5/2
