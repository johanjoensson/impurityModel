"""Tests for symmetry-related ground-state observables (symmetry plan, Phase 1)."""

import numpy as np

from impurityModel.ed.operator_algebra import addOps, c2i
from impurityModel.ed.observables import (
    apply_casimir,
    casimir_to_quantum_number,
    expect_casimir,
    get_LS_from_rho_spherical,
    make_spin_operators,
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
    for label in ("<E-E0>", "<N>", "<N(Dn)>", "<N(Up)>", "<Lz>", "<Sz>"):
        assert label in out
    ls_line = next(line for line in out.splitlines() if line.startswith("<L.S>"))
    assert np.isclose(float(ls_line.split("=")[1]), 1.0, atol=1e-6)


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
    s2_line = next(line for line in out.splitlines() if line.startswith("<S^2>"))
    assert np.isclose(float(s2_line.split("=")[1].split()[0]), 2.0, atol=1e-6)
    assert "S =  1.0000" in s2_line  # S(S+1)=2 -> S=1


def test_impurity_casimir_operators_rotated():
    """make_impurity_casimir_operators gives correct, rotation-invariant <L^2>,<S^2>,<J^2>.

    Stretched single d-electron |ml=2, up> = |j=5/2, mj=5/2>: L^2=6, S^2=3/4, J^2=35/4.
    """
    from impurityModel.ed.observables import make_impurity_casimir_operators, expect_casimir

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
    l_line = next(line for line in out.splitlines() if line.startswith("<L^2>"))
    assert "L =  2.0000" in l_line  # L(L+1)=6 -> L=2
    j_line = next(line for line in out.splitlines() if line.startswith("<J^2>"))
    assert "J =  2.5000" in j_line  # J(J+1)=35/4 -> J=5/2


def test_bath_spin_pairs_and_consistency():
    """bath_spin_pairs + spin_pairs_consistent_with_h validate/skip the spin assignment."""
    from impurityModel.ed.spin_pairs import (
        bath_spin_pairs,
        impurity_spin_pairs,
        spin_pairs_consistent_with_h,
    )
    from impurityModel.ed.observables import expect_spin_correlation, make_spin_operators
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

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
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

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
    assert any(line.startswith("<S_imp.S_bath>") for line in out.splitlines())
    header = next(line for line in out.splitlines() if "E-E0" in line and "Sz" in line)
    assert "Si.Sb" in header


def _hop(diag, hops):
    """Hermitian one-body ManyBodyOperator from on-site energies and hopping triples."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

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
        derive_spin_pairs,
        spin_pairs_consistent_with_h,
        _impurity_pairs_per_partition,
        _impurity_pairs_whole_shell,
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


def _thermal_sisb(out):
    """Parse the '<S_imp.S_bath> = value' thermal line from calc_gs output."""
    line = next(ln for ln in out.splitlines() if ln.startswith("<S_imp.S_bath>"))
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
    from impurityModel.ed.observables import make_impurity_casimir_operators
    from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant, inner
    from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix
    from impurityModel.ed.observables import apply_casimir

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
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
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
    Hop = ManyBodyOperator(addOps([h0, u_dict]))

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
        mixed_valence={g: 1 for g in impurity_orbitals},
        tau=0.01,
        dense_cutoff=4000,
        spin_flip_dj=True,
        comm=None,
        truncation_threshold=200000,
    )
    calc_gs(Hop, setup, bs, Rot.conj().T, verbose=True, slaterWeightMin=1e-12)  # must not raise
    out = capsys.readouterr().out
    assert "<S^2>" in out and "<L^2>" in out and "<J^2>" in out  # reported, not skipped
    s2 = float(next(ln for ln in out.splitlines() if ln.startswith("<S^2>")).split("=")[1].split("(")[0])
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
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

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
