"""Tests for the symmetry-discovery engine (symmetry plan, Phase 2)."""

import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import extract_tensors, tensors_to_operator


def test_tensor_extraction():
    """Round-trip: operator -> (h, V, const) -> operator reproduces every term."""
    n_orb = 4
    # A mix of one-body hopping (incl. complex, off-diagonal) and two-body Coulomb.
    terms = {
        ((0, "c"), (0, "a")): 0.5,
        ((0, "c"), (1, "a")): 0.3 + 0.2j,
        ((1, "c"), (0, "a")): 0.3 - 0.2j,
        ((2, "c"), (3, "a")): -1.1,
        # Coulomb-style c†_0 c†_1 c_1 c_0 and c†_2 c†_3 c_3 c_2
        ((0, "c"), (1, "c"), (1, "a"), (0, "a")): 1.7,
        ((2, "c"), (3, "c"), (3, "a"), (2, "a")): 0.9 + 0.1j,
        (): 2.5,  # constant
    }
    op = ManyBodyOperator(dict(terms))

    h, V, const = extract_tensors(op, n_orb=n_orb)

    # Spot-check the tensor entries against the known terms.
    assert np.isclose(h[0, 1], 0.3 + 0.2j)
    assert np.isclose(h[2, 3], -1.1)
    # The operator stores terms in canonical normal order, which sorts the two
    # annihilators ascending: c†_0 c†_1 c_1 c_0 is stored as -c†_0 c†_1 c_0 c_1. So the
    # tensor entry lands on V[i,j,k,l] with i=0, j=1, l=0, k=1 -> V[0,1,1,0] = -1.7.
    # Same operator, antisymmetric-conjugate representative: V[0,1,0,1] is now empty.
    assert np.isclose(V[0, 1, 1, 0], -1.7)
    assert np.isclose(V[0, 1, 0, 1], 0.0)
    assert np.isclose(const, 2.5)

    # Full round-trip.
    op2 = tensors_to_operator(h, V, const)
    d1 = op.to_dict()
    d2 = op2.to_dict()
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        assert np.isclose(complex(d1[key]), complex(d2[key])), key


def test_tensor_extraction_infers_n_orb():
    """n_orb defaults to max index + 1."""
    op = ManyBodyOperator({((0, "c"), (5, "a")): 1.0})
    h, _V, _ = extract_tensors(op)
    assert h.shape == (6, 6)
    assert np.isclose(h[0, 5], 1.0)


def test_tensor_extraction_rejects_3body():
    """A 3-body term raises rather than being silently dropped."""
    op = ManyBodyOperator({((0, "c"), (1, "c"), (2, "c"), (2, "a"), (1, "a"), (0, "a")): 1.0})
    with pytest.raises(ValueError, match="1- and"):
        extract_tensors(op, n_orb=3)


def test_tensor_extraction_rejects_non_number_conserving():
    """A bare creation operator (non-number-conserving) raises."""
    op = ManyBodyOperator({((0, "c"),): 1.0})
    with pytest.raises(ValueError):
        extract_tensors(op, n_orb=2)


# ---------------------------------------------------------------------------
# Phase 2.2: null-space symmetry discovery
# ---------------------------------------------------------------------------

# SU(2)-symmetric Hubbard dimer, 4 spin-orbitals.
# Layout: orbital = spin*2 + site, so up = {0,1}, down = {2,3}.
T_HOP = 0.7
_HSITE = np.array([[0.0, -T_HOP], [-T_HOP, 0.0]], dtype=complex)
H_DIMER = np.kron(np.eye(2), _HSITE)  # spin-diagonal hopping = I_spin (x) h_site

_I2 = np.eye(2)
N_GEN = np.eye(4, dtype=complex)
SZ_GEN = np.kron(np.array([[0.5, 0], [0, -0.5]], dtype=complex), _I2)
SX_GEN = np.kron(np.array([[0, 0.5], [0.5, 0]], dtype=complex), _I2)
SY_GEN = np.kron(np.array([[0, -0.5j], [0.5j, 0]], dtype=complex), _I2)


def test_symmetry_null_space():
    """The Hubbard-dimer commutant contains N, S_z, S_x, S_y (all one-body)."""
    from impurityModel.ed.symmetries import discover_one_body_symmetries, in_span

    gens = discover_one_body_symmetries(H_DIMER)

    # Every discovered generator actually commutes with h (convention-proof).
    for g in gens:
        assert np.linalg.norm(H_DIMER @ g - g @ H_DIMER) < 1e-10

    # N, S_z, S_x, S_y are all in the discovered algebra.
    for gen in (N_GEN, SZ_GEN, SX_GEN, SY_GEN):
        assert in_span(gens, gen)

    # h = I_spin (x) h_site with h_site non-degenerate => commutant is
    # span{I_site, h_site} (x) M_2(spin) = 2 * 4 = 8 dimensional.
    assert len(gens) == 8

    # S^2 is two-body: it cannot be one of these n x n one-body generators. Concretely,
    # a single-particle generator promoted back to an operator is purely one-body, so
    # S^2 (which needs a nonzero two-body V) is absent from the null space by
    # construction (see Phase 2.1 / extract_tensors).


def test_null_space_threshold_stability():
    """Discovered count is stable across cutoffs spanning orders of magnitude."""
    from impurityModel.ed.symmetries import discover_one_body_symmetries

    counts = {cut: len(discover_one_body_symmetries(H_DIMER, sigma_cut=cut)) for cut in (1e-10, 1e-8, 1e-6, 1e-4)}
    assert set(counts.values()) == {8}, counts


# ---------------------------------------------------------------------------
# Phase 2.3: Cartan subalgebra & joint diagonalization
# ---------------------------------------------------------------------------


def test_cartan_subalgebra_and_joint_diagonalization():
    """Cartan generators mutually commute and are simultaneously diagonalised."""
    from impurityModel.ed.symmetries import (
        cartan_subalgebra,
        discover_one_body_symmetries,
        joint_diagonalize,
    )

    gens = discover_one_body_symmetries(H_DIMER)
    cartan = cartan_subalgebra(gens)

    # u(2) + u(2) has rank 4 -> Cartan is 4-dimensional.
    assert len(cartan) == 4

    # All Cartan generators commute pairwise and with h.
    for a in range(len(cartan)):
        assert np.linalg.norm(H_DIMER @ cartan[a] - cartan[a] @ H_DIMER) < 1e-9
        for b in range(len(cartan)):
            assert np.linalg.norm(cartan[a] @ cartan[b] - cartan[b] @ cartan[a]) < 1e-9

    # Joint diagonalization: each generator is diagonal in the common basis U.
    u, diagonals = joint_diagonalize(cartan)
    for o, d in zip(cartan, diagonals):
        transformed = u.conj().T @ o @ u
        off = transformed - np.diag(np.diag(transformed))
        assert np.linalg.norm(off) < 1e-9
        assert np.allclose(np.diag(transformed), d, atol=1e-9)


def test_joint_diagonalization_degenerate():
    """Random-combination method handles a degenerate generator (S_z: +/-1/2 x2)."""
    from impurityModel.ed.symmetries import joint_diagonalize

    # S_z on the dimer is degenerate (eigenvalues +1/2,+1/2,-1/2,-1/2); N and S_z
    # commute. A random combo r1*N + r2*S_z is non-degenerate and diagonalises both.
    u, (dN, dSz) = joint_diagonalize([N_GEN, SZ_GEN])
    for o in (N_GEN, SZ_GEN):
        t = u.conj().T @ o @ u
        assert np.linalg.norm(t - np.diag(np.diag(t))) < 1e-9
    assert np.allclose(np.sort(dSz), [-0.5, -0.5, 0.5, 0.5], atol=1e-9)
    assert np.allclose(dN, 1.0, atol=1e-9)


def test_generator_weight_classification():
    """N has {0,1} weights (-> restriction); S_z has +/-1/2 (-> Phase 6)."""
    from impurityModel.ed.symmetries import joint_diagonalize, weights_are_01

    _, (dN, dSz) = joint_diagonalize([N_GEN, SZ_GEN])
    assert weights_are_01(dN)  # N: eigenvalues all 1 -> subset occupation
    assert not weights_are_01(dSz)  # S_z: +/-1/2 -> needs weighted-sum (Phase 6)


# ---------------------------------------------------------------------------
# Acceptance gate: discovery on a REAL d-shell Hamiltonian (codebase operators)
# ---------------------------------------------------------------------------

from collections import OrderedDict  # noqa: E402

from impurityModel.ed.atomic_physics import getSOCop  # noqa: E402
from impurityModel.ed.operator_algebra import c2i  # noqa: E402

_L = 2
_NBATHS = OrderedDict({_L: 0})  # single d-shell impurity, no baths
_N_D = 2 * (2 * _L + 1)  # 10 spin-orbitals


def _idx(s, m):
    """c2i index of d spin-orbital (s, m): s in {0=down,1=up}, m in -2..2."""
    return c2i(_NBATHS, (_L, s, m))


def _soc_matrix(xi):
    """Real codebase SOC operator getSOCop(xi, l=2) as a 10x10 matrix in c2i order.

    Note: l.s is real in the spherical-harmonic basis; the complexity of a real
    calculation comes from the spherical->cubic rotation (see the cubic-basis test).
    """
    h = np.zeros((_N_D, _N_D), dtype=complex)
    for term, val in getSOCop(xi, _L).items():
        (_l1, s1, m1), _ = term[0]
        (_l2, s2, m2), _ = term[1]
        h[_idx(s1, m1), _idx(s2, m2)] += val
    return h


def _spherical_to_cubic(l=_L):
    """Spin-doubled spherical->cubic rotation in the c2i [down, up] x [ml] layout."""
    from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix

    u_orb = get_spherical_2_cubic_matrix(spinpol=False, l=l)
    return np.kron(np.eye(2), u_orb)  # spin is the outer c2i index


def _crystal_field():
    """Axial, spin-independent crystal field (diagonal in ml), lifts orbital degeneracy."""
    eps = {-2: 0.0, -1: 0.15, 0: 0.30, 1: 0.45, 2: 0.60}
    h = np.zeros((_N_D, _N_D), dtype=complex)
    for s in (0, 1):
        for m in range(-_L, _L + 1):
            h[_idx(s, m), _idx(s, m)] = eps[m]
    return h


def _spin_and_orbital_generators():
    """N, S_x, S_y, S_z, L_z, J_z as 10x10 matrices in the c2i basis."""
    N = np.eye(_N_D, dtype=complex)
    Sz = np.zeros((_N_D, _N_D), dtype=complex)
    Lz = np.zeros((_N_D, _N_D), dtype=complex)
    Sp = np.zeros((_N_D, _N_D), dtype=complex)  # S_+ = c†_up c_dn
    for m in range(-_L, _L + 1):
        Sz[_idx(0, m), _idx(0, m)] = -0.5
        Sz[_idx(1, m), _idx(1, m)] = 0.5
        Lz[_idx(0, m), _idx(0, m)] = m
        Lz[_idx(1, m), _idx(1, m)] = m
        Sp[_idx(1, m), _idx(0, m)] = 1.0
    Sm = Sp.conj().T
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)
    Jz = Lz + Sz
    return N, Sx, Sy, Sz, Lz, Jz


def test_discovery_real_dshell_no_soc_has_su2():
    """No SOC: spin-independent CF -> discovery recovers N, S_x, S_y, S_z, L_z."""
    from impurityModel.ed.symmetries import discover_one_body_symmetries, in_span

    h = _crystal_field()
    gens = discover_one_body_symmetries(h)
    for g in gens:
        assert np.linalg.norm(h @ g - g @ h) < 1e-9

    N, Sx, Sy, Sz, Lz, _Jz = _spin_and_orbital_generators()
    for gen in (N, Sx, Sy, Sz, Lz):
        assert in_span(gens, gen)


def test_discovery_real_dshell_soc_breaks_su2():
    """SOC on: spin SU(2) is broken -> N and J_z survive, S_z alone does not."""
    from impurityModel.ed.symmetries import discover_one_body_symmetries, in_span

    h = _crystal_field() + _soc_matrix(0.5)
    assert np.linalg.norm(h - h.conj().T) < 1e-12  # Hermitian

    gens = discover_one_body_symmetries(h)
    for g in gens:
        assert np.linalg.norm(h @ g - g @ h) < 1e-9

    N, Sx, _Sy, Sz, _Lz, Jz = _spin_and_orbital_generators()
    # Charge and total J_z conserved; individual spin / orbital rotations broken.
    assert in_span(gens, N)
    assert in_span(gens, Jz)
    assert np.linalg.norm(h @ Jz - Jz @ h) < 1e-9
    assert not in_span(gens, Sz)
    assert not in_span(gens, Sx)
    assert np.linalg.norm(h @ Sz - Sz @ h) > 1e-6  # S_z genuinely no longer commutes


def test_discovery_basis_invariant_cubic():
    """Discovery is basis-covariant: a complex cubic-basis H has the same symmetry count.

    H is genuinely complex here (the spherical->cubic rotation), exercising the
    discovery engine on the kind of complex Hamiltonian real calculations use.
    """
    from impurityModel.ed.symmetries import discover_one_body_symmetries, in_span

    h_sph = _crystal_field() + _soc_matrix(0.5)
    u = _spherical_to_cubic()
    h_cub = u.conj().T @ h_sph @ u
    assert np.linalg.norm(h_cub.imag) > 1e-6  # genuinely complex in the cubic basis

    n_sph = len(discover_one_body_symmetries(h_sph))
    gens_cub = discover_one_body_symmetries(h_cub)
    # Symmetry dimension is invariant under a single-particle basis change.
    assert len(gens_cub) == n_sph
    # N = identity is invariant under any basis, so it is found in the cubic basis too.
    assert in_span(gens_cub, np.eye(_N_D, dtype=complex))
    for g in gens_cub:
        assert np.linalg.norm(h_cub @ g - g @ h_cub) < 1e-9


# ---------------------------------------------------------------------------
# Formal acceptance gate: discovered orbital blocks match/refine hand-coded ones
# ---------------------------------------------------------------------------

_NB_PD = OrderedDict({1: 0, 2: 0})  # p (l=1) at indices 0..5, d (l=2) at 6..15


def _soc_terms(xi, l):
    d = {}
    for term, val in getSOCop(xi, l).items():
        (l1, s1, m1), _ = term[0]
        (l2, s2, m2), _ = term[1]
        d[((c2i(_NB_PD, (l1, s1, m1)), "c"), (c2i(_NB_PD, (l2, s2, m2)), "a"))] = val
    return d


def _shell_ml_chain(l):
    """Couple consecutive ml within each spin of shell l (connects all ml of a spin)."""
    d = {}
    for s in (0, 1):
        for m in range(-l, l):
            i = c2i(_NB_PD, (l, s, m))
            j = c2i(_NB_PD, (l, s, m + 1))
            d[((i, "c"), (j, "a"))] = 0.2
            d[((j, "c"), (i, "a"))] = 0.2
    return d


def test_acceptance_gate_discovery_refines_pd_block_structure():
    """SOC p+d model: discovered orbital blocks refine the hand-coded [p, d] structure
    (SOC conserves J_z, so each shell decouples into mj sub-blocks)."""
    from impurityModel.ed.block_structure import build_block_structure
    from impurityModel.ed.symmetries import (
        blocks_refine_or_match,
        discovered_orbital_blocks,
        extract_tensors,
    )

    terms = {}
    terms.update(_soc_terms(0.1, 1))  # p SOC
    terms.update(_soc_terms(0.5, 2))  # d SOC
    op = ManyBodyOperator(terms)

    discovered = discovered_orbital_blocks(op, 16)
    hand_coded = [list(range(6)), list(range(6, 16))]  # get_spectra.py p/d blocks

    # Matches or refines the hand-coded structure (never coarsens it).
    assert blocks_refine_or_match(discovered, hand_coded)
    # Strictly finer here: SOC splits each shell into J_z sectors.
    assert len(discovered) > len(hand_coded)
    # Each discovered block stays within one shell.
    for block in discovered:
        assert block <= frozenset(range(6)) or block <= frozenset(range(6, 16))

    # Cross-check: identical to the existing build_block_structure(mat=h) machinery.
    h, _, _ = extract_tensors(op, 16)
    bbs_blocks = build_block_structure(None, mat=h).blocks
    assert blocks_refine_or_match(discovered, bbs_blocks)
    assert blocks_refine_or_match(bbs_blocks, discovered)


def test_acceptance_gate_fully_coupled_shells_match_exactly():
    """With intra-shell ml coupling + SOC each shell is fully connected, so the
    discovered blocks equal the hand-coded [p, d] structure exactly."""
    from impurityModel.ed.symmetries import blocks_refine_or_match, discovered_orbital_blocks

    terms = {}
    terms.update(_soc_terms(0.1, 1))
    terms.update(_soc_terms(0.5, 2))
    terms.update(_shell_ml_chain(1))  # connect all ml within p
    terms.update(_shell_ml_chain(2))  # connect all ml within d
    op = ManyBodyOperator(terms)

    discovered = discovered_orbital_blocks(op, 16)
    hand_coded = [frozenset(range(6)), frozenset(range(6, 16))]

    # Exact match: same partition both ways.
    assert blocks_refine_or_match(discovered, hand_coded)
    assert blocks_refine_or_match(hand_coded, discovered)
    assert set(discovered) == set(hand_coded)


def test_acceptance_gate_refines_single_impurity_block():
    """selfenergy.py uses a single impurity block; any discovered partition refines it."""
    from impurityModel.ed.symmetries import blocks_refine_or_match, discovered_orbital_blocks

    terms = {}
    terms.update(_soc_terms(0.5, 2))  # d shell only
    terms.update(_shell_ml_chain(2))
    # Re-index d to 0..9 for a standalone impurity.
    op = ManyBodyOperator(terms)
    discovered = discovered_orbital_blocks(op, 16)
    single_block = [list(range(16))]
    assert blocks_refine_or_match(discovered, single_block)
