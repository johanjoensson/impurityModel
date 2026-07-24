"""Tests for the unrestricted Hartree-Fock occupation seed (``ed/hartree_fock.py``).

The decisive test pins the two-body mean-field convention: for a single Slater determinant
the mean-field energy ``E[rho]`` must equal the exact operator expectation ``<D|H|D>``.
"""

import numpy as np
import pytest

from impurityModel.ed import hartree_fock as hf
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, inner
from impurityModel.ed.product_state_representation import bytes2bitarray  # noqa: F401  (parity w/ codebase)


def _det(occupied, n_orbs):
    """Single-determinant ManyBodyState with the given occupied orbitals (MSB-first bits)."""
    from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

    data = bytearray((n_orbs + 7) // 8)
    for orb in occupied:
        data[orb // 8] |= 1 << (7 - orb % 8)
    return ManyBodyState({SlaterDeterminant.from_bytes(bytes(data)): 1.0})


def _diag_rho(occupied, n_orb):
    rho = np.zeros((n_orb, n_orb), dtype=complex)
    for o in occupied:
        rho[o, o] = 1.0
    return rho


def _hermitian_hamiltonian():
    """A small Hermitian 0/1/2-body operator on 4 orbitals (all treated as impurity)."""
    n_orb = 4
    terms = {}
    # constant
    terms[()] = 0.37
    # one-body: diagonal levels + a Hermitian hopping pair (0<->2)
    for i, eps in enumerate([-1.1, -0.4, 0.6, 1.3]):
        terms[((i, "c"), (i, "a"))] = eps
    terms[((0, "c"), (2, "a"))] = 0.25
    terms[((2, "c"), (0, "a"))] = 0.25  # conjugate (real)

    # two-body: density-density U n_i n_j = U c†_i c†_j c_j c_i  ->  ((i,c),(j,c),(j,a),(i,a))
    def dens_dens(i, j, u):
        terms[((i, "c"), (j, "c"), (j, "a"), (i, "a"))] = terms.get(((i, "c"), (j, "c"), (j, "a"), (i, "a")), 0) + u

    dens_dens(0, 1, 2.0)
    dens_dens(0, 3, 1.5)
    dens_dens(1, 2, 1.7)
    # an exchange-type term c†_0 c†_1 c_0 c_1 (contributes to off-diagonal of the energy)
    terms[((0, "c"), (1, "c"), (0, "a"), (1, "a"))] = terms.get(((0, "c"), (1, "c"), (0, "a"), (1, "a")), 0) + 0.6
    return ManyBodyOperator(terms), n_orb


@pytest.mark.parametrize("occupied", [[], [0], [0, 1], [0, 1, 2], [1, 3], [0, 1, 2, 3]])
def test_mean_field_energy_matches_exact_expectation(occupied):
    """E[rho] from the tensors == <D|H|D> from the operator, for a single determinant."""
    h_op, n_orb = _hermitian_hamiltonian()
    h, V, imp, got_n_orb, const = hf.extract_hf_tensors(h_op, range(n_orb))
    assert got_n_orb == n_orb

    rho = _diag_rho(occupied, n_orb)
    e_mf = hf.mean_field_energy(h, V, imp, rho) + float(np.real(const))

    D = _det(occupied, n_orb)
    e_exact = float(np.real(inner(D, h_op(D, 0))))

    assert e_mf == pytest.approx(e_exact, abs=1e-10)


def test_non_interacting_limit_fills_lowest_orbitals():
    """With V=0, HF reduces to filling the lowest one-body eigenstates (occupations 0/1)."""
    n_orb = 5
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n_orb, n_orb)) + 1j * rng.standard_normal((n_orb, n_orb))
    h = a + a.conj().T
    V = np.zeros((0, 0, 0, 0), dtype=complex)
    imp = []  # no interacting orbitals
    n_tot = 2

    rho, converged, energy = hf.hartree_fock_density_matrix(h, V, imp, n_tot)
    assert converged

    w, u = np.linalg.eigh(h)
    # physical density <c†_i c_j> = conj of the projector U_occ U_occ†
    rho_ref = u[:, :n_tot].conj() @ u[:, :n_tot].T
    assert np.allclose(rho, rho_ref, atol=1e-8)
    assert energy == pytest.approx(float(np.real(np.sum(w[:n_tot]))), abs=1e-8)
    # occupations sum to the particle number
    assert np.real(np.trace(rho)) == pytest.approx(n_tot, abs=1e-8)


def test_strong_U_localises_impurity_occupation():
    """A deep impurity level with large U pins the impurity near half-filling (one electron).

    Single impurity orbital (index 0) coupled to one bath orbital (index 1). With the
    impurity level well below the bath and a large U, HF keeps ~1 electron on the impurity.
    """
    terms = {
        ((0, "c"), (0, "a")): -3.0,  # deep impurity level
        ((1, "c"), (1, "a")): 0.0,  # bath at zero
        ((0, "c"), (1, "a")): 0.4,  # hybridisation
        ((1, "c"), (0, "a")): 0.4,
        ((0, "c"), (0, "c"), (0, "a"), (0, "a")): 0.0,  # (single orbital: no same-orbital U here)
    }
    h_op = ManyBodyOperator(terms)
    impurity_orbitals = {0: [[0]]}
    bath_states = ({0: [[1]]}, {0: [[]]})  # one valence bath, no conduction
    N0 = {0: 1}

    winning, _energy, converged = hf.hartree_fock_occupation(h_op, impurity_orbitals, bath_states, N0)
    assert converged
    # total electrons = N0 (1) + valence baths (1) = 2; impurity keeps close to one.
    assert winning[0] in (1,)


def test_two_body_outside_impurity_raises():
    terms = {((0, "c"), (1, "c"), (1, "a"), (0, "a")): 1.0}
    with pytest.raises(ValueError):
        hf.extract_hf_tensors(ManyBodyOperator(terms), impurity_indices=[0])  # term touches orb 1


# --------------------------------------------------------------------------- #
# Constrained HF: keeping a frozen shell frozen
# --------------------------------------------------------------------------- #


def _core_valence_model():
    """A miniature of the NiO L-edge pathology: a 'core' that HF wants to ionize.

    Orbitals 0,1 = core (a shell with no bath, hence frozen, nominally full); 2..7 = valence
    (6 slots, nominally holding only 2); 8,9 = bath. Two ingredients make the core drain, and both
    are present in the real problem:

    * the core level sits *above* the valence -- what the MLFT double counting does to NiO, where
      the bare 3d (-106 eV) ends up below the 2p core (-74.5 eV); and
    * the valence has **empty slots below the core** for those electrons to fall into (NiO: a d8
      shell has two).

    Unconstrained HF duly empties the core into them. The core hybridizes with nothing (a frozen
    shell has no baths), which is what makes the block-diagonal constrained density an exact fixed
    point rather than an approximation.
    """
    e_core, e_val, e_bath, v, u, u_cv = -1.0, -8.0, -0.5, 0.3, 2.0, 1.5
    valence = list(range(2, 8))
    terms = {}
    for o in (0, 1):
        terms[((o, "c"), (o, "a"))] = e_core
    for o in valence:
        terms[((o, "c"), (o, "a"))] = e_val
    for o in (8, 9):
        terms[((o, "c"), (o, "a"))] = e_bath
    for a, b in ((2, 8), (3, 9)):  # valence hybridizes with the bath; the core with nothing
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    for a in valence:
        for b in valence:
            if a < b:
                terms[((a, "c"), (b, "c"), (b, "a"), (a, "a"))] = u
    for c in (0, 1):  # core-valence density-density repulsion (the Fpd analogue)
        for w in valence:
            terms[((c, "c"), (w, "c"), (w, "a"), (c, "a"))] = u_cv
    impurity_orbitals = {0: [[0, 1]], 1: [valence]}  # set 0 = core (frozen), set 1 = valence
    bath_states = ({0: [[]], 1: [[8, 9]]}, {0: [[]], 1: [[]]})
    return ManyBodyOperator(terms), impurity_orbitals, bath_states, {0: 2, 1: 2}


def test_unconstrained_hf_drains_the_core():
    """The bug this constraint exists for: with nothing pinning it, HF empties the frozen shell."""
    h_op, impurity_orbitals, bath_states, N0 = _core_valence_model()
    occ, _energy, _converged = hf.hartree_fock_occupation(h_op, impurity_orbitals, bath_states, N0)
    assert occ[0] < N0[0], "expected unconstrained HF to ionize the core -- the model no longer bites"


def test_constrained_hf_keeps_the_frozen_shell_exactly_full():
    """Pinning the core holds it at N0 *exactly* (not merely on rounding), and HF still converges.

    The valence is left free, so this also checks the constraint does not freeze the whole problem:
    HF still redistributes charge between valence and bath at the fixed total.
    """
    h_op, impurity_orbitals, bath_states, N0 = _core_valence_model()
    occ, _energy, converged = hf.hartree_fock_occupation(
        h_op, impurity_orbitals, bath_states, N0, frozen_occupations={0}
    )
    assert converged
    assert occ[0] == N0[0]

    # exact, not rounded: the core block of rho carries precisely N0[0] electrons
    h, V, imp, _n_orb, _c = hf.extract_hf_tensors(h_op, list(range(8)))
    rho, _conv, _e = hf.hartree_fock_density_matrix(h, V, imp, n_tot=6, constraints=[([0, 1], 2)])
    assert np.real(np.trace(rho)) == pytest.approx(6.0, abs=1e-10)
    assert float(np.real(np.diag(rho)[[0, 1]].sum())) == pytest.approx(2.0, abs=1e-12)


def test_constrained_hf_costs_energy_and_that_is_the_point():
    """The constrained solution is *higher* in mean-field energy than the unconstrained one.

    The unconstrained minimum really is lower -- it just gets there through a state the calculation
    has declared unphysical (an ionized core). Pinned here so nobody 'optimizes' the constraint away
    on the grounds that it raises the energy.
    """
    h_op, impurity_orbitals, bath_states, N0 = _core_valence_model()
    _, e_free, _ = hf.hartree_fock_occupation(h_op, impurity_orbitals, bath_states, N0)
    _, e_pinned, _ = hf.hartree_fock_occupation(h_op, impurity_orbitals, bath_states, N0, frozen_occupations={0})
    assert e_pinned > e_free


def test_unconstrained_path_is_bit_for_bit_unchanged():
    """constraints=None (and the empty list) must reduce exactly to the global aufbau."""
    h_op, _io, _bs, _N0 = _core_valence_model()
    h, V, imp, _n_orb, _c = hf.extract_hf_tensors(h_op, list(range(8)))
    ref, conv_ref, e_ref = hf.hartree_fock_density_matrix(h, V, imp, n_tot=6)
    for constraints in (None, []):
        rho, conv, e = hf.hartree_fock_density_matrix(h, V, imp, n_tot=6, constraints=constraints)
        assert np.array_equal(rho, ref)  # bit-for-bit, not allclose
        assert conv == conv_ref and e == e_ref


def test_constraint_validation():
    h_op, _io, _bs, _N0 = _core_valence_model()
    h, V, imp, _n_orb, _c = hf.extract_hf_tensors(h_op, list(range(8)))
    with pytest.raises(ValueError, match="overlap"):
        hf.hartree_fock_density_matrix(h, V, imp, 6, constraints=[([0, 1], 2), ([1, 2], 1)])
    with pytest.raises(ValueError, match="cannot hold"):
        hf.hartree_fock_density_matrix(h, V, imp, 6, constraints=[([0, 1], 3)])
    with pytest.raises(ValueError, match="unconstrained orbitals"):
        hf.hartree_fock_density_matrix(h, V, imp, 2, constraints=[([0, 1], 2), ([2, 3], 2)])


def test_classify_orbitals_basic():
    rho = np.diag([1.0, 1.0, 0.5, 0.5, 0.0, 0.0]).astype(complex)  # sum = 3
    filled, empty, partial, active_electrons, n_tot = hf.classify_orbitals(rho, eps=0.05)
    assert filled == [0, 1]
    assert empty == [4, 5]
    assert partial == [2, 3]
    assert n_tot == 3
    assert active_electrons == 1  # n_tot - #filled
    # the invariant the seed relies on: filled electrons + active electrons == total
    assert len(filled) + active_electrons == n_tot


def test_classify_orbitals_eps_controls_active_space():
    """A near-integer orbital (0.97) is 'filled' at eps=0.05 but 'partial' at eps=0.01.

    Smaller eps => larger (more generous) active space, the safe direction.
    """
    rho = np.diag([0.97, 0.03, 0.5]).astype(complex)
    _, _, partial_loose, _, _ = hf.classify_orbitals(rho, eps=0.05)
    _, _, partial_tight, _, _ = hf.classify_orbitals(rho, eps=0.01)
    assert partial_loose == [2]
    assert set(partial_tight) == {0, 1, 2}  # 0.97 and 0.03 pulled into the active space


def test_hf_active_space_strong_U():
    """hf_active_space returns a self-consistent classification on the deep-level + bath model."""
    terms = {
        ((0, "c"), (0, "a")): -3.0,
        ((1, "c"), (1, "a")): 0.0,
        ((0, "c"), (1, "a")): 0.4,
        ((1, "c"), (0, "a")): 0.4,
    }
    h_op = ManyBodyOperator(terms)
    impurity_orbitals = {0: [[0]]}
    bath_states = ({0: [[1]]}, {0: [[]]})
    N0 = {0: 1}
    filled, empty, partial, active_electrons, n_tot, _rho, converged, _energy = hf.hf_active_space(
        h_op, impurity_orbitals, bath_states, N0, eps=0.05
    )
    assert converged
    assert n_tot == 2  # N0(1) + valence bath(1)
    # partition is a partition of all orbitals; electron bookkeeping holds
    assert sorted(filled + empty + partial) == [0, 1]
    assert 0 <= active_electrons <= len(partial)
    assert len(filled) + active_electrons == n_tot


def _occ_set(b, n_orb):
    return frozenset(i for i in range(n_orb) if b[i // 8] & (1 << (7 - i % 8)))


def test_build_cas_seed():
    seeds = hf.build_cas_seed(filled_idx=[0, 1], partial_idx=[2, 3, 4], active_electrons=1, num_spin_orbitals=6)
    assert len(seeds) == 3  # C(3,1)
    occs = {_occ_set(s, 6) for s in seeds}
    assert occs == {frozenset({0, 1, 2}), frozenset({0, 1, 3}), frozenset({0, 1, 4})}
    # filled always occupied; exactly one active electron each => 3 electrons total
    assert all(len(o) == 3 for o in occs)


def test_build_cas_seed_edge_cases():
    assert len(hf.build_cas_seed([0, 1], [2, 3], 0, 4)) == 1  # single reference (filled only)
    assert _occ_set(hf.build_cas_seed([0, 1], [2, 3], 0, 4)[0], 4) == frozenset({0, 1})
    assert len(hf.build_cas_seed([0], [1, 2], 2, 4)) == 1  # all partial filled
    with pytest.raises(ValueError):
        hf.build_cas_seed([0], [1, 2], 3, 4)  # more active electrons than partial orbitals


def test_build_cas_seed_operator_compatible():
    """Seed determinants are accepted by Basis(initial_basis=...) and match its occupation set."""
    from impurityModel.ed.manybody_basis import Basis

    seeds = hf.build_cas_seed(filled_idx=[2, 3], partial_idx=[0, 1], active_electrons=1, num_spin_orbitals=6)
    basis = Basis(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        initial_basis=seeds,
        verbose=False,
    )
    assert len(basis) == 2  # C(2,1)
    got = {_occ_set(bytes(s.to_bytearray()), 6) for s in basis.local_basis}
    assert got == {frozenset({2, 3, 0}), frozenset({2, 3, 1})}
