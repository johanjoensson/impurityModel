"""Tests for symmetry-derived sectorization / restrictions (symmetry plan, Phase 3)."""

from itertools import combinations

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner
from impurityModel.ed.symmetries import (
    conserved_subset_charges,
    restrictions_from_charges,
    subset_occupations,
)


def _sd(occupied, n_orbs):
    n_bytes = (n_orbs + 7) // 8
    data = bytearray(n_bytes)
    for orb in occupied:
        data[orb // 8] |= 1 << (7 - (orb % 8))
    return SlaterDeterminant.from_bytes(bytes(data))


def _sector_basis(n_orb, n_elec):
    return [tuple(occ) for occ in combinations(range(n_orb), n_elec)]


def _dense_matrix(op, occs, n_orb):
    """<a|op|b> over an explicit list of occupied-orbital tuples."""
    states = [ManyBodyState({_sd(o, n_orb): 1.0}) for o in occs]
    n = len(states)
    mat = np.zeros((n, n), dtype=complex)
    for b, sb in enumerate(states):
        col = applyOp(op, sb)
        for a, sa in enumerate(states):
            mat[a, b] = inner(sa, col)
    return mat


def _hubbard_dimer():
    """0,1 = up sites; 2,3 = down sites. Spin-diagonal hopping + on-site U (density-density)."""
    t, u = 0.9, 2.3
    terms = {}
    for a, b in ((0, 1), (2, 3)):
        terms[((a, "c"), (b, "a"))] = -t
        terms[((b, "c"), (a, "a"))] = -t
    for up, dn in ((0, 2), (1, 3)):
        terms[((up, "c"), (dn, "c"), (dn, "a"), (up, "a"))] = u  # n_up n_dn
    return ManyBodyOperator(terms)


def test_conserved_charges_hubbard_separates_spin():
    """No spin mixing -> N_up and N_down are separately conserved subset charges."""
    op = _hubbard_dimer()
    charges = conserved_subset_charges(op, n_orb=4)
    assert charges == [frozenset({0, 1}), frozenset({2, 3})]


def _commutes_with_H(op, charge, n_orb):
    """True iff N_charge commutes with op on every fixed-N sector (block-diagonal)."""
    for n_elec in range(n_orb + 1):
        occs = _sector_basis(n_orb, n_elec)
        if not occs:
            continue
        h = _dense_matrix(op, occs, n_orb)
        n_s = np.diag([len(set(o) & charge) for o in occs]).astype(complex)
        if np.linalg.norm(h @ n_s - n_s @ h) > 1e-10:
            return False
    return True


def test_conserved_charges_actually_commute():
    """Each discovered subset charge commutes with the full many-body Hamiltonian."""
    op = _hubbard_dimer()
    charges = conserved_subset_charges(op, n_orb=4)
    for charge in charges:
        assert _commutes_with_H(op, charge, 4)


def test_one_body_spin_flip_merges_charges():
    """A one-body spin-flip term (SOC-like) couples up<->down -> only total N survives."""
    op = _hubbard_dimer()
    terms = op.to_dict()
    # add c†_0 c_2 + h.c. (up-site-0 <-> down-site-0): spin mixing
    terms[((0, "c"), (2, "a"))] = 0.4
    terms[((2, "c"), (0, "a"))] = 0.4
    op2 = ManyBodyOperator(terms)
    charges = conserved_subset_charges(op2, n_orb=4)
    assert charges == [frozenset({0, 1, 2, 3})]  # single charge = total N
    assert _commutes_with_H(op2, charges[0], 4)


def test_two_body_pair_hopping_merges_charges():
    """A pair-hopping term that moves 2 up <-> 2 down breaks N_up/N_down."""
    op = _hubbard_dimer()
    terms = op.to_dict()
    # c†_0 c†_1 c_2 c_3 : create two up, annihilate two down -> N_up not conserved
    terms[((0, "c"), (1, "c"), (2, "a"), (3, "a"))] = 0.3
    terms[((3, "c"), (2, "c"), (1, "a"), (0, "a"))] = 0.3  # h.c.
    op2 = ManyBodyOperator(terms)
    charges = conserved_subset_charges(op2, n_orb=4)
    assert charges == [frozenset({0, 1, 2, 3})]
    assert _commutes_with_H(op2, charges[0], 4)


def test_density_density_does_not_merge():
    """A genuine density-density cross term must NOT merge the spin charges."""
    op = _hubbard_dimer()
    terms = op.to_dict()
    # n_0 n_3 (up-site0 * down-site1) density-density: conserves both spin charges
    terms[((0, "c"), (3, "c"), (3, "a"), (0, "a"))] = 0.5
    op2 = ManyBodyOperator(terms)
    charges = conserved_subset_charges(op2, n_orb=4)
    assert charges == [frozenset({0, 1}), frozenset({2, 3})]


def test_restrictions_from_charges_and_occupations():
    """Mapping charges + a reference determinant to Basis.restrictions format."""
    op = _hubbard_dimer()
    charges = conserved_subset_charges(op, n_orb=4)
    # reference determinant: one electron per spin (orbital 0 up, orbital 2 down)
    occ = subset_occupations(charges, {0, 2})
    assert occ == [1, 1]

    restr = restrictions_from_charges(charges, occ)
    assert restr == {frozenset({0, 1}): (1, 1), frozenset({2, 3}): (1, 1)}

    restr_slack = restrictions_from_charges(charges, occ, slack=1)
    assert restr_slack == {frozenset({0, 1}): (0, 2), frozenset({2, 3}): (0, 2)}


# ---------------------------------------------------------------------------
# Phase 3.2: sectorized basis generation (CIPSI/Basis integration)
# ---------------------------------------------------------------------------

import impurityModel.ed.product_state_representation as psr  # noqa: E402
from impurityModel.ed.manybody_basis import Basis  # noqa: E402

_BASIS_KW = dict(impurity_orbitals={0: [[0, 1, 2, 3]]}, bath_states=({0: [[]]}, {0: [[]]}), verbose=False)


def _bytes(occ):
    b = bytearray(1)
    for o in occ:
        b[0] |= 1 << (7 - o)
    return bytes(b)


def _occset(det_bytes):
    return {k for k, bit in enumerate(psr.bytes2bitarray(det_bytes, 4)) if bit}


def _anderson_4():
    """4-orbital Anderson: 0=imp_up,1=imp_dn,2=bath_up,3=bath_dn. Spin-diagonal."""
    eps_i, eps_b, v, u = -1.0, 0.5, 0.7, 3.0
    terms = {
        ((0, "c"), (0, "a")): eps_i,
        ((1, "c"), (1, "a")): eps_i,
        ((2, "c"), (2, "a")): eps_b,
        ((3, "c"), (3, "a")): eps_b,
    }
    for a, b in ((0, 2), (1, 3)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u  # U n_imp_up n_imp_dn
    return ManyBodyOperator(terms)


def test_automatic_restrictions():
    """Auto-symmetry restrictions reproduce the unrestricted ground state, smaller basis."""
    op = _anderson_4()
    charges = conserved_subset_charges(op, n_orb=4)
    assert charges == [frozenset({0, 2}), frozenset({1, 3})]

    # Unrestricted: the full Fock space (all 16 determinants, every N sector).
    all_dets = [_bytes(occ) for ne in range(5) for occ in combinations(range(4), ne)]
    full = Basis(initial_basis=all_dets, **_BASIS_KW)
    h_full = np.asarray(full.build_dense_matrix(op))
    ev, evec = np.linalg.eigh(h_full)
    e0, gs = ev[0], evec[:, 0]

    # Read off the ground state's conserved-charge sector (charges are conserved, so
    # each <N_S> is integer for the non-degenerate GS).
    local = [bytes(d.to_bytearray()) for d in full.local_basis]
    gs_occ = [
        int(round(sum(abs(gs[i]) ** 2 * len(charge & _occset(local[i])) for i in range(len(local)))))
        for charge in charges
    ]
    assert gs_occ == [1, 1]  # half-filled singlet sector

    restrictions = restrictions_from_charges(charges, gs_occ)
    sector_dets = [
        d for d in all_dets if all(lo <= len(s & _occset(d)) <= hi for s, (lo, hi) in restrictions.items())
    ]
    restricted = Basis(initial_basis=sector_dets, restrictions=restrictions, **_BASIS_KW)
    e0_restricted = np.linalg.eigvalsh(np.asarray(restricted.build_dense_matrix(op)))[0]

    assert np.isclose(e0, e0_restricted, atol=1e-10)  # same ground-state energy
    assert len(sector_dets) < len(all_dets)  # 4 < 16 : smaller basis


def test_restrictions_refuse_out_of_sector_in_expand():
    """Restrictions actively prune: expand never generates out-of-sector determinants."""
    charges = [frozenset({0, 2}), frozenset({1, 3})]
    restrictions = restrictions_from_charges(charges, [1, 1])

    def in_sector(d):
        return all(lo <= len(s & _occset(d)) <= hi for s, (lo, hi) in restrictions.items())

    # Operator with an in-sector up-hop (0<->2) AND a spin-changing hop (1<->2) that
    # would leave the (N_up=1, N_dn=1) sector.
    cross = ManyBodyOperator(
        {
            ((2, "c"), (0, "a")): 1.0,
            ((0, "c"), (2, "a")): 1.0,
            ((2, "c"), (1, "a")): 1.0,
            ((1, "c"), (2, "a")): 1.0,
        }
    )
    seed = [_bytes((0, 1))]  # N_up=1, N_dn=1

    restricted = Basis(initial_basis=list(seed), restrictions=restrictions, **_BASIS_KW)
    restricted.expand(cross)
    assert all(in_sector(bytes(d.to_bytearray())) for d in restricted.local_basis)

    free = Basis(initial_basis=list(seed), restrictions=None, **_BASIS_KW)
    free.expand(cross)
    assert any(not in_sector(bytes(d.to_bytearray())) for d in free.local_basis)


# ---------------------------------------------------------------------------
# Phase 3.0: rough-CIPSI pre-scan -> identify GS sector -> refine
# ---------------------------------------------------------------------------


def test_measure_conserved_charges():
    """<N_S> on hand-built states rounds to the correct integer sector."""
    from impurityModel.ed.symmetries import measure_conserved_charges

    charges = [frozenset({0, 2}), frozenset({1, 3})]
    psi = ManyBodyState({_sd(o, 4): 0.5 for o in ([0, 1], [0, 3], [2, 1], [2, 3])})
    assert measure_conserved_charges(psi, charges, 4) == [1, 1]
    assert measure_conserved_charges(ManyBodyState({_sd([0, 2], 4): 1.0}), charges, 4) == [2, 0]


def _siam_6():
    """Single-impurity Anderson model, 6 spin-orbitals.

    0,1 = impurity (dn,up); 2,3 = valence bath (dn,up); 4,5 = conduction bath (dn,up).
    """
    ed, u, ev, ec, v = -1.0, 4.0, -3.0, 3.0, 0.5
    terms = {}
    for o in (0, 1):
        terms[((o, "c"), (o, "a"))] = ed
    for o in (2, 3):
        terms[((o, "c"), (o, "a"))] = ev
    for o in (4, 5):
        terms[((o, "c"), (o, "a"))] = ec
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u  # U n_imp_dn n_imp_up
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    return ManyBodyOperator(terms)


def test_prescan_finds_ground_state_sector():
    """Rough pre-scan finds the true GS energy/sector; restricting to it reproduces the
    ground state with a smaller basis."""
    from impurityModel.ed.groundstate import prescan_ground_state_sector

    op = _siam_6()
    impurity_orbitals = {0: [[0, 1]]}
    bath_states = ({0: [[2, 3]]}, {0: [[4, 5]]})

    # Ground truth: diagonalize the full Fock space (64 determinants).
    all_dets = [_bytes6(occ) for ne in range(7) for occ in combinations(range(6), ne)]
    full = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]}, bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=all_dets, verbose=False,
    )
    e_global = np.linalg.eigvalsh(np.asarray(full.build_dense_matrix(op)))[0]

    # Pre-scan: one rough solve over a broad window.
    winning_N0, restrictions, e_rough = prescan_ground_state_sector(
        op, impurity_orbitals, bath_states, {0: 2},
        mixed_valence={0: 1}, tau=0.01, chain_restrict=False, spin_flip_dj=False,
        dense_cutoff=1000, comm=None, truncation_threshold=100000, slaterWeightMin=1e-12,
    )
    assert np.isclose(e_rough, e_global, atol=1e-6)  # rough scan found the true GS

    # The identified sector has the right total electron number.
    total_n = sum(hi for (lo, hi) in restrictions.values())
    assert total_n == 3  # 2 (filled valence bath) + 1 (mixed-valence impurity)

    # Refine: restricting the full Fock space to the pre-scan sector reproduces the GS
    # with a strictly smaller basis.
    sector_dets = [
        d for d in all_dets if all(lo <= len(s & _occset6(d)) <= hi for s, (lo, hi) in restrictions.items())
    ]
    restricted = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]}, bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=sector_dets, restrictions=restrictions, verbose=False,
    )
    e_refined = np.linalg.eigvalsh(np.asarray(restricted.build_dense_matrix(op)))[0]
    assert np.isclose(e_refined, e_global, atol=1e-10)
    assert len(sector_dets) < len(all_dets)


def _bytes6(occ):
    b = bytearray(1)
    for o in occ:
        b[0] |= 1 << (7 - o)
    return bytes(b)


def _occset6(det_bytes):
    return {k for k, bit in enumerate(psr.bytes2bitarray(det_bytes, 6)) if bit}


def test_prescan_matches_brute_force_loop():
    """find_ground_state_basis with use_prescan=True matches the O(3^k) dN scan."""
    from impurityModel.ed.groundstate import find_ground_state_basis

    op = _siam_6()
    kw = dict(
        impurity_orbitals={0: [[0, 1]]}, bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}), N0={0: 2},
        mixed_valence={0: 1}, tau=0.01, chain_restrict=False, dense_cutoff=1000,
        spin_flip_dj=False, comm=None, truncation_threshold=100000, verbose=False, slaterWeightMin=1e-12,
    )

    def gs_energy(basis):
        return np.linalg.eigvalsh(np.asarray(basis.build_dense_matrix(op)))[0]

    e_brute = gs_energy(find_ground_state_basis(op, **kw, use_prescan=False))
    e_prescan = gs_energy(find_ground_state_basis(op, **kw, use_prescan=True))
    assert np.isclose(e_brute, e_prescan, atol=1e-8)
