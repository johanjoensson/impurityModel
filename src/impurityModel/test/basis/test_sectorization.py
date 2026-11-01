"""Tests for symmetry-derived sectorization / restrictions (symmetry plan, Phase 3)."""

from itertools import combinations

import numpy as np
import pytest

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
from impurityModel.ed.basis_transcription import build_dense_matrix  # noqa: E402
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
    h_full = np.asarray(build_dense_matrix(full, op))
    ev, evec = np.linalg.eigh(h_full)
    e0, gs = ev[0], evec[:, 0]

    # Read off the ground state's conserved-charge sector (charges are conserved, so
    # each <N_S> is integer for the non-degenerate GS).
    local = [bytes(d.to_bytearray()) for d in full.local_basis]
    gs_occ = [
        round(sum(abs(gs[i]) ** 2 * len(charge & _occset(local[i])) for i in range(len(local)))) for charge in charges
    ]
    assert gs_occ == [1, 1]  # half-filled singlet sector

    restrictions = restrictions_from_charges(charges, gs_occ)
    sector_dets = [d for d in all_dets if all(lo <= len(s & _occset(d)) <= hi for s, (lo, hi) in restrictions.items())]
    restricted = Basis(initial_basis=sector_dets, restrictions=restrictions, **_BASIS_KW)
    e0_restricted = np.linalg.eigvalsh(np.asarray(build_dense_matrix(restricted, op)))[0]

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


def _siam_6(u=4.0):
    """Single-impurity Anderson model, 6 spin-orbitals.

    0,1 = impurity (dn,up); 2,3 = valence bath (dn,up); 4,5 = conduction bath (dn,up).
    """
    ed, ev, ec, v = -1.0, -3.0, 3.0, 0.5
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


@pytest.mark.parametrize("u", [2.0, 8.0])
def test_hf_seed_finds_ground_state_sector(u):
    """The Hartree-Fock seed identifies the impurity occupation of the correlated ground state at
    the particle number HF solves at.

    HF works at fixed ``n_tot = N0 + n_valence`` (4 here), so that is the sector the oracle
    diagonalizes -- not the whole Fock space, whose ground state sits at a different total (3, at
    ``u = 4``) and agreed with the seed only by accident: until the two-body tensor was
    pair-symmetrized, the Fock matrix was not the energy's gradient and the SCF stopped at a
    non-stationary density near ``n_imp = 1`` (E = -4.221 where the true minimum is -4.656). ``u = 2``
    keeps the nominal occupation (2), ``u = 8`` moves it (to 1), so both answers are covered.
    """
    from impurityModel.ed.groundstate import hartree_fock_seed_occupation

    op = _siam_6(u=u)
    impurity_orbitals = {0: [[0, 1]]}
    bath_states = ({0: [[2, 3]]}, {0: [[4, 5]]})
    N0 = {0: 2}
    n_tot = N0[0] + 2

    dets = [_bytes6(occ) for occ in combinations(range(6), n_tot)]
    fixed_n = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=dets,
        verbose=False,
    )
    _, vecs = np.linalg.eigh(np.asarray(build_dense_matrix(fixed_n, op)))
    exact_imp_occ = sum(
        abs(amp) ** 2 * len({0, 1} & _occset6(bytes(d.to_bytearray())))
        for amp, d in zip(vecs[:, 0], fixed_n.local_basis)
    )
    # Precondition: the answer is not on a rounding boundary, where a mean-field miss by one is
    # expected and documented (hartree_fock_seed_occupation) rather than a defect.
    assert abs(exact_imp_occ - round(exact_imp_occ)) < 0.2, "the model drifted onto a rounding boundary"

    winning_N0, converged = hartree_fock_seed_occupation(op, impurity_orbitals, bath_states, N0)

    # The mean-field seed lands on the correct integer impurity occupation of the correlated GS,
    # and says so: an unconverged seed is not an occupation and callers must reject it.
    assert converged
    assert winning_N0 == {0: round(exact_imp_occ)}


def test_hf_seeded_walk_finds_the_whole_fock_space_ground_state():
    """From the HF seed, the occupation walk reaches the ground state of the whole Fock space.

    At ``u = 4`` the seed (2) is not that sector (1): the fixed-N exact occupation is 1.43, on the
    rounding boundary, and HF's 1.60 rounds the other way. The walk is what makes the seed's miss
    harmless, so it must land where the dN scan does.
    """
    from impurityModel.ed import groundstate

    op = _siam_6()
    kwargs = dict(impurity_orbitals={0: [[0, 1]]}, bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}), N0={0: 2}, verbose=False)
    seeded = groundstate.find_ground_state_sector(op, use_hf_seed=True, **kwargs)
    scanned = groundstate.find_ground_state_sector(op, use_hf_seed=False, **kwargs)
    assert seeded.occupation == scanned.occupation == {0: 1}
    assert seeded.energy == pytest.approx(scanned.energy, abs=1e-10)


def _two_group_siam(e_imp=-1.0, u=4.0, v=0.5, e_bath=-3.0):
    """Two impurity orbital pairs (0,1) and (2,3), each with its own valence bath pair ((4,5), (6,7)), and a
    Hubbard ``u`` within each pair: the smallest impurity whose two groups let the dN scan's diagonal move
    reach a total the +-1 walk does not."""
    terms = {}
    for o in range(4):
        terms[((o, "c"), (o, "a"))] = e_imp
        terms[((o + 4, "c"), (o + 4, "a"))] = e_bath
        terms[((o, "c"), (o + 4, "a"))] = v
        terms[((o + 4, "c"), (o, "a"))] = v
    for a in (0, 2):
        terms[((a, "c"), (a + 1, "c"), (a + 1, "a"), (a, "a"))] = u
    return ManyBodyOperator(terms)


def test_unconverged_hf_seed_is_rejected(monkeypatch, capsys):
    """A non-converged HF seed must never be used: it is the last iterate, not a minimiser.

    Measured on the NiO L-edge workload, an unconverged seed returned a 3d10 impurity paid for by
    emptying the (frozen) 2p core. Re-freezing the core then left two electrons from nowhere, the
    shell closed, the basis collapsed to a single determinant and every core-level spectrum came
    out identically zero -- while PES still looked healthy.

    find_ground_state_sector walks +-1 from the nominal N0 instead (review P8), and says so at the
    default verbosity. It used to fall back to the 3^k dN scan. The impurity is split into two groups
    so the two differ: the scan solves the diagonal move (+1, +1), total N0 + 2 = 4, which a descent
    to the winner (total 1) never reaches.
    """
    from impurityModel.ed import groundstate

    op = _two_group_siam()
    kwargs = dict(
        impurity_orbitals={0: [[0, 1]], 1: [[2, 3]]},
        bath_states=({0: [[4, 5]], 1: [[6, 7]]}, {0: [[]], 1: [[]]}),
        N0={0: 1, 1: 1},
        verbose=False,
    )
    solved = []
    calc_energy = groundstate.calc_energy

    def spy(h_op, impurity_orbitals, bath_states, trial_N0, *args, **kw):
        solved.append(sum(trial_N0.values()))
        return calc_energy(h_op, impurity_orbitals, bath_states, trial_N0, *args, **kw)

    monkeypatch.setattr(groundstate, "calc_energy", spy)

    # The oracle: the dN scan, which also shows the model separates the two (it solves total 4).
    scanned = groundstate.find_ground_state_sector(op, use_hf_seed=False, **kwargs)
    assert 4 in solved, "the scan no longer visits N0 + 2: the test cannot tell the walk from it"
    solved.clear()

    # Same problem, but HF now reports a garbage occupation and admits it did not converge.
    calls = []
    monkeypatch.setattr(
        groundstate,
        "hartree_fock_seed_occupation",
        lambda *a, **k: (calls.append(1), ({0: 0, 1: 0}, False))[1],
    )
    capsys.readouterr()
    fell_back = groundstate.find_ground_state_sector(op, **kwargs)

    assert calls, "the seed should still be consulted"
    # The rejected seed (impurity occupation 0) must not have been taken: the walk starts at N0.
    assert solved[0] == 2
    assert 4 not in solved, f"the rejected seed fell back to the dN scan (solved totals {solved})"
    assert sum(fell_back.occupation.values()) == sum(scanned.occupation.values()) == 2
    assert fell_back.energy == pytest.approx(scanned.energy, abs=1e-12)
    assert "HF seed rejected (HF did not converge)" in capsys.readouterr().out


@pytest.mark.xfail(
    reason="review step 4 open item: with >= 2 groups and non-convex E(N) (attractive U) the +-1 descent stops in a "
    "local minimum the scan's diagonal move escapes (2 eV high). Owned by review step 9 (MV 3's N+-2 extension).",
    strict=True,
)
def test_pm1_descent_finds_the_scan_minimum_on_a_non_convex_two_group_model():
    from impurityModel.ed import groundstate

    op = _two_group_siam(e_imp=1.0, u=-4.0, v=0.05)
    kwargs = dict(
        impurity_orbitals={0: [[0, 1]], 1: [[2, 3]]},
        bath_states=({0: [[4, 5]], 1: [[6, 7]]}, {0: [[]], 1: [[]]}),
        N0={0: 1, 1: 1},
        verbose=False,
    )
    scanned = groundstate.find_ground_state_sector(op, use_hf_seed=False, **kwargs)
    walked = groundstate.find_ground_state_sector(op, **kwargs)
    assert walked.energy == pytest.approx(scanned.energy, abs=1e-8)


def test_hf_seed_that_raids_a_frozen_shell_is_rejected(monkeypatch):
    """HF is free to move charge into or out of a shell this calculation holds frozen, paying for
    it elsewhere. Re-freezing afterwards keeps HF's answer for the *unfrozen* shells but undoes the
    compensation, so the impurity electron count silently drifts -- exactly the NiO failure (HF
    {2p: 4, 3d: 10}, sum 14; re-frozen {2p: 6, 3d: 10}, sum 16). Such a seed is not a seed for this
    problem and must be rejected even when HF converged."""
    from impurityModel.ed import groundstate

    op = _siam_6()
    impurity_orbitals = {0: [[0, 1]]}
    bath_states = ({0: [[2, 3]]}, {0: [[4, 5]]})
    N0 = {0: 2}

    scanned = groundstate.find_ground_state_sector(
        op,
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        N0=N0,
        frozen_occupations={0},
        verbose=False,
        use_hf_seed=False,
    )
    # A *converged* seed that moved charge out of the frozen set 0: re-freezing restores N0[0]=2
    # while keeping HF's answer elsewhere, changing the impurity count 0 -> 2. Reject it.
    monkeypatch.setattr(groundstate, "hartree_fock_seed_occupation", lambda *a, **k: ({0: 0}, True))
    fell_back = groundstate.find_ground_state_sector(
        op,
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        N0=N0,
        frozen_occupations={0},
        verbose=False,
    )
    assert fell_back.occupation == scanned.occupation
    assert fell_back.energy == scanned.energy


def _core_shell_toy(mix):
    """A valence pair (0, 1) with its valence bath pair (4, 5), and a bath-less core pair (2, 3) at -5 with a
    core-valence density-density repulsion: H conserves the core count, unless ``mix`` adds the one-body
    element ``c_0^+ c_2 + h.c.`` (RSPt's h_dft can carry one; the MS T4.4 fixture uses 1e-3)."""
    terms = {}
    for o in (0, 1):
        terms[((o, "c"), (o, "a"))] = -1.0
        terms[((o + 4, "c"), (o + 4, "a"))] = -3.0
        terms[((o, "c"), (o + 4, "a"))] = terms[((o + 4, "c"), (o, "a"))] = 0.5
    for o in (2, 3):
        terms[((o, "c"), (o, "a"))] = -5.0
        for w in (0, 1):
            terms[((o, "c"), (w, "c"), (w, "a"), (o, "a"))] = 1.0
    if mix:
        terms[((0, "c"), (2, "a"))] = terms[((2, "c"), (0, "a"))] = mix
    impurity_orbitals = {0: [[0, 1]], 1: [[2, 3]]}
    bath_states = ({0: [[4, 5]], 1: [[]]}, {0: [[]], 1: [[]]})
    return ManyBodyOperator(terms), impurity_orbitals, bath_states, {0: 1, 1: 2}


@pytest.mark.parametrize("mix", [0.0, 1e-3])
def test_a_frozen_shell_h_does_not_conserve_is_reported(mix, capsys):
    """A freeze pins only the seed, so it holds exactly only where H conserves the shell (review J1/O5). The
    one-body core-valence element is ``[H, N_core]``: measured by its coefficient, and warned about once, at the
    default verbosity. The conserved core (``mix = 0``) stays silent."""
    from impurityModel.ed import groundstate

    h_op, impurity_orbitals, bath_states, N0 = _core_shell_toy(mix)
    largest, total = groundstate.frozen_shell_leak(h_op, impurity_orbitals, {1})
    assert (largest, total) == (pytest.approx(mix), pytest.approx(2 * mix))
    assert groundstate.frozen_shell_leak(h_op, impurity_orbitals, set()) == (0.0, 0.0)

    capsys.readouterr()
    groundstate.solve_ground_state(
        h_op,
        impurity_orbitals,
        bath_states,
        N0,
        tau=0.01,
        slaterWeightMin=1e-12,
        frozen_occupations={1},
        truncation_threshold=np.inf,
        verbose=False,
    )
    out = capsys.readouterr().out
    assert ("does not conserve the electron count of the frozen shell(s) [1]" in out) == bool(mix), out
    if mix:
        assert "1.00e-03" in out


def test_the_frozen_shell_leak_counts_two_body_terms_and_is_relative(capsys):
    """Two-body terms: the core-valence exchange ``c2+ c0+ c3 c1`` conserves the core count, ``c2+ c1+ c1 c0``
    moves one core electron, the pair term ``c2+ c3+ c1 c0`` two -- each is ``[H, N_core]`` by its coefficient. And
    the threshold is relative to H's largest term: a 1e-12 element under a core energy of -5 is noise, silent."""
    from impurityModel.ed import groundstate

    h_op, impurity_orbitals, bath_states, N0 = _core_shell_toy(0.0)
    terms = dict(h_op)
    terms[((2, "c"), (0, "c"), (3, "a"), (1, "a"))] = 0.4
    assert groundstate.frozen_shell_leak(ManyBodyOperator(terms), impurity_orbitals, {1}) == (0.0, 0.0)
    terms[((2, "c"), (1, "c"), (1, "a"), (0, "a"))] = 0.3
    terms[((2, "c"), (3, "c"), (1, "a"), (0, "a"))] = 0.2
    largest, total = groundstate.frozen_shell_leak(ManyBodyOperator(terms), impurity_orbitals, {1})
    assert largest == pytest.approx(0.3) and total == pytest.approx(0.5)

    h_op, impurity_orbitals, bath_states, N0 = _core_shell_toy(1e-12)
    assert groundstate.frozen_shell_leak(h_op, impurity_orbitals, {1})[0] == pytest.approx(1e-12)
    capsys.readouterr()
    groundstate.solve_ground_state(
        h_op,
        impurity_orbitals,
        bath_states,
        N0,
        tau=0.01,
        slaterWeightMin=1e-12,
        frozen_occupations={1},
        truncation_threshold=np.inf,
        verbose=False,
    )
    assert "does not conserve" not in capsys.readouterr().out


def _bytes6(occ):
    b = bytearray(1)
    for o in occ:
        b[0] |= 1 << (7 - o)
    return bytes(b)


def _occset6(det_bytes):
    return {k for k, bit in enumerate(psr.bytes2bitarray(det_bytes, 6)) if bit}


def test_hf_seed_matches_brute_force_loop():
    """find_ground_state_sector with use_hf_seed=True matches the O(3^k) dN scan."""
    from impurityModel.ed.groundstate import find_ground_state_sector

    op = _siam_6()
    kw = dict(
        impurity_orbitals={0: [[0, 1]]},
        bath_states=({0: [[2, 3]]}, {0: [[4, 5]]}),
        N0={0: 2},
        chain_restrict=False,
        dense_cutoff=1000,
        comm=None,
        truncation_threshold=100000,
        verbose=False,
        slaterWeightMin=1e-12,
    )

    e_brute = find_ground_state_sector(op, **kw, use_hf_seed=False).energy
    e_hf = find_ground_state_sector(op, **kw, use_hf_seed=True).energy
    assert np.isclose(e_brute, e_hf, atol=1e-8)
