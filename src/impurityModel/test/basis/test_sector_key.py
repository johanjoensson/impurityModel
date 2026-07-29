"""The occupation walk's sector identity: which trial occupations are the *same* solve.

``generate_initial_basis`` pins each frozen group at its own value and, with two or more
unfrozen groups, lets every one of them range over its whole ``[0, group_size]`` while filtering
only the **total** impurity charge. So two trial occupations differing only in how one total is
split across unfrozen groups build the identical basis and are one sector solve --
``groundstate.sector_key`` is that equivalence, and ``find_ground_state_basis``'s energy cache
keys on it so the walk's diagonal probe stops paying for the same solve several times.

These tests pin the property the key *relies* on (identical determinant sets, before and after
the CIPSI expansion), the boundary where it stops holding (frozen groups, which do not
redistribute), and the one place merging trials could go wrong (an out-of-bounds split sharing a
key with a valid one).
"""

import numpy as np
import pytest

from impurityModel.ed.basis_generation import generate_initial_basis
from impurityModel.ed.groundstate import calc_energy, sector_key
from impurityModel.ed.lie_algebra import tensors_to_operator

# Two impurity groups of two spin-orbitals each, split in energy so the groups are
# distinguishable, plus one valence and one conduction bath orbital per group.
GROUP_SIZE = 2
N_ORB = 12


def _model():
    """(h_op, impurity_orbitals, bath_states) for a 2-group impurity with valence+conduction baths."""
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    impurity_orbitals = {0: [[0, 1]], 1: [[2, 3]]}
    valence = {0: [[4, 5]], 1: [[6, 7]]}
    conduction = {0: [[8, 9]], 1: [[10, 11]]}
    for group, (eps, imp) in enumerate(((-0.6, (0, 1)), (0.4, (2, 3)))):
        for orb in imp:
            h[orb, orb] = eps
    for orb in (4, 5, 6, 7):
        h[orb, orb] = -2.0
    for orb in (8, 9, 10, 11):
        h[orb, orb] = 2.0
    # Hybridize each impurity orbital with its own valence and conduction partner.
    for imp, val, con in ((0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)):
        for other in (val, con):
            h[imp, other] = h[other, imp] = 0.35
    return tensors_to_operator(h), impurity_orbitals, (valence, conduction)


def _seed(nominal, frozen=None):
    """The determinant set ``generate_initial_basis`` produces for one trial occupation."""
    _h_op, impurity_orbitals, bath_states = _model()
    states, _n = generate_initial_basis(
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        delta_valence_occ=dict.fromkeys(nominal, 0),
        delta_conduction_occ=dict.fromkeys(nominal, 0),
        delta_impurity_occ=dict.fromkeys(nominal, 0),
        nominal_impurity_occ=nominal,
        mixed_valence=dict.fromkeys(nominal, 0),
        n_bytes=(N_ORB + 7) // 8,
        verbose=False,
        frozen_occupations=frozen,
    )
    return {bytes(state.to_bytearray()) for state in states}


def _solve(nominal, frozen=None):
    """Energy and post-expansion support of one sector solve, through the production path."""
    h_op, impurity_orbitals, bath_states = _model()
    energy, basis = calc_energy(
        h_op,
        impurity_orbitals,
        bath_states,
        nominal,
        dict.fromkeys(nominal, 0),
        1e-3,
        False,
        False,
        1000,
        comm=None,
        verbose=False,
        truncation_threshold=np.inf,
        slaterWeightMin=1e-12,
        frozen_occupations=frozen,
    )
    return energy, {bytes(state.to_bytearray()) for state in basis.local_basis}


@pytest.mark.parametrize("split", [{0: 1, 1: 3}, {0: 3, 1: 1}, {0: 0, 1: 4}, {0: 4, 1: 0}])
def test_every_split_of_a_total_seeds_the_identical_basis(split):
    """The claim the cache key rests on, at the generator."""
    assert sum(split.values()) == 4
    assert _seed(split) == _seed({0: 2, 1: 2})


@pytest.mark.parametrize("split", [{0: 1, 1: 3}, {0: 3, 1: 1}])
def test_every_split_of_a_total_gives_the_identical_solve(split):
    """And after the CIPSI expansion, which is what the cached *energy* actually stands for.

    Pinning only the generator's seed would leave open that the expansion (which starts from
    that seed but is driven by the Hamiltonian) diverges between splits. It does not: same seed,
    same operator, same deterministic selection.
    """
    reference_energy, reference_support = _solve({0: 2, 1: 2})
    energy, support = _solve(split)
    assert energy == reference_energy
    assert support == reference_support


def test_a_different_total_is_a_different_basis():
    """The key would be worthless if it merged everything -- totals must still separate."""
    assert _seed({0: 2, 1: 2}) != _seed({0: 2, 1: 3})
    assert _solve({0: 2, 1: 2})[0] != _solve({0: 2, 1: 3})[0]


def test_a_frozen_group_does_not_redistribute():
    """Freezing group 0 pins it, so the split stops being free and the key must keep it.

    With one group frozen only one group is left unfrozen, so `redistribute` is False and each
    group's window is its own -- two occupations with the same total but different frozen values
    are genuinely different bases.
    """
    # Both groups hold 2 spin-orbitals, so these are the two reachable splits of total 3.
    frozen = {0}
    assert _seed({0: 1, 1: 2}, frozen) and _seed({0: 2, 1: 1}, frozen), "fixture generates nothing"
    assert _seed({0: 1, 1: 2}, frozen) != _seed({0: 2, 1: 1}, frozen)
    assert sector_key({0: 1, 1: 2}, frozen) != sector_key({0: 2, 1: 1}, frozen)
    # Unfrozen, the same two are one solve -- which is the whole point of the key.
    assert _seed({0: 1, 1: 2}) == _seed({0: 2, 1: 1})
    assert sector_key({0: 1, 1: 2}) == sector_key({0: 2, 1: 1})


def test_the_key_separates_exactly_what_the_generator_separates():
    unfrozen = sector_key({0: 0, 1: 2}), sector_key({0: 2, 1: 0}), sector_key({0: 1, 1: 1})
    assert len(set(unfrozen)) == 1, unfrozen
    assert sector_key({0: 2, 1: 2}) != sector_key({0: 2, 1: 3})
    # Frozen values ride along in the key, so a frozen group's own occupation still separates.
    assert sector_key({0: 1, 1: 2}, {0}) == (3, ((0, 1),))
    assert sector_key({0: 2, 1: 1}, {0}) == (3, ((0, 2),))


def test_the_key_is_all_integers():
    """A float in the key makes hit/miss depend on roundoff, so ranks can disagree on whether
    to run a collective sector solve -- an immediate deadlock rather than a wrong number."""
    total, pinned = sector_key({0: np.int64(2), 1: 2}, {0})
    assert isinstance(total, int)
    assert all(isinstance(group, int) and isinstance(occ, int) for group, occ in pinned)


def _walk(nominal, frozen=None, use_hf_seed=False):
    """One full occupation walk, with its cost accounting."""
    from impurityModel.ed import solver_trace
    from impurityModel.ed.groundstate import find_ground_state_basis

    h_op, impurity_orbitals, bath_states = _model()
    with solver_trace.tracing() as trace:
        basis = find_ground_state_basis(
            h_op,
            impurity_orbitals,
            bath_states,
            nominal,
            mixed_valence=dict.fromkeys(nominal, 0),
            tau=1e-3,
            chain_restrict=False,
            dense_cutoff=1000,
            spin_flip_dj=False,
            comm=None,
            verbose=False,
            truncation_threshold=np.inf,
            slaterWeightMin=1e-12,
            frozen_occupations=frozen,
            use_hf_seed=use_hf_seed,
        )
    return basis, trace


def test_the_walk_reuses_a_solve_across_splits_of_one_total():
    """The dedupe, measured rather than asserted.

    The diagonal 3^k probe visits every combination move from the seed, so on a 2-group impurity
    it reaches the same total by several routes -- (+1,-1) and (-1,+1) both land on the seed's
    own total. Those are one solve now, and the cache-hit counter the Phase-2 instrumentation
    added is what says so.
    """
    _basis, trace = _walk({0: 1, 1: 1})
    assert trace.count("sector_solve") > 0
    assert trace.count("sector_cache_hit") > 0

    # No sector is solved more than twice: once inside the walk, and -- for the winner only --
    # once more at the end, where find_ground_state_basis clears the energy cache and re-solves
    # because it needs the basis, not just the energy.
    solved = [sum(dict(event["n0"]).values()) for event in trace.of_kind("sector_solve")]
    repeats = {total for total in solved if solved.count(total) > 1}
    assert all(solved.count(total) == 2 for total in repeats), solved
    assert len(repeats) <= 1, solved

    # Every cache hit refers to a total that really was solved -- the key merges trials, it does
    # not invent them.
    hit_totals = {event["n0"][0] for event in trace.of_kind("sector_cache_hit")}
    assert hit_totals <= set(solved), (hit_totals, solved)


def test_the_walk_finds_the_true_minimum_despite_out_of_bounds_probes():
    """The one way merging trials could return a wrong answer, checked against a brute force.

    The per-group bounds test rejects {0: 0, 1: 3} (group 1 holds 2 spin-orbitals) while
    {0: 1, 1: 2} is the same, perfectly valid, total-3 sector -- and they share a cache key. The
    walk's diagonal probe generates the out-of-bounds one *unguarded*, so caching that rejection
    would hand `inf` to whichever of the two the walk asked for second, and the walk would
    silently skip a sector. Comparing against every reachable total solved directly is what
    catches that: a poisoned cache shows up as the walk settling above the true minimum.
    """
    reachable = {}
    for total in range(2 * GROUP_SIZE + 1):
        split = {0: min(total, GROUP_SIZE), 1: total - min(total, GROUP_SIZE)}
        reachable[total] = _solve(split)[0]
    best_total = min(reachable, key=reachable.get)

    basis, _trace = _walk({0: 1, 1: 1})
    assert sum(basis.ground_state_occupation.values()) == best_total, (
        basis.ground_state_occupation,
        reachable,
    )


def test_the_walk_reports_the_total_it_settled_on():
    """`ground_state_occupation` is what fixed_peak_dc sums; the total is the meaningful part."""
    basis, _trace = _walk({0: 1, 1: 1})
    occ = basis.ground_state_occupation
    assert set(occ) == {0, 1}
    assert all(isinstance(v, (int, np.integer)) for v in occ.values()), occ
    assert 0 <= sum(occ.values()) <= 2 * GROUP_SIZE, occ
