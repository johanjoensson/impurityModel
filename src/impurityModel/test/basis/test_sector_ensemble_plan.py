"""Membership and weights of the charge-sector ensemble (MV 5): pure functions of the walk's solved sectors."""

import numpy as np
import pytest

from impurityModel.ed.average import energy_cut
from impurityModel.ed.groundstate import SectorRecord
from impurityModel.ed.sector_ensemble import plan_ensemble, sector_weights

TAU = 2e-3


def _rec(total, energy, residual=0.0, certified=True, cap_bound=False):
    return SectorRecord({0: total}, energy, residual, certified, cap_bound, None)


def test_a_sector_inside_the_window_is_admitted_and_one_outside_is_not():
    cut = energy_cut(TAU)
    solved = [_rec(8, 0.0), _rec(9, 0.5 * cut), _rec(7, 2.0 * cut), _rec(10, 3 * cut), _rec(6, 3 * cut)]
    plan = plan_ensemble(solved, TAU, "on", max_error=1.0)
    assert [m.total for m in plan.members] == [8, 9]
    assert plan.needs == () and plan.refused_reason is None


def test_the_residual_is_optimistic_for_admission():
    cut = energy_cut(TAU)
    far = _rec(9, 1.5 * cut, residual=1.0 * cut)
    plan = plan_ensemble([_rec(8, 0.0), far, _rec(7, 5 * cut), _rec(10, 5 * cut)], TAU, "on", max_error=1e9)
    assert [m.total for m in plan.members] == [8, 9]


def test_an_admitted_edge_asks_for_its_outer_neighbour_and_respects_attainability():
    cut = energy_cut(TAU)
    solved = [_rec(8, 0.0), _rec(9, 0.5 * cut), _rec(7, 0.6 * cut)]
    plan = plan_ensemble(solved, TAU, "on")
    assert sorted(plan.needs) == [6, 10]
    plan = plan_ensemble(solved, TAU, "on", attainable=lambda t: t <= 9)
    assert plan.needs == (6,)


def test_the_winner_alone_needs_nothing_when_its_neighbours_are_solved():
    cut = energy_cut(TAU)
    plan = plan_ensemble([_rec(8, 0.0), _rec(7, 4 * cut), _rec(9, 4 * cut)], TAU, "auto")
    assert [m.total for m in plan.members] == [8] and plan.needs == () and plan.refused_reason is None


@pytest.mark.parametrize(
    "bad, word",
    [(dict(cap_bound=True), "cap"), (dict(certified=False), "certified")],
)
def test_a_capped_or_uncertified_member_refuses_the_ensemble(bad, word):
    cut = energy_cut(TAU)
    solved = [_rec(8, 0.0), _rec(9, 0.4 * cut, **bad), _rec(7, 6 * cut), _rec(10, 6 * cut)]
    plan = plan_ensemble(solved, TAU, "auto")
    assert plan.refused_reason and word in plan.refused_reason
    assert [m.total for m in plan.members] == [8]  # the winner alone


def test_the_error_bar_refuses_above_the_limit_and_scales_with_the_residual():
    cut = energy_cut(TAU)

    def plan(residual):
        solved = [_rec(8, 0.0, residual), _rec(9, 0.0, residual), _rec(7, 6 * cut), _rec(10, 6 * cut)]
        return plan_ensemble(solved, TAU, "auto")

    small, large = plan(1e-6), plan(1e-3)
    assert small.refused_reason is None and small.error_bar < 0.05
    assert large.refused_reason and "e by the sector energies" in large.refused_reason
    assert large.error_bar == pytest.approx(1e3 * small.error_bar)
    # two equal members: sum_s p_s |N_s - <N>| err_s / tau = 2 * 0.5 * 0.5 * err / tau
    assert large.error_bar == pytest.approx(0.5 * 1e-3 / TAU)


def test_the_merge_weights_are_invariant_under_a_shift_and_a_permutation():
    e0 = np.array([0.0, 0.003, 0.010])
    z = np.array([1.7, 1.2, 1.0])
    c = sector_weights(e0, z, TAU)
    assert c.sum() == pytest.approx(1.0)
    for shift in (-10 * TAU, 0.0, 50 * TAU):
        assert sector_weights(e0 + shift, z, TAU) == pytest.approx(c, rel=1e-12)
    perm = [2, 0, 1]
    assert sector_weights(e0[perm], z[perm], TAU) == pytest.approx(c[perm], rel=1e-12)
    assert list(sector_weights([3.0], [4.2], TAU)) == [1.0]  # one member: exactly 1.0


def test_the_weights_reproduce_the_grand_canonical_average_on_an_exact_spectrum():
    """Two sectors with spectra {0, 1e-3} and {2e-3, 5e-3}: c_s Z_s-weighted occupation = exact GC <N>."""
    spectra = {8: np.array([0.0, 1e-3]), 9: np.array([2e-3, 5e-3])}
    exact = sum(n * np.exp(-e / TAU).sum() for n, e in spectra.items()) / sum(
        np.exp(-e / TAU).sum() for e in spectra.values()
    )
    e0 = np.array([spectra[n].min() for n in (8, 9)])
    z = np.array([np.exp(-(spectra[n] - spectra[n].min()) / TAU).sum() for n in (8, 9)])
    c = sector_weights(e0, z, TAU)
    assert c @ np.array([8.0, 9.0]) == pytest.approx(exact, rel=1e-13)
    # mutation: giving every member the full weight (or ignoring e0_s) must NOT match
    wrong = np.array([0.5, 0.5])
    assert abs(wrong @ np.array([8.0, 9.0]) - exact) > 1e-3
    assert abs(sector_weights(np.zeros(2), z, TAU) @ np.array([8.0, 9.0]) - exact) > 1e-3


def test_a_capped_winner_alone_is_a_one_member_ensemble_not_a_refusal():
    cut = energy_cut(TAU)
    plan = plan_ensemble([_rec(8, 0.0, cap_bound=True, certified=False), _rec(7, 6 * cut), _rec(9, 6 * cut)], TAU, "on")
    assert plan.refused_reason is None and [m.total for m in plan.members] == [8]


# --- the walk's extension hook, on the toy model of test_mixed_valence_is_off (collective) ---------------------
from mpi4py import MPI  # noqa: E402

from impurityModel.ed import groundstate  # noqa: E402
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator  # noqa: E402
from impurityModel.test.support.comms import COMMS  # noqa: E402

IMPURITY = {0: [[0, 1]]}
BATHS = ({0: [[2, 3]]}, {0: [[4, 5]]})


def _toy():
    terms = {}
    for s in (0, 1):
        for orbital, energy in ((s, 0.9), (2 + s, -1.2), (4 + s, 1.0)):
            terms[((orbital, "c"), (orbital, "a"))] = energy
    return ManyBodyOperator(terms)


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_the_walk_solves_the_totals_the_plan_asks_for_and_every_rank_agrees(comm):
    kw = dict(slaterWeightMin=1e-12, truncation_threshold=np.inf, verbose=False, comm=comm)
    tau = 0.4  # window 3.7: N_imp = 1 (0.9 above) and N_imp = 2 (1.8 above) are inside
    plain = groundstate.find_ground_state_sector(_toy(), IMPURITY, BATHS, {0: 0}, **kw)
    assert sorted(r.total for r in plain.solved) == [0, 1]  # the plain walk stops at the winner's neighbours

    def extend(solved):
        return plan_ensemble(solved, tau, "on", attainable=lambda t: 0 <= t <= 2, max_error=1e9).needs

    wide = groundstate.find_ground_state_sector(_toy(), IMPURITY, BATHS, {0: 0}, extend_sectors=extend, **kw)
    assert sorted(r.total for r in wide.solved) == [0, 1, 2]
    plan = plan_ensemble(wide.solved, tau, "on", attainable=lambda t: 0 <= t <= 2, max_error=1e9)
    assert plan.needs == () and [m.total for m in plan.members] == [0, 1, 2]
    totals = (comm or MPI.COMM_SELF).allgather([m.total for m in plan.members])
    assert all(t == totals[0] for t in totals)
    # the walk's winner and energy do not depend on the hook
    assert wide.energy == plain.energy and wide.occupation == plain.occupation


def _exact_member_weights(spectra, n_val, tau, totals):
    """Exact weight of each member by dense diagonalisation: a member is a total-electron-number sector.

    A member's thermal states have several N_imp (the bath hybridises the impurity) but one electron count
    ``n_val + total`` (bath valence filled, conduction empty).  Its weight is ``Z_s / sum Z`` over the members.
    """
    z = np.array([np.exp(-spectra[n_val + t] / tau).sum() for t in totals])
    return z / z.sum(), z.sum()


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("index, expected", [(3, [1, 2]), (33, [0, 1]), (10, [0, 1, 2])])
def test_the_ensemble_weights_are_the_exact_grand_canonical_sector_weights_and_every_rank_agrees(comm, index, expected):
    from impurityModel.test.support.gs_oracle import exact_spectra, pinned_model

    tau = 0.15
    h_op, imp, bath, N0, n_orb, n_val, _ = pinned_model(20261016, index)
    spectra = exact_spectra(h_op, n_orb)
    kw = dict(slaterWeightMin=1e-12, truncation_threshold=np.inf, verbose=False, comm=comm, tau=tau)
    kept = []

    def consume(member, gs):
        kept.append((member.total, np.array(gs.es)))
        return member.total

    plan, results, realised = groundstate.solve_ground_state_ensemble(
        h_op, imp, bath, N0, mode="on", consume=consume, **kw
    )
    assert results == [m.total for m in plan.members] == [m for m, _ in kept]
    assert sorted(results) == expected
    totals = [t for t, _ in kept]
    e0 = np.array([e.min() for _, e in kept])
    z = np.array([np.exp(-(e - e.min()) / tau).sum() for _, e in kept])
    c = sector_weights(e0, z, tau)
    exact, norm = _exact_member_weights(spectra, n_val, tau, totals)
    assert c == pytest.approx(exact, abs=2e-3)
    # the unshifted union is the partition function of the members (to each window's truncation)
    assert sum(np.exp(-e / tau).sum() for _, e in kept) == pytest.approx(norm, rel=2e-3)
    # mutation: the winner alone (weight 1) or equal weights must NOT reproduce the sector weights
    assert np.abs(exact - 1 / len(exact)).max() > 0.02
    if exact.min() > 0.1:
        assert abs(1.0 - exact.max()) > 0.1
    agree = (comm or MPI.COMM_SELF).allgather([float(x) for x in realised])
    assert all(a == agree[0] for a in agree)
