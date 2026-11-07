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
    plan = plan_ensemble(solved, TAU)
    assert [m.total for m in plan.members] == [8, 9]
    assert plan.needs == () and plan.refused_reason is None


def test_the_residual_is_optimistic_for_admission():
    cut = energy_cut(TAU)
    far = _rec(9, 1.5 * cut, residual=1.0 * cut)
    plan = plan_ensemble([_rec(8, 0.0), far, _rec(7, 5 * cut), _rec(10, 5 * cut)], TAU)
    assert [m.total for m in plan.members] == [8, 9]


def test_an_admitted_edge_asks_for_its_outer_neighbour_and_respects_attainability():
    cut = energy_cut(TAU)
    solved = [_rec(8, 0.0), _rec(9, 0.5 * cut), _rec(7, 0.6 * cut)]
    plan = plan_ensemble(solved, TAU)
    assert sorted(plan.needs) == [6, 10]
    plan = plan_ensemble(solved, TAU, attainable=lambda t: t <= 9)
    assert plan.needs == (6,)


def test_the_winner_alone_needs_nothing_when_its_neighbours_are_solved():
    cut = energy_cut(TAU)
    plan = plan_ensemble([_rec(8, 0.0), _rec(7, 4 * cut), _rec(9, 4 * cut)], TAU)
    assert [m.total for m in plan.members] == [8] and plan.needs == () and plan.refused_reason is None


@pytest.mark.parametrize("bad", [dict(cap_bound=True), dict(certified=False), dict(cap_bound=True, certified=False)])
def test_a_capped_or_uncertified_member_still_enters_the_ensemble(bad):
    """Admission is by energy alone (S4): the truncation widens the weight interval, it does not drop the sector."""
    cut = energy_cut(TAU)
    solved = [_rec(8, 0.0), _rec(9, 0.4 * cut, **bad), _rec(7, 6 * cut), _rec(10, 6 * cut)]
    plan = plan_ensemble(solved, TAU)
    assert plan.refused_reason is None
    assert [m.total for m in plan.members] == [8, 9]


def test_the_error_bar_scales_with_the_residual_and_never_refuses():
    cut = energy_cut(TAU)

    def plan(residual):
        solved = [_rec(8, 0.0, residual), _rec(9, 0.0, residual), _rec(7, 6 * cut), _rec(10, 6 * cut)]
        return plan_ensemble(solved, TAU)

    small, large = plan(1e-6), plan(1e-3)
    assert small.refused_reason is None and large.refused_reason is None
    assert large.error_bar == pytest.approx(1e3 * small.error_bar)
    # two equal members: sum_s p_s |N_s - <N>| err_s / tau = 2 * 0.5 * 0.5 * err / tau
    assert large.error_bar == pytest.approx(0.5 * 1e-3 / TAU)


def test_the_weight_interval_is_exact_on_a_brute_force_vertex_scan():
    """<N> is monotone in each sector weight: the sorted scan finds the same extremes as all 2^K vertices."""
    import itertools

    from impurityModel.ed.sector_ensemble import weight_interval

    n = np.array([6.0, 7.0, 8.0, 9.0])
    e0 = np.array([0.003, 0.0, 0.002, 0.007])
    z = np.array([1.3, 1.0, 2.1, 1.0])
    shift = np.array([0.001, 0.0, 0.004, 0.002])
    lo, hi = weight_interval(n, e0, z, TAU, shift)
    values = []
    for corner in itertools.product([0, 1], repeat=4):
        a = z * np.exp(-(e0 - shift * np.array(corner) - e0.min()) / TAU)
        values.append(a @ n / a.sum())
    assert lo == pytest.approx(min(values)) and hi == pytest.approx(max(values))
    assert lo < float(sector_weights(e0, z, TAU) @ n) < hi
    # a sector with no uncertainty and one member: a point
    assert weight_interval([8.0], [0.0], [1.0], TAU, [0.0]) == (8.0, 8.0)


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
    plan = plan_ensemble([_rec(8, 0.0, cap_bound=True, certified=False), _rec(7, 6 * cut), _rec(9, 6 * cut)], TAU)
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
        return plan_ensemble(solved, tau, attainable=lambda t: 0 <= t <= 2).needs

    wide = groundstate.find_ground_state_sector(_toy(), IMPURITY, BATHS, {0: 0}, more_totals=extend, **kw)
    assert sorted(r.total for r in wide.solved) == [0, 1, 2]
    plan = plan_ensemble(wide.solved, tau, attainable=lambda t: 0 <= t <= 2)
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

    class Keep:
        def visit(self, member, state):
            kept.append((member.total, np.array(state.es)))
            return member.total

    result = groundstate.solve_charge_sector_ensemble(h_op, imp, bath, N0, mixed_valence=True, visitor=Keep(), **kw)
    realised = result.realised
    assert result.visited == [m.total for m in result.members] == [m for m, _ in kept]
    assert sorted(result.visited) == expected
    totals = [t for t, _ in kept]
    e0 = np.array([e.min() for _, e in kept])
    z = np.array([np.exp(-(e - e.min()) / tau).sum() for _, e in kept])
    c = sector_weights(e0, z, tau)
    assert result.weights == pytest.approx(c, abs=1e-12)  # the entry's own weights are these, in this order
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


def test_attainable_pins_a_window_group_at_the_walks_filling_not_the_nominal_one(monkeypatch):
    """A SeedWindow group sits where the walk put it (Hartree-Fock's filling), which need not be the nominal one.
    ``attainable`` used the nominal filling, so the top edge was reported attainable (and the plan refused for it)
    and the bottom neighbour was treated as closed (review X1)."""
    from types import SimpleNamespace

    from impurityModel.ed.basis_generation import SeedWindow

    imp = {0: [list(range(6))], 1: [list(range(6, 16))]}  # p (window) 6 orbitals, d 10
    walked = {0: 5, 1: 10}  # HF put a hole in p
    nominal = {0: 6, 1: 8}
    seen = {}

    class Stop(Exception):
        pass

    monkeypatch.setattr(
        groundstate,
        "find_ground_state_sector",
        lambda *a, **k: SimpleNamespace(occupation=walked, solved=[]),
    )

    def spy(solved, tau, attainable):
        seen["top"], seen["bottom"] = attainable(16), attainable(5)
        raise Stop

    monkeypatch.setattr(groundstate, "plan_ensemble", spy)
    with pytest.raises(Stop):
        groundstate.solve_charge_sector_ensemble(
            None, imp, ({}, {}), nominal, mixed_valence=True, visitor=None, tau=0.1, slaterWeightMin=0,
            frozen_occupations=SeedWindow({0}),
        )  # fmt: skip
    assert seen == {"top": False, "bottom": True}


def test_the_ensemble_truncation_report_is_not_the_dominant_members_when_a_lighter_one_is_capped():
    """``gs_info['truncation']`` took the dominant member's report, hiding a cap-bound lighter member (review MV7)."""
    from impurityModel.ed.groundstate import worst_truncation

    capped, memory = {"cap": 10}, {"cap": 5, "memory_bound": True}
    assert worst_truncation([None]) is None
    assert worst_truncation([capped]) is capped
    assert worst_truncation([None, capped]) is capped
    assert worst_truncation([capped, memory]) is memory


def test_a_sector_exactly_on_the_energy_cut_is_not_admitted():
    """The window is open at the cut (``E_s - residual - E_min < cut``); the C2 injection ``<`` -> ``<=`` escaped."""
    cut = float(energy_cut(TAU))
    solved = [_rec(8, 0.0), _rec(9, cut), _rec(7, 5 * cut), _rec(10, 5 * cut)]
    plan = plan_ensemble(solved, TAU)
    assert [m.total for m in plan.members] == [8]


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("index", [3, 33])
def test_the_switched_off_ensemble_is_solve_ground_states_state(comm, index):
    """``mixed_valence=False`` runs the same loop with the winner alone: one member, weight exactly 1, and the
    energies ``solve_ground_state`` finds (it is the same walk and the same member solve)."""
    from impurityModel.test.support.gs_oracle import pinned_model

    h_op, imp, bath, N0, *_ = pinned_model(20261016, index)
    kw = dict(slaterWeightMin=1e-12, truncation_threshold=np.inf, verbose=False, comm=comm, tau=0.15)
    seen = []

    class Keep:
        def visit(self, member, state):
            seen.append(np.array(state.es))
            return member.total

    result = groundstate.solve_charge_sector_ensemble(
        h_op, imp, bath, dict(N0), mixed_valence=False, visitor=Keep(), **kw
    )
    reference = groundstate.solve_ground_state(h_op, imp, bath, dict(N0), **kw)
    assert len(result.members) == 1 and list(result.weights) == [1.0] and result.refused_reason is None
    assert len(seen) == 1 and seen[0] == pytest.approx(np.asarray(reference.es), abs=1e-9)
    assert result.visited == [sum(reference.occupation.values())]


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("mixed_valence", [False, True])
def test_only_the_ensemble_asks_the_walk_to_solve_edge_sectors(comm, mixed_valence, monkeypatch):
    """The walk is told to extend to the plan's missing edge sectors (``more_totals``) only when the ensemble is on;
    off solves the winner's neighbourhood and nothing more, so the switched-off run costs what ``solve_ground_state``
    costs."""
    from impurityModel.test.support.gs_oracle import pinned_model

    h_op, imp, bath, N0, *_ = pinned_model(20261016, 3)
    seen = []
    real = groundstate.find_ground_state_sector

    def spy(*args, **kwargs):
        seen.append(kwargs.get("more_totals"))
        return real(*args, **kwargs)

    monkeypatch.setattr(groundstate, "find_ground_state_sector", spy)

    class Keep:
        def visit(self, member, state):
            return member.total

    groundstate.solve_charge_sector_ensemble(
        h_op,
        imp,
        bath,
        dict(N0),
        mixed_valence=mixed_valence,
        visitor=Keep(),
        slaterWeightMin=1e-12,
        truncation_threshold=np.inf,
        verbose=False,
        comm=comm,
        tau=0.15,
    )
    assert len(seen) == 1
    assert (seen[0] is not None) is mixed_valence


def test_the_alpha_ce_shape_forms_three_members_though_every_sector_is_capped_and_uncertified():
    """V5.2: on a cap-bound metal every sector is cap-bound and uncertified; the ensemble used to never form."""
    tau = 0.0015
    solved = [
        _rec(1, 0.0, residual=5.7e-6, certified=False, cap_bound=True),
        _rec(0, 2.64 * tau, residual=5.7e-6, certified=False, cap_bound=True),
        _rec(2, 3.70 * tau, residual=5.7e-6, certified=False, cap_bound=True),
        _rec(3, 20 * tau, residual=5.7e-6, certified=False, cap_bound=True),
        _rec(-1, 20 * tau, residual=5.7e-6, certified=False, cap_bound=True),
    ]
    plan = plan_ensemble(solved, tau)
    assert sorted(m.total for m in plan.members) == [0, 1, 2] and plan.refused_reason is None


def test_the_fe_shape_forms_two_members_and_its_interval_is_wide():
    """The residual (2.3e-3) is 0.9 tau: the neighbour at 0.82 tau is admitted and the interval says how little the
    weights are worth."""
    from impurityModel.ed.sector_ensemble import ENSEMBLE_RESIDUAL_FACTOR, weight_interval

    tau = 0.0025
    res = 2.3e-3
    solved = [
        _rec(6, 0.0, res, False, True),
        _rec(7, 0.82 * tau, res, False, True),
        _rec(5, 40 * tau, res, False, True),
        _rec(8, 40 * tau, res, False, True),
    ]
    plan = plan_ensemble(solved, tau)
    assert sorted(m.total for m in plan.members) == [6, 7]
    lo, hi = weight_interval([6, 7], [0.0, 0.82 * tau], [1.0, 1.0], tau, [ENSEMBLE_RESIDUAL_FACTOR * res] * 2)
    assert 6.0 <= lo < hi <= 7.0 and hi - lo > 0.3
