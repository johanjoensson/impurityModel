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
