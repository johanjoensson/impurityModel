"""The ranking cap of a sector certification (``GS_RANKING_CAP``): pass 1 ranks every class under a small cap,
pass 2 continues only the contenders under the full one."""

from types import SimpleNamespace

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import sector_certify
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.sector_certify import (
    RANK_FLOOR,
    CertificationJob,
    CertificationPlan,
    ClassVerdict,
    _cap_tail,
    _ranking_contenders,
    certification_plan,
    certify_sector,
    effective_ranking_cap,
)
from impurityModel.test.support.comms import COMMS
from impurityModel.test.symmetry.test_sector_certify import EXPAND, _hamiltonian, _seed


@pytest.mark.parametrize(
    "setting, cap, expected",
    [
        ("auto", 400_000, 50_000),
        ("auto", 100_000, 20_000),  # the floor
        ("auto", 15_000, None),  # the floor is above the cap: nothing to rank under
        ("off", 400_000, None),
        ("5000", 400_000, 5_000),
        ("5000", 4_000, None),  # never above the cap
        ("auto", float("inf"), None),
        ("auto", None, None),
    ],
)
def test_the_ranking_cap_follows_the_cap_in_force(monkeypatch, setting, cap, expected):
    monkeypatch.setenv("GS_RANKING_CAP", setting)
    assert effective_ranking_cap(cap) == expected


def test_a_class_stopped_at_the_cap_gets_a_margin_from_its_pt2_its_floor_and_its_trajectory():
    settled = [-1.0, -1.01, -1.015, -1.0175, -1.01875]  # drops 1e-2, 5e-3, 2.5e-3, 1.25e-3: ratio 0.5
    tail = _cap_tail(settled, {"residual_pt2": 4e-2, "discarded_de2_mass": 1e-2})
    assert tail == pytest.approx(2.0 * 5e-2)  # kappa * (residual + discarded) beats the extrapolation
    assert _cap_tail(settled, {"residual_pt2": 0.0}) == pytest.approx(
        max(RANK_FLOOR, 2.0 * 5e-3 * 0.9 / 0.1)  # the geometric tail of the largest recent drop
    )
    # Not monotone: it may fall as far again as it already has.
    bouncing = [-1.0, -1.2, -1.1, -1.3, -1.25]
    assert _cap_tail(bouncing, {"residual_pt2": 0.0}) == pytest.approx(0.25)
    # Too short to extrapolate: the same rule.
    assert _cap_tail([-1.0, -1.4], None) == pytest.approx(0.4)
    assert _cap_tail([], None) == RANK_FLOOR


def _capped(job, energy, tail, ranking_cap=100):
    return ClassVerdict(
        job, energy, ranking_cap, tail, False, False, ranked=True, lower=energy - tail, ranking_cap=ranking_cap
    )


def test_contenders_are_fixed_on_pass_1_energies_and_closed_under_ties():
    jobs = [CertificationJob(None, (k,), None, 1) for k in range(5)]
    verdicts = {
        0: ClassVerdict(jobs[0], -1.0, 50, 0.0, False, False),  # converged: the lowest
        1: _capped(jobs[1], -0.99, 0.02),  # lower = -1.01: reaches the lowest
        2: _capped(jobs[2], -0.9899, 0.001),  # lower = -0.9909: out by its own bound, but tied with 1
        3: _capped(jobs[3], -0.5, 0.001),  # far above
        4: ClassVerdict(jobs[4], -0.5, 50, 0.0, False, False, ranked=True, lower=-0.501),  # settled, not capped
    }
    found = _ranking_contenders(verdicts, [1, 2, 3, 4], reach=0.0)
    assert found == {1, 2}
    assert _ranking_contenders({0: verdicts[0], 4: verdicts[4]}, [4], 0.0) == set()


def test_pass_2_continues_a_capped_class_by_its_frozen_verdict_and_not_by_the_falling_lowest_energy(monkeypatch):
    """Class 1 reaches the lowest energy at pass 1; class 2 does not, but is tied with class 1 within the error
    bars. When class 1 is continued it falls by 0.03, and class 2's bound then misses the new lowest energy by
    far -- the frozen verdict still continues it (a degenerate level is continued whole or not at all)."""
    jobs = [CertificationJob(None, (k,), None, 1) for k in range(4)]
    plan = CertificationPlan(jobs, {}, [], False, {})
    calls = []

    def fake_expand_job(h_op, basis, plan, job, finished, *args, rank=False, keys=None, window=None, **_):
        k = jobs.index(job)
        calls.append((k, rank))
        if rank and k == 0:
            verdict = ClassVerdict(job, -1.0, 50, 0.0, False, False)
        elif rank and k == 1:
            verdict = _capped(job, -0.99, 0.02)
        elif rank and k == 2:
            verdict = _capped(job, -0.9899, 0.001)
        elif rank:
            verdict = _capped(job, -0.5, 0.001)
        else:  # pass 2: class 1 (and its partner) converge 0.03 lower
            verdict = ClassVerdict(job, -1.03, 200, 0.0, False, False)
        return verdict, np.array([verdict.energy]), None, SimpleNamespace(_keys={}, weighted_restrictions=None), None

    monkeypatch.setattr(sector_certify, "_expand_job", fake_expand_job)
    basis = SimpleNamespace(is_distributed=False, comm=None, weighted_restrictions=None, truncation_threshold=1000)
    h_op = SimpleNamespace(set_weighted_restrictions=lambda w: None)
    certify_sector(h_op, basis, plan, expand_kwargs={}, dense_cutoff=1000, slater_weight_min=0, ranking_cap=100)
    assert sorted(k for k, ranking in calls if not ranking) == [1, 2], calls
    assert all(ranking for k, ranking in calls if k == 3) and (3, False) not in calls


@pytest.mark.parametrize("comm", COMMS)
def test_pass_1_stops_at_the_ranking_cap_pass_2_reaches_the_full_cap_and_the_winner_carries_it(comm, monkeypatch):
    h_op = _hamiltonian()
    basis = _seed(h_op, comm)
    plan = certification_plan(h_op, basis)
    cap = basis.size + 1  # the two-orbital model closes at 8 determinants
    seen = []
    real = sector_certify._expand_job

    def spy(*args, rank=False, **kwargs):
        out = real(*args, rank=rank, **kwargs)
        seen.append((rank, out[0]))
        return out

    monkeypatch.setattr(sector_certify, "_expand_job", spy)
    es, _psis, winner, _solver, verdicts = certify_sector(
        h_op, basis, plan, expand_kwargs=EXPAND, dense_cutoff=1000, slater_weight_min=0, ranking_cap=cap
    )
    pass_1 = [v for ranking, v in seen if ranking]
    assert pass_1 and all(v.n_dets <= cap for v in pass_1), [v.n_dets for v in pass_1]
    assert any(v.ranking_cap == cap for v in pass_1), "no class was stopped by the ranking cap"
    # The winner's verdict is a pass-2 one, past the ranking cap, and its basis carries the caller's cap.
    winning = min(verdicts, key=lambda v: v.energy)
    assert winning.ranking_cap is None and not winning.ranked
    assert winning.n_dets > cap
    assert winner.truncation_threshold == basis.truncation_threshold
    uncapped = certify_sector(h_op, basis, plan, expand_kwargs=EXPAND, dense_cutoff=1000, slater_weight_min=0)[0]
    assert float(np.min(es)) == pytest.approx(float(np.min(uncapped)), abs=1e-9)


@pytest.mark.mpi
@pytest.mark.parametrize("cap, ranking_cap", [(700, 100), (894, 100)])
def test_a_cubic_d6_level_keeps_all_fifteen_states_when_its_classes_are_ranked_under_a_small_cap(
    monkeypatch, cap, ranking_cap
):
    from impurityModel.ed.eigensolvers import Within
    from impurityModel.ed.groundstate import solve_sector
    from impurityModel.test.support.cubic_d_shell import cubic_d_shell

    monkeypatch.setenv("GS_RANKING_CAP", str(ranking_cap))
    _h, h_op, imp, baths = cubic_d_shell(n_bath_sets=1, real_cubic=True)
    pass_1 = []
    real = sector_certify._expand_job

    def spy(*args, rank=False, **kwargs):
        out = real(*args, rank=rank, **kwargs)
        if rank:
            pass_1.append(out[0])
        return out

    monkeypatch.setattr(sector_certify, "_expand_job", spy)
    admitted = []
    select = CIPSISolver.determine_new_Dj

    def select_spy(self, *args, **kwargs):
        if self.stop_at_cap:
            admitted.append((kwargs.get("max_new"), self.basis.truncation_threshold - self.basis.size))
        return select(self, *args, **kwargs)

    monkeypatch.setattr(CIPSISolver, "determine_new_Dj", select_spy)
    result = solve_sector(
        h_op,
        imp,
        baths,
        {0: 6},
        tau=0,
        chain_restrict=False,
        dense_cutoff=1000,
        manifold=Within(1e-6),
        comm=MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None,
        verbose=False,
        slaterWeightMin=0,
        de2_min=0,
        e_pt2_tol=1e-8,
        truncation_threshold=cap,
    )
    assert any(v.ranking_cap == ranking_cap for v in pass_1), "the ranking cap never stopped a class"
    # A ranking expansion never admits past its cap, so nothing is truncated and the last selection round is a
    # standard PT2 estimate of the basis the class is ranked on.
    assert admitted and all(max_new <= room for max_new, room in admitted), admitted
    assert all(v.n_dets <= ranking_cap for v in pass_1 if v.ranking_cap is not None)
    check = result.solver.manifold_check
    assert check["found"] == check["expected"] == 15, check
    es = np.sort(np.real(result.es))
    assert np.ptp(es[:15]) < 1e-9 * max(1.0, abs(es[0])), np.ptp(es[:15])
    assert result.solver.basis.truncation_threshold == cap


def test_a_pass_2_solve_replaces_the_pass_1_solve_of_the_same_class_even_when_it_is_not_lower(monkeypatch):
    """The ranking-cap basis is small; the continued one is the basis a caller grows from, so it is returned even if
    its energy came out a hair above the capped one's (noise on a basis that cannot be worse variationally)."""
    jobs = [CertificationJob(None, (0,), None, 1)]
    plan = CertificationPlan(jobs, {}, [], False, {})

    def fake_expand_job(h_op, basis, plan, job, finished, *args, rank=False, keys=None, window=None, **_):
        verdict = _capped(job, -1.0, 0.02) if rank else ClassVerdict(job, -1.0 + 1e-10, 200, 0.0, False, False)
        marker = SimpleNamespace(_keys={}, weighted_restrictions=None, truncation_threshold=100 if rank else 1000)
        marker.pass_2 = not rank
        return verdict, np.array([verdict.energy]), None, marker, None

    monkeypatch.setattr(sector_certify, "_expand_job", fake_expand_job)
    basis = SimpleNamespace(is_distributed=False, comm=None, weighted_restrictions=None, truncation_threshold=1000)
    h_op = SimpleNamespace(set_weighted_restrictions=lambda w: None)
    _es, _psis, winner, _solver, verdicts = certify_sector(
        h_op, basis, plan, expand_kwargs={}, dense_cutoff=1000, slater_weight_min=0, ranking_cap=100
    )
    assert winner.pass_2 and winner.truncation_threshold == 1000
    assert verdicts[0].ranking_cap is None


def test_a_capped_partner_of_a_converged_lowest_class_is_continued_with_it():
    """The converged lowest class anchors the tie closure: a capped class whose bound misses the floor but which
    is tied with it within the error bars is a degenerate partner and must be continued (review finding 2)."""
    jobs = [CertificationJob(None, (k,), None, 1) for k in range(2)]
    verdicts = {
        0: ClassVerdict(jobs[0], -1.0, 50, 1e-4, False, False),
        1: _capped(jobs[1], -0.9999, 1e-6),  # lower = -0.999901: misses the floor by more than its own bar
    }
    assert verdicts[1].lower - (-1.0) > sector_certify._bound_error(
        verdicts[1]
    )  # the first test alone would not select it
    assert _ranking_contenders(verdicts, [1], reach=0.0) == {1}


@pytest.mark.parametrize("setting", ["20k", "-5", "banana", "1e-3"])
def test_a_malformed_ranking_cap_raises_and_the_cap_line_survives_it(monkeypatch, setting):
    from types import SimpleNamespace as NS

    from impurityModel.ed.memory_estimate import describe_gs_cap

    monkeypatch.setenv("GS_RANKING_CAP", setting)
    with pytest.raises(ValueError, match="GS_RANKING_CAP"):
        effective_ranking_cap(1_000_000)
    # Rank 0 alone prints the line: it must not raise there while the other ranks carry on.
    assert "ranked" not in describe_gs_cap(NS(source="auto", gs=1_000_000))


def test_the_cap_line_names_the_ranking_cap(monkeypatch):
    from types import SimpleNamespace as NS

    from impurityModel.ed.memory_estimate import describe_gs_cap

    monkeypatch.setenv("GS_RANKING_CAP", "auto")
    assert "first ranked at 125,000" in describe_gs_cap(NS(source="auto", gs=1_000_000))
    monkeypatch.setenv("GS_RANKING_CAP", "off")
    assert "ranked" not in describe_gs_cap(NS(source="auto", gs=1_000_000))
