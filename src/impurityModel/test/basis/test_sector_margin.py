"""The walk records every sector it solved and how firmly its winner beats the neighbours (review P11, MV 2).

Recorded only: no winner changes. ``SectorSearch.solved`` is the walk's solved set, one :class:`SectorRecord` per
sector in solve order; ``SectorSearch.diagnostics`` is the winner's margin to ``N +- 1`` in units of the two
energies' error bars, and the second difference of ``E(N)``.
"""

import math

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.groundstate import (
    SectorEnergy,
    SectorRecord,
    find_ground_state_sector,
    sector_diagnostics,
)
from impurityModel.test.support.comms import COMMS

IMPURITY = {0: [[0, 1, 2, 3]]}
BATHS: tuple = ({0: [[]]}, {0: [[]]})


def _rec(total, energy, residual=0.0, certified=True):
    return SectorRecord({0: total}, energy, residual, certified, False, None)


def test_a_clear_winner_has_an_infinite_margin_when_both_are_exact():
    solved = [_rec(1, -1.0), _rec(2, -3.0), _rec(3, -2.0)]
    d = sector_diagnostics(solved, 2)
    assert d.neighbour_totals == (1, 3)
    assert d.margin == math.inf
    assert d.second_difference == pytest.approx(-2.0 + 6.0 - 1.0)
    assert not d.undecided and not d.unknown_error


def test_the_margin_is_the_gap_in_units_of_the_two_error_bars():
    solved = [_rec(1, -1.0, 0.1), _rec(2, -1.5, 0.2), _rec(3, -1.2, 0.05)]
    d = sector_diagnostics(solved, 2)
    # gaps 0.5 and 0.3 over errors 0.3 and 0.25: the smaller ratio, 1.2
    assert d.margin == pytest.approx(0.3 / 0.25)
    assert not d.undecided


def test_a_gap_inside_the_error_bars_is_undecided():
    d = sector_diagnostics([_rec(1, -1.0, 1e-3), _rec(2, -1.0005, 1e-3)], 2)
    assert d.margin == pytest.approx(0.25)
    assert d.undecided
    assert d.second_difference is None, "only one neighbour was solved"


def test_a_lower_neighbour_is_negative_and_undecided():
    d = sector_diagnostics([_rec(2, -1.0, 1e-3), _rec(3, -2.0, 1e-3)], 2)
    assert d.margin < 0 and d.undecided
    d = sector_diagnostics([_rec(2, -1.0, 0.0), _rec(3, -2.0, 0.0)], 2)
    assert d.margin == -math.inf and d.undecided


def test_an_unknown_residual_is_said_so_and_counts_as_zero():
    d = sector_diagnostics([_rec(2, -1.0, None), _rec(3, -0.5, 0.1)], 2)
    assert d.unknown_error
    assert d.margin == pytest.approx(5.0)


def test_no_neighbour_means_no_margin_and_an_unsolved_winner_is_tolerated():
    assert sector_diagnostics([_rec(2, -1.0)], 2).margin is None
    assert sector_diagnostics([], 2).margin is None


def _walk(energies, residuals, comm=None, use_hf_seed=False, nominal=2, verbose=0):
    def sector_energy(trial):
        total = sum(trial.values())
        return SectorEnergy(energies[total], None, residuals.get(total, 0.0), True, False)

    return find_ground_state_sector(
        None,
        IMPURITY,
        BATHS,
        {0: nominal},
        chain_restrict=False,
        dense_cutoff=1,
        comm=comm,
        verbose=verbose,
        truncation_threshold=np.inf,
        use_hf_seed=use_hf_seed,
        sector_energy=sector_energy,
    )


ENERGIES = {0: 5.0, 1: -1.0, 2: -1.0005, 3: -0.2, 4: 3.0}


@pytest.mark.parametrize("comm", COMMS)
def test_the_walk_records_every_solved_sector_in_order_and_flags_the_close_neighbour(comm, capsys):
    search = _walk(ENERGIES, {1: 1e-3, 2: 1e-3, 3: 1e-3}, comm=comm, verbose=1)
    assert sum(search.occupation.values()) == 2
    totals = [rec.total for rec in search.solved]
    assert totals == list(dict.fromkeys(totals)), "a sector is recorded once"
    assert {1, 2, 3} <= set(totals), "the winner and both neighbours are in the solved set"
    by_total = {rec.total: rec for rec in search.solved}
    assert by_total[2].energy == ENERGIES[2] and by_total[2].residual_pt2 == 1e-3 and by_total[2].certified is True
    assert search.diagnostics.undecided
    assert search.diagnostics.margin == pytest.approx(0.25)
    out = capsys.readouterr().out
    # printed on rank 0 only (a double-counting search, which passes sector_energy, prints it from V_SUMMARY)
    assert ("not separated from its neighbour" in out) == (comm is None or comm.rank == 0)


@pytest.mark.parametrize("comm", COMMS)
def test_a_separated_winner_is_quiet_at_the_default_verbosity(comm, capsys):
    search = _walk(ENERGIES, {1: 1e-6, 2: 1e-6, 3: 1e-6}, comm=comm)
    assert not search.diagnostics.undecided
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("use_hf_seed", [False, True])
def test_both_branches_solve_the_neighbours_of_the_winner(use_hf_seed, monkeypatch):
    if use_hf_seed:
        import impurityModel.ed.groundstate as gs

        monkeypatch.setattr(gs, "hartree_fock_seed_occupation", lambda *a, **k: ({0: 3}, True))
    energies = {0: 5.0, 1: 0.0, 2: -2.0, 3: -1.0, 4: 3.0}
    search = _walk(energies, {}, use_hf_seed=use_hf_seed, nominal=3 if use_hf_seed else 2)
    assert sum(search.occupation.values()) == 2
    assert {1, 2, 3} <= {rec.total for rec in search.solved}
    assert search.diagnostics.neighbour_totals == (1, 3)


@pytest.mark.mpi
def test_every_rank_holds_the_same_diagnostics():
    comm = MPI.COMM_WORLD
    search = _walk(ENERGIES, {1: 1e-3, 2: 1e-3, 3: 1e-3}, comm=comm)
    gathered = comm.allgather((search.diagnostics, [rec.total for rec in search.solved]))
    assert all(g == gathered[0] for g in gathered)


def test_a_search_with_its_own_sector_solves_is_quiet_at_the_default_verbosity(capsys):
    """The double-counting search walks once per trial shift and sits at a crossover by construction."""
    _walk(ENERGIES, {1: 1e-3, 2: 1e-3, 3: 1e-3}, verbose=0)
    assert "WARNING" not in capsys.readouterr().out


def _undecided():
    solved = [_rec(1, -1.0, 1e-3), _rec(2, -1.0005, 1e-3)]
    return sector_diagnostics(solved, 2), solved[1]


@pytest.mark.parametrize("always, verbose, printed", [(True, 0, True), (False, 0, False), (False, 1, True)])
def test_report_margin_warns_at_every_verbosity_only_for_the_production_walk(always, verbose, printed, capsys):
    from impurityModel.ed.groundstate import _report_margin

    diagnostics, winner = _undecided()
    _report_margin(diagnostics, winner, verbose, always=always)
    assert ("not separated" in capsys.readouterr().out) == printed


def test_neighbour_gaps_are_in_units_of_tau():
    """P9: the gap to N +- 1 in tau, only for solved neighbours, empty without a positive tau."""
    from impurityModel.ed.groundstate import NEIGHBOUR_GAP_WARN, neighbour_gaps

    solved = [_rec(4, -1.0), _rec(3, -0.99), _rec(5, -0.5), _rec(6, 0.0)]
    gaps = neighbour_gaps(solved, 4, 0.005)
    assert gaps == pytest.approx({3: 2.0, 5: 100.0}) and set(gaps) == {3, 5}
    assert gaps[3] < NEIGHBOUR_GAP_WARN < gaps[5]
    assert neighbour_gaps(solved, 4, 0.0) == {} and neighbour_gaps(solved, 9, 0.005) == {}


def test_convergence_summary_names_residual_limit_and_margin():
    from impurityModel.ed.gs_report import convergence_summary

    assert convergence_summary(None) == "convergence: not recorded"
    line = convergence_summary(
        {
            "converged": False,
            "residual_pt2": 3e-6,
            "e_pt2_tol": 1e-8,
            "limited_by": "cap",
            "sector_diagnostics": {"winner_total": 8, "margin": 0.5, "undecided": True},
        }
    )
    assert "NOT converged" in line and "3.00e-06" in line and "limited by cap" in line
    assert "N_imp=8 (undecided)" in line
    ok = convergence_summary({"converged": True, "residual_pt2": 0.0, "e_pt2_tol": 1e-8})
    assert ok.startswith("convergence: converged;") and "limited" not in ok


def test_heartbeat_is_rate_limited_and_goes_to_stderr(capsys):
    from impurityModel.ed.utils import Heartbeat

    t = [0.0]
    hb = Heartbeat(interval=300.0, clock=lambda: t[0])
    assert not hb.beat("early")
    t[0] = 299.0
    assert not hb.beat("still early")
    t[0] = 301.0
    assert hb.beat("late")
    assert not hb.beat("just beat")
    out = capsys.readouterr()
    assert out.out == "" and "[still running] late" in out.err and "early" not in out.err


def test_degraded_warnings_only_name_unconverged_or_undecided():
    from impurityModel.ed.gs_report import degraded_warnings

    assert degraded_warnings(None) == []
    assert degraded_warnings({"converged": True, "sector_diagnostics": {"undecided": False}}) == []
    got = degraded_warnings(
        {
            "converged": False,
            "limited_by": "cap",
            "residual_pt2": 1e-5,
            "sector_diagnostics": {"undecided": True, "winner_total": 8},
        }
    )
    assert len(got) == 2 and "limited by cap" in got[0] and "N_imp=8" in got[1]
