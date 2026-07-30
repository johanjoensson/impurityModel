"""Tests for the double-counting cost accounting (``DC_DIAGNOSTICS`` / ``solver_trace``).

Phase 2 of the DC usability campaign: before anything is made faster, the search has to be able
to say where its time goes. The properties that matter are that the accounting is *free* when
off, that it counts the unit later phases will reduce (sector solves), and that it reports on
rank 0 only -- a per-rank report would bury a production log, and a per-rank *collective* would
be the very deadlock class the DC search was just fixed for.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import dc_search
from impurityModel.ed import solver_trace
from impurityModel.ed.selfenergy import fixed_occupation_dc, fixed_peak_dc

from .test_fixed_dc import common_kwargs


def test_hooks_are_inert_outside_a_tracing_block():
    """The production path must pay nothing: every hook short-circuits on a single None test."""
    assert not solver_trace.is_active()
    with solver_trace.timed("sector_solve") as fields:
        fields["n_dets"] = 7
    solver_trace.note("sector_cache_hit")
    with solver_trace.labelled(mu=1.0):
        solver_trace.note("sector_cache_hit")
    # Nothing was recorded anywhere, and nothing raised on the way through.
    with solver_trace.tracing() as trace:
        pass
    assert trace.events == []


def test_labels_attach_to_events_recorded_underneath():
    with solver_trace.tracing() as trace:
        with solver_trace.labelled(mu=0.25):
            with solver_trace.timed("sector_solve") as fields:
                fields["n_dets"] = 3
            solver_trace.note("sector_cache_hit")
        solver_trace.note("sector_cache_hit")

    labelled_events = [event for event in trace.events if "mu" in event]
    assert len(labelled_events) == 2 and all(event["mu"] == 0.25 for event in labelled_events)
    assert trace.count("sector_cache_hit") == 2
    assert trace.of_kind("sector_solve")[0]["n_dets"] == 3
    # The unlabelled event buckets under None, which is how the report skips events that belong
    # to no trial shift rather than mis-attributing them to one.
    assert set(trace.group_by("mu")) == {None, 0.25}


def test_a_raising_block_is_still_recorded():
    """A search that fails to reach its target is exactly the one whose cost matters."""
    with solver_trace.tracing() as trace:
        with pytest.raises(RuntimeError):
            with solver_trace.timed("sector_solve"):
                raise RuntimeError("boom")
    assert trace.count("sector_solve") == 1


def test_tracing_refuses_to_nest():
    """Nesting used to *divert* events, which reads downstream as zero measurements.

    The previous version of this test asserted the nesting semantics were correct. They were not:
    an inner block replaced the outer one, so `dc_diagnostics` -- which wraps a search in its own
    trace -- read zero events with `DC_DIAGNOSTICS=1`, reported `value = nan` for every row, and
    because rows are filtered on `np.isfinite` dropped its STABLE/DRIFTS verdict entirely while
    still printing a confident scaling exponent. Refusing is the smallest fix that cannot fail
    silently.
    """
    with solver_trace.tracing() as outer:
        with pytest.raises(RuntimeError, match="already active"):
            with solver_trace.tracing():
                pass
        # The refusal must not damage the enclosing trace.
        solver_trace.note("sector_cache_hit")
    assert solver_trace.is_active() is False
    assert outer.count("sector_cache_hit") == 1


def test_labels_do_not_outlive_their_block_and_the_trace_is_restored():
    """The two genuine invariants the old nesting test also happened to encode."""
    with solver_trace.tracing() as trace:
        with solver_trace.labelled(mu=1.0):
            solver_trace.note("sector_cache_hit")
        solver_trace.note("sector_cache_hit")
    labelled, unlabelled = trace.events
    assert labelled["mu"] == 1.0
    assert "mu" not in unlabelled
    # And the module is left with no active trace, so a later block starts clean.
    assert not solver_trace.is_active()


def test_chi_is_the_slope_of_the_closest_evaluated_pair():
    # A geometric scan leaves widely spaced early points and a tight final bracket; the slope
    # that matters for delta_mu = delta_n / chi is the local one, so the closest pair wins.
    samples = {-4.0: 0.0, 0.0: 1.0, 1.0: 3.0, 1.5: 4.0}
    assert dc_search._dc_chi(samples) == pytest.approx(2.0)
    assert dc_search._dc_chi({0.0: 1.0}) is None
    assert dc_search._dc_chi({}) is None


@pytest.mark.parametrize(
    "criterion, call, expected_slope",
    [
        ("fixed-occupation", lambda kw: fixed_occupation_dc(**kw, occupation=1.05), None),
        ("fixed-peak", lambda kw: fixed_peak_dc(**kw, peak_position=1.2), -1.0),
    ],
)
def test_the_search_reports_its_cost_when_the_knob_is_set(monkeypatch, capsys, criterion, call, expected_slope):
    """End to end: the report names the search, counts sector solves and measures the slope."""
    monkeypatch.setenv("DC_DIAGNOSTICS", "1")
    kwargs, _ = common_kwargs(v=0.4, tau=1e-3)
    # The fixed-occupation target here (1.05) sits on a charge-sector plateau (B3:
    # DoubleCountingUnreachable) -- irrelevant to this test, which only checks that the cost
    # report appears. _dc_search_trace reports from a `finally`, so the accounting is already
    # printed by the time the search raises.
    try:
        call(kwargs)
    except dc_search.DoubleCountingUnreachable:
        pass
    out = capsys.readouterr().out

    if MPI.COMM_WORLD.rank != 0:
        # Rank-0-only by design (see the report's own comment): a redundant non-root run of this
        # unmarked test must see nothing. Assert the absence rather than skipping, so the
        # rank-gating itself is what is under test.
        assert "cost accounting" not in out, out
        return

    assert f"{criterion} double-counting search: cost accounting" in out, out
    assert "sector solves" in out and "cache hits" in out, out
    # Every kind the later phases target has to be broken out separately, or a speedup cannot be
    # attributed: Phase 4 removes sector solves, Phase 5 removes expansions.
    for kind in ("build", "expand", "eigensolve"):
        assert kind in out, out
    # The peak criterion's observable is E_upper - E_lower, which a uniform shift moves with
    # slope exactly -1 (both sectors shift by -mu*n, differing by one electron). That the
    # measured slope reproduces it is the check that the samples are paired to the right mu.
    if expected_slope is not None:
        chi_line = next(line for line in out.splitlines() if "chi =" in line)
        measured = float(chi_line.split("=")[2].split("(")[0])
        assert measured == pytest.approx(expected_slope, abs=0.05), chi_line


def test_no_report_without_the_knob(capsys):
    kwargs, _ = common_kwargs(v=0.4, tau=1e-3)
    fixed_peak_dc(**kwargs, peak_position=1.2)
    assert "cost accounting" not in capsys.readouterr().out


@pytest.mark.mpi
def test_every_rank_runs_the_same_number_of_sector_solves():
    """The count is the cheapest witness that the ranks walked the same collective path.

    The residual is broadcast (P0-2), so the trial-mu sequence agrees by construction; a
    rank-dependent *cache hit* or occupation-bound test inside the walk would not, and would
    surface here as a count mismatch rather than as a hang somewhere downstream.
    """
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs at least 2 ranks")
    # No DC_DIAGNOSTICS here on purpose: the hooks record into whatever trace is open, so a
    # caller can account for a search without the search printing anything. Setting the knob
    # would make the criterion try to open its *own* trace, which now raises rather than
    # silently diverting this one (see test_tracing_refuses_to_nest).
    # occupation=1.05 sits on a charge-sector plateau (B3: DoubleCountingUnreachable) -- the
    # residual is broadcast, so every rank raises identically and skips the allgathers below
    # together (no partial-collective hang); this test only needs that every rank walked the
    # same trace before the raise, not that the search succeeded.
    with solver_trace.tracing() as trace:
        kwargs, _ = common_kwargs(v=0.4, tau=1e-3)
        try:
            fixed_occupation_dc(**kwargs, occupation=1.05)
        except dc_search.DoubleCountingUnreachable:
            pass
    counts = comm.allgather(trace.count("sector_solve"))
    assert len(set(counts)) == 1, counts
    assert counts[0] > 0
    mus = comm.allgather(sorted({event["mu"] for event in trace.events if "mu" in event}))
    assert all(np.allclose(m, mus[0]) for m in mus), mus
