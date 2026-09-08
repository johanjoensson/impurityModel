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

from impurityModel.ed import dc_search, solver_trace
from impurityModel.ed.selfenergy import fixed_occupation_dc, fixed_peak_dc
from impurityModel.test.support import dc_diagnostics

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
    with solver_trace.tracing() as trace, pytest.raises(RuntimeError), solver_trace.timed("sector_solve"):
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
        with pytest.raises(RuntimeError, match="already active"), solver_trace.tracing():
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


def test_chi_is_the_slope_of_the_closest_pair_straddling_the_answer():
    """chi describes ``dc``, so it has to be measured *where the answer is*.

    A geometric scan leaves widely spaced early points and a tight final cluster, and the closest
    evaluated pair is not necessarily near the returned ``mu``: measured on a search that hit its
    target exactly at ``mu = 2.0``, the closest pair sat 2.4 units away and reported ``chi =
    0.0009``, i.e. "``dc`` undetermined to +-11" -- a statement about a place the answer is not.
    Containment is what makes the number local; because the returned ``mu`` is always itself an
    evaluated point, it degrades to "the two evaluated neighbours of the answer".
    """
    samples = {-4.0: 0.0, 0.0: 1.0, 1.0: 3.0, 1.5: 4.0}
    assert dc_search._dc_chi(samples, 1.25) == (pytest.approx(2.0), pytest.approx(0.5))
    # The answer at an endpoint of the tight pair still gets the tight pair, not the wide one.
    assert dc_search._dc_chi(samples, 1.0)[0] == pytest.approx(2.0)
    # ... and an answer out at -4.0 gets the pair that straddles *it*, wide as it is, rather than
    # the unrelated tight one.
    assert dc_search._dc_chi(samples, -4.0) == (pytest.approx(0.25), pytest.approx(4.0))
    # Nothing straddles a point outside the evaluated range: not resolvable, not the nearest pair.
    assert dc_search._dc_chi(samples, 9.0) == (None, None)
    # A pair narrower than the search's own bracket resolution measures a discontinuity's jump
    # over its width, never a slope.
    assert dc_search._dc_chi({1.0: 3.0, 1.0001: 9.0}, 1.0, width_tol=1e-3) == (None, None)
    # Nor is a difference taken across a charge-sector boundary a derivative of either branch.
    assert dc_search._dc_chi(samples, 1.25, in_sector=lambda mu: mu < 1.2) == (None, None)
    assert dc_search._dc_chi({0.0: 1.0}, 0.0) == (None, None)
    assert dc_search._dc_chi({}, 0.0) == (None, None)
    assert dc_search._dc_chi(samples, None) == (None, None)


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


# ---- the cap ladder's verdict grades `mu`, not the controlled quantity --------------------
#
# `_mu_verdict` replaced a verdict computed on `row["value"]` against a flat 1e-2. That was the
# wrong axis for every criterion: `value` is what the search *drives to its target* at each cap,
# so its spread is bounded by the search tolerance rather than by the truncation the ladder
# varies. On SMO it reported STABLE while `mu` moved 0.255 and changed sign twice. Each test
# below is checked against the specific wrong behaviour it guards, not just against the fix.


def _ladder(*pairs):
    """Rows carrying only what `_mu_verdict` reads: the returned shift and its own resolution."""
    return [{"mu": mu, "mu_tol": mu_tol} for mu, mu_tol in pairs]


def test_mu_verdict_drifts_on_the_smo_ladder_the_old_verdict_called_stable():
    """The regression this function exists for, with the real numbers.

    SMO gap ladder at caps 2000/8000/32000: gap centres -0.00207/-0.00106/0.00034 (spread
    0.0024, which the old flat 1e-2 gate graded STABLE) against mu 0.141796/0.208290/-0.046748
    (spread 0.255, 24x the loosest per-cap resolution).
    """
    rows = _ladder((0.141796, 0.0025 / 1.2797), (0.208290, 0.0025 / 0.4730), (-0.046748, 0.0025 / 0.2384))
    line = dc_diagnostics._mu_verdict(rows)
    assert "DRIFTS" in line, line
    assert "0.255" in line
    # The old gate would have passed: the controlled quantity really does sit inside 1e-2.
    values = [-0.00207, -0.00106, 0.00034]
    assert max(values) - min(values) <= 1e-2


def test_mu_verdict_is_stable_when_the_shift_settles_inside_the_resolution():
    line = dc_diagnostics._mu_verdict(_ladder((0.10, 5e-3), (0.1005, 5e-3)))
    assert "STABLE" in line and "DRIFTS" not in line, line


def test_mu_verdict_grades_against_the_loosest_resolution_not_the_tightest():
    """A drift inside the loosest per-cap band is not evidence of anything, so it is not graded
    as one. Picking the tightest band instead would manufacture DRIFTS on this ladder."""
    rows = _ladder((0.10, 1e-3), (0.106, 1e-2))
    line = dc_diagnostics._mu_verdict(rows)
    assert "STABLE" in line, line
    # Pin the reading this rules out, off the same rows: graded against the tightest band the
    # spread is 6x over and the verdict would flip.
    spread = max(row["mu"] for row in rows) - min(row["mu"] for row in rows)
    assert spread > min(row["mu_tol"] for row in rows)


def test_mu_verdict_refuses_to_grade_a_flat_slope_rather_than_calling_it_stable():
    """`mu_tol_effective` is inf when the criterion measured a genuinely zero slope. Comparing
    against inf grades *any* drift STABLE, which is the opposite of informative."""
    rows = _ladder((0.1, float("inf")), (0.9, float("inf")))
    line = dc_diagnostics._mu_verdict(rows)
    assert "UNGRADED" in line and "STABLE" not in line, line
    # Pin what a naive `spread <= band` would have concluded off these same rows: an infinite
    # band accepts an arbitrarily large drift, so every ladder would read STABLE.
    spread = max(row["mu"] for row in rows) - min(row["mu"] for row in rows)
    assert spread <= max(row["mu_tol"] for row in rows)


def test_mu_verdict_refuses_to_grade_a_criterion_that_reports_no_resolution():
    """`fixed_peak_dc` writes `tol` but no `mu_tol_effective`; inventing a band for it is how
    this harness got the verdict wrong in the first place."""
    line = dc_diagnostics._mu_verdict(_ladder((0.1, None), (0.9, None)))
    assert "UNGRADED" in line and "STABLE" not in line, line


def test_mu_verdict_needs_two_finite_shifts():
    assert "need two finite" in dc_diagnostics._mu_verdict(_ladder((0.1, 5e-3)))
    assert "need two finite" in dc_diagnostics._mu_verdict(_ladder((float("nan"), 5e-3), (0.3, 5e-3)))


def test_mu_verdict_ignores_a_single_missing_resolution_when_others_have_one():
    """One rung failing to report `mu_tol_effective` must not ungrade the whole ladder."""
    line = dc_diagnostics._mu_verdict(_ladder((0.10, None), (0.50, 5e-3)))
    assert "DRIFTS" in line, line
