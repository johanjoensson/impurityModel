"""Tests for the double-counting cost accounting (``DC_DIAGNOSTICS`` / ``solver_trace``).

Phase 2 of the DC usability campaign: before anything is made faster, the search has to be able
to say where its time goes. The properties that matter are that the accounting is *free* when
off, that it counts the unit later phases will reduce (sector solves), and that it reports on
rank 0 only -- a per-rank report would bury a production log, and a per-rank *collective* would
be the very deadlock class the DC search was just fixed for.
"""

from dataclasses import dataclass

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
    return [{"cap": 1000 * (i + 1), "mu": mu, "mu_tol": mu_tol} for i, (mu, mu_tol) in enumerate(pairs)]


def test_mu_verdict_drifts_on_the_smo_ladder_the_old_verdict_called_stable():
    """The regression this function exists for.

    SMO gap ladder at caps 2000/8000/32000 (doc/plans/dc_smo_performance.md): the measured mu
    values are 0.141796/0.208290/-0.046748, and the gap centres -0.00207/-0.00106/0.00034 --
    a 0.0024 spread that the old flat 1e-2 gate graded STABLE.

    The resolutions here are `tol/|chi|` built from that run's recorded tol and chi columns,
    NOT its `mu_tol_effective`, which was not recorded per rung at the time and which for the
    gap criterion is `tol/(0.5*delta_sum)` -- a different slope estimator (see
    `_row_resolution`). Either is a resolution of the right order; the verdict arithmetic under
    test does not care which produced the band, and this test does not claim they are equal.
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


def test_mu_verdict_refuses_to_grade_when_no_rung_reports_a_resolution():
    """Inventing a band is how this harness got the verdict wrong in the first place.

    A row with neither `mu_tol` nor a usable `tol`/`chi` has nothing to be graded against. Its
    `mu` still counts toward the spread -- it is a well-determined answer whose precision is
    merely unknown -- but with no band anywhere the line refuses rather than picking a number.
    """
    line = dc_diagnostics._mu_verdict(_ladder((0.1, None), (0.9, None)))
    assert "UNGRADED" in line and "STABLE" not in line, line
    assert "0.800000" in line, line


def test_mu_verdict_falls_back_to_tol_over_chi_when_the_criterion_reports_no_mu_tol():
    """`fixed_peak_dc` records `tol` and `chi` but no `mu_tol_effective`. Refusing to grade a
    peak ladder forever would be a coverage regression against the verdict this replaced."""
    rows = [
        {"cap": 1000, "mu": 0.10, "mu_tol": None, "tol": 1e-3, "criterion_chi": -0.5},
        {"cap": 2000, "mu": 0.40, "mu_tol": None, "tol": 1e-3, "criterion_chi": -0.5},
    ]
    line = dc_diagnostics._mu_verdict(rows)
    assert "DRIFTS" in line and "tol/|chi|" in line, line
    # 1e-3/0.5 = 2e-3, and the spread is 0.3.
    assert "2.00e-03" in line, line


def test_the_fallback_uses_the_criterions_chi_not_the_harnesss_unfiltered_one():
    """The table's `chi` column is recomputed here with no `in_sector` predicate, so across a
    charge-sector boundary it is a jump over a vanishing width -- an arbitrarily large slope,
    hence an arbitrarily tight band, hence a manufactured DRIFTS. Only `criterion_chi` (measured
    by the criterion under its own sector predicate and width tolerance) may feed the fallback.
    """
    rows = [
        # A sector-crossing secant in the harness's column, 1000x the criterion's own slope.
        {"cap": 1000, "mu": 0.10, "mu_tol": None, "tol": 1e-3, "criterion_chi": -0.5, "chi": -500.0},
        {"cap": 2000, "mu": 0.1005, "mu_tol": None, "tol": 1e-3, "criterion_chi": -0.5, "chi": -500.0},
    ]
    line = dc_diagnostics._mu_verdict(rows)
    # criterion_chi gives a 2e-3 band, which the 5e-4 spread sits inside.
    assert "STABLE" in line and "2.00e-03" in line, line
    # The reading this rules out: the harness column would give 2e-6 and call it DRIFTS.
    assert 0.0005 > 1e-3 / 500.0


def test_mu_verdict_names_the_estimator_that_produced_the_band_not_every_estimator_present():
    """`band` is one rung's number. Joining every source reads as a single expression
    ("mu_tol_effective/tol/|chi|") and attributes the band to the wrong estimator."""
    rows = [
        {"cap": 1000, "mu": 0.10, "mu_tol": 1e-3},
        {"cap": 2000, "mu": 0.90, "mu_tol": None, "tol": 1e-3, "criterion_chi": -0.05},  # 2e-2, the max
    ]
    line = dc_diagnostics._mu_verdict(rows)
    assert "(tol/|chi|) = 2.00e-02" in line, line
    assert "mu_tol_effective/tol" not in line, line


def test_mu_verdict_names_the_cause_when_every_rung_is_flat():
    """The all-degenerate branch must say *why* nothing was gradable rather than reporting a
    bare shortfall, and must not claim the mu values were missing -- they were finite and were
    dropped for an unbounded resolution."""
    line = dc_diagnostics._mu_verdict(_ladder((0.1, float("inf")), (0.9, float("inf"))))
    assert "UNGRADED" in line, line
    assert "2 rung(s) dropped entirely for a measured zero slope" in line, line
    assert "finite mu" not in line, line


def test_mu_verdict_reports_a_nan_resolution_as_undefined_not_as_a_measured_zero_slope():
    """A NaN band is not a measurement; saying "measured zero slope" would assert one."""
    line = dc_diagnostics._mu_verdict(_ladder((0.1, float("nan")), (0.9, 5e-3)))
    assert "undefined (NaN)" in line, line
    assert "measured zero slope" not in line, line


def test_a_rung_degenerate_on_one_estimator_is_dropped_not_adjudicated():
    """`delta_sum == 0` makes the gap criterion's band infinite while `tol`/`criterion_chi` may
    still be finite. An earlier version preferred the finite one -- which silently resolved a
    disagreement `dc_criteria` deliberately *records* (`delta_sum_vs_chi`), and let that rung's
    mu back into the spread, exactly what the infinite-resolution drop exists to prevent."""
    row = {"cap": 1000, "mu": 0.1, "mu_tol": float("inf"), "tol": 1e-3, "criterion_chi": -0.5}
    resolution, source = dc_diagnostics._row_resolution(row)
    assert resolution == float("inf"), (resolution, source)
    assert "disagree" in source, source
    # The disagreement is surfaced, and the rung stays out of the spread.
    line = dc_diagnostics._mu_verdict([row, {"cap": 2000, "mu": 0.9, "mu_tol": 1e-3}])
    assert "disagrees" in line, line
    assert "need two rungs to grade, 1 left" in line, line
    # With no fallback at all the source carries no disagreement claim.
    assert dc_diagnostics._row_resolution({"mu_tol": float("inf")}) == (float("inf"), "mu_tol_effective")
    # And a NaN mu_tol measured nothing, so there is no degeneracy to disagree about -- labelling
    # it as a disagreement would print a claim to the operator that nothing supports.
    nan_row = {"mu_tol": float("nan"), "tol": 1e-3, "criterion_chi": -0.5}
    assert dc_diagnostics._row_resolution(nan_row)[1] == "mu_tol_effective"


def test_mu_verdict_needs_two_gradable_rungs():
    assert "need two rungs to grade, 1 left" in dc_diagnostics._mu_verdict(_ladder((0.1, 5e-3)))
    assert "need two rungs to grade, 1 left" in dc_diagnostics._mu_verdict(_ladder((float("nan"), 5e-3), (0.3, 5e-3)))


def test_the_short_ladder_message_counts_gradable_rungs_not_ones_with_a_resolution():
    """A rung reporting no resolution IS counted in the spread, so a message saying "need two
    rungs with a finite mu and a usable resolution" would misdescribe what is short."""
    line = dc_diagnostics._mu_verdict([{"cap": 1, "mu": 0.1, "mu_tol": float("inf")}, {"cap": 2, "mu": 0.9}])
    assert "need two rungs to grade, 1 left" in line, line
    assert "usable resolution" not in line, line


def test_mu_verdict_drops_an_unresolvable_rung_from_the_spread_not_only_from_the_band():
    """A rung with a measured zero slope may sit anywhere on its plateau, so counting its mu
    toward the spread while excluding it from the band would print DRIFTS and blame the cap."""
    rows = _ladder((0.10, 5e-3), (0.1005, 5e-3), (9.0, float("inf")))
    line = dc_diagnostics._mu_verdict(rows)
    assert "STABLE" in line, line
    assert "dropped entirely" in line, line
    # The reading this rules out: keeping that mu in the spread makes it 8.9, i.e. DRIFTS.
    assert max(row["mu"] for row in rows) - min(row["mu"] for row in rows) > 5e-3


def test_mu_verdict_guards_a_zero_resolution_instead_of_dividing_by_it():
    """print_ladder's last line, after a multi-hour benchmark whose rows are already flushed --
    a ZeroDivisionError inside the f-string would lose the whole summary."""
    line = dc_diagnostics._mu_verdict(_ladder((0.1, 0.0), (0.3, 0.0)))
    assert "UNGRADED" in line, line


def test_mu_verdict_ignores_a_single_missing_resolution_when_others_have_one():
    """One rung failing to report `mu_tol_effective` must not ungrade the whole ladder."""
    line = dc_diagnostics._mu_verdict(_ladder((0.10, None), (0.50, 5e-3)))
    assert "DRIFTS" in line, line


def test_run_dc_search_copies_the_criterions_resolution_into_the_row(monkeypatch):
    """The plumbing the verdict depends on, which the arithmetic tests above cannot see.

    `_mu_verdict` grades against `row["mu_tol"]`, which only exists because `run_dc_search`
    passes a `report` dict to whichever criterion runs and copies `mu_tol_effective` out of it.
    A revert of that -- or a rename of the record key -- would leave every test above green
    while every real ladder printed UNGRADED, so this drives `run_dc_search` itself with the
    archive load and the criterion stubbed out.
    """
    import numpy as _np

    from impurityModel.test.support import dc_diagnostics as diag

    dc_guess = _np.zeros((2, 2))

    class _Model:
        dc = None
        n_spin_orbitals = 4

    @dataclass
    class _Basis:
        truncation_threshold: object = None
        tau: float = 1e-3

    monkeypatch.setitem(diag.WORKLOADS, "_stub", "unused")
    monkeypatch.setattr(diag, "load_selfenergy_archive", lambda *a, **k: (_Model(), None, _Basis(), None, "stub"))
    monkeypatch.setattr(diag, "suggest_truncation_threshold", lambda *a, **k: 12345)

    def fake_gap_dc(**kwargs):
        # What a real criterion does: fill the caller's record, including the two fields the
        # verdict reads, and return the shifted dc.
        kwargs["report"].update({"tol": 2.5e-3, "tol_basis": "mu_resolution", "mu_tol_effective": 6.49e-3})
        solver_trace.note("dc_evaluation", mu=0.25, gap=1e-4)
        return dc_guess + 0.25 * _np.identity(2)

    monkeypatch.setattr(diag, "fixed_gap_dc", fake_gap_dc)

    caller_report = {"stale": "from a previous rung"}
    row = diag.run_dc_search("_stub", cap=1000, criterion="gap", gap_report=caller_report)

    assert row["mu"] == pytest.approx(0.25)
    assert row["mu_tol"] == pytest.approx(6.49e-3)
    assert row["tol"] == pytest.approx(2.5e-3)
    assert row["tol_basis"] == "mu_resolution"
    # And the caller's dict was refreshed, not layered over: a key from an earlier rung that this
    # rung did not write must be gone, or a rung that measured no slope would inherit one.
    assert "stale" not in caller_report
    assert caller_report["mu_tol_effective"] == pytest.approx(6.49e-3)


def test_run_dc_search_hands_each_rung_a_fresh_record(monkeypatch):
    """Two rungs through one shared `gap_report`: the second must not inherit the first's band.

    `dc_record.recording` fills the out-parameter with `report.update(record)` and never clears,
    and both criteria write `mu_tol_effective` only when a slope was measured -- so reusing the
    caller's dict would let a rung that measured none read back its predecessor's, and the
    verdict would grade against a band no rung produced.
    """
    import numpy as _np

    from impurityModel.test.support import dc_diagnostics as diag

    dc_guess = _np.zeros((2, 2))

    class _Model:
        dc = None
        n_spin_orbitals = 4

    @dataclass
    class _Basis:
        truncation_threshold: object = None
        tau: float = 1e-3

    monkeypatch.setitem(diag.WORKLOADS, "_stub", "unused")
    monkeypatch.setattr(diag, "load_selfenergy_archive", lambda *a, **k: (_Model(), None, _Basis(), None, "stub"))
    monkeypatch.setattr(diag, "suggest_truncation_threshold", lambda *a, **k: 12345)

    measured = iter([True, False])  # rung 1 measures a slope, rung 2 does not

    def fake_gap_dc(**kwargs):
        fields = {"tol": 2.5e-3}
        if next(measured):
            fields["mu_tol_effective"] = 6.49e-3
        kwargs["report"].update(fields)
        solver_trace.note("dc_evaluation", mu=0.25, gap=1e-4)
        return dc_guess + 0.25 * _np.identity(2)

    monkeypatch.setattr(diag, "fixed_gap_dc", fake_gap_dc)

    shared = {}
    first = diag.run_dc_search("_stub", cap=1000, criterion="gap", gap_report=shared)
    second = diag.run_dc_search("_stub", cap=2000, criterion="gap", gap_report=shared)

    assert first["mu_tol"] == pytest.approx(6.49e-3)
    # The bug this pins: with a shared dict, `second["mu_tol"]` came back as the first rung's.
    assert second["mu_tol"] is None


def _stub_archive(monkeypatch, diag):
    """Wire `run_dc_search`'s archive load out, leaving its own record/row assembly intact."""

    @dataclass
    class _Basis:
        truncation_threshold: object = None
        tau: float = 1e-3

    class _Model:
        dc = None
        n_spin_orbitals = 4

    monkeypatch.setitem(diag.WORKLOADS, "_stub", "unused")
    monkeypatch.setattr(diag, "load_selfenergy_archive", lambda *a, **k: (_Model(), None, _Basis(), None, "stub"))
    monkeypatch.setattr(diag, "suggest_truncation_threshold", lambda *a, **k: 12345)


def test_a_rung_that_raises_hands_back_its_own_record_not_the_previous_rungs(monkeypatch):
    """The stale-record bug, relocated onto the exception path by an earlier version of the fix.

    `dc_record.recording` fills its out-parameter from a `finally` so a criterion that raises
    still reports how far it got. Copying into the caller's dict *after* the search block undid
    that: a rung that raised left the caller holding the previous rung's record, which is the
    misattribution the fresh-dict change exists to prevent.
    """
    from impurityModel.test.support import dc_diagnostics as diag

    _stub_archive(monkeypatch, diag)
    calls = {"n": 0}

    def fake_gap_dc(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            kwargs["report"].update({"tol": 1.0, "mu_tol_effective": 1.0, "status": "rung1"})
            solver_trace.note("dc_evaluation", mu=0.25, gap=0.0)
            return 0.25 * np.identity(2)
        kwargs["report"].update({"tol": 2.0, "status": "unreachable-rung2"})
        raise dc_search.DoubleCountingUnreachable("rung 2 could not reach its target")

    monkeypatch.setattr(diag, "fixed_gap_dc", fake_gap_dc)

    shared = {}
    diag.run_dc_search("_stub", cap=1000, criterion="gap", gap_report=shared)
    assert shared["status"] == "rung1"
    with pytest.raises(dc_search.DoubleCountingUnreachable):
        diag.run_dc_search("_stub", cap=2000, criterion="gap", gap_report=shared)
    # Its own partial record, not rung 1's -- and rung 1's band must be gone, or a later reader
    # attributes a resolution to a rung that never measured one.
    assert shared["status"] == "unreachable-rung2"
    assert shared["tol"] == 2.0
    assert "mu_tol_effective" not in shared


def test_cap_ladder_forwards_gap_report_and_gap_offset(monkeypatch):
    """Both parameters existed on `cap_ladder` and were dropped: a caller asking for the record
    off a ladder run got an untouched dict, and a non-zero offset was silently ignored."""
    from impurityModel.test.support import dc_diagnostics as diag

    seen = []

    def fake_run(workload_key, cap, **kwargs):
        seen.append((cap, kwargs.get("gap_offset"), kwargs.get("gap_report")))
        if kwargs.get("gap_report") is not None:
            kwargs["gap_report"].clear()
            kwargs["gap_report"].update({"cap": cap})
        return {
            "workload": workload_key,
            "label": "stub",
            "criterion": "gap",
            "cap": cap,
            "production_cap": 0,
            "seconds": 1.0,
            "mu": 0.1 * cap,
            "value": 0.0,
            "chi": -1.0,
            "criterion_chi": -1.0,
            "tol": 1e-3,
            "tol_basis": "mu_resolution",
            "mu_tol": 1e-3,
            "evaluations": 1,
            "sector_solves": 1,
            "cache_hits": 0,
            "build_s": 0.0,
            "expand_s": 0.0,
            "eigensolve_s": 0.0,
            "max_dets": cap,
        }

    monkeypatch.setattr(diag, "run_dc_search", fake_run)
    report = {"stale": True}
    diag.cap_ladder("_stub", [10, 20], criterion="gap", gap_offset=0.75, gap_report=report, comm=None)

    assert [cap for cap, _off, _rep in seen] == [10, 20]
    assert all(offset == 0.75 for _cap, offset, _rep in seen), seen
    assert all(rep is report for _cap, _off, rep in seen), seen
    # And it holds the last rung's record alone.
    assert report == {"cap": 20}


def test_run_dc_search_drops_unresolved_evaluations_from_its_sample_map(monkeypatch):
    """A successful search can record an evaluation whose observable did not resolve.

    `gap_observable` writes `evaluation_fields["gap_centre"] = centre` even when
    `_gap_centre_at_mu` returned `None` at a shell edge, and `_solve_dc_shift` propagates that
    `None` and keeps searching. Unfiltered, such a point can be one of the two neighbours
    `_dc_chi` picks as straddling the answer, and the subtraction raises `TypeError` at the very
    end of a rung -- or `achieved` comes back `None` and breaks the table's float format.
    """
    from impurityModel.test.support import dc_diagnostics as diag

    _stub_archive(monkeypatch, diag)

    def fake_gap_dc(**kwargs):
        kwargs["report"].update({"tol": 2.5e-3, "chi": -0.5, "mu_tol_effective": 5e-3})
        # An unresolved neighbour on each side of the answer, which is what makes it the pair
        # `_dc_chi` would otherwise select.
        solver_trace.note("dc_evaluation", mu=0.20, gap_centre=None)
        solver_trace.note("dc_evaluation", mu=0.25, gap_centre=1e-4)
        solver_trace.note("dc_evaluation", mu=0.30, gap_centre=None)
        return 0.25 * np.identity(2)

    monkeypatch.setattr(diag, "fixed_gap_dc", fake_gap_dc)

    row = diag.run_dc_search("_stub", cap=1000, criterion="gap")
    assert row["value"] == pytest.approx(1e-4)
    assert row["mu"] == pytest.approx(0.25)
    # Formatting the row is where a None `value` would surface in production.
    assert "0.00010" in diag._format_row(row)
