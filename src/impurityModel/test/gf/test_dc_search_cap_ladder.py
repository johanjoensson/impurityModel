"""Unit tests for :func:`impurityModel.ed.dc_search.calibrate_truncation_threshold`.

The DC criteria's own sector expansions saturate whatever determinant cap they are given (the
``GS_DE2_MIN`` selection does not converge in size, it keeps admitting until the budget stops
it), so cost is linear in the cap while the criterion's own answer is not. Ported from branch
``DC_gap_perf`` (``f61bfaa``/``91109b8``, ``doc/plans/dc_smo_performance.md``'s Phase 5) as
pure-Python synthetic-sequence tests: what is under test is the stopping rule, which has nothing
to do with what a rung costs, so no eigensolver or MPI communicator is involved here.
"""

import pytest

from impurityModel.ed import config
from impurityModel.ed.dc_search import (
    CAP_CONVERGENCE_FRACTION,
    CAP_CONVERGENCE_RUNS,
    _cap_ladder_max_rungs,
    _cap_ladder_start,
    calibrate_truncation_threshold,
)

#: The ladder's shipped first rung. Read from the knob's ``default``, never from
#: ``_cap_ladder_start()``: that reads the environment, and the whole point of the fixture below
#: is that these tests describe the *shipped* ladder rather than whatever the shell exports.
CAP_LADDER_START = config.DC_CAP_LADDER_START.default


@pytest.fixture(autouse=True)
def _shipped_ladder_bounds(monkeypatch):
    """Run every test in this module against the declared defaults, not the ambient environment.

    Both ends of the ladder became environment knobs (:data:`config.DC_CAP_LADDER_START` /
    ``DC_CAP_LADDER_MAX_RUNGS``), read lazily on every call -- which is what an operator wants,
    and which silently made this module's expectations follow the shell. Exporting the very knob
    these tests cover broke 7 of them: ``DC_CAP_LADDER_MAX_RUNGS=3`` cuts the ladder short, so
    every test asserting on a rung sequence longer than three fails for a reason that has nothing
    to do with the stopping rule under test.

    Cleared rather than pinned to a literal, so the defaults stay declared in exactly one place
    (:mod:`impurityModel.ed.config`) and a deliberate change to them shows up here as a test
    failure rather than being masked by a second copy. The one test that *is* about the knobs
    sets them itself, and ``monkeypatch`` layers over this fixture's own removal.
    """
    monkeypatch.delenv("DC_CAP_LADDER_START", raising=False)
    monkeypatch.delenv("DC_CAP_LADDER_MAX_RUNGS", raising=False)


def _ladder(values):
    """A ``quantity(cap)`` returning ``values`` in order, plus the caps it was asked for."""
    seen = []

    def quantity(cap):
        seen.append(cap)
        return values[min(len(seen) - 1, len(values) - 1)]

    return quantity, seen


def test_the_cap_ladder_doubles_and_stops_when_the_window_stops_moving():
    # The last three values span 0, so the ladder stops at the rung that completed that window.
    quantity, seen = _ladder([0.0, 1.0, 2.0, 2.0, 2.0])
    cap, drift, rungs, _status = calibrate_truncation_threshold(quantity, tol=1.0, memory_cap=10**6)

    assert seen == [CAP_LADDER_START * 2**i for i in range(5)], seen
    # The accepted cap is the last one *evaluated*, not one the ladder merely reasoned about.
    assert cap == CAP_LADDER_START * 2**4
    assert drift == pytest.approx(0.0)
    assert len(rungs) == CAP_CONVERGENCE_RUNS + 3


def test_the_returned_cap_was_actually_evaluated():
    """The ladder must never return a cap it did not measure.

    A version that doubles ``cap`` at the end of its final iteration and returns *that* would
    hand back one doubling past the largest rung it ever ran -- an internally inconsistent
    triple, with ``cap`` describing one rung and ``drift`` another. Reachable in production,
    where the memory budget (~1e6-1e7) is far above the top of an eight-rung ladder from 500, so
    the ``cap >= memory_cap`` break never fires and only the rung-budget exit is reached.
    """
    # Never settles, and never reaches the memory ceiling: the max-rungs exit, which is the one
    # test_the_ladder_never_proposes_a_cap_the_machine_could_not_run cannot reach.
    quantity, _seen = _ladder([float(i) for i in range(20)])
    cap, _drift, rungs, _status = calibrate_truncation_threshold(quantity, tol=1e-9, memory_cap=10**7)

    assert cap in dict(rungs), f"returned an unevaluated cap {cap}; measured {[c for c, _ in rungs]}"
    assert cap == rungs[-1][0]


def test_a_staircase_within_the_pairwise_gate_does_not_stop_the_ladder():
    """A pairwise gate bounds nothing against the limit; the span over the window does.

    Every step here is exactly the pairwise target, so a "two consecutive agreements" rule
    accepts at the first opportunity while the quantity keeps walking away -- after eight rungs
    it has moved 7x the gate. That is the *expected* shape for a selection that does not converge
    in size (``E(cap) ~ E_inf + A cap**-p`` with small ``p``), not a contrived one.
    """
    tol = 1.0
    step = CAP_CONVERGENCE_FRACTION * tol
    quantity, seen = _ladder([i * step for i in range(20)])
    _cap, drift, rungs, _status = calibrate_truncation_threshold(quantity, tol=tol, memory_cap=10**7)

    assert len(seen) > 3, "accepted a staircase whose every step sits exactly on the gate"
    assert drift > CAP_CONVERGENCE_FRACTION * tol
    assert len(rungs) == len(seen)


def test_a_single_agreeing_pair_does_not_stop_the_ladder():
    """The regression measured against a real cap ladder (nio_5peeled).

    The gap centre there runs -1.55e-3, -1.33e-3, -4.0e-5, +4.3e-4 over caps 500/1000/2000/8000:
    the first step is small enough to pass a quarter-of-tolerance target while the very next one
    is six times larger. Stopping on one agreement accepted cap 1000 there and would have landed
    5.9e-3 in mu from the cap-8000 answer -- 70% of the whole acceptance band, spent on noise.
    """
    quantity, seen = _ladder([-1.55, -1.33, -0.04, 0.43, 0.43, 0.43])
    cap, _drift, _rungs, _status = calibrate_truncation_threshold(quantity, tol=1.0, memory_cap=10**6)

    # 0.25 * tol = 0.25: the -1.55 -> -1.33 step (0.22) agrees, -1.33 -> -0.04 (1.29) does not.
    assert len(seen) > 2, "stopped on the first agreeing pair"
    assert cap >= CAP_LADDER_START * 8


def test_the_ladder_never_proposes_a_cap_the_machine_could_not_run():
    quantity, seen = _ladder([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    cap, _drift, _rungs, _status = calibrate_truncation_threshold(quantity, tol=1e-9, memory_cap=1200)

    assert max(seen) <= 1200 and cap <= 1200


def test_an_undefined_rung_between_two_agreements_breaks_the_run():
    """A ``None`` rung must RESET the evidence, not be skipped over.

    A counter-based rule that only reset on a *disagreement* would let an undefined rung sit
    between two agreeing pairs and certify convergence from two non-consecutive ones -- the same
    false-convergence class the multi-rung requirement exists to kill. Undefined rungs are not
    hypothetical: the gap centre is ``None`` whenever a low cap starves one of the ``N +- 1``
    sectors, which is exactly what the ladder's small first rungs can do.
    """
    quantity, seen = _ladder([1.0, 1.0, None, 1.0, 1.0])
    cap, _drift, rungs, _status = calibrate_truncation_threshold(quantity, tol=1.0, memory_cap=10**6)

    # The window is cleared by the None, so the two rungs after it are not enough on their own.
    assert len(seen) >= 5, seen
    assert cap in dict(rungs)


def test_an_undefined_rung_does_not_count_as_agreement():
    """``None`` means the criterion is undefined at that cap, not that it agreed with the last one.

    Treating it as a value would let two undefined rungs certify convergence on nothing.
    """
    quantity, seen = _ladder([None, None, 1.0, 1.0, 1.0])
    cap, _drift, _rungs, _status = calibrate_truncation_threshold(quantity, tol=1.0, memory_cap=10**6)

    assert len(seen) >= 5, seen
    assert cap is not None


def test_an_undefined_rung_resets_drift_along_with_the_window():
    """``drift`` must reset to ``None`` alongside ``window`` on a ``None`` rung, not keep
    reporting the span of a window that no longer exists.

    Found by review: a version that only ever overwrote ``drift`` under ``len(window) > 1`` left
    it holding the last *pre-reset* value. ``[1.0, 1.0, None, 5.0]`` has a real, unmeasured jump
    of 4.0 across the ``None`` -- accepting it (the ladder settles at rung 4, since only one
    value follows the reset and the window never refills to ``CAP_CONVERGENCE_RUNS + 1``) must
    not report the stale ``drift = 0.0`` left over from the ``[1.0, 1.0]`` window the ``None``
    wiped out.
    """
    # memory_cap chosen so the ladder's 4th rung (cap 500*2**3 = 4000) hits the ceiling and
    # breaks right there -- before the None's reset window could refill and settle "for real".
    quantity, _seen = _ladder([1.0, 1.0, None, 5.0])
    _cap, drift, rungs, _status = calibrate_truncation_threshold(quantity, tol=1.0, memory_cap=4000)

    assert len(rungs) == 4
    # Only rung 4 (value 5.0) survives the None's reset -- a window of one value has no span.
    assert drift is None, drift


def test_the_ladder_reports_the_memory_parity_verbosely(capsys):
    """The accepted-cap progress line is gated on ``verbose``, not printed unconditionally."""
    quantity, _seen = _ladder([0.0, 1.0, 2.0, 2.0, 2.0])
    calibrate_truncation_threshold(quantity, tol=1.0, memory_cap=10**6, verbose=False, rank=0)
    assert capsys.readouterr().out == ""

    quantity, _seen = _ladder([0.0, 1.0, 2.0, 2.0, 2.0])
    calibrate_truncation_threshold(quantity, tol=1.0, memory_cap=10**6, verbose=True, rank=0)
    out = capsys.readouterr().out
    assert "calibrated to" in out and "memory budget would have allowed" in out


def test_the_ladder_warns_unconditionally_when_it_does_not_settle(capsys):
    """Reaching the rung budget without settling is a WARNING -- unconditional, like the rest of
    this module's warnings (``_report_unattainable_target``), never gated behind ``verbose``."""
    quantity, _seen = _ladder([float(i) for i in range(20)])
    calibrate_truncation_threshold(quantity, tol=1e-9, memory_cap=10**7, verbose=False, rank=0)
    assert "WARNING" in capsys.readouterr().out


def test_the_ladder_bounds_are_environment_knobs_read_lazily(monkeypatch):
    """Both ends of the ladder come from the environment, per read -- not captured at import.

    The rung budget is what binds on a workload whose answer has not settled (the memory-derived
    cap is typically orders of magnitude above the ladder's reach), so an operator has to be able
    to move it for one run without editing the package: SrMnO3's production search stopped at
    64,000 under the old hard-coded budget of 8 with a memory budget 2000x larger.
    """
    monkeypatch.setenv("DC_CAP_LADDER_START", "4000")
    monkeypatch.setenv("DC_CAP_LADDER_MAX_RUNGS", "3")
    assert (_cap_ladder_start(), _cap_ladder_max_rungs()) == (4000, 3)

    quantity, seen = _ladder([float(i) for i in range(20)])
    cap, _drift, rungs, _status = calibrate_truncation_threshold(quantity, tol=1e-9, memory_cap=10**9)

    assert seen == [4000, 8000, 16000], seen
    assert cap == 16000 and len(rungs) == 3


def test_the_default_ladder_climbs_past_the_64000_that_stopped_srmno3():
    """Driven, not computed: run the shipped ladder and see where it actually gets to.

    The *reason* the pre-2026-09 ladder returned an unconverged SrMnO3 double counting is that its
    ceiling (500 * 2**7 = 64,000) sat below the caps that workload needs, while its memory budget
    allowed 1.35e8. Asserting ``start * 2**(rungs - 1) > 64_000`` would only restate the two
    defaults back to themselves and could not fail for any reason worth knowing; this drives the
    real loop on a quantity that never settles, with a memory cap far above the ceiling so the
    rung budget is what binds -- the same shape as the run that was cut short.
    """
    quantity, seen = _ladder([float(i) for i in range(50)])
    cap, _drift, rungs, status = calibrate_truncation_threshold(quantity, tol=1e-9, memory_cap=10**9)

    assert status == "rung_budget", "the memory cap must not be what stops this ladder"
    assert cap == seen[len(seen) - 1] == max(seen)
    assert cap > 64_000, f"the shipped ladder tops out at {cap}, at or below the old ceiling"
    assert seen == [config.DC_CAP_LADDER_START.default * 2**i for i in range(len(rungs))], seen


def test_the_ladder_names_which_of_its_three_exits_produced_the_cap():
    """``status``, because ``cap`` and ``drift`` do not determine it.

    A ladder that settled on its last rung and one that merely stopped there return the same cap
    and the same drift, so a caller cannot tell them apart after the fact -- and re-deriving the
    acceptance test downstream would put a second copy of it in the tree. All three exits are
    driven here from the same synthetic ladder, varying only what stops it.
    """
    # Settles: a constant sequence clears the span gate as soon as the window is full.
    quantity, _seen = _ladder([1.0] * 10)
    _cap, _drift, _rungs, status = calibrate_truncation_threshold(quantity, tol=1.0, memory_cap=10**7)
    assert status == "settled"

    # Never settles, and the memory ceiling is far away: the rung budget is what binds.
    quantity, _seen = _ladder([float(i) for i in range(30)])
    _cap, _drift, _rungs, status = calibrate_truncation_threshold(quantity, tol=1e-9, memory_cap=10**9)
    assert status == "rung_budget"

    # Never settles, and the memory cap bites first -- a distinct answer, because no rung budget
    # can lift it.
    quantity, _seen = _ladder([float(i) for i in range(30)])
    _cap, _drift, _rungs, status = calibrate_truncation_threshold(quantity, tol=1e-9, memory_cap=1200)
    assert status == "memory_cap"
