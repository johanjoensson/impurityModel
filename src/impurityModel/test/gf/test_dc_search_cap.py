"""Unit tests for :func:`impurityModel.ed.dc_search.resolve_cap_at_max`.

The DC criteria's own sector expansions saturate whatever determinant cap they are given (the
``GS_DE2_MIN`` selection does not converge in size, it keeps admitting until the budget stops
it), so cost is linear in the cap while the criterion's own answer is not. Ported from branch
``DC_gap_perf`` (``f61bfaa``/``91109b8``, ``doc/plans/dc_smo_performance.md``'s Phase 5) as
pure-Python synthetic-sequence tests: what is under test is the stopping rule, which has nothing
to do with what a rung costs, so no eigensolver or MPI communicator is involved here.
"""

import pytest

from impurityModel.ed.dc_search import resolve_cap_at_max


def _ladder(values):
    """A ``quantity(cap)`` returning ``values`` in order, plus the caps it was asked for."""
    seen = []

    def quantity(cap):
        seen.append(cap)
        return values[min(len(seen) - 1, len(values) - 1)]

    return quantity, seen


# ---- run at the ceiling, ask whether it bound ------------------------


def test_an_unbound_expansion_needs_exactly_one_evaluation():
    """The whole point of the strategy: `truncation_report is None` means the expansion ran out of
    candidates above `de2_min`, not out of budget, so its basis is already the one any larger cap
    would build. No ladder rung can move that answer, so none is paid for."""
    seen = []
    cap, drift, rungs, status = resolve_cap_at_max(
        lambda c: (seen.append(c), 1.0)[1], 1e-2, memory_cap=512_000, bound_probe=lambda: False
    )
    assert seen == [512_000], "an unbound expansion must not be re-evaluated at any other cap"
    assert (cap, drift, status) == (512_000, 0.0, "unbound")
    assert rungs == [(512_000, 1.0)]


def test_a_bound_expansion_buys_one_half_rung_to_measure_the_drift():
    """When the cap *did* bind you learn "possibly unconverged" but not by how much -- so exactly
    one extra rung at half the cap puts a measured number on it. Two evaluations, not a ladder."""
    values = {512_000: 1.0, 256_000: 1.004}
    seen = []
    cap, drift, _rungs, status = resolve_cap_at_max(
        lambda c: (seen.append(c), values[c])[1], 1e-2, memory_cap=512_000, bound_probe=lambda: True
    )
    assert seen == [512_000, 256_000]
    assert cap == 512_000, "the accepted cap is the ceiling, not the half rung"
    assert drift == pytest.approx(0.004)
    assert status == "cap_bound"


def test_an_absent_probe_is_read_as_bound_not_as_converged():
    """`None` is "cannot tell", and the conservative reading costs one rung rather than claiming a
    convergence the run never established.

    Both criterion contexts now wire a real probe, so this is the *fallback* contract rather than
    any criterion's normal path: it is what a caller gets before anything has been solved, or if a
    future criterion uses this function without one."""
    for probe in (None, lambda: None):
        seen = []
        _cap, _drift, _rungs, status = resolve_cap_at_max(
            lambda c, seen=seen: (seen.append(c), 1.0)[1], 1e-2, memory_cap=8000, bound_probe=probe
        )
        assert status == "cap_bound", "an unprobed run must not report itself converged"
        assert len(seen) == 2


def test_the_cap_halves_when_the_ceiling_cannot_be_evaluated():
    """`memory_cap` is a prediction. A MemoryError retreats by halving rather than failing the
    search outright -- the graceful-degradation property the ascending ladder had for free."""
    seen = []

    def quantity(cap):
        seen.append(cap)
        if cap > 128_000:
            raise MemoryError("synthetic")
        return 1.0

    cap, _drift, _rungs, status = resolve_cap_at_max(quantity, 1e-2, memory_cap=512_000, bound_probe=lambda: False)
    assert seen[:3] == [512_000, 256_000, 128_000]
    assert cap == 128_000
    assert status == "memory_retreat", "a retreat must be reported, never silently absorbed"


def test_a_ceiling_unevaluable_even_at_the_floor_raises_rather_than_looping():
    with pytest.raises(MemoryError, match="floor"):
        resolve_cap_at_max(
            lambda c: (_ for _ in ()).throw(MemoryError("synthetic")),
            1e-2,
            memory_cap=8,
            bound_probe=lambda: False,
            min_cap=4,
        )
