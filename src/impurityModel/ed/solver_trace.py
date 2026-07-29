"""Opt-in accounting of where a solver run spends its wall clock.

The double-counting search evaluates its observable dozens of times, and each evaluation runs a
whole ground-state occupation walk underneath: several *sector solves*, each of which builds a
basis, expands it with CIPSI and diagonalizes it. Making that search affordable means knowing
which of those three costs dominates and how many of them are redundant -- neither of which is
visible from the outside, because the search only ever reports its final ``dc``.

This module is the recorder. It is a leaf: it imports nothing from the package, so any layer may
write into it (:func:`groundstate.calc_energy` and the criteria in ``double_counting`` both do).

Nothing is recorded unless a :func:`tracing` block is open, and the hooks are one ``is None``
test when it is not, so instrumented code pays nothing on the production path::

    with solver_trace.tracing() as trace:
        dc = fixed_occupation_dc(...)
    print(trace.summary())

Events carry whatever fields the call site passes plus the fields of every enclosing
:func:`labelled` block, which is how a ``sector_solve`` recorded three frames deep inside
``find_ground_state_basis`` still knows which trial ``mu`` it belongs to.

Traces are **rank-local**: every rank times its own work and no collective is issued from here.
A caller that wants a cross-rank comparison (the sector-solve count *must* agree, or the ranks
have taken different paths through the collectives) reduces :meth:`Trace.count` itself.
"""

from contextlib import contextmanager
from time import perf_counter

__all__ = ["Trace", "labelled", "note", "timed", "tracing"]

_ACTIVE = None
_LABELS: list[dict] = []


class Trace:
    """A flat list of recorded events, in the order they completed.

    Each event is a plain dict with at least ``kind`` and ``seconds``, merged with the fields of
    the enclosing :func:`labelled` blocks. Flat rather than nested because every question asked of
    it so far is an aggregate ("total expansion seconds", "sector solves per mu"), and a list of
    dicts answers those without a tree walk.
    """

    def __init__(self):
        self.events: list[dict] = []

    def add(self, kind, seconds=0.0, **fields):
        event = {"kind": kind, "seconds": seconds}
        for labels in _LABELS:
            event.update(labels)
        event.update(fields)
        self.events.append(event)
        return event

    def of_kind(self, kind):
        return [event for event in self.events if event["kind"] == kind]

    def count(self, kind):
        return len(self.of_kind(kind))

    def seconds(self, kind):
        """Total wall clock of one kind of event.

        Kinds nest (a ``sector_solve`` contains ``expand`` and ``eigensolve``), so summing across
        kinds double-counts. Compare within a kind, or against the enclosing kind's total.
        """
        return sum(event["seconds"] for event in self.of_kind(kind))

    def group_by(self, field):
        """Events bucketed by one label, e.g. ``group_by("mu")``. Missing label -> ``None`` key."""
        groups: dict = {}
        for event in self.events:
            groups.setdefault(event.get(field), []).append(event)
        return groups

    def summary(self):
        """Per-kind ``(count, seconds)``, for a one-line report."""
        return {kind: (self.count(kind), self.seconds(kind)) for kind in dict.fromkeys(e["kind"] for e in self.events)}


@contextmanager
def tracing(trace=None):
    """Collect events into ``trace`` (a fresh :class:`Trace` by default) for the duration.

    Re-entrant only in the sense that an inner block replaces the outer one and restores it on
    exit; events are never written to two traces at once. The label stack is reset on entry so a
    trace opened inside a labelled block does not inherit stale labels.
    """
    global _ACTIVE, _LABELS
    previous, previous_labels = _ACTIVE, _LABELS
    _ACTIVE = Trace() if trace is None else trace
    _LABELS = []
    try:
        yield _ACTIVE
    finally:
        _ACTIVE, _LABELS = previous, previous_labels


@contextmanager
def labelled(**fields):
    """Merge ``fields`` into every event recorded inside the block (e.g. ``mu=0.14``)."""
    if _ACTIVE is None:
        yield
        return
    _LABELS.append(fields)
    try:
        yield
    finally:
        _LABELS.pop()


@contextmanager
def timed(kind, **fields):
    """Record one event of ``kind`` spanning the block.

    Yields a dict the block may write into -- fields added there are recorded alongside the
    duration, which is how ``calc_energy`` reports the determinant count its expansion reached
    without measuring it twice.
    """
    if _ACTIVE is None:
        yield {}
        return
    # Bind the trace at entry, not at exit: the event belongs to the trace that was open when the
    # block started, and a block that raises must still land in it.
    trace = _ACTIVE
    extra: dict = {}
    start = perf_counter()
    try:
        yield extra
    finally:
        trace.add(kind, perf_counter() - start, **{**fields, **extra})


def note(kind, **fields):
    """Record a zero-duration event -- something that *happened*, e.g. a cache hit."""
    if _ACTIVE is None:
        return
    _ACTIVE.add(kind, 0.0, **fields)


def is_active():
    """Whether a :func:`tracing` block is open. For call sites that would do real work to report."""
    return _ACTIVE is not None
