r"""The root find both double-counting criteria share, and its cost accounting.

The double counting is parametrized as a uniform shift, ``dc(mu) = dc_guess + mu * identity``;
:func:`_solve_dc_shift` drives a caller-supplied scalar observable of ``mu`` onto a target with a
bidirectional geometric scan and safeguarded secant/bisection refinement. Nothing here knows what
the observable is -- :mod:`dc_criteria` supplies a peak position or an impurity occupation -- so
this module carries no ED machinery and imports no solver.

Two properties are load-bearing and easy to lose. The search assumes **no monotonicity**: the
residual can be a difference of two independently-monotone quantities, and today's criteria
re-select both the CIPSI space and the charge sector at every ``mu``, so the residual genuinely
is not monotone. And every branch is taken on a value produced by a *collective* observable that
is replicated only to roundoff, so the residual is broadcast once, in one place, before anything
reads it (CLAUDE.md's MPI rule: never gate a collective on rank-local state).

The reporting half answers "where did the search spend its time" through
:mod:`impurityModel.ed.solver_trace`, under the ``DC_DIAGNOSTICS`` knob.
"""

from contextlib import contextmanager

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config, solver_trace


def _refine_bracket(residual, mu_low, g_low, mu_high, g_high, tol, width_tol):
    r"""Safeguarded secant/bisection refinement of a bracket straddling a root.

    ``[mu_low, mu_high]`` must have residuals of opposite sign (``g_low`` and ``g_high``).
    A secant estimate that leaves the bracket, or hugs an endpoint, is replaced by the
    midpoint, guaranteeing a geometric decrease of the bracket width every step. This makes
    no assumption of monotonicity beyond the bracket invariant (opposite-sign endpoints),
    so it also handles a non-monotone residual with a single sign change in the bracket.

    Returns
    -------
    (mu, g) : tuple of float
        The best point found: the root itself, with ``|g| <= tol``, once met; otherwise the
        closer of the two endpoints of the fully narrowed bracket (a plateau/step in the
        residual, collapsed below ``width_tol`` without ever meeting ``tol``).
    """
    while mu_high - mu_low > width_tol:
        mu_mid = mu_high - g_high * (mu_high - mu_low) / (g_high - g_low) if g_high != g_low else np.inf
        # Safeguard: reject a secant estimate that leaves the bracket or hugs an endpoint,
        # keeping a guaranteed geometric decrease via bisection.
        margin = 0.01 * (mu_high - mu_low)
        if not (mu_low + margin <= mu_mid <= mu_high - margin):
            mu_mid = 0.5 * (mu_low + mu_high)
        g_mid = residual(mu_mid)
        if abs(g_mid) <= tol:
            return mu_mid, g_mid
        # Keep the sub-bracket that still straddles a root (opposite-sign endpoints);
        # correct regardless of whether the residual is monotone.
        if g_mid * g_low < 0:
            mu_high, g_high = mu_mid, g_mid
        else:
            mu_low, g_low = mu_mid, g_mid
    return (mu_low, g_low) if abs(g_low) <= abs(g_high) else (mu_high, g_high)


def _solve_dc_shift(
    observable,
    target,
    *,
    tol,
    width_tol,
    initial_step,
    max_shift,
    plateau_ok,
    unreachable_message,
    rank=0,
    comm=None,
):
    r"""Find the uniform shift ``mu`` that drives a scalar observable onto ``target``.

    Generic root-finder shared by :func:`fixed_peak_dc` and :func:`fixed_occupation_dc`. The
    double counting is parametrized as ``dc(mu) = dc_guess + mu * identity``; the caller passes an
    ``observable(mu)`` closure that builds ``dc(mu)``, solves the model and returns the scalar to
    control (the peak position or the impurity occupation). No monotonicity in ``mu`` is assumed:
    the residual ``observable(mu) - target`` can be a difference of two independently-monotone
    quantities (e.g. an interacting and a non-interacting occupation), which is not itself
    monotone.

    The search scans both directions from ``mu = 0`` in geometrically growing steps
    (``initial_step, 2*initial_step, ...``, evaluated in a fixed +/- order each level so every
    rank makes the same sequence of collective calls). At each level it checks both new points
    for a direct hit (``|residual| <= tol``, returned immediately) and for a bracket (a
    sign change against that direction's previous point); any brackets found at that level are
    refined right away, nearest-``mu=0``-first, by :func:`_refine_bracket` (a safeguarded secant
    step with bisection fallback). Only if every bracket found so far collapses without meeting
    ``tol`` (a plateau) does the scan grow to the next level -- so a well-behaved, near-``mu=0``
    root is found as cheaply as the old single-direction search, while a residual with a false
    near bracket (e.g. a non-monotone ``n(mu) - n0(mu)``) still finds a genuine root farther out.
    The scan stops once both directions have exceeded ``max_shift``.

    Parameters
    ----------
    observable : callable
        ``observable(mu) -> float``. Evaluated collectively (it runs the eigensolver); call it the
        same number of times on every rank.
    target : float
        Requested observable value.
    tol : float
        Convergence tolerance on ``|observable - target|``.
    width_tol : float
        Stop refining once a bracket in ``mu`` is narrower than this (plateau detection).
    initial_step : float
        First bracketing step for ``|mu|``.
    max_shift : float
        Scanning a direction gives up once ``|mu|`` exceeds this.
    plateau_ok : bool
        If every bracket collapses without meeting ``tol`` (the observable steps across the
        target -- a plateau) and no bracket is found at all: ``True`` returns the closest side
        seen and warns on rank 0, ``False`` raises ``RuntimeError``.
    unreachable_message : str
        ``RuntimeError`` message when the target cannot be reached.
    rank : int
        MPI rank, for rank-0-only logging.
    comm : MPI.Comm or None
        The communicator ``observable`` is collective on -- every residual is broadcast from its
        root so that the branch decisions driving the next collective call are rank-identical.
        Pass the *same* communicator the observable's own collectives use (both criteria here
        build their bases on ``MPI.COMM_WORLD`` regardless of the caller's ``comm``). ``None``
        skips the broadcast, for a genuinely serial observable (the unit tests).

    Returns
    -------
    float
        The shift ``mu``.

    Raises
    ------
    RuntimeError
        If the target cannot be bracketed within ``max_shift`` in either direction (or every
        bracket collapses without meeting ``tol`` and ``plateau_ok=False``).
    """
    evaluated = {}

    def residual(mu):
        if mu not in evaluated:
            g = observable(mu) - target
            # ``observable`` is collective, and its value is only replicated to roundoff: both
            # criteria end in Lanczos energies (``peak_observable`` returns an unbroadcast
            # ``np.min(es)``, ``occupation_observable`` weights Allreduced density matrices by
            # rank-local energies). Every decision below reads this float -- the two ``tol``
            # tests, the ``g * g_prev < 0`` bracket detection, _refine_bracket's secant and
            # sub-bracket choice, and both RuntimeErrors -- and each one decides whether the
            # next collective ``observable`` call happens. Ranks disagreeing by one ulp on any
            # of them therefore issue different sequences of collectives and deadlock. Broadcast
            # once, here, so all of them see the identical value (CLAUDE.md's MPI rule: never
            # gate a collective on rank-local state).
            if comm is not None:
                g = comm.bcast(g, root=0)
            evaluated[mu] = g
        return evaluated[mu]

    g0 = residual(0.0)
    if abs(g0) <= tol:
        return 0.0

    # Bidirectional geometric scan, fixed +1/-1 order per level (rank-invariant collective call
    # sequence). Brackets found at a level are refined immediately, before growing further.
    prev = {1: (0.0, g0), -1: (0.0, g0)}
    active = {1, -1}
    closest_unmet = None
    level = max(width_tol, initial_step)
    while active:
        level_brackets = []
        for direction in (1, -1):
            if direction not in active:
                continue
            mu = direction * level
            if abs(mu) > max_shift:
                active.discard(direction)
                continue
            g = residual(mu)
            if abs(g) <= tol:
                return mu
            mu_prev, g_prev = prev[direction]
            if g * g_prev < 0:
                bracket = (mu_prev, g_prev, mu, g) if mu_prev < mu else (mu, g, mu_prev, g_prev)
                level_brackets.append(bracket)
            prev[direction] = (mu, g)

        # Refine this level's brackets nearest-mu=0-first -- the smallest correction to the guess
        # wins when the residual has more than one root (the non-monotone case can bracket both
        # directions at the same level). Sorted by nearest bracket *endpoint*, not nearest root:
        # two brackets tied on that endpoint distance break ties by scan order (+1 before -1),
        # which is only guaranteed optimal when each bracket holds at most one root -- true for
        # every criterion in this module (peak position, occupation).
        level_brackets.sort(key=lambda b: min(abs(b[0]), abs(b[2])))
        for mu_low, g_low, mu_high, g_high in level_brackets:
            mu_c, g_c = _refine_bracket(residual, mu_low, g_low, mu_high, g_high, tol, width_tol)
            if abs(g_c) <= tol:
                return mu_c
            # Bracket collapsed without meeting tol (a plateau/step): remember the closest point
            # reached in case every bracket ever found does this.
            if closest_unmet is None or abs(g_c) < abs(closest_unmet[1]):
                closest_unmet = (mu_c, g_c)
        level *= 2

    if closest_unmet is not None:
        mu, g = closest_unmet
        if not plateau_ok:
            raise RuntimeError(unreachable_message.format(mu=mu, value=g + target, target=target))
        if rank == 0:
            print(
                f"WARNING: the requested double-counting target {target} falls on a plateau; the "
                f"closest achievable observable is {g + target:.4f} (mu = {mu:.6f})."
            )
        return mu

    # Neither direction ever bracketed the target within max_shift; report the closer of the two
    # farthest points actually probed.
    best_mu, best_g = min(prev.values(), key=lambda mu_g: abs(mu_g[1]))
    raise RuntimeError(unreachable_message.format(mu=best_mu, value=best_g + target, target=target))


def _dc_chi(samples):
    r"""Finite-difference slope :math:`\chi = dn/d\mu` from the two closest trial shifts.

    The reviews' point 5: the answer this module returns is ``dc``, not ``n``, and the error in
    ``dc`` is ``delta_mu = delta_n / chi``. On a plateau ``chi -> 0`` and an occupation converged
    to ``occ_tol`` still leaves an unbounded ``dc``, so an occupation residual reported without
    ``chi`` beside it does not say how well the double counting is determined.

    ``samples`` is a ``{mu: value}`` mapping of the points the search actually evaluated -- the
    slope is free, every one of them having cost a full solve. Returns ``None`` when fewer than
    two distinct shifts were evaluated (the ``mu = 0`` fast path).
    """
    shifts = sorted(samples)
    if len(shifts) < 2:
        return None
    pairs = zip(shifts, shifts[1:])
    mu_low, mu_high = min(pairs, key=lambda pair: pair[1] - pair[0])
    return (samples[mu_high] - samples[mu_low]) / (mu_high - mu_low)


def _report_dc_trace(trace, label, comm, rank):
    """Print the cost accounting of one double-counting search on rank 0.

    Two aggregates and a per-``mu`` table. The kinds nest -- a ``sector_solve`` contains its
    ``build``, ``expand`` and ``eigensolve`` -- so the inner three are reported as a share of the
    search's total rather than added to it.
    """
    solves = trace.count("sector_solve")
    # Unconditional collective (the caller already broadcast the decision to trace at all). The
    # sector-solve count is the cheapest available witness that every rank walked the same path
    # through the collectives: the residual is broadcast, so the mu sequence agrees by
    # construction, but a rank-dependent cache hit or occupation-bound test inside the walk would
    # show up here as a count mismatch instead of as a hang three phases later.
    if comm is not None:
        solves_min = comm.allreduce(solves, op=MPI.MIN)
        solves_max = comm.allreduce(solves, op=MPI.MAX)
    else:
        solves_min = solves_max = solves
    if rank != 0:
        return
    if solves_min != solves_max:
        print(
            f"WARNING: the {label} search ran a different number of sector solves on different "
            f"ranks ({solves_min} to {solves_max}). The ranks took different paths through the "
            "collective calls; the next such divergence is likely to deadlock rather than warn.",
            flush=True,
        )
    total = trace.seconds("dc_evaluation")
    print(f"--- {label} double-counting search: cost accounting ---")
    print(
        f"  {trace.count('dc_evaluation')} observable evaluations, {solves} sector solves "
        f"({trace.count('sector_cache_hit')} cache hits), {total:.1f} s total"
    )
    for kind in ("build", "expand", "eigensolve"):
        seconds = trace.seconds(kind)
        share = f"{100 * seconds / total:.0f}%" if total > 0 else "n/a"
        print(f"  {kind:<11} {trace.count(kind):>5} calls  {seconds:>9.1f} s  {share:>5} of total")
    print(f"  {'mu':>12} {'value':>12} {'solves':>7} {'hits':>6} {'dets':>8} {'seconds':>9}")
    for mu, events in sorted(trace.group_by("mu").items(), key=lambda item: (item[0] is None, item[0])):
        if mu is None:
            continue
        at_mu = [event for event in events if event["kind"] == "dc_evaluation"]
        sector_solves = [event for event in events if event["kind"] == "sector_solve"]
        value = at_mu[0].get("n", at_mu[0].get("gap")) if at_mu else float("nan")
        dets = max((event.get("n_dets", 0) for event in sector_solves), default=0)
        seconds = sum(event["seconds"] for event in at_mu)
        hits = sum(1 for event in events if event["kind"] == "sector_cache_hit")
        print(f"  {mu:>12.6f} {value:>12.6f} {len(sector_solves):>7} {hits:>6} {dets:>8} {seconds:>9.1f}")
    # The occupation criterion controls n, the peak criterion controls the sector-energy gap;
    # either way the slope of the controlled quantity in mu is what converts a residual into an
    # error on the answer, and it costs nothing once the points have been solved.
    evaluations = trace.of_kind("dc_evaluation")
    field, symbol = ("n", "dn/dmu") if any("n" in event for event in evaluations) else ("gap", "dgap/dmu")
    samples = {event["mu"]: event[field] for event in evaluations if field in event and "mu" in event}
    chi = _dc_chi(samples)
    if chi is not None:
        print(f"  chi = {symbol} = {chi:.4f} (closest evaluated pair); delta_mu = delta_residual / chi")
    print("--- end cost accounting ---", flush=True)


@contextmanager
def _dc_search_trace(label, comm, rank):
    """Account for a double-counting search when ``DC_DIAGNOSTICS`` is set; otherwise a no-op.

    The activation decision gates the collective inside :func:`_report_dc_trace`, so it is
    broadcast rather than read per rank: environment variables are uniform under ``mpiexec`` in
    every normal invocation, and "in every normal invocation" is how this repo's deadlocks got in.
    Reporting happens in a ``finally`` -- a search that fails to reach its target is exactly the
    one whose cost breakdown is worth having.
    """
    enabled = config.DC_DIAGNOSTICS.get()
    if comm is not None:
        enabled = comm.bcast(enabled, root=0)
    if not enabled:
        yield None
        return
    with solver_trace.tracing() as trace:
        try:
            yield trace
        finally:
            _report_dc_trace(trace, label, comm, rank)
