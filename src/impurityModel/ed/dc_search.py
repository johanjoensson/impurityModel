r"""The root find both double-counting criteria share, and its cost accounting.

The double counting is parametrized as a uniform shift, ``dc(mu) = dc_guess + mu * identity``;
:func:`_solve_dc_shift` drives a caller-supplied scalar observable of ``mu`` onto a target with a
sector-first bracketing and safeguarded secant/bisection refinement. Nothing here knows what the
observable is -- :mod:`dc_criteria` supplies a peak position or an impurity occupation -- so this
module carries no ED machinery and imports no solver.

**What is monotone, and what is not.** The residual is not monotone in ``mu``: the criteria
re-select both the CIPSI space and the charge sector at every ``mu``, and across a sector boundary
the residual is a different function. The *sector* is monotone, and that is what the search
exploits. On a fixed space ``E_N(mu) = E_N(0) - mu n_N`` is affine, so the winning sector is the
argmin of a family of lines -- a non-decreasing step function of ``mu``. Stage A therefore steps
one-sidedly on the integer sector (the direction follows from the sign of ``nominal - current``),
and Stage B solves the criterion inside the sector it lands in, where the charge state is fixed
and the residual *is* smooth with a known sign of slope. The old claim that the residual was "a
difference of two independently-monotone quantities" was simply wrong: the DFT reference ``n0`` is
computed once and does not depend on ``mu`` at all.

The bidirectional scan the search used to do came from that wrong premise and cost half its
evaluations to the side no root could lie on. It survives only as a *fallback*, tried after the
informed direction has failed -- i.e. on the failure path, never on the success path.

Every branch is taken on a value produced by a *collective* observable that is replicated only to
roundoff, so the residual and the sector are broadcast together, once, before anything reads them
(CLAUDE.md's MPI rule: never gate a collective on rank-local state).

The reporting half answers "where did the search spend its time" through
:mod:`impurityModel.ed.solver_trace`, under the ``DC_DIAGNOSTICS`` knob.
"""

import itertools
import math
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config, solver_trace


class DoubleCountingUnreachable(RuntimeError):
    """The double-counting target could not be reached -- a modelling verdict, not a solver
    failure.

    Distinguishing this from a bare :class:`RuntimeError` is the point (B3): the observable
    itself can also raise a bare ``RuntimeError`` on a genuine failure
    (:func:`average.thermal_average_scale_indep`, :func:`basis_transcription.build_density_matrices`),
    and a caller that catches ``RuntimeError`` indiscriminately cannot tell "the target is
    unattainable with this bath/truncation" from "the solver itself broke" -- the first should
    keep the old double counting and say so loudly; the second is a bug and should abort. Because
    this subclasses ``RuntimeError``, existing ``except RuntimeError`` handlers still catch it
    (nothing that worked before silently stops catching it) -- callers that need the distinction
    catch this type *first*.
    """


def bracket_width_tol(tau):
    """The ``width_tol`` both DC criteria pass to :func:`_solve_dc_shift`.

    Single source of truth for a literal that used to be written out (``max(tau, 1e-4)``) at
    every call site that needs it: both criteria's own ``_solve_dc_shift`` call, and anything
    filtering :func:`_dc_chi` against the same bracket-collapse resolution afterward (the search's
    own closing report, and the ``dc_diagnostics`` benchmark that reads its trace from outside).
    ``1e-4`` floors it away from zero at ``tau -> 0``, where a bracket could otherwise be asked to
    resolve to machine precision.
    """
    return max(tau, 1e-4)


def _refine_bracket(residual, mu_low, g_low, mu_high, g_high, tol, width_tol, max_iter=60):
    r"""Safeguarded secant/bisection refinement of a bracket straddling a root.

    ``[mu_low, mu_high]`` must have residuals of opposite sign (``g_low`` and ``g_high``).
    A secant estimate that leaves the bracket, or hugs an endpoint, is replaced by the
    midpoint, guaranteeing a geometric decrease of the bracket width every step. This makes
    no assumption of monotonicity beyond the bracket invariant (opposite-sign endpoints),
    so it also handles a non-monotone residual with a single sign change in the bracket.

    ``margin`` is a *quarter* of the bracket, not a hundredth. Every step here costs a full
    ground-state solve, and the old 1% guard accepted secant steps that shaved 1% off the
    bracket: measured 115 evaluations on a residual with 0.98 asymmetry where plain bisection
    needs 9, and up to ~916 in the worst case. Rejecting the outer quarters caps the per-step
    reduction at 0.75 and costs at most a few extra steps when the secant really is good.
    ``max_iter`` is a second belt: ``width_tol`` alone bounds the step count only if the bracket
    actually shrinks, which a pathological residual can defeat.

    Returns
    -------
    (mu, g, step) : tuple
        ``mu``/``g`` is the best point found: the root itself, with ``|g| <= tol``, once met;
        otherwise the closer of the two endpoints of the fully narrowed bracket. ``step`` is
        ``None`` when the root was met, and otherwise ``(mu_low, g_low, mu_high, g_high)`` of the
        collapsed bracket -- a **discontinuity** in the observable, not a root: it changes sign
        across an interval narrower than ``width_tol``, so the caller can report how large the
        jump is and how close the returned ``mu`` sits to it. That distance is the real hazard;
        see :func:`_solve_dc_shift`'s plateau branch.
    """
    for _ in range(max_iter):
        if mu_high - mu_low <= width_tol:
            break
        mu_mid = mu_high - g_high * (mu_high - mu_low) / (g_high - g_low) if g_high != g_low else np.inf
        # Safeguard: reject a secant estimate that leaves the bracket or lands in its outer
        # quarters, keeping a guaranteed geometric decrease via bisection.
        margin = 0.25 * (mu_high - mu_low)
        if not (mu_low + margin <= mu_mid <= mu_high - margin):
            mu_mid = 0.5 * (mu_low + mu_high)
        g_mid = residual(mu_mid)
        if abs(g_mid) <= tol:
            return mu_mid, g_mid, None
        # Keep the sub-bracket that still straddles a root (opposite-sign endpoints);
        # correct regardless of whether the residual is monotone.
        if g_mid * g_low < 0:
            mu_high, g_high = mu_mid, g_mid
        else:
            mu_low, g_low = mu_mid, g_mid
    step = (mu_low, g_low, mu_high, g_high)
    best = (mu_low, g_low) if abs(g_low) <= abs(g_high) else (mu_high, g_high)
    return best[0], best[1], step


def _report_unattainable_target(mu, g, step, target, width_tol):
    r"""Report a target the observable *steps across* rather than crosses.

    A collapsed bracket is not a root. The observable jumps from one branch to the other over an
    interval narrower than ``width_tol`` -- a charge-sector boundary -- and the target lies in the
    gap, so **no shift attains it**. The old warning said only "falls on a plateau", which reads
    as "the answer is a bit uncertain" when the truth is "the criterion has no solution here".

    Three numbers make that actionable, and none of them was reported before: how large the jump
    is (the honest measure of the miss), where the discontinuity sits, and how far the returned
    ``mu`` is from it. That last one is the hazard the review flagged -- ``mu`` lands within
    ``width_tol`` of a sector boundary, so the downstream solve can fall into either charge state,
    and a ``dc`` cached across CSC iterations from there is a d8/d9 limit-cycle generator.

    **Why the returned ``mu`` is the branch edge and not the branch midpoint.** The plan called
    for the midpoint of the plateau. Measured on the toy that exercises this path, the branches
    are not flat -- the lower one drifts from ``n = 0.000525`` at ``mu = -16`` to ``n = 0.011165``
    at ``mu = -1.470``, a factor of 20 -- so the midpoint (``mu ~ -8.7``, ``n = 0.0016``) misses a
    target of 0.2 by 0.198 where the edge misses it by 0.189. Moving inward off the edge makes the
    criterion *worse* on the only quantity it controls. The edge is kept; the distance to the
    discontinuity is reported instead, so the instability is visible rather than traded for a
    worse answer.
    """
    mu_low, g_low, mu_high, g_high = step
    jump = abs((g_high + target) - (g_low + target))
    distance = min(abs(mu - mu_low), abs(mu - mu_high))
    print(
        f"WARNING: no double-counting shift attains the requested target {target}. The observable "
        f"steps from {g_low + target:.4f} to {g_high + target:.4f} (a jump of {jump:.4f}) across a "
        f"discontinuity at mu = {0.5 * (mu_low + mu_high):.6f} -- a charge-sector boundary -- and "
        f"the target lies inside that gap. Returning the closer side: observable "
        f"{g + target:.4f} at mu = {mu:.6f}, missing by {abs(g):.4f}.",
        flush=True,
    )
    print(
        f"         That mu sits {distance:.2e} from the boundary (bracket resolved to "
        f"{width_tol:.1e}), so the charge state there is knife-edge: the downstream solve may "
        "fall either side, and a dc carried across self-consistency iterations from this point "
        "can limit-cycle between charge states. Prefer an explicitly reachable target.",
        flush=True,
    )


def _find_nominal_sector_point(evaluate, nominal_sector, *, initial_step, max_shift, width_tol, verbose, rank):
    r"""Stage A: *a* ``mu`` at which the ground state sits in the nominal charge sector.

    **Why the sector and not the residual.** On a fixed space ``E_N(mu) = E_N(0) - mu * n_N`` is
    affine in ``mu``, so the winning sector is the lower envelope's ``argmin`` -- and the argmin of
    a family of lines is a **monotone** step function of ``mu`` (larger ``mu`` favours larger
    ``n_N``). The residual is not monotone across sectors, which is what made the old search scan
    both directions geometrically and spend half its evaluations on the side no root could lie on.
    The sector is monotone, so one-sided stepping finds it.

    Deliberately *a point*, not the interval. Resolving both sector boundaries to ``width_tol``
    costs two bisections across the whole shift range -- measured at 22 of 50 evaluations, each a
    full ground-state solve -- and buys nothing: Stage B walks outward from here anyway and can
    simply stop when the sector changes. Every evaluation made here is shared with Stage B through
    the caller's cache.

    Returns
    -------
    float or None
        A ``mu`` inside the nominal sector, or ``None`` when no ``|mu| <= max_shift`` attains it.
    """

    def sector(mu):
        return evaluate(mu)[1]

    s0 = sector(0.0)
    if s0 == nominal_sector:
        return 0.0

    # One-sided, and the side is known: the sector is non-decreasing in mu, so a sector below the
    # nominal one can only be raised by increasing mu. This is the half of the old scan that could
    # not contain the answer -- measured 7 of 23 evaluations on the structurally impossible side.
    direction = 1 if s0 < nominal_sector else -1
    step = max(width_tol, initial_step)
    far, far_sector = 0.0, s0
    while True:
        mu = direction * step
        if abs(mu) > max_shift:
            mu = direction * max_shift
        s = sector(mu)
        if s == nominal_sector:
            return mu
        if (direction > 0 and s > nominal_sector) or (direction < 0 and s < nominal_sector):
            # Stepped straight over it: the nominal sector occupies a mu-window narrower than the
            # current step, so bisect between the last two points to land inside it.
            found = _bisect_into_sector(sector, far, far_sector, mu, s, nominal_sector, width_tol)
            if found is not None:
                return found
            break
        far, far_sector = mu, s
        if abs(mu) >= max_shift:
            break
        step *= 2

    if verbose and rank == 0:
        print(
            f"The nominal charge sector {nominal_sector} is not reached by any |mu| <= {max_shift}; "
            f"the search reached sector {far_sector} at mu = {far:.4f}.",
            flush=True,
        )
    return None


def _bisect_into_sector(sector, mu_a, s_a, mu_b, s_b, nominal_sector, width_tol):
    """A ``mu`` strictly between ``mu_a`` and ``mu_b`` whose sector is ``nominal_sector``.

    The sector is monotone between the two, so bisecting on the integer value converges onto the
    nominal one if it occurs at all; a window narrower than ``width_tol`` is reported absent
    (``None``) rather than chased below the search's own resolution.
    """
    lo, hi = (mu_a, mu_b) if mu_a < mu_b else (mu_b, mu_a)
    s_lo = s_a if mu_a < mu_b else s_b
    while hi - lo > width_tol:
        mid = 0.5 * (lo + hi)
        s_mid = sector(mid)
        if s_mid == nominal_sector:
            return mid
        if (s_mid < nominal_sector) == (s_lo < nominal_sector):
            lo, s_lo = mid, s_mid
        else:
            hi = mid
    return None


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
    sector_of=None,
    nominal_sector=None,
    slope_sign=1,
    slope=None,
    report=None,
    verbose=False,
):
    r"""Find the uniform shift ``mu`` that drives a scalar observable onto ``target``.

    Generic root-finder shared by :func:`fixed_peak_dc` and :func:`fixed_occupation_dc`. The
    double counting is parametrized as ``dc(mu) = dc_guess + mu * identity``; the caller passes an
    ``observable(mu)`` closure that builds ``dc(mu)``, solves the model and returns the scalar to
    control (the peak position or the impurity occupation).

    **Two stages, and neither of them scans both directions.**

    *Stage A* (:func:`_bracket_nominal_sector`, when ``sector_of`` and ``nominal_sector`` are
    given) bisects on the **integer charge sector**, which is a monotone step function of ``mu``
    because ``E_N(mu) = E_N(0) - mu n_N`` is affine and the winning sector is the lower envelope's
    argmin. That yields the ``mu``-interval on which the ground state sits in the nominal sector.

    *Stage B* solves the criterion **inside** that interval, where the charge state is fixed, so
    the residual is smooth with a known sign of slope and no branch structure -- a
    safeguarded secant converges in a couple of evaluations. Outside a known interval (the unit
    tests, and any caller that supplies no sector) the same one-sided logic applies to the whole
    range: ``slope_sign`` says which way the residual moves with ``mu``, so the search steps only
    toward the root instead of probing the side no root can lie on.

    The two-level preference is the user-facing contract: **prefer the nominal sector**; if it is
    unattainable within ``max_shift``, fall back to the solution closest to ``dc_guess``
    (``mu = 0``). Raise only when neither exists.

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
        Resolution in ``mu``: bracket refinement and sector-boundary bisection both stop here.
    initial_step : float
        First step for ``|mu|``.
    max_shift : float
        Give up once ``|mu|`` exceeds this.
    plateau_ok : bool
        Whether a *plateau* -- the criterion satisfied across an interval of ``mu`` rather than at
        a point -- counts as an answer.

        ``True`` (``fixed_occupation_dc``) accepts it: for a charge-transfer insulator the
        occupation really is flat across the gap, that is Karolak et al.'s Eq.-(2) breakdown, and
        the honest response is to return the resolved point and report ``status = plateau``. It
        also selects the richer unreachable diagnostic (:func:`_report_unattainable_target`), which
        reports the jump, the boundary and the distance to it.

        ``False`` (``fixed_peak_dc``, ``fixed_gap_dc``) **rejects** it, raising
        :class:`DoubleCountingUnreachable`. Those criteria pin an *energy*, and an energy that does
        not move with ``mu`` means no uniform shift can place it -- the impurity is decoupled from
        the shift, and the midpoint of the flat region is not an approximation of the answer, it is
        an arbitrary point in a region where the criterion says nothing. This flag used to select
        only the diagnostic, so a plateau was returned as ``converged`` regardless; the criteria's
        own comments claimed the behaviour that is now implemented.
    sector_of : callable, optional
        ``sector_of(mu) -> int``, the integer charge sector the ground-state search settled on at
        ``mu``. Read only *after* ``observable(mu)`` has run, so criteria that record it as a side
        effect can simply look it up.
    nominal_sector : int, optional
        The user's nominal occupation. Enables Stage A.
    slope_sign : int
        Sign of ``d(observable)/d(mu)``: ``+1`` for the impurity occupation (a positive ``mu``
        lowers the impurity level and so raises the occupation), ``-1`` for a peak position or a
        gap centre. Fixes the search *direction* from the sign of the residual alone.
    slope : float, optional
        Magnitude and sign of ``d(observable)/d(mu)`` when the caller knows it analytically. Used
        for the first step only -- a Newton step lands near the root at once, where a blind
        geometric walk needs several doublings to get there. A wrong value costs a step or two;
        it cannot cost correctness, because every step is still checked.
    report : dict, optional
        Filled in with ``status``, ``sector``, ``interval``, ``evaluations`` and ``plateau``, for
        the caller's result record.
    comm : MPI.Comm or None
        The communicator ``observable`` is collective on -- the residual *and* the sector are
        broadcast together from its root so that every branch decision driving the next collective
        call is rank-identical (CLAUDE.md's MPI rule: never gate a collective on rank-local state).

    Returns
    -------
    float
        The shift ``mu``.

    Raises
    ------
    DoubleCountingUnreachable
        The target cannot be attained. Raised identically on every rank.
    """
    evaluated = {}
    record = report if report is not None else {}

    def evaluate(mu):
        """(residual, sector) at ``mu``, computed once and replicated to every rank."""
        if mu not in evaluated:
            value = observable(mu)
            # `None` means the criterion is *undefined* at this mu -- a sector the model cannot
            # represent, not a large residual. Propagated as None all the way through: the moment
            # it becomes a number, it can flip a residual's sign and manufacture a bracket that is
            # not a root.
            g = None if value is None else value - target
            s = sector_of(mu) if sector_of is not None else None
            # One broadcast of the pair: `observable` ends in Lanczos energies that are only
            # replicated to roundoff, and every branch below reads these two values to decide
            # whether the next collective `observable` call happens. Ranks disagreeing by an ulp
            # would issue different sequences of collectives and deadlock.
            if comm is not None:
                g, s = comm.bcast((g, s), root=0)
            evaluated[mu] = (g, s)
        return evaluated[mu]

    def residual(mu):
        return evaluate(mu)[0]

    # Recorded before any early return: what the caller asked for is part of the report on every
    # path, including the one where the guess already satisfies it and nothing further is searched.
    record["nominal_sector"] = nominal_sector

    g0, s0 = evaluate(0.0)
    if g0 is not None and abs(g0) <= tol and (nominal_sector is None or s0 == nominal_sector):
        # Already converged at the guess, in the sector the user asked for. Both levels of the
        # tie-break are satisfied at once, so there is nothing to search: return before Stage A
        # spends a single further ground-state solve on bracketing a sector we are already in.
        record["status"] = "converged_at_guess"
        record["mu"] = 0.0
        record["sector"] = s0
        record["interval"] = None
        record["plateau"] = False
        record["evaluations"] = len(evaluated)
        return 0.0

    seed = None
    if sector_of is not None and nominal_sector is not None:
        seed = _find_nominal_sector_point(
            evaluate,
            nominal_sector,
            initial_step=initial_step,
            max_shift=max_shift,
            width_tol=width_tol,
            verbose=verbose,
            rank=rank,
        )
    record["nominal_sector_seed"] = seed

    # Two levels, in order. The nominal sector is a *preference among solutions*, never a
    # constraint that can make a reachable target unreachable -- so if the criterion has no root
    # inside it, the search widens to the whole range and takes the solution closest to
    # `dc_guess` (mu = 0) instead. Anything else would answer "no solution" to a question that
    # has one, which is strictly worse than answering it at another charge state and saying so.
    #
    # `in_sector` is a predicate rather than a pair of resolved boundaries: Stage B walks outward
    # and stops where it stops, so the boundaries never have to be located to `width_tol`.
    unconstrained_status = "outside_nominal_sector" if seed is not None else "unconstrained"
    attempts = []
    if seed is not None:
        attempts.append(("nominal_sector", seed, lambda mu: evaluate(mu)[1] == nominal_sector, False))
    attempts.append((unconstrained_status, 0.0, None, False))
    attempts.append((unconstrained_status, 0.0, None, True))

    mu = g = step = None
    plateau = False
    # The collapsed bracket that best explains a failure, kept across attempts: a later attempt
    # can end without one (it simply ran out of range) while an earlier one located the exact
    # discontinuity the target falls inside, which is the diagnostic worth printing.
    best_step = None
    best_miss = None
    for status, start_mu, in_sector, flip in attempts:
        mu, g, step, plateau = _solve_in_interval(
            residual,
            g0,
            -max_shift,
            max_shift,
            tol=tol,
            width_tol=width_tol,
            initial_step=initial_step,
            slope_sign=slope_sign,
            slope=slope,
            start_mu=start_mu,
            in_sector=in_sector,
            centre=(status == "nominal_sector"),
            flip=flip,
        )
        if mu is not None and abs(g) <= tol and (plateau_ok or not plateau):
            # A plateau is only an answer for a criterion that says so. For one pinning an energy
            # (`plateau_ok=False`) a flat observable means no uniform shift can place it, so the
            # loop falls through to the unreachable raise below rather than returning the midpoint
            # of a region the criterion cannot discriminate within.
            record["status"] = "plateau" if plateau else status
            record["mu"] = mu
            record["sector"] = evaluated[mu][1]
            # Both axes, not one. `status` answers "is this point determined"; `sector_verdict`
            # answers "is it the charge state that was asked for". Collapsing the second into the
            # first meant a plateau found outside the nominal sector reported only `plateau` and
            # dropped the sector miss entirely.
            record["sector_verdict"] = status if status != "nominal_sector" else None
            record["plateau"] = plateau
            record["evaluations"] = len(evaluated)
            if status == "outside_nominal_sector" and rank == 0:
                print(
                    f"WARNING: the requested target has no solution in the nominal charge sector "
                    f"{nominal_sector} (reached at mu = {seed:.4f}). Falling back to the solution "
                    f"nearest the double-counting guess: mu = {mu:.6f}, sector {evaluated[mu][1]}. "
                    "calc_selfenergy will find that sector at the returned dc.",
                    flush=True,
                )
            return mu
        if step is not None and best_step is None:
            best_step, best_miss = step, (mu, g)
        elif mu is not None and (best_miss is None or abs(g) < abs(best_miss[1])):
            best_miss = (mu, g)

    record["evaluations"] = len(evaluated)
    record["plateau"] = plateau
    if best_miss is not None:
        mu, g = best_miss
    # The point the search gave up on, recorded before the raise: the failure record and the cost
    # accounting both describe *somewhere*, and "as far as it got" is the only honest answer.
    record["mu"] = mu
    record["sector"] = evaluated.get(mu, (None, None))[1]
    if best_step is not None and plateau_ok and rank == 0:
        _report_unattainable_target(mu, g, best_step, target, width_tol)
    record["status"] = "unreachable"
    if plateau and not plateau_ok:
        # The target *was* met -- across an interval, which for a criterion pinning an energy is
        # not an answer. Say that, rather than reporting the generic "could not reach", which sends
        # the reader looking for a bracketing failure that did not happen.
        record["status"] = "plateau_rejected"
        raise DoubleCountingUnreachable(
            f"The criterion is satisfied across an interval of mu around {mu:.6f}, not at a point: "
            "the controlled quantity does not respond to the double-counting shift, so no uniform "
            "shift places it and the midpoint of that interval would be an arbitrary choice. For "
            "the peak and gap criteria this means the impurity is effectively decoupled from the "
            "shift -- both charge sectors move together. Check the impurity character of the "
            "sectors involved (the record's delta_sum), and prefer fixed_occupation_dc or a static "
            "scheme here."
        )
    raise DoubleCountingUnreachable(
        unreachable_message.format(
            mu=mu if mu is not None else 0.0,
            value=(g + target) if g is not None and math.isfinite(g) else float("nan"),
            target=target,
        )
    )


def _solve_in_interval(
    residual,
    g0,
    lo,
    hi,
    *,
    tol,
    width_tol,
    initial_step,
    slope_sign,
    start_mu=0.0,
    in_sector=None,
    centre=True,
    flip=False,
    slope=None,
):
    r"""Stage B: drive the residual to zero, walking outward from ``start_mu``.

    ``in_sector(mu)`` (optional) marks the region the answer must stay in -- the nominal charge
    sector. It is a predicate, not a resolved interval, precisely so the sector's boundaries never
    have to be bisected: the walk simply stops when it steps out. Within it the charge state is
    fixed, so the residual is smooth with a known sign of slope and no branch structure.

    ``slope_sign`` is the sign of ``d(observable)/d(mu)``, which fixes the direction from the sign
    of the residual alone. That is what removes the old bidirectional scan: there is exactly one
    way to move a residual toward zero, and it never has to be discovered by trying both.

    Returns ``(mu, g, collapsed_step, plateau)``.
    """
    sector_ok = (lambda mu: True) if in_sector is None else in_sector

    def inside(mu):
        """Is ``mu`` a place the answer may come from at all?

        Two ways it is not: the ground state there is in another charge sector, or the criterion
        is undefined there (``residual`` is ``None``). They are the same thing as far as the walk
        is concerned -- step over it and keep going -- so they are one predicate.
        """
        return residual(mu) is not None and sector_ok(mu)

    def polish(mu, g):
        """One safeguarded Newton step onto the root -- which is also the flatness probe.

        A hit within ``tol`` is not automatically a plateau: the acceptance band is ``tol /
        |slope|`` wide in ``mu`` all by itself, and mapping *that* by bisection measures the
        tolerance, not the physics. It matters as soon as a criterion has reason to set a loose
        energy tolerance -- :func:`dc_criteria.fixed_gap_dc` scales its tolerance to the measured
        gap width, because converging tighter than a percent of it is chasing the truncation noise
        of two independently expanded CIPSI sectors, and that made the band 30x wider than the
        ``mu`` resolution. Measured on the analytic toy: 17 evaluations landing 1.3e-3 from the
        exact root, against 2 landing 3.3e-5. The 15 saved there are bisections *removed* -- that
        fixture's residual is linear in the declared slope, so its Newton seed lands on the root
        and this settles at the ``width_tol`` return below without evaluating anything. The
        refinement itself pays off where the declared slope is only approximate, which is the
        realistic case (the gap criterion declares ``-1``, the safe end of ``[-1, -0.5]``).

        But "the caller supplied a slope" is not by itself proof the observable *has* one. A peak
        criterion whose two sectors carry equal impurity occupation is flat no matter what slope
        was declared (the ``delta_n ~ 0`` ill-conditioning), and there the standoff the plateau
        path applies is the right answer. So this does not assume: it takes the Newton step and
        checks whether the residual actually improved. It did -- the observable responds, and the
        polished point is the answer. It did not -- the declared slope is not the observable's,
        and ``None`` here hands the decision back to the plateau resolution, which is exactly what
        used to happen before any of this. One evaluation buys the distinction (two on the exact
        hit below, which has no step to read the answer off).

        Returns a settled ``(mu, g, None, False)``, or ``None`` for "cannot decide, fall through".
        """
        if g:
            candidate = mu - g / slope
            if abs(candidate - mu) <= width_tol:
                # The correction is below the resolution asked for, so this *is* the answer to
                # within `width_tol`. Not evidence of flatness -- settle here, and do not spend a
                # bisection discovering that the acceptance band is as wide as `tol` implies.
                return mu, g, None, False
            if not lo <= candidate <= hi:
                # The slope points at a root outside the region the answer may come from. That is
                # the boundary case the standoff below exists for; hand it over.
                return None
            g_new = residual(candidate)
            if g_new is not None and sector_ok(candidate) and abs(g_new) < abs(g):
                return candidate, g_new, None, False
            return None

        # An exact hit. There is nothing to step onto, and the residual alone cannot say whether
        # this is an isolated root or a flat region that happens to sit on the target -- both read
        # as zero. Probe *both* sides, one band away: a single-sided probe mistakes a plateau edge
        # for a root, because the walk always arrives at an edge from outside.
        probe = 2.0 * tol / abs(slope)
        if probe <= width_tol:
            return None
        for candidate in (mu - probe, mu + probe):
            if not lo <= candidate <= hi:
                # A side the answer could not have come from is not part of a plateau either.
                continue
            g_probe = residual(candidate)
            if g_probe is None or not sector_ok(candidate):
                continue
            if abs(g_probe) <= tol:
                # Still on target a whole band away: flat, whatever slope was declared.
                return None
        return mu, g, None, False

    def settle(mu, g):
        """Resolve a hit at ``mu`` into the answer the tie-break asks for.

        An isolated root is the answer. A *plateau* is not: the criterion is then satisfied across
        a range and determines nothing inside it -- Karolak et al.'s Eq. (2) "essentially breaks
        down" for a charge-transfer insulator, where the impurity occupation is flat across the
        gap. The choice then falls to the tie-break: ``mu = 0`` (leave ``dc_guess`` alone) when the
        plateau contains it, otherwise a point that is not hard against a charge-sector boundary,
        since a ``dc`` returned from within the search's own resolution of one is knife-edge and
        limit-cycles between charge states across self-consistency iterations.

        A caller that supplies ``slope`` gets one :func:`polish` step first: if the residual
        responds, the band was the tolerance rather than a plateau and the polished root is the
        answer, for one evaluation instead of a bisection. If it does not respond, the flat case
        is real whatever the caller declared, and the plateau resolution below runs unchanged.
        """
        if slope:
            polished = polish(mu, g)
            if polished is not None:
                return polished
        directions = (-1, +1) if centre else ((-1,) if mu > 0.0 else (+1,))
        # Resolve the edges only as finely as the answer needs. The standoff below is one
        # `initial_step`, so locating a boundary to `width_tol` (which can be 100x smaller)
        # buys nothing and costs a full ground-state solve per halving.
        edge_tol = max(width_tol, 0.5 * initial_step)
        a, b = _plateau_extent(residual, mu, lo, hi, tol, edge_tol, directions=directions, inside=inside)
        if b - a <= edge_tol:
            return mu, g, None, False
        if a <= 0.0 <= b:
            return 0.0, residual(0.0), None, True
        if centre:
            chosen = 0.5 * (a + b)
        else:
            near, far = (a, b) if abs(a) < abs(b) else (b, a)
            chosen = near + math.copysign(min(0.5 * (b - a), initial_step), far - near)
        return chosen, residual(chosen), None, True

    g_start = residual(start_mu)
    if g_start is not None and abs(g_start) <= tol and sector_ok(start_mu):
        return settle(start_mu, g_start)

    # The direction needs a defined residual to read its sign from. Prefer the start, fall back to
    # mu = 0, and if the criterion is undefined at both, step outward and let the flipped attempt
    # cover the other side.
    reference = g_start if g_start is not None else g0
    direction = 1 if reference is None else (-1 if (reference > 0) == (slope_sign > 0) else 1)
    if flip:
        # The informed direction found nothing. A physical observable is monotone in mu within
        # a charge sector, so this only fires when the model is not the physical one (a
        # discontinuous test observable) or the sector assumption broke down -- i.e. on the
        # failure path, never on the success path, which is where the old bidirectional scan
        # spent half its evaluations.
        direction = -direction
    # First step from the *slope*, not from a blind grid. Inside a fixed charge sector the
    # criterion is smooth with a known sensitivity -- `slope` is d(observable)/d(mu), which
    # `dc_criteria` knows analytically (-(delta+ + delta-)/2 ~ -1 for a peak or gap centre) or can
    # estimate (chi = dn/dmu for the occupation). A Newton step lands near the root immediately,
    # so the geometric walk below only has to clean up; `initial_step` bounds it so a bad slope
    # estimate cannot fling the search across the whole range on its first move.
    step = max(width_tol, initial_step)
    if slope and g_start is not None and abs(slope) > 0:
        newton = abs(g_start / slope)
        if math.isfinite(newton) and newton > 0:
            step = min(max(newton, width_tol), max(initial_step, abs(hi - lo)))
    mu_prev, g_prev = start_mu, g_start
    best = (start_mu, g_start) if (g_start is not None and sector_ok(start_mu)) else (start_mu, math.inf)
    collapsed_step = None
    # The secant can converge without ever bracketing (a one-sided approach), so the loop needs a
    # cap of its own: `width_tol` bounds `_refine_bracket`, not this walk.
    walked, max_walk = 0, 60
    while True:
        mu = mu_prev + direction * step
        clipped = False
        if not (lo <= mu <= hi):
            mu = hi if direction > 0 else lo
            clipped = True
        if mu == mu_prev:
            break
        g = residual(mu)
        if g is None:
            # Undefined here. Keep walking -- the criterion can be defined again farther out (the
            # N+-1 sector comes back into range) -- but never bracket across the gap.
            if clipped:
                break
            mu_prev, g_prev = mu, g
            step *= 2
            walked += 1
            if walked >= max_walk:
                break
            continue
        # Only in-sector points may become the answer. Updating `best` unconditionally let a
        # point one step *outside* the nominal sector be returned as though the sector-constrained
        # attempt had succeeded -- the caller then reported status "nominal_sector" beside a
        # sector that was not the nominal one.
        if inside(mu) and abs(g) < abs(best[1]):
            best = (mu, g)
        if abs(g) <= tol and inside(mu):
            return settle(mu, g)
        if g_prev is not None and g * g_prev < 0 and inside(mu):
            a, b = (mu_prev, mu) if mu_prev < mu else (mu, mu_prev)
            ga, gb = (g_prev, g) if mu_prev < mu else (g, g_prev)
            mu_c, g_c, collapsed = _refine_bracket(residual, a, ga, b, gb, tol, width_tol)
            if collapsed is None:
                return settle(mu_c, g_c)
            # The bracket collapsed without meeting tol: the observable *steps across* the target
            # rather than crossing it -- a charge-sector boundary, not a root. Remember it for the
            # unreachable diagnostic, then keep walking: a genuine root can lie farther out, and
            # settling for the near discontinuity is how the old search reported "unreachable" on
            # a target it could actually have hit.
            if collapsed_step is None or abs(g_c) < abs(collapsed_step[1]):
                collapsed_step = (mu_c, g_c, collapsed)
            if abs(g_c) < abs(best[1]):
                best = (mu_c, g_c)
        if not inside(mu):
            # Left the nominal sector: the criterion has no solution inside it on this side, and
            # the caller decides whether to widen. Stopping here is the whole reason the sector's
            # boundary never needs bisecting.
            break
        if clipped:
            break
        # Next step: a secant through the last two points, but *only* for a criterion whose slope
        # the caller vouched for. Doubling is a search for a bracket; the secant is a search for
        # the root, and it is the better move on a residual that is genuinely smooth -- measured
        # 5 -> 3 evaluations on the peak criterion. It is the worse move on one that is not: the
        # occupation criterion is plateau-dominated (that is the whole Karolak Eq.-2 breakdown),
        # a nearly-flat pair gives a secant slope near zero, and chasing it cost 6 -> 14
        # evaluations. `slope` is exactly the caller saying "this residual has a knowable slope",
        # so it gates both the Newton seed and this. Growth is capped at 4x regardless.
        if slope and g_prev is not None and g != g_prev and mu != mu_prev:
            secant_slope = (g - g_prev) / (mu - mu_prev)
            estimate = mu - g / secant_slope if secant_slope != 0 else None
            if estimate is not None and math.isfinite(estimate) and estimate != mu:
                proposed = min(abs(estimate - mu), 4 * max(step, initial_step))
                step = max(proposed, width_tol)
                direction = 1 if estimate > mu else -1
                mu_prev, g_prev = mu, g
                walked += 1
                if walked >= max_walk:
                    break
                continue
        mu_prev, g_prev = mu, g
        step *= 2
        walked += 1
        if walked >= max_walk:
            break
    if collapsed_step is not None:
        return collapsed_step[0], collapsed_step[1], collapsed_step[2], False
    return best[0], best[1], None, False


def _plateau_extent(residual, mu, lo, hi, tol, edge_tol, directions=(-1, +1), inside=None):
    r"""The largest ``[a, b] <= [lo, hi]`` around ``mu`` on which ``|residual| <= tol``.

    A hit is not automatically an answer. For a charge-transfer insulator the impurity occupation
    is flat across the whole gap -- Karolak et al.'s Eq. (2) "essentially breaks down", their
    words -- so the criterion is satisfied by an *interval* of ``mu`` and determines the double
    counting not at all. The old search returned whichever point of that interval its geometric
    grid happened to land on, which could sit right against a charge-sector boundary; a ``dc``
    returned from there is knife-edge, and carried across self-consistency iterations it is a
    d8/d9 limit-cycle generator.

    Geometric growth outward, then bisection onto each edge, so the cost is logarithmic in the
    plateau width rather than linear.
    """
    inside_pred = (lambda _mu: True) if inside is None else inside
    edges = [mu]
    for direction in directions:
        bound = lo if direction < 0 else hi
        inside, outside = mu, None
        step = edge_tol
        while True:
            probe = mu + direction * step
            if (direction < 0 and probe <= bound) or (direction > 0 and probe >= bound):
                bound_g = residual(bound)
                if bound_g is not None and abs(bound_g) <= tol and inside_pred(bound):
                    inside = bound
                else:
                    outside = bound
                break
            probe_g = residual(probe)
            if probe_g is None or abs(probe_g) > tol or not inside_pred(probe):
                outside = probe
                break
            inside = probe
            step *= 2
        if outside is not None:
            # Bisect between the last point known to be inside the plateau and the first known to
            # be outside. Named by role, not by position: the edge always lies between them, so
            # "inside" moves to a probe that is inside and "outside" to one that is not,
            # regardless of which of the two has the larger mu. Ordering them by position instead
            # inverts the update on one of the two directions and excludes the very edge being
            # hunted -- measured as a returned mu of 9.75 where the plateau began at 2.0.
            for _ in range(60):
                if abs(inside - outside) <= edge_tol:
                    break
                mid = 0.5 * (inside + outside)
                mid_g = residual(mid)
                if mid_g is not None and abs(mid_g) <= tol and inside_pred(mid):
                    inside = mid
                else:
                    outside = mid
        edges.append(inside)
    return min(edges), max(edges)


def _dc_chi(samples, at, width_tol=0.0, in_sector=None):
    r"""Finite-difference slope :math:`\chi = dn/d\mu` **at the answer**, and the span it spans.

    The reviews' point 5: the answer this module returns is ``dc``, not ``n``, and the error in
    ``dc`` is ``delta_mu = delta_n / chi``. On a plateau ``chi -> 0`` and an occupation converged
    to ``occ_tol`` still leaves an unbounded ``dc``, so an occupation residual reported without
    ``chi`` beside it does not say how well the double counting is determined.

    ``samples`` is a ``{mu: value}`` mapping of the points the search actually evaluated -- the
    slope is free, every one of them having cost a full solve -- and ``at`` is the ``mu`` the search
    returned. Three filters, each removing a way the old "closest evaluated pair" answer was a
    number about somewhere else:

    *Containment.* The pair must straddle ``at`` (``mu_low <= at <= mu_high``). A search that ends
    with a tight cluster far from its answer -- an exact hit at ``mu = 2.0`` reported ``chi =
    0.0009`` from a pair 2.4 units away, i.e. "``dc`` undetermined to +-11" measured somewhere the
    answer is not -- otherwise reports that cluster's slope as though it described the answer.
    Because the returned ``mu`` is always itself an evaluated point, containment degrades to "the
    two evaluated neighbours of the answer", and to ``None`` exactly where a slope genuinely was
    not measured: the single-point ``mu = 0`` fast path.

    *Resolution.* ``_refine_bracket`` narrows a bracket straddling a sector discontinuity down to
    exactly ``width_tol`` before giving up on it (see its docstring), so a pair that narrow is the
    one place a finite difference is guaranteed to measure a discontinuity's jump divided by its
    width rather than a slope. ``width_tol`` excludes it.

    *Sector.* Optional ``in_sector(mu) -> bool``. Across a charge-sector boundary the observable is
    a different function, so a difference taken across one is not a derivative of either branch.
    The criteria pass this (they record the sector at every ``mu``); the trace report does not,
    because trace events carry no sector and inventing one would be worse than the narrower claim.

    **The span is returned, not silently bounded.** Nothing constrains how *wide* the surviving pair
    may be: a Newton-seeded search lands near the root on its first step, so the narrowest pair
    containing the answer can still be the whole distance from ``mu = 0`` to it. That makes ``chi``
    a secant over a finite interval rather than a derivative at a point, and the difference matters
    because the record converts it into an error bar on the answer. Rejecting wide pairs was the
    alternative and is worse: it would report *no* slope in exactly the cheap searches where a
    coarse one is the only evidence there is. Reporting how far apart the two points were lets the
    reader discount it instead.

    Returns
    -------
    (chi, span) : tuple of (float, float) or (None, None)
        ``span`` is ``mu_high - mu_low`` of the pair used.
    """
    if at is None:
        return None, None
    shifts = sorted(samples)
    pairs = [
        (mu_low, mu_high)
        for mu_low, mu_high in itertools.pairwise(shifts)
        if mu_high - mu_low > width_tol
        and mu_low <= at <= mu_high
        and (in_sector is None or (in_sector(mu_low) and in_sector(mu_high)))
    ]
    if not pairs:
        return None, None
    mu_low, mu_high = min(pairs, key=lambda pair: pair[1] - pair[0])
    return (samples[mu_high] - samples[mu_low]) / (mu_high - mu_low), mu_high - mu_low


def _report_dc_trace(trace, label, comm, rank, width_tol=0.0, at=None):
    """Print the cost accounting of one double-counting search on rank 0.

    Two aggregates and a per-``mu`` table. The kinds nest -- a ``sector_solve`` contains its
    ``build``, ``expand`` and ``eigensolve`` -- so the inner three are reported as a share of the
    search's total rather than added to it. ``width_tol`` is the same bracket-collapse threshold
    the search itself used and ``at`` the ``mu`` it returned (see :func:`_dc_chi`); passing both is
    what keeps this report from attributing a slope measured elsewhere -- or a discontinuity's jump
    -- to the answer.
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
    chi, span = _dc_chi(samples, at, width_tol=width_tol)
    if chi is not None:
        print(
            f"  chi = {symbol} = {chi:.4f} (closest resolvable pair around mu = {at:.6f}, "
            f"measured over a span of {span:.4g}); delta_mu = delta_residual / chi"
        )
    else:
        print(
            f"  chi = {symbol}: not resolvable (no evaluated pair straddling the returned mu and "
            f"wider than {width_tol:.1e})"
        )
    print("--- end cost accounting ---", flush=True)


@contextmanager
def _dc_search_trace(label, comm, rank, width_tol=0.0, report=None):
    """Account for a double-counting search when ``DC_DIAGNOSTICS`` is set; otherwise a no-op.

    The activation decision gates the collective inside :func:`_report_dc_trace`, so it is
    broadcast rather than read per rank: environment variables are uniform under ``mpiexec`` in
    every normal invocation, and "in every normal invocation" is how this repo's deadlocks got in.
    Reporting happens in a ``finally`` -- a search that fails to reach its target is exactly the
    one whose cost breakdown is worth having. ``width_tol`` must be the same value the caller
    passes to ``_solve_dc_shift`` for this search, or :func:`_dc_chi` filters against the wrong
    resolution.

    ``report`` is the same dict the caller hands ``_solve_dc_shift``, held by reference rather than
    by value: the answer is not known when this block opens, and ``_dc_chi`` needs it to know where
    to measure. ``_solve_dc_shift`` writes ``mu`` into it on every exit including the raising one,
    so the accounting of a failed search still reports its slope at the point it gave up on.
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
            _report_dc_trace(
                trace, label, comm, rank, width_tol=width_tol, at=None if report is None else report.get("mu")
            )
