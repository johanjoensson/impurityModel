"""Unit tests for :func:`impurityModel.ed.dc_search._solve_dc_shift`.

Pure-Python synthetic observables (no eigensolver, no MPI) exercising the bidirectional,
non-monotonicity-safe bracketing search: a plain monotone root, a staircase (plateau), a
non-monotone observable with two roots, an unreachable target, and a near plateau that must
fall through to a genuine root farther out.
"""

import numpy as np
import pytest

from impurityModel.ed.dc_search import DoubleCountingUnreachable, _solve_dc_shift

UNREACHABLE = "unreachable: target={target} closest={value:.4f} at mu={mu:.4f}"


def test_solve_dc_shift_monotone_root():
    # observable(mu) = mu is trivially monotone; root at mu = target.
    mu = _solve_dc_shift(
        lambda mu: mu,
        2.5,
        tol=1e-8,
        width_tol=1e-10,
        initial_step=0.5,
        max_shift=100.0,
        plateau_ok=False,
        unreachable_message=UNREACHABLE,
    )
    assert np.isclose(mu, 2.5, atol=1e-6)


def test_solve_dc_shift_staircase_plateau_ok_raises_unreachable():
    # A single step at mu=3.7 (0 below, 5 above): target=2 sits strictly between the two
    # achievable values, so no mu meets tol. B3: a collapsed bracket is "no solution here", so
    # plateau_ok=True still raises -- it only changes which diagnostic precedes the raise
    # (_report_unattainable_target's jump/boundary/distance narration, on rank 0).
    def observable(mu):
        return 0.0 if mu < 3.7 else 5.0

    with pytest.raises(DoubleCountingUnreachable, match="unreachable"):
        _solve_dc_shift(
            observable,
            2.0,
            tol=0.1,
            width_tol=1e-6,
            initial_step=1.0,
            max_shift=20.0,
            plateau_ok=True,
            unreachable_message=UNREACHABLE,
        )


def test_solve_dc_shift_staircase_plateau_not_ok_raises():
    def observable(mu):
        return 0.0 if mu < 3.7 else 5.0

    with pytest.raises(DoubleCountingUnreachable, match="unreachable"):
        _solve_dc_shift(
            observable,
            2.0,
            tol=0.1,
            width_tol=1e-6,
            initial_step=1.0,
            max_shift=20.0,
            plateau_ok=False,
            unreachable_message=UNREACHABLE,
        )


def test_solve_dc_shift_non_monotone_nearest_root_chosen():
    # observable(mu) = -mu^2 + 6*mu - 5 has two roots, mu=1 and mu=5 (both on the positive
    # side); a fine enough initial_step brackets and converges to the nearer one (mu=1) without
    # the geometric scan ever reaching the farther root.
    def observable(mu):
        return -(mu**2) + 6 * mu - 5

    mu = _solve_dc_shift(
        observable,
        0.0,
        tol=1e-8,
        width_tol=1e-10,
        initial_step=0.1,
        max_shift=100.0,
        plateau_ok=False,
        unreachable_message=UNREACHABLE,
    )
    assert np.isclose(mu, 1.0, atol=1e-6)


def test_solve_dc_shift_unreachable_raises():
    # observable(mu) = mu can never reach target=100 within max_shift=5.
    with pytest.raises(DoubleCountingUnreachable, match="unreachable"):
        _solve_dc_shift(
            lambda mu: mu,
            100.0,
            tol=1e-6,
            width_tol=1e-6,
            initial_step=1.0,
            max_shift=5.0,
            plateau_ok=True,
            unreachable_message=UNREACHABLE,
        )


def test_solve_dc_shift_falls_through_near_plateau_to_far_root():
    # A flat step straddling mu=0 (never within tol of target=0) plus a genuine linear root at
    # mu=6, reachable only by growing past the near, non-converging bracket. The search must not
    # settle for the near plateau (as "unreachable" or a false plateau answer) when a real root
    # exists farther out.
    def observable(mu):
        if abs(mu) < 3:
            return 0.5 if mu >= 0 else -0.5
        return mu - 6.0

    mu = _solve_dc_shift(
        observable,
        0.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=1.0,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
    )
    assert np.isclose(mu, 6.0, atol=1e-5)


def test_solve_dc_shift_observable_evaluated_once_per_distinct_mu():
    # MPI rank-invariance requires observable(mu) to be called exactly once per distinct mu
    # (repeated calls at the same mu are not required to reproduce bit-identically for an
    # expensive collective eigensolver observable).
    calls = []

    def observable(mu):
        calls.append(mu)
        return mu

    _solve_dc_shift(
        observable,
        2.5,
        tol=1e-8,
        width_tol=1e-10,
        initial_step=0.5,
        max_shift=100.0,
        plateau_ok=False,
        unreachable_message=UNREACHABLE,
    )
    assert len(calls) == len(set(calls))


def test_solve_dc_shift_wrong_sign_secant_across_sector_jump_does_not_ping_pong():
    # The AFM NiO gap-DC shape: in the nominal sector the residual decays toward a positive floor
    # and never reaches the target before the ground state jumps to the next sector, where a
    # genuine root lies farther out. A secant through one point on each side of the jump has the
    # wrong slope sign and stepped *back* below the boundary; the next same-sector secant pointed
    # across it again, ping-ponging to `max_walk` (~120 ground-state solves) and then raising.
    boundary = 0.365

    def sector(mu):
        return 8 if mu < boundary else 9

    def observable(mu):
        if sector(mu) == 8:
            return 0.003 + 0.079 * np.exp(-mu / 0.09)
        return 0.086 - 0.45 * (mu - 0.41)

    calls = []

    def counted(mu):
        calls.append(mu)
        return observable(mu)

    report = {}
    tol = 1e-3
    mu = _solve_dc_shift(
        counted,
        0.0,
        tol=tol,
        width_tol=1e-4,
        initial_step=0.05,
        max_shift=10.0,
        plateau_ok=False,
        unreachable_message=UNREACHABLE,
        sector_of=sector,
        nominal_sector=8,
        slope_sign=-1,
        slope=-1.0,
        report=report,
    )
    assert abs(observable(mu)) <= tol
    assert np.isclose(mu, 0.41 + 0.086 / 0.45, atol=tol / 0.45)
    assert report["status"] == "outside_nominal_sector"
    assert report["sector"] == 9
    assert len(calls) < 20
