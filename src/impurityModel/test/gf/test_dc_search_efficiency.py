"""The sector-first search must not evaluate the direction that cannot hold the answer.

Every evaluation is a full ground-state solve, so the cost of the double-counting search *is* its
evaluation count. The old bidirectional geometric scan probed both directions at every level
because it believed the residual was non-monotone. It is not the residual that is monotone --
it is the **charge sector**: on a fixed space ``E_N(mu) = E_N(0) - mu n_N`` is affine, so the
winning sector is the argmin of a family of lines and therefore a non-decreasing step function of
``mu``. Knowing that fixes the direction, and knowing the sign of ``d(observable)/d(mu)`` fixes it
again inside the sector.

These tests use a cheap analytic observable rather than an ED solve: the property under test is
*which* ``mu`` the search visits, which has nothing to do with what the observable costs.
"""

import numpy as np

from impurityModel.ed.dc_search import _solve_dc_shift

UNREACHABLE = "unreachable: target={target} closest={value:.4f} at mu={mu:.4f}"


def _counting(observable):
    """(wrapped observable, list of visited mu)."""
    visited = []

    def wrapped(mu):
        visited.append(mu)
        return observable(mu)

    return wrapped, visited


def test_a_monotone_observable_is_never_probed_on_the_impossible_side():
    """n(mu) is non-decreasing, so a target above n(0) can only lie at mu > 0."""
    observable, visited = _counting(lambda mu: 1.0 + 0.5 * mu)

    mu = _solve_dc_shift(
        observable,
        2.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=0.5,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        slope_sign=1,
    )

    assert np.isclose(mu, 2.0, atol=1e-5)
    assert all(m >= 0.0 for m in visited), f"probed the structurally impossible side: {visited}"


def test_the_direction_follows_the_slope_sign():
    """A peak position moves *down* as mu rises, so the same residual sign implies the other way."""
    observable, visited = _counting(lambda mu: 1.0 - 0.5 * mu)

    mu = _solve_dc_shift(
        observable,
        2.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=0.5,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        slope_sign=-1,
    )

    assert np.isclose(mu, -2.0, atol=1e-5)
    assert all(m <= 0.0 for m in visited), f"probed the structurally impossible side: {visited}"


def test_an_already_converged_guess_costs_one_evaluation():
    """The double counting is usually carried between self-consistency iterations, so "already
    converged at the guess" is the common case and must not pay for a sector bracket."""
    observable, visited = _counting(lambda mu: 1.0 + 0.5 * mu)

    mu = _solve_dc_shift(
        observable,
        1.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=0.5,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        sector_of=lambda _mu: 8,
        nominal_sector=8,
        slope_sign=1,
    )

    assert mu == 0.0
    assert len(visited) == 1, f"an already-converged guess cost {len(visited)} evaluations: {visited}"


def test_the_nominal_sector_is_preferred_over_a_root_nearer_the_guess():
    """The two-level tie-break, on a residual with a root in two different charge sectors.

    Sector 7 holds a root at ``mu = -1``; the nominal sector 8 holds one at ``mu = +3``. The old
    search returned whichever sat nearest ``mu = 0`` -- the wrong charge state. Preferring the
    nominal sector is the whole point of bracketing it first.
    """

    def sector(mu):
        return 7 if mu < 1.0 else 8

    def observable(mu):
        return (mu + 1.0) if sector(mu) == 7 else (mu - 3.0)

    counted, _visited = _counting(observable)
    mu = _solve_dc_shift(
        counted,
        0.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=0.5,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        sector_of=sector,
        nominal_sector=8,
        slope_sign=1,
    )

    assert np.isclose(mu, 3.0, atol=1e-5), f"took the root at the wrong charge state: mu = {mu}"


def test_an_unattainable_nominal_sector_falls_back_instead_of_failing():
    """The sector preference is a tie-break among solutions, never a constraint that can turn a
    solvable problem into "unreachable"."""

    def observable(mu):
        return 1.0 + 0.5 * mu

    mu = _solve_dc_shift(
        observable,
        2.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=0.5,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        # A sector the search can never reach.
        sector_of=lambda _mu: 3,
        nominal_sector=99,
        slope_sign=1,
    )

    assert np.isclose(mu, 2.0, atol=1e-5)


def test_the_search_records_what_it_did():
    report = {}
    _solve_dc_shift(
        lambda mu: 1.0 + 0.5 * mu,
        2.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=0.5,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        sector_of=lambda mu: 8 if mu >= 0 else 7,
        nominal_sector=8,
        slope_sign=1,
        report=report,
    )

    assert report["status"] == "nominal_sector"
    assert report["sector"] == 8
    assert report["nominal_sector"] == 8
    assert report["evaluations"] >= 1
    assert report["plateau"] is False


def test_an_undefined_observable_is_stepped_over_not_scored():
    """A sector the model cannot represent means the criterion is *undefined*, not "far off".

    ``fixed_peak_dc`` needs ``E[N+1]`` and ``E[N-1]``; at a shell edge one of them does not exist.
    Returning a large finite penalty there put a number of the wrong sign next to a real one and
    manufactured a bracket that was not a root -- measured at 9 of 15 evaluations and 54 of 93
    sector solves spent refining it, after which the failure message blamed "a charge-sector
    boundary" that was the search's own artefact. ``None`` propagates instead: the walk steps over
    the region and never brackets across it.
    """

    def observable(mu):
        # Undefined on a band the walk must land in (its steps from 0 are 0.5, 1.5, 3.5, ...),
        # with the genuine root beyond it at mu = 4.
        if 0.4 < mu < 2.0:
            return None
        return mu - 4.0

    counted, visited = _counting(observable)
    mu = _solve_dc_shift(
        counted,
        0.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=0.5,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        slope_sign=1,
    )

    assert np.isclose(mu, 4.0, atol=1e-5), f"did not reach the root past the undefined band: {mu}"
    assert any(0.4 < m < 2.0 for m in visited), "fixture never probed the undefined band"


def test_an_undefined_start_does_not_crash_the_search():
    """The criterion can be undefined at the guess itself; the search must still find the root."""

    def observable(mu):
        return None if abs(mu) < 0.25 else mu - 3.0

    mu = _solve_dc_shift(
        observable,
        0.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=0.5,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        slope_sign=1,
    )
    assert np.isclose(mu, 3.0, atol=1e-5)


def test_a_declared_slope_does_not_override_a_measured_plateau():
    """Declaring a slope is a hint for stepping, never a claim that the observable has one.

    ``fixed_peak_dc`` declares ``slope = -1``, but its own docstring records the case where that
    is false: when the upper and lower sectors carry the same impurity occupation, a uniform
    shift cannot move the peak at all (the ``delta_n ~ 0`` ill-conditioning). The plateau
    resolution exists to keep the answer off a charge-sector boundary in exactly that case, and
    the Newton polish must not quietly replace it -- the polish tests whether the residual
    responded, and hands back to the plateau path when it did not.

    Here the observable is flat *on* the target across ``mu in [2, 5]`` and the guess is outside
    it, so the mu = 0 fast path cannot fire and the plateau branch is the one under test.
    """

    def observable(mu):
        if mu < 2.0:
            return 1.0 - 0.5 * (2.0 - mu)
        return 1.0 if mu <= 5.0 else 1.0 + 0.5 * (mu - 5.0)

    mu = _solve_dc_shift(
        observable,
        1.0,
        tol=1e-6,
        width_tol=1e-9,
        initial_step=0.5,
        max_shift=20.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        sector_of=lambda _mu: 8,
        nominal_sector=8,
        slope_sign=1,
        slope=0.5,
    )

    # Not hard against the near edge at mu = 2: the plateau path stands off from it, because a dc
    # returned from within the search's own resolution of a sector boundary limit-cycles between
    # charge states across self-consistency iterations.
    assert 2.0 < mu <= 5.0, f"returned a point on the plateau edge: {mu}"


def test_the_polish_refines_a_hit_the_tolerance_would_have_accepted():
    """The ``g != 0`` branch: a hit inside a *wide* tolerance band is refined, not returned.

    This is the shape ``fixed_gap_dc`` has by design -- its energy tolerance is a fraction of the
    measured gap width, so the acceptance band is wide in ``mu`` and the first point inside it can
    be far from the root. The declared slope is deliberately wrong here (``-1`` against a true
    ``-0.7``), which is also the realistic case: ``-1`` is the safe *end* of the gap criterion's
    ``[-1, -0.5]`` range, not its value. One step must still move the answer toward the root, and
    the safeguard is that it is kept only because the residual actually improved.
    """
    observable, visited = _counting(lambda mu: 2.0 - 0.7 * mu)
    report = {}
    root = 2.0 / 0.7

    mu = _solve_dc_shift(
        observable,
        0.0,
        tol=0.7,
        width_tol=1e-4,
        initial_step=0.01,
        max_shift=50.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        slope_sign=-1,
        slope=-1.0,
        report=report,
    )

    hit_before_polish = visited[-2]
    assert abs(mu - root) < abs(hit_before_polish - root), f"polish moved away from the root: {visited}"
    assert report["plateau"] is False, report
    assert len(visited) <= 4, f"took {len(visited)} evaluations: {visited}"


def test_an_exact_hit_is_probed_on_both_sides_before_being_called_a_root():
    """The ``g == 0`` branch, where nothing distinguishes a root from a plateau sitting on target.

    An exact zero has no direction to step in and no magnitude to compare against, so the polish
    probes one band to each side instead. Both sides are needed: the walk always arrives at a
    plateau *edge* from outside, so a single-sided probe pointing back the way it came would see
    the residual leave ``tol`` and call every plateau edge a root.
    """
    observable, visited = _counting(lambda mu: 2.0 - 1.0 * mu)
    report = {}

    mu = _solve_dc_shift(
        observable,
        0.0,
        tol=3e-2,
        width_tol=1e-5,
        initial_step=0.01,
        max_shift=50.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        slope_sign=-1,
        slope=-1.0,
        report=report,
    )

    assert report["plateau"] is False, report
    # Converged far tighter than `tol` would have allowed on its own.
    assert abs(mu - 2.0) < 1e-9, mu
    assert len(visited) <= 4, f"took {len(visited)} evaluations: {visited}"


def test_a_known_slope_seeds_the_first_step():
    """With the slope supplied, a linear residual is solved in a handful of evaluations."""
    observable, visited = _counting(lambda mu: 2.0 - 1.0 * mu)

    mu = _solve_dc_shift(
        observable,
        0.0,
        tol=1e-9,
        width_tol=1e-12,
        initial_step=0.01,
        max_shift=50.0,
        plateau_ok=True,
        unreachable_message=UNREACHABLE,
        slope_sign=-1,
        slope=-1.0,
    )

    assert np.isclose(mu, 2.0, atol=1e-8)
    # Without the seed this needs ~9 doublings of a 0.01 step to cross mu = 2 at all.
    assert len(visited) <= 4, f"slope-seeded search took {len(visited)} evaluations: {visited}"
