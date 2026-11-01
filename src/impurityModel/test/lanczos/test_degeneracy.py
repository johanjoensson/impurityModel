"""``ed.degeneracy``: the one answer to "which levels are one manifold".

The reference functions below are the rules as they stood in their five homes before the move (CIPSI's
CIPSI's degenerate-groups helper, the eigensolvers' copy, the observables' grouping, the last-kept loop of
``_energy_cut_indices`` and the chain loop of the TRLM window); the module must reproduce each exactly on the
inputs that matter, including levels sitting exactly ``tol`` apart and drifting runs.
"""

import numpy as np
import pytest

from impurityModel.ed.degeneracy import absorb_degenerate_tail, degenerate_groups


def _chain_groups_reference(e, tol):
    groups, current = [], [0]
    for i in range(1, len(e)):
        if abs(e[i] - e[i - 1]) <= tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    return groups + [current] if len(e) else []


def _chain_tail_reference(theta, n_in, deg_tol):
    while 0 < n_in < len(theta) and theta[n_in] - theta[n_in - 1] <= deg_tol:
        n_in += 1
    return n_in


def _levels(rng, n):
    """Ascending levels built from runs of near-equal values, gaps drawn at, below and above ``tol``."""
    steps = rng.choice([0.0, 0.4, 1.0, 1.0 + 1e-9, 2.5, 7.0], size=n)
    return np.cumsum(steps) * 1e-6


@pytest.mark.parametrize("seed", range(40))
def test_groups_and_tail_are_the_chain_rule(seed):
    rng = np.random.default_rng(seed)
    e = _levels(rng, int(rng.integers(0, 30)))
    for tol in (1e-9, 1e-6, 2e-6, 1e-5):
        assert degenerate_groups(e, tol) == _chain_groups_reference(e, tol)
        for n in range(len(e) + 1):
            assert absorb_degenerate_tail(e, n, tol) == _chain_tail_reference(e, n, tol)


def test_a_creeping_run_is_one_manifold():
    """The old first-level anchor cut this at 1.6 (two levels of the run in each half)."""
    e = np.array([0.0, 0.8, 1.6, 2.4, 10.0])
    assert degenerate_groups(e, 1.0) == [[0, 1, 2, 3], [4]]
    assert absorb_degenerate_tail(e, 1, 1.0) == 4


def test_a_gap_of_exactly_tol_stays_in_the_manifold():
    assert degenerate_groups([0.0, 1.0, 3.0], 1.0) == [[0, 1], [2]]


def test_empty_and_single_inputs():
    assert degenerate_groups([], 1.0) == []
    assert degenerate_groups([3.0], 1.0) == [[0]]
    assert absorb_degenerate_tail(np.array([0.0, 0.0]), 0, 1.0) == 0
    assert absorb_degenerate_tail(np.array([0.0, 0.0]), 2, 1.0) == 2


def test_the_levels_may_be_complex_with_a_real_part():
    assert degenerate_groups(np.array([0.0 + 1e-3j, 0.5 - 1e-3j, 3.0]), 1.0) == [[0, 1], [2]]
