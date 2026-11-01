"""``ed.degeneracy``: the one answer to "which levels are one manifold".

The reference functions below are the rules as they stood in their five homes before the move (CIPSI's
CIPSI's degenerate-groups helper, the eigensolvers' copy, the observables' grouping, the last-kept loop of
``_energy_cut_indices`` and the chain loop of the TRLM window); the module must reproduce each exactly on the
inputs that matter, including levels sitting exactly ``tol`` apart and drifting runs.
"""

import numpy as np
import pytest

from impurityModel.ed.degeneracy import absorb_degenerate_tail, degenerate_groups


def _first_anchor_reference(e, tol):
    e = np.asarray(e).real
    groups, start = [], 0
    for i in range(1, len(e) + 1):
        if i == len(e) or abs(e[i] - e[start]) > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


def _chain_reference(theta, n_in, deg_tol):
    while 0 < n_in < len(theta) and theta[n_in] - theta[n_in - 1] <= deg_tol:
        n_in += 1
    return n_in


def _last_kept_reference(e_sorted, n_keep, tol):
    original = n_keep
    while n_keep < len(e_sorted) and abs(e_sorted[n_keep] - e_sorted[original - 1]) <= tol:
        n_keep += 1
    return n_keep


def _levels(rng, n):
    """Ascending levels built from runs of near-equal values, gaps drawn at, below and above ``tol``."""
    steps = rng.choice([0.0, 0.4, 1.0, 1.0 + 1e-9, 2.5, 7.0], size=n)
    return np.cumsum(steps) * 1e-6


@pytest.mark.parametrize("seed", range(40))
def test_first_anchor_is_the_old_rule(seed):
    rng = np.random.default_rng(seed)
    e = _levels(rng, int(rng.integers(0, 30)))
    for tol in (1e-9, 1e-6, 2e-6, 1e-5):
        assert degenerate_groups(e, tol) == _first_anchor_reference(e, tol)
        assert degenerate_groups(e, tol, anchor="first") == _first_anchor_reference(e, tol)


@pytest.mark.parametrize("seed", range(40))
def test_the_cut_helpers_are_the_old_loops(seed):
    rng = np.random.default_rng(seed)
    e = _levels(rng, int(rng.integers(2, 30)))
    for tol in (1e-9, 1e-6, 2e-6):
        for n in range(1, len(e)):
            assert absorb_degenerate_tail(e, n, tol) == _last_kept_reference(e, n, tol)
            assert absorb_degenerate_tail(e, n, tol, anchor="chain") == _chain_reference(e, n, tol)


def test_chain_does_not_split_a_creeping_run_but_first_does():
    e = np.array([0.0, 0.8, 1.6, 2.4, 10.0])
    assert degenerate_groups(e, 1.0, anchor="first") == [[0, 1], [2, 3], [4]]
    assert degenerate_groups(e, 1.0, anchor="chain") == [[0, 1, 2, 3], [4]]


def test_a_gap_of_exactly_tol_stays_in_the_manifold():
    assert degenerate_groups([0.0, 1.0, 3.0], 1.0) == [[0, 1], [2]]
    assert degenerate_groups([0.0, 1.0, 3.0], 1.0, anchor="chain") == [[0, 1], [2]]


def test_empty_and_single_inputs():
    assert degenerate_groups([], 1.0) == []
    assert degenerate_groups([3.0], 1.0) == [[0]]
    assert absorb_degenerate_tail(np.array([0.0, 0.0]), 0, 1.0) == 0
    assert absorb_degenerate_tail(np.array([0.0, 0.0]), 2, 1.0) == 2


def test_the_levels_may_be_complex_with_a_real_part():
    assert degenerate_groups(np.array([0.0 + 1e-3j, 0.5 - 1e-3j, 3.0]), 1.0) == [[0, 1], [2]]


def test_an_unknown_anchor_raises():
    with pytest.raises(ValueError, match="anchor"):
        degenerate_groups([0.0], 1.0, anchor="last_kept")
    with pytest.raises(ValueError, match="anchor"):
        absorb_degenerate_tail(np.array([0.0, 1.0]), 1, 1.0, anchor="first")
