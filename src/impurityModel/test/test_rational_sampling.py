"""Tests for the set-valued AAA fit and the greedy adaptive sampler primitives."""

import numpy as np
import pytest

from impurityModel.ed.rational_sampling import barycentric_eval, greedy_next_samples, set_valued_aaa


def _shared_pole_map(x, poles, numerators):
    """F[i, k] = sum_j numerators[k, j] / (x_i - poles[j]) -- components sharing all poles."""
    cauchy = 1.0 / (x[:, None] - poles[None, :])  # (n, n_poles)
    return cauchy @ numerators.T  # (n, K)


POLES = np.array([-6.2 + 0.2j, -5.4 + 0.2j, -5.0 + 0.35j])
RNG_NUMERATORS = np.array(
    [
        [1.0, 0.3, 0.1],
        [0.2 + 0.1j, 1.5, 0.05],
        [0.7, 0.1j, 2.0],
        [0.05, 0.9, 0.4 + 0.2j],
    ]
)


def test_set_valued_aaa_recovers_shared_pole_function():
    """A K-component rational function with 3 shared poles needs few support points and is
    reproduced to fit tolerance on every node and component."""
    x = np.linspace(-7.5, -4.5, 61)
    F = _shared_pole_map(x, POLES, RNG_NUMERATORS)
    support, weights = set_valued_aaa(x, F, rtol=1e-12)
    assert len(support) <= 8, f"expected a compact support set for 3 poles, got {len(support)}"
    R = barycentric_eval(x, x[support], weights, F[support])
    assert np.max(np.abs(R - F)) <= 1e-10 * np.max(np.abs(F))


def test_barycentric_eval_interpolates_support_exactly():
    x = np.linspace(-7.5, -4.5, 31)
    F = _shared_pole_map(x, POLES, RNG_NUMERATORS)
    support, weights = set_valued_aaa(x, F, rtol=1e-12)
    R = barycentric_eval(x[support], x[support], weights, F[support])
    np.testing.assert_array_equal(R, F[support])


def test_shared_weights_reconstruct_unseen_components():
    """The production pattern: fit the weights on a component subsample, reconstruct the FULL
    component set from support-point values only (valid because the poles are shared)."""
    x = np.linspace(-7.5, -4.5, 61)
    F = _shared_pole_map(x, POLES, RNG_NUMERATORS)
    fit_cols = [0, 2]  # fit sees half the components
    support, weights = set_valued_aaa(x, F[:, fit_cols], rtol=1e-12)
    R_all = barycentric_eval(x, x[support], weights, F[support])  # full component set
    assert np.max(np.abs(R_all - F)) <= 1e-9 * np.max(np.abs(F))


def test_set_valued_aaa_rtol_controls_support_size():
    x = np.linspace(-7.5, -4.5, 121)
    # Many poles with geometrically decaying residues: loose tolerances only need the
    # strong poles, so the support size grows as rtol tightens.
    rng = np.random.default_rng(7)
    poles = np.linspace(-7.2, -4.8, 12) + 1j * rng.uniform(0.15, 0.4, 12)
    numerators = rng.standard_normal((3, 12)) * (10.0 ** -np.arange(12))[None, :]
    F = _shared_pole_map(x, poles, numerators)
    sizes = [len(set_valued_aaa(x, F, rtol=r)[0]) for r in (1e-2, 1e-6, 1e-12)]
    assert sizes[0] < sizes[1] <= sizes[2], f"support sizes not increasing with accuracy: {sizes}"


def test_set_valued_aaa_zero_function():
    x = np.linspace(0.0, 1.0, 10)
    support, weights = set_valued_aaa(x, np.zeros((10, 3), dtype=complex), rtol=1e-12)
    assert len(support) == 1


def test_greedy_next_samples_prefers_surrogate_then_space_fills():
    x = np.linspace(0.0, 10.0, 11)
    solved = [0, 10]
    err = np.zeros(11)
    err[3] = 1.0
    err[7] = 0.5
    picks = greedy_next_samples(x, solved, err, 3)
    assert picks[:2] == [3, 7]
    assert len(picks) == 3 and picks[2] not in solved and picks[2] not in picks[:2]
    # third pick is space-filling: farthest from solved+picked (x=5 area)
    assert picks[2] == 5


def test_greedy_next_samples_no_surrogate_space_fills():
    x = np.linspace(0.0, 10.0, 11)
    picks = greedy_next_samples(x, [0, 10], None, 1)
    assert picks == [5]


def test_greedy_next_samples_exhausts_grid():
    x = np.linspace(0.0, 1.0, 4)
    picks = greedy_next_samples(x, [0, 1, 2, 3], np.ones(4), 2)
    assert picks == []


def test_greedy_returns_fewer_when_grid_nearly_solved():
    x = np.linspace(0.0, 1.0, 4)
    picks = greedy_next_samples(x, [0, 1, 2], np.zeros(4), 3)
    assert picks == [3]


def test_set_valued_aaa_needs_two_nodes():
    with pytest.raises(ValueError):
        set_valued_aaa(np.array([1.0]), np.array([[1.0]]), rtol=1e-12)
