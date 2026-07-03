"""Pure (MPI-free) tests for the unit-packing math behind basis splitting."""

import numpy as np
import pytest

from impurityModel.ed.manybody_basis import _pack_units


def _check_valid_packing(subgroups, procs_per_color, n_units, comm_size):
    assert sum(len(g) for g in subgroups) == n_units
    assert sorted(u for g in subgroups for u in g) == list(range(n_units))
    assert all(len(g) > 0 for g in subgroups)
    assert np.sum(procs_per_color) == comm_size
    assert np.all(procs_per_color >= 1)


def test_uniform_weights_max_split():
    weights = [1.0] * 8
    subgroups, procs = _pack_units(weights, 8, 1.0)
    _check_valid_packing(subgroups, procs, 8, 8)
    assert len(subgroups) == 8
    assert all(p == 1 for p in procs)


def test_single_unit_no_split():
    subgroups, procs = _pack_units([5.0], 4, 1.0)
    assert subgroups is None and procs is None


def test_zero_threshold_forces_unified():
    subgroups, procs = _pack_units([1.0, 2.0, 3.0], 4, 0.0)
    assert subgroups is None and procs is None


def test_dominant_unit_caps_colors():
    # One unit carries almost all the mass: participation ~ 1, but ceil() grants a
    # second color for the residual mass. The dominant unit's bin gets nearly all
    # ranks; the split must stay valid.
    weights = [1000.0, 1e-3, 1e-3, 1e-3]
    subgroups, procs = _pack_units(weights, 4, 1.0)
    _check_valid_packing(subgroups, procs, 4, 4)
    assert len(subgroups) == 2
    heavy_color = next(c for c, g in enumerate(subgroups) if 0 in g)
    assert procs[heavy_color] == 3


def test_remainder_reclaim_terminates():
    # Regression for the fancy-index write-back bug: skewed weights where the
    # max(1, floor) floors over-allocate (sum > comm_size) used to spin forever in
    # the remainder loop. With split_threshold > 1 the participation cap no longer
    # masks the over-allocation.
    weights = [0.7, 0.1, 0.1, 0.1]
    subgroups, procs = _pack_units(weights, 4, 3.0)
    _check_valid_packing(subgroups, procs, 4, 4)


def test_remainder_reclaim_many_small_units():
    # Many near-zero weights next to a dominant one force floors of 0 -> max(1, .)
    # bumps -> over-allocation that must be reclaimed from the light bins.
    weights = [100.0] + [1e-6] * 7
    subgroups, procs = _pack_units(weights, 8, 8.0)
    _check_valid_packing(subgroups, procs, 8, 8)
    # The dominant unit's bin keeps the lion's share of the ranks.
    heavy_color = next(c for c, g in enumerate(subgroups) if 0 in g)
    assert procs[heavy_color] == max(procs)


@pytest.mark.parametrize("comm_size", [1, 2, 3, 4, 7, 16, 64])
@pytest.mark.parametrize("seed", range(5))
def test_randomized_packings_valid(comm_size, seed):
    rng = np.random.default_rng(seed)
    n_units = rng.integers(1, 40)
    weights = rng.pareto(1.0, size=n_units) + 1e-12
    for threshold in (0.5, 1.0, 2.0):
        packed = _pack_units(weights, comm_size, threshold)
        if packed[0] is None:
            continue
        _check_valid_packing(*packed, n_units, comm_size)


def test_deterministic():
    rng = np.random.default_rng(1)
    weights = rng.random(17)
    first = _pack_units(weights, 6, 1.0)
    second = _pack_units(weights.copy(), 6, 1.0)
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])


def test_lpt_no_worse_than_round_robin():
    # LPT's heaviest bin must not exceed round-robin dealing's on random weights.
    rng = np.random.default_rng(2)
    for _ in range(50):
        n_units = int(rng.integers(2, 30))
        weights = rng.pareto(1.0, size=n_units) + 1e-12
        packed = _pack_units(weights, 8, 1.0)
        if packed[0] is None:
            continue
        subgroups, _ = packed
        n_colors = len(subgroups)
        normalized = weights / np.sum(weights)
        lpt_max = max(np.sum(normalized[list(g)]) for g in subgroups)
        # Round-robin dealing of descending-sorted units.
        order = np.argsort(normalized, kind="stable")[::-1]
        rr_mass = np.zeros(n_colors)
        for i, u in enumerate(order):
            rr_mass[i % n_colors] += normalized[u]
        assert lpt_max <= np.max(rr_mass) + 1e-12
