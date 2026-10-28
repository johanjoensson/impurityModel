"""``hopping_cost_weights`` / ``cost_budget_restriction``: costs follow the coupling distance."""

from impurityModel.ed.basis_restrictions import (
    _impurity_coupling_distance,
    cost_budget_restriction,
    hopping_cost_weights,
)


def _chain_op(hops, n_orb):
    """Hopping operator over orbitals 0..n_orb-1 with ``hops[(i, j)]`` the (symmetric) amplitude."""
    op = {}
    for (i, j), t in hops.items():
        op[((i, "c"), (j, "a"))] = t
        op[((j, "c"), (i, "a"))] = t
    return op


def _dist(hops, n_orb):
    return _impurity_coupling_distance(_chain_op(hops, n_orb), n_orb, [0], 1e-6, 0)[0]


def test_costs_grow_along_a_decaying_chain_and_the_nearest_site_is_free():
    dist = _dist({(0, 1): 1.0, (1, 2): 0.5, (2, 3): 0.25, (3, 4): 0.1}, 5)
    cost = hopping_cost_weights(dist, [1, 2, 3, 4])
    assert cost[1] == 0
    assert 0 < cost[2] < cost[3] < cost[4]


def test_a_graded_star_still_costs():
    dist = _dist({(0, 1): 1.0, (0, 2): 0.3, (0, 3): 0.05}, 4)
    cost = hopping_cost_weights(dist, [1, 2, 3])
    assert cost[1] == 0 and 0 < cost[2] < cost[3]


def test_a_chain_with_hopping_larger_than_the_hybridization_has_no_negative_cost():
    dist = _dist({(0, 1): 0.2, (1, 2): 2.0, (2, 3): 2.0}, 4)
    cost = hopping_cost_weights(dist, [1, 2, 3])
    assert min(cost.values()) >= 0
    assert cost[1] == 0


def test_unreachable_orbitals_cost_as_much_as_the_dearest_reachable_one():
    dist = _dist({(0, 1): 1.0, (1, 2): 0.1}, 4)
    cost = hopping_cost_weights(dist, [1, 2, 3])
    assert cost[3] == cost[2] > 0


def test_cost_budget_restriction_is_the_weighted_sum_with_filled_valence_offset():
    dist = _dist({(0, 1): 1.0, (1, 2): 0.1, (0, 3): 1.0, (3, 4): 0.1}, 5)
    bath = ({0: [[1, 2]]}, {0: [[3, 4]]})
    weights, (lo, hi) = cost_budget_restriction(bath, 5, dist)
    cost = hopping_cost_weights(dist, [1, 2, 3, 4])
    assert weights == {2: -cost[2], 4: cost[4]}
    assert (lo, hi) == (-cost[2], 5 - cost[2])


def test_all_free_orbitals_give_no_restriction():
    dist = _dist({(0, 1): 1.0}, 2)
    assert cost_budget_restriction(({0: [[1]]}, {}), 3, dist) is None


def test_costs_are_rounded_not_truncated():
    # dist(2) = -log(0.9) = 0.1054 -> 8 * 0.1054 = 0.84: rounds to 1, truncates to 0
    dist = _dist({(0, 1): 1.0, (1, 2): 0.9}, 3)
    assert hopping_cost_weights(dist, [1, 2])[2] == 1
