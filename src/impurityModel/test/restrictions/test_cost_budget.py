"""``hopping_cost_weights`` / ``cost_budget_restriction``: costs follow the coupling distance."""

import numpy as np

from impurityModel.ed.basis_restrictions import (
    _impurity_coupling_distance,
    build_weighted_restrictions,
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


# --- the causality floor (collective) ---------------------------------------------------------------
import pytest  # noqa: E402
from mpi4py import MPI  # noqa: E402

from impurityModel.ed.basis_restrictions import floor_weighted_restrictions, get_effective_weighted_bounds  # noqa: E402
from impurityModel.ed.manybody_basis import Basis  # noqa: E402
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant  # noqa: E402


def _det(occupied, n_bytes=1):
    raw = bytearray(n_bytes)
    for i in occupied:
        raw[i // 8] |= 1 << (7 - i % 8)
    return SlaterDeterminant.from_bytes(bytes(raw))


def _floor_basis(dets, entries, comm):
    imp = {0: [[0, 1]]}
    baths = ({0: [[2, 3]]}, {0: [[4, 5]]})
    return Basis(imp, baths, initial_basis=sorted(dets), comm=comm, verbose=False, weighted_restrictions=entries)


# weight 1 on orbitals 2..5: q = number of electrons in the bath
_ENTRY = ({2: 1, 3: 1, 4: 1, 5: 1}, (0, 1))


@pytest.mark.mpi
@pytest.mark.parametrize("n_dets", [1, 3])
def test_the_observed_bounds_span_the_whole_basis_whatever_the_rank_count(n_dets):
    """One determinant leaves ranks empty at -n 3; the bounds are global either way."""
    dets = [_det([0, 2]), _det([0, 2, 3, 4]), _det([1, 2, 3])][:n_dets]
    basis = _floor_basis(dets, [_ENTRY], MPI.COMM_WORLD)
    qs = [sum(1 for o in (2, 3, 4, 5) if (d.to_bytearray()[0] >> (7 - o)) & 1) for d in dets]
    assert get_effective_weighted_bounds(basis) == [(min(qs), max(qs))]


@pytest.mark.mpi
def test_the_floor_widens_a_window_the_basis_exceeds_and_leaves_a_loose_one():
    basis = _floor_basis([_det([0, 2, 3, 4]), _det([0, 2])], [_ENTRY, ({2: 1}, (-3, 5))], MPI.COMM_WORLD)
    floored = floor_weighted_restrictions(basis)
    assert floored[0][1] == (0, 3)  # the basis holds a 3-electron bath determinant: floor above budget 1
    assert floored[1][1] == (-3, 5)


def test_no_weighted_restrictions_means_no_floor():
    basis = _floor_basis([_det([0, 2])], None, None)
    assert get_effective_weighted_bounds(basis) is None
    assert floor_weighted_restrictions(basis) is None


def test_build_weighted_restrictions_ands_the_two_budgets_and_needs_its_inputs():

    hops = {(0, 1): 1.0, (1, 2): 0.1, (0, 3): 1.0, (3, 4): 0.1}
    op = _chain_op(hops, 5)
    bath = ({0: [[1, 2]]}, {0: [[3, 4]]})
    imp = {0: [[0]]}
    both = build_weighted_restrictions(bath, 2, cost_budget=1.0, h_op=op, impurity_orbitals=imp, slater_weight_min=1e-6)
    assert len(both) == 2
    total = round(8 * 1.0 * -np.log(1e-6))
    dist = _dist(hops, 5)
    assert both[1] == cost_budget_restriction(bath, total, dist)
    assert len(build_weighted_restrictions(bath, 2)) == 1
    with pytest.raises(ValueError, match="needs h_op"):
        build_weighted_restrictions(bath, None, cost_budget=1.0)
    with pytest.raises(ValueError, match="slater_weight_min > 0"):
        build_weighted_restrictions(bath, None, cost_budget=1.0, h_op=op, impurity_orbitals=imp, slater_weight_min=0.0)


# --- the cost budget through calc_gs ----------------------------------------------------------------


def _chain_siam():
    """imp {0, 1}; valence chain 2 - 3 and conduction chain 4 - 5 hanging off orbital 0 only (weak second hops)."""
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    terms = {((o, "c"), (o, "a")): e for o, e in ((0, -2.0), (1, -2.0), (2, -4.0), (3, -4.0), (4, 4.0), (5, 4.0))}
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = 6.0
    for a, b, t in ((0, 2, 1.0), (0, 4, 1.0), (2, 3, 0.05), (4, 5, 0.05), (1, 3, 0.05)):
        terms[((a, "c"), (b, "a"))] = t
        terms[((b, "c"), (a, "a"))] = t
    bs = BlockStructure(
        blocks=[[0, 1]],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    return ManyBodyOperator(terms), {0: [[0, 1]]}, ({0: [[2, 3]]}, {0: [[4, 5]]}), bs


def _gs_basis(cost_budget, swm=1e-12, comm=None):
    from impurityModel.ed.groundstate import calc_gs

    op, imp, bath, bs = _chain_siam()
    weighted = build_weighted_restrictions(
        bath, None, cost_budget=cost_budget, h_op=op, impurity_orbitals=imp, slater_weight_min=swm
    )
    setup = dict(
        impurity_orbitals=imp,
        bath_states=bath,
        N0={0: 1},
        tau=0.01,
        dense_cutoff=1000,
        comm=comm,
        truncation_threshold=100000,
        weighted_restrictions=weighted,
    )
    _p, es, basis, _r, _i = calc_gs(op, setup, bs, np.eye(2, dtype=complex), verbose=False, slaterWeightMin=swm)
    return weighted, min(es), basis


def test_a_cost_budget_bounds_every_ground_state_determinant_and_a_large_one_changes_nothing():

    _w, e_free, basis_free = _gs_basis(None)
    weighted, _e_tight, basis_tight = _gs_basis(0.05)  # 8 * 0.05 * 27.6 = 11: below the cost of a far site (24)
    # The mixed-valence seeds are generated without the weighted restrictions (as with excitation_budget); the
    # expansion honours them, so the tight budget removes determinants.
    assert weighted[0] == ({3: -24, 5: 24}, (-24, -13))
    assert basis_tight.size < basis_free.size
    _w, e_big, basis_big = _gs_basis(50.0)
    np.testing.assert_allclose(e_big, e_free, atol=1e-10)
    assert basis_big.size == basis_free.size


def _basis_keys(basis):
    local = {bytes(d.to_bytearray()[:1]) for d in basis.local_basis}
    comm = basis.comm if basis.is_distributed else None
    return set().union(*(comm.allgather(local) if comm is not None else [local]))


@pytest.mark.mpi
def test_a_cost_restricted_ground_state_basis_is_the_same_set_on_every_rank_count():
    """The cost window is a pure function of the key: serial and distributed expansions end on one set."""
    _w, e_serial, serial = _gs_basis(0.05)
    _w, e_dist, dist = _gs_basis(0.05, comm=MPI.COMM_WORLD)
    assert _basis_keys(dist) == _basis_keys(serial)
    np.testing.assert_allclose(e_dist, e_serial, atol=1e-10)


@pytest.mark.mpi
@pytest.mark.parametrize("floor", [True, False])
def test_the_floored_window_admits_every_single_transition_of_a_ground_state_determinant(floor):
    """The causality floor: a basis that exceeds its nominal window (a 3-electron bath determinant against a
    budget of 1) must still admit c^+_j|D> and c_j|D> for every determinant ``D`` it holds. Without the floor
    the widened window excludes the 4-electron image of the large determinant."""
    from impurityModel.ed.symmetries import widen_weighted_restrictions

    basis = _floor_basis([_det([0, 2, 3, 4]), _det([0, 2])], [_ENTRY], MPI.COMM_WORLD)
    window = widen_weighted_restrictions(floor_weighted_restrictions(basis) if floor else basis.weighted_restrictions)
    weights, (lo, hi) = window[0][0], window[0][1]
    qs = [sum(weights[o] for o in weights if (d.to_bytearray()[0] >> (7 - o)) & 1) for d in basis.local_basis]
    comm = MPI.COMM_WORLD
    qs = [q for part in comm.allgather(qs) for q in part]
    images = [q + w for q in qs for w in (1, -1)]
    outside = [q for q in images if not lo <= q <= hi]
    assert (not outside) if floor else outside
