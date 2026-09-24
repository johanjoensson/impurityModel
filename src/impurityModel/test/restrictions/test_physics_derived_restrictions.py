"""P3: physics-derived (coupling-strength) occupation restrictions.

The ground-state restrictions freeze bath orbitals that couple *weakly* to the impurity,
using the coupling-strength-weighted distance rather than graph hop-count. So a strongly
hybridised long chain stays free, while an orbital past a weak link is frozen regardless of
how few hops away it is.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.basis_restrictions import (
    _impurity_coupling_distance,
    build_excited_restrictions,
    build_initial_restrictions,
)
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator


def _chain_op(hoppings, onsite):
    terms = {}
    for (i, j), t in hoppings.items():
        terms[((i, "c"), (j, "a"))] = t
        terms[((j, "c"), (i, "a"))] = t
    for i, e in onsite.items():
        terms[((i, "c"), (i, "a"))] = e
    return ManyBodyOperator(terms)


def _basis(valence_block):
    return Basis(
        impurity_orbitals={0: [[0]]},
        bath_states=({0: [valence_block]}, {0: [[]]}),
        nominal_impurity_occ={0: 1},
        comm=MPI.COMM_WORLD,
        verbose=False,
    )


def test_strongly_coupled_long_chain_is_not_frozen():
    """All hops equal (strong): even far orbitals stay free — hop-count min_dist would freeze them."""
    orbs = [1, 2, 3, 4, 5, 6]
    hop = {(i, i + 1): 1.0 for i in range(0, 6)}  # 0-1-2-...-6, all t = 1
    op = _chain_op(hop, dict.fromkeys(orbs, -1.0))
    basis = _basis(orbs)

    # Legacy hop-count (min_dist=4) freezes orbitals 5,6; coupling-based keeps everything free.
    assert build_initial_restrictions(basis, op, coupling_cutoff=1e-3) is None
    legacy = build_initial_restrictions(basis, op, coupling_cutoff=None, min_dist=4)
    assert legacy is not None and any(5 in k or 6 in k for k in legacy)


def test_orbitals_past_a_weak_link_are_frozen():
    """A weak link partway down the chain decouples the orbitals beyond it -> frozen."""
    orbs = [1, 2, 3]
    hop = {(0, 1): 1.0, (1, 2): 1e-4, (2, 3): 1.0}  # weak link between 1 and 2
    op = _chain_op(hop, dict.fromkeys(orbs, -1.0))
    basis = _basis(orbs)

    restr = build_initial_restrictions(basis, op, coupling_cutoff=1e-3)
    assert restr is not None
    frozen = set().union(*restr.keys())
    assert {2, 3} <= frozen  # beyond the weak link
    assert 1 not in frozen  # strongly coupled, stays free


def test_near_but_weakly_coupled_orbital_is_frozen():
    """A weakly-coupled orbital one hop away is frozen (hop-count distance would keep it free)."""
    orbs = [1, 2, 3]
    # Impurity couples strongly to 2,3 and very weakly to 1 (all one/two hops away).
    hop = {(0, 2): 1.0, (2, 3): 1.0, (0, 1): 1e-6}
    op = _chain_op(hop, dict.fromkeys(orbs, -1.0))

    dist, cutoff = _impurity_coupling_distance(
        op, tot_orb=4, all_impurity_orbitals=[0], coupling_cutoff=1e-3, min_dist=4
    )
    # Orbital 1 (weakly coupled) is beyond the cutoff; orbitals 2,3 (strong) are not.
    assert dist[0, 1] > cutoff
    assert dist[0, 2] <= cutoff and dist[0, 3] <= cutoff


def _grouped_star():
    """Star bath, two impurity groups whose orbitals are NOT in sorted order across groups.

    Group 0 holds impurity orbitals [0, 2], group 1 holds [1, 3] -- the eg/t2g grouping of a real
    d shell interleaves the same way. Every bath orbital couples directly (and equally strongly)
    to its own impurity orbital: nothing is far from the impurity, so nothing may be windowed.
    """
    spokes = {0: [4, 5, 6], 2: [7, 8, 9], 1: [10, 11, 12], 3: [13, 14, 15]}
    hop = {(imp, b): 0.1 for imp, baths in spokes.items() for b in baths}
    onsite = {b: -0.3 for baths in spokes.values() for b in baths}
    op = _chain_op(hop, onsite)
    basis = Basis(
        impurity_orbitals={0: [[0, 2]], 1: [[1, 3]]},
        bath_states=({0: [spokes[0] + spokes[2]], 1: [spokes[1] + spokes[3]]}, {0: [[]], 1: [[]]}),
        nominal_impurity_occ={0: 1, 1: 1},
        # Every distance lookup in build_excited_restrictions sits under `if basis.chain_restrict`;
        # without it the excited-side assertion below would pass whatever the row order.
        chain_restrict=True,
        comm=MPI.COMM_WORLD,
        verbose=False,
    )
    return op, basis


def test_coupling_distance_rows_are_indexed_by_orbital():
    """Row k of the distance matrix is impurity orbital k, whatever order the groups list them in.

    The callers index rows with impurity orbital numbers. Rows used to follow the order of
    ``all_impurity_orbitals`` instead, so with interleaved groups a bath looked up from its own
    group read the distance from a different impurity orbital -- infinite on a star.
    """
    op, _ = _grouped_star()
    dist, cutoff = _impurity_coupling_distance(
        op, tot_orb=16, all_impurity_orbitals=[0, 2, 1, 3], coupling_cutoff=1e-3, min_dist=4
    )
    for imp, bath in ((0, 4), (2, 7), (1, 10), (3, 13)):
        assert dist[imp, bath] <= cutoff
    assert dist[0, 7] == np.inf  # impurity 0 does not reach impurity 2's spoke on a star


def test_star_with_interleaved_groups_gets_no_chain_window():
    """A star has no chain, so neither the initial nor the ground-state window may restrict a bath.

    Regression: an SrMnO3 star (eg group [0,1,5,6], t2g group [2,3,4,7,8,9]) had the baths of
    impurity orbitals 3-6 windowed as if they were far chain sites, because the distance-matrix
    rows were looked up by orbital number but ordered group by group.
    """
    op, basis = _grouped_star()
    assert build_initial_restrictions(basis, op, coupling_cutoff=1e-3) is None
    bath = set(range(4, 16))
    # Plain (binary) windows, then the graded three-zone path (_emit_graded_chain_window).
    for slater_weight_min in (None, 1e-6):
        excited = build_excited_restrictions(basis, op, psis=None, es=None, slater_weight_min=slater_weight_min)
        assert not any(set(k) & bath for k in (excited or {})), (slater_weight_min, excited)
