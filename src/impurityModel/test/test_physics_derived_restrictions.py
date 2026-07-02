"""P3: physics-derived (coupling-strength) occupation restrictions.

The ground-state restrictions freeze bath orbitals that couple *weakly* to the impurity,
using the coupling-strength-weighted distance rather than graph hop-count. So a strongly
hybridised long chain stays free, while an orbital past a weak link is frozen regardless of
how few hops away it is.
"""

from mpi4py import MPI

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
    op = _chain_op(hop, {o: -1.0 for o in orbs})
    basis = _basis(orbs)

    # Legacy hop-count (min_dist=4) freezes orbitals 5,6; coupling-based keeps everything free.
    assert basis.build_initial_restrictions(op, coupling_cutoff=1e-3) is None
    legacy = basis.build_initial_restrictions(op, coupling_cutoff=None, min_dist=4)
    assert legacy is not None and any(5 in k or 6 in k for k in legacy)


def test_orbitals_past_a_weak_link_are_frozen():
    """A weak link partway down the chain decouples the orbitals beyond it -> frozen."""
    orbs = [1, 2, 3]
    hop = {(0, 1): 1.0, (1, 2): 1e-4, (2, 3): 1.0}  # weak link between 1 and 2
    op = _chain_op(hop, {o: -1.0 for o in orbs})
    basis = _basis(orbs)

    restr = basis.build_initial_restrictions(op, coupling_cutoff=1e-3)
    assert restr is not None
    frozen = set().union(*restr.keys())
    assert {2, 3} <= frozen  # beyond the weak link
    assert 1 not in frozen  # strongly coupled, stays free


def test_near_but_weakly_coupled_orbital_is_frozen():
    """A weakly-coupled orbital one hop away is frozen (hop-count distance would keep it free)."""
    orbs = [1, 2, 3]
    # Impurity couples strongly to 2,3 and very weakly to 1 (all one/two hops away).
    hop = {(0, 2): 1.0, (2, 3): 1.0, (0, 1): 1e-6}
    op = _chain_op(hop, {o: -1.0 for o in orbs})
    basis = _basis(orbs)

    dist, cutoff = basis._impurity_coupling_distance(
        op, tot_orb=4, all_impurity_orbitals=[0], coupling_cutoff=1e-3, min_dist=4
    )
    # Orbital 1 (weakly coupled) is beyond the cutoff; orbitals 2,3 (strong) are not.
    assert dist[0, 1] > cutoff
    assert dist[0, 2] <= cutoff and dist[0, 3] <= cutoff
