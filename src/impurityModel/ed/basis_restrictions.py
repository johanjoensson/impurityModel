"""
Occupation-restriction construction for a :class:`~impurityModel.ed.manybody_basis.Basis`:
effective (observed) restrictions of the current determinant set, connectivity-derived
ground-state restrictions, and widened restrictions for excited/spectral sectors.

The functions take the basis as their first argument; the ones that reduce over the
distributed determinant set (`get_effective_restrictions`, and through it
`build_excited_restrictions`) contain MPI collectives and must be called by all ranks
of ``basis.comm``.
"""

import itertools
from math import ceil
from typing import Optional

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.basis_transcription import build_density_matrices

# Tunable restriction knobs, gathered here as named constants (rather than literals scattered
# through the builders) so a measurement sweep or a driver can override them. Every default
# reproduces the historical behavior exactly:
#   * COUPLING_CUTOFF_DEFAULT / MIN_DIST_DEFAULT -- the freeze-eligibility threshold on the
#     physics-weighted / legacy hop-count impurity coupling distance (see
#     :func:`_impurity_coupling_distance`). Resolved when the corresponding argument is None.
#   * CHAIN_FILLED_HOLE_FRACTION / CHAIN_EMPTY_ELECTRON_FRACTION -- the excited-sector chain
#     window widths: a "filled" valence chain of length L may lose up to
#     ``round(L * hole_fraction)`` electrons (floor occupation ``int(L*(1-hole_fraction))``),
#     an "empty" conduction chain may gain up to ``ceil(L * electron_fraction)``. Both default
#     to 0.5, i.e. the historical ``(L//2, L)`` / ``(0, ceil(L/2))`` windows.
COUPLING_CUTOFF_DEFAULT = 1e-3
MIN_DIST_DEFAULT = 4
CHAIN_FILLED_HOLE_FRACTION = 0.5
CHAIN_EMPTY_ELECTRON_FRACTION = 0.5

# Sentinel for "resolve from the module default" -- distinct from an explicit ``None``, which
# is a *meaningful* value for ``coupling_cutoff`` (it selects the legacy unweighted hop-count
# metric in :func:`_impurity_coupling_distance`). A caller that omits the argument gets the
# module default; one that passes ``None`` still gets the legacy path.
_USE_DEFAULT = object()


def get_effective_restrictions(basis) -> dict[frozenset[int], tuple[int, int]]:
    """Calculate the actual min/max occupations observed across the current basis.

    Returns
    -------
    restrictions : dict of frozenset of int to (int, int)
        Dictionary mapping orbital subsets to their observed (min, max) occupations.
    """
    valence_baths, conduction_baths = basis.bath_states
    restrictions = {}

    # Impurity occupation is restricted on the *whole* impurity as a single subset (never
    # per orbital-symmetry group). A per-group impurity pin would fix S_z (spin groups) or
    # the eg/t2g ratio (manifold groups) and destroy spin-multiplet degeneracy / confine the
    # charge; the union bound confines only the total impurity charge, leaving the manifolds
    # free to redistribute at fixed count.
    all_impurity_indices = frozenset(
        sorted(ind for imp_blocks in basis.impurity_orbitals.values() for imp_ind in imp_blocks for ind in imp_ind)
    )
    valence_index_sets = {
        i: frozenset(sorted(ind for val_ind in valence_baths[i] for ind in val_ind)) for i in valence_baths
    }
    conduction_index_sets = {
        i: frozenset(sorted(ind for con_ind in conduction_baths[i] for ind in con_ind)) for i in conduction_baths
    }

    max_imp, min_imp = 0, len(all_impurity_indices)
    min_val = {i: len(valence_index_sets[i]) for i in valence_baths}
    max_con = {i: 0 for i in conduction_baths}
    for state in basis.local_basis:
        bits = psr.bytes2bitarray(bytes(state.to_bytearray()), basis.num_spin_orbitals)
        n_imp = sum(bits[i] for i in all_impurity_indices)
        max_imp = max(max_imp, n_imp)
        min_imp = min(min_imp, n_imp)
        for i in valence_baths:
            n_val = sum(bits[j] for j in valence_index_sets[i])
            n_con = sum(bits[j] for j in conduction_index_sets[i])
            min_val[i] = min(min_val[i], n_val)
            max_con[i] = max(max_con[i], n_con)
    if basis.is_distributed:
        max_imp = basis.comm.allreduce(max_imp, op=MPI.MAX)
        min_imp = basis.comm.allreduce(min_imp, op=MPI.MIN)
    if len(all_impurity_indices) > 0:
        restrictions[all_impurity_indices] = (min_imp, max_imp)
    for i in sorted(valence_baths.keys()):
        v_min, c_max = min_val[i], max_con[i]
        if basis.is_distributed:
            v_min = basis.comm.allreduce(v_min, op=MPI.MIN)
            c_max = basis.comm.allreduce(c_max, op=MPI.MAX)
        # Valence baths sit filled (max = full), conduction empty (min = 0) by convention.
        if len(valence_index_sets[i]) > 0:
            restrictions[valence_index_sets[i]] = (v_min, len(valence_index_sets[i]))
        if len(conduction_index_sets[i]) > 0:
            restrictions[conduction_index_sets[i]] = (0, c_max)
    return restrictions


def _get_updated_occ_restrictions(
    restrictions: dict[frozenset[int], tuple[int, int]], orbs: frozenset[int], dN: Optional[tuple[int, int]]
) -> tuple[int, int]:
    """Compute the updated occupation bounds after an operator excitation.

    Parameters
    ----------
    restrictions : dict of frozenset of int to (int, int)
        The existing occupation restrictions.
    orbs : frozenset of int
        The set of orbitals being modified.
    dN : tuple of (int, int), optional
        (occupations_decreased, occupations_increased) bounds, or None.

    Returns
    -------
    min_occ : int
        The new minimum occupation allowed.
    max_occ : int
        The new maximum occupation allowed.
    """
    if dN is None or orbs not in restrictions:
        return 0, len(orbs)
    occ_dec, occ_inc = dN
    min_occ, max_occ = restrictions[orbs]
    return max(min_occ - occ_dec, 0), min(max_occ + occ_inc, len(orbs))


def _impurity_coupling_distance(op, tot_orb, all_impurity_orbitals, coupling_cutoff, min_dist):
    r"""Distance-from-impurity matrix classifying how weakly each orbital couples in.

    The default (``coupling_cutoff`` set) is a *physics-derived* metric: each hopping edge
    gets weight :math:`-\log(|h_{ij}| / h_{\max})` (``h_max`` = largest off-diagonal
    hopping), so the weighted shortest path from the impurity accumulates
    :math:`-\log` of the product of couplings along the best path and
    :math:`e^{-\text{dist}}` is the effective coupling strength. An orbital counts as
    decoupled (freeze-eligible) when ``dist > -log(coupling_cutoff)``. This follows the
    actual coupling *strength* rather than graph hop-count, so a strongly-hybridised long
    chain stays free while a weakly-coupled near orbital is frozen.

    With ``coupling_cutoff=None`` it falls back to the legacy unweighted hop-count distance
    with threshold ``min_dist``.

    Returns ``(dist_matrix, threshold)`` with ``dist_matrix`` rows ordered as
    ``all_impurity_orbitals`` (the callers index it with impurity orbital indices, valid for
    the 0-based impurity layout) and an orbital frozen when ``dist > threshold``.
    """
    hop = np.zeros((tot_orb, tot_orb))
    for i, j in itertools.product(range(tot_orb), repeat=2):
        key = ((i, "c"), (j, "a"))
        if key in op:
            hop[i, j] = abs(op[key])
    if coupling_cutoff is None:
        graph = hop > 1e-8
        dist = sp.sparse.csgraph.shortest_path(graph, directed=False, unweighted=True, indices=all_impurity_orbitals)
        return dist, min_dist
    np.fill_diagonal(hop, 0.0)
    h_max = hop.max()
    mask = hop > 1e-8
    graph = np.zeros((tot_orb, tot_orb))
    if h_max > 0:
        # weight = -log(|h| / h_max) >= 0; + small offset so the strongest edges (weight 0)
        # are not read as "no edge" by csgraph (which drops values <~1e-8). The offset is
        # negligible for the cutoff (it takes thousands of hops to accumulate to it).
        graph[mask] = -np.log(hop[mask] / h_max) + 1e-3
    dist = sp.sparse.csgraph.shortest_path(graph, directed=False, indices=all_impurity_orbitals)
    return dist, -np.log(coupling_cutoff)


def build_initial_restrictions(
    basis, op: ManyBodyOperator, min_dist=_USE_DEFAULT, coupling_cutoff=_USE_DEFAULT
) -> Optional[dict[frozenset[int], tuple[int, int]]]:
    """Construct the initial occupation restrictions based on Hamiltonian connectivity.

    Parameters
    ----------
    op : ManyBodyOperator
        The Hamiltonian operator.
    min_dist : int, default 4
        Minimum shortest-path distance from the impurity to consider a bath state.

    Returns
    -------
    restrictions : dict of frozenset of int to (int, int), optional
        The initial ground state restrictions, or None if no restrictions were built.
    """
    if coupling_cutoff is _USE_DEFAULT:
        coupling_cutoff = COUPLING_CUTOFF_DEFAULT
    if min_dist is _USE_DEFAULT:
        min_dist = MIN_DIST_DEFAULT
    ground_state_restrictions = {}
    valence_baths, conduction_baths = basis.bath_states

    filled_bath_states = []
    empty_bath_states = []

    all_impurity_orbitals = [
        orb for orb_blocks in basis.impurity_orbitals.values() for orb_block in orb_blocks for orb in orb_block
    ]
    all_valence_orbitals = [
        orb for orb_blocks in valence_baths.values() for orb_block in orb_blocks for orb in orb_block
    ]
    all_conduction_orbitals = [
        orb for orb_blocks in conduction_baths.values() for orb_block in orb_blocks for orb in orb_block
    ]
    tot_orb = len(all_impurity_orbitals) + len(all_valence_orbitals) + len(all_conduction_orbitals)
    dist_matrix, dist_cutoff = _impurity_coupling_distance(
        op, tot_orb, all_impurity_orbitals, coupling_cutoff, min_dist
    )
    for i, impurity_orbitals in basis.impurity_orbitals.items():
        for imp_orb_block, val_orb_block, con_orb_block in zip(
            impurity_orbitals, valence_baths[i], conduction_baths[i]
        ):
            # Identify filled and empty bath states
            # Only restrict states that couple weakly to the impurity (dist above cutoff).
            filled_valence_states = [
                orb for orb in val_orb_block if np.min(dist_matrix[np.ix_(imp_orb_block, [orb])]) > dist_cutoff
            ]
            filled_states = frozenset(sorted(filled_valence_states))
            empty_conduction_states = [
                orb for orb in con_orb_block if np.min(dist_matrix[np.ix_(imp_orb_block, [orb])]) > dist_cutoff
            ]
            empty_states = frozenset(sorted(empty_conduction_states))
            filled_bath_states.append(filled_states)
            empty_bath_states.append(empty_states)
    for filled_orbitals, empty_orbitals in zip(filled_bath_states, empty_bath_states):
        if len(filled_orbitals) > 1:
            ground_state_restrictions[filled_orbitals] = (len(filled_orbitals) - 1, len(filled_orbitals))
        if len(empty_orbitals) > 1:
            ground_state_restrictions[empty_orbitals] = (0, 1)
    if sum(len(rest) for rest in ground_state_restrictions.keys()) == 0:
        return None
    if basis.verbose:
        print("Ground state restrictions:")
        for indices, occupations in ground_state_restrictions.items():
            print(f"---> {sorted(indices)} : {occupations}")
    return ground_state_restrictions


def build_excited_restrictions(
    basis,
    op: ManyBodyOperator,
    psis: Optional[list[ManyBodyState] | ManyBodyState] = None,
    es: Optional[list[float]] = None,
    imp_change: Optional[dict[int, tuple[int, int]]] = None,
    val_change: Optional[dict[int, tuple[int, int]]] = None,
    con_change: Optional[dict[int, tuple[int, int]]] = None,
    cutoff: float = 1e-6,
    min_dist=_USE_DEFAULT,
    coupling_cutoff=_USE_DEFAULT,
):
    """
    Construct restrictions for impurity occupation, valence bath occupation, and conduction bath occupation.
    Restrictions are formed by identifying which bath states are filled, empty, and partially filled, using the density matrices of states psis.
    Filled states are restricted to containing a maximum of one hole. Empty states can have a maximum of one electron. Partially filled states can be empty or filled.
    The total occupations of the impurity, valence band, and conduction band is limited by the ground state occupations with an optional occupation change.
    Arguments:
    =========
    psis: list[ManyBodyState] | ManyBodyState - Eigenstates used to calculate the density matrices.
    imp_change: Optional[tuple[int, int]] - Tuple containing the maximum deviation of the impurity occupation from the ground state occupations, (max_decrease, max_increasess). Default = None
    val_change: Optional[tuple[int, int]] - Tuple containing the maximum deviation of the valence bath occupation from the ground state occupations, (max_decrease, max_increasess). Default = None
    con_change: Optional[tuple[int, int]] - Tuple containing the maximum deviation of the conduction occupation from the ground state occupations, (max_decrease, max_increasess). Default = None
    occ_cutoff: float - Cutoff separating filled, partially filled, and empty states. Filled stated have occupation > 1-occ_cutoff, empty states have occupation < occ_cutoff, partially filled states lie in between
    Returns:
    ========
    excited restrictions: Optional[dict[frozenset[int], tuple[int, int]]] - The occupation restrictions of various orbital indices, or None if no restrictions are generated
    """
    if coupling_cutoff is _USE_DEFAULT:
        coupling_cutoff = COUPLING_CUTOFF_DEFAULT
    if min_dist is _USE_DEFAULT:
        min_dist = MIN_DIST_DEFAULT
    if isinstance(psis, ManyBodyState):
        psis = [psis]
    if psis is not None and len(psis) == 0:
        return None
    if imp_change is None:
        imp_change = dict.fromkeys(basis.impurity_orbitals)
    if val_change is None:
        val_change = dict.fromkeys(basis.impurity_orbitals)
    if con_change is None:
        con_change = dict.fromkeys(basis.impurity_orbitals)

    ground_state_restrictions = get_effective_restrictions(basis)
    excited_restrictions = {}

    valence_baths, conduction_baths = basis.bath_states

    filled_bath_states = []
    empty_bath_states = []

    all_impurity_orbitals = [
        orb for orb_blocks in basis.impurity_orbitals.values() for orb_block in orb_blocks for orb in orb_block
    ]
    all_valence_orbitals = [
        orb for orb_blocks in valence_baths.values() for orb_block in orb_blocks for orb in orb_block
    ]
    all_conduction_orbitals = [
        orb for orb_blocks in conduction_baths.values() for orb_block in orb_blocks for orb in orb_block
    ]
    tot_orb = len(all_impurity_orbitals) + len(all_valence_orbitals) + len(all_conduction_orbitals)
    dist_matrix, dist_cutoff = _impurity_coupling_distance(
        op, tot_orb, all_impurity_orbitals, coupling_cutoff, min_dist
    )
    # Impurity occupation is confined on the *whole* impurity as one subset (never per
    # orbital-symmetry group), so the window bounds only the total impurity charge and does
    # not pin S_z or the eg/t2g ratio. Combine the per-group requested changes into a single
    # (max decrease, max increase); if any group is unconstrained (None), the union is too.
    all_imp_orbs = frozenset(sorted(all_impurity_orbitals))
    imp_changes = [imp_change[i] for i in basis.impurity_orbitals]
    if imp_changes and all(c is not None for c in imp_changes):
        combined_imp_change = (max(c[0] for c in imp_changes), max(c[1] for c in imp_changes))
    else:
        combined_imp_change = None
    imp_min, imp_max = _get_updated_occ_restrictions(ground_state_restrictions, all_imp_orbs, combined_imp_change)

    for i, impurity_orbitals in basis.impurity_orbitals.items():
        val_orbs = frozenset(sorted(orb for block in valence_baths[i] for orb in block))
        min_val, max_val = _get_updated_occ_restrictions(ground_state_restrictions, val_orbs, val_change[i])
        con_orbs = frozenset(sorted(orb for block in conduction_baths[i] for orb in block))
        min_con, max_con = _get_updated_occ_restrictions(ground_state_restrictions, con_orbs, con_change[i])

        if basis.chain_restrict:
            for imp_orb_block, val_orb_block, con_orb_block in zip(
                impurity_orbitals, valence_baths[i], conduction_baths[i]
            ):
                if psis is not None:
                    val_rhos = build_density_matrices(basis, psis, val_orb_block, val_orb_block)
                    con_rhos = build_density_matrices(basis, psis, con_orb_block, con_orb_block)
                    valence_occupations = thermal_average_scale_indep(
                        es, np.diagonal(val_rhos.real, axis1=1, axis2=2), basis.tau
                    )
                    conduction_occupations = thermal_average_scale_indep(
                        es, np.diagonal(con_rhos.real, axis1=1, axis2=2), basis.tau
                    )
                else:
                    valence_occupations = np.ones(len(val_orb_block))
                    conduction_occupations = np.zeros(len(con_orb_block))

                # Identify filled and empty bath states
                # Ignore states that are too close to the impurity
                filled_valence_states = [
                    val_orb_block[orb]
                    for orb in np.nonzero(valence_occupations > 1 - cutoff)[0]
                    if np.min(dist_matrix[np.ix_(imp_orb_block, [val_orb_block[orb]])]) > dist_cutoff
                ]
                filled_conduction_states = [
                    con_orb_block[orb]
                    for orb in np.nonzero(conduction_occupations > 1 - cutoff)[0]
                    if np.min(dist_matrix[np.ix_(imp_orb_block, [con_orb_block[orb]])]) > dist_cutoff
                ]
                filled_states = frozenset(sorted(filled_valence_states + filled_conduction_states))
                empty_valence_states = [
                    val_orb_block[orb]
                    for orb in np.nonzero(valence_occupations < cutoff)[0]
                    if np.min(dist_matrix[np.ix_(imp_orb_block, [val_orb_block[orb]])]) > dist_cutoff
                ]
                empty_conduction_states = [
                    con_orb_block[orb]
                    for orb in np.nonzero(conduction_occupations < cutoff)[0]
                    if np.min(dist_matrix[np.ix_(imp_orb_block, [con_orb_block[orb]])]) > dist_cutoff
                ]
                min_val = max(min_val - len(filled_valence_states) - len(empty_valence_states), 0)
                max_con = max(max_con - len(empty_conduction_states) - len(filled_conduction_states), 0)
                empty_states = frozenset(sorted(empty_valence_states + empty_conduction_states))
                filled_bath_states.append(filled_states)
                empty_bath_states.append(empty_states)

            if basis.collapse_chains:
                filled_bath_states = [
                    frozenset(sorted(orbs for filled_orbs in filled_bath_states for orbs in filled_orbs))
                ]
                empty_bath_states = [frozenset(sorted(orbs for empty_orbs in empty_bath_states for orbs in empty_orbs))]
        else:
            empty_bath_states = [frozenset()]
            filled_bath_states = [frozenset()]
        new_valence_indices = frozenset(
            sorted(orb for orb in val_orbs if not any(orb in s for s in filled_bath_states + empty_bath_states))
        )
        new_conduction_indices = frozenset(
            sorted(orb for orb in con_orbs if not any(orb in s for s in filled_bath_states + empty_bath_states))
        )
        if len(new_valence_indices) > 0 and min_val > 0:
            excited_restrictions[new_valence_indices] = (min_val, len(new_valence_indices))
        if len(new_conduction_indices) > 0 and max_con < len(new_conduction_indices):
            excited_restrictions[new_conduction_indices] = (0, max_con)
        for filled_orbitals, empty_orbitals in zip(filled_bath_states, empty_bath_states):
            if len(filled_orbitals) > 2:
                # Floor occupation: allow up to round(L * hole_fraction) holes. At the default
                # hole_fraction 0.5 this is exactly the historical L // 2.
                filled_min = int(len(filled_orbitals) * (1.0 - CHAIN_FILLED_HOLE_FRACTION))
                excited_restrictions[filled_orbitals] = (filled_min, len(filled_orbitals))
            if len(empty_orbitals) > 2:
                # Cap occupation: allow up to ceil(L * electron_fraction) electrons. At the
                # default electron_fraction 0.5 this is exactly the historical ceil(L / 2).
                empty_max = ceil(len(empty_orbitals) * CHAIN_EMPTY_ELECTRON_FRACTION)
                excited_restrictions[empty_orbitals] = (0, empty_max)
    # Emit the single whole-impurity occupation window (only when it actually confines).
    if len(all_imp_orbs) > 0 and (imp_min > 0 or imp_max < len(all_imp_orbs)):
        excited_restrictions[all_imp_orbs] = (imp_min, imp_max)
    if sum(len(rest) for rest in excited_restrictions.keys()) == 0:
        return None
    return excited_restrictions


def excitation_budget_restriction(bath_states, budget, cost_fn=None, extra_orbitals=None):
    r"""Weighted restriction bounding the *total* bath-excitation cost to ``budget``.

    This is the Phase-3a generalization of the per-chain occupation windows: instead of
    windowing each chain separately, it caps a single weighted sum over the whole bath.
    Taking the reference occupation as "valence baths filled, conduction baths empty" (the
    :func:`get_effective_restrictions` convention), a determinant's excitation cost is

    .. math::
        E = \sum_{o\in\text{val}} c_o (1 - n_o) + \sum_{o\in\text{con}} c_o n_o,

    i.e. every hole in a filled-valence orbital and every electron in an empty-conduction
    orbital costs ``c_o = cost_fn(o)`` (default 1 -- then ``E`` is just the number of bath
    excitations, and ``budget`` is the maximum excitation order). Because
    :math:`E = C + \sum_o w_o n_o` with :math:`w_o = -c_o` on valence, :math:`+c_o` on
    conduction and :math:`C = \sum_{\text{val}} c_o`, the constraint ``0 <= E <= budget``
    is exactly the integer weighted restriction ``(w, (-C, budget - C))`` consumed by
    :meth:`ManyBodyOperator.set_weighted_restrictions`. It composes (AND) with any other
    weighted restriction (e.g. ``S_z``) and, on a Green's-function sector, widens correctly
    through :func:`symmetries.widen_weighted_restrictions` (max ``|weight|`` = max cost).

    A graded budget (weaker-coupled orbitals cheaper to excite) is obtained by passing a
    ``cost_fn`` that quantizes the impurity coupling distance to a small integer; the
    uniform default is the excitation-order budget the Phase-1 profiles motivate.

    Parameters
    ----------
    bath_states : tuple of dict
        ``(valence_baths, conduction_baths)`` as on a :class:`Basis`.
    budget : int
        Maximum total excitation cost admitted.
    cost_fn : callable, optional
        ``orbital -> non-negative int cost``. Default: 1 for every orbital.
    extra_orbitals : iterable of int, optional
        Additional orbitals to treat as conduction-like (cost on occupation); unused by the
        default construction, provided for symmetry with mixed valence/conduction layouts.

    Returns
    -------
    tuple or None
        A single weighted restriction ``(weights, (q_min, q_max))``, or ``None`` when there
        are no bath orbitals / every cost is zero.
    """
    valence_baths, conduction_baths = bath_states
    valence = [o for blocks in valence_baths.values() for blk in blocks for o in blk]
    conduction = [o for blocks in conduction_baths.values() for blk in blocks for o in blk]
    if extra_orbitals:
        conduction = conduction + list(extra_orbitals)
    cost = (lambda o: 1) if cost_fn is None else cost_fn
    weights: dict[int, int] = {}
    filled_cost = 0
    for o in valence:
        c = int(cost(o))
        if c == 0:
            continue
        weights[o] = -c
        filled_cost += c
    for o in conduction:
        c = int(cost(o))
        if c == 0:
            continue
        weights[o] = c
    if not weights:
        return None
    return (weights, (-filled_cost, int(budget) - filled_cost))


def build_weighted_restrictions(bath_states, excitation_budget=None):
    """Assemble the ground-state weighted restrictions a driver enables via its options.

    Currently just the optional excitation budget (see :func:`excitation_budget_restriction`),
    returned as a *list* so more weighted restrictions can be appended later and so it matches
    :meth:`ManyBodyOperator.set_weighted_restrictions`' contract. Returns ``None`` when nothing
    is requested (the drivers then leave ``Basis.weighted_restrictions`` unset).

    Parameters
    ----------
    bath_states : tuple of dict
        ``(valence_baths, conduction_baths)`` in the solver basis.
    excitation_budget : int, optional
        Maximum total bath excitations; ``None`` disables the budget.

    Returns
    -------
    list or None
        A list of ``(weights, (q_min, q_max))`` weighted restrictions, or ``None`` if empty.
    """
    restrictions = []
    if excitation_budget is not None:
        budget = excitation_budget_restriction(bath_states, excitation_budget)
        if budget is not None:
            restrictions.append(budget)
    return restrictions or None
