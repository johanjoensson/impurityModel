from math import ceil
from typing import Any, Optional, Union

try:
    from collections.abc import Iterable
except ModuleNotFoundError:
    from collections import Iterable
import itertools

import numpy as np
import scipy as sp
from mpi4py import MPI
from scipy.cluster.hierarchy import DisjointSet

from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.finite import (
    thermal_average_scale_indep,
)
from impurityModel.ed.manybody_state_containers import (
    SimpleDistributedStateContainer,
)
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
    SlaterDeterminant,
)
from impurityModel.ed.ManyBodyUtils import (
    applyOp as applyOp_test,
)
from impurityModel.ed.mpi_comm import graph_alltoall_psis
from impurityModel.ed.utils import matrix_print


def batched(iterable: Iterable, n: int) -> Iterable:
    """
    batched('ABCDEFG', 3) → ABC DEF G
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def reduce_states(a: list[dict], b: list[dict], _) -> list[dict]:
    """Reduce list of state dicts by summing amplitudes of identical states.

    Parameters
    ----------
    a : list of dict
        Accumulator list of state-to-amplitude dictionaries.
    b : list of dict
        Input list of state-to-amplitude dictionaries.
    _ : Any
        Unused MPI datatype parameter.

    Returns
    -------
    list of dict
        The updated accumulator list of dictionaries.
    """
    res = a.copy()
    for sa, sb in zip(res, b):
        for state, amp in sb.items():
            sa[state] = amp + sa.get(state, 0)
    return res


reduce_states_op = MPI.Op.Create(reduce_states, commute=True)


def reduce_disjoint_set(a: DisjointSet, b: DisjointSet, _) -> DisjointSet:
    """Reduce disjoint sets by merging subsets of b into a.

    Parameters
    ----------
    a : scipy.cluster.hierarchy.DisjointSet
        Accumulator disjoint set.
    b : scipy.cluster.hierarchy.DisjointSet
        Input disjoint set.
    _ : Any
        Unused MPI datatype parameter.

    Returns
    -------
    scipy.cluster.hierarchy.DisjointSet
        The merged disjoint set.
    """
    for subset in b.subsets():
        it = iter(subset)
        root = next(it)
        for item in it:
            a.merge(item, root)
    return a


reduce_disjoint_set_op = MPI.Op.Create(reduce_disjoint_set, commute=True)


def combine_sets(set_1: set, set_2: set, _) -> set:
    """Combine two sets using union.

    Parameters
    ----------
    set_1 : set
        First set.
    set_2 : set
        Second set.
    _ : Any
        Unused MPI datatype parameter.

    Returns
    -------
    set
        The union of the two sets.
    """
    return set_1 | set_2


combine_sets_op = MPI.Op.Create(combine_sets, commute=True)


def reduce_subscript(a: np.ndarray, b: np.ndarray, datatype) -> np.ndarray:
    """MPI reduction operator to combine subscript arrays.

    Replaces None elements in array a with elements from array b.

    Parameters
    ----------
    a : np.ndarray
        Accumulator array.
    b : np.ndarray
        Input array.
    datatype : Any
        The MPI datatype.

    Returns
    -------
    np.ndarray
        The combined subscript array.
    """
    res = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] is None:
                res[i][j] = b[i][j]
            else:
                res[i][j] = a[i][j]
    return res


reduce_subscript_op = MPI.Op.Create(reduce_subscript, commute=True)


def getitem_reduce(a: list, b: list, datatype) -> list:
    """MPI reduction operator to take element-wise maximum of two lists.

    Parameters
    ----------
    a : list
        First list.
    b : list
        Second list.
    datatype : Any
        The MPI datatype.

    Returns
    -------
    list
        List of element-wise maximum values.
    """
    return [max(val_a, val_b) for val_a, val_b in zip(a, b)]


getitem_reduce_op = MPI.Op.Create(getitem_reduce, commute=True)


def getitem_reduce_matrix(a: list[list], b: list[list], datatype) -> list[list]:
    """MPI reduction operator to take element-wise maximum of two 2D lists (matrices).

    Parameters
    ----------
    a : list of list
        First matrix.
    b : list of list
        Second matrix.
    datatype : Any
        The MPI datatype.

    Returns
    -------
    list of list
        Matrix of element-wise maximum values.
    """
    res = [[None for _ in row] for row in a]
    for i in range(len(a)):
        for j in range(len(a[i])):
            res[i][j] = max(a[i][j], b[i][j])
    return res


getitem_reduce_matrix_op = MPI.Op.Create(getitem_reduce_matrix, commute=True)


class Basis:
    """Many-body basis of Slater determinants.

    This class manages the Slater determinant basis states for exact diagonalization,
    supporting distributed states over MPI, restrictions, and basis extensions.
    """

    @property
    def offset(self):
        return self.state_container.offset

    @property
    def size(self):
        return self.state_container.size

    @property
    def local_indices(self):
        return self.state_container.local_indices

    @property
    def _index_dict(self):
        return self.state_container._index_dict

    @property
    def index_bounds(self):
        return self.state_container.index_bounds

    @property
    def state_bounds(self):
        return self.state_container.state_bounds

    @property
    def local_basis(self):
        return self.state_container.local_basis

    def _get_initial_basis(
        self,
        impurity_orbitals: dict[int, list[list[int]]],
        bath_states: tuple[dict[int, list[list[int]]], dict[int, list[list[int]]]],
        delta_valence_occ: Optional[dict[int, int]],
        delta_conduction_occ: Optional[dict[int, int]],
        delta_impurity_occ: Optional[dict[int, int]],
        nominal_impurity_occ: dict[int, int],
        mixed_valence: dict[int, int],
        verbose: bool,
    ) -> tuple[list[SlaterDeterminant], int]:
        """Construct the initial basis of Slater determinants.

        Parameters
        ----------
        impurity_orbitals : dict
            Impurity orbitals grouped by l quantum number.
        bath_states : tuple of dict
            Valence and conduction bath states grouped by l quantum number.
        delta_valence_occ : dict, optional
            Allowed valence bath occupation variations.
        delta_conduction_occ : dict, optional
            Allowed conduction bath occupation variations.
        delta_impurity_occ : dict, optional
            Allowed impurity occupation variations.
        nominal_impurity_occ : dict
            Nominal impurity occupations.
        mixed_valence : dict
            Allowed mixed valence variations.
        verbose : bool
            Whether to print configuration details.

        Returns
        -------
        basis : list of SlaterDeterminant
            The list of constructed initial Slater determinants.
        num_spin_orbitals : int
            The total number of spin orbitals.
        """
        valence_baths, conduction_baths = bath_states
        total_baths = {
            i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
            for i in valence_baths
        }

        if delta_valence_occ is None:
            delta_valence_occ = dict.fromkeys(impurity_orbitals.keys(), 0)
        if delta_conduction_occ is None:
            delta_conduction_occ = dict.fromkeys(impurity_orbitals.keys(), 0)
        if delta_impurity_occ is None:
            delta_impurity_occ = dict.fromkeys(impurity_orbitals.keys(), 0)

        total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
        # Per group, materialise the allowed configurations tagged with their impurity
        # occupation, as (impurity_occupation, occupied_orbital_tuple). Materialising (rather
        # than keeping lazy nested itertools iterators) avoids re-consuming an exhausted
        # iterator when several groups each admit multiple occupations, and lets the cross-group
        # combination below be filtered by *total* impurity charge.
        # When the impurity is split into several orbital-symmetry manifolds (this grouping),
        # they are one correlated shell that must freely redistribute charge among manifolds at
        # fixed *total* occupation. A single group already enumerates every whole-impurity
        # arrangement through ``combinations`` below, so its per-group occupation stays pinned to
        # ``nominal +/- mixed_valence`` (preserving the seed count for the un-grouped case); but
        # with >= 2 groups each group's occupation must range freely (the cross-group *total*
        # filter then keeps only the arrangements in the occupation window). Gating the per-group
        # range by ``mixed_valence[i]`` in the grouped case instead pins each manifold and
        # collapses the seed to a single frozen configuration -- the NiO covalency /
        # magnetic-moment regression. ``mixed_valence`` still widens the *total* window via
        # ``total_slack``.
        redistribute = len(impurity_orbitals) > 1
        group_configurations = {}
        for i in valence_baths:
            configs = []
            impurity_electron_indices = [orb for imp_orbs in impurity_orbitals[i] for orb in imp_orbs]
            valence_electron_indices = [orb for val_orbs in valence_baths[i] for orb in val_orbs]
            conduction_electron_indices = [orb for con_orbs in conduction_baths[i] for orb in con_orbs]
            occ_lo = 0 if redistribute else max(0, nominal_impurity_occ[i] - abs(mixed_valence[i]))
            occ_hi = (
                total_impurity_orbitals[i]
                if redistribute
                else min(total_impurity_orbitals[i], nominal_impurity_occ[i] + abs(mixed_valence[i]))
            )
            for nominal_occ in range(occ_lo, occ_hi + 1):
                for delta_valence in range(delta_valence_occ[i] + 1):
                    for delta_conduction in range(delta_conduction_occ[i] + 1):
                        delta_impurity = delta_valence - delta_conduction
                        if (
                            abs(delta_impurity) <= abs(delta_impurity_occ[i])
                            and nominal_occ + delta_impurity <= total_impurity_orbitals[i]
                            and nominal_occ + delta_impurity >= 0
                            and delta_valence <= len(valence_electron_indices)
                        ):
                            impurity_occupation = nominal_occ + delta_impurity
                            valence_occupation = len(valence_electron_indices) - delta_valence
                            conduction_occupation = delta_conduction
                            if verbose:
                                print(f"Partition {i} occupations")
                                print(f"Impurity occupation:   {impurity_occupation:d}")
                                print(f"Valence occupation:   {valence_occupation:d}")
                                print(f"Conduction occupation: {conduction_occupation:d}")
                            for imp_c, val_c, con_c in itertools.product(
                                itertools.combinations(impurity_electron_indices, impurity_occupation),
                                itertools.combinations(valence_electron_indices, valence_occupation),
                                itertools.combinations(conduction_electron_indices, conduction_occupation),
                            ):
                                configs.append((impurity_occupation, imp_c + val_c + con_c))
            group_configurations[i] = configs
        num_spin_orbitals = sum(total_impurity_orbitals[i] + total_baths[i] for i in total_baths)

        # Total impurity-occupation window: the seed spans a fixed *total* impurity charge (the
        # sum of the per-group nominal occupations), widened by the whole-impurity charge
        # excursion budget. The groups are the orbital-symmetry manifolds (eg / t2g, spin up /
        # down) of ONE correlated shell sharing ONE charge reservoir (the bath), so the total
        # excursion is bounded by the *largest* per-group budget (``max``), NOT their sum: the
        # per-group ``mixed_valence`` / ``delta_impurity_occ`` are wide enough to let each
        # manifold redistribute, but summing them would multiply a uniform search widening (e.g.
        # the prescan sets the same ``scan_width`` on every group) by the number of groups and
        # blow the window open — which let the ground-state prescan discover an unphysical
        # empty-impurity sector (the NiO d8 -> d2 regression). Filtering the cross-group product
        # on this bounded total keeps wide per-manifold windows from leaking total charge, so the
        # manifolds redistribute charge at fixed impurity count while a single group keeps its
        # full impurity/bath charge-transfer range (the filter is then a no-op).
        total_nominal = sum(int(nominal_impurity_occ[i]) for i in valence_baths)
        total_slack = max((abs(mixed_valence[i]) + abs(delta_impurity_occ[i]) for i in valence_baths), default=0)
        lo_tot = max(0, total_nominal - total_slack)
        hi_tot = total_nominal + total_slack

        basis = []
        # Combine the per-group configurations, keeping only determinants whose *total* impurity
        # occupation lies in the window.
        for combo in itertools.product(*group_configurations.values()):
            if not (lo_tot <= sum(imp_occ for imp_occ, _ in combo) <= hi_tot):
                continue
            occupied = tuple(idx for _imp_occ, orbs in combo for idx in orbs)
            basis.append(psr.tuple2bytes(occupied, 8 * self.n_bytes))

        return [SlaterDeterminant.from_bytes(bytestring) for bytestring in basis], num_spin_orbitals

    def get_effective_restrictions(self) -> dict[frozenset[int], tuple[int, int]]:
        """Calculate the actual min/max occupations observed across the current basis.

        Returns
        -------
        restrictions : dict of frozenset of int to (int, int)
            Dictionary mapping orbital subsets to their observed (min, max) occupations.
        """
        valence_baths, conduction_baths = self.bath_states
        restrictions = {}

        # Impurity occupation is restricted on the *whole* impurity as a single subset (never
        # per orbital-symmetry group). A per-group impurity pin would fix S_z (spin groups) or
        # the eg/t2g ratio (manifold groups) and destroy spin-multiplet degeneracy / confine the
        # charge; the union bound confines only the total impurity charge, leaving the manifolds
        # free to redistribute at fixed count.
        all_impurity_indices = frozenset(
            sorted(ind for imp_blocks in self.impurity_orbitals.values() for imp_ind in imp_blocks for ind in imp_ind)
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
        for state in self.local_basis:
            bits = psr.bytes2bitarray(bytes(state.to_bytearray()), self.num_spin_orbitals)
            n_imp = sum(bits[i] for i in all_impurity_indices)
            max_imp = max(max_imp, n_imp)
            min_imp = min(min_imp, n_imp)
            for i in valence_baths:
                n_val = sum(bits[j] for j in valence_index_sets[i])
                n_con = sum(bits[j] for j in conduction_index_sets[i])
                min_val[i] = min(min_val[i], n_val)
                max_con[i] = max(max_con[i], n_con)
        if self.is_distributed:
            max_imp = self.comm.allreduce(max_imp, op=MPI.MAX)
            min_imp = self.comm.allreduce(min_imp, op=MPI.MIN)
        if len(all_impurity_indices) > 0:
            restrictions[all_impurity_indices] = (min_imp, max_imp)
        for i in sorted(valence_baths.keys()):
            v_min, c_max = min_val[i], max_con[i]
            if self.is_distributed:
                v_min = self.comm.allreduce(v_min, op=MPI.MIN)
                c_max = self.comm.allreduce(c_max, op=MPI.MAX)
            # Valence baths sit filled (max = full), conduction empty (min = 0) by convention.
            if len(valence_index_sets[i]) > 0:
                restrictions[valence_index_sets[i]] = (v_min, len(valence_index_sets[i]))
            if len(conduction_index_sets[i]) > 0:
                restrictions[conduction_index_sets[i]] = (0, c_max)
        return restrictions

    def _get_updated_occ_restrictions(
        self, restrictions: dict[frozenset[int], tuple[int, int]], orbs: frozenset[int], dN: Optional[tuple[int, int]]
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

    def _impurity_coupling_distance(self, op, tot_orb, all_impurity_orbitals, coupling_cutoff, min_dist):
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
            dist = sp.sparse.csgraph.shortest_path(
                graph, directed=False, unweighted=True, indices=all_impurity_orbitals
            )
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
        self, op: ManyBodyOperator, min_dist: int = 4, coupling_cutoff: float = 1e-3
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
        ground_state_restrictions = {}
        valence_baths, conduction_baths = self.bath_states

        filled_bath_states = []
        empty_bath_states = []

        all_impurity_orbitals = [
            orb for orb_blocks in self.impurity_orbitals.values() for orb_block in orb_blocks for orb in orb_block
        ]
        all_valence_orbitals = [
            orb for orb_blocks in valence_baths.values() for orb_block in orb_blocks for orb in orb_block
        ]
        all_conduction_orbitals = [
            orb for orb_blocks in conduction_baths.values() for orb_block in orb_blocks for orb in orb_block
        ]
        tot_orb = len(all_impurity_orbitals) + len(all_valence_orbitals) + len(all_conduction_orbitals)
        dist_matrix, dist_cutoff = self._impurity_coupling_distance(
            op, tot_orb, all_impurity_orbitals, coupling_cutoff, min_dist
        )
        for i, impurity_orbitals in self.impurity_orbitals.items():
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
        if self.verbose:
            print("Ground state restrictions:")
            for indices, occupations in ground_state_restrictions.items():
                print(f"---> {sorted(indices)} : {occupations}")
        return ground_state_restrictions

    def build_excited_restrictions(
        self,
        op: ManyBodyOperator,
        psis: Optional[list[ManyBodyState] | ManyBodyState] = None,
        es: Optional[list[float]] = None,
        imp_change: Optional[dict[int, tuple[int, int]]] = None,
        val_change: Optional[dict[int, tuple[int, int]]] = None,
        con_change: Optional[dict[int, tuple[int, int]]] = None,
        cutoff: float = 1e-6,
        min_dist=4,
        coupling_cutoff: float = 1e-3,
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
        if isinstance(psis, ManyBodyState):
            psis = [psis]
        if psis is not None and len(psis) == 0:
            return None
        if imp_change is None:
            imp_change = dict.fromkeys(self.impurity_orbitals)
        if val_change is None:
            val_change = dict.fromkeys(self.impurity_orbitals)
        if con_change is None:
            con_change = dict.fromkeys(self.impurity_orbitals)

        ground_state_restrictions = self.get_effective_restrictions()
        excited_restrictions = {}

        valence_baths, conduction_baths = self.bath_states

        filled_bath_states = []
        empty_bath_states = []

        all_impurity_orbitals = [
            orb for orb_blocks in self.impurity_orbitals.values() for orb_block in orb_blocks for orb in orb_block
        ]
        all_valence_orbitals = [
            orb for orb_blocks in valence_baths.values() for orb_block in orb_blocks for orb in orb_block
        ]
        all_conduction_orbitals = [
            orb for orb_blocks in conduction_baths.values() for orb_block in orb_blocks for orb in orb_block
        ]
        tot_orb = len(all_impurity_orbitals) + len(all_valence_orbitals) + len(all_conduction_orbitals)
        dist_matrix, dist_cutoff = self._impurity_coupling_distance(
            op, tot_orb, all_impurity_orbitals, coupling_cutoff, min_dist
        )
        if False and self.verbose:
            matrix_print(dist_matrix[:, :], "Orbital distance matrix", flush=True)

        # Impurity occupation is confined on the *whole* impurity as one subset (never per
        # orbital-symmetry group), so the window bounds only the total impurity charge and does
        # not pin S_z or the eg/t2g ratio. Combine the per-group requested changes into a single
        # (max decrease, max increase); if any group is unconstrained (None), the union is too.
        all_imp_orbs = frozenset(sorted(all_impurity_orbitals))
        imp_changes = [imp_change[i] for i in self.impurity_orbitals]
        if imp_changes and all(c is not None for c in imp_changes):
            combined_imp_change = (max(c[0] for c in imp_changes), max(c[1] for c in imp_changes))
        else:
            combined_imp_change = None
        imp_min, imp_max = self._get_updated_occ_restrictions(ground_state_restrictions, all_imp_orbs, combined_imp_change)

        for i, impurity_orbitals in self.impurity_orbitals.items():
            val_orbs = frozenset(sorted(orb for block in valence_baths[i] for orb in block))
            min_val, max_val = self._get_updated_occ_restrictions(ground_state_restrictions, val_orbs, val_change[i])
            con_orbs = frozenset(sorted(orb for block in conduction_baths[i] for orb in block))
            min_con, max_con = self._get_updated_occ_restrictions(ground_state_restrictions, con_orbs, con_change[i])

            if self.chain_restrict:
                for imp_orb_block, val_orb_block, con_orb_block in zip(
                    impurity_orbitals, valence_baths[i], conduction_baths[i]
                ):
                    if psis is not None:
                        val_rhos = self.build_density_matrices(psis, val_orb_block, val_orb_block)
                        con_rhos = self.build_density_matrices(psis, con_orb_block, con_orb_block)
                        valence_occupations = thermal_average_scale_indep(
                            es, np.diagonal(val_rhos.real, axis1=1, axis2=2), self.tau
                        )
                        conduction_occupations = thermal_average_scale_indep(
                            es, np.diagonal(con_rhos.real, axis1=1, axis2=2), self.tau
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

                if self.collapse_chains:
                    filled_bath_states = [
                        frozenset(sorted(orbs for filled_orbs in filled_bath_states for orbs in filled_orbs))
                    ]
                    empty_bath_states = [
                        frozenset(sorted(orbs for empty_orbs in empty_bath_states for orbs in empty_orbs))
                    ]
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
                    excited_restrictions[filled_orbitals] = (len(filled_orbitals) // 2, len(filled_orbitals))
                    # excited_restrictions[filled_orbitals] = (len(filled_orbitals) - 1, len(filled_orbitals))
                if len(empty_orbitals) > 2:
                    excited_restrictions[empty_orbitals] = (0, ceil(len(empty_orbitals) / 2))
                    # excited_restrictions[empty_orbitals] = (0, 1)
        # Emit the single whole-impurity occupation window (only when it actually confines).
        if len(all_imp_orbs) > 0 and (imp_min > 0 or imp_max < len(all_imp_orbs)):
            excited_restrictions[all_imp_orbs] = (imp_min, imp_max)
        if sum(len(rest) for rest in excited_restrictions.keys()) == 0:
            return None
        return excited_restrictions

    def __init__(
        self,
        impurity_orbitals,
        bath_states,
        nominal_impurity_occ=None,
        mixed_valence=None,
        initial_basis=None,
        restrictions=None,
        weighted_restrictions=None,
        split_threshold=1.0,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        truncation_threshold=np.inf,
        spin_flip_dj=False,
        tau=0,
        chain_restrict=False,
        collapse_chains=False,
        comm=None,
        verbose=True,
        debug=False,
    ):
        """Initialize the Basis class.

        Parameters
        ----------
        impurity_orbitals : dict
            Impurity orbitals.
        bath_states : tuple of dict
            Valence and conduction bath states.
        nominal_impurity_occ : dict, optional
            Nominal impurity occupations.
        mixed_valence : dict, optional
            Mixed valence bounds.
        initial_basis : list of SlaterDeterminant or bytes, optional
            Predefined initial states.
        restrictions : dict, optional
            Initial occupation restrictions.
        delta_valence_occ : dict, optional
            Allowed valence occupation variations.
        delta_conduction_occ : dict, optional
            Allowed conduction occupation variations.
        delta_impurity_occ : dict, optional
            Allowed impurity occupation variations.
        truncation_threshold : float, default np.inf
            Threshold for truncating states.
        spin_flip_dj : bool, default False
            Whether to enable spin-flip states.
        tau : float, default 0
            Tau parameter.
        chain_restrict : bool, default False
            Whether to restrict chain states.
        collapse_chains : bool, default False
            Whether to collapse chains.
        comm : MPI.Comm, optional
            MPI communicator.
        verbose : bool, default True
            Whether to print info.
        debug : bool, default False
            Debug flag.
        """
        assert (
            impurity_orbitals is not None
        ), "You need to supply the number of impurity orbitals in each set in impurity_orbitals"
        assert bath_states is not None, "You need to supply the number of bath states for each l quantum number"

        self.num_spin_orbitals = sum(
            sum(len(orbs) for orbs in impurity_orbitals[i])
            + sum(len(orbs) for orbs in bath_states[0][i])
            + sum(len(orbs) for orbs in bath_states[1][i])
            for i in bath_states[0]
        )
        test = ManyBodyState({SlaterDeterminant.from_bytes(b"\x00"): 1.0})
        slater_det = list(test.keys())[0]
        self.type = type(slater_det)
        self.n_bytes = int(ceil(ceil(self.num_spin_orbitals / 8) / len(slater_det)) * len(slater_det))

        self.truncation_threshold = truncation_threshold
        self.is_distributed = comm is not None and comm.size > 1
        self.tau = tau

        if initial_basis is not None:
            assert nominal_impurity_occ is None
            assert delta_valence_occ is None
            assert delta_conduction_occ is None
            assert delta_impurity_occ is None
            initial_basis = [
                self.type.from_bytes(state) if isinstance(state, bytes) else state for state in initial_basis
            ]
        else:
            assert nominal_impurity_occ is not None
            initial_basis, num_spin_orbitals = self._get_initial_basis(
                impurity_orbitals=impurity_orbitals,
                bath_states=bath_states,
                delta_valence_occ=delta_valence_occ,
                delta_conduction_occ=delta_conduction_occ,
                delta_impurity_occ=delta_impurity_occ,
                nominal_impurity_occ=nominal_impurity_occ,
                mixed_valence=mixed_valence if mixed_valence is not None else dict.fromkeys(nominal_impurity_occ, 0),
                verbose=verbose,
            )
        self.impurity_orbitals = impurity_orbitals
        self.bath_states = bath_states
        self.spin_flip_dj = spin_flip_dj
        self.chain_restrict = chain_restrict
        self.collapse_chains = collapse_chains
        self.verbose = verbose
        self.debug = debug
        self.comm = comm
        self.restrictions = restrictions
        # Weighted-sum restrictions (e.g. S_z), list of (weights, (q_min, q_max)); see
        # ManyBodyOperator.set_weighted_restrictions. None = none.
        self.weighted_restrictions = weighted_restrictions
        # Adaptive MPI split policy (Phase 7): cap the number of split colors near the
        # participation ratio of the block costs, scaled by split_threshold. Larger =>
        # split more aggressively; 0 => never split (unified communicator). 1.0 keeps the
        # legacy max-split behaviour for equally-weighted blocks.
        self.split_threshold = split_threshold

        # self.state_container = CentralizedStateContainer(
        self.state_container = SimpleDistributedStateContainer(
            # self.state_container = DistributedStateContainer(
            initial_basis,
            bytes_per_state=self.n_bytes,
            comm=self.comm,
            verbose=verbose,
        )

    def clone(self, initial_basis=None, restrictions=None, weighted_restrictions=None, verbose=None, comm=None):
        """Create a new Basis instance, optionally overriding initial_basis and restrictions.

        If initial_basis is None, the new basis will start with self.local_basis.
        If restrictions is None, the new basis will inherit self.restrictions.
        If weighted_restrictions is None, the new basis inherits self.weighted_restrictions.
        If comm is None, the new basis will inherit self.comm.
        """
        return Basis(
            impurity_orbitals=self.impurity_orbitals,
            bath_states=self.bath_states,
            initial_basis=initial_basis if initial_basis is not None else list(self.local_basis),
            restrictions=restrictions if restrictions is not None else self.restrictions,
            weighted_restrictions=(
                weighted_restrictions if weighted_restrictions is not None else self.weighted_restrictions
            ),
            split_threshold=self.split_threshold,
            truncation_threshold=self.truncation_threshold,
            spin_flip_dj=self.spin_flip_dj,
            tau=self.tau,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            comm=comm if comm is not None else self.comm,
            verbose=verbose if verbose is not None else self.verbose,
            debug=self.debug,
        )

    def free_comm(self):
        """
        Free the split/custom MPI communicator associated with this Basis.
        This must be called collectively by all ranks sharing the communicator.
        """
        if self.comm is not None and self.comm != MPI.COMM_NULL:
            self.comm.Free()
            self.comm = None
        if hasattr(self, "state_container") and self.state_container is not None:
            self.state_container.comm = None

    def alltoall_states(self, send_list: list[list[bytes]], flatten: bool = False) -> list[list[bytes]] | list[bytes]:
        """Distribute basis states to their owners across MPI ranks.

        Parameters
        ----------
        send_list : list of list of bytes
            The states to send to each rank.
        flatten : bool, default False
            If True, return a flat list of bytes.

        Returns
        -------
        list of list of bytes or list of bytes
            The received states.
        """
        return self.state_container.alltoall_states(send_list, flatten)

    def add_states(self, new_states: Iterable[bytes], unique_sorted=False) -> None:
        """
        Extend the current basis by adding the new_states to it.
        """
        self.state_container.add_states(new_states, unique_sorted)

    def redistribute_psis(self, psis: list[ManyBodyState]) -> list[ManyBodyState]:
        """Redistribute wavefunctions across MPI ranks based on state ownership.

        Parameters
        ----------
        psis : list of ManyBodyState
            The wavefunctions to redistribute.

        Returns
        -------
        list of ManyBodyState
            The redistributed wavefunctions.
        """
        if isinstance(psis, ManyBodyState):
            print("WARNING in redistribute_psi:")
            print(
                "Expetced a list of ManyBodyStates, received a single ManyBodyState. Remaking into list of one ManyBodyState"
            )
            psis = [psis]
        psis = [
            (
                psi
                if isinstance(psi, ManyBodyState)
                else ManyBodyState(
                    {(SlaterDeterminant.from_bytes(k) if isinstance(k, bytes) else k): v for k, v in psi.items()}
                )
            )
            for psi in psis
        ]
        if not self.is_distributed:
            return psis

        comm = self.comm

        res = graph_alltoall_psis(psis, self.n_bytes, comm)
        return res

    def _generate_spin_flipped_determinants(self, determinants: Iterable[SlaterDeterminant]) -> set[SlaterDeterminant]:
        """Generate spin-flipped counterparts for a collection of determinants.

        Parameters
        ----------
        determinants : Iterable of SlaterDeterminant
            The starting Slater determinants to spin-flip.

        Returns
        -------
        set of SlaterDeterminant
            The original determinants plus their spin-flipped counterparts.
        """
        valence_baths, conduction_baths = self.bath_states
        n_dn_op = {
            ((i, "c"), (i, "a")): 1.0
            for l in self.impurity_orbitals
            for i in range(sum(len(orbs) for orbs in self.impurity_orbitals[l]) // 2)
        }
        n_up_op = {
            ((i, "c"), (i, "a")): 1.0
            for l in self.impurity_orbitals
            for i in range(
                sum(len(orbs) for orbs in self.impurity_orbitals[l]) // 2,
                sum(len(orbs) for orbs in self.impurity_orbitals[l]),
            )
        }
        n_dn_mbo = ManyBodyOperator(n_dn_op)
        n_up_mbo = ManyBodyOperator(n_up_op)
        spin_flip = set()
        for det in determinants:
            n_dn = int(applyOp_test(n_dn_mbo, ManyBodyState({det: 1.0}), cutoff=0).get(det, 0).real)
            n_up = int(applyOp_test(n_up_mbo, ManyBodyState({det: 1.0}), cutoff=0).get(det, 0).real)
            spin_flip.add(det)
            to_flip = {det}
            for l in self.impurity_orbitals:
                n_orb = sum(len(orbs) for orbs in self.impurity_orbitals[l])
                for i in range(n_orb // 2):
                    spin_flip_op = {
                        ((i + n_orb // 2, "c"), (i, "a")): 1.0,
                        ((i, "c"), (i + n_orb // 2, "a")): 1.0,
                    }
                    spin_flip_mbo = ManyBodyOperator(spin_flip_op)
                    for state in list(to_flip):
                        flipped = applyOp_test(spin_flip_mbo, ManyBodyState({state: 1.0}), cutoff=0)
                        to_flip.update(flipped.keys())
                        if len(flipped) == 0:
                            continue
                        flipped_state = list(flipped.keys())[0]
                        new_n_dn = int(
                            applyOp_test(n_dn_mbo, ManyBodyState({flipped_state: 1.0}), cutoff=0)
                            .get(flipped_state, 0)
                            .real
                        )
                        new_n_up = int(
                            applyOp_test(n_up_mbo, ManyBodyState({flipped_state: 1.0}), cutoff=0)
                            .get(flipped_state, 0)
                            .real
                        )
                        if (new_n_dn == n_dn and new_n_up == n_up) or (new_n_dn == n_up and new_n_up == n_dn):
                            spin_flip.update(flipped.keys())

        return spin_flip

    def expand(self, op, slaterWeightMin=0, max_it=5):
        """
        Expand the basis by repeatedly applying an operator
        to the basis states, thus generating new basis states.
        The basis will probably explode!

        Parameters:
        ===========
        op: ManyBodyOperator - The operator to apply, over and over again.
        slaterWeightMin: float - Minimum weight for slater determinants to be kept (default: 0).
        max_it: int - Apply the operator at most this number of times (default: 5)

        Returns:
        ========
        op_dict: dict - Dictionary of slater determinants as keys and ManyBodyStates as values.
                        Applying the operator to a key results in the ManyBodyState mapped to
                        by that key.
        """
        if isinstance(op, dict):
            op = ManyBodyOperator(op)
        op.set_restrictions(self.restrictions)
        # Unconditional (like set_restrictions): passing None clears any stale weighted
        # mask left on a reused operator object.
        op.set_weighted_restrictions(self.weighted_restrictions)
        old_size = self.size - 1

        it = 0
        max_inner_loops = 2

        local_states = set(self.local_basis)
        apply_h_to_these = local_states
        while old_size < self.size and it < max(max_it // max_inner_loops, 1):
            for _ in range(max_inner_loops):
                new_local_states = set()
                for state in apply_h_to_these:
                    res = applyOp_test(
                        op,
                        ManyBodyState({state: 1}),
                        cutoff=slaterWeightMin,
                    )
                    new_local_states |= set(state for state in res.keys()) - local_states
                if len(new_local_states) == 0:
                    break
                apply_h_to_these = new_local_states
                local_states |= new_local_states
            new_states = local_states - set(self.local_basis)
            if self.spin_flip_dj:
                new_states = self._generate_spin_flipped_determinants(new_states)
            old_size = self.size

            n_new_states = len(new_states)
            if self.is_distributed:
                n_new_states = self.comm.allreduce(n_new_states, op=MPI.SUM)
            if self.size + n_new_states > self.truncation_threshold:
                break
            self.add_states(new_states)
            apply_h_to_these = apply_h_to_these ^ (set(self.local_basis) - local_states)
            it += 1
        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.")
        # return self.build_operator_dict(op)

    def index(self, val: SlaterDeterminant) -> int:
        """Find the global index of a Slater determinant in the basis.

        Parameters
        ----------
        val : SlaterDeterminant
            The Slater determinant to look up.

        Returns
        -------
        int
            The global index of the determinant.
        """
        return self.state_container.index(val)

    def __getitem__(self, key: int | slice) -> SlaterDeterminant | list[SlaterDeterminant]:
        """Get the Slater determinant(s) at the specified index or slice.

        Parameters
        ----------
        key : int or slice
            The index or slice of basis states to retrieve.

        Returns
        -------
        SlaterDeterminant or list of SlaterDeterminant
            The Slater determinant at the index, or list of determinants for a slice.
        """
        return self.state_container[key]

    def __len__(self) -> int:
        """Get the total size of the basis.

        Returns
        -------
        int
            The total number of Slater determinants in the basis.
        """
        return self.state_container.size

    def __contains__(self, item: SlaterDeterminant | bytes) -> bool:
        """Check if a Slater determinant or its byte representation is in the basis.

        Parameters
        ----------
        item : SlaterDeterminant or bytes
            The state to search for.

        Returns
        -------
        bool
            True if the state is in the basis, False otherwise.
        """
        return item in self.state_container

    def contains(self, item: Iterable[SlaterDeterminant | bytes]) -> np.ndarray:
        """Check containment for an iterable of states.

        Parameters
        ----------
        item : Iterable of SlaterDeterminant or bytes
            The collection of states to check.

        Returns
        -------
        np.ndarray of bool
            Boolean array indicating containment for each state.
        """
        return self.state_container.contains(item)

    def __iter__(self) -> Iterable[SlaterDeterminant]:
        """Iterate over all Slater determinants in the basis.

        Yields
        ------
        SlaterDeterminant
            The next Slater determinant in the basis.
        """
        for state in self.state_container:
            yield state

    def copy(self) -> "Basis":
        """Create a copy of this Basis.

        Returns
        -------
        Basis
            A new Basis object with identical states and parameters.
        """
        return Basis(
            self.impurity_orbitals,
            self.bath_states,
            initial_basis=self.local_basis,
            restrictions=self.restrictions,
            weighted_restrictions=self.weighted_restrictions,
            split_threshold=self.split_threshold,
            spin_flip_dj=self.spin_flip_dj,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            verbose=self.verbose,
        )

    def clear(self) -> None:
        """Clear all states from the basis."""
        self.state_container.clear()
        self.add_states([])

    def build_vector(
        self, psis: list[ManyBodyState], root: Optional[int] = None, slaterWeightMin: float = 0
    ) -> np.ndarray:
        """Build a dense matrix representation of wavefunctions in the basis.

        Parameters
        ----------
        psis : list of ManyBodyState
            The wavefunctions to represent.
        root : int, optional
            MPI rank to reduce the vector to. If None, it is reduced to all ranks.
        slaterWeightMin : float, default 0
            Minimum amplitude threshold below which coefficients are ignored.

        Returns
        -------
        np.ndarray
            The 2D dense matrix representation of the wavefunctions.
        """
        v = np.zeros((len(psis), self.size), dtype=complex, order="C")
        # psis = self.redistribute_psis(psis)
        # row_states_in_basis: list[bytes] = []
        # row_dict = {state: self._index_dict[state] for state in self.local_basis}
        # col_dict = dict(zip(self.local_basis, range(self.local_indices.start, self.local_indices.stop)))
        _index_dict = self._index_dict
        for row, psi in enumerate(psis):
            for state, val in psi.items():
                idx = _index_dict.get(state)
                if idx is None or abs(val) < slaterWeightMin:
                    continue
                v[row, idx] = val

        if self.is_distributed and root is None:
            self.comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
        elif self.is_distributed:
            self.comm.Reduce(MPI.IN_PLACE if self.comm.rank == root else v, v, op=MPI.SUM, root=root)
        return v

    def build_distributed_vector(self, psis: list[ManyBodyState], dtype: Any = complex) -> np.ndarray:
        """Build the MPI-local portion of a wavefunction vector.

        Parameters
        ----------
        psis : list of ManyBodyState
            The wavefunctions to represent.
        dtype : Any, default complex
            The data type of the returned array.

        Returns
        -------
        np.ndarray
            The 2D array containing the local amplitudes.
        """
        psis = self.redistribute_psis(psis)
        v = np.empty((len(psis), len(self.local_basis)), dtype=dtype, order="C")
        for (row, psi), (col, state) in itertools.product(enumerate(psis), enumerate(self.local_basis)):
            v[row, col] = psi.get(state, 0)
        return v

    def build_state(self, vs: Union[list[np.ndarray], np.ndarray], slaterWeightMin: float = 0) -> list[ManyBodyState]:
        """Convert dense vectors back to a list of ManyBodyState objects.

        Parameters
        ----------
        vs : list of np.ndarray or np.ndarray
            The dense vector representations.
        slaterWeightMin : float, default 0
            Minimum amplitude threshold to keep.

        Returns
        -------
        list of ManyBodyState
            The corresponding list of many-body states.
        """
        if isinstance(vs, np.matrix):
            vs = vs.A
        if isinstance(vs, np.ndarray) and len(vs.shape) == 1:
            vs = vs.reshape((1, vs.shape[0]))
        if isinstance(vs, list):
            vs = np.array(vs)
        res = [ManyBodyState({}) for _ in range(vs.shape[0])]
        if vs.shape[1] == self.size:
            for j, i in np.argwhere(np.abs(vs[:, self.local_indices]) > slaterWeightMin):
                res[j][self.local_basis[i]] = vs[j, i + self.offset]
        elif vs.shape[1] == len(self.local_basis):
            for j, i in np.argwhere(np.abs(vs) > slaterWeightMin):
                res[j][self.local_basis[i]] = vs[j, i]
        else:
            raise RuntimeError(
                f"The dimensions of the input dense vector does not match a distributed, or full vector.\n{vs.shape} != ({vs.shape[0]}, {self.size}) || ({vs.shape[0]}, {len(self.local_basis)})"
            )
        return res

    def build_local_operator_list(self, op, slaterWeightMin):
        """
        Apply the operator to all (MPI local) basis states, in order.
        Return the results in a list.
        """
        res = []
        unit_state = ManyBodyState()
        for state in self.local_basis:
            unit_state[state] = 1.0
            res.append(applyOp_test(op, unit_state, cutoff=slaterWeightMin))
            unit_state.erase(state)
        return res

    def build_operator_dict(self, op, slaterWeightMin=0):
        """
        Express the operator, op, in the current basis. Do not expand the basis.
        Return a dict containing the results of applying op to the different basis states
        """
        if isinstance(op, dict):
            op = ManyBodyOperator(op)
        return dict(zip(self.local_basis, self.build_local_operator_list(op, slaterWeightMin)))

    def build_dense_matrix(self, op, distribute=True):
        """
        Get the operator as a dense matrix in the current basis.
        by default the dense matrix is distributed to all ranks.
        """
        h_local = self.build_sparse_matrix(op)
        if self.is_distributed:
            h = np.empty(h_local.shape, dtype=h_local.dtype)
            self.comm.Allreduce(h_local.todense(), h, op=MPI.SUM)
        else:
            h = h_local.todense(order="C")
        return h

    def build_sparse_matrix(self, op: ManyBodyOperator):
        """
        Get the operator as a sparse matrix in the current basis.
        The sparse matrix is distributed over all ranks.
        """
        if isinstance(op, dict):
            op = ManyBodyOperator(op)

        rows = []
        cols = []
        vals = []
        _index_dict = self._index_dict
        if not self.is_distributed:
            for ket, ket_state in zip(self.local_basis, self.build_local_operator_list(op, 0)):
                col = _index_dict[ket]
                for bra, val in ket_state.items():
                    row = _index_dict.get(bra)
                    if row is not None:
                        rows.append(row)
                        cols.append(col)
                        vals.append(val)
        else:
            columns = []
            bras = []
            values = []
            for ket, ket_state in zip(self.local_basis, self.build_local_operator_list(op, 0)):
                col = _index_dict[ket]
                for bra, val in ket_state.items():
                    columns.append(col)
                    bras.append(bra)
                    values.append(val)

            global_rows = list(self.state_container._index_sequence(bras))
            _size = self.size
            for row, col, val in zip(global_rows, columns, values):
                if row != _size:
                    rows.append(row)
                    cols.append(col)
                    vals.append(val)

        n = len(self)
        if rows:
            res = sp.sparse.csc_array((vals, (rows, cols)), shape=(n, n), dtype=complex)
        else:
            res = sp.sparse.csc_array((n, n), dtype=complex)
        return res

    @property
    def impurity_spin_orbital_indices(self):
        """Flat, sorted-by-orbital-set list of all impurity spin-orbital indices."""
        return [orb for blocks in self.impurity_orbitals.values() for block in blocks for orb in block]

    @property
    def valence_spin_orbital_indices(self):
        """Flat list of all valence-bath spin-orbital indices."""
        valence_baths, _conduction_baths = self.bath_states
        return [orb for blocks in valence_baths.values() for block in blocks for orb in block]

    @property
    def conduction_spin_orbital_indices(self):
        """Flat list of all conduction-bath spin-orbital indices."""
        _valence_baths, conduction_baths = self.bath_states
        return [orb for blocks in conduction_baths.values() for block in blocks for orb in block]

    def get_state_statistics(self, psis):
        """
        Calucluate some occupation statistics for the ManyBopdyState psi.
        Parameters:
        ===========
        Returns:
        ========
        stats: dict - Occupation statistics
        """
        impurity_indices = self.impurity_spin_orbital_indices
        valence_indices = self.valence_spin_orbital_indices
        conduction_indices = self.conduction_spin_orbital_indices
        psi_stats = [{} for _ in psis]
        for i, psi in enumerate(psis):
            for state, amp in psi.items():
                bits = psr.bytes2bitarray(bytes(state.to_bytearray()), self.num_spin_orbitals)
                n_imp = bits[impurity_indices].count()
                n_valence = bits[valence_indices].count()
                n_cond = bits[conduction_indices].count()
                psi_stats[i][(n_imp, n_valence, n_cond)] = abs(amp) ** 2 + psi_stats[i].get(
                    (n_imp, n_valence, n_cond), 0
                )
        if self.is_distributed:
            all_psi_stats = self.comm.gather(psi_stats)
            if self.comm.rank == 0:
                psi_stats = [{} for _ in psis]
                for local_psi_stats in all_psi_stats:
                    for i, psi_stat in enumerate(local_psi_stats):
                        for key in psi_stat:
                            psi_stats[i][key] = psi_stat[key] + psi_stats[i].get(key, 0)
            psi_stats = self.comm.bcast(psi_stats)
        return psi_stats

    def build_density_matrices(self, psis, orbital_indices_left=None, orbital_indices_right=None):
        r"""Compute single-particle density matrices for a list of many-body states.

        rho[n, i, j] = <psi_n| c_{orb_j}^dagger c_{orb_i} |psi_n>

        For the square case (orbital_indices_left == orbital_indices_right) the
        identity rho[i, j] = <phi_j | phi_i>  (where |phi_k> = c_{orb_k}|psi>)
        is exploited, cutting operator applications from O(n^2) to O(n) and
        halving inner products via Hermitian symmetry.

        For the general rectangular case we also exploit this decomposition
        rho[i, j] = <chi_j | phi_i> (where |chi_k> = c_{orb_k}|psi> and
        |phi_k> = c_{orb_k}|psi>), reducing operator applications to O(n).
        """
        if orbital_indices_left is None:
            orbital_indices_left = list(range(self.num_spin_orbitals))
        if orbital_indices_right is None:
            orbital_indices_right = list(range(self.num_spin_orbitals))
        n_left, n_right = len(orbital_indices_left), len(orbital_indices_right)
        rhos = np.zeros((len(psis), n_left, n_right), dtype=complex)

        square = orbital_indices_left == orbital_indices_right

        for n, psi_n in enumerate(psis):
            phi = [ManyBodyOperator({((orb, "a"),): 1.0})(psi_n, 0) for orb in orbital_indices_left]
            if square:
                chi = phi
            else:
                chi = [ManyBodyOperator({((orb, "a"),): 1.0})(psi_n, 0) for orb in orbital_indices_right]

            if self.is_distributed:
                phi = self.redistribute_psis(phi)
                if square:
                    chi = phi
                else:
                    chi = self.redistribute_psis(chi)

            from impurityModel.ed.ManyBodyUtils import inner_multi

            rhos[n] = inner_multi(chi, phi).T

        if self.is_distributed:
            self.comm.Allreduce(MPI.IN_PLACE, rhos, op=MPI.SUM)

        return rhos

    def determine_blocks(self, op, slaterWeightMin=0):
        """
        Determine the blockstructure of op in the ManyBodyBasis
        Return a list of lists containing the (MPI-)local basis states belonging to each block
        NB. The lists of local basis states for each block may be empty, but the block is not!
        Arguments:
        ==========
        op: ManyBodyOperator to determine the block structure of
        slaterWeightMin: float ignore matrix elements with magnitude < slaterWeightMin (|{op}_ij| < slaterWeightMin)
        Returns:
        ========
        list[list[slater determinants]] the (MPI-)local manybody basis states belonging to each block.
        """
        disjoint_sets = DisjointSet(list(range(self.size)))
        tmps = [None for _ in range(len(self.local_basis))]
        states = set()
        for i, state in enumerate(self.local_basis):
            tmps[i] = set(applyOp_test(op, ManyBodyState({state: 1.0}), cutoff=slaterWeightMin).keys())
            states |= tmps[i]

        indices = self.state_container._index_sequence(states)
        _size = self.size
        idx_map = {state: idx for state, idx in zip(states, indices) if idx != _size}
        for root, connected_states in enumerate(tmps):
            for state in connected_states:
                if state not in idx_map:
                    continue
                disjoint_sets.merge(root + self.offset, idx_map[state])
        if self.is_distributed:
            disjoint_sets = self.comm.allreduce(disjoint_sets, op=reduce_disjoint_set_op)

        return [subset.intersection(self.local_indices) for subset in disjoint_sets.subsets()]

    def split_basis_and_redistribute_psi(
        self, priorities: list[float], psis: Optional[list[ManyBodyState]]
    ) -> tuple[
        list[int], list[int], int, list[int], "Basis", Optional[list[ManyBodyState]], list[Optional[MPI.Intercomm]]
    ]:
        """Split the basis and redistribute wavefunctions over a split communicator.

        Parameters
        ----------
        priorities : list of float
            The split priority weights for each block.
        psis : list of ManyBodyState, optional
            The wavefunctions to redistribute, or None.

        Returns
        -------
        indices : list of int
            Representative indices.
        split_roots : list of int
            The roots for the split communicators.
        color : int
            The split communicator color rank.
        items_per_color : list of int
            Number of items assigned to each color group.
        split_basis : Basis
            The new Basis associated with the split communicator.
        psis : list of ManyBodyState, optional
            Redistributed wavefunctions.
        intercomms : list of MPI.Intercomm
            MPI intercommunicators (all set to None after being freed).
        """

        if (not self.is_distributed) or len(priorities) <= 1:
            return range(len(priorities)), [0], 0, [len(priorities)], self, psis, [None]

        comm = self.comm
        normalized_priorities = np.array([abs(p) for p in priorities], dtype=float)
        normalized_priorities /= np.sum(np.abs(normalized_priorities))
        # normalized_priorities[::-1].sort()
        sorted_idxs = np.argsort(normalized_priorities, kind="stable")[::-1]
        n_colors = min(comm.size, len(normalized_priorities))

        # Adaptive split policy (Phase 7). The participation ratio
        # (Σp)²/Σp² = 1/Σ(normalized_p²) is the effective number of equally-weighted
        # blocks; capping n_colors near it (scaled by split_threshold) avoids starving a
        # few dominant blocks of ranks — better to run them on a larger sub-communicator
        # (or unified). split_threshold=1 is the legacy max-split for equal blocks;
        # split_threshold=0 forces a single unified communicator.
        participation = 1.0 / np.sum(normalized_priorities**2)
        n_colors = min(n_colors, max(1, int(np.ceil(participation * self.split_threshold))))
        if n_colors <= 1:
            # Unified: all ranks process every block together (no actual split).
            return range(len(priorities)), [0], 0, [len(priorities)], self, psis, [None]

        subgroups = [tuple() for _ in range(n_colors)]
        for i in range(0, len(normalized_priorities), n_colors):
            for j in range(min(n_colors, len(normalized_priorities) - i)):
                subgroups[j] += (sorted_idxs[i + j],)
        merged_priorities = np.array([np.sum(normalized_priorities[list(subgroup)]) for subgroup in subgroups])
        procs_per_color = np.array([max(1, n) for n in np.floor(comm.size * merged_priorities)], dtype=int)
        remainder = comm.size - np.sum(procs_per_color)
        while remainder != 0:
            if remainder < 0:
                mask = np.nonzero(procs_per_color > 1)

                procs_per_color[mask][-abs(remainder) % n_colors :] -= 1
            else:
                procs_per_color[-remainder % n_colors :] += 1
            remainder = comm.size - np.sum(procs_per_color)

        assert sum(procs_per_color) == comm.size
        proc_cutoffs = np.cumsum(procs_per_color)
        color = np.argmax(comm.rank < proc_cutoffs)

        split_comm = comm.Split(color=color, key=comm.rank)
        split_roots = [0] + proc_cutoffs[:-1].tolist()
        items_per_color = [len(subgroup) for subgroup in subgroups]
        assert sum(items_per_color) == len(priorities)

        intercomms = []
        for c, c_root in enumerate(split_roots):
            if c == color:
                intercomms.append(None)
                continue
            intercomms.append(split_comm.Create_intercomm(0, comm, c_root))
        indices = sorted(subgroups[color])

        if split_comm.rank == 0:
            assert comm.rank in split_roots

        new_states = set(self.local_basis)
        # Distribute my local basis states among all other colors
        for c, c_root in enumerate(split_roots):
            # I will send  states to this color
            if color != c:
                serialized_local_basis = bytearray().join(
                    state.to_bytearray()[: self.n_bytes] for state in self.local_basis
                )
                intercomms[c].send(serialized_local_basis, dest=split_comm.rank % procs_per_color[c])
            # I will receive states from all other colors
            else:
                for send_color in range(len(split_roots)):
                    if send_color == color:
                        continue
                    for sender in range(procs_per_color[send_color]):
                        if sender % procs_per_color[c] != split_comm.rank:
                            continue
                        received_bytes = intercomms[send_color].recv(source=sender)
                        new_states.update(
                            self.type.from_bytes(bytes(received_bytes[i : i + self.n_bytes]))
                            for i in range(0, len(received_bytes), self.n_bytes)
                        )

        split_basis = Basis(
            self.impurity_orbitals,
            self.bath_states,
            initial_basis=list(new_states),
            restrictions=self.restrictions,
            weighted_restrictions=self.weighted_restrictions,
            split_threshold=self.split_threshold,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            comm=split_comm,
            verbose=self.verbose,
            truncation_threshold=self.truncation_threshold,
            tau=self.tau,
            spin_flip_dj=self.spin_flip_dj,
        )

        if psis is not None:
            new_psis = [p.copy() for p in psis]
            for c, c_root in enumerate(split_roots):
                if color != c:
                    serialized_psis = [{bytes(k.to_bytearray()[: self.n_bytes]): v for k, v in p.items()} for p in psis]
                    intercomms[c].send(serialized_psis, dest=split_comm.rank % procs_per_color[c])
                else:
                    for send_color in range(len(split_roots)):
                        if send_color == color:
                            continue
                        for sender in range(procs_per_color[send_color]):
                            if sender % procs_per_color[color] != split_comm.rank:
                                continue
                            received_psis = intercomms[send_color].recv(source=sender)
                            for i, received_psi in enumerate(received_psis):
                                new_psis[i] += ManyBodyState(
                                    {self.type.from_bytes(k): v for k, v in received_psi.items()}
                                )
            psis = split_basis.redistribute_psis(new_psis)

        # Free the intercommunicators collectively while all ranks are still
        # synchronised here.  MPI_Comm_free is collective — leaving the objects
        # for Python gc means they may be freed at different times on different
        # ranks, causing crashes.  The split_comm itself (split_basis.comm) must
        # NOT be freed here because the caller still needs split_basis.
        for ic in intercomms:
            if ic is not None and ic != MPI.COMM_NULL:
                ic.Free()

        return indices, split_roots, color, items_per_color, split_basis, psis, [None] * len(intercomms)
