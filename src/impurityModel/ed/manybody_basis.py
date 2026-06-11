from math import ceil
import sys
from typing import Optional, Union
from os import environ
from bisect import bisect_left

try:
    from collections.abc import Iterable
except ModuleNotFoundError:
    from collections import Iterable
import itertools
import numpy as np
import scipy as sp
from heapq import merge
from scipy.cluster.hierarchy import DisjointSet
from mpi4py import MPI
from bitarray import bitarray
from impurityModel.ed.manybody_state_containers import (
    SimpleDistributedStateContainer,
)
from impurityModel.ed.mpi_comm import graph_alltoall, graph_alltoall_psis



from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.finite import (
    c2i,
    c2i_op,
    eigensystem_new,
    norm2,
    build_density_matrix,
    thermal_average_scale_indep,
)

from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    ManyBodyOperator,
    SlaterDeterminant,
    applyOp as applyOp_test,
    inner,
)


from impurityModel.ed.finite import applyOp_new as applyOp
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
    def _get_offsets_and_local_lengths(self, total_length: int) -> tuple[int, int]:
        """Compute the MPI rank offsets and local lengths for distributing a total size.

        Parameters
        ----------
        total_length : int
            The total number of states/elements to distribute.

        Returns
        -------
        offset : int
            The global index offset for the local rank.
        local_len : int
            The number of states/elements assigned to the local rank.
        """
        offset = 0
        local_len = total_length
        if self.comm is not None:
            local_len = total_length // self.comm.size
            leftovers = total_length % self.comm.size
            if leftovers != 0 and self.comm.rank < leftovers:
                local_len += 1
            scanned_length = np.empty((1,), dtype=int)
            offset = self.comm.Scan(np.array([local_len], dtype=int), scanned_length, op=MPI.SUM)
        return offset, scanned_length[0] - local_len

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
        total_configurations = {}
        for i in valence_baths:
            valid_configurations = []
            impurity_electron_indices = [orb for imp_orbs in impurity_orbitals[i] for orb in imp_orbs]
            valence_electron_indices = [orb for val_orbs in valence_baths[i] for orb in val_orbs]
            conduction_electron_indices = [orb for con_orbs in conduction_baths[i] for orb in con_orbs]
            for nominal_occ in range(
                max(0, nominal_impurity_occ[i] - abs(mixed_valence[i])),
                min(total_impurity_orbitals[i], nominal_impurity_occ[i] + abs(mixed_valence[i])) + 1,
            ):
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
                            impurity_configurations = itertools.combinations(
                                impurity_electron_indices, impurity_occupation
                            )
                            valence_configurations = itertools.combinations(
                                valence_electron_indices, valence_occupation
                            )
                            conduction_configurations = itertools.combinations(
                                conduction_electron_indices, conduction_occupation
                            )
                            if verbose:
                                print(f"Partition {i} occupations")
                                print(f"Impurity occupation:   {impurity_occupation:d}")
                                print(f"Valence occupation:   {valence_occupation:d}")
                                print(f"Conduction occupation: {conduction_occupation:d}")
                            valid_configurations.append(
                                itertools.product(
                                    impurity_configurations,
                                    valence_configurations,
                                    conduction_configurations,
                                )
                            )
            total_configurations[i] = valid_configurations
        num_spin_orbitals = sum(total_impurity_orbitals[i] + total_baths[i] for i in total_baths)
        basis = []
        # Combine all valid configurations for all l-subconfigurations (ex. p-states and d-states)
        for config in itertools.product(*total_configurations.values()):
            for set_bits in itertools.product(*config):
                basis.append(
                    psr.tuple2bytes(
                        tuple(idx for subset in set_bits for part in subset for idx in part), 8 * self.n_bytes
                    ),
                )

        return [SlaterDeterminant.from_bytes(bytestring) for bytestring in basis], num_spin_orbitals

    def _get_restrictions(
        self,
        impurity_orbitals: dict[int, list[list[int]]],
        bath_states: tuple[dict[int, list[list[int]]], dict[int, list[list[int]]]],
        delta_valence_occ: dict[int, int],
        delta_conduction_occ: dict[int, int],
        delta_impurity_occ: dict[int, int],
        nominal_impurity_occ: dict[int, int],
        verbose: bool,
    ) -> dict[frozenset[int], tuple[int, int]]:
        """Determine the occupation restrictions for each orbital set.

        Parameters
        ----------
        impurity_orbitals : dict
            Impurity orbitals grouped by l quantum number.
        bath_states : tuple of dict
            Valence and conduction bath states grouped by l quantum number.
        delta_valence_occ : dict
            Allowed valence occupation variation.
        delta_conduction_occ : dict
            Allowed conduction occupation variation.
        delta_impurity_occ : dict
            Allowed impurity occupation variation.
        nominal_impurity_occ : dict
            Nominal impurity occupation.
        verbose : bool
            Whether to print restriction details.

        Returns
        -------
        restrictions : dict of frozenset of int to (int, int)
            A dictionary mapping sets of orbital indices to their (min, max) occupations.
        """
        valence_baths, conduction_baths = bath_states
        restrictions = {}
        total_baths = {
            i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
            for i in valence_baths
        }
        total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
        for i in total_baths:
            impurity_indices = frozenset(orb for imp_orbs in impurity_orbitals[i] for orb in imp_orbs)
            restrictions[impurity_indices] = (
                max(nominal_impurity_occ[i] - delta_impurity_occ[i], 0),
                min(nominal_impurity_occ[i] + delta_impurity_occ[i] + 1, total_impurity_orbitals[i] + 1),
            )
            valence_indices = frozenset(orb for val_orbs in valence_baths[i] for orb in val_orbs)
            restrictions[valence_indices] = (
                max(sum(len(orbs) for orbs in valence_baths[i]) - delta_valence_occ[i], 0),
                sum(len(orbs) for orbs in valence_baths[i]) + 1,
            )
            conduction_indices = frozenset(orb for con_orbs in conduction_baths[i] for orb in con_orbs)
            restrictions[conduction_indices] = (0, delta_conduction_occ[i] + 1)

            if verbose:
                print(f"l = {i}")
                print(f"|---Restrictions on the impurity orbitals = {restrictions[impurity_indices]}")
                print(f"|---Restrictions on the valence bath      = {restrictions[valence_indices]}")
                print(f"----Restrictions on the conduction bath   = {restrictions[conduction_indices]}")

        return restrictions

    def get_effective_restrictions(self) -> dict[frozenset[int], tuple[int, int]]:
        """Calculate the actual min/max occupations observed across the current basis.

        Returns
        -------
        restrictions : dict of frozenset of int to (int, int)
            Dictionary mapping orbital subsets to their observed (min, max) occupations.
        """
        valence_baths, conduction_baths = self.bath_states

        total_baths = {
            i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
            for i in valence_baths
        }
        total_impurity_orbitals = {
            i: sum(len(orbs) for orbs in self.impurity_orbitals[i]) for i in self.impurity_orbitals
        }
        restrictions = {}

        for i in sorted(total_baths.keys()):
            max_imp = 0
            min_imp = total_impurity_orbitals[i]
            max_val = 0
            min_val = sum(len(orbs) for orbs in valence_baths[i])
            max_con = 0
            min_con = sum(len(orbs) for orbs in conduction_baths[i])
            impurity_indices = frozenset(sorted(ind for imp_ind in self.impurity_orbitals[i] for ind in imp_ind))
            valence_indices = frozenset(sorted(ind for val_ind in valence_baths[i] for ind in val_ind))
            conduction_indices = frozenset(sorted(ind for con_ind in conduction_baths[i] for ind in con_ind))
            for state in self.local_basis:
                bits = psr.bytes2bitarray(bytes(state.to_bytearray()), self.num_spin_orbitals)
                n_imp = sum(bits[i] for i in impurity_indices)
                n_val = sum(bits[i] for i in valence_indices)
                n_con = sum(bits[i] for i in conduction_indices)
                max_imp = max(max_imp, n_imp)
                min_imp = min(min_imp, n_imp)
                max_val = max(max_val, n_val)
                min_val = min(min_val, n_val)
                max_con = max(max_con, n_con)
                min_con = min(min_con, n_con)
            max_val = len(valence_indices)
            min_con = 0
            if self.is_distributed:
                max_imp = self.comm.allreduce(max_imp, op=MPI.MAX)
                min_imp = self.comm.allreduce(min_imp, op=MPI.MIN)
                min_val = self.comm.allreduce(min_val, op=MPI.MIN)
                max_con = self.comm.allreduce(max_con, op=MPI.MAX)
            if len(impurity_indices) > 0:
                restrictions[impurity_indices] = (min_imp, max_imp)
            if len(valence_indices) > 0:
                restrictions[valence_indices] = (min_val, max_val)
            if len(conduction_indices) > 0:
                restrictions[conduction_indices] = (min_con, max_con)
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
        if dN is None:
            return 0, len(orbs)
        elif orbs not in restrictions:
            return 0, len(orbs)
        occ_dec, occ_inc = dN
        min_occ, max_occ = restrictions[orbs]
        return max(min_occ - occ_dec, 0), min(max_occ + occ_inc, len(orbs))

    def build_initial_restrictions(self, op: ManyBodyOperator, min_dist: int = 4) -> Optional[dict[frozenset[int], tuple[int, int]]]:
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
        graph = np.zeros((tot_orb, tot_orb), dtype=bool)

        for i, j in itertools.product(range(tot_orb), repeat=2):
            if ((i, "c"), (j, "a")) in op:
                graph[i, j] = abs(op[((i, "c"), (j, "a"))]) > 1e-8
        dist_matrix = sp.sparse.csgraph.shortest_path(
            graph, directed=False, unweighted=True, indices=all_impurity_orbitals
        )
        for i, impurity_orbitals in self.impurity_orbitals.items():
            for imp_orb_block, val_orb_block, con_orb_block in zip(
                impurity_orbitals, valence_baths[i], conduction_baths[i]
            ):
                # Identify filled and empty bath states
                # Ignore states that are too close to the impurity
                filled_valence_states = [
                    orb for orb in val_orb_block if np.min(dist_matrix[np.ix_(imp_orb_block, [orb])]) > min_dist
                ]
                filled_states = frozenset(sorted(filled_valence_states))
                empty_conduction_states = [
                    orb for orb in con_orb_block if np.min(dist_matrix[np.ix_(imp_orb_block, [orb])]) > min_dist
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
            print(f"Ground state restrictions:")
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
            imp_change = {i: None for i in self.impurity_orbitals}
        if val_change is None:
            val_change = {i: None for i in self.impurity_orbitals}
        if con_change is None:
            con_change = {i: None for i in self.impurity_orbitals}

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
        graph = np.zeros((tot_orb, tot_orb), dtype=bool)

        for i, j in itertools.product(range(tot_orb), repeat=2):
            if ((i, "c"), (j, "a")) in op:
                graph[i, j] = abs(op[((i, "c"), (j, "a"))]) > 1e-8

        dist_matrix = sp.sparse.csgraph.shortest_path(
            graph, directed=False, unweighted=True, indices=all_impurity_orbitals
        )
        if False and self.verbose:
            matrix_print(dist_matrix[:, :], "Orbital distance matrix", flush=True)

        for i, impurity_orbitals in self.impurity_orbitals.items():
            imp_orbs = frozenset(sorted(orb for block in impurity_orbitals for orb in block))
            min_imp, max_imp = self._get_updated_occ_restrictions(ground_state_restrictions, imp_orbs, imp_change[i])
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
                        if np.min(dist_matrix[np.ix_(imp_orb_block, [val_orb_block[orb]])]) > min_dist
                    ]
                    filled_conduction_states = [
                        con_orb_block[orb]
                        for orb in np.nonzero(conduction_occupations > 1 - cutoff)[0]
                        if np.min(dist_matrix[np.ix_(imp_orb_block, con_orb_block[orb])]) > min_dist
                    ]
                    filled_states = frozenset(sorted(filled_valence_states + filled_conduction_states))
                    empty_valence_states = [
                        val_orb_block[orb]
                        for orb in np.nonzero(valence_occupations < cutoff)[0]
                        if np.min(dist_matrix[np.ix_(imp_orb_block, val_orb_block[orb])]) > min_dist
                    ]
                    empty_conduction_states = [
                        con_orb_block[orb]
                        for orb in np.nonzero(conduction_occupations < cutoff)[0]
                        if np.min(dist_matrix[np.ix_(imp_orb_block, con_orb_block[orb])]) > min_dist
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
            if len(imp_orbs) > 0 and (min_imp > 0 or max_imp < len(imp_orbs)):
                excited_restrictions[frozenset(sorted(imp_orbs))] = (min_imp, max_imp)
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
                self.type.from_bytes(state) if isinstance(state, bytes) else state
                for state in initial_basis
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
                mixed_valence=mixed_valence if mixed_valence is not None else {i: 0 for i in nominal_impurity_occ},
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

        # self.state_container = CentralizedStateContainer(
        self.state_container = SimpleDistributedStateContainer(
            # self.state_container = DistributedStateContainer(
            initial_basis,
            bytes_per_state=self.n_bytes,
            comm=self.comm,
            verbose=verbose,
        )
        self.offset = self.state_container.offset
        self.size = self.state_container.size
        self.local_indices = self.state_container.local_indices
        self._index_dict = self.state_container._index_dict
        self.index_bounds = self.state_container.index_bounds
        self.state_bounds = self.state_container.state_bounds
        self.local_basis = self.state_container.local_basis

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

        self.offset = self.state_container.offset
        self.size = self.state_container.size
        self.local_indices = self.state_container.local_indices
        self._index_dict = self.state_container._index_dict
        self.index_bounds = self.state_container.index_bounds
        self.state_bounds = self.state_container.state_bounds
        self.local_basis = self.state_container.local_basis

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
                f"Expetced a list of ManyBodyStates, received a single ManyBodyState. Remaking into list of one ManyBodyState"
            )
            psis = [psis]
        psis = [
            psi if isinstance(psi, ManyBodyState) else ManyBodyState({
                (SlaterDeterminant.from_bytes(k) if isinstance(k, bytes) else k): v
                for k, v in psi.items()
            })
            for psi in psis
        ]
        if not self.is_distributed:
            return psis

        comm = self.comm

        def find_owner(state: SlaterDeterminant) -> int:
            """Determine the rank owning the state using a hash function.

            Parameters
            ----------
            state : SlaterDeterminant
                The state to locate.

            Returns
            -------
            int
                The MPI rank index.
            """
            return hash(state) % comm.size

        # Build a send list: for each target rank r, a list of dicts
        # (one dict per psi), mapping state_bytes -> amplitude.
        send_list = [[{} for _ in psis] for _ in range(comm.size)]
        unique_states = set()
        for psi in psis:
            unique_states.update(psi.keys())
        for state in unique_states:
            send_to = find_owner(state)
            state_bytes = bytes(state.to_bytearray()[:self.n_bytes])
            for s_psi, l_psi in zip(send_list[send_to], psis):
                if state not in l_psi:
                    continue
                s_psi[state_bytes] = l_psi[state]

        # Use the specialised zero-pickle path: raw bytes + complex128
        # arrays exchanged via Neighbor_alltoallv on a sparse graph comm.
        received_list = graph_alltoall_psis(send_list, self.n_bytes, comm)
        res = [ManyBodyState({}) for _ in psis]
        for received_psis in received_list:
            for res_n, psi_dict in zip(res, received_psis):
                if psi_dict:
                    res_n += ManyBodyState({
                        self.type.from_bytes(k): v
                        for k, v in psi_dict.items()
                    })
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
        spin_flip = set()
        for det in determinants:
            n_dn = int(applyOp(self.num_spin_orbitals, n_dn_op, {det: 1}).get(det, 0))
            n_up = int(applyOp(self.num_spin_orbitals, n_up_op, {det: 1}).get(det, 0))
            spin_flip.add(det)
            to_flip = {det}
            for l in self.impurity_orbitals:
                n_orb = sum(len(orbs) for orbs in self.impurity_orbitals[l])
                for i in range(n_orb // 2):
                    spin_flip_op = {
                        ((i + n_orb // 2, "c"), (i, "a")): 1.0,
                        ((i, "c"), (i + n_orb // 2, "a")): 1.0,
                    }
                    for state in list(to_flip):
                        flipped = applyOp(self.num_spin_orbitals, spin_flip_op, {state: 1})
                        to_flip.update(flipped.keys())
                        if len(flipped) == 0:
                            continue
                        flipped_state = list(flipped.keys())[0]
                        new_n_dn = int(applyOp(self.num_spin_orbitals, n_dn_op, {flipped_state: 1}).get(flipped_state, 0))
                        new_n_up = int(applyOp(self.num_spin_orbitals, n_up_op, {flipped_state: 1}).get(flipped_state, 0))
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
        # op.set_restrictions(self.restrictions)
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

    def copy(self) -> 'Basis':
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
        for row, psi in enumerate(psis):
            for state, val in psi.items():
                idx = self._index_dict.get(state)
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
        if not self.is_distributed:
            for ket, ket_state in zip(self.local_basis, self.build_local_operator_list(op, 0)):
                col = self._index_dict[ket]
                for bra, val in ket_state.items():
                    row = self._index_dict.get(bra)
                    if row is not None:
                        rows.append(row)
                        cols.append(col)
                        vals.append(val)
        else:
            columns = []
            bras = []
            values = []
            for ket, ket_state in zip(self.local_basis, self.build_local_operator_list(op, 0)):
                col = self._index_dict[ket]
                for bra, val in ket_state.items():
                    columns.append(col)
                    bras.append(bra)
                    values.append(val)

            global_rows = list(self.state_container._index_sequence(bras))
            for row, col, val in zip(global_rows, columns, values):
                if row != self.size:
                    rows.append(row)
                    cols.append(col)
                    vals.append(val)

        n = len(self)
        if rows:
            res = sp.sparse.csc_array(
                (vals, (rows, cols)), shape=(n, n), dtype=complex
            )
        else:
            res = sp.sparse.csc_array((n, n), dtype=complex)
        return res

    def get_state_statistics(self, psis):
        """
        Calucluate some occupation statistics for the ManyBopdyState psi.
        Parameters:
        ===========
        Returns:
        ========
        stats: dict - Occupation statistics
        """
        impurity_indices = self.impurity_orbitals
        (valence_indices, conduction_indices) = self.bath_states
        impurity_indices = [orb for blocks in impurity_indices.values() for block in blocks for orb in block]
        valence_indices = [orb for i, blocks in valence_indices.items() for block in blocks for orb in block]
        conduction_indices = [orb for blocks in conduction_indices.values() for block in blocks for orb in block]
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
            phi = [
                ManyBodyOperator({((orb, "a"),): 1.0})(psi_n, 0)
                for orb in orbital_indices_left
            ]
            if square:
                chi = phi
            else:
                chi = [
                    ManyBodyOperator({((orb, "a"),): 1.0})(psi_n, 0)
                    for orb in orbital_indices_right
                ]

            if self.is_distributed:
                phi = self.redistribute_psis(phi)
                if square:
                    chi = phi
                else:
                    chi = self.redistribute_psis(chi)

            if square:
                for i in range(n_left):
                    amp = inner(chi[i], phi[i])
                    if abs(amp) > 0:
                        rhos[n, i, i] = amp
                    for j in range(i + 1, n_left):
                        amp = inner(chi[j], phi[i])
                        if abs(amp) > 0:
                            rhos[n, i, j] = amp
                            rhos[n, j, i] = amp.conjugate()
            else:
                for i in range(n_left):
                    for j in range(n_right):
                        amp = inner(chi[j], phi[i])
                        if abs(amp) > 0:
                            rhos[n, i, j] = amp

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
        idx_map = {state: idx for state, idx in zip(states, indices) if idx != self.size}
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
    ) -> tuple[list[int], list[int], int, list[int], 'Basis', Optional[list[ManyBodyState]], list[Optional[MPI.Intercomm]]]:
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
        rank = comm.rank
        normalized_priorities = np.array([abs(p) for p in priorities], dtype=float)
        normalized_priorities /= np.sum(np.abs(normalized_priorities))
        # normalized_priorities[::-1].sort()
        sorted_idxs = np.argsort(normalized_priorities, kind="stable")[::-1]
        n_colors = min(comm.size, len(normalized_priorities))

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
                serialized_local_basis = bytearray().join(state.to_bytearray()[:self.n_bytes] for state in self.local_basis)
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
                    serialized_psis = [
                        {bytes(k.to_bytearray()[:self.n_bytes]): v for k, v in p.items()}
                        for p in psis
                    ]
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
                                new_psis[i] += ManyBodyState({
                                    self.type.from_bytes(k): v
                                    for k, v in received_psi.items()
                                })
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

    def split_into_block_basis_and_redistribute_psi(
        self, op, psis, min_group_size=1000, slaterWeightMin=0, verbose=False
    ):
        """
        Split the basis into blocks determined by op. Also split psis into these blocks.
        """

        if not self.is_distributed:
            return [0], [0], 0, [1], self, psis, [None]

        comm = self.comm
        rank = comm.rank

        blocks = self.determine_blocks(op, slaterWeightMin)
        block_lengths = np.array([len(block) for block in blocks], dtype=int)
        if self.is_distributed:
            self.comm.Allreduce(MPI.IN_PLACE, block_lengths)
        if verbose:
            print("Found block sizes:")
            print(block_lengths)

        tmp_basis = Basis(
            self.impurity_orbitals,
            self.bath_states,
            initial_basis=[],
            restrictions=self.restrictions,
            spin_flip_dj=self.spin_flip_dj,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            verbose=self.verbose,
        )

        block_indices, block_roots, block_color, blocks_per_color, block_basis, _, _ = (
            tmp_basis.split_basis_and_redistribute_psi(block_lengths**2, None)
        )
        block_roots = np.array(block_roots, dtype=int)
        procs_per_color = block_roots[1:] - block_roots[:-1]
        procs_per_color = np.append(procs_per_color, [comm.size - np.sum(procs_per_color)])

        # Recreate the intercommunicators between color groups.
        # split_basis_and_redistribute_psi frees them before returning (to avoid
        # non-collective gc freeing).  We need them for the communication below,
        # so we rebuild them here using the same split_comm (block_basis.comm).
        split_comm = block_basis.comm
        block_intercomms = []
        for c, c_root in enumerate(block_roots.tolist()):
            if c == block_color:
                block_intercomms.append(None)
            else:
                block_intercomms.append(split_comm.Create_intercomm(0, comm, int(c_root)))

        num_block_indices_per_color = np.empty((len(block_roots)), dtype=int)
        for c, c_root in enumerate(block_roots):
            if c == block_color and rank == 0:
                for send_color in range(len(block_roots)):
                    if send_color == c:
                        num_block_indices_per_color[c] = len(block_indices)
                        continue
                    block_intercomms[send_color].Recv(
                        num_block_indices_per_color[send_color : send_color + 1], source=0
                    )
            elif rank == c_root:
                block_intercomms[0].Send(np.array([len(block_indices)], dtype=int), dest=0)
        comm.Bcast(num_block_indices_per_color, root=0)

        block_index_color_offsets = [np.sum(num_block_indices_per_color[:c]) for c in range(len(block_roots))]
        block_indices_per_color = np.empty((np.sum(num_block_indices_per_color)), dtype=int)
        for c, c_root in enumerate(block_roots):
            if c == block_color and rank == 0:
                for send_color in range(len(block_roots)):
                    start = block_index_color_offsets[send_color]
                    stop = start + num_block_indices_per_color[send_color]
                    if send_color == c:
                        block_indices_per_color[start:stop] = block_indices
                        continue
                    block_intercomms[send_color].Recv(
                        block_indices_per_color[start:stop],
                        source=0,
                    )
            elif rank == c_root:
                block_intercomms[0].Send(np.array(block_indices, dtype=int), dest=0)
        comm.Bcast(block_indices_per_color, root=0)

        new_states = {
            self.local_basis[local_idx - self.offset] for block_idx in block_indices for local_idx in blocks[block_idx]
        }
        for c, c_root in enumerate(block_roots):
            # We will receive states from everyone else
            if c == block_color:
                for send_color in range(len(block_roots)):
                    if send_color == block_color:
                        continue
                    for sender in range(procs_per_color[send_color]):
                        if sender % procs_per_color[c] != block_basis.comm.rank:
                            continue
                        received_bytes = block_intercomms[send_color].recv(source=sender)
                        new_states.update(
                            self.type.from_bytes(bytes(received_bytes[i : i + self.n_bytes]))
                            for i in range(0, len(received_bytes), self.n_bytes)
                        )
            else:
                start = block_index_color_offsets[c]
                stop = start + num_block_indices_per_color[c]
                states_to_send = {
                    self.local_basis[local_idx - self.offset]
                    for block_idx in block_indices_per_color[start:stop]
                    for local_idx in blocks[block_idx]
                }
                serialized_states = bytearray().join(state.to_bytearray()[:self.n_bytes] for state in states_to_send)
                block_intercomms[c].send(
                    serialized_states,
                    dest=block_basis.comm.rank % procs_per_color[c],
                )
        block_basis.add_states(new_states)

        if psis is not None:
            states_in_blocks = {
                self.local_basis[state_idx - self.offset]
                for block_idx in block_indices
                for state_idx in blocks[block_idx]
            }
            new_psis = [
                ManyBodyState({state: amp for state, amp in psi.items() if state in states_in_blocks}) for psi in psis
            ]
            for c, c_roor in enumerate(block_roots):
                if c == block_color:
                    for send_color in range(len(block_roots)):
                        if send_color == block_color:
                            continue
                        for sender in range(procs_per_color[send_color]):
                            if sender % procs_per_color[c] != block_basis.comm.rank:
                                continue
                            received_psi_dict = block_intercomms[send_color].recv(source=sender)
                            for i, r_psi_dict in enumerate(received_psi_dict):
                                psi_state = ManyBodyState({
                                    self.type.from_bytes(k): v
                                    for k, v in r_psi_dict.items()
                                })
                                new_psis[i] += psi_state
                else:
                    start = block_index_color_offsets[c]
                    stop = start + num_block_indices_per_color[c]
                    send_states = {
                        self.local_basis[local_idx - self.offset]
                        for block_idx in block_indices_per_color[start:stop]
                        for local_idx in blocks[block_idx]
                    }
                    serialized_send_psis = [
                        {bytes(state.to_bytearray()[:self.n_bytes]): amp for state, amp in psi.items() if state in send_states}
                        for psi in psis
                    ]
                    block_intercomms[c].send(
                        serialized_send_psis,
                        dest=block_basis.comm.rank % procs_per_color[c],
                    )
            psis = block_basis.redistribute_psis(new_psis)

        # Free the intercommunicators and the split communicator collectively
        # before returning.  MPI_Comm_free is a collective operation — all ranks
        # in a communicator must call it at the same time.  Leaving these for
        # Python gc risks non-collective freeing (crash / protocol violation).
        for ic in block_intercomms:
            if ic is not None and ic != MPI.COMM_NULL:
                ic.Free()
        if block_basis is not None and block_basis.comm != comm:
            block_basis.free_comm()

        return block_indices, block_roots, block_color, blocks_per_color, block_basis, psis, [None] * len(block_intercomms)


class CIPSI_Basis(Basis):
    """Many-body basis implementing the CIPSI method.

    CIPSI (Configuration Interaction by Perturbation with Multi-Configurational
    Reference Selected by Perturbation) iteratively expands the basis by selecting
    important configuration determinants based on second-order perturbation theory.
    """
    def __init__(
        self,
        impurity_orbitals: dict[int, list[list[int]]],
        bath_states: tuple[dict[int, list[list[int]]], dict[int, list[list[int]]]],
        H: Optional[ManyBodyOperator] = None,
        nominal_impurity_occ: Optional[dict[int, int]] = None,
        initial_basis: Optional[list] = None,
        **kwargs,
    ):
        """Initialize the CIPSI basis.

        Parameters
        ----------
        impurity_orbitals : dict
            Impurity orbitals grouped by l quantum number.
        bath_states : tuple of dict
            Valence and conduction bath states grouped by l quantum number.
        H : ManyBodyOperator, optional
            The Hamiltonian operator.
        nominal_impurity_occ : dict, optional
            Nominal impurity occupation.
        initial_basis : list, optional
            Predefined initial states.
        **kwargs : dict
            Additional arguments passed to the parent Basis constructor.
        """
        if H is None:
            H = ManyBodyOperator({})
        if not isinstance(H, ManyBodyOperator):
            H = ManyBodyOperator(H)
        assert nominal_impurity_occ is not None or initial_basis is not None
        super().__init__(
            impurity_orbitals,
            bath_states,
            nominal_impurity_occ=nominal_impurity_occ,
            initial_basis=initial_basis,
            **kwargs,
        )

        if self.size > self.truncation_threshold and H is not None:
            if self.verbose:
                print("Truncating basis!")
            H_sparse = self.build_sparse_matrix(H)
            e_ref, psi_ref = eigensystem_new(
                H_sparse,
                e_max=-self.tau * np.log(1e-4),
                k=1,
                eigenValueTol=0,  # np.sqrt(np.finfo(float).eps),
                comm=self.comm,
                dense=False,
            )
            self.truncate(self.build_state(psi_ref))

    def truncate(self, psis: list[ManyBodyState]) -> list[ManyBodyState]:
        """Truncate the basis to fit within the truncation threshold.

        Parameters
        ----------
        psis : list of ManyBodyState
            The wavefunctions whose states are used to determine which basis elements to keep.

        Returns
        -------
        list of ManyBodyState
            The wavefunctions represented in the truncated basis.
        """
        cutoff = np.finfo(float).eps

        self.local_basis.clear()
        num_states = self.comm.allreduce(max(len(psi) for psi in psis))
        while num_states > self.truncation_threshold:
            psis = [{state: amp for state, amp in psi.items() if abs(amp) > cutoff} for psi in psis]
            num_states = self.comm.allreduce(max(len(psi) for psi in psis))
            cutoff *= 10
        self.add_states(state for psi in psis for state in psi)
        return self.redistribute_psis(psis)

    def _calc_de2(self, H, Hpsi_ref, e_ref: float, slaterWeightMin: float = 0):
        """
        Calculate the second-order variational energy contribution for each
        candidate Slater determinant Dj not yet in the basis.

        For each Dj and each reference state |Psi_i> with energy E_i this
        computes the perturbative correction

            de2[i, j] = |<Dj|H|Psi_i>|^2 / max(E_i - <Dj|H|Dj>, 1e-12)

        The diagonal expectation value <Dj|H|Dj> is evaluated by applying the
        *full* operator H to a uniform superposition of all candidate states and
        reading the diagonal amplitudes.  This is correct for arbitrary N-body
        operators (single-, double-, ... electron terms) because for any
        determinant |Dj> with amplitude 1 the result of H|Dj> contains the
        coefficient <Dj|H|Dj> at position Dj.
        """

        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        # Collect candidate determinants: states touched by H|Psi_i> that are
        # not already in the current basis.
        local_Djs = sorted(
            {state for hp in Hpsi_ref for state in hp if state not in self._index_dict}
        )

        if not local_Djs:
            return local_Djs, np.zeros((len(Hpsi_ref), 0), dtype=complex)

        # --- overlaps: <Dj|H|Psi_i> for all i, j ---
        # Build a (n_ref, n_Dj) matrix by reading amplitudes directly from
        # the already-computed H|Psi_i> dictionaries.
        Dj_index = {Dj: j for j, Dj in enumerate(local_Djs)}
        overlaps = np.zeros((len(Hpsi_ref), len(local_Djs)), dtype=complex)
        for i, Hpsi_i in enumerate(Hpsi_ref):
            for state, amp in Hpsi_i.items():
                j = Dj_index.get(state)
                if j is not None:
                    overlaps[i, j] = amp

        # --- diagonal elements: <Dj|H|Dj> for each candidate Dj ---
        # Apply H once to the uniform superposition |S> = sum_j |Dj>.  For a
        # general N-body operator the diagonal element <Dj|H|Dj> equals the
        # amplitude at Dj in H|Dj>, and since all |Dj> are orthogonal
        # determinants the contribution from |Dk> (k != j) to position Dj is
        # zero.  Reading H|S> at Dj therefore gives <Dj|H|Dj> correctly for
        # arbitrary operator rank.
        psi_all_Dj = ManyBodyState({Dj: 1.0 for Dj in local_Djs})
        H_psi_all = applyOp_test(H, psi_all_Dj, cutoff=slaterWeightMin)
        e_Dj = np.array(
            [np.real(H_psi_all.get(Dj, 0.0)) for Dj in local_Djs], dtype=float
        )

        # --- perturbative energy denominators ---
        # de[i, j] = E_i - <Dj|H|Dj>  (broadcast e_ref over j)
        de = e_ref[:, None] - e_Dj[None, :]
        de = np.maximum(de, 1e-12)

        # de2[i, j] = |<Dj|H|Psi_i>|^2 / (E_i - <Dj|H|Dj>)
        de2 = np.zeros_like(overlaps)
        mask = np.abs(overlaps) > 1e-12
        de2[mask] = np.square(np.abs(overlaps[mask])) / de[mask]
        return local_Djs, de2

    def determine_new_Dj(self, e_ref, psi_ref, H, de2_min, slater_cutoff=0, return_Hpsi_ref=False):
        """Apply H to each reference state, then select candidate determinants
        whose perturbative energy contribution exceeds *de2_min*."""
        Hpsi_ref = [
            applyOp_test(H, psi_i, cutoff=slater_cutoff)
            for psi_i in psi_ref
        ]
        Hpsi_ref = self.redistribute_psis(Hpsi_ref)
        local_Djs, de2 = self._calc_de2(H, Hpsi_ref, e_ref)
        de2_mask = np.any(np.abs(de2) >= de2_min, axis=0)
        new_Dj = set(itertools.compress(local_Djs, de2_mask))
        if return_Hpsi_ref:
            return new_Dj, Hpsi_ref
        return new_Dj

    def expand(self, H, de2_min=1e-10, dense_cutoff=1e3, slaterWeightMin=0):
        """
        Use the CIPSI method to expand the basis.

        Iteratively diagonalises H in the current basis, computes the
        second-order perturbative contribution of every Slater determinant
        outside the basis, and adds those whose contribution exceeds *de2_min*.
        Stops when no new determinants are added (convergence).

        Parameters
        ----------
        H : dict or ManyBodyOperator
            The many-body Hamiltonian.  Can contain single-, double-, or
            higher-body terms.
        de2_min : float
            Threshold for the perturbative energy contribution. Determinants
            with |de2| >= de2_min (for any reference state) are added.
        dense_cutoff : int
            Basis size below which dense diagonalisation is used.
        slaterWeightMin : float
            Amplitude cutoff passed to applyOp during H|Psi> evaluations.
        """
        de0_max = -self.tau * np.log(1e-4)
        psi_refs = None

        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        old_size = self.size - 1
        while old_size != self.size:

            H_mat = self.build_sparse_matrix(H)

            # Use previous eigenvectors as initial guess when doing sparse
            # diagonalisation (size >= dense_cutoff) to accelerate convergence.
            v0 = (
                self.build_vector(psi_refs).T
                if psi_refs is not None and self.size >= dense_cutoff
                else None
            )
            e_ref, psi_ref_dense = eigensystem_new(
                H_mat,
                e_max=de0_max,
                k=2 * len(psi_refs) if psi_refs is not None else 10,
                e0=None,
                v0=v0,
                eigenValueTol=0,
                comm=self.comm,
                dense=self.size < dense_cutoff,
            )

            psi_refs = self.build_state(psi_ref_dense.T)

            new_Dj = self.determine_new_Dj(
                e_ref, psi_refs, H, de2_min, slater_cutoff=slaterWeightMin
            )
            old_size = self.size
            self.add_states(new_Dj)
            psi_refs = self.redistribute_psis(psi_refs)
            if self.size > self.truncation_threshold:
                psi_refs = self.truncate(psi_refs)
                print("------> Basis truncated!")
                break

        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.", flush=True)

    def expand_at(self, E_ref: np.ndarray, psi_ref: list[ManyBodyState], H: ManyBodyOperator, de2_min: float = 1e-5) -> None:
        """Expand the basis at a specific reference energy and wavefunction.

        Parameters
        ----------
        E_ref : np.ndarray
            Reference energies.
        psi_ref : list of ManyBodyState
            Reference wavefunctions.
        H : ManyBodyOperator
            The Hamiltonian operator.
        de2_min : float, default 1e-5
            Second-order energy contribution threshold.
        """
        old_size = self.size - 1
        while old_size != self.size:
            new_Dj, psi_ref = self.determine_new_Dj(E_ref, psi_ref, H, de2_min, return_Hpsi_ref=True)

            old_size = self.size
            self.add_states(new_Dj)

            psi_ref = self.redistribute_psis(psi_ref)
            N2s = np.array([psi.norm2() for psi in psi_ref], dtype=float)
            if self.is_distributed:
                self.comm.Allreduce(MPI.IN_PLACE, N2s, op=MPI.SUM)
            psi_ref = [psi / np.sqrt(N2s[i]) for i, psi in enumerate(psi_ref)]

    def copy(self) -> 'CIPSI_Basis':
        """Create a copy of this CIPSI_Basis.

        Returns
        -------
        CIPSI_Basis
            A new CIPSI_Basis object with identical states and parameters.
        """
        new_basis = CIPSI_Basis(
            self.impurity_orbitals,
            self.bath_states,
            initial_basis=self.local_basis,
            restrictions=self.restrictions,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            spin_flip_dj=self.spin_flip_dj,
            tau=self.tau,
            verbose=self.verbose,
        )
        assert len(new_basis) == len(self)
        return new_basis

    def split_basis_and_redistribute_psi(
        self, priorities: list[float], psis: Optional[list[ManyBodyState]]
    ) -> tuple[list[int], list[int], int, list[int], 'CIPSI_Basis', Optional[list[ManyBodyState]], list[Optional[MPI.Intercomm]]]:
        """Split the CIPSI basis and redistribute wavefunctions.

        Parameters
        ----------
        priorities : list of float
            The priorities.
        psis : list of ManyBodyState, optional
            The wavefunctions.

        Returns
        -------
        tuple
            Split basis information. See Basis.split_basis_and_redistribute_psi.
        """
        indices, split_roots, color, items_per_color, split_basis, psis, intercomms = (
            super().split_basis_and_redistribute_psi(priorities, psis)
        )

        split_basis = CIPSI_Basis(
            split_basis.impurity_orbitals,
            split_basis.bath_states,
            initial_basis=split_basis.local_basis,
            restrictions=split_basis.restrictions,
            comm=split_basis.comm,
            truncation_threshold=split_basis.truncation_threshold,
            chain_restrict=split_basis.chain_restrict,
            collapse_chains=split_basis.collapse_chains,
            spin_flip_dj=split_basis.spin_flip_dj,
            tau=split_basis.tau,
            verbose=split_basis.verbose,
        )
        return indices, split_roots, color, items_per_color, split_basis, psis, intercomms
