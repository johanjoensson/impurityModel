from math import ceil
from time import perf_counter
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
from impurityModel.ed.manybody_state_containers import (
    DistributedStateContainer,
    CentralizedStateContainer,
    SimpleDistributedStateContainer,
)


from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.finite import (
    c2i,
    c2i_op,
    eigensystem_new,
    norm2,
    build_density_matrix,
    thermal_average_scale_indep,
)

from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, applyOp as applyOp_test, inner


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


def reduce_states(a: list[dict], b: list[dict], _):
    res = a.copy()
    for sa, sb in zip(res, b):
        for state, amp in sb.items():
            sa[state] = amp + sa.get(state, 0)
    return res


reduce_states_op = MPI.Op.Create(reduce_states, commute=True)


def reduce_disjoint_set(a: DisjointSet, b: DisjointSet, _):
    for subset in b.subsets():
        it = iter(subset)
        root = next(it)
        for item in it:
            a.merge(item, root)
    return a


reduce_disjoint_set_op = MPI.Op.Create(reduce_disjoint_set, commute=True)


def combine_sets(set_1, set_2, _):
    return set_1 | set_2


combine_sets_op = MPI.Op.Create(combine_sets, commute=True)


def reduce_subscript(a, b, datatype):
    res = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] is None:
                res[i][j] = b[i][j]
            else:
                res[i][j] = a[i][j]
    return res


reduce_subscript_op = MPI.Op.Create(reduce_subscript, commute=True)


def getitem_reduce(a, b, datatype):
    return [max(val_a, val_b) for val_a, val_b in zip(a, b)]


getitem_reduce_op = MPI.Op.Create(getitem_reduce, commute=True)


def getitem_reduce_matrix(a, b, datatype):
    res = [[None for _ in row] for row in a]
    for i in range(len(a)):
        for j in range(len(a[i])):
            res[i][j] = max(a[i][j], b[i][j])
    return res


getitem_reduce_matrix_op = MPI.Op.Create(getitem_reduce_matrix, commute=True)


class Basis:
    def _get_offsets_and_local_lengths(self, total_length):
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
        impurity_orbitals,
        bath_states,
        delta_valence_occ,
        delta_conduction_occ,
        delta_impurity_occ,
        nominal_impurity_occ,
        mixed_valence,
        verbose,
    ):
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
                    int.from_bytes(
                        psr.tuple2bytes(
                            tuple(idx for subset in set_bits for part in subset for idx in part), num_spin_orbitals
                        ),
                        byteorder="little",
                    ).to_bytes(self.n_bytes, "little")
                )

        return basis, num_spin_orbitals

    def _get_restrictions(
        self,
        impurity_orbitals,
        bath_states,
        delta_valence_occ,
        delta_conduction_occ,
        delta_impurity_occ,
        nominal_impurity_occ,
        verbose,
    ):
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

    def get_effective_restrictions(self):
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
                bits = psr.bytes2bitarray(state, self.num_spin_orbitals)
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
    ):
        if dN is None:
            return 0, len(orbs)
        elif orbs not in restrictions:
            return 0, len(orbs)
        occ_dec, occ_inc = dN
        min_occ, max_occ = restrictions[orbs]
        return max(min_occ - occ_dec, 0), min(max_occ + occ_inc, len(orbs))

    def build_excited_restrictions(
        self,
        psis: list[ManyBodyState] | ManyBodyState,
        es: list[float],
        op: ManyBodyOperator,
        imp_change: Optional[dict[int, tuple[int, int]]],
        val_change: Optional[dict[int, tuple[int, int]]],
        con_change: Optional[dict[int, tuple[int, int]]],
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
        if len(psis) == 0:
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
        if self.verbose:
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
                    imp_val_rho = self.build_density_matrices(psis, imp_orb_block, val_orb_block)
                    val_rhos = self.build_density_matrices(psis, val_orb_block, val_orb_block)
                    imp_con_rho = self.build_density_matrices(psis, imp_orb_block, con_orb_block)
                    con_rhos = self.build_density_matrices(psis, con_orb_block, con_orb_block)

                    valence_occupations = thermal_average_scale_indep(
                        es, np.diagonal(val_rhos.real, axis1=1, axis2=2), self.tau
                    )
                    conduction_occupations = thermal_average_scale_indep(
                        es, np.diagonal(con_rhos.real, axis1=1, axis2=2), self.tau
                    )
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
                if len(filled_orbitals) > 1:
                    excited_restrictions[filled_orbitals] = (len(filled_orbitals) - 1, len(filled_orbitals))
                if len(empty_orbitals) > 1:
                    excited_restrictions[empty_orbitals] = (0, 1)
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
        assert (
            impurity_orbitals is not None
        ), "You need to supply the number of impurity orbitals in each set in impurity_orbitals"
        assert bath_states is not None, "You need to supply the number of bath states for each l quantum number"

        test = ManyBodyState({b"\x00": 1.0})
        slater_det = list(test.keys())[0]
        self.type = type(slater_det)
        # self.type = type(psr.int2bytes(0, self.num_spin_orbitals))
        self.n_bytes = len(slater_det)
        # self.n_bytes = int(ceil(self.num_spin_orbitals / 8))
        self.truncation_threshold = truncation_threshold
        self.is_distributed = comm is not None and comm.size > 1
        self.tau = tau
        
        if initial_basis is not None:
            assert nominal_impurity_occ is None
            assert delta_valence_occ is None
            assert delta_conduction_occ is None
            assert delta_impurity_occ is None
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
        self.num_spin_orbitals = sum(
            sum(len(orbs) for orbs in impurity_orbitals[i])
            + sum(len(orbs) for orbs in bath_states[0][i])
            + sum(len(orbs) for orbs in bath_states[1][i])
            for i in bath_states[0]
        )
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

    def alltoall_states(self, send_list: list[list[bytes]], flatten=False):
        return self.state_container.alltoall_states(send_list, flatten)

    def add_states(self, new_states: Iterable[bytes]) -> None:
        """
        Extend the current basis by adding the new_states to it.
        """
        self.state_container.add_states(new_states)

        self.offset = self.state_container.offset
        self.size = self.state_container.size
        self.local_indices = self.state_container.local_indices
        self._index_dict = self.state_container._index_dict
        self.index_bounds = self.state_container.index_bounds
        self.state_bounds = self.state_container.state_bounds
        self.local_basis = self.state_container.local_basis

    def redistribute_psis(self, psis: list[ManyBodyState]):
        if isinstance(psis, ManyBodyState):
            print("WARNING in redistribute_psi:")
            print(
                f"Expetced a list of ManyBodyStates, received a single ManyBodyState. Remaking into list of one ManyBodyState"
            )
            psis = [ManyBodyState]
        if not self.is_distributed:
            return psis

        def find_owner(state):
            for r, bound in enumerate(self.state_bounds):
                if bound is None or state < bound:
                    return r
            return -1

        # for psi in psis:
        #     l = list(psi.keys())
        #     assert all(l[i] < l[i + 1] for i in range(len(psi) - 1)), f'{l}'
        send_list = [[ManyBodyState({}) for _ in psis] for _ in range(self.comm.size)]
        t0 = perf_counter()
        # for state, _ in itertools.groupby(merge(*tuple((state for state in psi.keys()) for psi in psis))):
        for state in sorted({state for psi in psis for state in psi}):

            send_to = find_owner(state)

            for s_psi, l_psi in zip(send_list[send_to], psis):
                if state not in l_psi:
                    continue
                s_psi[state] += l_psi[state]

        t0 = perf_counter()
        received_list = [None for _ in range(self.comm.size)]
        for r_offset in range(self.comm.size):
            send_to = (self.comm.rank + r_offset) % self.comm.size
            receive_from = (self.comm.rank + self.comm.size - r_offset) % self.comm.size

            if send_to == self.comm.rank:
                received_list[receive_from] = send_list[send_to]
                # received_list[receive_from] = [psi.to_dict() for psi in send_list[send_to]]
            else:
                received_list[receive_from] = self.comm.sendrecv(
                    send_list[send_to],
                    # [psi.to_dict() for psi in send_list[send_to]],
                    dest=send_to,
                    source=receive_from,
                )
        res = [ManyBodyState({}) for _ in psis]
        for received_psis in received_list:
            for res_n, psi_n in zip(res, received_psis):
                res_n += psi_n
                # res_n += ManyBodyState(psi_n)
        return res

    def redistribute_psis_old(self, psis: Iterable[dict]):
        if not self.is_distributed:
            return list(psis)

        res = []
        send_to_rank = [[] for _ in range(self.comm.size)]
        send_states = [[] for _ in range(self.comm.size)]
        send_amps = [[] for _ in range(self.comm.size)]
        n_psis = 0
        for n, psi in enumerate(psis):
            n_psis += 1
            for state, amp in psi.items():
                for r, state_bound in enumerate(self.state_bounds):
                    if state_bound is None or state < state_bound:
                        send_states[r].append(state)
                        send_amps[r].append(amp)
                        send_to_rank[r].append(n)
                        break
        send_counts = np.array([len(send_amps[r]) for r in range(self.comm.size)], dtype=np.int64)
        send_offsets = np.array([sum(send_counts[:r]) for r in range(self.comm.size)], dtype=np.int64)
        receive_counts = np.empty((self.comm.size), dtype=np.int64)
        self.comm.Alltoall(np.array(send_counts, dtype=np.int64), receive_counts)
        receive_offsets = np.array([sum(receive_counts[:r]) for r in range(self.comm.size)], dtype=np.int64)
        received_bytes = bytearray(sum(receive_counts) * self.n_bytes)
        received_amps = np.empty(sum(receive_counts), dtype=np.complex128)
        received_splits = np.empty(sum(receive_counts), dtype=np.int64)

        # numpy arrays of bytes do not play very nicely with MPI, sometimes data corruotion happens.
        # MPI4PYs Ialltoallv does not play nice with bytearrays, the call just freezes.
        # The solution to both these issues is to use bytes for sending and bytearrays for receiving.
        received_bytes = bytearray(sum(receive_counts) * self.n_bytes)
        state_request = self.comm.Ialltoallv(
            (
                bytes(byte for state_list in send_states for state in state_list for byte in state),
                send_counts * self.n_bytes,
                send_offsets * self.n_bytes,
                MPI.BYTE,
            ),
            (received_bytes, receive_counts * self.n_bytes, receive_offsets * self.n_bytes, MPI.BYTE),
        )

        received_amps_arr = np.empty((sum(receive_counts),), dtype=complex)
        amps_request = self.comm.Ialltoallv(
            (
                np.array(
                    [amp for amps in send_amps for amp in amps],
                    dtype=np.complex128,
                ),
                send_counts,
                send_offsets,
                MPI.C_DOUBLE_COMPLEX,
            ),
            (received_amps_arr, receive_counts, receive_offsets, MPI.C_DOUBLE_COMPLEX),
        )
        received_splits_arr = np.empty((sum(receive_counts),), dtype=int)
        splits_request = self.comm.Ialltoallv(
            (
                np.array([split for splits in send_to_rank for split in splits], dtype=np.int64),
                send_counts,
                send_offsets,
                MPI.LONG,
                # MPI.INT64_T,
            ),
            (received_splits_arr, receive_counts, receive_offsets, MPI.LONG),
            # (received_splits_arr, receive_counts, receive_offsets, MPI.INT64_T),
        )

        received_states: list[Iterable[bytes]] = [[] for _ in send_states]
        state_request.Wait()
        state_request.free()
        received_states = [
            (
                bytes(r_bytes)
                for r_bytes in batched(
                    received_bytes[
                        receive_offsets[r] * self.n_bytes : (receive_offsets[r] + receive_counts[r]) * self.n_bytes
                    ],
                    self.n_bytes,
                )
            )
            for r in range(self.comm.size)
        ]
        amps_request.Wait()
        amps_request.free()
        received_amps: list[Iterable[complex]] = [
            received_amps_arr[receive_offsets[r] : receive_offsets[r] + receive_counts[r]]
            for r in range(self.comm.size)
        ]
        splits_request.Wait()
        splits_request.free()
        received_splits: list[Iterable[int]] = [
            received_splits_arr[receive_offsets[r] : receive_offsets[r] + receive_counts[r]]
            for r in range(self.comm.size)
        ]
        res = [{} for _ in range(n_psis)]
        for n, state, amp in zip(
            itertools.chain.from_iterable(received_splits),
            itertools.chain.from_iterable(received_states),
            itertools.chain.from_iterable(received_amps),
        ):
            # if state in self.local_basis:
            res[n][state] = amp + res[n].get(state, 0)
        return res

    def _generate_spin_flipped_determinants(self, determinants):
        valence_baths, conduction_baths = self.bath_states
        n_imp_orbs = sum()
        n_dn_op = {
            ((i, "c"), (i, "a")): 1.0
            for l in self.impurity_orbitals
            for i in range(sum(len(orbs) for orbs in self.impurity_orbitals[l]) // 2)
        }
        n_up_op = {
            ((i, "c"), (i, "a")): 1.0
            for l in self.impurity_orbitals
            for i in range(self.impurity_orbitals[l] // 2, self.impurity_orbitals[l])
        }
        spin_flip = set()
        for det in determinants:
            n_dn = int(applyOp(self.num_spin_orbitals, n_dn_op, {det: 1}).get(det, 0))
            n_up = int(applyOp(self.num_spin_orbitals, n_up_op, {det: 1}).get(det, 0))
            spin_flip.add(det)
            to_flip = {det}
            for l in self.impurity_orbitals:
                n_orb = self.impurity_orbitals[l]
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
                        new_n_dn = int(applyOp(self.num_spin_orbitals, n_dn, {flipped_state: 1}).get(flipped_state, 0))
                        new_n_up = int(applyOp(self.num_spin_orbitals, n_up, {flipped_state: 1}).get(flipped_state, 0))
                        if (new_n_dn == n_dn and new_n_up == n_up) or (new_n_dn == n_up and new_n_up == n_dn):
                            spin_flip.update(flipped.keys())

        return spin_flip

    def expand(self, op, dense_cutoff=None, slaterWeightMin=0, max_it=5):
        if isinstance(op, dict):
            op = ManyBodyOperator(op)
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
                        restrictions=self.restrictions,
                    )
                    new_local_states |= set(res.keys()) - local_states
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
        return self.build_operator_dict(op)

    def index(self, val):
        return self.state_container.index(val)

    def __getitem__(self, key) -> Iterable:
        return self.state_container[key]

    def __len__(self):
        return self.state_container.size

    def __contains__(self, item):
        return item in self.state_container

    def contains(self, item) -> Iterable[bool]:
        return self.state_container.contains(item)

    def __iter__(self):
        for state in self.state_container:
            yield state

    def copy(self):
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

    def clear(self):
        self.state_container.clear()
        self.add_states([])

    def build_vector(self, psis: list[ManyBodyState], root: Optional[int] = None) -> np.ndarray:
        v = np.zeros((len(psis), self.size), dtype=complex, order="C")
        psis = self.redistribute_psis(psis)
        # row_states_in_basis: list[bytes] = []
        # row_dict = {state: self._index_dict[state] for state in self.local_basis}
        # col_dict = dict(zip(self.local_basis, range(self.local_indices.start, self.local_indices.stop)))
        for row, psi in enumerate(psis):
            for state, val in psi.items():
                if state not in self._index_dict:
                    continue
                v[row, self._index_dict[state]] = val

        if self.is_distributed and root is None:
            self.comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
        elif self.is_distributed:
            self.comm.Reduce(MPI.IN_PLACE if self.comm.rank == root else v, v, op=MPI.SUM, root=root)
        return v

    def build_distributed_vector(self, psis: list[ManyBodyState], dtype=complex) -> np.ndarray:
        psis = self.redistribute_psis(psis)
        v = np.empty((len(psis), len(self.local_basis)), dtype=dtype, order="C")
        for (row, psi), (col, state) in itertools.product(enumerate(psis), enumerate(self.local_basis)):
            v[row, col] = psi.get(state, 0)
        return v

    def build_state(self, vs: Union[list[np.ndarray], np.ndarray], slaterWeightMin=0) -> list[dict]:
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

    def build_operator_dict(self, op, slaterWeightMin=1e-16):
        """
        Express the operator, op, in the current basis. Do not expand the basis.
        Return a dict containing the results of applying op to the different basis states
        """
        if isinstance(op, dict):
            op = ManyBodyOperator(op)
        res = dict()

        t0 = perf_counter()
        for state in self.local_basis:
            res[state] = applyOp_test(
                op,
                ManyBodyState({state: 1}),
                cutoff=slaterWeightMin,
                restrictions=self.restrictions,
            )
        print(f"Applying the Hamiltonian took {perf_counter() - t0} seconds")
        assert len(res) == len(self.local_basis)
        return res

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

    def build_sparse_matrix(self, op):
        """
        Get the operator as a sparse matrix in the current basis.
        The sparse matrix is distributed over all ranks.
        """
        expanded_dict = self.build_operator_dict(op)

        res_dok = sp.sparse.dok_array((len(self), len(self)), dtype=complex)
        bras = []
        columns = []
        values = []
        for ket, ket_dict in expanded_dict.items():
            if ket not in self._index_dict:
                continue
            columns.extend([self._index_dict[ket]] * len(ket_dict))
            for bra, val in ket_dict.items():
                bras.append(bra)
                values.append(val)
        for row, col, val in zip(self.state_container._index_sequence(bras), columns, values):
            if row == self.size:
                continue
            res_dok[row, col] = val

        return res_dok.tocsc()

    def _state_statistics(self, psi, impurity_indices, valence_indices, conduction_indices, num_spin_orbitals):
        stat = {}
        for state, amp in psi.items():
            bits = psr.bytes2bitarray(state, num_spin_orbitals)
            n_imp = sum(bits[i] for i in impurity_indices)
            n_valence = sum(bits[i] for i in valence_indices)
            n_cond = sum(bits[i] for i in conduction_indices)
            stat[(n_imp, n_valence, n_cond)] = abs(amp) ** 2 + stat.get((n_imp, n_valence, n_cond), 0)
        return stat

    def get_state_statistics(self, psis):
        impurity_indices = [
            orb for blocks in self.impurity_orbitals.values() for imp_orbs in blocks for orb in imp_orbs
        ]
        valence_indices = [orb for blocks in self.bath_states[0].values() for val_orbs in blocks for orb in val_orbs]
        conduction_indices = [orb for blocks in self.bath_states[1].values() for con_orbs in blocks for orb in con_orbs]
        psi_stats = [
            self._state_statistics(psi, impurity_indices, valence_indices, conduction_indices, self.num_spin_orbitals)
            for psi in psis
        ]
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
        if orbital_indices_left is None:
            orbital_indices_left = list(range(self.num_spin_orbitals))
        if orbital_indices_right is None:
            orbital_indices_right = list(range(self.num_spin_orbitals))
        n_left, n_right = (len(orbital_indices_left), len(orbital_indices_right))
        rhos = np.zeros((len(psis), n_left, n_right), dtype=complex)
        for n, psi_n in enumerate(psis):

            psi_ps = []
            for (i, orb_i), (j, orb_j) in itertools.product(
                enumerate(orbital_indices_left), enumerate(orbital_indices_right)
            ):
                op = ManyBodyOperator({((orb_i, "c"), (orb_j, "a")): 1.0})
                psi_ps.append(op(psi_n))
            if self.is_distributed:
                psi_ps = self.redistribute_psis(psi_ps)
            for (i, j), psi_p in zip(itertools.product(range(n_left), range(n_right)), psi_ps):
                amp = inner(psi_n, psi_p)
                if abs(amp) > np.finfo(float).eps:
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
            tmps[i] = set(
                applyOp_test(
                    op, ManyBodyState({state: 1.0}), restrictions=self.restrictions, cutoff=slaterWeightMin
                ).keys()
            )
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

    def split_basis_and_redistribute_psi(self, priorities, psis, max_stddev=0.3):

        if (not self.is_distributed) or len(priorities) <= 1:
            return range(len(priorities)), [0], 0, [len(priorities)], self, psis, [None]

        comm = self.comm
        rank = comm.rank
        normalized_priorities = np.array([p for p in priorities], dtype=float)
        normalized_priorities /= np.sum(np.abs(normalized_priorities))
        n_colors = min(comm.size, len(normalized_priorities))

        # Group priorities into colors so that all colours have roughly the same priorities
        # Step 1:
        #        Each priority starts in its own subgroup
        # while not equal priorities
        #       merge the two subgroups with lowest priorities
        priority_groups = DisjointSet(list(range(len(normalized_priorities))))
        subgroups = priority_groups.subsets()
        weights = np.array([np.sum(normalized_priorities[list(idxs)]) for idxs in subgroups])
        avg_weight = np.sum(weights) / len(weights)
        stddev = np.sqrt(np.sum((weights - avg_weight) ** 2) / (len(subgroups) - 1))
        while len(subgroups) > n_colors or stddev > max_stddev:
            subgroups = sorted(subgroups, key=lambda idxs: np.sum(normalized_priorities[list(idxs)]))
            priority_groups.merge(next(iter(subgroups[0])), next(iter(subgroups[1])))
            subgroups = priority_groups.subsets()
            if len(subgroups) <= 1:
                break
            weights = np.array([np.sum(normalized_priorities[list(idxs)]) for idxs in subgroups])
            avg_weight = np.sum(weights) / len(weights)
            stddev = np.sqrt(np.sum((weights - avg_weight) ** 2) / (len(subgroups) - 1))
        subgroups = sorted(subgroups, key=lambda idxs: np.sum(normalized_priorities[list(idxs)]))
        n_colors = len(subgroups)
        merged_priorities = np.array([np.sum(normalized_priorities[list(subgroup)]) for subgroup in subgroups])
        procs_per_color = np.array([max(1, n) for n in np.floor(comm.size * merged_priorities)], dtype=int)
        remainder = comm.size - np.sum(procs_per_color)
        while remainder != 0:
            if remainder < 0:
                mask = np.nonzero(procs_per_color > 1)

                procs_per_color[mask[-(abs(remainder) % n_colors) :]] -= 1
            else:
                procs_per_color[-(abs(remainder) % n_colors) :] += 1
            remainder = comm.size - np.sum(procs_per_color)
        # comm.Bcast(procs_per_color, root=0)

        assert sum(procs_per_color) == comm.size
        proc_cutoffs = np.cumsum(procs_per_color)
        print(f"{proc_cutoffs=}")
        color = np.argmax(comm.rank < proc_cutoffs)
        print(f"{color=}")

        split_comm = comm.Split(color=color, key=comm.rank)
        split_roots = [0] + proc_cutoffs[:-1].tolist()
        print(f"{split_roots=}")
        items_per_color = [len(subgroup) for subgroup in subgroups]
        assert sum(items_per_color) == len(priorities)

        assert comm.is_intra
        assert not comm.is_inter
        assert split_comm.is_intra
        assert not split_comm.is_inter
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
                intercomms[c].send(self.local_basis, dest=split_comm.rank % procs_per_color[c])
            # I will receive states from all other colors
            else:
                for send_color in range(len(split_roots)):
                    if send_color == color:
                        continue
                    for sender in range(procs_per_color[send_color]):
                        if sender % procs_per_color[c] != split_comm.rank:
                            continue
                        new_states |= set(intercomms[send_color].recv(source=sender))

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
                    intercomms[c].send(psis, dest=split_comm.rank % procs_per_color[c])
                    # intercomms[c].send([p.to_dict() for p in psis], dest=split_comm.rank % procs_per_color[c])
                else:
                    for send_color in range(len(split_roots)):
                        if send_color == color:
                            continue
                        for sender in range(procs_per_color[send_color]):
                            if sender % procs_per_color[color] != split_comm.rank:
                                continue
                            received_psis = intercomms[send_color].recv(source=sender)
                            for i, received_psi in enumerate(received_psis):
                                new_psis[i] += received_psi
                                # new_psis[i] += ManyBodyState(received_psi)
            psis = split_basis.redistribute_psis(new_psis)
        return indices, split_roots, color, items_per_color, split_basis, psis, intercomms

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

        block_indices, block_roots, block_color, blocks_per_color, block_basis, _, block_intercomms = (
            tmp_basis.split_basis_and_redistribute_psi(block_lengths**2, None)
        )
        block_roots = np.array(block_roots, dtype=int)
        procs_per_color = block_roots[1:] - block_roots[:-1]
        procs_per_color = np.append(procs_per_color, [comm.size - np.sum(procs_per_color)])

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
                        new_states |= block_intercomms[send_color].recv(source=sender)
            else:
                start = block_index_color_offsets[c]
                stop = start + num_block_indices_per_color[c]
                block_intercomms[c].send(
                    {
                        self.local_basis[local_idx - self.offset]
                        for block_idx in block_indices_per_color[start:stop]
                        for local_idx in blocks[block_idx]
                    },
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
                                new_psis[i] += r_psi_dict
                                # new_psis[i] += ManyBodyState(r_psi_dict)
                else:
                    start = block_index_color_offsets[c]
                    stop = start + num_block_indices_per_color[c]
                    send_states = {
                        self.local_basis[local_idx - self.offset]
                        for block_idx in block_indices_per_color[start:stop]
                        for local_idx in blocks[block_idx]
                    }
                    block_intercomms[c].send(
                        [
                            ManyBodyState({state: amp for state, amp in psi.items() if state in send_states})
                            for psi in psis
                        ],
                        dest=block_basis.comm.rank % procs_per_color[c],
                    )
            psis = block_basis.redistribute_psis(new_psis)

        return block_indices, block_roots, block_color, blocks_per_color, block_basis, psis, block_intercomms


class CIPSI_Basis(Basis):
    def __init__(self, impurity_orbitals, bath_states, H=None, nominal_impurity_occ=None, initial_basis=None, **kwargs):
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

    def truncate(self, psis):
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
        calculate second variational energy contribution of the Slater determinants in states.
        """

        if isinstance(H, dict):
            H = ManyBodyOperator(H)
        local_Djs = sorted({state for hp in Hpsi_ref for state in hp if state not in self._index_dict})
        overlaps = np.zeros((len(Hpsi_ref), len(local_Djs)), dtype=complex)
        e_Dj = np.zeros((len(Hpsi_ref), len(local_Djs)), dtype=float)
        for (j, Dj), (i, Hpsi_i) in itertools.product(enumerate(local_Djs), enumerate(Hpsi_ref)):
            if Dj not in Hpsi_i:
                continue
            # <Dj|H|Psi_i>
            overlaps[i, j] = Hpsi_i[Dj]
            # <Dj|H|Dj>
            HDj = applyOp_test(
                H,
                ManyBodyState({Dj: 1}),
                cutoff=slaterWeightMin,
                restrictions=self.restrictions,
            )
            e_Dj[i, j] = np.real(HDj[Dj])
        # E_i - <Dj|H|Dj>
        de = e_ref[:, None] - e_Dj

        #      {Dj},      <Dj|H|Psi_i>^2 / (E_i - <Dj|H|Dj>)
        de = np.maximum(de, 1e-12)
        de2 = np.zeros_like(overlaps)
        mask = np.abs(overlaps) > 1e-12
        de2[mask] = np.square(np.abs(overlaps[mask])) / de[mask]
        return local_Djs, de2

    def determine_new_Dj(self, e_ref, psi_ref, H, de2_min, slater_cutoff=0, return_Hpsi_ref=False):
        if isinstance(H, dict):
            H = ManyBodyOperator(H)
        new_Dj = set()
        Hpsi_ref = [None for _ in psi_ref]
        for i, (e_i, psi_i) in enumerate(zip(e_ref, psi_ref)):
            Hpsi_ref[i] = applyOp_test(
                H,
                psi_i,
                slater_cutoff,
                restrictions=self.restrictions,
            )
        Hpsi_ref = self.redistribute_psis(Hpsi_ref)
        local_Djs, de2 = self._calc_de2(H, Hpsi_ref, e_ref)
        de2_mask = np.any(np.abs(de2) >= de2_min, axis=0)
        new_Dj = {Dj for Dj in itertools.compress(local_Djs, de2_mask)}
        if return_Hpsi_ref:
            return new_Dj, Hpsi_ref

        return new_Dj

    def expand(self, H, de2_min=1e-10, dense_cutoff=1e3, slaterWeightMin=0):
        """
        Use the CIPSI method to expand the basis. Keep adding Slater determinants until the CIPSI energy is converged.
        """
        de0_max = -self.tau * np.log(1e-4)
        psi_refs = None
        e_ref = None

        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        old_size = self.size - 1
        while old_size != self.size:

            H_mat = self.build_sparse_matrix(H)
            # e_ref = np.array([], dtype=float)
            # new_psi_refs = []

            e_ref, psi_ref_dense = eigensystem_new(
                H_mat,
                e_max=de0_max,
                k=2 * len(psi_refs) if psi_refs is not None else 10,
                e0=None,
                v0=self.build_vector(psi_refs).T if psi_refs is not None and self.size >= dense_cutoff else None,
                eigenValueTol=0,
                comm=self.comm,
                dense=self.size < dense_cutoff,
            )

            psi_refs = self.build_state(psi_ref_dense.T)

            new_Dj = self.determine_new_Dj(e_ref, psi_refs, H, de2_min, slater_cutoff=slaterWeightMin)
            old_size = self.size
            self.add_states(new_Dj)
            psi_refs = self.redistribute_psis(psi_refs)
            if self.size > self.truncation_threshold:
                psi_refs = self.truncate(psi_refs)
                print(f"-----> Basis truncated!")
                break

        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.", flush=True)

        return self.build_operator_dict(H)

    def expand_at(self, E_ref, psi_ref, H, de2_min=1e-5):

        if isinstance(H, dict):
            H = ManyBodyOperator(H)
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

        return self.build_operator_dict(H)

    def copy(self):
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

    def split_basis_and_redistribute_psi(self, priorities, psis, max_stddev=0.3):

        indices, split_roots, color, items_per_color, split_basis, psis, intercomms = (
            super().split_basis_and_redistribute_psi(priorities, psis, max_stddev)
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
