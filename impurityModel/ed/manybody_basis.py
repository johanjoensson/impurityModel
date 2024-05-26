from math import ceil
from time import perf_counter
import sys
from typing import Optional, Union

try:
    from collections.abc import Sequence, Iterable
except ModuleNotFoundError:
    from collections import Sequence, Iterable
import itertools
from heapq import merge
import numpy as np
import scipy as sp
from mpi4py import MPI
from impurityModel.ed.manybody_state_containers import DistributedStateContainer, CentralizedStateContainer


from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.finite import applyOp_new as applyOp, c2i, c2i_op, eigensystem_new, norm2

try:
    from petsc4py import PETSc
except ModuleNotFoundError:
    pass


def batched(iterable: Iterable, n: int) -> Iterable:
    """
    batched('ABCDEFG', 3) → ABC DEF G
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


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
        valence_baths,
        conduction_baths,
        delta_valence_occ,
        delta_conduction_occ,
        delta_impurity_occ,
        nominal_impurity_occ,
        verbose,
    ):
        total_baths = {i: valence_baths[i] + conduction_baths[i] for i in valence_baths}
        configurations = {}
        n_imp_orbs = 0
        n_val_orbs = sum(imp_orbs for imp_orbs in impurity_orbitals.values())
        n_cond_orbs = n_val_orbs + sum(val_orbs for val_orbs in valence_baths.values())
        for i in valence_baths:
            if verbose:
                print(f"{i=}")
            valid_configurations = []
            for delta_valence in range(delta_valence_occ[i] + 1):
                for delta_conduction in range(delta_conduction_occ[i] + 1):
                    delta_impurity = delta_valence - delta_conduction
                    if (
                        abs(delta_impurity) <= delta_impurity_occ[i]
                        and nominal_impurity_occ[i] + delta_impurity <= impurity_orbitals[i]
                        and nominal_impurity_occ[i] + delta_impurity >= 0
                    ):
                        impurity_occupation = nominal_impurity_occ[i] + delta_impurity
                        valence_occupation = valence_baths[i] - delta_valence
                        conduction_occupation = delta_conduction
                        if verbose:
                            print("Partition occupations")
                            print(f"Impurity occupation:   {impurity_occupation:d}")
                            print(f"Valence occupation:   {valence_occupation:d}")
                            print(f"Conduction occupation: {conduction_occupation:d}")
                        impurity_electron_indices = list(range(n_imp_orbs, n_imp_orbs + impurity_orbitals[i]))
                        impurity_configurations = itertools.combinations(impurity_electron_indices, impurity_occupation)
                        valence_electron_indices = list(range(n_val_orbs, n_val_orbs + valence_baths[i]))
                        valence_configurations = itertools.combinations(valence_electron_indices, valence_occupation)
                        conduction_electron_indices = list(range(n_cond_orbs, n_cond_orbs + conduction_baths[i]))
                        conduction_configurations = itertools.combinations(
                            conduction_electron_indices, conduction_occupation
                        )
                        valid_configurations.append(
                            itertools.product(
                                impurity_configurations, valence_configurations, conduction_configurations
                            )
                        )
            configurations[i] = [
                imp + val + cond for configuration in valid_configurations for (imp, val, cond) in configuration
            ]
            n_imp_orbs += impurity_orbitals[i]
            n_val_orbs += valence_baths[i]
            n_cond_orbs += conduction_baths[i]
        num_spin_orbitals = sum(impurity_orbitals[i] + total_baths[i] for i in total_baths)
        basis = []
        # Combine all valid configurations for all l-subconfigurations (ex. p-states and d-states)
        for system_configuration in itertools.product(*configurations.values()):
            basis.append(
                psr.tuple2bytes(
                    tuple(sorted(itertools.chain.from_iterable(system_configuration))),
                    num_spin_orbitals,
                )
            )
        return basis, num_spin_orbitals

    def _get_restrictions(
        self,
        impurity_orbitals,
        valence_baths,
        conduction_baths,
        delta_valence_occ,
        delta_conduction_occ,
        delta_impurity_occ,
        nominal_impurity_occ,
        verbose,
    ):
        restrictions = {}
        total_baths = {i: valence_baths[i] + conduction_baths[i] for i in valence_baths}
        impurity_orbs = 0
        valence_orbs = sum(imp_orbs for imp_orbs in impurity_orbitals.values())
        conduction_orbs = valence_orbs + sum(val_orbs for val_orbs in valence_orbitals.values())
        for i in total_baths:
            impurity_indices = frozenset(range(impurity_orbs, impurity_orbs + impurity_orbitals[i]))
            restrictions[impurity_indices] = (
                max(nominal_impurity_occ[i] - delta_impurity_occ[i], 0),
                min(nominal_impurity_occ[i] + delta_impurity_occ[i] + 1, impurity_orbitals[i] + 1),
            )
            valence_indices = frozenset(range(valence_orbs, valence_orbs + valence_baths[i]))
            restrictions[valence_indices] = (max(valence_baths[i] - delta_valence_occ[i], 0), valence_baths[i] + 1)
            conduction_indices = frozenset(range(conduction_orbs, conduction_orbs + conduction_baths[i]))
            restrictions[conduction_indices] = (0, delta_conduction_occ[i] + 1)
            impurity_orbs += impurity_orbitals[i]
            valence_orbs += valence_baths[i]
            conduction_orbs += conduction_baths[i]

            if verbose:
                print(f"l = {i}")
                print(f"|---Restrictions on the impurity orbitals = {restrictions[impurity_indices]}")
                print(f"|---Restrictions on the valence bath      = {restrictions[valence_indices]}")
                print(f"----Restrictions on the conduction bath   = {restrictions[conduction_indices]}")

        return restrictions

    def get_effective_restrictions(self):
        valence_baths, conduction_baths = self.bath_states

        total_baths = {i: valence_baths[i] + conduction_baths[i] for i in valence_baths}
        restrictions = {}
        n_imp_orbs = sum(num for num in self.impurity_orbitals.values())
        n_valence_states = sum(num for num in valence_baths.values())
        n_conduction_states = sum(num for num in conduction_baths.values())

        imp_orbs = 0
        val_orbs = n_imp_orbs
        con_orbs = n_imp_orbs + n_conduction_states
        for i in total_baths:
            max_imp = 0
            min_imp = self.impurity_orbitals[i]
            max_val = 0
            min_val = valence_baths[i]
            max_con = 0
            min_con = conduction_baths[i]
            impurity_indices = frozenset(ind for ind in range(imp_orbs, imp_orbs + self.impurity_orbitals[i]))
            valence_indices = frozenset(ind for ind in range(val_orbs, val_orbs + valence_baths[i]))
            conduction_indices = frozenset(ind for ind in range(con_orbs, con_orbs + conduction_baths[i]))
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
            max_imp = self.comm.allreduce(max_imp, op=MPI.MAX)
            min_imp = self.comm.allreduce(min_imp, op=MPI.MIN)
            max_val = valence_baths[i]
            min_val = self.comm.allreduce(min_val, op=MPI.MIN)
            max_con = self.comm.allreduce(max_con, op=MPI.MAX)
            min_con = 0
            restrictions[impurity_indices] = (min_imp, max_imp)
            restrictions[valence_indices] = (min_val, max_val)
            restrictions[conduction_indices] = (min_con, max_con)
        return restrictions

    def build_excited_restrictions(self, imp_change=(1, 1), val_change=(1, 0), con_change=(0, 1)):
        imp_reduce, imp_increase = imp_change
        val_reduce, _ = val_change
        _, con_increase = con_change
        valence_baths, conduction_baths = self.bath_states
        total_baths = {i: valence_baths[i] + conduction_baths[i] for i in valence_baths}
        restrictions = self.get_effective_restrictions()
        excited_restrictions = {}
        n_imp_orbs = sum(num for num in self.impurity_orbitals.values())
        n_valence_states = sum(num for num in valence_baths.values())
        n_conduction_states = sum(num for num in conduction_baths.values())
        imp_orbs = 0
        val_orbs = n_imp_orbs
        con_orbs = n_imp_orbs + n_valence_states
        for i in total_baths:
            impurity_indices = frozenset(ind for ind in range(imp_orbs, imp_orbs + self.impurity_orbitals[i]))
            valence_indices = frozenset(ind for ind in range(val_orbs, val_orbs + valence_baths[i]))
            conduction_indices = frozenset(ind for ind in range(con_orbs, con_orbs + conduction_baths[i]))
            r_min_imp, r_max_imp = restrictions[impurity_indices]
            min_imp = max(r_min_imp - imp_reduce, 0)
            max_imp = min(r_max_imp + imp_increase, self.impurity_orbitals[i])
            r_min_val, r_max_val = restrictions[valence_indices]
            min_val = max(r_min_val - val_reduce, 0)
            max_val = valence_baths[i]
            r_min_cond, r_max_cond = restrictions[conduction_indices]
            min_cond = 0
            max_cond = min(r_max_cond + con_increase, conduction_baths[i])
            excited_restrictions[impurity_indices] = (min_imp, max_imp)
            excited_restrictions[valence_indices] = (min_val, max_val)
            excited_restrictions[conduction_indices] = (min_cond, max_cond)
        return excited_restrictions

    def __init__(
        self,
        impurity_orbitals,
        initial_basis=None,
        restrictions=None,
        valence_baths=None,
        conduction_baths=None,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        nominal_impurity_occ=None,
        truncation_threshold=np.inf,
        spin_flip_dj=False,
        tau=0,
        comm=None,
        verbose=True,
        debug=False,
    ):
        t0 = perf_counter()
        assert (
            impurity_orbitals is not None
        ), "You need to supply the number of impurity orbitals in each set in impurity_orbitals"
        assert valence_baths is not None, "You need to supply the number of bath states for each l quantum number"
        assert conduction_baths is not None, "You need to supply the number of bath states for each l quantum number"
        bath_states = (valence_baths, conduction_baths)
        if initial_basis is not None:
            assert nominal_impurity_occ is None
            assert delta_valence_occ is None
            assert delta_conduction_occ is None
            assert delta_impurity_occ is None
        else:
            assert restrictions is None
            initial_basis, num_spin_orbitals = self._get_initial_basis(
                impurity_orbitals=impurity_orbitals,
                valence_baths=valence_baths,
                conduction_baths=conduction_baths,
                delta_valence_occ=delta_valence_occ,
                delta_conduction_occ=delta_conduction_occ,
                delta_impurity_occ=delta_impurity_occ,
                nominal_impurity_occ=nominal_impurity_occ,
                verbose=verbose,
            )
            restrictions = self._get_restrictions(
                impurity_orbitals=impurity_orbitals,
                valence_baths=valence_baths,
                conduction_baths=conduction_baths,
                delta_valence_occ=delta_valence_occ,
                delta_conduction_occ=delta_conduction_occ,
                delta_impurity_occ=delta_impurity_occ,
                nominal_impurity_occ=nominal_impurity_occ,
                verbose=verbose,
            )
        t0 = perf_counter() - t0
        t0 = perf_counter()
        self.impurity_orbitals = impurity_orbitals
        self.bath_states = bath_states
        self.spin_flip_dj = spin_flip_dj
        self.verbose = verbose
        self.debug = debug
        self.comm = comm
        self.num_spin_orbitals = sum(
            impurity_orbitals[i] + valence_baths[i] + conduction_baths[i] for i in impurity_orbitals
        )
        self.restrictions = restrictions
        self.type = type(psr.int2bytes(0, self.num_spin_orbitals))
        self.n_bytes = int(ceil(self.num_spin_orbitals / 8))
        self.truncation_threshold = truncation_threshold
        self.is_distributed = comm is not None
        t0 = perf_counter() - t0
        t0 = perf_counter()
        if comm is not None:
            seed_sequences = None
            if self.comm.rank == 0:
                seed_parent = np.random.SeedSequence()
                seed_sequences = seed_parent.spawn(comm.size)
            seed_sequence = comm.scatter(seed_sequences, root=0)
            self.rng = np.random.default_rng(seed_sequence)
        else:
            self.rng = np.random.default_rng()
        t0 = perf_counter() - t0
        self.tau = tau

        t0 = perf_counter()

        # self.state_container = CentralizedStateContainer(
        self.state_container = DistributedStateContainer(
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

    def redistribute_psis(self, psis: Iterable[dict]):
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
        received_amps: list[Iterable[complex]] = [
            received_amps_arr[receive_offsets[r] : receive_offsets[r] + receive_counts[r]]
            for r in range(self.comm.size)
        ]
        splits_request.Wait()
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
            res[n][state] = amp + res[n].get(state, 0)
        return res

    def _generate_spin_flipped_determinants(self, determinants):
        valence_baths, conduction_baths = self.bath_states
        n_dn_op = {
            ((i, "c"), (i, "a")): 1.0 for l in self.impurity_orbitals for i in range(self.impurity_orbitals[l] // 2)
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

    def expand(self, op, op_dict=None, dense_cutoff=None, slaterWeightMin=0):
        old_size = self.size - 1
        t0 = perf_counter()
        t_apply = 0
        t_filter = 0
        t_add = 0
        t_keys = 0
        # states_to_check = set(self.local_basis)
        new_states = set()
        # checked_states = set()
        while old_size != self.size and self.size < self.truncation_threshold:
            # while len(states_to_check) > 0:
            # checked_states |= states_to_check
            # for state in states_to_check:
            for state in self.local_basis:
                t_tmp = perf_counter()
                res = applyOp(
                    self.num_spin_orbitals,
                    op,
                    {state: 1},
                    restrictions=self.restrictions,
                    slaterWeightMin=slaterWeightMin,
                    opResult=op_dict,
                )
                t_apply += perf_counter() - t_tmp
                t_tmp = perf_counter()
                new_states |= set(res.keys())  #  - set(self.local_basis)
                t_keys += perf_counter() - t_tmp
                # if a state appears in op_dict it means it has already been evaluted
            t_tmp = perf_counter()
            filtered_states = new_states
            t_filter += perf_counter() - t_tmp
            if self.spin_flip_dj:
                filtered_states = self._generate_spin_flipped_determinants(filtered_states)
            t_tmp = perf_counter()
            old_size = self.size
            self.add_states(filtered_states)
            t_add += perf_counter() - t_tmp
        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.")
        op_dict = self.build_operator_dict(op, op_dict=op_dict)
        return op_dict

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
            impurity_orbitals=self.impurity_orbitals,
            valence_baths=self.bath_states[0],
            conduction_baths=self.bath_states[1],
            initial_basis=self.local_basis,
            restrictions=self.restrictions,
            spin_flip_dj=self.spin_flip_dj,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            verbose=self.verbose,
        )

    def clear(self):
        self.state_container.clear()
        self.add_states([])

    def build_vector(self, psis: list[dict], root: Optional[int] = None) -> np.ndarray:
        v_local = np.zeros((len(psis), self.size), dtype=complex)
        v = np.empty_like(v_local)
        psis = self.redistribute_psis(psis)
        # row_states_in_basis: list[bytes] = []
        row_dict = self._index_dict
        for row, psi in enumerate(psis):
            for state, val in psi.items():
                if state not in row_dict:
                    continue
                v_local[row, row_dict[state]] = val

        if self.is_distributed and root is None:
            self.comm.Allreduce(v_local, v, op=MPI.SUM)
        elif self.is_distributed:
            self.comm.Reduce(v_local, v, op=MPI.SUM, root=root)
        else:
            v = v_local
        return v

    def build_distributed_vector(self, psis: list[dict], dtype=complex) -> np.ndarray:
        v = np.zeros((len(psis), len(self.local_basis)), dtype=dtype, order="C")
        psis_new = self.redistribute_psis(psis)
        for row, psi in enumerate(psis_new):
            for state in psi:
                if state not in self._index_dict:
                    continue
                v[row, self._index_dict[state] - self.offset] = psi[state]
        return v

    def build_state(self, vs: Union[list[np.ndarray], np.ndarray], slaterWeightMin=0) -> list[dict]:
        if isinstance(vs, np.matrix):
            vs = vs.A
        if isinstance(vs, np.ndarray) and len(vs.shape) == 1:
            vs = vs.reshape((1, vs.shape[0]))
        if isinstance(vs, list):
            vs = np.array(vs)
        res = [{} for _ in range(vs.shape[0])]
        for row, i in itertools.product(range(vs.shape[0]), self.local_indices):
            psi = res[row]
            if abs(vs[row, i]) ** 2 > slaterWeightMin:
                psi[self.local_basis[i - self.offset]] = vs[row, i]
        return res

    def build_operator_dict(self, op, op_dict=None, slaterWeightMin=0):
        """
        Express the operator, op, in the current basis. Do not expand the basis.
        Return a dict containing the results of applying op to the different basis states
        """
        if op_dict is None:
            op_dict = {}

        for state in self.local_basis:
            _ = applyOp(
                self.num_spin_orbitals,
                op,
                {state: 1},
                restrictions=self.restrictions,
                slaterWeightMin=slaterWeightMin,
                opResult=op_dict,
            )
        # op_dict.clear()
        # op_dict.update(new_op_dict)
        # return {state: op_dict[state] for state in self.local_basis}
        return op_dict

    def build_dense_matrix(self, op, op_dict=None, distribute=True):
        """
        Get the operator as a dense matrix in the current basis.
        by default the dense matrix is distributed to all ranks.
        """
        h_local = self.build_sparse_matrix(op, op_dict, petsc=False)
        local_dok = h_local.todok()
        if self.is_distributed:
            reduced_dok = self.comm.reduce(local_dok, op=MPI.SUM, root=0)
            if self.comm.rank == 0:
                h = reduced_dok.todense()
            h = self.comm.bcast(h if self.comm.rank == 0 else None, root=0)
        else:
            h = h_local.todense()
        return h

    def build_sparse_matrix(
        self, op, op_dict: Optional[dict[bytes, dict[bytes, complex]]] = None, petsc="petsc4py" in sys.modules
    ):
        """
        Get the operator as a sparse matrix in the current basis.
        The sparse matrix is distributed over all ranks.
        """
        if petsc:
            return self._build_PETSc_matrix(op, op_dict)

        op_dict = self.build_operator_dict(op, op_dict)
        rows: list[int] = []
        columns: list[int] = []
        values: list[complex] = []
        if not self.is_distributed:
            for column in self.local_basis:
                for row in op_dict[column]:
                    if row not in self._index_dict:
                        continue
                    columns.append(self._index_dict[column])
                    rows.append(self._index_dict[row])
                    values.append(op_dict[column][row])
        else:
            rows_in_basis: set[bytes] = {row for column in self.local_basis for row in op_dict[column].keys()}
            row_dict = {
                state: index
                for state, index in zip(rows_in_basis, self.state_container._index_sequence(rows_in_basis))
                if index != self.size
            }

            for column in self.local_basis:
                for row in op_dict[column]:
                    if row not in row_dict:
                        continue
                    columns.append(self._index_dict[column])
                    rows.append(row_dict[row])
                    values.append(op_dict[column][row])
            if self.debug and len(rows) > 0:
                print(f"{self.size=} {max(rows)=}", flush=True)
        return sp.sparse.csc_matrix((values, (rows, columns)), shape=(self.size, self.size), dtype=complex)

    def _build_PETSc_vector(self, psis: list[dict], dtype=complex) -> PETSc.Mat:
        vs = PETSc.Mat().create(comm=self.comm)
        vs.setSizes([len(psis), self.size])
        row_dict = self._index_dict
        for row, psi in enumerate(psis):
            row_states = set(psi.keys())
            need_mpi = False
            if self.is_distributed:
                need_mpi = False
                if any(state not in row_dict for state in psi):
                    need_mpi = True
                need_mpi_arr = np.empty((1,), dtype=bool)
                self.comm.Allreduce(np.array([need_mpi], dtype=bool), need_mpi_arr, op=MPI.LOR)
                need_mpi = need_mpi_arr[0]
            if need_mpi:
                sorted_row_states = row_states  # sorted(row_states)
                row_dict = {
                    state: i for state, i in zip(sorted_row_states, self.index(sorted_row_states)) if i < self.size
                }
            vs.setUp()
            for state, val in psi.items():
                if state not in row_dict:
                    continue
                vs[row, row_dict[state]] = val
            vs.assemble()
        return vs

    def _build_PETSc_matrix(self, op, op_dict=None):
        """
        Get the operator as a sparse matrix in the current basis.
        The sparse matrix is distributed over all ranks.
        """

        M = PETSc.Mat().create(comm=self.comm)
        M.setSizes([self.size, self.size])

        expanded_dict = self.build_operator_dict(op, op_dict)
        rows: list[int] = []
        columns: list[int] = []
        values: list[complex] = []
        if not self.is_distributed:
            for column in self.local_basis:
                for row in expanded_dict[column]:
                    if row not in self._index_dict:
                        continue
                    columns.append(self._index_dict[column])
                    rows.append(self._index_dict[row])
                    values.append(expanded_dict[column][row])
        else:
            rows_in_basis: set[bytes] = {row for column in self.local_basis for row in op_dict[column].keys()}
            row_dict = {
                state: index
                for state, index in zip(rows_in_basis, self.state_container._index_sequence(rows_in_basis))
                if index != self.size
            }

            for column in self.local_basis:
                for row in op_dict[column]:
                    if row not in row_dict:
                        continue
                    columns.append(self._index_dict[column])
                    rows.append(row_dict[row])
                    values.append(op_dict[column][row])
        M.setUp()
        for i, j, val in zip(rows, columns, values):
            M[i, j] = val
        M.assemble()
        return M

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
        n_imp = sum(ni for ni in self.impurity_orbitals.values())
        n_val = sum(nv for nv in self.bath_states[0].values())
        impurity_indices = range(0, n_imp)
        valence_indices = range(n_imp, n_imp + n_val)
        conduction_indices = range(n_imp + n_val, self.num_spin_orbitals)
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


class CIPSI_Basis(Basis):
    def __init__(
        self,
        impurity_orbitals,
        valence_baths=None,
        conduction_baths=None,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        nominal_impurity_occ=None,
        initial_basis=None,
        restrictions=None,
        truncation_threshold=np.inf,
        spin_flip_dj=False,
        verbose=False,
        H=None,
        tau=0,
        comm=None,
    ):
        assert valence_baths is not None
        assert conduction_baths is not None
        bath_states = (valence_baths, conduction_baths)
        if initial_basis is None:
            assert nominal_impurity_occ is not None
            initial_basis, num_spin_orbitals = self._get_initial_basis(
                impurity_orbitals,
                valence_baths,
                conduction_baths,
                delta_valence_occ,
                delta_conduction_occ,
                delta_impurity_occ,
                nominal_impurity_occ,
                verbose,
            )
        super(CIPSI_Basis, self).__init__(
            impurity_orbitals=impurity_orbitals,
            valence_baths=bath_states[0],
            conduction_baths=bath_states[1],
            initial_basis=initial_basis,
            restrictions=restrictions,
            truncation_threshold=truncation_threshold,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
            verbose=verbose,
            comm=comm,
        )

        if self.size > self.truncation_threshold and H is not None:
            if self.verbose:
                print("Truncating basis!")
            H_sparse = self.build_sparse_matrix(H)
            e_ref, psi_ref = eigensystem_new(
                H_sparse,
                e_max=1e-12,
                k=sum(2 * (2 * l + 1) for l in self.ls),
                eigenValueTol=0,
                verbose=self.verbose,
            )
            self.truncate(self.build_state(psi_ref))

    def truncate(self, psis):
        self.local_basis.clear()
        basis_states = {state for psi in psis for state in psi}
        coefficients = np.empty(
            (
                len(
                    basis_states,
                )
            )
        )
        for i, state in enumerate(basis_states):
            coefficients[i] = np.max([abs(psi[state]) for psi in psis if state in psi])
        sort_order = np.argsort(coefficients)[::-1]
        new_basis = []
        for i in range(self.truncation_threshold):
            new_basis.append(list(basis_states)[sort_order[i]])
        self.add_states(new_basis)

    def _calc_de2(self, Djs: Basis, H: dict, H_dict: dict, Hpsi_ref: dict, e_ref: float, slaterWeightMin: float = 0):
        """
        calculate second variational energy contribution of the Slater determinants in states.
        """

        overlaps = np.empty((len(Djs.local_basis)), dtype=complex)
        e_Dj = np.empty((len(Djs.local_basis)), dtype=float)
        for j, (Dj, overlap) in enumerate((d, Hpsi_ref[d]) for d in Djs.local_basis):
            overlaps[j] = overlap
            HDj = applyOp(
                self.num_spin_orbitals,
                H,
                {Dj: 1},
                restrictions=self.restrictions,
                slaterWeightMin=slaterWeightMin,
                opResult=H_dict,
            )
            # <Dj|H|Dj>
            e_Dj[j] = np.real(HDj.get(Dj, 0))
        de = e_ref - e_Dj
        de[np.abs(de) < np.finfo(float).eps] = np.finfo(float).eps

        # <Dj|H|Psi_ref>^2 / (E_ref - <Dj|H|Dj>)
        return np.square(np.abs(overlaps)) / de

    def determine_new_Dj(self, e_ref, psi_ref, H, H_dict, de2_min, return_Hpsi_ref=False):
        new_Dj = set()
        Hpsi_ref = []
        for e_i, psi_i in zip(e_ref, psi_ref):
            Hpsi_i = applyOp(
                self.num_spin_orbitals,
                H,
                psi_i,
                restrictions=self.restrictions,
                opResult=H_dict,
            )
            Dj_candidates = Hpsi_i.keys()
            Dj_basis_mask = (not x for x in self.contains(Dj_candidates))
            Dj_basis = Basis(
                impurity_orbitals=self.impurity_orbitals,
                valence_baths=self.bath_states[0],
                conduction_baths=self.bath_states[1],
                initial_basis=itertools.compress(Dj_candidates, Dj_basis_mask),
                restrictions=None,
                comm=self.comm,
                verbose=False,
            )
            Hpsi_i = Dj_basis.redistribute_psis([Hpsi_i])[0]
            de2 = self._calc_de2(Dj_basis, H, H_dict, Hpsi_i, e_i)
            de2_mask = np.abs(de2) >= de2_min
            Dji = {Dj_basis.local_basis[i] for i, mask in enumerate(de2_mask) if mask}
            new_Dj |= Dji
            Hpsi_ref.append(Hpsi_i)
        if return_Hpsi_ref:
            return new_Dj, Hpsi_ref
        return new_Dj

    def expand(self, H, H_dict=None, de2_min=1e-10, dense_cutoff=1e3, slaterWeightMin=0):
        """
        Use the CIPSI method to expand the basis. Keep adding Slater determinants until the CIPSI energy is converged.
        """
        t0 = perf_counter()
        t_build_dict = 0
        t_build_mat = 0
        t_build_vec = 0
        t_build_state = 0
        t_eigen = 0
        t_Dj = 0
        t_add = 0
        psi_ref = None
        converge_count = 0
        de0_max = max(1e-6, -self.tau * np.log(1e-4))
        psi_ref = None
        t_tmp = perf_counter()
        H_dict = self.build_operator_dict(H, H_dict)
        t_build_dict += perf_counter() - t_tmp
        while converge_count < 1:
            t_tmp = perf_counter()
            H_mat = (
                self.build_sparse_matrix(H, op_dict=H_dict)
                if self.size > dense_cutoff
                else self.build_dense_matrix(H, op_dict=H_dict)
            )
            t_build_mat += perf_counter() - t_tmp
            t_tmp = perf_counter()
            if psi_ref is not None:
                v0 = self.build_vector(psi_ref).T
            else:
                v0 = None
            t_build_vec += perf_counter() - t_tmp
            t_tmp = perf_counter()
            e_ref, psi_ref_dense = eigensystem_new(
                H_mat,
                e_max=de0_max,
                k=len(psi_ref) + 1 if psi_ref is not None else 2,
                v0=v0,
                eigenValueTol=0,  # de2_min,
            )
            t_eigen += perf_counter() - t_tmp
            t_tmp = perf_counter()
            psi_ref = self.build_state(psi_ref_dense.T)
            t_build_state += perf_counter() - t_tmp
            t_tmp = perf_counter()
            new_Dj = self.determine_new_Dj(e_ref, psi_ref, H, H_dict, de2_min)
            t_Dj += perf_counter() - t_tmp
            old_size = self.size
            if self.spin_flip_dj:
                new_Dj = self._generate_spin_flipped_determinants(new_Dj)
            t_tmp = perf_counter()
            self.add_states(new_Dj)
            t_add += perf_counter() - t_tmp

            if old_size == self.size:
                converge_count += 1
            else:
                converge_count = 0

        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.")

        if self.size > self.truncation_threshold:
            H_sparse = self.build_sparse_matrix(H, op_dict=H_dict)
            e_ref, psi_ref = eigensystem_new(
                H_sparse,
                e_max=de0_max,
                k=sum(2 * (2 * l + 1) for l in self.ls),
            )
            self.truncate(self.build_state(psi_ref))
            if self.verbose:
                print(f"----->After truncation, the basis contains {self.size} elements.")
        t_tmp = perf_counter()
        H_dict = self.build_operator_dict(H, op_dict=H_dict)
        t_build_dict += perf_counter() - t_tmp
        return H_dict

    def expand_at(self, w, psi_ref, H, H_dict=None, de2_min=1e-3):
        old_size = self.size - 1
        while old_size != self.size:
            # Hpsi_ref = [[] for _ in psi_ref]
            # for i, psi in enumerate(psi_ref):
            #     Hpsi_ref[i] = applyOp(
            #         self.num_spin_orbitals,
            #         H,
            #         psi,
            #         restrictions=self.restrictions,
            #         opResult=H_dict,
            #     )
            new_Dj, Hpsi_ref = self.determine_new_Dj(
                [w] * len(psi_ref), psi_ref, H, H_dict, de2_min, return_Hpsi_ref=True
            )

            old_size = self.size
            # self.local_basis.clear()
            self.add_states(new_Dj)

            Hpsi_keys = set(state for psi in Hpsi_ref for state in psi)
            mask = list(self.contains(Hpsi_keys))
            psi_ref = [
                {state: psi[state] for state in itertools.compress(Hpsi_keys, mask) if state in psi} for psi in Hpsi_ref
            ]
            local_N2s = np.array([norm2(psi) for psi in psi_ref], dtype=float)
            N2s = np.empty_like(local_N2s)
            self.comm.Allreduce(local_N2s, N2s, op=MPI.SUM)
            psi_ref = [{state: psi[state] / np.sqrt(N2s[i]) for state in psi} for i, psi in enumerate(psi_ref)]

        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.")
        return self.build_operator_dict(H, op_dict=H_dict)

    def copy(self):
        new_basis = CIPSI_Basis(
            impurity_orbitals=self.impurity_orbitals,
            valence_baths=self.bath_states[0],
            conduction_baths=self.bath_states[1],
            initial_basis=self.local_basis,
            restrictions=self.restrictions,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            spin_flip_dj=self.spin_flip_dj,
            tau=self.tau,
            verbose=self.verbose,
        )
        return new_basis
