from math import ceil
from time import perf_counter
import sys
from typing import Optional, Union
from os import environ
from bisect import bisect_left

try:
    from collections.abc import Sequence, Iterable
except ModuleNotFoundError:
    from collections import Sequence, Iterable
import itertools
from heapq import merge
import numpy as np
import scipy as sp
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

from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, applyOp as applyOp_test


from impurityModel.ed.finite import applyOp_new as applyOp


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
            for delta_valence in range(delta_valence_occ[i] + 1):
                for delta_conduction in range(delta_conduction_occ[i] + 1):
                    delta_impurity = delta_valence - delta_conduction
                    if (
                        abs(delta_impurity) <= abs(delta_impurity_occ[i])
                        and nominal_impurity_occ[i] + delta_impurity <= total_impurity_orbitals[i]
                        and nominal_impurity_occ[i] + delta_impurity >= 0
                        and delta_valence <= len(valence_electron_indices)
                    ):
                        impurity_occupation = nominal_impurity_occ[i] + delta_impurity
                        valence_occupation = len(valence_electron_indices) - delta_valence
                        conduction_occupation = delta_conduction
                        impurity_configurations = itertools.combinations(impurity_electron_indices, impurity_occupation)
                        valence_configurations = itertools.combinations(valence_electron_indices, valence_occupation)
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
                        tuple(idx for subset in set_bits for part in subset for idx in part), num_spin_orbitals
                    )
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

    def _build_full_empty_bath_states(self, psis, occ_cutoff):

        valence_baths, conduction_baths = self.bath_states
        bath_indices = {
            i: [sorted(val_b + cond_b) for val_b, cond_b in zip(valence_baths[i], conduction_baths[i])]
            for i in valence_baths
        }
        _, bath_rhos = self.build_density_matrices(psis)
        bath_rhos = {i: [rho[0] for rho in br] for i, br in bath_rhos.items()}
        bath_occupations = {i: [np.diag(bath_rho) for bath_rho in brs] for i, brs in bath_rhos.items()}
        full_bath_states = {}
        empty_bath_states = {}
        for i in bath_occupations.keys():
            full_bath_states[i] = []
            empty_bath_states[i] = []
            for block_i, (block_orbs, block_occs) in enumerate(zip(bath_indices[i], bath_occupations[i])):
                filled_baths = [
                    orb
                    for orbs, occs in zip(
                        batched(block_orbs, len(self.impurity_orbitals[i][block_i])),
                        batched(block_occs, len(self.impurity_orbitals[i][block_i])),
                    )
                    for orb in orbs
                    if sum(occs) / len(orbs) >= 1 - occ_cutoff
                ]
                empty_baths = [
                    orb
                    for orbs, occs in zip(
                        batched(block_orbs, len(self.impurity_orbitals[i][block_i])),
                        batched(block_occs, len(self.impurity_orbitals[i][block_i])),
                    )
                    for orb in orbs
                    if sum(occs) / len(orbs) <= occ_cutoff
                ]
                full_bath_states[i].append(filled_baths[:-1])
                empty_bath_states[i].append(empty_baths[1:])
        return full_bath_states, empty_bath_states

    def build_excited_restrictions(self, psis, imp_change, val_change, con_change, occ_cutoff=1e-6):
        occ_restrict = imp_change is not None or val_change is not None or con_change is not None
        if not (occ_restrict or self.chain_restrict):
            return None

        valence_baths, conduction_baths = self.bath_states
        if self.chain_restrict:
            full_bath_states, empty_bath_states = self._build_full_empty_bath_states(psis, occ_cutoff)
        else:
            full_bath_states = {i: [] for i in self.impurity_orbitals.keys()}
            empty_bath_states = {i: [] for i in self.impurity_orbitals.keys()}

        new_valence_baths = {
            i: [
                [
                    orb
                    for orb in block_orbs
                    if not any(orb in full_block for full_block in full_bath_states[i])
                    and not any(orb in empty_block for empty_block in empty_bath_states[i])
                ]
                for block_orbs in valence_baths[i]
            ]
            for i in valence_baths.keys()
        }
        new_conduction_baths = {
            i: [
                [
                    orb
                    for orb in block_orbs
                    if not any(orb in full_block for full_block in full_bath_states[i])
                    and not any(orb in empty_block for empty_block in empty_bath_states[i])
                ]
                for block_orbs in conduction_baths[i]
            ]
            for i in conduction_baths.keys()
        }
        val_diff = {
            i: sum(len(orbs) for orbs in valence_baths[i]) - sum(len(orbs) for orbs in new_valence_baths[i])
            for i in valence_baths.keys()
        }
        restrictions = self.get_effective_restrictions()
        excited_restrictions = {}
        for i in self.impurity_orbitals.keys():
            impurity_indices = frozenset(ind for imp_ind in self.impurity_orbitals[i] for ind in imp_ind)
            if len(impurity_indices) > 0 and imp_change is not None:
                r_min_imp, r_max_imp = restrictions[impurity_indices]
                min_imp = max(r_min_imp - imp_change.get(i, (0, 0))[0], 0)
                max_imp = min(
                    r_max_imp + imp_change.get(i, (0, 0))[1], sum(len(orbs) for orbs in self.impurity_orbitals[i])
                )
                excited_restrictions[impurity_indices] = (min_imp, max_imp)

            min_val = 0
            max_val = 0
            new_valence_indices = frozenset()
            if val_change is not None:
                valence_indices = frozenset(ind for val_ind in valence_baths[i] for ind in val_ind)
                new_valence_indices = frozenset(ind for val_ind in new_valence_baths[i] for ind in val_ind)
                max_val = len(new_valence_indices)
                if len(valence_indices) > 0:
                    r_min_val, _ = restrictions[valence_indices]
                    min_val = max(r_min_val - val_diff[i] - val_change.get(i, (0, 0))[0], 0)

            max_cond = 0
            new_conduction_indices = frozenset()
            if con_change is not None:
                conduction_indices = frozenset(ind for con_ind in conduction_baths[i] for ind in con_ind)
                new_conduction_indices = frozenset(ind for con_ind in new_conduction_baths[i] for ind in con_ind)
                if len(conduction_indices) > 0:
                    _, r_max_cond = restrictions[conduction_indices]
                    max_cond = min(
                        r_max_cond + con_change.get(i, (0, 0))[1], sum(len(orbs) for orbs in new_conduction_baths[i])
                    )

            if val_change is not None or con_change is not None:
                new_fluctuating_indices = new_valence_indices.union(new_conduction_indices)
                if len(new_fluctuating_indices) == 0:
                    continue
                excited_restrictions[new_fluctuating_indices] = (min_val, max_val + max_cond)

            if self.collapse_chains:
                full_indices = frozenset(orb for full_indices in full_bath_states[i] for orb in full_indices)
                if len(full_indices) > 0:
                    excited_restrictions[full_indices] = (
                        len(full_indices) - 1,
                        len(full_indices),
                    )
                empty_indices = frozenset(orb for empty_indices in empty_bath_states[i] for orb in empty_indices)
                if len(empty_indices) > 0:
                    excited_restrictions[empty_indices] = (0, 1)
            else:
                for full_indices in full_bath_states.values():
                    for idx in full_indices:
                        if len(idx) == 0:
                            continue
                        excited_restrictions[frozenset(idx)] = (
                            len(idx) - 1,
                            len(idx),
                        )
                for empty_indices in empty_bath_states.values():
                    for idx in empty_indices:
                        if len(idx) == 0:
                            continue
                        excited_restrictions[frozenset(idx)] = (0, 1)
        return excited_restrictions

    def __init__(
        self,
        impurity_orbitals,
        bath_states,
        nominal_impurity_occ=None,
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
        t0 = perf_counter()
        assert (
            impurity_orbitals is not None
        ), "You need to supply the number of impurity orbitals in each set in impurity_orbitals"
        assert bath_states is not None, "You need to supply the number of bath states for each l quantum number"
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
                verbose=verbose,
            )
        t0 = perf_counter() - t0
        t0 = perf_counter()
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
        self.type = type(psr.int2bytes(0, self.num_spin_orbitals))
        self.n_bytes = int(ceil(self.num_spin_orbitals / 8))
        self.truncation_threshold = truncation_threshold
        self.is_distributed = comm is not None and comm.size > 1
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
        if not self.is_distributed:
            return psis

        res = [{} for _ in psis]
        states = sorted({state for psi in psis for state in psi})
        for r_offset in range(self.comm.size):
            send_to = (self.comm.rank + r_offset) % self.comm.size
            receive_from = (self.comm.rank + self.comm.size - r_offset) % self.comm.size

            if send_to > 0:
                lower_bound = self.state_bounds[send_to - 1]
            else:
                lower_bound = bytes(self.n_bytes)

            upper_bound = self.state_bounds[send_to]
            if upper_bound is None:
                upper_bound = bytes([0xFF] * (self.n_bytes + 1))

            send_list = [{} for _ in psis]
            if lower_bound is not None:
                for state in states:
                    if lower_bound <= state < upper_bound:
                        for send_n, psi_n in zip(send_list, psis):
                            send_n[state] = psi_n.get(state, 0)
            if send_to == self.comm.rank:
                received = send_list
            else:
                received = self.comm.sendrecv(send_list, dest=send_to, source=receive_from)
            for res_n, psi_n in zip(res, received):
                for state, amp in psi_n.items():
                    res_n[state] = amp + res_n.get(state, 0)
        return [ManyBodyState(p) for p in res]

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

    def expand(self, op, dense_cutoff=None, slaterWeightMin=0):
        if isinstance(op, dict):
            op = ManyBodyOperator(op)
        old_size = self.size - 1
        while old_size != self.size and self.size < self.truncation_threshold:
            new_states = set()
            local_states = set(self.local_basis)
            for i in range(5):
                new_local_states = set()
                for state in local_states:
                    res = applyOp_test(
                        op,
                        ManyBodyState({state: 1}),
                        cutoff=slaterWeightMin,
                        restrictions=self.restrictions,
                    )
                    new_local_states |= set(res.keys())
                if len(new_local_states) == len(local_states):
                    break
                local_states |= new_local_states
            new_states = local_states - set(self.local_basis)
            if self.spin_flip_dj:
                new_states = self._generate_spin_flipped_determinants(new_states)
            old_size = self.size
            self.add_states(new_states)
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
        res = [{} for _ in range(vs.shape[0])]
        if vs.shape[1] == self.size:
            # vs = vs[:, self.local_indices]
            for j, i in np.argwhere(np.abs(vs[:, self.local_indices]) > slaterWeightMin):
                res[j][self.local_basis[i]] = vs[j, i + self.offset]
            # for row, (i, state) in itertools.product(range(vs.shape[0]), zip(self.local_indices, self.local_basis)):
            #     psi = res[row]
            #     if abs(vs[row, i]) > slaterWeightMin:
            #         psi[state] = vs[row, i]
        elif vs.shape[1] == len(self.local_basis):
            for j, i in np.argwhere(np.abs(vs) > slaterWeightMin):
                res[j][self.local_basis[i]] = vs[j, i]
            # for row, (i, state) in itertools.product(range(vs.shape[0]), enumerate(self.local_basis)):
            #     psi = res[row]
            #     if abs(vs[row, i]) > slaterWeightMin:
            #         psi[state] = vs[row, i]
        else:
            raise RuntimeError(
                f"The dimensions of the input dense vector does not match a distributed, or full vector.\n{vs.shape} != ({vs.shape[0]}, {self.size}) || ({vs.shape[0]}, {len(self.local_basis)})"
            )
        return [ManyBodyState(p) for p in res]

    def build_operator_dict(self, op, slaterWeightMin=1e-16):
        """
        Express the operator, op, in the current basis. Do not expand the basis.
        Return a dict containing the results of applying op to the different basis states
        """
        if isinstance(op, dict):
            op = ManyBodyOperator(op)

        # for state in self.local_basis:
        _ = applyOp_test(
            op,
            ManyBodyState({state: 1 for state in self.local_basis}),
            cutoff=slaterWeightMin,
            restrictions=self.restrictions,
        )
        return op.memory()

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
            h = h_local.todense()
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

    def build_density_matrices(self, psis):
        local_psis = [{} for _ in psis]
        all_psis = self.comm.allgather([p.to_dict() for p in psis])
        for i, psi in enumerate(local_psis):
            for psis_r in all_psis:
                for state, amp in psis_r[i].items():
                    psi[state] = amp + psi.get(state, 0)
        rho_imps = {
            i: [
                np.array(
                    [
                        build_density_matrix(
                            sorted(block),
                            psi,
                            self.num_spin_orbitals,
                        )
                        for psi in local_psis
                    ]
                )
                for block in self.impurity_orbitals[i]
            ]
            for i in self.impurity_orbitals
        }
        valence, conduction = self.bath_states
        rho_baths = {
            i: [
                np.array(
                    [build_density_matrix(sorted(val_b + cond_b), psi, self.num_spin_orbitals) for psi in local_psis],
                    dtype=complex,
                )
                for val_b, cond_b in zip(valence[i], conduction[i])
            ]
            for i in valence
        }

        return rho_imps, rho_baths


class CIPSI_Basis(Basis):
    def __init__(self, H, impurity_orbitals, bath_states, nominal_impurity_occ=None, initial_basis=None, **kwargs):
        assert nominal_impurity_occ is not None or initial_basis is not None
        super().__init__(
            impurity_orbitals,
            bath_states,
            nominal_impurity_occ,
            initial_basis,
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
                eigenValueTol=0,
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
        local_Djs = [[state for state in hp if state not in self._index_dict] for hp in Hpsi_ref]
        overlaps = np.zeros((len(Hpsi_ref), max(len(Dj_row) for Dj_row in local_Djs)), dtype=complex)
        e_Dj = np.zeros((len(Hpsi_ref), max(len(Dj_row) for Dj_row in local_Djs)), dtype=float)
        # for i in range(len(Hpsi_ref)):
        for i, (Hpsi, Ds) in enumerate(zip(Hpsi_ref, local_Djs)):
            for j, (Dj, overlap) in enumerate((d, Hpsi[d]) for d in Ds):
                overlaps[i, j] = overlap
                HDj = applyOp_test(
                    H,
                    ManyBodyState({Dj: 1}),
                    cutoff=slaterWeightMin,
                    restrictions=self.restrictions,
                )
                # <Dj|H|Dj>
                e_Dj[i, j] = np.real(HDj[Dj])
        de = e_ref[:, None] - e_Dj
        de[np.abs(de) < np.finfo(float).eps] = np.finfo(float).eps

        # <Dj|H|Psi_ref>^2 / (E_ref - <Dj|H|Dj>)
        return local_Djs, np.square(np.abs(overlaps)) / de

    def determine_new_Dj(self, e_ref, psi_ref, H, de2_min, return_Hpsi_ref=False):
        if isinstance(H, dict):
            H = ManyBodyOperator(H)
        new_Dj = set()
        Hpsi_ref = [None for _ in psi_ref]
        for i, (e_i, psi_i) in enumerate(zip(e_ref, psi_ref)):
            Hpsi_ref[i] = applyOp_test(
                H,
                psi_i,
                restrictions=self.restrictions,
            )
        Hpsi_ref = self.redistribute_psis(Hpsi_ref)
        local_Djs, de2 = self._calc_de2(H, Hpsi_ref, e_ref)
        de2_mask = np.abs(de2) >= de2_min
        new_Dj = {Dj for i in range(len(Hpsi_ref)) for Dj, mask in zip(local_Djs[i], de2_mask[i]) if mask}
        if return_Hpsi_ref:
            return new_Dj, Hpsi_ref

        return new_Dj

    def expand(self, H, de2_min=1e-10, dense_cutoff=1e3, slaterWeightMin=0):
        """
        Use the CIPSI method to expand the basis. Keep adding Slater determinants until the CIPSI energy is converged.
        """
        psi_ref = None
        converge_count = 0
        de0_max = max(1e-6, -self.tau * np.log(1e-4))
        psi_ref = None

        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        i = 1
        while converge_count < 1:
            H_mat = self.build_sparse_matrix(H)
            e_ref, psi_ref_dense = eigensystem_new(
                H_mat,
                e_max=de0_max,
                k=len(psi_ref) if psi_ref is not None else 2,
                v0=self.build_vector(psi_ref).T if psi_ref is not None else None,
                eigenValueTol=de2_min,
                comm=self.comm,
                dense=self.size < dense_cutoff,
            )

            psi_ref = self.build_state(psi_ref_dense.T)
            if self.size > self.truncation_threshold:
                psi_ref = self.truncate(psi_ref)
                print(f"----->After truncation, the basis contains {self.size} elements.")

            new_Dj = self.determine_new_Dj(e_ref, psi_ref, H, de2_min)
            old_size = self.size
            if self.spin_flip_dj:
                new_Dj = self._generate_spin_flipped_determinants(new_Dj)
            self.add_states(new_Dj)
            psi_ref = self.redistribute_psis(psi_ref)

            if old_size == self.size:
                converge_count += 1
            else:
                converge_count = 0
            i += 1

        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.")

        return self.build_operator_dict(H)

    def expand_at(self, w, psi_ref, H, de2_min=1e-3):

        if isinstance(H, dict):
            H = ManyBodyOperator(H)
        old_size = self.size - 1
        while old_size != self.size:
            new_Dj, Hpsi_ref = self.determine_new_Dj([w] * len(psi_ref), psi_ref, H, de2_min, return_Hpsi_ref=True)

            old_size = self.size
            self.add_states(new_Dj)

            Hpsi_keys = {state for psi in Hpsi_ref for state in psi}
            mask = list(self.contains(Hpsi_keys))
            psi_ref = [
                {state: psi[state] for state in itertools.compress(Hpsi_keys, mask) if state in psi} for psi in Hpsi_ref
            ]
            N2s = np.array([norm2(psi) for psi in psi_ref], dtype=float)
            if self.is_distributed:
                self.comm.Allreduce(MPI.IN_PLACE, N2s, op=MPI.SUM)
            psi_ref = [{state: psi[state] / np.sqrt(N2s[i]) for state in psi} for i, psi in enumerate(psi_ref)]

        return self.build_operator_dict(H)

    def copy(self):
        new_basis = CIPSI_Basis(
            None,
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
        return new_basis
