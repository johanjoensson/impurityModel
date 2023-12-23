import numpy as np
import scipy as sp
from mpi4py import MPI
from math import ceil
from time import perf_counter
import sys
try:
    from petsc4py import PETSc
except:
    pass

try:
    from collections.abc import Sequence
except ModuleNotFoundError:
    from collections import Sequence

from impurityModel.ed import product_state_representation as psr
import itertools
from impurityModel.ed.finite import (
        c,
        a,
        c2i,
        eigensystem_new,
        norm2,
)
from impurityModel.ed.ac import (
        # create as c,
        # annihilate as a,
        applyOp,
)


def combine_sets(set_1, set_2, datatype):
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
            offset = self.comm.scan(local_len, op=MPI.SUM) - local_len
        return offset, local_len

    def _get_initial_basis(
        self,
        valence_baths,
        conduction_baths,
        delta_valence_occ,
        delta_conduction_occ,
        delta_impurity_occ,
        nominal_impurity_occ,
        verbose,
    ):
        total_baths = {l: valence_baths[l] + conduction_baths[l] for l in valence_baths}
        configurations = {}
        for l in valence_baths:
            if verbose:
                print(f"{l=}")
            valid_configurations = []
            for delta_valence in range(delta_valence_occ[l] + 1):
                for delta_conduction in range(delta_conduction_occ[l] + 1):
                    delta_impurity = delta_valence - delta_conduction
                    if (
                        abs(delta_impurity) <= delta_impurity_occ[l]
                        and nominal_impurity_occ[l] + delta_impurity <= 2 * (2 * l + 1)
                        and nominal_impurity_occ[l] + delta_impurity >= 0
                    ):
                        impurity_occupation = nominal_impurity_occ[l] + delta_impurity
                        valence_occupation = valence_baths[l] - delta_valence
                        conduction_occupation = delta_conduction
                        if verbose:
                            print("Partition occupations")
                            print(f"Impurity occupation:   {impurity_occupation:d}")
                            print(f"Valence onccupation:   {valence_occupation:d}")
                            print(f"Conduction occupation: {conduction_occupation:d}")
                        impurity_electron_indices = [
                            c2i(total_baths, (l, s, m)) for s in range(2) for m in range(-l, l + 1)
                        ]
                        impurity_configurations = itertools.combinations(impurity_electron_indices, impurity_occupation)
                        valence_electron_indices = [c2i(total_baths, (l, b)) for b in range(valence_baths[l])]
                        valence_configurations = itertools.combinations(valence_electron_indices, valence_occupation)
                        conduction_electron_indices = [
                            c2i(total_baths, (l, b)) for b in range(valence_baths[l], total_baths[l])
                        ]
                        conduction_configurations = itertools.combinations(
                            conduction_electron_indices, conduction_occupation
                        )
                        valid_configurations.append(
                            itertools.product(
                                impurity_configurations, valence_configurations, conduction_configurations
                            )
                        )
            configurations[l] = [
                imp + val + cond for configuration in valid_configurations for (imp, val, cond) in configuration
            ]
        num_spin_orbitals = sum(2 * (2 * l + 1) + total_baths[l] for l in total_baths)
        basis = []
        # Combine all valid configurations for all l-subconfigurations (ex. p-states and d-states)
        for system_configuration in itertools.product(*configurations.values()):
            basis.append(
                psr.tuple2bytes(tuple(sorted(itertools.chain.from_iterable(system_configuration))), num_spin_orbitals)
            )
        return basis, num_spin_orbitals

    def _get_restrictions(
        self,
        valence_baths,
        conduction_baths,
        delta_valence_occ,
        delta_conduction_occ,
        delta_impurity_occ,
        nominal_impurity_occ,
        verbose,
    ):
        restrictions = {}
        total_baths = {l: valence_baths[l] + conduction_baths[l] for l in valence_baths}
        for l in total_baths:
            impurity_indices = frozenset(c2i(total_baths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))
            restrictions[impurity_indices] = (
                max(nominal_impurity_occ[l] - delta_impurity_occ[l], 0),
                min(nominal_impurity_occ[l] + delta_impurity_occ[l] + 1, 2 * (2 * l + 1) + 1),
            )
            valence_indices = frozenset(c2i(total_baths, (l, b)) for b in range(valence_baths[l]))
            restrictions[valence_indices] = (max(valence_baths[l] - delta_valence_occ[l], 0), valence_baths[l] + 1)
            conduction_indices = frozenset(c2i(total_baths, (l, b)) for b in range(conduction_baths[l]))
            restrictions[conduction_indices] = (0, delta_conduction_occ[l] + 1)

            if verbose:
                print(f"l = {l}")
                print(f"|---Restrictions on the impurity orbitals = {restrictions[impurity_indices]}")
                print(f"|---Restrictions on the valence bath      = {restrictions[valence_indices]}")
                print(f"----Restrictions on the conduction bath   = {restrictions[conduction_indices]}")

        return restrictions

    def __init__(
        self,
        num_spin_orbitals=None,
        initial_basis=None,
        restrictions=None,
        valence_baths=None,
        conduction_baths=None,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        nominal_impurity_occ=None,
        truncation_threshold=np.inf,
        tau=0,
        comm=None,
        verbose=False,
    ):
        t0 = perf_counter()
        if initial_basis is not None:
            assert (
                num_spin_orbitals is not None
            ), "when supplying an initial basis, you also need to supply the num_spin_orbitals"
            assert nominal_impurity_occ is None
            assert valence_baths is None
            assert conduction_baths is None
            assert delta_valence_occ is None
            assert delta_conduction_occ is None
            assert delta_impurity_occ is None
            initial_basis = initial_basis
        else:
            assert initial_basis is None
            assert num_spin_orbitals is None
            assert restrictions is None
            initial_basis, num_spin_orbitals = self._get_initial_basis(
                valence_baths=valence_baths,
                conduction_baths=conduction_baths,
                delta_valence_occ=delta_valence_occ,
                delta_conduction_occ=delta_conduction_occ,
                delta_impurity_occ=delta_impurity_occ,
                nominal_impurity_occ=nominal_impurity_occ,
                verbose=verbose,
            )
            restrictions = self._get_restrictions(
                valence_baths=valence_baths,
                conduction_baths=conduction_baths,
                delta_valence_occ=delta_valence_occ,
                delta_conduction_occ=delta_conduction_occ,
                delta_impurity_occ=delta_impurity_occ,
                nominal_impurity_occ=nominal_impurity_occ,
                verbose=verbose,
            )
        t0 = perf_counter() - t0
        if verbose:
            print(f"===> T initial_basis : {t0}")
        t0 = perf_counter()
        self.verbose = verbose
        self.truncation_threshold = truncation_threshold
        self.comm = comm
        self.num_spin_orbitals = num_spin_orbitals
        self.local_basis = []
        self.restrictions = restrictions
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict = {}
        self.dtype = type(psr.int2bytes(0, self.num_spin_orbitals))
        self.n_bytes = int(ceil(self.num_spin_orbitals / 8))
        self.np_dtype = np.dtype(("B", self.n_bytes), align=False)

        self.index_bounds = [(None, None)] * comm.size if comm is not None else [(None, None)]
        self.state_bounds = [(None, None)] * comm.size if comm is not None else [(None, None)]

        self.is_distributed = comm is not None
        t0 = perf_counter() - t0
        if verbose:
            print(f"===> T init basic stuff : {t0}")
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
        if verbose:
            print(f"===> T init rng : {t0}")
        self.tau = tau

        t0 = perf_counter()
        self.add_states(initial_basis)
        t0 = perf_counter() - t0
        if verbose:
            print(f"===> T add_states : {t0}")

    def alltoall_states(self, send_list):
        # result = np.zeros((len(l) * self.n_bytes), dtype=np.byte)

        # self.comm.Alltoallv(
        #     (results, recv_counts * self.n_bytes, displacements * self.n_bytes, MPI.BYTE),
        #     (result, send_counts * self.n_bytes, send_offsets * self.n_bytes, MPI.BYTE),
        # )

        # result_new = np.zeros((len(l)), dtype=self.np_dtype)
        # for i in range(len(l)):
        #     if len(send_order) > 0:
        #         result_new[send_order[i]] = result[i * self.n_bytes: (i + 1) * self.n_bytes]
        #     else:
        #         result_new[0] = result[0: self.n_bytes]
        # result_new = [state.tobytes() for state in result_new]
        recv_counts = np.empty((self.comm.size), dtype=int)
        self.comm.Alltoall(np.array([len(l) for l in send_list], dtype=int), recv_counts)

        received_bytes = np.empty((sum(recv_counts) * self.n_bytes), dtype=np.byte)
        offsets = np.array([sum(recv_counts[:i]) for i in range(self.comm.size)], dtype=int)
        send_list_flat = np.array(
            [byte for state_list in send_list for byte_list in state_list for byte in byte_list], dtype=np.ubyte
        )

        send_counts = np.array([len(l) for l in send_list], dtype=int)
        send_offsets = np.array([sum(send_counts[:i]) for i in range(self.comm.size)], dtype=int)

        self.comm.Alltoallv(
            [send_list_flat, send_counts * self.n_bytes, send_offsets * self.n_bytes, MPI.BYTE],
            [received_bytes, recv_counts * self.n_bytes, offsets * self.n_bytes, MPI.BYTE],
        )

        received_states = [
            received_bytes[i * self.n_bytes : (i + 1) * self.n_bytes].tobytes() for i in range(sum(recv_counts))
        ]
        return received_states


    def _set_state_bounds(self, local_states):
        n_samples = min(100, max(len(local_states)//100, 2))
        state_bounds = None
        done = False
        while not done:
            if len(local_states) > 1:
                samples = list(itertools.chain(
                    [min(local_states)]
                    ,
                        self.rng.choice(
                            list(local_states), size=min(len(local_states) - 2, n_samples), replace=False
                        )
                    , [max(local_states)]
                ))
            else:
                samples = local_states

            samples_count = np.empty((self.comm.size), dtype=int)
            self.comm.Gather(np.array([len(samples)], dtype=int), samples_count, root=0)

            all_samples_bytes = None
            offsets = 0
            if self.comm.rank == 0:
                all_samples_bytes = np.empty((sum(samples_count) * self.n_bytes), dtype=np.byte)
                offsets = np.array([np.sum(samples_count[:i]) for i in range(self.comm.size)], dtype=int)

            self.comm.Gatherv(
                np.array([byte for state in samples for byte in state], dtype=np.byte),
                [all_samples_bytes, samples_count * self.n_bytes, offsets * self.n_bytes, MPI.BYTE],
                root=0,
            )

            if self.comm.rank == 0:
                all_states = sorted({
                    i.tobytes() for i in np.split(all_samples_bytes, np.sum(samples_count))
                    # all_samples_bytes[i * self.n_bytes: (i + 1) * self.n_bytes].tobytes()
                    # for i in range(sum(samples_count))
                })
                done = True

                sizes = np.array([len(all_states) // self.comm.size] * self.comm.size, dtype=int)
                sizes[: len(all_states) % self.comm.size] += 1

                bounds = [sum(sizes[:i]) for i in range(self.comm.size)]
                print(f"{len(all_states)=}")
                print(f"{bounds=}")
                state_bounds = [
                    all_states[bound] if bound < len(all_states) else all_states[-1] for bound in bounds
                ]
                state_bounds_bytes = np.array([byte for state in state_bounds for byte in state], dtype=np.byte)
            else:
                state_bounds_bytes = np.empty((self.comm.size * self.n_bytes), dtype=np.byte)
                state_bounds = None
            done = self.comm.bcast(done, root=0)

        self.comm.Bcast(state_bounds_bytes, root=0)
        state_bounds = [
            state_bounds_bytes[i * self.n_bytes: (i + 1) * self.n_bytes].tobytes() for i in range(self.comm.size)
            # i.tobytes() for i in np.split(state_bounds_bytes, self.comm.size)
        ]
        return state_bounds

    def add_states(self, new_states: list, distributed_sort=True):
        """
        Extend the current basis by adding the new_states to it.
        """
        if not self.is_distributed:
            self.local_basis = sorted(set(self.local_basis + new_states))
            self.size = len(self.local_basis)
            self.offset = 0
            self.local_indices = range(0, len(self.local_basis))
            self._index_dict = {state: i for i, state in enumerate(self.local_basis)}
            self.local_index_bounds = (0, len(self.local_basis))
            if len(self.local_basis) > 0:
                self.local_state_bounds = (self.local_basis[0], self.local_basis[-1])
            else:
                self.local_state_bounds = (None, None)
            return

        if not distributed_sort:
            old_basis = self.comm.reduce(set(self.local_basis), op=combine_sets_op, root=0)
            new_states = self.comm.reduce(set(new_states), op=combine_sets_op, root=0)
            if self.comm.rank == 0:
                new_basis = sorted(old_basis | new_states)
                send_basis = [[] for _ in range(self.comm.size)]
                start = 0
                for r in range(self.comm.size):
                    stop = start + len(new_basis) // self.comm.size
                    if r < len(new_basis) % self.comm.size:
                        stop += 1
                    send_basis[r] = new_basis[start:stop]
                    start = stop
            else:
                send_basis = None
            self.local_basis = self.comm.scatter(send_basis, root=0)
        else:
            t0 = perf_counter()
            local_states = set(itertools.chain(self.local_basis, new_states))
            # local_states = sorted(set(itertools.chain(self.local_basis, new_states)))
            t0 = perf_counter() - t0
            if self.verbose:
                print(f"=======> T sorting local states : {t0}")
            t0 = perf_counter()
            state_bounds = self._set_state_bounds(local_states)
            t0 = perf_counter() - t0
            if self.verbose:
                print(f"=======> T set_state_bounds : {t0}")

            t0 = perf_counter()
            last_rank = self.comm.size - 1
            for r in range(self.comm.size - 1):
                if state_bounds[r] == state_bounds[r + 1]:
                    last_rank = r
                    break
            send_list = [[] for _ in range(self.comm.size)]
            for state in local_states:
                for r in range(last_rank):
                    if state >= state_bounds[r] and state < state_bounds[r + 1]:
                        send_list[r].append(state)
                        break
                if state >= state_bounds[last_rank]:
                    send_list[last_rank].append(state)
            t0 = perf_counter() - t0
            if self.verbose:
                print(f"=======> T setting up send_list : {t0}")

            t0 = perf_counter()
            received_states = self.comm.alltoall(send_list)
            received_states = {state for rank_states in received_states for state in rank_states}

            # recv_counts = np.empty((self.comm.size), dtype=int)
            # self.comm.Alltoall(np.array([len(l) for l in send_list], dtype=int), recv_counts)

            # received_bytes = np.empty((sum(recv_counts) * self.n_bytes), dtype=np.byte)
            # offsets = np.array([sum(recv_counts[:i]) for i in range(self.comm.size)], dtype=int)
            # send_list_flat = np.array([byte for state in np.ravel(send_list) for byte in state])
            # # send_list_flat = np.array(
            # #     [byte for state_list in send_list for byte_list in state_list for byte in byte_list], dtype=np.byte
            # # )

            # send_counts = np.array([len(l) for l in send_list], dtype=int)
            # send_offsets = np.array([sum(send_counts[:i]) for i in range(self.comm.size)], dtype=int)

            # self.comm.Alltoallv(
            #     [send_list_flat, send_counts * self.n_bytes, send_offsets * self.n_bytes, MPI.BYTE],
            #     [received_bytes, recv_counts * self.n_bytes, offsets * self.n_bytes, MPI.BYTE],
            # )
            t0 = perf_counter() - t0
            if self.verbose:
                print(f"=======> T distributing new states : {t0}")

            t0 = perf_counter()
            # received_states = {
            #     received_bytes[i * self.n_bytes : (i + 1) * self.n_bytes].tobytes() for i in range(sum(recv_counts))
            # }
            t0 = perf_counter() - t0
            if self.verbose:
                print(f"=======> T bytes to states : {t0}")

            t0 = perf_counter()
            self.local_basis = sorted(received_states)
            t0 = perf_counter() - t0
            if self.verbose:
                print(f"=======> T sort received_states : {t0}")
            ########################################################################
            # The local lengths are not balanced! The basis is sorted, but not
            # evenly distributed among the ranks.
            ########################################################################

        t0 = perf_counter()
        local_length = len(self.local_basis)
        self.size = self.comm.allreduce(local_length, op=MPI.SUM)
        self.offset = self.comm.scan(local_length, op=MPI.SUM) - local_length
        self.local_indices = range(self.offset, self.offset + local_length)
        self._index_dict = {state: self.offset + i for i, state in enumerate(self.local_basis)}
        local_index_bounds = (self.offset, self.offset + local_length)
        if len(self.local_basis) > 0:
            local_state_bounds = (self.local_basis[0], self.local_basis[-1])
        else:
            local_state_bounds = (None, None)
        self.index_bounds = self.comm.allgather(local_index_bounds)
        self.state_bounds = self.comm.allgather(local_state_bounds)
        t0 = perf_counter() - t0
        if self.verbose:
            print(f"=======> T set bounds and stuff : {t0}")

    def expand(self, op, op_dict={}, dense_cutoff=None, slaterWeightMin=1e-20):
        done = False
        if self.comm is None:
            new_basis = set()
            new_states = set(op_dict.keys()) | set(self.local_basis)
            while len(new_states) > 0:
                states_to_check = new_states
                new_states = set()
                for state in states_to_check:
                    if state in op_dict:
                        res = op_dict[state]
                    else:
                        res = applyOp(
                            self.num_spin_orbitals,
                            op,
                            {state: 1},
                            restrictions=self.restrictions,
                            slaterWeightMin=slaterWeightMin,
                        )
                        if len(res) != 0:
                            op_dict[state] = res
                    new_states |= set(res.keys()) - new_basis

                new_basis += sorted(new_states)
                self.local_basis = list(sorted(new_basis))
        else:
            while not done:
                local_states = []

                new_local_states = []
                for state in set(self.local_basis + local_states):
                    res = applyOp(
                        self.num_spin_orbitals,
                        op,
                        {state: 1},
                        restrictions=self.restrictions,
                        slaterWeightMin=slaterWeightMin,
                    )
                    op_dict[state] = res
                    new_local_states.extend(res.keys())
                local_states.extend(new_local_states)

                old_size = self.size
                self.add_states(local_states)
                done = old_size == self.size
        if self.verbose:
            print(f"Expanded basis contains {self.size} elements")
        return self.build_operator_dict(op)

    def _getitem_sequence(self, l):
        if self.comm is None:
            return [self.local_basis[i] for i in l]

        l = np.array([i if i >= 0 else self.size + i for i in l], dtype=int)

        send_list = [np.array([], dtype=int) for _ in range(self.comm.size)]
        send_to_ranks = []
        for i in l:
            send_to_ranks.append(self.comm.size)
            for r in range(self.comm.size):
                if i >= self.index_bounds[r][0] and i < self.index_bounds[r][1]:
                    send_list[r] = np.append(send_list[r], [i])
                    send_to_ranks[-1] = r
        send_order = np.argsort(send_to_ranks, kind="stable")
        recv_counts = np.empty((self.comm.size), dtype=int)
        queries = None
        displacements = None

        self.comm.Alltoall(np.array([len(l) for l in send_list], dtype=int), recv_counts)

        queries = np.empty((sum(recv_counts)), dtype=int)
        displacements = np.array([sum(recv_counts[:p]) for p in range(self.comm.size)])
        send_list_flat = np.array([i for l in send_list for i in l], dtype=int)
        send_counts = np.array([len(l) for l in send_list], dtype=int)
        send_offsets = np.array([sum(send_counts[:r]) for r in range(self.comm.size)], dtype=int)

        self.comm.Alltoallv(
            (send_list_flat, send_counts, send_offsets, MPI.UINT64_T),
            (queries, recv_counts, displacements, MPI.UINT64_T),
        )

        results = np.empty((sum(recv_counts) * self.n_bytes), dtype=np.byte)
        for i, query in enumerate(queries):
            if query >= self.offset and query < self.offset + len(self.local_basis):
                results[i * self.n_bytes: (i + 1) * self.n_bytes] = np.frombuffer(
                    self.local_basis[query - self.offset], dtype="B"
                )
        result = np.zeros((len(l) * self.n_bytes), dtype=np.byte)

        self.comm.Alltoallv(
            (results, recv_counts * self.n_bytes, displacements * self.n_bytes, MPI.BYTE),
            (result, send_counts * self.n_bytes, send_offsets * self.n_bytes, MPI.BYTE),
        )

        result_new = np.zeros((len(l)), dtype=self.np_dtype)
        for i in range(len(l)):
            if len(send_order) > 0:
                result_new[send_order[i]] = result[i * self.n_bytes: (i + 1) * self.n_bytes]
            else:
                result_new[0] = result[0: self.n_bytes]
        for state in result_new:
            assert len(state) == self.n_bytes

        return [state.tobytes() for state in result_new]

    def index(self, val):
        if isinstance(val, self.dtype):
            res = self._index_sequence([val])[0]
            if res == self.size:
                raise ValueError(f"Could not find {val} in basis!")
        elif isinstance(val, Sequence):
            res = self._index_sequence(val)
            for i, v in enumerate(res):
                if v == self.size:
                    raise ValueError(f"Could not find {val[i]} in basis!")
        else:
            raise TypeError(f"Invalid query type {type(val)}! Valid types are {self.dtype} and sequences thereof.")
        return res

    def _index_sequence(self, s):
        if self.comm is None:
            results = [self._index_dict[val] if val in self._index_dict else self.size for val in s]
            return results

        send_list = [np.empty((0), dtype=self.np_dtype) for _ in range(self.comm.size)]
        send_to_ranks = []
        for i, val in enumerate(s):
            send_to_ranks.append(self.comm.size)
            for r in range(self.comm.size):
                if (
                    self.state_bounds[r][0] is not None
                    and val >= self.state_bounds[r][0]
                    and val <= self.state_bounds[r][1]
                ):
                    send_list[r] = np.append(send_list[r], [np.frombuffer(val, dtype="B")], axis=0)
                    send_to_ranks[-1] = r
                    break

        send_order = np.argsort(send_to_ranks, kind="stable")
        recv_counts = np.empty((self.comm.size), dtype=int)
        send_counts = np.array([len(send_list[r]) for r in range(self.comm.size)], dtype=int)
        send_displacements = np.array([sum(send_counts[:i]) for i in range(self.comm.size)], dtype=int)
        send_list_flat_bytes = np.array(
            [byte for states in send_list for state in states for byte in state], dtype=np.byte
        )

        self.comm.Alltoall(np.array([len(send_list[r]) for r in range(self.comm.size)], dtype=int), recv_counts)

        queries = np.empty((sum(recv_counts) * self.n_bytes), dtype=np.byte)
        displacements = np.array([sum(recv_counts[:p]) for p in range(self.comm.size)], dtype=int)

        self.comm.Alltoallv(
            [send_list_flat_bytes, send_counts * self.n_bytes, send_displacements * self.n_bytes, MPI.BYTE],
            [queries, recv_counts * self.n_bytes, displacements * self.n_bytes, MPI.BYTE],
        )

        results = np.empty((sum(recv_counts)), dtype=int)
        results[:] = self.size
        for i in range(sum(recv_counts)):
            query = queries[i * self.n_bytes: (i + 1) * self.n_bytes].tobytes()
            if query in self._index_dict:
                results[i] = self._index_dict[query]
        result = np.empty((len(s)), dtype=int)
        result[:] = self.size

        self.comm.Alltoallv(
            [results, recv_counts, displacements, MPI.UINT64_T], [result, send_counts, send_displacements, MPI.UINT64_T]
        )

        result[send_order] = result.copy()
        return result.tolist()

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start
            if start is None:
                start = 0
            elif start < 0:
                start = self.size + start
            stop = key.stop
            if stop is None:
                stop = self.size
            elif stop < 0:
                stop = self.size + stop
            step = key.step
            if step is None and start < stop:
                step = 1
            elif step is None:
                step = -1
            query = [i for i in range(start, stop, step)]
            result = self._getitem_sequence(query)
            for i, res in enumerate(result):
                if res == psr.int2bytes(0, self.num_spin_orbitals):
                    raise IndexError(f"Could not find index {query[i]} in basis with size {self.size}!")
        elif isinstance(key, Sequence):
            result = self._getitem_sequence(key)
            for i, res in enumerate(result):
                if res == psr.int2bytes(0, self.num_spin_orbitals):
                    raise IndexError(f"Could not find index {key[i]} in basis with size {self.size}!")
        elif isinstance(key, int):
            result = self._getitem_sequence([key])[0]
            if result == psr.int2bytes(0, self.num_spin_orbitals):
                raise IndexError(f"Could not find index {key} in basis with size {self.size}!")
        else:
            raise TypeError(f"Invalid index type {type(key)}. Valid types are slice, Sequence and int")
        return result

    def __len__(self):
        return self.size

    def __contains__(self, item):
        if self.comm is None:
            return item in self.local_basis
        index = self._index_sequence([item])[0]
        return index != self.size

    def _contains_sequence(self, items):
        if self.comm is None:
            return [item in self.local_indices for item in items]
        indices = self._index_sequence(items)
        return [index != self.size for index in indices]

    def contains(self, item):
        if isinstance(item, Sequence):
            return self._contains_sequence(item)
        elif isinstance(item, self.dtype):
            return self._contains_sequence([item])[0]

    def __iter__(self):
        for i in range(self.size):
            yield self[i]

    def copy(self):
        return Basis(
            initial_basis=self.local_basis,
            num_spin_orbitals=self.num_spin_orbitals,
            restrictions=self.restrictions,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            verbose=self.verbose,
        )

    def build_vector(self, psis, dtype=complex):
        v_local = np.zeros((self.size, len(psis)), dtype=dtype)
        v = np.empty_like(v_local)
        for col, psi in enumerate(psis):
            if len(psi) == 0:
                continue
            for state in psi:
                if state in self._index_dict and abs(psi[state]) > 0:
                    v_local[self._index_dict[state], col] = psi[state]

        self.comm.Allreduce(v_local, v, op=MPI.SUM)
        vs = self.comm.allgather(v)
        for v_i in vs:
            assert np.all(v == v_i)
        # if len(psis) == 1:
        #     v = v.reshape((self.size))
        return v

    def build_state(self, v, distribute=False):
        assert v.shape[0] == self.size
        if len(v.shape) == 1:
            v = v.reshape((v.shape[0], 1))
        ncols = v.shape[1]
        local_psis = []
        for col in range(ncols):
            state = {}
            for i in self.local_indices:
                if abs(v[i, col]) > 0:
                    state[self.local_basis[i - self.offset]] = v[i, col]
            local_psis.append(state)
        if distribute:
            return local_psis
        psis = self.comm.allgather(local_psis)
        res = []
        for col in range(ncols):
            state = {}
            for local_psi in psis:
                state.update(local_psi[col])
            res.append(state)
        self.comm.barrier()
        return res

    def build_operator_dict(self, op, op_dict=None, distributed=True, slaterWeightMin=1e-20):
        """
        Express the operator, op, in the current basis. Do not expand the basis.
        Return a dict containing the results of applying op to the different basis states
        """
        if op_dict is None:
            op_dict = {}

        for state in self.local_basis:
            if state not in op_dict:
                _ = applyOp(
                    self.num_spin_orbitals,
                    op,
                    {state: 1},
                    restrictions=self.restrictions,
                    slaterWeightMin=slaterWeightMin,
                    opResult=op_dict,
                )

        all_row_states = [state for column in op_dict for state in op_dict[column]]
        # row_indices = np.array(self._index_sequence(all_row_states), dtype=int)
        # in_basis_mask = row_indices != self.size
        # state_in_basis = {state: in_basis for state, in_basis in zip(all_row_states, in_basis_mask)}
        state_in_basis = {state: in_basis for state, in_basis in zip(all_row_states, self.contains(all_row_states))}
        for column in list(op_dict.keys()):
            if column not in self.local_basis:
                op_dict.pop(column, None)
                continue
            for row in list(op_dict[column].keys()):
                if not state_in_basis[row]:
                    op_dict[column].pop(row, None)

        return op_dict

    def build_dense_matrix(self, op, op_dict=None, distribute=True):
        """
        Get the operator as a dense matrix in the current basis.
        by default the dense matrix is distributed to all ranks.
        """
        h_local = self.build_sparse_matrix(op, op_dict)
        local_rows, local_columns = h_local.nonzero()
        local_data = np.array([h_local[row, col] for row, col in zip(local_rows, local_columns)], dtype=complex)
        data = None
        rows = None
        columns = None
        recv_counts = None
        offsets = None
        if self.comm.rank == 0:
            recv_counts = np.empty((self.comm.size), dtype=int)
        self.comm.Gather(np.array([len(local_data)]), recv_counts, root=0)

        if self.comm.rank == 0:
            offsets = [sum(recv_counts[:i]) for i in range(self.comm.size)]
            data = np.empty((sum(recv_counts)), dtype=h_local.dtype)
            rows = np.empty((sum(recv_counts)), dtype=local_rows.dtype)
            columns = np.empty((sum(recv_counts)), dtype=local_columns.dtype)
        self.comm.Gatherv(local_data, [data, recv_counts, offsets, MPI.DOUBLE_COMPLEX], root=0)
        self.comm.Gatherv(local_rows, [rows, recv_counts, offsets, MPI.INT], root=0)
        self.comm.Gatherv(local_columns, [columns, recv_counts, offsets, MPI.INT], root=0)
        if self.comm.rank == 0:
            h = sp.sparse.coo_matrix((data, (rows, columns)), shape=(h_local.shape[0], h_local.shape[0]))
            h = h.todense()
        else:
            h = None
        if distribute:
            h = self.comm.bcast(h, root=0)
        return h

    def build_sparse_matrix(self, op, op_dict=None):
        """
        Get the operator as a sparse matrix in the current basis.
        The sparse matrix is distributed over all ranks.
        """
        if "PETSc" in sys.modules:
            return self._build_PETSc_matrix(op, op_dict)

        expanded_op_dict = self.build_operator_dict(op, op_dict)

        columns = []
        rows = []
        values = []
        for column in expanded_op_dict:
            for row in expanded_op_dict[column]:
                columns.append(column)
                rows.append(row)
                values.append(expanded_op_dict[column][row])
        columns = self.index(columns)
        rows = self.index(rows)
        shape = (self.size, self.size)
        return sp.sparse.csr_matrix((values, (rows, columns)), shape=shape, dtype=complex)

    def _build_PETSc_matrix(self, op, op_dict=None):
        """
        Get the operator as a sparse matrix in the current basis.
        The sparse matrix is distributed over all ranks.
        """

        M = PETSc.Mat().create(comm=self.comm)
        M.setSizes([self.size, self.size])

        expanded_op_dict = self.build_operator_dict(op, op_dict)
        columns = []
        rows = []
        values = []
        for column in expanded_op_dict:
            for row in expanded_op_dict[column]:
                columns.append(column)
                rows.append(row)
                values.append(expanded_op_dict[column][row])
        columns = self.index(columns)
        rows = self.index(rows)
        M.setUp()
        for i, j, val in zip(rows, columns, values):
            M[i, j] = val
        M.assemble()
        return M

class CIPSI_Basis(Basis):
    def __init__(
        self,
        valence_baths=None,
        conduction_baths=None,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        nominal_impurity_occ=None,
        initial_basis=None,
        restrictions=None,
        num_spin_orbitals=None,
        truncation_threshold=np.inf,
        verbose=False,
        H=None,
        tau=0,
        comm=None,
    ):
        if initial_basis is None:
            assert valence_baths is not None
            assert conduction_baths is not None
            assert nominal_impurity_occ is not None
            initial_basis, num_spin_orbitals = self._get_initial_basis(
                valence_baths,
                conduction_baths,
                delta_valence_occ,
                delta_conduction_occ,
                delta_impurity_occ,
                nominal_impurity_occ,
                verbose,
            )
            restrictions = self._get_restrictions(
                valence_baths=valence_baths,
                conduction_baths=conduction_baths,
                delta_valence_occ=valence_baths,
                delta_conduction_occ=conduction_baths,
                delta_impurity_occ={l: max(abs(2 * (2 * l + 1) - N0), N0) for l, N0 in nominal_impurity_occ.items()},
                nominal_impurity_occ=nominal_impurity_occ,
                verbose=verbose,
            )
        else:
            assert num_spin_orbitals is not None
        Basis.__init__(
            self,
            initial_basis=initial_basis,
            num_spin_orbitals=num_spin_orbitals,
            restrictions=restrictions,
            truncation_threshold=truncation_threshold,
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
                basis=self,
                e_max=1e-12,
                k=min(1, self.size - 1),
                dk=min(10, self.size - 1),
                eigenValueTol=0,
                slaterWeightMin=np.finfo(float).eps ** 2,
                verbose=self.verbose,
            )
            self.truncate(psi_ref)

    def truncate(self, psis):
        basis_states = list(set(state for psi in psis for state in psi))
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
            new_basis.append(basis_states[sort_order[i]])

        self.local_basis = []
        self.add_states(new_basis)

    def _calc_de_2(self, Djs, H, H_dict, Hpsi_ref, e_ref, slaterWeightMin=1e-20):
        """
        calculate second variational energy contribution of the Slater determinants in states.
        """

        overlap = np.empty((len(Djs)), dtype=complex)
        e_state = np.empty((len(Djs)), dtype=float)
        for j, Dj in enumerate(Djs):
            # <Dj|H|Psi_ref>
            overlap[j] = Hpsi_ref[Dj] if Dj in Hpsi_ref else 0
            HDj = applyOp(
                self.num_spin_orbitals,
                H,
                {Dj: 1},
                restrictions=self.restrictions,
                slaterWeightMin=slaterWeightMin,
                opResult=H_dict,
            )
            # <Dj|H|Dj>
            e_state[j] = np.real(HDj[Dj] if Dj in HDj else 0)
        de = e_ref - e_state
        #  inf_mask = np.abs(de) < 1e-15
        #  overlap[inf_mask] = 1e15
        #  de[inf_mask] = 1

        return np.abs(overlap) ** 2 / de

    def _generate_spin_flipped_determinants(self, determinants):
        # X_lm = c(l, 0, m)*a(l, 1, m) + c(l, 1, m)*a(l, 0, m)
        # X = 1
        # for m in -l to l (inclusive)
        #   X = (1 + X_lm)*X
        # apply X to det
        # spin_flipped_dets =
        determinants = set(determinants)
        spin_flipped = set()
        for dn_state in range(self.num_spin_orbitals // 2):
            up_state = dn_state + self.num_spin_orbitals // 2
            for det in determinants:
                dn_to_up = c(self.num_spin_orbitals, up_state, a(self.num_spin_orbitals, dn_state, {det: 1}))
                up_to_dn = c(self.num_spin_orbitals, dn_state, a(self.num_spin_orbitals, up_state, {det: 1}))
                spin_flipped |= set(dn_to_up.keys()) | set(up_to_dn.keys())
            determinants |= spin_flipped
        return set(determinants)

    def expand(self, H, H_dict={}, e_conv=1e-15, dense_cutoff=1e3, slaterWeightMin=1e-20):
        """
        Use the CIPSI method to expand the basis. Keep adding Slater determinants until the CIPSI energy is converged.
        """
        psi_ref = None
        e_0_prev = np.inf
        e_0 = 0
        de_2_min = 1e-15
        converge_count = 0
        de0_max = -self.tau * np.log(1e-4)
        while converge_count < 1:
            t0 = perf_counter()
            H_sparse = self.build_sparse_matrix(H)
            t0 = perf_counter() - t0
            t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
            if self.verbose:
                print(f"->Time to build sparse H: {t0/self.comm.size:.3f} seconds")

            t0 = perf_counter()
            if psi_ref is not None:
                v0 = self.build_vector(psi_ref)
                # v0 = np.zeros((self.size, len(psi_ref)), dtype=complex)
                # v0_states = [list(pr.keys()) for pr in psi_ref]
                # v0_indices = [self.index(psi) for psi in v0_states]
                # for j in range(len(v0_indices)):
                #     for i, state in zip(v0_indices[j], v0_states[j]):
                #         v0[i, j] = psi_ref[j][state]
                # self.comm.Allreduce(v0.copy(), v0)
            else:
                v0 = None
            e_ref, psi_ref = eigensystem_new(
                H_sparse,
                basis=self,
                e_max=de0_max,
                k=1 if psi_ref is None else max(1, len(psi_ref)),
                dk=10,
                v0=v0,
                eigenValueTol=de_2_min if de_2_min > 1e-12 else 0,
                slaterWeightMin=slaterWeightMin,
                dense_cutoff=dense_cutoff,
                verbose=self.verbose,
                distribute_eigenvectors=False,
            )
            e_0_prev = e_0
            e_0 = np.min(e_ref)
            t0 = perf_counter() - t0
            t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
            if self.verbose:
                print(f"->Time to get psi_ref: {t0/self.comm.size:.3f} seconds")
                print("->Loop over eigenstates")

            new_Dj = set()
            de_2_corr = {}
            for e_i, psi_i in zip(e_ref, psi_ref):
                t0_loop = perf_counter()
                Hpsi_i = applyOp(
                    self.num_spin_orbitals,
                    H,
                    psi_i,
                    restrictions=self.restrictions,
                    slaterWeightMin=slaterWeightMin,
                    opResult=H_dict,
                )
                t0 = perf_counter() - t0_loop
                t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
                if self.verbose:
                    print(f"--->Time to calculate Hpsi_i: {t0/self.comm.size:.3f} seconds")

                t0 = perf_counter()
                # coupled_Dj = list(Hpsi_i.keys())
                # basis_mask = np.logical_not(self.contains(coupled_Dj))

                Dj_basis = Basis(
                    initial_basis=set(Hpsi_i.keys()),
                    # initial_basis=[Dj for j, Dj in enumerate(coupled_Dj) if basis_mask[j]],
                    num_spin_orbitals=self.num_spin_orbitals,
                    restrictions=self.restrictions,
                    comm=self.comm,
                )
                t0 = perf_counter() - t0
                t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
                if self.verbose:
                    print(f"--->Time to setup distributed Djs: {t0/self.comm.size:.3f} seconds")

                t0 = perf_counter()
                send_list = [[] for _ in range(self.comm.size)]
                for r in range(self.comm.size):
                    if Dj_basis.state_bounds[r][0] is None:
                        continue
                    send_list[r] = {
                        state: val
                        for state, val in Hpsi_i.items()
                        if state >= Dj_basis.state_bounds[r][0] and state <= Dj_basis.state_bounds[r][1]
                    }
                received_Hpsi_i = self.comm.alltoall(send_list)
                Hpsi_i = {}
                for received_dict in received_Hpsi_i:
                    for key in received_dict:
                        if key in Hpsi_i:
                            Hpsi_i[key] += received_dict[key]
                        else:
                            Hpsi_i[key] = received_dict[key]
                t0 = perf_counter() - t0
                t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
                if self.verbose:
                    print(f"--->Time to distribute H|psi_i>: {t0/self.comm.size:.3f} seconds")

                t0 = perf_counter()
                de_2 = self._calc_de_2(Dj_basis.local_basis, H, H_dict, Hpsi_i, e_i)

                de_2_mask = np.abs(de_2) >= de_2_min

                t0 = perf_counter() - t0
                t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
                if self.verbose:
                    print(f"--->Time to calculate de_2: {t0/self.comm.size:.3f} seconds")

                t0 = perf_counter()
                Dji = {Dj_basis.local_basis[i] for i, mask in enumerate(de_2_mask) if mask}
                de_2_corr.update({Dj_basis.local_basis[i]: max(de_2[i], de_2_corr.get(Dj_basis.local_basis[i], 0)) for i, mask in enumerate(de_2_mask) if mask})
                Dji = self._generate_spin_flipped_determinants(Dji)
                t0 = perf_counter() - t0
                t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
                if self.verbose:
                    print(f"--->Time to generate spin flipped determinants: {t0/self.comm.size:.3f} seconds")
                new_Dj |= Dji - set(self.local_basis)

                # t0 = perf_counter()
                # self.add_states(new_Dj)

                # t0 = perf_counter() - t0
                # t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
                # if self.verbose:
                #     print(f"----->Time to add new states: {t0/self.comm.size:.3f} seconds")

                t0 = perf_counter() - t0_loop
                t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
                if self.verbose:
                    print(f"->One loop took a total of: {t0/self.comm.size:.3f} seconds", flush=True)

            t0 = perf_counter()
            old_size = self.size
            self.add_states(new_Dj)
            t0 = perf_counter() - t0
            t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
            if self.verbose:
                print(f"--->Time to add new states: {t0/self.comm.size:.3f} seconds")

            if abs(e_0 - e_0_prev) <= e_conv or old_size == self.size:
                converge_count += 1
            else:
                converge_count = 0
            de_2_corrs = self.comm.gather(de_2_corr, root=0)
            if self.verbose:
                de_2_corr = {}
                for de_2s in de_2_corrs:
                    de_2_corr.update(de_2s)
                print(
                    f"-----> N = {self.size: 7,d}, log(de_2_min) = {np.log10(de_2_min): 5.1f}, log(|de_2|) = {np.log10(np.abs(np.sum(list(de_2_corr.values()))))}, log(|de_0|) = {np.log10(abs(e_0 - e_0_prev))}, {converge_count=}",
                )

        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.")

        if self.size > self.truncation_threshold:
            H_sparse = self.build_sparse_matrix(H)
            e_ref, psi_ref = eigensystem_new(
                H_sparse,
                basis=self,
                e_max=0,
                k=min(1, self.size - 1),
                dk=min(10, self.size - 1),
                eigenValueTol=0,
                slaterWeightMin=slaterWeightMin,
                verbose=False,
            )
            self.truncate(psi_ref)
            if self.verbose:
                print(f"----->After truncation, the basis contains {self.size} elements.")
        return self.build_operator_dict(H)

    def expand_at(self, w, psi_ref, H, H_dict={}, slaterWeightMin=1e-20):
        de_2_min = 1e-6

        t_applyOp = 0
        t_basis_mask = 0
        t_new_dj = 0
        t_distribute_Hpsi = 0
        t_de2 = 0
        t_de2_filter = 0
        t_spin_flip = 0
        t_add_dj = 0
        t_massage_Hpsi = 0

        t_tot = perf_counter()
        while True:
            t0 = perf_counter()
            Hpsi_ref = applyOp(
                self.num_spin_orbitals,
                H,
                psi_ref,
                restrictions=self.restrictions,
                slaterWeightMin=slaterWeightMin,
                opResult=H_dict,
            )
            t_applyOp += perf_counter() - t0

            # t_0 = perf_counter()
            # coupled_Dj = list(Hpsi_ref.keys())
            # basis_mask = np.logical_not(self.contains(coupled_Dj))
            # t_basis_mask += perf_counter() - t_0

            t_0 = perf_counter()
            Dj_basis = Basis(
                initial_basis=set(Hpsi_ref.keys()),
                # initial_basis=[Dj for j, Dj in enumerate(coupled_Dj) if basis_mask[j]],
                num_spin_orbitals=self.num_spin_orbitals,
                restrictions=self.restrictions,
                comm=self.comm,
            )
            t_new_dj += perf_counter() - t_0

            t_0 = perf_counter()
            send_list = [[] for _ in range(self.comm.size)]
            for r in range(self.comm.size):
                if Dj_basis.state_bounds[r][0] is None:
                    continue
                send_list[r] = {
                    state: val
                    for state, val in Hpsi_ref.items()
                    if state >= Dj_basis.state_bounds[r][0] and state <= Dj_basis.state_bounds[r][1]
                }
            received_Hpsi_ref = self.comm.alltoall(send_list)
            Hpsi_ref = {}
            for received_dict in received_Hpsi_ref:
                for key in received_dict:
                    Hpsi_ref[key] = received_dict[key] + Hpsi_ref.get(key, 0)
            t_distribute_Hpsi += perf_counter() - t_0

            t_0 = perf_counter()
            de_2 = self._calc_de_2(Dj_basis.local_basis, H, H_dict, Hpsi_ref, w)

            de_2_mask = np.abs(de_2) >= de_2_min
            t_de2 += perf_counter() - t_0

            t_0 = perf_counter()
            Dji = {Dj_basis.local_basis[i] for i, mask in enumerate(de_2_mask) if mask}
            t_de2_filter += perf_counter() - t_0

            # de_2_corr = {Dj_basis.local_basis[i]: de_2[i] for i, mask in enumerate(de_2_mask) if mask}
            t_0 = perf_counter()
            Dji = self._generate_spin_flipped_determinants(Dji)
            t_spin_flip += perf_counter() - t_0

            t_0 = perf_counter()
            new_Dj = Dji - set(self.local_basis)
            new_dj_sum = np.empty((1,), dtype=int)
            self.comm.Allreduce(np.array([len(new_Dj)]), new_dj_sum, op=MPI.SUM)
            self.add_states(new_Dj)
            if new_dj_sum[0] == 0:
                break
            t_add_dj += perf_counter() - t_0

            t_0 = perf_counter()
            Hpsi_keys = list(Hpsi_ref.keys())
            mask = self.contains(Hpsi_keys)
            psi_ref = {state: Hpsi_ref[state] for state, m in zip(Hpsi_keys, mask) if m}
            N = np.sqrt(norm2(psi_ref))
            psi_ref = {state: psi_ref[state] / N for state in psi_ref}
            t_massage_Hpsi += perf_counter() - t_0
        t_build_op_dict = perf_counter()
        h_dict = self.build_operator_dict(H, op_dict=H_dict)
        t_build_op_dict = perf_counter() - t_build_op_dict
        t_tot = perf_counter() - t_tot

        return h_dict

    def copy(self):
        new_basis = CIPSI_Basis(
            initial_basis=[],
            num_spin_orbitals=self.num_spin_orbitals,
            restrictions=self.restrictions,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            tau=self.tau,
            verbose=self.verbose
        )
        new_basis.add_states(self.local_basis)
        return new_basis
