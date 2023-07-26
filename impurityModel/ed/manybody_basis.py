import numpy as np
import scipy as sp
from mpi4py import MPI
from math import ceil
from time import perf_counter

try:
    from collections.abc import Sequence
except:
    from collections import Sequence

from impurityModel.ed import product_state_representation as psr
import itertools
from impurityModel.ed.finite import (
    c2i,
    applyOp,
    get_job_tasks,
    eigensystem_new,
    add,
    getTraceDensityMatrix,
    getEgT2gOccupation,
    thermal_average_scale_indep,
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
                            print(f"Partition occupations")
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
        truncation_threshold=None,
        comm=None,
        verbose=True,
    ):
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
            initial_basis = sorted(initial_basis)
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
        if comm is not None:
            seed_sequences = None
            if self.comm.rank == 0:
                seed_parent = np.random.SeedSequence()
                seed_sequences = seed_parent.spawn(comm.size)
            seed_sequence = comm.scatter(seed_sequences, root=0)
            self.rng = np.random.default_rng(seed_sequence)
        else:
            rng = np.random.default_rng()

        num_states = comm.allreduce(len(initial_basis), op=MPI.SUM)
        if num_states > 0:
            self.add_states(initial_basis)

    def add_states(self, new_states):
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

        local_states = sorted(set(self.local_basis) | set(new_states))

        n_samples = min(100, len(local_states) // 10)
        state_bounds = None
        done = False
        while not done:
            samples = self.rng.choice(
                local_states[1:-1], size=min(n_samples, max(0, len(local_states) - 2)), replace=False
            )
            if len(local_states) > 1:
                samples = np.append([local_states[0]], np.append(samples, [local_states[-1]]))
            elif len(local_states) == 1:
                samples = np.append([local_states[0]], samples)
            else:
                samples = []
            print (f"{self.comm.rank=} {len(samples)=}")
            all_samples = self.comm.gather(samples, root=0)

            if self.comm.rank == 0:
                all_states = sorted(set([state for samples in all_samples for state in samples]))
                print (f"{self.comm.rank=} {len(all_states)=}")
                done = True
                sizes = [
                    len(all_states) // self.comm.size + (1 if i < len(all_states) % self.comm.size else 0)
                    for i in range(self.comm.size)
                ]
                bounds = [sum(sizes[:i]) for i in range(self.comm.size)]
                state_bounds = [all_states[bound] if bound < len(all_states) else all_states[-1] for bound in bounds]
                print (f"{bounds=}")
            done = self.comm.bcast(done, root=0)
        state_bounds = self.comm.bcast(state_bounds, root=0)
        send_list = [[] for _ in range(self.comm.size)]
        for r in range(self.comm.size - 1):
            send_list[r] = [state for state in local_states if state >= state_bounds[r] and state < state_bounds[r + 1]]
        send_list[-1] = [state for state in local_states if state >= state_bounds[-1]]

        received_states = None
        for r in range(self.comm.size):
            if r == self.comm.rank:
                received_states = self.comm.gather(send_list[r], root=r)
            else:
                _ = self.comm.gather(send_list[r], root=r)

        self.local_basis = sorted(set(state for states in received_states for state in states))
        local_length = len(self.local_basis)
        ########################################################################
        # The local lengths are not balanced! The basis is sorted, but not
        # evenly distributed among the ranks.
        ########################################################################

        self.size = self.comm.allreduce(len(self.local_basis), op=MPI.SUM)
        self.offset = self.comm.scan(local_length, op=MPI.SUM) - local_length
        self.local_indices = range(self.offset, self.offset + local_length)
        self._index_dict = {state: self.offset + i for i, state in enumerate(self.local_basis)}
        local_index_bounds = (self.offset, self.offset + len(self.local_basis))
        if len(self.local_basis) > 0:
            local_state_bounds = (self.local_basis[0], self.local_basis[-1])
        else:
            local_state_bounds = (None, None)
        self.index_bounds = self.comm.allgather(local_index_bounds)
        self.state_bounds = self.comm.allgather(local_state_bounds)
        print (f"{self.size=}")
        print (f"{self.index_bounds=}")

    def expand(self, op, op_dict={}, dense_cutoff=None):
        done = False
        if self.comm is None:
            # serial algorithm
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
                            self.num_spin_orbitals, op, {state: 1}, restrictions=self.restrictions, slaterWeightMin=0
                        )
                        if len(res) != 0:
                            op_dict[state] = res
                    new_states |= set(res.keys()) - new_basis

                new_basis += sorted(new_states)
                self.local_basis = sorted(new_basis)
        else:
            # MPI distributed algorithm
            while not done:
                local_states = []
                # local_states.extend([state for state in op_dict] + [state for key in op_dict for state in op_dict[key]])
                # states_in_op_dicts = self.comm.allreduce(set(op_dict.keys()), op=combine_sets_op)

                new_local_states = []
                for state in set(self.local_basis + local_states):
                    # if state in states_in_op_dicts:
                    #     continue
                    res = applyOp(
                        self.num_spin_orbitals, op, {state: 1}, restrictions=self.restrictions, slaterWeightMin=0
                    )
                    op_dict[state] = res
                    new_local_states.extend(res.keys())
                local_states.extend(new_local_states)

                old_size = self.size
                self.add_states(local_states)
                done = old_size == self.size
        if self.comm.rank == 0:
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
        for r in range(self.comm.size):
            self.comm.Gather(np.array([len(send_list[r])], dtype=int), recv_counts, root=r)
            if self.comm.rank == r:
                queries = np.empty((sum(recv_counts)), dtype=int)
                displacements = np.array([sum(recv_counts[:p]) for p in range(self.comm.size)])
            self.comm.Gatherv(send_list[r], (queries, recv_counts, displacements, MPI.UINT64_T), root=r)

        results = np.empty((sum(recv_counts)), dtype=self.np_dtype)
        for i, query in enumerate(queries):
            if query >= self.offset and query < self.offset + len(self.local_basis):
                results[i] = np.frombuffer(self.local_basis[query - self.offset], dtype="B")
        result = np.zeros((len(l) * self.n_bytes), dtype=np.byte)
        for r in range(self.comm.size):
            receive_array = result[
                sum(len(l) for l in send_list[:r])
                * self.n_bytes : sum(len(l) for l in send_list[: r + 1])
                * self.n_bytes
            ]
            send_counts = recv_counts * self.n_bytes if r == self.comm.rank else np.zeros((self.comm.rank))
            send_displacements = displacements * self.n_bytes if r == self.comm.rank else np.zeros((self.comm.rank))
            self.comm.Scatterv((results.tobytes(), send_counts, send_displacements, MPI.BYTE), receive_array, root=r)
        result_new = np.zeros((len(l)), dtype=self.np_dtype)
        for i in range(len(l)):
            if len(send_order) > 0:
                result_new[send_order[i]] = result[i * self.n_bytes : (i + 1) * self.n_bytes]
            else:
                result_new[0] = result[0 : self.n_bytes]
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
            return results.tolist()

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
        queries = None
        displacements = 0
        for r in range(self.comm.size):
            self.comm.Gather(np.array([len(send_list[r])], dtype=int), recv_counts, root=r)
            if self.comm.rank == r:
                queries = np.empty((sum(recv_counts) * self.n_bytes), dtype=np.byte)
                displacements = np.array([sum(recv_counts[:p]) for p in range(self.comm.size)])
            self.comm.Gatherv(
                send_list[r].tobytes(),
                (queries, recv_counts * self.n_bytes, displacements * self.n_bytes, MPI.BYTE),
                root=r,
            )

        results = np.empty((sum(recv_counts)), dtype=int)
        results[:] = self.size
        for i in range(sum(recv_counts)):
            query = queries[i * self.n_bytes : (i + 1) * self.n_bytes].tobytes()
            if query in self._index_dict:
                results[i] = self._index_dict[query]
        result = np.empty((len(s)), dtype=int)
        result[:] = self.size
        for r in range(self.comm.size):
            receive_array = result[sum(len(l) for l in send_list[:r]) : sum(len(l) for l in send_list[: r + 1])]
            send_counts = recv_counts if r == self.comm.rank else np.zeros((self.comm.rank))
            send_displacements = displacements if r == self.comm.rank else np.zeros((self.comm.rank))
            self.comm.Scatterv((results, send_counts, send_displacements, MPI.UINT64_T), receive_array, root=r)
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

    def build_operator_dict(self, op, op_dict=None, distributed=True):
        """
        Express the operator, op, in the current basis. Do not expand the basis.
        Return a dict containing the results of applying op to the different basis states
        """
        if op_dict is None:
            op_dict = {}

        for state in self.local_basis:
            if state not in op_dict:
                res = applyOp(
                    self.num_spin_orbitals,
                    op,
                    {state: 1},
                    restrictions=self.restrictions,
                    slaterWeightMin=0,
                    opResult=op_dict,
                )

        all_row_states = [state for column in op_dict for state in op_dict[column]]
        row_indices = np.array(self._index_sequence(all_row_states))
        in_basis_mask = row_indices != self.size
        state_in_basis = {state: in_basis for state, in_basis in zip(all_row_states, in_basis_mask)}
        for column in list(op_dict.keys()):
            if column not in self.local_basis:
                op_dict.pop(column, None)
                continue
            for row in list(op_dict[column].keys()):
                if not state_in_basis[row]:
                    op_dict[column].pop(row, None)

        return op_dict

    def build_sparse_operator(self, op, op_dict=None):
        """
        Get the operator as a sparse matrix in the current basis.
        The sparse matrix is distributed over all ranks.
        """
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
        return sp.sparse.csr_matrix((values, (columns, rows)), shape=(self.size, self.size), dtype=complex)


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
        truncation_threshold=None,
        verbose=False,
        H=None,
        tau=None,
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
            verbose=verbose,
            comm=comm,
        )
        if tau is None:
            tau = 0.0
        self.tau = tau

        if self.size > self.truncation_threshold and H is not None:
            H_sparse = self.build_sparse_operator(H)
            e_ref, psi_ref = eigensystem_new(
                H_sparse,
                basis=self,
                e_max=1e-12,
                k=min(1, self.size - 1),
                dk=min(10, self.size - 1),
                eigenValueTol=0,
                slaterWeightMin=0,
                verbose=False,
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

    def _calc_de_2(self, Djs, H, H_dict, psi_ref, e_ref):
        """
        calculate second variational energy contribution of the Slater determinants in states.
        """

        overlap = np.empty((len(Djs)), dtype=complex)
        e_state = np.empty((len(Djs)), dtype=float)
        Hpsi = applyOp(
            self.num_spin_orbitals,
            H,
            psi_ref,
            restrictions=self.restrictions,
            slaterWeightMin=0,
            opResult=H_dict,
        )
        for j, Dj in enumerate(Djs):
            # <Dj|H|Psi_ref>
            overlap[j] = Hpsi[Dj] if Dj in Hpsi else 0
            HDj = applyOp(
                self.num_spin_orbitals, H, {Dj: 1}, restrictions=self.restrictions, slaterWeightMin=0, opResult=H_dict
            )
            # <Dj|H|Dj>
            e_state[j] = np.real(HDj[Dj] if Dj in HDj else 0)
        de = e_ref - e_state
        inf_mask = np.abs(de) < 1e-15
        overlap[inf_mask] = 1e15
        de[inf_mask] = 1

        return np.abs(overlap) ** 2 / de

    def expand(self, H, H_dict={}, e_conv=np.finfo(float).eps, dense_cutoff=1e3, slaterWeightMin=0):
        """
        Use the CIPSI method to expand the basis. Keep adding Slater determinants until the CIPSI energy is converged.
        """
        psi_ref = None
        e_cipsi_prev = np.inf
        e_cipsi = 0
        de_2_min = 1e-2
        converge_count = 0
        de0_max = -self.tau * np.log(np.finfo(float).eps)
        while converge_count < 3:
            t0 = perf_counter()
            H_sparse = self.build_sparse_operator(H)
            t0 = perf_counter() - t0
            t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
            # if self.comm.rank == 0:
            #     print (f"Time to build sparse H: {t0/self.comm.size:.3f} seconds")

            t0 = perf_counter()
            if self.size > dense_cutoff and psi_ref is not None:
                v0 = np.zeros((self.size, 1), dtype=complex)
                v0_states = psi_ref[0].keys()
                v0_indices = self.index(list(v0_states))
                for i, state in zip(v0_indices, v0_states):
                    v0[i, 0] = psi_ref[0][state]
            else:
                v0 = None
            e_ref, psi_ref = eigensystem_new(
                H_sparse,
                basis=self,
                e_max=de0_max,
                k=1,
                dk=1 if psi_ref is None else max(1, len(psi_ref)),
                v0=v0,
                eigenValueTol= de_2_min if de_2_min > 1e-8 else 0,
                slaterWeightMin=slaterWeightMin,
                dense_cutoff=dense_cutoff,
                verbose=self.comm.rank == 0 and False,
            )
            t0 = perf_counter() - t0
            t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
            if self.comm.rank == 0:
                print (f"Time to get psi_ref: {t0/self.comm.size:.3f} seconds")
            t0 = perf_counter()
            weights = np.exp(-(e_ref - e_ref[0]) / max(self.tau, 1e-15))
            weights /= sum(weights)
            psi_sum = {}
            N = len(psi_ref)
            for i, psi in enumerate(psi_ref):
                psi_sum = add(psi_sum, psi, weights[i])
            e_ref = np.sum(weights * e_ref)
            Hpsi_ref = applyOp(
                self.num_spin_orbitals,
                H,
                psi_sum,
                restrictions=self.restrictions,
                slaterWeightMin=0,
                opResult=H_dict,
            )
            t0 = perf_counter() - t0
            t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
            # if self.comm.rank == 0:
            #     print (f"Time to average psi_ref: {t0/self.comm.size:.3f} seconds")
            t0 = perf_counter()
            coupled_Dj = list(Hpsi_ref.keys())
            basis_mask = np.logical_not(self.contains(coupled_Dj))
            Djs = [Dj for j, Dj in enumerate(coupled_Dj) if basis_mask[j]]
            num_Djs = np.empty((1,), dtype=int)
            self.comm.Allreduce(np.array([len(Djs)]), num_Djs, op=MPI.SUM)
            if num_Djs == 0:
                break

            local_Djs = len(Djs) // self.comm.size
            if self.comm.rank < len(Djs) % self.comm.size:
                local_Djs += 1
            offset = self.comm.scan(local_Djs, op=MPI.SUM) - local_Djs
            Djs = Djs[offset : offset + local_Djs]

            de_2 = self._calc_de_2(Djs, H, H_dict, psi_sum, e_ref)
            de_2_max_arr = np.empty((1,))
            de_2_min_arr = np.empty((1,))
            if len(de_2) == 0:
                de_2 = np.array([0], dtype=float)
            self.comm.Allreduce(np.array([np.max(np.abs(de_2))]), de_2_max_arr, op=MPI.MAX)
            de_2_min = min(de_2_min, max(de_2_max_arr[0] ** 1.5, np.finfo(float).eps ** 2))
            de_2_mask = np.abs(de_2) > de_2_min
            de_2_mask_sum = np.empty((1,), dtype=int)
            self.comm.Allreduce(np.array([sum(de_2_mask)]), de_2_mask_sum, op=MPI.SUM)

            if de_2_mask_sum[0] == 0:
                break

            old_size = self.size
            new_Dj = [Djs[i] for i, mask in enumerate(de_2_mask) if mask]
            self.add_states(new_Dj)

            t0 = perf_counter() - t0
            t0 = self.comm.reduce(t0, op=MPI.SUM, root=0)
            # if self.comm.rank == 0:
            #     print (f"Time to add new Djs: {t0/self.comm.size:.3f} seconds")
            e_pt2 = np.empty((1,))
            self.comm.Allreduce(np.array([np.sum(de_2[de_2_mask])]), e_pt2, op=MPI.SUM)
            e_pt2 = e_pt2[0]

            de_cipsi = abs(e_cipsi - (e_ref + e_pt2))
            e_cipsi_prev = e_cipsi
            e_cipsi = e_ref + e_pt2
            if de_cipsi <= e_conv:
                converge_count += 1
            else:
                converge_count = 0
            if self.comm.rank == 0:
                print(
                    f"--------> N = {self.size: 7,d}, log(de_2_min) = {np.log10(de_2_min): 5.1f}, log(|e_pt2|) = {np.log10(abs(e_pt2)): 5.1f}, log(|de_cipsi|) = {np.log10(de_cipsi): 5.1f}"
                )

        if self.comm.rank == 0:
            print(f"After expansion, the basis contains {self.size} elements.")

        if self.size > self.truncation_threshold:
            H_sparse = self.build_sparse_operator(H)
            e_ref, psi_ref = eigensystem_new(
                H_sparse,
                basis=self,
                e_max=0,
                k=min(1, self.size - 1),
                dk=min(10, self.size - 1),
                eigenValueTol=0,
                slaterWeightMin=0,
                verbose=False,
            )
            self.truncate(psi_ref)
            if self.comm.rank == 0:
                print(f"After truncation, the basis contains {self.size} elements.")
        return self.build_operator_dict(H)

    def copy(self):
        new_basis = CIPSI_Basis(
            initial_basis=[],
            num_spin_orbitals=self.num_spin_orbitals,
            restrictions=self.restrictions,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            tau=self.tau,
        )
        new_basis.add_states(self.local_basis)
        return new_basis
