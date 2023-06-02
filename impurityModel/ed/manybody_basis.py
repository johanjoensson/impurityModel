import numpy as np
from mpi4py import MPI
from math import ceil

try:
    from collections.abc import Sequence
except:
    from collections import Sequence

from impurityModel.ed import product_state_representation as psr
import itertools
from impurityModel.ed.finite import c2i, applyOp, get_job_tasks


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
                    if abs(
                        delta_impurity <= delta_impurity_occ[l]
                        and nominal_impurity_occ[l] + delta_impurity <= 2 * (2 * l + 1)
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
                nominal_impurity_occ[l] - delta_impurity_occ[l],
                nominal_impurity_occ[l] + delta_impurity_occ[l] + 1,
            )
            valence_indices = frozenset(c2i(total_baths, (l, b)) for b in range(valence_baths[l]))
            restrictions[valence_indices] = (valence_baths[l] - delta_valence_occ[l], valence_baths[l] + 1)
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
        comm=None,
        verbose=True,
    ):
        if initial_basis is not None:
            initial_basis = sorted(initial_basis)
        else:
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
        self.comm = comm
        self.num_spin_orbitals = num_spin_orbitals
        offset, local_len = self._get_offsets_and_local_lengths(len(initial_basis))
        self.local_basis = sorted(initial_basis)[offset : offset + local_len]
        self.restrictions = restrictions
        self.offset = offset
        self.size = len(initial_basis)
        self.local_indices = range(offset, offset + local_len)
        self._index_dict = {state : self.offset + i for i, state in enumerate(self.local_basis)}
        self.dtype = type(psr.int2bytes(0, self.num_spin_orbitals))
        self.n_bytes = int(ceil(self.num_spin_orbitals/8))
        self.np_dtype = f"|S{self.n_bytes}"

        local_index_bounds = (self.offset, self.offset + len(self.local_basis))
        if len(self.local_basis) > 0:
            local_state_bounds = (self.local_basis[0], self.local_basis[-1])
        else:
            local_state_bounds = (None, None)
        self.index_bounds = self.comm.allgather(local_index_bounds)
        self.state_bounds = self.comm.allgather(local_state_bounds)

        self.is_distributed = comm is not None

    def expand(self, op, op_dict):
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
                        res = applyOp(self.num_spin_orbitals, op, {state: 1}, restrictions=self.restrictions)
                        if len(res) != 0:
                            op_dict[state] = res
                    new_states |= set(res.keys()) - new_basis

                new_basis += sorted(new_states)
                self.local_basis = sorted(new_basis)
        else:
            # MPI distributed algorithm
            while not done:
                local_states = list(self.local_basis)
                local_states.extend([state for state in op_dict] + [state for key in op_dict for state in op_dict[key]])
                states_in_op_dicts = self.comm.allreduce(set(op_dict.keys()), op=combine_sets_op)

                new_states = []
                for state in local_states:
                    if state in states_in_op_dicts:
                        continue
                    res = applyOp(self.num_spin_orbitals, op, {state: 1}, restrictions=self.restrictions)
                    if len(res) != 0:
                        op_dict[state] = res
                        new_states.extend(res.keys())
                local_states.extend(new_states)

                new_basis = self.comm.reduce(set(local_states), op=combine_sets_op, root=0)
                total_local_basis_len = self.comm.reduce(len(self.local_basis), op=MPI.SUM)
                if self.comm.rank == 0:
                    new_basis = sorted(new_basis)
                    send_basis = []
                    start = 0
                    for r in range(self.comm.size):
                        stop = start + len(new_basis) // self.comm.size
                        if r < len(new_basis) % self.comm.size:
                            stop += 1
                        send_basis.append(new_basis[start:stop])
                        start = stop
                    done = len(new_basis) == total_local_basis_len
                else:
                    send_basis = None
                self.local_basis = self.comm.scatter(send_basis, root=0)
                done = self.comm.bcast(done, root=0)
            # The new basis is determined and distributed over the MPI ranks
            # Make sure that op_dict contains all the local basis states
            for state in self.local_basis:
                if state not in op_dict:
                    res = applyOp(self.num_spin_orbitals, op, {state: 1}, restrictions=self.restrictions)
                    if len(res) != 0:
                        op_dict[state] = res
        if self.comm is None:
            self.offset = 0
            self.size = len(self.local_basis)
        else:
            self.offset = self.comm.scan(len(self.local_basis), op=MPI.SUM) - len(self.local_basis)
            self.size = self.comm.allreduce(len(self.local_basis), op=MPI.SUM)
        self._index_dict = {state : self.offset + i for i, state in enumerate(self.local_basis)}
        self.local_indices = range(self.offset, self.offset + len(self.local_basis))

        local_index_bounds = (self.offset, self.offset + len(self.local_basis))
        if len(self.local_basis) > 0:
            local_state_bounds = (self.local_basis[0], self.local_basis[-1])
        else:
            local_state_bounds = (None, None)
        self.index_bounds = self.comm.allgather(local_index_bounds)
        self.state_bounds = self.comm.allgather(local_state_bounds)

        return {key: op_dict[key] for key in op_dict if key in self.local_basis}

    def _getitem_sequence(self, l):
        if self.comm is None:
            return [self.local_basis[i] for i in l]

        l = np.array([i if i >= 0 else self.size - i for i in l], dtype = int)

        send_list = [np.array([], dtype = int) for _ in range(self.comm.size)]
        send_to_ranks = []
        for i in l:
            for r in range(self.comm.size):
                if i >= self.index_bounds[r][0] and i < self.index_bounds[r][1]:
                    send_list[r] = np.append(send_list[r], [i])
                    send_to_ranks.append(r)
        send_order = np.argsort(send_to_ranks, kind = 'stable')
        recv_counts = np.empty((self.comm.size), dtype = int)
        queries = None
        displacements = None
        for r in range(self.comm.size):
            self.comm.Gather(np.array([len(send_list[r])], dtype = int), recv_counts, root = r)
            if self.comm.rank == r:
                queries = np.empty((sum(recv_counts)), dtype = int)
                displacements = np.array([sum(recv_counts[:p]) for p in range(self.comm.size)])
            self.comm.Gatherv(send_list[r], (queries, recv_counts, displacements, MPI.UINT64_T), root = r)

        results = np.empty((sum(recv_counts)), dtype = self.np_dtype)
        for i, query in enumerate(queries):
            if query >= self.offset and query < self.offset + len(self.local_basis):
                results[i] = self.local_basis[query - self.offset]
        result = np.empty((len(l)*self.n_bytes), dtype = np.byte)
        for r in range(self.comm.size):
                receive_array = result[sum(len(l) for l in send_list[:r])*self.n_bytes : sum(len(l) for l in send_list[:r+1])*self.n_bytes]
                send_counts = recv_counts*self.n_bytes if r == self.comm.rank else np.zeros((self.comm.rank))
                send_displacements = displacements*self.n_bytes if r == self.comm.rank else np.zeros((self.comm.rank))
                self.comm.Scatterv((results.tobytes(), send_counts, send_displacements, MPI.BYTE), 
                                   receive_array,
                                   root = r)
        result_new = np.empty((len(l)), dtype = self.np_dtype)
        for i in range(len(l)):
            result_new[send_order[i]] = result[i*self.n_bytes : (i+1)*self.n_bytes].tobytes()

        for i, res in enumerate(result_new):
            if res == psr.int2bytes(0, self.num_spin_orbitals):
                raise IndexError(f"Could not find index {l[i]} in basis with size {self.size}!")

        return result_new.tolist()

    def _getitem_slice(self, s):
        start = s.start
        if start is None:
            start = 0
        elif start < 0:
            start = self.size + start
        stop = s.stop
        if stop is None:
            stop = self.size
        elif stop < 0:
            stop = self.size + stop
        step = s.step
        if step is None and start < stop:
            step = 1
        elif step is None:
            step = -1
        query = [i for i in range(start, stop, step)]
        return self._getitem_sequence(query)

    def _getitem_int(self, i):
        if self.comm is None:
            return self.local_basis[i]

        if i < 0:
            i = self.size + i

        send_list = [np.array([], dtype = int) for _ in range(self.comm.size)]
        for r in range(self.comm.size):
            if i >= self.index_bounds[r][0] and i < self.index_bounds[r][1]:
                send_list[r] = np.append(send_list[r], [i])
        recv_counts = np.empty((self.comm.size), dtype = int)
        queries = None
        displacements = None
        for r in range(self.comm.size):
            self.comm.Gather(np.array([len(send_list[r])], dtype = int), recv_counts, root = r)
            if self.comm.rank == r:
                queries = np.empty((sum(recv_counts)), dtype = int)
                displacements = np.array([sum(recv_counts[:p]) for p in range(self.comm.size)])
            self.comm.Gatherv(send_list[r], (queries, recv_counts, displacements, MPI.UINT64_T), root = r)

        results = np.empty((queries.shape[0]), dtype = self.np_dtype)
        for i, query in enumerate(queries):
            if query >= self.offset and query < self.offset + len(self.local_basis):
                results[i] = self.local_basis[query - self.offset]
        result = np.empty((self.n_bytes), dtype = np.byte)
        for r in range(self.comm.size):
            if self.comm.rank == r or len(send_list[r]) > 0:
                self.comm.Scatterv((results.tobytes(), recv_counts*self.n_bytes, displacements*self.n_bytes, MPI.BYTE), result, root = r)
            else:
                self.comm.Scatterv((results.tobytes(), recv_counts*self.n_bytes, displacements*self.n_bytes, MPI.BYTE), np.empty((0)), root = r)
        result = result.tobytes()

        if result == psr.int2bytes(0, self.num_spin_orbitals):
            raise IndexError(f"Could not find index {i} in basis with size {self.size}!")
        return result

    def index(self, val):
        if isinstance(val, self.dtype):
            # return self._index_single(val)
            return self._index_sequence([val])[0]
        elif isinstance(val, Sequence):
            return self._index_sequence(val)
        else:
            raise TypeError(f"Invalid query type {type(val)}! Valid types are {self.dtype} and sequences thereof.")

    def _index_sequence(self, s):
        if self.comm is None:
            results = [self._index_dict[val] if val in self._index_dict else self.size for val in s]
            return results.tolist()

        send_list = [[] for _ in range(self.comm.size)]
        send_to_ranks = []
        for val in s:
            for r in range(self.comm.size):
                if self.state_bounds[r][0] is not None and val >= self.state_bounds[r][0] and val <= self.state_bounds[r][1]:
                    send_list[r].append(val)
                    send_to_ranks.append(r)
        send_order = np.argsort(send_to_ranks, kind = 'stable')
        recv_counts = np.empty((self.comm.size), dtype = int)
        queries = None
        displacements = 0
        for r in range(self.comm.size):
            self.comm.Gather(np.array([len(send_list[r])], dtype = int), recv_counts, root = r)
            if self.comm.rank == r:
                queries = np.empty((sum(recv_counts)*self.n_bytes), dtype = np.byte)
                displacements = np.array([sum(recv_counts[:p]) for p in range(self.comm.size)])
            self.comm.Gatherv(np.array(send_list[r], dtype = self.np_dtype).tobytes(), (queries, recv_counts*self.n_bytes, displacements*self.n_bytes, MPI.BYTE), root = r)

        results = np.empty((sum(recv_counts)), dtype = int)
        results[:] = self.size
        for i in range(sum(recv_counts)):
            query = queries[i*self.n_bytes : (i+1)*self.n_bytes].tobytes()
            if query in self._index_dict:
                results[i] = self._index_dict[query]
        for r in range(self.comm.size):
            if r == self.comm.rank:
                for i in range(sum(recv_counts)):
                    query = queries[i*self.n_bytes: (i+1)*self.n_bytes].tobytes()
                    res = results[i]
        result = np.empty((len(s)), dtype = int)
        for r in range(self.comm.size):
            receive_array = result[sum(len(l) for l in send_list[:r]) : sum(len(l) for l in send_list[:r+1])]
            send_counts = recv_counts if r == self.comm.rank else np.zeros((self.comm.rank))
            send_displacements = displacements if r == self.comm.rank else np.zeros((self.comm.rank))
            self.comm.Scatterv((results, send_counts, send_displacements, MPI.UINT64_T), receive_array, root = r)

        result[send_order] = result.copy()
        for i, res in enumerate(result):
            if res == self.size:
                raise ValueError(f"Could not find {s[i]} in basis!")

        return result.tolist()

    def _index_single(self, val):
        if self.comm is None:
            return self._index_dict[val]
            if res != self.size and self.local_basis[res] == val:
                return res
            raise ValueError(f"Index {val} not in basis with size {self.size}")

        send_list = [[] for _ in range(self.comm.size)]
        for r in range(self.comm.size):
            if self.state_bounds[r][0] is not Nonte and val >= self.state_bounds[r][0] and val <= self.state_bounds[r][1]:
                send_list[r].extend(val)
        recv_counts = np.empty((self.comm.size), dtype = int)
        queries = None
        displacements = None
        for r in range(self.comm.size):
            self.comm.Gather(np.array([len(send_list[r])], dtype = int), recv_counts, root = r)
            if self.comm.rank == r:
                queries = np.empty((sum(recv_counts)), dtype = np.byte)
                displacements = np.array([sum(recv_counts[:p]) for p in range(self.comm.size)])
            self.comm.Gatherv(np.array(send_list[r], dtype = np.byte), (queries, recv_counts, displacements, MPI.BYTE), root = r)

        results = np.empty((sum(recv_counts)//self.n_bytes), dtype = int)
        results[:] = self.size
        for i in range(sum(recv_counts)):
            query = queries[i*self.n_bytes : (i+1)*self.n_bytes].tobytes()
            if query in self._index_dict:
                results[i] = self._index_dict[query]
        result = np.empty((1), dtype = int)
        for r in range(self.comm.size):
            self.comm.Scatterv((results, recv_counts//self.n_bytes, displacements//self.n_bytes, MPI.UINT64_T), result[sum(recv_counts[:r])//self.n_bytes : sum(recv_counts[:r+1])]//self.n_bytes, root = r)
        result = result[0]

        if result == self.size:
            raise ValueError(f"Item {val} not in basis")
        return result

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._getitem_slice(key)
        if isinstance(key, Sequence):
            return self._getitem_sequence(key)
        elif isinstance(key, int):
            return self._getitem_int(key)
        else:
            raise TypeError("Invalid index type {type(key)}. Valid types are slice, Sequence and int")

    def __len__(self):
        return self.size

    def __contains__(self, item):
        if self.comm is None:
            return item in self.local_basis
        queries = self.comm.allgather(item)
        results = np.array([False for _ in queries])
        for i, query in enumerate(queries):
            if query in self.local_basis:
                results[i] = True
        result = np.empty((1), dtype=bool)
        self.comm.Reduce_scatter(results, result, op=MPI.LOR)
        return result[0]

    def __iter__(self):
        for i in range(self.size):
            yield self.__getitem__(i)
