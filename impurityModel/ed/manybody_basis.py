import numpy as np
from bisect import bisect_left
from mpi4py import MPI
try:
    from collections.abc import Sequence
except:
    from collections import Sequence

from impurityModel.ed import product_state_representation as psr
import itertools
from impurityModel.ed.finite import c2i, applyOp, get_job_tasks

def combine_sets(set_1, set_2, datatype):
    return set_1 | set_2
combine_sets_op = MPI.Op.Create(combine_sets, commute = True)

def reduce_subscript(a, b, datatype):
    res = np.empty_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] is None:
                res[i][j] = b[i][j]
            else:
                res[i][j] = a[i][j]
    return res
reduce_subscript_op = MPI.Op.Create(reduce_subscript, commute = True)

class Basis:
    def _calculate_offsets_and_local_lengths(self, total_length):
        offset = 0
        local_len = total_length
        if self.comm is not None:
            local_len = total_length // self.comm.size
            leftovers = total_length % self.comm.size
            if leftovers != 0 and self.comm.rank < leftovers:
                local_len += 1
            offset = self.comm.scan(local_len, op = MPI.SUM) - local_len
        return offset, local_len
    
    def _initial_basis(self,
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
                print (f"{l=}")
            valid_configurations = []
            for delta_valence in range(delta_valence_occ[l] + 1):
                for delta_conduction in range(delta_conduction_occ[l] + 1):
                    delta_impurity = delta_valence - delta_conduction
                    if abs(delta_impurity <= delta_impurity_occ[l] and nominal_impurity_occ[l] + delta_impurity <= 2*(2*l + 1)):
                        impurity_occupation = nominal_impurity_occ[l] + delta_impurity
                        valence_occupation = valence_baths[l] - delta_valence
                        conduction_occupation = delta_conduction
                        if verbose:
                            print (f"Partition occupations")
                            print (f"Impurity occupation:   {impurity_occupation:d}")
                            print (f"Valence onccupation:   {valence_occupation:d}")
                            print (f"Conduction occupation: {conduction_occupation:d}")
                        impurity_electron_indices = [c2i(total_baths, (l, s, m)) for s in range(2) for m in range(-l, l+1)]
                        impurity_configurations = itertools.combinations(impurity_electron_indices, impurity_occupation)
                        valence_electron_indices = [c2i(total_baths, (l, b)) for b in range(valence_baths[l])]
                        valence_configurations = itertools.combinations(valence_electron_indices, valence_occupation)
                        conduction_electron_indices = [c2i(total_baths, (l, b)) for b in range(valence_baths[l], total_baths[l])]
                        conduction_configurations = itertools.combinations(conduction_electron_indices, conduction_occupation)
                        valid_configurations.append(itertools.product(impurity_configurations, valence_configurations, conduction_configurations))
            configurations[l] = [imp + val + cond for configuration in valid_configurations for (imp, val, cond) in configuration]
        num_spin_orbitals = sum(2 * (2*l + 1) + total_baths[l] for l in total_baths)
        basis = []
        # Combine all valid configurations for all l-subconfigurations (ex. p-states and d-states)
        for system_configuration in itertools.product(*configurations.values()):
            basis.append(psr.tuple2bytes(tuple(sorted(itertools.chain.from_iterable(system_configuration))), num_spin_orbitals))
        return basis


    def __init__(self,
                 initial_basis = None,
                 valence_baths = None,
                 conduction_baths = None,
                 delta_valence_occ = None,
                 delta_conduction_occ = None,
                 delta_impurity_occ = None,
                 nominal_impurity_occ = None,
                 comm = None,
                 verbose = True
                 ):
        if initial_basis is not None:
            initial_basis = sorted(initial_basis)
        else:
            initial_basis = self._initial_basis(
                    valence_baths = valence_baths,
                    conduction_baths = conduction_baths,
                    delta_valence_occ = delta_valence_occ,
                    delta_conduction_occ = delta_conduction_occ,
                    delta_impurity_occ =delta_impurity_occ,
                    nominal_impurity_occ = nominal_impurity_occ,
                    verbose = verbose,
                    )
        self.comm = comm
        offset, local_len = self._calculate_offsets_and_local_lengths(len(initial_basis))
        self.local_basis = np.array(sorted(initial_basis)[offset:offset+local_len])
        self.offset = offset
        self.size = len(initial_basis)
        self.local_indices = range(offset, offset + local_len)
        self.is_distributed = comm is not None
        self.dtype = type(initial_basis[0])
        

    def expand(self, num_spin_orbitals, op, op_dict, restrictions):
        done = False
        if self.comm is None:
            # serial algorithm
            while not done:
                states_to_check = new_states
                new_states = set()
                for state in states_to_check:
                    if state in op_dict:
                        res = op_dict[state]
                    else:
                        res = applyOp(num_spin_orbitals, op, {state: 1}, restrictions=restrictions)
                        op_dict[state] = res
                    new_states.update(set(res.keys()).difference(new_basis))

                new_basis += sorted(new_states)
                done = len(new_states) == 0
        else:
            # MPI distributed algorithm
            while not done:
                # new_rows = set()
                # states_local = set()
                # states_in_op_dict = set()
                states_in_op_dict = set(op_dict.keys())
                states_local = states_in_op_dict - set(self.local_basis)
                for state in op_dict:
                    res = op_dict[state]
                    states_local |= set(res.keys())
                for state in set(self.local_basis) - set(op_dict.keys()):
                    res = applyOp(num_spin_orbitals, op, {state: 1}, restrictions = restrictions)
                    op_dict[state] = res
                    states_local |= set(res.keys()) - set(self.local_basis)

                # Add unique elements of basis_new_local into basis_new
                new_basis = self.comm.reduce(states_local | set(self.local_basis),
                                           op = combine_sets_op, root = 0)
                total_local_basis_len = self.comm.reduce(len(self.local_basis), op = MPI.SUM)
                if self.comm.rank == 0:
                    new_basis = sorted(new_basis)
                    send_basis = []
                    start = 0
                    for r in range(self.comm.size):
                        stop = start + len(new_basis) // self.comm.size 
                        if r < len(new_basis) % self.comm.size:
                            stop += 1
                        send_basis.append(new_basis[start: stop])
                        start = stop
                    done = len(new_basis) == total_local_basis_len
                else:
                    send_basis = None
                self.local_basis = self.comm.scatter(send_basis, root = 0)
                done = self.comm.bcast(done, root = 0)
                # Update op_dict to only include entries from the local basis
                # We are done if we found no new states to add to the basis, on any rank
        if self.comm is None:
            self.offset = 0
            self.size = len(self.local_basis)
        else:
            self.offset = self.comm.scan(len(self.local_basis), op = MPI.SUM) - len(self.local_basis)
            self.size = self.comm.allreduce(len(self.local_basis), op = MPI.SUM)
        self.local_indices = range(self.offset, self.offset + len(self.local_basis))
        return {key:op_dict[key] for key in op_dict if key in self.local_basis}

    def _getitem_sequence(self, l):
        if self.comm is None:
            return [self.local_basis[i] for i in l] 

        local_query = l.copy()
        len_query = len(local_query)
        max_queries = self.comm.allreduce(len(local_query), op = MPI.MAX)
        for _ in range(len_query, max_queries):
            local_query.append(0)
        local_query = np.array(local_query)
        queries = np.empty((self.comm.size, max_queries), dtype = int)
        self.comm.Allgather(local_query, queries)

        results = np.empty_like(queries, dtype = object)
        results[:, :] = None
        for i, query in enumerate(queries):
            for j, q in enumerate(query):
                if (q is not None and 
                    q >= self.offset and 
                    q < self.offset + len(self.local_basis)):
                    results[i, j] = self.local_basis[q - self.offset]

        reduced_results = self.comm.reduce(results, op = reduce_subscript_op, root = 0)
        results = self.comm.scatter(reduced_results, root = 0)

        for res in results[:len_query]:
            if res is None:
                raise IndexError

        return [results[i] for i in range(len_query)]

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

        queries = np.empty((self.comm.size,), dtype = int)
        self.comm.Allgather(np.array([i]), queries)
        results = np.array([[None] for _ in queries])
        for i, query in enumerate(queries):
            if query < 0:
                query = self.size + query
            if query >= self.offset and query < self.offset + len(self.local_basis):
                results[i] = self.local_basis[query - self.offset]

        reduced_results = self.comm.reduce(results, op = reduce_subscript_op, root = 0)
        result = self.comm.scatter(reduced_results)
        assert len(result.shape) == 1
        assert result.shape[0] == 1
        if result[0] is None:
            raise IndexError
        return result[0]

    def index(self, val):
        if isinstance(val, self.dtype):
            return self._index_single(val)
        elif isinstance(val, Sequence):
            return self._index_sequence(val)
        else:
            raise TypeError

    def _index_sequence(self, s):
        if self.comm is None:
            results = [np.searchsorted(self.local_basis, val, side = 'left') for val in s]
            for i, res in enumerate(results):
                if res == self.size or self.local_basis[res] != s[i]:
                    raise ValueError
            return results

        local_query = s.copy()
        len_query = len(local_query)
        max_queries = self.comm.allreduce(len(local_query), op = MPI.MAX)
        for _ in range(len_query, max_queries):
            local_query.append(None)
        local_query = np.array(local_query)
        queries = self.comm.allgather(local_query)

        results = np.empty((self.comm.size, max_queries), dtype = int)
        results[:, :] = self.size

        for i, query in enumerate(queries):
            for j, q in enumerate(query):
                if q is not None:
                    res = np.searchsorted(self.local_basis, q)
                    if res != len(self.local_basis) and self.local_basis[res] == q:
                        results[i, j] = res + self.offset
        result = np.empty((max_queries), dtype = int)
        self.comm.Reduce_scatter(results, result, op = MPI.MIN)

        for i, res in enumerate(result[:len_query]):
            if res == self.size:
                raise ValueError

        return list(result[ :len_query])

    def _index_single(self, val):
        if self.comm is None:
            res = np.searchsorted(self.local_basis, val, side = 'left')
            if res != self.size and self.local_basis[res] == val:
                return res
            raise ValueError
        queries = self.comm.allgather(val)
        results = np.empty((self.comm.size), dtype = int)
        results[:] = self.size
        for i, query in enumerate(queries):
            res = np.searchsorted(self.local_basis, query, side = 'left')
            if res != len(self.local_basis) and self.local_basis[res] == query:
                results[i] = self.offset + res
        result = np.empty((1), dtype = int)
        self.comm.Reduce_scatter(results, result, op = MPI.MIN)
        if result[0] == self.size:
            raise ValueError
        return result

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._getitem_slice(key)
        if isinstance(key, Sequence):
            return self._getitem_sequence(key)
        elif isinstance(key, int):
            return self._getitem_int(key)
        else:
            raise TypeError

    def __len__(self):
        if self.comm is None:
            return self.size
        res = self.size
        return res
    
    def __contains__(self, item):
        if self.comm is None:
            return item in self.local_basis
        queries = self.comm.allgather(item)
        results = np.array([False for _ in queries])
        for i, query in enumerate(queries):
            if query in self.local_basis:
                results[i] = True
        result = np.empty((1), dtype = bool)
        self.comm.Reduce_scatter(results, result, op = MPI.LOR)
        return result[0]

    def __iter__(self):
        for i in range(self.size):
            yield self.__getitem__(i)

