from typing import Optional

try:
    from collections.abc import Iterable, Sequence
except ModuleNotFoundError:
    from collections import Iterable, Sequence
import itertools
from heapq import merge

import numpy as np
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import SlaterDeterminant
from impurityModel.ed.mpi_comm import graph_alltoall


def batched(iterable: Iterable, n: int) -> Iterable:
    """
    batched('ABCDEFG', 3) → ABC DEF G
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def hash_key(state):
    """Hash function for consistent state comparison across ranks."""
    return state.get_hash()


class StateContainer:
    """
    Base class for managing and mapping a collection of many-body states.

    This container supports indexing, iteration, and membership checks for
    many-body states (such as Slater determinants), with support for
    parallel execution and distributed storage across MPI ranks.
    """

    class IndexDict:
        """O(1) state → index mapping backed by a plain dict.

        Replaces the previous bisect-based implementation which was
        O(log n) per lookup and paid Python-level comparison overhead on
        every call.  SlaterDeterminant already implements __hash__ and
        __eq__ via the Cython layer, so dict lookup is safe and fast.

        The ``states`` list and ``offset`` attributes are kept for
        compatibility with code that inspects them directly.
        """

        def __init__(self, states: list, offset: int):
            """
            Initialize the IndexDict mapping.

            Parameters
            ----------
            states : list of SlaterDeterminant
                Sorted list of local states.
            offset : int
                Global index offset for the local states on this MPI rank.
            """
            self.states = states
            self.offset = offset
            # Build the O(1) lookup table from the sorted list.
            self._map: dict = {state: offset + i for i, state in enumerate(states)}

        def _rebuild(self):
            """Rebuild _map from self.states (call after mutating self.states)."""
            self._map = {state: self.offset + i for i, state in enumerate(self.states)}

        def __contains__(self, key):
            """
            Check if the state is contained in the mapping.

            Parameters
            ----------
            key : SlaterDeterminant or bytes
                The state to search for.

            Returns
            -------
            bool
                True if the state is in the mapping, False otherwise.
            """
            if isinstance(key, bytes) and self.states:
                key = type(self.states[0]).from_bytes(key)
            return key in self._map

        def __getitem__(self, key):
            """
            Get the global index of a state.

            Parameters
            ----------
            key : SlaterDeterminant or bytes
                The state whose index is retrieved.

            Returns
            -------
            int
                The global index of the state.

            Raises
            ------
            ValueError
                If the state is not found in the mapping.
            """
            if isinstance(key, bytes) and self.states:
                key = type(self.states[0]).from_bytes(key)
            try:
                return self._map[key]
            except KeyError:
                raise ValueError

        def get(self, key, default=None):
            """
            Get the global index of a state, or return a default value if not found.

            Parameters
            ----------
            key : SlaterDeterminant or bytes
                The state whose index is retrieved.
            default : Any, optional
                The default value to return if the state is not found. Default is None.

            Returns
            -------
            int or Any
                The global index of the state, or the default value.
            """
            if isinstance(key, bytes) and self.states:
                key = type(self.states[0]).from_bytes(key)
            return self._map.get(key, default)

        def __len__(self):
            """
            Return the number of mapped states in the dictionary.

            Returns
            -------
            int
                Number of states.
            """
            return len(self._map)

    def __init__(self, states, bytes_per_state, state_type, comm):
        """
        Initialize the StateContainer.

        Parameters
        ----------
        states : Iterable
            Initial collection of states to add.
        bytes_per_state : int
            Number of bytes representing a single state.
        state_type : type
            The class/type representing each state (e.g., SlaterDeterminant).
        comm : MPI.Comm or None
            MPI communicator for distributed parallel runs.
        """
        self.local_basis = []
        self.comm = comm
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict = StateContainer.IndexDict(self.local_basis, self.offset)
        self.type = state_type
        self.n_bytes = bytes_per_state
        self.is_distributed = comm is not None and comm.size > 1
        self.index_bounds = [None] * comm.size if self.is_distributed else [None]
        self.state_bounds = [None] * comm.size if self.is_distributed else [None]
        self.add_states(states)
        if self.size > 0 and self.type is None:
            self.type = type(self[0])

    def __iter__(self):
        """
        Iterate over all states in the container.

        Yields
        ------
        SlaterDeterminant
            Each state stored in the container in order of their global index.
        """
        chunk_size = 10000
        for i in range(0, self.size, chunk_size):
            chunk_end = min(i + chunk_size, self.size)
            chunk = self._getitem_sequence(range(i, chunk_end))
            for state in chunk:
                yield state

    def add_states(self, new_states: Iterable[SlaterDeterminant]) -> None:
        """
        Add new states to the container.

        This method must be implemented by subclasses.

        Parameters
        ----------
        new_states : Iterable[SlaterDeterminant]
            States to be added to the container.
        """
        pass

    def __getitem__(self, key) -> Iterable[SlaterDeterminant]:
        """
        Retrieve state(s) from the container by index, slice, or sequence of indices.

        Parameters
        ----------
        key : int, slice, Sequence[int], or Iterable[int]
            The index, slice, or collection of indices to retrieve.

        Returns
        -------
        SlaterDeterminant or Iterable[SlaterDeterminant]
            The state at the specified index, or an iterator yielding the states
            for the specified indices or slice.

        Raises
        ------
        IndexError
            If the index is out of bounds or the state cannot be found.
        TypeError
            If the index type is invalid.
        """
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
            query = range(start, stop, step)
            result = list(self._getitem_sequence(query))
            for i, res in enumerate(result):
                if res == SlaterDeterminant.from_bytes(bytes(0)):
                    raise IndexError(f"Could not find index {query[i]} in basis with size {self.size}!")
            return (state for state in result)
        elif isinstance(key, Sequence) or isinstance(key, Iterable):
            result = list(self._getitem_sequence(key))
            for i, res in enumerate(result):
                if res == SlaterDeterminant.from_bytes(bytes(0)):
                    raise IndexError(f"Could not find index {key[i]} in basis with size {self.size}!")
                # if res == psr.int2bytes(0, self.num_spin_orbitals):
                #     raise IndexError(f"Could not find index {key[i]} in basis with size {self.size}!")
            return (state for state in result)
        elif isinstance(key, int):
            result = next(self._getitem_sequence([key]))
            # if result is None:
            if result == SlaterDeterminant.from_bytes(bytes(0)):
                raise IndexError(f"Could not find index {key} in basis with size {self.size}!")
            return result
        else:
            raise TypeError(f"Invalid index type {type(key)}. Valid types are slice, Sequence and int")
        return None

    def __len__(self):
        """
        Get the total number of states in the container across all ranks.

        Returns
        -------
        int
            The global size of the container.
        """
        return self.size

    def index(self, val):
        """
        Get the global index or indices of the given state(s).

        Parameters
        ----------
        val : SlaterDeterminant, bytes, or Sequence/Iterable of them
            The state or collection of states to search for.

        Returns
        -------
        int or Iterable[int]
            The global index of the state, or an iterator of global indices
            if a sequence of states is queried.

        Raises
        ------
        ValueError
            If any state is not found in the container.
        TypeError
            If the query type is invalid.
        """
        if isinstance(val, bytes):
            val = self.type.from_bytes(val)
        if isinstance(val, self.type):
            res = next(self._index_sequence([val]))
            if res == self.size:
                raise ValueError(f"Could not find {val} in basis!")
            return res
        elif isinstance(val, Sequence) or isinstance(val, Iterable):
            converted = [self.type.from_bytes(x) if isinstance(x, bytes) else x for x in val]
            res = list(self._index_sequence(converted))
            for i, v in enumerate(res):
                if v >= self.size:
                    raise ValueError(f"Could not find {list(val)[i]} in basis!")
            return (i for i in res)
        else:
            raise TypeError(f"Invalid query type {type(val)}! Valid types are {self.type} and sequences thereof.")
        return None

    def _index_sequence(self, s: Iterable[SlaterDeterminant]) -> Iterable[int]:
        """
        Look up the global indices for a sequence of states.

        This method must be implemented by subclasses.

        Parameters
        ----------
        s : Iterable[SlaterDeterminant]
            The states to look up.

        Returns
        -------
        Iterable[int]
            The global indices corresponding to each state.
        """
        pass

    def contains(self, item) -> Iterable[bool]:
        """
        Check membership for a state or sequence of states.

        Parameters
        ----------
        item : SlaterDeterminant, bytes, or Sequence/Iterable of them
            The state or states to check for containment.

        Returns
        -------
        bool or Iterable[bool]
            True/False for a single state, or an iterable of booleans for a
            sequence of states.
        """
        if isinstance(item, bytes):
            item = self.type.from_bytes(item)
        if isinstance(item, self.type):
            return next(self._contains_sequence([item]))
        elif isinstance(item, Sequence) or isinstance(item, Iterable):
            converted = [self.type.from_bytes(x) if isinstance(x, bytes) else x for x in item]
            return self._contains_sequence(converted)
        return None

    def __contains__(self, item):
        """
        Check if a single state is in the container.

        Parameters
        ----------
        item : SlaterDeterminant or bytes
            The state to search for.

        Returns
        -------
        bool
            True if the state is in the container, False otherwise.
        """
        if isinstance(item, bytes):
            item = self.type.from_bytes(item)
        if not self.is_distributed:
            return item in self._index_dict
        return next(self._index_sequence([item])) != self.size

    def _contains_sequence(self, items):
        """
        Check membership for a sequence of states.

        This method must be implemented by subclasses.

        Parameters
        ----------
        items : Iterable[SlaterDeterminant]
            The states to check.

        Returns
        -------
        Iterable[bool]
            An iterable of booleans indicating if each state is present.
        """
        pass

    def clear(self):
        """
        Clear all states and reset the container.
        """
        self.local_basis.clear()
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict = StateContainer.IndexDict(self.local_basis, self.offset)
        self.index_bounds = [None] * self.comm.size if self.is_distributed else [None]
        self.state_bounds = [None] * self.comm.size if self.is_distributed else [None]

    def alltoall_states(self, send_list: list[list[SlaterDeterminant]], flatten=False):
        """
        Perform an MPI all-to-all exchange of many-body states.

        Parameters
        ----------
        send_list : list of list of SlaterDeterminant
            Outer list of size comm.size, where element `r` is a list of
            states to send to rank `r`.
        flatten : bool, optional
            If True, the received states are flattened into a single list.
            If False, returns a list of iterables of states, one per rank.
            Default is False.

        Returns
        -------
        list of Iterable of SlaterDeterminant or Iterable of SlaterDeterminant
            The received states.
        """
        recv_counts = np.empty((self.comm.size), dtype=int)
        request = self.comm.Ialltoall(
            (np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list)), MPI.LONG), recv_counts
        )

        send_counts = np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list))
        send_offsets = np.fromiter(
            (np.sum(send_counts[:i]) for i in range(self.comm.size)), dtype=int, count=self.comm.size
        )

        request.Wait()
        request.free()
        received_bytes = bytearray(sum(recv_counts) * self.n_bytes)
        offsets = np.fromiter((np.sum(recv_counts[:i]) for i in range(self.comm.size)), dtype=int, count=self.comm.size)

        # numpy arrays of bytes do not play nicely with MPI, sometimes datacorruption happens.
        # bytearrays seem to work though...
        # Do not use Ialltoallv with bytearrays though, the call seems to simply freeze
        self.comm.Alltoallv(
            (
                bytearray().join(state.to_bytearray() for state_list in send_list for state in state_list),
                send_counts * self.n_bytes,
                send_offsets * self.n_bytes,
                MPI.BYTE,
            ),
            (received_bytes, recv_counts * self.n_bytes, offsets * self.n_bytes, MPI.BYTE),
        )

        if not flatten:
            states: list[Iterable[SlaterDeterminant]] = [() for _ in range(len(send_list))]
            # states: list[Iterable[bytes]] = [()] * len(send_list)
            start = 0
            for r in range(len(recv_counts)):
                if recv_counts[r] == 0:
                    continue
                states[r] = [
                    SlaterDeterminant.from_bytes(r_bytes)
                    # bytes(r_bytes)
                    for r_bytes in batched(received_bytes[start : start + recv_counts[r] * self.n_bytes], self.n_bytes)
                ]
                start += recv_counts[r] * self.n_bytes
        else:
            states: Iterable[SlaterDeterminant] = [
                SlaterDeterminant.from_bytes(r_bytes) for r_bytes in batched(received_bytes, self.n_bytes)
            ]
        return states


class SimpleDistributedStateContainer(StateContainer):
    """
    A distributed state container that uses sparse point-to-point communication.

    This container distributes states across MPI ranks based on their hash value,
    using point-to-point MPI communication for index lookups and retrieval.
    """

    def _point2point(send_list, comm):
        """
        Perform point-to-point MPI exchange of data.

        Parameters
        ----------
        send_list : list of list
            Data lists to send to each MPI rank.
        comm : MPI.Comm
            MPI communicator.

        Returns
        -------
        list of list
            Data lists received from each MPI rank.
        """
        return graph_alltoall(send_list, comm)

    def __init__(self, states: Iterable, bytes_per_state, state_type=SlaterDeterminant, comm=None, verbose=True):
        """
        Initialize the SimpleDistributedStateContainer.

        Parameters
        ----------
        states : Iterable
            Initial collection of states.
        bytes_per_state : int
            Number of bytes representing a single state.
        state_type : type, optional
            The class representing the state. Default is SlaterDeterminant.
        comm : MPI.Comm or None, optional
            MPI communicator. Default is None.
        verbose : bool, optional
            If True, print verbose messages. Default is True.
        """
        self.rng = np.random.default_rng()
        super(SimpleDistributedStateContainer, self).__init__(states, bytes_per_state, state_type, comm)

    def _set_state_bounds(self, local_states) -> list[Optional[SlaterDeterminant]]:
        """
        Determine state boundaries for partitioning states across ranks.

        Parameters
        ----------
        local_states : Iterable
            The local states on this rank.

        Returns
        -------
        list of SlaterDeterminant or None
            A list containing boundary states for each rank.
        """
        local_states_list = list(local_states)
        total_local_states_len = self.comm.allreduce(len(local_states_list), op=MPI.SUM)
        samples = []
        if len(local_states_list) > 1:
            n_samples = min(len(local_states_list), int(self.comm.size * np.log10(total_local_states_len) / 0.05**2))
            for interval in batched(local_states_list, len(local_states_list) // n_samples):
                samples.append(self.rng.choice(list(interval)))
        else:
            samples = local_states_list

        all_states_received = self.comm.gather(samples)

        if self.comm.rank == 0:
            all_states_it = merge(*all_states_received)
            all_states = []
            for state, _ in itertools.groupby(all_states_it, key=lambda state: hash_key(state)):
                all_states.append(state)

            sizes = np.array([len(all_states) // self.comm.size] * self.comm.size, dtype=int)
            sizes[: len(all_states) % self.comm.size] += 1

            bounds = (sum(sizes[: i + 1]) for i in range(self.comm.size))
            state_bounds = [all_states[bound] if bound < len(all_states) else all_states[-1] for bound in bounds]
        else:
            state_bounds = [None] * self.comm.size

        state_bounds = self.comm.bcast(state_bounds, root=0)
        return [
            state_bounds[r] if r < self.comm.size - 1 and state_bounds[r] != state_bounds[r + 1] else None
            for r in range(self.comm.size)
        ]

    def add_states(self, new_states: Iterable[SlaterDeterminant], unique_sorted=False) -> None:
        """
        Extend the current basis by adding the new_states to it.
        """
        new_states = [self.type.from_bytes(state) if isinstance(state, bytes) else state for state in new_states]
        if not self.is_distributed:
            existing_set = self._index_dict._map
            unique_new = [s for s in sorted(set(new_states)) if s not in existing_set]
            if unique_new:
                self.local_basis = list(merge(self.local_basis, unique_new))
                self.size = len(self.local_basis)
                self.offset = 0
                self.local_indices = range(0, len(self.local_basis))
                self._index_dict = StateContainer.IndexDict(self.local_basis, self.offset)
                self.local_index_bounds = [(0, len(self.local_basis))]
                if len(self.local_basis) > 0:
                    self.state_bounds: Optional[SlaterDeterminant] = [None]
                else:
                    self.state_bounds = [None]
                if __debug__:
                    assert all(self.local_basis[i] < self.local_basis[i + 1] for i in range(len(self.local_basis) - 1))
            return

        from impurityModel.ed.mpi_comm import distribute_determinants

        unique_new_states = list(set(new_states))
        received_list = distribute_determinants(unique_new_states, self.n_bytes, self.comm)

        all_received = []
        for r_data in received_list:
            if r_data:
                all_received.extend(r_data)

        existing_set = self._index_dict._map
        unique_received = sorted(set(all_received))
        unique_new = [s for s in unique_received if s not in existing_set]

        local_added = len(unique_new)
        any_added = self.comm.allreduce(local_added, op=MPI.SUM)
        if any_added == 0:
            return

        if unique_new:
            self.local_basis = list(merge(self.local_basis, unique_new))

        size_arr = np.empty((self.comm.size,), dtype=int)
        local_length = len(self.local_basis)
        size_arr = np.array(self.comm.allgather(local_length), dtype=int)
        self.size = np.sum(size_arr)
        self.offset = np.sum(size_arr[: self.comm.rank])
        self.local_indices = range(self.offset, self.offset + len(self.local_basis))
        self.index_bounds = [np.sum(size_arr[: r + 1]) if size_arr[r] > 0 else None for r in range(self.comm.size)]
        self._index_dict = StateContainer.IndexDict(self.local_basis, self.offset)
        state_bounds = list(self._getitem_sequence([i for i in self.index_bounds if i is not None and i < self.size]))
        self.state_bounds = state_bounds + [None] * (self.comm.size - len(state_bounds))
        self.state_bounds = [
            (
                self.state_bounds[r]
                if r < self.comm.size - 1 and self.state_bounds[r] != self.state_bounds[r + 1]
                else None
            )
            for r in range(self.comm.size)
        ]
        if __debug__:
            assert all(self.local_basis[i] < self.local_basis[i + 1] for i in range(len(self.local_basis) - 1))

    def _getitem_sequence(self, l: Iterable[int]) -> Iterable[SlaterDeterminant]:
        """
        Retrieve states corresponding to a sequence of global indices.

        Parameters
        ----------
        l : Iterable[int]
            The global indices to retrieve.

        Returns
        -------
        Iterable[SlaterDeterminant]
            The states corresponding to the requested indices.
        """
        if not self.is_distributed:
            return (self.local_basis[i] for i in l)

        l = np.fromiter((i if i >= 0 else self.size + i for i in l), dtype=int)

        send_list: list[list[int]] = [[] for _ in range(self.comm.size)]
        send_to_ranks = np.empty((len(l)), dtype=int)
        send_to_ranks[:] = self.size
        for idx, i in enumerate(l):
            for r in range(self.comm.size):
                if self.index_bounds[r] is not None and i < self.index_bounds[r]:
                    send_list[r].append(i)
                    send_to_ranks[idx] = r
                    break
        send_order = np.argsort(send_to_ranks, kind="stable")

        queries = SimpleDistributedStateContainer._point2point(send_list, self.comm)

        results = [[] for _ in range(self.comm.size)]
        for r in range(len(queries)):
            for i, query in enumerate(queries[r]):
                if query >= self.offset and query < self.offset + len(self.local_basis):
                    results[r].append(self.local_basis[query - self.offset])

        result = [
            state
            for r_results in SimpleDistributedStateContainer._point2point(results, self.comm)
            for state in r_results
        ]

        return (result[i] for i in np.argsort(send_order))

    def _index_sequence(self, s: Iterable[SlaterDeterminant]) -> Iterable[int]:
        """
        Find global indices for a sequence of states.

        Parameters
        ----------
        s : Iterable[SlaterDeterminant]
            The states to look up.

        Returns
        -------
        Iterable[int]
            The global indices corresponding to each state.
        """
        if not self.is_distributed:
            return (self._index_dict.get(val, self.size) for val in s)
            # return (self._index_dict[val] if val in self._index_dict else self.size for val in s)

        s = list(s)
        send_list: list[list[SlaterDeterminant]] = [[] for _ in range(self.comm.size)]
        send_to_ranks = np.empty((len(s)), dtype=int)
        send_to_ranks[:] = self.size
        for i, val in enumerate(s):
            r = val.get_hash() % self.comm.size
            send_list[r].append(val)
            send_to_ranks[i] = r
            # for r in range(self.comm.size):
            #     if self.state_bounds[r] is None or val < self.state_bounds[r]:
            #         send_list[r].append(val)
            #         send_to_ranks[i] = r
            #         break

        send_order = np.argsort(send_to_ranks, kind="stable")

        queries = SimpleDistributedStateContainer._point2point(send_list, self.comm)

        results = [[] for _ in range(self.comm.size)]
        for r in range(self.comm.size):
            for query in queries[r]:
                results[r].append(self._index_dict.get(query, self.size))
        result = np.array(
            [i for r_i in SimpleDistributedStateContainer._point2point(results, self.comm) for i in r_i], dtype=int
        )
        if len(result) > 0:
            max_retries = 3
            retry_count = 0
            while np.any(np.logical_or(result > self.size, result < 0)) and retry_count < max_retries:
                mask = np.logical_or(result > self.size, result < 0)
                result[mask] = np.fromiter(
                    self._index_sequence(itertools.compress(s, mask)), dtype=int, count=int(np.sum(mask))
                )
                retry_count += 1

            if retry_count >= max_retries:
                import warnings

                warnings.warn(f"Failed to resolve all indices after {max_retries} retries")

        return (res for res in result[np.argsort(send_order)])

    def _contains_sequence(self, items) -> Iterable[bool]:
        """
        Check membership for a sequence of states.

        Parameters
        ----------
        items : Iterable[SlaterDeterminant]
            The states to check.

        Returns
        -------
        Iterable[bool]
            An iterable of booleans indicating if each state is present.
        """
        if not self.is_distributed:
            return (item in self._index_dict for item in items)
        return (index < self.size for index in self._index_sequence(items))

    def alltoall_states(self, send_list: list[list[SlaterDeterminant]], flatten=False):
        """
        Exchange states using sparse point-to-point all-to-all communication.

        Parameters
        ----------
        send_list : list of list of SlaterDeterminant
            Outer list of size comm.size, where element `r` is a list of
            states to send to rank `r`.
        flatten : bool, optional
            If True, the received states are flattened into a single list.
            If False, returns a list of iterables of states, one per rank.
            Default is False.

        Returns
        -------
        list of list of SlaterDeterminant or list of SlaterDeterminant
            The received states.
        """
        states = SimpleDistributedStateContainer._point2point(send_list, self.comm)
        if flatten:
            states = [state for r_states in states for state in r_states]
        return states

        # return (idx != len(self._full_basis) for idx in self._index_sequence(items))
