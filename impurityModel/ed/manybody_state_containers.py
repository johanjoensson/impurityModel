from bisect import bisect_left
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
from impurityModel.ed import product_state_representation as psr


def batched(iterable: Iterable, n: int) -> Iterable:
    """
    batched('ABCDEFG', 3) → ABC DEF G
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


class StateContainer:
    def __init__(self, states, bytes_per_state, state_type, comm):
        self.local_basis = []
        self.comm = comm
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict = {}
        self.type = state_type
        self.n_bytes = bytes_per_state
        self.is_distributed = comm is not None
        self.index_bounds = [None] * comm.size if self.is_distributed else [None]
        self.state_bounds = [None] * comm.size if self.is_distributed else [None]
        self.add_states(states)
        if self.size > 0 and self.type is None:
            self.type = type(self[0])

    def __iter__(self):
        for i in range(self.size):
            yield self.__getitem__(i)

    def add_states(self, new_states: Iterable[bytes]) -> None:
        pass

    def __getitem__(self, key) -> Iterable[bytes]:
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
                if res == bytes(0 for _ in range(self.n_bytes)):
                    raise IndexError(f"Could not find index {query[i]} in basis with size {self.size}!")
            return (state for state in result)
        elif isinstance(key, Sequence) or isinstance(key, Iterable):
            result = list(self._getitem_sequence(key))
            for i, res in enumerate(result):
                if res == psr.int2bytes(0, self.num_spin_orbitals):
                    raise IndexError(f"Could not find index {key[i]} in basis with size {self.size}!")
            return (state for state in result)
        elif isinstance(key, int):
            result = next(self._getitem_sequence([key]))
            # if result is None:
            if result == bytes(0 for _ in range(self.n_bytes)):
                raise IndexError(f"Could not find index {key} in basis with size {self.size}!")
            return result
        else:
            raise TypeError(f"Invalid index type {type(key)}. Valid types are slice, Sequence and int")
        return None

    def __len__(self):
        return self.size

    def index(self, val):
        if isinstance(val, self.type):
            res = next(self._index_sequence([val]))
            if res == self.size:
                raise ValueError(f"Could not find {val} in basis!")
            return res
        elif isinstance(val, Sequence) or isinstance(val, Iterable):
            res = list(self._index_sequence(val))
            for i, v in enumerate(res):
                if v >= self.size:
                    raise ValueError(f"Could not find {val[i]} in basis!")
            return (i for i in res)
        else:
            raise TypeError(f"Invalid query type {type(val)}! Valid types are {self.dtype} and sequences thereof.")
        return None

    def _index_sequence(self, s: Iterable[bytes]) -> Iterable[int]:
        pass

    def contains(self, item) -> Iterable[bool]:
        if isinstance(item, self.type):
            return next(self._contains_sequence([item]))
        elif isinstance(item, Sequence):
            return self._contains_sequence(item)
        elif isinstance(item, Iterable):
            return self._contains_sequence(item)
        return None

    def __contains__(self, item):
        if self.comm is None:
            return item in self._index_dict
        return next(self._index_sequence([item])) != self.size

    def _contains_sequence(self, items):
        pass

    def clear(self):
        self.local_basis.clear()
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict = {}
        self.index_bounds = [None] * self.comm.size if self.is_distributed else None
        self.state_bounds = [None] * self.comm.size if self.is_distributed else None

    def alltoall_states(self, send_list: list[list[bytes]], flatten=False):
        recv_counts = np.empty((self.comm.size), dtype=int)
        request = self.comm.Ialltoall(
            (np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list)), MPI.LONG), recv_counts
        )

        send_counts = np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list))
        send_offsets = np.fromiter(
            (np.sum(send_counts[:i]) for i in range(self.comm.size)), dtype=int, count=self.comm.size
        )

        request.Wait()
        received_bytes = bytearray(sum(recv_counts) * self.n_bytes)
        offsets = np.fromiter((np.sum(recv_counts[:i]) for i in range(self.comm.size)), dtype=int, count=self.comm.size)

        # numpy arrays of bytes do not play nicely with MPI, sometimes datacorruption happens.
        # bytearrays seem to work though...
        # Do not use Ialltoallv with bytearrays though, the call seems to simply freeze
        self.comm.Alltoallv(
            (
                bytearray(byte for state_list in send_list for state in state_list for byte in state),
                send_counts * self.n_bytes,
                send_offsets * self.n_bytes,
                MPI.BYTE,
            ),
            (received_bytes, recv_counts * self.n_bytes, offsets * self.n_bytes, MPI.BYTE),
        )

        if not flatten:
            states: list[Iterable[bytes]] = [()] * len(send_list)
            start = 0
            for r in range(len(recv_counts)):
                if recv_counts[r] == 0:
                    continue
                states[r] = [
                    bytes(r_bytes)
                    for r_bytes in batched(received_bytes[start : start + recv_counts[r] * self.n_bytes], self.n_bytes)
                ]
                start += recv_counts[r] * self.n_bytes
        else:
            states: Iterable[bytes] = [bytes(r_bytes) for r_bytes in batched(received_bytes, self.n_bytes)]
        return states


class DistributedStateContainer(StateContainer):
    def __init__(self, states: Iterable, bytes_per_state, state_type=bytes, comm=None, verbose=True):
        super(DistributedStateContainer, self).__init__(states, bytes_per_state, state_type, comm)

    def _set_state_bounds(self, local_states) -> list[Optional[bytes]]:
        local_states_list = local_states
        total_local_states_len = self.comm.allreduce(len(local_states_list), op=MPI.SUM)
        samples = []
        if len(local_states) > 1:
            n_samples = min(len(local_states), int(self.comm.size * np.log10(total_local_states_len) / 0.05**2))
            for interval in batched(local_states, len(local_states) // n_samples):
                samples.append(self.rng.choice(list(interval)))
        else:
            samples = local_states_list

        samples_count = np.empty((self.comm.size), dtype=int)
        self.comm.Gather((np.array([len(samples)], dtype=int), MPI.LONG), samples_count, root=0)

        all_samples_bytes = bytearray(0)
        offsets = np.array([0], dtype=int)
        if self.comm.rank == 0:
            all_samples_bytes = bytearray(sum(samples_count) * self.n_bytes)
            offsets = np.fromiter(
                (np.sum(samples_count[:i]) for i in range(self.comm.size)), dtype=int, count=self.comm.size
            )

        self.comm.Gatherv(
            (
                bytearray(byte for state in samples for byte in state),
                MPI.BYTE,
            ),
            (all_samples_bytes, samples_count * self.n_bytes, offsets * self.n_bytes, MPI.BYTE),
            root=0,
        )

        if self.comm.rank == 0:
            if sum(samples_count) == 0:
                state_bounds = [psr.int2bytes(0, self.num_spin_orbitals)] * self.comm.size
            else:
                all_states_received = (
                    [
                        bytes(
                            all_samples_bytes[
                                (sum(samples_count[:r]) + i)
                                * self.n_bytes : (sum(samples_count[:r]) + i + 1)
                                * self.n_bytes
                            ]
                        )
                        for i in range(sc)
                    ]
                    for r, sc in enumerate(samples_count)
                )
                all_states_it = merge(*all_states_received)
                all_states = []
                for state, _ in itertools.groupby(all_states_it):
                    all_states.append(state)

                sizes = np.array([len(all_states) // self.comm.size] * self.comm.size, dtype=int)
                sizes[: len(all_states) % self.comm.size] += 1

                bounds = (sum(sizes[: i + 1]) for i in range(self.comm.size))
                state_bounds = (all_states[bound] if bound < len(all_states) else all_states[-1] for bound in bounds)
            state_bounds_bytes = bytearray(byte for state in state_bounds for byte in state)
        else:
            state_bounds_bytes = bytearray(self.comm.size * self.n_bytes)
            state_bounds = None

        self.comm.Bcast(state_bounds_bytes, root=0)
        state_bounds: list[Optional[bytes]] = [
            bytes(state_bounds_bytes[i * self.n_bytes : (i + 1) * self.n_bytes]) for i in range(self.comm.size)
        ]
        return [
            state_bounds[r] if r < self.comm.size - 1 and state_bounds[r] != state_bounds[r + 1] else None
            for r in range(self.comm.size)
        ]

    def add_states(self, new_states: Iterable[bytes]) -> None:
        """
        Extend the current basis by adding the new_states to it.
        """
        if not self.is_distributed:
            local_it = merge(self.local_basis, sorted(set(new_states)))
            local_basis: list[bytes] = []
            for state, _ in itertools.groupby(local_it):
                local_basis.append(state)
            self.local_basis = local_basis
            self.size = len(self.local_basis)
            self.offset = 0
            self.local_indices = range(0, len(self.local_basis))
            self._index_dict = {state: i for i, state in enumerate(self.local_basis)}
            self.local_index_bounds = [(0, len(self.local_basis))]
            if len(self.local_basis) > 0:
                self.state_bounds: Optional[tuple[bytes, Optional[bytes]]] = [(self.local_basis[0], None)]
            else:
                self.state_bounds = [None]
            return

        local_it = merge(self.local_basis, sorted(set(new_states)))
        local_states = []
        for state, _ in itertools.groupby(local_it):
            local_states.append(state)
        local_sizes = np.empty((self.comm.size,), dtype=int)
        self.comm.Allgather(np.array([len(self.local_basis)], dtype=int), local_sizes)

        send_list: list[list[bytes]] = [[] for _ in range(self.comm.size)]
        treated_states = [False] * len(local_states)
        for (i, state), (r, state_bounds) in itertools.product(enumerate(local_states), enumerate(self.state_bounds)):
            if treated_states[i]:
                continue
            if state_bounds is None or state < state_bounds:
                send_list[r].append(state)
                treated_states[i] = True

        recv_counts = np.empty((self.comm.size), dtype=np.int64)
        request = self.comm.Ialltoall(
            (np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list)), MPI.LONG), recv_counts
        )

        send_counts = np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list))
        send_offsets = np.fromiter(
            (sum(send_counts[:i]) for i in range(self.comm.size)), dtype=np.int64, count=self.comm.size
        )

        request.Wait()
        received_bytes = bytearray(sum(recv_counts) * self.n_bytes)
        offsets = np.fromiter(
            (sum(recv_counts[:i]) for i in range(self.comm.size)), dtype=np.int64, count=self.comm.size
        )

        request = self.comm.Ialltoallv(
            [
                bytes(byte for states in send_list for state in states for byte in state),
                send_counts * self.n_bytes,
                send_offsets * self.n_bytes,
                MPI.BYTE,
            ],
            [received_bytes, recv_counts * self.n_bytes, offsets * self.n_bytes, MPI.BYTE],
        )
        request.Wait()
        received_states = []
        if sum(recv_counts) > 0:
            received_states = []
            offset = 0
            for r in range(self.comm.size):
                received_states.append(
                    [
                        bytes(received_bytes[(offset + i) * self.n_bytes : (offset + i + 1) * self.n_bytes])
                        for i in range(recv_counts[r])
                    ]
                )
                offset += recv_counts[r]

        local_basis = []
        for state, _ in itertools.groupby(merge(*received_states)):
            local_basis.append(state)

        size_arr = np.empty((self.comm.size,), dtype=int)
        self.local_basis = local_basis
        local_length = len(self.local_basis)
        self.comm.Allgather(np.array([local_length], dtype=int), size_arr)
        self.size = np.sum(size_arr)
        self.offset = np.sum(size_arr[: self.comm.rank])  # offset_arr[0] - local_length
        self.local_indices = range(self.offset, self.offset + len(self.local_basis))
        self._index_dict = {state: self.offset + i for i, state in enumerate(self.local_basis)}
        self.index_bounds = [np.sum(size_arr[: r + 1]) if size_arr[r] > 0 else None for r in range(self.comm.size)]
        if self.size > 0 and any(abs(size_arr - self.size // self.comm.size) / self.size > 0.10):
            n_states_per_rank = np.array(
                [
                    self.size // self.comm.size + (1 if r < self.size % self.comm.size else 0)
                    for r in range(self.comm.size)
                ]
            )
            local_indices = range(
                sum(n_states_per_rank[: self.comm.rank]), sum(n_states_per_rank[: self.comm.rank + 1])
            )
            self.local_basis = list(self._getitem_sequence([i for i in local_indices if i < self.size]))
            local_length = len(self.local_basis)
            self.comm.Allgather(np.array([local_length], dtype=int), size_arr)
            self.size = np.sum(size_arr)
            self.offset = np.sum(size_arr[: self.comm.rank])  # offset_arr[0] - local_length
            self.local_indices = range(self.offset, self.offset + local_length)
            self._index_dict = {state: self.offset + i for i, state in enumerate(self.local_basis)}
            self.index_bounds = [np.sum(size_arr[: r + 1]) if size_arr[r] > 0 else None for r in range(self.comm.size)]
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

    def _getitem_sequence(self, l: Iterable[int]) -> Iterable[bytes]:
        if self.comm is None:
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
        recv_counts = np.empty((self.comm.size), dtype=int)

        request = self.comm.Ialltoall(
            (np.fromiter((len(sl) for sl in send_list), dtype=int, count=len(send_list)), MPI.LONG), recv_counts
        )

        send_counts = np.fromiter((len(sl) for sl in send_list), dtype=int, count=len(send_list))
        send_offsets = np.fromiter(
            (sum(send_counts[:r]) for r in range(self.comm.size)), dtype=int, count=self.comm.size
        )

        request.Wait()

        queries = np.empty((sum(recv_counts)), dtype=int)
        displacements = np.fromiter(
            (sum(recv_counts[:p]) for p in range(self.comm.size)), dtype=int, count=self.comm.size
        )

        self.comm.Alltoallv(
            (
                np.fromiter((i for sl in send_list for i in sl), dtype=int, count=len(l)),
                send_counts,
                send_offsets,
                MPI.LONG,
            ),
            (queries, recv_counts, displacements, MPI.LONG),
        )

        results = bytearray(sum(recv_counts) * self.n_bytes)
        for i, query in enumerate(queries):
            if query >= self.offset and query < self.offset + len(self.local_basis):
                results[i * self.n_bytes : (i + 1) * self.n_bytes] = self.local_basis[query - self.offset]
        result = bytearray(len(l) * self.n_bytes)

        self.comm.Alltoallv(
            (results, recv_counts * self.n_bytes, displacements * self.n_bytes, MPI.BYTE),
            (result, send_counts * self.n_bytes, send_offsets * self.n_bytes, MPI.BYTE),
        )

        return (bytes(result[i * self.n_bytes : (i + 1) * self.n_bytes]) for i in np.argsort(send_order))

    def _index_sequence(self, s: Iterable[bytes]) -> Iterable[int]:
        if self.comm is None:
            return (self._index_dict[val] if val in self._index_dict else self.size for val in s)

        send_list: list[list[bytes]] = [[] for _ in range(self.comm.size)]
        send_to_ranks = np.empty((len(s)), dtype=int)
        send_to_ranks[:] = self.size
        for i, val in enumerate(s):
            for r in range(self.comm.size):
                if self.state_bounds[r] is None or val < self.state_bounds[r]:
                    send_list[r].append(val)
                    send_to_ranks[i] = r
                    break

        send_order = np.argsort(send_to_ranks, kind="stable")
        recv_counts = np.empty((self.comm.size), dtype=int)
        send_counts = np.fromiter((len(send_list[r]) for r in range(self.comm.size)), dtype=int, count=self.comm.size)
        send_displacements = np.fromiter(
            (sum(send_counts[:i]) for i in range(self.comm.size)), dtype=int, count=self.comm.size
        )

        self.comm.Alltoall(
            (
                np.fromiter((len(send_list[r]) for r in range(self.comm.size)), dtype=int, count=self.comm.size),
                MPI.LONG,
                # MPI.INT64_T,
            ),
            recv_counts,
        )

        queries = bytearray(sum(recv_counts) * self.n_bytes)
        displacements = np.fromiter(
            (sum(recv_counts[:p]) for p in range(self.comm.size)), dtype=int, count=self.comm.size
        )

        self.comm.Alltoallv(
            (
                bytearray(byte for states in send_list for state in states for byte in state),
                send_counts * self.n_bytes,
                send_displacements * self.n_bytes,
                MPI.BYTE,
            ),
            (queries, recv_counts * self.n_bytes, displacements * self.n_bytes, MPI.BYTE),
        )

        results = np.empty((sum(recv_counts)), dtype=int)
        for i in range(sum(recv_counts)):
            query = bytes(queries[i * self.n_bytes : (i + 1) * self.n_bytes])
            results[i] = self._index_dict.get(query, self.size)
        result = np.empty((len(s)), dtype=int)
        result[:] = self.size

        self.comm.Alltoallv(
            (results, recv_counts, displacements, MPI.LONG), (result, send_counts, send_displacements, MPI.LONG)
        )
        result[sum(send_counts) :] = self.size
        while np.any(np.logical_or(result > self.size, result < 0)):
            mask = np.logical_or(result > self.size, result < 0)
            result[mask] = np.from_iter(self._index_sequence(itertools.compress(s, mask)), dtype=int)

        return (res for res in result[np.argsort(send_order)])

    def _contains_sequence(self, items) -> Iterable[bool]:
        if self.comm is None:
            return (item in self._index_dict for item in items)
        return (index < self.size for index in self._index_sequence(items))

    def alltoall_states(self, send_list: list[list[bytes]], flatten=False):
        recv_counts = np.empty((self.comm.size), dtype=int)
        request = self.comm.Ialltoall(
            (np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list)), MPI.LONG), recv_counts
        )

        send_counts = np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list))
        send_offsets = np.fromiter(
            (np.sum(send_counts[:i]) for i in range(self.comm.size)), dtype=int, count=self.comm.size
        )

        request.Wait()
        received_bytes = bytearray(sum(recv_counts) * self.n_bytes)
        offsets = np.fromiter((np.sum(recv_counts[:i]) for i in range(self.comm.size)), dtype=int, count=self.comm.size)

        # numpy arrays of bytes do not play nicely with MPI, sometimes datacorruption happens.
        # bytearrays seem to work though...
        # Do not use Ialltoallv with bytearrays though, the call seems to simply freeze
        self.comm.Alltoallv(
            (
                bytearray(byte for state_list in send_list for state in state_list for byte in state),
                send_counts * self.n_bytes,
                send_offsets * self.n_bytes,
                MPI.BYTE,
            ),
            (received_bytes, recv_counts * self.n_bytes, offsets * self.n_bytes, MPI.BYTE),
        )

        if not flatten:
            states: list[Iterable[bytes]] = [()] * len(send_list)
            start = 0
            for r in range(len(recv_counts)):
                if recv_counts[r] == 0:
                    continue
                states[r] = [
                    bytes(r_bytes)
                    for r_bytes in batched(received_bytes[start : start + recv_counts[r] * self.n_bytes], self.n_bytes)
                ]
                start += recv_counts[r] * self.n_bytes
        else:
            states: Iterable[bytes] = [bytes(r_bytes) for r_bytes in batched(received_bytes, self.n_bytes)]
        return states


class SimpleDistributedStateContainer(StateContainer):
    def __init__(self, states: Iterable, bytes_per_state, state_type=bytes, comm=None, verbose=True):
        super(SimpleDistributedStateContainer, self).__init__(states, bytes_per_state, state_type, comm)

    def _set_state_bounds(self, local_states) -> list[Optional[bytes]]:
        local_states_list = local_states
        total_local_states_len = self.comm.allreduce(len(local_states_list), op=MPI.SUM)
        samples = []
        if len(local_states) > 1:
            n_samples = min(len(local_states), int(self.comm.size * np.log10(total_local_states_len) / 0.05**2))
            for interval in batched(local_states, len(local_states) // n_samples):
                samples.append(self.rng.choice(list(interval)))
        else:
            samples = local_states_list

        all_states_received = self.comm.gather(samples)

        if self.comm.rank == 0:
            all_states_it = merge(*all_states_received)
            all_states = []
            for state, _ in itertools.groupby(all_states_it):
                all_states.append(state)

            sizes = np.array([len(all_states) // self.comm.size] * self.comm.size, dtype=int)
            sizes[: len(all_states) % self.comm.size] += 1

            bounds = (sum(sizes[: i + 1]) for i in range(self.comm.size))
            state_bounds = (all_states[bound] if bound < len(all_states) else all_states[-1] for bound in bounds)
        else:
            state_bounds = None

        state_bounds = self.comm.bcast(state_bounds)
        return [
            state_bounds[r] if r < self.comm.size - 1 and state_bounds[r] != state_bounds[r + 1] else None
            for r in range(self.comm.size)
        ]

    def add_states(self, new_states: Iterable[bytes]) -> None:
        """
        Extend the current basis by adding the new_states to it.
        """
        if not self.is_distributed:
            local_it = merge(self.local_basis, sorted(set(new_states)))
            local_basis: list[bytes] = []
            for state, _ in itertools.groupby(local_it):
                local_basis.append(state)
            self.local_basis = local_basis
            self.size = len(self.local_basis)
            self.offset = 0
            self.local_indices = range(0, len(self.local_basis))
            self._index_dict = {state: i for i, state in enumerate(self.local_basis)}
            self.local_index_bounds = [(0, len(self.local_basis))]
            if len(self.local_basis) > 0:
                self.state_bounds: Optional[tuple[bytes, Optional[bytes]]] = [(self.local_basis[0], None)]
            else:
                self.state_bounds = [None]
            return

        local_it = merge(self.local_basis, sorted(set(new_states)))
        local_states = []
        for state, _ in itertools.groupby(local_it):
            local_states.append(state)

        send_list: list[list[bytes]] = [[] for _ in range(self.comm.size)]
        treated_states = [False] * len(local_states)
        for (i, state), (r, state_bounds) in itertools.product(enumerate(local_states), enumerate(self.state_bounds)):
            if treated_states[i]:
                continue
            if state_bounds is None or state < state_bounds:
                send_list[r].append(state)
                treated_states[i] = True

        received_states = self.comm.alltoall(send_list)

        local_basis = []
        for state, _ in itertools.groupby(merge(*received_states)):
            local_basis.append(state)

        size_arr = np.empty((self.comm.size,), dtype=int)
        self.local_basis = local_basis
        local_length = len(self.local_basis)
        size_arr = np.array(self.comm.allgather(local_length), dtype=int)
        self.size = np.sum(size_arr)
        self.offset = np.sum(size_arr[: self.comm.rank])  # offset_arr[0] - local_length
        self.local_indices = range(self.offset, self.offset + len(self.local_basis))
        self._index_dict = {state: self.offset + i for i, state in enumerate(self.local_basis)}
        self.index_bounds = [np.sum(size_arr[: r + 1]) if size_arr[r] > 0 else None for r in range(self.comm.size)]
        if self.size > 0 and any(abs(size_arr - self.size // self.comm.size) / self.size > 0.10):
            n_states_per_rank = np.array(
                [
                    self.size // self.comm.size + (1 if r < self.size % self.comm.size else 0)
                    for r in range(self.comm.size)
                ]
            )
            local_indices = range(
                sum(n_states_per_rank[: self.comm.rank]), sum(n_states_per_rank[: self.comm.rank + 1])
            )
            self.local_basis = list(self._getitem_sequence([i for i in local_indices if i < self.size]))
            local_length = len(self.local_basis)
            size_arr = np.array(self.comm.allgather(local_length), dtype=int)
            self.size = int(np.sum(size_arr))
            self.offset = int(np.sum(size_arr[: self.comm.rank]))  # offset_arr[0] - local_length
            self.local_indices = range(self.offset, self.offset + local_length)
            self._index_dict = {state: self.offset + i for i, state in enumerate(self.local_basis)}
            self.index_bounds = [np.sum(size_arr[: r + 1]) if size_arr[r] > 0 else None for r in range(self.comm.size)]
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

    def _getitem_sequence(self, l: Iterable[int]) -> Iterable[bytes]:
        if self.comm is None:
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

        queries = self.comm.alltoall(send_list)

        results = [[] for _ in range(self.comm.size)]
        for r in range(len(queries)):
            for i, query in enumerate(queries[r]):
                if query >= self.offset and query < self.offset + len(self.local_basis):
                    results[r].append(self.local_basis[query - self.offset])

        result = self.comm.alltoall(results)
        result = [state for r_results in result for state in r_results]

        return (result[i] for i in np.argsort(send_order))

    def _index_sequence(self, s: Iterable[bytes]) -> Iterable[int]:
        if self.comm is None:
            return (self._index_dict[val] if val in self._index_dict else self.size for val in s)

        send_list: list[list[bytes]] = [[] for _ in range(self.comm.size)]
        send_to_ranks = np.empty((len(s)), dtype=int)
        send_to_ranks[:] = self.size
        for i, val in enumerate(s):
            for r in range(self.comm.size):
                if self.state_bounds[r] is None or val < self.state_bounds[r]:
                    send_list[r].append(val)
                    send_to_ranks[i] = r
                    break

        send_order = np.argsort(send_to_ranks, kind="stable")

        queries = self.comm.alltoall(send_list)

        results = [[] for _ in range(self.comm.size)]
        for r in range(self.comm.size):
            for query in queries[r]:
                results[r].append(self._index_dict.get(query, self.size))
        result = self.comm.alltoall(results)
        result = np.array([i for r_i in result for i in r_i], dtype=int)
        while np.any(np.logical_or(result > self.size, result < 0)):
            mask = np.logical_or(result > self.size, result < 0)
            result[mask] = np.from_iter(self._index_sequence(itertools.compress(s, mask)), dtype=int)

        return (res for res in result[np.argsort(send_order)])

    def _contains_sequence(self, items) -> Iterable[bool]:
        if self.comm is None:
            return (item in self._index_dict for item in items)
        return (index < self.size for index in self._index_sequence(items))

    def alltoall_states(self, send_list: list[list[bytes]], flatten=False):
        states = self.comm.alltoall(send_list)
        if flatten:
            states = [state for r_states in states for state in r_states]
        return states


class CentralizedStateContainer(StateContainer):
    def __init__(self, states: Iterable, bytes_per_state, state_type=bytes, comm=None, verbose=True):
        self._full_basis = []
        super(CentralizedStateContainer, self).__init__(states, bytes_per_state, state_type, comm)

    def _set_state_bounds(self, local_states) -> list[Optional[bytes]]:
        local_sizes = (
            [self.size // self.comm.size + (1 if r < self.size % self.comm.size else 0) for r in range(self.comm.size)]
            if self.is_distributed
            else None
        )
        return (
            [self.local_basis[sum(local_sizes[: r + 1])] if local_sizes[r] > 0 else None for r in range(self.comm.size)]
            if self.is_distributed
            else [None]
        )

    def add_states(self, new_states) -> None:
        states_to_add = sorted(itertools.compress(new_states, self._contains_sequence(new_states)))
        if self.is_distributed:
            states_from_ranks = self.comm.allgather(states_to_add)
            merged_states_from_ranks = merge(*states_from_ranks)
            states_to_add = [state for state, _ in itertools.groupby(merged_states_from_ranks)]
        new_full_basis = merge(self._full_basis, states_to_add)
        # Remove duplicates from new_full_basis
        self._full_basis = [state for state, _ in itertools.groupby(new_full_basis)]
        self.size = len(self._full_basis)
        local_sizes = (
            [self.size // self.comm.size + (1 if r < self.size % self.comm.size else 0) for r in range(self.comm.size)]
            if self.is_distributed
            else None
        )
        self.offset = sum(local_sizes[: self.comm.rank]) if self.is_distributed else 0
        self.local_basis = (
            self._full_basis[self.offset : self.offset + local_sizes[self.comm.rank]]
            if self.is_distributed
            else self._full_basis
        )
        self.local_indices = (
            range(self.offset, self.offset + local_sizes[self.comm.rank])
            if self.is_distributed
            else range(0, self.size)
        )
        self._index_dict = {state: i for i, state in enumerate(self._full_basis)}
        # self._index_dict = {state: self.offset + i for i, state in enumerate(self.local_basis)}
        self.index_bounds = (
            [np.sum(local_sizes[: r + 1]) for r in range(self.comm.size)] if self.is_distributed else [self.size]
        )
        self.state_bounds = (
            [
                self._full_basis[sum(local_sizes[: r + 1])] if sum(local_sizes[: r + 1]) < self.size else None
                for r in range(self.comm.size - 1)
            ]
            + [None]
            if self.is_distributed
            else [None]
        )

    def _getitem_sequence(self, l: Iterable[int]) -> Iterable[bytes]:
        return (self._full_basis[i] for i in l)

    def _search_sorted(self, key):
        return bisect_left(self._full_basis, key)

    def _index_sequence(self, key: Iterable[bytes]) -> Iterable[int]:
        # return (self._search_sorted(k) for k in key)
        return (self._index_dict.get(k, len(self._full_basis)) for k in key)

    def _contains_sequence(self, items) -> Iterable[bool]:
        return (i in self._index_dict for i in items)
        # return (self._search_sorted(i) != self.size and self._full_basis[self._search_sorted(i)] == i for i in items)
