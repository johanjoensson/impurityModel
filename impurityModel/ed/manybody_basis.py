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


from impurityModel.ed import product_state_representation as psr
from impurityModel.ed.finite import applyOp_new as applyOp, c2i, c2i_op, eigensystem_new, norm2


def batched(iterable, n):
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
            conduction_indices = frozenset(
                c2i(total_baths, (l, b)) for b in range(valence_baths[l], valence_baths[l] + conduction_baths[l])
            )
            restrictions[conduction_indices] = (0, delta_conduction_occ[l] + 1)

            if verbose:
                print(f"l = {l}")
                print(f"|---Restrictions on the impurity orbitals = {restrictions[impurity_indices]}")
                print(f"|---Restrictions on the valence bath      = {restrictions[valence_indices]}")
                print(f"----Restrictions on the conduction bath   = {restrictions[conduction_indices]}")

        return restrictions

    def get_effective_restrictions(self):
        valence_baths, conduction_baths = self.bath_states

        total_baths = {l: valence_baths[l] + conduction_baths[l] for l in valence_baths}
        restrictions = {}
        for l in total_baths:
            max_imp = 0
            min_imp = 2 * (2 * l + 1)
            max_val = 0
            min_val = valence_baths[l]
            max_con = 0
            min_con = conduction_baths[l]
            impurity_indices = frozenset(c2i(total_baths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))
            valence_indices = frozenset(c2i(total_baths, (l, b)) for b in range(valence_baths[l]))
            conduction_indices = frozenset(
                c2i(total_baths, (l, b)) for b in range(valence_baths[l], valence_baths[l] + conduction_baths[l])
            )
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
            max_val = valence_baths[l]
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
        total_baths = {l: valence_baths[l] + conduction_baths[l] for l in valence_baths}
        restrictions = self.get_effective_restrictions()
        excited_restrictions = {}
        for l in total_baths:
            impurity_indices = frozenset(c2i(total_baths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))
            valence_indices = frozenset(c2i(total_baths, (l, b)) for b in range(valence_baths[l]))
            conduction_indices = frozenset(
                c2i(total_baths, (l, b)) for b in range(valence_baths[l], valence_baths[l] + conduction_baths[l])
            )
            r_min_imp, r_max_imp = restrictions[impurity_indices]
            min_imp = max(r_min_imp - imp_reduce, 0)
            max_imp = min(r_max_imp + imp_increase, 2 * (2 * l + 1))
            r_min_val, r_max_val = restrictions[valence_indices]
            min_val = max(r_min_val - val_reduce, 0)
            max_val = valence_baths[l]
            r_min_cond, r_max_cond = restrictions[conduction_indices]
            min_cond = 0
            max_cond = min(r_max_cond + con_increase, conduction_baths[l])
            excited_restrictions[impurity_indices] = (min_imp, max_imp)
            excited_restrictions[valence_indices] = (min_val, max_val)
            excited_restrictions[conduction_indices] = (min_cond, max_cond)
        return excited_restrictions

    def __init__(
        self,
        ls,
        bath_states=None,
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
        spin_flip_dj=False,
        tau=0,
        comm=None,
        verbose=True,
        debug=False,
    ):
        t0 = perf_counter()
        if initial_basis is not None:
            assert (
                num_spin_orbitals is not None
            ), "when supplying an initial basis, you also need to supply the num_spin_orbitals"
            assert (
                bath_states is not None
            ), "when supplying an initial basis, you also need to supply the number of bath states for each l quantum number"
            assert nominal_impurity_occ is None
            assert valence_baths is None
            assert conduction_baths is None
            assert delta_valence_occ is None
            assert delta_conduction_occ is None
            assert delta_impurity_occ is None
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
            bath_states = (valence_baths, conduction_baths)
        t0 = perf_counter() - t0
        t0 = perf_counter()
        self.ls = ls
        self.bath_states = bath_states
        self.spin_flip_dj = spin_flip_dj
        self.verbose = verbose
        self.debug = debug
        self.truncation_threshold = truncation_threshold
        self.comm = comm
        self.num_spin_orbitals = num_spin_orbitals
        self.local_basis = []
        self.restrictions = restrictions
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict = {}
        self.type = type(psr.int2bytes(0, self.num_spin_orbitals))
        self.n_bytes = int(ceil(self.num_spin_orbitals / 8))

        self.index_bounds = [None] * comm.size if comm is not None else None
        self.state_bounds = [None] * comm.size if comm is not None else None

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
        self.add_states(initial_basis)
        t0 = perf_counter() - t0

    def alltoall_states(self, send_list: list[list[bytes]], flatten=False):
        recv_counts = np.empty((self.comm.size), dtype=int)
        request = self.comm.Ialltoall(
            (np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list)), MPI.LONG),
            recv_counts
            # (np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list)), MPI.INT64_T), recv_counts
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
                    # bytes(received_bytes[start + i * self.n_bytes : start + (i + 1) * self.n_bytes])
                    # for i in range(recv_counts[r])
                ]
                start += recv_counts[r] * self.n_bytes
        else:
            states: Iterable[bytes] = [
                bytes(r_bytes)
                for r_bytes in batched(received_bytes, self.n_bytes)
                # bytes(received_bytes[i * self.n_bytes : (i + 1) * self.n_bytes]) for i in range(sum(recv_counts))
            ]
        return states

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
        # self.comm.Gather((np.array([len(samples)], dtype=int), MPI.INT64_T), samples_count, root=0)

        all_samples_bytes = bytearray(0)
        # all_samples_bytes = np.empty((0), dtype=np.ubyte)
        offsets = np.array([0], dtype=int)
        if self.comm.rank == 0:
            all_samples_bytes = bytearray(sum(samples_count) * self.n_bytes)
            # all_samples_bytes = np.empty((sum(samples_count) * self.n_bytes), dtype=np.ubyte)
            offsets = np.fromiter(
                (np.sum(samples_count[:i]) for i in range(self.comm.size)), dtype=int, count=self.comm.size
            )

        self.comm.Gatherv(
            (
                bytearray(byte for state in samples for byte in state),
                # np.array([byte for state in samples for byte in state], dtype=np.ubyte),
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
            # state_bounds_bytes = np.array([byte for state in state_bounds for byte in state], dtype=np.ubyte)
        else:
            state_bounds_bytes = bytearray(self.comm.size * self.n_bytes)
            # state_bounds_bytes = np.zeros((self.comm.size * self.n_bytes), dtype=np.ubyte)
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
            self.local_index_bounds = (0, len(self.local_basis))
            if len(self.local_basis) > 0:
                self.state_bounds: Optional[tuple[bytes, Optional[bytes]]] = (self.local_basis[0], None)
            else:
                self.state_bounds = None
            return

        t0 = perf_counter()
        local_it = merge(self.local_basis, sorted(set(new_states)))
        local_states = []
        for state, _ in itertools.groupby(local_it):
            local_states.append(state)
        t0 = perf_counter() - t0
        t0 = perf_counter()
        local_sizes = np.empty((self.comm.size,), dtype=int)
        self.comm.Allgather(np.array([len(self.local_basis)], dtype=int), local_sizes)
        t0 = perf_counter() - t0
        t0 = perf_counter()

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
            (np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list)), MPI.LONG),
            recv_counts
            # (np.fromiter((len(l) for l in send_list), dtype=int, count=len(send_list)), MPI.INT64_T), recv_counts
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
        t0 = perf_counter() - t0

        t0 = perf_counter()
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
        t0 = perf_counter() - t0

        t0 = perf_counter()
        local_basis = []
        for state, _ in itertools.groupby(merge(*received_states)):
            local_basis.append(state)
        t0 = perf_counter() - t0

        size_arr = np.empty((self.comm.size,), dtype=int)
        self.local_basis = local_basis
        t0 = perf_counter()
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
            # self.local_basis = list(local_states)
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
            self.state_bounds[r]
            if r < self.comm.size - 1 and self.state_bounds[r] != self.state_bounds[r + 1]
            else None
            for r in range(self.comm.size)
        ]

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
            (((l, 0, ml), "c"), ((l, 0, ml), "a")): 1.0
            for l in self.ls
            for ml in range(-l, l + 1)
            # (((2, 0, -2), "c"), ((2, 0, -2), "a")): 1.0,
            # (((2, 0, -1), "c"), ((2, 0, -1), "a")): 1.0,
            # (((2, 0, 0), "c"), ((2, 0, 0), "a")): 1.0,
            # (((2, 0, 1), "c"), ((2, 0, 1), "a")): 1.0,
            # (((2, 0, 2), "c"), ((2, 0, 2), "a")): 1.0,
        }
        n_dn_iop = c2i_op(
            {l: valence_baths[l] + conduction_baths[l] for l in valence_baths},
            n_dn_op,
        )
        n_up_op = {
            (((l, 1, ml), "c"), ((l, 1, ml), "a")): 1.0
            for l in self.ls
            for ml in range(-l, l + 1)
            # (((2, 1, -2), "c"), ((2, 1, -2), "a")): 1.0,
            # (((2, 1, -1), "c"), ((2, 1, -1), "a")): 1.0,
            # (((2, 1, 0), "c"), ((2, 1, 0), "a")): 1.0,
            # (((2, 1, 1), "c"), ((2, 1, 1), "a")): 1.0,
            # (((2, 1, 2), "c"), ((2, 1, 2), "a")): 1.0,
        }
        n_up_iop = c2i_op(
            {l: valence_baths[l] + conduction_baths[l] for l in valence_baths},
            n_up_op,
        )
        spin_flip = set()
        for det in determinants:
            n_dn = int(applyOp(self.num_spin_orbitals, n_dn_iop, {det: 1}).get(det, 0))
            n_up = int(applyOp(self.num_spin_orbitals, n_up_iop, {det: 1}).get(det, 0))
            spin_flip.add(det)
            to_flip = {det}
            for l in self.ls:
                for ml in range(-l, l + 1):
                    spin_flip_op = {
                        (((l, 1, ml), "c"), ((l, 0, ml), "a")): 1.0,
                        (((l, 0, ml), "c"), ((l, 1, ml), "a")): 1.0,
                    }
                    spin_flip_iop = c2i_op(
                        {l: valence_baths[l] + conduction_baths[l] for l in valence_baths},
                        spin_flip_op,
                    )
                    for state in list(to_flip):
                        flipped = applyOp(self.num_spin_orbitals, spin_flip_iop, {state: 1})
                        to_flip.update(flipped.keys())
                        if len(flipped) == 0:
                            continue
                        flipped_state = list(flipped.keys())[0]
                        new_n_dn = int(
                            applyOp(self.num_spin_orbitals, n_dn_iop, {flipped_state: 1}).get(flipped_state, 0)
                        )
                        new_n_up = int(
                            applyOp(self.num_spin_orbitals, n_up_iop, {flipped_state: 1}).get(flipped_state, 0)
                        )
                        if (new_n_dn == n_dn and new_n_up == n_up) or (new_n_dn == n_up and new_n_up == n_dn):
                            spin_flip.update(flipped.keys())
                    # spin_flip.update(flipped.keys())

        # for state in spin_flip.copy():
        #     new_bits = psr.bytes2bitarray(state, self.num_spin_orbitals)
        #     for bath_occ in itertools.permutations(new_bits[10:]):
        #         new_bits[10:] = psr.str2bitarray(''.join(f'{bit}' for bit in bath_occ))
        #         spin_flip.add(psr.bitarray2bytes(new_bits))

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
                # states_to_check = new_states - checked_states
            t_tmp = perf_counter()
            filtered_states = new_states
            # filtered_states = itertools.compress(new_states, (not x for x in self.contains(new_states)))
            t_filter += perf_counter() - t_tmp
            if self.spin_flip_dj:
                filtered_states = self._generate_spin_flipped_determinants(filtered_states)
            t_tmp = perf_counter()
            old_size = self.size
            self.add_states(filtered_states)
            t_add += perf_counter() - t_tmp
        # t_tmp = perf_counter()
        # filtered_states = list(itertools.compress(new_states, (not x for x in self.contains(new_states))))
        # t_filter += perf_counter() - t_tmp
        # if self.spin_flip_dj:
        # filtered_states = self._generate_spin_flipped_determinants(filtered_states)
        # new_states = self._generate_spin_flipped_determinants(new_states)
        # t_tmp = perf_counter()
        # self.add_states(filtered_states)
        # self.add_states(new_states)
        # t_add += perf_counter() - t_tmp

        # print(f"Basis.expand took {perf_counter() - t0} secondsds.")
        # print(f"===> getting new keys took {t_keys} secondsds.")
        # print(f"===> applyOp took {t_apply} secondsds.")
        # print(f"===> filter states took {t_filter} secondsds.")
        # print(f"===> add states took {t_add} secondsds.")
        if self.verbose:
            print(f"After expansion, the basis contains {self.size} elements.")
        t0 = perf_counter()
        op_dict = self.build_operator_dict(op, op_dict=op_dict)
        # print(f"Building operator took {perf_counter() - t0} seconds.")
        return op_dict

    def _getitem_sequence(self, l: Iterable[int]) -> Iterable[bytes]:
        if self.comm is None:
            return (self.local_basis[i] for i in l)

        l = np.fromiter((i if i >= 0 else self.size + i for i in l), dtype=int)
        # l = list(l)

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
            (np.fromiter((len(sl) for sl in send_list), dtype=int, count=len(send_list)), MPI.LONG),
            recv_counts
            # (np.fromiter((len(sl) for sl in send_list), dtype=int, count=len(send_list)), MPI.INT64_T), recv_counts
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
                # MPI.INT64_T,
            ),
            (queries, recv_counts, displacements, MPI.LONG),
            # (queries, recv_counts, displacements, MPI.INT64_T),
        )

        results = bytearray(sum(recv_counts) * self.n_bytes)
        # results = np.empty((sum(recv_counts) * self.n_bytes), dtype=np.ubyte)
        for i, query in enumerate(queries):
            if query >= self.offset and query < self.offset + len(self.local_basis):
                results[i * self.n_bytes : (i + 1) * self.n_bytes] = self.local_basis[query - self.offset]
        result = bytearray(len(l) * self.n_bytes)
        # result = np.zeros((len(l) * self.n_bytes), dtype=np.ubyte)

        self.comm.Alltoallv(
            (results, recv_counts * self.n_bytes, displacements * self.n_bytes, MPI.BYTE),
            (result, send_counts * self.n_bytes, send_offsets * self.n_bytes, MPI.BYTE),
        )

        # return (result[i * self.n_bytes : (i + 1) * self.n_bytes].tobytes() for i in np.argsort(send_order))
        return (bytes(result[i * self.n_bytes : (i + 1) * self.n_bytes]) for i in np.argsort(send_order))

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

        if self.debug:
            print("send_list:")
            for r in range(self.comm.size):
                print(f"    {r}: {send_list[r]}")
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
        # queries = np.empty((sum(recv_counts) * self.n_bytes), dtype=np.ubyte)
        displacements = np.fromiter(
            (sum(recv_counts[:p]) for p in range(self.comm.size)), dtype=int, count=self.comm.size
        )

        self.comm.Alltoallv(
            (
                bytearray(byte for states in send_list for state in states for byte in state),
                # np.array([byte for states in send_list for state in states for byte in state], dtype=np.ubyte),
                send_counts * self.n_bytes,
                send_displacements * self.n_bytes,
                MPI.BYTE,
            ),
            (queries, recv_counts * self.n_bytes, displacements * self.n_bytes, MPI.BYTE),
        )
        if self.debug:
            print("queries:")
            for r in range(self.comm.size):
                print(
                    f"    {r}: {[bytes(queries[i * self.n_bytes: (i + 1) * self.n_bytes]) for i in range(displacements[r], displacements[r] + recv_counts[r])]}"
                )

        results = np.empty((sum(recv_counts)), dtype=int)
        for i in range(sum(recv_counts)):
            query = bytes(queries[i * self.n_bytes : (i + 1) * self.n_bytes])
            results[i] = self._index_dict.get(query, self.size)
        result = np.empty((len(s)), dtype=int)
        result[:] = self.size

        self.comm.Alltoallv(
            (results, recv_counts, displacements, MPI.LONG),
            (result, send_counts, send_displacements, MPI.LONG)
            # (results, recv_counts, displacements, MPI.INT64_T), (result, send_counts, send_displacements, MPI.INT64_T)
        )
        result[sum(send_counts) :] = self.size
        while np.any(np.logical_or(result > self.size, result < 0)):
            mask = np.logical_or(result > self.size, result < 0)
            result[mask] = np.from_iter(self._index_sequence(itertools.compress(s, mask)), dtype=int)

        return (res for res in result[np.argsort(send_order)])

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
                if res == psr.int2bytes(0, self.num_spin_orbitals):
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
            if result == psr.int2bytes(0, self.num_spin_orbitals):
                raise IndexError(f"Could not find index {key} in basis with size {self.size}!")
            return result
        else:
            raise TypeError(f"Invalid index type {type(key)}. Valid types are slice, Sequence and int")
        return None

    def __len__(self):
        return self.size

    def __contains__(self, item):
        if self.comm is None:
            return item in self._index_dict
        return next(self._index_sequence([item])) != self.size

    def _contains_sequence(self, items):
        if self.comm is None:
            return (item in self._index_dict for item in items)
        return (index != self.size for index in self._index_sequence(items))

    def contains(self, item) -> Iterable[bool]:
        if isinstance(item, self.type):
            return next(self._contains_sequence([item]))
        elif isinstance(item, Sequence):
            return self._contains_sequence(item)
        elif isinstance(item, Iterable):
            return self._contains_sequence(item)
        return None

    def __iter__(self):
        for i in range(self.size):
            yield self.__getitem__(i)

    def copy(self):
        return Basis(
            ls=self.ls,
            bath_states=self.bath_states,
            initial_basis=self.local_basis,
            num_spin_orbitals=self.num_spin_orbitals,
            restrictions=self.restrictions,
            spin_flip_dj=self.spin_flip_dj,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            verbose=self.verbose,
        )

    def clear(self):
        self.local_basis.clear()
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
        h_local = self.build_sparse_matrix(op, op_dict)
        local_dok = h_local.todok()
        if self.is_distributed:
            reduced_dok = self.comm.reduce(local_dok, op=MPI.SUM, root=0)
            if self.comm.rank == 0:
                h = reduced_dok.todense()
            h = self.comm.bcast(h if self.comm.rank == 0 else None, root=0)
        else:
            h = h_local.todense()
        return h

    def build_sparse_matrix(self, op, op_dict: Optional[dict[bytes, dict[bytes, complex]]] = None):
        """
        Get the operator as a sparse matrix in the current basis.
        The sparse matrix is distributed over all ranks.
        """
        # if "PETSc" in sys.modules:
        #     return self._build_PETSc_matrix(op, op_dict)

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
                for state, index in zip(rows_in_basis, self._index_sequence(rows_in_basis))
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
        ls,
        bath_states=None,
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
        spin_flip_dj=False,
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
            bath_states = (valence_baths, conduction_baths)
        else:
            assert num_spin_orbitals is not None
            assert bath_states is not None
        super(CIPSI_Basis, self).__init__(
            ls=ls,
            bath_states=bath_states,
            initial_basis=initial_basis,
            num_spin_orbitals=num_spin_orbitals,
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
                ls={},
                bath_states={},
                initial_basis=itertools.compress(Dj_candidates, Dj_basis_mask),
                num_spin_orbitals=self.num_spin_orbitals,
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
                eigenValueTol=de2_min,
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

        # print(f"CIPSI_Basis.expand took {perf_counter() - t0} seconds.")
        # print(f"===> building matrix took {t_build_mat} secondsds.")
        # print(f"===> building vector took {t_build_vec} secondsds.")
        # print(f"===> building state took {t_build_state} secondsds.")
        # print(f"===> finding eigenstates took {t_eigen} secondsds.")
        # print(f"===> determining new Djs took {t_Dj} seconds.")
        # print(f"===> add states took {t_add} seconds.")
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
        # print(f"Building operator took {t_build_dict} seconds.")
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
            ls=self.ls,
            bath_states=self.bath_states,
            initial_basis=self.local_basis,
            num_spin_orbitals=self.num_spin_orbitals,
            restrictions=self.restrictions,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            spin_flip_dj=self.spin_flip_dj,
            tau=self.tau,
            verbose=self.verbose,
        )
        return new_basis
