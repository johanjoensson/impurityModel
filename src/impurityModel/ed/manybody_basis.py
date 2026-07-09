from math import ceil

try:
    from collections.abc import Iterable, Sequence
except ModuleNotFoundError:
    from collections import Iterable, Sequence
import itertools
from heapq import merge

import numpy as np
from mpi4py import MPI

from impurityModel.ed.basis_generation import generate_initial_basis, spin_flipped_determinants
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
    SlaterDeterminant,
)
from impurityModel.ed.ManyBodyUtils import (
    applyOp as applyOp_test,
)
from impurityModel.ed.mpi_comm import distribute_determinants, graph_alltoall, graph_alltoall_block, graph_alltoall_psis


def collective_amplitude_cutoff(scores, k, comm):
    """Smallest cutoff with at most ``k`` scores above it, across all ranks.

    Ranks candidates by their (nonnegative) importance ``scores`` and returns the
    cutoff such that the global number of entries with ``score > cutoff`` is <= ``k``:
    keeping everything strictly above the cutoff admits the top-``k`` candidates,
    under-admitting ties at the cutoff (the cap is never exceeded). Near-tie retained
    sets may differ across rank counts through summation-order rounding.

    The bisection runs a fixed iteration count on allreduce'd counts, so every rank
    computes the identical cutoff. It bisects geometrically over the nonzero score
    range, so the full floating-point dynamic range is resolved (a linear bisection
    from the maximum cannot reach scores below ``max / 2^45``). **Collective on**
    ``comm``: call unconditionally on all ranks (a rank may hold zero scores).

    Parameters
    ----------
    scores : np.ndarray
        Rank-local nonnegative importance scores (e.g. ``|amplitude|^2``).
    k : int
        Maximum global number of scores allowed above the returned cutoff.
    comm : MPI.Comm or None
        Communicator; ``None`` (or size 1) means serial.

    Returns
    -------
    float
        The cutoff; retain entries with ``score > cutoff``.
    """
    mpi = comm is not None and comm.size > 1
    positive = scores[scores > 0.0] if scores.size else scores
    local_max = float(positive.max()) if positive.size else 0.0
    hi = comm.allreduce(local_max, op=MPI.MAX) if mpi else local_max
    if hi == 0.0:
        return 0.0
    local_min = float(positive.min()) if positive.size else np.inf
    lo = comm.allreduce(local_min, op=MPI.MIN) if mpi else local_min
    # Floor just below the smallest nonzero score, so "retain everything" is reachable.
    lo *= 0.5
    for _ in range(45):
        mid = np.sqrt(lo * hi)
        count = int(np.count_nonzero(scores > mid))
        if mpi:
            count = comm.allreduce(count, op=MPI.SUM)
        if count <= k:
            hi = mid
        else:
            lo = mid
    return hi


class Basis:
    """Many-body basis of Slater determinants.

    This class manages the Slater determinant basis states for exact diagonalization,
    supporting distributed states over MPI, restrictions, and basis extensions.
    """

    def __init__(
        self,
        impurity_orbitals,
        bath_states,
        nominal_impurity_occ=None,
        mixed_valence=None,
        initial_basis=None,
        restrictions=None,
        weighted_restrictions=None,
        split_threshold=1.0,
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
        """Initialize the Basis class.

        Parameters
        ----------
        impurity_orbitals : dict
            Impurity orbitals.
        bath_states : tuple of dict
            Valence and conduction bath states.
        nominal_impurity_occ : dict, optional
            Nominal impurity occupations.
        mixed_valence : dict, optional
            Mixed valence bounds.
        initial_basis : list of SlaterDeterminant or bytes, optional
            Predefined initial states.
        restrictions : dict, optional
            Initial occupation restrictions.
        delta_valence_occ : dict, optional
            Allowed valence occupation variations.
        delta_conduction_occ : dict, optional
            Allowed conduction occupation variations.
        delta_impurity_occ : dict, optional
            Allowed impurity occupation variations.
        truncation_threshold : float, default np.inf
            Global cap on the number of Slater determinants (``np.inf`` = uncapped). The
            container itself only *stops growing* at the cap (``expand`` rejects a batch
            that would overflow); importance-based truncation is the solvers' job (CIPSI
            for the ground state, the capped GF drivers for spectra). ``None`` is
            normalized to ``np.inf``; drivers derive RAM-fitted values via
            :mod:`impurityModel.ed.memory_estimate`.
        spin_flip_dj : bool, default False
            Whether to enable spin-flip states.
        tau : float, default 0
            Tau parameter.
        chain_restrict : bool, default False
            Whether to restrict chain states.
        collapse_chains : bool, default False
            Whether to collapse chains.
        comm : MPI.Comm, optional
            MPI communicator.
        verbose : bool, default True
            Whether to print info.
        debug : bool, default False
            Debug flag.
        """
        assert (
            impurity_orbitals is not None
        ), "You need to supply the number of impurity orbitals in each set in impurity_orbitals"
        assert bath_states is not None, "You need to supply the number of bath states for each l quantum number"

        self.num_spin_orbitals = sum(
            sum(len(orbs) for orbs in impurity_orbitals[i])
            + sum(len(orbs) for orbs in bath_states[0][i])
            + sum(len(orbs) for orbs in bath_states[1][i])
            for i in bath_states[0]
        )
        test = ManyBodyState({SlaterDeterminant.from_bytes(b"\x00"): 1.0})
        slater_det = list(test.keys())[0]
        self.type = type(slater_det)
        self.n_bytes = int(ceil(ceil(self.num_spin_orbitals / 8) / len(slater_det)) * len(slater_det))

        self.truncation_threshold = np.inf if truncation_threshold is None else truncation_threshold
        self.is_distributed = comm is not None and comm.size > 1
        self.tau = tau

        if initial_basis is not None:
            assert nominal_impurity_occ is None
            assert delta_valence_occ is None
            assert delta_conduction_occ is None
            assert delta_impurity_occ is None
            initial_basis = [
                self.type.from_bytes(state) if isinstance(state, bytes) else state for state in initial_basis
            ]
        else:
            assert nominal_impurity_occ is not None
            initial_basis, num_spin_orbitals = generate_initial_basis(
                impurity_orbitals=impurity_orbitals,
                bath_states=bath_states,
                delta_valence_occ=delta_valence_occ,
                delta_conduction_occ=delta_conduction_occ,
                delta_impurity_occ=delta_impurity_occ,
                nominal_impurity_occ=nominal_impurity_occ,
                mixed_valence=mixed_valence if mixed_valence is not None else dict.fromkeys(nominal_impurity_occ, 0),
                n_bytes=self.n_bytes,
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
        self.restrictions = restrictions
        # Weighted-sum restrictions (e.g. S_z), list of (weights, (q_min, q_max)); see
        # ManyBodyOperator.set_weighted_restrictions. None = none.
        self.weighted_restrictions = weighted_restrictions
        # Adaptive MPI split policy (Phase 7): cap the number of split colors near the
        # participation ratio of the block costs, scaled by split_threshold. Larger =>
        # split more aggressively; 0 => never split (unified communicator). 1.0 keeps the
        # legacy max-split behaviour for equally-weighted blocks.
        self.split_threshold = split_threshold

        # Distributed determinant storage:
        # the rank-local sorted determinant list, its state -> global-index dict, and the
        # rank-partition bookkeeping. States are hash-distributed across ranks; lookups and
        # retrievals use sparse point-to-point communication (graph_alltoall).
        self.rng = np.random.default_rng()
        self.local_basis = []
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict: dict = {}
        self.index_bounds = [None] * comm.size if self.is_distributed else [None]
        self.state_bounds = [None] * comm.size if self.is_distributed else [None]
        self.add_states(initial_basis)

    def clone(self, initial_basis=None, restrictions=None, weighted_restrictions=None, verbose=None, comm=None):
        """Create a new Basis instance, optionally overriding initial_basis and restrictions.

        If initial_basis is None, the new basis will start with self.local_basis.
        If restrictions is None, the new basis will inherit self.restrictions.
        If weighted_restrictions is None, the new basis inherits self.weighted_restrictions.
        If comm is None, the new basis will inherit self.comm.
        """
        return Basis(
            impurity_orbitals=self.impurity_orbitals,
            bath_states=self.bath_states,
            initial_basis=initial_basis if initial_basis is not None else list(self.local_basis),
            restrictions=restrictions if restrictions is not None else self.restrictions,
            weighted_restrictions=(
                weighted_restrictions if weighted_restrictions is not None else self.weighted_restrictions
            ),
            split_threshold=self.split_threshold,
            truncation_threshold=self.truncation_threshold,
            spin_flip_dj=self.spin_flip_dj,
            tau=self.tau,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            comm=comm if comm is not None else self.comm,
            verbose=verbose if verbose is not None else self.verbose,
            debug=self.debug,
        )

    def free_comm(self):
        """
        Free the split/custom MPI communicator associated with this Basis.
        This must be called collectively by all ranks sharing the communicator.
        """
        if self.comm is not None and self.comm != MPI.COMM_NULL:
            self.comm.Free()
            self.comm = None

    @staticmethod
    def _point2point(send_list, comm):
        """Sparse point-to-point MPI exchange of per-rank data lists."""
        return graph_alltoall(send_list, comm)

    def add_states(self, new_states: Iterable[bytes], unique_sorted=False) -> None:
        """
        Extend the current basis by adding the new_states to it.
        """
        new_states = [self.type.from_bytes(state) if isinstance(state, bytes) else state for state in new_states]
        if not self.is_distributed:
            existing_set = self._index_dict
            unique_new = [s for s in sorted(set(new_states)) if s not in existing_set]
            if unique_new:
                self.local_basis = list(merge(self.local_basis, unique_new))
                self.size = len(self.local_basis)
                self.offset = 0
                self.local_indices = range(0, len(self.local_basis))
                self._index_dict = {state: i for i, state in enumerate(self.local_basis)}
                if __debug__:
                    assert all(self.local_basis[i] < self.local_basis[i + 1] for i in range(len(self.local_basis) - 1))
            return

        unique_new_states = list(set(new_states))
        received_list = distribute_determinants(unique_new_states, self.n_bytes, self.comm)

        all_received = []
        for r_data in received_list:
            if r_data:
                all_received.extend(r_data)

        existing_set = self._index_dict
        unique_received = sorted(set(all_received))
        unique_new = [s for s in unique_received if s not in existing_set]

        local_added = len(unique_new)
        any_added = self.comm.allreduce(local_added, op=MPI.SUM)
        if any_added == 0:
            return

        if unique_new:
            self.local_basis = list(merge(self.local_basis, unique_new))

        local_length = len(self.local_basis)
        size_arr = np.array(self.comm.allgather(local_length), dtype=int)
        self.size = np.sum(size_arr)
        self.offset = np.sum(size_arr[: self.comm.rank])
        self.local_indices = range(self.offset, self.offset + len(self.local_basis))
        self.index_bounds = [np.sum(size_arr[: r + 1]) if size_arr[r] > 0 else None for r in range(self.comm.size)]
        self._index_dict = {state: self.offset + i for i, state in enumerate(self.local_basis)}
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

    def redistribute_psis(self, psis: list[ManyBodyState]) -> list[ManyBodyState]:
        """Redistribute wavefunctions across MPI ranks based on state ownership.

        Parameters
        ----------
        psis : list of ManyBodyState
            The wavefunctions to redistribute.

        Returns
        -------
        list of ManyBodyState
            The redistributed wavefunctions.
        """
        if isinstance(psis, ManyBodyState):
            print("WARNING in redistribute_psi:")
            print(
                "Expetced a list of ManyBodyStates, received a single ManyBodyState. Remaking into list of one ManyBodyState"
            )
            psis = [psis]
        psis = [
            (
                psi
                if isinstance(psi, ManyBodyState)
                else ManyBodyState(
                    {(SlaterDeterminant.from_bytes(k) if isinstance(k, bytes) else k): v for k, v in psi.items()}
                )
            )
            for psi in psis
        ]
        if not self.is_distributed:
            return psis

        comm = self.comm

        res = graph_alltoall_psis(psis, self.n_bytes, comm)
        return res

    def redistribute_block(self, block):
        """Redistribute a ``ManyBodyBlockState`` across MPI ranks by state ownership.

        The block analogue of :meth:`redistribute_psis` (Phase 2.3 of the block-state
        matvec plan): one wire entry per shared-support row instead of one per
        (determinant, vector) pair; rows for the same determinant arriving from
        several ranks are summed per column. Non-distributed bases return the block
        unchanged, mirroring :meth:`redistribute_psis`.
        """
        if not self.is_distributed:
            return block
        return graph_alltoall_block(block, self.n_bytes, self.comm)

    def expand(self, op, slaterWeightMin=0, max_it=5):
        """
        Expand the basis in place by repeatedly applying an operator to the
        basis states, thus generating new basis states.

        Parameters
        ----------
        op : ManyBodyOperator or dict
            The operator to apply, over and over again.
        slaterWeightMin : float, default 0
            Minimum amplitude for generated Slater determinants to be kept.
        max_it : int, default 5
            Apply the operator at most this number of times.
        """
        if isinstance(op, dict):
            op = ManyBodyOperator(op)
        op.set_restrictions(self.restrictions)
        # Unconditional (like set_restrictions): passing None clears any stale weighted
        # mask left on a reused operator object.
        op.set_weighted_restrictions(self.weighted_restrictions)
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
                    )
                    new_local_states |= set(state for state in res.keys()) - local_states
                if len(new_local_states) == 0:
                    break
                apply_h_to_these = new_local_states
                local_states |= new_local_states
            new_states = local_states - set(self.local_basis)
            if self.spin_flip_dj:
                new_states = spin_flipped_determinants(self.impurity_orbitals, new_states)
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

    def index(self, val: SlaterDeterminant) -> int:
        """Find the global index of a Slater determinant in the basis.

        Parameters
        ----------
        val : SlaterDeterminant
            The Slater determinant to look up.

        Returns
        -------
        int
            The global index of the determinant.

        Raises
        ------
        ValueError
            If any state is not found in the basis.
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

    def __getitem__(self, key: int | slice) -> SlaterDeterminant | list[SlaterDeterminant]:
        """Get the Slater determinant(s) at the specified index or slice.

        Parameters
        ----------
        key : int or slice
            The index or slice of basis states to retrieve.

        Returns
        -------
        SlaterDeterminant or list of SlaterDeterminant
            The Slater determinant at the index, or list of determinants for a slice.

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
            return (state for state in result)
        elif isinstance(key, int):
            result = next(self._getitem_sequence([key]))
            if result == SlaterDeterminant.from_bytes(bytes(0)):
                raise IndexError(f"Could not find index {key} in basis with size {self.size}!")
            return result
        else:
            raise TypeError(f"Invalid index type {type(key)}. Valid types are slice, Sequence and int")
        return None

    def __len__(self) -> int:
        """Get the total size of the basis.

        Returns
        -------
        int
            The total number of Slater determinants in the basis.
        """
        return self.size

    def __contains__(self, item: SlaterDeterminant | bytes) -> bool:
        """Check if a Slater determinant or its byte representation is in the basis.

        Parameters
        ----------
        item : SlaterDeterminant or bytes
            The state to search for.

        Returns
        -------
        bool
            True if the state is in the basis, False otherwise.
        """
        if isinstance(item, bytes):
            item = self.type.from_bytes(item)
        if not self.is_distributed:
            return item in self._index_dict
        return next(self._index_sequence([item])) != self.size

    def contains_local(self, item: SlaterDeterminant) -> bool:
        """Check whether this rank already owns ``item``, without any communication.

        The rank-local, ``O(1)`` counterpart of :meth:`__contains__`, which runs a global
        index query when the basis is distributed. Callers that only want to avoid handing
        :meth:`add_states` a determinant it already has (and shrink the redistribution
        payload) want this: ``item in self.local_basis`` is the same predicate but scans a
        list, and at solver support sizes that scan dominates the matvec it accompanies.
        """
        return item in self._index_dict

    def contains(self, item: Iterable[SlaterDeterminant | bytes]) -> np.ndarray:
        """Check containment for an iterable of states.

        Parameters
        ----------
        item : Iterable of SlaterDeterminant or bytes
            The collection of states to check.

        Returns
        -------
        np.ndarray of bool
            Boolean array indicating containment for each state.
        """
        if isinstance(item, bytes):
            item = self.type.from_bytes(item)
        if isinstance(item, self.type):
            return next(self._contains_sequence([item]))
        elif isinstance(item, Sequence) or isinstance(item, Iterable):
            converted = [self.type.from_bytes(x) if isinstance(x, bytes) else x for x in item]
            return self._contains_sequence(converted)
        return None

    def __iter__(self) -> Iterable[SlaterDeterminant]:
        """Iterate over all Slater determinants in the basis.

        Yields
        ------
        SlaterDeterminant
            The next Slater determinant in the basis.
        """
        chunk_size = 10000
        for i in range(0, self.size, chunk_size):
            chunk_end = min(i + chunk_size, self.size)
            chunk = self._getitem_sequence(range(i, chunk_end))
            for state in chunk:
                yield state

    def _getitem_sequence(self, l: Iterable[int]) -> Iterable[SlaterDeterminant]:
        """Retrieve the states for a sequence of global indices (sparse point-to-point)."""
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

        queries = Basis._point2point(send_list, self.comm)

        results = [[] for _ in range(self.comm.size)]
        for r in range(len(queries)):
            for i, query in enumerate(queries[r]):
                if query >= self.offset and query < self.offset + len(self.local_basis):
                    results[r].append(self.local_basis[query - self.offset])

        result = [state for r_results in Basis._point2point(results, self.comm) for state in r_results]

        return (result[i] for i in np.argsort(send_order))

    def _index_sequence(self, s: Iterable[SlaterDeterminant]) -> Iterable[int]:
        """Find the global indices for a sequence of states (hash-routed lookups)."""
        if not self.is_distributed:
            return (self._index_dict.get(val, self.size) for val in s)

        s = list(s)
        send_list: list[list[SlaterDeterminant]] = [[] for _ in range(self.comm.size)]
        send_to_ranks = np.empty((len(s)), dtype=int)
        send_to_ranks[:] = self.size
        for i, val in enumerate(s):
            r = val.routing_hash() % self.comm.size
            send_list[r].append(val)
            send_to_ranks[i] = r

        send_order = np.argsort(send_to_ranks, kind="stable")

        queries = Basis._point2point(send_list, self.comm)

        results = [[] for _ in range(self.comm.size)]
        for r in range(self.comm.size):
            for query in queries[r]:
                results[r].append(self._index_dict.get(query, self.size))
        result = np.array([i for r_i in Basis._point2point(results, self.comm) for i in r_i], dtype=int)
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
        """Check membership for a sequence of states."""
        if not self.is_distributed:
            return (item in self._index_dict for item in items)
        return (index < self.size for index in self._index_sequence(items))

    def copy(self) -> "Basis":
        """Create a copy of this Basis.

        Returns
        -------
        Basis
            A new Basis object with identical states and parameters.
        """
        return Basis(
            self.impurity_orbitals,
            self.bath_states,
            initial_basis=self.local_basis,
            restrictions=self.restrictions,
            weighted_restrictions=self.weighted_restrictions,
            split_threshold=self.split_threshold,
            spin_flip_dj=self.spin_flip_dj,
            chain_restrict=self.chain_restrict,
            collapse_chains=self.collapse_chains,
            comm=self.comm,
            truncation_threshold=self.truncation_threshold,
            verbose=self.verbose,
        )

    def clear(self) -> None:
        """Clear all states from the basis."""
        self.local_basis.clear()
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
        self._index_dict = {}
        self.index_bounds = [None] * self.comm.size if self.is_distributed else [None]
        self.state_bounds = [None] * self.comm.size if self.is_distributed else [None]
        self.add_states([])

    @property
    def impurity_spin_orbital_indices(self):
        """Flat, sorted-by-orbital-set list of all impurity spin-orbital indices."""
        return [orb for blocks in self.impurity_orbitals.values() for block in blocks for orb in block]

    @property
    def valence_spin_orbital_indices(self):
        """Flat list of all valence-bath spin-orbital indices."""
        valence_baths, _conduction_baths = self.bath_states
        return [orb for blocks in valence_baths.values() for block in blocks for orb in block]

    @property
    def conduction_spin_orbital_indices(self):
        """Flat list of all conduction-bath spin-orbital indices."""
        _valence_baths, conduction_baths = self.bath_states
        return [orb for blocks in conduction_baths.values() for block in blocks for orb in block]
