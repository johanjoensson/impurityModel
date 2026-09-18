import itertools
from collections.abc import Iterable, Iterator, Sequence
from math import ceil
from typing import overload

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
from impurityModel.ed.mpi_comm import distribute_determinants, graph_alltoall, graph_alltoall_block


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


class _LocalBasisView:
    """A read-only sequence view of a rank's local determinants.

    ``Basis`` stores its determinants in a width-0 ``ManyBodyState`` -- a sorted C++ key
    vector -- rather than a Python list, so the per-determinant cost is the key itself
    instead of a key plus a ``SlaterDeterminant`` wrapper object plus a list slot. This
    view keeps ``basis.local_basis`` a sequence for the callers that index it, take its
    length or test membership, without rebuilding that list to answer them: ``len`` and
    ``in`` and ``[i]`` all go straight to the C++ block.

    Iteration is the one operation that must materialize, because the caller is asking for
    the determinant objects themselves.
    """

    __slots__ = ("_keys",)

    def __init__(self, keys):
        self._keys = keys

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys.keys())

    def __contains__(self, item) -> bool:
        return item in self._keys

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self._keys.keys()[index]
        if index < 0:
            index += len(self._keys)
        return self._keys.key_at(index)

    def __repr__(self) -> str:
        return f"<local basis of {len(self._keys)} determinants>"


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
        total_charge_slack=0,
        initial_basis=None,
        restrictions=None,
        weighted_restrictions=None,
        split_threshold=1.0,
        delta_valence_occ=None,
        delta_conduction_occ=None,
        delta_impurity_occ=None,
        frozen_occupations=None,
        truncation_threshold=np.inf,
        spin_flip_dj=False,
        tau=0,
        chain_restrict=False,
        collapse_chains=False,
        comm=None,
        verbose=False,
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
            Impurity charge fluctuation against the bath, at fixed total electron number.
        total_charge_slack : int, optional
            Half-width of the *total charge* window; ``0`` (the default) keeps the basis a
            single charge sector. See
            :func:`impurityModel.ed.basis_generation.generate_initial_basis`.
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
        frozen_occupations : set, optional
            Orbital-set keys whose impurity occupation is pinned at exactly
            ``nominal_impurity_occ[i]`` during basis generation (e.g. a bath-less core
            shell); see :func:`impurityModel.ed.basis_generation.generate_initial_basis`.
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
        verbose : bool, default False
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
        slater_det = SlaterDeterminant.from_bytes(b"\x00")
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
            initial_basis = [self._as_determinant(state) for state in initial_basis]
        else:
            assert nominal_impurity_occ is not None
            initial_basis, _num_spin_orbitals = generate_initial_basis(
                impurity_orbitals=impurity_orbitals,
                bath_states=bath_states,
                delta_valence_occ=delta_valence_occ,
                delta_conduction_occ=delta_conduction_occ,
                delta_impurity_occ=delta_impurity_occ,
                nominal_impurity_occ=nominal_impurity_occ,
                mixed_valence=mixed_valence if mixed_valence is not None else dict.fromkeys(nominal_impurity_occ, 0),
                n_bytes=self.n_bytes,
                # generate_initial_basis is deliberately comm-free (pure, rank-deterministic);
                # gate the rank here so its prints don't fire nranks times.
                verbose=verbose and (comm is None or comm.rank == 0),
                frozen_occupations=frozen_occupations,
                total_charge_slack=total_charge_slack,
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
        self._keys = ManyBodyState(width=0)
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
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

    @property
    def local_basis(self):
        """This rank's determinants, in sorted order, as a sequence view (see
        :class:`_LocalBasisView`). Assigning any iterable of determinants rebuilds the
        underlying sorted key block."""
        return _LocalBasisView(self._keys)

    @local_basis.setter
    def local_basis(self, states) -> None:
        self._keys = ManyBodyState.from_keys(states)

    def _as_determinant(self, state):
        """Convert ``state`` to this basis's determinant type at its canonical width.

        Determinant keys are ``std::vector<uint64_t>`` and compare element-wise, so a proper
        prefix sorts strictly before its zero-extended twin: the *same* physical occupation
        built from byte strings of different length yields two unequal keys with different
        hashes. Padding every input to ``n_bytes`` here makes the width an invariant of the
        container rather than a property of whatever the caller happened to pass, which is
        what lets ordering, hashing and index arithmetic agree across ranks.

        Production already satisfies this (``basis_generation.generate_initial_basis`` emits
        ``tuple2bytes(occupied, 8 * n_bytes)``); the normalization exists so that it cannot
        silently stop being true.
        """
        if not isinstance(state, bytes):
            return state
        if len(state) > self.n_bytes:
            # Over-wide input is accepted only when the excess carries no occupation, which
            # makes the trim lossless. This is not hypothetical: `SlaterDeterminant.to_bytearray`
            # returns eight bytes per chunk-byte (`_slater_state.pxi:44` allocates
            # `8 * n_bytes * len(self)` and fills an eighth of it), so any caller that
            # round-trips `from_bytes(to_bytearray())` hands us a determinant eight times too
            # wide -- which, unnormalized, is a key that compares unequal to the same
            # occupation built any other way.
            if any(state[self.n_bytes :]):
                raise ValueError(
                    f"determinant of {len(state)} bytes exceeds this basis's width of "
                    f"{self.n_bytes} ({self.num_spin_orbitals} spin-orbitals) and carries "
                    f"occupation beyond it; trimming would drop orbitals"
                )
            state = state[: self.n_bytes]
        return self.type.from_bytes(state.ljust(self.n_bytes, b"\x00"))

    def add_states(self, new_states: Iterable[bytes], unique_sorted=False) -> None:
        """
        Extend the current basis by adding the new_states to it.
        """
        new_states = [self._as_determinant(state) for state in new_states]
        if not self.is_distributed:
            unique_new = [s for s in sorted(set(new_states)) if not self._contains_local(s)]
            if unique_new:
                self._keys.merge_keys(ManyBodyState.from_keys(unique_new))
                self.size = len(self._keys)
                self.offset = 0
                self.local_indices = range(0, len(self._keys))
            return

        unique_new_states = list(set(new_states))
        received_list = distribute_determinants(unique_new_states, self.n_bytes, self.comm)

        all_received = []
        for r_data in received_list:
            if r_data:
                all_received.extend(r_data)

        unique_received = sorted(set(all_received))
        unique_new = [s for s in unique_received if not self._contains_local(s)]

        local_added = len(unique_new)
        any_added = self.comm.allreduce(local_added, op=MPI.SUM)
        if any_added == 0:
            return

        if unique_new:
            self._keys.merge_keys(ManyBodyState.from_keys(unique_new))

        local_length = len(self._keys)
        size_arr = np.array(self.comm.allgather(local_length), dtype=int)
        self.size = np.sum(size_arr)
        self.offset = np.sum(size_arr[: self.comm.rank])
        self.local_indices = range(self.offset, self.offset + len(self._keys))
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

    def redistribute_psis(self, *blocks):
        """Redistribute one or more ``ManyBodyState`` blocks across MPI ranks by
        determinant ownership, in a single fused collective.

        Parameters
        ----------
        *blocks : ManyBodyState
            Any number of blocks, of any (possibly differing) width ``w_i`` -- each a
            producer's own independently-supported block, not necessarily sharing
            support with the others. A single block (``redistribute_psis(blk)``) is
            first-class; :meth:`redistribute_block` is the direct one-block primitive
            this composes from.

        Returns
        -------
        list of ManyBodyState
            One redistributed block per input, same width ``w_i`` as the corresponding
            input, each pruned back to its own (sparse) support -- no union-support
            zero padding survives, unlike the old ``column``-based split this replaces.
            Deliberately a ``list``, not a tuple: ``block_tsqr``/``block_normalize``
            (``_reort.pxi``) dispatch on ``isinstance(wp, list)`` to decide whether
            they are looking at a list of width-1 states or an already-combined block,
            and a tuple would silently fall through to the "already an array" branch.

        All ``W = sum(w_i)`` columns are flattened into one block over the union
        support (``ManyBodyState.from_states``), redistributed in one fused
        ``Neighbor_alltoallv`` (:meth:`redistribute_block`, hashing/routing each
        determinant once regardless of ``W``), then sliced back by cumulative column
        offset (``select``) and re-pruned to each slice's own support
        (``prune_rows(0.0)``). Collective on ``self.comm`` when distributed; a
        non-distributed basis returns the inputs unchanged, mirroring
        :meth:`redistribute_block`.
        """
        if not blocks:
            return []
        if not self.is_distributed:
            return list(blocks)
        widths = [b.width for b in blocks]
        combined = ManyBodyState.from_states([c for b in blocks for c in b.to_states()])
        combined = self.redistribute_block(combined)
        out = []
        offset = 0
        for w in widths:
            sub = combined.select(list(range(offset, offset + w)))
            sub.prune_rows(0.0)
            out.append(sub)
            offset += w
        return out

    def redistribute_block(self, block):
        """Redistribute a ``ManyBodyState`` across MPI ranks by state ownership.

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
                    new_local_states |= set(res.keys()) - local_states
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
        if self.verbose and (self.comm is None or self.comm.rank == 0):
            print(f"After expansion, the basis contains {self.size} elements.")

    def index(self, val: SlaterDeterminant | bytes | Iterable[SlaterDeterminant | bytes]) -> int | Iterator[int]:
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
        elif isinstance(val, (Sequence, Iterable)):
            converted = [self.type.from_bytes(x) if isinstance(x, bytes) else x for x in val]
            res = list(self._index_sequence(converted))
            for i, v in enumerate(res):
                if v >= self.size:
                    raise ValueError(f"Could not find {list(val)[i]!r} in basis!")
            return (i for i in res)
        else:
            raise TypeError(f"Invalid query type {type(val)}! Valid types are {self.type} and sequences thereof.")

    def __getitem__(self, key: int | slice | Iterable[int]) -> SlaterDeterminant | Iterator[SlaterDeterminant]:
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
        elif isinstance(key, (Sequence, Iterable)):
            key_list = list(key)
            result = list(self._getitem_sequence(key_list))
            for i, res in enumerate(result):
                if res == SlaterDeterminant.from_bytes(bytes(0)):
                    raise IndexError(f"Could not find index {key_list[i]} in basis with size {self.size}!")
            return (state for state in result)
        elif isinstance(key, int):
            result = next(self._getitem_sequence([key]))
            if result == SlaterDeterminant.from_bytes(bytes(0)):
                raise IndexError(f"Could not find index {key} in basis with size {self.size}!")
            return result
        else:
            raise TypeError(f"Invalid index type {type(key)}. Valid types are slice, Sequence and int")

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
            return self._contains_local(item)
        return next(self._index_sequence([item])) != self.size

    def contains_local(self, item: SlaterDeterminant) -> bool:
        """Check whether this rank already owns ``item``, without any communication.

        The rank-local, ``O(1)`` counterpart of :meth:`__contains__`, which runs a global
        index query when the basis is distributed. Callers that only want to avoid handing
        :meth:`add_states` a determinant it already has (and shrink the redistribution
        payload) want this: ``item in self.local_basis`` is the same predicate but scans a
        list, and at solver support sizes that scan dominates the matvec it accompanies.
        """
        return self._contains_local(item)

    @overload
    def contains(self, item: SlaterDeterminant | bytes) -> bool: ...
    @overload
    def contains(self, item: Iterable[SlaterDeterminant | bytes]) -> Iterator[bool]: ...
    def contains(self, item):
        """Check containment for a single state or an iterable of states.

        Parameters
        ----------
        item : SlaterDeterminant, bytes, or an iterable thereof
            The state(s) to check.

        Returns
        -------
        bool or Iterator[bool]
            A single bool for a single state, or an iterator of bools (one per state) for an
            iterable of states.
        """
        if isinstance(item, bytes):
            item = self.type.from_bytes(item)
        if isinstance(item, self.type):
            return next(self._contains_sequence([item]))
        elif isinstance(item, (Sequence, Iterable)):
            converted = [self.type.from_bytes(x) if isinstance(x, bytes) else x for x in item]
            return self._contains_sequence(converted)
        raise TypeError(f"Invalid query type {type(item)}! Valid types are {self.type} and sequences thereof.")

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

    def _getitem_sequence(self, l: Iterable[int]) -> Iterator[SlaterDeterminant]:
        """Retrieve the states for a sequence of global indices (sparse point-to-point)."""
        if not self.is_distributed:
            return (self._keys.key_at(i) for i in l)

        l = np.fromiter((i if i >= 0 else self.size + i for i in l), dtype=int)

        send_list: list[list[int]] = [[] for _ in range(self.comm.size)]
        send_to_ranks = np.empty((len(l)), dtype=int)
        send_to_ranks[:] = self.size
        for idx, i in enumerate(l):
            for r in range(self.comm.size):
                if self.index_bounds[r] is not None and i < self.index_bounds[r]:
                    send_list[r].append(int(i))
                    send_to_ranks[idx] = r
                    break
        send_order = np.argsort(send_to_ranks, kind="stable")

        queries = Basis._point2point(send_list, self.comm)

        results: list[list[SlaterDeterminant]] = [[] for _ in range(self.comm.size)]
        for r in range(len(queries)):
            for query in queries[r]:
                if query >= self.offset and query < self.offset + len(self._keys):
                    results[r].append(self._keys.key_at(query - self.offset))

        result = [state for r_results in Basis._point2point(results, self.comm) for state in r_results]

        return (result[i] for i in np.argsort(send_order))

    def _local_index(self, state) -> int:
        """Global index of ``state`` if this rank owns it, else the miss sentinel ``self.size``.

        ``local_basis`` is kept sorted and ``local_indices`` is ``range(offset, offset + len)``,
        so the global index of an owned determinant is exactly ``offset + position`` -- which is
        what ``_index_dict`` stored. The dict is therefore derivable, not informative.

        **The sentinel is the whole subtlety.** A miss must return the *global* ``self.size``,
        never ``offset + len(local_basis)``: on any rank but the last non-empty one the latter is
        a perfectly valid global index belonging to another rank, and the retry loop in
        :meth:`_index_sequence` only re-queries results outside ``[0, size]``. A bogus-but-in-range
        answer would sail through it and land as a fabricated row or column in
        ``basis_transcription.build_sparse_matrix`` -- a wrong number, not a crash.
        """
        row = self._keys.find_row(state)
        return self.offset + row if row != len(self._keys) else self.size

    def _contains_local(self, state) -> bool:
        """Whether this rank owns ``state``; the membership half of :meth:`_local_index`."""
        return state in self._keys

    def _index_sequence(self, s: Iterable[SlaterDeterminant]) -> Iterator[int]:
        """Find the global indices for a sequence of states (hash-routed lookups)."""
        if not self.is_distributed:
            # Batched: one call into the extension for the whole sequence. A per-element
            # `find_row` is dominated by the Python/Cython call boundary rather than by the
            # search itself (measured), so the loop belongs on the other side of it.
            s = list(s)
            n_local = len(self._keys)
            return (self.offset + r if r != n_local else self.size for r in self._keys.find_rows(s))

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

        results: list[list[int]] = [[] for _ in range(self.comm.size)]
        for r in range(self.comm.size):
            n_local = len(self._keys)
            results[r] = [
                self.offset + row if row != n_local else self.size for row in self._keys.find_rows(queries[r])
            ]
        result = np.array([i for r_i in Basis._point2point(results, self.comm) for i in r_i], dtype=int)

        # The retry re-enters this method, which runs two `graph_alltoall` exchanges -- so
        # whether to retry is a COLLECTIVE decision and must not be taken from rank-local
        # state. Both of the old guards were rank-local: `len(result) > 0` is this rank's bra
        # count (a rank asked to resolve nothing skipped the loop entirely) and `np.any(...)`
        # is this rank's unresolved entries. A rank whose lookups all resolved would walk out
        # of the loop while another rank entered it and blocked in the exchange, waiting for a
        # partner that was never going to arrive. That is the failure mode CLAUDE.md's MPI
        # rules name first, and it was reachable here the moment the retry fired on a subset
        # of ranks.
        max_retries = 3
        for _ in range(max_retries):
            mask = np.logical_or(result > self.size, result < 0)
            # One scalar reduction, on every rank, every pass: the loop now runs the same
            # number of times everywhere. Ranks with nothing to re-resolve still enter the
            # recursive call with an empty sequence, because it is collective for them too.
            if not self.comm.allreduce(bool(mask.any()), op=MPI.LOR):
                break
            result[mask] = np.fromiter(
                self._index_sequence(itertools.compress(s, mask)), dtype=int, count=int(np.sum(mask))
            )
        else:
            if self.comm.rank == 0:
                import warnings

                warnings.warn(f"Failed to resolve all indices after {max_retries} retries", stacklevel=2)

        return (res for res in result[np.argsort(send_order)])

    def _contains_sequence(self, items) -> Iterator[bool]:
        """Check membership for a sequence of states."""
        if not self.is_distributed:
            return (self._contains_local(item) for item in items)
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
        self._keys = ManyBodyState(width=0)
        self.offset = 0
        self.size = 0
        self.local_indices = range(0, 0)
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
