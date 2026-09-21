"""
Adaptive MPI splitting of a :class:`~impurityModel.ed.manybody_basis.Basis`
into per-color sub-communicators, with redistribution of wavefunctions onto
the split bases. `_pack_units` is the pure packing math (color cap, LPT unit
packing, largest-remainder rank apportionment); every rank computes the
identical packing. `split_basis_and_redistribute_psi` is collective over
``basis.comm``: it replicates the basis (and the seeds) into every color with
one sparse all-to-all per phase (`mpi_comm.graph_alltoall`), then lets each
color's `Basis` hash-repartition its copy over the split sub-communicator.
"""

from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.mpi_comm import graph_alltoall


def _pack_units(
    weights, comm_size: int, split_threshold: float, max_colors: Optional[int] = None
) -> tuple[Optional[list[tuple[int, ...]]], Optional[np.ndarray]]:
    """Pack work units into per-color bins and allocate ranks to each color.

    Pure packing math behind :meth:`Basis.split_basis_and_redistribute_psi` — no MPI,
    so it is unit-testable, and every rank computes the identical packing from the
    (already Allreduced) weights.

    The number of colors is capped by the participation ratio (Σw)²/Σw² — the
    effective number of equally-weighted units — scaled by ``split_threshold``, so a
    few dominant units are not starved of ranks: better to run them on a larger
    sub-communicator (or unified). ``split_threshold=0`` forces a single unified
    communicator; ``=1`` is the legacy max-split for equal weights.

    Units are packed with LPT (Longest Processing Time): the next-heaviest unit goes
    to the currently lightest bin, ties to the lowest bin index. This bounds the
    heaviest bin at 4/3 of the optimal packing and reduces to round-robin dealing on
    uniform weights. Ranks are then apportioned to bins proportionally to bin mass by
    largest remainder, every bin keeping at least one rank.

    Parameters
    ----------
    weights : array_like of float
        Cost weight per unit (identical on every rank).
    comm_size : int
        Number of MPI ranks to distribute over.
    split_threshold : float
        Scale factor on the participation-ratio cap of the number of colors.
    max_colors : int, optional
        Hard cap on the color count (e.g. the memory budget cap from
        :func:`impurityModel.ed.memory_estimate.max_colors_within_budget` — every
        simultaneous color may fill the same ``truncation_threshold``, so memory can
        bound the concurrency below what the participation ratio allows).

    Returns
    -------
    subgroups : list of tuple of int, or None
        Unit indices assigned to each color; ``None`` when the packing collapses to
        a single color (the caller should not split).
    procs_per_color : ndarray of int, or None
        Ranks per color, each at least 1, summing to ``comm_size``.
    """
    normalized = np.abs(np.asarray(weights, dtype=float))
    total = np.sum(normalized)
    if not np.isfinite(total) or total <= 0.0:
        # A NaN/inf weight silently reorders the argsort below and a zero sum makes the
        # normalization NaN; either way ranks would disagree on the packing. Fail loudly
        # and identically on every rank instead (weights are already Allreduced, so this
        # check is rank-invariant).
        raise ValueError(f"_pack_units: weights must be finite and sum to a positive value, got sum {total}")
    normalized /= total
    n_colors = min(comm_size, len(normalized))
    participation = 1.0 / np.sum(normalized**2)
    n_colors = min(n_colors, max(1, int(np.ceil(participation * split_threshold))))
    if max_colors is not None:
        n_colors = min(n_colors, max(1, max_colors))
    if n_colors <= 1:
        return None, None

    # LPT packing. The first n_colors units land in distinct empty bins, so no bin
    # is ever empty (n_colors <= number of units).
    sorted_idxs = np.argsort(normalized, kind="stable")[::-1]
    subgroups: list[tuple[int, ...]] = [tuple() for _ in range(n_colors)]
    bin_mass = np.zeros(n_colors)
    for u in sorted_idxs:
        c = int(np.argmin(bin_mass))
        subgroups[c] += (int(u),)
        bin_mass[c] += normalized[u]

    # Largest-remainder rank apportionment on the bin masses (they sum to 1).
    raw = comm_size * bin_mass
    floors = np.floor(raw).astype(int)
    procs_per_color = np.maximum(floors, 1)
    remainder = comm_size - int(np.sum(procs_per_color))
    if remainder > 0:
        # The floors sum to within n_colors of comm_size, so one pass over the
        # largest fractional parts places every leftover rank.
        order = np.argsort(-(raw - floors), kind="stable")
        procs_per_color[order[:remainder]] += 1
    else:
        # The max(1, .) floors over-allocated: reclaim ranks from the lightest bins
        # that can spare one, so the heaviest bins keep their proportional share.
        order = np.argsort(bin_mass, kind="stable")
        while remainder < 0:
            reclaimed = False
            for c in order:
                if procs_per_color[c] > 1:
                    procs_per_color[c] -= 1
                    remainder += 1
                    reclaimed = True
                    if remainder == 0:
                        break
            assert reclaimed, "rank apportionment failed to converge"
    assert np.sum(procs_per_color) == comm_size
    return subgroups, procs_per_color


def split_basis_and_redistribute_psi(
    basis, priorities: list[float] | np.ndarray, psis: Optional[list[ManyBodyState]], max_colors: Optional[int] = None
) -> tuple[list[int], list[int], int, list[int], Basis, Optional[list[ManyBodyState]]]:
    """Split the basis and redistribute wavefunctions over a split communicator.

    Parameters
    ----------
    priorities : list of float
        The split priority weights for each block.
    psis : list of width-1 ManyBodyState, optional
        The wavefunctions to redistribute, or None.
    max_colors : int, optional
        Hard cap on the number of colors (see :func:`_pack_units`); must be identical
        on every rank of ``basis.comm``.

    Returns
    -------
    indices : list of int
        Representative indices.
    split_roots : list of int
        The roots for the split communicators.
    color : int
        The split communicator color rank.
    items_per_color : list of int
        Number of items assigned to each color group.
    split_basis : Basis
        The new Basis associated with the split communicator.
    psis : list of ManyBodyState, optional
        Redistributed wavefunctions.
    """

    if (not basis.is_distributed) or len(priorities) <= 1:
        return list(range(len(priorities))), [0], 0, [len(priorities)], basis, psis

    comm = basis.comm
    # All packing math (participation-ratio color cap, LPT unit packing,
    # largest-remainder rank apportionment) lives in _pack_units; it is pure and
    # deterministic, so every rank computes the identical packing.
    subgroups, procs_per_color = _pack_units(priorities, comm.size, basis.split_threshold, max_colors)

    # Every send/receive target below is derived from `procs_per_color`, so the packing
    # MUST be bit-identical on every rank. It is a pure function of (already Allreduced)
    # inputs, but nothing else checks that -- and a divergence deadlocks rather than
    # errors (a rank that computes the unified packing returns just below while the
    # others enter `comm.Split`). Verify here, before that early return, with an
    # unconditional collective on every rank (CLAUDE.md MPI rule).
    n_colors_local = 1 if procs_per_color is None else len(procs_per_color)
    if comm.allreduce(n_colors_local, op=MPI.MIN) != comm.allreduce(n_colors_local, op=MPI.MAX):
        raise RuntimeError(
            f"split_basis_and_redistribute_psi: ranks disagree on the unit packing "
            f"(rank {comm.rank} computed {n_colors_local} color(s)); the _pack_units inputs "
            f"(unit weights / max_colors / split_threshold) are not rank-invariant"
        )
    if procs_per_color is not None:
        ppc = np.ascontiguousarray(procs_per_color)
        lo = np.empty_like(ppc)
        hi = np.empty_like(ppc)
        comm.Allreduce(ppc, lo, op=MPI.MIN)
        comm.Allreduce(ppc, hi, op=MPI.MAX)
        if not (np.array_equal(lo, ppc) and np.array_equal(hi, ppc)):
            raise RuntimeError(
                f"split_basis_and_redistribute_psi: ranks disagree on procs_per_color "
                f"(rank {comm.rank} computed {list(ppc)}; communicator min {list(lo)}, max {list(hi)})"
            )

    if subgroups is None:
        # Unified: all ranks process every block together (no actual split).
        return list(range(len(priorities))), [0], 0, [len(priorities)], basis, psis

    # _pack_units returns (None, None) or two non-None values together; the guard above
    # rules out the None case, so procs_per_color is a real array from here on.
    assert procs_per_color is not None
    proc_cutoffs = np.cumsum(procs_per_color)
    color = int(np.argmax(comm.rank < proc_cutoffs))

    split_comm = comm.Split(color=color, key=comm.rank)
    split_roots = [0] + proc_cutoffs[:-1].tolist()
    items_per_color = [len(subgroup) for subgroup in subgroups]
    assert sum(items_per_color) == len(priorities)

    indices = sorted(subgroups[color])

    if split_comm.rank == 0:
        assert comm.rank in split_roots

    # Pure-Python ints: routing_hash() is a uint64 that can exceed C long, and
    # `big_int % np.int64` overflows on the numpy coercion.
    ppc_int = [int(p) for p in procs_per_color]
    roots_int = [int(r) for r in split_roots]

    def _group_by_destination(routing_hashes):
        """``(destination rank, row indices)`` for every other color, rows in table order.

        Every determinant I own goes to its owner-to-be in each color other than mine (my
        color already holds it); routing to the eventual within-color owner makes the
        ``Basis`` re-partition below a no-op for that determinant. Per determinant that
        owner is ``roots_int[other] + routing_hash % ppc_int[other]``, which is the scalar
        spelling the tests use as an oracle.

        The modulus is taken in ``uint64`` throughout: ``routing_hash()`` does not fit a C
        long, which is why the scalar path insisted on pure-Python ints rather than letting
        numpy coerce to ``int64``, and ``uint64 % uint64`` has the same no-overflow
        property. Grouping is a stable argsort, so each destination receives its rows in the
        order they were enumerated -- the order the receiving side accumulates duplicates
        in, and therefore the order the sum is taken in.
        """
        for other in range(len(roots_int)):
            if other == color:
                continue
            dest = (routing_hashes % np.uint64(ppc_int[other])).astype(np.int64) + roots_int[other]
            if dest.size == 0:
                continue
            order = np.argsort(dest, kind="stable")
            bounds = np.flatnonzero(np.diff(dest[order])) + 1
            for rows in np.split(order, bounds):
                yield int(dest[rows[0]]), rows

    # Replicate the full basis into every color with one sparse all-to-all over
    # ``comm``: every determinant I own goes to its owner-to-be in each other color.
    #
    # The payload is a packed key array per destination, not a list of `bytes` objects.
    # `graph_alltoall` pickles a numpy array as a raw buffer and a Python object as objects
    # on BOTH sides -- the distinction that decided `compute_impurity_rdm`, where the wire
    # format was only 1.4x and materializing the objects was the other 16x. Measured here
    # at 200k determinants: 87.2 B/entry to build the list form, 0.0 for the array.
    n_key_bytes = basis.n_bytes
    n_local = len(basis.local_basis)
    local_keys = np.empty((n_local, n_key_bytes), dtype=np.uint8)
    local_hashes = np.empty(n_local, dtype=np.uint64)
    for i, state in enumerate(basis.local_basis):
        local_keys[i] = np.frombuffer(bytes(state.to_bytearray()[:n_key_bytes]), dtype=np.uint8)
        local_hashes[i] = state.routing_hash()

    # A dict, never a bare array: `is_empty` only understands None/list/dict/set, and
    # `if chunk:` on a numpy array of more than one element raises rather than answering.
    det_send: list = [None] * comm.size
    for g, rows in _group_by_destination(local_hashes):
        det_send[g] = {"keys": local_keys[rows]}
    del local_keys, local_hashes

    new_states = set(basis.local_basis)
    for chunk in graph_alltoall(det_send, comm):
        if not chunk:
            continue
        received = chunk["keys"]
        new_states.update(basis.type.from_bytes(received[j].tobytes()) for j in range(received.shape[0]))
    del det_send

    split_basis = Basis(
        basis.impurity_orbitals,
        basis.bath_states,
        initial_basis=list(new_states),
        restrictions=basis.restrictions,
        weighted_restrictions=basis.weighted_restrictions,
        split_threshold=basis.split_threshold,
        chain_restrict=basis.chain_restrict,
        collapse_chains=basis.collapse_chains,
        comm=split_comm,
        verbose=basis.verbose,
        truncation_threshold=basis.truncation_threshold,
        tau=basis.tau,
        spin_flip_dj=basis.spin_flip_dj,
    )

    if psis is not None:
        # ManyBodyState is block storage throughout; the split-basis wire format only
        # supports width-1 (same convention as Basis.redistribute_psis). The payload
        # carries plain scalars (``v[0]`` unwraps the Row).
        for p in psis:
            if p.width != 1:
                raise ValueError(f"split_basis_and_redistribute_psi: expected width-1 blocks, got width {p.width}")
        # Same routing as the determinants: each (seed, determinant, amplitude) entry to
        # its owner-to-be in every other color; redistribute_psis then finalises placement
        # on the split communicator (and sums any duplicate rows).
        # One flat table over all seeds, sliced per destination -- the same packing as the
        # determinants above, and the bigger half of it: a `(int, bytes, complex)` tuple per
        # entry measured 158.3 B against 16.1 packed, 9.8x, at 800k entries.
        n_rows = sum(len(p) for p in psis)
        seed_index = np.empty(n_rows, dtype=np.int32)
        seed_keys = np.empty((n_rows, basis.n_bytes), dtype=np.uint8)
        seed_amps = np.empty(n_rows, dtype=complex)
        seed_hashes = np.empty(n_rows, dtype=np.uint64)
        at = 0
        for i, p in enumerate(psis):
            for k, v in p.items():
                seed_index[at] = i
                seed_keys[at] = np.frombuffer(bytes(k.to_bytearray()[: basis.n_bytes]), dtype=np.uint8)
                seed_amps[at] = v[0]
                seed_hashes[at] = k.routing_hash()
                at += 1
        assert at == n_rows, f"psi table filled {at} of {n_rows} rows"

        psi_send: list = [None] * comm.size
        for g, rows in _group_by_destination(seed_hashes):
            psi_send[g] = {"i": seed_index[rows], "keys": seed_keys[rows], "amp": seed_amps[rows]}
        del seed_index, seed_keys, seed_amps, seed_hashes

        received_rows: list[dict] = [dict() for _ in psis]
        for chunk in graph_alltoall(psi_send, comm):
            if not chunk:
                continue
            chunk_i, chunk_keys, chunk_amp = chunk["i"], chunk["keys"], chunk["amp"]
            for j in range(chunk_i.shape[0]):
                d = received_rows[int(chunk_i[j])]
                sd = basis.type.from_bytes(chunk_keys[j].tobytes())
                # `complex(...)`, not the numpy scalar: the accumulated values go into a
                # `ManyBodyState`, and the rows are summed in the same order as before, so
                # keeping the Python type keeps the arithmetic identical too.
                d[sd] = d.get(sd, 0) + complex(chunk_amp[j])
        del psi_send
        new_psis = [p.copy() for p in psis]
        for i, rows in enumerate(received_rows):
            if rows:
                new_psis[i] += ManyBodyState(rows)
        psis = split_basis.redistribute_psis(*new_psis)

    return indices, split_roots, color, items_per_color, split_basis, psis
