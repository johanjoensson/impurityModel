"""
Adaptive MPI splitting of a :class:`~impurityModel.ed.manybody_basis.Basis`
into per-color sub-communicators, with redistribution of wavefunctions onto
the split bases. `_pack_units` is the pure packing math (color cap, LPT unit
packing, largest-remainder rank apportionment); every rank computes the
identical packing. `split_basis_and_redistribute_psi` is collective over
``basis.comm`` and frees its intercommunicators collectively before returning.
"""

from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState


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
    normalized /= np.sum(normalized)
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
) -> tuple[list[int], list[int], int, list[int], Basis, Optional[list[ManyBodyState]], list[Optional[MPI.Intercomm]]]:
    """Split the basis and redistribute wavefunctions over a split communicator.

    Parameters
    ----------
    priorities : list of float
        The split priority weights for each block.
    psis : list of ManyBodyState, optional
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
    intercomms : list of MPI.Intercomm
        MPI intercommunicators (all set to None after being freed).
    """

    if (not basis.is_distributed) or len(priorities) <= 1:
        return list(range(len(priorities))), [0], 0, [len(priorities)], basis, psis, [None]

    comm = basis.comm
    # All packing math (participation-ratio color cap, LPT unit packing,
    # largest-remainder rank apportionment) lives in _pack_units; it is pure and
    # deterministic, so every rank computes the identical packing.
    subgroups, procs_per_color = _pack_units(priorities, comm.size, basis.split_threshold, max_colors)
    if subgroups is None:
        # Unified: all ranks process every block together (no actual split).
        return list(range(len(priorities))), [0], 0, [len(priorities)], basis, psis, [None]

    # _pack_units returns (None, None) or two non-None values together; the guard above
    # rules out the None case, so procs_per_color is a real array from here on.
    assert procs_per_color is not None
    proc_cutoffs = np.cumsum(procs_per_color)
    color = int(np.argmax(comm.rank < proc_cutoffs))

    split_comm = comm.Split(color=color, key=comm.rank)
    split_roots = [0] + proc_cutoffs[:-1].tolist()
    items_per_color = [len(subgroup) for subgroup in subgroups]
    assert sum(items_per_color) == len(priorities)

    # None for this rank's own color; a real intercommunicator for every other color.
    intercomms: list[MPI.Intercomm | None] = []
    for c, c_root in enumerate(split_roots):
        if c == color:
            intercomms.append(None)
            continue
        intercomms.append(split_comm.Create_intercomm(0, comm, c_root))
    indices = sorted(subgroups[color])

    if split_comm.rank == 0:
        assert comm.rank in split_roots

    new_states = set(basis.local_basis)
    # Distribute my local basis states among all other colors
    for c, _c_root in enumerate(split_roots):
        # I will send  states to this color
        if color != c:
            serialized_local_basis = bytearray().join(
                state.to_bytearray()[: basis.n_bytes] for state in basis.local_basis
            )
            target = intercomms[c]
            assert target is not None  # never None on the color != c branch
            target.send(serialized_local_basis, dest=split_comm.rank % procs_per_color[c])
        # I will receive states from all other colors
        else:
            for send_color in range(len(split_roots)):
                if send_color == color:
                    continue
                for sender in range(procs_per_color[send_color]):
                    if sender % procs_per_color[c] != split_comm.rank:
                        continue
                    source_comm = intercomms[send_color]
                    assert source_comm is not None
                    received_bytes = source_comm.recv(source=sender)
                    new_states.update(
                        basis.type.from_bytes(bytes(received_bytes[i : i + basis.n_bytes]))
                        for i in range(0, len(received_bytes), basis.n_bytes)
                    )

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
        new_psis = [p.copy() for p in psis]
        for c, _c_root in enumerate(split_roots):
            if color != c:
                serialized_psis = [{bytes(k.to_bytearray()[: basis.n_bytes]): v for k, v in p.items()} for p in psis]
                target = intercomms[c]
                assert target is not None  # never None on the color != c branch
                target.send(serialized_psis, dest=split_comm.rank % procs_per_color[c])
            else:
                for send_color in range(len(split_roots)):
                    if send_color == color:
                        continue
                    for sender in range(procs_per_color[send_color]):
                        if sender % procs_per_color[color] != split_comm.rank:
                            continue
                        source_comm = intercomms[send_color]
                        assert source_comm is not None
                        received_psis = source_comm.recv(source=sender)
                        for i, received_psi in enumerate(received_psis):
                            new_psis[i] += ManyBodyState({basis.type.from_bytes(k): v for k, v in received_psi.items()})
        psis = split_basis.redistribute_psis(new_psis)

    # Free the intercommunicators collectively while all ranks are still
    # synchronised here.  MPI_Comm_free is collective — leaving the objects
    # for Python gc means they may be freed at different times on different
    # ranks, causing crashes.  The split_comm itself (split_basis.comm) must
    # NOT be freed here because the caller still needs split_basis.
    for ic in intercomms:
        if ic is not None and ic != MPI.COMM_NULL:
            ic.Free()

    return indices, split_roots, color, items_per_color, split_basis, psis, [None] * len(intercomms)
