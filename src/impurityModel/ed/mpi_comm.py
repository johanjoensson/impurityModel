"""
This module contains help functions for MPI communication.
"""

import hashlib
import math
import pickle
import time
from collections import OrderedDict
from itertools import islice

import numpy as np
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    SlaterDeterminant,
    pack_block_fused_cy,
    pack_determinants_cy,
    unpack_block_fused_cy,
    unpack_determinants_cy,
)

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size

# Cache of distributed-graph communicators for graph_alltoall_block. Building a dist_graph
# is a collective with real setup cost, so we keep one per *topology* rather than one per
# parent communicator.
#
# The earlier version kept a single entry per parent comm and rebuilt whenever the local
# neighbourhood differed from the previous call. Measured on a NiO 10-bath workload build,
# that missed 51% of the time -- 81 of 159 calls at 2 ranks, 79 of 155 at 3 -- because the
# neighbourhood is derived from whichever block is being redistributed and legitimately
# oscillates between self-only, all-ranks and empty. Every miss cost a collective
# Comm_free plus Create_dist_graph_adjacent on every rank. Keying on the topology turns
# that oscillation into hits, since the same few topologies recur.
#
# Keyed by id(parent comm) -> {global signature: graph comm}. The parent comm is pinned in
# the entry so its id stays valid while the cache holds it. Measured live count after
# mpi_infra+gf+lanczos at -n 3: 22 graph comms over 9 parents, against ~10 for the old
# single-entry design -- roughly 2x, not the 8x the cap allows, because most parents only ever
# see two or three distinct topologies.
_MAX_CACHED_GRAPHS = 8
_graph_comm_cache: "dict[int, tuple]" = {}


def _cached_dist_graph(comm, sources, destinations):
    """Return a dist_graph communicator for this ``(sources, destinations)`` neighbourhood,
    reusing a cached one whenever this exact topology has been seen before.

    Both ``Create_dist_graph_adjacent`` and ``Comm_free`` are collective, so every rank must
    make the same create/evict decisions in the same order. A rank's own neighbourhood is not
    enough to decide that -- the graph is the *collection* of neighbourhoods -- so the lookup
    key is a global signature built with one ``allgather`` of the per-rank neighbourhoods.
    Every rank therefore computes an identical key, sees an identical cache state, and takes
    an identical branch. That ``Allgather`` replaces the previous version's ``Allreduce(LOR)``,
    so the per-call collective *count* is unchanged; it moves from O(log P) to O(P) bytes,
    which is why the signature is hashed to a fixed 8 bytes per rank rather than gathering the
    neighbourhood lists themselves. What changes is how often the expensive rebuild behind it
    fires: measured 51% -> 3% on a NiO 10-bath build.

    Eviction is FIFO past ``_MAX_CACHED_GRAPHS``. It is safe for the same reason: insertion
    order is identical on every rank, so all ranks free the same communicator together.
    """
    key = id(comm)
    local_sig = (tuple(sources), tuple(destinations))
    # Fixed-width digest, gathered through the *buffer* Allgather. The obvious spelling --
    # comm.allgather(local_sig) -- pickles each rank's neighbourhood lists, which are O(P) long
    # at P ranks, so the payload would be O(P^2) per call on every one of ~155 calls. That trades
    # 80 expensive rebuilds for 155 quadratic gathers, and a 3-rank gate would never show it.
    # Hashing to 8 bytes first keeps the collective at O(P) bytes.
    payload = repr(local_sig).encode()
    digest = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    sig_local = np.array([digest], dtype=np.uint64)
    sig_all = np.empty(comm.size, dtype=np.uint64)
    comm.Allgather(sig_local, sig_all)
    global_sig = sig_all.tobytes()

    entry = _graph_comm_cache.get(key)
    if entry is None:
        entry = (comm, OrderedDict())
        _graph_comm_cache[key] = entry
    _pinned_comm, graphs = entry

    cached = graphs.get(global_sig)
    if cached is not None:
        return cached

    graph_comm = comm.Create_dist_graph_adjacent(sources, destinations, reorder=False)
    graphs[global_sig] = graph_comm
    # Bounded, so a workload that keeps producing fresh topologies cannot leak communicators
    # into MPI's context-id space. Identical on every rank, so the Free below is collective.
    while len(graphs) > _MAX_CACHED_GRAPHS:
        _evicted_sig, evicted = graphs.popitem(last=False)
        evicted.Free()
    return graph_comm


def dict_chunks_from_one_MPI_rank(data, chunk_maxsize=1 * 10**6, root=0):
    """
    Divide up data in chunks for one MPI rank.

    Yields chunks of data.
    Each chunk will contain a maximum number of elements,
    which is determined by the user.
    The other MPI ranks yields the same number of chunks,
    but each such chunk is equal to None.

    Parameters
    ----------
    data : dict
    chunk_maxsize : int
    root : int

    """
    if rank == root:
        it = iter(data)
        n_chunks = math.ceil(len(data) / chunk_maxsize)
    else:
        n_chunks = None
    n_chunks = comm.bcast(n_chunks, root=root)
    for _ in range(n_chunks):
        if rank == root:
            yield {k: data[k] for k in islice(it, chunk_maxsize)}
        else:
            yield None


def allgather_dict(data, total, chunk_maxsize=1 * 10**6):
    """
    Distribute data from all ranks to all ranks into variable total.

    The function performs "Allgather".
    However, since Allgather requires the same amount of data
    for all MPI ranks, it's done through simpler communications.

    Parameters
    ----------
    data : dict
        Contains different information for each MPI rank.
        Unique keys for each rank, i.e.
        a key for rank r does not exist as a key in data
        for any other rank other than rank r.
        Neither does it exist in the variable total.
    total : dict
        Will be updated with data from all MPI ranks.
    chunk_maxsize : int
        The maximum number of dictionary elements to send at once.

    """
    # Measure time for constructing H in matrix form
    t0 = time.perf_counter()
    # Number of elements for each rank.
    n_ps_new = np.zeros(ranks, dtype=np.int64)
    comm.Allgather(np.array([len(data)], dtype=np.int64), n_ps_new)
    # Determine here if we can use a simple Allgather or need
    # to send the data in chunks.
    if max(n_ps_new) <= chunk_maxsize:
        if rank == 0:
            print("Allgather everything at once...")
        for r in range(ranks):
            total.update(comm.bcast(data, root=r))
    else:
        if rank == 0:
            print("Allgather chunks...")
        # MPI do not allow to messages bigger than about 2 GB.
        # Therefore we send the data in chunks.
        for r in range(ranks):
            # Data in rank r is broadcasted in chunks to all the other ranks.
            for chunk in dict_chunks_from_one_MPI_rank(data, chunk_maxsize, r):
                total.update(comm.bcast(chunk, root=r))
    if rank == 0:
        print("time(Allgather H_dict) = {:.5f} seconds.".format(time.perf_counter() - t0))


def is_empty(x):
    """
    Check if a structure is empty.

    A structure is considered empty if it is None, a zero-length list,
    dict, or set, or a list containing only empty lists, dicts, or sets.

    Parameters
    ----------
    x : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is empty, False otherwise.
    """
    if x is None:
        return True
    if isinstance(x, list):
        if len(x) == 0:
            return True
        if all(isinstance(i, (dict, list, set)) for i in x):
            return all(len(i) == 0 for i in x)
        return False
    if isinstance(x, (dict, set)):
        return len(x) == 0
    return False


def empty_clone(x):
    """
    Create an empty clone of the given data structure.

    Preserves the nested list/dict/set structure but strips the values.

    Parameters
    ----------
    x : Any
        The structure to clone.

    Returns
    -------
    Any
        An empty structure matching the container type of `x`.
    """
    if x is None:
        return None
    if isinstance(x, list):
        if len(x) > 0 and all(isinstance(i, dict) for i in x):
            return [{} for _ in x]
        if len(x) > 0 and all(isinstance(i, list) for i in x):
            return [[] for _ in x]
        return []
    if isinstance(x, dict):
        return {}
    if isinstance(x, set):
        return set()
    return None


def graph_alltoall(send_list, comm):
    """
    Perform sparse all-to-all communication of Python objects.

    Uses ``MPI_Dist_graph_create_adjacent`` (via
    ``comm.Create_dist_graph_adjacent``) to build a neighbourhood
    communicator that only connects ranks which actually exchange data.
    The actual transfer is done with ``Neighbor_alltoallv`` over raw
    bytes, so only participating pairs pay communication cost.

    If the communicator has 0 or 1 ranks the list is returned unchanged.

    Parameters
    ----------
    send_list : list[any]
        List of length ``comm.size``.  Element ``r`` is the object to
        send to rank ``r``.  Empty/None elements are skipped.
    comm : MPI.Comm

    Returns
    -------
    list[any]
        List of length ``comm.size``.  Element ``r`` is the object
        received from rank ``r``, or an empty clone of ``send_list[r]``
        when nothing was received.
    """
    if comm is None or comm.size <= 1:
        return send_list

    size = comm.size

    # ------------------------------------------------------------------
    # Step 1 - serialise the non-empty payloads and record their sizes.
    # ------------------------------------------------------------------
    send_bytes = [None] * size
    send_sizes = np.zeros(size, dtype=np.int64)
    for r in range(size):
        if not is_empty(send_list[r]):
            send_bytes[r] = pickle.dumps(send_list[r])
            send_sizes[r] = len(send_bytes[r])

    # ------------------------------------------------------------------
    # Step 2 - exchange sizes with a lean Alltoall so every rank learns
    #           how many bytes it will receive from every other rank.
    # ------------------------------------------------------------------
    recv_sizes = np.empty(size, dtype=np.int64)
    comm.Alltoall(send_sizes, recv_sizes)

    # ------------------------------------------------------------------
    # Step 3 - identify the sparse send/receive neighbourhood.
    # ------------------------------------------------------------------
    destinations = [r for r in range(size) if send_sizes[r] > 0]
    sources = [r for r in range(size) if recv_sizes[r] > 0]

    # ------------------------------------------------------------------
    # Step 4 - build a distributed-graph communicator so that only the
    #           actual (source, destination) pairs are represented.
    #           MPI_Dist_graph_create_adjacent lets MPI optimise the
    #           collective over this sparse neighbourhood.
    # ------------------------------------------------------------------
    graph_comm = comm.Create_dist_graph_adjacent(
        sources,  # in-edges  (ranks that will send to me)
        destinations,  # out-edges (ranks I will send to)
        reorder=False,
    )

    # ------------------------------------------------------------------
    # Step 5 - pack send buffer (bytes for each destination in order).
    # ------------------------------------------------------------------
    send_buf = bytearray().join(send_bytes[r] for r in destinations if send_bytes[r] is not None)
    s_counts = np.array([int(send_sizes[r]) for r in destinations], dtype=np.int64)
    s_displs = np.concatenate(([0], np.cumsum(s_counts[:-1]))) if len(s_counts) > 0 else np.array([], dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 6 - allocate receive buffer.
    # ------------------------------------------------------------------
    r_counts = np.array([int(recv_sizes[r]) for r in sources], dtype=np.int64)
    r_displs = np.concatenate(([0], np.cumsum(r_counts[:-1]))) if len(r_counts) > 0 else np.array([], dtype=np.int64)
    recv_buf = bytearray(int(np.sum(r_counts)))

    # ------------------------------------------------------------------
    # Step 7 - neighbourhood all-to-all-v over the sparse graph.
    # ------------------------------------------------------------------
    graph_comm.Neighbor_alltoallv(
        [send_buf, s_counts, s_displs, MPI.BYTE],
        [recv_buf, r_counts, r_displs, MPI.BYTE],
    )

    # ------------------------------------------------------------------
    # Step 8 - free the neighbourhood communicator.
    # ------------------------------------------------------------------
    graph_comm.Free()

    # ------------------------------------------------------------------
    # Step 9 - deserialise and build result list.
    # ------------------------------------------------------------------
    result = [None] * size
    offset = 0
    for r, cnt in zip(sources, r_counts):
        result[r] = pickle.loads(recv_buf[offset : offset + cnt])
        offset += cnt

    for r in range(size):
        if result[r] is None:
            result[r] = empty_clone(send_list[r])

    return result


def distribute_determinants(
    dets: "list[SlaterDeterminant]",
    n_bytes: int,
    comm: "MPI.Intracomm",  # graph-topology helpers (Create_dist_graph_adjacent) live on Intracomm
) -> "list[list[SlaterDeterminant]]":
    """
    Partition and distribute SlaterDeterminants across MPI ranks.
    """
    if comm is None or comm.size <= 1:
        return [dets]

    size = comm.size
    chunks_per_state = (n_bytes + 7) // 8

    # 1. Pack in Cython (handles hash partitioning)
    send_counts, state_send = pack_determinants_cy(dets, size)

    # 2. Exchange counts
    recv_counts = np.empty(size, dtype=np.int64)
    comm.Alltoall(send_counts, recv_counts)

    # 3. Build graph
    destinations = [r for r in range(size) if send_counts[r] > 0]
    sources = [r for r in range(size) if recv_counts[r] > 0]
    graph_comm = comm.Create_dist_graph_adjacent(sources, destinations, reorder=False)

    # 4. Buffers
    s_counts_nb = np.array([send_counts[r] for r in destinations], dtype=np.int64)
    s_displs_nb = (
        np.concatenate(([0], np.cumsum(s_counts_nb[:-1]))) if len(s_counts_nb) else np.array([], dtype=np.int64)
    )

    total_recv = int(np.sum(recv_counts))
    state_recv = np.empty(total_recv * chunks_per_state, dtype=np.uint64)

    r_counts_nb = np.array([recv_counts[r] for r in sources], dtype=np.int64)
    r_displs_nb = (
        np.concatenate(([0], np.cumsum(r_counts_nb[:-1]))) if len(r_counts_nb) else np.array([], dtype=np.int64)
    )

    # 5. Exchange
    graph_comm.Neighbor_alltoallv(
        [state_send, s_counts_nb * chunks_per_state * 8, s_displs_nb * chunks_per_state * 8, MPI.BYTE],
        [state_recv, r_counts_nb * chunks_per_state * 8, r_displs_nb * chunks_per_state * 8, MPI.BYTE],
    )
    graph_comm.Free()

    # 6. Unpack
    if total_recv > 0:
        return unpack_determinants_cy(size, recv_counts, state_recv, chunks_per_state)
    else:
        return [[] for _ in range(size)]


def graph_alltoall_block(
    block: "ManyBodyState",
    n_bytes: int,
    comm: "MPI.Comm",
) -> "ManyBodyState":
    """Redistribute a shared-support block state across MPI ranks (Phase 2.3).

    One wire entry per shared-support ROW (``[det | width x complex amp]``, no
    per-entry psi index), ``routing_hash`` ownership, a cached dist-graph + single
    fused ``Neighbor_alltoallv(MPI.BYTE)``. Rows for the same determinant arriving
    from several ranks are summed per column. The one redistribution primitive
    ``Basis.redistribute_psis``/``redistribute_block`` route through -- every state
    is a block (``p == 1`` an ordinary case), so there is no separate flat-list path.
    """
    if comm is None or comm.size <= 1:
        return block.copy()

    size = comm.size
    width = int(block.width)
    chunks_per_state = (n_bytes + 7) // 8
    # One interleaved entry = state chunks + width complex amps.
    bytes_per_entry = chunks_per_state * 8 + width * 16

    # 1. Cython packing into a single rank-ordered byte buffer (counts are rows/rank).
    send_counts, send_buf = pack_block_fused_cy(block, size, chunks_per_state)

    # 2. Exchange counts
    recv_counts = np.empty(size, dtype=np.int64)
    comm.Alltoall(send_counts, recv_counts)

    # 3. Reuse (or build) the graph communicator over the send/recv neighbourhood.
    destinations = [r for r in range(size) if send_counts[r] > 0]
    sources = [r for r in range(size) if recv_counts[r] > 0]
    graph_comm = _cached_dist_graph(comm, sources, destinations)

    s_counts_nb = np.array([send_counts[r] for r in destinations], dtype=np.int64)
    s_displs_nb = (
        np.concatenate(([0], np.cumsum(s_counts_nb[:-1]))) if len(s_counts_nb) else np.array([], dtype=np.int64)
    )

    # 4. Allocate the single receive byte buffer
    total_recv = int(np.sum(recv_counts))
    recv_buf = np.empty(total_recv * bytes_per_entry, dtype=np.uint8)

    r_counts_nb = np.array([recv_counts[r] for r in sources], dtype=np.int64)
    r_displs_nb = (
        np.concatenate(([0], np.cumsum(r_counts_nb[:-1]))) if len(r_counts_nb) else np.array([], dtype=np.int64)
    )

    # 5. One fused exchange (BYTE); all ranks participate unconditionally.
    graph_comm.Neighbor_alltoallv(
        [send_buf, s_counts_nb * bytes_per_entry, s_displs_nb * bytes_per_entry, MPI.BYTE],
        [recv_buf, r_counts_nb * bytes_per_entry, r_displs_nb * bytes_per_entry, MPI.BYTE],
    )

    # 6. Unpack into a fresh block (duplicate rows summed in arrival order).
    return unpack_block_fused_cy(size, width, recv_counts, recv_buf, chunks_per_state)


def gather_distributed_results(
    comm, sub_comm_rank, roots, items_per_color, local_res, is_array=True, shape=None, dtype=None
):
    """
    Gather results computed across sub-communicators into the root communicator (rank 0).

    Parameters
    ----------
    comm : MPI.Comm
        The global communicator.
    sub_comm_rank : int
        The rank of the local process in its sub-communicator.
    roots : list of int
        The global rank of the root process for each sub-communicator color.
    items_per_color : list of int
        The number of items (e.g. frequencies) handled by each color.
    local_res : ndarray or list
        The local result to be sent.
    is_array : bool, optional
        True if the data is a numpy array (uses comm.Recv/Send), False if python list (uses comm.recv/send).
    shape : tuple, optional
        The shape of the array to gather (if is_array is True). If not provided, it will be inferred from local_res.
    dtype : np.dtype, optional
        The data type (if is_array is True). If not provided, it will be inferred from local_res.

    Returns
    -------
    all_res : ndarray or list or None
        The gathered results on global rank 0, or None on other ranks.
    """
    if comm is None or comm.size <= 1:
        return local_res

    if comm.rank == 0:
        if is_array:
            if shape is None:
                shape = local_res.shape[1:] if len(local_res.shape) > 1 else ()
            if dtype is None:
                dtype = local_res.dtype
            total_items = sum(items_per_color)
            all_res = np.empty((total_items,) + shape, dtype=dtype)
        else:
            all_res = []

        offsets = [0] + list(np.cumsum(items_per_color))[:-1]

        for color, (count, root) in enumerate(zip(items_per_color, roots)):
            if count == 0:
                continue

            if root == 0:
                if is_array:
                    all_res[offsets[color] : offsets[color] + count] = local_res
                else:
                    all_res.extend(local_res)
            else:
                if is_array:
                    buf = np.empty((count,) + shape, dtype=dtype)
                    comm.Recv(buf, source=root)
                    all_res[offsets[color] : offsets[color] + count] = buf
                else:
                    res = comm.recv(source=root)
                    all_res.extend(res)
        return all_res
    else:
        if sub_comm_rank == 0:
            if is_array:
                comm.Send(np.asarray(local_res), dest=0)
            else:
                comm.send(local_res, dest=0)
        return None


def get_job_tasks(rank, ranks, tasks_tot):
    """
    Return a tuple of job task indices for a particular rank.

    This function distribute the job tasks in tasks_tot
    over all the ranks.

    Note
    ----
    This is a primerly a MPI help function.

    Parameters
    ----------
    rank : int
        Current MPI rank/worker.
    ranks : int
        Number of MPI ranks/workers in total.
    tasks_tot : list
        List of task indices.
        Length is the total number of job tasks.

    """
    n_tot = len(tasks_tot)
    nj = n_tot // ranks
    rest = n_tot % ranks
    tasks = [tasks_tot[i] for i in range(nj * rank, nj * rank + nj)]
    if rank < rest:
        tasks.append(tasks_tot[n_tot - rest + rank])
    return tuple(tasks)
