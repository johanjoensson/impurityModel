"""
This module contains help functions for MPI communication.
"""

import math
import pickle
import time
from itertools import islice

import numpy as np
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size

# Cache of distributed-graph communicators for graph_alltoall_psis, keyed by id(parent
# comm). Building a dist_graph is a collective with real setup cost; at 100s-1000s of
# ranks rebuilding it every matvec dominates. We reuse it whenever the per-rank
# (sources, destinations) neighbourhood is unchanged. The parent comm is pinned in the
# cache value so its id stays valid while the entry lives.
_graph_comm_cache: "dict[int, tuple]" = {}


def _cached_dist_graph(comm, sources, destinations):
    """Return a dist_graph communicator for this (sources, destinations) neighbourhood,
    reusing a cached one when the topology is unchanged.

    The rebuild decision is made **collectively** via ``Allreduce(LOR)``: if *any* rank's
    neighbourhood changed, *all* ranks rebuild together. This keeps the collective
    ``Create_dist_graph_adjacent`` call consistent across ranks (no deadlock), regardless
    of how each rank's local topology shifted.
    """
    key = id(comm)
    src_t = tuple(sources)
    dst_t = tuple(destinations)
    cached = _graph_comm_cache.get(key)
    changed_local = cached is None or cached[0] != src_t or cached[1] != dst_t
    changed = comm.allreduce(changed_local, op=MPI.LOR)
    if changed:
        if cached is not None:
            cached[2].Free()
        graph_comm = comm.Create_dist_graph_adjacent(sources, destinations, reorder=False)
        _graph_comm_cache[key] = (src_t, dst_t, graph_comm, comm)
        return graph_comm
    return cached[2]


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
    # Step 1 – serialise the non-empty payloads and record their sizes.
    # ------------------------------------------------------------------
    send_bytes = [None] * size
    send_sizes = np.zeros(size, dtype=np.int64)
    for r in range(size):
        if not is_empty(send_list[r]):
            send_bytes[r] = pickle.dumps(send_list[r])
            send_sizes[r] = len(send_bytes[r])

    # ------------------------------------------------------------------
    # Step 2 – exchange sizes with a lean Alltoall so every rank learns
    #           how many bytes it will receive from every other rank.
    # ------------------------------------------------------------------
    recv_sizes = np.empty(size, dtype=np.int64)
    comm.Alltoall(send_sizes, recv_sizes)

    # ------------------------------------------------------------------
    # Step 3 – identify the sparse send/receive neighbourhood.
    # ------------------------------------------------------------------
    destinations = [r for r in range(size) if send_sizes[r] > 0]
    sources = [r for r in range(size) if recv_sizes[r] > 0]

    # ------------------------------------------------------------------
    # Step 4 – build a distributed-graph communicator so that only the
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
    # Step 5 – pack send buffer (bytes for each destination in order).
    # ------------------------------------------------------------------
    send_buf = bytearray().join(send_bytes[r] for r in destinations if send_bytes[r] is not None)
    s_counts = np.array([int(send_sizes[r]) for r in destinations], dtype=np.int64)
    s_displs = np.concatenate(([0], np.cumsum(s_counts[:-1]))) if len(s_counts) > 0 else np.array([], dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 6 – allocate receive buffer.
    # ------------------------------------------------------------------
    r_counts = np.array([int(recv_sizes[r]) for r in sources], dtype=np.int64)
    r_displs = np.concatenate(([0], np.cumsum(r_counts[:-1]))) if len(r_counts) > 0 else np.array([], dtype=np.int64)
    recv_buf = bytearray(int(np.sum(r_counts)))

    # ------------------------------------------------------------------
    # Step 7 – neighbourhood all-to-all-v over the sparse graph.
    # ------------------------------------------------------------------
    graph_comm.Neighbor_alltoallv(
        [send_buf, s_counts, s_displs, MPI.BYTE],
        [recv_buf, r_counts, r_displs, MPI.BYTE],
    )

    # ------------------------------------------------------------------
    # Step 8 – free the neighbourhood communicator.
    # ------------------------------------------------------------------
    graph_comm.Free()

    # ------------------------------------------------------------------
    # Step 9 – deserialise and build result list.
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
    comm: "MPI.Comm",
) -> "list[list[SlaterDeterminant]]":
    """
    Partition and distribute SlaterDeterminants across MPI ranks.
    """
    if comm is None or comm.size <= 1:
        return [dets]

    from impurityModel.ed.ManyBodyUtils import pack_determinants_cy, unpack_determinants_cy

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


def graph_alltoall_psis(
    psis: "list[ManyBodyState]",
    n_bytes: int,
    comm: "MPI.Comm",
) -> "list[ManyBodyState]":
    """
    Efficiently redistribute many-body state amplitudes across MPI ranks.
    """
    if comm is None or comm.size <= 1:
        return [psi.copy() for psi in psis]

    from impurityModel.ed.ManyBodyUtils import ManyBodyState, pack_psis_fused_cy, unpack_psis_fused_cy

    size = comm.size
    chunks_per_state = (n_bytes + 7) // 8
    # One interleaved entry = state chunks + complex amp + int32 psi index.
    bytes_per_entry = chunks_per_state * 8 + 2 * 8 + 4

    # 1. Cython packing into a single rank-ordered byte buffer (counts are entries/rank).
    send_counts, send_buf = pack_psis_fused_cy(psis, size, chunks_per_state)

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

    # 5. One fused exchange (BYTE) instead of three -- 3x fewer latency-bound rounds, the
    #    dominant cost of the dense small-message all-to-all at 100s-1000s of ranks.
    graph_comm.Neighbor_alltoallv(
        [send_buf, s_counts_nb * bytes_per_entry, s_displs_nb * bytes_per_entry, MPI.BYTE],
        [recv_buf, r_counts_nb * bytes_per_entry, r_displs_nb * bytes_per_entry, MPI.BYTE],
    )

    # 6. graph_comm is cached for reuse (not freed here); see _cached_dist_graph.

    # 7. Unpack into result
    res = [ManyBodyState() for _ in psis]
    if total_recv > 0:
        unpack_psis_fused_cy(res, size, recv_counts, recv_buf, chunks_per_state)

    return res


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
