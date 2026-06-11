"""
This module contains help functions for MPI communication.
"""

import math
import pickle
import time
from itertools import islice

import numpy as np
from mpi4py import MPI

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


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
    Documentation for is_empty.
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
    Documentation for empty_clone.
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
        sources,       # in-edges  (ranks that will send to me)
        destinations,  # out-edges (ranks I will send to)
        reorder=False,
    )

    # ------------------------------------------------------------------
    # Step 5 – pack send buffer (bytes for each destination in order).
    # ------------------------------------------------------------------
    send_buf = bytearray().join(
        send_bytes[r] for r in destinations if send_bytes[r] is not None
    )
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
        result[r] = pickle.loads(recv_buf[offset: offset + cnt])
        offset += cnt

    for r in range(size):
        if result[r] is None:
            result[r] = empty_clone(send_list[r])

    return result


def graph_alltoall_psis(
    send_list: "list[list[dict]]",
    n_bytes: int,
    comm: "MPI.Comm",
) -> "list[list[dict]]":
    """
    Efficiently redistribute many-body state amplitudes across MPI ranks.

    This is a specialised version of :func:`graph_alltoall` tuned for the
    ``redistribute_psis`` hot-path.  Instead of pickling Python dicts it
    packs everything into two contiguous byte arrays:

    * a flat ``bytearray`` of fixed-length state keys
      (``n_bytes`` bytes each)
    * a flat ``complex128`` numpy array of the corresponding amplitudes

    Both buffers are exchanged with a single
    ``MPI_Neighbor_alltoallv`` call over the sparse distributed-graph
    communicator built from the non-empty communication partners, so
    only the actual sender/receiver pairs pay communication cost.

    Parameters
    ----------
    send_list :
        List of length ``comm.size``.  Element *r* is a list of dicts,
        one dict per "psi".  Each dict maps ``state_bytes`` (``bytes``,
        length ``n_bytes``) to a complex amplitude.
    n_bytes :
        Fixed byte-width of every state key.
    comm :
        MPI communicator.  Returned unchanged when ``size <= 1``.

    Returns
    -------
    list[list[dict]]
        Same shape as *send_list*.  Element *r* is the list of dicts
        received from rank *r*.
    """
    if comm is None or comm.size <= 1:
        return send_list

    size = comm.size
    n_psis = len(send_list[0])

    # ------------------------------------------------------------------
    # Step 1 – count states going to each destination (one counter per
    #           psi is unnecessary; we send all psis together and store
    #           a psi-index alongside each entry).
    # ------------------------------------------------------------------
    # Flatten: one entry = (psi_index, state_bytes, amplitude)
    # For each destination r we track how many (state, amp) pairs to send
    # across ALL psis (psi_index is packed into a separate int array).
    send_counts = np.zeros(size, dtype=np.int64)  # #entries per dest
    for r in range(size):
        for psi_dict in send_list[r]:
            send_counts[r] += len(psi_dict)

    # ------------------------------------------------------------------
    # Step 2 – exchange counts with Alltoall.
    # ------------------------------------------------------------------
    recv_counts = np.empty(size, dtype=np.int64)
    comm.Alltoall(send_counts, recv_counts)

    # ------------------------------------------------------------------
    # Step 3 – build the distributed-graph communicator.
    # ------------------------------------------------------------------
    destinations = [r for r in range(size) if send_counts[r] > 0]
    sources = [r for r in range(size) if recv_counts[r] > 0]

    graph_comm = comm.Create_dist_graph_adjacent(
        sources,
        destinations,
        reorder=False,
    )

    # ------------------------------------------------------------------
    # Step 4 – pack send buffers.
    #
    #   state_buf : flat bytearray, n_bytes per entry
    #   amp_buf   : flat complex128 array, 1 entry per state
    #   psi_buf   : flat int32 array, psi-index per state
    # ------------------------------------------------------------------
    total_send = int(np.sum(send_counts))
    state_send = bytearray(total_send * n_bytes)
    amp_send   = np.empty(total_send, dtype=np.complex128)
    psi_send   = np.empty(total_send, dtype=np.int32)

    pos = 0
    for r in range(size):
        for pi, psi_dict in enumerate(send_list[r]):
            for state_bytes, amp in psi_dict.items():
                state_send[pos * n_bytes: (pos + 1) * n_bytes] = state_bytes
                amp_send[pos] = amp
                psi_send[pos] = pi
                pos += 1

    s_counts_nb = np.array([send_counts[r] for r in destinations], dtype=np.int64)
    s_displs_nb = np.concatenate(([0], np.cumsum(s_counts_nb[:-1]))) if len(s_counts_nb) else np.array([], dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 5 – allocate receive buffers.
    # ------------------------------------------------------------------
    total_recv = int(np.sum(recv_counts))
    state_recv = bytearray(total_recv * n_bytes)
    amp_recv   = np.empty(total_recv, dtype=np.complex128)
    psi_recv   = np.empty(total_recv, dtype=np.int32)

    r_counts_nb = np.array([recv_counts[r] for r in sources], dtype=np.int64)
    r_displs_nb = np.concatenate(([0], np.cumsum(r_counts_nb[:-1]))) if len(r_counts_nb) else np.array([], dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 6 – neighbourhood all-to-all-v (three separate calls for
    #           state bytes, amplitudes, and psi indices).
    # ------------------------------------------------------------------
    # 6a. State bytes
    graph_comm.Neighbor_alltoallv(
        [state_send, s_counts_nb * n_bytes, s_displs_nb * n_bytes, MPI.BYTE],
        [state_recv, r_counts_nb * n_bytes, r_displs_nb * n_bytes, MPI.BYTE],
    )
    # 6b. Amplitudes (complex128 = 2×float64)
    graph_comm.Neighbor_alltoallv(
        [amp_send, s_counts_nb, s_displs_nb, MPI.C_DOUBLE_COMPLEX],
        [amp_recv, r_counts_nb, r_displs_nb, MPI.C_DOUBLE_COMPLEX],
    )
    # 6c. Psi indices (int32)
    graph_comm.Neighbor_alltoallv(
        [psi_send, s_counts_nb, s_displs_nb, MPI.INT],
        [psi_recv, r_counts_nb, r_displs_nb, MPI.INT],
    )

    # ------------------------------------------------------------------
    # Step 7 – free the neighbourhood communicator.
    # ------------------------------------------------------------------
    graph_comm.Free()

    # ------------------------------------------------------------------
    # Step 8 – unpack into result structure.
    # ------------------------------------------------------------------
    result: "list[list[dict]]" = [[{} for _ in range(n_psis)] for _ in range(size)]
    offset = 0
    for r, cnt in zip(sources, r_counts_nb):
        for k in range(int(cnt)):
            idx = offset + k
            sb = bytes(state_recv[idx * n_bytes: (idx + 1) * n_bytes])
            amp = complex(amp_recv[idx])
            pi = int(psi_recv[idx])
            result[r][pi][sb] = result[r][pi].get(sb, 0) + amp
        offset += int(cnt)

    # Fill empty slots with empty-list-of-dicts for ranks we got nothing from
    for r in range(size):
        if not any(result[r]):
            result[r] = [{} for _ in range(n_psis)]

    return result
