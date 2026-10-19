# ===========================================================================
# MPI pack/unpack + dense multi-state helpers
# ===========================================================================
# Buffer packing/unpacking of determinants and psis for the sparse graph-alltoall
# redistribution (MpiUtils.cpp), plus the dense multi-state utilities (inner_multi,
# add_scaled_multi).

from MpiUtils cimport pack_determinants as c_pack_determinants, unpack_determinants as c_unpack_determinants, pack_block_count as c_pack_block_count, pack_block_fill as c_pack_block_fill, unpack_block_fused as c_unpack_block_fused
import numpy as np


def pack_determinants_cy(list dets, int comm_size):
    cdef vector[SlaterDeterminant_cpp[uint64_t]] c_dets
    c_dets.reserve(len(dets))
    cdef SlaterDeterminant det
    for det in dets:
        c_dets.push_back(det.s)

    cdef vector[int64_t] send_counts
    cdef vector[uint64_t] state_buf

    with nogil:
        c_pack_determinants(c_dets, comm_size, send_counts, state_buf)

    cdef size_t total_states = state_buf.size()

    send_counts_np = np.zeros(comm_size, dtype=np.int64)
    state_buf_np = np.zeros(total_states, dtype=np.uint64)

    cdef int64_t[:] send_counts_view = send_counts_np
    cdef uint64_t[:] state_buf_view = state_buf_np

    if comm_size > 0:
        memcpy(&send_counts_view[0], <void*>send_counts.data(), comm_size * sizeof(int64_t))

    if total_states > 0:
        memcpy(&state_buf_view[0], <void*>state_buf.data(), total_states * sizeof(uint64_t))

    return send_counts_np, state_buf_np


def pack_determinants_ordered_cy(list dets, int64_t[:] order, size_t chunks_per_state):
    """Pack ``dets[order[i]]`` into one flat ``uint64`` buffer, in the order given.

    Unlike :func:`pack_determinants_cy` this preserves order and keeps duplicates. That one
    routes and **deduplicates** through an ``unordered_set``, which is what ``add_states``
    wants and what an index lookup must not do: a lookup has to answer for every query it was
    given, in the order it was given them.
    """
    cdef Py_ssize_t n = order.shape[0]
    out = np.empty(<Py_ssize_t>(n * <Py_ssize_t>chunks_per_state), dtype=np.uint64)
    if n == 0 or chunks_per_state == 0:
        return out
    cdef uint64_t[::1] view = out
    cdef SlaterDeterminant sd
    cdef Py_ssize_t i, c, base
    for i in range(n):
        sd = <SlaterDeterminant?>dets[order[i]]
        if sd.s.size() != chunks_per_state:
            raise ValueError(
                f"determinant of {sd.s.size()} chunks packed into a {chunks_per_state}-chunk buffer"
            )
        base = i * <Py_ssize_t>chunks_per_state
        for c in range(<Py_ssize_t>chunks_per_state):
            view[base + c] = sd.s[c]
    return out


def unpack_determinants_cy(int comm_size, int64_t[:] recv_counts, uint64_t[:] state_buf, size_t chunks_per_state):
    cdef vector[int64_t] c_recv_counts
    cdef vector[uint64_t] c_state_buf

    if comm_size > 0:
        c_recv_counts.assign(<int64_t*> &recv_counts[0], <int64_t*> &recv_counts[0] + comm_size)

    if state_buf.shape[0] > 0:
        c_state_buf.assign(<uint64_t*> &state_buf[0], <uint64_t*> &state_buf[0] + state_buf.shape[0])

    cdef vector[vector[SlaterDeterminant_cpp[uint64_t]]] c_res
    with nogil:
        c_res = c_unpack_determinants(comm_size, c_recv_counts, c_state_buf, chunks_per_state)

    cdef list res = []
    cdef SlaterDeterminant py_det
    for i in range(comm_size):
        rank_dets = []
        for j in range(c_res[i].size()):
            py_det = SlaterDeterminant()
            py_det.s = c_res[i][j]
            rank_dets.append(py_det)
        res.append(rank_dets)
    return res


def inner_multi(states_a, states_b):
    """
    Compute inner products between two sequences of width-1 ``ManyBodyState`` blocks.
    Returns a complex numpy array of shape (len(states_a), len(states_b)).

    Non-list sequences (e.g. a SparseKrylovDense column store) are materialized into a
    list first. Each pair is merge-joined on its own supports (``columns_inner_cy``):
    no block over the union of the columns' supports is built, which for independently
    supported columns was (union rows x width) dense amplitudes and most of the time.
    """
    if not isinstance(states_a, list):
        states_a = list(states_a)
    if not isinstance(states_b, list):
        states_b = list(states_b)
    return columns_inner_cy(states_a, states_b)


def support_stats(list states):
    """Union-support statistics of a list of width-1 ``ManyBodyState`` blocks
    (rank-local, no MPI).

    Returns ``(union_size, total_nnz)``: the number of distinct determinants in the
    union support of ``states`` and the total number of stored coefficients. The dense
    fill ratio of the block is ``total_nnz / (union_size * len(states))`` — the fraction
    of a dense ``(union_size x len(states))`` coefficient matrix that is actually
    nonzero, which is the break-even measure for a columnar (dense-over-support)
    Krylov storage vs a list of independent width-1 blocks.
    """
    cdef vector[SlaterDeterminant_cpp[uint64_t]] support
    cdef ManyBodyState ms
    cdef size_t total_nnz = 0
    cdef Py_ssize_t row, nr
    for obj in states:
        ms = <ManyBodyState?>obj
        nr = <Py_ssize_t>ms.b.rows()
        total_nnz += <size_t>nr
        for row in range(nr):
            support.push_back(ms.b.key(row))
    sort(support.begin(), support.end())
    support.erase(unique(support.begin(), support.end()), support.end())
    return (<size_t>support.size(), total_nnz)
