# ===========================================================================
# MPI pack/unpack + dense multi-state helpers
# ===========================================================================
# Buffer packing/unpacking of determinants and psis for the sparse graph-alltoall
# redistribution (MpiUtils.cpp), plus the dense multi-state utilities (inner_multi,
# add_scaled_multi, reorth_cgs2_dense) used by the array reorthogonalization path.

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


cdef ManyBodyState _as_width1_block(list states):
    """A list of width-1 ``ManyBodyState`` blocks (each its own, generally different,
    support), merged into ONE block via ``from_states`` -- the union-support
    representation ``block_inner_cy`` needs to compute a whole Gram matrix in one
    merge-join instead of one per pair. ``from_states`` itself validates that every
    element is width 1."""
    return ManyBodyState.from_states(states)


def inner_multi(states_a, states_b):
    """
    Compute inner products between two sequences of width-1 ``ManyBodyState`` blocks.
    Returns a complex numpy array of shape (len(states_a), len(states_b)).

    Non-list sequences (e.g. a SparseKrylovDense column store) are materialized into a
    list first. Each side's states (possibly independently-supported) are merged into
    one block via ``from_states`` and routed through ``block_inner_cy``'s single
    merge-join.
    """
    if not isinstance(states_a, list):
        states_a = list(states_a)
    if not isinstance(states_b, list):
        states_b = list(states_b)
    return block_inner_cy(_as_width1_block(states_a), _as_width1_block(states_b))


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


def reorth_cgs2_dense(list wp, list Q, int n_passes, object comm):
    r"""Dense (BLAS) block reorthogonalization of ``wp`` against ``Q``: each is a list
    of width-1 ``ManyBodyState`` blocks (the sparse, non-columnar-store path).

    Performs ``n_passes`` of classical block Gram-Schmidt

    .. math:: O = Q^\dagger\, wp \quad(\text{Allreduced over ranks});\qquad wp \leftarrow wp - Q\,O,

    but materializes ``wp`` and ``Q`` onto their merged determinant support and runs the two
    projections as ``zgemm`` instead of per-pair inner products / merges.
    Mathematically equivalent (to floating point) to repeating classical block Gram-Schmidt
    ``n_passes`` times -- the W-estimator / bad-block selection is unchanged; only the projection
    is accelerated. Returns ``(out, O_last)``: the new list of ``wp`` width-1 blocks and the
    final pass's measured (Allreduced) ``(nq x p)`` overlap (``None`` when nothing was done) —
    the caller's honest post-reorthogonalization W-estimate.
    """
    cdef int p = len(wp)
    cdef int nq = len(Q)
    if p == 0 or nq == 0 or n_passes <= 0:
        return wp, None

    cdef vector[ManyBodyBlockState_cpp*] wp_ptrs
    cdef vector[ManyBodyBlockState_cpp*] q_ptrs
    cdef ManyBodyState ms
    cdef int ci
    for obj in wp:
        ms = <ManyBodyState?>obj
        wp_ptrs.push_back(&ms.b)
    for obj in Q:
        ms = <ManyBodyState?>obj
        q_ptrs.push_back(&ms.b)

    # --- merged local support: sorted, unique determinant keys over wp ∪ Q ---
    cdef vector[SlaterDeterminant_cpp[uint64_t]] support
    cdef Py_ssize_t row, nr, pos
    for ci in range(p):
        nr = <Py_ssize_t>wp_ptrs[ci].rows()
        for row in range(nr):
            support.push_back(wp_ptrs[ci].key(row))
    for ci in range(nq):
        nr = <Py_ssize_t>q_ptrs[ci].rows()
        for row in range(nr):
            support.push_back(q_ptrs[ci].key(row))
    # NOTE: an empty local support must NOT return early under MPI -- p, nq and
    # n_passes are rank-identical (bad_cols is bcast), so every rank must join the
    # n_passes Allreduce(O) collectives below; a rank-local early return mispairs
    # the other ranks' Allreduce with this rank's next collective (wrong numerics,
    # then deadlock -- seen at 4 ranks where a rank owns no determinants). With
    # ns == 0 the GEMMs are zero-row no-ops and O contributes zeros, as intended.
    if support.size() == 0 and comm is None:
        return wp, None
    sort(support.begin(), support.end())
    support.erase(unique(support.begin(), support.end()), support.end())
    cdef Py_ssize_t ns = <Py_ssize_t>support.size()

    if _MBU_PROF_ON:
        # Transient dense footprint of this call: the (ns x p) W block + (ns x nq) Q block
        # materialized below (16 B/coeff). Track the peak across calls plus totals.
        _bytes = 16 * ns * (p + nq)
        _MBU_PROF["cgs2_dense_peak_bytes"] = max(_MBU_PROF.get("cgs2_dense_peak_bytes", 0), _bytes)
        _MBU_PROF["cgs2_dense_calls"] = _MBU_PROF.get("cgs2_dense_calls", 0) + 1
        _MBU_PROF["cgs2_dense_total_bytes"] = _MBU_PROF.get("cgs2_dense_total_bytes", 0) + _bytes

    # --- materialize dense (|S| x p) and (|S| x nq) over the local support ---
    Wd = np.zeros((ns, p), dtype=complex)
    Qd = np.zeros((ns, nq), dtype=complex)
    cdef complex[:, :] Wv = Wd
    cdef complex[:, :] Qv = Qd
    cdef ManyBodyBlockState_cpp.Value cval
    for ci in range(p):
        nr = <Py_ssize_t>wp_ptrs[ci].rows()
        for row in range(nr):
            pos = lower_bound(support.begin(), support.end(), wp_ptrs[ci].key(row)) - support.begin()
            cval = wp_ptrs[ci].data()[row]
            Wv[pos, ci].real = cval.real()
            Wv[pos, ci].imag = cval.imag()
    for ci in range(nq):
        nr = <Py_ssize_t>q_ptrs[ci].rows()
        for row in range(nr):
            pos = lower_bound(support.begin(), support.end(), q_ptrs[ci].key(row)) - support.begin()
            cval = q_ptrs[ci].data()[row]
            Qv[pos, ci].real = cval.real()
            Qv[pos, ci].imag = cval.imag()

    # --- CGS2 passes via BLAS; the small (nq x p) overlap is Allreduced each pass ---
    Qh = np.conj(Qd.T)
    if comm is not None:
        from mpi4py import MPI
    O = None
    for _ in range(n_passes):
        O = Qh @ Wd
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, O, op=MPI.SUM)
        Wd = Wd - Qd @ O

    # --- scatter back to width-1 ManyBodyStates (keep the nonzero rows of each column) ---
    cdef complex[:, :] Wout = Wd
    cdef list out = []
    cdef vector[SlaterDeterminant_cpp[uint64_t]] keys
    cdef vector[ManyBodyBlockState_cpp.Value] vals
    cdef ManyBodyState new_ms
    cdef double complex z
    for ci in range(p):
        keys.clear()
        vals.clear()
        for row in range(ns):
            z = Wout[row, ci]
            if z.real != 0 or z.imag != 0:
                keys.push_back(support[row])
                vals.push_back(ManyBodyBlockState_cpp.Value(z.real, z.imag))
        new_ms = ManyBodyState()
        new_ms.b = ManyBodyBlockState_cpp(keys, vals, <size_t>1)
        out.append(new_ms)
    return out, O
