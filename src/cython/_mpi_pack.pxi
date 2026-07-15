# ===========================================================================
# MPI pack/unpack + dense multi-state helpers
# ===========================================================================
# Buffer packing/unpacking of determinants and psis for the sparse graph-alltoall
# redistribution (MpiUtils.cpp), plus the dense multi-state utilities (inner_multi,
# add_scaled_multi, reorth_cgs2_dense) used by the array reorthogonalization path.

from MpiUtils cimport pack_determinants as c_pack_determinants, unpack_determinants as c_unpack_determinants, pack_psis as c_pack_psis, unpack_psis as c_unpack_psis, pack_psis_fused as c_pack_psis_fused, unpack_psis_fused as c_unpack_psis_fused, pack_block_count as c_pack_block_count, pack_block_fill as c_pack_block_fill, unpack_block_fused as c_unpack_block_fused
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


def pack_psis_cy(list psis, int comm_size):
    cdef vector[const ManyBodyState_cpp*] c_psis
    cdef ManyBodyState psi
    for psi in psis:
        c_psis.push_back(&(psi.v))

    cdef vector[int64_t] send_counts
    cdef vector[uint64_t] state_buf
    cdef vector[double] amp_buf_reim
    cdef vector[int32_t] psi_buf

    with nogil:
        c_pack_psis(c_psis, comm_size, send_counts, state_buf, amp_buf_reim, psi_buf)

    cdef size_t total_states = state_buf.size()
    cdef size_t n_entries = psi_buf.size()

    send_counts_np = np.zeros(comm_size, dtype=np.int64)
    state_buf_np = np.zeros(total_states, dtype=np.uint64)
    amp_buf_np = np.zeros(n_entries, dtype=np.complex128)
    psi_buf_np = np.zeros(n_entries, dtype=np.int32)

    cdef int64_t[:] send_counts_view = send_counts_np
    cdef uint64_t[:] state_buf_view = state_buf_np
    cdef complex[:] amp_buf_view = amp_buf_np
    cdef int32_t[:] psi_buf_view = psi_buf_np

    if comm_size > 0:
        memcpy(&send_counts_view[0], <void*>send_counts.data(), comm_size * sizeof(int64_t))

    if total_states > 0:
        memcpy(&state_buf_view[0], <void*>state_buf.data(), total_states * sizeof(uint64_t))

    if n_entries > 0:
        memcpy(&amp_buf_view[0], <void*>amp_buf_reim.data(), n_entries * 2 * sizeof(double))
        memcpy(&psi_buf_view[0], <void*>psi_buf.data(), n_entries * sizeof(int32_t))

    return send_counts_np, state_buf_np, amp_buf_np, psi_buf_np


def unpack_psis_cy(list psis, int comm_size, int64_t[:] recv_counts, uint64_t[:] state_buf, complex[:] amp_buf, int32_t[:] psi_buf, size_t chunks_per_state):
    """
    Unpacks vectors natively into ManyBodyState objects.
    """
    cdef vector[ManyBodyState_cpp*] c_psis
    cdef ManyBodyState psi
    for psi in psis:
        c_psis.push_back(&(psi.v))

    cdef vector[int64_t] c_recv_counts
    cdef vector[uint64_t] c_state_buf
    cdef vector[double] c_amp_buf_reim
    cdef vector[int32_t] c_psi_buf

    if comm_size > 0:
        c_recv_counts.assign(<int64_t*> &recv_counts[0], <int64_t*> &recv_counts[0] + comm_size)

    if state_buf.shape[0] > 0:
        c_state_buf.assign(<uint64_t*> &state_buf[0], <uint64_t*> &state_buf[0] + state_buf.shape[0])

    if amp_buf.shape[0] > 0:
        c_amp_buf_reim.assign(<double*> &amp_buf[0], <double*> &amp_buf[0] + amp_buf.shape[0] * 2)

    if psi_buf.shape[0] > 0:
        c_psi_buf.assign(<int32_t*> &psi_buf[0], <int32_t*> &psi_buf[0] + psi_buf.shape[0])

    with nogil:
        c_unpack_psis(c_psis, comm_size, c_recv_counts, c_state_buf, c_amp_buf_reim, c_psi_buf, chunks_per_state)


def pack_psis_fused_cy(list psis, int comm_size, size_t chunks_per_state):
    """Pack psis into a single interleaved byte buffer (state||amp||psi_idx per entry),
    rank-ordered, for a one-shot Neighbor_alltoallv(MPI.BYTE) redistribute."""
    cdef vector[const ManyBodyState_cpp*] c_psis
    cdef ManyBodyState psi
    for psi in psis:
        c_psis.push_back(&(psi.v))

    cdef vector[int64_t] send_counts
    cdef vector[char] send_buf

    with nogil:
        c_pack_psis_fused(c_psis, comm_size, chunks_per_state, send_counts, send_buf)

    cdef size_t nbytes = send_buf.size()
    send_counts_np = np.zeros(comm_size, dtype=np.int64)
    send_buf_np = np.zeros(nbytes, dtype=np.uint8)

    cdef int64_t[:] send_counts_view = send_counts_np
    cdef uint8_t[:] send_buf_view = send_buf_np
    if comm_size > 0:
        memcpy(&send_counts_view[0], <void*>send_counts.data(), comm_size * sizeof(int64_t))
    if nbytes > 0:
        memcpy(&send_buf_view[0], <void*>send_buf.data(), nbytes)

    return send_counts_np, send_buf_np


def unpack_psis_fused_cy(list psis, int comm_size, int64_t[:] recv_counts, uint8_t[:] recv_buf, size_t chunks_per_state):
    """Unpack the interleaved byte buffer produced by pack_psis_fused_cy."""
    cdef vector[ManyBodyState_cpp*] c_psis
    cdef ManyBodyState psi
    for psi in psis:
        c_psis.push_back(&(psi.v))

    cdef vector[int64_t] c_recv_counts
    cdef vector[char] c_recv_buf

    if comm_size > 0:
        c_recv_counts.assign(<int64_t*> &recv_counts[0], <int64_t*> &recv_counts[0] + comm_size)
    if recv_buf.shape[0] > 0:
        c_recv_buf.assign(<char*> &recv_buf[0], <char*> &recv_buf[0] + recv_buf.shape[0])

    with nogil:
        c_unpack_psis_fused(c_psis, comm_size, c_recv_counts, c_recv_buf, chunks_per_state)


def extract_new_states(list states, object existing_dict):
    """
    Extracts all unique SlaterDeterminant keys from a list of ManyBodyStates
    that are not already present in existing_dict.
    """
    cdef set new_states = set()
    cdef ManyBodyState s
    cdef ManyBodyState_cpp.const_iterator it
    cdef SlaterDeterminant key
    for obj in states:
        s = <ManyBodyState?>obj
        it = s.v.cbegin()
        while it != s.v.cend():
            key = SlaterDeterminant(tuple(chunk for chunk in dereference(it).first))
            if key not in existing_dict and key not in new_states:
                new_states.add(key)
            preincrement(it)
    return new_states


def inner_multi(states_a, states_b):
    """
    Compute inner products between two sequences of ManyBodyStates.
    Returns a complex numpy array of shape (len(states_a), len(states_b)).

    Non-list sequences (e.g. a SparseKrylovDense column store) are materialized into a
    list first: the loop below collects raw pointers into the states, so the Python
    objects must stay alive for the whole call — a bare iterator's temporaries would
    be collected mid-loop and leave the pointers dangling.
    """
    if not isinstance(states_a, list):
        states_a = list(states_a)
    if not isinstance(states_b, list):
        states_b = list(states_b)
    cdef int na = len(states_a)
    cdef int nb = len(states_b)
    cdef res = np.zeros((na, nb), dtype=complex)
    cdef complex[:, :] res_view = res
    cdef ManyBodyState_cpp.mapped_type val
    cdef int i, j

    cdef vector[ManyBodyState_cpp*] a_ptrs
    cdef vector[ManyBodyState_cpp*] b_ptrs
    cdef ManyBodyState state

    a_ptrs.reserve(na)
    for obj in states_a:
        state = <ManyBodyState?>obj
        a_ptrs.push_back(&state.v)

    b_ptrs.reserve(nb)
    for obj in states_b:
        state = <ManyBodyState?>obj
        b_ptrs.push_back(&state.v)

    with nogil:
        for i in range(na):
            for j in range(nb):
                val = inner_cpp(dereference(a_ptrs[i]), dereference(b_ptrs[j]))
                res_view[i, j].real = val.real()
                res_view[i, j].imag = val.imag()

    return res


def add_scaled_multi(list states_target, states_source, complex[:, :] coeffs):
    """
    Add a scaled sum of states_source to each state in states_target::

        for each j in range(len(states_target)):
            for i in range(len(states_source)):
                states_target[j] += coeffs[i, j] * states_source[i]

    ``states_target`` must be a real list (the states are mutated in place; a store's
    materialized temporaries would silently discard the update). ``states_source`` may
    be any sequence; non-lists are materialized first to keep the collected raw
    pointers alive for the whole call.
    """
    if not isinstance(states_source, list):
        states_source = list(states_source)
    cdef int n_target = len(states_target)
    cdef int n_source = len(states_source)

    cdef vector[ManyBodyState_cpp*] t_ptrs
    cdef vector[ManyBodyState_cpp*] s_ptrs
    cdef ManyBodyState state

    t_ptrs.reserve(n_target)
    for obj in states_target:
        state = <ManyBodyState?>obj
        t_ptrs.push_back(&state.v)

    s_ptrs.reserve(n_source)
    for obj in states_source:
        state = <ManyBodyState?>obj
        s_ptrs.push_back(&state.v)

    cdef int i, j
    cdef double complex coeff
    with nogil:
        for j in range(n_target):
            for i in range(n_source):
                coeff = coeffs[i, j]
                if coeff.real != 0 or coeff.imag != 0:
                    dereference(t_ptrs[j]).add_scaled(dereference(s_ptrs[i]), ManyBodyState_cpp.mapped_type(coeff.real, coeff.imag))


def support_stats(list states):
    """Union-support statistics of a list of ``ManyBodyState`` (rank-local, no MPI).

    Returns ``(union_size, total_nnz)``: the number of distinct determinants in the
    union support of ``states`` and the total number of stored coefficients. The dense
    fill ratio of the block is ``total_nnz / (union_size * len(states))`` — the fraction
    of a dense ``(union_size x len(states))`` coefficient matrix that is actually
    nonzero, which is the break-even measure for a columnar (dense-over-support)
    Krylov storage vs the per-state ``flat_map`` representation.
    """
    cdef vector[SlaterDeterminant_cpp[uint64_t]] support
    cdef ManyBodyState ms
    cdef ManyBodyState_cpp.iterator it
    cdef size_t total_nnz = 0
    for obj in states:
        ms = <ManyBodyState?>obj
        total_nnz += ms.v.size()
        it = ms.v.begin()
        while it != ms.v.end():
            support.push_back(dereference(it).first)
            preincrement(it)
    sort(support.begin(), support.end())
    support.erase(unique(support.begin(), support.end()), support.end())
    return (<size_t>support.size(), total_nnz)


def reorth_cgs2_dense(list wp, list Q, int n_passes, object comm):
    r"""Dense (BLAS) block reorthogonalization of ``wp`` against ``Q`` for the sparse
    (``ManyBodyState``) path.

    Performs ``n_passes`` of classical block Gram-Schmidt

    .. math:: O = Q^\dagger\, wp \quad(\text{Allreduced over ranks});\qquad wp \leftarrow wp - Q\,O,

    but materializes ``wp`` and ``Q`` onto their merged determinant support and runs the two
    projections as ``zgemm`` instead of per-pair ``flat_map`` inner products / merges.
    Mathematically equivalent (to floating point) to repeating ``block_orthogonalize_sparse``
    ``n_passes`` times -- the W-estimator / bad-block selection is unchanged; only the projection
    is accelerated. Returns ``(out, O_last)``: the new list of ``wp`` ManyBodyStates and the
    final pass's measured (Allreduced) ``(nq x p)`` overlap (``None`` when nothing was done) —
    the caller's honest post-reorthogonalization W-estimate.
    """
    cdef int p = len(wp)
    cdef int nq = len(Q)
    if p == 0 or nq == 0 or n_passes <= 0:
        return wp, None

    cdef vector[ManyBodyState_cpp*] wp_ptrs
    cdef vector[ManyBodyState_cpp*] q_ptrs
    cdef ManyBodyState ms
    cdef int ci
    for obj in wp:
        ms = <ManyBodyState?>obj
        wp_ptrs.push_back(&ms.v)
    for obj in Q:
        ms = <ManyBodyState?>obj
        q_ptrs.push_back(&ms.v)

    # --- merged local support: sorted, unique determinant keys over wp ∪ Q ---
    cdef vector[SlaterDeterminant_cpp[uint64_t]] support
    cdef ManyBodyState_cpp.iterator it
    for ci in range(p):
        it = wp_ptrs[ci].begin()
        while it != wp_ptrs[ci].end():
            support.push_back(dereference(it).first)
            preincrement(it)
    for ci in range(nq):
        it = q_ptrs[ci].begin()
        while it != q_ptrs[ci].end():
            support.push_back(dereference(it).first)
            preincrement(it)
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
    cdef Py_ssize_t row
    cdef ManyBodyState_cpp.mapped_type cval
    for ci in range(p):
        it = wp_ptrs[ci].begin()
        while it != wp_ptrs[ci].end():
            row = lower_bound(support.begin(), support.end(), dereference(it).first) - support.begin()
            cval = dereference(it).second
            Wv[row, ci].real = cval.real()
            Wv[row, ci].imag = cval.imag()
            preincrement(it)
    for ci in range(nq):
        it = q_ptrs[ci].begin()
        while it != q_ptrs[ci].end():
            row = lower_bound(support.begin(), support.end(), dereference(it).first) - support.begin()
            cval = dereference(it).second
            Qv[row, ci].real = cval.real()
            Qv[row, ci].imag = cval.imag()
            preincrement(it)

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

    # --- scatter back to ManyBodyStates (keep the nonzero rows of each column) ---
    cdef complex[:, :] Wout = Wd
    cdef list out = []
    cdef vector[ManyBodyState_cpp.key_type] keys
    cdef vector[ManyBodyState_cpp.mapped_type] vals
    cdef ManyBodyState new_ms
    cdef double complex z
    for ci in range(p):
        keys.clear()
        vals.clear()
        for row in range(ns):
            z = Wout[row, ci]
            if z.real != 0 or z.imag != 0:
                keys.push_back(support[row])
                vals.push_back(ManyBodyState_cpp.mapped_type(z.real, z.imag))
        new_ms = ManyBodyState()
        new_ms.v = ManyBodyState_cpp(keys, vals)
        out.append(new_ms)
    return out, O
