# ===========================================================================
# ManyBodyState-path block primitives + reorthogonalization
# ===========================================================================
# The is_array-dispatching block primitives (block_inner/apply/add_scaled/combine/
# orthogonalize/normalize) that let the array Lanczos kernel run on ManyBodyState bases, and
# the reorthogonalization machinery (selective_orthogonalize, apply_reort). Reort-estimator
# honesty: resets use measured overlaps, never an EPS post-act reset (which blinds it).

import scipy.sparse as sps
from impurityModel.ed.ManyBodyUtils import (
    inner_multi,
    add_scaled_multi,
    reorth_cgs2_dense,
    SparseKrylovDense,
    ManyBodyBlockState,
)

cpdef bint is_array(object V):
    if isinstance(V, (np.ndarray, sps.spmatrix, sps.sparray)):
        return True
    if isinstance(V, list) and len(V) > 0 and isinstance(V[0], np.ndarray):
        return True
    return False

cpdef object block_inner(object V, object W, bint mpi=False, object comm=None):
    if is_array(V):
        if isinstance(V, list):
            V = np.column_stack(V)
        if isinstance(W, list):
            W = np.column_stack(W)
        res = np.ascontiguousarray(np.conj(V.T) @ W)
        if mpi and comm is not None:
            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res
    else:
        res = inner_multi(V, W)
        if mpi and comm is not None:
            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res

cpdef object block_apply(object H, object V, object basis=None, bint mpi=False, double slaterWeightMin=0.0):
    if is_array(V) or getattr(H, "is_array_operator", False) or isinstance(H, np.ndarray) or isinstance(H, sps.spmatrix):
        if isinstance(V, list) and isinstance(V[0], np.ndarray):
            V_arr = np.column_stack(V)
            res = H @ V_arr
        else:
            res = H @ V

        if mpi and basis is not None and getattr(basis, 'comm', None) is not None:
            comm = basis.comm
            res_global = np.ascontiguousarray(res)
            comm.Allreduce(MPI.IN_PLACE, res_global, op=MPI.SUM)
            rank = comm.rank
            counts = np.empty(comm.size, dtype=int)
            local_N = V_arr.shape[0] if isinstance(V, list) else V.shape[0]
            comm.Allgather(np.array([local_N], dtype=int), counts)
            offsets = np.array([np.sum(counts[:r]) for r in range(comm.size)], dtype=int)
            return res_global[offsets[rank] : offsets[rank] + local_N, :]
        return res
    else:
        wp = H.apply_multi(V, cutoff=slaterWeightMin)
        if mpi and basis is not None and basis.comm is not None:
            wp = basis.redistribute_psis(wp)
        return wp

cpdef object block_add_scaled(object V, object W, object alpha, double slaterWeightMin=0.0):
    if is_array(V):
        V += W @ alpha
    else:
        add_scaled_multi(V, W, alpha)
        if slaterWeightMin > 0:
            for st in V:
                st.prune(slaterWeightMin)
    return V

cpdef object block_combine(object Q, object Y, double slaterWeightMin=0.0):
    if is_array(Q):
        if isinstance(Q, list):
            Q = np.column_stack(Q)
        return block_combine_array(Q, Y)
    elif isinstance(Q, SparseKrylovDense):
        # Columnar Krylov store: one zgemm over the dense buffer + scatter of only the
        # output columns — no materialization of the inputs.
        return Q.combine(Y, slaterWeightMin=slaterWeightMin)
    else:
        from impurityModel.ed.BlockLanczos import block_combine_sparse
        return block_combine_sparse(Q, Y, slaterWeightMin)

cpdef tuple block_orthogonalize(object wp, object Q, object overlaps=None, bint mpi=False, object comm=None):
    if is_array(wp):
        if isinstance(Q, list):
            Q = np.column_stack(Q)
        return block_orthogonalize_array(wp, Q, overlaps, comm if mpi else None)
    else:
        from impurityModel.ed.BlockLanczos import block_orthogonalize_sparse
        return block_orthogonalize_sparse(wp, Q, overlaps, comm if mpi else None)

cpdef tuple block_normalize(object wp, bint mpi=False, object comm=None, double slaterWeightMin=0.0):
    if is_array(wp):
        return block_normalize_array(wp, comm if mpi else None)
    else:
        from impurityModel.ed.BlockLanczos import block_normalize_sparse
        return block_normalize_sparse(wp, mpi, comm, slaterWeightMin)


def selective_orthogonalize(q_next, Q_basis, alphas, betas, W, block_widths,
                            it, n_curr, double beta_norm, double reort_eps,
                            int period, bint mpi, object comm):
    r"""EA16 §2.6.2 selective orthogonalization of the new Krylov block.

    Shared by both kernels: the eigensolve, the convergence/overlap gate and the
    rank-0/bcast decision are all representation-independent (pure numpy on the
    replicated ``alphas``/``betas``/``W``), and the projection itself dispatches on
    ``is_array`` through ``block_combine`` / ``block_inner`` / ``block_add_scaled``, so
    the same code drives the dense-array (``block_lanczos_array_cy``) and
    ``ManyBodyState`` (``block_lanczos_step_cy``) kernels.

    Gated to the ``period`` cadence (the per-step PARTIAL bad-block reorth keeps
    orthogonality between locks, so the O(m^3) Ritz check can run less often). For each
    Ritz pair that has **converged** (``err_bnd = beta_norm·|s_k[-1]| < reort_eps``) AND
    whose running W-estimate shows ``q_next`` has actually lost orthogonality to it
    (``max|w_ritz_k| > reort_eps``), the Ritz vector is projected out of ``q_next``
    (twice). The decision data is replicated, so it is computed on rank 0 and the index
    list broadcast to keep every rank's projection collectives in lock-step.

    Parameters
    ----------
    q_next : ndarray or list of ManyBodyState
        The freshly QR'd block to be cleaned; returned (possibly modified).
    Q_basis : ndarray or list of ManyBodyState
        The accumulated Krylov basis (all ``it+1`` blocks), indexed by ``s_k``.
    alphas, betas : ndarray
        Block-tridiagonal coefficients built so far (views ``[: it+1]`` are used).
    W : ndarray
        Paige-Simon estimator state (``W[-1]`` is the current-level overlap table).
    block_widths : list of int
        True width of every *completed* block (length ``it``); the current block
        width ``n_curr`` is appended internally.
    it : int
        Absolute iteration index.
    n_curr : int
        Width of the current block.
    beta_norm : float
        ``||beta_i||_2`` (the Ritz residual scale).
    reort_eps : float
        Trigger tolerance (``REORT_TOL``).
    period : int
        Cadence for the Ritz-convergence check.
    mpi : bool
    comm : mpi4py communicator or None

    Returns
    -------
    q_next : ndarray or list of ManyBodyState
        ``q_next`` with the flagged converged Ritz directions removed.
    """
    if not (it > 0 and it % period == 0):
        return q_next
    # Banded eigensolve of the pure block-tridiagonal T (no dense matrix).
    eigvals_T, conv_evec = eigh_block_tridiagonal(
        alphas[: it + 1], betas[: it + 1], block_widths=block_widths + [n_curr]
    )
    ritz_to_project = []
    if not mpi or comm is None or comm.rank == 0:
        widths_list = list(block_widths) + [n_curr]
        offsets_w = [0]
        off = 0
        for w_val in widths_list:
            off += int(w_val)
            offsets_w.append(off)
        for k_idx in range(len(eigvals_T)):
            err_bnd = beta_norm * np.abs(conv_evec[-1, k_idx])
            if err_bnd < reort_eps:
                w_ritz_k = np.zeros(n_curr, dtype=complex)
                for j in range(it + 1):
                    s_k_j = conv_evec[offsets_w[j] : offsets_w[j + 1], k_idx]
                    w_j = widths_list[j]
                    w_ritz_k += np.conj(s_k_j) @ np.conj(W[-1, j, :n_curr, :w_j].T)
                if np.max(np.abs(w_ritz_k)) > reort_eps:
                    ritz_to_project.append(k_idx)
    if mpi and comm is not None:
        ritz_to_project = comm.bcast(ritz_to_project, root=0)

    # Batched projection: the flagged Ritz vectors are mutually orthonormal (eigenvectors of
    # the Hermitian banded T over an orthonormal Q), so projecting them out as one block via
    # CGS2 is equivalent to the old one-at-a-time loop up to rounding — but it costs one
    # store sweep per RITZ_BATCH instead of one per Ritz vector. The batch keeps the
    # materialized Ritz block bounded at (n_rows x RITZ_BATCH).
    for batch_start in range(0, len(ritz_to_project), RITZ_BATCH):
        idx = ritz_to_project[batch_start : batch_start + RITZ_BATCH]
        ritz_blk = block_combine(Q_basis, np.ascontiguousarray(conv_evec[:, idx]))
        for _ in range(2):
            overlap = block_inner(ritz_blk, q_next, mpi, comm)
            q_next = block_add_scaled(q_next, ritz_blk, -overlap)
    return q_next


def block_lanczos_array(*args, **kwargs):
    return block_lanczos_array_cy(*args, **kwargs)


cpdef tuple apply_reort(object wp, object Q_list, object W, object reort, bint mpi, object comm, list block_widths, object krylov=None, bint force=False):
    """Reorthogonalize ``wp`` per the reort mode. Returns ``(wp, W, acted)``; ``acted`` is True
    iff a projection was actually applied (always for FULL/PERIODIC; for PARTIAL/SELECTIVE only
    when a bad block exceeded the trigger), so the caller can skip the follow-up renormalize
    when nothing changed.

    ``force=True`` (PARTIAL/SELECTIVE) skips the ``REORT_TOL`` trigger gate and projects
    against every block whose estimate exceeds ``BAD_BLOCK_TOL``. Drivers pass it on the
    step immediately after an acted step (the classic two-consecutive-steps rule, Simon
    1984 / PROPACK): the projection cleans only ``wp`` = q_{i+1}, so q_i's remaining
    overlap re-contaminates q_{i+2} through the three-term recurrence one step later —
    deterministically re-projecting there closes that channel."""
    from impurityModel.ed.BlockLanczosArray import Reort, REORT_TOL, BAD_BLOCK_TOL, EPS
    cdef list bad_block_idx = []
    cdef int j, col_start, col_end, w_j
    cdef list bad_cols
    cdef object Q_bad
    cdef int active_k
    cdef bint acted = False
    # A shared-support block (Phase 2.4) goes through the same list-based projection
    # machinery via a boundary conversion: on every call for FULL/PERIODIC, but for
    # PARTIAL/SELECTIVE only when a bad block actually triggers — the common no-op
    # case never materializes the block.
    cdef bint was_block = isinstance(wp, ManyBodyBlockState)

    if is_array(wp):
        active_k = wp.shape[1]
    elif was_block:
        active_k = wp.width
    else:
        active_k = len(wp)

    if reort == Reort.FULL or reort == Reort.PERIODIC:
        if is_array(wp):
            for _ in range(2):
                wp, _ = block_orthogonalize(wp, Q_list, mpi=mpi, comm=comm)
        else:
            if was_block:
                wp = wp.to_states()
            if krylov is not None:
                # Sparse path with a maintained dense Krylov basis: slice all columns, no gather.
                wp, _ = krylov.reort(wp, None, 2, comm if mpi else None)
            else:
                # Sparse path fallback: 2-pass CGS2 in dense BLAS (materialize Q from flat_maps).
                wp, _ = reorth_cgs2_dense(wp, Q_list, 2, comm if mpi else None)
            if was_block:
                wp = ManyBodyBlockState.from_states(wp)
        acted = True

    elif reort in (Reort.PARTIAL, Reort.SELECTIVE):
        if W is not None:
            n_blks = W.shape[1] - 1  # W[-1, :n_blks]
            if not mpi or comm is None or comm.rank == 0:
                if force or np.max(np.abs(W[-1, :n_blks])) > REORT_TOL:
                    bad_block_idx = [j for j in range(n_blks) if np.max(np.abs(W[-1, j])) > BAD_BLOCK_TOL]
            if mpi and comm is not None:
                bad_block_idx = comm.bcast(bad_block_idx, root=0)

            if _REORT_PROF_ON:
                _REORT_PROF["calls"] = _REORT_PROF.get("calls", 0.0) + 1.0
                _REORT_PROF["n_blocks_total"] = _REORT_PROF.get("n_blocks_total", 0.0) + float(n_blks)
                if bad_block_idx:
                    _REORT_PROF["acted"] = _REORT_PROF.get("acted", 0.0) + 1.0
                    _REORT_PROF["bad_blocks"] = _REORT_PROF.get("bad_blocks", 0.0) + float(len(bad_block_idx))
            if bad_block_idx:
                acted = True
                bad_cols = []
                for j in bad_block_idx:
                    col_start = sum(block_widths[:j])
                    col_end = col_start + block_widths[j]
                    bad_cols.extend(range(col_start, col_end))
                if _REORT_PROF_ON:
                    _REORT_PROF["bad_cols"] = _REORT_PROF.get("bad_cols", 0.0) + float(len(bad_cols))

                # Every path keeps the FINAL CGS pass's measured (Allreduced) overlap
                # O_last (rows ordered as bad_cols): it upper-bounds the overlap left
                # after the projection and becomes the post-reort W-estimate below.
                O_last = None
                if is_array(Q_list):
                    Q_mat = Q_list if not isinstance(Q_list, list) else Q_list[0]
                    Q_bad = Q_mat[:, bad_cols]
                    for _ in range(2):
                        wp, O_last = block_orthogonalize(wp, Q_bad, mpi=mpi, comm=comm)
                else:
                    if was_block:
                        wp = wp.to_states()
                    if krylov is not None:
                        # Sparse path with a maintained dense Krylov basis: slice the flagged columns.
                        wp, O_last = krylov.reort(wp, bad_cols, 2, comm if mpi else None)
                    else:
                        Q_bad = [Q_list[col] for col in bad_cols]
                        # Sparse path fallback: 2-pass CGS2 in dense BLAS over the flagged bad blocks.
                        wp, O_last = reorth_cgs2_dense(wp, Q_bad, 2, comm if mpi else None)
                    if was_block:
                        wp = ManyBodyBlockState.from_states(wp)

                # HONEST reset (the old ``W = EPS`` was a lie that lost production runs):
                # against a Krylov set whose own mutual orthogonality has degraded to
                # delta, CGS2 leaves a residual ~ delta * |overlap| >> EPS. Writing EPS
                # made the estimator blind right when it mattered (both live W rows get
                # chopped on consecutive acted steps), and the true loss then regrew
                # geometrically from the un-modeled residual until the recurrence
                # diverged (measured: estimate 1e-9 while the true overlap was 1e-2).
                # The final-pass overlap O_last is a measured, conservative bound on
                # that residual — in the healthy regime it is ~EPS-scale, so the
                # trigger cadence there is unchanged.
                row0 = 0
                for j in bad_block_idx:
                    w_j = block_widths[j]
                    if O_last is not None:
                        W[-1, j, :w_j, :active_k] = O_last[row0 : row0 + w_j, :active_k]
                    else:
                        W[-1, j, :w_j, :active_k] = EPS * np.eye(w_j, active_k, dtype=complex)
                    row0 += w_j

    return wp, W, acted
