# ===========================================================================
# Core block-Lanczos recurrence
# ===========================================================================
# One three-term block step (lanczos_step) and the full recurrence driver
# (block_lanczos) that serves the whole frequency mesh from a single recurrence. The
# reorthogonalization mode (NONE/FULL) and the EA16
# shrinking-block deflation policy are dispatched from here.

def _keep_global_top(st, max_size, comm):
    """Keep the ``max_size`` largest-amplitude rows of the distributed width-1 state ``st``, in place.

    Exact and layout-independent (:func:`manybody_basis.collective_top_k_mask`: whole near-tie groups,
    then the boundary group in key order), so the kept rows are the same at every rank count. It
    replaced a serial ``truncate`` (ties kept) against a multi-rank bisection to an absolute 1e-8 on
    ``|amp|^2``, which kept different rows by rank count. Collective on ``comm``.
    """
    rows = list(st.items())
    n_total = comm.allreduce(len(rows), op=MPI.SUM) if comm is not None else len(rows)
    if n_total <= max_size:
        return
    keys = [key for key, _amp in rows]
    scores = np.array([abs(complex(amp[0])) ** 2 for _key, amp in rows], dtype=float)
    mask = collective_top_k_mask(scores, keys, int(max_size), comm)
    st.keep_rows(ManyBodyState.from_keys([keys[i] for i in np.flatnonzero(mask)]))


def lanczos_step(
    matvec,
    q_prev,
    q_curr,
    store,
    alphas,
    betas,
    int it,
    reort,
    bint mpi,
    comm,
    double slaterWeightMin=0.0,
    truncation_threshold=0.0,
    locked=None,
    note_cut=None,
    double h_norm_est=0.0,
    double deflate_tol=-1.0,
):
    r"""One step of the block-Lanczos three-term recurrence -- the only one, for every block representation.

    .. math::

        W = A X_i, \qquad \alpha_i = X_i^\dagger W, \qquad
        W \leftarrow W - X_i \alpha_i - X_{i-1} \beta_{i-1}^\dagger, \qquad
        X_{i+1}, \beta_i = \mathrm{TSQR}(W)

    ``q_prev`` / ``q_curr`` are tall-skinny blocks in either representation -- ``ndarray`` or
    shared-support ``ManyBodyState`` -- and the step touches them only through the dispatching
    primitives of ``_block_ops.pxi`` (``block_inner``, ``block_add_scaled``, ``apply_full_reort``,
    ``factor_residual``); ``store`` is the Krylov store (``reort`` projects against it); ``matvec`` is the representation's ``A X`` (the driver builds it: a
    ``DistributedSparseMatrix`` exchange, or ``ManyBodyOperator`` apply + redistribute + the cut on the
    summed rows). Between the recurrence and the TSQR it projects out the ``locked`` columns and, in
    full-reort mode, the stored Krylov basis; after it, a ``ManyBodyState`` block is truncated
    (``slaterWeightMin``, ``truncation_threshold``) and re-factored.

    **Collective** when ``mpi``: the ``p x p`` overlap ``Allreduce``\ s, the TSQR's ``Allgather`` and
    the reorthogonalization's passes -- every rank takes the same branches because every decision is
    read off replicated data (``alpha``, the TSQR's ``R``).

    Writes ``alphas[it]`` and, unless the step breaks down, ``betas[it]``.

    Returns
    -------
    tuple
        ``(q_next, alpha_i, beta_i, beta_norm, active_k, breakdown)``. ``active_k`` is the retained
        width of ``q_next``; ``breakdown`` is True when the residual vanished against the operator
        scale (``active_k == 0``, an invariant subspace) or its factor came out non-finite
        (``active_k == -1``, a corrupted recurrence); ``q_next``, ``beta_i`` are then ``None``.
        ``beta_norm`` is :math:`\|\beta_i\|_2`.
    """
    cdef int p = block_cols(q_curr)
    cdef int n_prev, active_k
    cdef bint did_truncate = False

    # --- 1. W = A X_i ----------------------------------------------------
    wp = matvec(q_curr)

    # --- 2. alpha_i = <X_i | W> ------------------------------------------
    alpha_i = block_inner(q_curr, wp, mpi, comm)
    alphas[it, :p, :p] = alpha_i

    # --- 3. W -= X_i alpha_i + X_{i-1} beta_{i-1}^dagger -----------------
    wp = block_add_scaled(wp, q_curr, -alpha_i)
    # q_prev is always a valid block (a zero placeholder on the very first step of a fresh start),
    # so its width is well-defined at every call; it only contributes from the second step on.
    n_prev = block_cols(q_prev)
    if it > 0:
        wp = block_add_scaled(wp, q_prev, -np.conj(betas[it - 1, :p, :n_prev].T))

    # --- 3b. Locking deflation -------------------------------------------
    # Keep the residual orthogonal to the already-converged ("locked") Ritz vectors *before* forming
    # beta, so the stored beta and q_next are consistent with the deflated vector. The locked vectors
    # are only approximate eigenvectors, so A X_i leaks a small component along them that is
    # otherwise amplified across iterations, reintroducing locked eigenvalues (and their 2*theta
    # harmonics) as spurious Ritz values below the true spectral minimum. Twice, for robustness.
    #
    # Gate on block_cols(locked), never truthiness: ManyBodyState.__len__ is the rank-local row
    # count, so an empty rank diverges and the Allreduce below deadlocks (doc/lanczos_invariants.md,
    # "MPI collective gating").
    if locked is not None and block_cols(locked) > 0:
        if not is_array(wp) and not isinstance(locked, ManyBodyState):
            locked = ManyBodyState.from_states(list(locked))
        for _ in range(2):
            wp = block_add_scaled(wp, locked, -block_inner(locked, wp, mpi, comm))

    # --- 4. Full reorthogonalization against the stored basis ------------
    if reort:
        wp = apply_full_reort(wp, store, mpi, comm)

    # --- 5. X_{i+1}, beta_i = TSQR(W) --------------------------------------
    # The triangular factor is built from the block's own rows (panel Householder + Givens merges +
    # one Allgather), never from <W|W>, so it is backward stable at any conditioning and the rank
    # decision is made on the block's true singular values. Breakdown is measured against the
    # operator scale, not against 1: a residual block is negligible when it is small compared to A.
    # `h_norm_est` is the driver's running estimate (0 on the first step of a cold start, where
    # alpha_i carries the scale).
    q_next, beta_i, active_k, sv_i = factor_residual(
        wp, mpi, comm, max(float(h_norm_est), spectral_norm(alpha_i)), deflate_tol=deflate_tol,
    )
    if active_k < 0:
        # Non-finite factor => the recurrence is *corrupted*, not a genuine invariant subspace.
        # active_k == -1 lets the caller report "diverged" (and warn) instead of treating the
        # truncated result as exact; active_k == 0 is a real rank-deficient residual (the Krylov
        # space is closed) => invariant subspace.
        return None, alpha_i, None, 0.0, -1, True
    if active_k == 0:
        return None, alpha_i, None, 0.0, 0, True
    cdef double beta_norm = float(sv_i[0])

    # --- 5a. Truncation (ManyBodyState blocks) ---------------------------
    # A dense block has no per-determinant structure to prune: these cuts are absent there, not
    # unmirrored.
    if isinstance(q_next, ManyBodyState):
        if slaterWeightMin > 0.0:
            # Whole-row prune: a row survives when ANY column survives (keeps the block's shared
            # support; the per-column prune of independent states would desync it).
            if note_cut is not None:
                # In the matvec's units: the residual is q_next @ beta_i, so a dropped row of q_next
                # carried row beta_i beta_i^H row^H of it. Reaches the proxy with the next step's
                # admission.
                note_cut(q_next.prune_rows(slaterWeightMin, beta_i @ np.conj(beta_i.T)))
            else:
                q_next.prune_rows(slaterWeightMin)
            did_truncate = True
        if truncation_threshold > 0:
            _q_states = q_next.to_states()
            for st in _q_states:
                _keep_global_top(st, truncation_threshold, comm if mpi else None)
            q_next = ManyBodyState.from_states(_q_states)
            did_truncate = True

    # Dropping amplitudes breaks the orthonormality the factorization just established, so the
    # truncated block has to be re-factored and the correction folded into beta_i:
    # W = (q_next @ R2) @ beta_i. TSQR delivers orthonormality directly, so no second pass is
    # needed for well-conditioned blocks.
    if did_truncate:
        q_next, R2, active_k, _sv2 = block_tsqr(q_next, mpi, comm, 1.0, deflate_tol=deflate_tol)
        if active_k <= 0:
            return None, alpha_i, None, 0.0, 0, True
        beta_i = R2 @ beta_i
        beta_norm = spectral_norm(beta_i)

    betas[it, :active_k, :p] = beta_i
    return q_next, alpha_i, beta_i, beta_norm, active_k, False


cdef class DenseKrylovStore:
    """The retained Krylov basis of an array (``ndarray``-block) run: ``(N_local, columns)`` in one
    over-allocated buffer grown geometrically (amortized O(1) copies instead of one full
    ``np.concatenate`` per step) -- the dense counterpart of :class:`SparseKrylovDense`, with the same
    ``append_block`` / ``reort`` / ``__len__`` surface, so one driver loop serves both.

    ``from_array`` adopts a caller's basis with zero spare capacity: the first append reallocates, so
    the caller's array is never written to.
    """
    cdef np.ndarray _buf
    cdef Py_ssize_t _cols

    def __cinit__(self, Py_ssize_t rows=0, np.ndarray initial=None):
        if initial is not None:
            self._buf = initial
            self._cols = initial.shape[1]
        else:
            self._buf = np.empty((rows, 0), dtype=complex, order='C')
            self._cols = 0

    @staticmethod
    def from_array(np.ndarray Q):
        return DenseKrylovStore(Q.shape[0], Q)

    def __len__(self):
        return self._cols

    def append_block(self, block):
        cdef Py_ssize_t n_new = block.shape[1]
        cdef Py_ssize_t cap = self._buf.shape[1]
        if self._cols + n_new > cap:
            grown = np.empty((self._buf.shape[0], max(2 * cap, self._cols + n_new)), dtype=complex, order='C')
            grown[:, : self._cols] = self._buf[:, : self._cols]
            self._buf = grown
        self._buf[:, self._cols : self._cols + n_new] = block
        self._cols += n_new

    def reort(self, wp, cols, int n_passes, comm):
        """``n_passes`` of classical Gram-Schmidt of ``wp`` against the stored columns (``cols``: a
        subset); returns ``(wp, last overlap)``. ``wp`` is updated in place."""
        Qv = self._buf[:, : self._cols] if cols is None else self._buf[:, cols]
        overlap = None
        for _ in range(n_passes):
            wp, overlap = block_orthogonalize_array(wp, Qv, None, comm)
        return wp, overlap

    def array(self):
        """The basis as an ``(N_local, columns)`` C-contiguous array. A trimmed copy when the buffer has
        spare capacity: a view would pin it for as long as the caller (e.g. a TRLM restart) holds it."""
        if self._buf.shape[1] == self._cols:
            return self._buf
        return np.ascontiguousarray(self._buf[:, : self._cols])


def _block_lanczos_array(
    np.ndarray psi0,
    h_op,
    converged,
    verbose=False,
    reort=False,
    alphas=None,
    betas=None,
    Q=None,
    comm=None,
    return_widths=False,
    return_status=False,
    build_krylov_basis=None,
    deflate_tol=-1.0,
    max_iter=None,
    block_widths_init=None,
    locked=None,
    tail=None,
    return_tail=False,
):
    """The array setup of :func:`block_lanczos`: ``h_op`` (a :class:`DistributedSparseMatrix`) applied to
    ``(N_local, p)`` blocks. The array front end of :func:`_lanczos_loop`: this function only builds
    the starting blocks and the dense Krylov store and packs the result.

    ``tail=(q_prev, q_curr)`` resumes a non-reorthogonalized recurrence from its last two blocks -- the local
    rows of the block before the current one and of the current one, each at its own width -- with
    ``alphas``/``betas`` the coefficients so far (padded to the seed width) and ``block_widths_init``
    their widths. It is how a sparse recurrence whose support froze carries on as the CSR recurrence of
    ``P H P`` (gf_solvers.block_Green_sparse), and how a tail-only run continues after its budget ran
    out. ``return_tail=True`` appends the ``(q_prev, q_curr)`` a ``"max_iter"`` stop can be resumed
    from (``None`` after any other stop).
    """
    reort = bool(reort)

    # Whether to retain the accumulated Krylov basis. The historical default (True) stands:
    # restart machinery (TRLM) and direct callers consume the returned Q regardless of
    # the reort mode. Callers that only need the final residual block (the GF continued
    # fraction with reort NONE) opt out explicitly, dropping the O(N * n * k) growth; any
    # reort mode projects against the basis and the warm-start protocol slices its blocks,
    # so opting out requires reort=False and a fresh start.
    cdef bint keep_krylov
    if tail is not None:
        if reort or Q is not None or psi0 is not None or build_krylov_basis:
            raise ValueError("a tail resume needs reort=False, no psi0, no Q and no Krylov basis")
        if alphas is None or betas is None or np.ndim(alphas) != 3:
            raise ValueError("a tail resume needs the (n_blocks, p, p) alphas and betas so far")
        if len(alphas) > 0 and (block_widths_init is None or len(block_widths_init) != len(alphas)):
            raise ValueError("a tail resume needs block_widths_init, one width per block so far")
        build_krylov_basis = False
    if build_krylov_basis is None:
        keep_krylov = True
    else:
        keep_krylov = bool(build_krylov_basis)
        if not keep_krylov and (reort or Q is not None):
            raise ValueError("build_krylov_basis=False requires reort=False and no warm-start Q")
    cdef int N
    cdef int n
    if tail is not None:
        N = tail[1].shape[0]
        n = alphas.shape[1]
    else:
        N = psi0.shape[0] if psi0 is not None else Q.shape[0]
        n = (psi0.shape[1] if psi0.ndim == 2 else 1) if psi0 is not None else alphas[0].shape[0]

    cdef int start_it = 0
    store = None
    resuming = alphas is not None and betas is not None
    if tail is not None:
        start_it = len(alphas)
        q_prev = np.ascontiguousarray(tail[0], dtype=complex)
        q_curr = np.ascontiguousarray(tail[1], dtype=complex)
    elif resuming and Q is not None:
        start_it = len(alphas)
        q0_fallback = psi0.copy() if psi0 is not None else Q[:, :n]
        q_prev = Q[:, (start_it - 1) * n : start_it * n] if start_it > 0 else q0_fallback
        q_curr = Q[:, start_it * n : (start_it + 1) * n] if start_it > 0 else q0_fallback
        store = DenseKrylovStore.from_array(Q)
    else:
        q_prev = np.zeros((N, n), dtype=complex, order='C')
        q_curr = np.ascontiguousarray(psi0 if psi0.ndim == 2 else psi0.reshape(-1, 1))
        if keep_krylov:
            store = DenseKrylovStore(N)
            store.append_block(q_curr)

    if max_iter is None:
        max_iter = int(np.ceil(h_op.shape[0] / n))
    cdef int budget = int(max_iter)
    cdef int buf_size = start_it + budget
    alphas_buf = np.zeros((buf_size, n, n), dtype=complex)
    betas_buf = np.zeros((buf_size, n, n), dtype=complex)
    if start_it > 0:
        alphas_buf[:start_it] = alphas
        betas_buf[:start_it] = betas

    if not isinstance(h_op, DistributedSparseMatrix):
        raise TypeError(f"h_op must be a DistributedSparseMatrix, got {type(h_op).__name__}")
    if h_op.shape[1] != N:
        raise ValueError(f"h_op owns {h_op.shape[1]} columns but the block has {N} local rows")
    cdef bint mpi = comm is not None
    if (comm is not None and comm.size > 1) != h_op.is_distributed:
        raise ValueError("h_op is distributed over a different communicator than the one the kernel reduces over")

    # The divergence guard of a resumed call keeps the scale the earlier steps established (and the
    # TSQR breakdown reference built from it), exactly as the ManyBodyState driver's resume does;
    # otherwise its first step would re-seed the scale unguarded.
    cdef double t_norm_max = 0.0
    cdef double h_norm_est = 0.0
    cdef bint guard_seeded = False
    cdef int k
    if tail is not None:
        for k in range(start_it):
            _, t_norm_max, h_norm_est = divergence_guard(
                spectral_norm(betas[k]), spectral_norm(alphas[k]), k == 0, t_norm_max, h_norm_est
            )
            guard_seeded = True

    # One exchange plan per call (three tiny collectives; `n` is the widest block this call will see
    # -- the active width only ever shrinks, so every round fits the plan's buffers). None on one rank.
    plan = h_op.exchange_plan(n, "block_lanczos")
    # The product buffer is reallocated when the active width shrinks; the matvec writes into it.
    out_holder = [np.empty((N, n), dtype=complex, order='C')]

    def matvec(x):
        if out_holder[0].shape[1] != x.shape[1]:
            out_holder[0] = np.empty((N, x.shape[1]), dtype=complex, order='C')
        return h_op.apply(x, plan, out=out_holder[0])

    # `locked` is column-distributed exactly like the Krylov vectors (local rows).
    locked_arr = None if locked is None else np.ascontiguousarray(locked, dtype=complex)

    cdef list block_widths = list(block_widths_init) if block_widths_init is not None else [n] * start_it
    it, termination, q_prev, q_curr, q_next, deflated_tail = _lanczos_loop(
        matvec, q_prev, q_curr, store, alphas_buf, betas_buf, start_it, budget, block_widths, converged, reort,
        mpi, comm, None, 0.0, 0.0, locked_arr, deflate_tol, verbose, t_norm_max, h_norm_est, guard_seeded, None,
        append_before_check=False, keep_deflated=not keep_krylov,
    )
    # The loop's step counter is relative to start_it.
    it += start_it

    if verbose and (not mpi or comm.rank == 0):
        print(f"Block Lanczos terminated ({termination}) at iteration {it}")

    # Tail-only mode: q_curr is the last block ever appended in stored mode (the roll runs before
    # the append, and every break happens before the append), so callers slicing the final residual
    # columns see identical data. The blocks recorded before each narrowing (deflated_tail) come
    # first, so those trailing columns are unchanged.
    if keep_krylov:
        res_Q = store.array()
    elif deflated_tail:
        res_Q = np.ascontiguousarray(np.hstack(deflated_tail + [q_curr]))
    else:
        res_Q = q_curr
    res = pack_lanczos_result(alphas_buf[:it], betas_buf[:it], res_Q, block_widths, termination,
                              return_widths, return_status)
    if return_tail:
        # Only a budget stop has rolled q onto the next step's (previous, current) pair; every other stop
        # ends the recurrence (or, "converged", breaks before the roll).
        res = tuple(res) + (((q_prev, q_curr) if termination == "max_iter" else None),)
    return res


def _lanczos_loop(
    matvec, q_prev, q_curr, store, alphas_buf, betas_buf, int start_it, int budget, list block_widths,
    converged_fn, reort, bint mpi, comm, basis, double slaterWeightMin, truncation_threshold, locked,
    double deflate_tol, bint verbose, double t_norm_max, double h_norm_est, bint guard_seeded, note_cut,
    bint append_before_check, bint keep_deflated,
):
    """The recurrence loop -- the one driver of :func:`lanczos_step`, for every block representation.

    Runs at most ``budget`` further steps from blocks ``q_prev``/``q_curr`` (steps ``start_it``
    onward), appending each accepted block to ``store`` (``None``: nothing is retained, the tail-only
    mode of a non-reorthogonalized run), guarding divergence and asking ``converged_fn(alphas, betas,
    verbose=, block_widths=)`` after each step. ``block_widths`` is extended in place with the width of
    every block a step was taken from. ``append_before_check`` orders the append of the new block
    against the convergence test (the ManyBodyState driver retains the residual block of a converged
    stop, the array driver does not); ``keep_deflated`` collects, in ``deflated_tail``, each block whose
    successor came out narrower (tail-only array runs).

    Returns ``(steps_taken, termination, q_prev, q_curr, q_next, deflated_tail)`` with the blocks as
    the loop left them: after a ``"max_iter"`` or ``"frozen"`` stop the ``q_prev``/``q_curr`` roll has
    run (the state a resume continues from), after ``"converged"`` it has not (``q_next`` is the
    residual), after a breakdown ``q_next`` is ``None``.

    ``termination`` is one of ``"converged"``, ``"invariant_subspace"`` (the block-Krylov space is
    closed under the operator within the active restrictions: the result is *exact*), ``"diverged"``
    (the divergence guard truncated a corrupted tail: NOT exact), ``"max_iter"`` or ``"frozen"`` (the
    basis proxy froze during the last step).
    """
    cdef int it = 0
    cdef int it_abs, n_curr, active_k
    cdef double beta_norm, alpha_norm
    cdef bint breakdown, diverged
    cdef list deflated_tail = []
    termination = "max_iter"
    q_next = None
    # A growth-capping proxy whose caller asked to be handed control at the freeze (the GF
    # frozen-basis CSR fallback, gf_solvers.block_Green_sparse). Its `frozen` flag derives only
    # from allreduced data, so every rank of the communicator stops on the same step.
    stop_on_freeze = basis is not None and bool(getattr(basis, "stop_on_freeze", False))

    while it < budget:
        it_abs = start_it + it
        n_curr = block_cols(q_curr)

        # The matvec exchanges `rows x n_curr` blocks: if the ranks ever disagree about the width,
        # they post the same collective with different counts and MPICH memcpys past the receive
        # buffer -- a SIGSEGV inside MPI_Reduce, with no Python traceback, no core dump (the MPI
        # runtime's own SIGSEGV handler exits 1 instead of re-raising) and every rank dying mute.
        # Unconditional and collective on every rank, so it cannot itself desynchronize; one
        # 2-element Allreduce against a matvec is not measurable.
        if mpi:
            width_probe = np.array([n_curr, -n_curr], dtype=np.int64)
            comm.Allreduce(MPI.IN_PLACE, width_probe, op=MPI.MAX)
            if width_probe[0] != -width_probe[1]:
                raise RuntimeError(
                    f"block width disagrees across ranks at iteration {it_abs}: this rank has "
                    f"n_curr={n_curr}, the communicator spans [{-width_probe[1]}, {width_probe[0]}]. "
                    "The width is set by block_tsqr's retained rank, which is computed from a "
                    "replicated R and must therefore be rank-invariant; a disagreement means that "
                    "invariant has broken."
                )

        q_next, alpha_i, beta_i, beta_norm, active_k, breakdown = lanczos_step(
            matvec, q_prev, q_curr, store, alphas_buf, betas_buf, it_abs, reort, mpi, comm,
            slaterWeightMin=slaterWeightMin,
            truncation_threshold=truncation_threshold,
            locked=locked,
            note_cut=note_cut,
            h_norm_est=max(h_norm_est, t_norm_max),
            deflate_tol=deflate_tol,
        )

        if breakdown:
            # active_k < 0 marks a non-finite (corrupted) factor -> truncated, NOT exact;
            # active_k == 0 is a genuine rank-deficient residual -> the block-Krylov space is
            # closed under H -> the continued fraction is exact on it.
            termination = "diverged" if active_k < 0 else "invariant_subspace"
            # "diverged" is a problem the caller must see regardless of verbosity;
            # "invariant_subspace" is a legitimate exact-termination case, stays at -vv.
            if (termination == "diverged" or verbose) and (comm is None or comm.Get_rank() == 0):
                print(f"[BlockLanczos] {termination} detected at iteration {it_abs}.")
            block_widths.append(n_curr)
            it += 1
            break

        # --- Divergence safeguard ---------------------------------------
        # For a Hermitian operator every block norm is bounded by ||A||, so ||alpha_i||/||beta_i||
        # can never exceed the largest healthy block norm by more than rounding. A jump of
        # several orders of magnitude means the recurrence has been corrupted (lost
        # orthogonality the QR/deflation did not repair). Truncate *before* the offending
        # block -- exclude alpha_i/beta_i at it_abs -- so the returned T-matrix is the last
        # numerically trustworthy subspace and the diverged tail never reaches the Green's
        # function. The trailing beta then plays the role of the (ignored) residual coupling.
        alpha_norm = spectral_norm(alpha_i)
        # first_step = it == 0: this loop's own counter, always 0 on the first step executed by THIS
        # call -- see check_divergence's docstring for why this must not be "it_abs == 0" -- unless a
        # resume already replayed the guard.
        diverged, t_norm_max, h_norm_est = check_divergence(
            beta_norm, alpha_norm, it == 0 and not guard_seeded, t_norm_max, h_norm_est, it_abs, comm
        )
        if diverged:
            termination = "diverged"
            break

        # Retain q_next (last block = residual direction). The locking deflation is applied inside
        # lanczos_step (before beta), so q_next is orthogonal to the locked set.
        if append_before_check and store is not None:
            # append_block copies the coefficients into the store straight from the block rows -- no
            # per-state materialization, no defensive copies.
            store.append_block(q_next)

        if verbose and (comm is None or comm.Get_rank() == 0):
            print(
                f"[BlockLanczos] it={it_abs:4d}  " f"|beta|={beta_norm:.3e}  " f"alpha_diag={np.real(np.diag(alpha_i))}"
            )

        # Convergence check
        _converged = converged_fn(
            alphas_buf[: it_abs + 1], betas_buf[: it_abs + 1], verbose=verbose, block_widths=block_widths + [n_curr]
        )
        if mpi:
            # The verdict is computed rank-locally from alpha, which comes out of an Allreduce --
            # and MPI does not promise a bitwise-identical Allreduce result on every rank (a
            # recursive-doubling reduction sums in a rank-dependent order). A verdict that flips
            # on one rank near the threshold sends it out of the loop while the others enter the
            # next step's collectives. Root decides for everyone (review ledger M2).
            _converged = comm.bcast(bool(_converged), root=0)
        if _converged:
            termination = "converged"
            block_widths.append(n_curr)
            it += 1
            break

        if not append_before_check and store is not None:
            store.append_block(q_next)
        if keep_deflated and block_cols(q_next) < n_curr:
            # The last vectors of the columns that deflated here: a basis-growing caller
            # (gf_solvers.block_Green) must probe from them too -- a column whose chain closes only
            # because the still-incomplete basis truncates H deflates, and probing from the final
            # block alone never reaches the determinants that chain was missing (review ledger C12).
            deflated_tail.append(q_curr.copy())
        q_prev = q_curr
        q_curr = q_next
        block_widths.append(n_curr)
        it += 1

        # After the roll, so the returned state is exactly a "max_iter" stop's: a caller that
        # declines the hand-off resumes from it with the usual *_init protocol.
        if stop_on_freeze and basis.frozen:
            termination = "frozen"
            break

    return it, termination, q_prev, q_curr, q_next, deflated_tail


def _pack_result(
    alphas_buf, betas_buf, start_it, it, store_krylov, termination,
    q_prev, q_curr, q_next, Q_basis, block_widths, return_widths, return_status,
):
    """Build block_lanczos's return tuple: the finished alphas/betas views, the
    Krylov basis (or, with store_krylov=False, its two-block tail), and the
    return_widths/return_status opt-in appends. Reads only finished loop state --
    nothing here feeds back into the recurrence."""
    alphas_out = alphas_buf[:start_it + it]
    betas_out = betas_buf[:start_it + it]
    if not store_krylov:
        # Return the two-block tail [last accepted block, residual block], mirroring the
        # last two blocks of the full Q_basis: on a "converged" break the q_prev/q_curr
        # roll has not run yet (tail = q_curr + q_next); on breakdown/divergence q_next
        # was never appended, and on budget exhaustion the roll has run (both: tail =
        # q_prev + q_curr).
        if termination == "converged":
            Q_basis = q_curr.to_states() + q_next.to_states()
        else:
            Q_basis = q_prev.to_states() + q_curr.to_states()
    return pack_lanczos_result(alphas_out, betas_out, Q_basis, block_widths, termination, return_widths, return_status)


def _block_lanczos_mbs(
    psi0,
    h_op,
    basis,
    converged_fn,
    verbose: bool = False,
    reort=True,
    max_iter=None,
    slaterWeightMin: float = 0.0,
    truncation_threshold: int = 0,
    comm=None,
    alphas_init=None,
    betas_init=None,
    Q_init=None,
    return_widths=False,
    return_status=False,
    block_widths_init=None,
    locked=None,
    store_krylov=True,
    krylov_dtype=None,
    deflate_tol=-1.0,
):
    """The ``ManyBodyState`` setup of :func:`block_lanczos` (distributed block Lanczos iteration).

    Implements the block Lanczos three-term recurrence

    .. math::

        H Q_i = Q_i \\alpha_i + Q_{i+1} \\beta_i + Q_{i-1} \\beta_{i-1}^\\dagger

    building the block-tridiagonal matrix

    .. math::

        T_m = \\begin{pmatrix}
            \\alpha_0 & \\beta_0^\\dagger & & \\\\
            \\beta_0  & \\alpha_1 & \\ddots & \\\\
                     & \\ddots & \\ddots & \\beta_{m-2}^\\dagger \\\\
                     & & \\beta_{m-2} & \\alpha_{m-1}
        \\end{pmatrix}.

    ``alphas_buf``/``betas_buf`` are pre-allocated once; the returned arrays are
    *views* into them, not copies. Resumes from block ``len(alphas_init)`` when
    ``alphas_init``, ``betas_init`` and ``Q_init`` are all given (``None`` for all
    three starts fresh) — see "Warm-start / resume protocol" in
    doc/lanczos_invariants.md for the full protocol.

    **MPI collective operations** (called every iteration, all ranks must participate):

    * One ``MPI_Allreduce`` (``MPI.SUM``) for :math:`\\alpha_i` – shape
      ``(p, p)``; result is replicated on all ranks.
    * One ``MPI_Allgather`` of the packed triangular factor – ``p(p+1)/2``
      complex per rank; the TSQR of the residual block.
    * Additional ``MPI_Allreduce`` calls inside reorthogonalization (see
      ``lanczos_step`` for details).

    Note:
        ``converged_fn`` is called on **all ranks** with *replicated* data
        (``alphas``, ``betas`` are identical on every rank).  If the callable
        performs rank-dependent logic the caller is responsible for broadcasting
        the result.  When ``reort`` is ``False`` no reorthogonalization of any
        kind is performed; orthogonality degrades at the rate of floating-point
        rounding errors.

    Args:
        psi0: Initial block of ``p`` ``ManyBodyState`` objects (or an equivalent
            ``ManyBodyState``), or ``None`` when resuming from ``Q_init``
            (warm-start mode).
        h_op: ``ManyBodyOperator`` that implements ``apply_multi(psis, cutoff)``.
        basis: ``Basis`` object providing ``redistribute_psis`` and ``basis.comm``.
        converged_fn: Callable with signature
            ``converged_fn(alphas, betas, verbose=bool, block_widths=list[int]) -> bool``
            that returns ``True`` once the desired accuracy is reached.  Called after
            every accepted step with the *full* ``alphas``/``betas`` arrays built so
            far and ``block_widths`` extended with the current step's width -- a
            callback that only accepts ``(alphas, betas, verbose)`` raises
            ``TypeError`` (accept ``**kwargs`` if ``block_widths`` is unused).
        verbose: Print per-iteration diagnostics including ``|beta|`` norms and
            diagonal alpha values.  Default ``False``.
        reort: ``True`` projects each new block against the whole stored basis (two Gram-Schmidt
            passes); ``False`` performs no reorthogonalization.  Default ``True``.
        max_iter: Maximum number of Lanczos steps before returning.  Defaults to
            ``basis.size // p`` when ``basis.size`` is available, otherwise
            ``10 * p``.
        slaterWeightMin: Amplitude cutoff passed to ``ManyBodyState.prune``.
            Slater determinants with :math:`|c_k| < \\text{slaterWeightMin}` are
            dropped after each matvec.  Default ``0.0`` (no pruning).
        comm: ``mpi4py`` communicator.  If ``None``, falls back to ``basis.comm``
            and then to serial mode (``mpi=False``).
        alphas_init: Warm-start diagonal block array of shape ``(k0, p, p)``.
            If provided together with ``betas_init`` and ``Q_init``, iteration
            resumes from block ``k0``.  Default ``None``.
        betas_init: Warm-start off-diagonal block array of shape ``(k0, p, p)``.
            Default ``None``.
        Q_init: Warm-start Krylov basis: the ``SparseKrylovDense`` store returned
            by a previous run (adopted as-is, zero-copy resume) or a legacy flat
            list of ``ManyBodyState`` (length ``>= k0 * p``, ingested into a fresh
            store).  Default ``None``.
        store_krylov: When ``False`` (requires ``reort=False`` and no
            ``locked`` set) the accumulated Krylov basis is *not* retained; see
            "Krylov store" in doc/lanczos_invariants.md.  Default ``True``.
        krylov_dtype: Storage dtype of the retained Krylov basis --
            ``complex128`` (default) or ``complex64``.  Default
            ``None`` (complex128).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, list]: A 3-tuple
        ``(alphas, betas, Q_basis)`` where

        * ``alphas`` – complex array of shape ``(k, p, p)`` holding the
          diagonal blocks :math:`\\alpha_0, \\dots, \\alpha_{k-1}`.
        * ``betas`` – complex array of shape ``(k, p, p)`` holding the
          off-diagonal blocks :math:`\\beta_0, \\dots, \\beta_{k-1}`.
        * ``Q_basis`` – the Krylov basis as a ``SparseKrylovDense`` column store
          of ``(k + 1) * p`` columns (a sequence of ``ManyBodyState``: ``len``,
          indexing, slicing and iteration materialize columns on demand); the
          last ``p`` columns form the residual.  With ``store_krylov=False`` a
          plain two-block tail list instead (see below).

        ``return_widths=True`` appends ``block_widths`` (the per-block true widths)
        and ``return_status=True`` appends a ``termination`` string -- one of
        ``"converged"`` (``converged_fn`` satisfied), ``"invariant_subspace"`` (the
        block-Krylov space is closed under H within the active restrictions, so the
        result is *exact*), ``"diverged"`` (the divergence guard truncated a corrupted
        tail; NOT exact), ``"max_iter"`` (the ``max_iter`` budget was exhausted before
        the recurrence terminated naturally) or ``"frozen"`` (``basis`` is a growth-capping
        proxy with ``stop_on_freeze`` set and froze during the last step; the state is a
        ``"max_iter"`` stop's, so the recurrence can be resumed). Both flags are independent and opt-in, so
        call sites that pass neither get the 3-tuple.
    """
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    if isinstance(h_op, dict):
        h_op = ManyBodyOperator(h_op)

    # --- Resolve communicator / MPI flag --------------------------------
    if comm is None:
        comm = getattr(basis, "comm", None)
    mpi = comm is not None and comm.Get_size() > 1

    reort = bool(reort)

    # Tail-only mode: with reort off (and no locked set) nothing in the recurrence
    # ever projects against the accumulated Krylov basis, and the warm-start protocol
    # reads only the last two blocks -- so retaining the full O(N_det * p * k) basis is
    # pure dead memory (the dominant avoidable allocation in a Green's-function run).
    #
    # `locked` gates this the same way as the width test below (nlock, hoisted here so
    # both this check and the width test share one rank-invariant value):
    # `if locked` on a ManyBodyState reads its LOCAL ROW count, which can disagree
    # across ranks and turn this `raise` into a rank-divergent collective itself.
    nlock = 0 if locked is None else block_cols(locked)
    if not store_krylov and (reort or nlock > 0):
        raise ValueError("store_krylov=False requires reort=False and no locked vectors")

    # --- Resume or start fresh? -----------------------------------------
    resuming = alphas_init is not None and betas_init is not None and Q_init is not None

    cdef list block_widths = list(block_widths_init) if block_widths_init is not None else []
    if resuming:
        start_it = len(alphas_init)
        p = alphas_init[0].shape[0] if len(alphas_init) > 0 else len(Q_init[0] if Q_init else psi0)
        if len(block_widths) == 0:
            block_widths = [p] * start_it

        if not store_krylov:
            # Q_init is the two-block tail [prev block, residual block] returned by the
            # previous tail-only run; split it by the true width of the last accepted
            # block (deflation-safe: block_widths is authoritative, the tail may be
            # narrower than p).
            Q_basis = []
            if start_it == 0:
                q_prev = ManyBodyState(width=p)
                q_curr = ManyBodyState.from_states(list(Q_init))
            else:
                q_prev_len = block_widths[start_it - 1]
                q_prev = ManyBodyState.from_states(list(Q_init[:q_prev_len]))
                q_curr = ManyBodyState.from_states(list(Q_init[q_prev_len:]))
        else:
            # Columnar retention: the store is the ONLY copy of the Krylov basis (shared
            # determinant->row support + one dense coefficient buffer, ~16 B/coeff vs
            # ~72 B/coeff for a list of flat_map states). A store from a previous run
            # (the resume round-trip) is adopted as-is; a legacy list is ingested once.
            if isinstance(Q_init, SparseKrylovDense):
                Q_basis = Q_init
            else:
                Q_basis = SparseKrylovDense(krylov_dtype)
                # Row hint = the local basis size (an upper bound on the support after
                # redistribution): chunks are sized so row growth never forces a new one.
                _local_basis = getattr(basis, "local_basis", None)
                if _local_basis is not None:
                    Q_basis.reserve_rows(len(_local_basis))
                if isinstance(Q_init, ManyBodyState):
                    # A caller (IRLM's restart continuation) resuming from its own
                    # block-native Q_basis_new. list(Q_init) below is for the OTHER
                    # branch only (a legacy flat list of ManyBodyState): iterating a
                    # ManyBodyState yields its determinant KEYS, not states, so
                    # append(list(Q_init)) would silently corrupt the store instead of
                    # raising. append_block is the direct, no-round-trip ingestion.
                    if Q_init.width > 0:
                        Q_basis.append_block(Q_init)
                elif len(Q_init) > 0:
                    Q_basis.append(list(Q_init))
            if start_it == 0 or len(Q_basis) < p:
                q_prev = ManyBodyState(width=p)
                q_curr = Q_basis.slice_block(0, p)
            else:
                q_prev_start = sum(block_widths[:start_it - 1])
                q_prev_len = block_widths[start_it - 1]
                q_curr_start = sum(block_widths[:start_it])
                q_prev = Q_basis.slice_block(q_prev_start, q_prev_start + q_prev_len)
                q_curr = Q_basis.slice_block(q_curr_start, len(Q_basis))
    else:
        start_it = 0

        # psi0 may already be a ManyBodyState (a caller with its own block-native
        # seed, e.g. the TRLM entry points) or a legacy flat list of ManyBodyState --
        # len()/list() on a block would read its ROW count / iterate its determinant KEYS,
        # not its width/states, so the two must be told apart explicitly rather than
        # funneled through one call. Either way, adopt the shared-support block
        # representation for the live recurrence blocks.
        if isinstance(psi0, ManyBodyState):
            p = psi0.width
            q_curr = (basis.admit_seed if hasattr(basis, "admit_seed") else basis.redistribute_block)(psi0)
        else:
            p = len(psi0)
            q_curr = ManyBodyState.from_states(basis.redistribute_psis(*psi0))
        q_prev = ManyBodyState(width=p)
        if store_krylov:
            Q_basis = SparseKrylovDense(krylov_dtype)
            _local_basis = getattr(basis, "local_basis", None)
            if _local_basis is not None:
                Q_basis.reserve_rows(len(_local_basis))
            Q_basis.append_block(q_curr)
        else:
            Q_basis = []

    # --- Determine max_iter ---------------------------------------------
    if max_iter is None:
        max_iter = getattr(basis, "size", 10 * p) // p

    # Pre-allocate buffers to avoid allocation inside loop
    _buf_size = max(int(max_iter), 1)
    if start_it > 0:
        alphas_buf = np.zeros((start_it + _buf_size, p, p), dtype=complex)
        betas_buf = np.zeros((start_it + _buf_size, p, p), dtype=complex)
        alphas_buf[:start_it] = alphas_init
        betas_buf[:start_it] = betas_init
    else:
        alphas_buf = np.zeros((_buf_size, p, p), dtype=complex)
        betas_buf = np.zeros((_buf_size, p, p), dtype=complex)

    # --- Main Lanczos loop ----------------------------------------------
    # Largest healthy block-norm seen so far (proxy for ||H|| on the subspace), and the spectral-scale
    # estimate (seeded from beta_0 ~ ||H||, grown only by ||alpha_i||, which is bounded by ||H|| and
    # never runs away): the divergence guard's state. A resumed call (a doubled budget, a declined
    # freeze) replays the guard over the blocks it was handed, so it keeps the scale the earlier calls
    # built: starting both trackers from 0 left every resumed call's first step unguarded and let that
    # step re-seed the scale -- every 16 blocks while capped GF units ran a window at a time.
    t_norm_max = 0.0
    h_norm_est = 0.0
    guard_seeded = False
    if resuming:
        for _k in range(min(len(alphas_init), len(betas_init))):
            _, t_norm_max, h_norm_est = divergence_guard(
                spectral_norm(betas_init[_k]), spectral_norm(alphas_init[_k]), _k == 0, t_norm_max, h_norm_est
            )
            guard_seeded = True

    # The matvec: a growth-capping basis proxy (caps_growth=True) must see every step's residual even
    # serially -- its redistribute_block enforces the truncation_threshold there (the inner Basis
    # redistribute no-ops without MPI). Chunking only where a redistribute actually follows;
    # `APPLY_ROW_CHUNKS` defaults to 4 and `1` is the one-shot path. The matvec itself (chunking,
    # the cut on summed rows, a capping proxy's deferred admission) is `apply_and_redistribute`.
    _needs_redistribute = basis is not None and ((mpi and comm is not None) or getattr(basis, "caps_growth", False))
    _n_chunks = config.APPLY_ROW_CHUNKS.get() if _needs_redistribute else 1
    _n_chunks = 1 if _n_chunks is None else int(_n_chunks)

    def matvec(q):
        return apply_and_redistribute(h_op, q, basis, slaterWeightMin, _needs_redistribute, _n_chunks)

    # A proxy running the support freeze also weighs what the next block's prune drops (replicated flag).
    _note_cut = (
        basis.note_step_cut
        if _needs_redistribute and hasattr(basis, "begin_step") and getattr(basis, "support_freeze", False)
        else None
    )

    it, termination, q_prev, q_curr, q_next, _ = _lanczos_loop(
        matvec, q_prev, q_curr, Q_basis if store_krylov else None, alphas_buf, betas_buf, start_it, _buf_size,
        block_widths, converged_fn, reort, mpi, comm, basis, slaterWeightMin, truncation_threshold, locked,
        deflate_tol, verbose, t_norm_max, h_norm_est, guard_seeded, _note_cut,
        append_before_check=True, keep_deflated=False,
    )

    return _pack_result(
        alphas_buf, betas_buf, start_it, it, store_krylov, termination,
        q_prev, q_curr, q_next, Q_basis, block_widths, return_widths, return_status,
    )


def block_lanczos(
    psi0,
    h_op,
    basis=None,
    converged=None,
    verbose=False,
    reort=True,
    max_iter=None,
    slaterWeightMin=0.0,
    truncation_threshold=0,
    comm=None,
    alphas_init=None,
    betas_init=None,
    Q_init=None,
    block_widths_init=None,
    locked=None,
    store_krylov=None,
    krylov_dtype=None,
    deflate_tol=-1.0,
    tail=None,
    return_widths=False,
    return_status=False,
    return_tail=False,
):
    """Block Lanczos: the one entry point, for either block representation.

    The recurrence is :func:`lanczos_step` driven by :func:`_lanczos_loop` whatever the blocks are;
    this function only picks the representation's setup. An ``ndarray`` start block (or ``Q_init`` /
    ``tail``) means the array representation, with ``h_op`` a :class:`DistributedSparseMatrix` (a scipy
    or dense local matrix is converted once). Anything else is a ``ManyBodyState`` block (or a list of
    them) with ``h_op`` a ``ManyBodyOperator`` and ``basis`` the distributed basis.

    ``converged(alphas, betas, **kw) -> bool`` is called on every rank with replicated data after each
    accepted step; ``reort=True`` projects each new block against the stored Krylov basis (two
    passes), ``False`` does not. ``alphas_init``/``betas_init``/``Q_init``/``block_widths_init`` resume
    a previous run (see "Warm-start / resume protocol" in doc/lanczos_invariants.md);
    ``store_krylov=False`` (needs ``reort=False``, no ``locked`` set and a fresh start) keeps only the
    last two blocks. The return is ``(alphas, betas, Q)`` plus ``block_widths`` (``return_widths``),
    the termination string (``return_status``: ``"converged"``, ``"invariant_subspace"``,
    ``"diverged"``, ``"max_iter"`` or ``"frozen"``) and, for arrays, the resumable ``(q_prev, q_curr)``
    tail (``return_tail``).

    Array-only: ``tail`` (resume a non-reorthogonalized recurrence from its last two blocks) and
    ``return_tail``. ManyBodyState-only: ``basis``, ``slaterWeightMin``, ``truncation_threshold`` and
    ``krylov_dtype``; passing them with array blocks raises instead of being silently ignored.
    """
    is_array_run = (
        tail is not None
        or isinstance(psi0, np.ndarray)
        or (psi0 is None and isinstance(Q_init, np.ndarray))
    )
    if converged is None:
        raise TypeError("block_lanczos needs a `converged` callable")
    if is_array_run:
        if slaterWeightMin or truncation_threshold or krylov_dtype is not None:
            raise ValueError("slaterWeightMin, truncation_threshold and krylov_dtype apply to ManyBodyState blocks only")
        if comm is None:
            comm = getattr(basis, "comm", None)
        h_op = DistributedSparseMatrix.coerce(h_op, comm)
        return _block_lanczos_array(
            psi0, h_op, converged, verbose=verbose, reort=reort, alphas=alphas_init, betas=betas_init, Q=Q_init,
            comm=comm, return_widths=return_widths, return_status=return_status, build_krylov_basis=store_krylov,
            deflate_tol=deflate_tol, max_iter=max_iter, block_widths_init=block_widths_init, locked=locked,
            tail=tail, return_tail=return_tail,
        )
    if tail is not None or return_tail:
        raise ValueError("tail / return_tail apply to array blocks only")
    return _block_lanczos_mbs(
        psi0, h_op, basis, converged, verbose=verbose, reort=reort, max_iter=max_iter,
        slaterWeightMin=slaterWeightMin, truncation_threshold=truncation_threshold, comm=comm,
        alphas_init=alphas_init, betas_init=betas_init, Q_init=Q_init, return_widths=return_widths,
        return_status=return_status, block_widths_init=block_widths_init, locked=locked,
        store_krylov=True if store_krylov is None else store_krylov, krylov_dtype=krylov_dtype,
        deflate_tol=deflate_tol,
    )
