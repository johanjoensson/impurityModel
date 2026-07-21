# ===========================================================================
# Core block-Lanczos recurrence
# ===========================================================================
# One three-term block step (block_lanczos_step_cy) and the full recurrence driver
# (block_lanczos_cy) that serves the whole frequency mesh from a single recurrence. The
# reorthogonalization mode (NONE/PARTIAL/FULL/PERIODIC/SELECTIVE) and the EA16
# shrinking-block deflation policy are dispatched from here.

def block_lanczos_step_cy(
    h_op,
    q_prev,
    q_curr,
    Q_basis,
    alphas,
    betas,
    it: int,
    reort_mode,
    W,
    mpi: bool,
    comm,
    basis,
    slaterWeightMin: float = 0.0,
    truncation_threshold: float = 0.0,
    reort_period: int = 5,
    start_it: int = 0,
    block_widths=None,
    locked=None,
    locked_reort="full",
    krylov=None,
    w_out=None,
    beta_norm_hist=None,
    force_reort: bool = False,
    h_norm_est: float = 0.0,
):
    """Perform one step of the distributed block Lanczos iteration.

    Executes the three-term block recurrence

    .. math::

        W_p = H Q_i - Q_i \\alpha_i - Q_{i-1} \\beta_{i-1}^\\dagger

    where :math:`\\alpha_i = Q_i^\\dagger H Q_i` is the diagonal block and
    :math:`\\beta_{i-1}` is the off-diagonal block from the previous step.
    The residual block :math:`W_p` is then QR-factorised via the Cholesky
    decomposition of :math:`M = W_p^\\dagger W_p`:

    .. math::

        M = L L^\\dagger, \\quad
        \\beta_i = L^\\dagger, \\quad
        Q_{i+1} = W_p \\beta_i^{-1}.

    **MPI collective operations** (all ranks must call simultaneously):

    1. ``MPI_Allreduce`` (``MPI.SUM``) on :math:`\\alpha_i` – :math:`p \\times p`
       complex matrix; result is replicated on every rank.
    2. ``MPI_Allreduce`` (``MPI.SUM``) on :math:`M = W_p^\\dagger W_p` – :math:`p \\times p`
       complex matrix; used for Cholesky QR and breakdown detection.
    3. For ``FULL`` / ``PERIODIC`` modes, an additional ``MPI_Allreduce`` per
       reorthogonalization pass on the :math:`(it \\cdot p) \\times p` overlap matrix.
    4. For ``PARTIAL``, standard Paige-Simon tracking operates against all Lanczos vectors,
       triggering ``MPI_Allreduce`` calls over potentially large swaths of the Krylov
       basis when orthogonality loss exceeds :math:`\\sqrt{\\varepsilon_{\\text{mach}}}`.
    5. For ``SELECTIVE`` (EA16), tracking operates exclusively against converged Ritz vectors.
       The Ritz overlap is calculated as $w_{ritz, k} = s_k^\\dagger W[:, \\text{curr}]$, and
       the W-matrix geometric projection update is
       $W[:, \\text{curr}] \\leftarrow W[:, \\text{curr}] - (W \\cdot s_k) \\langle y_k \\mid w_p \\rangle$.
       Because ``SELECTIVE`` exclusively projects against a small subset of Ritz vectors rather
       than the entire basis, the ``MPI_Allreduce`` footprint is drastically lower.

    Note:
        ``h_op.apply_multi`` releases the GIL internally; all other operations in
        this function hold the GIL.

    Args:
        h_op: ``ManyBodyOperator`` Hamiltonian; must implement
            ``apply_multi(psis, cutoff)``.
        q_prev: ``ManyBodyBlockState`` of width ``p`` from iteration ``i-1``
            (a zero-row block of width ``p`` at ``it=0``).
        q_curr: ``ManyBodyBlockState`` of width ``p`` from iteration ``i``.
        Q_basis: Accumulated Krylov basis as a flat list of ``ManyBodyState``
            (length ``p * (it + 1)`` on entry, grows by ``p`` each step).
        alphas: Pre-allocated numpy array of shape ``(max_iter, p, p)`` that
            receives the diagonal block :math:`\\alpha_i` at index ``it``.
        betas: Pre-allocated numpy array of shape ``(max_iter, p, p)`` that
            receives the off-diagonal block :math:`\\beta_i` at index ``it``.
        it: Current iteration index into ``alphas`` / ``betas`` buffers
            (0-based, reset to 0 at the start of each fresh run).
        reort_mode: ``Reort`` enum value controlling reorthogonalization strategy.
            ``Reort.NONE`` skips all reorthogonalization entirely.
        W: Paige-Simon W-matrix estimator state array of shape
            ``(2, it+1, p, p)``, or ``None`` when not used.  Required for
            ``PARTIAL`` / ``SELECTIVE`` modes; ignored otherwise.
        mpi: ``True`` when running under MPI with more than one rank.
        comm: Active ``mpi4py.MPI.Comm`` communicator, or ``None`` when running
            serially.
        slaterWeightMin: Amplitude cutoff passed to ``ManyBodyState.prune``;
            SD coefficients below this value are dropped.  Default ``0.0``
            (no pruning).
        reort_period: Number of steps between full reorthogonalization sweeps
            for ``Reort.PERIODIC`` mode.  Full reorthogonalization is applied at
            step ``it`` when ``it > 0`` and ``it % reort_period == 0``.
            Default ``5``.

    Returns:
        tuple: A 7-tuple ``(q_next, alpha_i, beta_i, W_updated, active_k, breakdown,
        reort_acted)``; ``reort_acted`` is True iff the PARTIAL/SELECTIVE bad-block
        projection actually fired this step (the driver then forces a re-check on the
        next step — the two-consecutive-steps rule):

        * ``q_next`` – ``ManyBodyBlockState`` (width ``active_k``) forming the next
          Krylov block :math:`Q_{i+1}`, or ``None`` if breakdown occurred.
        * ``alpha_i`` – numpy complex array of shape ``(p, p)`` for the current
          diagonal block.
        * ``beta_i`` – numpy complex array of shape ``(p, p)`` for the current
          off-diagonal block, or ``None`` if breakdown occurred.
        * ``W_updated`` – updated Paige-Simon estimator array, or ``None`` if
          not applicable.
        * ``breakdown`` – ``True`` if an invariant subspace or an ill-conditioned
          block was detected (``NaN``/``Inf`` in :math:`M`, or condition number
          exceeding :math:`100 / \\varepsilon_{\\text{mach}}`).
    """
    # q_prev / q_curr are shared-support ManyBodyBlockStates (Phase 2.4): the matvec,
    # Gram products and axpy updates below run once per determinant ROW instead of once
    # per (determinant, vector) pair. All block primitives are bit-for-bit identical to
    # the old list-of-ManyBodyState ops (same accumulation order); only the pruning
    # keeps whole rows (any-column-survives) instead of per-column entries.
    p = q_curr.width

    # --- 1. Block matvec: wp = H q_curr ---------------------------------
    _t0 = _time.perf_counter()
    wp = h_op.apply_block(q_curr, slaterWeightMin)
    _prof_acc("matvec_apply", _t0)
    _t1 = _time.perf_counter()
    # A growth-capping basis proxy (caps_growth=True) must see every step's residual
    # even serially: its redistribute_block enforces the truncation_threshold there
    # (the inner Basis redistribute no-ops without MPI).
    if basis is not None and ((mpi and comm is not None) or getattr(basis, "caps_growth", False)):
        if hasattr(basis, "redistribute_block"):
            wp = basis.redistribute_block(wp)
        else:
            # Duck-typed basis without the block method (e.g. a test mock): fall back
            # to the scalar redistribute through a boundary conversion.
            wp = ManyBodyBlockState.from_states(basis.redistribute_psis(wp.to_states()))
    _prof_acc("matvec_redistribute", _t1)
    _prof_acc("matvec", _t0)

    # --- 2. alpha_i = <q_curr | wp> -------------------------------------
    _t0 = _time.perf_counter()
    alpha_i = block_inner_cy(q_curr, wp)
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, alpha_i, op=MPI.SUM)
    alphas[it, :p, :p] = alpha_i

    # --- 3. Subtract: wp = wp - q_curr * alpha_i - q_prev * beta_{i-1}^† -
    wp = block_add_scaled_cy(wp, q_curr, -alpha_i)
    if it > 0:
        n_prev = q_prev.width
        beta_prev_dag = np.conj(betas[it - 1, :p, :n_prev].T)
        wp = block_add_scaled_cy(wp, q_prev, -beta_prev_dag)

    # --- 3b. EA16 §2.6.2 locking deflation ------------------------------
    # Keep the residual block orthogonal to the already-converged ("locked") Ritz
    # vectors *before* forming M and beta, so the stored beta and the resulting
    # q_next are consistent with the deflated vector (matching the array kernel).
    # The locked Ritz vectors are only approximate eigenvectors, so H q_curr leaks
    # a small component along them that is otherwise amplified across iterations,
    # reintroducing locked eigenvalues (and their 2*theta harmonics) as spurious
    # Ritz values below the true spectral minimum on restarted sweeps. Twice for
    # numerical robustness. Skipped in the "partial" mode, where the estimate-driven
    # EA16 §2.6.2 reorth is applied to q_next in block_lanczos_cy instead.
    if locked and locked_reort != "partial":
        locked_blk = (
            locked if isinstance(locked, ManyBodyBlockState) else ManyBodyBlockState.from_states(list(locked))
        )
        for _ in range(2):
            _ovl = block_inner_cy(locked_blk, wp)
            if mpi and comm is not None:
                comm.Allreduce(MPI.IN_PLACE, _ovl, op=MPI.SUM)
            wp = block_add_scaled_cy(wp, locked_blk, -_ovl)

    # --- 4. Full / Periodic reorthogonalization -------------------------
    # The PERIODIC cadence gate stays in the caller; the reort action itself goes
    # through the shared apply_reort (single FULL implementation for both kernels;
    # bit-for-bit equal to the old 2x inner_multi/add_scaled_multi loop because
    # block_orthogonalize_sparse does exactly that with comm=comm-if-mpi).
    if reort_mode == Reort.FULL or (reort_mode == Reort.PERIODIC and it > 0 and it % reort_period == 0):
        wp, _, _ = apply_reort(wp, Q_basis, None, Reort.FULL, mpi, comm, block_widths or [], krylov)

    # --- 5. TSQR of the residual block ----------------------------------
    # The triangular factor is built from the block's own rows (panel Householder + Givens
    # merges + one Allgather), never from <wp|wp>, so it is backward stable at any
    # conditioning and the rank decision is made on the block's true singular values.
    # Breakdown is measured against the operator scale, not against 1: a residual block is
    # negligible when it is small compared to H. `h_norm_est` is the driver's running estimate
    # (0 on the first step of a cold start, where alpha_i carries the scale). Mirrors
    # block_lanczos_array_cy so both kernels deflate on the same criterion.
    q_next, beta_i, active_k, sv_i = block_tsqr(
        wp, mpi, comm, max(float(h_norm_est), float(np.linalg.norm(alpha_i, ord=2)))
    )
    if active_k < 0:
        # Non-finite factor => the recurrence is *corrupted*, not a genuine invariant
        # subspace. Signal it with active_k = -1 so the caller can report "diverged" (and
        # warn) instead of treating the truncated result as exact. active_k == 0 is a
        # real rank-deficient residual (the Krylov space is closed) => invariant subspace.
        return None, alpha_i, None, W, -1, True, False
    if active_k == 0:
        return None, alpha_i, None, W, 0, True, False
    _prof_acc("recurrence", _t0)

    # --- 5a. Forceful / Amplitude Truncation ----------------------------
    _t0 = _time.perf_counter()
    cdef bint did_truncate = False
    if slaterWeightMin > 0.0:
        # Whole-row prune: a row survives when ANY column survives (keeps the block's
        # shared support; the per-column prune of independent states would desync it).
        q_next.prune_rows(slaterWeightMin)
        did_truncate = True
    if truncation_threshold > 0:
        from impurityModel.ed.ManyBodyUtils import apply_global_truncation
        _q_states = q_next.to_states()
        for st in _q_states:
            apply_global_truncation(st, truncation_threshold, comm if mpi else None)
        q_next = ManyBodyBlockState.from_states(_q_states)
        did_truncate = True

    # Dropping amplitudes breaks the orthonormality the factorization just established, so
    # the truncated block has to be re-factored and the correction folded into beta_i:
    # wp = (q_next @ R2) @ beta_i. Unlike the CholeskyQR2 pass this replaces, no second pass
    # is needed for well-conditioned blocks — TSQR delivers orthonormality directly.
    if did_truncate:
        q_next, R2, active_k, _sv2 = block_tsqr(q_next, mpi, comm, 1.0)
        if active_k <= 0:
            return None, alpha_i, None, W, 0, True, False
        beta_i = R2 @ beta_i
    _prof_acc("tsqr", _t0)

    betas[it, :active_k, :p] = beta_i

    # --- 7. EA16 Selective Orthogonalization / Partial Reortho ---------
    _reort_acted = False
    if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
        _t0 = _time.perf_counter()
        if W is None:
            if start_it > 0:
                # Exact Overlap Restart (rare, resume-only): materialize the live blocks
                # once so the store slices (lists) meet lists in inner_multi.
                _wp_states = wp.to_states()
                _q_prev_states = q_prev.to_states()
                W = np.zeros((2, start_it + 1, alphas.shape[1], alphas.shape[1]), dtype=complex)
                for j in range(start_it):
                    w_j = block_widths[j]
                    Q_j = Q_basis[sum(block_widths[:j]) : sum(block_widths[:j+1])]
                    W[1, j, :w_j, :p] = inner_multi(Q_j, _wp_states)
                    if j < start_it - 1:
                        W[0, j, :w_j, :n_prev] = inner_multi(Q_j, _q_prev_states)
                if mpi and comm is not None:
                    comm.Allreduce(MPI.IN_PLACE, W, op=MPI.SUM)
                W[1, start_it, :p, :p] = np.eye(p)
                W[0, start_it - 1, :n_prev, :n_prev] = np.eye(n_prev)
            else:
                W = np.zeros((2, 1, p, p), dtype=complex)
                W[1, 0] = np.eye(p)

        W = estimate_orthonormality(
            W,
            alphas[: it + 1],
            betas[: it + 1],
            block_widths=block_widths + [p, active_k],
            eps=EPS,
            N=float(getattr(basis, "size", 0) or 1),
            out=w_out,
            beta_norms=beta_norm_hist,
        )
        _prof_acc("w_estimate", _t0)
        _t0 = _time.perf_counter()

        reort_eps = REORT_TOL

        if reort_mode == Reort.SELECTIVE and it > 0 and it % reort_period == 0:
            # EA16 §2.6.2 selective orthogonalization (shared with block_lanczos_array_cy).
            # beta_i's 2-norm is the Ritz residual scale; the driver has not computed it yet at
            # this point, so pass it explicitly. The cadence gate is replicated here (the
            # function gates internally too) so the block<->list boundary conversion only
            # happens on the steps where the Ritz check actually runs.
            q_next = ManyBodyBlockState.from_states(
                selective_orthogonalize(
                    q_next.to_states(), Q_basis, alphas, betas, W, block_widths,
                    it, p, np.linalg.norm(beta_i, ord=2), reort_eps, reort_period, mpi, comm,
                )
            )

        if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
            # Bad-block partial reorthogonalization via the shared apply_reort (single
            # implementation for both kernels). Pass block_widths + [p] so the current
            # block (index it) is included in the width table apply_reort indexes.
            # force_reort (set by the driver on the step after an acted one) bypasses
            # the trigger gate — the two-consecutive-steps rule, see apply_reort.
            q_next, W, _reort_acted = apply_reort(
                q_next, Q_basis, W, reort_mode, mpi, comm, block_widths + [p], krylov, force=force_reort
            )
            if _PROF_ON:
                _PROF["reort_total#n"] = _PROF.get("reort_total#n", 0.0) + 1.0
                if _reort_acted:
                    _PROF["reort_acted#n"] = _PROF.get("reort_acted#n", 0.0) + 1.0
            # Only when a bad block was actually projected does q_next need re-orthonormalizing;
            # otherwise it is unchanged and this would be an exact no-op (R2 == I), so skip the
            # factorization entirely. Mirrors block_lanczos_array_cy.
            if _reort_acted:
                q_next_2, R2, active_k, sv2 = block_tsqr(q_next, mpi, comm, 1.0)
                # Absolutely tiny residual after projection => block contained in the existing span
                # (invariant subspace); renormalizing it would amplify rounding. Treat as breakdown.
                # sqrt(EPS) is the largest column norm the old max(diag(<q|q>)) < EPS test admitted.
                if active_k <= 0 or float(sv2[0]) < np.sqrt(EPS):
                    return None, alpha_i, None, W, 0, True, False
                beta_i = R2 @ beta_i
                q_next = q_next_2
                # The renormalization q_next <- q_next @ R2^{-1} rescales its true overlaps
                # with every Krylov column by up to ||R2^{-1}||_2 = 1/sigma_min; propagate
                # the same bound into the just-written honest post-reort W estimates
                # (a no-op when the projection removed little: R2 ~ I).
                W[1, : W.shape[1] - 1] *= 1.0 / float(sv2[active_k - 1])
                betas[it, :active_k, :p] = beta_i
        _prof_acc("reort", _t0)

    return q_next, alpha_i, beta_i, W, active_k, False, _reort_acted


def block_lanczos_cy(
    psi0,
    h_op,
    basis,
    converged_fn,
    verbose: bool = False,
    reort="full",
    max_iter=None,
    slaterWeightMin: float = 0.0,
    truncation_threshold: int = 0,
    comm=None,
    reort_period: int = 5,
    alphas_init=None,
    betas_init=None,
    Q_init=None,
    W_init=None,
    return_widths=False,
    return_status=False,
    block_widths_init=None,
    locked=None,
    locked_evals=None,
    locked_res=0.0,
    locked_reort="full",
    store_krylov=True,
    krylov_dtype=None,
):
    """Run the distributed block Lanczos iteration with ``ManyBodyState``.

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

    **Pre-allocation**: ``alphas_buf`` and ``betas_buf`` of shape
    ``(max_iter, p, p)`` are allocated *once* before the main loop so that no
    NumPy allocation occurs inside the Lanczos body.  The convergence check and the
    returned arrays are *views* into these buffers (``alphas_buf[:it_abs+1]``) — no
    per-iteration list rebuild or ``np.array()`` copy.

    **Warm-start / resume protocol**: if ``alphas_init``, ``betas_init``, and
    ``Q_init`` are all provided the iteration resumes from block
    ``len(alphas_init)``.  The warm-start Q blocks are extracted from
    ``Q_init`` and the existing W-estimator (``W_init``) is reused.  If
    ``W_init`` is ``None`` but ``reort`` is ``'partial'`` or ``'selective'``,
    the Paige-Simon estimator array ``W`` is exactly initialized by computing the
    exact overlaps of the starting blocks against all prior blocks (Exact
    Overlap Restart - EOR).  Pass ``None`` for all three to start a fresh run.

    **MPI collective operations** (called every iteration, all ranks must participate):

    * One ``MPI_Allreduce`` (``MPI.SUM``) for :math:`\\alpha_i` – shape
      ``(p, p)``; result is replicated on all ranks.
    * One ``MPI_Allreduce`` (``MPI.SUM``) for :math:`M = W_p^\\dagger W_p` –
      shape ``(p, p)``; used for Cholesky QR.
    * Additional ``MPI_Allreduce`` calls inside reorthogonalization (see
      ``block_lanczos_step_cy`` for details).

    Note:
        ``converged_fn`` is called on **all ranks** with *replicated* data
        (``alphas``, ``betas`` are identical on every rank).  If the callable
        performs rank-dependent logic the caller is responsible for broadcasting
        the result.  When ``reort == Reort.NONE`` no reorthogonalization of any
        kind is performed; orthogonality degrades at the rate of floating-point
        rounding errors.

    Args:
        psi0: Initial block of ``p`` ``ManyBodyState`` objects, or ``None``
            when resuming from ``Q_init`` (warm-start mode).
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
        reort: Reorthogonalization mode.  Accepts ``Reort`` enum members or the
            equivalent strings ``'none'``, ``'partial'``, ``'selective'``,
            ``'full'``, ``'periodic'``.  ``Reort.NONE`` disables all
            reorthogonalization.  Default ``'full'``.
        max_iter: Maximum number of Lanczos steps before returning.  Defaults to
            ``basis.size // p`` when ``basis.size`` is available, otherwise
            ``10 * p``.
        slaterWeightMin: Amplitude cutoff passed to ``ManyBodyState.prune``.
            Slater determinants with :math:`|c_k| < \\text{slaterWeightMin}` are
            dropped after each matvec.  Default ``0.0`` (no pruning).
        comm: ``mpi4py`` communicator.  If ``None``, falls back to ``basis.comm``
            and then to serial mode (``mpi=False``).
        reort_period: Period for ``Reort.PERIODIC`` mode.  Full
            reorthogonalization is applied at steps that satisfy
            ``it > 0 and it % reort_period == 0``.  Default ``5``.
        alphas_init: Warm-start diagonal block array of shape ``(k0, p, p)``.
            If provided together with ``betas_init`` and ``Q_init``, iteration
            resumes from block ``k0``.  Default ``None``.
        betas_init: Warm-start off-diagonal block array of shape ``(k0, p, p)``.
            Default ``None``.
        Q_init: Warm-start Krylov basis: the ``SparseKrylovDense`` store returned
            by a previous run (adopted as-is, zero-copy resume) or a legacy flat
            list of ``ManyBodyState`` (length ``>= k0 * p``, ingested into a fresh
            store).  Default ``None``.
        W_init: Warm-start Paige-Simon W-estimator array.  Passed through to
            the step function; ``None`` causes exact initialisation via EOR
            when resuming with partial/selective reorthogonalization.  Default
            ``None``.
        store_krylov: When ``False`` (requires ``reort == 'none'`` and no
            ``locked`` set) the accumulated Krylov basis is *not* retained;
            the returned ``Q_basis`` holds only the last two blocks (previous
            block + residual) — exactly what the warm-start protocol reads.
            The alphas/betas are bit-identical to ``store_krylov=True``; only
            the O(N_det * p * k) dead retention is dropped.  On resume,
            ``Q_init`` is interpreted as that two-block tail (split by
            ``block_widths_init[-1]``).  Default ``True``.
        krylov_dtype: Storage dtype of the retained Krylov basis --
            ``complex128`` (default) or ``complex64``, which halves the store.
            A ``complex64`` basis can only be projected against to ~6e-8, i.e.
            *above* the ``REORT_TOL = sqrt(EPS) ~ 1.5e-8`` semi-orthogonality
            target that ``PARTIAL``/``SELECTIVE`` steer to, so those modes reject
            it: the target is unreachable, and their ``BAD_BLOCK_TOL ~ 1.8e-12``
            block selection sits five orders below the fp32 noise floor, so every
            block would be flagged.  ``FULL``/``PERIODIC`` accept it: they hold no estimator and
            simply settle at orthogonality ~6e-8, which perturbs the
            block-tridiagonal ``T`` by O(6e-8**2 * ||H||) -- far below the
            Green's-function broadening.  The live recurrence blocks, the
            overlaps and the residual all stay complex128; only the *stored*
            basis narrows.  Default ``None`` (complex128).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, list, numpy.ndarray]: A 4-tuple
        ``(alphas, betas, Q_basis, W)`` where

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
        tail; NOT exact) or ``"max_iter"`` (the ``max_iter`` budget was exhausted before
        the recurrence terminated naturally). Both flags are independent and opt-in, so
        existing call sites that pass neither keep the 4-tuple return.
    """
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    if isinstance(h_op, dict):
        h_op = ManyBodyOperator(h_op)

    # --- Resolve communicator / MPI flag --------------------------------
    if comm is None:
        comm = getattr(basis, "comm", None)
    mpi = comm is not None and comm.Get_size() > 1

    # --- Resolve reort mode ---------------------------------------------
    reort_mode = resolve_reort(reort)

    # Tail-only mode: with reort == NONE (and no locked set) nothing in the recurrence
    # ever projects against the accumulated Krylov basis, and the warm-start protocol
    # reads only the last two blocks -- so retaining the full O(N_det * p * k) basis is
    # pure dead memory (the dominant avoidable allocation in a Green's-function run).
    if not store_krylov and (reort_mode != Reort.NONE or locked):
        raise ValueError("store_krylov=False requires reort='none' and no locked vectors")

    # A complex64 store represents each Krylov vector to ~u32 = 6e-8, so a projection against
    # it cannot drive the residual overlap against the TRUE Krylov space below that (measured:
    # FULL + complex64 settles at ||Q^H Q - I|| = 6.0e-8, vs 1.1e-15 at complex128). But
    # PARTIAL/SELECTIVE steer to REORT_TOL = sqrt(EPS) ~ 1.5e-8, four times tighter: the target
    # is unreachable, so their control loop has nothing to converge to. Worse, once the trigger
    # fires they select blocks with BAD_BLOCK_TOL = EPS**0.75 ~ 1.8e-12 -- five orders below the
    # fp32 noise floor -- so every block is flagged and PARTIAL degenerates into FULL while
    # delivering worse orthogonality than FULL at complex128. Paying FULL's cost for a worse
    # answer is strictly dominated, so reject rather than warn.
    #
    # The estimator's own reading of the situation is regime dependent and was NOT measured
    # end to end (this guard fires first): O_last tracks the true residual within ~1.5x when
    # there is a real projection to do, but is measured against the *stored* (rounded) basis,
    # so on a near-no-op step it reads rounding-level while the true loss sits at ~u32. That is
    # the under-prediction failure mode that has cost production runs before (see the "HONEST
    # reset" note in BlockLanczosArray.apply_reort). Either way the combination is unusable.
    if krylov_dtype is not None and np.dtype(krylov_dtype) == np.dtype(np.complex64):
        if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
            raise ValueError(
                "krylov_dtype='complex64' is incompatible with reort='partial'/'selective': "
                "the stored basis is only accurate to ~6e-8, above the sqrt(EPS) ~ 1.5e-8 "
                "semi-orthogonality target these modes maintain. Use reort='full'/'periodic' "
                "(no Paige-Simon estimator) or keep complex128."
            )

    # --- Resume or start fresh? -----------------------------------------
    resuming = alphas_init is not None and betas_init is not None and Q_init is not None

    cdef list block_widths = list(block_widths_init) if block_widths_init is not None else []
    if resuming:
        start_it = len(alphas_init)
        p = alphas_init[0].shape[0] if len(alphas_init) > 0 else len(Q_init[0] if Q_init else psi0)
        if len(block_widths) == 0:
            block_widths = [p] * start_it
        W = W_init

        if not store_krylov:
            # Q_init is the two-block tail [prev block, residual block] returned by the
            # previous tail-only run; split it by the true width of the last accepted
            # block (deflation-safe: block_widths is authoritative, the tail may be
            # narrower than p).
            Q_basis = []
            if start_it == 0:
                q_prev = ManyBodyBlockState.from_states([ManyBodyState() for _ in range(p)])
                q_curr = ManyBodyBlockState.from_states(list(Q_init))
            else:
                q_prev_len = block_widths[start_it - 1]
                q_prev = ManyBodyBlockState.from_states(list(Q_init[:q_prev_len]))
                q_curr = ManyBodyBlockState.from_states(list(Q_init[q_prev_len:]))
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
                if len(Q_init) > 0:
                    Q_basis.append(list(Q_init))
            if start_it == 0 or len(Q_basis) < p:
                q_prev = ManyBodyBlockState.from_states([ManyBodyState() for _ in range(p)])
                q_curr = ManyBodyBlockState.from_states(Q_basis[0:p])
            else:
                q_prev_start = sum(block_widths[:start_it - 1])
                q_prev_len = block_widths[start_it - 1]
                q_curr_start = sum(block_widths[:start_it])
                q_prev = ManyBodyBlockState.from_states(Q_basis[q_prev_start : q_prev_start + q_prev_len])
                q_curr = ManyBodyBlockState.from_states(Q_basis[q_curr_start : len(Q_basis)])
    else:
        start_it = 0
        p = len(psi0)
        W = None

        # Redistribute initial states across ranks, then adopt the shared-support block
        # representation for the live recurrence blocks (Phase 2.4).
        q_curr = ManyBodyBlockState.from_states(basis.redistribute_psis(list(psi0)))
        q_prev = ManyBodyBlockState.from_states([ManyBodyState() for _ in range(p)])
        if store_krylov:
            Q_basis = SparseKrylovDense(krylov_dtype)
            _local_basis = getattr(basis, "local_basis", None)
            if _local_basis is not None:
                Q_basis.reserve_rows(len(_local_basis))
            Q_basis.append_block(q_curr)
        else:
            Q_basis = []

    # The store doubles as the dense reort mirror for every mode (apply_reort slices its
    # columns via store.reort); the old FULL/PERIODIC-only mirror and the PARTIAL-mode
    # transient reorth_cgs2_dense materialization are both subsumed by it.
    krylov = Q_basis if store_krylov else None

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
    # Seed W-estimator state
    if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE) and W is None:
        if start_it > 0:
            W = np.zeros((2, start_it + 1, p, p), dtype=complex)
            for j in range(start_it):
                w_j = block_widths[j]
                Q_j = Q_basis[sum(block_widths[:j]) : sum(block_widths[:j+1])]

                # q_curr/q_prev are blocks; inner_multi materializes them (resume-only path)
                ov_curr = inner_multi(Q_j, q_curr)
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, ov_curr, op=MPI.SUM)
                W[1, j, :w_j, :q_curr.width] = ov_curr

                if j < start_it - 1:
                    ov_prev = inner_multi(Q_j, q_prev)
                    if mpi:
                        comm.Allreduce(MPI.IN_PLACE, ov_prev, op=MPI.SUM)
                    W[0, j, :w_j, :q_prev.width] = ov_prev

            W[1, start_it, :q_curr.width, :q_curr.width] = np.eye(q_curr.width)
            W[0, start_it - 1, :q_prev.width, :q_prev.width] = np.eye(q_prev.width)
        else:
            W = np.zeros((2, 1, p, p), dtype=complex)
            W[1, 0] = np.eye(p)
    elif reort_mode in (Reort.PARTIAL, Reort.SELECTIVE) and W is not None:
        pass  # W expands dynamically in estimate_orthonormality

    # Bounded-W ping-pong buffers + the beta-norm history (Phase 1): the estimator
    # writes each step's W into the spare persistent buffer instead of allocating,
    # and reuses ||beta_j||_2 of completed blocks instead of re-factorizing them
    # every step. Resume prefixes the history with None placeholders (block index
    # aligned); the estimator falls back to computing those on demand.
    _w_bufs = None
    beta_norm_hist = None
    if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
        _w_bufs = (
            np.empty((2, start_it + _buf_size + 2, p, p), dtype=complex),
            np.empty((2, start_it + _buf_size + 2, p, p), dtype=complex),
        )
        beta_norm_hist = [None] * start_it

    # Estimate-driven locking reorthogonalization (EA16 §2.6.2) state. Only active when a
    # locked set is supplied and locked_reort == "partial"; otherwise the step does the
    # unconditional "full" projection. See ea16.locked_overlap_step for the recurrence.
    partial_locked = bool(locked) and locked_reort == "partial"
    nlock = len(locked) if locked else 0
    if partial_locked:
        from impurityModel.ed import ea16 as _ea16
        locked_evals_arr = np.ascontiguousarray(np.real(np.asarray(locked_evals)), dtype=float)
        _Ntot = getattr(basis, "size", 0) or 1
        omega_min_l = EPS * p * math.sqrt(_Ntot)
        xi_l = np.full(nlock, omega_min_l)
        xi_prev_l = np.full(nlock, omega_min_l)
        locked_rho = float(locked_res)

    # --- Main Lanczos loop ----------------------------------------------
    it = 0
    breakdown = False
    # Why the recurrence stops (returned when return_status=True). Default assumes the loop
    # runs out the max_iter budget without terminating naturally; the branches below overwrite
    # it. "invariant_subspace" => the block-Krylov space is closed under H (within the active
    # restrictions) and the result is *exact*; "diverged" => the divergence guard truncated a
    # corrupted tail (NOT exact); "converged" => converged_fn was satisfied.
    termination = "max_iter"
    # Largest healthy block-norm seen so far (proxy for ||H|| on the subspace); used to
    # detect a runaway recurrence that the deflation/QR safeguards did not catch.
    t_norm_max = 0.0
    # Spectral-scale estimate (seeded from beta_0 ~ ||H||, grown only by ||alpha_i||, which is
    # bounded by ||H|| and never runs away) — catches gradual beta growth the relative check misses.
    h_norm_est = 0.0

    # Two-consecutive-steps reort rule (Simon/PROPACK): when a step acts, the next step
    # bypasses the REORT_TOL gate (see apply_reort force=) so q_curr's remaining overlap
    # cannot silently re-contaminate the recurrence through the three-term coupling.
    _force_reort = False

    while it < _buf_size:
        it_abs = start_it + it
        n_curr = q_curr.width
        q_next, alpha_i, beta_i, W, active_k, breakdown, _step_acted = block_lanczos_step_cy(
            h_op=h_op,
            q_prev=q_prev,
            q_curr=q_curr,
            Q_basis=Q_basis,
            alphas=alphas_buf,
            betas=betas_buf,
            it=it_abs,
            reort_mode=reort_mode,
            W=W,
            mpi=mpi,
            comm=comm,
            basis=basis,
            slaterWeightMin=slaterWeightMin,
            truncation_threshold=truncation_threshold,
            reort_period=reort_period,
            start_it=start_it,
            block_widths=block_widths,
            locked=locked,
            locked_reort=locked_reort,
            krylov=krylov,
            w_out=_w_bufs[it % 2] if _w_bufs is not None else None,
            beta_norm_hist=beta_norm_hist,
            force_reort=_force_reort,
            h_norm_est=max(h_norm_est, t_norm_max),
        )
        _force_reort = _step_acted

        if breakdown:
            # active_k < 0 marks a non-finite (corrupted) Gram matrix -> truncated, NOT exact;
            # active_k == 0 is a genuine rank-deficient residual -> the block-Krylov space is
            # closed under H -> the continued fraction is exact on it.
            termination = "diverged" if active_k < 0 else "invariant_subspace"
            if verbose:
                if comm is None or comm.Get_rank() == 0:
                    print(f"[BlockLanczos] {termination} detected at iteration {it_abs}.")
            block_widths.append(n_curr)
            it += 1
            break

        # --- Divergence safeguard ---------------------------------------
        # For a Hermitian H every block norm is bounded by ||H||, so ||alpha_i||/||beta_i||
        # can never exceed the largest healthy block norm by more than rounding. A jump of
        # several orders of magnitude means the recurrence has been corrupted (lost
        # orthogonality the QR/deflation did not repair). Truncate *before* the offending
        # block — exclude alpha_i/beta_i at it_abs — so the returned T-matrix is the last
        # numerically trustworthy subspace and the diverged tail never reaches the Green's
        # function. The trailing beta then plays the role of the (ignored) residual coupling.
        # One SVD of beta_i per step: its largest singular value is the 2-norm (guard/verbose)
        # and its smallest gives ||beta_i^-1|| for the locked-reort estimate below.
        _svb = np.linalg.svd(beta_i, compute_uv=False)
        beta_norm = float(_svb[0])
        if beta_norm_hist is not None:
            beta_norm_hist.append(beta_norm)
        alpha_norm = np.linalg.norm(alpha_i, ord=2)
        diverged, t_norm_max, h_norm_est = divergence_guard(
            beta_norm, alpha_norm, it == 0, t_norm_max, h_norm_est
        )
        if diverged:
            if verbose and (comm is None or comm.Get_rank() == 0):
                print(
                    f"[BlockLanczos] Divergence detected at iteration {it_abs}: "
                    f"|beta|={beta_norm:.3e}, |alpha|={alpha_norm:.3e} >> spectral scale "
                    f"{h_norm_est:.3e}. Truncating to the last trustworthy block."
                )
            termination = "diverged"
            breakdown = True
            break

        # EA16 §2.6.2 estimate-driven locking reorth: propagate the per-pair overlap
        # estimate (cheap, no O(N) work) and reorthogonalize q_next only against the
        # locked vectors whose estimate now exceeds omega_TOL.
        if partial_locked:
            bj_inv_norm = 1.0 / max(float(_svb[len(_svb) - 1]), EPS)  # reuse the step's beta_i SVD
            bjm1_norm = float(np.linalg.norm(betas_buf[it_abs - 1], 2)) if it_abs > 0 else 0.0
            xi_new_l, xi_trigger, xi_mask = _ea16.locked_overlap_step(
                xi_l, xi_prev_l, locked_evals_arr, alpha_i,
                bj_inv_norm, bjm1_norm, locked_rho, REORT_TOL, BAD_BLOCK_TOL, EPS,
            )
            if xi_trigger:
                idx_l = np.nonzero(xi_mask)[0]
                Lm_blk = ManyBodyBlockState.from_states([locked[int(t)] for t in idx_l])
                for _ in range(2):
                    _lovl = block_inner_cy(Lm_blk, q_next)
                    if mpi and comm is not None:
                        comm.Allreduce(MPI.IN_PLACE, _lovl, op=MPI.SUM)
                    q_next = block_add_scaled_cy(q_next, Lm_blk, -_lovl)
                xi_new_l[idx_l] = omega_min_l
                xi_l[idx_l] = omega_min_l
            xi_prev_l = xi_l
            xi_l = xi_new_l

        # Append q_next to the basis (last block = residual direction). The locking
        # deflation is applied inside block_lanczos_step_cy (before M/beta) in "full"
        # mode, or just above in "partial" mode, so q_next is orthogonal to the locked set.
        # Tail-only mode keeps nothing here: q_prev/q_curr roll below and the two-block
        # tail is assembled at return.
        if store_krylov:
            # append_block copies the coefficients into the dense buffer straight from
            # the block rows — no per-state materialization, no defensive copies.
            Q_basis.append_block(q_next)

        if verbose:
            print(
                f"[BlockLanczos] it={it_abs:4d}  " f"|beta|={beta_norm:.3e}  " f"alpha_diag={np.real(np.diag(alpha_i))}"
            )

        # Convergence check
        alphas_np = alphas_buf[:it_abs+1]
        betas_np = betas_buf[:it_abs+1]
        _t0m = _time.perf_counter()
        _converged = converged_fn(alphas_np, betas_np, verbose=verbose, block_widths=block_widths + [n_curr])
        _prof_acc("monitor", _t0m)
        if _converged:
            termination = "converged"
            block_widths.append(n_curr)
            it += 1
            break

        q_prev = q_curr
        q_curr = q_next
        block_widths.append(n_curr)
        it += 1

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
    if return_widths and return_status:
        return alphas_out, betas_out, Q_basis, W, block_widths, termination
    if return_widths:
        return alphas_out, betas_out, Q_basis, W, block_widths
    if return_status:
        return alphas_out, betas_out, Q_basis, W, termination
    return alphas_out, betas_out, Q_basis, W
