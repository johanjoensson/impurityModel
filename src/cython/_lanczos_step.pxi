# ===========================================================================
# Core block-Lanczos recurrence
# ===========================================================================
# One three-term block step (block_lanczos_step_cy) and the full recurrence driver
# (block_lanczos_cy) that serves the whole frequency mesh from a single recurrence. The
# reorthogonalization mode (NONE/PARTIAL/FULL) and the EA16
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
    start_it: int = 0,
    block_widths=None,
    locked=None,
    locked_reort="full",
    krylov=None,
    w_out=None,
    beta_norm_hist=None,
    force_reort: bool = False,
    h_norm_est: float = 0.0,
    deflate_tol: float = -1.0,
):
    """Perform one step of the distributed block Lanczos iteration.

    Executes the three-term block recurrence

    .. math::

        W_p = H Q_i - Q_i \\alpha_i - Q_{i-1} \\beta_{i-1}^\\dagger

    where :math:`\\alpha_i = Q_i^\\dagger H Q_i` is the diagonal block and
    :math:`\\beta_{i-1}` is the off-diagonal block from the previous step.
    The residual block :math:`W_p = Q_{i+1}\\beta_i` is then QR-factorised by TSQR
    (``block_tsqr``), which builds :math:`\\beta_i` from :math:`W_p` itself and obtains
    :math:`Q_{i+1} = W_p\\beta_i^{-1}` by back substitution — the Gram matrix
    :math:`W_p^\\dagger W_p` is never formed.

    **MPI collective operations** (all ranks must call simultaneously):

    1. ``MPI_Allreduce`` (``MPI.SUM``) on :math:`\\alpha_i` – :math:`p \\times p`
       complex matrix; result is replicated on every rank.
    2. ``MPI_Allgather`` of the rank-local packed triangular factor
       (:math:`p(p+1)/2` complex) inside the TSQR; every rank merges them in rank order,
       so the resulting :math:`\\beta_i`, the retained rank and the breakdown decision are
       identical on all ranks.
    3. For ``FULL`` mode, an additional ``MPI_Allreduce`` per
       reorthogonalization pass on the :math:`(it \\cdot p) \\times p` overlap matrix.
    4. For ``PARTIAL``, standard Paige-Simon tracking operates against all Lanczos vectors,
       triggering ``MPI_Allreduce`` calls over potentially large swaths of the Krylov
       basis when orthogonality loss exceeds :math:`\\sqrt{\\varepsilon_{\\text{mach}}}`.

    Note:
        ``h_op.apply_multi`` releases the GIL internally; all other operations in
        this function hold the GIL.

    Args:
        h_op: ``ManyBodyOperator`` Hamiltonian; must implement
            ``apply_multi(psis, cutoff)``.
        q_prev: ``ManyBodyState`` of width ``p`` from iteration ``i-1``
            (a zero-row block of width ``p`` at ``it=0``).
        q_curr: ``ManyBodyState`` of width ``p`` from iteration ``i``.
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
            ``PARTIAL`` mode; ignored otherwise.
        mpi: ``True`` when running under MPI with more than one rank.
        comm: Active ``mpi4py.MPI.Comm`` communicator, or ``None`` when running
            serially.
        slaterWeightMin: Amplitude cutoff passed to ``ManyBodyState.prune``;
            SD coefficients below this value are dropped.  Default ``0.0``
            (no pruning).  On ``H q`` the cut is made on each row's summed amplitude
            when the step redistributes.

    Returns:
        tuple: A 7-tuple ``(q_next, alpha_i, beta_i, W_updated, active_k, breakdown,
        reort_acted)``; ``reort_acted`` is True iff the PARTIAL bad-block
        projection actually fired this step (the driver then forces a re-check on the
        next step — the two-consecutive-steps rule):

        * ``q_next`` – ``ManyBodyState`` (width ``active_k``) forming the next
          Krylov block :math:`Q_{i+1}`, or ``None`` if breakdown occurred.
        * ``alpha_i`` – numpy complex array of shape ``(p, p)`` for the current
          diagonal block.
        * ``beta_i`` – numpy complex array of shape ``(p, p)`` for the current
          off-diagonal block, or ``None`` if breakdown occurred.
        * ``W_updated`` – updated Paige-Simon estimator array, or ``None`` if
          not applicable.
        * ``breakdown`` – ``True`` if the residual block was numerically zero against the
          operator scale (an invariant subspace, ``active_k == 0``) or its factor came out
          non-finite (a corrupted recurrence, ``active_k == -1``).
    """
    # q_prev / q_curr are shared-support ManyBodyStates: the matvec, Gram products and
    # axpy updates below run once per determinant ROW instead of once per (determinant,
    # vector) pair; pruning keeps whole rows (any-column-survives) instead of
    # per-column entries.
    p = q_curr.width

    # --- 1. Block matvec: wp = H q_curr ---------------------------------
    # A growth-capping basis proxy (caps_growth=True) must see every step's residual
    # even serially: its redistribute_block enforces the truncation_threshold there
    # (the inner Basis redistribute no-ops without MPI).
    _needs_redistribute = basis is not None and ((mpi and comm is not None) or getattr(basis, "caps_growth", False))
    # Chunking only where a redistribute actually follows. `GF_APPLY_ROW_CHUNKS` defaults to 4;
    # `1` is the one-shot path. The matvec itself (chunking, the cut on summed rows, a capping
    # proxy's deferred admission) is `apply_and_redistribute`.
    _n_chunks = config.GF_APPLY_ROW_CHUNKS.get() if _needs_redistribute else 1
    wp = apply_and_redistribute(
        h_op, q_curr, basis, slaterWeightMin, _needs_redistribute, 1 if _n_chunks is None else int(_n_chunks),
    )
    # A proxy running the support freeze also weighs what the next block's prune below drops (replicated flag).
    _note_cut = (
        basis.note_step_cut
        if _needs_redistribute and hasattr(basis, "begin_step") and getattr(basis, "support_freeze", False)
        else None
    )

    # --- 2. alpha_i = <q_curr | wp> -------------------------------------
    alpha_i = block_inner_cy(q_curr, wp)
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, alpha_i, op=MPI.SUM)
    alphas[it, :p, :p] = alpha_i

    # --- 3. Subtract: wp = wp - q_curr * alpha_i - q_prev * beta_{i-1}^† -
    wp = block_add_scaled_cy(wp, q_curr, -alpha_i)
    # Bound unconditionally: q_prev is always a valid width-p block (a zero placeholder
    # on the very first step of a fresh start), so this is well-defined at every call.
    # Binding it only under "if it > 0" relied on that guard coinciding with the read
    # site's own "if start_it > 0" below -- true today only because both trace back to
    # the same resume/fresh-start decision, not because the two conditions are the same
    # test. See doc/lanczos_invariants.md.
    n_prev = q_prev.width
    if it > 0:
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
    #
    # Gate on block_cols(locked), never truthiness: ManyBodyState.__len__ is the rank-local
    # row count, so an empty rank diverges and the Allreduce below deadlocks. See
    # doc/lanczos_invariants.md ("MPI collective gating") for the full incident.
    if locked is not None and block_cols(locked) > 0 and locked_reort != "partial":
        locked_blk = (
            locked if isinstance(locked, ManyBodyState) else ManyBodyState.from_states(list(locked))
        )
        for _ in range(2):
            _ovl = block_inner_cy(locked_blk, wp)
            if mpi and comm is not None:
                comm.Allreduce(MPI.IN_PLACE, _ovl, op=MPI.SUM)
            wp = block_add_scaled_cy(wp, locked_blk, -_ovl)

    # --- 4. Full reorthogonalization ------------------------------------
    # The reort action goes through the shared apply_reort (classical 2-pass block
    # Gram-Schmidt, one implementation for both kernels).
    if reort_mode == Reort.FULL:
        wp, _, _ = apply_reort(wp, Q_basis, None, Reort.FULL, mpi, comm, block_widths or [], krylov)

    # --- 5. TSQR of the residual block ----------------------------------
    # The triangular factor is built from the block's own rows (panel Householder + Givens
    # merges + one Allgather), never from <wp|wp>, so it is backward stable at any
    # conditioning and the rank decision is made on the block's true singular values.
    # Breakdown is measured against the operator scale, not against 1: a residual block is
    # negligible when it is small compared to H. `h_norm_est` is the driver's running estimate
    # (0 on the first step of a cold start, where alpha_i carries the scale).
    q_next, beta_i, active_k, sv_i = factor_residual(
        wp, mpi, comm, max(float(h_norm_est), spectral_norm(alpha_i)),
        deflate_tol=deflate_tol,
    )
    if active_k < 0:
        # Non-finite factor => the recurrence is *corrupted*, not a genuine invariant
        # subspace. Signal it with active_k = -1 so the caller can report "diverged" (and
        # warn) instead of treating the truncated result as exact. active_k == 0 is a
        # real rank-deficient residual (the Krylov space is closed) => invariant subspace.
        return None, alpha_i, None, W, -1, True, False
    if active_k == 0:
        return None, alpha_i, None, W, 0, True, False

    # --- 5a. Forceful / Amplitude Truncation ----------------------------
    cdef bint did_truncate = False
    if slaterWeightMin > 0.0:
        # Whole-row prune: a row survives when ANY column survives (keeps the block's
        # shared support; the per-column prune of independent states would desync it).
        if _note_cut is not None:
            # In the matvec's units: the residual is q_next @ beta_i, so a dropped row of q_next carried
            # row beta_i beta_i^H row^H of it. Reaches the proxy with the next step's admission.
            _note_cut(q_next.prune_rows(slaterWeightMin, beta_i @ np.conj(beta_i.T)))
        else:
            q_next.prune_rows(slaterWeightMin)
        did_truncate = True
    if truncation_threshold > 0:
        _q_states = q_next.to_states()
        for st in _q_states:
            _keep_global_top(st, truncation_threshold, comm if mpi else None)
        q_next = ManyBodyState.from_states(_q_states)
        did_truncate = True

    # Dropping amplitudes breaks the orthonormality the factorization just established, so
    # the truncated block has to be re-factored and the correction folded into beta_i:
    # wp = (q_next @ R2) @ beta_i. Unlike the CholeskyQR2 pass this replaces, no second pass
    # is needed for well-conditioned blocks — TSQR delivers orthonormality directly.
    if did_truncate:
        q_next, R2, active_k, _sv2 = block_tsqr(q_next, mpi, comm, 1.0, deflate_tol=deflate_tol)
        if active_k <= 0:
            return None, alpha_i, None, W, 0, True, False
        beta_i = R2 @ beta_i

    betas[it, :active_k, :p] = beta_i

    # --- 6. W estimator (Paige-Simon PRO) -------------------------------
    _reort_acted = False
    if reort_mode == Reort.PARTIAL:
        if W is None:
            # Unreachable from block_lanczos_cy: the driver seeds W (Exact Overlap
            # Restart, seed_w_estimator) before entering the loop whenever reort is
            # PARTIAL, so W is never None here. This branch used to carry a
            # *third* inline copy of the EOR loop that no production path could reach,
            # and which seeded from a different block (the pre-TSQR residual) than the
            # driver's copy (q_curr) -- an asymmetry that never mattered precisely
            # because it was dead. Raise rather than silently seed under a second,
            # divergent convention: a direct caller of this step function owes it a
            # seeded W.
            raise ValueError(
                "block_lanczos_step_cy requires a seeded W for reort='partial'; "
                "callers other than block_lanczos_cy must seed it via "
                "BlockLanczosCore.seed_w_estimator first."
            )

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

        # --- 8. Bad-block reort + renormalize ----------------------------
        if reort_mode == Reort.PARTIAL:
            # Bad-block partial reorthogonalization + renormalize via the shared
            # finish_reort (single implementation for both kernels). Pass
            # block_widths + [p] so the current block (index it) is included in the
            # width table apply_reort indexes. force_reort (set by the driver on the
            # step after an acted one) bypasses the trigger gate — the
            # two-consecutive-steps rule, see apply_reort.
            q_next, W, beta_i, active_k, _reort_acted, _ok = finish_reort(
                q_next, Q_basis, W, beta_i, active_k, reort_mode, mpi, comm, block_widths + [p],
                krylov, force_reort, deflate_tol,
            )
            if _reort_acted and not _ok:
                # Absolutely tiny residual after projection => block contained in the existing
                # span (invariant subspace); renormalizing it would amplify rounding.
                return None, alpha_i, None, W, 0, True, False
            if _reort_acted:
                betas[it, :active_k, :p] = beta_i

    return q_next, alpha_i, beta_i, W, active_k, False, _reort_acted


def _pack_result(
    alphas_buf, betas_buf, start_it, it, store_krylov, termination,
    q_prev, q_curr, q_next, Q_basis, W, block_widths, return_widths, return_status,
):
    """Build block_lanczos_cy's return tuple: the finished alphas/betas views, the
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
    return pack_lanczos_result(alphas_out, betas_out, Q_basis, W, block_widths, termination, return_widths, return_status)


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
    alphas_init=None,
    betas_init=None,
    Q_init=None,
    W_init=None,
    return_widths=False,
    return_status=False,
    block_widths_init=None,
    locked=None,
    locked_evals=None,
    locked_res=REORT_TOL,
    locked_reort="full",
    store_krylov=True,
    krylov_dtype=None,
    deflate_tol=-1.0,
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

    ``alphas_buf``/``betas_buf`` are pre-allocated once; the returned arrays are
    *views* into them, not copies. Resumes from block ``len(alphas_init)`` when
    ``alphas_init``, ``betas_init`` and ``Q_init`` are all given (``None`` for all
    three starts fresh) — see "Warm-start / resume protocol" in
    doc/lanczos_invariants.md for the full protocol including the W-estimator's
    Exact Overlap Restart.

    **MPI collective operations** (called every iteration, all ranks must participate):

    * One ``MPI_Allreduce`` (``MPI.SUM``) for :math:`\\alpha_i` – shape
      ``(p, p)``; result is replicated on all ranks.
    * One ``MPI_Allgather`` of the packed triangular factor – ``p(p+1)/2``
      complex per rank; the TSQR of the residual block.
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
        reort: Reorthogonalization mode.  Accepts ``Reort`` enum members or the
            equivalent strings ``'none'``, ``'partial'``, ``'full'``.  ``Reort.NONE`` disables all
            reorthogonalization.  Default ``'full'``.
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
        W_init: Warm-start Paige-Simon W-estimator array.  Passed through to
            the step function; ``None`` causes exact initialisation via EOR
            when resuming with partial/selective reorthogonalization.  Default
            ``None``.
        store_krylov: When ``False`` (requires ``reort == 'none'`` and no
            ``locked`` set) the accumulated Krylov basis is *not* retained; see
            "Krylov store" in doc/lanczos_invariants.md.  Default ``True``.
        krylov_dtype: Storage dtype of the retained Krylov basis --
            ``complex128`` (default) or ``complex64``.  Rejected outright (not
            just warned) when combined with ``reort='partial'/'selective'`` --
            see "complex64 measurements" in doc/lanczos_invariants.md.  Default
            ``None`` (complex128).

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
        tail; NOT exact), ``"max_iter"`` (the ``max_iter`` budget was exhausted before
        the recurrence terminated naturally) or ``"frozen"`` (``basis`` is a growth-capping
        proxy with ``stop_on_freeze`` set and froze during the last step; the state is a
        ``"max_iter"`` stop's, so the recurrence can be resumed). Both flags are independent and opt-in, so
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
    #
    # `locked` gates this the same way as the width test below (nlock, hoisted here so
    # both this check and the partial_locked/xi setup share one rank-invariant value):
    # `if locked` on a ManyBodyState reads its LOCAL ROW count, which can disagree
    # across ranks and turn this `raise` into a rank-divergent collective itself.
    nlock = 0 if locked is None else block_cols(locked)
    if not store_krylov and (reort_mode != Reort.NONE or nlock > 0):
        raise ValueError("store_krylov=False requires reort='none' and no locked vectors")

    # complex64 storage (~6e-8 rounding) can never reach PARTIAL's REORT_TOL
    # target (~1.5e-8, sqrt(EPS)): the trigger is unreachable, and on the rare step it
    # does fire, BAD_BLOCK_TOL flags every block, degenerating to FULL at
    # worse-than-complex128 orthogonality. Reject rather than warn; see
    # doc/lanczos_invariants.md ("complex64 measurements") for the full measurement
    # table and the estimator under-prediction risk this guard also avoids.
    if krylov_dtype is not None and np.dtype(krylov_dtype) == np.dtype(np.complex64):
        if reort_mode == Reort.PARTIAL:
            raise ValueError(
                "krylov_dtype='complex64' is incompatible with reort='partial': "
                "the stored basis is only accurate to ~6e-8, above the sqrt(EPS) ~ 1.5e-8 "
                "semi-orthogonality target these modes maintain. Use reort='full' "
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
        W = None

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
    # Seed W-estimator state (Exact Overlap Restart, see seed_w_estimator). Unifies
    # this copy's previous per-j Allreduce pattern onto the one-bulk-Allreduce pattern
    # the array kernel already used -- Allreduce(SUM) reduces each element
    # independently, so batching cannot change a per-element result. Verified by
    # case_mbs_partial_eor_resume in the golden harness (added alongside this hoist:
    # the pre-existing warm-resume cases all pass W_init non-None and so never reach
    # this branch) plus test_block_lanczos_cy_resume_partial_reort_without_w_init.
    if reort_mode == Reort.PARTIAL and W is None:
        W = seed_w_estimator(Q_basis, q_curr, q_prev, block_widths, start_it, p, mpi, comm)
    elif reort_mode == Reort.PARTIAL and W is not None:
        pass  # W expands dynamically in estimate_orthonormality

    # Bounded-W ping-pong buffers + the beta-norm history: the estimator
    # writes each step's W into the spare persistent buffer instead of allocating,
    # and reuses ||beta_j||_2 of completed blocks instead of re-factorizing them
    # every step. Resume prefixes the history with None placeholders (block index
    # aligned); the estimator falls back to computing those on demand.
    _w_bufs = None
    beta_norm_hist = None
    if reort_mode == Reort.PARTIAL:
        _w_bufs = (
            np.empty((2, start_it + _buf_size + 2, p, p), dtype=complex),
            np.empty((2, start_it + _buf_size + 2, p, p), dtype=complex),
        )
        beta_norm_hist = [None] * start_it

    # Estimate-driven locking reorthogonalization (EA16 §2.6.2) state. Only active when a
    # locked set is supplied and locked_reort == "partial"; otherwise the step does the
    # unconditional "full" projection (block_lanczos_step_cy). See ea16.locked_overlap_step
    # for the recurrence. `nlock` (the rank-invariant column count) was hoisted above the
    # store_krylov guard; gate on it rather than `bool(locked)` -- a ManyBodyState defines
    # no __bool__, so `bool()`/`if locked` fall back to len(), its LOCAL ROW count, which
    # can be 0 on an empty rank (owns no determinants) while width > 0 elsewhere. That
    # mismatch let an empty rank skip the `_lovl` Allreduce below while other ranks
    # executed it -> deadlock at mpiexec -n 3 (test_block_lanczos_mbs_empty_rank.py).
    partial_locked = nlock > 0 and locked_reort == "partial"
    if partial_locked:
        locked_evals_arr = np.ascontiguousarray(np.real(np.asarray(locked_evals)), dtype=float)
        _Ntot = getattr(basis, "size", 0) or 1
        omega_min_l = omega_floor(p, _Ntot)
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
    # A resumed call (a doubled budget, a declined freeze) replays the guard over the blocks it was
    # handed, so it keeps the scale the earlier calls built: starting both trackers from 0 left
    # every resumed call's first step unguarded and let that step re-seed the scale -- every 16
    # blocks while capped GF units ran a window at a time.
    guard_seeded = False
    if resuming:
        for _k in range(min(len(alphas_init), len(betas_init))):
            _, t_norm_max, h_norm_est = divergence_guard(
                spectral_norm(betas_init[_k]), spectral_norm(alphas_init[_k]), _k == 0, t_norm_max, h_norm_est
            )
            guard_seeded = True

    # Two-consecutive-steps reort rule (Simon/PROPACK): when a step acts, the next step
    # bypasses the REORT_TOL gate (see apply_reort force=) so q_curr's remaining overlap
    # cannot silently re-contaminate the recurrence through the three-term coupling.
    _force_reort = False
    # A growth-capping proxy whose caller asked to be handed control at the freeze (the GF
    # frozen-basis CSR fallback, gf_solvers.block_Green_sparse). Its `frozen` flag derives only
    # from allreduced data, so every rank of the communicator stops on the same step.
    _stop_on_freeze = basis is not None and bool(getattr(basis, "stop_on_freeze", False))

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
            start_it=start_it,
            block_widths=block_widths,
            locked=locked,
            locked_reort=locked_reort,
            krylov=krylov,
            w_out=_w_bufs[it % 2] if _w_bufs is not None else None,
            beta_norm_hist=beta_norm_hist,
            force_reort=_force_reort,
            h_norm_est=max(h_norm_est, t_norm_max),
            deflate_tol=deflate_tol,
        )
        _force_reort = _step_acted

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
        # For a Hermitian H every block norm is bounded by ||H||, so ||alpha_i||/||beta_i||
        # can never exceed the largest healthy block norm by more than rounding. A jump of
        # several orders of magnitude means the recurrence has been corrupted (lost
        # orthogonality the QR/deflation did not repair). Truncate *before* the offending
        # block — exclude alpha_i/beta_i at it_abs — so the returned T-matrix is the last
        # numerically trustworthy subspace and the diverged tail never reaches the Green's
        # function. The trailing beta then plays the role of the (ignored) residual coupling.
        # One SVD of beta_i per step: its largest singular value is the 2-norm (guard/verbose)
        # and its smallest gives ||beta_i^-1|| for the locked-reort estimate below.
        _svb = robust_svd(beta_i, compute_uv=False)
        beta_norm = float(_svb[0])
        if beta_norm_hist is not None:
            beta_norm_hist.append(beta_norm)
        alpha_norm = spectral_norm(alpha_i)
        # first_step = it == 0: this driver's own loop-local counter, always 0 on the
        # first step executed by THIS call -- see check_divergence's docstring for why this
        # must not be "it_abs == 0" -- unless a resume already replayed the guard above.
        diverged, t_norm_max, h_norm_est = check_divergence(
            beta_norm, alpha_norm, it == 0 and not guard_seeded, t_norm_max, h_norm_est, it_abs, comm
        )
        if diverged:
            termination = "diverged"
            breakdown = True
            break

        # EA16 §2.6.2 estimate-driven locking reorth: propagate the per-pair overlap
        # estimate (cheap, no O(N) work) and reorthogonalize q_next only against the
        # locked vectors whose estimate now exceeds omega_TOL.
        if partial_locked:
            bjm1_norm = float(np.linalg.norm(betas_buf[it_abs - 1], 2)) if it_abs > 0 else 0.0
            q_next, xi_prev_l, xi_l = locked_reort_step(
                xi_l, xi_prev_l, locked_evals_arr, alpha_i, float(_svb[len(_svb) - 1]), bjm1_norm,
                locked_rho, locked, q_next, mpi, comm, omega_min_l,
            )

        # Append q_next to the basis (last block = residual direction). The locking
        # deflation is applied inside block_lanczos_step_cy (before M/beta) in "full"
        # mode, or just above in "partial" mode, so q_next is orthogonal to the locked set.
        # Tail-only mode keeps nothing here: q_prev/q_curr roll below and the two-block
        # tail is assembled at return.
        if store_krylov:
            # append_block copies the coefficients into the dense buffer straight from
            # the block rows — no per-state materialization, no defensive copies.
            Q_basis.append_block(q_next)

        if verbose and (comm is None or comm.Get_rank() == 0):
            print(
                f"[BlockLanczos] it={it_abs:4d}  " f"|beta|={beta_norm:.3e}  " f"alpha_diag={np.real(np.diag(alpha_i))}"
            )

        # Convergence check
        alphas_np = alphas_buf[:it_abs+1]
        betas_np = betas_buf[:it_abs+1]
        _converged = converged_fn(alphas_np, betas_np, verbose=verbose, block_widths=block_widths + [n_curr])
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

        q_prev = q_curr
        q_curr = q_next
        block_widths.append(n_curr)
        it += 1

        # After the roll, so the returned state is exactly a "max_iter" stop's: a caller that
        # declines the hand-off resumes from it with the usual *_init protocol.
        if _stop_on_freeze and basis.frozen:
            termination = "frozen"
            break

    return _pack_result(
        alphas_buf, betas_buf, start_it, it, store_krylov, termination,
        q_prev, q_curr, q_next, Q_basis, W, block_widths, return_widths, return_status,
    )
