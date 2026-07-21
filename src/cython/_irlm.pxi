# ===========================================================================
# Implicitly Restarted Block Lanczos (IRLM) \u2014 EA16 (Meerbergen & Scott,
# RAL-TR-2000-011). The whole business logic lives here (Cython); the Python
# module impurityModel.ed.irlm is a thin re-export. The core is path-agnostic:
# both the dense-array and ManyBodyState paths run through _irlm_core via the
# is_array-dispatching block_* primitives.
# ===========================================================================


# --- Path-agnostic basis helpers. The array path represents a basis as an
# ``(N, k)`` ndarray (column blocks); the ManyBodyState path as a length-``k``
# list of states. ---------------------------------------------------------
def _q_cols(Q):
    return Q.shape[1] if is_array(Q) else len(Q)


def _check_width_sync(Q, widths, where, exact=False):
    """Assert the block-width table and the stored Krylov basis agree.

    Every restarted kernel indexes ``T`` (sized from ``block_widths``) and ``Q_basis``
    (a column store) by the *same* cumulative offsets, so ``sum(widths)`` must never
    exceed ``Q``'s column count. A sweep ends either with a trailing residual block
    appended but not counted (``sum(widths) == cols - w_last``, the ``max_iter`` and
    ``diverged`` exits) or with the last block counted and no residual stored
    (``sum(widths) == cols``, the breakdown exits) -- never with a counted block whose
    vectors were never stored.

    Desynchronization used to surface as an opaque ``matmul`` shape error several
    restarts later (or, where the shapes happened to line up, as Ritz values silently
    paired with the wrong vectors), so check it where the two meet. Pass
    ``exact=True`` where the residual block has already been split off.

    Raises:
        RuntimeError: if the invariant is violated.
    """
    cols = _q_cols(Q)
    total = int(sum(widths)) if widths is not None else 0
    if total > cols:
        raise RuntimeError(
            f"{where}: block widths sum to {total} but the Krylov basis holds only {cols} "
            f"columns ({widths!r}). The recurrence counted a block whose vectors were "
            "never stored."
        )
    if exact and total != cols:
        raise RuntimeError(
            f"{where}: block widths sum to {total} but the Krylov basis holds {cols} "
            f"columns ({widths!r}). The residual block has already been split off here, so "
            "the stored columns must be exactly the ones T is expressed in."
        )


def _q_slice(Q, a, b):
    return Q[:, a:b] if is_array(Q) else Q[a:b]


def _q_concat(A, B):
    return np.concatenate([A, B], axis=1) if is_array(A) else (list(A) + list(B))


def _copy_block(V):
    return V.copy() if is_array(V) else [s.copy() for s in V]


def _implicitly_restarted_block_lanczos_array(
    psi0,
    h_op,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol,
    max_restarts,
    verbose,
    reort_mode,
    comm,
    cntl2=None,
    cntl3=0.0,
    locked_reort="full",
):
    """Array-path entry point: prepares the ``(N, p)`` start block and the sweep
    callable, then delegates to the shared :func:`_irlm_core`. See that function for
    the EA16 algorithm description."""
    from impurityModel.ed.BlockLanczosArray import block_lanczos_array

    mpi = comm is not None and comm.size > 1
    psi0 = np.ascontiguousarray(psi0 if psi0.ndim == 2 else psi0.reshape(-1, 1), dtype=complex)
    psi0, _ = block_normalize(psi0, mpi, comm, 0.0)

    def sweep(
        v0,
        max_iter,
        alphas=None,
        betas=None,
        Q=None,
        W=None,
        reort=None,
        locked=None,
        locked_evals=None,
        locked_res=0.0,
    ):
        res = block_lanczos_array(
            psi0=v0,
            h_op=h_op,
            converged=lambda a, b, **kw: False,
            max_iter=max_iter,
            verbose=verbose,
            reort=reort if reort is not None else reort_mode,
            return_W=True,
            return_widths=True,
            comm=comm,
            alphas=alphas,
            betas=betas,
            Q=Q,
            W=W,
            locked=locked,
            locked_evals=locked_evals,
            locked_res=locked_res,
            locked_reort=locked_reort,
        )
        # (alphas, betas, Q, W, block_widths)
        return res[0], res[1], res[2], res[3], res[4]

    return _irlm_core(
        psi0,
        basis,
        num_wanted,
        max_subspace_blocks,
        tol,
        max_restarts,
        verbose,
        reort_mode,
        comm,
        sweep,
        slater=0.0,
        cntl2=cntl2,
        cntl3=cntl3,
        tag="IRLM-array",
    )


def _implicitly_restarted_block_lanczos_manybody(
    psi0,
    h_op,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol,
    max_restarts,
    verbose,
    reort_mode,
    comm,
    slaterWeightMin=0.0,
    truncation_threshold=0,
    cntl2=None,
    cntl3=0.0,
    locked_reort="full",
):
    """ManyBodyState-path entry point: prepares the length-``p`` state block and the
    sweep callable (the Cython ``block_lanczos_cy`` kernel), then delegates to the
    shared :func:`_irlm_core`. Bit-for-bit consistent with the array path."""
    mpi = comm is not None and comm.size > 1
    psi0 = list(psi0) if isinstance(psi0, (list, tuple)) else psi0
    psi0, _ = block_normalize(psi0, mpi, comm, slaterWeightMin)

    def sweep(
        v0,
        max_iter,
        alphas=None,
        betas=None,
        Q=None,
        W=None,
        reort=None,
        locked=None,
        locked_evals=None,
        locked_res=0.0,
    ):
        r = reort if reort is not None else reort_mode
        r = r.name.lower() if hasattr(r, "name") else r
        res = block_lanczos_cy(
            psi0=v0,
            h_op=h_op,
            basis=basis,
            converged_fn=lambda a, b, **kw: False,
            verbose=verbose,
            reort=r,
            max_iter=max_iter,
            slaterWeightMin=slaterWeightMin,
            truncation_threshold=truncation_threshold,
            comm=comm,
            alphas_init=alphas,
            betas_init=betas,
            Q_init=Q,
            W_init=W,
            return_widths=True,
            locked=locked,
            locked_evals=locked_evals,
            locked_res=locked_res,
            locked_reort=locked_reort,
        )
        # (alphas, betas, Q, W, block_widths)
        return res[0], res[1], res[2], res[3], res[4]

    return _irlm_core(
        psi0,
        basis,
        num_wanted,
        max_subspace_blocks,
        tol,
        max_restarts,
        verbose,
        reort_mode,
        comm,
        sweep,
        slater=slaterWeightMin,
        cntl2=cntl2,
        cntl3=cntl3,
        tag="IRLM-mbs",
    )


def _irlm_core(
    psi0,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol,
    max_restarts,
    verbose,
    reort_mode,
    comm,
    sweep,
    slater,
    cntl2,
    cntl3,
    tag,
):
    """Path-agnostic EA16 implicitly restarted block Lanczos (locking + purging).

    Faithful to Meerbergen & Scott, RAL-TR-2000-011 (EA16) for the standard
    real-symmetric/Hermitian problem in regular mode. Drives both the array and
    ManyBodyState paths through the path-agnostic ``block_*`` helpers and a path-specific
    ``sweep`` callable. Combines the block Lanczos recurrence (``sweep``, with
    resumption); **locking** of converged Ritz pairs (\u00a72.2.2); **explicit purging**
    (\u00a72.2.1, eq. 6) as the restart compression (``ea16.purge_restart``); and the EA16
    eq. (15) acceptance test (\u00a73.2.4),
    ``res <= u*||T_k|| + |CNTL(2)| + |CNTL(3)|*|theta|``.

    Returns:
        tuple: ``(eigvals, eigvecs)`` \u2014 sorted-ascending eigenvalues (length
        ``num_wanted``) and the matching Ritz vectors in the path's basis representation.
    """
    from impurityModel.ed import ea16

    mpi = comm is not None and comm.size > 1
    rank0 = (not mpi) or comm.rank == 0
    u = ea16.EPS
    if cntl2 is None:
        cntl2 = tol

    is_arr = is_array(psi0)
    p = (psi0.shape[1] if psi0.ndim == 2 else 1) if is_arr else len(psi0)
    m = max_subspace_blocks
    k0 = int(np.ceil(num_wanted / p))
    if m <= k0:
        raise ValueError("max_subspace_blocks must be strictly greater than ceil(num_wanted / p).")

    Xl = np.zeros((psi0.shape[0], 0), dtype=complex) if is_arr else []
    theta_l = []

    def _nlock():
        return Xl.shape[1] if is_arr else len(Xl)

    def _orth_against_locked(V):
        if _nlock() == 0:
            return V
        for _ in range(2):
            V, _ = block_orthogonalize(V, Xl, mpi=mpi, comm=comm)
        return V

    def _lock_block(X, vals):
        """Lock the columns of ``X`` (Ritz vectors) one at a time, reorthogonalizing each
        against the running locked set (\u00a72.6.2) and skipping any column that collapses \u2014
        i.e. is already represented in ``Xl``. Stops at ``num_wanted``."""
        nonlocal Xl
        n_locked_now = 0
        for j in range(_q_cols(X)):
            if len(theta_l) >= num_wanted:
                break
            col = _orth_against_locked(_q_slice(X, j, j + 1))
            g = block_inner(col, col, mpi, comm)
            # ``col`` entered with unit norm, so ``g[0,0]`` is the fraction of it that survives
            # deflation against the locked set. Below the rank floor ``_cholesky_or_deflate``
            # itself uses -- ``DEFLATE_EVAL_TOL = DEFLATE_TOL**2``, on the *squared* norm -- the
            # column is already represented in ``Xl`` and locking it again would return a
            # duplicate eigenpair. Scale-free because the input is normalized.
            if float(np.abs(g[0, 0])) < DEFLATE_EVAL_TOL:
                continue
            try:
                col, _ = block_normalize(col, mpi, comm, slater)
            except ValueError:
                # Belt and braces: block_normalize's own breakdown test (1e-12 absolute, since a
                # unit-norm column wants scale=1) is looser than the guard above, so this is
                # reached only on an exactly-zero column. It reduces M with a collective
                # Allreduce, so every rank raises together and skipping is MPI-collective-safe.
                continue
            Xl = np.concatenate([Xl, col], axis=1) if is_arr else (list(Xl) + list(col))
            theta_l.append(float(vals[j]))
            n_locked_now += 1
        return n_locked_now

    def _locked_kwargs():
        # Locked Ritz pairs to deflate the inner sweep against (EA16 \u00a72.6.2). Empty locked
        # set -> locked=None so the kernel skips it. ``locked_evals``/``locked_res`` feed
        # the \u00a72.6.2 overlap estimate (used only by the "partial" locked-reort mode).
        if _nlock() == 0:
            return {"locked": None}
        return {
            "locked": Xl,
            "locked_evals": np.asarray(theta_l, dtype=float),
            "locked_res": float(cntl2),
        }

    # --- Initial Lanczos run --------------------------------------------
    alphas, betas, Q_basis, _W, widths = sweep(psi0, m, **_locked_kwargs())

    for restart in range(max_restarts):
        n_need = num_wanted - len(theta_l)
        if n_need <= 0:
            break
        m_act = len(alphas)
        if m_act < 1:
            break

        # The sweep may shrink blocks (rank-deficient beta -> deflation), so the true
        # subspace dimension is sum(block_widths), not m_act * p. Build T against the
        # real widths so T (and its eigenvectors Z) line up with the stored Q_basis.
        _check_width_sync(Q_basis, widths, f"{tag} restart {restart} sweep")
        total = int(sum(widths)) if widths is not None else m_act * p
        _betas_off = betas[: m_act - 1] if len(betas) == m_act else betas
        # Banded eigensolve straight from the block coefficients (no dense T); the IRLM
        # implicit-QR restart keeps T block-tridiagonal, so the band is valid.
        evals, Z = eigh_block_tridiagonal(alphas, _betas_off, block_widths=widths)
        if mpi:
            evals = comm.bcast(evals, root=0)
            Z = comm.bcast(Z, root=0)

        order = np.argsort(evals.real)
        if total < m_act * p:
            # Sweep deflated: the block Krylov recurrence broke down into a
            # (near-)invariant subspace, so the uniform-width purge/restart below does
            # not apply. Stop and let _assemble_results pull the lowest wanted Ritz
            # pairs from this width-consistent factorization.
            if verbose and rank0:
                print(
                    f"[{tag}] Restart {restart:3d} | sweep deflated to dim {total} (<{m_act * p}); extracting & stopping."
                )
            break

        beta_last = betas[m_act - 1]
        res = ea16.ritz_residual_norms(beta_last, Z, p)
        tnorm = ea16.operator_norm_estimate(evals, theta_l)

        wanted = order[:n_need]
        if rank0:
            locked_local = [int(i) for i in wanted if res[i] <= ea16.acceptance_tol(evals[i], tnorm, cntl2, cntl3, u)]
        else:
            locked_local = None
        if mpi:
            locked_local = comm.bcast(locked_local, root=0)

        if verbose and rank0:
            print(
                f"[{tag}] Restart {restart:3d} | locked={len(theta_l)} | "
                f"MinEig={evals[order[0]].real:.6f} | "
                f"MaxWantedRes={float(np.max(res[wanted])):.2e} | newly_locked={len(locked_local)}"
            )

        # --- Lock converged wanted pairs (\u00a72.2.2) ----------------------
        if locked_local:
            X_new = block_combine(_q_slice(Q_basis, 0, total), Z[:, locked_local], slater)
            _lock_block(X_new, [evals[i].real for i in locked_local])
            n_need = num_wanted - len(theta_l)
            if n_need <= 0:
                break

        k_blocks = int(np.ceil(n_need / p))
        n_keep = k_blocks * p
        if total - n_keep < p:
            v0 = _orth_against_locked(_copy_block(psi0))
            try:
                v0, _ = block_normalize(v0, mpi, comm, slater)
            except ValueError:
                # The start block, projected orthogonal to the locked Ritz vectors, has
                # collapsed: psi0's Krylov space lies entirely within the already-locked
                # subspace, so no further wanted pairs are reachable. Collective-safe break.
                break
            alphas, betas, Q_basis, _W, widths = sweep(v0, m, **_locked_kwargs())
            continue

        # --- Purge + restart in the Ritz basis (EA16 \u00a72.2.1, eq. 6) ----
        kept_idx, _ = ea16.select_restart_indices(evals, n_keep, locked_local, which="smallest")
        C, beta_new, alphas_new, betas_new = ea16.purge_restart(evals, Z, beta_last, p, kept_idx)
        Q_used = _q_slice(Q_basis, 0, total)
        Q_new = block_combine(Q_used, C, slater)
        if _nlock() > 0:
            Q_new = _orth_against_locked(Q_new)

        # The trailing residual block can itself be rank-deficient when the sweep reached
        # a (near-)invariant subspace without shrinking any *diagonal* block, so the
        # alpha-width guard above (total < m_act * p) did not fire. Its stored width is
        # then < p and the Sorensen residual rotation qres @ beta_new is undefined. Lock
        # the lowest wanted Ritz pairs and stop.
        res_width = _q_cols(Q_basis) - total
        if res_width < p:
            X_rem = block_combine(_q_slice(Q_basis, 0, total), Z[:, order], slater)
            _lock_block(X_rem, [evals[i].real for i in order])
            if verbose and rank0:
                print(
                    f"[{tag}] Restart {restart:3d} | trailing residual block deflated (width {res_width}<{p}). Locking remaining & stopping."
                )
            break

        # The trailing normalized residual block is always present after a sweep (the
        # recurrence stores m_act+1 blocks), so the Sorensen residual reduces to rotating
        # it by the re-banding coupling beta_new (EA16 \u00a72.2.1).
        qres = _q_slice(Q_basis, total, total + p)
        f_plus = block_combine(qres, beta_new, slater)
        f_plus = _orth_against_locked(f_plus)

        # ``f_plus`` is a *residual* block (the trailing Krylov block rotated by the re-banding
        # coupling ``beta_new``), so its norm is O(||H||), not O(1): its breakdown reference must
        # be the operator norm, like the two Lanczos sweeps and unlike ``block_normalize``.
        # ``tnorm`` is the proxy already in hand -- the largest-magnitude Ritz value including the
        # locked ones, as used by the eq. (15) acceptance test above and by ``_trlm_core``'s
        # ``stop_beta``. The default ``scale=1.0`` arms the guard at 1e-12 *absolute*, which for
        # an O(||H||) block means never.
        #
        # This is a consistency fix, not a live bug. The branch is unreachable today: eq. (15)
        # locks a Ritz pair as soon as ``res <= u*tnorm + |cntl2|``, so a residual block cannot
        # survive to here once it has shrunk that far, and an anisotropic one is caught by the
        # *relative* rank test first. Instrumented over the restart/Lanczos/CIPSI suite plus a
        # warm-start probe: 734 hits, 0 decisions changed, closest approach 1127x above the
        # branch. Keep the guard honest anyway -- an isotropic residual block that is numerically
        # zero against ||H|| would deflate to nothing yet still be normalized, amplifying its
        # rounding noise by ||H||/eps.
        q_k_next, beta_k, active_k, _sv_k = block_tsqr(f_plus, mpi, comm, tnorm, slater)
        if active_k < 0:
            if verbose and rank0:
                print(f"[{tag}] Breakdown at restart -- returning current Ritz pairs.")
            break
        if active_k < p:
            # Trailing block deflated => near-invariant subspace. Lock the lowest wanted
            # Ritz pairs (ascending; collapses against Xl are skipped) and stop.
            X_rem = block_combine(_q_slice(Q_basis, 0, total), Z[:, order], slater)
            _lock_block(X_rem, [evals[i].real for i in order])
            if verbose and rank0:
                print(f"[{tag}] Restart-block deflation (active_k={active_k}). Locking remaining & stopping.")
            break
        Q_basis_new = _q_concat(Q_new, q_k_next)

        betas_pass_list = list(betas_new) if len(betas_new) > 0 else []
        if len(betas_pass_list) < k_blocks:
            betas_pass_list.append(beta_k)
        else:
            betas_pass_list[len(betas_pass_list) - 1] = beta_k
        betas_pass = np.array(betas_pass_list) if betas_pass_list else np.empty((0, p, p), dtype=complex)
        alphas_pass = np.array(alphas_new)

        # Restart-PRO continuation: continue in PARTIAL/SELECTIVE, seeding the Paige-Simon
        # estimator W at REORT_TOL (EA16 \u00a72.6.3). NONE/PERIODIC/FULL restart with FULL reort.
        if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
            W_init = np.zeros((2, k_blocks + 1, p, p), dtype=complex)
            W_init[1, :k_blocks] = REORT_TOL
            if k_blocks >= 2:
                W_init[0, : k_blocks - 1] = REORT_TOL
            W_init[1, k_blocks] = np.eye(p)
            W_init[0, k_blocks - 1] = np.eye(p)
            reort_continuation = reort_mode
        else:
            W_init = None
            reort_continuation = Reort.FULL

        alphas, betas, Q_basis, _W, widths = sweep(
            None,
            m - k_blocks,
            alphas=alphas_pass,
            betas=betas_pass,
            Q=Q_basis_new,
            W=W_init,
            reort=reort_continuation,
            **_locked_kwargs(),
        )

    # --- Final extraction -----------------------------------------------
    return _assemble_results(Xl, theta_l, alphas, betas, Q_basis, num_wanted, p, mpi, comm, slater, is_arr, widths)


def _assemble_results(Xl, theta_l, alphas, betas, Q_basis, num_wanted, p, mpi, comm, slater, is_arr, widths=None):
    """Combine locked pairs with the best remaining active Ritz pairs, sorted ascending.

    Active Ritz candidates are deflated against the locked set (and each other) and any
    that collapse are skipped, so the result never contains duplicate eigenpairs. May
    return **fewer than ``num_wanted``** pairs when the reachable invariant subspace is
    smaller than ``num_wanted``; callers must use ``len(eigvals)``.
    """
    eigvals_list = list(theta_l)
    nlock = Xl.shape[1] if is_arr else len(Xl)
    eigvecs_cols = [_q_slice(Xl, j, j + 1) for j in range(nlock)]

    n_need = num_wanted - len(eigvals_list)
    if n_need > 0 and len(alphas) > 0:
        m_act = len(alphas)
        _check_width_sync(Q_basis, widths, "IRLM final extraction")
        total = int(sum(widths)) if widths is not None else m_act * p
        _betas_off = betas[: m_act - 1] if len(betas) == m_act else betas
        # Banded eigensolve straight from the block coefficients (no dense T).
        evals, Z = eigh_block_tridiagonal(alphas, _betas_off, block_widths=widths)
        if mpi:
            evals = comm.bcast(evals, root=0)
            Z = comm.bcast(Z, root=0)
        # Deflate active Ritz candidates against the locked set (and against each other)
        # before accepting them, so near-copies of locked Ritz vectors are not accepted as
        # spurious duplicate eigenpairs.
        accepted = Xl
        Q_used = _q_slice(Q_basis, 0, total)
        for k in np.argsort(evals.real):
            if len(eigvals_list) >= num_wanted:
                break
            col = block_combine(Q_used, Z[:, k : k + 1], slater)
            n_acc = accepted.shape[1] if is_arr else len(accepted)
            if n_acc > 0:
                for _ in range(2):
                    col, _ = block_orthogonalize(col, accepted, mpi=mpi, comm=comm)
            g = block_inner(col, col, mpi, comm)
            # Same test, same reason, same threshold as _lock_block's: a unit-norm Ritz column
            # retaining less than the rank floor after deflation against ``accepted`` is a
            # near-copy of one already taken.
            if float(np.abs(g[0, 0])) < DEFLATE_EVAL_TOL:
                continue
            try:
                col, _ = block_normalize(col, mpi, comm, slater)
            except ValueError:
                continue
            eigvals_list.append(float(evals[k].real))
            eigvecs_cols.append(col)
            accepted = np.concatenate([accepted, col], axis=1) if is_arr else (list(accepted) + list(col))

    eigvals = np.array(eigvals_list[:num_wanted])
    order = np.argsort(eigvals)
    cols = eigvecs_cols[:num_wanted]
    if is_arr:
        eigvecs = np.concatenate(cols, axis=1) if cols else np.zeros((0, 0))
        eigvecs = eigvecs[:, order]
    else:
        flat = [c[0] if isinstance(c, list) else c for c in cols]
        eigvecs = [flat[i] for i in order]
    return eigvals[order], eigvecs


def implicitly_restarted_block_lanczos(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0.0,
    reort=None,
    comm=None,
    locked_reort: str = "full",
):
    """Implicitly-restarted block Lanczos (IRLM), dispatching on the operator type.

    Finds the ``num_wanted`` algebraically smallest eigenvalues of ``h_op`` via the EA16
    block Lanczos algorithm with locking, explicit purging, partial reorthogonalization
    against locked Ritz vectors, and the eq. (15) stopping criterion. Routes to the array
    path (dense/sparse ``h_op``) or the ManyBodyState path (``ManyBodyOperator``); both
    share the path-agnostic :func:`_irlm_core`.

    Returns:
        tuple[numpy.ndarray, list | numpy.ndarray]: ``(eigvals, eigvecs)``.
    """
    if reort is None:
        reort_mode = Reort.PARTIAL
    elif isinstance(reort, Reort):
        reort_mode = reort
    else:
        reort_mode = Reort[str(reort).upper()]

    if comm is None:
        comm = getattr(basis, "comm", None)

    if is_array(h_op):
        return _implicitly_restarted_block_lanczos_array(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=tol,
            max_restarts=max_restarts,
            verbose=verbose,
            reort_mode=reort_mode,
            comm=comm,
            locked_reort=locked_reort,
        )

    return _implicitly_restarted_block_lanczos_manybody(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=num_wanted,
        max_subspace_blocks=max_subspace_blocks,
        tol=tol,
        max_restarts=max_restarts,
        verbose=verbose,
        reort_mode=reort_mode,
        comm=comm,
        slaterWeightMin=slaterWeightMin,
        locked_reort=locked_reort,
    )


def implicitly_restarted_block_lanczos_cy(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0.0,
    truncation_threshold: int = 0,
    reort="partial",
    comm=None,
):
    """Implicitly-restarted block Lanczos (IRLM) for the ManyBodyState path (EA16).

    The MBS-only entry point. Resolves ``reort`` and the communicator, then runs the
    shared path-agnostic :func:`_irlm_core` via the ManyBodyState sweep kernel
    ``block_lanczos_cy``.

    Returns:
        tuple[numpy.ndarray, list]: ``(eigvals, eigvecs)`` \u2014 ``num_wanted`` smallest
        eigenvalues (ascending) and matching ``ManyBodyState`` Ritz vectors.
    """
    if comm is None:
        comm = getattr(basis, "comm", None)
    if isinstance(reort, str):
        reort_mode = Reort[reort.upper()]
    else:
        reort_mode = reort

    return _implicitly_restarted_block_lanczos_manybody(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=num_wanted,
        max_subspace_blocks=max_subspace_blocks,
        tol=tol,
        max_restarts=max_restarts,
        verbose=verbose,
        reort_mode=reort_mode,
        comm=comm,
        slaterWeightMin=slaterWeightMin,
        truncation_threshold=truncation_threshold,
    )
