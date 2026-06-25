import numpy as np
import scipy.linalg as sp

from impurityModel.ed.BlockLanczosArray import (
    block_combine,
    block_inner,
    block_normalize,
    block_orthogonalize,
    is_array,
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
    reort=None,
    comm=None,
):
    """Implicitly-restarted block Lanczos (IRLM), dispatching on the operator type.

    Finds the ``num_wanted`` algebraically smallest eigenvalues of ``h_op`` via the
    EA16 (Meerbergen & Scott, RAL-TR-2000-011) block Lanczos algorithm with locking
    (§2.2.2), explicit purging (§2.2.1), partial reorthogonalization against locked
    Ritz vectors (§2.6.2), and the eq. (15) stopping criterion (§3.2.4). Both operator
    types share the path-agnostic core :func:`_irlm_core`; routing only selects the
    Lanczos sweep kernel and the basis representation:

    * the **array** path :func:`_implicitly_restarted_block_lanczos_array` when
      ``h_op`` is a dense/sparse matrix — row-block-distributed ``block_lanczos_array``;
    * the **ManyBodyState** path :func:`_implicitly_restarted_block_lanczos_manybody`
      otherwise — the hash-distributed Cython ``block_lanczos_cy`` sweep kernel.

    Both honor the requested reorthogonalization mode and continue PARTIAL/SELECTIVE
    runs across the restart in PRO (W seeded uniformly at ``REORT_TOL``).

    Args:
        psi0: Starting block — ``(N, p)`` ndarray (array path) or a length-``p`` list of
            ``ManyBodyState`` (ManyBodyState path).
        h_op: Hamiltonian — dense/sparse matrix (array path) or ``ManyBodyOperator``.
        basis: ``Basis`` providing ``comm`` (and, ManyBodyState path,
            ``redistribute_psis``); may be a light mock carrying only ``comm``.
        num_wanted: Number of wanted lowest eigenvalues.
        max_subspace_blocks: Maximum Krylov subspace size in blocks.
        tol: Convergence tolerance on the max wanted residual. Default ``1e-8``.
        max_restarts: Maximum implicit restarts. Default ``100``.
        verbose: Per-restart diagnostics. Default ``True``.
        slaterWeightMin: Amplitude cutoff (ManyBodyState path). Default ``0.0``.
        reort: Reorthogonalization mode (``Reort`` enum, string, or ``None`` →
            ``'partial'``).
        comm: ``mpi4py`` communicator; falls back to ``basis.comm`` or serial.

    Returns:
        tuple[numpy.ndarray, list | numpy.ndarray]: ``(eigvals, eigvecs)``.
    """
    from impurityModel.ed.BlockLanczosArray import Reort

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
    )


# ---------------------------------------------------------------------------
# Path-agnostic basis helpers. The array path represents a basis as an ``(N, k)``
# ndarray (column blocks); the ManyBodyState path as a length-``k`` list of states.
# ---------------------------------------------------------------------------
def _q_cols(Q):
    return Q.shape[1] if is_array(Q) else len(Q)


def _q_slice(Q, a, b):
    return Q[:, a:b] if is_array(Q) else Q[a:b]


def _q_concat(A, B):
    return np.concatenate([A, B], axis=1) if is_array(A) else (list(A) + list(B))


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
):
    """Array-path entry point: prepares the ``(N, p)`` start block and the sweep
    callable, then delegates to the shared :func:`_irlm_core`. See that function for
    the EA16 algorithm description."""
    from impurityModel.ed.BlockLanczosArray import block_lanczos_array

    mpi = comm is not None and comm.size > 1
    psi0 = np.ascontiguousarray(psi0 if psi0.ndim == 2 else psi0.reshape(-1, 1), dtype=complex)
    psi0, _ = block_normalize(psi0, mpi, comm, 0.0)

    def sweep(v0, max_iter, alphas=None, betas=None, Q=None, W=None, reort=None, locked=None):
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
    cntl2=None,
    cntl3=0.0,
):
    """ManyBodyState-path entry point: prepares the length-``p`` state block and the
    sweep callable (the Cython ``block_lanczos_cy`` kernel), then delegates to the
    shared :func:`_irlm_core`. Bit-for-bit consistent with the array path."""
    from impurityModel.ed.BlockLanczos import block_lanczos_cy

    mpi = comm is not None and comm.size > 1
    psi0 = list(psi0) if isinstance(psi0, (list, tuple)) else psi0
    psi0, _ = block_normalize(psi0, mpi, comm, slaterWeightMin)

    def sweep(v0, max_iter, alphas=None, betas=None, Q=None, W=None, reort=None, locked=None):
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
            comm=comm,
            alphas_init=alphas,
            betas_init=betas,
            Q_init=Q,
            W_init=W,
            return_widths=True,
            locked=locked,
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
    ``sweep`` callable. Combines:

    * the block Lanczos recurrence (``sweep``, with resumption);
    * **locking** of converged Ritz pairs (§2.2.2): each converged wanted pair is moved
      into the locked set ``Xl``; subsequent active vectors are reorthogonalized against
      ``Xl`` (§2.6.2, ``_orth_against_locked``), freeing subspace room for the remaining
      pairs — the mechanism that lets IRLM converge in a small subspace where the
      un-locked recurrence stalls;
    * **explicit purging** (§2.2.1, eq. 6) as the restart compression
      (``ea16.purge_restart``): the just-locked converged Ritz pairs and the unwanted
      extremal pairs are dropped, the ``n_keep`` lowest unconverged wanted pairs are
      retained, and the purged factorization is re-banded — mathematically the
      exact-shift implicit restart (Morgan 1996);
    * the EA16 eq. (15) acceptance test (§3.2.4),
      ``res <= u*||T_k|| + |CNTL(2)| + |CNTL(3)|*|theta|``.

    Args:
        psi0: Orthonormal start block — ``(N, p)`` ndarray (array) or length-``p`` list
            of ``ManyBodyState`` (MBS).
        basis: ``Basis`` (MBS path uses ``redistribute_psis``; array path unused).
        num_wanted: Number of algebraically smallest eigenvalues wanted.
        max_subspace_blocks: Maximum Krylov size in blocks (``> ceil(num_wanted/p)``).
        tol: Convenience tolerance; mapped onto ``CNTL(2)`` when ``cntl2`` is ``None``.
        max_restarts: Maximum implicit restarts.
        verbose: Per-restart diagnostics.
        reort_mode: ``Reort`` mode for the inner sweeps.
        comm: ``mpi4py`` communicator or ``None``.
        sweep: ``sweep(v0, max_iter, alphas, betas, Q, W, reort) -> (alphas, betas, Q, W)``.
        slater: ``slaterWeightMin`` amplitude cutoff (MBS path; ``0.0`` for arrays).
        cntl2: Absolute backward-error tolerance (eq. 15). Defaults to ``tol``.
        cntl3: Relative backward-error tolerance (eq. 15). Default ``0.0``.
        tag: Label for verbose output.

    Returns:
        tuple: ``(eigvals, eigvecs)`` — sorted-ascending eigenvalues (length
        ``num_wanted``) and the matching Ritz vectors in the path's basis representation.
    """
    from impurityModel.ed.BlockLanczosArray import Reort, REORT_TOL, _build_full_T, _cholesky_or_deflate
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
        against the running locked set (§2.6.2) and skipping any column that collapses —
        i.e. is already represented in ``Xl`` (the active spectrum still contains
        approximations to the already-locked eigenvalues under boundary reorthogonalization,
        so a converged-again duplicate must not be locked twice). Stops at ``num_wanted``."""
        nonlocal Xl
        n_locked_now = 0
        for j in range(_q_cols(X)):
            if len(theta_l) >= num_wanted:
                break
            col = _orth_against_locked(_q_slice(X, j, j + 1))
            g = block_inner(col, col, mpi, comm)
            if float(np.abs(g[0, 0])) < 1e-16:
                continue
            try:
                col, _ = block_normalize(col, mpi, comm, slater)
            except ValueError:
                # The column collapsed under block_normalize's stricter deflation floor
                # (DEFLATE_TOL ~ sqrt(eps), tighter than the cheap pre-check above): it is
                # already represented in the locked set. block_normalize reduces M with a
                # collective Allreduce, so every rank raises together and skipping is
                # MPI-collective-safe.
                continue
            Xl = np.concatenate([Xl, col], axis=1) if is_arr else (list(Xl) + list(col))
            theta_l.append(float(vals[j]))
            n_locked_now += 1
        return n_locked_now

    def _locked():
        # Locked Ritz vectors to deflate the inner sweep against (EA16 §2.6.2). Empty
        # locked set -> None so the kernel skips the projection entirely.
        return Xl if _nlock() > 0 else None

    # --- Initial Lanczos run --------------------------------------------
    alphas, betas, Q_basis, _W, widths = sweep(psi0, m, locked=_locked())

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
        total = int(sum(widths)) if widths is not None else m_act * p
        _betas_off = betas[: m_act - 1] if len(betas) == m_act else betas
        T = _build_full_T(alphas, _betas_off, block_widths=widths)
        evals, Z = sp.eigh(T)
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
                print(f"[{tag}] Restart {restart:3d} | sweep deflated to dim {total} (<{m_act * p}); extracting & stopping.")
            break

        beta_last = betas[m_act - 1]
        res = ea16.ritz_residual_norms(beta_last, Z, p)
        tnorm = ea16.operator_norm_estimate(evals, theta_l)

        wanted = order[:n_need]
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

        # --- Lock converged wanted pairs (§2.2.2) ----------------------
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
                # The start block, projected orthogonal to the locked Ritz vectors,
                # has collapsed: psi0's Krylov space lies entirely within the
                # already-locked subspace, so no further wanted pairs are reachable
                # from it. block_normalize reduces M with a collective Allreduce, so
                # every rank raises together and this break is MPI-collective-safe.
                # Stop restarting and return what has been locked / found so far.
                break
            alphas, betas, Q_basis, _W, widths = sweep(v0, m, locked=_locked())
            continue

        # --- Purge + restart in the Ritz basis (EA16 §2.2.1, eq. 6) ----
        # Ghost-of-locked filtering (locked_evals/ghost_tol) is intentionally left off:
        # the inner sweep already deflates every Lanczos vector against the locked set
        # (EA16 §2.6.2), which removes ghosts by *eigenvector* and so — unlike an
        # eigenvalue match — does not endanger true degeneracies. See
        # ea16.select_restart_indices for the disabled defense-in-depth fallback.
        kept_idx, _ = ea16.select_restart_indices(evals, n_keep, locked_local, which="smallest")
        C, beta_new, alphas_new, betas_new = ea16.purge_restart(evals, Z, beta_last, p, kept_idx)
        Q_used = _q_slice(Q_basis, 0, total)
        Q_new = block_combine(Q_used, C, slater)
        if _nlock() > 0:
            Q_new = _orth_against_locked(Q_new)

        # The trailing normalized residual block is always present after a sweep (the
        # recurrence stores m_act+1 blocks), so the Sorensen residual reduces to rotating
        # it by the re-banding coupling beta_new (EA16 §2.2.1).
        qres = _q_slice(Q_basis, total, total + p)
        f_plus = block_combine(qres, beta_new, slater)
        f_plus = _orth_against_locked(f_plus)

        M = block_inner(f_plus, f_plus, mpi, comm)
        if np.any(np.isnan(M)) or np.any(np.isinf(M)):
            if verbose and rank0:
                print(f"[{tag}] Breakdown at restart -- returning current Ritz pairs.")
            break

        beta_k, beta_k_inv, active_k = _cholesky_or_deflate(M, p)
        if active_k < p:
            # Trailing block deflated => near-invariant subspace. Lock the lowest wanted
            # Ritz pairs (ascending; collapses against Xl are skipped) and stop.
            X_rem = block_combine(_q_slice(Q_basis, 0, total), Z[:, order], slater)
            _lock_block(X_rem, [evals[i].real for i in order])
            if verbose and rank0:
                print(f"[{tag}] Restart-block deflation (active_k={active_k}). Locking remaining & stopping.")
            break

        q_k_next = block_combine(f_plus, beta_k_inv, slater)
        Q_basis_new = _q_concat(Q_new, q_k_next)

        betas_pass_list = list(betas_new) if len(betas_new) > 0 else []
        if len(betas_pass_list) < k_blocks:
            betas_pass_list.append(beta_k)
        else:
            betas_pass_list[len(betas_pass_list) - 1] = beta_k
        betas_pass = np.array(betas_pass_list) if betas_pass_list else np.empty((0, p, p), dtype=complex)
        alphas_pass = np.array(alphas_new)

        # Restart-PRO continuation: continue in PARTIAL/SELECTIVE, seeding the Paige-Simon
        # estimator W at REORT_TOL (EA16 §2.6.3). NONE/PERIODIC/FULL restart with FULL reort.
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
            locked=_locked(),
        )

    # --- Final extraction -----------------------------------------------
    return _assemble_results(
        Xl, theta_l, alphas, betas, Q_basis, num_wanted, p, mpi, comm, slater, is_arr, _build_full_T, widths
    )


def _copy_block(V):
    return V.copy() if is_array(V) else [s.copy() for s in V]


def _assemble_results(
    Xl, theta_l, alphas, betas, Q_basis, num_wanted, p, mpi, comm, slater, is_arr, _build_full_T, widths=None
):
    """Combine locked pairs with the best remaining active Ritz pairs, sorted ascending.

    ``widths`` carries the per-block dimensions from the final sweep so a deflated
    (shrunk-block) factorization is diagonalized at its true dimension ``sum(widths)``
    rather than the padded ``m_act * p`` — keeping ``T``/``Z`` consistent with the
    stored ``Q_basis``.

    Active Ritz candidates are deflated against the locked set (and each other) and any
    that collapse are skipped, so the result never contains duplicate eigenpairs.

    .. note::
        The returned arrays may contain **fewer than ``num_wanted``** pairs. When the
        reachable invariant subspace is smaller than ``num_wanted`` (e.g. IRLM seeded
        from exact eigenvectors, or a small sector), there simply are not that many
        distinct eigenpairs, and the deduplication drops the rest rather than returning
        spurious copies. Callers must not assume exactly ``num_wanted`` results — use
        ``len(eigvals)`` and filter by energy as appropriate.
    """
    eigvals_list = list(theta_l)
    nlock = Xl.shape[1] if is_arr else len(Xl)
    eigvecs_cols = [_q_slice(Xl, j, j + 1) for j in range(nlock)]

    n_need = num_wanted - len(eigvals_list)
    if n_need > 0 and len(alphas) > 0:
        m_act = len(alphas)
        total = int(sum(widths)) if widths is not None else m_act * p
        _betas_off = betas[: m_act - 1] if len(betas) == m_act else betas
        T = _build_full_T(alphas, _betas_off, block_widths=widths)
        evals, Z = sp.eigh(T)
        if mpi:
            evals = comm.bcast(evals, root=0)
            Z = comm.bcast(Z, root=0)
        # Deflate active Ritz candidates against the locked set (and against each other)
        # before accepting them. When the start space is exhausted by the locked set
        # (e.g. IRLM seeded from already-converged eigenvectors, as in
        # CIPSISolver.get_eigenvectors), the leftover active factorization still
        # contains near-copies of the locked Ritz vectors; without this deflation they
        # are accepted as spurious duplicate eigenpairs (each true eigenvalue returned
        # twice), which double-counts states in the downstream thermal average.
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
            if float(np.abs(g[0, 0])) < 1e-12:
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
