"""Implicitly Restarted Block Lanczos (IRLM) — EA16 (Meerbergen & Scott,
RAL-TR-2000-011). All IRLM business logic lives here, as plain Python: the
path-agnostic core (``_irlm_core``), the array / ManyBodyState entry points, locking,
explicit purging, and result assembly. Path-agnostic through the ``is_array``-dispatching
``block_*`` primitives from :mod:`impurityModel.ed.BlockLanczosCore`.

``impurityModel.ed.BlockLanczos`` (the compiled ManyBodyState step kernel) re-exports this
module's MBS entry point under the name ``implicitly_restarted_block_lanczos_cy`` so
``from impurityModel.ed.BlockLanczos import implicitly_restarted_block_lanczos_cy`` keeps
resolving. This module's *own* top-level name ``implicitly_restarted_block_lanczos_cy`` has
always meant something different -- the dispatching entry point below, not the MBS-only one
-- so the MBS-only function is defined here under
``_implicitly_restarted_block_lanczos_cy_manybody`` and the public name is bound to the
dispatcher at the bottom of this module. See that binding for the full contract.
"""

import numpy as np

from impurityModel.ed import ea16
from impurityModel.ed.BlockLanczosCore import (
    DEFAULT_EIGEN_TOL,
    DEFLATE_EVAL_TOL,
    REORT_TOL,
    Reort,
    block_combine,
    block_inner,
    block_normalize,
    block_orthogonalize,
    block_tsqr,
    eigh_block_tridiagonal,
    is_array,
    resolve_reort,
)
from impurityModel.ed.block_view import (
    block_cols,
    concat_cols,
    copy_block,
    slice_cols,
    trim_trailing_beta,
    width_synced_total,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyState

__all__ = [
    "_assemble_results",
    "_implicitly_restarted_block_lanczos_array",
    "_implicitly_restarted_block_lanczos_manybody",
    "_irlm_core",
    "implicitly_restarted_block_lanczos",
    "implicitly_restarted_block_lanczos_cy",
]


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
    # Function-local: BlockLanczos.pyx imports this module's MBS entry point back out
    # (see the bottom of this file), so a module-level import here would cycle at the
    # first `pip install -e .` depending on which module happens to import first. One
    # import per outer IRLM call (sweep, defined below, closes over the name), not per
    # step -- mirrors the array twin's function-local BlockLanczosArray import above.
    from impurityModel.ed.BlockLanczos import block_lanczos_cy

    mpi = comm is not None and comm.size > 1
    # Adopt the block-native seed representation up front (matching what the sweep's
    # fresh-start ingestion and every restart-loop primitive already work in) instead of
    # coercing to a list: a bare list only ever reaches block_normalize's list-only
    # fallback from here on out because of this conversion, not because anything downstream
    # still needs one.
    if not isinstance(psi0, ManyBodyState):
        psi0 = ManyBodyState.from_states(list(psi0))
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


class LockedSet:
    """Running set of locked (converged) Ritz pairs (EA16 §2.2.2), coupled to the
    sweep's basis representation.

    ``Xl`` is block-compatible with ``Q_basis`` from the start -- an empty
    ``ManyBodyState`` (width 0) on the MBS path, not an empty list -- since every
    column sliced out of it (``lock_block``'s ``col``, ``_assemble_results``' ``col``)
    is a block too.
    """

    def __init__(self, psi0, is_arr, mpi, comm, slater):
        self.is_arr = is_arr
        self.mpi = mpi
        self.comm = comm
        self.slater = slater
        self.Xl = np.zeros((psi0.shape[0], 0), dtype=complex) if is_arr else ManyBodyState()
        self.theta_l = []

    @property
    def n(self):
        return self.Xl.shape[1] if self.is_arr else block_cols(self.Xl)

    def orth_against(self, V):
        if self.n == 0:
            return V
        for _ in range(2):
            V, _ = block_orthogonalize(V, self.Xl, mpi=self.mpi, comm=self.comm)
        return V

    def lock_block(self, X, vals, num_wanted):
        """Lock the columns of ``X`` (Ritz vectors) one at a time, reorthogonalizing each
        against the running locked set (§2.6.2) and skipping any column that collapses —
        i.e. is already represented in ``Xl``. Stops at ``num_wanted``."""
        n_locked_now = 0
        for j in range(block_cols(X)):
            if len(self.theta_l) >= num_wanted:
                break
            col = self.orth_against(slice_cols(X, j, j + 1))
            g = block_inner(col, col, self.mpi, self.comm)
            # ``col`` entered with unit norm, so ``g[0,0]`` is the fraction of it that survives
            # deflation against the locked set. Below the rank floor the factorization itself
            # uses -- ``DEFLATE_EVAL_TOL = DEFLATE_TOL**2``, on the *squared* norm -- the
            # column is already represented in ``Xl`` and locking it again would return a
            # duplicate eigenpair. Scale-free because the input is normalized.
            if float(np.abs(g[0, 0])) < DEFLATE_EVAL_TOL:
                continue
            try:
                col, _ = block_normalize(col, self.mpi, self.comm, self.slater)
            except ValueError:
                # Belt and braces: block_normalize's own breakdown test (1e-12 absolute, since a
                # unit-norm column wants scale=1) is looser than the guard above, so this is
                # reached only on an exactly-zero column. It reduces M with a collective
                # Allreduce, so every rank raises together and skipping is MPI-collective-safe.
                continue
            self.Xl = np.concatenate([self.Xl, col], axis=1) if self.is_arr else concat_cols(self.Xl, col)
            self.theta_l.append(float(vals[j]))
            n_locked_now += 1
        return n_locked_now

    def sweep_kwargs(self, cntl2):
        # Locked Ritz pairs to deflate the inner sweep against (EA16 §2.6.2). Empty locked
        # set -> locked=None so the kernel skips it. ``locked_evals``/``locked_res`` feed
        # the §2.6.2 overlap estimate (used only by the "partial" locked-reort mode).
        if self.n == 0:
            return {"locked": None}
        return {
            "locked": self.Xl,
            "locked_evals": np.asarray(self.theta_l, dtype=float),
            "locked_res": float(cntl2),
        }


def lock_remaining_and_stop(locked, Q_basis, total, Z, order, evals, num_wanted, slater, tag, verbose, rank0, reason):
    """Lock the lowest wanted Ritz pairs from the current factorization and print
    why -- the shared tail of every "the sweep/residual/restart block deflated,
    nothing more to gain by continuing" exit in ``_irlm_core``/``_purge_and_reband``.
    The caller still does its own ``break``: this only performs the locking and the
    diagnostic, not the loop control.
    """
    X_rem = block_combine(slice_cols(Q_basis, 0, total), Z[:, order], slater)
    locked.lock_block(X_rem, [evals[i].real for i in order], num_wanted)
    if verbose and rank0:
        print(f"[{tag}] {reason}")


def _purge_and_reband(
    evals,
    Z,
    beta_last,
    p,
    n_keep,
    locked_local,
    Q_basis,
    total,
    slater,
    locked,
    mpi,
    comm,
    tnorm,
    k_blocks,
    verbose,
    rank0,
    tag,
    restart,
    order,
    num_wanted,
):
    """Purge + restart in the Ritz basis (EA16 §2.2.1, eq. 6): compress the ``m``
    active blocks down to ``n_keep`` in the Ritz basis, re-band the trailing
    residual against the compression, and re-factor the new trailing block via
    TSQR.

    Returns ``(stop, Q_basis_new, alphas_pass, betas_pass)``. ``stop`` is ``True``
    on breakdown or a deflated re-factored block (already locked by this call, via
    :func:`lock_remaining_and_stop`) -- the caller should ``break`` without using
    the other return values.
    """
    kept_idx, _ = ea16.select_restart_indices(evals, n_keep, locked_local, which="smallest")
    C, beta_new, alphas_new, betas_new = ea16.purge_restart(evals, Z, beta_last, p, kept_idx)
    Q_used = slice_cols(Q_basis, 0, total)
    Q_new = block_combine(Q_used, C, slater)
    if locked.n > 0:
        Q_new = locked.orth_against(Q_new)

    # The trailing normalized residual block is always present after a sweep (the
    # recurrence stores m_act+1 blocks), so the Sorensen residual reduces to rotating
    # it by the re-banding coupling beta_new (EA16 §2.2.1).
    qres = slice_cols(Q_basis, total, total + p)
    f_plus = block_combine(qres, beta_new, slater)
    f_plus = locked.orth_against(f_plus)

    # f_plus is a residual block -- O(||H||) -- so its breakdown reference is tnorm
    # (the largest-magnitude Ritz value, already in hand), not the default O(1) scale
    # block_tsqr otherwise arms at. See doc/lanczos_invariants.md ("deflation vs
    # breakdown scales") for why this branch is a consistency fix rather than a live
    # bug, and the instrumentation behind that claim.
    q_k_next, beta_k, active_k, _sv_k = block_tsqr(f_plus, mpi, comm, tnorm, slater)
    if active_k < 0:
        if verbose and rank0:
            print(f"[{tag}] Breakdown at restart -- returning current Ritz pairs.")
        return True, None, None, None
    if active_k < p:
        # Trailing block deflated => near-invariant subspace. Lock the lowest wanted
        # Ritz pairs (ascending; collapses against Xl are skipped) and stop.
        lock_remaining_and_stop(
            locked,
            Q_basis,
            total,
            Z,
            order,
            evals,
            num_wanted,
            slater,
            tag,
            verbose,
            rank0,
            f"Restart-block deflation (active_k={active_k}). Locking remaining & stopping.",
        )
        return True, None, None, None
    Q_basis_new = concat_cols(Q_new, q_k_next)

    betas_pass_list = list(betas_new) if len(betas_new) > 0 else []
    if len(betas_pass_list) < k_blocks:
        betas_pass_list.append(beta_k)
    else:
        betas_pass_list[len(betas_pass_list) - 1] = beta_k
    betas_pass = np.array(betas_pass_list) if betas_pass_list else np.empty((0, p, p), dtype=complex)
    alphas_pass = np.array(alphas_new)

    return False, Q_basis_new, alphas_pass, betas_pass


def _irlm_core(
    psi0,
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
    resumption); **locking** of converged Ritz pairs (§2.2.2); **explicit purging**
    (§2.2.1, eq. 6) as the restart compression (``ea16.purge_restart``); and the EA16
    eq. (15) acceptance test (§3.2.4),
    ``res <= u*||T_k|| + |CNTL(2)| + |CNTL(3)|*|theta|``.

    Returns:
        tuple: ``(eigvals, eigvecs)`` — sorted-ascending eigenvalues (length
        ``num_wanted``) and the matching Ritz vectors in the path's basis representation.
    """
    mpi = comm is not None and comm.size > 1
    rank0 = (not mpi) or comm.rank == 0
    u = ea16.EPS
    if cntl2 is None:
        cntl2 = tol

    is_arr = is_array(psi0)
    p = (psi0.shape[1] if psi0.ndim == 2 else 1) if is_arr else block_cols(psi0)
    m = max_subspace_blocks
    k0 = int(np.ceil(num_wanted / p))
    if m <= k0:
        raise ValueError("max_subspace_blocks must be strictly greater than ceil(num_wanted / p).")

    locked = LockedSet(psi0, is_arr, mpi, comm, slater)

    # --- Initial Lanczos run --------------------------------------------
    alphas, betas, Q_basis, _W, widths = sweep(psi0, m, **locked.sweep_kwargs(cntl2))

    for restart in range(max_restarts):
        n_need = num_wanted - len(locked.theta_l)
        if n_need <= 0:
            break
        m_act = len(alphas)
        if m_act < 1:
            break

        # The sweep may shrink blocks (rank-deficient beta -> deflation), so the true
        # subspace dimension is sum(block_widths), not m_act * p. Build T against the
        # real widths so T (and its eigenvectors Z) line up with the stored Q_basis.
        total = width_synced_total(Q_basis, widths, m_act, p, f"{tag} restart {restart} sweep")
        _betas_off = trim_trailing_beta(betas, m_act)
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
        tnorm = ea16.operator_norm_estimate(evals, locked.theta_l)

        wanted = order[:n_need]
        if rank0:
            locked_local = [int(i) for i in wanted if res[i] <= ea16.acceptance_tol(evals[i], tnorm, cntl2, cntl3, u)]
        else:
            locked_local = None
        if mpi:
            locked_local = comm.bcast(locked_local, root=0)

        if verbose and rank0:
            print(
                f"[{tag}] Restart {restart:3d} | locked={len(locked.theta_l)} | "
                f"MinEig={evals[order[0]].real:.6f} | "
                f"MaxWantedRes={float(np.max(res[wanted])):.2e} | newly_locked={len(locked_local)}"
            )

        # --- Lock converged wanted pairs (§2.2.2) ----------------------
        if locked_local:
            X_new = block_combine(slice_cols(Q_basis, 0, total), Z[:, locked_local], slater)
            locked.lock_block(X_new, [evals[i].real for i in locked_local], num_wanted)
            n_need = num_wanted - len(locked.theta_l)
            if n_need <= 0:
                break

        k_blocks = int(np.ceil(n_need / p))
        n_keep = k_blocks * p
        if total - n_keep < p:
            # psi0 is already block-native on the MBS path (the entry point converts it
            # before the first sweep), matching Xl's representation, and block_lanczos_cy's
            # fresh-start ingestion now accepts a ManyBodyState seed directly -- so
            # v0 can stay a block end to end instead of round-tripping through
            # list[ManyBodyState] just for the locked.orth_against projection.
            v0 = locked.orth_against(copy_block(psi0))
            try:
                v0, _ = block_normalize(v0, mpi, comm, slater)
            except ValueError:
                # The start block, projected orthogonal to the locked Ritz vectors, has
                # collapsed: psi0's Krylov space lies entirely within the already-locked
                # subspace, so no further wanted pairs are reachable. Collective-safe break.
                break
            alphas, betas, Q_basis, _W, widths = sweep(v0, m, **locked.sweep_kwargs(cntl2))
            continue

        # The trailing residual block can itself be rank-deficient when the sweep reached
        # a (near-)invariant subspace without shrinking any *diagonal* block, so the
        # alpha-width guard above (total < m_act * p) did not fire. Its stored width is
        # then < p and the Sorensen residual rotation qres @ beta_new is undefined. Lock
        # the lowest wanted Ritz pairs and stop.
        #
        # Must run BEFORE purge_restart: an exact breakdown (res_width == 0) zeros every
        # Ritz residual, so eq. (15) locks everything and purge_restart's kept_idx can come
        # back empty (used to crash on an empty np.concatenate) -- see
        # doc/lanczos_invariants.md ("deflation vs breakdown scales"). total/res_width are
        # rank-invariant by construction (TSQR's globally-reduced active_k); the bcast
        # below is defense-in-depth.
        res_width = block_cols(Q_basis) - total
        take_break = res_width < p
        if mpi:
            take_break = comm.bcast(take_break, root=0)
        if take_break:
            lock_remaining_and_stop(
                locked,
                Q_basis,
                total,
                Z,
                order,
                evals,
                num_wanted,
                slater,
                tag,
                verbose,
                rank0,
                f"Restart {restart:3d} | trailing residual block deflated (width {res_width}<{p}). Locking remaining & stopping.",
            )
            break

        # Purge + restart in the Ritz basis (EA16 §2.2.1, eq. 6) -- see
        # _purge_and_reband's docstring.
        stop, Q_basis_new, alphas_pass, betas_pass = _purge_and_reband(
            evals,
            Z,
            beta_last,
            p,
            n_keep,
            locked_local,
            Q_basis,
            total,
            slater,
            locked,
            mpi,
            comm,
            tnorm,
            k_blocks,
            verbose,
            rank0,
            tag,
            restart,
            order,
            num_wanted,
        )
        if stop:
            break

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
            **locked.sweep_kwargs(cntl2),
        )

    # --- Final extraction -----------------------------------------------
    return _assemble_results(
        locked.Xl, locked.theta_l, alphas, betas, Q_basis, num_wanted, p, mpi, comm, slater, is_arr, widths
    )


def _assemble_results(Xl, theta_l, alphas, betas, Q_basis, num_wanted, p, mpi, comm, slater, is_arr, widths=None):
    """Combine locked pairs with the best remaining active Ritz pairs, sorted ascending.

    Active Ritz candidates are deflated against the locked set (and each other) and any
    that collapse are skipped, so the result never contains duplicate eigenpairs. May
    return **fewer than ``num_wanted``** pairs when the reachable invariant subspace is
    smaller than ``num_wanted``; callers must use ``len(eigvals)``.
    """
    eigvals_list = list(theta_l)
    nlock = Xl.shape[1] if is_arr else block_cols(Xl)
    eigvecs_cols = [slice_cols(Xl, j, j + 1) for j in range(nlock)]

    n_need = num_wanted - len(eigvals_list)
    if n_need > 0 and len(alphas) > 0:
        m_act = len(alphas)
        total = width_synced_total(Q_basis, widths, m_act, p, "IRLM final extraction")
        _betas_off = trim_trailing_beta(betas, m_act)
        # Banded eigensolve straight from the block coefficients (no dense T).
        evals, Z = eigh_block_tridiagonal(alphas, _betas_off, block_widths=widths)
        if mpi:
            evals = comm.bcast(evals, root=0)
            Z = comm.bcast(Z, root=0)
        # Deflate active Ritz candidates against the locked set (and against each other)
        # before accepting them, so near-copies of locked Ritz vectors are not accepted as
        # spurious duplicate eigenpairs.
        accepted = Xl
        Q_used = slice_cols(Q_basis, 0, total)
        for k in np.argsort(evals.real):
            if len(eigvals_list) >= num_wanted:
                break
            col = block_combine(Q_used, Z[:, k : k + 1], slater)
            n_acc = accepted.shape[1] if is_arr else block_cols(accepted)
            if n_acc > 0:
                for _ in range(2):
                    col, _ = block_orthogonalize(col, accepted, mpi=mpi, comm=comm)
            g = block_inner(col, col, mpi, comm)
            # Same test, same reason, same threshold as LockedSet.lock_block's: a unit-norm Ritz column
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
            accepted = np.concatenate([accepted, col], axis=1) if is_arr else concat_cols(accepted, col)

    eigvals = np.array(eigvals_list[:num_wanted])
    order = np.argsort(eigvals)
    cols = eigvecs_cols[:num_wanted]
    if is_arr:
        eigvecs = np.concatenate(cols, axis=1) if cols else np.zeros((0, 0))
        eigvecs = eigvecs[:, order]
    else:
        # Every entry of cols is a width-1 ManyBodyState here (Xl and every col above
        # are blocks throughout); materialize to the documented list[ManyBodyState]
        # contract only at this actual return boundary.
        flat = [c.to_states()[0] for c in cols]
        eigvecs = [flat[i] for i in order]
    return eigvals[order], eigvecs


def implicitly_restarted_block_lanczos(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = DEFAULT_EIGEN_TOL,
    max_restarts: int = 100,
    verbose: bool = False,
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
    reort_mode = Reort.PARTIAL if reort is None else resolve_reort(reort)

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


def _implicitly_restarted_block_lanczos_cy_manybody(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = DEFAULT_EIGEN_TOL,
    max_restarts: int = 100,
    verbose: bool = False,
    slaterWeightMin: float = 0.0,
    truncation_threshold: int = 0,
    reort="partial",
    comm=None,
):
    """Implicitly-restarted block Lanczos (IRLM) for the ManyBodyState path (EA16).

    The MBS-only entry point. Resolves ``reort`` and the communicator, then runs the
    shared path-agnostic :func:`_irlm_core` via the ManyBodyState sweep kernel
    ``block_lanczos_cy``.

    Re-exported from :mod:`impurityModel.ed.BlockLanczos` as
    ``implicitly_restarted_block_lanczos_cy`` -- see the module docstring above for why
    this module's *own* top-level name of that spelling is bound to the dispatcher
    (:func:`implicitly_restarted_block_lanczos`) instead.

    Returns:
        tuple[numpy.ndarray, list]: ``(eigvals, eigvecs)`` — ``num_wanted`` smallest
        eigenvalues (ascending) and matching ``ManyBodyState`` Ritz vectors.
    """
    if comm is None:
        comm = getattr(basis, "comm", None)
    reort_mode = resolve_reort(reort)

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


# ed.irlm's own `implicitly_restarted_block_lanczos_cy` name has always meant the
# *dispatcher* (see module docstring): the pre-move shim aliased it explicitly
# (`from impurityModel.ed.BlockLanczos import implicitly_restarted_block_lanczos as
# implicitly_restarted_block_lanczos_cy`). Moving the MBS-only function's body into this
# module creates a genuine name collision between the two meanings under one spelling;
# resolved by giving the MBS-only function its own name
# (_implicitly_restarted_block_lanczos_cy_manybody) and keeping this module's public
# `implicitly_restarted_block_lanczos_cy` bound to the dispatcher, exactly as before.
implicitly_restarted_block_lanczos_cy = implicitly_restarted_block_lanczos
