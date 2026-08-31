"""Thick-Restart Block Lanczos (TRLM). All TRLM business logic lives here, as plain
Python: the path-agnostic ``_trlm_core``, the array / ManyBodyState entry points, and the
width-aware thick-restart loop. Path-agnostic through the ``is_array``-dispatching
``block_*`` primitives from :mod:`impurityModel.ed.BlockLanczosCore`.

``impurityModel.ed.BlockLanczos`` (the compiled ManyBodyState step kernel) re-exports this
module's public names (``thick_restart_block_lanczos_cy``, ``thick_restart_block_lanczos``,
and the internal entry points) so existing ``from impurityModel.ed.BlockLanczos import
...`` sites keep resolving. Unlike ``ed/irlm.py``, no name here collides between this
module's own top-level meaning and ``ed.BlockLanczos``'s re-export -- both point at the
same functions.
"""

import numpy as np
import scipy.linalg as sp

from impurityModel.ed.BlockLanczosCore import (
    BREAKDOWN_TOL,
    DEFAULT_EIGEN_TOL,
    RESTART_ORTH_TOL,
    Reort,
    _build_full_T,
    block_apply,
    block_combine,
    block_inner,
    block_normalize,
    block_orthogonalize,
    eigh_block_tridiagonal,
    is_array,
    resolve_reort,
)
from impurityModel.ed.block_view import (
    as_state_list,
    block_cols,
    check_width_sync,
    concat_cols,
    copy_block,
    slice_cols,
    trim_trailing_beta,
    width_synced_total,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyState

__all__ = [
    "_thick_restart_block_lanczos_array",
    "_trlm_core",
    "_trlm_extract",
    "thick_restart_block_lanczos",
    "thick_restart_block_lanczos_cy",
]


#: How many continuation blocks to add between mid-restart convergence checks.
#:
#: The thick-restart continuation (see :func:`_trlm_core`) rebuilds ``m - k_blocks`` blocks on
#: every restart, and used to test convergence only *after* all of them -- so a solve whose wanted
#: Ritz pairs stopped moving early still paid for the rest. Checking costs one dense ``eigh`` of
#: the current ``T``; a continuation block costs a matvec plus the two unconditional CGS2 passes
#: against the whole ``Q_basis``. Neither reliably dominates, so the check is amortized.
#:
#: Measured, not chosen: NiO 7-bath from ``impmod_tests``, production settings, single rank,
#: ``OPENBLAS_NUM_THREADS=1``, timing spent *inside* the eigensolver (total wall is dominated by
#: the Green's function and hides the effect), two repetitions per point.
#:
#: ===========  ======  ======  ======  ======  ======  ======  ======  =====
#: cadence           1       2       4       8      12      16      24    off
#: block matvecs  2677    2682    2698    2721    2744    2769    2804   2945
#: TRLM seconds   40.7    37.2    36.8    35.1    35.1    35.7    36.3   39.5
#: ===========  ======  ======  ======  ======  ======  ======  ======  =====
#:
#: Two things that table settles. Fewest matvecs is **not** fastest -- checking every block
#: (cadence 1) is slower than never checking, because the ``eigh`` outweighs the block it saves.
#: And the minimum is a broad plateau at 8-12 rather than a cliff: every cadence from 2 to 24
#: beats ``off``, so this is not tuned to one workload's edge. 8 is the matvec-conservative end
#: of the plateau. ``sigma_static`` is bit-identical to the un-checked run at every cadence.
#:
#: **TRLM only, deliberately.** The same check was written for IRLM (whose continuation is a
#: ``sweep`` call, so the hook is its ``converged_fn``) and measured: it fired **zero** times
#: across three problem shapes while adding 9-24 banded eigendecompositions, costing ~13% at
#: ``p=21, m=24`` (four reps, no overlap between the distributions). IRLM does not need it -- it
#: **locks** converged Ritz pairs and deflates them out of the active subspace between sweeps
#: (EA16 §2.2.2), so it has no equivalent of the thick restart's unconditional
#: ``m - k_blocks``-block rebuild for a mid-sweep test to catch.
CONTINUATION_CHECK_EVERY = 8


def _trlm_extract(T_full, Q, dim, num_wanted, comm, slater):
    """Diagonalize the leading ``dim`` x ``dim`` block of the (possibly arrowhead)
    ``T_full`` and form the ``num_wanted`` lowest Ritz vectors as combinations of
    ``Q[:dim]``. Path-agnostic (array ndarray, ManyBodyState list, or ManyBodyState)
    via ``block_combine``. Shared by the TRLM early-exit / breakdown / final-extraction
    paths so they all honor the true (possibly deflated) subspace dimension ``dim``
    instead of a padded ``m_actual * p``.

    Materializes a block result to ``list[ManyBodyState]`` here, at the actual return
    boundary: the documented ``eigvecs`` contract predates the block-native restart
    bookkeeping (see ``as_state_list``)."""
    eigvals_T, eigvecs_T = sp.eigh(T_full[:dim, :dim])
    if comm is not None:
        eigvals_T = comm.bcast(eigvals_T, root=0)
        eigvecs_T = comm.bcast(eigvecs_T, root=0)
    wanted = np.argsort(eigvals_T)[:num_wanted]
    eigvecs = block_combine(slice_cols(Q, 0, dim), eigvecs_T[:, wanted], slater)
    return eigvals_T[wanted], as_state_list(eigvecs)


def _extract_invariant_subspace(
    alphas, betas_off, widths, Q_basis, num_wanted, comm, slater, verbose, rank0, total, m_actual, m, deflated, q_m
):
    """Terminal early-exit for ``_trlm_core``'s initial sweep: a genuine invariant
    subspace (fewer blocks than asked), block deflation (rank-deficient residual),
    or a sweep that stored no trailing residual block at all. That last case
    (``q_m is None``) means the sweep broke out before appending ``q_next``, which
    it only does on breakdown -- the block-Krylov space is closed under H. In all
    three the spanned space is (near-)invariant, so its Ritz pairs are accurate
    eigenpairs and we extract directly. T here is a pure block-tridiagonal, so the
    banded solver suffices (no dense T); this also avoids the restart loop, which
    needs a residual block to hang the thick-restart spike on.

    Returns the ``(eigvals, eigvecs)`` result, or ``None`` if the restart loop
    should run instead.
    """
    if not (m_actual < m or deflated or q_m is None):
        return None
    if verbose and rank0:
        reason = "Invariant subspace" if m_actual < m or q_m is None else "Block deflation"
        print(f"[TRLM] {reason} (dim {total}). Extracting directly.")
    eigvals_T, eigvecs_T = eigh_block_tridiagonal(alphas, betas_off, block_widths=widths)
    if comm is not None:
        eigvals_T = comm.bcast(eigvals_T, root=0)
        eigvecs_T = comm.bcast(eigvecs_T, root=0)
    wanted = np.argsort(eigvals_T)[:num_wanted]
    return eigvals_T[wanted], as_state_list(block_combine(Q_basis, eigvecs_T[:, wanted], slater))


def _restart_coefficients(
    D,
    eigvals_T,
    keep,
    Y_last,
    beta_res,
    q_m,
    p_resid,
    nkeep,
    k_ret,
    orth_err,
    Q_ret,
    h_op,
    basis,
    mpi,
    comm,
    slater,
    stop_beta,
    num_wanted,
    restart,
    verbose,
    rank0,
):
    """Choose the thick-restart coefficients for the arrowhead ``T_full`` spike,
    via one of two arms decided by whether the retained Ritz block
    ``Q_ret ~ Q_basis[:D] @ Y_k`` stayed semi-orthogonal.

    Returns ``(early_result, T_lead, cross, q_m, p_resid, beta_res)``.
    ``early_result`` is ``None`` to continue the restart with the returned
    coefficients, or the ``(eigvals, eigvecs)`` tuple the caller should return
    immediately when the Rayleigh-Ritz residual itself collapses to an invariant
    subspace.
    """
    if k_ret == nkeep and orth_err <= RESTART_ORTH_TOL:
        # Healthy case. The textbook thick-restart coefficients follow from the recurrence
        # identity H Q = Q T + q_m beta_res E_last^H together with Q^H Q = I, and cost
        # nothing: the retained block's projected operator is diag(theta_keep), the spike
        # is beta_res @ Y_last, and the carried-over residual block q_m is already the
        # whole residual -- (I - P) H Q_ret = q_m beta_res Y_last has rank <= p.
        T_lead = np.asarray(np.diag(eigvals_T[keep]), dtype=complex)
        cross = beta_res @ Y_last  # (p_resid, nkeep)
        return None, T_lead, cross, q_m, p_resid, beta_res

    # The retained block lost semi-orthogonality (and possibly rank). Q^H Q != I, so
    # *neither* textbook coefficient is valid -- both derive from it -- and rescaling
    # them by the orthonormalizing factor only amplifies the error by 1/sigma_min.
    # Rebuild the restart from actual matvecs instead: a plain Rayleigh-Ritz step on
    # the orthonormalized retained basis, which needs no premise about Q_basis at all.
    #   T_lead = Q_ret^H H Q_ret
    #   q_m    = orthonormalized (I - Q_ret Q_ret^H) H Q_ret
    #   cross  = q_m^H H Q_ret = beta_res  (q_m _|_ Q_ret, so the projection drops out)
    # Costs k_ret matvecs, on the restarts that need it. Without Q^H Q = I the residual
    # is no longer rank <= p, so q_m can be up to k_ret wide -- hence T_full is sized
    # off p_resid below, not off p. Keeping the *whole* residual matters: the inner loop
    # stores only the sub-diagonal coupling, so a dropped piece of (I - P) H Q_ret would
    # leave H q_next with an uncaptured component on Q_ret.
    HQ = block_apply(h_op, Q_ret, basis, mpi, slater)
    ovl = block_inner(Q_ret, HQ, mpi, comm)
    T_lead = 0.5 * (ovl + np.conj(ovl.T))
    # Thick restart always full-reorthogonalizes the residual against the retained
    # basis (all modes): the arrowhead T_full requires it, and the PRO W-recurrence
    # is not maintained across a restart. The first pass reuses the overlaps already
    # formed for T_lead.
    wp, _ = block_orthogonalize(HQ, Q_ret, overlaps=ovl, mpi=mpi, comm=comm)
    wp, _ = block_orthogonalize(wp, Q_ret, mpi=mpi, comm=comm)
    try:
        q_m, beta_res = block_normalize(wp, mpi, comm, 0.0)
    except (sp.LinAlgError, ValueError):
        q_m = None
    if q_m is None or np.linalg.norm(beta_res, ord=2) <= stop_beta:
        if verbose and rank0:
            print(f"[TRLM] Invariant subspace found at restart {restart}. Stopping early.")
        return _trlm_extract(T_lead, Q_ret, k_ret, num_wanted, comm, slater), T_lead, None, q_m, p_resid, beta_res
    p_resid = block_cols(q_m)
    cross = beta_res  # (p_resid, k_ret)
    return None, T_lead, cross, q_m, p_resid, beta_res


def _trlm_core(
    psi0,
    h_op,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol,
    max_restarts,
    verbose,
    slater,
    comm,
    sweep,
):
    """Path-agnostic thick-restart block Lanczos (TRLM).

    Implements the thick-restart strategy of Knyazev (2001) and Wu & Simon (2000) to find
    the ``num_wanted`` algebraically smallest eigenvalues (and Ritz vectors) of ``H``.
    Drives both the dense-array and ``ManyBodyState`` paths through the path-agnostic
    ``block_*`` helpers (``block_apply`` / ``block_combine`` / ``block_inner`` /
    ``block_orthogonalize`` / ``block_normalize``) and a path-specific ``sweep`` callable
    (``block_lanczos_array`` for arrays, ``block_lanczos_cy`` for ``ManyBodyState``).

    Algorithm: run one block-Lanczos sweep of ``m`` blocks; diagonalise the
    block-tridiagonal ``T``; keep the ``k = ceil(n_w/p)`` lowest Ritz pairs as one retained
    super-block; reattach the residual block as a thick-restart spike; and continue the
    recurrence from block ``k`` until the maximum wanted residual ``||beta_res s_i||`` drops
    below ``tol`` or ``max_restarts`` is exhausted. A genuine invariant subspace (fewer
    blocks than requested) or a block deflation (rank-deficient residual) terminates early
    with a direct banded extraction.

    The loop is width-aware: blocks can shrink mid-restart (rank-deficient residual ->
    rectangular beta + narrower ``q_next``), so it tracks each block's actual width
    (``cur_widths``) and addresses ``T_full`` / ``Q_basis`` by cumulative offsets instead of
    a constant block width ``p``. ``nkeep = k_blocks * p`` is the *requested* number of
    retained Ritz vectors (one diagonal super-block coupled to the residual by the
    thick-restart spike), not a block width -- and not necessarily the number actually
    retained; see below.

    Two ways to restart, chosen per restart by the retained block's orthogonality:

    * ``||Q^H Q - I|| <= RESTART_ORTH_TOL`` at full rank -- use the textbook coefficients
      (``diag(theta_keep)`` for the retained block, ``beta_res @ Y_last`` for the spike).
      They follow from the recurrence identity ``H Q = Q T + q_m beta_res E_last^H``
      *together with* ``Q^H Q = I``, and cost nothing.
    * otherwise -- an explicit Rayleigh-Ritz step on the orthonormalized retained basis,
      which assumes nothing about ``Q_basis``, at the price of ``k_ret`` matvecs.
      ``reort=NONE`` needs this (measured ``||Q^H Q - I|| = 1.0`` on a spectrum with a 1e-9
      cluster, where the retained block is also rank deficient); the semi-orthogonal modes
      never take it.

    MPI: the dense ``sp.eigh`` results (replicated across ranks because ``T_full`` is built
    from Allreduced coefficients) are broadcast from rank 0 so every rank uses identical Ritz
    vectors and the restart bases stay in lock-step. The branch above is decided from an
    Allreduced Gram matrix, so it too is identical on every rank -- it must be, because the
    Rayleigh-Ritz arm issues four collectives.

    Returns:
        tuple[numpy.ndarray, list | numpy.ndarray]: ``(eigvals, eigvecs)`` — the smallest
        eigenvalues (ascending) and matching Ritz vectors in the path's basis
        representation. **May be fewer than ``num_wanted``** when the retained Ritz block
        deflates below that and the continuation then closes on an invariant subspace: only
        ``k_ret`` independent directions were ever there. Callers must use ``len(eigvals)``.
    """
    mpi = comm is not None and getattr(comm, "size", 1) > 1
    rank0 = (not mpi) or comm.rank == 0

    is_arr = is_array(psi0)
    p = (psi0.shape[1] if (is_arr and psi0.ndim == 2) else 1) if is_arr else block_cols(psi0)
    k_blocks = int(np.ceil(num_wanted / p))
    m = max_subspace_blocks
    if m <= k_blocks:
        raise ValueError("max_subspace_blocks must be strictly greater than ceil(num_wanted / p).")

    # --- Initial Lanczos run --------------------------------------------
    alphas, betas, Q_basis, widths = sweep(psi0, m)
    m_actual = len(alphas)
    if m_actual == 0:
        raise RuntimeError("Block Lanczos produced zero iterations.")

    # The sweep can shrink blocks (rank-deficient beta -> deflation), so the true subspace
    # dimension is sum(widths), not the padded m_actual * p. All slicing / T construction
    # must use the real widths or they desynchronize from Q_basis.
    total = width_synced_total(Q_basis, widths, m_actual, p, "TRLM initial sweep")
    deflated = total < m_actual * p

    # Split the trailing residual block off the basis.
    q_m = copy_block(slice_cols(Q_basis, total, block_cols(Q_basis))) if block_cols(Q_basis) > total else None
    Q_basis = slice_cols(Q_basis, 0, total)

    _betas_off = trim_trailing_beta(betas, m_actual)

    # A genuine invariant subspace, block deflation, or a sweep that never reached a
    # trailing residual block -- see _extract_invariant_subspace's docstring.
    early_result = _extract_invariant_subspace(
        alphas, _betas_off, widths, Q_basis, num_wanted, comm, slater, verbose, rank0, total, m_actual, m, deflated, q_m
    )
    if early_result is not None:
        return early_result

    # The thick restart below builds an *arrowhead* T (a spike couples the retained Ritz
    # block to the residual), which is not banded; this path keeps the dense T_full.
    T_full = _build_full_T(alphas, _betas_off, block_widths=widths)

    # --- Width-aware thick restart -------------------------------------------------
    nkeep = k_blocks * p
    p_resid = block_cols(q_m)  # q_m is not None here: the closed-space case extracted above
    # betas[-1] is the trailing coupling, padded to (p, p) by the kernel. The residual block
    # can deflate (rank p_resid < p) even when the diagonal blocks do not, leaving total ==
    # m_actual*p (so we still reach here); slice off the phantom padded rows.
    beta_res = betas[len(betas) - 1][:p_resid, :]
    cur_widths = list(widths)

    for restart in range(max_restarts):
        # Q_basis carries exactly the columns T_full[:D, :D] is expressed in: the trailing
        # residual block lives in q_m, not in Q_basis.
        check_width_sync(Q_basis, cur_widths, f"TRLM restart {restart}", exact=True)
        D = int(sum(cur_widths))
        p_last = cur_widths[len(cur_widths) - 1]
        eigvals_T, eigvecs_T = sp.eigh(T_full[:D, :D])
        if comm is not None:
            eigvals_T = comm.bcast(eigvals_T, root=0)
            eigvecs_T = comm.bcast(eigvecs_T, root=0)
        res_norms = np.linalg.norm(beta_res @ eigvecs_T[D - p_last : D, :], axis=0)
        wanted = np.argsort(eigvals_T)[:num_wanted]
        max_res = float(np.max(res_norms[wanted]))

        # Threshold below which a residual block means "stop here". Two reasons to stop,
        # and neither is a flat absolute tolerance (which would cap TRLM's residual at a
        # fixed value for any O(1)-norm operator, whatever `tol` asks):
        #   * ||beta|| < tol       -- every Ritz residual ||beta s_i|| is then already under tol;
        #   * ||beta|| <= BREAKDOWN_TOL * ||T||  -- the block is numerically zero against the
        #     operator scale, i.e. a genuine invariant subspace.
        t_scale = float(np.max(np.abs(eigvals_T))) if eigvals_T.size else 1.0
        stop_beta = max(float(tol), BREAKDOWN_TOL * t_scale)

        if verbose and rank0:
            print(f"[TRLM] Restart {restart:3d} | MinEigval={eigvals_T[0]:.6f} | MaxWantedRes={max_res:.2e}")

        done = max_res < tol
        if mpi:
            done = comm.bcast(done, root=0)
        if done:
            if verbose and rank0:
                print("[TRLM] Converged!")
            break

        keep = np.argsort(eigvals_T)[:nkeep]
        Y_k = eigvecs_T[:, keep]
        Y_last = Y_k[D - p_last : D, :]  # last-block rows -> thick-restart spike

        # The retained Ritz block X = Q_basis[:D] @ Y_k inherits Q_basis's orthonormality.
        # Both thick-restart coefficient shortcuts below are derived from Q^H Q = I, so they
        # are only usable while that holds; under reort=NONE (and, at its design tolerance,
        # PARTIAL) the recurrence loses orthogonality, spawns ghost copies of converged Ritz
        # values, and X can even become rank deficient -- block_normalize then deflates it to
        # k_ret < nkeep columns. cur_widths, T_full's leading block, the spike and the residual
        # must all follow the *actual* width k_ret, or T's dimension desynchronizes from
        # Q_basis and the next restart's block_combine raises (or, worse, silently pairs Ritz
        # values with the wrong vectors).
        X = block_combine(slice_cols(Q_basis, 0, D), Y_k, 0.0)
        gram = block_inner(X, X, mpi, comm)
        orth_err = float(np.linalg.norm(gram - np.eye(gram.shape[0]), ord=2))
        Q_ret, _ = block_normalize(X, mpi, comm, 0.0)
        k_ret = block_cols(Q_ret)
        if verbose and rank0:
            print(f"[TRLM] Restart {restart}: retained block ||Q^H Q - I|| = {orth_err:.2e}, rank {k_ret}/{nkeep}")

        # Two arms (textbook coefficients vs. an explicit Rayleigh-Ritz rebuild) --
        # see _restart_coefficients's docstring for when each applies.
        early_result, T_lead, cross, q_m, p_resid, beta_res = _restart_coefficients(
            D,
            eigvals_T,
            keep,
            Y_last,
            beta_res,
            q_m,
            p_resid,
            nkeep,
            k_ret,
            orth_err,
            Q_ret,
            h_op,
            basis,
            mpi,
            comm,
            slater,
            stop_beta,
            num_wanted,
            restart,
            verbose,
            rank0,
        )
        if early_result is not None:
            return early_result

        # The continuation adds at most (m - k_blocks) blocks, and block widths are
        # non-increasing under deflation, so none is wider than the residual block it starts
        # from. Sizing off p_resid (not the constant p) is what keeps the deflating branch --
        # whose residual can be wider than p -- inside T_full.
        dim = k_ret + (m - k_blocks) * p_resid
        T_full = np.zeros((dim, dim), dtype=complex)
        T_full[:k_ret, :k_ret] = T_lead

        Q_basis = concat_cols(Q_ret, copy_block(q_m))
        T_full[k_ret : k_ret + p_resid, :k_ret] = cross
        T_full[:k_ret, k_ret : k_ret + p_resid] = np.conj(cross.T)

        cur_widths = [k_ret, p_resid]
        off = k_ret  # column start of the current block q1
        w1 = p_resid
        q1 = q_m
        q_m = None  # consumed; the new trailing residual is set at the last inner step

        for i in range(k_blocks, m):
            wp = block_apply(h_op, q1, basis, mpi, slater)

            overlaps = block_inner(Q_basis, wp, mpi, comm)
            alpha_i = overlaps[overlaps.shape[0] - w1 :, :]  # q1^H H q1  (w1, w1)
            T_full[off : off + w1, off : off + w1] = alpha_i

            # First pass reuses the overlaps already formed for alpha_i (wp is unchanged), so
            # this is the same projection the per-pass recompute would give; the second pass
            # recomputes against the now-cleaned wp.
            wp, _ = block_orthogonalize(wp, Q_basis, overlaps=overlaps, mpi=mpi, comm=comm)
            wp, _ = block_orthogonalize(wp, Q_basis, mpi=mpi, comm=comm)

            try:
                q_next, beta_i = block_normalize(wp, mpi, comm, 0.0)
            except (sp.LinAlgError, ValueError):
                q_next = None

            # Full collapse, or a near-invariant subspace: extract from what we have.
            if q_next is None or np.linalg.norm(beta_i, ord=2) <= stop_beta:
                if verbose and rank0:
                    print(f"[TRLM] Invariant subspace found during restart at block {i}. Stopping early.")
                return _trlm_extract(T_full, Q_basis, off + w1, num_wanted, comm, slater)

            # Converged already? Exactly the test the restart-loop head runs at the top of this
            # function (`max ||beta s_i|| < tol` over the `num_wanted` lowest Ritz pairs, plain
            # `tol` and no scale rule), just applied before the subspace has been grown to its
            # full `m` blocks. Without it the continuation always rebuilds every one of the
            # `m - k_blocks` blocks, each a matvec plus the two unconditional CGS2 passes above,
            # even when the wanted pairs stopped moving many blocks ago.
            #
            # The estimate is honest here in a way it would not be inside the initial sweep: this
            # loop fully reorthogonalizes against `Q_basis` whatever `reort` asks for, so the
            # recurrence identity the residual rests on holds to working precision and a ghost
            # Ritz value cannot pass the test on lost orthogonality alone.
            D_now = off + w1
            if i < m - 1 and D_now > num_wanted and (i - k_blocks + 1) % CONTINUATION_CHECK_EVERY == 0:
                theta_now, s_now = sp.eigh(T_full[:D_now, :D_now])
                res_now = np.linalg.norm(beta_i @ s_now[D_now - w1 : D_now, :], axis=0)
                done_now = float(np.max(res_now[np.argsort(theta_now)[:num_wanted]])) < tol
                # `sp.eigh` is a rank-local LAPACK call on a replicated `T_full`; a multithreaded
                # BLAS can make its output differ in the last bits between ranks. Deciding
                # per-rank would let some ranks return here while the others enter the next
                # block's collectives -- decide on rank 0 and broadcast, per CLAUDE.md's MPI rule
                # and matching `done` at the head of the restart loop.
                if mpi:
                    done_now = comm.bcast(done_now, root=0)
                if done_now:
                    if verbose and rank0:
                        print(f"[TRLM] Converged during restart {restart} at block {i} (dim {D_now}).")
                    return _trlm_extract(T_full, Q_basis, D_now, num_wanted, comm, slater)

            # Partial deflation shrinks the block: beta_i is (w_next, w1), q_next has
            # w_next <= w1 columns; place the arrowhead with those widths.
            w_next = block_cols(q_next)
            if i < m - 1:
                T_full[off + w1 : off + w1 + w_next, off : off + w1] = beta_i
                T_full[off : off + w1, off + w1 : off + w1 + w_next] = np.conj(beta_i.T)
                Q_basis = concat_cols(Q_basis, copy_block(q_next))
                cur_widths.append(w_next)
                off += w1
                w1 = w_next
                q1 = q_next
            else:
                beta_res = beta_i
                q_m = q_next
                p_resid = w_next

    # --- Final extraction -----------------------------------------------
    final_eigvals, final_eigvecs = _trlm_extract(T_full, Q_basis, int(sum(cur_widths)), num_wanted, comm, slater)
    if verbose and rank0:
        print(f"[TRLM] Final eigvals:\n{final_eigvals}")
    return final_eigvals, final_eigvecs


def _thick_restart_block_lanczos_array(
    psi0, h_op, basis, num_wanted, max_subspace_blocks, tol, max_restarts, verbose, reort_mode, comm
):
    """Array-path entry point: prepares the ``(N, p)`` start block and the
    ``block_lanczos_array`` sweep, then delegates to the shared :func:`_trlm_core`."""
    from impurityModel.ed.BlockLanczosArray import block_lanczos_array

    mpi = comm is not None and getattr(comm, "size", 1) > 1
    # block_lanczos_array assumes an orthonormal start block (it does not normalize
    # internally); normalize here so the betas do not grow geometrically and overflow T.
    psi0 = np.ascontiguousarray(psi0 if psi0.ndim == 2 else np.reshape(psi0, (-1, 1)), dtype=complex)
    psi0, _ = block_normalize(psi0, mpi, comm, 0.0)

    def sweep(v0, max_iter):
        res = block_lanczos_array(
            psi0=v0,
            h_op=h_op,
            converged=lambda a, b, **kw: False,
            max_iter=max_iter,
            verbose=verbose,
            reort=reort_mode,
            return_W=False,
            return_widths=True,
            comm=comm,
        )
        # (alphas, betas, Q, block_widths)
        return res[0], res[1], res[2], res[3]

    return _trlm_core(
        psi0,
        h_op,
        basis,
        num_wanted,
        max_subspace_blocks,
        tol,
        max_restarts,
        verbose,
        0.0,
        comm,
        sweep,
    )


def thick_restart_block_lanczos_cy(
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
    """Thick-restart block Lanczos (TRLM) for the ManyBodyState path.

    The MBS-only entry point: prepares the length-``p`` state block and the
    ``block_lanczos_cy`` sweep, then runs the shared path-agnostic :func:`_trlm_core`.

    Args:
        psi0: Starting block of ``p`` ``ManyBodyState`` objects.
        h_op: ``ManyBodyOperator`` Hamiltonian implementing ``apply_multi(psis, cutoff)``.
        basis: ``Basis`` providing ``redistribute_psis`` and ``basis.comm``.
        num_wanted: Number of wanted lowest eigenvalues.
        max_subspace_blocks: Maximum Krylov subspace size in blocks
            (``> ceil(num_wanted / p)``).
        tol: Convergence tolerance on the maximum wanted residual. Default ``1e-8``.
        max_restarts: Maximum number of thick restarts. Default ``100``.
        verbose: Print restart diagnostics. Default ``False``.
        slaterWeightMin: Amplitude cutoff for ``ManyBodyState.prune``. Default ``0.0``.
        reort: Reorthogonalization mode (``Reort`` enum or string). Default ``'partial'``.
        comm: ``mpi4py`` communicator. Falls back to ``basis.comm`` or serial.

    Returns:
        tuple[numpy.ndarray, list]: ``(eigvals, eigvecs)`` — the smallest eigenvalues
        (ascending) and matching ``ManyBodyState`` Ritz vectors. May be fewer than
        ``num_wanted``; see :func:`_trlm_core`. Callers must use ``len(eigvals)``.
    """
    # Function-local: BlockLanczos.pyx imports this module's dispatcher back out (see
    # the bottom of this module and the `thick_restart_block_lanczos` re-export), so a
    # module-level import here would cycle at the first `pip install -e .` depending on
    # which module happens to import first. One import per outer TRLM call (sweep,
    # defined below, closes over the name), not per step -- mirrors
    # _thick_restart_block_lanczos_array's function-local BlockLanczosArray import above.
    from impurityModel.ed.BlockLanczos import block_lanczos_cy

    if comm is None:
        comm = getattr(basis, "comm", None)
    mpi = comm is not None and comm.Get_size() > 1
    reort_mode = resolve_reort(reort)

    # Adopt the block-native seed representation up front (matching what the sweep's
    # fresh-start ingestion and every restart-loop primitive already work in) instead of
    # coercing to a list: a bare list only ever reaches block_normalize's list-only
    # fallback from here on out because of this conversion, not because anything downstream
    # still needs one.
    if not isinstance(psi0, ManyBodyState):
        psi0 = ManyBodyState.from_states(list(psi0))
    psi0, _ = block_normalize(psi0, mpi, comm, slaterWeightMin)

    def sweep(v0, max_iter):
        r = reort_mode.name.lower() if hasattr(reort_mode, "name") else reort_mode
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
            return_widths=True,
        )
        # (alphas, betas, Q_basis, W, block_widths) -> drop W
        return res[0], res[1], res[2], res[4]

    return _trlm_core(
        psi0,
        h_op,
        basis,
        num_wanted,
        max_subspace_blocks,
        tol,
        max_restarts,
        verbose,
        slaterWeightMin,
        comm,
        sweep,
    )


def thick_restart_block_lanczos(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = DEFAULT_EIGEN_TOL,
    max_restarts: int = 100,
    verbose: bool = False,
    slaterWeightMin: float = 0,
    reort=Reort.PARTIAL,
):
    """Thick-restart block Lanczos (TRLM), dispatching on the operator type.

    Routes to the array path (dense/sparse ``h_op``) or the ManyBodyState path
    (``ManyBodyOperator``); both share the path-agnostic :func:`_trlm_core`.

    Returns:
        tuple[numpy.ndarray, list | numpy.ndarray]: ``(eigvals, eigvecs)``.
    """
    mpi = basis is not None and getattr(basis, "comm", None) is not None
    comm = basis.comm if mpi else None
    reort_mode = resolve_reort(reort)

    if not is_array(h_op):
        return thick_restart_block_lanczos_cy(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=tol,
            max_restarts=max_restarts,
            verbose=verbose,
            slaterWeightMin=slaterWeightMin,
            reort=reort_mode,
            comm=comm,
        )

    return _thick_restart_block_lanczos_array(
        psi0, h_op, basis, num_wanted, max_subspace_blocks, tol, max_restarts, verbose, reort_mode, comm
    )
