# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True, freethreading_compatible=True

"""
BiCGSTAB.pyx
============
Block BiCGSTAB linear solver for the many-body layer.

Solves ``A X = Y`` for a block right-hand side, on the same two representations the
block-Lanczos kernels support: dense ``numpy`` arrays and hash-distributed
``ManyBodyBlockState`` blocks. It is the memory-flat alternative to block Lanczos for
resolvent evaluation -- it carries a fixed handful of vectors instead of a retained
Krylov basis, and has no orthogonality to lose -- which is what the per-frequency
Green's-function driver is built on. Today its production caller is the RIXS
intermediate-state resolvent (``spectra._rixs_driver``).

``impurityModel.ed.cg`` re-exports the public entry point, so the historical import path
keeps working; this module is where the logic lives (the same arrangement as
``irlm.py`` / ``trlm.py`` over ``BlockLanczos.pyx``).
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import (
    _cholesky_or_deflate,
    block_add_scaled,
    block_apply,
    block_combine,
    block_inner,
    is_array,
)
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyBlockState,
    ManyBodyState,
    block_add_scaled_cy,
    block_inner_cy,
)


def block_bicgstab(A, x0, y, basis, double slaterWeightMin, atol=1e-8, rtol=1e-12, **kwargs):
    """
    Solve a linear system ``A X = Y`` with Block BiCGSTAB.

    Works for dense (``numpy`` array) and fully sparse (``ManyBodyState``) blocks, and for an
    arbitrary -- possibly **rank-deficient** -- right-hand side. A rank-deficient block RHS
    (linearly dependent columns) would otherwise make the block coefficient matrix singular and
    stall the solver at the initial guess; here the initial residual block is deflated to its
    independent directions, the full-rank reduced system is solved, and the dependent columns
    are reconstructed by linearity.

    ``atol`` is a tolerance on the residual **relative to the right-hand side**: the solve stops
    once ``||A X - Y||`` is about ``atol * ||Y||``, whatever ``x0`` was. That is what lets a warm
    start (``x0`` = the previous frequency's solution, as the per-frequency Green's function and
    the RIXS resolvent both use) pay off in iterations rather than in an ever-tighter target --
    the internal reduced system is normalized, so a tolerance applied directly there would scale
    with ``||Y - A x0||`` and silently demand more accuracy the better the warm start was.

    The sparse branch runs on ``ManyBodyBlockState`` (Phase 2 of the block-state matvec plan):
    one shared determinant support per block, ``ManyBodyOperator.apply_block`` matvecs
    (term/sign/accumulator work once per determinant, near-flat in the block width), the
    fused block redistribute, and per-column norms straight off the dense amplitude rows.
    Callers keep the ``list[ManyBodyState]`` interface — conversion happens at this boundary.

    Parameters
    ----------
    A : ManyBodyOperator or ndarray
        The linear operator.
    x0 : list of ManyBodyState or ndarray
        Initial guess block (warm start).
    y : list of ManyBodyState or ndarray
        Right-hand side block.
    basis : Basis
        The many-body state basis object (``None`` for the dense path).
    slaterWeightMin : float
        Slater determinant cutoff weight.
    atol : float, optional
        Residual tolerance relative to ``||Y||``. Default is 1e-8. ``x0`` is returned unrefined
        if it already meets it.
    rtol : float, optional
        Stagnation tolerance on the deflated system's residual, relative to its own initial
        value. Default is 1e-12.

    Returns
    -------
    list of ManyBodyState or ndarray
        The solution block ``X``.
    """
    cdef bint is_arr = is_array(x0)
    cdef Py_ssize_t n = x0.shape[1] if is_arr and len(x0.shape) == 2 else len(x0)
    cdef bint mpi = basis is not None and getattr(basis, "is_distributed", False)
    cdef Py_ssize_t rank
    cdef double y_scale
    cdef double r0_scale
    comm = basis.comm if mpi else None

    if not is_arr and hasattr(A, "set_restrictions"):
        A.set_restrictions(basis.restrictions)

    if is_arr:

        def matmat(v):
            return block_apply(A, v, basis=basis, mpi=mpi, slaterWeightMin=slaterWeightMin)

    else:

        def matmat(v):
            out = A.apply_block(v, slaterWeightMin)
            if mpi:
                out = basis.redistribute_block(out)
            return out

    eye_n = np.eye(n, dtype=complex)

    # Initial residual block R0 = Y - A x0.
    if is_arr:
        Axi = matmat(x0.copy())
        ri = y.copy()
        block_add_scaled(ri, Axi, -eye_n, slaterWeightMin=slaterWeightMin)
        y_cols2 = np.linalg.norm(y, axis=0) ** 2
    else:
        x_blk = ManyBodyBlockState.from_states(list(x0))
        y_blk = ManyBodyBlockState.from_states(list(y))
        Axi = matmat(x_blk)
        ri = block_add_scaled_cy(y_blk, Axi, -eye_n)
        if slaterWeightMin > 0:
            ri.prune_rows(slaterWeightMin)
        basis.add_states(state for state in ri.support_keys(slaterWeightMin) if not basis.contains_local(state))
        y_cols2 = y_blk.col_norm2()
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, y_cols2, op=MPI.SUM)
    y_scale = np.sqrt(np.max(y_cols2))

    # Deflate R0 = Q @ beta_j into rank independent directions (Q = R0 @ beta_inv, orthonormal).
    # Solve the full-rank reduced system A Zq = Q, then reconstruct the correction
    # Z = Zq @ beta_j so that A Z = A Zq @ beta_j = Q @ beta_j = R0, exactly (the dependent
    # directions are recovered by linearity, never solved for). Reuses the same Gram-matrix
    # deflation as the block-Lanczos recurrence (:func:`_cholesky_or_deflate`).
    if is_arr:
        gram = block_inner(ri, ri, mpi=mpi, comm=comm)
    else:
        gram = block_inner_cy(ri, ri)
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)

    # ``gram``'s diagonal holds the squared column norms of R0, so its largest entry is the
    # block's overall scale -- free, no extra reduction.
    r0_scale = np.sqrt(max(np.max(gram.diagonal().real), 0.0))

    # A zero RHS has no scale of its own to measure the residual against; fall back to
    # R0's, which makes `atol` absolute there (and keeps the degenerate y = 0, x0 != 0
    # case from demanding an exactly-zero residual).
    if y_scale == 0.0:
        y_scale = r0_scale

    # Converged before we started: a warm start (or a zero RHS) already meets the tolerance.
    # This test has to live here, explicitly. `_cholesky_or_deflate`'s rank floor is
    # *absolute* (`evals > DEFLATE_EVAL_TOL * max(evals[-1], 1.0)`), which is right for a
    # block-Lanczos residual block but would otherwise silently deflate any R0 with
    # ||R0|| < sqrt(DEFLATE_EVAL_TOL) ~ 6e-6 to rank 0 and return x0 unrefined, whatever
    # `atol` asked for.
    if r0_scale <= atol * y_scale:
        return x0

    # Deflate the *normalized* Gram: the rank of R0 is a property of its column directions,
    # not of its overall scale. beta_j/beta_inv are rescaled to keep Q = R0 @ beta_inv
    # orthonormal and R0 = Q @ beta_j exact.
    beta_j, beta_inv, rank = _cholesky_or_deflate(gram / (r0_scale * r0_scale), n)
    if rank == 0:
        return x0  # zero residual: x0 already solves the system
    beta_j = beta_j * r0_scale
    beta_inv = beta_inv / r0_scale

    if is_arr:
        q_block = block_combine(ri, beta_inv, slaterWeightMin)
    else:
        q_block = ri.combine_columns(beta_inv)
        if slaterWeightMin > 0:
            q_block.prune_rows(slaterWeightMin)

    # The core drives the residual of the *normalized* system A Zq = Q (unit columns) below
    # its tolerance, and the true residual is that times ||R0|| (Z = Zq @ beta_j). Scaling by
    # ||Y||/||R0|| makes the delivered accuracy ``atol * ||Y||`` regardless of the warm start,
    # so a good x0 buys fewer iterations rather than a silently tighter target. A cold start
    # has R0 = Y, hence the ratio is exactly 1 and the tolerance is unchanged.
    z_block = _block_bicgstab_core(
        matmat,
        q_block,
        basis,
        slaterWeightMin,
        atol * (y_scale / r0_scale),
        rtol,
        mpi,
        comm,
        is_arr,
        rank,
        kwargs.get("max_iter", np.inf),
    )
    if is_arr:
        correction = block_combine(z_block, beta_j, slaterWeightMin)
        xi = x0.copy()
        block_add_scaled(xi, correction, eye_n, slaterWeightMin=slaterWeightMin)
        return xi
    correction = z_block.combine_columns(beta_j)
    xi = block_add_scaled_cy(x_blk, correction, eye_n)
    if slaterWeightMin > 0:
        xi.prune_rows(slaterWeightMin)
    return xi.to_states()


def _block_bicgstab_core(
    matmat,
    rhs,
    basis,
    double slaterWeightMin,
    atol,
    rtol,
    bint mpi,
    comm,
    bint is_arr,
    Py_ssize_t n,
    max_iter,
):
    """Block BiCGSTAB inner iteration for a **full-rank** RHS block with a zero initial guess.

    Assumes ``rhs`` has full column rank (guaranteed by the deflation in
    :func:`block_bicgstab`), so the block coefficient systems are well posed. The active-column
    coefficient solves use least squares (robust to any rank loss that still develops as columns
    converge at different rates), and the loop exits only on genuine convergence, ``max_iter``,
    or basis exhaustion -- never discarding accumulated progress on a conditioning number.

    Sparse blocks are ``ManyBodyBlockState``: the axpy updates rebind to fresh union-support
    blocks (``block_add_scaled_cy``) instead of mutating in place, per-column norms come from
    ``col_norm2`` on the shared rows, and the two matvecs per iteration are single
    ``apply_block`` calls.
    """
    cdef int it
    cdef double r0_norm
    cdef double complex wi

    def b_inner(B1, B2):
        if is_arr:
            return block_inner(B1, B2, mpi=mpi, comm=comm)
        res = block_inner_cy(B1, B2)
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res

    def col_norms2(v):
        if is_arr:
            return np.linalg.norm(v, axis=0) ** 2
        norms = v.col_norm2()
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, norms, op=MPI.SUM)
        return norms

    def axpy(target, source, coeffs):
        """target + source @ coeffs as a new (rebound) block; prunes like the old
        in-place block_add_scaled dispatcher did."""
        if is_arr:
            block_add_scaled(target, source, coeffs, slaterWeightMin=slaterWeightMin)
            return target
        out = block_add_scaled_cy(target, source, np.ascontiguousarray(coeffs, dtype=complex))
        if slaterWeightMin > 0:
            out.prune_rows(slaterWeightMin)
        return out

    eye_n = np.eye(n, dtype=complex)
    if is_arr:
        xi = np.zeros_like(rhs)
        ri = rhs.copy()  # A x0 = 0 -> residual is the RHS
        r0_t = ri.copy()
        pi = ri.copy()
    else:
        xi = ManyBodyBlockState.from_states([ManyBodyState() for _ in range(n)])
        ri = rhs  # blocks are never mutated: rebinding replaces copying
        r0_t = rhs
        pi = rhs

    r0_norm = np.sqrt(np.max(col_norms2(r0_t)))
    if r0_norm < np.finfo(float).eps:
        return xi

    cdef double cutoff2 = slaterWeightMin * slaterWeightMin
    if not is_arr:
        # Two width-0 key-only mask blocks track the determinant bookkeeping. Both are
        # sorted C++ key vectors merged in nogil, so nothing here materializes a Python
        # object per determinant of the support (`support_keys` does, once per row).
        #
        # `seen_mask` -- every determinant ever touched, at cutoff 0. Only its *size*
        #   is consumed, by the `it * n < global_seen_size` exhaustion bound below.
        # `offered_mask` -- every determinant ever handed to `basis.add_states`, at
        #   `slaterWeightMin`. Kept separate rather than folded into `seen_mask`: a
        #   determinant can enter the support below the cutoff (so it is seen but not
        #   offered) and grow above it later, and one mask would drop it from the basis
        #   forever. The two coincide only when `slaterWeightMin == 0`.
        seen_mask = ManyBodyBlockState()
        offered_mask = ManyBodyBlockState()
        seen_mask.merge_keys(rhs.keys_new_above(seen_mask, 0.0))
        global_seen_size = np.array([len(seen_mask)], dtype=int)
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)
    else:
        global_seen_size = np.array([np.inf])

    def grow_basis_and_seen(v):
        # Only the determinants this solve has not offered before are materialized as
        # Python objects, so each one costs an allocation at most once over the whole
        # solve instead of once per iteration. `contains_local` (an O(1) dict lookup, not
        # the `in basis.local_basis` list scan) then drops the ones this rank already owns,
        # which is purely a redistribution-payload optimization -- `add_states` dedups too.
        new_offered = v.keys_new_above(offered_mask, cutoff2)
        offered_mask.merge_keys(new_offered)
        basis.add_states(state for state in new_offered.keys() if not basis.contains_local(state))
        seen_mask.merge_keys(v.keys_new_above(seen_mask, 0.0))
        global_seen_size[0] = len(seen_mask)
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, global_seen_size, op=MPI.SUM)

    def masked_lstsq(mat, rhs_mat, active):
        # Solve mat @ X = rhs_mat restricted to active rows/cols (inactive rows of X -> 0),
        # via least squares so a singular active sub-block is handled gracefully.
        out = np.zeros_like(rhs_mat)
        act = np.where(active)[0]
        if act.size:
            out[np.ix_(act, np.arange(out.shape[1]))] = np.linalg.lstsq(
                mat[np.ix_(act, act)], rhs_mat[act, :], rcond=None
            )[0]
        return out

    it = 0
    while it * n < global_seen_size[0] and it < max_iter:
        it += 1

        r_norms = np.sqrt(col_norms2(ri))

        active_mask = (r_norms >= atol) & (r_norms / r0_norm >= rtol)
        if not np.any(active_mask):
            break

        vi = matmat(pi)
        if not is_arr:
            grow_basis_and_seen(vi)

        R0_V = b_inner(r0_t, vi)
        R0_R = b_inner(r0_t, ri)
        R0_R[:, ~active_mask] = 0
        ai = masked_lstsq(R0_V, R0_R, active_mask)

        # s_i = r_i - v_i a_i (rebinding; the dense path updates ri's buffer in place).
        si = axpy(ri, vi, -ai)

        active_mask_s = np.sqrt(col_norms2(si)) >= atol
        if not np.any(active_mask_s):
            xi = axpy(xi, pi, ai)
            break

        ti = matmat(si)
        if not is_arr:
            grow_basis_and_seen(ti)

        if is_arr:
            ts = np.sum(np.conj(ti[:, active_mask_s]) * si[:, active_mask_s], axis=0).sum()
            tt = np.sum(np.conj(ti[:, active_mask_s]) * ti[:, active_mask_s], axis=0).sum()
        else:
            # Diagonals of the block Gram matrices; summed over the active columns
            # (the off-diagonal work is negligible at spectra block widths).
            G_ts = b_inner(ti, si)
            G_tt = b_inner(ti, ti)
            ts = complex(np.sum(G_ts.diagonal()[active_mask_s]))
            tt = complex(np.sum(G_tt.diagonal()[active_mask_s]))

        wi = 0.0 if abs(tt) < np.finfo(float).eps else ts / tt

        # x_{i+1} = x_i + s_i w_i + p_i a_i.
        xi = axpy(xi, si, wi * eye_n)
        xi = axpy(xi, pi, ai)

        R0_T = b_inner(r0_t, ti)
        R0_T[:, ~active_mask_s] = 0
        bi = masked_lstsq(R0_V, -R0_T, active_mask)

        # r_{i+1} = s_i - t_i w_i.
        ri = axpy(si, ti, -wi * eye_n)

        # p_{i+1} = r_{i+1} + (p_i - v_i w_i) b_i (the dense path copies here; the
        # block path rebinds fresh blocks anyway).
        if is_arr:
            pip = ri.copy()
            pip = axpy(pip, pi, bi)
            pi = axpy(pip, vi, -wi * bi)
        else:
            pi = axpy(axpy(ri, pi, bi), vi, -wi * bi)

    return xi
