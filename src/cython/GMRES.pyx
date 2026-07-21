# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True, freethreading_compatible=True

"""
GMRES.pyx
=========
Restarted block-GMRES (block Arnoldi) linear solver for the many-body layer.

The optimal-residual companion to :mod:`BiCGSTAB`: the same entry contract (dense
``numpy`` and hash-distributed ``ManyBodyBlockState`` blocks, rank-deficient right-hand
sides deflated and reconstructed by linearity, ``atol`` relative to ``||Y||``), but the
residual is *minimized* over the Krylov space at every step -- monotone, no shadow
residual, no breakdown modes. That makes it the fallback for the per-frequency resolvent
solves where BiCGSTAB's shadow-residual recurrence stagnates (real-axis points within
~``delta`` of a pole).

It is **not** memory-flat: each Arnoldi step retains one Krylov block, so a restart
cycle holds up to ``restart + 3`` live blocks. The fallback role is deliberate --
BiCGSTAB (7 live blocks) stays the workhorse; GMRES buys robustness on the few flagged
points.

``impurityModel.ed.gmres`` re-exports the public entry point, mirroring the
``cg.py`` / ``irlm.py`` / ``trlm.py`` thin-wrapper arrangement.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.BiCGSTAB import _fill_info, _make_matmat, _SupportTracker
from impurityModel.ed.BlockLanczosArray import (
    BREAKDOWN_TOL,
    block_add_scaled,
    block_combine,
    block_inner,
    block_tsqr,
    is_array,
)
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyBlockState,
    block_add_scaled_cy,
    block_inner_cy,
)

# Restarted GMRES on an indefinite spectrum can stall cycle-over-cycle (the restart
# forgets the subspace), and a stalled fallback should report failure instead of burning
# its full max_restarts budget. A cycle "makes progress" when it shrinks the true
# residual by at least 1%; two consecutive progress-free cycles end the solve. Two, not
# one: restarted GMRES legitimately alternates slow and fast cycles on such spectra, and
# a single slow cycle is not yet stagnation.
_GMRES_CYCLE_PROGRESS = 0.99
_GMRES_STALL_CYCLES = 2


def block_gmres(A, x0, y, basis, double slaterWeightMin, atol=1e-8, restart=40, max_restarts=25, info=None):
    """
    Solve ``A X = Y`` with restarted block GMRES (block Arnoldi + least squares).

    Contract-compatible with :func:`BiCGSTAB.block_bicgstab`: dense (``numpy``) and
    sparse (``ManyBodyState`` list) blocks, arbitrary -- possibly rank-deficient --
    right-hand sides (the residual block is deflated to its independent directions and
    dependent columns are reconstructed by linearity), warm starts pay off in iterations
    (``atol`` is relative to ``||Y||``, never to the warm-start residual), and an
    optional ``info`` dict receives the exit state.

    Each restart cycle deflates the current true residual, runs up to ``restart`` block
    Arnoldi steps (modified Gram-Schmidt against all retained blocks; rank loss in a new
    block *deflates the block width* going forward, width 0 is the happy breakdown --
    the Krylov space closed and the projected solution is exact), monitors the
    per-column residual of the projected least-squares system every step, and updates
    ``X`` from the minimizer. The true residual is recomputed once per cycle (one
    matvec), so the reported ``rel_residual`` is measured, not estimated.

    Parameters
    ----------
    A : ManyBodyOperator or ndarray
        The linear operator.
    x0 : list of ManyBodyState or ndarray
        Initial guess block (warm start).
    y : list of ManyBodyState or ndarray
        Right-hand side block.
    basis : Basis
        The many-body state basis object (``None`` for the dense path). A
        ``caps_growth`` basis (``_CappedBasisProxy``) bounds the solve's support exactly
        as it does for BiCGSTAB: post-freeze the solve is exact GMRES of ``P H P``.
    slaterWeightMin : float
        Slater determinant cutoff weight.
    atol : float, optional
        Residual tolerance relative to ``||Y||``. Default 1e-8.
    restart : int, optional
        Block Arnoldi steps per cycle (the Krylov blocks retained simultaneously).
        Default 40.
    max_restarts : int, optional
        Restart cycles before giving up (progress-gated: a cycle that fails to shrink
        the residual by :data:`_GMRES_CYCLE_PROGRESS` ends the solve early). Default 25.
    info : dict, optional
        Receives ``{"iterations", "converged", "rel_residual"}`` -- Arnoldi steps run,
        whether every column met ``atol``, and the final *measured* residual relative
        to ``||Y||``.

    Returns
    -------
    list of ManyBodyState or ndarray
        The solution block ``X``.
    """
    cdef bint is_arr = is_array(x0)
    cdef Py_ssize_t n = x0.shape[1] if is_arr and len(x0.shape) == 2 else len(x0)
    cdef bint mpi = basis is not None and getattr(basis, "is_distributed", False)
    cdef int total_iters = 0
    cdef int cycle
    cdef Py_ssize_t j, i, rank_r, w_next
    comm = basis.comm if mpi else None

    if not is_arr and hasattr(A, "set_restrictions"):
        A.set_restrictions(basis.restrictions)

    matmat = _make_matmat(A, basis, slaterWeightMin, is_arr, mpi)
    tracker = None if is_arr else _SupportTracker(basis, slaterWeightMin, mpi, comm)

    # --- small representation-dispatch helpers (the BiCGSTAB core's conventions) -----
    def b_inner(B1, B2):
        if is_arr:
            return block_inner(B1, B2, mpi=mpi, comm=comm)
        res = block_inner_cy(B1, B2)
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res

    def axpy(target, source, coeffs):
        """target + source @ coeffs (dense: in place; sparse: rebound block)."""
        if is_arr:
            block_add_scaled(target, source, coeffs, slaterWeightMin=slaterWeightMin)
            return target
        out = block_add_scaled_cy(target, source, np.ascontiguousarray(coeffs, dtype=complex))
        if slaterWeightMin > 0:
            out.prune_rows(slaterWeightMin)
        return out

    def combine(block, coeffs):
        if is_arr:
            return block_combine(block, coeffs, slaterWeightMin)
        out = block.combine_columns(np.ascontiguousarray(coeffs, dtype=complex))
        if slaterWeightMin > 0:
            out.prune_rows(slaterWeightMin)
        return out

    def residual(xi):
        """R = Y - A X (one matvec) -- the measured true residual."""
        Ax = matmat(xi)
        if is_arr:
            R = y.copy()
            block_add_scaled(R, Ax, -np.eye(n, dtype=complex), slaterWeightMin=slaterWeightMin)
            return R
        if tracker is not None:
            tracker.grow(Ax)
        R = block_add_scaled_cy(y_blk, Ax, -np.eye(n, dtype=complex))
        if slaterWeightMin > 0:
            R.prune_rows(slaterWeightMin)
        return R

    # --- entry: scales and the warm-start early exit (block_bicgstab's semantics) ----
    if is_arr:
        xi = x0.copy()
        y_cols2 = np.linalg.norm(y, axis=0) ** 2
    else:
        xi = ManyBodyBlockState.from_states(list(x0))
        y_blk = ManyBodyBlockState.from_states(list(y))
        y_cols2 = y_blk.col_norm2()
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, y_cols2, op=MPI.SUM)
        tracker.seed(y_blk)
    y_scale = np.sqrt(np.max(y_cols2))

    R = residual(xi)

    prev_res = np.inf
    stalled = 0
    converged = False
    rel_res = np.inf
    for cycle in range(max_restarts):
        # The cycle's scale is the largest column norm of R -- an O(n) reduction, where the
        # Gram it used to be read off the diagonal of is O(n^2).
        r_cols2 = np.linalg.norm(R, axis=0) ** 2 if is_arr else R.col_norm2()
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, r_cols2, op=MPI.SUM)
        r_scale = np.sqrt(max(np.max(r_cols2), 0.0))
        scale = y_scale if y_scale > 0.0 else r_scale
        rel_res = r_scale / scale if scale > 0.0 else 0.0
        if r_scale <= atol * scale:
            converged = True
            break
        # Stall gate (see _GMRES_CYCLE_PROGRESS): report honestly instead of spending
        # the remaining budget on cycles that no longer move the residual.
        if r_scale > _GMRES_CYCLE_PROGRESS * prev_res:
            stalled += 1
            if stalled >= _GMRES_STALL_CYCLES:
                break
        else:
            stalled = 0
        prev_res = min(prev_res, r_scale)

        # Rank is a property of the directions, not the scale, so only the breakdown test
        # needs R's scale as its reference; beta_j reconstructs R = V_1 @ beta_j so dependent
        # columns are recovered by linearity, exactly as in block_bicgstab's entry.
        V_1, beta_j, rank_r, _sv_r = block_tsqr(R, mpi, comm, r_scale, slaterWeightMin)
        if rank_r == 0:
            converged = True
            break
        if rank_r < 0:
            break  # non-finite residual block: stop and report the last honest residual

        V = [V_1]
        widths = [rank_r]
        col_off = [0]
        # Padded block Hessenberg: rows/cols indexed by the cumulative block widths.
        max_cols = restart * rank_r
        Hbar = np.zeros((max_cols + rank_r, max_cols), dtype=complex)
        # RHS of the projected system: ||R|| enters through beta_j at reconstruction,
        # so the projected system solves A Z = V_1 (unit columns): E1 = I.
        e1 = np.zeros((max_cols + rank_r, rank_r), dtype=complex)
        e1[:rank_r, :rank_r] = np.eye(rank_r)
        # The core tolerance on the normalized projected system: delivered accuracy is
        # atol * ||Y|| regardless of the warm start (cold start: ratio exactly 1).
        atol_core = atol * (scale / r_scale)

        y_sol = None
        for j in range(restart):
            W = matmat(V[j])
            if tracker is not None:
                tracker.grow(W)
            total_iters += 1
            # Block modified Gram-Schmidt against every retained block.
            for i in range(j + 1):
                Hij = b_inner(V[i], W)
                Hbar[col_off[i] : col_off[i] + widths[i], col_off[j] : col_off[j] + widths[j]] = Hij
                W = axpy(W, V[i], -Hij)
            w_cols2 = np.linalg.norm(W, axis=0) ** 2 if is_arr else W.col_norm2()
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, w_cols2, op=MPI.SUM)
            w_scale = np.sqrt(max(np.max(w_cols2), 0.0))
            # Two questions, two scales: the rank of W's *directions* is relative and the
            # factorization judges it on its own singular values, but "is W zero" (the
            # Arnoldi space closed) is an absolute question needing a reference -- here the
            # magnitude of this step's Hessenberg column, ~||A V_j||. That reference has to
            # be applied here rather than handed to the factorization, because W's own scale
            # is what it would otherwise be measured against, and a numerically-zero W still
            # has full-rank directions made of roundoff noise.
            h_scale = np.linalg.norm(Hbar[: col_off[j] + widths[j], col_off[j] : col_off[j] + widths[j]])
            V_next = None
            if w_scale > BREAKDOWN_TOL * max(h_scale, w_scale):
                V_next, bj, w_next, _sv_w = block_tsqr(W, mpi, comm, w_scale, slaterWeightMin)
                if w_next < 0:
                    w_next = 0  # non-finite: treat as a closed space and stop this cycle
            else:
                w_next = 0
            row = col_off[j] + widths[j]
            if w_next > 0:
                Hbar[row : row + w_next, col_off[j] : col_off[j] + widths[j]] = bj
            n_cols = col_off[j] + widths[j]
            y_sol, res_cols = _hessenberg_lstsq(Hbar, e1, n_cols, row + w_next)
            if w_next == 0:
                # Happy breakdown: the block Krylov space closed under A, the projected
                # solution is exact on it.
                break
            if np.max(res_cols) <= atol_core:
                break
            # No explicit Krylov-exhaustion bound (unlike the BiCGSTAB core): a closed
            # space announces itself through the breakdown test one step later (W
            # deflates to zero after MGS), which is exact rather than one step early.
            V.append(V_next)
            widths.append(w_next)
            col_off.append(n_cols)

        # X += sum_j V_j @ (y_j @ beta_j) -- the correction mapped back through the
        # entry deflation (dependent residual columns reconstructed by linearity).
        if y_sol is not None:
            for j in range(len(widths)):
                rows = y_sol[col_off[j] : col_off[j] + widths[j], :]
                if rows.size == 0 or j >= len(V):
                    continue
                xi = axpy(xi, V[j], rows @ beta_j)
        V = None
        R = residual(xi)

    else:
        # max_restarts exhausted: measure the final residual for the info record.
        r_cols2 = np.linalg.norm(R, axis=0) ** 2 if is_arr else R.col_norm2()
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, r_cols2, op=MPI.SUM)
        r_scale = np.sqrt(max(np.max(r_cols2), 0.0))
        scale = y_scale if y_scale > 0.0 else r_scale
        rel_res = r_scale / scale if scale > 0.0 else 0.0
        converged = r_scale <= atol * scale

    _fill_info(info, total_iters, converged, rel_res)
    return xi if is_arr else xi.to_states()


def _hessenberg_lstsq(Hbar, e1, Py_ssize_t n_cols, Py_ssize_t n_rows):
    """Least-squares solve of the (padded) block Hessenberg system.

    Returns ``(y, res_cols)``: the minimizer of ``||e1 - Hbar y||`` restricted to the
    first ``n_cols`` columns / ``n_rows`` rows, and the per-column norms of its
    residual on the projected system -- the quantity the per-step convergence test
    compares against the rescaled ``atol``. Sizes are ``(restart+1)p x restart*p`` at
    most (~205 x 200), so a dense ``lstsq`` per Arnoldi step is negligible next to one
    many-body matvec.
    """
    H = Hbar[:n_rows, :n_cols]
    rhs = e1[:n_rows, :]
    y, *_ = np.linalg.lstsq(H, rhs, rcond=None)
    res = rhs - H @ y
    return y, np.linalg.norm(res, axis=0)
