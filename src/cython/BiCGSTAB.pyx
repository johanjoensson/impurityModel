# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True, freethreading_compatible=True

"""
BiCGSTAB.pyx
============
Block BiCGSTAB linear solver for the many-body layer.

Solves ``A X = Y`` for a block right-hand side, on the same two representations the
block-Lanczos kernels support: dense ``numpy`` arrays and hash-distributed
``ManyBodyState`` blocks. It is the memory-flat alternative to block Lanczos for
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
    block_add_scaled,
    block_apply,
    block_combine,
    block_inner,
    block_tsqr,
    is_array,
)
from impurityModel.ed.TSQR import DEFLATE_TOL_SEEDS
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    block_add_scaled_cy,
    block_inner_cy,
)


def _fill_info(info, iterations, converged, rel_residual):
    """Populate the caller's optional ``info`` dict (no-op when ``info is None``)."""
    if info is None:
        return
    info["iterations"] = int(iterations)
    info["converged"] = bool(converged)
    info["rel_residual"] = float(rel_residual)


def _make_matmat(A, basis, double slaterWeightMin, bint is_arr, bint mpi):
    """The solver matvec, shared by ``block_bicgstab`` and ``block_gmres``.

    Dense path: :func:`BlockLanczosArray.block_apply`. Sparse path: one
    ``ManyBodyOperator.apply_block`` per call, routed through
    ``basis.redistribute_block`` when distributed **or** when the basis enforces a
    determinant cap (``caps_growth`` -- the per-frequency Green's function's
    ``_CappedBasisProxy`` binds inside ``redistribute_block``, so it must see every
    matvec output even in a serial run, where redistribution itself is a no-op).
    """
    if is_arr:

        def matmat(v):
            return block_apply(A, v, basis=basis, mpi=mpi, slaterWeightMin=slaterWeightMin)

    else:
        caps = getattr(basis, "caps_growth", False)

        def matmat(v):
            out = A.apply_block(v, slaterWeightMin)
            if mpi or caps:
                out = basis.redistribute_block(out)
            return out

    return matmat


class _SupportTracker:
    """Width-0 mask bookkeeping of a sparse solve's determinant support.

    Two sorted C++ key-vector masks, merged in nogil -- nothing here materializes a
    Python object per determinant of the support (``keys()`` does, once per *new*
    offered determinant):

    * ``seen_mask`` -- every determinant ever touched, at cutoff 0. Only its *size*
      is consumed, by the solvers' Krylov-exhaustion bound.
    * ``offered_mask`` -- every determinant ever handed to ``basis.add_states``, at
      ``slaterWeightMin``. Kept separate rather than folded into ``seen_mask``: a
      determinant can enter the support below the cutoff (so it is seen but not
      offered) and grow above it later, and one mask would drop it from the basis
      forever. The two coincide only when ``slaterWeightMin == 0``.

    Shared by ``block_bicgstab`` and ``block_gmres`` (sparse paths only; dense
    callers use no tracker and an infinite exhaustion bound).
    """

    def __init__(self, basis, double slaterWeightMin, bint mpi, comm):
        self.basis = basis
        self.cutoff2 = slaterWeightMin * slaterWeightMin
        self.mpi = mpi
        self.comm = comm
        self.seen_mask = ManyBodyState()
        self.offered_mask = ManyBodyState()
        self.global_seen_size = np.array([0], dtype=int)

    def seed(self, block):
        """Register the right-hand side's support (already in the basis: no add_states)."""
        self.seen_mask.merge_keys(block.keys_new_above(self.seen_mask, 0.0))
        self._reduce_seen()

    def grow(self, v):
        """Register one matvec output: offer new above-cutoff determinants to the basis.

        Only the determinants this solve has not offered before are materialized as
        Python objects, so each one costs an allocation at most once over the whole
        solve instead of once per iteration. ``contains_local`` (an O(1) dict lookup,
        not the ``in basis.local_basis`` list scan) then drops the ones this rank
        already owns, which is purely a redistribution-payload optimization --
        ``add_states`` dedups too.
        """
        new_offered = v.keys_new_above(self.offered_mask, self.cutoff2)
        self.offered_mask.merge_keys(new_offered)
        self.basis.add_states(state for state in new_offered.keys() if not self.basis.contains_local(state))
        self.seen_mask.merge_keys(v.keys_new_above(self.seen_mask, 0.0))
        self._reduce_seen()

    def _reduce_seen(self):
        self.global_seen_size[0] = len(self.seen_mask)
        if self.mpi:
            self.comm.Allreduce(MPI.IN_PLACE, self.global_seen_size, op=MPI.SUM)

    @property
    def seen_size(self):
        """Global count of determinants the solve has touched (the Krylov-dimension bound)."""
        return self.global_seen_size[0]


def block_bicgstab(A, x0, y, basis, double slaterWeightMin, atol=1e-8, rtol=1e-12, max_iter=np.inf, info=None):
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

    The sparse branch runs on ``ManyBodyState`` (Phase 2 of the block-state matvec plan):
    one shared determinant support per block, ``ManyBodyOperator.apply_block`` matvecs
    (term/sign/accumulator work once per determinant, near-flat in the block width), the
    fused block redistribute, and per-column norms straight off the dense amplitude rows.
    Callers pass and receive ``ManyBodyState`` directly -- no conversion at this
    boundary, so a warm-start chain (:func:`~impurityModel.ed.gf_solvers.solve_shifted_block`'s
    restart loop, the BiCGSTAB->GMRES escalation) carries the same block through every
    attempt instead of round-tripping through a list each call.

    Parameters
    ----------
    A : ManyBodyOperator or ndarray
        The linear operator.
    x0 : ManyBodyState or ndarray
        Initial guess block (warm start).
    y : ManyBodyState or ndarray
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
    max_iter : int or float, optional
        Hard bound on the number of BiCGSTAB iterations. The sparse path is additionally
        bounded by basis exhaustion (``it * n < |seen support|``); the dense path has **no**
        other bound, so a dense caller that can stagnate (e.g. a resolvent close to a pole)
        should always pass a finite ``max_iter``. Default ``np.inf``.
    info : dict, optional
        When given, receives the solve's exit state (the reliability contract the
        per-frequency Green's-function driver reports on):

        * ``"iterations"`` -- BiCGSTAB iterations run (0 for an already-converged warm start).
        * ``"converged"`` -- whether every residual column met ``atol`` (``False`` means
          ``max_iter``, basis exhaustion, or ``rtol`` stagnation ended the solve first).
        * ``"rel_residual"`` -- estimated final residual relative to ``||Y||`` (max over
          columns; measured on the deflated system and rescaled, no extra matvec).

    Returns
    -------
    ManyBodyState or ndarray
        The solution block ``X``.
    """
    cdef bint is_arr = is_array(x0)
    cdef Py_ssize_t n = x0.shape[1] if is_arr and len(x0.shape) == 2 else x0.width
    cdef bint mpi = basis is not None and getattr(basis, "is_distributed", False)
    cdef Py_ssize_t rank
    cdef double y_scale
    cdef double r0_scale
    comm = basis.comm if mpi else None

    if not is_arr and hasattr(A, "set_restrictions"):
        A.set_restrictions(basis.restrictions)

    matmat = _make_matmat(A, basis, slaterWeightMin, is_arr, mpi)

    eye_n = np.eye(n, dtype=complex)

    # Initial residual block R0 = Y - A x0.
    if is_arr:
        Axi = matmat(x0.copy())
        ri = y.copy()
        block_add_scaled(ri, Axi, -eye_n, slaterWeightMin=slaterWeightMin)
        y_cols2 = np.linalg.norm(y, axis=0) ** 2
    else:
        x_blk = x0
        y_blk = y
        Axi = matmat(x_blk)
        ri = block_add_scaled_cy(y_blk, Axi, -eye_n)
        if slaterWeightMin > 0:
            ri.prune_rows(slaterWeightMin)
        basis.add_states(state for state in ri.support_keys(slaterWeightMin) if not basis.contains_local(state))
        y_cols2 = y_blk.col_norm2()
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, y_cols2, op=MPI.SUM)
    y_scale = np.sqrt(np.max(y_cols2))

    # Deflate R0 = Q @ beta_j into rank independent directions (Q orthonormal). Solve the
    # full-rank reduced system A Zq = Q, then reconstruct the correction Z = Zq @ beta_j so
    # that A Z = A Zq @ beta_j = Q @ beta_j = R0, exactly (the dependent directions are
    # recovered by linearity, never solved for). Uses the same TSQR and the same deflation
    # policy as the block-Lanczos recurrence (:func:`block_tsqr`).
    #
    # R0's overall scale is the largest of its column norms -- an O(n) reduction, where the
    # Gram this used to read the scale off the diagonal of is O(n^2).
    r0_cols2 = np.linalg.norm(ri, axis=0) ** 2 if is_arr else ri.col_norm2()
    if mpi:
        comm.Allreduce(MPI.IN_PLACE, r0_cols2, op=MPI.SUM)
    r0_scale = np.sqrt(max(np.max(r0_cols2), 0.0))

    # A zero RHS has no scale of its own to measure the residual against; fall back to
    # R0's, which makes `atol` absolute there (and keeps the degenerate y = 0, x0 != 0
    # case from demanding an exactly-zero residual).
    if y_scale == 0.0:
        y_scale = r0_scale

    # Converged before we started: a warm start (or a zero RHS) already meets the tolerance.
    # This test has to live here, explicitly, because "R0 is small" is a statement about
    # convergence and the factorization's breakdown test is a statement about the block being
    # zero against a *reference* -- pass the wrong reference and any R0 below it is silently
    # deflated to rank 0 and x0 returned unrefined, whatever `atol` asked for.
    if r0_scale <= atol * y_scale:
        _fill_info(info, 0, True, r0_scale / y_scale if y_scale > 0.0 else 0.0)
        return x0

    # The rank of R0 is a property of its column directions, not of its overall scale, so the
    # relative rank test needs no normalization; only the breakdown test does, and it takes
    # R0's own scale as its reference. Q comes straight out of the factorization -- no
    # combine against an explicit inverse.
    #
    # DEFLATE_TOL_SEEDS, not the default floor: this solver's production caller is the RIXS
    # intermediate-state resolvent, whose right-hand side is the Cartesian polarization block.
    # Symmetry makes some of those components dependent and this deflation is what removes
    # them -- but they are zero only to within the rounding accumulated while the seeds were
    # built (measured up to 3.5e-11 relative on a *small* benchmark, and growing with the
    # basis). Judged against the default floor such a direction would be retained, and the
    # solve would carry a pure-noise column with sigma_min ~ 1e-11.
    q_block, beta_j, rank, _sv0 = block_tsqr(
        ri, mpi, comm, r0_scale, slaterWeightMin, deflate_tol=DEFLATE_TOL_SEEDS
    )
    if rank == 0:
        _fill_info(info, 0, True, r0_scale / y_scale if y_scale > 0.0 else 0.0)
        return x0  # zero residual: x0 already solves the system
    if rank < 0:
        # Non-finite residual block: the warm start or the operator is corrupted. Report the
        # failure rather than iterating on NaNs.
        _fill_info(info, 0, False, float("nan"))
        return x0

    # The core drives the residual of the *normalized* system A Zq = Q (unit columns) below
    # its tolerance, and the true residual is that times ||R0|| (Z = Zq @ beta_j). Scaling by
    # ||Y||/||R0|| makes the delivered accuracy ``atol * ||Y||`` regardless of the warm start,
    # so a good x0 buys fewer iterations rather than a silently tighter target. A cold start
    # has R0 = Y, hence the ratio is exactly 1 and the tolerance is unchanged.
    z_block, core_iters, core_norms, core_converged = _block_bicgstab_core(
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
        max_iter,
    )
    # The core's residual columns live on the normalized system A Zq = Q; the true residual is
    # R_core @ beta_j (see the deflation comment above), so ||R0|| ~ r0_scale rescales it back
    # and y_scale makes it relative to the RHS -- an estimate (sigma_max(beta_j) can exceed
    # r0_scale by up to sqrt(n)), consistent with the atol the core was driven to.
    _fill_info(info, core_iters, core_converged, float(np.max(core_norms)) * (r0_scale / y_scale))
    if is_arr:
        correction = block_combine(z_block, beta_j, slaterWeightMin)
        xi = x0.copy()
        block_add_scaled(xi, correction, eye_n, slaterWeightMin=slaterWeightMin)
        return xi
    correction = z_block.combine_columns(beta_j)
    xi = block_add_scaled_cy(x_blk, correction, eye_n)
    if slaterWeightMin > 0:
        xi.prune_rows(slaterWeightMin)
    return xi


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

    Sparse blocks are ``ManyBodyState``: the axpy updates rebind to fresh union-support
    blocks (``block_add_scaled_cy``) instead of mutating in place, per-column norms come from
    ``col_norm2`` on the shared rows, and the two matvecs per iteration are single
    ``apply_block`` calls.

    Returns
    -------
    tuple
        ``(xi, iterations, exit_norms, converged)`` -- the solution block, the iteration
        count, the final per-column residual norms (on this normalized system), and whether
        every column met ``atol`` (``False`` on ``max_iter``, basis exhaustion, or ``rtol``
        stagnation). ``exit_norms`` is exact at every exit: the two mid-loop breaks capture
        the norms they just measured, and a while-condition exit recomputes them (one extra
        reduction, no matvec) because ``ri`` was updated after the loop-top measurement.
    """
    cdef int it
    cdef double r0_norm
    cdef double complex wi
    cdef bint broke = False

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
        xi = ManyBodyState(width=n)
        ri = rhs  # blocks are never mutated: rebinding replaces copying
        r0_t = rhs
        pi = rhs

    r0_cols = np.sqrt(col_norms2(r0_t))
    r0_norm = np.max(r0_cols)
    last_norms = r0_cols
    if r0_norm < np.finfo(float).eps:
        return xi, 0, last_norms, True

    # Determinant bookkeeping (sparse path): the shared width-0 mask tracker feeds the
    # `it * n < seen_size` Krylov-exhaustion bound and offers newly discovered
    # determinants to the basis. Dense solves have no support to track and an infinite
    # exhaustion bound.
    if not is_arr:
        tracker = _SupportTracker(basis, slaterWeightMin, mpi, comm)
        tracker.seed(rhs)
        global_seen_size = tracker.global_seen_size
    else:
        tracker = None
        global_seen_size = np.array([np.inf])

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
        last_norms = r_norms

        active_mask = (r_norms >= atol) & (r_norms / r0_norm >= rtol)
        if not np.any(active_mask):
            broke = True
            break

        vi = matmat(pi)
        if not is_arr:
            tracker.grow(vi)

        R0_V = b_inner(r0_t, vi)
        R0_R = b_inner(r0_t, ri)
        R0_R[:, ~active_mask] = 0
        ai = masked_lstsq(R0_V, R0_R, active_mask)

        # s_i = r_i - v_i a_i (rebinding; the dense path updates ri's buffer in place).
        si = axpy(ri, vi, -ai)

        s_norms = np.sqrt(col_norms2(si))
        active_mask_s = s_norms >= atol
        if not np.any(active_mask_s):
            xi = axpy(xi, pi, ai)
            last_norms = s_norms
            broke = True
            break

        ti = matmat(si)
        if not is_arr:
            tracker.grow(ti)

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

    # A while-condition exit (max_iter / basis exhaustion) leaves `last_norms` one residual
    # update stale (ri changed at the end of the final iteration). Recompute -- one reduction,
    # no matvec, and collective-consistent: every rank exits the loop the same way because
    # `it`, `max_iter` and `global_seen_size` are replicated.
    if not broke:
        last_norms = np.sqrt(col_norms2(ri))

    return xi, it, last_norms, bool(np.all(last_norms < atol))
