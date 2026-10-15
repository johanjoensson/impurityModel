# ===========================================================================
# Path-agnostic block primitives + reorthogonalization
# ===========================================================================
# The is_array-dispatching block primitives (block_inner/apply/add_scaled/combine/
# orthogonalize/normalize) shared by the array and ManyBodyState Lanczos kernels, and
# the full reorthogonalization (apply_full_reort).

import scipy.sparse as sps
from impurityModel.ed.ManyBodyUtils import (
    inner_multi,
    SparseKrylovDense,
    ManyBodyState,
    block_inner_cy,
    block_add_scaled_cy,
)

cpdef bint is_array(object V):
    if isinstance(V, (np.ndarray, sps.spmatrix, sps.sparray)):
        return True
    if isinstance(V, list) and len(V) > 0 and isinstance(V[0], np.ndarray):
        return True
    return False

cpdef object block_cols(object Q):
    """Column count of a block, in whatever representation it comes.

    Check array first: a ``ManyBodyState``'s own ``len()`` is its ROW count (dict-like,
    matching ``len(dict)``), not its column count -- silently wrong for "how many Krylov
    columns", not a raise. Called from the restart layer's per-restart bookkeeping (not the
    per-step hot path), so it stays a plain dispatch rather than something perf-sensitive.

    ``is_array`` accepts a ``list[np.ndarray]`` as well as a bare array, and that case has no
    ``.shape`` -- a bare ``Q.shape[1]`` raised ``AttributeError`` on it rather than counting.
    It reached nothing while every caller passed an array, then became reachable from
    ``_block_normalize_failure``, where an ``AttributeError`` *from inside the error
    formatter* replaced the ``ValueError`` the caller was meant to see. Counted the way the
    rest of the module consumes that representation (``block_inner``/``block_tsqr``'s
    ``np.column_stack``): one column per 1-D entry, ``shape[1]`` per 2-D one -- without
    allocating the stacked copy just to read its width.
    """
    cdef Py_ssize_t cols = 0
    if is_array(Q):
        if isinstance(Q, list):
            # An explicit loop, not a generator expression: a closure inside a cpdef function
            # is a Cython compile error.
            for a in Q:
                cols += 1 if a.ndim == 1 else a.shape[1]
            return cols
        return Q.shape[1]
    if isinstance(Q, ManyBodyState):
        return Q.width
    return len(Q)

cpdef object block_inner(object V, object W, bint mpi=False, object comm=None):
    if is_array(V):
        if isinstance(V, list):
            V = np.column_stack(V)
        if isinstance(W, list):
            W = np.column_stack(W)
        res = adjoint_product(V, W)
        if mpi and comm is not None:
            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res
    elif isinstance(V, ManyBodyState):
        # A shared-support block: dispatch to the block-native Gram matrix directly.
        # ``inner_multi`` below assumes list[ManyBodyState] -- ``list(a_block)`` iterates
        # its determinant KEYS, not rows, so it would silently return garbage rather than
        # raise if a block ever reached it.
        res = block_inner_cy(V, W)
        if mpi and comm is not None:
            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res
    else:
        res = inner_multi(V, W)
        if mpi and comm is not None:
            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res


cpdef object block_col_norms2(object V, bint mpi=False, object comm=None):
    """Squared 2-norm of every column of a distributed block, summed over the ranks."""
    if is_array(V):
        norms = np.linalg.norm(V, axis=0) ** 2
    else:
        norms = V.col_norm2()
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, norms, op=MPI.SUM)
    return norms


cpdef object block_diag_inner(object A, object B, bint mpi=False, object comm=None):
    """The diagonal of ``A^H B`` -- ``<A_j|B_j>`` per column -- summed over the ranks.

    O(rows x width), where taking the diagonal of the full Gram matrix is O(rows x width^2)."""
    if is_array(A):
        d = np.sum(np.conj(A) * B, axis=0)
    else:
        d = np.array(np.diagonal(block_inner_cy(A, B)), dtype=complex)  # a copy: the diagonal view is read-only
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, d, op=MPI.SUM)
    return d


def matvec_cut_after_sum(bint redistributes, double cutoff):
    """Whether a sparse matvec cuts ``cutoff`` on its summed rows: exactly when its output is redistributed.

    The output is then summed over the ranks (and row chunks) before the cut, so what survives does not depend
    on the layout; that includes ``cutoff == 0``, where the cut drops the rows whose partials cancel exactly,
    as a serial apply (which never emits an exact-zero row) already does. Without a redistribute the one apply
    already sees whole sums. Depends only on replicated values, so every rank takes the same branch. (The
    ``MATVEC_PRUNE=before_sum`` alternative was retired; see ``doc/gf_support_pipeline.md``.)
    """
    return redistributes


def apply_and_redistribute(
    object H, object V, object basis, double cutoff, bint redistribute, int n_chunks=1, object prof=None,
):
    """``H V`` for a shared-support block, routed onto ``basis``'s owners when ``redistribute``.

    The one recurrence matvec: the block-Lanczos step, and the BiCGSTAB/GMRES matvec
    go through here. The ``cutoff`` row prune is made on each row's summed amplitude, so the result does
    not depend on how ``V``'s rows are spread over the ranks or chunks. Collective when ``redistribute`` is set.

    ``basis`` is either a plain basis (the null policy: redistribute, then cut) or a support policy
    (``gf_primitives._CappedBasisProxy``, recognised by ``decide``), which owns every decision on which rows
    may carry amplitude: it is armed with ``begin_step``, each chunk goes through ``transport``, and the cut
    and the admission both happen in ``decide``, once, on the summed output
    (``doc/gf_support_pipeline.md``). A capping policy must admit on the cut rows, not on the sub-cutoff ones
    the cut is about to drop (which counted toward the cap and froze it early).

    With ``redistribute`` and ``n_chunks > 1``, ``H`` is applied to one chunk of ``V``'s rows at a time and
    redistributed and accumulated before the next, so the raw apply output and its send and receive buffers are
    bounded to chunk size (``APPLY_ROW_CHUNKS``). Exact up to floating-point summation order. ``n_chunks``
    is replicated on every rank, so every rank makes the same number of collective ``redistribute_block``
    calls whatever its own row count: a rank with fewer rows than chunks sends explicit width-p empty
    chunks, never the width-0 polymorphic zero (``doc/lanczos_invariants.md``).

    ``prof(name, t0)``, when given, accumulates the apply, redistribute and total times.
    """
    cdef bint after = matvec_cut_after_sum(redistribute, cutoff)
    cdef object policy = basis if after and hasattr(basis, "decide") else None
    cdef object move = (
        policy.transport if policy is not None else (basis.redistribute_block if redistribute else None)
    )
    cdef double t0 = _time.perf_counter(), t1
    cdef double apply_cutoff = 0.0 if after else cutoff
    cdef object W, part, raw, piece, bounds
    cdef Py_ssize_t idx, lo, hi
    if policy is not None:
        policy.begin_step()
    if not redistribute or n_chunks <= 1:
        W = H.apply_block(V, apply_cutoff)
        if prof is not None:
            prof("matvec_apply", t0)
        t1 = _time.perf_counter()
        if redistribute:
            if hasattr(basis, "redistribute_block"):
                W = move(W)
            else:
                # Duck-typed basis without the block method (e.g. a test mock): fall back to the scalar
                # redistribute through a boundary conversion.
                W = ManyBodyState.from_states(basis.redistribute_psis(*W.to_states()))
        if prof is not None:
            prof("matvec_redistribute", t1)
    else:
        # `bounds[idx], bounds[idx + 1]` rather than the `[:-1]`/`[1:]` slice pair: this module carries
        # `wraparound=False` (CLAUDE.md), which the `x[-1]` idiom trips even on a slice boundary.
        bounds = np.linspace(0, len(V), n_chunks + 1).astype(int)
        W = None
        for idx in range(bounds.shape[0] - 1):
            lo = bounds[idx]
            hi = bounds[idx + 1]
            # `row_slice` allocates exactly this chunk's rows (a `copy()` then `keep_rows` would hold a
            # full-size duplicate of V per chunk, which is why chunking once measured slower with a HIGHER peak).
            part = V.row_slice(lo, hi)
            raw = H.apply_block(part, apply_cutoff)
            del part
            if hasattr(basis, "redistribute_block"):
                piece = move(raw)
            else:
                piece = ManyBodyState.from_states(basis.redistribute_psis(*raw.to_states()))
            del raw
            if W is None:
                W = piece
            elif len(piece):
                # Rank-local decision on a rank-local quantity (no collective inside), so an empty piece
                # on one rank cannot desynchronize the others.
                W += piece
            del piece
        if prof is not None:
            prof("matvec_apply", t0)
    if policy is not None:
        W = policy.decide(W, cutoff)
    elif after:
        W.prune_rows(cutoff)
    if prof is not None:
        prof("matvec", t0)
    return W


cpdef object block_apply(object H, object V, object basis=None, bint mpi=False, double slaterWeightMin=0.0):
    if is_array(V) or isinstance(H, DistributedSparseMatrix):
        V_arr = np.column_stack(V) if isinstance(V, list) and isinstance(V[0], np.ndarray) else V
        if not isinstance(H, DistributedSparseMatrix):
            raise TypeError(f"an array block needs a DistributedSparseMatrix operator, got {type(H).__name__}")
        # One exchange plan per call (collective on a distributed matrix); a kernel that applies H
        # every step holds its own plan instead.
        return H.apply(V_arr)
    elif isinstance(V, ManyBodyState):
        # apply_multi is list-typed (a plain ManyBodyState would raise TypeError
        # there); apply_block is the block-native counterpart, near-flat cost in
        # width vs. apply_multi's linear scaling.
        return apply_and_redistribute(
            H, V, basis, slaterWeightMin, mpi and basis is not None and getattr(basis, "comm", None) is not None
        )
    else:
        redistribute = mpi and basis is not None and basis.comm is not None
        after = matvec_cut_after_sum(redistribute, slaterWeightMin)
        wp = H.apply_multi(V, cutoff=0.0 if after else slaterWeightMin)
        if redistribute:
            wp = basis.redistribute_psis(*wp)
        if after:
            for psi in wp:
                psi.prune(slaterWeightMin)
        return wp

cdef void _gemm_accumulate(double complex[:, ::1] V, double complex[:, ::1] W, double complex[:, ::1] A) noexcept nogil:
    """``V += W @ A`` in place (zgemm with beta = 1): no ``N x p`` temporary."""
    matmul_nogil(V.shape[0], V.shape[1], W.shape[1], 1.0, W, b'N', A, b'N', 1.0, V)


cpdef object block_add_scaled(object V, object W, object alpha, double slaterWeightMin=0.0):
    cdef double complex[:, ::1] Vv, Wv, Av
    if is_array(V):
        if (
            isinstance(V, np.ndarray) and isinstance(W, np.ndarray)
            and V.dtype == np.complex128 and W.dtype == np.complex128
            and V.flags.c_contiguous and V.flags.writeable and V.ndim == 2 and W.ndim == 2
        ):
            Wc = np.ascontiguousarray(W)
            Ac = np.ascontiguousarray(alpha, dtype=complex)
            # The BLAS call reads `alpha` with W's width and V's width and cannot check them; and its
            # output must not overlap its inputs (numpy built the `W @ alpha` temporary first).
            if Ac.ndim != 2 or Ac.shape[0] != Wc.shape[1] or Ac.shape[1] != V.shape[1] or Wc.shape[0] != V.shape[0]:
                raise ValueError(
                    f"block_add_scaled: cannot add ({Wc.shape}) @ ({Ac.shape}) to a block of shape {V.shape}"
                )
            if np.may_share_memory(V, Wc):
                V += Wc @ Ac
                return V
            Vv = V
            Wv = Wc
            Av = Ac
            with nogil:
                _gemm_accumulate(Vv, Wv, Av)
        else:
            V += W @ alpha
    elif isinstance(V, ManyBodyState):
        # A shared-support block is never mutated in place (its union support can outgrow
        # V's own storage, same as every other block_add_scaled_cy caller) -- rebind V to
        # the new block instead. Every current caller of this dispatcher already captures
        # the return value for exactly this reason (see block_add_scaled's callers: the
        # array-only in-place calls are all guarded by an `is_array` check at the call site
        # and never reach here).
        V = block_add_scaled_cy(V, W, alpha)
        if slaterWeightMin > 0:
            V.prune_rows(slaterWeightMin)
    else:
        # Defensive only: every current caller (BiCGSTAB.pyx/GMRES.pyx) is block-native
        # end to end and passes a genuine ManyBodyState; the array-only in-place calls
        # are guarded by the is_array check above.
        raise TypeError(f"block_add_scaled: unsupported V type {type(V)!r}")
    return V

cpdef object block_combine(object Q, object Y, double slaterWeightMin=0.0):
    if is_array(Q):
        if isinstance(Q, list):
            Q = np.column_stack(Q)
        return block_combine_array(Q, Y)
    elif isinstance(Q, SparseKrylovDense):
        # Columnar Krylov store: one zgemm over the dense buffer + scatter directly into
        # ONE block over the union of nonzero rows (combine_block) — every current caller
        # of this branch (selective_orthogonalize) immediately feeds the result into
        # block_inner/block_add_scaled, which need a ManyBodyState, not a
        # list[ManyBodyState] (see combine() vs combine_block() on SparseKrylovDense).
        return Q.combine_block(Y, slaterWeightMin=slaterWeightMin)
    elif isinstance(Q, ManyBodyState):
        # Already a shared-support block: combine_columns is the block-native "self @ Y"
        # (bit-for-bit the same j-ascending accumulation as add_scaled_multi, see
        # test_combine_columns_matches_dense_reference).
        out = Q.combine_columns(Y)
        if slaterWeightMin > 0:
            # slaterWeightMin reaches this path from live TRLM callers (unlike
            # combine_block's caller, which always passes 0.0), so the pruning semantics
            # must be per-column, not per-row. prune_rows is row-granularity (a row
            # survives if ANY column is above cutoff) -- not the same thing, see
            # combine_block's docstring -- so we take the round trip through each output
            # state's own per-column prune instead.
            pruned = out.to_states()
            for st in pruned:
                st.prune(slaterWeightMin)
            out = ManyBodyState.from_states(pruned)
        return out
    else:
        # Defensive only: TRLM's psi0 seed is converted to a ManyBodyState at the
        # entry point before ever reaching this dispatcher, and the store_krylov=False
        # Q_basis shape (list[ManyBodyState]) never reaches block_combine either.
        raise TypeError(f"block_combine: unsupported Q type {type(Q)!r}")

cpdef tuple block_orthogonalize(object wp, object Q, object overlaps=None, bint mpi=False, object comm=None):
    if is_array(wp):
        if isinstance(Q, list):
            Q = np.column_stack(Q)
        return block_orthogonalize_array(wp, Q, overlaps, comm if mpi else None)
    elif isinstance(wp, ManyBodyState):
        # Both operands must be blocks (the caller's responsibility, same as every other
        # block-native pairing in this module, e.g. selective_orthogonalize's ritz_blk/
        # q_next): block_inner/block_add_scaled dispatch on their first argument only, so
        # a stray list Q here would silently reach inner_multi/add_scaled_multi with a
        # ManyBodyState instead of raising.
        if overlaps is None:
            overlaps = block_inner(Q, wp, mpi, comm)
        wp = block_add_scaled(wp, Q, -overlaps)
        return wp, overlaps
    else:
        # Defensive only: see the matching branch in block_combine -- TRLM's psi0
        # is converted to a block at the entry point before reaching this dispatcher.
        raise TypeError(f"block_orthogonalize: unsupported wp type {type(wp)!r}")


class BlockBreakdown(ValueError):
    """The rank code came back non-positive: ``block_normalize`` could not orthonormalize.

    A distinct type because the callers that *recover* from a breakdown (``irlm``'s three
    lock/restart sites, ``cipsi_solver._normalize_start_block``'s cold-start fallback) all sit
    around a **collective** call, and their recovery is only MPI-safe because this particular
    failure is rank-symmetric: it is decided by ``block_tsqr``'s ``k``, which TSQR replicates
    bitwise. A bare ``except ValueError`` there was wider than that argument. ``block_normalize``
    also raises before ever reaching the collective -- ``ManyBodyState.from_states`` rejects a
    width-0 block, and *that* is a rank-local condition (a rank whose local support is empty
    builds the polymorphic zero while its peers do not). Caught, it would send one rank into the
    recovery's own collectives while the others were still inside the first one: an asymmetric
    deadlock, the class CLAUDE.md's MPI rules name.

    Subclasses ``ValueError`` so any handler that predates it keeps working; the point is that
    handlers wrapping a collective can now name the narrower type.
    """


cpdef tuple block_normalize(object wp, bint mpi=False, object comm=None, double slaterWeightMin=0.0):
    """Orthonormalize a block, in whatever representation it comes.

    The columns here are already O(1) (a starting block, a restart block), so the
    breakdown reference is 1 rather than an operator norm; a block that collapses
    raises instead of returning a rank, because its callers have no shrinking-block
    path to fall back on. Absorbs the former block_normalize_array (dense arrays) and
    block_normalize_sparse (ManyBodyState / list[ManyBodyState], via block_tsqr's own
    representation dispatch) into one function -- they were the same 5-line body twice.
    """
    if is_array(wp):
        q_next, beta_j, active_k, sv = tsqr(wp, comm if mpi else None, 1.0)
    else:
        q_next, beta_j, active_k, sv = block_tsqr(wp, mpi, comm, 1.0, slaterWeightMin)
    if active_k <= 0:
        raise BlockBreakdown(_block_normalize_failure(wp, active_k, sv))
    return q_next, beta_j


cdef str _block_normalize_failure(object wp, int active_k, object sv):
    """Spell out *which* of ``tsqr``'s two failure codes fired, and on what block.

    ``tsqr`` distinguishes them deliberately -- ``k == 0`` is a numerically zero block (a
    genuine invariant subspace / closed Krylov space), ``k == -1`` is a **non-finite** factor,
    i.e. a corrupted recurrence upstream -- and this raise used to flatten both into one
    "Block collapsed to zero rank", which sent a production SrMnO3 double-counting crash
    (:mod:`impurityModel.ed.dc_criteria`'s cap ladder, a 262-column warm-started CIPSI start
    block) down the wrong diagnosis for want of one integer.

    ``active_k`` and ``sv`` are replicated (``block_tsqr`` returns the same ``k``/``sv`` on
    every rank), so every rank raises together: this builds a *message*, it never decides
    whether to raise. The row count is rank-local and labelled as such -- read, never branched
    on.
    """
    cdef object rows
    cdef str what = "local rows"
    if is_array(wp):
        # Every entry of a list-of-arrays shares the row partition, so the first one's height is
        # the block's.
        rows = wp[0].shape[0] if isinstance(wp, list) else wp.shape[0]
    elif isinstance(wp, ManyBodyState):
        rows = len(wp)
    else:
        # A list of states does NOT share a support: the block's row count is the union
        # `from_states` would build, and this is only the first column's. Labelled for what it is
        # rather than paying a merge on a failure path -- it is a size cue, and calling it "local
        # rows" made it read as the block's height, which it is not.
        rows = len(wp[0]) if len(wp) > 0 else 0
        what = "determinants in its first column"
    cdef str where = f"width {block_cols(wp)}, {rows} {what}"
    if active_k < 0:
        return (
            f"Block normalization got a non-finite factor ({where}): the block holds NaN/Inf, "
            "so the recurrence feeding it is corrupted -- this is not a closed Krylov space"
        )
    cdef object s_max = None if sv is None or len(sv) == 0 else float(sv[0])
    return (
        f"Block collapsed to zero rank ({where}): largest singular value {s_max} is at or "
        "below the breakdown floor, i.e. the block is numerically zero"
    )


cpdef tuple block_tsqr(object wp, bint mpi=False, object comm=None, double scale=1.0,
                       double slaterWeightMin=0.0, double deflate_tol=-1.0):
    """Orthonormalize a distributed block through TSQR, in whatever representation it comes.

    The single orthonormalization entry point of the Krylov layer: dispatches on ``is_array``
    exactly like :func:`block_normalize` / :func:`block_combine`, so the dense-array kernel,
    the ``ManyBodyState`` kernel and the ``ManyBodyState``-list callers all share one
    factorization. ``Q`` comes back in the same representation ``wp`` arrived in.

    The block representations reach the numerics through their buffer protocol, i.e. without a
    copy: the shared-support ``(rows x width)`` coefficients *are* the tall-skinny matrix. The
    exported view blocks row-structure mutation of ``wp`` while it is alive, so it is dropped
    before this returns (``Q`` is always freshly allocated), and a zero-row block is
    materialized as an empty array because there is no buffer to point at.

    ``deflate_tol`` overrides the rank floor for this block; see
    :func:`impurityModel.ed.TSQR.tsqr`. Structurally rank-deficient seed blocks (the
    spectroscopies' polarization components) pass ``DEFLATE_TOL_SEEDS``.

    **Collective** when ``mpi`` and ``comm`` are given: every rank must call it, and all ranks
    return the same ``beta``, ``k`` and ``sv``.

    Returns
    -------
    tuple
        ``(Q, beta, k, sv)``; ``Q`` is ``None`` when ``k <= 0`` (breakdown / non-finite).
    """
    cdef object src = None       # the block backing the array view, for the write-back
    cdef bint as_list = False
    if isinstance(wp, ManyBodyState):
        src = wp
    elif isinstance(wp, list) and len(wp) > 0 and not isinstance(wp[0], np.ndarray):
        src = ManyBodyState.from_states(list(wp))
        as_list = True

    if src is not None:
        A = np.zeros((0, src.width), dtype=complex) if len(src) == 0 else np.asarray(src)
    elif isinstance(wp, list):
        A = np.column_stack(wp)
    else:
        A = wp

    Q, beta, k, sv = tsqr(A, comm if mpi else None, scale, deflate_tol=deflate_tol)
    if k <= 0 or src is None:
        return Q, beta, k, sv

    Q_block = ManyBodyState.from_keys_and_amps(src, Q)
    if as_list:
        Q_states = Q_block.to_states()
        if slaterWeightMin > 0:
            for st in Q_states:
                st.prune(slaterWeightMin)
        return Q_states, beta, k, sv
    if slaterWeightMin > 0:
        Q_block.prune_rows(slaterWeightMin)
    return Q_block, beta, k, sv


cpdef object apply_full_reort(object wp, object store, bint mpi, object comm):
    """Two-pass classical block Gram-Schmidt of ``wp`` against every column of the Krylov ``store``
    (:class:`SparseKrylovDense` for a ``ManyBodyState`` block, :class:`DenseKrylovStore` for an array).
    Collective on ``comm`` when ``mpi``."""
    if store is None:
        raise ValueError("full reorthogonalization needs the Krylov store")
    wp, _ = store.reort(wp, None, 2, comm if mpi else None)
    return wp


def check_divergence(double beta_norm, double alpha_norm, bint first_step,
                     double t_norm_max, double h_norm_est, int it_abs, object comm):
    """Divergence guard + the ungated print, shared by both kernels' identical call
    sites (``divergence_guard`` above does the actual math). ``first_step`` must be
    computed by the caller in its own loop-variable idiom (MBS: ``it == 0`` on its
    loop-local counter; array: ``it == start_it`` on its absolute counter) -- both
    already mean "the first step executed by this call", not literally the same
    comparison, so do not copy one kernel's comparison into the other."""
    diverged, t_norm_max, h_norm_est = divergence_guard(beta_norm, alpha_norm, first_step, t_norm_max, h_norm_est)
    if diverged:
        if comm is None or comm.Get_rank() == 0:
            # flush=True (from the array kernel's copy): this diagnostic exists to be
            # seen when a run goes wrong, so it must not sit in a stdio buffer behind
            # whatever the rank does next.
            print(
                f"[BlockLanczos] Divergence detected at iteration {it_abs}: "
                f"|beta|={beta_norm:.3e}, |alpha|={alpha_norm:.3e} >> spectral scale "
                f"{h_norm_est:.3e}. Truncating to the last trustworthy block.",
                flush=True,
            )
    return diverged, t_norm_max, h_norm_est


def factor_residual(object wp, bint mpi, object comm, double h_scale, double deflate_tol=-1.0):
    """TSQR-factor the residual block (phase 5), shared by both kernels -- ``block_tsqr``
    under a name that matches the algorithmic step. The three-way active_k
    classification (< 0 non-finite/corrupted, == 0 rank-deficient/exact invariant
    subspace, > 0 normal) stays with each caller: MBS returns from the step function on
    breakdown, the array kernel breaks its own driver loop -- two different idioms for
    the same three outcomes, not unified here."""
    return block_tsqr(wp, mpi, comm, h_scale, deflate_tol=deflate_tol)


def pack_lanczos_result(alphas_out, betas_out, Q_basis, W, block_widths, termination,
                        bint return_widths, bint return_status, bint return_W=True):
    """The return_widths/return_status(/return_W) opt-in arity ladder, shared by both
    kernels' return-tuple assembly. Each caller finishes its own basis representation
    first (MBS substitutes the store_krylov=False two-block tail; the array kernel
    trims its growth buffer) before calling this.

    ``return_W`` defaults to True, matching the MBS kernel's ladder (``W`` is always
    in its base return). The array kernel additionally makes ``W``'s presence itself
    opt-in -- a third independent flag its own pre-hoist ladder already branched on
    -- so it passes ``return_W`` explicitly."""
    if return_widths:
        if return_W:
            if return_status:
                return alphas_out, betas_out, Q_basis, W, block_widths, termination
            return alphas_out, betas_out, Q_basis, W, block_widths
        if return_status:
            return alphas_out, betas_out, Q_basis, block_widths, termination
        return alphas_out, betas_out, Q_basis, block_widths
    else:
        if return_W:
            if return_status:
                return alphas_out, betas_out, Q_basis, W, termination
            return alphas_out, betas_out, Q_basis, W
        if return_status:
            return alphas_out, betas_out, Q_basis, termination
        return alphas_out, betas_out, Q_basis
