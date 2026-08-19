# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True, freethreading_compatible=True
# Permissive on purpose, unlike BlockLanczos.pyx -- see "Compiler directives" in
# doc/lanczos_invariants.md.

"""Block Lanczos kernel for numpy arrays / scipy sparse operators (MPI row-block).

The shared numerical layer (``Reort``, every tolerance constant, the Paige-Simon
``estimate_orthonormality`` estimator, the block-tridiagonal eigensolvers, and the
representation-dispatching block primitives / ``apply_reort``) lives in
``BlockLanczosCore.pyx``, imported here as a re-export block so every existing
``from impurityModel.ed.BlockLanczosArray import ...`` site keeps resolving. The
ManyBodyState kernel in ``BlockLanczos.pyx`` imports the same names directly from
``BlockLanczosCore``, not from here -- both kernels import downward from the shared
layer, never from each other. Both build a block-tridiagonal ``T`` (diagonal blocks
``alpha``, off-diagonal blocks ``beta``) whose eigenpairs approximate those of the
operator; the TRLM/IRLM drivers restart it.

Reorthogonalization modes (``Reort``), what each guarantees:

* ``NONE``      — no reorthogonalization. Cheapest; orthogonality is allowed to
  decay and ghost (spurious duplicate) eigenvalues may appear. Eigenvalues still
  converge for well-separated spectra. The restart machinery in TRLM/IRLM
  re-orthogonalizes regardless, so NONE only relaxes the *inner* recurrence.
* ``FULL``      — double-pass Gram–Schmidt against the whole basis every step.
  Most robust; ``‖QᴴQ − I‖`` stays at ~machine precision.
* ``PERIODIC``  — FULL, but only every ``REORT_PERIOD`` iterations.
* ``PARTIAL``   — partial reorthogonalization (PRO, Simon): the W-recurrence
  ``estimate_orthonormality`` cheaply *estimates* loss of orthogonality; a full
  reorth against the offending blocks fires only when an estimate exceeds
  ``REORT_TOL``. Matches FULL accuracy at lower cost; the default.
* ``SELECTIVE`` — PARTIAL plus periodic locking of converged Ritz vectors
  (gated to a ``REORT_PERIOD`` cadence; see ``block_lanczos_array_cy``).

Every mode retains all ``p * m`` Krylov columns. That is the dominant allocation of a
long Green's-function run, and it cannot be compressed away: reorthogonalization needs
the *converged Ritz vectors*, and a Ritz vector converging at step ``m`` has weight on
every block, including any an earlier compression would have discarded. Bounding the
retained basis therefore needs secondary storage, not a smaller basis — see the
"Ritz compression is unsound" section of ``doc/plans/blocklanczos_reort_memory.md``.

Module thresholds (all derived from machine ``eps``; see definitions below):
``REORT_TOL`` (loss-of-orthogonality trigger), ``BAD_BLOCK_TOL`` (which blocks to
reorth against), ``DEFLATE_TOL`` (relative rank floor on the block's singular values),
``BREAKDOWN_TOL`` (invariant-subspace detection), ``REORT_PERIOD`` (cadence). The
deflation floors are defined in the ``TSQR`` module that applies them.

Orthonormalization goes through ``TSQR`` (``tsqr`` here, ``block_tsqr`` in
``BlockLanczosCore.pyx``'s ``_block_ops.pxi`` for the other block representations): the
residual block's triangular factor is computed from the block itself, never from the
Gram matrix ``Wp^H Wp``, which is what makes it stable at any conditioning and its
singular values trustworthy.

Deflation policy: when a Lanczos block is rank-deficient (a singular value below the
``DEFLATE_TOL`` floor), the block size shrinks (EA16 shrinking-block, Meerbergen & Scott,
RAL-TR-2000-011) rather than zero-padding or terminating, so ``beta`` becomes rectangular
and ``T`` carries variable-size blocks; the recurrence keeps converging.
"""

cimport cython
import numpy as np
cimport numpy as np
import scipy.linalg as la
import scipy.sparse as sps

from scipy.linalg.cython_blas cimport zgemm

from mpi4py import MPI

# Shared numerical layer: Reort, every tolerance constant, the Paige-Simon estimator,
# the block-tridiagonal eigensolvers, and the representation-dispatching block
# primitives all now live in BlockLanczosCore.pyx -- re-exported here so every existing
# `from impurityModel.ed.BlockLanczosArray import ...` site (production and test) keeps
# resolving unchanged. Put new imports needed only by THIS module's own code above this
# block; put anything that belongs to the shared layer in BlockLanczosCore.pyx instead.
from impurityModel.ed.BlockLanczosCore import (  # noqa: F401
    Reort,
    REORT_TOL,
    BAD_BLOCK_TOL,
    RITZ_BATCH,
    BETA_BLOWUP_FACTOR,
    REORT_PERIOD,
    RESTART_ORTH_TOL,
    DEFAULT_EIGEN_TOL,
    THERMAL_GS_RESIDUAL_TOL,
    get_reort_profile,
    reset_reort_profile,
    enable_reort_profile,
    resolve_reort,
    divergence_guard,
    estimate_orthonormality,
    _build_full_T,
    _build_banded_lower,
    eigh_block_tridiagonal,
    eigsh,
    is_array,
    block_inner,
    block_apply,
    block_add_scaled,
    block_combine,
    block_orthogonalize,
    block_normalize,
    block_tsqr,
    selective_orthogonalize,
    apply_reort,
    block_combine_array,
    block_orthogonalize_array,
)

# The machine constant and the deflation floors are defined once, in the TSQR leaf that
# applies them; a second literal here would be a second source of truth. ``DEFLATE_TOL``
# itself is only read by callers, who import it straight from ``TSQR`` (cython-lint has no
# noqa escape for a pure re-export).
from impurityModel.ed.TSQR import (
    EPS,
    DEFLATE_EVAL_TOL,
    BREAKDOWN_TOL,
    tsqr,
)


def calculate_thermal_gs(h, block_size, e_max, v0=None, reort=Reort.FULL, comm=None, verbose=False):
    mpi = comm is not None
    rank = comm.rank if mpi else 0
    size = comm.size if mpi else 1

    def converged(alphas, betas, *args, **kwargs):
        e, s = eigsh(alphas, betas, de=e_max, select="m")
        sorted_indices = np.argsort(e)
        e = e[sorted_indices]
        s = s[:, sorted_indices]
        mask = e - np.min(e) <= e_max
        return np.linalg.norm(betas[-1] @ s[-block_size:, mask], ord=2) < THERMAL_GS_RESIDUAL_TOL

    if v0 is None:
        v0 = np.random.rand(h.shape[1], block_size) + 1j * np.random.rand(h.shape[1], block_size)
        if mpi:
            counts = np.empty((size), dtype=int)
            comm.Allgather(np.array([v0.size], dtype=int), counts)
            offsets = np.array([np.sum(counts[:r]) for r in range(size)], dtype=int)
            v0_full = np.empty((h.shape[0], block_size), dtype=complex, order="C") if rank == 0 else None
            comm.Gatherv(v0, (v0_full, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
            if rank == 0:
                v0_full, _ = la.qr(v0_full, mode="economic", overwrite_a=True, check_finite=False)
            comm.Scatterv((v0_full, counts, offsets, MPI.C_DOUBLE_COMPLEX), v0, root=0)
        else:
            v0, _ = la.qr(v0, mode="economic", overwrite_a=True, check_finite=False)
    elif v0.shape[1] < block_size:
        new_v0 = np.random.rand(v0.shape[0], block_size - v0.shape[1]) + 1j * np.random.rand(
            v0.shape[0], block_size - v0.shape[1]
        )
        new_v0 -= v0 @ np.conj(v0.T) @ new_v0
        new_v0, _ = la.qr(new_v0, mode="economic", overwrite_a=True, check_finite=False)
        v0 = np.append(v0, new_v0, axis=1)
    elif v0.shape[1] > block_size:
        v0 = v0[:, :block_size]

    alphas, betas, Q, *_ = block_lanczos_array_cy(
        psi0=v0,
        h_op=h,
        converged=converged,
        verbose=verbose,
        reort=reort,
        comm=comm,
    )
    if rank == 0:
        eigvals, eigvecs = eigsh(alphas, betas, de=e_max, Q=Q[:, : alphas.shape[0] * alphas.shape[1]], select="m")
    else:
        eigvals = None
        eigvecs = None
    if mpi:
        eigvals = comm.bcast(eigvals, root=0)
        eigvecs = comm.bcast(eigvecs, root=0)
    return eigvals, eigvecs


cdef extern from "complex.h":
    double complex conj(double complex z) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void matmul_nogil(
    int m, int n, int k,
    double complex alpha,
    double complex[:, ::1] A, char transA,
    double complex[:, ::1] B, char transB,
    double complex beta,
    double complex[:, ::1] C
) noexcept nogil:
    # C (m x n, row-major) = beta*C + alpha * opA(A) (m x k) @ opB(B) (k x n),
    # where op = conjugate-transpose when the trans flag is b'C'.
    # Implemented with BLAS-3 zgemm. BLAS is column-major; our arrays are C-contiguous
    # (row-major), so the row-major product C = opA(A) @ opB(B) is obtained by asking
    # BLAS for the column-major Cᵀ = opB(B)ᵀ @ opA(A)ᵀ: swap the operands (B then A),
    # swap (m, n) -> (n, m), keep the same trans flags, and use the physical row
    # lengths (A.shape[1], B.shape[1], n) as the leading dimensions. Validated against
    # numpy @ for all four trans combinations in test_zgemm_matmul (1e-13).
    cdef int i, j
    cdef int lda, ldb, ldc
    if m <= 0 or n <= 0:
        return
    if k <= 0:
        # No contraction: C = beta * C (zgemm with K=0 would still need valid A/B ptrs).
        if beta == 0.0:
            for i in range(m):
                for j in range(n):
                    C[i, j] = 0.0
        elif beta != 1.0:
            for i in range(m):
                for j in range(n):
                    C[i, j] = C[i, j] * beta
        return
    lda = A.shape[1]
    ldb = B.shape[1]
    ldc = n
    zgemm(&transB, &transA, &n, &m, &k, &alpha, &B[0, 0], &ldb, &A[0, 0], &lda, &beta, &C[0, 0], &ldc)


def _matmul_nogil_test(A, int transA, B, int transB, alpha, beta, C, int m, int n, int k):
    """Test-only Python wrapper around matmul_nogil (pass transA/transB as ord('N'/'C'))."""
    cdef double complex[:, ::1] Av = np.ascontiguousarray(A, dtype=complex)
    cdef double complex[:, ::1] Bv = np.ascontiguousarray(B, dtype=complex)
    cdef double complex[:, ::1] Cv = np.ascontiguousarray(C, dtype=complex)
    cdef char ta = transA
    cdef char tb = transB
    cdef double complex al = alpha
    cdef double complex be = beta
    matmul_nogil(m, n, k, al, Av, ta, Bv, tb, be, Cv)
    return np.asarray(Cv)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply_sparse_csr_nogil(
    int num_rows,
    int num_cols,
    int p,
    double complex[:] data,
    long[:] indices,
    long[:] indptr,
    double complex[:, ::1] X,
    double complex[:, ::1] Y
) noexcept nogil:
    cdef int i, j, k
    cdef long row_start, row_end
    cdef double complex val
    for i in range(num_rows):
        row_start = indptr[i]
        row_end = indptr[i+1]
        for k in range(p):
            Y[i, k] = 0.0
        for j in range(row_start, row_end):
            val = data[j]
            for k in range(p):
                Y[i, k] = Y[i, k] + val * X[indices[j], k]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply_dense_nogil(
    int M, int K, int p,
    double complex[:, ::1] H,
    double complex[:, ::1] X,
    double complex[:, ::1] Y
) noexcept nogil:
    matmul_nogil(M, p, K, 1.0, H, b'N', X, b'N', 0.0, Y)




def _cholesky_or_deflate(M, p_in, double scale=1.0):
    """Superseded by TSQR.tsqr, which factors the block directly and is stable at
    conditioning this Gram-based QR cannot survive; no production path calls this any
    more. Kept as the reference implementation the CholeskyQR2-era regression tests are
    written against (test_block_lanczos_blowup.py, test_tsqr.py). The
    deflation-vs-breakdown derivation this docstring used to carry moves to
    doc/lanczos_invariants.md (R2).
    """
    cdef double breakdown_lam = (BREAKDOWN_TOL * scale) ** 2
    # Try Cholesky first (fast path). diag(L)**2 are the (real, positive) Cholesky
    # pivots, an O(EPS)-faithful surrogate for the eigenvalues of the Hermitian-PD M.
    try:
        L = la.cholesky(M, lower=True)
        d2 = np.square(np.diag(L).real)
        d2_max = float(np.max(d2)) if d2.size else 0.0
        if d2_max <= breakdown_lam:                        # ||beta||_2 <= BREAKDOWN_TOL * scale
            return None, None, 0
        if np.any(d2 < DEFLATE_EVAL_TOL * d2_max):
            raise la.LinAlgError("Numeric singularity in Cholesky diagonal")
        beta_j = np.conj(L.T)
        beta_inv = la.inv(beta_j)
        p_next = p_in
        return beta_j, beta_inv, p_next
    except (la.LinAlgError, ValueError):
        # Fall back to eigh, using the same two tests as the fast path.
        evals, evecs = la.eigh(M)              # ascending
        lam_max = float(evals[evals.size - 1]) if evals.size else 0.0
        if lam_max <= breakdown_lam:                      # the whole block is numerically zero
            return None, None, 0
        keep = evals > DEFLATE_EVAL_TOL * lam_max         # boolean mask over p_in
        p_next = int(keep.sum())
        if p_next == 0:                                   # unreachable: lam_max always survives
            return None, None, 0
        V = evecs[:, keep]                                # (p_in, p_next)
        s = np.sqrt(evals[keep])                          # (p_next,)
        beta_j = (s[:, None] * np.conj(V.T))             # (p_next, p_in)   off-diag block
        beta_inv = V / s[None, :]                         # (p_in,  p_next)
        return beta_j, beta_inv, p_next


def _cholesky_qr2(M2, beta_j, active_k):
    r"""Second pass of CholeskyQR2 on the recomputed Gram of the once-QR'd block.

    Superseded together with :func:`_cholesky_or_deflate`, kept for the same regression
    tests. Re-orthonormalizes ``Q1 = Wp @ beta_inv`` (whose error can be :math:`O(1)`
    when ``M`` was ill-conditioned) back to :math:`O(\texttt{EPS})` using the Gram of
    the actual vectors; see doc/lanczos_invariants.md ("deflation vs breakdown scales")
    for the derivation.

    Given ``M2`` and the first-pass factor ``beta_j`` (so ``Wp = Q1 @ beta_j``), returns
    ``(beta2_inv, beta_j_new, k2)`` such that ``Q2 = Q1 @ beta2_inv`` is orthonormal and
    ``Wp = Q2 @ beta_j_new``.  Returns ``(None, None, 0)`` on full collapse.
    """
    beta2, beta2_inv, k2 = _cholesky_or_deflate(M2, active_k)
    if k2 == 0:
        return None, None, 0
    return beta2_inv, beta2 @ beta_j, k2


def block_lanczos_array_cy(
    np.ndarray psi0,
    h_op,
    converged,
    verbose=False,
    reort=Reort.NONE,
    alphas=None,
    betas=None,
    Q=None,
    W=None,
    return_W=False,
    comm=None,
    return_widths=False,
    return_status=False,
    build_krylov_basis=None,
    deflate_tol=-1.0,
    **kwargs
):
    # Resolve a string reort (e.g. "full") to the Reort enum, mirroring
    # block_lanczos_cy; otherwise the `reort == Reort.FULL` comparisons below never
    # match a string and the build silently runs with no reorthogonalization.
    reort = resolve_reort(reort)

    # Whether to retain the accumulated Krylov basis. The historical default (True) stands:
    # restart machinery (TRLM/IRLM) and direct callers consume the returned Q regardless of
    # the reort mode. Callers that only need the final residual block (the GF continued
    # fraction with reort NONE) opt out explicitly, dropping the O(N * n * k) growth; any
    # reort mode projects against the basis and the warm-start protocol slices its blocks,
    # so opting out requires reort='none' and a fresh start.
    cdef bint keep_krylov
    if build_krylov_basis is None:
        keep_krylov = True
    else:
        keep_krylov = bool(build_krylov_basis)
        if not keep_krylov and (reort != Reort.NONE or Q is not None):
            raise ValueError("build_krylov_basis=False requires reort='none' and no warm-start Q")
    cdef int N = psi0.shape[0] if psi0 is not None else Q.shape[0]
    cdef int n = (psi0.shape[1] if psi0.ndim == 2 else 1) if psi0 is not None else alphas[0].shape[0]

    cdef int start_it = 0
    cdef np.ndarray alphas_arr, betas_arr
    cdef list Q_list
    # Krylov storage: Q_buf is an over-allocated column buffer grown geometrically
    # (amortized O(1) copies instead of one full np.concatenate reallocation per step);
    # Q_list[0] is always the filled Q_buf[:, :q_cols] view the reort machinery reads.
    cdef np.ndarray Q_buf = None
    cdef int q_cols = 0

    if alphas is not None and betas is not None and Q is not None:
        start_it = len(alphas)
        alphas_arr = alphas
        betas_arr = betas

        q0_fallback = psi0.copy() if psi0 is not None else Q[:, :n]
        q = [Q[:, (start_it - 1) * n : start_it * n] if start_it > 0 else q0_fallback]
        q.append(Q[:, start_it * n : (start_it + 1) * n] if start_it > 0 else q0_fallback)
        # Zero spare capacity: the first append reallocates, so the caller's Q is never
        # written to in place.
        Q_buf = Q
        q_cols = Q.shape[1]
        Q_list = [Q]
    else:
        start_it = 0
        alphas_arr = np.empty((0, n, n), dtype=complex)
        betas_arr = np.empty((0, n, n), dtype=complex)

        q = [np.zeros((N, n), dtype=complex, order='C')]
        q.append(np.ascontiguousarray(psi0 if psi0.ndim == 2 else psi0.reshape(-1, 1)))
        if keep_krylov:
            Q_buf = q[1].copy()
            q_cols = Q_buf.shape[1]
            Q_list = [Q_buf]
        else:
            Q_list = None

    cdef int period = kwargs.get("reort_period", REORT_PERIOD)
    cdef int max_iter = kwargs.get("max_iter", int(np.ceil(h_op.shape[0] / n if sps.issparse(h_op) or isinstance(h_op, np.ndarray) else N / n)))
    cdef int _buf_size = start_it + max_iter
    cdef np.ndarray[double complex, ndim=3] alphas_buf = np.zeros((_buf_size, n, n), dtype=complex)
    cdef np.ndarray[double complex, ndim=3] betas_buf = np.zeros((_buf_size, n, n), dtype=complex)
    if start_it > 0:
        alphas_buf[:start_it] = alphas_arr
        betas_buf[:start_it] = betas_arr

    # Bounded-W ping-pong buffers + beta-norm history: the estimator writes into the
    # spare persistent buffer and reuses the completed blocks' ||beta_j||_2 instead of
    # re-factorizing them.
    _w_bufs = None
    beta_norm_hist = None
    if reort in (Reort.PARTIAL, Reort.SELECTIVE):
        _w_bufs = (
            np.empty((2, _buf_size + 2, n, n), dtype=complex),
            np.empty((2, _buf_size + 2, n, n), dtype=complex),
        )
        beta_norm_hist = [None] * start_it

    cdef bint is_sparse = sps.issparse(h_op)
    cdef bint is_dense = isinstance(h_op, np.ndarray)

    cdef int it = start_it
    cdef double t_norm_max = 0.0
    cdef double h_norm_est = 0.0
    cdef double _h_scale = 0.0
    cdef double beta_norm, alpha_norm
    cdef double reort_eps  # always assigned (from REORT_TOL) before its one read, below

    cdef bint mpi = comm is not None
    cdef int rank = comm.rank if mpi else 0
    cdef int size = comm.size if mpi else 1
    cdef np.ndarray counts, offsets
    cdef int global_N = N

    if mpi:
        counts = np.empty((size), dtype=int)
        comm.Allgather(np.array([N], dtype=int), counts)
        offsets = np.array([np.sum(counts[:r]) for r in range(size)], dtype=int)
        global_N = np.sum(counts)

    cdef double complex[:] h_data
    cdef long[:] h_indices
    cdef long[:] h_indptr
    cdef double complex[:, ::1] h_dense

    if is_sparse:
        h_op = h_op.tocsr()
        h_data = h_op.data
        h_indices = np.ascontiguousarray(h_op.indices, dtype=np.int64)
        h_indptr = np.ascontiguousarray(h_op.indptr, dtype=np.int64)
    elif is_dense:
        h_dense = np.ascontiguousarray(h_op)

    cdef np.ndarray wp_arr = np.empty((N, n), dtype=complex, order='C')
    cdef double complex[:, ::1] wp = wp_arr
    cdef double complex[:, ::1] q1, q0, beta_prev_dag_mv
    cdef double complex[:, ::1] alpha_i

    # GUARDRAIL: the MPI matvec forms the full (global_N, n) partial product on *every*
    # rank (column-distributed H -> Allreduce -> slice local rows), so per-rank memory
    # here scales with global_N, not local_N. This is intentional: the array kernel is
    # for small/dense sectors. For a large global_N use the sparse hash-distributed
    # kernel (BlockLanczos.pyx / block_lanczos_cy), which never forms a dense global
    # vector. We deliberately do NOT halo-exchange this (see blocklanczos_blas_
    # acceleration.md §3, "WON'T FIX"): the dense path already OOMs on the global_N^2
    # matrix first, and the memory-bound CIPSI/GF case runs on the sparse kernel anyway.
    cdef np.ndarray wp_global = np.empty((global_N, n), dtype=complex, order='C') if mpi else None
    cdef double complex[:, ::1] wp_g = wp_global

    cdef list block_widths = list(kwargs.get("block_widths_init", [n] * start_it))
    cdef int n_curr, n_prev, active_k

    # EA16 §2.6.2 locking deflation: keep every Lanczos vector orthogonal to the
    # already-converged ("locked") Ritz vectors. Without this the matvec keeps
    # amplifying the dominant locked directions back into the active subspace,
    # which reintroduces locked eigenvalues (and their 2*theta harmonics) as
    # spurious Ritz values *below* the true spectral minimum on restarted sweeps.
    # `locked` is column-distributed exactly like the Krylov vectors (local rows).
    cdef np.ndarray locked_arr = kwargs.get("locked", None)
    cdef np.ndarray locked_ovl
    cdef bint have_locked = locked_arr is not None and locked_arr.shape[1] > 0
    if have_locked:
        locked_arr = np.ascontiguousarray(locked_arr, dtype=complex)

    # Locking-deflation mode: "full" (default) unconditionally projects every Lanczos
    # vector against the locked set each step; "partial" implements the estimate-driven
    # EA16 §2.6.2 scheme — a cheap per-pair overlap recurrence (no O(N) inner products)
    # that reorthogonalizes only the locked vectors whose estimated overlap exceeds
    # omega_TOL. The default preserves the previous behaviour exactly.
    cdef str locked_reort = kwargs.get("locked_reort", "full")
    cdef bint partial_locked = have_locked and locked_reort == "partial"
    cdef np.ndarray locked_evals_arr = kwargs.get("locked_evals", None)
    cdef double locked_rho = float(kwargs.get("locked_res", REORT_TOL))
    cdef double omega_min_l = EPS * n * np.sqrt(global_N)
    cdef np.ndarray xi_l, xi_prev_l, xi_mask, q_next_ovl
    cdef double bj_inv_norm, bjm1_norm
    cdef int nlock = locked_arr.shape[1] if have_locked else 0
    if partial_locked:
        if locked_evals_arr is None:
            raise ValueError("locked_reort='partial' requires 'locked_evals'")
        locked_evals_arr = np.ascontiguousarray(np.real(locked_evals_arr), dtype=float)
        xi_l = np.full(nlock, omega_min_l)
        xi_prev_l = np.full(nlock, omega_min_l)
    from impurityModel.ed import ea16 as _ea16

    # Why the recurrence stops (returned when return_status=True), mirroring block_lanczos_cy:
    # "converged" (converged() satisfied), "invariant_subspace" (rank-deficient residual => the
    # block-Krylov space is closed under H => result is *exact*), "diverged" (non-finite Gram or
    # the divergence guard truncated a corrupted tail => NOT exact) or "max_iter" (budget spent
    # before the recurrence terminated naturally).
    termination = "max_iter"

    # Two-consecutive-steps reort rule (Simon/PROPACK): when a step acts, the next step
    # bypasses the REORT_TOL gate (see apply_reort force=) so q_curr's remaining overlap
    # cannot silently re-contaminate the recurrence through the three-term coupling.
    _force_reort = False

    while it < _buf_size:
        q1 = np.ascontiguousarray(q[1])
        n_curr = q1.shape[1]

        # Re-allocate wp buffers to match current active width for contiguous alignment
        if wp_arr.shape[1] != n_curr:
            wp_arr = np.empty((N, n_curr), dtype=complex, order='C')
            wp = wp_arr
            if mpi:
                wp_global = np.empty((global_N, n_curr), dtype=complex, order='C')
                wp_g = wp_global

        if is_sparse:
            with nogil:
                apply_sparse_csr_nogil(global_N, N, n_curr, h_data, h_indices, h_indptr, q1, wp_g if mpi else wp)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, wp_global, op=MPI.SUM)
                wp_arr[:] = wp_global[offsets[rank] : offsets[rank] + N, :]
        elif is_dense:
            with nogil:
                apply_dense_nogil(global_N, N, n_curr, h_dense, q1, wp_g if mpi else wp)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, wp_global, op=MPI.SUM)
                wp_arr[:] = wp_global[offsets[rank] : offsets[rank] + N, :]
        else:
            if hasattr(h_op, "dot"):
                wp_arr[:] = h_op.dot(q1)
            elif hasattr(h_op, "__matmul__"):
                wp_arr[:] = h_op @ q1
            else:
                wp_arr[:] = h_op(q1)

        # Allocate alpha_i as a contiguous array
        alpha_i_arr = np.empty((n_curr, n_curr), dtype=complex, order='C')
        alpha_i = alpha_i_arr
        with nogil:
            matmul_nogil(n_curr, n_curr, N, 1.0, q1, b'C', wp, b'N', 0.0, alpha_i)

        if mpi:
            comm.Allreduce(MPI.IN_PLACE, alpha_i_arr, op=MPI.SUM)

        alphas_buf[it, :n_curr, :n_curr] = alpha_i_arr

        with nogil:
            matmul_nogil(N, n_curr, n_curr, -1.0, q1, b'N', alpha_i, b'N', 1.0, wp)

        if it > 0:
            n_prev = q[0].shape[1]
            beta_prev_dag_arr = np.conj(betas_buf[it - 1, :n_curr, :n_prev].T).copy()
            beta_prev_dag_arr_c = np.ascontiguousarray(beta_prev_dag_arr)
            beta_prev_dag_mv = beta_prev_dag_arr_c
            q0 = np.ascontiguousarray(q[0])
            with nogil:
                matmul_nogil(N, n_curr, n_prev, -1.0, q0, b'N', beta_prev_dag_mv, b'N', 1.0, wp)

        # "full" locking deflation: unconditionally project against the locked set every
        # step (twice for robustness), for every reort mode. The "partial" mode skips this
        # and instead does the estimate-driven EA16 §2.6.2 reorth on q_next below.
        if have_locked and not partial_locked:
            for _ in range(2):
                locked_ovl = np.ascontiguousarray(locked_arr.conj().T @ wp_arr)
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, locked_ovl, op=MPI.SUM)
                wp_arr -= locked_arr @ locked_ovl
            wp = wp_arr

        if reort == Reort.FULL or (reort == Reort.PERIODIC and it > 0 and it % period == 0):
            if not keep_krylov:
                raise RuntimeError("Krylov basis must be built for reorthogonalization")
            wp_arr, _, _ = apply_reort(wp_arr, Q_list, None, Reort.FULL, mpi, comm, block_widths)

        # TSQR of the residual block: the triangular factor is built from the block itself
        # (panel Householder + Givens merges + one Allgather), never from wp^H wp, so it is
        # backward stable at any conditioning and the rank decision below is made on the
        # block's true singular values. A *residual* block is negligible when it is small
        # against H, not against 1, so the breakdown reference is the divergence guard's
        # running spectral-scale estimate (max of the block norms seen so far, all bounded by
        # ||H||), widened by this step's alpha since the guard has not run yet. On the first
        # step of a cold start h_norm_est is still 0 and alpha_0 carries the scale.
        _h_scale = max(h_norm_est, t_norm_max, float(np.linalg.norm(alpha_i_arr, ord=2)))
        q_next, beta_i, active_k, sv_i = tsqr(
            wp_arr, comm if mpi else None, _h_scale, deflate_tol=deflate_tol
        )
        if active_k < 0:
            # Non-finite factor => corrupted recurrence, truncated (NOT exact).
            termination = "diverged"
            block_widths.append(n_curr)
            it += 1
            break
        if active_k == 0:
            # Rank-deficient residual => the block-Krylov space is closed => exact.
            termination = "invariant_subspace"
            block_widths.append(n_curr)
            it += 1
            break

        betas_buf[it, :active_k, :n_curr] = beta_i

        # --- Divergence safeguard ---------------------------------------
        # For a Hermitian H every block norm is bounded by ||H||; a jump of several orders
        # of magnitude over the largest healthy block norm means the recurrence has been
        # corrupted past repair. Truncate *before* this block (exclude alpha_i/beta_i at
        # index it) so the diverged tail never reaches the Green's function.
        # ||beta_i||_2 comes out of the factorization: beta_i and the block share their
        # singular values, so sv_i[0] is the 2-norm (reused by the SELECTIVE Ritz-error check
        # below) with no SVD of its own.
        beta_norm = float(sv_i[0])
        _reort_acted = False
        alpha_norm = np.linalg.norm(alpha_i_arr, ord=2)
        diverged, t_norm_max, h_norm_est = divergence_guard(
            beta_norm, alpha_norm, it == start_it, t_norm_max, h_norm_est
        )
        if diverged:
            # Ungated from verbose: a truncated recurrence is a problem the caller must see
            # regardless of verbosity, not detail. Rank gate kept (mpi/comm.rank == 0).
            if not mpi or comm.rank == 0:
                print(
                    f"[BlockLanczos] Divergence detected at iteration {it}: "
                    f"|beta|={beta_norm:.3e}, |alpha|={alpha_norm:.3e} >> spectral scale "
                    f"{h_norm_est:.3e}. Truncating to the last trustworthy block.",
                    flush=True,
                )
            termination = "diverged"
            break

        # EA16 §2.6.2 estimate-driven locking reorthogonalization. Propagate the cheap
        # per-pair overlap estimate (no O(N) work) and reorthogonalize q_next only against
        # the locked vectors whose estimate now exceeds omega_TOL, resetting those to
        # omega_min. ||B_j^{-1}|| = 1/sigma_min of the retained block, ||B_{j-1}|| =
        # ||betas_buf[it-1]||_2.
        if partial_locked:
            bj_inv_norm = 1.0 / float(sv_i[active_k - 1])
            bjm1_norm = float(np.linalg.norm(betas_buf[it - 1], 2)) if it > 0 else 0.0
            xi_new_l, xi_trigger, xi_mask = _ea16.locked_overlap_step(
                xi_l, xi_prev_l, locked_evals_arr, alpha_i_arr,
                bj_inv_norm, bjm1_norm, locked_rho, REORT_TOL, BAD_BLOCK_TOL, EPS,
            )
            if xi_trigger:
                idx_l = np.nonzero(xi_mask)[0]
                Lm = np.ascontiguousarray(locked_arr[:, idx_l])
                for _ in range(2):
                    q_next_ovl = np.ascontiguousarray(Lm.conj().T @ q_next)
                    if mpi:
                        comm.Allreduce(MPI.IN_PLACE, q_next_ovl, op=MPI.SUM)
                    q_next = q_next - Lm @ q_next_ovl
                xi_new_l[idx_l] = omega_min_l
                xi_l[idx_l] = omega_min_l
            xi_prev_l = xi_l
            xi_l = xi_new_l

        if reort in (Reort.PARTIAL, Reort.SELECTIVE):
            if not keep_krylov:
                raise RuntimeError("Krylov basis must be built for reorthogonalization")
            if W is None:
                if start_it > 0:
                    W = np.zeros((2, start_it + 1, n, n), dtype=complex)
                    for j in range(start_it):
                        w_j = block_widths[j]
                        Q_j = Q_list[0][:, sum(block_widths[:j]) : sum(block_widths[:j+1])]
                        W[1, j, :w_j, :n_curr] = Q_j.conj().T @ wp_arr
                        if j < start_it - 1:
                            W[0, j, :w_j, :n_prev] = Q_j.conj().T @ q[0]
                    if mpi:
                        comm.Allreduce(MPI.IN_PLACE, W, op=MPI.SUM)
                    W[1, start_it, :n_curr, :n_curr] = np.eye(n_curr)
                    W[0, start_it - 1, :n_prev, :n_prev] = np.eye(n_prev)
                else:
                    W = np.zeros((2, 1, n, n), dtype=complex)
                    W[1, 0, :n_curr, :n_curr] = np.eye(n_curr)
            elif W.shape[1] < it + 1:
                W_new = np.zeros((2, it + 1, n, n), dtype=complex)
                W_new[:, : W.shape[1]] = W
                W = W_new

            block_widths.append(n_curr)
            block_widths.append(active_k)
            W = estimate_orthonormality(
                W, alphas_buf[: it + 1], betas_buf[: it + 1], block_widths=block_widths, eps=EPS, N=global_N,
                out=_w_bufs[it % 2] if _w_bufs is not None else None, beta_norms=beta_norm_hist,
            )
            block_widths.pop()
            block_widths.pop()

            reort_eps = REORT_TOL

            if reort == Reort.SELECTIVE:
                # EA16 §2.6.2 selective orthogonalization (shared with block_lanczos_step_cy).
                # Q_list[0] holds all it+1 completed blocks (the current block q1 is already its
                # last block), spanning the full subspace the Ritz vectors index. beta_norm is the
                # already-computed ||beta_i||_2.
                q_next = selective_orthogonalize(
                    q_next, Q_list[0], alphas_buf, betas_buf, W, block_widths,
                    it, n_curr, beta_norm, reort_eps, REORT_PERIOD, mpi, comm,
                )

            if reort in (Reort.PARTIAL, Reort.SELECTIVE):
                # Pass block_widths + [n_curr] so the current block (index it, already
                # appended to Q_list) is in the width table apply_reort indexes — matching
                # block_lanczos_cy. Without it the W-recurrence's bad-block columns are
                # mis-mapped, which silently loses orthogonality when resuming on a
                # restarted (e.g. IRLM-compressed) basis.
                q_next, W, _reort_acted = apply_reort(
                    q_next, Q_list, W, reort, mpi, comm, block_widths + [n_curr], force=_force_reort
                )
                _force_reort = _reort_acted
                # Only when the bad-block projection actually changed q_next does it need
                # re-orthonormalizing; when no block was projected q_next is unchanged and this
                # would be an exact no-op (M2 == I), so skip it and save the Gram + MPI Allreduce.
                if _reort_acted:
                    # The projection acts on the post-QR block, so it leaves q_next non-orthonormal
                    # and beta_i inconsistent; on a near-redundant block (tight clusters) it removes
                    # almost all of q_next, and that residual must be renormalized (or legitimately
                    # deflated) before it feeds the next iteration — else the recurrence amplifies
                    # the tiny leftover and the basis blows up. Re-factoring folds the correction
                    # into beta_i: wp = (q_next @ R2) @ beta_i.
                    q_next_2, R2, active_k, sv2 = tsqr(
                        q_next, comm if mpi else None, 1.0, deflate_tol=deflate_tol
                    )
                    # An *absolutely* tiny residual => the new block is fully contained in the
                    # existing Krylov span (invariant subspace, e.g. a tight cluster whose
                    # effective rank is exhausted). Renormalizing that noise would amplify rounding
                    # into the recurrence (beta blow-up), so stop here. sqrt(EPS) is the largest
                    # column norm the old max(diag(q_next^H q_next)) < EPS test admitted.
                    if active_k <= 0 or float(sv2[0]) < REORT_TOL:
                        termination = "invariant_subspace"
                        block_widths.append(n_curr)
                        it += 1
                        break
                    beta_i = R2 @ beta_i
                    q_next = q_next_2
                    # The renormalization rescales q_next's true overlaps with every
                    # Krylov column by up to ||R2^{-1}||_2 = 1/sigma_min; propagate the same
                    # bound into the just-written honest post-reort W estimates (a no-op when
                    # the projection removed little: R2 ~ I).
                    W[1, : W.shape[1] - 1] *= 1.0 / float(sv2[active_k - 1])
                    betas_buf[it, :active_k, :n_curr] = beta_i

        # Record ||beta_it||_2 of the FINAL stored beta (the acted-renormalize above may
        # have replaced beta_i) for the estimator's noise floor at later iterations.
        if beta_norm_hist is not None:
            beta_norm_hist.append(
                float(np.linalg.svd(beta_i, compute_uv=False)[0]) if _reort_acted else beta_norm
            )

        if converged(alphas_buf[: it + 1], betas_buf[: it + 1], verbose=verbose, block_widths=block_widths + [n_curr]):
            termination = "converged"
            block_widths.append(n_curr)
            it += 1
            break

        q[0] = q[1]
        q[1] = q_next
        if keep_krylov:
            n_new = q_next.shape[1]
            if q_cols + n_new > Q_buf.shape[1]:
                Q_grown = np.empty((Q_buf.shape[0], max(2 * Q_buf.shape[1], q_cols + n_new)), dtype=complex, order='C')
                Q_grown[:, :q_cols] = Q_buf[:, :q_cols]
                Q_buf = Q_grown
            Q_buf[:, q_cols : q_cols + n_new] = q_next
            q_cols += n_new
            Q_list[0] = Q_buf[:, :q_cols]
        block_widths.append(n_curr)
        it += 1

    if verbose and (not mpi or rank == 0):
        print(f"Block Lanczos terminated ({termination}) at iteration {it}")

    # Tail-only mode: q[1] is the last block ever appended in stored mode (the roll
    # q[0]=q[1]; q[1]=q_next runs before the append, and every break happens before the
    # append), so callers slicing the final residual columns see identical data.
    if keep_krylov:
        # Trim: returning a view of the over-allocated growth buffer would pin its spare
        # capacity for as long as the caller (e.g. TRLM restart) holds Q.
        res_Q = Q_buf if Q_buf.shape[1] == q_cols else np.ascontiguousarray(Q_buf[:, :q_cols])
    else:
        res_Q = q[1]
    res_alphas = alphas_buf[:it]
    res_betas = betas_buf[:it]
    # return_status appends `termination` as the final element, independent of the other
    # flags, so existing call sites that pass neither keep their original return arity.
    if return_widths:
        if return_W:
            if return_status:
                return res_alphas, res_betas, res_Q, W, block_widths, termination
            return res_alphas, res_betas, res_Q, W, block_widths
        if return_status:
            return res_alphas, res_betas, res_Q, block_widths, termination
        return res_alphas, res_betas, res_Q, block_widths
    else:
        if return_W:
            if return_status:
                return res_alphas, res_betas, res_Q, W, termination
            return res_alphas, res_betas, res_Q, W
        if return_status:
            return res_alphas, res_betas, res_Q, termination
        return res_alphas, res_betas, res_Q


def block_lanczos_array(*args, **kwargs):
    return block_lanczos_array_cy(*args, **kwargs)


