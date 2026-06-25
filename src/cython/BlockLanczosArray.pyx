# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True, freethreading_compatible=True

"""Block Lanczos kernel for numpy arrays / scipy sparse operators (MPI row-block).

This module is the numerical core shared with the ManyBodyState kernel in
``BlockLanczos.pyx`` (which imports the constants, ``estimate_orthonormality``,
``_build_full_T`` and ``apply_reort`` from here). Both build a block-tridiagonal
``T`` (diagonal blocks ``alpha``, off-diagonal blocks ``beta``) whose eigenpairs
approximate those of the operator; the TRLM/IRLM drivers restart it.

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

Module thresholds (all derived from machine ``eps``; see definitions below):
``REORT_TOL`` (loss-of-orthogonality trigger), ``BAD_BLOCK_TOL`` (which blocks to
reorth against), ``DEFLATE_TOL`` (relative rank floor for the block Cholesky),
``BREAKDOWN_TOL`` (invariant-subspace detection), ``REORT_PERIOD`` (cadence).

Deflation policy: when a Lanczos block is rank-deficient (Cholesky of the block
Gram matrix ``M`` hits the ``DEFLATE_TOL`` floor), the block size shrinks (EA16
shrinking-block, Meerbergen & Scott, RAL-TR-2000-011) rather than zero-padding or
terminating, so ``beta`` becomes rectangular and ``T`` carries variable-size
blocks; the recurrence keeps converging.
"""

cimport cython
import numpy as np
cimport numpy as np
import scipy.linalg as la
import scipy.sparse as sps

from libc.math cimport sqrt
from scipy.linalg.cython_blas cimport zgemm

from mpi4py import MPI
from enum import Enum


class Reort(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2
    PERIODIC = 3
    SELECTIVE = 4


cdef double EPS_VAL = np.finfo(float).eps
EPS = EPS_VAL          # ~2.22e-16
REORT_TOL = np.sqrt(EPS_VAL)        # ~1.49e-8  : trigger — reorth when max|W| exceeds this
BAD_BLOCK_TOL = EPS_VAL ** 0.75        # ~1.83e-12 : selection — reorth against blocks above this
DEFLATE_TOL = EPS_VAL ** 0.5          # ~1.49e-8  : relative rank floor for eigenvalues of M
BREAKDOWN_TOL = 1e-12              # absolute: ||beta||_2 below this ⇒ invariant subspace
REORT_PERIOD = 5                   # PERIODIC cadence, and SELECTIVE Ritz-check cadence


def _cholesky_or_deflate(M, p_in):
    # Try Cholesky first (fast path)
    try:
        L = la.cholesky(M, lower=True)
        if np.any(np.diag(L) < DEFLATE_TOL * max(np.max(np.diag(L)), 1.0)):
            raise la.LinAlgError("Numeric singularity in Cholesky diagonal")
        beta_j = np.conj(L.T)
        beta_inv = la.inv(beta_j)
        p_next = p_in
        return beta_j, beta_inv, p_next
    except (la.LinAlgError, ValueError):
        # Fall back to eigh
        evals, evecs = la.eigh(M)              # ascending
        keep = evals > DEFLATE_TOL * max(evals[-1], 1.0) # boolean mask over p_in
        p_next = int(keep.sum())
        if p_next == 0:                                   # whole block collapsed
            return None, None, 0
        V = evecs[:, keep]                                # (p_in, p_next)
        s = np.sqrt(evals[keep])                          # (p_next,)
        beta_j   = (s[:, None] * np.conj(V.T))             # (p_next, p_in)   off-diag block
        beta_inv = V / s[None, :]                         # (p_in,  p_next)
        return beta_j, beta_inv, p_next



def calculate_thermal_gs(h, block_size, e_max, v0=None, reort=Reort.FULL, comm=None):
    mpi = comm is not None
    rank = comm.rank if mpi else 0
    size = comm.size if mpi else 1

    def converged(alphas, betas, *args, **kwargs):
        e, s = eigsh(alphas, betas, de=e_max, select="m")
        sorted_indices = np.argsort(e)
        e = e[sorted_indices]
        s = s[:, sorted_indices]
        mask = e - np.min(e) <= e_max
        return np.linalg.norm(betas[-1] @ s[-block_size:, mask], ord=2) < 1e-6

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
        verbose=True,
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


cpdef np.ndarray estimate_orthonormality(
    np.ndarray[double complex, ndim=4] W,
    np.ndarray[double complex, ndim=3] alphas,
    np.ndarray[double complex, ndim=3] betas,
    object block_widths=None,
    double eps=0.0,
    int N=1
):
    cdef int i = alphas.shape[0] - 1
    cdef int n = alphas.shape[1]
    if eps == 0.0:
        eps = np.finfo(float).eps

    cdef list widths
    if block_widths is None:
        widths = [n] * (i + 2)
    else:
        widths = list(block_widths)

    cdef int w_curr = widths[i]
    cdef int w_i = w_curr
    cdef int w_next = widths[i+1]
    cdef int w_i_next = w_next
    cdef int w_0 = widths[0]

    cdef np.ndarray[double complex, ndim=4] W_out = np.zeros((2, i + 2, n, n), dtype=complex)
    cdef np.ndarray[double complex, ndim=3] w_bar = np.zeros((i + 2, n, n), dtype=complex)

    w_bar[i + 1, :w_next, :w_next] = np.identity(w_next)

    cdef np.ndarray beta_i_dag_inv = np.conj(la.pinv(betas[i, :w_next, :w_curr]).T) # shape (w_next, w_curr)
    w_bar[i, :w_next, :w_0] = eps * N * beta_i_dag_inv @ betas[0, :w_curr, :w_0]

    if i == 0:
        W_out[0, : i + 1] = W[1]
        W_out[1, : i + 2] = w_bar
        return W_out

    # j = 0
    cdef int w_j = widths[0]
    cdef int w_j_next = widths[1]
    cdef int w_i_prev = widths[i-1]
    cdef np.ndarray term1 = W[1, 1, :w_i, :w_j_next] @ betas[0, :w_j_next, :w_j]
    cdef np.ndarray term2 = W[1, 0, :w_i, :w_j] @ alphas[0, :w_j, :w_j]
    cdef np.ndarray term3 = alphas[i, :w_i, :w_i] @ W[1, 0, :w_i, :w_j]
    cdef np.ndarray term5 = betas[i-1, :w_i, :w_i_prev] @ W[0, 0, :w_i_prev, :w_j]
    cdef np.ndarray RHS_0 = term1 + term2 - term3 - term5
    w_bar[0, :w_next, :w_j] = beta_i_dag_inv @ RHS_0

    cdef int j, w_j_prev
    cdef np.ndarray term4
    cdef np.ndarray RHS
    for j in range(1, i):
        w_j = widths[j]
        w_j_prev = widths[j-1]
        w_j_next = widths[j+1]

        term1 = W[1, j+1, :w_i, :w_j_next] @ betas[j, :w_j_next, :w_j]
        term2 = W[1, j, :w_i, :w_j] @ alphas[j, :w_j, :w_j]
        term3 = alphas[i, :w_i, :w_i] @ W[1, j, :w_i, :w_j]
        term4 = W[1, j-1, :w_i, :w_j_prev] @ np.conj(betas[j-1, :w_j, :w_j_prev].T)
        term5 = betas[i-1, :w_i, :w_i_prev] @ W[0, j, :w_i_prev, :w_j]

        RHS = term1 + term2 - term3 + term4 - term5
        w_bar[j, :w_next, :w_j] = beta_i_dag_inv @ RHS

    w_bar[:i] += eps * (betas[i] + betas[:i])

    W_out[0, : i + 1] = W[1]
    W_out[1, : i + 2] = w_bar

    return W_out


cpdef np.ndarray build_banded_matrix(np.ndarray[double complex, ndim=3] alphas, np.ndarray[double complex, ndim=3] betas):
    cdef int k = alphas.shape[0]
    cdef int p = alphas.shape[1]
    cdef np.ndarray[double complex, ndim=2] bands = np.zeros((p + 1, k * p), dtype=alphas.dtype)
    cdef int i
    bands[0, :] = np.diagonal(alphas, offset=0, axis1=1, axis2=2).flatten()
    for i in range(1, p + 1):
        alpha_diags = np.diagonal(alphas, offset=-i, axis1=1, axis2=2)
        beta_diags = np.diagonal(betas, offset=p - i, axis1=1, axis2=2)
        bands[i, :] = np.concatenate([alpha_diags, beta_diags], axis=1).flatten()
    return bands


cpdef np.ndarray _build_full_T(np.ndarray[double complex, ndim=3] alphas, np.ndarray[double complex, ndim=3] betas, object block_widths=None, object comm=None):
    cdef int m = alphas.shape[0]
    if m == 0:
        return np.zeros((0, 0), dtype=complex)

    cdef list widths
    if block_widths is None:
        widths = [alphas.shape[1]] * m
    else:
        widths = list(block_widths)

    cdef int total_dim = sum(widths)
    cdef np.ndarray[double complex, ndim=2] T = np.zeros((total_dim, total_dim), dtype=complex)

    cdef list offsets = [0]
    cdef int off = 0
    cdef object w_val
    for w_val in widths:
        off += int(w_val)
        offsets.append(off)

    cdef int i, w_i, w_next, o_i, o_next
    for i in range(m):
        w_i = int(widths[i])
        o_i = offsets[i]
        T[o_i : o_i + w_i, o_i : o_i + w_i] = alphas[i, :w_i, :w_i]
        if i < m - 1:
            w_next = int(widths[i+1])
            o_next = offsets[i+1]
            T[o_next : o_next + w_next, o_i : o_i + w_i] = betas[i, :w_next, :w_i]
            T[o_i : o_i + w_i, o_next : o_next + w_next] = np.conj(betas[i, :w_next, :w_i].T)
    return T


cpdef tuple _extract_blocks(np.ndarray[double complex, ndim=2] T, int m, int n):
    cdef np.ndarray[double complex, ndim=3] alphas = np.zeros((m, n, n), dtype=complex)
    cdef np.ndarray[double complex, ndim=3] betas = np.zeros((m - 1, n, n), dtype=complex)
    cdef int i
    for i in range(m):
        alphas[i] = T[i * n : (i + 1) * n, i * n : (i + 1) * n]
        if i < m - 1:
            betas[i] = T[(i + 1) * n : (i + 2) * n, i * n : (i + 1) * n]
    return alphas, betas


cpdef tuple eigsh(
    np.ndarray[double complex, ndim=3] alphas,
    np.ndarray[double complex, ndim=3] betas,
    object de=None,
    np.ndarray Q=None,
    bint eigvals_only=False,
    str select="a",
    object select_range=None,
    int max_ev=0,
    object comm=None
):
    cdef bint within_gs = False
    if select == "m":
        assert de is not None
        select = "a"
        within_gs = True

    cdef np.ndarray Tm = build_banded_matrix(alphas, betas)
    cdef np.ndarray eigvals
    cdef np.ndarray eigvecs

    if eigvals_only:
        eigvals = la.eig_banded(
            Tm,
            lower=True,
            eigvals_only=True,
            overwrite_a_band=True,
            check_finite=False,
            select=select,
            select_range=select_range,
            max_ev=max_ev,
        )
        eigvals = np.sort(eigvals)
        if within_gs:
            return (eigvals[eigvals - eigvals[0] <= de], None)
        return (eigvals, None)

    eigvals, eigvecs = la.eig_banded(
        Tm,
        lower=True,
        eigvals_only=False,
        overwrite_a_band=True,
        check_finite=False,
        select=select,
        select_range=select_range,
        max_ev=max_ev,
    )

    cdef np.ndarray mask
    if within_gs:
        mask = eigvals - np.min(eigvals) <= de
    else:
        mask = np.ones(len(eigvals), dtype=bool)

    cdef np.ndarray mask_indices = np.where(mask)[0]
    cdef np.ndarray sort_indices = np.argsort(eigvals[mask_indices])
    cdef np.ndarray final_indices = mask_indices[sort_indices]

    eigvals = eigvals[final_indices]
    eigvecs = eigvecs[:, final_indices]

    if Q is not None:
        eigvecs = Q @ eigvecs

    return eigvals, eigvecs


cpdef np.ndarray block_combine_array(np.ndarray Q, np.ndarray Y):
    return Q @ np.ascontiguousarray(Y, dtype=complex)


cpdef tuple block_orthogonalize_array(np.ndarray wp, np.ndarray Q, object overlaps=None, object comm=None):
    if overlaps is None:
        overlaps = np.conj(Q.T) @ wp
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
    wp -= Q @ overlaps
    return wp, overlaps


cpdef tuple block_normalize_array(np.ndarray wp, object comm=None):
    cdef np.ndarray M = np.conj(wp.T) @ wp
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
    cdef np.ndarray beta_j, beta_inv
    cdef int active_k
    beta_j, beta_inv, active_k = _cholesky_or_deflate(M, wp.shape[1])
    if active_k == 0:
        raise ValueError("Block collapsed to zero rank")
    cdef np.ndarray q_next = wp @ beta_inv
    return q_next, beta_j

cdef extern from "complex.h":
    double complex conj(double complex z) nogil

# from scipy.linalg.cython_blas cimport zgemm, zgemv

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
    **kwargs
):
    # Resolve a string reort (e.g. "full") to the Reort enum, mirroring
    # block_lanczos_cy; otherwise the `reort == Reort.FULL` comparisons below never
    # match a string and the build silently runs with no reorthogonalization.
    if isinstance(reort, str):
        _map = {
            "none": Reort.NONE,
            "partial": Reort.PARTIAL,
            "selective": Reort.SELECTIVE,
            "full": Reort.FULL,
            "periodic": Reort.PERIODIC,
        }
        reort = _map.get(reort.lower())
        if reort is None:
            raise ValueError(f"Unknown reort string. Must be one of {list(_map.keys())}.")

    cdef bint build_krylov_basis = (reort != Reort.NONE) or True
    cdef int N = psi0.shape[0] if psi0 is not None else Q.shape[0]
    cdef int n = (psi0.shape[1] if psi0.ndim == 2 else 1) if psi0 is not None else alphas[0].shape[0]

    cdef int start_it = 0
    cdef np.ndarray alphas_arr, betas_arr
    cdef list Q_list

    if alphas is not None and betas is not None and Q is not None:
        start_it = len(alphas)
        alphas_arr = alphas
        betas_arr = betas

        q0_fallback = psi0.copy() if psi0 is not None else Q[:, :n]
        q = [Q[:, (start_it - 1) * n : start_it * n] if start_it > 0 else q0_fallback]
        q.append(Q[:, start_it * n : (start_it + 1) * n] if start_it > 0 else q0_fallback)
        Q_list = [Q]
    else:
        start_it = 0
        alphas_arr = np.empty((0, n, n), dtype=complex)
        betas_arr = np.empty((0, n, n), dtype=complex)

        q = [np.zeros((N, n), dtype=complex, order='C')]
        q.append(np.ascontiguousarray(psi0 if psi0.ndim == 2 else psi0.reshape(-1, 1)))
        Q_list = [q[1].copy()] if build_krylov_basis else None

    cdef int period = kwargs.get("reort_period", 5)
    cdef int max_iter = kwargs.get("max_iter", int(np.ceil(h_op.shape[0] / n if sps.issparse(h_op) or isinstance(h_op, np.ndarray) else N / n)))
    cdef int _buf_size = start_it + max_iter
    cdef np.ndarray[double complex, ndim=3] alphas_buf = np.zeros((_buf_size, n, n), dtype=complex)
    cdef np.ndarray[double complex, ndim=3] betas_buf = np.zeros((_buf_size, n, n), dtype=complex)
    if start_it > 0:
        alphas_buf[:start_it] = alphas_arr
        betas_buf[:start_it] = betas_arr

    cdef bint is_sparse = sps.issparse(h_op)
    cdef bint is_dense = isinstance(h_op, np.ndarray)

    cdef int it = start_it
    cdef double reort_eps = np.sqrt(np.finfo(float).eps)

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
    cdef int n_curr, n_prev, n_next, active_k
    cdef list bad_block_idx
    cdef np.ndarray Q_bad, overlap

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

        # Deflate against the locked Ritz vectors (twice for numerical robustness).
        # Applied for every reort mode, since locking is orthogonal to the Krylov
        # reorthogonalization strategy and must hold even for Reort.NONE.
        if have_locked:
            for _ in range(2):
                locked_ovl = np.ascontiguousarray(locked_arr.conj().T @ wp_arr)
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, locked_ovl, op=MPI.SUM)
                wp_arr -= locked_arr @ locked_ovl
            wp = wp_arr

        if reort == Reort.FULL or (reort == Reort.PERIODIC and it > 0 and it % period == 0):
            if not build_krylov_basis:
                raise RuntimeError("Krylov basis must be built for reorthogonalization")
            wp_arr, _ = apply_reort(wp_arr, Q_list, None, Reort.FULL, mpi, comm, block_widths)

        M = wp_arr.conj().T @ wp_arr
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)

        if np.any(np.isnan(M)) or np.any(np.isinf(M)):
            block_widths.append(n_curr)
            it += 1
            break

        beta_i, beta_inv, active_k = _cholesky_or_deflate(M, n_curr)
        if active_k == 0:
            block_widths.append(n_curr)
            it += 1
            break

        betas_buf[it, :active_k, :n_curr] = beta_i

        q_next = wp_arr @ beta_inv

        if reort in (Reort.PARTIAL, Reort.SELECTIVE):
            if not build_krylov_basis:
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
            W = estimate_orthonormality(W, alphas_buf[: it + 1], betas_buf[: it + 1], block_widths=block_widths, eps=EPS)
            block_widths.pop()
            block_widths.pop()

            reort_eps = REORT_TOL

            if reort == Reort.SELECTIVE:
                # Gate the O(m^3) Ritz-convergence check (build T_full + eigh) to a
                # REORT_PERIOD cadence: the PARTIAL bad-block reorth below still runs
                # every step to maintain orthogonality, so locking converged Ritz
                # vectors less often is safe but much cheaper.
                if it > 0 and it % REORT_PERIOD == 0:
                    T_full = _build_full_T(alphas_buf[: it + 1], betas_buf[: it + 1], block_widths=block_widths + [n_curr])
                    eigvals_T, conv_evec = la.eigh(T_full)
                    for k_idx in range(len(eigvals_T)):
                        err_bnd = np.linalg.norm(beta_i, ord=2) * np.abs(conv_evec[-1, k_idx])
                        if err_bnd < reort_eps:
                            s_k = conv_evec[:, k_idx]
                            # Q_list[0] holds all it+1 completed blocks (the current
                            # block q1 is already its last block), spanning the full
                            # subspace s_k indexes. The earlier split that added q1
                            # separately double-counted it and mismatched shapes whenever
                            # a Ritz value locks on a resumed run (where the retained Ritz
                            # pairs are already converged, so the lock fires immediately).
                            ritz_vec = Q_list[0] @ s_k[:, np.newaxis]
                            for _ in range(2):
                                overlap = ritz_vec.conj().T @ q_next
                                if mpi and comm is not None:
                                    overlap = np.ascontiguousarray(overlap, dtype=complex)
                                    comm.Allreduce(MPI.IN_PLACE, overlap, op=MPI.SUM)
                                q_next -= ritz_vec @ overlap

            if reort in (Reort.PARTIAL, Reort.SELECTIVE):
                # Pass block_widths + [n_curr] so the current block (index it, already
                # appended to Q_list) is in the width table apply_reort indexes — matching
                # block_lanczos_cy. Without it the W-recurrence's bad-block columns are
                # mis-mapped, which silently loses orthogonality when resuming on a
                # restarted (e.g. IRLM-compressed) basis.
                q_next, W = apply_reort(q_next, Q_list, W, reort, mpi, comm, block_widths + [n_curr])

        if converged(alphas_buf[: it + 1], betas_buf[: it + 1], verbose=verbose, block_widths=block_widths + [n_curr]):
            block_widths.append(n_curr)
            it += 1
            break

        q[0] = q[1]
        q[1] = q_next
        if build_krylov_basis:
            Q_list[0] = np.concatenate([Q_list[0], q_next], axis=1)
        block_widths.append(n_curr)
        it += 1

    if verbose:
        print(f"Converged at iteration {it}")

    res_Q = Q_list[0] if build_krylov_basis else None
    res_alphas = alphas_buf[:it]
    res_betas = betas_buf[:it]
    if return_widths:
        if return_W:
            return res_alphas, res_betas, res_Q, W, block_widths
        return res_alphas, res_betas, res_Q, block_widths
    else:
        if return_W:
            return res_alphas, res_betas, res_Q, W
        return res_alphas, res_betas, res_Q



import scipy.sparse as sps
from impurityModel.ed.ManyBodyUtils import inner_multi, add_scaled_multi

cpdef bint is_array(object V):
    if isinstance(V, (np.ndarray, sps.spmatrix, sps.sparray)):
        return True
    if isinstance(V, list) and len(V) > 0 and isinstance(V[0], np.ndarray):
        return True
    return False

cpdef object block_inner(object V, object W, bint mpi=False, object comm=None):
    if is_array(V):
        if isinstance(V, list):
            V = np.column_stack(V)
        if isinstance(W, list):
            W = np.column_stack(W)
        res = np.ascontiguousarray(np.conj(V.T) @ W)
        if mpi and comm is not None:
            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res
    else:
        res = inner_multi(V, W)
        if mpi and comm is not None:
            comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        return res

cpdef object block_apply(object H, object V, object basis=None, bint mpi=False, double slaterWeightMin=0.0):
    if is_array(V) or getattr(H, "is_array_operator", False) or isinstance(H, np.ndarray) or isinstance(H, sps.spmatrix):
        if isinstance(V, list) and isinstance(V[0], np.ndarray):
            V_arr = np.column_stack(V)
            res = H @ V_arr
        else:
            res = H @ V

        if mpi and basis is not None and getattr(basis, 'comm', None) is not None:
            comm = basis.comm
            res_global = np.ascontiguousarray(res)
            comm.Allreduce(MPI.IN_PLACE, res_global, op=MPI.SUM)
            rank = comm.rank
            counts = np.empty(comm.size, dtype=int)
            local_N = V_arr.shape[0] if isinstance(V, list) else V.shape[0]
            comm.Allgather(np.array([local_N], dtype=int), counts)
            offsets = np.array([np.sum(counts[:r]) for r in range(comm.size)], dtype=int)
            return res_global[offsets[rank] : offsets[rank] + local_N, :]
        return res
    else:
        wp = H.apply_multi(V, cutoff=slaterWeightMin)
        if mpi and basis is not None and basis.comm is not None:
            wp = basis.redistribute_psis(wp)
        return wp

cpdef object block_add_scaled(object V, object W, object alpha, double slaterWeightMin=0.0):
    if is_array(V):
        V += W @ alpha
    else:
        add_scaled_multi(V, W, alpha)
        if slaterWeightMin > 0:
            for st in V:
                st.prune(slaterWeightMin)
    return V

cpdef object block_combine(object Q, object Y, double slaterWeightMin=0.0):
    if is_array(Q):
        if isinstance(Q, list):
            Q = np.column_stack(Q)
        return block_combine_array(Q, Y)
    else:
        from impurityModel.ed.BlockLanczos import block_combine_sparse
        return block_combine_sparse(Q, Y, slaterWeightMin)

cpdef tuple block_orthogonalize(object wp, object Q, object overlaps=None, bint mpi=False, object comm=None):
    if is_array(wp):
        if isinstance(Q, list):
            Q = np.column_stack(Q)
        return block_orthogonalize_array(wp, Q, overlaps, comm if mpi else None)
    else:
        from impurityModel.ed.BlockLanczos import block_orthogonalize_sparse
        return block_orthogonalize_sparse(wp, Q, overlaps, comm if mpi else None)

cpdef tuple block_normalize(object wp, bint mpi=False, object comm=None, double slaterWeightMin=0.0):
    if is_array(wp):
        return block_normalize_array(wp, comm if mpi else None)
    else:
        from impurityModel.ed.BlockLanczos import block_normalize_sparse
        return block_normalize_sparse(wp, mpi, comm, slaterWeightMin)

def block_lanczos_array(*args, **kwargs):
    return block_lanczos_array_cy(*args, **kwargs)


cpdef tuple apply_reort(object wp, object Q_list, object W, object reort, bint mpi, object comm, list block_widths):
    from impurityModel.ed.BlockLanczosArray import Reort, REORT_TOL, BAD_BLOCK_TOL, EPS
    cdef list bad_block_idx = []
    cdef int j, col_start, col_end, w_j
    cdef list bad_cols
    cdef object Q_bad
    cdef int active_k

    if is_array(wp):
        active_k = wp.shape[1]
    else:
        active_k = len(wp)

    if reort == Reort.FULL or reort == Reort.PERIODIC:
        for _ in range(2):
            wp, _ = block_orthogonalize(wp, Q_list, mpi=mpi, comm=comm)

    elif reort in (Reort.PARTIAL, Reort.SELECTIVE):
        if W is not None:
            n_blks = W.shape[1] - 1  # W[-1, :n_blks]
            if not mpi or comm is None or comm.rank == 0:
                if np.max(np.abs(W[-1, :n_blks])) > REORT_TOL:
                    bad_block_idx = [j for j in range(n_blks) if np.max(np.abs(W[-1, j])) > BAD_BLOCK_TOL]
            if mpi and comm is not None:
                bad_block_idx = comm.bcast(bad_block_idx, root=0)

            if bad_block_idx:
                bad_cols = []
                for j in bad_block_idx:
                    col_start = sum(block_widths[:j])
                    col_end = col_start + block_widths[j]
                    bad_cols.extend(range(col_start, col_end))

                if is_array(Q_list):
                    Q_mat = Q_list if not isinstance(Q_list, list) else Q_list[0]
                    Q_bad = Q_mat[:, bad_cols]
                else:
                    Q_bad = [Q_list[col] for col in bad_cols]

                for _ in range(2):
                    wp, _ = block_orthogonalize(wp, Q_bad, mpi=mpi, comm=comm)

                for j in bad_block_idx:
                    w_j = block_widths[j]
                    W[-1, j, :w_j, :active_k] = EPS * np.eye(w_j, active_k, dtype=complex)

    return wp, W
