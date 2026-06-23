# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True, freethreading_compatible=True

cimport cython
import numpy as np
cimport numpy as np
import scipy.linalg as la
import scipy.sparse as sps

from libc.math cimport sqrt

from mpi4py import MPI
from enum import Enum


class Reort(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2
    PERIODIC = 3
    SELECTIVE = 4


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
    double eps=0.0,
    int N=1
):
    cdef int i = alphas.shape[0] - 1
    cdef int n = alphas.shape[1]
    if eps == 0.0:
        eps = np.finfo(float).eps
    
    cdef np.ndarray[double complex, ndim=4] W_out = np.zeros((2, i + 2, n, n), dtype=complex)
    cdef np.ndarray[double complex, ndim=3] w_bar = np.zeros((i + 2, n, n), dtype=complex)
    
    w_bar[i + 1, :, :] = np.identity(n)
    w_bar[i, :, :] = eps * N * la.solve_triangular(betas[i], betas[0], lower=False, trans="C", check_finite=False)
    
    if i == 0:
        W_out[0, : i + 1] = W[1]
        W_out[1, : i + 2] = w_bar
        return W_out

    w_bar[0] = W[1, 1] @ betas[0] + W[1, 0] @ alphas[0] - alphas[i] @ W[1, 0] - betas[i - 1] @ W[0, 0]
    w_bar[0] = la.solve_triangular(betas[i], w_bar[0], lower=False, trans="C", check_finite=False)
    
    cdef np.ndarray[double complex, ndim=3] betas_conj_T = np.conj(np.transpose(betas[: i - 1], axes=[0, 2, 1]))
    
    w_bar[1:i] = (
        W[1, 2 : i + 1] @ betas[1:i]
        + W[1, 1:i] @ alphas[1:i]
        - alphas[i][np.newaxis, :, :] @ W[1, 1:i]
        + W[1, : i - 1] @ betas_conj_T
        - betas[i - 1][np.newaxis, :, :] @ W[0, 1:i]
    )
    
    w_bar[1:i] = np.linalg.solve(betas[i][np.newaxis, :, :], w_bar[1:i])
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


cpdef np.ndarray _build_full_T(np.ndarray[double complex, ndim=3] alphas, np.ndarray[double complex, ndim=3] betas, comm=None):
    cdef int m = alphas.shape[0]
    cdef int n
    if m == 0:
        return np.zeros((0, 0), dtype=complex)
    n = alphas.shape[1]
    cdef np.ndarray[double complex, ndim=2] T = np.zeros((m * n, m * n), dtype=complex)
    cdef int i
    for i in range(m):
        T[i * n : (i + 1) * n, i * n : (i + 1) * n] = alphas[i]
        if i < m - 1:
            T[i * n : (i + 1) * n, (i + 1) * n : (i + 2) * n] = np.conj(betas[i].T)
            T[(i + 1) * n : (i + 2) * n, i * n : (i + 1) * n] = betas[i]
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
    cdef np.ndarray L = la.cholesky(M, lower=True)
    cdef np.ndarray beta = np.conj(L.T)
    cdef np.ndarray beta_inv = la.inv(beta)
    cdef np.ndarray q_next = wp @ beta_inv
    return q_next, beta

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
    cdef int i, j, l
    cdef double complex val
    
    # Initialize C with beta * C
    if beta == 0.0:
        for i in range(m):
            for j in range(n):
                C[i, j] = 0.0
    elif beta != 1.0:
        for i in range(m):
            for j in range(n):
                C[i, j] = C[i, j] * beta
                
    # Compute C += alpha * A * B
    if transA == b'N' and transB == b'N':
        for i in range(m):
            for l in range(k):
                val = alpha * A[i, l]
                for j in range(n):
                    C[i, j] = C[i, j] + val * B[l, j]
    elif transA == b'C' and transB == b'N':
        for l in range(k):
            for i in range(m):
                val = alpha * conj(A[l, i])
                for j in range(n):
                    C[i, j] = C[i, j] + val * B[l, j]
    elif transA == b'N' and transB == b'C':
        for i in range(m):
            for j in range(n):
                val = 0.0
                for l in range(k):
                    val = val + A[i, l] * conj(B[j, l])
                C[i, j] = C[i, j] + alpha * val
    elif transA == b'C' and transB == b'C':
        for i in range(m):
            for j in range(n):
                val = 0.0
                for l in range(k):
                    val = val + conj(A[l, i]) * conj(B[j, l])
                C[i, j] = C[i, j] + alpha * val

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
    **kwargs
):
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

    alphas = alphas_arr
    betas = betas_arr

    cdef bint is_sparse = sps.issparse(h_op)
    cdef bint is_dense = isinstance(h_op, np.ndarray)

    cdef int it = start_it
    cdef double reort_eps = np.sqrt(np.finfo(float).eps)
    cdef int period = kwargs.get("reort_period", 5)
    cdef int max_iter = kwargs.get("max_iter", int(np.ceil(h_op.shape[0] / n if is_sparse or is_dense else N / n)))
    
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
        # scipy chooses int32 index arrays for small/empty matrices (e.g. a rank
        # whose local column slice is empty) and int64 otherwise.  The typed
        # memoryviews below are ``long`` (int64), so coerce to avoid a
        # buffer-dtype-mismatch ValueError that would otherwise abort this rank
        # only -- deadlocking the collectives in the iteration loop.
        h_indices = np.ascontiguousarray(h_op.indices, dtype=np.int64)
        h_indptr = np.ascontiguousarray(h_op.indptr, dtype=np.int64)
    elif is_dense:
        h_dense = np.ascontiguousarray(h_op)

    cdef np.ndarray wp_arr = np.empty((N, n), dtype=complex, order='C')
    cdef double complex[:, ::1] wp = wp_arr
    cdef double complex[:, ::1] q1, q0
    cdef double complex[:, ::1] alpha_i = np.empty((n, n), dtype=complex, order='C')
    cdef double complex[:, ::1] beta_prev_dag = np.zeros((n, n), dtype=complex, order='C')

    cdef np.ndarray wp_global = np.empty((global_N, n), dtype=complex, order='C') if mpi else None
    cdef double complex[:, ::1] wp_g = wp_global

    while it < max_iter:
        q1 = np.ascontiguousarray(q[1])

        if is_sparse:
            with nogil:
                apply_sparse_csr_nogil(global_N, N, n, h_data, h_indices, h_indptr, q1, wp_g if mpi else wp)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, wp_global, op=MPI.SUM)
                wp_arr[:] = wp_global[offsets[rank] : offsets[rank] + N, :]
        elif is_dense:
            with nogil:
                apply_dense_nogil(global_N, N, n, h_dense, q1, wp_g if mpi else wp)
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
                
        with nogil:
            matmul_nogil(n, n, N, 1.0, q1, b'C', wp, b'N', 0.0, alpha_i)
            
        alpha_i_arr = np.asarray(alpha_i).copy()
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, alpha_i_arr, op=MPI.SUM)
            np.asarray(alpha_i)[:] = alpha_i_arr
        
        alphas = np.append(alphas, [alpha_i_arr], axis=0)
        betas = np.append(betas, np.zeros((1, n, n), dtype=complex), axis=0)

        with nogil:
            matmul_nogil(N, n, n, -1.0, q1, b'N', alpha_i, b'N', 1.0, wp)

        if it > 0:
            beta_prev_dag_arr = np.conj(betas[it - 1].T).copy()
            beta_prev_dag = np.ascontiguousarray(beta_prev_dag_arr)
            q0 = np.ascontiguousarray(q[0])
            with nogil:
                matmul_nogil(N, n, n, -1.0, q0, b'N', beta_prev_dag, b'N', 1.0, wp)

        if reort == Reort.FULL or (reort == Reort.PERIODIC and it > 0 and it % period == 0):
            if not build_krylov_basis:
                raise RuntimeError("Krylov basis must be built for reorthogonalization")
            
            Q_mat = np.ascontiguousarray(Q_list[0])
            overlap = Q_mat.conj().T @ wp_arr
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, overlap, op=MPI.SUM)
            wp_arr -= Q_mat @ overlap

        M = wp_arr.conj().T @ wp_arr
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)

        if np.any(np.isnan(M)) or np.any(np.isinf(M)):
            alphas = np.append(alphas, [np.asarray(alpha_i).copy()], axis=0)
            break

        try:
            eigvals, eigvecs = la.eigh(M)
        except Exception:
            alphas = np.append(alphas, [np.asarray(alpha_i).copy()], axis=0)
            break
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        max_eig = eigvals[0] if len(eigvals) > 0 else 0
        thresh = max(1e-12, 1e-12 * max_eig)

        k = 0
        for e in eigvals:
            if e > thresh:
                k += 1

        if k == 0:
            break

        beta_i = np.zeros((n, n), dtype=complex)
        beta_inv = np.zeros((n, n), dtype=complex)

        if k == n:
            try:
                L = la.cholesky(M, lower=True)
                beta_i = np.conj(L.T)
                beta_inv = la.inv(beta_i)
            except la.LinAlgError:
                L_sqrt = np.sqrt(eigvals[:k])
                beta_i[:k, :] = L_sqrt[:, np.newaxis] * np.conj(eigvecs[:, :k].T)
                beta_inv[:, :k] = eigvecs[:, :k] / L_sqrt[np.newaxis, :]
        else:
            L_sqrt = np.sqrt(eigvals[:k])
            beta_i[:k, :] = L_sqrt[:, np.newaxis] * np.conj(eigvecs[:, :k].T)
            beta_inv[:, :k] = eigvecs[:, :k] / L_sqrt[np.newaxis, :]
            
            # Since q_next = wp_arr @ beta_inv, vectors after k are 0.
            # We want to re-orthogonalize the active k vectors against Q if needed,
            # but to make sure they are orthonormal:
            # The mathematical definition gives us orthonormal vectors in the range [0:k].
            # However, wp_arr might have noise.

        betas[it] = beta_i

        q_next = wp_arr @ beta_inv
        
        # Zero out the parts that correspond to the deflated vectors
        for i in range(k, n):
            for j in range(N):
                q_next[j, i] = 0.0

        if reort in (Reort.PARTIAL, Reort.SELECTIVE):
            if not build_krylov_basis:
                raise RuntimeError("Krylov basis must be built for reorthogonalization")
            if W is None:
                if start_it > 0:
                    W = np.zeros((2, start_it + 1, n, n), dtype=complex)
                    for j in range(start_it):
                        Q_j = Q_list[0][:, j * n : (j + 1) * n]
                        W[1, j] = Q_j.conj().T @ wp_arr
                        if j < start_it - 1:
                            W[0, j] = Q_j.conj().T @ q[0]
                    if mpi:
                        comm.Allreduce(MPI.IN_PLACE, W, op=MPI.SUM)
                    W[1, start_it] = np.eye(n)
                    W[0, start_it - 1] = np.eye(n)
                else:
                    W = np.zeros((2, 1, n, n), dtype=complex)
                    W[1, 0] = np.eye(n)
            elif W.shape[1] < it + 1:
                W_new = np.zeros((2, it + 1, n, n), dtype=complex)
                W_new[:, : W.shape[1]] = W
                W = W_new

            W = estimate_orthonormality(W, alphas[: it + 1], betas[: it + 1], eps=np.finfo(float).eps)
            
            reort_eps = np.sqrt(np.finfo(float).eps)

            if reort == Reort.SELECTIVE:
                if it > 0:
                    T_full = _build_full_T(alphas[: it + 1], betas[: it + 1])
                    # Use deterministic dense eigh (no ARPACK) so the convergence
                    # decision is reproducible.  W has shape (2, k, n, n) where
                    # axis 0 is (prev_row, current_row) — index as W[1, j], not
                    # W[j, it+1] which confuses the two axes.
                    eigvals_T, conv_evec = la.eigh(T_full)
                    for k in range(len(eigvals_T)):
                        err_bnd = np.linalg.norm(betas[it], ord=2) * np.abs(conv_evec[-1, k])
                        if err_bnd < reort_eps:
                            s_k = conv_evec[:, k]
                            s_k_blocks = s_k.reshape(it + 1, n)
                            w_ritz_k = np.zeros(n, dtype=complex)
                            for j in range(it + 1):
                                w_ritz_k += np.conj(s_k_blocks[j]) @ W[1, j]
                            if np.max(np.abs(w_ritz_k)) > reort_eps:
                                ritz_vec = Q_list[0] @ s_k[:, np.newaxis]
                                for _ in range(2):
                                    overlap = ritz_vec.conj().T @ q_next
                                    q_next -= ritz_vec @ overlap

            if reort in (Reort.PARTIAL, Reort.SELECTIVE):
                n_blks = it + 1
                bad_block_idx = [j for j in range(n_blks) if np.max(np.abs(W[-1, j])) > reort_eps * 1e-2]
                if bad_block_idx:
                    bad_cols = []
                    for j in bad_block_idx:
                        bad_cols.extend(range(j * n, (j + 1) * n))
                    Q_bad = Q_list[0][:, bad_cols]
                    for _ in range(2):
                        overlap = Q_bad.conj().T @ q_next
                        q_next -= Q_bad @ overlap
                    for j in bad_block_idx:
                        W[-1, j] = np.finfo(float).eps * np.eye(n, dtype=complex)

        if converged(alphas, betas, verbose=verbose):
            break

        q[0] = q[1]
        q[1] = q_next
        if build_krylov_basis:
            Q_list[0] = np.concatenate([Q_list[0], q_next], axis=1)
        it += 1


    if verbose:
        print(f"Coverged at iteration {it + 1}")

    res_Q = Q_list[0] if build_krylov_basis else None
    if return_W:
        return alphas, betas, res_Q, W
    return alphas, betas, res_Q



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

