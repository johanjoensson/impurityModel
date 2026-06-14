import numpy as np
import scipy.linalg as sp
from typing import Callable, Tuple
from impurityModel.ed.ManyBodyUtils import ManyBodyState, inner_multi, add_scaled_multi
from impurityModel.ed.lanczos import block_lanczos_sparse, _reorthogonalize_sparse, Reort
from mpi4py import MPI
import scipy.sparse

def _build_full_T(alphas, betas):
    m = len(alphas)
    n = alphas[0].shape[0]
    T = np.zeros((m * n, m * n), dtype=complex)
    for i in range(m):
        T[i * n : (i + 1) * n, i * n : (i + 1) * n] = alphas[i]
        if i < m - 1 and i < len(betas):
            T[(i + 1) * n : (i + 2) * n, i * n : (i + 1) * n] = betas[i]
            T[i * n : (i + 1) * n, (i + 1) * n : (i + 2) * n] = np.conj(betas[i].T)
    return T

def thick_restarted_block_lanczos(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0,
    reort: Reort = Reort.PARTIAL,
):
    mpi = basis is not None and getattr(basis, 'comm', None) is not None
    comm = basis.comm if mpi else None
    
    is_array = isinstance(h_op, (np.ndarray, scipy.sparse.spmatrix))
    if is_array:
        n = psi0.shape[1] if len(psi0.shape) == 2 else 1
        inner_func = lambda x, y: x.conj().T @ y
    else:
        n = len(psi0)
        inner_func = inner_multi
        
    k_blocks = int(np.ceil(num_wanted / n))
    m = max_subspace_blocks

    track_W = reort in (Reort.PARTIAL, Reort.SELECTIVE)
    
    if is_array:
        from impurityModel.ed.lanczos import block_lanczos_array
        alphas, betas, Q_list, *W_res = block_lanczos_array(
            psi0, h_op, lambda a, b, **kw: False, max_iter=m, 
            reort=reort, verbose=verbose,
            orth_tol=1e-12, track_full_W=track_W, return_W=True
        )
    else:
        from impurityModel.ed.lanczos import block_lanczos_sparse
        alphas, betas, Q_list, *W_res = block_lanczos_sparse(
            psi0, h_op, basis, lambda a, b, **kw: False, slaterWeightMin=slaterWeightMin, max_iter=m, 
            reort=reort, verbose=verbose, inner_func=inner_func,
            orth_tol=1e-12, return_Q=True, track_full_W=track_W, return_W=True
        )
    W_full = W_res[0] if track_W and W_res else None
    
    m_actual = len(alphas)
    if is_array:
        if Q_list.shape[1] > m_actual * n:
            q_m = Q_list[:, m_actual * n : (m_actual + 1) * n].copy()
            Q_list = Q_list[:, :m_actual * n]
        else:
            q_m = None
    else:
        if len(Q_list) > m_actual * n:
            q_m = [Q_list[i].copy() for i in range(m_actual * n, (m_actual + 1) * n)]
            Q_list = Q_list[:m_actual * n]
        else:
            q_m = None
        
    T_full = _build_full_T(alphas, betas[:-1] if len(betas) == m_actual else betas)
    
    if m_actual <= k_blocks:
        if verbose:
            print(f"Invariant subspace of size {m_actual} blocks found. Stopping early.")
        eigvals, eigvecs = sp.eigh(T_full)
        wanted_indices = np.argsort(eigvals)[:num_wanted]
        final_eigvals = eigvals[wanted_indices]
        
        if is_array:
            final_eigvecs = Q_list @ np.ascontiguousarray(eigvecs[:, wanted_indices], dtype=complex)
        else:
            final_eigvecs = [ManyBodyState() for _ in range(len(wanted_indices))]
            add_scaled_multi(final_eigvecs, Q_list, np.ascontiguousarray(eigvecs[:, wanted_indices], dtype=complex))
        return final_eigvals, final_eigvecs
        
    beta_res = betas[-1]

    for restart in range(max_restarts):
        eigvals, eigvecs = sp.eigh(T_full)
        res_norms = np.linalg.norm(beta_res @ eigvecs[-n:, :], axis=0)
        wanted_indices = np.argsort(eigvals)[:num_wanted]
        max_wanted_res = np.max(res_norms[wanted_indices])
        
        if verbose and (not mpi or comm.Get_rank() == 0):
            print(f"Restart {restart:3d} | Min Eigval: {eigvals[0]:.6f} | Max Wanted Residual: {max_wanted_res:.2e}")
            
        if max_wanted_res < tol:
            if verbose: print("Converged!")
            break
            
        keep_indices = np.argsort(eigvals)[:k_blocks * n]
        Y_k = eigvecs[:, keep_indices]
        T_k = np.diag(eigvals[keep_indices])
        
        if is_array:
            Q_k_states = Q_list[:, :m_actual * n] @ np.ascontiguousarray(Y_k, dtype=complex)
            Q_list = Q_k_states
        else:
            Q_k_states = [ManyBodyState() for _ in range(k_blocks * n)]
            add_scaled_multi(Q_k_states, Q_list[:m_actual * n], np.ascontiguousarray(Y_k, dtype=complex))
            Q_list = Q_k_states
        
        T_full = np.zeros((m * n, m * n), dtype=complex)
        T_full[:k_blocks * n, :k_blocks * n] = T_k
        
        # New basis block is exactly q_m
        if q_m is None:
            if is_array:
                q_last = Q_list[:, (k_blocks-1)*n : k_blocks*n]
                wp = h_op @ q_last
            else:
                q_last = [Q_list[i] for i in range((k_blocks-1)*n, k_blocks*n)]
                wp = [ManyBodyState() for _ in range(n)]
                wp_tmp = h_op.apply_multi(q_last)
                if mpi: wp_tmp = basis.redistribute_psis(wp_tmp)
                for j in range(n): wp[j] += wp_tmp[j]
                
            if is_array:
                overlaps = Q_list.conj().T @ wp
                wp -= Q_list @ overlaps
                overlaps2 = Q_list.conj().T @ wp
                wp -= Q_list @ overlaps2
            else:
                _reorthogonalize_sparse(wp, Q_list, None, inner_func, mpi, comm, n)
            M = inner_func(wp, wp)
            if mpi and not is_array: comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
            L = sp.cholesky(M + np.eye(n)*1e-14, lower=True)
            beta_inv = sp.inv(np.conj(L.T))
            
            if is_array:
                q_m = wp @ beta_inv
            else:
                q_m = [ManyBodyState() for _ in range(n)]
                add_scaled_multi(q_m, wp, beta_inv)
                for st in q_m: st.prune(slaterWeightMin)
            beta_res = np.conj(L.T)
            
        if is_array:
            Q_list = np.concatenate([Q_list, q_m], axis=1)
        else:
            Q_list.extend([st.copy() for st in q_m])
        
        Y_last = Y_k[-n:, :]
        cross_term = beta_res @ Y_last
        
        T_full[k_blocks*n:(k_blocks+1)*n, :k_blocks*n] = cross_term
        T_full[:k_blocks*n, k_blocks*n:(k_blocks+1)*n] = np.conj(cross_term.T)
        
        q1 = q_m
        for i in range(k_blocks, m):
            if is_array:
                wp = h_op @ q1
                overlaps = inner_func(Q_list, wp)
            else:
                wp = [ManyBodyState() for _ in range(n)]
                wp_tmp = h_op.apply_multi(q1)
                if mpi: wp_tmp = basis.redistribute_psis(wp_tmp)
                for j in range(n): wp[j] += wp_tmp[j]
                overlaps = inner_func(Q_list, wp)
                if mpi: comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
            
            T_full[:(i+1)*n, i*n:(i+1)*n] = overlaps
            T_full[i*n:(i+1)*n, :(i+1)*n] = np.conj(overlaps.T)
            
            if is_array:
                wp -= Q_list @ overlaps
                overlaps2 = Q_list.conj().T @ wp
                wp -= Q_list @ overlaps2
            else:
                add_scaled_multi(wp, Q_list, -overlaps)
                _reorthogonalize_sparse(wp, Q_list, None, inner_func, mpi, comm, n)
            
            M = inner_func(wp, wp)
            if mpi and not is_array: comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
            
            try:
                L = sp.cholesky(M, lower=True)
            except sp.LinAlgError:
                L = sp.cholesky(M + np.eye(n) * 1e-14, lower=True)
            beta_i = np.conj(L.T)
            
            if i < m - 1:
                T_full[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = beta_i
                T_full[i*n:(i+1)*n, (i+1)*n:(i+2)*n] = np.conj(beta_i.T)
                
                beta_inv = sp.inv(beta_i)
                if is_array:
                    q_next = wp @ beta_inv
                    Q_list = np.concatenate([Q_list, q_next], axis=1)
                else:
                    q_next = [ManyBodyState() for _ in range(n)]
                    add_scaled_multi(q_next, wp, beta_inv)
                    for st in q_next: st.prune(slaterWeightMin)
                    Q_list.extend([st.copy() for st in q_next])
                q1 = q_next
            else:
                beta_res = beta_i
                beta_inv = sp.inv(beta_i)
                if is_array:
                    q_m = wp @ beta_inv
                else:
                    q_m = [ManyBodyState() for _ in range(n)]
                    add_scaled_multi(q_m, wp, beta_inv)
                    for st in q_m: st.prune(slaterWeightMin)

    eigvals, eigvecs = sp.eigh(T_full)
    wanted_indices = np.argsort(eigvals)[:num_wanted]
    final_eigvals = eigvals[wanted_indices]
    
    if is_array:
        final_eigvecs = Q_list @ np.ascontiguousarray(eigvecs[:, wanted_indices], dtype=complex)
    else:
        final_eigvecs = [ManyBodyState() for _ in range(len(wanted_indices))]
        add_scaled_multi(final_eigvecs, Q_list, np.ascontiguousarray(eigvecs[:, wanted_indices], dtype=complex))
        
    return final_eigvals, final_eigvecs

