import numpy as np
import scipy.linalg as sp
from typing import Callable, Tuple
from impurityModel.ed.ManyBodyUtils import ManyBodyState, inner_multi, add_scaled_multi
from impurityModel.ed.lanczos import block_lanczos_sparse, _reorthogonalize_sparse, Reort
from mpi4py import MPI
import scipy.sparse



from impurityModel.ed.block_math import (
    is_array, block_inner, block_apply, block_add_scaled, 
    block_combine, block_orthogonalize, block_normalize
)

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
    
    is_arr = is_array(h_op)
    n = psi0.shape[1] if is_arr and len(psi0.shape) == 2 else len(psi0)

        
    k_blocks = int(np.ceil(num_wanted / n))
    m = max_subspace_blocks

    track_W = reort in (Reort.PARTIAL, Reort.SELECTIVE)
    
    if is_arr:
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
            reort=reort, verbose=verbose, inner_func=inner_multi,
            orth_tol=1e-12, return_Q=True, track_full_W=track_W, return_W=True
        )
    W_full = W_res[0] if track_W and W_res else None
    
    m_actual = len(alphas)
    if is_arr:
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
        
    from impurityModel.ed.lanczos import _build_full_T
    T_full = _build_full_T(alphas, betas[:-1] if len(betas) == m_actual else betas)
    
    if m_actual <= k_blocks:
        if verbose:
            print(f"Invariant subspace of size {m_actual} blocks found. Stopping early.")
        eigvals, eigvecs = sp.eigh(T_full)
        wanted_indices = np.argsort(eigvals)[:num_wanted]
        final_eigvals = eigvals[wanted_indices]
        
        final_eigvecs = block_combine(Q_list, eigvecs[:, wanted_indices], slaterWeightMin)
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
        
        if is_arr:
            Q_list = Q_list[:, :m_actual * n]
        else:
            Q_list = Q_list[:m_actual * n]
        Q_list = block_combine(Q_list, Y_k, slaterWeightMin)
        
        T_full = np.zeros((m * n, m * n), dtype=complex)
        T_full[:k_blocks * n, :k_blocks * n] = T_k
        
        # New basis block is exactly q_m
        if q_m is None:
            if is_arr:
                q_last = Q_list[:, (k_blocks-1)*n : k_blocks*n]
            else:
                q_last = Q_list[(k_blocks-1)*n : k_blocks*n]
                
            wp = block_apply(h_op, q_last, basis, mpi)
            wp, _ = block_orthogonalize(wp, Q_list, mpi=mpi, comm=comm)
            if is_arr: wp, _ = block_orthogonalize(wp, Q_list, mpi=mpi, comm=comm) # array does 2x for stability
            
            q_m, beta_res = block_normalize(wp, mpi, comm, slaterWeightMin)
            
        if is_arr:
            Q_list = np.concatenate([Q_list, q_m], axis=1)
        else:
            Q_list.extend([st.copy() for st in q_m])
        
        Y_last = Y_k[-n:, :]
        cross_term = beta_res @ Y_last
        
        T_full[k_blocks*n:(k_blocks+1)*n, :k_blocks*n] = cross_term
        T_full[:k_blocks*n, k_blocks*n:(k_blocks+1)*n] = np.conj(cross_term.T)
        
        q1 = q_m
        for i in range(k_blocks, m):
            wp = block_apply(h_op, q1, basis, mpi)
            overlaps = block_inner(Q_list, wp, mpi, comm)
            
            T_full[:(i+1)*n, i*n:(i+1)*n] = overlaps
            T_full[i*n:(i+1)*n, :(i+1)*n] = np.conj(overlaps.T)
            
            wp, _ = block_orthogonalize(wp, Q_list, overlaps=overlaps, mpi=mpi, comm=comm)
            if is_arr: wp, _ = block_orthogonalize(wp, Q_list, mpi=mpi, comm=comm)
            
            q_next, beta_i = block_normalize(wp, mpi, comm, slaterWeightMin)
            
            if i < m - 1:
                T_full[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = beta_i
                T_full[i*n:(i+1)*n, (i+1)*n:(i+2)*n] = np.conj(beta_i.T)
                
                if is_arr:
                    Q_list = np.concatenate([Q_list, q_next], axis=1)
                else:
                    Q_list.extend([st.copy() for st in q_next])
                q1 = q_next
            else:
                beta_res = beta_i
                q_m = q_next

    eigvals, eigvecs = sp.eigh(T_full)
    wanted_indices = np.argsort(eigvals)[:num_wanted]
    final_eigvals = eigvals[wanted_indices]
    
    final_eigvecs = block_combine(Q_list, eigvecs[:, wanted_indices], slaterWeightMin)
    return final_eigvals, final_eigvecs

