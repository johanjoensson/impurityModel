import numpy as np
import scipy.linalg as sp
import time
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, inner_multi, add_scaled_multi

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

def _extract_blocks(T, m, n):
    alphas = np.zeros((m, n, n), dtype=complex)
    betas = np.zeros((m - 1, n, n), dtype=complex)
    for i in range(m):
        alphas[i] = T[i * n : (i + 1) * n, i * n : (i + 1) * n]
        if i < m - 1:
            betas[i] = T[(i + 1) * n : (i + 2) * n, i * n : (i + 1) * n]
    return alphas, betas

def implicitly_restarted_block_lanczos(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0,
    reort=None,
):
    from impurityModel.ed.lanczos import block_lanczos_sparse, _reorthogonalize_sparse, eigsh, Reort
    
    if reort is None:
        reort = Reort.PARTIAL

    n = len(psi0) if isinstance(psi0, list) else (psi0.shape[1] if len(psi0.shape) == 2 else 1)
    k_blocks = int(np.ceil(num_wanted / n))
    m = max_subspace_blocks

    if m <= k_blocks:
        raise ValueError("max_subspace_blocks must be strictly greater than ceil(num_wanted / n)")

    mpi = basis is not None and getattr(basis, 'comm', None) is not None
    comm = basis.comm if mpi else None

    # Initial Lanczos run to build subspace
    track_W = reort in (Reort.PARTIAL, Reort.SELECTIVE)
    
    if isinstance(psi0, np.ndarray) or (isinstance(psi0, list) and isinstance(psi0[0], np.ndarray)):
        from impurityModel.ed.lanczos import block_lanczos_array
        if isinstance(psi0, list):
            psi0_arr = np.concatenate(psi0, axis=1) if len(psi0[0].shape) == 2 else np.column_stack(psi0)
        else:
            psi0_arr = psi0
        alphas, betas, Q_list, *W_res = block_lanczos_array(
            psi0_arr, h_op, lambda a,b,**kw: False, max_iter=m,
            reort=reort, verbose=verbose, return_W=True, track_full_W=track_W,
            orth_tol=1e-12, return_Q=True
        )
        is_array = True
    else:
        alphas, betas, Q_list, *W_res = block_lanczos_sparse(
            psi0, h_op, basis, lambda a,b,**kw: False, max_iter=m, slaterWeightMin=slaterWeightMin, 
            reort=reort, verbose=verbose, inner_func=inner_multi,
            orth_tol=1e-12, return_Q=True, track_full_W=track_W, return_W=True
        )
        is_array = False
        
    W_full = W_res[0] if track_W and W_res else None

    for restart in range(max_restarts):
        m_actual = len(alphas)
        
        # Check if we hit invariant subspace before completing first loop
        if m_actual <= k_blocks:
            break
            
        T = _build_full_T(alphas, betas[:-1])
        
        # Compute Ritz pairs
        eigvals, eigvecs = sp.eigh(T)
        
        # Check convergence
        res_norms = np.linalg.norm(betas[-1] @ eigvecs[-n:, :], axis=0)
        
        wanted_indices = np.argsort(eigvals)[:num_wanted]
        unwanted_indices = np.argsort(eigvals)[num_wanted:]
        
        max_wanted_res = np.max(res_norms[wanted_indices])
        if verbose and (not mpi or comm.rank == 0):
            print(f"Restart {restart:3d} | Min Eigval: {eigvals[0]:.6f} | Max Wanted Residual: {max_wanted_res:.2e}")
            
        if max_wanted_res < tol:
            if verbose and (not mpi or comm.rank == 0):
                print("Converged!")
            break
            
        shifts = eigvals[unwanted_indices]
        from impurityModel.ed.givens_qr import implicit_qr_step_block
        
        U_total = np.eye(m_actual * n, dtype=complex)
        T_shifted = T.copy()
        for shift in shifts:
            T_shifted, U_total = implicit_qr_step_block(T_shifted, n, shift, U_total)
            
        U_k = U_total[:, :k_blocks * n]
        alphas_new, betas_new = _extract_blocks(T_shifted, k_blocks, n)
        
        if is_array:
            Q_new = Q_list[:, :m_actual * n] @ np.ascontiguousarray(U_k, dtype=complex)
        else:
            Q_new = [ManyBodyState() for _ in range(k_blocks * n)]
            add_scaled_multi(Q_new, Q_list[:m_actual * n], np.ascontiguousarray(U_k, dtype=complex))
            for st in Q_new:
                st.prune(slaterWeightMin)
        
        Q_list = Q_new
        alphas = alphas_new
        betas = betas_new
        
        if track_W and W_full is not None:
            W_full_2d = np.zeros((m_actual * n, m_actual * n), dtype=complex)
            for i_blk in range(m_actual):
                for j_blk in range(m_actual):
                    W_full_2d[i_blk*n:(i_blk+1)*n, j_blk*n:(j_blk+1)*n] = W_full[i_blk, j_blk]
            W_new_2d = U_k.conj().T @ W_full_2d @ U_k
            W_new = np.zeros((m, m, n, n), dtype=complex)
            for i_blk in range(k_blocks):
                for j_blk in range(k_blocks):
                    W_new[i_blk, j_blk] = W_new_2d[i_blk*n:(i_blk+1)*n, j_blk*n:(j_blk+1)*n]
            W_full = W_new

        alphas_pass = alphas_new
        
        # We must manually compute q_k so that block_lanczos can start at it = k_blocks
        # This preserves alpha_{k-1} exactly from the QR decomposition.
        if is_array:
            q1 = Q_new[:, -n:]
            q_k_unorth = h_op.apply_multi(q1) if hasattr(h_op, 'apply_multi') else h_op @ q1
            # Full orthogonalization against Q_new
            q_k_unorth -= Q_new @ (Q_new.conj().T @ q_k_unorth)
            # Second orthogonalization pass for stability
            q_k_unorth -= Q_new @ (Q_new.conj().T @ q_k_unorth)
            
            beta_i = sp.cholesky(q_k_unorth.conj().T @ q_k_unorth, lower=True)
            beta_inv = sp.inv(beta_i)
            q_k = q_k_unorth @ beta_inv
            Q_list = np.concatenate([Q_new, q_k], axis=1)
            betas_pass = np.concatenate([betas_new, [np.conj(beta_i.T)]], axis=0) if len(betas_new) > 0 else np.array([np.conj(beta_i.T)])
            
            if track_W and W_full is not None:
                W_pass = np.zeros((k_blocks + 1, k_blocks + 1, n, n), dtype=complex)
                W_pass[:k_blocks, :k_blocks] = W_full[:k_blocks, :k_blocks]
                W_pass[k_blocks, k_blocks] = np.eye(n)
            else:
                W_pass = None
                
            from impurityModel.ed.lanczos import block_lanczos_array
            alphas, betas, Q_list, *W_res = block_lanczos_array(
                None, h_op, lambda a,b,**kw: False, max_iter=m,
                reort=reort, verbose=verbose, return_W=True, track_full_W=track_W,
                alphas=alphas_pass, betas=betas_pass, Q=Q_list, W=W_pass,
                orth_tol=1e-12, return_Q=True
            )
        else:
            q1 = Q_new[-n:]
            wp = h_op.apply_multi(q1)
            if basis.comm is not None:
                wp = basis.redistribute_psis(wp)
            from impurityModel.ed.lanczos import _reorthogonalize_sparse
            _reorthogonalize_sparse(wp, Q_new, None, inner_multi, basis.comm is not None, basis.comm if basis.comm is not None else None, n)
            _reorthogonalize_sparse(wp, Q_new, None, inner_multi, basis.comm is not None, basis.comm if basis.comm is not None else None, n)
            
            M = inner_multi(wp, wp)
            if basis.comm is not None:
                basis.comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
            L = sp.cholesky(M, lower=True)
            beta_i = np.conj(L.T)
            beta_inv = sp.inv(beta_i)
            q_next = [ManyBodyState() for _ in range(n)]
            add_scaled_multi(q_next, wp, beta_inv)
            for st in q_next:
                st.prune(slaterWeightMin)
            Q_list = Q_new + q_next
            betas_pass = list(betas_new) + [beta_i]
            
            if track_W and W_full is not None:
                W_pass = np.zeros((k_blocks + 1, k_blocks + 1, n, n), dtype=complex)
                W_pass[:k_blocks, :k_blocks] = W_full[:k_blocks, :k_blocks]
                W_pass[k_blocks, k_blocks] = np.eye(n)
            else:
                W_pass = None
            
            from impurityModel.ed.lanczos import block_lanczos_sparse
            alphas, betas, Q_list, *W_res = block_lanczos_sparse(
                None, h_op, basis, lambda a,b,**kw: False, max_iter=m, slaterWeightMin=slaterWeightMin, 
                reort=reort, verbose=verbose, inner_func=inner_multi,
                alphas=alphas_pass, betas=betas_pass, Q=Q_list, W=W_pass,
                orth_tol=1e-12, return_Q=True, track_full_W=track_W, return_W=True
            )
        W_full = W_res[0] if track_W and W_res else None

    # Final Ritz vectors
    m_actual = len(alphas)
    T_final = _build_full_T(alphas, betas[:-1] if len(betas) == m_actual else betas)
    eigvals, eigvecs = sp.eigh(T_final)
    
    wanted_indices = np.argsort(eigvals)[:num_wanted]
    final_eigvals = eigvals[wanted_indices]
    
    if is_array:
        final_eigvecs = Q_list[:, :m_actual * n] @ np.ascontiguousarray(eigvecs[:, wanted_indices], dtype=complex)
    else:
        final_eigvecs = [ManyBodyState() for _ in range(len(wanted_indices))]
        add_scaled_multi(final_eigvecs, Q_list[:m_actual * n], np.ascontiguousarray(eigvecs[:, wanted_indices], dtype=complex))
        
    return final_eigvals, final_eigvecs
