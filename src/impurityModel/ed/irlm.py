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
        if i < m - 1:
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
    psi0: list,
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
    """
    Computes eigenvalues and eigenvectors using Implicitly Restarted Block Lanczos.
    
    Uses standard Implicit QR shifting using exact shifts. 
    Maintains full block-tridiagonal matrix using QR sweeps.
    """
    from impurityModel.ed.lanczos import block_lanczos_sparse, _reorthogonalize_sparse, eigsh, Reort
    
    if reort is None:
        reort = Reort.PARTIAL

    n = len(psi0)
    k_blocks = int(np.ceil(num_wanted / n))
    m = max_subspace_blocks
    
    if m <= k_blocks:
        raise ValueError("max_subspace_blocks must be strictly greater than ceil(num_wanted / n)")

    mpi = basis.comm is not None
    comm = basis.comm if mpi else None

    # Initial Lanczos run to build subspace
    track_W = reort in (Reort.PARTIAL, Reort.SELECTIVE)
    
    alphas, betas, Q_list, *W_res = block_lanczos_sparse(
        psi0, h_op, basis, lambda a,b,**kw: len(a) == m, slaterWeightMin=slaterWeightMin, 
        reort=reort, verbose=verbose, inner_func=inner_multi,
        orth_tol=1e-12, return_Q=True, track_full_W=track_W, return_W=True
    )
    W_full = W_res[0] if track_W and W_res else None
    
    m_actual = len(alphas)
    if m_actual <= k_blocks:
        if verbose:
            print(f"Invariant subspace of size {m_actual} blocks found. Stopping early.")
        T_final = _build_full_T(alphas, betas[:-1] if len(betas) == m_actual else betas)
        eigvals, eigvecs = sp.eigh(T_final)
        
        wanted_indices = np.argsort(eigvals)[:num_wanted]
        final_eigvals = eigvals[wanted_indices]
        final_eigvecs = [ManyBodyState() for _ in range(len(wanted_indices))]
        add_scaled_multi(final_eigvecs, Q_list, np.ascontiguousarray(eigvecs[:, wanted_indices], dtype=complex))
        return final_eigvals, final_eigvecs
    
    for restart in range(max_restarts):
        m_actual = len(alphas)
        T = _build_full_T(alphas, betas[:-1])
        
        # Compute Ritz pairs
        eigvals, eigvecs = sp.eigh(T)
        
        # Check convergence
        res_norms = np.linalg.norm(betas[-1] @ eigvecs[-n:, :], axis=0)
        
        wanted_indices = np.argsort(eigvals)[:num_wanted]
        unwanted_indices = np.argsort(eigvals)[num_wanted:]
        
        max_wanted_res = np.max(res_norms[wanted_indices])
        if verbose:
            print(f"Restart {restart:3d} | Min Eigval: {eigvals[0]:.6f} | Max Wanted Residual: {max_wanted_res:.2e}")
            
        if max_wanted_res < tol:
            if verbose:
                print("Converged!")
            break
            
        shifts = eigvals[unwanted_indices]
        U_total = np.eye(m_actual * n, dtype=complex)
        T_shifted = T.copy()
        
        for shift in shifts:
            Q_step, _ = sp.qr(T_shifted - shift * np.eye(m_actual * n))
            if np.any(np.isnan(Q_step)):
                print(f"NaN in Q_step! shift={shift}")
            T_shifted = Q_step.conj().T @ T_shifted @ Q_step
            U_total = U_total @ Q_step
            
        if np.any(np.isnan(U_total)):
            print("NaNs in U_total!")
            
        U_k = U_total[:, :k_blocks * n]
        T_k = T_shifted[:k_blocks * n, :k_blocks * n]
        alphas_new, betas_new = _extract_blocks(T_k, k_blocks, n)
        
        Q_k_states = [ManyBodyState() for _ in range(k_blocks * n)]
        add_scaled_multi(Q_k_states, Q_list[:m_actual * n], np.ascontiguousarray(U_k, dtype=complex))
            
        Q_list = Q_k_states
        alphas = list(alphas_new)
        betas = list(betas_new)
        
        q0 = [Q_list[i] for i in range((k_blocks-1)*n, k_blocks*n)] if k_blocks > 1 else [ManyBodyState() for _ in range(n)]
        q1 = [Q_list[i] for i in range((k_blocks-1)*n, k_blocks*n)]
        
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

        for i in range(k_blocks, m):
            wp = [ManyBodyState() for _ in range(n)]
            
            wp_tmp = h_op.apply_multi(q1)
            if mpi:
                wp_tmp = basis.redistribute_psis(wp_tmp)
            for j in range(n):
                wp[j] += wp_tmp[j]
            
            _reorthogonalize_sparse(wp, Q_list, None, inner_multi, mpi, comm, n)
            
            M = inner_multi(wp, wp)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
            try:
                L = sp.cholesky(M, lower=True)
            except sp.LinAlgError:
                L = sp.cholesky(M + np.eye(n) * 1e-14, lower=True)
            beta_i = np.conj(L.T)
            
            betas.append(beta_i)
            
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    beta_inv = sp.inv(beta_i)
            except Exception:
                break
            
            q_next = [ManyBodyState() for _ in range(n)]
            add_scaled_multi(q_next, wp, beta_inv)
            for st in q_next:
                st.prune(slaterWeightMin)
                
            Q_list.extend([st.copy() for st in q_next])
            
            wp_next = h_op.apply_multi(q_next)
            if mpi:
                wp_next = basis.redistribute_psis(wp_next)
            alpha_i = inner_multi(q_next, wp_next)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, alpha_i, op=MPI.SUM)
            alphas.append(alpha_i)
            
            if track_W and W_full is not None:
                # Approximate tracking for expanded blocks
                W_full[i, i] = np.eye(n, dtype=complex)
                eps = np.finfo(float).eps
                W_full[i, :i] = eps * np.eye(n, dtype=complex)
                W_full[:i, i] = eps * np.eye(n, dtype=complex)
            
            q0 = q1
            q1 = q_next
            
        # Compute the final residual beta
        wp_final = [ManyBodyState() for _ in range(n)]
        wp_tmp = h_op.apply_multi(q1)
        if mpi:
            wp_tmp = basis.redistribute_psis(wp_tmp)
        for j in range(n):
            wp_final[j] += wp_tmp[j]
        _reorthogonalize_sparse(wp_final, Q_list, None, inner_multi, mpi, comm, n)
        
        M_final = inner_multi(wp_final, wp_final)
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, M_final, op=MPI.SUM)
        try:
            L = sp.cholesky(M_final, lower=True)
        except sp.LinAlgError:
            L = sp.cholesky(M_final + np.eye(n) * 1e-14, lower=True)
        betas.append(np.conj(L.T))
            
    T_final = _build_full_T(alphas, betas[:-1] if len(betas) == m_actual else betas)
    eigvals, eigvecs = sp.eigh(T_final)
    
    wanted_indices = np.argsort(eigvals)[:num_wanted]
    final_eigvals = eigvals[wanted_indices]
    
    final_eigvecs = [ManyBodyState() for _ in range(len(wanted_indices))]
    add_scaled_multi(final_eigvecs, Q_list, np.ascontiguousarray(eigvecs[:, wanted_indices], dtype=complex))
        
    return final_eigvals, final_eigvecs
