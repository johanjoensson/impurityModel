import numpy as np
import scipy.linalg as sp

from impurityModel.ed.block_math import block_apply, block_combine, block_normalize, block_orthogonalize, is_array
from impurityModel.ed.ManyBodyUtils import inner_multi


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
    from impurityModel.ed.lanczos import Reort, block_lanczos_sparse

    if reort is None:
        reort = Reort.PARTIAL

    is_arr = is_array(h_op)
    n = psi0.shape[1] if is_arr and len(psi0.shape) == 2 else len(psi0)
    k_blocks = int(np.ceil(num_wanted / n))
    m = max_subspace_blocks

    if m <= k_blocks:
        raise ValueError("max_subspace_blocks must be strictly greater than ceil(num_wanted / n)")

    mpi = basis is not None and getattr(basis, "comm", None) is not None
    comm = basis.comm if mpi else None

    # Initial Lanczos run to build subspace
    track_W = reort in (Reort.PARTIAL, Reort.SELECTIVE)

    if is_arr:
        from impurityModel.ed.lanczos import block_lanczos_array

        if isinstance(psi0, list):
            psi0_arr = np.concatenate(psi0, axis=1) if len(psi0[0].shape) == 2 else np.column_stack(psi0)
        else:
            psi0_arr = psi0
        alphas, betas, Q_list, *W_res = block_lanczos_array(
            psi0_arr,
            h_op,
            lambda a, b, **kw: False,
            max_iter=m,
            reort=reort,
            verbose=verbose,
            return_W=True,
            track_full_W=track_W,
            orth_tol=1e-12,
            return_Q=True,
        )
    else:
        from impurityModel.ed.lanczos import block_lanczos_sparse

        alphas, betas, Q_list, *W_res = block_lanczos_sparse(
            psi0,
            h_op,
            basis,
            lambda a, b, **kw: False,
            max_iter=m,
            slaterWeightMin=slaterWeightMin,
            reort=reort,
            verbose=verbose,
            inner_func=inner_multi,
            orth_tol=1e-12,
            return_Q=True,
            track_full_W=track_W,
            return_W=True,
        )

    W_full = W_res[0] if track_W and W_res else None

    for restart in range(max_restarts):
        m_actual = len(alphas)

        # Check if we hit invariant subspace before completing first loop
        if m_actual <= k_blocks:
            break

        from impurityModel.ed.lanczos import _build_full_T

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

        U_k = U_total[:, : k_blocks * n]
        from impurityModel.ed.lanczos import _extract_blocks

        alphas_new, betas_new = _extract_blocks(T_shifted, k_blocks, n)

        if is_arr:
            Q_list = Q_list[:, : m_actual * n]
        else:
            Q_list = Q_list[: m_actual * n]
        Q_new = block_combine(Q_list, U_k, slaterWeightMin)

        Q_list = Q_new
        alphas = alphas_new
        betas = betas_new

        if track_W and W_full is not None:
            W_full_2d = np.zeros((m_actual * n, m_actual * n), dtype=complex)
            for i_blk in range(m_actual):
                for j_blk in range(m_actual):
                    W_full_2d[i_blk * n : (i_blk + 1) * n, j_blk * n : (j_blk + 1) * n] = W_full[i_blk, j_blk]
            W_new_2d = U_k.conj().T @ W_full_2d @ U_k
            W_new = np.zeros((m, m, n, n), dtype=complex)
            for i_blk in range(k_blocks):
                for j_blk in range(k_blocks):
                    W_new[i_blk, j_blk] = W_new_2d[i_blk * n : (i_blk + 1) * n, j_blk * n : (j_blk + 1) * n]
            W_full = W_new

        alphas_pass = alphas_new

        # We must manually compute q_k so that block_lanczos can start at it = k_blocks
        # This preserves alpha_{k-1} exactly from the QR decomposition.
        if is_arr:
            q1 = Q_new[:, -n:]
        else:
            q1 = Q_new[-n:]

        wp = block_apply(h_op, q1, basis, mpi)
        wp, _ = block_orthogonalize(wp, Q_new, mpi=mpi, comm=comm)
        wp, _ = block_orthogonalize(wp, Q_new, mpi=mpi, comm=comm)

        q_k, beta_i = block_normalize(wp, mpi, comm, slaterWeightMin)

        if np.linalg.norm(beta_i) < 1e-5:
            if verbose and (not mpi or comm.rank == 0):
                print(f"Invariant subspace found during IRLM restart. Stopping early.")
            
            from impurityModel.ed.lanczos import _build_full_T
            T_final = _build_full_T(alphas_new, betas_new)
            eigvals, eigvecs = sp.eigh(T_final)
            
            wanted_indices = np.argsort(eigvals)[:num_wanted]
            final_eigvals = eigvals[wanted_indices]
            
            final_eigvecs = block_combine(Q_new, eigvecs[:, wanted_indices], slaterWeightMin)
            return final_eigvals, final_eigvecs

        if is_arr:
            Q_list = np.concatenate([Q_new, q_k], axis=1)
        else:
            Q_list = Q_new + q_k

        if is_arr:
            betas_pass = np.concatenate([betas_new, [beta_i]], axis=0) if len(betas_new) > 0 else np.array([beta_i])
        else:
            betas_pass = list(betas_new) + [beta_i]

        if track_W and W_full is not None:
            W_pass = np.zeros((k_blocks + 1, k_blocks + 1, n, n), dtype=complex)
            W_pass[:k_blocks, :k_blocks] = W_full[:k_blocks, :k_blocks]
            W_pass[k_blocks, k_blocks] = np.eye(n)
        else:
            W_pass = None

        if is_arr:
            from impurityModel.ed.lanczos import block_lanczos_array

            alphas, betas, Q_list, *W_res = block_lanczos_array(
                None,
                h_op,
                lambda a, b, **kw: False,
                max_iter=m,
                reort=reort,
                verbose=verbose,
                return_W=True,
                track_full_W=track_W,
                alphas=alphas_pass,
                betas=betas_pass,
                Q=Q_list,
                W=W_pass,
                orth_tol=1e-12,
                return_Q=True,
            )
        else:
            from impurityModel.ed.lanczos import block_lanczos_sparse

            alphas, betas, Q_list, *W_res = block_lanczos_sparse(
                None,
                h_op,
                basis,
                lambda a, b, **kw: False,
                max_iter=m,
                slaterWeightMin=slaterWeightMin,
                reort=reort,
                verbose=verbose,
                inner_func=inner_multi,
                alphas=alphas_pass,
                betas=betas_pass,
                Q=Q_list,
                W=W_pass,
                orth_tol=1e-12,
                return_Q=True,
                track_full_W=track_W,
                return_W=True,
            )
        W_full = W_res[0] if track_W and W_res else None

    # Final Ritz vectors
    m_actual = len(alphas)
    from impurityModel.ed.lanczos import _build_full_T

    T_final = _build_full_T(alphas, betas[:-1] if len(betas) == m_actual else betas)
    eigvals, eigvecs = sp.eigh(T_final)

    wanted_indices = np.argsort(eigvals)[:num_wanted]
    final_eigvals = eigvals[wanted_indices]

    if is_arr:
        Q_list = Q_list[:, : m_actual * n]
    else:
        Q_list = Q_list[: m_actual * n]

    final_eigvecs = block_combine(Q_list, eigvecs[:, wanted_indices], slaterWeightMin)

    return final_eigvals, final_eigvecs
