import numpy as np
import scipy as sp

import time
from random import uniform

from mpi4py import MPI

from numba import njit

comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size

def get_block_Lanczos_matrices(
        psi0 : np.ndarray,
        h : sp.sparse.csr_array,
        converged : bool,
        h_local : bool = False,
        verbose : bool = True,
        partial_reort : bool = False,
        debug_ort: bool = False,
):
    krylovSize = h.shape[0]
    eps = np.finfo("float").eps
    t0 = time.perf_counter()

    t_reorth = 0
    t_estimate = 0
    t_matmul = 0
    t_conv = 0
    t_qr = 0

    N = psi0.shape[0]
    n = max(psi0.shape[1], 1)

    alphas = np.empty((0, n, n), dtype = complex)
    betas = np.empty((0, n, n), dtype = complex)

    if rank == 0:
        Q = np.empty((N, n), dtype = complex)
        Q[:, :] = psi0
        W = np.zeros((2, 1, n, n), dtype = complex)
        W[1] = np.identity(n)
        force_reort = None
        n_reort = 0
    q = np.zeros((2, N, n), dtype=complex)
    q[1, :, :] = psi0

    if rank == 0 and debug_ort:
        overlap_file = open('overlap.dat', 'a')
        overlap_file.write(f"{np.max(np.abs(W[1, : 1]))}  {np.max(np.abs( np.conj(Q.T) @ q[1]))}  {1}\n")

    if h_local:
        done = False
        # Run at least 1 iteration (to generate $\alpha_0$).
        # We can also not generate more than N Lanczos vectors, meaning we can take
        # at most N/n steps in total
        for i in range(int(np.ceil(krylovSize / n))):
            t_h = time.perf_counter()
            wp = h @ q[1]
            wp = comm.reduce(wp, root=0)
            t_matmul += time.perf_counter() - t_h

            if rank == 0:
                alphas = np.append(alphas, [np.conj(q[1].T) @ wp], axis=0)
                betas = np.append(betas, [np.zeros((n, n), dtype=complex)], axis=0)
                wp -= q[1] @ alphas[i] + q[0] @ np.conj(betas[i - 1].T)

                q[0] = q[1]
                t_qr_fact = time.perf_counter()
                q[1], betas[i] = sp.linalg.qr(wp, mode="economic", overwrite_a = True, check_finite = False)
                # v, r = sp.linalg.qr(wp, mode="full", overwrite_a = True, check_finite = False)
                # q[1], betas[i] = v[:,:n], r[:n, :]
                t_qr += time.perf_counter() - t_qr_fact
                # try:
                if i % int(np.ceil(krylovSize/n))//10 == 0:
                # if True:
                    t_converged = time.perf_counter()
                    # done = converged(alphas, betas)
                    delta = converged(alphas, betas)
                    done = delta < 1e-6
                    t_conv += time.perf_counter() - t_converged

            done = comm.bcast(done, root=0)
            if done:
                break

            if rank == 0 and partial_reort:
                t_overlap = time.perf_counter()
                # Clearly a function
                ####################
                w_bar = np.zeros((i + 2, n, n), dtype=complex)
                w_bar[i + 1, :, :] = np.identity(n)
                w_bar[i, :, :] = (
                    eps
                    * N
                    * sp.linalg.solve_triangular(np.conj(betas[i].T), betas[0], lower = True)
                    * np.random.normal(loc=0, scale=0.6, size=(n, n))
                )
                if i > 0:
                    w_bar[0, :, :] = (
                        W[1, 1] @ betas[0] + W[1, 0] @ alphas[0] - alphas[i] @ W[1, 0] - betas[i - 1] @ W[0, 0]
                    )
                    w_bar[0, :, :] = sp.linalg.solve_triangular(np.conj(betas[i].T), w_bar[0], lower = True)
                for j in range(1, i):
                    w_bar[j, :, :] = (
                        W[1, j + 1] @ betas[j]
                        + W[1, j] @ alphas[j]
                        - alphas[i] @ W[1, j]
                        + W[1, j - 1] @ np.conj(betas[j - 1].T)
                        - betas[i - 1] @ W[0, j]
                    )
                    w_bar[j, :, :] = sp.linalg.solve_triangular(np.conj(betas[i].T), w_bar[j], lower = True)

                w_bar[:i, :, :] += eps * (betas[i] + betas[:i]) * np.random.normal(loc=0, scale=0.3, size=(i, n, n))
                W_new = np.zeros((2, i + 2, n, n), dtype=complex)
                W_new[0, : i + 1] = W[1]
                W_new[1, : i + 2] = w_bar
                W = W_new
                ###
                t_estimate += time.perf_counter() - t_overlap

                if rank ==0 and debug_ort:
                    overlap_file.write(f"{np.max(np.abs(W[1, : i + 1]))}  {np.max(np.abs( np.conj(Q.T) @ q[1]))}  {delta}\n")
                reort = np.any(np.abs(W[1, : i + 1]) > np.sqrt(eps))
                if partial_reort and  (reort or force_reort is not None):
                    t_ortho = time.perf_counter()
                    n_reort += 1
                    # clearly a function
                    # mask = np.array([np.any(np.abs(m) > eps ** (3 / 4), axis=1) for m in W[1]])
                    # mask = np.array([[np.any(np.abs(m) > eps ** (3 / 4))]*n for m in W[1]])
                    # mask[-1:] = False
                    # mask = np.array([[True]*n for _ in W[1]])
                    # mask[-1, :] = False
                    # if force_reort is None:
                    #     Qm = Q[:, mask[:-1].flatten()]
                    # else:
                    #     force_reort = np.append(force_reort, [[False] * n], axis=0)
                    #     Qm = Q[:, np.logical_or(mask, force_reort)[:-1].flatten()]
                    q[1] = q[1] @ betas[i]
                    # q[1] -= Qm @ (np.conj(Qm.T) @ q[1])
                    q[1] -= Q @ (np.conj(Q.T) @ q[1])

                    q[1], betas[i] = sp.linalg.qr(q[1], mode="economic", overwrite_a = True, check_finite = False)
                    # v, r = sp.linalg.qr(q[1], mode="full", overwrite_a = True, check_finite = False)
                    # q[1], betas[i] = v[:,:n], r[:n, :]

                    # W[1, mask] = eps * np.random.normal(loc=0, scale=1.5, size=W[1, mask].shape)
                    # W[1, force_reort] = eps * np.random.normal(loc=0, scale=1.5, size=W[1, force_reort].shape)
                    W[1, : i + 1] = eps * np.random.normal(loc=0, scale=1.5, size=W[1, : i + 1].shape)

                    # force_reort = mask if reort else None
                    force_reort = True if reort else None
                    t_reorth += time.perf_counter() - t_ortho
                Q = np.append(Q, q[1], axis=1)

            q[1] = comm.bcast(q[1], root=0)

        # Distribute Lanczos matrices to all ranks
        alphas = comm.bcast(alphas, root=0)
        betas = comm.bcast(betas, root=0)
    else:
        # Run at least 1 iteration (to generate $\alpha_0$).
        # We can also not generate more than N Lanczos vectors, meaning we can take
        # at most N/n steps in total
        for i in range(max(krylovSize // n, 1)):
            # Update to PRO block Lanczos!!
            wp = h @ q[1]  # - q[0] @ np.conj(betas[i-1].T)
            # alphas[i] = np.conj(q[1].T) @ wp
            alphas = np.append(alphas, [np.conj(q[1].T) @ wp], axis=0)
            betas = np.append(betas, [np.zeros((n, n), dtype=complex)], axis=0)
            w = wp - q[1] @ alphas[i] - q[0] @ np.conj(betas[i - 1].T)
            q[0] = q[1]
            # q[1], betas[i] = np.linalg.qr(w)
            q[1], betas[i] = sp.linalg.qr(w, mode = 'economic', overwrite_a = True, check_finite = False)
            # q[1], betas[i] = my_qr(w)
            if converged(alphas, betas):
                break

    if rank == 0 and debug_ort:
        overlap_file.write("\n\n")
        overlap_file.close()
    if rank == 0 and verbose:
        print(f"Breaking after iteration {i}, blocksize = {n}")
        print(f"Matrix vector multiplication took {t_matmul:.4f} seconds")
        print(f"Estimating overlap took {t_estimate:.4f} seconds")
        print(f"Estimating convergence took {t_conv:.4f} seconds")
        print(f"QR factorization took {t_qr:.4f} seconds")
        print(f"Reorthogonalized {n_reort} times")
        print(f"Reorthogonalizing took {t_reorth:.4f} seconds")
        print(f"time(get_block_Lanczons_matrices) = {time.perf_counter() - t0:.4f} seconds.")
    return alphas, betas
