import numpy as np
import scipy as sp

import time
from random import uniform

from mpi4py import MPI

from impurityModel.ed.lanczos_cython import mydot
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
        partial_reort : bool = False
):
    krylovSize = h.shape[0]
    eps = np.finfo("float").eps
    if verbose:
        t0 = time.perf_counter()

    t_reorth = []
    t_estimate = 0

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

    if h_local:
        done = False
        # Run at least 1 iteration (to generate $\alpha_0$).
        # We can also not generate more than N Lanczos vectors, meaning we can take
        # at most N/n steps in total
        for i in range(int(np.ceil(krylovSize / n))):
            wp = h @ q[1]
            wp = comm.reduce(wp, root=0)

            if rank == 0:
                alphas = np.append(alphas, [np.conj(q[1].T) @ wp], axis=0)
                betas = np.append(betas, [np.zeros((n, n), dtype=complex)], axis=0)
                wp -= q[1] @ alphas[i] + q[0] @ np.conj(betas[i - 1].T)

                t_s = time.perf_counter()
                q[0] = q[1]
                q[1], betas[i] = sp.linalg.qr(wp, mode="economic", overwrite_a = True)
                # try:
                if i % (100//n) == 0:
                    done = converged(alphas, betas)
                # except FloatingPointError:
                #     alphas = alphas[:-1]
                #     betas = betas[:-1]
                #     done = True

            done = comm.bcast(done, root=0)
            if done:
                break

            if rank == 0 and partial_reort:
                # Clearly a function
                ####################
                w_bar = np.zeros((i + 2, n, n), dtype=complex)
                w_bar[i + 1, :, :] = np.identity(n)
                try:
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
                except FloatingPointError:
                    w_bar[:i+1] = 1
                W_new = np.zeros((2, i + 2, n, n), dtype=complex)
                W_new[0, : i + 1] = W[1]
                W_new[1, : i + 2] = w_bar
                W = W_new
                ###
                t_estimate += time.perf_counter() - t_s

                # reort = np.any(np.abs(W[1, : i + 1]) > 0.01)
                reort = np.any(np.abs(W[1, : i + 1]) > np.sqrt(eps))
                if reort or force_reort is not None:
                    n_reort += 1
                    # clearly a function
                    mask = np.array([np.any(np.abs(m) > eps ** (3 / 4), axis=0) for m in W[1]])
                    mask[-1:] = False
                    if force_reort is None:
                        Qm = Q[:, mask[:-1].flatten()]
                    else:
                        force_reort = np.append(force_reort, [[False] * n], axis=0)
                        Qm = Q[:, np.logical_or(mask, force_reort)[:-1].flatten()]
                    q[1] = q[1] @ betas[i]
                    t_s = time.perf_counter()
                    q[1] -= Qm @ (np.conj(Qm.T) @ q[1])
                    t_reorth.append(time.perf_counter() - t_s)
                    ###

                    q[1], betas[i] = sp.linalg.qr(q[1], mode="economic", overwrite_a = False)

                    W[1, mask] = eps * np.random.normal(loc=0, scale=1.5, size=W[1, mask].shape)
                    W[1, force_reort] = eps * np.random.normal(loc=0, scale=1.5, size=W[1, force_reort].shape)

                    force_reort = mask if reort else None
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
            # w = wp - mydot(q[1], alphas[i]) - mydot_triangular(np.conj(betas[i - 1]), q[0].T).T
            q[0] = q[1]
            # q[1], betas[i] = np.linalg.qr(w)
            q[1], betas[i] = sp.linalg.qr(w, mode = 'economic', overwrite_a = True, check_finite = False)
            # q[1], betas[i] = my_qr(w)
            if converged(alphas, betas):
                break

    if rank == 0 and verbose:
        print(f"Breaking after iteration {i}, blocksize = {n}")
        print(f"Estimating overlap took {t_estimate} seconds")
        if len(t_reorth) > 0:
            print(f"Reorthogonalized {n_reort} times")
            print(f"Reorthogonalizing took {sum(t_reorth)} seconds in total")
        print(f"time(get_block_Lanczons_matrices) = {time.perf_counter() - t0} seconds.")
    return alphas, betas
