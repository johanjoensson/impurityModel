import numpy as np
import scipy as sp
import time
from mpi4py import MPI
from impurityModel.ed.krylovBasis import KrylovBasis
from enum import Enum

comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


class Reort(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2


def calculate_thermal_gs(h, block_size, e_max, v0=None, reort=Reort.FULL):
    def converged(alphas, betas):
        e, s = eigsh(alphas, betas, de=e_max, select="m")
        sorted_indices = np.argsort(e)
        e = e[sorted_indices]
        s[:, sorted_indices]
        mask = e - np.min(e) <= e_max
        return np.linalg.norm(betas[-1] @ s[-block_size:, mask], ord=2)

    if v0 is None:
        v0 = np.random.rand(h.shape[0], block_size) + 1j * np.random.rand(h.shape[0], block_size)
        # v0 = np.ones((h.shape[0], block_size), dtype = complex)
        v0, _ = sp.linalg.qr(v0, mode="economic", overwrite_a=True, check_finite=False)
        # v0, _ = np.linalg.qr(v0, mode="reduced")
    elif v0.shape[1] < block_size:
        new_v0 = np.random.rand(v0.shape[0], block_size - v0.shape[1]) + 1j * np.random.rand(
            v0.shape[0], block_size - v0.shape[1]
        )
        new_v0 -= v0 @ np.conj(v0.T) @ new_v0
        for col in range(new_v0.shape[1]):
            new_v0[:, col] = new_v0[:, col] / np.linalg.norm(new_v0[:, col])
        v0 = np.append(v0, new_v0, axis=1)
    elif v0.shape[1] > block_size:
        v0 = v0[:, :block_size]
    alphas, betas, Q = get_block_Lanczos_matrices(
        psi0=v0,
        h=h,
        converged=converged,
        h_local=True,
        verbose=True,
        reort_mode=reort,
    )
    if comm.rank == 0:
        eigvals, eigvecs = eigsh(alphas, betas, de=e_max, Q=Q[:, : alphas.shape[0] * alphas.shape[1]], select="m")
    else:
        eigvals = None
        eigvecs = None
    eigvals = comm.bcast(eigvals, root=0)
    eigvecs = comm.bcast(eigvecs, root=0)
    return eigvals, eigvecs


def build_banded_matrix(alphas, betas):
    k = alphas.shape[0]
    p = alphas.shape[1]
    bands = np.zeros((p + 1, k * p), dtype=alphas.dtype)
    bands[0, :] = np.diagonal(alphas, offset=0, axis1=1, axis2=2).flatten()
    for i in range(1, p + 1):
        for j in range(k):
            bands[i, j * p : (j + 1) * p] = np.append(
                np.diagonal(alphas[j], offset=-i),
                [np.diagonal(betas[j], offset=p - i)],
            ).flatten()
    return bands


def eigsh(alphas, betas, de=None, Q=None, eigvals_only=False, select="a", select_range=None, max_ev=0):
    """
    Solve the (block) lanczos eigenvalue problem. Return the eigenvalues and
    (optionally) the corresponding eigenvectors. Return only eigenvalues (and,
    optionally eigenvectors) lying within de of the lowest eigenvalue.
    NOTE: len(alphas) == len(betas) == n, however the last element in betas
    will not be accessed.
         [alpha_0  beta_0* 0        ...      ]
    Tm = [beta_0   alpha_1 beta_1*  ...      ]
         [0        beta_1  ...      beta_n-2*]
         [...              beta_n-2 alpha_n-1  ]
    If eigvals_only is True, only return the eigenvalues.
    If Q is None, return the eigenvectors in the Krylov basis. Otherwise, the
    eigenvectors are transformed using Q, as such: eigvecs = Q @ eigvecs, and
    then returned.
    Parameters
    ==========
    alphas - (block) diagonal terms of the Matrix in the Krylov basis. [alpha_0
             ,alpha_1, ..., alpha_n]
    betas - (block) off-diagnal terms of the Matrix in the Krylov basis.[beta_0
            ,beta_1, ..., beta_n]
    """
    whithin_gs = False
    if select == "m":
        assert de is not None
        select = "a"
        whithin_gs = True

    Tm = build_banded_matrix(alphas, betas)
    if eigvals_only:
        eigvals = sp.linalg.eig_banded(
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
        return eigvals[eigvals - eigvals[0] <= de]
    eigvals, eigvecs = sp.linalg.eig_banded(
        Tm,
        lower=True,
        eigvals_only=False,
        overwrite_a_band=True,
        check_finite=False,
        select=select,
        select_range=select_range,
        max_ev=max_ev,
    )
    if whithin_gs:
        mask = eigvals - np.min(eigvals) <= de
    else:
        mask = [True] * len(eigvals)
    if Q is not None:
        eigvecs = Q @ eigvecs
    sort_indices = np.argsort(eigvals[mask])
    return eigvals[mask][sort_indices], eigvecs[:, mask][:, sort_indices]


def get_block_Lanczos_matrices(
    psi0: np.ndarray,
    h,
    reort_mode,
    converged,
    h_local: bool = False,
    verbose: bool = True,
    max_krylov_size: int = None,
    build_krylov_basis: bool = True,
):
    if max_krylov_size is None:
        krylovSize = h.shape[0]
    else:
        krylovSize = min(h.shape[0], max_krylov_size)
    eps = np.finfo("float").eps
    t0 = time.perf_counter()

    t_reorth = 0.0
    t_estimate = 0.0
    t_matmul = 0.0
    t_conv = 0.0
    t_qr = 0.0

    N = h.shape[0]
    n = psi0.shape[1] if len(psi0.shape) == 2 else 1

    alphas = np.empty((0, n, n), dtype=complex)
    betas = np.empty((0, n, n), dtype=complex)
    Q = None
    build_krylov_basis = build_krylov_basis or reort_mode != Reort.NONE
    n_reort = 0
    if rank == 0:
        if build_krylov_basis:
            Q = KrylovBasis(N, psi0.dtype, psi0)
        q = np.zeros((2, N, n), dtype=complex)
    else:
        q = np.empty((2, 0, 0))
        # q[1, :, :] = psi0
    counts = comm.allgather(n * psi0.shape[0])
    offsets = [sum(counts[:r]) for r in range(len(counts))]
    comm.Gatherv(
        psi0,
        (
            q[1, :, :],
            counts,
            offsets,
            MPI.DOUBLE_COMPLEX,
        ),
        root=0,
    )
    if rank == 0 and reort_mode != Reort.NONE:
        W = np.zeros((2, 1, n, n), dtype=complex)
        W[1] = np.identity(n)
        force_reort = None
    q_i = psi0

    if h_local:
        done = False
        wp = None
        if rank == 0:
            wp = np.empty((h.shape[0], q.shape[2]), dtype=complex)
        # Run at least 1 iteration (to generate $\alpha_0$).
        # We can also not generate more than N Lanczos vectors, meaning we can
        # take at most N/n steps in total
        for i in range(int(np.ceil(krylovSize / n))):
            t_h = time.perf_counter()
            comm.Reduce(
                h @ q_i,
                # h[:, offsets[comm.rank] : offsets[comm.rank] + psi0.shape[0]] @ q_i,
                wp,
                op=MPI.SUM,
                root=0,
            )
            t_matmul += time.perf_counter() - t_h

            if rank == 0:
                alphas = np.append(alphas, [np.conj(q[1].T) @ wp], axis=0)
                betas = np.append(betas, [np.empty((n, n), dtype=complex)], axis=0)
                if i == 0:
                    wp -= q[1] @ alphas[i]
                else:
                    wp -= q[1] @ alphas[i] + q[0] @ np.conj(betas[i - 1].T)
                if reort_mode == Reort.FULL:
                    t_ortho = time.perf_counter()
                    n_reort += 1
                    wp -= Q.calc_projection(wp)
                    t_reorth += time.perf_counter() - t_ortho

                q[0] = q[1]
                t_qr_fact = time.perf_counter()
                q[1], betas[i] = sp.linalg.qr(wp, mode="economic", overwrite_a=True, check_finite=False)
                t_qr += time.perf_counter() - t_qr_fact
                b_mask = np.abs(np.diagonal(betas[i])) < np.finfo(float).eps
                t_converged = time.perf_counter()
                delta = converged(alphas, betas)

                done = delta < 1e-12
                t_conv += time.perf_counter() - t_converged

            done = comm.bcast(done, root=0)
            if done:
                break

            if rank == 0 and reort_mode == Reort.PARTIAL:
                t_overlap = time.perf_counter()
                # Clearly a function
                ####################
                w_bar = np.zeros((i + 2, n, n), dtype=complex)
                w_bar[i + 1, :, :] = np.identity(n)
                w_bar[i, :, :] = (
                    eps
                    * N
                    * sp.linalg.solve_triangular(np.conj(betas[i].T), betas[0], lower=True)
                    * np.random.normal(loc=0, scale=0.6, size=(n, n))
                )
                if i > 0:
                    w_bar[0, :, :] = (
                        W[1, 1] @ betas[0] + W[1, 0] @ alphas[0] - alphas[i] @ W[1, 0] - betas[i - 1] @ W[0, 0]
                    )
                    w_bar[0, :, :] = sp.linalg.solve_triangular(np.conj(betas[i].T), w_bar[0], lower=True)
                for j in range(1, i):
                    w_bar[j, :, :] = (
                        W[1, j + 1] @ betas[j]
                        + W[1, j] @ alphas[j]
                        - alphas[i] @ W[1, j]
                        + W[1, j - 1] @ np.conj(betas[j - 1].T)
                        - betas[i - 1] @ W[0, j]
                    )
                    w_bar[j, :, :] = sp.linalg.solve_triangular(np.conj(betas[i].T), w_bar[j], lower=True)

                w_bar[:i, :, :] += eps * (betas[i] + betas[:i]) * np.random.normal(loc=0, scale=0.3, size=(i, n, n))
                W_new = np.zeros((2, i + 2, n, n), dtype=complex)
                W_new[0, : i + 1] = W[1]
                W_new[1, : i + 2] = w_bar
                W = W_new
                ###
                t_estimate += time.perf_counter() - t_overlap

                reort = np.any(np.abs(W[1, : i + 1]) > np.sqrt(eps))
                if reort or force_reort is not None:
                    t_ortho = time.perf_counter()
                    n_reort += 1
                    q[1] = q[1] @ betas[i]

                    q[1], betas[i] = sp.linalg.qr(
                        q[1] - Q.calc_projection(q[1]), mode="economic", overwrite_a=True, check_finite=False
                    )
                    # clearly a function
                    mask = np.array([np.any(np.abs(m) > eps ** (3 / 4), axis=1) for m in W[1]])
                    mask[-1:] = False
                    b_mask = np.abs(np.diagonal(betas[i])) < np.finfo(float).eps
                    while np.any(b_mask):
                        q[1] = q[1] @ betas[i]
                        q[1][:, b_mask] = np.random.rand(N, sum(b_mask)) + 1j * np.random.rand(N, sum(b_mask))
                        q[1], betas[i] = sp.linalg.qr(
                            q[1] - Q.calc_projection(q[1]), mode="economic", overwrite_a=True, check_finite=False
                        )
                        b_mask = np.abs(np.diagonal(betas[i])) < np.finfo(float).eps
                        W[1, -1] = 1

                    W[1, mask] = eps * np.random.normal(loc=0, scale=1.5, size=W[1, mask].shape)

                    force_reort = mask if reort else None
                    t_reorth += time.perf_counter() - t_ortho
            if rank == 0 and build_krylov_basis:
                Q.add(q[1])

            comm.Scatterv(
                (q[1], counts, offsets, MPI.DOUBLE_COMPLEX),
                q_i,
                root=0,
            )

        # Distribute Lanczos matrices to all ranks
        alphas = comm.bcast(alphas, root=0)
        betas = comm.bcast(betas, root=0)
    else:
        # Run at least 1 iteration (to generate $\alpha_0$).
        # We can also not generate more than N Lanczos vectors, meaning we can take
        # at most N/n steps in total
        for i in range(max(krylovSize // n, 1)):
            # Update to PRO block Lanczos!!
            wp = h @ q[1]
            alphas = np.append(alphas, [np.conj(q[1].T) @ wp], axis=0)
            betas = np.append(betas, [np.zeros((n, n), dtype=complex)], axis=0)
            w = wp - q[1] @ alphas[i] - q[0] @ np.conj(betas[i - 1].T)
            q[0] = q[1]
            q[1], betas[i] = sp.linalg.qr(w, mode="economic", overwrite_a=True, check_finite=False)
            delta = converged(alphas, betas)
            if delta < 1e-6:
                break

    if verbose:
        print(f"Breaking after iteration {i}, blocksize = {n}")
        print(f"Matrix vector multiplication took {t_matmul:.4f} seconds")
        print(f"Estimating overlap took {t_estimate:.4f} seconds")
        print(f"Estimating convergence took {t_conv:.4f} seconds")
        print(f"QR factorization took {t_qr:.4f} seconds")
        print(f"Reorthogonalized {n_reort} times")
        print(f"Reorthogonalizing took {t_reorth:.4f} seconds")
        print(f"time(get_block_Lanczons_matrices) = {time.perf_counter() - t0:.4f} seconds.")
    return alphas, betas, Q.vectors[: len(Q)].T if Q is not None else None
