import itertools
from enum import Enum
from time import perf_counter
import numpy as np
import scipy as sp
from typing import Optional, NamedTuple, Callable
from mpi4py import MPI
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.krylovBasis import KrylovBasis
from impurityModel.ed.finite import applyOp_new as applyOp, inner, matmul, removeFromFirst


class Reort(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2


def calculate_thermal_gs(h, block_size, e_max, v0=None, reort=Reort.FULL, comm=None):
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


def eigsh(alphas, betas, de=None, Q=None, eigvals_only=False, select="a", select_range=None, max_ev=0, comm=None):
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
    converged: Callable[[np.ndarray, np.ndarray], bool],
    h_local: bool = False,
    verbose: bool = True,
    max_krylov_size: int = None,
    build_krylov_basis: bool = True,
    comm=None,
):
    if max_krylov_size is None:
        krylovSize = h.shape[0]
    else:
        krylovSize = min(h.shape[0], max_krylov_size)
    eps = np.finfo("float").eps
    t0 = perf_counter()

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
    if comm.rank == 0:
        if build_krylov_basis:
            Q = KrylovBasis(N, psi0.dtype, psi0)
        q = np.zeros((2, N, n), dtype=complex, order="C")
    else:
        q = np.empty((2, 0, 0))
    counts = comm.allgather(n * psi0.shape[0])
    offsets = [sum(counts[:r]) for r in range(len(counts))]
    comm.Gatherv(
        psi0,
        (
            q[1],
            counts,
            offsets,
            MPI.C_DOUBLE_COMPLEX,
        ),
        root=0,
    )
    if comm.rank == 0 and reort_mode != Reort.NONE:
        W = np.zeros((2, 1, n, n), dtype=complex)
        W[1] = np.identity(n)
        force_reort = None
    q_i = np.array(psi0, copy=True, order="C")

    if h_local:
        done = False
        wp = None
        if comm.rank == 0:
            wp = np.empty((h.shape[0], q.shape[2]), dtype=complex, order="C")
        # Run at least 1 iteration (to generate $\alpha_0$).
        # We can also not generate more than N Lanczos vectors, meaning we can
        # take at most N/n steps in total
        for i in range(int(np.ceil(krylovSize / n))):
            t_h = perf_counter()
            comm.Reduce(
                h @ q_i,
                wp,
                op=MPI.SUM,
                root=0,
            )
            t_matmul += perf_counter() - t_h

            if comm.rank == 0:
                alphas = np.append(alphas, [np.conj(q[1].T) @ wp], axis=0)
                betas = np.append(betas, [np.empty((n, n), dtype=complex)], axis=0)
                if i == 0:
                    wp -= q[1] @ alphas[i]
                else:
                    wp -= q[1] @ alphas[i] + q[0] @ np.conj(betas[i - 1].T)
                if reort_mode == Reort.FULL:
                    t_ortho = perf_counter()
                    n_reort += 1
                    wp -= Q.calc_projection(wp)
                    t_reorth += perf_counter() - t_ortho

                q[0] = q[1]
                t_qr_fact = perf_counter()
                q[1], betas[i] = sp.linalg.qr(wp, mode="economic", overwrite_a=True, check_finite=False)
                t_qr += perf_counter() - t_qr_fact
                b_mask = np.abs(np.diagonal(betas[i])) < np.finfo(float).eps
                t_converged = perf_counter()

                done = converged(alphas, betas)
                t_conv += perf_counter() - t_converged

            done = comm.bcast(done, root=0)
            if done:
                break

            if comm.rank == 0 and reort_mode == Reort.PARTIAL:
                t_overlap = perf_counter()
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
                t_estimate += perf_counter() - t_overlap

                reort = np.any(np.abs(W[1, : i + 1]) > np.sqrt(eps))
                if reort or force_reort is not None:
                    t_ortho = perf_counter()
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
                    t_reorth += perf_counter() - t_ortho
            if comm.rank == 0 and build_krylov_basis:
                Q.add(q[1])

            comm.Scatterv(
                (q[1], counts, offsets, MPI.C_DOUBLE_COMPLEX),
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

    print(f"Breaking after iteration {i}, blocksize = {n}")
    if verbose:
        print(f"===> Matrix vector multiplication took {t_matmul:.4f} seconds")
        print(f"===> Estimating overlap took {t_estimate:.4f} seconds")
        print(f"===> Estimating convergence took {t_conv:.4f} seconds")
        print(f"===> QR factorization took {t_qr:.4f} seconds")
        print(f"===> Reorthogonalized {n_reort} times")
        print(f"===> Reorthogonalizing took {t_reorth:.4f} seconds")
        print(f"=> time(get_block_Lanczons_matrices) = {perf_counter() - t0:.4f} seconds.")
    return alphas, betas, Q.vectors[: len(Q)].T if Q is not None else None


def block_lanczos(
    psi0: list[dict],
    h_op: dict,
    basis: Basis,
    converged: Callable[[np.ndarray, np.ndarray], bool],
    h_mem: dict = None,
    verbose: bool = True,
    build_krylov_basis: bool = True,
    slaterWeightMin: float = 0,
) -> (np.ndarray, np.ndarray, Optional[list[dict]]):
    if h_mem is None:
        h_mem = {}
    mpi = basis.comm is not None
    rank = basis.comm.rank if mpi else 0
    n = len(psi0)
    N0 = basis.size
    if mpi:
        psi_len = basis.comm.allreduce(sum(len(psi) for psi in psi0), op=MPI.SUM)
    else:
        psi_len = sum(len(psi) for psi in psi0)
    if psi_len == 0:
        return (
            np.zeros((1, n, n), dtype=complex),
            np.zeros((1, n, n), dtype=complex),
            psi0 if build_krylov_basis else None,
        )

    alphas = np.empty((0, n, n), dtype=complex)
    betas = np.empty((0, n, n), dtype=complex)
    q = [[{}] * n, psi0]
    if build_krylov_basis:
        Q = [psi for psi in psi0]
    t_add = 0
    t_vec = 0
    t_apply = 0
    t_redist = 0
    t_linalg = 0
    t_conv = 0
    t_qr = 0
    t_state = 0
    t_tot = perf_counter()

    it = 0
    done = False
    converge_count = 0
    wp = [None] * n
    while True:
        t_tmp = perf_counter()
        t_add += perf_counter() - t_tmp
        t_tmp = perf_counter()
        wp = [
            applyOp(
                basis.num_spin_orbitals,
                h_op,
                psi_i,
                slaterWeightMin=0,  # slaterWeightMin,
                restrictions=basis.restrictions,
                opResult=h_mem,
            )
            for psi_i in q[1]
        ]
        t_apply += perf_counter() - t_tmp
        t_tmp = perf_counter()
        if basis.size > 2 * N0:
            basis.clear()
            basis.add_states((state for psis in q for psi in psis for state in psi))
            N0 = basis.size
        basis.add_states(state for psi in wp for state in psi)
        t_add += perf_counter() - t_tmp
        t_tmp = perf_counter()
        tmp = basis.redistribute_psis(itertools.chain(q[0], q[1], wp))
        q[0] = tmp[0:n]
        q[1] = tmp[n : 2 * n]
        wp = tmp[2 * n : 3 * n]
        t_redist += perf_counter() - t_tmp
        t_tmp = perf_counter()
        psi = np.empty((len(basis.local_basis), n), dtype=complex)
        psip = np.empty_like(psi)
        psim = np.empty_like(psi)
        for (i, state), j in itertools.product(enumerate(basis.local_basis), range(n)):
            psi[i, j] = q[1][j].get(state, 0)
            psim[i, j] = q[0][j].get(state, 0)
            psip[i, j] = wp[j].get(state, 0)
        t_vec += perf_counter() - t_tmp

        t_tmp = perf_counter()
        alpha = np.conj(psi.T) @ psip
        alphas = np.append(alphas, np.empty((1, n, n), dtype=complex), axis=0)
        if mpi:
            request = basis.comm.Iallreduce(alpha, alphas[-1], op=MPI.SUM)
        else:
            alphas[-1] = alpha

        if it > 0:
            psip -= psim @ np.conj(betas[-1].T)
        betas = np.append(betas, np.empty((1, n, n), dtype=complex), axis=0)
        if mpi:
            send_counts = np.empty((basis.comm.size), dtype=int)
            basis.comm.Gather(np.array([n * len(basis.local_basis)]), send_counts)
            request.Wait()

        psip -= psi @ alphas[-1]
        t_linalg += perf_counter() - t_tmp
        t_tmp = perf_counter()
        if mpi:
            offsets = np.fromiter(
                (np.sum(send_counts[:i]) for i in range(basis.comm.size)), dtype=int, count=basis.comm.size
            )

        qip = np.empty((basis.size, n), dtype=complex) if rank == 0 else None
        if mpi:
            basis.comm.Gatherv(psip, [qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], root=0)
        else:
            qip[:, :] = psip
        t_vec += perf_counter() - t_tmp
        if rank == 0:
            t_tmp = perf_counter()
            qip, betas[-1] = sp.linalg.qr(qip, mode="economic", overwrite_a=True, check_finite=False)
            t_qr += perf_counter() - t_tmp
            qip = np.array(qip, order="C")
            _, columns = qip.shape

        if mpi:
            basis.comm.Bcast(betas[-1], root=0)
            columns = basis.comm.bcast(columns if rank == 0 else None)
            request = basis.comm.Iscatterv([qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], psip[:, :columns], root=0)

        if it % 5 == 0 or converge_count > 0:
            t_tmp = perf_counter()
            done = converged(alphas, betas)
            t_conv += perf_counter() - t_tmp

        if mpi:
            done = basis.comm.allreduce(done, op=MPI.LAND)

        converge_count = (1 + converge_count) if done else 0
        if converge_count > 0 or it * n > N0:
            break

        t_tmp = perf_counter()
        q[0] = q[1]
        q[1] = [{} for _ in range(columns)]
        # q[1] = [{} for _ in range(n)]
        if mpi:
            request.Wait()
        for j, (i, state) in itertools.product(range(columns), enumerate(basis.local_basis)):
            # for j, (i, state) in itertools.product(range(n), enumerate(basis.local_basis)):
            if abs(psip[i, j]) ** 2 > slaterWeightMin:
                q[1][j][state] = psip[i, j]
        t_state += perf_counter() - t_tmp
        if build_krylov_basis:
            Q.append(psi for psi in q[1])
        it += 1
    if verbose:
        print(f"Breaking after iteration {it}, blocksize = {n}")
        print(f"===> Applying the hamiltonian took {t_apply:.4f} seconds")
        print(f"===> Adding states took {t_add:.4f} seconds")
        print(f"===> Redistributing states took {t_redist:.4f} seconds")
        print(f"===> Local linear algebra took {t_linalg:.4f} seconds")
        print(f"===> Building vectors took {t_vec:.4f} seconds")
        print(f"===> Estimating convergence took {t_conv:.4f} seconds")
        print(f"===> QR factorization took {t_qr:.4f} seconds")
        print(f"===> Building states took {t_state:.4f} seconds")
        print(f"=> time(get_block_Lanczons_matrices) = {perf_counter() - t_tot:.4f} seconds.", flush=True)
    return alphas, betas, Q if build_krylov_basis else None
