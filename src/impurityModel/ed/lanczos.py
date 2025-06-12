from bisect import bisect_left
import itertools
from enum import Enum
from heapq import merge
from time import perf_counter
import numpy as np
import scipy as sp
from typing import Optional, NamedTuple, Callable
from mpi4py import MPI
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.krylovBasis import KrylovBasis
from impurityModel.ed.finite import (
    applyOp_new as applyOp,
    inner,
    matmul,
    removeFromFirst,
    addOps,
    subtractOps,
    scale,
    norm2,
)
from cmath import phase, rect
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, applyOp as applyOp_test


class Reort(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2
    PERIODIC = 3


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


def estimate_orthonormality(W, alphas, betas, eps=np.finfo(float).eps, N=1, rng=np.random.default_rng()):
    """Estimate the overlap between obtained Lanczos vectors at a ceratin iteration.
    W, alphas and betas contain all the required information for estimating the overlap.
    The stats dictionary contains the following keys:
    * w_bar  - The absolute values of the estimated overlaps for the second row
               of W.
    Parameters:
    W      - Array containing the two latest etimates of overlap. Dimensions (2, i+1, n, n)
    alphas - Array containing the (block) diagonal elements obtained from the
             (block) Lanczos method. Dimensions (i+1, n, n)
    betas  - Array containing the (block) off diagonal elements obtained from the
             (block) Lanczos method. Dimensions (i+1, n, n)
    eps    - Precision of orthogonality. Default: machine precision
    A_norm - Estimate of the norm of the matrix A. Default: 1
    Returns:
    W_out  - Estimated overlaps of the last two vectors obtained from the (block)
             Lanczos method. Dimensions (2, i+1, n, n)
    """
    # i is the index of the latest calculated vector
    i = alphas.shape[0] - 1
    n = alphas.shape[1]
    W_out = np.zeros((2, i + 2, n, n), dtype=complex)
    w_bar = np.zeros((i + 2, n, n), dtype=complex)
    w_bar[i + 1, :, :] = np.identity(n)
    w_bar[i, :, :] = (
        eps * N * sp.linalg.solve_triangular(betas[i], betas[0], lower=False, trans="C", check_finite=False)
    )
    if i == 0:
        W_out[0, : i + 1] = W[1]
        W_out[1, : i + 2] = w_bar
        return W_out

    w_bar[0] = W[1, 1] @ betas[0] + W[1, 0] @ alphas[0] - alphas[i] @ W[1, 0] - betas[i - 1] @ W[0, 0]
    w_bar[0] = sp.linalg.solve_triangular(betas[i], w_bar[0], lower=False, trans="C", check_finite=False)
    w_bar[1:i] = (
        W[1, 2 : i + 1] @ betas[1:i]
        + W[1, 1:i] @ alphas[1:i]
        - alphas[i][np.newaxis, :, :] @ W[1, 1:i]
        + W[1, : i - 1] @ np.conj(np.transpose(betas[: i - 1], axes=[0, 2, 1]))
        - betas[i - 1][np.newaxis, :, :] @ W[0, 1:i]
    )
    w_bar[1:i] = np.linalg.solve(betas[i][np.newaxis, :, :], w_bar[1:i])

    w_bar[:i] += eps * (betas[i] + betas[:i])
    W_out[0, : i + 1] = W[1]
    W_out[1, : i + 2] = w_bar

    return W_out


def qr_decomp_new(basis, psi):
    print(f"{len(psi)=}")

    if basis.size <= len(psi):
        v = basis.build_vector(psi, root=0)
        if basis.comm.rank == 0:
            q, r, _ = qr_decomp(v.T)
        q = basis.comm.bcast(q if basis.comm.rank == 0 else None, root=0)
        r = basis.comm.bcast(r if basis.comm.rank == 0 else None, root=0)
        return basis.build_state(q.T), r

    v = basis.build_vector(psi, root=0)
    if basis.comm.rank == 0:
        q, r, _ = qr_decomp(v.T)
    q_check = basis.comm.bcast(q if basis.comm.rank == 0 else None, root=0)
    r_check = basis.comm.bcast(r if basis.comm.rank == 0 else None, root=0)
    q_check = basis.build_state(q_check.T)

    def reflect(v, x):
        p = inner(v, x)
        p = basis.comm.allreduce(p, op=MPI.SUM)
        return subtractOps(x, scale(v, 2 * p))

    k_max = min(len(psi), basis.size - 1)
    selected_states = list(basis[range(k_max)])
    A = psi.copy()
    vs = [None for _ in range(k_max)]
    for i, state in enumerate(selected_states):
        Ap = [{s: amp for s, amp in A[j].items() if s not in selected_states[:i]} for j in range(i, len(A))]
        x = Ap[0]
        x_norm = np.sqrt(basis.comm.allreduce(norm2(x)))
        if state in basis.local_basis:
            xi = x[state]
            alpha = -rect(x_norm, phase(xi))
        else:
            alpha = 0
        alpha = basis.comm.allreduce(alpha)
        u = subtractOps(x, scale({state: 1} if state in basis.local_basis else {}, alpha))
        u_norm = np.sqrt(basis.comm.allreduce(norm2(u)))
        v = scale(u, 1 / u_norm)
        # Q = lambda x: reflect(v, x)
        vs[i] = v
        # for j, y in enumerate(tmp):
        for j, y in enumerate(Ap):
            Ap[j] = reflect(v, y)
        for j in range(i, len(A)):
            for s, amp in Ap[j - i].items():
                A[j][s] = amp
    Q = [{} for _ in psi]
    for (i, state_i), (j, state_j) in itertools.product(enumerate(basis.local_basis), enumerate(selected_states)):
        x = {state_j: 1}
        for v in vs:
            p = np.conj(v.get(state_j, 0))
            x = subtractOps(x, scale(v, 2 * p))
        if state_i in x:
            Q[j][state_i] = x[state_i] + Q[j].get(state_i, 0)

    R = basis.build_vector(A)
    print(f"Q = {[{state: amp for state, amp in q.items() if abs(amp) > np.finfo(float).eps} for q in Q]}")
    print(f"Q_check = {[{state: amp for state, amp in q.items() if abs(amp) > np.finfo(float).eps} for q in q_check]}")
    print(f"R = {R[:k_max, :k_max]}")
    print(f"R_check = {r_check}")
    for (i, qi), (j, qj) in itertools.product(enumerate(Q), repeat=2):
        if i == j:
            assert abs(1 - basis.comm.allreduce(inner(qi, qj))) < 1e-12, f"{i=} {j=}: {inner(qi,qj)=}"
        else:
            assert abs(-basis.comm.allreduce(inner(qi, qj))) < 1e-12, f"{i=} {j=}: {inner(qi, qj)=}"
    return [{state: amp for state, amp in q.items() if abs(amp) > np.finfo(float).eps} for q in Q], R[:k_max, :k_max]


def qr_decomp(psi, pivoting=False):
    if pivoting:
        psi, beta, p = sp.linalg.qr(psi, mode="economic", overwrite_a=True, check_finite=False, pivoting=True)
        return np.ascontiguousarray(psi), beta, p
    psi, beta = sp.linalg.qr(psi, mode="economic", overwrite_a=True, check_finite=False, pivoting=False)
    return np.ascontiguousarray(psi), beta, None


def get_block_Lanczos_matrices(
    psi0: np.ndarray,
    h,
    reort_mode,
    converged: Callable[[np.ndarray, np.ndarray], bool],
    verbose: bool = True,
    max_krylov_size: int = None,
    build_krylov_basis: bool = False,
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
    # counts = comm.allgather(n * psi0.shape[0])
    # offsets = [sum(counts[:r]) for r in range(len(counts))]
    if comm.rank == 0:
        q[1] = psi0
    # comm.Gatherv(
    #     psi0,
    #     (
    #         q[1],
    #         counts,
    #         offsets,
    #         MPI.C_DOUBLE_COMPLEX,
    #     ),
    #     root=0,
    # )
    if comm.rank == 0 and reort_mode != Reort.NONE:
        W = np.zeros((2, 1, n, n), dtype=complex)
        W[1] = np.identity(n)
        force_reort = None
    q_i = np.array(psi0, copy=True, order="C")

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
            betas = np.append(betas, np.zeros((1, n, n), dtype=complex), axis=0)
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
            t_converged = perf_counter()

            done = converged(alphas, betas)
            t_conv += perf_counter() - t_converged

        done = comm.bcast(done, root=0)
        if done:
            break

        if comm.rank == 0:
            if reort_mode == Reort.PARTIAL:
                t_overlap = perf_counter()
                W = estimate_orthonormality(W, alphas, betas, N=N)
                t_estimate += perf_counter() - t_overlap

                reort = np.any(np.abs(W[1, : i + 1]) > np.sqrt(eps))
                if reort or force_reort is not None:
                    t_ortho = perf_counter()
                    n_reort += 1
                    mask = np.array([np.any(np.abs(m) > eps ** (3 / 4), axis=1) for m in W[1]])
                    mask[-1:] = False
                    combined_mask = mask
                    combined_mask[:-1] = np.logical_or(mask[:-1], force_reort) if force_reort is not None else mask[:-1]

                    # Qm = Q[combined_mask.flatten()]
                    wp = q[1] @ betas[i]
                    # wp -= Qm @ np.conj(Qm.T) @ wp
                    wp -= Q.calc_projection(wp)
                    q[1], betas[i] = sp.linalg.qr(wp, mode="economic", overwrite_a=True, check_finite=False)

                    W[1, :-1] = eps * np.random.normal(loc=0, scale=1.5, size=W[1, :-1].shape)
                    # W[1, combined_mask.flatten()] = eps * np.random.normal(loc=0, scale=1.5, size=W[1, mask].shape)

                    force_reort = mask if reort else None
                    t_reorth += perf_counter() - t_ortho
        if comm.rank == 0 and build_krylov_basis:
            Q.add(q[1])

        q_i = comm.bcast(q[1], root=0)
        # comm.Scatterv(
        #     (q[1], counts, offsets, MPI.C_DOUBLE_COMPLEX),
        #     q_i,
        #     root=0,
        # )

    # Distribute Lanczos matrices to all ranks
    alphas = comm.bcast(alphas, root=0)
    betas = comm.bcast(betas, root=0)

    if verbose:
        print()
        print(f"Breaking after iteration {i+1}, blocksize = {n}")
        print(f"===> Matrix vector multiplication took {t_matmul:.4f} seconds")
        print(f"===> Estimating overlap took {t_estimate:.4f} seconds")
        print(f"===> Estimating convergence took {t_conv:.4f} seconds")
        print(f"===> QR factorization took {t_qr:.4f} seconds")
        print(f"===> Reorthogonalized {n_reort} times")
        print(f"===> Reorthogonalizing took {t_reorth:.4f} seconds")
        print(f"=> time(get_block_Lanczons_matrices) = {perf_counter() - t0:.4f} seconds.")
        print("=" * 80)
    return alphas, betas, Q.vectors[: len(Q)].T if Q is not None else None


def block_lanczos_sparse(
    psi0: list[dict],
    h_op: dict,
    basis: Basis,
    converged: Callable[[np.ndarray, np.ndarray], bool],
    h_mem: Optional[dict] = None,
    verbose: bool = False,
    reort: Reort = Reort.NONE,
    slaterWeightMin: float = 0,
) -> (np.ndarray, np.ndarray, Optional[list[dict]]):
    if h_mem is None:
        h_mem = {}
    mpi = basis.comm is not None
    comm = basis.comm if mpi else MPI.COMM_SELF
    build_krylov_basis = reort != Reort.NONE
    n = len(psi0)
    N_max = basis.size
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
        Q = list(psi0)

    eps = max(slaterWeightMin, np.finfo(float).eps)

    it = 0
    done = False
    converge_count = 0
    wp = [None] * n
    while (it + 1) * n < basis.size or it == 0:
        wp = [
            applyOp(
                basis.num_spin_orbitals,
                h_op,
                psi_i,
                slaterWeightMin=slaterWeightMin,
                restrictions=basis.restrictions,
                opResult=h_mem,
            )
            for psi_i in q[1]
        ]
        # wp = basis.redistribute_psis(wp)

        wp_size = np.array([len(psi) for psi in wp], dtype=int)
        comm.Allreduce(MPI.IN_PLACE, wp_size, op=MPI.SUM)
        cutoff = eps
        n_trunc = 0
        while np.max(wp_size) > basis.truncation_threshold:
            wp = [{state: w for state, w in psi.items() if abs(w) >= cutoff} for psi in wp]
            comm.Allreduce(np.array([len(psi) for psi in wp]), wp_size, op=MPI.SUM)
            cutoff *= 5
            n_trunc += 1

        if n_trunc > 0:
            basis.clear()

        basis.add_states(
            state
            for psi in wp
            for state in psi
            for state_idx in [bisect_left(basis.local_basis, state)]
            if state_idx == len(basis.local_basis) or basis.local_basis[state_idx] != state
        )
        N_max = max(N_max, basis.size)
        tmp = basis.redistribute_psis(itertools.chain(q[0], q[1], wp))
        q[0] = tmp[0:n]
        q[1] = tmp[n : 2 * n]
        wp = tmp[2 * n :]
        alpha = np.empty((n, n), dtype=complex)
        for i, j in itertools.product(range(n), repeat=2):
            alpha[i, j] = inner(q[1][i], wp[j])
        alphas = np.append(alphas, np.empty((1, n, n), dtype=complex), axis=0)
        if mpi:
            request = comm.Iallreduce(alpha, alphas[-1], op=MPI.SUM)
        else:
            alphas[-1, :, :] = alpha

        betas = np.append(betas, np.zeros((1, n, n), dtype=complex, order="C"), axis=0)

        if mpi:
            request.wait()
        tmp = [{} for _ in wp]
        for i, j in itertools.product(range(n), repeat=2):
            tmp[i] = addOps((tmp[i], scale(q[1][j], alpha[j, i])))
            tmp[i] = addOps((tmp[i], scale(q[0][j], np.conj(betas[it - 1, i, j]))))
        for i in range(n):
            wp[i] = subtractOps(wp[i], tmp[i])

        if True:
            wp, betas[-1] = qr_decomp_new(basis, wp)
        else:
            psip = basis.build_vector(wp, root=0)
            if basis.comm.rank == 0:
                psip, betas[-1], _ = qr_decomp(psip.T)
            comm.bcast(betas[-1], root=0)
            psip_local = np.empty((len(basis.local_basis), betas.shape[-1]), dtype=complex, order="C")
            send_counts = np.empty((basis.comm.size), dtype=int) if basis.comm.rank == 0 else None
            basis.comm.Gather(np.array([psip_local.size]), send_counts if basis.comm.rank == 0 else None)
            offsets = (
                np.array([np.sum(send_counts[:r]) for r in range(comm.size)], dtype=int)
                if basis.comm.rank == 0
                else None
            )
            comm.Scatterv(
                (
                    [np.ascontiguousarray(psip), send_counts, offsets, MPI.C_DOUBLE_COMPLEX]
                    if basis.comm.rank == 0
                    else None
                ),
                psip_local,
                root=0,
            )
            wp = basis.build_state(psip_local.T, slaterWeightMin=np.finfo(float).eps)

        done = converged(alphas, betas, verbose=reort == Reort.PARTIAL)

        if mpi:
            done = comm.allreduce(done, op=MPI.LAND)

        converge_count = (1 + converge_count) if done else 0
        if converge_count > 0:
            break

        q[0] = q[1]
        q[1] = wp

        if build_krylov_basis:
            Q.extend(q[1])
        it += 1
    return alphas, betas, Q if build_krylov_basis else None


def block_lanczos(
    psi0: list[dict],
    h_op: dict,
    basis: Basis,
    converged: Callable[[np.ndarray, np.ndarray], bool],
    h_mem: Optional[dict] = None,
    verbose: bool = False,
    reort: Reort = Reort.NONE,
    slaterWeightMin: float = 0,
) -> (np.ndarray, np.ndarray, Optional[list[dict]]):
    CYTHON = isinstance(h_op, ManyBodyOperator)
    if h_mem is None:
        h_mem = {}
    mpi = basis.comm is not None
    comm = basis.comm if mpi else MPI.COMM_SELF
    rank = basis.comm.rank if mpi else 0
    build_krylov_basis = reort != Reort.NONE
    n = len(psi0)
    columns = n
    N_max = basis.size
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

    q = None
    if CYTHON:
        # psi0 = [ManyBodyState(psi) for psi in psi0]
        # h_op = ManyBodyOperator(h_op)
        q = [[ManyBodyState()] * n, psi0]
    else:
        q = [[{}] * n, psi0]
    alphas = np.empty((0, n, n), dtype=complex)
    betas = np.empty((0, n, n), dtype=complex)
    if build_krylov_basis:
        Q = list(psi0)

    orth_loss = False
    force_reort = False
    perform_reort = False
    mask = []  # [False] * n
    if reort == Reort.PARTIAL:
        W = np.zeros((2, 1, n, n), dtype=complex)
        W[1, 0, :, :] = np.identity(n, dtype=complex)
    eps = max(slaterWeightMin, np.finfo(float).eps)

    it = 0
    done = False
    converge_count = 0
    wp = [None] * n
    while (it + 1) * n < basis.size or it == 0:
        if CYTHON:
            wp = [applyOp_test(h_op, psi_i, cutoff=slaterWeightMin, restrictions=basis.restrictions) for psi_i in q[1]]
        else:
            wp = [
                applyOp(
                    basis.num_spin_orbitals,
                    h_op,
                    psi_i,
                    slaterWeightMin=slaterWeightMin,
                    restrictions=basis.restrictions,
                    opResult=h_mem,
                )
                for psi_i in q[1]
            ]

        wp_size = np.array([len(psi) for psi in wp], dtype=int)
        comm.Allreduce(MPI.IN_PLACE, wp_size, op=MPI.SUM)
        cutoff = eps
        n_trunc = 0
        while np.max(wp_size) > basis.truncation_threshold:
            if CYTHON:
                for psi in wp:
                    psi.prune(cutoff)
            else:
                wp = [{state: amp for state, amp in psi.items() if abs(amp) > cutoff} for psi in wp]
            comm.Allreduce(np.array([len(psi) for psi in wp]), wp_size, op=MPI.SUM)
            cutoff *= 5
            n_trunc += 1

        if n_trunc > 0:
            basis.clear()

        basis.add_states(
            state
            for psi in wp
            for state in psi
            for state_idx in [bisect_left(basis.local_basis, state)]
            if state_idx == len(basis.local_basis) or basis.local_basis[state_idx] != state
        )
        N_max = max(N_max, basis.size)
        if CYTHON:
            # q[0] = [ManyBodyState(out) for out in basis.redistribute_psis([p.to_dict() for p in q[0]])]
            # q[1] = [ManyBodyState(out) for out in basis.redistribute_psis([p.to_dict() for p in q[1]])]
            # wp = [ManyBodyState(out) for out in basis.redistribute_psis([p.to_dict() for p in wp])]
            psim = basis.build_distributed_vector([p.to_dict() for p in q[0]]).T
            psi = basis.build_distributed_vector([p.to_dict() for p in q[1]]).T
            psip = basis.build_distributed_vector([p.to_dict() for p in wp]).T
        else:
            q[0] = basis.redistribute_psis(q[0])
            q[1] = basis.redistribute_psis(q[1])
            wp = basis.redistribute_psis(wp)
            psi = np.empty((len(basis.local_basis), n), dtype=complex, order="C")
            psip = np.empty_like(psi)
            psim = np.empty_like(psi)
            for (i, state), j in itertools.product(enumerate(basis.local_basis), range(n)):
                psi[i, j] = q[1][j].get(state, 0)
                psim[i, j] = q[0][j].get(state, 0)
                psip[i, j] = wp[j].get(state, 0)
            # psim = basis.build_distributed_vector(q[0]).T
            # psi = basis.build_distributed_vector(q[1]).T
            # psip = basis.build_distributed_vector(wp).T

        alpha = np.conj(psi.T) @ psip
        alphas = np.append(alphas, np.empty((1, n, n), dtype=complex), axis=0)
        if mpi:
            request = comm.Iallreduce(alpha, alphas[-1], op=MPI.SUM)
        else:
            alphas[-1, :, :] = alpha

        betas = np.append(betas, np.zeros((1, n, n), dtype=complex, order="C"), axis=0)
        if mpi:
            send_counts = np.empty((comm.size), dtype=int)
            request.wait()
            comm.Gather(np.array([n * len(basis.local_basis)]), send_counts)

        psip -= psi @ alphas[it] + psim @ np.conj(betas[it - 1].T)
        psip = np.ascontiguousarray(psip)
        if mpi:
            offsets = np.fromiter((np.sum(send_counts[:i]) for i in range(comm.size)), dtype=int, count=comm.size)
        if reort == Reort.FULL or reort == Reort.PERIODIC and it % 5 < 2:
            Qt = basis.redistribute_psis(Q)
            Qm = basis.build_distributed_vector(Qt).T
            tmp = np.conj(Qm.T) @ psip
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
            psip -= Qm @ tmp

        qip = np.empty((basis.size, n), dtype=complex, order="C") if rank == 0 else None
        if mpi:
            comm.Gatherv(np.ascontiguousarray(psip), [qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], root=0)
        else:
            qip = psip

        if rank == 0:
            qip, betas[-1], _ = qr_decomp(qip)
            qip = np.ascontiguousarray(qip)
            assert columns == qip.shape[1]
            _, columns = qip.shape
        if mpi:
            comm.Bcast(betas[-1], root=0)

        if reort == Reort.PARTIAL:
            W = estimate_orthonormality(W, alphas, betas, N=max(basis.size, 100), eps=eps)

            mask = np.append(mask, [False] * n)
            orth_loss = np.any(np.abs(W[1, :-1]) > np.sqrt(eps))

            if orth_loss:
                block_mask = np.abs(W[1, :-1]) > eps ** (3 / 4)
                mask = np.logical_or(mask, np.any(block_mask, axis=2).flatten())
            perform_reort = orth_loss or force_reort
            force_reort = orth_loss
            if perform_reort:
                W[1, np.argwhere(block_mask)] = eps
                Qt = basis.redistribute_psis(itertools.compress(Q, mask))
                Qm = basis.build_distributed_vector(Qt).T
                tmp = np.conj(Qm.T) @ psip
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
                psip -= Qm @ tmp
                perform_reort = False

                qip = np.empty((basis.size, n), dtype=complex, order="C") if rank == 0 else None
                if mpi:
                    comm.Gatherv(np.ascontiguousarray(psip), [qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], root=0)
                else:
                    qip = psip
                if rank == 0:
                    qip, betas[-1], _ = qr_decomp(qip)
                    qip = np.ascontiguousarray(qip)
                    _, columns = qip.shape
                if mpi:
                    comm.Bcast(betas[-1], root=0)
        if mpi:
            columns = comm.bcast(columns if rank == 0 else None)
            request = comm.Iscatterv([qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], psip, root=0)
        else:
            psip = qip

        done = converged(alphas, betas, verbose=reort == Reort.PARTIAL)

        if mpi:
            request.wait()
            done = comm.allreduce(done, op=MPI.LAND)

        converge_count = (1 + converge_count) if done else 0
        if converge_count > 0:
            break

        q[0] = q[1]
        if CYTHON:
            q[1] = [ManyBodyState(psi) for psi in basis.build_state(psip.T, slaterWeightMin=np.finfo(float).eps)]
        else:
            q[1] = basis.build_state(psip.T, slaterWeightMin=np.finfo(float).eps)

        if build_krylov_basis:
            Q.extend(q[1])
        it += 1
    return alphas, betas, Q if build_krylov_basis else None


def block_lanczos_fixed_basis(
    psi0: list[dict],
    h_op: dict,
    basis: Basis,
    converged: Callable[[np.ndarray, np.ndarray], bool],
    h_mem: Optional[dict] = None,
    verbose: bool = True,
    reort: Reort = Reort.NONE,
    slaterWeightMin: float = 0,
) -> (np.ndarray, np.ndarray, Optional[list[dict]]):
    if h_mem is None:
        h_mem = {}
    mpi = basis.comm is not None
    rank = basis.comm.rank if mpi else 0
    build_krylov_basis = reort != Reort.NONE
    n = len(psi0)
    N_max = basis.size
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
        Q = list(psi0)
    orth_loss = False
    if reort == Reort.PARTIAL:
        W = np.zeros((2, 1, n, n), dtype=complex)
        W[1] = np.identity(n)
        force_reort = None
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
                slaterWeightMin=slaterWeightMin,
                restrictions=basis.restrictions,
                opResult=h_mem,
            )
            for psi_i in q[1]
        ]
        t_apply += perf_counter() - t_tmp
        t_tmp = perf_counter()
        wp = basis.redistribute_psis(wp)
        wp = [{state: amp for state, amp in psi if state in basis.local_basis} for psi in wp]

        t_add += perf_counter() - t_tmp
        t_tmp = perf_counter()
        t_redist += perf_counter() - t_tmp
        t_tmp = perf_counter()
        psi = np.empty((len(basis.local_basis), n), dtype=complex)
        psip = np.empty_like(psi)
        psim = np.empty_like(psi)
        for (i, state), j in itertools.product(enumerate(basis.local_basis), range(n)):
            psi[i, j] = q[1][j].get(state, 0)
            psim[i, j] = q[0][j].get(state, 0)
            psip[i, j] = wp[j].get(state, 0)
        if build_krylov_basis:
            Q = basis.redistribute_psis(Q)
        t_vec += perf_counter() - t_tmp

        t_tmp = perf_counter()
        alpha = np.conj(psi.T) @ psip
        alphas = np.append(alphas, np.empty((1, n, n), dtype=complex), axis=0)
        if mpi:
            request = basis.comm.Iallreduce(alpha, alphas[-1], op=MPI.SUM)
        else:
            alphas[-1, :, :] = alpha

        betas = np.append(betas, np.zeros((1, n, n), dtype=complex), axis=0)
        if mpi:
            send_counts = np.empty((basis.comm.size), dtype=int)
            basis.comm.Gather(np.array([n * len(basis.local_basis)]), send_counts)
            request.Wait()

        psip -= psi @ alphas[it] + psim @ np.conj(betas[it - 1].T)
        t_linalg += perf_counter() - t_tmp
        t_tmp = perf_counter()
        if mpi:
            offsets = np.fromiter(
                (np.sum(send_counts[:i]) for i in range(basis.comm.size)), dtype=int, count=basis.comm.size
            )

        if reort == Reort.FULL:
            Qm = basis.build_distributed_vector(Q).T
            if mpi:
                tmp = np.empty((Qm.shape[1], n), dtype=complex)
                basis.comm.Allreduce(np.conj(Qm.T) @ psip, tmp, op=MPI.SUM)
            else:
                tmp = np.conj(Qm.T) @ psip
            psip -= Qm @ tmp
        elif reort == Reort.PARTIAL and it > 0:
            W = estimate_orthonormality(W, alphas, betas, N=1)
            orth_loss = np.any(np.abs(W[1, :-1]) > np.sqrt(np.finfo(float).eps))
            if orth_loss or force_reort is not None:
                mask = np.any(np.abs(W[1, :-1]) > np.finfo(float).eps ** (3 / 4), axis=1)
                combined_mask = (
                    np.logical_or(mask, np.append(force_reort, [[False] * n], axis=0))
                    if force_reort is not None
                    else mask
                )
                Qm = basis.build_distributed_vector(list(itertools.compress(Q, combined_mask.flatten()))).T
                W[1, :-1][combined_mask] = np.finfo(
                    float
                ).eps  #  * np.random.normal(loc=0, scale=1.5, size=W[1, :-2].shape)
                basis.comm.Bcast(W[1])
                force_reort = None if force_reort is not None else mask
            else:
                Qm = np.zeros((len(basis.local_basis), 0), dtype=complex)
        else:
            Qm = np.zeros((len(basis.local_basis), 0), dtype=complex)

        if reort == Reort.FULL or reort == Reort.PARTIAL:
            if mpi:
                tmp = np.empty((Qm.shape[1], n), dtype=complex)
                basis.comm.Allreduce(np.conj(Qm.T) @ psip, tmp, op=MPI.SUM)
            else:
                tmp = np.conj(Qm.T) @ psip
            psip -= Qm @ tmp
        qip = np.empty((basis.size, n), dtype=complex) if rank == 0 else None
        if mpi:
            basis.comm.Gatherv(psip, [qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], root=0)
        else:
            qip = psip
        t_vec += perf_counter() - t_tmp
        if rank == 0:
            t_tmp = perf_counter()
            qip, betas[-1], _ = qr_decomp(qip)
            _, columns = qip.shape

        if mpi:
            basis.comm.Bcast(betas[it], root=0)
            columns = basis.comm.bcast(columns if rank == 0 else None)
            request = basis.comm.Iscatterv([qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], psip.T, root=0)
        else:
            psip = qip

        if it % 1 == 0 or converge_count > 0:
            t_tmp = perf_counter()
            done = converged(alphas, betas)
            t_conv += perf_counter() - t_tmp

        if mpi:
            done = basis.comm.allreduce(done, op=MPI.LAND)

        converge_count = (1 + converge_count) if done else 0
        if converge_count > 0:
            break

        t_tmp = perf_counter()
        q[0] = q[1]
        q[1] = [{} for _ in range(columns)]
        if mpi:
            request.Wait()
        for j, (i, state) in itertools.product(range(columns), enumerate(basis.local_basis)):
            q[1][j][state] = psip[i, j]

        t_state += perf_counter() - t_tmp
        if build_krylov_basis:
            Q.extend(q[1])
        it += 1
    if verbose:
        print()
        print(f"Breaking after iteration {it}, blocksize = {n}")
        print(f"===> Maximum basis size: {N_max} Slater determinants")
        print(f"===> Applying the hamiltonian took {t_apply:.4f} seconds")
        print(f"===> Adding states took {t_add:.4f} seconds")
        print(f"===> Redistributing states took {t_redist:.4f} seconds")
        print(f"===> Local linear algebra took {t_linalg:.4f} seconds")
        print(f"===> Building vectors took {t_vec:.4f} seconds")
        print(f"===> Estimating convergence took {t_conv:.4f} seconds")
        print(f"===> QR factorization took {t_qr:.4f} seconds")
        print(f"===> Building states took {t_state:.4f} seconds")
        print(f"=> time(get_block_Lanczons_matrices) = {perf_counter() - t_tot:.4f} seconds.")
        print("=" * 80)
    return alphas, betas, Q if build_krylov_basis else None
