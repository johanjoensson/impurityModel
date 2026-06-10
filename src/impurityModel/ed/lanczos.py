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
    # inner,
    matmul,
    removeFromFirst,
    addOps,
    subtractOps,
    scale,
    norm2,
)
from cmath import phase, rect
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, applyOp as applyOp_test, inner
from impurityModel.ed.utils import matrix_print


class Reort(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2
    PERIODIC = 3


def calculate_thermal_gs(h, block_size, e_max, v0=None, reort=Reort.FULL, comm=None):
    def converged(alphas, betas, *args, **kwargs):
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
    within_gs = False
    if select == "m":
        assert de is not None
        select = "a"
        within_gs = True

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
    if within_gs:
        mask = eigvals - np.min(eigvals) <= de
    else:
        mask = np.ones(len(eigvals), dtype=bool)
    if Q is not None:
        eigvecs = Q @ eigvecs
    sort_indices = np.argsort(eigvals[mask])
    mask_indices = np.where(mask)[0]
    return eigvals[mask_indices][sort_indices], eigvecs[:, mask_indices][:, sort_indices]


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


def qr_decomp(psi, pivoting=False):
    if pivoting:
        psi, beta, p = sp.linalg.qr(psi, mode="economic", overwrite_a=True, check_finite=False, pivoting=True)
        return np.ascontiguousarray(psi), beta, p
    psi, beta = sp.linalg.qr(psi, mode="economic", overwrite_a=True, check_finite=False, pivoting=False)
    return np.ascontiguousarray(psi), beta, None


def get_block_Lanczos_matrices_dense(
    psi0: np.ndarray,
    h,
    converged: Callable[[np.ndarray, np.ndarray], bool],
    verbose: bool = True,
):
    krylovSize = h.shape[0]

    N = h.shape[0]
    n = psi0.shape[1] if len(psi0.shape) == 2 else 1

    q = np.zeros((2, N, n), dtype=complex, order="C")
    wp = np.empty((N, n), dtype=complex, order="C")
    q[1] = psi0

    it_max = int(np.ceil(krylovSize / n))
    alphas = np.empty((it_max, n, n), dtype=complex)
    betas = np.empty((it_max, n, n), dtype=complex)

    # Run at least 1 iteration (to generate $\alpha_0$).
    # We can also not generate more than N Lanczos vectors, meaning we can
    # take at most N/n steps in total
    converge_count = 0
    for i in range(it_max):
        wp = h @ q[1]

        alphas[i] = np.conj(q[1].T) @ wp
        if i == 0:
            wp = wp - q[1] @ alphas[i]
        else:
            wp -= q[1] @ alphas[i] + q[0] @ np.conj(betas[i - 1].T)
        q[0] = q[1]
        q[1], betas[i] = sp.linalg.qr(wp, mode="economic", overwrite_a=True, check_finite=False)

        converge_count = 1 + converge_count if converged(alphas[: i + 1], betas[: i + 1], verbose=verbose) else 0
        if converge_count > 2:
            break

    return alphas[: i + 1], betas[: i + 1]


def get_block_Lanczos_matrices(
    psi0: np.ndarray,
    h,
    converged: Callable[[np.ndarray, np.ndarray], bool],
    verbose: bool = True,
    comm=None,
):
    mpi = comm is not None
    rank = comm.rank if mpi else 0
    krylovSize = h.shape[0]
    if mpi:
        counts = np.empty((comm.size), dtype=int)
        comm.Allgather(np.array([psi0.size], dtype=int), counts)
        offsets = np.array([np.sum(counts[:r]) for r in range(comm.size)], dtype=int)

    N = h.shape[0]
    n = psi0.shape[1] if len(psi0.shape) == 2 else 1
    it_max = int(np.ceil(krylovSize / n))

    n_reort = 0
    if rank == 0:
        q = np.zeros((2, N, n), dtype=complex, order="C")
        wp = np.empty((N, n), dtype=complex, order="C")
        alphas = np.empty((it_max, n, n), dtype=complex, order="C")
        betas = np.empty((it_max, n, n), dtype=complex, order="C")
    else:
        q = np.empty((2, 0, 0))
        wp = None
    if mpi:
        comm.Gatherv(psi0, (q[1], counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
    else:
        q[1] = psi0.copy()
    qi = np.ascontiguousarray(psi0).copy()

    converge_count = 0
    done = False
    # Run at least 1 iteration (to generate $\alpha_0$).
    # We can also not generate more than N Lanczos vectors, meaning we can
    # take at most N/n steps in total
    for i in range(it_max):
        wp = h @ qi

        if rank == 0:
            alphas[i] = np.conj(q[1].T) @ wp
            if i == 0:
                wp -= q[1] @ alphas[i]
            else:
                wp -= q[1] @ alphas[i] + q[0] @ np.conj(betas[i - 1].T)
            q[0] = q[1].copy()
            q[1], betas[i] = sp.linalg.qr(wp, mode="economic", overwrite_a=False, check_finite=False)
            q[1] = np.ascontiguousarray(q[1])
            qi[:] = q[1, : counts[0] // n]

            converge_count = 1 + converge_count if converged(alphas[: i + 1], betas[: i + 1], verbose=verbose) else 0
            done = converge_count > 2

        if mpi:
            done = comm.bcast(done, root=0)
            comm.Scatterv(
                (q[1], counts, offsets, MPI.C_DOUBLE_COMPLEX),
                qi,
                root=0,
            )

        else:
            qi = q[1].copy()
        if done:
            break

    if rank == 0:
        alphas = alphas[: i + 1]
        betas = betas[: i + 1]
    # Distribute Lanczos matrices to all ranks
    if mpi:
        if rank != 0:
            alphas = np.empty((i + 1, n, n), dtype=complex, order="C")
            betas = np.empty((i + 1, n, n), dtype=complex, order="C")
        comm.Bcast(alphas, root=0)
        comm.Bcast(betas, root=0)

    return alphas, betas


def block_lanczos_sparse(
    psi0: list[ManyBodyState],
    h_op: ManyBodyOperator,
    basis: Basis,
    converged: Callable[[np.ndarray, np.ndarray], bool],
    verbose: bool = False,
    reort: Reort = Reort.NONE,
    slaterWeightMin: float = 0,
) -> (np.ndarray, np.ndarray, Optional[list[dict]]):
    mpi = basis.comm is not None
    comm = basis.comm if mpi else MPI.COMM_SELF
    rank = comm.rank if mpi else 0
    build_krylov_basis = reort != Reort.NONE
    n = len(psi0)
    N_max = basis.size

    alphas = np.empty((0, n, n), dtype=complex, order="C")
    betas = np.empty((0, n, n), dtype=complex, order="C")
    q = [[ManyBodyState({}) for _ in range(n)], psi0]
    if build_krylov_basis:
        Q = list(psi0)

    it_max = basis.size // n + 1
    it = 0
    converge_count = 0
    expand_basis = True

    local_rows = len(basis.local_basis)
    q1_local = np.empty((local_rows, n), dtype=complex, order="C")
    while it * n < basis.size:
        t0 = perf_counter()
        wp = [
            applyOp_test(
                h_op,
                psi_i,
                cutoff=slaterWeightMin,
            )
            for psi_i in q[1]
        ]
        t_apply = perf_counter() - t0
        old_basis_size = basis.size
        t0 = perf_counter()
        if expand_basis:
            basis.add_states(
                set(state for p in wp for state in p if state not in basis.local_basis),
            )
            local_rows = len(basis.local_basis)
            q1_local = np.empty((local_rows, n), dtype=complex, order="C")
        t_add = perf_counter() - t0
        if old_basis_size == basis.size:
            it_max = basis.size // n
            # expand_basis = False
        if verbose:
            print(f"Added {basis.size - old_basis_size} states to the basis.")
            print(f"----> Currently the basis contains {basis.size} states.")
            print(f"----> Applying the hamiltonian took {t_apply} seconds.")
            print(f"----> Adding new states took {t_add} seconds.", flush=True)
        tmp = basis.redistribute_psis(q[0] + q[1] + wp)
        v_dense = basis.build_vector(tmp, slaterWeightMin=0, root=0).T
        if rank == 0:
            q0_dense = v_dense[:, :n]
            q1_dense = v_dense[:, n : 2 * n]
            wp_dense = v_dense[:, 2 * n :]
            alphas = np.append(alphas, [np.conj(q1_dense.T) @ wp_dense], axis=0)
            betas = np.append(betas, np.zeros((1, n, n), dtype=complex), axis=0)
            if it == 0:
                wp_dense -= q1_dense @ alphas[it]
            else:
                wp_dense -= q1_dense @ alphas[it] + q0_dense @ np.conj(betas[it - 1].T)
            q1_dense, betas[it] = sp.linalg.qr(wp_dense, mode="economic", overwrite_a=True, check_finite=False)
            q1_dense = np.ascontiguousarray(q1_dense)
            q1_local[:] = q1_dense[basis.local_indices]
        else:
            alphas = np.append(
                alphas,
                np.empty(
                    (1, n, n),
                    dtype=alphas.dtype,
                ),
                axis=0,
            )
            betas = np.append(
                betas,
                np.empty(
                    (1, n, n),
                    dtype=betas.dtype,
                ),
                axis=0,
            )
        if mpi:
            comm.Bcast(alphas[-1], root=0)
            comm.Bcast(betas[-1], root=0)

        if converged(alphas, betas, verbose=verbose):
            converge_count += 1
        else:
            converge_count = 0
        if converge_count > 2 or it > it_max:
            break

        if mpi:

            # Gather local row counts from all ranks on rank 0
            send_counts = np.empty((comm.size), dtype=int) if rank == 0 else None
            comm.Gather(np.array([local_rows], dtype=int), send_counts if rank == 0 else None, root=0)

            # Calculate correct offsets based on physical matrix rows
            offsets = np.array([np.sum(send_counts[:r]) for r in range(comm.size)], dtype=int) if rank == 0 else None

            # Multiply counts/offsets by block-width n to scatter full contiguous blocks
            if rank == 0:
                send_counts *= n
                offsets *= n
                q1_dense_flat = np.ascontiguousarray(q1_dense).reshape(-1)
            else:
                q1_dense_flat = None

            comm.Scatterv(
                [q1_dense_flat, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None,
                q1_local,
                root=0,
            )
        else:
            q1_local = np.ascontiguousarray(q1_dense)
        q[0] = tmp[n : 2 * n]
        q[1] = basis.build_state(q1_local.T, slaterWeightMin=0)
        if build_krylov_basis:
            Q.extend(q[1])
        it += 1
    if verbose:
        print(f"Coverged at iteration {it+1} out of a maximum of {basis.size//n}")
    return alphas, betas


def block_lanczos(
    psi0: list[ManyBodyState],
    h_op: ManyBodyOperator,
    basis: Basis,
    converged: Callable[[np.ndarray, np.ndarray], bool],
    h_mem: Optional[dict] = None,
    verbose: bool = False,
    reort: Reort = Reort.NONE,
    slaterWeightMin: float = 0,
) -> (np.ndarray, np.ndarray, Optional[list[dict]]):
    CYTHON = isinstance(h_op, ManyBodyOperator)
    if h_mem is None and not CYTHON:
        h_mem = {}
    mpi = basis.comm is not None
    comm = basis.comm if mpi else MPI.COMM_SELF
    rank = basis.comm.rank if mpi else 0
    build_krylov_basis = reort != Reort.NONE

    # Convert psi0 keys to SlaterDeterminant if they are bytes
    psi0 = [
        {basis.type.from_bytes(k) if isinstance(k, bytes) else k: v for k, v in psi.items()}
        for psi in psi0
    ]

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
        q = [[ManyBodyState() for _ in range(n)], [ManyBodyState(psi) for psi in psi0]]
    else:
        q = [[{} for _ in range(n)], psi0]
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
    t_op = 0
    t_add = 0
    t_build = 0
    t_algo = 0
    t_conv = 0
    t_dist = 0
    t_tot = perf_counter()
    while it * n < basis.size or it == 0:
        t0 = perf_counter()
        if CYTHON:
            wp = [applyOp_test(h_op, psi_i, cutoff=slaterWeightMin) for psi_i in q[1]]
        else:
            wp = []
            for psi_i in q[1]:
                psi_i_bytes = {
                    bytes(k.to_bytearray()) if hasattr(k, "to_bytearray") else k: v
                    for k, v in psi_i.items()
                }
                wp_i_bytes = applyOp(basis.num_spin_orbitals, h_op, psi_i_bytes, slaterWeightMin=slaterWeightMin)
                wp_i = {
                    basis.type.from_bytes(k) if isinstance(k, bytes) else k: v
                    for k, v in wp_i_bytes.items()
                }
                wp.append(wp_i)
        t_op += perf_counter() - t0

        wp_size = np.array([len(psi) for psi in wp], dtype=int)
        comm.Allreduce(MPI.IN_PLACE, wp_size, op=MPI.SUM)
        cutoff = eps
        n_trunc = 0
        while np.max(wp_size) > basis.truncation_threshold:
            for psi in wp:
                if CYTHON:
                    psi.prune(cutoff)
                else:
                    for state in list(psi.keys()):
                        if np.abs(psi[state]) < cutoff:
                            del psi[state]
            comm.Allreduce(np.array([len(psi) for psi in wp]), wp_size, op=MPI.SUM)
            cutoff *= 5
            n_trunc += 1

        if n_trunc > 0:
            basis.clear()

        t0 = perf_counter()
        basis.add_states(
            state
            for psi in wp
            for state in psi
            for state_idx in [bisect_left(basis.local_basis, state)]
            if state_idx == len(basis.local_basis) or basis.local_basis[state_idx] != state
        )
        t_add += perf_counter() - t0
        N_max = max(N_max, basis.size)
        t0 = perf_counter()
        psim = basis.build_distributed_vector(q[0]).T
        psi = basis.build_distributed_vector(q[1]).T
        psip = basis.build_distributed_vector(wp).T
        t_build += perf_counter() - t0

        t0 = perf_counter()
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
            comm.Gatherv(psip, [qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], root=0)
        else:
            qip = psip

        if rank == 0:
            qip, betas[-1], _ = qr_decomp(qip)
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
                    comm.Gatherv(psip, [qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], root=0)
                else:
                    qip = psip
                if rank == 0:
                    qip, betas[-1], _ = qr_decomp(qip)
                    _, columns = qip.shape
                if mpi:
                    comm.Bcast(betas[-1], root=0)
        if mpi:
            columns = comm.bcast(columns if rank == 0 else None)
            request = comm.Iscatterv([qip, send_counts, offsets, MPI.C_DOUBLE_COMPLEX], psip, root=0)
        else:
            psip = qip
        t_algo += perf_counter() - t0

        t0 = perf_counter()
        done = converged(alphas, betas, verbose=reort == Reort.PARTIAL)
        t_conv += perf_counter() - t0
        if mpi:
            request.wait()
            done = comm.allreduce(done, op=MPI.LAND)

        converge_count = (1 + converge_count) if done else 0
        if converge_count > 0:
            break

        t0 = perf_counter()
        q[0] = q[1]
        q[1] = basis.build_state(psip.T, slaterWeightMin=np.finfo(float).eps)
        t_dist += perf_counter() - t0

        if build_krylov_basis:
            Q.extend(q[1])
        it += 1
    t_tot = perf_counter() - t_tot
    if rank == 0:
        print(f"Basis size:        {N_max} determinants")
        print(f"block_lanczos took {t_tot} seconds")
        print(f"--> applyOp took   {t_op} seconds")
        print(f"--> add states     {t_add} seconds")
        print(f"--> build vecs     {t_build} seconds")
        print(f"--> algorithm took {t_algo} seconds")
        print(f"--> convergence    {t_conv} seconds")
        print(f"--> distribute     {t_dist} seconds")
        print(f"--> applyOp took   {t_op/(it+1)} seconds per iteration")
        print(f"--> add states     {t_add/(it+1)} second per iterations")
    return alphas, betas, Q if build_krylov_basis else None


def get_Lanczos_vectors(A, alphas, betas, v0, comm, which="all"):
    mpi = comm is not None
    rank = comm.rank if mpi else 0
    counts = np.array([v0.size], dtype=int)
    offsets = np.array([0], dtype=int)
    n_it = alphas.shape[0]
    N = v0.shape[0]
    n = alphas.shape[1]
    q = np.empty((2, 0, 0), dtype=complex)
    if rank == 0:
        q = np.zeros((2, A.shape[0], n), dtype=complex, order="C")
    if mpi:
        counts = np.empty((comm.size), dtype=int)
        comm.Allgather(np.array([v0.size], dtype=int), counts)
        offsets = np.array([np.sum(counts[:r]) for r in range(comm.size)], dtype=int)
        comm.Gatherv(v0, (q[1] if rank == 0 else None, counts, offsets, MPI.C_DOUBLE_COMPLEX))
    else:
        q[1] = v0
    assert (n_it + 1) * n == (alphas.shape[0] + 1) * alphas.shape[1]
    if which == "all":
        which = list(range(n_it))
    elif isinstance(which, int):
        if which < 0:
            which = [n_it + which]
        else:
            which = [which]
    elif isinstance(which, list):
        which = [w if w >= 0 else n_it + w for w in which]
    elif isinstance(which, tuple):
        which = list(range(min(which[0], which[1]), max(which[0], which[1])))
    else:
        raise RuntimeError(f"Unknown value for which: {which}")

    Q = np.empty((N, len(which) * n), dtype=alphas.dtype)
    qi = v0.copy()
    for i in range(n_it):
        if i in which:
            idx = which.index(i)
            Q[:, idx * n : (idx + 1) * n] = qi
        q_tmp = A @ qi
        if rank == 0:
            if i == 0:
                q_tmp -= q[1] @ alphas[i]
            else:
                q_tmp -= q[1] @ alphas[i] + q[0] @ np.conj(betas[i - 1].T)
            q[0] = q[1]
            q[1], _ = sp.linalg.qr(q_tmp, mode="economic", overwrite_a=True, check_finite=False)
        if mpi:
            comm.Scatterv(
                (q[1], counts, offsets, MPI.C_DOUBLE_COMPLEX),
                qi,
                root=0,
            )
        else:
            qi = q[1]
    if n_it in which:
        idx = which.index(n_it)
        Q[:, idx * n : (idx + 1) * n] = qi

    return Q
