import itertools
import numpy as np
import scipy as sp
import time
from typing import Optional, Iterable

from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.lanczos import get_block_Lanczos_matrices, block_lanczos
from impurityModel.ed.manybody_basis import CIPSI_Basis, Basis

from mpi4py import MPI


def split_comm_and_redistribute_basis(priorities: Iterable[float], basis: Basis, psis: list[dict]):
    """
    Split MPI communicator in order to divide MPI ranks among items, number of ranks per item is determined using the priorities.
    Higher priority means more MPI ranks assigned to the element.
    """
    comm = basis.comm
    normalized_priorities = np.array([p for p in priorities], dtype=float)
    normalized_priorities /= np.sum(normalized_priorities)
    n_colors = min(comm.size, len(normalized_priorities))
    if len(normalized_priorities) < comm.size:
        procs_per_color = np.array([max(1, n) for n in comm.size * normalized_priorities], dtype=int)
    else:
        procs_per_color = np.array([1] * n_colors, dtype=int)
    diff = np.sum(procs_per_color) - comm.size
    sorted_indices = np.argsort(normalized_priorities)[::-1]
    if diff != 0:
        for _ in range(1 + (abs(diff) // n_colors)):
            if diff > 0:
                mask = sorted_indices[procs_per_color > 1]
                procs_per_color[mask[-(diff % n_colors) :]] -= 1
                diff = np.sum(procs_per_color) - comm.size
            else:
                procs_per_color[sorted_indices[: abs(diff) % n_colors]] += 1
                diff = np.sum(procs_per_color) - comm.size
    assert sum(procs_per_color) == comm.size
    proc_cutoffs = np.cumsum(procs_per_color)
    color = np.argmax(comm.rank < proc_cutoffs)
    split_comm = comm.Split(color=color, key=0)
    split_roots = [0] + proc_cutoffs[:-1].tolist()
    items_per_color = np.array([len(priorities) // n_colors] * n_colors, dtype=int)
    items_per_color[: len(priorities) % n_colors] += 1
    indices_start = sum(items_per_color[:color])
    indices_end = sum(items_per_color[: color + 1])

    if split_comm.rank == 0:
        assert comm.rank in split_roots

    for color_root in split_roots:
        if comm.rank == color_root:
            all_psis = comm.gather(psis, root=color_root)
        else:
            _ = comm.gather(psis, root=color_root)
    psis = [{} for _ in range(len(psis))]
    if split_comm.rank == 0:
        for partial_psis in all_psis:
            for i, partial_psi in enumerate(partial_psis):
                for state in partial_psi:
                    psis[i][state] = partial_psi[state] + psis[i].get(state, 0)
    split_basis = Basis(
        impurity_orbitals=basis.impurity_orbitals,
        valence_baths=basis.bath_states[0],
        conduction_baths=basis.bath_states[1],
        initial_basis=(state for psi in psis for state in psi),
        restrictions=basis.restrictions,
        comm=split_comm,
        verbose=basis.verbose,
        truncation_threshold=basis.truncation_threshold,
        tau=basis.tau,
        spin_flip_dj=basis.spin_flip_dj,
    )
    psis = split_basis.redistribute_psis(psis)

    return slice(indices_start, indices_end), split_roots, n_colors, items_per_color, split_basis, psis


def get_Greens_function(
    matsubara_mesh,
    omega_mesh,
    psis,
    es,
    tau,
    basis,
    hOp,
    delta,
    blocks,
    verbose,
    reort,
):
    """
    Calculate interacting Greens function.
    """
    (
        block_indices,
        block_roots,
        n_colors,
        blocks_per_color,
        block_basis,
        psis,
    ) = split_comm_and_redistribute_basis([len(block) ** 2 for block in blocks], basis, psis)
    gs_matsubara_block = []
    gs_realaxis_block = []
    for opIPS, opPS in (
        (
            [{((orb, "c"),): 1} for orb in block],
            [{((orb, "a"),): 1} for orb in block],
        )
        for block in blocks[block_indices]
    ):
        gsIPS_matsubara, gsIPS_realaxis = calc_Greens_function_with_offdiag(
            hOp,
            opIPS,
            psis,
            es,
            tau,
            block_basis,
            matsubara_mesh,
            omega_mesh,
            delta,
            reort=reort,
            slaterWeightMin=1e-12,
            verbose=verbose,
        )
        gsPS_matsubara, gsPS_realaxis = calc_Greens_function_with_offdiag(
            hOp,
            opPS,
            psis,
            es,
            tau,
            block_basis,
            -matsubara_mesh if matsubara_mesh is not None else None,
            -omega_mesh if omega_mesh is not None else None,
            -delta,
            slaterWeightMin=1e-12,
            verbose=verbose,
            reort=reort,
        )

        if matsubara_mesh is not None and block_basis.comm.rank == 0:
            gs_matsubara_block.append(
                gsIPS_matsubara
                - np.transpose(
                    gsPS_matsubara,
                    (
                        0,
                        2,
                        1,
                    ),
                )
            )
        if omega_mesh is not None and block_basis.comm.rank == 0:
            gs_realaxis_block.append(
                gsIPS_realaxis
                - np.transpose(
                    gsPS_realaxis,
                    (
                        0,
                        2,
                        1,
                    ),
                )
            )
    gs_matsubara = (
        [np.empty((len(matsubara_mesh), len(block), len(block)), dtype=complex) for block in blocks]
        if basis.comm.rank == 0
        else None
    )
    gs_realaxis = (
        [np.empty((len(omega_mesh), len(block), len(block)), dtype=complex) for block in blocks]
        if basis.comm.rank == 0
        else None
    )
    requests = []
    if block_basis.comm.rank == 0:
        if matsubara_mesh is not None:
            for matsubara_block_gs in gs_matsubara_block:
                requests.append(basis.comm.Isend(matsubara_block_gs, 0))
        if omega_mesh is not None:
            for real_block_gs in gs_realaxis_block:
                requests.append(basis.comm.Isend(real_block_gs, 0))
    if basis.comm.rank == 0:
        for color, color_root in zip(range(n_colors), block_roots):
            block_is = range(sum(blocks_per_color[:color]), sum(blocks_per_color[: color + 1]))
            if matsubara_mesh is not None:
                for block_i in block_is:
                    requests.append(basis.comm.Irecv(gs_matsubara[block_i], color_root))
            if omega_mesh is not None:
                for block_i in block_is:
                    requests.append(basis.comm.Irecv(gs_realaxis[block_i], color_root))
    if len(requests) > 0:
        requests[-1].Waitall(requests)
    return gs_matsubara, gs_realaxis


def calc_Greens_function_with_offdiag(
    hOp,
    tOps,
    psis,
    es,
    tau,
    basis,
    iw,
    w,
    delta,
    reort,
    slaterWeightMin=0,
    verbose=True,
):
    r"""
    Return Green's function for states with low enough energy.

    For states :math:`|psi \rangle`, calculate:

    :math:`g(w+1j*delta) =
    = \langle psi| tOp^\dagger ((w+1j*delta+e)*\hat{1} - hOp)^{-1} tOp
    |psi \rangle`,

    where :math:`e = \langle psi| hOp |psi \rangle`

    Lanczos algorithm is used.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    hOp : dict
        Operator
    tOps : list
        List of dict operators
    psis : list
        List of Multi state dictionaries
    es : list
        Total energies
    w : list
        Real axis energy mesh
    delta : float
        Deviation from real axis.
        Broadening/resolution parameter.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    krylovSize : int
        Size of the Krylov space
    slaterWeightMin : float
        Restrict the number of product states by
        looking at `|amplitudes|^2`.
    parallelization_mode : str
            "eigen_states" or "H_build".

    """
    comm = basis.comm
    n = len(tOps)
    # excited_restrictions = None
    excited_restrictions = basis.build_excited_restrictions(imp_change=(1, 1), val_change=(1, 0), con_change=(0, 1))

    t_mems = [{} for _ in tOps]
    h_mem = {}
    if iw is not None:
        gs_matsubara_block = np.zeros((len(iw), n, n), dtype=complex)
    else:
        gs_matsubara_block = None
    if w is not None:
        gs_realaxis_block = np.zeros((len(w), n, n), dtype=complex)
    else:
        gs_realaxis_block = None

    (
        eigen_indices,
        eigen_roots,
        n_colors,
        eigen_per_color,
        eigen_basis,
        psis,
    ) = split_comm_and_redistribute_basis([1 for _ in es], basis, psis)
    gs_matsubara_received = np.empty((len(eigen_roots), len(iw), n, n), dtype=complex) if comm.rank == 0 else None
    gs_realaxis_received = np.empty((len(eigen_roots), len(w), n, n), dtype=complex) if comm.rank == 0 else None
    e0 = min(es)
    Z = np.sum(np.exp(-(es - e0) / tau))
    for psi, e in zip(psis[eigen_indices], es[eigen_indices]):
        block_v = []
        local_excited_basis = set()
        t0 = time.perf_counter()
        for i_tOp, tOp in enumerate(tOps):
            v = finite.applyOp_new(
                eigen_basis.num_spin_orbitals,
                tOp,
                psi,
                slaterWeightMin=slaterWeightMin,
                restrictions=basis.restrictions,
                opResult=t_mems[i_tOp],
            )
            local_excited_basis |= v.keys()
            block_v.append(v)

        excited_basis = Basis(
            impurity_orbitals=eigen_basis.impurity_orbitals,
            valence_baths=eigen_basis.bath_states[0],
            conduction_baths=eigen_basis.bath_states[1],
            initial_basis=local_excited_basis,
            restrictions=excited_restrictions,
            comm=eigen_basis.comm,
            verbose=verbose,
            truncation_threshold=eigen_basis.truncation_threshold,
            tau=eigen_basis.tau,
            spin_flip_dj=eigen_basis.spin_flip_dj,
        )

        if verbose:
            print(f"time(build excited state basis) = {time.perf_counter() - t0}")
        gs_matsubara_block_i, gs_realaxis_block_i = block_Green(
            n_spin_orbitals=excited_basis.num_spin_orbitals,
            hOp=hOp,
            psi_arr=block_v,
            basis=excited_basis,
            e=e,
            iws=iw,
            ws=w,
            delta=delta,
            h_mem=h_mem,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose,
            reort=reort,
        )
        if excited_basis.comm.rank == 0:
            if iw is not None:
                gs_matsubara_block += np.exp(-(e - e0) / tau) / Z * gs_matsubara_block_i
            if w is not None:
                gs_realaxis_block += np.exp(-(e - e0) / tau) / Z * gs_realaxis_block_i
    # Send calculated Greens functions to root
    requests = []
    if eigen_basis.comm.rank == 0:
        if iw is not None:
            requests.append(comm.Isend(gs_matsubara_block, 0))
        if w is not None:
            requests.append(comm.Isend(gs_realaxis_block, 0))
    if comm.rank == 0:
        for i, r in enumerate(eigen_roots):
            if iw is not None:
                requests.append(comm.Irecv(gs_matsubara_received[i], r))
        for i, r in enumerate(eigen_roots):
            if w is not None:
                requests.append(comm.Irecv(gs_realaxis_received[i], r))
        requests[-1].Waitall(requests)
        if iw is not None:
            gs_matsubara_block = np.sum(gs_matsubara_received, axis=0)
        if w is not None:
            gs_realaxis_block = np.sum(gs_realaxis_received, axis=0)
    # if len(requests) > 0:
    #     requests[-1].Waitall(requests)
    return gs_matsubara_block, gs_realaxis_block


def get_block_Green(
    n_spin_orbitals,
    hOp,
    psi_arr,
    basis,
    e,
    iws,
    ws,
    delta,
    reort,
    restrictions=None,
    h_mem=None,
    mode="sparse",
    slaterWeightMin=0,
    parallelization_mode="H_build",
    verbose=True,
    dense_cutoff=1e3,
):
    """
    Calculate one block of the inteacting Greens function. Including offdiagonal terms.
    """
    comm = basis.comm
    rank = comm.rank
    matsubara = iws is not None
    realaxis = ws is not None

    if not matsubara and not realaxis:
        if rank == 0:
            print("No Matsubara mesh or real frequency mesh provided. No Greens function will be calculated.")
        return None, None

    if h_mem is None:
        h_mem = {}
    h_local = True

    if verbose:
        t0 = time.perf_counter()
    h = basis.build_sparse_matrix(hOp, h_mem)

    if verbose:
        print(f"time(build Hamiltonian operator) = {time.perf_counter() - t0}")

    N = len(basis)
    n = len(psi_arr)

    if n == 0 or N == 0:
        return np.zeros((n, n, len(iws)), dtype=complex), np.zeros((n, n, len(ws)), dtype=complex)
    if verbose:
        t0 = time.perf_counter()
    psi_start = np.array(basis.build_distributed_vector(psi_arr).T, copy=False, order="C")
    counts = np.empty((comm.size,), dtype=int) if comm.rank == 0 else None
    comm.Gather(np.array([n * len(basis.local_basis)], dtype=int), counts, root=0)
    offsets = [sum(counts[:r]) for r in range(len(counts))] if comm.rank == 0 else None
    psi_start_0 = np.empty((N, n), dtype=complex, order="C") if comm.rank == 0 else None
    comm.Gatherv(psi_start, (psi_start_0, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
    r: Optional[np.ndarray] = None
    if comm.rank == 0:
        # Do a QR decomposition of the starting block.
        # Later on, use r to restore the block corresponding to
        psi0_0, r = sp.linalg.qr(psi_start_0, mode="economic", overwrite_a=True, check_finite=False, pivoting=False)
        psi0_0 = np.array(psi0_0, copy=False, order="C")
        # Find which columns (if any) are 0 in psi0_0
        rows, columns = psi0_0.shape
    rows = comm.bcast(rows if comm.rank == 0 else None, root=0)
    columns = comm.bcast(columns if comm.rank == 0 else None, root=0)
    if rows == 0 or columns == 0:
        return np.zeros((n, n, len(iws)), dtype=complex), np.zeros((n, n, len(ws)), dtype=complex)

    counts = np.empty((comm.size,), dtype=int) if comm.rank == 0 else None
    comm.Gather(np.array([columns * len(basis.local_basis)], dtype=int), counts, root=0)
    offsets = [sum(counts[:r]) for r in range(len(counts))] if comm.rank == 0 else None
    psi0 = np.zeros((len(basis.local_basis), columns), dtype=complex)
    comm.Scatterv((psi0_0, counts, offsets, MPI.C_DOUBLE_COMPLEX) if comm.rank == 0 else None, psi0, root=0)
    if verbose:
        print(f"time(set up psi_start) = {time.perf_counter() - t0}")

    # If we have a realaxis mesh, prefer to check convergence on that
    # if not, use the Matsubara mesh
    if realaxis:
        conv_w = ws
        delta_p = delta
    elif matsubara:
        conv_w = iws
        delta_p = 0

    # Select points from the frequency mesh, according to a Normal distribuition
    # centered on (value) 0.
    n_samples = max(len(conv_w) // 100, 1)

    def converged(alphas, betas):
        if alphas.shape[0] == 1:
            return False

        w = np.random.choice(conv_w, size=n_samples, replace=False)
        wIs = (w + 1j * delta_p + e)[:, np.newaxis, np.newaxis] * np.identity(alphas.shape[1], dtype=complex)[
            np.newaxis, :, :
        ]
        gs_new = wIs - alphas[-1]
        gs_new = (
            wIs
            - alphas[-2]
            - np.conj(betas[-2].T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, betas[-2][np.newaxis, :, :])
        )
        gs_prev = wIs - alphas[-2]
        for alpha, beta in zip(alphas[-3::-1], betas[-3::-1]):
            gs_new = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, beta[np.newaxis, :, :])
            gs_prev = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_prev, beta[np.newaxis, :, :])
        # print(rf"$\Delta$G = {np.max(np.abs(gs_new - gs_prev))}", flush=True)
        return np.all(np.abs(gs_new - gs_prev) < 1e-8)

    print("Get alpha and beta!", flush=True)
    # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
    alphas, betas, _ = get_block_Lanczos_matrices(
        psi0=psi0,
        h=h[:, basis.local_indices],
        converged=converged,
        h_local=h_local,
        verbose=verbose,
        reort_mode=reort,
        build_krylov_basis=False,
    )

    t0 = time.perf_counter()

    gs_matsubara, gs_realaxis = calc_mpi_Greens_function_from_alpha_beta(alphas, betas, iws, ws, e, delta, r, verbose)
    if rank == 0 and matsubara:
        gs_matsubara = np.moveaxis(gs_matsubara, 0, -1)
    if rank == 0 and realaxis:
        gs_realaxis = np.moveaxis(gs_realaxis, 0, -1)

    if verbose:
        print(f"time(G_from_alpha_beta) = {time.perf_counter() - t0: .4f} seconds.")

    return gs_matsubara, gs_realaxis


def block_Green(
    n_spin_orbitals,
    hOp,
    psi_arr,
    basis,
    e,
    iws,
    ws,
    delta,
    reort,
    h_mem=None,
    slaterWeightMin=0,
    verbose=True,
):
    """
    calculate  one block of the Greens function. This function builds the many body basis iteratively. Reducing memory requrements.
    """
    comm = basis.comm
    rank = comm.rank
    matsubara = iws is not None
    realaxis = ws is not None

    if not matsubara and not realaxis:
        if rank == 0:
            print("No Matsubara mesh or real frequency mesh provided. No Greens function will be calculated.")
        return None, None

    N = len(basis)
    n = len(psi_arr)

    if n == 0 or N == 0:
        return np.zeros((len(iws), n, n), dtype=complex), np.zeros((len(ws), n, n), dtype=complex)
    if verbose:
        t0 = time.perf_counter()
    psi_start = np.array(basis.build_distributed_vector(psi_arr).T, copy=False, order="C")
    counts = np.empty((comm.size,), dtype=int) if comm.rank == 0 else None
    comm.Gather(np.array([n * len(basis.local_basis)], dtype=int), counts, root=0)
    offsets = [sum(counts[:r]) for r in range(len(counts))] if comm.rank == 0 else None
    psi0 = np.empty((N, n), dtype=complex, order="C") if comm.rank == 0 else None
    comm.Gatherv(psi_start, (psi0, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
    r: Optional[np.ndarray] = None
    if comm.rank == 0:
        # Do a QR decomposition of the starting block.
        # Later on, use r to restore the block corresponding to
        psi0[:, :], r = sp.linalg.qr(psi0, mode="economic", overwrite_a=True, check_finite=False, pivoting=False)
        # Find which columns (if any) are 0 in psi0_0
        rows, columns = psi0.shape
    rows = comm.bcast(rows if comm.rank == 0 else None, root=0)
    columns = comm.bcast(columns if comm.rank == 0 else None, root=0)
    if rows == 0 or columns == 0:
        return np.zeros((len(iws), n, n), dtype=complex), np.zeros((len(ws), n, n), dtype=complex)
    if comm.rank != 0:
        psi0 = None
    comm.Scatterv((psi0, counts, offsets, MPI.C_DOUBLE_COMPLEX), psi_start, root=0)
    psi = [{} for _ in range(n)]
    for j, (i, state) in itertools.product(range(n), enumerate(basis.local_basis)):
        if abs(psi_start[i, j]) ** 2 > slaterWeightMin:
            psi[j][state] = psi_start[i, j]
    if verbose:
        print(f"time(set up psi_start) = {time.perf_counter() - t0}")

    # If we have a realaxis mesh, prefer to check convergence on that
    # if not, use the Matsubara mesh
    if realaxis:
        conv_w = ws
        delta_p = delta
    elif matsubara:
        conv_w = iws
        delta_p = 0

    # Select points from the frequency mesh, according to a Normal distribuition
    # centered on (value) 0.
    n_samples = max(len(conv_w) // 50, min(len(conv_w), 10))

    def converged(alphas, betas):
        if alphas.shape[0] == 1:
            return False

        w = np.random.choice(conv_w, size=max(n_samples // comm.size, 1), replace=False)
        wIs = (w + 1j * delta_p + e)[:, np.newaxis, np.newaxis] * np.identity(alphas.shape[1], dtype=complex)[
            np.newaxis, :, :
        ]
        gs_new = wIs - alphas[-1]
        gs_new = (
            wIs
            - alphas[-2]
            - np.conj(betas[-2].T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, betas[-2][np.newaxis, :, :])
        )
        gs_prev = wIs - alphas[-2]
        for alpha, beta in zip(alphas[-3::-1], betas[-3::-1]):
            gs_new = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, beta[np.newaxis, :, :])
            gs_prev = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_prev, beta[np.newaxis, :, :])
        return np.all(np.abs(gs_new - gs_prev) < 1e-8)

    t0 = time.perf_counter()
    # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
    alphas, betas, _ = block_lanczos(
        psi0=psi,
        h_op=hOp,
        basis=basis,
        converged=converged,
        h_mem=h_mem,
        verbose=verbose,
        slaterWeightMin=slaterWeightMin,
    )
    if verbose:
        print(f"time(block_lanczos) = {time.perf_counter() - t0: .4f} seconds.")

    t0 = time.perf_counter()

    gs_matsubara, gs_realaxis = calc_mpi_Greens_function_from_alpha_beta(
        alphas, betas, iws, ws, e, delta, r, verbose, comm=comm
    )

    if verbose:
        print(f"time(G_from_alpha_beta) = {time.perf_counter() - t0: .4f} seconds.")

    return gs_matsubara, gs_realaxis


def calc_mpi_Greens_function_from_alpha_beta(alphas, betas, iws, ws, e, delta, r, verbose, comm):
    """
    Calculate the Greens function from the diagonal and offdiagonal terms obtained from the Lanczos procedure.
    This function splits the frequency axes over MPI ranks.
    """
    matsubara = iws is not None
    realaxis = ws is not None
    if matsubara:
        num_indices = np.array([len(iws) // comm.size] * comm.size, dtype=int)
        num_indices[: len(iws) % comm.size] += 1
        iws_split = iws[sum(num_indices[: comm.rank]) : sum(num_indices[: comm.rank + 1])]
    if realaxis:
        num_indices = np.array([len(ws) // comm.size] * comm.size, dtype=int)
        num_indices[: len(ws) % comm.size] += 1
        ws_split = ws[sum(num_indices[: comm.rank]) : sum(num_indices[: comm.rank + 1])]
    gs_matsubara_local, gs_realaxis_local = calc_local_Greens_function_from_alpha_beta(
        alphas, betas, iws_split, ws_split, e, delta, verbose
    )
    # Multiply obtained Green's function with the upper triangular matrix to restore the original block
    # R^T* G R
    if matsubara:
        counts = np.empty((comm.size), dtype=int)
        comm.Gather(np.array([gs_matsubara_local.shape[1] ** 2 * len(iws_split)], dtype=int), counts)
        offsets = [sum(counts[:r]) for r in range(len(counts))] if comm.rank == 0 else None
        gs_matsubara = np.empty((len(iws), r.shape[0], r.shape[0]), dtype=complex) if comm.rank == 0 else None
        comm.Gatherv(gs_matsubara_local, (gs_matsubara, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
        if comm.rank == 0:
            gs_matsubara = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_matsubara, r[np.newaxis, :, :])
    if realaxis:
        counts = np.empty((comm.size), dtype=int)
        comm.Gather(np.array([gs_realaxis_local.shape[1] ** 2 * len(ws_split)], dtype=int), counts)
        offsets = [sum(counts[:r]) for r in range(len(counts))] if comm.rank == 0 else None
        gs_realaxis = np.empty((len(ws), r.shape[0], r.shape[0]), dtype=complex) if comm.rank == 0 else None
        comm.Gatherv(gs_realaxis_local, (gs_realaxis, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
        if comm.rank == 0:
            gs_realaxis = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_realaxis, r[np.newaxis, :, :])
    return gs_matsubara, gs_realaxis


def calc_local_Greens_function_from_alpha_beta(alphas, betas, iws, ws, e, delta, verbose):
    """
    Calculate the Greens function from alphas and betas, for all frequencies in iws and ws.
    """
    I = np.identity(alphas.shape[1], dtype=complex)
    matsubara = iws is not None
    realaxis = ws is not None
    if matsubara:
        iomegaP = iws + e
        gs_matsubara = np.zeros((len(iws), alphas.shape[1], alphas.shape[1]), dtype=complex)
        iwIs = iomegaP[:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
        gs_matsubara = iwIs - alphas[-1][np.newaxis, :, :]
    else:
        gs_matsubara = None
    if realaxis:
        omegaP = ws + 1j * delta + e
        gs_realaxis = np.zeros((len(ws), alphas.shape[1], alphas.shape[1]), dtype=complex)
        wIs = omegaP[:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
        gs_realaxis = wIs - alphas[-1][np.newaxis, :, :]
    else:
        gs_realaxis = None

    for alpha, beta in zip(alphas[-2::-1], betas[-2::-1]):
        if matsubara:
            gs_matsubara = (
                iwIs
                - alpha[np.newaxis, :, :]
                - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_matsubara, beta[np.newaxis, :, :])
            )
        if realaxis:
            gs_realaxis = (
                wIs
                - alpha[np.newaxis, :, :]
                - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_realaxis, beta[np.newaxis, :, :])
            )
    return gs_matsubara, gs_realaxis


def calc_Greens_function_with_offdiag_cg(
    n_spin_orbitals,
    hOp,
    tOps,
    psis,
    es,
    basis,
    iw,
    w,
    delta,
    reort,
    restrictions=None,
    blocks=None,
    krylovSize=None,
    slaterWeightMin=0,
    parallelization_mode="H_build",
    verbose=True,
    dense_cutoff=1e3,
    tau=0,
):
    """
    Use conjugate gradient method to calculate the interacting Greens function function with offdiagonal elements.
    Keep the manybody basis optimized for the excited state.
    """
    n = len(es)
    if iw is not None:
        gs_matsubara = np.zeros((n, len(tOps), len(tOps), len(iw)), dtype=complex)
    else:
        gs_matsubara = None
    if w is not None:
        gs_realaxis = np.zeros((n, len(tOps), len(tOps), len(w)), dtype=complex)
    else:
        gs_realaxis = None

    t_mems = [{} for _ in tOps]
    h_mem = {}

    local_excited_basis = set()
    for block in blocks:
        for i_tOp, tOp in [(orb, tOps[orb]) for orb in block]:
            for s in basis.local_basis:
                res = finite.applyOp_new(
                    n_spin_orbitals,
                    tOps[i_tOp],
                    {s: 1},
                    slaterWeightMin=0,  # slaterWeightMin,
                    restrictions=None,  # basis.restrictions,
                    opResult=t_mems[i_tOp],
                )
                local_excited_basis |= res.keys()
    excited_basis = CIPSI_Basis(
        impurity_orbitals=basis.impurity_orbitals,
        bath_states=basis.bath_states,
        initial_basis=local_excited_basis,
        restrictions=basis.restrictions,
        comm=basis.comm,
        verbose=verbose,
        truncation_threshold=basis.truncation_threshold,
        tau=basis.tau,
        spin_flip_dj=basis.spin_flip_dj,
    )
    for i, (psi, e) in enumerate(zip(psis, es)):
        for block in blocks:
            block_v = []
            local_excited_basis = set()
            t0 = time.perf_counter()
            for i_tOp, tOp in [(orb, tOps[orb]) for orb in block]:
                v = finite.applyOp_new(
                    n_spin_orbitals,
                    tOp,
                    {state: psi[state] for state in psi if state in basis.local_basis},
                    slaterWeightMin=slaterWeightMin,
                    restrictions=basis.restrictions,
                    opResult=t_mems[i_tOp],
                )
                vs = comm.allgather(v)
                v = {}
                for v_i in vs:
                    for state in v_i:
                        v[state] = v_i[state] + v.get(state, 0)
                block_v.append(v)
            if verbose:
                print(f"time(build excited state basis) = {time.perf_counter() - t0}")
            gs_matsubara_i, gs_realaxis_i = get_block_Green_cg(
                n_spin_orbitals=n_spin_orbitals,
                hOp=hOp,
                psi_arr=block_v,
                basis=excited_basis,
                e=e,
                iws=iw,
                ws=w,
                delta=delta,
                restrictions=restrictions,
                h_mem=h_mem,
                slaterWeightMin=slaterWeightMin,
                verbose=verbose,
                dense_cutoff=dense_cutoff,
            )
            if rank == 0:
                if iw is not None:
                    block_idx = np.ix_(block, block, range(gs_matsubara.shape[3]))
                    gs_matsubara[i][block_idx] = gs_matsubara_i
                if w is not None:
                    block_idx = np.ix_(block, block, range(gs_realaxis.shape[3]))
                    gs_realaxis[i][block_idx] = gs_realaxis_i
    return gs_matsubara, gs_realaxis


def get_block_Green_cg(
    n_spin_orbitals,
    hOp,
    psi_arr,
    basis,
    e,
    iws,
    ws,
    delta,
    restrictions=None,
    h_mem=None,
    slaterWeightMin=0,
    verbose=True,
    dense_cutoff=1e3,
):
    """
    Calculate one block of the Greens function using the conjugate gradient method.
    """
    matsubara = iws is not None
    realaxis = ws is not None

    if not matsubara and not realaxis:
        if rank == 0:
            print("No Matsubara mesh or real frequency mesh provided. No Greens function will be calculated.")
        return None, None

    if h_mem is None:
        h_mem = {}

    if verbose:
        t0 = time.perf_counter()
    h = basis.build_sparse_matrix(hOp, h_mem)

    if verbose:
        print(f"time(build Hamiltonian operator) = {time.perf_counter() - t0}")

    N = h.shape[0]
    n = len(psi_arr)

    if verbose:
        t0 = time.perf_counter()

    if verbose:
        print(f"time(set up psi_start) = {time.perf_counter() - t0}")

    if N == 0 or n == 0:
        return np.zeros((n, n, len(iws)), dtype=complex), np.zeros((n, n, len(ws)), dtype=complex)

    local_basis_lens = np.empty((comm.size), dtype=int)
    comm.Allgather(np.array([len(basis.local_basis)], dtype=int), local_basis_lens)
    if matsubara:
        gs_matsubara = np.zeros((len(iws), n, n), dtype=complex)
        local_basis = CIPSI_Basis(
            impurity_orbitals=basis.impurity_orbitals,
            bath_states=basis.bath_states,
            initial_basis=basis,
            restrictions=basis.restrictions,
            comm=None,
            verbose=verbose,
            truncation_threshold=basis.truncation_threshold,
            spin_flip_dj=basis.spin_flip_dj,
            tau=basis.tau,
        )
        for w_i, w in finite.get_job_tasks(comm.rank, comm.size, list(enumerate(iws))):
            shift = {((0, "i"),): w + e}
            A_op = finite.subtractOps(shift, hOp)
            A_dict = {}
            for col in range(n):
                tmp, info = cg_phys(A_op, A_dict, n_spin_orbitals, {}, psi_arr[col], 0, w.imag, local_basis)
                T_psi = local_basis.build_vector(psi_arr).T
                # T_psi = np.empty((len(T_psi_vs[0]), len(T_psi_vs)), dtype=T_psi_vs[0].dtype)
                # for col, v in enumerate(T_psi_vs):
                #     T_psi[:, col] = v
                gs_matsubara[w_i, :, col] = np.conj(T_psi.T) @ tmp
        comm.Allreduce(gs_matsubara.copy(), gs_matsubara, op=MPI.SUM)
        gs_matsubara = np.moveaxis(gs_matsubara, 0, -1)

    if realaxis:
        gs_realaxis = np.zeros((len(ws), n, n), dtype=complex)
        for w_i, w in finite.get_job_tasks(comm.rank, comm.size, list(enumerate(ws))):
            shift = {((0, "i"),): w + 1j * delta + e}
            A_op = finite.subtractOps(shift, hOp)
            for col in range(n):
                tmp, info = cg_phys(A_op, {}, n_spin_orbitals, {}, psi_arr[col], w, delta, local_basis)
                T_psi = local_basis.build_vector(psi_arr)
                gs_realaxis[w_i, :, col] = np.conj(T_psi.T) @ tmp
        comm.Allreduce(gs_realaxis.copy(), gs_realaxis, op=MPI.SUM)
        gs_realaxis = np.moveaxis(gs_realaxis, 0, -1)

    def matrix_print(m):
        print("\n".join(["  ".join([f"{np.real(el): 5.3f}  {np.imag(el):+5.3f}j" for el in row]) for row in m]))

    if verbose:
        print(f"time(G_cg) = {time.perf_counter() - t0: .4f} seconds.")

    return gs_matsubara, gs_realaxis


def rotate_matrix(M, T):
    """
    Rotate the matrix, M, using the matrix T.
    Returns M' = T^\dagger M T
    Parameters
    ==========
    M : NDArray - Matrix to rotate
    T : NDArray - Rotation matrix to use
    Returns
    =======
    M' : NDArray - The rotated matrix
    """
    return np.conj(T.T) @ M @ T


def rotate_Greens_function(G, T):
    """
    Rotate the Greens function, G, using the matrix T.
    Returns G'(\omega) = T^\dagger G(\omega) T
    Parameters
    ==========
    G : NDArray - Greens function to rotate
    T : NDArray - Rotation matrix to use
    Returns
    =======
    G' : NDArray - The rotated Greens function
    """
    return np.conj(T.T)[np.newaxis, :, :] @ G @ T[np.newaxis, :, :]


def rotate_4index_U(U4, T):
    """
    Rotate the four index tensor, U4, using the matrix T.
    Returns U4' = T^\daggerT^\dagger U4 TT
    Parameters
    ==========
    U4 : NDArray - Tensor function to rotate
    T : NDArray - Rotation matrix to use
    Returns
    =======
    U4' : NDArray - The rotated tensor function
    """
    return np.einsum("ij,kl, jlmo, mn, op", np.conj(T.T), np.conj(T.T), U4, T, T)


def save_Greens_function(gs, omega_mesh, label, e_scale=1, tol=1e-8):
    """
    Save Greens function to file, using RSPt .dat format. Including offdiagonal elements.
    """
    n_orb = gs.shape[1]
    axis_label = "realaxis"
    if np.all(np.abs(np.imag(omega_mesh)) > 1e-6):
        omega_mesh = np.imag(omega_mesh)
        axis_label = "Matsubara"

    off_diags = []
    for column in range(gs.shape[2]):
        for row in range(gs.shape[1]):
            if row == column:
                continue
            if np.any(np.abs(gs[:, row, column]) > tol):
                off_diags.append((row, column))

    print(f"Writing {axis_label} {label} to files")
    with open(f"real-{axis_label}-{label}.dat", "w") as fg_real, open(f"imag-{axis_label}-{label}.dat", "w") as fg_imag:
        header = "# Frequency, total, spin down, spin up\n"
        header += "# indexmap: (column index of projected elements)"
        for row in range(gs.shape[1]):
            header += "\n# "
            for column in range(gs.shape[2]):
                if row == column:
                    header += f"{5 + row:< 4d}"
                elif (row, column) in off_diags:
                    header += f"{5 + n_orb + off_diags.index((row, column)):< 4d}"
                else:
                    header += f"{0:< 4d}"
        fg_real.write(header + "\n")
        fg_imag.write(header + "\n")
        for i, w in enumerate(omega_mesh):
            fg_real.write(
                f"{w*e_scale} {np.real(np.sum(np.diag(gs[i, :, :])))} "
                + f"{np.real(np.sum(np.diag(gs[i, :n_orb//2, :n_orb//2])))} "
                + f"{np.real(np.sum(np.diag(gs[i, n_orb//2:, n_orb//2:])))} "
                + " ".join(f"{np.real(el)}" for el in np.diag(gs[i, :, :]))
                + " "
                + " ".join(f"{np.real(gs[i, row, column])}" for row, column in off_diags)
                + "\n"
            )
            fg_imag.write(
                f"{w*e_scale} {np.imag(np.sum(np.diag(gs[i, :, :])))} "
                + f"{np.imag(np.sum(np.diag(gs[i, :n_orb//2, :n_orb//2])))} "
                + f"{np.imag(np.sum(np.diag(gs[i, n_orb//2:, n_orb//2:])))} "
                + " ".join(f"{np.imag(el)}" for el in np.diag(gs[i, :, :]))
                + " "
                + " ".join(f"{np.imag(gs[i, row, column])}" for row, column in off_diags)
                + "\n"
            )
