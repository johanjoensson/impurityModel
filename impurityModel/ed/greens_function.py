import itertools
import numpy as np
import scipy as sp
import time
from typing import Optional, Iterable

from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.lanczos import get_block_Lanczos_matrices, block_lanczos
from impurityModel.ed.manybody_basis import CIPSI_Basis, Basis
from impurityModel.ed.cg import bicgstab

from mpi4py import MPI


def matrix_print(matrix: np.ndarray, label: str = None) -> None:
    """
    Pretty print the matrix, with optional label.
    """
    ms = "\n".join([" ".join([f"{np.real(val): .4f}{np.imag(val):+.4f}j" for val in row]) for row in matrix])
    if label is not None:
        print(label)
    print(ms)


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
    comm.Barrier()
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
        bath_states=basis.bath_states,
        initial_basis=(state for psi in psis for state in psi),
        restrictions=basis.restrictions,
        comm=split_comm,
        verbose=basis.verbose,
        truncation_threshold=basis.truncation_threshold,
        tau=basis.tau,
        spin_flip_dj=basis.spin_flip_dj,
    )
    psis = split_basis.redistribute_psis(psis)

    return slice(indices_start, indices_end), split_roots, color, items_per_color, split_basis, psis


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
    occ_restrict=True,
    chain_restrict=False,
    occ_cutoff=5e-2,
    slaterWeightMin=np.finfo(float).eps,
):
    """
    Calculate interacting Greens function.
    """
    (
        block_indices,
        block_roots,
        color,
        blocks_per_color,
        block_basis,
        psis,
    ) = split_comm_and_redistribute_basis([len(block) ** 2 for block in blocks], basis, psis)
    if verbose:
        print(f"New root ranks:{block_roots}")
        print(f"Number blocks per subgroup: {blocks_per_color}")
    bis = list(range(block_indices.start, block_indices.stop))
    gs_matsubara = [None for _ in blocks]
    gs_realaxis = [None for _ in blocks]
    for block_i, (opIPS, opPS) in enumerate(
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
            matsubara_mesh if matsubara_mesh is not None else None,
            omega_mesh if omega_mesh is not None else None,
            delta,
            reort=reort,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose,
            occ_restrict=occ_restrict,
            chain_restrict=chain_restrict,
            occ_cutoff=occ_cutoff,
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
            slaterWeightMin=slaterWeightMin,
            verbose=verbose,
            reort=reort,
            occ_restrict=occ_restrict,
            chain_restrict=chain_restrict,
            occ_cutoff=occ_cutoff,
        )

        if matsubara_mesh is not None and block_basis.comm.rank == 0:
            gs_matsubara[bis[block_i]] = gsIPS_matsubara - np.transpose(
                gsPS_matsubara,
                (
                    0,
                    2,
                    1,
                ),
            )
        if omega_mesh is not None and block_basis.comm.rank == 0:
            gs_realaxis[bis[block_i]] = gsIPS_realaxis - np.transpose(
                gsPS_realaxis,
                (
                    0,
                    2,
                    1,
                ),
            )
    all_gs_matsubara = basis.comm.gather(gs_matsubara, root=0)
    all_gs_realaxis = basis.comm.gather(gs_realaxis, root=0)
    if basis.comm.rank == 0:
        for gs_mats in all_gs_matsubara:
            for i, gm in enumerate(gs_mats):
                if gm is None:
                    continue
                gs_matsubara[i] = gm
        for gs_reals in all_gs_realaxis:
            for i, gr in enumerate(gs_reals):
                if gr is None:
                    continue
                gs_realaxis[i] = gr
        assert all(gs is not None for gs in gs_matsubara)
        assert all(gs is not None for gs in gs_realaxis)

    # gs_matsubara = (
    #     [np.empty((len(matsubara_mesh), len(block), len(block)), dtype=complex) for block in blocks]
    #     if basis.comm.rank == 0
    #     else None
    # )
    # gs_realaxis = (
    #     [np.empty((len(omega_mesh), len(block), len(block)), dtype=complex) for block in blocks]
    #     if basis.comm.rank == 0
    #     else None
    # )
    # # requests = []
    # if block_basis.comm.rank == 0 and basis.comm.rank != 0:
    #     assert basis.comm.rank in block_roots
    #     if matsubara_mesh is not None:
    #         for matsubara_block_gs in gs_matsubara_block:
    #             basis.comm.send(matsubara_block_gs, 0)
    #             # requests.append(basis.comm.isend(matsubara_block_gs, 0))
    #     if omega_mesh is not None:
    #     assert basis.comm.rank in block_roots
    #         for real_block_gs in gs_realaxis_block:
    #             basis.comm.isend(real_block_gs, 0)
    #             # requests.append(basis.comm.isend(real_block_gs, 0))
    # if basis.comm.rank == 0:
    #     for color, color_root in enumerate(block_roots):
    #         block_is = range(sum(blocks_per_color[:color]), sum(blocks_per_color[: color + 1]))
    #         if matsubara_mesh is not None:
    #             if color_root == block_basis.rank:
    #                 continue
    #             for block_i in block_is:
    #                 basis.comm.recv(gs_matsubara[block_i], color_root)
    #                 # requests.append(basis.comm.irecv(gs_matsubara[block_i], color_root))
    #         if omega_mesh is not None:
    #             for block_i in block_is:
    #                 basis.comm.recv(gs_realaxis[block_i], color_root)
    #                 # requests.append(basis.comm.irecv(gs_realaxis[block_i], color_root))
    # if len(requests) > 0:
    #     MPI.Request().waitall(requests)
    block_basis.comm.Free()
    return (gs_matsubara, gs_realaxis) if basis.comm.rank == 0 else (None, None)


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
    occ_restrict=True,
    chain_restrict=False,
    occ_cutoff=5e-2,
):
    r"""
        Return Green's function for states with low enough energy.

        For states :math:`|psi \rangle`, calculate:

        :math:`g(w+1j*delta) =
        = \langle psi| tOp^\dagger ((w+1j*delta+e)*\hat{1} - hOp)^{-1} tOp
        |psi \rangle`,

        where :math:`e = \langle psi| hOp |psi \rangle`
    ,
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

    # if chain_restrict:
    #     _, bath_rhos, bath_indices = basis.build_density_matrices(psis)
    #     thermal_bath_rhos = {
    #         i: [finite.thermal_average_scale_indep(es, block_rhos, tau) for block_rhos in bath_rhos[i]]
    #         for i in basis.impurity_orbitals.keys()
    #     }
    # else:
    #     thermal_bath_rhos = None
    #     bath_indices = None

    # if occ_restrict or chain_restrict:
    #     excited_restrictions = basis.build_excited_restrictions(
    #         bath_rhos=thermal_bath_rhos,
    #         bath_indices=bath_indices,
    #         imp_change=(1, 1),
    #         val_change=(1, 0),
    #         con_change=(0, 1),
    #         occ_cutoff=occ_cutoff,
    #     )
    # else:
    #     excited_restrictions = None

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
        color,
        eigen_per_color,
        eigen_basis,
        psis,
    ) = split_comm_and_redistribute_basis([1 for _ in es], basis, psis)
    if verbose:
        print(f"New root ranks for eigenstates:{eigen_roots}")
        print(f"Number of eigenstates per subgroup: {eigen_per_color}")
    e0 = min(es)
    Z = np.sum(np.exp(-(es - e0) / tau))
    for psi, e in zip(psis[eigen_indices], es[eigen_indices]):
        if chain_restrict:
            _, bath_rhos, bath_indices = eigen_basis.build_density_matrices([psi])
            bath_rhos = {i: [rho[0] for rho in br] for i, br in bath_rhos.items()}
        else:
            bath_rhos = None
            bath_indices = None

        if occ_restrict or chain_restrict:
            excited_restrictions = eigen_basis.build_excited_restrictions(
                bath_rhos=bath_rhos,
                bath_indices=bath_indices,
                imp_change=(1, 1),
                val_change=(1, 0),
                con_change=(0, 1),
                occ_cutoff=occ_cutoff,
            )
        else:
            excited_restrictions = None

        if verbose and excited_restrictions is not None:
            print("Excited state restrictions:")
            for indices, occupations in excited_restrictions.items():
                print(f"---> {sorted(indices)} : {occupations}")
            print()
        block_v = []
        local_excited_basis = set()
        # t0 = time.perf_counter()
        for i_tOp, tOp in enumerate(tOps):
            v = finite.applyOp_new(
                eigen_basis.num_spin_orbitals,
                tOp,
                psi,
                slaterWeightMin=0,
                restrictions=None,
                opResult=t_mems[i_tOp],
            )
            local_excited_basis |= v.keys()
            block_v.append(v)

        excited_basis = Basis(
            impurity_orbitals=eigen_basis.impurity_orbitals,
            bath_states=eigen_basis.bath_states,
            initial_basis=local_excited_basis,
            restrictions=excited_restrictions,
            comm=eigen_basis.comm.Clone(),
            verbose=verbose,
            truncation_threshold=eigen_basis.truncation_threshold,
            tau=eigen_basis.tau,
            spin_flip_dj=eigen_basis.spin_flip_dj,
        )

        # gs_matsubara_block_i, gs_realaxis_block_i = block_Green_freq(
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
        if eigen_basis.comm.rank == 0:
            if iw is not None:
                gs_matsubara_block += np.exp(-(e - e0) / tau) * gs_matsubara_block_i
            if w is not None:
                gs_realaxis_block += np.exp(-(e - e0) / tau) * gs_realaxis_block_i
        excited_basis.comm.Free()
    # Send calculated Greens functions to root
    basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == 0 else gs_matsubara_block, gs_matsubara_block, root=0)
    basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == 0 else gs_realaxis_block, gs_realaxis_block, root=0)
    # requests = []
    # if eigen_basis.comm.rank == 0:
    #     assert comm.rank in eigen_roots
    #     if iw is not None:
    #         requests.append(comm.isend(gs_matsubara_block, 0))
    #     if w is not None:
    #         requests.append(comm.isend(gs_realaxis_block, 0))
    # if comm.rank == 0:
    #     for i, r in enumerate(eigen_roots):
    #         if iw is not None:
    #             requests.append(comm.irecv(gs_matsubara_received[i], r))
    #     for i, r in enumerate(eigen_roots):
    #         if w is not None:
    #             requests.append(comm.irecv(gs_realaxis_received[i], r))
    #     MPI.Request().waitall(requests)
    #     if iw is not None:
    #         gs_matsubara_block = np.sum(gs_matsubara_received, axis=0)
    #     if w is not None:
    #         gs_realaxis_block = np.sum(gs_realaxis_received, axis=0)
    # if len(requests) > 0:
    #     MPI.Request().waitall(requests)
    eigen_basis.comm.Free()
    return gs_matsubara_block / Z, gs_realaxis_block / Z


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

    if verbose:
        t0 = time.perf_counter()
    h_mem = basis.expand(hOp, h_mem, slaterWeightMin=slaterWeightMin)
    h = basis.build_sparse_matrix(hOp, h_mem, petsc=False)

    if verbose:
        print(f"time(build Hamiltonian operator) = {time.perf_counter() - t0}")

    N = len(basis)
    n = len(psi_arr)

    if n == 0 or N == 0:
        return np.zeros((len(iws), n, n), dtype=complex), np.zeros((len(ws), n, n), dtype=complex)
    if verbose:
        t0 = time.perf_counter()
    psi0 = np.array(basis.build_distributed_vector(psi_arr).T, copy=False, order="C")
    counts = np.empty((comm.size,), dtype=int) if comm.rank == 0 else None
    comm.Gather(np.array([n * len(basis.local_basis)], dtype=int), counts, root=0)
    offsets = np.array([sum(counts[:r]) for r in range(len(counts))]) if comm.rank == 0 else None
    psi0_full = np.empty((N, n), dtype=complex, order="C") if comm.rank == 0 else None
    comm.Gatherv(psi0, (psi0_full, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
    r: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None
    if comm.rank == 0:
        # Do a QR decomposition of the starting block.
        # Later on, use r to restore the block corresponding to
        psi0_full, r, p = sp.linalg.qr(psi0_full, mode="economic", overwrite_a=True, check_finite=False, pivoting=True)
        psi0_full = np.array(psi0_full, copy=False, order="C")
        rows, columns = psi0_full.shape
    rows = comm.bcast(rows if comm.rank == 0 else None, root=0)
    columns = comm.bcast(columns if comm.rank == 0 else None, root=0)
    if rows == 0 or columns == 0:
        return np.zeros((len(iws), n, n), dtype=complex), np.zeros((len(ws), n, n), dtype=complex)

    comm.Gather(np.array([columns * len(basis.local_basis)], dtype=int), counts, root=0)
    comm.Scatterv(
        (psi0_full, counts * columns // n, offsets * columns // n, MPI.C_DOUBLE_COMPLEX) if comm.rank == 0 else None,
        psi0.T,
        root=0,
    )
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
    n_samples = max(len(conv_w) // 10, 1)

    def converged(alphas, betas):
        if np.any(np.linalg.norm(betas[-1], axis=1) < slaterWeightMin):
            return True

        if alphas.shape[0] == 1:
            return False
        w = np.zeros((n_samples), dtype=conv_w.dtype)
        intervals = np.linspace(start=conv_w[0], stop=conv_w[-1], num=n_samples + 1)
        for i in range(n_samples):
            w[i] = basis.rng.uniform(
                low=min(intervals[i], intervals[i + 1]), high=max(intervals[i], intervals[i + 1]), size=None
            )
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
        return np.all(np.abs(gs_new - gs_prev) < max(slaterWeightMin, 1e-8))

    # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
    alphas, betas, _ = get_block_Lanczos_matrices(
        psi0=psi0[:, :columns],
        h=h[:, basis.local_indices],
        converged=converged,
        verbose=verbose,
        reort_mode=reort,
        build_krylov_basis=False,
        comm=basis.comm,
    )

    t0 = time.perf_counter()

    gs_matsubara, gs_realaxis = calc_mpi_Greens_function_from_alpha_beta(
        alphas, betas, iws, ws, e, delta, r, p, verbose, comm=comm
    )
    if verbose:
        print(f"time(G_from_alpha_beta) = {time.perf_counter() - t0: .4f} seconds.")

    return gs_matsubara, gs_realaxis


def build_qrp(psi):
    # Do a QR decomposition of the starting block.
    # Later on, use r to restore the psi block
    # Allow for permutations of rows in psi as well
    psi, r, p = sp.linalg.qr(psi.copy(), mode="economic", overwrite_a=True, check_finite=False, pivoting=True)
    return np.ascontiguousarray(psi), r, p


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

    psi_dense = basis.build_vector(psi_arr, root=0).T
    if rank == 0:
        psi_dense, r, p = build_qrp(psi_dense)
    r = basis.comm.bcast(r if rank == 0 else None, root=0)
    p = basis.comm.bcast(p if rank == 0 else None, root=0)
    rows, columns = basis.comm.bcast(psi_dense.shape if rank == 0 else None, root=0)
    assert rows == basis.size
    psi_dense_local = np.empty((len(basis.local_basis), columns), dtype=complex, order="C")
    send_counts = np.empty((basis.comm.size), dtype=int) if rank == 0 else None
    basis.comm.Gather(np.array([psi_dense_local.size]), send_counts if rank == 0 else None)
    offsets = np.array([np.sum(send_counts[:r]) for r in range(comm.size)], dtype=int) if rank == 0 else None
    if rank == 0:
        print(f"{send_counts=} {offsets=}")
        print(f"{basis.size=} {columns=}")
    comm.Scatterv(
        [psi_dense, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None, psi_dense_local, root=0
    )
    psi = basis.build_state(psi_dense_local.T, slaterWeightMin=slaterWeightMin)

    if len(psi) == 0:
        return np.zeros((len(iws), n, n), dtype=complex), np.zeros((len(ws), n, n), dtype=complex)

    # If we have a realaxis mesh, prefer to check convergence on that
    # if not, use the Matsubara mesh
    if realaxis:
        conv_w = ws
    else:
        conv_w = np.linspace(start=-0.5, stop=0.5, num=501)

    n_samples = max(len(conv_w) // 20, min(len(conv_w), 10))

    def converged(alphas, betas, verbose=False):
        if alphas.shape[0] == 1:
            return False

        if np.any(np.abs(betas[-1]) > 1e6):
            return True
        if alphas.shape[0] % 10 != 0:
            return False

        w = np.zeros((n_samples), dtype=conv_w.dtype)
        intervals = np.linspace(start=conv_w[0], stop=conv_w[-1], num=n_samples + 1)
        for i in range(n_samples):
            w[i] = basis.rng.uniform(
                low=min(intervals[i], intervals[i + 1]), high=max(intervals[i], intervals[i + 1]), size=None
            )
        wIs = (w + 1j * delta + e)[:, np.newaxis, np.newaxis] * np.identity(alphas.shape[1], dtype=complex)[
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

        δ = np.max(np.abs(gs_new - gs_prev))
        if verbose:
            print(rf"{δ=}")
        return δ < max(slaterWeightMin, 1e-12)

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
        reort=reort,
    )
    if verbose:
        print(f"time(block_lanczos) = {time.perf_counter() - t0: .4f} seconds.")

    t0 = time.perf_counter()

    gs_matsubara, gs_realaxis = calc_mpi_Greens_function_from_alpha_beta(
        alphas, betas, iws, ws, e, delta, r, p, verbose, comm=comm
    )

    if verbose:
        print(f"time(G_from_alpha_beta) = {time.perf_counter() - t0: .4f} seconds.")

    return gs_matsubara, gs_realaxis


def block_Green_freq_2(
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
    matsubara = iws is not None
    realaxis = ws is not None

    if not matsubara and not realaxis:
        if basis.comm.rank == 0:
            print("No Matsubara mesh or real frequency mesh provided. No Greens function will be calculated.")
        return None, None

    N = len(basis)
    n = len(psi_arr)

    if n == 0 or N == 0:
        return np.zeros((len(iws), n, n), dtype=complex), np.zeros((len(ws), n, n), dtype=complex)
    if verbose:
        t0 = time.perf_counter()

    psi_orig, r, p = build_qrp(psi_arr, basis, slaterWeightMin)
    if len(psi_orig) == 0:
        return np.zeros((len(iws), n, n), dtype=complex), np.zeros((len(ws), n, n), dtype=complex)

    if verbose:
        print(f"time(set up psi_start) = {time.perf_counter() - t0}")

    def build_converged(w, delta):
        def converged(alphas, betas):
            if np.any(np.linalg.norm(betas[-1], axis=1) < max(slaterWeightMin, 1e-8)):
                return True

            if alphas.shape[0] == 1:
                return False

            gs_new = (w + 1j * delta + e) - alphas[-1]
            gs_new = (
                w
                + 1j * delta
                + e
                - alphas[-2]
                - np.conj(betas[-2].T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, betas[-2][np.newaxis, :, :])
            )
            gs_prev = (w + 1j * delta + e) - alphas[-2]
            for alpha, beta in zip(alphas[-3::-1], betas[-3::-1]):
                gs_new = (
                    (w + 1j * delta + e)
                    - alpha
                    - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, beta[np.newaxis, :, :])
                )
                gs_prev = (
                    (w + 1j * delta + e)
                    - alpha
                    - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_prev, beta[np.newaxis, :, :])
                )
            return np.all(np.abs(gs_new - gs_prev) < max(slaterWeightMin, 1e-6))

        return converged

    t0 = time.perf_counter()
    gs_matsubara = np.zeros((len(iws), len(psi_orig), len(psi_orig)), dtype=complex)
    gs_realaxis = np.zeros((len(ws), len(psi_orig), len(psi_orig)), dtype=complex)
    for w_mesh, gs in zip((iws, ws + 1j * delta), (gs_matsubara, gs_realaxis)):
        if w_mesh is None:
            continue
        _, freq_roots, color, _, split_basis, psi = split_comm_and_redistribute_basis(
            [1] * len(w_mesh), basis, psi_orig
        )
        w_indices = slice(color, len(w_mesh), len(freq_roots))
        freq_basis = CIPSI_Basis(
            impurity_orbitals=split_basis.impurity_orbitals,
            bath_states=split_basis.bath_states,
            initial_basis=split_basis.local_basis,
            restrictions=split_basis.restrictions,
            truncation_threshold=split_basis.truncation_threshold,
            spin_flip_dj=split_basis.spin_flip_dj,
            tau=split_basis.tau,
            verbose=verbose,
            comm=split_basis.comm.Clone(),
        )

        for w_i, w in itertools.islice(
            zip(range(len(w_mesh)), w_mesh), w_indices.start, w_indices.stop, w_indices.step
        ):

            freq_basis.expand_at(w + e, psi, hOp, H_dict=h_mem, de2_min=1e-12)

            if True:
                # Use fully sparse implementation
                # Build basis for each frequency
                h_local = freq_basis.build_sparse_matrix(hOp, h_mem, petsc=False)
                alphas, betas, _ = block_lanczos(
                    psi0=psi,
                    h_op=hOp,
                    basis=basis,
                    converged=build_converged(w, delta),
                    h_mem=h_mem,
                    verbose=False and verbose,
                    slaterWeightMin=slaterWeightMin,
                    reort=reort,
                )
            else:
                # Use build basis before building sparse matrix and ruhnning Lanczos
                h_local = freq_basis.build_sparse_matrix(hOp, h_mem, petsc=False)
                h = finite.create_linear_operator(h_local, freq_basis.comm)
                alphas, betas, _ = get_block_Lanczos_matrices(
                    psi0=freq_basis.build_vector(psi).T,
                    h=h,
                    converged=build_converged(w, delta),
                    verbose=False and verbose,
                    reort_mode=reort,
                    comm=freq_basis.comm,
                )
            if freq_basis.comm.rank == 0:
                gs[w_i, :, :] = (w + e) * np.eye(n) - alphas[-1]
                for alpha, beta in zip(alphas[-2::-1], betas[-2::-1]):
                    gs[w_i, :, :] = (w + e) * np.eye(n) - alpha - np.conj(beta.T) @ np.linalg.solve(gs[w_i], beta)
        freq_basis.comm.Free()
    basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == 0 else gs_matsubara, gs_matsubara, op=MPI.SUM)
    basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == 0 else gs_realaxis, gs_realaxis, op=MPI.SUM)
    if verbose:
        print(f"time(block_lanczos) = {time.perf_counter() - t0: .4f} seconds.")
    if basis.comm.rank == 0:
        ix = np.ix_(range(len(iws)), np.argsort(p), np.argsort(p))
        gs_matsubara = (np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_matsubara, r[np.newaxis, :, :]))[ix]
        ix = np.ix_(range(len(ws)), np.argsort(p), np.argsort(p))
        gs_realaxis = (np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_realaxis, r[np.newaxis, :, :]))[ix]

    return gs_matsubara, gs_realaxis


def block_Green_freq(
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

    if len(psi_arr) == 0 or len(basis) == 0:
        return np.zeros((len(iws), len(psi_arr), len(psi_arr)), dtype=complex), np.zeros(
            (len(ws), len(psi_arr), len(psi_arr)), dtype=complex
        )

    # psi_orig, r, p = build_qrp(psi_arr, basis, slaterWeightMin)
    psi_orig = psi_arr
    n_orb = len(psi_orig)

    if n_orb == 0:
        return np.zeros((len(iws), len(psi_arr), len(psi_arr)), dtype=complex), np.zeros(
            (len(ws), len(psi_arr), len(psi_arr)), dtype=complex
        )

    gs_matsubara = np.zeros((len(iws), n_orb, n_orb), dtype=complex)
    gs_realaxis = np.zeros((len(ws), n_orb, n_orb), dtype=complex)
    for w_mesh, gs in zip((iws, ws + 1j * delta), (gs_matsubara, gs_realaxis)):
        if w_mesh is None:
            continue
        _, freq_roots, color, _, split_basis, psi = split_comm_and_redistribute_basis(
            [1] * len(w_mesh), basis, psi_orig
        )
        w_indices = slice(color, len(w_mesh), len(freq_roots))
        freq_basis = CIPSI_Basis(
            impurity_orbitals=split_basis.impurity_orbitals,
            bath_states=split_basis.bath_states,
            initial_basis=split_basis.local_basis,
            restrictions=split_basis.restrictions,
            truncation_threshold=split_basis.truncation_threshold,
            spin_flip_dj=split_basis.spin_flip_dj,
            tau=split_basis.tau,
            verbose=verbose,
            comm=split_basis.comm.Clone(),
        )

        A_inv_psi = [{} for _ in psi]
        for w_i, w in itertools.islice(
            zip(range(len(w_mesh)), w_mesh), w_indices.start, w_indices.stop, w_indices.step
        ):

            freq_basis.expand_at(w + e, psi, hOp, H_dict=h_mem, de2_min=1e-12)
            A_op = finite.subtractOps({((0, "i"),): w + e}, hOp)
            if True:
                A_inv_psi = bicgstab(
                    A_op=A_op,
                    A_op_dict={},
                    x_0=A_inv_psi,
                    y=psi,
                    basis=freq_basis,
                    slaterWeightMin=0 * slaterWeightMin,
                    atol=max(np.sqrt(slaterWeightMin), 1e-5),
                )
                for (i, psi_i), (j, Ainvpsi_j) in itertools.product(enumerate(psi), enumerate(A_inv_psi)):
                    gs[w_i, i, j] = finite.inner(psi_i, Ainvpsi_j)
            else:
                A = freq_basis.build_sparse_matrix(A_op, petsc=False)
                A = finite.create_linear_operator(A, freq_basis.comm)
                psi_i = freq_basis.build_vector(psi)
                A_inv_psi_v = freq_basis.build_vector(A_inv_psi).T

                for j, psi_j in enumerate(psi_i):
                    if np.linalg.norm(psi_j) < np.finfo(float).eps:
                        continue
                    info = -1

                    while info != 0:
                        A_inv_psi_v[:, j], info = sp.sparse.linalg.gmres(
                            A, psi_j, x0=A_inv_psi_v[:, j], atol=np.sqrt(slaterWeightMin)
                        )
                        if info < 0:
                            raise RuntimeError("Parameter breakdown in bicgstab!")
                gs[w_i, :, :] = np.conj(psi_i) @ A_inv_psi_v
                A_inv_psi = freq_basis.build_state(A_inv_psi_v.T)
        freq_basis.comm.Free()

    basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == 0 else gs_matsubara, gs_matsubara, op=MPI.SUM, root=0)
    basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == 0 else gs_realaxis, gs_realaxis, op=MPI.SUM, root=0)

    return gs_matsubara, gs_realaxis


def calc_mpi_Greens_function_from_alpha_beta(alphas, betas, iws, ws, e, delta, r, p, verbose, comm):
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
        ix = np.ix_(range(len(iws_split)), p, p)
        gs_matsubara_local = (
            np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_matsubara_local, r[np.newaxis, :, :])
        )[ix]
        counts = np.empty((comm.size), dtype=int)
        comm.Gather(np.array([gs_matsubara_local.size], dtype=int), counts)
        offsets = [sum(counts[:rank]) for rank in range(len(counts))] if comm.rank == 0 else None
        gs_matsubara = (
            np.empty((len(iws), r.shape[1], r.shape[1]), dtype=complex, order="C") if comm.rank == 0 else None
        )
        comm.Gatherv(
            np.ascontiguousarray(gs_matsubara_local), (gs_matsubara, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0
        )
    else:
        gs_matsubara = None
    if realaxis:
        ix = np.ix_(range(len(ws_split)), p, p)
        gs_realaxis_local = (np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_realaxis_local, r[np.newaxis, :, :]))[
            ix
        ]
        counts = np.empty((comm.size), dtype=int)
        comm.Gather(np.array([gs_realaxis_local.size], dtype=int), counts)
        offsets = [sum(counts[:rank]) for rank in range(len(counts))] if comm.rank == 0 else None
        gs_realaxis = np.empty((len(ws), r.shape[1], r.shape[1]), dtype=complex, order="C") if comm.rank == 0 else None
        comm.Gatherv(
            np.ascontiguousarray(gs_realaxis_local), (gs_realaxis, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0
        )
    else:
        gs_realaxis = None
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
                    slaterWeightMin=slaterWeightMin,
                    restrictions=basis.restrictions,
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
    comm = basis.comm

    if not matsubara and not realaxis:
        if comm.rank == 0:
            print("No Matsubara mesh or real frequency mesh provided. No Greens function will be calculated.")
        return None, None

    if h_mem is None:
        h_mem = {}

    if verbose:
        t0 = time.perf_counter()
    # h = basis.build_sparse_matrix(hOp, h_mem, petsc=False)

    if verbose:
        print(f"time(build Hamiltonian operator) = {time.perf_counter() - t0}")

    # N = h.shape[0]
    n = len(psi_arr)

    if verbose:
        t0 = time.perf_counter()

    if verbose:
        print(f"time(set up psi_start) = {time.perf_counter() - t0}")

    # if N == 0 or n == 0:
    #     return np.zeros((n, n, len(iws)), dtype=complex), np.zeros((n, n, len(ws)), dtype=complex)

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
        comm.Allreduce(MPI.IN_PLACE, gs_matsubara, op=MPI.SUM)

        if gs_matsubara is not None:
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
        comm.Allreduce(MPI.IN_PLACE, gs_realaxis, op=MPI.SUM)
        if gs_realaxis is not None:
            gs_realaxis = np.moveaxis(gs_realaxis, 0, -1)

    def matrix_print(m):
        print("\n".join(["  ".join([f"{np.real(el): 5.3f}  {np.imag(el):+5.3f}j" for el in row]) for row in m]))

    if verbose:
        print(f"time(G_cg) = {time.perf_counter() - t0: .4f} seconds.")

    return gs_matsubara, gs_realaxis


def rotate_matrix(M, T):
    r"""
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


def block_diagonalize_hyb(hyb):
    hyb_herm = 1 / 2 * (hyb + np.conj(np.transpose(hyb, (0, 2, 1))))
    blocks = get_block_structure(hyb_herm)
    Q_full = np.zeros((hyb.shape[1], hyb.shape[2]), dtype=complex)
    treated_orbitals = 0
    for block in blocks:
        block_idx = np.ix_(range(hyb.shape[0]), block, block)
        if len(block) == 1:
            Q_full[block_idx[1:], treated_orbitals] = 1
            treated_orbitals += 1
            continue
        block_hyb = hyb_herm[block_idx]
        upper_triangular_hyb = np.triu(hyb_herm, k=1)
        ind_max_offdiag = np.unravel_index(np.argmax(np.abs(upper_triangular_hyb)), upper_triangular_hyb.shape)
        eigvals, Q = np.linalg.eigh(block_hyb[ind_max_offdiag[0], :, :])
        sorted_indices = np.argsort(eigvals)
        Q = Q[:, sorted_indices]
        for column in range(Q.shape[1]):
            j = np.argmax(np.abs(Q[:, column]))
            Q_full[block, treated_orbitals + column] = Q[:, column] * abs(Q[j, column]) / Q[j, column]
        treated_orbitals += Q.shape[1]
    phase_hyb = np.conj(Q_full.T)[np.newaxis, :, :] @ hyb @ Q_full[np.newaxis, :, :]
    return phase_hyb, Q_full


def get_block_structure(hyb: np.ndarray, hamiltonian=None, tol=1e-6):
    # Extract matrix elements with nonzero hybridization function
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    mask = np.logical_or(np.any(np.abs(hyb) > tol, axis=0), np.abs(hamiltonian) > tol)

    # Use the extracted mask to extract blocks

    n_blocks, block_idxs = sp.sparse.csgraph.connected_components(
        csgraph=sp.sparse.csr_matrix(mask), directed=False, return_labels=True
    )

    blocks = [[] for _ in range(n_blocks)]
    for orb_i, block_i in enumerate(block_idxs):
        blocks[block_i].append(orb_i)

    return blocks


def get_identical_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    identical_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in identical_blocks]):
            continue
        identical = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in identical_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if np.all(np.abs(hyb[idx_i] - hyb[idx_j]) < tol) and np.all(
                np.abs(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]]) < tol
            ):
                identical.append(j)
        identical_blocks[i] = identical
    return identical_blocks


def get_transposed_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    transposed_blocks = [[] for _ in blocks]
    for i, block_i in enumerate(blocks):
        if len(block_i) == 1 or np.any([i in b for b in transposed_blocks]):
            continue
        transposed = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i + 1 :]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp + 1
            if any(j in b for b in transposed_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if np.all(np.abs(hyb[idx_i] - np.transpose(hyb[idx_j], (0, 2, 1))) < tol) and np.all(
                np.abs(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]].T) < tol
            ):
                transposed.append(j)
        transposed_blocks.append(transposed)
    return transposed_blocks


def get_particle_hole_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[0], hyb.shape[1]))
    particle_hole_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in particle_hole_blocks]):
            continue
        particle_hole = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in particle_hole_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if (
                np.all(np.abs(np.real(hyb[idx_i] + hyb[idx_j])) < tol)
                and np.all(np.abs(np.imag(hyb[idx_i] - hyb[idx_j])) < tol)
                and np.all(np.abs(np.real(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]])) < tol)
                and np.all(np.abs(np.imag(hamiltonian[idx_i[1:]] - hamiltonian[idx_j[1:]])) < tol)
            ):
                particle_hole.append(j)
        particle_hole_blocks.append(particle_hole)
    return particle_hole_blocks


def get_particle_hole_and_transpose_blocks(blocks, hyb, hamiltonian=None, tol=1e-6):
    if hamiltonian is None:
        hamiltonian = np.zeros((hyb.shape[1], hyb.shape[2]))
    patricle_hole_and_transpose_blocks = []
    for i, block_i in enumerate(blocks):
        if np.any([i in b for b in patricle_hole_and_transpose_blocks]):
            continue
        patricle_hole_and_transpose = []
        idx_i = np.ix_(range(hyb.shape[0]), block_i, block_i)
        for jp, block_j in enumerate(blocks[i:]):
            if len(block_i) != len(block_j):
                continue
            j = i + jp
            if any(j in b for b in patricle_hole_and_transpose_blocks):
                continue
            idx_j = np.ix_(range(hyb.shape[0]), block_j, block_j)
            if np.all(np.abs(hyb[idx_i] + np.transpose(hyb[idx_j], (0, 2, 1))) < tol) and np.all(
                np.abs(hamiltonian[idx_i[1:]] + hamiltonian[idx_j[1:]].T) < tol
            ):
                patricle_hole_and_transpose.append(j)
        patricle_hole_and_transpose_blocks.append(patricle_hole_and_transpose)
    return patricle_hole_and_transpose_blocks


def rotate_Greens_function(G, T):
    r"""
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
    r"""
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
