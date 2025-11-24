import itertools
import numpy as np
import scipy as sp
import time
from typing import Optional, Iterable

# from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.lanczos import (
    block_lanczos,
    block_lanczos_sparse,
    get_block_Lanczos_matrices,
    get_block_Lanczos_matrices_dense,
    get_Lanczos_vectors,
)
from impurityModel.ed.manybody_basis import CIPSI_Basis, Basis
from impurityModel.ed.cg import bicgstab
from impurityModel.ed.block_structure import BlockStructure, get_blocks
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, applyOp as applyOp_test

from mpi4py import MPI


def build_full_greens_function(block_gf, block_structure: BlockStructure):
    (
        blocks,
        identical_blocks,
        transposed_blocks,
        particle_hole_blocks,
        particle_hole_transposed_blocks,
        inequivalent_blocks,
    ) = block_structure
    n_orb = sum(len(block) for block in block_structure.blocks)
    if len(block_gf[0].shape) == 2:
        res = np.zeros((n_orb, n_orb), dtype=block_gf[0].dtype)
    elif len(block_gf[0].shape) == 3:
        res = np.zeros((block_gf[0].shape[0], n_orb, n_orb), dtype=block_gf[0].dtype)
    else:
        raise RuntimeError(
            f"Unknown data shape {block_gf[0].shape}. Should be 3 index (n_freq, n_orb,n_orb) or 2 index (n_orb,n_orb)"
        )
    if len(block_gf) == len(inequivalent_blocks):
        # block_gf contains only symmetrically inequivalent blocks
        for inequiv_i, gf_i in enumerate(block_gf):
            for block_i in identical_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                res[block_idx] = gf_i
            for block_i in transposed_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                res[block_idx] = np.transpose(gf_i, (0, 2, 1))
            for block_i in particle_hole_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                res[block_idx] = -np.conj(gf_i)
            for block_i in particle_hole_transposed_blocks[inequivalent_blocks[inequiv_i]]:
                if len(gf_i.shape) == 2:
                    block_idx = np.ix_(blocks[block_i], blocks[block_i])
                elif len(gf_i.shape) == 3:
                    block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
                res[block_idx] = -np.transpose(np.conj(gf_i), (0, 2, 1))
    elif len(block_gf) == len(blocks):
        # block_gf contains all blocks
        for block_i, gf_i in enumerate(block_gf):
            if len(gf_i.shape) == 2:
                block_idx = np.ix_(blocks[block_i], blocks[block_i])
            elif len(gf_i.shape) == 3:
                block_idx = np.ix_(range(gf_i.shape[0]), blocks[block_i], blocks[block_i])
            res[block_idx] = gf_i
    else:
        raise RuntimeError(f"Block structure does not match block_gf.\n{block_structure=} {len(block_gf)=}")
    return res


def matrix_print(matrix: np.ndarray, label: str = None) -> None:
    """
    Pretty print the matrix, with optional label.
    """
    ms = "\n".join([" ".join([f"{np.real(val): .4f}{np.imag(val):+.4f}j" for val in row]) for row in matrix])
    if label is not None:
        print(label)
    print(ms)


def split_comm_and_redistribute_psi(priorities: Iterable[float], psis: list[ManyBodyState], comm):
    """
    Split MPI communicator in order to divide MPI ranks among items, number of ranks per item is determined using the priorities.
    Higher priority means more MPI ranks assigned to the element.
    """
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
    proc_cutoffs = np.cumsum(procs_per_color)
    color = np.argmax(comm.rank < proc_cutoffs)
    split_comm = comm.Split(color=color, key=0)
    split_roots = [0] + proc_cutoffs[:-1].tolist()
    items_per_color = np.array([len(priorities) // n_colors] * n_colors, dtype=int)
    items_per_color[: len(priorities) % n_colors] += 1
    indices_start = sum(items_per_color[:color])
    indices_end = sum(items_per_color[: color + 1])

    new_psis = [p.copy() for p in psis]
    for c, c_root in enumerate(split_roots):
        if color != c:
            comm.send(psis, dest=c_root + (comm.rank % procs_per_color[c]))
            # comm.send([p.to_dict() for p in psis], dest=c_root + (comm.rank % procs_per_color[c]))
        else:
            for sender in range(comm.size):
                if (
                    sum(procs_per_color[:c]) <= sender < sum(procs_per_color[: c + 1])
                    or sender % procs_per_color[c] != split_comm.rank
                ):
                    continue
                received_psis = comm.recv(source=sender)
                for i, received_psi in enumerate(received_psis):
                    new_psis[i] += ManyBodyState(received_psi)

    return slice(indices_start, indices_end), split_roots, color, items_per_color, split_comm, psis


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
    verbose_extra,
    reort,
    dN,
    occ_cutoff,
    slaterWeightMin,
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
        block_intercomms,
    ) = basis.split_basis_and_redistribute_psi([len(block) ** 3 for block in blocks], psis)
    if verbose:
        print(f"New block roots: {block_roots}")
        print(f"Blocks per color: {blocks_per_color}")
        print("=" * 80)
    if basis.comm.rank == 0:
        indices_for_colors = np.empty((sum(blocks_per_color)), dtype=int)
        offsets = np.array([sum(blocks_per_color[:r]) for r in range(len(block_roots))], dtype=int)
        for col, sender in enumerate(block_roots):
            if sender == 0:
                indices_for_colors[offsets[color] : offsets[color] + blocks_per_color[color]] = block_indices
                continue
            basis.comm.Recv(indices_for_colors[offsets[col] : offsets[col] + blocks_per_color[col]], source=sender)
    elif block_basis.comm.rank == 0:
        basis.comm.Send(np.array(block_indices), dest=0)

    bis = block_indices  # list(range(block_indices.start, block_indices.stop))
    excited_basis_sizes_IPS = np.empty((len(blocks), len(es)), dtype=int)
    excited_basis_sizes_PS = np.empty((len(blocks), len(es)), dtype=int)
    local_excited_basis_sizes_IPS = []
    local_excited_basis_sizes_PS = []
    local_gs_matsubara = []
    local_gs_realaxis = []
    for block_i, (opIPS, opPS) in enumerate(
        (
            [{((orb, "c"),): 1} for orb in block],
            [{((orb, "a"),): 1} for orb in block],
        )
        for block in (blocks[bi] for bi in block_indices)
    ):
        gsIPS_matsubara, gsIPS_realaxis, excited_basis_sizes = calc_Greens_function_with_offdiag(
            hOp,
            [ManyBodyOperator(o) for o in opIPS],
            psis,
            es,
            tau,
            block_basis,
            matsubara_mesh if matsubara_mesh is not None else None,
            omega_mesh if omega_mesh is not None else None,
            delta,
            reort=reort,
            dN_imp={i: (dN, dN) for i in basis.impurity_orbitals} if dN is not None else None,
            dN_val={i: (dN, dN) for i in basis.impurity_orbitals} if dN is not None else None,
            dN_con={i: (dN, dN) for i in basis.impurity_orbitals} if dN is not None else None,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose_extra,
            occ_cutoff=occ_cutoff,
        )
        local_excited_basis_sizes_IPS.append(excited_basis_sizes)
        gsPS_matsubara, gsPS_realaxis, excited_basis_sizes = calc_Greens_function_with_offdiag(
            hOp,
            [ManyBodyOperator(o) for o in opPS],
            psis,
            es,
            tau,
            block_basis,
            -matsubara_mesh if matsubara_mesh is not None else None,
            -omega_mesh if omega_mesh is not None else None,
            -delta,
            dN_imp={i: (dN, 0) for i in basis.impurity_orbitals} if dN is not None else None,
            dN_val={i: (dN, 0) for i in basis.impurity_orbitals} if dN is not None else None,
            dN_con={i: (0, dN) for i in basis.impurity_orbitals} if dN is not None else None,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose_extra,
            reort=reort,
            occ_cutoff=occ_cutoff,
        )
        local_excited_basis_sizes_PS.append(excited_basis_sizes)

        if matsubara_mesh is not None and block_basis.comm.rank == 0:
            local_gs_matsubara.append(
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
            local_gs_realaxis.append(
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
    if basis.comm.rank == 0:
        gs_matsubara = [None for _ in blocks]
        gs_realaxis = [None for _ in blocks]
        for i, block_i in enumerate(block_indices):
            gs_matsubara[block_i] = local_gs_matsubara[i]
            gs_realaxis[block_i] = local_gs_realaxis[i]
            excited_basis_sizes_IPS[block_i] = np.array(local_excited_basis_sizes_IPS[i])
            excited_basis_sizes_PS[block_i] = np.array(local_excited_basis_sizes_PS[i])
        for col, sender in enumerate(block_roots):
            if sender == 0:
                continue
            for block_idx in indices_for_colors[offsets[col] : offsets[col] + blocks_per_color[col]]:
                block = blocks[block_idx]

                gs_matsubara[block_idx] = np.empty((len(matsubara_mesh), len(block), len(block)), dtype=complex)
                basis.comm.Recv(gs_matsubara[block_idx], source=sender)

                gs_realaxis[block_idx] = np.empty((len(omega_mesh), len(block), len(block)), dtype=complex)
                basis.comm.Recv(gs_realaxis[block_idx], source=sender)

                basis.comm.Recv(excited_basis_sizes_IPS[block_idx], source=sender)
                basis.comm.Recv(excited_basis_sizes_PS[block_idx], source=sender)
    elif block_basis.comm.rank == 0:
        for gsm, gsr in zip(local_gs_matsubara, local_gs_realaxis):
            basis.comm.Send(gsm, dest=0)
            basis.comm.Send(gsr, dest=0)
            basis.comm.Send(np.array(local_excited_basis_sizes_IPS, dtype=int), dest=0)
            basis.comm.Send(np.array(local_excited_basis_sizes_PS, dtype=int), dest=0)
    basis.comm.Bcast(excited_basis_sizes_IPS, root=0)
    basis.comm.Bcast(excited_basis_sizes_PS, root=0)

    if verbose:
        print("=" * 80)
        print("Electron addition")
        for block_i, ebs in enumerate(excited_basis_sizes_IPS):
            print(f"   inequivalen  block {block_i}:")
            for ei, eb in enumerate(ebs):
                print(f"   ---> Excited basis for eigenstate {ei} contains {eb} states")
        print("Electron removal")
        for block_i, ebs in enumerate(excited_basis_sizes_PS):
            print(f"   inequivalen  block {block_i}:")
            for ei, eb in enumerate(ebs):
                print(f"   ---> Excited basis for eigenstate {ei} contains {eb} states")
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
    dN_imp: Optional[dict[int, tuple[int, int]]],
    dN_val: Optional[dict[int, tuple[int, int]]],
    dN_con: Optional[dict[int, tuple[int, int]]],
    slaterWeightMin: float,
    verbose: bool,
    occ_cutoff: float,
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
    n = len(tOps)

    excited_basis_sizes = np.zeros((len(es)), dtype=int)
    if iw is not None:
        gs_matsubara_block = np.zeros((len(iw), n, n), dtype=complex, order="C")
    else:
        gs_matsubara_block = None
    if w is not None:
        gs_realaxis_block = np.zeros((len(w), n, n), dtype=complex, order="C")
    else:
        gs_realaxis_block = None

    (
        eigen_indices,
        eigen_roots,
        color,
        eigen_per_color,
        eigen_basis,
        psis,
        eigen_intercomms,
    ) = basis.split_basis_and_redistribute_psi([1 for _ in es], psis)
    if verbose:
        print(f"New eigenstate roots: {eigen_roots}")
        print(f"Eigenstates  per color: {eigen_per_color}")
        print("=" * 80)
    e0 = min(es)
    Z = np.sum(np.exp(-(es - e0) / tau))

    for ei, psi, e in zip(eigen_indices, (psis[ei] for ei in eigen_indices), (es[ei] for ei in eigen_indices)):
        excited_restrictions = eigen_basis.build_excited_restrictions(
            psis=psis,
            imp_change=dN_imp,
            val_change=dN_val,
            con_change=dN_con,
            occ_cutoff=occ_cutoff,
        )
        excited_basis = Basis(
            eigen_basis.impurity_orbitals,
            eigen_basis.bath_states,
            initial_basis=[],
            restrictions=excited_restrictions,
            comm=eigen_basis.comm.Clone(),
            verbose=verbose,
            truncation_threshold=eigen_basis.truncation_threshold,
            tau=eigen_basis.tau,
            spin_flip_dj=eigen_basis.spin_flip_dj,
        )
        if verbose and excited_restrictions is not None:
            print("Excited state restrictions:")
            for indices, occupations in excited_restrictions.items():
                print(f"---> {sorted(indices)} : {occupations}", flush=True)

        block_v = []
        local_excited_basis = set()
        for i_tOp, tOp in enumerate(tOps):
            for state in eigen_basis.local_basis:
                v = applyOp_test(tOp, ManyBodyState({state: 1.0}), cutoff=0, restrictions=None)
                local_excited_basis |= set(v.keys())
            v = applyOp_test(
                tOp,
                psi,
                cutoff=0,
                restrictions=None,
            )
            # local_excited_basis |= set(v.keys())
            block_v.append(v)

        excited_basis.add_states(local_excited_basis)
        block_v = excited_basis.redistribute_psis(block_v)

        # gs_matsubara_block_i, gs_realaxis_block_i = block_Green_freq(
        gs_matsubara_block_i, gs_realaxis_block_i = block_Green(
            hOp=hOp,
            psi_arr=block_v,
            basis=excited_basis,
            e=e,
            iws=iw,
            ws=w,
            delta=delta,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose,
            reort=reort,
        )
        if eigen_basis.comm.rank == 0:
            if iw is not None:
                gs_matsubara_block += np.exp(-(e - e0) / tau) * gs_matsubara_block_i
            if w is not None:
                gs_realaxis_block += np.exp(-(e - e0) / tau) * gs_realaxis_block_i
        excited_basis_sizes[ei] = excited_basis.size
        hOp.clear_memory()

    # Send calculated Greens functions to root
    if basis.comm.rank == 0:
        for sender in eigen_roots:
            if sender == 0:
                continue
            if iw is not None:
                gs_iw_tmp = np.empty_like(gs_matsubara_block)
                basis.comm.Recv(gs_iw_tmp, source=sender)
                gs_matsubara_block += gs_iw_tmp
            if w is not None:
                gs_w_tmp = np.empty_like(gs_realaxis_block)
                basis.comm.Recv(gs_w_tmp, source=sender)
                gs_realaxis_block += gs_w_tmp
    elif eigen_basis.comm.rank == 0:
        if iw is not None:
            basis.comm.Send(gs_matsubara_block, dest=0)
        if w is not None:
            basis.comm.Send(gs_realaxis_block, dest=0)
    if iw is not None:
        gs_matsubara_block /= Z
    if w is not None:
        gs_realaxis_block /= Z
    basis.comm.Allreduce(MPI.IN_PLACE, excited_basis_sizes, op=MPI.MAX)
    return gs_matsubara_block, gs_realaxis_block, excited_basis_sizes


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
    # h_mem = basis.expand(hOp, slaterWeightMin=slaterWeightMin)
    h = basis.build_sparse_matrix(hOp, h_mem)

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
        alphas, betas, iws, ws, e, delta, r, verbose, comm=comm
    )

    # comm.barrier()
    return gs_matsubara, gs_realaxis


def build_qr(psi):
    # Do a QR decomposition of the starting block.
    # Later on, use r to restore the psi block
    # Allow for permutations of rows in psi as well
    psi, r = sp.linalg.qr(psi.copy(), mode="economic", overwrite_a=True, check_finite=False, pivoting=False)
    return np.ascontiguousarray(psi), r


def block_Green(
    hOp,
    psi_arr,
    basis,
    e,
    iws,
    ws,
    delta,
    reort,
    slaterWeightMin=0,
    verbose=True,
):
    """
    calculate  one block of the Greens function. This function builds the many body basis iteratively. Reducing memory requrements.
    """
    comm = basis.comm
    rank = comm.rank if comm is not None else 0
    matsubara = iws is not None
    realaxis = ws is not None
    if not realaxis:
        ws = np.linspace(-0.5, 0.5, num=int(2 / delta))

    if not matsubara and not realaxis:
        if rank == 0:
            print("No Matsubara mesh or real frequency mesh provided. No Greens function will be calculated.")
        return None, None

    N = len(basis)
    n = len(psi_arr)

    if n == 0 or N == 0:
        return np.zeros((len(iws), n, n), dtype=complex) if matsubara else None, (
            np.zeros((len(ws), n, n), dtype=complex) if realaxis else None
        )

    impurity_orbitals = basis.impurity_orbitals
    bath_states = basis.bath_states

    basis.expand(hOp, slaterWeightMin=slaterWeightMin, max_it=1)
    psi_arr = basis.redistribute_psis(psi_arr)
    # Calculate initial guess for Green's function
    gs_matsubara, gs_realaxis, last_state = block_green_impl(
        basis, hOp, psi_arr, iws, ws, e, delta, slaterWeightMin, verbose
    )
    done = False
    causal = False
    cutoff = slaterWeightMin
    while not done:
        old_size = basis.size
        # Add states connected to the first and last Krylov vector(s)
        new_states = set()
        for psi in itertools.chain(psi_arr, last_state):
            Hpsi = applyOp_test(hOp, psi, restrictions=basis.restrictions, cutoff=cutoff)
            new_states |= set(Hpsi.keys())
        basis.add_states(new_states)
        if basis.size == old_size:
            break
        while basis.size > basis.truncation_threshold:
            cutoff = max(10 * cutoff, np.finfo(float).eps)
            last_state = basis.redistribute_psis(last_state)
            basis.clear()
            new_states = set()
            for psi in last_state:
                Hpsi = applyOp_test(hOp, psi, restrictions=basis.restrictions, cutoff=cutoff)
                new_states |= set(Hpsi.keys())
            basis.add_states(new_states)
        if verbose:
            print(f"Expanded basis contains {basis.size} states")
        psi_arr = basis.redistribute_psis(psi_arr)
        gs_realaxis_prev = gs_realaxis
        gs_matsubara, gs_realaxis, last_state = block_green_impl(
            basis, hOp, psi_arr, iws, ws, e, delta, slaterWeightMin, verbose
        )
        if rank == 0:
            done = np.max(np.abs(gs_realaxis - gs_realaxis_prev)) < 1e-6
            causal = np.all(np.diagonal(gs_realaxis, axis1=1, axis2=2).imag) < 0
            done = done and causal
        if comm is not None:
            done = comm.bcast(done, root=0)
            causal = comm.bcast(causal, root=0)

    return gs_matsubara, gs_realaxis


def block_green_impl(basis, hOp, psi_arr, iws, ws, e, delta, slaterWeightMin, verbose):
    comm = basis.comm
    rank = comm.rank if comm is not None else 0
    matsubara = iws is not None
    realaxis = ws is not None
    N = len(basis)
    n = len(psi_arr)

    # Parallelization over blocks
    _, block_roots, block_color, _, block_basis, block_psis, block_intercomms = (
        basis.split_into_block_basis_and_redistribute_psi(hOp, psi_arr, verbose=verbose)
    )

    bcomm = block_basis.comm
    brank = bcomm.rank if bcomm is not None else 0

    last_state = [ManyBodyState() for _ in block_psis]
    dense = len(block_basis) < 500
    if dense:
        psi_dense = block_basis.build_vector(block_psis).T
        psi_dense_local, r = build_qr(psi_dense)
    else:
        psi_dense = block_basis.build_vector(block_psis, root=0).T
        if brank == 0:
            psi_dense, r = build_qr(psi_dense)
        r = bcomm.bcast(r if brank == 0 else None, root=0)
        rows, columns = bcomm.bcast(psi_dense.shape if brank == 0 else None, root=0)
        psi_dense_local = np.empty((len(block_basis.local_basis), columns), dtype=complex, order="C")
        send_counts = np.empty((bcomm.size), dtype=int) if brank == 0 else None
        bcomm.Gather(np.array([psi_dense_local.size]), send_counts, root=0)
        offsets = np.array([np.sum(send_counts[:r]) for r in range(bcomm.size)], dtype=int) if brank == 0 else None
        bcomm.Scatterv(
            [psi_dense, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if brank == 0 else None,
            psi_dense_local,
            root=0,
        )

    if psi_dense_local.shape[1] == 0:
        return np.zeros((len(iws), n, n), dtype=complex), np.zeros((len(ws), n, n), dtype=complex)

    it_max = block_basis.size // n
    if block_basis.size % n != 0:
        it_max += 1
    it_max = max(1, it_max)
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

        # delta_guess = np.linalg.norm(np.conj(betas[-1].T) @ betas[-1])
        if np.any(np.abs(betas[-1]) > 1e6):
            return True
        if it_max >= 10 and alphas.shape[0] % (it_max // 10) != 0:
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

        d_gs = np.max(np.abs(gs_new - gs_prev))
        if verbose:
            print(rf"$\delta$ = {d_gs}")
        return d_gs < max(slaterWeightMin, 1e-12)

    if dense:
        H = block_basis.build_dense_matrix(hOp)
        alphas, betas = get_block_Lanczos_matrices_dense(
            psi0=psi_dense_local,
            h=H,
            converged=converged,
            verbose=verbose,
        )
    else:
        h_local = block_basis.build_sparse_matrix(hOp)[:, block_basis.local_indices]

        def matmat(v):
            res = h_local @ v
            if bcomm is not None:
                bcomm.Reduce(MPI.IN_PLACE if brank == 0 else res, res, op=MPI.SUM, root=0)
            return res.reshape(h_local.shape[0], v.shape[1])

        H = sp.sparse.linalg.LinearOperator(
            (len(block_basis), len(block_basis.local_indices)),
            matvec=matmat,
            rmatvec=matmat,
            matmat=matmat,
            rmatmat=matmat,
            dtype=complex,
        )

        # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
        alphas, betas = get_block_Lanczos_matrices(
            psi0=psi_dense_local,
            h=H,
            converged=converged,
            verbose=False and verbose,
            comm=bcomm,
        )

    gs_matsubara, gs_realaxis = calc_mpi_Greens_function_from_alpha_beta(
        alphas, betas, iws, ws, e, delta, r, verbose, comm=bcomm
    )

    Q = get_Lanczos_vectors(H, alphas, betas, psi_dense_local, comm=bcomm if not dense else None, which=-1)
    # Combine the results from every block
    for i, qi in enumerate(block_basis.build_state(Q.T)):
        last_state[i] += qi
    last_state = basis.redistribute_psis(last_state)
    if rank == 0:
        tmp_gs_matsubara = np.empty_like(gs_matsubara)
        tmp_gs_realaxis = np.empty_like(gs_realaxis)
        for send_color in range(len(block_roots)):
            if send_color == 0:
                continue
            if matsubara:
                block_intercomms[send_color].Recv(tmp_gs_matsubara, source=0)
                gs_matsubara += tmp_gs_matsubara
            if realaxis:
                block_intercomms[send_color].Recv(tmp_gs_realaxis, source=0)
                gs_realaxis += tmp_gs_realaxis
    elif brank == 0:
        if matsubara:
            block_intercomms[0].Send(gs_matsubara, dest=0)
        if realaxis:
            block_intercomms[0].Send(gs_realaxis, dest=0)
    return gs_matsubara, gs_realaxis, last_state


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

    psi_orig, r = build_qr(psi_arr)
    if len(psi_orig) == 0:
        return np.zeros((len(iws), n, n), dtype=complex), np.zeros((len(ws), n, n), dtype=complex)

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
    gs_matsubara = np.zeros((len(iws), len(psi_orig), len(psi_orig)), dtype=complex, order="C")
    gs_realaxis = np.zeros((len(ws), len(psi_orig), len(psi_orig)), dtype=complex, order="C")
    for w_mesh, gs in zip((iws, ws + 1j * delta), (gs_matsubara, gs_realaxis)):
        if w_mesh is None:
            continue
        _, freq_roots, color, _, split_basis, psi = basis.split_and_redistribute_basis([1] * len(w_mesh), psi_orig)
        w_indices = slice(color, len(w_mesh), len(freq_roots))
        freq_basis = CIPSI_Basis(
            split_basis.impurity_orbitals,
            split_basis.bath_states,
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

            freq_basis.expand_at(w + e, psi, hOp, de2_min=1e-12)

            if True:
                # Use fully sparse implementation
                # Build basis for each frequency
                h_local = freq_basis.build_sparse_matrix(hOp)
                alphas, betas, _ = block_lanczos(
                    psi0=psi,
                    basis=basis,
                    converged=build_converged(w, delta),
                    h_mem=h_mem,
                    verbose=False and verbose,
                    slaterWeightMin=slaterWeightMin,
                    reort=reort,
                )
            else:
                # Use build basis before building sparse matrix and ruhnning Lanczos
                h_local = freq_basis.build_sparse_matrix(hOp)
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
    if basis.comm.rank == 0:
        gs_matsubara = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_matsubara, r[np.newaxis, :, :])
        gs_realaxis = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_realaxis, r[np.newaxis, :, :])
    # basis.comm.barrier()

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

    gs_matsubara = np.zeros((len(iws), n_orb, n_orb), dtype=complex, order="C")
    gs_realaxis = np.zeros((len(ws), n_orb, n_orb), dtype=complex, order="C")
    for w_mesh, gs in zip((iws, ws + 1j * delta), (gs_matsubara, gs_realaxis)):
        if w_mesh is None:
            continue
        _, freq_roots, color, _, split_basis, psi = basis.split_and_redistribute_basis([1] * len(w_mesh), psi_orig)
        w_indices = slice(color, len(w_mesh), len(freq_roots))
        freq_basis = CIPSI_Basis(
            split_basis.impurity_orbitals,
            split_basis.bath_states,
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
                    slaterWeightMin=slaterWeightMin,
                    atol=max(slaterWeightMin, 1e-5),
                )
                for (i, psi_i), (j, Ainvpsi_j) in itertools.product(enumerate(psi), enumerate(A_inv_psi)):
                    gs[w_i, i, j] = finite.inner(psi_i, Ainvpsi_j)
            else:
                A = freq_basis.build_sparse_matrix(A_op)
                A = finite.create_linear_operator(A, freq_basis.comm)
                psi_i = freq_basis.build_vector(psi)
                A_inv_psi_v = freq_basis.build_vector(A_inv_psi).T

                for j, psi_j in enumerate(psi_i):
                    if np.linalg.norm(psi_j) < np.finfo(float).eps:
                        continue
                    info = -1

                    while info != 0:
                        A_inv_psi_v[:, j], info = sp.sparse.linalg.gmres(
                            A, psi_j, x0=A_inv_psi_v[:, j], atol=slaterWeightMin
                        )
                        if info < 0:
                            raise RuntimeError("Parameter breakdown in bicgstab!")
                gs[w_i, :, :] = np.conj(psi_i) @ A_inv_psi_v
                A_inv_psi = freq_basis.build_state(A_inv_psi_v.T)
        freq_basis.comm.Free()

    basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == 0 else gs_matsubara, gs_matsubara, op=MPI.SUM, root=0)
    basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == 0 else gs_realaxis, gs_realaxis, op=MPI.SUM, root=0)
    # basis.comm.barrier()

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
    else:
        iws_split = None

    if realaxis:
        num_indices = np.array([len(ws) // comm.size] * comm.size, dtype=int)
        num_indices[: len(ws) % comm.size] += 1
        ws_split = ws[sum(num_indices[: comm.rank]) : sum(num_indices[: comm.rank + 1])]
    else:
        ws_split = None
    gs_matsubara_local, gs_realaxis_local = calc_local_Greens_function_from_alpha_beta(
        alphas, betas, iws_split, ws_split, e, delta, verbose
    )
    # Multiply obtained Green's function with the upper triangular matrix to restore the original block
    # R^T* G R
    if matsubara:
        gs_matsubara_local = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_matsubara_local, r[np.newaxis, :, :])
        counts = np.empty((comm.size), dtype=int)
        comm.Gather(np.array([gs_matsubara_local.size], dtype=int), counts)
        offsets = [sum(counts[:rank]) for rank in range(len(counts))] if comm.rank == 0 else None
        gs_matsubara = (
            np.empty((len(iws), r.shape[1], r.shape[1]), dtype=complex, order="C") if comm.rank == 0 else None
        )
        comm.Gatherv(gs_matsubara_local, (gs_matsubara, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
    else:
        gs_matsubara = None
    if realaxis:
        gs_realaxis_local = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_realaxis_local, r[np.newaxis, :, :])
        counts = np.empty((comm.size), dtype=int)
        comm.Gather(np.array([gs_realaxis_local.size], dtype=int), counts)
        offsets = [sum(counts[:rank]) for rank in range(len(counts))] if comm.rank == 0 else None
        gs_realaxis = np.empty((len(ws), r.shape[1], r.shape[1]), dtype=complex, order="C") if comm.rank == 0 else None
        comm.Gatherv(gs_realaxis_local, (gs_realaxis, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
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
        basis.impurity_orbitals,
        basis.bath_states,
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
    # h = basis.build_sparse_matrix(hOp, h_mem)

    # N = h.shape[0]
    n = len(psi_arr)

    if verbose:
        t0 = time.perf_counter()

    # if N == 0 or n == 0:
    #     return np.zeros((n, n, len(iws)), dtype=complex), np.zeros((n, n, len(ws)), dtype=complex)

    local_basis_lens = np.empty((comm.size), dtype=int)
    comm.Allgather(np.array([len(basis.local_basis)], dtype=int), local_basis_lens)
    if matsubara:
        gs_matsubara = np.zeros((len(iws), n, n), dtype=complex)
        local_basis = CIPSI_Basis(
            basis.impurity_orbitals,
            basis.bath_states,
            initial_basis=basis.local_basis,
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

    # comm.Barrier()
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
    blocks = get_blocks(hyb_herm)
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


def save_Greens_function(gs, omega_mesh, label, cluster_label, e_scale=1, tol=1e-8):
    """
    Save Greens function to file, using RSPt .dat format. Including offdiagonal elements.
    """
    n_orb = gs.shape[1]
    axis_label = "-realaxis"
    if np.all(np.abs(np.imag(omega_mesh)) > 1e-6):
        omega_mesh = np.imag(omega_mesh)
        axis_label = ""

    off_diags = []
    for column in range(gs.shape[2]):
        for row in range(gs.shape[1]):
            if row == column:
                continue
            if np.any(np.abs(gs[:, row, column]) > tol):
                off_diags.append((row, column))

    print(f"Writing {label}{axis_label}-{cluster_label} to files")
    with open(f"real-{label}{axis_label}-{cluster_label}.dat", "w") as fg_real, open(
        f"imag-{label}{axis_label}-{cluster_label}.dat", "w"
    ) as fg_imag:
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
                f"{w*e_scale} {np.real(np.sum(np.diag(gs[i, :, :])))/e_scale} "
                + f"{np.real(np.sum(np.diag(gs[i, :n_orb//2, :n_orb//2])))/e_scale} "
                + f"{np.real(np.sum(np.diag(gs[i, n_orb//2:, n_orb//2:])))/e_scale} "
                + " ".join(f"{np.real(el)/e_scale}" for el in np.diag(gs[i, :, :]))
                + " "
                + " ".join(f"{np.real(gs[i, row, column])/e_scale}" for row, column in off_diags)
                + "\n"
            )
            fg_imag.write(
                f"{w*e_scale} {np.imag(np.sum(np.diag(gs[i, :, :])))/e_scale} "
                + f"{np.imag(np.sum(np.diag(gs[i, :n_orb//2, :n_orb//2])))/e_scale} "
                + f"{np.imag(np.sum(np.diag(gs[i, n_orb//2:, n_orb//2:])))/e_scale} "
                + " ".join(f"{np.imag(el)/e_scale}" for el in np.diag(gs[i, :, :]))
                + " "
                + " ".join(f"{np.imag(gs[i, row, column])/e_scale}" for row, column in off_diags)
                + "\n"
            )
