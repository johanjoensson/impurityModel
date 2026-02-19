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
from impurityModel.ed.cg import bicgstab, block_bicgstab
from impurityModel.ed.block_structure import BlockStructure, get_blocks
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, applyOp as applyOp_test, inner

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
    matsubara_mesh: np.ndarray,
    omega_mesh: np.ndarray,
    psis: list[ManyBodyState],
    es: list[float],
    tau: float,
    basis: ManyBodyBasis,
    hOp: ManyBodyOperator,
    delta: float,
    blocks: list[list[int]],
    verbose: bool,
    verbose_extra: bool,
    reort: Optional,
    dN: Optional[int],
    occ_cutoff: float,
    slaterWeightMin: float,
    sparse: bool,
):
    """
    Calculate interacting Greens function.
    """
    (
        block_indices,
        block_roots,
        block_color,
        blocks_per_color,
        block_basis,
        psis,
        block_intercomms,
    ) = basis.split_basis_and_redistribute_psi([len(block) ** 2 for block in blocks], psis)
    if verbose:
        print(f"New block roots: {block_roots}")
        print(f"Blocks per color: {blocks_per_color}")
        print("=" * 80, flush=True)
    if basis.comm.rank == 0:
        block_indices_per_color = np.empty((sum(blocks_per_color)), dtype=int)
        block_offsets = np.array([sum(blocks_per_color[:r]) for r in range(len(block_roots))], dtype=int)
        for col, sender in enumerate(block_roots):
            if sender == 0:
                block_indices_per_color[
                    block_offsets[block_color] : block_offsets[block_color] + blocks_per_color[block_color]
                ] = block_indices
                continue
            basis.comm.Recv(
                block_indices_per_color[block_offsets[col] : block_offsets[col] + blocks_per_color[col]], source=sender
            )
    elif block_basis.comm.rank == 0:
        basis.comm.Send(np.array(block_indices), dest=0)

    local_gs_matsubara = []
    local_gs_realaxis = []
    IPS_ops = ([ManyBodyOperator({((orb, "c"),): 1}) for orb in blocks[bi]] for bi in block_indices)
    PS_ops = ([ManyBodyOperator({((orb, "a"),): 1}) for orb in blocks[bi]] for bi in block_indices)
    for block_i, IPS_op, PS_op in zip(block_indices, IPS_ops, PS_ops):
        alphas_IPS, betas_IPS, r_IPS = calc_Greens_function_with_offdiag(
            hOp,
            IPS_op,
            psis,
            es,
            block_basis,
            delta,
            dN=dN,
            occ_cutoff=occ_cutoff,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose_extra,
            sparse=sparse,
        )
        alphas_PS, betas_PS, r_PS = calc_Greens_function_with_offdiag(
            hOp,
            PS_op,
            psis,
            es,
            block_basis,
            -delta,
            dN=dN,
            occ_cutoff=occ_cutoff,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose_extra,
            sparse=sparse,
        )

        e0 = np.min(es)
        Z = np.sum(np.exp(-(es - e0) / tau))
        if matsubara_mesh is not None and block_basis.comm.rank == 0:
            G_IPS = np.zeros((len(matsubara_mesh), len(IPS_op), len(IPS_op)), dtype=complex)
            for e, alphas_e, betas_e, r_e in zip(es, alphas_IPS, betas_IPS, r_IPS):
                G_IPS += calc_G(alphas_e, betas_e, r_e, matsubara_mesh, e, 0) * np.exp(-(e - e0) / tau)

            G_PS = np.zeros((len(matsubara_mesh), len(PS_op), len(PS_op)), dtype=complex)
            for e, alphas_e, betas_e, r_e in zip(es, alphas_PS, betas_PS, r_PS):
                G_PS += calc_G(alphas_e, betas_e, r_e, -matsubara_mesh, e, 0) * np.exp(-(e - e0) / tau)
            local_gs_matsubara.append(
                (
                    G_IPS
                    - np.transpose(
                        G_PS,
                        (
                            0,
                            2,
                            1,
                        ),
                    )
                )
                / Z
            )
        if omega_mesh is not None and block_basis.comm.rank == 0:
            G_IPS = np.zeros((len(omega_mesh), len(IPS_op), len(IPS_op)), dtype=complex)
            for e, alphas_e, betas_e, r_e in zip(es, alphas_IPS, betas_IPS, r_IPS):
                G_IPS += calc_G(alphas_e, betas_e, r_e, omega_mesh, e, delta) * np.exp(-(e - e0) / tau)

            G_PS = np.zeros((len(omega_mesh), len(PS_op), len(PS_op)), dtype=complex)
            for e, alphas_e, betas_e, r_e in zip(es, alphas_PS, betas_PS, r_PS):
                G_PS += calc_G(alphas_e, betas_e, r_e, -omega_mesh, e, -delta) * np.exp(-(e - e0) / tau)
            local_gs_realaxis.append(
                (
                    G_IPS
                    - np.transpose(
                        G_PS,
                        (
                            0,
                            2,
                            1,
                        ),
                    )
                )
                / Z
            )
    if basis.comm.rank == 0:
        gs_matsubara = [np.empty((len(matsubara_mesh), len(block), len(block)), dtype=complex) for block in blocks]
        gs_realaxis = [np.empty((len(omega_mesh), len(block), len(block)), dtype=complex) for block in blocks]
        for col, sender in enumerate(block_roots):
            if sender == 0:
                for i, block_i in enumerate(block_indices):
                    gs_matsubara[block_i][:] = local_gs_matsubara[i]
                    gs_realaxis[block_i][:] = local_gs_realaxis[i]
                continue
            for block_idx in block_indices_per_color[block_offsets[col] : block_offsets[col] + blocks_per_color[col]]:
                # block = blocks[block_idx]

                basis.comm.Recv(gs_matsubara[block_idx], source=sender)

                basis.comm.Recv(gs_realaxis[block_idx], source=sender)

    elif block_basis.comm.rank == 0:
        for block_i, (gsm, gsr) in enumerate(zip(local_gs_matsubara, local_gs_realaxis)):
            basis.comm.Send(gsm, dest=0)
            basis.comm.Send(gsr, dest=0)

    return (gs_matsubara, gs_realaxis) if basis.comm.rank == 0 else (None, None)


def calc_Greens_function_with_offdiag(
    hOp,
    tOps,
    psis,
    es,
    block_basis,
    delta,
    dN: Optional[int],
    occ_cutoff: float,
    slaterWeightMin: float,
    verbose: bool,
    sparse: bool,
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

    excited_restrictions = [None for _ in psis]
    for ei, psi in enumerate(psis):
        excited_restrictions[ei] = block_basis.build_excited_restrictions(
            psi,
            [1],
            hOp,
            imp_change={i: (dN, dN) for i in block_basis.impurity_orbitals} if dN is not None else None,
            val_change={i: (dN, 0) for i in block_basis.impurity_orbitals} if dN is not None else None,
            con_change={i: (0, dN) for i in block_basis.impurity_orbitals} if dN is not None else None,
            cutoff=occ_cutoff,
        )
    block_v = [[ManyBodyState({}) for _ in tOps] for _ in psis]
    for (i_tOp, tOp), (j_psi, psi) in itertools.product(enumerate(tOps), enumerate(psis)):
        block_v[j_psi][i_tOp] += applyOp_test(
            tOp,
            psi,
            cutoff=slaterWeightMin,
            restrictions=excited_restrictions[j_psi],
        )
    block_v_lengths = np.array([sum(len(t_psi) for t_psi in t_psis) for t_psis in block_v])
    block_basis.comm.Allreduce(MPI.IN_PLACE, block_v_lengths, op=MPI.SUM)

    (
        excited_indices,
        excited_roots,
        excited_color,
        excited_states_per_color,
        original_excited_basis,
        excited_psis,
        excited_intercomms,
    ) = block_basis.split_basis_and_redistribute_psi(
        (block_v_lengths + 1) ** 2, [t_psi for t_psis in block_v for t_psi in t_psis]
    )
    if verbose:
        print(f"New excited state roots: {excited_roots}")
        print(f"excited states per color: {excited_states_per_color}")
        print("=" * 80, flush=True)
    if block_basis.comm.rank == 0:
        excited_indices_per_color = np.empty((sum(excited_states_per_color)), dtype=int)
        excited_offsets = np.array([sum(excited_states_per_color[:r]) for r in range(len(excited_roots))], dtype=int)
        for col, sender in enumerate(excited_roots):
            if sender == 0:
                excited_indices_per_color[
                    excited_offsets[excited_color] : excited_offsets[excited_color]
                    + excited_states_per_color[excited_color]
                ] = excited_indices
                continue
            block_basis.comm.Recv(
                excited_indices_per_color[excited_offsets[col] : excited_offsets[col] + excited_states_per_color[col]],
                source=sender,
            )
    elif original_excited_basis.comm.rank == 0:
        block_basis.comm.Send(np.array(excited_indices), dest=0)

    excited_block_psis = [[ManyBodyState({}) for _ in vs] for vs in block_v]
    for i, j in itertools.product(range(len(tOps)), range(len(psis))):
        excited_block_psis[j][i] += excited_psis[j * len(tOps) + i]
    local_alphas = []
    local_betas = []
    local_r = []
    for excited_psis, er in ((excited_block_psis[ei], excited_restrictions[ei]) for ei in excited_indices):
        if verbose and er is not None:
            print("Excited state restrictions:")
            for indices, occupations in er.items():
                print(f"---> {sorted(indices)} : {occupations}")
        excited_basis = Basis(
            original_excited_basis.impurity_orbitals,
            original_excited_basis.bath_states,
            initial_basis=set(state for psi in excited_psis for state in psi),
            restrictions=er,
            comm=original_excited_basis.comm,
            verbose=verbose,
            truncation_threshold=original_excited_basis.truncation_threshold,
            tau=original_excited_basis.tau,
            spin_flip_dj=original_excited_basis.spin_flip_dj,
        )
        excited_psis = excited_basis.redistribute_psis(excited_psis)

        if sparse:
            alphas, betas, r = block_Green_sparse(
                hOp=hOp,
                psi_arr=excited_psis,
                basis=excited_basis,
                delta=delta,
                slaterWeightMin=slaterWeightMin,
                verbose=verbose,
            )
        else:
            alphas, betas, r = block_Green(
                reort=None,
                hOp=hOp,
                psi_arr=excited_psis,
                basis=excited_basis,
                delta=delta,
                slaterWeightMin=slaterWeightMin,
                verbose=verbose,
            )
        local_alphas.append(alphas)
        local_betas.append(betas)
        local_r.append(r)

    excited_alphas = None
    excited_betas = None
    excited_r = None
    if block_basis.comm.rank == 0:
        excited_alphas = [None for _ in psis]
        excited_betas = [None for _ in psis]
        excited_r = [None for _ in psis]
        for col, sender in enumerate(excited_roots):
            if sender == 0:
                for i, excited_i in enumerate(excited_indices):
                    excited_alphas[excited_i] = local_alphas[i]
                    excited_betas[excited_i] = local_betas[i]
                    excited_r[excited_i] = local_r[i]
                continue
            received_alphas = block_basis.comm.recv(source=sender)
            received_betas = block_basis.comm.recv(source=sender)
            received_r = block_basis.comm.recv(source=sender)
            for i, excited_i in enumerate(
                excited_indices_per_color[excited_offsets[col] : excited_offsets[col] + excited_states_per_color[col]]
            ):
                excited_alphas[excited_i] = received_alphas[i]
                excited_betas[excited_i] = received_betas[i]
                excited_r[excited_i] = received_r[i]
        assert not any(alpha is None for alpha in excited_alphas), f"{excited_alphas=}"
        assert not any(beta is None for beta in excited_betas), f"{excited_betas=}"
        assert not any(r is None for r in excited_r), f"{excited_r=}"

    elif excited_basis.comm.rank == 0:
        block_basis.comm.send(local_alphas, dest=0)
        block_basis.comm.send(local_betas, dest=0)
        block_basis.comm.send(local_r, dest=0)

    return excited_alphas, excited_betas, excited_r


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
        return np.all(np.abs(gs_new - gs_prev) < max(slaterWeightMin, 1e-12))

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


def calc_continuants(diagonal, offdiagonal):
    """
    Calculate continued fraction continuants.

    """

    An = np.empty_like(diagonal)
    Bn = np.empty_like(An)
    An[-1] = np.eye(diagonal.shape[1])
    Bn[-1] = 0
    An[0] = diagonal[0]
    Bn[0] = 1  # np.eye(offdiagonal.shape[1])
    for n in range(1, diagonal.shape[0]):
        An[n] = diagonal[n] @ An[n - 1] - np.conj(offdiagonal[n].T) @ An[n - 2] @ offdiagonal[n]
        Bn[n] = diagonal[n] @ Bn[n - 1] - np.conj(offdiagonal[n].T) @ Bn[n - 2] @ offdiagonal[n]
    return An, Bn


def block_Green(
    hOp,
    psi_arr,
    basis,
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

    N = len(basis)
    n = len(psi_arr)

    basis.expand(hOp, slaterWeightMin=slaterWeightMin, max_it=10)
    psi_arr = basis.redistribute_psis(psi_arr)
    alphas, betas, r = block_green_impl(basis, hOp, psi_arr, delta, slaterWeightMin, verbose)
    done = False
    causal = False
    cutoff = slaterWeightMin
    while not done:
        old_size = basis.size
        basis.expand(hOp, slaterWeightMin=slaterWeightMin, max_it=1)
        if basis.size == old_size:
            break
        last_state = psi_arr
        while basis.size > basis.truncation_threshold:
            cutoff = max(10 * cutoff, np.finfo(float).eps)
            basis.clear()
            new_states = set()
            for psi in last_state:
                Hpsi = applyOp_test(hOp, psi, restrictions=basis.restrictions, cutoff=cutoff)
                new_states |= set(Hpsi.keys())
            basis.add_states(new_states)
            last_state = Hpsi
        if verbose:
            print(f"Expanded basis contains {basis.size} states")
        alphas_prev = alphas
        betas_prev = betas
        alphas, betas, r = block_green_impl(
            basis, hOp, basis.redistribute_psis(psi_arr), delta, slaterWeightMin, verbose
        )
        if n == 1:
            An_prev, Bn_prev = calc_continuants(
                np.diag(np.diag(alphas[-1]) + 1j * delta)[None] - alphas_prev, betas_prev
            )
            An, Bn = calc_continuants(np.diag(np.diag(alphas[-1]) + 1j * delta)[None] - alphas, betas_prev)
            done = np.abs(An_prev[-1] / Bn_prev[-1] - An[-1] / Bn[-1]) < 1e-12
        else:
            G_prev = calc_G(alphas_prev, betas_prev, np.identity(n), np.diag(np.diag(alphas[-1])), 0, delta)
            G = calc_G(alphas, betas, np.identity(n), np.diag(np.diag(alphas[-1])), 0, delta)
            done = np.max(np.abs(G - G_prev)) < 1e-12

    return alphas, betas, r


def block_green_impl(basis, hOp, psi_arr, delta, slaterWeightMin, verbose):
    comm = basis.comm
    rank = comm.rank if comm is not None else 0
    N = len(basis)
    n = len(psi_arr)

    # Parallelization over blocks
    # _, block_roots, block_color, _, block_basis, block_psis, block_intercomms = (
    #     basis.split_into_block_basis_and_redistribute_psi(hOp, psi_arr, verbose=verbose)
    # )
    block_basis = basis
    block_psis = psi_arr
    block_roots = [0]
    block_color = 0
    block_intercomms = None

    bcomm = block_basis.comm
    brank = bcomm.rank if bcomm is not None else 0

    last_state = [ManyBodyState() for _ in block_psis]
    dense = len(block_basis) < 500
    if dense:
        psi_dense = block_basis.build_vector(block_psis, slaterWeightMin=slaterWeightMin).T
        psi_dense_local, r = build_qr(psi_dense)
    else:
        psi_dense = block_basis.build_vector(block_psis, root=0, slaterWeightMin=slaterWeightMin).T
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
        return np.zeros((0, n, n), dtype=complex), np.zeros((0, n, n), dtype=complex), r

    it_max = block_basis.size // n
    if block_basis.size % n != 0:
        it_max += 1
    it_max = max(1, it_max)

    # delta_min = max(slaterWeightMin**2, np.finfo(float).eps)
    delta_min = max(slaterWeightMin**2, 1e-12)

    def converged(alphas, betas, verbose=False):
        if alphas.shape[0] <= 1:
            return False

        # For scalar valued Lanczos, calculate convergents to check for convergence.
        if n == 1:
            An, Bn = calc_continuants(np.diag(np.diag(alphas[-1]) + 1j * delta)[None] - alphas, betas)
            if verbose:
                print(f"delta = {np.abs(An[-2] / Bn[-2] - An[-1] / Bn[-1])}")
            if abs(Bn[-1]) < 1e-6:
                return abs(Bn[-1] * An[-2] - An[-1] * Bn[-2]) <= abs(delta_min * Bn[-1] * Bn[-2])

            return np.abs(An[-2] / Bn[-2] - An[-1] / Bn[-1]) < delta_min

        # For matrix valued (block) Lanczos, continued fractions are harder to estimate convergence for
        wIs = ((delta * 1j) * np.identity(alphas.shape[1], dtype=complex) + np.diag(np.diag(alphas[-1])))[np.newaxis]
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

        d_g = np.max(np.abs(gs_new - gs_prev))
        if verbose:
            print(f"delta = {d_g}")
        return d_g < delta_min

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
            verbose=verbose,
            comm=bcomm,
        )
    return alphas, betas, r


def block_Green_sparse(
    hOp,
    psi_arr,
    basis,
    delta,
    slaterWeightMin=0,
    verbose=True,
):
    """
    calculate  one block of the Greens function. This function builds the many body basis iteratively. Reducing memory requrements.
    """
    mpi = basis.comm is not None
    comm = basis.comm if mpi else None
    rank = comm.rank if mpi else 0

    N = len(basis)
    n = len(psi_arr)

    if N == 0 or n == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), np.zeros((n, n), dtype=complex)
    psi_arr = basis.redistribute_psis(psi_arr)
    psi_dense = basis.build_vector(psi_arr, root=0, slaterWeightMin=slaterWeightMin).T
    if rank == 0:
        psi_dense, r = build_qr(psi_dense)
    if mpi:
        r = comm.bcast(r if rank == 0 else None, root=0)
        rows, columns = comm.bcast(psi_dense.shape if rank == 0 else None, root=0)
        psi_dense_local = np.empty((len(basis.local_basis), columns), dtype=complex, order="C")
        send_counts = np.empty((comm.size), dtype=int) if rank == 0 else None
        comm.Gather(np.array([psi_dense_local.size]), send_counts, root=0)
        offsets = np.array([np.sum(send_counts[:r]) for r in range(comm.size)], dtype=int) if rank == 0 else None
        comm.Scatterv(
            [psi_dense, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None,
            psi_dense_local,
            root=0,
        )
    else:
        psi_dense_local = psi_dense
    psi_arr = basis.build_state(psi_dense_local.T, slaterWeightMin=0)
    if len(psi_arr) == 0:
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), r

    delta_min = max(slaterWeightMin**2, 1e-12)

    def converged(alphas, betas, verbose=False):
        if alphas.shape[0] <= 1:
            return False

        # For scalar valued Lanczos, calculate convergents to check for convergence.
        if n == 1:
            An, Bn = calc_continuants(np.diag(np.diag(alphas[-1]) + 1j * delta)[None] - alphas, betas)
            if abs(Bn[-1]) < 1e-6:
                # The An and Bn can become ridiculously tiny, in which case division becomes very unreliable
                # delta >= |A2/B2 - A1/Ba| = |(A2*B1 - A1*B2)/(B1*B2)| <=? |delta*B1*B2| >= |A2*B1 - A1*B2|
                if verbose:
                    print(f"delta = {Bn[-1] * An[-2] - An[-1] * Bn[-2]}/{Bn[-1] * Bn[-2]}")
                return abs(Bn[-1] * An[-2] - An[-1] * Bn[-2]) <= abs(delta_min * Bn[-1] * Bn[-2])
            if verbose:
                print(f"delta = {An[-2]/Bn[-2] - An[-1]/Bn[-1]}")
            return abs(An[-2] / Bn[-2] - An[-1] / Bn[-1]) <= delta_min

        # For matrix valued (block) Lanczos, continued fractions are harder to estimate convergence for
        wIs = ((delta * 1j) * np.identity(alphas.shape[1], dtype=complex) + np.diag(np.diag(alphas[-1])))[np.newaxis]
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

        d_g = np.max(np.abs(gs_new - gs_prev))
        if verbose:
            print(f"delta = {d_g}", flush=True)
        return d_g < delta_min

    alphas, betas = block_lanczos_sparse(psi_arr, hOp, basis, converged, verbose=verbose)

    return alphas, betas, r


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

        # Hpsi = [ManyBodyState() for _ in psi]
        # for Hps, ps in zip(Hpsi, psi):
        #     Hps += applyOp_test(hOp, ps, cutoff=slaterWeightMin, restrictions=freq_basis.restrictions)
        # E_psi = np.empty((len(psi)), dtype=float)
        # for i, (ps, Hps) in enumerate(zip(psi, Hpsi)):
        #     E_psi[i] = inner(psi, Hpsi).real

        for w_i, w in itertools.islice(
            zip(range(len(w_mesh)), w_mesh), w_indices.start, w_indices.stop, w_indices.step
        ):

            freq_basis.expand_at(np.zeros((len(psi))) - (w + e), psi, hOp, de2_min=1e-5)

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


def Green_freq_bicgstab_fixed_basis(w_mesh, hOp, psi, e, basis, slaterWeightMin):
    _, freq_roots, color, freq_per_color, split_basis, psi, _ = basis.split_basis_and_redistribute_psi(
        [1] * len(w_mesh), psi
    )
    offsets = [np.sum(freq_per_color[:c], dtype=int) for c in range(len(freq_roots))]
    w_indices = slice(int(offsets[color]), int(offsets[color] + freq_per_color[color]), 1)
    freq_basis = CIPSI_Basis(
        split_basis.impurity_orbitals,
        split_basis.bath_states,
        initial_basis=[],  # sorted(set(state for p in psi for state in p.keys())),
        restrictions=None,  # split_basis.restrictions,
        truncation_threshold=split_basis.truncation_threshold,
        spin_flip_dj=split_basis.spin_flip_dj,
        tau=split_basis.tau,
        verbose=False,
        comm=split_basis.comm,
    )
    # psi = freq_basis.redistribute_psis(psi)

    gs = np.zeros((len(w_mesh), len(psi), len(psi)), dtype=complex)
    max_basis_size = 0
    freq = None
    A_inv_psi = [ManyBodyState() for _ in psi]
    for w_i, w in zip(range(w_indices.start, w_indices.stop, w_indices.step), w_mesh[w_indices]):

        for aip in A_inv_psi:
            aip.prune(slaterWeightMin)
        freq_basis.clear()
        freq_basis.add_states(sorted(set(state for p in itertools.chain(psi, A_inv_psi) for state in p.keys())))
        psi = freq_basis.redistribute_psis(psi)

        freq_basis.expand_at(np.array([w.real + e] * len(psi), dtype=float), psi, hOp, de2_min=slaterWeightMin / 10)

        # A_op = ManyBodyOperator({processes: w.real + e - amp for processes, amp in hOp.items()})

        diag = np.zeros((len(freq_basis)), dtype=complex)
        diag[freq_basis.local_indices] = w + e
        A_local = sp.sparse.diags(diag) - freq_basis.build_sparse_matrix(hOp)
        A = sp.sparse.linalg.LinearOperator(
            dtype=A_local.dtype,
            shape=A_local.shape,
            matvec=finite.mpi_matmat(A_local, freq_basis.comm),
            matmat=finite.mpi_matmat(A_local, freq_basis.comm),
        )
        psi_dense = freq_basis.build_vector(freq_basis.redistribute_psis(psi)).T
        A_inv_psi_dense = np.empty_like(psi_dense)
        for i in range(len(psi)):
            A_inv_psi_dense[:, i], info = sp.sparse.linalg.bicgstab(
                A=A,
                b=psi_dense[:, i],
                x0=freq_basis.build_vector(freq_basis.redistribute_psis([A_inv_psi[i]])).T,
                atol=1e-5,
            )
            if info < 0:
                raise RuntimeError(f"Breakdown in bicgstab! {info}")
            elif info > 0:
                raise RuntimeError(f"bicgstab did not converge after {info} iterations.")
        gs[w_i] = np.conj(psi_dense.T) @ A_inv_psi_dense
        A_inv_psi = freq_basis.build_state(A_inv_psi_dense.T)
        if len(freq_basis) > max_basis_size:
            max_basis_size = len(freq_basis)
            freq = w

    print(f"Maximum basis size: {max_basis_size} at frequency {freq}")
    basis.comm.Allreduce(MPI.IN_PLACE, gs, op=MPI.SUM)
    return gs


def Green_freq_bicgstab(w_mesh, hOp, psi, e, basis, slaterWeightMin):
    _, freq_roots, color, freq_per_color, split_basis, psi, _ = basis.split_basis_and_redistribute_psi(
        [1] * len(w_mesh), psi
    )
    offsets = [np.sum(freq_per_color[:c], dtype=int) for c in range(len(freq_roots))]
    w_indices = slice(int(offsets[color]), int(offsets[color] + freq_per_color[color]), 1)
    freq_basis = CIPSI_Basis(
        split_basis.impurity_orbitals,
        split_basis.bath_states,
        initial_basis=[],  # sorted(set(state for p in psi for state in p.keys())),
        restrictions=None,  # split_basis.restrictions,
        truncation_threshold=split_basis.truncation_threshold,
        spin_flip_dj=split_basis.spin_flip_dj,
        tau=split_basis.tau,
        verbose=False,
        comm=split_basis.comm,
    )
    psi = freq_basis.redistribute_psis(psi)

    gs = np.zeros((len(w_mesh), len(psi), len(psi)), dtype=complex)
    max_basis_size = 0
    freq = None
    A_inv_psi = [ManyBodyState() for _ in psi]
    for w_i, w in zip(range(w_indices.start, w_indices.stop, w_indices.step), w_mesh[w_indices]):

        for aip in A_inv_psi:
            aip.prune(slaterWeightMin)
        freq_basis.clear()
        freq_basis.add_states(sorted(set(state for p in itertools.chain(psi, A_inv_psi) for state in p.keys())))

        A_op = ManyBodyOperator({((0, "c"), (0, "a")): w + e, ((0, "a"), (0, "c")): w + e}) - hOp
        A_inv_psi = block_bicgstab(
            A=A_op,
            x0=freq_basis.redistribute_psis(A_inv_psi),
            y=freq_basis.redistribute_psis(psi),
            basis=freq_basis,
            slaterWeightMin=slaterWeightMin,
            atol=1e-5,
        )
        for (i, psi_i), (j, Ainvpsi_j) in itertools.product(
            enumerate(freq_basis.redistribute_psis(psi)), enumerate(A_inv_psi)
        ):
            gs[w_i, i, j] = inner(psi_i, Ainvpsi_j)
        if len(freq_basis) > max_basis_size:
            max_basis_size = len(freq_basis)
            freq = w

    print(f"Maximum basis size: {max_basis_size} at frequency {freq}")
    basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == 0 else gs, gs, op=MPI.SUM, root=0)
    return gs


def block_Green_freq(
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

    if len(psi_arr) == 0 or len(basis) == 0:
        return np.zeros((len(iws), len(psi_arr), len(psi_arr)), dtype=complex), np.zeros(
            (len(ws), len(psi_arr), len(psi_arr)), dtype=complex
        )

    psi_dense, r = build_qr(basis.build_vector(psi_arr).T)
    psi_orig = basis.build_state(psi_dense.T)
    n_orb = len(psi_orig)

    if n_orb == 0:
        return np.zeros((len(iws), len(psi_arr), len(psi_arr)), dtype=complex), np.zeros(
            (len(ws), len(psi_arr), len(psi_arr)), dtype=complex
        )

    gs_matsubara = Green_freq_bicgstab_fixed_basis(iws, hOp, psi_orig, e, basis, slaterWeightMin)
    gs_realaxis = Green_freq_bicgstab_fixed_basis(ws + 1j * delta, hOp, psi_orig, e, basis, slaterWeightMin)

    return (
        np.conj(r.T)[np.newaxis] @ gs_matsubara @ r[np.newaxis],
        np.conj(r.T)[np.newaxis] @ gs_realaxis @ r[np.newaxis],
    )


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


def calc_G(alphas, betas, r, omega, e, delta):
    if alphas.shape[0] == 0:
        return np.zeros((len(omega), alphas.shape[1], alphas.shape[1]), dtype=complex)
    I = np.identity(alphas.shape[1], dtype=complex)
    omegaP = omega + 1j * delta + e
    # G_inv = np.zeros((len(omega), alphas.shape[1], alphas.shape[1]), dtype=complex)
    wIs = omegaP[:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
    G_inv = wIs - alphas[-1][np.newaxis]
    for alpha, beta in zip(alphas[-2::-1], betas[-2::-1]):
        G_inv = (
            wIs
            - alpha[np.newaxis, :, :]
            - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(G_inv, beta[np.newaxis, :, :])
        )
    return np.conj(r.T)[np.newaxis] @ np.linalg.solve(G_inv, r[np.newaxis])


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
