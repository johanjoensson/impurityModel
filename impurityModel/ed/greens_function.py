import numpy as np
import scipy as sp
import time

from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.lanczos import get_block_Lanczos_matrices
from impurityModel.ed.manybody_basis import CIPSI_Basis, Basis

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


def get_Greens_function(
    nBaths,
    matsubara_mesh,
    omega_mesh,
    es,
    psis,
    basis,
    l,
    hOp,
    delta,
    blocks,
    verbose,
    reort,
    mpi_distribute=False,
    dense_cutoff=1e3,
):
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    tOpsPS = spectra.getPhotoEmissionOperators(nBaths, l=l)
    tOpsIPS = spectra.getInversePhotoEmissionOperators(nBaths, l=l)
    gsIPS_matsubara, gsIPS_realaxis = calc_Greens_function_with_offdiag(
        n_spin_orbitals,
        hOp,
        tOpsIPS,
        psis,
        es,
        basis,
        matsubara_mesh,
        omega_mesh,
        delta,
        blocks=blocks,
        slaterWeightMin=1e-8,
        verbose=verbose,
        reort=reort,
        dense_cutoff=dense_cutoff,
    )
    gsPS_matsubara, gsPS_realaxis = calc_Greens_function_with_offdiag(
        n_spin_orbitals,
        hOp,
        tOpsPS,
        psis,
        es,
        basis,
        -matsubara_mesh if matsubara_mesh is not None else None,
        -omega_mesh if omega_mesh is not None else None,
        -delta,
        blocks=blocks,
        slaterWeightMin=1e-8,
        verbose=verbose,
        reort=reort,
        dense_cutoff=dense_cutoff,
    )

    if mpi_distribute:
        if matsubara_mesh is not None:
            gsIPS_matsubara = comm.bcast(gsIPS_matsubara, root=0)
            gsPS_matsubara = comm.bcast(gsPS_matsubara, root=0)
        if omega_mesh is not None:
            gsIPS_realaxis = comm.bcast(gsIPS_realaxis, root=0)
            gsPS_realaxis = comm.bcast(gsPS_realaxis, root=0)
    if matsubara_mesh is not None and (mpi_distribute or comm.rank == 0):
        gs_matsubara = gsIPS_matsubara - np.transpose(
            gsPS_matsubara,
            (
                0,
                2,
                1,
                3,
            ),
        )
    else:
        gs_matsubara = None
    if omega_mesh is not None and (mpi_distribute or comm.rank == 0):
        gs_realaxis = gsIPS_realaxis - np.transpose(
            gsPS_realaxis,
            (
                0,
                2,
                1,
                3,
            ),
        )
    else:
        gs_realaxis = None
    return gs_matsubara, gs_realaxis


def calc_Greens_function_with_offdiag(
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
    blocks=None,
    slaterWeightMin=0,
    parallelization_mode="H_build",
    verbose=True,
    dense_cutoff=1e3,
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
    n = len(es)
    excited_restrictions = basis.build_excited_restrictions(imp_change=(1, 1), val_change=(1, 0), con_change=(0, 1))

    if blocks is None:
        blocks = [list(range(len(tOps)))]
    t_mems = [{} for _ in tOps]
    if parallelization_mode == "eigen_states":
        gs_matsubara = np.zeros((n, len(tOps), len(tOps), len(iw)), dtype=complex)
        gs_realaxis = np.zeros((n, len(tOps), len(tOps), len(w)), dtype=complex)
        for i in finite.get_job_tasks(rank, ranks, range(len(psis))):
            psi = psis[i]
            e = es[i]

            v = []
            local_excited_basis = set()
            for block in blocks:
                block_v = []
                for i_tOp, tOp in [(orb, tOps[orb]) for orb in block]:
                    v = finite.applyOp_2(
                        n_spin_orbitals,
                        tOp,
                        psi,
                        slaterWeightMin=0,
                        restrictions=None,  # excited_restrictions,
                        opResult=t_mems[i_tOp],
                    )
                    local_excited_basis |= v.keys()
                    block_v.append(v)
            excited_basis = Basis(
                ls=basis.ls,
                bath_states=basis.bath_states,
                initial_basis=local_excited_basis,
                restrictions=excited_restrictions,
                num_spin_orbitals=basis.num_spin_orbitals,
                comm=basis.comm,
                verbose=False and verbose,
                truncation_threshold=basis.truncation_threshold,
                spin_flip_dj=basis.spin_flip_dj,
                tau=basis.tau,
            )

            h_mem = excited_basis.expand(hOp, slaterWeightMin=slaterWeightMin)

            gs_matsubara_i, gs_realaxis_i = get_block_Green(
                n_spin_orbitals=n_spin_orbitals,
                hOp=hOp,
                psi_arr=v,
                basis=excited_basis,
                e=e,
                iws=iw,
                ws=w,
                delta=delta,
                h_mem=h_mem,
                slaterWeightMin=slaterWeightMin,
                parallelization_mode="serial",
                verbose=verbose,
                reort=reort,
                dense_cutoff=dense_cutoff,
            )
            comm.Reduce(gs_matsubara_i, gs_matsubara)
            comm.Reduce(gs_realaxis_i, gs_realaxis)
    elif parallelization_mode == "H_build":
        if iw is not None:
            gs_matsubara = np.zeros((n, len(tOps), len(tOps), len(iw)), dtype=complex) if comm.rank == 0 else None
        else:
            gs_matsubara = None
        if w is not None:
            gs_realaxis = np.zeros((n, len(tOps), len(tOps), len(w)), dtype=complex) if comm.rank == 0 else None
        else:
            gs_realaxis = None
        for i, (psi, e) in enumerate(zip(psis, es)):
            for block in blocks:
                block_v = []
                local_excited_basis = set()
                t0 = time.perf_counter()
                for i_tOp, tOp in [(orb, tOps[orb]) for orb in block]:
                    v = finite.applyOp_2(
                        n_spin_orbitals,
                        tOp,
                        psi,
                        slaterWeightMin=0,
                        restrictions=None,  # excited_restrictions,
                        opResult=t_mems[i_tOp],
                    )
                    local_excited_basis |= v.keys()
                    block_v.append(v)

                excited_basis = Basis(
                    ls=basis.ls,
                    bath_states=basis.bath_states,
                    initial_basis=local_excited_basis,
                    restrictions=excited_restrictions,
                    num_spin_orbitals=basis.num_spin_orbitals,
                    comm=basis.comm,
                    verbose=verbose,
                    truncation_threshold=basis.truncation_threshold,
                    tau=basis.tau,
                    spin_flip_dj=basis.spin_flip_dj,
                )
                h_mem = excited_basis.expand(hOp, dense_cutoff=dense_cutoff, slaterWeightMin=slaterWeightMin)

                if verbose:
                    print(f"time(build excited state basis) = {time.perf_counter() - t0}")
                gs_matsubara_i, gs_realaxis_i = get_block_Green(
                    n_spin_orbitals=n_spin_orbitals,
                    hOp=hOp,
                    psi_arr=block_v,
                    basis=excited_basis,
                    e=e,
                    iws=iw,
                    ws=w,
                    delta=delta,
                    h_mem=h_mem,
                    slaterWeightMin=slaterWeightMin,
                    parallelization_mode=parallelization_mode,
                    verbose=verbose,
                    reort=reort,
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
    # psi_start = basis.build_vector(psi_arr).T  # /basis.comm.size
    psi_start = basis.build_distributed_vector(psi_arr).T  # /basis.comm.size
    counts = np.empty((comm.size,), dtype=int) if comm.rank == 0 else None
    comm.Gather(np.array([n * len(basis.local_basis)], dtype=int), counts, root=0)
    offsets = [sum(counts[:r]) for r in range(len(counts))] if comm.rank == 0 else None
    psi_start_0 = np.empty((N, n), dtype=complex) if comm.rank == 0 else None
    comm.Gatherv(psi_start, (psi_start_0, counts, offsets, MPI.DOUBLE_COMPLEX), root=0)
    if comm.rank == 0:
        # Do a QR decomposition of the starting block.
        # Later on, use r to restore the block corresponding to
        psi0_0, r = sp.linalg.qr(psi_start_0, mode="economic", overwrite_a=True, check_finite=False)

        # Find which columns (if any) are 0 in psi0
        column_mask = np.any(np.abs(psi0_0) > 1e-12, axis=0)
        rows, columns = psi0_0.shape
    else:
        r = None
    psi0 = np.empty_like(psi_start)
    comm.Scatterv((psi0_0, counts, offsets, MPI.DOUBLE_COMPLEX) if comm.rank == 0 else None, psi0, root=0)
    column_mask = comm.bcast(column_mask if comm.rank == 0 else None, root=0)
    rows = comm.bcast(rows if comm.rank == 0 else None, root=0)
    columns = comm.bcast(columns if comm.rank == 0 else None, root=0)
    if verbose:
        print(f"time(set up psi_start) = {time.perf_counter() - t0}")

    if rows == 0 or columns == 0:
        return np.zeros((n, n, len(iws)), dtype=complex), np.zeros((n, n, len(ws)), dtype=complex)

    # If we have a realaxis mesh, prefer to check convergence on that
    # if not, use the Matsubara mesh
    if realaxis:
        conv_w = ws
        delta_p = delta
    elif matsubara:
        conv_w = iws
        delta_p = 0

    def gaussian(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)

    # Select points from the frequency mesh, according to a Normal distribuition
    # centered on (value) 0.
    n_samples = max(len(conv_w) // 50, 1)
    weights = np.ones(conv_w.shape)
    weights /= np.sum(weights)

    def matrix_print(m):
        print("\n".join(["  ".join([f"{np.real(el): 5.3f}  {np.imag(el):+5.3f}j" for el in row]) for row in m]))

    def converged(alphas, betas):
        if alphas.shape[0] == 1:
            return 1.0

        w = np.random.choice(conv_w, size=min(n_samples, len(conv_w)), p=weights, replace=False)
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
        return np.max(
            np.abs(
                np.diagonal(np.linalg.inv(gs_new), axis1=1, axis2=2)
                - np.diagonal(np.linalg.inv(gs_prev), axis1=1, axis2=2)
            )
        )

    # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
    alphas, betas, _ = get_block_Lanczos_matrices(
        psi0=psi0[:, column_mask],
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


def calc_mpi_Greens_function_from_alpha_beta(alphas, betas, iws, ws, e, delta, r, verbose):
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
        gs_matsubara = np.empty((len(iws), r.shape[1], r.shape[1]), dtype=complex) if comm.rank == 0 else None
        comm.Gatherv(gs_matsubara_local, (gs_matsubara, counts, offsets, MPI.DOUBLE_COMPLEX), root=0)
        if comm.rank == 0:
            gs_matsubara = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_matsubara, r[np.newaxis, :, :])
    if realaxis:
        counts = np.empty((comm.size), dtype=int)
        comm.Gather(np.array([gs_realaxis_local.shape[1] ** 2 * len(ws_split)], dtype=int), counts)
        offsets = [sum(counts[:r]) for r in range(len(counts))] if comm.rank == 0 else None
        gs_realaxis = np.empty((len(ws), r.shape[1], r.shape[1]), dtype=complex) if comm.rank == 0 else None
        comm.Gatherv(gs_realaxis_local, (gs_realaxis, counts, offsets, MPI.DOUBLE_COMPLEX), root=0)
        if comm.rank == 0:
            gs_realaxis = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(gs_realaxis, r[np.newaxis, :, :])
    return gs_matsubara, gs_realaxis


def calc_local_Greens_function_from_alpha_beta(alphas, betas, iws, ws, e, delta, verbose):
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
                res = finite.applyOp_2(
                    n_spin_orbitals,
                    tOps[i_tOp],
                    {s: 1},
                    slaterWeightMin=0,  # slaterWeightMin,
                    restrictions=None,  # basis.restrictions,
                    opResult=t_mems[i_tOp],
                )
                local_excited_basis |= res.keys()
    excited_basis = CIPSI_Basis(
        ls=basis.ls,
        bath_states=basis.bath_states,
        initial_basis=local_excited_basis,
        restrictions=basis.restrictions,
        num_spin_orbitals=basis.num_spin_orbitals,
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
                v = finite.applyOp_2(
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
            ls=basis.ls,
            bath_states=basis.bath_states,
            initial_basis=list(basis),
            restrictions=basis.restrictions,
            num_spin_orbitals=basis.num_spin_orbitals,
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
    w_ind = np.argmax(G.shape)
    return np.moveaxis(
        np.conj(T.T)[np.newaxis, :, :] @ np.moveaxis(G, w_ind, 0) @ T[np.newaxis, :, :],
        0,
        w_ind,
    )


def rotate_4index_U(U4, T):
    return np.einsum("ij,kl, jlmo, mn, op", np.conj(T.T), np.conj(T.T), U4, T, T)


def save_Greens_function(gs, omega_mesh, label, e_scale=1, tol=1e-8):
    n_orb = gs.shape[0]
    axis_label = "realaxis"
    if np.all(np.abs(np.imag(omega_mesh)) > 1e-6):
        omega_mesh = np.imag(omega_mesh)
        axis_label = "Matsubara"

    off_diags = []
    for column in range(gs.shape[1]):
        for row in range(gs.shape[0]):
            if row == column:
                continue
            if np.any(np.abs(gs[row, column, :]) > tol):
                off_diags.append((row, column))

    print(f"Writing {axis_label} {label} to files")
    with open(f"real-{axis_label}-{label}.dat", "w") as fg_real, open(f"imag-{axis_label}-{label}.dat", "w") as fg_imag:
        header = "# 1 - Omega(Ry)  2 - Trace  3 - Spin down  4 - Spin up\n"
        header += "# Individual matrix elements given in the matrix below:"
        for row in range(gs.shape[0]):
            header += "\n# "
            for column in range(gs.shape[1]):
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
                f"{w*e_scale} {np.real(np.sum(np.diag(gs[:, :, i])))} "
                + f"{np.real(np.sum(np.diag(gs[:n_orb//2, :n_orb//2, i])))} "
                + f"{np.real(np.sum(np.diag(gs[n_orb//2:, n_orb//2:, i])))} "
                + " ".join(f"{np.real(el)}" for el in np.diag(gs[:, :, i]))
                + " "
                + " ".join(f"{np.real(gs[row, column, i])}" for row, column in off_diags)
                + "\n"
            )
            fg_imag.write(
                f"{w*e_scale} {np.imag(np.sum(np.diag(gs[:, :, i])))} "
                + f"{np.imag(np.sum(np.diag(gs[:n_orb//2, :n_orb//2, i])))} "
                + f"{np.imag(np.sum(np.diag(gs[n_orb//2:, n_orb//2:, i])))} "
                + " ".join(f"{np.imag(el)}" for el in np.diag(gs[:, :, i]))
                + " "
                + " ".join(f"{np.imag(gs[row, column, i])}" for row, column in off_diags)
                + "\n"
            )
