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
    restrictions,
    blocks,
    verbose,
    reort,
    mpi_distribute=False,
    dense_cutoff=1e3,
    tau=0,
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
        restrictions=restrictions,
        blocks=blocks,
        krylovSize=None,
        slaterWeightMin=1e-8,
        verbose=verbose,
        reort=reort,
        dense_cutoff=dense_cutoff,
        tau=tau,
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
        restrictions=restrictions,
        blocks=blocks,
        krylovSize=None,
        slaterWeightMin=1e-8,
        verbose=verbose,
        reort=reort,
        dense_cutoff=dense_cutoff,
        tau=tau,
    )

    if mpi_distribute:
        if matsubara_mesh is not None:
            gsIPS_matsubara = comm.bcast(gsIPS_matsubara, root=0)
            gsPS_matsubara = comm.bcast(gsPS_matsubara, root=0)
        if omega_mesh is not None:
            gsIPS_realaxis = comm.bcast(gsIPS_realaxis, root=0)
            gsPS_realaxis = comm.bcast(gsPS_realaxis, root=0)
    if matsubara_mesh is not None:
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
    if omega_mesh is not None:
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
    restrictions=None,
    blocks=None,
    krylovSize=None,
    slaterWeightMin=0,
    parallelization_mode="H_build",
    verbose=True,
    dense_cutoff=1e3,
    tau=0,
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

    if blocks is None:
        blocks = [list(range(len(tOps)))]
    if parallelization_mode == "eigen_states":
        gs_matsubara = np.zeros((n, len(tOps), len(tOps), len(iw)), dtype=complex)
        gs_realaxis = np.zeros((n, len(tOps), len(tOps), len(w)), dtype=complex)
        for i in finite.get_job_tasks(rank, ranks, range(len(psis))):
            psi = psis[i]
            e = es[i]

            v = []
            new_basis = Basis(
                initial_basis=[],
                restrictions=basis.restrictions,
                num_spin_orbitals=basis.num_spin_orbitals,
                comm=basis.comm,
                verbose=False and verbose,
                truncation_threshold=basis.truncation_threshold,
                tau=basis.tau,
            )
            for tOp in tOps:
                v.append(
                    finite.applyOp(
                        n_spin_orbitals,
                        tOp,
                        psi,
                        slaterWeightMin=slaterWeightMin,
                        restrictions=basis.restrictions,
                        opResult=t_mems[i_tOp],
                    )
                )
                for s in basis.local_basis:
                    res = finite.applyOp(
                        n_spin_orbitals,
                        tOp,
                        {s: 1},
                        slaterWeightMin=slaterWeightMin,
                        restrictions=basis.restrictions,
                        opResult=t_mems[i_tOp],
                    )
                    local_excited_basis |= res.keys()

            new_basis.add_states(new_local_basis)
            h_mem = new_basis.expand(hOp, dense_cutoff=dense_cutoff, slaterWeightMin=slaterWeightMin)

            gs_matsubara_i, gs_realaxis_i = get_block_Green(
                n_spin_orbitals=n_spin_orbitals,
                hOp=hOp,
                psi_arr=v,
                basis=new_basis,
                e=e,
                iw=iw,
                w=w,
                delta=delta,
                restrictions=restrictions,
                h_mem=h_mem,
                krylovSize=krylovSize,
                slaterWeightMin=slaterWeightMin,
                parallelization_mode="serial",
                verbose=verbose,
                reort=reort,
                dense_cutoff=dense_cutoff,
                tau=tau,
            )
            comm.Reduce(gs_matsubara_i, gs_matsubara)
            comm.Reduce(gs_realaxis_i, gs_realaxis)
    elif parallelization_mode == "H_build":
        if iw is not None:
            gs_matsubara = np.zeros((n, len(tOps), len(tOps), len(iw)), dtype=complex)
        else:
            gs_matsubara = None
        if w is not None:
            gs_realaxis = np.zeros((n, len(tOps), len(tOps), len(w)), dtype=complex)
        else:
            gs_realaxis = None

        t_mems = [{} for _ in tOps]

        local_excited_basis = set()
        for block in blocks:
            for i_tOp, tOp in [(orb, tOps[orb]) for orb in block]:
                for s in basis.local_basis:
                    res = finite.applyOp(
                        n_spin_orbitals,
                        tOps[i_tOp],
                        {s: 1},
                        slaterWeightMin=slaterWeightMin,
                        restrictions=basis.restrictions,
                        opResult=t_mems[i_tOp],
                    )
                    local_excited_basis |= res.keys()
        excited_basis = Basis(
            initial_basis=local_excited_basis,
            restrictions=basis.restrictions,
            num_spin_orbitals=basis.num_spin_orbitals,
            comm=basis.comm,
            verbose=False and verbose,
            truncation_threshold=basis.truncation_threshold,
            tau=basis.tau,
        )
        h_mem = excited_basis.expand(hOp, dense_cutoff=dense_cutoff, slaterWeightMin=slaterWeightMin)
        for i, (psi, e) in enumerate(zip(psis, es)):
            for block in blocks:
                block_v = []
                local_excited_basis = set()
                t0 = time.perf_counter()
                for i_tOp, tOp in [(orb, tOps[orb]) for orb in block]:
                    v = finite.applyOp(
                        n_spin_orbitals,
                        tOp,
                        {state: psi[state] for state in psi if state in basis.local_basis},
                        slaterWeightMin=slaterWeightMin,
                        restrictions=basis.restrictions,
                        opResult=t_mems[i_tOp],
                    )
                    # local_excited_basis |= v.keys() - set(basis.local_basis)
                    vs = comm.allgather(v)
                    v = {}
                    for v_i in vs:
                        for state in v_i:
                            v[state] = v_i[state] + v.get(state, 0)
                    block_v.append(v)

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
                    restrictions=restrictions,
                    h_mem=h_mem,
                    krylovSize=krylovSize,
                    slaterWeightMin=slaterWeightMin,
                    parallelization_mode=parallelization_mode,
                    verbose=verbose,
                    reort=reort,
                    dense_cutoff=dense_cutoff,
                    tau=tau,
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
    krylovSize=None,
    slaterWeightMin=0,
    parallelization_mode="H_build",
    verbose=True,
    dense_cutoff=1e3,
    tau=0,
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
    psi_states = [key for psi in psi_arr for key in psi.keys()]
    if np.any(np.logical_not(basis.contains(psi_states))):
        basis.add_states(psi_states[basis.comm.rank :: basis.comm.size])
        h_mem = basis.expand(hOp, h_mem, dense_cutoff=dense_cutoff, slaterWeightMin=slaterWeightMin)
    h = basis.build_sparse_matrix(hOp, h_mem)

    if verbose:
        print(f"time(build Hamiltonian operator) = {time.perf_counter() - t0}")

    N = h.shape[0]
    n = len(psi_arr)

    if verbose:
        t0 = time.perf_counter()
    psi_start = basis.build_vector(psi_arr)
    # psi_start = np.zeros((N, n), dtype=complex)
    # for i, psi in enumerate(psi_arr):
    #     states = []
    #     amps = []
    #     for ps, amp in psi.items():
    #         states.append(ps)
    #         amps.append(amp)
    #     indices = basis.index(states)
    #     psi_start[indices, i] = amps

    if verbose:
        print(f"time(set up psi_start) = {time.perf_counter() - t0}")

    rows, columns = psi_start.shape
    if rows == 0 or columns == 0:
        return np.zeros((n, n, len(iws)), dtype=complex), np.zeros((n, n, len(ws)), dtype=complex)
    # Do a QR decomposition of the starting block.
    # Later on, use r to restore the block corresponding to
    psi0, r = sp.linalg.qr(psi_start, mode="economic", overwrite_a=True, check_finite=False)

    # Find which columns (if any) are 0 in psi0
    column_mask = np.any(np.abs(psi0) > 1e-12, axis=0)

    rows, columns = psi0.shape
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
        return np.max(np.abs(np.diagonal(gs_new, axis1=1, axis2=2) - np.diagonal(gs_prev, axis1=1, axis2=2)))

    # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
    alphas, betas, _ = get_block_Lanczos_matrices(
        psi0=psi0[:, column_mask],
        h=h,
        converged=converged,
        h_local=h_local,
        verbose=verbose,
        reort_mode=reort,
        build_basis=False,
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
    iw_indices = None
    w_indices = None
    matsubara = iws is not None
    realaxis = ws is not None
    if matsubara:
        iw_indices = np.array(finite.get_job_tasks(rank, ranks, range(len(iws))))
    if realaxis:
        w_indices = np.array(finite.get_job_tasks(rank, ranks, range(len(ws))))
    gs_matsubara_local, gs_realaxis_local = calc_local_Greens_function_from_alpha_beta(
        alphas, betas, iws, ws, iw_indices, w_indices, e, delta, verbose
    )
    # Multiply obtained Green's function with the upper triangular matrix to restore the original block
    # R^T* G R
    if matsubara:
        gs_matsubara = np.zeros((len(iws), r.shape[1], r.shape[1]), dtype=complex)
        if len(iw_indices) > 0:
            gs_matsubara[iw_indices, :, :] = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(
                gs_matsubara_local[iw_indices], r[np.newaxis, :, :]
            )
    if realaxis:
        gs_realaxis = np.zeros((len(ws), r.shape[1], r.shape[1]), dtype=complex)
        if len(w_indices) > 0:
            gs_realaxis[w_indices, :, :] = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(
                gs_realaxis_local[w_indices], r[np.newaxis, :, :]
            )
    if matsubara:
        gs_matsubara = comm.reduce(gs_matsubara, root=0)
    else:
        gs_matsubara = None
    if realaxis:
        gs_realaxis = comm.reduce(gs_realaxis, root=0)
    else:
        gs_realaxis = None
    return gs_matsubara, gs_realaxis


def calc_local_Greens_function_from_alpha_beta(alphas, betas, iws, ws, iw_indices, w_indices, e, delta, verbose):
    I = np.identity(alphas.shape[1], dtype=complex)
    matsubara = iws is not None
    realaxis = ws is not None
    if matsubara:
        iomegaP = iws + e
        gs_matsubara_local = np.zeros((len(iws), alphas.shape[1], alphas.shape[1]), dtype=complex)
        if len(iw_indices) > 0:
            iwIs = iomegaP[iw_indices][:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
            gs_matsubara_local[iw_indices] = iwIs - alphas[-1][np.newaxis, :, :]
    else:
        gs_matsubara_local = None
    if realaxis:
        omegaP = ws + 1j * delta + e
        gs_realaxis_local = np.zeros((len(ws), alphas.shape[1], alphas.shape[1]), dtype=complex)
        if len(w_indices) > 0:
            wIs = omegaP[w_indices][:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
            gs_realaxis_local[w_indices] = wIs - alphas[-1][np.newaxis, :, :]
    else:
        gs_realaxis_local = None

    for alpha, beta in zip(alphas[-2::-1], betas[-2::-1]):
        if matsubara:
            if len(iw_indices) > 0:
                gs_matsubara_local[iw_indices] = (
                    iwIs
                    - alpha[np.newaxis, :, :]
                    - np.conj(beta.T)[np.newaxis, :, :]
                    @ np.linalg.solve(gs_matsubara_local[iw_indices], beta[np.newaxis, :, :])
                )
        if realaxis:
            if len(w_indices) > 0:
                gs_realaxis_local[w_indices] = (
                    wIs
                    - alpha[np.newaxis, :, :]
                    - np.conj(beta.T)[np.newaxis, :, :]
                    @ np.linalg.solve(gs_realaxis_local[w_indices], beta[np.newaxis, :, :])
                )
    return gs_matsubara_local, gs_realaxis_local

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
    return np.einsum("ij,kl, jlmo, mn, op", np.conj(T.T),
                   np.conj(T.T), U4, T, T)


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
