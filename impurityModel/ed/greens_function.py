import numpy as np
import scipy as sp
import time

from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.lanczos import get_block_Lanczos_matrices

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


def get_Greens_function(
    nBaths, matsubara_mesh, omega_mesh, es, psis, l, hOp, delta, restrictions, verbose, mpi_distribute=False
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
        matsubara_mesh,
        omega_mesh,
        delta,
        restrictions,
        krylovSize=None,
        verbose=verbose,
    )
    gsPS_matsubara, gsPS_realaxis = calc_Greens_function_with_offdiag(
        n_spin_orbitals,
        hOp,
        tOpsPS,
        psis,
        es,
        -matsubara_mesh if matsubara_mesh is not None else None,
        -omega_mesh if omega_mesh is not None else None,
        -delta,
        restrictions,
        krylovSize=None,
        verbose=verbose,
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
    iw,
    w,
    delta,
    restrictions=None,
    krylovSize=None,
    slaterWeightMin=1e-12,
    parallelization_mode="H_build",
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
    n = len(es)

    h_mem = {}
    if parallelization_mode == "eigen_states":
        # Green's functions
        gs_matsubara = np.zeros((n, len(tOps), len(tOps), len(iw)), dtype=complex)
        gs_realaxis = np.zeros((n, len(tOps), len(tOps), len(w)), dtype=complex)
        for i in finite.get_job_tasks(rank, ranks, range(len(psis))):
            psi = psis[i]
            e = es[i]

            v = []
            for tOp in tOps:
                v.append(finite.applyOp(n_spin_orbitals, tOp, psi, slaterWeightMin, restrictions, {}))

            gs_matsubara_i, gs_realaxis_i = get_block_Green(
                n_spin_orbitals=n_spin_orbitals,
                hOp=hOp,
                psi_arr=v,
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
        for i, (psi, e) in enumerate(zip(psis, es)):
            v = []
            for i_tOp, tOp in enumerate(tOps):
                v.append(finite.applyOp(n_spin_orbitals, tOp, psi, slaterWeightMin, restrictions, t_mems[i_tOp]))
            gs_matsubara_i, gs_realaxis_i = get_block_Green(
                n_spin_orbitals=n_spin_orbitals,
                hOp=hOp,
                psi_arr=v,
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
            )
            if rank == 0:
                if iw is not None:
                    gs_matsubara[i, :, :, :] = gs_matsubara_i
                if w is not None:
                    gs_realaxis[i, :, :, :] = gs_realaxis_i
    return gs_matsubara, gs_realaxis


def get_block_Green(
    n_spin_orbitals,
    hOp,
    psi_arr,
    e,
    iws,
    ws,
    delta,
    restrictions=None,
    h_mem=None,
    mode="sparse",
    krylovSize=None,
    slaterWeightMin=1e-7,
    parallelization_mode="H_build",
    verbose=True,
):
    matsubara = iws is not None
    realaxis = ws is not None

    if not matsubara and not realaxis:
        print("No Matsubara mesh or real frequency mesh provided. No Greens function will be calculated.")
        return np.array([[[0]]]), np.array([[[0]]])

    if h_mem is None:
        h_mem = {}
    h_local = True

    states = set(key for psi in psi_arr for key in psi.keys())
    h, basis_index = finite.expand_basis_and_hamiltonian(
        n_spin_orbitals,
        h_mem,
        hOp,
        sorted(list(states)),
        restrictions,
        parallelization_mode=parallelization_mode,
        return_h_local=h_local,
        verbose=True,
    )

    N = len(basis_index)
    n = len(psi_arr)

    import numpy as np

    psi_start = np.zeros((N, n), dtype=complex)
    for i, psi in enumerate(psi_arr):
        for ps, amp in psi.items():
            psi_start[basis_index[ps], i] = amp

    # Do a QR decomposition of the starting block.
    # Later on, use r to restore the block corresponding to
    # psi_start
    psi0, r = np.linalg.qr(psi_start, mode="reduced")
    # The QR decomposition can, in some rare instances, reduce the number of
    # independent vectors in our block, m is the number of linearly independent
    # vectors in the block. The full block structure is obtained by multiplying
    # with r
    m = psi0.shape[1]

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
    n_samples = 50
    weights = gaussian(conv_w, mu=0, sigma=0.5)
    weights /= np.sum(weights)

    def converged(alphas, betas):
        if alphas.shape[0] == 1:
            return False

        w = np.random.choice(conv_w, size=min(n_samples, len(conv_w)), p=weights, replace=False)
        wIs = (w + 1j * delta_p + e)[:, np.newaxis, np.newaxis] * np.identity(alphas.shape[1], dtype=complex)[
            np.newaxis, :, :
        ]
        gs_new = wIs - alphas[-1]  # [np.newaxis, :, :]
        for alpha, beta in zip(alphas[-2::-1], betas[-2::-1]):
            # gs_new = wIs - alpha[np.newaxis, :, :] - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve( gs_new, beta[np.newaxis, :, :])
            gs_new = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_new, beta[np.newaxis, :, :])

        gs_prev = wIs - alphas[-2][np.newaxis, :, :]
        for alpha, beta in zip(alphas[-3::-1], betas[-3::-1]):
            # gs_prev = wIs - alpha[np.newaxis, :, :] - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve( gs_prev, beta[np.newaxis, :, :])
            gs_prev = wIs - alpha - np.conj(beta.T)[np.newaxis, :, :] @ np.linalg.solve(gs_prev, beta[np.newaxis, :, :])

        return np.all(np.abs(gs_new - gs_prev) < 1e-12)

    # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
    alphas, betas = get_block_Lanczos_matrices(
        psi0=psi0,
        h=h,
        n_spin_orbitals=n_spin_orbitals,
        slaterWeightMin=slaterWeightMin,
        restrictions=restrictions,
        krylovSize=krylovSize,
        converged=converged,
        h_local=h_local,
        mode=mode,
        verbose=verbose,
    )
    I = np.identity(alphas.shape[1], dtype=complex)
    if matsubara:
        iomegaP = iws + e
        # Parallelize over omega mesh
        iw_indices = np.array(finite.get_job_tasks(rank, ranks, range(len(iomegaP))))
        iwIs = iomegaP[iw_indices][:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
        gs_matsubara_local = np.zeros((len(iws), m, m), dtype=complex)
        gs_matsubara_local[iw_indices] = iwIs - alphas[-1][np.newaxis, :, :]
    if realaxis:
        omegaP = ws + 1j * delta + e
        # Parallelize over omega mesh
        w_indices = np.array(finite.get_job_tasks(rank, ranks, range(len(omegaP))))
        wIs = omegaP[w_indices][:, np.newaxis, np.newaxis] * I[np.newaxis, :, :]
        gs_realaxis_local = np.zeros((len(ws), m, m), dtype=complex)
        gs_realaxis_local[w_indices] = wIs - alphas[-1][np.newaxis, :, :]

    if verbose and rank == 0:
        t0 = time.perf_counter()

    # for alpha, beta in zip(reversed(alphas), reversed(betas)):
    for alpha, beta in zip(alphas[-2::-1], betas[-2::-1]):
        if matsubara:
            gs_matsubara_local[iw_indices] = (
                iwIs
                - alpha[np.newaxis, :, :]
                - np.conj(beta.T)[np.newaxis, :, :]
                @ np.linalg.solve(gs_matsubara_local[iw_indices], beta[np.newaxis, :, :])
            )
        if realaxis:
            gs_realaxis_local[w_indices] = (
                wIs
                - alpha[np.newaxis, :, :]
                - np.conj(beta.T)[np.newaxis, :, :]
                @ np.linalg.solve(gs_realaxis_local[w_indices], beta[np.newaxis, :, :])
            )
    # Multiply obtained Green's function with the upper triangular matrix to restore the original block
    # R^T* G R
    if matsubara:
        gs_matsubara = np.zeros((len(iws), n, n), dtype=complex)
        gs_matsubara[iw_indices, :, :] = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(
            gs_matsubara_local[iw_indices], r[np.newaxis, :, :]
        )
    if realaxis:
        gs_realaxis = np.zeros((len(ws), n, n), dtype=complex)
        gs_realaxis[w_indices, :, :] = np.conj(r.T)[np.newaxis, :, :] @ np.linalg.solve(
            gs_realaxis_local[w_indices], r[np.newaxis, :, :]
        )
    # Reduce Green's function to rank 0
    if matsubara:
        gs_matsubara = comm.reduce(gs_matsubara, root=0)
        if rank == 0:
            gs_matsubara = np.moveaxis(gs_matsubara, 0, -1)
    if realaxis:
        gs_realaxis = comm.reduce(gs_realaxis, root=0)
        if rank == 0:
            gs_realaxis = np.moveaxis(gs_realaxis, 0, -1)
    if verbose and rank == 0:
        print(f"time(G_from_alpha_beta) = {time.perf_counter() - t0} seconds.")
    return gs_matsubara, gs_realaxis


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

    if rank == 0:
        print(f"Writing {axis_label} {label} to files")
        with open(f"real-{axis_label}-{label}.dat", "w") as fg_real, open(
            f"imag-{axis_label}-{label}.dat", "w"
        ) as fg_imag:
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
