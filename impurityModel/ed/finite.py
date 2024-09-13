"""
This module contains functions doing the bulk of the calculations.
"""

from math import pi, sqrt
import numpy as np
from sympy.physics.wigner import gaunt
import itertools
from collections import OrderedDict
import scipy.sparse
from mpi4py import MPI
import time
from multiprocessing import Process, Queue, current_process, freeze_support
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from os import environ

try:
    from petsc4py import PETSc
    from slepc4py import SLEPc
    from slepc4py.SLEPc import EPS
except ModuleNotFoundError:
    pass

# Local imports
from impurityModel.ed import product_state_representation as psr
from impurityModel.ed import create
from impurityModel.ed import remove
from impurityModel.ed.average import k_B, thermal_average, thermal_average_scale_indep

from scipy.sparse.linalg import ArpackNoConvergence, ArpackError, eigsh
from scipy.linalg import qr


# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


def get_job_tasks(rank, ranks, tasks_tot):
    """
    Return a tuple of job task indices for a particular rank.

    This function distribute the job tasks in tasks_tot
    over all the ranks.

    Note
    ----
    This is a primerly a MPI help function.

    Parameters
    ----------
    rank : int
        Current MPI rank/worker.
    ranks : int
        Number of MPI ranks/workers in total.
    tasks_tot : list
        List of task indices.
        Length is the total number of job tasks.

    """
    n_tot = len(tasks_tot)
    nj = n_tot // ranks
    rest = n_tot % ranks
    tasks = [tasks_tot[i] for i in range(nj * rank, nj * rank + nj)]
    if rank < rest:
        tasks.append(tasks_tot[n_tot - rest + rank])
    return tuple(tasks)


def rotate_matrix(M, T):
    r"""
    Rotate the matrix, M, using the matrix T.
    Returns :math:`M' = T^{\dagger} M T`
    Parameters
    ==========
    M : NDArray - Matrix to rotate
    T : NDArray - Rotation matrix to use
    Returns
    =======
    M' : NDArray - The rotated matrix
    """
    return np.conj(T.T) @ M @ T


def setup_hamiltonian(
    n_spin_orbitals,
    hOp,
    basis,
    verbose=False,
    mode="sparse",
):
    if verbose:
        print("Create Hamiltonian matrix...")
    if mode != "sparse":
        h = get_hamiltonian_matrix(n_spin_orbitals, hOp, basis, verbose=verbose)
        nonzero = len(h.nonzero()[0])
    elif mode == "sparse":
        h_local, h_dict, expanded_basis = expand_basis_and_build_hermitian_hamiltonian_new(
            n_spin_orbitals,
            {},
            hOp,
            basis,
            restrictions=basis.restrictions,
            verbose=verbose,
            parallelization_mode="H_build",
            return_h_local=True,
        )

        nonzero = comm.reduce(h_local.nnz, root=0, op=MPI.SUM)
    if verbose:
        print(f"h_local :\n{h_local}")
        print(f"<#Hamiltonian elements/column> = {int(nonzero / len(expanded_basis))}")
    return expanded_basis, h_dict, h_local


def mpi_matmul(h_local, comm):
    """
    MPI parallelized matrix multiplication.
    Each rank has a number of columns of the matrix and the full vector.
    """

    def matmat(m):
        if len(m.shape) == 1:
            m = m.reshape((m.shape[0], 1))
        n_cols = m.shape[1]
        res = np.empty((h_local.shape[0], n_cols), dtype=np.result_type(h_local.dtype, m.dtype))
        comm.Allreduce(h_local @ m, res, op=MPI.SUM)
        return res

    return matmat


def eigensystem_new(
    h_local,
    e_max,
    k=10,
    v0=None,
    eigenValueTol=0,
    return_eigvecs=True,
):
    """
    Return eigen-energies and eigenstates.

    Parameters
    ----------
    h_local : HermitianOperator object
        Contains part of the full many-body Hamiltonian, local to this MPI rank.
    e_max : float
        Maximum energy difference for excited states
    k : int
        Calculate at least k eigenstates above e_max, helps ensure convergence of eigenvalues and eigenstates.
    eigenValueTol : float
        The precision of the returned eigenvalues.
    return_eigvecs : bool
        If True, return eigenvalues and eigenvectors for all states with energy within e_max of the lowest energy state.
        If False, return only the calculated eigenvalues.
    """

    t0 = time.perf_counter()
    if isinstance(h_local, np.ndarray):
        if h_local.shape[0] == 0:
            return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=h_local.dtype)
        if comm.rank == 0:
            es, vecs = np.linalg.eigh(h_local, UPLO="L")
        else:
            es = np.empty((h_local.shape[0],))
            vecs = np.empty_like(h_local)
        comm.Bcast(es, root=0)
        comm.Bcast(vecs, root=0)
        mask = es - es[0] <= e_max
    elif isinstance(h_local, scipy.sparse._csr.csr_matrix) or isinstance(h_local, scipy.sparse._csc.csc_matrix):
        h = scipy.sparse.linalg.LinearOperator(
            (h_local.shape[0], h_local.shape[0]),
            matvec=mpi_matmul(h_local, comm),
            rmatvec=mpi_matmul(h_local, comm),
            matmat=mpi_matmul(h_local, comm),
            rmatmat=mpi_matmul(h_local, comm),
            dtype=h_local.dtype,
        )

        dk = 1
        v0_guess = v0[:, 0] if v0 is not None else None
        es = []
        mask = [True]
        ncv = None
        while len(es) <= sum(mask):
            try:
                es, vecs = eigsh(
                    h,
                    k=min(k + dk, h.shape[0] - 2),
                    which="SA",
                    tol=eigenValueTol,
                    v0=v0_guess,
                    ncv=ncv,
                )
            except ArpackNoConvergence:
                eigenValueTol = max(np.sqrt(eigenValueTol), 1e-6)
                dk = 1
                vecs = None
                es = []
                mask = [True]
                continue
            except ArpackError:
                eigenValueTol = max(np.sqrt(eigenValueTol), 1e-6)
                if ncv is None:
                    ncv = min(h.shape[0], max(5 * (k + dk) + 1, 50))
                else:
                    ncv = min(h.shape[0], max(2 * (k + dk) + 1, 2 * ncv))
                vecs = None
                es = []
                mask = [True]
                continue
            mask = es - np.min(es) <= e_max
            dk += k
            v0_guess = vecs[:, mask][:, 0]
    elif isinstance(h_local, PETSc.Mat):
        dk = 1
        es = []
        mask = [True]

        eig_solver = SLEPc.EPS()
        eig_solver.create()
        eig_solver.setOperators(h_local, None)
        eig_solver.setProblemType(SLEPc.EPS.ProblemType.HEP)
        eig_solver.setWhichEigenpairs(EPS.Which.SMALLEST_REAL)
        eig_solver.setTolerances(tol=max(eigenValueTol, np.finfo(float).eps))
        while len(es) - sum(mask) <= 0:
            eig_solver.setDimensions(k + dk, PETSc.DECIDE)
            eig_solver.solve()
            nconv = eig_solver.getConverged()
            es = np.empty((nconv), dtype=float)
            if nconv > 0:
                for i in range(nconv):
                    es[i] = eig_solver.getEigenvalue(i).real
                mask = es - np.min(es) <= e_max
                dk += k
        vecs = None
        if nconv > 0:
            vecs = np.empty((h_local.size[0], nconv), dtype=complex) if comm.rank == 0 else None
            vr, wr = h_local.getVecs()
            vi, wi = h_local.getVecs()
            for i in range(nconv):
                _ = eig_solver.getEigenpair(i, vr, vi)
                offsets = vr.owner_ranges[:-1]
                counts = [vr.owner_ranges[i] - vr.owner_ranges[i - 1] for i in range(1, len(vr.owner_ranges))]
                v_real = np.empty((h_local.size[0]), dtype=complex)
                v_imag = np.empty((h_local.size[0]), dtype=complex)
                comm.Gatherv(vr.array_r, [v_real, counts, offsets, MPI.DOUBLE_COMPLEX], root=0)
                comm.Gatherv(vi.array_r, [v_imag, counts, offsets, MPI.DOUBLE_COMPLEX], root=0)
                if comm.rank == 0:
                    vecs[:, i] = v_real + 1j * v_imag
        vecs = comm.bcast(vecs, root=0)
    indices = np.argsort(es)
    es = es[indices]
    vecs = vecs[:, indices]
    mask = es - np.min(es) <= e_max
    t0 = time.perf_counter() - t0

    if not return_eigvecs:
        return es[: sum(mask)]

    # the scipy eigsh function does not guarantee that degenerate eigenvalues get orthogonal eigenvectors.
    if isinstance(h_local, scipy.sparse._csr.csr_matrix) or isinstance(h_local, scipy.sparse._csc.csc_matrix):
        vecs[:, : sum(mask)], _ = qr(vecs[:, : sum(mask)], mode="economic", overwrite_a=True, check_finite=False)

    t0 = time.perf_counter() - t0

    return es[: sum(mask)], vecs[:, : sum(mask)]


def eigensystem(
    n_spin_orbitals,
    hOp,
    basis,
    nPsiMax,
    groundDiagMode="Lanczos",
    eigenValueTol=1e-12,
    slaterWeightMin=1e-12,
    verbose=True,
    lock=None,
):
    """
    Return eigen-energies and eigenstates.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    hOp : dict
        tuple : float or complex
        The Hamiltonian operator to diagonalize.
        Each keyword contains ordered instructions
        where to add or remove electrons.
        Values indicate the strengths of
        the corresponding processes.
    basis : tuple
        All product states included in the basis.
    nPsiMax : int
        Number of eigenvalues to find.
    groundDiagMode : str
        'Lanczos' or 'full' diagonalization.
    eigenValueTol : float
        The precision of the returned eigenvalues.
    slaterWeightMin : float
        Minimum product state weight for product states to be kept.

    """
    if rank == 0 and verbose:
        print("Create Hamiltonian matrix...")
    h = get_hamiltonian_matrix(n_spin_orbitals, hOp, basis, verbose=verbose)
    if rank == 0 and verbose:
        print("Checking if Hamiltonian is Hermitian!")
        err_max = np.max(np.abs(np.conj(h.T) - h))
        if err_max > 1e-12:
            print(f"Warning! Hamiltonian matrix is not very Hermitian!\nLargest error = {err_max}")
        else:
            print("Hamiltonian matrix is Hermitian!")
        print("<#Hamiltonian elements/column> = {:d}".format(int(len(np.nonzero(h)[0]) / len(basis))))
        print("Diagonalize the Hamiltonian...")
    if groundDiagMode == "full":
        es, vecs = np.linalg.eigh(h.todense())
        es = es[:nPsiMax]
        vecs = vecs[:, :nPsiMax]
    elif groundDiagMode == "Lanczos":
        es, vecs = scipy.sparse.linalg.eigsh(h, k=nPsiMax, which="SA", tol=eigenValueTol)
        # Sort the eigenvalues and eigenvectors in ascending order.
        indices = np.argsort(es)
        es = np.array([es[i] for i in indices])
        vecs = np.array([vecs[:, i] for i in indices]).T
    else:
        print(f"Unknown diagonalization mode: {groundDiagMode}")
    if rank == 0 and verbose:
        V = np.array([ev / np.linalg.norm(ev) for ev in vecs.T]).T
        err_max = np.max(np.abs(np.conj(V.T) @ V - np.eye(V.shape[1])))
        if err_max > 1e-12:
            print(f"Warning! Obtained eigenvectors are not very orthogonal!\nMaximum overlap {err_max}")

        print(f"Proceed with {len(es)} eigenstates.\n")

    psis = [
        ({basis[i]: vecs[i, vi] for i in range(len(basis)) if slaterWeightMin <= abs(vecs[i, vi]) ** 2})
        for vi in range(len(es))
    ]
    return es, psis


def printSlaterDeterminantsAndWeights(psis, nPrintSlaterWeights):
    print("Slater determinants/product states and correspoinding weights")
    weights = []
    for i, psi in enumerate(psis):
        print("Eigenstate {:d}.".format(i))
        print("Consists of {:d} product states.".format(len(psi)))
        ws = np.array([abs(a) ** 2 for a in psi.values()])
        s = np.array(list(psi.keys()))
        j = np.argsort(ws)
        ws = ws[j[-1::-1]]
        s = s[j[-1::-1]]
        weights.append(ws)
        if nPrintSlaterWeights > 0:
            print("Highest (product state) weights:")
            print(ws[:nPrintSlaterWeights])
            print("Corresponding product states:")
            print(s[:nPrintSlaterWeights])
            print("")


def printExpValues(rhos, es, rot_to_spherical):
    """
    print several expectation values, e.g. E, N, L^2.
    """
    if rank == 0:
        print("E0 = {:7.4f}".format(es[0]))
        print(
            # "{:^3s} {:>11s} {:>8s} {:>8s} {:>8s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
            "{:^3s} {:>11s} {:>8s} {:>8s} {:>8s} {:>9s} {:>9s}".format(
                "i",
                "E-E0",
                "N",
                "N(Dn)",
                "N(Up)",
                "Lz",
                "Sz",
                # "L^2",
                # "S^2",
            )
        )
    #        print(('  i  E-E0  N(3d) N(egDn) N(egUp) N(t2gDn) '
    #               'N(t2gUp) Lz(3d) Sz(3d) L^2(3d) S^2(3d)'))
    if rank == 0:
        for i, (e, rho) in enumerate(zip(es - es[0], rhos)):
            rho_spherical = rotate_matrix(rho, rot_to_spherical)
            N, Ndn, Nup = get_occupations_from_rho_spherical(rho_spherical)
            print(
                # ("{:3d} {:11.8f} {:8.5f} {:8.5f} {:8.5f}" " {: 9.6f} {: 9.6f} {:9.5f} {:9.5f}").format(
                ("{:3d} {:11.8f} {:8.5f} {:8.5f} {:8.5f}" " {: 9.6f} {: 9.6f}").format(
                    i,
                    e,
                    N,
                    Ndn,
                    Nup,
                    get_Lz_from_rho_spherical(rho_spherical),
                    get_Sz_from_rho_spherical(rho_spherical),
                    # get_L_from_rho_spherical(rho_spherical),
                    # get_S_from_rho_spherical(rho_spherical),
                )
            )
        print("\n", flush=True)


def get_occupations_from_rho_spherical(rho):
    """
    Calculate the (spin polarized) occupation from the density matrix.
    """
    n_orbs = rho.shape[0]
    return (
        np.real(np.trace(rho)),
        np.real(np.trace(rho[: n_orbs // 2, : n_orbs // 2])),
        np.real(np.trace(rho[n_orbs // 2 :, n_orbs // 2 :])),
    )


def get_Lz_from_rho_spherical(rho):
    l = (rho.shape[0] // 2 - 1) // 2
    return np.real(
        sum(ml * (rho[i, i] + rho[i + (2 * l + 1), i + (2 * l + 1)]) for i, ml in enumerate(range(-l, l + 1)))
    )


def get_Lplus_from_rho_spherical(rho, l):
    # L+ |l, ml> = sqrt(l*(l+1) - ml*(ml+1))|l, ml+1>
    llp1 = l * (l + 1)
    #   L+    |2, -2>,  |2, -1>, |2,  0>, |2,  1>, |2,  2>
    # <2, -2|    0         0        0        0        0
    # <2, -1| sqrt(8)      0        0        0        0
    # <2,  0|    0      sqrt(6)     0        0        0
    # <2,  1|    0         0     sqrt(6)     0        0
    # <2,  2|    0         0        0     sqrt(8)     0
    Lplus = np.diag([np.sqrt(llp1 - ml * (ml + 1)) for ml in range(-l, l)], k=-1)
    return np.trace(
        rho @ np.block([[Lplus, np.zeros((2 * l + 1, 2 * l + 1))], [np.zeros((2 * l + 1, 2 * l + 1)), Lplus]])
    )


def get_Sminus_from_rho_spherical(rho, l, s=1 / 2):
    # S+ |s, ms> = sqrt(s*(s+1) - ms*(ms+1))|s, ms+1>
    ssp1 = s * (s + 1)
    ms = +1 / 2
    #   S-      |1/2,-1/2>,  |1/2, 1/2>
    # <1/2,-1/2|    0            1
    # <1/2, 1/2|    0            0
    # S- = [[0   S-],
    #        0   0 ]]
    Sminus = np.diag(np.repeat(np.sqrt(ssp1 - ms * (ms - 1)), 2 * l), k=1)
    return np.trace(
        rho
        @ np.block(
            [
                [np.zeros((2 * l + 1, 2 * l + 1)), Sminus],
                [np.zeros((2 * l + 1, 2 * l + 1)), np.zeros((2 * l + 1, 2 * l + 1))],
            ]
        )
    )


def get_Lminus_from_rho_spherical(rho, l):
    # L- |l, ml> = sqrt(l*(l+1) - ml*(ml-1))|l, ml-1>
    llp1 = l * (l + 1)
    #   L+    |2, -2>,  |2, -1>, |2,  0>, |2,  1>, |2,  2>
    # <2, -2|    0      sqrt(4)     0        0        0
    # <2, -1|    0         0     sqrt(6)     0        0
    # <2,  0|    0         0        0     sqrt(6)     0
    # <2,  1|    0         0        0        0     sqrt(4)
    # <2,  2|    0         0        0        0        0
    Lminus = np.diag([np.sqrt(llp1 - ml * (ml - 1)) for ml in range(-l + 1, l + 1)], k=1)
    return np.trace(
        rho @ np.block([[Lminus, np.zeros((2 * l + 1, 2 * l + 1))], [np.zeros((2 * l + 1, 2 * l + 1)), Lminus]])
    )


def get_Splus_from_rho_spherical(rho, l, s=1 / 2):
    # S+ |s, ms> = sqrt(s*(s+1) - ms*(ms+1))|s, ms+1>
    ssp1 = s * (s + 1)
    ms = -1 / 2
    #   S+      |1/2,-1/2>,  |1/2, 1/2>
    # <1/2,-1/2|    0            0
    # <1/2, 1/2|    1            0
    # S+ = [[0   0],
    #        S+  0]]
    Splus = np.diag(np.repeat(np.sqrt(ssp1 - ms * (ms + 1)), 2 * l), k=-1)
    return np.trace(
        rho
        @ np.block(
            [
                [np.zeros((2 * l + 1, 2 * l + 1)), np.zeros((2 * l + 1, 2 * l + 1))],
                [Splus, np.zeros((2 * l + 1, 2 * l + 1))],
            ]
        )
    )


def get_L_from_rho_spherical(rho, l):
    return np.sqrt(
        (0.5 * (get_Lplus_from_rho_spherical(rho, l) + get_Lminus_from_rho_spherical(rho, l))) ** 2
        + (-1j / 2 * (get_Lplus_from_rho_spherical(rho, l) - get_Lminus_from_rho_spherical(rho, l))) ** 2
        + get_Lz_from_rho_spherical(rho, l) ** 2
    )


def get_S_from_rho_spherical(rho, l, s):
    return np.sqrt(
        (0.5 * (get_Splus_from_rho_spherical(rho, l, s=s) + get_Sminus_from_rho_spherical(rho, l, s=s)) ** 2)
        + (-1j / 2 * (get_Splus_from_rho_spherical(rho, l, s=s) - get_Sminus_from_rho_spherical(rho, l, s=s)) ** 2)
        + get_Sz_from_rho_spherical(rho, l) ** 2
    )


def get_L2_from_rho_spherical(rho):
    l = (rho.shape[0] // 2 - 1) // 2
    llp1 = l * (l + 1)
    Lz = get_Lz_from_rho_spherical(rho)
    Lplus = np.zeros((2 * (2 * l + 1), 2 * (2 * l + 1)))
    for i, ml in enumerate(range(-l, l)):
        Lplus[i + 1, i] = np.sqrt(llp1 + ml * (ml + 1))
    Lminus = np.zeros((2 * (2 * l + 1), 2 * (2 * l + 1)))
    for i, ml in enumerate(range(-l + 1, l + 1)):
        Lminus[i - 1, i] = np.sqrt(llp1 + ml * (ml - 1))
    Lz2 = np.identity(2 * (2 * l + 1))
    for i, ml in enumerate(range(-l, l + 1)):
        Lz2[i, i] = ml**2
    return np.trace(rho @ Lz2) + 2 * Lz + np.trace(rho @ Lplus @ Lminus)


def get_Sz_from_rho_spherical(rho):
    l = (rho.shape[0] // 2 - 1) // 2
    return 1 / 2 * np.real(sum(-rho[i, i] + rho[i + (2 * l + 1), i + (2 * l + 1)] for i in range(2 * l + 1)))


def get_S2_from_rho_spherical(rho):
    l = (rho.shape[0] // 2 - 1) // 2
    ssp1 = 3 / 4
    Sz = get_Sz_from_rho_spherical(rho)
    Splus = np.zeros((2 * (2 * l + 1), 2 * (2 * l + 1)))
    for i, ms in enumerate(np.repeat([-0.5], 2 * l + 1)):
        Splus[i + 2 * l + 1, i] = np.sqrt(ssp1 + ms * (ms + 1))
    Sminus = np.zeros((2 * (2 * l + 1), 2 * (2 * l + 1)))
    for i, ms in enumerate(np.repeat([0.5], 2 * l + 1)):
        Sminus[i, i + 2 * l + 1] = np.sqrt(ssp1 + ms * (ms - 1))
    Sz2 = np.identity(2 * (2 * l + 1))
    return np.trace(rho @ Sz2) + 2 * Sz + np.trace(rho @ Splus @ Sminus)


def printThermalExpValues_new(rhos, es, tau, rot_to_spherical):
    """
    print several thermal expectation values, e.g. E, N, Sz, Lz.

    cutOff - float. Energies more than cutOff*kB*T above the
            lowest energy is not considered in the average.
    """
    e = es - es[0]
    rho_thermal = thermal_average_scale_indep(es, rhos, tau)
    rho_thermal_spherical = rotate_matrix(rho_thermal, rot_to_spherical)
    N, Ndn, Nup = get_occupations_from_rho_spherical(rho_thermal_spherical)
    print("<E-E0> = {:8.7f}".format(thermal_average_scale_indep(e, e, tau=tau)))
    print("<N(3d)> = {:8.7f}".format(N))
    print("<N(Dn)> = {:8.7f}".format(Ndn))
    print("<N(Up)> = {:8.7f}".format(Nup))
    print("<Lz> = {:8.7f}".format(get_Lz_from_rho_spherical(rho_thermal_spherical)))
    print("<Sz> = {:8.7f}".format(get_Sz_from_rho_spherical(rho_thermal_spherical)))
    # print("<L> = {:8.7f}".format(get_L_from_rho_spherical(rho_thermal_spherical)))
    # print("<S> = {:8.7f}".format(get_S_from_rho_spherical(rho_thermal_spherical)))


def printThermalExpValues(nBaths, es, psis, T=300, cutOff=10):
    """
    print several thermal expectation values, e.g. E, N, L^2.

    cutOff - float. Energies more than cutOff*kB*T above the
            lowest energy is not considered in the average.
    """
    e = es - es[0]
    # Select relevant energies
    mask = e < cutOff * k_B * T
    e = e[mask]
    psis = np.array(psis)[mask]
    occs = thermal_average(e, np.array([getEgT2gOccupation(nBaths, psi) for psi in psis]), T=T)
    if rank == 0:
        print("<E-E0> = {:8.7f}".format(thermal_average(e, e, T=T)))
        print("<N(3d)> = {:8.7f}".format(thermal_average(e, [getTraceDensityMatrix(nBaths, psi) for psi in psis], T=T)))
        print("<N(egDn)> = {:8.7f}".format(occs[0]))
        print("<N(egUp)> = {:8.7f}".format(occs[1]))
        print("<N(t2gDn)> = {:8.7f}".format(occs[2]))
        print("<N(t2gUp)> = {:8.7f}".format(occs[3]))
        print("<Lz(3d)> = {:8.7f}".format(thermal_average(e, [getLz3d(nBaths, psi) for psi in psis], T=T)))
        print("<Sz(3d)> = {:8.7f}".format(thermal_average(e, [getSz3d(nBaths, psi) for psi in psis], T=T)))
        # print("<L^2(3d)> = {:8.7f}".format(thermal_average(e, [getLsqr3d(nBaths, psi) for psi in psis], T=T)))
        # print("<S^2(3d)> = {:8.7f}".format(thermal_average(e, [getSsqr3d(nBaths, psi) for psi in psis], T=T)))


def dc_MLFT(n3d_i, c, Fdd, n2p_i=None, Fpd=None, Gpd=None):
    r"""
    Return double counting (DC) in multiplet ligand field theory.

    Parameters
    ----------
    n3d_i : int
        Nominal (integer) 3d occupation.
    c : float
        Many-body correction to the charge transfer energy.
    n2p_i : int
        Nominal (integer) 2p occupation.
    Fdd : list
        Slater integrals {F_{dd}^k}, k \in [0,1,2,3,4]
    Fpd : list
        Slater integrals {F_{pd}^k}, k \in [0,1,2]
    Gpd : list
        Slater integrals {G_{pd}^k}, k \in [0,1,2,3]

    Notes
    -----
    The `c` parameter is related to the charge-transfer
    energy :math:`\Delta_{CT}` by:

    .. math:: \Delta_{CT} = (e_d-e_b) + c.

    """
    if not int(n3d_i) == n3d_i:
        raise ValueError("3d occupation should be an integer")
    if n2p_i is not None and int(n2p_i) != n2p_i:
        raise ValueError("2p occupation should be an integer")

    # Average repulsion energy defines Udd and Upd
    Udd = Fdd[0] - 14.0 / 441 * (Fdd[2] + Fdd[4])
    if n2p_i is None and Fpd is None and Gpd is None:
        return {2: Udd * n3d_i - c}
    if n2p_i == 6 and Fpd is not None and Gpd is not None:
        Upd = Fpd[0] - (1 / 15.0) * Gpd[1] - (3 / 70.0) * Gpd[3]
        return {2: Udd * n3d_i + Upd * n2p_i - c, 1: Upd * (n3d_i + 1) - c}
    else:
        raise ValueError("double counting input wrong.")


def get_spherical_2_cubic_matrix(spinpol=False, l=2):
    r"""
    Return unitary ndarray for transforming from spherical to cubic harmonics.

    Parameters
    ----------
    spinpol : boolean
        If transformation involves spin.
    l : integer
        Angular momentum number. p: l=1, d: l=2.

    Returns
    -------
    u : (M,M) ndarray
        The unitary matrix from spherical to cubic harmonics.

    Notes
    -----
    Element :math:`u_{i,j}` represents the contribution of spherical
    harmonics :math:`i` to the cubic harmonic :math:`j`:

    .. math:: \lvert l_j \rangle  = \sum_{i=0}^4 u_{d,(i,j)}
        \lvert Y_{d,i} \rangle.

    """
    if l == 1:
        # u = np.zeros((3,3),dtype=complex)
        u = np.zeros((3, 3), dtype=complex)
        u[0, 0] = 1j / np.sqrt(2)
        u[2, 0] = 1j / np.sqrt(2)
        u[0, 1] = 1 / np.sqrt(2)
        u[2, 1] = -1 / np.sqrt(2)
        u[1, 2] = 1
    elif l == 2:
        # u = np.zeros((5,5),dtype=complex)
        u = np.zeros((5, 5), dtype=complex)
        u[2, 0] = 1
        u[[0, -1], 1] = 1 / np.sqrt(2)
        u[1, 2] = -1j / np.sqrt(2)
        u[-2, 2] = -1j / np.sqrt(2)
        u[1, 3] = 1 / np.sqrt(2)
        u[-2, 3] = -1 / np.sqrt(2)
        u[0, 4] = 1j / np.sqrt(2)
        u[-1, 4] = -1j / np.sqrt(2)
    if spinpol:
        n, m = np.shape(u)
        # U = np.zeros((2*n,2*m),dtype=complex)
        U = np.zeros((2 * n, 2 * m), dtype=complex)
        U[0:n, 0:m] = u
        U[n:, m:] = u
        u = U
    return u


def daggerOp(op):
    """
    return op^dagger
    """
    opDagger = {}
    for process, value in op.items():
        processNew = []
        for e in process[::-1]:
            if e[1] == "a":
                processNew.append((e[0], "c"))
            elif e[1] == "c":
                processNew.append((e[0], "a"))
            else:
                raise Exception("Operator type unknown: {}".format(e[1]))
        processNew = tuple(processNew)
        opDagger[processNew] = value.conjugate()
    return opDagger


def get_basis(nBaths, valBaths, dnValBaths, dnConBaths, dnTol, n0imp, verbose=True):
    """
    Return restricted basis of product states.

    Parameters
    ----------
    nBaths : ordered dict
    valBaths : ordered dict
    dnValBaths : ordered dict
    dnConBaths : ordered dict
    dnTol : ordered dict
    n0imp : ordered dict

    """
    # Sanity check
    for l in nBaths.keys():
        assert valBaths[l] <= nBaths[l]

    # For each partition, create all configurations
    # given the occupation in that partition.
    basisL = {}
    for l in nBaths.keys():
        if rank == 0 and verbose:
            print("l=", l)
        # Add configurations to this list
        basisL[l] = []
        # Loop over different occupation partitions
        for dnVal in range(dnValBaths[l] + 1):
            for dnCon in range(dnConBaths[l] + 1):
                deltaNimp = dnVal - dnCon
                if abs(deltaNimp) <= dnTol[l] and n0imp[l] + deltaNimp <= 2 * (2 * l + 1):
                    nImp = n0imp[l] + deltaNimp
                    nVal = valBaths[l] - dnVal
                    nCon = dnCon

                    if rank == 0 and verbose:
                        print("New partition occupations:")
                    # if rank == 0:
                    #    print('nImp,dnVal,dnCon = {:d},{:d},{:d}'.format(
                    #        nImp,dnVal,dnCon))
                    if rank == 0 and verbose:
                        print("New partition occupations:")
                        print("nImp,nVal,nCon = {:d},{:d},{:d}".format(nImp, nVal, nCon))
                    # Impurity electron indices
                    indices = [c2i(nBaths, (l, s, m)) for s in range(2) for m in range(-l, l + 1)]
                    basisImp = tuple(itertools.combinations(indices, nImp))
                    # Valence bath electrons
                    if valBaths[l] == 0:
                        # One way of having zero electrons
                        # in zero spin-orbitals
                        basisVal = ((),)
                    else:
                        # Valence bath state indices
                        indices = [c2i(nBaths, (l, b)) for b in range(valBaths[l])]
                        basisVal = tuple(itertools.combinations(indices, nVal))
                    # Conduction bath electrons
                    if nBaths[l] - valBaths[l] == 0:
                        # One way of having zero electrons
                        # in zero spin-orbitals
                        basisCon = ((),)
                    else:
                        # Conduction bath state indices
                        indices = [c2i(nBaths, (l, b)) for b in range(valBaths[l], nBaths[l])]
                        basisCon = tuple(itertools.combinations(indices, nCon))
                    # Concatenate partitions
                    for bImp in basisImp:
                        for bVal in basisVal:
                            for bCon in basisCon:
                                basisL[l].append(bImp + bVal + bCon)
                                # print (f"basis state : {bImp+bVal+bCon}")
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    basis = []
    for configuration in itertools.product(*basisL.values()):
        # Convert product state representation from a tuple to a object
        # of the class bytes. Then add this product state to the basis.
        basis.append(psr.tuple2bytes(tuple(sorted(itertools.chain.from_iterable(configuration))), n_spin_orbitals))
    # return tuple(sorted(basis))
    return list(sorted(basis))


def printOp(nBaths, pOp, printstr):
    print(printstr)
    a = arrayOp(nBaths, pOp)
    print(np.array2string(a, max_line_width=2000, threshold=1000, precision=3, suppress_small=True))
    print("Eigenvalues: ")
    print(np.array_str(np.linalg.eigvalsh(a), max_line_width=368, precision=3, suppress_small=True))
    print()


def inner(a: dict, b: dict) -> complex:
    r"""
    Return :math:`\langle a | b \rangle`

    Parameters
    ----------
    a : dict
        Multi configurational state
    b : dict
        Multi configurational state

    Acknowledgement: Written entirely by Petter Saterskog
    """
    acc = 0
    for state, amp in b.items():
        acc += np.conj(a.get(state, 0)) * amp
    return acc


def matmul(psis: list[dict], mat: np.ndarray) -> list[dict]:
    n = len(psis)
    assert mat.shape == (n, n)
    res = [{} for _ in psis]
    for j, (i, psi_i) in itertools.product(range(n), enumerate(psis)):
        addToFirst(res[j], psi_i, mat[i, j])
    return res


def removeFromFirst(psi1, psi2, mul=1):
    r"""
    From state :math:`|\psi_1\rangle`, remove  :math:`mul * |\psi_2\rangle`.

    Parameters
    ----------
    psi1 : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    psi2 : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    mul : int, float or complex
        Optional

    """
    for state, amp in psi2.items():
        psi1[state] = psi1.get(state, 0) - mul * psi2[state]


def scale(psi, mul):
    """
    return mul*|\psi\rangle
    """
    return {s: a * mul for s, a in psi.items()}


def addToFirst(psi1, psi2, mul=1):
    r"""
    To state :math:`|\psi_1\rangle`, add  :math:`mul * |\psi_2\rangle`.

    Acknowledgement: Written by Petter Saterskog.

    Parameters
    ----------
    psi1 : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    psi2 : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    mul : int, float or complex
        Optional

    """
    for s, a in psi2.items():
        psi1[s] = a * mul + psi1.get(s, 0)


def a(n_spin_orbitals, i, psi):
    r"""
    Return :math:`|psi' \rangle = c_i |psi \rangle`.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    i : int
        Spin-orbital index
    psi : dict
        Multi configurational state

    Returns
    -------
    ret : dict
        New multi configurational state

    """
    ret = {}
    for state, amp in psi.items():
        state_new, sign = remove.ubytes(n_spin_orbitals, i, state)
        if sign != 0:
            ret[state_new] = amp * sign
    return ret


def c(n_spin_orbitals, i, psi):
    r"""
    Return :math:`|psi' \rangle = c_i^\dagger |psi \rangle`.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    i : int
        Spin-orbital index
    psi : dict
        Multi configurational state

    Returns
    -------
    ret : dict
        New multi configurational state

    """
    ret = {}
    for state, amp in psi.items():
        state_new, sign = create.ubytes(n_spin_orbitals, i, state)
        if sign != 0:
            ret[state_new] = amp * sign
    return ret


def identity(n_spin_orbitals, i, psi):
    return psi


def gauntC(k, l, m, lp, mp, prec=16):
    """
    return "nonvanishing" Gaunt coefficients of
    Coulomb interaction expansion.
    """
    c = sqrt(4 * pi / (2 * k + 1)) * (-1) ** m * gaunt(l, k, lp, -m, m - mp, mp, prec=prec)
    return float(c)


def getU(l1, m1, l2, m2, l3, m3, l4, m4, R):
    r"""
    Return Hubbard U term for four spherical harmonics functions.

    Scattering process:

    :math:`u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    * c_{l_1,m_1}^\dagger c_{l_2,m_2}^\dagger c_{l_3,m_3} c_{l_4,m_4}`.

    Parameters
    ----------
    l1 : int
        angular momentum of orbital 1
    m1 : int
        z projected angular momentum of orbital 1
    l2 : int
        angular momentum of orbital 2
    m2 : int
        z projected angular momentum of orbital 2
    l3 : int
        angular momentum of orbital 3
    m3 : int
        z projected angular momentum of orbital 3
    l4 : int
        angular momentum of orbital 4
    m4 : int
        z projected angular momentum of orbital 4
    R : list
        Slater-Condon parameters.
        Elements R[k] fullfill
        :math:`0<=k<=\textrm{min}(|l_1+l_4|,|l_2+l_3|)`.
        Note, U is nonzero if :math:`k+l_1+l_4` is an even integer
        and :math:`k+l_3+l_2` is an even integer.
        For example: if :math:`l_1=l_2=l_3=l_4=2`,
        R = [R0,R1,R2,R3,R4] and only R0,R2 and R4 will
        give nonzero contribution.

    Returns
    -------
    u - float
        Hubbard U term.
    """
    # Check if angular momentum is conserved
    if m1 + m2 == m3 + m4:
        u = 0
        for k, Rk in enumerate(R):
            u += Rk * gauntC(k, l1, m1, l4, m4) * gauntC(k, l3, m3, l2, m2)
    else:
        u = 0
    return u


def printGaunt(l=2, lp=2):
    """
    print Gaunt coefficients.

    Parameters
    ----------
    l : int
        angular momentum
    lp : int
        angular momentum
    """
    # Print Gauent coefficients
    for k in range(l + lp + 1):
        if rank == 0:
            print("k={:d}".format(k))
        for m in range(-l, l + 1):
            s = ""
            for mp in range(-lp, lp + 1):
                s += " {:3.2f}".format(gauntC(k, l, m, lp, mp))
            if rank == 0:
                print(s)
        if rank == 0:
            print("")


def getNoSpinUop(l1, l2, l3, l4, R):
    r"""
    Return non-spin polarized U operator.

    Scattering processes:

    :math:`1/2 \sum_{m_1,m_2,m_3,m_4}
    u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    c_{l_1,m_1}^\dagger c_{l_2,m_2}^\dagger c_{l_3,m_3} c_{l_4,m_4}`.

    No spin polarization considered, thus basis is: (l,m)

    """
    uDict = {}
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for m3 in range(-l3, l3 + 1):
                for m4 in range(-l4, l4 + 1):
                    u = getU(l1, m1, l2, m2, l3, m3, l4, m4, R)
                    if u != 0:
                        uDict[((l1, m1), (l2, m2), (l3, m3), (l4, m4))] = u / 2.0
    return uDict


def getUop_from_rspt_u4(u4):
    l1, l2, l3, l4 = u4.shape
    l1 = ((l1 // 2) - 1) // 2
    l2 = ((l2 // 2) - 1) // 2
    l3 = ((l3 // 2) - 1) // 2
    l4 = ((l4 // 2) - 1) // 2
    uDict = {}
    # for i, m1 in enumerate(range(-l1, l1 + 1)):
    #     for j, m2 in enumerate(range(-l2, l4 + 1)):
    #         for k, m3 in enumerate(range(-l3, l3 + 1)):
    #             for l, m4 in enumerate(range(-l4, l2 + 1)):
    for i, j, k, l in itertools.product(range(u4.shape[0]), range(u4.shape[1]), range(u4.shape[2]), range(u4.shape[3])):
        u = u4[i, j, k, l]
        if abs(u) > 1e-10:
            proccess = (
                (i, "c"),
                (j, "c"),
                (k, "a"),
                (l, "a"),
            )
            uDict[proccess] = u / 2
            # for s in range(2):
            #     for sp in range(2):
            #         proccess = (
            #             ((l1, s, m1), "c"),
            #             ((l2, sp, m2), "c"),
            #             ((l3, sp, m3), "a"),
            #             ((l4, s, m4), "a"),
            #         )
            #         uDict[proccess] = u / 2
    return uDict


def getUop(l1, l2, l3, l4, R):
    r"""
    Return U operator.

    Scattering processes:
    :math:`1/2 \sum_{m_1,m_2,m_3,m_4} u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    * \sum_{s,sp} c_{l_1, s, m_1}^\dagger c_{l_2, sp, m_2}^\dagger
    c_{l_3, sp, m_3} c_{l_4, s, m_4}`.

    Spin polarization is considered, thus basis: (l, s, m),
    where :math:`s \in \{0, 1 \}` and these indices respectively
    corresponds to the physical values
    :math:`\{-\frac{1}{2},\frac{1}{2} \}`.

    Returns
    -------
    uDict : dict
        Elements of the form:
        ((sorb1,'c'),(sorb2,'c'),(sorb3,'a'),(sorb4,'a')) : u/2
        where sorb1 is a superindex of (l, s, m).

    """
    uDict = {}
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for m3 in range(-l3, l3 + 1):
                for m4 in range(-l4, l4 + 1):
                    u = getU(l1, m1, l2, m2, l3, m3, l4, m4, R)
                    if u != 0:
                        for s in range(2):
                            for sp in range(2):
                                proccess = (
                                    ((l1, s, m1), "c"),
                                    ((l2, sp, m2), "c"),
                                    ((l3, sp, m3), "a"),
                                    ((l4, s, m4), "a"),
                                )
                                # Pauli exclusion principle
                                # if not (s == sp and ((l1, m1) == (l2, m2) or (l3, m3) == (l4, m4))):
                                uDict[proccess] = u / 2.0
    return uDict


def addOps(ops):
    """
    Return one operator, represented as a dictonary.

    Parameters
    ----------
    ops : list
        Operators

    Returns
    -------
    opSum : dict

    """
    opSum = {}
    for op in ops:
        for sOp, value in op.items():
            if np.abs(value) > 1e-12:
                if sOp in opSum:
                    opSum[sOp] += value
                else:
                    opSum[sOp] = value
    return opSum


def subtractOps(A, B):
    """
    Return the operator A - B
    """
    opDiff = A.copy()
    for sOp, value in B.items():
        if np.abs(value) > 0:
            opDiff[sOp] = opDiff.get(sOp, 0) - value
            # if sOp in opDiff:
            #     opDiff[sOp] -= value
            # else:
            #     opDiff[sOp] = -value
    return opDiff


def get2p3dSlaterCondonUop(Fdd=(9, 0, 8, 0, 6), Fpp=(20, 0, 8), Fpd=(10, 0, 8), Gpd=(0, 3, 0, 2)):
    """
    Return a 2p-3d U operator containing a sum of
    different Slater-Condon proccesses.

    Parameters
    ----------
    Fdd : tuple
    Fpp : tuple
    Fpd : tuple
    Gpd : tuple

    """
    # Calculate F_dd^{0,2,4}
    FddOp = {}
    if Fdd is not None:
        FddOp = getUop(l1=2, l2=2, l3=2, l4=2, R=Fdd)
    # Calculate F_pp^{0,2}
    FppOp = {}
    if Fpp is not None:
        FppOp = getUop(l1=1, l2=1, l3=1, l4=1, R=Fpp)
    # Calculate F_pd^{0,2}
    FpdOp = {}
    if Fpd is not None:
        FpdOp1 = getUop(l1=1, l2=2, l3=2, l4=1, R=Fpd)
        FpdOp2 = getUop(l1=2, l2=1, l3=1, l4=2, R=Fpd)
        FpdOp = addOps([FpdOp1, FpdOp2])
    # Calculate G_pd^{1,3}
    GpdOp = {}
    if Gpd is not None:
        GpdOp1 = getUop(l1=1, l2=2, l3=1, l4=2, R=Gpd)
        GpdOp2 = getUop(l1=2, l2=1, l3=2, l4=1, R=Gpd)
        GpdOp = addOps([GpdOp1, GpdOp2])
    # Add operators
    uOp = addOps([FddOp, FppOp, FpdOp, GpdOp])
    return uOp


def getSOCop(xi, l=2):
    """
    Return SOC operator for one l-shell.

    Returns
    -------
    uDict : dict
        Elements of the form:
        ((sorb1,'c'), (sorb2,'a') : h_value
        where sorb1 is a superindex of (l, s, m).

    """
    if abs(xi) > 1e-10:
        return {}
    opDict = {}
    for m in range(-l, l + 1):
        for s in range(2):
            value = xi * m * (1 / 2.0 if s == 1 else -1 / 2.0)
            opDict[(((l, s, m), "c"), ((l, s, m), "a"))] = value
    for m in range(-l, l):
        value = xi / 2.0 * sqrt((l - m) * (l + m + 1))
        opDict[(((l, 1, m), "c"), ((l, 0, m + 1), "a"))] = value
        opDict[(((l, 0, m + 1), "c"), ((l, 1, m), "a"))] = value
    return opDict


def gethHfieldop(hx, hy, hz, l=2):
    """
    Return magnetic field operator for one l-shell.

    Returns
    -------
    hHfieldOperator : dict
        Elements of the form:
        ((sorb1,'c'), (sorb2,'a') : h_value
        where sorb1 is a superindex of (l, s, m).

    """
    hHfieldOperator = {}
    for m in range(-l, l + 1):
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] = hx / 2
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] = hx / 2
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] += -hy * 1j / 2
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] += hy * 1j / 2
        for s in range(2):
            hHfieldOperator[(((l, s, m), "c"), ((l, s, m), "a"))] = hz / 2 if s == 1 else -hz / 2
    return hHfieldOperator


def c2i(nBaths, spinOrb):
    """
    Return an index, representing a spin-orbital or a bath state.

    Parameters
    ----------
    nBaths : ordered dict
        An elements is either of the form:
        angular momentum : number of bath spin-orbitals
        or of the form:
        (angular momentum_a, angular momentum_b, ...) : number of bath states.
        The latter form is used if impurity orbitals from different
        angular momenta share the same bath states.
    spinOrb : tuple
        (l, s, m), (l, b) or ((l_a, l_b, ...), b)

    Returns
    -------
    i : int
        An index denoting a spin-orbital or a bath state.

    """
    # Counting index and return variable.
    i = 0
    # Check if spinOrb is an impurity spin-orbital.
    # Loop through all impurity spin-orbitals.
    for lp in nBaths.keys():
        if isinstance(lp, int):
            for sp in range(2):
                for mp in range(-lp, lp + 1):
                    if (lp, sp, mp) == spinOrb:
                        return i
                    i += 1
        elif isinstance(lp, tuple):
            # Loop over all different angular momenta in lp.
            for lp_int in lp:
                for sp in range(2):
                    for mp in range(-lp_int, lp_int + 1):
                        if (lp_int, sp, mp) == spinOrb:
                            return i
                        i += 1
    # If reach this point it means spinOrb is a bath state.
    # Need to figure out which one index is has.
    for lp, nBath in nBaths.items():
        for b in range(nBath):
            if (lp, b) == spinOrb:
                return i
            i += 1
    print(spinOrb)
    raise Exception("Can not find index corresponding to spin-orbital state")


def i2c(nBaths, i):
    """
    Return an coordinate tuple, representing a spin-orbital.

    Parameters
    ----------
    nBaths : ordered dict
        An elements is either of the form:
        angular momentum : number of bath spin-orbitals
        or of the form:
        (angular momentum_a, angular momentum_b, ...) : number of bath states.
        The latter form is used if impurity orbitals from different
        angular momenta share the same bath states.
    i : int
        An index denoting a spin-orbital or a bath state.

    Returns
    -------
    spinOrb : tuple
        (l, s, m), (l, b) or ((l_a, l_b, ...), b)

    """
    # Counting index.
    k = 0
    # Check if index "i" belong to an impurity spin-orbital.
    # Loop through all impurity spin-orbitals.
    for lp in nBaths.keys():
        if isinstance(lp, int):
            # Check if index "i" belong to impurity spin-orbital having lp.
            if i - k < 2 * (2 * lp + 1):
                for sp in range(2):
                    for mp in range(-lp, lp + 1):
                        if k == i:
                            return (lp, sp, mp)
                        k += 1
            k += 2 * (2 * lp + 1)
        elif isinstance(lp, tuple):
            # Loop over all different angular momenta in lp.
            for lp_int in lp:
                # Check if index "i" belong to impurity spin-orbital having lp_int.
                if i - k < 2 * (2 * lp_int + 1):
                    for sp in range(2):
                        for mp in range(-lp_int, lp_int + 1):
                            if k == i:
                                return (lp_int, sp, mp)
                            k += 1
                k += 2 * (2 * lp_int + 1)
    # If reach this point it means index "i" belong to a bath state.
    # Need to figure out which one.
    for lp, nBath in nBaths.items():
        b = i - k
        # Check if bath state belong to bath states having lp.
        if b < nBath:
            # The index "b" will have a value between 0 and nBath-1
            return (lp, b)
        k += nBath
    print(i)
    raise Exception("Can not find spin-orbital state corresponding to index.")


def getLz3d(nBaths, psi):
    r"""
    Return expectation value :math:`\langle psi| Lz_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        Multi configurational state.

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    Lz = 0
    for state, amp in psi.items():
        tmp = 0
        for i in psr.bytes2tuple(state, n_spin_orbitals):
            spinOrb = i2c(nBaths, i)
            # Look for spin-orbitals of the shape: spinOrb = (l, s, ml), with l=2.
            if len(spinOrb) == 3 and spinOrb[0] == 2:
                tmp += spinOrb[2]
        Lz += tmp * abs(amp) ** 2
    return Lz


def getSz3d(nBaths, psi):
    r"""
    Return expectation value :math:`\langle psi| Sz_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        Multi configurational state.

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    Sz = 0
    for state, amp in psi.items():
        tmp = 0
        for i in psr.bytes2tuple(state, n_spin_orbitals):
            spinOrb = i2c(nBaths, i)
            # Look for spin-orbitals of the shape: spinOrb = (l, s, ml), with l=2.
            if len(spinOrb) == 3 and spinOrb[0] == 2:
                tmp += -1 / 2 if spinOrb[1] == 0 else 1 / 2
        Sz += tmp * abs(amp) ** 2
    return Sz


def getSsqr3d(nBaths, psi, tol=1e-8):
    r"""
    Return expectation value :math:`\langle psi| S^2_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        normalized multi configurational state.

    """
    psi1 = applySz3d(nBaths, psi)
    psi2 = applySplus3d(nBaths, psi)
    psi3 = applySminus3d(nBaths, psi)
    S2 = norm2(psi1) + 1 / 2 * (norm2(psi2) + norm2(psi3))
    if S2.imag > tol:
        print("Warning: <S^2> complex valued!")
    return S2.real


def getLsqr3d(nBaths, psi, tol=1e-8):
    r"""
    Return expectation value :math:`\langle psi| L^2_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        normalized multi configurational state.

    """
    psi1 = applyLz3d(nBaths, psi)
    psi2 = applyLplus3d(nBaths, psi)
    psi3 = applyLminus3d(nBaths, psi)
    L2 = norm2(psi1) + 1 / 2 * (norm2(psi2) + norm2(psi3))
    if L2.imag > tol:
        print("Warning: <L^2> complex valued!")
    return L2.real


def getTraceDensityMatrix(nBaths, psi, l=2):
    r"""
    Return  :math:`\langle psi| \sum_i c_i^\dagger c_i |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states
    psi : dict
        Multi configurational state.
    l : int (optional)
        Angular momentum

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    n = 0
    for state, amp in psi.items():
        s = psr.bytes2str(state, n_spin_orbitals)
        nState = 0
        for spin in range(2):
            for m in range(-l, l + 1):
                i = c2i(nBaths, (l, spin, m))
                if s[i] == "1":
                    nState += 1
        nState *= abs(amp) ** 2
        n += nState
    return n


def build_density_matrix(orbital_indices, psi, n_spin_orbitals):
    rho = np.zeros((len(orbital_indices), len(orbital_indices)), dtype=complex)
    for i, j in itertools.product(range(len(orbital_indices)), range(len(orbital_indices))):
        psi_new = a(n_spin_orbitals, orbital_indices[i], psi)
        psi_new = c(n_spin_orbitals, orbital_indices[j], psi_new)
        tmp = inner(psi, psi_new)
        if tmp != 0:
            rho[i, j] = tmp
    return rho


def build_impurity_density_matrix(n_imp_orbitals, n_bath_orbitals, psi):
    n_spin_orbitals = n_imp_orbitals + n_bath_orbitals
    return build_density_matrix(range(n_imp_orbitals), psi, n_spin_orbitals)


def build_bath_density_matrix(n_imp_orbitals, n_bath_orbitals, psi):
    n_spin_orbitals = n_imp_orbitals + n_bath_orbitals
    return build_density_matrix(range(n_imp_orbitals, n_spin_orbitals), psi, n_spin_orbitals)


def getDensityMatrix(nBaths, psi, l=2):
    r"""
    Return density matrix in spherical harmonics basis.

    :math:`n_{ij} = \langle i| \tilde{n} |j \rangle =
    \langle psi| c_j^\dagger c_i |psi \rangle`.

    Returns
    -------
    densityMatrix : dict
        keys of the form: :math:`((l,mi,si),(l,mj,sj))`.
        values of the form: :math:`\langle psi| c_j^\dagger c_i |psi \rangle`.

    Notes
    -----
    The perhaps suprising index notation is because
    of the following four equations:

    :math:`G_{ij}(\tau->0^-) = \langle c_j^\dagger c_i \rangle`.

    :math:`G_ij(\tau->0^-) = \langle i|\tilde{G}(\tau->0^-)|j \rangle`.

    :math:`\tilde{G}(\tau->0^-) = \tilde{n}`.

    :math:`n_{ij} = \langle i| \tilde{n} |j \rangle`.

    Note: Switched index order compared to the order of operators,
    where :math:`op[((li,mi,si),(lj,mj,sj))] = value`
    means operator: :math:`value * c_{li,mi,si}^\dagger c_{lj,mj,sj}`

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    densityMatrix = OrderedDict()
    for si in range(2):
        for sj in range(2):
            for mi in range(-l, l + 1):
                for mj in range(-l, l + 1):
                    i = c2i(nBaths, (l, si, mi))
                    j = c2i(nBaths, (l, sj, mj))
                    psi_new = a(n_spin_orbitals, i, psi)
                    psi_new = c(n_spin_orbitals, j, psi_new)
                    tmp = inner(psi, psi_new)
                    if tmp != 0:
                        densityMatrix[((l, si, mi), (l, sj, mj))] = tmp
    return densityMatrix


def getDensityMatrixCubic(nBaths, psi):
    r"""
    Return density matrix in cubic harmonics basis.

    :math:`n_{ic,jc} = \langle ic| \tilde{n} |jc \rangle =
    \langle psi| c_{jc}^\dagger c_{ic} |psi \rangle`,
    where ic is a index containing a cubic harmonics and a spin.

    :math:`c_{ic}^\dagger = \sum_j u[j,i] c_j^\dagger`

    This gives:
    :math:`\langle psi| c_{jc}^\dagger c_{ic} |psi \rangle
    = \sum_{k,m} u[k,j] u[m,i]^{*}
    * \langle psi| c_{k,sj}^\dagger c_{m,si} |psi \rangle
    = \sum_{k,m} u[m,i]^* n[{m,si},{k,sj}] u[k,j]`

    Returns
    -------
    densityMatrix : dict
        keys of the form: :math:`((i,si),(j,sj))`.
        values of the form: :math:`\langle psi| c_{jc}^\dagger c_{ic}
        |psi \rangle`.

    """
    # density matrix in spherical harmonics
    nSph = getDensityMatrix(nBaths, psi)
    l = 2
    # |i(cubic)> = sum_j u[j,i] |j(spherical)>
    u = get_spherical_2_cubic_matrix()
    nCub = OrderedDict()
    for i in range(2 * l + 1):
        for j in range(2 * l + 1):
            for si in range(2):
                for sj in range(2):
                    for k, mk in enumerate(range(-l, l + 1)):
                        for m, mm in enumerate(range(-l, l + 1)):
                            eSph = ((l, si, mm), (l, sj, mk))
                            if eSph in nSph:
                                tmp = np.conj(u[m, i]) * nSph[eSph] * u[k, j]
                                if tmp != 0:
                                    eCub = ((si, i), (sj, j))
                                    nCub[eCub] = tmp + nCub.get(eCub, 0)
                                    # if eCub in nCub:
                                    #     nCub[eCub] += tmp
                                    # else:
                                    #     nCub[eCub] = tmp
    return nCub


def printDensityMatrixCubic(nBaths, psis, tolPrintOccupation):
    if rank == 0:
        # Calculate density matrix
        print("Density matrix (in cubic harmonics basis):")
        for i, psi in enumerate(psis):
            print("Eigenstate {:d}".format(i))
            n = getDensityMatrixCubic(nBaths, psi)
            print("#density matrix elements: {:d}".format(len(n)))
            for e, ne in n.items():
                if abs(ne) > tolPrintOccupation:
                    if e[0] == e[1]:
                        print(
                            "Diagonal: (i,s) =",
                            e[0],
                            ", occupation = {:7.2f}".format(ne),
                        )
                    else:
                        print("Off-diagonal: (i,si), (j,sj) =", e, ", {:7.2f}".format(ne))
            print("")


def getEgT2gOccupation(nBaths, psi):
    r"""
    Return occupations of :math:`eg_\downarrow, eg_\uparrow,
    t2g_\downarrow, t2g_\uparrow` states.

    Calculate from density matrix diagonal:
    :math:`n_{ic,ic} = \langle psi| c_{ic}^\dagger c_{ic} |psi \rangle`,
    where `ic` is a cubic harmonics index, and
    :math:`c_{ic}^\dagger = \sum_j u[j,ic] c_j^\dagger`,
    where `j` is a spherical harmonics index.

    This gives:
    :math:`\langle psi| c_{ic,s}^\dagger c_{ic,s} |psi \rangle
    = \sum_{j,k} u[j,ic] u[k,ic]^{*}
    * \langle psi| c_{j,s}^\dagger c_{k,s} |psi \rangle
    = \sum_{j,k} u[k,ic]^*  n[{k,s},{j,s}] u[j,ic]`

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    l = 2
    # |i(cubic)> = sum_j u[j,i] |j(spherical)>
    u = get_spherical_2_cubic_matrix()
    eg_dn, eg_up, t2g_dn, t2g_up = 0, 0, 0, 0
    for i in range(2 * l + 1):
        for j, mj in enumerate(range(-l, l + 1)):
            for k, mk in enumerate(range(-l, l + 1)):
                for s in range(2):
                    jj = c2i(nBaths, (l, s, mj))
                    kk = c2i(nBaths, (l, s, mk))
                    psi_new = a(n_spin_orbitals, kk, psi)
                    psi_new = c(n_spin_orbitals, jj, psi_new)
                    v = u[j, i] * np.conj(u[k, i]) * inner(psi, psi_new)
                    if i < 2:
                        if s == 0:
                            eg_dn += v
                        else:
                            eg_up += v
                    else:
                        if s == 0:
                            t2g_dn += v
                        else:
                            t2g_up += v
    occs = [eg_dn, eg_up, t2g_dn, t2g_up]
    for i in range(len(occs)):
        if abs(occs[i].imag) < 1e-12:
            occs[i] = occs[i].real
        else:
            print("Warning: Complex occupation")
    return occs


def applySz3d(nBaths, psi):
    r"""
    Return :math:`|psi' \rangle = S^{z}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    psiNew = {}
    l = 2
    for s in range(2):
        for m in range(-l, l + 1):
            i = c2i(nBaths, (l, s, m))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, i, psi))
            addToFirst(psiNew, psiP, 1 / 2 if s == 1 else -1 / 2)
    return psiNew


def applyLz3d(nBaths, psi):
    r"""
    Return :math:`|psi' \rangle = L^{z}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    psiNew = {}
    l = 2
    for s in range(2):
        for m in range(-l, l + 1):
            i = c2i(nBaths, (l, s, m))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, i, psi))
            addToFirst(psiNew, psiP, m)
    return psiNew


def applySplus3d(nBaths, psi):
    r"""
    Return :math:`|psi' \rangle = S^{+}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l, l + 1):
        i = c2i(nBaths, (l, 1, m))
        j = c2i(nBaths, (l, 0, m))
        psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
        # sQ = 1/2.
        # sqrt((sQ-(-sQ))*(sQ+(-sQ)+1)) == 1
        addToFirst(psiNew, psiP)
    return psiNew


def applyLplus3d(nBaths, psi):
    r"""
    Return :math:`|psi' \rangle = L^{+}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    psiNew = {}
    l = 2
    for s in range(2):
        for m in range(-l, l):
            i = c2i(nBaths, (l, s, m + 1))
            j = c2i(nBaths, (l, s, m))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
            addToFirst(psiNew, psiP, sqrt((l - m) * (l + m + 1)))
    return psiNew


def applySminus3d(nBaths, psi):
    r"""
    Return :math:`|psi' \rangle = S^{-}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    psiNew = {}
    l = 2
    for m in range(-l, l + 1):
        i = c2i(nBaths, (l, 0, m))
        j = c2i(nBaths, (l, 1, m))
        psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
        # sQ = 1/2.
        # sqrt((sQ+sQ)*(sQ-sQ+1)) == 1
        addToFirst(psiNew, psiP)
    return psiNew


def applyLminus3d(nBaths, psi):
    r"""
    Return :math:`|psi' \rangle = L^{-}_{3d} |psi \rangle`.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath states.
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.

    Returns
    -------
    psiNew : dict
        With the same format as psi.

    """
    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    psiNew = {}
    l = 2
    for s in range(2):
        for m in range(-l + 1, l + 1):
            i = c2i(nBaths, (l, s, m - 1))
            j = c2i(nBaths, (l, s, m))
            psiP = c(n_spin_orbitals, i, a(n_spin_orbitals, j, psi))
            addToFirst(psiNew, psiP, sqrt((l + m) * (l - m + 1)))
    return psiNew


def applyOp(n_spin_orbitals, op, psi, slaterWeightMin=0, restrictions=None, opResult=None):
    r"""
        Return :math:`|psi' \rangle = op |psi \rangle`.

        If opResult is not None, it is updated to contain information of how the
        operator op acted on the product states in psi.

        Parameters
    :
    ----------
        n_spin_orbitals : int
            Total number of spin-orbitals in the system.
        op : dict
            Operator of the format
            tuple : amplitude,

            where each tuple describes a scattering

            process. Examples of possible tuples (and their meanings) are:

            ((i, 'c'))  <-> c_i^dagger

            ((i, 'a'))  <-> c_i

            ((i, 'c'), (j, 'a'))  <-> c_i^dagger c_j

            ((i, 'c'), (j, 'c'), (k, 'a'), (l, 'a')) <-> c_i^dagger c_j^dagger c_k c_l
        psi : dict
            Multi-configurational state.
            Product states as keys and amplitudes as values.
        slaterWeightMin : float
            Restrict the number of product states by
            looking at `|amplitudes|^2`.
        restrictions : dict
            Restriction the occupation of generated
            product states.
        opResult : dict
            In and output argument.
            If present, the results of the operator op acting on each
            product state in the state psi is added and stored in this
            variable.

        Returns
        -------
        psiNew : dict
            New state of the same format as psi.


        Note
        ----
        Different implementations exist.
        They return the same result, but calculations vary a bit.

    """
    psiNew = {}
    if opResult is None and restrictions is not None:
        # Loop over product states in psi.
        for state, amp in psi.items():
            # assert amp != 0
            bits = psr.bytes2bitarray(state, n_spin_orbitals)
            for process, h in op.items():
                # assert h != 0
                # Initialize state
                state_new = bits.copy()
                signTot = 1
                # for i, action in process[-1::-1]:
                for i, action in process[-1::-1]:
                    if action == "a":
                        sign = remove.ubitarray(i, state_new)
                    elif action == "c":
                        sign = create.ubitarray(i, state_new)
                    elif action == "i":
                        sign = 1
                    if sign == 0:
                        break
                    signTot *= sign
                else:
                    stateB = psr.bitarray2bytes(state_new)
                    if stateB in psiNew:
                        psiNew[stateB] += amp * h * signTot
                    else:
                        # Convert product state to the tuple representation.
                        stateB_tuple = psr.bitarray2tuple(state_new)
                        # Check that product state sB fulfills
                        # occupation restrictions.
                        for restriction, occupations in restrictions.items():
                            n = len(restriction.intersection(stateB_tuple))
                            if n < occupations[0] or occupations[1] < n:
                                break
                        else:
                            # Occupations ok, so add contributions
                            psiNew[stateB] = amp * h * signTot
    elif opResult is None and restrictions is None:
        # Loop over product states in psi.
        for state, amp in psi.items():
            # assert amp != 0
            bits = psr.bytes2bitarray(state, n_spin_orbitals)
            for process, h in op.items():
                # assert h != 0
                # Initialize state
                state_new = bits.copy()
                signTot = 1
                for i, action in process[-1::-1]:
                    if action == "a":
                        sign = remove.ubitarray(i, state_new)
                    elif action == "c":
                        sign = create.ubitarray(i, state_new)
                    elif action == "i":
                        sign = 1
                    if sign == 0:
                        break
                    signTot *= sign
                else:
                    stateB = psr.bitarray2bytes(state_new)
                    psiNew[stateB] = amp * h * signTot + psiNew.get(stateB, 0)
                    # if stateB in psiNew:
                    #     psiNew[stateB] += amp * h * signTot
                    # else:
                    #     psiNew[stateB] = amp * h * signTot
    elif opResult is not None and restrictions is not None:
        # Loop over product states in psi.
        for state, amp in psi.items():
            # assert amp != 0
            if state in opResult:
                addToFirst(psiNew, opResult[state], amp)
            else:
                bits = psr.bytes2bitarray(state, n_spin_orbitals)
                # Create new element in opResult
                # Store H|PS> for product states |PS> not yet in opResult
                opResult[state] = {}
                for process, h in op.items():
                    # assert h != 0
                    # Initialize state
                    state_new = bits.copy()
                    signTot = 1
                    for i, action in process[-1::-1]:
                        if action == "a":
                            sign = remove.ubitarray(i, state_new)
                        elif action == "c":
                            sign = create.ubitarray(i, state_new)
                        elif action == "i":
                            sign = 1
                        if sign == 0:
                            break
                        signTot *= sign
                    else:
                        stateB = psr.bitarray2bytes(state_new)
                        if stateB in psiNew:
                            # Occupations ok, so add contributions
                            psiNew[stateB] += amp * h * signTot
                            opResult[state][stateB] = h * signTot + opResult[state].get(stateB, 0)
                            # if stateB in opResult[state]:
                            #     opResult[state][stateB] += h * signTot
                            # else:
                            #     opResult[state][stateB] = h * signTot
                        else:
                            # Convert product state to the tuple representation.
                            stateB_tuple = psr.bitarray2tuple(state_new)
                            # Check that product state sB fulfills the
                            # occupation restrictions.
                            for restriction, occupations in restrictions.items():
                                n = len(restriction.intersection(stateB_tuple))
                                if n < occupations[0] or occupations[1] < n:
                                    break
                            else:
                                # Occupations ok, so add contributions
                                psiNew[stateB] = amp * h * signTot
                                opResult[state][stateB] = h * signTot
                # Make sure amplitudes in opResult are bigger than
                # the slaterWeightMin cutoff.
                for ps, amp in list(opResult[state].items()):
                    # Remove product states with small weight
                    if abs(amp) ** 2 < slaterWeightMin:
                        opResult[state].pop(ps)
    elif opResult is not None and restrictions is None:
        # Loop over product states in psi.
        for state, amp in psi.items():
            # assert amp != 0
            if state in opResult:
                addToFirst(psiNew, opResult[state], amp)
            else:
                bits = psr.bytes2bitarray(state, n_spin_orbitals)
                # Create new element in opResult
                # Store H|PS> for product states |PS> not yet in opResult
                opResult[state] = {}
                for process, h in op.items():
                    # assert h != 0
                    # Initialize state
                    state_new = bits.copy()
                    signTot = 1
                    for i, action in process[-1::-1]:
                        if action == "a":
                            sign = remove.ubitarray(i, state_new)
                        elif action == "c":
                            sign = create.ubitarray(i, state_new)
                        elif action == "i":
                            sign = 1
                        if sign == 0:
                            break
                        signTot *= sign
                    else:
                        stateB = psr.bitarray2bytes(state_new)
                        opResult[state][stateB] = h * signTot + opResult[state].get(stateB, 0)
                        # if stateB in opResult[state]:
                        #     opResult[state][stateB] += h * signTot
                        # else:
                        #     opResult[state][stateB] = h * signTot
                        psiNew[stateB] = amp * h * signTot + psiNew.get(stateB, 0)
                        # if stateB in psiNew:
                        #     psiNew[stateB] += amp * h * signTot
                        # else:
                        #     psiNew[stateB] = amp * h * signTot
                # Make sure amplitudes in opResult are bigger than
                # the slaterWeightMin cutoff.
                for ps, amp in list(opResult[state].items()):
                    # Remove product states with small weight
                    if abs(amp) ** 2 < slaterWeightMin:
                        opResult[state].pop(ps)
    else:
        raise Exception("Method not implemented.")
    # Remove product states with small weight
    for state, amp in list(psiNew.items()):
        if abs(amp) ** 2 < slaterWeightMin:
            psiNew.pop(state)
    # print (f"op: psiNew\n\t{op}: {psiNew}")
    return psiNew


def occupation_is_within_restrictions(state, n_spin_orbitals, restrictions):
    """
    Return True if the occupations in state are within the restrictions. Otherwise, return False.
    """
    if restrictions is None:
        return True
    # state_new_tuple = psr.bytes2tuple(state, n_spin_orbitals)
    bits = psr.bytes2bitarray(state, n_spin_orbitals)
    for restriction, occupations in restrictions.items():
        n = sum(bits[i] for i in restriction)
        # n = len(restriction.intersection(state_new_tuple))
        if n < occupations[0] or occupations[1] < n:
            return False
    return True


def applyOp_new(n_spin_orbitals: int, op: dict, psi: dict, slaterWeightMin=0, restrictions=None, opResult=None):
    r"""
    Return :math:`|psi' \rangle = op |psi \rangle`.

    If opResult is not None, it is updated to contain information of how the
    operator op acted on the product states in psi.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    op : dict
        Operator of the format
        tuple : amplitude,

        where each tuple describes a scattering

        process. Examples of possible tuples (and their meanings) are:

        ((i, 'c'))  <-> c_i^dagger

        ((i, 'a'))  <-> c_i

        ((i, 'c'), (j, 'a'))  <-> c_i^dagger c_j

        ((i, 'c'), (j, 'c'), (k, 'a'), (l, 'a')) <-> c_i^dagger c_j^dagger c_k c_l
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    slaterWeightMin : float
        Restrict the number of product states by
        looking at `|amplitudes|^2`.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    opResult : dict
        In and output argument.
        If present, the results of the operator op acting on each
        product state in the state psi is added and stored in this
        variable.

    Returns
    -------
    psiNew : dict
        New state of the same format as psi.


    """
    psiNew = dict()
    if opResult is None:
        opResult = dict()
    solved_states = psi.keys() & opResult.keys()
    for state in solved_states:
        addToFirst(psiNew, opResult[state], psi[state])

    newResults = dict()
    for state, (process, h) in itertools.product(psi.keys() - solved_states, op.items()):
        amp = psi[state]
        state_bits_new = psr.bytes2bitarray(state, n_spin_orbitals)
        # state_bits_new = state_bits.copy()
        signTot = 1
        for i, action in process[-1::-1]:
            if action == "a":
                sign = remove.ubitarray(i, state_bits_new)
            elif action == "c":
                sign = create.ubitarray(i, state_bits_new)
            elif action == "i":
                sign = 1
            signTot *= sign
            if signTot == 0:
                break
        if signTot == 0:
            newResults[state] = newResults.get(state, {})
            continue
        state_new = psr.bitarray2bytes(state_bits_new)
        if not occupation_is_within_restrictions(state_new, n_spin_orbitals, restrictions):
            continue
        psiNew[state_new] = amp * h * signTot + psiNew.get(state_new, 0)
        if state not in newResults:
            newResults[state] = {}
        newResults[state][state_new] = h * signTot + newResults[state].get(state_new, 0)
    opResult.update(newResults)

    return {state: amp for state, amp in psiNew.items() if abs(amp) ** 2 > slaterWeightMin}


def applyOp_thread_worker(op, psi, n_spin_orbitals, restrictions):
    opResult = dict()
    psiNew = dict()
    for (process, h), (state, amp) in itertools.product(op.items(), psi.items()):
        state_bits_new = psr.bytes2bitarray(state, n_spin_orbitals)
        signTot = 1
        for i, action in process[-1::-1]:
            if action == "a":
                sign = remove.ubitarray(i, state_bits_new)
            elif action == "c":
                sign = create.ubitarray(i, state_bits_new)
            elif action == "i":
                sign = 1
            signTot *= sign
            if signTot == 0:
                break
        if signTot == 0:
            opResult[state] = opResult.get(state, {})
            continue
        state_new = psr.bitarray2bytes(state_bits_new)
        if not occupation_is_within_restrictions(state_new, n_spin_orbitals, restrictions):
            continue
        psiNew[state_new] = amp * h * signTot + psiNew.get(state_new, 0)
        if state not in opResult:
            opResult[state] = {}
        opResult[state][state_new] = h * signTot + opResult[state].get(state_new, 0)
    return psiNew, opResult


def applyOp_threadpool(n_spin_orbitals: int, op: dict, psi: dict, slaterWeightMin=0, restrictions=None, opResult=None):
    r"""
    Return :math:`|psi' \rangle = op |psi \rangle`.

    If opResult is not None, it is updated to contain information of how the
    operator op acted on the product states in psi.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    op : dict
        Operator of the format
        tuple : amplitude,

        where each tuple describes a scattering

        process. Examples of possible tuples (and their meanings) are:

        ((i, 'c'))  <-> c_i^dagger

        ((i, 'a'))  <-> c_i

        ((i, 'c'), (j, 'a'))  <-> c_i^dagger c_j

        ((i, 'c'), (j, 'c'), (k, 'a'), (l, 'a')) <-> c_i^dagger c_j^dagger c_k c_l
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    slaterWeightMin : float
        Restrict the number of product states by
        looking at `|amplitudes|^2`.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    opResult : dict
        In and output argument.
        If present, the results of the operator op acting on each
        product state in the state psi is added and stored in this
        variable.

    Returns
    -------
    psiNew : dict
        New state of the same format as psi.


    """
    NUM_PROCESSES = int(environ.get("OMP_NUM_THREADS", 1))

    psiNew = dict()

    if opResult is None:
        opResult = dict()
    solved_states = psi.keys() & opResult.keys()
    for state in solved_states:
        addToFirst(psiNew, opResult[state], psi[state])

    step = len(op) // NUM_PROCESSES
    if len(op) % NUM_PROCESSES != 0:
        step += 1
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        op_i = iter(op)
        tasks = {
            executor.submit(
                applyOp_thread_worker,
                {process: op[process] for process in itertools.islice(op_i, step)},
                {state: psi[state] for state in psi if state not in solved_states},
                n_spin_orbitals,
                restrictions,
            )
            for _ in range(0, len(op), step)
        }
        for future in as_completed(tasks):
            psiNew_p, opResult_p = future.result()
            addToFirst(psiNew, psiNew_p)
            for state, res in opResult_p.items():
                tmp = opResult.get(state, {})
                addToFirst(tmp, res)
                opResult[state] = tmp

    return {state: amp for state, amp in psiNew.items() if abs(amp) ** 2 > slaterWeightMin}


def applyOp_worker(output, op, psi, n_spin_orbitals, restrictions):
    opResult = dict()
    psiNew = dict()
    for (process, h), (state, amp) in itertools.product(op, psi.items()):
        state_bits_new = psr.bytes2bitarray(state, n_spin_orbitals)
        signTot = 1
        for i, action in process[-1::-1]:
            if action == "a":
                sign = remove.ubitarray(i, state_bits_new)
            elif action == "c":
                sign = create.ubitarray(i, state_bits_new)
            elif action == "i":
                sign = 1
            signTot *= sign
            if signTot == 0:
                break
        if signTot == 0:
            opResult[state] = opResult.get(state, {})
            continue
        state_new = psr.bitarray2bytes(state_bits_new)
        if not occupation_is_within_restrictions(state_new, n_spin_orbitals, restrictions):
            continue
        psiNew[state_new] = amp * h * signTot + psiNew.get(state_new, 0)
        if state not in opResult:
            opResult[state] = {}
        opResult[state][state_new] = h * signTot + opResult[state].get(state_new, 0)
    output.put((psiNew, opResult))


def applyOp_multiprocess(
    n_spin_orbitals: int, op: dict, psi: dict, slaterWeightMin=0, restrictions=None, opResult=None
):
    r"""
    Return :math:`|psi' \rangle = op |psi \rangle`.

    If opResult is not None, it is updated to contain information of how the
    operator op acted on the product states in psi.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    op : dict
        Operator of the format
        tuple : amplitude,

        where each tuple describes a scattering

        process. Examples of possible tuples (and their meanings) are:

        ((i, 'c'))  <-> c_i^dagger

        ((i, 'a'))  <-> c_i

        ((i, 'c'), (j, 'a'))  <-> c_i^dagger c_j

        ((i, 'c'), (j, 'c'), (k, 'a'), (l, 'a')) <-> c_i^dagger c_j^dagger c_k c_l
    psi : dict
        Multi-configurational state.
        Product states as keys and amplitudes as values.
    slaterWeightMin : float
        Restrict the number of product states by
        looking at `|amplitudes|^2`.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    opResult : dict
        In and output argument.
        If present, the results of the operator op acting on each
        product state in the state psi is added and stored in this
        variable.

    Returns
    -------
    psiNew : dict
        New state of the same format as psi.


    """
    NUM_PROCESSES = int(environ.get("OMP_NUM_THREADS", 1))
    psiNew = dict()
    if opResult is None:
        opResult = dict()
    solved_states = psi.keys() & opResult.keys()
    for state in solved_states:
        addToFirst(psiNew, opResult[state], psi[state])

    done_queue = Queue()

    processes = [None] * NUM_PROCESSES
    for i in range(NUM_PROCESSES):
        processes[i] = Process(
            target=applyOp_worker,
            args=(
                done_queue,
                [op_i for op_i in itertools.islice(op.items(), i, None, NUM_PROCESSES)],
                {state: amp for state, amp in psi.items() if state not in solved_states},
                n_spin_orbitals,
                restrictions,
            ),
        )
        processes[i].start()
    for i in range(NUM_PROCESSES):
        psiNew_p, opResult_p = done_queue.get()
        addToFirst(psiNew, psiNew_p)
        for state, res in opResult_p.items():
            tmp = opResult.get(state, {})
            addToFirst(tmp, res)
            opResult[state] = tmp
    for proc in processes:
        proc.join()

    return {state: amp for state, amp in psiNew.items() if abs(amp) ** 2 > slaterWeightMin}


def get_hamiltonian_matrix(n_spin_orbitals, hOp, basis, mode="sparse_MPI", verbose=True):
    """
    Return Hamiltonian expressed in the provided basis of product states.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    hOp : dict
        tuple : float or complex
        The Hamiltonain operator to diagonalize.
        Each keyword contains ordered instructions
        where to add or remove electrons.
        Values indicate the strengths of
        the corresponding processes.
    basis : tuple
        All product states included in the basis.
    mode : str
        Algorithm for calculating the Hamiltonian.

    """
    # Number of basis states
    n = len(basis)
    basis_index = {basis[i]: i for i in range(n)}
    if rank == 0 and verbose:
        print("Filling the Hamiltonian...")
    progress = 0
    if mode == "dense_serial":
        # h = np.zeros((n,n),dtype=complex)
        h = np.zeros((n, n), dtype=complex)
        for j in range(n):
            if rank == 0 and progress + 10 <= int(j * 100.0 / n):
                progress = int(j * 100.0 / n)
                if verbose:
                    print("{:d}% done".format(progress))
            res = applyOp(n_spin_orbitals, hOp, {basis[j]: 1})
            for k, v in res.items():
                if k in basis_index:
                    h[basis_index[k], j] = v
    elif mode == "dense_MPI":
        # h = np.zeros((n,n),dtype=complex)
        h = np.zeros((n, n), dtype=complex)
        hRank = {}
        jobs = get_job_tasks(rank, ranks, range(n))
        for j in jobs:
            hRank[j] = {}
            if rank == 0 and progress + 10 <= int(j * 100.0 / len(jobs)):
                progress = int(j * 100.0 / len(jobs))
                if verbose:
                    print("{:d}% done".format(progress))
            res = applyOp(n_spin_orbitals, hOp, {basis[j]: 1})
            for k, v in res.items():
                if k in basis_index:
                    hRank[j][basis_index[k]] = v
        # Broadcast Hamiltonian dicts
        for r in range(ranks):
            hTmp = comm.bcast(hRank, root=r)
            for j, hj in hTmp.items():
                for i, hij in hj.items():
                    h[i, j] = hij
    elif mode == "sparse_serial":
        data = []
        row = []
        col = []
        for j in range(n):
            if rank == 0 and progress + 10 <= int(j * 100.0 / n):
                progress = int(j * 100.0 / n)
                if verbose:
                    print("{:d}% done".format(progress))
            res = applyOp(n_spin_orbitals, hOp, {basis[j]: 1})
            for k, v in res.items():
                if k in basis_index:
                    data.append(v)
                    col.append(j)
                    row.append(basis_index[k])
        h = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
    elif mode == "sparse_MPI":
        h = scipy.sparse.csr_matrix(([], ([], [])), shape=(n, n))
        data = []
        row = []
        col = []
        jobs = get_job_tasks(rank, ranks, range(n))
        for j, job in enumerate(jobs):
            res = applyOp(n_spin_orbitals, hOp, {basis[job]: 1})
            for k, v in res.items():
                if k in basis_index:
                    data.append(v)
                    col.append(job)
                    row.append(basis_index[k])
            if rank == 0 and progress + 10 <= int((j + 1) * 100.0 / len(jobs)):
                progress = int((j + 1) * 100.0 / len(jobs))
                if verbose:
                    print("{:d}% done".format(progress))
        # Print out that the construction of Hamiltonian is done
        if rank == 0 and progress != 100:
            progress = 100
            if verbose:
                print("{:d}% done".format(progress))
        hSparse = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
        # Different ranks have information about different basis states.
        # Therefor, need to broadcast and append sparse Hamiltonians
        # h = comm.allreduce(hSparse)
        for r in range(ranks):
            h += comm.bcast(hSparse, root=r)
    return h


def get_hamiltonian_hermitian_operator_from_h_dict(h_dict, basis, parallelization_mode="serial", return_h_local=False):
    """
    Return Hamiltonian expressed in the provided basis of product states
    in matrix format.

    Also return dictionary with product states in basis as keys,
    and basis indices as values.

    Parameters
    ----------
    h_dict : dict
        Elements of the form `|PS> : {hOp|PS>}`,
        where `|PS>` is a product state,
        and `{hOp|PS>}` is a dictionary containing the result of
        the (Hamiltonian) operator hOp acting on the product state `|PS>`.
        The dictionary `{hOp|PS>}` has product states as keys.
        h_dict may contain some product states (as keys) that are not
        part of the active basis.
        Also, if parallelization_mode == 'H_build', each product state in
        the active basis exists as a key in h_dict for only one MPI rank.
    basis : tuple
        All product states included in the basis.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".
    return_h_local : boolean
        If parallelization_mode is not serial, whether to return the
        MPI local Hamiltonian or the full Hamiltonian.

    """
    if parallelization_mode == "serial":
        # In serial mode, the full Hamiltonian is returned.
        assert return_h_local is False
    # Number of basis states
    n = len(basis)
    if parallelization_mode == "serial":
        diagonal = []
        diagonal_indices = []
        data = []
        rows = []
        cols = []
        for col in range(n):
            res = h_dict[basis[col]]
            for key, value in res.items():
                # row = basis_index[key]
                # row = basis.index(key)
                # row = bisect_left(basis,key)
                row = np.searchsorted(basis, key)
                if row == col:
                    diagonal.append(np.real(value))
                    diagonal_indices.append(row)
                elif col < row:
                    data.append(value)
                    cols.append(col)
                    rows.append(row)
        h_triangular = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=complex)
        diagonal = np.array(diagonal, dtype=float)
        diagonal_indices = np.array(diagonal_indices, dtype=np.ulonglong)
        sort_indices = np.argsort(diagonal_indices)
        h = NewHermitianOperator(diagonal[sort_indices], diagonal_indices[sort_indices], h_triangular)
    elif parallelization_mode == "H_build":
        n = comm.allreduce(n, op=MPI.MAX)
        # Loop over product states from the basis
        # which are also stored in h_dict.
        diagonal = []
        diagonal_indices = []
        data = []
        rows = []
        cols = []
        for ps in set(basis).intersection(h_dict.keys()):
            col = np.searchsorted(basis, ps)
            for key, value in h_dict[ps].items():
                row = np.searchsorted(basis, key)
                if row == col:
                    diagonal.append(np.real(value))
                    diagonal_indices.append(row)
                elif col < row:
                    data.append(value)
                    cols.append(col)
                    rows.append(row)
        h_triangular = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=complex)
        diagonal = np.array(diagonal, dtype=float)
        diagonal_indices = np.array(diagonal_indices, dtype=np.ulonglong)
        sort_indices = np.argsort(diagonal_indices)
        h_local = NewHermitianOperator(diagonal[sort_indices], diagonal_indices[sort_indices], h_triangular)
        if return_h_local:
            h = h_local
        else:
            # Different ranks have information about different basis states.
            # Broadcast and append local sparse Hamiltonians.
            h = comm.allreduce(h_local)
    else:
        raise Exception("Wrong parallelization mode!")
    return h


def get_hamiltonian_hermitian_operator_columns(N, comm, verbose):
    """
    Distrubute the columns of the Hamiltonian matrix among the MPI ranks so
    that each rank has roughly the same number of elements.
        [ 0 0 0 0 0 0 0 0 ... ]
        [ 0 1 1 1 1 1 1 1 ... ]
        [ 0 1 2 2 2 2 2 2 ... ]
        [ 0 1 2 3 3 3 3 3 ... ]
    H = [ 0 1 2 3 4 4 4 4 ... ]
        [ 0 1 2 3 4 5 5 5 ... ]
        [ 0 1 2 3 4 5 6 6 ... ]
        [ 0 1 2 3 4 5 6 7 ... ]
        [ . . . . . . . . ... ]
    """
    p = comm.size
    start = np.empty((p))
    stop = np.empty((p))
    if p > N:
        stop = np.arange(1, p + 1)
        stop[N:] = N
    else:
        stop[-1] = 0
        for i in range(p):
            stop[i] = int((N - 1 - np.sqrt((stop[i - 1] - N) ** 2 - N**2 / p)))
        remainder = N - stop[-1]
        inc_rank = 0
        inc = 1
        while remainder > 0 and inc_rank < p - 1:
            stop[-1] = 0
            try:
                for i in range(inc_rank, p):
                    stop[i] = int((N - np.sqrt((stop[i - 1] - N) ** 2 - N**2 / p))) + inc
                remainder = N - stop[-1]
                inc += 1
            except ValueError:
                stop[inc_rank] -= 1
                inc_rank += 1
                inc = 1
    start[1:] = stop[:-1]
    start[0] = 0
    stop[-1] = N
    average_number_of_elements = N**2 / p
    number_of_elements = (start - N) ** 2 - (stop - N) ** 2
    if verbose:
        print(f"{N=}, {p=}")
        print("New")
        print(f"|-->max(|N - N_avg|): {np.max(np.abs(number_of_elements - average_number_of_elements))}")
        print(
            f"--->max(|N - N_avg|/N_avg): {np.max(np.abs(number_of_elements - average_number_of_elements))/average_number_of_elements}"
        )
    return start[comm.rank], stop[comm.rank]


def get_hamiltonian_hermitian_operator_from_h_dict_new(h_dict, basis, return_h_local=False, comm=None):
    """
    Return Hamiltonian expressed in the provided basis of product states
    in matrix format.

    Also return dictionary with product states in basis as keys,
    and basis indices as values.

    Parameters
    ----------
    h_dict : dict
        Elements of the form `|PS> : {hOp|PS>}`,
        where `|PS>` is a product state,
        and `{hOp|PS>}` is a dictionary containing the result of
        the (Hamiltonian) operator hOp acting on the product state `|PS>`.
        The dictionary `{hOp|PS>}` has product states as keys.
        h_dict may contain some product states (as keys) that are not
        part of the active basis.
        Also, if parallelization_mode == 'H_build', each product state in
        the active basis exists as a key in h_dict for only one MPI rank.
    basis : tuple
        All product states included in the basis.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".
    return_h_local : boolean
        If parallelization_mode is not serial, whether to return the
        MPI local Hamiltonian or the full Hamiltonian.

    """
    if comm is None:
        assert not return_h_local
    if return_h_local:
        assert comm is not None
    # Number of basis states
    n = basis.size

    # col_start, col_end = get_hamiltonian_hermitian_operator_columns(n, comm, verbose = True)
    # col_states = basis[int(col_start) : int(col_end)]

    unique_states = list(set([state for state in h_dict] + [state for col in h_dict for state in h_dict[col]]))
    unique_indices = basis.index(unique_states)

    state_to_index = {}
    for i in range(len(unique_states)):
        state = unique_states[i]
        index = unique_indices[i]
        state_to_index[state] = index

    flat_column_indices = []
    flat_row_indices = []
    flat_values = []
    for col in h_dict.keys():
        for row in (h_dict[col]).keys():
            flat_column_indices.append(state_to_index[col])
            flat_row_indices.append(state_to_index[row])
            flat_values.append(h_dict[col][row])

    cols = np.empty((0), dtype=int)
    rows = np.empty((0), dtype=int)
    data = np.empty((0), dtype=complex)
    diagonal_indices = np.empty((0), dtype=np.ulonglong)
    diagonal = np.empty((0), dtype=float)
    for i in range(len(flat_values)):
        row = flat_row_indices[i]
        col = flat_column_indices[i]
        val = flat_values[i]
        if row == col:
            diagonal_indices = np.append(diagonal_indices, [np.ulonglong(row)])
            diagonal = np.append(diagonal, [np.real(val)])
        elif col < row:
            cols = np.append(cols, [col])
            rows = np.append(rows, [row])
            data = np.append(data, [val])

    h_triangular = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=complex)
    h_local = NewHermitianOperator(diagonal, diagonal_indices, h_triangular)
    if return_h_local:
        h = h_local
    else:
        # Different ranks have information about different basis states.
        # Broadcast and append local sparse Hamiltonians.
        h = comm.allreduce(h_local, op=MPI.SUM)
    return h


def get_hamiltonian_matrix_from_h_dict(
    h_dict, basis, parallelization_mode="serial", return_h_local=False, mode="sparse"
):
    """
    Return Hamiltonian expressed in the provided basis of product states
    in matrix format.

    Also return dictionary with product states in basis as keys,
    and basis indices as values.

    Parameters
    ----------
    h_dict : dict
        Elements of the form `|PS> : {hOp|PS>}`,
        where `|PS>` is a product state,
        and `{hOp|PS>}` is a dictionary containing the result of
        the (Hamiltonian) operator hOp acting on the product state `|PS>`.
        The dictionary `{hOp|PS>}` has product states as keys.
        h_dict may contain some product states (as keys) that are not
        part of the active basis.
        Also, if parallelization_mode == 'H_build', each product state in
        the active basis exists as a key in h_dict for only one MPI rank.
    basis : tuple
        All product states included in the basis.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".
    return_h_local : boolean
        If parallelization_mode is not serial, whether to return the
        MPI local Hamiltonian or the full Hamiltonian.
    mode : str
        Algorithm for calculating the Hamiltonian and type format of
        returned Hamiltonian.
        'dense' or 'sparse'.

    """
    if parallelization_mode == "serial":
        # In serial mode, the full Hamiltonian is returned.
        assert return_h_local is False
    # Number of basis states
    n = len(basis)
    basis_index = {basis[i]: i for i in range(n)}
    if mode == "dense" and parallelization_mode == "serial":
        h = np.zeros((n, n), dtype=complex)
        for j in range(n):
            res = h_dict[basis[j]]
            for k, v in res.items():
                h[basis_index[k], j] = v
    elif mode == "sparse" and parallelization_mode == "serial":
        data = []
        row = []
        col = []
        for j in range(n):
            res = h_dict[basis[j]]
            for k, v in res.items():
                data.append(v)
                col.append(j)
                row.append(basis_index[k])
        h = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
    elif mode == "sparse" and parallelization_mode == "H_build":
        n = comm.allreduce(n, op=MPI.MAX)
        # Loop over product states from the basis
        # which are also stored in h_dict.
        data = []
        row = []
        col = []
        for ps in set(basis).intersection(h_dict.keys()):
            for k, v in h_dict[ps].items():
                data.append(v)
                col.append(basis_index[ps])
                row.append(basis_index[k])
        h_local = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
        if return_h_local:
            h = h_local
        else:
            # Different ranks have information about different basis states.
            # Broadcast and append local sparse Hamiltonians.
            h = comm.allreduce(h_local)
    else:
        raise Exception("Wrong input parameters")
    return h, basis_index


def expand_basis(n_spin_orbitals, h_dict, hOp, basis0, restrictions, parallelization_mode="serial"):
    """
    Return basis.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    h_dict : dict
        Elements of the form `|PS> : {hOp|PS>}`,
        where `|PS>` is a product state,
        and `{hOp|PS>}` is a dictionary containing the result of
        the (Hamiltonian) operator hOp acting on the product state `|PS>`.
        The dictionary `{hOp|PS>}` has product states as keys.
        New elements might be added to this variable.
        h_dict may contain some product states (as keys) that will not
        be part of the final active basis.
        Also, if parallelization_mode == 'H_build', each product state in
        the active basis exists as a key in h_dict for only one MPI rank.
    hOp : dict
        The Hamiltonian. With elements of the form:
        process : h_value
    basis0 : tuple
        List of product states.
        These product states are used to generate more basis states.
    restrictions : dict
        Restriction the occupation of generated product states.
    parallelization_mode : str
        Parallelization mode. Either: 'serial' or 'H_build'.

    Returns
    -------
    basis : tuple
        The restricted active space basis of product states.

    """
    # Copy basis0, to avoid changing it when the basis grows
    basis = list(basis0)
    i = 0
    n = len(basis)
    if parallelization_mode == "serial":
        while i < n:
            basis_set = frozenset(basis)
            basis_new = set()
            for b in basis[i:n]:
                if b in h_dict:
                    res = h_dict[b]
                else:
                    res = applyOp(n_spin_orbitals, hOp, {b: 1}, restrictions=restrictions)
                    h_dict[b] = res
                basis_new.update(set(res.keys()).difference(basis_set))
            i = n
            # Add basis_new to basis.
            basis += sorted(basis_new)
            n = len(basis)
    elif parallelization_mode == "H_build":
        h_dict_new_local = {}
        while i < n:
            basis_set = frozenset(basis)
            basis_new_local = set()

            # Among the product states in basis[i:n], first consider
            # the product states which exist in h_dict.
            states_setA_local = set(basis[i:n]).intersection(h_dict.keys())
            # Loop through these product states
            for ps in states_setA_local:
                res = h_dict[ps]
                basis_new_local.update(set(res.keys()).difference(basis_set))

            # Now consider the product states in basis[i:n] which
            # does not exist in h_dict for any MPI rank.
            if rank == 0:
                states_setB = set(basis[i:n]) - states_setA_local
                for r in range(1, ranks):
                    states_setB.difference_update(comm.recv(source=r, tag=0))
                states_tupleB = tuple(states_setB)
            else:
                # Send product states to rank 0.
                comm.send(states_setA_local, dest=0, tag=0)
                states_tupleB = None
            states_tupleB = comm.bcast(states_tupleB, root=0)
            # Distribute and then loop through "unknown" product states
            for ps_indexB in get_job_tasks(rank, ranks, range(len(states_tupleB))):
                # One product state.
                ps = states_tupleB[ps_indexB]
                res = applyOp(n_spin_orbitals, hOp, {ps: 1}, restrictions=restrictions)
                h_dict_new_local[ps] = res
                basis_new_local.update(set(res.keys()).difference(basis_set))

            # Add unique elements of basis_new_local into basis_new
            basis_new = set()
            for r in range(ranks):
                basis_new.update(comm.bcast(basis_new_local, root=r))
            # Add basis_new to basis.
            # It is important that all ranks use the same order of the
            # product states. This is one way to ensure the same ordering.
            # But any ordering is fine, as long it's the same for all MPI ranks.
            basis += sorted(basis_new)
            # Updated total number of product states |PS> in
            # the basis where know H|PS>.
            i = n
            # Updated total number of product states needed to consider.
            n = len(basis)
        # Add new elements to h_dict, but only local contribution.
        h_dict.update(h_dict_new_local)
    else:
        raise Exception("Wrong parallelization parameter.")
    # return tuple(sorted(basis))
    return list(sorted(basis))


def combine_sets(set_1, set_2, datatype):
    return set_1 | set_2


combine_sets_op = MPI.Op.Create(combine_sets, commute=True)


def expand_basis_new(n_spin_orbitals, h_dict, hOp, basis0, restrictions, parallelization_mode="serial"):
    """
    Return basis.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    h_dict : dict
        Elements of the form `|PS> : {hOp|PS>}`,
        where `|PS>` is a product state,
        and `{hOp|PS>}` is a dictionary containing the result of
        the (Hamiltonian) operator hOp acting on the product state `|PS>`.
        The dictionary `{hOp|PS>}` has product states as keys.
        New elements might be added to this variable.
        h_dict may contain some product states (as keys) that will not
        be part of the final active basis.
        Also, if parallelization_mode == 'H_build', each product state in
        the active basis exists as a key in h_dict for only one MPI rank.
    hOp : dict
        The Hamiltonian. With elements of the form:
        process : h_value
    basis0 : tuple
        List of product states.
        These product states are used to generate more basis states.
    restrictions : dict
        Restriction the occupation of generated product states.
    parallelization_mode : str
        Parallelization mode. Either: 'serial' or 'H_build'.

    Returns
    -------
    basis : tuple
        The restricted active space basis of product states.

    """
    # Copy basis0, to avoid changing it when the basis grows
    basis = list(basis0)
    i = 0
    n = len(basis)
    if parallelization_mode == "serial":
        while i < n:
            basis_set = frozenset(basis)
            basis_new = set()
            for b in basis[i:n]:
                if b in h_dict:
                    res = h_dict[b]
                else:
                    res = applyOp(n_spin_orbitals, hOp, {b: 1}, restrictions=restrictions)
                    h_dict[b] = res
                basis_new.update(set(res.keys()).difference(basis_set))
            i = n
            # Add basis_new to basis.
            basis += sorted(basis_new)
            n = len(basis)
    elif parallelization_mode == "H_build":
        h_dict_new_local = {}
        while i < n:
            basis_set = frozenset(basis)
            basis_new_local = set()

            # Among the product states in basis[i:n], first consider
            # the product states which exist in h_dict.
            states_setA_local = set(basis[i:n]).intersection(h_dict.keys())
            # Loop through these product states
            for ps in states_setA_local:
                res = h_dict[ps]
                basis_new_local.update(set(res.keys()).difference(basis_set))

            # Now consider the product states in basis[i:n] which
            # does not exist in h_dict for any MPI rank.
            states_setB = comm.allreduce(
                states_setA_local,
                op=combine_sets_op,
            )
            states_tupleB = tuple(sorted(set(basis[i:n]) - states_setB))
            # All ranks must agree on the order of the product states, so sort them

            # Distribute and then loop through "unknown" product states
            for ps_indexB in get_job_tasks(rank, ranks, range(len(states_tupleB))):
                # One product state.
                ps = states_tupleB[ps_indexB]
                res = applyOp(n_spin_orbitals, hOp, {ps: 1}, restrictions=restrictions)
                h_dict_new_local[ps] = res
                basis_new_local.update(set(res.keys()).difference(basis_set))

            # Add unique elements of basis_new_local into basis_new

            basis_new = comm.allreduce(basis_new_local, op=combine_sets_op)
            # Add basis_new to basis.
            # It is important that all ranks use the same order of the
            # product states. This is one way to ensure the same ordering.
            # But any ordering is fine, as long it's the same for all MPI ranks.
            basis += sorted(basis_new)
            # Updated total number of product states |PS> in
            # the basis where we know H|PS>.
            i = n
            # Updated total number of product states needed to consider.
            n = len(basis)
        # Add new elements to h_dict, but only local contribution.
        h_dict.update(h_dict_new_local)
    else:
        raise Exception("Wrong parallelization parameter.")
    # return tuple(sorted(basis))
    return list(sorted(basis))
    # return np.array(sorted(basis))


def expand_basis_and_build_hermitian_hamiltonian(
    n_spin_orbitals,
    h_dict,
    hOp,
    basis0,
    restrictions,
    parallelization_mode="serial",
    return_h_local=False,
    verbose=True,
):
    """
    Return Hamiltonian in matrix format.

    Also return dictionary with product states in basis as keys,
    and basis indices as values.

    Also possibly to add new product state keys to h_dict.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    h_dict : dict
        Elements of the form `|PS> : {hOp|PS>}`,
        where `|PS>` is a product state,
        and {hOp|PS>} is a dictionary containing the result of
        the (Hamiltonian) operator hOp acting on the product state `|PS>`.
        The dictionary `{hOp|PS>}` has product states as keys.
        New elements might be added to this variable.
        h_dict may contain some product states (as keys) that will not
        be part of the final active basis.
        Also, if parallelization_mode == 'H_build', each product state in
        the active basis exists as a key in h_dict for only one MPI rank.
    hOp : dict
        The Hamiltonian. With elements of the form process : h_value
    basis0 : tuple
        List of product states.
        These product states are used to generate more basis states.
    restrictions : dict
        Restriction the occupation of generated product states.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".
    return_h_local : boolean
        If parallelization_mode is not serial, whether to return the
        MPI local Hamiltonian or the full Hamiltonian.

    Returns
    -------
    h : scipy sparse csr_matrix
        The Hamiltonian acting on the relevant product states.
    basis_index : dict
        Elements of the form `|PS> : i`,
        where `|PS>` is a product state and i an integer.

    """
    # Measure time to expand basis
    if rank == 0:
        t0 = time.perf_counter()
    # Obtain tuple containing different product states.
    # Possibly add new product state keys to h_dict.
    basis = np.array([state for state in basis0])
    # expanded_basis = expand_basis_new(n_spin_orbitals, h_dict, hOp, [state for state in basis], restrictions, parallelization_mode)
    expanded_basis = expand_basis(n_spin_orbitals, h_dict, hOp, basis, restrictions, parallelization_mode)
    # basis0.expand(n_spin_orbitals, hOp, h_dict, restrictions)
    # expanded_basis = np.array([state for state in basis0])
    if rank == 0 and verbose:
        print("time(expand_basis) = {:.3f} seconds.".format(time.perf_counter() - t0))
        t0 = time.perf_counter()
    # Obtain Hamiltonian in HermitianOperator form.
    h = get_hamiltonian_hermitian_operator_from_h_dict(h_dict, expanded_basis, parallelization_mode, return_h_local)
    if verbose:
        print("time(get_hamiltonian_matrix_from_h_dict) = {:.3f} seconds.".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    if parallelization_mode == "H_build":
        # Total Hamiltonian size. Only used for printing it.
        len_h_dict_total = comm.reduce(len(h_dict))
        if rank == 0 and verbose:
            print(
                "Hamiltonian basis sizes: "
                f"len(basis_index) = {len(expanded_basis)}, "
                f"np.shape(h)[0] = {np.shape(h)[0]}, "
                f"len(h_dict) = {len(h_dict)}, "
                f"len(h_dict_total) = {len_h_dict_total}"
            )
    elif parallelization_mode == "serial":
        if rank == 0 and verbose:
            print(
                "Hamiltonian basis sizes: "
                f"len(basis_index) = {len(expanded_basis)}, "
                f"np.shape(h)[0] = {np.shape(h)[0]}, "
                f"len(h_dict) = {len(h_dict)}, "
            )

    return h, expanded_basis


def expand_basis_and_build_hermitian_hamiltonian_new(
    n_spin_orbitals,
    h_dict,
    hOp,
    basis,
    restrictions,
    parallelization_mode="serial",
    return_h_local=False,
    verbose=True,
):
    """
    Return Hamiltonian in matrix format.

    Also return dictionary with product states in basis as keys,
    and basis indices as values.

    Also possibly to add new product state keys to h_dict.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    h_dict : dict
        Elements of the form `|PS> : {hOp|PS>}`,
        where `|PS>` is a product state,
        and {hOp|PS>} is a dictionary containing the result of
        the (Hamiltonian) operator hOp acting on the product state `|PS>`.
        The dictionary `{hOp|PS>}` has product states as keys.
        New elements might be added to this variable.
        h_dict may contain some product states (as keys) that will not
        be part of the final active basis.
        Also, if parallelization_mode == 'H_build', each product state in
        the active basis exists as a key in h_dict for only one MPI rank.
    hOp : dict
        The Hamiltonian. With elements of the form process : h_value
    basis0 : tuple
        List of product states.
        These product states are used to generate more basis states.
    restrictions : dict
        Restriction the occupation of generated product states.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".
    return_h_local : boolean
        If parallelization_mode is not serial, whether to return the
        MPI local Hamiltonian or the full Hamiltonian.

    Returns
    -------
    h : scipy sparse csr_matrix
        The Hamiltonian acting on the relevant product states.
    basis_index : dict
        Elements of the form `|PS> : i`,
        where `|PS>` is a product state and i an integer.

    """
    # Measure time to expand basis
    if verbose:
        t0 = time.perf_counter()
    h_dict = basis.expand(hOp, h_dict)

    if verbose:
        print("time(expand_basis) = {:.3f} seconds.".format(time.perf_counter() - t0))
        t0 = time.perf_counter()
    # Obtain Hamiltonian in HermitianOperator form.
    h = get_hamiltonian_hermitian_operator_from_h_dict_new(h_dict, basis, return_h_local, comm=comm)
    if verbose:
        print("time(get_hamiltonian_matrix_from_h_dict) = {:.3f} seconds.".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    if parallelization_mode == "H_build":
        # Total Hamiltonian size. Only used for printing it.
        len_h_dict_total = comm.reduce(len(h_dict))
        if verbose:
            print(
                "Hamiltonian basis sizes: "
                f"len(basis_index) = {len(basis)}, "
                f"np.shape(h)[0] = {np.shape(h)[0]}, "
                f"len(h_dict) = {len(h_dict)}, "
                f"len(h_dict_total) = {len_h_dict_total}",
            )
    elif parallelization_mode == "serial":
        if rank == 0 and verbose:
            print(
                "Hamiltonian basis sizes: "
                f"len(basis_index) = {len(basis)}, "
                f"np.shape(h)[0] = {np.shape(h)[0]}, "
                f"len(h_dict) = {len(h_dict)}, "
            )

    return h, h_dict, basis


def expand_basis_and_hamiltonian(
    n_spin_orbitals,
    h_dict,
    hOp,
    basis0,
    restrictions,
    parallelization_mode="serial",
    return_h_local=False,
    verbose=True,
):
    """
    Return Hamiltonian in matrix format.

    Also return dictionary with product states in basis as keys,
    and basis indices as values.

    Also possibly to add new product state keys to h_dict.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    h_dict : dict
        Elements of the form `|PS> : {hOp|PS>}`,
        where `|PS>` is a product state,
        and {hOp|PS>} is a dictionary containing the result of
        the (Hamiltonian) operator hOp acting on the product state `|PS>`.
        The dictionary `{hOp|PS>}` has product states as keys.
        New elements might be added to this variable.
        h_dict may contain some product states (as keys) that will not
        be part of the final active basis.
        Also, if parallelization_mode == 'H_build', each product state in
        the active basis exists as a key in h_dict for only one MPI rank.
    hOp : dict
        The Hamiltonian. With elements of the form process : h_value
    basis0 : tuple
        List of product states.
        These product states are used to generate more basis states.
    restrictions : dict
        Restriction the occupation of generated product states.
    parallelization_mode : str
        Parallelization mode. Either: "serial" or "H_build".
    return_h_local : boolean
        If parallelization_mode is not serial, whether to return the
        MPI local Hamiltonian or the full Hamiltonian.

    Returns
    -------
    h : scipy sparse csr_matrix
        The Hamiltonian acting on the relevant product states.
    basis_index : dict
        Elements of the form `|PS> : i`,
        where `|PS>` is a product state and i an integer.

    """
    # Measure time to expand basis
    if rank == 0:
        t0 = time.perf_counter()
    # Obtain tuple containing different product states.
    # Possibly add new product state keys to h_dict.
    basis = expand_basis(n_spin_orbitals, h_dict, hOp, basis0, restrictions, parallelization_mode)
    # basis = expand_basis(n_spin_orbitals, h_dict, hOp, basis0, restrictions, 'serial')
    if rank == 0 and verbose:
        print("time(expand_basis) = {:.3f} seconds.".format(time.perf_counter() - t0))
        t0 = time.perf_counter()
    # Obtain Hamiltonian in matrix format.
    h, basis_index = get_hamiltonian_matrix_from_h_dict(h_dict, basis, parallelization_mode, return_h_local)
    if rank == 0 and verbose:
        print("time(get_hamiltonian_matrix_from_h_dict) = {:.3f} seconds.".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    if parallelization_mode == "H_build":
        # Total Hamiltonian size. Only used for printing it.
        len_h_dict_total = comm.reduce(len(h_dict))
        if rank == 0 and verbose:
            print(
                "Hamiltonian basis sizes: "
                f"len(basis_index) = {len(basis_index)}, "
                f"np.shape(h)[0] = {np.shape(h)[0]}, "
                f"len(h_dict) = {len(h_dict)}, "
                f"len(h_dict_total) = {len_h_dict_total}"
            )
    elif parallelization_mode == "serial":
        if rank == 0 and verbose:
            print(
                "Hamiltonian basis sizes: "
                f"len(basis_index) = {len(basis_index)}, "
                f"np.shape(h)[0] = {np.shape(h)[0]}, "
                f"len(h_dict) = {len(h_dict)}, "
            )

    return h, basis_index


def get_tridiagonal_krylov_vectors(h, psi0, krylovSize, h_local=False, mode="sparse", verbose=True, tol=1e-16):
    r"""
    return tridiagonal elements of the Krylov Hamiltonian matrix.

    Parameters
    ----------
    h : sparse matrix (N,N)
        Hamiltonian.
    psi0 : complex array(N)
        Initial Krylov vector.
    krylovSize : int
        Size of the Krylov space.
    mode : str
        'dense' or 'sparse'
        Option 'sparse' should be best.

    """
    if rank == 0:
        # Measure time to get tridiagonal krylov vectors.
        t0 = time.perf_counter()
    # This is probably not a good idea in terms of computational speed
    # since the Hamiltonians typically are extremely sparse.
    if mode == "dense":
        h = h.toarray()
    # Number of basis states
    n = len(psi0)
    # Unnecessary (and impossible) to find more than n Krylov basis vectors.
    krylovSize = min(krylovSize, n)

    # Allocate tri-diagonal matrix elements
    alpha = np.zeros(krylovSize, dtype=float)
    beta = np.zeros(krylovSize - 1, dtype=float)
    # Allocate space for Krylov state vectors.
    # Do not save all Krylov vectors to save memory.
    # v = np.zeros((2,n), dtype=complex)
    v = np.zeros((2, n), dtype=complex)
    # Initialization...
    v[0, :] = psi0
    Q = np.zeros((psi0.shape[0], krylovSize), dtype=complex)
    Q[:, 0] = v[0, :]

    # Start with Krylov iterations.
    if h_local:
        if rank == 0 and verbose:
            print("MPI parallelization in the Krylov loop...")
        # The Hamiltonian matrix is distributed over MPI ranks,
        # i.e. H = sum_r Hr
        # This means a multiplication of the Hamiltonian matrix H
        # with a vector x can be written as:
        # y = H*x = sum_r Hr*x = sum_r y_r

        # Initialization...
        wp_local = h.dot(v[0, :])
        # Reduce vector wp_local to the vector wp at rank 0.
        wp = np.zeros_like(wp_local)
        comm.Reduce(wp_local, wp)
        if rank == 0:
            alpha[0] = np.dot(np.conj(wp), v[0, :]).real
            w = wp - alpha[0] * v[0, :]
            w -= np.linalg.multi_dot([Q, np.conj(Q.T), w])
        # Construct Krylov states,
        # and more importantly the vectors alpha and beta
        converged = False
        for j in range(1, krylovSize):
            if rank == 0:
                beta[j - 1] = sqrt(np.sum(np.abs(w) ** 2))
                if abs(beta[j - 1]) > tol:
                    v[1, :] = w / beta[j - 1]
                    Q[:, j] = v[1, :]
                else:
                    # Pick normalized state v[j],
                    # orthogonal to v[0],v[1],v[2],...,v[j-1]
                    # raise ValueError(('Warning: beta==0, '
                    #                   + 'implementation absent!'))
                    # print ("ValueError(\'Warning: beta==0, implementation absent!\')")
                    converged = True
            converged = comm.bcast(converged, root=0)
            if converged:
                break
            # Broadcast vector v[1,:] from rank 0 to all ranks.
            comm.Bcast(v[1, :], root=0)
            wp_local = h.dot(v[1, :])
            # Reduce vector wp_local to the vector wp at rank 0.
            wp = np.zeros_like(wp_local)
            comm.Reduce(wp_local, wp)
            if rank == 0:
                alpha[j] = np.dot(np.conj(wp), v[1, :]).real
                w = wp - alpha[j] * v[1, :] - beta[j - 1] * v[0, :]
                w -= np.linalg.multi_dot([Q[:, : j + 1], np.conj(Q[:, : j + 1].T), w])
                v[0, :] = v[1, :]
    else:
        # Initialization...
        wp = h.dot(v[0, :])
        alpha[0] = np.dot(np.conj(wp), v[0, :]).real
        w = wp - alpha[0] * v[0, :]
        w -= np.linalg.multi_dot([Q, np.conj(Q.T), w])
        # Construct Krylov states,
        # and more importantly the vectors alpha and beta
        for j in range(1, krylovSize):
            beta[j - 1] = sqrt(np.sum(np.abs(w) ** 2))
            if abs(beta[j - 1]) > tol:
                v[1, :] = w / beta[j - 1]
                Q[:, j] = v[1, :]
            else:
                # Pick normalized state v[j],
                # orthogonal to v[0],v[1],v[2],...,v[j-1]
                # raise ValueError('Warning: beta==0, implementation absent!')
                # print ("ValueError(\'Warning: beta==0, implementation absent!\')")
                break
            wp = h.dot(v[1, :])
            alpha[j] = np.dot(np.conj(wp), v[1, :]).real
            w = wp - alpha[j] * v[1, :] - beta[j - 1] * v[0, :]
            w -= np.linalg.multi_dot([Q[:, : j + 1], np.conj(Q[:, : j + 1].T), w])
            v[0, :] = v[1, :]
    if rank == 0 and verbose:
        print("time(get_tridiagonal_krylov_vectors) = {:.5f} seconds.".format(time.perf_counter() - t0))
    return alpha, beta


def add(psi1, psi2, mul=1):
    r"""
    Return :math:`|\psi\rangle = |\psi_1\rangle + mul * |\psi_2\rangle`

    Parameters
    ----------
    psi1 : dict
    psi2 : dict
    mul : int, float or complex
        Optional

    Returns
    -------
    psi : dict

    """
    psi = psi1.copy()
    for s, a in psi2.items():
        psi[s] = mul * a + psi.get(s, 0)
        # if s in psi:
        #     psi[s] += mul * a
        # else:
        #     psi[s] = mul * a
    return psi


def norm2(psi):
    r"""
    Return :math:`\langle psi|psi \rangle`.

    Parameters
    ----------
    psi : dict
        Multi configurational state.

    """
    return sum(abs(a) ** 2 for a in psi.values())


def arrayOp(nBaths, pOp):
    r"""
    Returns the array A of pOp = sum_{i,j} A_{ij} c^dagger_i c_j, where i,j are given by the c2i function.

    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    pOp: dict
        Multi configurational state.

    """
    dsize = 0
    for l, nb in nBaths.items():
        dsize += nb + (2 * l + 1) * 2
    a = np.zeros((dsize, dsize), dtype=complex)
    for t, val in pOp.items():
        # Only one particle terms
        if len(t) == 2:
            # Accept both ((l,m,s,b),'c/a') and (l,m,s,b)
            if len(t[0]) == 2:
                if t[0][1] == "c" and t[1][1] == "a":
                    a[c2i(nBaths, t[0][0]), c2i(nBaths, t[1][0])] = val
                elif t[0][1] == "a" and t[1][1] == "c":
                    if t[0][0] == t[1][0]:
                        a[c2i(nBaths, t[1][0]), c2i(nBaths, t[0][0])] = 1.0 - val
                    else:
                        a[c2i(nBaths, t[1][0]), c2i(nBaths, t[0][0])] = -val
            else:
                a[c2i(nBaths, t[0]), c2i(nBaths, t[1])] = val
    return a


def arrayOp2Dict(nbaths, opsArray):
    r"""
    Return an array of dicts of the form {(i,j):val ...} corresponding to the
    operators c^dagger_i c_j, stored in the opsArray
    Parameters
    ----------
    nbaths : dict
        l : nb
    opsArray : [dict]
        [{(((l, m, s, b), 'c'), ((l, m, s, b), 'a'))}, ...]
    """
    res = []
    for ops in opsArray:
        res.append(op2Dict(nbaths, ops))
    return res


def op2Dict(nBaths, ops):
    r"""
    returns a dict of the form {(i,j):val,...} correspoding to the opeator c^dagger_i c_j
    where i, j are obtained from c2i
    Parameters
    ----------
    nBaths : dict
        angular momentum : number of bath sets
    ops: dict
        Multi configurational state.
    """
    d = {}
    for t, val in ops.items():
        # Only one particle terms
        if len(t) == 2:
            # Accept both ((l,m,s,b),'c/a') and (l,m,s,b)
            if len(t[0]) == 2:
                if t[0][1] == "c" and t[1][1] == "a":
                    d[((c2i(nBaths, t[0][0]), "c"), (c2i(nBaths, t[1][0]), "a"))] = val
                elif t[0][1] == "a" and t[1][1] == "c":
                    if t[0][0] == t[1][0]:
                        d[((c2i(nBaths, t[1][0]), "c"), (c2i(nBaths, t[0][0]), "a"))] = 1.0 - val
                    else:
                        d[((c2i(nBaths, t[1][0]), "c"), (c2i(nBaths, t[0][0]), "a"))] = -val
            else:
                d[((c2i(nBaths, t[0]), "c"), (c2i(nBaths, t[1]), "a"))] = val
    return d


def combineOp(nBaths, op1, op2):
    r"""
    Return a dict of the form {(i, j) : val, ...} corresponding to the
    operator op1*op2
    Parameters
    ----------
    op1 : dict
     (i, j) : val
    op2 : dict
     (i, j) : val
    """
    mOp1 = iOpToMatrix(nBaths, op1)
    mOp2 = iOpToMatrix(nBaths, op2)

    newOp = np.matmul(mOp1, mOp2)

    return matrixToIOp(newOp)


def iOpToMatrix(nBaths, op):
    r"""
    Return the matrix representation of op
    Parameters
    ----------
    nBaths : dict
     l : nb
    op : dict
     (i, j) : val
    """
    dsize = 0
    for l, nb in nBaths.items():
        dsize += nb + (2 * l + 1) * 2
    m = np.zeros((dsize, dsize), dtype=complex)
    for ((i, opi), (j, opj)), val in op.items():
        if opi == "c" and opj == "a":
            m[i, j] = val
        elif opj == "c" and opi == "a":
            if i == j:
                m[i, j] = 1 - val
            else:
                m[i, j] = -val
    return m


def matrixToIOp(mat):
    r"""
    Return a dict containing the non-zero elements of the matrix mat
    Parameters
    ----------
    mat : numpy matrix
    """
    rows, columns = mat.shape
    res = {}
    for i in range(rows):
        for j in range(columns):
            if abs(mat[i, j]) > 0:
                res[((i, "c"), (j, "a"))] = mat[i, j]
    return res


def c2i_op(nBaths, c_op):
    i_op = {}
    for process, value in c_op.items():
        i_op[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value
    return i_op


def i2c_op(nBaths, i_op):
    c_op = {}
    for ((i, opi), (j, opj)), val in iDict.items():
        c_op[((i2c(nBaths, i), opi), (i2c(nBaths, j), opj))] = val
    return c_op


def i2cDict2Array(nBaths, i_ops):
    res = []
    for i_op in i_ops:
        res.append(i2c_op(nBaths, i_op))
    return res
