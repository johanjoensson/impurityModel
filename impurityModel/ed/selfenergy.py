from math import sqrt
import numpy as np
from collections import OrderedDict
from mpi4py import MPI
import pickle
import time
import argparse
import h5py

import scipy as sp

from impurityModel.ed.get_spectra import get_hamiltonian_operator, get_h0_operator, get_restrictions
from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.finite import c2i
from impurityModel.ed.finite import daggerOp, applyOp, inner, add, norm2
from impurityModel.ed.average import k_B


class UnphysicalSelfenergy(Exception):
    pass

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size

eV_to_Ry = 1/13.605693122994

def get_selfenergy(
        clustername,
        h0_filename,
        ls, 
        nBaths, 
        nValBaths,
        n0imps, 
        dnTols, 
        dnValBaths, 
        dnConBaths,
        Fdd, 
        xi, 
        chargeTransferCorrection,
        hField, 
        nPsiMax,
        nPrintSlaterWeights, 
        tolPrintOccupation,
        T,
        energy_cut,
        delta,
        verbose):

    # omega_mesh = np.linspace(-25, 25, 4000)
    omega_mesh = 1j*np.pi*T*np.arange(start = 1, step = 2, stop = 1024)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        t0 = time.time()
    # -- System information --
    nBaths = {ls: nBaths}
    nBaths[1] = 0
    nValBaths = {ls: nValBaths}
    nValBaths[1] = 0

    # -- Basis occupation information --
    n0imps = {ls: n0imps}
    n0imps[1] = 6
    dnTols = {ls: dnTols}
    dnTols[1] = 0
    dnValBaths = {ls: dnValBaths}
    dnValBaths[1] = 0
    dnConBaths = {ls: dnConBaths}
    dnConBaths[1] = 0

    # -- Spectra information --
    # Energy cut in eV.
    energy_cut *= k_B * T

    # -- Occupation restrictions for excited states --
    restrictions = get_restrictions(    l = ls, 
                                        n0imps = n0imps, 
                                        nBaths = nBaths, 
                                        nValBaths = nValBaths, 
                                        dnTols = dnTols, 
                                        dnValBaths = dnValBaths, 
                                        dnConBaths = dnConBaths)

    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    if rank == 0 and verbose:
        print("#spin-orbitals:", n_spin_orbitals)

    # Hamiltonian
    if rank == 0:
        print("Construct the Hamiltonian operator...")
    hOp = get_hamiltonian_operator(
        nBaths,
        nValBaths,
        [Fdd, [0 for _ in range(3)], [0 for _ in range(3)], [0 for _ in range(4)]],
        [0, xi],
        [n0imps, chargeTransferCorrection],
        hField,
        h0_filename,
        rank,
        verbose = verbose
    )
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0 and verbose:
        print("{:d} processes in the Hamiltonian.".format(len(hOp)))
    # Many body basis for the ground state
    if rank == 0:
        print("Create basis...")
    basis = finite.get_basis(nBaths, nValBaths, dnValBaths, dnConBaths, dnTols, n0imps, verbose = verbose)
    if rank == 0 and verbose:
        print("#basis states = {:d}".format(len(basis)))
    # Diagonalization of restricted active space Hamiltonian
    es, psis = finite.eigensystem(n_spin_orbitals, hOp, basis, nPsiMax, verbose = verbose)

    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())

    # Print Slater determinants and weights
    if rank == 0 and verbose:
        finite.printSlaterDeterminantsAndWeights(psis = psis, nPrintSlaterWeights = nPrintSlaterWeights)
        finite.printDensityMatrixCubic(nBaths = nBaths, psis = psis, tolPrintOccupation = tolPrintOccupation)

    # Consider from now on only eigenstates with low energy
    es = tuple(e for e in es if e - es[0] < energy_cut)
    psis = tuple(psis[i] for i in range(len(es)))
    if rank == 0:
        print("Consider {:d} eigenstates for the spectra \n".format(len(es)))
    if rank == 0:
        print("Calculate Interacting Green's function...")
    gs = get_Greens_functions(  nBaths = nBaths, 
                                omega_mesh = omega_mesh,
                                es = es, 
                                psis = psis, 
                                l = 2,
                                hOp = hOp,
                                delta = delta,
                                restrictions = restrictions,
                                verbose = verbose)

    gs_thermal_avg = spectra.thermal_average(es[:np.shape(gs)[0]], gs, T=T)

    h0op = get_non_interacting_hamiltonian(h0_filename, nBaths)
    if rank == 0:
        print("Calculate self-energy...")
    sigma = get_sigma(  omega_mesh = omega_mesh, 
                        nBaths = nBaths, 
                        g = gs_thermal_avg, 
                        h0op = h0op,
                        delta = delta)
    try:
        check_sigma(sigma)
    except UnphysicalSelfenergy as err:
        print (f"ERROR Unphysical selfenergy:\n\t{err}")
    if rank == 0:
        save_Greens_function(gs = gs_thermal_avg, omega_mesh = omega_mesh, clustername = clustername)
        save_selfenergy(sigma = sigma, omega_mesh = omega_mesh, clustername = clustername)

def check_sigma(sigma):
    diagonals = [np.diag(sigma[:,:, i]) for i in range(sigma.shape[-1])]
    if np.any(np.imag(diagonals) > 0):
        raise UnphysicalSelfenergy("Diagonal term has positive imaginary part.")
         
#    L = sigma.shape[-1]//2
#    for i in range(L):
#        if(np.any(np.abs(np.imag(np.diag(sigma[:, :, i] - sigma[:, :, L + i]))) > 1e-12)):
#            raise UnphysicalSelenergy("Imaginary part of self energy is not symmetric!")
#        if(np.any(np.abs(np.real(np.diag(sigma[:, :, i] + sigma[:, :, L + i]))) > 1e-12)):
#            raise UnphysicalSelenergy("Real part of self energy is not anti-symmetric!")

def get_non_interacting_hamiltonian(h0_filename, nBaths):
    h0 = get_h0_operator(h0_filename, nBaths)

    h0Op = {}
    for process, value in h0.items():
        h0Op[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value
    return h0Op

def get_h0_v_hb(h0op, nBaths):
    h0Matrix = finite.iOpToMatrix(nBaths, h0op)
    h0 = h0Matrix[6:16,6:16]
    v_dagger = h0Matrix[16:, 6:16]
    v = h0Matrix[6:16, 16:]
    h0_bath = np.diagonal(h0Matrix[16:, 16:])
    return h0, v, v_dagger, h0_bath

def get_sigma(omega_mesh, nBaths, g, h0op, delta):
    h0_d, v, v_dagger, h0_bath = get_h0_v_hb(h0op, nBaths)

    n = h0_d.shape[0]
    N = h0_bath.shape[0]

    g_inv = np.linalg.inv(np.moveaxis(g, -1, 0))
    g_inv = np.moveaxis(g_inv, 0, -1)

    def hyb(w):
        return np.dot(v, np.dot(np.linalg.inv(w*np.eye(N,N) - h0_bath), v_dagger))

    g0_inv = np.array(
             [ (w + 1j*delta)*np.eye(n, n) - h0_d - hyb(w) for w in omega_mesh ] 
             )
    g0_inv = np.moveaxis(g0_inv, 0, -1)

    return g0_inv - g_inv

def get_Greens_functions(nBaths, omega_mesh, es, psis, l, hOp, delta, restrictions, verbose):
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    tOpsPS = spectra.getPhotoEmissionOperators(nBaths, l=2)
    tOpsIPS = spectra.getInversePhotoEmissionOperators(nBaths, l=2)
    gsIPS = calc_Greens_function_with_offdiag(
            n_spin_orbitals,
            hOp, 
            tOpsIPS, 
            psis, 
            es, 
            omega_mesh, 
            delta, 
            restrictions, 
            verbose = verbose)
    gsPS = calc_Greens_function_with_offdiag(
                n_spin_orbitals, 
                hOp, 
                tOpsPS, 
                psis, 
                es, 
                -omega_mesh, 
                delta, 
                restrictions,
                verbose = verbose)
    for i_psi in range(len(psis)):
        for i_w in range(len(omega_mesh)):
            gsPS[i_psi, :, :, i_w] = np.conj(gsPS[i_psi, :, :, i_w].T)
    return gsIPS - gsPS
    
def save_Greens_function(gs, omega_mesh, clustername):
    print ("Writing Greens function to files")
    off_diags = []
    for column in range(len(gs[0,:,0])):
        for row in range(len(gs[:,0,0])):
            if row == column:
                continue
            if np.any(np.abs(gs[row, column, :]) > 1e-12):
                off_diags.append((row, column))
    with open(f"real-G-{clustername}.dat", "w") as fg_real, open(f"imag-G-{clustername}.dat", "w") as fg_imag:
        for i, w in enumerate(omega_mesh):
            fg_real.write(f"{w*eV_to_Ry} {np.real(np.sum(np.diag(gs[:, :, i])))} " + " ".join(f"{np.real(el)}" for el in np.diag(gs[:, :, i])) + " ".join(f"{np.real(gs[row, column, i])}" for row, column in off_diags) + "\n")
            fg_imag.write(f"{w*eV_to_Ry} {np.imag(np.sum(np.diag(gs[:, :, i])))} " + " ".join(f"{np.imag(el)}" for el in np.diag(gs[:, :, i])) + " ".join(f"{np.imag(gs[row, column, i])}" for row, column in off_diags) + "\n")

def save_selfenergy(sigma, omega_mesh, clustername):
    print ("Writing Selfenergy to files")
    off_diags = []
    for column in range(len(sigma[0,:,0])):
        for row in range(len(sigma[:,0,0])):
            if row == column:
                continue
            if np.any(np.abs(sigma[row, column, :]) > 1e-12):
                off_diags.append((row, column))
    with open(f"real-realaxis-Sigma-{clustername}.dat", "w") as fs_real, open(f"imag-realaxis-Sigma-{clustername}.dat", "w") as fs_imag:
        for i, w in enumerate(omega_mesh):
            fs_real.write(f"{w*eV_to_Ry} {np.real(np.sum(np.diag(sigma[:, :, i])))} " + " ".join(f"{np.real(el)}" for el in np.diag(sigma[:, :, i])) + " ".join(f"{np.real(sigma[row, column, i])}" for row, column in off_diags) + "\n")
            fs_imag.write(f"{w*eV_to_Ry} {np.imag(np.sum(np.diag(sigma[:, :, i])))} " + " ".join(f"{np.imag(el)}" for el in np.diag(sigma[:, :, i])) + " ".join(f"{np.imag(sigma[row, column, i])}" for row, column in off_diags) + "\n")


def calc_Greens_function_with_offdiag(
        n_spin_orbitals,
        hOp,
        tOps,
        psis,
        es,
        w,
        delta,
        restrictions=None,
        krylovSize=150,
        slaterWeightMin=1e-7,
        parallelization_mode="H_build",
        verbose = True):
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
    # Green's functions
    gs = np.zeros((n,len(tOps), len(tOps), len(w)),dtype=complex)

    # Hamiltonian dict of the form  |PS> : {H|PS>}
    # New elements are added each time getGreen is called.
    # Also acts as an input to getGreen and speed things up dramatically.
    h = {}

    for i, (psi, e) in enumerate(zip(psis, es)):
        v = []
        for tOp in tOps:
            v.append(applyOp(n_spin_orbitals, tOp, psi, slaterWeightMin, restrictions, {}))

        gs[i, :, :, :] = get_block_Green(
                                n_spin_orbitals = n_spin_orbitals,
                                hOp = hOp,
                                psi_arr = v,
                                e = e,
                                w = w,
                                delta = delta,
                                restrictions = restrictions,
                                krylovSize = krylovSize,
                                slaterWeightMin = slaterWeightMin,
                                parallelization_mode = parallelization_mode,
                                verbose = verbose
                                )
    return gs


    if parallelization_mode == "eigen_states":
        g = {}
        # Loop over eigen states, unique for each MPI rank
        for i in get_job_tasks(rank, ranks, range(n)):
            psi =  psis[i]
            e = es[i]
            # Initialize Green's functions
            g[i] = np.zeros((len(tOps), len(tOps),len(w)), dtype=complex)

            # Calculate Diagonal elements!
            for t_right, tOp_right in enumerate(tOps):
                psiR = applyOp(n_spin_orbitals, tO_right, psi, slaterWeightMin,
                        restrictions)
                normalization = sqrt(norm2(psiR))
                for state in psiR.keys():
                    psiR[state] /= normalization
                g_RR = spectra.getGreen(
                                n_spin_orbitals, e, psiR, hOp, w, delta, krylovSize,
                                slaterWeightMin, restrictions, h,
                                parallelization_mode="serial", verbose = verbose)
                g[i][t_right, t_right, :] = normalization**2*g_RR

            #for t_right, tOp_right in enumerate(tOps):
            #    for t_left in range(t_right + 1, len(tÓps)):
            #        tOp_left = tOps[t_left]
            #        opSum = finite.addOps([tOp_right, tOp_left])
            #        opDif = finite.addOps([tOp_right, {key: -val for key, val in tOp_left.items()}])
            #        psiSum = applyOp(n_spin_orbitals, opSum, psi, slaterWeightMin,
            #                restrictions)
            #        psiDif = applyOp(n_spin_orbitals, opDif, psi, slaterWeightMin,
            #                restrictions)

            #        sumNorm = sqrt(norm2(psiSum))
            #        difNorm = sqrt(norm2(psiDif))
            #        for state in psiSum.keys():
            #            psiSum[state] /= sumNorm
            #        for state in psiDif.keys():
            #            psiDif[state] /= difNorm
            #        g_LR =  0.25*(sumNorm**2*spectra.getGreen(
            #                        n_spin_orbitals, e, psiSum, hOp, w, delta, krylovSize,
            #                        slaterWeightMin, restrictions, h,
            #                        parallelization_mode="serial", verbose = verbose) 
            #                    - difNorm**2*spectra.getGreen(
            #                        n_spin_orbitals, e, psiDif, hOp, w, delta, krylovSize,
            #                        slaterWeightMin, restrictions, h,
            #                        parallelization_mode="serial", verbose = verbose)
            #                   )
            #        g[i][t_left, t_right, :] = g_LR
            #        g[i][t_right, t_left, :] = g_LR
        # Distribute the Green's functions among the ranks
        for r in range(ranks):
            gTmp = comm.bcast(g, root=r)
            for i,gValue in gTmp.items():
                gs[i,:,:,:] = gValue
    elif parallelization_mode == "H_build":
        for i in range(n):
            psi =  psis[i]
            e = es[i]
            # Loop over transition operators
            for t_right, tOp_right in enumerate(tOps):
                t_big = {}
                psiR = applyOp(n_spin_orbitals, tOp_right, psi, slaterWeightMin, restrictions, t_big)
                # if rank == 0: print("len(t_big) = {:d}".format(len(t_big)))
                normalization = sqrt(norm2(psiR))
                for state in psiR.keys():
                    psiR[state] /= normalization
                gs[i, t_right, t_right, :] = normalization**2*spectra.getGreen(
                    n_spin_orbitals, e, psiR, hOp, w, delta, krylovSize,
                    slaterWeightMin, restrictions, h,
                    parallelization_mode=parallelization_mode, verbose = verbose)
            # for t_right, tOp_right in enumerate(tOps):
            #     for t_left in range(t_right + 1, len(tOps)):
            #         tOp_left = tOps[t_left]
            #         t_big = {}
            #         opSum = finite.addOps([tOp_right, tOp_left])
            #         opDif = finite.addOps([tOp_right, {key: -val for key, val in tOp_left.items()}])
            #         psiSum = applyOp(n_spin_orbitals, opSum, psi, slaterWeightMin,
            #                 restrictions)
            #         psiDif = applyOp(n_spin_orbitals, opDif, psi, slaterWeightMin,
            #                 restrictions)

            #         sumNorm = sqrt(norm2(psiSum))
            #         for state in psiSum.keys():
            #             psiSum[state] /= sumNorm
            #         difNorm = sqrt(norm2(psiDif))
            #         for state in psiDif.keys():
            #             psiDif[state] /= difNorm

            #         g_LR =  0.25*(sumNorm**2*spectra.getGreen(
            #                         n_spin_orbitals, e, psiSum, hOp, w, delta, krylovSize,
            #                         slaterWeightMin, restrictions, h,
            #                         parallelization_mode="serial", verbose = verbose) 
            #                     - difNorm**2*spectra.getGreen(
            #                         n_spin_orbitals, e, psiDif, hOp, w, delta, krylovSize,
            #                         slaterWeightMin, restrictions, h,
            #                         parallelization_mode="serial", verbose = verbose)
            #                    )

            #         gs[i, t_left, t_right, :] = g_LR
            #         gs[i, t_right, t_left, :] = g_LR
    else:
        raise Exception("Incorrect value of variable parallelization_mode.")
    return gs

def get_block_Green(
        n_spin_orbitals,
        hOp,
        psi_arr,
        e,
        w,
        delta,
        restrictions=None,
        krylovSize=150,
        slaterWeightMin=1e-7,
        parallelization_mode="H_build",
        verbose = True):

    states = set([key for psi in psi_arr for key in psi.keys()])

    h, basis_index = finite.expand_basis_and_hamiltonian(
        n_spin_orbitals, {}, hOp, list(states), restrictions,
        parallelization_mode = 'serial', return_h_local = False, verbose = verbose)

    N = len(basis_index)
    n = len(psi_arr)

    gs = np.zeros((len(w), n, n), dtype = complex)


    psi_start = np.zeros((N,n), dtype= complex)
    for i, psi in enumerate(psi_arr):
        for ps, amp in psi.items():
            psi_start[basis_index[ps], i] = amp
    krylovSize = min(krylovSize,N)

    # Do a QR decomposition of the target block, to ensure that we start with an orthonormal block
    psi0, r = np.linalg.qr(psi_start)

    if rank == 0:
        print (f"Starting block Lanczos!")
    # Run Lanczos on Q^T* [wI - j*delta - H]^-1 Q
    alphas, betas = get_block_Lanczons_matrices(psi0, h, n_spin_orbitals, slaterWeightMin, restrictions, krylovSize)

    omegaP = w + 1j*delta + e
    for i in range(krylovSize - 1, -1, -1):
        if i == krylovSize - 1:
            gs = np.linalg.inv([wP*np.eye(n, n, dtype = complex) - alphas[:, :, i] for wP in omegaP])
        else:
            gs = np.linalg.inv([wP*np.eye(n, n, dtype = complex) - alphas[:, :, i] - np.linalg.multi_dot([np.conj(betas[:, :, i].T), gs[i_w], betas[:, :, i]]) for i_w, wP in enumerate(omegaP)])
    # Multiply obtained Green's function with the upper triangular matrix to restore the original block
    # R^T* G R
    gs = [np.linalg.multi_dot([np.conj(r.T), gs[i_w, :, :], r]) for i_w in range(len(w))]
    return np.moveaxis(gs, 0, -1)

def get_block_Lanczons_matrices(psi0, h, n_spin_orbitals, slaterWeightMin, restrictions, KrylovSize):
    h_dict = {}
    h_local = False
    verbose = True

    n = psi0.shape[1]
    N = psi0.shape[0]

    KrylovSize = min(KrylovSize, N)
    alphas = np.zeros((n, n, KrylovSize), dtype = complex)
    betas = np.zeros((n, n, KrylovSize), dtype = complex)
   
    q = np.zeros((2, N, n), dtype = complex) 
    q[1, :, :] = psi0

    for i in range(KrylovSize):
        wp = h.dot(q[1])
        alphas[:, :, i] = np.dot(np.conj(q[1].T), wp)
        w = wp - np.dot(q[1], alphas[:,:,i]) - np.dot(q[0], np.conj(betas[:,:,i-1].T))
        q[0] = q[1]
        q[1], betas[:, :, i] = np.linalg.qr(w)
        if np.any(np.abs(np.diag(betas[:, :, i])) < 1e-10):
            break

    return alphas, betas

if __name__== "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser(description="Spectroscopy simulations")
    parser.add_argument(
        "--clustername",
        type=str,
        default="cluster",
        help="Id of cluster, used for generating the filename in which to store the calculated self-energy.",
    )
    parser.add_argument(
        "h0_filename",
        type=str,
        help="Filename of non-interacting Hamiltonian, in pickle-format.",
    )
    parser.add_argument(
        "--ls",
        type=int,
        default=2,
        help="Angular momenta of correlated orbitals.",
    )
    parser.add_argument(
        "--nBaths",
        type=int,
        default=10,
        help="Number of bath states, for each angular momentum.",
    )
    parser.add_argument(
        "--nValBaths",
        type=int,
        default=10,
        help="Number of valence bath states, for each angular momentum.",
    )
    parser.add_argument(
        "--n0imps",
        type=int,
        default=8,
        help="Initial impurity occupation, for each angular momentum.",
    )
    parser.add_argument(
        "--dnTols",
        type=int,
        default=2,
        help=("Max devation from initial impurity occupation, " "for each angular momentum."),
    )
    parser.add_argument(
        "--dnValBaths",
        type=int,
        default=2,
        help=("Max number of electrons to leave valence bath orbitals, " "for each angular momentum."),
    )
    parser.add_argument(
        "--dnConBaths",
        type=int,
        default=0,
        help=("Max number of electrons to enter conduction bath orbitals, " "for each angular momentum."),
    )
    parser.add_argument(
        "--Fdd",
        type=float,
        nargs="+",
        default=[7.5, 0, 9.9, 0, 6.6],
        help="Slater-Condon parameters Fdd. d-orbitals are assumed.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=0.096,
        help="SOC value for valence orbitals. Assumed to be d-orbitals",
    )
    parser.add_argument(
        "--chargeTransferCorrection",
        type=float,
        default=1.5,
        help="Double counting parameter.",
    )
    parser.add_argument(
        "--hField",
        type=float,
        nargs="+",
        default=[0, 0, 0.0001],
        help="Magnetic field. (h_x, h_y, h_z)",
    )
    parser.add_argument(
        "--nPsiMax",
        type=int,
        default=5,
        help="Maximum number of eigenstates to consider.",
    )
    parser.add_argument(
            "--nPrintSlaterWeights", 
            type=int, 
            default=3, 
            help="Printing parameter."
            )
    parser.add_argument(
            "--tolPrintOccupation", 
            type=float, 
            default=0.5, 
            help="Printing parameter."
            )
    parser.add_argument(
            "--T", 
            type=float, 
            default=300, 
            help="Temperature (Kelvin)."
            )
    parser.add_argument(
        "--energy_cut",
        type=float,
        default=10,
        help="How many k_B*T above lowest eigenenergy to consider.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help=("Smearing, half width half maximum (HWHM). " "Due to short core-hole lifetime."),
    )
    parser.add_argument(
        "-v", "--verbose",
        action='store_true',
        help=("Set verbose output (very loud...)"),
    )
    args = parser.parse_args()

    # Sanity checks
    assert args.nBaths >= args.nValBaths
    assert args.n0imps <= 2 * (2 * args.ls + 1)  # Full occupation
    assert args.n0imps >= 0
    assert len(args.Fdd) == 5
    assert len(args.hField) == 3

    get_selfenergy(
            clustername=args.clustername,
            h0_filename=args.h0_filename,
            ls=(args.ls), 
            nBaths=(args.nBaths),
            nValBaths=(args.nValBaths), 
            n0imps=(args.n0imps),
            dnTols=(args.dnTols), 
            dnValBaths=(args.dnValBaths),
            dnConBaths=(args.dnConBaths),
            Fdd=(args.Fdd), 
            xi=args.xi,
            chargeTransferCorrection=args.chargeTransferCorrection,
            hField=tuple(args.hField), 
            nPsiMax=args.nPsiMax,
            nPrintSlaterWeights=args.nPrintSlaterWeights,
            tolPrintOccupation=args.tolPrintOccupation,
            T=args.T, 
            energy_cut=args.energy_cut,
            delta=args.delta,
            verbose = args.verbose
            )
