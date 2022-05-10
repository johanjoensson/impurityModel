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
from impurityModel.ed.average import k_B, thermal_average


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

    omega_mesh = np.linspace(-25, 25, 2000)
    # omega_mesh = 1j*np.pi*k_B*T*np.arange(start = 1, step = 2, stop = 3001)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        t0 = time.time()
    # -- System information --

    nBaths = OrderedDict({ls: nBaths})
    nValBaths = OrderedDict({ls: nValBaths})
    dnValBaths = OrderedDict({ls: dnValBaths})
    dnConBaths = OrderedDict({ls: dnConBaths})

    # -- Basis occupation information --
    n0imps = OrderedDict({ls: n0imps})
    dnTols = OrderedDict({ls: dnTols})

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
        [Fdd, None, None, None],
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
    es, psis = finite.eigensystem(n_spin_orbitals, hOp, basis, nPsiMax, verbose = verbose, groundDiagMode = 'full')

    finite.printThermalExpValues(nBaths, es, psis)
    finite.printExpValues(nBaths, es, psis)
    # Print Slater determinants and weights
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
                                l = ls,
                                hOp = hOp,
                                delta = delta,
                                restrictions = restrictions,
                                verbose = verbose,
                                mpi_distribute = False)
    # gs[np.abs(gs) < 1e-3] = 0

    # gs_thermal_avg = thermal_average(es, gs, T=T)
    gs_thermal_avg = thermal_average(es[:np.shape(gs)[0]], gs, T=T)

    h0op = get_non_interacting_hamiltonian(h0_filename, nBaths)
    if rank == 0:
        print("Calculate self-energy...")
        sigma = get_sigma(  omega_mesh = omega_mesh, 
                            nBaths = nBaths, 
                            g = gs_thermal_avg, 
                            h0op = h0op,
                            delta = delta,
                            save_G0 = True,
                            save_hyb = True, 
                            clustername = clustername)
        try:
            check_sigma(sigma)
        except UnphysicalSelfenergy as err:
            print (f"ERROR Unphysical selfenergy:\n\t{err}")
    if rank == 0:
        save_Greens_function(gs = gs_thermal_avg, omega_mesh = omega_mesh, label ='G-'+ clustername)
        save_Greens_function(gs = sigma, omega_mesh = omega_mesh, label ='Sigma-'+ clustername, tol = 1e-2)

def check_sigma(sigma):
    diagonals = [np.diag(sigma[:,:, i]) for i in range(sigma.shape[-1])]
    if np.any(np.imag(diagonals) > 0):
        raise UnphysicalSelfenergy("Diagonal term has positive imaginary part.")

def get_non_interacting_hamiltonian(h0_filename, nBaths):
    h0 = get_h0_operator(h0_filename, nBaths)

    h0Op = {}
    for process, value in h0.items():
        h0Op[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value
    return h0Op

def get_hcorr_v_hbath(h0op, nBaths):
    #   The matrix form of h0op can be written
    #   [  hcorr  V^+    ]
    #   [  V      hbath  ]
    # TODO: 
    # Remove assumption that hp can be ignored, 
    # allow for other l quantum numbers of correlated orbitals than 2
    # allow for hcorr to contain a mixture of l quantum numbers
    h0Matrix = finite.iOpToMatrix(nBaths, h0op)
    hcorr = h0Matrix[0:10,0:10]
    v_dagger = h0Matrix[0:10, 10:]
    v = h0Matrix[10:, 0:10]
    h_bath = h0Matrix[10:, 10:]
    return hcorr, v, v_dagger, h_bath

def get_sigma(omega_mesh, nBaths, g, h0op, delta, mpi_distribute = False, save_G0 = False, save_hyb = False, clustername = ""):
    hcorr, v, v_dagger, hbath = get_hcorr_v_hbath(h0op, nBaths)

    def matrix_print(matrix, label: str = None):
        ms = "\n".join([ "  ".join([f"{val: .3f}" for val in row]) for row in matrix])
        if label:
            print (label)
        print (ms)

    matrix_print (hcorr, r'H$_d$')
    matrix_print (v, 'V')
    matrix_print (hbath, r'H_b')
    n = hcorr.shape[0]
    N = hbath.shape[0]

    g_inv = np.linalg.inv(np.moveaxis(g, -1, 0))
    g_inv = np.moveaxis(g_inv, 0, -1)


    def hyb(w):
        return np.linalg.multi_dot([v_dagger, np.linalg.inv((w + 1j*delta)*np.identity(N) - hbath), v])
    if save_hyb:
        hybridization_function = np.array([hyb(w) for w in omega_mesh])
        save_Greens_function(np.moveaxis(hybridization_function, 0, -1), omega_mesh, label = "Hyb-" + clustername)

    g0_inv = np.array(
             [(w + 1j*delta)*np.identity(n) - hcorr - hybridization_function[i_w] for i_w, w in enumerate(omega_mesh)]
             )

    if save_G0:
        save_Greens_function(np.moveaxis(np.linalg.inv(g0_inv), 0, -1), omega_mesh, "G0-" + clustername)

    g0_inv = np.moveaxis(g0_inv, 0, -1)

    return g0_inv - g_inv

def get_Greens_functions(nBaths, omega_mesh, es, psis, l, hOp, delta, restrictions, verbose, mpi_distribute = False):
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    tOpsPS = spectra.getPhotoEmissionOperators(nBaths, l=l)
    tOpsIPS = spectra.getInversePhotoEmissionOperators(nBaths, l=l)
    gsIPS = calc_Greens_function_with_offdiag(
                                    n_spin_orbitals,
                                    hOp, 
                                    tOpsIPS, 
                                    psis, 
                                    es, 
                                    omega_mesh, 
                                    delta, 
                                    restrictions, 
                                    krylovSize = 150,
                                    verbose = verbose)
    gsPS  = np.transpose(calc_Greens_function_with_offdiag(
                                    n_spin_orbitals, 
                                    hOp, 
                                    tOpsPS, 
                                    psis, 
                                    es, 
                                    -omega_mesh, 
                                    -delta, 
                                    restrictions,
                                    krylovSize = 150,
                                    verbose = verbose)
            , (0, 2, 1, 3))
    if mpi_distribute:
        comm.Bcast(gsIPS, root = 0)
        comm.Bcast(gsPS, root = 0)
    return gsIPS - gsPS
    
def save_Greens_function(gs, omega_mesh, label, tol = 1e-12):
    axis_label = 'realaxis'
    if np.all(np.abs(np.imag(omega_mesh)) > 1e-12):
        axis_label = 'Matsubara'

    nl = int(gs.shape[1]/2)
    off_diags = []
    for row in range(gs.shape[0]):
        for column in range(gs.shape[1]):
            if row == column:
                continue
            if np.any(np.abs(gs[row, column, :]) > tol):
                off_diags.append((row, column))

    print (f"Writing {axis_label} {label} to files")
    with open(f"real-{axis_label}-{label}.dat", "w") as fg_real, open(f"imag-{axis_label}-{label}.dat", "w") as fg_imag:
        header = '# 1 - Omega(Ry)  2 - Trace  3 - Spin down  4 - Spin up\n'
        header +=  '# Individual matrix elements given in the matrix below:'
        for row in range(gs.shape[0]):
            header += '\n# '
            for column in range(gs.shape[1]):
                if row == column:
                    header += f'{5 + row:< 4d}'
                elif (row, column) in off_diags:
                    header += f'{5 + 2*nl + off_diags.index((row, column)):< 4d}'
                else:
                    header += ' '*4
        fg_real.write(header + '\n')
        fg_imag.write(header + '\n')
        for i, w in enumerate(omega_mesh):
            fg_real.write(f"{w*eV_to_Ry} {np.real(np.sum(np.diag(gs[:, :, i])))} {np.real(np.sum(np.diag(gs[0:nl, 0:nl, i])))} {np.real(np.sum(np.diag(gs[nl:, nl:, i])))} " + " ".join(f"{np.real(el)}" for el in np.diag(gs[:, :, i])) + " " + " ".join(f"{np.real(gs[row, column, i])}" for row, column in off_diags) + "\n")
            fg_imag.write(f"{w*eV_to_Ry} {np.imag(np.sum(np.diag(gs[:, :, i])))} {np.imag(np.sum(np.diag(gs[0:nl, 0:nl, i])))} {np.imag(np.sum(np.diag(gs[nl:, nl:, i])))} " + " ".join(f"{np.imag(el)}" for el in np.diag(gs[:, :, i])) + " " + " ".join(f"{np.imag(gs[row, column, i])}" for row, column in off_diags) + "\n")

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

    h_mem = {}
    if parallelization_mode == "eigen_states":
        # Green's functions
        gs = np.zeros((n,len(tOps), len(tOps), len(w)),dtype=complex)
        for i in finite.get_job_tasks(rank, ranks, range(len(psis))):
            psi = psis[i]
            e = es[i]

            v = []
            for tOp in tOps:
                v.append(applyOp(n_spin_orbitals, tOp, psi, slaterWeightMin, restrictions, {}))

            gs_i[:, :, :] = get_block_Green(
                                    n_spin_orbitals = n_spin_orbitals,
                                    hOp = hOp,
                                    psi_arr = v,
                                    e = e,
                                    w = w,
                                    delta = delta,
                                    restrictions = restrictions,
                                    h_mem = h_mem,
                                    krylovSize = krylovSize,
                                    slaterWeightMin = slaterWeightMin,
                                    parallelization_mode = 'serial',
                                    verbose = verbose
                                    )
            comm.Reduce(gs_i, gs[i])
    elif parallelization_mode == "H_build":
        gs = np.zeros((n,len(tOps), len(tOps), len(w)), dtype=complex)
        t_mems = [{} for _ in tOps]
        for i, (psi, e) in enumerate(zip(psis, es)):
            v = []
            for i_tOp, tOp in enumerate(tOps):
                v.append(applyOp(n_spin_orbitals, tOp, psi, slaterWeightMin, 
                    #restrictions, 
                    t_mems[i_tOp]))
            gs[i, :, :, :] = get_block_Green(
                                    n_spin_orbitals = n_spin_orbitals,
                                    hOp = hOp,
                                    psi_arr = v,
                                    e = e,
                                    w = w,
                                    delta = delta,
                                    restrictions = restrictions,
                                    h_mem = h_mem,
                                    krylovSize = krylovSize,
                                    slaterWeightMin = slaterWeightMin,
                                    parallelization_mode = parallelization_mode,
                                    verbose = verbose
                                    )
    return gs

def get_block_Green(
        n_spin_orbitals,
        hOp,
        psi_arr,
        e,
        w,
        delta,
        restrictions=None,
        h_mem = None,
        mode = 'sparse',
        krylovSize=150,
        slaterWeightMin=1e-7,
        parallelization_mode="H_build",
        verbose = True):

    if h_mem == None:
        h_mem = {}
        
    parallelization_mode = 'serial'
    h_local = False
    
    states = set(key for psi in psi_arr for key in psi.keys())
    h, basis_index = finite.expand_basis_and_hamiltonian(
        n_spin_orbitals, h_mem, hOp, list(states), restrictions,
        parallelization_mode = parallelization_mode, return_h_local = h_local, verbose = True)

    N = len(basis_index)
    n = len(psi_arr)


    psi_start = np.zeros((N,n), dtype= complex)
    for i, psi in enumerate(psi_arr):
        for ps, amp in psi.items():
            psi_start[basis_index[ps], i] = amp
    krylovSize = min(krylovSize,N)

    mask = np.linalg.norm(psi_start, axis = 0) > 1e-12
    if rank == 0:
        print (mask)
        print (psi_start[:,mask])
    psi0, r = np.linalg.qr(psi_start[:, mask])

    if rank == 0:
        print (f"Starting block Lanczos!")
    # Run Lanczos on psi0^T* [wI - j*delta - H]^-1 psi0
    alphas, betas = get_block_Lanczons_matrices(
                                            psi0, 
                                            h, 
                                            n_spin_orbitals, 
                                            slaterWeightMin, 
                                            restrictions, 
                                            krylovSize, 
                                            h_local,
                                            mode,
                                            verbose
                                            )

    gs_local = np.zeros((len(w), n, n), dtype = complex)
    # Parallelize over omega mesh
    omegaP = w + 1j*delta + e
    # omegaP = w
    mask_slice = np.ix_(mask, mask)
    for i_wP, wP in finite.get_job_tasks(rank, ranks, list(enumerate(omegaP))):
    #for i_wP, wP in enumerate(omegaP):
        for i in range(krylovSize - 1, -1, -1):
            if i == krylovSize - 1:
                gs_local[i_wP][mask_slice] = sp.linalg.inv(wP*np.identity(alphas.shape[0], dtype = complex) - alphas[:, :, i])
            else:
                gs_local[i_wP][mask_slice] = sp.linalg.inv(wP*np.identity(alphas.shape[0], dtype = complex) - alphas[:, :, i] - np.linalg.multi_dot([np.conj(betas[:, :, i].T), gs_local[i_wP][mask_slice], betas[:, :, i]]))
        # Multiply obtained Green's function with the upper triangular matrix to restore the original block
        # R^T* G R, only on rank 0
        gs_local[i_wP][mask_slice] = np.linalg.multi_dot([np.conj(r).T, gs_local[i_wP][mask_slice], r])
    # Reduce Green's function to rank 0
    gs = np.zeros_like(gs_local, dtype = complex)
    comm.Reduce(gs_local, gs, op = MPI.SUM, root = 0)
    # gs = np.zeros_like(gs_local)
    # gs = gs_local
    return np.moveaxis(gs, 0, -1)

def get_block_Lanczons_matrices(
        psi0, 
        h, 
        n_spin_orbitals, 
        slaterWeightMin, 
        restrictions, 
        KrylovSize, 
        h_local = False,
        mode = 'sparse',
        verbose = True):

    if mode == 'dense':
        h = h.to_array()

    N = psi0.shape[0]
    n = psi0.shape[1]

    KrylovSize = min(KrylovSize, N)
    alphas = np.zeros((n, n, KrylovSize), dtype = complex)
    betas = np.zeros((n, n, KrylovSize), dtype = complex)
   
    q = np.zeros((2, N, n), dtype = complex) 
    q[1] = psi0

    if h_local:
        for i in range(KrylovSize):
            wp_local = h.dot(q[1])
            if rank == 0:
                wp = np.zeros_like(wp_local)
            else:
                wp = None
            comm.Reduce(wp_local, wp, root = 0)

            if rank == 0:
                alphas[:, :, i] = np.dot(np.conj(q[1].T), wp)
                w = wp - np.dot(q[1], alphas[:,:,i]) - np.dot(q[0], np.conj(betas[:,:,i-1].T))
                q[0] = q[1]
                q[1], betas[:, :, i] = np.linalg.qr(w)
            comm.Bcast(q[1], root = 0)
        # Distribute Lanczos matrices to all ranks
        comm.Bcast(alphas, root = 0)
        comm.Bcast(betas, root = 0)
    else:
        for i in range(KrylovSize):
            wp = h.dot(q[1])
            alphas[:, :, i] = np.dot(np.conj(q[1].T), wp)
            w = wp - np.dot(q[1], alphas[:,:,i]) - np.dot(q[0], np.conj(betas[:,:,i-1].T))
            q[0] = q[1]
            q[1], betas[:, :, i] = np.linalg.qr(w)

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
        help="Total number of bath states, for the correlated orbitals.",
    )
    parser.add_argument(
        "--nValBaths",
        type=int,
        default=10,
        help="Number of valence bath states for the correlated orbitals.",
    )
    parser.add_argument(
        "--n0imps",
        type=int,
        default=8,
        help="Nominal impurity occupation.",
    )
    parser.add_argument(
        "--dnTols",
        type=int,
        default=2,
        help=("Max devation from nominal impurity occupation."),
    )
    parser.add_argument(
        "--dnValBaths",
        type=int,
        default=2,
        help=("Max number of electrons to leave valence bath orbitals."),
    )
    parser.add_argument(
        "--dnConBaths",
        type=int,
        default=0,
        help=("Max number of electrons to enter conduction bath orbitals."),
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
        default=None,
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
    assert args.n0imps >= 0
    assert args.n0imps <= 2 * (2 * args.ls + 1)
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
