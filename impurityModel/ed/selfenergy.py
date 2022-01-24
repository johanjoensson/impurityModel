from math import sqrt
import numpy as np
from collections import OrderedDict
from mpi4py import MPI
import pickle
import time
import argparse
import h5py

from impurityModel.ed.get_spectra import get_hamiltonian_operator, get_h0_operator, get_restrictions
from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.finite import c2i
from impurityModel.ed.finite import daggerOp, applyOp, inner, add, norm2
from impurityModel.ed.average import k_B

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size

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
        xi_3d, 
        chargeTransferCorrection,
        hField, 
        nPsiMax,
        nPrintSlaterWeights, 
        tolPrintOccupation,
        T,
        energy_cut,
        delta):

    omega_mesh = np.linspace(-25, 25, 4000)

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        t0 = time.time()
    # -- System information --
    nBaths = OrderedDict(zip(ls, nBaths))
    nValBaths = OrderedDict(zip(ls, nValBaths))

    # -- Basis occupation information --
    n0imps = OrderedDict(zip(ls, n0imps))
    dnTols = OrderedDict(zip(ls, dnTols))
    dnValBaths = OrderedDict(zip(ls, dnValBaths))
    dnConBaths = OrderedDict(zip(ls, dnConBaths))

    # -- Spectra information --
    # Energy cut in eV.
    energy_cut *= k_B * T

    # -- Occupation restrictions for excited states --
    restrictions = get_restrictions(    l = 2, 
                                        n0imps = n0imps, 
                                        nBaths = nBaths, 
                                        nValBaths = nValBaths, 
                                        dnTols = dnTols, 
                                        dnValBaths = dnValBaths, 
                                        dnConBaths = dnConBaths)

    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    if rank == 0:
        print("#spin-orbitals:", n_spin_orbitals)

    # Hamiltonian
    if rank == 0:
        print("Construct the Hamiltonian operator...")
    hOp = get_hamiltonian_operator(
        nBaths,
        nValBaths,
        [Fdd, [0 for _ in range(3)], [0 for _ in range(3)], [0 for _ in range(4)]],
        [0, xi_3d],
        [n0imps, chargeTransferCorrection],
        hField,
        h0_filename,
        rank
    )
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0:
        print("{:d} processes in the Hamiltonian.".format(len(hOp)))
    # Many body basis for the ground state
    if rank == 0:
        print("Create basis...")
    basis = finite.get_basis(nBaths, nValBaths, dnValBaths, dnConBaths, dnTols, n0imps)
    if rank == 0:
        print("#basis states = {:d}".format(len(basis)))
    # Diagonalization of restricted active space Hamiltonian
    es, psis = finite.eigensystem(n_spin_orbitals, hOp, basis, nPsiMax)

    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())

    # Print Slater determinants and weights
    if rank == 0:
        finite.printSlaterDeterminantsAndWeights(psis = psis, nPrintSlaterWeights = nPrintSlaterWeights)
        finite.printDensityMatrixCubic(nBaths = nBaths, psis = psis, tolPrintOccupation = tolPrintOccupation)

    # Consider from now on only eigenstates with low energy
    es = tuple(e for e in es if e - es[0] < energy_cut)
    psis = tuple(psis[i] for i in range(len(es)))
    if rank == 0:
        print("Consider {:d} eigenstates for the spectra \n".format(len(es)))
    gs = get_Greens_functions(  nBaths = nBaths, 
                                omega_mesh = omega_mesh,
                                es = es, 
                                psis = psis, 
                                l = 2,
                                hOp = hOp,
                                delta = delta,
                                restrictions = restrictions)

    gs_thermal_avg = spectra.thermal_average(es[:np.shape(gs)[0]], gs, T=T)

    h0op = get_non_interacting_hamiltonian(h0_filename, nBaths)
    sigma = get_sigma(  omega_mesh = omega_mesh, 
                        nBaths = nBaths, 
                        g = gs_thermal_avg, 
                        h0op = h0op,
                        delta = delta)

    if rank == 0:
        save_Greens_function(gs = gs_thermal_avg, omega_mesh = omega_mesh, clustername = clustername)
        save_selfenergy(sigma = sigma, omega_mesh = omega_mesh, clustername = clustername)

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

    g_inv = np.zeros((h0_d.shape[0], h0_d.shape[1], len(omega_mesh)), dtype = complex)
    g0_inv = np.zeros((h0_d.shape[0], h0_d.shape[1], len(omega_mesh)), dtype = complex)

    n = h0_d.shape[0]
    N = h0_bath.shape[0]
    for i, w in enumerate(omega_mesh):
        g_inv[:,:, i] = np.linalg.inv(g[:,:,i])
        g0_inv[:,:,i] = (w + 1j*delta)*np.eye(n, n) - h0_d - np.dot(v, np.dot(np.linalg.inv(w*np.eye(N,N) - h0_bath), v_dagger))

    return g0_inv - g_inv

def get_Greens_functions(nBaths, omega_mesh, es, psis, l, hOp, delta, restrictions):
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    gs = np.zeros((len(es), len(omega_mesh)), dtype=np.complex)
    tOpsPS = spectra.getPhotoEmissionOperators(nBaths, l=2)
    tOpsIPS = spectra.getInversePhotoEmissionOperators(nBaths, l=2)
    gs = calc_Greens_function_with_offdiag(
            n_spin_orbitals,
            hOp, 
            tOpsIPS, 
            psis, 
            es, 
            omega_mesh, 
            delta, 
            restrictions)
    gs -= calc_Greens_function_with_offdiag(
            n_spin_orbitals, 
            hOp, 
            tOpsPS, 
            psis, 
            es, 
            -omega_mesh, 
            -delta, 
            restrictions)
    return gs
    
def save_Greens_function(gs, omega_mesh, clustername):
    print ("Writing Greens function to files")
    with open(f"real-G-{clustername}.dat", "w") as fg_real, open(f"imag-G-{clustername}.dat", "w") as fg_imag:
        for i, w in enumerate(omega_mesh):
            fg_real.write(f"{w} {np.real(np.sum(gs[:, :, i]))} " + " ".join(f"{np.real(el)}" for el in np.diag(gs[:, :, i])) + "\n")
            fg_imag.write(f"{w} {np.imag(np.sum(gs[:, :, i]))} " + " ".join(f"{np.imag(el)}" for el in np.diag(gs[:, :, i])) + "\n")

def save_selfenergy(sigma, omega_mesh, clustername):
    print ("Writing Selfenergy to files")
    with open(f"real-realaxis-Sigma-{clustername}.dat", "w") as fs_real, open(f"imag-realaxis-Sigma-{clustername}.dat", "w") as fs_imag:
        for i, w in enumerate(omega_mesh):
            fs_real.write(f"{w} {np.real(np.sum(sigma[:, :, i]))} " + " ".join(f"{np.real(el)}" for el in np.diag(sigma[:, :, i])) + "\n")
            fs_imag.write(f"{w} {np.imag(np.sum(sigma[:, :, i]))} " + " ".join(f"{np.imag(el)}" for el in np.diag(sigma[:, :, i])) + "\n")


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
        parallelization_mode="H_build"):
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
    gs = np.zeros((n,len(tOps), len(tOps),len(w)),dtype=np.complex)
    # Hamiltonian dict of the form  |PS> : {H|PS>}
    # New elements are added each time getGreen is called.
    # Also acts as an input to getGreen and speed things up dramatically.
    h = {}
    if parallelization_mode == "eigen_states":
        g = {}
        # Loop over eigen states, unique for each MPI rank
        for i in get_job_tasks(rank, ranks, range(n)):
            psi =  psis[i]
            e = es[i]
            # Initialize Green's functions
            g[i] = np.zeros((len(tOps),len(w)), dtype=np.complex)
            # Loop over transition operators
            for t_left, tOp_left in enumerate(tOps):
                psiL = applyOp(n_spin_orbitals, tO_left, psi, slaterWeightMin,
                        restrictions)
                for t_right, tOp_right in enumerate(tOps):
                    psiR = applyOp(n_spin_orbitals, tOp_right, psi, slaterWeightMin,
                            restrictions)
                    normalization = sqrt(norm2(psiR))
                    for state in psiR.keys():
                        psiR[state] /= normalization
                    g[i][t_left, t_right, :] = normalization*finite.inner(psiL, psiR)*spectra.getGreen(
                            n_spin_orbitals, e, psiR, hOp, w, delta, krylovSize,
                            slaterWeightMin, restrictions, h,
                            parallelization_mode="serial")
        # Distribute the Green's functions among the ranks
        for r in range(ranks):
            gTmp = comm.bcast(g, root=r)
            for i,gValue in gTmp.items():
                gs[i,:,:,:] = gValue
    elif parallelization_mode == "H_build":
        # Loop over transition operators
        for t_left, tOp_left in enumerate(tOps):
            t_big_left = {}
            for t_right, tOp_right in enumerate(tOps):
                t_big_right = {}
                # Loop over eigen states
                for i in range(n):
                    psi =  psis[i]
                    e = es[i]
                    psiL = applyOp(n_spin_orbitals, tOp_left, psi, slaterWeightMin, restrictions, t_big_left)
                    psiR = applyOp(n_spin_orbitals, tOp_right, psi, slaterWeightMin, restrictions, t_big_right)
                    # if rank == 0: print("len(t_big) = {:d}".format(len(t_big)))
                    normalization = sqrt(norm2(psiR))
                    for state in psiR.keys():
                        psiR[state] /= normalization
                    gs[i, t_left, t_right, :] = normalization*finite.inner(psiL, psiR)*spectra.getGreen(
                        n_spin_orbitals, e, psiR, hOp, w, delta, krylovSize,
                        slaterWeightMin, restrictions, h,
                        parallelization_mode=parallelization_mode)
    else:
        raise Exception("Incorrect value of variable parallelization_mode.")
    return gs

if __name__== "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser(description="Spectroscopy simulations")
    parser.add_argument(
        "clustername",
        type=str,
        help="Id of cluster, used for generating the filename in which to store the calculated self-energy.",
    )
    parser.add_argument(
        "h0_filename",
        type=str,
        help="Filename of non-interacting Hamiltonian, in pickle-format.",
    )
    parser.add_argument(
        "radial_filename",
        type=str,
        help="Filename of radial part of correlated orbitals.",
    )
    parser.add_argument(
        "--ls",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Angular momenta of correlated orbitals.",
    )
    parser.add_argument(
        "--nBaths",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Number of bath states, for each angular momentum.",
    )
    parser.add_argument(
        "--nValBaths",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Number of valence bath states, for each angular momentum.",
    )
    parser.add_argument(
        "--n0imps",
        type=int,
        nargs="+",
        default=[6, 8],
        help="Initial impurity occupation, for each angular momentum.",
    )
    parser.add_argument(
        "--dnTols",
        type=int,
        nargs="+",
        default=[0, 2],
        help=("Max devation from initial impurity occupation, " "for each angular momentum."),
    )
    parser.add_argument(
        "--dnValBaths",
        type=int,
        nargs="+",
        default=[0, 2],
        help=("Max number of electrons to leave valence bath orbitals, " "for each angular momentum."),
    )
    parser.add_argument(
        "--dnConBaths",
        type=int,
        nargs="+",
        default=[0, 0],
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
        "--Fpp",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0],
        help="Slater-Condon parameters Fpp. p-orbitals are assumed.",
    )
    parser.add_argument(
        "--Fpd",
        type=float,
        nargs="+",
        default=[8.9, 0, 6.8],
        help="Slater-Condon parameters Fpd. p- and d-orbitals are assumed.",
    )
    parser.add_argument(
        "--Gpd",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 0, 2.8],
        help="Slater-Condon parameters Gpd. p- and d-orbitals are assumed.",
    )
    parser.add_argument(
        "--xi_2p",
        type=float,
        default=11.629,
        help="SOC value for p-orbitals. p-orbitals are assumed.",
    )
    parser.add_argument(
        "--xi_3d",
        type=float,
        default=0.096,
        help="SOC value for d-orbitals. d-orbitals are assumed.",
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
    parser.add_argument("--nPrintSlaterWeights", type=int, default=3, help="Printing parameter.")
    parser.add_argument("--tolPrintOccupation", type=float, default=0.5, help="Printing parameter.")
    parser.add_argument("--T", type=float, default=300, help="Temperature (Kelvin).")
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
        "--deltaRIXS",
        type=float,
        default=0.050,
        help=("Smearing, half width half maximum (HWHM). " "Due to finite lifetime of excited states."),
    )
    parser.add_argument(
        "--deltaNIXS",
        type=float,
        default=0.100,
        help=("Smearing, half width half maximum (HWHM). " "Due to finite lifetime of excited states."),
    )
    parser.add_argument('--RIXS_projectors_filename', type=str, default=None,
                        help=('File containing the RIXS projectors. Separated by newlines.'))

    args = parser.parse_args()

    # Sanity checks
    assert len(args.ls) == len(args.nBaths)
    assert len(args.ls) == len(args.nValBaths)
    for nBath, nValBath in zip(args.nBaths, args.nValBaths):
        assert nBath >= nValBath
    for ang, n0imp in zip(args.ls, args.n0imps):
        assert n0imp <= 2 * (2 * ang + 1)  # Full occupation
        assert n0imp >= 0
    assert len(args.Fdd) == 5
    assert len(args.Fpp) == 3
    assert len(args.Fpd) == 3
    assert len(args.Gpd) == 4
    assert len(args.hField) == 3

    get_selfenergy(
            clustername=args.clustername,
            h0_filename=args.h0_filename,
            ls=tuple(args.ls), 
            nBaths=tuple(args.nBaths),
            nValBaths=tuple(args.nValBaths), 
            n0imps=tuple(args.n0imps),
            dnTols=tuple(args.dnTols), 
            dnValBaths=tuple(args.dnValBaths),
            dnConBaths=tuple(args.dnConBaths),
            Fdd=tuple(args.Fdd), 
            xi_3d=args.xi_3d,
            chargeTransferCorrection=args.chargeTransferCorrection,
            hField=tuple(args.hField), 
            nPsiMax=args.nPsiMax,
            nPrintSlaterWeights=args.nPrintSlaterWeights,
            tolPrintOccupation=args.tolPrintOccupation,
            T=args.T, energy_cut=args.energy_cut,
            delta=args.delta,
            )
