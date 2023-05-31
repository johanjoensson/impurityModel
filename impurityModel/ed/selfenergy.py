from math import sqrt
import numpy as np
from collections import OrderedDict
from mpi4py import MPI
import pickle
import time
import argparse
import h5py

import scipy as sp

from impurityModel.ed.get_spectra import get_noninteracting_hamiltonian_operator, read_h0_operator, get_restrictions
from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.finite import daggerOp, applyOp, inner, add, norm2
from impurityModel.ed.average import k_B, thermal_average_scale_indep
from impurityModel.ed.manybody_basis import Basis

from impurityModel.ed.greens_function import get_Greens_function, save_Greens_function

eV_to_Ry = 1 / 13.605693122994


class UnphysicalSelfenergy(Exception):
    pass
class UnphysicalGreensFunction(Exception):
    pass

def matrix_print(matrix, label: str = None):
    ms = "\n".join([" ".join([f"{np.real(val): .4f}{np.imag(val):+.4f}j" for val in row]) for row in matrix])
    if label:
        print(label)
    print(ms)

def find_gs(h_op, N0, delta_occ, bath_states, num_spin_orbitals, rank, verbose = True):
    delta_imp_occ, delta_val_occ, delta_con_occ = delta_occ
    num_val_baths, num_cond_baths = bath_states
    sum_bath_states = {l: num_val_baths[l] + num_cond_baths[l] for l in num_val_baths}
    e_gs = np.inf
    basis_gs = None
    h_gs = None
    selected = 0
    energies = []
    # set up for N0 +- 1, 0
    dN = [-1, 0, 1]
    for i, d in enumerate(dN):
        basis = Basis(
                valence_baths        = num_val_baths,
                conduction_baths     = num_cond_baths,
                delta_valence_occ    = delta_val_occ,
                delta_conduction_occ = delta_con_occ,
                delta_impurity_occ   = delta_imp_occ,
                nominal_impurity_occ = {l: N0[0][l] + d for l in N0[0]},
                verbose = verbose,
                comm = MPI.COMM_WORLD
                )
        expanded_basis, h_dict, h = finite.setup_hamiltonian(num_spin_orbitals, h_op, basis, verbose = verbose)
        e_trial, _ = finite.eigensystem_new(
                h,
                expanded_basis,
                e_max = 0,
                k = 1,
                verbose = verbose,
                eigenValueTol = 0
                )
        energies.append(e_trial[0])
        if e_trial[0] < e_gs:
            e_gs = e_trial[0]
            basis_gs = expanded_basis
            h_gs = h
            selected = i
    underline = {0: ' ', 1: ' ', 2: ' '}
    underline[selected] = '='
    if rank == 0:
        l = [l for l in N0[0]][0]
        print (f"N0:    {N0[0][l] - 1: ^10d}  {N0[0][l]: ^10d}  {N0[0][l] + 1: ^10d}")
        print (f"E0:    {energies[0]: ^10.6f}  {energies[1]: ^10.6f}  {energies[2]: ^10.6f}")
        print (f"       {underline[0]*10}  {underline[1]*10}  {underline[2]*10}")


    return ({ l: N0[0][l] + dN[selected] for l in N0[0] }, N0[1], N0[2]), basis_gs, h_gs




def run(cluster, h0, iw, w, delta, tau, verbosity=0, partial_reort = False):
    """
    cluster     -- The impmod_cluster object containing loads of data.
    h0          -- Non-interacting hamiltonian.
    iw          -- Matsubara frequency mesh.
    w           -- Real frequency mesh.
    delta       -- Real frequency quantities are evaluated a frequency w_n + =j*delta
    tau         -- Temperature (in units of energy, i.e., tau = k_B*T)
    verbosity   -- How much output should be produced?
                   0 - quiet, very little output generated. (default)
                   1 - loud, detailed output generated
                   2 - SCREAM, insanely detailed output generated
    """
    energy_cut = 10
    num_psi_max = sum(2*(2*l + 1) for l in cluster.nominal_occ[0])
    nPrintSlaterWeights = 0
    tolPrintOccupation = 0.5

    sig_tmp, sig_real_tmp, sig_static_tmp = calc_selfenergy(
        h0,
        cluster.slater,
        iw,
        w,
        delta,
        cluster.nominal_occ,
        cluster.delta_occ,
        cluster.bath_states,
        tau,
        energy_cut,
        num_psi_max,
        nPrintSlaterWeights,
        tolPrintOccupation,
        verbosity,
        blocks = None, # cluster.blocks,
        rotation=cluster.rot_spherical,
        cluster_label=cluster.label,
        partial_reort = partial_reort,
    )
    cluster.sig[:,:,:] = 0
    cluster.sig_real[:,:,:] = 0
    cluster.sig_static[:,:] = 0
    block_idxs = [np.ix_(block,block) for block in cluster.blocks]
    for block in block_idxs:
        cluster.sig[block] = sig_tmp[block]
        cluster.sig_real[block] = sig_real_tmp[block]
        cluster.sig_static[block] = sig_static_tmp[block]


def calc_selfenergy(
    h0,
    slater_params,
    iw,
    w,
    delta,
    nominal_occ,
    delta_occ,
    num_bath_states,
    tau,
    energy_cut,
    num_psi_max,
    nPrintSlaterWeights,
    tolPrintOccupation,
    verbosity,
    blocks = None,
    rotation=None,
    cluster_label=None,
    partial_reort = False,
):
    """ """
    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ranks = comm.size

    num_val_baths, num_con_baths = num_bath_states
    sum_bath_states = {l: num_val_baths[l] + num_con_baths[l] for l in num_val_baths}

    ls = [l for l in num_val_baths]
    l = ls[0]

    # construct local, interacting, hamiltonian
    u = finite.getUop(l, l, l, l, slater_params)
    h = finite.addOps([h0, u])
    if rank == 0 and verbosity >= 2:
        finite.printOp(sum_bath_states, h, "Local Hamiltonian: ")
    h = finite.c2i_op(sum_bath_states, h)
    if rank == 0:
        with open(f"{cluster_label}_hamiltonian.pickle", 'wb') as f:
            pickle.dump(h, f)

    num_spin_orbitals = 2*(2*l + 1) + sum(num_val_baths[l] + num_con_baths[l] for l in num_val_baths)

    (n0_imp, n0_val, n0_con), basis, h_gs = find_gs(h, nominal_occ, delta_occ, num_bath_states, num_spin_orbitals, rank = rank, verbose = verbosity)
    delta_imp_occ, delta_val_occ, delta_con_occ = delta_occ

    restrictions = basis.restrictions
    
    if rank == 0 and verbosity >= 2:
        print("Restrictions on occupation")
        for key, res in restrictions.items():
            print(f"{key} : {res}")
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0 and verbosity >= 1:
        print("{:d} processes in the Hamiltonian.".format(len(h)))
        print("Create basis...")
        print("#basis states = {:d}".format(len(basis)))

    energy_cut *= tau

    es, psis = finite.eigensystem_new(
            h_gs,
            basis,
            energy_cut,
            k = 2*(2*l + 1),
            verbose = verbosity >= 1,
            eigenValueTol = 0
            )
    if verbosity >= 2:
        finite.printThermalExpValues(sum_bath_states, es, psis)
        finite.printExpValues(sum_bath_states, es, psis)
        # Print Slater determinants and weights
        finite.printSlaterDeterminantsAndWeights(psis=psis, nPrintSlaterWeights=nPrintSlaterWeights)

    if rank == 0 and verbosity >= 1:
        print("Consider {:d} eigenstates for the spectra \n".format(len(es)), flush=True)
        print("Calculate Interacting Green's function...")

    gs_matsubara, gs_realaxis = get_Greens_function(
        nBaths=sum_bath_states,
        matsubara_mesh=iw,
        omega_mesh=w,
        es=es,
        psis=psis,
        l=l,
        hOp=h,
        delta=delta,
        restrictions=restrictions,
        blocks = blocks,
        verbose = verbosity >= 2,
        mpi_distribute = True,
        partial_reort = partial_reort,
    )
    if iw is not None:
        gs_matsubara_thermal_avg = thermal_average_scale_indep(es[: np.shape(gs_matsubara)[0]], gs_matsubara, tau=tau)
        try:
            check_greens_function(gs_matsubara_thermal_avg, iw)
        except UnphysicalGreensFunction as err:
            if rank == 0:
                print(f"WARNING! Unphysical Matsubara-axis Greens function:\n\t{err}")
        if verbosity >= 2:
            save_Greens_function(gs=gs_matsubara_thermal_avg, omega_mesh=iw, label=f"G-{cluster_label}", e_scale=1)
            if rotation is not None:
                gs_rot = np.moveaxis(rotation[np.newaxis, :, :] @ np.moveaxis(gs_matsubara_thermal_avg, -1, 0) @ np.conj(rotation.T)[np.newaxis, :, :], 0, -1)
                save_Greens_function(gs=gs_rot, omega_mesh=iw, label=f"rotated-G-{cluster_label}", e_scale=1)
    if w is not None:
        gs_realaxis_thermal_avg = thermal_average_scale_indep(es[: np.shape(gs_realaxis)[0]], gs_realaxis, tau=tau)
        try:
            check_greens_function(gs_realaxis_thermal_avg, w)
        except UnphysicalGreensFunction as err:
            if rank == 0:
                print(f"WARNING! Unphysical real-axis Greens function:\n\t{err}")
        if verbosity >= 2:
            save_Greens_function(gs=gs_realaxis_thermal_avg, omega_mesh=w, label=f"G-{cluster_label}", e_scale=1)
            if rotation is not None:
                gs_rot = np.moveaxis(rotation[np.newaxis, :, :] @ np.moveaxis(gs_realaxis_thermal_avg, -1, 0) @ np.conj(rotation.T)[np.newaxis, :, :], 0, -1)
                save_Greens_function(gs=gs_rot, omega_mesh=w, label=f"rotated-G-{cluster_label}", e_scale=1)
    if rank == 0 and verbosity >= 1:
        print("Calculate self-energy...")
    if w is not None:
        sigma_real = get_sigma(
            omega_mesh=w,
            nBaths=sum_bath_states,
            g=gs_realaxis_thermal_avg,
            h0op=h0,
            delta=delta,
            save_G0=True,
            save_hyb=True,
            clustername=cluster_label,
            rotation=rotation,
        )
        try:
            check_sigma(sigma_real)
        except UnphysicalSelfenergy as err:
            if rank == 0:
                print(f"WARNING! Unphysical realaxis selfenergy:\n\t{err}")
    else:
        sigma_real = None
    if iw is not None:
        sigma = get_sigma(
            omega_mesh=iw,
            nBaths=sum_bath_states,
            g=gs_matsubara_thermal_avg,
            h0op=h0,
            delta=0,
            save_G0=True,
            save_hyb=True,
            clustername=cluster_label,
            rotation=rotation,
        )
        try:
            check_sigma(sigma)
        except UnphysicalSelfenergy as err:
            if rank == 0:
                print(f"WARNING! Unphysical Matsubara axis selfenergy:\n\t{err}")
    else:
        sigma = None
    if rank == 0 and verbosity >= 1:
        print(f"Calculating sig_static.")
    sigma_static = get_Sigma_static(sum_bath_states, slater_params, es, psis, l, tau)

    if verbosity >= 2:
        if iw is not None:
            save_Greens_function(gs=sigma, omega_mesh=iw, label=f"Sigma-{cluster_label}", e_scale=1)
        if w is not None:
            save_Greens_function(gs=sigma_real, omega_mesh=w, label=f"Sigma-{cluster_label}", e_scale=1)
        np.savetxt(f"real-Sigma_static-{cluster_label}.dat", np.real(sigma_static))
        np.savetxt(f"imag-Sigma_static-{cluster_label}.dat", np.imag(sigma_static))

    return sigma, sigma_real, sigma_static


def check_sigma(sigma):
    diagonals = [np.diag(sigma[:, :, i]) for i in range(sigma.shape[-1])]
    if np.any(np.imag(diagonals) > 0):
        raise UnphysicalSelfenergy("Diagonal term has positive imaginary part.")

def check_greens_function(G, energies):
    diagonals = [np.diag(G[:, :, i]) for i in range(G.shape[-1])]
    if np.any(np.imag(diagonals) > 0):
        raise UnphysicalGreensFunction("Diagonal term has positive imaginary part.")
    # norms = -1/np.pi*np.trapz(diagonals, energies)
    # if np.any(np.abs(np.imag(diagonals) - 1) > 5*(energies[1] - energies[0])):
    #     raise UnphysicalGreensFunction("Imaginary part of diagonal term is not norm-conserving.\n"
    #                                    "Integrating it does not give 1.")
    # if np.any(np.abs(np.real(diagonals) + 1) > 5*(energies[1] - energies[0])):
    #     raise UnphysicalGreensFunction("Real part of diagonal term is not norm-conserving.\n"
    #                                    "Integrating it does not give 0.")



def get_hcorr_v_hbath(h0op, sum_bath_states):
    #   The matrix form of h0op can be written
    #   [  hcorr  V^+    ]
    #   [  V      hbath  ]
    # where:
    #       - hcorr is the Hamiltonian for the correlated, impurity, orbitals.
    #       - V/V^+ is the hopping between impurity and bath orbitals.
    #       - hbath is the hamiltonian for the non-interacting, bath, orbitals.
    h0_i = finite.c2i_op(sum_bath_states, h0op)
    h0Matrix = finite.iOpToMatrix(sum_bath_states, h0_i)
    n_corr = sum([2 * (2 * l + 1) for l in sum_bath_states.keys()])
    hcorr = h0Matrix[0:n_corr, 0:n_corr]
    v_dagger = h0Matrix[0:n_corr, n_corr:]
    v = h0Matrix[n_corr:, 0:n_corr]
    h_bath = h0Matrix[n_corr:, n_corr:]
    return hcorr, v, v_dagger, h_bath


def get_sigma(
    omega_mesh, nBaths, g, h0op, delta, return_g0=False, save_G0=False, save_hyb=False, clustername="", rotation=None
):
    hcorr, v, v_dagger, hbath = get_hcorr_v_hbath(h0op, nBaths)

    n = hcorr.shape[0]
    N = hbath.shape[0]

    def hyb(ws):
        hyb = np.conj(v.T)[np.newaxis, :, :] @ np.linalg.solve(
            (ws + 1j * delta)[:, np.newaxis, np.newaxis] * np.identity(N, dtype=complex)[np.newaxis, :, :]
            - hbath[np.newaxis, :, :],
            v[np.newaxis, :, :],
        )
        return hyb

    if save_hyb:
        hybridization_function = hyb(omega_mesh)
        save_Greens_function(
            np.moveaxis(hybridization_function, 0, -1), omega_mesh, label="Hyb-" + clustername, e_scale=1
        )
        if rotation is not None:
            rotated_hyb = rotation[np.newaxis, :, :] @ hybridization_function @ np.conj(rotation.T)[np.newaxis, :, :]
            save_Greens_function(
                np.moveaxis(rotated_hyb, 0, -1), omega_mesh, label="rotated-Hyb-" + clustername, e_scale=1
            )

    wIs = (omega_mesh + 1j * delta)[:, np.newaxis, np.newaxis] * np.eye(n)[np.newaxis, :, :]
    hcorrs = hcorr[np.newaxis, :, :]
    g0_inv = wIs - hcorrs - hyb(omega_mesh)

    if save_G0:
        save_Greens_function(np.moveaxis(np.linalg.inv(g0_inv), 0, -1), omega_mesh, "G0-" + clustername, e_scale=1)
        if rotation is not None:
            rotated_g0_inv = rotation[np.newaxis, :, :] @ g0_inv @ np.conj(rotation.T)[np.newaxis, :, :]
            save_Greens_function(
                np.moveaxis(np.linalg.inv(rotated_g0_inv), 0, -1), omega_mesh, label="rotated-G0-" + clustername, e_scale=1
            )

    g_inv = np.linalg.inv(np.moveaxis(g, -1, 0))
    return np.moveaxis(g0_inv - g_inv, 0, -1)


def get_Sigma_static(nBaths, Fdd, es, psis, l, tau):
    n = 2 * (2 * l + 1)

    rhos = [finite.getDensityMatrix(nBaths, psi, l) for psi in psis]
    rhomats = np.zeros((len(rhos), n, n), dtype=complex)
    for mat, rho in zip(rhomats, rhos):
        for (state1, state2), val in rho.items():
            i = finite.c2i(nBaths, state1)
            j = finite.c2i(nBaths, state2)
            mat[i, j] = val
    rho = thermal_average_scale_indep(es, rhomats, tau)

    U = finite.getUop(l1=l, l2=l, l3=l, l4=l, R=Fdd)
    Umat = np.zeros((n, n, n, n), dtype=complex)
    for ((state1, op1), (state2, op2), (state3, op3), (state4, op4)), val in U.items():
        a = finite.c2i(nBaths, state1)
        b = finite.c2i(nBaths, state2)
        c = finite.c2i(nBaths, state3)
        d = finite.c2i(nBaths, state4)
        Umat[a, b, c, d] = 2 * val

    sigma_static = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            sigma_static += (Umat[j, :, :, i] - Umat[j, :, i, :]) * rho[i, j]

    return sigma_static


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
    tau,
    energy_cut,
    delta,
    verbose,
):
    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank
    ranks = comm.size

    # omega_mesh = np.linspace(-25, 25, 2000)
    omega_mesh = np.linspace(-1.83, 1.83, 2000)
    # omega_mesh = 1j*np.pi*tau*np.arange(start = 1, step = 2, stop = 2*375)

    if rank == 0:
        t0 = time.perf_counter()
    # -- System information --

    sum_baths = OrderedDict({ls: nBaths})
    nValBaths = OrderedDict({ls: nValBaths})
    dnValBaths = OrderedDict({ls: dnValBaths})
    dnConBaths = OrderedDict({ls: dnConBaths})

    # -- Basis occupation information --
    n0imps = OrderedDict({ls: n0imps})
    dnTols = OrderedDict({ls: dnTols})
    nominal_occ = (n0imps, {ls: nBaths}, {ls: 0})
    delta_occ = (dnTols, dnValBaths, dnConBaths)

    num_bath_states = ({ls: nValBaths[ls]}, {ls: sum_baths[ls] - nValBaths[ls]})

    # Hamiltonian
    if rank == 0:
        print("Construct the Hamiltonian operator...")
    hOp = get_noninteracting_hamiltonian_operator(
        sum_baths,
        [Fdd, None, None, None],
        [0, xi],
        [n0imps, chargeTransferCorrection],
        hField,
        h0_filename,
        rank=rank,
        verbose=verbose,
    )

    sigma, sigma_real, sigma_static = calc_selfenergy(
        h0=hOp,
        slater_params=Fdd,
        iw=None,
        w=omega_mesh,
        delta=delta,
        nominal_occ=nominal_occ,
        delta_occ=delta_occ,
        num_bath_states=num_bath_states,
        tau=tau,
        energy_cut=energy_cut,
        num_psi_max=nPsiMax,
        nPrintSlaterWeights=nPrintSlaterWeights,
        tolPrintOccupation=tolPrintOccupation,
        verbosity=2 if verbose else 0,
        cluster_label=clustername,
    )
    if rank == 0:
        print(f"Writing sig_static to files")
        np.savetxt(f"real-sig_static-{clustername}.dat", np.real(sigma_static))
        np.savetxt(f"imag-sig_static-{clustername}.dat", np.imag(sigma_static))
    if rank == 0:
        # save_Greens_function(gs = gs_thermal_avg, omega_mesh = omega_mesh, label =f'G-{clustername}', e_scale = 1)
        save_Greens_function(gs=sigma_real, omega_mesh=omega_mesh, label=f"Sigma-{clustername}", e_scale=1)


if __name__ == "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser(description="Calculate selfenergy")
    parser.add_argument(
        "h0_filename",
        type=str,
        help="Filename of non-interacting Hamiltonian.",
    )
    parser.add_argument(
        "--clustername",
        type=str,
        default="cluster",
        help="Id of cluster, used for generating the filename in which to store the calculated self-energy.",
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
        default=0,
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
    parser.add_argument("--nPrintSlaterWeights", type=int, default=3, help="Printing parameter.")
    parser.add_argument("--tolPrintOccupation", type=float, default=0.5, help="Printing parameter.")
    parser.add_argument("--tau", type=float, default=0.002, help="Fundamental temperature (kb*T).")
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
        "-v",
        "--verbose",
        action="store_true",
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
        tau=args.tau,
        energy_cut=args.energy_cut,
        delta=args.delta,
        verbose=args.verbose,
    )
