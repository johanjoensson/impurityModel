from collections import OrderedDict
import itertools
import time
import argparse

import h5py as h5
from mpi4py import MPI
import numpy as np
import scipy as sp

# from impurityModel.ed.get_spectra import get_noninteracting_hamiltonian_operator
from impurityModel.ed import finite
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.manybody_basis import CIPSI_Basis, Basis
import impurityModel.ed.product_state_representation as psr

from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.greens_function import get_Greens_function, save_Greens_function
from impurityModel.ed.block_structure import print_block_structure
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.utils import matrix_print

EV_TO_RY = 1 / 13.605693122994


class UnphysicalGreensFunctionError(Exception):
    """
    Excpetion signalling an unphysical Greens function, i.e. the imaginary part is positive for some frequencies.
    """


def fixed_peak_dc(
    h0_op,
    N0,
    mixed_valence,
    impurity_orbitals,
    bath_states,
    u4,
    peak_position,
    dc_guess,
    spin_flip_dj,
    tau,
    rank,
    verbose,
    dense_cutoff,
):
    n_orb = sum(len(block) for imp_orbs in impurity_orbitals.values() for block in imp_orbs)
    peak_position = max(peak_position, 4 * tau)
    valence_baths, conduction_baths = bath_states
    u = finite.getUop_from_rspt_u4(u4)
    h_op_i = ManyBodyOperator(finite.addOps([h0_op, u]))
    dc_trial = dc_guess

    Np = {l: N0[l] + 1 for l in N0}
    Nm = {l: N0[l] - 1 for l in N0}
    if peak_position >= 0:
        basis_upper = CIPSI_Basis(
            impurity_orbitals,
            bath_states,
            H=h_op_i,
            nominal_impurity_occ=Np,
            mixed_valence=mixed_valence,
            truncation_threshold=1e5,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
        )
        basis_lower = CIPSI_Basis(
            impurity_orbitals,
            bath_states,
            H=h_op_i,
            nominal_impurity_occ=N0,
            mixed_valence=mixed_valence,
            truncation_threshold=1e5,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
        )
    else:
        basis_upper = CIPSI_Basis(
            impurity_orbitals,
            bath_states,
            H=h_op_i,
            nominal_impurity_occ=N0,
            mixed_valence=mixed_valence,
            truncation_threshold=1e5,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
        )
        basis_lower = CIPSI_Basis(
            impurity_orbitals,
            bath_states,
            H=h_op_i,
            nominal_impurity_occ=Nm,
            mixed_valence=mixed_valence,
            truncation_threshold=1e5,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
            tau=tau,
        )

    dc_op_i = ManyBodyOperator(
        {
            ((i, "c"), (j, "a")): -dc_trial[i, j] + 0j
            for i in range(dc_trial.shape[0])
            for j in range(dc_trial.shape[1])
            if abs(dc_trial[i, j]) > 0
        }
    )
    h_op = h_op_i + dc_op_i
    _ = basis_upper.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-2, slaterWeightMin=1e-8)
    _ = basis_lower.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-2, slaterWeightMin=1e-8)

    energy_cut = 0  # -tau * np.log(1e-4)

    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]
    impurity_ix = np.ix_(impurity_indices, impurity_indices)

    def F(dc_fac):
        dc = dc_fac * dc_trial
        dc_op_i = {
            ((i, "c"), (j, "a")): -dc[i, j] + 0j
            for i in range(dc_trial.shape[0])
            for j in range(dc_trial.shape[1])
            if abs(dc_trial[i, j]) > 0
        }
        h_op = h_op_i + ManyBodyOperator(dc_op_i)

        h = basis_upper.build_sparse_matrix(h_op)
        e_upper, psi_upper = finite.eigensystem_new(
            h,
            e_max=energy_cut,
            k=1,
            eigenValueTol=np.sqrt(np.finfo(float).eps),
            return_eigvecs=True,
            comm=basis_upper.comm,
            dense=basis_upper.size < dense_cutoff,
        )
        h = basis_lower.build_sparse_matrix(h_op)
        e_lower, psi_lower = finite.eigensystem_new(
            h,
            e_max=energy_cut,
            k=1,
            eigenValueTol=np.sqrt(np.finfo(float).eps),
            return_eigvecs=True,
            comm=basis_upper.comm,
            dense=basis_lower.size < dense_cutoff,
        )
        rho_lower = basis_lower.build_density_matrices(
            basis_lower.build_state(psi_lower.T),
            orbital_indices_left=impurity_indices,
            orbital_indices_right=impurity_indices,
        )
        rho_upper = basis_upper.build_density_matrices(
            basis_upper.build_state(psi_upper.T),
            orbital_indices_left=impurity_indices,
            orbital_indices_right=impurity_indices,
        )
        rho_lower = finite.thermal_average_scale_indep(e_lower, rho_lower, basis_lower.tau)
        rho_upper = finite.thermal_average_scale_indep(e_upper, rho_upper, basis_upper.tau)
        avg_dc_lower = np.real(np.trace(rho_lower @ dc))
        avg_dc_upper = np.real(np.trace(rho_upper @ dc))
        if abs(avg_dc_upper - avg_dc_lower) < max(tau, 1e-2):
            return 0
        return (e_upper[0] - e_lower[0] - peak_position) / (avg_dc_upper - avg_dc_lower)

    dc_fac = 1
    for _ in range(5):
        dc_fac += F(dc_fac)
    if verbose:
        print(f"Peak position {peak_position}")
        matrix_print(dc_guess, label="DC guess")
        matrix_print(dc_fac * dc_trial, label="dc found")
        print("=" * 80)

    return dc_fac * dc_trial


def calc_selfenergy(
    h0,
    u4,
    iw,
    w,
    delta,
    nominal_occ,
    mixed_valence,
    impurity_orbitals,
    bath_states,
    tau,
    verbosity,
    block_structure,
    rot_to_spherical,
    cluster_label,
    reort,
    dense_cutoff,
    spin_flip_dj,
    comm,
    chain_restrict,
    occ_cutoff,
    truncation_threshold,
    slaterWeightMin,
    dN,
    sparse_green,
):
    """
    Calculate the self energy of the impurity.
    """
    # MPI variables
    rank = comm.rank if comm is not None else 0

    valence_baths, conduction_baths = bath_states
    total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
    sum_bath_states = {
        i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
        for i in valence_baths
    }

    # construct local, interacting, hamiltonian
    u = finite.getUop_from_rspt_u4(u4)
    h = ManyBodyOperator(h0) + ManyBodyOperator(u)
    # h = finite.addOps([h0, u])
    basis_information = {
        "impurity_orbitals": impurity_orbitals,
        "bath_states": bath_states,
        "N0": nominal_occ,
        "mixed_valence": mixed_valence,
        "tau": tau,
        "chain_restrict": chain_restrict,
        "dense_cutoff": dense_cutoff,
        "spin_flip_dj": spin_flip_dj,
        "rank": rank,
        "comm": comm,
        "truncation_threshold": truncation_threshold,
    }
    psis, es, ground_state_basis, thermal_rho, gs_info = calc_gs(
        h, basis_information, block_structure, rot_to_spherical, verbosity >= 1
    )
    restrictions = ground_state_basis.restrictions

    if restrictions is not None and verbosity >= 2:
        print("Restrictions GS on occupation")
        for indices, limits in restrictions.items():
            print(f"---> {sorted(indices)} : {limits}")

    if verbosity >= 1:
        print(f"Consider {len(es):d} eigenstates for the spectra \n")
        print("Calculate Interacting Green's function...", flush=verbosity >= 2)

    gs_matsubara, gs_realaxis = get_Greens_function(
        matsubara_mesh=iw,
        omega_mesh=w,
        psis=psis,
        es=es,
        tau=tau,
        basis=ground_state_basis,
        hOp=h,
        delta=delta,
        blocks=[block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
        verbose=verbosity >= 1,
        verbose_extra=verbosity >= 2,
        reort=reort,
        dN=dN,
        occ_cutoff=occ_cutoff,
        slaterWeightMin=slaterWeightMin,
        sparse=sparse_green,
    )
    if gs_matsubara is not None:
        try:
            for gs in gs_matsubara:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            raise UnphysicalGreensFunctionError("Matsubara interacting Greens function:\n" + str(err)) from None
    if gs_realaxis is not None:
        try:
            for gs in gs_realaxis:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            raise UnphysicalGreensFunctionError("Real frequency interacting Greens function:\n" + str(err)) from None

    if verbosity >= 1:
        print("Calculate self-energy...")
    if gs_realaxis is not None:
        sigma_real = get_sigma(
            omega_mesh=w,
            impurity_orbitals=total_impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gs_realaxis,
            h0op=h0,
            delta=delta,
            clustername=cluster_label,
            blocks=[block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
        )
        try:
            for sig in sigma_real:
                check_greens_function(sig)
        except UnphysicalGreensFunctionError as err:
            raise UnphysicalGreensFunctionError("Real frequency self-energy:\n" + str(err)) from None
    else:
        sigma_real = None
    if gs_matsubara is not None:
        sigma = get_sigma(
            omega_mesh=iw,
            impurity_orbitals=total_impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gs_matsubara,
            h0op=h0,
            delta=0,
            clustername=cluster_label,
            blocks=[block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
        )
        try:
            for sig in sigma:
                check_greens_function(sig)
        except UnphysicalGreensFunctionError as err:
            raise UnphysicalGreensFunctionError("Matsubara self-energy:\n" + str(err)) from None
    else:
        sigma = None
    if verbosity >= 1:
        print("Calculating sig_static.")
    impurity_indices = [
        orb
        for impurity_blocks in ground_state_basis.impurity_orbitals.values()
        for block in impurity_blocks
        for orb in block
    ]
    impurity_ix = np.ix_(impurity_indices, impurity_indices)
    sigma_static = get_Sigma_static(u4, thermal_rho[impurity_ix])

    return {
        "sigma": sigma,
        "sigma_real": sigma_real,
        "sigma_static": sigma_static,
        "gs_matsubara": gs_matsubara,
        "gs_realaxis": gs_realaxis,
        "thermal_rho": thermal_rho,
        "rhos": gs_info["rhos"],
    }


def check_greens_function(G):
    """
    Verify that G makes physical sense.
    """
    if np.any(np.diagonal(G, axis1=1, axis2=2).imag > 0):
        raise UnphysicalGreensFunctionError("Diagonal term has positive imaginary part.")


def get_hcorr_v_hbath(h0op, impurity_orbitals, sum_bath_states):
    """
    The matrix form of h0op can be written
      [  hcorr  V^+    ]
      [  V      hbath  ]
    where:
          - hcorr is the Hamiltonian for the correlated, impurity, orbitals.
          - V/V^+ is the hopping between impurity and bath orbitals.
          - hbath is the hamiltonian for the non-interacting, bath, orbitals.
    """
    # h0_i = finite.c2i_op(sum_bath_states, h0op)
    # h0Matrix = finite.iOpToMatrix(sum_bath_states, h0op)

    # n_corr = sum([2 * (2 * l + 1) for l in sum_bath_states.keys()])
    num_spin_orbitals = sum(impurity_orbitals[i] + sum_bath_states[i] for i in impurity_orbitals)
    n_corr = sum(ni for ni in impurity_orbitals.values())
    h0Matrix = np.zeros((num_spin_orbitals, num_spin_orbitals), dtype=complex)
    for ((i, opi), (j, opj)), val in h0op.items():
        if opi == "c" and opj == "a":
            h0Matrix[i, j] = val
        elif opj == "c" and opi == "a":
            if i == j:
                h0Matrix[i, j] = 1 - val
            else:
                h0Matrix[i, j] = -val
    hcorr = h0Matrix[0:n_corr, 0:n_corr]
    v_dagger = h0Matrix[0:n_corr, n_corr:]
    v = h0Matrix[n_corr:, 0:n_corr]
    h_bath = h0Matrix[n_corr:, n_corr:]
    return hcorr, v, v_dagger, h_bath


def hyb(ws, v, hbath, delta):
    """
    Calculate hybridization function from hopping parameters and bath energies.
    Δ = v^dag [(ws+i*delta)I - hbath]^-1 V
    """
    return np.conj(v.T) @ np.linalg.solve(
        (ws + 1j * delta)[:, None, None] * np.identity(hbath.shape[0], dtype=complex)[None, :, :] - hbath[None, :, :],
        v[None, :, :],
    )


def get_sigma(
    omega_mesh,
    impurity_orbitals,
    nBaths,
    gs,
    h0op,
    delta,
    blocks,
    clustername="",
):
    """
    Calculate self-energy from interacting Greens function and local hamiltonian.
    """
    hcorr, v_full, _, h_bath = get_hcorr_v_hbath(h0op, impurity_orbitals, nBaths)

    res = []
    for block, g in zip(blocks, gs):
        block_ix = np.ix_(block, block)
        wIs = (omega_mesh + 1j * delta)[:, np.newaxis, np.newaxis] * np.eye(len(block))[np.newaxis, :, :]
        g0_inv = wIs - hcorr[block_ix] - hyb(omega_mesh, v_full[:, block], h_bath, delta)
        res.append(g0_inv - np.linalg.inv(g))

    return res


def get_Sigma_static(U4, rho):
    """
    Calculate the static (Hartree-Fock) self-energy.
    """
    sigma_static = np.zeros_like(rho)
    for i, j in itertools.product(range(rho.shape[0]), range(rho.shape[1])):
        sigma_static += (U4[j, :, :, i] - U4[j, :, i, :]) * rho[i, j]

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
    tau,
    energy_cut,
    delta,
    verbose,
):
    """
    Calculate the self energy starting from a large number of arguments.
    """
    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # omega_mesh = np.linspace(-25, 25, 2000)
    omega_mesh = np.linspace(-1.83, 1.83, 2000)
    # omega_mesh = 1j*np.pi*tau*np.arange(start = 1, step = 2, stop = 2*375)

    # if rank == 0:
    #     t0 = time.perf_counter()
    # -- System information --

    sum_baths = OrderedDict({ls: nBaths})
    nValBaths = OrderedDict({ls: nValBaths})

    # -- Basis occupation information --
    n0imps = OrderedDict({ls: n0imps})
    nominal_occ = (n0imps, {ls: nBaths}, {ls: 0})

    num_bath_states = ({ls: nValBaths[ls]}, {ls: sum_baths[ls] - nValBaths[ls]})

    # Hamiltonian
    if rank == 0 and verbose:
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
        iw=None,
        w=omega_mesh,
        delta=delta,
        nominal_occ=nominal_occ,
        bath_states=num_bath_states,
        tau=tau,
        energy_cut=energy_cut,
        nPrintSlaterWeights=nPrintSlaterWeights,
        verbosity=2 if verbose else 0,
        cluster_label=clustername,
    )

    # if rank == 0:
    #     print("Writing sig_static to files")
    #     np.savetxt(f"real-sig_static-{clustername}.dat", np.real(sigma_static))
    #     np.savetxt(f"imag-sig_static-{clustername}.dat", np.imag(sigma_static))
    # if rank == 0:
    #     save_Greens_function(gs=sigma_real, omega_mesh=omega_mesh, label=f"Sigma-{clustername}", e_scale=1)


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
        tau=args.tau,
        energy_cut=args.energy_cut,
        delta=args.delta,
        verbose=args.verbose,
    )
