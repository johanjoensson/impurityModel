from collections import OrderedDict
import itertools
import time
import argparse

import h5py as h5
from mpi4py import MPI
import numpy as np
import scipy as sp
from impurityModel.ed.get_spectra import get_noninteracting_hamiltonian_operator
from impurityModel.ed import finite
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.manybody_basis import CIPSI_Basis
import impurityModel.ed.product_state_representation as psr

from impurityModel.ed.greens_function import get_Greens_function, save_Greens_function

EV_TO_RY = 1 / 13.605693122994


def matrix_print(matrix: np.ndarray, label: str = None) -> None:
    """
    Pretty print the matrix, with optional label.
    """
    ms = "\n".join([" ".join([f"{np.real(val): .4f}{np.imag(val):+.4f}j" for val in row]) for row in matrix])
    if label is not None:
        print(label)
    print(ms)


class UnphysicalSelfenergyError(Exception):
    """
    Excpetion signalling an unphysical self-energy, i.e. the imaginary part is positive for some frequencies.
    """


class UnphysicalGreensFunctionError(Exception):
    """
    Excpetion signalling an unphysical Greens function, i.e. the imaginary part is positive for some frequencies.
    """


def fixed_peak_dc(h0_op, dc_struct, rank, verbose, dense_cutoff):
    N0 = dc_struct.nominal_occ
    delta_impurity_occ, delta_valence_occ, delta_conduction_occ = dc_struct.delta_occ
    peak_position = max(dc_struct.peak_position, 4 * dc_struct.tau)
    num_valence_bath_states, num_conduction_bath_states = dc_struct.bath_states
    sum_bath_states = {l: num_valence_bath_states[l] + num_conduction_bath_states[l] for l in num_valence_bath_states}
    u = finite.getUop_from_rspt_u4(dc_struct.u4)
    dc_trial = dc_struct.dc_guess
    # dc_trial = np.identity(dc_struct.dc_guess.shape[0], dtype=complex)

    Np = ({l: N0[0][l] + 1 for l in N0[0]}, N0[1], N0[2])
    Nm = ({l: N0[0][l] - 1 for l in N0[0]}, N0[1], N0[2])
    if peak_position >= 0:
        basis_upper = CIPSI_Basis(
            impurity_orbitals=dc_struct.impurity_orbitals,
            valence_baths=num_valence_bath_states,
            conduction_baths=num_conduction_bath_states,
            delta_valence_occ=delta_valence_occ,
            delta_conduction_occ=delta_conduction_occ,
            delta_impurity_occ=delta_impurity_occ,
            nominal_impurity_occ=Np[0],
            truncation_threshold=1e9,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=dc_struct.spin_flip_dj,
        )
        basis_lower = CIPSI_Basis(
            impurity_orbitals=dc_struct.impurity_orbitals,
            valence_baths=num_valence_bath_states,
            conduction_baths=num_conduction_bath_states,
            delta_valence_occ=delta_valence_occ,
            delta_conduction_occ=delta_conduction_occ,
            delta_impurity_occ=delta_impurity_occ,
            nominal_impurity_occ=N0[0],
            truncation_threshold=1e9,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=dc_struct.spin_flip_dj,
        )
    else:
        basis_upper = CIPSI_Basis(
            impurity_orbitals=dc_struct.impurity_orbitals,
            valence_baths=num_valence_bath_states,
            conduction_baths=num_conduction_bath_states,
            delta_valence_occ=delta_valence_occ,
            delta_conduction_occ=delta_conduction_occ,
            delta_impurity_occ=delta_impurity_occ,
            nominal_impurity_occ=N0[0],
            truncation_threshold=1e9,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=dc_struct.spin_flip_dj,
        )
        basis_lower = CIPSI_Basis(
            impurity_orbitals=dc_struct.impurity_orbitals,
            valence_baths=num_valence_bath_states,
            conduction_baths=num_conduction_bath_states,
            delta_valence_occ=delta_valence_occ,
            delta_conduction_occ=delta_conduction_occ,
            delta_impurity_occ=delta_impurity_occ,
            nominal_impurity_occ=Nm[0],
            truncation_threshold=1e9,
            verbose=verbose,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=dc_struct.spin_flip_dj,
        )
    h_op_i = finite.addOps([h0_op, u])
    # h_op_i = finite.c2i_op(sum_bath_states, h_op_c)
    # h_dict = basis_upper.expand(h_op_i, dense_cutoff=dense_cutoff, de2_min=1e-6)
    # h_dict = basis_lower.expand(h_op_i, dense_cutoff=dense_cutoff, de2_min=1e-6)

    def F(dc_fac):
        dc = dc_fac * dc_trial
        bu = basis_upper.copy()
        bl = basis_lower.copy()
        dc_op_i = {
            ((i, "c"), (j, "a")): -dc[i, j] + 0j
            for i in range(dc_trial.shape[0])
            for j in range(dc_trial.shape[1])
            if abs(dc_trial[i, j]) > 0
        }
        h_op = finite.addOps([h_op_i, dc_op_i])
        h_dict = bu.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-8)
        h = bu.build_sparse_matrix(h_op, {}) if bu.size > dense_cutoff else bu.build_dense_matrix(h_op, h_dict)
        e_upper, psi_upper = finite.eigensystem_new(
            h,
            e_max=0,
            k=1,
            eigenValueTol=0,
            return_eigvecs=True,
        )
        h_dict = bl.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-8)
        h = bl.build_sparse_matrix(h_op, {}) if bl.size > dense_cutoff else bl.build_dense_matrix(h_op, h_dict)
        e_lower, psi_lower = finite.eigensystem_new(
            h,
            e_max=0,
            k=1,
            eigenValueTol=0,
            return_eigvecs=True,
        )
        psi_lower_local = bl.build_state(psi_lower[:, 0].T)[0]
        psi_upper_local = bu.build_state(psi_upper[:, 0].T)[0]
        psi_lowers = bl.comm.allgather(psi_lower_local)
        psi_uppers = bu.comm.allgather(psi_upper_local)
        psi_lower = {}
        psi_upper = {}
        for psi_lower_local, psi_upper_local in zip(psi_lowers, psi_uppers):
            for state in psi_lower_local:
                psi_lower[state] = psi_lower_local[state] + psi_lower.get(state, 0)
            for state in psi_upper_local:
                psi_upper[state] = psi_upper_local[state] + psi_upper.get(state, 0)
        rho_lower = finite.build_impurity_density_matrix(
            sum(no for no in bl.impurity_orbitals.values()), sum(nb for nb in sum_bath_states.values()), psi_lower
        )
        rho_upper = finite.build_impurity_density_matrix(
            sum(no for no in bu.impurity_orbitals.values()), sum(nb for nb in sum_bath_states.values()), psi_upper
        )
        avg_dc_lower = np.real(np.trace(rho_lower @ dc))
        avg_dc_upper = np.real(np.trace(rho_upper @ dc))
        if abs(avg_dc_upper - avg_dc_lower) < min(dc_struct.tau, 1e-2):
            return dc_trial
        return (e_upper[0] - e_lower[0] - peak_position) / (avg_dc_upper - avg_dc_lower)
        # return (e_upper[0] - e_lower[0] - peak_position + avg_dc_upper - avg_dc_lower) / (avg_dc_upper - avg_dc_lower)

    # res = sp.optimize.root_scalar(F, x0=1)
    # dc_fac = res.root
    # dc_fac = sp.optimize.newton(F, x0=1)
    dc_fac = 1
    for _ in range(5):
        dc_fac += F(dc_fac)
    if verbose:
        print(f"Peak position {dc_struct.peak_position}")
        print(f"DC guess {dc_struct.dc_guess}")
        print(f"dc found : {dc_fac*dc_trial}")

    return dc_fac * dc_trial


def matrix_print(matrix: np.ndarray, label: str = None) -> None:
    """
    Pretty print the matrix, with optional label.
    """
    ms = "\n".join([" ".join([f"{np.real(val): .4f}{np.imag(val):+.4f}j" for val in row]) for row in matrix])
    if label is not None:
        print(label)
    print(ms)


def calc_occ_e(
    h_op, N0, N_imp, N_val, N_con, delta_imp, delta_val, delta_con, spin_flip_dj, dense_cutoff, comm, verbose
):
    basis = CIPSI_Basis(
        impurity_orbitals=N_imp,
        H=h_op,
        valence_baths=N_val,
        conduction_baths=N_con,
        delta_impurity_occ=delta_imp,
        delta_valence_occ=delta_val,
        delta_conduction_occ=delta_con,
        nominal_impurity_occ=N0,
        truncation_threshold=1e9,
        verbose=verbose,
        spin_flip_dj=spin_flip_dj,
        comm=comm,
    )
    h_dict = basis.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-8)
    h = basis.build_sparse_matrix(h_op, h_dict) if basis.size > dense_cutoff else basis.build_dense_matrix(h_op, h_dict)

    e_trial = finite.eigensystem_new(
        h,
        e_max=0,
        k=basis.num_spin_orbitals,
        eigenValueTol=0,
        return_eigvecs=False,
    )
    return e_trial[0], basis, h_dict


def find_gs(h_op, N0, delta_occ, bath_states, num_impurity_orbitals, rank, verbose, dense_cutoff, spin_flip_dj, comm):
    """
    Find the occupation corresponding to the lowest energy, compare N0 - 1, N0 and N0 + 1
    Returns:
    gs_impurity_occ, dict: Impurity occupation corresponding to the lowest energy.
    basis_gs, ManybodyBasis: Initial basis for the ground state
    h_dict_gs, dict: Memoized states for the hamiltonian operator.
    """
    delta_imp_occ, delta_val_occ, delta_con_occ = delta_occ
    num_val_baths, num_cond_baths = bath_states
    e_gs = np.inf
    basis_gs = None
    gs_impurity_occ = None
    for dN in [0, -1, 1]:
        e_trial, basis, h_dict = calc_occ_e(
            h_op,
            {i: N0[0][i] + dN for i in N0[0]},
            num_impurity_orbitals,
            num_val_baths,
            num_cond_baths,
            delta_imp_occ,
            delta_val_occ,
            delta_con_occ,
            spin_flip_dj,
            dense_cutoff,
            comm=comm,
            verbose=False,
        )
        if e_trial < e_gs:
            e_gs = e_trial
            basis_gs = basis
            h_dict_gs = h_dict
            dN_gs = dN
            gs_impurity_occ = {i: N0[0][i] + dN for i in N0[0]}
    while dN_gs != 0:
        e_trial, basis, h_dict = calc_occ_e(
            h_op,
            {i: gs_impurity_occ[i] + dN_gs for i in gs_impurity_occ},
            num_impurity_orbitals,
            num_val_baths,
            num_cond_baths,
            delta_imp_occ,
            delta_val_occ,
            delta_con_occ,
            spin_flip_dj,
            dense_cutoff,
            comm=comm,
            verbose=False,
        )
        if e_trial >= e_gs:
            break
        e_gs = e_trial
        basis_gs = basis
        h_dict_gs = h_dict
        gs_impurity_occ = {i: gs_impurity_occ[i] + dN_gs for i in gs_impurity_occ}
    if verbose:
        print(f"Nominal GS occupation; impurity: {gs_impurity_occ}, valence: {N0[1]}, conduction: {N0[2]}")
    return (gs_impurity_occ, N0[1], N0[2]), basis_gs, h_dict_gs


def run(cluster, h0, iw, w, delta, tau, verbosity, reort, dense_cutoff, comm):
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
    cluster.sig[:, :, :] = 0
    cluster.sig_real[:, :, :] = 0
    cluster.sig_static[:, :] = 0

    sigma, sigma_real, cluster.sig_static[:, :] = calc_selfenergy(
        h0,
        cluster.u4,
        iw,
        w,
        delta,
        cluster.nominal_occ,
        cluster.delta_occ,
        cluster.bath_states,
        cluster.h_star_bath,
        cluster.v_star,
        tau,
        verbosity,
        blocks=[cluster.blocks[i] for i in cluster.inequivalent_blocks],
        rot_to_spherical=np.conj(cluster.corr_to_cf.T) @ cluster.corr_to_spherical,
        cluster_label=cluster.label,
        reort=reort,
        dense_cutoff=dense_cutoff,
        spin_flip_dj=cluster.spin_flip_dj,
        comm=comm,
    )

    if comm.rank == 0:
        for inequiv_i, (sig, sig_real) in enumerate(zip(sigma, sigma_real)):
            for block_i in cluster.identical_blocks[inequiv_i]:
                block_idx_matsubara = np.ix_(range(sig.shape[0]), cluster.blocks[block_i], cluster.blocks[block_i])
                cluster.sig[block_idx_matsubara] = sig
                block_idx_real = np.ix_(range(sig_real.shape[0]), cluster.blocks[block_i], cluster.blocks[block_i])
                cluster.sig_real[block_idx_real] = sig_real
            for block_i in cluster.transposed_blocks[inequiv_i]:
                block_idx_matsubara = np.ix_(range(sig.shape[0]), cluster.blocks[block_i], cluster.blocks[block_i])
                cluster.sig[block_idx_matsubara] = np.transpose(sig, (0, 2, 1))
                block_idx_real = np.ix_(range(sig_real.shape[0]), cluster.blocks[block_i], cluster.blocks[block_i])
                cluster.sig_real[block_idx_real] = np.transpose(sig_real, (0, 2, 1))


def calc_selfenergy(
    h0,
    u4,
    iw,
    w,
    delta,
    nominal_occ,
    delta_occ,
    num_bath_states,
    h_star_bath,
    v_star,
    tau,
    verbosity,
    blocks,
    rot_to_spherical,
    cluster_label,
    reort,
    dense_cutoff,
    spin_flip_dj,
    comm,
):
    """
    Calculate the self energy of the impurity.
    """
    # MPI variables
    rank = comm.rank

    n_imp_orbs = rot_to_spherical.shape[0]
    num_val_baths, num_con_baths = num_bath_states
    sum_bath_states = {i: num_val_baths[i] + num_con_baths[i] for i in num_val_baths}

    # construct local, interacting, hamiltonian
    u = finite.getUop_from_rspt_u4(u4)
    h = finite.addOps([h0, u])

    (n0_imp, n0_val, n0_con), basis, h_dict = find_gs(
        h,
        nominal_occ,
        delta_occ,
        num_bath_states,
        {i: n_imp_orbs for i in num_val_baths},
        rank=rank,
        verbose=verbosity,
        dense_cutoff=dense_cutoff,
        spin_flip_dj=spin_flip_dj,
        comm=comm,
    )
    delta_imp_occ, delta_val_occ, delta_con_occ = delta_occ
    restrictions = basis.restrictions

    if restrictions is not None and verbosity >= 2:
        print("Restrictions on occupation")
        for key, res in restrictions.items():
            print(f"---> {key} : {res}")

    energy_cut = -tau * np.log(1e-4)

    basis.tau = tau
    h_dict = basis.expand(h, H_dict=h_dict, dense_cutoff=dense_cutoff, de2_min=1e-8)
    if basis.size <= dense_cutoff:
        h_gs = basis.build_dense_matrix(h, h_dict)
    else:
        h_gs = basis.build_sparse_matrix(h, h_dict)
    es, psis_dense = finite.eigensystem_new(
        h_gs,
        e_max=energy_cut,
        k=n_imp_orbs,
        eigenValueTol=0,
    )
    psis = basis.build_state(psis_dense.T, slaterWeightMin=np.finfo(float).eps)
    basis.clear()
    basis.add_states(set(state for psi in psis for state in psi))
    if verbosity >= 1:
        print(f"{len(h)} processes in the Hamiltonian.")
        print(f"#basis states = {len(basis)}")
    gs_stats = basis.get_state_statistics(psis)
    all_psis = comm.gather(psis)
    local_psis = [{} for _ in psis]
    if rank == 0:
        for psis_r in all_psis:
            for i in range(len(local_psis)):
                for state in psis_r[i]:
                    local_psis[i][state] = psis_r[i][state] + local_psis[i].get(state, 0)
        rho_imps = np.array(
            [finite.build_impurity_density_matrix(n_imp_orbs, sum(sum_bath_states.values()), psi) for psi in local_psis]
        )
        rho_baths = np.array(
            [finite.build_bath_density_matrix(n_imp_orbs, sum(sum_bath_states.values()), psi) for psi in local_psis]
        )
    else:
        rho_imps = np.empty((len(psis), n_imp_orbs, n_imp_orbs), dtype=complex)
        rho_baths = np.empty((len(psis), sum(sum_bath_states.values()), sum(sum_bath_states.values())), dtype=complex)
    comm.Bcast(rho_imps)
    comm.Bcast(rho_baths)
    if verbosity >= 1:
        finite.printThermalExpValues_new(
            {i: n_imp_orbs for i in num_val_baths}, sum_bath_states, es, local_psis, tau, rot_to_spherical
        )
        finite.printExpValues(sum_bath_states, es, local_psis, rot_to_spherical, sum(n_imp_orbs for _ in num_val_baths))
        print("Ground state occupation statistics:")
        for psi_stats in gs_stats:
            print(f"{psi_stats}")
        print("Ground state bath occupation statistics:")
        thermal_bath_rho = finite.thermal_average_scale_indep(es, rho_baths, tau)
        matrix_print(thermal_bath_rho, "Bath density matrix:")

    excited_restrictions = basis.build_excited_restrictions()
    if verbosity >= 1:
        if verbosity >= 2:
            print("Restrictions when calculating the excited states:")
            for indices, occupations in excited_restrictions.items():
                print(f"---> {indices} : {occupations}")
            print()
        print(f"Consider {len(es):d} eigenstates for the spectra \n")
        print("Calculate Interacting Green's function...")

    gs_matsubara, gs_realaxis = get_Greens_function(
        matsubara_mesh=iw,
        omega_mesh=w,
        psis=psis,
        es=es,
        tau=tau,
        basis=basis,
        hOp=h,
        delta=delta,
        blocks=blocks,
        verbose=verbosity >= 2,
        reort=reort,
    )
    if gs_matsubara is not None:
        try:
            for gs in gs_matsubara:
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            if rank == 0:
                print(f"WARNING! Unphysical Matsubara-axis Greens function:\n\t{err}")
    if gs_realaxis is not None:
        try:
            for gs in gs_realaxis:
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            if rank == 0:
                print(f"WARNING! Unphysical real-axis Greens function:\n\t{err}")
    if verbosity >= 1:
        print("Calculate self-energy...", flush=True)
    if gs_realaxis is not None:
        sigma_real = get_sigma(
            omega_mesh=w,
            impurity_orbitals=basis.impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gs_realaxis,
            h0op=h0,
            h_bath=h_star_bath,
            v_full=v_star,
            delta=delta,
            clustername=cluster_label,
            blocks=blocks,
        )
        try:
            for sig in sigma_real:
                check_sigma(sig)
        except UnphysicalSelfenergyError as err:
            if rank == 0:
                print(f"WARNING! Unphysical realaxis selfenergy:\n\t{err}")
    else:
        sigma_real = None
    if gs_matsubara is not None:
        sigma = get_sigma(
            omega_mesh=iw,
            impurity_orbitals=basis.impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gs_matsubara,
            h0op=h0,
            h_bath=h_star_bath,
            v_full=v_star,
            delta=0,
            clustername=cluster_label,
            blocks=blocks,
        )
        try:
            for sig in sigma:
                check_sigma(sig)
        except UnphysicalSelfenergyError as err:
            if rank == 0:
                print(f"WARNING! Unphysical Matsubara axis selfenergy:\n\t{err}")
    else:
        sigma = None
    if verbosity >= 1:
        print("Calculating sig_static.")
    if rank == 0:
        sigma_static = get_Sigma_static(basis.impurity_orbitals, sum_bath_states, u4, es, local_psis, tau)
    else:
        sigma_static = 0
    if rank == 0:
        with h5.File("impurityModel_solver.h5", "a") as ar:
            it = 1
            if f"{cluster_label}/last_iteration" in ar:
                it = ar[f"{cluster_label}/last_iteration"][0] + 1
            else:
                ar.create_dataset(f"{cluster_label}/last_iteration", (1,), dtype=int)
            ar[f"{cluster_label}/last_iteration"][0] = it
            group = f"{cluster_label}/it_{it}"
            ar.create_dataset(f"{group}/tau", data=np.array([tau], dtype=float))
            ar.create_dataset(f"{group}/delta", data=np.array([delta], dtype=float))
            ar.create_dataset(f"{group}/gs_vecs", data=psis_dense)
            ar.create_dataset(f"{group}/gs_es", data=es)
            ar.create_dataset(f"{group}/iw", data=iw)
            ar.create_dataset(f"{group}/w", data=w)
            ar.create_dataset(f"{group}/rot_to_spherical", data=rot_to_spherical)
            ar.create_dataset(f"{group}/num_blocks", data=np.array([len(blocks)], dtype=int))
            ar.create_dataset(f"{group}/rho_imps", data=rho_imps, dtype=complex)
            ar.create_dataset(f"{group}/rho_baths", data=rho_baths, dtype=complex)

            for block_i, block in enumerate(blocks):
                ar.create_dataset(f"{group}/block_{block_i}/orbs", data=np.array(block, dtype=int))
                ar.create_dataset(f"{group}/block_{block_i}/gs_matsubara", data=gs_matsubara[block_i])
                ar.create_dataset(f"{group}/block_{block_i}/gs_real", data=gs_realaxis[block_i])
                ar.create_dataset(f"{group}/block_{block_i}/sigma_static", data=sigma_static[block_i])
                ar.create_dataset(f"{group}/block_{block_i}/sigma", data=sigma[block_i])
                ar.create_dataset(f"{group}/block_{block_i}/sigma_real", data=sigma_real[block_i])

    return sigma, sigma_real, sigma_static


def check_sigma(sigma: np.ndarray):
    """
    Verify that sigma makes physical sense.
    """
    diagonals = (np.diag(sigma[i, :, :]) for i in range(sigma.shape[0]))
    if np.any(np.imag(diagonals) > 0):
        raise UnphysicalSelfenergyError("Diagonal term has positive imaginary part.")


def check_greens_function(G):
    """
    Verify that G makes physical sense.
    """
    diagonals = (np.diag(G[i, :, :]) for i in range(G.shape[0]))
    if np.any(np.imag(diagonals) > 0):
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
    """
    n_imp = v.shape[1]
    hyb = np.zeros((n_imp, n_imp, len(ws)), dtype=complex)
    for bi, eb in enumerate(np.diagonal(hbath)):
        hyb += np.outer(np.conj(v[bi]), v[bi])[:, :, np.newaxis] * 1 / (ws + 1j * delta - eb)
    return np.moveaxis(hyb, -1, 0)


def get_sigma(
    omega_mesh,
    impurity_orbitals,
    nBaths,
    gs,
    h0op,
    h_bath,
    v_full,
    delta,
    blocks,
    clustername="",
):
    """
    Calculate self-energy from interacting Greens function and local hamiltonian.
    """
    hcorr, _, _, _ = get_hcorr_v_hbath(h0op, impurity_orbitals, nBaths)

    res = []
    for block, g in zip(blocks, gs):
        block_idx = np.ix_(block, block)
        wIs = (omega_mesh + 1j * delta)[:, np.newaxis, np.newaxis] * np.eye(len(block))[np.newaxis, :, :]
        g0_inv = wIs - hcorr[block_idx] - hyb(omega_mesh, v_full[:, block], h_bath, delta)
        res.append(g0_inv - np.linalg.inv(g))

    return res


def get_Sigma_static(n_impurity_orbitals, nBaths, U4, es, psis, tau):
    """
    Calculate the static (Hartree-Fock) self-energy.
    """
    n = sum(ni for ni in n_impurity_orbitals.values())
    rhos = [
        finite.build_impurity_density_matrix(
            sum(ni for ni in n_impurity_orbitals.values()), sum(nb for nb in nBaths.values()), psi
        )
        for psi in psis
    ]
    rho = thermal_average_scale_indep(es, rhos, tau)

    sigma_static = np.zeros((n, n), dtype=complex)
    for i, j in itertools.product(range(n), range(n)):
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
        iw=None,
        w=omega_mesh,
        delta=delta,
        nominal_occ=nominal_occ,
        delta_occ=delta_occ,
        num_bath_states=num_bath_states,
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
