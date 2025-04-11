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


def fixed_peak_dc(
    h0_op,
    N0,
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
    peak_position = max(peak_position, 4 * tau)
    valence_baths, conduction_baths = bath_states
    u = finite.getUop_from_rspt_u4(u4)
    dc_trial = dc_guess

    Np = {l: N0[l] + 1 for l in N0}
    Nm = {l: N0[l] - 1 for l in N0}
    if peak_position >= 0:
        basis_upper = CIPSI_Basis(
            impurity_orbitals=impurity_orbitals,
            bath_states=bath_states,
            nominal_impurity_occ=Np,
            truncation_threshold=1e5,
            verbose=False,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
        )
        basis_lower = CIPSI_Basis(
            impurity_orbitals=impurity_orbitals,
            bath_states=bath_states,
            nominal_impurity_occ=N0,
            truncation_threshold=1e5,
            verbose=False,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
        )
    else:
        basis_upper = CIPSI_Basis(
            impurity_orbitals=impurity_orbitals,
            bath_states=bath_states,
            nominal_impurity_occ=N0,
            truncation_threshold=1e5,
            verbose=False,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
        )
        basis_lower = CIPSI_Basis(
            impurity_orbitals=impurity_orbitals,
            bath_states=bath_states,
            nominal_impurity_occ=Nm,
            truncation_threshold=1e5,
            verbose=False,
            comm=MPI.COMM_WORLD,
            spin_flip_dj=spin_flip_dj,
        )
    h_op_i = finite.addOps([h0_op, u])

    dc_op_i = {
        ((i, "c"), (j, "a")): -dc_trial[i, j] + 0j
        for i in range(dc_trial.shape[0])
        for j in range(dc_trial.shape[1])
        if abs(dc_trial[i, j]) > 0
    }
    h_op = finite.addOps([h_op_i, dc_op_i])
    _ = basis_upper.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-3)
    _ = basis_lower.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-3)

    def F(dc_fac):
        dc = dc_fac * dc_trial
        dc_op_i = {
            ((i, "c"), (j, "a")): -dc[i, j] + 0j
            for i in range(dc_trial.shape[0])
            for j in range(dc_trial.shape[1])
            if abs(dc_trial[i, j]) > 0
        }
        h_op = finite.addOps([h_op_i, dc_op_i])
        _ = basis_upper.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-3)
        _ = basis_lower.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-3)

        h = basis_upper.build_sparse_matrix(h_op)
        e_upper, psi_upper = finite.eigensystem_new(
            h,
            e_max=0,
            k=2,
            eigenValueTol=1e-6,
            return_eigvecs=True,
            comm=basis_upper.comm,
            dense=basis_upper.size < dense_cutoff,
        )
        h = basis_lower.build_sparse_matrix(h_op)
        e_lower, psi_lower = finite.eigensystem_new(
            h,
            e_max=0,
            k=2,
            eigenValueTol=1e-6,
            return_eigvecs=True,
            comm=basis_upper.comm,
            dense=basis_lower.size < dense_cutoff,
        )
        psi_lower_local = basis_lower.build_state(psi_lower[:, 0].T)[0]
        psi_upper_local = basis_upper.build_state(psi_upper[:, 0].T)[0]
        psi_lowers = basis_lower.comm.allgather(psi_lower_local)
        psi_uppers = basis_upper.comm.allgather(psi_upper_local)
        psi_lower = {}
        psi_upper = {}
        for psi_lower_local, psi_upper_local in zip(psi_lowers, psi_uppers):
            for state in psi_lower_local:
                psi_lower[state] = psi_lower_local[state] + psi_lower.get(state, 0)
            for state in psi_upper_local:
                psi_upper[state] = psi_upper_local[state] + psi_upper.get(state, 0)
        rho_lower = finite.build_density_matrix(
            sorted([orb for blocks in basis_lower.impurity_orbitals.values() for block in blocks for orb in block]),
            psi_lower,
            basis_lower.num_spin_orbitals,
        )
        rho_upper = finite.build_density_matrix(
            sorted([orb for blocks in basis_upper.impurity_orbitals.values() for block in blocks for orb in block]),
            psi_upper,
            basis_upper.num_spin_orbitals,
        )
        avg_dc_lower = np.real(np.trace(rho_lower @ dc))
        avg_dc_upper = np.real(np.trace(rho_upper @ dc))
        if abs(avg_dc_upper - avg_dc_lower) < min(tau, 1e-2):
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


def calc_occ_e(
    h_op,
    N0,
    N_imp,
    bath_states,
    spin_flip_dj,
    dense_cutoff,
    comm,
    verbose,
    truncation_threshold,
):
    basis = CIPSI_Basis(
        impurity_orbitals=N_imp,
        H=h_op,
        bath_states=bath_states,
        delta_impurity_occ={i: 0 for i in N0},
        delta_valence_occ={i: 0 for i in N0},
        delta_conduction_occ={i: 0 for i in N0},
        nominal_impurity_occ=N0,
        truncation_threshold=truncation_threshold,
        verbose=verbose,
        spin_flip_dj=spin_flip_dj,
        comm=comm,
    )
    h_dict = basis.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-6)
    h = basis.build_sparse_matrix(h_op, h_dict)

    e_trial = finite.eigensystem_new(
        h,
        e_max=0,
        k=2,
        eigenValueTol=0,
        return_eigvecs=False,
        comm=basis.comm,
        dense=basis.size < dense_cutoff,
    )
    return e_trial[0], basis, h_dict


def find_gs(
    h_op,
    N0,
    bath_states,
    impurity_orbitals,
    rank,
    dense_cutoff,
    spin_flip_dj,
    comm,
    truncation_threshold,
    verbose,
):
    """
    Find the occupation corresponding to the lowest energy, compare N0 - 1, N0 and N0 + 1
    Returns:
    gs_impurity_occ, dict: Impurity occupation corresponding to the lowest energy.
    basis_gs, ManybodyBasis: Initial basis for the ground state
    h_dict_gs, dict: Memoized states for the hamiltonian operator.
    """
    (
        num_val_baths,
        num_cond_baths,
    ) = bath_states
    basis_gs = None
    gs_impurity_occ = N0.copy()
    dN_gs = dict.fromkeys(N0.keys(), 0)
    for i in N0:
        if verbose >= 2:
            print(f"{i=}")
        e_gs = np.inf
        for dN in [0, -1, 1]:
            e_trial, basis, h_dict = calc_occ_e(
                h_op,
                {j: N0[j] + dN for j in N0},
                impurity_orbitals,
                bath_states,
                spin_flip_dj,
                dense_cutoff,
                comm=comm,
                verbose=False,
                truncation_threshold=truncation_threshold,
            )
            if e_trial < e_gs:
                if verbose >= 2:
                    print(f"N0 {N0[i] + dN:^5d}: E = {e_trial:^7.4f} ")
                e_gs = e_trial
                basis_gs = basis
                h_dict_gs = h_dict.copy()
                dN_gs[i] = dN
                gs_impurity_occ[i] = N0[i] + dN
    for i in N0:
        while (
            dN_gs[i] != 0
            and all(imp_occ + dN_gs[j] > 0 for j, imp_occ in gs_impurity_occ.items())
            and all(
                imp_occ + dN_gs[j] <= sum(len(block) for block in impurity_orbitals[j])
                for j, imp_occ in gs_impurity_occ.items()
            )
        ):
            e_trial, basis, h_dict = calc_occ_e(
                h_op,
                {j: gs_impurity_occ[j] + dN_gs[j] for j in gs_impurity_occ},
                impurity_orbitals,
                bath_states,
                spin_flip_dj,
                dense_cutoff,
                comm=comm,
                verbose=False,
                truncation_threshold=truncation_threshold,
            )
            if e_trial >= e_gs:
                break
            e_gs = e_trial
            basis_gs = basis
            h_dict_gs = h_dict.copy()
            gs_impurity_occ[i] += dN_gs[i]
            if verbose >= 2:
                print(f"N0 {gs_impurity_occ[i]:^5d}: E = {e_gs:^7.4f} ")
    if verbose >= 1:
        print("Ground state occupation")
        print("\n".join((f"{i:^3d}: {gs_impurity_occ[i]: ^5d}" for i in gs_impurity_occ)))
        print(rf"E$_{{GS}}$ = {e_gs:^7.4f}")
        print("=" * 80)
    return gs_impurity_occ, basis_gs, h_dict_gs


def calc_selfenergy(
    h0,
    u4,
    iw,
    w,
    delta,
    nominal_occ,
    impurity_orbitals,
    bath_states,
    tau,
    verbosity,
    blocks,
    rot_to_spherical,
    cluster_label,
    reort,
    dense_cutoff,
    spin_flip_dj,
    comm,
    occ_restrict,
    chain_restrict,
    occ_cutoff,
    truncation_threshold,
    slaterWeightMin,
):
    """
    Calculate the self energy of the impurity.
    """
    # MPI variables
    rank = comm.rank

    valence_baths, conduction_baths = bath_states
    total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
    sum_bath_states = {
        i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
        for i in valence_baths
    }

    # construct local, interacting, hamiltonian
    u = finite.getUop_from_rspt_u4(u4)
    h = finite.addOps([h0, u])

    gs_impurity_occ, basis, h_dict = find_gs(
        h,
        nominal_occ,
        bath_states,
        impurity_orbitals,
        rank=rank,
        dense_cutoff=dense_cutoff,
        spin_flip_dj=spin_flip_dj,
        comm=comm,
        truncation_threshold=truncation_threshold,
        verbose=verbosity,
    )
    restrictions = basis.restrictions

    if restrictions is not None and verbosity >= 2:
        print("Restrictions GS on occupation")
        for indices, limits in restrictions.items():
            print(f"---> {sorted(indices)} : {limits}")

    energy_cut = -tau * np.log(1e-4)

    basis.tau = tau
    h_dict = basis.expand(h, H_dict=h_dict, dense_cutoff=dense_cutoff, de2_min=1e-6)
    h_gs = basis.build_sparse_matrix(h, h_dict)
    es, psis_dense = finite.eigensystem_new(
        h_gs,
        e_max=energy_cut,
        k=total_impurity_orbitals[0],
        eigenValueTol=0,
        comm=basis.comm,
        dense=basis.size < dense_cutoff,
    )
    psis = basis.build_state(psis_dense.T)
    if verbosity >= 1:
        print(f"{len(basis)} Slater determinants in the basis.")
    gs_stats = basis.get_state_statistics(psis)
    sum_bath_states = {
        i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
        for i in valence_baths
    }
    rho_imps, rho_baths, _ = basis.build_density_matrices(psis)
    n_orb = sum(len(block) for blocks in basis.impurity_orbitals.values() for block in blocks)
    full_rho_imps = np.zeros((len(psis), n_orb, n_orb), dtype=complex)
    for i, i_blocks in basis.impurity_orbitals.items():
        for k in range(len(psis)):
            for j, block_orbs in enumerate(i_blocks):
                idx = np.ix_([k], block_orbs, block_orbs)
                full_rho_imps[idx] = rho_imps[i][j][k]
    thermal_rho_imps = {
        i: [finite.thermal_average_scale_indep(es, block_rhos, tau) for block_rhos in rho_imps[i]]
        for i in basis.impurity_orbitals.keys()
    }
    thermal_rho_baths = {
        i: [finite.thermal_average_scale_indep(es, block_rhos, tau) for block_rhos in rho_baths[i]]
        for i in basis.impurity_orbitals.keys()
    }
    if verbosity >= 1:
        finite.printThermalExpValues_new(full_rho_imps, es, tau, rot_to_spherical)
        finite.printExpValues(full_rho_imps, es, rot_to_spherical)
        print("Occupation statistics for each eigenstate in the thermal ground state")
        print("Impurity, Valence, Conduction: Weight (|amp|^2)")
        for i, psi_stats in enumerate(gs_stats):
            print(f"{i}:")
            for imp_occ, val_occ, con_occ in sorted(psi_stats.keys()):
                print(f"{imp_occ:^8d},{val_occ:^8d},{con_occ:^11d}: {psi_stats[(imp_occ, val_occ, con_occ)]}")
            print("=" * 80)
            print()
        print("Ground state bath occupation statistics:")
        for i in basis.impurity_orbitals.keys():
            print(f"orbital set {i}:")
            for block_i, (imp_rho, bath_rho) in enumerate(zip(thermal_rho_imps[i], thermal_rho_baths[i])):
                print(f"Block {block_i} (impurity orbitals {basis.impurity_orbitals[i][block_i]})")
                matrix_print(imp_rho, "Impurity density matrix:")
                matrix_print(bath_rho, "Bath density matrix:")
                print("=" * 80)
            print()

    effective_restrictions = basis.get_effective_restrictions()
    if verbosity >= 1:
        print("Effective GS restrictions:")
        for indices, occupations in effective_restrictions.items():
            print(f"---> {sorted(indices)} : {occupations}")
        print("=" * 80)
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
        occ_restrict=occ_restrict,
        chain_restrict=chain_restrict,
        occ_cutoff=occ_cutoff,
        slaterWeightMin=slaterWeightMin,
    )
    if gs_matsubara is not None:
        try:
            for gs in gs_matsubara:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            if rank == 0:
                print(f"WARNING! Unphysical Matsubara-axis Greens function:\n\t{err}")
    if gs_realaxis is not None:
        try:
            for gs in gs_realaxis:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            if rank == 0:
                print(f"WARNING! Unphysical real-axis Greens function:\n\t{err}")
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
            impurity_orbitals=total_impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gs_matsubara,
            h0op=h0,
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
    sigma_static = get_Sigma_static(u4, full_rho_imps, es, tau)

    return {
        "sigma": sigma,
        "sigma_real": sigma_real,
        "sigma_static": sigma_static,
        "gs_matsubara": gs_matsubara,
        "gs_realaxis": gs_realaxis,
        "rho_imps": rho_imps,
        "rho_baths": rho_baths,
        "thermal_rho_imps": thermal_rho_imps,
        "thermal_rho_baths": thermal_rho_baths,
    }


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
        block_idx = np.ix_(block, block)
        wIs = (omega_mesh + 1j * delta)[:, np.newaxis, np.newaxis] * np.eye(len(block))[np.newaxis, :, :]
        g0_inv = wIs - hcorr[block_idx] - hyb(omega_mesh, v_full[:, block], h_bath, delta)
        res.append(g0_inv - np.linalg.inv(g))

    return res


def get_Sigma_static(U4, rho_imps, es, tau):
    """
    Calculate the static (Hartree-Fock) self-energy.
    """
    rho = finite.thermal_average_scale_indep(es, rho_imps, tau)
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
