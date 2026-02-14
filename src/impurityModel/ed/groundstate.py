from itertools import product, compress
import numpy as np
from mpi4py import MPI
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator
from impurityModel.ed.manybody_basis import CIPSI_Basis
from impurityModel.ed.finite import (
    eigensystem_new as eigensystem,
    thermal_average_scale_indep,
    print_thermal_expectation_values,
    print_expectation_values,
)
from impurityModel.ed.density_matrix import calc_density_matrices
from impurityModel.ed.block_structure import BlockStructure, print_block_structure
from impurityModel.ed.utils import matrix_print


def calc_energy(
    h_op,
    impurity_indices,
    bath_states,
    N0,
    mixed_valence,
    tau,
    chain_restrict,
    spin_flip_dj,
    dense_cutoff,
    comm,
    verbose,
    truncation_threshold,
):
    basis = CIPSI_Basis(
        impurity_indices,
        bath_states,
        H=h_op,
        delta_impurity_occ={i: 0 for i in N0},
        delta_valence_occ={i: 0 for i in N0},
        delta_conduction_occ={i: 0 for i in N0},
        nominal_impurity_occ=N0,
        mixed_valence=mixed_valence,
        tau=tau,
        chain_restrict=chain_restrict,
        truncation_threshold=truncation_threshold,
        verbose=verbose,
        spin_flip_dj=spin_flip_dj,
        comm=comm,
    )
    if len(basis) == 0:
        return np.inf, basis, {}
    _ = basis.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-4, slaterWeightMin=np.sqrt(np.finfo(float).eps))

    energy_cut = -tau * np.log(1e-4)

    # _, block_roots, _, _, block_basis, _, _ = basis.split_into_block_basis_and_redistribute_psi(h_op, None)
    h = basis.build_sparse_matrix(h_op)
    es = eigensystem(
        h,
        e_max=energy_cut,
        k=10,
        eigenValueTol=0,  # np.finfo(float).eps,
        return_eigvecs=False,
        comm=basis.comm,
        dense=basis.size < dense_cutoff,
    )
    # e_trial = basis.comm.allreduce(np.min(e_block), op=MPI.MIN)
    return np.min(es), basis


def find_ground_state_basis(
    h_op,
    impurity_orbitals,
    bath_states,
    N0,
    mixed_valence,
    tau,
    chain_restrict,
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
    basis_gs, ManybodyBasis: Initial basis for the ground state
    """
    (
        num_val_baths,
        num_cond_baths,
    ) = bath_states
    basis_gs = None
    gs_impurity_occ = N0.copy()
    dN_gs = dict.fromkeys(N0.keys(), 0)

    keys = list(N0.keys())
    dN_trials = [{keys[i]: dN[i] for i in range(len(keys))} for dN in product([0, -1, 1], repeat=len(keys))]
    e_gs = np.inf
    for dN in dN_trials:
        e_trial, basis = calc_energy(
            h_op,
            impurity_orbitals,
            bath_states,
            {i: N0[i] + dN[i] for i in N0},
            mixed_valence,
            tau,
            chain_restrict,
            spin_flip_dj,
            dense_cutoff,
            comm=comm,
            verbose=verbose,
            truncation_threshold=truncation_threshold,
        )
        if verbose:
            print("{" + " ".join(f" {i} : {N0[i] + dN[i]}" for i in dN) + f"}} ~ {e_trial}")
        if e_trial < e_gs:
            e_gs = e_trial
            basis_gs = basis.copy()
            dN_gs = dN
            gs_impurity_occ = {i: N0[i] + dN[i] for i in N0}
    for i in N0:
        while (
            dN_gs[i] != 0
            and all(imp_occ + dN_gs[j] > 0 for j, imp_occ in gs_impurity_occ.items())
            and all(
                imp_occ + dN_gs[j] <= sum(len(block) for block in impurity_orbitals[j])
                for j, imp_occ in gs_impurity_occ.items()
            )
        ):
            e_trial, basis = calc_energy(
                h_op,
                impurity_orbitals,
                bath_states,
                {j: n + dN_gs[i] if i == j else n for j, n in gs_impurity_occ.items()},
                mixed_valence,
                tau,
                chain_restrict,
                spin_flip_dj,
                dense_cutoff,
                comm=comm,
                verbose=True,
                truncation_threshold=truncation_threshold,
            )
            if verbose:
                print(
                    "{" + " ".join(f" {i} : {gs_impurity_occ[i] + dN_gs[i]}" for i in dN_gs) + f"}} ~ {e_trial}",
                )
            if e_trial >= e_gs:
                break
            gs_impurity_occ[i] += dN_gs[i]
            e_gs = e_trial
            basis_gs = basis.copy()
    if verbose:
        print("Ground state occupation")
        print("\n".join((f"{i:^3d}: {gs_impurity_occ[i]: ^5d}" for i in gs_impurity_occ)))
        print(rf"E$_{{GS}}$ = {e_gs:^7.4f}")
        print("=" * 80)
    return basis_gs


def calc_gs(
    Hop: ManyBodyOperator,
    basis_setup: dict,
    block_structure: BlockStructure,
    rot_to_spherical: np.ndarray,
    verbose: bool,
    **kwargs,
):

    tau = basis_setup["tau"]
    basis_setup["tau"] /= 100
    dense_cutoff = basis_setup["dense_cutoff"]
    ground_state_basis = find_ground_state_basis(
        Hop,
        verbose=verbose,
        **basis_setup,
    )

    ground_state_basis.tau = tau
    energy_cut = -tau * np.log(1e-4)
    _ = ground_state_basis.expand(Hop, dense_cutoff=dense_cutoff, de2_min=1e-6, slaterWeightMin=np.finfo(float).eps)
    _, block_roots, block_color, _, block_basis, _, _ = ground_state_basis.split_into_block_basis_and_redistribute_psi(
        Hop, None
    )
    h_gs = block_basis.build_sparse_matrix(Hop)
    block_es, block_psis_dense = eigensystem(
        h_gs,
        e_max=energy_cut,
        k=10,
        eigenValueTol=0,
        comm=block_basis.comm,
        dense=block_basis.size < dense_cutoff,
    )
    psis = []
    es = np.array([], dtype=float)
    for c, c_root in enumerate(block_roots):
        es_c = ground_state_basis.comm.bcast(block_es, root=c_root)
        es = np.append(es, es_c)
        if c != block_color:
            psi_c = ground_state_basis.redistribute_psis([ManyBodyState() for _ in es_c])
        else:
            psi_c = ground_state_basis.redistribute_psis(block_basis.build_state(block_psis_dense.T, slaterWeightMin=0))
        psis.extend(psi_c)

    sort_idx = np.argsort(es)
    es = es[sort_idx]
    mask = es <= (es[0] + energy_cut)
    es = es[mask]
    psis = [psis[idx] for idx in compress(sort_idx, mask)]

    effective_restrictions = ground_state_basis.get_effective_restrictions()
    if verbose:
        print("Effective GS restrictions:")
        for indices, occupations in effective_restrictions.items():
            print(f"---> {sorted(indices)} : {occupations}")
        print("=" * 80)
        print(f"{len(ground_state_basis)} Slater determinants in the basis.")
    gs_stats = ground_state_basis.get_state_statistics(psis)

    comm = ground_state_basis.comm
    rank = comm.rank if comm is not None else 0
    if ground_state_basis.is_distributed:
        gathered_psis = ground_state_basis.comm.gather(psis, root=0)
        if rank == 0:
            full_psis = [ManyBodyState() for _ in psis]
            for r_psis in gathered_psis:
                for psi_i, r_psi_i in zip(full_psis, r_psis):
                    psi_i += r_psi_i
    else:
        full_psis = psis
    if rank == 0:
        rhos = calc_density_matrices(full_psis, list(range(ground_state_basis.num_spin_orbitals)))
    else:
        rhos = np.empty(
            (len(psis), ground_state_basis.num_spin_orbitals, ground_state_basis.num_spin_orbitals), dtype=complex
        )
    if ground_state_basis.is_distributed:
        ground_state_basis.comm.Bcast(rhos, root=0)

    e_avg = thermal_average_scale_indep(es, es, tau)
    thermal_rho = thermal_average_scale_indep(es, rhos, tau)
    if verbose:
        impurity_indices = [
            orb
            for impurity_orbital_blocks in ground_state_basis.impurity_orbitals.values()
            for block in impurity_orbital_blocks
            for orb in block
        ]
        print(f"{impurity_indices=}")
        impurity_ix = np.ix_(impurity_indices, impurity_indices)
        print("Block structure")
        print_block_structure(block_structure)
        print_thermal_expectation_values(thermal_rho[impurity_ix], e_avg, rot_to_spherical, block_structure)
        impurity_ix = np.ix_(np.arange(len(rhos)), impurity_indices, impurity_indices)
        print_expectation_values(rhos[impurity_ix], es, rot_to_spherical, block_structure)
        print("Occupation statistics for each eigenstate in the thermal ground state")
        print("Impurity, Valence, Conduction: Weight (|amp|^2)")
        for i, psi_stats in enumerate(gs_stats):
            print(f"{i}:")
            for imp_occ, val_occ, con_occ in sorted(psi_stats.keys()):
                print(f"{imp_occ:^8d},{val_occ:^8d},{con_occ:^11d}: {psi_stats[(imp_occ, val_occ, con_occ)]}")
            print("=" * 80)
            print()
        print("Ground state bath occupation statistics:")
        valence_bath_states, conduction_bath_states = ground_state_basis.bath_states
        for i in ground_state_basis.impurity_orbitals.keys():
            print(f"orbital set {i}:")
            impurity_orbital_blocks = ground_state_basis.impurity_orbitals[i]
            valence_bath_orbital_blocks = valence_bath_states[i]
            conduction_bath_orbital_blocks = conduction_bath_states[i]
            for block_i, (imp_orbs, val_orbs, con_orbs) in enumerate(
                zip(impurity_orbital_blocks, valence_bath_orbital_blocks, conduction_bath_orbital_blocks)
            ):
                print(f"Block {block_i} (impurity orbitals {imp_orbs})")
                print(f"Block {block_i} (valence orbitals {val_orbs})")
                print(f"Block {block_i} (conduction orbitals {con_orbs})")
                impurity_ix = np.ix_(imp_orbs, imp_orbs)
                bath_ix = np.ix_(val_orbs + con_orbs, val_orbs + con_orbs)
                matrix_print(thermal_rho[impurity_ix], "Impurity density matrix:")
                matrix_print(thermal_rho[bath_ix], "Bath density matrix:")
                print("=" * 80)
            print("", flush=verbose >= 2)
        print()
    return psis, es, ground_state_basis, thermal_rho, {"rhos": rhos}
