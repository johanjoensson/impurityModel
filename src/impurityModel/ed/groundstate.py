from itertools import product, compress
import numpy as np
from mpi4py import MPI
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator
from impurityModel.ed.manybody_basis import Basis
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
    slaterWeightMin,
):
    """
    Calculate the ground-state energy of the system for a given charge sector.

    This function initializes a CIPSI basis for a nominal occupation config `N0`,
    expands the basis variationally using the Hamiltonian `h_op`, constructs the
    sparse Hamiltonian matrix, solves the eigensystem to obtain the lowest
    eigen-energies and states within a threshold of the ground state, and returns
    the minimum eigenvalue along with the optimized basis.

    Parameters
    ----------
    h_op : ManyBodyOperator
        The Hamiltonian operator of the system.
    impurity_indices : dict
        Mapping of orbital set indices to impurity orbital indices.
    bath_states : tuple of dicts
        Valence and conduction bath states coupled to the impurity.
    N0 : dict
        Nominal impurity orbital occupations.
    mixed_valence : dict
        The mixed valence occupation bounds per orbital set.
    tau : float
        Characteristic energy scale used for basis selection (temperature scale).
    chain_restrict : bool
        If True, restricts the basis to states generated along hopping chains.
    spin_flip_dj : bool
        If True, enables spin flip basis excitation configurations.
    dense_cutoff : int
        Dimension threshold below which a dense eigensolver is used.
    comm : MPI.Comm or None
        MPI communicator for distributed calculation.
    verbose : bool
        If True, prints progress details.
    truncation_threshold : int
        Maximum basis size allowed during initialization and expansion.
    slaterWeightMin : float
        Minimum weight (|amplitude|^2) below which Slater determinants are pruned.

    Returns
    -------
    energy : float
        The lowest eigenvalue (ground state energy) found for this charge sector.
    basis : Basis
        The optimized many-body basis.
    """

    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.ed.cipsi_solver import CIPSISolver

    basis = Basis(
        impurity_indices,
        bath_states,
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
    solver = CIPSISolver(basis)
    solver.truncate_initial(h_op)

    basis.restrictions = basis.build_excited_restrictions(h_op, psis=None, es=None)
    if len(basis) == 0:
        return np.inf, basis
    solver.expand(h_op, dense_cutoff=dense_cutoff, de2_min=1e-4, slaterWeightMin=slaterWeightMin)

    energy_cut = -tau * np.log(1e-4)

    h = basis.build_sparse_matrix(h_op)
    es, eigvecs = eigensystem(
        h,
        e_max=energy_cut,
        k=10,
        eigenValueTol=slaterWeightMin,
        return_eigvecs=True,
        comm=basis.comm,
        dense=basis.size < dense_cutoff,
    )
    eigen_psis = basis.build_state(eigvecs.T, slaterWeightMin=slaterWeightMin)
    basis.clear()
    basis.add_states(set(state for psi in eigen_psis for state in psi))
    return np.min(es), basis


def find_ground_state_basis(
    h_op,
    impurity_orbitals,
    bath_states,
    N0,
    frozen_occupations=None,
    mixed_valence=False,
    tau=0.01,
    chain_restrict=False,
    rank=0,
    dense_cutoff=1000,
    spin_flip_dj=True,
    comm=None,
    truncation_threshold=1000000,
    verbose=True,
    slaterWeightMin=0,
):
    """
    Find the occupation corresponding to the lowest energy, compare N0 - 1, N0 and N0 + 1
    Returns:
    basis_gs, ManybodyBasis: Initial basis for the ground state
    """
    if mixed_valence is None or mixed_valence is False:
        mixed_valence = {i: 0 for i in N0}
    (
        num_val_baths,
        num_cond_baths,
    ) = bath_states
    if frozen_occupations is None:
        frozen_occupations = set()
    basis_gs = None
    gs_impurity_occ = N0.copy()
    dN_gs = dict.fromkeys(N0.keys(), 0)

    energy_cache = {}

    def get_energy(trial_N0):
        """
        Helper function to calculate, cache, and return the energy and basis for a trial N0.

        Parameters
        ----------
        trial_N0 : dict
            The trial nominal occupations for each orbital set.

        Returns
        -------
        energy : float
            The ground state energy.
        basis : Basis
            The optimized many-body basis.
        """

        key = tuple(sorted(trial_N0.items()))
        if key in energy_cache:
            e_trial, basis = energy_cache[key]
            return e_trial, (basis.copy() if basis is not None else None)

        # Check bounds: 0 <= occupation <= max possible orbitals
        for orbital_idx, occ in trial_N0.items():
            max_occ = sum(len(block) for block in impurity_orbitals[orbital_idx])
            if occ < 0 or occ > max_occ:
                energy_cache[key] = (np.inf, None)
                return np.inf, None

        e_trial, basis = calc_energy(
            h_op,
            impurity_orbitals,
            bath_states,
            trial_N0,
            mixed_valence,
            tau,
            chain_restrict,
            spin_flip_dj,
            dense_cutoff,
            comm=comm,
            verbose=verbose,
            truncation_threshold=truncation_threshold,
            slaterWeightMin=slaterWeightMin,
        )
        # energy_cache[key] = (e_trial, basis.copy() if basis is not None else None)
        return e_trial, basis

    keys = list(N0.keys())
    dN_trials = [
        {keys[i]: dN[i] if keys[i] not in frozen_occupations else 0 for i in range(len(keys))}
        for dN in product([0, -1, 1], repeat=len(keys))
    ]
    e_gs = np.inf
    for dN in dN_trials:
        trial_N0 = {i: N0[i] + dN[i] for i in N0}
        e_trial, basis = get_energy(trial_N0)
        if verbose:
            print("{" + " ".join(f" {i} : {trial_N0[i]}" for i in dN) + f"}} ~ {e_trial:6.3f}")
        if e_trial < e_gs:
            e_gs = e_trial
            basis_gs = basis.copy()
            dN_gs = dN
            gs_impurity_occ = trial_N0
    for i in N0:
        while (
            dN_gs[i] != 0
            and all(imp_occ + dN_gs[j] > 0 for j, imp_occ in gs_impurity_occ.items())
            and all(
                imp_occ + dN_gs[j] <= sum(len(block) for block in impurity_orbitals[j])
                for j, imp_occ in gs_impurity_occ.items()
            )
        ):
            trial_N0 = {j: n + dN_gs[i] if i == j else n for j, n in gs_impurity_occ.items()}
            e_trial, basis = get_energy(trial_N0)
            if verbose:
                print(
                    "{" + " ".join(f" {j} : {trial_N0[j]}" for j in dN_gs) + f"}} ~ {e_trial:6.3f}",
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
    # Explicitly clear the energy_cache to break the closure reference cycle.
    # get_energy captures energy_cache (closure), which holds Basis objects whose
    # .comm may be a split MPI communicator.  Without this, the cycle cannot be
    # freed by CPython's reference-counting and survives until Python shutdown,
    # where MPI has already been finalised -> segfault.
    for _cached_e, _cached_basis in energy_cache.values():
        if _cached_basis is not None:
            _cached_basis.comm = None
    energy_cache.clear()
    return basis_gs


def calc_gs(
    Hop: ManyBodyOperator,
    basis_setup: dict,
    block_structure: BlockStructure,
    rot_to_spherical: np.ndarray,
    verbose: bool,
    slaterWeightMin=0,
    **kwargs,
):
    """
    Calculate the ground-state wavefunction, eigen-energies, and density matrices.

    This function determines the ground-state charge sector, optimizes the
    variational many-body basis, solves the eigensystem for the low-energy
    states, and computes the thermally-averaged density matrix and expectation
    values.

    Parameters
    ----------
    Hop : ManyBodyOperator
        The Hamiltonian operator.
    basis_setup : dict
        Configuration dictionary containing parameters for the basis setup,
        such as 'impurity_orbitals', 'bath_states', 'nominal_impurity_occ', etc.
    block_structure : BlockStructure
        The block structure defining mapping and symmetry relationships.
    rot_to_spherical : ndarray
        Transformation matrix from local to spherical harmonics.
    verbose : bool
        If True, prints detailed statistics and expectation values.
    slaterWeightMin : float
        Minimum weight threshold for determinants in the basis.

    Returns
    -------
    psis : list of ManyBodyState
        The low-energy eigenstates.
    es : ndarray
        The corresponding eigen-energies.
    ground_state_basis : Basis
        The optimized many-body basis.
    thermal_rho : ndarray
        The thermally-averaged density matrix.
    gs_info : dict
        A dictionary containing additional ground-state info (e.g. 'rhos' list).
    """

    basis_setup = dict(basis_setup)
    if "impurity_orbital" in basis_setup:
        basis_setup["impurity_orbitals"] = basis_setup.pop("impurity_orbital")
    if "nominal_impurity_occ" in basis_setup:
        basis_setup["N0"] = basis_setup.pop("nominal_impurity_occ")

    tau = basis_setup["tau"]
    basis_setup["tau"] /= 100
    dense_cutoff = basis_setup.get("dense_cutoff", 1000)
    ground_state_basis = find_ground_state_basis(
        Hop,
        verbose=verbose,
        slaterWeightMin=np.sqrt(slaterWeightMin),
        **basis_setup,
    )

    # if ground_state_basis.restrictions is not None:
    # Hop.set_restrictions(ground_state_basis.restrictions)
    ground_state_basis.tau = tau
    energy_cut = -tau * np.log(1e-4)
    from impurityModel.ed.cipsi_solver import CIPSISolver
    solver = CIPSISolver(ground_state_basis)
    solver.expand(Hop, dense_cutoff=dense_cutoff, de2_min=1e-6, slaterWeightMin=slaterWeightMin)
    h_gs = ground_state_basis.build_sparse_matrix(Hop)
    es, psis_dense = eigensystem(
        h_gs,
        e_max=energy_cut,
        k=10,
        eigenValueTol=0,
        comm=ground_state_basis.comm,
        dense=ground_state_basis.size < dense_cutoff,
    )

    psis = ground_state_basis.build_state(psis_dense.T, slaterWeightMin=slaterWeightMin)
    ground_state_basis.clear()
    ground_state_basis.add_states(set(state for p in psis for state in p))
    psis = ground_state_basis.redistribute_psis(psis)

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
