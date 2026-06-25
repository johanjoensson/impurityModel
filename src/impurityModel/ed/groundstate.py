from itertools import product

import numpy as np

from impurityModel.ed.block_structure import BlockStructure, print_block_structure
from impurityModel.ed.density_matrix import calc_density_matrices
from impurityModel.ed.finite import (
    apply_casimir,
    apply_spin_correlation,
    bath_spin_pairs,
    casimir_to_quantum_number,
    impurity_spin_pairs,
    make_impurity_casimir_operators,
    make_spin_operators,
    manifold_observable_values,
    print_expectation_values,
    print_thermal_expectation_values,
    spin_pairs_consistent_with_h,
    thermal_average_scale_indep,
    thermal_observable_value,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.symmetries import (
    conserved_subset_charges,
    measure_conserved_charges,
    restrictions_from_charges,
)
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
    cipsi_solver_method="trlm",
    reort="full",
    return_state=False,
    weighted_restrictions=None,
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

    from impurityModel.ed.cipsi_solver import CIPSISolver
    from impurityModel.ed.manybody_basis import Basis

    basis = Basis(
        impurity_indices,
        bath_states,
        delta_impurity_occ=dict.fromkeys(N0, 0),
        delta_valence_occ=dict.fromkeys(N0, 0),
        delta_conduction_occ=dict.fromkeys(N0, 0),
        nominal_impurity_occ=N0,
        mixed_valence=mixed_valence,
        tau=tau,
        chain_restrict=chain_restrict,
        truncation_threshold=truncation_threshold,
        verbose=verbose,
        spin_flip_dj=spin_flip_dj,
        comm=comm,
        weighted_restrictions=weighted_restrictions,
    )
    solver = CIPSISolver(basis)
    solver.truncate_initial(h_op)

    basis.restrictions = basis.build_excited_restrictions(h_op, psis=None, es=None)
    if len(basis) == 0:
        return (np.inf, basis, None) if return_state else (np.inf, basis)
    solver.expand(
        h_op,
        dense_cutoff=dense_cutoff,
        de2_min=1e-4,
        slaterWeightMin=slaterWeightMin,
        solver=cipsi_solver_method,
        reort=reort,
    )

    energy_cut = -tau * np.log(1e-4)

    es, eigen_psis = solver.get_eigenvectors(
        h_op,
        num_wanted=10,
        max_energy=energy_cut,
        dense_cutoff=dense_cutoff,
        slaterWeightMin=slaterWeightMin,
        solver=cipsi_solver_method,
        reort=reort,
    )
    gs_state = eigen_psis[int(np.argmin(es))] if return_state and len(eigen_psis) > 0 else None
    basis.clear()
    basis.add_states(set(state for psi in eigen_psis for state in psi))
    if return_state:
        return np.min(es), basis, gs_state
    return np.min(es), basis


def prescan_ground_state_sector(
    h_op,
    impurity_orbitals,
    bath_states,
    N0,
    mixed_valence,
    tau,
    chain_restrict,
    spin_flip_dj,
    dense_cutoff,
    comm,
    truncation_threshold,
    slaterWeightMin,
    cipsi_solver_method="trlm",
    scan_width=1,
    verbose=False,
    weighted_restrictions=None,
):
    """Rough CIPSI over a broad occupation window to locate the ground-state sector.

    Instead of running a full accurate solve for every candidate impurity occupation
    (the ``O(3^k)`` ``dN`` scan in :func:`find_ground_state_basis`), this runs **one**
    rough solve over a window wide enough to span the candidates and then *measures*:

    - the rough ground state's **impurity occupation** per orbital set
      (:math:`\\langle N_{\\mathrm{imp}}\\rangle`, rounded) — the winning nominal ``N0``
      to seed the accurate basis; and
    - its **conserved-charge sector** (:func:`conserved_subset_charges` +
      :func:`measure_conserved_charges`) — used to restrict the accurate refine to the
      correct sector (Phase 3).

    Parameters
    ----------
    h_op, impurity_orbitals, bath_states, N0, mixed_valence, tau, chain_restrict,
    spin_flip_dj, dense_cutoff, comm, truncation_threshold, slaterWeightMin,
    cipsi_solver_method
        As for :func:`calc_energy` / :func:`find_ground_state_basis`.
    scan_width : int, default 1
        How far (in electrons per orbital set) to widen the valence window beyond
        ``mixed_valence`` for the scan, so the candidate sectors are spanned.
    verbose : bool, default False

    Returns
    -------
    winning_N0 : dict
        Nominal impurity occupation per orbital set (rounded rough-GS occupation).
    restrictions : dict of frozenset to (int, int)
        Conserved-charge sector restrictions for the accurate refine, or ``None`` if the
        scan produced no state.
    energy : float
        The rough ground-state energy.
    """
    prescan_mixed_valence = {i: abs(mixed_valence.get(i, 0)) + scan_width for i in N0}
    energy, basis, gs_state = calc_energy(
        h_op,
        impurity_orbitals,
        bath_states,
        N0,
        prescan_mixed_valence,
        tau,
        chain_restrict,
        spin_flip_dj,
        dense_cutoff,
        comm,
        verbose,
        truncation_threshold,
        slaterWeightMin,
        cipsi_solver_method=cipsi_solver_method,
        weighted_restrictions=weighted_restrictions,
        return_state=True,
    )
    if gs_state is None:
        return dict(N0), None, energy

    n_orb = basis.num_spin_orbitals
    impurity_subsets = [frozenset(orb for block in impurity_orbitals[i] for orb in block) for i in N0]
    impurity_occ = measure_conserved_charges(gs_state, impurity_subsets, n_orb, comm=comm)
    winning_N0 = {i: impurity_occ[k] for k, i in enumerate(N0)}

    charges = conserved_subset_charges(h_op, n_orb)
    charge_occ = measure_conserved_charges(gs_state, charges, n_orb, comm=comm)
    restrictions = restrictions_from_charges(charges, charge_occ)
    return winning_N0, restrictions, energy


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
    slaterWeightMin=1e-12,
    cipsi_solver_method="trlm",
    use_prescan=True,
    weighted_restrictions=None,
):
    """
    Find the occupation corresponding to the lowest energy, compare N0 - 1, N0 and N0 + 1

    use_prescan (default True): locate the ground-state occupation with a single rough
    CIPSI over a broad window and measure the impurity occupation, instead of the
    O(3^k) accurate scan over every dN combination. Set False for the legacy scan.

    Returns:
    basis_gs, ManybodyBasis: Initial basis for the ground state
    """
    if mixed_valence is None or mixed_valence is False:
        mixed_valence = dict.fromkeys(N0, 0)
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
            cipsi_solver_method=cipsi_solver_method,
            weighted_restrictions=weighted_restrictions,
        )
        energy_cache[key] = (e_trial, basis.copy() if basis is not None else None)
        return e_trial, basis

    keys = list(N0.keys())
    if use_prescan:
        # One rough CIPSI over a broad window locates the GS sector by *measuring* the
        # ground state's impurity occupation, replacing the O(3^k) dN scan; then a single
        # accurate solve at that occupation refines it.
        winning_N0, _restrictions, _e_rough = prescan_ground_state_sector(
            h_op,
            impurity_orbitals,
            bath_states,
            N0,
            mixed_valence,
            tau,
            chain_restrict,
            spin_flip_dj,
            dense_cutoff,
            comm,
            truncation_threshold,
            slaterWeightMin,
            cipsi_solver_method=cipsi_solver_method,
            verbose=verbose,
            weighted_restrictions=weighted_restrictions,
        )
        winning_N0 = {i: N0[i] if i in frozen_occupations else winning_N0[i] for i in N0}
        e_gs, basis = get_energy(winning_N0)
        gs_impurity_occ = winning_N0
        basis_gs = basis.copy() if basis is not None else None
        if verbose and (comm is None or comm.rank == 0):
            print("Pre-scan ground state occupation:", gs_impurity_occ, f"~ {e_gs:6.3f}")
    else:
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
    cipsi_solver_method="trlm",
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
        cipsi_solver_method=cipsi_solver_method,
        **basis_setup,
    )

    # if ground_state_basis.restrictions is not None:
    # Hop.set_restrictions(ground_state_basis.restrictions)
    ground_state_basis.tau = tau
    energy_cut = -tau * np.log(1e-4)
    from impurityModel.ed.cipsi_solver import CIPSISolver

    solver = CIPSISolver(ground_state_basis)
    solver.expand(
        Hop, dense_cutoff=dense_cutoff, de2_min=1e-6, slaterWeightMin=slaterWeightMin, solver=cipsi_solver_method
    )
    es, psis = solver.get_eigenvectors(
        Hop,
        num_wanted=10,
        max_energy=energy_cut,
        dense_cutoff=dense_cutoff,
        slaterWeightMin=slaterWeightMin,
        solver=cipsi_solver_method,
    )
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
        # Impurity S^2 / L^2 / J^2: two-body observables, so they need the actual
        # eigenstates (gathered on rank 0). The L/S/J operators are built in the
        # spherical basis and rotated to the computational basis via rot_to_spherical
        # (symmetry-plan Phase 5), which makes the ml-dependence of L correct and is
        # robust to the computational spin layout. Computed on rank 0 only and passed
        # in; None on other ranks leaves the output unchanged there.
        s_values = l_values = j_values = None
        s2_thermal = l2_thermal = j2_thermal = None
        if rank == 0:
            try:
                l_ops, s_ops, j_ops = make_impurity_casimir_operators(
                    ground_state_basis.impurity_orbitals, rot_to_spherical
                )
            except ValueError:
                # Impurity is not a clean spin-doubled l-shell: skip the Casimirs
                # (the rho-based <L.S>/<Lz>/<Sz> etc. still print).
                l_ops = None
            if l_ops is not None:
                casimir = {}
                for name, ops in (("S", s_ops), ("L", l_ops), ("J", j_ops)):
                    vals = manifold_observable_values(full_psis, es, lambda psi, _ops=ops: apply_casimir(psi, *_ops))
                    casimir[name] = (
                        np.array([casimir_to_quantum_number(v) for v in vals]),
                        thermal_observable_value(vals, es, tau),
                    )
                s_values, s2_thermal = casimir["S"]
                l_values, l2_thermal = casimir["L"]
                j_values, j2_thermal = casimir["J"]
        # Kondo impurity-bath spin correlation <S_imp . S_bath>. The bath spin pairing
        # follows the down-then-up convention, but is only trusted if the induced global
        # spin operators commute with the one-body Hamiltonian (so the spin assignment
        # is consistent with the model's spin symmetry); otherwise (SOC, non-standard
        # ordering) it is skipped rather than reported wrong.
        sisb_values = None
        sisb_thermal = None
        if rank == 0:
            imp_pairs = impurity_spin_pairs(ground_state_basis.impurity_orbitals)
            bath_pairs = bath_spin_pairs(ground_state_basis.bath_states)
            n_orb = ground_state_basis.num_spin_orbitals
            if imp_pairs and bath_pairs and spin_pairs_consistent_with_h(Hop, imp_pairs + bath_pairs, n_orb):
                ops_imp = make_spin_operators(imp_pairs)
                ops_bath = make_spin_operators(bath_pairs)
                sisb_raw = manifold_observable_values(
                    full_psis, es, lambda psi: apply_spin_correlation(psi, ops_imp, ops_bath)
                )
                sisb_values = np.real(sisb_raw)
                sisb_thermal = thermal_observable_value(sisb_raw, es, tau)
        print_thermal_expectation_values(
            thermal_rho[impurity_ix],
            e_avg,
            rot_to_spherical,
            block_structure,
            s_thermal=s2_thermal,
            l_thermal=l2_thermal,
            j_thermal=j2_thermal,
            sisb_thermal=sisb_thermal,
        )
        impurity_ix = np.ix_(np.arange(len(rhos)), impurity_indices, impurity_indices)
        print_expectation_values(
            rhos[impurity_ix],
            es,
            rot_to_spherical,
            block_structure,
            s_values=s_values,
            l_values=l_values,
            j_values=j_values,
            sisb_values=sisb_values,
        )
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
