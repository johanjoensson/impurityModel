from itertools import product

import numpy as np

from impurityModel.ed.basis_restrictions import build_excited_restrictions, get_effective_restrictions
from impurityModel.ed.block_structure import BlockStructure, print_block_structure
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.hartree_fock import hartree_fock_occupation
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.memory_estimate import log_memory_budget, suggest_truncation_threshold
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.spin_pairs import (
    bath_spin_pairs,
    derive_spin_pairs,
    impurity_spin_pairs,
    spin_pairs_consistent_with_h,
)
from impurityModel.ed.observables import (
    apply_casimir,
    apply_spin_correlation,
    casimir_to_quantum_number,
    make_impurity_casimir_operators,
    make_spin_operators,
    manifold_observable_values,
    print_expectation_values,
    print_thermal_expectation_values,
    thermal_observable_value,
)
from impurityModel.ed.gs_statistics import compute_gs_statistics, print_gs_statistics, save_gs_statistics
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.utils import matrix_print
from impurityModel.ed.basis_transcription import build_density_matrices


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
    truncation_threshold : int or float
        Global cap on the number of Slater determinants in the basis; on overflow the CIPSI
        solver keeps only the determinants with the largest eigenvector amplitudes
        (``np.inf`` disables capping).
    slaterWeightMin : float
        Minimum weight (``|amplitude|^2``) below which Slater determinants are pruned.

    Returns
    -------
    energy : float
        The lowest eigenvalue (ground state energy) found for this charge sector.
    basis : Basis
        The optimized many-body basis.
    """

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

    basis.restrictions = build_excited_restrictions(basis, h_op, psis=None, es=None)
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


def hartree_fock_seed_occupation(h_op, impurity_orbitals, bath_states, N0, comm=None, verbose=False):
    """Nominal impurity occupation ``N0`` from a cheap unrestricted Hartree-Fock solve.

    This is the default seed for :func:`find_ground_state_basis`. Instead of running an
    accurate solve for every candidate impurity occupation (the legacy ``O(3^k)`` ``dN``
    scan) — or a rough many-body CIPSI over a *broadened* occupation window, which for long
    bath chains can build a massive basis and exhaust memory — it solves the problem at
    mean-field level (:func:`impurityModel.ed.hartree_fock.hartree_fock_occupation`). The
    unrestricted-HF single determinant variationally minimises the mean-field energy, so
    **its impurity occupation is "the occupation corresponding to the lowest energy" at
    mean-field level**, obtained from a handful of small ``(n_orb x n_orb)`` dense
    diagonalisations. It is quick and hard-bounded in memory regardless of chain length. HF
    is deterministic and its input (``h_op``) is replicated, so every rank computes the
    identical ``N0`` with no communication.

    The mean-field seed can miss the true integer sector by ``±1`` in strongly-correlated,
    near-degenerate cases; the ``mixed_valence`` window of the subsequent accurate solve
    absorbs such a miss.

    Parameters
    ----------
    h_op, impurity_orbitals, bath_states, N0
        As for :func:`calc_energy` / :func:`find_ground_state_basis`. ``N0`` sets the total
        (conserved) particle number: ``sum_i N0[i]`` impurity electrons plus the nominally
        full valence baths.
    comm : MPI.Comm or None
        Used only to gate the verbose print to rank 0; HF itself is uncommunicated.
    verbose : bool, default False

    Returns
    -------
    winning_N0 : dict
        Nominal impurity occupation per orbital set (rounded HF occupation).
    """
    winning_N0, energy, converged = hartree_fock_occupation(h_op, impurity_orbitals, bath_states, N0)
    if verbose and (comm is None or comm.rank == 0):
        status = "converged" if converged else "NOT converged"
        print(f"HF seed occupation: {winning_N0}  (E_HF ~ {energy:6.3f}, {status})")
    return winning_N0


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
    truncation_threshold=None,
    verbose=True,
    slaterWeightMin=1e-12,
    cipsi_solver_method="trlm",
    use_hf_seed=True,
    weighted_restrictions=None,
):
    """
    Find the occupation corresponding to the lowest energy, compare N0 - 1, N0 and N0 + 1

    use_hf_seed (default True): seed the nominal occupation with a cheap, memory-bounded
    unrestricted Hartree-Fock solve (the mean-field lowest-energy determinant), instead of
    the O(3^k) accurate scan over every dN combination. Set False for the legacy scan.

    truncation_threshold (default None): global cap on the number of Slater determinants in
    the basis; when the basis would grow past it, only the currently most important
    determinants are kept. ``None`` derives the cap from the available per-rank memory
    (:func:`impurityModel.ed.memory_estimate.suggest_truncation_threshold`; collective on
    ``comm``), ``np.inf`` disables capping.

    Returns:
    basis_gs, ManybodyBasis: Initial basis for the ground state
    """
    if truncation_threshold is None:
        # Same spin-orbital count formula as Basis.__init__ (blocked orbital lists).
        num_spin_orbitals = sum(
            sum(len(orbs) for orbs in impurity_orbitals[i])
            + sum(len(orbs) for orbs in bath_states[0][i])
            + sum(len(orbs) for orbs in bath_states[1][i])
            for i in bath_states[0]
        )
        truncation_threshold = suggest_truncation_threshold(num_spin_orbitals, comm=comm)
        log_memory_budget(
            truncation_threshold, num_spin_orbitals, comm=comm, verbose=verbose, label="ground-state basis"
        )
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
    # Cache key of the single entry allowed to hold a Basis (the running best). Every other
    # entry stores (energy, None): a revisited occupation only needs its basis when it is
    # strictly better than the current best, which cannot happen for a superseded entry, so
    # keeping one Basis instead of one per trial bounds the memory of the occupation scan.
    best_cached_key = None

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

        nonlocal best_cached_key

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
        if basis is not None and (best_cached_key is None or e_trial < energy_cache[best_cached_key][0]):
            if best_cached_key is not None:
                prev_e, prev_basis = energy_cache[best_cached_key]
                if prev_basis is not None:
                    prev_basis.comm = None
                energy_cache[best_cached_key] = (prev_e, None)
            energy_cache[key] = (e_trial, basis.copy())
            best_cached_key = key
        else:
            energy_cache[key] = (e_trial, None)
        return e_trial, basis

    keys = list(N0.keys())
    if use_hf_seed:
        # A cheap unrestricted Hartree-Fock solve locates the GS impurity occupation
        # (the mean-field lowest-energy determinant), replacing the O(3^k) dN scan; then a
        # single accurate solve at that occupation refines it. HF is quick and memory-bounded
        # (no broad-window CIPSI expansion), which matters for long bath chains.
        winning_N0 = hartree_fock_seed_occupation(h_op, impurity_orbitals, bath_states, N0, comm=comm, verbose=verbose)
        winning_N0 = {i: N0[i] if i in frozen_occupations else winning_N0[i] for i in N0}
        e_gs, basis = get_energy(winning_N0)
        gs_impurity_occ = winning_N0
        basis_gs = basis.copy() if basis is not None else None
        if verbose and (comm is None or comm.rank == 0):
            print("HF-seeded ground state occupation:", gs_impurity_occ, f"~ {e_gs:6.3f}", flush=True)
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
    cipsi_solver_method="irlm",
    num_wanted: int = 10,
    stats_path: str = "ground_state_statistics.json",
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
    solver = CIPSISolver(ground_state_basis)
    solver.expand(
        Hop, dense_cutoff=dense_cutoff, de2_min=1e-6, slaterWeightMin=slaterWeightMin, solver=cipsi_solver_method
    )
    es, psis = solver.get_eigenvectors(
        Hop,
        num_wanted=num_wanted,
        max_energy=energy_cut,
        dense_cutoff=dense_cutoff,
        slaterWeightMin=slaterWeightMin,
        solver=cipsi_solver_method,
    )
    ground_state_basis.clear()
    ground_state_basis.add_states(set(state for p in psis for state in p))
    psis = ground_state_basis.redistribute_psis(psis)

    effective_restrictions = get_effective_restrictions(ground_state_basis)
    if verbose:
        print("Effective GS restrictions:")
        for indices, occupations in effective_restrictions.items():
            print(f"---> {sorted(indices)} : {occupations}")
        print("=" * 80)
        print(f"{len(ground_state_basis)} Slater determinants in the basis.")

    comm = ground_state_basis.comm
    rank = comm.rank if comm is not None else 0
    # Single-particle density matrices, built distributed (each rank applies c_orb to its
    # local partition, redistributes, computes local inner products, then Allreduce); the
    # full rho is returned replicated on every rank. No full-state-vector gather needed.
    rhos = build_density_matrices(ground_state_basis, psis)

    e_avg = thermal_average_scale_indep(es, es, tau)
    thermal_rho = thermal_average_scale_indep(es, rhos, tau)

    # Occupation-weight statistics of the thermal ground state, built distributed
    # (collective call); returns the dict on rank 0 and None elsewhere.
    gs_stats = compute_gs_statistics(
        ground_state_basis,
        psis,
        es,
        tau,
        thermal_rho,
        ground_state_basis.impurity_spin_orbital_indices,
    )
    if rank == 0 and gs_stats is not None and stats_path is not None:
        save_gs_statistics(gs_stats, stats_path)

    # Sorted (original computational) order, matching the convention of block_structure
    # (local indices over the sorted impurity orbitals) and rot_to_spherical. The
    # impurity_orbitals dict is grouped (e.g. eg then t2g), so iterating it would slice the
    # density matrix into a *reordered* basis that no longer matches rot_to_spherical /
    # block_structure — corrupting the spherical rotation and the Sz / N(Up) split.
    impurity_indices = sorted(
        orb
        for impurity_orbital_blocks in ground_state_basis.impurity_orbitals.values()
        for block in impurity_orbital_blocks
        for orb in block
    )
    impurity_ix = np.ix_(impurity_indices, impurity_indices)
    # Impurity S^2 / L^2 / J^2 and <S_imp.S_bath> are two-body observables, so they
    # need the actual eigenstates rather than the density matrix. They are evaluated
    # *distributed*: each rank applies the operator to its local partition,
    # redistribute_psis realigns it, and manifold_observable_values Allreduces the
    # small <m|O|n> matrix (so this is a collective call with an identical result on
    # every rank — no state-vector gather). The L/S/J operators are built in the
    # spherical basis and rotated to the computational basis via rot_to_spherical
    # (symmetry-plan Phase 5).
    mov_comm = ground_state_basis.comm if ground_state_basis.is_distributed else None
    mov_redistribute = ground_state_basis.redistribute_psis if ground_state_basis.is_distributed else None
    s_values = l_values = j_values = None
    s2_thermal = l2_thermal = j2_thermal = None
    try:
        l_ops, s_ops, j_ops = make_impurity_casimir_operators(ground_state_basis.impurity_orbitals, rot_to_spherical)
    except ValueError:
        # The impurity is grouped into orbital-symmetry manifolds (e.g. eg / t2g), none of
        # which is individually a full spin-doubled l-shell, so the per-partition build raised.
        # L/S/J are shell-*total* operators, so aggregate the manifolds into the whole shell
        # (the sorted impurity_indices, which match the single rot_to_spherical matrix) and
        # retry. Only meaningful for a single shared rotation; a dict rotation is per-shell
        # (get_spectra's multi-l case) and is already correct per partition.
        l_ops = None
        if not isinstance(rot_to_spherical, dict):
            try:
                l_ops, s_ops, j_ops = make_impurity_casimir_operators({0: [impurity_indices]}, rot_to_spherical)
            except ValueError:
                # Genuinely not a spin-doubled l-shell: skip the Casimirs
                # (the rho-based <L.S>/<Lz>/<Sz> etc. still print).
                l_ops = None
    if l_ops is not None:
        # Evaluation is deterministic and identical on every rank (manifold_observable_values
        # Allreduces), so a failure raises on all ranks together -> the try/except stays
        # collective-safe and the report degrades instead of crashing the ground-state solve.
        try:
            casimir = {}
            for name, ops in (("S", s_ops), ("L", l_ops), ("J", j_ops)):
                vals = manifold_observable_values(
                    psis,
                    es,
                    lambda psi, _ops=ops: apply_casimir(psi, *_ops),
                    comm=mov_comm,
                    redistribute=mov_redistribute,
                )
                casimir[name] = (
                    np.array([casimir_to_quantum_number(v) for v in vals]),
                    thermal_observable_value(vals, es, tau),
                )
            s_values, s2_thermal = casimir["S"]
            l_values, l2_thermal = casimir["L"]
            j_values, j2_thermal = casimir["J"]
        except Exception as exc:  # noqa: BLE001 - reporting must not crash the GS solve
            if rank == 0:
                print(f"S^2/L^2/J^2 not reported: {exc}")
            s_values = l_values = j_values = None
            s2_thermal = l2_thermal = j2_thermal = None
    # Kondo impurity-bath spin correlation <S_imp . S_bath>. The bath spin pairing
    # follows the down-then-up convention, but is only trusted if the induced global
    # spin operators commute with the one-body Hamiltonian (so the spin assignment
    # is consistent with the model's spin symmetry); otherwise (SOC, non-standard
    # ordering) it is skipped rather than reported wrong.
    sisb_values = None
    sisb_thermal = None
    sisb_skip_reason = None
    try:
        n_orb = ground_state_basis.num_spin_orbitals
        # Fast path: the down-then-up index convention (valid in the spherical/c2i layout).
        imp_pairs = impurity_spin_pairs(ground_state_basis.impurity_orbitals)
        bath_pairs = bath_spin_pairs(ground_state_basis.bath_states)
        spin_pairs = None
        if imp_pairs and bath_pairs and spin_pairs_consistent_with_h(Hop, imp_pairs + bath_pairs, n_orb):
            spin_pairs = (imp_pairs, bath_pairs)
        else:
            # Fallback: derive the pairing from the Hamiltonian's spin symmetry (geometry-
            # agnostic, e.g. the linked double-chain / Haverkort bath where the computational
            # order is not down-then-up). Confirmed by the same [h, S] = 0 check.
            derived = derive_spin_pairs(Hop, ground_state_basis.impurity_orbitals, rot_to_spherical, n_orb)
            if derived is not None and spin_pairs_consistent_with_h(Hop, derived[0] + derived[1], n_orb):
                spin_pairs = derived
        if spin_pairs is None:
            sisb_skip_reason = (
                "could not determine a (down,up) spin pairing that commutes with the one-body "
                "Hamiltonian. The down-then-up index convention only holds in the spherical-harmonics "
                "representation, and the symmetry-derived fallback did not yield a consistent pairing "
                "(spin-orbit coupling, or a bath connectivity it cannot resolve)."
            )
        else:
            imp_pairs, bath_pairs = spin_pairs
            ops_imp = make_spin_operators(imp_pairs)
            ops_bath = make_spin_operators(bath_pairs)
            sisb_raw = manifold_observable_values(
                psis,
                es,
                lambda psi: apply_spin_correlation(psi, ops_imp, ops_bath),
                comm=mov_comm,
                redistribute=mov_redistribute,
            )
            sisb_values = np.real(sisb_raw)
            sisb_thermal = thermal_observable_value(sisb_raw, es, tau)
    except Exception as exc:  # noqa: BLE001 - reporting must not crash the GS solve
        # Deterministic + identical on every rank (collective Allreduce inside), so this raises
        # on all ranks together and stays collective-safe.
        sisb_values = None
        sisb_thermal = None
        sisb_skip_reason = f"spin-correlation evaluation failed: {exc}"

    if rank == 0:
        # The ground state is fully computed by here; formatting/printing the observable report
        # must never crash the solve. Any failure degrades to a warning and still returns the GS.
        # Rank-0-only (no collectives inside), so the guard cannot desync an MPI run.
        try:
            print(f"{impurity_indices=}")
            print("Block structure")
            print_block_structure(block_structure)
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
            full_impurity_ix = np.ix_(np.arange(len(rhos)), impurity_indices, impurity_indices)
            print_expectation_values(
                rhos[full_impurity_ix],
                es,
                rot_to_spherical,
                block_structure,
                s_values=s_values,
                l_values=l_values,
                j_values=j_values,
                sisb_values=sisb_values,
            )
            if sisb_skip_reason is not None:
                print(f"<S_imp.S_bath> not reported: {sisb_skip_reason}")
            if gs_stats is not None:
                print_gs_statistics(gs_stats)
                print("Ground state impurity / bath density matrices:")
                valence_bath_states, conduction_bath_states = ground_state_basis.bath_states
                for i in ground_state_basis.impurity_orbitals.keys():
                    print(f"orbital set {i}:")
                    impurity_orbital_blocks = ground_state_basis.impurity_orbitals[i]
                    valence_bath_orbital_blocks = valence_bath_states[i]
                    conduction_bath_orbital_blocks = conduction_bath_states[i]
                    for block_i, (imp_orbs, val_orbs, con_orbs) in enumerate(
                        zip(impurity_orbital_blocks, valence_bath_orbital_blocks, conduction_bath_orbital_blocks)
                    ):
                        print(f"Block {block_i}: impurity {imp_orbs}, valence {val_orbs}, conduction {con_orbs}")
                        impurity_ix = np.ix_(imp_orbs, imp_orbs)
                        bath_ix = np.ix_(val_orbs + con_orbs, val_orbs + con_orbs)
                        matrix_print(thermal_rho[impurity_ix], "Impurity density matrix:", n_prec=5)
                        matrix_print(thermal_rho[bath_ix], "Bath density matrix:", n_prec=5)
                        print("=" * 80)
                    print("", flush=verbose >= 2)
                print()
        except Exception as exc:  # noqa: BLE001 - reporting must not crash the GS solve
            print(f"[warning] ground-state observable report incomplete (GS still returned): {exc}")
    return psis, es, ground_state_basis, thermal_rho, {"rhos": rhos, "statistics": gs_stats}
