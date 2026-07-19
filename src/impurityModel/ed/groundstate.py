from itertools import product

import numpy as np

from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_restrictions import build_excited_restrictions, get_effective_restrictions
from impurityModel.ed.basis_transcription import build_density_matrices
from impurityModel.ed.block_structure import BlockStructure, get_equivalent_blocks, print_block_structure
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.gs_statistics import (
    compute_entanglement_entropy,
    compute_gs_statistics,
    print_gs_statistics,
    save_gs_statistics,
)
from impurityModel.ed.hartree_fock import hartree_fock_occupation
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.memory_estimate import log_memory_budget, suggest_truncation_threshold
from impurityModel.ed.observables import (
    apply_casimir,
    apply_spin_correlation,
    apply_spin_z_correlation,
    casimir_to_quantum_number,
    compute_correlation_diagnostics,
    compute_energy_decomposition,
    compute_magnetic_summary,
    compute_screening_diagnostics,
    compute_state_summary,
    compute_static_susceptibilities,
    get_Sz_from_rho_pairs,
    make_impurity_casimir_operators,
    make_spin_operators,
    manifold_observable_values,
    print_correlation_diagnostics,
    print_expectation_values,
    print_screening_diagnostics,
    print_state_summary,
    print_thermal_expectation_values,
    static_susceptibility_rows,
    thermal_observable_value,
)
from impurityModel.ed.spin_pairs import resolve_spin_pairs
from impurityModel.ed.symmetries import extract_tensors
from impurityModel.ed.utils import matrix_print, print_density_matrix_summary, report_banner, report_rule


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
    frozen_occupations=None,
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
    frozen_occupations : set, optional
        Orbital-set keys whose impurity occupation is pinned at exactly ``N0[i]`` during
        basis generation (a bath-less core shell); see
        :func:`impurityModel.ed.basis_generation.generate_initial_basis`.
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
        frozen_occupations=frozen_occupations,
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

    basis.restrictions = build_excited_restrictions(basis, h_op, psis=None, es=None, slater_weight_min=slaterWeightMin)
    if len(basis) == 0:
        return (np.inf, basis, None) if return_state else (np.inf, basis)
    solver.expand(
        h_op,
        dense_cutoff=dense_cutoff,
        # de2_min is a per-determinant Epstein-Nesbet PT2 energy threshold (|<Dj|H|psi>|^2
        # / |E_ref - E_Dj|). It was recalibrated ~2 orders down from the historical 1e-4
        # when the de2 denominator was corrected (previously a frozen ~1e-12 clamp made
        # de2_min a meaningless coupling filter); this occupation-search value is one order
        # looser than the final-GS solve in calc_gs.
        de2_min=1e-6,
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
    # Remember whether the truncation_threshold bound this occupation's expansion, so the
    # caller can report a capped ground-state determination even though the final basis
    # (reduced to the eigenstate support below) may fit under the cap.
    basis.occupation_search_truncation = solver.truncation_report
    basis.clear()
    basis.add_states({state for psi in eigen_psis for state in psi})
    if return_state:
        return np.min(es), basis, gs_state
    return np.min(es), basis


def hartree_fock_seed_occupation(
    h_op, impurity_orbitals, bath_states, N0, frozen_occupations=None, comm=None, verbose=False
):
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
    converged : bool
        Whether the HF iteration converged. A non-converged seed is **not** an occupation: it
        is whatever the last iterate happened to be, and callers must not use it (see
        :func:`find_ground_state_basis`, which falls back to the ``dN`` scan). Measured on the
        NiO L-edge workload, a non-converged seed returned ``{1: 4, 2: 10}`` -- a 3d10 impurity
        paid for by emptying the 2p core -- which, once the frozen core is restored, closes the
        shell, collapses the basis to a single determinant and zeroes every core-level spectrum.
    """
    winning_N0, energy, converged = hartree_fock_occupation(
        h_op, impurity_orbitals, bath_states, N0, frozen_occupations=frozen_occupations
    )
    if verbose and (comm is None or comm.rank == 0):
        status = "converged" if converged else "NOT converged"
        print(f"HF seed occupation: {winning_N0}  (E_HF ~ {energy:6.3f}, {status})")
    return winning_N0, converged


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
        _num_val_baths,
        _num_cond_baths,
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
    # Per-trial truncation report of the (memory-capped) occupation-search expansion, keyed
    # by trial occupation. The winning occupation's report is attached to basis_gs so a capped
    # ground-state determination is auditable even when the final basis fits under the cap.
    occ_search_reports = {}

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
            frozen_occupations=frozen_occupations,
        )
        if basis is not None:
            occ_search_reports[key] = getattr(basis, "occupation_search_truncation", None)
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

    def _hf_seed():
        """The HF seed occupation, or ``None`` when it cannot be trusted.

        The frozen shells are passed *into* HF, which solves at their fixed occupation
        (:func:`hartree_fock.hartree_fock_density_matrix`), so on a core-level workload the seed
        is now converged and usable and neither guard below fires. The guards remain because they
        are the difference between a wrong answer and a loud one:

        **(1) HF did not converge** -- the returned occupation is then merely the last iterate,
        not a minimiser. **(2) Re-freezing the frozen shells changes the impurity electron
        count** -- a seed that did not respect the freeze paid for the charge it moved somewhere
        else; restoring the frozen shells keeps its answer for the *unfrozen* ones while undoing
        that compensation, so the total silently drifts.

        Both fired on the NiO L-edge workload before the constraint existed, and either alone was
        fatal: unconstrained HF returned ``{2p: 4, 3d: 10}`` (sum 14 = the correct impurity count,
        a 3d10 paid for out of the core), and re-freezing the core to 6 left ``{2p: 6, 3d: 10}``
        = 16 -- two electrons from nowhere. That closes the shell (a filled system has exactly one
        determinant), so the basis collapsed and every core-level spectrum came out identically
        zero, while PES still looked healthy. Measured against the true d8 ground state, which
        lies inside the scan's own candidate space, the accepted state was 8.7 eV too high.
        """
        seed_N0, converged = hartree_fock_seed_occupation(
            h_op,
            impurity_orbitals,
            bath_states,
            N0,
            frozen_occupations=frozen_occupations,
            comm=comm,
            verbose=verbose,
        )
        refrozen = {i: N0[i] if i in frozen_occupations else seed_N0[i] for i in N0}
        leaked = sum(refrozen.values()) != sum(seed_N0.values())
        if converged and not leaked:
            return refrozen
        if verbose and (comm is None or comm.rank == 0):
            why = (
                "HF did not converge"
                if not converged
                else (
                    f"re-freezing {sorted(frozen_occupations)} changed the impurity count "
                    f"{sum(seed_N0.values())} -> {sum(refrozen.values())}"
                )
            )
            print(f"HF seed rejected ({why}); falling back to the dN occupation scan.", flush=True)
        return None

    hf_seed = _hf_seed() if use_hf_seed else None
    if hf_seed is not None:
        # A cheap unrestricted Hartree-Fock solve locates the GS impurity occupation
        # (the mean-field lowest-energy determinant), replacing the O(3^k) dN scan; then a
        # single accurate solve at that occupation refines it. HF is quick and memory-bounded
        # (no broad-window CIPSI expansion), which matters for long bath chains.
        winning_N0 = hf_seed
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
    if basis_gs is not None:
        # Whether the winning occupation's expansion was memory-capped (None if not), for
        # downstream truncation auditing (calc_gs merges this with its own final expand).
        basis_gs.occupation_search_truncation = occ_search_reports.get(tuple(sorted(gs_impurity_occ.items())))
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
    # de2_min: per-determinant PT2 energy tolerance for the final ground-state expansion
    # (recalibrated from 1e-6 when the de2 denominator was corrected in cipsi_solver; see
    # calc_energy). Tighter than the occupation search so the returned GS is well-converged.
    solver.expand(
        Hop, dense_cutoff=dense_cutoff, de2_min=1e-8, slaterWeightMin=slaterWeightMin, solver=cipsi_solver_method
    )
    # Record whether the truncation_threshold bound the ground-state determination (and how
    # the fixed-budget refinement resolved it), so a capped GS is auditable downstream
    # (returned in gs_info and saved to the statistics JSON). None when the cap never bound.
    # The cap can bind either the final expansion here or the earlier occupation search
    # (whose final basis may then fit under the cap); report either.
    gs_truncation_report = solver.truncation_report or getattr(ground_state_basis, "occupation_search_truncation", None)
    es, psis = solver.get_eigenvectors(
        Hop,
        num_wanted=num_wanted,
        max_energy=energy_cut,
        dense_cutoff=dense_cutoff,
        slaterWeightMin=slaterWeightMin,
        solver=cipsi_solver_method,
    )
    ground_state_basis.clear()
    ground_state_basis.add_states({state for p in psis for state in p})
    psis = ground_state_basis.redistribute_psis(psis)

    # The effective restrictions are printed in the ground-state-report overview below.
    effective_restrictions = get_effective_restrictions(ground_state_basis)

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
    if gs_stats is not None:
        # Record whether the ground-state basis was truncation-capped (None = not capped),
        # so it lands in the saved statistics JSON alongside the occupation weights.
        gs_stats["truncation"] = gs_truncation_report

    # Many-body impurity-bath entanglement from the impurity RDM (collective call; its
    # memory guard may skip it, deterministically on every rank). Degrades to None.
    try:
        entanglement = compute_entanglement_entropy(ground_state_basis, psis, es, tau)
    except Exception:
        entanglement = None
    if gs_stats is not None:
        gs_stats["entanglement"] = entanglement
    # The statistics JSON is saved after the remaining observable diagnostics below, so
    # they can be bundled into the same file.

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
        except Exception as exc:
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
    sisb_z_values = None
    sisb_z_thermal = None
    sisb_z_connected = None
    sisb_pairing_approx = False
    sisb_skip_reason = None
    spin_pairs = None
    try:
        n_orb = ground_state_basis.num_spin_orbitals
        # Validation cascade (down-then-up convention / symmetry-derived / collinear
        # spin-polarized bath) shared with the susceptibility driver. For the collinear
        # case (pairing_approx=True) the transverse (S_±) parts are flagged and the exact
        # longitudinal correlation is reported alongside.
        resolved = resolve_spin_pairs(
            Hop, ground_state_basis.impurity_orbitals, ground_state_basis.bath_states, rot_to_spherical, n_orb
        )
        if resolved is not None:
            imp_pairs, bath_pairs, sisb_pairing_approx = resolved
            spin_pairs = (imp_pairs, bath_pairs)
        if spin_pairs is None:
            sisb_skip_reason = (
                "could not determine a trustworthy (down,up) spin labelling. The down-then-up "
                "index convention only holds in the spherical-harmonics representation, the "
                "symmetry-derived fallback did not yield a consistent pairing, and the collinear "
                "spin-polarized-bath check also failed (spin-orbit coupling, or a bath "
                "connectivity/order the derivation cannot resolve)."
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
            if sisb_pairing_approx:
                # Longitudinal part <Sz_imp Sz_bath>: exact under the collinear check (needs only
                # the verified spin labels), reported next to the pairing-dependent full value,
                # plus its connected form against the thermal single-particle polarizations.
                sisb_z_raw = manifold_observable_values(
                    psis,
                    es,
                    lambda psi: apply_spin_z_correlation(psi, ops_imp, ops_bath),
                    comm=mov_comm,
                    redistribute=mov_redistribute,
                )
                sisb_z_values = np.real(sisb_z_raw)
                sisb_z_thermal = np.real(thermal_observable_value(sisb_z_raw, es, tau))
                sz_imp = get_Sz_from_rho_pairs(thermal_rho, imp_pairs)
                sz_bath = get_Sz_from_rho_pairs(thermal_rho, bath_pairs)
                sisb_z_connected = sisb_z_thermal - sz_imp * sz_bath
    except Exception as exc:
        # Deterministic + identical on every rank (collective Allreduce inside), so this raises
        # on all ranks together and stays collective-safe.
        sisb_values = None
        sisb_thermal = None
        sisb_z_values = None
        sisb_z_thermal = None
        sisb_z_connected = None
        sisb_pairing_approx = False
        spin_pairs = None
        sisb_skip_reason = f"spin-correlation evaluation failed: {exc}"

    # Static (Curie) susceptibilities of the retained thermal manifold. Charge always
    # works; spin/orbital need the many-body Sz/Lz from the Casimir build. Collective and
    # deterministic — separate degrade guard so it survives a spin-pairing failure.
    static_susceptibilities = None
    try:
        static_susceptibilities = compute_static_susceptibilities(
            psis,
            es,
            tau,
            impurity_indices,
            s_z_op=s_ops[2] if l_ops is not None else None,
            l_z_op=l_ops[2] if l_ops is not None else None,
            comm=mov_comm,
            redistribute=mov_redistribute,
        )
    except Exception:
        static_susceptibilities = None

    # Correlation-strength, screening and energy-decomposition diagnostics. Collective
    # (manifold_observable_values Allreduces inside) and deterministic on identical inputs,
    # so every rank takes the same path — same degrade-to-warning pattern as above.
    corr_diagnostics = None
    screening_diagnostics = None
    energy_decomposition = None
    diagnostics_skip_reason = None
    try:
        h1, _, _ = extract_tensors(Hop, n_orb=ground_state_basis.num_spin_orbitals, two_body=False)
        energy_decomposition = compute_energy_decomposition(thermal_rho, h1, impurity_indices, e_avg)
        if spin_pairs is not None:
            imp_pairs, bath_pairs = spin_pairs
            corr_diagnostics = compute_correlation_diagnostics(
                psis, es, tau, thermal_rho, imp_pairs, comm=mov_comm, redistribute=mov_redistribute
            )
            # Impurity channels grouped like the N(...) columns (equivalent-block groups).
            # Block orbitals are positions into the sorted impurity list (offset like
            # print_expectation_values); map them back to global spin-orbital indices.
            orb_offset = min(orb for block in block_structure.blocks for orb in block)
            imp_groups = {}
            for blocks in get_equivalent_blocks(block_structure):
                label = ",".join(str(b) for b in blocks)
                group_orbs = {impurity_indices[orb - orb_offset] for b in blocks for orb in block_structure.blocks[b]}
                in_group = [p for p in imp_pairs if p[0] in group_orbs and p[1] in group_orbs]
                if in_group:
                    imp_groups[label] = in_group
            if len(imp_groups) < 2:
                imp_groups = None  # a single channel equals the total <S_imp.S_bath>; skip
            screening_diagnostics = compute_screening_diagnostics(
                psis,
                es,
                tau,
                thermal_rho,
                imp_pairs,
                bath_pairs,
                h1,
                imp_groups=imp_groups,
                z_only=sisb_pairing_approx,
                comm=mov_comm,
                redistribute=mov_redistribute,
            )
        else:
            diagnostics_skip_reason = "no validated spin pairing (see the <S_imp.S_bath> message)"
    except Exception as exc:
        corr_diagnostics = None
        screening_diagnostics = None
        energy_decomposition = None
        diagnostics_skip_reason = f"evaluation failed: {exc}"

    # Bundle the observable diagnostics into the statistics dict (rank 0) and save the
    # JSON only now, so ground_state_statistics.json carries the complete report.
    full_impurity_ix = np.ix_(np.arange(len(rhos)), impurity_indices, impurity_indices)
    try:
        state_summary = compute_state_summary(
            rhos[full_impurity_ix], es, rot_to_spherical, s_values, l_values, j_values, entanglement
        )
    except Exception:
        state_summary = None
    if gs_stats is not None:
        try:
            magnetic_summary = compute_magnetic_summary(
                thermal_rho[impurity_ix], rot_to_spherical, s2_thermal, l2_thermal, j2_thermal
            )
        except Exception:
            magnetic_summary = None
        gs_stats["observables"] = {
            "magnetic": magnetic_summary,
            "per_state": state_summary,
            "sisb": {
                "thermal": None if sisb_thermal is None else float(np.real(sisb_thermal)),
                "z_thermal": sisb_z_thermal,
                "z_connected": sisb_z_connected,
                "pairing_approx": sisb_pairing_approx,
            },
            "correlation": corr_diagnostics,
            "screening": screening_diagnostics,
            "energy_decomposition": energy_decomposition,
            "static_susceptibilities": static_susceptibilities,
        }
    if rank == 0 and gs_stats is not None and stats_path is not None:
        save_gs_statistics(gs_stats, stats_path)

    if rank == 0:
        # The ground state is fully computed by here; formatting/printing the observable report
        # must never crash the solve. Any failure degrades to a warning and still returns the GS.
        # Rank-0-only (no collectives inside), so the guard cannot desync an MPI run.
        try:
            report_banner("Ground-state report")
            print(f"E0 = {np.min(es):.6f}   retained states = {len(es)}   tau = {tau:.4g}")
            print(f"{len(ground_state_basis)} Slater determinants in the basis.")
            print(f"impurity spin-orbitals: {impurity_indices}")
            print("Effective GS restrictions:")
            for indices, occupations in effective_restrictions.items():
                print(f"  {sorted(indices)} : {occupations}")
            print("Block structure:")
            print_block_structure(block_structure)

            report_rule("Thermal expectation values")
            extra_groups = []
            if energy_decomposition is not None:
                extra_groups.append(
                    (
                        "Energy decomposition",
                        [
                            ("<H_1body>", energy_decomposition["e_one_body"], ""),
                            ("<H_imp,1b>", energy_decomposition["e_imp_1b"], ""),
                            ("<H_bath>", energy_decomposition["e_bath"], ""),
                            ("<E_hyb>", energy_decomposition["e_hyb"], ""),
                            ("<H_Coulomb>", energy_decomposition["e_coulomb"], "(= <E> - <H_1body>)"),
                        ],
                    )
                )
            if static_susceptibilities is not None:
                extra_groups.append(
                    ("Static susceptibilities (Curie)", static_susceptibility_rows(static_susceptibilities))
                )
            print_thermal_expectation_values(
                thermal_rho[impurity_ix],
                e_avg,
                rot_to_spherical,
                block_structure,
                s_thermal=s2_thermal,
                l_thermal=l2_thermal,
                j_thermal=j2_thermal,
                sisb_thermal=sisb_thermal,
                sisb_z_thermal=sisb_z_thermal,
                sisb_z_connected=sisb_z_connected,
                sisb_pairing_approx=sisb_pairing_approx,
                extra_groups=extra_groups or None,
            )
            report_rule("Eigenstates")
            print_expectation_values(
                rhos[full_impurity_ix],
                es,
                rot_to_spherical,
                block_structure,
                s_values=s_values,
                l_values=l_values,
                j_values=j_values,
                sisb_values=sisb_values,
                sisb_z_values=sisb_z_values,
            )
            if state_summary is not None:
                print_state_summary(state_summary)
                print()
            if sisb_skip_reason is not None:
                print(f"<S_imp.S_bath> not reported: {sisb_skip_reason}")
            if corr_diagnostics is not None:
                report_rule("Correlation strength")
                print_correlation_diagnostics(corr_diagnostics)
            if screening_diagnostics is not None:
                report_rule("Screening")
                print_screening_diagnostics(screening_diagnostics)
            if diagnostics_skip_reason is not None:
                print(f"correlation/screening diagnostics not reported: {diagnostics_skip_reason}")
            if gs_stats is not None:
                report_rule("Configurations & entanglement")
                print_gs_statistics(gs_stats, verbose=2 if verbose >= 2 else 1)
                report_rule("Density matrices")
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
                        if verbose >= 2:
                            matrix_print(thermal_rho[bath_ix], "Bath density matrix:", n_prec=5)
                        else:
                            print_density_matrix_summary(thermal_rho[bath_ix], "Bath density matrix (summary):")
                    print("", flush=verbose >= 2)
                print()
        except Exception as exc:
            print(f"[warning] ground-state observable report incomplete (GS still returned): {exc}")
    return (
        psis,
        es,
        ground_state_basis,
        thermal_rho,
        {
            "rhos": rhos,
            "statistics": gs_stats,
            "truncation": gs_truncation_report,
            "correlation_diagnostics": corr_diagnostics,
            "screening_diagnostics": screening_diagnostics,
            "energy_decomposition": energy_decomposition,
            "static_susceptibilities": static_susceptibilities,
        },
    )
