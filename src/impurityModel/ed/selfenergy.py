from dataclasses import dataclass

import numpy as np

from impurityModel.ed import atomic_physics
from impurityModel.ed.symmetries import (
    classify_bath_occupation,
    extract_tensors,
    group_orbitals_by_blocks,
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)
from impurityModel.ed.basis_restrictions import build_weighted_restrictions
from impurityModel.ed.greens_function import build_full_greens_function, get_Greens_function, save_Greens_function
from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.memory_estimate import log_memory_budget, log_peak_vs_predicted, suggest_truncation_threshold
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

# The double-counting search and the self-energy extraction were split into their own modules;
# re-export their public entry points so calc_selfenergy's calls and existing
# selfenergy.<name> callers (and their test patches) resolve here unchanged.
from impurityModel.ed.double_counting import fixed_occupation_dc, fixed_peak_dc  # noqa: F401
from impurityModel.ed.sigma import (  # noqa: F401
    UnphysicalGreensFunctionError,
    check_greens_function,
    get_hcorr_v_hbath,
    get_Sigma_static,
    get_sigma,
    hyb,
)

# Adaptive symmetry-adapted-basis rotation (calc_selfenergy): drop rotated operator terms below
# this magnitude (eV; removes rotation round-off fill), and rotate into the symmetry-adapted
# basis only if it keeps the operator term count within this factor of the input basis.
_ROTATION_TRIM_TOL = 1e-8
_MAX_ROTATION_FILL = 2.0


def _per_group_occupation(nominal_occ, impurity_orbitals, h=None):
    """Map ``nominal_occ`` onto the derived orbital-symmetry groups.

    Accepts a dict already keyed by the group indices (used as-is), or any other dict / a
    scalar interpreted as the *total* impurity occupation. When the one-body Hamiltonian ``h``
    is supplied, the total is distributed by **energetic filling** — the lowest on-site-energy
    impurity spin-orbitals (``h[o, o]``) are occupied first — so e.g. a cubic d-shell fills the
    lower ``t2g`` manifold before ``eg`` (giving ``t2g=6``, ``eg=2`` for ``d8``) and the split
    is spin-symmetric. Without ``h`` it falls back to a size-proportional split (remainder to
    the largest groups). The prescan refines the per-group split, so this only needs to be a
    sensible starting point.
    """
    keys = list(impurity_orbitals)
    if isinstance(nominal_occ, dict) and set(nominal_occ) == set(keys):
        return {k: int(nominal_occ[k]) for k in keys}
    total = int(sum(nominal_occ.values()) if isinstance(nominal_occ, dict) else nominal_occ)

    if h is not None:
        # Energetic filling: occupy the lowest on-site-energy impurity spin-orbitals first and
        # count how many land in each group. Ties broken by orbital index for determinism.
        orb_to_group = {orb: k for k in keys for block in impurity_orbitals[k] for orb in block}
        ordered = sorted(orb_to_group, key=lambda o: (np.real(h[o, o]), o))
        alloc = {k: 0 for k in keys}
        for orb in ordered[: max(0, min(total, len(ordered)))]:
            alloc[orb_to_group[orb]] += 1
        return alloc

    sizes = {k: sum(len(block) for block in impurity_orbitals[k]) for k in keys}
    tot_size = sum(sizes.values()) or 1
    alloc = {k: int(total * sizes[k] // tot_size) for k in keys}
    remainder = total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: sizes[k], reverse=True):
        if remainder <= 0:
            break
        alloc[k] += 1
        remainder -= 1
    return alloc


def _per_group_scalar(value, impurity_orbitals, default=0):
    """Map a per-group scalar setting (e.g. mixed_valence) onto the derived group keys."""
    keys = list(impurity_orbitals)
    if isinstance(value, dict) and set(value) == set(keys):
        return dict(value)
    return {k: default for k in keys}


def _raise_together(comm, message):
    """Turn a rank-local validation verdict into a *collective* raise.

    .. warning:: **Collective on** ``comm`` (broadcast of the verdict from rank 0). Call it
       unconditionally on every rank, outside the ``if gs is not None`` guard.

    ``get_Greens_function`` gathers its results to global rank 0, so ``gs_matsubara`` /
    ``gs_realaxis`` (and the self-energies built from them) are ``None`` on every other rank.
    The physicality checks therefore run on rank 0 alone. Raising there unwinds rank 0 out of
    :func:`calc_selfenergy` and into ``MPI_Finalize`` while the remaining ranks walk on to the
    next collective -- ``log_peak_vs_predicted``'s ``Allreduce`` -- and block there forever.
    An unphysical Green's function then presents as a *hang* rather than an error.

    Broadcasting the verdict makes every rank raise the same exception at the same point.

    Parameters
    ----------
    comm : MPI communicator or None
    message : str or None
        The failure message on rank 0; ``None`` on the other ranks and when the check passed.
        Only rank 0's value is used.
    """
    if comm is not None:
        message = comm.bcast(message, root=0)
    if message is not None:
        raise UnphysicalGreensFunctionError(message)


@dataclass(frozen=True)
class _SolverBasis:
    """Solver-basis Hamiltonian and derived orbital/block layout for a self-energy run.

    Produced by :func:`_prepare_solver_basis`: the (optionally symmetry-adapted) solver
    Hamiltonian ``h`` and the matching non-interacting operator ``h0_solve``, the impurity/bath
    orbital grouping and block structure derived from it, the per-group occupation windows, and
    the rotations (``rotation_full`` full-space, ``u_imp`` impurity block) that carry results
    back to the caller's input basis.
    """

    h: object
    h0_solve: object
    n_spin_orbitals: int
    block_structure: object
    impurity_orbitals: dict
    bath_states: tuple
    nominal_occ: dict
    mixed_valence: dict
    rotation_full: "np.ndarray"
    u_imp: "np.ndarray"
    rot_to_spherical: "np.ndarray"
    total_impurity_orbitals: dict
    sum_bath_states: dict


def _prepare_solver_basis(h0, u4, impurity_orbitals, nominal_occ, mixed_valence, rot_to_spherical, verbosity):
    """Build the solver-basis Hamiltonian and derive its orbital/block layout.

    Assembles the interacting Hamiltonian ``H = h0 + U(u4)`` in the caller's input basis, then
    adaptively rotates into the impurity-diagonalising basis when that does not densify the
    Coulomb tensor (fill ratio ``<= _MAX_ROTATION_FILL``; the input basis is kept otherwise).
    Derives the bath valence/conduction split, the Green's-function block structure, and the
    per-group impurity/bath orbital grouping and occupation windows -- all in whichever basis is
    solved in. Returns a :class:`_SolverBasis`.
    """
    # construct local, interacting, hamiltonian (in the caller's input/correlated basis B)
    u = atomic_physics.getUop_from_rspt_u4(u4)
    h_input = ManyBodyOperator(h0) + ManyBodyOperator(u)

    # Flatten the impurity orbital dict (dict[int, list[int]]) into a plain spin-orbital index
    # list; the total orbital count is inferred from the Hamiltonian (impurity + bath). The bath
    # orbitals and their valence/conduction split are derived below, not passed in.
    impurity_indices = sorted(o for orbs in impurity_orbitals.values() for o in orbs)
    h_input_matrix = extract_tensors(h_input, two_body=False)[0]
    n_spin_orbitals = h_input_matrix.shape[0]

    # Adaptive symmetry-adapted basis: diagonalising the impurity one-body block collapses the
    # Green's-function block structure to its finest form (e.g. 1x1 eg/t2g blocks) BUT can
    # express the Coulomb interaction more densely. h0 and u4 are in the caller's "correlated"
    # input basis (NOT assumed spherical); the fill test below is measured *relative to that
    # input basis*, so we rotate only when it does not densify the operator (fill <= threshold)
    # and keep the input basis otherwise (e.g. a j,m_j eigenbasis under spin-orbit coupling
    # densifies the Coulomb tensor). Every output is rotated back to the input basis B before
    # returning; nothing here presumes a spherical-harmonic input.
    rotation_full, u_imp = impurity_symmetry_rotation(
        h_input, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_input_matrix
    )
    h_rotated = rotate_hamiltonian(h_input, rotation_full, tol=_ROTATION_TRIM_TOL)
    n_terms_input = sum(1 for v in h_input.values() if abs(v) > _ROTATION_TRIM_TOL)
    fill_ratio = len(h_rotated) / max(n_terms_input, 1)

    rotate = fill_ratio <= _MAX_ROTATION_FILL
    if rotate:
        h = h_rotated
        h0_solve = rotate_hamiltonian(ManyBodyOperator(h0), rotation_full, tol=_ROTATION_TRIM_TOL).to_dict()
        # Observable rotation for the solve (spherical -> S): compose the caller's input rotation
        # R_in (spherical -> B) with W^dag (B -> S). On the impurity block, R = u_imp^dag @ R_in.
        rot_to_spherical = u_imp.conj().T @ np.asarray(rot_to_spherical, dtype=complex)
    else:
        # Stay in the input basis; make the output rotation below a no-op.
        h = h_input
        h0_solve = h0
        rotation_full = np.eye(n_spin_orbitals, dtype=complex)
        u_imp = np.eye(len(impurity_indices), dtype=complex)

    # One-body matrix of the solver-basis Hamiltonian, extracted once and shared by the
    # classification/grouping helpers below (each would otherwise re-walk the full operator
    # and allocate its own dense n_orb x n_orb copy).
    h_matrix = extract_tensors(h, n_orb=n_spin_orbitals, two_body=False)[0] if rotate else h_input_matrix

    # Derive the bath orbitals (complement of the impurity set) and their initial occupation:
    # baths below the Fermi level (h[o, o] < 0) are valence (initially occupied), the rest are
    # conduction (initially empty). The bath one-body diagonal is unchanged by the impurity-only
    # rotation, so this is consistent whether measured in the input or solver basis.
    valence_flat, conduction_flat = classify_bath_occupation(
        h, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_matrix
    )

    # GF block structure from the hybridization-dressed impurity matrix (h[imp,imp] + V^dag V),
    # in whichever basis we solve in (fixes bath-mediated coupling; 1x1 blocks when rotated).
    # Derived from h *after* any rotation, so the blocks label the sectors of the solver basis.
    block_structure = impurity_block_structure(h, impurity_indices, h0_matrix=h_matrix)

    # Group the flat orbital lists into orbital-symmetry manifolds (the inequivalent blocks and
    # their spin-degenerate partners, e.g. eg / t2g) **in the solver basis** h. Grouping by the
    # block structure keeps both spins of a manifold in one group, so the many-body basis spans
    # all S_z sectors (spin multiplets stay degenerate); the impurity occupation window is tied
    # across groups by the restriction machinery, not pinned per group.
    impurity_orbitals, bath_states = group_orbitals_by_blocks(
        h, impurity_indices, valence_flat, conduction_flat, block_structure, n_orb=n_spin_orbitals, h0_matrix=h_matrix
    )
    nominal_occ = _per_group_occupation(nominal_occ, impurity_orbitals, h_matrix)
    mixed_valence = _per_group_scalar(mixed_valence, impurity_orbitals, default=0)

    valence_baths, conduction_baths = bath_states
    total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
    sum_bath_states = {
        i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
        for i in valence_baths
    }

    if verbosity > 0:
        basis_note = f"symmetry-adapted (fill {fill_ratio:.1f}x)" if rotate else f"input basis (fill {fill_ratio:.1f}x)"
        print(f"Block structure: {len(block_structure.blocks)} blocks, solving in {basis_note}")
    return _SolverBasis(
        h=h,
        h0_solve=h0_solve,
        n_spin_orbitals=n_spin_orbitals,
        block_structure=block_structure,
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        nominal_occ=nominal_occ,
        mixed_valence=mixed_valence,
        rotation_full=rotation_full,
        u_imp=u_imp,
        rot_to_spherical=rot_to_spherical,
        total_impurity_orbitals=total_impurity_orbitals,
        sum_bath_states=sum_bath_states,
    )


def _check_gf_physical(comm, gss, label):
    """Collectively verify a list of Green's functions is physical.

    Runs :func:`check_greens_function` on each block (skipping ``None`` and the ``None`` the
    non-root ranks hold), then broadcasts the verdict via :func:`_raise_together` so every rank
    raises (or continues) as one. Call it unconditionally on every rank.
    """
    message = None
    if gss is not None:
        try:
            for gs in gss:
                if gs is None:
                    continue
                check_greens_function(gs)
        except UnphysicalGreensFunctionError as err:
            message = f"{label} interacting Greens function:\n" + str(err)
    _raise_together(comm, message)


def _self_energy_on_mesh(
    mesh, gss, *, delta, total_impurity_orbitals, sum_bath_states, h0_solve, cluster_label, blocks, comm, label
):
    """Compute (and collectively physicality-check) the self-energy on one frequency mesh.

    Returns the per-inequivalent-block self-energy list, or ``None`` when ``gss`` is ``None``
    (``get_Greens_function`` gathers to rank 0, so the non-root ranks hold ``None``). On an
    unphysical result the offending blocks are saved to disk before the collective raise.

    .. warning:: **Collective on** ``comm`` (:func:`_raise_together`). ``get_sigma`` and the
       check run on rank 0 only, but ``_raise_together`` must run on *every* rank -- it is
       therefore called outside the ``gss is not None`` guard, never short-circuited by an
       early return.
    """
    sigma = None
    message = None
    if gss is not None:
        sigma = get_sigma(
            omega_mesh=mesh,
            impurity_orbitals=total_impurity_orbitals,
            nBaths=sum_bath_states,
            gs=gss,
            h0op=h0_solve,
            delta=delta,
            clustername=cluster_label,
            blocks=blocks,
        )
        try:
            for sig in sigma:
                check_greens_function(sig)
        except UnphysicalGreensFunctionError as err:
            for i, sig in enumerate(sigma):
                save_Greens_function(sig, mesh, f"sig+dc-{i}", cluster_label)
            message = f"{label} self-energy:\n" + str(err)
    _raise_together(comm, message)
    return sigma


def calc_selfenergy(model, meshes, basis, solver, *, comm, verbosity=0, cluster_label="cluster"):
    """Calculate the self energy of the impurity.

    Parameters
    ----------
    model : impurityModel.ed.model.ImpurityModel
        The impurity problem: the non-interacting Hamiltonian ``h0`` (single-index operator
        form), the Coulomb tensor ``u4``, the impurity orbital layout ``impurity_orbitals``
        (flat per-group spin-orbital index lists; the bath orbitals and their valence/
        conduction split are derived from ``h0`` internally), and ``rot_to_spherical``.
    meshes : impurityModel.ed.model.Meshes
        Matsubara (``iw``) and real (``w``) frequency meshes and the real-axis smearing
        (``delta``); either mesh may be ``None`` to skip that output.
    basis : impurityModel.ed.model.BasisOptions
        Many-body basis construction: nominal occupation, mixed valence, the occupation
        window ``dN``, the determinant budget ``truncation_threshold`` (``None`` derives the
        cap from available per-rank memory; ``np.inf`` disables capping), chain restrictions,
        spin-flip determinants, occupation cutoff, minimum Slater weight and temperature.
    solver : impurityModel.ed.model.SolverOptions
        Green's-function kernel (``gf_method`` -- ``"lanczos"``/``"bicgstab"``/``"sliced"``/``"cipsi"``),
        reorthogonalization mode, dense cutoff and the sparse-Green flag. See
        :func:`impurityModel.ed.greens_function.get_Greens_function`.
    comm : MPI.Comm or None
        MPI communicator.
    verbosity : int, optional
        Verbosity level.
    cluster_label : str, optional
        Label for the cluster.

    Returns
    -------
    dict
        Dictionary containing self-energy, Green's function, thermal density matrix, and ground state info.
    """
    # Unpack the grouped parameters into the local names used throughout the body.
    h0 = model.h0
    u4 = model.u4
    impurity_orbitals = model.impurity_orbitals
    rot_to_spherical = model.rot_to_spherical
    iw = meshes.iw
    w = meshes.w
    delta = meshes.delta
    nominal_occ = basis.nominal_occ
    mixed_valence = basis.mixed_valence
    tau = basis.tau
    chain_restrict = basis.chain_restrict
    spin_flip_dj = basis.spin_flip_dj
    occ_cutoff = basis.occ_cutoff
    truncation_threshold = basis.truncation_threshold
    slaterWeightMin = basis.slater_weight_min
    dN = basis.dN
    excitation_budget = basis.excitation_budget
    reort = solver.reort
    dense_cutoff = solver.dense_cutoff
    sparse_green = solver.sparse_green
    gf_method = solver.gf_method

    # MPI variables
    rank = comm.rank if comm is not None else 0

    def log(msg="", *, level=1, **kwargs):
        if verbosity >= level:
            print(msg, **kwargs)

    def banner(title, *, level=1):
        if verbosity >= level:
            print("\n" + "=" * 80)
            print(f"  {title}")
            print("=" * 80, flush=verbosity >= 2)

    sb = _prepare_solver_basis(h0, u4, impurity_orbitals, nominal_occ, mixed_valence, rot_to_spherical, verbosity)
    h = sb.h
    h0_solve = sb.h0_solve
    n_spin_orbitals = sb.n_spin_orbitals
    block_structure = sb.block_structure
    impurity_orbitals = sb.impurity_orbitals
    bath_states = sb.bath_states
    nominal_occ = sb.nominal_occ
    mixed_valence = sb.mixed_valence
    rotation_full = sb.rotation_full
    u_imp = sb.u_imp
    rot_to_spherical = sb.rot_to_spherical
    total_impurity_orbitals = sb.total_impurity_orbitals
    sum_bath_states = sb.sum_bath_states
    # Resolve the basis cap: None means "as many determinants as fit in RAM". Both the
    # suggestion and the budget log are collective on comm (memory probe + allreduce), so
    # they run unconditionally on every rank; only the printing is verbosity-gated.
    gf_block_width = max(4, *(len(block) for block in block_structure.blocks))
    if truncation_threshold is None:
        truncation_threshold = suggest_truncation_threshold(
            n_spin_orbitals, comm=comm, block_width=gf_block_width, reort=reort, method=gf_method
        )
    memory_budget = log_memory_budget(
        truncation_threshold,
        n_spin_orbitals,
        comm=comm,
        block_width=gf_block_width,
        reort=reort,
        verbose=verbosity > 0,
        label=cluster_label,
        method=gf_method,
    )
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
        # Optional excitation-budget weighted restriction on the ground-state basis; the GF
        # excited bases inherit it (widened) via greens_function._build_excited_restrictions.
        "weighted_restrictions": build_weighted_restrictions(bath_states, excitation_budget),
    }
    # Compute the thermal ground state and the interacting Green's function, with a single
    # auto-retry: the diagnostics report (gf_diagnostics) can detect that the thermal
    # ensemble was truncated (the highest retained state still carries Boltzmann weight); if
    # so we re-run the eigensolver with more requested states (num_wanted) once.
    num_wanted = 10
    max_retries = 2
    for _attempt in range(max_retries + 1):
        psis, es, ground_state_basis, thermal_rho, gs_info = calc_gs(
            h,
            basis_information,
            block_structure,
            rot_to_spherical,
            verbosity >= 2,
            slaterWeightMin=slaterWeightMin,
            num_wanted=num_wanted,
        )
        restrictions = ground_state_basis.restrictions

        if restrictions is not None:
            log("Restrictions on ground-state occupation:", level=2)
            for indices, limits in restrictions.items():
                log(f"  {sorted(indices)} : {limits}", level=2)

        banner("Interacting Green's function")
        log(f"Considering {len(es)} eigenstate(s) for the spectra.")
        log("Calculating interacting Green's function ...", flush=verbosity >= 2)

        gs_matsubara, gs_realaxis, gf_report = get_Greens_function(
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
            num_wanted=num_wanted,
            gf_method=gf_method,
        )

        # Root rank renders the diagnostics report and decides whether to retry; the decision
        # is broadcast so every rank re-enters calc_gs collectively (or all break).
        retry = False
        if rank == 0 and gf_report is not None:
            log(gf_report.render())
            retry = gf_report.needs_more_states and _attempt < max_retries
        if comm is not None:
            retry = comm.bcast(retry, root=0)
        if not retry:
            break
        num_wanted *= 2
        log(f"\nThermal ensemble appears truncated; retrying with num_wanted = {num_wanted}.\n", flush=True)
    # Physicality checks run on rank 0 (where the gathered results live); _check_gf_physical
    # broadcasts each verdict so every rank raises (or continues) as one.
    _check_gf_physical(comm, gs_matsubara, "Matsubara")
    _check_gf_physical(comm, gs_realaxis, "Real frequency")

    banner("Self-energy")
    log("Calculating self-energy ...")
    inequivalent_blocks = [block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks]
    sigma_real = _self_energy_on_mesh(
        w,
        gs_realaxis,
        delta=delta,
        total_impurity_orbitals=total_impurity_orbitals,
        sum_bath_states=sum_bath_states,
        h0_solve=h0_solve,
        cluster_label=cluster_label,
        blocks=inequivalent_blocks,
        comm=comm,
        label="Real frequency",
    )
    sigma = _self_energy_on_mesh(
        iw,
        gs_matsubara,
        delta=0,
        total_impurity_orbitals=total_impurity_orbitals,
        sum_bath_states=sum_bath_states,
        h0_solve=h0_solve,
        cluster_label=cluster_label,
        blocks=inequivalent_blocks,
        comm=comm,
        label="Matsubara",
    )

    log("Calculating static self-energy ...")
    # Sort the flattened indices: the groups enumerate the impurity orbitals in
    # block order (e.g. eg [0,1,5,6] before t2g [2,3,4,7,8,9]), but thermal_rho
    # and u4 below are in the input-basis orbital order. Indexing rho with the
    # unsorted list permutes it against u4 and yields a wrong static self-energy
    # (Sigma_static would no longer equal the iw -> inf limit of Sigma).
    impurity_indices = sorted(
        orb
        for impurity_blocks in ground_state_basis.impurity_orbitals.values()
        for block in impurity_blocks
        for orb in block
    )
    impurity_ix = np.ix_(impurity_indices, impurity_indices)

    # Rotate every result from the solver basis S back to the caller's input basis B
    # (O_B = W O_S W^dag; impurity block u_imp). When the adaptive test kept the input basis,
    # W and u_imp are identity and these are no-ops. The density matrix is full-space (rotate
    # with W); the self-energies / Green's functions are impurity-only (rotate with u_imp).
    thermal_rho = rotation_full @ thermal_rho @ rotation_full.conj().T

    def _to_input_basis(block_list):
        """Reassemble per-inequivalent-block matrices (basis S) and rotate to input basis B."""
        if block_list is None:
            return None
        full_s = build_full_greens_function(block_list, block_structure)
        if full_s.ndim == 3:  # (n_omega, n_imp, n_imp)
            return np.einsum("ij,wjk,lk->wil", u_imp, full_s, u_imp.conj())
        return u_imp @ full_s @ u_imp.conj().T

    sigma_full = _to_input_basis(sigma)
    sigma_real_full = _to_input_basis(sigma_real)
    gs_matsubara_full = _to_input_basis(gs_matsubara)
    gs_realaxis_full = _to_input_basis(gs_realaxis)

    # Static (Hartree-Fock) self-energy from the input-basis density matrix and u4 (input basis).
    sigma_static = get_Sigma_static(u4, thermal_rho[impurity_ix])

    # Predicted-vs-measured peak feedback for re-calibrating the byte model on
    # production-size runs (doc/plans/truncation_reliability.md). Collective on comm,
    # so it runs unconditionally; only the printing is verbosity-gated.
    log_peak_vs_predicted(memory_budget, comm=comm, verbose=verbosity > 0, label=cluster_label)

    return {
        "sigma": sigma_full,
        "sigma_real": sigma_real_full,
        "sigma_static": sigma_static,
        "gs_matsubara": gs_matsubara_full,
        "gs_realaxis": gs_realaxis_full,
        "thermal_rho": thermal_rho,
        "rhos": gs_info["rhos"],
        "gs_energies": np.asarray(es),
        "block_structure": block_structure,
        # None unless the truncation_threshold bound the ground-state basis; a dict with
        # the fixed-budget CIPSI refinement summary otherwise (see CIPSISolver.expand).
        "gs_truncation": gs_info.get("truncation"),
    }
