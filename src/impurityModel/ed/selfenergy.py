import time

import numpy as np

from impurityModel.ed import config
from impurityModel.ed.basis_restrictions import build_weighted_restrictions
from impurityModel.ed.cipsi_solver import _degeneracy_tol, _degenerate_groups

# The double-counting criteria live in their own modules; re-export their public entry points so
# calc_selfenergy's calls and existing selfenergy.<name> callers (and their test patches) resolve
# here unchanged.
from impurityModel.ed.dc_criteria import (  # noqa: F401
    fixed_gap_dc,
    fixed_occupation_dc,
    fixed_peak_dc,
    occupation_and_energy_at_mu,
)
from impurityModel.ed.dc_reference import (  # noqa: F401
    discretized_impurity_occupation,
    report_continuum_reference,
)
from impurityModel.ed.dc_search import DoubleCountingUnreachable  # noqa: F401
from impurityModel.ed.dc_static import amf_dc, fll_dc, nominal_dc, sigma_inf_dc  # noqa: F401
from impurityModel.ed.gf_diagnostics import check_ground_state_truncation
from impurityModel.ed.greens_function import (
    build_full_greens_function,
    get_Greens_function,
    get_greens_function_moments,
    save_Greens_function,
)
from impurityModel.ed.groundstate import GS_DE2_MIN, GS_E_PT2_TOL, calc_gs
from impurityModel.ed.memory_estimate import (
    log_peak_vs_predicted,
    resolve_cap_policy,
)
from impurityModel.ed.sigma import (  # noqa: F401
    UnphysicalGreensFunctionError,
    check_greens_function,
    get_hcorr_v_hbath,
    get_sigma,
    get_Sigma_moments,
    get_Sigma_static,
    hyb,
)
from impurityModel.ed.sigma_estimators import make_estimator
from impurityModel.ed.solver_basis import (  # noqa: F401
    _MAX_ROTATION_FILL,
    _ROTATION_TRIM_TOL,
    _per_group_occupation,
    _per_group_scalar,
    prepare_solver_basis,
)
from impurityModel.ed.utils import V_DETAIL, V_RESULT, V_SUMMARY, Reporter


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


def _self_energy_on_mesh(mesh, gss, *, delta, estimator, solver_basis, cluster_label, blocks, comm, label):
    """Compute (and collectively physicality-check) the self-energy on one frequency mesh.

    ``gss`` are the Green's functions of ``estimator``'s operator families, and ``estimator``
    reads the self-energy off them. Returns the per-inequivalent-block self-energy list, or
    ``None`` when ``gss`` is ``None``
    (``get_Greens_function`` gathers to rank 0, so the non-root ranks hold ``None``). On an
    unphysical result the offending blocks are saved to disk before the collective raise.

    .. warning:: **Collective on** ``comm`` (:func:`_raise_together`). The estimator and the
       check run on rank 0 only, but ``_raise_together`` must run on *every* rank -- it is
       therefore called outside the ``gss is not None`` guard, never short-circuited by an
       early return.
    """
    sigma = None
    message = None
    if gss is not None:
        sigma, components = estimator.sigma(
            mesh,
            gss,
            delta=delta,
            solver_basis=solver_basis,
            blocks=blocks,
            cluster_label=cluster_label,
            return_components=True,
        )
        tol = config.SIGMA_CAUSALITY_TOL.get()
        try:
            for sig, (g0_inv, ginv) in zip(sigma, components):
                check_greens_function(
                    sig, tol=tol, omega_mesh=mesh, label=f"{label} self-energy", g0_inv=g0_inv, ginv=ginv
                )
        except UnphysicalGreensFunctionError as err:
            for i, sig in enumerate(sigma):
                # "sig", not "sig+dc": the returned self-energy is the pure interaction term.
                # The double counting is already removed from the operator it is extracted
                # against (SolverBasis.h0_solve) and is applied to the lattice by RSPt.
                save_Greens_function(sig, mesh, f"sig-{i}", cluster_label)
            message = f"{label} self-energy:\n" + str(err)
    _raise_together(comm, message)
    return sigma


def _report_phase_time(report, phase, t_start):
    """One ``Wall time:`` line per solver phase at ``-v``, measured on the reporting rank.

    Every phase ends in collectives, so the root rank's elapsed time is the phase's wall time to
    within the last collective's skew. No collective here: a timing line must not add one.
    """
    report(f"Wall time: {phase} {time.perf_counter() - t_start:.1f} s", level=V_SUMMARY, flush=True)


def _drop_low_weight_manifolds(psis, es, tau, min_weight, slaterWeightMin, comm=None):
    """Drop whole degenerate manifolds whose Boltzmann weight per state is below ``min_weight``.

    The eigensolver keeps every state inside the energy window ``-tau*ln(1e-4)``
    (:func:`average.energy_cut`), which the ground state and the double-counting search also use.
    For the Green's function each retained state costs a full set of work units, whatever its
    weight; ``gf_min_weight`` lets a run spend them only on the states that matter. A manifold is
    kept or dropped whole -- splitting one would make the result depend on which members the
    eigensolver happened to return -- and the ground manifold is always kept. The thermal
    average over what remains is renormalised by the Green's-function accumulators themselves.

    The decision is taken on the root rank and broadcast, so every rank keeps the same states
    (``comm`` collective; ``None`` for a serial call).

    Returns
    -------
    (psis, es, dropped)
        The kept states and energies, in their original order (``es`` keeps its type), and one
        ``(E - E0, normalised weight)`` pair per dropped state.
    """
    e = np.real(np.asarray(es, dtype=complex))
    order = np.argsort(e, kind="stable")
    e0 = e[order[0]]
    weights = np.exp(-(e - e0) / tau)
    weights /= weights.sum()
    e_sorted = e[order]
    groups = _degenerate_groups(e_sorted, tol=_degeneracy_tol(e_sorted, slaterWeightMin))
    keep = sorted(
        int(order[j])
        for g_i, group in enumerate(groups)
        if g_i == 0 or max(weights[order[j]] for j in group) >= min_weight
        for j in group
    )
    if comm is not None:
        keep = comm.bcast(keep, root=0)
    kept = set(keep)
    dropped = [(float(e[i] - e0), float(weights[i])) for i in range(len(e)) if i not in kept]
    kept_es = es[keep] if isinstance(es, np.ndarray) else [es[i] for i in keep]
    return [psis[i] for i in keep], kept_es, dropped


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
        Green's-function kernel (``gf_method`` -- ``"lanczos"`` or ``"bicgstab"``),
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
    dc = model.dc
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
    occ_cutoff = basis.occ_cutoff
    truncation_threshold = basis.truncation_threshold
    slaterWeightMin = basis.slater_weight_min
    dN = basis.dN
    excitation_budget = basis.excitation_budget
    reort = solver.reort
    dense_cutoff = solver.dense_cutoff
    sparse_green = solver.sparse_green
    gf_method = solver.gf_method
    gf_admission = solver.gf_admission
    gf_admit_tol = solver.gf_admit_tol
    gf_tol = solver.gf_tol
    gf_real_tol = solver.gf_real_tol
    gf_min_weight = solver.gf_min_weight
    estimator = make_estimator(solver.sigma_method)

    # MPI variables
    rank = comm.rank if comm is not None else 0
    report = Reporter(verbosity, rank)

    sb = prepare_solver_basis(
        h0, dc, u4, impurity_orbitals, nominal_occ, mixed_valence, rot_to_spherical, verbosity, rank=rank
    )
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
    # Resolve the basis cap: None means "as many determinants as fit in RAM". Collective on comm
    # (provenance broadcast + memory probe), so unconditional on every rank; only the printing is
    # verbosity-gated. An auto cap is sized on the ground-state path alone, exactly as the
    # double-counting search sizes it, so the dc a search found and the ground state solved here
    # at that dc use the same determinant budget; the GF units size themselves at GF entry. The
    # policy, not a bare number, travels down so the solves know whether memory may hold the
    # basis below this cap.
    cap_policy, memory_budget = resolve_cap_policy(
        truncation_threshold,
        n_spin_orbitals,
        comm=comm,
        verbose=verbosity > 0,
        label=cluster_label,
    )
    basis_information = {
        "impurity_orbitals": impurity_orbitals,
        "bath_states": bath_states,
        "N0": nominal_occ,
        "mixed_valence": mixed_valence,
        "tau": tau,
        "chain_restrict": chain_restrict,
        "dense_cutoff": dense_cutoff,
        "rank": rank,
        "comm": comm,
        "truncation_threshold": cap_policy,
        # Optional excitation-budget weighted restriction on the ground-state basis; the GF
        # excited bases inherit it (widened) via greens_function._build_excited_restrictions.
        "weighted_restrictions": build_weighted_restrictions(bath_states, excitation_budget),
        # The residual PT2 energy the ground-state refinement converges to.
        "e_pt2_tol": GS_E_PT2_TOL if basis.e_pt2_tol is None else basis.e_pt2_tol,
        # ...and its optional per-determinant floor.
        "de2_min": GS_DE2_MIN if basis.de2_min is None else basis.de2_min,
    }
    # Compute the thermal ground state and the interacting Green's function, with a single
    # auto-retry: the diagnostics report (gf_diagnostics) can detect that the thermal
    # ensemble was truncated (the highest retained state still carries Boltzmann weight); if
    # so we re-run the eigensolver with more requested states (num_wanted) once.
    num_wanted = 10
    max_retries = 2
    for _attempt in range(max_retries + 1):
        t_phase = time.perf_counter()
        psis, es, ground_state_basis, thermal_rho, gs_info = calc_gs(
            h,
            basis_information,
            block_structure,
            rot_to_spherical,
            verbosity,
            slaterWeightMin=slaterWeightMin,
            num_wanted=num_wanted,
        )
        _report_phase_time(report, "ground state", t_phase)
        restrictions = ground_state_basis.restrictions

        if restrictions is not None:
            report("Restrictions on ground-state occupation:", level=V_DETAIL)
            for indices, limits in restrictions.items():
                report(f"  {sorted(indices)} : {limits}", level=V_DETAIL)

        ensemble_es = es
        if gf_min_weight is not None:
            psis, es, dropped = _drop_low_weight_manifolds(psis, es, tau, gf_min_weight, slaterWeightMin, comm)
            if dropped:
                report(
                    f"gf_min_weight {gf_min_weight:g}: dropped {len(dropped)} of {len(ensemble_es)} thermal state(s) "
                    f"(E-E0 = {', '.join(f'{de:.4g}' for de, _w in dropped)}; total Boltzmann weight "
                    f"{sum(w for _de, w in dropped):.3e}) from the Green's function and self-energy.",
                    level=V_RESULT,  # it changes the result, so it is never hidden
                )

        report.banner("Interacting Green's function")
        report(f"Considering {len(es)} eigenstate(s) for the spectra.")
        report("Calculating interacting Green's function ...", flush=True)

        t_phase = time.perf_counter()
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
            gf_admission=gf_admission,
            gf_admit_tol=gf_admit_tol,
            gf_tol=gf_tol,
            gf_real_tol=gf_real_tol,
            ensemble_es=ensemble_es,
            operator_families=lambda block: estimator.operator_families(block, sb),
        )

        _report_phase_time(report, "Green's function", t_phase)
        # Root rank renders the diagnostics report and decides whether to retry; the decision
        # is broadcast so every rank re-enters calc_gs collectively (or all break).
        retry = False
        if rank == 0 and gf_report is not None:
            gf_report.add(
                "ground state", check_ground_state_truncation(gs_info.get("truncation"), gs_info.get("convergence"))
            )
            # Always shown (this is the diagnostics report itself, not detail): only
            # problem rows at the terse default, the full table from -v.
            report(gf_report.render(only_problems=not report.enabled(V_SUMMARY)), level=V_RESULT)
            retry = gf_report.needs_more_states and _attempt < max_retries
        if comm is not None:
            retry = comm.bcast(retry, root=0)
        if not retry:
            break
        num_wanted *= 2
        report(f"\nThermal ensemble appears truncated; retrying with num_wanted = {num_wanted}.\n", flush=True)
    # get_Greens_function resolved the estimator's operator families; the impurity Green's
    # function is the leading block of each (the whole of it for the Dyson estimator).
    inequivalent_blocks = [block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks]
    family_matsubara, family_realaxis = gs_matsubara, gs_realaxis

    def _impurity_gf(families):
        if families is None:
            return None
        return [estimator.impurity_gf(g, block) for g, block in zip(families, inequivalent_blocks)]

    gs_matsubara = _impurity_gf(family_matsubara)
    gs_realaxis = _impurity_gf(family_realaxis)

    # Physicality checks run on rank 0 (where the gathered results live); _check_gf_physical
    # broadcasts each verdict so every rank raises (or continues) as one.
    _check_gf_physical(comm, gs_matsubara, "Matsubara")
    _check_gf_physical(comm, gs_realaxis, "Real frequency")

    report.banner("Self-energy")
    report("Calculating self-energy ...")
    sigma_real = _self_energy_on_mesh(
        w,
        family_realaxis,
        delta=delta,
        estimator=estimator,
        solver_basis=sb,
        cluster_label=cluster_label,
        blocks=inequivalent_blocks,
        comm=comm,
        label="Real frequency",
    )
    sigma = _self_energy_on_mesh(
        iw,
        family_matsubara,
        delta=0,
        estimator=estimator,
        solver_basis=sb,
        cluster_label=cluster_label,
        blocks=inequivalent_blocks,
        comm=comm,
        label="Matsubara",
    )

    # Sort the flattened indices: the groups enumerate the impurity orbitals in
    # block order (e.g. eg [0,1,5,6] before t2g [2,3,4,7,8,9]), but h/thermal_rho below
    # are in the input-basis orbital order. get_greens_function_moments below indexes h
    # with this list, so an unsorted list would permute the extracted moments against it.
    impurity_indices = sorted(
        orb
        for impurity_blocks in ground_state_basis.impurity_orbitals.values()
        for block in impurity_blocks
        for orb in block
    )

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

    # High-frequency self-energy moments Sigma_1, Sigma_2 (coefficients of 1/(iw), 1/(iw)^2).
    # Built from the exact interacting Green's-function spectral moments M1..M3 -- collective
    # (applies h + Allreduce), so run unconditionally on every rank. The M tensor is replicated
    # after the reduction, so the conversion and block-symmetrisation below are identical on all
    # ranks (pure numpy on replicated data).
    report("Calculating self-energy moments ...")
    t_phase = time.perf_counter()
    M = get_greens_function_moments(psis, es, tau, ground_state_basis, h, impurity_indices)
    _report_phase_time(report, "self-energy moments", t_phase)
    hcorr, v_full, _, h_bath = get_hcorr_v_hbath(h0_solve, total_impurity_orbitals, sum_bath_states)
    sigma_inf_s, sigma_1_s, sigma_2_s = get_Sigma_moments(M, hcorr, v_full, h_bath)

    def _moment_to_input_basis(full_s):
        """Symmetry-match a full solver-basis moment matrix against the GF blocks, then rotate to B."""
        return _to_input_basis([full_s[np.ix_(block, block)] for block in inequivalent_blocks])

    sigma_static = _moment_to_input_basis(sigma_inf_s)
    sigma_moment_1 = _moment_to_input_basis(sigma_1_s)
    sigma_moment_2 = _moment_to_input_basis(sigma_2_s)

    # Predicted-vs-measured peak feedback for re-calibrating the byte model on
    # production-size runs (doc/plans/truncation_reliability.md). Collective on comm,
    # so it runs unconditionally; only the printing is verbosity-gated.
    log_peak_vs_predicted(memory_budget, comm=comm, verbose=verbosity > 0, label=cluster_label)

    return {
        "sigma": sigma_full,
        "sigma_real": sigma_real_full,
        "sigma_static": sigma_static,
        "sigma_moment_1": sigma_moment_1,
        "sigma_moment_2": sigma_moment_2,
        "gs_matsubara": gs_matsubara_full,
        "gs_realaxis": gs_realaxis_full,
        "thermal_rho": thermal_rho,
        "rhos": gs_info["rhos"],
        "gs_energies": np.asarray(es),
        "block_structure": block_structure,
        # None unless the truncation_threshold bound the ground-state basis; a dict with
        # the fixed-budget CIPSI refinement summary otherwise (see CIPSISolver.expand).
        "gs_truncation": gs_info.get("truncation"),
        # The ground state's residual PT2 energy against its tolerance (CIPSISolver.expand's
        # `convergence_report`): the error bar on the ground-state energy.
        "gs_convergence": gs_info.get("convergence"),
    }
