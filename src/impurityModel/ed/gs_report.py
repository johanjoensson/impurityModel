"""Ground-state observables and the printed report.

Everything ``calc_gs`` computes *about* a ground state it has already found: single-particle density
matrices, thermal averages, occupation statistics, entanglement, spin/orbital Casimirs, correlation and
screening diagnostics, static susceptibilities, and the report that prints them. Kept apart from
:mod:`impurityModel.ed.groundstate`, which only decides *which* state is the ground state.
"""

import numpy as np

from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_restrictions import get_effective_restrictions
from impurityModel.ed.basis_transcription import comm_support_filter
from impurityModel.ed.block_structure import get_equivalent_blocks, print_block_structure
from impurityModel.ed.density_matrices import build_density_matrices
from impurityModel.ed.gs_statistics import (
    compute_entanglement_entropy,
    compute_gs_statistics,
    print_gs_statistics,
    save_gs_statistics,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.observables import (
    block_group_labels,
    casimir_operator,
    casimir_to_quantum_number,
    compute_correlation_diagnostics,
    compute_energy_decomposition,
    compute_magnetic_summary,
    compute_screening_diagnostics,
    compute_state_summary,
    compute_static_susceptibilities,
    get_Sz_from_rho_pairs,
    is_j_sharp,
    make_impurity_casimir_operators,
    make_spin_operators,
    manifold_observable_values,
    print_correlation_diagnostics,
    print_expectation_values,
    print_impurity_orbital_groups,
    print_screening_diagnostics,
    print_state_summary,
    print_thermal_expectation_values,
    spin_correlation_operator,
    static_susceptibility_rows,
    thermal_observable_value,
)
from impurityModel.ed.spin_pairs import resolve_spin_pairs
from impurityModel.ed.symmetries import extract_tensors
from impurityModel.ed.utils import (
    V_DETAIL,
    V_SUMMARY,
    Reporter,
    format_index_ranges,
    matrix_print,
    print_density_matrix_summary,
    report_banner,
    report_rule,
)


def report_ground_state(
    Hop,
    ground_state_basis,
    es,
    psis,
    tau,
    block_structure,
    rot_to_spherical,
    verbose,
    stats_path,
    gs_truncation_report,
    gs_convergence_report,
):
    """Observables of the thermal ground manifold ``(es, psis)``, and the report that prints them.

    Collective on ``ground_state_basis.comm``. Returns ``(psis, es, ground_state_basis, thermal_rho,
    gs_info)`` -- the tuple ``calc_gs`` returns, with ``psis`` redistributed onto the basis and
    materialized as a list.
    """
    # Redistribute onto the basis as it stands; do not first collapse it onto the support of
    # `psis`. The basis is also the seed space for the excited/Green's-function expansion
    # downstream, and shrinking it to the thermal eigenvectors' support throws away exactly the
    # determinants a transition operator is most likely to reach.
    # `psis` is the block `get_eigenvectors` returns, so it redistributes as a block --
    # `*psis` would unpack its determinant KEYS, and at 1 rank `redistribute_psis` no-ops and
    # hands them straight back rather than raising. `prune_rows(0.0)` matches what
    # `redistribute_psis` did per column (it pruned each slice, and the caller's `from_states`
    # re-unioned), so the support is unchanged.
    psis_blk = ground_state_basis.redistribute_block(
        psis if isinstance(psis, ManyBodyState) else ManyBodyState.from_states(list(psis))
    )
    psis_blk.prune_rows(0.0)
    # Shared-support block view of the same redistributed states, built once and used
    # by every observable diagnostic below (Phase 6a of the state-unification refactor,
    # doc/plans/manybodystate_block_unification.md) instead of `width` independent
    # per-state calls. `psis` itself stays a list alongside it: `calc_gs`'s own return
    # contract is `list[ManyBodyState]`, and it is still needed as a list for
    # `add_states`/`redistribute_psis` above, so keeping both avoids a redundant
    # to_states() round trip at the return.
    # `calc_gs`'s documented return contract is `list[ManyBodyState]` (the spectra / GF / RIXS
    # seed paths consume it as one), so the list is materialized here from the block rather than
    # the other way round. Migrating that contract outward is a separate, deliberate step.
    psis = psis_blk.to_states()

    # The effective restrictions are printed in the ground-state-report overview below.
    effective_restrictions = get_effective_restrictions(ground_state_basis)

    comm = ground_state_basis.comm
    rank = comm.rank if comm is not None else 0
    report = Reporter(verbose, rank)
    # Single-particle density matrices, built distributed (each rank applies c_orb to its
    # local partition, redistributes, computes local inner products, then Allreduce); the
    # full rho is returned replicated on every rank. No full-state-vector gather needed.
    rhos = build_density_matrices(ground_state_basis, psis_blk)

    e_avg = thermal_average_scale_indep(es, es, tau)
    thermal_rho = thermal_average_scale_indep(es, rhos, tau)

    # Occupation-weight statistics of the thermal ground state, built distributed
    # (collective call); returns the dict on rank 0 and None elsewhere.
    gs_stats = compute_gs_statistics(
        ground_state_basis,
        psis_blk,
        es,
        tau,
        thermal_rho,
        ground_state_basis.impurity_spin_orbital_indices,
    )
    if gs_stats is not None:
        # Record whether the ground-state basis was truncation-capped (None = not capped),
        # so it lands in the saved statistics JSON alongside the occupation weights.
        gs_stats["truncation"] = gs_truncation_report
        gs_stats["convergence"] = gs_convergence_report

    # Many-body impurity-bath entanglement from the impurity RDM (collective call; its
    # memory guard may skip it, deterministically on every rank). Degrades to None.
    try:
        entanglement = compute_entanglement_entropy(ground_state_basis, psis_blk, es, tau)
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
    # *distributed*: each rank block-applies the operator to its local partition,
    # redistribute_block realigns it, and manifold_observable_values Allreduces the
    # small <m|O|n> matrix (so this is a collective call with an identical result on
    # every rank — no state-vector gather). The L/S/J operators are built in the
    # spherical basis and rotated to the computational basis via rot_to_spherical
    # (symmetry-plan Phase 5).
    mov_comm = ground_state_basis.comm if ground_state_basis.is_distributed else None
    mov_redistribute = ground_state_basis.redistribute_block if ground_state_basis.is_distributed else None
    # Every <m|O|n> below is read on the eigenstates' own support, so each apply drops the rest of
    # O|n> as it is generated. Built here, collectively and unconditionally, once for all of them.
    support_keep = comm_support_filter([psis_blk], mov_comm)
    s_values = l_values = j_values = None
    s2_thermal = l2_thermal = j2_thermal = None
    # Whether J is a sharp quantum number across the ground manifold (is_j_sharp); True
    # (ungated) is the safe default when the Casimirs are unavailable -- there is then no
    # signal to gate on, and the term/g_J/mu_eff rows already don't print in that case.
    j_sharp = True
    # Per-shell S^2/L^2/J^2 (Stage 4 consumes this for the per-shell report table):
    # partition -> {"l", "s2_thermal", "l2_thermal", "j2_thermal"}. Only populated when
    # there is more than one shell to distinguish -- a single shell's per-shell values
    # would just duplicate the totals computed below.
    shell_casimir = {}
    try:
        l_ops, s_ops, j_ops, per_shell_ops = make_impurity_casimir_operators(
            ground_state_basis.impurity_orbitals, rot_to_spherical, per_shell=True
        )
    except ValueError:
        # The impurity is grouped into orbital-symmetry manifolds (e.g. eg / t2g), none of
        # which is individually a full spin-doubled l-shell, so the per-partition build raised.
        # L/S/J are shell-*total* operators, so aggregate the manifolds into the whole shell
        # (the sorted impurity_indices, which match the single rot_to_spherical matrix) and
        # retry. Only meaningful for a single shared rotation; a dict rotation is per-shell
        # (get_spectra's multi-l case) and is already correct per partition.
        l_ops = None
        per_shell_ops = None
        if not isinstance(rot_to_spherical, dict):
            try:
                l_ops, s_ops, j_ops, per_shell_ops = make_impurity_casimir_operators(
                    {0: [impurity_indices]}, rot_to_spherical, per_shell=True
                )
            except ValueError:
                # Genuinely not a spin-doubled l-shell: skip the Casimirs
                # (the rho-based <L.S>/<Lz>/<Sz> etc. still print).
                l_ops = None
                per_shell_ops = None
    if l_ops is not None:
        # Evaluation is deterministic and identical on every rank (manifold_observable_values
        # Allreduces), so a failure raises on all ranks together -> the try/except stays
        # collective-safe and the report degrades instead of crashing the ground-state solve.
        try:
            casimir = {}
            for name, ops in (("S", s_ops), ("L", l_ops), ("J", j_ops)):
                op2 = casimir_operator(*ops)
                vals = manifold_observable_values(
                    psis_blk,
                    es,
                    lambda blk, _op=op2: _op.apply_block(blk, 0, support_keep),
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
            # Without SOC, J is generically not a good quantum number: the ground
            # manifold can be energy-degenerate while its members' J differ (see
            # is_j_sharp's docstring). Gate the printed/persisted term/g_J/mu_eff on
            # this rather than reporting a single J that isn't actually well-defined.
            j_sharp = is_j_sharp(j_values, es)
            if per_shell_ops is not None and len(per_shell_ops) > 1:
                for partition, (shell_l, shell_l_ops, shell_s_ops, shell_j_ops) in per_shell_ops.items():
                    entry = {"l": shell_l}
                    for name, ops in (("s2", shell_s_ops), ("l2", shell_l_ops), ("j2", shell_j_ops)):
                        op2 = casimir_operator(*ops)
                        vals = manifold_observable_values(
                            psis_blk,
                            es,
                            lambda blk, _op=op2: _op.apply_block(blk, 0, support_keep),
                            comm=mov_comm,
                            redistribute=mov_redistribute,
                        )
                        entry[f"{name}_thermal"] = thermal_observable_value(vals, es, tau)
                    shell_casimir[partition] = entry
        except Exception as exc:
            if rank == 0:
                print(f"S^2/L^2/J^2 not reported: {exc}")
            s_values = l_values = j_values = None
            s2_thermal = l2_thermal = j2_thermal = None
            j_sharp = True
            shell_casimir = {}
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
                "index convention only holds in the spherical-harmonics representation -- a "
                "symmetry-adapted-basis rotation of the correlated shell (rot_to_spherical set "
                "to a symmetry-adapted, not spherical, unitary) breaks it -- and the "
                "symmetry-derived fallback did not yield a consistent pairing, and the collinear "
                "spin-polarized-bath check also failed (spin-orbit coupling, or a bath "
                "connectivity/order the derivation cannot resolve, are other possible causes)."
            )
        else:
            imp_pairs, bath_pairs = spin_pairs
            ops_imp = make_spin_operators(imp_pairs)
            ops_bath = make_spin_operators(bath_pairs)
            sisb_op = spin_correlation_operator(ops_imp, ops_bath)
            sisb_raw = manifold_observable_values(
                psis_blk,
                es,
                lambda blk: sisb_op.apply_block(blk, 0, support_keep),
                comm=mov_comm,
                redistribute=mov_redistribute,
            )
            sisb_values = np.real(sisb_raw)
            sisb_thermal = thermal_observable_value(sisb_raw, es, tau)
            if sisb_pairing_approx:
                # Longitudinal part <Sz_imp Sz_bath>: exact under the collinear check (needs only
                # the verified spin labels), reported next to the pairing-dependent full value,
                # plus its connected form against the thermal single-particle polarizations.
                sisb_z_op = spin_correlation_operator(ops_imp, ops_bath, z_only=True)
                sisb_z_raw = manifold_observable_values(
                    psis_blk,
                    es,
                    lambda blk: sisb_z_op.apply_block(blk, 0, support_keep),
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
            psis_blk,
            es,
            tau,
            impurity_indices,
            s_z_op=s_ops[2] if l_ops is not None else None,
            l_z_op=l_ops[2] if l_ops is not None else None,
            comm=mov_comm,
            redistribute=mov_redistribute,
            keep=support_keep,
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
                psis_blk,
                es,
                tau,
                thermal_rho,
                imp_pairs,
                comm=mov_comm,
                redistribute=mov_redistribute,
                keep=support_keep,
            )
            # Impurity channels grouped like the N(...) columns (equivalent-block groups),
            # using the same orbital-group labels (e.g. "group 1") as the printed report, not
            # the raw block indices (which are positions in block_structure.blocks, not
            # orbital indices, and name nothing the rest of the run refers to).
            # Block orbitals are positions into the sorted impurity list (offset like
            # print_expectation_values); map them back to global spin-orbital indices.
            orb_offset = min(orb for block in block_structure.blocks for orb in block)
            equivalent_blocks = get_equivalent_blocks(block_structure)
            block_labels, _ = block_group_labels(
                equivalent_blocks, block_structure, ground_state_basis.impurity_orbitals
            )
            imp_groups = {}
            for blocks, label in zip(equivalent_blocks, block_labels):
                group_orbs = {impurity_indices[orb - orb_offset] for b in blocks for orb in block_structure.blocks[b]}
                in_group = [p for p in imp_pairs if p[0] in group_orbs and p[1] in group_orbs]
                if in_group:
                    imp_groups[label] = in_group
            if len(imp_groups) < 2:
                imp_groups = None  # a single channel equals the total <S_imp.S_bath>; skip
            screening_diagnostics = compute_screening_diagnostics(
                psis_blk,
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
                keep=support_keep,
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
            rhos[full_impurity_ix],
            es,
            rot_to_spherical,
            s_values,
            l_values,
            j_values,
            entanglement,
            impurity_orbitals=ground_state_basis.impurity_orbitals,
        )
    except Exception as exc:
        state_summary = None
        if rank == 0:
            print(f"per-state summary not reported: {exc}")
    if gs_stats is not None:
        try:
            magnetic_summary = compute_magnetic_summary(
                thermal_rho[impurity_ix],
                rot_to_spherical,
                s2_thermal,
                l2_thermal,
                j2_thermal,
                impurity_orbitals=ground_state_basis.impurity_orbitals,
                j_sharp=j_sharp,
            )
        except Exception as exc:
            magnetic_summary = None
            if rank == 0:
                print(f"magnetic summary not reported: {exc}")
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
        # Terse run record: always shown (V_RESULT), independent of the full report below.
        gs_occ = getattr(ground_state_basis, "ground_state_occupation", None)
        occ_str = f"; ground-state impurity occupation: {sum(gs_occ.values())}" if gs_occ else ""
        print(f"E0 = {np.min(es):.6f}   retained states = {len(es)}   tau = {tau:.4g}")
        print(f"{len(ground_state_basis):,} Slater determinants{occ_str}")
        if not report.enabled(V_SUMMARY):
            note = f"; statistics saved to {stats_path}" if (gs_stats is not None and stats_path is not None) else ""
            print(f"(full report: -v{note})")
    if report.enabled(V_SUMMARY):
        # The ground state is fully computed by here; formatting/printing the observable report
        # must never crash the solve. Any failure degrades to a warning and still returns the GS.
        # Rank-0-only (report.enabled already checks rank == 0; no collectives inside), so the
        # guard cannot desync an MPI run.
        try:
            report_banner("Ground-state report")
            print(f"impurity spin-orbitals: {impurity_indices}")
            # Range-compressed: a bath set spans ~90 spin-orbitals, and spelled out in full the
            # index list is all one sees of a line whose point is the occupation window at its end.
            print("Effective GS restrictions (observed electron count per orbital set):")
            for indices, occupations in effective_restrictions.items():
                print(f"  {format_index_ranges(indices)} : {occupations[0]}-{occupations[1]}")
            print("Block structure:")
            print_block_structure(block_structure)
            print_impurity_orbital_groups(
                get_equivalent_blocks(block_structure), block_structure, ground_state_basis.impurity_orbitals
            )

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
                impurity_orbitals=ground_state_basis.impurity_orbitals,
                shell_casimir=shell_casimir,
                j_sharp=j_sharp,
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
                impurity_orbitals=ground_state_basis.impurity_orbitals,
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
                print_gs_statistics(gs_stats, verbose=2 if verbose >= V_DETAIL else 1)
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
                        if verbose >= V_DETAIL:
                            matrix_print(thermal_rho[bath_ix], "Bath density matrix:", n_prec=5)
                        else:
                            print_density_matrix_summary(thermal_rho[bath_ix], "Bath density matrix (summary):")
                    print("", flush=verbose >= V_DETAIL)
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
            "convergence": gs_convergence_report,
            "correlation_diagnostics": corr_diagnostics,
            "screening_diagnostics": screening_diagnostics,
            "energy_decomposition": energy_decomposition,
            "static_susceptibilities": static_susceptibilities,
        },
    )
