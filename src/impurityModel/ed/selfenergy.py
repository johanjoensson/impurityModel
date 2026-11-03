import time
from dataclasses import replace

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.cipsi_solver import _degeneracy_tol

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
from impurityModel.ed.degeneracy import degenerate_groups
from impurityModel.ed.gf_diagnostics import check_ground_state_truncation
from impurityModel.ed.gf_refinement import REFINE_SWAP, REFINE_TOL, RefineSpec, g0_check_shifts, g0_zero_shifts
from impurityModel.ed.greens_function import (
    build_full_greens_function,
    get_Greens_function,
    get_greens_function_moments,
    save_Greens_function,
)
from impurityModel.ed.groundstate import GS_DE2_MIN, GS_E_PT2_TOL, GroundStateSpec, calc_gs
from impurityModel.ed.gs_connectivity import solver_basis_connectivity
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.memory_estimate import (
    log_peak_vs_predicted,
    resolve_cap_policy,
)
from impurityModel.ed.sector_ensemble import resolve_mixed_valence
from impurityModel.ed.sigma import (  # noqa: F401
    UnphysicalGreensFunctionError,
    _format_causality_report,
    causality_report,
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


def _sigma_collect(mesh, gss, *, delta, estimator, solver_basis, cluster_label, blocks):
    """The self-energy on one mesh with its per-block causality records, without judging it.

    Rank 0 (where ``gss`` lives): ``(sigma, components, reports)`` with ``reports = {block_i: record}`` for every
    block whose diagonal has a positive imaginary part somewhere (:func:`sigma.causality_report`). Elsewhere
    ``(None, None, {})``. Rank-local: no collective."""
    if gss is None:
        return None, None, {}
    sigma, components = estimator.sigma(
        mesh,
        gss,
        delta=delta,
        solver_basis=solver_basis,
        blocks=blocks,
        cluster_label=cluster_label,
        return_components=True,
    )
    reports = {}
    for block_i, (sig, (g0_inv, ginv)) in enumerate(zip(sigma, components)):
        record = causality_report(sig, mesh, g0_inv, ginv)
        if record is not None:
            reports[block_i] = record
    return sigma, components, reports


def _judge_sigma(sigma, mesh, reports, *, label, cluster_label, note=""):
    """Rank 0: print the within-tolerance warnings, and return the failure message (or ``None``).

    A violation within ``SIGMA_CAUSALITY_TOL`` (relative, per diagonal orbital) is printed and tolerated; the first
    block above it fails the run: every block's self-energy is saved (``sig-i``) and the message names it
    (``note`` -- the retry's attribution -- is appended). Same verdicts and output as the check it replaces."""
    tol = config.SIGMA_CAUSALITY_TOL.get()
    for block_i in sorted(reports):
        record = reports[block_i]
        text = _format_causality_report(record, f"{label} self-energy")
        if record["worst_ratio"] > tol:
            for i, sig in enumerate(sigma):
                # "sig", not "sig+dc": the returned self-energy is the pure interaction term.
                # The double counting is already removed from the operator it is extracted
                # against (SolverBasis.h0_solve) and is applied to the lattice by RSPt.
                save_Greens_function(sig, mesh, f"sig-{i}", cluster_label)
            message = f"{label} self-energy:\nDiagonal term has positive imaginary part.\n{text}"
            return message + (f"\n{note}" if note else "")
        print(f"warning: {label} self-energy causality violated within tolerance (tol={tol:.1e}):\n{text}", flush=True)
    return None


def _self_energy_on_mesh(mesh, gss, *, delta, estimator, solver_basis, cluster_label, blocks, comm, label, note=""):
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
    sigma, _components, reports = _sigma_collect(
        mesh,
        gss,
        delta=delta,
        estimator=estimator,
        solver_basis=solver_basis,
        cluster_label=cluster_label,
        blocks=blocks,
    )
    message = None
    if sigma is not None:
        message = _judge_sigma(sigma, mesh, reports, label=label, cluster_label=cluster_label, note=note)
    _raise_together(comm, message)
    return sigma


#: A violation within this many broadenings of a bath level coupled to the block is reported as sitting on that pole.
_BATH_POLE_WIDTHS = 2.0
#: A bath eigenmode counts as coupled to an impurity orbital when its weight there is above this fraction of the
#: largest one.
_BATH_WEIGHT_FLOOR = 1e-6


def _bath_poles(sb):
    """``(levels, modes, v_full)``: the eigenvalues and eigenvectors of the bath block and the impurity-bath hopping
    (columns per impurity orbital), so a block's own pole weights are ``|U^dag V[:, orbital]|^2``. Rank-local."""
    _hcorr, v_full, _vd, h_bath = get_hcorr_v_hbath(sb.h0_solve, sb.total_impurity_orbitals, sb.sum_bath_states)
    levels, modes = np.linalg.eigh(h_bath)
    return levels, modes, np.asarray(v_full)


def _refine_tables(sb, blocks, w, iw, delta, comm):
    """``(tables, check_tables)``: the scoring shifts (:func:`gf_refinement.g0_zero_shifts`) and the acceptance
    test's check shifts (:func:`gf_refinement.g0_check_shifts`) of every Green's-function block, on its
    ``G0^-1 = z - h_corr - Delta(z)``, computed on rank 0 and broadcast: the tables' lengths decide the size of
    collective buffers downstream, so they must not depend on a rank-local ``eigh``."""
    tables = None
    if comm is None or comm.rank == 0:
        checks = []
        hcorr, v_full, _vd, h_bath = get_hcorr_v_hbath(sb.h0_solve, sb.total_impurity_orbitals, sb.sum_bath_states)
        bath_eig = np.linalg.eigh(h_bath)
        tables = []
        for block in blocks:
            ix = np.ix_(block, block)

            def g0_inv(zs, block=block, ix=ix):
                zs = np.asarray(zs, dtype=complex)
                eye = zs[:, np.newaxis, np.newaxis] * np.eye(len(block))[np.newaxis]
                return eye - hcorr[ix] - hyb(zs, v_full[:, block], h_bath, 0.0, bath_eig=bath_eig)

            zs, weights = g0_zero_shifts(g0_inv, w, iw, delta)
            tables.append(tuple(zip(zs, weights)))
            checks.append(g0_check_shifts(g0_inv, w, iw, delta))
        tables = (tuple(tables), tuple(checks))
    if comm is not None and comm.size > 1:
        tables = comm.bcast(tables, root=0)
    return tables


def _pole_note(omega, record, orbital, poles, delta):
    """One line: the nearest bath level that couples to ``orbital``, how far the worst point is from it, and whether
    that pole dominates ``G0^-1`` there (its own term ``|W|^2 delta / ((w-eps)^2 + delta^2)`` above ``|Re G^-1|``).
    Information only: a violation on a pole can still be the settle's (``|G|^-2`` amplifies there most)."""
    levels, modes, v_full = poles
    if v_full.size == 0 or len(levels) == 0:
        return None  # no bath
    weights = np.abs(modes.conj().T @ v_full[:, orbital]) ** 2
    coupled = np.flatnonzero(weights > _BATH_WEIGHT_FLOOR * max(float(weights.max()), np.finfo(float).tiny))
    if not len(coupled):
        return None
    k = coupled[int(np.argmin(np.abs(levels[coupled] - omega)))]
    widths = abs(omega - levels[k]) / delta
    own = weights[k] * delta / ((omega - levels[k]) ** 2 + delta**2)
    gi = record.get("ginv_at_worst")
    dominated = gi is not None and own > abs(gi.real)
    text = f"  nearest coupled bath level {levels[k]:.4g} (|W|^2 = {weights[k]:.1e}), {widths:.1f} broadenings away"
    if widths <= _BATH_POLE_WIDTHS and dominated:
        text += (
            "; G0^-1 is dominated by that pole: if the settle is not the cause this is the bath-discretisation "
            "artefact of doc/known_issues.md (raise delta or the number of bath states, or accept it with "
            "SIGMA_CAUSALITY_TOL)"
        )
    return text


def _attribute_block(block_i, block, failures, units, poles, delta, retry, have_run):
    """``(lines, retry_unit_ids)`` for one failing block: what the violation looks like and whether the support
    settle can be blamed. ``failures`` are the block's ``(mesh label, mesh, record)`` above tolerance. Rank 0 only;
    the lines are what a user reads when the run stops."""
    tol = config.SIGMA_CAUSALITY_TOL.get()
    lines = []
    real_axis = False
    for label, mesh, record in failures:
        omega = complex(mesh[record["worst_freq_index"]])
        where = f"{omega.real:.4g}" if omega.imag == 0 else f"i*{omega.imag:.4g}"
        lines.append(
            f"{label} self-energy, block {block}: worst Im Sigma = {record['worst_val']:+.3e} at w = {where} "
            f"(orbital {record['worst_orbital']}), relative {record['worst_ratio']:.2e} > SIGMA_CAUSALITY_TOL {tol:.0e}"
        )
        g0i, gi = record.get("g0_inv_at_worst"), record.get("ginv_at_worst")
        if g0i is not None:
            lines.append(f"  there Im(G0^-1) = {g0i.imag:.3e}, Im(G^-1) = {gi.imag:.3e}")
        if omega.imag == 0:
            real_axis = True
            if poles is not None and delta > 0:
                note = _pole_note(omega.real, record, block[record["worst_orbital"]], poles, delta)
                if note:
                    lines.append(note)
    mine = [u for u in units if u.block == block_i and u.retained is not None]
    settled = [u for u in mine if u.settled]
    if not settled:
        capped = [u for u in mine if u.cap_hit]
        if capped:
            lines.append(
                f"  support settle: no unit of this block settled, so it is not the cause. {len(capped)} of "
                f"{len(mine)} units froze at the determinant cap, which bounds the accuracy here, not slater_min: "
                "raise the cap, use importance-pruned growth (gf_prune_tol) or swap refinement (gf_refine_rounds), "
                "or raise delta (doc/known_issues.md)."
            )
        else:
            lines.append(
                "  support settle: no unit of this block settled, so it is not the cause. "
                "Lower slater_min, raise excitation_budget, or raise delta (doc/known_issues.md)."
            )
        return lines, []
    lines.append(
        f"  support settle: {len(settled)} of {len(mine)} units settled (gf_settle_ratio {settled[0].ratio:g}); "
        f"they drop determinants whose next-block amplitude stayed <= {settled[0].ratio:g} x slater_min"
    )
    if not real_axis:
        lines.append(
            "  not re-solved: a Matsubara-only failure; the settle changed G(i w) by less than 1e-13 on every "
            "measured run, so it cannot be the cause"
        )
        return lines, []
    if any(u.cap_hit or u.memory_frozen for u in settled):
        lines.append("  not re-solved: a unit of this block is also held by the cap or the memory guard")
        return lines, []
    if all(u.ratio == 0 for u in settled):
        lines.append("  not re-solved: the units already ran closure-only (gf_settle_ratio 0)")
        return lines, []
    if not retry:
        lines.append(
            "  not re-solved (gf_settle_retry is off). To test the settle, rerun with gf_settle_retry on, or with "
            "gf_settle_ratio 1 (closer to closure; expect a longer Green's-function phase)."
        )
        return lines, []
    if not have_run:
        lines.append("  not re-solved: this Green's-function kernel cannot re-solve units (gf_method)")
        return lines, []
    return lines, [u.unit for u in settled]


def _settle_retry(
    *,
    run,
    gf_report,
    family_matsubara,
    family_realaxis,
    iw,
    w,
    delta,
    estimator,
    sb,
    blocks,
    comm,
    report,
    cluster_label,
    retry,
    gf_seconds,
):
    """Attribute a self-energy causality failure and, when the support settle can plausibly be blamed and
    ``gf_settle_retry`` is on, solve the settled units again once, closure-only (``gf_settle_ratio = 0``).

    Re-solved only if: the block has settled units, none held by the cap or the memory guard, at least one failure
    is on the real axis (the settle does not reach ``G(i w)``), the units did not already run at ratio 0, and the
    kernel can re-solve units. A nearby bath pole is reported, not used as a veto. The closure-only re-solve can take
    as long as an unsettled run (3,271 s against 280 s on the small SMO case), which is why it is off by default.

    Returns ``(family_matsubara, family_realaxis, note, retried)``. ``retried`` is replicated (every rank must repeat
    the physicality check on the new ``G``); ``note`` (rank 0) is the attribution text the final failure message
    carries. Nothing is raised here: the caller's checks judge the (possibly new) ``G`` as they always did.
    The re-solve is taken only if every unit comes back uncapped, unguarded and converged, and the closure-only
    result is kept even if the violation got worse (it is the more complete solve at the same cutoff).
    **Collective** on ``comm``: the plan is broadcast from rank 0 before any rank branches, and
    :meth:`GFRun.resolve` is entered by every rank or by none."""
    rank0 = comm is None or comm.rank == 0
    tol = config.SIGMA_CAUSALITY_TOL.get()
    plan = {"units": []}
    note_lines = []
    failing = {}

    def collect(fam_m, fam_r):
        out = {}
        for label, mesh, fam, d in (("Real frequency", w, fam_r, delta), ("Matsubara", iw, fam_m, 0)):
            sigma, _comp, reports = _sigma_collect(
                mesh, fam, delta=d, estimator=estimator, solver_basis=sb, cluster_label=cluster_label, blocks=blocks
            )
            for block_i, record in reports.items():
                if record["worst_ratio"] > tol:
                    out.setdefault(block_i, []).append((label, mesh, sigma, record))
        return out

    if rank0:
        failing = collect(family_matsubara, family_realaxis)
        if failing:
            poles = _bath_poles(sb) if w is not None else None
            units = list(gf_report.units) if gf_report is not None else []
            for block_i in sorted(failing):
                lines, ids = _attribute_block(
                    block_i,
                    blocks[block_i],
                    [(label, mesh, record) for label, mesh, _sigma, record in failing[block_i]],
                    units,
                    poles,
                    delta,
                    retry,
                    run is not None,
                )
                note_lines += lines
                plan["units"] += ids
            plan["units"] = sorted(set(plan["units"]))
    if comm is not None:
        plan = comm.bcast(plan, root=0)
    if rank0 and note_lines:
        report("\n".join(note_lines), level=V_RESULT, flush=True)
    if not plan["units"]:
        return family_matsubara, family_realaxis, "\n".join(note_lines), False

    if rank0:
        report(
            f"Re-solving {len(plan['units'])} settled unit(s) closure-only (gf_settle_ratio 0); this can take as long "
            f"as an unsettled Green's function (the first pass took {gf_seconds:.1f} s). Set gf_settle_retry off to "
            "fail fast. The self-energy before the re-solve is saved as sig-*-before-retry.",
            level=V_RESULT,
            flush=True,
        )
        # The self-energy the failing run would have produced, kept for the before/after comparison.
        for _label, mesh, sigma, _record in {id(e[2]): e for entries in failing.values() for e in entries}.values():
            for i, sig in enumerate(sigma):
                save_Greens_function(sig, mesh, f"sig-{i}-before-retry", cluster_label)
    t0 = time.perf_counter()
    accepted, changes, new_m, new_r, new_report = run.resolve(plan["units"], 0.0)
    seconds = time.perf_counter() - t0
    if comm is not None:
        seconds = comm.allreduce(seconds, op=MPI.MAX)
    if rank0:

        def size(n):
            return f"{n:,}" if n is not None else "n/a"

        pieces = [f"unit {u}: {size(b)} -> {size(a)}" for u, b, a, _r in changes]
        report(
            f"Re-solved in {seconds:.1f} s; determinants {'; '.join(pieces)}",
            level=V_RESULT,
            flush=True,
        )
        if accepted:
            family_matsubara, family_realaxis = new_m, new_r
            if new_report is not None:
                report("Green's-function diagnostics after the re-solve:", level=V_RESULT)
                report(new_report.render(only_problems=not report.enabled(V_SUMMARY)), level=V_RESULT, flush=True)
            after = collect(family_matsubara, family_realaxis)
            worst_before = max(e[3]["worst_ratio"] for entries in failing.values() for e in entries)
            worst_after = max((e[3]["worst_ratio"] for entries in after.values() for e in entries), default=0.0)
            if not after:
                msg = (
                    f"sigma_settle_retry: cured (worst relative violation {worst_before:.2e} -> {worst_after:.2e}); "
                    f"the re-solve took {seconds:.1f} s. For the next iterations set gf_settle_ratio 1 (or 0 if 1 "
                    "is not enough), or every failing iteration pays this re-solve again."
                )
            else:
                msg = (
                    f"The closure-only support did not cure it ({worst_before:.2e} -> {worst_after:.2e}; the "
                    "closure-only result is kept and the failure stands), so the settle is not the cause. Lower "
                    "slater_min, raise excitation_budget, or raise delta; see doc/known_issues.md ('causality "
                    "violations at sharp hybridization resonances')."
                )
        else:
            why = "; ".join(sorted({r for *_x, r in changes if r}))
            msg = f"The re-solve was not taken ({why}); the first pass stands."
        note_lines.append(msg)
        report(msg, level=V_RESULT, flush=True)
    return family_matsubara, family_realaxis, "\n".join(note_lines), True


def _report_phase_time(report, phase, t_start):
    """One ``Wall time:`` line per solver phase at ``-v``, measured on the reporting rank.

    Every phase ends in collectives, so the root rank's elapsed time is the phase's wall time to
    within the last collective's skew. No collective here: a timing line must not add one.
    """
    report(f"Wall time: {phase} {time.perf_counter() - t_start:.1f} s", level=V_SUMMARY, flush=True)


def _state_list(psis):
    """The thermal states as a list of single-column states (a solve returns them as one block)."""
    return psis.to_states() if isinstance(psis, ManyBodyState) else list(psis)


def _drop_low_weight_manifolds(
    psis, es, tau, min_weight, slaterWeightMin, comm=None, *, e_ref=None, z_before=0.0, exempt_ground=True
):
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

    For a charge-sector ensemble the weights are normalised across the members solved so far: ``e_ref`` is the
    first member's realised ground energy and ``z_before`` the sum of ``exp(-(E - e_ref)/tau)`` over the earlier
    members' states, so a manifold's weight is ``exp(-(E - e_ref)/tau) / (z_before + this member's sum)``.  Later
    members only lower it, so the rule stays conservative.  Only the global ground manifold is exempt
    (``exempt_ground=False`` for every later member, whose states may all be dropped).  With the defaults the
    rule is the single-member one, unchanged.

    Returns
    -------
    (psis, es, dropped)
        The kept states and energies, in their original order (``es`` keeps its type), and one
        ``(E - E0, normalised weight)`` pair per dropped state.
    """
    e = np.real(np.asarray(es, dtype=complex))
    order = np.argsort(e, kind="stable")
    e0 = e[order[0]] if e_ref is None else e_ref
    weights = np.exp(-(e - e0) / tau)
    weights /= z_before + weights.sum()
    e_sorted = e[order]
    groups = degenerate_groups(e_sorted, tol=_degeneracy_tol(e_sorted, slaterWeightMin))
    keep = sorted(
        int(order[j])
        for g_i, group in enumerate(groups)
        if (g_i == 0 and exempt_ground) or max(weights[order[j]] for j in group) >= min_weight
        for j in group
    )
    if comm is not None:
        keep = comm.bcast(keep, root=0)
    kept = set(keep)
    dropped = [(float(e[i] - e0), float(weights[i])) for i in range(len(e)) if i not in kept]
    kept_es = es[keep] if isinstance(es, np.ndarray) else [es[i] for i in keep]
    return [psis[i] for i in keep], kept_es, dropped


def calc_selfenergy(model, meshes, basis, solver, *, comm, verbosity=0, cluster_label="cluster", cap_ledger=None):
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
    cap_ledger : impurityModel.ed.memory_estimate.CapLedger, optional
        Carries an auto ground-state cap between calls (P10): its ``ceiling`` lowers this call's auto cap and
        ``resolved`` receives the cap used. A cap the user set is never touched.

    Returns
    -------
    dict
        Dictionary containing self-energy, Green's function, thermal density matrix, and ground state info.
    """
    config.assert_uniform(comm)  # collective: a rank-divergent env knob is an error, not a deadlock
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
    mixed_valence, mv_note = resolve_mixed_valence(basis.mixed_valence, driver_default="auto")
    tau = basis.tau
    chain_restrict = basis.chain_restrict
    occ_cutoff = basis.occ_cutoff
    truncation_threshold = basis.truncation_threshold
    slaterWeightMin = basis.slater_weight_min
    dN = basis.dN
    excitation_budget = basis.excitation_budget
    dense_cutoff = solver.dense_cutoff
    sparse_green = solver.sparse_green
    gf_method = solver.gf_method
    gf_admission = solver.gf_admission
    gf_admit_tol = solver.gf_admit_tol
    gf_settle = solver.gf_settle
    gf_settle_ratio = solver.gf_settle_ratio
    gf_settle_retry = solver.gf_settle_retry
    gf_prune_tol = solver.resolved_gf_prune_tol
    gf_first_shell_tol = solver.gf_first_shell_tol
    gf_refine_rounds = solver.resolved_gf_refine_rounds
    gf_refine = (
        RefineSpec(
            rounds=gf_refine_rounds,
            tol=REFINE_TOL if solver.gf_refine_tol is None else solver.gf_refine_tol,
            swap=REFINE_SWAP if solver.gf_refine_swap is None else solver.gf_refine_swap,
        )
        if gf_refine_rounds
        else None
    )
    gf_tol = solver.gf_tol
    gf_real_tol = solver.gf_real_tol
    gf_min_weight = solver.gf_min_weight
    estimator = make_estimator(solver.sigma_method)

    # MPI variables
    rank = comm.rank if comm is not None else 0
    report = Reporter(verbosity, rank)
    if mv_note is not None:
        report(mv_note, level=V_RESULT)

    sb = prepare_solver_basis(
        h0,
        dc,
        u4,
        impurity_orbitals,
        nominal_occ,
        rot_to_spherical=rot_to_spherical,
        verbosity=verbosity,
        rank=rank,
        comm=comm,
        shell_ls=model.shell_ls,
    )
    h = sb.h
    h0_solve = sb.h0_solve
    n_spin_orbitals = sb.n_spin_orbitals
    block_structure = sb.block_structure
    impurity_orbitals = sb.impurity_orbitals
    bath_states = sb.bath_states
    nominal_occ = sb.nominal_occ
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
        connectivity=solver_basis_connectivity(sb, comm),
        ledger=cap_ledger,
    )
    # What decides which state is found, explicit (A1): this driver's values.
    gs_spec = GroundStateSpec(
        tau=tau,
        chain_restrict=chain_restrict,
        dense_cutoff=dense_cutoff,
        truncation_threshold=cap_policy,
        # Optional excitation-budget weighted restriction on the ground-state basis; the GF
        # excited bases inherit it (widened) via greens_function._build_excited_restrictions.
        excitation_budget=excitation_budget,
        cost_budget=basis.cost_budget,
        restriction_slater_weight_min=slaterWeightMin,
        # The residual PT2 energy the ground-state refinement converges to, and its optional
        # per-determinant floor.
        e_pt2_tol=GS_E_PT2_TOL if basis.e_pt2_tol is None else basis.e_pt2_tol,
        de2_min=GS_DE2_MIN if basis.de2_min is None else basis.de2_min,
        mixed_valence=mixed_valence,
    )
    basis_information = {
        "impurity_orbitals": impurity_orbitals,
        "bath_states": bath_states,
        "N0": nominal_occ,
        "comm": comm,
        "spin_layout": sb.spin_layout,
    }
    if sb.seed_window is not None:
        # A bath-less shell beside a correlated one: its seed is a window, not the whole shell (review J1).
        basis_information["frozen_occupations"] = sb.seed_window
    if gf_refine is not None:
        tables, check_tables = _refine_tables(
            sb,
            [block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks],
            w,
            iw,
            delta,
            comm,
        )
        gf_refine = replace(gf_refine, tables=tables, check_tables=check_tables)
    inequivalent_blocks = [block_structure.blocks[block_i] for block_i in block_structure.inequivalent_blocks]
    # The retry is off unless asked for: it can cost as much as an unsettled Green's function.
    settle_retry = gf_settle_retry is True and gf_settle is not False
    # Sort the flattened indices: the groups enumerate the impurity orbitals in
    # block order (e.g. eg [0,1,5,6] before t2g [2,3,4,7,8,9]), but h/thermal_rho below
    # are in the input-basis orbital order. get_greens_function_moments indexes h
    # with this list, so an unsorted list would permute the extracted moments against it.
    impurity_indices = sorted(orb for blocks_ in sb.impurity_orbitals.values() for block in blocks_ for orb in block)
    running = {
        "e_ref": None,
        "z": 0.0,
        "gf_caps": None,
    }  # the drop rule's normalisation across the charge-sector members so far

    def _impurity_gf(families):
        if families is None:
            return None
        return [estimator.impurity_gf(g, block) for g, block in zip(families, inequivalent_blocks)]

    def member_moments(psis, es, ground_state_basis):
        """The spectral moments M1..M3 of one set of thermal states (collective; replicated result).

        Built from the exact interacting Green's-function spectral moments -- collective (applies h + Allreduce),
        so run unconditionally on every rank. Only the inequivalent blocks' moments are consumed (the entries
        between blocks vanish by symmetry), so only those blocks' seeds are applied.
        """
        report("Calculating self-energy moments ...")
        t_phase = time.perf_counter()
        M = get_greens_function_moments(
            psis, es, tau, ground_state_basis, h, impurity_indices, blocks=inequivalent_blocks
        )
        _report_phase_time(report, "self-energy moments", t_phase)
        return M

    def member_gf(psis, es, ground_state_basis, truncation, convergence, label="", with_moments=True):
        """The interacting Green's function and moments of one set of thermal states (a charge-sector member).

        Collective. Returns ``None`` when the drop rule removes every state of the member (a minority member
        whose whole manifold is below ``gf_min_weight``), else a dict of the member's own thermally averaged
        families, spectral moments ``M`` and the settle-retry note. The averages are normalised over the member's
        kept states, exactly as for a single sector; the caller weights the members.
        """
        if running["e_ref"] is None:
            running["e_ref"] = float(np.min(np.real(np.asarray(es))))
        first = running["z"] == 0.0
        # The auto GF cap is sized once, by the dominant member's first stage, and every later member runs at the
        # same cap: a calculation has one cap, not one per member (the pins live on the basis object).
        if running["gf_caps"] is not None and getattr(ground_state_basis, "_auto_gf_caps", None) is None:
            ground_state_basis._auto_gf_caps = running["gf_caps"]
        restrictions = ground_state_basis.restrictions
        if restrictions is not None:
            report("Restrictions on ground-state occupation:", level=V_DETAIL)
            for indices, limits in restrictions.items():
                report(f"  {sorted(indices)} : {limits}", level=V_DETAIL)

        ensemble_es = es
        if gf_min_weight is not None:
            psis, es, dropped = _drop_low_weight_manifolds(
                psis,
                es,
                tau,
                gf_min_weight,
                slaterWeightMin,
                comm,
                e_ref=running["e_ref"],
                z_before=running["z"],
                exempt_ground=first,
            )
            if dropped:
                report(
                    f"gf_min_weight {gf_min_weight:g}: dropped {len(dropped)} of {len(ensemble_es)} thermal "
                    f"state(s) (E-E0 = {', '.join(f'{de:.4g}' for de, _w in dropped)}; total Boltzmann weight "
                    f"{sum(w for _de, w in dropped):.3e}) from the Green's function and self-energy.",
                    level=V_RESULT,  # it changes the result, so it is never hidden
                )
        running["z"] += float(np.exp(-(np.real(np.asarray(ensemble_es)) - running["e_ref"]) / tau).sum())
        if len(es) == 0:
            report(
                f"{label}every thermal state of this member is below gf_min_weight; the member is skipped.",
                level=V_RESULT,  # it changes the result, so it is never hidden
            )
            return None

        report.banner("Interacting Green's function")
        report(f"{label}Considering {len(es)} eigenstate(s) for the spectra.")
        report("Calculating interacting Green's function ...", flush=True)
        t_phase = time.perf_counter()
        gf_run = {}
        gs_matsubara, gs_realaxis, gf_report = get_Greens_function(
            matsubara_mesh=iw,
            omega_mesh=w,
            psis=psis,
            es=es,
            tau=tau,
            basis=ground_state_basis,
            hOp=h,
            delta=delta,
            blocks=inequivalent_blocks,
            verbose=verbosity >= 1,
            verbose_extra=verbosity >= 2,
            reort=None,  # the Green's-function recurrence never reorthogonalizes
            dN=dN,
            occ_cutoff=occ_cutoff,
            slaterWeightMin=slaterWeightMin,
            sparse=sparse_green,
            gf_method=gf_method,
            gf_admission=gf_admission,
            gf_admit_tol=gf_admit_tol,
            gf_settle=gf_settle,
            gf_settle_ratio=gf_settle_ratio,
            gf_prune_tol=gf_prune_tol,
            gf_first_shell_tol=gf_first_shell_tol,
            gf_refine=gf_refine,
            gf_tol=gf_tol,
            gf_real_tol=gf_real_tol,
            ensemble_es=ensemble_es,
            operator_families=lambda block: estimator.operator_families(block, sb),
            run_out=gf_run if settle_retry else None,
        )

        gf_seconds = time.perf_counter() - t_phase
        if running["gf_caps"] is None:
            running["gf_caps"] = getattr(ground_state_basis, "_auto_gf_caps", None)
        _report_phase_time(report, "Green's function", t_phase)
        if rank == 0 and gf_report is not None:
            gf_report.add("ground state", check_ground_state_truncation(truncation, convergence))
            # Always shown (this is the diagnostics report itself, not detail): only
            # problem rows at the terse default, the full table from -v.
            report(gf_report.render(only_problems=not report.enabled(V_SUMMARY)), level=V_RESULT)
        # get_Greens_function resolved the estimator's operator families; the impurity Green's
        # function is the leading block of each (the whole of it for the Dyson estimator).
        family_matsubara, family_realaxis = gs_matsubara, gs_realaxis

        # Physicality checks run on rank 0 (where the gathered results live); _check_gf_physical
        # broadcasts each verdict so every rank raises (or continues) as one.
        _check_gf_physical(comm, _impurity_gf(family_matsubara), "Matsubara")
        _check_gf_physical(comm, _impurity_gf(family_realaxis), "Real frequency")

        # A self-energy that fails the causality check is attributed, and when the support settle can plausibly be
        # blamed its units are solved again once, closure-only; the checks below then judge the (possibly new) G.
        family_matsubara, family_realaxis, retry_note, retried = _settle_retry(
            run=gf_run.get("run"),
            gf_report=gf_report,
            family_matsubara=family_matsubara,
            family_realaxis=family_realaxis,
            iw=iw,
            w=w,
            delta=delta,
            estimator=estimator,
            sb=sb,
            blocks=inequivalent_blocks,
            comm=comm,
            report=report,
            cluster_label=cluster_label,
            retry=settle_retry,
            gf_seconds=gf_seconds,
        )
        gf_run.clear()  # the run held every unit's seeds and results; nothing needs them past this point
        if retried:
            _check_gf_physical(comm, _impurity_gf(family_matsubara), "Matsubara")
            _check_gf_physical(comm, _impurity_gf(family_realaxis), "Real frequency")

        M = member_moments(psis, es, ground_state_basis) if with_moments else None
        return {
            "matsubara": family_matsubara,
            "realaxis": family_realaxis,
            "M": M,
            "note": retry_note,
            "es": es,
            "psis": None if with_moments else psis,
        }

    # The thermal ground state and the interacting Green's function. The ground-state solve returns
    # every eigenstate inside the thermal window, certified complete (or says it could not), so the
    # ensemble is never re-requested with a larger count. Under mixed_valence auto/on it returns the
    # dominant charge sector's states, and ``member_gf`` runs once per member inside it.
    t_phase = time.perf_counter()
    psis, es, ground_state_basis, thermal_rho, gs_info = calc_gs(
        h,
        basis_information,
        block_structure,
        rot_to_spherical,
        verbosity,
        slaterWeightMin=slaterWeightMin,
        shell_ls=sb.shell_ls,
        spec=gs_spec,
        member_hook=lambda member, basis, es, psis, trunc, conv: member_gf(
            _state_list(psis), es, basis, trunc, conv, label=f"[N_total = {member.total}] "
        ),
    )
    _report_phase_time(report, "ground state", t_phase)
    ensemble = gs_info.get("ensemble")
    if ensemble is None:  # mixed_valence off: one sector, no hook call
        # One sector: its moments come after the self-energy, as they always did (no basis is released meanwhile).
        member_results = [
            member_gf(
                psis,
                es,
                ground_state_basis,
                gs_info.get("truncation"),
                gs_info.get("convergence"),
                with_moments=False,
            )
        ]
        weights = [1.0]
    else:
        member_results = gs_info["member_results"]
        weights = [m["weight"] for m in ensemble["members"]]
    live = [(wt, r) for wt, r in zip(weights, member_results) if r is not None]
    if len(live) == 1:
        merged = live[0][1]
    else:
        total = sum(wt for wt, _ in live)
        coeff = [wt / total for wt, _ in live]

        def _combine(key):
            parts = [r[key] for _, r in live]
            if parts[0] is None:  # non-root ranks hold None
                return None
            return [sum(c * part[i] for c, part in zip(coeff, parts)) for i in range(len(parts[0]))]

        merged = {
            "matsubara": _combine("matsubara"),
            "realaxis": _combine("realaxis"),
            "M": sum(c * r["M"] for c, (_, r) in zip(coeff, live)),
            "psis": None,
            "note": "\n".join(r["note"] for _, r in live if r["note"]),
            "es": live[0][1]["es"],
        }
    es = merged["es"]
    family_matsubara, family_realaxis = merged["matsubara"], merged["realaxis"]
    retry_note, M = merged["note"], merged["M"]
    gs_matsubara = _impurity_gf(family_matsubara)
    gs_realaxis = _impurity_gf(family_realaxis)
    if len(live) > 1:
        # The merged Green's function is judged as one; a failure is reported, not retried.
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
        note=retry_note,
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
        note=retry_note,
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

    # High-frequency self-energy moments Sigma_1, Sigma_2 from the spectral moments M, replicated after the
    # reduction (so the conversion and block-symmetrisation below are identical on all ranks: pure numpy).
    if M is None:
        M = member_moments(merged["psis"], merged["es"], ground_state_basis)
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
        # None unless the charge-sector ensemble was formed: members, weights c_s, global e0 (mixed_valence).
        "gs_ensemble": ensemble,
    }
