r"""Replay one archived solver call with the Green's-function support freeze (settle) switched, and grade it.

The measurement rig behind ``doc/plans/gf_support_freeze_redesign.md``. One run = one (archive, cluster,
iteration, ``slaterWeightMin``, freeze mode) point: the archived ``calc_selfenergy`` call is rebuilt through
:mod:`real_workload` (same options, ``gf_tol``/``gf_real_tol`` from the archive), the self-energy causality guard
is left non-fatal (``selfenergy.DEFAULT_CAUSALITY_TOL``) so the Sigma comes out whatever it is, and the arrays
are written to an ``.npz``. :func:`grade` then compares two such runs against the acceptance rule of the plan:

* (A) ``max |dSigma_mats|`` on the Matsubara mesh;
* (B) per orbital, pointwise on the real mesh, ``|dSigma_ii(w)| / h_i(w)`` with the causality guard's own headroom
  ``h_i(w) = -Im Sigma_ref,ii(w) + tol * max_w |Im Sigma_ref,ii|`` (``tol`` = the causality tolerance);
* ``max |dG| / max |G|`` per axis, reported and not gated.

Run (one process per point; ``-n 3`` is what the numbers in the write-up used)::

    mpiexec -n 3 python -m impurityModel.test.support.freeze_replay run ARCHIVE.h5 --iteration 1 \
        --swm 1e-8 --freeze off --out debug/gf_freeze/swm8_nofreeze
    python -m impurityModel.test.support.freeze_replay grade debug/gf_freeze/swm8.npz \
        debug/gf_freeze/swm8_nofreeze.npz

``--freeze`` is ``on`` (the options' defaults), ``off`` (``gf_settle=False``) or a number (``gf_settle_ratio``).
The archive path is always an argument: nothing here hard-codes a workload.
"""

import argparse
import os
import sys
import time

# One thread per rank: set before numpy loads (OpenBLAS reads these once, at import), or a 3-rank replay
# runs ~30 threads and oversubscribes the machine. Overridable from the environment.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

#: Per-orbital quantities :func:`grade` reports, in print order.
GRADE_FIELDS = ("mats_abs", "real_abs", "real_vs_headroom", "g_real_rel", "g_mats_rel", "worst_im_ref", "worst_im_new")


def headroom(sigma_ref, tol):
    r"""``h_i(w) = -Im Sigma_ref,ii(w) + tol * max_w |Im Sigma_ref,ii|`` -- the room the causality guard leaves.

    ``sigma_ref`` has shape ``(n_w, n, n)``; the result ``(n_w, n)``. Points with ``h < tol * max`` already
    violate causality in the reference (the known-issues artefact) and are not a freeze effect.
    """
    im = np.imag(np.diagonal(np.asarray(sigma_ref), axis1=1, axis2=2))
    return -im + tol * np.max(np.abs(im), axis=0, keepdims=True)


def grade(ref, new, tol=1e-3):
    """Compare two result dicts (``sigma``, ``sigma_real``, ``gs_matsubara``, ``gs_realaxis``) per the plan's rule.

    Returns a dict with ``mats_abs`` (A), ``real_abs``, ``real_vs_headroom`` (B: the largest
    ``|dSigma_ii| / h_i`` over orbitals and frequencies, with ``(w, orbital)`` of the binding point in
    ``real_binding``), the relative G changes per axis and the worst ``Im Sigma_ii`` of both runs.
    """
    out = {}
    out["mats_abs"] = float(np.max(np.abs(new["sigma"] - ref["sigma"])))
    d_real = np.abs(np.diagonal(new["sigma_real"] - ref["sigma_real"], axis1=1, axis2=2))
    out["real_abs"] = float(np.max(np.abs(new["sigma_real"] - ref["sigma_real"])))
    h = headroom(ref["sigma_real"], tol)
    ratio = d_real / np.where(h > 0, h, np.nan)
    ratio = np.where(np.isfinite(ratio), ratio, np.inf)
    i = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    out["real_vs_headroom"] = float(ratio[i])
    out["real_binding"] = (int(i[0]), int(i[1]))
    out["artefact_points"] = int(
        np.sum(
            h
            < tol
            * np.max(np.abs(np.imag(np.diagonal(ref["sigma_real"], axis1=1, axis2=2))), axis=0, keepdims=True)
            * (1 - 1e-12)
        )
    )
    for key, name in (("gs_realaxis", "g_real_rel"), ("gs_matsubara", "g_mats_rel")):
        out[name] = float(np.max(np.abs(new[key] - ref[key])) / max(np.max(np.abs(ref[key])), np.finfo(float).tiny))
    for label, res in (("ref", ref), ("new", new)):
        im = np.imag(np.diagonal(res["sigma_real"], axis1=1, axis2=2))
        out[f"worst_im_{label}"] = float(np.max(im))
    return out


def grade_zero(new, tol=1e-5):
    """Grade a ``U = 0`` result against its exact self-energy, which is identically zero.

    There is no reference run: ``Sigma = G0^-1 - G^-1`` vanishes for a non-interacting model at the
    particle-hole symmetric point, so every nonzero value is truncation (or the solver's own noise floor, ~5e-6
    on a model that is not truncated at all -- compare against the uncapped run, not against zero).
    Returns ``mats_abs`` / ``real_abs`` (max |Sigma|), ``worst_im`` (max Im Sigma_ii on the real mesh, with
    ``(mesh point, orbital)`` in ``worst_im_at``) and ``positive_points``, the number of (point, orbital) pairs
    with Im Sigma_ii above ``tol`` (an acausal point).
    """
    im = np.imag(np.diagonal(np.asarray(new["sigma_real"]), axis1=1, axis2=2))
    i = np.unravel_index(int(np.argmax(im)), im.shape)
    return {
        "mats_abs": float(np.max(np.abs(new["sigma"]))),
        "real_abs": float(np.max(np.abs(new["sigma_real"]))),
        "worst_im": float(im[i]),
        "worst_im_at": (int(i[0]), int(i[1])),
        "positive_points": int(np.sum(im > tol)),
    }


def format_grade_zero(g):
    """One block of text for :func:`grade_zero`'s result."""
    return "\n".join(
        [
            f"U = 0 (exact Sigma = 0)    max|Sigma_mats| {g['mats_abs']:.3e}   max|Sigma_real| {g['real_abs']:.3e}",
            f"    max Im Sigma_ii real   {g['worst_im']:+.3e} at mesh point {g['worst_im_at'][0]}, "
            f"orbital {g['worst_im_at'][1]}   (points above tolerance: {g['positive_points']})",
        ]
    )


def format_grade(g):
    """One block of text for :func:`grade`'s result."""
    lines = [
        f"(A) max|dSigma_mats|           {g['mats_abs']:.3e}",
        f"    max|dSigma_real|           {g['real_abs']:.3e}",
        f"(B) max |dSigma_ii|/headroom   {g['real_vs_headroom']:.3e} at mesh point {g['real_binding'][0]}, "
        f"orbital {g['real_binding'][1]}   (reference points already inside the guard: {g['artefact_points']})",
        f"    max|dG|/max|G| real, mats  {g['g_real_rel']:.3e}, {g['g_mats_rel']:.3e}   (context, not gated)",
        f"    max Im Sigma_ii real       ref {g['worst_im_ref']:+.3e}   new {g['worst_im_new']:+.3e}",
    ]
    return "\n".join(lines)


def load_result(path):
    """An ``.npz`` written by :func:`run` as a plain dict of arrays."""
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def settle_options(mode):
    """``(gf_settle, gf_settle_ratio, text)`` for ``--freeze``: ``on`` (the options' defaults), ``off``, or a ratio."""
    if mode == "on":
        return None, None, "on"
    if mode == "off":
        return False, None, "off"
    ratio = float(mode)
    return None, ratio, f"ratio {ratio:g}"


class _UnitsDone(Exception):
    """Raised on every rank once the selected Green's-function units have been solved (``--units``)."""


def _solve_only(units, comm):
    """Wrap ``greens_function._gf_solve_units`` so that only ``units`` run, then report them and stop.

    ``units`` is a list of global unit indices, or ``"heaviest"`` for the ``comm.size`` heaviest by the cost
    weight (one per colour, as in production). Measurement aid: the self-energy is not assembled. Collective."""
    from impurityModel.ed import greens_function as gfm

    original = gfm._gf_solve_units

    def solve(plan, *args, subset=None, **kw):
        if subset is not None:
            return original(plan, *args, subset=subset, **kw)
        chosen = units
        if chosen == "heaviest":
            order = sorted(range(len(plan.unit_weights)), key=lambda u: -plan.unit_weights[u])
            chosen = sorted(order[: max(comm.size if comm is not None else 1, 1)])
        t0 = time.perf_counter()
        res = original(plan, *args, subset=list(chosen), **kw)
        wall = time.perf_counter() - t0
        if res is not None:
            for pos, u in enumerate(chosen):
                a, _b, _r, cap_stats, conv_stats = res[pos]
                print(
                    f"UNIT {u}: blocks {len(a)} retained {cap_stats.get('retained_size')} "
                    f"cap_hit {cap_stats.get('cap_hit')} memory_frozen {cap_stats.get('memory_frozen')} "
                    f"engine {cap_stats.get('engine')} converged {conv_stats.get('converged')}",
                    flush=True,
                )
                if cap_stats.get("refine") is not None:
                    print(f"UNIT {u} refine: {cap_stats['refine']}", flush=True)
            print(f"UNITS {list(chosen)} solved in {wall:.1f} s", flush=True)
        raise _UnitsDone

    gfm._gf_solve_units = solve


def cap_policy(cap, gs_cap=None, gf_cap=None):
    """The ``truncation_threshold`` a replay hands the driver: ``cap`` unless a separate GS / GF cap is named.

    ``gs_cap`` / ``gf_cap`` give a :class:`~impurityModel.ed.memory_estimate.CapPolicy`, an unset side being
    ``inf``. This is how a Green's-function cap that binds is pinned on a model whose ground state converges below
    it, with the ground state left alone -- ``cap`` alone pins both, which on a cap-bound ground state truncates
    it too. With both sides finite the caps are final (``from_memory=False``, the measured-RSS guards do not move
    them); with one side unlimited the policy keeps ``from_memory=True``, because ``CapPolicy`` has no
    "final but unlimited" state (``CapPolicy.coerce`` never makes one and the log line cannot describe it).
    """
    if gs_cap is None and gf_cap is None:
        return cap
    from impurityModel.ed.memory_estimate import CapPolicy

    inf = float("inf")
    return CapPolicy(
        gs=inf if gs_cap is None else gs_cap,
        gf=inf if gf_cap is None else gf_cap,
        from_memory=gs_cap is None or gf_cap is None,
    )


def _solver_overrides(
    gf_settle, gf_settle_ratio, prune_tol, first_shell_tol, refine_rounds, refine_tol, refine_swap, growth_tol=None
):
    """``SolverOptions`` fields a TOML replay overrides: :func:`real_workload.build_options`'s rule, on built options.

    An explicit ``prune_tol`` replaces the settle (the sweep's option beats the input's record), so the settle
    fields are then not touched.
    """
    out = {}
    if prune_tol is None:
        if gf_settle is not None:
            out["gf_settle"] = gf_settle
        if gf_settle_ratio is not None:
            out["gf_settle_ratio"] = gf_settle_ratio
    else:
        out["gf_prune_tol"] = prune_tol
    for key, value in (
        ("gf_first_shell_tol", first_shell_tol),
        ("gf_growth_tol", growth_tol),
        ("gf_refine_rounds", refine_rounds),
        ("gf_refine_tol", refine_tol),
        ("gf_refine_swap", refine_swap),
    ):
        if value is not None:
            out[key] = value
    return out


def _toml_inputs(path, comm, verbosity, *, swm, cap, cost_budget, solver_overrides):
    """``(model, basis, solver, meshes, cluster_label)`` of a TOML input, with the replay's overrides applied.

    The input goes through the same ``load_input`` / ``build`` as ``impurityModel run``, so a synthetic model
    (:mod:`metal_models`) is solved exactly as a user's would be. ``cap`` replaces the input's
    ``truncation_threshold`` unless it is ``"archive"`` (the input's own value). Collective (``build`` is).
    """
    from dataclasses import replace

    from impurityModel.inputformat.build import build
    from impurityModel.inputformat.reader import load_input

    built = build(load_input(str(path), comm=comm), comm=comm, verbosity=verbosity)
    if built.meshes is None:
        raise SystemExit(f"{path}: not a self-energy input (no frequency meshes)")
    basis = built.basis
    if cap != "archive":
        basis = replace(basis, truncation_threshold=cap)
    if swm is not None:
        basis = replace(basis, slater_weight_min=float(swm))
    if cost_budget is not None:
        basis = replace(basis, cost_budget=cost_budget)
    solver = replace(built.solver, **solver_overrides) if solver_overrides else built.solver
    return built.model, basis, solver, built.meshes, built.extra.get("cluster_label", "cluster")


def run(
    h5_path,
    out,
    iteration=1,
    cluster=None,
    swm=None,
    freeze="on",
    comm=None,
    verbosity=1,
    matrix_mode="auto",
    cap="archive",
    units=None,
    prune_tol=None,
    first_shell_tol=None,
    prune_rule=None,
    cost_budget=None,
    refine_rounds=None,
    refine_tol=None,
    refine_swap=None,
    refine_stop_rtol=None,
    refine_max_shifts=None,
    refine_weighted=None,
    gs_cap=None,
    gf_cap=None,
    growth_tol=None,
):
    """Replay one archived call and write ``out + '.npz'`` on rank 0. Returns the result dict (rank 0).

    ``cap`` pins ``truncation_threshold`` (an integer; the log's ``determinant cap`` line then says "set by you");
    ``"archive"`` keeps the recorded setting. ``gs_cap`` / ``gf_cap`` pin the ground-state and the Green's-function
    determinant caps *separately* (an unset side is uncapped, see :func:`cap_policy`) and override ``cap``: a model
    whose ground state needs the cap cannot otherwise be given a Green's-function cap that binds. ``h5_path`` may be
    a TOML input instead of an archive (:func:`_toml_inputs`); ``iteration`` and ``cluster`` are then ignored.
    ``units`` (a list of global unit indices, or ``"heaviest"``) solves
    only those Green's-function units and returns ``None`` after printing one ``UNIT`` line each.

    ``prune_tol`` / ``first_shell_tol`` run pruned growth (``gf_prune_tol`` / ``gf_first_shell_tol``, replacing
    the settle) and ``prune_rule`` (``eta`` or ``mass``) sets ``gf_recurrence._PRUNE_RULE``, identically on
    every rank. ``refine_rounds`` / ``refine_tol`` / ``refine_swap`` swap-refine the units the cap froze;
    ``refine_stop_rtol`` / ``refine_max_shifts`` / ``refine_weighted`` set ``gf_refinement.REFINE_STOP_RTOL`` /
    ``REFINE_MAX_SHIFTS`` / ``REFINE_WEIGHTED`` (the residual the recurrence-fed scoring solutions must reach, the
    number of scoring shifts per block, and whether each shift's scores are weighted by ``|G0^-1|^2``), identically
    on every rank. ``growth_tol`` (``gf_growth_tol``, needs pruning) stops the pruned growth pass at that looser
    tolerance; the restart on the final support converges fully.
    """
    if prune_rule is not None:
        from impurityModel.ed import gf_recurrence

        gf_recurrence._PRUNE_RULE = prune_rule
    from impurityModel.ed import selfenergy

    selfenergy.DEFAULT_CAUSALITY_TOL = 1e9  # non-fatal: the Sigma comes out whatever it is
    from impurityModel.test.support import real_workload as rw

    gf_settle, gf_settle_ratio, mode = settle_options(freeze)
    from impurityModel.ed import gf_solvers

    gf_solvers._MATRIX_MODE = matrix_mode  # harness switch, identical on every rank
    from impurityModel.ed import gf_refinement

    for name, value in (
        ("REFINE_STOP_RTOL", refine_stop_rtol),
        ("REFINE_MAX_SHIFTS", refine_max_shifts),
        ("REFINE_WEIGHTED", refine_weighted),
    ):
        if value is not None:
            setattr(gf_refinement, name, value)
    cap = cap_policy(cap, gs_cap, gf_cap)
    if (gs_cap is not None or gf_cap is not None) and (comm is None or comm.rank == 0):
        # The driver's own `determinant cap:` lines call a pinned side "auto" when the other side is unlimited
        # (CapPolicy has no "final but unlimited"), so say here what was pinned.
        print(
            f"REPLAY CAPS: GS {gs_cap or 'unlimited'}, GF {gf_cap or 'unlimited'} (pinned by --gs-cap/--gf-cap)",
            flush=True,
        )
    toml = str(h5_path).endswith(".toml")
    if toml:
        inputs = _toml_inputs(
            h5_path,
            comm,
            verbosity,
            swm=swm,
            cap=cap,
            cost_budget=cost_budget,
            solver_overrides=_solver_overrides(
                gf_settle,
                gf_settle_ratio,
                prune_tol,
                first_shell_tol,
                refine_rounds,
                refine_tol,
                refine_swap,
                growth_tol,
            ),
        )
        w_mesh, swm_used = inputs[3].w, inputs[1].slater_weight_min
    else:
        wl = rw.load_workload(h5_path, cluster=cluster, iteration=iteration)
        if wl["ignored_attrs"]:
            raise SystemExit(f"archive records options this replay does not consume: {wl['ignored_attrs']}")
        if swm is not None:
            wl["slaterWeightMin"] = float(swm)
        w_mesh, swm_used = wl["w_mesh"], wl["slaterWeightMin"]
    if units is not None:
        _solve_only(units, comm)
    t0 = time.perf_counter()
    try:
        if toml:
            from impurityModel.ed.selfenergy import calc_selfenergy

            model, basis, solver, meshes, label = inputs
            res = calc_selfenergy(model, meshes, basis, solver, comm=comm, verbosity=verbosity, cluster_label=label)
        else:
            res = rw.run_selfenergy(
                wl,
                comm=comm,
                verbosity=verbosity,
                truncation_threshold=cap,
                gf_tol="archive",
                gf_real_tol="archive",
                gf_settle=gf_settle,
                gf_settle_ratio=gf_settle_ratio,
                gf_prune_tol=prune_tol,
                gf_first_shell_tol=first_shell_tol,
                gf_growth_tol=growth_tol,
                cost_budget=cost_budget,
                gf_refine_rounds=refine_rounds,
                gf_refine_tol=refine_tol,
                gf_refine_swap=refine_swap,
            )
    except _UnitsDone:
        return None
    wall = time.perf_counter() - t0
    if comm is not None and comm.rank != 0:
        return None
    arrays = {}
    for key, value in res.items():
        try:
            arr = np.asarray(value)
        except Exception:  # non-array result entries are simply not saved
            continue
        if arr.dtype != object:
            arrays[key] = arr
    np.savez(
        out + ".npz",
        w=w_mesh,
        freeze=np.array(mode),
        swm=np.array(swm_used),
        wall=np.array(wall),
        **arrays,
    )
    print(f"RESULT {out}: swm {swm_used:g} freeze {mode} wall {wall:.1f} s", flush=True)
    return arrays


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("archive", help="an impurityModel_data.h5 archive, or a self-energy TOML input")
    r.add_argument("--iteration", type=int, default=1)
    r.add_argument("--cluster", default=None)
    r.add_argument("--swm", type=float, default=None)
    r.add_argument("--freeze", default="on")
    r.add_argument("--matrix-mode", default="auto", choices=["auto", "discovered", "retained", "none"])
    r.add_argument("--cap", default="archive", help="pin truncation_threshold to this integer")
    r.add_argument("--gs-cap", default=None, type=int, help="ground-state determinant cap (separate from --gf-cap)")
    r.add_argument("--gf-cap", default=None, type=int, help="Green's-function unit cap (separate from --gs-cap)")
    r.add_argument("--units", default=None, help="comma list of GF unit indices, or 'heaviest': solve only those")
    r.add_argument("--prune-tol", type=float, default=None, help="gf_prune_tol: pruned growth (replaces the settle)")
    r.add_argument("--refine-rounds", type=int, default=None, help="gf_refine_rounds: swap refinement of capped units")
    r.add_argument("--refine-tol", type=float, default=None, help="gf_refine_tol (needs --refine-rounds)")
    r.add_argument("--refine-swap", type=float, default=None, help="gf_refine_swap (needs --refine-rounds)")
    r.add_argument(
        "--refine-stop-rtol", type=float, default=None, help="residual the scoring solutions must reach (default 1e-4)"
    )
    r.add_argument("--refine-max-shifts", type=int, default=None, help="scoring shifts per block (default 8)")
    r.add_argument(
        "--refine-weighted",
        action="store_true",
        default=None,
        help="weight each scoring shift's scores by |G0^-1|^2 (default off)",
    )
    r.add_argument("--cost-budget", type=float, default=None, help="BasisOptions.cost_budget (alpha); off when absent")
    r.add_argument("--first-shell-tol", type=float, default=None, help="gf_first_shell_tol (needs --prune-tol)")
    r.add_argument("--growth-tol", type=float, default=None, help="gf_growth_tol (needs --prune-tol)")
    r.add_argument("--prune-rule", default=None, choices=["eta", "mass"], help="prune rule for the sweep")
    r.add_argument("--out", required=True)
    r.add_argument("--verbosity", type=int, default=1)
    g = sub.add_parser("grade")
    g.add_argument("ref", help="a reference .npz, or 'zero' for a U = 0 run whose exact Sigma is 0")
    g.add_argument("new")
    g.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args(argv)
    if a.cmd == "grade":
        if a.ref == "zero":
            print(format_grade_zero(grade_zero(load_result(a.new))))
            return 0
        print(format_grade(grade(load_result(a.ref), load_result(a.new), a.tol)))
        return 0
    from mpi4py import MPI

    cap = a.cap if a.cap == "archive" else int(a.cap)
    units = a.units
    if units is not None and units != "heaviest":
        units = [int(u) for u in units.split(",")]
    run(
        a.archive,
        a.out,
        a.iteration,
        a.cluster,
        a.swm,
        a.freeze,
        MPI.COMM_WORLD,
        a.verbosity,
        a.matrix_mode,
        cap,
        units,
        a.prune_tol,
        a.first_shell_tol,
        a.prune_rule,
        a.cost_budget,
        a.refine_rounds,
        a.refine_tol,
        a.refine_swap,
        a.refine_stop_rtol,
        a.refine_max_shifts,
        a.refine_weighted,
        a.gs_cap,
        a.gf_cap,
        a.growth_tol,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
