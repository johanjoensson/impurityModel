"""Phase-2 double-counting baseline: what does a fixed-XX DC search actually cost?

The fixed-occupation / fixed-peak double counting is the self-consistent way to run wide-window
p-d models, and today it is too slow to use: one NiO 15-bath ``find_ground_state_basis`` is
~2000 s and a full occupation bracket is hours, per cluster per CSC iteration. Before any of that
is made faster there has to be a baseline, and it has to be measured at **more than one
determinant cap**.

That last point is the whole design of this module. The cap decides which determinants survive
the CIPSI truncation, and therefore decides the impurity occupation ``n`` the search is driving
onto its target -- not just how long the search takes. A speedup demonstrated at one cheap cap
does not transfer to production unless ``n`` is stable across the ladder, so every row here
carries **achieved ``n`` and ``mu`` next to the seconds**. If ``n`` drifts, that is the finding,
and it invalidates cap-2000 measurements of everything Phases 4-6 do; the scaling exponent is
the secondary product.

The accounting comes from :mod:`impurityModel.ed.solver_trace`, which the production search
writes into unconditionally -- so what is measured here is the production path, not a
reimplementation of it. The workloads are the real archived RSPt runs shared with
:mod:`restriction_diagnostics`.

Run as an opt-in pytest module (one workload per process for honest timings)::

    RUN_DC_DIAG=1 DC_DIAG_WORKLOAD=nio_15 DC_DIAG_CAPS=2000,8000,32000 \\
        pytest -s -m benchmark src/impurityModel/test/support/dc_diagnostics.py

or directly::

    python -m impurityModel.test.support.dc_diagnostics nio_15 2000 8000 32000

A second mode, ``occupation_convergence`` (B2), answers a different question: is the achieved
occupation converged with respect to the ED's own variational truncation, at a FIXED mu -- not
whether the search's answer is stable across caps. It needs no search (`mu` is given, not solved
for), so it is affordable across a full (cap, excitation_budget, chain_restrict) grid where the
search-based ladder above is not::

    RUN_DC_DIAG=1 DC_DIAG_MODE=occupation_convergence DC_DIAG_WORKLOAD=nio_15 DC_DIAG_MU=0.41 \\
        DC_DIAG_CAPS=2000,4000,8000,16000 pytest -s -m benchmark src/impurityModel/test/support/dc_diagnostics.py

or directly::

    DC_DIAG_MODE=occupation_convergence DC_DIAG_MU=0.41 \\
        python -m impurityModel.test.support.dc_diagnostics nio_15 2000 4000 8000 16000
"""

import os
from dataclasses import replace
from time import perf_counter

import numpy as np

from impurityModel.ed import config, solver_trace
from impurityModel.ed.memory_estimate import suggest_truncation_threshold
from impurityModel.ed.model import load_selfenergy_archive
from impurityModel.ed.selfenergy import fixed_occupation_dc, fixed_peak_dc, occupation_and_energy_at_mu
from impurityModel.test.support.restriction_diagnostics import DEFAULT_ITERATION, WORKLOADS

DEFAULT_CAPS = (2000, 4000, 8000)
DEFAULT_CONVERGENCE_CAPS = (2000, 4000, 8000, 16000)
DEFAULT_EXCITATION_BUDGETS = (3, 4, 5)
DEFAULT_CHAIN_RESTRICTS = (True, False)


def _uniform_shift(dc_found, dc_guess):
    """The scalar ``mu`` in ``dc = dc_guess + mu * identity``, read back off the result.

    Both searches return the full matrix, not ``mu``; the shift is what the diagnostics compare
    across caps, and taking it from the diagonal (rather than plumbing a second return value
    through the production signature) keeps this module read-only with respect to the solver.

    The archived NiO double counting is a full matrix, not a scalar, so the difference is
    *asserted* uniform before a single number is taken from it -- averaging a non-uniform
    diagonal would silently produce a shift the search never evaluated, and the achieved value
    is then looked up at the wrong trial point.
    """
    guess = np.zeros_like(dc_found) if dc_guess is None else np.asarray(dc_guess, dtype=complex)
    shift = dc_found - guess
    identity = np.identity(shift.shape[0])
    if not np.allclose(shift, shift[0, 0] * identity, atol=1e-10):
        raise AssertionError(f"the search returned a non-uniform shift, which it cannot do:\n{shift}")
    return float(np.real(shift[0, 0]))


def run_dc_search(
    workload_key,
    cap,
    criterion="occupation",
    peak_position=None,
    comm=None,
    verbosity=0,
    iteration=DEFAULT_ITERATION,
):
    """One double-counting search at one determinant cap; return its cost accounting.

    Parameters
    ----------
    workload_key : str
        A key of :data:`restriction_diagnostics.WORKLOADS`.
    cap : int or float
        ``BasisOptions.truncation_threshold`` for this run. The scaling parameter.
    iteration : int
        Which CSC iteration of the archive to load. Pinned rather than left at the loader's
        ``last`` default: the iterations of one run can differ by several eV, and ``nio_15``'s
        last iteration is a runaway whose DFT reference occupation is 1.54 against a nominal 8
        (iteration 1: 8.6258). Benchmarking the newest iterate silently benchmarks a diverging
        one. The loaded label is reported in the table.
    criterion : {"occupation", "peak"}
        Which search to baseline. ``"occupation"`` runs self-consistently (target = the DFT
        reference filling), which is the production CSC path.
    peak_position : float, optional
        Required for ``criterion="peak"``.

    Returns
    -------
    dict
        ``cap``, ``seconds``, ``mu``, ``value`` (achieved ``n`` or gap), ``chi``, and the
        per-kind counts and seconds the trace recorded.
    """
    from impurityModel.ed.dc_search import _dc_chi, bracket_width_tol

    # This harness derives its whole result from a trace it opens itself, and DC_DIAGNOSTICS makes
    # the search open one too. They are mutually exclusive by construction -- solver_trace refuses
    # to nest -- so say so here rather than letting the search raise mid-run. Historically this
    # combination silently emptied the outer trace: every row came back `value = nan` and the
    # STABLE/DRIFTS verdict, the module's primary product, was dropped without a word.
    if config.DC_DIAGNOSTICS.get():
        raise RuntimeError(
            "DC_DIAGNOSTICS is set, which makes the search open its own solver_trace block; this "
            "benchmark opens one around the search and reads its counts from it. Unset "
            "DC_DIAGNOSTICS to run the benchmark (its table already reports the same per-kind "
            "seconds), or run the search directly to get the per-mu report."
        )

    model, _meshes, basis, solver, label = load_selfenergy_archive(WORKLOADS[workload_key], iteration=iteration)
    basis = replace(basis, truncation_threshold=cap)
    dc_guess = model.dc

    with solver_trace.tracing() as trace:
        start = perf_counter()
        if criterion == "occupation":
            dc = fixed_occupation_dc(model=model, basis=basis, solver=solver, comm=comm, verbosity=verbosity)
        elif criterion == "peak":
            if peak_position is None:
                raise ValueError('criterion="peak" needs a peak_position.')
            dc = fixed_peak_dc(
                model=model,
                basis=basis,
                solver=solver,
                peak_position=peak_position,
                comm=comm,
                verbosity=verbosity,
                # This harness measures search cost, not charge-state fidelity; a real workload's
                # peak criterion landing on a non-nominal sector is not a benchmark failure.
                allow_charge_state_change=True,
            )
        else:
            raise ValueError(f"unknown criterion {criterion!r}")
        seconds = perf_counter() - start

    from impurityModel.ed.lie_algebra import extract_tensors

    guess_dense = extract_tensors(dc_guess or {}, n_orb=dc.shape[0], two_body=False)[0]
    mu = _uniform_shift(dc, guess_dense)

    evaluations = trace.of_kind("dc_evaluation")
    field = "n" if any("n" in event for event in evaluations) else "gap"
    samples = {event["mu"]: event[field] for event in evaluations if field in event and "mu" in event}
    # The achieved value is the one at the shift actually returned; the search always evaluated
    # it (mu is either the mu=0 fast path, a direct scan hit, or a refined bracket point), but
    # match on the closest recorded mu rather than on equality, since mu came back through a
    # dense matrix diagonal.
    achieved = samples[min(samples, key=lambda m: abs(m - mu))] if samples else float("nan")

    return {
        "workload": workload_key,
        "label": label,
        "criterion": criterion,
        "cap": cap,
        # What production would actually have used: the archive records truncation_threshold=None,
        # which means the cap is derived from available per-rank memory at run time. The ladder
        # is an extrapolation and this is the point it extrapolates *to*; without it the exponent
        # has no destination.
        "production_cap": suggest_truncation_threshold(model.n_spin_orbitals, comm=comm),
        "seconds": seconds,
        "mu": mu,
        "value": achieved,
        # Same width_tol both criteria pass to _solve_dc_shift internally (dc_search.bracket_width_tol),
        # so a collapsed bracket at a sector boundary is excluded here exactly as it is in the
        # search's own closing report, rather than read as a slope.
        "chi": _dc_chi(samples, width_tol=bracket_width_tol(basis.tau)),
        "evaluations": trace.count("dc_evaluation"),
        "sector_solves": trace.count("sector_solve"),
        "cache_hits": trace.count("sector_cache_hit"),
        "build_s": trace.seconds("build"),
        "expand_s": trace.seconds("expand"),
        "eigensolve_s": trace.seconds("eigensolve"),
        "max_dets": max((event.get("n_dets", 0) for event in trace.of_kind("sector_solve")), default=0),
    }


def _scaling_exponent(rows):
    r"""Least-squares :math:`p` in ``seconds ~ cap**p`` over the measured ladder.

    Needs at least two caps -- which is the point of running a ladder rather than a single
    baseline. ``None`` when there are fewer, or when a row is degenerate (zero seconds).
    """
    usable = [row for row in rows if row["cap"] > 0 and row["seconds"] > 0 and np.isfinite(row["cap"])]
    if len(usable) < 2:
        return None
    log_cap = np.log([row["cap"] for row in usable])
    log_seconds = np.log([row["seconds"] for row in usable])
    return float(np.polyfit(log_cap, log_seconds, 1)[0])


def cap_ladder(
    workload_key,
    caps=DEFAULT_CAPS,
    criterion="occupation",
    peak_position=None,
    comm=None,
    verbosity=0,
    iteration=DEFAULT_ITERATION,
):
    """Baseline one workload across a ladder of determinant caps and print the table."""
    rank = comm.rank if comm is not None else 0
    rows = []
    for cap in caps:
        row = run_dc_search(
            workload_key,
            cap,
            criterion=criterion,
            peak_position=peak_position,
            comm=comm,
            verbosity=verbosity,
            iteration=iteration,
        )
        rows.append(row)
        if rank == 0:
            # Print as we go: the top of the ladder can take hours, and a run killed by a wall
            # clock limit should still have published everything below it.
            print(f"[dc-baseline] {_format_row(row)}", flush=True)
    if rank == 0:
        print_ladder(rows)
    return rows


# (header, width, row format) per column. The header and the row share the width, so a column
# added here lines up by construction.
_COLUMNS = (
    ("cap", 9, "{cap:>9}"),
    ("seconds", 10, "{seconds:>10.1f}"),
    ("evals", 6, "{evaluations:>6}"),
    ("solves", 7, "{sector_solves:>7}"),
    ("hits", 6, "{cache_hits:>6}"),
    ("dets", 8, "{max_dets:>8}"),
    ("build_s", 9, "{build_s:>9.1f}"),
    ("expand_s", 10, "{expand_s:>10.1f}"),
    ("eigen_s", 9, "{eigensolve_s:>9.1f}"),
    ("mu", 10, "{mu:>10.6f}"),
    ("value", 10, "{value:>10.5f}"),
)


def _format_row(row):
    return " ".join(fmt.format(**row) for _, _, fmt in _COLUMNS)


def print_ladder(rows):
    """The per-cap table, the achieved-value drift across it, and the scaling exponent."""
    if not rows:
        return
    head = rows[0]
    # The label names the (cluster, iteration) actually loaded -- the one thing that made an
    # earlier baseline meaningless, since the loader's default follows the newest iterate.
    print(f"\n=== double-counting baseline: {head['workload']} [{head['label']}], {head['criterion']} ===")
    print(" ".join(f"{name:>{width}}" for name, width, _ in _COLUMNS))
    for row in rows:
        print(_format_row(row))

    values = [row["value"] for row in rows if np.isfinite(row["value"])]
    if len(values) >= 2:
        drift = max(values) - min(values)
        # occ_tol, the tolerance the search itself converges to, is the scale that decides
        # whether the cap ladder is measuring one physical answer or several.
        verdict = "STABLE" if drift <= 1e-2 else "DRIFTS -- low-cap speedups do not transfer"
        print(f"achieved value across the ladder: spread {drift:.4f} ({verdict})")
    mus = [row["mu"] for row in rows]
    if len(mus) >= 2:
        print(f"mu across the ladder: spread {max(mus) - min(mus):.6f}")
    exponent = _scaling_exponent(rows)
    if exponent is not None:
        print(f"scaling: seconds ~ cap**{exponent:.2f}")
        production_cap = head.get("production_cap")
        if production_cap:
            # Extrapolate from the largest measured cap, not from the fit's intercept: the fit is
            # two or three points wide and its intercept carries all of their noise.
            top = max(rows, key=lambda row: row["cap"])
            projected = top["seconds"] * (production_cap / top["cap"]) ** exponent
            print(
                f"production cap on this machine: {production_cap} determinants "
                f"-> projected {projected:.0f} s ({projected / 3600:.1f} h) per DC search"
            )
    chis = [row["chi"] for row in rows if row["chi"] is not None]
    if chis:
        print("chi = d(value)/dmu per cap: " + ", ".join(f"{chi:.4f}" for chi in chis))
    print("", flush=True)


# ---- B2: fixed-mu occupation/energy convergence sweep ---------------------------------
#
# The cap ladder above asks whether the *search* transfers across caps; it does not settle
# whether the occupation the ED reports at a converged mu is itself converged with respect to
# its own variational truncation. Both caps in the ladder ran the same 7 evaluations on a
# deterministic grid, so identical mu only shows both first crossed |g| <= tol at the same grid
# point -- it says nothing about whether the *achieved n* would still agree at a materially
# different truncation_threshold/excitation_budget/chain_restrict. If it does not, dc is
# absorbing the ED's own variational error, which is exactly the truncation-as-fitting-parameter
# pathology Haule (2015) warns about, and it would drift with the charge density across a CSC
# loop for reasons that have nothing to do with the physics.
#
# This needs no search at all, which is why it is affordable where the cap ladder was not:
# occupation_and_energy_at_mu evaluates ONE fixed mu, so the whole grid below costs
# len(caps) * len(excitation_budgets) * len(chain_restricts) solves, not search-many-times-that.


def run_occupation_at_fixed_mu(
    workload_key,
    mu,
    cap,
    excitation_budget,
    chain_restrict,
    comm=None,
    verbosity=0,
    iteration=DEFAULT_ITERATION,
):
    """One (cap, excitation_budget, chain_restrict) point of the B2 convergence grid at fixed mu.

    Returns a dict: ``cap``, ``excitation_budget``, ``chain_restrict``, ``n``, ``E0``, ``sector``,
    ``seconds``. ``sector`` is the walk's winning charge state (M1) -- worth watching alongside
    ``n``, since a converged thermal occupation can still hide the walk landing on a different
    sector at a different cap.
    """
    model, _meshes, basis, solver, label = load_selfenergy_archive(WORKLOADS[workload_key], iteration=iteration)
    trial_basis = replace(
        basis,
        truncation_threshold=cap,
        excitation_budget=excitation_budget,
        chain_restrict=chain_restrict,
    )
    start = perf_counter()
    n, e0, sector = occupation_and_energy_at_mu(model, trial_basis, solver, mu, comm=comm, verbosity=verbosity)
    seconds = perf_counter() - start
    return {
        "workload": workload_key,
        "label": label,
        "mu": mu,
        "cap": cap,
        "excitation_budget": excitation_budget,
        "chain_restrict": chain_restrict,
        "n": n,
        "E0": e0,
        "sector": sector,
        "seconds": seconds,
    }


_CONVERGENCE_COLUMNS = (
    ("cap", 9, "{cap:>9}"),
    ("budget", 7, "{excitation_budget:>7}"),
    ("chain_r", 8, "{chain_restrict!s:>8}"),
    ("seconds", 10, "{seconds:>10.1f}"),
    ("n", 12, "{n:>12.6f}"),
    ("E0", 14, "{E0:>14.6f}"),
    ("sector", 7, "{sector:>7}"),
)


def _format_convergence_row(row):
    return " ".join(fmt.format(**row) for _, _, fmt in _CONVERGENCE_COLUMNS)


def occupation_convergence_sweep(
    workload_key,
    mu,
    caps=DEFAULT_CONVERGENCE_CAPS,
    excitation_budgets=DEFAULT_EXCITATION_BUDGETS,
    chain_restricts=DEFAULT_CHAIN_RESTRICTS,
    comm=None,
    verbosity=0,
    iteration=DEFAULT_ITERATION,
    occ_tol=1e-2,
):
    """Evaluate ``n``/``E0`` at one fixed ``mu`` across a (cap, excitation_budget,
    chain_restrict) grid -- B2's answer to "is the achieved occupation converged with respect to
    the ED's own truncation, independent of whatever the search happened to do".

    Prints the grid table and, per (excitation_budget, chain_restrict) row-group, the spread of
    ``n`` over the top two caps -- the acceptance criterion the plan sets: converged if that
    spread is below ``occ_tol``.
    """
    rank = comm.rank if comm is not None else 0
    rows = []
    for excitation_budget in excitation_budgets:
        for chain_restrict in chain_restricts:
            for cap in caps:
                row = run_occupation_at_fixed_mu(
                    workload_key,
                    mu,
                    cap,
                    excitation_budget,
                    chain_restrict,
                    comm=comm,
                    verbosity=verbosity,
                    iteration=iteration,
                )
                rows.append(row)
                if rank == 0:
                    # Print as we go: the grid can take a long time, and a run killed by a wall
                    # clock limit should still have published everything below it.
                    print(f"[dc-occ-convergence] {_format_convergence_row(row)}", flush=True)
    if rank == 0:
        print_occupation_convergence(rows, occ_tol=occ_tol)
    return rows


def print_occupation_convergence(rows, occ_tol=1e-2):
    """The (cap, budget, chain_restrict) grid, and a converged/drifts verdict per group."""
    if not rows:
        return
    head = rows[0]
    print(f"\n=== occupation convergence: {head['workload']} [{head['label']}], mu={head['mu']:.6f} ===")
    print(" ".join(f"{name:>{width}}" for name, width, _ in _CONVERGENCE_COLUMNS))
    for row in rows:
        print(_format_convergence_row(row))

    groups = {}
    for row in rows:
        groups.setdefault((row["excitation_budget"], row["chain_restrict"]), []).append(row)

    print()
    for (excitation_budget, chain_restrict), group_rows in groups.items():
        by_cap = sorted(group_rows, key=lambda row: row["cap"])
        top_two = by_cap[-2:]
        if len(top_two) < 2:
            continue
        n_spread = abs(top_two[-1]["n"] - top_two[0]["n"])
        e0_spread = abs(top_two[-1]["E0"] - top_two[0]["E0"])
        verdict = "CONVERGED" if n_spread <= occ_tol else "DRIFTS -- dc would absorb this"
        sector_note = (
            ""
            if top_two[-1]["sector"] == top_two[0]["sector"]
            else f" -- SECTOR CHANGED ({top_two[0]['sector']} -> {top_two[-1]['sector']})"
        )
        print(
            f"excitation_budget={excitation_budget}, chain_restrict={chain_restrict}: "
            f"top-two-cap spread n={n_spread:.4f}, E0={e0_spread:.4f} ({verdict}){sector_note}"
        )
    print("", flush=True)


# ---- opt-in pytest entry -------------------------------------------------------------

RUN = os.environ.get("RUN_DC_DIAG") == "1"


def test_dc_baseline():
    import pytest

    if not RUN:
        pytest.skip("Set RUN_DC_DIAG=1 to run the double-counting baseline.")
    from mpi4py import MPI

    key = os.environ.get("DC_DIAG_WORKLOAD", "nio_20")
    mode = os.environ.get("DC_DIAG_MODE", "cap_ladder")
    verbosity = int(os.environ.get("DC_DIAG_VERBOSITY", "0"))
    iteration = int(os.environ.get("DC_DIAG_ITERATION", DEFAULT_ITERATION))

    if mode == "occupation_convergence":
        mu_env = os.environ.get("DC_DIAG_MU")
        if mu_env is None:
            raise RuntimeError("DC_DIAG_MODE=occupation_convergence needs DC_DIAG_MU set.")
        caps = [int(c) for c in os.environ.get("DC_DIAG_CAPS", "").split(",") if c.strip()] or list(
            DEFAULT_CONVERGENCE_CAPS
        )
        budgets = [
            int(b) for b in os.environ.get("DC_DIAG_EXCITATION_BUDGETS", "").split(",") if b.strip()
        ] or list(DEFAULT_EXCITATION_BUDGETS)
        occupation_convergence_sweep(
            key,
            float(mu_env),
            caps=caps,
            excitation_budgets=budgets,
            comm=MPI.COMM_WORLD,
            verbosity=verbosity,
            iteration=iteration,
        )
        return

    caps = [int(c) for c in os.environ.get("DC_DIAG_CAPS", "").split(",") if c.strip()] or list(DEFAULT_CAPS)
    cap_ladder(
        key,
        caps,
        criterion=os.environ.get("DC_DIAG_CRITERION", "occupation"),
        comm=MPI.COMM_WORLD,
        verbosity=verbosity,
        iteration=iteration,
    )


test_dc_baseline.benchmark = True  # type: ignore[attr-defined]  # pytest-benchmark marker


if __name__ == "__main__":
    import sys

    try:
        from mpi4py import MPI

        _comm = MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None
    except ImportError:
        _comm = None
    _key = sys.argv[1] if len(sys.argv) > 1 else "nio_20"
    _verbosity = int(os.environ.get("DC_DIAG_VERBOSITY", "0"))
    _iteration = int(os.environ.get("DC_DIAG_ITERATION", DEFAULT_ITERATION))

    if os.environ.get("DC_DIAG_MODE") == "occupation_convergence":
        _mu_env = os.environ.get("DC_DIAG_MU")
        if _mu_env is None:
            raise RuntimeError("DC_DIAG_MODE=occupation_convergence needs DC_DIAG_MU set.")
        _caps = [int(c) for c in sys.argv[2:]] or list(DEFAULT_CONVERGENCE_CAPS)
        occupation_convergence_sweep(
            _key,
            float(_mu_env),
            caps=_caps,
            comm=_comm,
            verbosity=_verbosity,
            iteration=_iteration,
        )
    else:
        _caps = [int(c) for c in sys.argv[2:]] or list(DEFAULT_CAPS)
        cap_ladder(
            _key,
            _caps,
            comm=_comm,
            verbosity=_verbosity,
            iteration=_iteration,
        )
