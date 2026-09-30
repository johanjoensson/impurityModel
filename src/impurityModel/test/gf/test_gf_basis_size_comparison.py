"""The basis-size comparison: smoke tests of the machinery, and the opt-in full-size benchmark.

What these tests do and do not claim (``doc/plans/gf_basis_size_comparison.md``):

* the machinery tests (always run, seconds) pin that a cell measures what it says -- every method
  reproduces the exact reference when uncapped, a binding cap is reported and costs accuracy, and
  the Pareto summary keeps only points that improve on every smaller basis;
* the F-pos test is the **positive control**: a model built so that most of the closure is
  determinants G does not need. If importance-ranked admission cannot shrink the basis *there* the
  harness, not the physics, is at fault;
* the benchmark (``RUN_BASIS_SIZE_BENCH=1``) prints the tables and does not assert what they say.
  Whether per-frequency solves need fewer determinants than Lanczos is the question it exists to
  answer, and a test that pinned the answer would only pin the first guess.
"""

import json
import os

import numpy as np
import pytest

from impurityModel.test.support.aim_fixtures import build_nio_like
from impurityModel.test.support.basis_size_harness import (
    METHODS,
    Mesh,
    format_report,
    pareto,
    run_cell,
    run_grid,
    smallest_size_below,
)

DELTA = 0.3
OMEGA = np.linspace(-30.0, 10.0, 161)
MATSUBARA = 1j * np.pi * 0.5 * (2 * np.arange(12) + 1)


@pytest.fixture(scope="module")
def small():
    aim = build_nio_like(3, target_d9L=0.15)
    return aim, Mesh(aim, OMEGA, MATSUBARA, DELTA, n_real=8, n_mats=4)


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    # The harness sets every knob it depends on; an exported one would leak into the cells it does not set.
    for name in (
        "GF_BICGSTAB_ADMISSION",
        "GF_BICGSTAB_ADMIT_TOL_AMP",
        "GF_BICGSTAB_ADMIT_SCORER",
        "GF_BICGSTAB_ADMIT_ROUNDS",
        "GF_BICGSTAB_ADMIT_SHELLS",
        "GF_BICGSTAB_ADMIT_EN_TOL",
        "GF_BICGSTAB_WARM_HISTORY",
        "GF_LANCZOS_ADMIT_TOL",
        "GF_ADMIT_FIRST_SHELL_TOL",
        "GF_APPLY_ROW_CHUNKS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("method", METHODS)
def test_uncapped_every_method_reproduces_the_exact_reference(small, method):
    aim, mesh = small
    cell = run_cell(aim, mesh, method)
    assert cell["size"] is not None and cell["unconverged"] == 0
    assert cell["dG_mats"] < 1e-8 and cell["dG_real"] < 1e-8
    assert cell["dSigma"] < 1e-7


def test_every_method_needs_the_same_uncapped_basis(small):
    """Uncapped, all five reach the whole connectivity closure of the seeds."""
    aim, mesh = small
    sizes = {m: run_cell(aim, mesh, m)["size"] for m in METHODS}
    assert len(set(sizes.values())) == 1, sizes


@pytest.mark.parametrize("method", ["lanczos-cap", "bicgstab-cap"])
def test_a_binding_cap_is_reported_and_costs_accuracy(small, method):
    aim, mesh = small
    full = run_cell(aim, mesh, method)
    cell = run_cell(aim, mesh, method, cap=int(0.6 * full["size"]))
    assert cell["cap_hit"] and cell["size"] < full["size"]
    assert cell["dSigma"] > 1e-4 > full["dSigma"]


def test_a_cap_below_the_seed_support_freezes_on_the_seeds(small):
    """The seed-saturation regime (prior P3): no cap below the seed support can shrink the basis below it."""
    aim, mesh = small
    cell = run_cell(aim, mesh, "lanczos-cap", cap=10)
    assert cell["cap_hit"] and cell["size"] == cell["seed_size"]


def test_a_singular_G_is_a_failed_cell_not_a_crash_or_a_small_error(small):
    """An amplitude cutoff above the seed amplitudes removes the seeds: G is singular, Sigma undefined."""
    aim, mesh = small
    cell = run_cell(aim, mesh, "bicgstab-swm", eta=0.5)
    assert np.isinf(cell["dSigma_mats"]) or cell["dSigma_mats"] > 1e-2


def test_pareto_keeps_only_points_that_improve_on_every_smaller_basis():
    cells = [
        {"method": "m", "size": 100, "dSigma": 1e-1, "unconverged": 0},
        {"method": "m", "size": 200, "dSigma": 2e-1, "unconverged": 0},  # larger and worse: dominated
        {"method": "m", "size": 300, "dSigma": 1e-3, "unconverged": 0},
        {"method": "m", "size": 50, "dSigma": 5e-2, "unconverged": 1},  # unconverged: not a measurement
        {"method": "m", "size": None, "dSigma": 1e-9, "unconverged": 0},  # size unknown: not a measurement
    ]
    front = pareto(cells)["m"]
    assert front == [(100, 1e-1), (300, 1e-3)]
    assert smallest_size_below(front, 5e-2) == 300
    assert smallest_size_below(front, 1e-6) is None


# --- the positive control ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spectators():
    """Two spectator levels per spin-orbital, coupled at 1e-5: the closure is ~2x what G needs."""
    aim = build_nio_like(5, target_d9L=0.05, n_spectator=2)
    return aim, Mesh(aim, OMEGA, MATSUBARA, DELTA, n_real=8, n_mats=4)


@pytest.mark.parametrize("method", ["bicgstab-outer", "lanczos-pruned"])
def test_importance_admission_excludes_the_spectator_determinants(spectators, method):
    """The positive control. With the first shell kept whole the spectator-hole rows are in the start
    set (a coupling of 1e-5 still reaches the determinant), so nothing can be excluded; relaxing the
    first-shell cut is what lets the admission see that they do not matter."""
    aim, mesh = spectators
    closure = run_cell(aim, mesh, "lanczos-cap")["size"]
    whole = run_cell(aim, mesh, method, eta=1e-3)
    assert whole["size"] == closure, "the strict first shell should hold every spectator row"
    relaxed = run_cell(aim, mesh, method, eta=1e-3, shell_tol=1e-3)
    assert relaxed["size"] < 0.6 * closure
    assert relaxed["dSigma_mats"] < 1e-3


def test_an_equal_cap_tells_the_two_methods_apart_or_the_prior_is_recorded(small):
    """Prior P1 says capped Lanczos and capped BiCGSTAB keep the same basis for the same answer. The
    harness records the ratio rather than asserting it, because the measurement is the point; this
    only pins that both cells exist, are capped, and report errors at the same cap."""
    aim, mesh = small
    full = run_cell(aim, mesh, "lanczos-cap")["size"]
    cap = int(0.75 * full)
    lan, bic = run_cell(aim, mesh, "lanczos-cap", cap=cap), run_cell(aim, mesh, "bicgstab-cap", cap=cap)
    assert lan["cap_hit"] and bic["cap_hit"]
    assert lan["size"] <= cap and bic["size"] <= cap
    assert np.isfinite(lan["dSigma"]) and np.isfinite(bic["dSigma"])


# --- the benchmark -------------------------------------------------------------------------------

RUN = os.environ.get("RUN_BASIS_SIZE_BENCH", "0") not in ("0", "", "false", "False")


@pytest.mark.benchmark
@pytest.mark.skipif(not RUN, reason="Set RUN_BASIS_SIZE_BENCH=1 to run the full-size comparison.")
@pytest.mark.parametrize("fixture", ["F-NiO", "F-pos"])
def test_full_size_comparison(fixture):
    """Env: ``BENCH_N_B`` (bath levels per spin-orbital, default 9), ``BENCH_WEIGHTS`` (d9L targets,
    default ``0.05,0.15,0.27``), ``BENCH_DELTAS`` (broadenings, default ``0.06,0.4``: the NiO and
    FCC Ni production ratios delta/W = 0.02 and 0.13 of the W = 3 bath), ``BENCH_OUT`` (JSON path)."""
    n_b = int(os.environ.get("BENCH_N_B", "9"))
    weights = [float(w) for w in os.environ.get("BENCH_WEIGHTS", "0.05,0.15,0.27").split(",")]
    deltas = [float(d) for d in os.environ.get("BENCH_DELTAS", "0.06,0.4").split(",")]
    dump = {}
    for weight in weights if fixture == "F-NiO" else [0.05]:
        aim = build_nio_like(n_b, target_d9L=weight, n_spectator=2 if fixture == "F-pos" else 0)
        for delta in deltas:
            # Mesh spacing <= delta/2 resolves the Lorentzians (the trapezoid error is ~exp(-2 pi delta/h)).
            omega = np.arange(-40.0, 10.0, delta / 2)
            mesh = Mesh(aim, omega, MATSUBARA, delta, n_real=12, n_mats=6)
            title = f"{fixture} n_b={n_b} d9L={weight:.2f} v_eff={aim.params['v_eff']:.3f} delta={delta}"
            result = run_grid(
                aim,
                mesh,
                progress=lambda c: print(
                    f"   {c['method']:15s} cap={c['cap']:>10} eta={c['eta']:<8} size={c['size']}", flush=True
                ),
            )
            print("\n" + format_report(title, result), flush=True)
            dump[title] = result
    if os.environ.get("BENCH_OUT"):
        with open(os.environ["BENCH_OUT"], "w") as f:
            json.dump(dump, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
