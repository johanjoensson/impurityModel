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

from impurityModel.test.support.aim_fixtures import build_nio_like, build_semicircle_siam, geometry_variants
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


# --- F-metal: the semicircular-bath SIAM with a truncated ground state ---------------------------------


@pytest.fixture(scope="module")
def metal():
    """A strongly hybridized half-filled SIAM, ground state cut to 40 of its 400 determinants (the way a
    CIPSI ground state is), so the seeds are a small part of a closure the exact state would fill."""
    aim = build_semicircle_siam(5, v=0.5, gs_keep=40)
    omega = np.arange(-1.5, 1.5, 0.05)
    mats = 1j * np.pi * 0.05 * (2 * np.arange(8) + 1)
    return aim, Mesh(aim, omega, mats, 0.1, n_real=6, n_mats=4)


@pytest.mark.parametrize("method", METHODS)
def test_metal_uncapped_every_method_reproduces_the_exact_reference(metal, method):
    aim, mesh = metal
    cell = run_cell(aim, mesh, method)
    assert cell["unconverged"] == 0 and cell["dSigma"] < 1e-6


def test_metal_seeds_are_a_small_fraction_of_the_closure(metal):
    """The regime the truncated ground state exists for: caps and thresholds have room below the closure."""
    aim, mesh = metal
    cell = run_cell(aim, mesh, "lanczos-cap")
    assert cell["seed_size"] <= 3 * 40 and cell["size"] > 4 * cell["seed_size"]


def test_a_single_axis_mesh_scores_only_that_axis(metal):
    aim, _ = metal
    mats_only = Mesh(aim, np.array([0.0]), 1j * np.pi * 0.05 * (2 * np.arange(8) + 1), 0.1, n_real=0, n_mats=4)
    cell = run_cell(aim, mats_only, "bicgstab-cap")
    assert cell["dSigma_mats"] is not None and cell["dSigma_real"] is None
    assert cell["dSigma"] == cell["dSigma_mats"] and cell["dSigma"] < 1e-6
    result = run_grid(aim, mats_only, cap_fractions=(0.3,), etas=(1e-2,), swm_etas=(), methods=("lanczos-cap",))
    assert list(result["pareto_by"]) == ["dSigma_mats"]
    assert "dSigma_real" not in format_report("x", result)


# --- the bath basis (F-geom) --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def geometries():
    """The small metal in four bath bases, each truncated to the same ground-state accuracy."""
    exact = build_semicircle_siam(5, v=0.5)
    return exact, geometry_variants(exact, lost_weight=1e-4)


def test_the_natural_basis_needs_far_fewer_ground_state_determinants_than_the_star_and_chain(geometries):
    _exact, variants = geometries
    kept = {name: v.keep for name, v in variants.items()}
    assert kept["natural"] < kept["star"] < kept["chain"], kept


@pytest.mark.parametrize("name", ["star", "chain", "natural", "natural-chains"])
def test_each_basis_reproduces_its_own_reference_and_stays_near_the_physical_answer(geometries, name):
    """Solver error against the cell's own seeds is round-off in every basis; the end-to-end error against
    the exact-ground-state truth is the truncation's, at the accuracy the weight cut allows."""
    exact, variants = geometries
    aim = variants[name]
    omega = np.arange(-1.5, 1.5, 0.1)
    mesh = Mesh(aim, omega, 1j * np.pi * 0.05 * (2 * np.arange(8) + 1), 0.1, n_real=6, n_mats=4, truth=exact)
    cell = run_cell(aim, mesh, "lanczos-cap")
    assert cell["dSigma"] < 1e-6
    assert cell["dSigma_truth_mats"] is not None and cell["dSigma_truth_mats"] < 0.5


def test_a_compact_ground_state_means_a_small_seed_support(geometries):
    """The lever: the same model, the same ground-state accuracy, a much smaller seed support."""
    _exact, variants = geometries
    omega = np.arange(-1.5, 1.5, 0.1)
    mats = 1j * np.pi * 0.05 * (2 * np.arange(8) + 1)
    seeds = {
        name: run_cell(v, Mesh(v, omega, mats, 0.1, n_real=0, n_mats=4), "lanczos-cap")["seed_size"]
        for name, v in variants.items()
    }
    assert seeds["natural"] < seeds["star"] < seeds["chain"], seeds


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


@pytest.mark.benchmark
@pytest.mark.skipif(not RUN, reason="Set RUN_BASIS_SIZE_BENCH=1 to run the full-size comparison.")
def test_full_size_metal():
    """The scaled-down ``examples/semicircular_siam`` (V = 2 D there; D = U = 0.5 Ry).

    Env: ``BENCH_METAL_N_B`` (odd bath levels per spin, default 7: a 7,840-determinant closure),
    ``BENCH_METAL_K`` (ground-state determinants kept, default 150), ``BENCH_METAL_V`` (hybridizations,
    default ``0.5,1.0,0.25``), ``BENCH_DELTAS`` (default ``0.02,0.13``: delta/W of 0.02 and 0.13 for the
    bandwidth W = 2 D = 1 Ry), ``BENCH_OUT``. The Matsubara axis does not depend on delta, so it is run
    once per V and the real axis once per delta; the JSON is rewritten after each configuration."""
    n_b = int(os.environ.get("BENCH_METAL_N_B", "7"))
    keep = int(os.environ.get("BENCH_METAL_K", "150"))
    couplings = [float(v) for v in os.environ.get("BENCH_METAL_V", "0.5,1.0,0.25").split(",")]
    deltas = [float(d) for d in os.environ.get("BENCH_DELTAS", "0.02,0.13").split(",")]
    matsubara = 1j * np.pi * 0.005 * (2 * np.arange(12) + 1)
    out, dump = os.environ.get("BENCH_OUT"), {}

    def progress(c):
        fmt = lambda x: "-" if x is None else f"{x:.1e}"  # noqa: E731
        print(
            f"   {c['method']:15s} cap={c['cap']:>10} eta={c['eta']:<8} size={c['size']} "
            f"mats={fmt(c['dSigma_mats'])} real={fmt(c['dSigma_real'])} {c['wall']:.0f}s",
            flush=True,
        )

    for v in couplings:
        aim = build_semicircle_siam(n_b, v=v, gs_keep=keep)
        label = f"F-metal n_b={n_b} V={v} K={keep} (e0 error {aim.e0 - aim.exact_e0:.1e})"
        runs = [("Matsubara", Mesh(aim, np.array([0.0]), matsubara, deltas[0], n_real=0, n_mats=6))]
        for delta in deltas:
            omega = np.arange(-1.5, 1.5, delta / 2)
            runs.append((f"real delta={delta}", Mesh(aim, omega, matsubara, delta, n_real=8, n_mats=0)))
        for axis, mesh in runs:
            title = f"{label} {axis}"
            result = run_grid(aim, mesh, swm_etas=(1e-4, 1e-5), progress=progress)
            print("\n" + format_report(title, result), flush=True)
            dump[title] = result
            if out:
                with open(out, "w") as f:
                    json.dump(dump, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


@pytest.mark.benchmark
@pytest.mark.skipif(not RUN, reason="Set RUN_BASIS_SIZE_BENCH=1 to run the full-size comparison.")
@pytest.mark.parametrize("fixture", ["F-NiO", "F-metal"])
def test_full_size_geometry(fixture):
    """The same model in four bath bases (star, chain, natural orbitals, natural orbitals re-chained).

    Each basis keeps the ground-state determinants *its own* basis needs for ``BENCH_LOST`` (default 1e-6) of
    discarded weight, so the ground-state accuracy is held fixed and the seed support is what the basis
    changes. ``dSigma`` is the solver's error against that truncated state; ``dSigma_truth`` the end-to-end
    error against the exact ground state (``G`` does not depend on the basis). Env: ``BENCH_LOST``,
    ``BENCH_GEOM_N_B`` (default 9 for F-NiO, 7 for F-metal), ``BENCH_METAL_V`` (default 0.5),
    ``BENCH_DELTAS`` (F-NiO ``0.06,0.4``, F-metal ``0.02,0.13``), ``BENCH_BASES`` (default
    ``star,chain,natural,natural-chains``; add ``linked-chain``, which needs ``rspt2spectra``), ``BENCH_OUT``."""
    lost = float(os.environ.get("BENCH_LOST", "1e-6"))
    wanted = os.environ.get("BENCH_BASES", "star,chain,natural,natural-chains").split(",")
    if fixture == "F-NiO":
        n_b = int(os.environ.get("BENCH_GEOM_N_B", "9"))
        exact = build_nio_like(n_b, target_d9L=0.15)
        deltas = [float(d) for d in os.environ.get("BENCH_DELTAS", "0.06,0.4").split(",")]
        matsubara, lo, hi, label = MATSUBARA, -40.0, 10.0, f"F-NiO n_b={n_b} d9L=0.15"
    else:
        n_b = int(os.environ.get("BENCH_GEOM_N_B", "7"))
        v = float(os.environ.get("BENCH_METAL_V", "0.5"))
        exact = build_semicircle_siam(n_b, v=v)
        deltas = [float(d) for d in os.environ.get("BENCH_DELTAS", "0.02,0.13").split(",")]
        matsubara, lo, hi, label = 1j * np.pi * 0.005 * (2 * np.arange(12) + 1), -1.5, 1.5, f"F-metal n_b={n_b} V={v}"
    out, dump = os.environ.get("BENCH_OUT"), {}

    def progress(c):
        fmt = lambda x: "-" if x is None else f"{x:.1e}"  # noqa: E731
        print(
            f"   {c['method']:15s} cap={c['cap']:>10} eta={c['eta']:<8} size={c['size']} "
            f"own={fmt(c['dSigma'])} truth_m={fmt(c['dSigma_truth_mats'])} truth_r={fmt(c['dSigma_truth_real'])} "
            f"{c['wall']:.0f}s",
            flush=True,
        )

    for name, aim in geometry_variants(exact, lost_weight=lost, names=wanted).items():
        head = f"{label} basis={name} lost<={lost:.0e} K={aim.keep} (of {len(aim._sector_cache[aim.sector0][4])})"
        runs = [("Matsubara", Mesh(aim, np.array([0.0]), matsubara, deltas[0], n_real=0, n_mats=6, truth=exact))]
        for delta in deltas:
            omega = np.arange(lo, hi, delta / 2)
            runs.append((f"real delta={delta}", Mesh(aim, omega, matsubara, delta, n_real=8, n_mats=0, truth=exact)))
        for axis, mesh in runs:
            title = f"{head} {axis}"
            result = run_grid(aim, mesh, swm_etas=(1e-4, 1e-5), progress=progress)
            print("\n" + format_report(title, result), flush=True)
            dump[title] = result
            if out:
                with open(out, "w") as f:
                    json.dump(dump, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
