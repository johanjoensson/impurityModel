"""Truncation-threshold reliability sweep (opt-in benchmark).

Measures how capping the many-body basis (``truncation_threshold``) degrades the NiO
d-shell self-energy, across reorthogonalization modes, and what it buys in memory —
the empirical basis for the HPC sizing heuristics in
``doc/plans/truncation_reliability.md``.

One configuration per process invocation, so the ``VmHWM`` peak-RSS readings are
honest (a high-water mark never decays inside a process). The reference
(``TRUNCATION_BENCH_TRUNC=inf``) must run first; every later run compares against
its saved ``.npz``. Typical ladder::

    OUT=truncation-bench
    for T in inf 4000 2000 1000 500 250; do
      for R in none partial full; do
        RUN_TRUNCATION_BENCH=1 TRUNCATION_BENCH_TRUNC=$T TRUNCATION_BENCH_REORT=$R \\
        TRUNCATION_BENCH_OUTDIR=$OUT pytest -s -m benchmark \\
            src/impurityModel/test/test_truncation_reliability.py
      done
    done

and the same under ``mpiexec -n 2 ... --with-mpi`` / ``-n 3`` (results land in
per-rank-count JSON rows). Render the accumulated table with::

    python -m impurityModel.test.test_truncation_reliability $OUT

Tunables (env): ``TRUNCATION_BENCH_TRUNC`` (required: ``inf`` or an int),
``TRUNCATION_BENCH_REORT`` (``none``/``partial``/``full``; default ``none``),
``TRUNCATION_BENCH_NBATH`` (10; use 20 for caps that actually bind),
``TRUNCATION_BENCH_NW`` (500), ``TRUNCATION_BENCH_DENSE_CUTOFF`` (500 — the
production Lanczos ground-state path), ``TRUNCATION_BENCH_OUTDIR`` (default
``truncation-bench/`` — a calculation dropping, do not commit).
"""

import json
import os
import time

import numpy as np
import pytest

RUN = os.environ.get("RUN_TRUNCATION_BENCH") == "1"

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not RUN, reason="Set RUN_TRUNCATION_BENCH=1 to run the truncation reliability sweep."),
]


def _env(name, default):
    return os.environ.get(name, default)


def _peak_rss_bytes():
    """Process high-water-mark RSS (VmHWM) in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return 0


def _reference_path(outdir, nbaths, ranks):
    return os.path.join(outdir, f"reference_nb{nbaths}_np{ranks}.npz")


def _results_path(outdir):
    return os.path.join(outdir, "results.jsonl")


def test_truncation_reliability_config():
    """Run one (threshold, reort) configuration and record metrics vs the reference."""
    # Deferred (like test_selfenergy_perf): default CI runs skip this module and must
    # not pay the heavy solver imports.
    from mpi4py import MPI  # noqa: PLC0415

    from impurityModel.ed.selfenergy import calc_selfenergy  # noqa: PLC0415
    from impurityModel.test._nio_workload import build_selfenergy_inputs  # noqa: PLC0415

    comm = MPI.COMM_WORLD
    ranks = comm.size
    rank = comm.rank

    trunc_env = _env("TRUNCATION_BENCH_TRUNC", None)
    assert trunc_env is not None, "Set TRUNCATION_BENCH_TRUNC=inf (reference) or an integer cap."
    threshold = np.inf if trunc_env.lower() in ("inf", "ref") else int(float(trunc_env))
    reort = _env("TRUNCATION_BENCH_REORT", "none").lower()
    nbaths = int(_env("TRUNCATION_BENCH_NBATH", "10"))
    n_omega = int(_env("TRUNCATION_BENCH_NW", "500"))
    # 500 exercises the production Lanczos eigensolver for the ground state; the
    # _nio_workload default (100000) would keep calc_gs on the dense fallback.
    dense_cutoff = int(_env("TRUNCATION_BENCH_DENSE_CUTOFF", "500"))
    outdir = _env("TRUNCATION_BENCH_OUTDIR", "truncation-bench")
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
    comm.Barrier()

    kwargs = build_selfenergy_inputs(
        nBaths=nbaths,
        n_omega=n_omega,
        dense_cutoff=dense_cutoff,
        truncation_threshold=threshold,
        reort=None if reort == "none" else reort,
        rank=rank,
    )
    t0 = time.perf_counter()
    result = calc_selfenergy(comm=comm, **kwargs)
    wall = time.perf_counter() - t0
    peak_rss = comm.allreduce(_peak_rss_bytes(), op=MPI.MAX)

    if rank != 0:
        return

    sigma_real = result["sigma_real"]
    gs_real = result["gs_realaxis"]
    e0 = float(np.min(result["gs_energies"]))
    causality_violation = float(np.max(np.diagonal(gs_real.imag, axis1=1, axis2=2)))

    is_reference = not np.isfinite(threshold)
    ref_file = _reference_path(outdir, nbaths, ranks)
    row = {
        "threshold": None if is_reference else int(threshold),
        "reort": reort,
        "nbaths": nbaths,
        "ranks": ranks,
        "e0": e0,
        "causality_violation": causality_violation,
        "wall_s": round(wall, 3),
        "peak_rss_bytes": int(peak_rss),
    }
    if is_reference:
        np.savez_compressed(ref_file, sigma_real=sigma_real, gs_realaxis=gs_real, e0=e0)
        row.update(sigma_max_dev=0.0, sigma_l2_dev=0.0, e0_err=0.0)
    else:
        assert os.path.exists(ref_file), f"Reference missing: run TRUNCATION_BENCH_TRUNC=inf at -n {ranks} first."
        ref = np.load(ref_file)
        dev = np.abs(sigma_real - ref["sigma_real"])
        scale = max(float(np.max(np.abs(ref["sigma_real"]))), np.finfo(float).tiny)
        row.update(
            sigma_max_dev=float(np.max(dev)) / scale,
            sigma_l2_dev=float(np.linalg.norm(dev)) / max(float(np.linalg.norm(ref["sigma_real"])), 1e-300),
            e0_err=abs(e0 - float(ref["e0"])),
        )
    with open(_results_path(outdir), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        f"[truncation-bench] T={trunc_env} reort={reort} ranks={ranks}: "
        f"e0={e0:.6f} (err {row['e0_err']:.2e}), sigma max dev {row['sigma_max_dev']:.2e}, "
        f"causality {causality_violation:.1e}, peak RSS {peak_rss / 2**20:.0f} MiB, {wall:.1f} s",
        flush=True,
    )


def render_table(outdir):
    """Aligned text table of all recorded sweep rows (rank-count then threshold order)."""
    rows = []
    with open(_results_path(outdir)) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: (r["ranks"], r["reort"], -(r["threshold"] or 10**18)))
    header = (
        f"{'ranks':>5} {'reort':>8} {'threshold':>10} {'e0_err':>10} {'sig_max':>10} "
        f"{'sig_l2':>10} {'causal':>9} {'RSS_MiB':>8} {'wall_s':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        thr = "inf" if r["threshold"] is None else str(r["threshold"])
        lines.append(
            f"{r['ranks']:>5} {r['reort']:>8} {thr:>10} {r.get('e0_err', float('nan')):>10.2e} "
            f"{r.get('sigma_max_dev', float('nan')):>10.2e} {r.get('sigma_l2_dev', float('nan')):>10.2e} "
            f"{r['causality_violation']:>9.1e} {r['peak_rss_bytes'] / 2**20:>8.0f} {r['wall_s']:>8.1f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    print(render_table(sys.argv[1] if len(sys.argv) > 1 else "truncation-bench"))
