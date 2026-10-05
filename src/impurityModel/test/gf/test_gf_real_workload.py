"""Opt-in benchmark: ``calc_selfenergy`` on a real ``impurityModel_data.h5`` workload.

The measurement rig for the per-frequency BiCGSTAB memory question
(``doc/plans/bicgstab_per_frequency_gf.md``, Phase 3b): one run = one
(workload, gf_method, reort, cap, mesh subset) point, reporting wall time, peak RSS
(``VmHWM``) and, when asked, dumping ``sigma`` / ``sigma_real`` / ``sigma_static`` to an
``.npz`` so separate processes can be compared at fixed accuracy. Runs are separate
processes on purpose -- VmHWM is a process-lifetime high-water mark, so a second method
run in the same process would be hidden under the first one's peak.

Usage::

    RUN_REAL_WORKLOAD_BENCH=1 WORKLOAD_H5=path/to/impurityModel_data.h5 \
    GF_METHOD=bicgstab N_IW=64 N_W=0 BENCH_OUT=/tmp/ni_bicgstab.npz \
    mpiexec -n 2 python -m pytest src/impurityModel/test/test_gf_real_workload.py \
        -m benchmark --with-mpi -s

Environment knobs: ``WORKLOAD_H5`` (required), ``GF_METHOD`` (default lanczos),
``REORT`` / ``CAP`` (default: the archive's production settings; ``CAP`` accepts a
number or ``none``), ``N_IW`` / ``N_W`` (mesh subsampling; ``0`` drops the axis,
unset keeps the full mesh), ``BENCH_OUT`` (``.npz`` dump path), ``VERBOSITY``, and
``PHASES=1``, which prints where the wall clock went (see :func:`_phase_timers`).
"""

import os
import time
from contextlib import contextmanager

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.memory_estimate import format_bytes, peak_rss_bytes

RUN = os.environ.get("RUN_REAL_WORKLOAD_BENCH", "0") not in ("0", "", "false", "False")

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not RUN, reason="Set RUN_REAL_WORKLOAD_BENCH=1 (and WORKLOAD_H5) to run."),
]


#: The phases :func:`_phase_timers` records, in the order they are printed.
_PHASE_NAMES = ["calc_gs", "get_Greens_function", "gf_units", "moments", "dyson"]


def _env_int(name):
    value = os.environ.get(name)
    return None if value in (None, "") else int(value)


@contextmanager
def _phase_timers(phases):
    """Time the self-energy phases at their call sites, and each rank's busy time in GF units.

    ``phases[name]`` accumulates rank-local seconds for the ground state (``calc_gs``), the
    Green's function (``get_Greens_function``), the exact moments and the Dyson step, plus
    ``gf_units``: the time this rank spent inside unit kernels. ``gf_units`` over
    ``get_Greens_function`` is the rank's busy fraction of the GF phase; the rest is split
    overhead, assembly and waiting for the slowest color -- the idle fraction the Phase 8
    packing work (review ledger P2) is gated on. Patches module attributes only and restores
    them on exit, so production code is untouched.
    """
    from impurityModel.ed import gf_engine, gf_units, selfenergy, sigma_estimators, solver_trace
    from impurityModel.ed.memory_estimate import current_rss_bytes, release_freed_heap

    unit_seconds = phases.setdefault("unit_seconds", [])
    peaks_after = phases.setdefault("peak_after", {})
    trace = None

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                seconds = time.perf_counter() - t0
                phases[name] = phases.get(name, 0.0) + seconds
                # The high-water mark as each phase ends attributes a rank's peak to a phase.
                peaks_after.setdefault(name, peak_rss_bytes())
                if name == "gf_units":
                    # _block_green_group records one gf_unit_memory note per unit (collective, so
                    # it is there on every rank of the color); pair it with the unit's time.
                    notes = trace.of_kind("gf_unit_memory") if trace is not None else []
                    last = notes[-1] if notes else {}
                    unit_seconds.append((seconds, last.get("n_blocks"), last.get("retained_size")))

        return wrapper

    splits = phases.setdefault("splits", [])
    real_split = gf_units.split_basis_and_redistribute_psi

    def recorded_split(basis, *args, **kwargs):
        # What the split itself costs this rank: every color receives the whole parent basis
        # (review ledger P1), so the RSS step across the call is that replica plus the seeds.
        # Trim first: glibc would otherwise place the replica in heap the CIPSI rounds already freed,
        # and the step would read about zero while the replica really occupies memory.
        release_freed_heap()
        before = current_rss_bytes()
        out = real_split(basis, *args, **kwargs)
        split_basis = out[4]
        splits.append(
            {
                "parent_size": int(basis.size),
                "color_size": int(split_basis.size),
                "color_ranks": 1 if split_basis.comm is None else int(split_basis.comm.size),
                "n_colors": len(out[3]),
                "rss_step": current_rss_bytes() - before,
            }
        )
        return out

    targets = [
        (selfenergy, "calc_gs", "calc_gs"),
        (selfenergy, "get_Greens_function", "get_Greens_function"),
        (selfenergy, "get_greens_function_moments", "moments"),
        (sigma_estimators, "get_sigma", "dyson"),
        (gf_engine, "_block_green_group", "gf_units"),
    ]
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in targets]
    originals.append((gf_units, "split_basis_and_redistribute_psi", real_split))
    try:
        for (module, attr, label), (_, _, fn) in zip(targets, originals):
            setattr(module, attr, timed(label, fn))
        gf_units.split_basis_and_redistribute_psi = recorded_split
        with solver_trace.tracing() as trace:
            yield
    finally:
        for module, attr, fn in originals:
            setattr(module, attr, fn)


@pytest.mark.mpi
def test_real_workload_selfenergy():
    from impurityModel.test.support.real_workload import load_workload, run_selfenergy

    comm = MPI.COMM_WORLD
    h5_path = os.environ.get("WORKLOAD_H5")
    assert h5_path, "WORKLOAD_H5 must point to an impurityModel_data.h5 archive"

    gf_method = os.environ.get("GF_METHOD", "lanczos")
    reort = os.environ.get("REORT", "archive")
    cap = os.environ.get("CAP", "archive")
    if cap != "archive":
        cap = None if cap.lower() == "none" else float(cap)
    verbosity = int(os.environ.get("VERBOSITY", "1"))

    workload = load_workload(h5_path)
    phases = {}
    timers_on = os.environ.get("PHASES", "0") not in ("0", "")
    timers = _phase_timers(phases) if timers_on else _no_timers()
    t0 = time.perf_counter()
    with timers:
        result = run_selfenergy(
            workload,
            comm=comm,
            gf_method=gf_method,
            reort=reort,
            truncation_threshold=cap,
            n_iw=_env_int("N_IW"),
            n_w=_env_int("N_W"),
            verbosity=verbosity,
        )
    wall = time.perf_counter() - t0
    all_phases = comm.gather(phases, root=0)

    peaks = comm.gather(peak_rss_bytes(), root=0)
    if comm.rank == 0:
        print(
            f"\n[real-workload] {workload['label']} ({os.path.basename(os.path.dirname(h5_path))}) "
            f"gf_method={gf_method} reort={reort} cap={cap} "
            f"N_IW={os.environ.get('N_IW', 'full')} N_W={os.environ.get('N_W', 'full')}"
        )
        print(f"[real-workload] wall {wall:.1f} s, peak RSS per rank: {[format_bytes(p) for p in peaks]}")
        if phases:
            names = _PHASE_NAMES
            # Non-root ranks leave get_Greens_function as soon as their units are gathered, so
            # their own GF time understates the phase; the phase wall is the slowest rank's.
            gf_wall = max(ph.get("get_Greens_function", 0.0) for ph in all_phases)
            print(f"[real-workload] phase seconds per rank (GF phase wall {gf_wall:.1f} s; busy = in GF units / wall):")
            for rank, ph in enumerate(all_phases):
                busy = ph.get("gf_units", 0.0) / gf_wall if gf_wall > 0 else float("nan")
                cells = "  ".join(f"{n}={ph.get(n, 0.0):7.1f}" for n in names)
                print(f"  rank {rank}: {cells}  gf_busy={busy:.1%}")
                for split in ph.get("splits", []):
                    print(
                        f"    split: {split['n_colors']} colors, this color {split['color_ranks']} ranks, "
                        f"basis {split['parent_size']:,} -> {split['color_size']:,} dets per color, "
                        f"RSS step {format_bytes(split['rss_step'])}"
                    )
                peak_after = ph.get("peak_after", {})
                print("    peak RSS after first " + ", ".join(f"{k}: {format_bytes(v)}" for k, v in peak_after.items()))
                for seconds, n_blocks, size in sorted(ph.get("unit_seconds", []), key=lambda u: -u[0])[:8]:
                    print(f"    unit {seconds:7.1f} s  n_blocks={n_blocks}  retained_size={size}")
        out = os.environ.get("BENCH_OUT")
        if out:
            np.savez(
                out,
                sigma=result["sigma"] if result["sigma"] is not None else np.zeros(0),
                sigma_real=result["sigma_real"] if result["sigma_real"] is not None else np.zeros(0),
                sigma_static=result["sigma_static"],
                wall=wall,
                peaks=np.array(peaks, dtype=float),
            )
            print(f"[real-workload] results written to {out}")
        if timers_on:
            # A patch target that production no longer reaches through its module attribute (a
            # rename, or a `from ... import` at the call site) records nothing and the run looks
            # clean. Fail instead: the cluster kit's first round ran an install whose harness
            # had no PHASES support and spent ~6 h of 32-128 ranks producing none of these lines.
            recorded = {name for ph in all_phases for name, seconds in ph.items() if isinstance(seconds, float)}
            missing = [n for n in _PHASE_NAMES if n not in recorded]
            assert not missing, f"PHASES=1 recorded no phase timing for {missing}; the timer patches missed"


@contextmanager
def _no_timers():
    yield
