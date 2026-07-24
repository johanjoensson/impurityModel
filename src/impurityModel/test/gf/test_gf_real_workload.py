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
unset keeps the full mesh), ``BENCH_OUT`` (``.npz`` dump path), ``VERBOSITY``.
"""

import os
import time

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.memory_estimate import format_bytes, peak_rss_bytes

RUN = os.environ.get("RUN_REAL_WORKLOAD_BENCH", "0") not in ("0", "", "false", "False")

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not RUN, reason="Set RUN_REAL_WORKLOAD_BENCH=1 (and WORKLOAD_H5) to run."),
]


def _env_int(name):
    value = os.environ.get(name)
    return None if value in (None, "") else int(value)


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
    t0 = time.perf_counter()
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

    peaks = comm.gather(peak_rss_bytes(), root=0)
    if comm.rank == 0:
        print(
            f"\n[real-workload] {workload['label']} ({os.path.basename(os.path.dirname(h5_path))}) "
            f"gf_method={gf_method} reort={reort} cap={cap} "
            f"N_IW={os.environ.get('N_IW', 'full')} N_W={os.environ.get('N_W', 'full')}"
        )
        print(f"[real-workload] wall {wall:.1f} s, peak RSS per rank: {[format_bytes(p) for p in peaks]}")
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
