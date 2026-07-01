"""Benchmark harness for ``selfenergy.calc_selfenergy`` (Part A of the
``calc_selfenergy`` profiling plan).

Mirrors the oracle-first Phase-0 convention of ``test_apply_perf.py``: the heavy body
only runs when opted in (``RUN_SELFENERGY_BENCH=1``), so default CI skips it. When
enabled it

* builds the anchor workload (NiO d-shell, ``ls=2``, ``nBaths=100`` from
  ``h0/h0_NiO_100bath.pickle``) the way ``selfenergy.get_selfenergy`` does internally,
  then calls ``calc_selfenergy`` **directly** (``get_selfenergy`` currently mis-calls
  ``get_noninteracting_hamiltonian_operator`` with a stale positional arg order, so it
  cannot be reused as the driver);
* installs **non-invasive per-phase timers** by monkeypatching the phase entry points
  in the ``selfenergy`` namespace (``calc_gs``, ``get_Greens_function``, ``get_sigma``,
  ``get_Sigma_static``) with pass-through wrappers that accumulate wall time — no edits
  to production code;
* runs the whole solve under ``cProfile`` for function-level attribution.

Run serially::

    RUN_SELFENERGY_BENCH=1 pytest -s src/impurityModel/test/test_selfenergy_perf.py

Strong-scaling sweep (fixed problem, more ranks)::

    RUN_SELFENERGY_BENCH=1 mpirun -n 2 pytest -s \
        src/impurityModel/test/test_selfenergy_perf.py

.. warning::

   The default 100-bath anchor workload **does not currently complete**: ``calc_gs``
   fails for any basis large enough to exceed ``dense_cutoff`` (≈20 bath and up), on
   *both* solver paths — the IRLM/TRLM array Lanczos collapses on its seed block
   ("Block collapsed to zero rank"), and the dense fallback hits an ``IndexError`` in
   ``Basis.build_state`` because the post-truncation basis has ``size > 0`` but an empty
   ``local_basis`` (``CIPSISolver.truncate`` clears ``local_basis`` and the
   >truncation_threshold break path leaves it desynced). These are correctness bugs that
   must be fixed before the 100-bath benchmark can run; this harness then re-runs
   unchanged. The **currently-runnable** config is the small 10-bath workload::

       RUN_SELFENERGY_BENCH=1 SELFENERGY_BENCH_NBATH=10 SELFENERGY_BENCH_NVALBATH=10 \
           pytest -s src/impurityModel/test/test_selfenergy_perf.py

Tunables (env vars, defaults in parens): ``SELFENERGY_BENCH_NBATH`` (100),
``SELFENERGY_BENCH_NVALBATH`` (= nBath; the valence/conduction split is a physics
choice not encoded in the pickle, defaulted to all-valence), ``SELFENERGY_BENCH_N0IMP``
(8, NiO d8), ``SELFENERGY_BENCH_NW`` (2000 real-axis mesh points),
``SELFENERGY_BENCH_DENSE_CUTOFF`` (100000 — keeps ``calc_gs`` on the dense eigensolver;
set to 500 to exercise the production Lanczos eigensolver), ``SELFENERGY_BENCH_TRUNC``
(1000), ``SELFENERGY_BENCH_REPS`` (1). cProfile ``.pstats`` dumps land in
``SELFENERGY_BENCH_OUTDIR`` (a scratch dir by default), one file per rank.
"""

import cProfile
import io
import os
import pstats
import time
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import pytest

RUN = os.environ.get("RUN_SELFENERGY_BENCH") == "1"

# `benchmark` keeps this out of standard `pytest` runs (see pytest.ini); the
# RUN_SELFENERGY_BENCH gate is a second guard so even `pytest -m benchmark` skips
# the known-broken 100-bath default workload unless it is explicitly requested.
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not RUN, reason="Set RUN_SELFENERGY_BENCH=1 to run the calc_selfenergy benchmark."),
]

# Repo root: this file is src/impurityModel/test/<this>; the h0 pickles live in <root>/h0.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _env_int(name, default):
    return int(os.environ.get(name, default))


# Accumulators filled by the phase-timer wrappers (keyed by phase name).
_PHASE_TIME: "dict[str, float]" = {}
_PHASE_CALLS: "dict[str, int]" = {}


def _timed(name, func):
    """Wrap ``func`` so each call adds its wall time to the ``name`` accumulator."""

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _PHASE_TIME[name] = _PHASE_TIME.get(name, 0.0) + (time.perf_counter() - t0)
            _PHASE_CALLS[name] = _PHASE_CALLS.get(name, 0) + 1

    return wrapper


@contextmanager
def _phase_timers():
    """Monkeypatch the phase entry points in the ``selfenergy`` namespace with timers.

    Restores the originals on exit. Patches the *names as looked up inside*
    ``calc_selfenergy`` (module-level attributes of ``selfenergy``), so the real call
    sites are timed without touching production code.
    """
    from impurityModel.ed import selfenergy as se

    targets = ["calc_gs", "get_Greens_function", "get_sigma", "get_Sigma_static"]
    originals = {name: getattr(se, name) for name in targets}
    try:
        for name in targets:
            setattr(se, name, _timed(name, originals[name]))
        yield
    finally:
        for name, orig in originals.items():
            setattr(se, name, orig)


def _resolve_reort():
    """Optional GF reort override via SELFENERGY_BENCH_REORT (none/partial/selective/full)."""
    name = os.environ.get("SELFENERGY_BENCH_REORT")
    if not name:
        return None
    from impurityModel.ed.BlockLanczosArray import Reort

    return {
        "none": Reort.NONE,
        "partial": Reort.PARTIAL,
        "selective": Reort.SELECTIVE,
        "full": Reort.FULL,
        "periodic": Reort.PERIODIC,
    }[name.lower()]


def _build_inputs(ls, nBaths, nValBaths, n0imp, n_omega, dense_cutoff, truncation_threshold, *, rank, verbose):
    """Construct the calc_selfenergy arguments for the NiO d-shell anchor workload.

    Mirrors the input construction inside ``selfenergy.get_selfenergy`` but calls
    ``get_noninteracting_hamiltonian_operator`` with the correct keyword arguments.
    """
    from impurityModel.ed import finite
    from impurityModel.ed.block_structure import BlockStructure
    from impurityModel.ed.get_spectra import get_noninteracting_hamiltonian_operator

    Fdd = [7.5, 0, 9.9, 0, 6.6]
    xi = 0.0
    hField = (0.0, 0.0, 0.0001)

    sum_baths = OrderedDict({ls: nBaths})
    nValBaths_d = OrderedDict({ls: nValBaths})
    n_imp = 2 * (2 * ls + 1)

    # Coulomb U as a rank-4 tensor in the impurity spin-orbital index space.
    u4 = np.zeros((n_imp, n_imp, n_imp, n_imp), dtype=complex)
    uOp = finite.getUop(l1=ls, l2=ls, l3=ls, l4=ls, R=Fdd)
    nBaths_for_c2i = OrderedDict({ls: 0})
    for process, val in uOp.items():
        i = finite.c2i(nBaths_for_c2i, process[0][0])
        j = finite.c2i(nBaths_for_c2i, process[1][0])
        k = finite.c2i(nBaths_for_c2i, process[2][0])
        m = finite.c2i(nBaths_for_c2i, process[3][0])
        u4[i, j, k, m] = 2.0 * val

    impurity_orbitals = {ls: [list(range(n_imp))]}
    offset = n_imp
    valence_baths = {ls: [[offset + i for i in range(nValBaths)]]}
    offset += nValBaths
    conduction_baths = {ls: [[offset + i for i in range(nBaths - nValBaths)]]}
    bath_states = (valence_baths, conduction_baths)
    mixed_valence = {ls: 0}

    # calc_gs/find_ground_state_basis expect the impurity nominal-occupation dict
    # ({l: n0}), not get_selfenergy's stale 3-tuple form.
    nominal_occ = {ls: n0imp}

    block_structure = BlockStructure(
        blocks=[list(range(n_imp))],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    rot_to_spherical = np.eye(n_imp, dtype=complex)

    h0_filename = os.path.join(REPO_ROOT, "h0", f"h0_NiO_{nBaths}bath.pickle")
    hOp = get_noninteracting_hamiltonian_operator(
        nBaths=sum_baths,
        nValBaths=nValBaths_d,
        SOCs=[0, xi],
        hField=hField,
        h0_filename=h0_filename,
        rank=rank,
        verbose=verbose,
    )
    # Map (l,s,m) / (l,b) labels to single integer indices. Drop identically-zero terms
    # first: get_noninteracting_hamiltonian_operator unconditionally adds a 2p (l=1) SOC
    # operator whose terms are all 0.0 when xi_2p=0; those carry unmappable l=1 labels and
    # contribute nothing to H.
    hOp_int = {}
    for process, value in hOp.items():
        if abs(value) == 0:
            continue
        hOp_int[tuple((finite.c2i(sum_baths, spinOrb), action) for spinOrb, action in process)] = value

    omega_mesh = np.linspace(-1.83, 1.83, n_omega)

    return dict(
        h0=hOp_int,
        u4=u4,
        iw=None,
        w=omega_mesh,
        delta=0.2,
        nominal_occ=nominal_occ,
        mixed_valence=mixed_valence,
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        tau=0.002,
        verbosity=2 if verbose else 0,
        block_structure=block_structure,
        rot_to_spherical=rot_to_spherical,
        cluster_label="bench",
        reort=_resolve_reort(),
        dense_cutoff=dense_cutoff,
        spin_flip_dj=False,
        chain_restrict=False,
        occ_cutoff=float(os.environ.get("SELFENERGY_BENCH_OCC_CUTOFF", 1e-12)),
        truncation_threshold=truncation_threshold,
        slaterWeightMin=float(os.environ.get("SELFENERGY_BENCH_SWMIN", 1e-12)),
        dN=(int(os.environ["SELFENERGY_BENCH_DN"]) if os.environ.get("SELFENERGY_BENCH_DN") else None),
        sparse_green=True,
    )


def test_calc_selfenergy_benchmark():
    """Profile a full ``calc_selfenergy`` solve and report per-phase + cProfile attribution."""
    from mpi4py import MPI

    from impurityModel.ed import selfenergy

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    ls = 2
    nBaths = _env_int("SELFENERGY_BENCH_NBATH", 100)
    nValBaths = _env_int("SELFENERGY_BENCH_NVALBATH", nBaths)
    n0imp = _env_int("SELFENERGY_BENCH_N0IMP", 8)
    n_omega = _env_int("SELFENERGY_BENCH_NW", 2000)
    # Default dense_cutoff is large enough to keep calc_gs on the dense eigensolver: the
    # IRLM/TRLM array path collapses on its seed block ("Block collapsed to zero rank")
    # for the 100-bath workload (a known Lanczos-seed fragility, separate from the GF
    # block-Lanczos). Set SELFENERGY_BENCH_DENSE_CUTOFF=500 to exercise the (currently
    # failing) Lanczos eigensolver in calc_gs.
    dense_cutoff = _env_int("SELFENERGY_BENCH_DENSE_CUTOFF", 100000)
    truncation_threshold = _env_int("SELFENERGY_BENCH_TRUNC", 1000)
    reps = _env_int("SELFENERGY_BENCH_REPS", 1)
    outdir = os.environ.get(
        "SELFENERGY_BENCH_OUTDIR",
        os.path.join(REPO_ROOT, "debug", "selfenergy_bench"),
    )
    os.makedirs(outdir, exist_ok=True)

    verbose = rank == 0

    kwargs = _build_inputs(
        ls, nBaths, nValBaths, n0imp, n_omega, dense_cutoff, truncation_threshold, rank=rank, verbose=verbose
    )
    kwargs["comm"] = comm

    if rank == 0:
        print(
            f"\n[selfenergy-bench] ls={ls} nBaths={nBaths} nValBaths={nValBaths} "
            f"n0imp={n0imp} n_omega={n_omega} dense_cutoff={dense_cutoff} "
            f"trunc={truncation_threshold} ranks={size} reps={reps}",
            flush=True,
        )

    wall = []
    result = None
    for rep in range(reps):
        _PHASE_TIME.clear()
        _PHASE_CALLS.clear()
        profiler = cProfile.Profile()
        comm.Barrier()
        t0 = time.perf_counter()
        # A pure *performance* benchmark: if the physics post-checks fail (e.g. an unphysical
        # self-energy from an under-resolved excited sector at large bath with a tight dN), the
        # phase timings (calc_gs / get_Greens_function) were still collected, so report them.
        # Set SELFENERGY_BENCH_ALLOW_UNPHYSICAL=1 to swallow that and time the run anyway.
        result = None
        with _phase_timers():
            profiler.enable()
            try:
                result = selfenergy.calc_selfenergy(**kwargs)
            except selfenergy.UnphysicalGreensFunctionError:
                if os.environ.get("SELFENERGY_BENCH_ALLOW_UNPHYSICAL") != "1":
                    profiler.disable()
                    raise
                if rank == 0:
                    print("[selfenergy-bench] NOTE: unphysical self-energy (timing-only run).", flush=True)
            profiler.disable()
        elapsed = time.perf_counter() - t0
        comm.Barrier()
        wall.append(elapsed)

        # Per-rank cProfile dump (binary, for offline snakeviz/pstats inspection).
        stats_path = os.path.join(outdir, f"calc_selfenergy_n{size}_rank{rank}_rep{rep}.pstats")
        profiler.dump_stats(stats_path)

        # Optional: dump the real-axis self-energy on rank 0 for serial-vs-MPI comparison.
        if (
            rank == 0
            and os.environ.get("SELFENERGY_BENCH_DUMP")
            and result is not None
            and result.get("sigma_real") is not None
        ):
            arr = np.array(result["sigma_real"])
            np.save(os.path.join(outdir, f"sigma_real_n{size}.npy"), arr)
            print(f"[selfenergy-bench] sigma_real dumped: shape={arr.shape} |.|={np.linalg.norm(arr):.6e}", flush=True)

        if rank == 0:
            total = elapsed
            other = total - sum(_PHASE_TIME.values())
            print(f"\n[selfenergy-bench] === rep {rep}: wall = {total:.2f} s ===", flush=True)
            print(f"  {'phase':<22}{'time(s)':>10}{'%':>8}{'calls':>8}")
            for name in ["calc_gs", "get_Greens_function", "get_sigma", "get_Sigma_static"]:
                t = _PHASE_TIME.get(name, 0.0)
                print(f"  {name:<22}{t:>10.2f}{100 * t / total:>8.1f}{_PHASE_CALLS.get(name, 0):>8d}")
            print(f"  {'other (build/IO)':<22}{other:>10.2f}{100 * other / total:>8.1f}{'':>8}")

            # Optional per-step block-Lanczos split (BLOCKLANCZOS_PROFILE=1; env-gated in the
            # Cython kernel). Splits the GF block-Lanczos step into matvec / recurrence-LA /
            # W-estimator / triggered reort / CholeskyQR2 / convergence monitor.
            try:
                from impurityModel.ed.BlockLanczos import get_block_lanczos_profile

                prof = get_block_lanczos_profile()
            except Exception:
                prof = {}
            if prof:
                ops = ["matvec", "recurrence", "choleskyqr2_cond", "w_estimate", "reort", "monitor"]
                tot = sum(prof.get(o, 0.0) for o in ops)
                print("\n[selfenergy-bench] --- block-Lanczos per-op split (GF kernel) ---")
                print(f"  {'op':<18}{'time(s)':>10}{'%':>8}{'calls':>10}")
                for o in ops:
                    t = prof.get(o, 0.0)
                    print(f"  {o:<18}{t:>10.2f}{100 * t / tot if tot else 0:>8.1f}{int(prof.get(o + '#n', 0)):>10d}")
                acted = int(prof.get("reort_acted#n", 0))
                rtot = int(prof.get("reort_total#n", 0))
                print(f"  reort triggered: {acted}/{rtot} steps ({100 * acted / rtot if rtot else 0:.0f}%)", flush=True)
                try:
                    from impurityModel.ed.BlockLanczosArray import get_reort_profile

                    rp = get_reort_profile()
                except Exception:
                    rp = {}
                if rp.get("acted", 0):
                    a = rp["acted"]
                    print(
                        f"  reort fan-out: avg {rp.get('bad_blocks', 0) / a:.1f} bad blocks / "
                        f"{rp.get('bad_cols', 0) / a:.1f} bad cols per acting call; "
                        f"avg {rp.get('n_blocks_total', 0) / max(rp.get('calls', 1), 1):.1f} blocks present",
                        flush=True,
                    )

            # Top cProfile entries by cumulative and by tottime (rank-0 view).
            for sort_key in ("cumulative", "tottime"):
                buf = io.StringIO()
                pstats.Stats(profiler, stream=buf).sort_stats(sort_key).print_stats(30)
                print(f"\n[selfenergy-bench] --- cProfile top 30 by {sort_key} (rank 0) ---")
                print(buf.getvalue(), flush=True)

    if rank == 0:
        print(
            f"[selfenergy-bench] wall over {reps} rep(s): "
            f"min={min(wall):.2f}s median={sorted(wall)[len(wall) // 2]:.2f}s max={max(wall):.2f}s",
            flush=True,
        )
        # Light sanity: a completed solve produced a self-energy and a thermal density matrix.
        # (Skipped for a deliberate timing-only run that swallowed an unphysical self-energy.)
        if result is not None:
            assert result["sigma_real"] is not None
            assert result["thermal_rho"] is not None
