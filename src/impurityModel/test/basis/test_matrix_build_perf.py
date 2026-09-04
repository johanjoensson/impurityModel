"""Where ground-state time actually goes: sparse-matrix build vs. eigensolve attribution.

``CIPSISolver.get_eigenvectors`` rebuilds the *whole* Hamiltonian every CIPSI cycle
(``cipsi_solver.py:1040``/``:1160`` -> ``basis_transcription.build_sparse_matrix``), even
though cycle ``i+1``'s matrix differs from cycle ``i``'s only in the rows and columns of the
newly admitted determinants. That looks like an obvious thing to make incremental. This
harness was written to check whether it is worth it, and the answer -- on the FCC Ni 15-bath
production workload, serial, 214 builds over 11 expands -- was no:

===========================================  ==========================
build (all 214, whole basis each time)        48.2 s   (4.5%)
eigensolve                                    1021.3 s (95.5%)
per-expand ceiling  sum(|B_i|)/|B_final|      5.7x - 18.7x
final basis / nnz / matrix                    16,532 dets, 129k nnz, 3.1 MiB
===========================================  ==========================

A perfect incremental build -- one operator apply per *newly admitted* determinant instead of
one per determinant in the basis -- therefore buys at most ~4.4% of the ground-state solve,
for a cache that has to survive an index permutation on every ``add_states`` and two new
collectives. The verdict strengthens with basis size rather than weakening: build cost per
determinant is roughly constant, while solve cost is ``n_iter x nnz x p`` and nnz *per
determinant* grows as more of the connected space falls inside the basis.

The same run refutes the opposite idea -- dropping the matrix and running the recurrence
matrix-free, applying ``ManyBodyOperator`` and discarding out-of-basis determinants. One
projected matrix-free step (apply + ``redistribute_block`` + ``keep_rows``) measured 162 ms
against 0.64 ms for the equivalent sparse matvec: **252x**, so the matrix pays for its own
construction after 1.4 matvecs. There is no memory ceiling to rescue it either, the matrix
being 3.1 MiB.

Keep this harness for the question, not the answer: re-run it on a bigger or distributed
workload before concluding anything about a different regime. Note especially that the
``_index_sequence`` leg is *zero* in serial runs -- the routed all-to-all that resolves bras
to global row indices only exists in the distributed branch.

Nothing here is production code and nothing is monkeypatched permanently: the wrappers are
installed for the duration of one run and removed in a ``finally``. It is opt-in
(``RUN_MATRIX_BUILD_BENCH=1``) and ``benchmark``-marked, so the default suite skips it.

What it records
---------------
* the *call histogram*, one row per ``build_sparse_matrix`` call with the basis size at the
  time, grouped by which ``expand`` it belongs to (``solve_ground_state`` runs one expand per
  trial occupation in ``walk_to_ground_state_sector`` before the final refinement expand, and
  ``sum|B_i|/|B_final|`` is only meaningful *within* one expand);
* the split inside each build: (a) ``build_local_operator_list`` -- the C++ apply plus the
  per-determinant Python ``ManyBodyState``/``.items()`` walk, (b) ``Basis._index_sequence`` --
  the routed all-to-all resolving bras to global row indices (pickled determinant *bytes*, so
  its volume is nnz x n_bytes; zero in serial), (c) the remainder, i.e. the Python-list -> COO
  -> CSC assembly;
* the split outside it: state<->array marshalling (``build_state`` /
  ``build_distributed_vector``, per-determinant Python loops that run inside every solve)
  against the Krylov work proper;
* ``nnz`` and the CSC's bytes per cycle, the per-solve CSC->CSR conversion
  (``BlockLanczosArray.pyx:482``, an O(nnz) copy on every solve), and one projected
  matrix-free step against one sparse matvec of the same width.

Run it::

    RUN_MATRIX_BUILD_BENCH=1 pytest -s -m benchmark \
        src/impurityModel/test/basis/test_matrix_build_perf.py

    RUN_MATRIX_BUILD_BENCH=1 mpiexec -n 2 python -u -m pytest -s -m benchmark --with-mpi \
        src/impurityModel/test/basis/test_matrix_build_perf.py

Tunables (env, defaults in parens): ``MATRIX_BENCH_ARCHIVE`` (the FCC Ni 15-bath archive --
metallic, the regime that motivated the work), ``MATRIX_BENCH_TRUNC`` (``"archive"``; a
number overrides the recorded basis cap), ``MATRIX_BENCH_WIDTH`` (4, the block width used for
the step timings), ``MATRIX_BENCH_JSON`` (a scratch path for the machine-readable dump).
"""

import json
import os
import time

import numpy as np
import pytest
from mpi4py import MPI

pytestmark = pytest.mark.benchmark

_IMPMOD_ROOT = os.environ.get("IMPMOD_TESTS_DIR", "/home/johan/Programming/impmod_tests")
_DEFAULT_ARCHIVE = os.path.join(
    _IMPMOD_ROOT, "FCC_Ni/impmod/15_BathStates_HaverGeometry_partialReorthonormalization/impurityModel_data.h5"
)
_ARCHIVE = os.environ.get("MATRIX_BENCH_ARCHIVE", _DEFAULT_ARCHIVE)
_ENABLED = os.environ.get("RUN_MATRIX_BUILD_BENCH") == "1"
_WIDTH = int(os.environ.get("MATRIX_BENCH_WIDTH", "4"))


class _StopAfterGroundState(Exception):
    """Raised in place of the Green's-function phase: this harness only measures the GS."""


class _Recorder:
    """Accumulates one row per ``build_sparse_matrix`` call, plus the enclosing solve times."""

    def __init__(self):
        self.builds = []
        self.solves = []
        self._legs = None
        # Re-entrancy guard: `_index_sequence` calls itself on its retry path, and timing
        # both levels would double-count leg (b).
        self._index_depth = 0
        # Which `expand` a build belongs to. `solve_ground_state` runs one expand per trial
        # occupation in `walk_to_ground_state_sector` before the final refinement expand, and
        # sum|B_i|/|B_final| is only meaningful *within* one expand -- pooling the walk's small
        # bases with the refinement's large ones would report a ceiling no cache could reach.
        self.expand_index = -1
        self.marshal_seconds = 0.0

    @property
    def in_build(self):
        return self._legs is not None

    # -- leg accounting ---------------------------------------------------- #
    def open_build(self):
        self._legs = {"apply": 0.0, "index": 0.0}
        self._lookups = 0

    def add_lookups(self, n):
        if self._legs is not None:
            self._lookups += int(n)

    def add_leg(self, name, seconds):
        # Calls from outside a build (e.g. `determine_new_Dj`'s own lookups) are ignored:
        # this is attribution *within* the build, not a global profile.
        if self._legs is not None:
            self._legs[name] += seconds

    def close_build(self, seconds, basis_size, matrix):
        legs = self._legs or {"apply": 0.0, "index": 0.0}
        lookups = self._lookups
        self._legs = None
        nnz = int(matrix.nnz)
        nbytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
        self.builds.append(
            {
                "expand": self.expand_index,
                "basis_size": int(basis_size),
                "seconds": seconds,
                "apply_seconds": legs["apply"],
                "index_seconds": legs["index"],
                # The remainder: Python list -> COO -> CSC assembly.
                "assembly_seconds": max(seconds - legs["apply"] - legs["index"], 0.0),
                # Contention-immune: the number of determinants pushed through the routed
                # lookup does not depend on how the ranks were scheduled. Divided into
                # `index_seconds` it says whether that leg is serialization work or waiting --
                # `graph_alltoall` pickles `SlaterDeterminant` objects, so ~10 us per lookup is
                # the signature of real per-object cost, while far more is rank descheduling
                # showing up inside the collective.
                "index_lookups": int(lookups),
                "nnz": nnz,
                "matrix_bytes": nbytes,
            }
        )


def _install(recorder):
    """Wrap the three timed entry points. Returns a callable that restores them."""
    from impurityModel.ed import basis_transcription, cipsi_solver
    from impurityModel.ed.manybody_basis import Basis

    original = {
        "build": cipsi_solver.build_sparse_matrix,
        "apply": basis_transcription.build_local_operator_list,
        "index": Basis._index_sequence,
        "eigen": cipsi_solver.CIPSISolver.get_eigenvectors,
        "state": cipsi_solver.build_state,
        "vector": cipsi_solver.build_distributed_vector,
    }

    def timed_build(basis, op):
        recorder.open_build()
        t0 = time.perf_counter()
        result = original["build"](basis, op)
        recorder.close_build(time.perf_counter() - t0, len(basis), result)
        return result

    def timed_apply(basis, op, slaterWeightMin):
        t0 = time.perf_counter()
        result = original["apply"](basis, op, slaterWeightMin)
        recorder.add_leg("apply", time.perf_counter() - t0)
        return result

    def timed_index(self, s):
        # Outside a build, hand back the untouched lazy generator. `_index_sequence` wraps a
        # routed all-to-all, and forcing it eagerly would move a collective relative to its
        # consumer -- `contains_sequence` (manybody_basis.py:700) yields lazily, and reordering
        # a collective is how this repo's deadlocks start. Inside a build the sole call site
        # already materialises immediately (basis_transcription.py:195), so timing it eagerly
        # there changes nothing.
        if not recorder.in_build or recorder._index_depth:
            return original["index"](self, s)
        recorder._index_depth += 1
        try:
            t0 = time.perf_counter()
            result = list(original["index"](self, s))
            recorder.add_leg("index", time.perf_counter() - t0)
            recorder.add_lookups(len(result))
        finally:
            recorder._index_depth -= 1
        return iter(result)

    def timed_state(basis, vs, slaterWeightMin=0):
        t0 = time.perf_counter()
        result = original["state"](basis, vs, slaterWeightMin)
        recorder.marshal_seconds += time.perf_counter() - t0
        return result

    def timed_vector(basis, psis):
        t0 = time.perf_counter()
        result = original["vector"](basis, psis)
        recorder.marshal_seconds += time.perf_counter() - t0
        return result

    def timed_eigen(self, *args, **kwargs):
        n_before = len(recorder.builds)
        marshal_before = recorder.marshal_seconds
        t0 = time.perf_counter()
        result = original["eigen"](self, *args, **kwargs)
        total = time.perf_counter() - t0
        build = sum(b["seconds"] for b in recorder.builds[n_before:])
        # `build_state`/`build_distributed_vector` are per-determinant Python loops that run
        # inside every solve (dict setitem per (state, column)). Without this leg they hide
        # inside "eigensolve" and get mistaken for Krylov work.
        marshal = recorder.marshal_seconds - marshal_before
        recorder.solves.append(
            {
                "seconds": total,
                "build_seconds": build,
                "marshal_seconds": marshal,
                "solve_seconds": total - build - marshal,
            }
        )
        return result

    cipsi_solver.build_sparse_matrix = timed_build
    basis_transcription.build_local_operator_list = timed_apply
    Basis._index_sequence = timed_index
    cipsi_solver.build_state = timed_state
    cipsi_solver.build_distributed_vector = timed_vector
    cipsi_solver.CIPSISolver.get_eigenvectors = timed_eigen

    def restore():
        cipsi_solver.build_sparse_matrix = original["build"]
        basis_transcription.build_local_operator_list = original["apply"]
        Basis._index_sequence = original["index"]
        cipsi_solver.build_state = original["state"]
        cipsi_solver.build_distributed_vector = original["vector"]
        cipsi_solver.CIPSISolver.get_eigenvectors = original["eigen"]

    return restore


def _capture_final_state(store, recorder):
    """Wrap ``expand``: number the expands, and keep the basis/operator the last one finished on."""
    from impurityModel.ed import cipsi_solver

    original = cipsi_solver.CIPSISolver.expand

    def wrapped(self, H, *args, **kwargs):
        recorder.expand_index += 1
        result = original(self, H, *args, **kwargs)
        store["basis"] = self.basis
        store["H"] = H
        store["psi_refs"] = getattr(self, "psi_refs", None)
        return result

    cipsi_solver.CIPSISolver.expand = wrapped

    def restore():
        cipsi_solver.CIPSISolver.expand = original

    return restore


def _stop_after_gs():
    """Abort the run once the ground state is done -- the GF phase is not being measured."""
    from impurityModel.ed import selfenergy

    original = selfenergy.get_Greens_function

    def stop(*args, **kwargs):
        raise _StopAfterGroundState

    selfenergy.get_Greens_function = stop

    def restore():
        selfenergy.get_Greens_function = original

    return restore


def _hash_block(basis, width):
    """A block with the whole local basis as support -- what a Krylov vector looks like here."""
    from impurityModel.ed.cipsi_solver import _amplitude_from_hash
    from impurityModel.ed.ManyBodyUtils import ManyBodyState

    columns = []
    for k in range(width):
        amps = {state: _amplitude_from_hash(state.get_hash() + k) for state in basis.local_basis}
        columns.append(ManyBodyState(amps, width=1))
    return ManyBodyState.from_states(columns)


def _time_matrix_free_step(basis, H, width, repeats=3):
    """One projected matrix-free step: apply, redistribute, drop the out-of-basis rows.

    This is the step a frozen "admit nothing" basis proxy would run every iteration -- the
    projection is not free, and timing a bare apply would understate it.
    """
    from impurityModel.ed.ManyBodyUtils import ManyBodyState

    block = _hash_block(basis, width)
    mask = ManyBodyState.from_states([ManyBodyState(dict.fromkeys(basis.local_basis, 1.0 + 0j), width=1)])
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = H.apply_block(block, 0.0)
        out = basis.redistribute_block(out)
        out.keep_rows(mask)
        best = min(best, time.perf_counter() - t0)
    return best


def _time_spmv(basis, h_local, width, repeats=5):
    """One distributed sparse matvec: ``H[:, local] @ V`` plus the full-length reduction."""
    v = np.asfortranarray(np.random.default_rng(0).random((len(basis.local_basis), width)) + 0j)
    h_csr = h_local.tocsr()
    out = np.empty((len(basis), width), dtype=complex)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        partial = h_csr @ v
        if basis.is_distributed:
            basis.comm.Allreduce(np.ascontiguousarray(partial), out, op=MPI.SUM)
        best = min(best, time.perf_counter() - t0)
    return best


def _time_tocsr(h_local, repeats=3):
    """The CSC -> CSR conversion the array kernel does on every solve.

    ``BlockLanczosArray.pyx:482`` converts inside ``block_lanczos_array_cy``, i.e. once per
    ``restarted_lanczos`` call -- once per CIPSI cycle, right after each build. It is an
    O(nnz) full copy that the attribution otherwise hides inside ``solve_seconds``, and it
    decides what an incremental cache should *store*: caching a CSC that gets re-converted
    every cycle would leave this cost in place.
    """
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        h_local.tocsr()
        best = min(best, time.perf_counter() - t0)
    return best


def _report(recorder, extra, comm):
    rank = comm.rank if comm is not None else 0
    builds = recorder.builds
    payload = {"builds": builds, "solves": recorder.solves, **extra}
    if rank != 0:
        return payload
    print("\n=== P0.2  build attribution ===")
    print(
        f"{'exp':>4} {'#':>3} {'|B|':>10} {'build s':>9} {'apply':>8} {'index':>8} "
        f"{'assembly':>9} {'nnz':>12} {'MiB':>8} {'lookups':>10} {'us/look':>8}"
    )
    for i, b in enumerate(builds):
        n_look = b.get("index_lookups", 0)
        per = 1e6 * b["index_seconds"] / n_look if n_look else 0.0
        print(
            f"{b['expand']:>4} {i:>3} {b['basis_size']:>10,} {b['seconds']:>9.3f} {b['apply_seconds']:>8.3f} "
            f"{b['index_seconds']:>8.3f} {b['assembly_seconds']:>9.3f} {b['nnz']:>12,} "
            f"{b['matrix_bytes'] / 2**20:>8.1f} {n_look:>10,} {per:>8.1f}"
        )
    # Per expand: the walk's trial expands and the refinement expand have different sizes, and
    # a cache lives inside one expand, so the ceiling is a per-expand quantity.
    print("\nincremental ceiling, per expand (sum|B_i|/|B_final|):")
    for expand_id in sorted({b["expand"] for b in builds}):
        rows = [b for b in builds if b["expand"] == expand_id]
        total_sizes = sum(b["basis_size"] for b in rows)
        final = rows[len(rows) - 1]["basis_size"]
        seconds = sum(b["seconds"] for b in rows)
        print(
            f"  expand {expand_id:>3}: calls={len(rows):>3}  sum|B_i|={total_sizes:>12,}  "
            f"|B_final|={final:>10,}  ceiling={total_sizes / max(final, 1):>5.2f}x  build={seconds:>7.2f}s"
        )
    t_build = sum(s["build_seconds"] for s in recorder.solves)
    t_marshal = sum(s.get("marshal_seconds", 0.0) for s in recorder.solves)
    t_solve = sum(s["solve_seconds"] for s in recorder.solves)
    total = t_build + t_marshal + t_solve
    if total > 0:
        print(
            f"\nbuild {t_build:.2f}s ({100 * t_build / total:.1f}%)   "
            f"state<->array marshalling {t_marshal:.2f}s ({100 * t_marshal / total:.1f}%)   "
            f"Krylov {t_solve:.2f}s ({100 * t_solve / total:.1f}%)"
        )
    if "matrix_free_seconds" in extra:
        print("\n=== P0.3  matrix vs matrix-free ===")
        t_free, t_spmv = extra["matrix_free_seconds"], extra["spmv_seconds"]
        print(f"width={extra['width']}  matrix-free step {t_free * 1e3:.2f} ms   spmv {t_spmv * 1e3:.2f} ms")
        if t_free > t_spmv and builds:
            breakeven = builds[len(builds) - 1]["seconds"] / (t_free - t_spmv)
            print(f"matrix pays for itself after {breakeven:.1f} matvecs (ratio {t_free / max(t_spmv, 1e-12):.0f}x)")
        print(
            f"per-solve CSC->CSR conversion {extra['tocsr_seconds'] * 1e3:.2f} ms "
            f"({extra['tocsr_seconds'] / max(builds[len(builds) - 1]['seconds'], 1e-12):.1%} of one build)"
        )
    return payload


@pytest.mark.skipif(not _ENABLED, reason="set RUN_MATRIX_BUILD_BENCH=1 to run the build attribution")
@pytest.mark.skipif(not os.path.exists(_ARCHIVE), reason=f"workload archive not available: {_ARCHIVE}")
def test_matrix_build_attribution():
    """Measure, do not predict: the plan's Phase-0 gates all read off this run."""
    from impurityModel.ed.basis_transcription import build_sparse_matrix
    from impurityModel.test.support.real_workload import load_workload, run_selfenergy

    comm = MPI.COMM_WORLD
    workload = load_workload(_ARCHIVE)
    trunc = os.environ.get("MATRIX_BENCH_TRUNC", "archive")
    if trunc != "archive":
        trunc = float(trunc)

    recorder = _Recorder()
    final = {}
    restores = [_install(recorder), _capture_final_state(final, recorder), _stop_after_gs()]
    try:
        try:
            run_selfenergy(workload, comm=comm, truncation_threshold=trunc, n_iw=0, n_w=0, verbosity=0)
        except _StopAfterGroundState:
            pass
    finally:
        for restore in reversed(restores):
            restore()

    assert recorder.builds, "no build_sparse_matrix call was recorded -- the wrapper did not take"

    extra = {"archive": _ARCHIVE, "truncation_threshold": None if trunc == "archive" else trunc, "width": _WIDTH}
    basis, H = final.get("basis"), final.get("H")
    if basis is not None:
        h_local = build_sparse_matrix(basis, H)
        if basis.is_distributed:
            h_local = h_local[:, basis.local_indices]
        extra["matrix_free_seconds"] = _time_matrix_free_step(basis, H, _WIDTH)
        extra["spmv_seconds"] = _time_spmv(basis, h_local, _WIDTH)
        extra["tocsr_seconds"] = _time_tocsr(h_local)

    payload = _report(recorder, extra, comm)
    out_path = os.environ.get("MATRIX_BENCH_JSON")
    if out_path and (comm is None or comm.rank == 0):
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {out_path}")
