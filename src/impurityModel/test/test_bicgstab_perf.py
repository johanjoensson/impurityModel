"""Benchmark + memory baseline for ``cg.block_bicgstab`` on the sparse (``ManyBodyBlockState``) path.

Phase 0 of the per-frequency Green's-function plan. The target workload is *not* the RIXS
resolvent (where the solve is amortized over other work) but the Green's-function solve the
per-frequency driver will run once per mesh point:

    (w + i*delta + E0 - H) X = c_j^dag |psi0>,   G[i, j] = <seed_i | X_j>

so the harness builds the NiO d-shell ground state, applies the addition transition operators
of one symmetry block, and times a single ``block_bicgstab`` on the excited basis.

Two frequencies are measured because they bracket the conditioning the driver will face:

* **real axis**, ``delta = 0.2`` -- the production broadening; well conditioned.
* **first fermionic Matsubara**, ``i*pi*T`` at ``T = tau`` -- a much smaller imaginary shift,
  which is where a per-frequency solve gets hard. This is the axis the plan wants benchmarked
  against block Lanczos first.

Marked ``benchmark`` (skipped by default) with a second ``RUN_BICGSTAB_BENCH=1`` env gate,
mirroring ``test_selfenergy_perf.py`` / ``test_rixs_tensor_perf.py``::

    RUN_BICGSTAB_BENCH=1 pytest -s -m benchmark src/impurityModel/test/test_bicgstab_perf.py
    RUN_BICGSTAB_BENCH=1 mpiexec -n 2 pytest -s -m benchmark --with-mpi \
        src/impurityModel/test/test_bicgstab_perf.py

Set ``BICG_BENCH_PROFILE=1`` to also dump a ``cProfile`` table of the solve. On the baseline
(50 bath, width 5, 3863 determinants, 5 iterations, 0.77 s) that profile reads:

===========================================  ========  =====
frame                                        tottime      %
===========================================  ========  =====
``cg.py:212 <genexpr>``  (``not in`` scan)     0.477 s    62
``ManyBodyOperator.apply_block`` (12 calls)    0.223 s    29
everything else                                0.067 s     9
===========================================  ========  =====

The top frame is ``state not in basis.local_basis`` -- a linear scan of a Python *list*, run
once per determinant of the support, twice per iteration. It costs twice what all twelve
matvecs cost together, and it grows quadratically with the basis. The ``seen_states`` hash set
and the ``support_keys`` materialization, by contrast, are a few percent at this size: their
case is memory, not time.

Sizing knobs (environment):
    BICG_NBATHS   which ``h0/h0_NiO_<n>bath.pickle`` to load (default 50)
    BICG_MV       mixed-valence window (default 2; 0 collapses the GS sector)
    BICG_TRUNC    CIPSI basis-size cap (default 200000)
    BICG_DE2_MIN  CIPSI Epstein-Nesbet selection threshold (default 1e-12)
    BICG_WIDTHS   comma-separated block widths to sweep (default "1,5")
    BICG_DELTA    real-axis broadening (default 0.2)
    BICG_ATOL     block_bicgstab absolute tolerance (default 1e-8)
    BICG_RTOL     block_bicgstab relative tolerance (default 1e-12)
"""

import cProfile
import io
import os
import pstats
import threading
import time

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.greens_function import _build_excited_restrictions
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, applyOp, inner

RUN = os.environ.get("RUN_BICGSTAB_BENCH") == "1"
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not RUN, reason="Set RUN_BICGSTAB_BENCH=1 to run the block_bicgstab benchmark."),
]

NBATHS = int(os.environ.get("BICG_NBATHS", "50"))
MIXED_VALENCE = int(os.environ.get("BICG_MV", "2"))
TRUNC = int(os.environ.get("BICG_TRUNC", "200000"))
# The CIPSI selection threshold, not the bath count, sets the ground-state basis size:
# at the 1e-6 default the NiO anchor selects ~600 determinants and the excited basis the
# solver grows from it never leaves the toy regime.
DE2_MIN = float(os.environ.get("BICG_DE2_MIN", "1e-12"))
WIDTHS = [int(w) for w in os.environ.get("BICG_WIDTHS", "1,5").split(",")]
DELTA = float(os.environ.get("BICG_DELTA", "0.2"))
ATOL = float(os.environ.get("BICG_ATOL", "1e-8"))
RTOL = float(os.environ.get("BICG_RTOL", "1e-12"))
PROFILE = os.environ.get("BICG_BENCH_PROFILE") == "1"

DENSE_CUTOFF = 50
SLATER_WEIGHT_MIN = 1e-12
DN = 2  # excited-sector impurity occupation window, as get_Greens_function uses

MB = float(2**20)


def _comm():
    return MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None


def _rank0(comm):
    return comm is None or comm.rank == 0


def _report(comm, title, rows):
    if not _rank0(comm):
        return
    width = max(len(k) for k, _ in rows)
    print(f"\n=== {title} ===")
    for key, val in rows:
        print(f"  {key:<{width}}  {val}")


class RssSampler:
    """Sample this process's resident set size from ``/proc/self/statm`` in a thread.

    ``peak_delta_bytes`` is the peak RSS during the sampled window minus the RSS at
    ``start()``. Heap freed back to the allocator but not to the OS is invisible, so
    treat the result as a lower bound on transients.
    """

    def __init__(self, interval=0.01):
        self._interval = interval
        self._page = os.sysconf("SC_PAGE_SIZE")
        self._stop = threading.Event()
        self._thread = None
        self.baseline_bytes = 0
        self.peak_bytes = 0

    def _read_rss(self):
        with open("/proc/self/statm", "rb") as f:
            return int(f.read().split()[1]) * self._page

    def _run(self):
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._read_rss())
            self._stop.wait(self._interval)

    def start(self):
        self.baseline_bytes = self._read_rss()
        self.peak_bytes = self.baseline_bytes
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._read_rss())

    @property
    def peak_delta_bytes(self):
        return self.peak_bytes - self.baseline_bytes


class CountingOperator:
    """Pass-through wrapper counting ``apply_block`` matvecs.

    ``block_bicgstab`` only ever touches ``set_restrictions`` and ``apply_block`` on the
    sparse path, so this is enough to recover the iteration count without editing the
    solver: one matvec for the initial residual, then two per iteration.
    """

    def __init__(self, op):
        self.op = op
        self.n_apply = 0

    def set_restrictions(self, restrictions):
        self.op.set_restrictions(restrictions)

    def apply_block(self, block, cutoff=0.0):
        self.n_apply += 1
        return self.op.apply_block(block, cutoff)

    @property
    def iterations(self):
        return max(0, (self.n_apply - 1) // 2)


@pytest.fixture(scope="module")
def gf_workload():
    """NiO ground state + the excited basis and seeds of one symmetry block's addition GF."""
    from impurityModel.test._nio_workload import build_ground_state_workload

    comm = _comm()
    work = build_ground_state_workload(
        nBaths=NBATHS,
        mixed_valence=MIXED_VALENCE,
        truncation_threshold=TRUNC,
        dense_cutoff=DENSE_CUTOFF,
        slater_weight_min=SLATER_WEIGHT_MIN,
        de2_min=DE2_MIN,
        comm=comm,
    )
    h, basis, solver = work["h"], work["basis"], work["solver"]

    es, psis = solver.get_eigenvectors(
        h, num_wanted=1, dense_cutoff=DENSE_CUTOFF, slaterWeightMin=SLATER_WEIGHT_MIN, solver="trlm"
    )
    e0 = float(np.min(es))
    psi0 = psis[int(np.argmin(es))]

    excited_restrictions, excited_weighted = _build_excited_restrictions(
        basis, h, [psi0], [e0], dN=DN, occ_cutoff=1e-12
    )
    # Addition (c^dag) seeds, ranked by seed norm. The NiO d8 ground state fills one spin
    # channel of the d shell completely, so c^dag annihilates those orbitals outright and a
    # naive prefix of the impurity orbitals would hand the solver a zero right-hand side.
    orbitals = sorted(o for sub in basis.impurity_orbitals.values() for orbs in sub for o in orbs)
    excitations = [applyOp(ManyBodyOperator({((orb, "c"),): 1}), psi0, SLATER_WEIGHT_MIN) for orb in orbitals]
    # psi0 is hash-distributed, so the seed norms are rank-local: reduce before ranking, or
    # different ranks would pick different orbitals and the collective solve would diverge.
    norms = np.array([s.norm2() for s in excitations], dtype=float)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, norms, op=MPI.SUM)
    order = np.argsort(-norms)[: max(WIDTHS)]
    seeds = [excitations[i] for i in order]
    block = [orbitals[i] for i in order]
    assert norms[order].min() > 1e-6, f"degenerate addition seeds on orbitals {block}"

    return {
        "h": h,
        "basis": basis,
        "comm": comm,
        "e0": e0,
        "seeds": seeds,
        "excited_restrictions": excited_restrictions,
        "excited_weighted": excited_weighted,
        "tau": work["tau"],
    }


def _solve(workload, width, z, sample_rss=True):
    """One warm-start-free ``block_bicgstab`` solve of ``(z + E0 - H) X = seeds[:width]``.

    ``sample_rss=False`` skips the sampler thread: cProfile attributes the sampler's blocking
    reads and lock acquisitions to the profiled call tree, which buries the solver.

    Returns ``(seconds, iterations, excited_basis, X, seeds, peak_rss_delta)``.
    """
    h, basis = workload["h"], workload["basis"]
    seeds = workload["seeds"][:width]

    excited_basis = basis.clone(
        initial_basis={state for p in seeds for state in p},
        restrictions=workload["excited_restrictions"],
        weighted_restrictions=workload["excited_weighted"],
        verbose=False,
    )
    if excited_basis.weighted_restrictions is not None:
        h.set_weighted_restrictions(excited_basis.weighted_restrictions)

    shift = z + workload["e0"]
    a_op = CountingOperator(shift - h)

    y = excited_basis.redistribute_psis(list(seeds))
    x0 = [ManyBodyState() for _ in y]

    sampler = RssSampler() if sample_rss else None
    if sampler is not None:
        sampler.start()
    t0 = time.perf_counter()
    x = block_bicgstab(a_op, x0, y, basis=excited_basis, slaterWeightMin=SLATER_WEIGHT_MIN, atol=ATOL, rtol=RTOL)
    elapsed = time.perf_counter() - t0
    if sampler is not None:
        sampler.stop()

    return elapsed, a_op.iterations, excited_basis, x, y, sampler.peak_delta_bytes if sampler else 0


def _residual_norm(workload, excited_basis, z, x, y):
    """Global ``||(z + E0 - H) X - Y||`` over the block, as a solve-quality check.

    ``applyOp`` is rank-local: a rank's amplitudes scatter contributions onto determinants
    other ranks own. Those must be summed (``redistribute_psis``) *before* the per-determinant
    residual is squared, or the norm double-counts partial contributions.
    """
    shift = z + workload["e0"]
    a_op = shift - workload["h"]
    ax = excited_basis.redistribute_psis([applyOp(a_op, xi, SLATER_WEIGHT_MIN) for xi in x])
    total = sum((axi - yi).norm2() for axi, yi in zip(ax, y))
    if excited_basis.comm is not None:
        total = excited_basis.comm.allreduce(total, op=MPI.SUM)
    return np.sqrt(total)


def _green_block(excited_basis, y, x):
    """``G[i, j] = <Y_i | X_j>`` -- what the per-frequency driver will actually return.

    ``y`` must be the seeds as redistributed onto ``excited_basis``, not the originals:
    ``inner`` is a rank-local sum over the shared support, so both sides need the same
    ownership layout for the ``Allreduce`` to reassemble the global inner product.
    """
    n = len(y)
    g = np.zeros((n, n), dtype=complex)
    for i, yi in enumerate(y):
        for j, xj in enumerate(x):
            g[i, j] = inner(yi, xj)
    if excited_basis.comm is not None:
        excited_basis.comm.Allreduce(MPI.IN_PLACE, g, op=MPI.SUM)
    return g


#: Blocks the sparse recurrence holds at once: xi, ri, pi, vi, si, ti, and the pinned r0_t/rhs.
_LIVE_BLOCKS = 7


def _block_working_set(rows, width):
    """Amplitude bytes of the blocks the recurrence keeps live, on the largest rank.

    Ignores the key vectors (shared support, ``n_bytes`` per row) and the per-iteration
    union-support temporaries ``block_add_scaled_cy`` allocates -- so this is the floor the
    algorithm implies, not the peak. Below ~1e5 determinants it stays under the RSS
    sampler's resolution, which is why both numbers are reported.
    """
    return _LIVE_BLOCKS * rows * width * 16


@pytest.mark.parametrize("axis", ["realaxis", "matsubara"])
def test_block_bicgstab_bench(gf_workload, axis):
    comm = gf_workload["comm"]
    if axis == "realaxis":
        z = 0.0 + 1j * DELTA
        label = f"w=0, delta={DELTA}"
    else:
        # First fermionic Matsubara frequency, T = tau.
        z = 1j * np.pi * gf_workload["tau"]
        label = f"i*pi*T, T={gf_workload['tau']}"

    rows = []
    for width in WIDTHS:
        elapsed, iters, excited_basis, x, y, peak = _solve(gf_workload, width, z)
        res = _residual_norm(gf_workload, excited_basis, z, x, y)
        g = _green_block(excited_basis, y, x)

        # The solver must actually have solved the system: the block residual has to sit at
        # the requested tolerance, otherwise the timings below measure a premature exit.
        assert np.isfinite(res)
        assert res < max(ATOL * 10 * np.sqrt(width), 1e-6), f"width {width}: residual {res:.3e}"
        assert np.all(np.isfinite(g))

        n_local = len(excited_basis.local_basis)
        locals_n = [n_local] if comm is None else comm.gather(n_local, root=0)
        peaks = [peak] if comm is None else comm.gather(peak, root=0)
        if _rank0(comm):
            per_iter = 1e3 * elapsed / max(iters, 1)
            rows += [
                (f"[width {width}] wall time", f"{elapsed:.3f} s over {iters} iterations"),
                (f"[width {width}] per iteration", f"{per_iter:.1f} ms"),
                (f"[width {width}] excited basis", f"{excited_basis.size} dets, local {locals_n}"),
                (f"[width {width}] block working set", f"{_block_working_set(max(locals_n), width) / MB:.1f} MiB"),
                (f"[width {width}] peak RSS delta", f"{max(peaks) / MB:.1f} MiB"),
                (f"[width {width}] block residual", f"{res:.3e}"),
                (f"[width {width}] Tr G", f"{np.trace(g):.6f}"),
            ]
        # No free_comm: Basis.clone inherits the parent's communicator rather than cloning it
        # (only the RIXS driver passes comm=sub_comm.Clone()), so freeing it here would tear
        # down the communicator the workload fixture and the next width still hold.

    ranks = "serial" if comm is None else f"{comm.size} ranks"
    _report(comm, f"block_bicgstab {axis} ({label}, NiO {NBATHS} bath, {ranks})", rows)


def test_block_bicgstab_profile(gf_workload):
    """Function-level attribution of one solve. Opt-in via ``BICG_BENCH_PROFILE=1``."""
    if not PROFILE:
        pytest.skip("Set BICG_BENCH_PROFILE=1 to profile the solve.")
    comm = gf_workload["comm"]
    width = max(WIDTHS)
    z = 0.0 + 1j * DELTA

    profiler = cProfile.Profile()
    profiler.enable()
    _elapsed, iters, _excited_basis, _x, _y, _peak = _solve(gf_workload, width, z, sample_rss=False)
    profiler.disable()

    if _rank0(comm):
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(25)
        print(f"\n=== cProfile: block_bicgstab width {width}, {iters} iterations ===")
        print(stream.getvalue())
