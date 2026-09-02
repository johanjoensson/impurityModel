"""Serial and MPI must agree on the ground-state basis and on the Green's function.

`de08d77` made the adaptively-selected CIPSI basis independent of `comm.size`, but nothing in the
suite pins that end to end, so it can rot silently: the ground-state *energy* is reproducible even
when the selected determinants are not, which is exactly why the original bug survived so long.

Each test runs the production path twice -- once on the real communicator, once on `MPI.COMM_SELF`
-- and compares. Every rank recomputes the serial reference on its own `COMM_SELF`, which is
redundant but symmetric: no collective can be reached by a subset of ranks, so a mistake here fails
rather than deadlocks. The reference is computed *in the test*, not hard-coded, because a frozen
constant rots the moment the workload is retuned.

Tolerances. The determinant set must be **bit-identical** -- it is a discrete object, and any
difference is structural. The Green's function must not be: MPI reduction order alone perturbs it.
Measured on this workload, serial vs 2 and 3 ranks:

    gs_energies    rel 9.1e-16 / 1.7e-15
    gs_realaxis    rel 1.9e-12 / 2.8e-12
    sigma_real     rel 2.1e-12 / 3.0e-12

so `_GF_RTOL = 1e-10` leaves ~30x margin over roundoff while still catching the class of bug this
file exists for: before `de08d77` the same comparison read 5.3e-9 at 2 ranks and 1.6e-7 at 3.
`_E_RTOL = 1e-10` gives the same margin discipline for the ground-state energy sum: measured
agreement here is ~1e-15, but a bigger production workload sums over many more Slater
determinants in a different order, so the *measured* margin on this small fixture shouldn't be
mistaken for the guarantee -- match `_GF_RTOL`'s scale rather than pin to what this run delivers.
"""

import itertools

import numpy as np
import pytest

try:
    from mpi4py import MPI

    _has_mpi = True
except ImportError:  # pragma: no cover - mpi4py is a hard dependency in practice
    _has_mpi = False

from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.eigensolvers import scipy_eigensystem
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant
from impurityModel.ed.selfenergy import calc_selfenergy
from impurityModel.test.support._nio_workload import (
    as_calc_selfenergy_args,
    build_ground_state_workload,
    build_selfenergy_inputs,
)

# Re-enabled 2026-09-01 after the reproduction attempt below came back empty. Kept in one
# piece deliberately: what is written here is the evidence, so the next person who sees an
# exit-139 in CI does not repeat this hunt -- and does not reach for the skip marker first.
#
# The history. This file was skipped (not xfailed -- an xfail still runs the body, and an
# intermittent SIGSEGV kills the whole process rather than raising something xfail can absorb)
# behind an MPICH "BAD TERMINATION", exit 139, in CI. It first appeared once the numeric
# test-suite fixes let CI reach the MPI steps at all. It hit three different legs (intel/c++17,
# gcc-12/c++20/coverage, gcc-12/c++17) at three different points *within this one file* (0, 2
# and 2 dots in), and the crash rate scaled with rank count (5/8 legs at -n2/-n3 on one run
# against 1-2/8 normally) -- the signature of heap corruption that surfaces wherever the
# corrupted memory next gets touched, not of a logic bug at a fixed place.
#
# What CI has shown since, which corrects the paragraph above. Twelve crashes are on record, over
# six crashing runs (a seventh full run came back 0 of 8 -- the rate is 0-3 legs per run).
# **Every one of the twelve happened on a step where these tests were skipped, deselected, or had
# already finished**, so the crash is not in them and skipping them was never going to stop it.
# (The deselect is confirmed by the steps' own counts: 23 deselected against serial's 18, i.e.
# these five.)
#
# Where they land -- three sites, none of them random:
#
#   * inputformat/test_f_shell_crystal_field.py, 5 crashes, at 7-8 of its 10 dots.
#   * restrictions/test_excitation_budget.py, 3 crashes, at 10-11 of its 19 dots.
#   * End of run, 4 crashes: three just after symmetry/test_symmetry_observables.py (the last file
#     the suite collects), one after this file's own isolated -n 3 step reached [100%]. None
#     printed pytest's `N passed in Xs` summary, though earlier steps on the same leg did -- so
#     these are finalize-time, after the last test and before teardown finished.
#
# Each mid-suite crash landed on its file's one heavyweight full-stack test, not anywhere within
# it: test_f_shell_crystal_field.py's 9th of 10 (test_an_f_shell_crystal_field_model_solves) is
# the only one there that runs a solver at all -- the other nine are input-format validation --
# and test_excitation_budget.py's 12th of 19 (test_calc_selfenergy_excitation_budget_oracle) runs
# calc_selfenergy twice through the full driver and GF stack. Dots are flushed as tests finish and
# a partial line is lost when the process dies, so the counts are lower bounds; the windows are
# tests 8-10 and 11-13.
#
# Nothing in the configuration explains any of it. Across the twelve: gcc-12 eight times,
# clang-15 twice, intel twice; -std=c++17 five and -std=c++20 seven; `parallel` on three,
# `coverage` on three. And **every launch mode has now produced one** -- serial three times,
# mpiexec -n 1 once, -n 2 twice, -n 3 six times -- which closes rank count as a variable the way
# the others were already closed. They share no ingredient the passing legs lack: what organizes
# these crashes is *where in the run* they happen and *what the test does*, not how the extension
# was compiled or how it was launched.
#
# What the serial ones do and do not rule out. Three of the twelve ran under a bare `pytest` --
# two finalize-time, one mid-suite in test_excitation_budget.py -- so both clusters are reachable
# with no ranks, no message passing and no communicator lifetimes. That retires the _graph_comm_cache
# theory below on its own terms as well as by the measurement. It does **not** rule out MPI: 72
# modules import mpi4py at module scope, so a bare `pytest` still initializes MPI on import and
# finalizes it at exit -- verified locally, importing manybody_basis alone leaves
# MPI.Is_initialized() true. So "multi-rank MPI", never "MPI", is the thing the serial crashes
# exclude.
#
# That also makes the next probe much cheaper than anything tried below. A single-process,
# single-rank crash needs no MPICH build and no rank sweep -- run the whole suite under
# AddressSanitizer serially and watch the exit path.
#
# What was tried, all negative:
#   * The whole file, 500 iterations alternating -n 2 and -n 3, plus 50 interleaved full-suite
#     --with-mpi runs, under MPICH 4.2.2 (ch4:ofi) with mpi4py built from source against it --
#     i.e. under the CI MPI family, which the old skip comment's "not locally reproducible
#     under Open MPI" had never actually tested. 550 runs over ~9.5 hours, every one rc=0.
#     At the CI rate (1-2 of 8 legs, once 5 of 8) a few dozen runs should have been enough,
#     so this is a negative about *this machine*, not a bound on the crash's rate in CI.
#   * The same file and the full suite under AddressSanitizer, MPICH-linked, at -n 2 and -n 3.
#     Zero reports. This is the stronger negative: ASan traps the bad read/write when it
#     happens, so a run that corrupts the heap but would have survived is still caught.
#     (The one failure in that run was test_suggest_threshold_fits_budget, the known
#     free-RAM-derived flake, over budget by 0.015% -- not a memory-safety finding.)
#   * The narrowed target, once CI named it: 120 runs of test_f_shell_crystal_field.py alone,
#     alternating -n 2 and -n 3 under Open MPI. All rc=0. Consistent with every other local
#     attempt -- whatever discriminates is not on this machine, so the next probe belongs in CI.
#   * The _graph_comm_cache-leak theory (mpi_comm.py keys it on id(comm) and never evicts, and
#     the GF layer clones a communicator per unit): measured at 10 live entries after an entire
#     -n 3 suite run, across 18387 lookups. Nowhere near MPICH's context-id space. Refuted.
#
# What is still untested locally, reordered by the clusters above: the exit path of a serial run
# (never instrumented here -- the ASan work all ran under mpiexec), coverage instrumentation,
# compiler (gcc-12/clang-15/icpx at -std=c++17/20/2b in CI, gcc 16.2.1 locally), and 4 vCPUs
# against 8 cores. MPICH version (ubuntu-22.04 ships 4.0, the reproduction ran 4.2.2) drops near
# the bottom now that a single-rank process has crashed the same way.
#
# The probe for all of that is now live: the `test-asan` job in .github/workflows/tests.yml was
# commented out and is re-enabled, pointed at the two named files rather than the whole suite --
# which is what makes it affordable to loop, and looping is the point when the crash needs 1-3 of
# 8 legs to fire. It also runs one serial full-suite pass, for the other hypothesis (heap
# corrupted earlier, those tests merely touch it first) and for the finalize-time pair.
# `workflow_dispatch` takes an iteration count, so another sample costs a button rather than a
# push. Why it never worked before: `g++ -print-file-name=libasan.so` resolves to an ASCII
# *linker script*, not a preloadable object, so LD_PRELOAD-ing it silently did nothing and every
# run came back clean while uninstrumented. It now asks for the SONAME, verifies ELF magic, and
# checks a built extension carries __asan_ symbols -- because a self-test that compiles its own
# binary with -fsanitize=address proves ASan works, not that this build was instrumented. And
# mpi4py is installed with the sanitizer flags unset: measured, importing it aborts ASan even
# with rc.initialize=False, because LDFLAGS reaches its link step while CXXFLAGS never reaches
# its C sources, leaving it linked against libasan without being instrumented.
#
# This file is nonetheless deselected from the three standing MPI legs in
# .github/workflows/tests.yml and run in three steps of its own at the end of that job. Read that
# as cheap insurance, not as containment: the evidence above says the crash is not this file's, so
# the isolation bounds only what a recurrence *here* would cost -- one failed step, after the rest
# of the leg has already reported. That collateral damage, not the crash itself, is what the skip
# was really buying, and it is now bought for the price of three extra steps.
#
# One already has, and it behaved as designed: on run 33673571249 the gcc-12/-std=c++17 leg
# segfaulted in "Run rank-invariance guards (3 ranks)" -- *after* the step reached [100%], all
# five tests passed, i.e. at finalize, which is the third site above and not a failure of these
# tests. One step went red; the serial, -n 1 and -n 2 results on that leg had already reported,
# which under the old arrangement they would not have.
#
# So: diagnose a 139 from that step, do not skip the file again. These are the only end-to-end
# rank-invariance guards there are, and two solver bugs got through the rest of the suite while
# they were switched off.

_MASK = (1 << 64) - 1
_GF_RTOL = 1e-10
_E_RTOL = 1e-10

_NBATHS = 10
_DE2_MIN = 1e-6
_DENSE_CUTOFF = 50
# Was 1000, but the two build_ground_state_workload() call sites below never actually passed
# it through (dead constant): they silently ran at that function's own default of 30000,
# which lets the 4-impurity-group NiO-10-bath walk (groundstate.py's diagonal probe) grow
# ~132 trial CIPSI solves to a 1000+-determinant basis each, ~88s per call. Now threaded
# through explicitly; 300 reproduces the same rank-independence property (a differential
# world-vs-COMM_SELF comparison, not a golden numeric target) in ~18s.
_TRUNCATION = 300
_N_OMEGA = 32

# One-body Hamiltonian over 10 spin-orbitals for the fast truncate_initial regression below:
# a fixed (seeded, not per-run-random) generic Hermitian one-body matrix, deliberately NOT
# built from a symmetric physical model (eg/t2g-like couplings gave the ground eigenvector
# exact fourfold weight ties -- e.g. four determinants each carrying exactly 1/4 of the norm --
# which truncate()'s cutoff-at-target selection resolves by comparing floating-point importance
# scores; under an *exact* tie, roundoff from a different MPI reduction order can push different
# members of the tied group across the cutoff on different rank counts. That is a genuine
# indeterminacy of ranking a degenerate manifold, not a bug this file's fix addresses, so the
# test avoids it: a generic random one-body matrix gives every determinant's ground-eigenvector
# weight a distinct value with large (>1e-3 relative) gaps at the truncation boundary, which
# floating-point roundoff cannot cross.
_rng = np.random.default_rng(7)
_h1 = _rng.normal(size=(10, 10)) + 1j * _rng.normal(size=(10, 10))
_h1 = _h1 + _h1.conj().T
_TRUNCATE_INITIAL_HOP = {((i, "c"), (j, "a")): complex(_h1[i, j]) for i in range(10) for j in range(10)}
del _rng, _h1


def _det(occupied):
    """SlaterDeterminant with the given orbitals occupied (MSB-first bit convention)."""
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _basis_fingerprint(basis, comm):
    """Order-independent fingerprint ``(count, sum, xor)`` of the *global* determinant set.

    Determinants are hash-routed with one owner per rank, so the local sets are disjoint and a
    SUM/XOR reduction reconstructs the global set exactly, independently of how it was partitioned.
    Both a sum and an xor, because either alone can collide.
    """
    total = 0
    parity = 0
    for det in basis.local_basis:
        h = det.get_hash() & _MASK
        total = (total + h) & _MASK
        parity ^= h
    count = len(basis.local_basis)
    if comm is not None and comm.size > 1:
        total = comm.allreduce(total, op=MPI.SUM) & _MASK
        parity = comm.allreduce(parity, op=MPI.BXOR)
        count = comm.allreduce(count, op=MPI.SUM)
    return count, total, parity


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return float(np.max(np.abs(a - b))) / scale


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
def test_cipsi_ground_state_basis_is_rank_independent():
    """The selected determinant set is a discrete object: it must be bit-identical, not close."""
    world = MPI.COMM_WORLD

    distributed = build_ground_state_workload(
        nBaths=_NBATHS,
        de2_min=_DE2_MIN,
        dense_cutoff=_DENSE_CUTOFF,
        truncation_threshold=_TRUNCATION,
        comm=world,
        verbose=False,
    )
    serial = build_ground_state_workload(
        nBaths=_NBATHS,
        de2_min=_DE2_MIN,
        dense_cutoff=_DENSE_CUTOFF,
        truncation_threshold=_TRUNCATION,
        comm=MPI.COMM_SELF,
        verbose=False,
    )

    got = _basis_fingerprint(distributed["basis"], world)
    ref = _basis_fingerprint(serial["basis"], MPI.COMM_SELF)

    assert got == ref, (
        f"the CIPSI ground-state basis depends on the rank count: {world.size} ranks selected "
        f"(count, sum, xor) = {got}, serial selected {ref}"
    )


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
def test_greens_function_and_selfenergy_are_rank_independent():
    """Transitively pins the *excited* basis too.

    `_block_green_group` seeds it from the whole thermal manifold, which is rotation invariant.
    Seeded instead from a single eigenvector of a degenerate manifold -- as a benchmark harness
    once did -- it differed by one determinant across rank counts and `G` moved by 1.6e-7, three
    orders above the tolerance here.
    """
    world = MPI.COMM_WORLD

    kwargs = build_selfenergy_inputs(
        nBaths=_NBATHS, n_omega=_N_OMEGA, truncation_threshold=_TRUNCATION, rank=world.rank, verbose=False
    )
    distributed = calc_selfenergy(**as_calc_selfenergy_args(kwargs), comm=world)

    kwargs_serial = build_selfenergy_inputs(
        nBaths=_NBATHS, n_omega=_N_OMEGA, truncation_threshold=_TRUNCATION, rank=0, verbose=False
    )
    serial = calc_selfenergy(**as_calc_selfenergy_args(kwargs_serial), comm=MPI.COMM_SELF)

    if world.rank != 0:
        return  # calc_selfenergy returns the result on the root of the passed communicator

    assert _rel(serial["gs_energies"], distributed["gs_energies"]) < _E_RTOL

    for key in ("gs_realaxis", "sigma_real", "sigma_static"):
        ref, got = serial.get(key), distributed.get(key)
        if ref is None or np.asarray(ref).shape == ():
            continue
        err = _rel(ref, got)
        assert err < _GF_RTOL, f"{key} depends on the rank count: rel {err:.3e} at {world.size} ranks"


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
def test_warm_started_eigensolver_delivers_more_than_its_start_block():
    """A warm start must not cap the number of eigenstates at its own block width.

    Before the deflation fix, `get_eigenvectors` warm-started from `psi_refs` produced a first
    Lanczos block whose residual `beta_0` *is* the eigenpair residual; `_cholesky_or_deflate`'s
    absolute rank floor deflated it to rank 0, the sweep declared an invariant subspace, and TRLM
    returned exactly `len(psi_refs)` pairs whatever `num_wanted` asked for. Measured on the 50-bath
    workload: 4 states returned for `num_wanted` = 1 *and* 10, against 20 from a cold start.

    That capped the thermal manifold at the warm start's width, and a partially-kept degenerate
    manifold has no rotation-invariant basis -- which is precisely how the Green's function became
    rank dependent.
    """
    world = MPI.COMM_WORLD
    workload = build_ground_state_workload(
        nBaths=_NBATHS,
        de2_min=_DE2_MIN,
        dense_cutoff=_DENSE_CUTOFF,
        truncation_threshold=_TRUNCATION,
        comm=world,
        verbose=False,
    )
    solver = workload["solver"]
    assert solver.psi_refs is not None, "the workload no longer leaves a warm start; the test is moot"
    width = len(solver.psi_refs)

    e_ref, _ = solver.get_eigenvectors(
        workload["h"],
        num_wanted=10,
        max_energy=None,
        dense_cutoff=_DENSE_CUTOFF,
        slaterWeightMin=1e-12,
        psi_refs=solver.psi_refs,
    )

    assert len(e_ref) > width, (
        f"warm-started eigensolver returned {len(e_ref)} states from a width-{width} start block; "
        "the sweep is deflating the warm start away instead of refining it"
    )


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
def test_truncate_initial_basis_is_rank_independent():
    """Direct, fast regression for the -n3 ARPACK deadlock in ``CIPSISolver.truncate_initial``.

    Guards the deadlock directly instead of through the ~18s end-to-end workload the other tests
    in this file use: builds a basis well above ``truncation_threshold`` so ``truncate_initial``
    actually truncates, with ``dense_cutoff`` low enough that it engages TRLM via
    ``get_eigenvectors`` -- the path that replaced the multi-rank ARPACK call which used to hang
    here (see the memory note n3-arpack-truncate-initial-deadlock). The surviving determinant set
    must be bit-identical to the serial (``COMM_SELF``) reference.
    """
    world = MPI.COMM_WORLD

    def _make_basis(comm):
        basis = Basis(
            impurity_orbitals={2: [list(range(10))]},
            bath_states=({2: [[]]}, {2: [[]]}),
            initial_basis=[],
            truncation_threshold=20,
            comm=comm,
            verbose=False,
        )
        basis.add_states([_det(occ) for occ in itertools.combinations(range(10), 5)])
        return basis

    distributed = _make_basis(world)
    CIPSISolver(distributed).truncate_initial(_TRUNCATE_INITIAL_HOP, dense_cutoff=30)

    serial = _make_basis(MPI.COMM_SELF)
    CIPSISolver(serial).truncate_initial(_TRUNCATE_INITIAL_HOP, dense_cutoff=30)

    got = _basis_fingerprint(distributed, world)
    ref = _basis_fingerprint(serial, MPI.COMM_SELF)
    assert got == ref, (
        f"truncate_initial's surviving basis depends on the rank count: {world.size} ranks kept "
        f"(count, sum, xor) = {got}, serial kept {ref}"
    )


@pytest.mark.mpi
@pytest.mark.skipif(not _has_mpi, reason="mpi4py not available")
def test_scipy_eigensystem_root_driven_matches_serial_with_an_empty_rank():
    """Regression for the root-driven ``scipy_eigensystem`` rewrite (Phase 2 of the -n3 deadlock
    fix): the hardened ARPACK driver must still return the correct spectrum -- and, above all,
    must not hang -- when one rank owns no columns at all.

    ``h_local`` is deliberately column-distributed so that the *last* rank owns zero columns
    whenever ``comm.size > 1``: the empty-rank edge case that has bitten this codebase before
    (CLAUDE.md; memory note array-lanczos-empty-rank-int32-deadlock). At -n3 this lands exactly
    on rank 2, mirroring the original failure's rank/size combination.
    """
    world = MPI.COMM_WORLD
    n = 8
    rng = np.random.default_rng(1234)
    H_full = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H_full = H_full + H_full.conj().T
    ref = np.sort(np.linalg.eigvalsh(H_full))

    size = world.size
    h_local = np.zeros_like(H_full)
    if size == 1 or world.rank != size - 1:
        owners = size - 1 if size > 1 else 1
        owner_id = world.rank if size > 1 else 0
        my_cols = [c for c in range(n) if c % owners == owner_id]
        h_local[:, my_cols] = H_full[:, my_cols]
    # else: the last rank of a size > 1 run deliberately owns no columns.

    n_target = 3
    e_max = ref[n_target - 1] - ref[0] + 1e-6
    es, _ = scipy_eigensystem(h_local, e_max=e_max, k=n_target, comm=world, return_eigvecs=True)

    assert len(es) >= n_target
    got = np.sort(es)[:n_target]
    assert np.allclose(got, ref[:n_target], atol=1e-6), (
        f"root-driven scipy_eigensystem at {size} ranks (one empty) disagrees with the serial "
        f"reference: got {got}, expected {ref[:n_target]}"
    )
