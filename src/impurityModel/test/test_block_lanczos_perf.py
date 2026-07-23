"""Benchmark + memory baseline for the Block Lanczos loops.

Phase 0 of ``doc/plans/blocklanczos_partial_perf_memory.md``: measures, on the same
NiO d-shell CIPSI ground-state basis (the production PARTIAL-reorthogonalization
workload), both loops that can run it:

* ``test_partial_trlm_array_bench`` — the path CIPSI actually takes today
  (``cipsi_solver.get_eigenvectors``): sparse CSR ``H_mat`` (column-sliced under
  MPI) through the dispatching ``thick_restart_block_lanczos`` →
  ``block_lanczos_array_cy``.
* ``test_partial_sparse_kernel_bench`` — the hash-distributed ``ManyBodyState``
  kernel ``block_lanczos_cy`` (the memory-scalable alternative), with the
  per-phase time profile and the memory breakdown (Krylov-basis bytes, W buffer,
  dense fill ratio, transient reort spikes).

Both tests print a rank-0 report and assert correctness (orthonormality of the
Krylov basis, ground-state energy agreement between the two kernels), so the
harness doubles as a PARTIAL-mode regression.

Sizing knobs (environment):
    BLBENCH_NBATHS   which ``h0/h0_NiO_<n>bath.pickle`` to load (default 10)
    BLBENCH_MV       mixed-valence window (impurity occupation slack; default 1 —
                     the zero window of the 10-bath anchor pins every group's
                     occupation and collapses the GS sector to a trivial basis)
    BLBENCH_TRUNC    CIPSI basis-size cap / truncation threshold (default 30000)
    BLBENCH_MAXITER  sparse-kernel fixed iteration budget (default 40)
    BLBENCH_BLOCK    sparse-kernel block width p (default 2)
    BLBENCH_REPS     timing repetitions for the TRLM bench (default 3)

Run:
    python -m pytest -m benchmark -s src/impurityModel/test/test_block_lanczos_perf.py
    mpiexec -n 2 python -m pytest -m benchmark --with-mpi -s src/impurityModel/test/test_block_lanczos_perf.py
"""

import os
import random
import threading
import time

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import atomic_physics
from impurityModel.ed.basis_transcription import build_distributed_vector, build_sparse_matrix
from impurityModel.ed.BlockLanczos import (
    block_lanczos_cy,
    enable_block_lanczos_profile,
    get_block_lanczos_profile,
    reset_block_lanczos_profile,
)
from impurityModel.ed.BlockLanczosArray import (
    Reort,
    block_normalize,
    eigh_block_tridiagonal,
    enable_reort_profile,
    get_reort_profile,
    reset_reort_profile,
)
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.groundstate import find_ground_state_basis
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
    enable_manybody_profile,
    get_manybody_profile,
    inner_multi,
    reset_manybody_profile,
    support_stats,
)
from impurityModel.ed.selfenergy import _MAX_ROTATION_FILL, _ROTATION_TRIM_TOL, _per_group_occupation, _per_group_scalar
from impurityModel.ed.symmetries import (
    classify_bath_occupation,
    extract_tensors,
    group_orbitals_by_blocks,
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)
from impurityModel.ed.trlm import thick_restart_block_lanczos

pytestmark = pytest.mark.benchmark

NBATHS = int(os.environ.get("BLBENCH_NBATHS", "10"))
MIXED_VALENCE = int(os.environ.get("BLBENCH_MV", "1"))
TRUNC = int(os.environ.get("BLBENCH_TRUNC", "30000"))
MAXITER = int(os.environ.get("BLBENCH_MAXITER", "40"))
BLOCK = int(os.environ.get("BLBENCH_BLOCK", "2"))
REPS = int(os.environ.get("BLBENCH_REPS", "3"))
# Force the restarted-Lanczos path in the CIPSI expansion regardless of basis size
# (production hits it once the basis outgrows dense_cutoff; the bench always should).
DENSE_CUTOFF = 50
SLATER_WEIGHT_MIN = 1e-12

MB = float(2**20)


def _comm():
    return MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None


class RssSampler:
    """Sample this process's resident set size from /proc/self/statm in a thread.

    ``peak_delta_bytes`` is the peak RSS observed during the sampled window minus the
    RSS at ``start()`` — the incremental footprint of the benchmarked region (heap
    already freed back to the allocator but not the OS is invisible; treat results as
    a lower bound on transients, cross-checked by the explicit byte accounting).
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
            rss = self._read_rss()
            self.peak_bytes = max(self.peak_bytes, rss)
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
        rss = self._read_rss()
        self.peak_bytes = max(self.peak_bytes, rss)

    @property
    def peak_delta_bytes(self):
        return self.peak_bytes - self.baseline_bytes


@pytest.fixture(scope="module")
def nio_workload():
    """NiO d-shell Hamiltonian (solver basis) + converged CIPSI ground-state basis.

    Mirrors the production plumbing: ``_nio_workload.build_selfenergy_inputs`` →
    ``calc_selfenergy``'s symmetry-rotation / bath-classification / orbital-grouping
    preamble → ``find_ground_state_basis`` + one ``CIPSISolver.expand`` (both running
    the PARTIAL restarted Lanczos, exactly like ``calc_gs``). The returned basis is
    then *frozen*: the benchmarks re-run the Lanczos loops on it under controlled
    settings.
    """
    from impurityModel.test._nio_workload import build_selfenergy_inputs

    comm = _comm()
    inputs = build_selfenergy_inputs(nBaths=NBATHS, truncation_threshold=TRUNC, verbose=False)

    u = atomic_physics.getUop_from_rspt_u4(inputs["u4"])
    h_input = ManyBodyOperator(inputs["h0"]) + ManyBodyOperator(u)
    impurity_indices = sorted(o for orbs in inputs["impurity_orbitals"].values() for o in orbs)
    h_input_matrix = extract_tensors(h_input, two_body=False)[0]
    n_orb = h_input_matrix.shape[0]

    rotation_full, _u_imp = impurity_symmetry_rotation(h_input, impurity_indices, n_orb=n_orb, h0_matrix=h_input_matrix)
    h_rotated = rotate_hamiltonian(h_input, rotation_full, tol=_ROTATION_TRIM_TOL)
    n_terms_input = sum(1 for v in h_input.values() if abs(v) > _ROTATION_TRIM_TOL)
    if len(h_rotated) / max(n_terms_input, 1) <= _MAX_ROTATION_FILL:
        h = h_rotated
        h_matrix = extract_tensors(h, n_orb=n_orb, two_body=False)[0]
    else:
        h = h_input
        h_matrix = h_input_matrix

    valence_flat, conduction_flat = classify_bath_occupation(h, impurity_indices, n_orb=n_orb, h0_matrix=h_matrix)
    block_structure = impurity_block_structure(h, impurity_indices, h0_matrix=h_matrix)
    impurity_orbitals, bath_states = group_orbitals_by_blocks(
        h, impurity_indices, valence_flat, conduction_flat, block_structure, n_orb=n_orb, h0_matrix=h_matrix
    )
    nominal_occ = _per_group_occupation(inputs["nominal_occ"], impurity_orbitals, h_matrix)
    # _per_group_scalar maps a dict keyed by the derived group indices through unchanged;
    # anything else collapses to the default — so key the window by group explicitly.
    mixed_valence = _per_group_scalar(dict.fromkeys(impurity_orbitals, MIXED_VALENCE), impurity_orbitals, default=0)

    tau = inputs["tau"]
    basis = find_ground_state_basis(
        h,
        impurity_orbitals,
        bath_states,
        nominal_occ,
        mixed_valence=mixed_valence,
        tau=tau / 100,  # calc_gs runs the occupation search at tau/100
        chain_restrict=False,
        dense_cutoff=DENSE_CUTOFF,
        spin_flip_dj=False,
        comm=comm,
        truncation_threshold=TRUNC,
        verbose=False,
        slaterWeightMin=np.sqrt(SLATER_WEIGHT_MIN),
        cipsi_solver_method="trlm",
    )
    basis.tau = tau
    solver = CIPSISolver(basis)
    solver.expand(
        h,
        dense_cutoff=DENSE_CUTOFF,
        de2_min=1e-6,
        slaterWeightMin=SLATER_WEIGHT_MIN,
        solver="trlm",
        reort=Reort.PARTIAL,
    )
    if basis.restrictions is not None:
        h.set_restrictions(basis.restrictions)
    return {"h": h, "basis": basis, "comm": comm, "shared": {}}


def _rank0(comm):
    return comm is None or comm.rank == 0


def _report(comm, title, rows):
    """Print a rank-0 aligned key/value report."""
    if not _rank0(comm):
        return
    width = max(len(k) for k, _ in rows)
    print(f"\n=== {title} ===")
    for key, val in rows:
        print(f"  {key:<{width}}  {val}")


def _random_block(basis, comm, width):
    """Rank-seeded random start block, mirroring cipsi_solver.get_eigenvectors."""
    rank = comm.rank if comm is not None else 0
    random.seed(42 + rank)
    local_states = list(basis.local_basis)
    psi0 = [
        ManyBodyState({state: random.random() + 1j * random.random() for state in local_states}) for _ in range(width)
    ]
    psi0, _ = block_normalize(psi0, basis.is_distributed, basis.comm, SLATER_WEIGHT_MIN)
    return psi0


def test_partial_trlm_array_bench(nio_workload):
    """Production path: TRLM (PARTIAL) on the column-sliced CSR matrix (array kernel)."""
    h, basis, comm = nio_workload["h"], nio_workload["basis"], nio_workload["comm"]

    n_global = len(basis)
    num_wanted = min(10 + 10, n_global)
    max_subspace = min(max(2 * num_wanted, num_wanted + 10), n_global)
    psi0 = _random_block(basis, comm, 1)
    max_subspace_blocks = min(2 * int(np.ceil(max_subspace / len(psi0))) + 20, max(2, n_global // len(psi0) - 1))
    num_wanted = min(num_wanted, (max_subspace_blocks - 1) * len(psi0))

    sampler = RssSampler()
    sampler.start()
    H_mat = build_sparse_matrix(basis, h)
    if basis.is_distributed:
        H_mat = H_mat[:, basis.local_indices]
    h_mat_bytes = H_mat.data.nbytes + H_mat.indices.nbytes + H_mat.indptr.nbytes
    psi0_arr = build_distributed_vector(basis, psi0).T

    times = []
    e_ref = None
    for _ in range(REPS):
        t0 = time.perf_counter()
        e_ref, _psi_arr = thick_restart_block_lanczos(
            psi0=psi0_arr,
            h_op=H_mat,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=1e-8,
            max_restarts=100,
            verbose=False,
            slaterWeightMin=SLATER_WEIGHT_MIN,
            reort=Reort.PARTIAL,
        )
        times.append(time.perf_counter() - t0)
    sampler.stop()

    assert len(e_ref) > 0 and np.all(np.isfinite(e_ref))
    nio_workload["shared"]["e0_trlm"] = float(np.min(e_ref))

    local_n = len(basis.local_basis)
    peaks = [sampler.peak_delta_bytes] if comm is None else comm.gather(sampler.peak_delta_bytes, root=0)
    locals_n = [local_n] if comm is None else comm.gather(local_n, root=0)
    _report(
        comm,
        f"TRLM/array PARTIAL (NiO {NBATHS} bath, basis {n_global}, "
        f"{'serial' if comm is None else f'{comm.size} ranks'})",
        [
            ("E0", f"{np.min(e_ref):.6f}"),
            ("median wall time", f"{np.median(times):.3f} s over {REPS} reps"),
            ("H_mat CSR bytes (rank0)", f"{h_mat_bytes / MB:.1f} MiB"),
            ("local basis sizes", str(locals_n)),
            ("peak RSS delta / rank", str([f"{p / MB:.1f} MiB" for p in peaks])),
        ],
    )


def test_partial_sparse_kernel_bench(nio_workload):
    """Sparse ManyBodyState kernel: block_lanczos_cy PARTIAL, fixed iteration budget."""
    h, basis, comm = nio_workload["h"], nio_workload["basis"], nio_workload["comm"]

    enable_block_lanczos_profile(True)
    enable_reort_profile(True)
    enable_manybody_profile(True)
    reset_block_lanczos_profile()
    reset_reort_profile()
    reset_manybody_profile()

    psi0 = _random_block(basis, comm, BLOCK)

    sampler = RssSampler()
    sampler.start()
    t0 = time.perf_counter()
    alphas, betas, Q_basis, W, block_widths = block_lanczos_cy(
        psi0=psi0,
        h_op=h,
        basis=basis,
        converged_fn=lambda *a, **k: False,
        reort="partial",
        max_iter=MAXITER,
        slaterWeightMin=SLATER_WEIGHT_MIN,
        comm=comm,
        return_widths=True,
    )
    wall = time.perf_counter() - t0
    sampler.stop()

    enable_block_lanczos_profile(False)
    enable_reort_profile(False)
    enable_manybody_profile(False)

    n_it = len(alphas)

    # --- correctness: semi-orthogonality of the Krylov basis + E0 vs the array kernel ---
    ov = inner_multi(Q_basis, Q_basis)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, ov, op=MPI.SUM)
    orth_err = np.linalg.norm(ov - np.eye(len(Q_basis)))
    assert orth_err < 1e-6, f"PARTIAL lost semi-orthogonality: |Q^H Q - I| = {orth_err:.3e}"

    eigvals, _ = eigh_block_tridiagonal(alphas, betas, block_widths=block_widths, eigvals_only=True)
    e0_sparse = float(np.min(eigvals))
    e0_trlm = nio_workload["shared"].get("e0_trlm")
    if e0_trlm is not None:
        assert abs(e0_sparse - e0_trlm) < 1e-4, f"kernel E0 mismatch: sparse {e0_sparse} vs array {e0_trlm}"

    # --- memory breakdown ---
    # Q_basis is the columnar store (Phase 3); report its dense footprint, plus what the
    # same basis would cost as a list of flat_map states (the pre-Phase-3 retention).
    q_bytes = Q_basis.memory_bytes()
    q_states = list(Q_basis)
    legacy_bytes = sum(st.memory_bytes() for st in q_states)
    union_size, total_nnz = support_stats(q_states)
    fill = total_nnz / (union_size * len(Q_basis)) if union_size else 0.0
    w_bytes = W.nbytes if W is not None else 0
    del q_states

    prof = get_block_lanczos_profile()
    reort_prof = get_reort_profile()
    mbu_prof = get_manybody_profile()

    def per_it(key):
        n = prof.get(key + "#n", 0.0)
        return f"{1e3 * prof.get(key, 0.0) / n:8.2f} ms/call x {int(n):3d}" if n else "       - "

    local_n = len(basis.local_basis)
    gathered = (
        [(local_n, q_bytes, sampler.peak_delta_bytes)]
        if comm is None
        else comm.gather((local_n, q_bytes, sampler.peak_delta_bytes), root=0)
    )
    _report(
        comm,
        f"sparse-kernel PARTIAL (NiO {NBATHS} bath, basis {len(basis)}, p={BLOCK}, "
        f"{n_it} its, {'serial' if comm is None else f'{comm.size} ranks'})",
        [
            ("wall time", f"{wall:.3f} s  ({1e3 * wall / max(n_it, 1):.1f} ms/iter)"),
            ("E0 (T eigvals)", f"{e0_sparse:.6f}" + (f"  (TRLM: {e0_trlm:.6f})" if e0_trlm is not None else "")),
            ("|Q^H Q - I|", f"{orth_err:.3e}"),
            ("matvec_apply", per_it("matvec_apply")),
            ("matvec_redistribute", per_it("matvec_redistribute")),
            ("recurrence (LA)", per_it("recurrence")),
            ("tsqr", per_it("tsqr")),
            ("w_estimate", per_it("w_estimate")),
            ("reort", per_it("reort")),
            ("monitor", per_it("monitor")),
            ("reort acted/total", f"{int(prof.get('reort_acted#n', 0))}/{int(prof.get('reort_total#n', 0))}"),
            ("apply_reort stats", str(reort_prof)),
            ("cgs2_dense transients", str(mbu_prof)),
            ("Q_basis store (local)", f"{q_bytes / MB:.1f} MiB, {len(Q_basis)} columns"),
            ("union support / fill", f"{union_size} determinants, fill {fill:.2f}"),
            ("legacy list equivalent", f"{legacy_bytes / MB:.1f} MiB ({legacy_bytes / max(q_bytes, 1):.1f}x larger)"),
            ("W buffer", f"{w_bytes / MB:.2f} MiB"),
            (
                "per-rank (local_n, Q MiB, peak RSS delta MiB)",
                str([(n, f"{q / MB:.1f}", f"{p / MB:.1f}") for n, q, p in gathered]),
            ),
        ],
    )


def test_apply_block_width_scaling(nio_workload):
    """Phase 2.0 of the block-state matvec plan: apply_multi cost vs block width p on
    the real NiO Hamiltonian's term mix (restrictions set, basis-confined states).
    Today's independent per-state applies scale ~linearly in p; the ManyBodyState
    target is near-flat. Baseline for the block-container A/B."""
    h, basis, comm = nio_workload["h"], nio_workload["basis"], nio_workload["comm"]

    rows = []
    for p in (1, 2, 4, 8):
        psis = _random_block(basis, comm, p)
        blk = ManyBodyState.from_states(psis)
        times, btimes = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            out = h.apply_multi(psis, SLATER_WEIGHT_MIN)
            times.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            h.apply_block(blk, SLATER_WEIGHT_MIN)
            btimes.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        btimes.sort()
        rows.append((p, times[len(times) // 2], btimes[len(btimes) // 2], sum(len(st) for st in out)))
    t1 = rows[0][1]
    _report(
        comm,
        f"apply_multi p-scaling (NiO {NBATHS} bath, basis {len(basis)}, "
        f"{'serial' if comm is None else f'{comm.size} ranks'})",
        [
            (
                f"p={p}",
                f"multi {med:8.2f} ms ({med / t1:5.2f}x p=1)   block {bmed:8.2f} ms   "
                f"speedup {med / bmed:5.2f}x   nnz_out {nnz}",
            )
            for p, med, bmed, nnz in rows
        ],
    )
