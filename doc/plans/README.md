# Implementation Plans

Design and implementation plans for in-progress and proposed work on the
`impurityModel` ED/DMFT package. Each plan uses checkboxes so progress is verifiable
at a glance.

> **Status (2026-06-24):** the **symmetry program is complete** — the main
> `symmetry_implementation_plan.md` (Phases 1–7 + the cross-phase acceptance gate) and
> its `nonabelian_symmetry_casimir.md` companion are all implemented and tested (serial
> + MPI), with `use_prescan` defaulted on and `auto_block_structure` available as a
> validated opt-in. See each plan's top-of-file progress block for details. The
> remaining open work is in the Block Lanczos plans (#1–#3) and a few explicitly-scoped
> perf/won't-fix follow-ups noted inline.

## Reading order

The plans have dependencies. Read and execute in this order:

1. **[blocklanczos_hotloop_hardening.md](blocklanczos_hotloop_hardening.md)** —
   Correctness and numerical stability of the Block Lanczos hot loops (sparse
   `BlockLanczos.pyx` and array `BlockLanczosArray.pyx`): no silent failures, no MPI
   deadlocks. *Prerequisite for leaning on the core harder in the symmetry work.*
   Items 1–3 (SELECTIVE reortho deadlock, blanket `try/except`, array W-index bug)
   and §3b (empty-local-partition int32-CSR deadlock + regression test) are **done**.
   Its open Items 4–6 and §8 guardrails are now carried by the reort-reliability plan
   (#2) — see there for the live worklist.

2. **[blocklanczos_reort_reliability.md](blocklanczos_reort_reliability.md)** — Make
   TRLM/IRLM reliable across **every** reorthogonalization mode: pick any `Reort` mode
   (NONE/PARTIAL/FULL/PERIODIC/SELECTIVE) and the run is stable, accurate,
   deadlock-free, and ghost-band-free, serial + MPI, on both the array and
   ManyBodyState paths — with **PARTIAL (PRO) the safe default**. Incremental
   unification of the two divergent kernels; **oracle-first** sequencing (Phase 0 ships
   a cross-product test matrix + ghost-band regression as its own PR, capturing the
   failing cells as the documented worklist). Subsumes hotloop-hardening Items 4–6/§8;
   its Phase-0 oracle is the regression gate the BLAS work (#3) depends on.

3. **[blocklanczos_blas_acceleration.md](blocklanczos_blas_acceleration.md)** — Real
   BLAS in the array kernel (replace the hand-rolled `matmul_nogil` triple loop with
   `cython_blas.zgemm`, pre-allocate workspaces, shrink the distributed memory
   footprint). *Prerequisite for using the array path as a multi-tOp engine (symmetry
   Phase 7); gated on the reort-reliability Phase-0 oracle (#2).*

4. **[symmetry_implementation_plan.md](symmetry_implementation_plan.md)** — ✅ **DONE.**
   Automated symmetry discovery & exploitation. Seven phases, all implemented (new
   `symmetries.py` + wiring into `finite.py`/`groundstate.py`/`cipsi_solver.py`/
   `greens_function.py`/`manybody_basis.py` and a C++ weighted-restriction extension):
   observables (`<L.S>`/`<S²>`/`<L²>`/`<J²>`/`<S_imp·S_bath>` reporting) → discovery
   (`[H,O]=0` null space, Cartan, joint diagonalisation) → sectorized CIPSI (subset
   restrictions + rough-scan prescan, now default) → GF auto-routing (selection rules,
   sector confinement, `auto_block_structure`) → basis rotation (`U†HU`, DMFT cache) →
   extended restrictions (weighted-sum `S_z`/`L_z`, C++) → MPI adaptive split policy. The
   cross-phase acceptance gate (auto-discovery matches/refines every hand-coded
   `block_structure`) passes. Also fixed two latent bugs found along the way
   (`get_inequivalent_blocks` PH-reduction, dense-GF `block_green_impl` signature).

5. **[nonabelian_symmetry_casimir.md](nonabelian_symmetry_casimir.md)** — ✅ **DONE.**
   Companion to the symmetry plan: detect non-abelian symmetry (`is_abelian`,
   `structure_constants` → `f = iε_{abc}` for su(2)) and reconstruct Casimirs
   (`apply_reconstructed_casimir`, `Ĉ = Σ_a Ô_a²`) for exact multiplet labeling, with
   `[Ĉ,H]=0` as the correctness gate and the reconstructed `Ŝ²` proven identical to the
   hand-built one. Builds on symmetry-plan Phase 2; shares the Phase-1.2 observables.

6. **[manybodyoperator_apply_performance.md](manybodyoperator_apply_performance.md)** —
   Throughput of `ManyBodyOperator::apply` (`src/cython/ManyBodyOperator.cpp`), the
   second-quantized matvec under every Block Lanczos sweep / CIPSI / GF build. Six
   phases: oracle-first (Phase 0 ships a representative benchmark + golden-output
   regression gate), then allocation/accumulator micro-opts → term classification with
   a diagonal fast path and masked between-sign kernels → build-time normal ordering;
   with MPI bucketed-return and threading as parallel tracks. **Independent** of plans
   #1–#3 (different code: the C++ operator, not the Cython Lanczos drivers). `n_orbs` is
   runtime-determined and unbounded; the hot path must not assume a fixed chunk count.

7. **[bicgstab_per_frequency_gf.md](bicgstab_per_frequency_gf.md)** — Compute the interacting
   Green's function with a **per-frequency block BiCGSTAB solve** (real and Matsubara axes)
   instead of a block-Lanczos recurrence plus continued fraction. This is the memory-flat
   alternative `blocklanczos_reort_memory.md` (#1's companion) names and asks to benchmark
   before more effort goes into paging the retained Krylov basis `Q` to disk: a solve carries
   seven blocks and has no orthogonality to lose. Phases 0–2 are **done** — `cg.py` is
   consolidated into `src/cython/BiCGSTAB.pyx`, and after removing a quadratic
   `basis.local_basis` list scan the solver is **matvec-bound (93%)**, so any further speedup
   of this path belongs to plan #6, not here. Phase 3a (**done**) prices the benchmark that plan
   #1 asked for: warm starts (quadratic extrapolation in `z`) cut a solve from 12 to 2.9 matvecs
   per mesh point, and per-frequency BiCGSTAB still costs **10x the wall time** of one Lanczos
   recurrence on the production 375-point Matsubara mesh, for **7x less memory**, at identical
   accuracy. So it is a memory escape hatch (a better one than paging `Q`), not a faster GF.
   Phase 3b (the driver, as an opt-in `gf_method`) is the open work — but price *shifted*
   BiCGSTAB first, which would keep the flat memory and get the whole mesh from one Krylov space.

## Reference

- **[apply_perf_deferred_designs.md](apply_perf_deferred_designs.md)** — Design sketches
  (derivations + code) for the two items deferred from the apply-performance plan: the
  off-diagonal **two-body masked sign** (single masked popcount + build-time constant,
  probe-validated) and the **NBX sparse count exchange** (dynamic sparse data exchange,
  adaptive vs the dense collective). Not yet scheduled work.


- **[symmetry_plan_review.md](symmetry_plan_review.md)** — Archived code-grounded
  review (KEEP/FIX/REMOVE/ADD) of the original symmetry plan. Its conclusions are now
  incorporated into `symmetry_implementation_plan.md`; retained for rationale.

## Cross-cutting architectural facts (apply to all plans)

- **Two incompatible MPI distribution models.** Sparse hash-distributed path
  (`block_lanczos_cy`, CIPSI, density matrices): a Slater determinant is owned by rank
  `det.get_hash() % comm.size` via the **C++ splitmix64 hash** — *not* Python `hash()`
  (per-process salted, would corrupt routing). Dense row-block path
  (`block_lanczos_array_cy`): contiguous row-block partition with a full `global_N × p`
  buffer `Allreduce`d on every rank. They are not drop-in replacements.
- **Any rank can own an empty local partition** (`local_N == 0`) when ranks outnumber
  the distributed states. Distributed kernels must stay dtype/shape robust to this and
  every rank must reach the same collectives — an empty `(global_N, 0)` CSR slice
  silently picks int32 indices (vs int64 elsewhere), the class of bug fixed in hot-loop
  hardening §3b. Covered by `test_block_lanczos_array_empty_rank.py`.
- **Restriction masks** now support both subset occupation bounds (`set_restrictions`,
  "orbital-subset S holds between min and max electrons") **and** weighted charges
  `Σ wᵢ nᵢ = q` (`set_weighted_restrictions`, e.g. `S_z`/`L_z`) — the latter added by
  symmetry-plan Phase 6 (C++ `build_weighted_restriction_mask`). Both are checked
  together in `state_is_within_restrictions` and threaded through `Basis`/CIPSI/GF.
- **Unitary symmetries only.** Anti-unitary symmetries (time reversal / Kramers) are
  not detectable by the `[H,O]=0` null-space method.
- **Reorthogonalization mode is a first-class knob.** Any `Reort` mode must work as
  well as its algorithm allows; **PARTIAL (PRO) is the default** and must not introduce
  ghost bands. The two kernels share `estimate_orthonormality` / `_build_full_T` today
  and (post-unification, reort-reliability Phase 4) one reort engine. Don't add
  mode-specific behavior to only one kernel — they must stay behaviorally identical.

## Implementation conventions for weak-model execution (read before editing any plan)

These plans are written to be executable by a small/fast model with minimal judgment.
Every plan in this directory follows these rules; they are stated once here, not
repeated per task.

1. **Decisions are already made.** Each checkbox is a single concrete instruction. If
   you ever encounter a real choice ("do X *or* Y", "pick a value", "if needed") that
   the plan has *not* resolved, **stop and ask the human** — do not improvise. The
   plans have been combed to remove open choices; an unresolved one is a plan bug.
2. **Anchor on code, not line numbers.** Line numbers in these plans are *hints only*
   and drift as edits land. Each task names a **unique code snippet**; locate the real
   site with `grep -n "<snippet>" <file>` (or the Grep tool) immediately before
   editing. Never trust a bare line number.
3. **One checkbox = one small, self-contained change**, followed by a **checkpoint**:
   - After editing any `.pyx`/`.cpp`/`.h` (Cython/C++): rebuild with
     `pip install -e . --no-build-isolation` (the `--no-build-isolation` flag is needed
     so the env's scipy `cython_blas` is visible to `BlockLanczosArray.pyx`; set
     `BOOST_ROOT` or `pip install boost-headers` first if Boost headers are missing),
     then run the test named in that task.
   - After editing any `.py`: run the test named in that task (no rebuild needed).
   - Serial test: `pytest src/impurityModel/test/<file> -q`. MPI test:
     `mpirun -n 3 pytest --with-mpi src/impurityModel/test/<file>`.
   - **A red checkpoint blocks the next checkbox.** Revert your edit and report; do not
     stack changes on top of a failure.
4. **Cython/MPI landmines** (these have already bitten this codebase):
   - **CSR index dtype.** `scipy.tocsr()` picks **int32** for small/empty matrices and
     **int64** otherwise; the kernels bind `cdef long[:]` (int64). Always coerce:
     `np.ascontiguousarray(m.indices, dtype=np.int64)` (and `.indptr`). Mismatch =
     per-rank `ValueError` = MPI deadlock.
   - **Every MPI collective must be reached by *all* ranks.** Never put an
     `Allreduce`/`bcast`/`Allgather`/`Reduce` inside a branch that some ranks skip
     (e.g. `if rank == 0:` or a mode-dependent branch that isn't identical on every
     rank). Decide on rank 0 → `bcast` the decision → act uniformly.
   - **Cython ordering:** `cdef` declarations must precede executable statements;
     `array.shape` is a C `npy_intp*` and is not f-string-formattable (use
     `np.asarray(x).shape`).
   - **Splitmix64, not `hash()`.** SD ownership is `det.get_hash() % comm.size` (C++
     splitmix64); never substitute Python `hash()` (per-process salted).
5. **Don't broaden scope.** Implement exactly the checkboxes in order. If you discover
   adjacent work, note it; don't do it.
