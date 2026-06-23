# Review of `symmetry_implementation_plan.md` (archived)

> **This review has been incorporated.** The updated plan lives at
> `doc/plans/symmetry_implementation_plan.md`. This file is retained for reference.

---

# Review of `symmetry_implementation_plan.md`

This is a code-grounded review of the automated symmetry-discovery plan. It marks
each task **KEEP**, **FIX**, **REMOVE**, or **ADD**, with the architectural reason.
Two cross-cutting facts drive most of the comments:

1. **There are two completely different distribution models in this code.**
   * *Sparse `ManyBodyState` path* (`block_lanczos_cy`, CIPSI, density matrices):
     a Slater determinant is **owned by rank `det.get_hash() % comm.size`**, using
     the **C++ splitmix64 hash** in `SlaterDeterminant.h` (`std::hash<SlaterDeterminant>`),
     *not* Python `hash()`. This is essential: Python's `hash()` is per-process
     salted (`PYTHONHASHSEED`) and would place the same SD on different ranks on
     different processes, breaking `redistribute_psis`, `_index_sequence`,
     `pack_psis_cy`/`pack_determinants_cy`. Global indices are rank-major contiguous
     blocks; within a rank `local_basis` is sorted.
   * *Dense/CSR array path* (`block_lanczos_array_cy`, used by `greens_function.py`
     dense branch and CIPSI's `irlm`/`trlm`): a **contiguous row-block partition**
     (`offsets`/`counts` via `Allgather`), with a global `wp_global` `Allreduce`d and
     then sliced. This is a different ownership rule *and* a heavier memory model
     (the full `global_N × p` block is materialised on every rank).

   Any plan that says "use `BlockLanczosArray` for the distributed multi-tOp loop"
   (Phase 5/6.1) is silently switching between these two models. That must be made
   explicit or it will deadlock / give wrong reductions.

2. **The single-particle basis is fixed and occupation-encoded.** Restrictions
   (`ManyBodyOperator::build_restriction_mask`, `Restrictions =
   vector<pair<vector<size_t>, pair<size_t,size_t>>>`) can only express
   "orbital-subset *S* holds between *min* and *max* electrons." They **cannot**
   express a general weighted charge `sum_i w_i n_i = q`. This is the central
   limiter for Phase 3.

---

## Phase 1 — Symmetry Discovery Engine

* **1.1 Tensor extraction — KEEP.** `ManyBodyOperator` stores
  `vector<pair<OPS, SCALAR>>` with `OPS = vector<int64_t>` of (orb, c/a) tokens; a
  dense `h_ij`, `V_ijkl` extraction with a round-trip test is sound. Add: assert the
  operator is purely 1- and 2-body before extraction (reject 3-body terms instead
  of silently dropping them).
* **1.2 Null space via SVD — KEEP, reword the verification.** The commutant
  `[H, O] = 0` over **one-body** `O` yields the one-body symmetry algebra. `N̂`
  and `Ŝ_z` are one-body and *will* appear. **`Ŝ²` is two-body and will not** — the
  checkpoint that mentions "SU(2) … Total spin" is misleading; restrict the claim to
  `N̂`, `Ŝ_z`, and any one-body orbital symmetries (point-group/CF). Test against a
  Hubbard dimer where the answer is known by hand.
* **1.3 Cartan subalgebra & joint diagonalization — KEEP.** Extract a maximal
  mutually-commuting set and jointly diagonalize → single-particle `U`. Add a
  checkpoint that the resulting generators are **integer-valued occupation
  operators** in the new basis when possible (see Phase 3 limiter); flag any
  generator with non-`{0,1}` orbital weights, because those cannot become
  restrictions.

## Phase 2 — High-Performance Basis Rotation

* **2.1 Python reference `H' = U†HU` — KEEP.**
* **2.2 Cython `O(N⁵)` 2-body rotation — FIX the premise.** The file claims a
  `nogil` `zgemm` kernel, but the existing array kernel (`matmul_nogil` in
  `BlockLanczosArray.pyx`) is a **hand-rolled triple loop with the
  `scipy.linalg.cython_blas` import commented out** (line ~257). Before adding a
  *new* `O(N⁵)` kernel, either reuse a real BLAS path or you will benchmark against
  the wrong baseline. The 2-body rotation is a one-time setup cost, not a hot loop —
  **`numpy.einsum`/`tensordot` (which call BLAS) is almost certainly sufficient**;
  a bespoke Cython kernel here is premature optimization. Recommend: implement with
  `np.einsum` first, only port to Cython if profiling proves it dominates.
* **ADD (critical integration point the plan omits):** after rotation the *entire*
  pipeline runs in the rotated basis — `Basis._get_initial_basis`, `op_parser`,
  restrictions, the C++ hash distribution, density matrices, observable operators.
  The rotation only "block-diagonalizes" if the discovered symmetries are diagonal
  occupation numbers in the new basis. Add an explicit checkpoint: re-express `H` as
  a `ManyBodyOperator` in the rotated basis and verify the **C++ hash distribution
  still load-balances** (rotation can change SD bit patterns and thus the
  `hash % size` balance).

## Phase 3 — Sectorized CIPSI

* **3.1 Symmetry → restriction mapping — FIX / scope down.** As written
  ("`O = Σ w_i n̂_i` → `Basis.restrictions`") this is **only possible when each
  generator's weights are 0/1 on an orbital subset** (e.g. particle number on a
  subspace, `n_↑`/`n_↓` counts). For `S_z = Σ ±½ n_i` you would need a *signed*
  weighted-sum constraint, which `build_restriction_mask` does not support. Two
  honest options — pick one and write it down:
  * **(a) Subset-only:** restrict discovery output to symmetries expressible as
    subset occupation bounds; map those. Simple, covers `N`, valence/conduction
    counts, point-group-disconnected orbital sets.
  * **(b) Extend the mask:** add a weighted-sum restriction kind to the C++
    `Restrictions` machinery and `state_is_within_restrictions`. Larger, but unlocks
    `S_z`/general abelian charges. (Separate implementation plan recommended.)
* **3.2 Sectorized H generation — KEEP, with a caveat.** The existing CIPSI already
  refuses out-of-restriction SDs through `op.set_restrictions(...)` +
  `build_restriction_mask`. The "strictly block-diagonal `H_mat`" checkpoint is good
  but note `build_sparse_matrix` already only stores connections within the basis;
  it is the *basis generation* (`expand`, `determine_new_Dj`) that must respect the
  sector, which it does once restrictions are set. Reuse, don't rebuild.

## Phase 4 — Auto-Routing Green's Functions

* **4.1 Selection rules `Δq_ij = w_i − w_j` — KEEP.** This is the right
  generalization of the connectivity graph already in `build_initial_restrictions`
  (`scipy.sparse.csgraph.shortest_path` over the hopping graph).
* **4.2 Advanced block equivalence — MOSTLY ALREADY EXISTS; reframe as "wire up".**
  `block_structure.py` *already* computes `identical_blocks`, `transposed_blocks`,
  `particle_hole_blocks`, `particle_hole_transposed_blocks`, `inequivalent_blocks`
  and reconstructs via `build_matrix`/`build_greens_function`. The task should be
  "drive the existing equivalence detection from the discovered symmetry quantum
  numbers and use `inequivalent_blocks` in the spectra loop," not "expand detection."
  **BUG to fix first (separate task):** `build_greens_function` lines ~761/767 call
  `np.transpose(tuple(...))` — transposing a *tuple* literal, not the block `m`;
  the transposed/PH-transposed Green's-function reconstruction is broken. Cover it
  with `test_advanced_block_equivalence` (good that the plan adds this test).
* **4.3 Confining the Krylov space — FIX the mechanism.** `block_lanczos_cy` does
  **not** expand the basis; it runs on a fixed `basis` and applies
  `h_op.apply_multi` + `redistribute_psis`. There is no "dynamic expansion" to feed
  quantum numbers into. The sector confinement must be applied **before** the run by
  setting the excitation operator's restrictions (the existing
  `build_excited_restrictions` path) so that `apply_multi` never produces
  out-of-sector SDs. Reword 4.3 to: "set `hOp`/`tOp` restrictions to the target
  sector `q_Ψ − w_j` before `block_Green_sparse`, and verify the Krylov basis stops
  growing into other sectors." The "refuses off-sector states" assertion is then
  testable on the operator's restriction mask.

## Phase 5 — MPI Load Balancing

* **5.1 "Strip out `MPI.Comm.Split()`" — REMOVE as stated; replace with a heuristic.**
  The split is load-balancing **task parallelism**: `split_basis_and_redistribute_psi`
  assigns different tOp-blocks (`spectra.py:779`), energies (`:950`), frequencies
  (`:972`), and GF blocks (`greens_function.py:130,354`) to disjoint sub-communicators.
  Removing it forces every block through a *global* collective Lanczos. That is
  better **only** when there are few, large blocks; it is worse for the common case
  of many small independent blocks/frequencies (turns N cheap independent solves into
  N global-synchronized solves). Recommended task instead:
  * Make splitting a **heuristic policy** (`split when n_blocks · cost_per_block
    spread > threshold`, else unified), not a removal.
  * Keep the existing intercomm-freeing discipline (already carefully done in
    `get_Greens_function` and `split_basis_and_redistribute_psi`) — that code is
    correct and subtle; don't lose it.
* **5.1 "use vectorized `BlockLanczosArray`" — FIX.** `block_lanczos_array_cy` uses
  the **row-block** distribution and the **naive `matmul_nogil`**, and it
  `Allreduce`s a full `global_N × p` buffer every iteration. It is not a drop-in for
  the hash-distributed sparse path and is not currently BLAS-accelerated. Do not
  promise it as the unified multi-tOp engine until (a) its kernel is real BLAS and
  (b) its memory model is acceptable — see `blocklanczos_blas_acceleration.md`.
* **5.2 Memory profiling — KEEP.** Good. Note the actual Rank-0 OOM risk today is
  `build_vector(..., root=0)` and `build_dense_matrix` (`Allreduce` of a dense
  `size × size`), and the array path's `wp_global`. Target those specifically.

## Phase 6 — Ground-State Observables & Sector Labeling

* **6.1 ⟨L·S⟩ from ρ — KEEP.** `build_density_matrices` already gives the 1-RDM
  efficiently (it exploits `ρ_ij = ⟨c_i ψ | c_j ψ⟩`, O(n) applies). L·S is a 1-body
  contraction of ρ. Straightforward and correct.
* **6.2 S², L², J² as `ManyBodyOperator` — KEEP.** Build once, evaluate
  `⟨ψ|Ô|ψ⟩` via `apply` + `inner_multi` (+ `Allreduce` when distributed). Correct and
  cheap relative to the eigensolve. **This phase does not depend on Phases 1–5** and
  can ship first as a self-contained, high-value deliverable.
* **6.3 Local impurity observables — KEEP.** Restrict the operators to impurity
  orbitals; reuse `impurity_orbitals`.
* **6.4 Kondo ⟨S_imp·S_bath⟩ — KEEP.**

---

## Suggested resequencing (lowest-risk, highest-value first)

1. **Phase 6** (observables) — independent, immediately useful, no architectural risk.
2. **Block-Lanczos hot-loop hardening** (separate plan) — correctness/stability of
   the core *before* you lean on it harder for sectorization.
3. **Phase 4.2 bug fix** + wiring existing `block_structure` equivalences.
4. **Phase 1 + Phase 3(a) (subset-only)** — discovery + the part of sectorization
   that the restriction machinery already supports.
5. **Phase 2** (basis rotation, einsum-first), then **Phase 3(b)** (weighted
   restrictions) and **Phase 4.1/4.3** if needed.
6. **Phase 5** last, as a *heuristic split policy*, not a rewrite.

See companion plans:
* `blocklanczos_hotloop_hardening.md` — MPI-safety, deflation, reortho correctness.
* `blocklanczos_blas_acceleration.md` — real BLAS in the array kernel.
