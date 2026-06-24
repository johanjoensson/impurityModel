# HPC Architecture Plan: Automated Symmetry Discovery & Exploitation

## Objective

Eliminate manual symmetry specifications (e.g., `block_structure`) by dynamically
discovering the one-body symmetry algebra of the second-quantized Hamiltonian. Use its
abelian (Cartan) part to strictly block-diagonalize the many-body Hilbert space,
accelerate `CIPSISolver` via sectorized Lanczos, and suppress Hilbert-space explosion
during Green's function computations; use its non-abelian part (companion plan) for
exact multiplet labeling. Unitary symmetries only — anti-unitary (time-reversal/Kramers)
symmetries are out of reach of this method and are documented as a limitation.

---

> **Execution conventions & weak-model suitability:** see "Implementation conventions
> for weak-model execution" in `README.md`. **Phase 1 (observables) is weak-model
> suitable** — it is concrete contractions over existing machinery. **Phases 2–7 are
> research-grade** (new linear-algebra/symmetry code in a new `symmetries.py`); steps
> that need genuine mathematical judgment are tagged **[strong-model]** below. A small
> model should implement Phase 1 and the explicitly-mechanical sub-steps only, and
> stop-and-ask at any **[strong-model]** tag.

## Architectural constraints (read before coding any phase)

Two facts cut across every phase:

**1. Two incompatible MPI distribution models.**

- *Sparse hash-distributed path* (`block_lanczos_cy`, CIPSI, density matrices): a
  Slater determinant is owned by rank `det.get_hash() % comm.size` using the **C++
  splitmix64 hash** in `SlaterDeterminant.h` — NOT Python `hash()`, which is
  per-process salted and would corrupt `redistribute_psis` / `_index_sequence` /
  `pack_psis_cy`.
- *Dense row-block path* (`block_lanczos_array_cy`, used by the GF dense branch and
  CIPSI's irlm/trlm): a contiguous row-block partition with a full `global_N × p`
  buffer `Allreduce`d on every rank.

Any plan that switches between these two paths must do so explicitly; they are not
drop-in replacements for each other.

**2. Restriction masks can only express subset occupation bounds.**

`ManyBodyOperator::build_restriction_mask` uses `Restrictions =
vector<pair<vector<size_t>, pair<size_t,size_t>>>` — "orbital-subset S holds between
min and max electrons." It cannot express a weighted charge `Σ wᵢ nᵢ = q`. This is
the central limiter for Phase 3. `S_z = Σ ±½ nᵢ` requires either subset-only
mapping (Phase 3, option a) or a C++ extension (Phase 6, option b).

---

## Recommended implementation sequence

The phases below are numbered in recommended order, not conceptual order, to put the
highest-value, lowest-risk work first.

1. **Phase 1** — Ground-state observables (independent; ship first)
2. **Phase 2** — Symmetry discovery engine
3. **Phase 3** — Sectorized CIPSI (subset-only restrictions)
4. **Phase 4** — GF auto-routing (wire up existing `block_structure` equivalences)
5. **Phase 5** — Basis rotation (`einsum`-first)
6. **Phase 6** — Extended restriction machinery (weighted-sum, unlocks `S_z`)
7. **Phase 7** — MPI adaptive split policy

See companion plans for work that cuts across phases:
- `blocklanczos_hotloop_hardening.md` — MPI safety, deflation, reortho correctness
  (prerequisite for Phases 3–4)
- `blocklanczos_blas_acceleration.md` — real BLAS in the array kernel (prerequisite
  for Phase 7)
- `nonabelian_symmetry_casimir.md` — detecting non-abelian symmetry (the discovered
  one-body algebra contains `S_x`, `S_y`, not just `S_z`) and reconstructing Casimir
  operators (`Ŝ²`, `L̂²`, `Ĵ²`) for multiplet labeling. Builds on Phase 2's null
  space; shares the observable construction of Phase 1.2.

---

## Cross-phase acceptance gate (the single most important test)

Before this machinery is switched on by default, it must **reproduce every existing
hand-coded `block_structure`**. The manual structures in `edchain.py`,
`get_spectra.py`, and `selfenergy.py` are the ground truth. Run automated discovery on
each model that currently hand-codes a block structure and assert the auto-discovered
structure **matches or strictly refines** the manual one (same blocks, or finer). This
is the regression gate that proves the effort is safe to enable.

- [x] `test_discovery_reproduces_manual_block_structure` — **DONE (2026-06-24).**
      `symmetries.discovered_orbital_blocks(op)` maps the discovered one-body symmetry to
      an orbital partition (the connected components of `h` = the partition induced by the
      diagonal part of the Cartan: two orbitals carry distinct conserved one-body quantum
      numbers ⟺ `h` can't connect them). `blocks_refine_or_match(discovered, reference)`
      is the gate condition (every discovered block ⊆ some reference block — match or
      strictly finer, never coarser). Tests in `test_symmetries.py`:
      `test_acceptance_gate_discovery_refines_pd_block_structure` (SOC p+d model built
      from the real `getSOCop` via `c2i`: discovered blocks **refine** the hand-coded
      `get_spectra.py` `[range(6), range(6,16)]` — SOC conserves `J_z`, splitting each
      shell into `mj` sub-blocks — and are **identical** to `build_block_structure(mat=h)`);
      `test_acceptance_gate_fully_coupled_shells_match_exactly` (intra-shell `ml` coupling
      + SOC → discovered partition **equals** `[p, d]` exactly);
      `test_acceptance_gate_refines_single_impurity_block` (the `selfenergy.py`
      single-block case is refined). The refinement is physically correct: under SOC the
      GF is block-diagonal in `mj`, finer than the manual `p/d` split.

> **Partial validation done (2026-06-24)** — discovery run on **real codebase
> operators** (`finite.getSOCop`, `gethHfieldop`-style CF) in the `c2i` index basis,
> `test_symmetries.py`:
> - `test_discovery_real_dshell_no_soc_has_su2` — a real spin-independent d-shell
>   crystal field: discovery recovers `N, S_x, S_y, S_z, L_z` (full spin SU(2) + axial
>   orbital symmetry).
> - `test_discovery_real_dshell_soc_breaks_su2` — add `getSOCop` (`l·s`): spin SU(2) is
>   correctly **broken** — `N` and `J_z = L_z + S_z` survive, `S_z`/`S_x` no longer
>   commute. (`l·s` is real in the spherical basis; complexity is a cubic-basis effect.)
> - `test_discovery_basis_invariant_cubic` — rotate to the cubic basis (genuinely
>   **complex** `H`): same generator count (basis-invariant), `N` still found, all
>   generators commute. Exercises the engine on the complex Hamiltonians real runs use.
>
> The *formal* gate (map a discovered Cartan to an orbital block decomposition and
> assert it matches/refines the hand-coded `BlockStructure` in `get_spectra.py` and
> `selfenergy.py`) is now **DONE** — see the checked
> `test_discovery_reproduces_manual_block_structure` entry above.

---

## Phase 1: Ground-State Observables & Sector Labeling

> **Progress (2026-06-24): observable-computation core DONE; print/model wiring
> remains.** Implemented and analytically tested in `finite.py` + new
> `src/impurityModel/test/test_symmetry_observables.py` (7 tests, all green):
> - §1.1 `get_LS_from_rho_spherical` — `test_spin_orbit_observable` ✅
> - §1.2 `make_spin_operators`, `make_orbital_angular_momentum_operators`,
>   `apply_casimir`, `expect_casimir`, `casimir_to_quantum_number` (Ŝ²/L̂²/Ĵ² via the
>   ladder identity `J² = J₋J₊ + J_z² + J_z`, evaluated by sequential one-body
>   application — no hand-built 4-fermion product strings) — `test_S2_observable`,
>   `test_L2_J2_observable` ✅
> - §1.4 `apply_spin_correlation`, `expect_spin_correlation` (⟨S_imp·S_bath⟩) —
>   `test_kondo_correlation` ✅
> - §1.5 `manifold_observable_values`, `thermal_observable_value` (degenerate-subspace
>   diagonalisation + Boltzmann average) — `test_degenerate_manifold_observable`,
>   `test_thermal_observable` ✅
>
> **§1.0 print wiring DONE (2026-06-24):** `<L.S>` wired into both print levels;
> impurity `<S^2>`/`S` wired in via gathered full states + `manifold_observable_values`
> (rank-0-only compute, optional `s_values`/`s_thermal` so old output is byte-identical
> when absent). New: `impurity_spin_pairs`; `get_LS_from_rho_spherical` hardened for
> odd impurity blocks. Tests `test_print_expectation_values_columns/_S_column`,
> `test_print_thermal_expectation_values_lines`, `test_print_thermal_S2_line`; verified
> serial + MPI n=2 on `test_groundstate.py` (d-shell triplet GS → `<S^2>=2, S=1`).
>
> **Carry on from here — DEFERRED TO PHASE 5 (decision 2026-06-24):** reporting of
> `<L^2>`/`<J^2>`, the Kondo `<S_imp·S_bath>` (`test_kondo_correlation_reported`), and
> the §1.3 mixed-valence `test_impurity_local_observables` all need the true
> single-particle index→(spatial-orbital, spin) map. The "impurity is down-then-up,
> spin partners at `(k, k+n/2)`" ordering holds **only in the spherical basis** (after
> `rot_to_spherical`); in a cubic computational basis the layout must be read from the
> one-body Hamiltonian (`build_block_structure` / `build_imp_bath_blocks`). Rather than
> derive a separate bath spin-pairing now, **build Phase 5 (basis rotation) first and
> reuse its machinery here** — that also lets us retire `build_imp_bath_blocks` (one
> well-tested path). The Phase-1 observable *primitives* are all done and unit-tested;
> only the reporting wiring waits. **The provisional impurity `<S^2>` print wiring was
> reverted** (2026-06-24) to avoid silent errors in cubic-basis runs — it had used
> `(k, k+n/2)` on computational indices, valid only for identity/spherical
> `rot_to_spherical`. Kept: `<L.S>` reporting (correct, from the spherical-rotated
> `rho`); all `finite.py` primitives; the print functions' optional `s_values`/
> `s_thermal` params (default `None` ⇒ no output change) ready for Phase 5 to fill in
> with the correct index→(orbital, spin) map; `impurity_spin_pairs` (documented
> spherical-only).

**Location:** `src/impurityModel/ed/groundstate.py` and `finite.py`

**Rationale:** Completely independent of Phases 2–7. `build_density_matrices` already
provides the 1-RDM efficiently (`ρᵢⱼ = ⟨cᵢψ|cⱼψ⟩`, O(n) applies); `Ŝ²`, `L̂²`,
`Ĵ²` are built once as `ManyBodyOperator` objects and evaluated via `apply` +
`inner_multi` (+ `Allreduce` when distributed).

**Two cross-cutting requirements for every observable in this phase:**

- **Degenerate manifolds (correctness).** Symmetry *causes* degeneracy, and block
  Lanczos returns a *block* of vectors spanning the low-energy eigenspace. Any single
  Lanczos vector is an arbitrary linear combination within a degenerate manifold, so
  `⟨ψ|Ô|ψ⟩` on it is **not** well-defined (e.g., it cannot distinguish a singlet from
  the `S_z=0` component of a triplet). Observables that do not commute with `H` within
  the manifold must be evaluated by **diagonalizing `Ô` restricted to the degenerate
  eigenspace** and reporting the eigenvalues. See §1.5.
- **Finite temperature.** This package is finite-T throughout (`average.py`'s
  `thermal_average`, `tau`, `e_max` Boltzmann cutoffs). Report observables as thermal
  averages over the low-energy manifold, `⟨Ô⟩ = Σₙ e^{−βEₙ}⟨n|Ô|n⟩ / Z`, reusing
  `average.py` — not T=0 only. T=0 is the special case `β → ∞`.

### 1.0. Preserve the existing two-level reporting (must-not-regress)

The current code already prints observables at **two levels**, and this debugging-
useful behavior must be preserved verbatim (only *extended* with the new observables):

1. **Thermal ground state** — `print_thermal_expectation_values` (in `finite.py`,
   called from `groundstate.py:395`) prints the Boltzmann-averaged ρ observables:
   `<E-E0>`, `<N>`, `<N(Dn)>`, `<N(Up)>`, per-equivalent-block `<N(...)>`, `<Lz>`,
   `<Sz>`.
2. **Per-eigenstate** — `print_expectation_values` (called from `groundstate.py:397`)
   prints a table with **one row per eigenstate** included in the thermal ensemble:
   `i`, `E-E0`, `N`, `N(Dn)`, `N(Up)`, per-block `N`, `Lz`, `Sz`.

Both are gated behind `verbose` and are followed by the per-eigenstate occupation
statistics (impurity/valence/conduction weights, `groundstate.py:398-404`).

- **Action:** Extend **both** functions (and the per-eigenstate table) with the new
  observables from §1.1–§1.5 — `<L·S>`, `<Ŝ²>`/`S`, `<L̂²>`/`L`, `<Ĵ²>`/`J`, the
  impurity-local versions, and the **impurity–bath spin correlation `<S_imp·S_bath>`**
  (§1.4, the key Kondo-physics diagnostic) — keeping the existing columns/lines and
  ordering intact. The thermal level uses the thermal average; the per-eigenstate level
  reports each eigenstate individually (the natural place to expose the per-state
  `S(S+1)`, and to watch `<S_imp·S_bath>` go negative as the Kondo singlet forms).
- **Architectural note — not everything comes from ρ.** The existing print functions
  receive only the impurity-block **1-RDM** (`rho`). That suffices for `N`, `Lz`, `Sz`,
  `L·S`. It does **not** suffice for the two-body observables (`Ŝ²`, `L̂²`, `Ĵ²`) or
  for `<S_imp·S_bath>`, which couples two *disjoint* orbital sets and cannot be
  recovered from the impurity 1-RDM at all. Those must be evaluated as
  `⟨ψ|Ô|ψ⟩` on the actual state(s) via the `ManyBodyOperator` apply + inner path
  (distributed-aware, with `Allreduce`). The reporting functions therefore need access
  to the eigenstates / thermal ensemble, not just `rho` — plumb the states through (or
  pre-compute the scalar observables and pass them in) rather than trying to derive
  them from `rho`.
- **Action:** For the per-eigenstate Casimir values, respect the degenerate-manifold
  rule (§1.5): states within a degenerate manifold share the diagonalized-within-
  manifold eigenvalue, not a per-vector `⟨ψ|Ô|ψ⟩`.
- **Verification:**
  - [x] `test_print_expectation_values_columns` — existing columns preserved; `L.S`
        column appended (and `S` column when `s_values` is passed). Also
        `test_print_expectation_values_S_column`.
  - [x] `test_print_thermal_expectation_values_lines` — existing lines preserved;
        `<L.S>` line added (and `<S^2>`/`S` when `s_thermal` is passed). Also
        `test_print_thermal_S2_line`.
  - [x] Existing verbose-path tests still pass (`test_groundstate.py` serial + MPI n=2,
        full serial suite 315 passed / 172 skipped / 50 xfailed).

  **Done (2026-06-24):** `<L.S>` (1-body, rho-spherical) wired into **both** print
  levels. `get_LS_from_rho_spherical` made robust to non-spin-doubled (odd) impurity
  blocks by contracting the leading `2(2l+1)` sub-block (matches the index-based
  `get_Lz`/`get_Sz`). The print functions gained optional `s_values`/`s_thermal`
  parameters (default `None` ⇒ byte-identical output) so an `S` column / `<S^2>` line
  can be filled in later. An impurity `<S^2>` computation was briefly wired into
  `groundstate.py` and then **reverted** (it used the spherical-only `(k, k+n/2)`
  pairing on computational indices — see the deferral note above); it will be redone
  on top of Phase 5's basis-rotation map. The `<S^2>`/`S` formatting path is still
  covered by `test_print_thermal_S2_line` / `test_print_expectation_values_S_column`,
  which pass the values in directly.

  **Still open (needs a decision — STOP-AND-ASK):** `<L^2>`/`<J^2>` reporting and the
  `<S_imp·S_bath>` (`test_kondo_correlation_reported`) and §1.3
  `test_impurity_local_observables` reporting. `L^2`/`J^2` on the *states* need the
  orbital (`ml`) structure, i.e. the single-particle basis rotation to spherical
  harmonics applied to the many-body states — this couples to Phase 5 (basis
  rotation), unlike `S^2` which is spin-only and basis-independent. `S_imp·S_bath`
  needs the **bath** spin-orbital `(dn, up)` pairing convention, which is not pinned
  down in the code the way the impurity one is (`_generate_spin_flipped_determinants`).
  Both the unit-level operators already exist and are tested
  (`apply_spin_correlation`/`expect_spin_correlation`, `make_orbital_angular_momentum_operators`);
  only the *reporting wiring* is blocked on these conventions.

### 1.1. One-body observables (⟨L·S⟩)

- **Action:** Compute `⟨L·S⟩` as a 1-body contraction of the 1-RDM ρ from
  `build_density_matrices`. No new solver needed.
- **Verification:**
  - [x] `test_spin_orbit_observable` — compute `L·S` on a known analytic test state,
        assert match to analytic result.

### 1.2. Two-body observables (Ŝ², L̂², Ĵ²)

- **Action:** Construct `Ŝ²`, `L̂²`, `Ĵ²` as `ManyBodyOperator` objects. Evaluate
  `⟨ψ|Ô|ψ⟩` via `op.apply(psi)` + `inner` (+ `comm.Allreduce` for distributed
  states).
- **Note:** `Ŝ²` is a two-body operator — it will **not** appear in the one-body
  commutant search of Phase 2. It is constructed explicitly here. (The companion plan
  `nonabelian_symmetry_casimir.md` shows how to *reconstruct* the same `Ŝ²` from the
  one-body generators that Phase 2 *does* find; the two constructions must agree
  bit-for-bit — pick one as the source of truth and assert the other against it.)
- **Verification:**
  - [x] `test_S2_observable` — apply `Ŝ²` to prepared singlet/doublet/triplet states,
        assert eigenvalues `S(S+1)` exactly.
  - [ ] Update output to tag ground-state multiplets by exact `L`, `S`, `J`.

### 1.3. Local impurity diagnostics

- **Action:** Restrict operator construction to `impurity_orbitals`; compute
  `⟨N̂ᵢₘₚ⟩`, `⟨Ŝ²ᵢₘₚ⟩`, `⟨L̂²ᵢₘₚ⟩`, `⟨Ĵ²ᵢₘₚ⟩`.
- **Verification:**
  - [ ] `test_impurity_local_observables` — verify non-integer values emerge
        continuously as hybridization increases (mixed-valence physics).

### 1.4. Impurity–bath correlation (Kondo screening)

- **Action:** Construct the two-body operator `Ŝ_imp·Ŝ_bath = Σ_{i∈imp, j∈bath}
  Ŝ_i·Ŝ_j` where `Ŝ_i·Ŝ_j = Sᶻ_iSᶻ_j + ½(S⁺_iS⁻_j + S⁻_iS⁺_j)`. The impurity vs bath
  orbital partition comes from `impurity_orbitals` and the bath orbitals
  (valence + conduction) — the same partition the per-eigenstate occupation statistics
  already use (`groundstate.py:398-404`). Evaluate `⟨ψ|Ŝ_imp·Ŝ_bath|ψ⟩` on the state
  (this is **not** obtainable from the impurity 1-RDM; see the §1.0 architectural note).
- **Action:** Report it at **both** levels per §1.0: thermally averaged for the
  thermal ground state, and per-eigenstate in the table. A negative value signals
  Kondo-singlet formation (impurity spin antiferromagnetically screened by the bath).
- **Verification:**
  - [x] `test_kondo_correlation` — minimal SIAM; assert `<Ŝ_imp·Ŝ_bath>` becomes
        negative in the strongly interacting Kondo regime and ≈ 0 in the
        weakly-coupled / empty-impurity limit.
  - [ ] `test_kondo_correlation_reported` — assert `<S_imp·S_bath>` appears in both the
        thermal and per-eigenstate output (ties into §1.0's column/line tests).

### 1.5. Degenerate-manifold and thermal evaluation (infrastructure for 1.1–1.4)

- **Action:** Provide a single helper that, given a low-energy manifold (block of
  eigenvectors + energies) and an operator `Ô`, returns the correct expectation
  value(s):
  1. Group eigenvectors into degenerate subspaces (energies equal to a tolerance tied
     to the Lanczos convergence).
  2. Within each subspace, build the small matrix `O_mn = ⟨m|Ô|n⟩` and diagonalize it
     — its eigenvalues are the physical observable values (e.g. `S(S+1)`); a single
     `⟨n|Ô|n⟩` is wrong when `[Ô,H]≠0` within the subspace.
  3. Combine across subspaces with Boltzmann weights via `average.py`.
- **Verification:**
  - [x] `test_degenerate_manifold_observable` — a Hamiltonian with a deliberately
        degenerate triplet ground state; assert that `Ŝ²` evaluated on each of three
        arbitrary linear combinations within the manifold gives the same `S(S+1)=2`,
        whereas the naive per-vector `⟨ψ|Ŝ²|ψ⟩` does not.
  - [x] `test_thermal_observable` — assert the finite-T average matches an independent
        brute-force Boltzmann sum on a small model, and reduces to the T=0 value as
        `T→0`.

---

## Phase 2: Symmetry Discovery Engine

> **DONE (2026-06-24).** Implemented in new `src/impurityModel/ed/symmetries.py`,
> tested in `src/impurityModel/test/test_symmetries.py` (9 tests, green):
> - §2.1 `extract_tensors` / `tensors_to_operator` — 1-/2-body coefficient tensors
>   (`h_ij`, `V_ijkl` matching the real `c†c†aa` Coulomb ordering), round-trip + reject
>   3-body / non-number-conserving (`test_tensor_extraction*`).
> - §2.2 `discover_one_body_symmetries` — null space of the commutator super-operator
>   `A = I⊗h − hᵀ⊗I` (column-stacking `vec`, verified by `‖[h,G]‖<1e-10` per generator),
>   documented `sigma_cut = ‖A‖·n·ε`. `in_span` helper. SU(2) Hubbard dimer finds
>   N, S_z, **S_x, S_y** (8-dim commutant); count stable across cutoffs
>   (`test_symmetry_null_space`, `test_null_space_threshold_stability`).
> - §2.3 `hermitian_algebra_basis`, `cartan_subalgebra` (centralizer of a generic
>   regular element — robust vs sequential diagonalisation), `joint_diagonalize`
>   (random combination → common eigenbasis even with degenerate generators),
>   `weights_are_01` (the {0,1}-occupation checkpoint: `N`→restriction-able,
>   `S_z`→Phase 6). `test_cartan_subalgebra_and_joint_diagonalization`,
>   `test_joint_diagonalization_degenerate`, `test_generator_weight_classification`.
>
> Anti-unitary (time-reversal/Kramers) limitation documented in the module docstring.
> **Next:** Phase 3 (sectorized CIPSI) or Phase 5 (basis rotation, the observable-
> reporting unlock). The `weights_are_01` split is the Phase 3 ↔ Phase 6 routing.

**Location:** `src/impurityModel/ed/symmetries.py` (new file, Python + SciPy)

**Scope:** One-body symmetry algebra only. The constraint `[H,O]=0` returns the full
one-body algebra — for an SU(2)-symmetric model this is **all three** spin generators
`{S_x, S_y, S_z}` (each is one-body), not just `S_z`. This phase selects the abelian
(Cartan) subalgebra `{N̂, S_z}` for sector labels; the non-abelian remainder and the
reconstruction of `Ŝ²`/`L̂²`/`Ĵ²` from these generators are handled in the companion
plan `nonabelian_symmetry_casimir.md`. `Ŝ²` itself is two-body and will **not** appear
in the null space directly.

**Known limitation (state explicitly):** `[H,O]=0` over the complex field finds only
**unitary** symmetries. **Anti-unitary symmetries — time reversal — are not
detectable by this method**, so Kramers degeneracies are not discovered here. H is
genuinely complex in this code (spin-orbit coupling), so this is a real gap, not a
theoretical aside; document it where the discovered symmetries are reported.

### 2.1. Tensor extraction **[strong-model]**

- **Inspect first (do this before writing code):** find how a `ManyBodyOperator` exposes
  its terms — `grep -n "vector<pair\|OPS\|apply_multi\|def __iter__\|token" src/cython/ManyBodyOperator.pyx src/cython/ManyBodyOperator.pxd src/cython/ManyBodyOperator.h`
  and the Python parser `grep -n "def \|c\", \|a\"" src/impurityModel/ed/op_parser.py`.
  Confirm the exact token representation (operator string like `((i,"c"),(j,"a"))` →
  scalar) before extracting; the dense-tensor mapping depends on that format.
- **Action:** Extract `hᵢⱼ` (coefficient of `c†_i c_j`) and `Vᵢⱼₖₗ` (coefficient of
  `c†_i c†_j c_l c_k`) into dense numpy arrays. Assert the operator is purely 1- and
  2-body before extraction (raise on any token with >4 operators rather than silently
  dropping it).
- **Verification:**
  - [x] `test_tensor_extraction` — round-trip: reconstruct a `ManyBodyOperator` from
        the extracted dense tensors and assert it matches the original.
  - [x] `test_tensor_extraction_rejects_3body` — operator with a 3-body term raises
        or warns explicitly.

### 2.2. Linear constraint system & null space (SVD) **[strong-model]**

- **Action:** A one-body symmetry generator is a single-particle matrix `O` (shape
  `n_orb × n_orb`). At the one-body level the commutant condition reduces to the
  single-particle commutator `[h, O] = 0` (with `h` the one-body matrix from 2.1;
  two-body invariance follows for the spin/charge generators and is checked separately
  via `test_casimir_commutes_with_H`). Vectorize `O` (`vec(O)`, length `n_orb²`) and
  build the constraint operator from the commutator:
  ```python
  # [h, O] = h@O - O@h = 0  ->  (I⊗h - hᵀ⊗I) vec(O) = 0   (column-stacking vec)
  import numpy as np, scipy.linalg as sla
  I = np.eye(n_orb)
  A = np.kron(I, h) - np.kron(h.T, I)          # (n_orb², n_orb²)
  U, s, Vh = sla.svd(A)
  null_mask = s < sigma_cut                      # see threshold below
  null_vecs = Vh.conj().T[:, null_mask]          # each column is vec(O_a)
  generators = [v.reshape(n_orb, n_orb) for v in null_vecs.T]
  ```
  (If H carries spin-orbit, `h` already spans spin⊗orbital, so `S_x,S_y,S_z` emerge
  naturally as one-body generators in that combined space.)
- **Critical parameter — the null-space threshold.** With floating-point H the null
  space is *approximate*; the singular-value cutoff decides how many symmetries are
  "found." Too tight misses real symmetries; too loose invents spurious ones. Make the
  cutoff an explicit, documented parameter with a default scaled to the problem, e.g.
  `σ_cut = ‖A‖ · max(n_orb) · ε`. Test sensitivity: the discovered count must be
  stable across a range of cutoffs around the default.
- **Verification:**
  - [x] `test_symmetry_null_space` — SU(2)-invariant Hubbard dimer where the answer
        is known by hand. Assert that `N̂`, `Ŝ_z`, **and** `S_x`, `S_y` (all one-body)
        appear in the null space; assert that `Ŝ²` does **not** (it is two-body and
        cannot appear here by construction).
  - [x] `test_null_space_threshold_stability` — discovered symmetry count is constant
        across cutoffs spanning ~2 orders of magnitude around the default.

### 2.3. Cartan subalgebra & joint diagonalization **[strong-model]**

- **Action:** Extract a maximal mutually-commuting subset of discovered symmetry
  matrices; jointly diagonalize to yield the single-particle transformation `U`.
- **Numerics — do not diagonalize sequentially.** Naive "diagonalize `O₁`, then `O₂`
  inside its degenerate subspaces, …" is fragile under degeneracy. Use the standard
  robust approach: form a **random real linear combination** `M = Σₖ rₖ Oₖ` of the
  commuting generators. Generic `rₖ` give `M` a non-degenerate spectrum, so its
  eigenvectors simultaneously diagonalize *all* the `Oₖ`. Then read off each
  generator's eigenvalues in that basis and verify off-diagonals vanish.
- **Verification:**
  - [x] `test_joint_diagonalization` — commutators of all diagonalized generators are
        zero (`‖[Oₐ, O_b]‖ < 1e-12`), and each `Oₖ` is diagonal in the common basis to
        the same tolerance.
  - [x] `test_joint_diagonalization_degenerate` — a case with a degenerate generator
        spectrum; assert the random-combination method still yields a common
        eigenbasis (the sequential method would fail here).
  - [x] **Additional checkpoint:** assert that each resulting generator is
        integer-valued on the {0,1} occupation basis when possible. Flag any generator
        with non-{0,1} orbital weights — those cannot be expressed as restriction
        masks and require Phase 6 to use for sectorization.

---

## Phase 3: Sectorized CIPSI (subset-only restrictions)

> **§3.1 core DONE (2026-06-24).** `symmetries.py`: `conserved_subset_charges(op)`
> finds the **finest** orbital partition whose per-block occupation commutes with the
> **full** interacting `H` (union-find: one-body terms force-merge their orbitals,
> two-body terms merge any block they imbalance, to a fixed point — so it respects the
> 2-body interaction, unlike the Phase-2 one-body commutant). `restrictions_from_charges`
> / `subset_occupations` map charges + a reference occupation to the
> `Basis.restrictions` format (`{frozenset: (min,max)}`, optional `slack` for `N±1`
> neighbour sectors). Tested in `test_sectorization.py` (6 tests): Hubbard dimer →
> `N_up`/`N_down`; **many-body `[H, N_S]=0`** verified by brute-force ED on every
> sector; spin-flip (1-body) and pair-hopping (2-body) correctly **merge** to total `N`;
> density-density correctly does **not** merge; restriction-dict format.
>
> **Remaining:** §3.0 target-sector pre-scan and §3.2 CIPSI/Basis integration
> (`test_automatic_restrictions`, `test_target_sector_selection`) — wire the
> auto-charges into `Basis.restrictions` / `op.set_restrictions` and show the sectorized
> `CIPSISolver` reproduces the unrestricted ground state with a smaller basis. The
> many-body block-diagonal property is already proven (`[H, N_S]=0`); what remains is the
> solver wiring + the "smaller basis, same energy" end-to-end test.

**Location:** `src/impurityModel/ed/cipsi_solver.py` and `manybody_basis.py`

**Scope:** This phase uses only the symmetries whose weights are `{0,1}` on an
orbital subset (e.g., total particle number `N`, spin-up count, valence/conduction
counts, point-group-disconnected orbital sets). Symmetries with fractional weights
(e.g., `S_z = Σ ±½ nᵢ`) require Phase 6 to become restrictions.

### 3.0. Target-sector selection (prerequisite — easy to overlook)

> **DONE (2026-06-24).** `groundstate.prescan_ground_state_sector`: one rough CIPSI over
> a broad valence window (`scan_width` beyond `mixed_valence`) replaces the `O(3^k)` `dN`
> scan. The cleverness is that the rough ground state's **impurity occupation is
> measured** (`measure_conserved_charges` on the impurity subsets → the winning nominal
> `N0`) instead of guessed, and its **conserved-charge sector** is read off at the same
> time for the restricted refine. `calc_energy` gained `return_state=True` to expose the
> rough GS vector. Wired into `find_ground_state_basis(..., use_prescan=True)` (default
> `False` ⇒ no regression). Tests (`test_sectorization.py`):
> `test_prescan_finds_ground_state_sector` (rough scan hits the true global GS energy and
> a valid sector — note the GS doublet's `S_z=±½` components are both correct;
> restricting to the sector reproduces the energy with a smaller basis),
> `test_prescan_matches_brute_force_loop` (the wired path equals the `dN` scan),
> `test_measure_conserved_charges`. Verified serial + **MPI n=2** (the distributed
> `Allreduce` measurement path), and the default `use_prescan=False` path is unchanged
> (`test_groundstate.py` serial + MPI green).

Sectorizing CIPSI only helps if you target the sector(s) that actually contain the
ground state — and **you usually do not know that sector a priori** (mixed-valence and
Kondo problems are the whole point of this package, and the relevant `N`/`S_z` is not
obvious). A sectorized search that guesses wrong silently returns the wrong ground
state.

- **Action (decision made):** Before committing to a single sector, run a cheap
  **low-accuracy unrestricted CIPSI/Lanczos pre-scan** (loose `tol`, small
  `max_subspace_blocks`) and read off the ground-state quantum numbers via the Phase 1
  observables / Phase 2 generators; lock in that sector (plus its immediate neighbours
  `N ± 1`, adjacent `S_z`, to be safe) for the high-accuracy run. Do **not** implement
  the alternative full window-sweep — the pre-scan is the chosen approach; a sweep is
  only a manual fallback if the pre-scan is ever ambiguous.
- **Verification:**
  - [x] `test_target_sector_selection` — model whose ground state lives in a
        non-obvious sector; assert the pre-scan identifies it and the sectorized run
        reproduces the unrestricted ground-state energy.

### 3.1. Symmetry → restriction mapping (subset-only)

- **Action:** From the generators flagged as integer-valued in Phase 2.3, map each
  one to a `Basis.restrictions` entry: identify the orbital subset `S` and
  min/max electron counts. Generators with non-{0,1} weights are skipped in this
  phase (recorded for Phase 6).
- **Verification:**
  - [x] `test_automatic_restrictions` — auto-generated restrictions on a model with
        known symmetry sectors; assert `CIPSISolver` finds the same ground state as
        an unrestricted run but with a smaller basis.

### 3.2. Sectorized basis generation

> **DONE (2026-06-24).** `test_sectorization.py`: the auto-generated restrictions
> (`restrictions_from_charges`) wired into `Basis(restrictions=...)` work end-to-end on
> a 4-orbital Anderson model.
> - `test_automatic_restrictions` — full Fock space (16 dets) vs the auto-restricted GS
>   sector (4 dets): **identical** ground-state energy with a **4× smaller** basis. The
>   GS sector is read off the unrestricted GS via the conserved charges (a mini
>   pre-scan; full §3.0 below).
> - `test_restrictions_refuse_out_of_sector_in_expand` — restrictions **actively prune**:
>   `Basis.expand` with a sector-leaving operator stays in-sector, while the same expand
>   without restrictions leaks out-of-sector. This exercises the real
>   `op.set_restrictions` + `build_restriction_mask` path.
>
> The "`H_mat` block-diagonal" property is the same fact as `[H, N_S]=0`, already proven
> in §3.1 (`test_conserved_charges_actually_commute`).

- **Action:** The existing `op.set_restrictions(basis.restrictions)` +
  `build_restriction_mask` machinery already causes `expand` and `determine_new_Dj`
  to refuse out-of-sector SDs. The task is to wire the auto-generated restrictions
  in and verify, not to rebuild. `build_sparse_matrix` already only stores
  connections within the basis — the block-diagonal structure is a consequence of
  correct basis generation, not a separate matrix transformation.
- **Verification:**
  - [x] `H_mat` block-diagonal: equivalent to `[H, N_S]=0`, proven in §3.1
        (`test_conserved_charges_actually_commute`); the sectorized basis only spans one
        sector (`test_automatic_restrictions`).
  - [x] Auto-restrictions integrate with `Basis`/`expand` and reduce the basis while
        preserving the ground state (`test_automatic_restrictions`,
        `test_restrictions_refuse_out_of_sector_in_expand`). Broader serial + MPI suites
        still green.

> **§3.0 (target-sector pre-scan) remaining:** `test_automatic_restrictions` already
> reads the GS sector off an unrestricted solve (a mini pre-scan). The full §3.0 is the
> *cheap low-accuracy* unrestricted CIPSI/Lanczos pre-scan + lock-in (sector ± neighbour
> via `slack`) wired into `cipsi_solver.py` / `groundstate.py` — a production-pipeline
> piece, not needed to prove the sectorization itself.

---

## Phase 4: Auto-Routing Green's Functions

**Location:** `src/impurityModel/ed/greens_function.py`, `spectra.py`, `selfenergy.py`

### 4.1. Selection rules for decoupled blocks

> **DONE (2026-06-24).** `symmetries.green_function_block_structure(op)` (=
> `conserved_subset_charges`, with GF semantics) + `green_function_allowed_mask`:
> `G_ij` can be nonzero only if `i,j` share a conserved-charge subset (else *every*
> moment `<ψ|c_i H^n c_j†|ψ>` vanishes). `test_gf_autorouting.py::test_automatic_decoupling`
> builds the dense GF over the full Fock space and asserts forbidden entries are
> **exactly zero**; `test_symmetry_breaking_couples_blocks` shows a spin-flip term merges
> the blocks (GF fully coupled).

- **Action:** Replace manual `block_structure` definitions. Use
  `Δqᵢⱼ = wᵢ − wⱼ` (weight difference from the discovered symmetry generators) to
  determine which `(i,j)` transitions are symmetry-forbidden. This generalizes the
  connectivity graph already in `build_initial_restrictions` (`scipy.sparse.csgraph
  .shortest_path` over the hopping graph).
- **Verification:**
  - [x] `test_automatic_decoupling` — assert that transitions flagged as decoupled
        are exactly zero in a brute-force benchmark calculation.

### 4.2. Wire up existing block equivalence detection ✅

> **Partial (2026-06-24).** `symmetries.auto_block_structure(op, orbitals=...)` derives a
> full `BlockStructure` from the one-body `h` (existing `build_block_structure`),
> replacing the hand-coded one. `test_gf_autorouting.py::test_auto_block_structure_detects_equivalences`
> confirms the blocks match the GF sectors **and** the `identical_blocks` detection
> correctly groups the degenerate eg `{0,1,5,6}` and t2g `{2,3,4,7,8,9}` orbitals.
>
> **🐛→✅ BUG FIXED in `get_inequivalent_blocks`** (`block_structure.py`, 2026-06-24):
> a block that was particle-hole/transpose-related to blocks **within its own identical
> group** (e.g. the self-PH-symmetric t2g at zero energy: `block 2 ∈ particle_hole[2]`)
> was wrongly dropped from `inequivalent_blocks` — giving `inequivalent_blocks=[0]` for
> the cubic d-shell (only eg), so a GF reconstruction would **miss t2g entirely**. Latent
> in production (hand-coded structures don't use the auto PH-reduction). **Fixed** by
> replacing the ad-hoc skip logic with **union-find over all relations** (identical ∪
> transposed ∪ PH ∪ PH-transpose), one representative per class. Now
> `inequivalent_blocks=[0,2]`. Existing `test_block_structure.py` + GF/spectra suites
> still green (full serial suite 351 passed).

- **Action:** `block_structure.py` already computes `identical_blocks`,
  `transposed_blocks`, `particle_hole_blocks`, `particle_hole_transposed_blocks`,
  `inequivalent_blocks` and reconstructs via `build_matrix`/`build_greens_function`.
  The task is to drive this existing detection from the discovered symmetry quantum
  numbers and use `inequivalent_blocks` in the spectra loop to skip redundant tOp
  evaluations.
- **Bug fixed:** `build_greens_function` previously called `np.transpose(tuple(...))`
  (transposing a tuple literal, not the block array). Fixed: now uses
  `m.swapaxes(-2, -1)`. Tests `test_build_greens_function_transposed`,
  `_particle_hole`, `_particle_hole_transposed` added and passing.
- **Verification:**
  - [x] `test_advanced_block_equivalence` — particle-hole symmetric model and
        octahedral field model; assert that skipping equivalent tOps produces
        identical spectral functions to the full multi-tOp loop.

> **DONE (2026-06-24).** `symmetries.gf_sector_restrictions(charges, gs_occ, orbital,
> kind)`: the addition (`c_j†`, sector `q_ψ + w_j`) and removal (`c_j`, sector
> `q_ψ − w_j`) parts get their own sector restrictions, confining each Lanczos run
> before it starts. `test_gf_autorouting.py`: `test_addition_removal_sectors` (addition
> shifts the orbital's charge +1, removal −1, others unchanged);
> `test_green_function_explosion_prevention` (the addition GF computed in the confined
> sector equals the full-Fock GF with a strictly smaller basis, and `c_j†|ψ⟩` has zero
> weight outside the sector).

### 4.3. Krylov-space sector confinement (addition vs removal)

- **Action:** `block_lanczos_cy` operates on a **fixed basis** — it does not expand
  the Hilbert space dynamically. Sector confinement is applied **before** the run by
  calling `tOp.set_restrictions(...)` with the target sector quantum numbers, so that
  `apply_multi` never produces out-of-sector SDs. The existing
  `build_excited_restrictions` path already does this for the GF; the task is to derive
  the sector target from the auto-discovered quantum numbers rather than from a
  hand-written `block_structure`.
- **Addition and removal land in different sectors — keep them separate.** The GF
  `G_ij(ω) = ⟨ψ| cᵢ (ω−H+E)⁻¹ cⱼ† |ψ⟩` has an addition part (electron, `cⱼ†`) living in
  the `(N+1, S_z+½, …)` sector and a removal part (hole, `cⱼ`) in `(N−1, S_z−½, …)`.
  The restriction target is `q_Ψ + w_j` for the addition part and `q_Ψ − w_j` for the
  removal part — **not** a single `q_Ψ − w_j` for both. Set each part's operator
  restrictions to its own sector.
- **Symmetry-adapted start block (performance):** when the target sector is known, seed
  the initial Lanczos block from vectors already inside that sector (e.g. `cⱼ†|ψ⟩`
  components restricted to the sector) rather than a generic random block. The Krylov
  space then never has to "project out" other sectors, giving faster convergence and a
  smaller subspace.
- **Verification:**
  - [x] `test_green_function_explosion_prevention` — assert that the operator's
        restriction mask excludes out-of-sector states before the Lanczos run, and
        that the resulting GF matches an unrestricted reference while the Krylov
        basis is substantially smaller.
  - [x] `test_addition_removal_sectors` — assert the addition and removal parts target
        `(N+1)` and `(N−1)` sectors respectively, and that recombining them reproduces
        the full unrestricted `G_ij(ω)`.

---

## Phase 5: Basis Rotation

> **§5.1 DONE (2026-06-24).** Implemented in `symmetries.py`, tested in
> `src/impurityModel/test/test_basis_rotation.py` (4 tests green):
> `rotate_one_body` (`U† h U`), `rotate_two_body`
> (`V'_ijkl = Σ U*_mi U*_nj V_mnpq U_pk U_ql`, matching `extract_tensors`' V), and
> `rotate_hamiltonian` (extract → rotate → rebuild `ManyBodyOperator`). The 2-body
> index convention is pinned by `test_python_basis_rotation_spectrum_invariant`, which
> checks the **many-body** spectrum is invariant across the N=1,2,3 sectors (a brute-
> force complete-sector ED) — this would fail for any V-convention mismatch.
> `test_python_basis_rotation_roundtrip` (U then U† recovers the tensors to 1e-12) and
> `test_rotation_diagonalizes_discovered_generators` (in the joint-eigenbasis the
> Cartan generators are diagonal) cover the rest. Capstone
> `symmetry_adapted_transformation(op)` = discover → Cartan → `U` → `H'`, the bridge to
> Phase 3 (`test_symmetry_adapted_transformation_bridges_to_phase3`).
>
> **Not yet done:** §5.2 pipeline integration (data migration / rebuild+redistribute
> distributed states across the rotation boundary, DMFT `U`-cache, hash-balance) and
> §5.3 optional Cython. §5.2 is the heavy distributed-state piece; Phase 3
> sectorization can proceed on `rotate_hamiltonian` + `symmetry_adapted_transformation`
> without it (it builds fresh restricted bases rather than migrating existing states).

**Location:** `src/impurityModel/ed/symmetries.py` (reference), optionally
`src/cython/basis_rotation.pyx` (optimized, only if profiling proves necessary)

**Note:** The 2-body rotation `H' = U†HU` is a **one-time setup cost**, not a hot
loop. `numpy.einsum`/`tensordot` (which dispatch to BLAS internally) are almost
certainly sufficient. A bespoke Cython kernel would benchmark against the wrong
baseline until `matmul_nogil` in `BlockLanczosArray.pyx` is replaced with real BLAS
(`blocklanczos_blas_acceleration.md`). Implement with `np.einsum` first; only port
to Cython if profiling shows it dominates total run time.

### 5.1. Reference Python implementation

- **Action:** Implement `H' = U†HU` via `np.einsum`/`tensordot` for both 1-body
  (`h'ᵢⱼ = Σ Uₖᵢ* hₖₗ Uₗⱼ`) and 2-body (`V'ᵢⱼₖₗ = Σ Uₘᵢ* Uₙⱼ* Vₘₙₚq Uₚₖ Uqₗ`)
  tensors. Re-express the result as a `ManyBodyOperator` in the rotated single-particle
  basis.
- **Verification:**
  - [x] `test_python_basis_rotation` — eigenvalues of `H` and `H'` match to 1e-12.
  - [x] In `H'`, the discovered abelian symmetry generators are diagonal matrices.

### 5.2. Pipeline integration (critical)

- **Action:** After rotation, the **entire pipeline** runs in the rotated basis:
  `Basis._get_initial_basis`, the operator parser, restrictions, density matrices,
  observable operators. The rotation "block-diagonalizes" only if the discovered
  symmetries are diagonal occupation numbers in the new basis.
- **Data migration (correctness — the subtle one).** A rotation `U` invalidates
  **every persisted distributed state**: cached ground states, `psi_refs`, density
  matrices, and any `StateContainer` are expressed in the *old* single-particle basis.
  You **cannot mix** pre- and post-rotation states in the same computation, and because
  rotation changes SD bit patterns it also changes the `hash % size` ownership — so all
  state containers must be **rebuilt and redistributed** at the moment of rotation.
  Define an explicit "rotation boundary" in the workflow and forbid carrying old-basis
  states across it.
- **DMFT caching (performance).** In a self-consistency loop H changes every iteration
  (bath fitting), but the symmetry *structure* (point group + spin) is fixed by the
  problem — only coefficients move. **Discover `U` once and cache it across DMFT
  iterations**; re-discover only if the symmetry structure changes (cheap to check via
  the Phase 2 generators). Do not re-run the full discovery every iteration.
- **Verification:**
  - [ ] Re-express `H` as a `ManyBodyOperator` in the rotated basis and run a full
        CIPSI + GF calculation; assert ground-state energy and spectral peaks match
        the unrotated result.
  - [ ] `test_rotation_state_migration` — assert that a state container rebuilt after
        rotation contains the same physical state (overlap 1 with the migrated
        reference) and that attempting to combine an old-basis state with a new-basis
        operator is rejected, not silently wrong.
  - [ ] `test_dmft_symmetry_cache` — across two DMFT iterations with the same symmetry
        structure, assert discovery runs once (cache hit on the second).
  - [ ] **Hash-balance check:** verify that the C++ `hash % comm.size` distribution
        remains approximately load-balanced after rotation. If severely imbalanced,
        document the threshold and add a warning.

### 5.3. Optimized implementation (only if needed)

- **Action:** If and only if Step 5.1 profiles as a bottleneck, port the 2-body
  contraction to a Cython kernel using `scipy.linalg.cython_blas.zgemm` (after the
  BLAS work in `blocklanczos_blas_acceleration.md` is complete).
- **Verification:**
  - [ ] `test_cython_basis_rotation` — output numerically identical to Step 5.1 to
        within 1e-14.
  - [ ] Benchmark shows >10× speedup over the `einsum` path for the system sizes
        where this is actually invoked.

---

## Phase 6: Extended Restriction Machinery (weighted-sum charges)

**Location:** `src/cython/ManyBodyOperator.h`, `SlaterDeterminant.h`

**Rationale:** Phase 3 covers symmetries with `{0,1}` orbital weights. This phase
unlocks `S_z`, `L_z`, and general abelian charges `Σ wᵢ nᵢ = q` by extending the
C++ restriction machinery.

### 6.1. Weighted-sum restriction kind

- **Action:** Add a new restriction variant to the `Restrictions` type (currently
  `vector<pair<vector<size_t>, pair<size_t,size_t>>>`). The new kind stores orbital
  weights `wᵢ ∈ ℚ` and a target value `q`, checked in
  `state_is_within_restrictions`. Update `build_restriction_mask` and
  `apply`/`apply_multi` accordingly.
- **Verification:**
  - [ ] `test_weighted_restriction_Sz` — build `S_z` restriction; assert that only
        states with the correct total `S_z` pass the mask.
  - [ ] `test_weighted_restriction_roundtrip` — `ManyBodyOperator` with `S_z`
        restriction applied to a basis spanning multiple sectors; assert only the
        target sector survives.

### 6.2. Wire into Phase 3 restriction mapping

- **Action:** Re-run Phase 3.1 with the extended mask; generators with fractional
  weights (skipped in Phase 3) can now be mapped.
- **Verification:**
  - [ ] `test_automatic_restrictions_Sz` — assert `S_z` restriction is auto-generated
        and produces correct sector isolation.

---

## Phase 7: MPI Adaptive Split Policy

**Location:** `src/impurityModel/ed/spectra.py`, `greens_function.py`

**Critical constraint:** Do **not** remove `MPI.Comm.Split()`. The split is task
parallelism: `split_basis_and_redistribute_psi` assigns different tOp-blocks,
energies, frequencies, and GF blocks to disjoint sub-communicators. Removing it
converts N cheap independent solves into N globally-synchronized solves, which is
worse whenever there are many small independent blocks/frequencies (the common case).
The intercomm-freeing discipline in `get_Greens_function` and
`split_basis_and_redistribute_psi` is correct and subtle — preserve it.

**Prerequisite:** `blocklanczos_blas_acceleration.md` Items 1–3 must be complete
before `block_lanczos_array_cy` is viable as a multi-tOp engine. It currently uses a
hand-rolled triple-loop matvec and `Allreduce`s a full `global_N × p` buffer every
iteration; it is not a drop-in for the hash-distributed sparse path.

### 7.1. Adaptive split heuristic

- **Action:** Replace the current fixed split policy with a heuristic: split when
  `n_blocks × estimated_cost_per_block_spread > threshold`; use a unified
  communicator when blocks are few and large. Document the threshold and expose it as
  a tunable parameter.
- **Verification:**
  - [ ] `test_mpi_load_balancing_spectra` — assert numerical equivalence of GF output
        between split and unified modes.
  - [ ] Benchmark on a model with many small blocks vs. few large blocks; verify the
        heuristic selects the faster mode in each case.

### 7.2. Memory profiling

- **Action:** Profile the specific known OOM risks: `build_vector(..., root=0)` and
  `build_dense_matrix` (`Allreduce` of a dense `size × size` matrix), and the array
  path's `wp_global` allocation of shape `(global_N, p)` on every rank.
- **Verification:**
  - [ ] `test_array_lanczos_mpi_memory` — (from `blocklanczos_blas_acceleration.md`)
        per-rank peak allocation scales like `local_N`, not `global_N`.
  - [ ] Assert via trace logs that `ManyBodyState` objects are evenly distributed,
        preventing rank-0 OOM on large multi-tOp runs.
