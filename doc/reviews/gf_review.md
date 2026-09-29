# Green's-function execution paths: review ledger

This file is the running record of the 2026-09 review of the GF execution paths. The paths covered are:

- `selfenergy` → `greens_function` → `gf_units` / `gf_solvers` / `gf_primitives` / `gf_convergence` / `gf_shift_recycling` → the Cython Lanczos, BiCGSTAB and GMRES kernels;
- `spectra`, `rixs` and `susceptibility`;
- the supporting layers `basis_split`, `basis_restrictions` and `block_structure`.

Every row starts as a hypothesis found by reading the code, and only a test settles it.

**Verdicts:**
- `open`: not yet tested.
- `confirmed`: a test reproduces it.
- `refuted`: a test shows the claim is wrong. It is kept here so nobody re-derives it.
- `fixed`: a commit resolves it, and the test is now a regression guard.

**Severity:**
- `!!`: a silent wrong result in production is plausible.
- `!`: a crash, hang or leak.
- `·`: quality or performance only.

## Safety net

`src/impurityModel/test/gf/test_gf_branch_matrix.py` runs a pairwise covering array over every path-selecting switch of the GF engine and compares each cell against an exact oracle. The full cartesian product is opt-in with `-m branch_matrix_full` and takes about 10 min serially.

- **Switches covered:** `gf_method`, `sparse`, `GF_EIGENSTATE_GROUP`, `reort`, the mesh axes, the dense and operator array branches, and serial versus split execution.
- **Oracle:** `support/gf_branch_oracle.py`, a sector-wise dense Lehmann sum that shares no code with the pipeline. It is validated against the sum rule `zG → 1` and the analytic single-level limit.
- **Model:** its intra-block hopping is complex, so `G_ab ≠ G_ba` and a transpose error is visible. With real hopping, an injected dropped transpose went undetected.

The discrimination check (Phase 0) injects six bugs into `greens_function.py`: a dropped transpose (iw and w), a flipped removal δ, a 0.1% error in Z, the wrong eigenstate's seed slice, and a wrong Boltzmann weight on the bicgstab path. Each one turns cells red, or hangs (R1).

Known failures are pinned as xfails that name their ledger row. They are strict wherever the failure is a property of the cell, so a fix flips them and forces the pin to be removed. The M1 pins are non-strict: M1 needs a color with more than one rank, and the packing, not the cell, decides that.

## Ledger

| id | sev | claim | where | test | verdict |
|---|---|---|---|---|---|
| C11 | !! | Array Lanczos kernel caps `max_iter=ceil(N/p)`, which assumes a constant block width. Deflation shrinks it (widths `[4,4,2,2,2,2,2]` span 20 of 28 dims), so the recurrence stops at `max_iter` with a truncated continued fraction and only a warning. Measured G error up to 0.07 (real axis, δ=0.2) with stacked eigenstates, and 3.6e-5 on a 3-operator tensor. Hits RIXS R2 (always the array `block_Green`) and `sparse_green=False`. | `gf_solvers.py:237,292` | branch matrix `lanczos/sparse=False/group=2/model=1`; `test_calc_greens_function_with_offdiag_matches_oracle` | **fixed** (Phase 3): `block_green_impl` doubles `max_iter` (capped at N) and reruns while the kernel stops on `max_iter`; the branch-matrix cells and the tensor oracle test pass |
| C1 | ! (latent) | Particle-hole equivalence: `_particle_hole_blocks_matrix` scans `blocks[i:]`, so a block with Re M≈0 is its own particle-hole partner and gets overwritten with `-conj(G)` after the causality check. Correct pairs are also wrong on the real axis (no ω reversal). The detector never checks that U is particle-hole invariant. The moment round-trip flips the sign of Σ₁. | `block_structure.py:526`, `greens_function.py:122-134`, `selfenergy.py:393` | `test/gf/test_block_structure_particle_hole.py` (4 strict xfails; the Matsubara image passes, as predicted) | **confirmed**: self-inclusion, the overwritten representative block, the real-axis image and the Σ₁ sign all fail. Latent: every production archive in `impmod_tests` (NiO, AFM-NiO, FCC Ni, BCC Fe) prints empty particle-hole lists |
| C10 | !! | Identical-block detection uses only `h_imp + V†V` and ignores `h_bath`. `reconcile_block_structure_with_interaction` is never wired in. | `symmetries.py:776`, `solver_basis.py:197` | `test/gf/test_block_structure_bath_equivalence.py` | **confirmed**: orbitals with baths at -1.0 and +1.5 are declared identical and share one G. Not active on NiO 8-bath (its identical pairs also agree on `V† h_bathᵏ V` for k=1-3 exactly). The AFM-NiO archive lacks `H solver` (crashed early), so checking it needs a rebuild from `Ham-NiUp.inp`. **fixed** (Phase 3): `impurity_block_structure` also probes the bath moments `V†h_bathᵏV` (k=1,2), whose couplings merge blocks and whose values must agree for an equivalence, and it now runs `reconcile_block_structure_with_interaction` on the impurity two-body tensor. Production block structures are unchanged (NiO 8-bath: 2 inequivalent blocks; FCC Ni: 4) |
| C2 | !! | The near-conduction upper bound subtracts the counts of far-empty and far-filled orbitals, which can pin it to (0,0) when `con_change` is set (spectra, RIXS, self-energy with `dN`). | `basis_restrictions.py:476` | `test/restrictions/test_excited_window_conduction_bound.py` | **confirmed**: on a 7-site conduction chain the near pair {1,2} is pinned to (0,0), excluding a determinant of the ground-state basis. **fixed** (Phase 3): the near-conduction upper bound stays `max_con` (a subset of the group). **Output change:** spectra, RIXS, and self-energies with `dN`, on chains long enough to have freeze-eligible conduction orbitals, now keep the near conduction bath |
| C3 | !! | Masks set on the shared `hOp` are sticky and are set only when non-None, so the later moments (M2, M3 → Σ₁, Σ₂) run as P·H. The retry `calc_gs` inherits a stale weighted mask. | `greens_function.py:1229`, `cipsi_solver.py:1306`, `selfenergy.py:389` | `test/gf/test_gf_leaves_hamiltonian_unmasked.py` | **confirmed**: after one GF with a dN=1 window, M₂ and M₃ from the same `h` are off by up to 18% (M₁ unaffected). With the default dN=None the leftover mask is the chain window, which is looser but still a mask on a quantity documented as exact. **fixed** (Phase 3): every mask site sets unconditionally (`None` clears): `_block_green_group`, `CIPSISolver` (three sites), `dc_frozen`. `get_greens_function_moments` clears both masks before applying H. **Output change:** `sigma_moment_1`/`_2` move wherever a window used to bind (including the ground-state/chain window `calc_gs` left on `h`) |
| C4 | · | `None` means unrestricted, inherit, or empty depending on the helper, and nothing checks for `lo > hi`. | `gf_units.py:710`, `manybody_basis.py:373`, `greens_function.py:1144` | `test/restrictions/test_window_none_semantics.py` | **confirmed**: an unrestricted unit clone inherits the ground-state window, and an infeasible (0,1)∧(2,3) intersection passes silently. Reachable only with `GF_EIGENSTATE_GROUP>1` (union over disjoint keys) |
| C5 | ! | Seeds are cut by the ensemble window while the recurrence runs under the per-state window; P·H is non-Hermitian on seed rows outside it. RIXS seeds are unrestricted. | `greens_function.py:442`, `gf_units.py:394`, `rixs.py:546` |  | open: needs a long-chain oracle comparison; lands with the Phase 5b window work |
| C6 | ! | Excited chain caps are not widened by dN, and the budget grows only +1; neither was validated for N±1. | `basis_restrictions.py:292-322,500-508` |  | open: measured in Phase 5b against an unrestricted oracle |
| C7 | !! | Sector restrictions come from rounded averages without a definite-charge check. RIXS uses the `in_ops[0]` sector for every component. A vector mixing two sectors (a degenerate multiplet has no preferred basis, and the eigensolver's rotation depends on the rank count) is confined to a sector it does not have. Consumers are `calc_spectra`, the tensor path and RIXS. The self-energy does not use it, despite `gf_sector_restrictions`' docstring. | `symmetries.py:1202`, `rixs.py:552` | `test/spectra/test_sector_needs_definite_charge.py` | **confirmed**: an equal mix of the n₀ and n₁ sectors is confined to (n₀=1, n₁=0); the only seed determinant has n₁=1, so the whole seed is pruned and the spectrum is 0. **fixed** (Phase 3): `symmetries.definite_conserved_charges` checks each charge's variance (one Allreduce) and returns `None` for a mixed state; `calc_spectra`/the tensor path and RIXS then use the plain occupation window. RIXS confines only when every in-component gives the same sector |
| C8 | · | `simulate_spectra` ignores `restrictions` and `dN`. RIXS uses occ cutoff 1e-6 vs 1e-12. `dense_cutoff` is ignored (literal 500). A float `reort` passes untranslated. | `spectra.py:138,145`, `rixs.py:518`, `gf_solvers.py:186`, `BlockLanczosCore.pyx:180` |  | **confirmed by reading**: `simulate_spectra`'s `restrictions`/`dN` appear only in the signature and docstring; RIXS passes no `cutoff=` (the builder default is 1e-6); `dense_cutoff` never reaches the GF; `resolve_reort` returns any non-string unchanged. Tests land with the fixes |
| C9 | · | `calc_thermally_averaged_G` returns shape `(n_w,0,0)` on the empty path. | `gf_primitives.py:273` |  | **refuted in effect**: `len(alphas)==0` is the number of thermal states, which no caller can make 0; a dead branch. Remove it in Phase 5 |
| M1 | !! | The array operator branch has a `LinearOperator (N, N_local)` plus `Reduce` to root, and fails on colors with more than one rank (RIXS R3). | `gf_solvers.py:244-275` | `test_block_green_array_multirank.py` (strict xfail); branch matrix `model=2/comm=world` | **confirmed** (pre-existing) |
| M2 | ! | Rank-local break decisions (converged, invariant subspace, deflation width) come from replicated floats with no consensus. | `_lanczos_step.pxi:864`, `BlockLanczosArray.pyx:980-998` |  | **confirmed by reading** (`BlockLanczosArray.pyx:980-998`, `_lanczos_step.pxi:864`); triggering it needs a fault-injection hook in the kernel, so the test lands with the Phase 3 fix |
| M3 | ! | `_graph_comm_cache` is keyed by `id(parent)` and never evicted, leaking dist-graph comms per Clone. | `mpi_comm.py:48-101` | `test/mpi_infra/test_graph_comm_cache_leak.py` (skips at -n 1) | **confirmed**: 5 freed cloned communicators leave 5 cache entries (parent pinned plus graphs) |
| M4 | ! | The split-comm free sits outside the `finally`, and Clone frees happen only on the normal path. | `gf_units.py:648`, `gf_solvers.py:852`, `rixs.py:636` | `test/gf/test_gf_split_comm_freed_on_error.py` (skips at -n 1) | **confirmed**: a kernel raising on every rank leaves the split communicator unfreed |
| M5 | · | RIXS `solver_stats` are overcounted by ranks-per-color. | `rixs.py:246` |  | **confirmed by reading**: `eval_out` increments on every rank of a color (`rixs.py:805`, `:960`) and `_report_rixs_solver_stats` SUM-reduces over the whole comm |
| S2 | · | `gf_method="cipsi"` (resolvent-targeted CIPSI per-frequency selection) matched, but did not beat, the freeze-growth cap at equal budget on NiO, at 1.5-2x the wall cost (`doc/plans/gf_cipsi_frequency_truncation.md`, Verdict). Its one unique asset was a *measured* boundary residual, a per-run upper bound on the G error from truncation. | `gf_solvers.block_Green_cipsi` | plan verdict (measured 2026-08) | **retired** (Phase 1b): kernel, `CIPSISolver.select_at`, `GF_CIPSI_*`, `check_cipsi_boundary` removed. The boundary-residual idea becomes a requirement on the Phase 5b truncation diagnostic. |
| S3 | · | The pairwise operator split (`GF_OPERATOR_SPLIT`: scalar continued fractions per diagonal seed plus two polarization seeds per off-diagonal pair, recombined by the polarization identity) was an opt-in load-balancing mode. It multiplied the Krylov work (no shared subspace), disabled the sum-rule, convergence and integrated-weight diagnostics, and cannot serve the improved-estimator seam, which needs whole-block recurrences. | `gf_units.enumerate_gf_units`, `gf_primitives.PairwiseGF` | — | **retired** (Phase 1c) by your decision: code, knob and tests removed. `config.RETIRED_KNOBS` now makes every retired knob fail loudly (a warning at GF entry, an error in `[environment]`). |
| R1 | ! | A causality-violating recurrence (injected: removal side run with +δ) never satisfies the convergence monitor, and the sparse resume loop doubles its budget with no upper bound, so the unit runs until the Krylov space closes (>150 s on a 792-det sector) instead of failing fast. | `gf_solvers.py:443-480` | bug-injection run (Phase 0) | observed; decide in Phase 3 (bounded budget + diagnostic) |
| S1 | · | `gf_method="sliced"` misses the exact G by about 1e-5 relative on the real axis (above the 2e-6 cell tolerance). The path is being retired. | `greens_function.py:930` | branch matrix `method=sliced/model=1` | **retired** (Phase 1a): driver, `GF_SLICE_*`, `bra_seeds`, the slicing diagnostic and memory branch removed; the Chebyshev filter module stays |

## Performance baseline

Recorded 2026-09-29 at commit `6386b6a`. Phase 1 only removed opt-in paths, so the default
Lanczos path is unchanged from `aba6908`. Harness: `test/gf/test_gf_real_workload.py`
(`RUN_REAL_WORKLOAD_BENCH=1 -m benchmark`), `OPENBLAS_NUM_THREADS=1`, archive settings (auto cap,
`reort=none`, full meshes of 2048 Matsubara and 1001 real points).

| workload | ranks | wall | peak RSS / rank | GF layout |
|---|---|---|---|---|
| NiO 8 bath/orbital (`impmod_tests/NiO/impmod/8_BathStates_peeledGeometry_noneReorthonormalization_3_processors_`, 88 spin-orbitals) | 3 | 273.3 s | 734 / 373 / 374 MiB | 12 units on 3 one-rank colors (5/3/4); removal units 22-24k dets, ~948 Lanczos blocks; addition units 0-20 dets |

The 15-bath NiO replay is too heavy for a per-phase loop: its ground state alone took more than
10 min at `-n 1`, dominated by X1 below.

## Performance and parallelism

Each of these is adopted only with a measurement against the Phase 0 baseline.

| id | claim | verdict |
|---|---|---|
| N1 | Pole representation of T via `eig_banded`, for evaluation and the future DSR tier. | open |
| N2 | The convergence monitor rebuilds the continued fraction, O(k²) in total. | open |
| N3 | `hyb` solves an n_bath system per frequency. | open |
| N4 | The array expansion restarts Lanczos from scratch. | open |
| N5 | Spectra run the band-wide monitor (`eval_meshes=None`). | open |
| N6 | The thermal retry recomputes the whole GF. | open |
| P1 | The GS basis is replicated into every color although kernels clone only the seed support. | open |
| P2 | Static LPT packing on seed-mass estimates; ranks ∝ mass. | open |
| P3 | Serial rank-0 work (mesh evaluation, diagnostics, Dyson); `simulate_spectra` re-splits per spectrum. | open |

## Out of scope, found along the way

| id | claim | evidence | verdict |
|---|---|---|---|
| X1 | The ground-state TRLM continuation loop is dominated by bookkeeping, not the matvec. About 53% of samples go to full CGS2 against the whole retained basis (`trlm.py:528,535-536`, unconditional whatever `reort` says). About 22% go to re-copying the growing Krylov basis on every block (`concat_cols` → `np.concatenate`, `trlm.py:585`), which is quadratic in the Krylov dimension. The sparse matvec is about 2%. | py-spy, 30 s / 1484 samples, NiO 15-bath replay, cap 50k, `-n 1` | observed; preallocating `Q_basis` is a candidate follow-up (ground state, not GF) |
