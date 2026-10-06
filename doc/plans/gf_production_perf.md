# Green's-function performance for production DFT+DMFT (SrMnO3, CrI3)

Status 2026-10-06. Branch `gf-production-perf`: Steps 1-3 shipped, measured locally; the cluster read-out
(Step 6 kit) is pending.

## The problem

Two production runs with fixed-gap DC on `ground_state_manifold` were too slow for a converged DFT+DMFT loop.
DC is cheap; the interacting Green's function dominates.

| | SrMnO3 cubic (Arrhenius, 128 ranks) | CrI3 FM (Dardel, 256 ranks) |
|---|---|---|
| DC / GS | 2 min / 7 min (277,966 dets) | 32 min / 3 min (613,751 dets) |
| GF | killed by the 24 h wall clock after 14 h | 13.8 h |
| units | 40 on 32 colours (3-6 ranks) | 12 on 10 colours (20-40 ranks) |
| retained states | 10: 0.4998, 0.4998, then 8 x 5.7-6.8e-5 (5.0e-4 in total) | 2: Sz = +-1.65, split 1e-4 eV |
| expensive units | all 20 removal units (~42 s/block on 6 ranks) | removal units, frozen at the 10M cap |

Units are group-major (block0-add, block0-rem, block1-add, ...) x eigenstates. Every unit still running when SrMnO3
was killed was a removal unit.

## Root causes

1. **One tolerance for both axes.** The monitor drives every unit to a relative change of `G` below
   `max(slaterWeightMin**2, 1e-9)` on every requested axis. The real axis at broadening `eim` sets the Lanczos depth.
   RSPt's self-consistency uses only the Matsubara self-energy; the real-axis one is DOS/spectra output.
2. **Low-weight thermal states.** The energy window `-tau*ln(1e-4)` keeps states down to weight ~1e-4, and each
   costs a full set of units.
3. **Static, side-blind scheduling.** `unit_cost_weights = seed_mass x width` cannot tell a removal unit from an
   addition unit. SrMnO3's 2-unit colours finished in ~2 h and then sat idle for ~12 h.
4. **After the cap freezes the basis, the matvec stays a full apply.** See the profile below.

## Measured locally (SrMnO3 archive `Mn 1`, -n 3, one rank per colour)

Harness: `RUN_REAL_WORKLOAD_BENCH=1 WORKLOAD_H5=... CAP=... PHASES=1 VERBOSITY=2 BLOCKLANCZOS_PROFILE=1
pytest -m benchmark src/impurityModel/test/gf/test_gf_real_workload.py`. Note that the harness does not replay the
archive's excitation budget, so absolute sizes differ from production; the comparisons are like for like.

**Depth is the real axis.** At cap 1e5, Matsubara-only units converge in 26-42 blocks. The two full-mesh units that
finished needed 506 and 1327.

**Per-axis tolerance, cap 2e4.** The cap-2e4 ground state is truncated (residual PT2 4e-4), so the real-axis
self-energy is acausal at every tolerance. Runs used `SIGMA_CAUSALITY_TOL=2` to compare tolerances against each other.

| setting | blocks/unit (rank 0) | GF phase wall | Matsubara Sigma | real Sigma | sigma_static |
|---|---|---|---|---|---|
| default (1e-9 both) | 604-1625 | 1663 s | reference | reference | reference |
| `gf_real_tol 1e-6` | 1.6-1.8x fewer | 1270 s | **bit-identical** | 6.8e-6 | bit-identical |
| `gf_real_tol 1e-4` | 2.5-3.3x fewer | 893 s | **bit-identical** | 8.7e-4 | bit-identical |
| `gf_min_weight 1e-3` (drops 2 of 4 states, weight 1.0e-4) | as default | 1082 s | 1.5e-5 | 5.5e-4 | 7.3e-6 |

Errors are max-relative against the reference. The Matsubara self-energy is bit-identical under a looser real-axis
tolerance: the Matsubara axis converges within ~30-40 blocks, and deeper continued-fraction levels change `G(iw_n)`
below one ulp.

**Profile of a unit** (`BLOCKLANCZOS_PROFILE=1`, cap 2e4, one rank): `matvec_apply` is 95% of the unit time
(0.35-0.4 s per apply on 20k frozen determinants), recurrence 2-3%, TSQR 1-2%, monitor 1-2%.

**Moments.** `get_greens_function_moments` took 890 s against a GF phase of 1092 s at cap 1e5 (Matsubara-only), and
set the run's RSS peak (3.6 GiB vs 0.5 GiB for the GF). At cap 2e4 it took 22-54 s. It applies `(H - E)` twice to
`2 x n_corr` seeds per retained state, unrestricted, into the `N +- 1` sectors.

## Shipped

| commit | what |
|---|---|
| `03a1d9a2` | per-unit `wall=` on the `-vv` unit line; per-unit `BLOCKLANCZOS_PROFILE` split |
| `062c6e73` | `gf_tol` / `gf_real_tol` (SolverOptions, TOML, CLI, archive; `GF_TOL` / `GF_REAL_TOL` knobs); default bit-identical |
| `0e92e7b8` | `gf_min_weight`: drops whole degenerate manifolds below the weight from GF, Sigma, moments; ground manifold kept |
| impmod_interface `31a7703` | `gf_tol`, `gf_real_tol`, `gf_min_weight` on the RSPt solver line, archived, in the header |

## Recommendation for the production inputs

* Both systems: `gf_real_tol 1e-4` on the solver line. The Matsubara self-energy that drives the loop is unchanged;
  the real-axis one moves by ~1e-3 relative, well inside the DOS plot resolution. Use `1e-6` for a final,
  publication-quality spectrum iteration.
* SrMnO3: `gf_min_weight 1e-3`. It drops the eight excited states (normalised weight 5.7-6.8e-5 each, 5.0e-4 in
  total; commit 0e92e7b8's message quotes the unnormalised 1.4e-4) -- 40 -> 8 units, 20 -> 4 removal units.
  Matsubara Sigma moves by ~1e-5 relative per 1e-4 of dropped weight, far below `sigma_acc = 1e-2`.
* SrMnO3 cap: production ran `truncation_threshold auto`, which sized the GF cap (17.9M) for the smallest colour
  (3 ranks). With 8 units each colour gets ~16 ranks and `auto` would raise the cap several-fold. Pin
  `truncation_threshold 1.8e7` for a like-for-like comparison; raising it is a separate, deliberate decision.
* CrI3: `gf_min_weight` does not apply (both states weigh ~0.5). The near-degenerate +-Sz pair is a physics issue to
  look at separately (the FM moment is averaged out of the impurity).

## Next levers (not yet implemented), in expected order of payoff

1. **Frozen-basis CSR for capped units -- SHIPPED (`GF_FROZEN_CSR`, default on).** Once `_CappedBasisProxy`
   freezes, every step still applied H to all retained rows and dropped the image outside P (the apply is 95% of a
   unit's time; the dropped share of the image is only ~40% -- in-P nnz 11-12/row against a total fan-out of 16-28 --
   so the cost is the apply's per-row overhead, not the discarded rows). The whole capped recurrence is the exact
   Lanczos of `P H P` from the seeds, so `block_Green_sparse` now stops at the freeze (kernel status `"frozen"`),
   adopts the retained mask as a `Basis` (`Basis.clone_from_keys`) and *restarts* the recurrence on the distributed
   CSR with the array kernel -- no sparse-kernel state is carried over. Not taken for a memory-guard freeze, a CSR
   that does not fit the budget, or a `krylov_dtype` store. The distributed `build_sparse_matrix` now resolves its
   bras in batches, so its transient is bounded at multi-million determinants.

   Measured (SrMnO3 archive, cap 2e4, `-n 3`, `gf_real_tol 1e-4`, `gf_min_weight 1e-3`, `SIGMA_CAUSALITY_TOL=2`):
   GF phase **167.2 -> 9.0 s** (18.6x), units 21-70 s -> 0.8-2.6 s (build included), peak RSS unchanged (1.2 GiB).
   Matsubara Sigma agrees to 3.1e-11; real-axis Sigma to 1.1e-3, which is the `gf_real_tol 1e-4` stopping noise
   (block counts differ by a few per unit; that tolerance alone moves real Sigma by 8.7e-4, see above). CSR memory
   ~290 B per retained determinant. On one 3-rank colour for all units (distributed build + graph-exchange SpMV):
   GF phase 278.3 -> 14.1 s, Matsubara Sigma to 5.3e-11. The real-axis gap comes from the archive's `reort none`
   (orthogonality loss amplifies the two paths' different summation order), its `slaterWeightMin 1.5e-8` (pruned
   inside the sparse apply, absent from the exact CSR) and a 1e-4 monitor stopping a few blocks apart. The fallback's
   first block budget is bounded (`_CSR_INITIAL_BLOCKS`, or 4x the blocks reached at the freeze): the array kernel
   preallocates its coefficient buffers at the budget, and `ceil(N/p)` is GiBs at 10M. CrI3's removal units (frozen
   at 10M) are the cluster read-out still to do.
2. **Moments.** Cheap: evaluate only what is needed (`<s1|(H-e)|s1>` needs `H s1` restricted to `supp(s1)`, not the
   full fan-out), and fewer states via `gf_min_weight`. Measure at production GS size first.
3. **Scheduling (plan Step 4).** History-based unit costs (previous DMFT iteration's n_blocks x size per
   (block, side, state)). Re-measure after 1-2 change the unit mix.
4. `GF_APPLY_ROW_CHUNKS=1` vs 4 at GF colour rank counts (cluster read-out).
