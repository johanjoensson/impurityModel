# Ground-state redesign: one certified window solve per basis

Decision log for the rewrite of the CIPSI ground-state determination. Code comments state contracts;
the reasoning and the measurements live here.

## Why

The ground-state path had grown guards for problems that no longer existed and paid for them on every
call:

* A completeness probe (`_complete_low_spectrum`) ran after every eigensolve that had a thermal cut.
  It locked only the states at or below the boundary, so the lowest Ritz value of the complement was
  the first state *above* the boundary and the cheap "clear" certificate rarely fired; a full locked
  TRLM then re-converged states the solve already held. (Mechanism inferred from the code; the
  `theta0 - boundary` trace that would confirm it was never added before the probe was deleted.)
* `calc_energy` passed `tau=0, max_energy=0, num_wanted=1` (c5415f01). That put the probe on every walk
  cycle, made the walk's `tau/100` convention dead (`energy_cut(0) == 0`), and disabled the cold retry
  for sector crossings (it was gated on `need_more or max_energy is None`).
* The same determinant set was solved repeatedly: `_solve_sector_core`'s follow-up
  `get_eigenvectors` after `expand`, the walk's rebuild of its own winner, a cold first solve in the
  refinement, and the `gs_thermal` widening loop.

## Design

`eigensolvers.solve_window` returns every eigenstate within `max_energy` of the lowest, plus one
bracket state above, certified complete:

* TRLM *window mode* (`trlm.Window`): retains as many Ritz vectors as the window needs (grows
  `nkeep` at a restart head, nothing already built is re-solved) and converges the in-window pairs plus
  one bracket pair above the edge. A degenerate group straddling the edge is absorbed whole.
* Start block: the warm columns plus `R` generic hash-seeded columns (the all-positive cold vector and
  zero-centred random columns). They are rank-independent and full-support, so every charge sector is
  reachable from the start block; this retires the cold retry.
* Certificate: Krylov space `K(H, Q0)` meets the eigenspace of a level in `range(P Q0)`, of dimension
  `min(g, rank(P W) + R)` for `g` copies, `W` the warm columns. `k` converged copies certify the level
  when `k < rank(F^H W) + R`. The rank counts singular values above `max(1e-6, 100 tol)`: `F` is only
  `tol`-accurate, and counting noise would over-certify while undercounting only triggers the fallback.
  Probabilistic (the random columns are generic); assumes the copies of an exactly degenerate level
  converge together.
* Fallback, only for a saturated level: lock the converged in-window vectors, restart from `2 k_max`
  fresh random columns in the complement with the same absolute window (`Window.e0`), certify what
  turns up there with `k_new < R'`. Bounded rounds.
* `R` is adaptive: `max(4, max_multiplicity + 1)`, remembered across CIPSI cycles on the solver.
* An `EigenResult` carries the `Basis.generation` and window it is valid for. `generation` is bumped on
  every change to the determinant set (a capped cycle can turn determinants over at constant size, so
  size is not a key).
* Kernel: the `locked` projection in `lanczos_step` now runs *after* the full reorthogonalization.
  The stored basis carries an O(eps) locked component that the reorthogonalization puts back.

Ground-state flow: `expand`'s last solve is read by `get_eigenvectors(reuse_last=True)`; the walk
returns its winning `(basis, solver)`; `solve_ground_state` continues that solver warm.

## Measurements

One `solve_ground_state` (walk + refinement), NiO workload, 1 rank, `OPENBLAS_NUM_THREADS=1`, against
the tree at bc49879d with the same compiled kernels:

| workload | old | new | E0 |
|---|---|---|---|
| 10 baths, cap 5,000 | 17.1 s | 2.9 s | -62.67772709 (identical) |
| 20 baths, cap 20,000 | 45.7 s | 13.0 s | -130.34757357 (identical) |
| 50 baths, cap 30,000 | 186.4 s | 56.4 s | -208.69918153 (identical) |

SMO cubic archive (`impmod_tests/SMO/cubic`, cluster `Mn 1`), `calc_gs` alone (`calc_selfenergy` aborted after
it), 1 rank, cap 20,000 (the basis converges at 7,969 determinants, so a larger cap changes nothing):

| | old | new |
|---|---|---|
| `calc_gs` wall | 51.8 s | 21.4 s |
| time in `get_eigenvectors` | 48.9 s | 18.5 s |
| `get_eigenvectors` calls | 76 | 69 (7 of them reads of a previous solve) |
| E0 | -10.00697056 | -10.00697056 |
| thermal states / basis | 6 / 7,969 | 6 / 7,969 |

What is left in the new run: 53 real eigensolves (11.7 s), CIPSI selection 1.8 s, the remainder outside the
solver. The remaining eigensolve time is spread over the cycles (0.05-0.7 s each, the last refinement cycles
on ~8,000 determinants costing most); no single dominant probe is left.

`CONTINUATION_CHECK_EVERY = 8`: NiO 7-bath, production settings, one rank, time inside the eigensolver
(block matvecs / TRLM seconds): cadence 1: 2677 / 40.7, 2: 2682 / 37.2, 4: 2698 / 36.8, 8: 2721 / 35.1,
12: 2744 / 35.1, 16: 2769 / 35.7, 24: 2804 / 36.3, off: 2945 / 39.5. Checking every block is slower than
never checking (the `eigh` outweighs the block it saves); the minimum is a plateau at 8-12.
`sigma_static` was bit-identical to the unchecked run at every cadence.

## Not done / open

* The pad-state mechanism of the old probe was never confirmed with a trace (the probe was deleted first).
* Goldens (E0 / sigma_static / DC mu across the example tests) were not recorded as JSON; equivalence was
  checked on the NiO and SMO benchmarks above and by the existing test suite.
* Walk semantics (3^k diagonal probe, legacy `dN_gs` fallback) are untouched by design: changing them changes which
  sector is found and needs its own oracle (round 2 below changed how a trial is *solved*, not which trials run).

## Round 2: two manifold kinds (2026-10)

### Why

Round 1 left three things wrong or wasteful:

* `solve_ground_state` continued the walk's `(basis, solver)`. That basis was expanded for the ground
  *level* alone (`calc_energy` ran at `tau = 0`), so a low-lying state outside the level sat at seed
  quality in it: never a reference, its determinants never selected. The thermal window was solved on a
  basis never built for it.
* Every double-counting sector solve ran the full thermal window and kept `min(es)`; on SMO the block
  reached p = 171 and the selection round's memory and the matvec cost scale with p.
* The walk ran window mode at width 0, absorbed the whole degenerate multiplet and ran the completeness
  certificate for an energy it only compares.

### Design

Two request types (`eigensolvers.Lowest`, `eigensolvers.Within`; `Within.thermal(tau)`):

* **`Lowest`** -- one converged pair (`solve_lowest`, count-mode TRLM, warm column + the hash vector), no claim
  about the level's degeneracy. The occupation walk and every DC sector energy.
* **`Within(max_energy)`** -- every state in the window, certified complete (round 1's `solve_window`). The
  production thermal ground state, solved **from the seed basis at the winning sector**; the walk's basis is
  never reused (`walk_solver`, `best_entry` and the cap-restore plumbing are gone).

`CIPSISolver.get_eigenvectors/expand/truncate_initial` take the request; `de0_max` comes from it, never from
`basis.tau`; the dense and Krylov branches cut with the same `_energy_cut_indices` (the dense branch used to
bisect a degenerate group at the window edge). `groundstate.solve_sector` is the single sector body;
`calc_energy` is its `Lowest` head and returns `(e0, truncation_report)`; `find_ground_state_sector` replaces
`find_ground_state_basis` + `walk_to_ground_state_sector` (the `sqrt(slaterWeightMin)` convention lives there,
the dead `tau/100` is gone); `solve_ground_state` returns a `GroundState`. The `ground_state_manifold` flag
and its record fields (`manifold_spread`, `manifold_states`) are removed: a sector is solved for its lowest
state, so the reported occupation is that state's (the one Hellmann-Feynman relates to `dE/dmu`).

### Measurements

NiO workload, 1 rank, `OPENBLAS_NUM_THREADS=1`, `solve_ground_state` (walk + production solve), against the tree
just before round 2 (57f55cad) with the same compiled kernels. The October table above is not the baseline: later
commits on the branch moved E0 in the 4th decimal.

| workload | wall old | wall new | E0 (both) | dets | max residual new |
|---|---|---|---|---|---|
| 10 baths, cap 5,000 | 2.0 s | 4.0 s | -62.67740635 | 1076 | 2.3e-14 |
| 20 baths, cap 20,000 | 6.3 s | 7.4 s | -130.34574436 | 3760 | 2.3e-13 |
| 50 baths, cap 30,000 | 43.0 s | 44.8 s | -208.69462591 | 23910 | 3.0e-13 |

E0, basis size and the returned manifold agree with the old tree; the production solve is a full expansion from
the seed, which is the (small) price of building the basis for the thermal window rather than inheriting one
built for the ground level. These workloads' thermal window holds a single state, so they do not exercise the
completeness claim.

### Not done / open

* No small model reproduces the old incompleteness: an uncapped solve saturates the whole sector, so a ground-level
  -only walk basis finds the same states. `test_thermal_manifold_complete.py` pins the contract against a dense-ED
  oracle and the structure (walk never runs window mode; the winner is rebuilt), not the old failure.
* A state outside the window *in the seed basis* still cannot be followed (intrinsic to CIPSI).
* Not measured here: SMO `calc_gs` and the SMO `fixed_gap_dc` replay (the data lives on the cluster side); the
  expected effect is the DC sector blocks shrinking from p up to 171 to p <= 2.
* `impmod_interface`'s `_parse_dc_line` still accepts a `ground_state_manifold` token; a follow-up in that repo.
