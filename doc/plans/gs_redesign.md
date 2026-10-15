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

`CONTINUATION_CHECK_EVERY = 8`: NiO 7-bath, production settings, one rank, time inside the eigensolver
(block matvecs / TRLM seconds): cadence 1: 2677 / 40.7, 2: 2682 / 37.2, 4: 2698 / 36.8, 8: 2721 / 35.1,
12: 2744 / 35.1, 16: 2769 / 35.7, 24: 2804 / 36.3, off: 2945 / 39.5. Checking every block is slower than
never checking (the `eigh` outweighs the block it saves); the minimum is a plateau at 8-12.
`sigma_static` was bit-identical to the unchecked run at every cadence.

## Not done / open

* The SMO archive tail cycle was not benchmarked, and the pad-state mechanism above was not confirmed
  with a trace.
* Walk semantics (3^k diagonal probe, legacy `dN_gs` fallback) are untouched: changing them changes which
  sector is found and needs its own oracle.
