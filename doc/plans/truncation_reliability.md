# truncation_threshold: semantics, reliability, and HPC sizing heuristics

`truncation_threshold` caps the **global number of Slater determinants** a many-body
basis may hold: "keep as many determinants as fit in RAM; when more are generated,
keep only the currently important ones." This document records what the cap does on
each solver path, why it is safe under every reorthogonalization mode, the empirical
reliability sweep, and how to choose the threshold on a cluster.

## What the cap does per path

| Path | Mechanism | Importance criterion |
|---|---|---|
| Ground state (CIPSI, `cipsi_solver.py`) | on overflow: `basis.clear()` + re-add, cutoff raised ×10 until it fits | eigenvector amplitude |
| GF sparse kernel (`block_Green_sparse`, production self-energy) | `_CappedBasisProxy`: admit-while-under-cap → one importance-ranked boundary admission → freeze + project (`keep_rows`) | residual max column \|amp\|² at the overflow step |
| GF array kernel (`block_Green`, spectra XAS/RIXS) | probe expansion stops within one H-application batch of the cap | order of discovery |

`None` (the driver default) resolves the threshold from available per-rank RAM via
`impurityModel.ed.memory_estimate` (collective probe: `MemAvailable` ÷ ranks-per-node,
min-reduced); `np.inf` disables capping. The `Basis` container itself never truncates.

## Why the capped GF is reliable (the reort contract)

Once the sparse-kernel cap freezes the retained determinant set, every later residual
is projected onto it (diagonal projector `P`). Two facts make this safe:

1. Every previously accepted Krylov block has support inside the retained set, so
   `⟨Q_j, P wp⟩ = ⟨Q_j, wp⟩` — the projection cannot damage orthogonality against the
   existing Krylov basis, in exact arithmetic *and* in floating point (it only zeroes
   rows the earlier blocks never had).
2. From the freeze on, the recurrence is an **exact block Lanczos of the Hermitian
   projected operator `P H P`**: the block-tridiagonal `T` stays Hermitian, the
   continued fraction stays causal, moments up to the freeze step are exact w.r.t.
   `H`, and the recurrence terminates naturally as `invariant_subspace` within
   `ceil(cap / block_width)` blocks.

Consequences per reort mode:

- **NONE** (GF default): valid; only ~3 live blocks exist and the importance measure
  is the current residual, so the cap adds **zero** memory overhead.
- **FULL / PERIODIC**: valid; the `SparseKrylovDense` rows are bounded by the retained
  set and the store never needs removal (removal is unsupported there — this design
  never removes).
- **PARTIAL / SELECTIVE**: valid post-freeze (exact recurrence of `PHP`; the
  Paige–Simon estimator sees a genuine Lanczos process). Confirmed empirically by the
  oracle tests and the sweep below.

The **oracle test** (`test_gf_truncation.py`): the capped GF equals the dense resolvent
of `PHP` on the retained set to ~1e-9, for reort ∈ {none, full, partial}, serially and
at 2–3 ranks including ranks that retain zero rows.

Ground-state path: CIPSI truncation happens only *between* fixed-basis Lanczos solves
(after `basis.clear()`), so a truncation never coexists with a live Krylov store —
reort interaction is safe by construction. The historical `size>0`/empty-`local_basis`
desync was fixed in `5c2b37e`.

What a hit cap costs physically: spectral weight reachable only through the discarded
determinants is missing (the result is exact on the retained subspace — still causal
and sum-rule-consistent on it). The `basis_cap` diagnostic (`gf_diagnostics`) WARNs
when any GF solve froze; it deliberately does **not** trigger the thermal-ensemble
retry, because more eigenstates cannot widen the basis.

## Reliability sweep

Harness: `src/impurityModel/test/test_truncation_reliability.py` (opt-in; one config
per process so `VmHWM` is honest). NiO d-shell (2026-07 sweep), 20 bath
spin-orbitals, production Lanczos GS path (`dense_cutoff=500`), thresholds
{∞, 2000, 1000, 500, 250, 100} × reort {none, partial, full} × ranks {1, 2, 3}.
Metrics vs the uncapped reference: |ΔE₀|, max relative σ(ω) deviation, causality.

Serial results (relative σ(ω) max deviation vs uncapped; |ΔE₀| = 0 and causality
preserved in **every** run):

| threshold | reort=none | reort=partial | reort=full |
|---:|---:|---:|---:|
| 2000 | 0 | 3e-18 | 4e-16 |
| 1000 | 0 | 3e-18 | 4e-16 |
| 500  | 0 | 3e-18 | 4e-16 |
| 250  | 2.36e-08 | 2.36e-08 | 2.36e-08 |
| 100  | 8.70e-06 | 8.70e-06 | 8.70e-06 |

Readings:

- **The truncation error is reort-independent** — at every binding cap the deviation
  is bit-for-bit identical across none/partial/full. This is the empirical
  confirmation of the `PHP` contract above (in particular, PARTIAL's Paige–Simon
  estimator shows no freeze-step degradation).
- Caps at or above the natural reachable-space size are exactly free (deviation 0;
  the ≤1e-15 partial/full entries are reort-mode arithmetic noise present with or
  without a cap).
- Below the natural size the degradation is smooth and monotone: ~60 % of the
  reachable space costs ~1e-8 in σ(ω), ~25 % costs ~1e-5. The importance-ranked
  boundary admission keeps the error graceful rather than cliff-like.
- The 20-bath workload completes on the production Lanczos path — the historical
  `calc_gs` truncation crashes (pre-`5c2b37e`) are gone.
- **Rank-count independence**: at 2 and 3 MPI ranks the deviations match the serial
  values to the printed precision (2.36e-08 at T=250, 8.70e-06 at T=100, reort=none)
  — the collective bisection picks the same retained sets when candidate amplitudes
  are distinct; only near-ties may differ across rank counts (see item 6 below).

## HPC sizing heuristics

Use `memory_estimate.suggest_truncation_threshold(...)` (what the drivers do when
`truncation_threshold=None`), or size by hand:

1. **Bytes per determinant** (flat_map `ManyBodyState`): `72 B` up to 192
   spin-orbitals (40 B entry + ≥32 B key heap block); Python `Basis` bookkeeping adds
   ~160 B per *local* determinant. Live GF blocks cost
   `16·block_width + key_heap + 24` B per local row, ×3 live blocks.
2. **Per-rank scaling**: sparse-kernel structures scale as `threshold / ranks`
   (hash-distributed). **Exception:** the ground-state array kernel replicates the
   full `(global_N, block_width)` matvec product on **every** rank
   (`BlockLanczosArray.pyx`) — `16·threshold·block_width` bytes per rank that do *not*
   shrink with more ranks. This usually binds the GS estimate on wide runs.
3. **Krylov retention**: reort ≠ none on the GF path retains
   `16·block_width·n_blocks` B per local row (worst case `n_blocks =
   ceil(threshold/block_width)`). At the GF default reort=none this term is zero.
4. **Parallel units multiply memory**: under `run_units_distributed` each color's unit
   basis carries the same threshold, so a rank's share is
   `threshold / (ranks / n_colors)` — budget for `n_parallel_units` when suggesting.
5. **Safety factor 0.5**: absorbs the transient one-matvec-fanout overshoot at the
   freeze step, allocator slack (up to ~2× on flat_map arrays after growth), and the
   CSR Hamiltonian snapshot variability (`nnz_per_state` is model-dependent — measure
   on a small run).
6. **Ties/reproducibility**: near-tie boundary admissions may retain slightly
   different sets across rank counts (summation-order rounding), like the CIPSI basis
   trajectory; physics agreement holds to the sweep tolerances above.

Rule of thumb (GF path, reort=none, 20–200 spin-orbitals):
`threshold ≈ 0.5 · RAM_per_rank · ranks_per_unit / (~1 KiB per determinant)` —
e.g. 4 GiB/rank, 16 ranks, one unit color → threshold ≈ 3·10⁷.

Worked examples (`estimate_*_peak_bytes`, 120 spin-orbitals, 16 ranks, GF width 10 /
GS width 4):

| threshold | GF/rank (reort=none) | GS/rank |
|---:|---:|---:|
| 10⁶ | 53 MiB | 332 MiB |
| 10⁷ | 525 MiB | 3.2 GiB |
| 10⁸ | 5.1 GiB | 32.5 GiB |

The GF slope at reort=none is ~880 B per determinant per unit rank (width 10). The GS
column grows faster because of the array-kernel replication (item 2). At reort≠none
the default `n_blocks` is the invariant-subspace worst case
`ceil(threshold/block_width)` — astronomically conservative for converged runs; pass
a realistic Lanczos depth (tens to hundreds of blocks) when sizing a reort≠none run.

Calibration status: the 20-bath sweep peaks at ~305 MiB `VmHWM`, which is the Python
/ import / mesh floor, not determinant storage — the byte *constants* are instead
pinned exactly against the Cython `memory_bytes()` estimators
(`test_memory_estimate.py::test_bytes_per_determinant_matches_cython`). Re-calibrate
`_PY_BASIS_OVERHEAD_BYTES` and `nnz_per_state` against `VmHWM` on the first
production-size run (≥10⁶ determinants), where the per-determinant terms dominate.
