# truncation_threshold: semantics, reliability, and HPC sizing heuristics

`truncation_threshold` caps the **global number of Slater determinants** a many-body
basis may hold: "keep as many determinants as fit in RAM; when more are generated,
keep only the currently important ones." This document records what the cap does on
each solver path, why it is safe under every reorthogonalization mode, the empirical
reliability sweep, and how to choose the threshold on a cluster.

## What the cap does per path

| Path | Mechanism | Importance criterion |
|---|---|---|
| Ground state (CIPSI, `cipsi_solver.py`) | fixed-budget CIPSI: on overflow, prune to top-K by amplitude (collective bisection) ↔ admit best de2-ranked candidates ↔ re-diagonalize, until E₀ stabilises | eigenvector amplitude (prune) + Epstein–Nesbet PT2 de2 (admit) |
| GF sparse kernel (`block_Green_sparse`, production self-energy) | `_CappedBasisProxy`: admit-while-under-cap → one importance-ranked boundary admission → freeze + project (`keep_rows`) | residual max column \|amp\|² at the overflow step |
| GF array kernel (`block_Green`, spectra XAS/RIXS) | probe expansion stops within one H-application batch of the cap | order of discovery |

The ground-state and GF paths now share one collective top-K primitive,
`manybody_basis.collective_amplitude_cutoff(scores, k, comm)` (geometric bisection over
the nonzero score range on allreduce'd counts — every rank derives the identical cutoff,
ties under-admitted so the cap is never exceeded).

`None` (the driver default) resolves the threshold from available per-rank RAM via
`impurityModel.ed.memory_estimate` (collective probe: `min(MemAvailable, cgroup
headroom)` ÷ ranks-per-node, min-reduced); `np.inf` disables capping. The `Basis`
container itself never truncates.

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
  is the current residual, so the cap adds **zero Krylov-store** overhead (there is no
  store). This does *not* make the path free: the recurrence still holds every retained
  determinant as live support, measured at ~1.3–1.7 kB/det on the real workloads (FCC Ni
  OOMs here, not on the store). See `memory_estimate._GF_RECURRENCE_OVERHEAD_BYTES`.
- **FULL / PERIODIC**: valid; the `SparseKrylovDense` rows are bounded by the retained
  set and the store never needs removal (removal is unsupported there — this design
  never removes).
- **PARTIAL / SELECTIVE**: valid post-freeze (exact recurrence of `PHP`; the
  Paige–Simon estimator sees a genuine Lanczos process). Confirmed empirically by the
  oracle tests and the sweep below.

The **oracle test** (`test_gf_truncation.py`): the capped GF equals the dense resolvent
of `PHP` on the retained set to ~1e-9, for reort ∈ {none, full, partial}, serially and
at 2–3 ranks including ranks that retain zero rows.

Ground-state path (fixed-budget CIPSI): CIPSI truncation happens only *between*
fixed-basis Lanczos solves (after `basis.clear()`), so a truncation never coexists with
a live Krylov store — reort interaction is safe by construction. The historical
`size>0`/empty-`local_basis` desync was fixed in `5c2b37e`.

When `truncation_threshold` binds, `CIPSISolver.expand` no longer stops at the first
overflow. It runs a **fixed-budget CIPSI** loop: prune the basis to the top-K
determinants by eigenvector amplitude (`collective_amplitude_cutoff`, filling the cap
exactly instead of the old ×10 amplitude ladder that overshot), admit the best
`de2`-ranked new candidates into the freed room, re-diagonalise, and repeat until E₀
changes by less than `cap_e_tol` (default `1e-8`) or `max_cap_cycles` (default 10) is
reached; the best basis seen across cycles is kept. Two prerequisites make the ranking
trustworthy:

- The selection score `de2` is the Epstein–Nesbet PT2 magnitude
  `|⟨Dⱼ|H|ψ⟩|² / |E_ref − E_Dⱼ|`. The denominator previously collapsed to a `1e-12`
  clamp (ground-state candidates sit *above* `E_ref`, so `max(E_ref − E_Dⱼ, ε)` was
  always `ε`), degrading selection to a bare coupling filter blind to the energy gap;
  the corrected `|·|` denominator makes `de2_min` a genuine per-determinant energy
  tolerance. Its defaults were recalibrated ~2 orders down when the denominator was
  fixed (`calc_gs` 1e-6→1e-8, occupation search 1e-4→1e-6, DC solvers 1e-3→1e-5) so the
  admitted basis — hence the physics — is unchanged for uncapped runs (verified
  bit-for-bit against the NiO SOC + charge-transfer fingerprint).
- `truncate` retains the globally top-K determinants by max `|amplitude|²` over the
  low-energy manifold (each determinant counted once on its hash owner), the same
  collective bisection the GF cap uses. `truncate_initial` ranks over the 10-state
  manifold, not a single eigenvector.

A capped GS determination is auditable: `CIPSISolver.truncation_report`
(`{cap_hit, cycles, retained, threshold, discarded_de2_mass}`) is surfaced through
`calc_gs`'s `gs_info["truncation"]`, the saved `ground_state_statistics.json`, and the
`calc_selfenergy` result's `gs_truncation` key. The report fires whether the cap binds
the final expansion *or* the earlier occupation search (whose final basis may then fit
under the cap); rank 0 logs a one-line summary when `verbose`.

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

The caps above (100–2000) bound only the **Green's-function** bases; the natural
ground-state basis for this workload is 45 determinants, so `E₀` is exact in every row
of that table. The ground-state cap is exercised separately below.

### Ground-state-binding sweep

Same NiO d-shell (20 baths, `dense_cutoff=500`), thresholds chosen to bind the 45-det
ground state, reort {none, partial, full}. `sig_max` is the relative σ(ω) deviation vs
the uncapped reference; `GS_cap` is `retained/cycles` from the fixed-budget CIPSI report
(`-` = GS not capped).

| threshold | \|ΔE₀\| | sig_max | causality | GS_cap (none/partial/full) |
|---:|---:|---:|---:|:--|
| ∞  | 0        | 0        | −8.3e-05 | − / − / − |
| 44 | 4.3e-14  | 5.15e-02 | −7.4e-05 | 13·2c / 13·2c / 13·2c |
| 40 | 4.3e-14  | 5.15e-02 | −7.4e-05 | 40·3c / 40·3c / 40·3c |
| 35 | 4.3e-14  | 5.15e-02 | −7.4e-05 | 13·4c / 18·3c / 35·2c |

Readings:

- **The ground-state energy is essentially exact even when the cap binds the GS.** At
  every binding threshold `|ΔE₀|` is ~4e-14 (round-off), because the importance-ranked
  fixed-budget CIPSI retains the determinants that carry the correlation energy; the
  determinants it drops contribute below round-off. This is the direct answer to "is a
  truncated CIPSI ground state reliable?" — yes, and gracefully so.
- **The cap binds the ground state here through the occupation search** (the final
  `calc_gs` basis is smaller than the natural 45 and often fits under the cap), which is
  why `retained`/`cycles` vary while `E₀` does not; the report fires regardless (it
  merges the occupation-search and final-expansion caps).
- **σ(ω) degrades on the GF side, not the GS side** — `sig_max` is flat at 5.15e-02
  across 35–44 because these caps bind the (larger) GF bases to a similar effective
  size; it is reort-independent, as in the GF-only table.
- **Causality is preserved** at every binding cap (same sign and order as uncapped).
  Below threshold ≈ 30 the GF `check_greens_function` causality guard raises first (the
  Green's function needs more determinants than the ground state), so that is the
  practical floor for this workload, not a ground-state failure.

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
4. **Parallel units multiply memory — enforced at split time**: under
   `run_units_distributed` each color's unit basis carries the same threshold, so a
   rank's share is `threshold / (ranks / n_colors)`. The split site caps the color
   count via `memory_estimate.max_colors_within_budget` (collective probe) so a
   cap-filling unit basis still fits the per-rank budget; a rank-0 log line reports
   when memory (rather than the participation ratio) binds the split.
   `n_parallel_units` remains available for sizing by hand.
5. **Safety factor 0.5** (`DEFAULT_MEMORY_SAFETY`): absorbs the transient
   one-matvec-fanout overshoot at the freeze step, allocator slack (up to ~2× on
   flat_map arrays after growth), the CSR Hamiltonian snapshot variability
   (`nnz_per_state` is model-dependent — measure on a small run), and unmodeled
   at-scale MPI overheads (e.g. Cray-MPICH per-connection buffers after the first
   alltoallv, transient redistribute buffers).
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

Cluster caveats (SLURM/Cray, e.g. PDC Dardel):

- The probe takes `min(MemAvailable, cgroup memory headroom)` per node, so
  `--mem`-constrained or shared-node allocations are budgeted against the limit the
  kernel actually OOM-kills on, not the whole node. On exclusive nodes the cgroup trim
  (SLURM `RealMemory`) is small and was previously absorbed by the safety factor.
- `[mpiexec -n R] python -m impurityModel.ed.memory_estimate` prints the probe
  (MemAvailable, cgroup headroom, per-rank budget) and the suggested threshold —
  useful for checking a job script's sizing interactively on a compute node.

Calibration status: the 20-bath sweep peaks at ~305 MiB `VmHWM`, which is the Python
/ import / mesh floor, not determinant storage — the byte *constants* are instead
pinned exactly against the Cython `memory_bytes()` estimators
(`test_memory_estimate.py::test_bytes_per_determinant_matches_cython`). Re-calibrate
`_PY_BASIS_OVERHEAD_BYTES` and `nnz_per_state` against `VmHWM` on the first
production-size run (≥10⁶ determinants), where the per-determinant terms dominate:
`calc_selfenergy` now ends by printing the measured communicator-max `VmHWM` next to
the predicted peaks (`memory_estimate.log_peak_vs_predicted`).
