# Unified MPI distribution for Green's-function calculations

Status: **implemented** (self-energy, PES/IPS/XPS/NIXS/XAS spectra, RIXS). This note records
the shared scheme and the deliberately deferred improvements.

## The one scheme

Every Green's-function-like calculation now decomposes into **flat work units** and runs them
through the same engine in `greens_function.py`:

1. `enumerate_gf_units(op_groups, psis, ...)` — one unit per (operator group × eigenstate
   chunk), honoring `GF_EIGENSTATE_GROUP` (wide blocks) and `GF_OPERATOR_SPLIT` (pairwise
   scalar recurrences). RIXS enumerates its own units = (eigenstate × contiguous wIn-chunk,
   `GF_RIXS_WIN_CHUNK`) since its kernel is a resolvent chain, not a single Lanczos run.
2. `unit_cost_weights(unit_seeds, comm)` — the **single source of truth** for split weights:
   Allreduced seed mass × block width + 1. No caller may hand-roll priorities (`[1]*n` and
   `log10(len)+1` are retired).
3. `run_units_distributed(basis, unit_seeds, unit_weights, kernel)` — ONE
   `Basis.split_basis_and_redistribute_psi` over all units; each color runs `kernel` for its
   units on its sub-communicator; per-unit results are gathered to global rank 0 in global
   unit order (serial path short-circuits). The kernel must be collective on
   `split_basis.comm` only and picklable in its return.

Clients: `get_Greens_function` (self-energy), `calc_Greens_function_with_offdiag` (thin
wrapper, used by `calc_spectra_tensor` and tests), `calc_spectra`, `_rixs_map_flat`
(behind `calc_map` / `calc_tensor_map`).

The packing math lives in `manybody_basis._pack_units(weights, comm_size, split_threshold)`:
participation-ratio cap on the number of colors, **LPT packing** (next-heaviest unit into the
lightest bin), largest-remainder rank apportionment (each color ≥ 1 rank). Pure and
deterministic — every rank recomputes the identical packing; unit-tested without MPI in
`test_pack_units.py`.

## Fixed along the way

- `procs_per_color[mask][...] -= 1` wrote to a fancy-index *copy*; when the `max(1, floor)`
  floors over-allocated, the remainder loop never terminated → all-rank MPI hang. Replaced by
  one-pass largest-remainder apportionment.
- Round-robin dealing guaranteed bin 0 the heaviest unit of every round; LPT bounds the
  straggler bin at 4/3 of optimal and reduces to round-robin on uniform weights.
- `calc_map`/`_tensor` referenced `gs` on ranks that never allocated it
  (`UnboundLocalError` on non-root sub-ranks at ≥ 3 ranks); both now return the map on global
  rank 0 (and serially) and `None` elsewhere.
- The RIXS kernel now redistributes the resolvent seeds and warm starts onto the freshly
  rebuilt solver basis each wIn point — `block_bicgstab` assumes its states are aligned with
  `basis`'s ownership layout, and the rebuilt basis need not match where the amplitudes live
  (silent amplitude drop → zero spectra on multi-rank colors).

## Deferred suggestions (not implemented, by decision)

1. **Rank-allocation exponent α** — allocate ranks ∝ `mass_c**alpha` instead of ∝ mass.
   `alpha = 1.0` reproduces today's behavior; `alpha ≈ 0.5–0.7` models the sub-linear parallel
   scaling of a Lanczos run (a color with 2× the ranks does not finish 2× sooner), shifting
   ranks from a few huge colors toward the many medium ones. Tune on the NiO workload before
   enabling; zero-risk to add as a `Basis` attribute defaulting to 1.0.
2. **Per-color imbalance diagnostic** — one rank-0 line under `verbose` after each split:
   predicted per-color mass, ranks, and the max/mean mass-per-rank ratio. With the true-cost
   weights the distribution is far more skewed than under the old log10 compression, so the
   participation cap now fires more often; it should be observable, not inferred.
3. **Participation ratio post-packing** — the color cap is computed from raw unit weights;
   when units ≫ ranks, packing small units together makes the *achievable* balance better than
   the raw participation suggests. Computing the cap on LPT-merged bin masses would allow more
   colors in that regime. Measure before changing — interacts with (1).
4. **Per-color seed migration** — `split_basis_and_redistribute_psi` replicates every psi to
   every color; each color only consumes its own units' seeds. Sending each unit's seeds only
   to its color would cut migration volume and per-color memory ~n_colors×, but changes the
   primitive's contract (per-color psi subsets). Medium-risk performance follow-up, not needed
   for correctness.
