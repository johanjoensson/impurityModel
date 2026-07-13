# Spectrum slicing + Chebyshev filtering for the real-axis Green's function

**Status (2026-07-13): COMPLETE — the answer is no.** All phases built, tested (serial + MPI)
and measured. `gf_method="sliced"` is correct and slice-count invariant, and it **buys no
memory on either production workload**: on FCC Ni it costs *2x* the Lanczos baseline's peak
(2.75 vs 1.4 GiB) before a single solve runs, and on NiO the per-point basis is identical at
every slice tolerance. The premise — that energy localization implies support localization —
is false: the live basis is the **H-connectivity closure** of the seed support, invariant
under any reweighting of the seed, and `(z-H)^{-1}` re-expands to that closure however sharply
the seed is filtered. Slicing also *loses* accuracy (27x worse `sigma_real` than Lanczos),
because it multiplies the number of near-singular solves. Full verdict in Phase 3 below.

*(Superseded — kept for the record.)* **Phase 0 (2026-07-12)** read the bet as settled "with
nuance": filtered seeds localize strongly in their *dominant* amplitudes (80–260x below the
union support at the 1e-4 level) but their tails delocalize; at the production
`slaterWeightMin` (1.5e-8) the per-slice support is the union support, so slicing looked like
it would buy 2x at a 1e-6 cutoff and 4–8x at 1e-5, in exchange for accuracy. **That
projection did not survive the end-to-end measurement**: it priced the *filtered seed's*
support, but the memory is set by the closure of that support under `H`, which the solve
re-grows regardless. The seed-support tables below are still accurate as far as they go —
they simply measure the wrong quantity.

## Why

`doc/plans/bicgstab_per_frequency_gf.md` (Phase 3b, the FCC Ni decision gate) established that
the live determinant support is the memory wall and that no solver choice, occupation window
or per-point basis discard reduces it — the seeds inherit the ground state's cap-saturating
support and a single frequency's solution mixes every pole. The one mechanism left is energy
localization **by construction**: a Chebyshev partition of unity `sum_s p_s(H) = 1` splits

    G_ij(z) = sum_s <v_i| (z - H)^{-1} p_s(H) v_j>,

each filtered seed `v^s = p_s(H) v` carrying spectral weight only in slice `s`, each slice an
independent work unit with its own discarded basis, solved by the per-frequency BiCGSTAB
driver with an unfiltered bra (its natural cross-element form). Design details, user
decisions (block engine = `block_Green_bicgstab` + `bra_seeds`; Phase 0 as calibration) and
the phase plan are recorded in the approved implementation plan (2026-07-12); the theory
section is `doc/greens_function_theory.md`, section 5.

## Phase 0 — support-locality calibration (measured 2026-07-12)

Probe (`test/test_slicing_probe.py`, opt-in): intercept `calc_selfenergy` at its
`get_Greens_function` call (production ground state, restrictions and seeds), estimate the
spectral bounds with 40 Lanczos steps, run one Jackson-damped Chebyshev recurrence of degree
1500 under a 400k `_CappedBasisProxy` cap, accumulate 8 window filters spanning the caller's
real-axis band (+ rest windows), and measure each filtered seed's ε-support.

**FCC Ni 5-bath, block [0], cap 400k.** Union (unfiltered recurrence) support: 400,000
(cap-frozen; the true union is larger). Per-slice ε-support of the weight-carrying windows:

| side | seed | ε=1e-4 | ε=1e-5 | ε=1e-6 | ε=1e-8 (~production) |
|---|---|---|---|---|---|
| removal (`a`) | 12,152 | 5k–21k (**20–80x**) | 47k–102k (4–8.5x) | 167k–251k (1.6–2.4x) | 387k–397k (**≈1x**) |
| addition (`c`) | 4,302 | 1.5k–12k (**33–260x**) | 10k–62k (6–40x) | 50k–186k (2–8x) | 255k–386k (1.04–1.6x) |

Spectral weight is strongly front-loaded: the 2–3 slices nearest the Fermi level carry ~95%
of the seed norm on both sides, and the far-band rest window carries 6e-4 (removal) / 4e-5
(addition) — the exactness-completing far-band term is cheap by *weight*, though not by
support at tight cutoffs.

**Discard-tail arithmetic** (what the cutoffs mean for a slice basis truncated at amplitude
ε with n_ε retained determinants): the discarded tail norm is ≤ sqrt(n_tail)·ε, so a 1e-6
truncation of a ~0.5-norm slice seed is accurate to ~1e-3 relative; 1e-5 to ~5e-3. Hence:

* **High-precision Σ (1e-8): slicing does not help on this metal.** The tail *is* the
  support.
* **Moderate-accuracy real-axis spectra (plots, ~1e-3): 2–8x per-slice memory reduction is
  real**, with the slice tolerance as an explicit, reported knob.
* Insulators / narrow XAS windows remain the plausible strong-locality regime (untested —
  NiO 1-bath is too small to discriminate, its whole sector fits in 32 determinants).

The driver's memory model must therefore price a slice at `support(slice tolerance)`, not at
a fixed fraction of the union, and the slice tolerance must appear in the diagnostics next to
the partition-of-unity and leakage errors.

## Phases 1–3

As per the approved plan: Phase 1 `ChebyshevFilter.pyx` (`spectral_bounds`,
`chebyshev_apply`, `partition_of_unity`; the probe is the reference implementation), Phase 2
`gf_method="sliced"` (slices as work units; `block_Green_bicgstab` + `bra_seeds`;
`check_slice_partition` diagnostics; the calibrated memory model), Phase 3 verification
(SIAM-6 oracle, slice-count invariance, NiO σ parity, the FCC Ni support/VmHWM/causality
table) with the verdict recorded here.

## Phase 2 — the driver (built 2026-07-12, commits `ce9eeb4`, `09c5e6a`, `647bb09`)

`_get_greens_function_sliced` filters each GF unit's seeds once (one Chebyshev recurrence
serves all windows), then fans out **one engine unit per window**, each an ordinary
per-frequency BiCGSTAB/GMRES solve on its own rebuild-and-discard basis, with the *filtered*
ket and the *unfiltered* bra (`block_Green_bicgstab(bra_seeds=...)`). The window terms are
summed back by the shared `_run_evaluated_gf_units` accumulator — a plain sum, which is what
lets several units target the same `(block, side, eigenstate)` and add up.

**Spectral bounds are per unit.** Each unit's excited sector has its own reachable spectrum,
and a Chebyshev polynomial evaluated outside its interval grows as `cosh(n·arccosh|x|)`: at
degree ~10^3 a 1% bounds violation is a ~1e100 blowup. Sharing one unit's bounds across all
units produced `|G| ~ 1e+111` (measured). A norm guard on the filtered seeds (`|p_s| ≲ 1.1`
on the interval, so a filtered norm above ~2x the seed norm means the recurrence left the
interval) now turns that whole class into a hard error rather than a plausible-looking number.

### Two bugs the sliced path had no test for (both found by adding the MPI driver test)

1. **The filter ran on never-redistributed seeds.** Seeds are built by applying `c`/`c†`
   rank-locally, so each amplitude sits on the rank that *generated* it, not the rank that
   *owns* that determinant (`routing_hash % size`). Every other solver reaches its basis
   through a `redistribute_psis`; the Chebyshev recurrence is the one stage that runs before
   it — and it redistributes `H·t` but not `t`. A misplaced row therefore leaves `H·t` on the
   owner and `t` on the generator: the three-term recurrence **decouples across ranks and
   diverges to `inf`**. The Phase-1 MPI filter test missed it by seeding a *closed* basis,
   where generator == owner. The divergence guard is what turned this into an exception
   instead of a number.

2. **The unfiltered bras were admitted into the per-slice basis.** They belong only in the
   closing Gram, and `block_inner_cy` merge-joins the two key vectors, so a determinant in
   `supp(bra)\supp(X)` contributes nothing; ownership is by hash, so the merge-join is
   MPI-consistent *without* basis membership. Admitting them pinned every per-slice basis to
   the **unfiltered** seed support — precisely the quantity slicing exists to avoid paying.
   On FCC Ni, where the seeds saturate the cap, this would have capped every slice at the
   union support and yielded a "slicing buys nothing" verdict for a self-inflicted reason.

The lesson generalizes: *the memory claim and the correctness claim have to be measured on
the same code path the MPI test exercises.* Both bugs were invisible serially.

## Phase 3 — verification and verdict (measured 2026-07-13)

Correctness oracles (unit level, `test_gf_bicgstab_driver.py` / `test_chebyshev_filter.py`):

* **Partition of unity is exact by construction** — the tiling windows telescope, Jackson
  damping included, so `sum_s p_s ≡ 1` and the slicing adds *no* partition error. Tested
  directly on the coefficients and end-to-end.
* **Slice-count invariance**: 1 window vs 4 give the same `G` to 1e-6. This is the oracle for
  the whole scheme — and it was **vacuous until `09c5e6a`**: the `GF_SLICE*` knobs were
  import-time constants, so both legs silently ran the default 8 windows and the test compared
  a number against itself. The test now asserts the reported window counts actually differ.
* **Sliced == PARTIAL-reort Lanczos** on both meshes, serial and under MPI (2 and 3 ranks).
* **`GF_SLICE_TOL`** stays within its predicted discard tail and is reported as a `WARN`, so
  the memory-for-accuracy trade can never be taken silently.

The scheme is therefore *correct*. It is also, on both production workloads, **useless** —
and for a reason worth stating precisely.

### Measured: NiO 1-bath, real axis, 32 points, serial (accuracy)

Against the same run at `gf_method="lanczos"`, `reort=full`. `sigma_static` is bit-identical
in every leg.

| leg | `sigma_real` abs | wall | VmHWM | max per-point basis | solves | above atol / max residual |
|---|---|---|---|---|---|---|
| Lanczos `partial` | 5.3e-6 | 20 s | 324 M | — | — | — |
| BiCGSTAB + GMRES | 1.0e-5 | 1,031 s | 320 M | 718 / 734 | 576/blk | 0 / 1.8e-8 |
| **sliced, exact partition** | **1.5e-4** | 3,988 s | 323 M | **718 / 734** | 2,880/blk | 3 / 4.8e-6 |
| sliced, slice tol 1e-6 | 9.7e-5 | 3,838 s | 323 M | **718 / 734** | 2,880/blk | 5 / 4.7e-6 |
| sliced, slice tol 1e-5 | 1.5e-3 | 3,784 s | 323 M | **718 / 734** | 2,880/blk | 8 / 4.3e-6 |

Two results, both negative:

1. **The per-point basis is identical at every slice tolerance** (718/734, rebuild floor
   710–734) — and identical to the *unfiltered* BiCGSTAB run. Pruning the filtered seed does
   not shrink the basis, because the solve's Krylov growth re-expands it. This is the finding
   Phase 0 could not see: it measured the *filtered seed's* support, but the quantity that
   sets the memory is the **H-connectivity closure** of that support, and repeated
   application of `H` spreads any seed across the whole connected sector. Energy locality is
   not determinant locality. Filtering changes a seed's *amplitudes*; the basis is a
   *support* object, and `(z-H)^{-1}` couples the seed to everything `H` can reach.

2. **Accuracy is 27x worse than Lanczos, not better.** The partition is exact, so this is not
   a slicing error — the diagnostics name it: 3–8 of 2,880 solves per block finish *above*
   atol, at residuals ~4e-6 against BiCGSTAB's 1.8e-8 on the same workload. Slicing
   multiplies the solve count by the window count *and* hands each solve a worse-conditioned
   system: the filtered ket `p_s(H)v` concentrates its weight exactly in the window that
   contains `z`, so the slice carrying the answer is by construction the near-singular one.
   And `sigma_real` is famously unforgiving of per-point residuals (9.2e-3 → 2.0e-4 → 5.4e-7
   as residuals tightened, Phase 3b). Slicing spends its extra solves buying worse residuals.

### Measured: FCC Ni 5-bath, real axis 5 points, cap 400k (memory — the decision gate)

| leg | wall | VmHWM | outcome |
|---|---|---|---|
| Lanczos `none` (baseline) | 14,972 s | **1.4 GiB** | frozen at 399,999; block [0] converged 9.3e-10 in 181 blocks, block [2] not converged at max_iter |
| sliced, exact partition | killed at 8 h 16 m | **2.75 GiB** | still in the *filter stage* (unit 12 of 16); **no solve had started** |

**Slicing costs 2x the memory of the recurrence it was meant to replace, before doing any of
the work.** The mechanism is the one Phase 0 predicted and NiO confirmed: at the production
`slaterWeightMin` the filtered seeds' support *is* the union support, so the filter must hold
3 recurrence blocks **plus one accumulator per window** over that same full support — 6
block-rows where the Lanczos recurrence needs 3. The leg was killed rather than run to
completion (it needed ~18 more hours to reproduce a verdict three independent measurements
already agreed on: the Phase-0 probe, NiO end-to-end, and its own pre-solve peak).

### Verdict

**Spectrum slicing is not the memory lever for these workloads, and it never could have
been.** The bet was that energy localization implies support localization. It does not: the
live basis is the H-connectivity closure of the seed support, which is invariant under any
reweighting of the seed — and the resolvent re-expands to that closure no matter how sharply
the seed is filtered. Slicing buys nothing on memory, costs 4x wall against per-frequency
BiCGSTAB (200x against Lanczos), and loses two orders of magnitude of `sigma_real` accuracy
to the extra ill-conditioned solves.

`gf_method="sliced"` is kept — it is correct, tested (serial + MPI, slice-count invariant),
and it is the honest home for the Chebyshev machinery (`spectral_bounds`, `partition_of_unity`,
`chebyshev_apply`), which is reusable for KPM densities of states and for filtered
eigensolvers. It is **not** a recommended production mode and the decision guide says so.
The one regime never tested and still open in principle: a **narrow XAS/RIXS window on a
gapped insulator**, where the seed's reachable set may genuinely be a small subsector — but
that is a *restriction/sector* argument, not a filtering one, and it should be attacked with
occupation windows, which are free.
