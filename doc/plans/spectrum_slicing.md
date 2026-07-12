# Spectrum slicing + Chebyshev filtering for the real-axis Green's function

**Status (2026-07-12): Phase 0 measured on FCC Ni.** The support-locality bet is settled with
nuance: filtered seeds localize strongly in their *dominant* amplitudes (up to ~80–260x below
the union support at the 1e-4 amplitude level) but their tails delocalize across the whole
reachable set — at the production `slaterWeightMin` (1.5e-8) the per-slice support is the
union support. Slicing therefore buys memory **only at relaxed per-slice accuracy**: ~2x at a
1e-6 amplitude cutoff (slice-seed accuracy ~1e-3), 4–8x at 1e-5 (~5e-3). Per the user's
decision (calibration, not a gate) the filter stage and sliced driver are built regardless;
the calibration table below is their memory model.

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
