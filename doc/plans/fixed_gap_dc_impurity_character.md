# Does `fixed_gap_dc` measure the impurity gap?

**Verdict: it measures the *cluster* charge gap. On NiO that is the p→d charge-transfer gap — a
defensible quantity, and not the gap of `A_imp` that the prescription is usually quoted for. The
two are several eV apart. No reweighting of these sector energies turns one into the other, and a
weight-thresholded edge is not a usable replacement criterion.**

This closes the item deferred out of the previous review pass (commit `9c45f9b`), which recorded:

> resolve `ω±` from the impurity Green's function — require `|⟨N±1|c_d†|N⟩|²` above a threshold, or
> *refuse to converge* when `δ₊+δ₋ < 0.5` rather than merely warning. […] If the NiO number holds
> up at a production determinant cap, that deferral should be revisited.

It does not hold up in the form it was written, and the reason is more interesting than the claim.

## What was measured

`fixed_gap_dc` drives `centre(mu) = (E₀[N+1] − E₀[N−1])/2` onto an offset, where each `E₀` is the
**lowest** eigenvalue of a total-charge sector. Karolak et al. (arXiv:1004.4569) prescribe centring
the gap of the local spectral function `A_imp`, whose addition edge is, by Lehmann,

```
ω₊ = min{ E_n[N+1] − E₀[N] : w_n > 0 },     w_n = Σ_d |⟨n|c_d†|0⟩|²
```

Taking `min` over **all** `n` instead of the weighted ones is an assumption. `test/support/dc_weights.py`
measures `w_n` and `⟨N_imp⟩_n` per sector eigenstate, so the assumption can be checked rather than
argued about.

**The instrument is validated against closed form first.** On the two-level fixture at `v → 0`,
where `ω₊ = eps + U − dc = 1.5` and `ω₋ = eps − dc = −1.5` exactly, it returns `+1.500018` and
`−1.499938`, `δ = ±1.0000`, weights `1.0000`, representability `1.0000`, and sum rules closing to
100 %. The exact first moments (`M₁/M₀`, computed by completeness with one `H` apply per orbital)
reproduce the same `±1.5`. Tests in `test/gf/test_dc_weights.py`.

## NiO 5-bath, iteration 1 (58 spin-orbitals, the archived production run)

The archive already carries a converged DC, so the criterion satisfies `|g₀| ≤ tol` at `mu = 0` and
returns after **one** evaluation.

| | value |
|---|---|
| cluster gap | `ω₊ = +0.811 eV`, `ω₋ = −0.813 eV`, centred at `−0.0007 eV`, width 1.62 eV |
| edge character | **`δ₊ = 0.561`, `δ₋ = 0.032`** |
| lowest 4 removal states | `δ ≈ −0.032`, weight **exactly 0**, spread over 4 meV |
| first removal state with weight | `k = 4` at `−0.848 eV`, 2.4 % of the sum rule |
| any removal state ≥ 5 % | **none** |
| exact `A_imp` centres of gravity | addition `+1.75 eV`, removal `−5.81 eV` |
| cap 2000 → 8000 | centre moves 6 meV; `occ` `mu` identical to 6 decimals |

The four lowest removal states are within 4 meV of each other and carry no impurity weight: that is
a **band**, not an isolated level — the O-2p valence band. NiO's valence-band top being ligand is
what makes it a charge-transfer insulator. So the criterion is centring the p→d gap, correctly and
stably. It is simply not centring `A_imp`, whose weight sits at `+1.75` and `−5.81 eV`.

The charge-transfer toy fixture fails on the *other* edge and for the *other* reason:
`δ₊ = 0.075` against `δ₋ = 0.951`, with the lowest addition state an isolated fitted level carrying
15 % of the addition sum rule while the state carrying 80 % sits 1.45 above it. Band versus
artefact — and the enumeration distinguishes them.

## Why the deferred fix was not built

**F-a (weight-thresholded Krylov edges) has no root on NiO.** No removal state reaches 5 % of the
sum rule, so `centre_w` is undefined at every `mu`. The Krylov construction is sound — the lowest
Ritz value of `K_m(H, {c_d†|0⟩})` is a variational upper bound on the lowest weighted pole — but
there is nothing near `E_F` for it to find. `A_imp`'s removal weight is 5.8 eV down.

**F-b (refuse below a floor) would have fired on the wrong thing.** NiO's `δ₊+δ₋ = 0.593` sits
*above* the old floor of 0.5 while its removal edge is pure ligand. The sum was the wrong statistic.

## What changed

1. **`groundstate.solve_sector`** — `calc_energy`'s body with the eigenvectors kept; `calc_energy`
   is now that plus `min` plus its broadcast. Verbatim move, so the two cannot drift, and the
   diagnostic measures the solve the criterion actually runs.
2. **`δ₊` and `δ₋` are measured, not inferred, and reported separately.** They came from `chi`, the
   secant of the gap centre against `mu`, via Hellmann–Feynman — exact for an exact eigenstate,
   approximate for a CIPSI space re-selected at every `mu`, and needing two evaluations in one
   sector to exist at all. On this workload the search converged in one evaluation, so
   `chi = None` and `delta_sum = None`: **the diagnostic was silent on exactly the run that
   accepted a DC.** Now three sector solves at the converged shift produce both edges directly.
3. **The warning fires on `min(δ₊, δ₋)`, floor 0.3**, and names the edge and both explanations
   (charge-transfer band vs discretization artefact) rather than asserting the answer is wrong.
   Calibration: fires on 0.032 and 0.075; silent on 0.44, 0.56, 0.95, 1.00.
4. **`tol_basis` no longer says `measured_gap` when the `mu` resolution won the `max()`.** It did on
   both NiO workloads measured (`6.5e-4` and `1.2e-3` against `mu_tol = 2.5e-3`) — the one field
   whose job is to say where a number came from, misreporting in the only case it was read.
5. **Docstring** rewritten around the measurement, including that a weighted edge is not a usable
   replacement here.

## Two process notes worth keeping

**The union-space design this investigation was planned around does not work.** The plan built one
determinant space spanning `N−1, N, N+1` (`dc_criteria.build_union_space`) so the overlaps would
share an index space. `H` conserves the electron number and CIPSI expands from the centre sector's
reference, so the `N±1` blocks never grow past what `generate_initial_basis` seeded — **one
determinant each** against 18 in the centre, on the charge-transfer fixture. A `ManyBodyState` is
keyed by determinant, so the overlap needs no shared `Basis` at all, and each sector then keeps its
own properly expanded space. That is strictly better: the energies are the ones the criterion runs.

**Rank 0's summary hid a 7-failure gate.** Under `mpiexec` pytest prints a dashed rule *after* the
summary line, so `tail -1` on the gate script returned the rule and the run looked clean. Every
failure was in the new serial-only diagnostic tests, which the MPI gate runs on every rank. Take
the summary as the last line matching `passed|failed|error`, never as the last line.

## The last DC↔GS parity gap, closed

Every DC sector energy ran at `de2_min = 1e-6`, a literal inside `calc_energy`, while the
production ground state runs at `1e-8`. The module docstring recorded this as the one convention
still unshared and said closing it was "worth doing, measured, not assumed". It is now closed:
`calc_energy`/`solve_sector` take `de2_min` per call (`SECTOR_WALK_DE2_MIN = 1e-6` by default, so
the occupation walk is unchanged — only the *ordering* of charge sectors is decided there and a
common error largely cancels), and the DC criteria pass `GS_DE2_MIN = 1e-8`, because they
**difference** these energies, where it does not cancel and is then amplified by `1/|chi|`.

Measured on `nio_5peeled`:

| | cap 2000 | cap 8000 |
|---|---|---|
| determinants used, before | 2000 (cap-bound) | **5693** of 8000 |
| determinants used, after | 2000 (cap-bound) | **8000** of 8000 |
| gap centre, before → after | −0.000051 → −0.000051 (bit-identical) | 0.00041022 → 0.00043186 |
| gap width, before → after | 0.119384 → 0.119384 | 0.11846163 → 0.11841800 |
| cost | 52 s (unchanged) | 43 s → 167 s |

**At a binding determinant cap the change is invisible**, because the cap cuts the expansion off
before the PT2 threshold does — which is why this was easy to leave open. Where the cap does *not*
bind, the loose threshold was stopping the expansion with **29 % of the allotted determinant budget
unspent**. The answer moves by 0.3 meV and the cost by 4×. The point is not the 0.3 meV; it is that
the DC is now determined on the same variational space as the self-energy run that consumes it,
which was the whole premise of the parity list.

## Cost

Measured end to end on `nio_5peeled` at cap 2000: the `gap` search went from **9 sector solves /
26 s** to **12 solves / 62 s**. Three of those solves are `_measure_edge_character`; they re-solve
all three sectors from scratch, because `sector_energy` goes through `calc_energy` and discards
eigenvectors, and the rest of the gap is machine contention (a live RSPt run shared the box).
The overhead is worth stating plainly: it is not free, and on a search that terminates in one
evaluation it is a third of the total work. It buys a number that was previously **absent** on this
workload, not merely less accurate.

## Loose ends

- `nio_15`, the workload the earlier `delta_sum = 0.218` came from, was removed from disk on
  2026-08-05. That number cannot be reproduced; `nio_5peeled` replaces it in `WORKLOADS`.
- I2.2 (`A_imp(ω)` from the production PES/IPS path, overlaid on `ω±`) was **not** run. The exact
  first moments answer the same question more cheaply and without a broadening choice, but a
  spectrum would show the *shape* — whether the 5.8 eV removal centroid is one band or two.
- `dc_weights.py` is serial only.
