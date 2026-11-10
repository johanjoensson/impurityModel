# Excitation bounds on real workloads: planned runs

Status: planned, 2026-11-06. Nothing has been replayed yet. The findings in §4 come from production logs the user
ran on the new code.

## 1. Why

`excitation_budget` was removed on 2026-11-05 (series `a1896fd5..2d9fb230`, gated green, not pushed). The ground
state is now unbudgeted. The excited bases (GF, spectra, RIXS, susceptibility) run under three bounds that are read off
the converged GS and widened by +1:

- max valence holes;
- max conduction electrons;
- max total.

This investigation asks two questions, accuracy first:

1. **Accuracy.** Do the derived bounds + 1 lose accuracy in E0 and Σ against an unrestricted excited basis? Does Σ
   stay causal?
2. **Cost.** What do wall time, peak memory (VmHWM), and GS/GF basis sizes cost now, compared with budget 6?

Systems with bath states close to E_F hit the determinant cap on the local machine (8 cores, 15 GB). Systems with a
gapped bath mostly do not. The runs are therefore split into a **gapped track**, which answers question 1, and a
**conduction track**, where nothing converges and the questions are different (§6).

## 2. What is compared

| arm | code | excited bases |
|---|---|---|
| **OLD** | `92df4a91` (parent of the bounds series), worktree `debug/budget_ab/old/` | budget 6, **always passed explicitly** (`--budget 6`): the new archives record no budget, and OLD's loader would then fall back to the library default `EXCITATION_BUDGET_DEFAULT = 4` without saying so. Each OLD record reports the budget OLD's `BasisOptions` actually used. |
| **NEW** | `2d9fb230` (HEAD), worktree `debug/budget_ab/new/` | GS-derived bounds + 1 |
| **REF** | NEW with `groundstate.excitation_bound_restrictions` returning `[]` | unrestricted (only `dN`/chain restrictions); **identical GS to NEW**, so every difference is the excited basis |

Fixed in every arm:

- **Build:** `safe` (`-O3` only), compiled in place in each worktree and pinned with `PYTHONPATH`. The driver asserts
  `impurityModel.__file__`. `release` does not give bitwise-identical results across ranks.
- **Caps:** an explicit `truncation_threshold`, identical in every arm of a system. This sets a final GS cap and GF
  cap. Auto caps depend on rank count and free memory, and `max_total` depends on the cap whenever the GS is cap-bound.
- **Threads:** `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, 3 ranks unless stated.
- **Fidelity:** each replay goes through `test/support/real_workload.py`. Its `ignored_attrs` must be empty. One
  exception: `warm_start`/`warm_tol` are bath-fit options the loader does not yet list in `_BATH_FIT_ONLY`
  (loader gap, small repo fix pending). `run_arm.py` allows exactly these two and refuses any other.

Driver: `debug/budget_ab/run_arm.py ARM H5 --cap N [--n-iw K --n-w K]`. It writes `<arm>.npz` (Σ on both axes,
energies) and `<arm>.json` (scalars).

## 3. What is measured per run

- **Ground state:** E0, determinants, the three observed bounds, `[cap-bound]`, discarded PT2 at the cap, the number of
  states in the lowest level vs the certified degeneracy.
- **Excited bases:** size per GF unit, cap hits.
- **Accuracy:** worst Im Σ on the Matsubara and real axes (causality). ‖Σ_NEW − Σ_REF‖ and ‖Σ_NEW − Σ_OLD‖ on both axes.
- **Cost:** GS wall, GF wall, VmHWM (max over ranks).

Before any small NEW-vs-REF difference is trusted, the ‖ΔΣ‖ measure must be shown to discriminate. OLD at budget 2
must come out clearly different from OLD at budget 6.

## 4. Findings already in hand (production logs, new code)

| # | system | finding | consequence |
|---|---|---|---|
| F1 | **FeO** | The GS stopped at its cap (164,559) with 0 discarded PT2. Residual PT2 was within tolerance *before* the final truncation. **The thermal manifold holds a partial orbit: 5 of the 15 states** of the lowest level (⁵T₂, 3 × 5). Certification also found classes making at least 17 states in the thermal window. The log blames either the cap or the thermal window. | **Breaks the rule that the GS keeps every symmetry of H**, whatever the cause. R0 finds the cause, and whether the budget removal is behind it. |
| F2 | **BCC Fe** 5-bath (62 spin-orbitals, 6 conduction) | Sector-6 certification has 35 classes. Each runs to the 718k cap at 180–260 s, about 2.5 h per sector. Residual PT2 at the cap is 4–7e-3, while the best classes differ by 1.4e-4. | The class ranking costs a full-cap solve per class and resolves nothing. Candidate fix: rank under a small cap, give the full cap only to contenders. |
| F3 | **CeO₂** (56 spin-orbitals, gapped, SOC) | Hit its 266k cap, but discarded PT2 was only 1.75e-6 (Fe 1.6e-4). Fan-out 58.6 rows per determinant: SOC removes S_z, and the f⁰/f¹L/f²L² mixing is strong (n_f 0.69). | A cap hit is a matter of degree, so grade it by discarded PT2. CeO₂ belongs to the gapped track. |
| F4 | **NiO 20-bath** (204 spin-orbitals, gapped) | Classes converge at 3–16k determinants in 1–14 s. Pass 2 reaches PT2 1e-8 at 26k. | Size does not decide the cost. Bath levels near E_F do. |
| F5 | **CeO₂** input | The 3-bath fit deviates by 0.12, and the n_f discretization error is +0.20. | A DFT/fit-side issue that will dominate any Σ comparison on CeO₂. Use it for GS and bounds questions, and compare Σ only between arms. (Correction 2026-11-06: an earlier version called Δ acausal on the strength of `dc_reference`'s "causality" line. That line is the elementwise max Im Δ, off-diagonal elements included, so it is not a causality test. The anti-Hermitian part of Δ has eigenvalues ≤ −1.7e-5, so Δ is causal.) |

## 5. Systems

Sizes are spin-orbitals: impurity / valence bath / conduction bath, from the archives' `H solver`.

| system | imp/val/con | N0 | track | archive | what it adds |
|---|---|---|---|---|---|
| SMO 2-bath | 10/18/0 | 3 | gapped | ✓ | smallest; accuracy anchor where REF is cheap |
| SMO 3-, 4-bath | 10/28/0, 10/38/0 | 3 | gapped | ✓ | how the bounds grow with bath count; 4-bath has a large thermal manifold |
| CrI₃ (SOC) | 10/40/0 | 3 | gapped, off-diagonal | ✓ | 24 off-diagonal impurity terms; 12 GF units, all at the cap |
| CoO | 10/48/0 | 7 | gapped | ✓ | 12-fold lowest level (⁴T₁), fan-out 1.8, the cheapest d case |
| FeO | 10/48/0 | 6 | gapped | ✓ | 15-fold lowest level, finding F1 |
| CeO₂ (SOC) | 14/42/0 | 0 | gapped, off-diagonal | ✓ | covalent f⁰, high fan-out, finding F3 |
| UO₂ (SOC) | 14/42/0 | 2 | gapped, off-diagonal | ✓ | 5f² with a multiplet; fan-out 61.3 |
| NiO 20-bath | 10/194/0 | 8 | gapped | ✓ (no budget recorded) | largest gapped system; GS measured cheap (F4) |
| AFM NiO | 20 total | 8 | gapped, off-diagonal | needs rebuild from `Ham-NiUp.inp` | GF blocks up to 4 wide |
| FM-Gd `impmod` (2 baths, `fit_occ`) | 14/24/0 | 7 | gapped | ✓ | weak hybridization (max \|V\| 0.0065 Ry); GS ≈ HF seed; did not cap; 12 certification stalls |
| FM-Gd `impmod_unocc` (`fit_unocc`) | 14/0/32 | 7 | conduction only | ✓ | **no valence bath**: the fit put all 32 bath states above E_F (+0.019…+0.36 Ry), so this is not "gapped + conduction"; did not cap |
| NiO 10-bath, first SCF round | 10/92/4 | 8 | conduction | ✓ (`.bak/outdir-2026-10-24/outdir-1/impurityModel_data.h5`, group `Ni 1`; logs in `solver-it-1/`) | the known cap-limited acausal Σ |
| BCC Fe 5-bath | 10/46/6 | 5 | conduction | ✓ | d levels at E_F; finding F2 |
| α-Ce | 14/20/22 | 1 | conduction, off-diagonal | ✓ | bath straddles E_F (−0.005 / +0.015 Ry); mixed valence |
| FCC Ni 15-bath | 10/89/59 | 8 | conduction | ✓ | largest metal; GS only |

Dropped: Yb (user decision). Nd deferred: no verdict depends on it. α-Ce covers the f-shell metal, and FeO covers a large manifold under a cap. It is a later validation case for the R0/F2 fixes (a 10-fold J = 9/2 manifold in a metal).

## 6. Runs, in order, and what each tells us

### R0. FeO manifold split (GS only, 3 ranks)
Four runs:
- OLD at the production cap 164,559;
- NEW at the production cap 164,559;
- NEW with the cap raised, same thermal window;
- NEW with the thermal window widened, same cap.

Whichever change restores all 15 states names the cause: the cap or the window.
- **OLD keeps all 15 and NEW keeps 5:** removing the budget let each state's basis grow until the cap could hold only
  part of the manifold. That is a regression, and a cap that cuts a symmetry orbit is a bug in either arm.
- **Both split:** a pre-existing cap/certification bug, independent of the budget.

Either way it opens its own fix series (orbit-closed truncation under the cap) before FeO's Σ is used for anything.

### R1. Fe: did the budget keep metals below the cap? (GS only, OLD, `--budget 6`, auto cap)
The answer is in the first class's cycle trace: it either converges below the cap or doesn't. The run is stopped after one or two classes, because a capped OLD would also take hours.
- **Below the cap:** removing the budget turned a calculation that converged locally into a capped one. Then
  metals need a structural limit again, either a default or bounds taken from a converged small-cap solve, not from the
  capped GS.
- **At the cap anyway:** the budget was hiding nothing, and the only fix is the ranking cost (F2).

Also gives OLD's class count and cost per class for comparison with F2.

### R2. Rank invariance (SMO 2-bath, NEW, GS only, `-n 1/2/3`)
The three bounds and E0 must be identical across rank counts. If they are not, the excited problem depends on the
rank count, and that is fixed before anything else is read.

### R3. Accuracy anchor (SMO 2-bath, OLD/NEW/REF, full GF)
**Precondition for every REF verdict (R3, R4, R7): no REF GF unit may hit the cap.** A capped REF unit is just another capped arm, filled with different determinants, so it is not more accurate than NEW. If any unit caps, record "no reference" for that system and draw no verdict on +1.
- ‖Σ_NEW − Σ_REF‖ small, and well below the budget-2 vs budget-6 spread: **+1 is enough** on a gapped d³ system.
- NEW visibly short of REF: the widening is too narrow. Look at which bound binds, holes vs electrons vs total, to
  decide between +2 and per-side widths.

Also the first measure of cost against budget 6.

### R4. Gapped ladder (OLD/NEW, plus REF where it fits; full GF at pinned caps)
SMO 3/4-bath, CoO, CrI₃, CeO₂, UO₂, then NiO 20-bath, then FeO after R0.
- **Bounds vs system:** do the bounds grow with bath count and filling, or saturate?
- **Large thermal manifolds** (SMO 4-bath, CoO): the bounds come from the whole manifold. Check that no member's
  excitations exceed them.
- **Off-diagonal blocks** (CrI₃, CeO₂, UO₂): a unit seeds several orbitals into one excited basis under one set of
  bounds. Does REF agree on the off-diagonal Σ too?
- **Cost:** wall time and VmHWM against budget 6, at equal caps.

### R5. Conduction track (NEW cap ladder 50k / 100k / 200k / 400k, GS first, then GF where affordable)
NiO 10-bath (first round), Fe, α-Ce, FCC Ni 15-bath (GS only).
- **Bounds vs cap:** if the bounds keep growing with the cap, they are cap artifacts, and the excited problem on a
  metal is set by the hardware.
- **E0 and Σ vs cap:** do they converge? Where does Σ turn causal, if it does?
- **Discarded PT2 at each cap:** a graded measure of how cap-bound each system is.
- **Ranking cost per class** at each cap: the data for the F2 fix.

### R6. Gd pair
Before running, from the archives only: the bath levels nearest E_F, their V, the estimate
V⁴ / (E_add² · ΔE_bath), and the probe fan-out. Then GS for both fits.
- **`fit_occ` converges and `fit_unocc` caps:** the conduction bath itself drives the cost, through the second-order
  bath particle-hole channel, so the fix also concerns how bath pairs near E_F are admitted.
- **Both converge:** what makes a metal capped is impurity levels at E_F (Fe, α-Ce), not hybridization alone.

### R7. AFM NiO (once rebuilt)
The cheapest off-diagonal test, with blocks up to 4 wide: OLD/NEW/REF, full GF.

## 7. Outcome of the investigation (verdicts, 2026-11-07)

Status: closed 2026-11-07. Every planned run is done except the FM-Gd GFs, which the user stopped as too slow for one data point: the valence fit after about 9 h (a second ensemble member pending), the conduction fit after 4.4 h of NEW's GF. The verdicts below rest on the completed runs.

### V1. Is +1 sufficient? **Yes when the GS is converged. Untested when the GS is cap-bound.**
- **REF checks passed: 3 of 3, all bitwise.** SMO2 (R3), CoO (R4-GF) and AFM NiO (R7, off-diagonal width-2 units included) give NEW = REF exactly, in E0 and in Σ on both axes, and no REF unit hit the cap. CoO and AFM NiO have the tightest bounds tested (excited bases at 4 and 3 valence holes).
- **The bound is never active.** CIPSI's GF selection stays inside the bound by itself, so NEW's excited bases equal REF's to the determinant. The bounds act as a safety rail, not as a size control.
- **The check can see a binding limit:** OLD at budget 2 on SMO2 gives a grossly acausal Σ (Im Σ = +1.49), and the library rejects it.
- **Not covered, most important first:**
  1. **A cap-bound GS.** All three REF checks had a GS converged below the cap. When the GS is cap-bound, the bounds are cap artifacts (V4: SMO3 holes 5 at 300k against 7 at 1M), and that is where GS-derived bounds could be too tight for the GF. It is also the common production case for conduction-bath systems. Closing test (about 1 h): take the bounds from SMO2 GS-only runs at 50k/100k (cap-bound), force them onto a NEW GF at 1M (the same monkeypatch pattern REF uses), and compare against the existing `r3_smo2/ref` Σ.
  2. **GF units at the cap.** SMO3 at 1M froze 8 of 16 units, so REF cannot be a reference there within local memory. The metals were not GF-tested.

### V2. Did removing the budget cost accuracy or time? **No accuracy loss. A small time cost only where the budget used to bind.**
- **GS, 7 gapped systems at a pinned 300k cap** (SMO3, SMO4, CoO, CrI3, UO2, CeO2, NiO20): OLD and NEW give the same E0, or NEW is lower (SMO3, by 1.5e-7). Wall times agree to within about 7%.
- **GS, metals at a pinned 50k cap:** NEW is lower in E0 (Fe by 2.5e-4, FCC Ni by 2.4e-5), and OLD is 10–16% faster.
- **GF, SMO2:** OLD's budget 6 truncated units by up to 2.5k determinants and shifted the real-axis Σ by 6e-5. NEW removes that truncation and costs about 13% more wall time.

### V3. Rank invariant? **Yes** (R2: 1/2/3 ranks give the same E0 to 1e-14, the same basis and the same bounds).

### V4. Metals and other cap-bound systems: **restoring a budget would not help. The ranking cost (F2) is the problem.**
- **The budget never kept a metal below the cap:** Fe at 718k (R1) and at 50k (R5c), FCC Ni at 50k (R5d).
- **When the GS is cap-bound, the bounds are order statistics of the cap,** not physics:
  - α-Ce: all three bounds grow from 50k → 200k (R5b);
  - SMO3: holes 5 at 300k, 7 at 1M.
  They converge only when the GS does (NiO10 first round: holes = 6 from 100k on; CoO, NiO20, Gd).
- **The cost grows faster than the cap** (α-Ce ×3.1 per doubling) and is set by bath levels near E_F, covalency and fan-out, not by orbital count.
- **Fix (own series):** rank pass-1 classes under a small ranking cap, and give the full cap only to classes whose energy bound reaches the best.

### V5. Cap-made physics defects (pre-existing, not caused by the budget removal; each needs its own series)
1. **A degenerate level is cut** (symmetry rule violated): **FeO only** confirmed. It keeps 5 of the 15 states of its lowest level (R0, thermal-union cut). Fix: a level-wide equal-share cut that keeps orbits whole.
2. **The cap truncates the thermal ensemble.**
   (a) **Higher levels.** Fe keeps 1 state where certification found classes making ≥ 3, and FCC Ni 1 where they make ≥ 15, at 50k. Their lowest level is a single state (FM, no S_z symmetry), so no orbit is split, but higher levels inside the thermal window are missing.
   (b) **Neighbouring sectors.** Because a sector is cap-bound, `mixed_valence` keeps a single N_imp sector and drops neighbours that lie inside the thermal window. This happens in the metals (α-Ce 2.6/3.7τ, Fe 0.8τ, FCC Ni 2.0/2.3τ) and **also in gapped systems** (CrI3 7.3τ, SMO4 4.2τ, NiO20 2.5/4.0τ, NiO10 first round). For Fe, the sector choice itself is within the truncation error.
3. **The `[cap-bound]` flag is set by intermediate expansions,** so it fires even when the final basis is well below the cap (NiO10 at 400k, NiO20, SMO3 at 1M with 0 discarded PT2).

### Recommendation
Keep the excitation bounds +1 unchanged. Keep the budget retired. Spend the next effort, in this order:
1. the GS ranking cost (V4);
2. the symmetry-preserving level cut (V5.1);
3. sector selection under a cap (V5.2);
4. the flag, the loader gaps and the `dc_reference` causality diagnostic (§8).

## 8. Open items
- **Loader** (`real_workload`):
  - add `warm_start`, `warm_tol`, `symm_prec` and `DC ground state cap` to the ignorable set;
  - `load_workload` defaults to the archive's `last iteration`, a trap for first-round replays (NiO10: iteration 84 has no conduction bath).
- **`dc_reference` "causality" line:** it is the elementwise max Im Δ, not a matrix causality test. Use the eigenvalues of (Δ − Δ†)/2i.
- **`run_arm.py --gs-only`** records only the first ensemble member. Gd's valence fit is a two-member f⁶/f⁷ ensemble.
- **Gd:** no GF verdict (both fits stopped). A fit that straddles E_F would still be needed to separate valence-side from conduction-side effects.
- **GF tail cost:** on Gd, one discovered-mode unit ran alone for hours while the other ranks idled in the gather. That is worth profiling in the GF queue work.
- **impmod_tests `green.inp` files** still carry the old budget token, which the interface now refuses.

## 9. Results

### R0. FeO manifold split (2026-11-06, 3 ranks, cap 164,559, GS only)

| arm | E_GS | states kept / level | thermal basis | bounds (h/e/t) | GS wall | VmHWM/rank |
|---|---|---|---|---|---|---|
| OLD, budget 6 | −32.794002912 | 5 / 15 | 47,200 | (budget) | 570 s | 836 MB |
| NEW | −32.794002912 | 5 / 15 | 47,200 | 4 / 0 / 4 [cap-bound] | 571 s | 837 MB |

- The budget never bound FeO's GS: P(4 bath excitations) = 1.4e-7 and none at 5–6. OLD and NEW solve the same problem.
- **The split is pre-existing, not caused by the budget removal.** The 5 kept states are degenerate to 1e-13, so they
  are one complete S = 2 quintet, i.e. one orbital component of ⁵T₂. The other two orbital components are missing.
- The thermal basis (47,200) is far below the cap, so the final union was not cut at the cap. The cap was hit
  earlier, in the class solves.
- **Cause: the cap cut of the thermal union.** In sector 6 every class converges alone (pass 2 PT2 1e-8 at
  12–27k determinants). The lowest level appears in 6 classes plus mirrors (13 in the window). Their union needs
  more than 164,559, so `sector_solver` logs "the 13 class(es) inside the window were cut to a common share"
  (163,974 kept). The eigensolve on the cut basis keeps one quintet and trims to 47,200.
- **At cap 330k the union fits** (244,314 determinants, no cut) and **all 15 states** come out degenerate to 1e-14.
  The window-completeness warning remains (15 of ≥ 17).
- Why the cut is not orbit-closed for this level: the sector's symmetry group is spin SU(2) only ("4 hard labels
  (spin: SU(2))"). The cubic operations that map the three T₂ orbital components onto each other are not in the
  group, so a cut by importance keeps whole spin multiplets but not whole orbital triplets. In the code, the cut
  importance is also passed only to the first level source (`importance if n == 0 else None` in the thermal-window
  union).
- **Verdict:** a pre-existing violation of the all-symmetries rule, independent of the bounds. Fix as its own
  series: when the level does not fit, cut every member of the degenerate level to the same share, using a
  level-wide importance, or refuse the cut. Validation: FeO at 164,559 must keep 15 equal-quality states.

### R1. Fe: did the budget keep metals below the cap? (OLD, `--budget 6`, cap 718,449 pinned = NEW production's auto cap, 3 ranks)

First class of sector 6 (2Sz = −4, parity 0000); the run was stopped after it:

| | OLD, budget 6 | NEW (production log) |
|---|---|---|
| E | −11.03410627 | **−11.03496353** (8.6e-4 lower) |
| determinants | 718,449, cap-bound | 718,449, cap-bound |
| PT2 left | 3.6e-3 | 7.0e-3 |
| discarded PT2 at the cap | 1.8e-5 | 1.6e-4 |
| cycles to the cap | 6 (same growth: 46 → 523 → 4.6k → 33k → 189k → cap) | 6 |

- **No.** Budget 6 does not keep Fe below the cap: the growth curve is the same up to 189k, and the cap binds at
  cycle 6 in both arms. Removing the budget did not turn a converged problem into a capped one.
- At the same cap, NEW reaches a lower energy. The budget made the space smaller, not the solve converged.
- Consequence: metals need the ranking-cost fix (F2), not a restored structural limit. (Wall times are not
  comparable: OLD shared the machine with other runs.)

### R2. Rank invariance (SMO 2-bath, NEW, cap 200,000, GS only)

| ranks | E_GS | states | determinants | bounds h/e/t | wall |
|---|---|---|---|---|---|
| 1 | −6.81113617656556 | 4 | 176,760 | 7 / 0 / 7 | 637 s |
| 2 | −6.81113617656556 (Δ 6e-15) | 4 | 176,760 | 7 / 0 / 7 | 259 s |
| 3 | −6.81113617656556 (Δ 2e-15) | 4 | 176,760 | 7 / 0 / 7 | 337 s (shared machine) |

**Verdict: rank invariant** at 1/2/3 ranks: same E0 to 1e-14, same basis size, same three bounds.

- Not cap-bound. The hole bound 7 exceeds the old budget 6, but the excitation distribution is P(5) = 3.2e-6,
  P(6) = 1.3e-8, **P(7) = 9.9e-12**. The bound is set by determinants with 1e-11 total weight that the PT2
  tolerance admitted: the order-statistic sensitivity in practice. The excited bases then run at 8 holes,
  where OLD used 6. R3 measures whether those extra excitation levels cost anything or buy any accuracy.

### R3. Accuracy anchor (SMO 2-bath, OLD/NEW/REF, full GF, cap 1,000,000 GS and GF, 3 ranks, 2048 iw / 2001 w)
The three arms ran one at a time on an otherwise idle machine.

| arm | E0 | excited-basis restriction | largest GF unit | cap hit | wall | VmHWM/rank |
|---|---|---|---|---|---|---|
| REF | −6.811136176566 | none | 371,250 | no unit | 59:52 | 526 MB |
| NEW | −6.811136176566 | valence holes ≤ 8 | 371,250 | no unit | 61:03 | 532 MB |
| OLD (budget 6) | −6.811136176477 (+9e-11) | budget 6 | 368,721 | no unit | 54:02 | 522 MB |

ΔΣ against REF (relative Frobenius / max |ΔΣ|); worst diagonal Im Σ is the same for every arm (−7.8e-4 iw, −1.15e-3 w, all causal):

| arm | Matsubara | real axis |
|---|---|---|
| NEW | 0 / 0 (bitwise) | 0 / 0 (bitwise) |
| OLD budget 6 | 8.6e-9 / 1.9e-7 | 6.3e-5 / 7.5e-4 |

- **REF precondition met:** no REF unit reached the cap. The REF log has no bounds line, while NEW prints `valence holes <= 8` on all 34 excited bases, so the patch worked.
- **Verdict: on SMO2 the +1 bound is never active.** CIPSI's GF selection stays below 9 valence holes by itself, so every NEW unit equals REF to the determinant and Σ matches bitwise. The 1e-11-weight tail (R2) that sets the bound costs nothing here, because the bound only limits and never forces the basis to grow.
- OLD's budget 6 did bind slightly: units were ≤ 2.5k determinants smaller and the real-axis Σ was 6e-5 off. That is small, but it is a real truncation that NEW removes, and NEW pays about 7 min more wall time for it (about 13%; unit CSR times dominate).
- **Discrimination check (OLD, budget 2):** the largest unit falls to 50,646 determinants (REF: 371,250). Σ comes out grossly acausal: real-axis worst Im Σ is +1.49 at ω = −0.276 (relative 1.0), and Matsubara is +8.0e-3 (relative 0.37). The library raises `UnphysicalGreensFunctionError` after 7:43. So a binding excitation limit on this system shows up as a large, loud error, not a quiet 1e-5 shift. That makes the bitwise NEW = REF result above meaningful: the comparison would have seen a binding bound.

### R5a. NiO 10-bath, first SCF round (`outdir-1`, `--iteration 1`, group `Ni 1`; 10 imp / 92 val / 4 con), NEW cap ladder, GS only, 3 ranks
Pitfall: `load_workload` defaults to the archive's `last iteration` attribute, which is 84 for this archive and has 0 conduction orbitals, so `--iteration 1` is required. The first launch made exactly this mistake and was discarded.

| cap | E0 | E0 − E0(400k) | GS dets | holes / elec / total | wall | VmHWM/rank | discarded PT2 at cap |
|---|---|---|---|---|---|---|---|
| 50k | −53.329991487314 | 3.0e-7 | 49,918 | 5 / 4 / 9 | 11:46 | 846 MB | 2.7e-8 |
| 100k | −53.329991760628 | 2.7e-8 | 99,998 | 6 / 4 / 10 | 18:01 | 1014 MB | 1.6e-7 |
| 200k | −53.329991787574 | 4.1e-11 | 162,918 | 6 / 4 / 10 | 22:10 | 1137 MB | 1.1e-7 |
| 400k | −53.329991787615 | — | 162,048 | 6 / 4 / 10 | 24:39 | 1137 MB | 0 |

(Wall times were measured with the R4 queue running at the same time on 3 other cores.)
- **E0 and the bounds converge with the cap.** The hole bound settles at 6 from 100k on, and E0 is within 4e-11 of the 400k value at 200k. The final GS basis (~162k) sits below the 400k cap. Four conduction orbitals are too few to make the problem open-ended, so the user's rule ("conduction baths hit the cap") is about the *size* of the conduction bath, not its presence.
- **The electron bound is saturated:** 4 = all 4 conduction orbitals, and P(4) ≈ 0. The +1 widening to 5 is vacuous. It restricts nothing, so the electron side runs unrestricted.
- **The `[cap-bound]` flag is still set at 400k**, even though the final basis is 162k and the discarded PT2 is 0. The log says the N_imp = 8 thermal solve was "bound by the determinant cap", so an intermediate expansion or union reached 400k before e_pt2 truncated it. The flag reports a transient, not the final basis. Open item: check whether `cap_bound` should mean "final basis at the cap".

**Diagnostic bug (found 2026-11-06, user correction):** `dc_reference._continuum_fermi_filling` (`dc_reference.py:312`) reports `max(Im Δ)` over every matrix element as "causality (expect <= 0)". A causal matrix Δ can have off-diagonal Im > 0, so the test must use the eigenvalues of (Δ − Δ†)/2i. On the fitted Δ, which is causal by construction, it prints +1.69e-1 for UO2 and +2.6e-2 for CeO2, while the true maximum eigenvalues are −2.0e-5 and −1.7e-5. Every "acausal input Δ" reading taken from that line is void. Fix it in a separate series.

### R5b. α-Ce (14 imp / 20 val / 22 con, N0 = 1), NEW cap ladder, GS only, 3 ranks
| cap | E0 | ΔE0 vs previous | states in window | holes / elec / total | discarded PT2 at cap | winner residual PT2 | wall | VmHWM/rank |
|---|---|---|---|---|---|---|---|---|
| 50k | −1.524064466 | — | 3 | 4 / 3 / 6 | 6.4e-6 | 3.1e-5 | 1:15 h | 687 MB |
| 100k | −1.524098470 | −3.4e-5 | 3 | 4 / 3 / 7 | 3.2e-6 | 1.4e-5 | 2:20 h | 874 MB |
| 200k | −1.524113492 | −1.5e-5 | 4 | 5 / 4 / 7 | 1.2e-5 | 5.7e-6 | 7:18 h | 1422 MB |

(Every run shared the machine with an R4 job on 3 other cores.)
- **No convergence with the cap.** E0 drops 3.4e-5 and then 1.5e-5 per doubling, so it shrinks by only ~2× per doubling and stays well above the 1e-8 target. **All three bounds keep growing with the cap**: holes 4→5, electrons 3→4, total 6→7. Here the bounds are order statistics of the cap, as F1 predicted for conduction-bath systems. Bounds read from any affordable cap are not physics.
- **Cost is super-linear in the cap**: ×1.9 from 50k to 100k, then ×3.1 from 100k to 200k. This is the ranking cost (F2), and it is worse on an f-shell metal than on Fe per determinant: fan-out 30, raw connectivity 77.
- **Physics flag (not caused by the budget removal):** at every cap the solver keeps only the N_imp = 1 sector *because that sector is cap-bound*. Yet N_imp = 0 and N_imp = 2 lie inside the thermal window, 2.6τ and 3.7τ above. For α-Ce, a mixed-valence system, that drops the f⁰/f² admixture from the thermal ensemble. The cap decides this, not the physics.
- The level structure is stable: excitations at +8.89 and +12.5 mRy at every cap, and a fourth state at +13.4 mRy enters the window at 200k.

### R5c. BCC Fe 5-bath (10 imp / 46 val / 6 con), GS only, cap 50k pinned, NEW vs OLD budget 6, 3 ranks
| arm | E0 (N_imp = 5) | discarded PT2 at cap | winner residual PT2 | sector margin over N_imp = 4/6 | wall | VmHWM/rank |
|---|---|---|---|---|---|---|
| NEW | −11.100972999 (2.5e-4 lower) | 9.0e-4 | 2.3e-3 | 0.44 | 89 min | 683 MB |
| OLD b6 | −11.100722122 | 7.5e-4 | 1.7e-3 | 0.62 | 81 min | 685 MB |

- **The budget does not keep Fe below the cap at 50k either**, and both arms show the same pathologies. NEW is better in E0 by 2.5e-4, and OLD is 10% faster.
- **Both arms share the same cap-made physics defects:**
  - **Sector:** the choice between N_imp = 5 and 6 is not separated beyond the truncation error (margin < 1).
  - **Valence:** N_imp = 6 sits 0.8τ above and is dropped from the ensemble.
  - **Degenerate level:** the window keeps 1 state while certification found classes that make ≥ 3, the FeO-type cut of a degenerate level.
- NEW bounds: 6 / 3 / 8 (cap-bound).
- (R1's −11.034 was the first class of the N_imp = 6 sector at 718k. It is not comparable to these numbers.)

### R4. Gapped ladder, GS only, cap 300k pinned, NEW vs OLD budget 6, 3 ranks (complete)
| system | E0 NEW | E0 OLD − NEW | GS dets | NEW bounds h/e/t | discarded PT2 at cap | wall NEW / OLD |
|---|---|---|---|---|---|---|
| SMO3 | −9.601956338650 | +1.5e-7 | 299,962 | 5 / 0 / 5 | 1.3e-6 | 36 / 36 min |
| SMO4 | −11.565096265269 | 0 (same basis) | 299,952 | 5 / 0 / 5 | 4.9e-7 | 65 / 65 min |
| CoO | −38.479194077742 (12 states) | 0 | 20,648 (converged) | 3 / 0 / 3 | — | 1.5 / 1.5 min |
| CrI3 SOC | −7.623354318273 | +1.1e-11 | 300,000 | 7 / 0 / 7 | 1.4e-8 | 78 / 80 min |
| UO2 | −11.724540065703 | 0 (same basis) | 300,000 | 4 / 0 / 4 | 1.6e-4 | 212 / 214 min |
| CeO2 | −7.682145709663 | 0 (same basis) | 300,000 | 5 / 0 / 5 | 1.5e-6 | 146 / 150 min |
| NiO20 | −86.762768542141 (18 states) | 0 (same basis) | 189,304 (final below cap; an intermediate expansion reached 300k, discarding 2.0e-7) | 4 / 0 / 4, not cap-bound | — | 108 / 115 min |

**On gapped systems, OLD and NEW give the same ground state at the same cost.** The budget was not limiting their GS. Where the arms differ at all, NEW is lower.

### R4-GF. CoO full GF, OLD/NEW/REF, cap 300k (GS and GF), 3 ranks, archive meshes (2048 iw / 1001 w)
| arm | excited restriction | largest unit | cap hit | wall | VmHWM/rank |
|---|---|---|---|---|---|
| REF | none | 133,394 | no unit | 13:27 | 499 MB |
| NEW | valence holes ≤ 4 (48 orbitals), all 49 excited bases | 133,394 | no unit | 13:12 | 503 MB |
| OLD b6 | budget 6 | 133,394 | no unit | 13:42 | 498 MB |

E0 −38.479194077742 (12 states) in all three arms. ‖ΔΣ‖ against REF is **0 on both axes for NEW and OLD (bitwise)**. Every Σ is causal (worst Im Σ −2.2e-3 iw, −1.6e-3 w).
- **REF precondition met** (no unit capped). **Verdict: the tightest bound measured (holes ≤ 4, GS bound 3) is still never active.** CIPSI's GF selection does not reach 5 valence holes by itself. This is the second gapped system where NEW = REF bitwise.
- (Shared machine: CeO2 R4 ran on the other 3 cores.)

### R4-GF. SMO3 full GF, NEW, cap 1,000,000 (GS and GF), 3 ranks
E0 is −9.601960135, 3.8e-6 below the 300k GS. The GS is still `[cap-bound]` at 1M (the message reports 0 discarded PT2, the same transient-flag behaviour as NiO10). The NEW **hole bound is 7 at 1M against 5 at 300k**: even on a gapped system, a cap-bound GS gives cap-dependent bounds. **8 of the 16 GF units (all removal units) froze at 1,000,000.** The REF precondition cannot be met within local memory, so **no REF verdict for SMO3** and REF was not run. Σ is causal (worst Im −9.7e-4 iw, −1.1e-3 w). Wall time 2:34 h (GF 1:29 h), VmHWM 1.4 GB/rank.

### R6. FM-Gd pair, GS only, cap 300k, NEW vs OLD budget 6, 3 ranks
| fit | E0 | states | GS dets | NEW bounds h/e/t | cap-bound | wall NEW / OLD |
|---|---|---|---|---|---|---|
| `impmod` (valence only, 14/24/0) | −20.633936120787 | 8 | 63,996 | 2 / 0 / 2 | no | 11.6 / 11.7 min |
| `impmod_unocc` (conduction only, 14/0/32) | −14.501365341551 | 8 | 97,293 | 0 / 2 / 2 | no | 4.5 / 4.5 min |

Both fits are identical between OLD and NEW and converged below the cap. The 8-fold f⁷ (S = 7/2) octet is kept whole. **These are the tightest bounds of the investigation**: 2 → excited bases run at 3, with a weak V. They are the best test of whether +1 is ever too narrow, so both go to full GF with REF (R6-GF).

### R5d. FCC Ni 15-bath (10 imp / 89 val / 59 con, N0 = 8), GS only, cap 50k pinned, NEW vs OLD budget 6, 3 ranks
| arm | E0 (N_imp = 7) | discarded PT2 at cap | winner residual PT2 | sector margin | wall | VmHWM/rank |
|---|---|---|---|---|---|---|
| NEW | −31.561130781 (2.4e-5 lower) | 1.0e-4 | 4.7e-4 | 2.44 | 3:11 h | 880 MB |
| OLD b6 | −31.561107217 | 1.0e-4 | 4.0e-4 | 3.46 | 2:41 h | 872 MB |

- This is the same picture as Fe. **The budget does not keep it below the cap.** NEW is lower in E0, and OLD is about 16% faster (the ranking cost, F2).
- **Both arms share the same cap-made defects.** The thermal window keeps **1 state while certification found classes that make at least 15**, so a large degenerate level is cut, the same violation as FeO, here much larger. N_imp = 6 and 8 lie at 2.3τ and 2.0τ and are dropped from the ensemble.
- NEW bounds: 4 / 4 / 7 (cap-bound, so they are order statistics of the cap).

**Loader gap (R7, 2026-11-07):** the AFM NiO archive carries `DC ground state cap`, which `real_workload` does not consume. It only lowers an *auto* cap (`impmod_interface/lib.py` ~840) and has no effect on a replay with a pinned cap, so the driver whitelists it only when `--cap` is numeric. Add it to the loader fix together with `warm_start` / `warm_tol` / `symm_prec`.

### R7. AFM NiO 5-bath (`AFM_NiO/impmod/5_BathStates_linked_chainGeometry/outdir-2026-11-05/outdir-1`, group `NiUp 1`; 10 imp / 64 val / 0 con), full GF, OLD/NEW/REF, cap 300k, 3 ranks
GS: E0 −45.148667360355, 3 states, bounds 2 / 0 / 2, so the excited bases run at **valence holes ≤ 3**. GF blocks: [0], [1, 2], [5], [6, 7]. The two width-2 blocks seed two orbitals into one excited basis and give off-diagonal Σ. Largest unit 8,469 dets, no cap hit in any arm, about 2.4 min per arm.

‖ΔΣ‖ against REF is **0 on both axes for NEW and OLD (bitwise)**, off-diagonal elements included. Σ is causal (worst Im −1.5e-3 iw, −3.0e-4 w).
- **Verdict: +1 covers the multi-orbital (off-diagonal) units too.** This is the third system where NEW = REF bitwise and the second with a tight bound (≤ 3). Caveat: it is a small model (8k-det units), so the bound has little room to bind.

### R6-GF. FM-Gd full GF (OLD dropped, user 2026-11-07)
- **`impmod` (valence fit): stopped, no verdict.** The N_imp = 7 member's GF (8 states, 64 units) took about 9 h. Its last unit ran alone on one rank in discovered matvec mode, and the machine was oversubscribed overnight. The run then started a **second ensemble member** at E = −20.63171, 2.2 mRy up (f⁶, 48 states in the TRLM window). The second member and REF together would have taken about 2 more days, which the user ruled too long for one data point.
- **Driver blind spot this exposed:** `run_arm.py --gs-only` stops at the *first* GF call, so a GS-only record shows only the first ensemble member. Every R4 record holds one member, because `mixed_valence` kept a single sector in every R4 system. In CrI3, SMO4 and NiO20 the cap made that choice (V5.2b). The R6 GS row ("8 states") is incomplete, though: the valence fit is a two-member mixed-valence ensemble.
- **`impmod_unocc` (conduction fit): stopped by the user** after 4.4 h of NEW's GF (about 55 of 64 units). No REF verdict.
