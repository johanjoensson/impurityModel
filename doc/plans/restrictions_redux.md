# restrictions_redux: more restrictions, tighter cutoffs — measured verdict

> **Status (2026-07-17): measurement complete on the accessible workloads; verdict below.**
> Implemented and shipped: named/overridable freeze-and-chain-window constants and the
> `excitation_budget_restriction` weighted restriction in `basis_restrictions.py`, plus the
> `restriction_diagnostics.py` / `restriction_sweep.py` opt-in measurement harnesses and
> `test_excitation_budget.py`. **No production defaults were changed** — the measurements say
> the current cutoffs are already close to the accuracy bar. The split-cutoff idea (a looser
> *ground-state* amplitude cutoff) was **tested and refuted**: on the metal the ground-state
> eigenvector moves ~4–6e-3 (above the 1e-3 bar) when the cutoff is loosened to 1e-4/1e-3,
> even though ΔE₀ stays ~1e-5 — the energy is flat along the directions the cutoff prunes, so
> ΔE₀ is a misleading proxy and the Green's function (built from those eigenvectors) is not
> safe. The default `slaterWeightMin` is appropriately tight.

## The question

The solver bounds memory by (a) discarding determinants with `|amplitude| < slaterWeightMin`,
(b) occupation-window restrictions — subset windows plus weighted `Σᵢ wᵢ nᵢ` windows such as
`S_z`, (c) chain-freezing of weakly-coupled baths, and (d) the global `truncation_threshold`.
Are there **more** restrictions to enforce, and can the **current cutoffs be tightened**
without moving the spectra / `G(z)` / `Σ(z)` beyond the physical broadening (~1e-3 relative)?

## Method

Two opt-in harnesses driving the production path on real `impurityModel_data.h5` workloads:

- `restriction_diagnostics.py` — runs the production ground state
  (`selfenergy._prepare_solver_basis` + `groundstate.calc_gs`) and measures, on the converged
  basis: enforced-vs-observed occupation-window slack; thermally-weighted determinant weight
  by bath-excitation order / coupling-distance shell / bath energy; and a `slaterWeightMin`
  and excitation-budget scan.
- `restriction_sweep.py` — reruns `calc_selfenergy` under a ladder tightening one knob at a
  time from the production default, reporting ΔE₀, max relative ΔΣ / ΔG on the mesh,
  causality, ground-state basis size and peak RSS.

Workloads: metal **FCC Ni** (`fcc_ni_5`, 59 spin-orbitals), insulator **NiO** (`nio_20`,
20 orbitals — near-exact reference; `nio_15chain`, 144 orbitals — production scale), and
**SMO** (58 orbitals). No `AFM_NiO` archive carries an `H solver` group, so SMO stands in for
the off-diagonal tier. The 144-orbital NiO and the full metal `calc_selfenergy` sweep did not
finish on the 15 GB test machine (see *Caveats*); the metal is characterised by its
ground-state profile and a GS-only cutoff scan instead.

## Findings

### 1. Occupation-window slack is real but one-sided (the chain lower bound)
The chain-freeze **lower** bound on "filled" valence chains carries large never-used slack;
the impurity window and the small blocks are already tight.

| workload | enforced chain window | observed range | unused slack |
|---|---|---|---|
| `fcc_ni_5` | (9, 19) on a 20-orbital valence chain | (17, 19) | **8** (lower) |
| `smo` | (10, 20) on an 18-orbital valence chain | (17, 20) | **7** (lower) |
| `smo` | (2, 4) on a 4-orbital chain | (2, 4) | 0 (tight) |

The chain floor `L//2` sits far below what the ground state occupies. This is the clearest
tightening target and motivates both the chain-window fraction knob and the excitation budget.

### 2. Excitation weight decays steeply and is already bounded by `slaterWeightMin`
Thermally-weighted determinant weight by number of bath excitations:

| order | `nio_20` | `fcc_ni_5` | `smo` |
|---|---|---|---|
| 0 | 97.3% | 29.6% | 57.7% |
| 1 | 2.67% | 36.2% | 35.4% |
| 2 | 0.005% | 26.9% | 6.37% |
| 3 | — | 6.68% | 0.45% |
| 4 | — | 0.59% | 0.014% |
| ≥5 | — | <0.03% | <2e-4 |

The metal spreads to order ~4; the insulators are far more compact. In every case weight
beyond order ~4–5 is already <1e-4 — the amplitude cutoff prunes the high-excitation tail.

### 3. The coupling-distance metric is ~binary in the solver basis
~100% of the excitation weight sits in the **nearest** coupling-distance shell: after the
impurity-diagonalising rotation, essentially every bath couples within the freeze cutoff. The
threshold barely discriminates and there is almost nothing for a *graded distance* weight to
grade — a **negative** result for a perturbation-aware metric (§3b).

## Cutoff frontier

### `nio_20` full self-energy sweep (max|Im G| = 23.5, trustworthy; GS = 80 dets = full space)
| config | ΔE₀ | ΔΣ_rel | ΔG_rel | causality | gs_size |
|---|---|---|---|---|---|
| default (ref) | 0 | 0 | 0 | −7.31e-3 | 80 |
| swmin=1e-6 | 0 | 3.7e-4 | **1.36e-3** | −7.31e-3 | 80 |
| swmin=1e-4 | 1e-14 | 7.1e-4 | **4.02e-3** | −7.31e-3 | 80 |
| chain 0.34 / 0.25 | 0 | 0 | 0 | −7.31e-3 | 80 |
| couple=1e-2 | 0 | 0 | 0 | −7.31e-3 | 80 |

The **Green's function is the binding constraint**: ΔG hits the 1e-3 bar already at
`slaterWeightMin = 1e-6` and clearly exceeds it at 1e-4 — *even though E₀ stays exact*. On
this minimal basis the chain/coupling knobs are no-ops (there is no slack to cut).

### `fcc_ni_5` metal ground-state `slaterWeightMin` scan (E₀ metric, GS only)
| swmin | gs_size | ref/size | ΔE₀ |
|---|---|---|---|
| √ε≈1.5e-8 (default) | 16455 | 1.00 | 0 |
| 1e-6 | 16444 | 1.00 | 4e-7 |
| 1e-5 | 17706 | 0.93 | −2e-6 |
| 1e-4 | 8038 | **2.05** | 9e-6 |
| 1e-3 | 943 | **17.45** | 7.9e-4 |

The **ground-state energy tolerates a far looser cutoff** than the default: 1e-4 halves the
basis at essentially exact E₀, 1e-3 shrinks it 17× while E₀ still moves <1e-3. (The 1e-5 row
is slightly *larger* — the CIPSI fixed-budget selection is non-monotonic in the cutoff.) This
looks like an opportunity — but ΔE₀ is a **misleading proxy** (see the split-cutoff test next).

### Split-cutoff confirmation — REFUTED (the ΔE₀ signal is a mirage)
Because the Green's function is a deterministic function of `(H, psis, es)`, a loosened GS
cutoff is safe for the spectra only if the low-energy **eigenvectors** are unchanged. Comparing
the metal ground state built at the default √ε against looser GS cutoffs (GF cutoff irrelevant
— this is a GS-only eigenvector comparison):

| GS cutoff | gs_size | ref/size | 1 − \|⟨ψ_default\|ψ_loose⟩\| | max\|ΔEₖ\| |
|---|---|---|---|---|
| 1e-4 | 8038 | 2.05× | **4.35e-3** | 9.1e-6 |
| 1e-3 | 943 | 17.45× | **5.67e-3** | 7.9e-4 |

The ground-state **eigenvector moves ~4–6e-3 — above the 1e-3 bar — while ΔE₀ stays ~1e-5**.
Specific observables shift hard (⟨T_z⟩ 0.0037→0.062, a >10× change; ⟨S²⟩ ~1.5%). The metal's
near-degenerate manifold makes the *energy* flat along exactly the low-amplitude directions the
cutoff prunes, so the state rotates ~0.4% within that manifold at almost no energy cost. Since
the GF/Σ are built from these eigenvectors, loosening the GS cutoff would move the spectra above
tolerance. **The ground state is not safely over-resolved; the default cutoff is right.** (On
`nio_20` the same comparison is trivially clean — its 80-determinant GS is the complete space,
so nothing is pruned; the effect only appears where the cutoff actually bites, i.e. the metal.)

## Phase 3 — new restrictions

### 3a. Excitation budget — IMPLEMENTED + WIRED, a real win on metals
`basis_restrictions.excitation_budget_restriction(bath_states, budget, cost_fn=None)` builds a
single weighted restriction `(w, (q_min, q_max))` with `wₒ = ∓cost` on filled-valence /
empty-conduction orbitals, so `Σ excitations ≤ budget`. It composes (AND) with any other
weighted restriction (`S_z`) and widens correctly on a Green's-function sector via
`symmetries.widen_weighted_restrictions`. The uniform default is the excitation-order budget
the §2 profiles motivate; a graded (per-orbital `cost_fn`) form is supported but §3 shows the
distance metric does not discriminate post-rotation.

**Wired into the production drivers (2026-07-18).** `BasisOptions.excitation_budget`
(default `None` = off) is threaded by `calc_selfenergy` and `run_spectra` onto the
ground-state `Basis.weighted_restrictions` via the `build_weighted_restrictions` helper. The
self-energy GF and XAS/PES bases inherit it automatically (the shared
`greens_function._build_excited_restrictions` widens it); RIXS attaches the widened budget
explicitly to its excited-basis clones (`rixs.py`). Exposed on the `spectra`/`selfenergy` CLIs
(`--excitation_budget`) and read from the hdf5 archive. Sz-based GF restrictions were
considered and **not** added — `H` conserves Sz, so the GF closure never leaves the seed's Sz
sector (nothing to prune), and under SOC an Sz window is invalid.

Ground-state gate experiment:

| workload | budget | gs_size | ref/size | ΔE₀ |
|---|---|---|---|---|
| `nio_20` | ≥2 | 80 | 1.00 | 0 |
| `nio_20` | 1 | 60 | 1.33 | 9.4e-5 |
| `fcc_ni_5` | 6 | 16372 | 1.01 | 5.7e-7 |
| `fcc_ni_5` | 5 | 15368 | 1.07 | 8.1e-6 |
| `fcc_ni_5` | 4 | 11477 | **1.43** | 2.5e-4 |
| `fcc_ni_5` | 3 | 4767 | **3.45** | 3.8e-3 |

The insulator's natural GS is already order ≤2, so a budget barely bites. The **metal has real
excitation-order slack**: budget 4 gives 1.4×, budget 3 gives 3.45×. The ΔE₀ column is shown
for scale but — per the split-cutoff refutation above — **ΔE₀ under-states the accuracy cost on
the metal** (the energy is flat along the pruned directions); a budget should be accepted on an
eigenvector/spectral criterion, not ΔE₀. The budget is a genuine *memory* knob for metals and,
unlike `truncation_threshold`, an *a-priori* bound (it constrains the basis before it is built,
not after it overflows) — shipped **off by default** as an opt-in, with the guidance that its
accuracy be checked against the eigenvector overlap on the target workload.

### 3b. Perturbation-aware distance metric — NOT PURSUED
The coupling distance is ~uniform in the solver basis (§3); there is nothing to grade.

### 3c. Auto-enforced conserved charges — NO NEW CONFINEMENT
`symmetries.conserved_subset_charges` returns only the spin split `{N↑, N↓}` on all three
workloads: total `N` (already enforced by the occupation window) plus `S_z`. `S_z` must not be
pinned (spin multiplets — the recorded NiO-triplet regression), and the basis already spans
exactly the thermally-needed `S_z` via `spin_flip_dj`, so even an `S_z` *window* removes
nothing. No additional conserved subset charge exists to confine with. Wired as nothing;
the auto-`S_z` restriction remains available as the tested opt-in it already was.

### 3d. Long chains — the fraction window is the wrong knob; the budget is the lever (2026-07-18)
Prompted by a 25-bath linked-chain NiO DFT+DMFT run (`nio_25chain`, valence chain length 234,
much longer than anything in §1–§3), re-examined on the metal and insulator tiers with a
**rigorous eigenvector-overlap accuracy gate** (`restriction_diagnostics.eigenvector_overlap_experiment`):
the GF/Σ is a deterministic function of `(psis, es)`, so the worst-case rotation of the
ground-state manifold (smallest cross-overlap singular value) is the accuracy criterion — and it
sidesteps the FCC Ni GF over-convergence that makes a production Σ sweep intractable here (6+ min
CPU for a *5-bath* GF).

| workload | knob | gs/window shrink | 1 − fidelity (rotation) | verdict |
|---|---|---|---|---|
| `fcc_ni_5` | budget 4 | 1.43× | 1.8e-4 | **lossless** |
| `fcc_ni_5` | budget 3 | 3.45× | 5.8e-3 | over bar (and > ΔE₀ 3.8e-3) |
| `nio_25chain` | budget ≥2 | 1.00× | 2.2e-16 | lossless but **inert** |
| `nio_25chain` | budget 1 | 2.4× | 1.8e-3 | marginal |

Two measured facts reframe the "tighten the chain restrictions" request:
- **The chain-window *fraction* (`CHAIN_FILLED_HOLE_FRACTION`) never binds** on either tier. The
  `√ε` amplitude cutoff prunes per-chain excitation order to ≤2 before the `L//2` window matters:
  on `nio_25chain` the enforced valence window `(78,156)` has **76 units of unused slack** (the
  basis reaches only 2 holes), and on `fcc_ni_5` `gs_size` is *identical* (16455) at hole-fraction
  0.5 and 0.25. A metal's basis size is the *combinatorial spread of low-order excitations across
  many chains* plus the active conduction side, not deep holes in one chain.
- **The binding lever is the global excitation budget** (§3a). It bounds *total* excitations, which
  handles both the metal's combinatorial spread and long-chain inflation (a budget `B` caps a
  length-156 chain far tighter than `L//2 = 78`). Lossless where `B` exceeds the physical
  excitation order; on the localised insulator the cutoff is already tighter, so the budget is inert.

**New length-independent per-chain cap — `CHAIN_MAX_HOLES` / `CHAIN_MAX_ELECTRONS`** (default `None`
= the historical fraction-only behavior). In `build_excited_restrictions` the tighter of the
fraction floor `int(L·(1−frac))` and the absolute floor `L − CHAIN_MAX_HOLES` wins, so a long chain
no longer inflates the excited window with its length. On the `nio_25chain` excited sector the
valence group (L=146) window goes `(73,146)` → `(144,146)` at `CHAIN_MAX_HOLES=2` — a ~35× tighter
window — and is lossless-by-construction wherever per-chain order ≤ cap (both tiers measured ≤2).
It is the safeguard for the regime the budget shares but the fraction fails outright: **long chains
/ small-gap systems where the amplitude cutoff no longer prunes the deep chain by itself**. Shipped
off by default, same discipline as §3a.

### 3e. Graded hopping-derived three-zone chain restriction — IMPLEMENTED (2026-07-18)
The `CHAIN_MAX_HOLES` cap (§3d) is a two-zone (free/capped) special case. The **motivating**
observation: the GF/block-Lanczos basis-build has *no amplitude importance ranking* of the
determinants it generates (unlike the CIPSI ground state, which drops low-amplitude ones), so the
GF basis must be pruned a priori by the occupation restrictions. `CHAIN_GRADED_RESTRICT` (opt-in,
default off) does this with a physically-motivated **three-zone** split derived from the block
off-diagonal hopping terms, reusing the existing accumulated-hopping metric: `_impurity_coupling_distance`
already returns `dist(o) = -log(∏|h_ij|/h_max)` along the best impurity→`o` path, so
`a(o) = exp(-dist(o))` is the accumulated hopping amplitude. Two thresholds on `a(o)` define:

- **Head / free** (`a ≥ free_cutoff = COUPLING_CUTOFF_DEFAULT`) — unrestricted;
- **Intermediate** (`freeze_cutoff ≤ a < free_cutoff`) — capped at `CHAIN_INTERMEDIATE_MAX_EXCITATIONS`;
- **Deep / frozen** (`a < freeze_cutoff`) — hard-pinned to the reference occupation (0 excitations).

`freeze_cutoff` is **auto-calibrated** to the amplitude cutoff as
`slater_weight_min ** CHAIN_FREEZE_WEIGHT_EXPONENT` (default 0.5): freeze exactly where a single
excitation's amplitude `~ a(o)` can no longer survive `slater_weight_min` — "freeze the deep chain
where it cannot matter". A guard falls back to the §3d binary window when a loose cutoff leaves no
gap (`freeze_cutoff ≥ free_cutoff`). `slater_weight_min` is threaded into `build_excited_restrictions`
from all four call sites (GF `greens_function._build_excited_restrictions`, GS `groundstate.calc_energy`,
`spectra.calc_spectra`, `rixs`), so both the GS and GF/spectral builds use it.

Measured (2026-07-18):
- **A-priori GF-window prune** — on the real `nio_25chain` excited sector, block-0's far valence
  chain (L=146) goes from the binary `(73,146)` (up to 73 holes) to **36 orbitals hard-frozen
  (0 holes) + 110 capped at ≤2**, with block-1's 4 coupling orbitals capped at ≤2: the reachable
  far-region excitation count drops from 73 to ≤4.
- **Lossless on the ground state** (eigenvector-overlap gate): rotation `2.2e-16` on `nio_25chain`
  and `2.8e-13` on `fcc_ni_5`, with `gs_size` unchanged on both. The GS is *inert* because it is
  already amplitude/CIPSI-pruned and its weight sits in the free head zone — confirming the graded
  zones touch nothing with weight; the payoff is the un-ranked GF basis-build. The global
  `excitation_budget` (§3a) composes with the per-zone windows (conjunctive).

Shipped **off by default** (`CHAIN_GRADED_RESTRICT=False` reproduces the §3d binary behavior
exactly). Recommended for the long-chain / small-gap GF regime; validate on the target system's
Σ / eigenvector overlap before enabling as a default.

## Recommendations

1. **Do not split the amplitude cutoff, and do not loosen `slaterWeightMin`.** The split-cutoff
   idea was tested (§split-cutoff confirmation) and refuted: loosening the ground-state cutoff to
   1e-4 changes the metal ground-state eigenvector by ~4e-3 (above the 1e-3 bar) with ΔE₀ only
   ~1e-5. The energy is flat along the pruned directions, so ΔE₀ badly under-reports the cost;
   the Green's function is built from those eigenvectors and would move above tolerance. The
   default `√ε` is correct.
2. **Excitation budget as an opt-in metal knob.** Ship `excitation_budget` (default off): it is
   a real *memory* lever on metals (1.4× at budget 4, 3.45× at budget 3) and the principled
   a-priori replacement for the one-sided chain lower-bound slack (§1). Judge its accuracy on the
   target workload by the eigenvector overlap / spectral deviation, **not** ΔE₀ (same mirage as
   #1).
3. **The chain-window *fraction* is the wrong knob; use hopping-derived per-chain caps for long
   chains (§3d, §3e).** The `L//2` fraction never binds — the amplitude cutoff prunes per-chain
   order first — so narrowing the fraction is a no-op. For the long-chain / small-gap regime where
   the cutoff no longer prunes the deep chain, prefer the **graded three-zone restriction**
   (`CHAIN_GRADED_RESTRICT`, §3e): a free head, a capped intermediate band, and a hard-frozen deep
   zone, with the freeze boundary auto-calibrated to `slater_weight_min` from the accumulated
   hopping. It is the a-priori pruning the GF path needs (no amplitude ranking there) and measured
   lossless on the GS eigenvectors of both tiers; `CHAIN_MAX_HOLES` (§3d) is its two-zone special
   case. The global budget (#2) composes and bounds the total. Leave the coupling cutoff alone.
   All three ship off by default — enable per system after an eigenvector/Σ check.
4. **Note 3b is superseded on chains.** The graded distance metric was negative in the *rotated
   star* basis (§3b), but the *chain* geometry has genuinely graded accumulated hopping, which is
   exactly what §3e exploits. Default-on 3c (conserved charges) stays negative on every workload.

Net: the current cutoffs are already well-tuned; the genuinely new levers are the opt-in
excitation budget for metals (§3a) and the hopping-derived graded per-chain restriction for the
long-chain / small-gap GF regime (§3e). No production defaults changed.

## Caveats

- The full metal `calc_selfenergy` split sweep did not complete on the 15 GB test machine — the
  metal Green's function hits the documented FCC Ni over-convergence pathology (the "m≈833
  divergent recurrence"; ~2 h for one 3-run set), and SMO's full GF was similarly slow. The
  split-cutoff question was instead answered *directly and more cheaply* by comparing the
  ground-state **eigenvectors** between cutoffs (the GF is a deterministic function of them) — a
  GS-only comparison that needs no GF and is the rigorous test. The 144-orbital `nio_15chain` GS
  and the pure-Python `O(N_det)` diagnostic loop also do not scale past ~2e4 determinants.
- **`ΔE₀` is a misleading accuracy proxy for correlated metals** — the headline lesson. The
  energy is variational and flat along low-amplitude directions, so a determinant cutoff (or an
  excitation budget) can look "exact to 1e-5 in E₀" while rotating the wavefunction ~0.4% and
  shifting observables like ⟨T_z⟩ by >10×. Accept any basis-shrinking knob on the eigenvector
  overlap / spectral deviation, never on ΔE₀ alone.

## Files

- `src/impurityModel/ed/basis_restrictions.py` — `COUPLING_CUTOFF_DEFAULT`, `MIN_DIST_DEFAULT`,
  `CHAIN_FILLED_HOLE_FRACTION`, `CHAIN_EMPTY_ELECTRON_FRACTION` (behavior-preserving named
  constants; `_USE_DEFAULT` sentinel keeps `coupling_cutoff=None` meaning the legacy metric);
  `CHAIN_MAX_HOLES` / `CHAIN_MAX_ELECTRONS` (default `None`, length-independent per-chain caps, §3d);
  `CHAIN_GRADED_RESTRICT` / `CHAIN_INTERMEDIATE_MAX_EXCITATIONS` / `CHAIN_FREEZE_WEIGHT_EXPONENT`
  (default off, the graded three-zone restriction + its `_emit_graded_chain_window` helper, §3e);
  `excitation_budget_restriction`. `build_excited_restrictions` gained `slater_weight_min` (threaded
  from the GF/GS/spectra/RIXS call sites for the freeze-cutoff auto-calibration).
- `src/impurityModel/test/test_excitation_budget.py` — excitation-budget, `CHAIN_MAX_*`, and graded
  three-zone cap tests.
- `src/impurityModel/test/restriction_diagnostics.py`, `restriction_sweep.py` — opt-in harnesses
  (`eigenvector_overlap_experiment` = the rigorous accuracy gate; `nio_25chain` long-chain workload;
  `CHAIN_MAX_*` + `excitation_budget` threaded through the sweep).
