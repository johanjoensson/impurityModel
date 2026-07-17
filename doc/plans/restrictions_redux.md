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
3. **Do not change the chain window or coupling cutoff either.** The chain/coupling knobs are
   no-ops on a compact basis; only the chain lower-bound has slack, and the budget addresses that
   more principledly than narrowing the window.
4. **Drop 3b (graded distance) and default-on 3c (conserved charges)** — measured negative on
   every accessible workload.

Net: the current cutoffs are already well-tuned; the only genuinely new lever is the opt-in
excitation budget for metals. No production defaults changed.

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
  `excitation_budget_restriction`.
- `src/impurityModel/test/test_excitation_budget.py` (6 tests, serial + MPI).
- `src/impurityModel/test/restriction_diagnostics.py`, `restriction_sweep.py` — opt-in harnesses.
