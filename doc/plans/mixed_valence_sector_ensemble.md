# Mixed valence in impurityModel: what works, what is missing, and how to add it (rev. 2)

**Status (2026-10-30):** planned, not started. **Rev. 2 repurposes the existing `mixed_valence` flag as the ensemble switch, `"auto"|true|false`, and removes its seed-widening behaviour. See "Repurpose `mixed_valence`" below, which supersedes every mention of a new `sector_ensemble` flag.** To be merged with the ground-state determination plans before implementation (see §J "Joint decisions"). Rev. 2 folds in five adversarial reviews (code/MPI, architecture, numerics/memory, physics, UX). This file is rev. 2 (the stored rev. 1 copy was overwritten). Doc consistency pass 2026-11-02 (J9 of `gs_review_2026-10.md`).

## Context

**(A) Valence fluctuation at fixed total N (d^n + d^{n+1}L) works, with conditions.**
- The GS restrictions are built with `imp_change=None` (`groundstate.py:168` → `basis_restrictions.py:541,589-596`), so the impurity window is open.
- `BasisOptions.mixed_valence` only widens the seed, and it disables plan-store reuse (`solve_sector`'s `single_occupation`).
- **Condition 1:** the bath needs levels near E_F with spacing below Γ. Without them, (A) gives a cluster singlet, and fractional ⟨N⟩ can appear only through (B) as a thermal artifact.
- **Condition 2:** `excitation_budget=4` is "measured lossless" only on 3d oxides; it is unvalidated for 4f Kondo/Sm/Yb mixing. l=3 shells and `dc_MLFT` are otherwise supported.

**(B) Several total-N sectors inside `energy_cut(tau)` are not supported, and the code does not warn.** `solve_ground_state` keeps only the winning sector, so ρ, ⟨N⟩, G, Σ, spectra, χ and the occupation DC all drop the others.

Measured with one rank on the `test_fixed_dc.common_kwargs` 4-orbital fixture (tau=1e-2): the code reproduces the exact *canonical* ⟨N_imp⟩, but misses the exact grand-canonical (GC) value by up to 0.37 e within about 0.1 of the crossing (table in rev. 1).

The physics reviewer confirmed this is purely the two-level Boltzmann population n_GC ≈ n_can + g e^{-x}/(1+g e^{-x}), with g=2. Hybridization contributes only the 0.011 baseline.

**Physics verdict.** For a DMFT bath (a reservoir at mu), GC at T>0 is the correct ensemble, as in finite-T ED-DMFT (Capone/de' Medici/Georges 2007; Liebsch–Ishida 2012). Today's hybrid (pick the sector by the T=0 rule, then average canonically at finite T) is a bug. But GC does **not** make thermal sector mixing physical mixed valence. As τ→0 it collapses back to a step. A fractional ⟨N⟩ produced mainly by inter-sector weight is a bath-discretization symptom. For closed-cluster models (MLFT XAS of ionic insulators, molecules), fixed N is the correct model.

**Numerics verdict (measured).** The weights are exponentially sensitive to sector-relative variational error: δ⟨N⟩ ≈ −p(1−p)ΔN·δE/τ.
- At the default τ=0.002, a one-sided error δE=1e-3 in one sector moves ⟨N⟩ by 0.11; δE=1e-2 moves it by 0.59.
- CIPSI/cap errors are always ≥0 and differ between sectors. SMO walk residuals are 1.3e-3 to 6e-2, and uncertified classes can sit 0.17 eV high.
- So the harder sector is systematically under-weighted. **Membership and weights must be residual-aware, and must refuse cap-bound or uncertified members.**

## Design (rev. 2)

### D1. One ensemble seam in `groundstate.py`, a callback (not a generator), consumed by every driver
- `solve_ground_state` and `calc_gs` keep their single-sector contract: the lowest-energy member, plus the warning.
- New: `plan_ensemble(search, tau, ...) -> EnsemblePlan`. It holds the broadcast member list `[(trial_N0 dict, E_s, residual_pt2_s, certified_s, cap_bound_s)]`, a fixed order, and each member's window.
- New: `for_each_member(plan, consume)`. It runs `solve_sector(Within(...))` per member, calls `consume(member, SectorSolve)`, then drops references.
  - **Never `basis.free_comm()`**: the GS basis holds `COMM_WORLD` itself (`manybody_basis.py:474,557-567`).
  - A callback is required because a generator abandoned on one rank leaves the other ranks inside a collective.
- Every driver calls `for_each_member` once with its own consumer: `calc_selfenergy`, `run_spectra`, the susceptibility workflow, and `_evaluate_occupation_and_energy_at_mu`. The DC and Σ paths therefore see the same ensemble, which preserves the "single definition of the ground state" invariant (`groundstate.py:944ff`).

### D2. Merge normalised per-member results; never thread a global e0 into kernels
- Each member runs today's code unchanged and returns normalised X_s with (e0_s, Z_s), where e0_s is the sector's own realised thermal minimum.
- Merge with c_s = Z_s·exp(−(e0_s − e0)/τ) / Σ_t(…) and X = Σ c_s X_s, on rank 0. G is gathered to rank 0, so the merge happens there.
- **Linear quantities, merged this way:** G (add and remove sides kept separate), estimator-family matrices, moments M, ρ, and P(N) histograms.
- **Nonlinear quantities** are applied once to the merged inputs: `get_Sigma_moments`, Dyson, the Curie coefficient, the charge, spin and orbital static χ (`observables.py:1200ff,1280-1290`), P(N) variance, N_eff/participation, entanglement entropies (accumulate per-N_imp RDM blocks), `sisb_z_connected`, and `energy_decomposition`.
- **Per member only:** term symbol, `is_j_sharp`, and the Casimir→quantum-number labels. The combined Casimir ⟨·⟩ is printed only as a number.
- For one member c=1.0 exactly, so the result is bit-identical with **no separate code path**. This replaces rev. 1's short-circuit.
- `average.py` gains only the scalar helper `sector_weights(e0_s, Z_s, tau)`, which keeps the layer rule intact.

### D3. Membership: residual-aware, MPI-safe, and checked after the solve
- `find_ground_state_sector` returns `SectorSearch.solved: list[(trial_N0, E_s, residual_pt2, certified, truncation_report)]`. `energy_cache` today stores only `(E, report)` (`groundstate.py:710`), so the residual and the certification must be added.
- Admission rule: `E_s − residual_s − E_min < cut`.
- Refusal: if any member or near-member has `residual > c·τ` (c≈0.1), or is cap-bound or uncertified, refuse with an actionable error, or allow it under an explicit override. Record a per-member error bar p(1−p)ΔN·residual/τ.
- Always make sure the winner's ±1 neighbours are solved. The legacy dN branch (`groundstate.py:868-898`) can miss one. Extend to N±2 only while the edge member is inside the window. Expose the walk's `get_energy` for this; it is a closure today.
- Moot (review J1): nothing is frozen on the Σ/DC path, so no ensemble ever crosses frozen occupations. A bath-less deep shell is a seed window (`basis_generation.SeedWindow`, step 11.0b); it is held out of the walk, so every member carries the same deep-shell count and `sector_key` pins it identically.
- **After the loop**, compare each realised e0_s and the absolute window `e0 + cut` against the walk's values. Filter returned states in absolute terms on rank 0 and broadcast. If the realised minimum moved by more than the tolerance, re-solve the affected members or raise. "Dominant member" means the lowest realised energy, and the weights are reported separately.
- Walk energies are broadcast and therefore exactly replicated; realised `es` are replicated only to roundoff. Every decision taken from `es`/Z_s is made on rank 0 and broadcast.
- Refuse `sector_ensemble` when `energy_reference` is not `fermi`. Legacy h0 loads as `"unknown"` (`h0_format.py:595`); the existing refusal covers only the c2i path.

### D4. Selfenergy-driver specifics (`selfenergy.py`)
- `_drop_low_weight_manifolds` (`:465-503`) and `check_thermal_weight_cutoff` (`gf_diagnostics.py:334`): decide on rank 0 with the global e0 and Z, then broadcast. Only the *global* ground manifold is exempt.
- Moments (`:847`), `_check_gf_physical` and the per-member diagnostics run inside the consumer. `gf_report` merges per member, and unit IDs become `(sector, unit)`.
- `_settle_retry` (`:323-371, 412`) cannot re-solve freed members: **decision S** below.
- Auto GF cap: resolve it once, for the dominant member, and reuse it for every member (`basis._auto_gf_caps`, `gf_units.py:228,264`). Free members at a synchronised point. RSS left over from member k−1 shrinks member k's measured budget, so log both budgets in verification.
- Output: `gs_energies`/`rhos`/`thermal_rho` in HDF5 and the result dict (`scripts/selfenergy.py:325`, `selfenergy.py:865-883`) get member labels, global e0 and the c_s. Truncation and convergence reports take the worst case over members.

### D5. Double counting
- Occupation DC: `dc_search` is sector-first. `sector_of`/`nominal_sector` (`dc_criteria.py:2356`), the "charge-sector boundary" verdict (`dc_search.py:285-400`) and `_dc_chi`'s `in_sector` veto (`:2404`) all assume a step. Under the flag, add a **continuous mode** (`sector_of=None`) with its own oracle. This is an algorithm change in layer 1.
- `cap_bound_at`/`memory_bound_at` are OR-ed over members.
- **Physics guard (decision P):** a root reached mainly through inter-sector weight lands the dc at a level crossing to within about τ. The result depends on τ, and the DMFT loop can oscillate. Split Var(N_imp) into intra-sector (quantum) and inter-sector (thermal) parts, and record both.
- Peak/gap DC: if N±1 is itself a member, the criterion is ill-posed, so warn or refuse in the DC record.
- With the flag off, `_report_unattainable_target` (`dc_search.py:306-318`) and the `DoubleCountingUnreachable` text name `sector_ensemble` and give the predicted GC n at the boundary.
- `dc_MLFT`: allow a non-integer valence `n_val_i` (the formula is linear, and Haule's nominal n0 is real). Keep the filled-core integer check. Integer nominal/MLFT DC stays the default; FLL at the actual n is not made the default.

### D6. Warning (always on) and user experience
- The warning has two levels:
  - an estimated dropped weight below about 1e-2 goes to a JSON/DC-record note only;
  - above that, a rank-0 WARNING at default verbosity, also sent to `sys.__stderr__` (RSPt rank-0 stdout goes to a file, `impmod_interface lib.py:1209`).
- It prints once per `calc_selfenergy` and for the *accepted* DC mu only. Per-trial findings are aggregated into the DC record.
- ΔE is shown in the run's units. It carries a "lower bound (degeneracy g ignored)" label, or uses the lowest-manifold g if that is cheap. It adds a cap caveat when the walk or truncation report is bound.
- The advice depends on the driver: Σ/DC say "set sector_ensemble"; spectroscopy says "competing charge state near-degenerate; check nominal/Δ, or set the flag for a reservoir bath".
- No rename: `mixed_valence` is repurposed (see "Repurpose `mixed_valence`"), so there is no `RENAMED_KEYS` alias. Keep the `mv` archive attribute and solver token readable forever. Relabel the RSPt header. Fix the schema description and kind (today an electron count, `schema.py:1020`; becomes the tri-state).
- Labels: ensemble rows read "charge sector (seed N_imp=…, N_total=…)". Docs keep P(N_imp), meaning (A), distinct from the sector weights, meaning (B).

### D7. Wiring (from commit 1 of the flag, not deferred)
- Schema `Key` (`inputformat/schema.py`), `inputformat/build.py:706`, the three CLIs that build `BasisOptions` (`scripts/selfenergy.py:453`, `scripts/spectra.py:274`, `scripts/susceptibility.py:145`), and archive read/write (`model.py:1225-1260`, `test/support/real_workload.py`).
- An impmod_interface solver-line token, archive attribute and header label (`lib.py:358,514,561`). This is a Python-side change in the sibling repo, not RSPt.
- `required_features` entry so old readers refuse the key instead of running canonical.
- **Owed to the `schema_version` bump and `required_features` before step 12 starts (J8):** the MV fields (tri-state `mv`, `charge_sector_ensemble`), the MS archive datasets (`Spherical shell l`, per-shell N0), the reinterpreted `mv` token, and the P10 recorded cap. RSPt's `mixed_valence={0: mv}` (`lib.py:439`) is keyed on shell 0 and breaks under MS unless the scalar tri-state lands first.
- Until a driver is migrated, it **raises** on `sector_ensemble=True`.
- `gs_statistics` JSON: add `schema_version` and a `charge_sector_ensemble` list. Don't reuse `sector_weights` (`gs_statistics.py:715`, impurity-N blocks). `per_state` gets a `sector` tag and a global `energy_rel`.

### D8. KMS diagnostic
Pointwise A_rem = e^{−ω/τ}A_add fails under broadening δ≫τ and overflows for ω ≳ 700τ. If kept, check Lehmann/Ritz pole–residue pairs for |ω| < cut, with a tolerance tied to the member residuals (**decision K**).

## User decisions (2026-10-30, after the reviews)
- **Default:** set per driver. `sector_ensemble = "auto" | true | false`:
  - selfenergy, DC and RSPt default to `auto`, which switches on when a member is above the warning level;
  - get_spectra, RIXS and χ default to `false` (closed cluster);
  - an explicit value overrides the default everywhere.
- **DC guard:** converge, but flag loudly. The DC record carries the intra-sector/inter-sector variance split, and a warning says "bath lacks states near E_F".
- **Settle retry:** per member. The causality check and retry run on each member's own Σ_s before it is freed. Peak memory stays one member. The merged Σ is checked again, and a failure there is reported, not retried.
- **In scope:** the KMS pole–residue check (commit 12), and the impmod_interface solver token, archive attribute and header label (D7).

## Repurpose `mixed_valence` (user, 2026-10-30) — supersedes `sector_ensemble` everywhere above
`mixed_valence` was always *meant* to handle several charge sectors. Its current behaviour (seed widening at fixed total N, via `generate_initial_basis`) was never intended, nobody uses it, and nobody should. So:

- **There is no new flag.** `mixed_valence` becomes the ensemble switch. Read `sector_ensemble` in D1–D8, the decisions and the tests as `mixed_valence`. Its per-driver defaults are as decided above: Σ, DC and RSPt `auto`; spectra, RIXS and χ off.
- **Remove the seed-widening behaviour entirely**, including:
  - the `mixed_valence` parameter and the `delta_*`/`excitation_budget` folding in `basis_generation.generate_initial_basis` (`:32,54-75,98-118`);
  - `Basis.__init__`'s `mixed_valence` argument (`manybody_basis.py:358-460`) and `basis.mixed_valence`;
  - the `single_occupation` plan-store exemption in `solve_sector` (`groundstate.py:~287`), so plan reuse becomes unconditional;
  - the `sector_key` docstring branch "nominal +/- mixed_valence" (`groundstate.py:68`);
  - the `mixed_valence` positional arguments of `build_basis_and_solver`, `solve_sector`, `calc_energy`, `find_ground_state_sector` and `solve_ground_state`. The flag moves to the ensemble seam (D1), and the sector solves no longer see it;
  - the `solver_basis`/`symmetries.py:259` keying on it.
- **Value and compatibility:**
  - **Value (decision V, user):** `"auto" | true | false`. Membership comes from the thermal window plus the residual guard (D3); `auto` switches on only when a member is above the warning level. The RSPt token becomes `mv auto|on|off`.
  - Archives and the RSPt `mv` token currently carry `None` or an int. Map `None` → the driver default; `0` → off; an int > 0 → on, with a one-line note that the old seed-width meaning is gone. That keeps replay of every existing archive canonical as it was, because they all hold `None` or 0.
  - The `mv` token stays and accepts the new values (`impmod_interface lib.py:256,358,439,514,561`, plus `test_parse_solver_line.py:109` and `interface_test.py:82-93`).
- **Test fallout.** About 30 test and support files pass `mixed_valence`. Most pass `None`/`0`/`False`; drop the argument there.
  - **These pass non-zero, so their seeds change:**
    - `test/support/_nio_workload.py` (default `mixed_valence=1`, `:216,297`);
    - `test_block_lanczos_perf.py` (`MIXED_VALENCE`);
    - `test_bicgstab_perf.py`.

    Re-baseline any recorded numbers in these benchmark and workload fixtures, and note it in the commit.
  - `test_fixed_total_n.py`, `test_sector_key.py` and `test_sectorization.py` test the old seed semantics. Delete the cases that only test widening, and keep the fixed-total-N invariants.
- `total_charge_slack` (one basis spanning several N) is also unused. Keep it internal, and put it in the docs as *not* the mixed-valence mechanism, since the reviews rejected a single mixed-N basis.
- Commit 1 changes to "docs + remove the seed-widening behaviour of `mixed_valence`" (one concern: the behaviour that should not exist). It runs `-n 1/2/3` because it changes seeds. Commit 5's flag wiring repurposes the existing key instead of adding one. Commit 13 drops the rename.

## Commit series (rev. 2)
Each commit runs the targeted tests. **`-n 1/2/3` runs on every commit that touches collectives** (CLAUDE.md:38-40), and the full gate runs at the end.
1. Docs and docstrings: (A) and its conditions; `mixed_valence` is a seed knob; the fixed-N limitation; fix `dc_criteria.py:44/384` (`tau/100`) and `:2029`; correct the "absolute is refused" wording.
2. `SectorSearch.solved` with residual and certification; guarantee winner ±1 neighbours; expose the walk's `get_energy`. No behaviour change.
3. The always-on warning (needs commit 2), with JSON and DC-record fields.
4. A bit-identical split of `report_ground_state` into a per-sector `measure` (collective, linear partials) and a rank-0 `finalize`. Proved bit-identical on existing fixtures and NiO.
5. `plan_ensemble`, `for_each_member`, `sector_weights`, the post-solve window check, and the flag wiring (D7). Every driver raises on the flag.
6. GS report under the flag.
7. Selfenergy consumer: G/M/ρ merge, global drop rule, diagnostics, settle-retry decision, and GF cap reuse.
8. DC continuous mode, the physics guard, peak/gap warnings, and the plateau message naming the flag.
9. Susceptibility consumer (Curie from raw moments).
10. Spectra consumer. This needs `simulate_spectra` refactored to return arrays instead of writing HDF5 mid-run (`spectra.py:319-534`), and XAS components expanded per member before merging.
11. RIXS consumer.
12. The KMS pole check, if decision K keeps it.
13. The user-guide section (no rename: the key is repurposed), an `examples/` input, and regenerating `input_format.md` and `examples/ground_state_statistics.json`.

## Verification (sharpened by the numerics review)
1. **Exact oracle, extended:** a fixture with 2 impurity orbitals and 2–3 bath levels (Fock ≤1024) at τ=2e-3. Pick mu so that one member has an excited state at 0.3–0.9 of its reduced window. Compare ⟨N⟩ **and the per-sector Z_s/Z** against the exact values, truncated at the same cut.
   - Mutation: give every member the full cut; the test must fail.
2. **Perturbation:** monkeypatch one member's walk energy, raising it by x or marking it capped. The error bar must cover the induced δ⟨N⟩, the post-solve check must fire, and the refusal must trigger.
3. **Three-member fixture** (small U): N−1, N and N+1 members, to exercise the N±2 extension.
4. **Oracles for the nonlinear quantities:** the Curie coefficient and χ_charge against exact GC values. Mutation: weighted-sum them per sector instead; the test must fail.
5. Run with `gf_min_weight` set and the flag on: the global drop rule must apply.
6. **Merge helper:** shift e0 by −10τ, 0 and +50τ, and permute the member order; results must be identical.
7. **DC:** n(mu) must be monotone on a τ/5 grid across the crossing at `-n 3`, with an empty rank in the minority member. A fractional target converges in continuous mode, and the physics guard fires.
8. **Oracle with conduction bath levels near E_F:** n(dc) must be continuous at fixed N, case (A), and the guard must *not* fire.
9. **One-member bit-identity** for every driver: E0, ρ, G and Σ. Stdout and JSON are excluded, since they gain fields.
10. **MPI:** every rank sees the same member list and order. Drivers not yet migrated raise. Use `-n 1/2/3`.
11. **Memory:** log the expand budget and GF budget at each member's entry (two members, NiO-scale) to confirm peak memory is one member.

## J. Joint decisions with the GS-determination plans
1. One GS spec built from `BasisOptions`; the flag lives at the ensemble seam (D1), not in `solve_ground_state`'s signature. This fixes the existing parameter drift: spectra drops `slaterWeightMin`/`chain_restrict`, and susceptibility hard-codes `chain_restrict=False`.
2. Walk semantics: the N±2 extension and the solved set as a first-class output, decided together with the greedy local-minimum issue (`gs_redesign.md` P7) and its oracle.
3. Which energies decide membership, and at what accuracy (the DC walk can run at a different `DC_E_PT2_TOL`, P5).
4. Cost: decide membership from Lowest energies alone; solve the thermal ensemble only at the accepted mu; reuse the DC's final ensemble in `calc_selfenergy` (R1/R6). K× applies to both DC trials and the GF stage, which is hours on SMO.
5. Cap probe and manifold completeness for the N±1 members (`gs_cap_measured_inputs.md:42-51`).
6. Plan-store traffic with K members per shift (C4/C5).
7. Opt-in vs always-on, revisited after item 4.
8. Versioning of the `gs_statistics` JSON, the DC-record vocabulary and the archive attributes.

## Critical files
`ed/groundstate.py`, `ed/gs_report.py`, `ed/gs_statistics.py`, `ed/observables.py`, `ed/average.py`, `ed/selfenergy.py`, `ed/greens_function.py`, `ed/gf_units.py`, `ed/gf_diagnostics.py`, `ed/dc_criteria.py`, `ed/dc_search.py`, `ed/atomic_physics.py`, `ed/susceptibility.py`, `ed/spectra.py`, `ed/rixs.py`, `ed/get_spectra.py`, `ed/model.py`, `inputformat/{schema,reader,build}.py`, `scripts/{selfenergy,spectra,susceptibility}.py`, and the sibling `impmod_interface/src/impurityModel_interface/lib.py`.
