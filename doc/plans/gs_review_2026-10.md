# Ground-state determination review (2026-10-30, rev. 2)

**Status:** review and plan. Nothing in the code has changed.
- **Rev. 1** went through three read-only Opus reviewers. Their corrections are marked *(review)*.
- **Rev. 2** went through six (physics, numerics/memory/performance, architecture, MPI, code/testing, UX/cross-plan).
  Their corrections are marked *(r2)*. Several rev. 1 claims did not survive and are corrected in place: M1's fix,
  B1, B2's cause, B4's size, B5's rule, the SMO "thermal worse than walk" reading, and the "printed once prints
  twice" item.

**Decisions taken** (user): J1, what "frozen" means (§6); O1–O5 (§8).
Requirement **SYM** (user): symmetry over energy convergence, non-negotiable (above; drives S2 and O3).
**Still open:** O6's parameters.

> **Requirement SYM (user, 2026-10-30, non-negotiable).** The ground state, and every state of the thermal manifold,
> must have **all the symmetries of the Hamiltonian**, even at the price of a less converged energy. Every selection,
> cap, memory-guard, trim or amplitude cut acts on whole symmetry orbits (S± partners, symmetry images). When the
> budget cannot hold the symmetric basis, every member loses accuracy equally; a partial orbit is never kept.
> Fixes that trade symmetry for energy are rejected.

Evidence lives in `debug/gs_review_2026-10/` (gitignored):
- `findings.md`, the running log;
- `reviews_round1.md`, the condensed rev. 2 reviews;
- the harness scripts and `logs/`;
- the reviewers' probes in `review_{physics,numerics,mpi,code}/`.

Baseline: the GS test subset passes at 1, 2 and 3 ranks (508, 515 and 515 passed; `logs/baseline_*`). **None of the
defects below is caught by the current tests.**

## 1. What each consumer needs

| consumer | needs | gets today (printed arguments, `logs/parity_*.log`) | defect |
|---|---|---|---|
| self-energy / GF | the complete thermal manifold; the basis and cap the GF stage inherits; the *same state* the DC search measured | TOML options: chain_restrict=True, excitation budget, swmin 1.5e-8, dense_cutoff 500, tolerances | cap re-derived per call (P10); thermal manifold can be incomplete (C7, S1) |
| spectra (XAS/PES/RIXS) | the complete initial multiplet at T with the core pinned; dichroism is sensitive to every degenerate copy | swmin **0** (calc_gs default), chain_restrict **False**, dense_cutoff **1000**, no mixed_valence or tolerances; weighted restrictions built at swmin 1.5e-8. RIXS reuses this GS and widens its restrictions (`rixs.py:597`). | user `BasisOptions`/`SolverOptions` partly ignored (`run_spectra` takes no `SolverOptions`, `get_spectra.py:233`); inconsistent swmin (P2); `energy_cut` dead (P4) |
| DC peak/gap | E(N±1) per sector, certified, at *equal* accuracy with error bars (a one-sided cap error in N−1 biases the gap midpoint) | walk at `dc_criteria.py:498`, then **`solve_sector(Lowest)` directly** at `:577` (no walk, no thermal solve); bath-less groups frozen (wrongly, P1a); `DC_*` tolerance knobs | P1a, P5, P10, C4, **S1** (a closed-shell N±1 can return an excited level) |
| DC occupation | a smooth ⟨N⟩(μ): the full thermal GS at every μ | `solve_ground_state` at `:2052`; bath-less groups frozen (P1a); `basis.e_pt2_tol` only, ignoring `DC_*` (`:~2061`); hard-coded `MPI.COMM_WORLD` | P1a, P9, B8 |
| susceptibility | complete degenerate manifolds (the Curie split depends on the degeneracy grouping) | chain_restrict hard-coded **False** (`susceptibility.py:495`; `BasisOptions` defaults True), **no** weighted restrictions, default tolerances | different variational space from Σ (P3); degeneracy rules (H3) |

The cap probe (`gs_connectivity.py:94`, via `build_probe_seed`) and `test/support/dc_weights.py:160` also enter the
solver directly. A1 must cover all of these entries, not only `calc_gs`/`solve_ground_state`.

The two other plans (§6) add needs:
- **Ensemble default (MV).** Self-energy, both DC families and RSPt default to `mixed_valence = auto`; spectra, RIXS
  and χ default to off. *(r2, physics)* χ `--from-archive` reads a Σ/DMFT archive whose bath is a reservoir, and its
  charge Curie term is exactly what inter-sector weight feeds. The default should follow the input's provenance
  (O4, user: χ and spectra read the setting from the archive, else use their own default). The occupation DC also needs a continuous ⟨N⟩(μ) across a sector crossing (MV D5).
- **Multishell (MS).** The self-energy and DC paths need `shell_ls`, a per-shell N0, and DC shifts and N_imp over the
  active shells only (T4.7). They need **no** frozen core (J1).

Legitimate differences exist (spectra is a closed cluster with a pinned core). Make them **explicit**, with one spec
builder per driver (A1), not implicit in call sites.

## 2. Findings, most severe first

Status: **C** = confirmed by a run (or by reading, where noted); **R** = refuted; **P** = plausible.

### Wrong answers

| ID | status / evidence | finding | proposed modification |
|---|---|---|---|
| **S2** *(new, violates SYM)* | **C** run: thermal levels in `logs/prof_smo*_n*.log` (per-state table, `prof_smo4_n4.log:113-117`); **C** read: `sector_solver.py:1001-1002` | **A capped thermal solve breaks the Hamiltonian's spin symmetry.** `solve_certified` builds a symmetric union: classes plus S± partner rows and mirror images (`_add_units_within_cap`, `:714`, regenerates partners from a cut class "so a multiplet keeps its shape"). It then hands that union to a plain `CIPSISolver.expand(h_op, Within)` (`:1001-1002`), whose selection, `CapRefiner` cycles and memory-guard truncation pick determinants one by one, with no orbit closure. Evidence by workload: (a) **SMO 4-bath** (cap-bound, 6 cycles): the S=3/2 quartet splits into S_z=−3/2 at 0, +3/2 at 4.2e-7, −1/2 at 6.4e-6 and +1/2 at 7.6e-6. Both SU(2) (\|S_z\|=1/2 vs 3/2) and the spin-flip pairing (±S_z) are broken. (b) **SMO 3-bath** (memory-bound): a spread of 5e-8 at -n 4 and 1e-8 at -n 1. (c) **Uncapped, exactly degenerate:** SMO 2-bath and NiO, where the `Within` expand admitted nothing. The GF stage inherits the asymmetric states, so G and Σ can carry spurious spin polarization. Rev. 1 (B2) and O3 had read this as an accuracy issue. Under SYM it is a correctness defect. | **Every growth and every cut in the `Within` expand acts on orbits.** Close each admitted candidate set under S± (`plan.su2`, `_spin_partner_rows`) and the symmetry images (`plan.images`, `_image_keys`), the same way the union does. `CapRefiner`, the memory guard's `_lower`, `truncate` and the swmin cut keep or drop whole orbits, ranked by the orbit's maximum weight. If the cap cannot hold the symmetric basis, admit fewer orbits: every member loses accuracy equally. The same rule applies to any other truncation that sees a symmetric manifold (audit: `truncate_initial`, `get_eigenvectors(slaterWeightMin)`, the GF support truncation). **Acceptance:** on an SU(2) fixture under a binding cap, the members of a degenerate level agree to roundoff, ⟨S²⟩ is exact for each, and Σ↑ = Σ↓ for a paramagnetic input. Inject an orbit-breaking cut → the test fails. |
| **S1** *(r2, new)* | **C** run on toys (`review_code/boundary3.py`); production exposure **P** | **A one-determinant seed is "certified as it stands" and returns an excited level.** At full impurity occupation the seed is one determinant, so `certification_plan` yields one job, and `groundstate.py:318-321` takes the "one class, nothing to be trapped away from" branch. The expansion closes on an H-invariant span (15 of 28 determinants) on which the exact ground state has weight 2e-30. The returned energy is exact *level 1*: −0.20474 vs −0.237399; −7.148735 vs −7.175867; −3.036391 vs −3.244187. Root cause: `seed_sectors` enumerates only the classes present in the seed span (bath full), so a class whose low states need bath holes or conduction electrons is never seeded. Exposure: any closed-shell sector (d10 as the N+1 of a Cu d9 gap DC, d0, FCC Ni's d9→d10 step). The oracle missed it because it checks only the winning sector. | A one-job plan whose seed span is smaller than the label set is not a certificate. Seed the missing S_z/parity labels from single bath excitations of the seed, or mark the sector uncertified. The promoted oracle compares **every** solved sector's E(N) with the exact value; those values are already computed. |
| **S3** *(new, traced)* | **C** run: `logs/trace_cri3_fast.log`, `prof_cri3_n1.log`; **C** read: `sector_solver.py:~980-1000`, `_add_units_within_cap` (`:714`) | **The thermal union drops whole in-window classes when they do not fit under the cap, and reports the result as converged and untruncated.** Classes at the winning level (within both error bars) are truncated to fit; every *later* in-window class is added whole or "stops the union" (`truncate=False`). On CrI3 the first class holds 405,687 dets, so the second (494,833) cannot fit under the auto cap (607,191, sized "for the ground state alone"). Two of the four quartet states (0.108 and 0.226 mRy above, inside a 23 mRy window, about half the Boltzmann weight) are discarded. The only trace is a verbose line. `truncation` is `null` and `converged` is `True`, because the union cut is not recorded in the truncation report. That misled the review too: the 1-rank rerun looked "untruncated". | **(1) Never drop an in-window class.** Truncate every in-window class to the same share, with partners and images (as is already done for the winning level), so all window states lose accuracy equally: the completeness analogue of SYM. **(2) Record the cut** in the truncation report, and set `converged: False` / `limited_by: cap` when the union was cut. **(3) Size the `Within` cap for the window** (O3, P10). **(4)** The C7 detection compares the number of states returned with the number of in-window verdicts. |
| **C7** | **C** run: oracle at τ=0.05 (real model 8: missing state at +0.296 = 6τ); τ=0.2: 10/100 real (*soft*, near the edge); 0/100 complex. **CrI3 production: C** (`review_physics/cri3_reduced.py`, `logs/cri3_reduced_series.log`, `logs/prof_cri3_n1.log`) | The `Within.thermal` manifold misses states and still reports `converged: True`. `manifold_deficit` (`sector_solver.py:869`) checks only the lowest level, and without an S_z label (CrI3: "spin layout: none") it expects multiplicity 1. *(r2)* **Production evidence:** CrI3 keeps 2 states (0 and 0.36 meV, both \|S_z\|≈1.6) in a 23 meV window. A reduced cluster built from the captured production h_op (N=13, dimension 77,520) has 4 states inside the window, exactly and in the uncapped code (0, 0.11, 0.24, 0.37 meV: the S=3/2 quartet; next state at 65 meV). The ±1/2 members would carry about half the Boltzmann weight, which biases ⟨S_z⟩, the moment and the Curie χ. **Root cause traced (2026-10-31, `trace/trace_cri3.py`, `logs/trace_cri3_fast.log`) → finding S3.** With SOC there is no S_z label: all 120 seed determinants form **one** hard label with 3 seed classes. The class expansions **do find** the middle members: class 1 at +0.108 mRy, class 2 at +0.226 mRy (exact: 0.105, 0.221; robust as the reduced bath grows, `logs/cri3_reduced_series.log`). The union then drops them: "thermal window: held at the cap (405,765 of 607,191 determinants): 2 later class(es) left out". Production keeps the outer pair only (0 and 3.55e-4). | **Detection first (step 3):** count states per hard label inside the window, and warn when a cap trim dropped any class within cut + error. **Fix later (step 8):** certify the window per label (k lowest per class). |
| **P1** | **C** run: `p1_frozen.py`, `frozen_drain.py`; `core_conserve.py` | **(a)** The DC criteria freeze bath-less groups (`dc_criteria.py:824,1958`); `calc_selfenergy` freezes nothing. With one bath-less shell the DC search measures a different state: E0 −8.97/−7.94/−6.81 frozen vs −9.34/−9.30/−9.28 unfrozen. The comment at `:822` ("pinned exactly as get_spectra") is false: get_spectra pins the *declared core* (`get_spectra.py:392`). **(b)** "Frozen" only pins the **seed** (`generate_initial_basis` and `sector_key`); the restrictions bound only the total impurity charge. Under core–valence mixing the pinned group leaks (3.3% left at n₁=2). No production archive has a bath-less group (`logs/bathless.log`), so (a) is latent. | **Per J1.** (a) Delete the bath-less rule. (b) The spectra Hamiltonian conserves N_core *structurally* (*(r2)* `slater_condon_Uop`, `atomic_physics.py:571-626`, builds only F_cc, F_cv and G_cv; 413/413 terms on NiO L₂,₃), so the seed pin is exact there. Where the user freezes a shell, compute ‖[H, N_core]‖ on every rank from the replicated h_op, and warn once if it is non-zero (O5). |
| **P7** | **C** run: `dbg_sector.py`, oracle (real 3/240, all with the non-default `scan`; complex 17/160). *(r2)* fix verified by patching (`review_code/p7_patch.py`) | In the legacy dN scan, the guard at `groundstate.py:856-863` tests `imp_occ + dN_gs[j] > 0` for **every** group. It can never step to occupation 0, and any group already at 0 blocks every extension. That includes SMO's group 0 at nominal 0 (`{0:0, 1:3}`). In production it is reached only via P8: every wrong-sector failure is an HF-not-converged fallback. | *(r2)* **Delete both `all(...)` clauses.** `dN_gs` is already 0 for frozen groups, and `get_energy` (~`:668-677`) rejects out-of-range totals and frozen values. With the patch, all 11 failing models flip to the exact E0, a frozen empty shell goes −4.490 → −6.632, and two-group models work. Canonicalise the per-group split before returning: negative splits like `{0:-1, 1:1}` already occur, and MS's per-shell N0 would inherit them. Alternative: run the HF branch's bounded axis walk (`:807-834`) from the scan winner, which is also P8's fallback. |
| **P8** | **C** run: `hf_conv.py`, `hf_dbg.py`; *(r2)* `review_code/hf_gap.py` | HF converges for 92–97% of real random models but only 33–43% of complex/SOC ones. The Fock matrix is the exact gradient (‖F−∂E‖ ≈ 1e-9). *(r2)* It is not a degenerate Fermi level (0 of 45 non-converged last iterates have a gap < 1e-3); Fermi smearing recovers 22 of 45, which points at sloshing between fixed points. A single-call count gave 75/120 for complex models; recheck how `hf_conv.py` counts before quoting the rate. Every failure falls back to the scan (P7). The seed is also **off by one on every SMO workload** even when converged (seed N4, winner N3; 371 s = 19% on SMO 4-bath). | Level shifting / DIIS / ODA and more starts. On failure, walk ±1 from the nominal N0. Make "HF seed rejected" default-visible. For the off-by-one, start the walk cheaply (see B1). Keep `winning_N0`/`converged` broadcast as the only source (`groundstate.py:510-512`). |
| **P11** *(review)* | **C** read | The walk decides between cap-bound, variational `Lowest` energies with **no margin check**. It never compares the gap with the error bar, and it never records convexity. *(r2, physics)* N_imp is not conserved, so E_N(μ) is **concave** (a −½χμ² term), not affine. | Record the margin-to-error ratio and the second difference. Below 1: tighten that sector or extend to N±2. Cross-check E_N(μ) ≤ E_N(0) − μn_N(0) **one-sided**, with a χ-sized margin. Decide on rank 0 and broadcast. One summary line. |
| **P9** | **C** run: oracle notes. *(r2)* **production evidence:** the losing sectors of 3 of 5 workloads are inside the cut (§3) | Neighbouring N sectors inside the thermal window are dropped silently at fixed N. | MV plan. Cheapest step now: warn with ΔE/τ from the walk's energies. *(r2, physics)* **Drop** rev. 1's free-energy re-ranking (E − τ ln g). It needs Z_N over a window the walk never solves, and it only matters in MV's regime. |

### False certification

| ID | status | finding | proposed modification |
|---|---|---|---|
| **C2'** | **C** run *(r2)*: `review_code/c2_small.py` | `eigensolvers.py:496`: `except ValueError` around the locked fallback `run(...)` sets `saturated=[]`, and the earlier `certified=True` stands. *(r2)* The swallowed error in practice is `trlm.py:374` ("max_subspace_blocks must be strictly greater…"). The fallback's fresh block (r_new=8) deflates against `locked` to rank 6, while `krylov_depth` was sized for the width before deflation (`gs_krylov_blocks(10,8,8)=2` vs k_blocks=ceil(8/6)=2). Repro: n=10 with a 6-fold ground level returns n_in=4, certified=True. `LinAlgError` is a `ValueError` subclass, so `eigh` failures are swallowed too. Production goes dense below `dense_cutoff`, so exposure is **P**. | Size the fallback depth, and clip r_new, on the complement dimension `n_global − locked.shape[1]`. Treat only `BlockBreakdown` with code k==0 as "empty complement". Everything else gives `certified=False`, not a re-raise. Verify at -n 3, since the catch wraps a collective. |
| C1 | **R** | TRLM's breakdown → `invariant_subspace_continuation` is correct for a genuine invariant subspace. | Optional: narrow to `BlockBreakdown`. |
| **C4** | **C** run: `c4_store.py` | `SectorPlanStore.record` keeps the first record (`sector_solver.py:1073`). After `violated()` and a recertification (`groundstate.py:337-353`), the new verdicts are dropped. `window_overtake` stays skipped because `only` stays non-None. | Replace the record after a full recertification, and set `only = None` there. Test: monkeypatch `solve_certified` to trip `violated()`, then assert that the record's μ and verdicts are replaced. A unit test of the store alone does not discriminate. |
| **M4** | **C** run: `prof_nio10_n1/n4.log`, `union_nio10_n4.log` | `memory_bound` depends on the rank count: NiO is null at -n 1 and true at -n 4 on the same basis and E0. *(r2)* The rev. 1 cause (shm counted, `cipsi_budget.py:135`) is **inferred, not measured**: no log has the `-v` anon/shm lines. More: the union run sets `memory_bound: true` with zero discarded mass and zero cycles, and on SMO the "tightening" lowered the cap to the value already in force (509,546 → 509,546; 440,512 → 440,512), while `_report_convergence` (`cipsi_solver.py:~1268`) still labels it memory. `_record_truncation` (`:1229`) reads only the last round's discarded mass. | Set `memory_bound` only when `_lower` actually lowers the threshold (`cipsi_budget.py:~165`). Accumulate a replicated "memory withheld" flag across rounds. Measure anon/shm with `-v` before changing the budget basis. If changed, use `peak − shmem` against an anon-based resident term on both sides (`memory_estimate.py:1029`), MAX/MIN-reduced. The code has no MPI shared windows, so anon budgeting loses no real cost. Depends on M1 (the discarded-mass allreduce sits under a branch replicated only if the budget is). |

### MPI hazards

| ID | status | finding | proposed modification |
|---|---|---|---|
| **M1/M2** | **C** run: `mpi_env.py`; *(r2)* `review_mpi/m1_fixed.py` | When the env differs between ranks, the solve deadlocks for `GS_MEMORY_BUDGET_SAFETY=0`, `GS_CERTIFY_MAX_SEED` and `GS_SECTOR_CERTIFY`. *(r2)* **Rev. 1's fix is insufficient, measured.** With the early return at `groundstate.py:113` moved below both collectives, the MPMD run still hangs (exit 124; the uniform control passes in 5.6 s). Whether a budget exists is rank-local. `measuring` (`cipsi_solver.py:1033`) runs two `_allreduce_max` calls in `determine_new_Dj` (`:889-901`) on one rank only, and `_frontier_matrix_fits` returns early at `:1318` while the other rank waits in the allreduce at `:1327`. *(r2)* The hazard covers more knobs: `APPLY_ROW_CHUNKS` (`_lanczos_step.pxi:847`, the number of collective `redistribute_block` calls), `GF_SUPPORT_TRACE` (an extra matvec), and an invalid `GS_LOWEST_SOLVER` raising rank-locally (`eigensolvers.py:807`). | **(1)** In `expand_memory_budget`, broadcast `safety` from root before the `<= 0` test; make `_frontier_matrix_fits` reach its allreduce unconditionally. **(2)** *(r2)* Instead of the rev. 1 snapshot, which overrides env process-wide and breaks `gf_oracles.py:179` and `basis_size_harness.py:48`: `config.assert_uniform(comm)`. It allgathers a 64-bit digest of the raw env strings of every registered knob (Cython reads included), always, never cached, and raises the same error on every rank naming the knobs. Call it at `solve_ground_state`, `calc_selfenergy`, `run_spectra`, `calc_susceptibility(_workflow)`, rixs `calc_map`/`calc_tensor_map` and the DC entries. A per-knob `uniform` attribute exempts `GF_SECTOR_CACHE_DIR`. **Test:** set the knob on rank 1 only, assert that the `expand_memory_budget` results allgather equal and that every rank raises; `skipif(size == 1)`. |
| **M5** *(r2, new)* | **C** read | GF-side twin of M1, not env-driven. `KrylovShiftedResolvent.solve` (`gf_shift_recycling.py:563,637`) calls `available_bytes_per_rank()` **without a comm**: node-local MemAvailable, sampled per rank. It then returns `None` on some ranks while the others loop back into `block_lanczos`. It is reached from `rixs.py:350` on a colour comm. That is a latent multi-node RIXS deadlock, and near the memory threshold a single-node one too. | `available_bytes_per_rank(basis.comm)`. Same for `_sector_dense_max` (serial path only). |
| M3 | **C** run | `solve_ground_state(rank=0)` is the default, so spectra and the occupation DC print on every rank. | Use `comm.rank`; drop the parameter. *(r2)* Callers that pass `rank=` must change in the same commit: `oracle.py`, `dc_criteria.py:300/767/845/1064`, `test_fixed_dc`, `test_sector_certify`, `_nio_workload`. |

### Inconsistencies between consumers

| ID | status | finding | proposed modification |
|---|---|---|---|
| **P3** | **C** run: `parity_susc.log` | Susceptibility: chain_restrict hard-coded False, no weighted restrictions or tolerances (`susceptibility.py:489-509`). | A1, with its own χ baseline. |
| **P2** | **C** run: `parity_spectra.log` | Spectra solves the GS at swmin 0 but builds restrictions at 1.5e-8, and drops chain_restrict, dense_cutoff, mixed_valence and the tolerances. | A1. Unifying changes the XAS/RIXS final-state bases, so it needs a spectrum A/B **including RIXS**. `run_spectra` gains `SolverOptions` (a public, TOML and CLI signature change; RSPt is unaffected). |
| **P4** | **C** read | `SpectraOptions.energy_cut` (`model.py:1767`, `scripts/spectra.py:271`) and `ctx.energy_cut` (`dc_criteria.py:1869/2015`) are written and never read. | Wire it to `Within(...)`, or delete it. |
| **P10** | **C** run: caps 727105/693785/734146 on consecutive runs; 579,384 vs 626,972 on NiO | The auto cap is re-derived from live free memory on every call, so the DC search, Σ and each DMFT iteration run at different caps. *(r2, numerics)* It also confounds every 1-rank vs 4-rank comparison in §3. | *(r2, architecture)* Resolve it in the **caller**: RSPt calls the DC and `calc_selfenergy` as separate API calls (`impmod_interface lib.py:1402-1426, 1542`). Pass a resolved `CapPolicy` through `BasisOptions.truncation_threshold`. Hidden dependency: `cap_from_memory = bcast(threshold is None)` (`dc_criteria.py:790, 1954`) would read a passed policy as user-set and switch off recalibration, so change it to `CapPolicy.coerce(...).from_memory` first. Replaying min(recorded, current) needs an archive attribute and an RSPt header field, read on rank 0 and broadcast. |
| P5/P6 | **C** read/run | *(r2)* The DC paths use **three** tolerance rules: `_prepare_sector_context` takes argument > `DC_*` > basis > GS (`dc_criteria.py:847-848`); the `_SectorContext` default (`:457-459`); the occupation context uses `basis.e_pt2_tol` only. Σ ignores `DC_*`. dense_cutoff has **four** default sources: schema 500 (`schema.py:1173`), `SolverOptions` 500 (`model.py:1569`), the archive reader 1000 (`model.py:1251`), the GS functions 1000. dense_cutoff changes the basis (21,257 vs 23,832 dets, `stag2.py`). | Name one owner per default. One precedence: the DC knob if set (opt-in, for a rough DC; kept per O2), else the spec. B8 reuses states only when the knobs are unset. |

### Ruled out

- **C6, rank-layout dependence: R.** `logs/rankinv.log` agrees at -n 1/2/3 to the 9 decimals it prints *(r2: not bitwise)*.
- **R3** as stated: **R**. **R4**, dense solves not cached: negligible (0.28 s on NiO).
- *(r2)* **"Printed once prints twice": R, a harness artefact.** `emit_memory_warning` (`memory_estimate.py:1353-1366`) prints to stdout *and* `sys.__stderr__` by design, as the RSPt channel. The profiles merged both streams (`2>&1`). Do not "fix" it. The real asymmetry: the per-sub-solve "not converged" warning (`cipsi_solver.py:1301`) goes to stdout only.

## 3. Bottlenecks (measured)

Trace kinds nest: `sector_solve ⊃ certify ⊃ {class_expand, class_eigensolve, union expand}`, with `eigensolve`,
`h_matrix_build` and `cipsi_selection` as siblings inside those. Do not sum across them.

*(r2)* Caveats on the numbers:
- **peakRSS** comes from `ru_maxrss`, which `clear_refs` resets, so it is a lower bound since the last reset. NiO
  -n 4 reports 1009 MiB while the guard measured 1.5 GiB.
- **The 1-rank vs 4-rank speedups** compare different caps and bases (P10).
- **"Report & rest"** covers more than the GS report.

| workload | ranks | calc_gs | walk | thermal | report & rest | dets: walk winner → thermal | notes | log |
|---|---|---|---|---|---|---|---|---|
| NiO 10-bath it.1 | 1 | 411 s | 220 s (d8 50 s, **d7 165 s**, d9 3 s) | 138 s | ~53 s | 47,080 → **624,574** (3-fold) | | `prof_nio10_n1.log` |
| NiO 10-bath it.1 | 4 | 344 s | 192 s | 119 s | ~33 s | same | | `prof_nio10_n4.log` |
| NiO 10-bath it.1, rerun | 4 | 351 s | 210 s | 107 s | ~34 s | 47,080 → **379,129** (cap 579,384) | same E0, χ_zz and per-state observables | `union_nio10_n4.log` |
| SMO 2-bath | 4 | 170 s | 137 s (N4 18 s, N3 30 s, N5 4 s, **N2 85 s**) | 25 s | ~8 s | 37,925 → 153,998 (4-fold) | HF seed N4, winner N3 | `prof_smo2_n4.log` |
| SMO 3-bath | 4 | 1112 s | 827 s (N4 87 s, N3 210 s, N5 17 s, **N2 513 s**, cap-bound) | 255 s | ~30 s | 247,974 → 452,240 (4-fold), `converged: False` | thermal 1.5e-8 above walk, residual 5e-8 | `prof_smo3_n4.log` |
| SMO 3-bath | 1 | 1463 s | 1070 s (N2 718 s) | 340 s | ~50 s | 247,974 → 509,546 | thermal 6e-9 above walk, residual 2.2e-8 | `prof_smo3_n1.log` |
| SMO 4-bath | 4 | 1986 s | 1566 s (N4 371 s, N3 511 s, N5 40 s, **N2 644 s**) | 384 s | ~35 s | 440,512 (cap) → 440,512, residual 1.75e-5 | thermal 3.8e-6 above walk; 4-fold level spread 7.6e-6 | `prof_smo4_n4.log` |
| CrI3 SOC | 4 | 2346 s | 1557 s (seed 506 s, **N2 954 s**, at the cap; N4 96 s) | 748 s | ~40 s | 405,687 → 487,647 | **2 of 4 quartet states kept (S3)** | `prof_cri3_n4.log` |
| CrI3 SOC | 1 | 2953 s | 1905 s | 979 s | ~70 s | → 518,555, reported untruncated and converged | **still 2 states: the union cut is unreported (S3)** | `prof_cri3_n1.log` |
| FCC Ni 15-bath it.1 | 4 | **> 5400 s (timed out)** | the d9 seed sector alone ~47 min: 5 cap-bound class expansions at 723,036, residual ~1.2e-5 | — | — | — | not attributed (B8 note) | `prof_fccni15_n4.log` |

**Losing sectors and the thermal cut** *(r2)*. With cut = 9.21τ, the losing sectors that dominate the walk are:

| sector | gap | cut | inside the cut? | share of calc_gs |
|---|---|---|---|---|
| SMO 4-bath N2 | 0.0079 | 0.0175 | **yes**, Boltzmann ≈1.5% before multiplicity | 32% |
| CrI3 N2 | 0.0183 | 0.0230 | **yes**, ≈7e-4 | 41% |
| NiO d9 | 0.02283 | 0.02303 | at the 1e-4 design edge | — |
| SMO 3-bath N2 | 0.034 | | no | 46% |
| NiO d7 | 0.044 | | no | 41% (-n 1) |

So under MV, 3 of 5 workloads have K = 2, and the expensive loser is a member, not waste.

**"Thermal worse than walk" on SMO** *(r2, numerics, physics)*:
- **The deficits sit inside the error bars:** 6e-9 vs 2.2e-8, 1.5e-8 vs 5e-8, 3.8e-6 vs 1.75e-5.
- **`converged: False` is printed.** The walk prints E to four decimals, though, which hides the comparison.
- **The `limited_by: memory` label is wrong there** (see M4).
- **The cause is the auto cap, sized "for the ground state alone", applied to a 4-state union.** The winning level
  alone needs about 4 × 248k determinants.
- **The harm is broken symmetry (S2), not the energy.** \|S_z\|=1/2 sits 6e-6 above 3/2 and ±3/2 differ by 4.2e-7.
  The Hamiltonian's SU(2) and spin-flip symmetry are lost in the states handed to the GF stage. Under SYM this is a
  correctness defect, whatever its Boltzmann size (0.4% at τ = 0.0019).
- **Fixes:** S2 (orbit-closed growth and cuts), and a cap resolved for the `Within` manifold (with P10). Detection:
  any split of a degenerate level beyond roundoff, decided on rank 0 and broadcast. Do **not** seed the thermal solve
  from the walk basis (that breaks the symmetry).

**Matrix-build share** *(r2, numerics)*. `h_matrix_build` is 22% of calc_gs (NiO), 31–32% (SMO) and 50% (CrI3). That
arithmetic is right. But builds equal sparse plus dense eigensolve calls *by construction* (NiO 399 + 74 = 473), so
"builds > selections" proves nothing about rebuilds. Measured split on SMO 2-bath at -n 2 (`build_probe_smo2_n2.log`):

| build kind | time | share of build time |
|---|---|---|
| incremental reuse (`set_p_keys`) | 68.6 s | 58% |
| verdict rebuilds | 34.4 s | 29% |
| fresh frontier builds | 15.1 s | 13% |

`matrix_hook` (ClassPenalty) is nested inside the build, not additive. At -n 4 the frontier snapshot (0.1 × budget)
may be refused. CrI3's 6.9 s per build suggests a full rebuild every cycle, so re-measure at production budget.

Levers. *(r2)* All expected gains are **upper bounds** until measured on the change.

- **B1. Stop clearly losing sectors early, using a bound that exists.**
  - *(r2)* Rev. 1's ceiling "E_var − c·residual_PT2 > E_best + margin" fails twice. Its margin (0.17) exceeds every
    measured gap, so it never fires. And the code's `residual_pt2` is the mass of *un-admitted* candidates
    (`cipsi_solver.py:810`), which sits near e_pt2_tol by construction during growth: it underestimates the remaining
    drop by 90–2.8e4× early on (`review_numerics/pt2_bound.py`).
  - **Lever:** the pass-1 trajectory bound `lower` from `_ranking_remaining` (`sector_solver.py:95`, RANK_FLOOR 1e-3).
    Pass the walk's E_best into `certify_sector`, and skip pass 2 for classes with `lower > E_best + margin`.
  - **Expected:** about 55–66% of a losing sector (that function's own NiO replay). **Nothing** on sectors inside
    `cut + margin` (J2), so nothing on SMO 4-bath N2, CrI3 N2 or NiO d9 under MV. Not for the DC peak/gap path
    (a per-call option).
  - Implement it through `cycle_hook` (stop broadcast at `cipsi_solver.py:1077`), composed with `_expand_job`'s hook
    (`sector_solver.py:450`), not overwriting it.
- **B2. Thermal union margin.** Must keep orbit closure (SYM): drop or continue whole classes with their partners and images.
  - *(r2)* **Rev. 1's cause was wrong.** NiO's growth is neither the PT2 budget split nor "the cap sets the size".
    `solve_certified` admits classes with `margin = within + max(residual over ALL verdicts)` (`sector_solver.py:603`).
    On NiO, window 0.02303 becomes margin 0.028, inflated by two ranking-only classes with residual ~5e-3.
  - **Measured** (`union_probe_n2.log`): the winning triplet (classes of 47,080 + 25,584 dets plus partner rows) is
    **134,804 dets, 2.9× the walk's 47k**. The rest (244k–490k) is whole classes of a level at +0.02646, *outside* the
    window (Boltzmann 2.5e-5), admitted through the inflated margin. The cap only decided whether one or two of them
    fit: 379,129 vs 624,574. The `Within` expand admits nothing (cycles 0). The identical E0 is trivial, because those
    classes carry a different parity label. The same holds for χ_zz and every per-state observable to 6 digits.
  - **Lever:** a per-class error bar (`_energy_error(v)`) instead of the global maximum. Resolve ranked or leaked
    classes with an uncertain bound by continuing them (~91k dets), rather than admitting them whole with partners.
  - **Expected:** NiO thermal basis 624k → ~135k (**4.6×**). B6 and the GF stage shrink with it. Judge on Σ (Σ/G are
    untested); report the discarded norm, not the determinant count.
  - Rev. 1's alternative ("expand one row, close under S±") is what `solve_certified` already does.
- **B3. Hand the walk's winning-sector class keys to the thermal solve.** `jobs_at` re-expands every in-window class
  from the seed at the same shift and accuracy: ~64 s on NiO.
- **B4. Matrix reuse in verdict solves.** Pass the held, un-hooked matrix in as `h_matrix` (`expand` drops `_h_matrix`
  at `cipsi_solver.py:1176`). *(r2)* **≤ ~11% of calc_gs** on SMO 2-bath. Re-measure at production budget with
  `review_numerics/build_probe.py` before ranking. The incremental path (58% of build time) is the bigger target.
- **B5. Capped refinement stopping rule.**
  - *(r2)* The actual rule is "two consecutive cycles without improving by more than e_tol, or 10 cycles"
    (`cipsi_budget.py:197-207`).
  - Rev. 1's lever loses accuracy (`review_numerics/r3b.py`):
    - stopping at gain < κ·residual leaves 9.5e-6 instead of 6.0e-6 at cap 2000;
    - a 1e-5 floor is 5–26× worse at cap 5000;
    - under a cap the residual underestimates the true error by 0.8–5×. The cycles are not wasted.
  - **Lever:** stop when a geometric-tail prediction of the remaining gain falls below the accuracy that matters: the
    sector margin, or 0.1τ for MV.
- **B6. GS report.** Unpriced (part of "report & rest"). It shrinks with B2.
- **B7. Task parallelism.**
  - 399 small Davidson/TRLM solves are latency-bound (`eigensolve` 176 → 168 s from 1 to 4 ranks).
  - Lever: run independent class jobs on sub-communicators.
  - *(r2, MPI)* Requirements:
    - take the budget from the parent comm before the split and divide it among colours (`available_bytes_per_rank`
      divides by the sub-comm's ranks per node, so colours overcommit);
    - assign jobs to colours statically and merge verdicts in job order, because the leak verdict is
      order-dependent (`sector_solver.py:476-478`);
    - gather every colour's verdicts before `record`/`violated`/`jobs_at`/`manifold_deficit`;
    - broadcast results rather than recompute them, since the release build is not bitwise-replicated;
    - free the comms at a synchronized point, never from the GC;
    - no `dc_criteria` COMM_WORLD inside a colour;
    - test at -n 3 and -n 4.
- **B8. DC search followed by a full re-solve in Σ (R1 = MV §J4).** Every RSPt DMFT iteration re-runs the prep, cap
  probe, walk and thermal solve.
  - **Bounded, not measured** (§5): ~350 s NiO, ~1100 s SMO 3-bath, ~2000 s SMO 4-bath, ~2350 s CrI3 per iteration at
    4 ranks.
  - Needs: A1 spec equality, the same dc, a caller-resolved cap (P10), and *(r2, architecture)* a new
    `calc_selfenergy` input that accepts a precomputed ground state or ensemble.
  - O2: the DC tolerances stay. Reuse the states when they are unset (the default), and only the sector choice when
    the user loosened them.
- **FCC Ni** *(r2)*. It took over 90 min and the walk did not finish. Its cost structure is classes × cap (five
  cap-bound class expansions in the seed sector alone). Attribute it to B1/B5/B7 only after a per-class, per-cycle trace.

## 4. Architecture and UX

- **A1. `GroundStateSpec`** (frozen dataclass) in `groundstate.py` (Layer 5). *(r2, architecture)*
  - **The test for membership:** whatever decides *which state is found* belongs in the spec. That is the
    `solve_ground_state` docstring's own invariant (`groundstate.py:946-952`).
  - **Carries:** tolerances; dense_cutoff; chain_restrict; the restriction **inputs** (excitation_budget, cost_budget,
    swmin), not the built list (it depends on h_op: built at dc_guess for DC, at the final dc for Σ); `use_hf_seed`;
    the resolved cap (P10); the frozen set (spectra builder only, J1).
  - **Leaves out:**
    - `comm`, `rank`, `plan_store`, `shift` and `verbose`;
    - `shell_ls` and the per-shell N0, which `SolverBasis` owns (a copy would be a second source of truth);
    - the MV ensemble switch until MV 5 adds it, either as a `plan_ensemble` argument or as a field excluded from
      the solve key (`auto` resolves only after the walk).
  - **One builder per driver**, taking the `*Options` dataclasses duck-typed (`groundstate` must not import `model`)
    plus the group keys. All three front ends converge on those dataclasses, which is why one spec serves TOML, the
    CLIs and RSPt. Touch points: `inputformat/{schema,build}.py`, `lib.py`, and `model.py`'s archive round trip, which
    also drops `de2_min`.
  - It replaces the untyped `basis_setup` dict and `calc_gs`'s pass-through, including the legacy renames at
    `groundstate.py:1124-1127`. It feeds every entry in §1: `solve_ground_state`, the DC walk (`:498`) and
    `solve_sector` (`:577`).
  - **Plan-store key:** `(sector_key, len(basis), spec.solve_key)`, which removes the `float(e_pt2_tol or 0.0)`
    collision structurally.
  - **Series** (each step its own commit):
    - (a) bit-identical plumbing: each driver builds today's values explicitly, so the differences become visible;
    - (b) one unification per commit, each with its A/B: P3 χ, P2 spectra + RIXS, P5 DC tolerances, P6 dense_cutoff;
    - (c) P10 separately.
- **M1's knob check** lives in `config` (Layer 0), not the spec (§2).
- **UX** *(r2, UX)*:
  - **One summary at the end of `calc_gs`**, not after the walk: the thermal events (regression, cap tightening)
    happen after it. DC prints it for the accepted μ only (as MV D6), at V_RESULT.
  - **Walk table columns:** sector, E (to the error bar's digits), dets, error bar, limited_by, certified, wall time.
    That makes a separate "k of n cap-bound" line redundant.
  - **Every warning names its knob** for each front end: CLI flag, TOML key, RSPt solver-line token, env var. State
    the effective accuracy ("E0 limited by the cap to ~1e-5") rather than "not converged against 1e-8".
  - **stderr is tiered.** It is the only channel the RSPt DC and Σ passes share (they write different files,
    `lib.py:1209-1218`). Only four events go there:
    - the thermal result regressed beyond its error bars;
    - margin/error below 1;
    - MV dropped weight above 1e-2;
    - memory-bound with withheld mass.
  - **Progress:** print "solving sector N=…" at each trial; a rate-limited heartbeat (every ~5 min: cycle, dets,
    error, elapsed); an early "cap-bound at cycle k, error ~1000× tol, result will be cap-limited" line. FCC Ni
    printed nothing for 47 minutes.
  - **Also:** HF seed vs winner ("seed N=4 → winner N=3"); name the certified labels next to `converged`; state the
    frozen set and its provenance; print ΔE to N±1 in τ. Drop the "GS_MAX_BLOCK_WIDTH is unset… set it" nag, which
    advertises an internal env var on every run, and fix "1 states".
- **Hygiene (read):**
  - stale docs: `dc_criteria.py:33-46,384,1954`; `gf_units.py:47`; `gs_report.py:88-91`;
    `cipsi_solver.py:549,620,981,1414`; `config.py:602,721,766`; `groundstate.py:116`; `test_fixed_dc.py:232`
    ("basis_generation freezes any bath-less group" is false); the lazy-import comment at `dc_criteria.py:1987`;
  - unused parameters: `_candidate_overlaps_and_energies(slaterWeightMin)`, `_local_diagonal(offset)`
    (`eigensolvers.py:567`), trlm `slaterWeightMin`/`basis`;
  - `solve_window(max_keep)` is never passed, so retention is unbounded;
  - `load_selfenergy_archive` drops `de2_min`;
  - Boltzmann weights are computed in `gs_statistics._boltzmann_weights` and `ThermalEnsemble`; MV D2 would add
    `average.sector_weights` as a third, so consolidate in MV 5;
  - the occupation DC hard-codes `MPI.COMM_WORLD` (`dc_criteria.py:1960, 2052`).
- **H3. Degeneracy grouping. Architecture, not hygiene; schedule it before C7.** *(r2)*
  - Four rules: `cipsi_solver.py:166-180` and its verbatim copy `eigensolvers.degenerate_groups` (`:53`) anchor on
    the first member; `cipsi_solver.py:384` anchors on the last kept; `trlm.py:211` chains neighbours.
  - Tolerances vary: 1e-9, 1e-10 (seed_sectors), 1e-6 (`susceptibility.py:57`, `observables.py:711`,
    `gs_connectivity:132`).
  - `selfenergy.py:9` imports the private helpers.
  - Fix: one public helper in a leaf module. C7, B2 and the MV χ merge all group levels.

## 5. Measurements (2026-10-30)

- **Profiles:** the §3 table. FCC Ni timed out after 90 min inside the walk.
- **SMO 3-bath at -n 1:** same regression pattern, inside the error bars (§3).
- **NiO union:** the cause is the margin, not the cap (B2, `review_numerics/union_probe_n2.log`). The §3 rerun row is
  that measurement. `memory_bound: true` with zero withheld mass (M4).
- **R1/B8 price: not measured.** Both DC configs failed in the search on the input: the example
  `h0_NiO_10bath.pickle` is not Fermi-referenced (every level below E_F). Fixed-gap landed at n_center = 1;
  fixed-occupation saturated at n_ref = 10. Those searches took 393 s and 471 s (`logs/r1_fixed_*_n4.log`). The bound
  is in B8. A measured price needs an archive-based DC with an explicit occupation target.
- **Bath-less group survey:** none in any production archive (`logs/bathless.log`).
- **Core conservation in spectra:** 413/413 terms (`logs/core_conserve.log`); structural (P1).
- **CrI3 quartet: traced (S3).** The class expansions find all members. The thermal union then drops the two later
  in-window classes whole at the cap, and the run still reports itself untruncated and converged. That report hid the
  cause from the 1-rank rerun.

## 6. Joint plan with the mixed-valence and multishell plans

- **MV:** `doc/plans/mixed_valence_sector_ensemble.md` (rev. 2). It repurposes `mixed_valence` as the grand-canonical
  multi-sector switch and removes seed widening.
- **MS:** `impmod_interface/doc/plans/multishell_cluster_support.md`, G4 (T4.1–T4.7). It adds the shell model,
  per-shell N0 and DC shifts on the active shells.

Neither has landed (`shell_ls`, `frozen_groups` and `active_impurity_indices` do not exist in `src/`).

**ID crosswalk** *(r2)*. MV's references use older numbering. It cites "`gs_redesign.md` P7", but that doc has no
P-IDs; it means this review's P7.

| this review | MV | MS |
|---|---|---|
| A1 | §J1 (GS spec) | T4.4 (`SolverBasis` keeps `shell_ls`/N0) |
| P7, P11 | §J2, commit 2 (`SectorSearch.solved`) | — |
| P5 | §J3 | — |
| B8 (= R1) | §J4 (DC ensemble reuse) | — |
| P10 | §J5 (cap probe), D4 (GF cap) | — |
| C4 | §J6 (plan store; "C5" there = C4 here) | — |
| P9 | the plan itself | — |
| J1 | D3 "never ensemble across frozen" (moot) | D3, T4.4, T4.6, F1 |
| J5 | — | T4.7 |
| J8 | D6 | T4.5 |

### 6.1 Where the plans meet

**J1. What "frozen" means (decided by the user, 2026-10-30).**
- **Purpose.** `frozen_occupations` exists for a shell whose projected local Hamiltonian carries **no orbital
  energies**: the X-ray core, whose only one-body term is the SOC. Without the pin the walk empties core orbitals
  (2p⁴d¹⁰ wins by ~23 eV on NiO).
- **A proper local Hamiltonian needs no freezing.** RSPt's h_dft carries the core energies (2p ≈ −60 Ry). Nothing is
  frozen on the self-energy/DC path.
- **The seed pin is exact for its purpose.** The spectra U conserves N_core by construction (P1b), so CIPSI cannot
  leave the core-full span. The GF excited bases must **not** pin the core, because the XAS core hole is the
  excitation. That answers MS Q2: correct as is.
- **Consequences:**
  - **P1a is a bug.** Delete the DC bath-less rule and its `:822` comment. Rewrite `test_fixed_dc.py:1918-2010` and
    `:2080ff`, which assert the bath-less set. The frozen handling in `_SectorContext.trial_occupation` becomes dead
    for DC. Bath-less valence shells (e.g. Hubbard-I 4f) should not be pinned either: energetics and DC set them, unless the user declares the shell frozen (O5).
  - **Freeze only on explicit user request** (O5, user): a shell the user declares core, as `get_spectra.py:392`
    already does. No automatic detection. Where a user-frozen shell has ‖[H, N_core]‖ ≠ 0 (possible on RSPt h_dft
    with one-body core–valence mixing; the MS T4.4 test itself uses a 1e-3 p–d element), warn once with its size,
    because the seed-only pin then leaks. MS F1 freezes a shell only when the user declares it core.
  - **MS edits:** drop the frozen core from MS §1, §3 ("DC/N0/frozen logic"), D3, T4.4 (`frozen_groups`), the T4.6
    `frozen_occupations` addition and excited-basis pin, the T4.6 note, the test `frozen_groups == {p}`, T5.1 ("pinned
    at 6") and §7.2 ("frozen = {}"). Mark Q2 answered. **Keep** `shell_has_bath` and `active_impurity_indices` from
    T4.4, since T4.7 and J5 need them. Keep "`calc_selfenergy` keeps Tr ρ_p = 6" as a check that h_dft holds the core.
  - **Measure before J1 ships on multishell** *(r2, physics; plausible)*. With an unfrozen deep core:
    - `generate_initial_basis` lets the core range over [0, 6], so core-hole determinants hundreds of eV up enter the
      seed;
    - in `seed_class_plan`, the penalty 2·max(width, window) grows to ~10³ eV, and `scale = max|eig|` grows ~100×,
      loosening PARTNER_TOL/COUPLING_TOL;
    - seed growth heads towards `max_seed = 1000`, above which the sector silently becomes uncertified.

    *(r2, UX)* The walk or dN scan may also open trial sectors with a core hole and run full CIPSI solves on them.
    Measure all of this on the MS 2p@−60 Ry fixture. If any effect shows, add a deep-shell seed/restriction
    **window**, which is not a freeze and stays consistent with J1, and restrict the walk to hybridizing shells. Σ on
    the core block adds about 6 removal GF units at −60 Ry, and the core Hartree term goes uncorrected by the
    active-shell DC (harmless for occupations).

**J2. Who decides the sector set: walk bound (P11/B1) vs ensemble membership (MV D3).**
- MV admits `E_s − err_s − E_min < cut`, so every sector inside `cut + margin` needs a real error bar. B1 may stop a
  sector only **beyond** `cut + margin`. On 3 of 5 workloads the expensive loser is inside (§3).
- P11's record and MV commit 2's `SectorSearch.solved` are one structure:
  `(trial_N0, E, err, certified, cap_bound, truncation_report)`.
- MV's "guarantee ±1 neighbours" needs P7. It also needs S1, because a closed-shell neighbour may return an excited
  level.

**J3. Refusal rules need truthful flags and τ-scaled thresholds.**
- **Flags.** MV D3 refuses a cap-bound, uncertified or high-residual member. M4 (spurious `memory_bound`), C2', C7
  and S1 (spurious `certified`) feed it false flags. All four go **before MV 3**.
- **Thresholds** *(r2, physics)*. "Realised e0 moved by more than the tolerance" and a residual above c·τ fire on every
  cap-bound run: SMO 4-bath is 2000× over e_pt2_tol, yet that is only 0.002τ.
  - Express both in τ and weight: the weight error bar p(1−p)ΔN·err/τ, and \|Δe0\|/τ.
  - Refuse only when the weight error bar exceeds a threshold.
  - **`auto` never raises.** It resolves to off plus the warning when a driver is unmigrated (MV D7), the reference
    is not Fermi (MV D3; legacy h0 loads as "unknown"), or a member fails the guard. Only an explicit `true` raises.
    Without this rule an RSPt DMFT run aborts mid-iteration.

**J4. One spec object (A1 = MV §J1).** Contents are in §4. Two deviations from MV §J1:
- the ensemble switch is not in `solve_ground_state`'s signature; MV 5 adds it at the seam;
- `shell_ls`/N0 stay in `SolverBasis`.

**Before MV 1, a keyword-only commit** *(r2, architecture)*. `mixed_valence` is *positional* in
`build_basis_and_solver` (`groundstate.py:129`), `solve_sector` (`:193`) and `prepare_solver_basis`. Positional
callers include `dc_criteria.py:577-584`, `:~1935`, `gs_connectivity.py:94`, `test/support/dc_weights.py:160`,
`selfenergy.py:587` and `susceptibility.py:476`. Removing the parameter would silently shift `tau` into its slot.
From MV 1 until MV 5, a non-zero `mixed_valence`/`mv` must raise, not be ignored.

**J5. Plan store (C4, MV §J6).**
- MV 1 makes plan reuse unconditional, so C4 must land first. Today the exemption covers only formerly non-zero `mv`
  runs.
- With MS T4.7, give `SectorPlanStore` the active-shell projector **at construction** (it belongs to the shift
  operator), not in the key.
- The Hellmann-Feynman prediction in `jobs_at` is first order. Because E_N(μ) is concave, it is an upper bound, so a
  class predicted just outside the window can be inside. Add a χ-sized margin.

**J6. Reusing the DC result in Σ (B8 = MV §J4).** Needs A1 equality, the same dc, a caller-resolved cap (P10), the
`calc_selfenergy` precomputed-GS input. Per O2: reuse the states when the DC tolerances are unset; with loosened
DC tolerances, reuse the sector choice only. Deleting P1a
removes today's frozen mismatch. Under MV the reusable object is the DC's final ensemble.

**J7. Thermal re-solve per member.** Each member runs `solve_sector(Within(...))` and inherits:
- B2's margin inflation (NiO 4.6× too big, *(r2)* not "13×");
- C7/S1's missing states;
- the GS-alone cap applied to a multi-state union, and with it S2's symmetry breaking (SMO).

The GS cost of K members is one more cap-bound re-solve per member (the N2 `Lowest` solves alone took 513–954 s). MV
§J4's "hours on SMO" covers DC trials plus the GF stage. S2 lands in step 3. Fix B2, C7 and the `Within` cap before MV 7.

**J8. Warnings.** One end-of-`calc_gs` summary (§4 UX), merging:
- the review's items;
- MV D6's dropped-weight warning, with *(r2, physics)* the tail estimated from window counts, because the per-state
  cut −τ ln 1e-4 ignores the density of states;
- MS T4.5's per-shell lines.

The stderr tiers are in §4.

**J9. MV §J items rev. 1 left open** *(r2)*:
- **§J5, cap probe for the N±1 members.** The probe measures the seed sector only. Resolve the cap per member, or for
  the `Within` union of the dominant member with headroom (P10).
- **§J7, default revisited after cost:** O1, `auto` (user).
- **§J8, versioning.** J8's summary fields, MS's archive datasets (`Spherical shell l`, per-shell N0), the
  reinterpreted `mv` token and the P10 recorded cap all need a `schema_version` bump and `required_features`
  entries. RSPt's `mixed_valence={0: mv}` (`lib.py:439`) is keyed on shell 0 and breaks under MS unless MV's scalar
  tri-state lands first.
- **MV's own inconsistencies to fix in that doc (done 2026-11-02, step 11.6):**
  - the header still says rev. 1;
  - D6, Verification and commit 13 still describe the `seed_valence_width` rename that the repurpose section dropped;
  - §J1 puts the flag in `solve_ground_state`'s signature;
  - D3's frozen rule is moot.

### 6.2 Which plan changes

| plan | change |
|---|---|
| MS §1, §3, D3, T4.4, T4.6, T5.1, §7.2, tests, Q2 | Remove the frozen core (J1). Keep `shell_has_bath`, `active_impurity_indices`, and the Tr ρ_p = 6 test. Implement T4.4 without `frozen_groups`, rather than ship it and delete it. |
| MS F1 | Freeze a shell only when the user declares it core (O5); otherwise check Tr ρ_core. |
| MS T4.7 | Projector into `SectorPlanStore` at construction (J5). Until T4.7 lands, `dc_criteria` raises on `len(shell_ls) > 1` (§6.3 step 7). |
| MV header, D3, D6, D7, §J1, commit 13 | J3 thresholds in τ; `auto` never raises; the flag stays out of `solve_ground_state`; drop the rename remnants (J9). |
| MV D3 | Depends on P7, S1, M4, C2', C7 (J2, J3). |
| MV commit 1 | Preceded by the keyword-only commit; non-zero values raise until MV 5 (J4). |
| MV commit 4 | Take its bit-identity baseline after MS T4.5: both edit `gs_report.py:183-248`. |
| MV commit 5 | Consolidate the Boltzmann helpers; add the ensemble switch to the spec here. |
| review P1 | Per J1. |
| review B1 | Trajectory bound; only beyond `cut + margin` (J2). |

### 6.3 Merged order

One concern per commit. *Steps* group commits. Each commit lists its repository when it is not impurityModel. Every
step touching a collective runs `-n 1/2/3`.

| step | commits | collective? | acceptance |
|---|---|---|---|
| 1 **DONE 2026-10-31** (commits `d6758b22` P7, `7b5059cd` S1, oracle `9f1b0609`/`7d4fb258`; gate -n 1/2/3 green). S1 landed as widening by *single excitations*, not H\|seed\>: H conserves the class, so its Krylov space never leaves it. Skipped (with a `widen_skipped` trace note) above `GS_CERTIFY_MAX_SEED`. Open from the oracle: seed 1 m0 (HF, sector N=2 0.008 high), seed 1 m8 (window one level short). | **P7** (delete both guards, canonicalise the split); **S1** (one-job plan ≠ certificate); `oracle.py` → test (fix `n_found=None`, straddle tolerance; ~10 pinned models ~10 s, `use_hf_seed=False`, plus frozen-empty and two-group toys; full sweep under `-m benchmark`; compares every solved sector) | yes: 1-det seeds hit empty ranks, so **-n 3** | pinned models exact; inject the old guard → test fails |
| 2 **DONE 2026-10-31** (commits `9d8e3ca2` bcast, `655ece90` frontier-fit + budget-presence check, `e73f3ed1` `config.assert_uniform`, `0e6cfb96` entries, `ca81a714` M5; gate -n 1/2/3 see below). Every public entry (solve_ground_state, calc_selfenergy, run_spectra, susceptibility, RIXS maps on `basis.comm`, DC entries on COMM_WORLD) raises `NonUniformKnobError` on every rank when a knob differs; `GF_SECTOR_CACHE_DIR` is `uniform=False`. M5: RIXS probes the parent comm once. The GF budget's None-ness is covered by the entry check, no extra bcast. Open: `IMPMOD_MATVEC_PROF` is unregistered (profiler, no collective); GF-side consumers outside an asserting entry would not be checked. | **M1** bcast `safety` + unconditional `_frontier_matrix_fits` allreduce; `config.assert_uniform(comm)` at every public entry; **M5** (`gf_shift_recycling` comm) | yes | MPMD knob-on-rank-1 test raises on all ranks; inject a rank-local knob read → test hangs or fails |
| 3 **DONE 2026-10-31, profiles outstanding** (commits `2b1d3962` C4, `37d7a2ad` M4, `776e187d` C2', `5c5c86a3`/`289e1680` S3, `69c71c1a` C7 detection, `5697f8eb`..`90d7120e` S2 orbits + review/gate fixes; gate -n 1/2/3 green after `62d450fd`, targeted re-runs after `90d7120e`). Orbits (`ed/orbits.py`, `GS_ORBIT_CLOSURE`): full spin arrangements of a pair pattern at \|2Sz\|<=M plus the closure of the detected signed permutations, ranked by summed score; cubic d7 at cap 160: quartet equal to 2e-13 vs 1e-7 split without. Not done: orbit-aware `slaterWeightMin` (effect ~1e-12, skipped), GF-side closure and the Σ↑=Σ↓ acceptance (later step), SMO 4-bath / CrI3 / NiO -n1 vs -n4 profiles: **run 2026-11-01 (step 4.5, `debug/gs_review_2026-10/logs/prof_*_step45.log`)**. SMO 4-bath -n 4: the quartet is degenerate in the printed table (old log: spread 7.6e-6), no split line, manifold_check 4/4; wall 2175 s vs 1986 s. CrI3 -n 4: still 2 window states, but now reported (converged False, limited_by cap, window_complete False 2 of 3, WARNING naming the cap; old log said limited_by memory); wall 2466 s vs 2346 s; the missing members are the step-10 fix (C7/B2/Within cap). NiO -n 1 vs -n 4: same E0, converged True, limited_by None, 3 of 3 window states at both (M4 holds); 452 s vs 356 s. Open: retain() generates every arrangement of each kept orbit (cost at production size unmeasured); budget charges an orbit its full size, so a budget smaller than every candidate orbit stalls the expansion (reported cap-bound). | **S2** (orbit-closed growth and truncation in the `Within` expand; SYM); **S3** (never drop an in-window class; record the union cut; `converged: False`); **C2'** (depth on complement; certified=False); **C4** (replace record, `only=None`); **M4** (`memory_bound` only on an actual lowering, accumulated); **C7 detection** (per-label window counts, trim warning) | yes, -n 3 (orbit closure adds collectives) | S2 acceptance (degenerate members equal to roundoff under a binding cap; SMO 4-bath quartet exact); S3: CrI3 keeps 4 states (or reports the cut); `c2_small` unit test; C4 call-site test; NiO -n 1 vs -n 4 same flag; CrI3 flags its window |
| 4 **DONE 2026-11-01** (commits `ddaf34ce` target, `addb2806` P8 DIIS, `2b5d3834` P8 ±1 fallback, `1392f630` split detection, `a179bfc3` P1a, `f7fc6969` [H, N_core] warning; full gate -n 1/2/3 green on every rank file: 3786/3894/3902 passed; `make check`: ruff findings identical to `b663c7fd`, mypy clean, black flags only the untouched `gs_oracle.py`). P8: DIIS rung after the mixing ladder: complex 39/40, 99/100, real 60/60, 100/100 (convergence target met). **Off-by->1 target missed**: 11/100 real, 8/100 complex (target ≤ 10, ≤ 4). The new cases are newly converged models; on those off-by ones the DIIS point is the lowest of 60 random-start HF fixed points (`step4/hf_offby.log`), but on other rescued models DIIS often lands on a *higher* aufbau fixed point (Opus review: 11/56, 9/26, 6/8, ΔE up to 2.29). End to end 0/200 wrong E0 before and after, and 0 on every ladder-failed model vs the old scan path. Review (Opus) open item: with ≥ 2 groups and non-convex E(N) (attractive U) the ±1 descent stops in a local minimum the old scan's diagonal move escaped (`_two_group_siam(e_imp=1, u=-4, v=0.05)`: 2 eV high); the accepted-seed path already had this, 2b5d3834 extends it to rejected seeds; 0/120 on Kanamori U>0 two-group models. Level shift rescued nothing, not shipped. Rejected seed: ±1 walk from N0, notice at V_RESULT. Split detection: SU(2) only (`manifold_check.multiplet_split`, ERROR line); CrI3/SOC not covered. J1/O5: DC freezes nothing; the commutator warning uses `PARITY_TERM_TOL` (relative), which closes that O6 item. Open: `get_spectra._resolve_shell_roles` still infers the core from bath counts for models without `valence_l`/`core_l` (legacy fallback, raises on ambiguity; out of step 4). **P8 target set 2026-11-01, before the change** (`hf_conv.py` through `hartree_fock_occupation`, i.e. with both starts and the mixing ladder; `debug/gs_review_2026-10/step4/hf_conv_baseline.log`). Baseline: real 58/60 and 92/100, complex 13/40 and 43/100; converged seed off by >1 sector 10/100 real, 4/100 complex. **Target:** complex ≥ 90% on both seeds (≥ 36/40, ≥ 90/100); real does not drop; off-by->1 does not rise (≤ 10/100 real, ≤ 4/100 complex); every model the ladder converges today is bit-identical (the new solver only runs after the ladder fails). | **P8** (level shift/DIIS, ±1 fallback, visible rejection); degenerate-split **detection** (rank 0 + bcast); **J1/P1a** (delete the DC bath-less rule, rewrite `test_fixed_dc`; spectra `[H, N_core]` warning on user-frozen shells) | yes | complex/SOC HF convergence target set before the change; detection fires on an injected orbit-breaking cut |
| 5 **DONE 2026-11-01** (commits `a5e7e971` keyword-only, `68c264f0` MV 1, `09c5489f` MV 2 + P11, `137f0b7d` review fixes; full gate -n 1/2/3 green on every rank file: 3815/4008/4016 passed). MV 1: seed widening removed, non-zero `mixed_valence` raises `ValueError` at `prepare_solver_basis`/`find_ground_state_sector`/`solve_ground_state` and (after review) at spectroscopy input build; `_nio_workload` and the `BLBENCH_MV`/`BICG_MV` benchmarks no longer widen (their recorded numbers need re-baselining on next use). MV 2 + P11: `calc_energy` returns `SectorEnergy`, `SectorSearch.solved` (`SectorRecord`s in solve order) and `.diagnostics` (margin to N±1 in error bars, second difference); recorded only, WARNING when margin < 1. **Not done:** the one-sided check E_N(μ) ≤ E_N(0) − μ n_N(0) (needs n_N at a second shift: belongs with the DC consumer, MV 3/B1); `get_energy` as an object for N±2 (MV 3); the legacy-scan ±1 pass was dropped as dead (the scan always reaches the winner's neighbours). The `test_orbits` quartet control failed once under `-n 1` before MV 1 and passed in every later run; cause not found. | keyword-only `mixed_valence`; **MV 1** (non-zero raises); **MV 2 + P11** (`SectorSearch.solved`, ±1 neighbours, one-sided HF check) | MV 1 changes seeds: -n 3 | `mixed_valence=1` fixtures re-baselined, noted in the commit |
| 6 **DONE 2026-11-01** (commits `d486fa20` H3 move, `5b25f220` H3 unify, `436c6a24`+`8d5af590` M3, `f2ffdac9` P4; gate -n 1/2 green: 3861/3969 passed). H3: `ed/degeneracy.py`, one chain rule everywhere; threshold written first (χ Curie 1e-10, GS/Σ fixtures bit-identical) and nothing moved on the suite. M3: `rank=` gone. P4: `energy_cut` deleted (never a TOML key, so no REMOVED_KEYS entry; the CLI flag now fails as unrecognised). | **H3** public degeneracy helper; **M3** (and the `rank=` callers); **P4** | no | χ Curie term unchanged on fixtures |
| 7′ (new, runs first; plan `~/.claude/plans/tidy-cooking-flask.md`) | **B2** per-verdict margin `within + _energy_error(v)` (own A/B, before C7); **C7 fix**: a `within=` class expansion converges every in-window state (`_expand_job`), and `window_deficit` counts states per class (CrI3 reduced cluster expects 4); GF-side SYM **measurement** only (G summed over each degenerate level, Σ↑=Σ↓ under binding GF caps; xfail if it fails) | yes, -n 3 | oracle seed 1 m8 exact; CrI3 reduced cluster expects 4; NiO thermal basis moves with Σ inside a tolerance written first; SMO 4-bath quartet exact; profiles `prof_*_step7p.log` |
| 7 | **MS T4.1–T4.5** (impurityModel side; impmod_interface G2 and rspt2spectra G3 in their repos); `dc_criteria` raises on multishell until T4.7; MS 2p@−60 Ry fixture measurements (J1); Tr ρ_p = 6 test and T5.1 | no | single-shell archives bit-identical |
| 8 | **A1** series (a) plumbing, (b) P3 χ, P2 spectra+RIXS, P5, P6 one per commit, (c) **P10** caller-resolved cap (impmod_interface + archive attribute) | (c) yes | (a) bit-identical; (b) each with its A/B threshold written down first |
| 9 | **B1** (trajectory bound, beyond `cut + margin`); **MV 3** warning + **J8** summary and stderr tiers | yes | winners unchanged on all six profiled workloads; WARNING lines on SMO 4-bath ≤ the stated budget, each event once per channel |
| 10b | **`Within` cap (O3-2)** (multiplicity × walk per-state basis, memory-bounded, resolved with P10); **B3** (reuse the walk's window-class keys: priced first; the plan-store key includes `truncation_threshold` and the store holds replicated scalars only) with J5's χ-sized `jobs_at` margin. *(C7 fix and B2 moved to step 7′.)* | yes, -n 3 | CrI3 -n 4 keeps 4 window states (or the memory guard is named); SMO 3-bath per-member accuracy ≥ the walk's; quartet exact |
| 10c | **GF-side SYM fix**, only if the 7′ measurement fails: spin-image GF units placed in one MPI colour, support cut decided on pairs, tie-break in `collective_first_keys` respects spin flip; `assert_uniform` on GF consumers outside an entry | yes, -n 3, `skipif(size == 1)` | the 7′ xfail flips to pass; an injected unpaired unit fails |
| 11 | **MS T4.7** (active projector, J5 store) | yes | single-shell DC numerics bit-identical (projector = I) |
| 12 **MV 4–8 DONE 2026-11-03** (commits through `298dc02e`; tri-state `mixed_valence` auto/on/off, Σ/DC default `auto`; `calc_gs`, `calc_selfenergy` and the occupation DC consume the charge-sector ensemble; full gate -n 1/2/3 run, two stale tests re-pinned; Opus reviews at 12A and 12B/C). **MV 9–13** (χ, spectra, RIXS, KMS, docs) remain. Original scope: | (report split, baseline after T4.5), **MV 5** (seam, Boltzmann consolidation, spec switch), **MV 6–8** (GS report, Σ consumer, DC continuous). Then **MV 9–13** (χ, spectra, RIXS, KMS, docs) | yes; consumers raise collectively (`_raise_together`, `selfenergy.py:62`) | MV's own verification list |
| 13 | **B4** (after a production-budget re-measure), **B5** (geometric-tail rule), **B6**, **B7** (-n 4); **B8** (states reused with default DC tolerances, sector choice only when loosened) | yes | each gain measured on the change |
| 14 | hygiene | no | — |

**Step 7 (2026-11-02), MS T4.1-T4.5 DONE, remainder open.** Commits `8a3eae49`..`9c32a7e1` (shell helpers, `impurity_shell_rhos` /
Casimir `shell_ls`, `ImpurityModel.shell_ls`, `spin_pairs`, `prepare_solver_basis`, callers + `dc_criteria` guard, reporters),
then `4a08bf47` (guard reads the declared `shell_ls`: the full -n 1/-n 2 gates each failed one test, `eye(4)` rotation),
`19eb680f`/`48b38a73`/`2d3e848d` (Opus review: Casimir operators used group-dict order instead of sorted rotation rows; a
shell-keyed `nominal_occ` was read as group-keyed when the keys coincided; both fixed with tests, bug-injection fails them).
Gate: -n 1 3890 passed + 1 failed, -n 2 3998 passed + 1 failed (same test, fixed in `4a08bf47`), -n 3 4007 passed (with that
fix); after the review fixes only targeted runs (symmetry + `test_fixed_dc`) at -n 1/2/3, all green. No full re-gate yet.
`make check`: 61 ruff findings (baseline 62). `shell_ls=None` stays the single-shell path.
- **Open:** 2p@-60 Ry fixture measurements (J1), Tr rho_p = 6 test and T5.1, `get_spectra._resolve_shell_roles` explicit
  roles, impmod_interface G2 / rspt2spectra G3 (needs the user before touching impmod_interface).
- **Open (review, low):** `gs_report` "skipped (partial shell)" text prints whenever per-shell ops are `None`, even if both builds
  failed for another reason; `ImpurityModel.resolved_shell_ls` still raises on a non-shell column count (no production caller);
  a single partial shell collapses to `SolverBasis.shell_ls=None`; `_per_group_occupation` falls back silently to total-based
  filling when a shell owns no group; the per-shell Casimir block moved out of the totals `try`, so a degraded legacy path now
  also runs it (numerics unchanged).

**Step 7′ (2026-11-02), DONE with open items.** Commits `4d5dbae1` (B2 per-verdict margin), `28aa0889` (C7 counting:
`ClassVerdict.levels`, `window_deficit` expects the per-class sum), `b456415e` (GF-side SYM measurement). Gate: -n 1
3871 passed, -n 2 3979, -n 3 3987, 9 xfailed, every rank file green (`debug/gs_review_2026-10/step7p/`).
- **Not met:** oracle seed-1 m8 is *not* exact. The 4th window state (0.2958) has no overlap with the class basis, so CIPSI
  cannot reach it; a `Within` class expansion passed at -n 1/2 and failed at -n 3 (layout-dependent), so it was dropped.
  Kept as a non-strict xfail (`UNREACHABLE_WINDOW_STATE`). User decision: defer (options left: a widening round per
  in-window class, or a deflated restart from the class seed).
- **Not met:** CrI3 reports `window_expected 3, found 2`, not 4 (`prof_cri3_n4_step7p.log`, unchanged from step 4.5; limited by
  the cap, a 10b target).
- Profiles: SMO 4-bath quartet exact (4/4). NiO 10-bath -n 1 identical; -n 4 basis 136,765 -> 136,097 dets but `converged`
  flipped True -> False (`limited_by: cap`), winner and window unchanged: investigate in 10b.
- GF-side SYM: **holds.** G_up = G_down to ~1e-15 with all four units cut to 3,000 of 3,920 dets, at -n 1/2/3
  (`test_gf_spin_symmetry_under_caps.py`; injecting a unit-dependent cut fails it). Suggests 10c may be unnecessary; the
  fixture is a unique singlet with no exact boundary ties, so this is not proof for degenerate manifolds.
- Opus review fixes: `found` now counts to the same absolute window edge as `expected`; the SYM test uses CAP 2500 with a
  sensitivity guard (capped G must differ from uncapped by > 100x the tolerance). Open: the levels solve runs in pass 2 only
  and is untested end to end; its cost on dense low spectra (SMO regime) is unmeasured; `expected = max(...)` drops
  cross-label information (still a sound bound).
- `make check`: 62 ruff errors repo-wide, pre-existing; none new in the files this step touched except one fixed.

**Step 8 (2026-11-02), A1/P10 DONE with open items.** Commits `32d6ac5b` (A1 a, GroundStateSpec), `8816783e`+`91332395` (P10 CapLedger,
`CapLedger` exported), `c99e4c3d` (A1 a for the DC entries), `3987988b` (P5), `0b3b8670` (P6), `f6f70abb` (P3), `f2c1ec69` (P2),
`429fde12`, `170cc323`, `d14922bc` (review fixes). Gate: -n 1 / 2 / 3 green on every rank file (4011-4020 passed, 9 xfailed); the
-n 2 run ended by killing the idle PMIx-finalize hang after all tests had passed. Targeted DC/memory tests re-run at -n 1/2/3 after
the last two fixes. impmod_interface (not pushed, by instruction): `ffddb20` writes the attribute `"DC ground state cap"`.
- A1 a, P5, P6: bit-identical with the knobs unset. P3 (Curie 0.2412685545777933 within 1e-8) and P2 change defaults, see below.
- P10: `CapLedger(ceiling)` lowers an auto cap to the ceiling (rank 0's, broadcast) and records the resolved cap; the cap ladder
  re-records the cap it settled on (`170cc323`). A user-set cap is never touched. Carries DC -> Σ within one iteration only.
- **Decisions (user, 2026-11-02): keep the unified defaults.** (1) χ now takes `BasisOptions.chain_restrict` (default True), so every
  default χ run changes from False. The Curie A/B (SIAM fixture, no cap binding) cannot see this; the review argues the
  distance-based freeze is symmetric, unverified. P2 keeps spectra at `chain_restrict=False` for the complete multiplet, so the
  two now disagree. (2) TOML spectroscopy takes `solver.dense_cutoff` (500, was 1000); `scripts/spectra.py` and the end-to-end tests
  still use 1000. No spectrum/RIXS A/B was run for P2 (tests are source-text checks plus the suite).
- Review (Opus) fixed: unlimited cap ran the ladder (`OverflowError` risk) `429fde12`; ledger recorded the pre-ladder cap `170cc323`.
  Open: the ledger-ceiling test compares three live `MemAvailable` reads (may flake); the `determinant cap:` log line ignores the
  ceiling; intermediate commits `8816783e`/`91332395` were red (NameError, fixed in `c99e4c3d`); source-text tests do not catch a
  dropped name in an explicit `solve_kwargs` subset; ceiling `bcast` is gated on the argument `ledger is not None` (uniform at
  every caller); no `schema_version`/`required_features` bump or archive reader for the recorded cap.
- `make check`: ruff reports 22 errors under `src` and more in untracked dirs, not re-triaged; files touched here pass ruff.

**Step 10b (2026-11-02), started: finding + one fix.** The auto cap already prices the window solve (`estimate_gs_peak_bytes`
takes `window_states`/`gs_num_wanted` from the measured connectivity), so O3-2's "sized for one state" is only partly true today.
CrI3 -n 4 (`prof_cri3_n4_step7p.log`) is memory-bound, not cap-bound: the measured-memory guard tightened the auto cap
663,716 -> 661,666 at 1.6 GiB/rank, so a larger cap cannot add the missing states; the window warning now says so
(`_raise_cap_advice`, `sector_solver.py`). `window_expected` is 3, not 4, by design: it is a lower bound that does not read a
class's multiplicity above its lowest level (`window_deficit` docstring). **SMO 3-bath -n 4 run (`prof_smo3_n4_step10b.log`, 965 s):**
quartet 4/4, `multiplet_split: None`, `limited_by: memory`; the measured-memory guard tightened the auto cap (292,684 -> 290,600 at
RSS 1.2 GiB vs a 915 MiB budget), as it did in the old log (486,788 -> 452,240; this run had less free RAM, so the caps differ and the
bases are not comparable). **Verdict: the `Within` cap is not raised.** On both memory-bound workloads (CrI3, SMO 3-bath) the guard,
not the auto cap, ends the expansion, so a larger resolved cap has nowhere to go; O3-2 is closed as "memory-limited, no change"
unless a machine with more RAM per rank shows the auto cap binding first. **B3 deferred** (not built): reusing the walk's
window-class keys holds distributed key arrays across the DC walk, raising peak memory exactly where memory is the binding resource,
and needs a new store design (the store holds replicated scalars only; its key includes the cap).

**Step 11 (2026-11-02), started: 11.0 measured, 11.1 and 11.2 landed; the guard stays until 11.0b.** *Decision (user):* the DC
shift acts on the **hybridising shells** (`SolverBasis.active_impurity_indices`, the rule of `HYB_WEIGHT_REL_TOL`), "for now".
RSPt sends shell structure and a complete `u4` but no core flag, so U cannot carry a role; hybridisation is the de facto
declaration. 11.4 must make it visible (a log line naming covered and excluded shells) and raise if an excluded shell has levels
near E_F. An explicit solver-line role (option 2) is deferred. **11.0 (J1 measurement, `calc_gs` on a 2p + 3d + 4-bath fixture,
p at -60, guard bypassed, nothing frozen, d-d U 0.5, p-d U 0.6):** the unfrozen deep shell **does** change the search, three ways.
(1) The seed is the whole `N_imp` combinatorial space over p+d, not the d shell: C(16,8) = 12,870 determinants at the nominal
sector (11,440 at 7 and 9, 8,008 at 6, 4,368 at 5), against C(10,2) = 45 for the d block alone (arithmetic, not run), so every
sector exceeds `GS_CERTIFY_MAX_SEED` = 1,000 and is solved by one expansion, **not certified**; `seed_class_plan` returns `None`
above `max_seed`, so no penalty or scale exists to report. (2) The occupation walk visits N_imp = 7, 9, 6, 5, i.e. sectors whose
change is a p<->d charge move as often as a bath one. (3) The seed spans N_active = 0..N, so the plan store's
"shift is constant on the seed" premise fails once P is active-only. The GS still keeps Tr rho_p = 6 (natural occupations
1.0000 x6), from the -60 level and not from a constraint, as J1 intends. **Consequence:** lifting the guard (11.4) needs
**11.0b** first: a *seed-only* per-shell window for the bath-less shell (width >= 1 so CIPSI can still reach core-hole
determinants; not a freeze) built in `generate_initial_basis`, plumbed through `ManyBodyBasis`/`sector_key`, and the walk and
`SectorPlanStore` counting N over the active shells (11.3). **Landed:** 11.1 `4ccb4d9a` (`fll`/`amf`/`nominal` refuse a declared
multi-shell model; `sigma_inf_dc` is the multi-shell scheme), 11.2 (`shift_input`/`shift_solver`/`active_indices` in the two
DC contexts, exact `np.identity` and `None` on a single shell; `test_dc_multishell.py`, bug-injected with `P = I`).
Not done: 11.0b, 11.3-11.6.

**11.0b landed (2026-11-02).** `basis_generation.SeedWindow` (a `set` subclass passed as `frozen_occupations`) seeds the
groups of a bath-less *shell* (decided per shell: a hybridising shell has bath-less groups too) as a window: each group
ranges freely, the total over the window stays within `DEEP_SEED_WIDTH = 1` of nominal, and the other groups move to
compensate. It is not a freeze: HF does not pin it, the report does not call it frozen, the leak warning skips it, and
CIPSI may leave the window. `SolverBasis.seed_window` is `None` on a single shell (bit-identical) and `selfenergy`
passes it. Measured on the J1 fixture (2p + 3d + 4 baths): seed 765 determinants (was 12,870, over
`GS_CERTIFY_MAX_SEED`), `calc_gs` Tr rho_p = 6. Still to do: pass it from the DC search (11.4).

**Step 11 DONE (2026-11-02): MS T4.7, multi-shell double counting.** Commits 11.1–11.4 plus 11.0b. The DC search accepts a
multi-shell model. *Rule (user decision, option 1, "for now"):* the shift, the occupation target and the DFT reference act on the
hybridizing shells (`SolverBasis.active_impurity_indices`, the rule `HYB_WEIGHT_REL_TOL` uses); rank 0 logs "DC shift acts on
shell(s) l=… (hybridizing); excluded: l=… (no bath)". A skipped bath-less shell with a level within
`dc_criteria.DC_INACTIVE_SHELL_EF_WINDOW` (1.0, units of `h0`) of E_F raises `ValueError` on every rank before any collective: that
is a valence shell the hybridization cut dropped, not a core. An explicit solver-line role (option 2) stays deferred.
`SectorPlanStore(active_orbitals=…)` is fixed at construction and the class `n_imp` (the energy's slope in the shift) is read over
those orbitals (J5); the existing `N_IMP_SPREAD_SAFETY` margin plus `violated()` recertification is the χ-sized margin, no second
one was added. The deep shell reaches the solves as `SeedWindow` (11.0b), in the walk, the sector solves and the final ground
state. `dc_static` `fll/amf/nominal` refuse a multi-shell model; `sigma_inf_dc` is the scheme. Totals (`nominal_total`,
`max_occ_total`) stay whole-impurity on purpose: the deep shell is held out of the walk, so every sector carries the same p count.
Measured: `fixed_occupation_dc` on p(6)+d with U=3, target 1.0, gives dc = 0 on p and 1.5 on d, sector 7; 13 evaluations, one sector
seed above `GS_CERTIFY_MAX_SEED` (solved uncertified). `-n 1/2/3` green on the new tests; bug injection (P = I) fails seven of them.
Single shell: `seed_window` and `active_impurity_indices` are `None`, the projector is the identity object, nothing changed.
Open: which sector is the uncertified one (not looked at); the interface-side MS T4.7 text (impmod_interface repo, doc only, not
touched here).


**Step 9 (2026-11-02), started: MV 3 field landed, B1 deferred with a measurement.** `solve_ground_state` now puts the walk's
`SectorDiagnostics` into `convergence_report["sector_diagnostics"]` (`3c726664`; `-n 1/2/3`, bug-injected). **B1 (O6 gate):**
the step-7′ logs carry no per-class lines (`certify` is 2107 s of the 2149 s SMO 4-bath wall), so the B1 margin cannot be set from
them; the walk's per-sector times are enough to bound the upside instead. SMO 4-bath: the HF-seed sector (363 s) is certified
first, so B1 has no E_best to cut against there; the sector at +0.008 Ry (667 s) is inside any margin that is safe; only the
+0.59 Ry sector (38 s) is skippable. Pass 1 (`class_expand`, 1254 s of 71 expansions) is where the time goes, and B1 does not
touch it. **B1 is not built**: at most tens of seconds of 2149 s, against a new `e_cut` channel through a collective, bounded
sector energies leaking into `neighbour_totals`/`margin`, and the store/DC guard. Revisit if a workload whose first sector is
far above the winner shows pass 2 dominating.

**Step 9 DONE (2026-11-02), partial.** Commits: `3c726664` (MV 3 `sector_diagnostics` in the convergence report), `4d38c514` (B1
note), `a6c6a8b2` (P9 neighbour-in-window warning with ΔE/τ), `fb03064b` + test (J8: unset-`GS_MAX_BLOCK_WIDTH` nag dropped, plural fix),
and the review fixes (warning off for the DC search via `warn_neighbours=False`; infinite margin stored as null so the
statistics file is strict JSON; negative gap worded "BELOW"). Gate: -n 1 3911 / -n 2 4018 / -n 3 4026 passed, 9 xfailed, rank files
green. Opus review: no MPI hazard; three findings, all fixed. **Step 9 finish (2026-11-02), J8:** `convergence_summary` (one end-of-`calc_gs` line: PT2 residual vs tolerance, limiter, winning
sector and margin), `degraded_warnings` (stderr copy, rank 0 once, only for an unconverged expansion or an undecided sector),
and `utils.Heartbeat` (rank-local, no collective, 300 s rate limit) in the CIPSI cycle loop. **MV 3 N±2:** investigated and
deliberately not built: at a ±1 descent end the second difference is positive (attractive-U zigzag, measured on 14 models), so a
convexity trigger is vacuous, and the margin<1 trigger does not fire for exact solves; only an unconditional N±2 probe or a physics
signal would catch it. The u=-4 case stays a strict xfail with that reason. **Not built, still open:** B1 (measured, deferred above),
the "solving sector N=…" start line, the thermal-regression-beyond-error-bars and memory-bound-withheld-mass stderr events,
MV 3's one-sided E_N(μ) ≤ E_N(0) − μ n_N(0) check (needs n_N at a second shift: a DC-consumer item) and `get_energy` as an
object (a refactor with no consumer yet), and the HF off-by->1 / DIIS / oracle seed-1 m0 items. Steps 11–14 not started.


**Step 0 (2026-11-01).** Full gate on the tree at `84d9d9ad`+xfail: -n 1 3867 passed / 125 skipped / 6 xfailed (695 s),
-n 2 3975/17/7 (1211 s), -n 3 3983/9/7 (2230 s), every rank file green. The step-5 `test_orbits` -n 1 quartet-control
flake did not reproduce: 40 isolated -n 1 runs, 10 runs of `test/symmetry/` at -n 1, 10 at -n 3, plus the gate (0 failures).
Left unexplained; if it recurs, the control `np.ptp(split) > 1e-8` is the assertion that fails (a layout-dependent
tie). Oracle CI budget (O6): the whole suite is 11.5 min at -n 1; the oracle adds no separate budget beyond that.
`_two_group_siam(e_imp=1, u=-4, v=0.05)` is now a strict xfail in `test_sectorization.py`, owned by step 9.

**Orphans from steps 1–6, placed (2026-11-01).** `test_orbits` -n 1 flake: step 0. Σ↑=Σ↓ acceptance, GF-side closure,
`slaterWeightMin` audit: measured in 7′, fixed in 10c. `retain()` cost at production size and the stalled-budget case:
measured in the 7′ profiles, fixed in 10b if they bind. HF off-by->1 misses, DIIS landing on a higher fixed point, the
±1-descent local minimum (add `_two_group_siam(u=-4)` to the oracle as xfail in step 0), oracle seed 1 m0, the step-5
deferrals (one-sided E_N(μ) check, `get_energy` for N±2): step 9. `_resolve_shell_roles` core inference: step 7.
`IMPMOD_MATVEC_PROF` registration: step 14. B8 and the `_nio_workload`/`BLBENCH_MV`/`BICG_MV` re-baseline: start of
step 12 status (2026-11-03): MV 4-8 plus the Phase A review fixes are committed; full gate -n1/2/3 green (4026/4134/4142 passed). MV 9-13 (χ, spectra, RIXS, KMS pole-residue diagnostic, docs) committed 2026-11-03; Phase B gate -n1/2/3 green (4021/4129/4137 passed) plus Opus-review fixes (blocked-only window fallback, KMS log-ratio, docs, rank-0 merge failure raised together). Open: KMS not wired into Σ/χ reports, weight-discriminating GC oracles for χ/spectra, streamed merge. step 13 C0 re-baseline (2026-11-03, -n 4, NiO 10 bath): self-energy benchmark ~150 s (acausal, cap-limited; timing only, SELFENERGY_BENCH_ALLOW_UNPHYSICAL=1); block-Lanczos 40 s (TRLM/array 0.09 s; sparse-kernel FULL p=2, 40 its 12.5 s = 311 ms/it, |QHQ-I| 3e-14); BiCGSTAB at BICG_NBATHS=10 BICG_TRUNC=30000: width 1 0.75 s/29 its (25.7 ms/it), width 5 5.2 s/34 its (realaxis), 5.3 s/37 its (matsubara); default size not measured (>49 min). Benchmark API rot fixed in b8194ca4. step 13 C1 (2026-11-03, commits 5af9c510, 8e4a70b5): `fixed_occupation_dc(handoff=GroundStateHandoff())` keeps one ground state (the solve at the returned shift; the search measurably returns a non-last point, so it re-solves once then) and `calc_selfenergy(handoff=...)` reuses it only if dc, the spec/cap/orbital key and the communicator match, `mixed_valence` off, verdict allreduced; else a note and a fresh solve. Limits found: (1) Sigma defaults to `auto`, which forms the ensemble path, so the default configuration does not reuse (one-member/ensemble handoff not built); (2) impmod_interface damps the DC against the previous iteration (`previous + alpha*(dc - previous)`) and runs DC and Sigma as separate calls, so the dc Sigma sees equals the solved dc only on the first iteration or with alpha = 1 -- the call site is not wired and the saving is unmeasured on NiO/SMO. step 13 C1 SKIPPED by user decision (2026-11-03): the sector search must be fast enough that DC-to-Sigma reuse does not matter; the opt-in handoff stays in the tree, unused. step 13 C2 measured (NiO 10 bath linked chain, mixed_valence off, -n 4, rank 0): calc_gs 368 s (walk 189 s, thermal 148 s); h_matrix builds: incremental reuse 384 builds 45.3 s, verdict-solve PHP builds 50 builds 21.2 s (5.7% of calc_gs, max 186k dets), fresh frontier-kept 45 builds 10.3 s. Reusing the held matrix can remove only part of the 21 s (a reuse is an incremental build, not free), so the realistic saving is ~3% < the 5% threshold: not built. step 13 C3 (geometric-tail stop) NOT built: the lever needs a capped refinement to measure (not available locally) and the earlier experiment (`review_numerics/r3b.py`) shows it loses accuracy. step 13 C4 (NiO 10 bath, off, -n 4): GS report 30.6 s of 362.7 s (8.4%), of which `build_density_matrices` 23.9 s; inside it `np.unique(axis=0)` on the scattered (N-1) keys was 12-18 s/rank, alltoall 3-8 s. Replaced by a uint64-word `lexsort` grouping (same partition): report 16.7 s of 352.7 s (4.7%, below threshold), density-matrix build 9.8 s; the alltoall (3-8 s) is the remainder and was not touched. step 13 C5 (class-job task parallelism on sub-communicators) NOT built: `certify_sector` jobs are not independent -- pass 1 feeds `finished` (leak detection) from earlier jobs of the same label, `e_min` prunes the kept keys and orders pass 2, and the best solve is kept alive across jobs -- so a static job-to-colour assignment changes the verdicts unless pass 1 is made order-free first; no local -n 4 SMO 4-bath measurement of the latency-bound eigensolve exists to price it. Left as a design item. step 13 C6 (O2, DC tolerance knobs): unchanged -- C1 skipped, C2/C3/C5 not built, C4 only 3.7 s/rank-independent gain; nothing found that makes the loose DC tolerances unnecessary. step 14 (hygiene, 2026-11-03/04): IMPMOD_MATVEC_PROF registered (uniform=False); the `determinant cap:` line names a ledger ceiling that held the cap down; the ledger-ceiling test pins the memory probe; a test pins the `solve_kwargs` subsets of the DC call sites (the sector search now also passes `use_hf_seed`, no change at the default); `make check` green under `src` (ruff, black, mypy). `_resolve_shell_roles` already prefers explicit `valence_l`/`core_l` and raises on ambiguity; the partial-shell wording and the `_per_group_occupation` fallback (docstring) were judged accurate. The three remaining xfails still fail under `--runxfail` (not stale). Phase C/D gate -n1/2/3: 4041 / 4149 / 4157 passed; the C1 handoff tests were red on ranks >= 1 (they asserted rank-0 output on every rank) and were fixed in the commit after the gate, targeted -n1/2/3 green. Deferred by earlier user decisions and not built: C7 unreachable window state, B1, B3, the 10c GF-side SYM fix, MS option 2, impmod G2 / rspt2spectra G3, C3 (geometric tail), C5 (class-job parallelism), C1 call-site wiring. Opus review of Phase C/D: the C1 handoff key compared layout, spec, cap and dc but not the solver-basis gauge (`impurity_symmetry_rotation`'s `eigh` changes eigenvectors when `h` is shifted by mu*I), so a reused state could sit in a different orbital basis than the self-energy's `h`; it also sized the auto cap while the handed state was still held. C1 was skipped by decision and nothing called `handoff=`, so the C1 commits (5af9c510, 8e4a70b5, c77b3f72) were reverted. A future reuse must key on `rotation_full` and a digest of `h`, or hand over the DC's `SolverBasis`. Leftovers closed afterwards (2026-11-04): continuous-mode DC search pinned (no `sector_of`, finer bracket under `on`; both mutations fail), one settle retry per member on its own run (test), `redistributed_states` moved to `groundstate` (no private cross-driver import), legacy `selfenergy`/`susceptibility` scripts take `--mixed_valence` (default off), dedicated test that a rank-0 spectra-merge failure is raised on every rank, dense grand-canonical oracle for the chi ensemble (<N_imp> and the charge Curie coefficient; uniform weights fail it), `ensemble/total` member labels in the get_spectra file. Not built, with reasons: KMS check in the reports (the GF engine keeps no poles/residues, only mesh G), streamed merge for chi/spectra (member results are small arrays; Sigma already streams its Boltzmann normalisation), a spectra GC oracle (the weights path is shared with the Sigma and chi oracles), 'predicted GC n' advice (needs extra solves at the stuck shift), the bracket-tolerance trial-count record (the tolerance was not changed). Gate after all of this: -n 1/2/3 = 4044 / 4152 / 4160 passed on rank 0, every rank file clean; the earlier rc=1 at -n 2/3 was the C1 handoff tests failing on ranks >= 1. C4 and Phase D: no findings; `IMPMOD_MATVEC_PROF=0` still turns profiling on in the readers (`bool(os.environ.get)`) while `config.dump()` reports False (cosmetic). step 13. J9 (MV-doc fixes, `schema_version`/`required_features`): a doc commit before step 12. O6 oracle CI budget:
step 0.

Steps 1–4 are independent of both plans and can ship first. The first point where the plans touch the same code is
step 5 (MV 1).

## 7. Order without the other plans

1. P7 + S1 + oracle (step 1).
2. M1 + M5 (step 2).
3. S2, S3, C2', C4, M4, C7 detection (step 3).
4. P8, degenerate-split detection, P1a + spectra commutator warning (step 4).
5. P11, H3, M3, P4, P9's ΔE/τ warning, and the §4 UX summary.
6. A1 series incl. P10.
7. B1.
8. C7 fix, B2, B3, `Within` cap.
9. B4–B7; B8.
10. Hygiene.

## 8. Decisions

- **O1. `mixed_valence` default for RSPt/Σ/DC: `auto`** (user, 2026-10-30). The J3 rule applies: `auto` never raises.
- **O2. DC-only tolerance overrides: keep them** (user, 2026-10-30).
  - **Why they exist:** a rough DC estimate, for cases where the DC search would otherwise burn enormous compute just
    finding the lowest sector.
  - **When they can go:** once no reasonably simple calculation can spend unreasonable time on that. Today one can:
    FCC Ni's walk did not finish in 90 min, and the losing N2 sectors take 513–954 s. Revisit after S1, B1, B5 and B7
    are measured.
  - `DC_E_PT2_TOL`/`DC_DE2_MIN` default to unset, which means the GS values (`config.py:897-973`). So they are opt-in,
    and DC→Σ reuse (B8) applies whenever they are unset.
  - When the user loosens them, reuse only the DC's **sector choice**: start the Σ walk at the DC winner, still checking
    its ±1 neighbours. Do not reuse its states.
  - The three DC tolerance rules (P5) still collapse to one precedence: DC knob if set, else the spec.
- **O3. Thermal re-solve below the walk's quality: how it happens, and the rule (user: SYM, non-negotiable).**
  - **Mechanism.** The walk's `Lowest` solve converges *one* state of the winning sector on its own basis: SMO 3-bath
    247,974 determinants. The thermal `Within` solve rebuilds from the seed and must hold the whole window in one
    basis. On SMO that is a 4-fold S=3/2 level whose S_z members live on disjoint determinant sets. The auto cap is
    sized "for the ground state alone", so each member gets about a quarter of the walk's room:

    | run | thermal basis | per member | walk's single state |
    |---|---|---|---|
    | SMO 3-bath -n 4 | 452,240 | ~113k | 248k |
    | SMO 3-bath -n 1 | 509,546 | ~127k | 248k |
    | SMO 4-bath | cap 440,512 | ~110k | the full 440,512 |

  - **The energy deficit is acceptable; the symmetry breaking is not.** A less converged but symmetric thermal manifold
    is the correct outcome under SYM. The defect is S2: once the cap binds, the expansion grows and cuts members
    unevenly, and the quartet splits.
  - **Resolution:**
    - (1) S2: symmetric (orbit-closed) growth and truncation in the `Within` expand, so a binding cap costs every
      member the same accuracy;
    - (2) resolve the cap for the `Within` manifold (the winning level's multiplicity × the walk's per-state basis,
      bounded by memory) together with P10, so the symmetric solve is as accurate as memory allows;
    - (3) no re-solve and no "never above the walk" floor: seeding from the walk's single state would itself break
      the symmetry.

    A degenerate-level split beyond roundoff is reported as an error-level warning, because under SYM it can only
    mean a bug.
- **O4. χ and spectra read `mixed_valence` from the archive** (user, 2026-10-30).
  - When the archive has it, use it. When it is missing, use the χ or spectra default (off).
  - The archive records the *setting* (`auto`/`true`/`false`), not the resolved membership, so a replay re-decides
    membership on its own model.
  - Archives written before MV hold `None` or 0, which map to the reading driver's default as MV already specifies.
- **O5. Freeze only on explicit user request** (user, 2026-10-30), e.g. a shell declared `role = "core"`. No automatic
  detection.
  - Where the user freezes a shell, check ‖[H, N_core]‖ once and **warn** with its size if it is not zero. The seed-only
    pin then leaks, and the user should know.
  - MS F1 freezes a shell only if the user declares it core, never on its own.
- **O6. Parameters still to choose:** B1's margin; the J3 weight-error threshold; MV decision K (KMS); the CI
  runtime budget of the promoted oracle. (The J1/O5 commutator warning tolerance: `PARITY_TERM_TOL`, step 4.)

## 9. Cross-cutting review 2026-11 (ground state × mixed valence × multishell)

Four read-only Opus reviews (one per theme plus a cross-cutting lens) over `9f1b0609..HEAD`; findings confirmed
by probe or by reading, fixed in single-concern commits with a test that fails on the old code. Nothing pushed.

**Fixed** (review id: effect)

| id | defect | fix |
|---|---|---|
| (earlier) | seed truncation cut orbits over a part of the basis | `964b60b9` truncate over the whole basis |
| GS1/MV3 | rank-0-only blocks and merges hung the other ranks on an exception | `on_root` / `on_every_rank` (`174da019`, `18f288b9`) |
| MS1 | χ did not pass the bath-less shell's `seed_window` to the GS | `b8ba3844` |
| MS2 | cap connectivity probe seeded without the frozen window | `666dda3f` |
| MS3 | `n0` read input-basis `h0` at solver-basis indices | `ccdb4426` |
| GS9 | `DC_DE2_MIN=0` treated as unset | `10af5449` |
| X1 | `attainable` pinned window groups at the nominal filling | `dfd4e03f` |
| MV2 | a refused ensemble fed the cap ladder although the cap was not the cause | `2c7e1daa` (`EnsemblePlan.refused_by_cap`) |
| GS8 | split / incomplete / uncertified manifold reported `converged` | `858adaa5` |
| GS10 | seed-size truncation orbits built without the spin layout | `fa1b977a` |
| GS11 | `Orbits.notes` degradations silent | `381b68b8` (printed once, rank 0) |
| MV7 | `gs_info["truncation"]` was the dominant member's | `b3867376` (`worst_truncation`) |
| docs | X3, X4, X5, MV10, GS7, MS5 | doc/test hygiene commits |

**Refuted / not needed:** GS2, GS3, GS4, GS5 (optional hardening only), GS6, MV1.

**Open, for the owner's decision (not changed):**
H-MV12 (`auto` DC mode can raise a false `DoubleCountingUnreachable`), H-MV4 (`gs_energies` is the dominant
member's, `rhos` the union), H-MV6 (dominant member by walk energy, not realised energy), H-MV11 (should `on`
raise on a cap-bound thermal member; should `auto` keep the winner), H-MS4 (DC sector splits for a SeedWindow
are not shell-keyed), H-MS6 (tolerance mismatch refusing p–d couplings ≥ 1e-5), H-MS7 (rank-local rotate/refusal
decision), X7 (legacy `mv N>0` meaning), default-verbosity visibility of an uncertified winner, H-MV5
(streaming K-member merges). impmod_interface X6/X8 (header "Mixed valence |> None", `_parse_mv("-1")`) are
fixed in impmod_interface `fc456a1`, not open.

**Hang introduced and fixed in this pass.** GS8 (`858adaa5`) stored a numpy bool in `manifold_check`. Rank 0's
`save_gs_statistics` then raised `TypeError`, while the other ranks went on into the next collective, so the `-n 2` gate
stalled on `test_calc_gs_ensemble`. The machine then froze (the journal shows an i915 GPU hang, no OOM). The targeted
runs had missed it: they were run without `--with-mpi`, so the `[mpi]` variants were skipped. Fixes: `a5c78de9`
(`bool()`, the encoder writes `np.bool_`) and `67dd4a37` (the statistics write goes through `on_root`, gated on the
replicated `stats_path`). Every long MPI run since then has been under `timeout` + `systemd-run -p MemoryMax=… -p MemorySwapMax=0`.

**Final gate** (at `67dd4a37`, capped; after it only `6f1c1728`, a type annotation, and `23ebf47a`, one pure-function test run on its own): `-n 1` 4067 passed; `-n 2` 4176 / 4175 passed; `-n 3` 4184 / 4183 / 4183
passed; 9 xfailed on every rank file. `-n 3` took 46.5 min, against 58 before this pass. `make check` scoped to
`src` is no worse than the baseline (one mypy annotation added, `6f1c1728`).

**C2 bug injections** (`debug/review_2026-11/probes/c2_inject.py`; one edit at a time, restored from a copy):

| inj | edit | caught by |
|---|---|---|
| a | `sector_weights` without `z` | sector-plan GC weights (6), GF GC oracle (2) |
| b | `_member_coefficients` without `kept_share` | `test_member_coefficients_…`, partly-dropped-member oracle |
| c1 | admission `<` → `<=` on the cut | **escaped**; caught after `23ebf47a` (`test_a_sector_exactly_on_the_energy_cut_is_not_admitted`) |
| c2 | `e_min` not the winner's | needs/one-member tests (6) |
| d | `_active_shift` P = I | multishell DC (6) |
| e | `DEEP_SEED_WIDTH = 0` | `test_default_width` only (a constant pin) |
| f | `_group_rows` sorts on the first word only | `test_group_rows_is_the_partition_of_equal_rows` (3) |
| g | `retain` keeps half an orbit | `test_truncate_keeps_orbits_whole_…` |
| h | width-0 padding dropped | `test_width_zero_block_…` (at -n 2) |
| i | intra/inter variance swapped | between-sector variance warning test |
| j | `_holds_the_global_ground` always False | later-member-realises-lowest test |

**Profiles** (`debug/review_2026-11/logs/prof_*`; `-n 4` unless stated; reference step 7′):

| run | total | walk | thermal | E0 / states / dets | vs reference |
|---|---|---|---|---|---|
| NiO 10-bath off | 343 s | 182 s | 146 s | −53.3299644, 3, 710,493 | C4 352.7 s, 7′ 419 s; same E0 |
| NiO 10-bath off, -n 1 | 405 s | 216 s | 170 s | identical | 7′ 453 s |
| NiO auto | 4231 s | 188 s | 190 s | K=2: N=8 (0.9999), N=9 (1e-4) | GS ~380 s; ~3850 s are GFs of both members |
| NiO auto, -n 1 | timeout 5400 s | | | inside the first member's GF | |
| SMO 4-bath off | 2201 s | 1612 s | 564 s | −11.5651958, quartet 4/4, 489,322 | 7′ 2147 s (+2.5 %), E0 2.5 µRy lower |
| SMO 4-bath auto | 3562 s | 2012 s | 526 s | K=1 (N=3), 484,890, E0 0.5 µRy above off | walk +25 %: one extra N=1 solve (387 s); cap 497,260 vs 506,172, see below |
| CrI3 off | 2534 s | 1685 s | 825 s | −7.62335438, 2 states, 517,698 | 7′ 2472 s (+2.5 %), identical E |

No `off` regression beyond noise. Two new performance items for decision:
- **PERF-MV-a.** A member with weight 1e-4 (NiO N=9) gets a full Green's function, the same as the dominant one.
- **PERF-MV-b.** Under `auto`, a sector admitted provisionally on its optimistic (residual-subtracted) energy makes the plan solve
  its edge neighbour. On SMO that cost 387 s, and the final ensemble was the winner alone. The doc's prediction of K = 2 on
  SMO 4-bath did not hold (K = 1). SMO auto vs off is not a one-member identity break. The auto cap is sized at start from measured available memory and
  the deterministic connectivity probe, with no mode input (`resolve_cap_policy`). The two runs read about 2.9 GiB/rank but
  sized 497,260 against 506,172 dets. Both then tightened at the same 1.7 GiB RSS, and the first sector solved is identical.
  The 0.5 µRy is that measured-memory noise; it is also why off drifted from 462k to 489k dets between step 7′ and now.
- The 35→58 min `-n 3` growth did not reproduce (46.5 min in this capped run) and is still unexplained; P3 was not run.
- impmod_interface (`fc456a1`, local) suite at this HEAD: 231 passed. No archive (409 `mv` attributes) or solver line in
  `impmod_tests` has a negative `mv`, so rejecting `mv -1` (X8) changes no real input.

**Follow-up series 1 (2026-11-04), no-decision fixes:**

| id | defect | fix |
|---|---|---|
| H-MS7 | symmetry rotation and rotate decision taken from a rank-local `eigh` | `ab92eaec` broadcast from rank 0 |
| H-MV11 | a member whose own thermal solve was cap-bound or unconverged was merged; `on` did not raise | `041ced88` checked before `consume`; `on` raises, `auto` keeps the winner (owner decision B3) |
| H-GS13 | the connectivity probe reset the shared operator's restrictions on rank 0 only | `dc9952f4` probe on a copy |
| H-GS3 | `solve_sector` / `find_ground_state_sector` lacked `assert_uniform` | `bf954cb4` |
| H-MV9, H-MS8 | `mixed_valence_tristate` undocumented; discretized `n0` docstring | `c8cebf23` |

H-MV13 (sorted union `es`) moves to the `gs_energies` contract change (H-MV4/MV6). Each fix has a test that fails on
the old code (injected). Gate at `bf954cb4`: `-n 1` 4076 passed; `-n 2` 4187 / 4186; `-n 3` 4195 / 4194 / 4194;
9 xfailed on every rank file; `-n 3` took 47.6 min. Logs: `debug/review_2026-11/series1/logs`.

**Follow-up series 2 (2026-11-04), owner decisions:**

| id | decision | fix |
|---|---|---|
| H-MV11 (narrowed) | refuse on a bound basis or a *current* residual above tol; a stale residual or widened manifold warns | `e8027095`; a member held by the measured-memory guard is refused but not as a cap verdict, `841e38ef` |
| H-MV12 | `auto`: a bracket that collapses on a member flip is searched continuously, not raised | `64cd1a22`; the fallback records its own width, evaluations and failure, `dd3c06f3` |
| H-MS6 | message only: name the direct input-basis element or the bath that joins the shells | `55746235`, `09086bf0` (group's shells only, roundoff floor) |
| X7 | a legacy non-zero `mv` means `auto` | `5dcaccd7`; the input-file reader did the same, `cae06086` |
| H-MV4/MV6/MV13 | the dominant member is the realised lowest; `gs_energies` is the ascending labelled union | `3cd81ba1`; orders are rank 0's (rank-local `es`), `93a6a95c` |
| PERF-MV-a | members below weight 1e-3 get no Green's function, reported at every verbosity | `3092418e`; the global-ground member is exempt, `5a07ef8a` |
| H-MS4 | DC trial sectors hold a bath-less window at the walk's filling | `195ebac3`, `be82a810` |
| B10 | an uncertified ground-state sector is announced once at every verbosity | `374e5d62` |
| PERF-MV-b | accepted, no change | — |
| H-MV5 | deferred (modest cost) | — |

An Opus review of the series diff found the follow-up items above (rank-local ordering, the cut's exemption, the reader,
the fallback record, the memory-guard verdict, the window filling, the operator-copy comments). Still open from that
review: when a member at index ≥ 2 is refused, a Green's function already computed for an earlier member is wasted (7b); the
provisional neighbour note (7c) is cosmetic. Docs: user guide ensemble section `e7ac312c`.
Gate at `09086bf0`: `-n 1` 4110 passed; `-n 2` 4222 / 4221; `-n 3` 4230 / 4229 / 4229; 9 xfailed on every rank file;
`-n 3` took 46.9 min (the 58 min outlier again did not reproduce). The one failure on every rank file was
`test_gs_report_merge`, still asserting the pre-H-MV13 union order; fixed in `151b9552` (test only, re-run at -n 1/3).
Logs: `debug/review_2026-11/series2/logs`.
