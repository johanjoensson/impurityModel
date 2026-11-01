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
- **MV's own inconsistencies to fix in that doc:**
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
| 3 **DONE 2026-10-31, profiles outstanding** (commits `2b1d3962` C4, `37d7a2ad` M4, `776e187d` C2', `5c5c86a3`/`289e1680` S3, `69c71c1a` C7 detection, `5697f8eb`..`90d7120e` S2 orbits + review/gate fixes; gate -n 1/2/3 green after `62d450fd`, targeted re-runs after `90d7120e`). Orbits (`ed/orbits.py`, `GS_ORBIT_CLOSURE`): full spin arrangements of a pair pattern at \|2Sz\|<=M plus the closure of the detected signed permutations, ranked by summed score; cubic d7 at cap 160: quartet equal to 2e-13 vs 1e-7 split without. Not done: orbit-aware `slaterWeightMin` (effect ~1e-12, skipped), GF-side closure and the Σ↑=Σ↓ acceptance (later step), SMO 4-bath / CrI3 / NiO -n1 vs -n4 profiles (~33/40 min each, not run). Open: retain() generates every arrangement of each kept orbit (cost at production size unmeasured); budget charges an orbit its full size, so a budget smaller than every candidate orbit stalls the expansion (reported cap-bound). | **S2** (orbit-closed growth and truncation in the `Within` expand; SYM); **S3** (never drop an in-window class; record the union cut; `converged: False`); **C2'** (depth on complement; certified=False); **C4** (replace record, `only=None`); **M4** (`memory_bound` only on an actual lowering, accumulated); **C7 detection** (per-label window counts, trim warning) | yes, -n 3 (orbit closure adds collectives) | S2 acceptance (degenerate members equal to roundoff under a binding cap; SMO 4-bath quartet exact); S3: CrI3 keeps 4 states (or reports the cut); `c2_small` unit test; C4 call-site test; NiO -n 1 vs -n 4 same flag; CrI3 flags its window |
| 4 **DONE 2026-11-01** (commits `ddaf34ce` target, `addb2806` P8 DIIS, `2b5d3834` P8 ±1 fallback, `1392f630` split detection, `a179bfc3` P1a, `f7fc6969` [H, N_core] warning; full gate -n 1/2/3 green on every rank file: 3786/3894/3902 passed; `make check`: ruff findings identical to `b663c7fd`, mypy clean, black flags only the untouched `gs_oracle.py`). P8: DIIS rung after the mixing ladder: complex 39/40, 99/100, real 60/60, 100/100 (convergence target met). **Off-by->1 target missed**: 11/100 real, 8/100 complex (target ≤ 10, ≤ 4). The new cases are newly converged models; on those off-by ones the DIIS point is the lowest of 60 random-start HF fixed points (`step4/hf_offby.log`), but on other rescued models DIIS often lands on a *higher* aufbau fixed point (Opus review: 11/56, 9/26, 6/8, ΔE up to 2.29). End to end 0/200 wrong E0 before and after, and 0 on every ladder-failed model vs the old scan path. Review (Opus) open item: with ≥ 2 groups and non-convex E(N) (attractive U) the ±1 descent stops in a local minimum the old scan's diagonal move escaped (`_two_group_siam(e_imp=1, u=-4, v=0.05)`: 2 eV high); the accepted-seed path already had this, 2b5d3834 extends it to rejected seeds; 0/120 on Kanamori U>0 two-group models. Level shift rescued nothing, not shipped. Rejected seed: ±1 walk from N0, notice at V_RESULT. Split detection: SU(2) only (`manifold_check.multiplet_split`, ERROR line); CrI3/SOC not covered. J1/O5: DC freezes nothing; the commutator warning uses `PARITY_TERM_TOL` (relative), which closes that O6 item. Open: `get_spectra._resolve_shell_roles` still infers the core from bath counts for models without `valence_l`/`core_l` (legacy fallback, raises on ambiguity; out of step 4). **P8 target set 2026-11-01, before the change** (`hf_conv.py` through `hartree_fock_occupation`, i.e. with both starts and the mixing ladder; `debug/gs_review_2026-10/step4/hf_conv_baseline.log`). Baseline: real 58/60 and 92/100, complex 13/40 and 43/100; converged seed off by >1 sector 10/100 real, 4/100 complex. **Target:** complex ≥ 90% on both seeds (≥ 36/40, ≥ 90/100); real does not drop; off-by->1 does not rise (≤ 10/100 real, ≤ 4/100 complex); every model the ladder converges today is bit-identical (the new solver only runs after the ladder fails). | **P8** (level shift/DIIS, ±1 fallback, visible rejection); degenerate-split **detection** (rank 0 + bcast); **J1/P1a** (delete the DC bath-less rule, rewrite `test_fixed_dc`; spectra `[H, N_core]` warning on user-frozen shells) | yes | complex/SOC HF convergence target set before the change; detection fires on an injected orbit-breaking cut |
| 5 | keyword-only `mixed_valence`; **MV 1** (non-zero raises); **MV 2 + P11** (`SectorSearch.solved`, ±1 neighbours, one-sided HF check) | MV 1 changes seeds: -n 3 | `mixed_valence=1` fixtures re-baselined, noted in the commit |
| 6 | **H3** public degeneracy helper; **M3** (and the `rank=` callers); **P4** | no | χ Curie term unchanged on fixtures |
| 7 | **MS T4.1–T4.5** (impurityModel side; impmod_interface G2 and rspt2spectra G3 in their repos); `dc_criteria` raises on multishell until T4.7; MS 2p@−60 Ry fixture measurements (J1); Tr ρ_p = 6 test and T5.1 | no | single-shell archives bit-identical |
| 8 | **A1** series (a) plumbing, (b) P3 χ, P2 spectra+RIXS, P5, P6 one per commit, (c) **P10** caller-resolved cap (impmod_interface + archive attribute) | (c) yes | (a) bit-identical; (b) each with its A/B threshold written down first |
| 9 | **B1** (trajectory bound, beyond `cut + margin`); **MV 3** warning + **J8** summary and stderr tiers | yes | winners unchanged on all six profiled workloads; WARNING lines on SMO 4-bath ≤ the stated budget, each event once per channel |
| 10 | **C7 fix** (per-label window), **B2** (per-class margins), **B3**, the `Within` cap (O3) | yes, -n 3 | NiO thermal basis ~135k with Σ unchanged within its tolerance; CrI3 keeps the quartet |
| 11 | **MS T4.7** (active projector, J5 store) | yes | single-shell DC numerics bit-identical (projector = I) |
| 12 | **MV 4** (report split, baseline after T4.5), **MV 5** (seam, Boltzmann consolidation, spec switch), **MV 6–8** (GS report, Σ consumer, DC continuous). Then **MV 9–13** (χ, spectra, RIXS, KMS, docs) | yes; consumers raise collectively (`_raise_together`, `selfenergy.py:62`) | MV's own verification list |
| 13 | **B4** (after a production-budget re-measure), **B5** (geometric-tail rule), **B6**, **B7** (-n 4); **B8** (states reused with default DC tolerances, sector choice only when loosened) | yes | each gain measured on the change |
| 14 | hygiene | no | — |

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
