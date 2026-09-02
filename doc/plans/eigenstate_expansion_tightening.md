# Tightening eigenstate over-computation in the ground-state search

**Status: closed (2026-09-01). Mostly a documented failure, retained with its verdicts.**

The premise was that the ground-state search computes far more eigenstates, and far more
Krylov vectors, than it retains, and that five nested growth loops were the cause. Measured
on the real NiO workloads, **four of the five loops never fire in production and the fifth
was mis-sized by a cost model that counted the wrong thing.** Two genuine solver bugs turned
up while measuring, and those are what the campaign is actually worth.

Read this before proposing eigensolver work of this shape again — the numbers below are the
reason not to.

## Verdicts

| # | Lever | Verdict |
|---|---|---|
| C9 | Derive the degeneracy tolerance from the achieved residual | **SHIPPED** `3c085a8` |
| C5 | Early exit in the TRLM restart continuation | **SHIPPED** `de3ac3a` — measured -7.6% block matvecs, -11% eigensolver time |
| C5 | Same for IRLM | **REFUTED** — fired zero times, +13% |
| C1 | Drop the thermal cut on `calc_energy`'s min-only path | **SHIPPED** `b177f6a` / `9e90f37`, but **inert**: zero measured effect |
| C4b | A relative floor on TRLM's convergence test | **REFUTED** — the premise was arithmetically wrong |
| C6 | Warm-start the doubling rungs | **REFUTED** — there are no doubling rungs |
| C8 | Shrink the Krylov subspace `m` | **REFUTED** — would move work into the loop that dominates |
| — | Block deflation treated as a closed Krylov space (TRLM) | **BUG, FIXED** `8a9f337` |
| — | IRLM reporting Ritz pairs it never converged | **BUG, FIXED** `3715862` |
| — | The sparse-kernel bench's cross-kernel E0 assertion | **BUG, FIXED** `afdca32` (test bug; both kernels correct) |
| C2 | Restore the rank-invariance guard rail | **SHIPPED** `d96f5cf`..`af5059b` + the re-enable — two new tests, one strict xfail |
| C2 | Reproduce the CI-only SIGSEGV locally | **NEGATIVE** — 550 MPICH runs, clean ASan; and CI then crashed 4x with the file skipped, once with no MPI at all |
| — | `_graph_comm_cache` context-id exhaustion | **REFUTED** — 10 live entries after a whole `-n 3` suite |
| — | Cold `p = 1` cannot certify a degenerate manifold | **REFUTED by measurement** — see `doc/lanczos_invariants.md` |

C2, C3, C7, C11 were not reached or were narrowed to nothing by the measurements; the
successor lever at the end is unstarted.

## The measurements

All on NiO 7-bath and 10-bath through `test/support/real_workload.py`, `n_iw=1 n_w=0`,
`OPENBLAS_NUM_THREADS=1`.

**The growth loops never iterate.** Instrumenting both predicates of `get_eigenvectors`'
doubling loop:

| workload | Krylov solves | `need_more` true | `exhausted` true |
|---|---|---|---|
| NiO 7-bath | 14 | 0 | 0 |
| NiO 10-bath | 18 | 0 | 0 |

The first solve always lands a state outside the thermal cut, so loop B runs exactly once.
Loop A (`solve_ground_state`'s `wanted += 10`) only iterates when `len(es) >= wanted`, and
the cut retains 3 states against a request of 10, so it terminates immediately too. C1 and
C6 both target machinery that is already a no-op.

**In every solve `len(e_ref) == num_wanted` exactly**, at a padded request of 14, 18 or 22
states — of which the thermal cut retains **3**. So the pad is real waste; the question is
where it costs.

**Not in the subspace size.** Counting block-matvecs inside `_trlm_core`, 14 solves:

| | blocks | columns |
|---|---|---|
| initial sweep | 538 (15%) | 2625 (15%) |
| restart continuation | 2975 (**85%**) | 14597 (**85%**) |

Identical split either way; deflation is rare on this workload, so blocks and columns tell
the same story. `rr_matvecs = 0` — the Rayleigh-Ritz rebuild arm never fires in production.

**This refutes the campaign's central premise.** The plan's Finding 1 asserted the initial
sweep dominates ("at `p >= 21` the flat `+20` *is* `m` — ~90% of it"), and C4, C6 and C8 all
follow from it. It is 15%. Shrinking `m` shaves that 15% and lengthens the loop that is 85%,
because `d = m - k_blocks` is the polynomial degree added per restart and a shallower
restart needs more restarts. The recommended `m = k_blocks + max(4, k_blocks)` would have
taken the outlier solve below from `d = 30` to `d = 4`.

> **The reusable fact: a vector count is not a cost model.** Finding 1 counted vectors in a
> sweep and never counted the restart continuation, which is where the work is.

**Where the pad actually costs.** One solve (`p=6, num_wanted=20, m=34`) used **38 restarts
and 1126 continuation blocks — 38% of all continuation work in the campaign**; every other
solve finished in 3-10. Its residual trajectory is a clean linear decay at **0.67 per
restart**, against 0.001-0.09 for every other solve, with a period-2 oscillation
(5.4e-5, 6.1e-5, 3.1e-5, 3.9e-5, ...) — the signature of a near-degenerate pair straddling
the wanted/unwanted boundary, not of a shallow restart.

`done = max(res[wanted]) < tol` is taken over **all** `num_wanted = 20` states, so the
hardest of 20 sets the cost, and the cut then keeps 3. The `+10` pad costs through the
*stopping criterion*, not through `m`.

**C4b's premise was arithmetically wrong.** The plan claimed `done` could be unsatisfiable
because "`tol/||T|| = 1e-14 < EPS`". `EPS = 2.2e-16`, so 1e-14 is two orders *above* it.
Measured: `tol = 1e-8`, `||T|| ~ 48.5`, six orders of margin, and the most restarts any solve
used was 38 of 100. The floor would only bind for `||H|| > tol/EPS ~ 4500 eV`. (IRLM already
has exactly this floor — EA16 eq. (15)'s `u*||T_k||`. If it is ever wanted in TRLM, reuse
`ea16.acceptance_tol`.)

## The two solver bugs

Both were found by probing the **true** residual `max_i ||H v_i - e_i v_i||` at the
`get_eigenvectors` call site, not by reading code, and the full pytest suite caught neither.
Every tolerance above the kernels — including `cipsi_solver._degeneracy_tol`, whose whole job
is to be no tighter than the achieved accuracy — floors on the tolerance the solver was
*asked* for. Nothing checked what it achieved.

See `doc/lanczos_invariants.md` ("Deflation is not closure", "Never report a Ritz pair that
failed the acceptance test") for the invariants and the numbers.

## Restoring the guard rail (C2)

The campaign's own safety net was off: `test/mpi_infra/test_rank_independence.py`, which holds
the only end-to-end rank-invariance guards, was skipped behind an intermittent CI-only SIGSEGV.
Two solver bugs went through the rest of the suite while it was.

**Two tests that should have existed already.**

- `test/lanczos/test_manifold_completeness_krylov.py` — the degenerate-manifold guarantee, on
  the Krylov branch. Every previous test of it ran *dense*, which does not execute the same
  code: the dense path applies a hard `es - min(es) <= e_max` mask with no manifold-absorbing
  step, so `_energy_cut_indices` is reachable only through Krylov. Both the expansion and the
  solve now run through TRLM; expanding densely would only have shown that Krylov *preserves* a
  manifold handed to it.
- `test/lanczos/test_subspace_sizing.py` — the first test of the block-count arithmetic every
  production solve runs at, enabled by lifting it out of `get_eigenvectors` into
  `_size_subspace` (verified bit-identical against both original spellings over a 4,500-point
  grid). Seven green properties plus a **strict xfail**: the final clamp is not a no-op. When
  the basis-size bound wins it trims `num_wanted` — the *certified* output of the manifold
  search — and it does so on ~30% of that grid. The remedy is to give up block width or fall
  through to the dense branch, never the certified count.

**The SIGSEGV did not reproduce.** 500 whole-file iterations alternating `-n 2`/`-n 3` plus 50
interleaved full-suite runs under MPICH 4.2.2 (the CI MPI family, which the old skip note's
"not locally reproducible under Open MPI" had never tested): **550 runs over ~9.5 hours, all
rc=0**. Against the CI rate of 1-2 of 8 legs — once 5 of 8 — a few dozen should have sufficed,
so what this establishes is that the discriminating variable is not on this machine. Then the file
*and* the whole suite under AddressSanitizer, MPICH-linked, at `-n 2` and `-n 3`: zero reports.
That is the stronger negative — ASan traps the bad access when it happens, so a run that
corrupts the heap without crashing is still caught. The file is re-enabled, with the evidence
and the remaining untested variables recorded in its module comment.

**Re-enabled, but isolated.** A failed reproduction is not a fix, and the failure mode is a
whole-process SIGSEGV: a recurrence would not fail five tests, it would kill the step and take
every other result on that leg with it — which is the damage the skip was really avoiding, at a
rate of 1-2 of 8 legs normally and 5 of 8 once. So the file is deselected from the three
standing MPI legs and runs in three steps of its own at the end of the job, after the coverage
upload. It runs exactly once either way; a recurrence now fails one step whose predecessors
have already reported.

**And CI answered the question the reproduction could not.** Between 2026-08-28 and 08-29, with the
file still skipped, master crashed with exit 139 **four times across three runs** -- so the crash is
not confined to these tests, and skipping them was never going to stop it. Two landed at 41%, just
after `inputformat/test_f_shell_crystal_field.py`, on the `-n 3` step (runs 33192921138 and
33256083896). Two landed at 97%, just after `symmetry/test_symmetry_observables.py` -- the *last*
file collected -- in run 33215746448: one on the intel `-n 3` step, and one on the step named **"Run
serial tests"**, whose command is `pytest --cache-clear $PYTEST_COV_ARGS`, no `mpiexec`, no ranks
(`3202 Segmentation fault (core dumped) pytest --cache-clear`).

Nothing in the build explains it: gcc-12 three times and intel once, `-std=c++17` twice and
`-std=c++20` twice, `parallel` on one leg and `coverage` on one. What organizes these crashes is
*where in the run* they happen, not how the extension was compiled.

**And it recurred three times immediately, with the guard rail deselected.** The first CI run after
the re-enable -- run 33665523145, PR #2 -- crashed on **3 of 8 legs**, every one on a step where
this file is deselected (those steps report 23 deselected against serial's 18: the five tests).
gcc-12/`-std=c++20`/coverage at `-n 2` and clang-15/`-std=c++17` at `-n 3` both died in
`inputformat/test_f_shell_crystal_field.py`; clang-15/`-std=c++20` at `-n 3` died in
`restrictions/test_excitation_budget.py`, a new site.

**Seven crashes now, and they resolve into a pattern.** Five are mid-suite, and each landed on its
file's one heavyweight full-stack test rather than anywhere within it -- four in
`test_f_shell_crystal_field.py`, whose 9th of 10 tests
(`test_an_f_shell_crystal_field_model_solves`) is the only one there that runs a solver at all, the
other nine being input-format validation; and one on `test_excitation_budget.py`'s 12th of 19,
`test_calc_selfenergy_excitation_budget_oracle`, which runs `calc_selfenergy` twice through the
full driver and GF stack. Dot counts are lower bounds (a partial line is lost when the process
dies), so the windows are tests 8-10 and 12-13. The other two are the 97% finalize-time pair.
Neither rank count nor build discriminates.

A targeted local loop -- 120 runs of `test_f_shell_crystal_field.py` alone, alternating `-n 2` and
`-n 3` -- came back all rc=0, consistent with every other local attempt. Whatever discriminates is
not on this machine, so the probe had to go to CI.

**That probe is now live.** The `test-asan` job in `.github/workflows/tests.yml` was commented out;
it is re-enabled and pointed at the two named files instead of the whole suite, which is what makes
looping affordable -- and looping is the point when the crash needs 1-3 of 8 legs to fire.
Iterations are split by cost and crash share (10 per rank count on the f-shell file at ~10 s a run,
2 on the excitation-budget file at ~45 s), plus one serial full-suite pass for the other hypothesis
and the finalize-time pair. `workflow_dispatch` takes an iteration count, so another sample costs a
button rather than a push, and the job is `continue-on-error` with any ASan report copied into
`$GITHUB_STEP_SUMMARY` -- non-blocking, because the matrix legs already gate, but loud.

Why it had never worked: `g++ -print-file-name=libasan.so` returns an ASCII **linker script**, not
a preloadable object, so the old recipe's `LD_PRELOAD` silently did nothing and any clean result
was meaningless. It now resolves the SONAME, verifies ELF magic, and checks a built extension
carries `__asan_` symbols -- the self-test alone would not, since it compiles its own binary with
`-fsanitize=address` and passes whether or not `CXXFLAGS` reached the build.

The serial crash is the one that reframes the hunt, but read it precisely. It rules out **multi-rank
MPI** -- no ranks, no message passing, no communicator lifetimes -- not MPI itself: 72 modules
import mpi4py at module scope, so a bare `pytest` still runs `MPI_Init` on import and `MPI_Finalize`
at exit. Both 97% crashes are therefore consistent with a single finalize-time fault, and neither
crashing step printed pytest's `N passed` summary while every earlier step on the same leg did.
So the 550 whole-file iterations looped the wrong unit; only the 50 full-suite runs were on the
right one, and the ASan work all ran under `mpiexec`, never on a serial exit path. The cheapest
remaining probe is correspondingly cheap: the whole suite under ASan, **single process**, watching
the exit path -- no MPICH build, no rank sweep.

**Two theories killed on the way.** `_graph_comm_cache` is keyed on `id(comm)` and never
evicts, and the GF layer clones a communicator per unit — but it holds 10 live entries after an
entire `-n 3` suite across 18387 lookups, nowhere near MPICH's context-id space. And the
block-width bound `dim(K ∩ E_λ) ≤ p`, which says a cold `p = 1` solve cannot resolve a 3-fold
manifold and so could certify a third of one, does not bite in practice: cold and warm both
return all 3 states, capped *and* uncapped. Recorded in `doc/lanczos_invariants.md` rather than
acted on.

## Successor lever — unstarted, unmeasured

**Converge the manifold, not the pad.** Require `tol` of the states inside the thermal cut,
and of the boundary state only enough accuracy to *place* it outside. `theta[0]` is available
at every restart, so the in-cut set is known there, and the completeness guarantee is
untouched: it still needs a computed state provably outside the cut, not degenerate with the
last kept one.

- [ ] Establish whether the in-cut set is stable across restarts, or whether the criterion
      could ratchet (a state leaving the set would loosen the requirement it had already met).
- [ ] Measure on the 38-restart outlier specifically — that solve is the whole hypothesis.
- [ ] Gate on the **true** residual of the retained manifold, measured at the call site.
      "Achieved residual vs `tol`" is not sufficient: `tol` is itself looser than the grouping
      tolerance on three call paths (see `_degeneracy_tol`).
