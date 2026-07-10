# Scale-invariant deflation in `_cholesky_or_deflate`

**Status (2026-07-10): DONE.** Landed in three commits, each green on the full gate
(839 serial / 1004 at `-n 2` / 1004 at `-n 3`):

1. `a6ddb3d` — the `_trlm_core` width blocker (retained Ritz block, not the trailing residual).
2. `ec39d6f` — the partial-reort estimator's `omega_{i+1,i}` seed, scaled by `||H||` not `||beta_0||`.
3. this one — the deflation floor itself: relative rank test, `BREAKDOWN_TOL` relative to `||H||`.

Payoff: TRLM warm-started from `CIPSISolver.expand`'s `psi_refs` converges to `||r|| = 3.0e-14`.
Before, it returned its input unimproved at `2.2e-9` — whatever `tol` asked for — so
`cipsi_solver._eigen_tol` was a no-op on the production path. It bites now.

The three defects were independent, each sufficient to break a warm start on its own, and none
was covered by a test. Diagnosing them required three passes: each fix exposed the next.

## The defect

`_cholesky_or_deflate` (`src/cython/BlockLanczosArray.pyx`) answered two different questions with
one test:

```python
keep = evals > DEFLATE_EVAL_TOL * max(evals[-1], 1.0)     # DEFLATE_EVAL_TOL = EPS**(2/3) ~ 3.67e-11
```

* **Rank deficiency** — are some column *directions* of the residual block linearly dependent?
  That is a statement about `lambda_min / lambda_max` and must be judged *relative* to the block's
  own largest singular value.
* **Breakdown** — is the block numerically *zero*, i.e. is the block-Krylov space closed
  (invariant subspace)? That is an *absolute* statement — and therefore needs a *reference*. A
  Lanczos residual block is zero when it is negligible against `||H||`, not against `1`.

The `max(..., 1.0)` clamp fused them. Any block whose largest singular value fell below
`DEFLATE_TOL = EPS**(1/3) ~ 6.06e-6` was declared rank 0 **regardless of its conditioning**. A
small, perfectly well-conditioned block was treated as an invariant subspace. Consequence:
**every warm-started solver was silently handed back its own input.**

`BREAKDOWN_TOL = 1e-12` was declared at `BlockLanczosArray.pyx:86`, documented in the module
docstring, and **referenced nowhere**. Invariant-subspace detection rode entirely on the clamp.

The clamp also made breakdown detection *scale dependent* in the other direction: on an operator
with `||H|| = 1e-6`, a perfectly healthy **cold** start deflates to rank 0 on the first step
(`test_a_cold_start_never_breaks_down_early[1e-06]`).

## What landed

```python
def _cholesky_or_deflate(M, p_in, double scale=1.0):
    breakdown_lam = (BREAKDOWN_TOL * scale) ** 2
    ...
    if lam_max <= breakdown_lam:                  # absolute, against the caller's reference
        return None, None, 0
    keep = evals > DEFLATE_EVAL_TOL * lam_max     # relative, scale invariant
```

(and the mirror of both in the Cholesky fast path, on `d2 = diag(L)**2`). The conditioning
guarantee is untouched — a retained block still has `cond <~ EPS**(-1/3)`, inside the CholeskyQR2
recovery regime — because the relative test is exactly the one that bounds
`lambda_min / lambda_max`. Only the "small ⇒ rank 0" behaviour is removed.

`scale` defaults to `1.0`, which is right for every caller that normalizes O(1) columns:
`block_normalize`, the CholeskyQR2 second pass, and `block_bicgstab` (which pre-normalizes its
Gram). The two Lanczos sweeps pass the operator scale
`max(h_norm_est, t_norm_max, ||alpha_i||_2)` — the divergence guard's own running estimate,
widened by the current step's `alpha` because the guard has not run yet. On the first step of a
cold start `h_norm_est` is still 0 and `alpha_0` carries the scale; an exactly-zero block gives
`lam_max = 0 <= 0` and still breaks down.

The same reasoning fixed the two absolute `1e-5` invariant-subspace tests in `_trlm_core`, which
capped TRLM's residual at `1e-5` for any O(1)-norm operator whatever `tol` asked. They are now
`||beta|| <= max(tol, BREAKDOWN_TOL * ||T||)` — stop either because the Ritz residuals are already
under `tol`, or because the block is numerically zero against the operator.

## Why it took three passes

The reverted patch of 2026-07-09 was correct in the small and wrong in what it implied. Applying
it exposed, in order:

1. **`_trlm_core` width bookkeeping** (`a6ddb3d`). The note originally blamed the trailing residual
   block; that was wrong. The sweeps exclude it correctly on every exit. The desync was
   `block_normalize(Q_ret)` deflating the *retained Ritz block* while `cur_widths` recorded
   `nkeep`, and it needed no deflation change to reach. Fixing it exposed that the thick-restart
   coefficient shortcuts need `Q^H Q = I`, not merely full rank.

2. **The partial-reort estimator** (`ec39d6f`). With deflation relative, NiO's warm start no longer
   crashed — it *diverged*, at `it=37`, `|beta| = 1.17e5` against a spectral scale of `1.11e2`. I
   wrote in this file that `BREAKDOWN_TOL` was on the wrong scale and that a residual of `2.2e-9`
   against `||H|| = 1.1e2` "is a near-invariant subspace by any sensible measure". **That was
   wrong too.** `reort=FULL` sails through the same start block (`||Q^H Q - I|| = 8.9e-16`), so the
   recurrence is fine and the residual is a legitimate direction. What failed was PARTIAL's
   estimator: `omega_{i+1,i}` was seeded with `eps * beta_i^-H @ betas[0]`, using `beta_0` as a
   stand-in for `||A||`. True for a cold start; warm-started, `beta_0` is the eigenpair residual and
   the expression cancels to `eps` at `i = 0`, exactly where the true overlap
   `eps*||A||/||beta_0||` is largest. The trigger never fired and PARTIAL silently did *no*
   reorthogonalization.

3. **The deflation floor**, at last, with a `scale` argument rather than a bare constant.

Lesson worth keeping: a divergence in `PARTIAL` that `FULL` does not reproduce is an estimator
bug, not a breakdown-threshold bug. Check `FULL` first.

## Verification

`test_deflation_scale_invariance.py` (20 tests; 8 fail on the old floor), plus
`test_reort_oracle.py::test_partial_reort_survives_a_warm_start_from_converged_eigenvectors` and
`test_trlm_restart_widths.py`.

**Warm-started TRLM on the NiO ground state** (`_nio_workload`, `dense_cutoff=50`,
`_eigen_tol(1e-12) = 1e-12`), in-basis residual of the returned eigenvectors:

| | state 0 | state 1 | state 2 |
|---|---|---|---|
| before (old floor: returns its input) | `2.180e-09` | — | — |
| deflation only (diverges at `it=37`) | `7.860e+03` | — | — |
| deflation + estimator + width fix | `2.991e-14` | `6.839e-11` | `3.317e-13` |

**Cold starts are unaffected.** Identical reort trigger rates (`38/120`, `8/60`, `9/60` blocks
acted) and identical `||Q^H Q - I||` across three cold-start spectra.

**`reort=NONE` serial vs MPI**, NiO Green's function, `delta=0.01`, width 1 — re-measured after the
change (the old note recorded `m = 114/98/138` and `G` diffs `5.257e-09` / `1.611e-07`):

```
reort=NONE      1 rank: m= 98   2 ranks: m=138, dG=8.960e-09   3 ranks: m= 98, dG=1.510e-07
reort=PARTIAL   1 rank: m=106   2 ranks: m=106, dG=8.960e-09   3 ranks: m=130, dG=1.513e-07
```

The `G` differences are **identical under NONE and PARTIAL**, and track the excited basis (1216
determinants at 1–2 ranks, 1215 at 3), not deflation and not reorthogonalization. `E0` agrees to
`5.7e-14` across rank counts. The old docstring claim that the absolute floor "keeps the
`reort=NONE` recurrence on the *same* convergent trajectory serially and under MPI" was false
before this change and is false after it; what the *relative* half of the test buys is protection
against divergence (the `O(kappa)` amplification of `Allreduce` rank-order rounding), and that is
retained. The `BETA_BLOWUP_FACTOR = 1e3` guard never fired.

## Left behind

* `block_bicgstab`'s local Gram normalization (`f4d2aea`) is now redundant for the *rank* test —
  that test is scale invariant on its own. It still earns its place: it is what makes the default
  `scale = 1.0` the correct breakdown reference for a residual block. Its explicit
  `||R0|| <= atol * ||Y||` early exit is the tolerance contract and must stay regardless.
* **`reort=NONE` corrupts the sweep, not just the restart.** On a spectrum with a `1e-9` cluster a
  `reort=NONE` sweep returns Ritz values off by `~4.0` (on `|E| <= 5`) *before any restart*
  (`max_restarts=0` reproduces it at `n = 80, 100, 120, 160`). TRLM's Rayleigh-Ritz rebuild repairs
  it when it fires, but a sweep that terminates on an invariant subspace never restarts and the
  direct extraction inherits the garbage. That is the documented limitation of the mode (no
  orthogonality guarantee), not a bug — but it is why `test_trlm_restart_widths.py` asserts accuracy
  only in the configurations that do restart.
* TRLM may return **fewer than `num_wanted`** pairs when the retained Ritz block deflates below it
  and the continuation closes on an invariant subspace. Documented on `_trlm_core`; the one
  production caller (`cipsi_solver`) already uses `len(e_ref)`.
* The `1e-5` floors are gone from `_trlm_core`. `_irlm_core` has no norm-based floor — its restart
  guards are width-based (`total < m_act * p`, `res_width < p`, `active_k < p`) and are now driven
  by the scale-invariant deflation, so a small-but-full-rank residual block no longer stops it.
  `_assemble_results` still rejects a candidate Ritz vector on an absolute `|<c,c>| < 1e-12`, which
  is a duplicate-detection test on a normalized column, not an operator-scale one. Fine as it is.
