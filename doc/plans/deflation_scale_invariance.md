# Scale-invariant deflation in `_cholesky_or_deflate`

**Status (2026-07-09):** the `_trlm_core` blocker is **fixed and landed** (see the worklist); the
`_cholesky_or_deflate` change itself is analysed, attempted, **still reverted**. The attempted
patch is preserved in the session scratchpad as `BlockLanczosArray_relative_deflation.pyx`;
everything needed to redo it is below. Note the original diagnosis of the blocker was wrong — the
correction is in "The fix, and why it cannot land yet".

## The defect

`_cholesky_or_deflate` (`src/cython/BlockLanczosArray.pyx`) answers two different questions with
one test:

```python
keep = evals > DEFLATE_EVAL_TOL * max(evals[-1], 1.0)     # DEFLATE_EVAL_TOL = EPS**(2/3) ~ 3.67e-11
```

* **Rank deficiency** — are some column *directions* of the residual block linearly dependent?
  That is a statement about `lambda_min / lambda_max` and must be judged *relative* to the block's
  own largest singular value.
* **Breakdown** — is the block numerically *zero*, i.e. is the block-Krylov space closed
  (invariant subspace)? That is an *absolute* statement.

The `max(..., 1.0)` clamp fuses them. Any block whose largest singular value falls below
`DEFLATE_TOL = EPS**(1/3) ~ 6.06e-6` is declared rank 0 **regardless of its conditioning**. A
small, perfectly well-conditioned block is treated as an invariant subspace.

Consequence: **every warm-started solver is silently handed back its own input.**

* `block_bicgstab`: once `||R0|| < 6.06e-6` the initial residual deflated to rank 0 and `x0` was
  returned unrefined, whatever `atol` asked. Fixed *locally* in commit `f4d2aea` by normalizing
  the Gram inside `block_bicgstab` before calling `_cholesky_or_deflate`, plus an explicit
  `||R0|| <= atol * ||Y||` early exit. See `doc/plans/bicgstab_per_frequency_gf.md` Phase 3a.
* **TRLM**: `CIPSISolver.expand` leaves `self.psi_refs` populated, and `get_eigenvectors` reuses
  them as the Lanczos start block. Those vectors are already eigenvectors to `||r|| ~ 2.2e-9`, so
  the first Krylov residual block has `||beta|| ~ 2.2e-9`, well under the `6.06e-6` clamp. The
  block deflates, `invariant_subspace` fires, and TRLM returns the warm-start vectors
  **unimproved, before a single restart** — measured: `2.180e-09` in, `2.180e-09` out, bit for bit.

That last one caps the production ground-state accuracy at ~`2e-9` no matter what `tol` says, and
it is why `cipsi_solver._eigen_tol` (which derives TRLM's `tol` from `slaterWeightMin`) is
currently a **no-op on the production path**. Perturbing the warm start by `1e-6` breaks the
spurious invariance and the same solve reaches `8.3e-11`; a fresh start reaches `1.75e-11`;
`reort=FULL` reaches `3e-13`.

## `BREAKDOWN_TOL` is dead code

```python
BREAKDOWN_TOL = 1e-12   # absolute: ||beta||_2 below this => invariant subspace
```

Declared at `BlockLanczosArray.pyx:86`, documented in the module docstring, and **referenced
nowhere**. Invariant-subspace detection rides entirely on the `max(..., 1.0)` clamp returning
rank 0. Any fix must switch `BREAKDOWN_TOL` on — and switching it on is what exposes the bug below.

## The fix, and why it cannot land yet

```python
# breakdown: absolute, on ||beta||_2 = sqrt(lambda_max)
if lam_max <= BREAKDOWN_TOL ** 2:
    return None, None, 0
# rank deficiency: relative, scale invariant
keep = evals > DEFLATE_EVAL_TOL * lam_max
```

(and the mirror of both in the Cholesky fast path, on `d2 = diag(L)**2`). The conditioning
guarantee is untouched — a retained block still has `cond <~ EPS**(-1/3)`, inside the CholeskyQR2
recovery regime — because the relative test is exactly the one that bounds
`lambda_min / lambda_max`. Only the "small ⇒ rank 0" behaviour is removed.

Applied, the NiO ground-state solve dies:

```
_trlm_core (BlockLanczos.pyx:1194)
    Q_ret = block_combine(_q_slice(Q_basis, 0, D), Y_k, 0.0)
ValueError: matmul: size 108 is different from 105
```

**Resolved (2026-07-09), and the diagnosis above was wrong.** The trailing residual block is
*correctly* excluded from `block_widths` on every sweep exit; `sum(widths) <= cols(Q)` always
holds. The culprit was inside the restart loop: `Q_ret, _ = block_normalize(Q_ret, ...)` can
**deflate the retained Ritz block** from `nkeep` to `k_ret < nkeep` columns, while
`cur_widths = [nkeep, p_resid]` recorded `nkeep` regardless. The desync then surfaced one restart
later as the opaque `matmul` error above (`nkeep = 12 -> k_ret = 9`, hence `108` vs `105`).

Nothing about that requires the deflation change — it is reachable on today's kernel, and the
Ritz-block deflation is a *relative* test (`lambda_max(Gram) ~ 1`), so the `max(..., 1.0)` clamp
never applied to it. See `test_trlm_restart_widths.py`.

Two further defects came out of fixing it:

1. **The thick-restart coefficient shortcuts need `Q^H Q = I`, not just full rank.** Both
   `T_k = diag(theta_keep)` and `cross = beta_res @ Y_last` are derived from the recurrence
   identity `H Q = Q T + q_m beta_res E_last^H` *together with* orthonormality of `Q`. Under
   `reort=NONE` the retained block's measured `||Q^H Q - I||` is **1.0** — total loss — while
   still testing full rank, so the old code sailed into the shortcut and produced eigenvalues
   off by `4.1e3` on a spectrum bounded by `|E| <= 5`. Silently.
2. **Rescaling the shortcut by the orthonormalizing factor does not rescue it.** The first
   attempt set `T_k = S^H diag(theta) S` with `S = pinv(beta_ret)`; because `||S|| = 1/sigma_min`
   it amplified the very error it was meant to correct (returned `-3.2e9`).

The fix is to gate on semi-orthogonality (`RESTART_ORTH_TOL = sqrt(EPS)`, Simon's criterion, the
level `PARTIAL` maintains by construction) and, when it fails, rebuild the restart with an
explicit Rayleigh-Ritz step on the orthonormalized retained basis — `T_lead = Q_ret^H H Q_ret`,
`q_m` = orthonormalized `(I - P) H Q_ret`, `cross = beta_res` — which assumes nothing about
`Q_basis`. It costs `k_ret` matvecs and only fires where the premise is violated. Measured
retained-block `||Q^H Q - I||`: `FULL` ~1e-14, `PARTIAL` ~5e-10, `NONE` 1.0. Note the residual is
then no longer rank `<= p` (that bound also needed `Q^H Q = I`), so `T_full` must be sized off
`p_resid`, not `p`.

`_irlm_core` and the Green's-function paths were audited and do **not** violate the invariant:
IRLM guards both `total < m_act * p` (diagonal deflation) and `res_width < p` (trailing residual),
and the GF drivers never build a retained Ritz block. The invariant is now *checked* rather than
assumed, by `_check_width_sync` at every point where widths and a stored `Q` meet, plus a length
guard in `_trim_blocks` (whose `k = len(widths)` would otherwise silently shorten the continued
fraction).

**No existing test covered any of it.** `pytest -k "lanczos or trlm or irlm or cg"` was 195 passed
with all three defects live.

### What the deflation change does now that the blocker is gone

Re-applied on top of the width fix, `_cholesky_or_deflate`'s relative rank test no longer crashes,
and it delivers the promised accuracy — but **only when the start block is not itself a
near-invariant subspace**. Measured on the NiO ground state (`scratchpad/trlm_gate.py`):

| start block | outcome |
|---|---|
| `psi_refs` from `CIPSISolver.expand` (`\|\|r\|\| = 2.2e-9`) | sweep **diverges** at `it=37` (`\|beta\| = 1.17e5` vs spectral scale `1.11e2`); guard truncates; direct extraction gives `\|\|r\|\| = 7.86e3` |
| the same, perturbed by `1e-6` | converges in 8 restarts, `\|\|r\|\| = 8.36e-14` |

So the change *works* (`8.4e-14` beats the fresh start's `1.75e-11` and `reort=FULL`'s `3e-13` under
the old floor, and `_eigen_tol` finally bites), and the width fix was a real prerequisite — but
`BREAKDOWN_TOL = 1e-12` is on the wrong scale. A warm-start residual of `2.2e-9` against
`||H|| ~ 1.1e2` is `2e-11` *relative*: a near-invariant subspace by any sensible measure, yet the
absolute test lets it through, `beta_inv ~ 4.5e8` amplifies it, and 37 steps later the recurrence
is gone. **Breakdown must be judged relative to the operator scale** (`||T||`, or the running
`h_norm_est` the divergence guard already maintains), not absolutely. That is the next thing to
design — not a mechanical port of the reverted patch.

### Still open: `reort=NONE` corrupts the sweep, not just the restart

Independent of everything above, a `reort=NONE` sweep on a spectrum with a `1e-9` cluster returns
Ritz values off by `~4.0` (`|E| <= 5`) **before any restart** — `max_restarts=0` reproduces it at
`n = 80, 100, 120, 160`. The restart's Rayleigh-Ritz rebuild happens to repair it when it fires,
but when the sweep terminates on an invariant subspace there is no restart and the direct
extraction inherits the garbage. That is the documented limitation of the mode (no orthogonality
guarantee), not a bookkeeping bug, and it is why `test_trlm_restart_widths.py` asserts accuracy
only in the configurations that do restart.

### Still open: two more absolute floors in `_trlm_core`

`np.linalg.norm(beta_i, ord=2) < 1e-5` (inner loop) and the same test on `beta_res` (rebuild
branch) declare an invariant subspace on an **absolute** threshold. For NiO (`||H|| ~ 2e4`) that is
a relative `5e-10` and harmless; for an `O(1)`-norm operator it caps TRLM's residual at `1e-5`
whatever `tol` asks. Same family as `_cholesky_or_deflate`'s floor — fix them in the same commit,
relative to `max(|eigvals_T|)`.

## Worklist

- [x] Make `_trlm_core`'s Krylov dimension the single source of truth. Done: `cur_widths[0]` is
      now `k_ret = _q_cols(Q_ret)`, the *actual* retained width, and `_check_width_sync` asserts
      `sum(widths) == cols(Q_basis)` at the top of every restart. `Q_basis`'s column count is the
      invariant; `T`'s dimension follows it.
- [x] Add a regression test that drives a restarted recurrence into the broken branch.
      `test_trlm_restart_widths.py`, both paths (the array kernel and `ManyBodyState`, which share
      `_trlm_core`) plus MPI. Fails on the pre-fix kernel: array raises
      `ValueError: matmul: ... 120 is different from 118`, ManyBodyState silently returns
      eigenvalues off by `5.1e3`. *(The warm-start-from-converged-eigenvectors scenario the note
      originally proposed is a `_cholesky_or_deflate` test, not a `_trlm_core` one — it belongs to
      the next item.)*
- [ ] Give breakdown an **operator-relative** scale before switching it on. `BREAKDOWN_TOL = 1e-12`
      absolute is not the test: NiO's warm start (`||beta_0|| = 2.2e-9`, `||H|| = 1.1e2`) passes it,
      then diverges 37 steps later. Candidates: `||beta|| <= tol * h_norm_est` reusing the
      divergence guard's running estimate, or `tol * ||alpha_0||` on the first step. Whatever is
      chosen must also cover the two absolute `1e-5` invariant-subspace tests in `_trlm_core`.
- [ ] Then switch `_cholesky_or_deflate` to the relative rank test + the breakdown test above
      (both paths: Cholesky fast path on `diag(L)**2`, `eigh` fallback on `evals`). Add the
      small-but-full-rank residual-block test (`||beta|| ~ 1e-9`, `cond(beta) ~ 1`) there, and a
      warm-start test asserting NiO reaches `~1e-13` rather than diverging.
- [ ] Re-check `reort=NONE` serial vs MPI. The docstring claims the absolute floor "keeps the
      `reort=NONE` recurrence on the *same* convergent trajectory serially and under MPI". **That
      claim is already false** on the NiO workload with the floor in place:

          reort=NONE, delta=0.01, width 1
            1 rank : m = 114 blocks
            2 ranks: m =  98 blocks   max rel diff in G vs serial = 5.257e-09
            3 ranks: m = 138 blocks   max rel diff in G vs serial = 1.611e-07

      The `G` differences are identical under `reort=PARTIAL` and `FULL`, so they come from the
      excited basis, not from deflation. What the floor buys is protection against *divergence*
      (the `O(kappa)` amplification of `Allreduce` rank-order rounding), not trajectory identity —
      and that protection is the *relative* half of the test, which the fix keeps. Re-measure
      against these numbers after the change; watch for the `BETA_BLOWUP_FACTOR = 1e3` guard
      firing, which is the backstop if `BREAKDOWN_TOL = 1e-12` proves too permissive.
- [ ] Once landed, `block_bicgstab`'s local Gram normalization (commit `f4d2aea`) becomes
      redundant — the shared deflation is then scale invariant on its own. Its explicit
      `||R0|| <= atol * ||Y||` early exit must stay: that is the tolerance contract, not a
      deflation artifact.
- [ ] Confirm `_eigen_tol` (`cipsi_solver.py`) starts having an effect: the warm-started TRLM
      residual should fall from `2.180e-09` toward the requested `tol`. Compare against
      `reort=FULL`, which already reaches `3e-13` at `tol=1e-12`. Note `PARTIAL` underpredicts the
      true residual by ~15x (`1.75e-11` delivered for `1e-12` requested), so `tol` is an estimate
      in that mode, not a contract.

## Why this matters beyond tidiness

Every warm-started Krylov solve in the codebase is affected: the RIXS resolvent sweep, the
per-frequency Green's-function driver (`bicgstab_per_frequency_gf.md` Phase 3b), and the CIPSI
ground state via `psi_refs`. Each one silently stops improving once its residual crosses `6.06e-6`,
and reports success. The BiCGSTAB instance was caught only because a frequency sweep made the warm
start good enough to sit *below* the floor on the second mesh point.
