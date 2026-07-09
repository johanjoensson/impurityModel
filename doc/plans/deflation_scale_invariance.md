# Scale-invariant deflation in `_cholesky_or_deflate`

**Status (2026-07-09):** analysed, attempted, **reverted**. The change is correct and necessary;
it cannot land until a latent width-bookkeeping bug in `_trlm_core` is fixed. The attempted patch
is preserved in the session scratchpad as `BlockLanczosArray_relative_deflation.pyx`; everything
needed to redo it is below.

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

`D = int(sum(cur_widths))` (`BlockLanczos.pyx:1168`) says the accumulated Krylov space is
108-dimensional; `Q_basis` holds 105 columns. **The width array counts directions whose `Q`
columns were never appended** — the trailing residual block. While small-but-full-rank blocks were
being deflated to rank 0, the recurrence never reached the state where the two disagree; the
absolute floor was, accidentally, keeping TRLM out of a broken code path. That is not a reason to
keep the floor, but it is the reason this change cannot land alone.

This is the same family as the two IRLM deflation sites (see
`blocklanczos_reort_reliability.md`): a width guard that covers the `alpha` path and misses the
trailing residual block.

**No existing test covers it.** With the broken deflation in place, `pytest -k "lanczos or trlm or
irlm or cg"` is 195 passed. Only the real NiO ground state (`_nio_workload`, `dense_cutoff=50`, so
TRLM rather than the dense path) reaches the failing branch.

## Worklist

- [ ] Make `_trlm_core`'s Krylov dimension the single source of truth. Either derive `D` from
      `Q_basis`'s column count rather than `sum(cur_widths)`, or append `Q` columns for every
      direction counted in `block_widths`. Determine which of the two is the actual invariant —
      `eigh_block_tridiagonal(alphas, betas, block_widths)` and `_build_full_T` both size `T` from
      the widths, so `T`'s dimension and `Q_basis`'s width must agree by construction.
- [ ] Add a regression test that drives a restarted recurrence past a **small but full-rank**
      residual block (`||beta|| ~ 1e-9`, `cond(beta) ~ 1`), i.e. exactly a warm start from
      converged eigenvectors. Must fail on today's `_trlm_core`.
- [ ] Then switch `_cholesky_or_deflate` to the relative rank test + explicit `BREAKDOWN_TOL`
      breakdown test (both paths: Cholesky fast path on `diag(L)**2`, `eigh` fallback on `evals`).
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
