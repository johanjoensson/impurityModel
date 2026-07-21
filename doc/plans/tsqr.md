# TSQR: block orthonormalization without the Gram matrix

> **Status (2026-07-21): implemented and shipped**, including the `DEFLATE_TOL` relaxation
> that was originally deferred. Every block-Krylov call site in the package factors through
> `src/cython/TSQR.pyx`; `_cholesky_or_deflate` / `_cholesky_qr2` survive only as the
> reference implementation their regression tests are written against.

## Why

Every block-Krylov routine here has to factor a tall, skinny block `A` (`n_local x p`, rows
partitioned disjointly over the communicator) into `A = Q beta`. Until now that went through
the Gram matrix: `M = A^H A`, one `Allreduce`, then Cholesky (`_cholesky_or_deflate`), giving
`beta = L^H`, `beta_inv = inv(beta)`, `Q = A beta_inv`.

Forming `A^H A` squares the condition number, and everything unpleasant follows from that:

- **It has a hard ceiling.** Above `kappa(A) ~ EPS^(-1/2)` the Gram is not numerically
  positive definite and Cholesky fails outright — the `eigh` fallback exists only for this.
- **Its `Q` is not orthonormal.** A single Cholesky-QR leaves `||Q^H Q - I|| = O(kappa^2 EPS)`.
  That is what forced the conditional CholeskyQR2 second pass, at the cost of a second Gram
  and a second `Allreduce` per step.
- **It corrupted the deflation policy.** `DEFLATE_TOL` was tightened to `EPS^(1/3)` (~6.06e-6)
  *so that the retained block would stay inside CholeskyQR2's `kappa <~ EPS^(-1/2)` recovery
  regime* — i.e. blocks were being deflated to protect the factorization rather than because
  their directions were dependent.
- **Its rank decisions were made on squared quantities**, where a singular value is resolved
  only to `EPS sigma_max^2 / sigma`.

Measured, on a block at `kappa = 1e5` — inside the range the rank floor deliberately
*retains*, so this is a block the recurrence keeps and uses:

| | `\|\|Q^H Q - I\|\|` |
|---|---|
| single Cholesky-QR | 3.6e-7 |
| TSQR | 2.7e-16 |

3.6e-7 is already above `REORT_TOL = sqrt(EPS)` — the semi-orthogonality level the whole
partial-reorthogonalization apparatus assumes it is maintaining. The `||beta|| -> 1e90`
blow-ups recorded in `blocklanczos_reort_reliability.md` and `test_block_lanczos_blowup.py`
are this weakness surfacing.

## What it does

Two passes over the tall data (`src/cython/TSQR.pyx`):

**Pass 1 — the triangular factor, `A` read-only.** Walk the local rows in panels
(`panel_rows`, default 512): copy each panel into a Fortran scratch, factor with LAPACK
`zgeqrf`, keep only its `p x p` triangular factor, discard the reflectors, and merge it into
a running triangle. The merge is a hand-rolled complex Givens sweep over *row-wise packed
flat* triangles — the stacked `2p x p` matrix is never materialized, and because a packed row
is contiguous from its diagonal onward, each rotation touches two contiguous runs. Cost
`O(n p^2)` flops in `O(panel_rows * p + p^2)` scratch, with no copy of the block.

The rank-local triangles are combined by **one `Allgather`** of the packed triangles followed
by the same Givens merge replayed **in rank order on every rank**. That is deliberate: the
global `R` comes out *bitwise identical* everywhere, and each rank then decides the deflated
block width from it independently without a further collective. A rank owning no rows
contributes `R = 0`, which the merge skips entry by entry.

**Pass 2 — `Q` from `R`.** Canonicalize `R` to a real non-negative diagonal (making the
factorization unique and matching the convention the stored `beta`s always had), then take its
`p x p` SVD — tiny, and it yields the block's true singular values:

- non-finite entry → `k = -1` (**corrupted recurrence**, not a closed space),
- `sigma_max <= BREAKDOWN_TOL * scale` → `k = 0` (**invariant subspace**),
- otherwise `k = #{sigma_i > DEFLATE_TOL * sigma_max}`.

At full rank `Q = A R^{-1}` by back substitution (one `ztrsm`; no inverse is formed). Under
deflation the retained directions are `Q = A V_k Sigma_k^{-1}` with `beta = Sigma_k V_k^H`,
which needs no invertible `R` and whose conditioning is bounded by `1/DEFLATE_TOL`.

The triangular solve inherits `||Q^H Q - I|| = O(kappa EPS)`, so `tsqr` repeats itself once,
folding the correction into `beta`, when `kappa > REFINE_TOL = EPS^(-1/4)` (~1.5e4) — below
that the first pass is already orthonormal to better than `EPS^(3/4)`. Unlike CholeskyQR2 this
repetition is never a *rescue*: pass 1 cannot fail.

## Where it is used

`block_tsqr` (in `src/cython/_reort.pxi`) dispatches on the block representation — dense
array, `ManyBodyBlockState`, or `ManyBodyState` list — so all of these run the same
factorization:

| call site | what it replaced |
|---|---|
| `block_lanczos_array_cy` (`BlockLanczosArray.pyx`) | Gram + Cholesky + conditional CholeskyQR2 |
| `block_lanczos_step_cy` (`_lanczos_step.pxi`) | same, plus the forced CholeskyQR2 after truncation |
| `block_normalize_array` / `block_normalize_sparse` | a *single* Cholesky-QR, no second pass at all |
| IRLM restart block (`_irlm.pxi`) | Gram + NaN screen + Cholesky + `block_combine(beta_k_inv)` |
| `block_bicgstab` entry (`BiCGSTAB.pyx`) | normalized-Gram deflation + rescaling of `beta_j`/`beta_inv` |
| `block_gmres` entry and Arnoldi block (`GMRES.pyx`) | same |

The `ManyBodyBlockState` path costs no conversion: the block's shared-support
`(rows x width)` coefficients are exported through its buffer protocol and the factor is
written back with the new `from_keys_and_amps`.

Two things fall out of the factorization that call sites used to compute for themselves:
`||beta||_2 = sv[0]` (so the per-step `np.linalg.svd(beta_i)` is gone) and
`||beta^+||_2 = 1/sv[k-1]` for the locked-overlap estimator. In BiCGSTAB/GMRES the Gram was
also being built just to read the largest column norm off its diagonal; that is now an `O(n)`
column-norm reduction instead of an `O(n^2)` Gram.

## The `DEFLATE_TOL` relaxation (done, and it bought something)

The `EPS^(1/3)` floor (6.06e-6) existed to keep the retained block inside CholeskyQR2's
recovery regime. TSQR needs no such protection, so the floor was free to say what it is
supposed to say — which directions are numerically independent — and it is now `EPS^(2/3)`
(3.67e-11).

**What it fixes.** Ten cells of `test_no_ghost_bands.py` (five serial, five MPI) were marked
xfail with *"restarted block Lanczos cannot resolve a degeneracy exceeding the block size
within a tight restart subspace (partial T_full -> spurious Ritz values)"*. They now pass.
The mechanism is not subtle: that spectrum splits its eigenvalues by **1e-9 relative**, and a
floor of 6.06e-6 sits three orders *above* the splitting, so the second copy of each
near-degenerate pair was deflated away as rank-deficient — leaving `T_full` partially filled
and generating exactly the spurious Ritz values the test is named after. Scanning the floor
puts the threshold precisely at the splitting: xfail at 1e-9, xpass at 1e-10 and below.

**Why `EPS^(2/3)` specifically.** TSQR resolves a singular value of `R` to `~EPS*sigma_max`,
so 3.67e-11 is five orders above the noise it could be measuring while below any physical
scale a calculation is likely to care about.

**What it does *not* change.** Across the whole suite (23825 factorizations) deflation fires
2927 times, and **94.2% of those discard an exactly-zero singular value** — closed sectors,
where no floor matters. Only 57 sat above `EPS^(1/2)`, nearly all in IRLM restart paths. On
the FCC Ni production ground state the floor never fired at all, and the relaxation left the
converged energies bit-identical (`-14.985551211434395`), at the cost of ~10% more blocks
(3209 vs 2911) because the restart locks later.

**The trade, and why the floor became a per-call argument.** Checking the RIXS path found the
limit of a single global value. RIXS builds its right-hand side from the Cartesian
polarization components, symmetry makes some of them dependent, and the solvers *rely* on
deflation to remove them (group-rule dedup was refuted for rank-4 tensors — automatic rank
deflation is the mechanism). Those directions are zero only to the rounding accumulated
while the seeds were built: on the RIXS tensor benchmark they reach `sigma/sigma_max = 1.2e-9`,
four orders **above** `EPS^(2/3)`. At the default floor six of them were being **retained as
genuine**, each injecting a noise column with `sigma_min ~ 1e-10` into the solve.

So the floor is squeezed by two opposing physical requirements — below any splitting worth
resolving, above the construction noise of a structurally rank-deficient block — and the
window between them is under half an order of magnitude and workload-dependent. `tsqr` and
`block_tsqr` therefore take `deflate_tol` as an argument, exactly as they already take
`scale`: both are properties of *the block the caller is handing over*, not of the
factorization. `scale` answers "zero compared to what?"; `deflate_tol` answers "how much
construction noise do these columns carry?".

* default `DEFLATE_TOL = EPS^(2/3)` — recurrences, where the question is which directions the
  factorization can still resolve;
* `DEFLATE_TOL_SEEDS = EPS^(1/3)` — transition-operator seed blocks and the solves built on
  them (`block_bicgstab`, `block_gmres`, the seeded `block_Green*` recurrences,
  `KrylovShiftedResolvent`). A genuinely distinct polarization component sits at O(1)
  relative, five orders above this floor, so nothing physical is at risk from it.

Measured effect on the RIXS benchmark: the margin between the largest discarded direction and
the floor applied goes from **1.06x to 4901x**, and three more directions deflate that should
always have done.

The residual trade is unchanged for the recurrences: retained `kappa` is bounded by
`1/DEFLATE_TOL ~ 2.7e10` rather than 1.7e5, which propagates into `||beta^+||` in the
W-estimator and into the conditioning of `T`.

**Consequence for the calibrated tests.** `EPS^(2/3)` is *below* `BREAKDOWN_TOL * ||H||` for
any `||H|| > 37`, so the window the CholeskyQR2-era regression tests were written against —
"small enough that the old absolute clamp deflated it, large enough not to be breakdown" — no
longer exists. Those tests (`test_warm_restart_refines`, `test_deflation_scale_invariance`,
`test_block_lanczos_blowup`) now size their scenarios against the constant that actually
governs what each asserts: a local `HISTORIC_ABSOLUTE_FLOOR = EPS^(1/3)` where the point is
the historic bug, and `BREAKDOWN_TOL` where the point is "tiny in absolute terms".

## Open follow-up

**Reduction shape at very large rank counts.** The `Allgather` + in-rank-order merge is
`O(P p^3)` local work and `O(P p^2)` buffer. For `p <= 32` that is negligible up to a few
hundred ranks. A butterfly (recursive-doubling) reduction would make it `O(p^3 log P)` at the
cost of hand-rolled point-to-point code and a canonical pairing rule to preserve the
bitwise-identical-`R` property. Not needed at present rank counts.

## Tests

`src/impurityModel/test/test_tsqr.py` (serial + `@pytest.mark.mpi`, run at `-n 2` and `-n 3`):
packed-storage round trips; the Givens merge against a dense stacked QR; panel-height
invariance; local blocks with fewer rows than columns and with no rows at all; exactness and
orthonormality against `numpy.linalg.qr`; the deflation / breakdown / corruption contract; the
Cholesky-QR comparison quoted above; and, distributed, that `R` matches the serial factor, is
**bitwise** identical on every rank, that `Q` is globally orthonormal, and that empty ranks and
poisoned ranks are handled consistently everywhere.
