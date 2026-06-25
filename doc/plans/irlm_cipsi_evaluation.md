# IRLM + CIPSI implementation review (2026-06-25)

Scope: the EA16-style implicitly restarted block Lanczos (`irlm.py`, `ea16.py`, the
`BlockLanczosArray`/`BlockLanczos` Cython kernels) and the CIPSI selected-CI driver
(`cipsi_solver.py`, `groundstate.py`), in both the dense/sparse array path and the
distributed `ManyBodyState` path, serial and MPI.

This review was done while fixing the spurious-eigenvalue bug documented in
`distributed_irlm_groundstate_mpi.md`. Numbers below are from the 10-orbital/5-electron
random-Hermitian sector (full dimension C(10,5)=252) used by
`test_groundstate_and_density_matrix_mpi`, plus targeted unit reproductions.

## Correctness

**Fixed (this change).** The IRLM violated the Rayleigh-Ritz lower bound (returned
eigenvalues below the dense minimum) for intermediate subspace sizes, in both reort
modes, serial and MPI. Root cause: the inner sweep was not deflated against the locked
Ritz vectors `Xl` (EA16 §2.6.2), so the matvec regenerated locked directions into the
active subspace. A second, related defect returned each eigenvalue twice when IRLM was
seeded from converged eigenvectors. Both are fixed and pinned by
`test_irlm_locking_deflation.py`. After the fix, direct IRLM reproduces the dense
spectrum to ~1e-6 for all `max_subspace_blocks` and both reort modes, with orthonormal
eigenvectors and no duplicates.

**Verified good.**
- The distributed matvec is correct. The column-distributed `H_mat[:, local_indices]`,
  `apply_sparse_csr_nogil` -> `Allreduce` -> slice-local-rows path reproduces the serial
  product; the interception test showed every `H_mat` handed to the solver is Hermitian
  with the correct dense minimum. The earlier note's worry about the operator was
  unfounded.
- `ea16.purge_restart` (explicit-shift re-banding, eq. 6) and `ritz_residual_norms` /
  `acceptance_tol` (eq. 15) are faithful to the report and are not implicated in the bug.
- Deflation/shrinking-block handling (`_cholesky_or_deflate`, width-aware `T`) is
  consistent between the sweep, `_irlm_core`, and `_assemble_results`.

## Stability

- **PRO / PARTIAL under MPI is now stable.** The previously reported `maxM ~ 5e165`
  overflow was a *symptom* of the missing locking deflation (the regenerated locked
  direction dominating the recurrence), not an independent PRO failure. With deflation,
  the default `reort=PARTIAL` path that `calc_gs` uses for `expand` / `get_eigenvectors`
  runs cleanly at np=2.
- **MPI determinism.** Energies now agree serial vs np=2 to <1e-10 on the regression
  case. The non-associative `Allreduce` still perturbs the trajectory at the ULP level;
  the fix removed the instability that *amplified* that perturbation, but exact
  bit-for-bit cross-rank reproducibility is not guaranteed (and is not required by the
  tests, which use `atol=1e-10`).

## Performance

- **Deflation cost.** The new locked projection adds, per Lanczos step, two
  `(nlock x N)(N x p)` GEMMs plus one `Allreduce` of an `(nlock x p)` overlap, where
  `nlock <= num_wanted`. Negligible against the matvec for the array path; for the MBS
  path it is two `block_orthogonalize_sparse` passes over the locked state-list per step.
  Cheap relative to the sweep, and only active once pairs start locking.
- **Array MPI path forms the global vector on every rank** (documented guardrail at
  `BlockLanczosArray.pyx`): per-rank memory scales with `global_N`, not `local_N`. Fine
  for the small/dense sectors this path targets (`dense_cutoff`), but it is *not* a
  scalable distributed eigensolver — large sectors must use the hash-distributed MBS
  kernel. Worth a one-line assertion/doc at the CIPSI call site so nobody routes a large
  sector through the array IRLM by accident.
- **CIPSI subspace heuristic is loose.** For the 252-dim sector,
  `get_eigenvectors` chose `max_subspace_blocks=100` and `num_wanted=20` (i.e. a Krylov
  space nearly half the whole sector before restart). That is what put the solver in the
  fragile intermediate-`msb` regime. Now safe, but the heuristic
  (`2*ceil(max_subspace/len(psi0)) + 20`, then clamped) is generous; tightening it would
  cut work and keep restarts frequent (the self-correcting regime).

## Latent issues / future improvements

Status updated 2026-06-25 — items 1-4 addressed in the follow-up pass.

1. **MBS-path projection placement.** ✅ FIXED. The MBS kernel now projects the residual
   block against the locked set *inside* `block_lanczos_step_cy`, before forming
   `M`/`beta` (`locked=` argument), so the stored `beta` and `q_next` are consistent with
   the deflated vector — identical placement to the array kernel.
2. **`select_restart_indices` only excludes `locked_local`.** ✅ ADDRESSED (capability
   added, intentionally left disabled). `select_restart_indices` gained an optional
   `locked_evals`/`ghost_tol` ghost filter, but the IRLM driver keeps it **off** on
   purpose: eigenvalue-based filtering cannot tell a loss-of-orthogonality ghost (same
   eigenvalue *and* eigenvector) from a genuine degeneracy (same eigenvalue, orthogonal
   eigenvector that must be kept). The inner-sweep deflation removes ghosts by
   *eigenvector*, which is the correct discriminator and is degeneracy-safe. The filter
   is documented + unit-tested for callers that cannot deflate the sweep.
3. **Reseed-collapse path.** ✅ DOCUMENTED. `_assemble_results` now carries a docstring
   note that it may return *fewer* than `num_wanted` pairs when the reachable invariant
   subspace is smaller (deduplication drops copies rather than returning spurious ones);
   callers must use `len(eigvals)` (CIPSI already filters by `max_energy`).
4. **CIPSI start vector.** ✅ DOCUMENTED. `expand`'s per-rank `random.seed(42+rank)` seed
   now carries a comment: the final energy is reproducible across rank counts but the
   basis-selection *trajectory* is not; seed from a rank-independent global vector if
   cross-`np` basis reproducibility is ever needed.
5. **`trlm` vs `irlm`.** ✅ FIXED. The thick-restart solver (`CIPSISolver`/`expand`
   default) was confirmed broken — it crashed or diverged to ~1e150 on every audited
   configuration — from two bugs, now fixed in both the array (`trlm.py`) and
   ManyBodyState (`thick_restart_block_lanczos_cy`) paths:
   - **Unnormalized start block (array path only).** `thick_restart_block_lanczos` fed
     `psi0` straight into `block_lanczos_array`, which assumes an orthonormal start and
     does *not* normalize internally. A large-norm random start made the three-term
     recurrence operate on non-unit vectors, so the betas grew geometrically and `T`
     overflowed within ~10–15 steps. Fixed by normalizing `psi0` (as the IRLM driver
     already did). The MBS path already normalized.
   - **Not deflation-aware (both paths).** Both assumed a uniform block width
     (`m_actual * n`); once the sweep shrank a block (rank-deficient residual), the
     `T_full`/`Q` bookkeeping desynchronized and `block_combine` crashed on a dimension
     mismatch. Fixed by building `T` with `block_widths` and, when the sweep deflates
     (or returns an invariant subspace), extracting the Ritz pairs directly instead of
     entering the uniform-width restart loop (whose arrowhead assumes constant width).

   - **Uniform-width restart loop (both paths).** ✅ FIXED (follow-up). The thick-restart
     *continuation* loop was rewritten to be fully width-agnostic: it tracks each block's
     actual width (`cur_widths`) and addresses `T_full`/`Q` by cumulative column offsets
     instead of a constant `n`/`p`, so a block that deflates **mid-restart** (shrinking
     from `n` to `0 < k < n`) places a rectangular arrowhead beta correctly. This also
     fixed a latent crash where the *residual* block deflates while the diagonal blocks do
     not (leaving `total == m_actual*n`, so the run enters the restart loop with a
     kernel-padded trailing `beta` of phantom zero rows): `beta_res` is now sliced to the
     true residual rank `p_resid`. The retained Ritz pairs are one diagonal super-block of
     `k_blocks*n` columns coupled to the residual by the standard spike.

   Pinned by `test_trlm_*` in `test_irlm_locking_deflation.py` (array + MBS + MPI, an
   explicit unnormalized-start guard, a width-agnostic restart-loop test across start
   widths 1/2/3, and a block-deflation-in-restart test). Note: for `num_wanted` approaching
   the sector dimension `N` with a multi-vector start, the block-Krylov subspace may not
   span the highest wanted eigenvalues, so those few may be under-converged — a fundamental
   Krylov limitation, not a TRLM bug (the GS and the no-spurious invariant always hold).

## Kernel precondition discovered

`block_lanczos_array` (array kernel) **assumes an orthonormal start block** and does not
normalize it internally — an unnormalized start diverges (see TRLM bug above). Both
drivers (IRLM, TRLM) now normalize before calling. If a future caller is added, normalize
first, or add a defensive `block_normalize` at the top of `block_lanczos_array_cy`'s
fresh-start branch. The MBS kernel (`block_lanczos_cy`) is reached only through
`thick_restart`/IRLM, both of which normalize.

## One-line summary

The distributed IRLM had a single real bug — it locked converged Ritz pairs without
deflating the inner Lanczos sweep against them — which produced eigenvalues below the
true spectral minimum (and, once corrected, exposed a duplicate-eigenpair path). Both are
fixed in the array and ManyBodyState kernels and pinned by regression tests; the CIPSI
matvec and restart math were correct.
