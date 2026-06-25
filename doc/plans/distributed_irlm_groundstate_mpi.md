# Distributed IRLM ground-state correctness & stability under MPI

**Status:** RESOLVED (2026-06-25). The failing test
`test_groundstate_and_density_matrix_mpi` now passes; full suites green
(serial 421 passed, MPI np=2 560 passed). The fix is a single root cause — see below.

> **Correction to the earlier diagnosis.** The previous version of this note framed the
> bug as "two independent root causes" and took the serial energy `-17.78` (later
> `-20.34`) as the truth and the MPI energy `-10.42` as wrong. **That was backwards.**
> The dense diagonalization of the full C(10,5)=252 sector gives a ground-state energy of
> **-10.41746**. A Rayleigh-Ritz projection (which is what block Lanczos computes) can
> *never* return a value below the true minimum, so `-17.78` / `-20.34` were the spurious
> answers and `-10.42` was correct all along. There was **one** bug, with two symptoms
> (an overflow runaway, and a "clean" convergence to a spurious value), both serial and
> MPI — MPI just tipped over more readily because the non-associative Allreduce perturbs
> the trajectory.

## Root cause: missing locking deflation (EA16 §2.6.2)

The IRLM driver (`_irlm_core`) **locks** converged Ritz pairs into `Xl` and continues in a
compressed subspace, but the inner block-Lanczos sweep (`block_lanczos_array_cy` /
`block_lanczos_cy`) was **not kept orthogonal to `Xl`**. Only the restart *seed* was
orthogonalized against the locked set; the subsequent matvec-generated Lanczos vectors
were free to redevelop a component along the locked directions. Because the locked
ground state is the dominant low eigenvector, `H @ q` re-amplifies it every step, so it
(and its `2*theta` harmonic) re-enters the active subspace and the projected matrix `T`
acquires Ritz values **below** the true minimum — the classic loss-of-orthogonality
signature.

Diagnostic that pinned it (serial, no MPI needed — `max_subspace_blocks` swept on the
252-dim sector, `num_wanted=20`):

```
msb= 30 FULL    min=-10.41746  (correct)        msb= 30 PARTIAL min=-10.80   (1 below min)
msb= 60 FULL    min=-20.83492  (= 2*lambda_min!) msb= 60 PARTIAL min=-25.43  (4 below min)
msb=100 FULL    min=-17.25     (2 below min)      msb=100 PARTIAL min=-31.73  (6 below min)
msb=200 FULL    min=-10.41746  (correct)         msb=200 PARTIAL min=-10.41746(correct)
```

Two tells: (a) it fails for *intermediate* subspace sizes only — small `msb` restarts
often enough to self-correct, large `msb` captures everything before the destructive
restart; (b) it fails for **FULL** reort too, so it was never a partial-reorthogonalization
(PRO) problem. The verbose restart trace showed the algorithm locking 18 of 20 wanted
pairs and then spinning, periodically emitting `MinEig = -20.834925 = 2 x -10.417462` —
the *doubled, already-locked* ground state.

The earlier "PRO runaway / overflow" symptom (`maxM` growing ~3x/step to `5e165`) was the
same mechanism seen from the other side: once the regenerated locked direction dominates
and orthogonality collapses, the three-term recurrence diverges geometrically. It was not
an independent PRO bug — deflating against `Xl` removes both symptoms, and the default
`reort=PARTIAL` path is now stable serial and under MPI.

## The fix

1. **Inner-sweep locking deflation.** Both Lanczos kernels gained a `locked=` argument and
   project every new Lanczos block orthogonal to the locked Ritz vectors (twice, for
   robustness; MPI-collective via `Allreduce` on the overlap):
   - `src/cython/BlockLanczosArray.pyx`: project `wp` against `locked` before forming
     `M = wp^H wp` (so `beta` and `q_next` are consistent), for every reort mode.
   - `src/cython/BlockLanczos.pyx`: project `q_next` against `locked` after each step via
     `block_orthogonalize_sparse`.
   - `src/impurityModel/ed/irlm.py`: `_irlm_core` passes `locked=Xl` (or `None` when
     nothing is locked) into all three sweep call-sites (initial, reseed, continuation).
2. **Deflated final extraction.** `_assemble_results` now orthogonalizes each active Ritz
   candidate against the locked set (and against already-accepted actives) and skips
   collapses. This fixes a *second*, related symptom newly exposed once the energies were
   correct: seeding IRLM from already-converged eigenvectors (which
   `CIPSISolver.get_eigenvectors` does, restarting from `psi_refs`) made the leftover
   active factorization hold near-copies of the locked pairs, so every eigenvalue was
   returned **twice** — double-counting states in the thermal average.

## Regression tests

`src/impurityModel/test/test_irlm_locking_deflation.py` (13 serial + 2 MPI):
- no Ritz value below the dense minimum across `msb in {30,60,100,200}` x {FULL, PARTIAL};
- random start recovers the distinct lowest eigenvalues, no duplicates;
- converged-eigenvector start returns each reachable eigenvalue once (not twice);
- the ManyBodyState kernel is deflated too (no spurious MBS eigenvalue);
- row-block-distributed (np=2) IRLM matches serial and stays above the spectral minimum.

Run:
```bash
python -m pytest src/impurityModel/test/test_irlm_locking_deflation.py -q
mpirun -np 2 python -m pytest src/impurityModel/test/test_irlm_locking_deflation.py --with-mpi -q
```

## Follow-ups (not blocking; see evaluation note)

See `doc/plans/irlm_cipsi_evaluation.md` for the full stability/correctness/performance
review and the list of latent issues (MBS-path projection placement, `select_restart_indices`
only excluding this-restart locks, CIPSI's large `max_subspace_blocks` heuristic).
