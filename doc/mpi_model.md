# The MPI distribution model

This package parallelizes an exact-diagonalization solver across MPI ranks without ever
materializing a full state vector on any one rank. The model is simple to state and easy to
violate; the violations have caused real deadlocks, so this document states both the model and
the invariants that protect it.

## Determinants are hash-distributed

A many-body state is a superposition over Slater determinants. Each determinant `sd` has
exactly one owner rank, `hash(sd) % size`. A rank stores only the amplitudes of the
determinants it owns. There is **no full-vector representation** anywhere — a "state" is a
distributed object whose pieces live on different ranks.

Consequences:

- **Inner products and observables are three-step**: apply the operator locally
  (`ManyBodyOperator` on the local determinants), `redistribute_psis` to move amplitudes to
  their owners, take the local inner product, then `Allreduce` the scalar. No gather.
  (`manybody_basis.Basis.redistribute_psis`, and the pattern throughout `observables.py` /
  `gs_statistics.py`.)
- **A determinant produced on the wrong rank is silently misplaced.** Any recurrence that
  moves `H·t` between ranks must also move `t` (see the "Chebyshev recurrence seed ownership"
  war story) or seeds diverge.

## `redistribute_psis` sums, it does not deduplicate

`redistribute_psis` routes each amplitude to its owner and **sums** colliding contributions.
If the same state is fed on several ranks (a replicated seed), its amplitude is multiplied by
the replica count, and bilinear maps by its square. The rule: **feed input states on rank 0
only**, then redistribute. A same-pipeline A/B comparison cannot catch a double-count because
both sides double it equally — check against an independent reference.

## The distribution engine: colors

Large runs split the communicator so several independent work units run concurrently, each on
a subset of ranks. `basis_split.split_basis_and_redistribute_psi` partitions
`MPI.COMM_WORLD` into **colors** sized to fit the per-rank memory budget, gives each color its
own sub-communicator and its own clone of the `Basis`, and redistributes the seeds onto the
rebuilt per-color basis. `greens_function.run_units_distributed` drives this for every GF /
spectra / RIXS run (see [`gf_solver_architecture.md`](gf_solver_architecture.md)).

The pure packing math (which units go in which color) is `basis_split._pack_units`; it is unit
tested in isolation (`test_pack_units.py`) precisely because getting it wrong only shows up at
multi-rank scale.

## The invariants (violations have deadlocked)

These are load-bearing. Hold them when changing distributed code.

1. **Never gate an MPI collective on rank-local state.** A `verbose` flag that is 0 on
   non-master ranks, a `if gs is not None` that is only true on rank 0 — gating a collective on
   either makes some ranks skip it and the rest block forever. If a decision must gate a
   collective, broadcast it first (`comm.bcast(decision, root=0)`), then branch on the
   broadcast value on every rank. See `_raise_together` in `selfenergy.py` for the canonical
   fix (turn a rank-0-only verdict into a collective raise).

2. **No full state-vector gathers.** Determinants are hash-distributed; there is one owner per
   determinant. Observables go apply-local → `redistribute_psis` → local-inner → `Allreduce`.

3. **`MPI_Comm_free` is collective.** Free communicators and intercommunicators at
   synchronized points where every rank of the communicator runs the same free, never from a
   destructor or the garbage collector (which fires at rank-dependent times). See the
   collective free in `basis_split.py`.

4. **Empty-rank edge cases are real.** A rank can own zero determinants. Keep collective calls
   unconditional and buffer dtypes fixed regardless of the local partition size — an empty
   local partition once produced int32 CSR indices where the rest of the ranks had int64,
   raising a buffer-dtype mismatch that hung the run.

## Testing distributed code

- The gate runs serial, `-n 1`, and `-n 2`; add `-n 3` when touching `basis_split.py` or
  `run_units_distributed` (splitting only activates with multiple ranks — and a single color
  hides the packing bugs).
- MPI tests are marked `@pytest.mark.mpi` and run under
  `mpiexec -n 2 python -m pytest --with-mpi`. `conftest.py` redirects non-root ranks' output
  to `.pytest_mpi_rank*.out`, so **a green terminal summary is only rank 0** — a failure on a
  non-root rank shows up as a nonzero `mpiexec` exit code and a "process exited with non-zero
  status" banner, with the traceback in the per-rank file. Check the exit code, not just the
  printed summary.
- An unmarked (serial) test still runs on every rank under `--with-mpi`. If it asserts on
  rank-0-only state (e.g. datasets a driver writes only on rank 0), guard those assertions with
  the rank, or it fails on the non-root ranks.
