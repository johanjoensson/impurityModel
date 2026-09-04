# impurityModel — project guide

Exact-diagonalization solver for Anderson impurity models (Python + Cython/C++ + MPI).
Architecture: see `doc/architecture_overview.md` (module map, layer diagram, execution flow).

## Build

```bash
pip install --no-build-isolation -e .                      # release (default)
IMPURITYMODEL_BUILD=debug pip install --no-build-isolation -e .   # Cython checks on
```

`--no-build-isolation` requires the build prerequisites in the environment first:
`pip install numpy scipy cython "setuptools>=77.0.3" setuptools-scm`.
The C++ layer needs Boost headers (env var `BOOST_ROOT` for custom locations) and C++17+
(`CXX`/`CXXFLAGS` respected). Threaded apply: `IMPURITYMODEL_PARALLEL=1` at install time.

**Build modes** (`IMPURITYMODEL_BUILD`, all directives centralized in `setup.py`'s
`_DIRECTIVES` — a `# cython:` header in a `.pyx` would override them and silently defeat
the mode, which is why there are none):

| mode | what it is for |
|---|---|
| `release` (default) | `-O3 -march=native -ffast-math`. Fastest. Not portable off this CPU, and `-ffast-math` may let the compiler drop NaN/Inf guards. |
| `debug` | `boundscheck`/`initializedcheck` on at `-O1 -g`. **The only build that catches an out-of-bounds kernel access** — otherwise such a read just returns whatever is there (measured: `1.4e-309`, a subnormal that looks like a plausible number). |
| `safe` | `-O3` only. For heterogeneous clusters, and to reproduce pre-2026-09 numerics. |

`release` deliberately trades the bitwise-identical-across-ranks property that
`cipsi_solver.py` relies on; use `safe` when building per node on a cluster.

Switching mode re-cythonizes: cythonize decides staleness from timestamps and does *not*
notice a directive change, so `setup.py` records the mode in `src/cython/.build-mode` and
forces regeneration when it differs. Without that, switching modes silently keeps the old one.

Editing anything under `src/cython/` requires re-running the pip install to recompile.
Extras: `.[dev]` (pytest, pytest-mpi, black, ruff, mypy, cython-lint), `.[doc]` (Sphinx).

## Test gate

Run both after every change; each commit should be green on both:

```bash
python -m pytest
mpiexec -n 2 python -m pytest --with-mpi
```

CI runs serial, `-n 1`, `-n 2`, and `-n 3`. Benchmarks are opt-in: `pytest -m benchmark`.
MPI tests are marked `@pytest.mark.mpi`; non-root rank output goes to `.pytest_mpi_rank*.out`.
An empty rank (owns zero local determinants under `routing_hash() % comm.size`) only
appears at `-n 3`+ for small hash-distributed test fixtures — `-n 2` never exercises it.
Run `-n 3` locally when touching `basis_split.py` / `run_units_distributed` (splitting
only activates multi-rank) or the Cython Lanczos kernels / anything gating an MPI
collective (a rank-local truthiness/early-return check that should be rank-invariant
deadlocked the locked-reort Allreduce in `_lanczos_step.pxi` exactly this way).

Docs build: `make -s -C doc/sphinx html` (needs `.[doc]`).

**Diagnostic CI jobs** (`continue-on-error` — they report, they do not gate):

| job | what it covers |
|---|---|
| `test` leg `build: debug` | the checked Cython build — bounds and initialized-memoryview checks. The only configuration that catches an out-of-bounds kernel access; elsewhere such a read silently returns a plausible-looking subnormal. ~0% overhead. |
| `test-asan` | ASan + UBSan over one serial suite run |
| `test-tsan` | ThreadSanitizer with `IMPURITYMODEL_PARALLEL=1`, scoped to `operators`/`lanczos`/`basis` — the only places `ManyBodyOperator::apply`'s three `#if defined(PARALLEL)` blocks run, and the only threaded code in the repo. Also the sole CI coverage of the threaded build. |

Every one of these asserts it is actually active before its clean result counts — a
deliberate out-of-bounds/UB/race must be trapped first. That is not ceremony: ASan ran 68
instrumented times against a real crash while silently misconfigured, and a build-mode
switch was once a no-op because cythonize ignores directive changes. A sanitizer that is
not running and one that finds nothing produce identical output.

**Read sanitizer logs with line-anchored patterns.** `grep "data race"` matches the step's
own script text where it greps for that string; anchor to the report body
(`^.*WARNING: ThreadSanitizer`) or you will report findings that do not exist. This
misfired twice in one session.

## Layering rule

Modules only import downward (see the layer diagram in `doc/architecture_overview.md`):
physics/operator-algebra (`operator_algebra`, `atomic_physics`) never imports solvers;
the basis layer (`manybody_basis` + `basis_generation`/`basis_restrictions`/
`basis_transcription`/`basis_split`) sits below the solvers (`groundstate`,
`greens_function`, `spectra`, `selfenergy`); the CLIs (`get_spectra`, `selfenergy`)
sit strictly on top. `ea16.py` looks like a leaf but is load-bearing: both Cython
Lanczos kernels import it at runtime.

## MPI rules (violations have caused real deadlocks)

- Never gate an MPI collective on rank-local state (e.g. a `verbose` flag that is 0 on
  non-master ranks). If needed, broadcast the decision first.
- No full state-vector gathers. Determinants are hash-distributed (`hash(sd) % size`,
  one owner per determinant); observables go apply-local → `redistribute_psis` →
  local inner product → `Allreduce`.
- `MPI_Comm_free` is collective: free communicators at synchronized points, never from
  the garbage collector (see `basis_split.py`).
- Empty-rank edge cases (a rank owning zero determinants) have bitten before — keep
  collective calls unconditional and buffer dtypes fixed.

## Conventions

- Formatting: black (line length 120, target py311) + cython-lint via pre-commit;
  ruff/mypy via `make check` (configs in `pyproject.toml`/`setup.cfg`).
- **Never write `x[-1]` in `src/cython/`.** Every kernel there carries a module-wide
  `# cython: ... wraparound=False ...` header, which applies to plain `def` functions too:
  on an inferred ndarray, `x[-1]` reaches numpy as `-1 - len(x)` and raises `IndexError`
  (or, on a slice, silently addresses the wrong rows). Spell it `x[len(x) - 1]` /
  `x[x.shape[0] - 1]` — the idiom the kernels already use. This has landed twice: once in
  a perf pass, and again when `robust_svd` and `calculate_thermal_gs` took out 21 tests.
- Docstrings: numpy style (Sphinx napoleon).
- Commits: small, single-concern steps (see the R-numbered history on this branch);
  refactors move code verbatim and keep every commit green on the test gate.
- Temp/debug outputs (`h0.txt`, `debug/`, `*-realaxis-bench.dat`, …) are gitignored;
  don't commit calculation droppings.
