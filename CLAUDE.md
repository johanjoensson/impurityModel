# impurityModel — project guide

Exact-diagonalization solver for Anderson impurity models (Python + Cython/C++ + MPI).
Architecture: see `doc/architecture_overview.md` (module map, layer diagram, execution flow).

## Build

```bash
pip install --no-build-isolation -e .
```

`--no-build-isolation` requires the build prerequisites in the environment first:
`pip install numpy scipy cython "setuptools>=77.0.3" setuptools-scm`.
The C++ layer needs Boost headers (env var `BOOST_ROOT` for custom locations) and C++17+
(`CXX`/`CXXFLAGS` respected). Threaded apply: `IMPURITYMODEL_PARALLEL=1` at install time.

Editing anything under `src/cython/` requires re-running the pip install to recompile.
Extras: `.[dev]` (pytest, pytest-mpi, black, ruff, mypy, cython-lint), `.[doc]` (Sphinx),
`.[rspt]` (only for the `build_h0` script, which also needs the local `pyRSPthon` project).

## Test gate

Run both after every change; each commit should be green on both:

```bash
python -m pytest
mpiexec -n 2 python -m pytest --with-mpi
```

CI runs serial, `-n 1`, and `-n 2`. Benchmarks are opt-in: `pytest -m benchmark`.
MPI tests are marked `@pytest.mark.mpi`; non-root rank output goes to `.pytest_mpi_rank*.out`.
When touching `basis_split.py` / `run_units_distributed`, also run once at `-n 3`
(splitting only activates multi-rank).

Docs build: `make -s -C doc/sphinx html` (needs `.[doc]`).

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
- Docstrings: numpy style (Sphinx napoleon).
- Commits: small, single-concern steps (see the R-numbered history on this branch);
  refactors move code verbatim and keep every commit green on the test gate.
- Temp/debug outputs (`h0.txt`, `debug/`, `*-realaxis-bench.dat`, …) are gitignored;
  don't commit calculation droppings.
