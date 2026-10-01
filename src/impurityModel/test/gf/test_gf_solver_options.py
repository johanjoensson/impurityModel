"""The per-frequency kernel's basis-growth policy as a solver option (``SolverOptions.gf_admission``).

Until now ``outer`` admission was an environment knob: not part of any input format, not recorded in an
archive, so a run that used it could not be reproduced from its output. These tests pin the option's
contract at the layers it passes through:

* ``SolverOptions`` refuses a combination that would be silently ignored two layers down;
* an explicit argument wins over the matching environment knob, in both directions;
* choosing ``outer`` switches the measured error bound on (and says so in the diagnostics report),
  unless the environment forces it off;
* a whole ``calc_selfenergy`` run with ``outer`` agrees with the Lanczos one.
"""

import contextlib
import io

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import solver_trace
from impurityModel.ed.model import SolverOptions
from impurityModel.ed.selfenergy import calc_selfenergy
from impurityModel.test.gf.test_selfenergy_end_to_end import _basis_and_solver, _meshes, _siam_model
from impurityModel.test.support.gf_oracles import _run_driver

KNOBS = (
    "GF_BICGSTAB_ADMISSION",
    "GF_BICGSTAB_ADMIT_TOL_AMP",
    "GF_BICGSTAB_ADMIT_TOL_JACOBI",
    "GF_BICGSTAB_ADMIT_SCORER",
    "GF_BICGSTAB_RESIDUAL_CHECK",
    "GF_BICGSTAB_WARM_HISTORY",
    "GF_ADMIT_FIRST_SHELL_TOL",
)


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)


# --- the options ------------------------------------------------------------------------------------


def test_the_defaults_leave_the_policy_unspecified_so_the_environment_still_decides():
    options = SolverOptions()
    assert (options.gf_method, options.gf_admission, options.gf_admit_tol) == ("lanczos", None, None)


def test_outer_admission_with_a_tolerance_is_accepted():
    options = SolverOptions(gf_method="bicgstab", gf_admission="outer", gf_admit_tol=1e-5)
    assert options.gf_admission == "outer" and options.gf_admit_tol == 1e-5


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(gf_admission="sometimes"), "expected one of"),
        (dict(gf_method="lanczos", gf_admission="outer"), "needs gf_method='bicgstab'"),
        (dict(gf_method="bicgstab", gf_admit_tol=1e-4), "only applies to gf_admission='outer'"),
        (dict(gf_method="bicgstab", gf_admission="all", gf_admit_tol=1e-4), "only applies to gf_admission='outer'"),
        (dict(gf_method="bicgstab", gf_admission="outer", gf_admit_tol=0.0), "must be positive"),
        (dict(gf_method="bicgstab", gf_admission="outer", gf_admit_tol=-1e-4), "must be positive"),
    ],
)
def test_a_combination_that_would_be_ignored_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SolverOptions(**kwargs)


# --- the argument reaches the kernel, and beats the environment -------------------------------------


def _driver(monkeypatch, env=None, **extra):
    """``(points, diagnostics)`` of one small per-frequency run: the per-point records from the
    root-side unit notes, and the names of the diagnostics the report carries."""
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    with solver_trace.tracing() as trace, contextlib.redirect_stdout(io.StringIO()):
        _mat, _real, report = _run_driver("bicgstab", None, extra=extra)
    points = [p for note in trace.of_kind("gf_unit_basis") for p in note.get("points", [])]
    return points, {d.name for d in report.diagnostics}


def test_the_default_run_has_no_admission_records_and_no_error_bound(monkeypatch):
    points, diagnostics = _driver(monkeypatch)
    assert points and all("admission" not in p for p in points)
    assert "truncation_error_bound" not in diagnostics


def test_outer_by_argument_records_each_point_and_reports_the_error_bound(monkeypatch):
    points, diagnostics = _driver(monkeypatch, gf_admission="outer", gf_admit_tol=1e-3)
    assert points and all("admission" in p for p in points)
    assert "truncation_error_bound" in diagnostics


def test_an_explicit_argument_beats_the_environment_in_both_directions(monkeypatch):
    points, _ = _driver(monkeypatch, env={"GF_BICGSTAB_ADMISSION": "all"}, gf_admission="outer")
    assert all("admission" in p for p in points)
    points, _ = _driver(monkeypatch, env={"GF_BICGSTAB_ADMISSION": "outer"}, gf_admission="all")
    assert all("admission" not in p for p in points)


def test_the_environment_alone_still_selects_outer_for_the_benchmarks_that_use_it(monkeypatch):
    points, diagnostics = _driver(monkeypatch, env={"GF_BICGSTAB_ADMISSION": "outer"})
    assert all("admission" in p for p in points) and "truncation_error_bound" in diagnostics


def test_the_tolerance_argument_beats_the_environment_threshold(monkeypatch):
    """On SIAM-6, whose start set is smaller than its closure: a threshold above every score admits nothing
    beyond the start set whatever the environment says, and the environment's own tiny one admits."""
    from impurityModel.ed.gf_solvers import block_Green_bicgstab
    from impurityModel.ed.greens_function import _gf_signed_axes
    from impurityModel.test.support.gf_oracles import DELTA, OMEGA, _seed_basis, _seeds, _siam_6

    monkeypatch.setenv("GF_BICGSTAB_ADMIT_TOL_AMP", "1e-12")
    z_axes = _gf_signed_axes(None, OMEGA[:4], 0, DELTA)

    def rounds(**kwargs):
        _G, stats = block_Green_bicgstab(
            _siam_6(), _seeds(), _seed_basis(), [0.0], 2, z_axes, atol=1e-10, admission="outer", **kwargs
        )
        return [p["admission"]["rounds"] for p in stats["points"]]

    assert all(r == 0 for r in rounds(admit_tol=1e6))
    assert any(r > 0 for r in rounds())


def test_the_environment_can_force_the_error_bound_either_way(monkeypatch):
    _, diagnostics = _driver(
        monkeypatch, env={"GF_BICGSTAB_RESIDUAL_CHECK": "0"}, gf_admission="outer", gf_admit_tol=1e-3
    )
    assert "truncation_error_bound" not in diagnostics
    _, diagnostics = _driver(monkeypatch, env={"GF_BICGSTAB_RESIDUAL_CHECK": "1"})
    assert "truncation_error_bound" in diagnostics


def test_outer_with_the_lanczos_kernel_is_refused_where_it_would_otherwise_be_ignored(monkeypatch):
    with pytest.raises(ValueError, match="needs gf_method='bicgstab'"):
        _run_driver("lanczos", "full", extra={"gf_admission": "outer"})
    with pytest.raises(ValueError, match="Unknown gf_admission"):
        _run_driver("bicgstab", None, extra={"gf_admission": "banana"})


def test_all_by_argument_is_bit_identical_to_not_passing_it(monkeypatch):
    base = _run_driver("bicgstab", None)
    explicit = _run_driver("bicgstab", None, extra={"gf_admission": "all", "gf_admit_tol": None})
    for a, b in zip(base[:2], explicit[:2]):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# --- a whole self-energy run ------------------------------------------------------------------------


def _sigma(solver, comm):
    basis, _ = _basis_and_solver()
    with contextlib.redirect_stdout(io.StringIO()):
        return calc_selfenergy(_siam_model(), _meshes(), basis, solver, comm=comm, verbosity=0)


def test_calc_selfenergy_with_outer_admission_matches_the_lanczos_run(monkeypatch):
    reference = _sigma(
        SolverOptions(reort=None, dense_cutoff=500, sparse_green=True, gf_method="lanczos"), MPI.COMM_SELF
    )
    outer = _sigma(
        SolverOptions(gf_method="bicgstab", gf_admission="outer", gf_admit_tol=1e-9, dense_cutoff=500),
        MPI.COMM_SELF,
    )
    np.testing.assert_allclose(outer["sigma"], reference["sigma"], atol=1e-5)
    np.testing.assert_allclose(outer["sigma_real"], reference["sigma_real"], atol=1e-5)


@pytest.mark.mpi
def test_calc_selfenergy_with_outer_admission_is_the_same_distributed(monkeypatch):
    solver = SolverOptions(gf_method="bicgstab", gf_admission="outer", gf_admit_tol=1e-9, dense_cutoff=500)
    serial = _sigma(solver, MPI.COMM_SELF)
    distributed = _sigma(solver, MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 0:
        np.testing.assert_allclose(distributed["sigma"], serial["sigma"], atol=1e-6)
        np.testing.assert_allclose(distributed["sigma_real"], serial["sigma_real"], atol=1e-6)
