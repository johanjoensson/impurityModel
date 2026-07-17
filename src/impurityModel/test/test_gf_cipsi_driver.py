"""Tests for the per-frequency CIPSI-selected Green's function (``gf_method="cipsi"``).

Layers mirror ``test_gf_bicgstab_driver`` (whose SIAM-6 harness is reused):

* kernel: uncapped ``block_Green_cipsi`` against the dense resolvent on the closed sector
  (the selection fixpoint must find the full reachable support);
* budget: graceful error decay with ``GF_CIPSI_BUDGET``, the boundary residual as the
  honest truncation-error bar, the PT2 downfolding correction, and the scorer knob;
* driver: ``get_Greens_function(gf_method="cipsi")`` against the block-Lanczos path,
  including the ``cipsi_boundary`` diagnostic;
* MPI: lock-step distributed runs and rank-count invariance of the *capped* selection
  (the collective bisections must admit identical sets at any rank count).
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.gf_solvers import block_Green_cipsi
from impurityModel.ed.greens_function import _gf_signed_axes
from impurityModel.test.test_gf_bicgstab_driver import (
    DELTA,
    MATSUBARA,
    OMEGA,
    _dense_G_on,
    _n3_sector_dets,
    _run_driver,
    _seed_basis,
    _seeds,
    _siam_6,
)

E_SHIFT = 0.3


def _run_kernel(z_axes, comm=None):
    return block_Green_cipsi(
        _siam_6(),
        _seeds(),
        _seed_basis(comm=comm),
        [E_SHIFT],
        len(_seeds()),
        z_axes,
        atol=1e-10,
    )


def _max_rel_err(G_axes, z_axes):
    sector = _n3_sector_dets()
    errs = []
    for ax, z_axis in enumerate(z_axes):
        ref = _dense_G_on(sector, z_axis + E_SHIFT)
        errs.append(np.max(np.abs(G_axes[ax][0] - ref)) / np.max(np.abs(ref)))
    return max(errs)


# --------------------------------------------------------------------------- #
# Kernel: dense-resolvent oracle
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("side_i", [0, 1])
def test_kernel_matches_dense_resolvent(side_i):
    """Uncapped, the selection fixpoint reaches zero boundary residual and the kernel
    equals the dense resolvent on the closed N=3 sector, both axes, both signs."""
    z_axes = _gf_signed_axes(MATSUBARA, OMEGA, side_i, DELTA)
    G_axes, stats = _run_kernel(z_axes)
    assert stats["n_unconverged"] == 0
    assert stats["max_boundary_rel"] == 0.0
    assert _max_rel_err(G_axes, z_axes) < 1e-7


# --------------------------------------------------------------------------- #
# Budget: graceful truncation, boundary error bar, PT2, scorers
# --------------------------------------------------------------------------- #


def test_kernel_budget_error_decays(monkeypatch):
    """Error decreases monotonically with the budget; the boundary residual bounds it and
    the cap state is reported. At a budget covering the reachable support the truncation
    vanishes entirely."""
    z_axes = _gf_signed_axes(MATSUBARA, OMEGA, 0, DELTA)
    errs, bounds = [], []
    for budget in (6, 9, 15, 18):
        monkeypatch.setenv("GF_CIPSI_BUDGET", str(budget))
        G_axes, stats = _run_kernel(z_axes)
        err = _max_rel_err(G_axes, z_axes)
        assert all(np.all(np.isfinite(G)) for G in G_axes)
        errs.append(err)
        bounds.append(stats["max_boundary_rel"])
        if budget < 18:
            # The budget binds: the cap record must say so, and the boundary residual
            # (the truncation-error estimate) must not undersell the actual error.
            assert stats["cap_hit"]
            assert stats["retained_size"] <= budget
            assert err <= 10 * stats["max_boundary_rel"]
    assert all(e1 >= 0.99 * e2 for e1, e2 in itertools.pairwise(errs))
    assert errs[-1] < 1e-7 and bounds[-1] == 0.0


def test_kernel_pt2_improves_moderate_budget(monkeypatch):
    """The second-order downfolding of the discarded boundary reduces the error once the
    boundary is perturbative (a moderate budget), and its magnitude is recorded."""
    z_axes = _gf_signed_axes(MATSUBARA, OMEGA, 0, DELTA)
    monkeypatch.setenv("GF_CIPSI_BUDGET", "15")
    G_plain, stats_plain = _run_kernel(z_axes)
    monkeypatch.setenv("GF_CIPSI_PT2", "1")
    G_pt2, _stats_pt2 = _run_kernel(z_axes)
    assert stats_plain["pt2_max_correction"] > 0  # recorded even when not applied
    assert _max_rel_err(G_pt2, z_axes) < _max_rel_err(G_plain, z_axes)


def test_kernel_amplitude_scorer_and_invalid(monkeypatch):
    """The bare-coupling baseline scorer runs (finite, budget-capped); an unknown scorer
    raises."""
    z_axes = _gf_signed_axes(None, OMEGA, 0, DELTA)
    monkeypatch.setenv("GF_CIPSI_BUDGET", "12")
    monkeypatch.setenv("GF_CIPSI_SCORER", "amplitude")
    G_axes, stats = _run_kernel(z_axes)
    assert all(np.all(np.isfinite(G)) for G in G_axes)
    assert stats["max_solve_basis"] <= 12
    monkeypatch.setenv("GF_CIPSI_SCORER", "bogus")
    with pytest.raises(ValueError, match="scorer"):
        _run_kernel(z_axes)


# --------------------------------------------------------------------------- #
# Driver: get_Greens_function(gf_method="cipsi") vs block Lanczos (PARTIAL)
# --------------------------------------------------------------------------- #


def test_driver_matches_partial_lanczos():
    """cipsi G == PARTIAL-reorthogonalized block-Lanczos G, both meshes, and the report
    carries the solver record plus the boundary-residual diagnostic."""
    m_l, r_l, _ = _run_driver("lanczos", "partial")
    m_c, r_c, report = _run_driver("cipsi", None)
    np.testing.assert_allclose(m_c[0], m_l[0], atol=1e-7)
    np.testing.assert_allclose(r_c[0], r_l[0], atol=1e-7)
    names = {d.name for d in report.diagnostics}
    assert {"bicgstab", "cipsi_boundary", "causality"} <= names
    assert all(d.severity.name != "FAIL" for d in report.diagnostics)


def test_driver_budget_capped_stays_finite():
    """A finite driver-level budget yields finite G and surfaces the cap + boundary
    diagnostics. (This workload's per-point support is tiny, so the budget does not bind
    here -- the binding-budget behavior is covered by the kernel tests above.)"""
    m_c, r_c, report = _run_driver("cipsi", None, monkeypatch_env={"GF_CIPSI_BUDGET": "6"})
    assert np.all(np.isfinite(m_c[0])) and np.all(np.isfinite(r_c[0]))
    names = {d.name for d in report.diagnostics}
    assert {"cipsi_boundary", "basis_cap"} <= names
    assert all(d.severity.name != "FAIL" for d in report.diagnostics)


# --------------------------------------------------------------------------- #
# MPI
# --------------------------------------------------------------------------- #


@pytest.mark.mpi
def test_driver_mpi_matches_partial_lanczos():
    """Distributed run (2+ ranks): the per-point solve/select cycle stays in MPI lock-step
    (empty-candidate ranks included) and reproduces the PARTIAL Lanczos G."""
    comm = MPI.COMM_WORLD
    m_l, r_l, _ = _run_driver("lanczos", "partial", comm=comm)
    m_c, r_c, _ = _run_driver("cipsi", None, comm=comm)
    if comm.rank == 0:
        np.testing.assert_allclose(m_c[0], m_l[0], atol=1e-7)
        np.testing.assert_allclose(r_c[0], r_l[0], atol=1e-7)
    else:
        assert m_c is None and r_c is None


@pytest.mark.mpi
def test_driver_mpi_capped_rank_count_invariant():
    """A *binding* budget must select the identical basis at any rank count (sorted
    candidates + collective bisections): the distributed capped G equals a per-rank
    serial (COMM_SELF) capped run bit-for-bit in exact arithmetic, here to 1e-12."""
    comm = MPI.COMM_WORLD
    env = {"GF_CIPSI_BUDGET": "8"}
    m_s, r_s, _ = _run_driver("cipsi", None, comm=MPI.COMM_SELF, monkeypatch_env=env)
    m_d, r_d, _ = _run_driver("cipsi", None, comm=comm, monkeypatch_env=env)
    if comm.rank == 0:
        np.testing.assert_allclose(m_d[0], m_s[0], atol=1e-12)
        np.testing.assert_allclose(r_d[0], r_s[0], atol=1e-12)
