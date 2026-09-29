"""Tests for the per-frequency BiCGSTAB Green's function (``gf_method="bicgstab"``).

Three layers, mirroring the Lanczos-path suites:

* kernel: ``block_Green_bicgstab`` against the dense resolvent on a closed sector
  (``test_greens_function``-style), including signs on both spectral sides and axes;
* cap: the ``_CappedBasisProxy`` + ``block_bicgstab`` freeze-growth interaction against the
  dense ``P H P`` resolvent on the retained set — the same strong oracle as
  ``test_gf_truncation``;
* driver: ``get_Greens_function(gf_method="bicgstab")`` against the block-Lanczos path at
  ``reort="partial"`` on a hybridizing system with a two-body term, on both meshes.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.gf_solvers import block_Green_bicgstab
from impurityModel.ed.greens_function import _gf_signed_axes
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.support.gf_oracles import (
    DELTA,
    MATSUBARA,
    OMEGA,
    _capped_solve,
    _dense_G_on,
    _n3_sector_dets,
    _run_driver,
    _seed_basis,
    _seeds,
    _siam_6,
)

# --------------------------------------------------------------------------- #
# Kernel: dense-resolvent oracle, both sides and axes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("side_i", [0, 1])
def test_kernel_matches_dense_resolvent(side_i):
    """Uncapped kernel == dense resolvent on the closed N=3 sector, both axes, both signs."""
    e_shift = 0.3
    z_axes = _gf_signed_axes(MATSUBARA, OMEGA, side_i, DELTA)
    G_axes, stats = block_Green_bicgstab(
        _siam_6(),
        _seeds(),
        _seed_basis(),
        [e_shift],
        len(_seeds()),
        z_axes,
        atol=1e-10,
    )
    assert stats["n_unconverged"] == 0
    sector = _n3_sector_dets()
    for ax, z_axis in enumerate(z_axes):
        ref = _dense_G_on(sector, z_axis + e_shift)
        np.testing.assert_allclose(G_axes[ax][0], ref, atol=1e-7 * np.max(np.abs(ref)))


def test_kernel_zero_seed_column():
    """A zero seed column (an annihilated orbital) yields zero G entries, no crash."""
    seeds = [_seeds()[0], ManyBodyState(width=1)]
    z_axes = _gf_signed_axes(MATSUBARA, None, 0, DELTA)
    G_axes, stats = block_Green_bicgstab(_siam_6(), seeds, _seed_basis(), [0.0], 2, z_axes, atol=1e-10)
    assert stats["n_unconverged"] == 0
    np.testing.assert_array_equal(G_axes[0][0][:, 1, :], 0.0)
    np.testing.assert_array_equal(G_axes[0][0][:, :, 1], 0.0)
    assert np.max(np.abs(G_axes[0][0][:, 0, 0])) > 0


def test_kernel_gmres_fallback_rescues_failed_points():
    """With BiCGSTAB disabled (max_iter=0, every point 'fails'), the GMRES fallback must
    solve every point on its own: G still equals the dense resolvent, no point is left
    unconverged, and the stats record the fallback. (The default GF_GMRES_RESTART of 40
    exceeds this sector's dimension, so the rescue is a guaranteed full-GMRES solve.)"""
    e_shift = 0.3
    z_axes = _gf_signed_axes(MATSUBARA, OMEGA, 0, DELTA)
    G_axes, stats = block_Green_bicgstab(
        _siam_6(),
        _seeds(),
        _seed_basis(),
        [e_shift],
        len(_seeds()),
        z_axes,
        atol=1e-10,
        max_iter=0,
    )
    assert stats["gmres_points"] == stats["n_points"]
    assert stats["gmres_iterations"] > 0
    assert stats["n_unconverged"] == 0
    sector = _n3_sector_dets()
    for ax, z_axis in enumerate(z_axes):
        ref = _dense_G_on(sector, z_axis + e_shift)
        np.testing.assert_allclose(G_axes[ax][0], ref, atol=1e-7 * np.max(np.abs(ref)))


def test_kernel_bounded_by_cap():
    """A finite truncation_threshold bounds every per-point basis; the result stays causal."""
    cap = 12
    z_axes = _gf_signed_axes(None, OMEGA, 0, DELTA)
    G_axes, stats = block_Green_bicgstab(_siam_6(), _seeds(), _seed_basis(cap=cap), [0.0], 2, z_axes, atol=1e-10)
    assert stats["cap_hit"]
    assert not stats["seed_overflow"]
    assert stats["max_solve_basis"] <= cap
    assert stats["retained_size"] <= cap
    # retarded axis: Im G_ii <= 0 even under truncation (exact on the retained subspace)
    assert np.all(np.diagonal(G_axes[0][0].imag, axis1=1, axis2=2) <= 1e-12)


# --------------------------------------------------------------------------- #
# Cap: the P H P oracle at the block_bicgstab + _CappedBasisProxy level
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cap", [6, 12, 17])
def test_capped_solve_equals_dense_php_resolvent(cap):
    """The strong oracle: a capped solve is the exact resolvent of P H P on the retained set."""
    z = 1.7 + 1j * DELTA
    G, proxy = _capped_solve(cap, z)
    assert proxy.cap_hit and proxy.retained_size <= cap
    retained = proxy.retained_keys()
    assert len(retained) == proxy.retained_size
    ref = _dense_G_on(retained, [z])[0]
    np.testing.assert_allclose(G, ref, atol=1e-9)


def test_uncapped_solve_matches_dense_full_sector():
    """Sanity: with the cap above the reachable space the solve is exact on the sector."""
    z = -2.4 + 1j * DELTA
    G, proxy = _capped_solve(1000, z)
    assert not proxy.cap_hit
    ref = _dense_G_on(_n3_sector_dets(), [z])[0]
    np.testing.assert_allclose(G, ref, atol=1e-9)


# --------------------------------------------------------------------------- #
# Driver: get_Greens_function(gf_method="bicgstab") vs block Lanczos (PARTIAL)
# --------------------------------------------------------------------------- #


def test_driver_matches_partial_lanczos():
    """bicgstab G == PARTIAL-reorthogonalized block-Lanczos G, both meshes."""
    m_l, r_l, _ = _run_driver("lanczos", "partial")
    m_b, r_b, report = _run_driver("bicgstab", None)
    np.testing.assert_allclose(m_b[0], m_l[0], atol=1e-7)
    np.testing.assert_allclose(r_b[0], r_l[0], atol=1e-7)
    # the report carries the solver record and the representation-independent checks
    names = {d.name for d in report.diagnostics}
    assert "bicgstab" in names and "causality" in names
    assert all(d.severity.name != "FAIL" for d in report.diagnostics)


def test_driver_grouping_and_operator_split_invariance():
    """Eigenstate grouping is a unit-shape knob only, and the operator-split env (a Lanczos
    decomposition) is ignored on the bicgstab path -- all give the identical G."""
    _, r_ref, _ = _run_driver("bicgstab", None)
    _, r_grouped, _ = _run_driver("bicgstab", None, monkeypatch_env={"GF_EIGENSTATE_GROUP": "2"})
    _, r_split, _ = _run_driver("bicgstab", None, monkeypatch_env={"GF_OPERATOR_SPLIT": "1"})
    np.testing.assert_allclose(r_grouped[0], r_ref[0], atol=1e-9)
    np.testing.assert_allclose(r_split[0], r_ref[0], atol=1e-9)


def test_driver_rejects_unknown_method():
    with pytest.raises(ValueError, match="gf_method"):
        _run_driver("haydock", None)


@pytest.mark.parametrize("retired", sorted(config.RETIRED_GF_METHODS))
def test_driver_rejects_a_retired_method_with_its_reason(retired):
    """A retired kernel is not merely unknown: the error says it was retired, and why."""
    with pytest.raises(ValueError, match=f"'{retired}' was retired"):
        _run_driver(retired, None)


# --------------------------------------------------------------------------- #
# MPI
# --------------------------------------------------------------------------- #


@pytest.mark.mpi
def test_driver_mpi_matches_partial_lanczos():
    """Distributed run (2+ ranks): the per-point rebuild/redistribute/solve cycle stays in
    MPI lock-step (empty ranks included) and reproduces the PARTIAL Lanczos G."""
    comm = MPI.COMM_WORLD
    m_l, r_l, _ = _run_driver("lanczos", "partial", comm=comm)
    m_b, r_b, _ = _run_driver("bicgstab", None, comm=comm)
    if comm.rank == 0:
        np.testing.assert_allclose(m_b[0], m_l[0], atol=1e-7)
        np.testing.assert_allclose(r_b[0], r_l[0], atol=1e-7)
    else:
        assert m_b is None and r_b is None


@pytest.mark.mpi
def test_capped_solve_mpi_matches_dense_php():
    """Distributed capped solve: collective cap decisions consistent, P H P oracle holds."""
    comm = MPI.COMM_WORLD
    z = 1.7 + 1j * DELTA
    for cap in (6, 12):
        G, proxy = _capped_solve(cap, z, comm=comm)
        assert proxy.cap_hit and proxy.retained_size <= cap
        gathered = comm.allgather(proxy.retained_keys())
        retained = sorted({k for part in gathered for k in part})
        assert len(retained) == proxy.retained_size
        ref = _dense_G_on(retained, [z])[0]
        np.testing.assert_allclose(G, ref, atol=1e-9)


# ---- seed overflow: the right-hand side alone does not fit the budget -------------------------
#
# Both paths below had never run: every capped test left room for the seeds. The contract is
# that the seeds are never truncated -- a budget below their support is solved frozen on the
# support, exactly, and flagged -- while warm-start-only support is what gets cut.


def test_a_cap_below_the_seed_support_solves_frozen_on_it_and_says_so():
    seed_support = sorted({state for s in _seeds() for state in s})
    cap = len(seed_support) - 1
    e_shift = 0.3
    z_axes = _gf_signed_axes(None, OMEGA, 0, DELTA)
    G_axes, stats = block_Green_bicgstab(_siam_6(), _seeds(), _seed_basis(cap=cap), [e_shift], 2, z_axes, atol=1e-10)
    assert stats["seed_overflow"]
    assert stats["n_unconverged"] == 0
    # Frozen on the seed support: the exact resolvent of P H P there, at every point.
    ref = _dense_G_on(seed_support, z_axes[0] + e_shift)
    np.testing.assert_allclose(G_axes[0][0], ref, atol=1e-8 * np.max(np.abs(ref)))
