"""Tests for the truncation_threshold memory sizing helpers."""

import math

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import memory_estimate as me
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant


def _make_state(n_spin_orbitals, n_dets):
    """A ManyBodyState with n_dets single-bit determinants of the right chunk width."""
    n_chunks = math.ceil(n_spin_orbitals / 64)
    dets = {}
    for i in range(n_dets):
        chunks = [0] * n_chunks
        chunks[i // 64] = 1 << (63 - (i % 64))
        dets[SlaterDeterminant(tuple(chunks))] = 1.0 + 0.5j
    return ManyBodyState(dets)


@pytest.mark.parametrize("n_spin_orbitals", [16, 60, 100, 160, 250])
def test_bytes_per_determinant_matches_cython(n_spin_orbitals):
    """The Python formula must mirror ManyBodyState.memory_bytes exactly."""
    n_dets = min(8, n_spin_orbitals)
    ms = _make_state(n_spin_orbitals, n_dets)
    assert ms.memory_bytes() == n_dets * me.bytes_per_determinant(n_spin_orbitals)


def test_estimates_scale_linearly_at_reort_none():
    a = me.estimate_gf_peak_bytes(10_000, 100, block_width=10, reort="none")
    b = me.estimate_gf_peak_bytes(20_000, 100, block_width=10, reort="none")
    assert b == 2 * a


def test_gf_reort_retention_costs_more():
    none = me.estimate_gf_peak_bytes(10_000, 100, block_width=10, reort="none")
    full = me.estimate_gf_peak_bytes(10_000, 100, block_width=10, reort="full")
    assert full > none


def test_gf_ranks_reduce_per_rank_cost():
    one = me.estimate_gf_peak_bytes(10_000, 100, block_width=10, reort="none", ranks=1)
    four = me.estimate_gf_peak_bytes(10_000, 100, block_width=10, reort="none", ranks=4)
    assert four < one
    assert four >= one // 4


def test_gs_array_kernel_replication_does_not_shrink_with_ranks():
    """The replicated (global_N, width) term must persist at high rank counts."""
    many_ranks = me.estimate_gs_peak_bytes(100_000, 100, block_width=4, ranks=1024)
    assert many_ranks >= 100_000 * 4 * 16


def test_suggest_threshold_monotone_in_safety():
    lo = me.suggest_truncation_threshold(100, safety=0.1)
    hi = me.suggest_truncation_threshold(100, safety=0.5)
    assert 0 < lo <= hi


def test_suggest_threshold_fits_budget():
    n = me.suggest_truncation_threshold(100, block_width=4, reort="none", safety=0.25)
    budget = 0.25 * me.available_bytes_per_rank(None)
    assert me.estimate_gs_peak_bytes(n, 100, 4, 1) <= budget
    assert me.estimate_gs_peak_bytes(n + 1, 100, 4, 1) > budget or n >= 10**13


def test_available_bytes_serial_positive():
    assert me.available_bytes_per_rank(None) > 0


def test_log_memory_budget_serial(capsys):
    report = me.log_memory_budget(100_000, 100, comm=None, block_width=4, verbose=True, label="test")
    out = capsys.readouterr().out
    assert "truncation_threshold=100,000" in out
    assert report["available_per_rank"] > 0
    assert report["gs_peak"] > 0 and report["gf_peak"] > 0


def test_log_memory_budget_uncapped(capsys):
    for uncapped in (None, np.inf, float("inf")):
        report = me.log_memory_budget(uncapped, 100, comm=None, verbose=True)
        assert report["gs_peak"] is None and report["gf_peak"] is None
    assert "uncapped" in capsys.readouterr().out


def test_log_memory_budget_warns_when_too_big(capsys):
    report = me.log_memory_budget(10**12, 60, comm=None, block_width=4, verbose=True)
    assert not report["fits"]
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.mpi
def test_available_bytes_per_rank_mpi_consistent():
    """Collective probe: every rank gets the same positive budget."""
    comm = MPI.COMM_WORLD
    per_rank = me.available_bytes_per_rank(comm)
    assert per_rank > 0
    gathered = comm.allgather(per_rank)
    assert all(g == gathered[0] for g in gathered)
    if comm.size > 1:
        assert per_rank <= me._node_available_bytes()


@pytest.mark.mpi
def test_log_memory_budget_mpi_verbose_rank_local():
    """Per-rank verbose flags must be safe: collectives run unconditionally."""
    comm = MPI.COMM_WORLD
    # Drivers commonly set verbosity only on rank 0 -- this must not deadlock.
    report = me.log_memory_budget(10_000, 100, comm=comm, verbose=comm.rank == 0)
    assert report["available_per_rank"] > 0
