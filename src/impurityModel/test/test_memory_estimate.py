"""Tests for the truncation_threshold memory sizing helpers."""

import math

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import memory_estimate as me
from impurityModel.ed.groundstate import find_ground_state_basis
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant


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


def _write_cgroup_v2(tmp_path, entries):
    """Build a fake cgroup v2 tree; entries = {relpath: (max, current)} with str/int values."""
    root = tmp_path / "cgroup"
    for rel, (limit, current) in entries.items():
        d = root / rel if rel else root
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.max").write_text(f"{limit}\n")
        (d / "memory.current").write_text(f"{current}\n")
    return root


def test_cgroup_v2_tightest_ancestor_headroom(tmp_path):
    """min(limit - current) over the ancestor chain wins, root without files skipped."""
    root = _write_cgroup_v2(
        tmp_path,
        {
            "slurm/job_1": (8 * 2**30, 2**30),  # 7 GiB headroom
            "slurm/job_1/step_0": (16 * 2**30, 0),  # looser child
        },
    )
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/slurm/job_1/step_0\n")
    headroom = me._cgroup_available_bytes(proc_path=str(proc), v2_root=str(root))
    assert headroom == 7 * 2**30


def test_cgroup_v2_unlimited_returns_none(tmp_path):
    root = _write_cgroup_v2(tmp_path, {"user": ("max", 12345)})
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/user\n")
    assert me._cgroup_available_bytes(proc_path=str(proc), v2_root=str(root)) is None


def test_cgroup_v1_limit_and_huge_means_unlimited(tmp_path):
    v1 = tmp_path / "memory" / "slurm" / "job_2"
    v1.mkdir(parents=True)
    proc = tmp_path / "proc_cgroup"
    proc.write_text("3:cpu:/ignored\n2:memory:/slurm/job_2\n")
    (v1 / "memory.limit_in_bytes").write_text(f"{2**63 - 4096}\n")
    (v1 / "memory.usage_in_bytes").write_text("0\n")
    assert me._cgroup_available_bytes(proc_path=str(proc), v1_root=str(tmp_path / "memory")) is None
    (v1 / "memory.limit_in_bytes").write_text(f"{4 * 2**30}\n")
    (v1 / "memory.usage_in_bytes").write_text(f"{2**30}\n")
    headroom = me._cgroup_available_bytes(proc_path=str(proc), v1_root=str(tmp_path / "memory"))
    assert headroom == 3 * 2**30


def test_cgroup_missing_proc_file_returns_none(tmp_path):
    assert me._cgroup_available_bytes(proc_path=str(tmp_path / "nope")) is None


def test_node_available_bytes_respects_cgroup(monkeypatch):
    """A binding cgroup limit must cap the node availability figure."""
    unconstrained = me._node_available_bytes()
    monkeypatch.setattr(me, "_cgroup_available_bytes", lambda **kw: 12345)
    assert me._node_available_bytes() == 12345
    monkeypatch.setattr(me, "_cgroup_available_bytes", lambda **kw: None)
    assert me._node_available_bytes() >= min(unconstrained, 12345)


def test_max_colors_within_budget(monkeypatch):
    """The color cap must invert estimate_gf_peak_bytes against the safety-scaled budget."""
    from types import SimpleNamespace

    comm = SimpleNamespace(size=16)
    n, nso, width = 100_000, 100, 4
    target = me.estimate_gf_peak_bytes(n, nso, width, "none", ranks=16 // 4)
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: target / me.DEFAULT_MEMORY_SAFETY)
    assert me.max_colors_within_budget(n, nso, width, "none", comm, 16) == 4
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 1)
    assert me.max_colors_within_budget(n, nso, width, "none", comm, 16) == 1
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 2**60)
    assert me.max_colors_within_budget(n, nso, width, "none", comm, 16) == 16


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


def test_log_peak_vs_predicted_serial(capsys):
    budget = me.log_memory_budget(100_000, 100, comm=None, verbose=False)
    measured = me.log_peak_vs_predicted(budget, comm=None, verbose=True, label="test")
    out = capsys.readouterr().out
    assert measured > 0
    assert "measured per-rank peak RSS" in out and "predicted" in out


def test_log_peak_vs_predicted_uncapped(capsys):
    budget = me.log_memory_budget(np.inf, 100, comm=None, verbose=False)
    me.log_peak_vs_predicted(budget, comm=None, verbose=True)
    assert "predicted uncapped" in capsys.readouterr().out


@pytest.mark.mpi
def test_log_peak_vs_predicted_mpi_rank_local_verbose():
    """Collectives must run unconditionally under per-rank verbose flags."""
    comm = MPI.COMM_WORLD
    budget = me.log_memory_budget(10_000, 100, comm=comm, verbose=False)
    measured = me.log_peak_vs_predicted(budget, comm=comm, verbose=comm.rank == 0)
    assert measured >= me.peak_rss_bytes()


def test_log_memory_budget_warns_when_too_big(capsys):
    report = me.log_memory_budget(10**12, 60, comm=None, block_width=4, verbose=True)
    assert not report["fits"]
    assert "WARNING" in capsys.readouterr().out


def _siam_6_pieces():
    """Single-impurity Anderson model, 6 spin-orbitals (see test_sectorization)."""
    ed_, u, ev, ec, v = -1.0, 4.0, -3.0, 3.0, 0.5
    terms = {}
    for o in (0, 1):
        terms[((o, "c"), (o, "a"))] = ed_
    for o in (2, 3):
        terms[((o, "c"), (o, "a"))] = ev
    for o in (4, 5):
        terms[((o, "c"), (o, "a"))] = ec
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = u
    for a, b in ((0, 2), (1, 3), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = v
        terms[((b, "c"), (a, "a"))] = v
    return terms, {0: [[0, 1]]}, ({0: [[2, 3]]}, {0: [[4, 5]]})


def test_basis_normalizes_none_threshold_to_inf():
    _, impurity_orbitals, bath_states = _siam_6_pieces()
    basis = Basis(
        impurity_orbitals,
        bath_states,
        nominal_impurity_occ={0: 2},
        truncation_threshold=None,
        verbose=False,
    )
    assert basis.truncation_threshold == np.inf


def test_find_ground_state_basis_resolves_none_threshold():
    """The default truncation_threshold=None must resolve to a finite RAM-derived cap."""
    terms, impurity_orbitals, bath_states = _siam_6_pieces()
    basis = find_ground_state_basis(
        ManyBodyOperator(terms),
        impurity_orbitals,
        bath_states,
        N0={0: 2},
        tau=0.01,
        dense_cutoff=1000,
        comm=None,
        verbose=False,
    )
    assert np.isfinite(basis.truncation_threshold)
    assert basis.truncation_threshold >= 1


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


def test_krylov_dtype_halves_the_store_term_and_raises_the_cap():
    """complex64 storage must reach the sizing model, or the 2x never reaches the user.

    The Krylov store is the term that forces ``truncation_threshold`` down when
    reorthogonalization is on, so halving it must raise the suggested cap
    (see ``doc/plans/blocklanczos_reort_memory.md``).
    """
    kw = dict(n_dets=100_000, n_spin_orbitals=106, block_width=2, reort="full", ranks=4, n_blocks=400)
    wide = me.estimate_gf_peak_bytes(**kw)
    narrow = me.estimate_gf_peak_bytes(**kw, krylov_dtype=np.complex64)
    assert narrow < wide
    # Everything but the store is dtype independent, so the saving is exactly half of it.
    store_only = me.estimate_gf_peak_bytes(**{**kw, "reort": "none"})
    assert wide - narrow == pytest.approx((wide - store_only) * 0.5, rel=0.02)

    budget = 8 * 2**30
    lo = me._suggest_for_budget(budget, 106, 2, "full", 1, 100, 4)
    hi = me._suggest_for_budget(budget, 106, 2, "full", 1, 100, 4, np.complex64)
    assert hi > lo


@pytest.mark.parametrize("reort", ["partial", "selective"])
def test_krylov_dtype_complex64_rejected_for_estimator_modes(reort):
    """The model must refuse a combination the kernel refuses to run."""
    with pytest.raises(ValueError, match="incompatible with reort"):
        me.estimate_gf_peak_bytes(1000, 106, 2, reort=reort, krylov_dtype=np.complex64)
