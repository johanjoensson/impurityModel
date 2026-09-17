"""Tests for the truncation_threshold memory sizing helpers."""

import math
from itertools import pairwise

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


def test_gf_reort_none_per_det_matches_measured_slope(monkeypatch):
    """reort=none per-det stays near the VmHWM-calibrated ~550 B/det (width 1), once the
    round-8 matvec-fanout term (:data:`me._GF_MATVEC_ROW_FANOUT_DEFAULT`, divided by the
    chunking credit) is subtracted back out.

    Guards the ``s_live`` constant against a wild miscalibration: a prior recalibration put it
    at ~1.4 kB/det (3x the measured slope). See doc/plans/truncation_reliability.md. The fanout
    term is a real, separately-derived addition (doc/plans/dc_smo_memory.md, round 8) and is
    subtracted here rather than folded into a wider bound, so a future miscalibration of
    *this* constant still trips this test.
    """
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", "4")
    row_bytes = 16 * 1 + me._key_heap_bytes(124) + me._SD_STRUCT_BYTES
    fanout_per_det = me._GF_MATVEC_ROW_FANOUT_DEFAULT / me._gf_chunk_divisor(4) * row_bytes
    total = me.estimate_gf_peak_bytes(100_000, 124, block_width=1, reort="none")
    per_det = total / 100_000 - fanout_per_det
    assert 450 <= per_det <= 700, per_det


def test_gf_ranks_reduce_per_rank_cost():
    one = me.estimate_gf_peak_bytes(10_000, 100, block_width=10, reort="none", ranks=1)
    four = me.estimate_gf_peak_bytes(10_000, 100, block_width=10, reort="none", ranks=4)
    assert four < one
    assert four >= one // 4


def test_gs_array_kernel_replication_shrinks_with_ranks():
    """Post-Phase-1 (row-chunked reduce-scatter, doc/plans/dc_smo_performance.md), the array
    kernel's matvec transient is bounded by ``max(counts) ~ local``, not ``global_N`` -- the
    inverse of this test's pre-fix name and assertion, which locked in the very replication bug
    the fix removes. A few ranks should already beat the un-chunked whole-basis bound; many
    ranks should shrink further still, not merely fail to grow.
    """
    few_ranks = me.estimate_gs_peak_bytes(100_000, 100, block_width=4, ranks=4)
    more_ranks = me.estimate_gs_peak_bytes(100_000, 100, block_width=4, ranks=64)
    many_ranks = me.estimate_gs_peak_bytes(100_000, 100, block_width=4, ranks=1024)
    assert many_ranks < more_ranks < few_ranks
    # At 1024 ranks (local ~ 98 determinants) the whole estimate -- not just the replicated
    # term -- should be far below the pre-Phase-1 un-chunked bound on the replicated term alone.
    assert many_ranks < 100_000 * 4 * 16


def test_gs_graph_exchange_term_is_bounded_by_the_byte_budget(monkeypatch):
    """Under ``graph`` the exchange buffers are sized by the neighbour count and capped by
    ``GS_MATVEC_EXCHANGE_BYTES`` (both alive at once, so 2x); ``reduce`` keeps the single
    chunk buffer. A tiny budget must therefore remove the difference between the two modes."""
    monkeypatch.delenv("GS_MATVEC_EXCHANGE_BYTES", raising=False)
    monkeypatch.setenv("GS_MATVEC_EXCHANGE", "reduce")
    loop = me.estimate_gs_peak_bytes(1_000_000, 100, block_width=110, ranks=256)
    monkeypatch.setenv("GS_MATVEC_EXCHANGE", "graph")
    graph = me.estimate_gs_peak_bytes(1_000_000, 100, block_width=110, ranks=256)
    local = -(-1_000_000 // 256)
    # 37 neighbours x local x 110 x 16 B ~ 254 MiB per buffer, well over the 64 MiB default cap.
    assert graph - loop == 2 * me.config.GS_MATVEC_EXCHANGE_BYTES.default - local * 110 * 16
    monkeypatch.setenv("GS_MATVEC_EXCHANGE_BYTES", "16")
    assert me.estimate_gs_peak_bytes(1_000_000, 100, block_width=110, ranks=256) < loop
    # Below the cap the term is degree-proportional: two ranks have one neighbour each.
    monkeypatch.delenv("GS_MATVEC_EXCHANGE_BYTES", raising=False)
    monkeypatch.setenv("GS_MATVEC_EXCHANGE", "reduce")
    loop2 = me.estimate_gs_peak_bytes(1000, 100, block_width=4, ranks=2)
    monkeypatch.setenv("GS_MATVEC_EXCHANGE", "graph")
    assert me.estimate_gs_peak_bytes(1000, 100, block_width=4, ranks=2) - loop2 == 500 * 4 * 16


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
    # Sample the reference *before* the call. VmHWM is a high-water mark, so it only ever grows:
    # comparing the MAX-allreduce taken inside log_peak_vs_predicted against a reading taken
    # after it raced the rank-0 print's own allocations and failed by two pages (143147008 vs
    # 143155200) roughly one run in three at -n 3. The property under test is that the collective
    # ran on this rank and returned at least this rank's own peak, which a before-reading pins
    # without racing the process against itself.
    before = me.peak_rss_bytes()
    measured = me.log_peak_vs_predicted(budget, comm=comm, verbose=comm.rank == 0)
    assert measured >= before


def test_log_memory_budget_warns_when_too_big(capsys):
    report = me.log_memory_budget(10**12, 60, comm=None, block_width=4, verbose=True)
    assert not report["fits"]
    assert "WARNING" in capsys.readouterr().out


def test_resolve_gs_block_width_uses_default_when_the_knob_is_unset(monkeypatch):
    monkeypatch.delenv("GS_MAX_BLOCK_WIDTH", raising=False)
    assert me.resolve_gs_block_width() == 4
    assert me.resolve_gs_block_width(default=7) == 7


def test_resolve_gs_block_width_uses_the_configured_cap_when_set(monkeypatch):
    monkeypatch.setenv("GS_MAX_BLOCK_WIDTH", "12")
    assert me.resolve_gs_block_width() == 12
    assert me.resolve_gs_block_width(default=7) == 12


def test_log_memory_budget_warns_when_gs_max_block_width_is_unset(capsys, monkeypatch):
    monkeypatch.delenv("GS_MAX_BLOCK_WIDTH", raising=False)
    me.log_memory_budget(100_000, 100, comm=None, block_width=4, verbose=True, label="test")
    assert "GS_MAX_BLOCK_WIDTH is unset" in capsys.readouterr().out


def test_log_memory_budget_warns_when_gs_num_wanted_not_supplied(capsys, monkeypatch):
    """Knob set but no measured num_wanted is the dangerous configuration: the Krylov term
    silently keeps the pre-Phase-4 num_wanted~=2*block_width assumption."""
    monkeypatch.setenv("GS_MAX_BLOCK_WIDTH", "8")
    me.log_memory_budget(100_000, 100, comm=None, block_width=4, verbose=True, label="test")
    out = capsys.readouterr().out
    assert "gs_num_wanted was not supplied" in out
    assert "GS_MAX_BLOCK_WIDTH is unset" not in out


def test_log_memory_budget_is_quiet_when_gs_num_wanted_is_supplied(capsys, monkeypatch):
    monkeypatch.setenv("GS_MAX_BLOCK_WIDTH", "8")
    me.log_memory_budget(100_000, 100, comm=None, block_width=4, verbose=True, label="test", gs_num_wanted=20)
    out = capsys.readouterr().out
    assert "GS_MAX_BLOCK_WIDTH is unset" not in out
    assert "gs_num_wanted was not supplied" not in out


def test_log_memory_budget_does_not_warn_when_uncapped(capsys, monkeypatch):
    """No estimate_gs_peak_bytes call happens on the uncapped path, so there is no "block_width
    used above" to warn about -- neither warning must fire there (review finding)."""
    monkeypatch.delenv("GS_MAX_BLOCK_WIDTH", raising=False)
    me.log_memory_budget(None, 100, comm=None, block_width=4, verbose=True, label="test")
    out = capsys.readouterr().out
    assert "GS_MAX_BLOCK_WIDTH is unset" not in out
    assert "gs_num_wanted" not in out


def test_resolve_sizing_block_width_matches_gf_width_when_the_knob_is_unset(monkeypatch):
    """Preserves today's behaviour exactly on the unset path (review finding)."""
    monkeypatch.delenv("GS_MAX_BLOCK_WIDTH", raising=False)
    assert me.resolve_sizing_block_width(6) == 6


def test_resolve_sizing_block_width_takes_the_larger_of_gf_and_gs_widths(monkeypatch):
    monkeypatch.setenv("GS_MAX_BLOCK_WIDTH", "20")
    assert me.resolve_sizing_block_width(6) == 20
    monkeypatch.setenv("GS_MAX_BLOCK_WIDTH", "2")
    assert me.resolve_sizing_block_width(6) == 6


def test_gs_num_wanted_none_keeps_the_pre_phase4_coupled_default():
    """gs_num_wanted=None (the default, whether or not GS_MAX_BLOCK_WIDTH is set) must be a
    total no-op: byte-identical to bceaef5's pre-this-fix behaviour. A review round found that
    substituting n_dets as a "worst case" for an unmeasured num_wanted makes the Krylov term
    scale like n_dets^2 (_gs_krylov_columns's min(...,n_dets) clamp binds and blocks~n_dets/p),
    inverting the whole point of GS_MAX_BLOCK_WIDTH -- reverted; see
    doc/plans/dc_smo_performance.md. The honest fix is a measured value, like nnz_per_state."""
    n, nso, p = 200_000, 106, 4
    baseline = me.estimate_gs_peak_bytes(n, nso, block_width=p)
    assert me.estimate_gs_peak_bytes(n, nso, block_width=p, num_wanted=None) == baseline


def test_gs_num_wanted_when_supplied_changes_the_krylov_term():
    n, nso, p = 200_000, 106, 4
    coupled = me.estimate_gs_peak_bytes(n, nso, block_width=p)
    measured = me.estimate_gs_peak_bytes(n, nso, block_width=p, num_wanted=250)
    assert measured != coupled


def test_suggest_truncation_threshold_gs_num_wanted_reaches_estimate_gs_peak_bytes():
    """gs_num_wanted must actually thread through _suggest_for_budget down to
    estimate_gs_peak_bytes, not just sit unused on the outer signature."""
    nso, p = 106, 4
    budget = 5 * me.estimate_gs_peak_bytes(50_000, nso, block_width=p, num_wanted=2 * p)
    with_default = me._suggest_for_budget(budget, nso, p, "none", 1, 100, ranks=1)
    with_measured = me._suggest_for_budget(budget, nso, p, "none", 1, 100, ranks=1, gs_num_wanted=400)
    assert with_measured != with_default


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


# ---------------------------------------------------------------------------------------
# routing_hash skew (doc/plans/dc_smo_memory.md, "GF unit memory")
# ---------------------------------------------------------------------------------------


def test_routing_skew_factor_matches_measured_anchors():
    """The measured points themselves must come back exactly, not just interpolate near them."""
    for ranks, skew in me._ROUTING_SKEW_ANCHORS:
        assert me._routing_skew_factor(ranks) == pytest.approx(skew)


def test_routing_skew_factor_is_one_for_a_single_rank():
    """ranks<=1 is not a partition at all; the ratio is exactly 1 by definition."""
    assert me._routing_skew_factor(1) == 1.0
    assert me._routing_skew_factor(0) == 1.0


def test_routing_skew_factor_clamps_outside_the_measured_range():
    """Outside [2, 256] the factor clamps to the nearest anchor rather than extrapolating."""
    assert me._routing_skew_factor(1000) == me._routing_skew_factor(256)


def test_routing_skew_factor_is_monotone_increasing_in_ranks():
    """Skew grows with rank count (doc/plans/dc_smo_memory.md); interpolation must not reverse
    that between anchors."""
    ranks = [2, 3, 4, 6, 8, 12, 16, 32, 64, 96, 128, 200, 256]
    skews = [me._routing_skew_factor(r) for r in ranks]
    assert all(b >= a for a, b in pairwise(skews)), skews


def test_estimate_gf_peak_bytes_scales_local_rows_by_the_skew(monkeypatch):
    """Peak bytes at a given `ranks` must equal the unskewed estimate scaled by the same
    factor `max_colors_within_budget`/`max_unit_dets_within_budget` see -- otherwise the two
    inversions and the direct estimate would disagree on what they are budgeting.

    This is also the test that pins the round-8 matvec-fanout term itself: it asserts the exact
    byte formula, so removing the term, or changing its constant or its chunking credit without
    updating this, fails here. (`test_round8_smo_crash_geometry_is_refused` does NOT pin it --
    that geometry is refused with or without the term.)"""
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", "4")
    n, nso, width = 100_000, 100, 4
    for ranks in (2, 5, 16, 64):
        skew = me._routing_skew_factor(ranks)
        unskewed_local_rows = math.ceil(n / ranks)
        skewed_local_rows = math.ceil(n / ranks * skew)
        row_bytes = 16 * width + me._key_heap_bytes(nso) + me._SD_STRUCT_BYTES
        expected = skewed_local_rows * (me.bytes_per_determinant(nso) + me._PY_BASIS_OVERHEAD_BYTES)
        expected += 3 * skewed_local_rows * row_bytes
        expected += (
            math.ceil(skewed_local_rows * me._GF_MATVEC_ROW_FANOUT_DEFAULT / me._gf_chunk_divisor(4)) * row_bytes
        )
        got = me.estimate_gf_peak_bytes(n, nso, width, "none", ranks=ranks)
        assert got == expected, (ranks, skew, unskewed_local_rows, skewed_local_rows)


def test_gf_chunk_divisor_credits_chunking_but_never_the_full_chunk_count(monkeypatch):
    """The chunked apply bounds the fanout transient, but by less than the chunk count.

    Both of this round's earlier positions were wrong and unmeasured: that chunking saves
    nothing here (shipped first), and that it saves the full chunk count. Measured on the real
    SrMnO3 archive it is ~1.9x at the default of 4 chunks. Guards the direction and the bound.
    """
    assert me._gf_chunk_divisor(1) == 1.0
    assert me._gf_chunk_divisor(None) == 1.0
    for n_chunks in (2, 4, 8):
        d = me._gf_chunk_divisor(n_chunks)
        assert 1.0 < d < n_chunks, (n_chunks, d)
    # Monotone in the chunk count, and clamped (not extrapolated) beyond the measured range.
    ds = [me._gf_chunk_divisor(c) for c in (1, 2, 4, 8)]
    assert all(a < b for a, b in pairwise(ds)), ds
    assert me._gf_chunk_divisor(64) == me._gf_chunk_divisor(8)

    # The knob is read, so the one-shot escape hatch is priced without the credit.
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", "1")
    one_shot = me.estimate_gf_peak_bytes(100_000, 58, 1, "none", ranks=4)
    monkeypatch.setenv("GF_APPLY_ROW_CHUNKS", "4")
    chunked = me.estimate_gf_peak_bytes(100_000, 58, 1, "none", ranks=4)
    assert one_shot > chunked, (one_shot, chunked)


# ---------------------------------------------------------------------------------------
# max_unit_dets_within_budget: the complement of max_colors_within_budget
# ---------------------------------------------------------------------------------------


def test_max_unit_dets_within_budget_inverts_estimate_gf_peak_bytes(monkeypatch):
    """The returned cap must fit the budget, and one more determinant must not."""
    from types import SimpleNamespace

    comm = SimpleNamespace(size=5)
    nso, width, ranks = 100, 4, 5
    target_n = 20_000
    budget = me.estimate_gf_peak_bytes(target_n, nso, width, "none", ranks=ranks)
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: budget / me.DEFAULT_MEMORY_SAFETY)

    cap = me.max_unit_dets_within_budget(nso, width, "none", ranks, comm)
    assert me.estimate_gf_peak_bytes(cap, nso, width, "none", ranks=ranks) <= budget
    assert me.estimate_gf_peak_bytes(cap + 1, nso, width, "none", ranks=ranks) > budget
    assert cap == pytest.approx(target_n, rel=0.01)


def test_max_unit_dets_within_budget_grows_with_ranks(monkeypatch):
    """More ranks sharing a unit basis must afford a larger cap under the same node budget."""
    from types import SimpleNamespace

    comm = SimpleNamespace(size=64)
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 1 * 2**30)
    nso, width = 100, 4
    few = me.max_unit_dets_within_budget(nso, width, "none", 2, comm)
    many = me.max_unit_dets_within_budget(nso, width, "none", 32, comm)
    assert many > few


def test_max_unit_dets_within_budget_floor_is_one(monkeypatch):
    """An unaffordable budget still returns a usable (if pessimistic) positive cap."""
    from types import SimpleNamespace

    comm = SimpleNamespace(size=4)
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 1)
    assert me.max_unit_dets_within_budget(100, 4, "none", 4, comm) >= 1


def test_max_unit_dets_without_residency_is_structurally_a_no_op(monkeypatch):
    """Given the SAME budget, the per-unit cap can never tighten a cap the color inversion
    already approved -- and this test exists to say that out loud rather than dress it up.

    `max_colors_within_budget` returns `n_colors >= 2` only from inside its loop, i.e. having
    verified the cap fits at that color's rank count; the split can only reduce the color
    count, which only raises `ranks`; `estimate_gf_peak_bytes` is monotone non-increasing in
    `ranks`. So `unit_cap >= cap` identically. The first shipped version of the per-unit cap
    asserted exactly this and read it as evidence the design was sound -- it is in fact proof
    the design was inert (an adversarial review caught it). `resident_bytes` is what makes it
    bind; see the test below."""
    from types import SimpleNamespace

    comm = SimpleNamespace(size=128)
    nso, width = 100, 4
    cap = 40_000
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 2 * 2**30)

    max_candidate = 32
    n_colors = me.max_colors_within_budget(cap, nso, width, "none", comm, max_candidate)
    ranks_per_color = max(1, comm.size // n_colors)
    unit_cap = me.max_unit_dets_within_budget(nso, width, "none", ranks_per_color, comm)
    assert unit_cap >= cap, (n_colors, ranks_per_color, unit_cap)


def test_resident_bytes_tightens_the_unit_cap(monkeypatch):
    """The resident set is the information the color inversion does not have, so passing it
    must produce a strictly smaller budget -- and hence a strictly smaller cap -- than the
    same call without it. Without this the per-unit cap is inert (test above)."""
    from types import SimpleNamespace

    comm = SimpleNamespace(size=128)
    nso, width, ranks = 58, 1, 5
    available = 8 * 2**30
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: available)

    loose = me.max_unit_dets_within_budget(nso, width, "none", ranks, comm)
    tight = me.max_unit_dets_within_budget(nso, width, "none", ranks, comm, resident_bytes=3 * 2**30)
    assert tight < loose

    # The budget is `safety * (available + resident) - resident`, so a bigger resident set
    # must tighten monotonically.
    caps = [
        me.max_unit_dets_within_budget(nso, width, "none", ranks, comm, resident_bytes=r * 2**30) for r in (1, 2, 3, 4)
    ]
    assert all(b <= a for a, b in pairwise(caps)), caps

    # And the predicted peak at the derived cap must actually fit that tighter budget.
    resident = 3 * 2**30
    budget = me.DEFAULT_MEMORY_SAFETY * (available + resident) - resident
    assert me.estimate_gf_peak_bytes(tight, nso, width, "none", ranks=ranks) <= budget


def test_resident_bytes_over_the_safety_share_falls_back_rather_than_flooring(monkeypatch):
    """A process already past its safety share must not drive the cap to the 1-determinant
    floor: that memory is spent either way, and a 1-determinant GF is garbage physics, not a
    safety measure. The budget falls back to `safety * available`."""
    from types import SimpleNamespace

    comm = SimpleNamespace(size=8)
    nso, width, ranks = 58, 1, 4
    available = 1 * 2**30
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: available)

    # resident so large that safety*(available+resident) - resident <= 0
    huge = 100 * 2**30
    assert me.DEFAULT_MEMORY_SAFETY * (available + huge) - huge <= 0
    fallback = me.max_unit_dets_within_budget(nso, width, "none", ranks, comm, resident_bytes=huge)
    baseline = me.max_unit_dets_within_budget(nso, width, "none", ranks, comm)
    assert fallback == baseline
    assert fallback > 1


def test_round8_smo_crash_geometry_is_refused(monkeypatch):
    """The round-8 SrMnO3 crash's own numbers must be refused by the model.

    Pinned geometry (doc/plans/dc_smo_memory.md, round 8): job-wide cap 40,340,864, nso=58,
    block width 1, available 9.5 GiB/rank, resident ~2.2 GiB at GF entry. 4/5/6 are the
    smallest color sizes in the crash's own split ([6,6,6,6,6, 5x8, 4,4, 5x10]) -- the colors
    that lost ranks (16, 25, 32, 33) to the OOM killer.

    **This test does NOT discriminate the round-8 fanout term**, and an earlier version of it
    claimed to. It passes identically with `_GF_MATVEC_ROW_FANOUT_DEFAULT = 0` (verified),
    because round 7's mechanism -- the resident-adjusted budget at `safety=0.5`, which is
    3.65 GiB here, not the 9.5 GiB a first draft of the round-8 write-up mistakenly used --
    already refuses all three colors on its own. Its value is as a **guard on the production
    geometry**: these are real numbers off a real crashed job, and the model must never start
    calling this configuration affordable. The fanout term's own presence and magnitude are
    pinned by `test_estimate_gf_peak_bytes_scales_local_rows_by_the_skew`, which asserts the
    exact formula and fails if the term is removed.
    """
    from types import SimpleNamespace

    comm = SimpleNamespace(size=128)
    cap, nso, width = 40_340_864, 58, 1
    available = int(9.5 * 2**30)
    resident = int(2.2 * 2**30)
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: available)

    budget = me._resident_adjusted_budget(me.DEFAULT_MEMORY_SAFETY, available, resident)
    for ranks in (4, 5, 6):
        peak = me.estimate_gf_peak_bytes(cap, nso, width, "none", ranks=ranks)
        assert peak > budget, (
            ranks,
            peak,
            budget,
            "the corrected model must predict this configuration does NOT fit -- it is "
            "exactly what killed ranks 16, 25, 32, 33 on the round-8 archive",
        )
        unit_cap = me.max_unit_dets_within_budget(nso, width, "none", ranks, comm, resident_bytes=resident)
        assert unit_cap < cap, (ranks, unit_cap)


# ---------------------------------------------------------------------------------------
# The ranks-per-node count must not outlive the communicator it was measured on
# (doc/plans/dc_smo_memory.md, round 9)
# ---------------------------------------------------------------------------------------


@pytest.mark.mpi
@pytest.mark.skipif(
    MPI.COMM_WORLD.size == 1,
    reason="`available_bytes_per_rank` returns the node's bytes before it reaches the cache when "
    "`comm.size == 1`, so no count is ever attached and there is nothing here to assert",
)
def test_the_ranks_per_node_count_is_attached_to_the_communicator_not_to_its_handle():
    """This was cached in a module-level dict keyed on ``comm.py2f()``.

    That is an MPI *Fortran handle*, and MPI may reuse it once the communicator is freed -- which
    this stack does in several places, including inside ``available_bytes_per_rank`` itself. A
    recycled handle would then read back the previous communicator's ranks-per-node count and
    scale every memory budget derived from it by the ratio of the two.

    An MPI attribute is destroyed with its communicator, so the stale read is impossible by
    construction rather than merely unlikely. Asserted here on a communicator that is freed and
    replaced; the test reports whether the handle was actually recycled on this run, because that
    is the case the old key got wrong and it is not reproducible on demand.
    """
    # Freed, not leaked: `MPI_Comm_free` is collective, and a communicator left to the garbage
    # collector can be freed after `MPI_Finalize` (CLAUDE.md's MPI rules).
    shared = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    expected_ranks_on_node = shared.size
    shared.Free()

    a = MPI.COMM_WORLD.Dup()
    me.available_bytes_per_rank(a)
    assert a.Get_attr(me._RANKS_PER_NODE_KEYVAL) == expected_ranks_on_node
    handle = a.py2f()
    a.Free()

    b = MPI.COMM_WORLD.Dup()
    recycled = b.py2f() == handle
    try:
        assert (
            b.Get_attr(me._RANKS_PER_NODE_KEYVAL) is None
        ), f"a fresh communicator saw a freed one's cached count (handle recycled: {recycled})"
        # And it re-derives the same value from scratch, i.e. dropping the stale read costs
        # correctness nothing.
        assert me.available_bytes_per_rank(b) == me.available_bytes_per_rank(MPI.COMM_WORLD)
    finally:
        b.Free()


def test_the_rss_breakdown_sums_to_the_resident_set():
    """`VmRSS` -- what both CIPSI memory guards measure -- is anon + file + shmem."""
    parts = me.rss_breakdown()
    assert set(parts) == {"anon", "file", "shmem"}
    if any(parts.values()):  # 0 everywhere only if /proc/self/status is unreadable
        # Sampled a moment apart, so allow the process to have moved a little in between.
        assert abs(sum(parts.values()) - me.current_rss_bytes()) < 8 * 2**20


def test_an_anonymous_allocation_lands_in_anon_and_not_in_shmem():
    """The distinction the whole ledger rests on.

    Round 5 read a ratcheting `VmRSS` as the solver's own allocations and round 9 called the
    crashed run's 2.5 GiB "retained heap"; both were largely `RssShmem`, which is a node-wide
    pool every rank counts, does not move `MemAvailable`, and no allocator tuning returns
    (`malloc_trim` measured 0% at 256 ranks). A site's cost is its `anon` delta.
    """
    before = me.rss_breakdown()
    block = np.ones((3000, 3000))  # ~68.7 MiB, touched by `ones` so it is resident
    after = me.rss_breakdown()
    try:
        assert after["anon"] - before["anon"] > 32 * 2**20, (before, after)
        # File-backed and shared pages have no reason to follow a heap allocation.
        assert after["shmem"] - before["shmem"] < 8 * 2**20, (before, after)
        assert after["file"] - before["file"] < 8 * 2**20, (before, after)
    finally:
        del block


def test_selfenergy_module_actually_forwards_gs_num_wanted():
    """`GS_NUM_WANTED` must reach the cap sizing on the SELF-ENERGY path, not just the DC's.

    It was wired in `dc_criteria` and missing here, so a production job exporting the variable saw
    it honoured during the double-counting search and silently ignored when the main solver chose
    its own cap -- which then fell back to assuming `2 * block_width` (~10) eigenstates against a
    kept manifold in the hundreds. That under-count is what approved the 20,358,272 cap behind the
    SrMnO3 crash, and the only outward sign was `log_memory_budget`'s own "gs_num_wanted was not
    supplied" line in a log nobody was diffing (`doc/plans/dc_smo_memory.md`, round 9).

    Asserted against the module source rather than by driving the calls. A test that builds the
    two calls itself and checks they carry the argument passes whether or not `selfenergy` forwards
    it -- the first draft of this test did exactly that and was green against the bug. The defect
    is a dropped argument at a specific call site, so the call site is what has to be pinned.
    """
    import inspect

    from impurityModel.ed import selfenergy

    src = inspect.getsource(selfenergy)
    head = src[src.index("sizing_block_width = resolve_sizing_block_width") :]
    head = head[: head.index("basis_information")]
    assert "gs_num_wanted = resolve_gs_num_wanted()" in head
    assert head.count("gs_num_wanted=gs_num_wanted") == 2, "both suggest_* and log_memory_budget"
