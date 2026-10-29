"""``gf_recurrence``: the Green's-function recurrence on a discovered-key matrix, against the ManyBodyState
kernel (``block_Green_sparse``) and the exact resolvent on the 14-orbital chain of the support-freeze tests.

Pinned: the continued fraction equals the sparse kernel's at every cap, with the settle, and with importance
pruning (where the sparse kernel bans and this loop does not, so only the unpruned-equivalent limit is compared);
the retained count equals the sparse kernel's where the rule is the same; two runs agree bit for bit; the result
is the same at every rank count.
"""

import functools

import numpy as np
import pytest

from impurityModel.ed import config
from impurityModel.ed.gf_recurrence import block_Green_discovered as _block_Green_discovered
from impurityModel.test.gf.test_gf_support_freeze import _exact_sector_g, _g
from impurityModel.test.support.comms import COMMS
from impurityModel.test.support.gf_chain_fixtures import run_chain

#: The recurrence with the discovered-key matrix (``"auto"`` takes the cheaper retained mode without a consumer).
block_Green_discovered = functools.partial(_block_Green_discovered, matrix_mode="discovered")


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in (
        "GF_SUPPORT_TRACE",
        "GF_SUPPORT_FREEZE",
        "GF_SETTLE_RATIO",
        "GF_LANCZOS_ADMIT_TOL",
        "APPLY_ROW_CHUNKS",
    ):
        monkeypatch.delenv(name, raising=False)


def _both(comm, **kw):
    sparse = run_chain(comm, **kw)
    matrix = run_chain(comm, kernel=block_Green_discovered, **kw)
    return sparse, matrix


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("cap", [10**9, 2500, 900])
def test_the_continued_fraction_equals_the_sparse_kernels(comm, cap):
    (a0, b0, r0, i0), (a1, b1, r1, i1) = _both(comm, cap=cap, support_settle=False)
    np.testing.assert_allclose(_g((a1, b1, r1, i1)), _g((a0, b0, r0, i0)), atol=1e-9)
    assert i1["retained_size"] == i0["retained_size"]
    assert i1["cap_hit"] == i0["cap_hit"]
    assert i1["engine"] == "discovered"


@pytest.mark.parametrize("comm", COMMS)
def test_uncapped_it_is_the_exact_resolvent(comm):
    _s, (a1, b1, r1, i1) = _both(comm, cap=10**9, support_settle=False)
    np.testing.assert_allclose(_g((a1, b1, r1, i1)), _exact_sector_g(seed=3), atol=1e-6)


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("swm", [1e-3, 1e-4])
def test_a_settled_unit_is_no_farther_from_the_exact_g_than_the_sparse_kernels(comm, swm):
    """At ``slaterWeightMin > 0`` this loop cuts only the frontier rows (the array kernel's semantics), where the
    sparse kernel also cuts the ``P`` rows and the next block, so the two differ by the cut amounts -- measured
    here against the exact resolvent, where this loop is the closer one."""
    sparse, matrix = _both(comm, slater_min=swm, settle_ratio=1.0)
    exact = _exact_sector_g(seed=3)
    err_sparse = abs(_g(sparse) - exact).max()
    err_matrix = abs(_g(matrix) - exact).max()
    assert err_matrix <= 1.5 * err_sparse + 1e-9
    assert matrix[3]["support_frozen"] and sparse[3]["support_frozen"]


@pytest.mark.parametrize("comm", COMMS)
def test_two_runs_agree_bit_for_bit(comm):
    one = run_chain(comm, kernel=block_Green_discovered, cap=2500, support_settle=False)
    two = run_chain(comm, kernel=block_Green_discovered, cap=2500, support_settle=False)
    assert np.array_equal(one[0], two[0]) and np.array_equal(one[1], two[1])


@pytest.mark.parametrize("comm", COMMS)
def test_full_reorthogonalization_equals_the_sparse_kernels(comm):
    (a0, b0, r0, i0), (a1, b1, r1, i1) = _both(comm, cap=10**9, reort=True)
    np.testing.assert_allclose(_g((a1, b1, r1, i1)), _g((a0, b0, r0, i0)), atol=1e-8)


@pytest.mark.parametrize("comm", COMMS)
def test_importance_pruning_without_the_ban_admits_at_least_what_the_banned_rule_does(comm, monkeypatch):
    """The sparse kernel bans every row a step did not admit; here a deferred row stays in the frontier and
    re-enters when a stronger path reaches it, so the retained set is a superset in size and G no farther from
    the exact one."""
    monkeypatch.setenv("GF_LANCZOS_ADMIT_TOL", "2e-2")
    sparse, matrix = _both(comm)
    exact = _exact_sector_g(seed=3)
    assert matrix[3]["retained_size"] >= sparse[3]["retained_size"]
    assert abs(_g(matrix) - exact).max() <= abs(_g(sparse) - exact).max() + 1e-9


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("mode", ["retained", "none"])
@pytest.mark.parametrize("kw", [{"cap": 2500}, {"cap": 10**9}, {"cap": 10**9, "slater_min": 1e-3, "settle_ratio": 1.0}])
def test_every_matvec_mode_gives_the_same_green_function(comm, mode, kw):
    """The three modes differ in how ``H q`` is formed, not in what it is: the matrix over every discovered key,
    the operator applied to the block (with the ``P H P`` matrix built at the freeze), or the operator applied
    every block."""
    ref = run_chain(comm, kernel=block_Green_discovered, matrix_mode="discovered", **kw)
    got = run_chain(comm, kernel=block_Green_discovered, matrix_mode=mode, **kw)
    np.testing.assert_allclose(_g(got), _g(ref), atol=1e-7)
    assert got[3]["engine"] == mode and ref[3]["engine"] == "discovered"
    assert got[3]["retained_size"] == ref[3]["retained_size"]
    assert got[3]["cap_hit"] == ref[3]["cap_hit"] and got[3]["support_frozen"] == ref[3]["support_frozen"]
    assert got[3]["matrix"] is None


@pytest.mark.parametrize("comm", COMMS)
def test_block_green_sparse_runs_the_recurrence_on_a_matrix(comm, monkeypatch):
    """``block_Green_sparse`` is the recurrence: by default it prices the matvec mode (the retained mode without a
    consumer of the frontier rows), and ``set_gf_switches(matrix_mode=...)`` forces one."""
    from impurityModel.test.support.gf_switches import set_gf_switches

    routed = run_chain(comm, cap=2500, support_settle=False)
    assert routed[3]["engine"] == "retained"
    direct = run_chain(
        comm, kernel=functools.partial(_block_Green_discovered, matrix_mode="retained"), cap=2500, support_settle=False
    )
    assert np.array_equal(routed[0], direct[0]) and np.array_equal(routed[1], direct[1])
    set_gf_switches(monkeypatch, matrix_mode="discovered")
    assert run_chain(comm, cap=2500, support_settle=False)[3]["engine"] == "discovered"


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("mode", ["discovered", "retained"])
def test_the_stage_table_names_the_recurrences_stages(comm, mode, monkeypatch):
    """``IMPMOD_MATVEC_PROF`` stage names are read by the measurement harness, so they are pinned."""
    from impurityModel.ed import DistributedSparseMatrix as discovered_matrix

    monkeypatch.setattr(discovered_matrix, "_PROF", True)
    discovered_matrix._STAGES.clear()
    run_chain(comm, kernel=block_Green_discovered, matrix_mode=mode, cap=2500)
    names = set(discovered_matrix._STAGES)
    assert {"apply", "step_dense", "tsqr"} <= names
    if mode == "discovered":
        assert "matvec_all" in names
    else:
        assert {"grow_apply", "grow_keys", "decide", "build_columns", "frozen_kernel"} <= names
    assert all(calls > 0 and sec >= 0.0 for sec, calls, _rss in discovered_matrix._STAGES.values())


@pytest.mark.parametrize("comm", COMMS)
def test_the_mode_follows_the_memory_budget(comm):
    """No budget: the matrix. A budget the matrix does not fit but the ``P H P`` matrix does: retained. Neither:
    the operator on every block. Replicated on every rank."""
    from impurityModel.ed.gf_recurrence import MATRIX_MEMORY_SAFETY, choose_matrix_mode
    from impurityModel.test.basis.test_sparse_matrix_build import _basis
    from impurityModel.test.support.gf_chain_fixtures import N_ORB, chain_operator, chain_seeds

    basis = _basis(sorted({d for s in chain_seeds() for d in s}), comm, n_orb=N_ORB)
    op = chain_operator()
    assert choose_matrix_mode(basis, op, 3000, 2, None) == "discovered"
    report = {}
    assert choose_matrix_mode(basis, op, 3000, 2, 10**12, report) == "discovered"
    scale = MATRIX_MEMORY_SAFETY / config.GF_CSR_MEMORY_SAFETY.get()
    rss = report["rss"]
    assert report["need_discovered"] > report["need_retained"]  # the matrix holds the frontier rows too
    # The retained matrix fits (rss + need_retained <= budget) while the discovered one, held to a smaller share
    # of the same budget, does not.
    budget = rss + report["need_retained"] * 1.01
    assert rss + report["need_discovered"] > budget * scale
    assert choose_matrix_mode(basis, op, 3000, 2, budget) == "retained"
    assert choose_matrix_mode(basis, op, 3000, 2, 1) == "none"
    # Without a consumer of the frontier rows the discovered matrix is never chosen: it costs more memory and a
    # slower frozen matvec for nothing.
    assert choose_matrix_mode(basis, op, 3000, 2, None, want_discovered=False) == "retained"
    assert choose_matrix_mode(basis, op, 3000, 2, 10**12, want_discovered=False) == "retained"
    assert choose_matrix_mode(basis, op, 3000, 2, 1, want_discovered=False) == "none"
    if comm is not None:
        assert len(set(comm.allgather(choose_matrix_mode(basis, op, 3000, 2, budget)))) == 1


@pytest.mark.parametrize("comm", COMMS)
def test_the_scoring_accumulator_is_priced_and_a_unit_that_cannot_hold_it_runs_retained(comm):
    from impurityModel.ed.gf_recurrence import MATRIX_MEMORY_SAFETY, REFINE_SKEW, choose_matrix_mode
    from impurityModel.test.basis.test_sparse_matrix_build import _basis
    from impurityModel.test.support.gf_chain_fixtures import N_ORB, chain_operator, chain_seeds

    basis = _basis(sorted({d for s in chain_seeds() for d in s}), comm, n_orb=N_ORB)
    op = chain_operator()
    plain, priced = {}, {}
    choose_matrix_mode(basis, op, 3000, 2, 10**12, plain)
    choose_matrix_mode(basis, op, 3000, 2, 10**12, priced, True, 16)
    assert plain["need_accumulator"] == 0
    assert priced["need_accumulator"] == int(REFINE_SKEW * 2 * 16 * plain["n_p_local"] * 2 * 16)  # x and p, complex
    assert priced["need_discovered"] == plain["need_discovered"] + priced["need_accumulator"]
    # A budget that holds the matrix but not the matrix plus the accumulator: retained, with the accumulator priced.
    scale = MATRIX_MEMORY_SAFETY / config.GF_CSR_MEMORY_SAFETY.get()
    budget = (plain["rss"] + plain["need_discovered"] + priced["need_accumulator"] / 2) / scale
    assert choose_matrix_mode(basis, op, 3000, 2, budget) == "discovered"
    assert choose_matrix_mode(basis, op, 3000, 2, budget, None, True, 16) == "retained"


@pytest.mark.parametrize("comm", COMMS)
def test_the_retained_mode_stores_p_rows_only(comm, monkeypatch):
    """``"retained"`` is priced at ``need_CSR`` because its matrix has no frontier rows; one that kept them would
    overrun the budget it was chosen for and still give the same G, so only the matrix itself can show it."""
    from impurityModel.ed import gf_recurrence

    made = []
    real = gf_recurrence.DistributedSparseMatrix

    def spy(*args, **kw):
        made.append(real(*args, **kw))
        return made[-1]

    monkeypatch.setattr(gf_recurrence, "DistributedSparseMatrix", spy)
    run_chain(comm, kernel=block_Green_discovered, matrix_mode="retained", cap=2500)
    assert made and not any(m.keep_frontier for m in made)
    assert all(m.n_p_local in (m.n_k_local, 0) for m in made)
    made.clear()
    run_chain(comm, kernel=block_Green_discovered, matrix_mode="discovered", cap=2500)
    assert made and all(m.keep_frontier for m in made)
