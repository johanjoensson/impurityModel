r"""The support settle: a sparse GF unit stops admitting new determinants once its frontier is below the cutoff.

Every recurrence step records the largest amplitude the normalised next block carries on a determinant the step
reached for the first time; once, over a window of 16 steps, that stays at or below ``ratio x slaterWeightMin`` and
is not growing, the unit's support is final and the recurrence carries on on the ``P H P`` CSR. The contract these
tests pin:

* the rule itself, on synthetic records (window not full; ``<=`` at the boundary; the growth refusal; the floor;
  ``swm = 0`` and ratio 0 are closure);
* every recorded quantity against a dense numpy replay of the recurrence on the whole sector, step by step;
* it settles a support that trickles and never closes, where the unit's own trace says;
* the settled ``G`` is no farther from the exact one than the unsettled sparse ``G`` by more than a few percent;
* reort other than 'none' and the off switch (``support_settle=False``) leave the unit alone; a matrix that does
  not fit undoes the settle and leaves the result as if it had never fired;
* every rank holds the same records and decides alike, and the rank count moves nothing (``-n 2``/``-n 3``);
* the end-of-unit probe reports the frontier a settled unit left behind;
* the retired freezes (``GF_CLOSURE_FREEZE``, ``GF_STAGNATION_FREEZE``, ``GF_STAGNATION_WINDOW``) warn that they
  are no longer read.
"""

import itertools
import warnings

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import config, gf_primitives, gf_solvers
from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.gf_primitives import FreezeRule
from impurityModel.ed.greens_function import calc_G
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.basis.test_sparse_matrix_build import _basis, _det
from impurityModel.test.support.gf_chain_fixtures import (
    DELTA,
    N_ORB,
    OMEGA,
    SECTOR,
    chain_operator,
    chain_seeds,
    exact_chain_g,
    run_chain,
)

WINDOW = gf_primitives.SETTLE_WINDOW


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in (
        "GF_SUPPORT_TRACE",
        "GF_SUPPORT_FREEZE",
        "GF_SETTLE_RATIO",
        "GF_FROZEN_CSR",
        "APPLY_ROW_CHUNKS",
        "MATVEC_PRUNE",
        *config.RETIRED_KNOBS,
    ):
        monkeypatch.delenv(name, raising=False)


def _trickle(comm=None, **kw):
    """A chain whose support trickles: it reaches ~2,700 of the sector's 3432 determinants in ~45 blocks, then
    adds a few per block for hundreds of blocks (cutoff 1e-4)."""
    kw.setdefault("settle_ratio", 1.0)  # explicit: the rule's own tests do not move with the knob's default
    return run_chain(comm=comm, op=chain_operator(seed=5), slater_min=1e-4, **kw)


def _g(result):
    alphas, betas, r, _ = result
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)


def _exact_sector_g(seed=5):
    return exact_chain_g([_det(list(c)) for c in itertools.combinations(range(N_ORB), 7)], seed=seed)


def _first_settling(trace, ratio, swm, window=WINDOW):
    """The first step whose window of next-block frontiers passes the rule, replayed from a unit's own trace."""
    rule = FreezeRule(window, ratio, swm)
    for row in trace:
        if rule.push(row[8], row[3], row[4]):
            return row[0]
    return None


def _rule(max2s, ratio=1.0, swm=1e-4, window=4):
    rule = FreezeRule(window, ratio, swm)
    return [rule.push(m, 0.0, 0.0) for m in max2s], rule


def test_the_rule_needs_a_full_window_and_settles_at_the_boundary():
    swm = 1e-4
    flat = [(1.0 * swm) ** 2] * 4  # exactly ratio x swm: <= settles
    verdicts, rule = _rule(flat, ratio=1.0, swm=swm)
    assert verdicts == [False, False, False, True], "not full, not full, not full, then <= at the boundary"
    assert _rule(flat, ratio=0.999, swm=swm)[0][-1] is False, "just above the threshold does not"
    assert _rule([0.0] * 4, ratio=0.0, swm=swm)[0][-1] is True, "ratio 0 is closure: all-zero records settle"
    assert _rule([1e-30] + [0.0] * 3, ratio=0.0, swm=swm)[0][-1] is False
    v = rule.verdict()
    assert v["window"] == 4 and v["max_amp"] == pytest.approx(swm) and v["max_amp_over_swm"] == pytest.approx(1.0)


def test_the_rule_refuses_a_growing_frontier_and_forgives_a_shrinking_one():
    swm = 1e-4
    top = (swm * 3) ** 2
    growing = [top * 1e-4, top * 1e-3, top * 1e-2, top * 1e-1, top]  # below 3x, but climbing
    assert _rule(growing, ratio=3.0, swm=swm, window=5)[0][-1] is False
    shrinking = growing[::-1]
    assert _rule(shrinking, ratio=3.0, swm=swm, window=5)[0][-1] is True
    # a frontier at exactly 0 that shows noise at the floor (1e-3 x swm) is flat; one decade of amplitude above
    # the floor, late in the window, is a rise and refuses (documented: the trend term is a heuristic)
    assert _rule([0.0, 0.0, 0.0, (1e-3 * swm) ** 2], ratio=1.0, swm=swm)[0][-1] is True
    assert _rule([0.0, 0.0, 0.0, (1e-2 * swm) ** 2], ratio=1.0, swm=swm)[0][-1] is False


def test_at_swm_zero_only_a_closed_window_settles():
    assert _rule([0.0] * 4, ratio=5.0, swm=0.0)[0][-1] is True
    assert _rule([0.0, 0.0, 0.0, 1e-300], ratio=5.0, swm=0.0)[0][-1] is False
    with pytest.raises(ValueError):
        FreezeRule(4, -1.0, 1e-4)


def test_it_settles_a_trickling_support_where_its_own_trace_says():
    info = _trickle()[3]
    trace = info["support_trace"]
    assert info["support_frozen"] and info["csr_fallback"]
    assert _first_settling(trace, 1.0, 1e-4) == len(trace), "it settled on the first step whose window passed"
    assert len(trace) < 120, f"it settled after {len(trace)} blocks"
    assert info["retained_size"] < SECTOR, "premise: the support had not closed"
    window = trace[len(trace) - WINDOW :]
    assert sum(row[3] for row in window) > 0.0, "premise: weight still reached new rows when it settled"


def test_both_prunes_are_counted(monkeypatch):
    """Every step notes the matvec prune before its admission and the next block's prune after it."""
    calls = []
    real = gf_primitives._CappedBasisProxy.note_step_cut

    def recording(self, weight):
        calls.append(weight)
        return real(self, weight)

    monkeypatch.setattr(gf_primitives._CappedBasisProxy, "note_step_cut", recording)
    history = _trickle()[3]["support_trace"]
    assert len(history) * 2 - 1 <= len(calls) <= len(history) * 2 + 1, (len(calls), len(history))
    assert all(w >= 0.0 for w in calls)


@pytest.mark.parametrize("ratio", [0.0, 1.0, 3.0])
def test_the_settled_g_costs_no_more_than_the_cutoff_already_does(ratio):
    """The settle drops only amplitude the cutoff itself works at: the settled G is no farther from the exact
    sector G than the unsettled sparse G by more than a few percent (measured 1.32e-5 against 1.30e-5 at ratio 1,
    1.14e-5 at ratio 3: the error is the cutoff's, not the settle's)."""
    settled = _g(_trickle(settle_ratio=ratio))
    plain = _g(_trickle(support_settle=False))
    exact = _exact_sector_g()
    err_settled = np.abs(settled - exact).max()
    err_plain = np.abs(plain - exact).max()
    assert err_plain > 1e-7, "premise: the cutoff costs the sparse kernel something"
    assert err_settled <= 1.1 * err_plain, (err_settled, err_plain)


def test_it_is_on_by_default():
    assert _trickle()[3]["support_frozen"]


def test_with_no_cutoff_it_freezes_only_on_closure():
    info = run_chain()[3]
    assert info["support_frozen"] and info["retained_size"] == SECTOR
    window = info["support_trace"][len(info["support_trace"]) - WINDOW :]
    assert all(row[3] == 0.0 and row[4] == 0.0 and row[8] == 0.0 for row in window)


@pytest.mark.parametrize("reort", [True])
def test_it_needs_reort_none(reort):
    info = _trickle(reort=reort)[3]
    assert not info["support_frozen"] and info["support_trace"] == []


def test_off_switch():
    info = _trickle(support_settle=False)[3]
    assert not info["support_frozen"] and info["support_trace"] == []


def test_a_matrix_that_does_not_fit_undoes_the_freeze(monkeypatch):
    plain = _trickle(support_settle=False)
    monkeypatch.setattr(gf_solvers, "_frozen_csr_fits", lambda *a, **k: False)
    declined = _trickle()
    assert not declined[3]["support_frozen"] and not declined[3]["csr_fallback"]
    np.testing.assert_array_equal(declined[0], plain[0])
    np.testing.assert_array_equal(declined[1], plain[1])


@pytest.mark.mpi
def test_every_rank_holds_the_same_history_and_decides_alike():
    comm = MPI.COMM_WORLD
    result = _trickle(comm=comm)
    info = result[3]
    assert info["support_frozen"]
    # d_g is computed per rank from allreduced coefficients: logged, not replicated, so it is left out.
    trace = np.array([row[:6] + row[7:] for row in info["support_trace"]], dtype=np.float64).tobytes()
    mine = (trace, info["retained_size"], len(result[0]))
    assert len(set(comm.allgather(mine))) == 1
    assert _first_settling(info["support_trace"], 1.0, 1e-4) == len(info["support_trace"])
    # the rank count moves nothing: the same settle step, size and rows as the serial run
    serial = _trickle()[3]
    assert len(info["support_trace"]) == len(serial["support_trace"])
    assert info["retained_size"] == serial["retained_size"]
    np.testing.assert_allclose(_g(result), _g(_trickle()), atol=1e-6)


@pytest.mark.parametrize("name", ["GF_CLOSURE_FREEZE", "GF_STAGNATION_FREEZE", "GF_STAGNATION_WINDOW"])
def test_the_retired_freezes_warn_that_they_are_no_longer_read(name, monkeypatch):
    monkeypatch.setenv(name, "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.warn_retired_knobs()
    assert any(name in str(w.message) and "settle rule" in str(w.message) for w in caught)


def _dense_chain(seed):
    """The chain's H on its whole 7-electron sector, the seed columns there, and the seed rows."""
    keys = sorted(_det(list(c)) for c in itertools.combinations(range(N_ORB), 7))
    basis = _basis(keys, None, n_orb=N_ORB)
    h = np.asarray(build_dense_matrix(basis, chain_operator(seed=seed)))
    index = {bytes(d.to_bytearray()[:2]): i for i, d in enumerate(basis.local_basis)}
    v = np.zeros((h.shape[0], len(chain_seeds())), dtype=complex)
    for j, s in enumerate(chain_seeds()):
        for det, amp in s.items():
            v[index[bytes(det.to_bytearray()[:2])], j] = amp[0]
    return h, v, np.flatnonzero(np.any(v != 0, axis=1))


def test_every_steps_weights_are_the_dense_ones(monkeypatch):
    """The leak and both cuts, step by step, against the same recurrence replayed in numpy on the whole sector:
    the matvec's cut is the summed H q rows below the cutoff, the leak the kept rows outside the support so far
    (every column), the next block's cut its dropped rows weighed through beta (= the residual's rows there),
    and the block is re-factored after its prune. Pins the quantities, the Frobenius norm, the gram's
    orientation and the one-step lag of the next block's cut -- not only the bookkeeping. Every weight is
    invariant under the column phases a QR may choose. Compared from step 10 on, where the cuts are real
    weights rather than cancellation round-off."""
    notes = []
    real = gf_primitives._CappedBasisProxy.note_step_cut

    def recording(self, weight):
        notes.append(weight)
        return real(self, weight)

    monkeypatch.setattr(gf_primitives._CappedBasisProxy, "note_step_cut", recording)
    cutoff = 1e-4
    info = _trickle()[3]
    trace = info["support_trace"]

    h, v, seed_rows = _dense_chain(seed=5)
    support = np.zeros(h.shape[0], dtype=bool)
    support[seed_rows] = True
    q, _ = np.linalg.qr(v)
    q_prev, beta_prev, cut_next_prev = np.zeros_like(q), None, 0.0
    compared = 0
    for step in range(30):
        wp = h @ q
        hq2 = float(np.sum(np.abs(wp) ** 2))
        kept = np.any(np.abs(wp) > cutoff, axis=1)
        cut_matvec = float(np.sum(np.abs(wp[~kept]) ** 2))
        new_rows = kept & ~support
        leak = float(np.sum(np.abs(wp[new_rows]) ** 2))
        max2 = float(np.max(np.abs(wp[new_rows]) ** 2, initial=0.0))
        support |= kept
        wp[~kept] = 0.0
        wp -= q @ (q.conj().T @ wp)
        if beta_prev is not None:
            wp -= q_prev @ beta_prev.conj().T
        q_next, beta = np.linalg.qr(wp)
        max2_next = float(np.max(np.abs(q_next[new_rows]) ** 2, initial=0.0))
        dropped = ~np.any(np.abs(q_next) > cutoff, axis=1)
        cut_next = float(np.sum(np.abs(wp[dropped]) ** 2))
        q_next[dropped] = 0.0
        q_next, r2 = np.linalg.qr(q_next)
        if step >= 10:
            assert trace[step][3] == pytest.approx(leak, rel=1e-9), step
            assert trace[step][4] == pytest.approx(cut_matvec + cut_next_prev, rel=1e-9), step
            assert trace[step][5] == pytest.approx(hq2, rel=1e-9), step
            assert trace[step][7] == pytest.approx(max2, rel=1e-9, abs=1e-30), step
            assert trace[step][8] == pytest.approx(max2_next, rel=1e-6, abs=1e-30), step
            assert notes[2 * step] == pytest.approx(cut_matvec, rel=1e-9), step
            assert notes[2 * step + 1] == pytest.approx(cut_next, rel=1e-9), step
            compared += cut_matvec > 0 and cut_next > 0
        q_prev, q, beta_prev, cut_next_prev = q, q_next, r2 @ beta, cut_next
    assert compared >= 15, "premise: both cuts are real weights on most compared steps"


def test_the_trace_records_every_step_and_its_sizes_add_up():
    """``support_trace`` carries ``(step, retained_before, n_new, leak, cut, |Hq|^2, d_g, max|amp|^2,
    next-block max|amp|^2)`` per step: the steps count from 1, the retained sizes add up to the unit's support,
    and the monitor's ``d_g`` is logged (NaN until the monitor has measured one)."""
    info = _trickle()[3]
    trace = info["support_trace"]
    assert len(trace) > 0
    retained = trace[0][1]
    for k, (step, before, n_new, leak, cut, hq2, _d_g, _max2, _max2_next) in enumerate(trace):
        assert step == k + 1 and before == retained
        assert hq2 >= leak >= 0.0 and cut >= 0.0
        retained += n_new
    assert retained == info["retained_size"]


def test_a_settled_unit_carries_the_verdict_it_decided_on():
    info = _trickle()[3]
    v = info["settle_verdict"]
    assert v["step"] == len(info["support_trace"]) and v["window"] == WINDOW and v["ratio"] == 1.0
    assert v["max_amp"] <= v["ratio"] * 1e-4 * (1 + 1e-12) and v["max_amp_over_swm"] <= 1.0
    window = info["support_trace"][len(info["support_trace"]) - WINDOW :]
    assert v["leak"] == pytest.approx(sum(row[3] for row in window)) and v["cut"] == pytest.approx(
        sum(row[4] for row in window)
    )
    assert _trickle(support_settle=False)[3]["settle_verdict"] is None


def test_the_step_trace_prints_one_line_per_block(monkeypatch, capsys):
    monkeypatch.setenv("GF_SUPPORT_TRACE", "1")
    history = _trickle()[3]["support_trace"]
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("GF support step ")]
    assert len(lines) == len(history)


def test_with_no_cutoff_the_closed_support_gives_the_exact_g():
    np.testing.assert_allclose(_g(run_chain()), _exact_sector_g(seed=3), atol=1e-8)


def test_a_declined_freeze_is_asked_once_and_a_later_cap_still_takes_the_csr(monkeypatch):
    """The support freeze declines (the matrix does not fit); the unit carries on unfrozen, and when its cap binds
    later that freeze is offered the CSR again -- and fits."""
    asked = []
    real_fits = gf_solvers._frozen_csr_fits

    def first_declines(*args, **kwargs):
        asked.append(1)
        return len(asked) > 1 and real_fits(*args, **kwargs)

    monkeypatch.setattr(gf_solvers, "_frozen_csr_fits", first_declines)
    # ratio 10 settles at ~2,265 determinants, below the 2,600 cap (ratio 1 would settle at ~3,074, past it)
    info = _trickle(cap=2600, settle_ratio=10.0)[3]
    assert not info["support_frozen"] and info["cap_hit"] and info["csr_fallback"]
    assert len(asked) == 2, "the support freeze once, the cap freeze once"


def test_boundary_frontier_is_the_dense_image_outside_the_support():
    """``boundary_frontier`` against a dense ``H`` on the whole sector: the largest ``|amp|^2`` on a row outside
    ``P`` and the summed weight there, with and without the cutoff (summed rows cut, not partials)."""
    h, _v, _ = _dense_chain(seed=5)
    keys = sorted(_det(list(c)) for c in itertools.combinations(range(N_ORB), 7))
    full = _basis(keys, None, n_orb=N_ORB)
    rng = np.random.default_rng(0)
    inside = np.zeros(h.shape[0], dtype=bool)
    inside[rng.choice(h.shape[0], 900, replace=False)] = True
    sub = _basis([k for k, m in zip(full.local_basis, inside) if m], None, n_orb=N_ORB)
    x = np.zeros((h.shape[0], 2), dtype=complex)
    x[inside] = rng.normal(size=(int(inside.sum()), 2)) + 1j * rng.normal(size=(int(inside.sum()), 2))
    op = chain_operator(seed=5)
    order = {bytes(d.to_bytearray()[:2]): i for i, d in enumerate(full.local_basis)}
    sub_rows = [order[bytes(d.to_bytearray()[:2])] for d in sub.local_basis]
    block = gf_primitives.local_dense_to_block(sub, x[sub_rows])
    image = h @ x
    for cutoff in (0.0, 1e-1):
        out = image[~inside]
        kept = np.any(np.abs(out) > cutoff, axis=1) if cutoff > 0 else np.ones(len(out), dtype=bool)
        want_max = float(np.max(np.abs(out[kept]) ** 2))
        want_weight = float(np.sum(np.abs(out[kept]) ** 2))
        got_max, got_weight = gf_primitives.boundary_frontier(op, block, sub, cutoff, None)
        assert got_max == pytest.approx(want_max, rel=1e-9)
        assert got_weight == pytest.approx(want_weight, rel=1e-9)


def test_a_settled_unit_reports_the_frontier_it_left_behind(monkeypatch):
    monkeypatch.setenv("GF_SUPPORT_TRACE", "1")  # the probe is an opt-in diagnostic
    info = run_chain(slater_min=1e-4)[3]
    assert info["support_frozen"] and info["csr_fallback"]
    last = info["support_trace"][-1]
    assert info["leak_settle"] == last.leak and info["frontier_settle_matvec"] == pytest.approx(last.max2_matvec**0.5)
    assert np.isfinite(info["leak_end"]) and info["leak_end"] >= 0
    assert 0 <= info["frontier_end_matvec"] <= info["leak_end"] ** 0.5 * (1 + 1e-12)
    assert np.isfinite(info["frontier_end"]) and info["frontier_end"] >= info["frontier_end_matvec"] / 1e3


def test_the_probe_is_off_unless_traced_and_never_runs_for_a_capped_unit(monkeypatch):
    """Without GF_SUPPORT_TRACE the settling step's own record is reported and no extra matvec is made; a unit
    frozen by the cap is not a settle and never probed, traced or not."""
    calls = []
    real = gf_primitives.boundary_frontier
    monkeypatch.setattr(gf_solvers, "boundary_frontier", lambda *a, **k: calls.append(1) or real(*a, **k))
    info = run_chain(slater_min=1e-4)[3]
    assert info["support_frozen"] and np.isfinite(info["leak_settle"])
    assert np.isnan(info["frontier_end_matvec"]) and calls == []
    monkeypatch.setenv("GF_SUPPORT_TRACE", "1")
    capped = _trickle(cap=2000)[3]
    assert capped["cap_hit"] and capped["csr_fallback"] and calls == []
    assert np.isnan(capped["frontier_end_matvec"])


def test_a_closed_support_leaves_no_frontier(monkeypatch):
    monkeypatch.setenv("GF_SUPPORT_TRACE", "1")
    info = run_chain()[3]  # no cutoff: freezes only on exact closure
    assert info["csr_fallback"] and info["frontier_end_matvec"] == 0.0 and info["leak_end"] == 0.0


def test_a_unit_that_never_settled_reports_no_frontier(monkeypatch):
    monkeypatch.setenv("GF_SUPPORT_TRACE", "1")
    info = run_chain(slater_min=1e-4, support_settle=False)[3]
    assert np.isnan(info["frontier_end_matvec"]) and np.isnan(info["leak_settle"])


def test_a_half_finished_record_is_dropped_by_the_next_step_and_never_decides():
    """A step that breaks down before its next block leaves a pending record; ``begin_step`` drops it, and the hook
    with nothing pending neither records nor reduces."""
    basis = _basis([_det([0, 1, 2, 3, 4, 5, 6])], None, n_orb=N_ORB)
    policy = gf_primitives.SupportPolicy(basis, 10**9)
    policy.enable_support_freeze(4, 1.0, 1e-4)
    policy._pending = (None, 0, 0, 0.0, 0.0, 0.0, 0.0)
    policy.begin_step()
    assert policy._pending is None
    policy.note_next_block(ManyBodyState.from_keys([]))
    assert policy.support_trace == [] and not policy.frozen


def test_the_default_ratio_is_the_knob_and_an_explicit_one_wins(monkeypatch):
    assert config.GF_SETTLE_RATIO.get() == 3.0
    info = run_chain(op=chain_operator(seed=5), slater_min=1e-4)[3]
    assert info["settle_verdict"]["ratio"] == 3.0
    monkeypatch.setenv("GF_SETTLE_RATIO", "3")
    assert run_chain(op=chain_operator(seed=5), slater_min=1e-4)[3]["settle_verdict"]["ratio"] == 3.0
    explicit = run_chain(op=chain_operator(seed=5), slater_min=1e-4, settle_ratio=0.5)[3]
    assert explicit["settle_verdict"]["ratio"] == 0.5
