r"""``GF_SUPPORT_FREEZE``: freeze a sparse GF unit's support once growing it buys no more than the cutoff costs.

Every step weighs, in the units of its matvec, the weight landing on rows outside the support (the *leak*)
against what the step's ``slaterWeightMin`` prunes dropped (the *cut*); once the leak over the last 16 steps
is no larger than the cut over them, the support freezes and the recurrence carries on on the ``P H P`` CSR.
The contract these tests pin:

* the rule itself, on synthetic windows (not full: no; ``<=``; rolling);
* it fires on a support that trickles and never closes -- where it fires is the rule applied to the run's
  own per-step history, far earlier than a zero-growth window ever comes;
* both prunes are counted, the next block's weighed back through ``beta beta^H``;
* the frozen ``G`` is no farther from the exact one than the unfrozen sparse ``G``;
* with no cutoff it freezes only on exact closure;
* reort other than 'none' and the off switch leave the unit alone; a matrix that does not fit undoes the
  freeze and leaves the result as if it had never fired;
* every rank holds the same history bits and decides alike (``-n 2``/``-n 3``);
* it is on by default, and the freezes it replaces (``GF_CLOSURE_FREEZE``, ``GF_STAGNATION_FREEZE``,
  ``GF_STAGNATION_WINDOW``) warn that they are no longer read.
"""

import itertools
import warnings
from collections import deque

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import config, gf_primitives, gf_solvers
from impurityModel.ed.gf_primitives import support_window_freezes
from impurityModel.ed.greens_function import calc_G
from impurityModel.test.basis.test_sparse_matrix_build import _det
from impurityModel.test.support.gf_chain_fixtures import (
    DELTA,
    N_ORB,
    OMEGA,
    SECTOR,
    chain_operator,
    exact_chain_g,
    run_chain,
)

WINDOW = gf_solvers._SUPPORT_WINDOW


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in ("GF_SUPPORT_FREEZE", "GF_FROZEN_CSR", "GF_APPLY_ROW_CHUNKS", *config.RETIRED_KNOBS):
        monkeypatch.delenv(name, raising=False)


def _trickle(comm=None, **kw):
    """A chain whose support trickles: it reaches ~2,700 of the sector's 3432 determinants in ~45 blocks, then
    adds a few per block for hundreds of blocks (cutoff 1e-4)."""
    return run_chain(comm=comm, op=chain_operator(seed=5), slater_min=1e-4, **kw)


def _g(result):
    alphas, betas, r, _ = result
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)


def _exact_sector_g(seed=5):
    return exact_chain_g([_det(list(c)) for c in itertools.combinations(range(N_ORB), 7)], seed=seed)


def _first_firing(history):
    window = deque(maxlen=WINDOW)
    for step, entry in enumerate(history, start=1):
        window.append(entry)
        if support_window_freezes(window)[0]:
            return step
    return None


def test_the_window_rule():
    window = deque(maxlen=3)
    window.extend([(0.0, 1.0), (0.0, 1.0)])
    assert not support_window_freezes(window)[0], "a window that is not full never freezes"
    window.append((2.0, 0.0))
    assert support_window_freezes(window)[0], "leak == cut freezes (<=)"
    window.append((1e-300, 0.0))
    assert not support_window_freezes(window)[0], "leak > cut does not"
    window.extend([(0.0, 0.0)] * 3)
    assert support_window_freezes(window)[0], "rolling: the old leak has left the window; 0 <= 0 is closure"


def test_it_freezes_a_trickling_support_where_its_own_history_says():
    result = _trickle()
    info = result[3]
    history = info["support_history"]
    assert info["support_frozen"] and info["csr_fallback"]
    assert _first_firing(history) == len(history), "it froze on the first step whose window passed the rule"
    assert len(history) < 100, f"it froze after {len(history)} blocks"
    assert info["retained_size"] < SECTOR, "premise: the support had not closed"
    window = history[len(history) - WINDOW :]
    assert sum(leak for leak, _ in window) > 0.0, "premise: still growing when it froze"


def test_both_prunes_are_counted(monkeypatch):
    """Every step notes the matvec prune before its admission and the next block's prune after it."""
    calls = []
    real = gf_primitives._CappedBasisProxy.note_step_cut

    def recording(self, weight):
        calls.append(weight)
        return real(self, weight)

    monkeypatch.setattr(gf_primitives._CappedBasisProxy, "note_step_cut", recording)
    history = _trickle()[3]["support_history"]
    assert len(history) * 2 - 1 <= len(calls) <= len(history) * 2 + 1, (len(calls), len(history))
    assert all(w >= 0.0 for w in calls)


def test_the_frozen_g_is_no_farther_from_the_exact_one_than_the_sparse_one(monkeypatch):
    frozen = _g(_trickle())
    monkeypatch.setenv("GF_SUPPORT_FREEZE", "0")
    plain = _g(_trickle())
    exact = _exact_sector_g()
    err_frozen = np.abs(frozen - exact).max()
    err_plain = np.abs(plain - exact).max()
    assert err_plain > 1e-7, "premise: the cutoff costs the sparse kernel something"
    assert err_frozen <= err_plain, (err_frozen, err_plain)


def test_it_is_on_by_default():
    assert config.GF_SUPPORT_FREEZE.default is True
    assert _trickle()[3]["support_frozen"]


def test_with_no_cutoff_it_freezes_only_on_closure():
    info = run_chain()[3]
    assert info["support_frozen"] and info["retained_size"] == SECTOR
    window = info["support_history"][len(info["support_history"]) - WINDOW :]
    assert all(leak == 0.0 and cut == 0.0 for leak, cut in window)


@pytest.mark.parametrize("reort", ["full", "partial"])
def test_it_needs_reort_none(reort):
    info = _trickle(reort=reort)[3]
    assert not info["support_frozen"] and info["support_history"] == []


def test_off_switch(monkeypatch):
    monkeypatch.setenv("GF_SUPPORT_FREEZE", "0")
    info = _trickle()[3]
    assert not info["support_frozen"] and info["support_history"] == []


def test_a_matrix_that_does_not_fit_undoes_the_freeze(monkeypatch):
    monkeypatch.setenv("GF_SUPPORT_FREEZE", "0")
    plain = _trickle()
    monkeypatch.setenv("GF_SUPPORT_FREEZE", "1")
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
    mine = (np.array(info["support_history"]).tobytes(), info["retained_size"], len(result[0]))
    assert len(set(comm.allgather(mine))) == 1
    assert _first_firing(info["support_history"]) == len(info["support_history"])
    np.testing.assert_allclose(_g(result), _g(_trickle()), atol=1e-6)


@pytest.mark.parametrize("name", ["GF_CLOSURE_FREEZE", "GF_STAGNATION_FREEZE", "GF_STAGNATION_WINDOW"])
def test_the_retired_freezes_warn_that_they_are_no_longer_read(name, monkeypatch):
    monkeypatch.setenv(name, "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.warn_retired_knobs()
    assert any(name in str(w.message) and "GF_SUPPORT_FREEZE" in str(w.message) for w in caught)
