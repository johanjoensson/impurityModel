r"""``GF_FREEZE_TRIAL_WINDOW``: freeze the support when a solve on it stops changing ``G``.

Every ``W`` sparse blocks the recurrence runs to convergence on the ``P H P`` of the support it has so far (a trial,
tightened to ``GF_FREEZE_TRIAL_TOL`` times the unit's tolerance) and compares ``G`` on the evaluation meshes with the
previous trial's; ``GF_FREEZE_TRIAL_CONSEC`` agreeing pairs within ``GF_FREEZE_BUDGET`` times the tolerance freeze the
support, and the accepted trial is the unit's answer. The contract these tests pin:

* it is off by default, and needs the unit's evaluation meshes;
* an accepted trial is the exact resolvent of ``P H P`` on the support it froze (the dense oracle), to the trial's
  tolerance;
* a rejected trial leaves the run as it was: with a budget nothing can meet, the result is the run without trials;
* a matrix that does not fit stops the trials, and the run finishes as if there had been none;
* every rank decides alike, at every rank count;
* the support it freezes at is smaller than the leak-versus-cut freeze's on a support that trickles, and the error
  stays inside the budget it was given.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import config, gf_solvers
from impurityModel.ed.gf_convergence import EvalMeshes
from impurityModel.ed.greens_function import calc_G
from impurityModel.test.basis.test_sparse_matrix_build import _det
from impurityModel.test.support.gf_chain_fixtures import DELTA, N_ORB, OMEGA, chain_operator, exact_chain_g, run_chain

TOL = 1e-4
CUTOFF = 1e-4


@pytest.fixture(autouse=True)
def _hermetic_knobs(monkeypatch):
    for name in (
        "GF_FREEZE_TRIAL_WINDOW",
        "GF_FREEZE_BUDGET",
        "GF_FREEZE_TRIAL_CONSEC",
        "GF_FREEZE_TRIAL_TOL",
        "GF_SUPPORT_TRACE",
        "APPLY_ROW_CHUNKS",
        *config.RETIRED_KNOBS,
    ):
        monkeypatch.delenv(name, raising=False)


def _meshes(tol=TOL):
    return EvalMeshes([OMEGA + 1j * DELTA], [tol])


def _trickle(comm=None, tol=TOL, **kw):
    return run_chain(comm=comm, op=chain_operator(seed=5), slater_min=CUTOFF, eval_meshes=_meshes(tol), **kw)


def _g(result):
    alphas, betas, r, _ = result
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)


def _on(monkeypatch, window=4, budget=0.1, consec=2, tol=0.1):
    monkeypatch.setenv("GF_FREEZE_TRIAL_WINDOW", str(window))
    monkeypatch.setenv("GF_FREEZE_BUDGET", str(budget))
    monkeypatch.setenv("GF_FREEZE_TRIAL_CONSEC", str(consec))
    monkeypatch.setenv("GF_FREEZE_TRIAL_TOL", str(tol))


def test_it_is_off_by_default():
    info = _trickle()[3]
    assert info["freeze_trials"] == [] and not info["trial_frozen"]


def test_it_needs_the_evaluation_meshes(monkeypatch):
    _on(monkeypatch)
    info = run_chain(op=chain_operator(seed=5), slater_min=CUTOFF)[3]
    assert info["freeze_trials"] == [] and not info["trial_frozen"]


def test_it_freezes_a_trickling_support_earlier_than_the_leak_versus_cut_rule(monkeypatch):
    shipped = _trickle()[3]
    assert shipped["support_frozen"] and not shipped["trial_frozen"], "premise: the shipped rule freezes this support"
    _on(monkeypatch)
    trial = _trickle()[3]
    assert trial["trial_frozen"] and trial["support_frozen"] and trial["csr_fallback"]
    assert trial["retained_size"] < shipped["retained_size"], (trial["retained_size"], shipped["retained_size"])
    assert trial["freeze_trials"][-1][3], "the last trial is the accepted one"
    assert not any(t[3] for t in trial["freeze_trials"][:-1])


def test_an_accepted_trial_is_the_exact_resolvent_of_php(monkeypatch):
    """The accepted trial IS the answer: its G equals the dense resolvent of H restricted to the frozen support."""
    # An easy-to-meet budget so the freeze happens early, and a trial converged to 1e-4 x the unit tolerance (1e-8)
    # so the comparison with the dense resolvent is limited by the trial, not by the stopping rule.
    _on(monkeypatch, budget=1e4, consec=1, tol=1e-4)
    result = _trickle()
    info = result[3]
    assert info["trial_frozen"]
    keys = list(info["proxy"].retained_keys())
    assert len(keys) == info["retained_size"]
    np.testing.assert_allclose(_g(result), exact_chain_g(keys, seed=5), atol=1e-6)


def test_a_budget_nothing_meets_leaves_the_run_as_it_was(monkeypatch):
    """Rejected trials throw their work away: the result is bit-for-bit the run without trials."""
    plain = _trickle()
    _on(monkeypatch, budget=0.0)
    trial = _trickle()
    assert trial[3]["freeze_trials"] and not any(t[3] for t in trial[3]["freeze_trials"])
    assert not trial[3]["trial_frozen"]
    np.testing.assert_allclose(np.asarray(trial[0]), np.asarray(plain[0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial[1]), np.asarray(plain[1]), rtol=1e-12, atol=1e-12)
    assert trial[3]["retained_size"] == plain[3]["retained_size"]


def test_a_matrix_that_does_not_fit_stops_the_trials(monkeypatch):
    """No matrix fits, so neither freeze can take the CSR: the run is the sparse run, with or without the trials."""
    asked = []

    def declining(*args, **kwargs):
        asked.append(1)
        return False

    monkeypatch.setattr(gf_solvers, "_frozen_csr_fits", declining)
    plain = _trickle()
    plain_asks = len(asked)
    _on(monkeypatch)
    asked.clear()
    trial = _trickle()
    assert trial[3]["freeze_trials"] == [] and not trial[3]["trial_frozen"]
    assert len(asked) <= plain_asks + 1, "the trials asked once and gave up for good"
    np.testing.assert_allclose(np.asarray(trial[0]), np.asarray(plain[0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(trial[1]), np.asarray(plain[1]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("window,consec", [(4, 2), (8, 1)])
def test_the_error_stays_inside_the_tolerance(monkeypatch, window, consec):
    """Against the exact resolvent of the whole sector, the frozen unit is within the unit's own tolerance -- the
    cutoff's error (slaterWeightMin) and the freeze's together. A tolerance below the cutoff's own error (1e-6 here)
    is one no support smaller than the closure can meet, and the trials then never accept; that is checked too."""
    exact_sector = exact_chain_g([_det(list(c)) for c in itertools.combinations(range(N_ORB), 7)], seed=5)
    _on(monkeypatch, window=window, consec=consec, budget=0.3, tol=0.1)
    result = _trickle(tol=TOL)
    assert result[3]["trial_frozen"]
    assert np.abs(_g(result) - exact_sector).max() / np.abs(exact_sector).max() < TOL
    tight = _trickle(tol=1e-9)
    assert not tight[3]["trial_frozen"], "a tolerance under the cutoff's error freezes nothing the cutoff does not"


@pytest.mark.mpi
def test_every_rank_decides_alike(monkeypatch):
    comm = MPI.COMM_WORLD
    _on(monkeypatch)
    result = _trickle(comm=comm)
    info = result[3]
    assert info["trial_frozen"]
    # the verdicts and the support sizes are replicated; the per-axis changes are rank-local bits and are left out
    mine = ([(t[0], t[1], t[3]) for t in info["freeze_trials"]], info["retained_size"], len(result[0]))
    assert len(set(map(repr, comm.allgather(mine)))) == 1
    serial = _trickle()
    assert info["retained_size"] == serial[3]["retained_size"]
    np.testing.assert_allclose(_g(result), _g(serial), atol=1e-6)
