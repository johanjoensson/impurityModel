"""Attribute a self-energy causality failure and, when the support settle can be blamed, solve its units again once.

``selfenergy._settle_retry`` is exercised against a fake :class:`GFRun` and a faked self-energy verdict, so each branch
of the decision is pinned without a many-body solve: the gate (no settled unit, bath pole, held unit, estimate below
10% of the violation, retry off), the one closure-only re-solve (cured; not cured; rejected), and the collective shape
(the plan is broadcast, every rank enters ``resolve`` or none does). The real-solver end of the chain is
``test_gf_plan_solve_assemble`` (``GFRun.resolve``) and the replay of the NiO archive.
"""

import dataclasses

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_diagnostics as gd
from impurityModel.ed import selfenergy
from impurityModel.ed.utils import Reporter

DELTA = 0.01
W = np.linspace(-1.0, 1.0, 201)
IW = 1j * np.pi * 0.01 * (2 * np.arange(10) + 1)
BLOCKS = [[0, 1], [2, 3, 4]]


def _unit(unit, block, settled=True, held=False, retained=1000, leak_window=1e-6, leak_end=1e-7, n_blocks=500):
    return gd.UnitRecord(
        unit=unit, block=block, side="removal", states=(0,), e_minus_e0=(0.0,), boltzmann=(1.0,), seed_weight=1.0,
        retained=retained, settled=settled, settle_step=40 if settled else None, n_blocks=n_blocks,
        frontier_over_swm=5.0 if settled else float("nan"), ratio=10.0 if settled else float("nan"),
        window=16 if settled else 0, cap_hit=held, memory_frozen=False, leak_window=leak_window,
        frontier_end_over_swm=2.0, leak_end=leak_end,
    )  # fmt: skip


def _record(val=3e-3, ratio=7e-3, freq=100, g0=-0.5 + 0j, gi=-0.4 + 0j):
    """A full :func:`sigma.causality_report` record (the message formatter reads every field)."""
    return {
        "n_viol": 3, "n_total": 1000, "frac": 3e-3, "worst_val": val, "worst_freq_index": freq, "worst_orbital": 0,
        "worst_orbital_ratio": 0, "worst_ratio": ratio, "windows": [], "g0_inv_at_worst": g0, "ginv_at_worst": gi,
    }  # fmt: skip


class _Report:
    def __init__(self, units):
        self.units = units


class _SB:
    h0_solve = None
    total_impurity_orbitals = None
    sum_bath_states = None


class _FakeRun:
    def __init__(self, accepted=True, cures=True, reason=""):
        self.calls = []
        self.accepted, self.cures, self.reason = accepted, cures, reason
        self.cured = False

    def resolve(self, subset, ratio):
        self.calls.append((list(subset), ratio))
        if self.accepted and self.cures:
            self.cured = True
        before = [(u, 1000, 3000 if self.accepted else 1000, "" if self.accepted else self.reason) for u in subset]
        return (self.accepted, before, "NEW_M", "NEW_R", None) if self.accepted else (False, before, None, None, None)


@pytest.fixture
def fake_sigma(monkeypatch):
    """``_sigma_collect`` reports block 0 violating (real axis) until the fake run has cured it."""
    state = {"run": None, "violate": True, "matsubara": False}

    def collect(mesh, gss, *, delta, estimator, solver_basis, cluster_label, blocks):
        if gss is None:
            return None, None, {}
        sigma = [np.zeros((len(mesh), len(b), len(b)), dtype=complex) for b in blocks]
        cured = state["run"] is not None and state["run"].cured
        if mesh is W and state["violate"] and not cured and not state["matsubara"]:
            return sigma, None, {0: state["record"]}
        if mesh is IW and state["violate"] and state["matsubara"]:
            return sigma, None, {0: state["record_m"]}
        return sigma, None, {}

    monkeypatch.setattr(selfenergy, "_sigma_collect", collect)
    monkeypatch.setattr(selfenergy, "save_Greens_function", lambda *a, **k: None)
    # two bath levels, -0.6 coupled to every impurity orbital, -0.3 coupled to none (a foreign level)
    v_full = np.zeros((2, 5))
    v_full[0, :] = 0.2
    monkeypatch.setattr(selfenergy, "get_hcorr_v_hbath", lambda *a, **k: (None, v_full, None, np.diag([-0.6, -0.3])))
    state["record"] = _record()
    state["record_m"] = _record(freq=2)
    return state


def _call(state, run, units, comm=None, retry=True, record=None):
    state["run"] = run
    if record is not None:
        state["record"] = record
    return selfenergy._settle_retry(
        run=run, gf_report=_Report(units), family_matsubara=["M"], family_realaxis=["R"], iw=IW, w=W, delta=DELTA,
        estimator=None, sb=_SB(), blocks=BLOCKS, comm=comm, report=Reporter(0, 0 if comm is None else comm.rank),
        cluster_label="x", retry=retry, gf_seconds=12.0, tol=1e-3,
    )  # fmt: skip


def test_no_violation_changes_nothing_and_never_solves(fake_sigma):
    fake_sigma["violate"] = False
    run = _FakeRun()
    fm, fr, note, retried = _call(fake_sigma, run, [_unit(0, 0)])
    assert (fm, fr, note, retried) == (["M"], ["R"], "", False) and run.calls == []


def test_a_settle_that_can_be_blamed_is_solved_again_once_closure_only_and_can_cure(fake_sigma, capsys):
    run = _FakeRun()
    fm, fr, note, retried = _call(fake_sigma, run, [_unit(0, 0), _unit(1, 0), _unit(2, 1)])
    assert retried and run.calls == [([0, 1], 0.0)], "only the failing block's settled units, ratio 0"
    assert (fm, fr) == ("NEW_M", "NEW_R")
    assert "sigma_settle_retry: cured" in note and "set gf_settle_ratio 1" in note
    out = capsys.readouterr().out
    assert "Re-solving 2 settled unit(s) closure-only" in out and "unit 0: 1,000 -> 3,000" in out and "12.0 s" in out
    assert "sig-*-before-retry" in out and "gf_settle_retry off to fail fast" in out


def test_a_retry_that_does_not_cure_says_the_settle_is_not_the_cause(fake_sigma):
    run = _FakeRun(cures=False)
    fm, fr, note, retried = _call(fake_sigma, run, [_unit(0, 0)])
    assert retried and (fm, fr) == ("NEW_M", "NEW_R"), "the closure-only G is kept: it is the more exact one"
    assert "the settle is not the cause" in note and "closure-only result is kept" in note
    assert "doc/known_issues.md" in note


def test_a_retry_the_memory_budget_refused_leaves_the_first_pass_standing(fake_sigma):
    run = _FakeRun(accepted=False, reason="capped or held by the memory guard")
    fm, fr, note, retried = _call(fake_sigma, run, [_unit(0, 0)])
    assert retried and (fm, fr) == (["M"], ["R"])
    assert "not taken" in note and "memory guard" in note


@pytest.mark.parametrize(
    "units,retry,expect",
    [
        ([_unit(0, 0, settled=False)], True, "no unit of this block settled"),
        ([_unit(0, 0, settled=False, held=True)], True, "froze at the determinant cap"),
        ([_unit(0, 0, settled=False, held=True)], True, "swap refinement (gf_refine_rounds)"),
        ([_unit(0, 0, held=True)], True, "held by the cap or the memory guard"),
        ([_unit(0, 0)], False, "gf_settle_retry is off"),
    ],
)
def test_the_gate_does_not_spend_a_re_solve_where_the_settle_cannot_be_the_cause(
    fake_sigma, units, retry, expect, capsys
):
    run = _FakeRun()
    fm, fr, note, retried = _call(fake_sigma, run, units, retry=retry)
    assert not retried and run.calls == [] and (fm, fr) == (["M"], ["R"])
    assert expect in note
    assert "worst Im Sigma" in capsys.readouterr().out, "the attribution is printed whether or not it re-solves"


def test_a_matsubara_only_failure_is_never_re_solved(fake_sigma):
    fake_sigma["matsubara"] = True
    run = _FakeRun()
    _fm, _fr, note, retried = _call(fake_sigma, run, [_unit(0, 0)])
    assert not retried and run.calls == [] and "Matsubara-only failure" in note


def test_units_already_closure_only_are_not_re_solved_again(fake_sigma):
    run = _FakeRun()
    closed = dataclasses.replace(_unit(0, 0), ratio=0.0)
    _fm, _fr, note, retried = _call(fake_sigma, run, [closed])
    assert not retried and run.calls == [] and "already ran closure-only" in note


def test_a_bath_pole_is_reported_but_does_not_veto_the_re_solve(fake_sigma):
    """The worst point on the coupled level at -0.6 with G0^-1 dominated by it: reported as such, and still
    re-solved, because |G|^-2 amplifies the settle's error most there."""
    idx = int(np.argmin(np.abs(W + 0.6)))
    record = _record(freq=idx, gi=complex(1e-4, -0.4))
    run = _FakeRun()
    _fm, _fr, note, retried = _call(fake_sigma, run, [_unit(0, 0)], record=record)
    assert retried and run.calls == [([0], 0.0)]
    assert "nearest coupled bath level -0.6" in note and "dominated by that pole" in note


def test_a_foreign_bath_level_is_not_named_as_the_blocks_pole(fake_sigma):
    """-0.3 is not coupled to the block: the nearest COUPLED level (-0.6) is named even at w = -0.3."""
    idx = int(np.argmin(np.abs(W + 0.3)))
    _fm, _fr, note, _retried = _call(fake_sigma, _FakeRun(), [_unit(0, 0)], record=_record(freq=idx))
    assert "nearest coupled bath level -0.6" in note


def test_the_retry_is_off_by_default_in_calc_selfenergy():
    from impurityModel.ed.model import SolverOptions

    assert SolverOptions().gf_settle_retry is None
    # calc_selfenergy turns None into off: the decision lives in one expression, pinned here
    import inspect

    src = inspect.getsource(selfenergy.calc_selfenergy)
    assert "settle_retry = gf_settle_retry is True and gf_settle is not False" in src


@pytest.mark.mpi
def test_every_rank_enters_the_re_solve_or_none_does(fake_sigma):
    comm = MPI.COMM_WORLD
    run = _FakeRun()
    # only rank 0 holds the families and the report (as after get_Greens_function); the plan is broadcast
    fam_m, fam_r = (["M"], ["R"]) if comm.rank == 0 else (None, None)
    fake_sigma["run"] = run
    out = selfenergy._settle_retry(
        run=run, gf_report=_Report([_unit(0, 0)]) if comm.rank == 0 else None, family_matsubara=fam_m,
        family_realaxis=fam_r, iw=IW, w=W, delta=DELTA, estimator=None, sb=_SB(), blocks=BLOCKS, comm=comm,
        report=Reporter(0, comm.rank), cluster_label="x", retry=True, gf_seconds=1.0, tol=1e-3,
    )  # fmt: skip
    assert out[3] is True and run.calls == [([0], 0.0)], f"rank {comm.rank}"
    assert len(set(comm.allgather(out[3]))) == 1


def test_a_failing_self_energy_stops_the_run_with_the_attribution_in_the_message(monkeypatch, capsys):
    """Through ``calc_selfenergy``: an acausal verdict on the real axis is attributed (printed and appended to the
    exception), and nothing is re-solved when no unit settled. The verdict is forced; the solve is the real one."""
    from impurityModel.ed.sigma import UnphysicalGreensFunctionError
    from impurityModel.test.gf.test_selfenergy_end_to_end import _basis_and_solver, _meshes, _siam_model

    real = selfenergy._sigma_collect

    def forced(mesh, gss, **kw):
        sigma, components, reports = real(mesh, gss, **kw)
        if sigma is not None and np.isrealobj(mesh):  # the real axis
            reports = {0: _record(freq=10)}
        return sigma, components, reports

    monkeypatch.setattr(selfenergy, "_sigma_collect", forced)
    monkeypatch.setattr(selfenergy, "save_Greens_function", lambda *a, **k: None)
    basis, solver = _basis_and_solver()
    with pytest.raises(UnphysicalGreensFunctionError) as err:
        selfenergy.calc_selfenergy(_siam_model(), _meshes(), basis, solver, comm=None, verbosity=0)
    text = str(err.value)
    assert "Real frequency self-energy" in text and "positive imaginary part" in text
    assert "support settle: no unit of this block settled" in text
    assert "support settle: no unit" in capsys.readouterr().out


def test_the_default_run_is_unchanged_when_the_self_energy_is_causal():
    from impurityModel.test.gf.test_selfenergy_end_to_end import _basis_and_solver, _meshes, _siam_model

    basis, solver = _basis_and_solver()
    result = selfenergy.calc_selfenergy(_siam_model(), _meshes(), basis, solver, comm=None, verbosity=0)
    assert np.all(np.isfinite(result["sigma_real"])) and np.all(np.isfinite(result["sigma"]))
