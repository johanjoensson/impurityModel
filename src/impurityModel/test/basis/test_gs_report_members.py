"""``measure_member`` + ``finalize_members`` against the one-shot ``report_ground_state`` over the merged members.

One member must reproduce the one-shot report bit for bit; several must agree to rounding (the union's weights
are rebuilt from the members', and the per-state tables follow the union's energy order).
"""

import numpy as np
import pytest

from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.gs_report import (
    finalize_members,
    measure_ground_state,
    measure_member,
    merge_measurements,
    report_ground_state,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.basis.test_calc_gs_ensemble import _hook, _setup
from impurityModel.test.support.comms import COMMS


def _numerically_equal(x, y, tol):
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_numerically_equal(x[k], y[k], tol) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_numerically_equal(p, q, tol) for p, q in zip(x, y))
    if isinstance(x, (float, np.floating, complex, np.complexfloating)):
        return abs(x - y) <= tol * max(1.0, abs(y))
    if isinstance(x, np.ndarray):
        return x.shape == y.shape and np.allclose(x, y, rtol=0, atol=tol)
    return x == y


def _without_zero_rows(stats):
    """A member does not list the configs of states it does not hold; the one-shot union lists them with weight 0."""
    if stats is None:
        return None
    stats = dict(stats)
    stats["per_state"] = [{**s, "rows": [r for r in s["rows"] if r["weight"] != 0.0]} for s in stats["per_state"]]
    return stats


def _holds_states(obj):
    if isinstance(obj, ManyBodyState):
        return True
    if isinstance(obj, dict):
        return any(_holds_states(v) for v in list(obj.keys()) + list(obj.values()))
    if isinstance(obj, (list, tuple)):
        return any(_holds_states(v) for v in obj)
    return False


def _redistributed(basis, psis):
    blk = basis.redistribute_block(psis if isinstance(psis, ManyBodyState) else ManyBodyState.from_states(list(psis)))
    blk.prune_rows(0.0)
    return blk


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("index, n_members, tol", [(20, 1, 0.0), (3, 2, 1e-10)])
def test_members_replayed_equal_the_one_shot_report(comm, index, n_members, tol, tmp_path):
    h_op, setup, bs, rot, _, _ = _setup(index, comm, "on")
    captured = []

    def hook(member, basis, es, psis, truncation, convergence):
        captured.append((member.total, basis, np.array(es), _redistributed(basis, psis).to_states()))
        return len(es)

    calc_gs(h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, stats_path=str(tmp_path / "gs.json"), visitor=_hook(hook))
    assert len(captured) == n_members
    tau = setup["tau"]
    common = dict(verbose=0, stats_path=None, gs_truncation_report=None, gs_convergence_report={"converged": True})

    merged = merge_measurements([measure_ground_state(b, e, p, total=t) for t, b, e, p in captured])
    _psis, es_ref, _b, rho_ref, info_ref = report_ground_state(h_op, *merged, tau, bs, rot, **common)

    members = []
    for k, (total, basis, es, psis) in enumerate(captured):
        blk = _redistributed(basis, psis)
        members.append(measure_member(h_op, basis, es, blk, tau, bs, rot, total=total, keep_basis=k == 0))
    for m in members:  # a measurement is small replicated per-state data: no state, no block, anywhere in it
        assert not _holds_states(m.tape)
    none, es_new, _b, rho_new, info_new = finalize_members(h_op, members, tau, bs, rot, **common)

    assert none is None
    assert _numerically_equal(es_new, es_ref, tol) and _numerically_equal(rho_new, rho_ref, tol)
    assert _numerically_equal(info_new["rhos"], info_ref["rhos"], tol)
    for key in ("correlation_diagnostics", "screening_diagnostics", "energy_decomposition", "static_susceptibilities"):
        assert _numerically_equal(info_new[key], info_ref[key], tol), key
    a, b = info_new["statistics"], info_ref["statistics"]
    assert (a is None) == (b is None)
    if a is not None:
        assert _numerically_equal(_without_zero_rows(a), _without_zero_rows(b), tol)
