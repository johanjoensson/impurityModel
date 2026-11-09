"""``measure_member`` + ``finalize_members`` against the one-shot ``report_ground_state`` over the merged members.

One member must reproduce the one-shot report bit for bit; several must agree to rounding (the union's weights
are rebuilt from the members', and the per-state tables follow the union's energy order).
"""

import json

import numpy as np
import pytest

from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.gs_report import (
    finalize_members,
    measure_member,
    report_ground_state,
    sharpen_for_report,
)
from impurityModel.ed.gs_statistics import _NumpyEncoder
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.basis.test_calc_gs_ensemble import _hook, _setup
from impurityModel.test.support.comms import COMMS


def _merged(captured, comm):
    """The oracle: the members' states as one list ascending in rank 0's energies (one member passes through)."""
    if len(captured) == 1:
        _, basis, es, psis = captured[0]
        return basis, es, psis
    es = np.concatenate([e for _, _, e, _ in captured])
    es = es if comm is None else comm.bcast(es, root=0)
    states = [s for _, _, _, ps in captured for s in ps]
    order = np.argsort(es, kind="stable")
    return captured[0][1], es[order], [states[k] for k in order]


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
    if n_members > 1:  # the union order must really permute the concatenated members
        union = np.concatenate([e for _, _, e, _ in captured])
        assert not np.array_equal(np.argsort(union, kind="stable"), np.arange(len(union)))
    tau = setup["tau"]
    common = dict(verbose=0, stats_path=None, gs_truncation_report=None, gs_convergence_report={"converged": True})

    merged = _merged(captured, comm)
    _psis, es_ref, _b, rho_ref, info_ref = report_ground_state(h_op, *merged, tau, bs, rot, **common)

    members = []
    for k, (total, basis, es, psis) in enumerate(captured):
        blk = _redistributed(basis, psis)
        blk, mb = sharpen_for_report(h_op, basis, es, blk, rot)  # as the visitor does, before measuring
        members.append(
            measure_member(h_op, basis, es, blk, tau, bs, rot, total=total, keep_basis=k == 0, manifold_basis=mb)
        )
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
        expected = [
            {
                "n_imp": int(total),
                "max_holes": int(bnd.max_holes),
                "max_electrons": int(bnd.max_electrons),
                "max_total": int(bnd.max_total),
                "cap_bound": bool(bnd.cap_bound),
            }
            for total, basis, _, _ in captured
            if (bnd := getattr(basis, "excitation_bounds", None)) is not None
        ]
        # The replay reports one record per member, from the captured bases. The one-shot oracle sees only the lead
        # member's basis, so it can be compared directly only when there is a single member.
        assert (a.pop("excitation_bounds", None) or []) == expected
        if n_members > 1:
            b.pop("excitation_bounds", None)
        else:
            assert b.pop("excitation_bounds", None) == expected
        rep_a, rep_b = a.pop("report"), b.pop("report")
        assert rep_a["schema_version"] == 4 and rep_b["schema_version"] == 4
        json.dumps(rep_a, cls=_NumpyEncoder)  # the model is JSON-serialisable
        # manifolds and per-state rows are the same data, replayed or one-shot (the one-shot oracle has no
        # per-member sector, so ``sectors``/``n_imp_sector`` are compared only for the replay)
        for key in ("e0", "tau", "n_states", "degeneracy_tol", "thermal"):
            assert _numerically_equal(rep_a[key], rep_b[key], max(tol, 1e-12)), (key, rep_a[key], rep_b[key])
        drop = ("sectors", "n_imp_sector")
        for key in ("manifolds", "members"):
            strip = [{k: v for k, v in row.items() if k not in drop} for row in rep_a[key]]
            ref = [{k: v for k, v in row.items() if k not in drop} for row in rep_b[key]]
            assert _numerically_equal(strip, ref, max(tol, 1e-12)), key
        if n_members > 1:
            assert {m["n_imp_sector"] for m in rep_a["members"]} == {int(t) for t, _, _, _ in captured}
        assert sum(m["g"] for m in rep_a["manifolds"]) == len(es_new)
        assert _numerically_equal(_without_zero_rows(a), _without_zero_rows(b), tol)


def test_a_taped_failure_does_not_keep_the_states_alive():
    """The tape outlives the block (it is held until calc_gs returns); a taped exception must not pin it."""
    import gc
    import weakref

    from impurityModel.ed.gs_report import _LiveSource

    class Block:
        pass

    class Boom(RuntimeError):
        pass

    block = Block()
    src = object.__new__(_LiveSource)
    src.psis, src.tape = block, {}

    def compute():
        held = src.psis  # a frame that references the block, as manifold_observable_values' does
        raise Boom(f"failed on {type(held).__name__}")

    with pytest.raises(Boom):
        src.part("k", compute)
    ref = weakref.ref(block)
    del block, src.psis
    gc.collect()
    assert ref() is None
    kind, exc = src.tape["k"]
    assert kind == "err" and isinstance(exc, Boom) and exc.__traceback__ is None and "failed on Block" in str(exc)


def test_driver_text_is_held_back_but_warnings_go_to_stderr_at_once(capsys):
    from impurityModel.ed.groundstate import _HeldBack, _replay_driver_output

    buf, empty = _HeldBack(), _HeldBack()
    buf.write("plain line\nWARNING: unconverged\npartial")
    err = capsys.readouterr().err
    assert "WARNING: unconverged" in err and "plain" not in err
    buf.write(" line two\n")
    _replay_driver_output([(1, buf), (2, empty)], True)
    out = capsys.readouterr().out
    assert out.startswith("--- N_imp = 1 ---\nplain line\nWARNING: unconverged\npartial line two\n")
    assert "N_imp = 2" not in out  # a member that wrote nothing gets no header
