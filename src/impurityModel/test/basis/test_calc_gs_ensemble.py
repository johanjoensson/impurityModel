"""``calc_gs`` under ``mixed_valence``: the merged thermal density matrix is the grand-canonical one (MV 6)."""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.groundstate import calc_gs
from impurityModel.test.support.comms import COMMS
from impurityModel.test.support.gs_oracle import exact_spectra, pinned_model

TAU = 0.15


def _setup(index, comm, mixed_valence):
    h_op, imp, bath, N0, n_orb, n_val, _ = pinned_model(20261016, index)
    n_imp = len(imp[0][0])
    block_structure = BlockStructure(
        blocks=[list(range(n_imp))],
        identical_blocks=[[0]],
        transposed_blocks=[[]],
        particle_hole_blocks=[[]],
        particle_hole_transposed_blocks=[[]],
        inequivalent_blocks=[0],
    )
    setup = {
        "impurity_orbitals": imp,
        "bath_states": bath,
        "N0": N0,
        "tau": TAU,
        "chain_restrict": False,
        "dense_cutoff": 1000,
        "comm": comm,
        "truncation_threshold": np.inf,
        "mixed_valence": mixed_valence,
    }
    return h_op, setup, block_structure, np.eye(n_imp, dtype=complex), n_orb, n_val


def _run(index, comm, mixed_valence, tmp_path):
    h_op, setup, bs, rot, n_orb, n_val = _setup(index, comm, mixed_valence)
    out = calc_gs(h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, stats_path=str(tmp_path / "gs.json"))
    return out, n_orb, n_val


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_a_single_member_ensemble_is_bit_identical_to_off(comm, tmp_path):
    (_psis0, es0, _, rho0, _info0), _, _ = _run(20, comm, "off", tmp_path)
    (_psis1, es1, _, rho1, info1), _, _ = _run(20, comm, "auto", tmp_path)
    assert np.array_equal(es0, es1) and np.array_equal(rho0, rho1)
    assert len(info1["ensemble"]["members"]) == 1 and info1["ensemble"]["members"][0]["weight"] == 1.0


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("index, members", [(3, [1, 2]), (33, [0, 1])])
def test_the_merged_density_matrix_carries_the_grand_canonical_electron_count(comm, index, members, tmp_path):
    (psis, es, _basis, rho, info), n_orb, n_val = _run(index, comm, "on", tmp_path)
    ens = info["ensemble"]
    assert sorted(m["total"] for m in ens["members"]) == members
    h_op, *_ = _setup(index, comm, "on")
    spectra = exact_spectra(h_op, n_orb)
    z = {t: np.exp(-spectra[n_val + t] / TAU).sum() for t in members}
    exact = sum(z[t] * (n_val + t) for t in members) / sum(z.values())
    assert np.trace(rho).real == pytest.approx(exact, abs=5e-3)
    # the dominant member's own states are what the tuple returns; the winner alone would be an integer count
    dominant = ens["members"][0]
    assert len(es) == dominant["n_states"] and len(psis) == len(es)
    assert abs((n_val + dominant["total"]) - exact) > 0.05
    assert sum(m["weight"] for m in ens["members"]) == pytest.approx(1.0)
    assert len(ens["state_sector"]) == sum(m["n_states"] for m in ens["members"])
    assert (comm or MPI.COMM_SELF).allgather(float(np.trace(rho).real)) == [float(np.trace(rho).real)] * (
        comm.size if comm else 1
    )


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("index, n_members", [(20, 1), (3, 2)])
def test_the_hook_runs_after_the_solver_is_released(comm, index, n_members, tmp_path, monkeypatch):
    """A member's solver (and its Krylov store) must be gone before the hook builds its own workspace."""
    import gc
    import weakref

    from impurityModel.ed import groundstate

    solvers = []
    real = groundstate.refined_truncation_report

    def spy(solver, walk_report):
        solvers.append(weakref.ref(solver))
        return real(solver, walk_report)

    monkeypatch.setattr(groundstate, "refined_truncation_report", spy)
    alive_at_hook, calls, widths = [], [], []

    def hook(member, basis, es, psis, truncation, convergence):
        gc.collect()
        alive_at_hook.append(solvers[len(calls)]() is not None)
        calls.append(member.total)
        widths.append(len(es))
        return len(es)

    h_op, setup, bs, rot, _, _ = _setup(index, comm, "on")
    out = calc_gs(
        h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, stats_path=str(tmp_path / "gs.json"), member_hook=hook
    )
    assert len(calls) == n_members and out[4]["member_results"] == widths
    assert not any(alive_at_hook)


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_a_refused_auto_ensemble_keeps_the_dominant_member_and_names_it(comm, tmp_path, monkeypatch, capsys):
    from impurityModel.ed import groundstate

    real = groundstate.plan_ensemble
    plans = []

    def refusing(solved, tau, mode, **kw):
        plan = real(solved, tau, mode, **{**kw, "max_error": -1.0})
        plans.append(plan)
        return plan

    monkeypatch.setattr(groundstate, "plan_ensemble", refusing)
    (_, _, _, _, info), _, _ = _run(3, comm, "auto", tmp_path)
    final = plans[-1]
    assert final.refused_reason is not None and len(final.members) == 1
    assert len(info["ensemble"]["members"]) == 1
    out = capsys.readouterr().out
    if comm is None or comm.rank == 0:
        assert f"keeps the single N_imp = {final.members[0].total} sector" in out, out


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("mode", ["auto", "on"])
def test_a_member_whose_thermal_solve_is_cap_bound_is_not_merged(comm, mode, tmp_path, capsys):
    """H-MV11: at a cap of 10 the walk admits N = 1 and 2 unflagged, but both thermal solves are cap-bound."""
    h_op, setup, bs, rot, _, _ = _setup(3, comm, mode)
    setup["truncation_threshold"] = 10
    hooks = []

    def hook(member, *_):
        hooks.append(member.total)

    run = lambda: calc_gs(  # noqa: E731
        h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, stats_path=str(tmp_path / "gs.json"), member_hook=hook
    )
    if mode == "on":
        with pytest.raises(ValueError, match="thermal solve is bound by the determinant cap"):
            run()
        return
    *_, info = run()
    ens = info["ensemble"]
    assert "thermal solve is bound by the determinant cap" in ens["refused_reason"]
    assert [m["total"] for m in ens["members"]] == [1] and ens["members"][0]["weight"] == 1.0
    assert hooks == [1] and len(info["member_results"]) == 1
    assert set(ens["state_sector"]) == {1}
    if comm is None or comm.rank == 0:
        assert "keeps the single N_imp = 1 sector" in capsys.readouterr().out


@pytest.mark.mpi
def test_a_failed_statistics_write_on_rank_0_is_raised_on_every_rank(tmp_path, monkeypatch):
    """Rank 0 alone writes the statistics JSON; a failure there left the other ranks in the next collective
    (the GS8 numpy-bool hang at -n 2). It must be raised on every rank."""
    from impurityModel.ed import gs_report

    def failing(stats, path):
        raise TypeError("not serialisable")

    monkeypatch.setattr(gs_report, "save_gs_statistics", failing)
    with pytest.raises((TypeError, RuntimeError)):
        _run(20, MPI.COMM_WORLD, "off", tmp_path)


def _fake_member_state(truncation=None, **convergence):
    from types import SimpleNamespace

    report = {"converged": True, "residual_pt2": 1e-9, "residual_is_current": True, "limited_by": None, **convergence}
    return SimpleNamespace(solver=SimpleNamespace(truncation_report=truncation, convergence_report=report))


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize(
    "state, refused, by_cap",
    [
        (dict(), False, False),
        (dict(truncation={"cap": 10}), True, True),
        (dict(truncation={"cap": 10, "memory_bound": True}), True, False),  # the memory guard is not a cap verdict
        (dict(converged=False, residual_pt2=3e-3, limited_by="cap"), True, False),
        (dict(converged=False, residual_pt2=3e-9, residual_is_current=False), False, False),
        (dict(converged=False, residual_pt2=1e-9, limited_by="manifold_widened"), False, False),
    ],
    ids=["converged", "cap-bound", "memory-bound", "current-residual-large", "stale-residual", "manifold-widened"],
)
def test_only_a_bound_basis_or_a_current_large_residual_refuses_a_member(comm, state, refused, by_cap, capsys):
    """H-MV11 as narrowed by the owner: a stale residual or a widened manifold warns, it does not refuse."""
    from types import SimpleNamespace

    from impurityModel.ed.groundstate import _thermal_member_defect

    got = _thermal_member_defect(SimpleNamespace(total=2), _fake_member_state(**state), comm)
    assert (got is not None) == refused
    if refused:
        assert got[1] == by_cap and "N_imp = 2" in got[0]
    elif state.get("converged") is False and (comm is None or comm.rank == 0):
        assert "is not certified" in capsys.readouterr().out


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("index", [3, 33])
def test_the_union_is_ascending_labelled_and_the_dominant_member_holds_the_ground(comm, index, tmp_path):
    """H-MV4/MV6/MV13: rhos, the union energies and their sector labels line up; the union is ascending (es[0] is
    the global ground state); members[0] is the realised lowest and the returned states are all its own."""
    (psis, es, _basis, _rho, info), _, _ = _run(index, comm, "on", tmp_path)
    ens = info["ensemble"]
    union = np.asarray(ens["es"])
    assert len(union) == len(ens["state_sector"]) == len(info["rhos"]) > len(es)
    assert np.all(np.diff(union) >= 0)
    e0 = [m["e0"] for m in ens["members"]]
    assert e0[0] == min(e0) == union[0]
    dominant = ens["members"][0]["total"]
    mine = union[[k for k, total in enumerate(ens["state_sector"]) if total == dominant]]
    assert np.array_equal(np.asarray(es), mine) and len(psis) == len(es)
    assert {m["total"] for m in ens["members"]} == set(ens["state_sector"])


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_the_dominant_member_is_the_realised_lowest_whatever_the_plans_order(comm, tmp_path, monkeypatch):
    """H-MV6: the plan is walk-ordered. With it reversed, the realised lowest member is solved second; calc_gs must
    still put it first, its hook result with it, and hand back its states."""
    import dataclasses

    from impurityModel.ed import groundstate

    real = groundstate.plan_ensemble

    def reversed_plan(solved, tau, mode, **kw):
        plan = real(solved, tau, mode, **kw)
        return plan if plan.needs else dataclasses.replace(plan, members=plan.members[::-1])

    monkeypatch.setattr(groundstate, "plan_ensemble", reversed_plan)
    h_op, setup, bs, rot, _, _ = _setup(3, comm, "on")
    hooks = []
    _, es, _, _, info = calc_gs(
        h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, stats_path=str(tmp_path / "gs2.json"),
        member_hook=lambda member, *_: hooks.append(member.total) or member.total,
    )  # fmt: skip
    ens = info["ensemble"]
    e0 = [m["e0"] for m in ens["members"]]
    assert e0[0] == min(e0)
    assert info["member_results"] == [m["total"] for m in ens["members"]]  # the hook results follow the members
    assert hooks == [m["total"] for m in ens["members"]][::-1]  # solved in the reversed plan's order
    assert min(es) == min(ens["es"])


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs a second rank whose energies differ in the last bits")
def test_the_union_order_is_rank_zeros_whatever_the_local_energies():
    """Review of H-MV13: the eigensolver's energies are rank-local; a near-degenerate pair sorted locally would
    permute the columns differently on each rank, and every collective over them would then pair wrong states."""
    from types import SimpleNamespace

    from impurityModel.ed.gs_report import union_order

    comm = MPI.COMM_WORLD
    tie = -1.0 + (1e-15 if comm.rank == 1 else 0.0)  # rank 1 sees the pair in the other order
    measurements = [SimpleNamespace(es=np.array([-1.0, 0.3])), SimpleNamespace(es=np.array([tie, 0.5]))]
    order, es = union_order(measurements, comm)
    assert comm.allgather(order.tolist()) == [order.tolist()] * comm.size
    assert np.array_equal(es, comm.bcast(es, root=0))


def _failing_member(monkeypatch, total):
    """The thermal check refuses the member with ``total`` electrons (as rank 0 would broadcast it)."""
    from impurityModel.ed import groundstate

    real = groundstate._thermal_member_defect

    def check(member, ground_state, comm):
        verdict = real(member, ground_state, comm)
        return (f"the N_imp = {total} member's thermal solve is injected", False) if member.total == total else verdict

    monkeypatch.setattr(groundstate, "_thermal_member_defect", check)


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_an_ensemble_cut_to_its_winner_still_warns_about_a_near_neighbour(comm, tmp_path, monkeypatch, capsys):
    """7c: the winner was solved as an ensemble member (no neighbour note); once the cut leaves it alone, the N = 1
    neighbour at weight ~0.39 is outside the ground state, and the single-sector note says so."""
    h_op, setup, bs, rot, _, _ = _setup(33, comm, "auto")
    setup["tau"] = 0.3
    _failing_member(monkeypatch, 1)
    *_, info = calc_gs(h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, stats_path=str(tmp_path / "gs.json"))
    assert [m["total"] for m in info["ensemble"]["members"]] == [0]
    if comm is None or comm.rank == 0:
        out = capsys.readouterr().out
        assert "keeps the single N_imp = 0 sector" in out, out
        assert "neighbouring sector lies within the thermal window (N_imp = 1:" in out, out
