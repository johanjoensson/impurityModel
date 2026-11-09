"""``calc_gs`` under ``mixed_valence``: the merged thermal density matrix is the grand-canonical one (MV 6)."""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.gs_report import ReportFiles
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
    out = calc_gs(
        h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, report_files=ReportFiles(json=str(tmp_path / "gs.json"))
    )
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
        h_op,
        setup,
        bs,
        rot,
        0,
        slaterWeightMin=1e-12,
        report_files=ReportFiles(json=str(tmp_path / "gs.json")),
        visitor=_hook(hook),
    )
    assert len(calls) == n_members and out[4]["member_results"] == widths
    assert not any(alive_at_hook)


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_a_refused_auto_ensemble_keeps_the_dominant_member_and_names_it(comm, tmp_path, monkeypatch, capsys):
    from impurityModel.ed import groundstate

    real_plan = groundstate.plan_ensemble
    plans = []

    def recording(solved, tau, **kw):
        plans.append(real_plan(solved, tau, **kw))
        return plans[-1]

    def refusing(member, truncation, convergence, comm):
        return f"the N_imp = {member.total} member's thermal solve is held by the measured-memory guard"

    monkeypatch.setattr(groundstate, "plan_ensemble", recording)
    monkeypatch.setattr(groundstate, "member_refusal", refusing)
    (_, _, _, _, info), _, _ = _run(3, comm, "auto", tmp_path)
    final = plans[-1]
    assert len(final.members) > 1 and "measured-memory guard" in info["ensemble"]["refused_reason"]
    assert len(info["ensemble"]["members"]) == 1
    out = capsys.readouterr().out
    if comm is None or comm.rank == 0:
        assert f"keeps the single N_imp = {final.members[0].total} sector" in out, out


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_a_member_whose_thermal_solve_is_held_by_the_memory_guard_is_not_merged(comm, tmp_path, capsys, monkeypatch):
    """H-MV11 (S4): at a cap of 10 the walk admits N = 1 and 2 unflagged, but both thermal solves are held.

    The window is solved per symmetry block there (each block fits the cap), which is complete; the blocks'
    verdict is forced cap-bound here, since what is under test is the ensemble's refusal of such a member."""
    from impurityModel.ed import sector_solver

    real = sector_solver._solve_window_per_label

    def cap_bound_blocks(*args, **kwargs):
        merged, solver, es, psis, info = real(*args, **kwargs)
        solver.truncation_report = dict(
            solver.truncation_report or {"cap_hit": True}, cap_bound=True, memory_bound=True
        )
        return merged, solver, es, psis, info

    monkeypatch.setattr(sector_solver, "_solve_window_per_label", cap_bound_blocks)
    h_op, setup, bs, rot, _, _ = _setup(3, comm, True)
    setup["truncation_threshold"] = 10
    hooks = []

    def hook(member, *_):
        hooks.append(member.total)

    run = lambda: calc_gs(  # noqa: E731
        h_op,
        setup,
        bs,
        rot,
        0,
        slaterWeightMin=1e-12,
        report_files=ReportFiles(json=str(tmp_path / "gs.json")),
        visitor=_hook(hook),
    )
    *_, info = run()
    ens = info["ensemble"]
    assert "held by the measured-memory guard" in ens["refused_reason"]
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


def _hook(function):
    """A visitor calling ``function(member, basis, es, psis, truncation, convergence)`` (the pre-visitor hook shape)."""

    class Visitor:
        def visit(self, member, state):
            return function(member, state.basis, state.es, state.states, state.truncation, state.convergence)

    return Visitor()


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
        (dict(truncation={"cap": 10}), False, False),  # S4: a cap-bound basis widens the interval, it does not refuse
        (dict(truncation={"cap": 10, "memory_bound": True}), True, False),  # only the memory guard refuses
        (dict(converged=False, residual_pt2=3e-3, limited_by="cap"), False, False),
        (dict(converged=False, residual_pt2=3e-9, residual_is_current=False), False, False),
        (dict(converged=False, residual_pt2=1e-9, limited_by="manifold_widened"), False, False),
    ],
    ids=["converged", "cap-bound", "memory-bound", "current-residual-large", "stale-residual", "manifold-widened"],
)
def test_only_a_bound_basis_or_a_current_large_residual_refuses_a_member(comm, state, refused, by_cap, capsys):
    """H-MV11 as narrowed by the owner: a stale residual or a widened manifold warns, it does not refuse."""
    from types import SimpleNamespace

    from impurityModel.ed.groundstate import member_refusal

    solver = _fake_member_state(**state).solver
    got = member_refusal(SimpleNamespace(total=2), solver.truncation_report, solver.convergence_report, comm)
    assert (got is not None) == refused
    if refused:
        assert "N_imp = 2" in got and "memory guard" in got
    elif state.get("converged") is False and (comm is None or comm.rank == 0):
        assert "is not converged" in capsys.readouterr().out


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

    def reversed_plan(solved, tau, **kw):
        plan = real(solved, tau, **kw)
        return plan if plan.needs else dataclasses.replace(plan, members=plan.members[::-1])

    monkeypatch.setattr(groundstate, "plan_ensemble", reversed_plan)
    h_op, setup, bs, rot, _, _ = _setup(3, comm, "on")
    hooks = []
    _, es, _, _, info = calc_gs(
        h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, report_files=ReportFiles(json=str(tmp_path / "gs2.json")),
        visitor=_hook(lambda member, *_: hooks.append(member.total) or member.total),
    )  # fmt: skip
    ens = info["ensemble"]
    e0 = [m["e0"] for m in ens["members"]]
    assert e0[0] == min(e0)
    assert info["member_results"] == [m["total"] for m in ens["members"]]  # the hook results follow the members
    assert hooks == [m["total"] for m in ens["members"]][::-1]  # solved in the reversed plan's order
    assert min(es) == min(ens["es"])


def _failing_member(monkeypatch, total):
    """The thermal check refuses the member with ``total`` electrons (as rank 0 would broadcast it)."""
    from impurityModel.ed import groundstate

    real = groundstate.member_refusal

    def check(member, truncation, convergence, comm):
        verdict = real(member, truncation, convergence, comm)
        return f"the N_imp = {total} member's thermal solve is injected" if member.total == total else verdict

    monkeypatch.setattr(groundstate, "member_refusal", check)


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_a_failing_later_member_cuts_the_ensemble_there_keeping_the_members_before_it(
    comm, tmp_path, monkeypatch, capsys
):
    """7b (owner decision): at tau 0.3 model 33 forms N = 0, 1, 2. When N = 2's thermal solve fails, N = 0 and 1 were
    already handed to the hook; they stay in the ensemble instead of being thrown away for the winner alone."""
    h_op, setup, bs, rot, _, _ = _setup(33, comm, "auto")
    setup["tau"] = 0.3
    _failing_member(monkeypatch, 2)
    hooks = []

    def hook(member, *_):
        hooks.append(member.total)
        return member.total

    *_, info = calc_gs(
        h_op,
        setup,
        bs,
        rot,
        0,
        slaterWeightMin=1e-12,
        report_files=ReportFiles(json=str(tmp_path / "gs.json")),
        visitor=_hook(hook),
    )
    ens = info["ensemble"]
    assert "injected" in ens["refused_reason"]
    assert sorted(m["total"] for m in ens["members"]) == [0, 1]
    assert hooks == [0, 1] and sorted(info["member_results"]) == [0, 1]
    assert set(ens["state_sector"]) == {0, 1}
    assert sum(m["weight"] for m in ens["members"]) == pytest.approx(1.0)
    if comm is None or comm.rank == 0:
        out = capsys.readouterr().out
        assert "keeps the N_imp = [0, 1] members solved before it" in out, out
        assert "neighbouring sector lies" not in out  # N = 1 is kept; N = 2 is no neighbour of N = 0


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_an_ensemble_cut_to_its_winner_still_warns_about_a_near_neighbour(comm, tmp_path, monkeypatch, capsys):
    """7c: the winner was solved as an ensemble member (no neighbour note); once the cut leaves it alone, the N = 1
    neighbour at weight ~0.39 is outside the ground state, and the single-sector note says so."""
    h_op, setup, bs, rot, _, _ = _setup(33, comm, "auto")
    setup["tau"] = 0.3
    _failing_member(monkeypatch, 1)
    *_, info = calc_gs(
        h_op, setup, bs, rot, 0, slaterWeightMin=1e-12, report_files=ReportFiles(json=str(tmp_path / "gs.json"))
    )
    assert [m["total"] for m in info["ensemble"]["members"]] == [0]
    if comm is None or comm.rank == 0:
        out = capsys.readouterr().out
        assert "keeps the single N_imp = 0 sector" in out, out
        assert "neighbouring sector lies within the thermal window (N_imp = 1:" in out, out


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("index, n_members", [(20, 1), (3, 2)])
def test_the_driver_gets_the_only_copy_of_the_thermal_states(comm, index, n_members, tmp_path, monkeypatch):
    """The raw block goes to the measure visitor alone; the driver works on converted ``states``, not on the block."""
    from impurityModel.ed import groundstate
    from impurityModel.ed.ManyBodyUtils import ManyBodyState

    taken, members, driven = [], [], []
    real_take = groundstate.MemberState.take_psis

    def spy(self):
        members.append(self)
        raw = real_take(self)
        taken.append(raw)
        return raw

    monkeypatch.setattr(groundstate.MemberState, "take_psis", spy)

    class Driver:
        def visit(self, member, state):
            driven.append(state)
            return len(state.states)

    h_op, setup, bs, rot, _, _ = _setup(index, comm, "on")
    psis, _es, _, _, info = calc_gs(
        h_op,
        setup,
        bs,
        rot,
        0,
        slaterWeightMin=1e-12,
        report_files=ReportFiles(json=str(tmp_path / "gs.json")),
        visitor=Driver(),
    )
    assert len(taken) == len(driven) == n_members
    assert all(m.psis is None for m in members)  # the member holds no reference to its raw block afterwards
    for raw, state in zip(taken, driven):
        assert not hasattr(state, "psis") and state.states is not raw
        assert isinstance(state.states, list) and all(isinstance(s, ManyBodyState) for s in state.states)
        assert len(state.states) == len(state.es)
    assert psis is None  # a driver consumed the states; none is held for the return
    assert info["member_results"] == [len(d.states) for d in driven]


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("index, n_members", [(20, 1), (3, 2)])
def test_driver_output_follows_the_ground_state_report(comm, index, n_members, tmp_path, capfd):
    """Progress lines first, then the ground-state report, then each driver's held-back text (headed when K > 1)."""

    class Driver:
        def visit(self, member, state):
            print(f"DRIVER-TEXT {member.total}")
            return 0

    h_op, setup, bs, rot, _, _ = _setup(index, comm, "on")
    calc_gs(
        h_op,
        setup,
        bs,
        rot,
        1,
        slaterWeightMin=1e-12,
        report_files=ReportFiles(json=str(tmp_path / "gs.json")),
        visitor=Driver(),
    )
    out = capfd.readouterr().out
    if (comm.rank if comm is not None else 0) != 0:
        assert "DRIVER-TEXT" in out and "E0 = " not in out
        return
    progress, report, driver = out.find("running the driver"), out.find("E0 = "), out.find("DRIVER-TEXT")
    assert 0 <= progress < report < driver
    assert out.count("DRIVER-TEXT") == n_members
    assert ("--- N_imp =" in out) == (n_members > 1)


class _Boom(RuntimeError):
    pass


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_a_failing_member_prints_the_reports_gathered_so_far(comm, tmp_path, capfd):
    """A driver failing on member 2 prints a PARTIAL report and member 1's text, writes no JSON, re-raises.

    Every rank raises: with only one failing rank the others would wait in the solve's next collective, which no
    handler can prevent. The handler itself communicates nothing.
    """

    class Driver:
        def __init__(self):
            self.n = 0

        def visit(self, member, state):
            self.n += 1
            print(f"DRIVER-TEXT {member.total}")
            if self.n == 2:
                raise _Boom("member two broke")
            return 0

    stats = tmp_path / "gs.json"
    h_op, setup, bs, rot, _, _ = _setup(3, comm, "on")
    with pytest.raises(_Boom):
        calc_gs(
            h_op, setup, bs, rot, 1, slaterWeightMin=1e-12, report_files=ReportFiles(json=str(stats)), visitor=Driver()
        )
    out = capfd.readouterr().out
    assert not stats.exists()
    if (comm.rank if comm is not None else 0) == 0:
        assert "PARTIAL GROUND-STATE REPORT" in out and "_Boom: member two broke" in out
        assert "E0 = " in out and out.count("DRIVER-TEXT") == 2
        assert out.find("PARTIAL") < out.find("DRIVER-TEXT")


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_a_failing_second_solve_prints_the_first_members_reports(comm, tmp_path, capfd, monkeypatch):
    """A failure before member 2's visit (in its solve) still prints member 1's report and driver text."""
    from impurityModel.ed import groundstate

    class Driver:
        def visit(self, member, state):
            print(f"DRIVER-TEXT {member.total}")
            return 0

    real = groundstate._MeasureVisitor.visit
    seen = []

    def visit(self, member, state):
        out = real(self, member, state)
        seen.append(member.total)
        if len(seen) == 1:
            raise _Boom("between members")  # after member 1 is measured and driven, before member 2
        return out

    monkeypatch.setattr(groundstate._MeasureVisitor, "visit", visit)
    h_op, setup, bs, rot, _, _ = _setup(3, comm, "on")
    with pytest.raises(_Boom):
        calc_gs(
            h_op,
            setup,
            bs,
            rot,
            1,
            slaterWeightMin=1e-12,
            report_files=ReportFiles(json=str(tmp_path / "gs.json")),
            visitor=Driver(),
        )
    out = capfd.readouterr().out
    if (comm.rank if comm is not None else 0) == 0:
        assert "PARTIAL GROUND-STATE REPORT" in out and "E0 = " in out and "DRIVER-TEXT" in out


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
def test_driver_text_survives_a_failing_report(comm, tmp_path, capfd, monkeypatch):
    """A failure after every member is done (in the report) must not lose what the drivers printed."""
    from impurityModel.ed import groundstate

    class Driver:
        def visit(self, member, state):
            print(f"DRIVER-TEXT {member.total}")
            return 0

    def broken(*args, **kwargs):
        raise _Boom("report broke")

    monkeypatch.setattr(groundstate, "finalize_members", broken)
    h_op, setup, bs, rot, _, _ = _setup(3, comm, "on")
    with pytest.raises(_Boom):
        calc_gs(
            h_op,
            setup,
            bs,
            rot,
            1,
            slaterWeightMin=1e-12,
            report_files=ReportFiles(json=str(tmp_path / "gs.json")),
            visitor=Driver(),
        )
    out = capfd.readouterr().out
    assert out.count("DRIVER-TEXT") == 2
    if (comm.rank if comm is not None else 0) == 0:
        assert "GROUND-STATE REPORT FAILED" in out and "report broke" in out
