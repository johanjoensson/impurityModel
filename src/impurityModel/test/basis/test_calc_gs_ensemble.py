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
def test_a_failed_statistics_write_on_rank_0_is_raised_on_every_rank(tmp_path, monkeypatch):
    """Rank 0 alone writes the statistics JSON; a failure there left the other ranks in the next collective
    (the GS8 numpy-bool hang at -n 2). It must be raised on every rank."""
    from impurityModel.ed import gs_report

    def failing(stats, path):
        raise TypeError("not serialisable")

    monkeypatch.setattr(gs_report, "save_gs_statistics", failing)
    with pytest.raises((TypeError, RuntimeError)):
        _run(20, MPI.COMM_WORLD, "off", tmp_path)
