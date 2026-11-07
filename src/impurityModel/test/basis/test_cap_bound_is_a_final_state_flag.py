"""``truncation_report["cap_bound"]`` describes the *final* basis, ``cap_ever_hit`` the history.

A cap cycle that truncated nothing the wavefunction needed (a transient: the cap was reached,
the kept basis re-admitted, the discarded PT2 mass is zero) must not read as ``[cap-bound]``;
a cap that dropped weight or candidate mass must.
"""

import pytest
from mpi4py import MPI

from impurityModel.ed.eigensolvers import Within
from impurityModel.ed.groundstate import GS_DE2_MIN
from impurityModel.ed.truncation_flags import cap_ever_hit, is_cap_bound
from impurityModel.test.restrictions.test_cipsi_memory_budget_guard import (  # noqa: F401
    _fresh_warning_latch,
    _hamiltonian,
    _make_solver,
)

COMMS = [pytest.param(MPI.COMM_SELF, id="self"), pytest.param(MPI.COMM_WORLD, id="world", marks=pytest.mark.mpi)]


def _expand(comm, cap):
    solver = _make_solver(comm, truncation_threshold=cap)
    solver.basis.verbose = False
    solver.expand(_hamiltonian(), Within.thermal(solver.basis.tau), de2_min=GS_DE2_MIN)
    return solver


@pytest.mark.parametrize("comm", COMMS)
def test_a_binding_cap_is_cap_bound(comm, capfd):
    solver = _expand(comm, 6)
    report = solver.truncation_report
    assert cap_ever_hit(report) and is_cap_bound(report)
    assert report["cap_bound"] is True


@pytest.mark.parametrize("comm", COMMS)
def test_a_transient_cap_cycle_is_hit_but_not_bound(comm, capfd):
    solver = _expand(comm, 9)
    report = solver.truncation_report
    assert cap_ever_hit(report), "the cap cycle ran, so the history flag is set"
    assert not is_cap_bound(report), report
    assert report["cap_bound"] is False
    if comm.rank == 0:
        assert "gs-cap" not in capfd.readouterr().err


@pytest.mark.parametrize("comm", COMMS)
def test_the_flag_is_reset_by_each_expand(comm):
    solver = _expand(comm, 6)
    assert is_cap_bound(solver.truncation_report)
    solver.basis.truncation_threshold = 1000
    solver.expand(_hamiltonian(), Within.thermal(solver.basis.tau), de2_min=GS_DE2_MIN)
    assert not is_cap_bound(solver.truncation_report)


def test_predicates_on_absent_and_legacy_reports():
    assert not cap_ever_hit(None) and not is_cap_bound(None)
    assert is_cap_bound({"cap_hit": True}), "a pre-flag report keeps meaning cap-bound"


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs a second rank to disagree")
def test_the_flag_is_identical_on_every_rank():
    solver = _expand(MPI.COMM_WORLD, 6)
    flags = MPI.COMM_WORLD.allgather(bool(solver.truncation_report["cap_bound"]))
    assert len(set(flags)) == 1


def test_a_selection_on_the_snapshot_basis_marks_the_snapshot_cap_bound():
    """Restoring the best basis must bring back what the cap was measured to do to *that* basis.

    The snapshot is taken before the cycle's selection runs, so it first carries the previous cycle's flag;
    ``note_selection_on_snapshot`` folds in the selection that ran on the snapshotted basis itself.
    """
    import numpy as np

    from impurityModel.ed.cipsi_budget import CapRefiner

    solver = _make_solver(MPI.COMM_SELF, truncation_threshold=None)
    basis = solver.basis
    refiner = CapRefiner(e_tol=1e-9, max_cycles=5)
    refiner.observe(basis, -1.0, None, np.array([-1.0]), 1, cap_bound=False)
    refiner.note_selection_on_snapshot(True)
    basis.add_states([])  # the basis is mutated by later cycles
    refiner.observe(basis, -0.5, None, np.array([-0.5]), 2, cap_bound=False)  # worse: no new snapshot
    refiner.note_selection_on_snapshot(False)
    assert refiner._cap_bound is True
    # A selection after a cycle that did not snapshot must not leak into the older snapshot.
    refiner2 = CapRefiner(e_tol=1e-9, max_cycles=5)
    refiner2.observe(basis, -1.0, None, np.array([-1.0]), 1, cap_bound=False)
    refiner2.observe(basis, -0.5, None, np.array([-0.5]), 2, cap_bound=False)
    refiner2.note_selection_on_snapshot(True)
    assert refiner2._cap_bound is False
