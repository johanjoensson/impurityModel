"""The look-ahead half of ``CIPSISolver.expand``'s memory guard (``doc/plans/dc_smo_memory.md``,
round 6).

The after-the-fact trip-wire compares a high-water mark *after* the selection round that set it.
An expansion that admits everything grows its basis 5-10x per cycle and the selection round's
transient grows with it, so one cycle takes a run from under budget to OOM-killed: the SrMnO3
double-counting search read 2.4 GiB against a 2.5 GiB budget, then 5.8 GiB, then died. The
look-ahead bound instead measures each round's own transient (``VmHWM`` reset via
``/proc/self/clear_refs`` before the round) and caps *this* admission so that the *next* round,
on the basis it creates, is predicted to fit.

These tests fake the RSS samplers so the guard sees a transient proportional to the basis it ran
on -- the scaling the bound assumes and the crash log confirms -- on a toy system whose real memory
is far too small to trip anything.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import cipsi_solver
from impurityModel.ed.cipsi_solver import CIPSISolver, _memory_growth_bound
from impurityModel.ed.groundstate import GS_DE2_MIN
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3]]}, {0: [[4, 5]]})
N_SPIN_ORBITALS = 6
N_ELECTRONS = 3
MiB = 2**20


def _det(occupied):
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _make_solver(comm, truncation_threshold=None):
    basis = Basis(
        IMPURITY_ORBITALS,
        BATH_STATES,
        nominal_impurity_occ={0: 1},
        comm=comm,
        verbose=True,
        truncation_threshold=truncation_threshold,
    )
    basis.add_states([_det(range(N_ELECTRONS))])
    return CIPSISolver(basis)


def _hamiltonian():
    terms = {((i, "c"), (i, "a")): 0.3 * (i + 1) for i in range(N_SPIN_ORBITALS)}
    for a, b in ((0, 2), (1, 3), (2, 4), (3, 5), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = 0.15 + 0.05j
        terms[((b, "c"), (a, "a"))] = 0.15 - 0.05j
    return ManyBodyOperator(terms)


class _FakeMemory:
    """RSS samplers that report a selection-round transient of ``bytes_per_det`` per determinant
    of the basis the round runs on, above a flat ``floor``. Installed into ``cipsi_solver``'s
    namespace so only the guard's own samples are faked."""

    def __init__(self, monkeypatch, solver, bytes_per_det, floor=100 * MiB):
        self.solver = solver
        self.bytes_per_det = bytes_per_det
        self.floor = floor
        self.peak = floor
        self.resets = 0
        monkeypatch.setattr(cipsi_solver, "current_rss_bytes", lambda: self.floor)
        monkeypatch.setattr(cipsi_solver, "reset_peak_rss", self._reset)
        monkeypatch.setattr(cipsi_solver, "peak_rss_bytes", self._peak)

    def _reset(self):
        self.resets += 1
        # The round about to run enumerates the candidates of the *current* basis.
        self.peak = self.floor + self.bytes_per_det * self.solver.basis.size
        return True

    def _peak(self):
        return self.peak


# ---------------------------------------------------------------------------------------
# The bound's arithmetic, in isolation
# ---------------------------------------------------------------------------------------


def test_bound_scales_the_measured_transient_by_basis_and_width_growth():
    # A round on 1000 determinants with 10 references peaked 100 MiB above a 400 MiB RSS; the
    # next eigensolve will ask for 20. Budget 1000 MiB leaves 600 MiB of headroom, the transient
    # doubles with the width, so the next basis may be 3x this one: 2000 new determinants.
    affordable = _memory_growth_bound(1000 * MiB, 1000, p_now=10, p_next=20)
    assert affordable(100 * MiB, 400 * MiB) == 2000


def test_bound_returns_zero_when_even_the_same_size_does_not_fit():
    affordable = _memory_growth_bound(1000 * MiB, 1000, 10, 10)
    assert affordable(700 * MiB, 400 * MiB) == 0  # 400 + 700 > 1000: no growth at all
    assert affordable(100 * MiB, 1000 * MiB) == 0  # no headroom left


def test_bound_has_no_opinion_without_a_measurement():
    affordable = _memory_growth_bound(1000 * MiB, 1000, 10, 10)
    assert affordable(-1, 400 * MiB) is None
    assert affordable(0, 400 * MiB) is None


def test_bound_never_shrinks_the_width_ratio_below_one():
    # A shrinking manifold must not predict a *cheaper* next round.
    a = _memory_growth_bound(1000 * MiB, 1000, p_now=20, p_next=10)
    b = _memory_growth_bound(1000 * MiB, 1000, p_now=20, p_next=20)
    assert a(100 * MiB, 400 * MiB) == b(100 * MiB, 400 * MiB)


# ---------------------------------------------------------------------------------------
# Through expand
# ---------------------------------------------------------------------------------------


def _run(comm, monkeypatch, budget, bytes_per_det):
    H = _hamiltonian()
    solver = _make_solver(comm)
    fake = _FakeMemory(monkeypatch, solver, bytes_per_det)
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=budget)
    return solver, fake


def test_no_budget_never_measures_and_matches_the_unguarded_run(monkeypatch):
    H = _hamiltonian()
    ref = _make_solver(None)
    ref.expand(H, de2_min=GS_DE2_MIN, solver="trlm")

    solver = _make_solver(None)
    fake = _FakeMemory(monkeypatch, solver, bytes_per_det=1)
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=None)

    assert fake.resets == 0, "without a budget the guard must not even sample"
    assert set(solver.basis.local_basis) == set(ref.basis.local_basis)
    assert solver.truncation_report is None
    assert "memory_admit_cap" not in (solver.last_selection or {})


def test_generous_budget_measures_but_never_binds(monkeypatch):
    H = _hamiltonian()
    ref = _make_solver(None)
    ref.expand(H, de2_min=GS_DE2_MIN, solver="trlm")

    # 1 KiB per determinant on a basis of at most 20: the round never comes near 2^60.
    solver, fake = _run(None, monkeypatch, budget=2**60, bytes_per_det=1024)
    assert fake.resets > 0, "with a budget every round is measured"
    assert set(solver.basis.local_basis) == set(ref.basis.local_basis)
    assert solver.truncation_report is None
    assert not np.isfinite(solver.basis.truncation_threshold)
    sel = solver.last_selection
    assert sel["memory_admit_cap"] is not None and sel["memory_admit_cap"] >= sel["n_candidates"]
    assert sel["round_transient_bytes"] == 1024 * solver.basis.size or sel["n_candidates"] == 0


def test_tight_budget_stops_growth_before_the_predicted_overrun(monkeypatch):
    """A transient of 40 MiB per determinant above a 100 MiB floor against a 1 GiB budget: the
    bound predicts that a basis beyond ~23 determinants cannot run its selection round, so the
    expansion must adopt a cap there -- before any round would have exceeded the budget."""
    budget = 1024 * MiB
    solver, fake = _run(None, monkeypatch, budget=budget, bytes_per_det=40 * MiB)

    rep = solver.truncation_report
    assert rep is not None and rep["cap_hit"] and rep["memory_bound"]
    assert np.isfinite(solver.basis.truncation_threshold)
    assert solver.basis.size <= solver.basis.truncation_threshold
    # The prediction the guard acts on: every round that actually ran fit the budget.
    assert fake.peak <= budget, "a round ran whose faked transient exceeded the budget"
    # And it bound *ahead* of the overrun, not after it: the largest basis any round ran on is
    # within what the bound allowed from the round before.
    assert solver.basis.size < 1024 * MiB / (40 * MiB)


def test_the_look_ahead_never_loosens_a_caller_cap(monkeypatch):
    all_dets = [_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), N_ELECTRONS)]
    cap = 5
    H = _hamiltonian()
    solver = _make_solver(None, truncation_threshold=cap)
    solver.basis.clear()
    solver.basis.add_states([all_dets[0]])
    _FakeMemory(monkeypatch, solver, bytes_per_det=1024)
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=2**60)
    assert solver.basis.truncation_threshold == cap
    assert solver.basis.size <= cap


@pytest.mark.mpi
def test_tight_budget_binds_identically_on_every_rank(monkeypatch):
    comm = MPI.COMM_WORLD
    budget = 1024 * MiB
    solver, fake = _run(comm, monkeypatch, budget=budget, bytes_per_det=40 * MiB)
    rep = solver.truncation_report
    assert rep is not None and rep["memory_bound"]
    thresholds = comm.allgather(float(solver.basis.truncation_threshold))
    assert len(set(thresholds)) == 1, thresholds
    sizes = comm.allgather(int(solver.basis.size))
    assert len(set(sizes)) == 1, sizes
    assert fake.peak <= budget
