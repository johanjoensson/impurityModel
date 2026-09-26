"""``CIPSISolver.expand``'s ``memory_budget_bytes`` guard (Phase 4, ``doc/plans/dc_smo_memory.md``).

An uncapped expansion (``basis.truncation_threshold`` left at ``inf``) has no way to stop
growing before a kernel OOM kill, which is uncatchable and is what actually happened to the
SrMnO3 double-counting search this guard exists for. ``memory_budget_bytes`` is a measured
trip-wire, not a formula: it compares the expansion's own already-sampled peak RSS against a
budget and, the first time that budget is reached, retroactively adopts a fixed-budget cap at
the current basis size -- handing off to the existing, independently-tested fixed-budget CIPSI
machinery rather than inventing new truncation logic.

``memory_budget_bytes=None`` (the default) must be an exact no-op: every other test in this
suite calls ``expand`` without it, so this is what keeps the guard's existence from being a
silent behaviour change.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.groundstate import GS_DE2_MIN
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3]]}, {0: [[4, 5]]})
N_SPIN_ORBITALS = 6
N_ELECTRONS = 3


@pytest.fixture(autouse=True)
def _fresh_warning_latch():
    """The user-cap memory warning prints once per calculation; each test is one."""
    from impurityModel.ed.memory_estimate import reset_user_cap_memory_warnings

    reset_user_cap_memory_warnings()
    yield
    reset_user_cap_memory_warnings()


def _det(occupied):
    """SlaterDeterminant with the given orbitals occupied (MSB-first bit convention)."""
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
    seed = _det(range(N_ELECTRONS))
    basis.add_states([seed])
    return CIPSISolver(basis)


def _hamiltonian():
    terms = {((i, "c"), (i, "a")): 0.3 * (i + 1) for i in range(N_SPIN_ORBITALS)}
    for a, b in ((0, 2), (1, 3), (2, 4), (3, 5), (0, 4), (1, 5)):
        terms[((a, "c"), (b, "a"))] = 0.15 + 0.05j
        terms[((b, "c"), (a, "a"))] = 0.15 - 0.05j
    return ManyBodyOperator(terms)


def test_memory_budget_none_matches_no_argument_at_all_serial():
    """The explicit `None` default must be bit-for-bit identical to omitting the argument."""
    H = _hamiltonian()
    solver_a = _make_solver(None)
    solver_a.expand(H, de2_min=GS_DE2_MIN, solver="trlm")

    solver_b = _make_solver(None)
    solver_b.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=None)

    assert set(solver_a.basis.local_basis) == set(solver_b.basis.local_basis)
    assert solver_a.truncation_report is None
    assert solver_b.truncation_report is None
    assert not np.isfinite(solver_b.basis.truncation_threshold)


def test_impossible_budget_adopts_a_cap_at_the_current_basis_size():
    """A budget of 1 byte is reached on the very first diagnostic sample (process VmHWM alone
    is always > 1 B), so the guard must fire immediately and the expansion must finish capped."""
    H = _hamiltonian()
    solver = _make_solver(None)
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=1)

    assert np.isfinite(solver.basis.truncation_threshold), "the guard must have adopted a cap"
    assert solver.truncation_report is not None, "the adopted cap must actually have bound"
    assert solver.basis.size <= solver.basis.truncation_threshold


def test_huge_budget_never_fires_and_matches_the_unguarded_run():
    """A budget far above anything this toy system could reach must be a no-op, exactly like
    `memory_budget_bytes=None` -- the guard's *presence* must not change behaviour, only its
    triggering should."""
    H = _hamiltonian()
    solver_a = _make_solver(None)
    solver_a.expand(H, de2_min=GS_DE2_MIN, solver="trlm")

    solver_b = _make_solver(None)
    solver_b.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=2**60)

    assert set(solver_a.basis.local_basis) == set(solver_b.basis.local_basis)
    assert solver_b.truncation_report is None
    assert not np.isfinite(solver_b.basis.truncation_threshold)


def test_the_guard_never_loosens_a_caller_cap():
    """A caller's finite cap is an instruction, and a budget that is never reached must leave it
    exactly alone -- neither raised nor lowered."""
    H = _hamiltonian()
    all_dets = [_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), N_ELECTRONS)]

    solver_a = _make_solver(None, truncation_threshold=len(all_dets))
    solver_a.basis.clear()
    solver_a.basis.add_states([all_dets[0]])
    solver_a.expand(H, de2_min=GS_DE2_MIN, solver="trlm")

    solver_b = _make_solver(None, truncation_threshold=len(all_dets))
    solver_b.basis.clear()
    solver_b.basis.add_states([all_dets[0]])
    solver_b.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=1 << 62)

    assert solver_a.basis.truncation_threshold == solver_b.basis.truncation_threshold == len(all_dets)
    assert set(solver_a.basis.local_basis) == set(solver_b.basis.local_basis)


def test_the_guard_tightens_a_cap_that_memory_cannot_afford():
    """**Deliberate behaviour change.** This replaces an earlier test asserting the guard must
    never engage once any finite cap is set (`not capped` gated it).

    That rule made the guard useless for the run it exists for. Production set
    ``truncation_threshold=119,555,328`` -- a number produced by
    ``memory_estimate.suggest_truncation_threshold``, not by a human -- and was OOM-killed at
    949,834 determinants, **0.79% of its own cap**. The cap never bound, but its mere existence
    made ``capped`` true and skipped the memory check entirely. The estimator behind that number is
    measured to under-predict by 372-1028x at 256 ranks (``doc/plans/dc_smo_memory.md``), so a cap
    derived from it is a target, whereas the memory budget is a physical constraint.

    The guard therefore now fires whatever the cap, and only ever **tightens** it -- the
    never-loosen invariant is kept by ``test_the_guard_never_loosens_a_caller_cap`` above.

    That is ``memory_policy="tighten"`` (the default), the policy for a cap derived from memory.
    A cap the **user** set is final and runs under ``"warn"`` instead -- see
    ``test_warn_mode_keeps_a_user_cap_and_only_warns`` below.
    """
    H = _hamiltonian()
    all_dets = [_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), N_ELECTRONS)]
    absurd_cap = 10**6  # finite, and far beyond anything this basis could reach: production's shape

    solver = _make_solver(None, truncation_threshold=absurd_cap)
    solver.basis.clear()
    solver.basis.add_states([all_dets[0]])
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=1)

    assert solver.basis.truncation_threshold < absurd_cap, "an unaffordable cap must be tightened"
    assert solver.basis.size <= solver.basis.truncation_threshold


def test_the_guard_fires_once_and_does_not_ratchet_the_cap_down(monkeypatch):
    """``peak_rss`` is a high-water mark: once it exceeds the budget it stays above it forever, so
    without a latch the guard would re-fire every cycle and walk the threshold down toward the seed.

    A budget of 1 byte cannot test this -- it fires at cycle 0 while the basis is still the seed, so
    firing once and firing repeatedly are indistinguishable. Instead the RSS reading is driven
    directly: below budget for the first two cycles, far above it from then on.
    """
    from impurityModel.ed import cipsi_solver as _cs

    budget = 1 << 40
    # Keyed on cycles (one eigensolve each), not on calls to `peak_rss_bytes`: a cycle samples the
    # high-water mark more than once (after the eigensolve, in the selection round, at the
    # trip-wire), so counting calls would move the "first two cycles" boundary.
    cycles = {"n": 0}
    real_eigenvectors = _cs.CIPSISolver.get_eigenvectors

    def counting_eigenvectors(self, *args, **kwargs):
        cycles["n"] += 1
        return real_eigenvectors(self, *args, **kwargs)

    def fake_peak_rss():
        return 0 if cycles["n"] <= 2 else budget * 2

    monkeypatch.setattr(_cs.CIPSISolver, "get_eigenvectors", counting_eigenvectors)
    monkeypatch.setattr(_cs, "peak_rss_bytes", fake_peak_rss)

    H = _hamiltonian()
    all_dets = [_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), N_ELECTRONS)]
    solver = _make_solver(None, truncation_threshold=10**6)
    solver.basis.clear()
    solver.basis.add_states([all_dets[0]])
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=budget)

    adopted = solver.basis.truncation_threshold
    assert adopted < 10**6, "the unaffordable cap should have been tightened"
    # Re-firing would have dragged the threshold down with the basis on every later cycle.
    assert adopted > 1, f"guard ratcheted the cap down to {adopted}"
    assert solver.basis.size <= adopted


@pytest.mark.mpi
def test_impossible_budget_adopts_a_cap_at_the_current_basis_size_mpi():
    """Same guard, distributed -- run at -n 2 and -n 3. `Basis.size` and the MAX-allreduced
    VmHWM are both already global at the point the guard reads them, so every rank must reach
    the same adopted cap."""
    comm = MPI.COMM_WORLD
    H = _hamiltonian()
    solver = _make_solver(comm)
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=1)

    assert np.isfinite(solver.basis.truncation_threshold)
    thresholds = comm.allgather(solver.basis.truncation_threshold)
    assert all(t == thresholds[0] for t in thresholds)


def test_warn_mode_keeps_a_user_cap_and_only_warns(capfd):
    """A cap the user set is final: at a budget every sample exceeds, ``memory_policy="warn"``
    must leave the cap, the admissions and the truncation report exactly as an unguarded run
    has them, and say so -- on stderr as well as stdout, at any verbosity."""
    H = _hamiltonian()
    all_dets = [_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), N_ELECTRONS)]
    user_cap = 10**6

    reference = _make_solver(None, truncation_threshold=user_cap)
    reference.basis.clear()
    reference.basis.add_states([all_dets[0]])
    reference.expand(H, de2_min=GS_DE2_MIN, solver="trlm")

    solver = _make_solver(None, truncation_threshold=user_cap)
    solver.basis.verbose = False
    solver.basis.clear()
    solver.basis.add_states([all_dets[0]])
    capfd.readouterr()
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=1, memory_policy="warn")
    out, err = capfd.readouterr()

    assert solver.basis.truncation_threshold == user_cap
    assert set(solver.basis.local_basis) == set(reference.basis.local_basis)
    assert solver.truncation_report is None, "warn mode must not report a memory-bound cap"
    assert solver.memory_warning is not None and solver.memory_warning["cap"] == user_cap
    assert "WARNING determinant cap" in err
    assert "WARNING determinant cap" in out
    assert err.count("WARNING determinant cap") == 1, "the warning is latched once per expansion"


def test_warn_mode_is_a_no_op_under_budget():
    """No budget exceeded, no warning and nothing recorded."""
    H = _hamiltonian()
    solver = _make_solver(None, truncation_threshold=10**6)
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=2**60, memory_policy="warn")
    assert solver.memory_warning is None
    assert solver.truncation_report is None


def test_unknown_memory_policy_is_rejected():
    with pytest.raises(ValueError, match="memory_policy"):
        _make_solver(None).expand(_hamiltonian(), memory_policy="shrink")


def test_groundstate_maps_cap_provenance_to_memory_policy():
    """Only a finite cap the user set runs warn-only; auto and unlimited keep the guard."""
    from impurityModel.ed.groundstate import _memory_policy
    from impurityModel.ed.memory_estimate import CapPolicy

    assert _memory_policy(None) == "tighten"
    assert _memory_policy(np.inf) == "tighten"
    assert _memory_policy(5000) == "warn"
    assert _memory_policy(CapPolicy(gs=5000, gf=5000, from_memory=True)) == "tighten"
    assert _memory_policy(CapPolicy(gs=5000, gf=5000, from_memory=False)) == "warn"


@pytest.mark.mpi
def test_warn_mode_keeps_a_user_cap_mpi():
    """Warn mode's latch rides on the same replicated condition as the trip-wire, so every rank
    must keep the cap and record the warning together -- run at -n 2 and -n 3."""
    comm = MPI.COMM_WORLD
    H = _hamiltonian()
    solver = _make_solver(comm, truncation_threshold=10**6)
    solver.expand(H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=1, memory_policy="warn")

    assert solver.basis.truncation_threshold == 10**6
    flags = comm.allgather(solver.memory_warning is not None)
    assert all(flags), flags


def test_the_trip_wire_sees_a_peak_that_happens_during_the_eigensolve(monkeypatch):
    """The selection round resets the high-water mark to measure its own transient; a spike that
    lives only inside the eigensolve (at scale ~95% of an expansion's cost) was wiped by that reset
    before the trip-wire looked. Modelled directly: the fake high-water mark is huge only between an
    eigensolve and the next reset."""
    from impurityModel.ed import cipsi_solver as _cs

    budget = 1 << 40
    state = {"spiked": False}
    real_eigenvectors = _cs.CIPSISolver.get_eigenvectors

    def spiking_eigenvectors(self, *args, **kwargs):
        result = real_eigenvectors(self, *args, **kwargs)
        state["spiked"] = True
        return result

    def fake_reset():
        state["spiked"] = False
        return True

    monkeypatch.setattr(_cs.CIPSISolver, "get_eigenvectors", spiking_eigenvectors)
    monkeypatch.setattr(_cs, "reset_peak_rss", fake_reset)
    monkeypatch.setattr(_cs, "peak_rss_bytes", lambda: budget * 2 if state["spiked"] else 0)

    solver = _make_solver(None)
    solver.expand(_hamiltonian(), de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=budget)
    assert solver.truncation_report is not None and solver.truncation_report["memory_bound"]


def test_the_user_cap_warning_prints_once_per_calculation(capfd):
    """A double-counting search runs dozens of expansions; one warning per calculation, counted,
    and a new calculation (a driver resolving its cap) warns again."""
    from impurityModel.ed.memory_estimate import reset_user_cap_memory_warnings, resolve_cap_policy

    H = _hamiltonian()
    for _ in range(3):
        _make_solver(None, truncation_threshold=10**6).expand(
            H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=1, memory_policy="warn"
        )
    _out, err = capfd.readouterr()
    assert err.count("WARNING determinant cap") == 1
    resolve_cap_policy(10**6, N_SPIN_ORBITALS, log="never")  # the next calculation starts
    _make_solver(None, truncation_threshold=10**6).expand(
        H, de2_min=GS_DE2_MIN, solver="trlm", memory_budget_bytes=1, memory_policy="warn"
    )
    _out, err = capfd.readouterr()
    assert err.count("WARNING determinant cap") == 1
    assert reset_user_cap_memory_warnings() == 1
