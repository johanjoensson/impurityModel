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
    calls = {"n": 0}

    def fake_peak_rss():
        calls["n"] += 1
        return 0 if calls["n"] <= 2 else budget * 2

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
