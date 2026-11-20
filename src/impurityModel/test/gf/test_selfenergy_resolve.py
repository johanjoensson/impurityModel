"""``selfenergy._resolve_failing_blocks``: the one-shot re-solve of a block that fails the self-energy check.

The judge is faked (``_sigma_collect``), so the mechanics are pinned without a many-body solve: only a failing block
with an uncapped unit is re-solved, with a tighter grow budget and more rounds, once; its families and units are
spliced in; nothing is re-solved when pruning or refinement is off; every rank makes the same call.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_diagnostics as gd
from impurityModel.ed import selfenergy
from impurityModel.ed.gf_refinement import GROW_TOL, RefineSpec
from impurityModel.ed.utils import Reporter

W = np.linspace(-1.0, 1.0, 11)
BLOCKS = [[0], [1], [2]]


def _unit(block, cap_hit=False):
    return gd.UnitRecord(
        unit=0, block=block, side="removal", states=(0,), e_minus_e0=(0.0,), boltzmann=(1.0,), seed_weight=1.0,
        retained=10, n_blocks=5, cap_hit=cap_hit, memory_frozen=False,
    )  # fmt: skip


class _Report:
    def __init__(self, units):
        self.units = units


def _record():
    return {"worst_ratio": 1.0}


@pytest.fixture
def fake_judge(monkeypatch):
    """Blocks 0 and 1 fail on the real axis; whatever family entry is "new" is causal."""

    def collect(mesh, gss, *, delta, estimator, solver_basis, cluster_label, blocks):
        if gss is None:
            return None, None, {}
        return None, None, {i: _record() for i, g in enumerate(gss) if mesh is W and g == "old" and i < 2}

    monkeypatch.setattr(selfenergy, "_sigma_collect", collect)


def _judge():
    return dict(iw=None, w=W, delta=0.01, estimator=None, sb=None, blocks=BLOCKS, cluster_label="x", tol=1e-3)


def _run(refine, prune, units, comm=None, calls=None):
    comm_rank0 = comm is None or comm.rank == 0
    real = ["old", "old", "old"] if comm_rank0 else None
    solved = (None, real, _Report(units) if comm_rank0 else None)

    def solve_gf(blocks, tight):
        calls.append((blocks, tight))
        n = len(blocks)
        return (
            None,
            (["new"] * n if comm_rank0 else None),
            (_Report([_unit(k) for k in range(n)]) if comm_rank0 else None),
        )

    out = selfenergy._resolve_failing_blocks(
        solve_gf, refine, prune, solved, judge=_judge(), comm=comm, report=Reporter(0, 0)
    )
    return out


def test_only_a_failing_block_with_an_uncapped_unit_is_re_solved_once_and_spliced_in(fake_judge):
    calls = []
    refine = RefineSpec(rounds=3)
    units = [_unit(0), _unit(1, cap_hit=True), _unit(2)]
    _m, real, report = _run(refine, 1e-3, units, calls=calls)
    assert [b for b, _t in calls] == [[BLOCKS[0]]], "block 1 is capped, block 2 passes: one call, one block"
    tight = calls[0][1]
    assert tight.grow_tol == pytest.approx(0.1 * GROW_TOL) and tight.rounds == 6
    assert real == ["new", "old", "old"]
    assert sorted((u.block, u.cap_hit) for u in report.units) == [(0, False), (1, True), (2, False)]


@pytest.mark.parametrize("refine, prune", [(RefineSpec(rounds=0), 1e-3), (RefineSpec(rounds=3), 0.0), (None, 1e-3)])
def test_nothing_is_re_solved_when_a_tighter_selection_cannot_change_anything(fake_judge, refine, prune):
    calls = []
    _run(refine, prune, [_unit(0), _unit(1)], calls=calls)
    assert calls == []


def test_a_re_solve_with_the_same_budget_would_change_nothing_by_construction():
    """The tight spec differs from the original only in the budget and the rounds."""
    from dataclasses import replace

    base = RefineSpec(rounds=3, tol=1e-4, swap=0.2)
    tight = replace(base, grow_tol=0.1 * GROW_TOL, rounds=6)
    assert (tight.tol, tight.swap, tight.stop_rtol) == (base.tol, base.swap, base.stop_rtol)


@pytest.mark.mpi
def test_every_rank_makes_the_same_call(fake_judge):
    comm = MPI.COMM_WORLD
    calls = []
    _run(RefineSpec(rounds=3), 1e-3, [_unit(0), _unit(1)], comm=comm, calls=calls)
    assert [b for b, _t in calls] == [[BLOCKS[0], BLOCKS[1]]], "rank 0's verdict, on every rank"
