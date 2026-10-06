"""``run_units_distributed`` under ``GF_SCHEDULER=queue``: ordering, both result paths, balance."""

import time

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import memory_estimate as me
from impurityModel.ed import work_queue
from impurityModel.ed.gf_units import run_units_distributed
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant

pytestmark = [
    pytest.mark.mpi,
    pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="the unit split only runs on more than one rank"),
]


@pytest.fixture(autouse=True)
def _queue_scheduler(monkeypatch):
    monkeypatch.setenv("GF_SCHEDULER", "queue")
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda c: 2**60)  # memory never cuts colors


def _basis_and_seed():
    comm = MPI.COMM_WORLD
    states = [b"\x80", b"\x40", b"\x20", b"\x10"]
    basis = Basis(
        {0: [[0, 1, 2, 3]]}, ({0: [[]]}, {0: [[]]}), initial_basis=states, comm=comm, truncation_threshold=100
    )
    psi = ManyBodyState.from_states(
        [ManyBodyState({SlaterDeterminant.from_bytes(states[0]): 1.0} if comm.rank == 0 else {}, width=1)]
    )
    return basis, psi


def _ran_on_split(comm):
    return MPI.Comm.Compare(comm, MPI.COMM_WORLD) != MPI.IDENT


@pytest.mark.parametrize("streaming", [False, True])
def test_queued_results_come_back_in_global_unit_order(streaming):
    basis, psi = _basis_and_seed()
    n_units = 3 * MPI.COMM_WORLD.size + 1
    weights = np.linspace(1.0, 2.0, n_units)  # dispatch order is the reverse of unit order

    def kernel(split_basis, u, seeds):
        assert _ran_on_split(split_basis.comm), "premise: the units ran on a split communicator"
        return ("unit", u)

    got = {}
    reduce_fn = got.__setitem__ if streaming else None
    results = run_units_distributed(basis, [[psi]] * n_units, weights, kernel, reduce_fn=reduce_fn)
    assert not work_queue._hosted, "the queue was not freed"
    if MPI.COMM_WORLD.rank == 0:
        if streaming:
            assert results is True
            results = [got[u] for u in range(n_units)]
        assert results == [("unit", u) for u in range(n_units)]
    else:
        assert results is None


def test_a_long_unit_does_not_hold_back_the_short_ones():
    """Equal predicted weights, one unit 1 s and the rest 0.03 s. A static round-robin would give
    the long unit's color its share of short units too; the queue gives them to the idle colors,
    so the color running the long unit runs nothing else. The kernel sleeps in slices and pokes
    the counter's progress, as the GF kernels do once per block -- without that, a counter host
    running the long unit would hold every other color's next fetch on MPICH."""
    comm = MPI.COMM_WORLD
    basis, psi = _basis_and_seed()
    n_short = 6
    walls = [1.0] + [0.03] * n_short

    def kernel(split_basis, u, seeds):
        end = time.perf_counter() + walls[u]
        while time.perf_counter() < end:
            time.sleep(0.005)
            work_queue.queue_progress()
        return (u, MPI.COMM_WORLD.rank, split_basis.comm.rank)

    results = run_units_distributed(basis, [[psi]] * len(walls), np.ones(len(walls)), kernel)
    if comm.rank == 0:
        ran_by = {}
        for u, world_rank, sub_rank in results:
            assert sub_rank == 0, "every result is reported by its color root"
            ran_by.setdefault(world_rank, []).append(u)
        long_color = next(r for r, units in ran_by.items() if 0 in units)
        assert ran_by[long_color] == [0], f"the long unit's color also ran {ran_by[long_color]}"
        assert sorted(u for units in ran_by.values() for u in units) == list(range(len(walls)))


def test_an_unknown_scheduler_is_rejected_on_every_rank(monkeypatch):
    monkeypatch.setenv("GF_SCHEDULER", "dynamic")
    basis, psi = _basis_and_seed()
    with pytest.raises(ValueError, match="GF_SCHEDULER"):
        run_units_distributed(basis, [[psi]] * 2, np.ones(2), lambda b, u, s: u)
