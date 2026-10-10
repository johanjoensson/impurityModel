"""The GF unit pull queue: ordering, replay, exactly-once hand-out, and host progress."""

import time

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import work_queue
from impurityModel.ed.work_queue import UnitQueue, queue_makespan, queue_order, queue_progress


def test_queue_order_is_heaviest_first_with_stable_ties():
    assert queue_order([1.0, 5.0, 3.0, 5.0]).tolist() == [1, 3, 2, 0]


def test_queue_makespan_is_list_scheduling():
    # Two colors: 4 -> c0, 3 -> c1, 2 -> c1 (free at 3), 1 -> c0 (free at 4) => both end at 5.
    assert queue_makespan([1.0, 2.0, 3.0, 4.0], [3, 2, 1, 0], 2) == 5.0
    # Light units first strand the heavy one at the end: 1,2 | 3 then 4 on the color free at 2.
    assert queue_makespan([1.0, 2.0, 3.0, 4.0], [0, 1, 2, 3], 2) == 6.0


def test_heavy_first_beats_a_static_split_of_bimodal_units():
    """The SrMnO3 shape: a few long units among many short ones, misjudged by a static packer."""
    walls = np.array([100.0] * 4 + [1.0] * 12)
    # A static packer that sees equal weights deals round-robin; with 4 colors two heavy units can
    # share a color. The queue in heavy-first order gives each heavy unit its own color.
    assert queue_makespan(walls, queue_order(walls), 4) == pytest.approx(103.0)


@pytest.mark.mpi
def test_every_index_is_handed_out_exactly_once():
    comm = MPI.COMM_WORLD
    n_units = 3 * comm.size + 1
    queue = UnitQueue(comm, n_units)
    try:
        mine = []
        while (k := queue.fetch()) < n_units:
            mine.append(k)
        queue.finish()
    finally:
        queue.free()
    every = [k for got in comm.allgather(mine) for k in got]
    assert sorted(every) == list(range(n_units))
    assert not work_queue._hosted, "the host's progress registration outlived the queue"


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs a rank other than the counter host")
def test_a_busy_host_that_calls_queue_progress_does_not_stall_fetches():
    """The host answers requests only when it enters MPI: without the poke a fetch waits out the
    host's whole compute stretch."""
    comm = MPI.COMM_WORLD
    busy = 1.5
    queue = UnitQueue(comm, 3 * comm.size, is_client=comm.rank != work_queue.HOST)
    try:
        comm.Barrier()
        if comm.rank == work_queue.HOST:
            start = time.perf_counter()
            a = np.random.default_rng(0).random((100, 100))
            while time.perf_counter() - start < busy:
                a = np.tanh(a @ a / 100.0)  # no MPI call of its own
                queue_progress()
        else:
            time.sleep(0.1)  # the host is inside its compute loop
            for _ in range(3):
                queue.fetch()
        waits = comm.gather(queue.max_wait, root=0)
        comm.Barrier()
    finally:
        queue.free()
    if comm.rank == 0:
        assert max(waits) < busy / 3, f"a fetch waited {max(waits):.2f} s on a host that pokes progress"


def test_queue_order_breaks_ties_on_the_secondary_key():
    # Units 0 and 2 share the largest primary key; the secondary key puts 2 first.
    assert queue_order([5.0, 1.0, 5.0], tiebreak=[1.0, 9.0, 3.0]).tolist() == [2, 0, 1]


class _NoOneSidedMPI:
    """``mpi4py.MPI`` with the RMA window class removed."""

    def __getattr__(self, name):
        if name == "Win":
            raise AssertionError("the unit queue made a one-sided MPI call")
        return getattr(MPI, name)


@pytest.mark.mpi
def test_the_queue_makes_no_one_sided_mpi_call(monkeypatch):
    """A cross-node ``Fetch_and_op`` failed on Intel MPI over Slingshot (cxi) on every color root of
    the second node, while one-node runs passed; the counter uses two-sided messages only."""
    monkeypatch.setattr(work_queue, "MPI", _NoOneSidedMPI())
    comm = MPI.COMM_WORLD
    queue = UnitQueue(comm, comm.size)
    try:
        while queue.fetch() < comm.size:
            pass
        queue.finish()
    finally:
        queue.free()


@pytest.mark.mpi
def test_a_host_that_runs_out_first_answers_the_others_before_its_next_collective():
    """The host takes its indices at once and heads for a collective while the others still fetch.
    ``finish`` keeps it answering; without it the others wait on a reply and the host on them."""
    comm = MPI.COMM_WORLD
    n_units = 2 * comm.size
    queue = UnitQueue(comm, n_units)
    try:
        mine = []
        while (k := queue.fetch()) < n_units:
            mine.append(k)
            if comm.rank != work_queue.HOST:
                time.sleep(0.02)
        queue.finish()
        every = [k for got in comm.allgather(mine) for k in got]
    finally:
        queue.free()
    assert sorted(every) == list(range(n_units))


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs a client other than the counter host")
def test_free_answers_a_client_still_fetching_when_every_rank_raises():
    """The host's own unit raised before a client's first request arrived. The client still needs an
    answer to reach its own unit, raise there, and join the collective free."""
    comm = MPI.COMM_WORLD
    queue = UnitQueue(comm, 100)
    try:
        if comm.rank != work_queue.HOST:
            time.sleep(0.2)  # the host is already in free
        queue.fetch()
        raise RuntimeError("the unit failed")
    except RuntimeError:
        pass
    finally:
        queue.free()
    assert not work_queue._hosted
