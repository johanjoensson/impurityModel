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
    queue = UnitQueue(comm)
    try:
        mine = []
        while (k := queue.fetch()) < n_units:
            mine.append(k)
    finally:
        queue.free()
    every = [k for got in comm.allgather(mine) for k in got]
    assert sorted(every) == list(range(n_units))
    assert not work_queue._hosted, "the host's progress registration outlived the queue"


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs a rank other than the counter host")
def test_a_busy_host_that_calls_queue_progress_does_not_stall_fetches():
    """Without the poke, MPICH (one node, default) and Open MPI over TCP held the fetch for the
    host's whole compute stretch. Under Open MPI's shared-memory transport the fetch never stalls,
    so this test only discriminates on MPICH -- the CI's MPI."""
    comm = MPI.COMM_WORLD
    busy = 1.5
    queue = UnitQueue(comm)
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
