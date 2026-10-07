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


def test_unit_sector_dimensions_reads_the_electron_count_from_any_rank():
    """Each unit's seed lives on one rank only; the count must still be the same on every rank."""
    from math import comb

    from impurityModel.ed.gf_units import unit_sector_dimensions

    comm = MPI.COMM_WORLD
    basis, _psi = _basis_and_seed()  # 4 spin-orbitals
    one = ManyBodyState({SlaterDeterminant.from_bytes(b"\x80"): 1.0}, width=1)  # 1 electron
    two = ManyBodyState({SlaterDeterminant.from_bytes(b"\xc0"): 1.0}, width=1)  # 2 electrons
    empty = ManyBodyState(width=1)
    last = comm.size - 1
    seeds = [[one if comm.rank == 0 else empty], [two if comm.rank == last else empty], [empty]]
    windows = [None, {frozenset({0, 1}): (1, 1)}, None]
    dims = unit_sector_dimensions(seeds, windows, basis)
    # Unit 1: two electrons with exactly one in {0, 1} -> 2 * 2 determinants.
    assert dims.tolist() == [float(comb(4, 1)), 4.0, 0.0]
    assert comm.allgather(dims.tolist()) == [dims.tolist()] * comm.size


@pytest.mark.skipif(MPI.COMM_WORLD.size < 3, reason="needs two colors of different widths")
@pytest.mark.parametrize("scheduler, same_cap", [("queue", True), ("static", False)])
def test_queued_units_get_one_cap_whatever_color_ran_them(monkeypatch, scheduler, same_cap):
    """Under the queue, colors differ in width by up to one rank and a unit lands on any of them; a
    user cap lowered per color width would truncate the same unit differently from run to run.
    Two units at -n 3 give colors of 2 and 1 ranks. The per-color memory bound is made to depend on
    the width (500 determinants per rank); static keeps the per-color bound (the premise), the
    queue sizes every color for the narrowest."""
    from impurityModel.ed import gf_units

    monkeypatch.setenv("GF_SCHEDULER", scheduler)

    def budget(n_orb, width, reort, ranks, comm, safety=None, **kwargs):
        return 10**9 if safety is not None else 500 * int(ranks)  # color count unbounded; unit cap per width

    monkeypatch.setattr(gf_units, "max_unit_dets_within_budget", budget)
    basis, psi = _basis_and_seed()
    basis.truncation_threshold = 10**6

    def kernel(split_basis, u, seeds):
        return (split_basis.comm.size, float(split_basis.truncation_threshold))

    results = run_units_distributed(basis, [[psi], [psi]], np.ones(2), kernel)
    if MPI.COMM_WORLD.rank == 0:
        if MPI.COMM_WORLD.size == 3:
            assert sorted(size for size, _ in results) in ([1, 2], [1, 1], [2, 2]), results
        caps = {cap for _, cap in results}
        widths = {size for size, _ in results}
        if same_cap:
            assert caps == {500.0}, results
        elif len(widths) > 1:
            assert len(caps) > 1, f"premise: static caps each color by its own width, got {results}"
