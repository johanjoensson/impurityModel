import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.mpi_comm import (
    allgather_dict,
    dict_chunks_from_one_MPI_rank,
    gather_distributed_results,
)


def test_dict_chunks_from_one_MPI_rank():
    data = {1: "a", 2: "b", 3: "c"}
    chunks = list(dict_chunks_from_one_MPI_rank(data, chunk_maxsize=2, root=0))
    if MPI.COMM_WORLD.rank == 0:
        assert len(chunks) == 2
        assert chunks[0] == {1: "a", 2: "b"}
        assert chunks[1] == {3: "c"}
    else:
        assert chunks[0] is None


@pytest.mark.mpi
def test_dict_chunks_from_one_MPI_rank_mpi():
    comm = MPI.COMM_WORLD
    data = {1: "a", 2: "b", 3: "c"} if comm.rank == 0 else {}
    chunks = list(dict_chunks_from_one_MPI_rank(data, chunk_maxsize=2, root=0))
    if comm.rank == 0:
        assert len(chunks) == 2
        assert chunks[0] == {1: "a", 2: "b"}
    else:
        assert len(chunks) == 2
        assert chunks[0] is None


def test_allgather_dict():
    if MPI.COMM_WORLD.size > 1:
        return
    total = {}
    data = {1: "a"}
    allgather_dict(data, total, chunk_maxsize=10)
    assert total == {1: "a"}


@pytest.mark.mpi
def test_allgather_dict_mpi():
    comm = MPI.COMM_WORLD
    total = {}
    data = {comm.rank: comm.rank * 10}
    # Test small chunk size to trigger chunking logic
    allgather_dict(data, total, chunk_maxsize=1)
    assert len(total) == comm.size
    for i in range(comm.size):
        assert total[i] == i * 10


def test_gather_distributed_results():
    local_res = np.array([1.0, 2.0])
    items = [2]
    roots = [0]
    res = gather_distributed_results(None, 0, roots, items, local_res)
    np.testing.assert_array_equal(res, local_res)


@pytest.mark.mpi
def test_gather_distributed_results_mpi():
    comm = MPI.COMM_WORLD
    local_res = np.array([float(comm.rank)])
    items = [1] * comm.size
    roots = list(range(comm.size))
    res = gather_distributed_results(comm, 0, roots, items, local_res)
    if comm.rank == 0:
        assert len(res) == comm.size
        for i in range(comm.size):
            assert res[i] == float(i)
    else:
        assert res is None


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs a non-root color to report zero items")
@pytest.mark.parametrize("is_array", [True, False])
def test_a_color_with_zero_items_sends_nothing(is_array):
    """A zero-count color must not leave a send behind for the next gather to receive.

    Rank 0 skips receiving from a color with ``count == 0``; if that color's root sent its empty
    result anyway, the message stayed queued and the *next* receive from that rank took it. A
    pull-queue colour can legitimately finish with no units.
    """
    comm = MPI.COMM_WORLD
    roots = list(range(comm.size))
    empty_rank = comm.size - 1

    def local(value):
        if is_array:
            return np.array([value], dtype=float) if value is not None else np.empty(0, dtype=float)
        return [value] if value is not None else []

    first = [0 if r == empty_rank else 1 for r in roots]
    gather_distributed_results(
        comm, 0, roots, first, local(None if comm.rank == empty_rank else 1.0), is_array=is_array
    )
    second = [1] * comm.size
    res = gather_distributed_results(comm, 0, roots, second, local(42.0 + comm.rank), is_array=is_array)
    if comm.rank == 0:
        assert list(res) == [42.0 + r for r in roots]
