import numpy as np
import pytest
from mpi4py import MPI
from impurityModel.ed.mpi_comm import (
    dict_chunks_from_one_MPI_rank,
    allgather_dict,
    gather_distributed_results,
    distribute_determinants
)
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

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
    if MPI.COMM_WORLD.size > 1: return
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
