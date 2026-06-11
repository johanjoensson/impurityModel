import gc
try:
    from mpi4py import MPI
    _has_mpi = True
except ImportError:
    _has_mpi = False


def pytest_runtest_teardown(item, nextitem):
    """Synchronise all MPI ranks before garbage-collecting.

    MPI_Comm_free is a collective operation: every rank in a communicator must
    call it simultaneously.  Without the barrier, one rank may be inside
    gc.collect() (calling MPI_Comm_free on a split communicator) while another
    rank has already moved on to the next test — leading to protocol violations
    and segmentation faults.
    """
    if _has_mpi and MPI.Is_initialized() and not MPI.Is_finalized():
        MPI.COMM_WORLD.Barrier()
    gc.collect()
