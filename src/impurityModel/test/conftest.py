import gc
import faulthandler
import sys

try:
    from mpi4py import MPI
    _has_mpi = True
except ImportError:
    _has_mpi = False


def pytest_runtest_setup(item):
    # Dump traceback to stderr if any test hangs for more than 15 seconds
    faulthandler.dump_traceback_later(15, file=sys.stderr)


def pytest_runtest_teardown(item, nextitem):
    faulthandler.cancel_dump_traceback_later()
    if _has_mpi and MPI.Is_initialized() and not MPI.Is_finalized():
        MPI.COMM_WORLD.Barrier()
    gc.collect()

