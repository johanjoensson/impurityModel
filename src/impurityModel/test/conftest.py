import faulthandler
import gc
import sys

import pytest

try:
    from mpi4py import MPI

    _has_mpi = True
except ImportError:
    _has_mpi = False


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # Under ``mpiexec -n N`` every rank runs the full pytest session and writes its
    # own progress dots, ``MPI Information`` header and summary line to the shared
    # terminal, interleaving into unreadable output. Keep the real report on rank 0
    # and redirect every other rank's terminal writer to a per-rank file, so the
    # terminal stays clean while non-root failures are preserved on disk rather than
    # discarded. Plain ``pytest`` and ``mpiexec -n 1`` (size == 1) are untouched.
    if not (_has_mpi and MPI.Is_initialized() and MPI.COMM_WORLD.Get_size() > 1):
        return
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        return
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is None:
        return
    out = open(f".pytest_mpi_rank{rank}.out", "w")
    # Keep a reference on config so the file object survives the whole session.
    config._mpi_rank_out = out
    reporter._tw._file = out
    reporter._tw.hasmarkup = False


def pytest_runtest_setup(item):
    # Dump traceback to stderr if any test hangs for more than 15 seconds
    faulthandler.dump_traceback_later(15, file=sys.stderr)


def pytest_runtest_teardown(item, nextitem):
    faulthandler.cancel_dump_traceback_later()
    if _has_mpi and MPI.Is_initialized() and not MPI.Is_finalized():
        MPI.COMM_WORLD.Barrier()
    gc.collect()
