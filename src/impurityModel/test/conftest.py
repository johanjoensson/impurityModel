import faulthandler
import gc
import os
import sys
import tempfile

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
    # Every rank also creates and garbage-collects numbered tmpdirs under the same
    # /tmp/pytest-of-<user> root; the concurrent cleanup (rename to garbage-<uuid>
    # + rmtree) races between ranks, and the resulting "(rm_rf) error removing"
    # PytestWarning is a hard session error under filterwarnings = error even when
    # every test passed. Point each rank at a private temp root instead. Read
    # lazily by TempPathFactory.getbasetemp() and only when --basetemp is not
    # given, so an explicit --basetemp still wins; the directory must pre-exist
    # because getbasetemp() creates pytest-of-<user> without parents=True.
    temproot = os.path.join(tempfile.gettempdir(), f"pytest-mpi-rank{rank}")
    os.makedirs(temproot, exist_ok=True)
    os.environ["PYTEST_DEBUG_TEMPROOT"] = temproot
    if rank == 0:
        return
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is None:
        return
    out = open(f".pytest_mpi_rank{rank}.out", "w")
    # Keep a reference on config so the file object survives the whole session,
    # and close it at unconfigure so interpreter shutdown doesn't emit an
    # unclosed-file ResourceWarning.
    config._mpi_rank_out = out
    config.add_cleanup(out.close)
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
