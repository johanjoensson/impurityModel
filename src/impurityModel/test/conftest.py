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
    out = open(f".pytest_mpi_rank{rank}.out", "w")  # noqa: SIM115  (see below)
    # Keep a reference on config so the file object survives the whole session,
    # and close it at unconfigure so interpreter shutdown doesn't emit an
    # unclosed-file ResourceWarning.
    config._mpi_rank_out = out
    config.add_cleanup(out.close)
    reporter._tw._file = out
    reporter._tw.hasmarkup = False


def _watchdog_file(config):
    # faulthandler needs a file object that is NOT one of the process's fd 0/1/2:
    # pytest's default "fd" capture dup2()s a temp file over the real stderr fd for
    # the duration of each test, so writes to sys.stderr (or even sys.__stderr__,
    # which still names the same, now-redirected, fd number) land in the captured
    # buffer and are invisible unless the test fails. A plain open() file uses an
    # independent fd, so it bypasses the redirect entirely. Cached on config so a
    # long test session reuses one handle instead of reopening per item.
    cached = getattr(config, "_hang_watchdog_file", None)
    if cached is not None:
        return cached
    rank = MPI.COMM_WORLD.Get_rank() if (_has_mpi and MPI.Is_initialized()) else 0
    size = MPI.COMM_WORLD.Get_size() if (_has_mpi and MPI.Is_initialized()) else 1
    suffix = f"_rank{rank}" if size > 1 else ""
    f = open(f".pytest_watchdog{suffix}.out", "w")  # noqa: SIM115  (closed via add_cleanup)
    config._hang_watchdog_file = f
    config.add_cleanup(f.close)
    return f


def pytest_runtest_setup(item):
    # Dump traceback if any test runs longer than 15 seconds. This is a diagnostic
    # snapshot, not a failure: a legitimately slow test just gets one recorded
    # stack showing it mid-computation (e.g. inside eigh), distinguishing "slow"
    # from "actually stuck on an MPI collective" without a manual py-spy bisect.
    faulthandler.dump_traceback_later(15, file=_watchdog_file(item.config))


def pytest_runtest_teardown(item, nextitem):
    faulthandler.cancel_dump_traceback_later()
    # MPI_Comm_free is collective: without a barrier, one rank could be inside gc.collect()
    # (freeing a split communicator -- see basis_split.py) while another rank has already
    # moved on to the next test, a protocol violation that segfaults. gc.collect() is what
    # actually calls MPI_Comm_free (via the communicator wrapper's finalizer), so both must
    # run together, and only when a split communicator could exist in the first place --
    # basis_split only activates at world.size > 1 (see CLAUDE.md), so a plain serial run or
    # `mpiexec -n 1` has nothing to synchronize. gc.collect() alone measured ~33 ms/item here
    # (~50 s across the full suite); gating it to real multi-rank runs recovers that for the
    # serial gate without touching the invariant it exists for.
    if _has_mpi and MPI.Is_initialized() and not MPI.Is_finalized() and MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.Barrier()
        gc.collect()
