r"""Pull queue for distributed work units: idle colors take the next unit from a shared counter.

A static packing has to know each unit's cost before the run, and the costs that matter are the
hardest to predict: an SrMnO3 removal unit runs 35-100x longer than an addition unit with a similar
seed, and NiO's addition side or an empty conduction band can invert that. A pull queue only needs
the *order* to be roughly right. Replayed on the measured SrMnO3 unit walls, a queue in random order
already beat the static seed-mass packing 1.36x at 32 ranks and 1.65x at 64
(doc/plans/gf_load_balancing.md).

The counter is one int64 in an RMA window hosted on rank 0 of the communicator, incremented with
``Fetch_and_op``. Passive-target RMA only progresses when the host enters MPI: measured with the
host computing for 3.8 s, a fetch waited the full 3.8 s on MPICH 4.2.2 (one node, default
settings) and on Open MPI 5.0.9 over TCP; ``MPI.Request.Testall([])`` did not help, an ``Iprobe``
on the window's communicator brought it to 1-17 ms. A 1-rank color makes no MPI call for a whole
unit -- its collectives short-circuit, and a size-1 ``Allreduce`` measured no progress either -- so
the host must poke: :func:`queue_progress`, called from the kernels' per-step loops. A progress
thread is not an option: under RSPt, MPI is initialised by a plain ``MPI_Init`` (THREAD_SINGLE).

This module has no Cython imports, so its MPI behaviour can be tested against any mpi4py build.
"""

import time

import numpy as np
from mpi4py import MPI

#: Rank (in the queue's communicator) that hosts the counter.
HOST = 0

# Communicators this rank hosts a live counter on, innermost last. A stack rather than a flag so a
# nested queue (none exists today) could not switch off its parent's progress.
_hosted: list = []


def queue_progress() -> None:
    """Let MPI progress pending counter requests on a rank that hosts a live :class:`UnitQueue`.

    A no-op on every other rank (one list check), so it is cheap enough for every Lanczos block.
    Local, not collective: callers may invoke it any number of times on any rank.
    """
    if _hosted:
        _hosted[-1].Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)


def queue_order(weights, tiebreak=None) -> np.ndarray:
    """Dispatch order for a queue: heaviest predicted unit first.

    Ties in ``weights`` go to the larger ``tiebreak`` (when given), then to the lower index. Only
    the ranking matters. Replicated inputs give a replicated order.
    """
    primary = -np.asarray(weights, dtype=float)
    if tiebreak is None:
        return np.argsort(primary, kind="stable")
    return np.lexsort((-np.asarray(tiebreak, dtype=float), primary))


def queue_makespan(walls, order, n_colors: int) -> float:
    """Wall time of a pull queue that hands units out in ``order`` to ``n_colors`` identical colors.

    Pure list scheduling: each unit goes to the color that frees up first. Used to judge an
    ordering against measured per-unit walls without an MPI run.
    """
    walls = np.asarray(walls, dtype=float)
    free_at = np.zeros(max(1, int(n_colors)))
    for u in order:
        c = int(np.argmin(free_at))
        free_at[c] += walls[int(u)]
    return float(free_at.max())


class UnitQueue:
    """A shared counter handing out ``0, 1, 2, ...`` to whichever rank asks next.

    Construction and :meth:`free` are collective on ``comm`` (a ``Dup`` and a window are created and
    freed); :meth:`fetch` is not. Free it at a synchronized point -- a ``finally`` that every rank
    reaches -- never from the garbage collector.

    ``max_wait`` is the longest single :meth:`fetch` on this rank, in seconds: a long wait means the
    host computed without calling :func:`queue_progress`.
    """

    def __init__(self, comm):
        self._comm = comm.Dup()
        self._is_host = self._comm.rank == HOST
        self._counter = np.zeros(1, dtype=np.int64)
        self._win = MPI.Win.Create(self._counter if self._is_host else None, disp_unit=8, comm=self._comm)
        if self._is_host:
            _hosted.append(self._comm)
        self.max_wait = 0.0

    def fetch(self) -> int:
        """The next index; every index is returned exactly once across all ranks."""
        one = np.ones(1, dtype=np.int64)
        got = np.zeros(1, dtype=np.int64)
        start = time.perf_counter()
        self._win.Lock(HOST, MPI.LOCK_SHARED)
        self._win.Fetch_and_op(one, got, HOST, 0, MPI.SUM)
        self._win.Unlock(HOST)
        self.max_wait = max(self.max_wait, time.perf_counter() - start)
        return int(got[0])

    def free(self) -> None:
        """Release the window and the communicator. Collective."""
        if self._win is None:
            return
        if self._is_host:
            _hosted.remove(self._comm)
        self._win.Free()
        self._comm.Free()
        self._win = None
