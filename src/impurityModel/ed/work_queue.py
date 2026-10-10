r"""Pull queue for distributed work units: idle colors take the next unit from a shared counter.

A static packing has to know each unit's cost before the run, and the costs that matter are the
hardest to predict: an SrMnO3 removal unit runs 35-100x longer than an addition unit with a similar
seed, and NiO's addition side or an empty conduction band can invert that. A pull queue only needs
the *order* to be roughly right. Replayed on the measured SrMnO3 unit walls, a queue in random order
already beat the static seed-mass packing 1.36x at 32 ranks and 1.65x at 64
(doc/plans/gf_load_balancing.md).

The counter is one int64 held by rank 0 of the communicator. A client asks for the next index
with a point-to-point request and waits for the answer; the host answers whenever it enters
:func:`queue_progress` (or :meth:`UnitQueue.fetch`, or :meth:`UnitQueue.finish`). It was an RMA
window taken with ``Fetch_and_op`` until a 2-node SrMnO3 run on Intel MPI over Slingshot (cxi)
failed the first cross-node fetch with "Other MPI error" in the active-message path, on every
color root of the second node; every one-node test had passed. Two-sided messages are what the
rest of the code already relies on across nodes, so the counter now uses nothing else.

Either way the host has to take part: measured with the host computing for 3.8 s, an RMA fetch
waited the full 3.8 s on MPICH 4.2.2 (one node, default settings) and on Open MPI 5.0.9 over TCP;
an ``Iprobe`` brought it to 1-17 ms. A 1-rank color makes no MPI call for a whole unit -- its
collectives short-circuit -- so the host must poke: :func:`queue_progress`, called from the
kernels' per-step loops, answers the pending requests. A progress thread is not an option: under
RSPt, MPI is initialised by a plain ``MPI_Init`` (THREAD_SINGLE).
"""

import time

import numpy as np
from mpi4py import MPI

#: Rank (in the queue's communicator) that hosts the counter.
HOST = 0

# Message tags on the queue's own communicator (a Dup, so they cannot meet any other traffic).
_REQUEST = 1
_REPLY = 2

# Queues this rank hosts, innermost last. A stack rather than a flag so a nested queue (none
# exists today) could not switch off its parent's progress.
_hosted: list = []


def queue_progress() -> None:
    """Answer the pending counter requests on a rank that hosts a live :class:`UnitQueue`.

    A no-op on every other rank (one list check), so it is cheap enough for every Lanczos block.
    Local, not collective: callers may invoke it any number of times on any rank.
    """
    if _hosted:
        _hosted[-1]._serve_pending()


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
    """A shared counter handing out ``0, 1, 2, ...`` to whichever client asks next.

    ``limit`` is the number of real indices: a client stops fetching once it is handed an index
    ``>= limit``, and that is how the host knows a client is done. ``is_client`` says whether this
    rank calls :meth:`fetch` (in the GF engine only the color roots do).

    Construction and :meth:`free` are collective on ``comm`` (a ``Dup`` is created and freed);
    :meth:`fetch` and :meth:`finish` are not. Every rank calls :meth:`finish` after its last fetch
    on the success path: the host keeps answering there until every other client has been handed
    an index past ``limit``, so a host that runs out of work first cannot strand a client in its
    fetch, and the caller's collectives after the loop can follow. (It is not part of
    :meth:`free`: on a path where every rank raised, clients stop fetching without reaching
    ``limit``, and a host waiting for that would hang.) Free at a synchronized point -- a
    ``finally`` that every rank reaches -- never from the garbage collector.

    A host that reaches :meth:`free` without :meth:`finish` is failing: it answers every fetch still
    to come with :attr:`ABORTED` instead of a real index, so the other colors stop (and raise) at
    their next fetch rather than start units whose results nobody will collect.

    ``max_wait`` is the longest single :meth:`fetch` that returned a real index on this rank, in
    seconds: a long wait means the host computed without calling :func:`queue_progress`.
    """

    #: Returned by :meth:`fetch` after the host failed. Negative, so callers must test for it
    #: before using the value as an index (``order[-1]`` would silently run the last unit).
    ABORTED = -1

    def __init__(self, comm, limit: int, is_client: bool = True):
        self._comm = comm.Dup()
        # A transport error on the counter must end the job with MPI's message. Returned as a
        # Python exception it would raise on one client only, and the host would then wait in
        # finish() for that client's last request until the job's wall time ran out.
        self._comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
        self._is_host = self._comm.rank == HOST
        self._limit = int(limit)
        # Clients the host answers over the wire; its own fetches are local.
        self._open_clients = self._comm.allreduce(int(bool(is_client) and not self._is_host), op=MPI.SUM)
        # Clients already handed an index past `limit`: a client that asked again (an API misuse)
        # must not be counted out twice, or finish() would return while another still waits.
        self._done_clients: set = set()
        self._aborting = False
        self._finished = False
        self._next = 0
        self._request = np.zeros(1, dtype=np.int64)
        self._reply = np.zeros(1, dtype=np.int64)
        self._status = MPI.Status()
        if self._is_host:
            _hosted.append(self)
        self.max_wait = 0.0

    def _take(self) -> int:
        got = self._next
        self._next += 1
        return got

    def _answer(self) -> None:
        """Receive the request already probed into ``self._status`` and send it the next index."""
        source = self._status.Get_source()
        self._comm.Recv(self._request, source=source, tag=_REQUEST)
        self._reply[0] = self.ABORTED if self._aborting else self._take()
        if self._reply[0] >= self._limit and source not in self._done_clients:
            self._done_clients.add(source)
            self._open_clients -= 1
        self._comm.Send(self._reply, dest=source, tag=_REPLY)

    def _serve_pending(self) -> None:
        while self._comm.Iprobe(source=MPI.ANY_SOURCE, tag=_REQUEST, status=self._status):
            self._answer()

    def fetch(self) -> int:
        """The next index; every index is returned exactly once across all ranks."""
        start = time.perf_counter()
        if self._is_host:
            self._serve_pending()  # first come, first served
            got = self._take()
        else:
            self._comm.Send(self._request, dest=HOST, tag=_REQUEST)
            self._comm.Recv(self._reply, source=HOST, tag=_REPLY)
            got = int(self._reply[0])
        if 0 <= got < self._limit:
            # Only a wait for real work costs anything: a color told to stop has nothing to do,
            # and counting its wait would flag a missing poke where there is none.
            self.max_wait = max(self.max_wait, time.perf_counter() - start)
        return got

    def finish(self) -> None:
        """On the host, answer requests until every client has been handed an index past ``limit``.

        A no-op on every other rank. Not collective.
        """
        if not self._is_host:
            return
        self._finished = True
        while self._open_clients > 0:
            self._comm.Probe(source=MPI.ANY_SOURCE, tag=_REQUEST, status=self._status)
            self._answer()

    def free(self) -> None:
        """Release the communicator. Collective.

        The host keeps answering requests until every rank has arrived. On the success path
        :meth:`finish` already answered them all. Without it the host is failing, and a client may
        still be waiting on its fetch (or start one later): it is answered :attr:`ABORTED`, so it
        raises and gets here. (That is why the arrival is a
        non-blocking barrier: a blocking ``Free`` on the host would wait for that client forever.)
        """
        if self._comm is None:
            return
        try:
            arrived = self._comm.Ibarrier()
            if self._is_host:
                self._aborting = not self._finished
                while not arrived.Test():
                    self._serve_pending()
            else:
                arrived.Wait()
        finally:
            # Unregistered on every path, so later queue_progress calls never poke a dead queue.
            if self._is_host:
                _hosted.remove(self)
        self._comm.Free()
        self._comm = None
