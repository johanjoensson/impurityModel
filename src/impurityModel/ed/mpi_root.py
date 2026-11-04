"""Rank-0 computation with a collective verdict (layer 0: imports nothing from the package)."""


def on_root(comm, compute):
    """``compute()`` on rank 0 (collective): its result there, ``None`` elsewhere.

    An exception on rank 0 is announced to every rank before anything else is exchanged and raised
    everywhere, so no rank is left waiting in the broadcast that would have carried the result.
    """
    result = error = None
    if comm is None or comm.rank == 0:
        try:
            result = compute()
        except Exception as exc:  # re-raised below, on every rank
            error = exc
    if comm is not None:
        failure = comm.bcast(None if error is None else f"{type(error).__name__}: {error}", root=0)
        if failure is not None:
            if error is not None:
                raise error
            raise RuntimeError(f"rank 0 failed: {failure}")
    elif error is not None:
        raise error
    return result


def on_every_rank(comm, compute):
    """``compute()`` on every rank (collective): its result, or the same failure raised on all ranks.

    For a step that runs everywhere but can only fail on some ranks (a merge that works on rank 0's data and
    on ``None`` elsewhere): each rank's verdict is exchanged before anyone moves on, so a failure on one rank
    is raised on every rank instead of leaving the others in the next collective.
    """
    result = error = None
    try:
        result = compute()
    except Exception as exc:  # re-raised below, on every rank
        error = exc
    if comm is not None:
        verdicts = comm.allgather(None if error is None else f"rank {comm.rank}: {type(error).__name__}: {error}")
        failure = next((v for v in verdicts if v is not None), None)
        if failure is not None:
            if error is not None:
                raise error
            raise RuntimeError(failure)
    elif error is not None:
        raise error
    return result
