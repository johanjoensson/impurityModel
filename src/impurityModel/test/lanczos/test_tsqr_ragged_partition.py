r"""``tsqr`` on a wide block whose row partition is ragged, including ranks owning nothing.

Determinants are owned by ``routing_hash() % comm.size``, so at a large rank count a block can
be *wider than the rows a rank owns* -- and some ranks can own none at all while the global
block is perfectly healthy. Both regimes are a change in the local panel QR: with ``n_local <
p`` the panel is shorter than it is wide, and with ``n_local == 0`` there is no panel to factor.

Why these exist as tests rather than as a one-off probe: they are the elimination step behind a
production diagnosis. A SrMnO3 double-counting search died in ``block_normalize`` on a
262-column start block at 128 ranks and survived the same solve at 6, which puts the row
partition squarely under suspicion -- at 128 ranks that block *is* wider than a rank's share of
the basis. If ``tsqr`` mishandled either regime, the failure would be a partitioning bug and the
cold-start fallback in ``cipsi_solver`` would be aimed at the wrong thing. It does not: the
retained rank is ``min(p, total_rows)`` and orthonormality holds to machine precision in every
shape below, which is what leaves a non-finite block (``tsqr``'s ``k == -1``) as the only
remaining explanation.

The all-empty row (``total_rows == 0``) is the genuinely-zero block, and ``k == 0`` there is the
correct answer, not a failure -- the distinction ``block_normalize``'s message now carries.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.TSQR import tsqr

#: Wide enough to exceed a rank's share of the rows in the shapes below, which is the point.
P = 40


def _check(n_local, comm):
    """Factor an ``(n_local, P)`` block distributed over ``comm``; return nothing, assert plenty."""
    rng = np.random.default_rng(99 + (comm.rank if comm is not None else 0))
    total = comm.allreduce(n_local, op=MPI.SUM) if comm is not None else n_local
    A = rng.normal(size=(n_local, P)) + 1j * rng.normal(size=(n_local, P))

    Q, beta, k, _sv = tsqr(A, comm, 1.0)

    assert k == min(P, total), f"retained rank {k} for {total} rows over {P} columns"
    if k <= 0:
        return
    gram = Q.conj().T @ Q
    if comm is not None:
        gram = comm.allreduce(gram, op=MPI.SUM)
    assert np.max(np.abs(gram - np.eye(k))) < 1e-12
    local_err = float(np.max(np.abs(Q @ beta - A))) if n_local else 0.0
    assert (comm.allreduce(local_err, op=MPI.MAX) if comm is not None else local_err) < 1e-12


@pytest.mark.parametrize("rows", [2 * P, P, P // 4, 1])
def test_tsqr_is_exact_when_a_rank_owns_fewer_rows_than_the_block_is_wide(rows):
    """Serial cover of the short-panel regime; the rank count only changes how short."""
    _check(rows, None)


@pytest.mark.mpi
@pytest.mark.parametrize("shape", ["one_rank_owns_everything", "alternating_empty", "ragged"])
def test_tsqr_is_exact_on_a_ragged_partition_with_empty_ranks(shape):
    """The distributed regimes, with at least one rank contributing a zero-row panel.

    ``alternating_empty`` and ``ragged`` both leave ranks empty at every size >= 2, so unlike
    the hash-distributed fixtures elsewhere in this directory these do not need ``-n 3`` to
    exercise the empty-rank path -- the emptiness is constructed, not left to hash luck.
    """
    comm = MPI.COMM_WORLD
    rows = {
        "one_rank_owns_everything": 4 * P if comm.rank == 0 else 0,
        "alternating_empty": 3 * P if comm.rank % 2 == 0 else 0,
        "ragged": max(0, (comm.rank - 1) * P),
    }[shape]
    if comm.allreduce(rows, op=MPI.SUM) == 0:
        pytest.skip("this shape distributes no rows at this rank count")
    _check(rows, comm)
