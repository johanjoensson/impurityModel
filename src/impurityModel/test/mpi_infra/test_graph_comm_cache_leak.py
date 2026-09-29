"""The dist-graph cache must not outlive the communicators it was built for (ledger row M3).

``mpi_comm._cached_dist_graph`` keys its cache by ``id(parent comm)`` and pins the parent in the
entry. The number of graphs *per parent* is capped, but nothing ever removes a parent: every
``Clone()`` that redistributes states (one per RIXS work unit, one per bicgstab unit, one per
split color) leaves an entry -- the freed parent plus up to eight dist-graph communicators --
for the rest of the process. On a long RIXS run that is a steady drain on MPI's context ids.
"""

import pytest
from mpi4py import MPI

from impurityModel.ed import mpi_comm
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.support.gf_branch_oracle import _det

M3 = "M3: _graph_comm_cache is never evicted when its parent communicator is freed"

N_CYCLES = 5


def _cycle(comm):
    """What a GF unit does: clone the communicator, build a basis on it, redistribute, free."""
    sub = comm.Clone()
    dets = [_det(occ, 8) for occ in ((0, 1), (0, 2), (1, 3), (2, 5), (4, 6), (3, 7))]
    basis = Basis({0: [[0, 1]]}, ({0: [[2, 3, 4]]}, {0: [[5, 6, 7]]}), initial_basis=dets, comm=sub, verbose=False)
    state = ManyBodyState({d: 1.0 + i for i, d in enumerate(dets)}) if comm.rank == 0 else ManyBodyState(width=1)
    basis.redistribute_psis(ManyBodyState.from_states([state]) if comm.rank == 0 else state)
    basis.free_comm()


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="dist-graph communicators are only built on >1 rank")
@pytest.mark.xfail(strict=True, reason=M3)
def test_freed_communicators_leave_no_cached_graphs():
    comm = MPI.COMM_WORLD
    _cycle(comm)  # warm: anything the first call caches for good (e.g. on COMM_WORLD) is not a leak
    before = len(mpi_comm._graph_comm_cache)
    for _ in range(N_CYCLES):
        _cycle(comm)
    grown = len(mpi_comm._graph_comm_cache) - before
    # The verdict must be rank-invariant (it is: every rank runs the same cycles), and it is
    # reduced anyway so a divergence would surface as a failure rather than an XPASS on one rank.
    grown = comm.allreduce(grown, op=MPI.MAX)
    assert grown == 0, f"{grown} cache entries survived {N_CYCLES} freed communicators"
