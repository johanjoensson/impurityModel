"""The Lanczos kernels' stop decision must be the same on every rank (ledger row M2).

The GF convergence monitor is evaluated rank-locally on alpha, which comes out of an
``Allreduce`` -- and MPI does not promise a bitwise-identical ``Allreduce`` result on every rank
(recursive doubling sums in a rank-dependent order; heterogeneous ``-march=native`` builds add
more). A verdict that flips on one rank near the threshold sends it out of the loop while the
others enter the next iteration's collectives: a deadlock. The kernels now take root's verdict.

The test injects the disagreement directly: a monitor that says "converged" on every non-root
rank three blocks before root does. Every rank must stop where root stops.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import block_normalize
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.test.support.gf_branch_oracle import cached_oracle

ROOT_STOPS_AT, OTHERS_STOP_AT = 6, 3


def _disagreeing_monitor(comm):
    stop_at = ROOT_STOPS_AT if comm.rank == 0 else OTHERS_STOP_AT

    def converged(alphas, betas, **kwargs):
        return len(alphas) >= stop_at

    return converged


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="a rank-local verdict needs a second rank to disagree with")
def test_every_rank_stops_where_root_stops():
    comm = MPI.COMM_WORLD
    oracle, (hOp, imp, baths, _n_orb, _n0, _blocks) = cached_oracle(1)
    basis = Basis(imp, baths, initial_basis=oracle.gs.dets, comm=comm, verbose=False)
    rng = np.random.default_rng(7)
    amps = rng.normal(size=len(oracle.gs.dets)) + 1j * rng.normal(size=len(oracle.gs.dets))
    seed = ManyBodyState(dict(zip(oracle.gs.dets, amps))) if comm.rank == 0 else ManyBodyState(width=1)
    (seed,) = basis.redistribute_psis(ManyBodyState.from_states([seed]) if comm.rank == 0 else seed)
    psi0, _ = block_normalize(seed, mpi=True, comm=comm)

    alphas, _betas, _q, _w = block_lanczos_cy(
        psi0=psi0,
        h_op=hOp,
        basis=basis,
        converged_fn=_disagreeing_monitor(comm),
        reort="none",
        max_iter=12,
        verbose=False,
    )
    n_blocks = comm.allgather(len(alphas))
    assert n_blocks == [ROOT_STOPS_AT] * comm.size, n_blocks
