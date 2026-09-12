"""``gf_solvers.block_Green``'s operator branch (>= 500 determinants) on a multi-rank communicator.

Pinned as a **strict xfail**: the branch wraps the ``(global_N, N_local)`` CSR in a
``LinearOperator`` whose ``matmat`` returns the full ``global_N``-row product (reduced to rank 0),
which the array kernel cannot store in its ``N_local``-row buffer -- ``ValueError: could not
broadcast`` on every rank. Serially the same call is fine and agrees with ``block_Green_sparse``
to 1e-9. RIXS's R3 stage reaches this branch on any colour spanning two or more ranks
(``doc/plans/rixs_mbs_migration.md``); when R3 moves to the MBS driver, or the branch hands the
kernel the CSR directly, delete the marker and this test becomes the regression guard.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczosCore import Reort
from impurityModel.ed.gf_solvers import block_Green, block_Green_sparse
from impurityModel.ed.greens_function import calc_G
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

NSO, NE = 12, 6  # C(12, 6) = 924 determinants: above block_Green's dense threshold of 500
OMEGA = np.linspace(-6.0, 6.0, 25)
DELTA = 0.2


def _det(occ):
    b = [0, 0]
    for i in occ:
        b[i // 8] |= 1 << (7 - i % 8)
    return SlaterDeterminant.from_bytes(bytes(b))


def _model():
    terms = {((o, "c"), (o, "a")): -1.0 + 0.3 * o for o in range(NSO)}
    for a in (0, 1):
        for b in range(2, NSO):
            terms[((a, "c"), (b, "a"))] = 0.4
            terms[((b, "c"), (a, "a"))] = 0.4
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = 3.0
    return ManyBodyOperator(terms)


def _setup(comm):
    basis = Basis(
        {0: [[0, 1]]},
        ({0: [list(range(2, 7))]}, {0: [list(range(7, 12))]}),
        initial_basis=[_det(o) for o in itertools.combinations(range(NSO), NE)],
        comm=comm,
        verbose=False,
    )
    full = [
        ManyBodyState({_det([0, 2, 3, 4, 5, 6]): 1.0 + 0j, _det([1, 2, 3, 4, 5, 6]): 0.5 + 0j}),
        ManyBodyState({_det([0, 1, 2, 3, 4, 5]): 1.0 + 0j}),
    ]
    # redistribute_psis SUMS per-rank contributions: only rank 0 supplies amplitudes, the
    # others an explicit width-1 empty (a width-0 placeholder would deadlock the exchange).
    seeds = [ManyBodyState(dict(s.items()) if comm is None or comm.rank == 0 else {}, width=1) for s in full]
    if comm is not None:
        seeds = basis.redistribute_psis(*seeds)
    return basis, seeds


def _green(fn, comm):
    basis, seeds = _setup(comm)
    assert basis.size == 924
    alphas, betas, r = fn(_model(), seeds, basis, DELTA, Reort.NONE, verbose=False)
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA)


def test_array_and_mbs_drivers_agree_serially():
    np.testing.assert_allclose(_green(block_Green, None), _green(block_Green_sparse, None), atol=1e-8)


@pytest.mark.mpi
def test_mbs_driver_matches_serial_on_many_ranks():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs a multi-rank communicator")
    np.testing.assert_allclose(_green(block_Green_sparse, comm), _green(block_Green_sparse, None), atol=1e-8)


@pytest.mark.mpi
@pytest.mark.xfail(
    strict=True, raises=ValueError, reason="block_Green's operator branch returns global rows into a local buffer"
)
def test_array_driver_operator_branch_on_many_ranks():
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs a multi-rank communicator")
    np.testing.assert_allclose(_green(block_Green, comm), _green(block_Green_sparse, None), atol=1e-8)
