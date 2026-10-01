"""The array block-Green's basis growth must reach every determinant its Krylov space needs.

:func:`gf_solvers.block_Green` runs the block-Lanczos recurrence on H restricted to a basis it
grows by probing: apply H to some Lanczos vectors and add what they reach. A column whose chain
closes only because the incomplete basis truncates H deflates, and the determinants it was
missing lie next to *its* last vectors. Probing from the final Lanczos vector alone never found
them, so the growth stopped short and G came back silently wrong (review ledger C12).
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.gf_primitives import calc_G
from impurityModel.ed.gf_solvers import block_Green
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.test.support.gf_branch_oracle import cached_oracle

TAU = 0.3
DELTA = 0.2
IW = 1j * np.pi * TAU * (2 * np.arange(10) + 1)
#: Eigenstate 2 of the nb1 model, block [2, 3] plus bath orbital 4 (which couples to block [0, 1]
#: only): the bath column's chain closes after 11 blocks inside a 46-determinant basis whose
#: H-closure has 48, and the block narrows 3 -> 1 there.
STATE = 2
ORBITALS = (2, 3, 4)


@pytest.mark.parametrize("reort", [None, "full"])
def test_a_column_that_deflates_early_still_has_its_basis_grown(reort):
    oracle, (hOp, imp, baths, _n, _n0, _blocks) = cached_oracle(1)
    psi = oracle.psis([STATE])[0]
    ops = [ManyBodyOperator({((o, "c"),): 1}) for o in ORBITALS]
    seeds = [op(psi, 0) for op in ops]
    support = sorted({d for s in seeds for d in s.to_dict()})
    basis = Basis(imp, baths, initial_basis=support, comm=MPI.COMM_SELF, verbose=False)

    alphas, betas, r = block_Green(hOp, seeds, basis, DELTA, reort, slaterWeightMin=0, verbose=False)

    e = oracle.gs.e[STATE]
    g = calc_G(alphas, betas, r, IW, e, 0.0)
    ref = oracle.transition_tensor(ops, +1, [STATE], TAU, IW)
    np.testing.assert_allclose(g, ref, rtol=0, atol=1e-10)
    assert basis.size == 48  # the H-closure of the seed support
