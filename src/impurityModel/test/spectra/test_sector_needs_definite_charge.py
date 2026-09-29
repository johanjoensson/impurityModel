"""Sector confinement needs a state with a definite conserved charge (ledger row C7).

``spectra._sector_restrictions_per_top`` measures each thermal state's conserved charges with
``symmetries.measure_conserved_charges``, which *rounds* the weighted average to an integer and
never checks that the state actually has a definite charge. A vector mixing two sectors -- what
an eigensolver may return inside a degenerate multiplet whose members span different
``(N_up, N_down)`` sectors -- gets a sector it does not have (0.5 rounds to 0), and the seeds are
then confined to it, pruning the part of every seed that lives in the other sector.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.spectra import _sector_restrictions_per_top
from impurityModel.test.support.gf_branch_oracle import _det

C7 = "C7: sector restrictions from rounded, unchecked charge averages (doc/reviews/gf_review.md)"


@pytest.mark.xfail(strict=True, reason=C7)
def test_a_state_mixing_two_sectors_gets_no_sector_confinement():
    # Two decoupled levels: H conserves n_0 and n_1 separately.
    hOp = ManyBodyOperator({((0, "c"), (0, "a")): -0.4, ((1, "c"), (1, "a")): -0.4})
    dets = [_det((0,), 2), _det((1,), 2)]
    basis = Basis({0: [[0, 1]]}, ({0: [[]]}, {0: [[]]}), initial_basis=dets, comm=MPI.COMM_SELF, verbose=False)
    mixed = ManyBodyState({d: 1 / np.sqrt(2) for d in dets})  # degenerate: a valid eigenvector
    tOps = [ManyBodyOperator({((0, "c"),): 1.0})]
    sectors = _sector_restrictions_per_top(hOp, tOps, [mixed], basis)
    assert sectors is None or sectors[0] is None, f"mixed state confined to {sectors}"
