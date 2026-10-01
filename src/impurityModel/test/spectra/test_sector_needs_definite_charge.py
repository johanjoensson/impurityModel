"""Sector confinement needs a state with a definite conserved charge (ledger row C7).

``spectra._sector_restrictions_per_top`` (and RIXS's intermediate-state confinement) used to
measure each thermal state's conserved charges with ``symmetries.measure_conserved_charges``,
which *rounds* the weighted average to an integer and never checks that the state actually has
a definite charge. A vector mixing two sectors -- what an eigensolver may return inside a
degenerate multiplet whose members span different ``(N_up, N_down)`` sectors -- got a sector it
does not have (0.5 rounds to 0), and the seeds were confined to it: measured, the whole seed was
pruned and the spectrum came out 0. ``symmetries.definite_conserved_charges`` checks the charge
variance and declines instead.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.spectra import _sector_restrictions_per_top
from impurityModel.ed.symmetries import definite_conserved_charges
from impurityModel.test.support.gf_branch_oracle import _det

# Two decoupled levels: H conserves n_0 and n_1 separately.
H = ManyBodyOperator({((0, "c"), (0, "a")): -0.4, ((1, "c"), (1, "a")): -0.4})
DETS = [_det((0,), 2), _det((1,), 2)]
CHARGES = [frozenset({0}), frozenset({1})]


def _basis():
    return Basis({0: [[0, 1]]}, ({0: [[]]}, {0: [[]]}), initial_basis=DETS, comm=MPI.COMM_SELF, verbose=False)


def test_a_state_mixing_two_sectors_gets_no_sector_confinement():
    mixed = ManyBodyState({d: 1 / np.sqrt(2) for d in DETS})  # degenerate: a valid eigenvector
    tOps = [ManyBodyOperator({((0, "c"),): 1.0})]
    sectors = _sector_restrictions_per_top(H, tOps, [mixed], _basis())
    assert sectors is None or sectors[0] is None, f"mixed state confined to {sectors}"


def test_a_state_in_one_sector_keeps_its_confinement():
    """The control: a definite state is still confined, to the sector its seed actually has."""
    pure = ManyBodyState({DETS[1]: 1.0})  # n_0 = 0, n_1 = 1
    tOps = [ManyBodyOperator({((0, "c"),): 1.0})]
    (sector,) = _sector_restrictions_per_top(H, tOps, [pure], _basis())
    assert sector is not None
    seed_occupation = {0, 1}  # c_0^dag |n_1 = 1>
    for key, (lo, hi) in sector.items():
        assert lo <= len(seed_occupation & set(key)) <= hi, (key, (lo, hi))


def test_definite_conserved_charges_reads_the_variance():
    assert definite_conserved_charges(ManyBodyState({DETS[0]: 1.0}), CHARGES, 2) == [1, 0]
    assert definite_conserved_charges(ManyBodyState({d: 1 / np.sqrt(2) for d in DETS}), CHARGES, 2) is None
    # A negligible admixture (weight 1e-12) is roundoff, not a second sector.
    nearly = ManyBodyState({DETS[0]: np.sqrt(1 - 1e-12), DETS[1]: 1e-6})
    assert definite_conserved_charges(nearly, CHARGES, 2) == [1, 0]
    assert definite_conserved_charges(ManyBodyState({}), CHARGES, 2) is None
