"""``CIPSISolver.get_eigenvectors(Lowest())`` on the dense branch, for a degenerate ground level.

``eigh`` returns an arbitrary orthonormal basis of a degenerate level, so column 0 alone is a
roundoff-chosen member -- often a single determinant in one symmetry sector. On the NiO gap-DC
search that member changed between trial shifts of the same seed matrix, and expansions started
from the wrong one converged 0.15-0.17 too high. The reference must instead overlap the whole
level, and be the same state whatever constant the Hamiltonian is shifted by.
"""

import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.eigensolvers import Lowest
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3]]}, {0: [[4, 5]]})
N_SPIN_ORBITALS = 6
#: Orbitals 3 and 4 share an energy, so {0, 1, 2, 3} and {0, 1, 2, 4} are a degenerate ground level.
ENERGIES = [1.0, 2.0, 3.0, 4.0, 4.0, 6.0]


def _det(occupied):
    """SlaterDeterminant with the given orbitals occupied (MSB-first bit convention)."""
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _make_solver():
    basis = Basis(
        IMPURITY_ORBITALS,
        BATH_STATES,
        nominal_impurity_occ={0: 2},
        comm=MPI.COMM_WORLD,
        verbose=False,
    )
    basis.add_states([_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), 4)])
    return basis, CIPSISolver(basis)


def _h_op(shift=0.0):
    """``sum_i (e_i - shift) n_i``: at fixed particle number, a constant shift of the spectrum."""
    return ManyBodyOperator({((i, "c"), (i, "a")): ENERGIES[i] - shift for i in range(N_SPIN_ORBITALS)})


def _amplitudes(basis, psi_refs, dets):
    """The reference's amplitude on each of ``dets``, summed over the ranks (each has one owner)."""
    (state,) = psi_refs.to_states()
    local = dict(state.items())
    amps = np.array([np.ravel(local.get(det, [0.0]))[0] for det in dets], dtype=complex)
    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, amps, op=MPI.SUM)
    return amps


def test_the_dense_lowest_reference_overlaps_every_member_of_a_degenerate_ground_level():
    basis, solver = _make_solver()

    e_ref, psi_refs = solver.get_eigenvectors(_h_op(), Lowest(), dense_cutoff=1e3)

    np.testing.assert_allclose(e_ref.real, [10.0])
    amps = _amplitudes(basis, psi_refs, [_det((0, 1, 2, 3)), _det((0, 1, 2, 4))])
    assert np.all(np.abs(amps) > 1e-3), amps
    np.testing.assert_allclose(np.linalg.norm(amps), 1.0)


def test_the_dense_lowest_reference_is_the_same_state_at_every_shift():
    dets = [_det((0, 1, 2, 3)), _det((0, 1, 2, 4))]
    refs = []
    for shift in (0.0, 0.025, 0.1875, -0.327):
        basis, solver = _make_solver()
        _e_ref, psi_refs = solver.get_eigenvectors(_h_op(shift), Lowest(), dense_cutoff=1e3)
        refs.append(_amplitudes(basis, psi_refs, dets))

    for amps in refs[1:]:
        assert abs(np.vdot(refs[0], amps)) > 1 - 1e-10, (refs[0], amps)

