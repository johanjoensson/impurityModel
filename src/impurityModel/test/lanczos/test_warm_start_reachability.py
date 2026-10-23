"""``CIPSISolver.get_eigenvectors`` warm-start sector-blindness guards.

A block warm-started from converged eigenvectors spans a (near-)invariant subspace. When the
ground state has moved to another near-decoupled charge sector between solves (the
fixed-occupation DC search jumps across a charge-transfer crossing between trial shifts), Lanczos
restarted from that block alone converges the wrong sector's states only. The start block
therefore always carries generic full-support columns next to the warm ones, so every sector is
reachable from it: these tests run real solves on a diagonal Hamiltonian whose exact spectrum is
known, and hand the solver warm blocks in the wrong place, corrupted, or empty.

The last two tests pin what ``max_energy=None`` returns on the *dense* branch, which
``calc_energy`` takes on every sector basis below ``dense_cutoff``.
"""

import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.eigensolvers import Lowest, Within
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3]]}, {0: [[4, 5]]})
N_SPIN_ORBITALS = 6
CUT = 1e-3  # thermal energy cut handed to get_eigenvectors as max_energy


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
    assert len(basis) >= 5  # num_wanted below must exceed the fake's 3 exhausted states
    return basis, CIPSISolver(basis)


def _warm_refs(basis):
    """A single-determinant reference block, fed on rank 0 only (redistribute_psis SUMS replicas)."""
    local = list(basis.local_basis)
    parts = basis.comm.allgather(local) if basis.is_distributed else [local]
    det0 = sorted(det for part in parts for det in part)[0]
    rank = basis.comm.rank if basis.is_distributed else 0
    seed = ManyBodyState({det0: 1.0} if rank == 0 else {}, width=1)
    (out,) = basis.redistribute_psis(seed)[0].to_states()
    return [out]


def _h_op():
    return ManyBodyOperator({((i, "c"), (i, "a")): float(i + 1) for i in range(N_SPIN_ORBITALS)})


def test_lowest_on_the_dense_branch_returns_exactly_one_state():
    """``Lowest`` on the dense branch builds one state, however large the basis.

    ``eigensystem`` diagonalizes the whole space, so without a bound in ``get_eigenvectors`` a
    ``Lowest`` caller would build one ``ManyBodyState`` per determinant -- a cost set by the basis
    size rather than by the request. This is the branch every double-counting sector solve and
    every occupation-walk trial takes on a basis below ``dense_cutoff``.
    """
    _basis, solver = _make_solver()

    e_ref, psi_refs = solver.get_eigenvectors(_h_op(), Lowest())

    assert len(e_ref) == psi_refs.width == 1
    np.testing.assert_allclose(e_ref.real, _exact_energies()[:1])


def test_a_window_wider_than_the_spectrum_is_clamped_not_an_error():
    basis, solver = _make_solver()

    e_ref, psi_refs = solver.get_eigenvectors(_h_op(), Within(1e9))

    assert len(e_ref) == psi_refs.width == len(basis)


def _uniform_refs(basis, value):
    """A one-column warm block holding ``value`` on every determinant this rank owns.

    Built from ``local_basis`` directly rather than through ``redistribute_psis``: the point is
    to hand ``get_eigenvectors`` a block in a specific numerical state on every rank, and a
    redistribute would sum replicas and change the value (see the note on ``_warm_refs``).
    """
    return [ManyBodyState(dict.fromkeys(basis.local_basis, value), width=1)]


def _exact_energies():
    """H is diagonal in this basis, so the exact spectrum is the per-determinant occupation sum."""
    return sorted(sum(orb + 1 for orb in occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), 4))


def _high_energy_refs(basis):
    """A one-column warm block on the highest-energy determinant (fed on rank 0 only)."""
    local = list(basis.local_basis)
    parts = basis.comm.allgather(local) if basis.is_distributed else [local]
    dets = sorted(det for part in parts for det in part)

    def energy(det):
        return sum(orb + 1 for orb in range(N_SPIN_ORBITALS) if (int(det[0]) >> (63 - orb)) & 1)

    top = max(dets, key=energy)
    rank = basis.comm.rank if basis.is_distributed else 0
    seed = ManyBodyState({top: 1.0} if rank == 0 else {}, width=1)
    (out,) = basis.redistribute_psis(seed)[0].to_states()
    return [out]


def test_a_warm_block_in_the_wrong_sector_still_finds_the_true_ground_state():
    """The reachability oracle: warm columns are an eigenvector of the *highest* state."""
    basis, solver = _make_solver()
    e_ref, psi_refs = solver.get_eigenvectors(_h_op(), Within(CUT), dense_cutoff=1, psi_refs=_high_energy_refs(basis))
    assert abs(float(np.min(e_ref.real)) - _exact_energies()[0]) < 1e-9
    assert psi_refs.width == len(e_ref) > 0


def test_a_cold_start_finds_the_true_ground_state():
    _basis, solver = _make_solver()
    e_ref, _ = solver.get_eigenvectors(_h_op(), Within(CUT), dense_cutoff=1)
    assert abs(float(np.min(e_ref.real)) - _exact_energies()[0]) < 1e-9


def test_a_non_finite_warm_block_falls_back_to_the_generic_start():
    """A corrupted warm block costs iterations, not the whole calculation.

    The regression: a production double-counting search died when ``block_normalize`` raised on the
    warm-started start block and nothing caught it.
    """
    basis, solver = _make_solver()
    e_ref, psi_refs = solver.get_eigenvectors(
        _h_op(), Within(CUT), dense_cutoff=1, psi_refs=_uniform_refs(basis, complex(np.nan, 0.0))
    )
    assert abs(float(np.min(e_ref.real)) - _exact_energies()[0]) < 1e-9
    assert psi_refs.width == len(e_ref) > 0


def test_a_zero_warm_block_needs_no_fallback_because_the_generic_columns_carry_it():
    basis, solver = _make_solver()
    e_ref, _ = solver.get_eigenvectors(_h_op(), Within(CUT), dense_cutoff=1, psi_refs=_uniform_refs(basis, 0.0))
    assert abs(float(np.min(e_ref.real)) - _exact_energies()[0]) < 1e-9
