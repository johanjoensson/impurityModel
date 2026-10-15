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
import pytest
from mpi4py import MPI

from impurityModel.ed.cipsi_solver import CIPSISolver
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


def test_no_cut_dense_branch_returns_exactly_num_wanted_states():
    """The dense branch is bounded by ``num_wanted`` when there is no cut.

    ``eigensystem`` reads ``e_max=None`` as "no cutoff" and returns the *whole* dense spectrum,
    so without a bound in ``get_eigenvectors`` a cut-less caller would build one ``ManyBodyState``
    per determinant -- a cost set by the basis size rather than by the request. This is the branch
    ``calc_energy`` takes on every sector basis below ``dense_cutoff``, and the tests above run
    the Krylov branch (``dense_cutoff=1``), so it needs its own.

    The slice is by count alone: with no cut there is no manifold to keep whole, and it can land
    inside a degenerate group. That is sound here only because ``calc_energy``, the sole caller
    of the no-cut path, keeps ``min(es)``.
    """
    basis, solver = _make_solver()
    h_op = _h_op()

    e_ref, psi_refs = solver.get_eigenvectors(h_op, num_wanted=3, max_energy=None)

    assert len(e_ref) == psi_refs.width == 3
    # H is diagonal in this basis, so the exact spectrum is the per-determinant occupation sum.
    exact = sorted(sum(orb + 1 for orb in occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), 4))
    np.testing.assert_allclose(sorted(e_ref.real), exact[:3])


def test_no_cut_request_above_the_basis_size_is_clamped_not_an_error():
    basis, solver = _make_solver()

    e_ref, psi_refs = solver.get_eigenvectors(_h_op(), num_wanted=10 * len(basis), max_energy=None)

    assert len(e_ref) == psi_refs.width == len(basis)


def _uniform_refs(basis, value):
    """A one-column warm block holding ``value`` on every determinant this rank owns.

    Built from ``local_basis`` directly rather than through ``redistribute_psis``: the point is
    to hand ``get_eigenvectors`` a block in a specific numerical state on every rank, and a
    redistribute would sum replicas and change the value (see the note on ``_warm_refs``).
    """
    return [ManyBodyState({det: value for det in basis.local_basis}, width=1)]


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
    e_ref, psi_refs = solver.get_eigenvectors(
        _h_op(), num_wanted=1, max_energy=CUT, dense_cutoff=1, psi_refs=_high_energy_refs(basis)
    )
    assert abs(float(np.min(e_ref.real)) - _exact_energies()[0]) < 1e-9
    assert psi_refs.width == len(e_ref) > 0


def test_a_cold_start_finds_the_true_ground_state():
    basis, solver = _make_solver()
    e_ref, _ = solver.get_eigenvectors(_h_op(), num_wanted=1, max_energy=CUT, dense_cutoff=1)
    assert abs(float(np.min(e_ref.real)) - _exact_energies()[0]) < 1e-9


def test_a_non_finite_warm_block_falls_back_to_the_generic_start():
    """A corrupted warm block costs iterations, not the whole calculation.

    The regression: a production double-counting search died when ``block_normalize`` raised on the
    warm-started start block and nothing caught it.
    """
    basis, solver = _make_solver()
    e_ref, psi_refs = solver.get_eigenvectors(
        _h_op(), num_wanted=1, max_energy=CUT, dense_cutoff=1, psi_refs=_uniform_refs(basis, complex(np.nan, 0.0))
    )
    assert abs(float(np.min(e_ref.real)) - _exact_energies()[0]) < 1e-9
    assert psi_refs.width == len(e_ref) > 0


def test_a_zero_warm_block_needs_no_fallback_because_the_generic_columns_carry_it():
    basis, solver = _make_solver()
    e_ref, _ = solver.get_eigenvectors(
        _h_op(), num_wanted=1, max_energy=CUT, dense_cutoff=1, psi_refs=_uniform_refs(basis, 0.0)
    )
    assert abs(float(np.min(e_ref.real)) - _exact_energies()[0]) < 1e-9
