r"""``sector_solver``: every symmetry class that can hold the ground state is expanded, and the lowest wins.

The fixture is ``test_seed_sectors``' two-orbital crossover model, where a single expansion from the
seed's lowest state is trapped in the atomic (high-spin) multiplet. The oracle is exact
diagonalization of the whole sector.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_sparse_matrix
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.eigensolvers import Lowest
from impurityModel.ed.groundstate import build_basis_and_solver
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.sector_solver import certification_plan, certify_sector
from impurityModel.test.support.comms import COMMS
from impurityModel.test.symmetry.test_seed_sectors import (
    BATH_STATES,
    IMPURITY_ORBITALS,
    _exact_ground_energy,
    _hamiltonian,
)

EXPAND = dict(dense_cutoff=1000, slaterWeightMin=0, de2_min=0, e_pt2_tol=1e-12)


def _seed(h_op, comm):
    basis, _solver = build_basis_and_solver(
        h_op, IMPURITY_ORBITALS, BATH_STATES, {0: 2}, None, 0, False, None, comm, False, None, None, 0,
        manifold=Lowest(),
    )  # fmt: skip
    return basis


def _certify(h_op, comm):
    basis = _seed(h_op, comm)
    plan = certification_plan(h_op, basis)
    es, _psis, _cb, _cs, verdicts = certify_sector(
        h_op, basis, plan, expand_kwargs=EXPAND, dense_cutoff=1000, slater_weight_min=0
    )
    e0 = float(np.min(es))
    return (e0 if comm is None else comm.bcast(e0, root=0)), plan, verdicts


def _single_expansion(h_op, comm):
    basis = _seed(h_op, comm)
    solver = CIPSISolver(basis)
    solver.expand(h_op, Lowest(), **EXPAND)
    es, _psis = solver.get_eigenvectors(h_op, Lowest(), dense_cutoff=1000, slaterWeightMin=0, reuse_last=True)
    e0 = float(np.min(es))
    return e0 if comm is None else comm.bcast(e0, root=0)


@pytest.mark.parametrize("comm", COMMS)
def test_certification_reaches_the_ground_state_a_single_expansion_misses(comm):
    h_op = _hamiltonian()
    exact = _exact_ground_energy(h_op)
    trapped = _single_expansion(_hamiltonian(), comm)
    assert trapped > exact + 1e-3, "the fixture no longer traps a single expansion"
    energy, plan, _verdicts = _certify(_hamiltonian(), comm)
    assert plan.su2
    assert abs(energy - exact) < 1e-8


@pytest.mark.parametrize("comm", COMMS)
def test_certification_without_spin_symmetry_uses_the_parity_labels(comm):
    """``spin_mixing`` breaks ``S`` and ``S_z``: only the orbital parity is left to split the seed."""
    exact = _exact_ground_energy(_hamiltonian(spin_mixing=0.3))
    energy, plan, _verdicts = _certify(_hamiltonian(spin_mixing=0.3), comm)
    assert not plan.su2
    assert all(job.two_sz is None and job.spin is None for job in plan.jobs)
    assert abs(energy - exact) < 1e-8


@pytest.mark.mpi
def test_the_verdicts_are_identical_on_every_rank():
    _energy, plan, verdicts = _certify(_hamiltonian(), MPI.COMM_WORLD)
    gathered = MPI.COMM_WORLD.allgather([(v.job, v.energy) for v in verdicts])
    assert all(g == gathered[0] for g in gathered)
    assert all(p == plan.jobs for p in MPI.COMM_WORLD.allgather(plan.jobs))


@pytest.mark.parametrize("comm", COMMS)
def test_the_matrix_penalty_is_added_to_the_built_matrix(comm):
    h_op = _hamiltonian()
    basis = _seed(h_op, comm)
    solver = CIPSISolver(basis)
    penalty = ManyBodyOperator({((0, "c"), (0, "a")): 0.75, ((2, "c"), (2, "a")): 0.25})
    solver.matrix_penalty = penalty
    got = solver._build_h_matrix(h_op).to_scipy()
    expected = build_sparse_matrix(basis, h_op + penalty).to_scipy()
    assert abs(got - expected).max() < 1e-14 if (got - expected).nnz else True


def _spin_squared(psis, comm):
    from impurityModel.ed.observables import casimir_operator, expect_casimir, make_spin_operators
    from impurityModel.ed.spin_pairs import bath_spin_pairs, impurity_spin_pairs
    from impurityModel.ed.ManyBodyUtils import ManyBodyState

    pairs = impurity_spin_pairs(IMPURITY_ORBITALS) + bath_spin_pairs(BATH_STATES)
    sp, sm, sz = make_spin_operators(pairs)
    (state,) = psis.select([0]).to_states() if psis.width > 1 else psis.to_states()
    local = {det: complex(np.ravel(row)[0]) for det, row in state.items()}
    parts = comm.allgather(local) if comm is not None else [local]
    whole = {k: v for part in parts for k, v in part.items()}
    norm = sum(abs(v) ** 2 for v in whole.values())
    return expect_casimir(ManyBodyState(whole), sp, sm, sz) / norm


@pytest.mark.parametrize("comm", COMMS)
def test_the_penalty_steers_the_expansion_references(comm):
    """``matrix_penalty`` acts in every eigensolve of ``expand``. A plain expansion of this model is
    the trapped one: its reference stays in the seed's high-spin multiplet. ``lambda S^2`` must
    steer it to the low-spin state."""
    from impurityModel.ed.observables import casimir_operator, make_spin_operators
    from impurityModel.ed.spin_pairs import bath_spin_pairs, impurity_spin_pairs

    h_op = _hamiltonian()
    pairs = impurity_spin_pairs(IMPURITY_ORBITALS) + bath_spin_pairs(BATH_STATES)
    s2 = casimir_operator(*make_spin_operators(pairs))
    plain = CIPSISolver(_seed(h_op, comm))
    plain.expand(h_op, Lowest(), **EXPAND)
    steered = CIPSISolver(_seed(h_op, comm))
    steered.matrix_penalty = 10.0 * s2
    steered.expand(h_op, Lowest(), **EXPAND)
    assert _spin_squared(plain.psi_refs, comm) > 1.5
    assert _spin_squared(steered.psi_refs, comm) < 0.5


def test_every_spin_tier_job_expands_with_its_penalty(monkeypatch):
    seen = []
    real = CIPSISolver.expand

    def recording(self, *args, **kwargs):
        seen.append(self.matrix_penalty is not None)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(CIPSISolver, "expand", recording)
    _energy, plan, _verdicts = _certify(_hamiltonian(), None)
    expected = [job.spin is not None for job in plan.jobs if job.duplicate_of is None]
    assert any(expected), "the fixture has no spin tier below the top spin"
    assert seen == expected
