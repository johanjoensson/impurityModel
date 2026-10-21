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
    assert all(job.two_sz is None for job in plan.jobs)
    assert abs(energy - exact) < 1e-8


@pytest.mark.mpi
def test_the_verdicts_are_identical_on_every_rank():
    _energy, plan, verdicts = _certify(_hamiltonian(), MPI.COMM_WORLD)
    gathered = MPI.COMM_WORLD.allgather([(v.job, v.energy) for v in verdicts])
    assert all(g == gathered[0] for g in gathered)
    assert all(p == plan.jobs for p in MPI.COMM_WORLD.allgather(plan.jobs))


def _solve_sector(h_op, comm, manifold):
    from impurityModel.ed.groundstate import solve_sector

    result = solve_sector(
        h_op, IMPURITY_ORBITALS, BATH_STATES, {0: 2}, None, 0, False, 1000, manifold=manifold, comm=comm,
        verbose=False, slaterWeightMin=0, de2_min=0, e_pt2_tol=1e-12,
    )  # fmt: skip
    e0 = float(np.min(np.real(result.es)))
    return (e0 if comm is None else comm.bcast(e0, root=0)), result


@pytest.mark.parametrize("comm", COMMS)
def test_solve_sector_certifies_by_default_and_the_switch_restores_a_single_expansion(comm, monkeypatch):
    exact = _exact_ground_energy(_hamiltonian())
    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    certified, _result = _solve_sector(_hamiltonian(), comm, Lowest())
    monkeypatch.setenv("GS_SECTOR_CERTIFY", "0")
    single, _result = _solve_sector(_hamiltonian(), comm, Lowest())
    assert abs(certified - exact) < 1e-8
    assert single > exact + 1e-3


@pytest.mark.parametrize("comm", COMMS)
def test_the_certified_thermal_manifold_holds_the_whole_ground_multiplet(comm, monkeypatch):
    """``Within`` merges the winner, its spin partners and the classes in the window: a degenerate
    ground multiplet comes back complete, whatever row the winning class expanded."""
    from impurityModel.ed.eigensolvers import Within

    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    h_op = _hamiltonian(mu=-5.0)  # a triplet ground level: its rows span three S_z
    exact = np.sort(np.real(_exact_spectrum(h_op)))
    multiplicity = int(np.sum(exact - exact[0] < 1e-8))
    assert multiplicity > 1, "the fixture's ground level is not degenerate"
    _e0, result = _solve_sector(_hamiltonian(mu=-5.0), comm, Within(1e-6))
    got = np.sort(np.real(result.es))
    assert len(got) == multiplicity
    np.testing.assert_allclose(got, exact[:multiplicity], atol=1e-8)


def _exact_spectrum(h_op):
    import itertools

    from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant, applyOp
    from impurityModel.test.symmetry.test_seed_sectors import N_ELECTRONS, N_ORB

    dets = []
    for occupied in itertools.combinations(range(N_ORB), N_ELECTRONS):
        chunk = 0
        for orb in occupied:
            chunk |= 1 << (63 - orb)
        dets.append(SlaterDeterminant((chunk,)))
    index = {det: i for i, det in enumerate(dets)}
    h_op.set_restrictions(None)
    h_op.set_weighted_restrictions(None)
    h = np.zeros((len(dets), len(dets)), dtype=complex)
    for j, det in enumerate(dets):
        for key, row in applyOp(h_op, ManyBodyState({det: 1.0}), cutoff=0).items():
            h[index[key], j] += row[0]
    return np.linalg.eigvalsh(h)


def _verdict(k, energy, n_imp, residual=0.0):
    from impurityModel.ed.sector_solver import CertificationJob, ClassVerdict

    job = CertificationJob(two_sz=k, parity=(), seed_class=None, n_seed=1)
    return job, ClassVerdict(job, energy, 10, residual, False, False, n_imp)


def test_the_store_keeps_the_classes_that_can_still_win_and_widens_with_the_shift():
    from impurityModel.ed.sector_solver import CertificationPlan, SectorPlanStore

    jobs, verdicts = zip(_verdict(0, -1.00, 8.6), _verdict(1, -0.95, 8.7), _verdict(2, -0.50, 8.6))
    store = SectorPlanStore()
    assert store.jobs_at("N6", 0.0) is None, "an uncertified sector must be certified in full"
    store.record("N6", CertificationPlan(list(jobs), {}, [], False, 0.0), 0.0, verdicts)
    # At the certified shift only the leader is within its error bar.
    assert store.jobs_at("N6", 0.0) == {0}
    # 0.1 away: the runner-up (0.05 above, falling 0.1 * 0.1 faster) is predicted 0.04 above, inside
    # the margin 0.1 * (0.1 spread + 0.5 safety) = 0.06; the third class (0.5 above) is not.
    assert store.jobs_at("N6", 0.1) == {0, 1}
    # Far away every class can have crossed: certify them all again.
    assert store.jobs_at("N6", 2.0) == {0, 1, 2}
    store.clear()
    assert store.jobs_at("N6", 0.05) is None


@pytest.mark.parametrize("comm", COMMS)
def test_a_second_shift_through_the_store_expands_fewer_classes_and_finds_the_same_energy(comm, monkeypatch):
    from impurityModel.ed.groundstate import solve_sector
    from impurityModel.ed.sector_solver import SectorPlanStore

    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    expansions = {"n": 0}
    real = CIPSISolver.expand

    def counting(self, *args, **kwargs):
        expansions["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(CIPSISolver, "expand", counting)

    def solve(mu, store):
        result = solve_sector(
            _hamiltonian(mu=mu), IMPURITY_ORBITALS, BATH_STATES, {0: 2}, None, 0, False, 1000, manifold=Lowest(),
            comm=comm, verbose=False, slaterWeightMin=0, de2_min=0, e_pt2_tol=1e-12, plan_store=store, shift=mu,
        )  # fmt: skip
        e0 = float(np.min(np.real(result.es)))
        return e0 if comm is None else comm.bcast(e0, root=0)

    store = SectorPlanStore()
    solve(0.0, store)
    full = expansions["n"]
    expansions["n"] = 0
    stored = solve(0.01, store)
    partial = expansions["n"]
    fresh = solve(0.01, None)
    assert partial < full
    assert abs(stored - fresh) < 1e-10
    assert abs(stored - _exact_ground_energy(_hamiltonian(mu=0.01))) < 1e-8


def test_with_spin_orbit_coupling_neither_spin_label_is_assumed_and_the_answer_is_exact():
    """Spin need not be a good quantum number: with a spin-mixing term neither ``S`` nor ``S_z`` is
    detected, only the parities split the seed, and the certified energy is still the exact one."""
    import itertools

    from impurityModel.ed.basis_transcription import build_dense_matrix
    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.test.support.cubic_d_shell import cubic_d_shell

    _h, h_op, imp, baths = cubic_d_shell(n_bath_sets=1, soc=0.08)
    n_orb, n_imp_el = 20, 6
    dets = [
        sum((1 << (7 - o % 8)) << (8 * (2 - o // 8)) for o in occ).to_bytes(3, "big")
        for occ in itertools.combinations(range(n_orb), n_imp_el + 10)
    ]
    h_op.set_restrictions(None)
    h_op.set_weighted_restrictions(None)
    exact = float(np.linalg.eigvalsh(build_dense_matrix(Basis(imp, baths, initial_basis=dets, verbose=False), h_op))[0])
    basis, _solver = build_basis_and_solver(
        h_op, imp, baths, {0: n_imp_el}, None, 0, False, None, None, False, None, None, 1e-12, manifold=Lowest()
    )
    plan = certification_plan(h_op, basis)
    # A per-determinant floor, not a residual target: 29 class expansions, seconds rather than minutes.
    loose = dict(dense_cutoff=1000, slaterWeightMin=1e-12, de2_min=1e-6, e_pt2_tol=None)
    assert not plan.su2 and all(job.two_sz is None for job in plan.jobs)
    es, *_rest = certify_sector(h_op, basis, plan, expand_kwargs=loose, dense_cutoff=1000, slater_weight_min=1e-12)
    assert abs(float(np.min(es)) - exact) < 1e-5


def test_a_cycle_hook_ends_the_expansion():
    """``CIPSISolver.cycle_hook`` is what lets a leaked class stop: ``True`` ends ``expand`` there."""
    h_op = _hamiltonian()
    solver = CIPSISolver(_seed(h_op, None))
    seed_size = len(solver.basis)
    calls = []
    solver.cycle_hook = lambda sv, e_ref, psi_refs: calls.append(len(sv.basis)) or True
    solver.expand(h_op, Lowest(), **EXPAND)
    assert solver.stopped_early and calls == [seed_size] and len(solver.basis) == seed_size


def test_a_spin_and_orbitally_degenerate_thermal_level_comes_back_whole(monkeypatch):
    """Review regression: cubic d7 with one bath set has an 8-fold ground level (spin x orbital). The
    certified ``Within`` returned 5 of its states -- spin partners were generated from the winning
    class only, so the other in-window class's rows stayed at seed quality -- where a plain
    expansion returns all 8. Exact diagonalization of the whole sector is the oracle."""
    import itertools

    from impurityModel.ed.basis_transcription import build_dense_matrix
    from impurityModel.ed.eigensolvers import Within
    from impurityModel.ed.groundstate import solve_sector
    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.test.support.cubic_d_shell import cubic_d_shell

    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    n_imp_el, n_orb = 7, 20
    _h, h_op, imp, baths = cubic_d_shell(n_bath_sets=1)
    dets = [
        sum((1 << (7 - o % 8)) << (8 * (2 - o // 8)) for o in occ).to_bytes(3, "big")
        for occ in itertools.combinations(range(n_orb), n_imp_el + 10)
    ]
    h_op.set_restrictions(None)
    h_op.set_weighted_restrictions(None)
    exact = np.linalg.eigvalsh(build_dense_matrix(Basis(imp, baths, initial_basis=dets, verbose=False), h_op))
    multiplicity = int(np.sum(exact - exact[0] < 1e-6))
    assert multiplicity == 8 and exact[multiplicity] - exact[0] > 0.1
    _h, h_op, imp, baths = cubic_d_shell(n_bath_sets=1)
    result = solve_sector(
        h_op, imp, baths, {0: n_imp_el}, None, 0, False, 1000, manifold=Within(1e-6), comm=None, verbose=False,
        slaterWeightMin=0, de2_min=0, e_pt2_tol=1e-8,
    )  # fmt: skip
    got = np.sort(np.real(result.es))
    assert len(got) == multiplicity
    np.testing.assert_allclose(got, exact[:multiplicity], atol=1e-6)


@pytest.mark.parametrize("eps_b", [-1.0, -1.5, -3.0])
def test_labels_with_equal_seed_spectra_but_different_baths_are_both_expanded(eps_b):
    """Review regression: two orbitals with the same onsite level and hybridization but differently
    fitted bath levels give two parity labels with identical seed spectra of ``P H P`` and
    ``P H Q H P`` -- the bath energies enter only through ``Q H Q``. Deduplicating on those spectra
    expanded one label only and missed the ground state by 0.04 (``eps_b = -1.5``) to 0.12 (-3.0)."""
    from impurityModel.ed import atomic_physics
    from impurityModel.ed.interaction_models import kanamori_u4
    from impurityModel.ed.lie_algebra import tensors_to_operator
    from impurityModel.test.symmetry.test_seed_sectors import N_IMP, N_ORB

    def hamiltonian():
        h = np.zeros((N_ORB, N_ORB), dtype=complex)
        for spin in range(2):
            for orbital, eps in enumerate((-1.0, eps_b)):
                i = orbital + 2 * spin
                h[i + N_IMP, i + N_IMP] = eps
                h[i, i + N_IMP] = h[i + N_IMP, i] = 0.8
        u = ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(kanamori_u4(2, 4.0, 0.8)))
        return tensors_to_operator(h) + u

    exact = _exact_ground_energy(hamiltonian(), n_electrons=5)
    h_op = hamiltonian()
    basis, _solver = build_basis_and_solver(
        h_op, IMPURITY_ORBITALS, BATH_STATES, {0: 1}, None, 0, False, None, None, False, None, None, 0,
        manifold=Lowest(),
    )  # fmt: skip
    plan = certification_plan(h_op, basis)
    es, *_rest = certify_sector(h_op, basis, plan, expand_kwargs=EXPAND, dense_cutoff=1000, slater_weight_min=0)
    assert abs(float(np.min(es)) - exact) < 1e-8


def test_a_class_held_by_the_cap_marks_the_sector_solve_cap_bound(monkeypatch):
    """A class can lose *because* the cap held it high. The sector solve callers read (the DC's cap
    calibration through ``truncation_report``) must say so even when the winner itself was unbound."""
    from impurityModel.ed import groundstate
    from impurityModel.ed.sector_solver import CertificationJob, ClassVerdict

    real = groundstate.solve_certified

    def with_a_capped_loser(*args, **kwargs):
        es, psis, sb, solver, verdicts = real(*args, **kwargs)
        assert solver.truncation_report is None, "the fixture's winner is capped itself"
        loser = ClassVerdict(
            CertificationJob(0, (1,), None, 1), es[0] + 1.0, 10, 0.0, True, False,
            truncation_report={"cap_hit": True, "threshold": 10, "memory_bound": False},
        )  # fmt: skip
        return es, psis, sb, solver, [*verdicts, loser]

    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    monkeypatch.setattr(groundstate, "solve_certified", with_a_capped_loser)
    _e0, result = _solve_sector(_hamiltonian(), None, Lowest())
    report = result.solver.truncation_report
    assert report is not None and report["cap_hit"] and "losing_class" in report
