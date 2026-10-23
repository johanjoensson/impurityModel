r"""``sector_solver``: every symmetry class that can hold the ground state is expanded, and the lowest wins.

The fixture is ``test_seed_sectors``' two-orbital crossover model, where a single expansion from the
seed's lowest state is trapped in the atomic (high-spin) multiplet. The oracle is exact
diagonalization of the whole sector.
"""

import numpy as np
import pytest
from mpi4py import MPI

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


def _verdict(k, energy, n_imp, residual=0.0, stopped=False):
    from impurityModel.ed.sector_solver import CertificationJob, ClassVerdict

    job = CertificationJob(two_sz=k, parity=(), seed_class=None, n_seed=1)
    return job, ClassVerdict(job, energy, 10, residual, False, False, n_imp, stopped)


def test_the_store_keeps_the_classes_that_can_still_win_and_widens_with_the_shift():
    from impurityModel.ed.sector_solver import CertificationPlan, SectorPlanStore

    jobs, verdicts = zip(_verdict(0, -1.00, 8.6), _verdict(1, -0.95, 8.7), _verdict(2, -0.50, 8.6))
    store = SectorPlanStore()
    assert store.jobs_at("N6", 0.0) is None, "an uncertified sector must be certified in full"
    store.record("N6", CertificationPlan(list(jobs), {}, [], False, {}), 0.0, verdicts)
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


@pytest.mark.parametrize("comm", COMMS)
def test_the_store_is_reused_only_at_its_own_accuracy_and_never_for_a_mixed_valence_seed(comm, monkeypatch):
    """A record says which classes can win *at the accuracy and seed it was certified with*. At a
    different ``e_pt2_tol`` the sector is certified in full again; a mixed-valence seed spans several
    impurity occupations, on which a shift is not a constant, so it is counted but never reused."""
    from impurityModel.ed.groundstate import solve_sector
    from impurityModel.ed.sector_solver import SectorPlanStore

    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    expansions = {"n": 0}
    real = CIPSISolver.expand

    def counting(self, *args, **kwargs):
        expansions["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(CIPSISolver, "expand", counting)

    def solve(mu, store, e_pt2_tol=1e-12, mixed_valence=None):
        expansions["n"] = 0
        solve_sector(
            _hamiltonian(mu=mu), IMPURITY_ORBITALS, BATH_STATES, {0: 2}, mixed_valence, 0, False, 1000,
            manifold=Lowest(), comm=comm, verbose=False, slaterWeightMin=0, de2_min=0, e_pt2_tol=e_pt2_tol,
            plan_store=store, shift=mu,
        )  # fmt: skip
        return expansions["n"]

    store = SectorPlanStore()
    full = solve(0.0, store)
    assert solve(0.01, store) < full, "the store did not skip a class at its own accuracy"
    assert solve(0.01, store, e_pt2_tol=1e-11) == full, "a record was reused at another accuracy"
    mixed = SectorPlanStore()
    full_mixed = solve(0.0, mixed, mixed_valence={0: 1})
    assert solve(0.01, mixed, mixed_valence={0: 1}) == full_mixed, "a mixed-valence record was reused"
    assert mixed.summary()["certified_sectors"] == 1, "a certified mixed-valence sector went uncounted"


@pytest.mark.parametrize("comm", COMMS)
def test_a_sector_too_large_to_certify_is_counted_as_uncertified(comm, monkeypatch):
    """A seed above ``GS_CERTIFY_MAX_SEED`` is solved by one expansion; the store says so, so the
    result record can, and a sector that is certified is counted as such."""
    from impurityModel.ed.groundstate import solve_sector
    from impurityModel.ed.sector_solver import SectorPlanStore

    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)

    def solve(store):
        solve_sector(
            _hamiltonian(), IMPURITY_ORBITALS, BATH_STATES, {0: 2}, None, 0, False, 1000, manifold=Lowest(),
            comm=comm, verbose=False, slaterWeightMin=0, de2_min=0, e_pt2_tol=1e-12, plan_store=store,
        )  # fmt: skip
        counts = store.summary()
        return {key: counts[key] for key in ("certified_sectors", "uncertified_sectors", "recertified_sectors")}

    monkeypatch.setenv("GS_CERTIFY_MAX_SEED", "1")
    counts = solve(SectorPlanStore())
    assert counts == {"certified_sectors": 0, "uncertified_sectors": 1, "recertified_sectors": 0}, counts
    monkeypatch.delenv("GS_CERTIFY_MAX_SEED")
    counts = solve(SectorPlanStore())
    assert counts == {"certified_sectors": 1, "uncertified_sectors": 0, "recertified_sectors": 0}, counts


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
    assert result.solver.manifold_check == {"found": 8, "expected": 8}


def test_a_thermal_manifold_short_of_its_level_is_reported(monkeypatch, capsys):
    """The completeness check, fed the 8-fold cubic d7 level with three of its states dropped: the
    two hard labels at the winning energy, times the ``2S + 1 = 4`` of its ``<S^2>``, bound it at 8."""
    from impurityModel.ed import sector_solver
    from impurityModel.ed.eigensolvers import Within
    from impurityModel.ed.groundstate import solve_sector
    from impurityModel.test.support.cubic_d_shell import cubic_d_shell

    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    seen = {}
    real = sector_solver.manifold_deficit

    def keep(plan, verdicts, es, psis, basis):
        seen.update(plan=plan, verdicts=verdicts, es=es, psis=psis, basis=basis)
        return real(plan, verdicts, es, psis, basis)

    monkeypatch.setattr(sector_solver, "manifold_deficit", keep)
    _h, h_op, imp, baths = cubic_d_shell(n_bath_sets=1)
    solve_sector(
        h_op, imp, baths, {0: 7}, None, 0, False, 1000, manifold=Within(1e-6), comm=None, verbose=False,
        slaterWeightMin=0, de2_min=0, e_pt2_tol=1e-8,
    )  # fmt: skip
    order = np.argsort(np.real(seen["es"]))[:5]
    short = real(
        seen["plan"], seen["verdicts"], np.asarray(seen["es"])[order], seen["psis"].select(list(order)), seen["basis"]
    )
    assert short == (5, 8)


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


def test_a_leaked_class_stops_only_once_it_has_settled_near_a_finished_one():
    from impurityModel.ed.sector_solver import LEAK_ENERGY_TOL, LEAK_PLATEAU_TOL, _has_leaked

    finished = [-1.0]
    near = -1.0 + 0.5 * LEAK_ENERGY_TOL
    assert _has_leaked(0.0, near, near + 0.1 * LEAK_PLATEAU_TOL, finished)
    # Still descending through the band: it may be on its way to a new, lower level.
    assert not _has_leaked(0.0, near, near + 10 * LEAK_PLATEAU_TOL, finished)
    # Still in its own class, or not near any finished level.
    assert not _has_leaked(0.9, near, near, finished)
    assert not _has_leaked(0.0, -0.5, -0.5, finished)


def test_the_store_expands_a_stopped_class_with_the_class_it_leaked_into():
    from impurityModel.ed.sector_solver import CertificationPlan, SectorPlanStore

    # Class 2 leaked into class 1's state (stopped near -0.95): its energy bounds nothing.
    jobs, verdicts = zip(_verdict(0, -1.00, 8.6), _verdict(1, -0.95, 8.7), _verdict(2, -0.9496, 8.6, stopped=True))
    store = SectorPlanStore()
    store.record("N6", CertificationPlan(list(jobs), {}, [], False, {}), 0.0, verdicts)
    assert store.jobs_at("N6", 0.0) == {0}
    assert store.jobs_at("N6", 0.1) == {0, 1, 2}


def test_a_solve_above_its_certified_tangent_is_a_violation():
    from impurityModel.ed.sector_solver import CertificationPlan, SectorPlanStore

    jobs, verdicts = zip(_verdict(0, -1.00, 8.6), _verdict(1, -0.95, 8.7))
    store = SectorPlanStore()
    store.record("N6", CertificationPlan(list(jobs), {}, [], False, {}), 0.0, verdicts)
    # At mu = 0.01 the tangent predicts -1.086 for class 0; landing there is fine, 0.01 above is not.
    _job, on_tangent = _verdict(0, -1.0 - 0.01 * 8.6, 8.6)
    _job, above = _verdict(0, -1.0 - 0.01 * 8.6 + 0.01, 8.6)
    assert not store.violated("N6", 0.01, [on_tangent])
    assert store.violated("N6", 0.01, [above])


def test_the_ground_state_certifies_its_winning_sector_once(monkeypatch):
    """``solve_ground_state``'s walk and thermal solve share one store: the thermal solve expands only
    the classes inside its window, not every class of the sector a second time."""
    from impurityModel.ed import sector_solver
    from impurityModel.ed.groundstate import solve_ground_state

    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    calls = {"n": 0}
    real = sector_solver._expand_job

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sector_solver, "_expand_job", counted)

    def run():
        calls["n"] = 0
        gs = solve_ground_state(
            _hamiltonian(), IMPURITY_ORBITALS, BATH_STATES, {0: 2}, tau=1e-3, slaterWeightMin=0, comm=None,
            verbose=0, de2_min=0, e_pt2_tol=1e-12, use_hf_seed=False,
        )  # fmt: skip
        return np.sort(np.real(gs.es)), calls["n"]

    es_shared, n_shared = run()
    monkeypatch.setattr(sector_solver.SectorPlanStore, "jobs_at", lambda self, key, mu, window=0.0: None)
    es_fresh, n_fresh = run()
    assert n_shared < n_fresh
    np.testing.assert_allclose(es_shared, es_fresh, atol=1e-10)


def test_the_window_overtake_is_the_crossover_measured_against_the_window(monkeypatch):
    """The crossover toy's winner overtakes the seed-lowest class (the atomic triplet): the ratio is
    that overtake, in units of the winner's own spread, over the window's ``window / sigma_max`` --
    so it halves when the window factor doubles, and is positive here."""
    from impurityModel.ed.sector_solver import window_overtake

    monkeypatch.delenv("GS_CERTIFY_WINDOW", raising=False)
    ratios = []
    for factor in ("3", "6"):
        monkeypatch.setenv("GS_CERTIFY_WINDOW_FACTOR", factor)
        _energy, plan, verdicts = _certify(_hamiltonian(), None)
        ratios.append(window_overtake(plan, verdicts))
    assert ratios[0] is not None and ratios[0] > 0.0, ratios
    assert ratios[1] == pytest.approx(ratios[0] / 2, rel=1e-6)


def test_the_ranking_rule_never_settles_the_nio_d6_winner_and_bounds_a_loser_above_it():
    """Pass 1's stopping rule against recorded NiO 10-bath d6 trajectories (per-cycle class energies).

    The winner's drops shrink once (0.021) before it crosses into the ground state (0.042, 0.064):
    no prefix of its trajectory may count as settled, or pass 1 would rank the ground state 0.13
    high. A loser that has settled must be bounded above the winner's energy, or pass 2 would waste
    a full expansion on it."""
    from impurityModel.ed.sector_solver import _ranking_remaining

    winner = [-41.09834, -42.45474, -42.97355, -43.13595, -43.15667, -43.19871, -43.26283, -43.32773, -43.32777]
    assert all(_ranking_remaining(winner[: k + 1]) is None for k in range(len(winner) - 1))
    loser = [-41.38547, -42.46083, -43.01510, -43.14906, -43.16474, -43.16816, -43.18348, -43.19211, -43.19666,
             -43.19897, -43.20010, -43.20066, -43.20094, -43.20103, -43.20105]  # fmt: skip
    settled = [k for k in range(len(loser)) if _ranking_remaining(loser[: k + 1]) is not None]
    assert settled, "the loser never settled"
    k = settled[0]
    assert loser[k] - _ranking_remaining(loser[: k + 1]) > winner[-1] + 0.05
    assert _ranking_remaining([-1.0, -1.0005, -1.0004, -1.0006]) is None  # a rise is not settling


def test_a_class_whose_state_left_its_class_is_marked_leaked_even_unstopped(monkeypatch):
    """A class that converged into another class's state without being stopped (nothing had finished
    there yet) carries that state's energy; at another shift it may stay in its own state, far
    above, which the store read as a broken tangent (NiO d5: 0.18 above, the sector re-certified).
    Its verdict must say it leaked, so the store neither checks nor trusts it."""
    from impurityModel.ed import seed_sectors

    monkeypatch.setattr(seed_sectors, "class_weight", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr("impurityModel.ed.sector_solver._has_leaked", lambda *args: False)
    _energy, _plan, verdicts = _certify(_hamiltonian(), None)
    classed = [v for v in verdicts if v.job.seed_class is not None]
    assert classed, "the fixture has no class expansions"
    assert all(v.stopped_early for v in classed)


def test_davidson_runs_only_where_a_class_penalty_holds_the_expansion(monkeypatch):
    """Davidson stays in its start's symmetry sector, so it is safe only under a ``ClassPenalty``.

    A label with at most one class in the window is expanded whole (``seed_class=None``) with no
    penalty: a plain expansion inside the hard label, where spin and orbital traps remain and
    only Lanczos growing the generic start vector escapes them. A dense cutoff of 1 sends every
    solve down the sparse path, where the choice is made.
    """
    h_op = _hamiltonian()
    basis = _seed(h_op, None)
    plan = certification_plan(h_op, basis)
    assert any(job.seed_class is None for job in plan.jobs)
    calls = []
    original = CIPSISolver._solve_lowest

    def record(self, *args, **kwargs):
        calls.append((self.davidson, self.matrix_hook is not None))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CIPSISolver, "_solve_lowest", record)
    sparse = dict(EXPAND, dense_cutoff=1)
    es, *_rest = certify_sector(h_op, basis, plan, expand_kwargs=sparse, dense_cutoff=1, slater_weight_min=0)
    assert any(davidson for davidson, _ in calls)
    assert any(not hooked for _, hooked in calls)
    assert all(hooked for davidson, hooked in calls if davidson), calls
    assert abs(float(np.min(es)) - _exact_ground_energy(h_op)) < 1e-8


def test_a_class_verdict_below_the_dense_cutoff_is_a_lanczos_solve_on_rank_0(monkeypatch):
    """The verdict -- the unpenalized lowest state of a class basis, in whatever sector -- is solved
    by Lanczos from the generic start vector (never Davidson), serially on rank 0 below the dense
    cutoff, and is exact. ``CLASS_DENSE_MAX=1`` puts this fixture's small bases above that threshold."""
    import impurityModel.ed.cipsi_solver as cs

    monkeypatch.setattr(cs, "CLASS_DENSE_MAX", 1)
    h_op = _hamiltonian()
    basis = _seed(h_op, None)
    plan = certification_plan(h_op, basis)
    calls = []
    original = CIPSISolver._solve_lowest

    def record(self, *args, serial=False, **kwargs):
        calls.append((self.davidson, serial))
        return original(self, *args, serial=serial, **kwargs)

    monkeypatch.setattr(CIPSISolver, "_solve_lowest", record)
    es, *_rest = certify_sector(h_op, basis, plan, expand_kwargs=EXPAND, dense_cutoff=1000, slater_weight_min=0)
    assert (False, True) in calls, calls
    assert abs(float(np.min(es)) - _exact_ground_energy(h_op)) < 1e-8


@pytest.mark.parametrize("comm", COMMS)
def test_a_sparse_lowest_solve_of_few_nonzeros_runs_on_rank_0(comm, monkeypatch):
    """Above the dense cutoff, a ``Lowest`` solve whose operator has at most ``LOWEST_SERIAL_NNZ``
    nonzeros is gathered onto rank 0 (``serial``): distributed, every iteration of so small a
    problem is collective waiting. Above the bound it stays distributed; the energy is the same."""
    import impurityModel.ed.cipsi_solver as cs

    h_op = _hamiltonian()
    energies = []
    calls = []
    original = cs.solve_lowest

    def record(*args, serial=False, **kwargs):
        calls.append(serial)
        return original(*args, serial=serial, **kwargs)

    monkeypatch.setattr(cs, "solve_lowest", record)
    for bound, expected in ((cs.LOWEST_SERIAL_NNZ, True), (0, False)):
        monkeypatch.setattr(cs, "LOWEST_SERIAL_NNZ", bound)
        calls.clear()
        solver = CIPSISolver(_seed(h_op, comm))
        es, _psis = solver.get_eigenvectors(h_op, Lowest(), dense_cutoff=1, slaterWeightMin=0)
        assert calls == [expected]
        energies.append(float(np.min(es)))
    assert abs(energies[0] - energies[1]) < 1e-10


def test_spin_partners_stop_at_the_largest_seed_two_sz():
    """The thermal union's partner rows reach every ``|2 S_z| <= max_two_sz`` and nothing beyond: no
    state of the window has a row there, and on NiO those rows were a quarter of the union."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant
    from impurityModel.ed.sector_solver import _bits, _spin_partner_keys

    pairs = [(0, 1), (2, 3), (4, 5)]  # (dn, up)
    det = SlaterDeterminant(((1 << 63) | (1 << 61) | (1 << 58),))  # 0 (dn), 2 (dn), 5 (up): 2 S_z = -1
    found = _spin_partner_keys(ManyBodyState.from_keys([det]), pairs, max_two_sz=1, n_orb=8)
    sign = np.zeros(8, dtype=np.int64)
    for dn, up in pairs:
        sign[up], sign[dn] = 1, -1
    rows = set((_bits(list(found), 8).astype(np.int64) @ sign).tolist())
    assert rows == {1}, rows
    # The unbounded ladder would also have reached 2 S_z = 3 from here.
    wider = _spin_partner_keys(ManyBodyState.from_keys([det]), pairs, max_two_sz=3, n_orb=8)
    assert 3 in set((_bits(list(wider), 8).astype(np.int64) @ sign).tolist())


def _union_size_at_the_window_solve(monkeypatch, cap, capsys=None):
    """``(sizes of the basis the thermal window's expansion starts on, the solve's result)`` on cubic d7."""
    from impurityModel.ed import sector_solver
    from impurityModel.ed.eigensolvers import Within
    from impurityModel.ed.groundstate import solve_sector
    from impurityModel.test.support.cubic_d_shell import cubic_d_shell

    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    sizes = []
    real = sector_solver.CIPSISolver

    class Spy(real):
        def expand(self, h_op, manifold, **kwargs):
            sizes.append(self.basis.size)
            return super().expand(h_op, manifold, **kwargs)

    monkeypatch.setattr(sector_solver, "CIPSISolver", Spy)
    _h, h_op, imp, baths = cubic_d_shell(n_bath_sets=1)
    result = solve_sector(
        h_op, imp, baths, {0: 7}, None, 0, False, 1000, manifold=Within(1e-6), comm=None, verbose=False,
        slaterWeightMin=0, de2_min=0, e_pt2_tol=1e-8, truncation_threshold=cap,
    )  # fmt: skip
    return sizes, result


def test_the_thermal_union_is_built_within_the_cap_and_unchanged_when_it_fits(monkeypatch):
    """The window solve costs more per determinant than a class expansion, so the union it starts on must
    not pass the cap (it used to: whole classes and all their spin partners went in before the first check,
    and only the expansion's exit truncated). A union that fits is the same set as before."""
    natural, wide = _union_size_at_the_window_solve(monkeypatch, 100_000)
    union = natural[-1]
    assert wide.solver.manifold_check == {"found": 8, "expected": 8}
    assert union == 240  # the seed's 120 determinants and the partner rows of the 8-fold level

    cap = union - 40  # under what the winning level's classes and their partner rows need
    sizes, capped = _union_size_at_the_window_solve(monkeypatch, cap)
    assert sizes[-1] <= cap < union, (sizes[-1], cap, union)
    # The classes at the winning energy are cut by one common share, not one starved for another, and
    # the shortfall is reported rather than hidden: a cut union cannot hold the whole 8-fold level.
    assert cap - 20 <= sizes[-1] <= cap
    found, expected = capped.solver.manifold_check["found"], capped.solver.manifold_check["expected"]
    assert 0 < found < expected == 8
