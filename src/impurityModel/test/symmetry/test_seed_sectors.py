"""Seed symmetry classes (``seed_sectors``), and the sector solve the certifier builds on them.

A CIPSI expansion cannot leave the symmetry sectors of its reference, and the first reference is
the seed's ground level -- the atomic multiplet, before hybridization. The fixture here is built so
that hybridization reorders the multiplets: two orbitals with Hund's coupling and a crystal field
just below the atomic high-spin/low-spin crossover, the upper orbital strongly hybridized. Its
atomic ground state is the S = 1 triplet; the true ground state is a singlet that gains more
covalency, because it leaves the strongly hybridized orbital empty. Every expected energy is an
exact diagonalization of the full 28-determinant sector.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import atomic_physics
from impurityModel.ed.eigensolvers import Lowest, Within
from impurityModel.ed.groundstate import build_basis_and_solver, solve_sector
from impurityModel.ed.interaction_models import kanamori_u4
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp
from impurityModel.ed.seed_sectors import _partition, seed_class_plan

N_IMP = 4  # two spatial orbitals, down-then-up: 0, 1 down; 2, 3 up
N_ORB = 2 * N_IMP  # one valence bath site per impurity spin-orbital
N_ELECTRONS = 6  # d2 with the bath full
IMPURITY_ORBITALS = {0: [list(range(N_IMP))]}
BATH_STATES: tuple[dict, dict] = ({0: [list(range(N_IMP, N_ORB))]}, {0: []})


@pytest.fixture(autouse=True)
def _certifier_defaults(monkeypatch):
    """The certifier on (its default) and the window knobs cleared, so a value in the developer's
    environment cannot leak in."""
    monkeypatch.delenv("GS_SECTOR_CERTIFY", raising=False)
    monkeypatch.delenv("GS_CERTIFY_WINDOW", raising=False)
    monkeypatch.delenv("GS_CERTIFY_WINDOW_FACTOR", raising=False)


def _hamiltonian(spin_mixing=0.0, mu=0.0):
    """The two-orbital crossover model; ``spin_mixing`` breaks S but keeps the orbital parity."""
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    for spin in range(2):
        for orbital, (energy, hopping) in enumerate(((-1.0, 0.2), (1.0, 1.5))):
            i = orbital + 2 * spin
            h[i, i] = energy - mu
            h[i + N_IMP, i + N_IMP] = -1.0
            h[i, i + N_IMP] = h[i + N_IMP, i] = hopping
    for orbital in range(2):
        h[orbital, orbital + 2] = h[orbital + 2, orbital] = spin_mixing
    u_op = ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(kanamori_u4(2, 4.0, 0.8)))
    return tensors_to_operator(h) + u_op


def _exact_ground_energy(h_op, n_orb=N_ORB, n_electrons=N_ELECTRONS):
    dets = []
    for occupied in itertools.combinations(range(n_orb), n_electrons):
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
    return float(np.linalg.eigvalsh(h)[0])


def _sector_energy(h_op, manifold=None):
    result = solve_sector(
        h_op,
        IMPURITY_ORBITALS,
        BATH_STATES,
        {0: 2},
        None,
        0,
        False,
        1000,
        manifold=Lowest() if manifold is None else manifold,
        comm=MPI.COMM_WORLD,
        verbose=False,
        slaterWeightMin=0,
    )
    return float(np.min(np.real(result.es)))


def _plan(h_op):
    basis, _solver = build_basis_and_solver(
        h_op, IMPURITY_ORBITALS, BATH_STATES, {0: 2}, None, 0, False, None, MPI.COMM_WORLD, False, None, None, 0,
        manifold=Lowest(),
    )  # fmt: skip
    return seed_class_plan(h_op, basis)


@pytest.mark.mpi
def test_a_single_expansion_is_trapped_in_the_seed_ground_multiplet(monkeypatch):
    """The failure this module exists for, pinned so a fixture change cannot quietly remove it."""
    h_op = _hamiltonian()
    monkeypatch.setenv("GS_SECTOR_CERTIFY", "0")
    assert _sector_energy(h_op) - _exact_ground_energy(h_op) > 0.1


@pytest.mark.mpi
def test_the_class_expansions_reach_the_exact_ground_state_across_a_spin_crossover():
    h_op = _hamiltonian()
    assert _sector_energy(h_op) == pytest.approx(_exact_ground_energy(h_op), abs=1e-6)


@pytest.mark.mpi
def test_the_thermal_manifold_is_built_on_the_winning_class():
    h_op = _hamiltonian()
    assert _sector_energy(h_op, Within(1e-3)) == pytest.approx(_exact_ground_energy(h_op), abs=1e-6)


@pytest.mark.mpi
def test_the_partition_separates_the_multiplets_without_labelling_them():
    """No spin operator is used, yet the seed's triplet comes out as one class -- its three rows are
    a degenerate level every operator is a scalar on -- and each singlet as a class of its own:
    ``P H P`` and ``P H Q H P`` alone see the symmetry."""
    plan = _plan(_hamiltonian())
    assert not hasattr(plan.classes[0], "spin")
    assert [c.span.shape[1] for c in plan.classes] == [3, 2, 1]
    assert all(c.status == "expand" for c in plan.classes)


@pytest.mark.mpi
def test_a_symmetry_the_framework_cannot_see_still_separates_the_classes():
    """With spin mixed, the orbital-parity sectors still split.

    Each orbital has its own bath and the interaction only moves electron pairs between orbitals,
    so the parity of each orbital's (impurity + bath) count is conserved: a discrete symmetry with no
    Lie generator, found by the symmetry-agnostic seed partition alone.
    """
    h_op = _hamiltonian(spin_mixing=0.05)
    plan = _plan(h_op)
    assert len(plan.expanded) >= 2
    assert _sector_energy(h_op) == pytest.approx(_exact_ground_energy(h_op), abs=1e-6)


@pytest.mark.mpi
def test_the_classes_do_not_depend_on_the_trial_shift():
    """A double-counting shift moves every seed determinant by the same constant."""
    reference = _plan(_hamiltonian())
    shifted = _plan(_hamiltonian(mu=0.37))
    assert [(c.span.shape[1], c.status) for c in reference.classes] == [
        (c.span.shape[1], c.status) for c in shifted.classes
    ]
    np.testing.assert_allclose(
        [c.energy for c in reference.classes], [c.energy + 2 * 0.37 for c in shifted.classes], atol=1e-9
    )


@pytest.mark.mpi
def test_the_energy_is_identical_on_every_rank():
    energy = _sector_energy(_hamiltonian())
    assert len(set(MPI.COMM_WORLD.allgather(energy))) == 1


def test_partner_rows_are_one_class_and_unrelated_blocks_are_not():
    """Two copies of one block (partners) merge; a third block with an accidentally shared eigenvalue does not."""
    rng = np.random.default_rng(3)
    block = rng.standard_normal((3, 3))
    block = block + block.T
    other = np.diag(np.linalg.eigvalsh(block)[[0, 1]] + [0.0, 0.5])  # shares the lowest eigenvalue exactly
    a = np.zeros((8, 8))
    a[:3, :3] = a[3:6, 3:6] = block
    a[6:, 6:] = other
    m = np.zeros((8, 8))
    coupling = rng.standard_normal((3, 3))
    m[:3, :3] = m[3:6, 3:6] = coupling @ coupling.T
    m[6:, 6:] = np.diag([0.3, 0.7])
    groups, _w, ok = _partition([a.astype(complex), m.astype(complex)])
    assert ok
    sizes = sorted(len(g) for g in groups)
    # The two partner blocks come out as one class of 6 or as two classes of 3, never mixed with the
    # third block, whose two states are not coupled and so may stay apart.
    assert sizes in ([1, 1, 6], [1, 1, 3, 3])


def _three_orbital_hole(onsite_c):
    """One hole in three orbitals: a and b degenerate in the crystal field, hybridized 0.2 and 1.5.

    Their seed classes share the ``P H P`` spectrum but are no symmetry's partners, and the hole wins
    in b, the strongly hybridized one.
    """
    n_imp, n_orb = 6, 12
    h = np.zeros((n_orb, n_orb), dtype=complex)
    for spin in range(2):
        for orbital, (energy, hopping) in enumerate(((0.0, 0.2), (0.0, 1.5), (onsite_c, 0.3))):
            i = orbital + 3 * spin
            h[i, i] = energy
            h[i + n_imp, i + n_imp] = -1.0
            h[i, i + n_imp] = h[i + n_imp, i] = hopping
    return tensors_to_operator(h), {0: [list(range(n_imp))]}, ({0: [list(range(n_imp, n_orb))]}, {0: []})


@pytest.mark.mpi
@pytest.mark.parametrize("onsite_c", [-0.05, 0.05])
def test_degenerate_but_differently_hybridized_classes_are_not_partners(onsite_c):
    h_op, imp, bath = _three_orbital_hole(onsite_c)
    result = solve_sector(
        h_op, imp, bath, {0: 5}, None, 0, False, 1000, manifold=Lowest(), comm=MPI.COMM_WORLD, verbose=False,
        slaterWeightMin=0,
    )  # fmt: skip
    assert float(np.min(np.real(result.es))) == pytest.approx(_exact_ground_energy(h_op, 12, 11), abs=1e-8)


@pytest.mark.mpi
def test_the_thermal_manifold_holds_every_copy_of_a_degenerate_ground_multiplet():
    """Partner classes are expanded for ``Within``: a partner left at seed quality falls outside the window."""
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    for spin in range(2):
        for orbital, (energy, hopping) in enumerate(((-1.0, 0.2), (1.0, 0.2))):
            i = orbital + 2 * spin
            h[i, i] = energy
            h[i + N_IMP, i + N_IMP] = -1.0
            h[i, i + N_IMP] = h[i + N_IMP, i] = hopping
    h_op = tensors_to_operator(h) + ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(kanamori_u4(2, 4.0, 0.8)))
    result = solve_sector(
        h_op, IMPURITY_ORBITALS, BATH_STATES, {0: 2}, None, 0, False, 1000, manifold=Within(1e-3),
        comm=MPI.COMM_WORLD, verbose=False, slaterWeightMin=0,
    )  # fmt: skip
    assert len(result.es) == 3  # the S = 1 triplet
    np.testing.assert_allclose(result.es, _exact_ground_energy(h_op), atol=1e-8)
