"""Label symmetries (``label_symmetries``): a verified signed orbital permutation lets certification
expand one hard label per orbit.

The fixture is ``test_seed_sectors``' two-orbital model with the two orbitals made equivalent (same
level, same hybridization, Kanamori ``U``), at 7 electrons: the per-orbital occupation parities are
then conserved and odd in total, so the labels ``(0, 1)`` and ``(1, 0)`` both occur and the orbital
swap maps one onto the other. Every expected energy is an exact diagonalization of the whole sector.
"""

import itertools

import numpy as np
import pytest

from impurityModel.ed import atomic_physics
from impurityModel.ed import label_symmetries as ls
from impurityModel.ed.eigensolvers import Lowest, Within
from impurityModel.ed.groundstate import build_basis_and_solver, solve_sector
from impurityModel.ed.interaction_models import kanamori_u4
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp
from impurityModel.ed.sector_solver import certification_plan, certify_sector
from impurityModel.ed.spin_pairs import bath_spin_pairs, impurity_spin_pairs
from impurityModel.test.support.comms import COMMS
from impurityModel.test.symmetry.test_seed_sectors import BATH_STATES, IMPURITY_ORBITALS, N_IMP, N_ORB

N_ELECTRONS = 7  # d3 with the bath full: odd, so the two orbital parities differ
N0 = {0: 3}
EXPAND = dict(dense_cutoff=1000, slaterWeightMin=0, de2_min=0, e_pt2_tol=1e-12)


@pytest.fixture(autouse=True)
def _certifier_defaults(monkeypatch):
    for knob in ("GS_SECTOR_CERTIFY", "GS_CERTIFY_WINDOW", "GS_CERTIFY_WINDOW_FACTOR", "GS_CERTIFY_LABEL_SYMMETRY"):
        monkeypatch.delenv(knob, raising=False)


def _hamiltonian(bath_split=0.0):
    """Two equivalent orbitals; ``bath_split`` moves orbital 1's bath level and breaks the swap."""
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    for spin in range(2):
        for orbital in range(2):
            i = orbital + 2 * spin
            h[i, i] = -0.5
            h[i + N_IMP, i + N_IMP] = -1.0 + (bath_split if orbital == 1 else 0.0)
            h[i, i + N_IMP] = h[i + N_IMP, i] = 0.7
    u_op = ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(kanamori_u4(2, 4.0, 0.8)))
    return tensors_to_operator(h) + u_op


def _generators(h_op, restrictions=None):
    return ls.symmetry_generators(
        h_op,
        impurity_spin_pairs(IMPURITY_ORBITALS),
        bath_spin_pairs(BATH_STATES),
        N_ORB,
        restrictions=restrictions,
    )


def _swaps(generators):
    return [sp for sp in generators if not sp.flips_spin]


def test_the_orbital_swap_and_the_spin_flip_are_found_and_verified():
    generators = _generators(_hamiltonian())
    assert any(sp.flips_spin for sp in generators)
    (swap,) = _swaps(generators)
    # Orbital 0 <-> 1 in both spins, the bath sites carried along.
    assert swap.perm[0] == 1 and swap.perm[2] == 3 and swap.perm[4] == 5 and swap.perm[6] == 7
    assert ls.is_symmetry(_hamiltonian(), swap)


def test_a_bath_level_fitted_slightly_apart_breaks_the_swap():
    """Equal seed spectra are not enough: a bath level 1e-3 apart is invisible to the seed but makes
    the two labels different problems, and the full-operator check sees it."""
    assert not _swaps(_generators(_hamiltonian(bath_split=1e-3)))


def test_a_restriction_on_one_orbital_breaks_the_swap():
    restrictions = {frozenset({0, 2, 4, 6}): (3, 4)}  # orbital 0 and its bath only
    assert _swaps(_generators(_hamiltonian()))
    assert not _swaps(_generators(_hamiltonian(), restrictions=restrictions))


def test_the_swap_maps_the_two_odd_parity_labels_onto_each_other():
    from impurityModel.ed.symmetries import conserved_parities

    h_op = _hamiltonian()
    parities = conserved_parities(h_op)
    (swap,) = _swaps(_generators(h_op))
    mapping = ls.label_map(swap, parities)
    assert mapping is not None
    moved = {
        label: mapping(label)
        for label in [(1, p) for p in itertools.product((0, 1), repeat=len(parities))]
        if mapping(label) != label
    }
    assert moved and all(mapping(image) == label for label, image in moved.items())


def _seed(h_op, comm):
    basis, _solver = build_basis_and_solver(
        h_op, IMPURITY_ORBITALS, BATH_STATES, N0, None, 0, False, None, comm, False, None, None, 0,
        manifold=Lowest(),
    )  # fmt: skip
    return basis


def _certified(comm):
    h_op = _hamiltonian()
    basis = _seed(h_op, comm)
    plan = certification_plan(h_op, basis)
    es, _psis, _cb, _cs, verdicts = certify_sector(
        h_op, basis, plan, expand_kwargs=EXPAND, dense_cutoff=1000, slater_weight_min=0
    )
    e0 = float(np.min(np.real(es)))
    return (e0 if comm is None else comm.bcast(e0, root=0)), plan, verdicts


def _spectrum(h_op):
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


@pytest.mark.parametrize("comm", COMMS)
def test_mirrored_labels_are_not_expanded_and_reach_the_same_answer(comm, monkeypatch):
    exact = float(_spectrum(_hamiltonian())[0])
    monkeypatch.setenv("GS_CERTIFY_LABEL_SYMMETRY", "0")
    e_all, plan_all, verdicts_all = _certified(comm)
    monkeypatch.delenv("GS_CERTIFY_LABEL_SYMMETRY")
    e_dedup, plan, verdicts = _certified(comm)

    mirrors = [job for job in plan.jobs if job.image_of is not None]
    assert mirrors, "the fixture no longer has a label the swap mirrors"
    assert all(job.image_of is None for job in plan_all.jobs)
    # The same jobs, by label and class; the mirrors' verdicts are their sources' energies.
    assert {(j.two_sz, j.parity, j.seed_class) for j in plan.jobs} == {
        (j.two_sz, j.parity, j.seed_class) for j in plan_all.jobs
    }
    by_job = {(v.job.two_sz, v.job.parity, v.job.seed_class): v.energy for v in verdicts_all}
    for v in verdicts:
        assert v.energy == pytest.approx(by_job[(v.job.two_sz, v.job.parity, v.job.seed_class)], abs=1e-8)
    assert abs(e_dedup - exact) < 1e-8 and abs(e_all - exact) < 1e-8


@pytest.mark.parametrize("comm", COMMS)
def test_the_thermal_manifold_keeps_the_mirrored_labels_states(comm, monkeypatch):
    """The ground level is degenerate across the two mirrored labels; ``Within`` must return all of it
    -- the mirror label's rows come from mapping the representative's determinants."""
    exact = np.sort(np.real(_spectrum(_hamiltonian())))
    multiplicity = int(np.sum(exact - exact[0] < 1e-8))
    assert multiplicity > 1, "the fixture's ground level is not degenerate"

    def thermal():
        result = solve_sector(
            _hamiltonian(), IMPURITY_ORBITALS, BATH_STATES, N0, None, 0, False, 1000, manifold=Within(1e-6),
            comm=comm, verbose=False, slaterWeightMin=0, de2_min=0, e_pt2_tol=1e-12,
        )  # fmt: skip
        return np.sort(np.real(result.es))

    got = thermal()
    assert len(got) == multiplicity
    np.testing.assert_allclose(got, exact[:multiplicity], atol=1e-8)


def test_a_mirror_image_follows_its_source_in_the_thermal_union():
    """The union adds the kept classes in order and stops at the cap, so an image must sit where the
    mirror's own degenerate class would have sorted -- right behind its source -- or a binding cap
    would drop the degenerate partner before a higher class."""
    from impurityModel.ed.sector_solver import _thermal_union_order

    # Winner 2 (mirrored by 7), then in-window classes 0 and 3 by energy; 3 is mirrored by 6.
    order = _thermal_union_order(winner=2, inside=[2, 0, 3], mirrored={6: 3, 7: 2})
    assert order == [(2, None), (2, 7), (0, None), (3, None), (3, 6)]


def test_the_thermal_union_carries_the_winners_image():
    from impurityModel.ed.sector_solver import _image_keys

    h_op = _hamiltonian()
    basis = _seed(h_op, None)
    plan = certification_plan(h_op, basis)
    *_rest, kept = certify_sector(
        h_op, basis, plan, expand_kwargs=EXPAND, dense_cutoff=1000, slater_weight_min=0, within=1e-6
    )
    images = [
        set(_image_keys(kept[0], plan.images[k], basis.num_spin_orbitals, basis.n_bytes).keys()) for k in plan.images
    ]
    assert len(kept) == 2 and set(kept[1].keys()) in images and set(kept[1].keys()) != set(kept[0].keys())
