"""``spin_pairs.spin_layout``: the (dn, up) pairing of impurity and bath, and what ``H`` verifies of it.

The model is a five-orbital d shell (dn-then-up: 0-4 down, 5-9 up) with a Kanamori interaction, pair
hopping included, and one bath chain of ``DEPTH`` sites per impurity spin-orbital. ``GROUPS`` regroups
the impurity per spin, as ``prepare_solver_basis`` does when a spin-dependent shift splits the one-body
levels: the dn-then-up *index* convention applied inside each group then pairs same-spin orbitals.
"""

import numpy as np
import pytest

from impurityModel.ed import atomic_physics
from impurityModel.ed.interaction_models import kanamori_u4
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.spin_pairs import spin_layout

N_IMP = 10
DEPTH = 2
N_ORB = N_IMP * (1 + DEPTH)
#: eg-dn, t2g-dn, eg-up, t2g-up, the grouping of the FCC Ni archive.
GROUPS = {0: [[0, 1]], 1: [[2, 3, 4]], 2: [[5, 6]], 3: [[7, 8, 9]]}
WHOLE = {0: [list(range(N_IMP))]}
PAIRS = tuple((k, k + 5) for k in range(5))


def _site(i, depth):
    """Bath orbital ``depth`` (0-based) of impurity spin-orbital ``i``."""
    return N_IMP + depth * N_IMP + i


def _bath_states(groups):
    """Valence baths: one dn-then-up block for the whole shell, else blocked like ``groups``."""
    if len(groups) == 1:
        return ({0: [[_site(i, d) for spin in (0, 5) for d in range(DEPTH) for i in range(spin, spin + 5)]]}, {0: []})
    blocks = [[_site(i, d) for d in range(DEPTH) for i in block] for blocks_ in groups.values() for block in blocks_]
    return ({0: blocks}, {0: []})


def _model(bath_shift=(0.0, 0.0), level_shift=(0.0, 0.0), density_density=False, two_body=True):
    """``bath_shift`` / ``level_shift``: (dn, up) energy added to every bath site / impurity level."""
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    levels = np.array([-0.3, -0.3, 0.2, 0.2, 0.2] * 2)
    for i in range(N_IMP):
        spin = int(i >= 5)
        h[i, i] = levels[i] + level_shift[spin]
        for d in range(DEPTH):
            s = _site(i, d)
            h[s, s] = -1.0 - 0.7 * d + 0.1 * (i % 5) * 0 + bath_shift[spin]
            prev = i if d == 0 else _site(i, d - 1)
            h[prev, s] = h[s, prev] = 0.6 / (d + 1)
    op = tensors_to_operator(h)
    if two_body:
        u4 = kanamori_u4(5, 4.0, 0.8, J_pair=None if not density_density else 0.0)
        op = op + ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(u4))
    return op


def _layout(op, groups=GROUPS, rot=None):
    rot = np.eye(N_IMP) if rot is None else rot
    return spin_layout(op, groups, _bath_states(groups), rot, N_ORB)


def test_a_grouped_polarized_shell_is_paired_by_rotation_and_chain_position():
    layout = _layout(_model(bath_shift=(0.0, 0.05), level_shift=(0.0, 5e-5)))
    assert layout.imp_pairs == PAIRS
    assert layout.imp_source == "rotation"
    assert "index: impurity orbitals 4, 9 unpaired" in layout.rejected
    assert layout.bath_source == "chain_position"
    assert sorted(layout.bath_pairs) == sorted((_site(k, d), _site(k + 5, d)) for k in range(5) for d in range(DEPTH))
    assert layout.grade == "collinear"
    assert layout.impurity_split == pytest.approx(5e-5)


def test_an_unpolarized_shell_is_su2_with_index_sources():
    layout = _layout(_model(), groups=WHOLE)
    assert layout.grade == "su2"
    assert layout.imp_pairs == PAIRS
    assert (layout.imp_source, layout.bath_source) == ("index", "index")
    assert layout.impurity_split == 0.0
    assert layout.provenance_note() is None


def test_an_unpolarized_grouped_shell_is_still_su2_but_not_by_index():
    layout = _layout(_model())
    assert layout.grade == "su2"
    assert layout.imp_source == "rotation"
    assert layout.bath_source in ("matched_energies", "chain_position")


def test_a_polarized_bath_alone_is_collinear():
    layout = _layout(_model(bath_shift=(0.0, 0.2)), groups=WHOLE)
    assert layout.grade == "collinear"
    assert layout.impurity_split == 0.0


def test_orbital_dependent_split_is_not_confused_with_the_pairing():
    op = _model(bath_shift=(0.0, 0.2))
    h_extra = np.zeros((N_ORB, N_ORB), dtype=complex)
    for k, e in enumerate([1e-3, 2e-3, 3e-3, 4e-3, 5e-3]):
        h_extra[k + 5, k + 5] = e
    layout = _layout(op + tensors_to_operator(h_extra), groups=WHOLE)
    assert layout.grade == "collinear"
    assert layout.impurity_split == pytest.approx(5e-3)


def test_a_spin_dependent_rotation_of_the_levels_is_only_sz_conserving():
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    h[5, 7] = h[7, 5] = 0.05  # mixes eg with t2g in the up block only
    layout = _layout(_model() + tensors_to_operator(h), groups=WHOLE)
    assert layout.grade == "sz_only"


def test_spin_orbit_coupling_leaves_no_pairing():
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    h[0, 5] = h[5, 0] = 0.05  # dn <-> up
    layout = _layout(_model() + tensors_to_operator(h), groups=WHOLE)
    assert layout.grade == "none"
    assert not layout.labelled
    assert layout.imp_pairs == () and layout.rejected
    assert "spin layout: none" in layout.provenance_note()


def test_the_rotation_decides_when_a_density_density_u_cannot():
    """Orbitals 5 and 6 are equivalent copies and ``U`` is orbital independent, so the index pairs
    (0, 5), (1, 6) and the swapped (0, 6), (1, 5) both pass every gate on ``H``. Only the rotated
    ``S_+`` says which one is the spin."""
    op = _model(density_density=True, two_body=False)
    u4 = kanamori_u4(5, 4.0, 0.0, U_prime=4.0, J_pair=0.0)
    op = op + ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(u4))
    rot = np.eye(N_IMP)
    rot[[5, 6]] = rot[[6, 5]]
    layout = _layout(op, groups=WHOLE, rot=rot)
    assert layout.imp_pairs == ((0, 6), (1, 5), (2, 7), (3, 8), (4, 9))
    assert layout.imp_source == "rotation"
    assert "index: contradicts the rotation pairing" in layout.rejected


def _unequal_chains(op, link=0.0):
    """``op`` plus one extra bath site hanging on the dn chain of impurity orbital 0 only, so the dn chain of
    orbital 0 is one site longer than the up chain of its partner 5. ``link`` couples the extra site to the up
    chain as well (a spin-mixing bath)."""
    extra = np.zeros((N_ORB + 1, N_ORB + 1), dtype=complex)
    s = N_ORB
    extra[s, s] = -2.0
    extra[_site(0, DEPTH - 1), s] = extra[s, _site(0, DEPTH - 1)] = 0.3
    extra[_site(5, DEPTH - 1), s] = extra[s, _site(5, DEPTH - 1)] = link
    big = ManyBodyOperator(dict(op.items())) + tensors_to_operator(extra)
    return big, s


def test_unequal_chains_are_labelled_and_partially_paired_not_given_up():
    """A spin-polarised fit can leave dn and up chains of different length: no pairing of the bath exists,
    but H conserves S_z exactly. The layout keeps the S_z labels and pairs what can be paired."""
    big, s = _unequal_chains(_model(bath_shift=(0.0, 0.2)))
    layout = spin_layout(big, WHOLE, ({0: [list(range(N_IMP, N_ORB + 1))]}, {0: []}), np.eye(N_IMP), N_ORB + 1)
    assert layout.labelled and not layout.paired and not layout.bath_paired
    assert layout.grade == "collinear"
    assert layout.imp_pairs == PAIRS
    assert layout.bath_source == "chain_position_partial"
    assert layout.unpaired == (s,)
    weights = layout.sz_weights
    assert weights[s] == -1  # reached from the dn impurity orbital 0
    assert all(weights[dn] == -1 and weights[up] == 1 for dn, up in layout.pairs)
    assert len(weights) == N_ORB + 1
    assert any("differ in length" in r for r in layout.rejected)
    from impurityModel.ed.symmetries import PARITY_TERM_TOL, conserves_sz

    assert conserves_sz(big, weights, PARITY_TERM_TOL)
    assert not conserves_sz(big, layout.pairs, PARITY_TERM_TOL)  # the pairs alone leave s without a spin
    note = layout.provenance_note()
    assert "S_z exact" in note and "1 surplus site(s) unpaired" in note
    # S_z over the labels counts the surplus site; over the pairs it would not.
    sz = layout.sz_operator()
    assert ((s, "c"), (s, "a")) in dict(sz.items())


def test_a_bath_site_reached_from_both_spins_still_gives_none():
    big, _ = _unequal_chains(_model(bath_shift=(0.0, 0.2)), link=0.2)
    layout = spin_layout(big, WHOLE, ({0: [list(range(N_IMP, N_ORB + 1))]}, {0: []}), np.eye(N_IMP), N_ORB + 1)
    assert layout.grade == "none" and not layout.labelled
    assert any("reached by hopping from both" in r for r in layout.rejected)


def test_an_empty_bath_is_a_valid_layout():
    op = _model()
    layout = spin_layout(op, WHOLE, ({}, {}), np.eye(N_IMP), N_ORB)
    # The chains still couple to the impurity, so unpaired bath orbitals break S_z: the layout is
    # judged on the whole operator, not on the impurity alone.
    assert layout.grade in ("none", "su2")


def _basis(groups):
    from types import SimpleNamespace

    return SimpleNamespace(
        impurity_orbitals=groups,
        bath_states=_bath_states(groups),
        num_spin_orbitals=N_ORB,
        is_distributed=False,
        comm=None,
    )


def test_the_susceptibility_operators_follow_the_layout():
    from impurityModel.ed.susceptibility import build_susceptibility_operators

    op = _model(bath_shift=(0.0, 0.05), level_shift=(0.0, 5e-5))
    ops, skipped, notes = build_susceptibility_operators(op, _basis(GROUPS), np.eye(N_IMP, dtype=complex))
    assert "transverse" in ops and "transverse" not in skipped
    assert "spin-flip response" in notes["transverse"]  # collinear: physics note, not a warning

    soc = np.zeros((N_ORB, N_ORB), dtype=complex)
    soc[0, 5] = soc[5, 0] = 0.05
    ops, skipped, notes = build_susceptibility_operators(
        _model() + tensors_to_operator(soc), _basis(WHOLE), np.eye(N_IMP, dtype=complex)
    )
    assert "transverse" in skipped and "transverse" not in ops


def test_a_sector_with_unequal_chains_is_solved_exactly_under_the_sz_labels():
    """End to end: the certifier labels by 2 S_z from the labels and builds every class window from them. A
    window built from the partial pairs leaves out the surplus site and rejects every determinant (E = 0)."""
    import itertools

    from mpi4py import MPI

    from impurityModel.ed.eigensolvers import Lowest
    from impurityModel.ed.groundstate import solve_sector
    from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant, applyOp
    from impurityModel.ed.sector_certify import certification_plan
    from impurityModel.ed.spin_pairs import spin_layout as resolve
    from impurityModel.test.symmetry import test_seed_sectors as small

    n_orb = small.N_ORB + 1
    surplus = small.N_ORB
    h = np.zeros((n_orb, n_orb), dtype=complex)
    for spin in range(2):
        for orbital, (energy, hopping) in enumerate(((-1.0, 0.2), (1.0, 1.5))):
            i = orbital + 2 * spin
            h[i, i] = energy
            h[i + small.N_IMP, i + small.N_IMP] = -1.0 + 0.2 * spin  # spin-polarised bath: no matched energies
            h[i, i + small.N_IMP] = h[i + small.N_IMP, i] = hopping
    h[surplus, surplus] = -1.5
    h[small.N_IMP, surplus] = h[surplus, small.N_IMP] = 0.4  # a second site on the dn chain of orbital 0 only
    u_op = ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(kanamori_u4(2, 4.0, 0.8)))
    h_op = tensors_to_operator(h) + u_op
    bath = ({0: [list(range(small.N_IMP, n_orb))]}, {0: []})
    n_electrons = small.N_ELECTRONS + 1

    layout = resolve(h_op, small.IMPURITY_ORBITALS, bath, None, n_orb)
    assert layout.labelled and not layout.bath_paired and layout.unpaired == (surplus,)

    dets = []
    for occupied in itertools.combinations(range(n_orb), n_electrons):
        chunk = 0
        for orb in occupied:
            chunk |= 1 << (63 - orb)
        dets.append(SlaterDeterminant((chunk,)))
    index = {det: i for i, det in enumerate(dets)}
    h_op.set_restrictions(None)
    h_op.set_weighted_restrictions(None)
    dense = np.zeros((len(dets), len(dets)), dtype=complex)
    for j, det in enumerate(dets):
        for key, row in applyOp(h_op, ManyBodyState({det: 1.0}), cutoff=0).items():
            dense[index[key], j] += row[0]
    exact = float(np.linalg.eigvalsh(dense)[0])

    result = solve_sector(
        h_op,
        small.IMPURITY_ORBITALS,
        bath,
        {0: 2},
        tau=0,
        chain_restrict=False,
        dense_cutoff=1000,
        manifold=Lowest(),
        comm=MPI.COMM_WORLD,
        verbose=False,
        slaterWeightMin=0,
        de2_min=0,
        e_pt2_tol=1e-12,
        spin_layout=layout,
    )
    assert float(np.min(np.real(result.es))) == pytest.approx(exact, abs=1e-8)
