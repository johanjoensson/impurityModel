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


def test_overlapping_chains_are_paired_only_if_consistent_and_unequal_chains_give_none():
    op = _model(bath_shift=(0.0, 0.2))
    # An extra bath site hanging on impurity orbital 0 only: its up partner chain is shorter.
    extra = np.zeros((N_ORB + 1, N_ORB + 1), dtype=complex)
    s = N_ORB
    extra[s, s] = -2.0
    extra[_site(0, DEPTH - 1), s] = extra[s, _site(0, DEPTH - 1)] = 0.3
    big = ManyBodyOperator(dict(op.items())) + tensors_to_operator(extra)
    layout = spin_layout(big, WHOLE, ({0: [list(range(N_IMP, N_ORB + 1))]}, {0: []}), np.eye(N_IMP), N_ORB + 1)
    assert layout.grade == "none"
    assert any("differ in length" in r for r in layout.rejected)


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
