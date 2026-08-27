"""A 1s -> 2p dipole operator, against the analytic answer.

Gate 2 for the shell generalisation. The L2,3 gate
(``test_shell_generalisation.test_l23_hamiltonian_is_bit_identical_to_the_2p3d_assembly``)
only proves the new code agrees with the old code on the one edge the old code could do; it
cannot catch an error that is uniform across edges. This file checks a *different* edge
against something the codebase did not produce: the closed-form dipole matrix elements

.. math:: \\langle 1\\,m | \\hat T_\\varepsilon | 0\\,0 \\rangle
          = \\sum_q \\varepsilon_q\\, c^1(1, m; 0, 0),

where :math:`c^1(1, m; 0, 0) = 1/\\sqrt 3` for every :math:`m`, from

.. math:: c^k(l, m; l', m') = (-1)^m \\sqrt{(2l+1)(2l'+1)}
          \\begin{pmatrix} l & k & l' \\\\ 0 & 0 & 0 \\end{pmatrix}
          \\begin{pmatrix} l & k & l' \\\\ -m & m-m' & m' \\end{pmatrix},

with :math:`\\begin{pmatrix} 1 & 1 & 0 \\\\ 0 & 0 & 0 \\end{pmatrix} = -1/\\sqrt 3`.

``l_core = 0`` is the case a length-derived core angular momentum gets silently wrong: an
omitted core array used to be replaced by a p-shaped zero tuple, which asserts ``l_core = 1``.
"""

from collections import OrderedDict
from math import sqrt

import numpy as np
import pytest

from impurityModel.ed.operator_algebra import c2i
from impurityModel.ed.transition_operators import dipole_operator, dipole_operators

#: A bare 1s + 2p atom: 2 + 6 spin-orbitals, no bath.
NBATHS = OrderedDict({0: 0, 1: 0})

#: The single reduced Gaunt coefficient of a 1s -> 2p dipole, c^1(l=1, m; l'=0, m'=0).
#: Independent of m, and positive: sqrt(3) * 3j(1,1,0;0,0,0) * 3j(1,1,0;-m,m,0) = 1/sqrt(3).
C1 = 1 / sqrt(3)


def _index(l, s, m):
    return c2i(NBATHS, (l, s, m))


@pytest.mark.parametrize(
    "polarization, q",
    [
        ([0.0, 0.0, 1.0], 0),  # linear z  -> only m = 0
        ([1.0, 0.0, 0.0], None),  # linear x  -> m = -1 and +1
    ],
)
def test_k_edge_dipole_matches_the_closed_form(polarization, q):
    """Every amplitude, not just the pattern of non-zeros."""
    n = polarization
    n_spherical = {
        -1: (n[0] + 1j * n[1]) / sqrt(2),
        0: n[2],
        1: (-n[0] + 1j * n[1]) / sqrt(2),
    }

    t_op = dipole_operator(NBATHS, n, l_core=0, l_valence=1)

    expected = {}
    for s in range(2):
        for m in (-1, 0, 1):
            amplitude = n_spherical[m] * C1
            if amplitude != 0:
                expected[((_index(1, s, m), "c"), (_index(0, s, 0), "a"))] = amplitude

    assert set(t_op) == set(expected)
    for process, amplitude in expected.items():
        assert t_op[process] == pytest.approx(amplitude, abs=1e-12)

    if q is not None:
        # z polarization drives m = 0 alone -- the pi transition.
        assert {process[0][0] for process in t_op} == {_index(1, s, 0) for s in range(2)}


def test_k_edge_dipole_is_spin_diagonal_and_raises_l_by_one():
    """The two structural facts a wrongly-identified core shell would break."""
    t_op = dipole_operator(NBATHS, [0.3, -0.7, 0.5], l_core=0, l_valence=1)
    assert t_op
    for (i, create), (j, annihilate) in t_op:
        assert create == "c" and annihilate == "a"
        # Creation lands in the 2p block (indices 2..7), annihilation in the 1s block (0..1).
        assert 2 <= i < 8 and 0 <= j < 2
        # Spin is conserved: the 1s block is (s=0, s=1) and the 2p block three m's per spin.
        assert (i - 2) // 3 == j


def test_the_cartesian_components_sum_to_the_total_dipole_strength():
    """Sum rule: sum_alpha |T_alpha|^2 over all matrix elements is 2 * 3 * |C1|^2 * (2 spins).

    Each of the three 2p orbitals is reachable from 1s with weight |C1|^2 once the three
    Cartesian polarizations are summed, for each of the two spins.
    """
    components = dipole_operators(NBATHS, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0, 1)
    total = sum(abs(value) ** 2 for op in components for value in op.values())
    assert total == pytest.approx(2 * 3 * C1**2, rel=1e-12)


def test_a_forbidden_pair_is_refused_rather_than_returned_empty():
    """1s -> 3d is zero by selection rule; a silently empty operator would look like a bug."""
    with pytest.raises(ValueError, match="Gaunt selection rule"):
        dipole_operator(OrderedDict({0: 0, 2: 0}), [0, 0, 1], l_core=0, l_valence=2)


def test_the_l23_dipole_is_unchanged_by_naming_the_shells():
    """The generalisation must not move the edge it was built from."""
    nBaths = OrderedDict({1: 0, 2: 10})
    t_op = dipole_operator(nBaths, [0, 0, 1], l_core=1, l_valence=2)
    # 2p -> 3d, z polarization: m_core = m_valence, so 3 m-pairs x 2 spins.
    assert len(t_op) == 6
    for (i, _), (j, _) in t_op:
        assert 6 <= i < 16 and 0 <= j < 6
    # Spot-check one amplitude against the Gaunt coefficient it is defined by.
    from impurityModel.ed.atomic_physics import gauntC

    key = ((c2i(nBaths, (2, 0, 0)), "c"), (c2i(nBaths, (1, 0, 0)), "a"))
    assert t_op[key] == pytest.approx(gauntC(k=1, l=2, m=0, lp=1, mp=0), abs=1e-12)


def test_reversing_the_roles_reverses_the_transition():
    """The role arguments, not the dict order, decide which way the operator runs."""
    forward = dipole_operator(NBATHS, [0, 0, 1], l_core=0, l_valence=1)
    backward = dipole_operator(NBATHS, [0, 0, 1], l_core=1, l_valence=0)
    assert {(process[1][0], process[0][0]) for process in forward} == {
        (process[0][0], process[1][0]) for process in backward
    }
    assert not np.isclose(len(forward), 0)
