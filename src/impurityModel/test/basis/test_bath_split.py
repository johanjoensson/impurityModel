"""``prepare_solver_basis(bath_split=...)`` uses the caller's valence/conduction split."""

import numpy as np
import pytest

from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.solver_basis import prepare_solver_basis


def _build(bath_split=None, onsite=(-0.5, -0.5, -0.1, 0.4)):
    """Two impurity orbitals (0, 1), each hopping to a chain: 2 -> 3 (orbital 0) and 4 -> 5 (orbital 1)."""
    h = np.zeros((6, 6), dtype=complex)
    h[0, 0] = h[1, 1] = 0.0
    for o, e in zip((2, 3, 4, 5), onsite):
        h[o, o] = e
    for a, b in ((0, 2), (2, 3), (1, 4), (4, 5)):
        h[a, b] = h[b, a] = 0.3
    return prepare_solver_basis(
        tensors_to_operator(h).to_dict(),
        {},
        {},
        {0: [0, 1]},
        {0: 1},
        rot_to_spherical=np.eye(2, dtype=complex),
        verbosity=0,
        bath_split=bath_split,
    )


def _flat(parts):
    return sorted(o for blocks in parts.values() for block in blocks for o in block)


def test_none_is_the_sign_rule():
    sb = _build()
    assert _flat(sb.bath_states[0]) == [2, 3, 4]
    assert _flat(sb.bath_states[1]) == [5]


@pytest.mark.parametrize("shape", ["flat", "nested"])
def test_a_negative_onsite_site_declared_conduction_stays_conduction(shape):
    val, con = [2, 4], [3, 5]
    split = ({0: val}, {0: con}) if shape == "flat" else ({0: [[2], [4]]}, {0: [[3], [5]]})
    sb = _build(split)
    assert _flat(sb.bath_states[0]) == val
    assert _flat(sb.bath_states[1]) == con


@pytest.mark.parametrize(
    "split",
    [
        ({0: [2, 3]}, {0: [3, 4, 5]}),  # overlap
        ({0: [2, 3]}, {0: [4]}),  # missing 5
        ({0: [2, 3, 0]}, {0: [4, 5]}),  # impurity orbital
        ({0: [2, 3]}, {0: [4, 5, 9]}),  # out of range
    ],
)
def test_an_invalid_split_raises(split):
    with pytest.raises(ValueError, match="bath_split"):
        _build(split)
