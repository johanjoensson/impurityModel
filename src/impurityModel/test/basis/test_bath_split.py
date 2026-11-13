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


def test_none_derives_the_split():
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


def test_every_driver_hands_the_model_split_to_prepare_solver_basis():
    """DC, self-energy and susceptibility must classify the bath the same way: all pass ``model.bath_states``."""
    import ast
    import pathlib

    import impurityModel.ed as ed_pkg

    callers = ("selfenergy.py", "dc_occupation.py", "dc_sector_context.py", "susceptibility.py")
    root = pathlib.Path(ed_pkg.__file__).parent
    for name in callers:
        calls = [
            n
            for n in ast.walk(ast.parse((root / name).read_text()))
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "prepare_solver_basis"
        ]
        assert calls, name
        for call in calls:
            kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
            assert kw.get("bath_split") == "model.bath_states", name


def _chain_end(mix):
    """Bath sites 1, 2 under impurity orbital 0: levels at -1.0 and +0.1, rotated by ``mix`` into a chain."""
    c, s = np.cos(mix), np.sin(mix)
    rot = np.array([[c, -s], [s, c]])
    h = np.zeros((3, 3))
    h[1:, 1:] = rot @ np.diag([-1.0, 0.1]) @ rot.T
    h[0, 1] = h[1, 0] = 0.3
    return h


def test_a_conduction_chain_end_with_negative_onsite_energy_is_derived_conduction():
    """The NiO linked-chain shape: 12% of a deep level makes the on-site energy negative, the site is still empty."""
    from impurityModel.ed.symmetries import classify_bath_occupation

    h = _chain_end(np.arcsin(np.sqrt(0.12)))
    assert h[2, 2] < 0, "premise: the sign of the on-site energy would fill the site"
    messages = []
    assert classify_bath_occupation(None, [0], h0_matrix=h, report=messages.append) == ([1], [2])
    assert messages == []


def test_a_diagonal_bath_block_gives_the_sign_of_the_onsite_energy():
    from impurityModel.ed.symmetries import classify_bath_occupation

    rng = np.random.default_rng(3)
    eps = rng.uniform(-1, 1, 9)
    eps[4] = 0.0
    h = np.diag(np.concatenate([[0.0], eps])).astype(complex)
    h[0, 1:] = h[1:, 0] = 0.2
    valence, conduction = classify_bath_occupation(None, [0], h0_matrix=h)
    assert valence == [o + 1 for o in range(9) if eps[o] < 0]
    assert conduction == [o + 1 for o in range(9) if eps[o] >= 0]


def test_equally_mixed_sites_stay_together_and_the_count_mismatch_is_reported():
    """Two sites sharing one filled level hold half an electron each: both are filled, one electron too many."""
    from impurityModel.ed.symmetries import classify_bath_occupation

    messages = []
    valence, conduction = classify_bath_occupation(None, [0], h0_matrix=_chain_end(np.pi / 4), report=messages.append)
    assert (valence, conduction) == ([1, 2], [])
    assert len(messages) == 2
    assert "off by 1" in messages[0]
    assert "poor reference" in messages[1]
