"""sharpen_manifolds: rotate a degenerate manifold to a basis with sharp S_z (and S^2), changing nothing else."""

from itertools import combinations
from types import SimpleNamespace

import numpy as np

from impurityModel.ed.gs_report import sharpen_manifolds
from impurityModel.ed.ManyBodyUtils import ManyBodyState, inner
from impurityModel.ed.observables import make_spin_operators

PAIRS = [(0, 1), (2, 3)]
LAYOUT = SimpleNamespace(labelled=True, pairs=PAIRS)


def _make_sd(occ):
    from impurityModel.test.symmetry.test_nonabelian_casimir import _sd

    return _sd(sorted(occ))


def _random_rotated_manifold(seed):
    dets = list(combinations(range(4), 2))
    basis_states = [ManyBodyState({_make_sd(d): 1.0}) for d in dets]
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    blk = ManyBodyState.from_states(basis_states).combine_columns(q)
    return blk


def _casimirs():
    ops = make_spin_operators(PAIRS)
    return (ops, ops, ops)


def test_rotated_spin_manifold_gets_sharp_sz_and_s2():
    blk = _random_rotated_manifold(3)
    es = np.zeros(6)
    rot, info = sharpen_manifolds(blk, es, LAYOUT, _casimirs(), 1e-6)
    assert info.rotated
    assert sorted(info.two_sz.tolist()) == [-2, 0, 0, 0, 0, 2]
    s2 = info.values["S2"]
    # S = 1 exactly where |2Sz| = 2, and the trace over the manifold (6 states: 3 triplet + 3 singlet) is conserved.
    np.testing.assert_allclose(s2[np.abs(info.two_sz) == 2], 2.0, atol=1e-10)
    np.testing.assert_allclose(np.sum(s2), 6.0, atol=1e-10)
    # The values describe the rotated block itself.
    sz_op = make_spin_operators(PAIRS)[2]
    out = sz_op.apply_block(rot, 0)
    m = np.array([[inner(rot.to_states()[i], out.to_states()[j]) for j in range(6)] for i in range(6)])
    np.testing.assert_allclose(np.real(np.diag(m)), info.values["Sz"], atol=1e-10)
    np.testing.assert_allclose(m - np.diag(np.diag(m)), 0.0, atol=1e-10)


def test_singlet_and_triplet_in_one_manifold_are_not_one_s():
    blk = _random_rotated_manifold(5)
    _rot, info = sharpen_manifolds(blk, np.zeros(6), LAYOUT, _casimirs(), 1e-6)
    assert set(np.round(info.values["S2"], 6)) == {0.0, 2.0}


def test_split_levels_are_left_alone():
    blk = _random_rotated_manifold(7)
    es = np.arange(6, dtype=float)
    rot, info = sharpen_manifolds(blk, es, LAYOUT, _casimirs(), 1e-6)
    assert not info.rotated
    assert rot is blk


def test_disabled_returns_the_block_untouched():
    blk = _random_rotated_manifold(9)
    rot, info = sharpen_manifolds(blk, np.zeros(6), LAYOUT, _casimirs(), 1e-6, enabled=False)
    assert rot is blk and not info.rotated and info.keys == ()
