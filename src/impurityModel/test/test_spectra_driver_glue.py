"""Driver-glue helpers for rotated spectra: operator rotation + PES/IPS dedup grouping.

These cover the pieces get_spectra.main / simulate_spectra use to run spectra in the
symmetry-adapted basis: :func:`spectra._rotate_op` (rotate a one-body transition operator)
and :func:`spectra._pes_ips_equivalence_groups` (label degenerate PES/IPS operators for B2a).
"""

from collections import OrderedDict

import numpy as np

from impurityModel.ed import spectra
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import (
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)


def test_rotate_op_identity_is_noop():
    op = {((2, "c"), (0, "a")): 0.7, ((3, "c"), (1, "a")): -0.4}
    # Accepts either a dict (what the transition_operators builders return) or an operator.
    for arg in (op, ManyBodyOperator(op)):
        rotated = spectra._rotate_op(arg, np.eye(4, dtype=complex))
        assert isinstance(rotated, ManyBodyOperator)
        assert set(rotated.keys()) == set(op.keys())
        for k, value in op.items():
            assert np.isclose(rotated[k], value)


def test_rotate_op_swaps_under_permutation():
    # Swap orbitals 0 and 1: c_0 <-> c_1. Rotating c^dagger_0 c_0 gives c^dagger_1 c_1.
    W = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    rotated = spectra._rotate_op({((0, "c"), (0, "a")): 1.0}, W)
    assert set(rotated.keys()) == {((1, "c"), (1, "a"))}
    assert np.isclose(rotated[((1, "c"), (1, "a"))], 1.0)


def test_pes_ips_groups_collapse_nio_d_shell_to_four_classes():
    """On the rotated NiO 3d shell the 10 PES/IPS operators fall into 4 symmetry classes."""
    from impurityModel.test._nio_workload import build_selfenergy_inputs

    kw = build_selfenergy_inputs(nBaths=10, verbose=False)
    op = ManyBodyOperator(dict(kw["h0"]))
    d_indices = list(range(10))
    W, _ = impurity_symmetry_rotation(op, d_indices, n_orb=30)
    op_rot = rotate_hamiltonian(op, W, tol=1e-8)
    bs = impurity_block_structure(op_rot, d_indices, n_orb=30)

    nBaths = OrderedDict({2: 0})
    groups = spectra._pes_ips_equivalence_groups(nBaths, 2, bs)

    assert len(groups) == 10
    # One label per inequivalent class (t2g up/dn, eg up/dn -> 4).
    assert len(set(groups)) == len(bs.inequivalent_blocks) == 4
    # Class sizes match the block-structure degeneracies (two triplets + two doublets).
    counts = sorted([groups.count(g) for g in set(groups)])
    assert counts == [2, 2, 3, 3]
