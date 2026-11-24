"""Driver glue for spectra: :func:`spectra._pes_ips_equivalence_groups` labels degenerate PES/IPS operators (B2a)."""

from collections import OrderedDict

from impurityModel.ed import spectra
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import (
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)


def test_pes_ips_groups_collapse_nio_d_shell_to_four_classes():
    """On the rotated NiO 3d shell the 10 PES/IPS operators fall into 4 symmetry classes."""
    from impurityModel.test.support._nio_workload import build_selfenergy_inputs

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


def test_the_orbitals_of_one_coupled_block_are_not_labelled_equivalent():
    """Only the k-th orbital of each identical block is equivalent; the orbitals of one block are distinct.

    A block of several orbitals used to give all of them one label, so ``calc_spectra`` computed the first and
    copied it into every column of the PES spectrum.
    """
    from types import SimpleNamespace

    nBaths = OrderedDict({1: 0})
    one_block = SimpleNamespace(blocks=[[0, 1, 2, 3, 4, 5]], identical_blocks=[[0]])
    assert len(set(spectra._pes_ips_equivalence_groups(nBaths, 1, one_block))) == 6
    # Two identical blocks of three: orbital k of the first block pairs with orbital k of the second.
    two_blocks = SimpleNamespace(blocks=[[0, 1, 2], [3, 4, 5]], identical_blocks=[[0, 1]])
    groups = spectra._pes_ips_equivalence_groups(nBaths, 1, two_blocks)
    assert groups[:3] == groups[3:] and len(set(groups)) == 3
