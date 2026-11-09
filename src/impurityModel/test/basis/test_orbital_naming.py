"""OrbitalNaming: one validated map from spin-orbital index to role, group, spin, block."""

from types import SimpleNamespace

import numpy as np
import pytest

from impurityModel.ed.block_structure import BlockStructure, blocks_as_positions
from impurityModel.ed.gs_report_model import OrbitalNaming


def _bs(blocks):
    n = len(blocks)
    none = [[] for _ in range(n)]
    return BlockStructure(
        blocks=blocks,
        identical_blocks=[[i] for i in range(n)],
        transposed_blocks=none,
        particle_hole_blocks=none,
        particle_hole_transposed_blocks=none,
        inequivalent_blocks=list(range(n)),
    )


def _layout(pairs):
    return SimpleNamespace(labelled=True, pairs=tuple(pairs))


def test_noncontiguous_impurity_with_interleaved_bath():
    imp = {0: [[3, 8]]}
    bath = ({0: [[5, 9]]}, {0: [[1, 6]]})
    h1 = np.zeros((10, 10))
    h1[3, 5] = h1[5, 3] = 0.4
    h1[8, 9] = h1[9, 8] = 0.4
    h1[5, 5] = -1.0
    nm = OrbitalNaming.build(imp, bath, _bs([[0], [1]]), layout=_layout([(3, 8), (5, 9)]), h1=h1)
    assert nm.indices("impurity") == [3, 8]
    assert nm.indices("valence") == [5, 9] and nm.indices("conduction") == [1, 6]
    assert nm[3].spin == "dn" and nm[8].spin == "up" and nm[3].partner == 8
    assert nm[5].coupled_to == (3,) and nm[9].coupled_to == (8,)
    assert nm[5].one_body_level == -1.0 and nm[1].spin is None
    assert nm[3].block == 0 and nm[8].block == 1


def test_up_first_layout_names_spins_from_the_pairs_not_the_index_order():
    imp = {0: [[0, 1]]}
    nm = OrbitalNaming.build(imp, ({}, {}), _bs([[0, 1]]), layout=_layout([(1, 0)]))
    assert nm[1].spin == "dn" and nm[0].spin == "up"


def test_global_blocks_are_translated_to_positions():
    imp = {0: [[10, 11, 12]]}
    nm = OrbitalNaming.build(imp, ({}, {}), _bs([[10], [11, 12]]))
    assert nm.block_structure.blocks == [[0], [1, 2]]
    assert nm[12].block == 1
    assert blocks_as_positions(nm.block_structure, [10, 11, 12]) is nm.block_structure


def test_group_split_by_the_block_structure_is_labelled():
    imp = {1: [[0, 1, 2, 3]]}
    nm = OrbitalNaming.build(imp, ({}, {}), _bs([[0, 1], [2, 3]]))
    assert nm[0].group_label == "group 1.a" and nm[3].group_label == "group 1.b"


@pytest.mark.parametrize(
    "imp, bath, blocks, kw, match",
    [
        ({0: [[0, 1]]}, ({0: [[1]]}, {}), [[0, 1]], {}, "both"),  # an index with two roles
        (
            {0: [[0, 1]]},
            ({}, {}),
            [[0, 5]],
            {},
            "neither the positions",
        ),  # blocks that are neither positions nor indices
        ({0: [[0, 1]]}, ({0: [[2], [3]]}, {}), [[0, 1]], {}, "bath blocks"),  # bath blocks do not match impurity blocks
        ({0: [[0, 1]]}, ({}, {}), [[0, 1]], {"shell_ls": [1]}, "tile"),  # 2(2l+1) = 6 != 2
        ({0: [[0, 1]]}, ({}, {}), [[0, 1]], {"rot_to_spherical": np.eye(3)}, "rows"),
        (
            {0: [[0, 1]]},
            ({}, {}),
            [[0, 1]],
            {"layout": _layout([(0, 7)])},
            "unknown spin-orbital",
        ),  # pair names an unknown orbital
    ],
)
def test_inconsistent_inputs_raise(imp, bath, blocks, kw, match):
    with pytest.raises(ValueError, match=match):
        OrbitalNaming.build(imp, bath, _bs(blocks), **kw)
