"""The full report folds symmetry copies: bath levels, density-matrix entries, pure natural orbitals."""

from typing import ClassVar
import numpy as np

from impurityModel.ed.block_structure import BlockStructure, block_structure_summary
from impurityModel.ed.gs_report import _restriction_label
from impurityModel.ed.observables import collapse_screening_levels, print_screening_diagnostics
from impurityModel.ed.utils import print_density_matrix_summary


def _row(pair, eps, v, n=0.9, sisb=-0.1):
    return dict(pair=pair, eps_dn=eps, eps_up=eps, n_dn=n, n_up=n, v_dn=v, v_up=v, sisb=sisb)


def test_equal_bath_levels_fold_and_a_broken_symmetry_stays_expanded():
    levels = [
        _row((2, 3), -1.0, 0.5),
        _row((4, 5), -1.0, 0.5),
        _row((6, 7), -1.0, 0.5),
        _row((8, 9), -1.0, 0.5000 + 1e-3),
    ]
    classes = collapse_screening_levels(levels)
    assert [c for _, c in classes] == [3, 1]


def test_uncoupled_levels_are_counted_not_listed(capsys):
    scr = dict(
        z_only=False,
        with_correlation=True,
        channels=[("group 0", -0.1)],
        levels=[_row((2, 3), -1.0, 0.5), _row((4, 5), -2.0, 0.0), _row((6, 7), -3.0, 0.0)],
    )
    print_screening_diagnostics(scr)
    out = capsys.readouterr().out
    assert "2 bath level(s) with |V| = 0" in out
    assert "(4,5)" not in out and "(2,3)" in out


def test_equal_density_matrix_entries_print_once_with_a_multiplicity(capsys):
    m = np.eye(8, dtype=complex)
    for i, j in ((0, 1), (2, 3), (4, 5)):
        m[i, j] = m[j, i] = 0.02
    m[6, 7] = m[7, 6] = 0.03
    print_density_matrix_summary(m)
    out = capsys.readouterr().out
    assert "x3" in out and out.count("0.02000") == 1 and "0.03000" in out
    assert "4 of 28" in out


def test_restrictions_are_named_by_the_group_they_equal():
    class _Info:
        def __init__(self, role, key):
            self.role, self.group_key = role, key

    class _Naming:
        infos: ClassVar[dict] = {
            0: _Info("impurity", 0),
            1: _Info("impurity", 0),
            2: _Info("valence", 0),
            3: _Info("conduction", 0),
        }

        def indices(self, role=None, group_key=None):
            return sorted(i for i, o in self.infos.items() if (role is None or o.role == role))

    assert _restriction_label([2], _Naming()).startswith("group 0 valence bath (1)")
    assert _restriction_label([0, 1], _Naming()).startswith("impurity (2)")
    assert _restriction_label([1, 2], _Naming()) == "1-2"


def test_block_structure_summary_counts_copies():
    bs = BlockStructure(
        blocks=[[0], [1], [2]],
        identical_blocks=[[1, 2], [], []],
        transposed_blocks=[[], [], []],
        particle_hole_blocks=[[], [], []],
        particle_hole_transposed_blocks=[[], [], []],
        inequivalent_blocks=[0],
    )
    assert block_structure_summary(bs) == "3 block(s), 1 inequivalent: 3 x 1-orbital"
