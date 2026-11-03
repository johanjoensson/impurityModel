"""``merge_measurements``: one member is passed through untouched; several concatenate unshifted energies."""

import numpy as np
import pytest

from impurityModel.ed.gs_report import measure_ground_state, merge_measurements
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant


def _det(i):
    return SlaterDeterminant((i,))


def _member(keys, es, total):
    psis = [ManyBodyState({k: 1.0}) for k in keys]
    return measure_ground_state(f"basis{total}", np.array(es), psis, total=total)


def test_one_member_is_returned_untouched():
    m = _member([_det(1)], [-1.0], 1)
    basis, es, psis = merge_measurements([m])
    assert basis is m.basis and es is m.es and psis is m.psis


def test_members_concatenate_with_absolute_energies_dominant_first():
    a = _member([_det(1), _det(2)], [-2.0, -1.9], 1)
    b = _member([_det(4)], [-1.95], 2)
    basis, es, psis = merge_measurements([a, b])
    assert basis == "basis1"
    assert list(es) == [-2.0, -1.9, -1.95]
    assert len(psis) == 3


def test_empty_raises():
    with pytest.raises(ValueError):
        merge_measurements([])
