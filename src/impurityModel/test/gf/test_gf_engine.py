"""Direct tests of the shared GF engine helpers (gf_engine.py, GF review Phase 5).

The drivers built on them are covered end to end by the branch matrix against the exact oracle;
these pin the helpers' own contracts, which a driver-level test would only see as a wrong G.
"""

import numpy as np
import pytest

from impurityModel.ed.gf_engine import combine_sides, states_by_group
from impurityModel.ed.gf_units import GFUnit


def _result(tag, n_states_in_unit):
    alphas = np.full((1, 1, 1), tag, dtype=complex)
    return (alphas, alphas, [np.full((1, 1), tag + p, dtype=complex) for p in range(n_states_in_unit)], {}, {})


def test_states_by_group_places_each_stacked_state_in_its_group():
    units = [GFUnit(0, (0, 1), 1, 0.1), GFUnit(1, (0,), 1, -0.1), GFUnit(1, (1,), 1, -0.1)]
    results = [_result(10, 2), _result(20, 1), _result(30, 1)]
    (a0, _b0, r0), (a1, _b1, r1) = states_by_group(units, results, 2, 2)
    assert a0[0] is a0[1] and a0[0][0, 0, 0] == 10  # stacked states share their unit's coefficients
    assert [r[0, 0] for r in r0] == [10, 11]  # but keep their own seed-projection slice
    assert [a[0, 0, 0] for a in a1] == [20, 30] and [r[0, 0] for r in r1] == [20, 30]


def test_states_by_group_rejects_an_uncovered_state():
    with pytest.raises(RuntimeError, match="eigenstate"):
        states_by_group([GFUnit(0, (0,), 1, 0.1)], [_result(1, 1)], 1, 2)


def test_combine_sides_transposes_the_removal_side():
    g_add = np.arange(8, dtype=complex).reshape(2, 2, 2)
    g_rem = np.arange(8, 16, dtype=complex).reshape(2, 2, 2)
    got = combine_sides(g_add, g_rem, 2.0)
    np.testing.assert_array_equal(got, (g_add - np.transpose(g_rem, (0, 2, 1))) / 2.0)
    assert got[0, 0, 1] == (g_add[0, 0, 1] - g_rem[0, 1, 0]) / 2.0
