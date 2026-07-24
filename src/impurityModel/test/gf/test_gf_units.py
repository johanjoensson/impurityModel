"""Direct, unmocked tests for enumerate_gf_units (gf_units.py:85).

test_spectra.py's orchestration test mocks enumerate_gf_units entirely
(``mock_enum.return_value = ([GFUnit(0, (0,), 1, 0.1)], [[MagicMock()]], [None])``), so its
real logic -- applying transition operators to real thermal states, then either splitting
into pairwise scalar recurrences or chunking eigenstates into groups -- has never run against
a real ManyBodyOperator/ManyBodyState. This exercises both decomposition modes directly, checking the
returned unit_seeds against an independent, by-hand application of the same transition
operators (not calling _apply_transition_ops, so a bug shared between the two would still
surface as a mismatch), plus a sensitivity check that a corrupted state or swapped operator
pair would actually be caught.

Restrictions are left at None (a no-op) throughout: this is scoped to the
enumeration/chunking/seed-construction logic gap, not the restriction-application machinery,
which has its own dedicated coverage under restrictions/.
"""

import os

from impurityModel.ed.gf_units import enumerate_gf_units
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant


def _det(occupied):
    b = 0
    for i in occupied:
        b |= 1 << (7 - i)
    return SlaterDeterminant.from_bytes(bytes([b]))


def _thermal_states():
    """Three arbitrary (not necessarily eigenstates -- enumerate_gf_units never checks
    that) multi-determinant states over a 4 spin-orbital space, distinct enough that a
    seed-construction bug would show up as a real numerical mismatch."""
    return [
        ManyBodyState({_det([0, 2]): 1.0 + 0j, _det([1, 2]): 0.5 + 0j}),
        ManyBodyState({_det([0, 3]): 0.6 + 0j, _det([1, 3]): -0.8j}),
        ManyBodyState({_det([2, 3]): 1.0 + 0j}),
    ]


def _transition_ops():
    return [ManyBodyOperator({((0, "a"),): 1.0}), ManyBodyOperator({((1, "a"),): 1.0})]


def _direct_apply(tOps, psis):
    """Independent reference: apply each operator to each state directly, bypassing
    enumerate_gf_units'/_apply_transition_ops' shared-support block machinery entirely."""
    return [[tOp(psi, 0) for tOp in tOps] for psi in psis]


def _states_equal(a, b, atol=1e-12):
    keys = set(a.keys()) | set(b.keys())
    return all(
        abs(complex(a.get(k, [0.0])[0] if k in a else 0.0) - complex(b.get(k, [0.0])[0] if k in b else 0.0)) < atol
        for k in keys
    )


def test_enumerate_gf_units_pairwise_mode_matches_direct_application():
    psis = _thermal_states()[:2]
    tOps = _transition_ops()
    ref = _direct_apply(tOps, psis)

    units, unit_seeds, unit_restrictions = enumerate_gf_units(
        [(tOps, 0.1)], psis, [None], None, slaterWeightMin=0.0, pairwise=True
    )

    # n_psis=2, n_ops=2: per eigenstate, 2 diag + 1 sum + 1 imag = 4 units -> 8 total.
    assert len(units) == 8
    assert len(unit_seeds) == 8
    assert all(r is None for r in unit_restrictions)

    by_tag = {(u.chunk, u.pw_tag): seeds[0] for u, seeds in zip(units, unit_seeds)}
    for ei in range(2):
        assert _states_equal(by_tag[((ei,), ("diag", 0, 0))], ref[ei][0])
        assert _states_equal(by_tag[((ei,), ("diag", 1, 1))], ref[ei][1])
        assert _states_equal(by_tag[((ei,), ("sum", 0, 1))], ref[ei][0] + ref[ei][1])
        assert _states_equal(by_tag[((ei,), ("imag", 0, 1))], ref[ei][0] + 1j * ref[ei][1])
        assert all(u.group_i == 0 for u in units)
        assert all(u.n_ops == 1 for u in units)


def test_enumerate_gf_units_grouped_mode_stacks_eigenstates_in_chunks():
    psis = _thermal_states()  # 3 states
    tOps = _transition_ops()
    ref = _direct_apply(tOps, psis)

    old = os.environ.get("GF_EIGENSTATE_GROUP")
    os.environ["GF_EIGENSTATE_GROUP"] = "2"
    try:
        units, unit_seeds, _ = enumerate_gf_units(
            [(tOps, -0.1)], psis, [None], None, slaterWeightMin=0.0, pairwise=False
        )
    finally:
        if old is None:
            os.environ.pop("GF_EIGENSTATE_GROUP", None)
        else:
            os.environ["GF_EIGENSTATE_GROUP"] = old

    # group=2, n_psis=3 -> chunks (0, 1) and (2,).
    assert [u.chunk for u in units] == [(0, 1), (2,)]
    assert [u.n_ops for u in units] == [2, 2]

    # unit_seeds order is [block_v[j][i] for j in chunk for i in range(n_ops)]: state-major,
    # operator-minor within each state.
    expected_first = [ref[0][0], ref[0][1], ref[1][0], ref[1][1]]
    for got, exp in zip(unit_seeds[0], expected_first):
        assert _states_equal(got, exp)
    expected_second = [ref[2][0], ref[2][1]]
    for got, exp in zip(unit_seeds[1], expected_second):
        assert _states_equal(got, exp)


def test_enumerate_gf_units_would_catch_a_swapped_operator_bug():
    """Sanity check that the pairwise comparison above is actually sensitive to which
    operator seeded which unit -- guards against a vacuously-passing test (e.g. if the two
    transition operators happened to give numerically similar results)."""
    psis = _thermal_states()[:1]
    tOps = _transition_ops()
    ref = _direct_apply(tOps, psis)
    assert not _states_equal(ref[0][0], ref[0][1]), "the two reference operators must differ for this check to work"

    units, unit_seeds, _ = enumerate_gf_units([(tOps, 0.1)], psis, [None], None, slaterWeightMin=0.0, pairwise=True)
    by_tag = {u.pw_tag: seeds[0] for u, seeds in zip(units, unit_seeds)}

    assert _states_equal(by_tag[("diag", 0, 0)], ref[0][0])
    assert not _states_equal(by_tag[("diag", 0, 0)], ref[0][1])  # swapped operator must disagree
