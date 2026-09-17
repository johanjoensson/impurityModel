"""The per-unit basis report a completed RIXS map prints.

RIXS reaches two bases through three solver tiers, and in both cases the obvious read is wrong:

* the **intermediate (core-excited) resolvent** -- ``KrylovShiftedResolvent.solve`` leaves
  ``tmp_basis`` at the right-hand side's support and tracks the recurrence only in its cap
  proxy, so ``tmp_basis.size`` reports the seed. Each tier reports its own support instead
  (the spectral cache its sector, the recycler its proxy count, the BiCGSTAB fallback the
  basis it grew in place).
* the **final-state basis** -- the tensor path's out-resolvent cache serves a hit *without*
  regrowing ``green_basis``, which was cleared at the top of the unit, so reading it back
  gives 0 for a solve that really ran on the cached sector. ``eval_out`` reports what it used.

Both are the seed-vs-support trap that the self-energy path already hit once; these tests pin
that the two RIXS drivers agree with each other, which they cannot do if either reverts.
"""

import numpy as np
import pytest

from impurityModel.ed import rixs

from .test_rixs_tensor import (
    EPS_IN,
    EPS_OUT,
    _model,
    _run_rixs,
    _run_rixs_tensor,
    _thermal_states,
    _tin_tout,
)


@pytest.fixture
def fixture():
    op = _model()
    psis, es, dets, _states, _vecs = _thermal_states(op, 2)
    tin, tout = _tin_tout()
    return op, psis, es, dets, tin, tout


def _reports(out):
    """``{heading: {row label: size or None}}`` for every report in ``out``."""
    reports, current = {}, None
    for line in out.splitlines():
        if line.startswith("Maximum ") and line.endswith(":"):
            current = reports.setdefault(line[:-1], {})
        elif current is not None and line.startswith("  ") and "maximum over" not in line:
            label, _, rest = line.strip().rpartition("  ")
            current[label.strip()] = None if "not tracked" in rest else int(rest.split()[0].replace(",", ""))
        elif not line.startswith("  "):
            current = None
    return reports


def _only(reports, needle):
    matches = [v for k, v in reports.items() if needle in k]
    assert len(matches) == 1, f"expected one {needle!r} report, got {len(matches)}: {list(reports)}"
    return matches[0]


def test_calc_map_reports_both_bases_per_eigenstate(capsys, fixture):
    op, psis, es, dets, tin, tout = fixture
    _run_rixs(op, psis, es, tin, tout, dets)
    reports = _reports(capsys.readouterr().out)
    intermediate = _only(reports, "intermediate-resolvent")
    final = _only(reports, "final-state")
    assert set(intermediate) == {f"eigenstate {e}" for e in range(len(es))}
    assert set(final) == set(intermediate)
    # Every row is measured -- the fixture's cap is finite, as both CLIs always arrange.
    assert all(size is not None for size in {**intermediate, **final}.values())
    assert max(final.values()) > 0


def test_the_two_rixs_drivers_agree_on_the_intermediate_support(capsys, fixture):
    """``calc_map`` takes the BiCGSTAB fallback and ``calc_tensor_map`` the spectral cache on
    this fixture, so agreement here is agreement *across two different solver tiers* -- which
    is exactly what a stale ``tmp_basis.size`` read would destroy."""
    op, psis, es, dets, tin, tout = fixture
    _run_rixs(op, psis, es, tin, tout, dets)
    plain = _only(_reports(capsys.readouterr().out), "intermediate-resolvent")
    _run_rixs_tensor(op, psis, es, tin, tout, dets, EPS_IN, EPS_OUT)
    tensor = _only(_reports(capsys.readouterr().out), "intermediate-resolvent")
    assert plain == tensor, (plain, tensor)


def test_the_tensor_path_does_not_report_a_cleared_final_basis(capsys, fixture):
    """The out-resolvent cache serves this fixture, and ``green_basis`` is cleared per unit and
    never regrown on a hit. Reporting its size would give 0 for units that solved on the cached
    sector; the number must instead match what the per-operator driver measures."""
    op, psis, es, dets, tin, tout = fixture
    _run_rixs(op, psis, es, tin, tout, dets)
    plain = _only(_reports(capsys.readouterr().out), "final-state")
    _run_rixs_tensor(op, psis, es, tin, tout, dets, EPS_IN, EPS_OUT)
    tensor = _only(_reports(capsys.readouterr().out), "final-state")
    assert max(tensor.values()) == max(plain.values()) > 0
    # The trap this pins: an eigenstate that solved off the cache reported 0 before the fix.
    solved = [e for e, size in tensor.items() if plain[e] > 0]
    assert solved and all(tensor[e] > 0 for e in solved), (plain, tensor)


def test_the_adaptive_sampler_reports_once_for_the_whole_map(capsys, fixture):
    """The adaptive wIn sampler runs the flat driver once per refinement pass; the report
    belongs to the finished map, so the accumulator is owned by the caller across passes."""
    op, psis, es, dets, tin, tout = fixture
    passes = []
    real_flat = rixs._rixs_map_flat

    def counting(*args, **kwargs):
        passes.append(1)
        return real_flat(*args, **kwargs)

    rixs._rixs_map_flat = counting
    try:
        _run_rixs_tensor(op, psis, es, tin, tout, dets, EPS_IN, EPS_OUT)
    finally:
        rixs._rixs_map_flat = real_flat
    out = capsys.readouterr().out
    assert out.count("Maximum intermediate-resolvent basis size per unit") == 1
    assert out.count("Maximum final-state basis size per unit") == 1
    assert len(passes) >= 1


def test_a_chain_with_an_unmeasured_solve_reports_not_tracked():
    """A unit whose intermediate support went unmeasured says so rather than reporting a lower
    bound over an unknown subset of its wIn points."""
    chain = rixs._R1SolverChain(None, eigenstate=0, counters=None)
    chain._record_support(12)
    assert rixs._chain_support(chain) == 12
    chain._record_support(None)
    assert rixs._chain_support(chain) is None


def test_the_shift_recycler_reports_its_proxy_count_not_the_seed_support():
    """``KrylovShiftedResolvent.solve`` clears its basis, refills it from the right-hand side
    and then runs the recurrence on a cap proxy, so ``basis.size`` afterwards is the SEED
    support. ``cap_info`` is the only honest source. An eight-orbital chain seeded with a single
    determinant makes the two differ: reading the basis back would report 1."""
    from impurityModel.ed import greens_function as gf
    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

    terms = {((i, "c"), (i, "a")): -1.0 + 0.3 * i for i in range(8)}
    for i in range(7):
        terms[((i, "c"), (i + 1, "a"))] = 0.7
        terms[((i + 1, "c"), (i, "a"))] = 0.7
    seed = b"\xc0"  # orbitals 0 and 1 occupied
    basis = Basis(
        impurity_orbitals={0: [list(range(8))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[seed],
        comm=None,
        truncation_threshold=10_000,  # finite, non-binding: switches tracking on
    )
    rhs = [ManyBodyState({SlaterDeterminant.from_bytes(seed): 1.0})]
    info = {}
    sols = gf.KrylovShiftedResolvent().solve(
        basis,
        ManyBodyOperator(terms),
        rhs,
        np.array([0.7 + 0.3j, -1.1 + 0.2j]),
        slaterWeightMin=0.0,
        atol=1e-10,
        cap_info=info,
    )
    assert sols is not None
    assert set(info) >= {"cap_hit", "retained_size"}
    assert info["cap_hit"] is False
    # The hopping chain reaches far beyond the single seed determinant `basis.size` holds.
    assert basis.size == 1
    assert info["retained_size"] > 1


def test_the_shift_recycler_fills_cap_info_on_an_early_decline():
    """Every return path leaves the caller a complete dict, so a decline is distinguishable
    from a solve that simply was not tracked."""
    from impurityModel.ed import greens_function as gf

    from .test_rixs_tensor import _basis

    info = {}
    gf.KrylovShiftedResolvent().solve(_basis([]), _model(), [], np.array([0.1 + 0.1j]), cap_info=info)
    assert info == {"cap_hit": False, "retained_size": None}
