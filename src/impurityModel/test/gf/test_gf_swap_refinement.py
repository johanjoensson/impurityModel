"""Swap refinement of a capped unit (``gf_refinement.refine_frozen_unit``) on the 6-orbital SIAM, against the dense
resolvent of the whole reachable space."""

import numpy as np
import pytest

from impurityModel.ed.gf_refinement import RefineSpec as _RefineSpec
from impurityModel.ed.gf_solvers import block_Green_sparse
from impurityModel.ed.greens_function import calc_G
from impurityModel.test.gf.test_gf_truncation import _redistribute_as_width1
from impurityModel.test.support.comms import COMMS
from impurityModel.test.support.gf_oracles import DELTA, OMEGA, _dense_G_on, _seed_basis, _seeds, _siam_6


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("APPLY_ROW_CHUNKS", "1")
    for name in ("GF_LANCZOS_ADMIT_TOL", "GF_ADMIT_FIRST_SHELL_TOL", "GF_PRUNE_RULE"):
        monkeypatch.delenv(name, raising=False)


def _mesh():
    return [OMEGA + 1j * DELTA, 1j * np.linspace(0.05, 2.0, 6)]


#: The scoring shifts a block's table would give this unit (addition side, no thermal energy: the mesh's own frame).
SHIFTS = tuple((complex(z), 1.0) for z in list(OMEGA + 1j * DELTA) + list(1j * np.linspace(0.05, 2.0, 6)))


#: The acceptance test's check shifts (``(z, weight, axis)``, the same frame): every other real-mesh point and two
#: Matsubara points, unit weights.
CHECKS = tuple((complex(z), 1.0, 0) for z in OMEGA[::2] + 1j * DELTA) + ((0.05j, 1.0, 1), (0.44j, 1.0, 1))


def RefineSpec(**kw):
    kw.setdefault("shifts", SHIFTS)
    kw.setdefault("checks", CHECKS)
    return _RefineSpec(**kw)


def _run(cap, refine=None, comm=None, prune_tol=None):
    basis = _seed_basis(cap=cap, comm=comm)
    full = _seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    info = {}
    alphas, betas, r = block_Green_sparse(
        _siam_6(),
        seeds,
        basis,
        DELTA,
        verbose=False,
        cap_info=info,
        eval_meshes=_mesh(),
        refine=refine,
        prune_tol=prune_tol,
    )
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA), info


def _gather(keys, comm):
    if comm is not None:
        keys = [k for part in comm.allgather(keys) for k in part]
    return sorted(set(keys))


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("cap", [10, 11, 15, 16])
def test_refinement_keeps_the_cap_and_beats_breadth_first_growth(cap, comm):
    """On the 18-determinant closure of the fixture a swap round at these caps cuts the error against the dense
    resolvent of the whole space by well over a half (measured 2.5x to 10x). The estimate is first order, so this is
    not claimed for every cap (12 and 14 get worse by about 40 %: see the doc)."""
    truth, closure = _run(np.inf, comm=comm)
    assert closure["retained_size"] == 18
    g_breadth, info0 = _run(cap, comm=comm)
    assert info0["cap_hit"] and info0["retained_size"] == cap
    g_ref, info = _run(cap, RefineSpec(rounds=6, tol=1e-12, swap=0.4), comm=comm)
    assert info["retained_size"] == cap
    rec = info["refine"]
    assert rec["declined"] is None and sum(rec["swapped"]) >= 1, rec
    assert np.abs(g_ref - truth).max() < 0.6 * np.abs(g_breadth - truth).max()


@pytest.mark.parametrize("comm", COMMS)
def test_a_refinement_that_finds_nothing_to_swap_leaves_the_unit_alone(comm):
    g_breadth, _ = _run(17, comm=comm)
    g_ref, info = _run(17, RefineSpec(rounds=3, tol=1e-12, swap=0.4), comm=comm)
    assert info["refine"]["swapped"] == [] and info["refine"]["rounds"] == 0
    np.testing.assert_allclose(g_ref, g_breadth, atol=1e-10)  # another matvec mode: equal to rounding


def test_no_refinement_without_a_cap_hit():
    _g, info = _run(np.inf, RefineSpec(rounds=3))
    assert "refine" not in info


@pytest.mark.parametrize("comm", COMMS)
def test_the_refined_support_is_the_exact_resolvent_of_PHP(comm):
    g_ref, info = _run(10, RefineSpec(rounds=3, tol=1e-12, swap=0.4), comm=comm)
    kept = _gather(info["proxy"].retained_keys(), comm)
    np.testing.assert_allclose(g_ref, _dense_G_on(kept, OMEGA + 1j * DELTA), atol=1e-9)


def _spy_scores(monkeypatch, residuals=None):
    """Record the arguments of every ``importance_scores`` call; ``residuals`` replaces the solves' residuals."""
    from impurityModel.ed import gf_refinement

    calls, real = [], gf_refinement.importance_scores

    def spy(matrix, seeds, zs, **kw):
        out = real(matrix, seeds, zs, **kw)
        calls.append((list(zs), kw))
        if residuals is not None:
            out.residuals = list(residuals)
        return out

    monkeypatch.setattr(gf_refinement, "importance_scores", spy)
    return calls


@pytest.mark.parametrize("comm", COMMS)
def test_the_scores_are_taken_at_the_spec_shifts_with_their_weights_and_nothing_is_solved(monkeypatch, comm):
    calls = _spy_scores(monkeypatch)
    weighted = tuple((z, 0.25 * (i + 1)) for i, (z, _w) in enumerate(SHIFTS))
    _run(10, RefineSpec(rounds=1, tol=1e-12, swap=0.4, shifts=weighted, weighted=True), comm=comm)
    zs, kw = calls[0]
    assert zs == [z for z, _w in weighted]
    assert list(kw["weights"]) == [w for _z, w in weighted]
    assert kw["xs"] is not None and "rtol" not in kw  # the recurrence's solutions, no BiCGSTAB


@pytest.mark.parametrize("comm", COMMS)
def test_an_unweighted_spec_scores_every_shift_at_weight_one(monkeypatch, comm):
    calls = _spy_scores(monkeypatch)
    weighted = tuple((z, 0.3) for z, _w in SHIFTS)
    _run(10, RefineSpec(rounds=1, tol=1e-12, swap=0.4, shifts=weighted), comm=comm)
    assert set(calls[0][1]["weights"]) == {1.0}


@pytest.mark.parametrize("comm", COMMS)
def test_a_spec_without_shifts_is_declined_with_the_reason(comm):
    _g, info = _run(10, _RefineSpec(rounds=2), comm=comm)
    assert "refine" in info and info["refine"]["swapped"] == [] and info["refine"]["declined"]


@pytest.mark.parametrize("comm", COMMS)
def test_the_spec_stop_tolerance_reaches_every_recurrence_and_later_rounds_stop_on_the_residual_alone(
    monkeypatch, comm
):
    from impurityModel.ed import gf_recurrence

    calls, real = [], gf_recurrence.block_lanczos_discovered

    def spy(*args, **kw):
        calls.append((kw.get("shifted") is not None, kw.get("shifted_rtol"), kw.get("score_only", False)))
        return real(*args, **kw)

    monkeypatch.setattr(gf_recurrence, "block_lanczos_discovered", spy)
    _run(10, RefineSpec(rounds=2, tol=1e-12, swap=0.4, stop_rtol=3e-5), comm=comm)
    with_acc = [c for c in calls if c[0]]
    assert with_acc and all(c[1] == 3e-5 for c in with_acc)
    assert with_acc[0][2] is False and any(c[2] for c in with_acc[1:]), calls  # round 0 full, later rounds score-only
    assert calls[-1][0] is False  # the final recurrence, to Green's-function convergence, carries no accumulator


@pytest.mark.parametrize("comm", COMMS)
def test_the_unit_settles_on_the_tolerance_and_says_so(comm):
    """``gain <= tol * max|G|``: a huge tolerance settles at once (no swap made), a tiny one with one round does
    not and swaps once."""
    _g, loose = _run(10, RefineSpec(rounds=3, tol=1e9, swap=0.4), comm=comm)
    assert loose["refine"]["settled"] and loose["refine"]["swapped"] == []
    _g, tight = _run(10, RefineSpec(rounds=1, tol=1e-12, swap=0.4), comm=comm)
    assert not tight["refine"]["settled"] and len(tight["refine"]["swapped"]) == 1


@pytest.mark.parametrize("comm", COMMS)
def test_a_refinement_whose_estimate_grows_falls_back_to_round_zeros_support(monkeypatch, comm):
    """The fallback restores round 0's P (the matrix, the support and the returned G agree on it)."""
    from impurityModel.ed import gf_refinement

    g_breadth, info0 = _run(10, comm=comm)
    real, calls = gf_refinement._select_swap, []

    def worse(matrix, scores, exempt, k_max):
        retire, admit, gain = real(matrix, scores, exempt, k_max)
        calls.append(1)
        return retire, admit, gain * (1.0 if len(calls) == 1 else 1e6)

    monkeypatch.setattr(gf_refinement, "_select_swap", worse)
    g_ref, info = _run(10, RefineSpec(rounds=3, tol=1e-12, swap=0.4), comm=comm)
    assert info["refine"].get("fell_back") is True, info["refine"]
    np.testing.assert_allclose(g_ref, g_breadth, atol=1e-10)
    assert _gather(info["proxy"].retained_keys(), comm) == _gather(info0["proxy"].retained_keys(), comm)


@pytest.mark.parametrize("prune_tol", [None, 1e-6])
@pytest.mark.parametrize("comm", COMMS)
def test_a_unit_with_nothing_swappable_is_declined_before_any_scoring_is_computed(monkeypatch, comm, prune_tol):
    """Review R4: when the exemption covers every retained row, the scoring tail is skipped: no scores, no swaps,
    breadth-first's G."""
    from impurityModel.ed import gf_refinement

    g_breadth, _ = _run(10, comm=comm, prune_tol=prune_tol)
    calls = _spy_scores(monkeypatch)
    monkeypatch.setattr(
        gf_refinement,
        "_exempt_pslots",
        lambda matrix, seed_keys, seed_block, extra: np.ones(matrix.n_p_local, dtype=bool),
    )
    g_ref, info = _run(10, RefineSpec(rounds=3, tol=1e-12, swap=0.4), comm=comm, prune_tol=prune_tol)
    assert "nothing is swappable" in info["refine"]["declined"], info["refine"]
    assert calls == [] and info["refine"]["swapped"] == []
    np.testing.assert_allclose(g_ref, g_breadth, atol=1e-10)


@pytest.mark.parametrize("comm", COMMS)
def test_a_final_recurrence_that_diverges_falls_back_to_round_zeros_support(monkeypatch, comm):
    """The Green's-function rerun on the swapped support is checked for divergence like the scoring reruns."""
    from impurityModel.ed import gf_recurrence

    g_breadth, info0 = _run(10, comm=comm)
    real, calls = gf_recurrence.block_lanczos_discovered, []

    def spy(*args, **kw):
        out = real(*args, **kw)
        calls.append((kw.get("shifted") is None, kw.get("score_only", False)))
        if len(calls) > 1 and calls[-1] == (True, False):  # the final rerun: no accumulator, not score-only
            return out[:3] + ("diverged",) + out[4:]
        return out

    monkeypatch.setattr(gf_recurrence, "block_lanczos_discovered", spy)
    # Stops on "no swap gains" before the round limit, so the Green's function needs its own recurrence.
    g_ref, info = _run(10, RefineSpec(rounds=8, tol=1e-12, swap=0.4), comm=comm)
    assert info["refine"].get("fell_back") is True and "final" in info["refine"]["stopped"], info["refine"]
    np.testing.assert_allclose(g_ref, g_breadth, atol=1e-10)
    assert _gather(info["proxy"].retained_keys(), comm) == _gather(info0["proxy"].retained_keys(), comm)


@pytest.mark.parametrize("comm", COMMS)
def test_seeds_that_are_not_real_up_to_a_phase_are_refined_with_the_conjugate_shift_solve(monkeypatch, comm):
    """A column with a relative phase between its determinants (the real archives' seeds) is not declined: the
    scores need ``Y = R(z*) s`` as well, and the refinement still keeps the cap, swaps, and beats breadth-first
    growth against the dense resolvent."""
    import sys

    from impurityModel.ed.ManyBodyUtils import ManyBodyState
    from impurityModel.test.support.gf_oracles import _det

    module = sys.modules[__name__]
    real_seeds = module._seeds

    def complex_seeds():
        _a, b = real_seeds()
        return [ManyBodyState({_det([0, 2, 3]): 1.0 + 0j, _det([1, 2, 3]): 0.5j}), b]

    monkeypatch.setattr(module, "_seeds", complex_seeds)
    truth, _ = _run(np.inf, comm=comm)
    g_breadth, _ = _run(10, comm=comm)
    g_ref, info = _run(10, RefineSpec(rounds=6, tol=1e-12, swap=0.4), comm=comm)
    rec = info["refine"]
    assert rec["declined"] is None and sum(rec["swapped"]) >= 1, rec
    assert info["retained_size"] == 10
    assert np.abs(g_ref - truth).max() < 0.8 * np.abs(g_breadth - truth).max()


@pytest.mark.parametrize("comm", COMMS)
def test_a_pruned_unit_restarts_with_the_accumulator_and_is_refined(comm):
    """With pruning the first run is discarded and the restart carries the accumulator (the production default)."""
    truth, _ = _run(np.inf, comm=comm)
    g_ref, info = _run(10, RefineSpec(rounds=6, tol=1e-12, swap=0.4), comm=comm, prune_tol=1e-6)
    rec = info["refine"]
    assert rec["declined"] is None and info["retained_size"] == 10, rec
    assert np.isfinite(g_ref).all() and np.abs(g_ref - truth).max() < 1.0


def test_the_unit_shifts_are_signed_for_the_side_offset_by_the_thermal_energies_and_deduplicated():
    from impurityModel.ed.gf_convergence import _gf_signed_axes
    from impurityModel.ed.gf_refinement import unit_refine_spec

    table = ((0.3 + 0.02j, 1.0), (0.0 + 0.5j, 0.5))
    spec = _RefineSpec(rounds=1, tables=(table,))
    for side, sign in ((0, 1.0), (1, -1.0)):
        got = unit_refine_spec(spec, 0, side, [2.5]).shifts
        assert [z for z, _w in got] == [sign * z + 2.5 for z, _w in table]
    real = _gf_signed_axes(None, np.array([0.3]), 1, 0.02)[0][0]
    assert unit_refine_spec(spec, 0, 1, [0.0]).shifts[0][0] == pytest.approx(real)  # the monitor's own frame
    assert len(unit_refine_spec(spec, 0, 0, [1.0, 1.0 + 1e-12]).shifts) == len(table)  # degenerate: no repeats
    stacked = unit_refine_spec(spec, 0, 0, [0.0, 2.5])  # review F1/R3: no cross-state scoring
    assert stacked.shifts == () and "stacked thermal states" in stacked.note
    from impurityModel.ed.gf_refinement import scoring_plan

    assert "stacked thermal states" in scoring_plan(stacked, None, None, True)[1]
    assert unit_refine_spec(spec, 5, 0, [0.0]).shifts == ()  # no table for the block
    checks = ((0.3 + 0.02j, 4.0, 0), (0.5j, 2.0, 1))
    spec = _RefineSpec(rounds=1, tables=(table,), check_tables=(checks,))
    for side, sign in ((0, 1.0), (1, -1.0)):
        got = unit_refine_spec(spec, 0, side, [2.5]).checks
        assert got == tuple((sign * z + 2.5, w, a) for z, w, a in checks)  # the scoring shifts' frame
    assert unit_refine_spec(spec, 0, 0, [0.0, 2.5]).checks == ()  # declined units carry none


_REAL_SIAM_6 = _siam_6


def _complex_siam_6():
    """The fixture's SIAM with complex Hermitian hopping (phases on every impurity-bath term)."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    terms = dict(_REAL_SIAM_6().items())
    for (a, b), phase in zip(((0, 2), (1, 3), (0, 4), (1, 5)), (0.7, -0.4, 1.9, 0.3)):
        v = terms[((a, "c"), (b, "a"))]
        terms[((a, "c"), (b, "a"))] = v * np.exp(1j * phase)
        terms[((b, "c"), (a, "a"))] = v * np.exp(-1j * phase)
    return ManyBodyOperator(terms)


@pytest.mark.parametrize("comm", COMMS)
def test_a_complex_hermitian_hamiltonian_is_refined_with_the_conjugate_shifts(monkeypatch, comm):
    """Step 3.7: nothing assumes a real ``H`` or complex-symmetric solutions. The scores take ``Y = R(z*) s`` from the
    conjugate shifts the accumulator carries, the unit keeps its cap, swaps, and beats breadth-first growth against the
    dense resolvent of the same complex operator."""
    import sys

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_siam_6", _complex_siam_6)
    truth, _ = _run(np.inf, comm=comm)
    g_breadth, _ = _run(10, comm=comm)
    seen = []
    from impurityModel.ed import gf_refinement

    real = gf_refinement.importance_scores

    def spy(matrix, seeds, zs, xs, ys=None, **kw):
        seen.append(ys is not None)
        return real(matrix, seeds, zs, xs, ys, **kw)

    monkeypatch.setattr(gf_refinement, "importance_scores", spy)
    g_ref, info = _run(10, RefineSpec(rounds=6, tol=1e-12, swap=0.4), comm=comm)
    rec = info["refine"]
    assert rec["declined"] is None and sum(rec["swapped"]) >= 1, rec
    assert seen and all(seen), "a complex H needs the conjugate-shift solutions"
    assert info["retained_size"] == 10
    assert np.abs(g_ref - truth).max() < 0.8 * np.abs(g_breadth - truth).max()


def test_the_default_scoring_residual_is_inside_the_range_the_oracle_ranks_exactly():
    """Review S1: ranks are exact at residual <= 2e-3, so the default must not exceed it (and 1e-4 was ~20x tighter
    than needed at the bath poles, where the residual falls slowly)."""
    from impurityModel.ed import gf_refinement

    assert gf_refinement.REFINE_STOP_RTOL <= 2e-3


@pytest.mark.parametrize("comm", COMMS)
def test_the_last_permitted_round_is_also_the_green_function_run(monkeypatch, comm):
    """Review R6: a refinement that uses all its rounds ends on a recurrence that is both scored and converged, so
    there is no further accumulator-free recurrence; one that stops early still needs it."""
    from impurityModel.ed import gf_recurrence

    real = gf_recurrence.block_lanczos_discovered

    def run_with(rounds):
        calls = []

        def spy(*args, **kw):
            calls.append((kw.get("shifted") is None, kw.get("score_only", False)))
            return real(*args, **kw)

        monkeypatch.setattr(gf_recurrence, "block_lanczos_discovered", spy)
        _g, info = _run(10, RefineSpec(rounds=rounds, tol=1e-12, swap=0.4), comm=comm)
        return calls, info["refine"]

    calls, rec = run_with(1)
    assert rec["rounds"] == 1 and (True, False) not in calls[1:], calls
    calls, rec = run_with(8)
    assert rec["rounds"] < 8 and calls[-1] == (True, False), (calls, rec)


@pytest.mark.parametrize("comm", COMMS)
def test_the_settle_threshold_is_taken_in_the_frame_the_gain_is_measured_in(monkeypatch, comm):
    """Review R7: the scores see the orthonormal seeds, so the gain does not scale with the seed norm while G does.
    The threshold's reference |G| must not either (it was ``|r|^2`` times too small for small seeds)."""
    import sys

    module = sys.modules[__name__]
    real_seeds = module._seeds

    def run(c):
        monkeypatch.setattr(module, "_seeds", lambda: [c * s for s in real_seeds()])
        _g, info = _run(10, RefineSpec(rounds=1, tol=1e-12, swap=0.4), comm=comm)
        return info["refine"]

    big, small = run(1.0), run(0.1)
    assert small["scale"] == pytest.approx(0.01 * big["scale"], rel=1e-6)
    assert small["scale_gain"] == pytest.approx(big["scale_gain"], rel=1e-6)
    assert small["gain"][0] == pytest.approx(big["gain"][0], rel=1e-6)


def _worst_swaps(monkeypatch, from_call):
    """From the ``from_call``-th selection on, retire the *most* important retained rows for the *least* important
    candidates (the importance order inverted; the candidates keep a large offset so the swap is still taken)."""
    from impurityModel.ed import gf_refinement

    real, calls = gf_refinement._select_swap, []

    def bad(matrix, scores, exempt, k_max):
        calls.append(1)
        if len(calls) >= from_call:
            scores.s_ret = 1.0 / (1.0 + np.nan_to_num(scores.s_ret, neginf=0.0))
            scores.s_cand = 1e6 + 1.0 / (1.0 + np.nan_to_num(scores.s_cand, neginf=0.0))
        return real(matrix, scores, exempt, k_max)

    monkeypatch.setattr(gf_refinement, "_select_swap", bad)


@pytest.mark.parametrize("comm", COMMS)
def test_a_round_that_raises_the_estimated_sigma_error_is_undone_and_ends_the_refinement(monkeypatch, comm):
    """Review F4/R1. A first swap that worsens the estimated self-energy error returns round 0's G; a later one returns
    the previous round's (same as stopping there)."""
    g_breadth, info0 = _run(10, comm=comm)
    _worst_swaps(monkeypatch, from_call=1)
    g_bad, info = _run(10, RefineSpec(rounds=3, tol=1e-12, swap=0.4), comm=comm)
    rec = info["refine"]
    assert rec["rejected"] == 1 and rec["swapped"] == [], rec
    from impurityModel.ed.gf_recurrence import _refine_line

    assert "round 1 rejected" in _refine_line(rec)  # the -v line says why nothing was swapped
    np.testing.assert_allclose(g_bad, g_breadth, atol=1e-10)
    assert _gather(info["proxy"].retained_keys(), comm) == _gather(info0["proxy"].retained_keys(), comm)

    monkeypatch.undo()
    g_one, info_one = _run(10, RefineSpec(rounds=1, tol=1e-12, swap=0.4), comm=comm)
    _worst_swaps(monkeypatch, from_call=2)
    g_later, info_later = _run(10, RefineSpec(rounds=3, tol=1e-12, swap=0.4), comm=comm)
    rec = info_later["refine"]
    assert rec["rejected"] == 2 and len(rec["swapped"]) == 1, rec
    np.testing.assert_allclose(g_later, g_one, atol=1e-10)
    assert _gather(info_later["proxy"].retained_keys(), comm) == _gather(info_one["proxy"].retained_keys(), comm)


@pytest.mark.parametrize("comm", COMMS)
def test_without_the_acceptance_test_the_same_bad_swap_is_kept(monkeypatch, comm):
    _worst_swaps(monkeypatch, from_call=1)
    g_breadth, _ = _run(10, comm=comm)
    _g, info = _run(10, RefineSpec(rounds=1, tol=1e-12, swap=0.4, verify=False), comm=comm)
    assert "rejected" not in info["refine"] and sum(info["refine"]["swapped"]) >= 1
