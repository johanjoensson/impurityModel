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


def RefineSpec(**kw):
    kw.setdefault("shifts", SHIFTS)
    return _RefineSpec(**kw)


def _run(cap, refine=None, comm=None):
    basis = _seed_basis(cap=cap, comm=comm)
    full = _seeds()
    seeds = _redistribute_as_width1(basis, full if comm is None or comm.rank == 0 else None, len(full))
    info = {}
    alphas, betas, r = block_Green_sparse(
        _siam_6(), seeds, basis, DELTA, verbose=False, cap_info=info, eval_meshes=_mesh(), refine=refine
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
    assert info["refine"]["declined"] == "no scoring shifts" and info["refine"]["swapped"] == []


@pytest.mark.parametrize("comm", COMMS)
def test_the_recurrence_runs_until_the_scoring_residual_is_met(monkeypatch, comm):
    """The stop tolerance is the recurrence's: a tight one costs blocks, a loose one stops with the Green's function."""
    blocks = {}
    for tol in (1e-1, 1e-12):
        _g, info = _run(10, RefineSpec(rounds=1, tol=1e9, swap=0.4, stop_rtol=tol), comm=comm)
        blocks[tol] = info["refine"]["blocks"][0]
    assert blocks[1e-12] >= blocks[1e-1]


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
