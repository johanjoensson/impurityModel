"""Swap refinement of a capped unit (``gf_refinement.refine_frozen_unit``) on the 6-orbital SIAM, against the dense
resolvent of the whole reachable space."""

import numpy as np
import pytest

from impurityModel.ed.gf_refinement import RefineSpec
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


def test_the_real_axis_scoring_shifts_are_broadened_and_the_matsubara_ones_are_not():
    from impurityModel.ed.gf_refinement import pick_shifts

    real = np.linspace(-1.1, 1.1, 9) + 1j * DELTA
    matsubara = 1j * np.linspace(0.05, 2.0, 6)

    def g_of(zs):  # one sharp peak on the real axis
        return np.array([[[1.0 / (z - 0.3 + 1e-3j)]] for z in np.asarray(zs)])

    plain, hold = pick_shifts(g_of, [real, matsubara])
    wide, hold_wide = pick_shifts(g_of, [real, matsubara], broaden=4.0)
    assert len(plain) == len(wide)
    on_matsubara = set(1j * np.linspace(0.05, 2.0, 6))
    for p, w in zip(plain, wide):
        assert w.real == p.real
        assert w.imag == pytest.approx(p.imag if p in on_matsubara else 4.0 * p.imag)
    assert any(w.imag == pytest.approx(4 * DELTA) for w in wide), "a real-axis shift must be broadened"
    assert hold == hold_wide, "the holdout is judged on the mesh"
    assert {w for w in wide if w in on_matsubara} == {p for p in plain if p in on_matsubara}


@pytest.mark.parametrize("comm", COMMS)
def test_the_refinement_scores_at_the_spec_broadening_and_the_spec_tolerances(monkeypatch, comm):
    calls = _spy_scores(monkeypatch)
    _run(10, RefineSpec(rounds=1, tol=1e-12, swap=0.4, rtol=1e-3, maxit=7, broaden=4.0), comm=comm)
    zs, kw = calls[0]
    assert (kw["rtol"], kw["maxit"]) == (1e-3, 7)
    matsubara = set(_mesh()[1])
    real_axis = [z for z in zs if z not in matsubara]
    assert real_axis and all(z.imag == pytest.approx(4 * DELTA) for z in real_axis)
    assert [z for z in zs if z in matsubara], "premise: the Matsubara points are scored too, unbroadened"


@pytest.mark.parametrize("comm", COMMS)
def test_a_spec_that_says_nothing_reads_the_module_constants(monkeypatch, comm):
    from impurityModel.ed import gf_refinement

    monkeypatch.setattr(gf_refinement, "REFINE_RTOL", 1e-4)
    monkeypatch.setattr(gf_refinement, "REFINE_MAXIT", 11)
    monkeypatch.setattr(gf_refinement, "REFINE_BROADEN", 3.0)
    calls = _spy_scores(monkeypatch)
    _run(10, RefineSpec(rounds=1, tol=1e-12, swap=0.4), comm=comm)
    zs, kw = calls[0]
    assert (kw["rtol"], kw["maxit"]) == (1e-4, 11)
    assert any(z.imag == pytest.approx(3 * DELTA) for z in zs)
    assert not any(z.imag == pytest.approx(3 * DELTA) for z in zs if z in set(_mesh()[1]))


@pytest.mark.parametrize("comm", COMMS)
def test_the_record_keeps_the_worst_residual_of_every_pass(monkeypatch, comm):
    """The worst over the shifts (not the first, not the last) of each pass, one entry per pass."""
    _spy_scores(monkeypatch, residuals=[1e-9, 3e-2, 1e-4])
    _g, info = _run(10, RefineSpec(rounds=2, tol=1e-12, swap=0.4), comm=comm)
    rec = info["refine"]
    assert rec["residual"] == [3e-2] * len(rec["gain"])


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
