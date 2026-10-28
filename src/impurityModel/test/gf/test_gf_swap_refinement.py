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
    assert info["refine"]["swapped"] == [0]
    np.testing.assert_allclose(g_ref, g_breadth, atol=1e-10)  # another matvec mode: equal to rounding


def test_no_refinement_without_a_cap_hit():
    _g, info = _run(np.inf, RefineSpec(rounds=3))
    assert "refine" not in info


@pytest.mark.parametrize("comm", COMMS)
def test_the_refined_support_is_the_exact_resolvent_of_PHP(comm):
    g_ref, info = _run(10, RefineSpec(rounds=3, tol=1e-12, swap=0.4), comm=comm)
    kept = _gather(info["proxy"].retained_keys(), comm)
    np.testing.assert_allclose(g_ref, _dense_G_on(kept, OMEGA + 1j * DELTA), atol=1e-9)
