"""Every growth recurrence of the sparse Green's function ends by its own rules, whatever the monitor says.

The monitor's relative-change tolerance can sit below the rounding floor of an unreorthogonalized recurrence
(NiO 2p+3d, 0 bath states: the real-axis change stalls at 1e-8..1e-7 against a tolerance of 1e-9). A recurrence
whose only way out is the monitor then never ends. That is what happened when the support was *closed*: ``H``
mapped the seeds' determinants into themselves, nothing could ever be admitted, the support never froze, and every
frozen-support stop (``exhausted``, the plateau, the array-kernel hand-off) stayed out of reach.

The stand-in monitor here never converges and has no plateau hook, so only the recurrence's own stops can end a unit.
The budget seam ``gf_recurrence._GROWTH_MAX_BLOCKS`` turns a missing stop into ``"max_iter"`` instead of a hang.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import gf_convergence, gf_recurrence
from impurityModel.ed.gf_solvers import block_Green_sparse
from impurityModel.ed.greens_function import calc_G
from impurityModel.test.gf.test_gf_truncation import (
    DELTA,
    OMEGA,
    _MODES,
    _dense_reference_on,
    _excited_basis,
    _redistribute_as_width1,
    _seeds,
    _siam_6,
)
from impurityModel.test.support.gf_switches import set_gf_switches

_NO_FULL_REORT_YET = pytest.mark.xfail(strict=True, reason="small frozen supports are not reorthogonalized yet")

#: Far above any block count the stop rules need on the 18-determinant fixture, far below a hang.
_BUDGET = 400


@pytest.fixture(autouse=True)
def _shipped_defaults(monkeypatch):
    for knob in ("GF_FROZEN_CSR", "APPLY_ROW_CHUNKS", "GF_SUPPORT_FREEZE"):
        monkeypatch.delenv(knob, raising=False)
    monkeypatch.setattr(gf_recurrence, "_GROWTH_MAX_BLOCKS", _BUDGET)


def _never_converging(monkeypatch):
    """Replace the GF monitor by one that is never satisfied (a rounding floor above the tolerance)."""

    def make(delta, slaterWeightMin, eval_meshes=None):
        def converged(alphas, betas, verbose=False, block_widths=None, **kwargs):
            return False

        return converged, [False], 1e-9, [None]

    monkeypatch.setattr(gf_recurrence, "_make_gf_convergence_monitor", make)


def _run(comm=None, **kwargs):
    basis = _excited_basis(np.inf, comm=comm)
    seeds_full = _seeds()
    seeds = _redistribute_as_width1(basis, seeds_full if comm is None or comm.rank == 0 else None, len(seeds_full))
    info = {}
    kwargs.setdefault("verbose", False)
    alphas, betas, r = block_Green_sparse(_siam_6(), seeds, basis, DELTA, cap_info=info, **kwargs)
    return calc_G(alphas, betas, r, OMEGA, 0.0, DELTA), info


def _assert_ended_by_a_rule(info):
    """The unit stopped on its own, in about one pass over the space: no ``max_iter``, and the growth and restart
    passes each well inside a few multiples of ``|P|`` vectors (width >= 1, so blocks <= vectors)."""
    assert info["status"] != "max_iter", info["status"]
    n_p = int(info["retained_size"])
    passes = info["passes"]
    bound = 2 * n_p + 4
    assert passes["growth_blocks"] <= bound, (passes, n_p)
    assert passes["restart_blocks"] <= bound, (passes, n_p)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("prune_tol", [None, 1e-3])
def test_a_closed_support_ends_without_the_monitor(mode, prune_tol, monkeypatch):
    """The fixture's 18 reachable determinants are a closed space: after the first steps nothing is left to admit."""
    set_gf_switches(monkeypatch, matrix_mode=mode)
    _never_converging(monkeypatch)
    g, info = _run(prune_tol=prune_tol)
    _assert_ended_by_a_rule(info)
    assert info["retained_size"] == 18
    assert info["proxy"].decisions.closed
    assert not info["cap_hit"], "a closed support is not cap-bound"


@_NO_FULL_REORT_YET
@pytest.mark.parametrize("mode", _MODES)
def test_a_small_closed_support_is_solved_exactly(mode, monkeypatch):
    """Below ``GF_FULL_REORT_MAX`` the frozen recurrence is reorthogonalized: it ends on a genuine invariant subspace
    and its continued fraction is the dense resolvent on the support, whatever the monitor says."""
    set_gf_switches(monkeypatch, matrix_mode=mode)
    _never_converging(monkeypatch)
    g, info = _run()
    _assert_ended_by_a_rule(info)
    assert info["status"] == "invariant_subspace", info["status"]
    np.testing.assert_allclose(g, _dense_reference_on(info["proxy"].retained_keys()), atol=1e-10)


@_NO_FULL_REORT_YET
@pytest.mark.parametrize("mode", _MODES)
def test_a_large_closed_support_ends_exhausted(mode, monkeypatch):
    """Above the bound (forced here: ``GF_FULL_REORT_MAX = 0``) the frozen recurrence stays unreorthogonalized and
    ends ``exhausted`` after ``|P|`` vectors."""
    set_gf_switches(monkeypatch, matrix_mode=mode)
    monkeypatch.setattr(gf_convergence, "GF_FULL_REORT_MAX", 0)
    _never_converging(monkeypatch)
    _g, info = _run()
    _assert_ended_by_a_rule(info)
    assert info["status"] in ("exhausted", "invariant_subspace"), info["status"]


@pytest.mark.mpi
@_NO_FULL_REORT_YET
@pytest.mark.parametrize("mode", _MODES)
def test_a_closed_support_ends_on_every_rank(mode, monkeypatch):
    """Multi-rank (an empty rank at -n 3): closure is decided on the replicated frontier count, so every rank stops
    on the same block and the result is the dense resolvent."""
    comm = MPI.COMM_WORLD
    set_gf_switches(monkeypatch, matrix_mode=mode)
    _never_converging(monkeypatch)
    g, info = _run(comm=comm)
    _assert_ended_by_a_rule(info)
    assert len(set(comm.allgather((info["status"], info["passes"]["growth_blocks"], info["n_vectors"])))) == 1
    np.testing.assert_allclose(g, _dense_reference_on(info["proxy"].retained_keys(), comm), atol=1e-9)


def test_a_closed_unit_is_not_re_solved_with_a_tighter_selection():
    """A closed unit holds every determinant ``H`` reaches: the acausal-block re-solve has nothing to select."""
    from impurityModel.ed.gf_diagnostics import UnitRecord
    from impurityModel.ed.selfenergy import _blocks_to_tighten

    def unit(block, **flags):
        flags = {"cap_hit": False, "memory_frozen": False, "closed": False} | flags
        return UnitRecord(0, block, "addition", (0,), (0.0,), (1.0,), 1.0, 18, 10, **flags)

    failing = {0: [], 1: [], 2: []}
    units = [unit(0), unit(1, closed=True), unit(2, cap_hit=True)]
    assert _blocks_to_tighten(failing, units) == [0]
