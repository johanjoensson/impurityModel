"""``selfenergy._attribute_failures``: the text a user reads when the self-energy is acausal.

It is exercised against a faked self-energy verdict (``_sigma_collect``), so each branch is pinned without a
many-body solve: nothing failing gives no text, a failing block is named with its worst point, the nearest bath level
is named only when it couples to the block, capped units point at the cap, and nothing is re-solved.
"""

import numpy as np
import pytest

from impurityModel.ed import gf_diagnostics as gd
from impurityModel.ed import selfenergy
from impurityModel.ed.utils import Reporter

DELTA = 0.01
W = np.linspace(-1.0, 1.0, 201)
IW = 1j * np.pi * 0.01 * (2 * np.arange(10) + 1)
BLOCKS = [[0, 1], [2, 3, 4]]


def _unit(unit, block, cap_hit=False, memory_frozen=False):
    return gd.UnitRecord(
        unit=unit, block=block, side="removal", states=(0,), e_minus_e0=(0.0,), boltzmann=(1.0,), seed_weight=1.0,
        retained=1000, n_blocks=500, cap_hit=cap_hit, memory_frozen=memory_frozen,
    )  # fmt: skip


def _record(val=3e-3, ratio=7e-3, freq=100, g0=-0.5 + 0j, gi=-0.4 + 0j):
    return {
        "n_viol": 3, "n_total": 1000, "frac": 3e-3, "worst_val": val, "worst_freq_index": freq, "worst_orbital": 0,
        "worst_orbital_ratio": 0, "worst_ratio": ratio, "windows": [], "g0_inv_at_worst": g0, "ginv_at_worst": gi,
    }  # fmt: skip


class _Report:
    def __init__(self, units):
        self.units = units


class _SB:
    h0_solve = None
    total_impurity_orbitals = None
    sum_bath_states = None


@pytest.fixture
def fake_sigma(monkeypatch):
    state = {"violate": True, "record": _record(), "levels": (-0.6, -0.3)}

    def collect(mesh, gss, *, delta, estimator, solver_basis, cluster_label, blocks):
        if gss is None:
            return None, None, {}
        sigma = [np.zeros((len(mesh), len(b), len(b)), dtype=complex) for b in blocks]
        if mesh is W and state["violate"]:
            return sigma, None, {0: state["record"]}
        return sigma, None, {}

    monkeypatch.setattr(selfenergy, "_sigma_collect", collect)
    # level -0.6 couples to every impurity orbital, the other level to none (a foreign level)
    v_full = np.zeros((2, 5))
    v_full[0, :] = 0.2
    monkeypatch.setattr(
        selfenergy, "get_hcorr_v_hbath", lambda *a, **k: (None, v_full, None, np.diag(np.array(state["levels"])))
    )
    return state


def _call(units):
    return selfenergy._attribute_failures(
        gf_report=_Report(units), family_matsubara=["M"], family_realaxis=["R"], iw=IW, w=W, delta=DELTA,
        estimator=None, sb=_SB(), blocks=BLOCKS, comm=None, report=Reporter(0, 0), cluster_label="x", tol=1e-3,
    )  # fmt: skip


def test_a_causal_self_energy_gives_no_text(fake_sigma):
    fake_sigma["violate"] = False
    assert _call([_unit(0, 0)]) == ""


def test_a_failing_block_is_named_with_its_worst_point_and_the_cap_when_units_froze(fake_sigma):
    text = _call([_unit(0, 0, cap_hit=True), _unit(1, 0), _unit(2, 1)])
    assert "Real frequency self-energy, block [0, 1]" in text and "+3.000e-03" in text
    assert "1 of 2 units of this block froze at the determinant cap" in text and "gf_refine_rounds" in text
    assert "block [2, 3, 4]" not in text


def test_without_frozen_units_the_advice_is_the_pruning_tolerance(fake_sigma):
    text = _call([_unit(0, 0)])
    assert (
        "lower gf_prune_tol (gf_pt2_tol, when the second-order rule is on) or slater_min" in text
        and "Likely cause" in text
        and "froze at the determinant cap" not in text
    )


def test_the_coupled_bath_level_is_named_and_a_foreign_one_is_not(fake_sigma):
    fake_sigma["record"] = _record(freq=60)  # w = -0.4, closest to the coupled level -0.6 (the foreign -0.3 is nearer)
    assert "nearest coupled bath level -0.6" in _call([_unit(0, 0)])
    assert "-0.3 " not in _call([_unit(0, 0)])
