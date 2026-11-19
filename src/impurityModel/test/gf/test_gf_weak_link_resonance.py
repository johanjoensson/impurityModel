"""Step-7 fixture of the GF redesign: the amplitude rule loses determinants that carry a pole of ``G_bb``.

``build_two_channel_siam``: a half-filled spin-up impurity beside an empty spin-down one, coupled by ``U n_a n_b``
alone. The ground state is exact and its reference causal; the determinants that carry the ``b``-chain poles
(one ``b``-chain site, spread over the up configurations) are not seeds, and their individual weights are small
next to the seeds', so a pinned amplitude threshold drops them and the real-axis Green's function loses the pole.
This is the failure the resonance score of the redesign (plan steps 8-11) has to fix at the same ``|P|``; the test
pins that it exists, so the fix cannot be tested against a model where the amplitude rule already succeeds.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.test.support.aim_fixtures import build_two_channel_siam
from impurityModel.test.support.basis_size_harness import Mesh, run_cell


@pytest.fixture(scope="module")
def model():
    aim = build_two_channel_siam(5, U=0.5, chain_b=([0.7, 0.6, 0.8, 0.8, 0.8], [0.3, 0.25, 0.25, 0.25, 0.25]))
    omega = np.linspace(-1.5, 2.0, 351)
    mats = 1j * np.pi * 0.05 * (2 * np.arange(8) + 1)
    return aim, Mesh(aim, omega, mats, 0.01, n_real=30, n_mats=4)


def test_the_ground_state_is_exact_and_the_closure_is_larger_than_the_seeds(model):
    aim, mesh = model
    cell = run_cell(aim, mesh, "lanczos-cap")
    assert cell["acausal_real"] < 1e-8 and cell["dSigma"] < 1e-8
    assert cell["size"] > 3 * cell["seed_size"]


def test_a_coarse_threshold_loses_the_b_chain_pole_and_a_fine_one_does_not(model):
    aim, mesh = model
    closure = run_cell(aim, mesh, "lanczos-cap")["size"]
    fine = run_cell(aim, mesh, "lanczos-pruned", eta=0.03)
    coarse = run_cell(aim, mesh, "lanczos-pruned", eta=0.1)
    assert fine["size"] == closure and fine["dSigma_real"] < 1e-6
    assert coarse["size"] < closure and coarse["dSigma_real"] > 0.1


def _refine_spec(aim, mesh, rounds, swap, grow_tol=None):
    """The shift tables of the fixture's one block, as ``selfenergy`` builds them from ``G0^-1``."""
    from impurityModel.ed.gf_refinement import RefineSpec, g0_check_shifts, g0_zero_shifts
    from impurityModel.test.support.aim_fixtures import free_G_inverse

    def g0_inv(zs):
        return free_G_inverse(aim, np.asarray(zs, dtype=complex))

    zs, weights = g0_zero_shifts(g0_inv, mesh.omega, mesh.matsubara, mesh.delta)
    checks = g0_check_shifts(g0_inv, mesh.omega, mesh.matsubara, mesh.delta)
    return RefineSpec(
        rounds=rounds, swap=swap, grow_tol=grow_tol, tables=(tuple(zip(zs, weights)),), check_tables=(checks,)
    )


_RECORDS = []


def _refined(aim, mesh, monkeypatch, eta, rounds, swap, scores=None, cap=10**5, comm=None, grow_tol=None):
    import impurityModel.ed.gf_refinement as refinement

    _RECORDS.clear()
    refine_unit = refinement.refine_frozen_unit

    def recording(*args, **kwargs):
        outcome = refine_unit(*args, **kwargs)
        _RECORDS.append((outcome.record, comm.size if comm is not None else 1, args[0].comm))
        return outcome

    monkeypatch.setattr(refinement, "refine_frozen_unit", recording)
    import impurityModel.test.support.basis_size_harness as harness

    knobs = harness._knobs

    def with_refinement(*args, **kwargs):
        spec = knobs(*args, **kwargs)
        spec["options"] = {
            **spec["options"],
            "gf_refine": _refine_spec(aim, mesh, rounds=rounds, swap=swap, grow_tol=grow_tol),
        }
        return spec

    monkeypatch.setattr(harness, "_knobs", with_refinement)
    if scores is not None:
        score = refinement.importance_scores

        def altered(*args, **kwargs):
            out = score(*args, **kwargs)
            out.s_cand = scores(out.s_cand)
            return out

        monkeypatch.setattr(refinement, "importance_scores", altered)
    return run_cell(aim, mesh, "lanczos-pruned", eta=eta, cap=cap, comm=comm)


def test_a_pruned_unit_with_room_admits_the_rows_the_resonance_scores_and_recovers_the_pole(model, monkeypatch):
    """Grow mode of the swap refinement: a pruned unit below its cap scores the discovered rows at the shifts of
    ``G0^-1`` and admits the best ones, retiring none. Given room for the whole closure it recovers the pole."""
    aim, mesh = model
    assert run_cell(aim, mesh, "lanczos-pruned", eta=0.1, cap=10**5)["dSigma_real"] > 0.1
    refined = _refined(aim, mesh, monkeypatch, 0.1, rounds=8, swap=0.5)
    assert refined["dSigma_real"] < 1e-6 and refined["acausal_real"] < 1e-8


def test_the_grow_budget_is_the_specs_and_a_loose_one_stops_before_the_resonance(model, monkeypatch):
    """``RefineSpec.grow_tol`` is the mass the cut may leave out: a budget above the whole frontier admits nothing
    (the pole stays lost, reported settled), the default admits the resonance."""
    aim, mesh = model
    loose = _refined(aim, mesh, monkeypatch, 0.1, rounds=8, swap=0.5, grow_tol=1e3)
    assert loose["dSigma_real"] > 0.1
    assert sum(_RECORDS[-1][0]["swapped"]) == 0
    tight = _refined(aim, mesh, monkeypatch, 0.1, rounds=8, swap=0.5)
    assert tight["dSigma_real"] < 1e-6


def test_the_resonance_score_ranks_the_rows_it_admits(model, monkeypatch):
    """Room for 12 of the 40 missing rows, so the order matters: the score halves the error that the amplitude rule
    leaves, and the same admissions in the inverse order do not."""
    aim, mesh = model
    plain = run_cell(aim, mesh, "lanczos-pruned", eta=0.1, cap=10**5)["dSigma_real"]
    scored = _refined(aim, mesh, monkeypatch, 0.1, rounds=8, swap=0.1, cap=95 + 12)["dSigma_real"]
    assert scored < 0.6 * plain
    inverse = _refined(
        aim,
        mesh,
        monkeypatch,
        0.1,
        rounds=8,
        swap=0.1,
        cap=95 + 12,
        scores=lambda c: np.where(c > 0, 1.0 / np.maximum(c, 1e-300), 0.0),
    )["dSigma_real"]
    assert inverse > 0.9 * plain


def test_grow_mode_stops_when_the_next_admission_would_not_fit_the_memory_budget(model, monkeypatch):
    """The growth is checked against the memory budget before every admission: with the budget exhausted by the
    resident set after the unit's own growth, the refinement admits nothing and the pole stays lost."""
    import impurityModel.ed.gf_recurrence as recurrence

    aim, mesh = model
    grown = _refined(aim, mesh, monkeypatch, 0.1, rounds=8, swap=0.5)["dSigma_real"]
    assert grown < 1e-6
    make_basis = aim.basis

    def budgeted(*args, **kwargs):
        basis = make_basis(*args, **kwargs)
        basis.gf_memory_budget = 10**12
        return basis

    monkeypatch.setattr(aim, "basis", budgeted, raising=False)
    monkeypatch.setattr(recurrence, "current_rss_bytes", lambda: 2 * 10**12)
    starved = _refined(aim, mesh, monkeypatch, 0.1, rounds=8, swap=0.5)["dSigma_real"]
    assert starved > 0.1
    # ... and it is the memory guard that stopped it, not a refinement that never ran
    assert _RECORDS and all(r["stopped"] == "the next admission would exceed the memory budget" for r, _, _ in _RECORDS)


def test_a_coarser_cut_still_recovers_the_pole_when_the_frontier_regrows(model, monkeypatch):
    """At eta = 0.2 the frontier grows as rows are admitted; a fraction of |P| per round (the old rule) stalls at 95
    rows, the cumulative cut reaches the whole closure."""
    aim, mesh = model
    refined = _refined(aim, mesh, monkeypatch, 0.2, rounds=8, swap=0.1)
    assert refined["dSigma_real"] < 1e-6


class _Key(int):
    def to_bytearray(self):
        return int(self).to_bytes(8, "big")


class _StubMatrix:
    """Just what :func:`_select_grow` reads: ``n`` discovered rows, none of them in P."""

    comm = None
    n_k_local = 0

    def __init__(self, n):
        self.n_k_local = n

    def alive(self):
        return np.zeros(0, dtype=np.int8)

    def p_to_k_slot(self):
        return np.zeros(0, dtype=np.int64)

    def keys_of(self, slots):
        return [_Key(int(s)) for s in slots]


def _grow(scores, budget, k_cap, fits=None):
    from types import SimpleNamespace

    from impurityModel.ed.gf_refinement import _select_grow

    return _select_grow(_StubMatrix(len(scores)), SimpleNamespace(s_cand=np.asarray(scores)), budget, k_cap, fits)


def test_the_cumulative_cut_admits_the_heavy_rows_and_not_the_sea_of_tiny_ones():
    scores = np.concatenate([np.full(10, 1.0), np.full(10**5, 1e-9)])
    admit, remaining, stopped = _grow(scores, budget=1e-2, k_cap=10**6)
    assert sorted(admit) == list(range(10)) and stopped is None
    assert remaining == pytest.approx(10.0 + 1e-4)
    admit, _, _ = _grow(scores, budget=1e-5, k_cap=10**6)  # the tail now carries more than the budget: it goes in
    assert len(admit) == len(scores)


def test_a_binding_room_reports_the_mass_it_leaves_out_and_the_memory_guard_shrinks_instead_of_refusing():
    scores = np.concatenate([np.full(10, 1.0), np.full(1000, 1e-6)])
    admit, remaining, _ = _grow(scores, budget=1e-4, k_cap=500)
    assert len(admit) == 500 and remaining > 1e-4  # stopped by the room, and the mass left is reported
    admit, _, stopped = _grow(scores, budget=1e-4, k_cap=500, fits=lambda k: k <= 5)
    assert len(admit) == 3 and stopped is None  # 500 halved until it fits: 250, 125, 62, 31, 15, 7, 3
    assert set(admit) <= set(range(10))  # and the best rows are the ones kept


def test_a_spec_with_no_rounds_reports_the_peak_shifts_and_does_not_refine(model, monkeypatch):
    """The shift tables ride on every lanczos unit (step 3): with ``rounds=0`` the unit still gets its peak-shift
    line, and the coarse support is left as it is (the pole stays lost)."""
    aim, mesh = model
    import impurityModel.ed.gf_recurrence as recurrence

    checks = []
    check = recurrence.peak_shift_check
    monkeypatch.setattr(recurrence, "peak_shift_check", lambda *a, **k: checks.append(check(*a, **k)) or checks[-1])
    plain = run_cell(aim, mesh, "lanczos-pruned", eta=0.1, cap=10**5)
    assert not checks  # no tables, no line
    cell = _refined(aim, mesh, monkeypatch, 0.1, rounds=0, swap=0.5)
    assert cell["unit_sizes"] == plain["unit_sizes"] and cell["dSigma_real"] > 0.1
    assert checks and all(c["n_shifts"] > 0 for c in checks)


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="a distributed matrix needs more than one rank")
def test_grow_mode_on_a_distributed_matrix_admits_the_same_rows_as_one_rank(model, monkeypatch):
    """The discovered matrix, its scores and the cumulative cut are spread over the ranks: the unit must end with the
    same size and the same self-energy as on one rank."""
    aim, mesh = model
    make_basis = aim.basis

    def unified(*args, **kwargs):  # never split the communicator: the unit's matrix is spread over every rank
        basis = make_basis(*args, **kwargs)
        basis.split_threshold = 0
        return basis

    monkeypatch.setattr(aim, "basis", unified, raising=False)
    for cap in (10**5, 95 + 12):  # room for the whole closure, then a room that binds
        serial = _refined(aim, mesh, monkeypatch, 0.1, rounds=8, swap=0.5, cap=cap)
        spread = _refined(aim, mesh, monkeypatch, 0.1, rounds=8, swap=0.5, cap=cap, comm=MPI.COMM_WORLD)
        assert _RECORDS and all(c.size == MPI.COMM_WORLD.size for _, _, c in _RECORDS)  # really distributed
        assert spread["unit_sizes"] == serial["unit_sizes"]
        assert spread["dSigma_real"] == pytest.approx(serial["dSigma_real"], abs=1e-8)
        if cap == 10**5:
            assert serial["dSigma_real"] < 1e-6
