"""The measured-RSS trip-wire is actually wired into both ``expand`` call sites.

The guard itself (what ``expand`` does once the budget is reached) is covered by
``test_cipsi_memory_budget_guard.py``. What is tested here is the thing that was missing for a
whole investigation: the budget being *derived and passed at all*. It shipped implemented, tested
and unwired, so an uncapped production expansion grew until the kernel killed the rank --
see ``doc/plans/dc_smo_memory.md``.
"""

from pathlib import Path

import pytest

from impurityModel.ed import config, groundstate
from impurityModel.ed.cipsi_solver import _memory_growth_bound
from impurityModel.ed.memory_estimate import DEFAULT_MEMORY_SAFETY


@pytest.fixture(autouse=True)
def _no_inherited_knob(monkeypatch):
    """A knob read from the environment makes its own tests non-hermetic; clear it."""
    monkeypatch.delenv("GS_MEMORY_BUDGET_SAFETY", raising=False)


def test_the_default_budget_is_the_shared_safety_fraction_of_the_ranks_whole_share(monkeypatch):
    """`safety * (available + resident)`, not `safety * available`. See `absolute_rss_budget`:
    `available` is *free* memory, so the fraction of it alone is an increment allowance, and both
    guards it feeds measure absolute RSS."""
    available, resident = _crashed_run_numbers(monkeypatch)
    assert groundstate.expand_memory_budget(None) == int(DEFAULT_MEMORY_SAFETY * (available + resident))


def test_the_safety_fraction_is_not_a_second_hard_coded_literal():
    """It must come from ``DEFAULT_MEMORY_SAFETY``, not a copy of 0.5 in ``groundstate``."""
    assert config.GS_MEMORY_BUDGET_SAFETY.default is None, "an explicit default would shadow the shared one"


@pytest.mark.parametrize("safety", ["0", "0.0"])
def test_zero_disables_the_guard(monkeypatch, safety):
    monkeypatch.setenv("GS_MEMORY_BUDGET_SAFETY", safety)
    assert groundstate.expand_memory_budget(None) is None


def test_a_set_fraction_scales_the_budget(monkeypatch):
    available, resident = _crashed_run_numbers(monkeypatch)
    monkeypatch.setenv("GS_MEMORY_BUDGET_SAFETY", "0.25")
    assert groundstate.expand_memory_budget(None) == int(0.25 * (available + resident))


def test_a_negative_fraction_is_clamped_not_honoured(monkeypatch):
    """``minimum=0.0`` on the knob must stop a typo becoming a negative budget, which would trip
    the guard on cycle 0 and cap every expansion at its seed basis."""
    monkeypatch.setenv("GS_MEMORY_BUDGET_SAFETY", "-1")
    assert groundstate.expand_memory_budget(None) is None


def test_both_expand_call_sites_pass_the_budget():
    """Source-level, deliberately: the two call sites are inside collective functions that need a
    full solve to reach, and what regressed before was the *argument*, not the behaviour."""
    # `read_text`, not a bare `open(...).read()`: the unclosed handle raises
    # PytestUnraisableExceptionWarning, and pytest.ini turns warnings into errors.
    src = Path(groundstate.__file__).read_text()
    n_expand = src.count("solver.expand(")
    n_wired = src.count("memory_budget_bytes=expand_memory_budget(comm)")
    assert n_expand == n_wired == 2, f"{n_expand} expand call sites, {n_wired} pass a budget"
    # ...and both say whether the cap may be tightened: a cap the user set is final, so its guard
    # only warns. Omitting the argument would silently default to "tighten" on that path.
    n_policy = src.count("memory_policy=_memory_policy(truncation_threshold)")
    assert n_policy == 2, f"{n_policy} of 2 expand call sites pass the cap's memory policy"


# ---------------------------------------------------------------------------------------
# Regression: the budget must not be swallowed by the process's own resident set
# (doc/plans/dc_smo_memory.md, round 9 -- the SrMnO3 cubic gap-DC collapse)
# ---------------------------------------------------------------------------------------

MiB = 2**20


def _crashed_run_numbers(monkeypatch, available=4900 * MiB, resident=2500 * MiB):
    """The SrMnO3 crash's own readings: 4.9 GiB/rank still free, 2.5 GiB already resident.

    `groundstate` binds both names with `from ... import`, so they must be patched *there*
    rather than in `memory_estimate` (the same trap `test_greens_function_and_basis_split`
    documents).

    It must be `resident_bytes_per_rank`, not `current_rss_bytes`: the budget reads the former,
    so patching the latter silently does nothing and the test falls through to the *host's* real
    RSS. Measured while writing this: that left it passing by 46 MiB of pytest-process footprint,
    i.e. green for the wrong reason and one allocation away from flaking
    (cf. the flaky memory-budget test).
    """
    monkeypatch.setattr(groundstate, "available_bytes_per_rank", lambda comm: available)
    monkeypatch.setattr(groundstate, "resident_bytes_per_rank", lambda comm: resident)
    return available, resident


def test_the_budget_exceeds_a_resident_set_that_already_fills_free_memory(monkeypatch):
    """`available_bytes_per_rank` is `MemAvailable / ranks_on_node` -- memory that is *free*,
    already net of what this process holds. `safety * available` is therefore an **increment**
    allowance, but `_memory_growth_bound` compares it against **absolute** RSS. Once the process
    is resident above that fraction the guard refuses all growth forever.

    On the crashed run that is 0.5 * 4.9 = 2.45 GiB against a 2.5 GiB resident set, and every
    sector after the first was pinned at its seed basis (10-252 determinants) while its own
    selection round cost 4-68 KiB.
    """
    _available, resident = _crashed_run_numbers(monkeypatch)
    budget = groundstate.expand_memory_budget(None)
    assert budget > resident, (
        f"budget {budget / MiB:.0f} MiB does not even cover the {resident / MiB:.0f} MiB already "
        "resident, so the look-ahead bound has negative headroom before any work is done"
    )


def test_a_kilobyte_selection_round_can_still_grow_a_seed_basis(monkeypatch):
    """The defect as the crash log shows it, end to end.

    `impurityModel-Mn-dc.out:509`: a 120-determinant basis, p=86, whose selection round peaked
    **8 KiB** above its resident set, was told it could afford **0** of 4,032 candidates. Nothing
    about 8 KiB is unaffordable; the bound was comparing incommensurable quantities.
    """
    _available, resident = _crashed_run_numbers(monkeypatch)
    budget = groundstate.expand_memory_budget(None)

    # The crashed round's own shape: 120 determinants, 86 references, next request 96.
    affordable = _memory_growth_bound(budget, basis_size=120, p_now=86, p_next=96)
    cap, reason = affordable(8 * 1024, resident)
    assert reason == "transient", "the budget alone decided this, without consulting the round"
    assert cap > 0, "a selection round costing 8 KiB was refused all growth on a 120-determinant basis"


def test_the_guard_still_refuses_growth_when_memory_is_genuinely_gone(monkeypatch):
    """The safety property the fix must not trade away: when the resident set really has consumed
    the rank's share, headroom is zero and the expansion stops."""
    _available, resident = _crashed_run_numbers(monkeypatch, available=64 * MiB, resident=8192 * MiB)
    budget = groundstate.expand_memory_budget(None)
    affordable = _memory_growth_bound(budget, basis_size=120, p_now=86, p_next=96)
    assert affordable(8 * 1024, resident) == (0, "budget")


# The repo's `_parse_bool` treats only these as false ("only explicit falsehoods are false"),
# so these are the spellings a user can actually roll back with.
@pytest.mark.parametrize("off", ["0", "false", "False"])
def test_the_rollback_knob_restores_the_free_memory_arithmetic(monkeypatch, off):
    """`GS_MEMORY_BUDGET_INCLUDE_RESIDENT=0` is the way back to the pre-2026-09 budget.

    This guard's job is preventing an uncatchable OOM kill, so changing the arithmetic it runs on
    needs an escape hatch that does not also switch the guard off entirely
    (`GS_MEMORY_BUDGET_SAFETY=0` is the separate, blunter lever).
    """
    available, _resident = _crashed_run_numbers(monkeypatch)
    monkeypatch.setenv("GS_MEMORY_BUDGET_INCLUDE_RESIDENT", off)
    assert groundstate.expand_memory_budget(None) == int(DEFAULT_MEMORY_SAFETY * available)


def test_the_rollback_knob_is_registered():
    """An unregistered knob reads from the environment but never appears in `config.dump`, so a
    run cannot be reproduced from its own log."""
    assert "GS_MEMORY_BUDGET_INCLUDE_RESIDENT" in config.KNOBS


class _StopAtRefine(Exception):
    pass


def _refine_start_cap(monkeypatch, walked_cap, policy):
    """The cap `solve_ground_state`'s refinement expand starts from, after a walk that left its
    basis at `walked_cap` and attached `policy`."""
    from impurityModel.ed import cipsi_solver
    from impurityModel.ed.manybody_basis import Basis

    basis = Basis(
        {0: [[0, 1]]}, ({0: [[2, 3]]}, {0: [[]]}), nominal_impurity_occ={0: 1}, truncation_threshold=walked_cap
    )
    basis.cap_policy = policy
    monkeypatch.setattr(groundstate, "walk_to_ground_state_sector", lambda *a, **k: basis)
    seen = {}

    def fake_expand(self, *args, **kwargs):
        seen["cap"] = self.basis.truncation_threshold
        raise _StopAtRefine

    monkeypatch.setattr(cipsi_solver.CIPSISolver, "expand", fake_expand)
    with pytest.raises(_StopAtRefine):
        groundstate.solve_ground_state(
            None, {0: [[0, 1]]}, ({0: [[2, 3]]}, {0: [[]]}), {0: 1}, tau=0.01, slaterWeightMin=0
        )
    return seen["cap"]


def test_the_refinement_does_not_inherit_a_cap_the_walk_lowered(monkeypatch):
    """The walk keeps several sectors alive and may be held lower by the memory guard; the
    refinement solves one sector on one basis and starts from the resolved (auto) cap, with its own
    guard measuring afresh."""
    from impurityModel.ed.memory_estimate import CapPolicy

    assert _refine_start_cap(monkeypatch, 120, CapPolicy(gs=50_000, gf=50_000, from_memory=True)) == 50_000


def test_the_refinement_reset_is_for_memory_derived_caps_only(monkeypatch):
    """A cap the user set is never lowered by the guard, so there is nothing to undo: whatever the
    basis carries is left alone (a reset would be the one way this code could *change* a user
    cap)."""
    from impurityModel.ed.memory_estimate import CapPolicy

    assert _refine_start_cap(monkeypatch, 120, CapPolicy(gs=2000, gf=2000, from_memory=False)) == 120


def test_a_walk_the_memory_guard_held_is_not_reported_memory_bound_after_the_refinement():
    """The refinement restarts from the resolved cap, so the walk's memory verdict no longer
    describes the ground state: carrying it would raise a false gs_memory warning and a spurious
    double-counting refusal. The cap verdict and the walk's own flag are kept."""
    from types import SimpleNamespace

    walk = {"cap_hit": True, "memory_bound": True, "retained": 120}
    basis = SimpleNamespace(occupation_search_truncation=walk)
    report = groundstate.refined_truncation_report(SimpleNamespace(truncation_report=None), basis)
    assert report["memory_bound"] is False and report["walk_memory_bound"] is True
    assert report["cap_hit"] is True
    refined = {"cap_hit": True, "memory_bound": True}
    assert groundstate.refined_truncation_report(SimpleNamespace(truncation_report=refined), basis) is refined
