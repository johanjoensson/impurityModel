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
from impurityModel.ed.memory_estimate import DEFAULT_MEMORY_SAFETY, available_bytes_per_rank


@pytest.fixture(autouse=True)
def _no_inherited_knob(monkeypatch):
    """A knob read from the environment makes its own tests non-hermetic; clear it."""
    monkeypatch.delenv("GS_MEMORY_BUDGET_SAFETY", raising=False)


def test_the_default_budget_is_the_shared_safety_fraction_of_available_ram():
    assert groundstate.expand_memory_budget(None) == int(DEFAULT_MEMORY_SAFETY * available_bytes_per_rank(None))


def test_the_safety_fraction_is_not_a_second_hard_coded_literal():
    """It must come from ``DEFAULT_MEMORY_SAFETY``, not a copy of 0.5 in ``groundstate``."""
    assert config.GS_MEMORY_BUDGET_SAFETY.default is None, "an explicit default would shadow the shared one"


@pytest.mark.parametrize("safety", ["0", "0.0"])
def test_zero_disables_the_guard(monkeypatch, safety):
    monkeypatch.setenv("GS_MEMORY_BUDGET_SAFETY", safety)
    assert groundstate.expand_memory_budget(None) is None


def test_a_set_fraction_scales_the_budget(monkeypatch):
    monkeypatch.setenv("GS_MEMORY_BUDGET_SAFETY", "0.25")
    assert groundstate.expand_memory_budget(None) == int(0.25 * available_bytes_per_rank(None))


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
