"""``GS_NUM_WANTED`` reaches the cap sizing, and changes the answer where it mattered.

``memory_estimate`` has accepted a ``gs_num_wanted`` argument all along, and ``log_memory_budget``
has warned when it was missing -- but **no caller ever passed it**. That dead plumbing is what let
the SrMnO3 double-counting search approve ``truncation_threshold=119,555,328``: sized with the
default assumption of ``2 * block_width`` the model predicted 2.51 GiB/rank against 5.0 GiB
available, where the manifold the run actually reached predicts 8.43 GiB. See
``doc/plans/dc_smo_memory.md``.
"""

from pathlib import Path

import pytest

from impurityModel.ed import config, dc_criteria, memory_estimate


@pytest.fixture(autouse=True)
def _no_inherited_knob(monkeypatch):
    """An env-backed knob makes its own tests non-hermetic; clear it."""
    monkeypatch.delenv("GS_NUM_WANTED", raising=False)


def _gs_cap(num_wanted):
    """The auto ground-state cap at production's rank count (256) and budget (0.5 x 5 GiB)."""
    return memory_estimate._largest_fitting(
        lambda n: memory_estimate.estimate_gs_peak_bytes(n, 58, 5, 256, 100, num_wanted=num_wanted) <= 0.5 * 5.0 * 2**30
    )


def test_unset_resolves_to_none_so_todays_behaviour_is_unchanged():
    assert memory_estimate.resolve_gs_num_wanted() is None
    assert config.GS_NUM_WANTED.default is None, "a default would silently resize every cap"


def test_a_set_value_is_resolved(monkeypatch):
    monkeypatch.setenv("GS_NUM_WANTED", "105")
    assert memory_estimate.resolve_gs_num_wanted() == 105


def test_every_driver_honours_it_through_the_one_resolver(monkeypatch):
    """What regressed before was the *argument*: the parameter existed and one driver never passed
    it. Every driver, the double-counting search included, now sizes through `resolve_cap_policy`,
    which reads the knob itself -- so setting it must change the cap they all get."""
    src = Path(dc_criteria.__file__).read_text()
    assert src.count("resolve_cap_policy(") == 2
    monkeypatch.setattr(memory_estimate, "available_bytes_per_rank", lambda c: 5 * 2**30)
    monkeypatch.setenv("GS_MAX_BLOCK_WIDTH", "5")
    unset, _ = memory_estimate.resolve_cap_policy(None, 58, log="never")
    monkeypatch.setenv("GS_NUM_WANTED", "105")
    supplied, _ = memory_estimate.resolve_cap_policy(None, 58, log="never")
    assert supplied.gs < unset.gs


def test_supplying_it_shrinks_the_suggested_cap():
    """The measured effect on the current tree, at production's rank count and budget.

    Note what this does **not** claim. An earlier version of this test asserted that supplying
    ``gs_num_wanted`` flips the verdict on ``truncation_threshold=119,555,328`` from fits to
    refused. That was true of the estimator *before* ``selection_bytes`` was added; on the current
    tree every path already refuses that cap (6.86 GiB unset, 12.78 GiB at 222, against 5.0 GiB
    available). What remains is that the knob shrinks the cap the search *chooses*.
    """
    unset = _gs_cap(num_wanted=None)
    supplied = _gs_cap(num_wanted=105)

    assert supplied < unset
    assert unset / supplied > 1.3, f"expected a materially smaller cap, got {unset / supplied:.2f}x"


def test_it_does_not_by_itself_make_the_cap_safe():
    """Honesty guard, and the reason the measured-RSS trip-wire is the primary mitigation.

    Even with the manifold supplied, the suggested cap stays far above the 949,834 determinants that
    actually exhausted memory, because ``estimate_gs_peak_bytes`` under-predicts the per-rank peak by
    372-1028x at 256 ranks (``doc/plans/dc_smo_memory.md``). Anything claiming this knob makes cap
    sizing correct should fail here.
    """
    supplied = _gs_cap(num_wanted=222)
    assert supplied > 10 * 949_834, (
        "if this ever fails the estimator has improved enough to revisit the claim that "
        "gs_num_wanted is a partial mitigation only"
    )
