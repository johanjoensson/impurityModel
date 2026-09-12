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


def test_unset_resolves_to_none_so_todays_behaviour_is_unchanged():
    assert memory_estimate.resolve_gs_num_wanted() is None
    assert config.GS_NUM_WANTED.default is None, "a default would silently resize every cap"


def test_a_set_value_is_resolved(monkeypatch):
    monkeypatch.setenv("GS_NUM_WANTED", "105")
    assert memory_estimate.resolve_gs_num_wanted() == 105


def test_both_dc_sizing_sites_pass_it():
    """Source-level, deliberately: both sites sit inside collective functions that need a full DC
    search to reach, and what regressed before was the *argument*, not the behaviour -- the
    parameter existed and was simply never supplied."""
    src = Path(dc_criteria.__file__).read_text()
    assert src.count("gs_num_wanted=gs_num_wanted") == 4, "two sizing calls x two sites"
    assert src.count("gs_num_wanted = resolve_gs_num_wanted()") == 2


def test_supplying_it_shrinks_the_suggested_cap():
    """The measured effect on the current tree, at production's rank count and budget.

    Note what this does **not** claim. An earlier version of this test asserted that supplying
    ``gs_num_wanted`` flips the verdict on ``truncation_threshold=119,555,328`` from fits to
    refused. That was true of the estimator *before* ``selection_bytes`` was added; on the current
    tree every path already refuses that cap (6.86 GiB unset, 12.78 GiB at 222, against 5.0 GiB
    available). What remains is that the knob shrinks the cap the search *chooses*.
    """
    budget = 0.5 * 5.0 * 2**30
    common = dict(
        budget=budget,
        n_spin_orbitals=58,
        block_width=5,
        reort="none",
        n_parallel_units=1,
        nnz_per_state=100,
        krylov_dtype=None,
        method="lanczos",
        ranks=256,
    )
    unset = memory_estimate._suggest_for_budget(gs_num_wanted=None, **common)
    supplied = memory_estimate._suggest_for_budget(gs_num_wanted=105, **common)

    assert supplied < unset
    assert unset / supplied > 1.3, f"expected a materially smaller cap, got {unset / supplied:.2f}x"


def test_it_does_not_by_itself_make_the_cap_safe():
    """Honesty guard, and the reason the measured-RSS trip-wire is the primary mitigation.

    Even with the manifold supplied, the suggested cap stays far above the 949,834 determinants that
    actually exhausted memory, because ``estimate_gs_peak_bytes`` under-predicts the per-rank peak by
    372-1028x at 256 ranks (``doc/plans/dc_smo_memory.md``). Anything claiming this knob makes cap
    sizing correct should fail here.
    """
    common = dict(
        budget=0.5 * 5.0 * 2**30,
        n_spin_orbitals=58,
        block_width=5,
        reort="none",
        n_parallel_units=1,
        nnz_per_state=100,
        krylov_dtype=None,
        method="lanczos",
        ranks=256,
    )
    supplied = memory_estimate._suggest_for_budget(gs_num_wanted=222, **common)
    assert supplied > 10 * 949_834, (
        "if this ever fails the estimator has improved enough to revisit the claim that "
        "gs_num_wanted is a partial mitigation only"
    )
