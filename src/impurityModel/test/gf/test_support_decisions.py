"""``gf_primitives.SupportDecisions``: the block-agnostic decisions of a capped support, on scalars and arrays.

:class:`SupportPolicy` holds one and forwards its state under the old names; the recurrence on a
discovered-key matrix uses the same object on dense frontier rows. Pinned: the forwarding is two-way,
the boundary ranking admits exactly the free slots, and a quiet window settles the support.
"""

import numpy as np

from impurityModel.ed.gf_primitives import SupportDecisions, SupportPolicy
from impurityModel.test.gf.test_gf_lanczos_pruning import _seed_basis


def test_the_policy_forwards_its_state_both_ways():
    policy = SupportPolicy(_seed_basis(), 10**9)
    d = policy.decisions
    assert policy.retained_size == d.global_count and not policy.cap_hit
    policy.cap_hit = True  # gf_admission assigns it
    assert d.cap_hit
    d.global_count += 5
    assert policy.retained_size == d.global_count
    policy.note_step_cut(0.25)
    assert d.pending_cut == 0.25


def test_the_boundary_ranking_fills_exactly_the_free_slots():
    d = SupportDecisions(None, cap=10, global_count=7)
    norms2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    above, boundary, n_fill = d.rank_boundary(norms2)
    admitted = int((norms2 > above).sum()) + n_fill
    assert admitted == 3
    d.freeze_at_cap(3)
    assert d.frozen and d.cap_hit and d.global_count == 10
    assert d.stop_phrase() == "GF basis frozen at the cap at"


def test_a_quiet_window_settles_the_support():
    d = SupportDecisions(None, cap=10**9, global_count=100)
    d.enable_support_freeze(4, 1.0, 1e-4)
    for _ in range(4):
        d.settle_step(1e-12, 100, 0, 0.0, 0.0, 1.0, 1e-12)
    assert d.support_frozen and d.frozen and not d.cap_hit
    assert d.settle_verdict()["step"] == 4
    assert d.stop_phrase() == "GF unit support settled at"
    d.unfreeze_support()
    assert not d.frozen and d.settle_verdict() is None
