"""Consumers of the final-state cap flag: a transient report neither refuses a member nor hides a bound one."""

from impurityModel.ed.gf_diagnostics import check_ground_state_truncation
from impurityModel.ed.gs_ensemble import member_refusal, worst_truncation

TRANSIENT = {"cap_hit": True, "cap_bound": False, "memory_bound": False, "residual_pt2": 0.0}
BOUND = {"cap_hit": True, "cap_bound": True, "memory_bound": False, "residual_pt2": 1e-3}
MEMORY = {"cap_hit": True, "cap_bound": True, "memory_bound": True}


def test_worst_truncation_prefers_memory_then_bound_over_transient():
    assert worst_truncation([TRANSIENT, BOUND]) is BOUND
    assert worst_truncation([TRANSIENT, BOUND, MEMORY]) is MEMORY
    assert worst_truncation([TRANSIENT]) is TRANSIENT
    assert worst_truncation([None, None]) is None


class _Member:
    total = 7


def test_member_refusal_ignores_a_transient_report():
    assert member_refusal(_Member(), TRANSIENT, {"converged": True, "residual_pt2": 0.0}, None) is None
    # a cap-bound basis no longer refuses a member (S4): it widens the weight interval
    assert member_refusal(_Member(), BOUND, {"converged": True}, None) is None
    guard = dict(BOUND, memory_bound=True)
    assert "memory guard" in member_refusal(_Member(), guard, {"converged": True}, None)


def test_gf_diagnostic_does_not_flag_a_transient():
    ok = check_ground_state_truncation(TRANSIENT, {"residual_pt2": 0.0})
    flagged = check_ground_state_truncation(BOUND, {"residual_pt2": 1e-3})
    assert ok.severity < flagged.severity
