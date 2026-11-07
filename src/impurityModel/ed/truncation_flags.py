"""Predicates on a ``truncation_report`` (Layer 0: imports nothing).

A report is written whenever a cap cycle ran or the memory guard limited an expansion, so its mere existence
says only that the cap *was reached at some point*. Two questions need different answers:

* :func:`cap_ever_hit` -- the **history**: did the cap or the memory guard bind in any cycle? An intermediate
  cut changes which candidates later cycles see, so the double-counting search (which asks "could raising the
  cap move this answer?") needs this.
* :func:`is_cap_bound` -- the **final state**: does the basis the expansion returned still miss candidates, or
  was it truncated with weight above tolerance by its last cycle? A basis that hit the cap, truncated and then
  re-admitted everything (discarded PT2 mass zero) is a transient: not cap-bound. The judgement uses the
  last selection that ran, which may be on the previous basis: a cap-bound basis the final cycle newly admitted into
  is not re-measured (``residual_is_current`` says so). The ``[cap-bound]`` display,
  sector verdicts and the ensemble weight interval need this.

A report without the ``cap_bound`` key (an old pickle) falls back to its ``cap_hit`` (cap-bound when that too is
absent), the conservative answer.
"""


def cap_ever_hit(report):
    """Whether the cap or the memory guard bound in any cycle of the expansion that wrote ``report``."""
    return report is not None


def is_cap_bound(report):
    """Whether the expansion that wrote ``report`` ended cap-bound (final state); ``False`` for ``None``."""
    if report is None:
        return False
    return bool(report.get("cap_bound", report.get("cap_hit", True)))
