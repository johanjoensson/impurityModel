"""Test seams for the two always-on Green's-function switches (``gf_solvers._SUPPORT_FREEZE`` and
``_FROZEN_CSR``): a test turns one off to exercise the plain sparse path it replaces. Not user knobs."""

from contextlib import contextmanager


def _truthy(value):
    return str(value).strip().lower() not in ("0", "false", "no", "")


def set_gf_switches(monkeypatch, *, support_freeze=None, frozen_csr=None):
    """Set the switches for the current test (restored by ``monkeypatch``); ``None`` leaves one alone."""
    from impurityModel.ed import gf_solvers

    if support_freeze is not None:
        monkeypatch.setattr(gf_solvers, "_SUPPORT_FREEZE", _truthy(support_freeze))
    if frozen_csr is not None:
        monkeypatch.setattr(gf_solvers, "_FROZEN_CSR", _truthy(frozen_csr))


@contextmanager
def gf_switches(*, support_freeze=None, frozen_csr=None):
    """The same as :func:`set_gf_switches`, as a context manager (for tests without ``monkeypatch``)."""
    from impurityModel.ed import gf_solvers

    old = (gf_solvers._SUPPORT_FREEZE, gf_solvers._FROZEN_CSR)
    try:
        if support_freeze is not None:
            gf_solvers._SUPPORT_FREEZE = _truthy(support_freeze)
        if frozen_csr is not None:
            gf_solvers._FROZEN_CSR = _truthy(frozen_csr)
        yield
    finally:
        gf_solvers._SUPPORT_FREEZE, gf_solvers._FROZEN_CSR = old
