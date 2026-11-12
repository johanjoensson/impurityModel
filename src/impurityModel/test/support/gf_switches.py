"""Test seams for the Green's-function switches (``gf_solvers._BICGSTAB_CLOSURE_FREEZE`` and ``_MATRIX_MODE``): a
test turns the closure freeze off to exercise the plain path it replaces, or forces a matvec mode of the Lanczos
GF's recurrence. Not user knobs.

``_BICGSTAB_CLOSURE_FREEZE`` gates the per-frequency kernel's closure freeze (and the RIXS R1 chain's).
``_MATRIX_MODE`` is ``"auto"`` (the mode is priced per unit), ``"discovered"``, ``"retained"`` or ``"none"``. The
block-Lanczos support settle is not a seam: switch it with the ``gf_settle`` solver option, or
``block_Green_sparse(support_settle=False)``.
"""

from contextlib import contextmanager


def _truthy(value):
    return str(value).strip().lower() not in ("0", "false", "no", "")


def set_gf_switches(monkeypatch, *, closure_freeze=None, matrix_mode=None):
    """Set the switches for the current test (restored by ``monkeypatch``); ``None`` leaves one alone."""
    from impurityModel.ed import gf_solvers

    if closure_freeze is not None:
        monkeypatch.setattr(gf_solvers, "_BICGSTAB_CLOSURE_FREEZE", _truthy(closure_freeze))
    if matrix_mode is not None:
        monkeypatch.setattr(gf_solvers, "_MATRIX_MODE", matrix_mode)


@contextmanager
def gf_switches(*, closure_freeze=None, matrix_mode=None):
    """The same as :func:`set_gf_switches`, as a context manager (for tests without ``monkeypatch``)."""
    from impurityModel.ed import gf_solvers

    old = (gf_solvers._BICGSTAB_CLOSURE_FREEZE, gf_solvers._MATRIX_MODE)
    try:
        if closure_freeze is not None:
            gf_solvers._BICGSTAB_CLOSURE_FREEZE = _truthy(closure_freeze)
        if matrix_mode is not None:
            gf_solvers._MATRIX_MODE = matrix_mode
        yield
    finally:
        gf_solvers._BICGSTAB_CLOSURE_FREEZE, gf_solvers._MATRIX_MODE = old


@contextmanager
def gf_tuning(*, eigenstate_group=None, per_state_restrict=None):
    """Patch the unit-shape constants (``gf_units._EIGENSTATE_GROUP``, ``greens_function._PER_STATE_RESTRICT``)
    for a block; ``None`` leaves one alone. Not user knobs."""
    from impurityModel.ed import gf_units, greens_function

    old = (gf_units._EIGENSTATE_GROUP, greens_function._PER_STATE_RESTRICT)
    try:
        if eigenstate_group is not None:
            gf_units._EIGENSTATE_GROUP = int(eigenstate_group)
        if per_state_restrict is not None:
            greens_function._PER_STATE_RESTRICT = bool(per_state_restrict)
        yield
    finally:
        gf_units._EIGENSTATE_GROUP, greens_function._PER_STATE_RESTRICT = old
