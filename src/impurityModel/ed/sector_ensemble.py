"""Charge-sector ensemble: the ``mixed_valence`` switch and (later) the member plan.

``mixed_valence`` is a tri-state grand-canonical multi-sector switch: ``"auto"``, ``"on"`` or ``"off"``.
This module resolves the user-facing spellings to one of those three strings. The value is replicated
input, so every rank resolves identically and may do so before any collective.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal

import numpy as np

MixedValence = Literal["auto", "on", "off"]
_MODES = ("auto", "on", "off")

#: One-line note for a legacy positive integer / dict value (the seed-width meaning is gone).
LEGACY_NOTE = (
    "mixed_valence: an integer used to widen the seed's impurity window; that is gone. A non-zero value now "
    "means 'on' (the charge-sector ensemble); use true/false/'auto'."
)


def resolve_mixed_valence(value: Any, *, driver_default: MixedValence) -> tuple[MixedValence, str | None]:
    """Resolve a user ``mixed_valence`` value to ``"auto"``, ``"on"`` or ``"off"``.

    Parameters
    ----------
    value : None, bool, int, str or dict
        ``None`` takes ``driver_default``. ``"auto"``/``"on"``/``"off"`` (case-insensitive), ``True``/``False``
        and ``0`` are taken as named. A positive int is ``"on"`` plus a note. A legacy ``{group: int|bool}``
        dict is ``"off"`` if every entry is falsy, else ``"on"`` plus a note.
    driver_default : {"auto", "on", "off"}
        What ``None`` means for the calling driver (Sigma/DC: ``"auto"``; spectra/RIXS/chi: ``"off"``).

    Returns
    -------
    mode, note : tuple
        The resolved mode and an optional one-line note for rank 0 to print.

    Raises
    ------
    ValueError
        For an unknown string, a negative int, a dict holding anything but int/bool, or an unsupported type.
    """
    if driver_default not in _MODES:
        raise ValueError(f"driver_default must be one of {_MODES}, got {driver_default!r}")
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.generic):  # an HDF5 attribute arrives as a numpy scalar
        value = value.item()
    if value is None:
        return driver_default, None
    if isinstance(value, dict):
        if not all(isinstance(v, (Integral, bool)) or v is None for v in value.values()):
            raise ValueError(f"mixed_valence={value!r}: a dict is accepted only with int/bool values (legacy form)")
        return ("on", LEGACY_NOTE) if any(bool(v) for v in value.values()) else ("off", None)
    if isinstance(value, bool):
        return ("on" if value else "off"), None
    if isinstance(value, Integral):
        if value < 0:
            raise ValueError(f"mixed_valence={value!r}: must not be negative")
        return ("on", LEGACY_NOTE) if value > 0 else ("off", None)
    if isinstance(value, str):
        key = value.strip().lower()
        key = {"true": "on", "false": "off", "0": "off"}.get(key, key)
        if key in _MODES:
            return key, None  # type: ignore[return-value]
    raise ValueError(f"mixed_valence={value!r}: expected 'auto', true/'on', false/'off' (or None)")


def reject_unmigrated_ensemble(value: Any, *, driver: str, driver_default: MixedValence = "off") -> MixedValence:
    """Resolve ``mixed_valence`` for a driver that cannot form the ensemble (yet).

    ``"off"`` and ``"auto"`` both run the winning sector alone (``auto`` never raises: it resolves to off);
    an explicit ``"on"`` raises, because ignoring it would hand a single-sector answer to a caller who asked
    for the ensemble. Replicated input: every rank raises together, so call it before any collective.

    Returns
    -------
    {"auto", "off"}
        The resolved mode (``"on"`` never returns).

    Raises
    ------
    ValueError
        If the value resolves to ``"on"``.
    """
    mode, _note = resolve_mixed_valence(value, driver_default=driver_default)
    if mode == "on":
        raise ValueError(
            f"mixed_valence={value!r}: the charge-sector ensemble is not implemented for {driver} yet; "
            "use 'auto', false or leave it unset."
        )
    return mode
