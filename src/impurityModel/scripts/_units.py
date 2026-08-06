"""Shared eV/Ry unit conversion for CLI energy-valued arguments.

``spectra``/``selfenergy``/``susceptibility`` all default to eV (the ``.h0`` format's own
internal convention -- see ``doc/h0_file_format.md``). ``--unit Ry`` lets a caller pass
RSPt-native Rydberg values directly instead of converting by hand first, which is exactly the
mistake this exists to remove: Finding 1 of the NiO tutorial investigation was Rydberg values
passed to this eV-only interface with no way to say so.
"""

from impurityModel.ed.h0_format import RY_TO_EV


def convert_energy_args(args, fields, unit):
    """Convert the named ``argparse`` ``Namespace`` attributes from ``unit`` to eV, in place.

    Call this once, immediately after parsing and before any other code (including validation
    such as a unit-sanity warning) touches the named attributes -- everything downstream must
    see eV regardless of which unit the caller gave.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments, mutated in place.
    fields : sequence of str
        Attribute names on ``args`` to convert. Each may be a scalar float or a list of floats
        (e.g. ``--Fdd``); both are handled uniformly. A ``None`` value (an unset optional
        argument) is left untouched.
    unit : str
        ``"eV"`` (no-op) or ``"Ry"``.
    """
    if unit == "eV":
        return
    for name in fields:
        value = getattr(args, name)
        if value is None:
            continue
        if isinstance(value, list):
            setattr(args, name, [v * RY_TO_EV for v in value])
        else:
            setattr(args, name, value * RY_TO_EV)
