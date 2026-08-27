"""What the solver can actually do today, as opposed to what the format can express.

The input format deliberately puts **no** restriction on the shells' angular momenta, so
there is in principle a gap between a valid file and a runnable one. This module is that gap,
held in one declarative table so a generalisation lands by widening a row rather than by
editing the schema.

The gap is currently empty on the angular momenta: the spectroscopy path used to run only the
2p -> 3d L2,3 edge, and now takes any dipole-allowed core/valence pair. What is left is the
*shape* of a calculation -- a self-energy run still builds one correlated shell and no core
shell -- and the selection rule, which is not a gap at all.

Two kinds of "no" are distinguished, because they mean opposite things to the user:

* :class:`UnsupportedCalculation` -- the file is fine, the solver is not general enough yet.
  Any blocking assumption is named with its location so the message doubles as the to-do list
  for lifting it. There are none at present.
* :class:`InvalidShellCombination` -- the request is wrong at any level of generality (a
  dipole transition that is zero by selection rule, say), and no amount of generalisation
  would make it work. This one does not go away.

A leaf module: standard library only.
"""

from dataclasses import dataclass
from typing import Optional, Union

__all__ = [
    "ANY",
    "UnsupportedCalculation",
    "InvalidShellCombination",
    "SUPPORTED",
    "CORE_LEVEL_TECHNIQUES",
    "check",
    "edge_name",
]


class UnsupportedCalculation(NotImplementedError):
    """The input is valid but the current solver cannot run it."""


class InvalidShellCombination(ValueError):
    """The requested shells cannot produce the requested spectroscopy, at any generality."""


#: Sentinel: this row places no restriction on that angular momentum. Distinct from ``None``,
#: which for ``core_l`` means the concrete requirement "there must be no core shell". The two
#: used to be spelled the same way, which only worked while no row wanted "any core shell".
ANY = "any"


@dataclass(frozen=True)
class Supported:
    """One row of the support table.

    Parameters
    ----------
    calculation : str
        ``"spectroscopy"``, ``"selfenergy"`` or ``"susceptibility"``.
    core_l : int, None or ANY
        Required core angular momentum; ``None`` requires *no* core shell, :data:`ANY`
        accepts either.
    valence_l : int or ANY
        Required valence angular momentum; :data:`ANY` accepts any.
    why : str
        What makes this the supported set -- shown when a request misses.
    """

    calculation: str
    core_l: Union[int, None, str]
    valence_l: Union[int, str]
    why: str


#: Techniques that need a core hole, and therefore a ``role = "core"`` shell.
CORE_LEVEL_TECHNIQUES = ("xps", "xas", "rixs")

#: What runs today. Every path now takes the shells it is given: the interacting assembly
#: (:func:`atomic_physics.slater_condon_Uop`, :func:`atomic_physics.dc_MLFT`), the spin-orbit
#: and Zeeman terms, the transition operators and the basis-expansion windows are all told
#: which shell is the core one rather than assuming l=1 and l=2. What remains here is the
#: shape of a calculation, not its angular momenta.
SUPPORTED = (
    Supported(
        "spectroscopy",
        core_l=ANY,
        valence_l=ANY,
        why="any dipole-allowed core/valence pair, or a valence shell alone (PES/IPS/NIXS)",
    ),
    Supported(
        "selfenergy",
        core_l=None,
        valence_l=ANY,
        why="any single correlated shell, with no core shell",
    ),
    Supported(
        "susceptibility",
        core_l=None,
        valence_l=ANY,
        why="any single correlated shell, with no core shell",
    ),
)

#: Where an assumption that narrows the support table lives, so the error message doubles as
#: the to-do list for lifting it. Empty: the 2p/3d assumptions the spectroscopy path used to
#: carry are gone. Keep it that way -- a new row here is a new restriction, not a note.
_BLOCKERS = ()

#: Spectroscopic edge names, for messages. Purely cosmetic, but "L2,3" is what a user calls
#: the thing they asked for.
_EDGE_NAMES = {
    (0, 1): "K",
    (1, 2): "L2,3",
    (2, 3): "M4,5",
    (3, 4): "N6,7",
}


def edge_name(core_l: Optional[int], valence_l: Optional[int]) -> str:
    """Human name of a core -> valence transition, e.g. ``"L2,3 (2p -> 3d)"``."""
    if core_l is None or valence_l is None:
        return "no core-level transition"
    shells = "spdfghi"
    label = _EDGE_NAMES.get((core_l, valence_l))
    detail = f"{shells[core_l] if core_l < len(shells) else f'l={core_l}'} -> " + (
        shells[valence_l] if valence_l < len(shells) else f"l={valence_l}"
    )
    return f"{label} ({detail})" if label else detail


def _format_blockers() -> str:
    return "\n".join(f"    - {text}\n      ({where})" for text, where in _BLOCKERS)


def check(calculation, core_l, valence_l, techniques=()):
    """Refuse a valid-but-unrunnable request, with a message that says which it is.

    Parameters
    ----------
    calculation : str
        The calculation table that was present.
    core_l : int or None
        Angular momentum of the ``role = "core"`` shell, or ``None`` if there is none.
    valence_l : int
        Angular momentum of the ``role = "valence"`` shell.
    techniques : sequence of str, optional
        Enabled spectroscopy techniques, lower-case.

    Raises
    ------
    InvalidShellCombination
        If the shells cannot yield the requested spectroscopy at any level of generality.
    UnsupportedCalculation
        If the request is well-formed but outside the current support table.
    """
    techniques = tuple(techniques)
    needs_core = any(t in CORE_LEVEL_TECHNIQUES for t in techniques)

    # --- Wrong at any generality, so not a "not yet" ------------------------------------
    if needs_core and core_l is None:
        raise InvalidShellCombination(
            f"{', '.join(t for t in techniques if t in CORE_LEVEL_TECHNIQUES)} needs a core "
            'hole, but no shell declares role = "core".'
        )
    if needs_core:
        if abs(core_l - valence_l) != 1:
            raise InvalidShellCombination(
                f"A dipole transition between l={core_l} and l={valence_l} is zero by the "
                "Gaunt selection rule, which requires |l_core - l_valence| = 1. This is not a "
                "missing feature: the spectrum would be identically zero."
            )
        if core_l > valence_l:
            raise InvalidShellCombination(
                f"l_core={core_l} exceeds l_valence={valence_l}. A core level lies below the "
                "valence shell it is excited into, so this is a mislabelled pair of shells "
                "rather than an unsupported one -- check which [[shell]] carries "
                'role = "core".'
            )

    # --- Valid, but maybe not yet runnable ----------------------------------------------
    rows = [row for row in SUPPORTED if row.calculation == calculation]
    if not rows:
        raise UnsupportedCalculation(f"Unknown calculation {calculation!r}.")
    for row in rows:
        if row.core_l is not ANY and row.core_l != core_l:
            continue
        if row.valence_l is not ANY and row.valence_l != valence_l:
            continue
        return

    requested = edge_name(core_l, valence_l) if core_l is not None else f"a valence l={valence_l} shell"
    supported = "; ".join(f"{row.why}" for row in rows)
    message = (
        f"{calculation} with {requested} is not supported yet.\n"
        "  The input file is valid. The solver is not general enough yet.\n"
        f"  Supported today: {supported}."
    )
    if _BLOCKERS:
        message += "\n  Blocking assumptions in the current code:\n" + _format_blockers()
    raise UnsupportedCalculation(message)
