"""What the solver can actually do today, as opposed to what the format can express.

The input format deliberately puts **no** restriction on the shells' angular momenta: it has
to be able to describe the general case before the solver can execute it, or it would need
replacing on the day the 2p/3d restriction lifts. That leaves a gap between a valid file and
a runnable one, and this module is that gap, held in one declarative table so a
generalisation lands by deleting a row rather than by editing the schema.

Two kinds of "no" are distinguished, because they mean opposite things to the user:

* :class:`UnsupportedCalculation` -- the file is fine, the solver is not general enough yet.
  Every blocking assumption is named with its location so the message doubles as the to-do
  list for lifting it.
* :class:`InvalidShellCombination` -- the request is wrong at any level of generality (a
  dipole transition that is zero by selection rule, say), and no amount of generalisation
  would make it work.

A leaf module: standard library only.
"""

from dataclasses import dataclass
from typing import Optional

__all__ = [
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


@dataclass(frozen=True)
class Supported:
    """One row of the support table.

    Parameters
    ----------
    calculation : str
        ``"spectroscopy"``, ``"selfenergy"`` or ``"susceptibility"``.
    core_l : int or None
        Required core angular momentum, or ``None`` for "no core shell".
    valence_l : int or None
        Required valence angular momentum; ``None`` means any.
    why : str
        What makes this the supported set -- shown when a request misses.
    """

    calculation: str
    core_l: Optional[int]
    valence_l: Optional[int]
    why: str


#: Techniques that need a core hole, and therefore a ``role = "core"`` shell.
CORE_LEVEL_TECHNIQUES = ("xps", "xas", "rixs")

#: What runs today. Deliberately precise rather than pessimistic: only the *spectroscopy*
#: path is 2p/3d-bound. ``ImpurityModel.from_h0_file`` passes ``l`` through to
#: :func:`atomic_physics.getSOCop`, :func:`model.atomic_u4` and ``_add_soc_and_field``, all of
#: which are already general, so the self-energy and susceptibility paths take any single
#: correlated shell.
SUPPORTED = (
    Supported(
        "spectroscopy",
        core_l=1,
        valence_l=2,
        why="the transition-metal L2,3 edges (2p -> 3d)",
    ),
    Supported(
        "selfenergy",
        core_l=None,
        valence_l=None,
        why="any single correlated shell, with no core shell",
    ),
    Supported(
        "susceptibility",
        core_l=None,
        valence_l=None,
        why="any single correlated shell, with no core shell",
    ),
)

#: Where each 2p/3d assumption lives, so the error message is also the to-do list. Keep this
#: in step with the code: a lifted assumption is a deleted line here.
_BLOCKERS = (
    ("ImpurityModel.from_shells accepts only l in {1, 2}", "ed/model.py:461"),
    (
        "the Coulomb assembly is hardcoded to 2p/3d (the general slater_condon_Uop exists " "but is not wired up)",
        "ed/hamiltonian_io.py:256",
    ),
    ("the core and valence spin-orbit couplings are pinned to l=1 and l=2", "ed/hamiltonian_io.py:49-50"),
    ("the magnetic field is hardcoded to l=2", "ed/hamiltonian_io.py:53"),
    ("the double counting (dc_MLFT) is hardcoded to 2p/3d", "ed/atomic_physics.py"),
    ("PES/IPS transition operators default to l=2", "ed/transition_operators.py:259,279"),
)

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
                f"l_core={core_l} exceeds l_valence={valence_l}. The Slater-Condon assembly "
                "expects the core shell to be the smaller of the two (its array lengths are "
                "set by l_core), so this is a mislabelled pair of shells rather than an "
                "unsupported one."
            )

    # --- Valid, but maybe not yet runnable ----------------------------------------------
    rows = [row for row in SUPPORTED if row.calculation == calculation]
    if not rows:
        raise UnsupportedCalculation(f"Unknown calculation {calculation!r}.")
    for row in rows:
        if row.core_l is not None and row.core_l != core_l:
            continue
        if row.core_l is None and core_l is not None:
            continue
        if row.valence_l is not None and row.valence_l != valence_l:
            continue
        return

    requested = edge_name(core_l, valence_l) if core_l is not None else f"a valence l={valence_l} shell"
    supported = "; ".join(f"{row.why}" for row in rows)
    raise UnsupportedCalculation(
        f"{calculation} with {requested} is not supported yet.\n"
        "  The input file is valid. The solver is not general enough yet.\n"
        f"  Supported today: {supported}.\n"
        "  Blocking assumptions in the current code:\n" + _format_blockers()
    )
