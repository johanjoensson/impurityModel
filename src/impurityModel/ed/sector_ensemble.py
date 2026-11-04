"""Charge-sector ensemble: the ``mixed_valence`` switch and (later) the member plan.

``mixed_valence`` is a tri-state grand-canonical multi-sector switch: ``"auto"``, ``"on"`` or ``"off"``.
This module resolves the user-facing spellings to one of those three strings. The value is replicated
input, so every rank resolves identically and may do so before any collective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Literal

import numpy as np

from impurityModel.ed import config
from impurityModel.ed.average import energy_cut

MixedValence = Literal["auto", "on", "off"]
_MODES = ("auto", "on", "off")

#: One-line note for a legacy positive integer / dict value (the seed-width meaning is gone).
LEGACY_NOTE = (
    "mixed_valence: an integer used to widen the seed's impurity window; that is gone. A non-zero value now "
    "means 'auto' (the charge-sector ensemble where it can be formed, else the winning sector with a warning); "
    "write 'on' to make a refused ensemble an error, or false/'off'."
)


def resolve_mixed_valence(value: Any, *, driver_default: MixedValence) -> tuple[MixedValence, str | None]:
    """Resolve a user ``mixed_valence`` value to ``"auto"``, ``"on"`` or ``"off"``.

    Parameters
    ----------
    value : None, bool, int, str or dict
        ``None`` takes ``driver_default``. ``"auto"``/``"on"``/``"off"`` (case-insensitive), ``True``/``False``
        and ``0`` are taken as named. A positive int is ``"auto"`` plus a note, not ``"on"``: an old input that
        asked for a wider seed never asked for a refused ensemble to abort the run (review X7). A legacy
        ``{group: int|bool}`` dict is ``"off"`` if every entry is falsy, else ``"auto"`` plus a note.
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
        return ("auto", LEGACY_NOTE) if any(bool(v) for v in value.values()) else ("off", None)
    if isinstance(value, bool):
        return ("on" if value else "off"), None
    if isinstance(value, Integral):
        if int(value) < 0:
            raise ValueError(f"mixed_valence={value!r}: must not be negative")
        return ("auto", LEGACY_NOTE) if int(value) > 0 else ("off", None)
    if isinstance(value, str):
        key = value.strip().lower()
        key = {"true": "on", "false": "off", "0": "off"}.get(key, key)
        if key in _MODES:
            return key, None  # type: ignore[return-value]
    raise ValueError(f"mixed_valence={value!r}: expected 'auto', true/'on', false/'off' (or None)")


# ---------------------------------------------------------------------------------------------------------------
# Membership: which charge sectors enter the ensemble, and how well the walk's energies fix their weights.
# Pure functions of the walk's replicated ``solved`` records: no MPI, so every rank computes the same plan.
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleMember:
    """One admitted charge sector.

    ``walk_energy`` and ``residual`` are the walk's ground-state energy of the sector and its PT2 error bar;
    ``weight`` is the Boltzmann weight ``p_s`` of the sector's lowest level among the members (degeneracy
    ignored: it only sizes the error bar), ``error_bar`` its contribution ``p_s |N_s - <N>| err_s / tau`` to the
    error on ``<N_imp>``.
    """

    trial_N0: dict
    total: int
    walk_energy: float
    residual: float
    certified: bool | None
    cap_bound: bool
    weight: float = 0.0
    error_bar: float = 0.0


@dataclass(frozen=True)
class EnsemblePlan:
    """Who is in the ensemble. ``members`` are ordered dominant first (lowest walk energy).

    ``mode`` is the resolved ``mixed_valence`` (``"auto"`` or ``"on"``). ``needs`` lists totals outside the
    solved range that must be solved before membership is decided (an admitted edge member's outer neighbour);
    while it is non-empty the plan is provisional. ``refused_reason`` is set when the ensemble cannot be formed
    (``members`` is then the winner alone, or, when a member's own thermal solve fails after lower members were
    consumed, the members before it); ``refused_by_cap`` says the refusal is a determinant-cap
    limit, the one reason a smaller or larger cap bears on; ``error_bar`` is the total error on ``<N_imp>`` in
    electrons.
    """

    mode: str
    members: tuple
    e_min: float
    cut: float
    error_bar: float = 0.0
    needs: tuple = ()
    refused_reason: str | None = None
    refused_by_cap: bool = False
    solved_totals: tuple = field(default=())


def plan_ensemble(
    solved,
    tau: float,
    mode: str,
    *,
    attainable: Callable[[int], bool] = lambda total: True,
    max_error: float | None = None,
) -> EnsemblePlan:
    """Decide the ensemble from the walk's solved sectors.

    A sector is admitted when its optimistic energy lies inside the thermal window of the lowest one:
    ``E_s - residual_s - E_min < energy_cut(tau)`` with ``E_min`` the lowest walk energy (the winner's). When the
    lowest or highest admitted total has an unsolved
    neighbour that is ``attainable``, the plan reports it in ``needs`` and is provisional: solve it and plan
    again. Once complete, the plan is refused (members = winner alone, ``refused_reason`` set) when an admitted
    member is cap-bound or uncertified, or when the error bar on ``<N_imp>`` exceeds ``max_error``
    (:data:`~impurityModel.ed.config.ENSEMBLE_WEIGHT_ERROR_MAX`).

    Parameters
    ----------
    solved : sequence of SectorRecord
        ``SectorSearch.solved`` (replicated).
    tau : float
        Temperature scale.
    mode : {"auto", "on"}
        Only recorded: the caller decides what a refusal means (warn for auto, raise for on).
    attainable : callable
        ``attainable(total)`` is False for a total the model cannot reach (bounds, frozen groups).
    max_error : float, optional
        Overrides the config threshold.
    """
    if not solved:
        raise ValueError("plan_ensemble: no solved sector")
    limit = config.ENSEMBLE_WEIGHT_ERROR_MAX if max_error is None else max_error
    cut = float(energy_cut(tau))
    records = sorted(solved, key=lambda r: (r.energy, r.total))
    e_min = float(records[0].energy)
    admitted = [r for r in records if r.energy - (r.residual_pt2 or 0.0) - e_min < cut]
    totals = {r.total for r in solved}
    needs = []
    lo, hi = min(r.total for r in admitted), max(r.total for r in admitted)
    for edge in (lo - 1, hi + 1):
        if edge not in totals and attainable(edge):
            needs.append(edge)
    solved_totals = tuple(sorted(totals))
    if needs:
        return EnsemblePlan(mode, _members(admitted, tau), e_min, cut, needs=tuple(needs), solved_totals=solved_totals)
    members = _members(admitted, tau)
    error = sum(m.error_bar for m in members)
    reason = None
    by_cap = False
    # A lone winner is a one-member ensemble however it was solved: cap / certification matter only when they
    # would corrupt a merge.
    for r in admitted if len(admitted) > 1 else ():
        if r.cap_bound:
            reason = f"the N_imp = {r.total} sector is bound by the determinant cap"
            by_cap = True
            break
        if r.certified is False:
            reason = f"the N_imp = {r.total} sector is not certified"
            break
    if reason is None and len(members) > 1 and error > limit:
        reason = (
            f"the occupation weights are fixed only to {error:.3f} e by the sector energies' truncation error "
            f"(limit {limit:g} e)"
        )
    if reason is not None:
        return EnsemblePlan(
            mode,
            members[:1],
            e_min,
            cut,
            error_bar=error,
            refused_reason=reason,
            refused_by_cap=by_cap,
            solved_totals=solved_totals,
        )
    return EnsemblePlan(mode, members, e_min, cut, error_bar=error, solved_totals=solved_totals)


def _members(admitted, tau) -> tuple:
    """:class:`EnsembleMember`s with their approximate weights and error-bar contributions."""
    e = np.array([r.energy for r in admitted], dtype=float)
    p = np.exp(-(e - e.min()) / tau)
    p /= p.sum()
    n = np.array([r.total for r in admitted], dtype=float)
    n_avg = float(p @ n)
    out = []
    for r, ps, ns in zip(admitted, p, n):
        res = float(r.residual_pt2 or 0.0)
        out.append(
            EnsembleMember(
                r.trial_N0,
                r.total,
                float(r.energy),
                res,
                r.certified,
                bool(r.cap_bound),
                weight=float(ps),
                error_bar=float(ps * abs(ns - n_avg) * res / tau),
            )
        )
    return tuple(out)


def sector_weights(e0, z, tau) -> "np.ndarray":
    """Normalised sector weights ``c_s ∝ Z_s exp(-(e0_s - e0_min)/tau)``.

    ``e0`` are the members' realised ground-state energies and ``z`` their partition functions *relative to their
    own e0* (``Z_s = sum_i exp(-(E_i - e0_s)/tau)``). Shifting every ``e0`` by a constant, or permuting the
    members, leaves ``c`` unchanged; one member gives exactly ``[1.0]``.
    """
    e0 = np.asarray(e0, dtype=float)
    z = np.asarray(z, dtype=float)
    w = z * np.exp(-(e0 - e0.min()) / tau)
    return w / w.sum()
