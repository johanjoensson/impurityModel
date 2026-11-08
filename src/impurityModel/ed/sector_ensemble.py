"""Charge-sector ensemble: the ``mixed_valence`` switch and the member plan.

``mixed_valence`` is a boolean grand-canonical multi-sector switch: on lets the code pick the total-N sectors inside
the thermal window (a one-member ensemble is fine), off keeps the winning sector alone. This module resolves the
user-facing spellings to that bool. The value is replicated input, so every rank resolves identically and may do so
before any collective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Protocol

import numpy as np

from impurityModel.ed.average import energy_cut

_SPELLINGS = {"auto": True, "on": True, "true": True, "off": False, "false": False, "0": False}

#: One-line note for a legacy positive integer / dict value (the seed-width meaning is gone).
LEGACY_NOTE = (
    "mixed_valence: an integer used to widen the seed's impurity window; that is gone. A non-zero value now "
    "means on (the charge-sector ensemble where it can be formed, else the winning sector with a warning); "
    "write false/'off' to keep the winning sector alone."
)


def resolve_mixed_valence(value: Any, *, driver_default: Any = None, default: bool = True) -> tuple[bool, str | None]:
    """Resolve a user ``mixed_valence`` value to a bool.

    Parameters
    ----------
    value : None, bool, int, str or dict
        ``None`` takes the default. ``"on"``/``"auto"``/``"true"`` (case-insensitive) and ``True`` are on;
        ``"off"``/``"false"``, ``False`` and ``0`` are off (``"auto"`` stays accepted: archives and the ``mv``
        token written since 2026-11-03 carry it). A positive int is on plus a note. A legacy
        ``{group: int|bool}`` dict is off if every entry is falsy, else on plus a note.
    driver_default : {"auto", "on", "off"} or bool, optional
        What ``None`` means for the calling driver; takes precedence over ``default`` while a caller still passes
        one.
    default : bool
        What ``None`` means otherwise.

    Returns
    -------
    on, note : tuple
        The resolved switch and an optional one-line note for rank 0 to print.

    Raises
    ------
    ValueError
        For an unknown string, a negative int, a dict holding anything but int/bool, or an unsupported type.
    """
    if driver_default is not None:
        default = _coerce_default(driver_default)
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.generic):  # an HDF5 attribute arrives as a numpy scalar
        value = value.item()
    if value is None:
        return bool(default), None
    if isinstance(value, dict):
        if not all(isinstance(v, (Integral, bool)) or v is None for v in value.values()):
            raise ValueError(f"mixed_valence={value!r}: a dict is accepted only with int/bool values (legacy form)")
        return (True, LEGACY_NOTE) if any(bool(v) for v in value.values()) else (False, None)
    if isinstance(value, bool):
        return value, None
    if isinstance(value, Integral):
        if int(value) < 0:
            raise ValueError(f"mixed_valence={value!r}: must not be negative")
        return (True, LEGACY_NOTE) if int(value) > 0 else (False, None)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _SPELLINGS:
            return _SPELLINGS[key], None
    raise ValueError(f"mixed_valence={value!r}: expected true/'on'/'auto', false/'off' (or None)")


def _coerce_default(driver_default: Any) -> bool:
    if isinstance(driver_default, (bool, np.bool_)):
        return bool(driver_default)
    if isinstance(driver_default, str) and driver_default in ("auto", "on", "off"):
        return driver_default != "off"
    raise ValueError(f"driver_default must be 'auto', 'on', 'off' or a bool, got {driver_default!r}")


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

    ``needs`` lists totals outside the
    solved range that must be solved before membership is decided (an admitted edge member's outer neighbour);
    while it is non-empty the plan is provisional. ``refused_reason`` is set when the ensemble cannot be formed
    (``members`` is then the winner alone, or, when a member's own thermal solve fails after lower members were
    consumed, the members before it); ``error_bar`` is the total error on ``<N_imp>`` in
    electrons.
    """

    members: tuple
    e_min: float
    cut: float
    error_bar: float = 0.0
    needs: tuple = ()
    refused_reason: str | None = None
    solved_totals: tuple = field(default=())


def plan_ensemble(
    solved,
    tau: float,
    *,
    attainable: Callable[[int], bool] = lambda total: True,
) -> EnsemblePlan:
    """Decide the ensemble from the walk's solved sectors.

    A sector is admitted when its optimistic energy lies inside the thermal window of the lowest one:
    ``E_s - residual_s - E_min < energy_cut(tau)`` with ``E_min`` the lowest walk energy (the winner's). When the
    lowest or highest admitted total has an unsolved
    neighbour that is ``attainable``, the plan reports it in ``needs`` and is provisional: solve it and plan
    again. Once complete, the plan is never refused: a cap-bound or uncertified member
    enters, and the weight uncertainty is an interval reported after the solves (:func:`weight_interval`).

    Parameters
    ----------
    solved : sequence of SectorRecord
        ``SectorSearch.solved`` (replicated).
    tau : float
        Temperature scale.
    attainable : callable
        ``attainable(total)`` is False for a total the model cannot reach (bounds, frozen groups).
    """
    if not solved:
        raise ValueError("plan_ensemble: no solved sector")
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
        return EnsemblePlan(_members(admitted, tau), e_min, cut, needs=tuple(needs), solved_totals=solved_totals)
    members = _members(admitted, tau)
    error = sum(m.error_bar for m in members)
    # Admission is by energy alone (review S4): a cap-bound or uncertified member is still a member. What its
    # truncation does to the weights is an interval, computed from the realised energies after the solves
    # (:func:`weight_interval`), not a reason to drop the sector.
    return EnsemblePlan(members, e_min, cut, error_bar=error, solved_totals=solved_totals)


def winner_plan(solved, winner_total: int, tau: float) -> EnsemblePlan:
    """The plan of ``mixed_valence`` off: the winning sector alone. Never provisional, never refused."""
    records = [r for r in solved if r.total == winner_total]
    if not records:
        raise ValueError(f"winner_plan: sector {winner_total} was not solved")
    e_min = float(records[0].energy)
    return EnsemblePlan(
        _members(records, tau),
        e_min,
        float(energy_cut(tau)),
        solved_totals=tuple(sorted({r.total for r in solved})),
    )


class MemberVisitor(Protocol):
    """What a driver does with one ensemble member (level 3 of ``solve_charge_sector_ensemble``).

    ``visit(member, state)`` runs once per member, in plan order, after the member's solver and Krylov store have
    been dropped; ``state`` is a :class:`~impurityModel.ed.groundstate.MemberState` (raw ``psis``), or, for a driver
    visitor passed to ``calc_gs``, a :class:`~impurityModel.ed.groundstate.DrivenState` (redistributed ``states``,
    no ``psis``). It is collective on the
    ensemble's communicator, on every rank, and its return value is collected in ``EnsembleResult.visited``.
    """

    def visit(self, member: EnsembleMember, state: Any) -> Any: ...


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


#: How many residual PT2 a member's true energy may lie below its realised one (calibrated, not a bound).
ENSEMBLE_RESIDUAL_FACTOR = 3.0


def weight_interval(n, e0, z, tau, lo_shift, hi_shift=None):
    """``(lo, hi)`` of ``<N>`` when each sector's energy may lie in ``[e0_s - lo_shift_s, e0_s + hi_shift_s]``.

    ``<N> = sum a_s N_s / sum a_s`` with ``a_s = Z_s exp(-E_s/tau)`` is linear-fractional, hence monotone in each
    ``a_s``: its extremes sit at box vertices. Sorting by ``N_s``, the maximum puts the sectors of the high-``N``
    tail at their largest ``a`` and the rest at their smallest (and the minimum the reverse), so a scan over the
    ``K + 1`` thresholds is exact. ``lo_shift`` is how far a sector's true energy may lie *below* the realised one
    (truncation leaves energies as upper bounds), ``hi_shift`` above it (default 0).
    """
    n = np.asarray(n, dtype=float)
    e0 = np.asarray(e0, dtype=float)
    z = np.asarray(z, dtype=float)
    lo_shift = np.asarray(lo_shift, dtype=float)
    hi_shift = np.zeros_like(lo_shift) if hi_shift is None else np.asarray(hi_shift, dtype=float)
    # Log weights: each vertex is normalised by its own largest term, so neither a wide box (exp overflow)
    # nor a vertex whose terms all sit far below the reference (0 / 0) can produce a NaN.
    log_max = np.log(z) - (e0 - lo_shift) / tau
    log_min = np.log(z) - (e0 + hi_shift) / tau
    order = np.argsort(n, kind="stable")
    n, log_max, log_min = n[order], log_max[order], log_min[order]
    k = len(n)

    def extreme(sign):
        best = None
        for cut in range(k + 1):
            high = np.arange(k) >= cut  # the sectors at the heavy end
            log_a = np.where(high == (sign > 0), log_max, log_min)
            a = np.exp(log_a - log_a.max())
            value = float(a @ n / a.sum())
            if best is None or sign * value > sign * best:
                best = value
        return best

    return extreme(-1), extreme(+1)
