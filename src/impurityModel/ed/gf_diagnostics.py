r"""Convergence and consistency diagnostics for the interacting Green's function.

The exact-diagonalization self-energy pipeline (``selfenergy.calc_selfenergy`` ->
``greens_function.get_Greens_function``) can silently produce a wrong result in two ways
that no longer go unnoticed thanks to this module:

* **Truncated thermal ground-state ensemble.** The eigensolver is asked for a fixed number
  of low-lying states (``num_wanted``) and keeps those within
  :math:`\Delta E_{\mathrm{cut}} = -\tau\ln(\epsilon_{\mathrm{B}})`.  A dense / near-degenerate
  low-energy spectrum can hold *more* thermal states than were computed, so the highest
  retained state still carries a non-negligible Boltzmann weight and the thermal average is
  biased.
* **Under-resolved block Lanczos.** The continued fraction can stop before the spectral
  function is converged, or the real-frequency mesh can fail to cover the spectral support.

Each check returns a :class:`Diagnostic` (``OK`` / ``WARN`` / ``FAIL`` plus a measured value,
threshold, human-readable message, and a concrete suggestion).  A :class:`DiagnosticReport`
collects them per Green's-function block and renders one compact table, and exposes the
aggregate ``needs_more_states`` / ``needs_more_iterations`` flags that drive the single
auto-retry in ``calc_selfenergy``.

Severity policy: checks are advisory (``WARN``) by default; ``FAIL`` is reserved for results
that are outright unphysical (a violated exact sum rule, a non-causal Green's function).

.. note::
    A frequency-domain addition/removal consistency check (peaks in
    :math:`G^{\mathrm{add}} - w\,G^{\mathrm{rem}}` flagging truncated eigenstates) was
    designed but **deferred**: the underlying KMS detailed-balance relation
    :math:`A^{\mathrm{rem}}(\omega) = e^{-\omega/\tau} A^{\mathrm{add}}(\omega)` holds for a
    grand-canonical (mixed particle number) ensemble, whereas this code uses a fixed
    total-N ensemble.  See :func:`addition_removal_peak_check` for the placeholder.
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum

import numpy as np

# Boltzmann-weight threshold used to define the energy cut for the thermal states
# (``energy_cut = -tau * ln(BOLTZMANN_DESIGN_WEIGHT)`` in groundstate.py / selfenergy.py).
BOLTZMANN_DESIGN_WEIGHT = 1e-4


class Severity(IntEnum):
    """Ordered severity levels (higher = worse) so ``max`` selects the worst."""

    OK = 0
    WARN = 1
    FAIL = 2


_SEVERITY_TAG = {Severity.OK: "OK  ", Severity.WARN: "WARN", Severity.FAIL: "FAIL"}


@dataclasses.dataclass
class Diagnostic:
    r"""Result of a single convergence/consistency check.

    Args:
        name: Short identifier of the check (column header in the report).
        severity: :class:`Severity` level of the outcome.
        value: The measured quantity the check is based on.
        threshold: The value beyond which the check is no longer ``OK``.
        message: One-line human-readable summary of the outcome.
        suggestion: Concrete, actionable hint shown when ``severity`` is not ``OK``.
        needs_more_states: Hint to the auto-retry that the thermal ensemble is too small.
        needs_more_iterations: Hint to the auto-retry that the Lanczos run was too short.
    """

    name: str
    severity: Severity
    value: float
    threshold: float
    message: str
    suggestion: str = ""
    needs_more_states: bool = False
    needs_more_iterations: bool = False


class DiagnosticReport:
    """Collection of :class:`Diagnostic` entries grouped by Green's-function block."""

    def __init__(self):
        self._by_block: dict[str, list[Diagnostic]] = {}

    def add(self, block_label: str, diagnostic: Diagnostic) -> None:
        """Record ``diagnostic`` under ``block_label`` (creating the group if needed)."""
        self._by_block.setdefault(str(block_label), []).append(diagnostic)

    def extend(self, block_label: str, diagnostics) -> None:
        """Record several diagnostics under ``block_label``."""
        for diagnostic in diagnostics:
            self.add(block_label, diagnostic)

    @property
    def diagnostics(self) -> list[Diagnostic]:
        """All recorded diagnostics, flattened."""
        return [d for group in self._by_block.values() for d in group]

    @property
    def worst_severity(self) -> Severity:
        """The highest :class:`Severity` over all recorded diagnostics."""
        return max((d.severity for d in self.diagnostics), default=Severity.OK)

    @property
    def needs_more_states(self) -> bool:
        """True if any non-OK diagnostic asks for a larger thermal ensemble."""
        return any(d.needs_more_states and d.severity != Severity.OK for d in self.diagnostics)

    @property
    def needs_more_iterations(self) -> bool:
        """True if any non-OK diagnostic asks for more Lanczos iterations."""
        return any(d.needs_more_iterations and d.severity != Severity.OK for d in self.diagnostics)

    def render(self, only_problems: bool = False) -> str:
        """Return a compact aligned table of the recorded diagnostics.

        Args:
            only_problems: If True, omit ``OK`` rows (show only ``WARN`` / ``FAIL``).

        Returns:
            A multi-line string suitable for printing on the root MPI rank.
        """
        rows = []
        for block_label, group in self._by_block.items():
            for d in group:
                if only_problems and d.severity == Severity.OK:
                    continue
                rows.append((block_label, d))
        if not rows:
            return "Green's-function diagnostics: all checks passed."

        name_w = max(len(d.name) for _, d in rows)
        block_w = max(len(b) for b, _ in rows)
        lines = ["Green's-function convergence / consistency diagnostics:"]
        header = f"  {'block':<{block_w}}  {'check':<{name_w}}  stat  {'value':>11}  message"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for block_label, d in rows:
            lines.append(
                f"  {block_label:<{block_w}}  {d.name:<{name_w}}  "
                f"{_SEVERITY_TAG[d.severity]}  {d.value:>11.3e}  {d.message}"
            )
            if d.severity != Severity.OK and d.suggestion:
                lines.append(f"  {'':<{block_w}}  {'':<{name_w}}        {'':>11}  -> {d.suggestion}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _seed_weight(r_e) -> float:
    r"""Zeroth spectral moment of one seed block: :math:`\operatorname{tr}(r^\dagger r)`.

    For the addition seed :math:`r` (the QR factor of the block
    :math:`[c_i^\dagger|s\rangle]_i`) this equals
    :math:`\sum_i \lVert c_i^\dagger|s\rangle\rVert^2 = \sum_i(1-n_i)`; for the removal seed
    it equals :math:`\sum_i n_i`.  ``r`` is the small ``(n_ops, n_ops)`` upper-triangular
    factor, so this is cheap and exact.
    """
    r_arr = np.asarray(r_e)
    return float(np.real(np.sum(np.abs(r_arr) ** 2)))


def _thermal_seed_weight(r_list, es, e0, tau) -> float:
    r"""Boltzmann-weighted seed moment :math:`\sum_s e^{-(e_s-e_0)/\tau}\operatorname{tr}(r_s^\dagger r_s)`."""
    return float(sum(np.exp(-(e - e0) / tau) * _seed_weight(r_s) for e, r_s in zip(es, r_list)))


# --------------------------------------------------------------------------- #
# Checks                                                                       #
# --------------------------------------------------------------------------- #


def check_spectral_sum_rule(r_IPS_list, r_PS_list, es, e0, tau, block_dim, rtol: float = 1e-6) -> Diagnostic:
    r"""Exact anticommutator sum rule per thermal state: addition + removal weight = orbitals.

    For every thermal state :math:`|s\rangle` and orbital :math:`i` in the block,
    :math:`\langle s|c_i c_i^\dagger|s\rangle + \langle s|c_i^\dagger c_i|s\rangle = 1` (from
    :math:`\{c_i,c_i^\dagger\}=1`).  Summed over the block this is
    :math:`\operatorname{tr}(r^{\mathrm{IPS}\dagger}_s r^{\mathrm{IPS}}_s)
    + \operatorname{tr}(r^{\mathrm{PS}\dagger}_s r^{\mathrm{PS}}_s) = \texttt{block\_dim}`,
    independent of any Lanczos truncation.  A violation means a seed state is not a
    normalized eigenstate (a real bug), hence ``FAIL`` severity.

    Args:
        r_IPS_list, r_PS_list: Per-thermal-state QR seed factors for addition / removal.
        es: Thermal-state energies (only their count/order must match the seed lists).
        e0, tau: Ground energy and temperature scale (unused except for signature symmetry).
        block_dim: Number of orbitals in the block (the expected per-state weight sum).
        rtol: Relative tolerance on the deviation from ``block_dim``.

    Returns:
        Diagnostic: ``OK`` within tolerance, ``FAIL`` otherwise.
    """
    worst = 0.0
    for r_ips, r_ps in zip(r_IPS_list, r_PS_list):
        total = _seed_weight(r_ips) + _seed_weight(r_ps)
        worst = max(worst, abs(total - block_dim))
    threshold = rtol * max(block_dim, 1)
    ok = worst <= threshold
    return Diagnostic(
        name="sum_rule",
        severity=Severity.OK if ok else Severity.FAIL,
        value=worst,
        threshold=threshold,
        message=("addition+removal seed weight = #orbitals" if ok else "anticommutator sum rule violated"),
        suggestion="" if ok else "a thermal seed state is not a normalized eigenstate; check the eigensolver/basis",
    )


def _trapezoid_lorentzian_error(mesh, delta) -> float:
    r"""Worst-case relative trapezoidal-quadrature error for a Lorentzian of width ``|delta|``.

    Integrating a Lorentzian of HWHM :math:`\delta` on a uniform mesh of spacing :math:`h`
    has error :math:`\approx 2\,e^{-2\pi|\delta|/h}` (Poisson summation; the Fourier transform
    of a Lorentzian is :math:`e^{-\delta|t|}`).  ``h`` is taken as the largest gap in ``mesh``
    (the binding spacing).  Returns ``0`` when there is no broadening (``delta == 0``).
    """
    mesh = np.real(np.asarray(mesh))
    d = abs(float(delta)) if delta is not None else 0.0
    if d == 0.0 or mesh.size < 2:
        return 0.0
    h = float(np.max(np.diff(np.sort(mesh))))
    return float(2.0 * np.exp(-2.0 * np.pi * d / h)) if h > 0 else 0.0


def check_mesh_density(mesh, delta, rtol: float = 2e-2) -> Diagnostic:
    r"""Is the real-frequency mesh fine enough to integrate the Lorentzian-broadened spectrum?

    The zeroth-moment (sum-rule / weight) checks integrate spectral functions broadened by
    :math:`\delta` with the trapezoidal rule, whose error is
    :math:`\approx 2\,e^{-2\pi|\delta|/h}` for spacing :math:`h`.  Staying below ``rtol`` needs

    .. math:: h \;\lesssim\; \frac{2\pi|\delta|}{\ln(2/\texttt{rtol})}

    i.e. roughly **one to a few mesh points per broadening width** (``h <~ delta`` for ~1%,
    ``h <~ delta/2`` for machine-negligible error).  A coarser mesh makes a *weight deficit a
    quadrature artifact rather than lost spectral weight* — this check reports that cause
    distinctly so the ``weight_*`` checks are not misread.

    Args:
        mesh: Real-frequency grid (any ordering; the largest gap is used as ``h``).
        delta: Lorentzian broadening (HWHM).  ``0`` ⇒ not applicable (e.g. Matsubara).
        rtol: The weight-check tolerance this density must support.

    Returns:
        Diagnostic: ``value``/``threshold`` are ``h/|delta|`` and its maximum; ``WARN`` when
        the mesh is too coarse for ``rtol`` (``OK`` when fine or when ``delta == 0``).
    """
    mesh = np.real(np.asarray(mesh))
    d = abs(float(delta)) if delta is not None else 0.0
    if d == 0.0 or mesh.size < 2:
        return Diagnostic("mesh_density", Severity.OK, 0.0, float("inf"), "broadening 0 / single point: not applicable")
    h = float(np.max(np.diff(np.sort(mesh))))
    h_over_d = h / d
    h_max_over_d = 2.0 * np.pi / np.log(2.0 / rtol)
    ok = h_over_d <= h_max_over_d
    return Diagnostic(
        name="mesh_density",
        severity=Severity.OK if ok else Severity.WARN,
        value=h_over_d,
        threshold=h_max_over_d,
        message=(
            f"mesh resolves the broadening (h/delta={h_over_d:.2f})"
            if ok
            else f"mesh too coarse for the broadening (h/delta={h_over_d:.2f}); weight checks are quadrature-limited"
        ),
        suggestion=(
            "" if ok else f"refine omega_mesh to h/delta <~ {h_max_over_d:.2f} (~2-3 points per broadening delta)"
        ),
    )


def check_integrated_weight(
    G_side, r_list, es, e0, tau, mesh, label: str, delta=None, rtol: float = 2e-2
) -> Diagnostic:
    r"""Spectral-weight (zeroth moment) conservation = real-frequency mesh coverage.

    The continued fraction reproduces the zeroth moment
    :math:`m_0 = \operatorname{tr}(r^\dagger r)` *exactly* for any number of Lanczos blocks,
    so :math:`\int -\operatorname{Im}\operatorname{tr}G(\omega)/\pi\,d\omega` recovered on the
    mesh equals the (Boltzmann-weighted) seed moment **iff the mesh covers the spectral
    support**.  A deficit therefore flags a too-narrow ``omega_mesh`` (poles lie outside it),
    not under-convergence.

    Args:
        G_side: Thermally-averaged (un-normalized, no ``1/Z``) one-sided GF on ``mesh``,
            shape ``(len(mesh), n, n)`` (addition ``G_IPS`` or removal ``G_PS``).
        r_list: Matching per-state QR seed factors.
        es, e0, tau: Thermal-state energies, ground energy, temperature scale.
        mesh: Real-frequency grid the GF was evaluated on (ascending).
        label: ``"add"`` or ``"rem"`` for the message.
        delta: Lorentzian broadening; when given, the threshold is relaxed to the trapezoidal
            quadrature floor ``2*exp(-2*pi*|delta|/h)`` so a coarse mesh does not masquerade as
            lost spectral weight (the cause is reported by :func:`check_mesh_density`).
        rtol: Allowed relative weight deficit (when the mesh is fine enough to resolve it).

    Returns:
        Diagnostic: ``OK`` if the recovered weight matches the seed moment, else ``WARN``.
    """
    mesh = np.real(np.asarray(mesh))
    order = np.argsort(mesh)
    # A one-sided spectral function is sign-definite, but the removal GF (G_PS) is evaluated
    # with -delta, which flips the sign of Im. Take the magnitude of the integral so the same
    # zeroth-moment test works for both the addition (+delta) and removal (-delta) sides.
    spectral = np.imag(np.trace(np.asarray(G_side), axis1=1, axis2=2)) / np.pi
    recovered = abs(float(np.trapezoid(spectral[order], mesh[order])))
    expected = _thermal_seed_weight(r_list, es, e0, tau)
    deficit = (expected - recovered) / max(abs(expected), 1e-30)
    # Do not flag a deficit that is within the trapezoidal-quadrature error of the broadening:
    # that is a mesh-density artifact (see check_mesh_density), not off-mesh weight loss.
    eff = max(rtol, _trapezoid_lorentzian_error(mesh, delta))
    ok = deficit <= eff
    return Diagnostic(
        name=f"weight_{label}",
        severity=Severity.OK if ok else Severity.WARN,
        value=float(deficit),
        threshold=eff,
        message=(f"{label} spectral weight conserved on mesh" if ok else f"{label} spectral weight lost off-mesh"),
        suggestion="" if ok else "widen the real-frequency window (omega_mesh) to cover the full spectral support",
    )


def check_thermal_weight_cutoff(
    es, e0, tau, n_returned: int, num_wanted=None, boltzmann_floor: float = 1e-3
) -> Diagnostic:
    r"""Detect a truncated thermal ensemble from the highest retained state's weight.

    The eigensolver keeps states up to
    :math:`\Delta E_{\mathrm{cut}}=-\tau\ln(\epsilon_{\mathrm{B}})` (design weight
    :math:`\epsilon_{\mathrm{B}}=\texttt{BOLTZMANN\_DESIGN\_WEIGHT}`).  If the window had been
    filled, the highest state would carry weight :math:`\approx\epsilon_{\mathrm{B}}`.  A
    *much larger* highest-state weight together with ``n_returned == num_wanted`` (the
    energy cut never bound — the solver simply ran out of requested states) means thermal
    states beyond what was computed were dropped.

    Args:
        es: Retained thermal-state energies.
        e0: Ground-state energy (``min(es)``).
        tau: Temperature scale.
        n_returned: Number of states the eigensolver actually returned.
        num_wanted: Number of states that were requested.
        boltzmann_floor: Highest-state weight above which truncation is suspected.

    Returns:
        Diagnostic: ``WARN`` + ``needs_more_states`` when truncation is suspected.
    """
    es = np.real(np.asarray(es))
    w_highest = float(np.exp(-(np.max(es) - e0) / tau)) if len(es) else 0.0
    # The energy cut did not bind (the solver returned everything it was asked for, so more
    # states may lie within the window). Only decidable when ``num_wanted`` is known: if the
    # cut had bound, the solver would have returned *fewer* than requested.
    window_unfilled = num_wanted is not None and n_returned >= num_wanted
    truncated = (w_highest > boltzmann_floor) and window_unfilled
    return Diagnostic(
        name="thermal_cut",
        severity=Severity.WARN if truncated else Severity.OK,
        value=w_highest,
        threshold=boltzmann_floor,
        message=(
            "highest thermal state has negligible weight"
            if not truncated
            else "highest thermal state still thermally relevant (ensemble truncated)"
        ),
        suggestion="" if not truncated else "increase num_wanted so the energy window -tau*ln(eps) is fully populated",
        needs_more_states=truncated,
    )


def check_lanczos_convergence(converged: bool, d_g: float, n_blocks: int, max_iter: int) -> Diagnostic:
    r"""Surface the block-Lanczos Green's-function convergence-monitor result.

    The monitor stops when the relative change of the resolvent on a frozen mesh drops below
    its tolerance.  Reaching ``max_iter`` without that (``converged is False``) means the
    spectral function is not fully resolved.

    Args:
        converged: Whether the GF Lanczos reached its convergence tolerance.
        d_g: Last relative-change value reported by the monitor (``nan`` if unavailable).
        n_blocks: Number of Lanczos blocks actually run.
        max_iter: The block cap that was in effect.

    Returns:
        Diagnostic: ``OK`` if converged, else ``WARN`` + ``needs_more_iterations``.
    """
    value = float(d_g) if d_g is not None and np.isfinite(d_g) else float("nan")
    return Diagnostic(
        name="lanczos",
        severity=Severity.OK if converged else Severity.WARN,
        value=value,
        threshold=float("nan"),
        message=(
            f"converged in {n_blocks} block(s)"
            if converged
            else f"not converged at max_iter={max_iter} ({n_blocks} blocks)"
        ),
        suggestion="" if converged else "raise the Lanczos block cap (max_iter) for this Green's-function solve",
        needs_more_iterations=not converged,
    )


def check_basis_truncation(cap_hit: bool, retained, cap) -> Diagnostic:
    r"""Surface a Green's-function basis frozen by ``truncation_threshold``.

    When the excited-basis determinant cap is hit, the recurrence continues as an exact
    block Lanczos of the projected operator :math:`PHP` (see
    ``greens_function._CappedBasisProxy``): the result is causal and exact *on the
    retained subspace*, but spectral weight reachable only through the discarded
    determinants is missing.  That is a resource limit, not a solver failure, so this is
    a ``WARN``.  It deliberately does **not** set ``needs_more_states`` — that flag
    triggers the thermal-ensemble eigensolver retry (more eigenstates), which cannot
    widen the retained subspace; the remedy is a larger ``truncation_threshold`` (more
    memory or more ranks).

    Args:
        cap_hit: Whether any of the block's GF solves froze at the cap.
        retained: Global retained determinant count (the largest over the block's solves).
        cap: The ``truncation_threshold`` in effect (``inf`` or ``None`` if uncapped).

    Returns:
        Diagnostic: ``OK`` if the cap never bound, else ``WARN``.
    """
    cap_value = float(cap) if cap is not None else float("inf")
    if not cap_hit:
        return Diagnostic(
            name="basis_cap",
            severity=Severity.OK,
            value=float(retained) if retained is not None else float("nan"),
            threshold=cap_value,
            message="determinant cap not reached",
        )
    return Diagnostic(
        name="basis_cap",
        severity=Severity.WARN,
        value=float(retained),
        threshold=cap_value,
        message=f"GF basis frozen at {int(retained):,} determinants (truncation_threshold)",
        suggestion=(
            "result is exact on the retained subspace; raise truncation_threshold "
            "(more memory per rank or more ranks) to recover the missing spectral weight"
        ),
    )


def check_causality(G, label: str = "G") -> Diagnostic:
    r"""Causality: every diagonal spectral weight must be non-negative.

    The retarded Green's function has :math:`\operatorname{Im}G_{ii}(\omega)\le 0` (the
    advanced/Matsubara conventions flip the sign).  This mirrors the existing
    ``selfenergy.check_greens_function`` but returns a :class:`Diagnostic` (``FAIL``) rather
    than raising, so it composes with the rest of the report; the orchestrator still treats
    a ``FAIL`` as fatal.

    Args:
        G: Green's function of shape ``(len(mesh), n, n)``.
        label: Name used in the message.

    Returns:
        Diagnostic: ``OK`` if causal, ``FAIL`` otherwise.
    """
    G = np.asarray(G)
    diag_imag = np.imag(np.diagonal(G, axis1=1, axis2=2))
    # Retarded convention used throughout the pipeline: Im G_ii <= 0 (matches
    # selfenergy.check_greens_function, which raises on any positive diagonal Im).
    worst_pos = float(np.max(diag_imag)) if diag_imag.size else 0.0
    tol = 1e-8 * max(np.max(np.abs(G)), 1.0)
    causal = worst_pos <= tol
    return Diagnostic(
        name="causality",
        severity=Severity.OK if causal else Severity.FAIL,
        value=max(worst_pos, 0.0),
        threshold=tol,
        message=(f"{label} diagonal Im <= 0 (causal)" if causal else f"{label} has a positive Im on the diagonal"),
        suggestion="" if causal else "result is unphysical; check thermal ensemble / Lanczos convergence above",
    )


def addition_removal_peak_check(*args, **kwargs):
    r"""Deferred: KMS addition/removal peak diagnostic (see module note).

    Would flag truncated eigenstates as sharp peaks in
    :math:`G^{\mathrm{add}}(\omega) - w\,G^{\mathrm{rem}}(\omega)`.  Deferred because the
    detailed-balance relation it relies on holds for a grand-canonical ensemble, whereas
    this code uses a fixed total-N ensemble; the correct fixed-N reweighting must be pinned
    down before implementing it.  Truncation is meanwhile covered by
    :func:`check_thermal_weight_cutoff` and :func:`check_spectral_sum_rule`.
    """
    raise NotImplementedError(
        "addition/removal KMS peak check is deferred pending the fixed-N reweighting formula; "
        "use check_thermal_weight_cutoff for truncation detection."
    )
