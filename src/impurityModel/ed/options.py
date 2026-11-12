"""The option groups the solvers consume: meshes, basis construction, Green's-function kernel, spectra.

Leaf module (numpy and :mod:`config` only), so a front end can parse and validate options without
pulling in any many-body machinery.
"""

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from impurityModel.ed import config

__all__ = [
    "OPTION_SPECS",
    "BasisOptions",
    "Meshes",
    "OptionSpec",
    "SolverOptions",
    "SpectraOptions",
    "describe_options",
    "options_from_attrs",
    "options_to_attrs",
    "parse_option_tokens",
]


@dataclass(frozen=True)
class Meshes:
    """Frequency meshes and the real-axis broadening.

    Attributes
    ----------
    iw : numpy.ndarray or None
        Complex Matsubara mesh :math:`z = i\\nu_n` (the solver evaluates the resolvent at
        ``iw`` directly, so pass ``1j * nu``). ``None`` skips the Matsubara output.
    w : numpy.ndarray or None
        Real frequency mesh (eV). ``None`` skips the real-axis output.
    delta : float
        Smearing above the real axis (HWHM, eV).
    """

    iw: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    delta: float = 0.1


@dataclass(frozen=True)
class BasisOptions:
    """Many-body basis construction: occupation, restrictions and the determinant budget.

    Attributes
    ----------
    nominal_occ : dict or int
        Nominal impurity occupation. A dict keyed by the model's impurity groups, or a
        scalar/total that the solver distributes by energetic filling. For a multi-shell model
        (``ImpurityModel.shell_ls``) the keys are shell indices ``a`` and the values the nominal
        occupation of shell ``a``.
    mixed_valence : None, bool, int, str or dict
        The charge-sector ensemble switch: ``"auto"``, ``"on"``/``True`` or ``"off"``/``False`` (``None`` takes
        the driver's default). A legacy int / ``{group: int}`` is read as off (0) or on. See
        :func:`~impurityModel.ed.sector_ensemble.resolve_mixed_valence`.
    dN : int or None
        Allowed impurity occupation window (+-dN) for the Green's-function excited bases.
    truncation_threshold : int, float or None
        Global cap on the number of determinants per basis. ``None`` derives the cap from
        available per-rank memory; ``numpy.inf`` disables capping.
    chain_restrict : bool
        Whether to apply chain occupation restrictions.
    occ_cutoff : float
        Occupation cutoff.
    slater_weight_min : float
        Minimum Slater-determinant weight retained.
    tau : float
        Fundamental temperature ``k_B * T`` (eV).
    cost_budget : float or None
        ``alpha`` of the hopping-cost budget: a weighted restriction whose per-bath-orbital cost is
        ``round(8 * (d(o) - d_min))`` on the coupling distance ``d`` (``-log`` of the best-path product of
        hoppings), bounded by ``round(COST_KAPPA * alpha * -log(slater_weight_min))`` and widened for the
        Green's-function bases to contain the whole ground-state basis. ``None`` (default) disables it.
    e_pt2_tol : float or None
        Residual Epstein-Nesbet PT2 energy the ground-state CIPSI expansion is converged to (the
        *summed* PT2 contribution of every determinant left out, which is what the energy error
        follows; see ``doc/plans/cipsi_pt2_convergence.md``). ``None`` uses
        :data:`groundstate.GS_E_PT2_TOL` (``1e-8``). The double-counting search's charge-sector
        solves inherit it unless the double-counting line sets their own.
    de2_min : float or None
        Optional per-determinant Epstein-Nesbet PT2 floor for the same expansion. It *loosens* a
        solve -- it bounds each refused determinant, not their sum -- and an expansion it stops
        short of ``e_pt2_tol`` warns with the residual it left. ``None`` uses
        :data:`groundstate.GS_DE2_MIN` (0, no floor). Inherited by the double-counting search
        like ``e_pt2_tol``.
    interaction_merges_blocks : bool
        ``True`` (default): an interaction that couples one-body blocks merges them into one
        Green's-function block. ``False``: the one-body blocks are authoritative and such
        couplings are neglected (reported). Set it when the one-body part was deliberately
        truncated to block-diagonal form. Only the RSPt interface sets it; the standalone scripts
        (spectra, selfenergy, susceptibility) and input files keep the default.
    """

    nominal_occ: Any
    mixed_valence: Any = None
    dN: Optional[int] = None
    truncation_threshold: Optional[Union[int, float]] = None
    chain_restrict: bool = True
    occ_cutoff: float = 1e-12
    slater_weight_min: float = float(np.sqrt(np.finfo(float).eps))
    tau: float = 0.002
    cost_budget: Optional[float] = None
    e_pt2_tol: Optional[float] = None
    de2_min: Optional[float] = None
    interaction_merges_blocks: bool = True

    def __post_init__(self):
        if self.e_pt2_tol is not None and not self.e_pt2_tol > 0:
            raise ValueError(f"e_pt2_tol must be positive (or None for the default), got {self.e_pt2_tol}")
        if self.de2_min is not None and not self.de2_min >= 0:
            raise ValueError(f"de2_min must be non-negative (or None for the default), got {self.de2_min}")
        if self.cost_budget is not None and not self.cost_budget > 0:
            raise ValueError(f"cost_budget must be positive (or None to disable it), got {self.cost_budget}")


#: The Green's-function support options a run uses when the options leave them unset (``gf_method="lanczos"``):
#: importance-pruned growth at this tolerance and this many swap-refinement rounds (a unit the cap does not bind is
#: not refined). ``0`` switches either off; archives that predate the options read as ``0``.
DEFAULT_GF_PRUNE_TOL = 1e-3
DEFAULT_GF_REFINE_ROUNDS = 3


@dataclass(frozen=True)
class SolverOptions:
    """Green's-function kernel and eigensolver knobs.

    Attributes
    ----------
    dense_cutoff : int
        Use a dense eigensolver below this matrix size.
    sparse_green : bool
        Whether the Green's function uses the sparse block-Lanczos path.
    gf_method : {"lanczos", "bicgstab"}
        Green's-function kernel. See :func:`impurityModel.ed.greens_function.get_Greens_function`.
    gf_admission : {"all", "outer"} or None
        Basis-growth policy of the per-frequency ``"bicgstab"`` kernel: ``"all"`` admits everything
        the solver produces, ``"outer"`` admits only determinants whose residual scores above
        ``gf_admit_tol`` (:mod:`impurityModel.ed.gf_admission`), solving on a frozen basis between
        rounds, and records a measured error bound. ``None`` (default) is ``"all"``. ``"outer"`` needs
        ``gf_method="bicgstab"``.
    gf_admit_tol : float or None
        Admission threshold of ``gf_admission="outer"``, relative to the seed norm (amplitude
        scorer). ``None`` takes 1e-4; tighter admits more and is
        more accurate, and the error bound reports what was left out.
    gf_tol : float or None
        Convergence tolerance of the block-Lanczos Green's function (relative change of ``G``
        between checks) on every evaluation axis. ``None`` (default):
        ``max(slaterWeightMin**2, 1e-9)``. ``gf_method="lanczos"`` only.
    gf_real_tol : float or None
        The same tolerance on the real-frequency axis only; ``None`` (default):
        the ``gf_tol`` value. A looser real-axis tolerance shortens the recurrence without
        touching the Matsubara accuracy. In RSPt the real-axis self-energy still feeds the next
        bath fit, so loosen it moderately (e.g. ``1e-6``) rather than drop it.
        ``gf_method="lanczos"`` only.
    gf_min_weight : float or None
        Drop thermal states whose normalised Boltzmann weight is below this from the Green's
        function, the self-energy and its moments (whole degenerate manifolds only; the ground
        manifold is always kept, and the remaining weights are renormalised). ``None`` (default)
        keeps every state of the eigensolver's energy window ``-tau*ln(1e-4)``, each of which costs
        a full set of Green's-function work units whatever its weight. The ground-state report and
        the returned ``thermal_rho`` / ``rhos`` keep the whole ensemble: they describe the thermal
        state itself, not the subset the Green's function was built from.
    gf_settle : bool or None
        Whether a sparse block-Lanczos Green's-function unit may stop admitting new determinants once its
        frontier has fallen below the cutoff-scaled threshold (the support settle,
        :class:`impurityModel.ed.gf_primitives.FreezeRule`) and carry on as the exact recurrence on what it
        has. ``None`` (default) is on. It drops the determinants whose next-block amplitude stayed at or below
        ``gf_settle_ratio x slater_weight_min``; switching it off keeps every unit growing until it closes or hits
        a cap, which is far slower on a unit
        whose support trickles. ``gf_method="lanczos"`` only.
    gf_settle_ratio : float or None
        The settle threshold as a **multiple of** ``slater_weight_min`` (not a fraction of the seed norm, unlike
        ``gf_admit_tol``): a unit settles when, over 16 blocks, the largest amplitude the normalised next block
        carries on a determinant reached for the first time stays at or below ``ratio x slater_weight_min`` and
        is not growing. ``1`` settles when every excluded determinant would have been pruned from the next
        block anyway, larger settles sooner, ``0`` is closure only (no new determinant at all). ``None``
        (default): 3. Needs the settle on; ``gf_method="lanczos"`` only.
    gf_settle_retry : bool or None
        If the real-axis self-energy fails the causality check and the support settle can plausibly be blamed (the
        block has settled units, none held by the cap or the memory guard, and they did not already run closure-only),
        solve those units again once with the closure-only recurrence (``gf_settle_ratio = 0``) and judge the new
        ``G``. The attribution is printed either way; the retry never falls back to "settle off". ``None`` (default)
        is **off**: the closure-only re-solve can take as long as an unsettled Green's function (hours on a large
        unit). Needs the settle on; ``gf_method="lanczos"`` only.
    gf_prune_tol : float or None
        Importance-pruned growth of the sparse block-Lanczos Green's function: a step admits a new determinant only
        if its largest amplitude on the normalised step block exceeds this (the ``mass`` prune rule reads it as a
        fraction of the step's frontier weight instead), so a capped unit reaches a greater H-depth where the
        amplitude is. A deferred determinant is not banned: it re-enters when a stronger path reaches it. ``None``
        (default): :data:`DEFAULT_GF_PRUNE_TOL` (1e-3) for ``gf_method="lanczos"``, or off when a settle option
        is set; ``0``: off. Exclusive with the support settle: it switches the settle off, and ``gf_settle=True``,
        ``gf_settle_ratio`` and ``gf_settle_retry`` are refused with an explicit value.
    gf_first_shell_tol : float or None
        With ``gf_prune_tol``: the same cut for the seeds' first H-shell, which is kept whole by default (the
        moments of G through ``H^2``, hence the self-energy tail, stay exact). ``>= 0``; ``None``:
        0 by default. Needs ``gf_prune_tol``.
    gf_growth_tol : float or None
        With pruned growth: the relative-change tolerance the growth pass converges to (looser than the full
        ``gf_tol``). The restart from the seeds on the final support, which pruning always needs, then converges
        ``G`` to the full tolerance, so a larger value ends the matrix-free growth earlier and the support it
        finds is the one that earlier tolerance saw. ``None`` (default): the growth converges fully. In ``(0, 1)``.
    gf_refine_rounds : int or None
        Swap refinement of a unit the determinant cap froze: up to this many rounds of scoring the retained and the
        discovered determinants on the discovered matrix and swapping the least important retained for the most
        important discovered ones at a fixed cap, then one recurrence on the final support. ``None`` (default):
        :data:`DEFAULT_GF_REFINE_ROUNDS` (3) for ``gf_method="lanczos"``; ``0``: off.
        A unit the cap does not bind, one the memory guard froze, and a complex Hamiltonian are left alone. Needs
        the discovered matrix (more memory than the retained one) and costs minutes per capped unit.
        ``gf_method="lanczos"`` only.
    gf_refine_tol : float or None
        Stop refining when the estimated remaining first-order gain (the summed Epstein-Nesbet scores of the swap
        set) is below this fraction of the largest ``|G|`` on the scoring shifts. ``None``: 1e-3. Needs
        ``gf_refine_rounds``.
    gf_refine_swap : float or None
        The largest fraction of the cap one round swaps, in (0, 0.5]. ``None``: 0.1. Needs ``gf_refine_rounds``.
    sigma_causality_tol : float or None
        Relative tolerance of the self-energy causality check (per diagonal orbital, ``Im Sigma_ii <= 0``).
        A violation below it is printed as a warning and tolerated; above it the run fails. ``None``: 1e-3;
        ``0`` always fails on a violation.
    sigma_method : {"dyson"}
        Self-energy estimator (:mod:`impurityModel.ed.sigma_estimators`): which operator family
        the Green's-function engine resolves and how the self-energy is read off it. Only the
        Dyson estimator exists today; the field is the extension point, not yet a user knob.
    """

    dense_cutoff: int = config.DEFAULT_DENSE_CUTOFF
    sparse_green: bool = True
    gf_method: str = "lanczos"
    sigma_method: str = "dyson"
    gf_admission: Optional[str] = None
    gf_admit_tol: Optional[float] = None
    gf_tol: Optional[float] = None
    gf_real_tol: Optional[float] = None
    gf_min_weight: Optional[float] = None
    sigma_causality_tol: Optional[float] = None
    gf_settle: Optional[bool] = None
    gf_settle_ratio: Optional[float] = None
    gf_settle_retry: Optional[bool] = None
    gf_prune_tol: Optional[float] = None
    gf_first_shell_tol: Optional[float] = None
    gf_growth_tol: Optional[float] = None
    gf_refine_rounds: Optional[int] = None
    gf_refine_tol: Optional[float] = None
    gf_refine_swap: Optional[float] = None

    def __post_init__(self):
        # One place for every front-end (RSPt solver line, TOML, CLI, archive replay): a bad
        # combination is refused here rather than silently ignored two layers down.
        if self.gf_method not in config.GF_METHODS:
            raise ValueError(
                f"Unknown gf_method {self.gf_method!r}; expected one of {', '.join(map(repr, config.GF_METHODS))}"
            )
        if self.gf_admission is not None and self.gf_admission not in config.GF_ADMISSIONS:
            raise ValueError(
                f"gf_admission {self.gf_admission!r}: expected one of {', '.join(map(repr, config.GF_ADMISSIONS))}"
            )
        if self.gf_admit_tol is not None and not self.gf_admit_tol > 0.0:
            raise ValueError(f"gf_admit_tol must be positive (got {self.gf_admit_tol!r}); omit it for the default")
        if self.gf_admission == "outer" and self.gf_method != "bicgstab":
            raise ValueError(
                f"gf_admission='outer' is the basis-growth policy of the per-frequency kernel and needs "
                f"gf_method='bicgstab' (got {self.gf_method!r})"
            )
        if self.gf_admit_tol is not None and self.gf_admission != "outer":
            raise ValueError("gf_admit_tol only applies to gf_admission='outer' (it is set, but the policy is not)")
        for name in ("gf_tol", "gf_real_tol"):
            value = getattr(self, name)
            if value is None:
                continue
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1) (got {value!r}); omit it for the default")
            if self.gf_method != "lanczos":
                raise ValueError(
                    f"{name} is the block-Lanczos convergence tolerance; gf_method={self.gf_method!r} ignores it"
                )
        if self.gf_min_weight is not None and not 0.0 < self.gf_min_weight < 1.0:
            raise ValueError(
                f"gf_min_weight must lie in (0, 1) (got {self.gf_min_weight!r}); omit it to keep every state"
            )
        if self.sigma_causality_tol is not None and not self.sigma_causality_tol >= 0.0:
            raise ValueError(f"sigma_causality_tol must be >= 0 (got {self.sigma_causality_tol!r})")
        for name in ("gf_settle", "gf_settle_ratio", "gf_settle_retry"):
            if getattr(self, name) is not None and self.gf_method != "lanczos":
                raise ValueError(f"{name} applies only to gf_method='lanczos'; remove it or switch gf_method")
        if self.gf_settle_ratio is not None and 0.0 < self.gf_settle_ratio < 1.0:
            warnings.warn(
                f"gf_settle_ratio {self.gf_settle_ratio:g} is a MULTIPLE of slater_min, not a fraction: values below 1 "
                "are nearly closure-only and make the Green's function several times slower",
                stacklevel=2,
            )
        if self.gf_settle_ratio is not None and not (
            math.isfinite(self.gf_settle_ratio) and self.gf_settle_ratio >= 0.0
        ):
            raise ValueError(
                f"gf_settle_ratio must be >= 0 (got {self.gf_settle_ratio!r}); 0 is closure only, "
                "omit it for the default"
            )
        for name in ("gf_prune_tol", "gf_first_shell_tol", "gf_growth_tol"):
            value = getattr(self, name)
            if value is None or (value == 0.0 and name == "gf_prune_tol"):
                continue  # 0 is the explicit off: it is also what an archive without the option reads as
            if self.gf_method != "lanczos":
                raise ValueError(f"{name} applies only to gf_method='lanczos'; remove it or switch gf_method")
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite (got {value!r}); omit it for the default")
        if self.gf_refine_rounds is not None and (
            isinstance(self.gf_refine_rounds, bool)
            or not isinstance(self.gf_refine_rounds, int)
            or self.gf_refine_rounds < 0
        ):
            raise ValueError(f"gf_refine_rounds must be a non-negative integer (got {self.gf_refine_rounds!r})")
        refining = bool(self.gf_refine_rounds)
        if (self.gf_refine_tol is not None or self.gf_refine_swap is not None) and self.gf_refine_rounds == 0:
            raise ValueError("gf_refine_tol / gf_refine_swap need gf_refine_rounds > 0 (refinement is switched off)")
        if refining and self.gf_method != "lanczos":
            raise ValueError("gf_refine_rounds applies only to gf_method='lanczos'; remove it or switch gf_method")
        if self.gf_refine_tol is not None and not (
            math.isfinite(self.gf_refine_tol) and 0.0 < self.gf_refine_tol < 1.0
        ):
            raise ValueError(f"gf_refine_tol must be in (0, 1) (got {self.gf_refine_tol!r})")
        if self.gf_refine_swap is not None and not (0.0 < self.gf_refine_swap <= 0.5):
            raise ValueError(f"gf_refine_swap must be in (0, 0.5] (got {self.gf_refine_swap!r})")
        if self.gf_growth_tol is not None and not 0.0 < self.gf_growth_tol < 1.0:
            raise ValueError(f"gf_growth_tol must lie in (0, 1) (got {self.gf_growth_tol!r}); omit it for off")
        if self.gf_growth_tol is not None and self.gf_prune_tol == 0.0:
            raise ValueError("gf_growth_tol shortens the growth of pruned growth; pruning is switched off")
        if self.gf_first_shell_tol is not None and self.gf_prune_tol == 0.0:
            raise ValueError("gf_first_shell_tol relaxes the first shell of pruned growth; pruning is switched off")
        if self.gf_prune_tol and (self.gf_settle is True or self.gf_settle_ratio is not None or self.gf_settle_retry):
            raise ValueError(
                "gf_prune_tol is exclusive with the support settle (gf_settle, gf_settle_ratio, gf_settle_retry): "
                "pruned growth switches the settle off; remove the settle options"
            )
        if self.gf_settle_retry and self.gf_settle is False:
            raise ValueError(
                "gf_settle_retry on contradicts gf_settle off: the retry re-solves settled units; remove one of the two"
            )
        if self.gf_settle_ratio is not None and self.gf_settle is False:
            raise ValueError(
                "gf_settle_ratio only applies with the support settle on (it is set, but gf_settle is off); "
                "remove one of the two"
            )

    @property
    def resolved_gf_prune_tol(self):
        """The pruned-growth tolerance a run uses; ``None`` leaves pruning off.

        An unset ``gf_prune_tol`` is :data:`DEFAULT_GF_PRUNE_TOL` for ``gf_method="lanczos"``, except where the
        options ask for the support settle (``gf_settle`` on, ``gf_settle_ratio``, ``gf_settle_retry``), which
        pruning replaces: the explicit request wins and pruning is off (``0.0``).
        """
        if self.gf_method != "lanczos":
            return self.gf_prune_tol
        if self.gf_prune_tol is not None:
            return self.gf_prune_tol
        if self.gf_settle is True or self.gf_settle_ratio is not None or self.gf_settle_retry:
            return 0.0
        return DEFAULT_GF_PRUNE_TOL

    @property
    def resolved_gf_refine_rounds(self):
        """The refinement rounds a run uses: :data:`DEFAULT_GF_REFINE_ROUNDS` when unset for ``"lanczos"``, else 0."""
        if self.gf_refine_rounds is not None:
            return self.gf_refine_rounds
        return DEFAULT_GF_REFINE_ROUNDS if self.gf_method == "lanczos" else 0


@dataclass(frozen=True)
class SpectraOptions:
    """Spectroscopy meshes, broadenings, polarizations and projectors for the spectra driver.

    Every field defaults to ``None`` for the arrays the driver fills in with its standard
    grids/polarizations (so an API caller can override just the pieces it cares about) and to
    the historical literal for the scalars. The frequency meshes are resolved once inside
    :func:`impurityModel.ed.get_spectra.run_spectra`.

    Attributes
    ----------
    w : numpy.ndarray or None
        XAS / PES / XPS energy mesh (eV). ``None`` -> ``linspace(-25, 25, 3001)``.
    delta : float
        Core-hole-lifetime broadening (HWHM, eV).
    epsilons : sequence or None
        XAS polarization vectors. ``None`` -> the Cartesian ``x, y, z`` triple.
    wLoss : numpy.ndarray or None
        RIXS energy-loss mesh. ``None`` -> ``linspace(-2, 12, 4000)``.
    wIn : numpy.ndarray or None
        RIXS incoming-energy mesh. ``None`` with ``deltaRIXS > 0`` -> ``linspace(-10, 20, 50)``;
        an empty mesh disables RIXS.
    deltaRIXS, deltaNIXS : float
        Excited-state-lifetime broadenings for RIXS / NIXS (HWHM, eV). ``deltaRIXS <= 0``
        disables RIXS.
    epsilonsRIXSin, epsilonsRIXSout : sequence or None
        RIXS in/out polarization vectors. ``None`` -> the Cartesian ``x, y, z`` triple.
    qsNIXS : sequence or None
        NIXS momentum-transfer vectors. ``None`` -> the driver's two default ``|q|``.
    liNIXS, ljNIXS : int
        Angular momenta of the final/initial orbitals in the NIXS excitation.
    radial : tuple or None
        ``(radial_mesh, Ri, Rj)`` for the NIXS radial integral. ``None`` skips NIXS.
    auto_block_structure : bool
        Derive the block structure from the
        hybridization-dressed impurity matrix instead of the hand-coded one.
    XAS_projectors, RIXS_projectors : object or None
        Optional transition projectors; ``None`` computes the full polarization tensor.
    rixs_adaptive_tol : float or None
        Stop tolerance of the greedy adaptive sampling of the RIXS incoming-energy grid
        (:func:`impurityModel.ed.rixs.calc_tensor_map`). ``None`` (default) solves the whole grid.
    pes, xps, xas, rixs, nixs : bool or None
        Which spectra to compute, one switch per technique. ``None`` (the default) reproduces
        the historical behaviour exactly: PES, XPS and XAS unconditionally, RIXS when
        ``deltaRIXS > 0`` and ``wIn`` is non-empty, NIXS when ``radial`` was supplied. Setting
        one is what lets a caller ask for, say, XAS alone -- which was previously impossible,
        since PES and XPS had no switch and the others were disabled only as a side effect of
        a non-positive broadening, an empty mesh or withheld data.
    """

    w: Optional[np.ndarray] = None
    delta: float = 0.2
    epsilons: Any = None
    wLoss: Optional[np.ndarray] = None
    wIn: Optional[np.ndarray] = None
    deltaRIXS: float = 0.050
    deltaNIXS: float = 0.100
    epsilonsRIXSin: Any = None
    epsilonsRIXSout: Any = None
    qsNIXS: Any = None
    liNIXS: int = 2
    ljNIXS: int = 2
    radial: Any = None
    auto_block_structure: bool = True
    XAS_projectors: Any = None
    RIXS_projectors: Any = None
    rixs_adaptive_tol: Optional[float] = None
    pes: Optional[bool] = None
    xps: Optional[bool] = None
    xas: Optional[bool] = None
    rixs: Optional[bool] = None
    nixs: Optional[bool] = None


# ----------------------------------------------------------------------------------------------
# The registry: one row per user-settable option.  The solver-line grammar, the option header of
# an output file and the archive attribute record are all derived from it, so adding an option
# is one row here plus the dataclass field.
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionSpec:
    """How one :class:`BasisOptions` / :class:`SolverOptions` field is spelled, read and shown.

    Attributes
    ----------
    name : str
        Field name on the dataclass.
    group : {"basis", "solver"}
        Which dataclass owns the field.
    kind : {"int", "float", "onoff", "enum", "cap", "bool"}
        How a value is read from text.  ``"bool"`` is a field that has only flag spellings.
    token : str or None
        The word that takes a value (``None`` for a field that has no valued spelling).
    aliases : tuple of str
        Other words for the same valued option.
    flags : dict
        ``{word: value}``: words that set the field without a value (``no_mv`` -> ``False``).
    attr : str
        The key in the archive attribute record (historical name where it differs from ``name``).
    label : str
        The row label of the option header.
    unset : str or None
        What the header shows for ``None``; ``None`` shows the raw value.
    choices : tuple of str
        The words an ``"enum"`` takes; ``auto_none`` is read as ``None``.
    """

    name: str
    group: str
    kind: str
    label: str
    token: Optional[str] = None
    aliases: tuple = ()
    flags: dict = field(default_factory=dict)
    attr: Optional[str] = None
    unset: Optional[str] = None
    choices: tuple = ()
    auto_none: Optional[str] = None
    positive: bool = False

    @property
    def attr_name(self):
        return self.attr or self.name

    @property
    def words(self):
        return () if self.token is None else (self.token, *self.aliases)


def _spec(name, group, kind, label, **kw):
    return OptionSpec(name, group, kind, label, **kw)


#: In header print order: the Green's-function kernel first, then the basis construction.
OPTION_SPECS = (
    _spec("interaction_merges_blocks", "basis", "bool", "Interaction merges GF blocks"),
    _spec("dense_cutoff", "solver", "int", "Dense matrix size cutoff", token="dense_cutoff"),
    _spec("sparse_green", "solver", "bool", "Sparse Green's function", flags={"dense_green": False}),
    _spec("gf_method", "solver", "enum", "Green's function method", token="gf_method", choices=config.GF_METHODS),
    _spec(
        "gf_admission",
        "solver",
        "enum",
        "GF basis admission",
        token="gf_admission",
        choices=config.GF_ADMISSIONS,
        auto_none="auto",
        unset="auto",
    ),
    _spec(
        "gf_admit_tol",
        "solver",
        "float",
        "GF admission threshold",
        token="gf_admit_tol",
        unset="default",
        positive=True,
    ),
    _spec("gf_settle", "solver", "onoff", "GF support settle", token="gf_settle", unset="on"),
    _spec(
        "gf_settle_ratio",
        "solver",
        "float",
        "GF settle ratio (x slater_min)",
        token="gf_settle_ratio",
        unset="default",
    ),
    _spec("gf_settle_retry", "solver", "onoff", "GF settle retry", token="gf_settle_retry", unset="off"),
    _spec(
        "gf_prune_tol",
        "solver",
        "float",
        "GF pruned growth threshold",
        token="gf_prune_tol",
        unset="default (1e-3)",
    ),
    _spec(
        "gf_first_shell_tol",
        "solver",
        "float",
        "GF first-shell threshold",
        token="gf_first_shell_tol",
        unset="whole shell",
    ),
    _spec("gf_growth_tol", "solver", "float", "GF growth tolerance", token="gf_growth_tol", unset="full"),
    _spec(
        "gf_refine_rounds",
        "solver",
        "int",
        "GF swap refinement rounds",
        token="gf_refine_rounds",
        unset="default (3)",
    ),
    _spec(
        "gf_refine_tol",
        "solver",
        "float",
        "GF refinement stop (x max|G|)",
        token="gf_refine_tol",
        unset="default",
    ),
    _spec(
        "gf_refine_swap",
        "solver",
        "float",
        "GF refinement swap fraction",
        token="gf_refine_swap",
        unset="default",
    ),
    _spec("cost_budget", "basis", "float", "Hopping-cost budget", token="cost_budget", unset="off"),
    _spec("gf_tol", "solver", "float", "GF convergence tolerance", token="gf_tol", unset="default"),
    _spec("gf_real_tol", "solver", "float", "GF real-axis tolerance", token="gf_real_tol", unset="default"),
    _spec("gf_min_weight", "solver", "float", "GF minimal thermal weight", token="gf_min_weight", unset="off"),
    _spec(
        "sigma_causality_tol",
        "solver",
        "float",
        "Self-energy causality tolerance",
        token="sigma_causality_tol",
        unset="default (1e-3)",
    ),
    _spec("occ_cutoff", "basis", "float", "Occupation cutoff", token="occ_cutoff"),
    _spec("dN", "basis", "int", "dN", token="dn"),
    _spec(
        "mixed_valence",
        "basis",
        "bool",
        "Mixed valence allowed",
        flags={"no_mv": False},
        attr="mv",
        unset="auto (default)",
    ),
    _spec("chain_restrict", "basis", "bool", "Chain occ. restrictions", flags={"no_chain_restrict": False}),
    _spec(
        "slater_weight_min",
        "basis",
        "float",
        "Minimal Slater weight",
        token="slater_min",
        aliases=("slater_weight_min",),
        attr="slater_min",
    ),
    _spec(
        "e_pt2_tol",
        "basis",
        "float",
        "GS residual PT2 tolerance",
        token="e_pt2",
        aliases=("e_pt2_tol",),
        unset="default",
        positive=True,
    ),
    _spec("de2_min", "basis", "float", "GS per-determinant PT2 floor", token="de2_min", unset="default"),
    _spec(
        "truncation_threshold",
        "basis",
        "cap",
        "Determinant cap (truncation_threshold)",
        token="truncation_threshold",
        unset="auto",
    ),
)

#: Fields that were never options: ``nominal_occ`` and ``tau`` are the caller's, the rest of
#: ``BasisOptions`` is reached through :data:`OPTION_SPECS`.
_SPEC_BY_WORD = {}
for _s in OPTION_SPECS:
    for _w in _s.words:
        _SPEC_BY_WORD[_w] = (_s, None)
    for _w, _v in _s.flags.items():
        _SPEC_BY_WORD[_w] = (_s, _v)
del _s, _w


def _read_number(word, text):
    # Fortran writes 1d-2 for 1e-2
    try:
        return float(re.sub(r"(?<=[0-9.])[dD](?=[+-]?\d)", "e", text))
    except ValueError:
        raise ValueError(f"'{word}' needs a number, got {text!r}") from None


def _read_value(spec, word, text):
    kind = spec.kind
    if kind == "float":
        value = _read_number(word, text)
        if spec.positive and not value > 0:
            raise ValueError(f"'{word}' must be positive, got {text!r}")
        return value
    if kind == "int":
        try:
            return int(text)
        except ValueError:
            # 2e3 style
            number = _read_number(word, text)
            if number != int(number):
                raise ValueError(f"'{word}' needs an integer, got {text!r}") from None
            return int(number)
    if kind == "onoff":
        lowered = text.lower()
        if lowered not in ("on", "off"):
            raise ValueError(f"'{word}' must be on or off, got {text!r}")
        return lowered == "on"
    if kind == "enum":
        lowered = text.lower()
        if spec.auto_none is not None and lowered == spec.auto_none:
            return None
        return lowered  # the dataclass refuses a word it does not know, with the reason
    if kind == "cap":
        from impurityModel.ed.memory_estimate import parse_truncation_threshold

        return parse_truncation_threshold(text)
    raise ValueError(f"'{word}' cannot take a value")


def parse_option_tokens(tokens):
    """Read basis and solver options out of solver-line words.

    Parameters
    ----------
    tokens : sequence of str
        Whitespace-separated words, comments already removed.  Words are case-insensitive.

    Returns
    -------
    basis_kwargs, solver_kwargs : dict
        Keyword arguments for :class:`BasisOptions` / :class:`SolverOptions`; only options that
        appeared are present.
    unused : list of str
        Every word that is not an option, in order.  A caller that shares the line with other
        consumers (the bath fit) hands the remainder on; one that does not should refuse it.

    Raises
    ------
    ValueError
        An option without a value, or with one that cannot be read.
    """
    found = {"basis": {}, "solver": {}}
    unused = []
    i = 0
    while i < len(tokens):
        word = tokens[i].lower()
        entry = _SPEC_BY_WORD.get(word)
        if entry is None:
            unused.append(tokens[i])
        else:
            spec, flag_value = entry
            if word in spec.flags:
                found[spec.group][spec.name] = flag_value
            else:
                if i + 1 >= len(tokens):
                    raise ValueError(f"'{tokens[i]}' on the solver line needs a value")
                found[spec.group][spec.name] = _read_value(spec, tokens[i], tokens[i + 1])
                i += 1
        i += 1
    return found["basis"], found["solver"], unused


def options_to_attrs(basis, solver):
    """Flat ``{attr: value}`` record of the registered options, for the archive."""
    groups = {"basis": basis, "solver": solver}
    return {spec.attr_name: getattr(groups[spec.group], spec.name) for spec in OPTION_SPECS}


def options_from_attrs(attrs):
    """Inverse of :func:`options_to_attrs`: ``(basis_kwargs, solver_kwargs)`` from an attribute record.

    HDF5 cannot store ``None``, so an unset option is an absent attribute; absent keys are left out
    and the dataclass default applies.
    """
    out = {"basis": {}, "solver": {}}
    for spec in OPTION_SPECS:
        if spec.attr_name not in attrs:
            continue
        value = attrs[spec.attr_name]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode()
        out[spec.group][spec.name] = value
    return out["basis"], out["solver"]


def _show(spec, value):
    if value is None:
        return spec.unset if spec.unset is not None else value
    if spec.kind == "onoff":
        return "on" if value else "off"
    if spec.kind == "cap":
        return "unlimited" if not value < float("inf") else f"{int(value):,} (final)"
    if spec.name in ("gf_prune_tol", "gf_refine_rounds") and value == 0:
        return "off"
    return value


def describe_options(basis, solver):
    """``[(label, text)]`` rows for a run header, in print order."""
    groups = {"basis": basis, "solver": solver}
    return [(spec.label, _show(spec, getattr(groups[spec.group], spec.name))) for spec in OPTION_SPECS]
