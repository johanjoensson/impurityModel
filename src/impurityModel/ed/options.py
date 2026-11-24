"""The option groups the solvers consume: meshes, basis construction, Green's-function kernel, spectra.

Leaf module (numpy and :mod:`config` only), so a front end can parse and validate options without
pulling in any many-body machinery.
"""

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union

import numpy as np

from impurityModel.ed import config

__all__ = [
    "OPTION_SPECS",
    "BasisOptions",
    "Meshes",
    "NIXSOptions",
    "OptionSpec",
    "PESOptions",
    "RIXSOptions",
    "SolverOptions",
    "XASOptions",
    "XPSOptions",
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


#: The Green's-function support options a run uses when the options leave them unset (always):
#: importance-pruned growth at this tolerance and this many swap-refinement rounds (a unit the cap does not bind is
#: not refined). ``0`` switches either off. An unset option is an absent archive attribute and reads back as unset, so a replay uses these defaults.
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
    gf_tol : float or None
        Convergence tolerance of the block-Lanczos Green's function (relative change of ``G``
        between checks) on every evaluation axis. ``None`` (default):
        ``max(slaterWeightMin**2, 1e-9)``.
    gf_real_tol : float or None
        The same tolerance on the real-frequency axis only; ``None`` (default):
        the ``gf_tol`` value. A looser real-axis tolerance shortens the recurrence without
        touching the Matsubara accuracy. In RSPt the real-axis self-energy still feeds the next
        bath fit, so loosen it moderately (e.g. ``1e-6``) rather than drop it.
    gf_min_weight : float or None
        Drop thermal states whose normalised Boltzmann weight is below this from the Green's
        function, the self-energy and its moments (whole degenerate manifolds only; the ground
        manifold is always kept, and the remaining weights are renormalised). ``None`` (default)
        keeps every state of the eigensolver's energy window ``-tau*ln(1e-4)``, each of which costs
        a full set of Green's-function work units whatever its weight. The ground-state report and
        the returned ``thermal_rho`` / ``rhos`` keep the whole ensemble: they describe the thermal
        state itself, not the subset the Green's function was built from.
    gf_prune_tol : float or None
        Importance-pruned growth of the sparse block-Lanczos Green's function: a step admits a new determinant only
        if its largest amplitude on the normalised step block exceeds this (the ``mass`` prune rule reads it as a
        fraction of the step's frontier weight instead), so a capped unit reaches a greater H-depth where the
        amplitude is. A deferred determinant is not banned: it re-enters when a stronger path reaches it. ``None``
        (default): :data:`DEFAULT_GF_PRUNE_TOL` (1e-3); ``0``: off (the unit then
        grows until it closes or hits the determinant cap).
    gf_pt2_tol : float or None
        Second-order importance rule for the pruned growth, **replacing** the amplitude rule: a frontier determinant is
        scored by its coherent second-order weight, maximised over the unit's real-frequency window, and the growth
        admits the highest scores until the score mass left out is at most this fraction (one collective tail cut).
        A budget on the relative error of the unit's ``G``. In ``(0, 1)``; ``None`` (default): the amplitude rule.
        Cannot be combined with ``gf_prune_tol`` (either value) or ``gf_first_shell_tol``: the first shell is kept
        whole. See ``doc/plans/pt2_growth_rule.md``.
    gf_first_shell_tol : float or None
        With ``gf_prune_tol``: the same cut for the seeds' first H-shell, which is kept whole by default (the
        moments of G through ``H^2``, hence the self-energy tail, stay exact). ``>= 0``; ``None``:
        0 by default. Needs ``gf_prune_tol``.
    gf_refine_rounds : int or None
        Swap refinement of a unit the determinant cap froze: up to this many rounds of scoring the retained and the
        discovered determinants on the discovered matrix and swapping the least important retained for the most
        important discovered ones at a fixed cap, then one recurrence on the final support. ``None`` (default):
        :data:`DEFAULT_GF_REFINE_ROUNDS` (3); ``0``: off.
        A unit the cap does not bind, one the memory guard froze, and a complex Hamiltonian are left alone. Needs
        the discovered matrix (more memory than the retained one) and costs minutes per capped unit.
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
    sigma_method: str = "dyson"
    gf_tol: Optional[float] = None
    gf_real_tol: Optional[float] = None
    gf_min_weight: Optional[float] = None
    sigma_causality_tol: Optional[float] = None
    gf_prune_tol: Optional[float] = None
    gf_first_shell_tol: Optional[float] = None
    gf_pt2_tol: Optional[float] = None
    gf_refine_rounds: Optional[int] = None
    gf_refine_tol: Optional[float] = None
    gf_refine_swap: Optional[float] = None

    def __post_init__(self):
        # One place for every front-end (RSPt solver line, TOML, CLI, archive replay): a bad
        # combination is refused here rather than silently ignored two layers down.
        for name in ("gf_tol", "gf_real_tol"):
            value = getattr(self, name)
            if value is None:
                continue
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1) (got {value!r}); omit it for the default")
        if self.gf_min_weight is not None and not 0.0 < self.gf_min_weight < 1.0:
            raise ValueError(
                f"gf_min_weight must lie in (0, 1) (got {self.gf_min_weight!r}); omit it to keep every state"
            )
        if self.sigma_causality_tol is not None and not self.sigma_causality_tol >= 0.0:
            raise ValueError(f"sigma_causality_tol must be >= 0 (got {self.sigma_causality_tol!r})")
        for name in ("gf_prune_tol", "gf_first_shell_tol"):
            value = getattr(self, name)
            if value is None or (value == 0.0 and name == "gf_prune_tol"):
                continue  # 0 is the explicit off
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be non-negative and finite (got {value!r}); omit it for the default")
        if self.gf_refine_rounds is not None and (
            isinstance(self.gf_refine_rounds, bool)
            or not isinstance(self.gf_refine_rounds, int)
            or self.gf_refine_rounds < 0
        ):
            raise ValueError(f"gf_refine_rounds must be a non-negative integer (got {self.gf_refine_rounds!r})")
        if (self.gf_refine_tol is not None or self.gf_refine_swap is not None) and self.gf_refine_rounds == 0:
            raise ValueError("gf_refine_tol / gf_refine_swap need gf_refine_rounds > 0 (refinement is switched off)")
        if self.gf_refine_tol is not None and not (
            math.isfinite(self.gf_refine_tol) and 0.0 < self.gf_refine_tol < 1.0
        ):
            raise ValueError(f"gf_refine_tol must be in (0, 1) (got {self.gf_refine_tol!r})")
        if self.gf_refine_swap is not None and not (0.0 < self.gf_refine_swap <= 0.5):
            raise ValueError(f"gf_refine_swap must be in (0, 0.5] (got {self.gf_refine_swap!r})")
        if self.gf_pt2_tol is not None:
            if not (math.isfinite(self.gf_pt2_tol) and 0.0 < self.gf_pt2_tol < 1.0):
                raise ValueError(
                    f"gf_pt2_tol must lie in (0, 1) (got {self.gf_pt2_tol!r}); omit it for the amplitude rule"
                )
            if self.gf_prune_tol is not None:
                raise ValueError(
                    "gf_pt2_tol replaces the amplitude rule gf_prune_tol; set one of them "
                    f"(got gf_pt2_tol={self.gf_pt2_tol!r}, gf_prune_tol={self.gf_prune_tol!r})"
                )
            if self.gf_first_shell_tol is not None:
                raise ValueError("gf_first_shell_tol does not apply to gf_pt2_tol: the first shell is kept whole")
        if self.gf_first_shell_tol is not None and self.gf_prune_tol == 0.0:
            raise ValueError("gf_first_shell_tol relaxes the first shell of pruned growth; pruning is switched off")

    @property
    def resolved_gf_prune_tol(self):
        """The pruned-growth tolerance a run uses; ``None`` leaves pruning off.

        An unset ``gf_prune_tol`` is :data:`DEFAULT_GF_PRUNE_TOL`, except under ``gf_pt2_tol``, whose rule replaces
        the amplitude rule (``0.0``: the amplitude rule is off).
        """
        if self.gf_pt2_tol is not None:
            return 0.0
        if self.gf_prune_tol is not None:
            return self.gf_prune_tol
        return DEFAULT_GF_PRUNE_TOL

    @property
    def resolved_gf_refine_rounds(self):
        """The refinement rounds a run uses: :data:`DEFAULT_GF_REFINE_ROUNDS` when unset."""
        if self.gf_refine_rounds is not None:
            return self.gf_refine_rounds
        return DEFAULT_GF_REFINE_ROUNDS


# ----------------------------------------------------------------------------------------------
# One option class per spectrum.  Each owns its own mesh and broadening -- there is no shared bag in
# which one number plays two roles -- and validates itself on construction, so a bad request raises
# on every rank before any collective is entered.
# ----------------------------------------------------------------------------------------------


def _default_w():
    return np.linspace(-25, 25, 3001)


def _check_mesh(name, mesh):
    if mesh is None or len(mesh) == 0:
        raise ValueError(f"{name} is empty: a spectrum needs at least one mesh point")


def _check_broadening(name, value):
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite broadening (got {value!r})")


def _check_dipole(dipole):
    if dipole is None:
        return
    shape = np.shape(dipole)
    if len(shape) != 3 or shape[0] != 3 or shape[1] != shape[2]:
        raise ValueError("dipole must have shape (3, n_imp, n_imp): the x, y, z transition matrices")


@dataclass(frozen=True)
class PESOptions:
    """Valence photoemission + inverse photoemission (``PS/spectra``).

    Attributes
    ----------
    w : numpy.ndarray
        Real-energy mesh (eV). Default ``linspace(-25, 25, 3001)``.
    delta : float
        Broadening above the real axis (HWHM, eV).
    """

    w: np.ndarray = field(default_factory=_default_w)
    delta: float = 0.2

    def __post_init__(self):
        _check_mesh("PES mesh w", self.w)
        _check_broadening("PES delta", self.delta)


@dataclass(frozen=True)
class XPSOptions:
    """Core-level x-ray photoemission (``XPS/spectra``). Needs a core shell.

    Attributes
    ----------
    w : numpy.ndarray
        Real-energy mesh (eV). Default ``linspace(-25, 25, 3001)``.
    delta : float
        Core-hole lifetime broadening (HWHM, eV).
    """

    w: np.ndarray = field(default_factory=_default_w)
    delta: float = 0.2

    def __post_init__(self):
        _check_mesh("XPS mesh w", self.w)
        _check_broadening("XPS delta", self.delta)


@dataclass(frozen=True)
class XASOptions:
    """X-ray absorption (``XAS/tensor``, or ``XAS/projected`` when ``projectors`` is given). Needs a core shell.

    Attributes
    ----------
    w : numpy.ndarray
        Real-energy mesh (eV). Default ``linspace(-25, 25, 3001)``.
    delta : float
        Core-hole lifetime broadening (HWHM, eV).
    projectors : dict or None
        Transition projectors. ``None`` stores the polarization-free Cartesian tensor.
    epsilons : sequence or None
        Polarization vectors; only read together with ``projectors``. ``None`` -> Cartesian x, y, z.
        The frame: crystal (global Cartesian) coordinates when the model carries operators with
        ``frame="global"`` (RSPt's), whose ``L`` fixes each shell's site rotation; otherwise the shells' local
        frame, the one ``rot_to_spherical`` defines. The tensor's x, y, z axes follow the same rule.
    dipole : numpy.ndarray or None
        ``(3, n_imp, n_imp)`` x, y, z dipole matrices in the model's input basis, mapping core to valence
        orbitals (RSPt's ``Dxyz``). ``None`` uses the model's own dipole if it fills the core->valence block,
        else the atomic (Gaunt) dipole operator. An explicit matrix is taken as given: its x, y, z are whatever
        frame it was built in (crystal axes for RSPt's ``Dxyz``).
    """

    w: np.ndarray = field(default_factory=_default_w)
    delta: float = 0.2
    projectors: Any = None
    epsilons: Any = None
    dipole: Any = None

    def __post_init__(self):
        _check_mesh("XAS mesh w", self.w)
        _check_broadening("XAS delta", self.delta)
        _check_dipole(self.dipole)


@dataclass(frozen=True)
class RIXSOptions:
    """Resonant inelastic x-ray scattering (``RIXS/tensor``, or ``RIXS/projected``). Needs a core shell.

    Attributes
    ----------
    w_in : numpy.ndarray
        Incoming-energy mesh (eV). Default ``linspace(-10, 20, 50)``.
    w_loss : numpy.ndarray
        Energy-loss mesh (eV). Default ``linspace(-2, 12, 4000)``.
    delta_intermediate : float
        Core-hole lifetime broadening of the intermediate state (HWHM, eV).
    delta_final : float
        Final-state lifetime broadening (HWHM, eV).
    adaptive_tol : float or None
        Stop tolerance of the greedy adaptive sampling of ``w_in``. ``None`` solves the whole grid.
    projectors : dict or None
        Transition projectors. ``None`` stores the polarization-free Kramers-Heisenberg tensor.
    epsilons_in, epsilons_out : sequence or None
        Polarization vectors; only read together with ``projectors``. ``None`` -> Cartesian x, y, z.
    dipole : numpy.ndarray or None
        As :attr:`XASOptions.dipole`.
    """

    w_in: np.ndarray = field(default_factory=lambda: np.linspace(-10, 20, 50))
    w_loss: np.ndarray = field(default_factory=lambda: np.linspace(-2.0, 12.0, 4000))
    delta_intermediate: float = 0.2
    delta_final: float = 0.050
    adaptive_tol: Optional[float] = None
    projectors: Any = None
    epsilons_in: Any = None
    epsilons_out: Any = None
    dipole: Any = None

    def __post_init__(self):
        _check_mesh("RIXS incoming-energy mesh w_in", self.w_in)
        _check_mesh("RIXS energy-loss mesh w_loss", self.w_loss)
        _check_broadening("RIXS delta_intermediate", self.delta_intermediate)
        _check_broadening("RIXS delta_final", self.delta_final)
        _check_dipole(self.dipole)


@dataclass(frozen=True)
class NIXSOptions:
    """Non-resonant inelastic x-ray scattering (``NIXS/spectra``).

    Attributes
    ----------
    w_loss : numpy.ndarray
        Energy-loss mesh (eV). Default ``linspace(-2, 12, 4000)``.
    delta : float
        Broadening (HWHM, eV).
    qs : sequence or None
        Momentum-transfer vectors. ``None`` -> two default ``|q|`` along (1, 1, 1).
    l_final, l_initial : int
        Angular momenta of the final / initial orbitals of the excitation.
    radial : tuple
        ``(radial_mesh, R_final, R_initial)``: the radial integral's input. Required.
    """

    radial: Any
    w_loss: np.ndarray = field(default_factory=lambda: np.linspace(-2.0, 12.0, 4000))
    delta: float = 0.100
    qs: Any = None
    l_final: int = 2
    l_initial: int = 2

    def __post_init__(self):
        if self.radial is None:
            raise ValueError("NIXS needs radial data (radial_mesh, R_final, R_initial)")
        _check_mesh("NIXS mesh w_loss", self.w_loss)
        _check_broadening("NIXS delta", self.delta)

    @property
    def resolved_qs(self):
        """The momentum transfers: the given ones, else the two default ``|q|``."""
        if self.qs is not None:
            return self.qs
        return [2 * np.array([1, 1, 1]) / np.sqrt(3), 7 * np.array([1, 1, 1]) / np.sqrt(3)]


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
    cli : str or None
        The ``impurityModel selfenergy`` command-line flag (``None``: not a flag).  A ``"bool"``
        field is a store-false switch (``--no-chain-restrict``); the others take a value.
    help : str
        The flag's help text.
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
    cli: Optional[str] = None
    help: str = ""

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
    _spec(
        "gf_pt2_tol",
        "solver",
        "float",
        "GF second-order growth tolerance",
        token="gf_pt2_tol",
        unset="off (amplitude rule)",
    ),
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

#: ``impurityModel selfenergy`` flags: ``{field: (flag, help)}``.  Kept beside the rows so that a new
#: option is one place; ``scripts/selfenergy.py`` builds its arguments and overrides from them.
_CLI = {
    "dN": ("--dN", "Impurity occupation window (+-dN) for the excited bases."),
    "cost_budget": (
        "--cost_budget",
        "Hopping-cost budget alpha: bath excitations cost round(8*(d-d_min)) on the coupling distance, "
        "bounded by round(COST_KAPPA*alpha*-log(slater_weight_min)), COST_KAPPA = 8. "
        "default: off (see doc/plans/gf_relevant_support.md).",
    ),
    "chain_restrict": ("--no-chain-restrict", "Disable chain occupation restrictions."),
    "dense_cutoff": ("--dense-cutoff", "Use a dense eigensolver below this size."),
    "sparse_green": ("--no-sparse-green", "Disable the sparse block-Lanczos path."),
    "gf_prune_tol": (
        "--gf-prune-tol",
        "Importance-pruned growth of the block-Lanczos Green's function: a step admits a new determinant only if "
        "its largest amplitude on the normalised step block exceeds this. Deferred determinants are not banned. "
        "(default: 1e-3; 0 is off; an archive that did not set it replays with the default).",
    ),
    "gf_first_shell_tol": (
        "--gf-first-shell-tol",
        "With pruning on: prune the seeds' first H-shell by this cut too (0 keeps it whole, the default; "
        "the self-energy tail stays exact only then).",
    ),
    "gf_pt2_tol": (
        "--gf-pt2-tol",
        "Second-order growth rule replacing the amplitude rule: admit the determinants with the highest coherent "
        "second-order weight until the weight left out is at most this fraction (a budget on the relative error "
        "of G). In (0, 1); not with --gf-prune-tol or --gf-first-shell-tol (default: off).",
    ),
    "gf_refine_rounds": (
        "--gf-refine-rounds",
        "Swap refinement of a unit the determinant cap froze: up to this many rounds of swapping the least "
        "important retained determinants for the most important discovered ones at the same cap (default: 3; 0 "
        "is off; an archive that did not set it replays with the default). Needs the discovered matrix; costs minutes per "
        "capped unit.",
    ),
    "gf_refine_tol": (
        "--gf-refine-tol",
        "With --gf-refine-rounds: stop when the estimated remaining gain is below this fraction of |G| (1e-3).",
    ),
    "gf_refine_swap": (
        "--gf-refine-swap",
        "With --gf-refine-rounds: the largest fraction of the cap one round swaps, in (0, 0.5] (0.1).",
    ),
    "gf_tol": (
        "--gf-tol",
        "Block-Lanczos Green's-function convergence tolerance (relative change of G) on every axis "
        "(default: max(slater_weight_min^2, 1e-9), or the archive's).",
    ),
    "gf_real_tol": ("--gf-real-tol", "The same tolerance on the real-frequency axis only (default: --gf-tol's value)."),
    "gf_min_weight": (
        "--gf-min-weight",
        "Drop thermal states (whole degenerate manifolds, never the ground one) whose normalised "
        "Boltzmann weight is below this from the Green's function and self-energy (default: keep all).",
    ),
    "sigma_causality_tol": (
        "--sigma-causality-tol",
        "Relative tolerance of the self-energy causality check; smaller violations only warn (default: 1e-3).",
    ),
}
OPTION_SPECS = tuple(
    replace(spec, cli=_CLI[spec.name][0], help=_CLI[spec.name][1]) if spec.name in _CLI else spec
    for spec in OPTION_SPECS
)
assert set(_CLI) <= {spec.name for spec in OPTION_SPECS}

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
            if not math.isfinite(number) or number != int(number):
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
    attrs = {spec.attr_name: getattr(groups[spec.group], spec.name) for spec in OPTION_SPECS}
    if isinstance(attrs.get("mv"), dict):
        raise ValueError("a per-group mixed_valence dict cannot be recorded in an archive; use None, a bool or a word")
    return attrs


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
