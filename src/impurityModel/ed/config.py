"""Central registry of the environment-variable tuning knobs.

Every runtime-tunable parameter of the solver stack is declared exactly once here: its
environment-variable name, type, default, clamp and the rationale for the default. Call
sites read a knob through its :meth:`Knob.get` accessor instead of reaching into
``os.environ`` themselves, so a default lives in one place and cannot drift between the
module that consumes it and the module that models its cost (``memory_estimate`` used to
re-hardcode two of them).

Knobs are read **lazily**, on every ``get()``: a caller (or a test) may set the variable at
any point and the next read sees it.

A knob whose ``default`` is ``None`` has no static default -- the value is derived at the
call site from information only available there (available memory, communicator size). The
knob then only carries the *override*: ``get()`` returns ``None`` when the variable is unset
and the call site derives as usual.

``dump()`` renders the whole registry as a table; ``doc/configuration.md`` is generated from
it, so a new knob is documented by declaring it here.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["KNOBS", "Knob", "dump"]


def _parse_bool(raw: str) -> bool:
    """Truthiness of an environment-variable string. Only explicit falsehoods are false."""
    return raw not in ("0", "", "false", "False")


_PARSERS: dict[str, Callable[[str], Any]] = {
    "int": int,
    "float": float,
    "bool": _parse_bool,
    "str": str,
}


@dataclass(frozen=True)
class Knob:
    """One environment-variable-backed tuning parameter.

    Parameters
    ----------
    name : str
        The environment variable, e.g. ``"GF_SLICES"``.
    kind : {"int", "float", "bool", "str"}
        How the raw string is parsed.
    default : Any
        Value when the variable is unset. ``None`` means the call site derives the value
        (the knob is an override only).
    doc : str
        What the knob tunes, and why the default is what it is.
    minimum : Any, optional
        Lower clamp applied after parsing (numeric knobs only).
    group : str
        Section heading used by :func:`dump`.
    """

    name: str
    kind: str
    default: Any
    doc: str
    minimum: Any = None
    group: str = "general"

    def get(self) -> Any:
        """The knob's current value: the parsed environment variable, else the default.

        Read lazily -- every call re-reads ``os.environ``. An empty string counts as unset
        for the non-string knobs, so ``GF_RIXS_ADAPTIVE_TOL=`` disables the sampler rather
        than raising.
        """
        raw = os.environ.get(self.name)
        if raw is None or (raw == "" and self.kind != "str"):
            return self.default
        value = _PARSERS[self.kind](raw)
        if self.minimum is not None and value < self.minimum:
            value = self.minimum
        return value


def _register(*knobs: Knob) -> dict[str, Knob]:
    return {knob.name: knob for knob in knobs}


# --- Green's function: per-frequency BiCGSTAB solver (gf_method="bicgstab") -----------------

GF_BICGSTAB_ATOL = Knob(
    name="GF_BICGSTAB_ATOL",
    kind="float",
    default=1e-8,
    group="bicgstab",
    doc="""Absolute residual tolerance of a per-frequency BiCGSTAB solve. The default sits at
    the block-Lanczos reference's accuracy (doc/plans/bicgstab_per_frequency_gf.md, Phase 3a)
    -- inside the 2.5e-8 spread PARTIAL-vs-FULL reorthogonalization itself shows on the real
    workloads. The reliability diagnostics (gf_diagnostics.check_bicgstab_convergence) derive
    their thresholds from the value actually used; never re-hardcode it.""",
)

GF_BICGSTAB_MAX_ITER = Knob(
    name="GF_BICGSTAB_MAX_ITER",
    kind="int",
    default=500,
    minimum=1,
    group="bicgstab",
    doc="""Hard per-point iteration bound. Warm-started production solves measure ~3 iterations
    and a cold start ~6, so 500 is pathology headroom: a stagnating solve (a real-axis point
    within `delta` of a pole) ends and is *reported* by the diagnostics instead of iterating
    until the growing seen-support exhaustion bound -- which a solve that keeps discovering
    determinants may never reach.""",
)

GF_BICGSTAB_RESTARTS = Knob(
    name="GF_BICGSTAB_RESTARTS",
    kind="int",
    default=10,
    minimum=0,
    group="bicgstab",
    doc="""Restarts granted to a non-converged point before the GMRES fallback. Restarting is
    the standard cure for BiCGSTAB's r0-orthogonality stagnation, which real-axis points within
    ~delta of a pole do hit. Progress-gated: a restart must shrink the residual by at least
    half to earn the next one, so a genuinely stuck point stops early and is reported.""",
)

GF_GMRES_RESTART = Knob(
    name="GF_GMRES_RESTART",
    kind="int",
    default=40,
    minimum=1,
    group="bicgstab",
    doc="""Krylov restart length of the GMRES fallback (the solver for points BiCGSTAB leaves
    unconverged: its shadow-residual recurrence stagnates near a pole, GMRES minimizes the
    residual and has no such mode). Bounds the fallback's live Krylov blocks, so the memory
    model (memory_estimate.estimate_gf_peak_bytes, method="bicgstab") reads the same knob.""",
)

GF_GMRES_MAX_RESTARTS = Knob(
    name="GF_GMRES_MAX_RESTARTS",
    kind="int",
    default=25,
    minimum=1,
    group="bicgstab",
    doc="Maximum GMRES restart cycles before the point is reported as unconverged.",
)

# --- Green's function: per-frequency CIPSI-selected solver (gf_method="cipsi") --------------
# Experimental (doc/plans/gf_cipsi_frequency_truncation.md): the per-point basis is grown by
# resolvent-targeted CIPSI selection (CIPSISolver.select_at) instead of the H-connectivity
# closure, so the retained determinants are the *important* ones at each frequency rather
# than the first-discovered ones the freeze-growth cap keeps.

GF_CIPSI_BUDGET = Knob(
    name="GF_CIPSI_BUDGET",
    kind="int",
    default=None,  # unset = inherit the basis truncation_threshold
    minimum=1,
    group="cipsi",
    doc="""Per-point determinant budget of a CIPSI-selected solve: selection rounds stop
    admitting candidates once the basis reaches this size. Unset inherits the basis
    ``truncation_threshold`` (possibly unbounded).""",
)

GF_CIPSI_MAX_NEW = Knob(
    name="GF_CIPSI_MAX_NEW",
    kind="int",
    default=None,  # unset = only the remaining budget caps a round
    minimum=1,
    group="cipsi",
    doc="""Global cap on the candidates admitted per selection round (collective bisection on
    the importance scores). Unset admits every candidate passing ``GF_CIPSI_DE2_MIN`` up to
    the remaining budget; a finite value staggers the growth so later rounds select with a
    better-converged iterate.""",
)

GF_CIPSI_DE2_MIN = Knob(
    name="GF_CIPSI_DE2_MIN",
    kind="float",
    default=0.0,
    minimum=0.0,
    group="cipsi",
    doc="""Importance floor of the resolvent CIPSI selection: candidates below it are never
    admitted regardless of budget. 0 (default) leaves the truncation entirely to the budget
    and the boundary-residual stop.""",
)

GF_CIPSI_MAX_ROUNDS = Knob(
    name="GF_CIPSI_MAX_ROUNDS",
    kind="int",
    default=8,
    minimum=1,
    group="cipsi",
    doc="""Solve->select->re-solve rounds per frequency point. Each round solves exactly on
    the frozen basis, then admits the highest-importance boundary determinants; the loop
    also stops on the boundary-residual tolerance or an exhausted budget.""",
)

GF_CIPSI_BOUNDARY_TOL = Knob(
    name="GF_CIPSI_BOUNDARY_TOL",
    kind="float",
    default=None,  # unset = the solver atol (GF_BICGSTAB_ATOL)
    minimum=0.0,
    group="cipsi",
    doc="""Stop tolerance on the boundary residual (the true-residual norm outside the basis,
    relative to the seed norm) -- the selection loop's convergence measure, and the honest
    truncation-error estimate of the returned G. Unset uses the in-basis solver tolerance
    (``GF_BICGSTAB_ATOL``), so in-basis and out-of-basis errors are balanced by default.""",
)

GF_CIPSI_SCORER = Knob(
    name="GF_CIPSI_SCORER",
    kind="str",
    default="de2",
    group="cipsi",
    doc="""Candidate importance: ``de2`` is the resolvent weight
    ``sum_i |<Dj|H|X_i>|^2 / |z - E_Dj|^2`` (frequency-targeted); ``amplitude`` drops the
    energy denominator (the bare-coupling baseline the frequency targeting must beat).""",
)

GF_CIPSI_PT2 = Knob(
    name="GF_CIPSI_PT2",
    kind="bool",
    default=False,
    group="cipsi",
    doc="""Add the second-order (Loewdin downfolding) correction of the discarded boundary to
    G: ``dG_ij = sum_D <D|H|X_i> <D|H|X_j> / (z - E_D)`` over the final round's unadmitted
    candidates (complex-symmetric approximation, exact for a real Hamiltonian matrix). Its
    magnitude is recorded in the stats either way -- it doubles as a truncation-error bar.""",
)

# --- Green's function: spectrum slicing (gf_method="sliced") --------------------------------
# Retained as a documented failure: doc/plans/spectrum_slicing.md records why the projected
# 2-8x win never materialized (the live basis is the H-connectivity closure of the seed
# support, invariant under filtering).

GF_SLICES = Knob(
    name="GF_SLICES",
    kind="int",
    default=8,
    minimum=1,
    group="sliced",
    doc="Number of Chebyshev windows tiling the real-axis evaluation band.",
)

GF_SLICE_DEGREE = Knob(
    name="GF_SLICE_DEGREE",
    kind="int",
    default=0,
    minimum=0,
    group="sliced",
    doc="Chebyshev filter degree; 0 = auto (derived from the bandwidth / slice-width ratio).",
)

GF_SLICE_TOL = Knob(
    name="GF_SLICE_TOL",
    kind="float",
    default=0.0,
    minimum=0.0,
    group="sliced",
    doc="Amplitude truncation applied to the filtered slice seeds; 0 = no truncation.",
)

# --- Green's function: work-unit decomposition ---------------------------------------------

GF_EIGENSTATE_GROUP = Knob(
    name="GF_EIGENSTATE_GROUP",
    kind="int",
    default=1,
    minimum=1,
    group="units",
    doc="""Eigenstates stacked into one block-Lanczos work unit. Stacking shares the
    matvec/Krylov build across eigenstates but grows the per-step reorthogonalization with the
    block width, so the optimum is workload-dependent (doc/plans/calc_selfenergy_performance.md).
    The default (1) gives each eigenstate its own unit and its own Krylov space.""",
)

GF_OPERATOR_SPLIT = Knob(
    name="GF_OPERATOR_SPLIT",
    kind="bool",
    default=False,
    group="units",
    doc="""Split each orbital block's Green's function into scalar (pairwise) continued
    fractions, one per operator column, instead of one block recurrence. Multiplies the number
    of independent work units -- better load balance for few large blocks -- at the cost of
    redundant Krylov building (no subspace shared across columns). Mutually exclusive with
    eigenstate grouping; the operator split wins when both are requested.""",
)

GF_PER_STATE_RESTRICT = Knob(
    name="GF_PER_STATE_RESTRICT",
    kind="bool",
    default=None,  # falls back to the basis's chain_restrict flag
    group="units",
    doc="""Build the excited-sector occupation window per eigenstate rather than once for the
    thermal ensemble. Unset, it follows the basis's ``chain_restrict`` flag. It only matters
    when the bath classification is state-dependent (long chains, where distant sites clear the
    coupling-distance filter); for a directly-hybridizing single bath shell the per-state and
    ensemble windows are identical and this is a no-op.""",
)

# --- Green's function: block-Lanczos convergence monitor ------------------------------------

GF_CHECK_EVERY = Knob(
    name="GF_CHECK_EVERY",
    kind="int",
    default=8,
    minimum=1,
    group="convergence",
    doc="""Blocks between convergence tests during the long approach. The test rebuilds the
    block continued fraction each call -- the single largest cost of the block-Lanczos Green's
    function (~53% of runtime at reort=NONE, measured) -- so while convergence is still far
    away it is sampled sparsely. Set to 1 to test every block. Once a check lands within
    GF_NEAR_FACTOR x tol the monitor switches to every block regardless, so the exact
    convergence point is caught with no added Lanczos steps and the converged G is unchanged.""",
)

GF_NEAR_FACTOR = Knob(
    name="GF_NEAR_FACTOR",
    kind="float",
    default=2.0,
    minimum=1.0,
    group="convergence",
    doc="""Switch from sparse to per-block convergence sampling once the relative change is
    within this factor of the tolerance. Kept small: the relative change typically sits on a
    long noisy plateau a decade or two above tolerance before its final descent, and that
    plateau must stay in the sparse regime for the sampling to pay off.""",
)

# --- RIXS: shift-recycling solver tiers -----------------------------------------------------

GF_SECTOR_DENSE_MAX = Knob(
    name="GF_SECTOR_DENSE_MAX",
    kind="int",
    default=None,  # derived: sqrt(0.25 * available_bytes_per_rank / (3 * 16))
    minimum=0,
    group="rixs-solvers",
    doc="""Largest sector the RIXS R1 spectral cache (SectorResolventCache) may densify and
    eigendecompose. The eigendecomposition holds ~3 dense (N, N) complex arrays (H, the
    eigenvector matrix, LAPACK workspace); unset, the cap is derived so that fits in a quarter
    of the available per-rank memory. 0 disables the tier.""",
)

GF_SECTOR_CACHE_DIR = Knob(
    name="GF_SECTOR_CACHE_DIR",
    kind="str",
    default="",
    group="rixs-solvers",
    doc="""Directory persisting SectorResolventCache eigendecompositions across runs. Empty =
    in-memory only. With it, the dominant one-time `eigh` cost (measured ~450 s at 5565
    determinants; OpenBLAS's Hermitian eigensolvers are bound by their non-parallelizing
    reduction stage, and the measured alternatives are no faster with eigenvectors) is paid
    once per material instead of once per run.""",
)

GF_KRYLOV_RECYCLE_MAX_BYTES = Knob(
    name="GF_KRYLOV_RECYCLE_MAX_BYTES",
    kind="int",
    default=None,  # derived: available_bytes_per_rank // 4
    minimum=0,
    group="rixs-solvers",
    doc="""Per-rank byte cap on a recycled Krylov store (KrylovShiftedResolvent: one
    block-Lanczos recurrence serving every shift of a fixed right-hand side). The retained
    Krylov basis is that tier's dominant allocation; unset, it is capped at a quarter of the
    available per-rank memory, mirroring GF_SECTOR_DENSE_MAX's budget. 0 disables the tier.""",
)

# --- RIXS: incoming-energy sampling ---------------------------------------------------------

GF_RIXS_WIN_CHUNK = Knob(
    name="GF_RIXS_WIN_CHUNK",
    kind="int",
    default=None,  # derived from the eigenstate count, mesh size and communicator size
    minimum=1,
    group="rixs-sampling",
    doc="""Incoming-energy points per RIXS work unit. A unit is (eigenstate x contiguous
    wIn-chunk); contiguity preserves the warm-start locality of consecutive points, and a unit
    is atomic (the engine never reorders within one). Unset, the default targets ~3 units per
    rank so the packing has slack to balance without fragmenting the warm-start chains; a
    serial run gets one unit per eigenstate (maximal locality).""",
)

GF_RIXS_ADAPTIVE_TOL = Knob(
    name="GF_RIXS_ADAPTIVE_TOL",
    kind="float",
    default=None,  # unset = the adaptive sampler is off, the wIn grid is swept densely
    minimum=0.0,
    group="rixs-sampling",
    doc="""Stop tolerance of the greedy adaptive wIn sampler (set-valued AAA): solve only the
    incoming energies the rational interpolant cannot yet predict to within this tolerance.
    Unset/empty disables it (dense sweep). Measured on NiO L3: 28 of 121 solves at 1e-4
    relative error.""",
)

GF_RIXS_ADAPTIVE_BATCH = Knob(
    name="GF_RIXS_ADAPTIVE_BATCH",
    kind="int",
    default=1,
    minimum=1,
    group="rixs-sampling",
    doc="""New wIn solves per adaptive round. Above 1 trades interpolation sharpness (each
    round's greedy pick is made with less information) for parallel width.""",
)


KNOBS: dict[str, Knob] = _register(
    GF_BICGSTAB_ATOL,
    GF_BICGSTAB_MAX_ITER,
    GF_BICGSTAB_RESTARTS,
    GF_GMRES_RESTART,
    GF_GMRES_MAX_RESTARTS,
    GF_CIPSI_BUDGET,
    GF_CIPSI_MAX_NEW,
    GF_CIPSI_DE2_MIN,
    GF_CIPSI_MAX_ROUNDS,
    GF_CIPSI_BOUNDARY_TOL,
    GF_CIPSI_SCORER,
    GF_CIPSI_PT2,
    GF_SLICES,
    GF_SLICE_DEGREE,
    GF_SLICE_TOL,
    GF_EIGENSTATE_GROUP,
    GF_OPERATOR_SPLIT,
    GF_PER_STATE_RESTRICT,
    GF_CHECK_EVERY,
    GF_NEAR_FACTOR,
    GF_SECTOR_DENSE_MAX,
    GF_SECTOR_CACHE_DIR,
    GF_KRYLOV_RECYCLE_MAX_BYTES,
    GF_RIXS_WIN_CHUNK,
    GF_RIXS_ADAPTIVE_TOL,
    GF_RIXS_ADAPTIVE_BATCH,
)

GROUP_TITLES = {
    "bicgstab": 'Per-frequency BiCGSTAB solver (``gf_method="bicgstab"``)',
    "cipsi": 'Per-frequency CIPSI-selected solver (``gf_method="cipsi"``)',
    "sliced": 'Spectrum slicing (``gf_method="sliced"``)',
    "units": "Green's-function work-unit decomposition",
    "convergence": "Block-Lanczos convergence monitor",
    "rixs-solvers": "RIXS shift-recycling solver tiers",
    "rixs-sampling": "RIXS incoming-energy sampling",
}


def _summary(knob: Knob) -> str:
    """The knob's docstring collapsed to a single whitespace-normalized line."""
    return " ".join(knob.doc.split())


def dump() -> str:
    """Render the registry as a Markdown table, grouped by section.

    The source of ``doc/configuration.md``; a knob declared here is documented by construction.
    """
    lines = []
    for group, title in GROUP_TITLES.items():
        knobs = [k for k in KNOBS.values() if k.group == group]
        if not knobs:
            continue
        lines.append(f"## {title}\n")
        lines.append("| Variable | Type | Default | Description |")
        lines.append("| --- | --- | --- | --- |")
        for knob in knobs:
            default = "*derived*" if knob.default is None else f"`{knob.default!r}`"
            lines.append(f"| `{knob.name}` | {knob.kind} | {default} | {_summary(knob)} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - developer convenience
    print(dump())
