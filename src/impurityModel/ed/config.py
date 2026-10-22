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
        The environment variable, e.g. ``"GF_EIGENSTATE_GROUP"``.
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


# --- Green's-function kernels (the ``gf_method`` solver option) ----------------------------
# Not knobs -- ``gf_method`` is a solver option -- but every front end (the driver, the TOML
# schema, the CLI, archive replay) must accept the same set, and this leaf module is the one
# they all import.

#: Accepted ``gf_method`` values.
GF_METHODS = ("lanczos", "bicgstab")

#: Accepted ``SolverOptions.gf_admission`` values, the basis-growth policy of the per-frequency
#: ``"bicgstab"`` kernel: ``"all"`` admits every determinant the solver produces, ``"outer"`` solves on a
#: frozen basis, measures the residual outside it and admits only what scores above a threshold
#: (:mod:`impurityModel.ed.gf_admission`). ``"outer"`` needs ``gf_method="bicgstab"``.
GF_ADMISSIONS = ("all", "outer")

#: Accepted ``GF_SCHEDULER`` values (see that knob).
GF_SCHEDULERS = ("static", "queue")

#: Accepted ``SolverOptions.sigma_method`` values: the self-energy estimators of
#: :mod:`impurityModel.ed.sigma_estimators` (its ``ESTIMATORS`` registry has exactly these keys).
SIGMA_METHODS = ("dyson",)

#: Retired ``gf_method`` values, each with the reason and the replacement. Archive replay maps
#: them to ``"lanczos"`` with a warning; every other front end rejects them with this message.
RETIRED_GF_METHODS = {
    "sliced": (
        "was retired: spectrum slicing never delivered its projected memory win (the live basis "
        "is the H-connectivity closure of the seed support, doc/plans/spectrum_slicing.md) and "
        "missed the exact G by ~1e-5 (doc/reviews/gf_review.md, S1); use 'lanczos' or 'bicgstab'"
    ),
    "cipsi": (
        "was retired: resolvent-targeted CIPSI selection matched, but did not beat, the plain "
        "freeze-growth cap at equal determinant budget, at 1.5-2x the wall cost "
        "(doc/plans/gf_cipsi_frequency_truncation.md; doc/reviews/gf_review.md, S2); use "
        "'bicgstab' with a truncation_threshold, or 'lanczos'"
    ),
}

#: Knobs that no longer exist, each with the reason. ``[environment]`` rejects them with this
#: message, and a GF run warns when one is still set in the process environment
#: (:func:`warn_retired_knobs`) -- an exported variable nobody reads any more would otherwise
#: be silently ignored.
RETIRED_KNOBS = {
    **dict.fromkeys(("GF_SLICES", "GF_SLICE_DEGREE", "GF_SLICE_TOL"), "retired with gf_method='sliced'"),
    **dict.fromkeys(
        (
            "GF_CIPSI_BUDGET",
            "GF_CIPSI_MAX_NEW",
            "GF_CIPSI_DE2_MIN",
            "GF_CIPSI_MAX_ROUNDS",
            "GF_CIPSI_BOUNDARY_TOL",
            "GF_CIPSI_SCORER",
            "GF_CIPSI_PT2",
        ),
        "retired with gf_method='cipsi'",
    ),
    **dict.fromkeys(
        ("GF_CLOSURE_FREEZE", "GF_STAGNATION_FREEZE", "GF_STAGNATION_WINDOW"),
        "retired with the zero-growth and absolute-threshold freezes; the support freeze replaces both, "
        "weighing the weight reaching new determinants against what slaterWeightMin already cuts",
    ),
    "MATVEC_PRUNE": (
        "retired: the slaterWeightMin cut of a redistributed sparse matvec is always made on the summed rows "
        "('after_sum'); 'before_sum' made the result depend on the rank and chunk count "
        "(doc/gf_support_pipeline.md)"
    ),
    "GF_FROZEN_CSR": "retired: the frozen-basis CSR fallback is always on",
    "GF_SUPPORT_FREEZE": "retired: the support freeze is always on (it is off only under importance pruning, GF_LANCZOS_ADMIT_TOL)",
    "GS_SELECTION_PRUNE": "retired: the CIPSI selection round always cuts the summed amplitude ('after_sum'); 'before_sum' made the selected set depend on the rank and chunk count",
    "GS_MEMORY_BUDGET_INCLUDE_RESIDENT": "retired: the ground-state memory budget always includes the resident set",
    "GF_CHECK_EVERY": "retired: a fixed tuning constant of the convergence monitor (every 8 blocks until within 2x of the tolerance)",
    "GF_NEAR_FACTOR": "retired: a fixed tuning constant of the convergence monitor (every 8 blocks until within 2x of the tolerance)",
    "GF_BICGSTAB_MAX_ITER": "retired: a fixed bound (500 iterations per point)",
    "GF_GMRES_MAX_RESTARTS": "retired: a fixed bound (25 GMRES restart cycles)",
    "GF_BICGSTAB_RESTARTS": "retired: a fixed bound (10 BiCGSTAB restarts before the GMRES fallback)",
    "GF_BICGSTAB_WARM_HISTORY": "retired: a fixed constant (3 warm-start solutions)",
    "GF_RIXS_ADAPTIVE_BATCH": "retired: the adaptive wIn sampler always solves one new point per round",
    "GS_APPLY_ROW_CHUNKS": "retired: merged with GF_APPLY_ROW_CHUNKS into APPLY_ROW_CHUNKS",
    "GF_APPLY_ROW_CHUNKS": "retired: merged with GS_APPLY_ROW_CHUNKS into APPLY_ROW_CHUNKS",
    "GF_WEIGHTED_TOL": "retired: weighting the convergence tolerance by thermal weight was removed (it never left experimental)",
    "GF_WEIGHTED_TOL_CEILING": "retired: weighting the convergence tolerance by thermal weight was removed (it never left experimental)",
    "DC_CAP_STRATEGY": "retired: the double-counting cap is always resolved by running at the ceiling; the ascending cap ladder was removed",
    "DC_CAP_LADDER_START": "retired: the double-counting cap ladder was removed",
    "DC_CAP_LADDER_MAX_RUNGS": "retired: the double-counting cap ladder was removed",
    "GS_MATVEC_EXCHANGE": (
        "retired: the distributed array matvec always uses the sparse graph exchange; the per-root 'reduce' "
        "spelling was the A/B arm it replaced (mpi_comm.MatvecExchangePlan)"
    ),
    "GF_OPERATOR_SPLIT": (
        "retired: the pairwise operator split (scalar continued fractions per operator pair) was an "
        "opt-in load-balancing mode that multiplied the Krylov work, and the improved-estimator "
        "self-energy needs whole-block recurrences (doc/reviews/gf_review.md, S3)"
    ),
}


def warn_retired_knobs():
    """Warn once per call for every retired knob still set in ``os.environ``."""
    import warnings

    for name, reason in RETIRED_KNOBS.items():
        if os.environ.get(name, "") != "":
            warnings.warn(f"environment variable {name} is set but no longer read: {reason}.", stacklevel=2)


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

GF_BICGSTAB_RESIDUAL_CHECK = Knob(
    name="GF_BICGSTAB_RESIDUAL_CHECK",
    kind="bool",
    default=None,
    group="bicgstab",
    doc="""After every per-frequency solve, apply the operator once more at cutoff 0 and record the
    true residual split at the solve basis: the in-basis part (the solver's residual, recomputed)
    and the boundary residual (1-P) H X that a truncated basis leaves unseen, with a bound on the
    error of G built from them (``gf_primitives.resolvent_error_bound``). Costs one unchunked
    matvec per point -- its peak is the full cutoff-0 image of the iterate, larger than a solve
    iteration's. Unset (derived), it follows the admission policy: on for ``outer`` admission,
    whose point is to trade basis size for a measured error, off for ``all``. ``1``/``0`` force it
    either way.""",
)

GF_BICGSTAB_ADMISSION = Knob(
    name="GF_BICGSTAB_ADMISSION",
    kind="str",
    default="all",
    group="bicgstab",
    doc="""How a per-frequency point's basis grows. ``all`` (default): every determinant the solver
    produces is admitted (``slaterWeightMin`` aside), up to the cap. ``outer``: solve on a frozen
    basis, measure the residual outside it, admit only the determinants that score above
    ``GF_BICGSTAB_ADMIT_TOL_*`` (at most the cap allows), and re-solve -- importance-ranked
    admission between frozen solves, never inside the BiCGSTAB recurrence (dropping rows between
    steps breaks its recursive residual).""",
)

GF_BICGSTAB_ADMIT_SCORER = Knob(
    name="GF_BICGSTAB_ADMIT_SCORER",
    kind="str",
    default="amplitude",
    group="bicgstab",
    doc="""Importance score of an outside determinant D under ``outer`` admission, per column j and
    maximised over the block: ``amplitude`` is ``|b_Dj| / ||Y_j||`` (boundary residual over seed
    norm); ``jacobi`` is ``|b_Dj| / (||X_j|| |z - H_DD|)`` (the first-order correction it would
    receive, so near-resonant determinants rank first). ``jacobi`` only means something where
    ``H_DD`` is a configuration energy, i.e. a star bath.""",
)

GF_BICGSTAB_ADMIT_TOL_AMP = Knob(
    name="GF_BICGSTAB_ADMIT_TOL_AMP",
    kind="float",
    default=1e-4,
    minimum=0.0,
    group="bicgstab",
    doc="Admission threshold of the ``amplitude`` scorer (dimensionless: relative to the seed norm).",
)

GF_BICGSTAB_ADMIT_TOL_JACOBI = Knob(
    name="GF_BICGSTAB_ADMIT_TOL_JACOBI",
    kind="float",
    default=1e-4,
    minimum=0.0,
    group="bicgstab",
    doc="Admission threshold of the ``jacobi`` scorer (dimensionless: relative to the iterate norm).",
)

GF_BICGSTAB_ADMIT_ROUNDS = Knob(
    name="GF_BICGSTAB_ADMIT_ROUNDS",
    kind="int",
    default=20,
    minimum=0,
    group="bicgstab",
    doc="""Solve-measure-admit rounds per point before the final solve. Each round reaches
    ``GF_BICGSTAB_ADMIT_SHELLS`` H-shells beyond the current basis, so this bounds the depth the
    admission can reach; the retired per-frequency CIPSI was round-limited at 8 shells.""",
)

GF_BICGSTAB_ADMIT_SHELLS = Knob(
    name="GF_BICGSTAB_ADMIT_SHELLS",
    kind="int",
    default=2,
    minimum=1,
    group="bicgstab",
    doc="""H-shells admitted per round. The first is scored on the measured boundary residual; each
    further one on the residual of the Jacobi-extended correction of the shell before it, so a
    round reaches the boundary of the boundary without a solve in between.""",
)

GF_BICGSTAB_ADMIT_CARRY_TOL = Knob(
    name="GF_BICGSTAB_ADMIT_CARRY_TOL",
    kind="float",
    default=1e-3,
    minimum=0.0,
    group="bicgstab",
    doc="""Under ``outer`` admission a point starts from its seeds, their first H-shell and the
    determinants of the warm-start guess with ``|x0_D| / ||x0|| >= this``, re-scored at the new z so
    the set can shrink -- not from the whole extrapolation support, which only ever grows along the
    sweep.""",
)

GF_BICGSTAB_ADMIT_EN_TOL = Knob(
    name="GF_BICGSTAB_ADMIT_EN_TOL",
    kind="float",
    default=0.0,
    minimum=0.0,
    group="bicgstab",
    doc="""Stop admitting when the Epstein-Nesbet estimate of the remaining G error,
    ``sum_D b_Di* b_Dj / (z - H_DD)``, is below this fraction of ``max_j |G_jj|``. 0 (default)
    disables it. An estimate, not a bound: it is reported and never added to G.""",
)

GF_LANCZOS_ADMIT_TOL = Knob(
    name="GF_LANCZOS_ADMIT_TOL",
    kind="float",
    default=0.0,
    minimum=0.0,
    group="bicgstab",
    doc="""Importance pruning of the block-Lanczos recurrence's growth -- the comparator for
    ``GF_BICGSTAB_ADMISSION=outer``. A step's new determinants are admitted only if their largest
    amplitude exceeds this; every other row the step reached is banned for good -- including the rows
    the ``slaterWeightMin`` cut dropped -- so no rejected row ever enters the retained set. At
    ``slaterWeightMin=0`` that keeps the recurrence the exact Lanczos of the final retained set under
    every reorthogonalization mode; at ``slaterWeightMin>0`` the cuts inside the support still perturb it,
    as in every capped run. The decision is taken once, on the step's summed output, so it works with
    ``APPLY_ROW_CHUNKS`` > 1. Exclusive with the support freeze. 0 (default) is off.""",
)

GF_ADMIT_FIRST_SHELL_TOL = Knob(
    name="GF_ADMIT_FIRST_SHELL_TOL",
    kind="float",
    default=0.0,
    minimum=0.0,
    group="bicgstab",
    doc="""Importance-pruned admission (``GF_BICGSTAB_ADMISSION=outer``, ``GF_LANCZOS_ADMIT_TOL``) keeps
    the seeds' first H-shell whole by default (0): that is what keeps the moments of G through
    ``H^2``, hence the ``Sigma`` tail, exact. A determinant reached from the seeds by a coupling
    ``V`` still enters that shell, however small ``V`` is, so a bath level hybridized at 1e-5 puts
    its whole hole-space in the start set. Above 0, first-shell rows with amplitude below this
    (relative to the seed norm) are pruned like any other: the ``H^2`` moment then errs by at most
    the summed squared amplitudes of what was dropped.""",
)

GF_SUPPORT_TRACE = Knob(
    name="GF_SUPPORT_TRACE",
    kind="bool",
    default=False,
    group="bicgstab",
    doc="""Diagnostic. Print one ``GF support step k: ...`` line per block of a sparse Lanczos Green's-function
    unit (rank 0): the retained support, the new rows, the weight on them (leak), the weight the
    ``slaterWeightMin`` prunes cut, ``||Hq||^2`` and the convergence monitor's last change. The same numbers are
    kept in ``cap_info["support_trace"]``; the lines let a production log be replayed offline against any
    freeze rule. It never changes a result.""",
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

APPLY_ROW_CHUNKS = Knob(
    name="APPLY_ROW_CHUNKS",
    kind="int",
    default=4,
    minimum=1,
    group="units",
    doc="""How many row chunks a sparse matvec splits its reference block into before applying `H` and
    redistributing: the block-Lanczos step's `H q` (`_lanczos_step.pxi`) and the CIPSI selection round
    (`CIPSISolver._apply_block_and_redistribute`). `1` applies the whole block at once. With `n` chunks one
    chunk's rows are applied, redistributed and accumulated in turn, so only one chunk's raw apply output,
    packed send buffer and receive buffer are alive at a time (measured on SrMnO3: the one-shot selection
    round holds 6x the owned candidate block, 243 MiB vs 673 at 4 chunks; `doc/plans/dc_smo_memory.md`,
    round 6). The result is exact up to floating-point summation order -- the same class of difference a
    change of MPI rank count makes. The memory model prices the saving (`memory_estimate._gf_chunk_divisor`),
    so changing this changes every cap derived from it. Ignored when the matvec has no redistribute to
    bound (serial). Costs `n` operator walks over the rows in total, not `n` times the work.""",
)


GF_SCHEDULER = Knob(
    name="GF_SCHEDULER",
    kind="str",
    default="queue",
    group="units",
    doc="""How Green's-function units are handed to the MPI colors. ``queue`` (default): equal-width
    colors take the next unit, largest excited sector first, from a shared counter as they go idle
    (:mod:`impurityModel.ed.work_queue`), so a wrong cost prediction costs at most the tail instead
    of a whole color's backlog. ``static``: packed up front by predicted cost (LPT on
    ``unit_cost_weights``), each color's rank count proportional to its packed cost. Measured on the
    SrMnO3 self-energy, the queue cut the GF phase 1.32x at 64 ranks and matched ``static`` at 128
    (doc/plans/gf_load_balancing.md). Read on rank 0 and broadcast.""",
)

GF_CSR_MEMORY_SAFETY = Knob(
    name="GF_CSR_MEMORY_SAFETY",
    kind="float",
    default=0.75,
    minimum=0.0,
    group="units",
    doc="""Fraction of the rank's whole share of node RAM (``safety * (available + resident)``, the
    form ``GS_MEMORY_BUDGET_SAFETY`` uses) the frozen ``P H P`` recurrence (``GF_FROZEN_CSR``) may
    reach. Separate from, and looser than, the guard's fraction that sizes the auto cap: the auto cap
    fills memory on the sparse path, and the frozen matrix then needs more than that path's half
    of the share, so at the guard's 0.5 every capped unit was declined (SrMnO3, auto cap 18.6M: 20
    of 20, GF 39 ks instead of ~1.4 ks). The fit check's estimate over-predicted the measured peak in
    all 76 frozen units of the 2026-10 cluster runs (measured/predicted 0.71-0.91), and a decline
    costs only speed -- the sparse kernel continues -- so the decision may use more of the share
    than the cap does. Ignored when the guard is off (``GS_MEMORY_BUDGET_SAFETY=0``).""",
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

GF_TOL = Knob(
    name="GF_TOL",
    kind="float",
    default=None,  # unset: max(slaterWeightMin**2, 1e-9) -- gf_convergence._gf_rel_tol
    group="convergence",
    doc="""Relative-change tolerance of the block-Lanczos Green's function on every evaluation
    axis (``SolverOptions.gf_tol``; an explicit option wins). Unset keeps the default
    ``max(slaterWeightMin**2, 1e-9)``. The monitor stops a work unit once two consecutive checks
    move ``G`` by less than this, relative to ``max|G|`` on the axis.""",
)

GF_REAL_TOL = Knob(
    name="GF_REAL_TOL",
    kind="float",
    default=None,  # unset: the same tolerance as the Matsubara axis (GF_TOL / its default)
    group="convergence",
    doc="""Relative-change tolerance on the **real-frequency** axis only
    (``SolverOptions.gf_real_tol``; an explicit option wins). Unset uses the ``GF_TOL`` value.
    The real axis at broadening ``delta`` sets the Lanczos depth of a production self-energy
    run, and a looser real-axis tolerance shortens every unit without touching the Matsubara
    accuracy. Loosen it, never drop it: RSPt builds the next bath fit's hybridization from the
    real-axis self-energy (``G0^-1 = G^-1 + sig_real``), so its error enters the DMFT loop.
    ``1e-6`` measured ~7e-6 relative error at 1.6-1.8x fewer blocks.""",
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

# --- Ground state: block-Lanczos width -------------------------------------------------------

GS_MAX_BLOCK_WIDTH = Knob(
    name="GS_MAX_BLOCK_WIDTH",
    kind="int",
    default=None,  # unset = uncapped (today's behaviour)
    minimum=1,
    group="groundstate",
    doc="""Caps the ground-state array-kernel Lanczos block width (`cipsi_solver.CIPSISolver.
    get_eigenvectors`'s warm-start block, `p = len(psi_refs) + 1`). Unset (the default) leaves
    it uncapped, which is what grows unboundedly through `expand`'s own warm-start feedback loop
    (`num_wanted = 2 * len(psi_refs)` feeding back into the next solve's block width) -- measured
    on SrMnO3 reaching p=105 (sweep) / 315 (TRLM's retained block after a restart) at the
    ~1M-determinant production cap, and this is what the memory model (`memory_estimate`,
    `estimate_gs_peak_bytes`'s `krylov_bytes` term) has to size for once its remaining call
    sites stop defaulting `block_width=4`. Only the warm block fed *into* the next Lanczos solve
    is truncated (to the lowest `min(len(psi_refs), GS_MAX_BLOCK_WIDTH)` energies -- already
    ordered ascending); the cold full-support vector is still appended (a correctness guard, not
    an optimization: it keeps every charge sector reachable), and `num_wanted` -- how many states
    the caller actually asked for -- is untouched, so `expand`'s manifold does not shrink, only
    the block width the solver uses to find it. See `doc/plans/dc_smo_performance.md`'s Phase 4
    discussion for the wall-clock/gap-centre-stability gate a chosen value should pass before it
    becomes the default rather than an opt-in override.""",
)

GS_SECTOR_CERTIFY = Knob(
    name="GS_SECTOR_CERTIFY",
    kind="bool",
    default=True,
    group="groundstate",
    doc="""Certify every sector solve (`sector_solver`): split the seed by the hard symmetry labels
    ``H`` is detected to conserve (``2 S_z``, the occupation parities), partition each label's seed
    into the classes of ``P H P`` and ``P H Q H P`` (no symmetry assumed), expand every class inside
    the seed window, and keep the lowest. A single expansion from the seed's lowest state never
    leaves that state's symmetry class, and on bath-fitted impurities the seed's class is routinely
    the wrong one -- on the NiO 10-bath archive every sector from d5 to d9 converged on an excited
    level, up to 0.17 high. ``0`` restores the single expansion.""",
)

GS_CERTIFY_MAX_SEED = Knob(
    name="GS_CERTIFY_MAX_SEED",
    kind="int",
    default=1000,
    minimum=0,
    group="groundstate",
    doc="""Largest seed (determinants) a sector solve certifies. The certification plan is built
    from dense seed matrices on rank 0 (``n0^2`` complex, twice), and a search's plan store keeps
    each certified sector's class spans on every rank (up to ``n0^2`` complex per sector); a larger
    seed is solved by a single expansion, uncertified.""",
)

GS_CERTIFY_WINDOW = Knob(
    name="GS_CERTIFY_WINDOW",
    kind="float",
    default=None,
    minimum=0.0,
    group="groundstate",
    doc="""Energy window, above the lowest seed class of a hard label, inside which the sector
    certifier expands a seed symmetry class (Hamiltonian units). Unset, it is
    `GS_CERTIFY_WINDOW_FACTOR` times the largest hybridization spread sqrt(<v|PHQHP|v>) of the
    classes' lowest seed states.""",
)

GS_CERTIFY_WINDOW_FACTOR = Knob(
    name="GS_CERTIFY_WINDOW_FACTOR",
    kind="float",
    default=3.0,
    minimum=0.0,
    group="groundstate",
    doc="""Multiple of the largest seed hybridization spread that sets the certifier's seed-class
    window when `GS_CERTIFY_WINDOW` is unset. Hybridization lowers a seed state by about sigma^2 / Delta,
    which reaches sigma as the charge-transfer gap Delta closes; three times the largest sigma
    covers every crossover that a single hybridization shell can drive, with margin.""",
)

GS_SELECTION_CHUNK = Knob(
    name="GS_SELECTION_CHUNK",
    kind="int",
    default=None,  # unset = unchunked (today's behaviour): the whole (p, n_Dj) score stack at once
    minimum=1,
    group="groundstate",
    doc="""Caps how many reference rows (`p`, `len(psi_ref)` in `CIPSISolver.determine_new_Dj`)
    the CIPSI selection round's Epstein-Nesbet score computation processes at once
    (`cipsi_solver._score_candidates`), instead of materializing the whole `(p, n_Dj)` de2/mask
    temporary stack in one shot. `n_Dj` here is this rank's **local** candidate count, not the
    global one the cycle log reports (that field is an `_allreduce_sum`), so the stack this knob
    bounds is smaller than the log suggests by roughly the rank count over the routing skew: see
    `doc/plans/dc_smo_memory.md`, written against the SrMnO3 double-counting search that was
    OOM-killed with a selection round at p~68-104. **Measured verdict: leave it unset -- it is a
    no-op at production rank counts.** At 1 rank, once the CIPSI candidate energies became exact
    and this site started setting the peak, the knob moved a full solve 1306.3 -> 1109.6 MiB at
    chunk 8 (-15%) with `e0` bit-identical. That did not survive contact with the cluster: on the
    128-rank SrMnO3 gap-DC run, a chunk-8 job and an unchunked one hit the *identical* memory-guard
    message (a 2.9 GiB round transient over a 1.9 GiB resident set) and tightened to the *same*
    3,630,778 determinants.

    **The reason is not that the score stack is small, and the verdict is conditional on
    `GS_APPLY_ROW_CHUNKS`.** A 128-rank site ledger on the N-1 sector (`p` around 320) measures
    `_score_candidates` growing by **1,331 MiB**: `n_Dj` is rank-local, but `p` is not, and it
    reaches the hundreds at production, so the `(p, n_Dj)` stack is over a gigabyte. At the
    default `GS_APPLY_ROW_CHUNKS=4` bounding it still buys nothing, because it is not the
    peak-setter -- `_apply_block_and_redistribute` reaches **5,508 MiB** absolute against
    `_score_candidates`' 4,747. **At `GS_APPLY_ROW_CHUNKS=8` that inverts**: the apply drops to
    4,323 MiB and `_score_candidates` becomes the high-water mark at 4,778, with
    `_candidate_overlaps_and_energies` next at 4,114. So on a workload whose `p` is in the
    hundreds, set the apply chunking first; this knob is then the one acting on the peak, and
    its remaining headroom is the ~660 MiB between those two sites. Measured at 128 ranks, cap
    2,745,510 (`doc/plans/dc_smo_memory.md`, round 9 step 7). Chunking happens on
    group boundaries only
    (`_degenerate_groups`) -- a degenerate manifold is never split across a chunk -- which is what
    keeps the result exact: the manifold-summed score is `max over independent groups of
    (group-summed de2)`, and an elementwise running max over already-processed groups equals
    stacking every group and maxing once at the end. Unset (the default) processes every group in
    one chunk, i.e. today's behaviour, bit-for-bit; a chosen chunk size should come from the same
    width sweep that sets `GS_MAX_BLOCK_WIDTH`, not from a guess.""",
)


GS_MEMORY_BUDGET_SAFETY = Knob(
    name="GS_MEMORY_BUDGET_SAFETY",
    kind="float",
    # Derived from `memory_estimate.DEFAULT_MEMORY_SAFETY` rather than repeating 0.5: one source of
    # truth for "what fraction of available RAM is it safe to be holding" (see the no-duplicated-
    # tolerance-literals convention). 0 disables the trip-wire.
    default=None,
    minimum=0.0,
    group="groundstate",
    doc="""Fraction of the rank's **whole share** of node RAM at which a CIPSI ground-state
    expansion stops growing its basis, measured against its own peak RSS rather than against a
    predicted one (`CIPSISolver.expand`'s `memory_budget_bytes`). This applies when the cap was
    derived from memory (`auto`) or there is none (`unlimited`). A `truncation_threshold` the user
    set is final: the guard then only warns, on stdout and stderr, when the budget is reached.
    The same fraction budgets the Green's-function units' measured guard and sizes their auto cap
    (`gf_units`); `0` switches both guards off. Unset uses
    `memory_estimate.DEFAULT_MEMORY_SAFETY`; `0` disables the guard and restores the pre-2026-09
    behaviour, in which an uncapped expansion grows until the kernel OOM-kills the rank.

    The share is `memory_estimate.absolute_rss_budget` = `safety * (available + resident)`, not
    `safety * available_bytes_per_rank`. `available` is `MemAvailable / ranks_on_node`, i.e.
    memory that is *free* and already net of what this process holds, so a fraction of it is an
    **increment** allowance -- and both guards below measure **absolute** RSS. Budgeting one
    against the other pinned every SrMnO3 gap-DC sector after the first at its seed basis
    (`doc/plans/dc_smo_memory.md`, round 9). `GS_MEMORY_BUDGET_INCLUDE_RESIDENT=0` rolls that
    arithmetic back without switching the guard off.

    This exists because the *predictive* path cannot be trusted at scale. Measured on the SrMnO3
    double-counting search that was OOM-killed (`doc/plans/dc_smo_memory.md`):
    `estimate_gs_peak_bytes` under-predicted the per-rank peak by 4-5x on a laptop-sized basis and
    by **372-1028x** at 256 ranks, because the dominant term is not proportional to a rank's own
    determinant count -- a 14.5x spread in local determinants moves per-rank RSS by ~10%. A
    trip-wire on measured RSS is indifferent to every one of those modelling errors, which is why
    it is the default rather than a tuning option.

    Two guards share the budget, both on measured RSS. The **look-ahead** one
    (`cipsi_solver._memory_growth_bound`) resets the process high-water mark before every CIPSI
    selection round, measures that round's own transient, and caps the admission so the *next*
    round -- whose cost scales with the basis this admission creates and with the reference-block
    width -- is predicted to fit; when it binds, the affordable size becomes the fixed budget. It
    exists because the after-the-fact trip-wire cannot catch an expansion that admits everything:
    the basis grows 5-10x per cycle, so the crashed SrMnO3 run read 2.4 GiB against a 2.5 GiB
    budget and was killed at 5.8 GiB one cycle later. The **after-the-fact** trip-wire stays as a
    backstop: the first cycle whose measured peak reaches the budget tightens the cap to the current
    basis size. The two overlap by construction -- the look-ahead reports a `"budget"`-reason zero
    only when the round's starting RSS is already over budget, and the trip-wire's `peak_rss` is
    never below that -- so in that case the backstop has always fired first. Both only ever
    *tighten* a memory-derived cap, never loosen it, never touch a cap the user set, and a run
    that stays under budget is bit-identical to one without the guard.""",
)


GS_MATVEC_EXCHANGE_BYTES = Knob(
    name="GS_MATVEC_EXCHANGE_BYTES",
    kind="int",
    default=64 * 2**20,
    minimum=16,
    group="groundstate",
    doc="""Per-buffer byte bound on one round of the sparse graph matvec exchange.
    The send buffer holds one row block per destination neighbour and the receive buffer one
    `(local rows x columns)` block per source neighbour, so both scale with the graph *degree*; the
    exchange is chunked over the block's columns so that neither exceeds this bound (peak 2x, both
    are alive at once). This is a correctness guard, not a tuning: on a complete graph (every small
    rank count, or a Hamiltonian whose terms reach every rank) an unchunked send buffer would be the
    whole `(global rows - local rows) x width` product -- the several-GiB-per-rank buffer the
    row-chunked reduce-scatter was introduced to remove (`doc/plans/dc_smo_performance.md`). 64 MiB
    is at most +128 MiB per rank, 2.5% of the 5.2 GiB/rank the SrMnO3 double-counting job runs at;
    the production exchange (`GS_MAX_BLOCK_WIDTH=5`, 28 neighbours) is ~3 MiB and runs in one round,
    and the exchange is bandwidth-bound so extra rounds cost little down to a few MiB. Clamped at
    1 GiB at the call site so `Neighbor_alltoallv`'s C-int element counts cannot overflow.""",
)


GS_NUM_WANTED = Knob(
    name="GS_NUM_WANTED",
    kind="int",
    default=None,
    minimum=1,
    group="groundstate",
    doc="""Eigenstates the ground-state CIPSI solve actually converges, used *only* to size memory
    estimates (`memory_estimate.estimate_gs_peak_bytes`'s `num_wanted`). It does not change what the
    solver computes -- `CIPSISolver.expand` derives that from the thermal manifold it measures.

    Unset, `estimate_gs_peak_bytes` assumes `2 * block_width` (~10 at the production
    `GS_MAX_BLOCK_WIDTH=5`), and `log_memory_budget` warns that it is guessing. That guess approved
    the cap behind the SrMnO3 double-counting OOM: the pre-`selection_bytes` model predicted
    2.51 GiB/rank at `truncation_threshold=119,555,328` against 5.0 GiB available, where the
    manifold the run actually reached (222) predicts 8.43 GiB.

    **What it buys now, measured.** With `selection_bytes` in the model that particular cap is
    already refused on every path, so the remaining effect is on the cap the search *chooses*: at
    256 ranks and a 2.5 GiB budget, `suggest_truncation_threshold` returns 43,570,432 unset against
    31,447,552 at 105 and 23,396,096 at 222 -- a 1.4-1.9x reduction. **It does not make the cap
    safe**: all of those remain 24-46x above the 949,834 determinants that actually exhausted
    memory, because the estimate is structurally low by 372-1028x at that rank count. The
    measured-RSS trip-wire (`GS_MEMORY_BUDGET_SAFETY`) is the mitigation that does not depend on the
    model being right; this one narrows the gap it has to cover (`doc/plans/dc_smo_memory.md`).

    Supply the value measured by the same width sweep that sets `GS_MAX_BLOCK_WIDTH`, in the same
    place (`job.rspt`). For SrMnO3 the kept manifold grows with basis size and saturates near 105
    (44 at cap 20,000, 87 at 100,000, ~105 at 949,834), so it is a per-workload number and
    deliberately has no default: a wrong default here silently resizes every cap, and the existing
    warning is a better failure mode than a confident guess.""",
)


# --- Double counting: search diagnostics -----------------------------------------------------

DC_DIAGNOSTICS = Knob(
    name="DC_DIAGNOSTICS",
    kind="bool",
    default=False,
    group="double-counting",
    doc="""Everything about a double-counting calculation beyond its one-block result record.
    Every criterion emits that record unconditionally (:mod:`impurityModel.ed.dc_record`); this
    knob adds the two things too bulky to print every time. **Where the time went:** a search
    evaluates its observable dozens of times and each evaluation runs a whole ground-state
    occupation walk underneath, so the record's ``walltime`` says nothing about which of build /
    CIPSI expansion / diagonalization dominates, nor how many sector solves were redundant --
    with this set the search runs inside a ``solver_trace.tracing()`` block and prints a
    per-trial-``mu`` table plus per-kind aggregates on rank 0. **What the dc matrix looks like:**
    the full ``dc_guess`` and ``dc`` matrices, which the record can only summarise by their
    per-orbital level, and which matter for a scheme whose ``dc`` is not a uniform shift
    (``sigma_inf``, or a ``dc_guess`` arriving from RSPt). Off by default: the accounting is free
    (one ``is None`` test per hook) but the report is several lines per ``mu`` and two complex
    matrices per call.""",
)


# --- Double counting: the determinant-cap ladder ----------------------------------------------

DC_ALLOW_MEMORY_BOUND = Knob(
    name="DC_ALLOW_MEMORY_BOUND",
    kind="bool",
    default=False,
    group="double-counting",
    doc="""Accept a double counting whose charge-sector ground states were stopped by the
    **memory guard** instead of by convergence. Unset, such a search raises
    `dc_search.DoubleCountingUnreachable` rather than returning a number.

    The failure this exists for: the SrMnO3 cubic gap-DC run reported `status = converged` and
    handed `dc_level = 0.580041` to the DMFT loop from ground states of 10 to 252 determinants,
    because the memory guard had pinned every sector after the first at its seed basis. The
    energies the bisection ran on were those of whatever basis each expansion happened to reach.
    See `doc/plans/dc_smo_memory.md`, round 9.

    Note what does *not* trip this: a `truncation_threshold` that binds. Spending a determinant
    budget deliberately is the normal mode, and it is reported separately as `dc_cap_bound`.
    Only `truncation_report["memory_bound"]` -- the rank actually running out of room -- gates
    the answer.

    Set it when you want the number anyway, knowing it is not converged in the basis.""",
)


DC_DE2_MIN = Knob(
    name="DC_DE2_MIN",
    kind="float",
    default=None,  # unset = GS_DE2_MIN (0: no floor)
    minimum=0.0,
    group="double-counting",
    doc="""Per-determinant Epstein-Nesbet PT2 floor for the double-counting search's **charge-sector
    solves** (`dc_criteria`'s `solve_sector` calls). Unset uses the solver's own
    `BasisOptions.de2_min` (TOML `[many_body_basis].de2_min`), else :data:`groundstate.GS_DE2_MIN`,
    which is 0: no floor. The double-counting line's / criterion table's own `de2_min` beats it. The solves are converged by `DC_E_PT2_TOL` instead; a floor bounds each
    candidate it refuses but not their sum, so it is a way to *loosen* a solve, not to converge one.

    The notes below predate that change: they were measured when this floor, at 1e-8, was the only
    selection criterion.

    **Why this is separable, and when to loosen it.** Matching `GS_DE2_MIN` buys *parity* -- the DC
    determined on the same variational space as the self-energy run that consumes it -- not
    accuracy; `groundstate`'s own note records the measurement, on `nio_5peeled`: moving these
    sector energies from `1e-6` to `1e-8` shifts the gap centre 0.3 meV, i.e. ~2.7 meV in `mu`
    after the `1 / |chi|` amplification, **at 4x the cost**, while raising the determinant cap from
    2,000 to 8,000 shifts it ~54 meV. Truncation drift dominates; the PT2 floor does not.

    Weigh 2.7 meV against the tolerance the search itself already admits. On the SrMnO3 gap-DC the
    record reports `chi = -0.4537` with *"dc determined to +- 5.51e-02 by the search tolerance
    alone"* -- so at `1e-8` the sector energies are converged roughly **20x finer than the search
    resolves**, for 4x the time and a proportionally larger basis.

    **And on that workload the parity argument does not apply**, because parity is not achieved:
    the production ground state was PT2-converged at 200,565 determinants (`candidates=0`), while
    the DC's N-1 sector was *memory-bound* at 2,745,510 with 1.6 of PT2 importance discarded. What
    sets that sector's space is the memory guard, not this threshold. Loosening it there costs
    little that was being bought, and shrinks a basis whose size is the reason the search needs
    64 ranks per node (`doc/plans/dc_smo_memory.md`, round 9).

    Keep the default when the DC's answer is the deliverable and cost is not binding; raise it to
    `1e-6` when the search is what stands between you and a self-energy. It loosens the search's
    sector walk with it (the walk is the search's own sector solves), not the production ground
    state.""",
)

DC_GAP_LEVER_MIN = Knob(
    name="DC_GAP_LEVER_MIN",
    kind="float",
    default=0.1,
    minimum=0.0,
    group="double-counting",
    doc="""Smallest response of the gap centre to the double-counting shift the `gap` criterion
    searches on: ``(delta_+ + delta_-) / 2``, the electrons the two gap edges move onto the
    impurity, measured at the guess from solves the search has already paid for. Below it, and with
    the guess not already converged, the criterion raises `DoubleCountingUnreachable` at once,
    with the measured edges, instead of searching: the centre is then a property of the bath, not
    of the impurity, and the shift it asks for is far outside what the solves at the guess say
    anything about. On the NiO 10-bath archive the lever is 0.02 (impurity-like edges give 0.5-1),
    and the search ran 22 evaluations without converging. `0` searches anyway.""",
)

DC_E_PT2_TOL = Knob(
    name="DC_E_PT2_TOL",
    kind="float",
    default=None,  # unset = GS_E_PT2_TOL, the production ground state's own tolerance
    minimum=0.0,
    group="double-counting",
    doc="""Residual Epstein-Nesbet PT2 energy the double-counting search's **charge-sector solves**
    are converged to (`CIPSISolver.expand`'s `e_pt2_tol`). Unset uses the solver's own
    `BasisOptions.e_pt2_tol` (the RSPt solver line's `e_pt2 X`), else
    :data:`groundstate.GS_E_PT2_TOL` (1e-8): the tolerance the production ground state is
    converged to, so the DC is determined on a space converged as far as the self-energy run's.
    The double-counting line's own `e_pt2 X` beats this knob. The `occupation` criterion takes
    none of these overrides, only the solver's tolerance: it solves on the production path.

    This, not `DC_DE2_MIN`, is the accuracy control: `DC_DE2_MIN` bounds each candidate left out,
    this bounds their sum -- the quantity the energy error follows (measured on a SIAM against
    exact diagonalization: the residual predicts the error to 2%). Loosen it, e.g. to `1e-5`, when
    the search is what stands between you and a self-energy; the sector energies are then
    variational upper bounds with at most that much PT2 energy left out, and a solve that stops
    short of it (cap, memory guard) warns with its residual.""",
)

# --- Self-energy: causality tolerance --------------------------------------------------------

SIGMA_CAUSALITY_TOL = Knob(
    name="SIGMA_CAUSALITY_TOL",
    kind="float",
    default=1e-3,
    minimum=0.0,
    group="sigma",
    doc="""Relative tolerance for the self-energy causality check (sigma.check_greens_function,
    called on Sigma from selfenergy._self_energy_on_mesh). Causality requires Im Sigma_ii(w) <=
    0 on every diagonal element. Since Sigma = G0^-1 - G^-1, a positive Im Sigma_ii means the
    interacting G is narrower than the bare hybridized G0 there -- a real solver artifact (basis
    truncation, a restricted excited sector, or finite pole discreteness at small delta), not
    roundoff. The normalizer is *per diagonal element*: for
    orbital i compare max_w Im(Sigma_ii(w)) against max_w |Im(Sigma_ii(w))|, worst over i -- not
    a whole-block max, which lets a wide, high-weight orbital mask a violation on a narrow one.
    Below this threshold the violation is reported (gf_diagnostics-style: violating window,
    worst value/ratio, and the Im(G0^-1) vs Im(G^-1) breakdown at the worst point) as a WARN and
    the run continues; above it the run raises, as it always has. Default 1e-3: comfortably
    above roundoff, and below the ~7.6e-3 lobe that a real 128-rank Mn self-energy run hit at a
    sharp hybridization resonance (see doc/, the Lanczos-diagnostics writeup) -- so the default
    does not silently rescue a run with a genuine artifact, only ones with a much smaller one.
    0 restores the historical always-raise behaviour.""",
)


KNOBS: dict[str, Knob] = _register(
    GF_BICGSTAB_ATOL,
    GF_BICGSTAB_RESIDUAL_CHECK,
    GF_BICGSTAB_ADMISSION,
    GF_BICGSTAB_ADMIT_SCORER,
    GF_BICGSTAB_ADMIT_TOL_AMP,
    GF_BICGSTAB_ADMIT_TOL_JACOBI,
    GF_BICGSTAB_ADMIT_ROUNDS,
    GF_BICGSTAB_ADMIT_SHELLS,
    GF_BICGSTAB_ADMIT_CARRY_TOL,
    GF_BICGSTAB_ADMIT_EN_TOL,
    GF_LANCZOS_ADMIT_TOL,
    GF_ADMIT_FIRST_SHELL_TOL,
    GF_SUPPORT_TRACE,
    GF_GMRES_RESTART,
    GF_EIGENSTATE_GROUP,
    GF_SCHEDULER,
    GF_CSR_MEMORY_SAFETY,
    GF_PER_STATE_RESTRICT,
    GF_TOL,
    GF_REAL_TOL,
    GF_SECTOR_DENSE_MAX,
    GF_SECTOR_CACHE_DIR,
    GF_KRYLOV_RECYCLE_MAX_BYTES,
    GF_RIXS_WIN_CHUNK,
    GF_RIXS_ADAPTIVE_TOL,
    GS_MAX_BLOCK_WIDTH,
    GS_SECTOR_CERTIFY,
    GS_CERTIFY_MAX_SEED,
    GS_CERTIFY_WINDOW,
    GS_CERTIFY_WINDOW_FACTOR,
    GS_SELECTION_CHUNK,
    GS_MATVEC_EXCHANGE_BYTES,
    APPLY_ROW_CHUNKS,
    GS_NUM_WANTED,
    GS_MEMORY_BUDGET_SAFETY,
    DC_ALLOW_MEMORY_BOUND,
    DC_DE2_MIN,
    DC_GAP_LEVER_MIN,
    DC_E_PT2_TOL,
    DC_DIAGNOSTICS,
    SIGMA_CAUSALITY_TOL,
)

GROUP_TITLES = {
    "bicgstab": 'Per-frequency BiCGSTAB solver (``gf_method="bicgstab"``)',
    "units": "Green's-function work-unit decomposition",
    "convergence": "Block-Lanczos convergence monitor",
    "rixs-solvers": "RIXS shift-recycling solver tiers",
    "rixs-sampling": "RIXS incoming-energy sampling",
    "groundstate": "Ground-state block-Lanczos width, matvec exchange and CIPSI selection sizing",
    "double-counting": "Double-counting search: the cap ladder and diagnostics",
    "sigma": "Self-energy causality tolerance",
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
