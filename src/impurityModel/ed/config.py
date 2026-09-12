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

GS_SELECTION_CHUNK = Knob(
    name="GS_SELECTION_CHUNK",
    kind="int",
    default=None,  # unset = unchunked (today's behaviour): the whole (p, n_Dj) score stack at once
    minimum=1,
    group="groundstate",
    doc="""Caps how many reference rows (`p`, `len(psi_ref)` in `CIPSISolver.determine_new_Dj`)
    the CIPSI selection round's Epstein-Nesbet score computation processes at once
    (`cipsi_solver._score_candidates`), instead of materializing the whole `(p, n_Dj)` de2/mask
    temporary stack in one shot. `n_Dj` -- the candidate count -- reaches the hundreds of
    thousands at production scale, so that stack (several same-shape arrays, measured
    ~50 B/element combined) is a real per-cycle memory peak: see `doc/plans/dc_smo_memory.md`,
    written against the SrMnO3 double-counting search that was OOM-killed with a selection round
    at p~68-104 and n_Dj in the hundreds of thousands. Chunking happens on group boundaries only
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
    doc="""Fraction of `memory_estimate.available_bytes_per_rank` at which an **uncapped** CIPSI
    ground-state expansion stops growing its basis, measured against its own peak RSS rather than
    against a predicted one (`CIPSISolver.expand`'s `memory_budget_bytes`). Unset uses
    `memory_estimate.DEFAULT_MEMORY_SAFETY`; `0` disables the guard and restores the pre-2026-09
    behaviour, in which an uncapped expansion grows until the kernel OOM-kills the rank.

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
    basis size. Both only ever *tighten* a caller's cap, never loosen it, and a run that stays under
    budget is bit-identical to one without the guard.""",
)


GS_APPLY_ROW_CHUNKS = Knob(
    name="GS_APPLY_ROW_CHUNKS",
    kind="int",
    default=None,  # unset = one shot (today's behaviour): apply to the whole reference block at once
    minimum=1,
    group="groundstate",
    doc="""How many row chunks `CIPSISolver._apply_block_and_redistribute` splits the reference
    block into before applying `H` and redistributing. Unset (or 1) applies to the whole block at
    once. With `n` chunks, each chunk of the local reference rows is applied, pruned, redistributed
    and accumulated into the owned candidate block in turn, so only one chunk's raw output, packed
    send buffer and receive buffer are alive at a time. Those three, plus the merged block, are
    the selection round's peak: measured on the SrMnO3 double-counting workload at 4 ranks the step
    holds **6x** the owned candidate block, and on the 256-rank job that was OOM-killed the same
    step accounted for the 2.4 -> 5.8 GiB jump in one cycle (`doc/plans/dc_smo_memory.md`, round 6).
    Exact up to floating-point summation order: a candidate reached from reference rows in
    different chunks has its partial sums added in a different order than the one-shot apply, and
    the per-column `slater_weight_min` prune acts on those partial sums -- the same class of
    difference a change of MPI rank count already makes. Costs `n` operator walks over the
    reference rows in total (each row is walked once), not `n` times the work. Off by default until
    measured on more than one workload; set it in the job script alongside `GS_MAX_BLOCK_WIDTH`.""",
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

DC_CAP_STRATEGY = Knob(
    name="DC_CAP_STRATEGY",
    kind="str",
    default="max",
    group="double-counting",
    doc="""How the double-counting search picks its determinant cap when
    `BasisOptions.truncation_threshold` was left at `None`.

    `max` (the default) runs once at the memory-derived ceiling and asks the expansion whether the
    cap bound it -- `CIPSISolver.truncation_report is None` means it did not, i.e. the expansion
    ran out of candidates above `de2_min` rather than out of budget, so the basis is already the
    one any larger cap would build and the answer cannot move by raising the cap. That is an
    exact per-sector test from a single evaluation. Only when something *did* bind is a second
    rung at half the cap evaluated, to put a measured number on how far from converged the answer
    is. Retreats by halving if the ceiling cannot be evaluated (`MemoryError`, broadcast so every
    rank retreats together; a hard OOM kill is not catchable).

    `ladder` is the previous behaviour: double the cap from `DC_CAP_LADDER_START` until the
    criterion's own answer stops moving (`DC_CAP_LADDER_MAX_RUNGS`, `CAP_CONVERGENCE_RUNS`).

    **Why the default changed.** Measured on SrMnO3 cubic at 6 ranks, one gap-centre evaluation:
    the full ladder 32k->512k costs 4299 s to reach a conclusion the 1890 s top rung reports
    directly, and the N+1 sector returns a *bit-identical* energy at 128k, 256k and 512k because
    it self-limits at 88,164 determinants -- it never binds, so no ladder rung above 128k could
    have told anyone anything. Only N-1 binds. The ladder also cannot distinguish "the cap is the
    limit" from "de2_min is the limit", which is the question a caller actually has to act on.

    Keep `ladder` when the operating cap matters more than the answer -- it accepts the *smallest*
    sufficient cap and so makes every subsequent trial-mu evaluation cheaper, where `max` runs
    them all at the ceiling. That trade is only worth taking when the answer is known to converge
    below the ceiling; on a workload that is still truncation-limited there (SrMnO3 is, up to at
    least 512,000) the ladder buys nothing and costs 2.3x.""",
)

DC_CAP_LADDER_START = Knob(
    name="DC_CAP_LADDER_START",
    kind="int",
    default=500,
    minimum=1,
    group="double-counting",
    doc="""First rung of the determinant-cap ladder :func:`dc_search.calibrate_truncation_threshold`
    climbs, doubling, until the criterion's answer stops moving (see ``DC_CAP_LADDER_MAX_RUNGS``).
    Small enough that the first rungs are cheap relative to the production caps the ladder exists
    to avoid, so a workload that settles quickly pays almost nothing for the calibration. Raising
    it skips rungs known to be meaningless on a given workload -- SrMnO3 discards PT2 weight of
    order 1 at caps of 500-2,000 -- but the saving is small and the ladder's *reach* is what
    usually matters: a rung costs roughly linearly in its cap (measured ``cap**0.98``), so the
    whole geometric ladder costs about twice its top rung whatever the first one is. It is not a
    free change either -- it raises the smallest cap the ladder can *accept*, which moves the
    answer on workloads that do settle early.""",
)

DC_CAP_LADDER_MAX_RUNGS = Knob(
    name="DC_CAP_LADDER_MAX_RUNGS",
    kind="int",
    default=11,
    minimum=1,
    group="double-counting",
    doc="""How many times the determinant-cap ladder doubles before giving up and reporting its
    largest rung as truncation-limited rather than search-limited. The reachable ceiling is
    ``DC_CAP_LADDER_START * 2**(N-1)`` -- 512,000 determinants at the defaults. The memory-derived
    cap bounds the ladder as well and is usually far above that (1.35e8 on a 128-rank SrMnO3 run),
    so on a workload whose answer has not settled it is *this* budget that binds: the pre-2026-09
    default of 8 stopped the ladder at 64,000 with the answer still moving, and the record could
    not distinguish that from a converged one. Cost is what this trades, and not only the
    calibration's: ``dc_criteria._calibrate_cap`` applies the accepted cap to
    ``ctx.truncation_threshold``, so it is the operating cap of every subsequent trial-``mu`` too.
    A rung costs roughly linearly in its cap (measured ``cap**0.98``), so raising the ceiling from
    64,000 to 512,000 is ~7.7x on the dominant cost of a search whose answer never settles --
    which is why this is a knob and not just a bigger constant. Lower it when a search has to fit
    inside a per-iteration time budget; raise it when the record's ``dc_cap_status`` comes back
    ``rung_budget``, which is the ladder saying in as many words that it stopped because it ran
    out of doublings and the answer is still truncation-limited. (``memory_cap`` there means the
    opposite: the run is already at the largest cap it can afford and no rung budget will lift
    it.)""",
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
    GS_MAX_BLOCK_WIDTH,
    GS_SELECTION_CHUNK,
    GS_APPLY_ROW_CHUNKS,
    GS_NUM_WANTED,
    GS_MEMORY_BUDGET_SAFETY,
    DC_CAP_STRATEGY,
    DC_CAP_LADDER_START,
    DC_CAP_LADDER_MAX_RUNGS,
    DC_DIAGNOSTICS,
    SIGMA_CAUSALITY_TOL,
)

GROUP_TITLES = {
    "bicgstab": 'Per-frequency BiCGSTAB solver (``gf_method="bicgstab"``)',
    "cipsi": 'Per-frequency CIPSI-selected solver (``gf_method="cipsi"``)',
    "sliced": 'Spectrum slicing (``gf_method="sliced"``)',
    "units": "Green's-function work-unit decomposition",
    "convergence": "Block-Lanczos convergence monitor",
    "rixs-solvers": "RIXS shift-recycling solver tiers",
    "rixs-sampling": "RIXS incoming-energy sampling",
    "groundstate": "Ground-state block-Lanczos width and CIPSI selection sizing",
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
