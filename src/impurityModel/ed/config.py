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

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Callable

#: The matrix size below which an eigensolve is dense -- owned here, once per role (P6). ``DEFAULT_DENSE_CUTOFF`` is
#: what a user-facing entry gets when nothing is set (``SolverOptions``, the input schema, which mirrors it as a leaf
#: module -- a test holds them equal); ``GS_DENSE_CUTOFF`` is the library-level fallback of the ground-state functions
#: and of an archive that records none. They differ for historical reasons; changing either changes the bases.
DEFAULT_DENSE_CUTOFF = 500
GS_DENSE_CUTOFF = 1000

#: Largest tolerated error bar, in electrons, on the ensemble-averaged impurity occupation that the solved sectors'
#: residual PT2 energies leave (sum over members of p_s |N_s - <N>| err_s / tau). Above it the charge-sector
#: ensemble is refused (``mixed_valence``: cut at the failing member, or the winner alone, with a warning).
ENSEMBLE_WEIGHT_ERROR_MAX = 0.05

#: A charge-sector ensemble member whose weight is below this gets no Green's function: the self-energy merges
#: the other members, renormalised, and says so at every verbosity. The weight is the plan's (walk energies,
#: degeneracy ignored), the only one known before the member's Green's function would be built. A member at
#: 1e-4 (NiO 10-bath, N = 9) cost a full Green's function, ~3850 s, for a 1e-4 contribution (review PERF-MV-a).
#: The member still enters the ground-state report, ``thermal_rho`` and ``rhos``.
ENSEMBLE_GF_MIN_MEMBER_WEIGHT = 1e-3

__all__ = ["KNOBS", "Knob", "NonUniformKnobError", "assert_uniform", "dump"]


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
    uniform : bool
        Whether every MPI rank must read the same value (:func:`assert_uniform`). True for
        everything that can steer a collective; False only for a knob that is legitimately
        per-rank, such as a scratch path.
    """

    name: str
    kind: str
    default: Any
    doc: str
    minimum: Any = None
    group: str = "general"
    uniform: bool = True

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

#: Accepted ``GS_LOWEST_SOLVER`` values (see that knob).
GS_LOWEST_SOLVERS = ("auto", "davidson", "trlm")

#: Accepted ``SolverOptions.sigma_method`` values: the self-energy estimators of
#: :mod:`impurityModel.ed.sigma_estimators` (its ``ESTIMATORS`` registry has exactly these keys).
SIGMA_METHODS = ("dyson",)


def effective_ranking_cap(cap):
    """The cap pass 1 of a certification ranks under, or ``None`` (rank at the full cap).

    ``GS_RANKING_CAP``: ``off`` (``None``), ``auto`` (``max(cap // 8, 20000)``) or a positive integer; never
    above ``cap``, and ``None`` for an infinite cap. Resolved from the cap in force, so it follows the DC
    ladder. A malformed value raises ``ValueError`` naming the knob.
    """
    import math

    setting = str(GS_RANKING_CAP.get()).strip().lower()
    if setting in ("off", "0", "none", "false") or cap is None or not math.isfinite(cap):
        return None
    if setting == "auto":
        ranking = max(int(cap) // 8, 20_000)
    else:
        try:
            ranking = int(float(setting))
        except ValueError:
            ranking = 0
        if ranking <= 0:
            raise ValueError(f"GS_RANKING_CAP must be 'auto', 'off' or a positive integer, got {setting!r}")
    ranking = min(ranking, int(cap))
    return None if ranking >= int(cap) else ranking


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

GF_SUPPORT_TRACE = Knob(
    name="GF_SUPPORT_TRACE",
    kind="bool",
    default=False,
    group="bicgstab",
    doc="""Diagnostic. Print one ``GF support step k: ...`` line per block of a sparse Lanczos Green's-function
    unit (rank 0): the retained support, the new rows, the weight on them (leak), the weight the
    ``slaterWeightMin`` prunes cut, ``||Hq||^2``, the largest amplitude squared landing on a new row and the
    convergence monitor's last change. The same numbers are
    kept in ``cap_info["support_trace"]``; the lines let a production log be replayed offline against any
    settle rule. It also switches on the end-of-unit frontier probe of a settled unit (one extra unfrozen matvec).
    It never changes a result.""",
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


ROUTING_HASH = Knob(
    name="IMPURITYMODEL_ROUTING_HASH",
    kind="str",
    default="nibble",
    group="groundstate",
    doc="""Which hash decides determinant ownership (``routing_hash() % comm.size``): ``nibble`` (default) or
    ``orbital``. ``nibble`` weights each 4-orbital nibble by its popcount, so on an LMO-shaped basis it takes only
    ~1,500 distinct values and the busiest rank holds 2.6x the mean at 240 ranks (logged ``local[min,max]``);
    its communication graph is sparse at ground-state rank counts (57-80 peers of 240). ``orbital`` gives every
    orbital its own 64-bit weight: balanced to within 1.05x at every rank count tried (8-240), same cost per
    determinant, still an additive hash (a hop shifts it by a determinant-independent constant, which is what
    ``MatvecExchangePlan`` relies on), but a denser graph (every peer at 240 ranks, ~7.6 MiB/rank of MPI
    state on arrhenius). At GF colour widths (<= 40 ranks) the nibble graph is already complete.
    Read once per process by ``SlaterDeterminant::routing_hash`` (C++), so set it before Python starts;
    every rank must agree (:func:`assert_uniform`). Any other value aborts the process.""",
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

GF_QUEUE_COLOR_MARGIN = Knob(
    name="GF_QUEUE_COLOR_MARGIN",
    kind="float",
    default=2.0,
    minimum=1.0,
    group="units",
    doc="""Queue colour-count bound: with ``GF_SCHEDULER=queue`` no colour may be narrower than
    ``margin`` heaviest units' worth of ranks, i.e. the colour count is at most
    ``floor(sum(w) / (margin * max(w)))`` over the unit cost weights (on top of the participation-ratio
    bound). A queue colour runs whole units, so more colours than ``sum/max`` leaves the heaviest unit on a
    colour narrower than its fair share and it alone sets the makespan -- measured on the LMO logs: 18 colours
    of 13 ranks 3,128 s against 6-8 colours of 30-40 ranks 1,490 s (doc/plans/gf_load_balancing.md). The
    margin absorbs the error of the predicted weights. Static packing ignores it.""",
)

GF_CSR_MEMORY_SAFETY = Knob(
    name="GF_CSR_MEMORY_SAFETY",
    kind="float",
    default=0.9,
    minimum=0.0,
    group="units",
    doc="""Fraction of the rank's whole share of node RAM (``safety * (available + resident)``, the
    form ``GS_MEMORY_BUDGET_SAFETY`` uses) the frozen ``P H P`` recurrence may
    reach. Separate from, and looser than, the guard's fraction that sizes the auto cap: the auto cap
    fills memory on the sparse path, and the frozen matrix then needs more than that path's half
    of the share, so at the guard's 0.5 every capped unit was declined (SrMnO3, auto cap 18.6M: 20
    of 20, GF 39 ks instead of ~1.4 ks). 0.9, not 0.75: on SrMnO3 at 128 one-rank colours (2026-10-23,
    auto cap 6.96M) every first capped unit fit, but a rank's resident set stays ~1.5 GiB higher after
    a frozen unit and the share, re-sampled per unit from the node's free memory, falls as every rank
    on it does the same, so 30 of the 166 capped units were declined at 0.75 -- all of them within
    0.90 of the share -- and ran the sparse kernel at 47 s a block against 0.74 s on the matrix,
    ~9 h a unit instead of ~10 min, past the job's limit. The fit check's estimate over-predicts the
    measured peak (measured/predicted 0.71-0.91 over the 76 frozen units of the 2026-10 runs,
    0.78-0.83 on the 36 first units of that job), so 0.9 of the share is ~0.72 of it in practice,
    the margin 0.75 was meant to give. Ignored when the guard is off (``GS_MEMORY_BUDGET_SAFETY=0``).""",
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
    uniform=False,
    doc="""Directory persisting SectorResolventCache eigendecompositions across runs. Empty =
    in-memory only. With it, the dominant one-time `eigh` cost (measured ~450 s at 5565
    determinants; OpenBLAS's Hermitian eigensolvers are bound by their non-parallelizing
    reduction stage, and the measured alternatives are no faster with eigenvectors) is paid
    once per material instead of once per run.""",
)

IMPMOD_MATVEC_PROF = Knob(
    name="IMPMOD_MATVEC_PROF",
    kind="bool",
    default=False,
    group="rixs-solvers",
    uniform=False,
    doc="""Opt-in stage timer of the Green's-function recurrence (seconds, calls and the resident-set
    high-water per stage, printed to stderr every 30 s). A measurement aid: it has no collective inside
    and nothing reads its result back, so ranks may legitimately differ. The Cython kernel and
    DistributedSparseMatrix read the raw environment variable (any non-empty value switches it on).""",
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

GS_SHARPEN_MANIFOLDS = Knob(
    name="GS_SHARPEN_MANIFOLDS",
    kind="bool",
    default=True,
    group="groundstate",
    doc="""Rotate the degenerate ground manifold to the basis that diagonalises the conserved and the
    impurity symmetry labels inside it (``2 S_z``, then ``S^2``, ``L^2`` -- or ``J^2`` when ``2 S_z``
    is not conserved), so every reported state is one vector with sharp quantum numbers and the
    states ``calc_gs`` returns are those same vectors. The solver returns an arbitrary orthonormal
    basis of a degenerate manifold, which the report would otherwise have to describe state by
    state. Energies are untouched and any observable summed over the whole manifold is unchanged.
    ``0`` returns the solver's own vectors.""",
)

GS_LOWEST_SOLVER = Knob(
    name="GS_LOWEST_SOLVER",
    kind="str",
    default="auto",
    group="groundstate",
    doc="""Solver for a ``Lowest`` eigenpair above the dense size (every CIPSI cycle of an expansion).
    ``auto`` (default): Davidson with the diagonal preconditioner inside a certified class expansion,
    thick-restart Lanczos everywhere else. Davidson needed 2-4x fewer matrix-vector products than
    Lanczos on the NiO 10-bath d6 cycles, with a quarter of the stored vectors, and cut the certified
    d5 sector from 1423 s to 375 s; but it stays in the symmetry sector of its start's lowest Ritz
    vector, which a class expansion intends and a plain expansion must not (its only defence against
    its seed's trap is Lanczos growing every sector of the generic start vector). ``davidson`` /
    ``trlm`` force one everywhere. Davidson falls back to Lanczos when it does not converge; thermal
    (``Within``) solves always use Lanczos, which certifies the manifold complete.""",
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

GS_RANKING_CAP = Knob(
    name="GS_RANKING_CAP",
    kind="str",
    default="auto",
    group="groundstate",
    doc="""Determinant cap under which the sector certifier first *ranks* its symmetry classes. Pass 1 stops
    a class once its basis reaches this size (the cap binds only there: nothing is truncated) and bounds
    what it may still fall to; pass 2 continues only the classes that can still win, from their pass-1
    determinants, under the full ``truncation_threshold``. On a cap-bound system (a metal) every class
    otherwise runs to the full cap before any is compared. ``auto`` (default): ``max(cap // 8, 20000)``,
    never above the cap; ``off`` ranks at the full cap (no pass-1 stop); an integer sets it. A class
    inside the ranking tail of the lowest, or tied with a contender within the error bars, is always
    continued, so a degenerate level keeps every member.""",
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

GS_ORBIT_CLOSURE = Knob(
    name="GS_ORBIT_CLOSURE",
    kind="bool",
    default=True,
    group="groundstate",
    doc="""A `Within` (thermal) ground-state solve selects and truncates whole symmetry orbits of determinants
    (`orbits`): the closure of the detected symmetries of H and, with total spin conserved, every spin arrangement
    of a pair-occupation pattern at |2Sz| up to the seed's. The thermal states then keep every symmetry of H under a
    binding cap, at the cost of some energy; per-determinant selection splits a degenerate multiplet there.
    ``0`` selects and truncates determinant by determinant.""",
)

GS_CERTIFY_LABEL_SYMMETRY = Knob(
    name="GS_CERTIFY_LABEL_SYMMETRY",
    kind="bool",
    default=True,
    group="groundstate",
    doc="""Expand one hard label per orbit of the label symmetries (`label_symmetries`): signed
    orbital permutations -- the spin flip, and transpositions of impurity orbitals carrying their
    bath chains -- verified on the whole operator and on the basis restrictions, which map one
    label's problem unitarily onto another's. The partner labels reuse the representative's
    verdicts, and their thermal states are mapped from its determinants. On the NiO linked-chain
    iteration 1 a t2g swap made two of four labels in two sectors bit-identical repeats. The
    swap is built from the run's spin layout (`spin_pairs.spin_layout`), so a spin-polarized run
    whose impurity is grouped per spin finds it too. ``0`` expands every label.""",
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
    (`degeneracy.degenerate_groups`) -- a degenerate manifold is never split across a chunk -- which is what
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

# --- Self-energy: causality tolerance --------------------------------------------------------


KNOBS: dict[str, Knob] = _register(
    GF_BICGSTAB_ATOL,
    GF_BICGSTAB_RESIDUAL_CHECK,
    GF_SUPPORT_TRACE,
    GF_GMRES_RESTART,
    GF_EIGENSTATE_GROUP,
    ROUTING_HASH,
    GF_SCHEDULER,
    GF_QUEUE_COLOR_MARGIN,
    GF_CSR_MEMORY_SAFETY,
    GF_PER_STATE_RESTRICT,
    GF_SECTOR_DENSE_MAX,
    GF_SECTOR_CACHE_DIR,
    IMPMOD_MATVEC_PROF,
    GF_KRYLOV_RECYCLE_MAX_BYTES,
    GF_RIXS_WIN_CHUNK,
    GF_RIXS_ADAPTIVE_TOL,
    GS_MAX_BLOCK_WIDTH,
    GS_SECTOR_CERTIFY,
    GS_SHARPEN_MANIFOLDS,
    GS_LOWEST_SOLVER,
    GS_CERTIFY_MAX_SEED,
    GS_RANKING_CAP,
    GS_CERTIFY_WINDOW,
    GS_CERTIFY_WINDOW_FACTOR,
    GS_CERTIFY_LABEL_SYMMETRY,
    GS_ORBIT_CLOSURE,
    GS_SELECTION_CHUNK,
    GS_MATVEC_EXCHANGE_BYTES,
    APPLY_ROW_CHUNKS,
    GS_NUM_WANTED,
    GS_MEMORY_BUDGET_SAFETY,
    DC_GAP_LEVER_MIN,
    DC_DIAGNOSTICS,
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


class NonUniformKnobError(RuntimeError):
    """Raised on every rank by :func:`assert_uniform` when a knob reads differently on some of them."""


def _raw_knob_values() -> dict[str, str | None]:
    """This rank's raw value of every uniform knob, normalised the way :meth:`Knob.get` reads it:
    unset, or empty for a non-string knob, is ``None``."""
    values: dict[str, str | None] = {}
    for name in sorted(KNOBS):
        knob = KNOBS[name]
        if not knob.uniform:
            continue
        raw = os.environ.get(name)
        values[name] = None if raw is None or (raw == "" and knob.kind != "str") else raw
    return values


def _digest(values: dict[str, str | None]) -> str:
    # blake2b, not hash(): str hashing is salted per process, so ranks would never agree.
    return hashlib.blake2b(repr(sorted(values.items())).encode(), digest_size=8).hexdigest()


def assert_uniform(comm) -> None:
    """Raise :class:`NonUniformKnobError` on **every** rank if a uniform knob differs between ranks.

    Collective on ``comm`` (one 8-byte allgather when the ranks agree). Knobs are re-read from
    each rank's own environment on every :meth:`Knob.get`, and many of them decide which
    collectives a rank runs, so a value that differs between ranks is a deadlock at some later
    point. Call this at the top of every public entry, on the communicator that entry's
    collectives run on. Not cached: harnesses change the environment between calls.

    ``comm`` is anything with ``size`` and ``allgather`` (this module imports no mpi4py);
    ``None`` and one-rank communicators return immediately.
    """
    if comm is None or comm.size == 1:
        return
    values = _raw_knob_values()
    digests = comm.allgather(_digest(values))
    if all(d == digests[0] for d in digests):
        return
    # Every rank saw the same digests, so every rank takes this branch together.
    everyone = comm.allgather(values)
    lines = []
    for name in sorted(values):
        by_value: dict[str | None, list[int]] = {}
        for rank, vals in enumerate(everyone):
            by_value.setdefault(vals[name], []).append(rank)
        if len(by_value) > 1:
            parts = []
            for value, ranks in by_value.items():
                shown = ",".join(map(str, ranks[:8])) + (",..." if len(ranks) > 8 else "")
                parts.append(f"{'<unset>' if value is None else repr(value)} on rank(s) {shown}")
            lines.append(f"  {name}: " + "; ".join(parts))
    raise NonUniformKnobError(
        "environment knobs differ between MPI ranks (each rank re-reads them, and they steer collectives):\n"
        + "\n".join(lines)
        + "\nSet them identically on every rank."
    )
