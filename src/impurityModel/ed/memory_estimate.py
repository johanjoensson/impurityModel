"""
Memory sizing helpers for choosing ``truncation_threshold``.

``truncation_threshold`` caps the *global* number of Slater determinants in a
:class:`~impurityModel.ed.manybody_basis.Basis`. This module turns that count into
predicted per-rank peak memory (and back), so drivers can pick a threshold that fits
in RAM instead of guessing.

The byte formulas mirror the authoritative ``memory_bytes()`` estimators in
``src/cython/ManyBodyUtils.pyx`` (``ManyBodyState``, ``ManyBodyState`` and
``SparseKrylovDense``); the pure-Python overhead constants are rough and are
calibrated against measured RSS in ``doc/plans/truncation_reliability.md``.

Layering: this module sits at the bottom of the stack (numpy/mpi4py/os only) so both
the basis layer and the solver drivers may import it.

Two per-rank scaling regimes matter (see ``doc/architecture_overview.md``):

* the sparse (hash-distributed) kernels hold ``~n_dets / ranks`` determinants per
  rank, so per-rank memory shrinks with the communicator size;
* the array block-Lanczos kernel's matvec (``BlockLanczosArray.pyx`` / ``_block_ops.pxi``'s
  ``block_apply``) used to replicate the full ``(global_N, block_width)`` product on every
  rank; since the row-chunked reduce-scatter fix (``doc/plans/dc_smo_performance.md``) it
  is bounded by ``max(counts, over ranks) * block_width`` instead, so this term now shrinks
  with rank count like the others. ``block_width`` itself is bounded at every call site via
  :func:`resolve_gs_block_width` when ``GS_MAX_BLOCK_WIDTH`` (Phase 4) is set; unset, it falls
  back to a placeholder (see :func:`estimate_gs_peak_bytes`'s docstring) and, at production
  widths, the retained dense Krylov term dominates instead.

Under ``run_units_distributed`` the communicator is split into colors and every unit
basis inherits the same numeric ``truncation_threshold``, so each rank's share of a
unit basis is ``threshold / (ranks / n_colors)`` — parallel units multiply per-rank
memory accordingly. The split site enforces this: :func:`max_colors_within_budget`
caps the color count so a cap-filling unit basis still fits the per-rank budget
(``n_parallel_units`` remains available for sizing by hand).

The available-memory probe respects the enforced cgroup memory limit (SLURM ``--mem``
and shared-node allocations), taking the minimum of ``MemAvailable`` and the cgroup
headroom. Unmodeled at-scale overheads (MPI library buffers, transient redistribution
buffers) are absorbed by ``DEFAULT_MEMORY_SAFETY``.
"""

import os
from itertools import pairwise
from math import ceil, exp, log, log2

from mpi4py import MPI

from impurityModel.ed import config

# sizeof(pair<SlaterDeterminant, complex<double>>) in the flat_map entry array:
# a std::vector<uint64_t> header (24 B) + complex<double> (16 B).
_FLAT_MAP_ENTRY_BYTES = 40
# sizeof(std::vector<uint64_t>): the in-line key header stored per ManyBodyState row.
_SD_STRUCT_BYTES = 24
# SparseKrylovDense support-map node overhead per registered row (ManyBodyUtils.pyx).
_KRYLOV_NODE_BYTES = 72
_COMPLEX_BYTES = 16

#: Neighbours per rank the ``graph`` matvec exchange is sized for when the real coupling graph
#: is not in hand (the estimator runs before any Hamiltonian exists). A *measured* input, like
#: ``nnz_per_state``: the maximum source count over ranks on the SrMnO3 double-counting
#: workload at 256 ranks (mean ~28, ``doc/plans/dc_smo_memory.md`` round 5). ``routing_hash``
#: is linear in the occupied orbitals, so the degree is set by the operator's term set, not the
#: basis size, and saturates at ``ranks - 1`` on small communicators. The ``matvec_exchange``
#: trace note reports the real degree of every run; replace this when a workload measures larger.
_MATVEC_EXCHANGE_DEGREE_DEFAULT = 37
# scipy CSC complex128: 16 B value + index/indptr (int32 or int64) per stored element.
_CSR_BYTES_PER_NNZ = 24
# Default candidate fan-out per basis determinant during a CIPSI selection round
# (`CIPSISolver.determine_new_Dj`'s `H|psi_ref>` block) -- a *different*, larger quantity than
# `nnz_per_state` above: this is the raw H-connectivity a selection round explores, before the
# pruning that produces the smaller, *stored* nnz. Measured directly on the SrMnO3 workload this
# term exists for (`doc/plans/dc_smo_memory.md`): `build_local_operator_list` gives a mean raw
# fan-out of 41.1 rows/determinant (max 47); the CIPSI cycles' own `H·psi_ref` row counts ranged
# 16-35 rows/determinant early in the expansion (partial-overlap effects shrink it below the raw
# figure as the basis grows). 40 sits at the conservative (raw-connectivity) end on purpose --
# this term exists because the model was 15-23x *too optimistic* once, and the fix must not
# repeat that in the opposite corner case.
_SELECTION_FANOUT_DEFAULT = 40
# WHOLE-STEP dedup'd ROW fanout of one GF block-Lanczos matvec (`h_op.apply_block(q_curr, ...)`
# inside `_lanczos_step.pxi`'s `block_lanczos_step_cy`) before `redistribute_block` routes rows to
# their owners and the truncation cap prunes. `ManyBodyOperator::apply` returns a fully
# materialized `ManyBodyBlockState` BY VALUE, keyed one row per REACHED determinant -- a
# different, SMALLER quantity than `_SELECTION_FANOUT_DEFAULT` above, which counts
# source-determinant/candidate PAIRS before CIPSI's own dedup. Multiple source rows in `q_curr`
# reaching the same target determinant collapse into one row here; they do not in a pair count.
#
# "WHOLE-STEP" is load-bearing and is what the first shipped version of this constant got wrong.
# Fanout is measured per determinant of the block it is applied to, and it FALLS as that block
# grows, because a larger block's reachable set overlaps itself more. Measuring it on quarter-
# blocks and then applying the result to a whole step over-counts by exactly that dedup. The
# first version shipped 20 from a four-chunk probe; the whole-block value on the same basis is
# 13.30. Measure the block you are modelling.
#
# Measured on the real SrMnO3 crash archive (NOT the WORKLOADS["smo"] key -- a different archive,
# see arrhenius-smo-crash-archive-is-not-the-workloads-key) via `h_op.apply_block(q, 0)` on a
# width-1 block spanning the whole of its own converged ground-state basis
# (`doc/plans/dc_smo_memory.md`, round 8):
#
#     basis size      5,000    20,000   100,000
#     whole fanout    13.30    10.68      8.06
#
# monotone decreasing, log-log slope -0.167. Production caps are >= 100x the largest basis
# measured, so extrapolating would put this near 3.7 -- but this module does not extrapolate past
# its measured range (see `_routing_skew_factor`, which clamps for the same reason), and clamping
# at the largest measured anchor OVER-predicts relative to the trend, which is the safe direction.
# 8.1 is that anchor, rounded up.
#
# A block spanning the entire basis at unit amplitude is deliberate: it is the largest support the
# recurrence can reach, and this is a PEAK model. Two residual assumptions, neither measured: the
# production `q_curr` lives on the *excited* basis rather than the ground-state basis (same H and
# restrictions, similar occupation character, but not identical), and it is a Krylov superposition
# whose support early in a recurrence is smaller than the full basis (which makes this an upper
# bound there, not a lower one).
_GF_MATVEC_ROW_FANOUT_DEFAULT = 8.1

#: Measured saving from the row-chunked apply (`config.GF_APPLY_ROW_CHUNKS`), as
#: ``whole_step_rows / max_over_chunks(chunk_rows)``. The chunked branch of
#: `block_lanczos_step_cy` applies H to one `row_slice` at a time and frees each raw result before
#: the next (`del _raw`), so the PEAK unpartitioned transient over a step is the largest single
#: chunk's, not the whole step's -- chunking genuinely bounds the term this module models.
#:
#: It does NOT bound it by the chunk count, because chunks reach overlapping sets: measured on the
#: same archive and bases as the fanout above, at the 100,000-determinant basis (the largest, and
#: the lowest divisors of the three), ``whole/max`` was 1.35 / 1.96 / 3.49 at 2 / 4 / 8 chunks
#: against the ideal 2 / 4 / 8. Rounded down here, both because the divisors themselves decline
#: mildly with basis size (2.41 -> 2.15 -> 1.96 at 4 chunks over 5k -> 20k -> 100k, i.e. production
#: is likely lower still) and because under-crediting the saving over-predicts memory, which is the
#: safe direction for this term.
#:
#: Two earlier positions on this were both wrong and both unmeasured: that chunking saves nothing
#: here (it saves ~2x at the default), and that it saves the full chunk count (it does not, by
#: roughly half). The knob is a supported escape hatch -- `GF_APPLY_ROW_CHUNKS=1` recovers the
#: one-shot path bit-for-bit -- so the model reads it rather than assuming the default.
_GF_CHUNK_DIVISOR_ANCHORS = ((1, 1.0), (2, 1.3), (4, 1.9), (8, 3.4))
# Per (determinant, reference-column) pair, the CIPSI selection round's own temporaries
# (`CIPSISolver._apply_block_and_redistribute`, `_candidate_overlaps_and_energies`,
# `_score_candidates`) hold at once, summed from the arrays that survive Phase 2's rewrite
# (`doc/plans/dc_smo_memory.md`): the redistributed H|psi_ref> block and the derived coupling
# matrix `overlaps` (complex128, 16 B each -- roughly equal-sized and briefly coexistent, so
# counted as 2x16), the Epstein-Nesbet denominator `de` (float64, 8 B) and numerator `de2`
# (float64, 8 B -- was complex128 before the dtype fix Phase 2 also made) and the `>1e-12` mask
# (bool, 1 B). `_score_candidates` chunks this last group over `GS_SELECTION_CHUNK`, but this
# constant does not assume the knob is set (its default, unchunked, is what every call site gets
# unless someone opts in) -- see `estimate_gs_peak_bytes`'s `selection_bytes` term.
#
# 50 is derived from these array sizes, not fitted to a VmHWM measurement: the pre-Phase-2
# round trip this replaced measured 80-170 B/pair end to end (see the growth-cycle fit in
# `doc/plans/dc_smo_memory.md`), and summing what Phase 2 actually removed --
# `ManyBodyState.to_states`/`from_states`'s ~72 B/pair key-copy round trip, `de2_abs`'s 8 B/pair,
# the group-stack's up to 8 B/pair, and the 8 B/pair the complex-to-float `de2` fix recovers --
# against that range lands close to 50, not further out at the sweep's own uncertainty. A clean
# controlled sweep (isolating this term the way `doc/plans/truncation_reliability.md`'s VmHWM
# sweep isolated `_PY_BASIS_OVERHEAD_BYTES`) should replace this once cluster time allows it.
_SELECTION_BYTES_PER_PAIR = 50
# Python-side Basis bookkeeping per local determinant: SlaterDeterminant wrapper object,
# local_basis list slot and _index_dict entry (over and above bytes_per_determinant, which
# is the flat_map entry + key heap). Calibrated by the VmHWM sweep in
# ``doc/plans/truncation_reliability.md``: a synthetic Basis of N distinct determinants at
# nso=124 has resident RSS slope 273 B/det and VmHWM slope 333 B/det above a stable 213 MiB
# floor (N in {1,2,4}e5, R^2 ~ 1). With bytes_per_determinant(124)=72 that leaves
# 333-72 = 261 B/det of Python/allocator overhead; 260 matches the peak (VmHWM), which is
# what OOM-kills. This is the same Basis object on the GS and GF paths, so both estimators
# use it. (A prior recalibration to 1100 was wrong: it came from raw delta-RSS figures that
# were floor-contaminated, not clean per-determinant slopes.)
_PY_BASIS_OVERHEAD_BYTES = 260

#: Fallback cap used by drivers when no memory probe is possible (matches the historical
#: ``groundstate.calc_gs`` default).
DEFAULT_TRUNCATION_THRESHOLD = 1_000_000

#: Fraction of the available per-rank RAM the sizing helpers budget by default; the rest
#: absorbs transient matvec fanout, allocator slack, unmodeled overheads (MPI buffers), and
#: hash-partition skew: ``estimate_gs_peak_bytes``'s ``replicated_bytes`` term approximates
#: the Phase 1 chunked matvec transient's true bound, ``max(counts)`` over *all* ranks, with
#: *this* rank's own ``local`` share (``doc/plans/dc_smo_performance.md``) -- exact only for a
#: perfectly balanced hash partition. A rank computing its own estimate cannot see another
#: rank's share, so no closed-form correction is possible without measuring the real partition
#: at plan time; this margin is what stands in for it.
DEFAULT_MEMORY_SAFETY = 0.5

# cgroup v1 reports "no limit" as a huge number (PAGE_COUNTER_MAX); anything this large
# is unlimited in practice.
_CGROUP_UNLIMITED = 1 << 60

_ranks_per_node_cache: dict = {}


def _retains_krylov(reort):
    """Whether a reort mode (string, ``Reort`` enum member or None) retains the Krylov store."""
    name = getattr(reort, "name", reort)
    return name is not None and str(name).lower() != "none"


def _krylov_itemsize(reort, krylov_dtype):
    """Bytes per stored Krylov coefficient for a (reort mode, dtype) pair.

    ``complex64`` halves the store but only ``FULL``/``PERIODIC`` may use it: the
    Paige-Simon estimator behind ``PARTIAL``/``SELECTIVE`` steers to
    ``REORT_TOL = sqrt(EPS)``, which a basis stored to ~6e-8 cannot support (the kernel
    raises on that combination). Mirroring that rule here keeps the predicted peak from
    promising a cap the solver will refuse to run.
    """
    if krylov_dtype is None:
        return _COMPLEX_BYTES
    import numpy as _np

    if _np.dtype(krylov_dtype) != _np.dtype(_np.complex64):
        return _COMPLEX_BYTES
    name = str(getattr(reort, "name", reort)).lower()
    if name in ("partial", "selective"):
        raise ValueError(f"krylov_dtype='complex64' is incompatible with reort='{name}'")
    return _COMPLEX_BYTES // 2


def _key_heap_bytes(n_spin_orbitals):
    """Heap bytes of one determinant key allocation (16-byte glibc classes, min 32 B)."""
    n_chunks = max(1, ceil(n_spin_orbitals / 64))
    key_heap = (8 * n_chunks + 8 + 15) & ~15
    return max(key_heap, 32)


def bytes_per_determinant(n_spin_orbitals):
    """Heap bytes per (determinant, coefficient) entry in a flat_map ``ManyBodyState``.

    Mirrors ``ManyBodyState.memory_bytes``: the contiguous entry array element plus one
    heap block per key vector. 72 B for up to 192 spin-orbitals.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals (determinant bit width).

    Returns
    -------
    int
        Bytes per stored determinant.
    """
    return _FLAT_MAP_ENTRY_BYTES + _key_heap_bytes(n_spin_orbitals)


# `SlaterDeterminant::routing_hash` is deliberately locality-preserving, not dispersing (see
# its own comment): a fixed-electron-number charge sector lives on a restricted popcount
# lattice, so `hash % ranks` is not uniform and the skew GROWS with rank count. Measured
# directly (not fitted) on 20,000 real determinants from the SrMnO3 workload against a
# uniform-null Monte-Carlo baseline, cross-validated against two independent GS solves'
# reported `local[min,max]` (doc/plans/dc_smo_memory.md, "routing_hash is deliberately
# non-uniform"): the busiest rank's share over the mean share, at rank count:
_ROUTING_SKEW_ANCHORS = (
    (2, 1.002),
    (4, 1.077),
    (8, 1.092),
    (16, 1.378),
    (64, 1.808),
    (128, 1.901),
    (256, 2.790),
)


def _routing_skew_factor(ranks):
    """Measured max/mean local-determinant-count ratio at ``ranks`` (:data:`_ROUTING_SKEW_ANCHORS`).

    Log-log interpolated between the measured anchors (the skew is a measurement at each
    rank count, not a fitted curve -- ``doc/plans/dc_smo_memory.md`` explicitly found no
    power law that survives extrapolation for the *related* local-count-response exponent,
    so this deliberately does not extrapolate either): outside ``[2, 256]`` it clamps to the
    nearest anchor rather than guess beyond the measured range. ``ranks <= 1`` is not a
    partition at all, so the ratio is exactly 1 by definition.
    """
    ranks = int(ranks)
    if ranks <= 1:
        return 1.0
    lo_r, lo_s = _ROUTING_SKEW_ANCHORS[0]
    hi_r, hi_s = _ROUTING_SKEW_ANCHORS[-1]
    if ranks <= lo_r:
        return lo_s
    if ranks >= hi_r:
        return hi_s
    x = log2(ranks)
    for (r0, s0), (r1, s1) in pairwise(_ROUTING_SKEW_ANCHORS):
        x0, x1 = log2(r0), log2(r1)
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return exp(log(s0) + t * (log(s1) - log(s0)))
    return hi_s  # unreachable given the clamps above


def _gf_chunk_divisor(n_chunks):
    """Measured bound the row-chunked apply puts on the matvec fanout transient at
    ``n_chunks`` chunks (:data:`_GF_CHUNK_DIVISOR_ANCHORS`).

    Same anchor-and-clamp shape as :func:`_routing_skew_factor`, and for the same reason: these
    are measurements at each chunk count, not a fitted curve, so outside the measured ``[1, 8]``
    this clamps to the nearest anchor rather than guess. ``None`` or ``<= 1`` means the one-shot
    path (``GF_APPLY_ROW_CHUNKS=1``), which bounds nothing, so the divisor is exactly 1.
    """
    if n_chunks is None:
        return 1.0
    n_chunks = int(n_chunks)
    if n_chunks <= 1:
        return 1.0
    hi_c, hi_d = _GF_CHUNK_DIVISOR_ANCHORS[-1]
    if n_chunks >= hi_c:
        return hi_d
    for (c0, d0), (c1, d1) in pairwise(_GF_CHUNK_DIVISOR_ANCHORS):
        if c0 <= n_chunks <= c1:
            t = (log2(n_chunks) - log2(c0)) / (log2(c1) - log2(c0))
            return exp(log(d0) + t * (log(d1) - log(d0)))
    return hi_d  # unreachable given the clamps above


def estimate_gf_peak_bytes(
    n_dets,
    n_spin_orbitals,
    block_width,
    reort="none",
    ranks=1,
    n_blocks=None,
    krylov_dtype=None,
    method="lanczos",
    gmres_restart=None,
):
    """Predicted per-rank peak bytes of the sparse (MBS-kernel) Green's-function path.

    Peak ``~ C * (s_live + itemsize * p * n_blocks)`` per rank, where ``C`` is the local
    determinant count. **Both** reort modes pay ``s_live`` — the excited ``Basis`` bookkeeping
    (measured ~330 B/det VmHWM, see :data:`_PY_BASIS_OVERHEAD_BYTES`) plus the ~3 live
    ``ManyBodyState`` blocks of the recurrence (``q_prev``, ``q_curr``, ``wp``, ~216 B/det
    at block width 1). So ``s_live ~ 450-550 B/det`` and ``reort="none"`` is not free, but it
    is *bounded* — not the multi-kB/det figure an earlier miscalibration claimed. At
    ``reort != "none"`` the ``SparseKrylovDense`` store adds ``itemsize * p * n_blocks`` bytes
    per retained determinant *on top* (rows bounded by the retained set, columns by the Lanczos
    block count) — it cannot be compressed away, see ``doc/plans/blocklanczos_reort_memory.md``.

    Which term leads depends on the run: the store scales with ``n_blocks`` (``m``), so at
    width 1 it overtakes ``s_live`` once ``m`` passes ~30, but at ``reort="none"`` (the
    production self-energy path) ``s_live`` is the whole cost -- **plus** the recurrence's
    transient matvec fanout (below), which at ``reort="none"`` is comparable to it. Neither
    ``s_live`` nor the store universally dominates the *other* -- the earlier "the store
    dominates against ~450 for everything else" framing was only half right (the ~450 is real;
    the store does not always win).

    The fanout term (:data:`_GF_MATVEC_ROW_FANOUT_DEFAULT`, divided by
    :data:`_GF_CHUNK_DIVISOR_ANCHORS`) closes a gap this docstring previously conceded: the
    matvec's raw output is materialized whole, unpartitioned, before routing and pruning. It was
    added in round 8 (``doc/plans/dc_smo_memory.md``) on the strength of being a real, code-
    verified allocation -- **not**, as a retracted first draft of that round claimed, because it
    was needed to explain an OOM. It was not: round 7's resident-adjusted budget already refuses
    that crash's configuration without it. Sizing it wrongly has a real cost in the other
    direction -- an over-priced version of this term cut the affordable color count 25 -> 5 on
    the same geometry, i.e. most of the GF phase's concurrency -- so it is sized on measurement,
    at the largest basis measured, with the chunking credit the chunked apply actually earns.

    Since the reort projection now streams the store chunk by chunk, the old
    ``(n_rows x n_cols)`` gather transient (which peaked at ~1.85x the store) is gone and
    is no longer modelled.

    Parameters
    ----------
    n_dets : int
        Global determinant count (the ``truncation_threshold`` being considered).
    n_spin_orbitals : int
        Determinant bit width.
    block_width : int
        Block width ``p`` of the GF block Lanczos (number of seed vectors of the unit).
    reort : str
        Reorthogonalization mode; anything but ``"none"`` retains the Krylov store.
    ranks : int
        MPI ranks sharing this basis (the unit's sub-communicator size under
        ``run_units_distributed``). Peak bytes scale with the *binding* rank, not the
        mean: ``local_rows`` is the mean local determinant count times
        :func:`_routing_skew_factor`, ``routing_hash``'s measured max/mean skew at this
        rank count -- the OOM killer fires on the heaviest rank's share of a hash-
        distributed basis, not the average one. This is why a color split into few ranks
        (a small unit under ``run_units_distributed``) is not simply "mean x safety
        margin": the skew is small at few ranks (1.08x at 4) and large at many (2.79x at
        256), so it must be evaluated at the unit's own rank count, not the job's.
        :func:`estimate_gs_peak_bytes` deliberately does **not** apply this factor: the
        ground-state solve spans the full communicator, its measured skew there (1.62x
        on the SrMnO3 archive's own final cycle) sits inside ``DEFAULT_MEMORY_SAFETY``,
        and the cap it produces is what the whole double-counting search is calibrated
        against -- moving it would change ``dc_search.resolve_cap_at_max``'s behaviour on
        every DC iteration, a much larger change than this function's docstring should
        decide by itself.
    n_blocks : int, optional
        Krylov blocks retained at ``reort != "none"``. Defaults to the invariant-subspace
        bound ``ceil(n_dets / block_width)`` (worst case).
    krylov_dtype : optional
        Storage dtype of the Krylov basis. ``complex64`` halves the store and is legal
        only for ``FULL``/``PERIODIC`` (see :func:`_krylov_itemsize`).
    method : str
        ``"lanczos"`` (default) or ``"bicgstab"`` -- the per-frequency BiCGSTAB driver
        retains **no** Krylov store (``reort``/``n_blocks``/``krylov_dtype`` are ignored)
        but carries more live blocks per solve: the 7 solver blocks (``xi, ri, r0_t, pi,
        vi, si, ti``) plus the seeds, the 3 warm-start history solutions and the
        extrapolated guess -- ~12 block-rows against the recurrence's 3 -- plus, on the
        points BiCGSTAB leaves unconverged, the GMRES fallback's transient Arnoldi space
        of ``gmres_restart + 3`` block-rows. That transient is what a worst-case point
        peaks at, and peaks are what OOM-kill, so it is modeled rather than footnoted.
        The basis term is the *per-point* rebuilt support, still bounded by the same
        ``n_dets`` cap.
    gmres_restart : int, optional
        The fallback's block-Arnoldi restart length; only read for ``method="bicgstab"``.
        ``None`` (default) reads :data:`config.GF_GMRES_RESTART` -- the same knob
        :func:`gf_solvers.solve_shifted_block` resolves at solve time, so a caller who
        overrides it cannot silently get a peak estimate for the un-overridden length.

    Returns
    -------
    int
        Predicted per-rank peak bytes.
    """
    local_rows = ceil(n_dets / max(1, ranks) * _routing_skew_factor(ranks))
    key_heap = _key_heap_bytes(n_spin_orbitals)
    basis_bytes = local_rows * (bytes_per_determinant(n_spin_orbitals) + _PY_BASIS_OVERHEAD_BYTES)
    row_bytes = _COMPLEX_BYTES * block_width + key_heap + _SD_STRUCT_BYTES
    # "cipsi" shares the bicgstab live-vector model: same per-point solver, and the
    # selection loop's basis is bounded by the same cap (GF_CIPSI_BUDGET defaults to it).
    if method in ("bicgstab", "sliced", "cipsi"):
        if gmres_restart is None:
            gmres_restart = config.GF_GMRES_RESTART.get()
        live = 12 + gmres_restart + 3
        if method == "sliced":
            # The filter stage's transient (3 recurrence blocks + one accumulator per
            # window) runs before the solves; the peak is whichever transient is larger.
            # At most 2 rest windows complete the partition -- one collapses whenever the
            # evaluation band reaches a spectral bound, so this is an upper bound, which is
            # what a peak model wants.
            n_windows = config.GF_SLICES.get() + 2
            live = max(live, 3 + n_windows)
        return basis_bytes + live * local_rows * row_bytes
    live_bytes = 3 * local_rows * row_bytes
    # The recurrence's transient matvec fanout: `wp_raw = h_op.apply_block(q_curr, ...)` inside
    # `block_lanczos_step_cy` is NOT hash-partitioned -- it is everything this rank's rows reach
    # under H, before `redistribute_block` routes rows to their eventual owners and the cap
    # prunes admissions (see `_GF_MATVEC_ROW_FANOUT_DEFAULT`'s derivation). Priced at the same
    # `row_bytes` per row as the other live blocks, and divided by the measured saving the
    # row-chunked apply actually delivers at the configured chunk count
    # (`_GF_CHUNK_DIVISOR_ANCHORS` -- ~1.9x at the default of 4, NOT the full chunk count, since
    # chunks reach overlapping sets). Reading the knob matters: `GF_APPLY_ROW_CHUNKS=1` is the
    # supported escape hatch for a bit-identical run, and on that path the divisor is 1.
    #
    # Chunking always applies wherever this estimate is used: `_lanczos_step.pxi` chunks whenever
    # the step has a redistribute to bound, which is true for any basis under a finite
    # `truncation_threshold` (a `_CappedBasisProxy` sets `caps_growth`, serial or not), and a
    # finite cap is exactly the case this function is called to size.
    #
    # Scoped to the Lanczos recurrence only (not the bicgstab/sliced/cipsi branch above, which
    # returned already): that branch runs a different per-point solver whose own live-block model
    # is separate, and its transient has not been measured.
    fanout_bytes = (
        ceil(local_rows * _GF_MATVEC_ROW_FANOUT_DEFAULT / _gf_chunk_divisor(config.GF_APPLY_ROW_CHUNKS.get()))
        * row_bytes
    )
    store_bytes = 0
    if _retains_krylov(reort):
        if n_blocks is None:
            n_blocks = ceil(n_dets / max(1, block_width))
        itemsize = _krylov_itemsize(reort, krylov_dtype)
        store_bytes = local_rows * (itemsize * block_width * n_blocks + 2 * key_heap + _KRYLOV_NODE_BYTES)
    return basis_bytes + live_bytes + fanout_bytes + store_bytes


#: Additive eigenstate padding TRLM's initial sizing applies before certifying a manifold
#: complete (mirrors ``cipsi_solver._EIGENSTATE_PAD``; not imported -- this module sits below
#: the solver stack, see the module docstring -- so keep the two in sync by hand if that
#: constant changes; both are 10 as of this writing).
_GS_EIGENSTATE_PAD = 10

#: The pre-Phase-4 coupled-regime assumption ``num_wanted ~= _GS_COUPLED_NUM_WANTED_RATIO *
#: block_width`` (``cipsi_solver.expand``'s ``num_wanted = 2 * len(psi_refs)`` against
#: ``block_width = len(psi_refs) + 1``, true only while ``GS_MAX_BLOCK_WIDTH`` is unset). A
#: single source of truth for both :func:`_gs_krylov_columns`'s default and
#: :func:`log_memory_budget`'s warning message, so the two cannot silently drift apart.
_GS_COUPLED_NUM_WANTED_RATIO = 2


def _gs_krylov_columns(n_dets, block_width, num_wanted=None):
    """Peak column count of the retained dense Krylov store, at the ground-state default
    ``reort="full"``.

    Mirrors ``cipsi_solver._size_subspace`` **exactly** (re-read from source, not from memory --
    an earlier version of this function got the formula wrong by omitting a factor of 2 and a
    flat +20 headroom the real one carries, caught by review; see the git history of this file
    and ``doc/plans/dc_smo_performance.md``'s "Phase resequencing" section):

    .. code-block:: text

        max_subspace = min(max(2*nw, nw+10), n_dets)
        blocks = min(2*ceil(max_subspace/p) + 20, max(2, n_dets//p - 1))

    where ``nw = num_wanted + _GS_EIGENSTATE_PAD``. Not imported (this module sits below the
    solver stack, see the module docstring); keep the two in sync by hand if
    ``_size_subspace`` changes -- there is no automated guard against drift here, only the
    cross-reference.

    **This replaced a flat ``n_blocks`` constant**, itself replacing a still-earlier guess of
    30: neither a constant ratio nor (as this function's first version assumed) a ratio-free
    formula covers the real behaviour, because ``num_wanted`` and ``block_width`` are two
    different quantities that happen to move together 1:1 only when ``GS_MAX_BLOCK_WIDTH`` is
    unset. Now that it exists (Phase 4), ``num_wanted`` can stay large while ``block_width`` is
    capped -- callers pass the real, resolved ``block_width`` (:func:`resolve_gs_block_width`)
    to keep this honest, and ``num_wanted``'s ``None`` default here still assumes the pre-cap
    1:1 relationship for any caller that has not measured its own ``num_wanted``.

    Parameters
    ----------
    n_dets : int
        Global determinant count -- ``_size_subspace``'s ``cap`` argument (``len(self.basis)``
        in production; the same basis-size bound applies here since the Krylov store cannot
        hold more orthonormal columns than there are determinants).
    block_width : int
        The Lanczos block width ``p``.
    num_wanted : int, optional
        The *unpadded* eigenstate request (``get_eigenvectors``'s own argument). ``None``
        (default) assumes ``2 * block_width`` -- the relationship that holds while
        ``num_wanted`` tracks the block width 1:1, i.e. before ``GS_MAX_BLOCK_WIDTH`` exists.

    Returns
    -------
    int
        Column count (a multiple of ``block_width``).
    """
    p = max(1, block_width)
    nw = (_GS_COUPLED_NUM_WANTED_RATIO * p if num_wanted is None else num_wanted) + _GS_EIGENSTATE_PAD
    max_subspace = min(max(2 * nw, nw + 10), n_dets)
    blocks = min(2 * ceil(max_subspace / p) + 20, max(2, n_dets // p - 1))
    return p * blocks


def resolve_gs_block_width(default=4):
    """The ground-state Lanczos block width to size a memory estimate with.

    Returns the configured :data:`config.GS_MAX_BLOCK_WIDTH` cap when set (Phase 4,
    ``doc/plans/dc_smo_performance.md``); otherwise ``default``, a placeholder -- the real,
    uncapped width grows with the warm-start manifold (``cipsi_solver.get_eigenvectors``) and
    has no static bound this function can report honestly. :func:`log_memory_budget` prints a
    warning on that unset path rather than letting the fallback look like a measured number.

    Parameters
    ----------
    default : int
        Value to return when the knob is unset (the historical ``block_width=4``, or a
        driver's own Green's-function block width when it wants ``max(gf, gs)`` sizing).

    Returns
    -------
    int
        Block width to feed :func:`estimate_gs_peak_bytes` / :func:`suggest_truncation_threshold`.
    """
    configured = config.GS_MAX_BLOCK_WIDTH.get()
    return default if configured is None else configured


def resolve_gs_num_wanted():
    """Eigenstate count to size a ground-state memory estimate with, or ``None`` when unset.

    The twin of :func:`resolve_gs_block_width`, and deliberately shaped the same way: it reports
    the configured :data:`config.GS_NUM_WANTED` and otherwise ``None``, letting
    :func:`estimate_gs_peak_bytes` fall back to its ``2 * block_width`` assumption and
    :func:`log_memory_budget` warn that the number is a guess.

    No default is invented here. The kept manifold grows with basis size and saturates at a
    workload-specific value (measured 44 / 87 / ~105 at 20,000 / 100,000 / 949,834 determinants on
    SrMnO3), so any constant would silently mis-size every cap it touched; an honest warning beats
    a confident wrong number. See ``doc/plans/dc_smo_memory.md``.

    Returns
    -------
    int or None
        ``num_wanted`` for :func:`estimate_gs_peak_bytes` / :func:`suggest_truncation_threshold`.
    """
    return config.GS_NUM_WANTED.get()


def resolve_sizing_block_width(gf_block_width):
    """Block width to size a call site that estimates both a GF and a GS solve with one shared
    ``block_width`` parameter (``selfenergy.py``/``susceptibility.py``): the larger of the
    driver's own GF block width and the resolved GS width (:func:`resolve_gs_block_width`,
    falling back to ``gf_block_width`` itself when ``GS_MAX_BLOCK_WIDTH`` is unset -- so this
    equals ``gf_block_width`` exactly, unchanged, on that path).
    """
    return max(gf_block_width, resolve_gs_block_width(gf_block_width))


def estimate_gs_peak_bytes(
    n_dets,
    n_spin_orbitals,
    block_width=4,
    ranks=1,
    nnz_per_state=100,
    num_wanted=None,
    selection_fanout=_SELECTION_FANOUT_DEFAULT,
):
    """Predicted per-rank peak bytes of the ground-state (CIPSI + array-kernel) path.

    Counts the ``Basis`` bookkeeping and the CSR Hamiltonian snapshot (both hash
    distributed, ~1/ranks per rank), the array kernel's chunked reduce-scatter matvec
    transient (``_block_ops.pxi``'s ``block_apply``, post row-chunking: bounded by
    ``max(counts)`` -- the largest single rank's local row/column count under the hash
    partition, approximated here as ``local`` since a materially skewed partition, not
    ``global_N``, is now the risk this term misses -- see :data:`DEFAULT_MEMORY_SAFETY`,
    which is what absorbs it: this function has no way to see another rank's share), the
    retained dense Krylov blocks at the ground-state default ``reort="full"``, and the
    CIPSI selection round's own transient (``selection_bytes`` below).

    The selection term is what this function was missing until
    ``doc/plans/dc_smo_memory.md``: a production SrMnO3 double-counting search was
    OOM-killed at a basis this function (pre-fix) predicted needed 2.5 GiB/rank, of a
    5 GiB/rank budget -- 15-23x too optimistic across every plausible choice of its other
    inputs, because nothing here charged for ``CIPSISolver.determine_new_Dj``'s own
    per-cycle arrays (``_apply_block_and_redistribute``'s redistributed ``H|psi_ref>``
    block, ``_candidate_overlaps_and_energies``'s coupling matrix, ``_score_candidates``'s
    Epstein-Nesbet temporaries), which scale with the *raw* candidate connectivity, not
    with the basis or the stored Hamiltonian this function already counted.

    Parameters
    ----------
    n_dets : int
        Global determinant count (the ``truncation_threshold`` being considered).
    n_spin_orbitals : int
        Determinant bit width.
    block_width : int
        Lanczos block width (number of sought eigenvectors). Every production call site now
        resolves this via :func:`resolve_gs_block_width`, which reads the configured
        ``GS_MAX_BLOCK_WIDTH`` cap (Phase 4) when set. The default ``4`` here is only reached
        when that knob is unset, in which case it is a placeholder, not a measurement: Phase 0
        found the real width growing with the determinant cap itself (2-16 at cap 2000,
        105-315 at the ~1M production cap that crashed), so it has no honest static bound until
        the knob is set. :func:`log_memory_budget` warns on that path. Also the width
        ``selection_bytes`` sizes: the selection round's own block width is
        ``len(psi_ref)``, which tracks this same quantity (see
        ``cipsi_solver.CIPSISolver.expand``'s warm-start feedback loop).
    ranks : int
        MPI ranks sharing the basis.
    nnz_per_state : int
        Stored Hamiltonian elements per basis state (measure on a small run; grows with
        the number of one-/two-body terms). Not the same quantity as ``selection_fanout``
        below -- this is the *pruned, stored* matrix; measured 6.4 on SrMnO3 against a raw
        connectivity of ~41 (``doc/plans/dc_smo_memory.md``), so conflating the two would
        undercount the selection term by an order of magnitude.
    num_wanted : int, optional
        Forwarded to :func:`_gs_krylov_columns`; ``None`` assumes ``2 * block_width`` (the
        pre-Phase-4 coupled regime -- still the best available default even when
        ``GS_MAX_BLOCK_WIDTH`` caps ``block_width``, since ``num_wanted`` is *not* capped by that
        knob and has no static bound tighter than ``n_dets`` itself; substituting ``n_dets`` here
        turns this term quadratic in ``n_dets`` and makes the estimate useless rather than safe
        (found by review -- do not reintroduce it). The honest fix is a measured value from the
        same width sweep that sets ``GS_MAX_BLOCK_WIDTH`` (Phase 0 recorded the real ratio
        growing to ~30x this default at production scale); pass it here once measured, the same
        way ``nnz_per_state`` is a measured, not derived, input.
    selection_fanout : int
        Candidate determinants a CIPSI selection round connects to, per basis determinant,
        *before* pruning (see :data:`_SELECTION_FANOUT_DEFAULT`). Deliberately not threaded
        through :func:`suggest_truncation_threshold`/:func:`log_memory_budget` yet -- unlike
        ``nnz_per_state`` and ``num_wanted``, no measured value from a real width sweep exists
        for it (that sweep needs the multi-rank run ``doc/plans/dc_smo_memory.md`` Phase 1
        describes); every production call site therefore gets this conservative default,
        which is deliberately biased toward *over*-predicting the peak -- the failure mode
        this term exists to close was the model being too optimistic, not too pessimistic.

    Returns
    -------
    int
        Predicted per-rank peak bytes.
    """
    local = ceil(n_dets / max(1, ranks))
    basis_bytes = local * (bytes_per_determinant(n_spin_orbitals) + _PY_BASIS_OVERHEAD_BYTES)
    csr_bytes = local * nnz_per_state * _CSR_BYTES_PER_NNZ
    # The matvec's reduce-scatter transient, plus the (local, w) result. Under
    # GS_MATVEC_EXCHANGE=graph (the default) the send and receive buffers each hold one row
    # block per neighbour -- degree x local x w complex, both alive at once -- chunked over
    # columns so neither exceeds GS_MATVEC_EXCHANGE_BYTES (mpi_comm.MatvecExchangePlan). Under
    # `reduce` it is the one (max(counts), w) chunk buffer of the per-root Reduce loop
    # (Phase 1); max(counts) ~ local under a balanced hash partition either way.
    if config.GS_MATVEC_EXCHANGE.get() == "graph":
        degree = min(max(ranks - 1, 0), _MATVEC_EXCHANGE_DEGREE_DEFAULT)
        per_buffer = min(config.GS_MATVEC_EXCHANGE_BYTES.get(), degree * local * block_width * _COMPLEX_BYTES)
        exchange_bytes = 2 * per_buffer
    else:
        exchange_bytes = local * block_width * _COMPLEX_BYTES
    replicated_bytes = local * block_width * _COMPLEX_BYTES + exchange_bytes
    krylov_bytes = local * _gs_krylov_columns(n_dets, block_width, num_wanted) * _COMPLEX_BYTES
    # CIPSI selection round transient (doc/plans/dc_smo_memory.md): `local * selection_fanout`
    # approximates this rank's local candidate-row count the same way `local * nnz_per_state`
    # approximates the stored CSR nnz above -- a different, larger quantity (raw connectivity
    # before pruning), times `block_width` reference columns, times the per-pair cost that
    # survives Phase 2's rewrite of `_apply_block_and_redistribute` /
    # `_candidate_overlaps_and_energies` / `_score_candidates`.
    # Calibrated against the one-shot apply; the row-chunked default (GS_APPLY_ROW_CHUNKS=4)
    # holds ~1/2.8 of that on the step that set the constant, so this is now an upper bound
    # (deliberately: the model's failure mode was optimism, see GS_MEMORY_BUDGET_SAFETY).
    selection_bytes = local * selection_fanout * block_width * _SELECTION_BYTES_PER_PAIR
    return basis_bytes + csr_bytes + replicated_bytes + krylov_bytes + selection_bytes


def _read_cgroup_int(path):
    """Integer content of a cgroup file; ``None`` on missing file or non-numeric (``max``)."""
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _cgroup_available_bytes(proc_path="/proc/self/cgroup", v2_root="/sys/fs/cgroup", v1_root="/sys/fs/cgroup/memory"):
    """Tightest memory headroom (limit - current usage) over this process's cgroup ancestors.

    Job schedulers (SLURM) enforce ``--mem`` through cgroup limits, which can sit far
    below the node's ``MemAvailable`` on shared allocations. Handles cgroup v2
    (``memory.max``/``memory.current``) and v1 (``memory.limit_in_bytes``/
    ``memory.usage_in_bytes``); returns ``None`` when no limit applies (unlimited,
    non-Linux, or unreadable hierarchy).
    """
    try:
        with open(proc_path) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    headroom = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy_id, controllers, cgroup_path = parts
        if hierarchy_id == "0" and not controllers:
            base, limit_name, usage_name = v2_root, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            base, limit_name, usage_name = v1_root, "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        stop = os.path.normpath(base)
        node = os.path.normpath(os.path.join(base, cgroup_path.lstrip("/")))
        while node.startswith(stop):
            limit = _read_cgroup_int(os.path.join(node, limit_name))
            if limit is not None and limit < _CGROUP_UNLIMITED:
                usage = _read_cgroup_int(os.path.join(node, usage_name)) or 0
                headroom = min(headroom, max(0, limit - usage)) if headroom is not None else max(0, limit - usage)
            if node == stop:
                break
            node = os.path.dirname(node)
    return headroom


def _node_available_bytes():
    """Available bytes on this node, respecting the enforced cgroup limit.

    The minimum of ``MemAvailable`` from /proc/meminfo (sysconf fallback) and the
    cgroup memory headroom (:func:`_cgroup_available_bytes`) — the latter is what the
    kernel OOM-kills against under a scheduler-constrained allocation.
    """
    available = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if available is None:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    cgroup = _cgroup_available_bytes()
    return available if cgroup is None else min(available, cgroup)


def available_bytes_per_rank(comm=None):
    """Available RAM per MPI rank, consistent across the communicator.

    .. warning:: **Collective on** ``comm``: every rank must call this (it splits a
       shared-memory sub-communicator to count ranks per node and min-reduces the
       result). Never gate the call on rank-local state.

    The node's ``MemAvailable`` is divided by the number of ranks on that node
    (``MPI.COMM_TYPE_SHARED`` split, freed immediately at this synchronized point;
    the ranks-per-node count is cached per communicator). The global minimum is
    returned so all ranks agree on one budget.

    Parameters
    ----------
    comm : MPI communicator, optional
        ``None`` means serial: the node's available bytes are returned directly.

    Returns
    -------
    int
        Available bytes per rank (communicator-wide minimum).
    """
    node_bytes = _node_available_bytes()
    if comm is None or comm.size == 1:
        return node_bytes
    cache_key = comm.py2f()
    ranks_on_node = _ranks_per_node_cache.get(cache_key)
    if ranks_on_node is None:
        shared = comm.Split_type(MPI.COMM_TYPE_SHARED)
        ranks_on_node = shared.size
        shared.Free()
        _ranks_per_node_cache[cache_key] = ranks_on_node
    return comm.allreduce(node_bytes // max(1, ranks_on_node), op=MPI.MIN)


def suggest_truncation_threshold(
    n_spin_orbitals,
    comm=None,
    block_width=4,
    reort="none",
    n_parallel_units=1,
    nnz_per_state=100,
    safety=DEFAULT_MEMORY_SAFETY,
    krylov_dtype=None,
    method="lanczos",
    gs_num_wanted=None,
):
    """Largest ``truncation_threshold`` whose predicted peak fits in per-rank RAM.

    .. warning:: **Collective on** ``comm`` (calls :func:`available_bytes_per_rank`).

    The budget is ``safety * available_bytes_per_rank``; the safety factor absorbs
    transient overshoot (one matvec fanout past the cap), allocator slack (up to ~2x on
    the flat_map entry arrays after growth) and everything this model does not count.
    The threshold is the largest ``n`` for which both :func:`estimate_gs_peak_bytes`
    and :func:`estimate_gf_peak_bytes` stay within budget, found by bisection.

    Parameters
    ----------
    n_spin_orbitals : int
        Determinant bit width.
    comm : MPI communicator, optional
    block_width : int
        Lanczos block width used for both path estimates.
    reort : str
        GF reorthogonalization mode (``"none"`` on the production self-energy path).
    n_parallel_units : int
        Simultaneous ``run_units_distributed`` colors; divides the ranks per unit basis.
    nnz_per_state : int
        Stored Hamiltonian elements per basis state for the ground-state CSR estimate.
    safety : float
        Fraction of available RAM to budget (default ``DEFAULT_MEMORY_SAFETY``).
    krylov_dtype : optional
        Krylov store dtype; ``complex64`` halves the store and so raises the cap.
    gs_num_wanted : int, optional
        Forwarded to :func:`estimate_gs_peak_bytes`'s ``num_wanted``. ``None`` (default) keeps
        the pre-Phase-4 ``num_wanted ~ 2*block_width`` assumption, which under-counts the
        ground-state Krylov term once ``GS_MAX_BLOCK_WIDTH`` caps ``block_width`` below the real
        (uncapped) manifold width -- see :func:`estimate_gs_peak_bytes`'s docstring for why the
        fix is a measured value, not a derived worst case. :func:`log_memory_budget` warns when
        the knob is set and this is not supplied.

    Returns
    -------
    int
        Suggested global determinant cap (at least 1).
    """
    budget = safety * available_bytes_per_rank(comm)
    ranks = comm.size if comm is not None else 1
    return _suggest_for_budget(
        budget,
        n_spin_orbitals,
        block_width,
        reort,
        n_parallel_units,
        nnz_per_state,
        ranks,
        krylov_dtype,
        method,
        gs_num_wanted,
    )


def absolute_rss_budget(safety, available, resident_bytes):
    """The rank's share of node RAM as an **absolute** RSS ceiling: ``safety * (available +
    resident)``.

    ``available_bytes_per_rank`` is ``MemAvailable / ranks_on_node`` -- memory that is *free*,
    already net of what this process holds. ``safety * available`` is therefore an *increment*
    allowance, and comparing it against a process's *absolute* RSS mixes two quantities. Adding
    ``resident`` back before applying ``safety`` recovers the rank's whole share, which is what
    an absolute RSS reading may be measured against.

    That confusion is not hypothetical: it pinned the SrMnO3 cubic gap-DC search at its seed
    basis in every sector after the first, because the process sat at 2.5 GiB resident against a
    ``0.5 * 4.9 = 2.45 GiB`` budget and the guard read negative headroom before doing any work
    (``doc/plans/dc_smo_memory.md``, round 9).

    ``resident_bytes`` of ``None`` or ``0`` degrades to ``safety * available``, the pre-2026-09
    behaviour, so a caller that cannot measure its own RSS is no worse off than before.

    **On the reduction directions**: production callers pass a MIN-reduced ``available``
    (:func:`available_bytes_per_rank`) and a MAX-reduced ``resident``
    (:func:`resident_bytes_per_rank`). That is deliberately the worst case on both terms, and it
    means the sum is *not* any single rank's quantity -- do not reason about it as one.
    """
    resident = 0.0 if resident_bytes is None else max(0.0, float(resident_bytes))
    return safety * (float(available) + resident)


def _resident_adjusted_budget(safety, available, resident_bytes):
    """The head*room* form of :func:`absolute_rss_budget`: how much more this process may add.

    ``safety * available`` by default, tightened to ``absolute_rss_budget(...) - resident`` when
    ``resident_bytes`` is given and that tightening is actually binding (see
    :func:`max_unit_dets_within_budget`'s ``resident_bytes`` parameter for the derivation).

    Never lets an already-over-budget process drive the result to a non-positive headroom --
    that memory is spent either way, and callers use this as a bisection bound, not a signal
    to shrink toward zero. Callers that compare against an *absolute* RSS reading want
    :func:`absolute_rss_budget` instead; the two differ by exactly ``resident``, and picking the
    wrong one is the defect described there.
    """
    budget = safety * available
    if resident_bytes is not None and resident_bytes > 0:
        headroom = absolute_rss_budget(safety, available, resident_bytes) - float(resident_bytes)
        if headroom > 0:
            budget = headroom
    return budget


def resident_bytes_per_rank(comm=None):
    """This process's resident set (:func:`current_rss_bytes`), MAX-reduced over ``comm``.

    .. warning:: **Collective on** ``comm``. Every rank must reach it; never gate the call on
       rank-local state (CLAUDE.md's MPI rules). The lowercase ``allreduce`` is safe here only
       because the payload is a plain Python ``int`` -- an ndarray payload deadlocks
       (see the ``mpi4py`` lowercase-allreduce note).

    MAX rather than mean: the budget has to hold on the heaviest rank, and ``routing_hash`` is
    deliberately locality-preserving, so the busiest rank owns ~2.79x the mean at 256 ranks.
    """
    resident = current_rss_bytes()
    if comm is None or comm.size == 1:
        return resident
    return comm.allreduce(resident, op=MPI.MAX)


def max_colors_within_budget(
    n_dets,
    n_spin_orbitals,
    block_width,
    reort,
    comm,
    max_candidate,
    safety=DEFAULT_MEMORY_SAFETY,
    krylov_dtype=None,
    method="lanczos",
):
    """Largest unit-color count whose predicted per-rank GF peak fits the memory budget.

    .. warning:: **Collective on** ``comm`` (calls :func:`available_bytes_per_rank`).

    Under ``run_units_distributed`` each color's unit basis may fill the same
    ``truncation_threshold`` on only ``comm.size / n_colors`` ranks, so per-rank memory
    grows with the color count. This inverts :func:`estimate_gf_peak_bytes`: the largest
    ``n_colors <= max_candidate`` for which a cap-filling unit basis still fits the budget
    (see :func:`_resident_adjusted_budget` -- the same policy
    :func:`max_unit_dets_within_budget` uses, so the two no longer diverge on whether the
    resident set counts). At ``reort != "none"`` the estimate uses the invariant-subspace
    worst case for the Krylov store (very conservative), consistent with
    :func:`suggest_truncation_threshold`.

    Parameters
    ----------
    n_dets : int
        The basis cap (``truncation_threshold``) each unit basis may fill.
    n_spin_orbitals : int
        Determinant bit width.
    block_width : int
        Widest unit's seed count (GF block width).
    reort : str or None
        GF reorthogonalization mode.
    comm : MPI communicator
        The full communicator about to be split.
    max_candidate : int
        Upper bound on the color count (``min(comm.size, n_units)`` at the split site).
    safety : float
        Fraction of available RAM to budget (default ``DEFAULT_MEMORY_SAFETY``).

    Returns
    -------
    int
        Color count in ``[1, max_candidate]``.

    Notes
    -----
    This deliberately does **not** take a ``resident_bytes`` argument the way
    :func:`max_unit_dets_within_budget` does. Passing the resident set here would tighten the
    *concurrency* bound as well as the per-unit cap, and the two are not interchangeable: the
    per-unit cap trades basis size (accuracy) for safety, while this one trades color count
    (wall clock) for safety, and nothing has measured that the second trade is wanted. It also
    underpins the composition argument in :func:`max_unit_dets_within_budget`'s docstring,
    which assumes the two inversions run against the *same* budget. Both share
    :func:`_resident_adjusted_budget` so the policy has one definition; only this call site
    passes no resident set.
    """
    budget = _resident_adjusted_budget(safety, available_bytes_per_rank(comm), None)
    for n_colors in range(max_candidate, 1, -1):
        # The mean, not each color's real count: `_pack_units` (basis_split.py) apportions
        # ranks to colors proportionally to bin mass with a floor of 1, not evenly, so colors
        # genuinely differ -- a round-8 SrMnO3 archive had colors on 4, 5 *and* 6 ranks at
        # n_colors=25 (mean 5.12). This function runs *before* `_pack_units` (it decides
        # `max_colors`, one of `_pack_units`'s own inputs) and sits below `basis_split` in the
        # layering (CLAUDE.md), so it cannot call the real packer to learn the true spread, and
        # the only bound it *could* guarantee -- 1 rank/color -- would make it always return 1.
        # The real per-color bound lives where it can actually be seen: `run_units_distributed`
        # sizes each color's own cap on `split_basis.comm.size` after the real split runs
        # (doc/plans/dc_smo_memory.md, "GF unit memory", item 3) and never loosens what this
        # function's mean-based `max_colors` allows -- this is deliberately the coarser of the
        # two bounds, not a second, independent one to fix.
        ranks_per_color = max(1, comm.size // n_colors)
        if (
            estimate_gf_peak_bytes(
                n_dets,
                n_spin_orbitals,
                block_width,
                reort,
                ranks=ranks_per_color,
                krylov_dtype=krylov_dtype,
                method=method,
            )
            <= budget
        ):
            return n_colors
    return 1


def max_unit_dets_within_budget(
    n_spin_orbitals,
    block_width,
    reort,
    ranks,
    comm,
    safety=DEFAULT_MEMORY_SAFETY,
    krylov_dtype=None,
    method="lanczos",
    resident_bytes=None,
):
    """Largest per-unit ``truncation_threshold`` whose predicted GF peak fits ``ranks`` ranks.

    .. warning:: **Collective on** ``comm`` (calls :func:`available_bytes_per_rank`).

    The complement of :func:`max_colors_within_budget`: that function fixes the cap and
    finds the largest color count (so the smallest ``ranks``) that still fits the budget;
    this fixes ``ranks`` -- the color a unit actually landed on, from
    its own ``split_basis.comm.size`` -- and finds the largest cap that basis can fill.
    A unit basis inheriting the *job-wide* ``truncation_threshold`` verbatim (today's
    behaviour, ``basis_split.py``'s ``truncation_threshold=basis.truncation_threshold``) was
    sized for all of ``comm.size`` ranks, not the ``ranks`` its color actually has, which is
    the root cause this function exists to fix (see ``doc/plans/dc_smo_memory.md``, "GF
    unit memory").

    Both this function and :func:`max_colors_within_budget` invert the same
    :func:`estimate_gf_peak_bytes` (which is where :func:`_routing_skew_factor` lives), so
    they compose rather than double-count: a skew-tightened color count means *more* ranks
    per color, which this function then reports as a *larger* affordable per-unit cap.

    **That composition is why ``resident_bytes`` exists.** Given the *same* budget and the
    *mean* rank count, this function could not tighten anything: whenever
    ``max_colors_within_budget`` returns ``n_colors >= 2`` it returned from inside its loop,
    i.e. it already verified the cap fits at the rank count it assumed; the split can only
    *reduce* the color count, which only *raises* the mean ``ranks``; and
    :func:`estimate_gf_peak_bytes` is monotone non-increasing in ``ranks``. So
    ``unit_cap >= cap`` identically and ``min(cap, unit_cap) == cap`` -- a no-op. (Measured
    over a 400-cell grid of rank count x unit count x cap x budget: 385 no-op, and all 15
    binding cells had ``n_colors == 1``.) An adversarial review caught that this made the
    first shipped version of this function inert in exactly the production geometry it was
    written for. To bind, it must be given information the color inversion did not have --
    which is what the resident set is.

    **That no-op argument assumed the mean, and is now only half true.** Since
    ``run_units_distributed`` sizes each color on its own ``split_basis.comm.size``, a color
    apportioned *fewer* ranks than ``comm.size // n_colors`` (``_pack_units``'s floor-of-1
    step can do this whenever the mean is >= 2) evaluates this function at a *smaller*
    ``ranks`` than the color inversion assumed, so ``unit_cap < cap`` can bind on the real
    rank count alone, with no resident set involved. Colors at or above the mean still fall
    under the original argument. Both mechanisms tighten; neither loosens.

    Exponential-then-bisection, mirroring :func:`_suggest_for_budget`.

    Parameters
    ----------
    n_spin_orbitals : int
        Determinant bit width.
    block_width : int
        Widest unit's seed count (GF block width) about to run on this color.
    reort : str or None
        GF reorthogonalization mode.
    ranks : int
        Ranks in this unit's color -- the color's own ``split_basis.comm.size`` at the split
        site, **not** ``comm.size`` and **not** the mean ``comm.size // n_colors``
        (``_pack_units`` apportions ranks proportionally to bin mass, so colors differ).
    comm : MPI communicator
        The full communicator (``available_bytes_per_rank`` reads the per-node memory
        budget from it; the search itself is rank-local once the budget is known).
    safety : float
        Fraction of available RAM to budget (default ``DEFAULT_MEMORY_SAFETY``).
    krylov_dtype : optional
        Krylov store dtype; ``complex64`` halves the store and so raises the cap.
    method : str
        See :func:`estimate_gf_peak_bytes`.
    resident_bytes : int, optional
        What this rank already holds when the GF phase starts (``current_rss_bytes()``,
        MAX-reduced over the communicator by the caller). ``None`` (default) reproduces the
        pre-2026-09-14 budget, ``safety * available``, and with it this function cannot bind
        (see above).

        Given it, the budget becomes ``safety * (available + resident) - resident``: the
        process may occupy at most ``safety`` of its total per-rank share, it already holds
        ``resident``, so the GF phase may *add* only the difference. This is a strictly
        tighter bound than ``safety * available`` (by ``resident * (1 - safety)``) and it is
        the constraint the color inversion structurally cannot express -- that inversion runs
        against *remaining* headroom, which is a snapshot taken before every rank on the node
        grows its unit basis at once.

        ``available`` is node ``MemAvailable`` divided by ranks-per-node, so it is already
        net of what this process holds; adding ``resident`` back reconstructs the rank's total
        share before applying ``safety`` to it. If the process is already over its safety
        share the difference is non-positive, which would cap the GF at the 1-determinant
        floor and silently destroy the physics; that case falls back to ``safety * available``
        and is the caller's cue to warn.

    Returns
    -------
    int
        Largest determinant count (at least 1) whose predicted per-rank GF peak, at
        ``ranks``, fits the budget described above.
    """
    budget = _resident_adjusted_budget(safety, available_bytes_per_rank(comm), resident_bytes)

    def fits(n):
        return (
            estimate_gf_peak_bytes(
                n, n_spin_orbitals, block_width, reort, ranks=ranks, krylov_dtype=krylov_dtype, method=method
            )
            <= budget
        )

    lo, hi = 1, 1024
    while fits(hi) and hi < 10**13:
        lo, hi = hi, hi * 2
    if hi >= 10**13:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _suggest_for_budget(
    budget,
    n_spin_orbitals,
    block_width,
    reort,
    n_parallel_units,
    nnz_per_state,
    ranks,
    krylov_dtype=None,
    method="lanczos",
    gs_num_wanted=None,
):
    """Largest ``n`` with both path estimates within ``budget``, by bisection. Rank-local."""
    ranks_per_unit = max(1, ranks // max(1, n_parallel_units))

    def fits(n):
        gs = estimate_gs_peak_bytes(n, n_spin_orbitals, block_width, ranks, nnz_per_state, num_wanted=gs_num_wanted)
        gf = estimate_gf_peak_bytes(
            n, n_spin_orbitals, block_width, reort, ranks_per_unit, krylov_dtype=krylov_dtype, method=method
        )
        return max(gs, gf) <= budget

    lo, hi = 1, 1024
    while fits(hi) and hi < 10**13:
        lo, hi = hi, hi * 2
    if hi >= 10**13:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def log_memory_budget(
    truncation_threshold,
    n_spin_orbitals,
    comm=None,
    block_width=4,
    reort="none",
    n_parallel_units=1,
    nnz_per_state=100,
    verbose=True,
    label="",
    krylov_dtype=None,
    method="lanczos",
    gs_num_wanted=None,
):
    """Predict peak memory for a chosen threshold, print it on rank 0, warn if it won't fit.

    .. warning:: **Collective on** ``comm`` (calls :func:`available_bytes_per_rank`).
       Call it unconditionally on every rank; only the printing is gated on
       ``verbose``/rank 0, so per-rank verbosity flags are safe.

    Parameters
    ----------
    truncation_threshold : int or float
        The chosen global determinant cap (``inf`` reports "uncapped" and only the
        availability figure).
    n_spin_orbitals, comm, block_width, reort, n_parallel_units, nnz_per_state
        See :func:`suggest_truncation_threshold`.
    verbose : bool
        Gate for the rank-0 print (may safely differ across ranks).
    label : str
        Prefix for the log lines (e.g. the cluster name).
    gs_num_wanted : int, optional
        See :func:`suggest_truncation_threshold`.

    Returns
    -------
    dict
        ``{"available_per_rank", "gs_peak", "gf_peak", "fits"}`` in bytes/bool
        (``gs_peak``/``gf_peak`` are ``None`` when uncapped).
    """
    ranks = comm.size if comm is not None else 1
    rank = comm.rank if comm is not None else 0
    available = available_bytes_per_rank(comm)
    uncapped = truncation_threshold is None or not (truncation_threshold < float("inf"))
    if uncapped:
        gs = gf = None
        fits = False
    else:
        n = int(truncation_threshold)
        ranks_per_unit = max(1, ranks // max(1, n_parallel_units))
        gs = estimate_gs_peak_bytes(n, n_spin_orbitals, block_width, ranks, nnz_per_state, num_wanted=gs_num_wanted)
        gf = estimate_gf_peak_bytes(
            n, n_spin_orbitals, block_width, reort, ranks_per_unit, krylov_dtype=krylov_dtype, method=method
        )
        fits = max(gs, gf) <= available
    prefix = f"{label}: " if label else ""
    if verbose and rank == 0:
        if uncapped:
            print(f"{prefix}truncation_threshold=inf (uncapped); {format_bytes(available)}/rank available.", flush=True)
        else:
            print(
                f"{prefix}truncation_threshold={int(truncation_threshold):,}: predicted per-rank peak "
                f"{format_bytes(gs)} (ground state) / {format_bytes(gf)} (Green's function), "
                f"{format_bytes(available)}/rank available.",
                flush=True,
            )
    # Only meaningful once a peak was actually computed above -- the uncapped branch never
    # calls estimate_gs_peak_bytes, so there is nothing above to warn about.
    if verbose and rank == 0 and not uncapped:
        knob_set = config.GS_MAX_BLOCK_WIDTH.get() is not None
        if not knob_set:
            print(
                f"{prefix}GS_MAX_BLOCK_WIDTH is unset: the ground-state Lanczos block width grows "
                f"with the manifold and is not bounded by the block_width={block_width} used above -- "
                "the ground-state peak figure is a placeholder, not a measured bound. Set "
                "GS_MAX_BLOCK_WIDTH for a budget that reflects the real solve.",
                flush=True,
            )
        elif gs_num_wanted is None:
            assumed = _GS_COUPLED_NUM_WANTED_RATIO * block_width
            print(
                f"{prefix}GS_MAX_BLOCK_WIDTH is set but gs_num_wanted was not supplied: the ground-state "
                f"Krylov term assumes num_wanted~={assumed} ({_GS_COUPLED_NUM_WANTED_RATIO}*block_width), "
                "which under-counts by the manifold-to-width ratio measured at production scale (up to "
                "~30x, see doc/plans/dc_smo_performance.md) -- pass the value measured by the same width "
                "sweep that set GS_MAX_BLOCK_WIDTH.",
                flush=True,
            )
    # Ungated: an OOM prediction is a warning about a real problem, not detail. The
    # informational budget lines above stay behind `verbose`.
    if not uncapped and not fits and rank == 0:
        suggestion = _suggest_for_budget(
            DEFAULT_MEMORY_SAFETY * available,
            n_spin_orbitals,
            block_width,
            reort,
            n_parallel_units,
            nnz_per_state,
            ranks,
            krylov_dtype,
            method,
            gs_num_wanted,
        )
        print(
            f"{prefix}WARNING: predicted peak exceeds available memory; consider "
            f"truncation_threshold<={suggestion:,} or more ranks.",
            flush=True,
        )
    return {"available_per_rank": available, "gs_peak": gs, "gf_peak": gf, "fits": fits}


def _proc_status_bytes(key):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def peak_rss_bytes():
    """This process's high-water-mark RSS (``VmHWM`` from /proc/self/status); 0 if unreadable."""
    return _proc_status_bytes("VmHWM:")


def current_rss_bytes():
    """This process's resident set right now (``VmRSS`` from /proc/self/status); 0 if unreadable."""
    return _proc_status_bytes("VmRSS:")


def reset_peak_rss():
    """Reset this process's ``VmHWM`` to its current RSS, so the next :func:`peak_rss_bytes` reads
    the peak of what runs *after* this call rather than of the whole process lifetime.

    Writes ``5`` to ``/proc/self/clear_refs`` (Linux >= 4.0; needs no privilege). Returns whether
    it worked -- a caller that cannot reset must treat the peak as cumulative, not as its own.

    Why it exists: ``VmHWM`` never decreases, so once one solve has set a high mark every later
    step's own peak is invisible behind it. The SrMnO3 double-counting crash log shows exactly
    that -- three CIPSI cycles whose selection rounds grew 10x each all reported the previous
    sector's "2.0 GiB" (``doc/plans/dc_smo_memory.md``, round 6).
    """
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def log_peak_vs_predicted(memory_budget, comm=None, verbose=True, label=""):
    """Print the measured per-rank peak RSS next to the predicted peaks, for re-calibration.

    .. warning:: **Collective on** ``comm`` (MAX-allreduce of the per-rank ``VmHWM``).
       Call it unconditionally on every rank; only the printing is gated on
       ``verbose``/rank 0.

    The measured figure includes the Python/import floor (~hundreds of MiB) that the
    byte model deliberately does not count; on production-size runs the determinant
    terms dominate and the comparison calibrates ``_PY_BASIS_OVERHEAD_BYTES`` and
    ``nnz_per_state`` (see ``doc/plans/truncation_reliability.md``).

    Parameters
    ----------
    memory_budget : dict
        The return value of :func:`log_memory_budget` for the run being measured.
    comm : MPI communicator, optional
    verbose : bool
        Gate for the rank-0 print (may safely differ across ranks).
    label : str
        Prefix for the log line (e.g. the cluster name).

    Returns
    -------
    int
        Measured peak RSS in bytes (communicator-wide maximum).
    """
    measured = peak_rss_bytes()
    if comm is not None and comm.size > 1:
        measured = comm.allreduce(measured, op=MPI.MAX)
    if verbose and (comm is None or comm.rank == 0):
        prefix = f"{label}: " if label else ""
        gs, gf = memory_budget.get("gs_peak"), memory_budget.get("gf_peak")
        if gs is None:
            predicted = "uncapped"
        else:
            predicted = f"{format_bytes(gs)} (ground state) / {format_bytes(gf)} (Green's function)"
        print(
            f"{prefix}measured per-rank peak RSS {format_bytes(measured)} (includes the Python/import floor); "
            f"predicted {predicted}.",
            flush=True,
        )
    return measured


def format_bytes(n):
    """Human-readable bytes (binary units)."""
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TiB"


def _main():
    """Interactive sizing probe: ``[mpiexec -n R] python -m impurityModel.ed.memory_estimate``."""
    import argparse

    parser = argparse.ArgumentParser(description="Probe per-rank memory and suggest a truncation_threshold.")
    parser.add_argument("--n-spin-orbitals", type=int, default=120, help="determinant bit width (default 120)")
    parser.add_argument("--block-width", type=int, default=4, help="Lanczos block width (default 4)")
    parser.add_argument("--reort", default="none", help="GF reorthogonalization mode (default none)")
    parser.add_argument("--n-parallel-units", type=int, default=1, help="simultaneous unit colors (default 1)")
    parser.add_argument("--nnz-per-state", type=int, default=100, help="stored H elements per state (default 100)")
    parser.add_argument("--safety", type=float, default=0.5, help="fraction of available RAM to budget (default 0.5)")
    parser.add_argument(
        "--gs-num-wanted",
        type=int,
        default=None,
        help="measured ground-state num_wanted (from the same width sweep as --block-width when "
        "GS_MAX_BLOCK_WIDTH is set); default None assumes num_wanted ~= 2*block_width",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD if MPI.COMM_WORLD.size > 1 else None
    suggestion = suggest_truncation_threshold(
        args.n_spin_orbitals,
        comm=comm,
        block_width=args.block_width,
        reort=args.reort,
        n_parallel_units=args.n_parallel_units,
        nnz_per_state=args.nnz_per_state,
        safety=args.safety,
        gs_num_wanted=args.gs_num_wanted,
    )
    if comm is None or comm.rank == 0:
        cgroup = _cgroup_available_bytes()
        print(f"node available (min of MemAvailable and cgroup headroom): {format_bytes(_node_available_bytes())}")
        print(f"cgroup memory headroom: {format_bytes(cgroup) if cgroup is not None else 'unlimited'}")
    log_memory_budget(
        suggestion,
        args.n_spin_orbitals,
        comm=comm,
        block_width=args.block_width,
        reort=args.reort,
        n_parallel_units=args.n_parallel_units,
        nnz_per_state=args.nnz_per_state,
        label=f"suggested (safety {args.safety})",
        gs_num_wanted=args.gs_num_wanted,
    )


if __name__ == "__main__":
    _main()
