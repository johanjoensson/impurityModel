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
from math import ceil

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
# scipy CSC complex128: 16 B value + index/indptr (int32 or int64) per stored element.
_CSR_BYTES_PER_NNZ = 24
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
    production self-energy path) ``s_live`` is the whole cost. Neither universally dominates —
    the earlier "the store dominates against ~450 for everything else" framing was only half
    right (the ~450 is real; the store does not always win). The recurrence's transient matvec
    fanout past the cap is deliberately *not* modelled here; ``DEFAULT_MEMORY_SAFETY`` absorbs it.

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
        ``run_units_distributed``).
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
    local_rows = ceil(n_dets / max(1, ranks))
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
    store_bytes = 0
    if _retains_krylov(reort):
        if n_blocks is None:
            n_blocks = ceil(n_dets / max(1, block_width))
        itemsize = _krylov_itemsize(reort, krylov_dtype)
        store_bytes = local_rows * (itemsize * block_width * n_blocks + 2 * key_heap + _KRYLOV_NODE_BYTES)
    return basis_bytes + live_bytes + store_bytes


#: Additive eigenstate padding TRLM's initial sizing applies before certifying a manifold
#: complete (mirrors ``cipsi_solver._EIGENSTATE_PAD``; not imported -- this module sits below
#: the solver stack, see the module docstring -- so keep the two in sync by hand if that
#: constant changes; both are 10 as of this writing).
_GS_EIGENSTATE_PAD = 10


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
    nw = (2 * p if num_wanted is None else num_wanted) + _GS_EIGENSTATE_PAD
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


def estimate_gs_peak_bytes(n_dets, n_spin_orbitals, block_width=4, ranks=1, nnz_per_state=100, num_wanted=None):
    """Predicted per-rank peak bytes of the ground-state (CIPSI + array-kernel) path.

    Counts the ``Basis`` bookkeeping and the CSR Hamiltonian snapshot (both hash
    distributed, ~1/ranks per rank), the array kernel's chunked reduce-scatter matvec
    transient (``_block_ops.pxi``'s ``block_apply``, post row-chunking: bounded by
    ``max(counts)`` -- the largest single rank's local row/column count under the hash
    partition, approximated here as ``local`` since a materially skewed partition, not
    ``global_N``, is now the risk this term misses -- see :data:`DEFAULT_MEMORY_SAFETY`,
    which is what absorbs it: this function has no way to see another rank's share), and
    the retained dense Krylov blocks at the ground-state default ``reort="full"``.

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
        the knob is set. :func:`log_memory_budget` warns on that path.
    ranks : int
        MPI ranks sharing the basis.
    nnz_per_state : int
        Stored Hamiltonian elements per basis state (measure on a small run; grows with
        the number of one-/two-body terms).
    num_wanted : int, optional
        Forwarded to :func:`_gs_krylov_columns`; ``None`` assumes ``2 * block_width``.

    Returns
    -------
    int
        Predicted per-rank peak bytes.
    """
    local = ceil(n_dets / max(1, ranks))
    basis_bytes = local * (bytes_per_determinant(n_spin_orbitals) + _PY_BASIS_OVERHEAD_BYTES)
    csr_bytes = local * nnz_per_state * _CSR_BYTES_PER_NNZ
    # Chunked reduce-scatter transient (Phase 1): one (max(counts), w) chunk buffer plus the
    # (local, w) result live at once; max(counts) ~ local under a balanced hash partition.
    replicated_bytes = 2 * local * block_width * _COMPLEX_BYTES
    krylov_bytes = local * _gs_krylov_columns(n_dets, block_width, num_wanted) * _COMPLEX_BYTES
    return basis_bytes + csr_bytes + replicated_bytes + krylov_bytes


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

    Returns
    -------
    int
        Suggested global determinant cap (at least 1).
    """
    budget = safety * available_bytes_per_rank(comm)
    ranks = comm.size if comm is not None else 1
    return _suggest_for_budget(
        budget, n_spin_orbitals, block_width, reort, n_parallel_units, nnz_per_state, ranks, krylov_dtype, method
    )


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
    ``n_colors <= max_candidate`` for which a cap-filling unit basis still fits
    ``safety * available_bytes_per_rank``. At ``reort != "none"`` the estimate uses the
    invariant-subspace worst case for the Krylov store (very conservative), consistent
    with :func:`suggest_truncation_threshold`.

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
    """
    budget = safety * available_bytes_per_rank(comm)
    for n_colors in range(max_candidate, 1, -1):
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
):
    """Largest ``n`` with both path estimates within ``budget``, by bisection. Rank-local."""
    ranks_per_unit = max(1, ranks // max(1, n_parallel_units))

    def fits(n):
        gs = estimate_gs_peak_bytes(n, n_spin_orbitals, block_width, ranks, nnz_per_state)
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
        gs = estimate_gs_peak_bytes(n, n_spin_orbitals, block_width, ranks, nnz_per_state)
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
    if verbose and rank == 0 and config.GS_MAX_BLOCK_WIDTH.get() is None:
        print(
            f"{prefix}GS_MAX_BLOCK_WIDTH is unset: the ground-state Lanczos block width grows "
            f"with the manifold and is not bounded by the block_width={block_width} used above -- "
            "the ground-state peak figure is a placeholder, not a measured bound. Set "
            "GS_MAX_BLOCK_WIDTH for a budget that reflects the real solve.",
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
        )
        print(
            f"{prefix}WARNING: predicted peak exceeds available memory; consider "
            f"truncation_threshold<={suggestion:,} or more ranks.",
            flush=True,
        )
    return {"available_per_rank": available, "gs_peak": gs, "gf_peak": gf, "fits": fits}


def peak_rss_bytes():
    """This process's high-water-mark RSS (``VmHWM`` from /proc/self/status); 0 if unreadable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


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
    )


if __name__ == "__main__":
    _main()
