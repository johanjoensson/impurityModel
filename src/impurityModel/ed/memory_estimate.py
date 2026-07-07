"""
Memory sizing helpers for choosing ``truncation_threshold``.

``truncation_threshold`` caps the *global* number of Slater determinants in a
:class:`~impurityModel.ed.manybody_basis.Basis`. This module turns that count into
predicted per-rank peak memory (and back), so drivers can pick a threshold that fits
in RAM instead of guessing.

The byte formulas mirror the authoritative ``memory_bytes()`` estimators in
``src/cython/ManyBodyUtils.pyx`` (``ManyBodyState``, ``ManyBodyBlockState`` and
``SparseKrylovDense``); the pure-Python overhead constants are rough and are
calibrated against measured RSS in ``doc/plans/truncation_reliability.md``.

Layering: this module sits at the bottom of the stack (numpy/mpi4py/os only) so both
the basis layer and the solver drivers may import it.

Two per-rank scaling regimes matter (see ``doc/architecture_overview.md``):

* the sparse (hash-distributed) kernels hold ``~n_dets / ranks`` determinants per
  rank, so per-rank memory shrinks with the communicator size;
* the array block-Lanczos kernel replicates the full ``(global_N, block_width)``
  matvec product on every rank (``BlockLanczosArray.pyx``), so that term does *not*
  shrink with the rank count. It usually dominates the ground-state estimate.

Under ``run_units_distributed`` the communicator is split into colors and every unit
basis inherits the same numeric ``truncation_threshold``, so each rank's share of a
unit basis is ``threshold / (ranks / n_colors)`` — parallel units multiply per-rank
memory accordingly (pass ``n_parallel_units``).
"""

import os
from math import ceil

from mpi4py import MPI

# sizeof(pair<SlaterDeterminant, complex<double>>) in the flat_map entry array:
# a std::vector<uint64_t> header (24 B) + complex<double> (16 B).
_FLAT_MAP_ENTRY_BYTES = 40
# sizeof(std::vector<uint64_t>): the in-line key header stored per ManyBodyBlockState row.
_SD_STRUCT_BYTES = 24
# SparseKrylovDense support-map node overhead per registered row (ManyBodyUtils.pyx).
_KRYLOV_NODE_BYTES = 72
_COMPLEX_BYTES = 16
# scipy CSC complex128: 16 B value + index/indptr (int32 or int64) per stored element.
_CSR_BYTES_PER_NNZ = 24
# Python-side Basis bookkeeping per local determinant: SlaterDeterminant wrapper object,
# local_basis list slot and _index_dict entry. Rough; calibrated against measured RSS.
_PY_BASIS_OVERHEAD_BYTES = 160

#: Fallback cap used by drivers when no memory probe is possible (matches the historical
#: ``groundstate.calc_gs`` default).
DEFAULT_TRUNCATION_THRESHOLD = 1_000_000

_ranks_per_node_cache: dict = {}


def _retains_krylov(reort):
    """Whether a reort mode (string, ``Reort`` enum member or None) retains the Krylov store."""
    name = getattr(reort, "name", reort)
    return name is not None and str(name).lower() != "none"


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


def estimate_gf_peak_bytes(n_dets, n_spin_orbitals, block_width, reort="none", ranks=1, n_blocks=None):
    """Predicted per-rank peak bytes of the sparse (MBS-kernel) Green's-function path.

    Counts the excited ``Basis`` bookkeeping, the ~3 live ``ManyBodyBlockState`` blocks
    of the recurrence (``q_prev``, ``q_curr``, ``wp``) and, at ``reort != "none"``, the
    ``SparseKrylovDense`` retention (its rows are bounded by the retained determinant
    set, its columns by the number of Lanczos blocks).

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

    Returns
    -------
    int
        Predicted per-rank peak bytes.
    """
    local_rows = ceil(n_dets / max(1, ranks))
    key_heap = _key_heap_bytes(n_spin_orbitals)
    basis_bytes = local_rows * (bytes_per_determinant(n_spin_orbitals) + _PY_BASIS_OVERHEAD_BYTES)
    row_bytes = _COMPLEX_BYTES * block_width + key_heap + _SD_STRUCT_BYTES
    live_bytes = 3 * local_rows * row_bytes
    store_bytes = 0
    if _retains_krylov(reort):
        if n_blocks is None:
            n_blocks = ceil(n_dets / max(1, block_width))
        store_bytes = local_rows * (_COMPLEX_BYTES * block_width * n_blocks + 2 * key_heap + _KRYLOV_NODE_BYTES)
    return basis_bytes + live_bytes + store_bytes


def estimate_gs_peak_bytes(n_dets, n_spin_orbitals, block_width=4, ranks=1, nnz_per_state=100, n_blocks=30):
    """Predicted per-rank peak bytes of the ground-state (CIPSI + array-kernel) path.

    Counts the ``Basis`` bookkeeping and the CSR Hamiltonian snapshot (both hash
    distributed, ~1/ranks per rank), the array kernel's replicated full
    ``(global_N, block_width)`` matvec product (per rank, *not* divided by ranks — see
    ``BlockLanczosArray.pyx``), and the retained dense Krylov blocks at the ground-state
    default ``reort="full"``.

    Parameters
    ----------
    n_dets : int
        Global determinant count (the ``truncation_threshold`` being considered).
    n_spin_orbitals : int
        Determinant bit width.
    block_width : int
        Lanczos block width (number of sought eigenvectors).
    ranks : int
        MPI ranks sharing the basis.
    nnz_per_state : int
        Stored Hamiltonian elements per basis state (measure on a small run; grows with
        the number of one-/two-body terms).
    n_blocks : int
        Typical converged Lanczos depth for the retained dense Krylov basis.

    Returns
    -------
    int
        Predicted per-rank peak bytes.
    """
    local = ceil(n_dets / max(1, ranks))
    basis_bytes = local * (bytes_per_determinant(n_spin_orbitals) + _PY_BASIS_OVERHEAD_BYTES)
    csr_bytes = local * nnz_per_state * _CSR_BYTES_PER_NNZ
    replicated_bytes = n_dets * block_width * _COMPLEX_BYTES
    krylov_bytes = local * block_width * n_blocks * _COMPLEX_BYTES
    return basis_bytes + csr_bytes + replicated_bytes + krylov_bytes


def _node_available_bytes():
    """Available bytes on this node: ``MemAvailable`` from /proc/meminfo, sysconf fallback."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


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
    safety=0.5,
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
        Fraction of available RAM to budget (default 0.5).

    Returns
    -------
    int
        Suggested global determinant cap (at least 1).
    """
    budget = safety * available_bytes_per_rank(comm)
    ranks = comm.size if comm is not None else 1
    return _suggest_for_budget(budget, n_spin_orbitals, block_width, reort, n_parallel_units, nnz_per_state, ranks)


def _suggest_for_budget(budget, n_spin_orbitals, block_width, reort, n_parallel_units, nnz_per_state, ranks):
    """Largest ``n`` with both path estimates within ``budget``, by bisection. Rank-local."""
    ranks_per_unit = max(1, ranks // max(1, n_parallel_units))

    def fits(n):
        gs = estimate_gs_peak_bytes(n, n_spin_orbitals, block_width, ranks, nnz_per_state)
        gf = estimate_gf_peak_bytes(n, n_spin_orbitals, block_width, reort, ranks_per_unit)
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
        gf = estimate_gf_peak_bytes(n, n_spin_orbitals, block_width, reort, ranks_per_unit)
        fits = max(gs, gf) <= available
    if verbose and rank == 0:
        prefix = f"{label}: " if label else ""
        if uncapped:
            print(f"{prefix}truncation_threshold=inf (uncapped); {_fmt_bytes(available)}/rank available.", flush=True)
        else:
            print(
                f"{prefix}truncation_threshold={int(truncation_threshold):,}: predicted per-rank peak "
                f"{_fmt_bytes(gs)} (ground state) / {_fmt_bytes(gf)} (Green's function), "
                f"{_fmt_bytes(available)}/rank available.",
                flush=True,
            )
            if not fits:
                suggestion = _suggest_for_budget(
                    0.5 * available, n_spin_orbitals, block_width, reort, n_parallel_units, nnz_per_state, ranks
                )
                print(
                    f"{prefix}WARNING: predicted peak exceeds available memory; consider "
                    f"truncation_threshold<={suggestion:,} or more ranks.",
                    flush=True,
                )
    return {"available_per_rank": available, "gs_peak": gs, "gf_peak": gf, "fits": fits}


def _fmt_bytes(n):
    """Human-readable bytes (binary units)."""
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TiB"
