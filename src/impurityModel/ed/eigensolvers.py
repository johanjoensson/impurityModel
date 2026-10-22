"""
Low-spectrum eigensolvers of the ground-state path.

* :func:`solve_window` -- every eigenstate inside an energy window, certified complete, from the
  thick-restart block Lanczos solver (:mod:`impurityModel.ed.trlm`). Used above ``dense_cutoff``.
* :func:`solve_lowest` -- the lowest eigenpair alone, with no claim about its degeneracy.
* :func:`eigensystem` -- dense diagonalization, complete by construction, below ``dense_cutoff``.

A caller says which of two manifolds it needs: :class:`Lowest` (the energy of the ground level, which
the occupation walk and every double-counting sector solve use) or :class:`Within` (every state inside
an energy window, certified complete, which the production thermal ground state needs). The CIPSI
solver cuts a dense or window result with one rule (``_energy_cut_indices``).
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed.average import energy_cut
from impurityModel.ed import config
from impurityModel.ed.BlockLanczosCore import block_apply, block_inner, block_normalize, block_orthogonalize
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.memory_estimate import gs_krylov_blocks
from impurityModel.ed.trlm import STATUS_CONVERGED, Window, trlm

_U64 = (1 << 64) - 1
#: Salt separating the imaginary part's hash stream from the real part's (the splitmix64 "golden
#: gamma" of a second stream), shared by every hash-seeded start block.
_IMAG_STREAM_SALT = 0xD1B54A32D192ED03
#: Well-spread odd multiplier deriving a column's salt from (round, column): the 64-bit PCG LCG constant.
_SALT_MULTIPLIER = 0x5851F42D4C957F2D


def _splitmix64(x: int) -> int:
    """One splitmix64 round: a well-mixed 64-bit hash of a 64-bit integer."""
    x = (x + 0x9E3779B97F4A7C15) & _U64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64
    return x ^ (x >> 31)


def _splitmix64_array(x):
    """:func:`_splitmix64` over a ``uint64`` array, bit for bit (numpy's unsigned arithmetic wraps mod 2**64)."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def degenerate_groups(e, tol):
    """Partition ascending energies ``e`` into manifolds: runs whose members lie within ``tol`` of the first."""
    e = np.asarray(e).real
    groups, start = [], 0
    for i in range(1, len(e) + 1):
        if i == len(e) or abs(e[i] - e[start]) > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


def cold_vector(hashes):
    """The all-positive hash-seeded start vector, ``(n_local, 1)``.

    One value per determinant from its hash alone -- never from the rank, the iteration order or
    ``PYTHONHASHSEED`` -- so the Lanczos trajectory is identical at every rank count. Full support:
    every charge sector is reachable from it.
    """
    h = np.asarray(hashes, dtype=np.uint64)
    with np.errstate(over="ignore"):
        re = _splitmix64_array(h).astype(float) / 2.0**64
        im = _splitmix64_array(h ^ np.uint64(_IMAG_STREAM_SALT)).astype(float) / 2.0**64
    return (re + 1j * im).reshape(-1, 1)


def random_block(hashes, width, round_):
    """``(n_local, width)`` rank-independent pseudo-random block, centred on zero.

    Like :func:`cold_vector`, with a different salt per column and per ``round_`` so successive
    rounds explore new directions. Zero-centred so no column is biased toward the uniform vector.
    """
    h = np.asarray(hashes, dtype=np.uint64)
    out = np.empty((len(h), width), dtype=complex)
    scale = 1.0 / 2.0**64
    with np.errstate(over="ignore"):
        for k in range(width):
            salted = h ^ np.uint64(_splitmix64((_SALT_MULTIPLIER * (round_ + 1) + k) & _U64))
            re = _splitmix64_array(salted).astype(float) * scale
            im = _splitmix64_array(salted ^ np.uint64(_IMAG_STREAM_SALT)).astype(float) * scale
            out[:, k] = (re - 0.5) + 1j * (im - 0.5)
    return out


@dataclass(frozen=True)
class Lowest:
    """Request for the lowest eigenpair. Nothing is claimed about the degeneracy of its level."""


@dataclass(frozen=True)
class Within:
    """Request for every eigenstate within ``max_energy`` of the lowest, certified complete."""

    max_energy: float

    @classmethod
    def thermal(cls, tau):
        """The window that holds every state with a Boltzmann weight above the design weight at scale ``tau``."""
        return cls(energy_cut(tau))


@dataclass
class EigenResult:
    """The low spectrum of one operator on one basis.

    ``e`` ascending and ``vecs`` ``(n_local, len(e))``: the states inside the window followed by
    one bracket state just above it (``n_in`` counts the former). ``certified`` is True when
    every in-window level is shown complete (see :func:`solve_window`). ``max_multiplicity`` is
    the largest degenerate group in the window. ``generation`` is the ``Basis.generation`` the
    result belongs to and ``manifold`` the request it answers (:class:`Lowest` or :class:`Within`);
    a consumer re-solves when the generation no longer matches or it needs more than the request covered.
    """

    e: np.ndarray
    vecs: np.ndarray
    n_in: int
    certified: bool
    max_multiplicity: int
    status: str
    generation: int | None = None
    manifold: Lowest | Within | None = None


#: Singular values of ``F_G^H W`` below this are not counted as warm overlap with a level. ``F`` is
#: only ``tol``-accurate, so the product carries O(tol) mixing from neighbouring levels; counting
#: that noise would inflate the reachable-copies estimate and certify a level with missing copies,
#: while undercounting only triggers the (safe) fallback.
_OVERLAP_RANK_FLOOR = 1e-6
#: Fallback rounds before a still-saturating level is reported uncertified.
_MAX_FALLBACK_ROUNDS = 3


def _on_root(comm, compute):
    """``compute()`` on rank 0 (collective): its result there, ``None`` elsewhere.

    An exception on rank 0 is announced to every rank before anything else is exchanged and raised
    everywhere, so no rank is left waiting in the broadcast that would have carried the result.
    """
    result = error = None
    if comm is None or comm.rank == 0:
        try:
            result = compute()
        except Exception as exc:  # re-raised below, on every rank
            error = exc
    if comm is not None:
        failure = comm.bcast(None if error is None else f"{type(error).__name__}: {error}", root=0)
        if failure is not None:
            if error is not None:
                raise error
            raise RuntimeError(f"rank 0 failed: {failure}")
    elif error is not None:
        raise error
    return result


def _gather_to_root(h_op, comm):
    """The whole operator as a scipy CSC on rank 0 and ``None`` elsewhere (collective).

    Each rank sends its own columns' entries (global row, global column, value) with typed
    ``Gatherv`` -- ``O(nnz)`` -- where densifying ``n x n`` on every rank and summing them costs
    ``O(n^2)`` per rank. The column offsets come from the ranks' local sizes (the rank-contiguous
    numbering), not from ``col_offset``. Overlapping segments are summed, as :meth:`to_scipy` does.
    """
    h_op.unbind_slots()
    local = h_op.to_scipy().tocoo()
    offset, _counts = _rank_offsets(h_op.shape[1], comm)
    parts = [
        (np.ascontiguousarray(local.row, dtype=np.int64), MPI.INT64_T),
        (np.ascontiguousarray(local.col, dtype=np.int64) + offset, MPI.INT64_T),
        (np.ascontiguousarray(local.data, dtype=complex), MPI.C_DOUBLE_COMPLEX),
    ]
    sizes = np.empty(comm.size, dtype=np.int64)
    comm.Allgather(np.array([local.nnz], dtype=np.int64), sizes)
    displs = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    root = comm.rank == 0
    gathered = []
    for array, mpi_type in parts:
        out = np.empty(int(sizes.sum()), dtype=array.dtype) if root else None
        comm.Gatherv([array, mpi_type], [out, sizes, displs, mpi_type] if root else None, root=0)
        gathered.append(out)
    if not root:
        return None
    rows, cols, data = gathered
    n = h_op.shape[0]
    full = sps.coo_array((data, (rows, cols)), shape=(n, n)).tocsc()
    full.sum_duplicates()
    return full


def _distributed(h_local, comm):
    return isinstance(h_local, DistributedSparseMatrix) and comm is not None and comm.size > 1


def dense_eigensystem(
    h_local: Any, return_eigvecs: bool = True, comm: Optional[MPI.Comm] = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Solve the eigenvalue problem using dense matrix diagonalization.

    Parameters
    ----------
    h_local : Any
        The matrix to diagonalize.
    return_eigvecs : bool, default True
        Whether to return eigenvectors.
    comm : MPI.Comm, optional
        MPI communicator.

    Returns
    -------
    es : np.ndarray
        Array of eigenvalues.
    vecs : np.ndarray, optional
        Array of eigenvectors, returned if return_eigvecs is True.
    """
    rank = comm.rank if comm is not None else 0
    if _distributed(h_local, comm):
        full = _gather_to_root(h_local, comm)
        n = h_local.shape[0]
        h = full.toarray() if rank == 0 else None
        dtype = complex
    else:
        if hasattr(h_local, "toarray"):
            h = h_local.toarray()
        elif hasattr(h_local, "todense"):
            h = h_local.todense()
        else:
            h = h_local
        if comm is not None:
            comm.Reduce(h if rank != 0 else MPI.IN_PLACE, h, root=0, op=MPI.SUM)
        n, dtype = h.shape[0], h.dtype
    if return_eigvecs:
        solved = _on_root(comm, lambda: np.linalg.eigh(h, UPLO="L"))
        if rank == 0:
            es, vecs = solved
        else:
            es = np.empty(n, dtype=float, order="C")
            vecs = np.empty((n, n), dtype=dtype, order="C")
    else:
        solved = _on_root(comm, lambda: np.linalg.eigvalsh(h, UPLO="L"))
        es = solved if rank == 0 else np.empty(n, dtype=float)
    if comm is not None:
        comm.Bcast(es, root=0)
        if return_eigvecs:
            comm.Bcast(vecs, root=0)
    if return_eigvecs:
        return es, vecs
    return es


def dense_lowest(h_local, level_size, comm=None, first=16):
    """The lowest eigenpairs of a dense matrix, as many as its lowest level needs: ``(es, vecs)``.

    :func:`dense_eigensystem` computes every eigenpair and broadcasts the ``n x n`` vectors, where a
    ``Lowest`` solve reads only the lowest level. This computes the ``k`` lowest
    (``scipy.linalg.eigh(subset_by_index=...)``, the MRRR driver), ``k`` starting at ``first`` and
    doubling while ``level_size(es)`` -- the size of the lowest level among the ``k`` values -- fills
    them all, so a degenerate level is never cut. Measured at ``n = 1000`` (complex, one thread):
    0.7 s against 3.0 s, and the broadcast is ``n x k``. Gathered onto rank 0 like
    :func:`dense_eigensystem`; ``level_size`` must be a pure function of its argument, which rank 0
    alone evaluates.
    """
    import scipy.linalg

    rank = comm.rank if comm is not None else 0
    if _distributed(h_local, comm):
        full = _gather_to_root(h_local, comm)
        h = full.toarray() if rank == 0 else None
        n = h_local.shape[0]
    else:
        h = h_local.toarray() if hasattr(h_local, "toarray") else np.asarray(h_local)
        if comm is not None:
            comm.Reduce(h if rank != 0 else MPI.IN_PLACE, h, root=0, op=MPI.SUM)
        n = h.shape[0]

    def lowest_level():
        k = min(n, first)
        while True:
            es, vecs = scipy.linalg.eigh(h, lower=True, subset_by_index=[0, k - 1], driver="evr")
            if k == n or level_size(es) < k:
                return es, np.ascontiguousarray(vecs)
            k = min(n, 2 * k)

    solved = _on_root(comm, lowest_level)
    shape = None
    if rank == 0:
        es, vecs = solved
        shape = (len(es), vecs.dtype)
    if comm is not None:
        k, dtype = comm.bcast(shape, root=0)
        if rank != 0:
            es = np.empty(k, dtype=float)
            vecs = np.empty((n, k), dtype=dtype)
        comm.Bcast(es, root=0)
        comm.Bcast(vecs, root=0)
    return es, vecs


def eigensystem(h_local, e_max, eigenValueTol=0, return_eigvecs=True, comm=None):
    """
    Return the eigen-energies and eigenstates of a Hamiltonian matrix, densely diagonalized.

    Parameters
    ----------
    h_local : scipy.sparse sparse array (any kind) or ndarray
        Contains part of the full many-body Hamiltonian, local to this MPI rank; the parts are
        summed onto rank 0 and the result broadcast (see :func:`dense_eigensystem`).
    e_max : float or None
        Maximum energy difference for excited states. ``None`` keeps every computed state.
    eigenValueTol : float
        Floor on ``e_max``: the precision of the returned eigenvalues.
    return_eigvecs : bool
        If True, return eigenvalues and eigenvectors for all states with energy within e_max of the lowest energy state.
        If False, return only the calculated eigenvalues.
    comm : MPI communicator to use for any MPI communication
    """

    # e_max is limited by the accuracy of the calculated eigenvalues and machine precision.
    # e_max=None means "no energy cutoff" (get_eigenvectors passes max_energy=None): keep every
    # computed state. Guard the None here, otherwise max(None, ...) raises TypeError -- a live
    # crash on the dense (basis < dense_cutoff) path, which does not otherwise touch e_max.
    e_max = np.inf if e_max is None else max(e_max, eigenValueTol, np.finfo(float).eps * 100)

    if return_eigvecs:
        es, vecs = dense_eigensystem(h_local, return_eigvecs, comm)
    else:
        es = dense_eigensystem(h_local, return_eigvecs, comm)
        vecs = None
    indices = np.argsort(es)
    es = es[indices]
    if return_eigvecs and vecs is not None:
        vecs = vecs[:, indices]
    mask = es - np.min(es) <= e_max

    if return_eigvecs:
        return es[mask], vecs[:, mask]
    return es[mask]


def _global_gram_rank(W, mpi, comm, floor=1e-10):
    """Numerical rank of the columns of ``W`` (rows distributed over ``comm``)."""
    if W.shape[1] == 0:
        return 0
    gram = block_inner(W, W, mpi, comm)
    w = np.linalg.eigvalsh(0.5 * (gram + gram.conj().T))
    return int(np.count_nonzero(w > floor * max(float(w.max()), 1.0)))


def _level_overlap_ranks(F, groups, W, mpi, comm, floor):
    """``rank(F_G^H W)`` for every group ``G`` of columns of ``F``: how many copies the warm block reaches."""
    if W.shape[1] == 0 or not groups:
        return [0] * len(groups)
    M = block_inner(F, W, mpi, comm)  # (n_in, w), Allreduced: replicated
    return [int(np.count_nonzero(np.linalg.svd(M[g], compute_uv=False) > floor)) for g in groups]


def solve_window(
    h_op,
    comm,
    hashes,
    *,
    warm,
    max_energy,
    deg_tol,
    tol,
    num_wanted,
    n_global,
    n_random=4,
    max_keep=None,
    max_restarts=100,
    verbose=False,
):
    """All eigenstates within ``max_energy`` of the lowest, with a completeness certificate.

    One thick-restart block-Lanczos solve in window mode (:class:`impurityModel.ed.trlm.Window`):
    it retains as many Ritz vectors as the window needs, so the multiplicity of the levels inside
    need not be known beforehand.

    **Start block.** The ``warm`` columns (a previous result's eigenvectors, possibly empty) plus
    ``n_random`` generic columns: the all-positive hash vector :func:`cold_vector` and zero-centred
    :func:`random_block` columns. They are rank-independent and have full support, so every charge
    sector is reachable from them whatever the warm block spans.

    **Certificate.** A block Krylov space started from ``Q0`` meets the eigenspace of a level
    ``lam`` in ``range(P_lam Q0)``, of dimension ``min(g, rank(P_lam W) + R)`` for ``g`` copies,
    ``W`` the warm columns and ``R`` generic random ones. Having found ``k`` converged copies, the
    level is therefore complete when ``k < rank(F^H W) + R``: otherwise the Krylov space was
    saturated and copies may be missing. Probabilistic (the random columns are generic), and it
    assumes the copies of an exactly degenerate level converge together. ``rank`` counts singular
    values above ``max(1e-6, 100 tol)``.

    **Fallback**, only for a saturated level: lock the converged in-window vectors, restart from
    ``2 k_max`` fresh random columns in their orthogonal complement with the same absolute window,
    and certify what turns up there with ``k_new < R'``. Bounded rounds.

    Parameters
    ----------
    h_op
        ``DistributedSparseMatrix`` (or coercible) of the operator on the current basis.
    hashes : ndarray of uint64, ``(n_local,)``
        The routing hash of every locally owned determinant, in row order.
    warm : ndarray ``(n_local, w)`` or None
    max_energy, deg_tol, tol : float
        Window width above the lowest state, degeneracy tolerance, residual tolerance.
    num_wanted : int
        Initial number of retained Ritz vectors; grows with the window.
    n_global : int
        Dimension of the operator (bounds the Krylov depth).
    n_random : int
        Generic columns ``R`` in the start block.
    max_keep : int, optional
        Ceiling on retained vectors; reaching it returns ``certified=False``.

    Returns
    -------
    EigenResult
        Collective on ``comm``; every field is replicated except ``vecs`` (row-distributed).
    """
    mpi = comm is not None and getattr(comm, "size", 1) > 1
    n_local = len(hashes)
    warm = np.zeros((n_local, 0), dtype=complex) if warm is None else np.ascontiguousarray(warm, dtype=complex)
    n_random = max(1, int(n_random))

    def start_block(with_warm):
        cols = [warm] if with_warm else []
        cols.append(cold_vector(hashes))
        if n_random > 1:
            cols.append(random_block(hashes, n_random - 1, 0))
        return np.ascontiguousarray(np.hstack(cols))

    def krylov_depth(width, wanted):
        return gs_krylov_blocks(n_global, width, wanted)

    def run(psi0, wanted, locked=None, window=None):
        return trlm(
            psi0,
            h_op,
            comm,
            num_wanted=wanted,
            max_subspace_blocks=krylov_depth(psi0.shape[1], wanted),
            tol=tol,
            max_restarts=max_restarts,
            verbose=verbose,
            locked=locked,
            window=window or Window(max_energy, deg_tol),
            max_keep=max_keep,
        )

    with_warm = warm.shape[1] > 0
    psi0 = start_block(with_warm)
    try:
        q0, _ = block_normalize(psi0, mpi, comm, 0.0)
    except ValueError:  # BlockBreakdown, rank-symmetric: decided by the replicated TSQR factor
        with_warm = False
        psi0 = start_block(False)
        q0, _ = block_normalize(psi0, mpi, comm, 0.0)
    rank_w = _global_gram_rank(warm, mpi, comm) if with_warm else 0
    r_eff = max(0, q0.shape[1] - rank_w)
    wanted = max(1, min(int(num_wanted), n_global - 1))
    res = run(q0, wanted)
    e, vecs, n_in = res.eigvals, res.eigvecs, res.n_in
    certified = res.status in STATUS_CONVERGED
    status = res.status
    if len(e) == 0:
        return EigenResult(e, vecs, 0, False, 0, status)

    floor = max(_OVERLAP_RANK_FLOOR, 100.0 * tol)
    groups = degenerate_groups(e[:n_in], deg_tol)
    ranks = _level_overlap_ranks(vecs[:, :n_in], groups, warm if with_warm else warm[:, :0], mpi, comm, floor)
    saturated = [len(g) >= c + r_eff for g, c in zip(groups, ranks)]
    k_max = max((len(g) for g in groups), default=0)

    rounds = 0
    while certified and any(saturated) and rounds < _MAX_FALLBACK_ROUNDS:
        rounds += 1
        e0 = float(e[0])
        top = max(e0 + max_energy, float(e[n_in - 1])) + deg_tol
        locked = np.ascontiguousarray(vecs[:, :n_in])
        r_new = max(4, 2 * k_max)
        fresh = random_block(hashes, r_new, rounds)
        win = Window(top - e0, deg_tol, e0=e0)
        if locked.shape[1] >= n_global:
            saturated = []  # the window is the whole space: nothing left to miss
            break
        try:
            res2 = run(fresh, max(r_new, 1), locked=locked, window=win)
        except ValueError:  # BlockBreakdown, rank-symmetric: the complement of the locked set is empty
            saturated = []
            break
        certified = res2.status in STATUS_CONVERGED
        status = res2.status
        e_new, v_new, n_new = res2.eigvals[: res2.n_in], res2.eigvecs[:, : res2.n_in], res2.n_in
        bracket_e = res2.eigvals[n_new : n_new + 1]
        bracket_v = res2.eigvecs[:, n_new : n_new + 1]
        keep_bracket = bracket_e.size and (e.size <= n_in or bracket_e[0] < e[n_in])
        e_all = np.concatenate([e[:n_in], e_new])
        v_all = np.hstack([vecs[:, :n_in], v_new])
        order = np.argsort(e_all, kind="stable")
        e_all, v_all = e_all[order], np.ascontiguousarray(v_all[:, order])
        if keep_bracket:
            e_br, v_br = bracket_e, bracket_v
        else:
            e_br, v_br = e[n_in : n_in + 1], vecs[:, n_in : n_in + 1]
        e = np.concatenate([e_all, e_br])
        vecs = np.hstack([v_all, v_br])
        n_in = len(e_all)
        groups = degenerate_groups(e[:n_in], deg_tol)
        k_max = max((len(g) for g in groups), default=0)
        # Copies found in the complement are certified against the fresh columns alone: ``k_new < R'``.
        new_groups = degenerate_groups(e_new, deg_tol)
        saturated = [len(g) >= r_new for g in new_groups]

    return EigenResult(
        e, vecs, n_in, bool(certified and not any(saturated)), max((len(g) for g in groups), default=0), status
    )


#: Most search vectors :func:`davidson_lowest` holds (``V`` and ``H V`` each); at the cap it restarts
#: on the current and previous Ritz vectors. On the 93 captured NiO 5-bath d7 class solves, 8, 12, 20
#: and 30 vectors took 29.2, 29.3, 28.5 and 28.0 matrix-vector products on average, while the
#: per-iteration work grows with the count.
DAVIDSON_MAX_VECTORS = 8

#: Iterations before :func:`davidson_lowest` gives up and :func:`solve_lowest` falls back to TRLM.
DAVIDSON_MAX_ITER = 2000

#: Iterations the residual may go without halving before :func:`davidson_lowest` declares it stagnant
#: and hands over to TRLM. On the 93 captured NiO 5-bath d7 class solves the longest such plateau in
#: a solve that went on to converge was 30 iterations (a near-degenerate pair 2e-8 apart, at a tol of
#: 1e-8); a 25-iteration window failed two of them.
DAVIDSON_STAGNATION = 100

#: The diagonal preconditioner's denominator ``theta - H_ii`` is floored at this magnitude, keeping
#: its sign.
_DAVIDSON_SHIFT_FLOOR = 1e-8

#: A correction whose norm falls below this fraction of its norm before projection lies in the
#: search space already (up to roundoff) and is not added.
_DAVIDSON_INDEPENDENCE = 1e-10

#: The residual tolerance is floored at this many ulps of the operator's scale (its largest diagonal
#: magnitude): below that the residual is roundoff and no iteration can reach it.
_DAVIDSON_TOL_ULPS = 64


def _rank_offsets(n_local, comm):
    """``(offset, counts)``: this rank's first global row and every rank's row count (rank order).

    The global numbering is rank-contiguous, as :meth:`DistributedSparseMatrix.exchange_plan` assumes.
    """
    if comm is None:
        return 0, np.array([n_local], dtype=np.int64)
    counts = np.empty(comm.size, dtype=np.int64)
    comm.Allgather(np.array([n_local], dtype=np.int64), counts)
    return int(np.sum(counts[: comm.rank])), counts


def _local_diagonal(h_op, offset):
    """The diagonal entries of ``h_op``'s local columns, ``(n_local,)``; ``offset`` is the first local
    column's global index. Unbinds a bound matrix (row indices must be global)."""
    h_op.unbind_slots()
    diag = np.zeros(h_op.shape[1], dtype=complex)
    for c0, indptr, rows, data in h_op.segments:
        cols = np.repeat(np.arange(c0, c0 + len(indptr) - 1, dtype=np.int64), np.diff(indptr))
        on = np.asarray(rows, dtype=np.int64) == cols + offset
        np.add.at(diag, cols[on], np.asarray(data)[on])
    return diag.real


def _gather_sum(local, comm):
    """``Allreduce(SUM)`` of a small vector, bitwise identical on every rank.

    An Allgather followed by a sum in rank order (the idiom :mod:`TSQR` uses), so every rank holds the
    same bytes and every decision read from them -- converged, restart, independence, the Ritz
    problem itself -- comes out the same everywhere without a broadcast.
    """
    local = np.ascontiguousarray(local, dtype=complex).ravel()
    if comm is None:
        return local
    buf = np.empty(comm.size * local.size, dtype=complex)
    comm.Allgather(local, buf)
    return buf.reshape(comm.size, local.size).sum(axis=0)


def davidson_lowest(h_op, comm, start, *, tol, max_vectors=None, max_iter=None):
    """The lowest eigenpair of ``h_op`` by Davidson with the diagonal preconditioner: ``(theta, x, converged)``.

    ``start`` ``(n_local, b)`` spans the start space; the lowest Ritz vector is corrected by
    ``(theta - diag H)^{-1} r`` every iteration until ``||r|| < tol`` (floored at roundoff of the
    operator's scale). From a CIPSI cycle's warm start this took 2-4x fewer matrix-vector products
    than thick-restart Lanczos on every captured NiO d6 cycle.

    It stays in the symmetry sector its start's lowest Ritz vector is in -- which is why
    :func:`solve_lowest` uses it only where that is intended (a certified class expansion, held in its
    class by its penalty): a plain expansion relies on Lanczos growing every sector of the generic
    start vector to escape its seed's sector (a plain NiO d6 expansion stopped at -43.145 under
    Davidson, -43.275 under Lanczos).

    Per iteration: one matrix-vector product (with an exchange plan built once per call) and three
    small Allgathers -- the new column of ``V^H H V``, the projection of the correction together with
    its norm and the residual norm, and a second Gram-Schmidt pass with the norm by Pythagoras. The
    projected matrix grows by one column per iteration and is recombined, not recomputed, at a
    restart. Every reduction is summed in rank order (:func:`_gather_sum`), so every rank takes the
    same path through the collectives without broadcasts.

    Returns ``converged=False`` -- for :func:`solve_lowest` to continue with Lanczos -- when the
    iteration cap is reached, when the residual has not halved in :data:`DAVIDSON_STAGNATION`
    iterations, or when the correction lies in the search space already.
    """
    mpi = comm is not None and getattr(comm, "size", 1) > 1
    comm = comm if mpi else None
    h_op = DistributedSparseMatrix.coerce(h_op, comm)
    start = np.ascontiguousarray(start, dtype=complex)
    n_local = start.shape[0]
    offset, _counts = _rank_offsets(n_local, comm)
    diag = _local_diagonal(h_op, offset)
    scale = float(np.max(np.abs(diag))) if diag.size else 0.0
    if comm is not None:
        scale = comm.allreduce(scale, op=MPI.MAX)
    tol = max(float(tol), _DAVIDSON_TOL_ULPS * np.finfo(float).eps * scale)
    plan = h_op.exchange_plan(1, "davidson")  # after _local_diagonal, which unbinds

    max_vectors = max(int(DAVIDSON_MAX_VECTORS if max_vectors is None else max_vectors), 3)
    max_iter = int(DAVIDSON_MAX_ITER if max_iter is None else max_iter)
    # The space is stored by rows -- vector i is V[i], contiguous -- and V^H t is formed as
    # conj(V t*), which conjugates one vector instead of copying the space: with column storage the
    # skinny products and their `.conj().T` copies cost 2-3x the sparse product per iteration.
    V = np.zeros((max_vectors, n_local), dtype=complex)
    AV = np.zeros((max_vectors, n_local), dtype=complex)
    S = np.zeros((max_vectors, max_vectors), dtype=complex)
    m = 0

    def project(t):
        """``V^H t``, this rank's part."""
        return (V[:m] @ t.conj()).conj()

    def orthonormal(t, g):
        """``t`` minus its projection on the space ``V[:m]``, normalized, or ``None`` when nothing is left.

        ``g`` is the summed ``[V^H t, t^H t]`` of the first pass; the second pass (CGS2) gathers again
        and takes the norm by Pythagoras, so this costs one Allgather.
        """
        t = t - g[:m] @ V[:m]
        g2 = _gather_sum(np.append(project(t), np.vdot(t, t)), comm)
        t = t - g2[:m] @ V[:m]
        norm2 = g2[m].real - float(np.vdot(g2[:m], g2[:m]).real)
        if norm2 <= _DAVIDSON_INDEPENDENCE**2 * g[m].real:
            return None
        return t / np.sqrt(norm2)

    def append(t):
        """Add the orthonormal column ``t`` to the space: one product, one Allgather for its ``S`` column."""
        nonlocal m
        At = h_op.apply(t.reshape(-1, 1), plan)[:, 0]
        col = _gather_sum(np.append(project(At), np.vdot(t, At)), comm)
        V[m], AV[m] = t, At
        S[:m, m] = col[:m]
        S[m, :m] = col[:m].conj()
        S[m, m] = col[m].real
        m += 1

    for j in range(start.shape[1]):
        t = start[:, j]
        t = orthonormal(t, _gather_sum(np.append(project(t), np.vdot(t, t)), comm))
        if t is not None and m < max_vectors:
            append(t)

    if m == 0:  # the start block spans nothing
        return np.inf, V[:1].T.copy(), False
    y_prev = None
    best = np.inf
    best_at = 0
    theta, x = np.inf, V[:1].T.copy()
    for it in range(max_iter):
        thetas, Y = np.linalg.eigh(S[:m, :m])  # S is bitwise replicated, so is this
        y = Y[:, 0]
        theta = float(thetas[0])
        x = y @ V[:m]
        r = y @ AV[:m] - theta * x
        denom = theta - diag
        small = np.abs(denom) < _DAVIDSON_SHIFT_FLOOR
        t = r / np.where(small, np.where(denom < 0, -_DAVIDSON_SHIFT_FLOOR, _DAVIDSON_SHIFT_FLOOR), denom)
        g = _gather_sum(np.concatenate([project(t), [np.vdot(t, t), np.vdot(r, r)]]), comm)
        residual = np.sqrt(max(g[m + 1].real, 0.0))
        if residual < tol:
            return theta, x.reshape(-1, 1), True
        if residual < 0.5 * best:
            best, best_at = residual, it
        elif it - best_at >= DAVIDSON_STAGNATION:
            return theta, x.reshape(-1, 1), False
        if m == max_vectors:
            # Restart on the current and the previous Ritz vector; V, H V and S are recombined.
            keep = [y] if y_prev is None else [y, np.append(y_prev, np.zeros(m - len(y_prev)))]
            Q, _ = np.linalg.qr(np.column_stack(keep))
            k = Q.shape[1]
            V[:k], AV[:k] = Q.T @ V[:m], Q.T @ AV[:m]
            S_k = Q.conj().T @ S[:m, :m] @ Q
            S[:, :] = 0.0
            S[:k, :k] = 0.5 * (S_k + S_k.conj().T)
            g = np.concatenate([Q.conj().T @ g[:m], g[m:]])
            y = Q.conj().T @ y
            m = k
        y_prev = y
        t = orthonormal(t, g)
        if t is None:
            return theta, x.reshape(-1, 1), False
        append(t)
    return theta, x.reshape(-1, 1), False


def _lowest_on_root(h_op, comm, hashes, warm, solve):
    """:func:`solve_lowest` on rank 0 alone, for a basis too small to pay for distributed iterations.

    The operator (:func:`_gather_to_root`), the hashes and the warm block go to rank 0 with typed
    ``Gatherv``; ``solve(h, hashes, warm)`` runs there with no communicator, and each rank gets its own
    rows of the vector back by ``Scatterv``: three collectives per solve where a distributed Davidson
    iteration costs four. The result does not depend on the rank count.
    """
    h_op = DistributedSparseMatrix.coerce(h_op, comm)
    full = _gather_to_root(h_op, comm)
    n_local = len(hashes)
    _offset, counts = _rank_offsets(n_local, comm)
    displs = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    root = comm.rank == 0
    width = warm.shape[1]
    g_hashes = np.empty(int(counts.sum()), dtype=np.uint64) if root else None
    comm.Gatherv([np.ascontiguousarray(hashes, dtype=np.uint64), MPI.UINT64_T],
                 [g_hashes, counts, displs, MPI.UINT64_T] if root else None, root=0)  # fmt: skip
    g_warm = np.empty((int(counts.sum()), width), dtype=complex) if root else None
    if width:
        comm.Gatherv([np.ascontiguousarray(warm, dtype=complex), MPI.C_DOUBLE_COMPLEX],
                     [g_warm, counts * width, displs * width, MPI.C_DOUBLE_COMPLEX] if root else None, root=0)  # fmt: skip
    res = _on_root(comm, lambda: solve(DistributedSparseMatrix.from_scipy(full, None), g_hashes, g_warm))
    head = comm.bcast(
        None if res is None else (res.e, res.n_in, res.certified, res.max_multiplicity, res.status), root=0
    )
    e, n_in, certified, multiplicity, status = head
    k = len(e)
    vecs = np.empty((n_local, k), dtype=complex)
    send = np.ascontiguousarray(res.vecs, dtype=complex) if root else None
    comm.Scatterv([send, counts * k, displs * k, MPI.C_DOUBLE_COMPLEX] if root else None,
                  [vecs, MPI.C_DOUBLE_COMPLEX], root=0)  # fmt: skip
    return EigenResult(e, vecs, n_in, certified, multiplicity, status)


def solve_lowest(
    h_op, comm, hashes, *, warm, tol, n_global, max_restarts=100, verbose=False, davidson=False, serial=False
):
    """The lowest eigenpair of ``h_op``: one converged Ritz pair, no completeness claim.

    Count-mode thick-restart block Lanczos retaining a single pair. The start block is the ``warm``
    column (the previous lowest state, at most the first column of ``warm``) plus the all-positive
    :func:`cold_vector`, which has full support, so every charge sector stays reachable whatever the
    warm column spans. A degenerate ground level needs no special handling: any vector of its eigenspace
    converges, and the caller keeps the energy.

    Parameters
    ----------
    h_op
        ``DistributedSparseMatrix`` (or coercible) of the operator on the current basis.
    hashes : ndarray of uint64, ``(n_local,)``
        The routing hash of every locally owned determinant, in row order.
    warm : ndarray ``(n_local, w)`` or None
    tol : float
        Residual tolerance.
    n_global : int
        Dimension of the operator (bounds the Krylov depth).
    davidson : bool
        Try :func:`davidson_lowest` first (under ``GS_LOWEST_SOLVER=auto``).
    serial : bool
        Solve on rank 0 alone (:func:`_lowest_on_root`): for a basis so small that every distributed
        iteration is collective latency.

    Returns
    -------
    EigenResult
        One pair (none if the Krylov space is empty); ``certified`` means the pair reached ``tol``.
        Collective on ``comm``; ``e`` is replicated and ``vecs`` row-distributed.
    """
    mpi = comm is not None and getattr(comm, "size", 1) > 1
    n_local = len(hashes)
    warm = np.zeros((n_local, 0), dtype=complex) if warm is None else np.asarray(warm, dtype=complex)
    if mpi and serial:
        return _lowest_on_root(
            h_op,
            comm,
            hashes,
            warm[:, :1],
            lambda h, hs, w: solve_lowest(
                h, None, hs, warm=w, tol=tol, n_global=n_global, max_restarts=max_restarts, verbose=verbose,
                davidson=davidson,
            ),
        )  # fmt: skip

    def start_block(with_warm):
        cols = [warm[:, :1]] if with_warm else []
        cols.append(cold_vector(hashes))
        return np.ascontiguousarray(np.hstack(cols))

    with_warm = warm.shape[1] > 0
    try:
        q0, _ = block_normalize(start_block(with_warm), mpi, comm, 0.0)
    except ValueError:  # BlockBreakdown, rank-symmetric: decided by the replicated TSQR factor
        q0, _ = block_normalize(start_block(False), mpi, comm, 0.0)
    choice = config.GS_LOWEST_SOLVER.get()
    if choice not in config.GS_LOWEST_SOLVERS:  # a typo would otherwise silently mean Lanczos
        raise ValueError(f"GS_LOWEST_SOLVER={choice!r}; expected one of {config.GS_LOWEST_SOLVERS}")
    if (choice == "davidson" or (choice == "auto" and davidson)) and n_global > 1:
        theta, x, converged = davidson_lowest(h_op, comm, q0, tol=tol)
        if converged:
            return EigenResult(np.array([theta]), np.ascontiguousarray(x), 1, True, 1, "converged")
        # Not converged within its iteration cap: thick-restart Lanczos from where it got to.
        q0, _ = block_normalize(np.hstack([x, cold_vector(hashes)]), mpi, comm, 0.0)
    res = trlm(
        q0,
        h_op,
        comm,
        num_wanted=1,
        max_subspace_blocks=gs_krylov_blocks(n_global, q0.shape[1], 1),
        tol=tol,
        max_restarts=max_restarts,
        verbose=verbose,
    )
    n = min(1, len(res.eigvals))
    return EigenResult(
        res.eigvals[:n],
        np.ascontiguousarray(res.eigvecs[:, :n]),
        n,
        bool(n and res.status in STATUS_CONVERGED),
        1,
        res.status,
    )
