"""
Low-spectrum eigensolvers of the ground-state path.

* :func:`solve_window` -- every eigenstate inside an energy window, certified complete, from the
  thick-restart block Lanczos solver (:mod:`impurityModel.ed.trlm`). Used above ``dense_cutoff``.
* :func:`eigensystem` -- dense diagonalization, complete by construction, below it.

Both answer the same question, "all states within ``max_energy`` of the lowest", and the
CIPSI solver cuts the result with one rule (``_energy_cut_indices``).
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed.BlockLanczosCore import block_inner, block_normalize
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


@dataclass
class EigenResult:
    """The low spectrum of one operator on one basis.

    ``e`` ascending and ``vecs`` ``(n_local, len(e))``: the states inside the window followed by
    one bracket state just above it (``n_in`` counts the former). ``certified`` is True when
    every in-window level is shown complete (see :func:`solve_window`). ``max_multiplicity`` is
    the largest degenerate group in the window. ``generation`` is the ``Basis.generation`` the
    result belongs to and ``window`` the ``max_energy`` it is complete for; a consumer re-solves
    when the generation no longer matches or it needs a wider window.
    """

    e: np.ndarray
    vecs: np.ndarray
    n_in: int
    certified: bool
    max_multiplicity: int
    status: str
    generation: int | None = None
    window: float | None = None


#: Singular values of ``F_G^H W`` below this are not counted as warm overlap with a level. ``F`` is
#: only ``tol``-accurate, so the product carries O(tol) mixing from neighbouring levels; counting
#: that noise would inflate the reachable-copies estimate and certify a level with missing copies,
#: while undercounting only triggers the (safe) fallback.
_OVERLAP_RANK_FLOOR = 1e-6
#: Fallback rounds before a still-saturating level is reported uncertified.
_MAX_FALLBACK_ROUNDS = 3


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
    if hasattr(h_local, "toarray"):
        h = h_local.toarray()
    elif hasattr(h_local, "todense"):
        h = h_local.todense()
    else:
        h = h_local
    if comm is not None:
        comm.Reduce(h if rank != 0 else MPI.IN_PLACE, h, root=0, op=MPI.SUM)
    if return_eigvecs:
        if rank == 0:
            es, vecs = np.linalg.eigh(h, UPLO="L")
        else:
            es = np.empty((h.shape[0]), dtype=float, order="C")
            vecs = np.empty(h.shape, dtype=h.dtype, order="C")
    else:
        es = np.linalg.eigvalsh(h, UPLO="L") if rank == 0 else np.empty(h.shape[0], dtype=float)
    if comm is not None:
        comm.Bcast(es, root=0)
        if return_eigvecs:
            comm.Bcast(vecs, root=0)
    if return_eigvecs:
        return es, vecs
    return es


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
    """``rank(F_G^H W)`` for every group ``G`` of columns of ``F``: how many of the level's copies the warm block reaches."""
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
        k_blocks = int(np.ceil(wanted / width))
        return max(k_blocks + 1, min(4 * k_blocks + 20, max(2, n_global // width - 1)))

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
