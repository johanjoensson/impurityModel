"""Thick-restart block Lanczos (TRLM) on distributed dense blocks.

One implementation, written against the ``block_*`` primitives of
:mod:`impurityModel.ed.BlockLanczosCore`. A block is an ``(n_local, k)`` complex array whose rows are
distributed over ``comm``; the operator is a ``DistributedSparseMatrix`` (or anything
:meth:`DistributedSparseMatrix.coerce` accepts).

Two stopping rules share one restart loop:

* **count mode** (``window=None``): the ``num_wanted`` lowest Ritz pairs, ``num_converge`` of which
  must reach ``tol``;
* **window mode** (``window=Window(...)``): *every* Ritz pair inside an energy window, plus one
  bracket pair just above it. The number of retained vectors follows the window -- it grows when
  the window turns out to hold more states than were being kept -- so the caller never has to
  guess a multiplicity beforehand.

Collective on ``comm``. Every branch is decided from replicated data (the Allreduced ``T`` and a
root-broadcast ``eigh``), so all ranks leave the loop together.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.linalg as sp

from impurityModel.ed.BlockLanczosCore import (
    BREAKDOWN_TOL,
    DEFAULT_EIGEN_TOL,
    _build_full_T,
    block_apply,
    block_cols,
    block_combine,
    block_inner,
    block_normalize,
    block_orthogonalize,
    eigh_block_tridiagonal,
)
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.solver_trace import note as _trace_note
from impurityModel.ed.solver_trace import timed as _trace_timed

__all__ = ["TRLMResult", "Window", "thick_restart_block_lanczos", "trlm"]

#: Continuation blocks between mid-restart convergence checks. One check is a dense ``eigh`` of the
#: current ``T``; one block is a matvec plus two CGS passes against the whole basis. Checking every
#: block costs more than it saves, never checking wastes the tail of every restart; the minimum is a
#: broad plateau around 8-12 (``doc/plans/gs_redesign.md``).
CONTINUATION_CHECK_EVERY = 8

#: Restarts of no real progress before the loop gives up. The residual is not monotone across
#: restarts, so progress is judged on the best value seen so far.
STAGNATION_PATIENCE = 20
#: Factor the best-so-far residual must improve by within one patience window to count as progress.
STAGNATION_FACTOR = 10.0

#: Exit statuses of :func:`trlm`. ``"ceiling"``, ``"stagnated"`` and ``"restart_cap"`` mean the
#: requested accuracy (and, in window mode, the completeness of the window) was *not* reached.
STATUS_CONVERGED = (
    "converged",
    "continuation_converged",
    "invariant_subspace_sweep",
    "invariant_subspace_continuation",
)


def check_width_sync(Q, widths, where, exact=False):
    """Assert the block-width table and the stored Krylov basis agree.

    The restart loop indexes ``T`` (sized from ``widths``) and ``Q`` by the *same* cumulative offsets, so
    ``sum(widths)`` must never exceed ``Q``'s column count. A sweep ends either with a trailing residual
    block appended but not counted (``sum(widths) == cols - w_last``) or with the last block counted and
    no residual stored (``sum(widths) == cols``) -- never with a counted block whose vectors were never
    stored. ``exact=True`` where the residual block has already been split off.

    Raises
    ------
    RuntimeError
        If the invariant is violated.
    """
    cols = block_cols(Q)
    total = int(sum(widths)) if widths is not None else 0
    if total > cols:
        raise RuntimeError(
            f"{where}: block widths sum to {total} but the Krylov basis holds only {cols} "
            f"columns ({widths!r}). The recurrence counted a block whose vectors were never stored."
        )
    if exact and total != cols:
        raise RuntimeError(
            f"{where}: block widths sum to {total} but the Krylov basis holds {cols} columns ({widths!r}). "
            "The residual block has already been split off here, so the stored columns must be exactly "
            "the ones T is expressed in."
        )


def width_synced_total(Q_basis, widths, m_act, p, where, exact=False):
    """:func:`check_width_sync`, then the true (possibly deflation-shrunk) subspace dimension ``sum(widths)``."""
    check_width_sync(Q_basis, widths, where, exact=exact)
    return int(sum(widths)) if widths is not None else m_act * p


def trim_trailing_beta(betas, m_act):
    """Drop the trailing residual-block coupling from ``betas`` (a running sweep has one entry per step,
    including the not-yet-accepted residual); a sweep that already split it off is returned unchanged."""
    return betas[: m_act - 1] if len(betas) == m_act else betas


class KrylovColumnStore:
    """Append-only column store for the Krylov basis of one TRLM restart.

    A preallocated C-ordered buffer whose filled prefix is handed out as a view: numpy passes the strided
    view to BLAS through its leading dimension, so products against it are bit-identical to ones against a
    concatenated copy, and appending a block copies that block and nothing else. The buffer is kept across
    restarts and reallocated only when a restart asks for more columns than it holds. :attr:`columns` taken
    before :meth:`reset` is overwritten by the next restart's columns, so callers drop it first.
    """

    def __init__(self):
        self._buf = None
        self._filled = 0

    def reset(self, like, capacity, row_hint=None):
        """Empty the store, with room for ``capacity`` columns shaped like the block ``like``."""
        self._filled = 0
        n_rows = like.shape[0]
        if self._buf is None or self._buf.shape[0] != n_rows or self._buf.shape[1] < capacity:
            self._buf = None  # free the old buffer before allocating the new one
            self._buf = np.empty((n_rows, capacity), dtype=complex)

    def append(self, block):
        """Copy the columns of ``block`` in after the ones already stored."""
        w = block.shape[1]
        if self._filled + w > self._buf.shape[1]:
            grown = np.empty((self._buf.shape[0], max(2 * self._buf.shape[1], self._filled + w)), dtype=complex)
            grown[:, : self._filled] = self._buf[:, : self._filled]
            self._buf = grown
        self._buf[:, self._filled : self._filled + w] = block
        self._filled += w

    @property
    def columns(self):
        """A view of the filled prefix."""
        return self._buf[:, : self._filled]

    def project(self, wp, mpi=False, comm=None):
        """One classical Gram-Schmidt pass of ``wp`` against the stored columns.

        Returns ``(wp - Q O, O)`` with ``O = Q^H wp`` taken *before* the pass and summed over ranks, so the
        first of two passes also yields the Lanczos ``alpha`` (its last rows). Collective on ``comm``: one
        ``Allreduce`` of the ``(n_cols, p)`` overlaps, on every rank.
        """
        Q = self.columns
        overlaps = block_inner(Q, wp, mpi, comm)
        wp, _ = block_orthogonalize(wp, Q, overlaps=overlaps, mpi=mpi, comm=comm)
        return wp, overlaps


class StagnationMonitor:
    """Windowed no-progress detector for the restart loop.

    The residual is not monotone across restarts, so the test is on the best value seen so far,
    compared across a window of ``patience`` restarts.
    """

    def __init__(self, patience=None, factor=None):
        # Read at construction, not as default arguments, so a test can monkeypatch the module-level
        # calibration.
        self.patience = STAGNATION_PATIENCE if patience is None else int(patience)
        self.factor = STAGNATION_FACTOR if factor is None else float(factor)
        self.best = float("inf")
        self._window_ref = float("inf")
        self._window_start = None

    def update(self, restart, residual):
        """Record one restart; return True when the loop has stopped making progress."""
        residual = float(residual)
        self.best = min(self.best, residual)
        if self._window_start is None:
            self._window_ref = residual
            self._window_start = restart
            return False
        if restart - self._window_start < self.patience:
            return False
        advanced = self.best <= self._window_ref / self.factor
        self._window_ref = self.best
        self._window_start = restart
        return not advanced


@dataclass(frozen=True)
class Window:
    """Energy window ``[ref, ref + max_energy]`` of Ritz values that must be returned whole.

    ``ref`` is the lowest Ritz value, or ``e0`` when the caller knows the absolute origin (a solve
    in the complement of locked vectors, whose own lowest Ritz value is not the spectrum's).
    A degenerate group (gaps ``<= deg_tol``) straddling the upper edge is absorbed whole, never
    bisected. ``bracket_fraction``: the first Ritz pair above the window counts as converged once
    its residual is below this fraction of its distance to the window edge.
    """

    max_energy: float
    deg_tol: float
    e0: float | None = None
    bracket_fraction: float = 0.1

    def split(self, theta):
        """``(n_in, top)`` for ascending ``theta``: the number of Ritz values inside the window and its upper edge."""
        ref = theta[0] if self.e0 is None else self.e0
        edge = ref + self.max_energy
        n_in = int(np.count_nonzero(theta <= edge))
        while 0 < n_in < len(theta) and theta[n_in] - theta[n_in - 1] <= self.deg_tol:
            n_in += 1
        return n_in, (max(edge, float(theta[n_in - 1])) if n_in else edge)


class TRLMResult(NamedTuple):
    """``eigvals`` ascending, ``eigvecs`` ``(n_local, len(eigvals))``, exit ``status``, and ``n_in``.

    ``n_in`` is the number of returned pairs inside the window (``None`` in count mode); the one
    pair after them is the bracket. In window mode ``status in STATUS_CONVERGED`` certifies that the
    in-window pairs reached ``tol`` and the bracket sits above them.
    """

    eigvals: np.ndarray
    eigvecs: np.ndarray
    status: str
    n_in: int | None


def _assess(theta, res, tol, window, n_converge):
    """``(metric, n_ret, n_in)``: the quantity the loop drives below ``tol``, and what to return.

    Count mode: the largest residual of the ``n_converge`` lowest pairs. Window mode: the largest
    in-window residual, and the bracket pair's residual rescaled so that "bracket converged" is
    ``metric < tol`` too (infinite when no pair lies above the window yet).
    """
    if window is None:
        return float(np.max(res[:n_converge])), None, None
    n_in, top = window.split(theta)
    if n_in >= len(theta):
        return float("inf"), n_in, n_in
    r_in = float(np.max(res[:n_in])) if n_in else 0.0
    gap = float(theta[n_in]) - top
    threshold = max(tol, window.bracket_fraction * gap)
    return max(r_in, float(res[n_in]) * tol / threshold), n_in + 1, n_in


def _extract(T, Q, dim, num_wanted, comm, window):
    """Ritz pairs of the leading ``dim`` x ``dim`` block of ``T`` formed from ``Q[:, :dim]``.

    ``num_wanted`` lowest pairs (count mode), or the window plus its bracket. The ``eigh`` runs on
    rank 0 and is broadcast so every rank forms identical Ritz vectors.
    """
    theta, S = sp.eigh(T[:dim, :dim])
    if comm is not None:
        theta = comm.bcast(theta, root=0)
        S = comm.bcast(S, root=0)
    if window is None:
        n_ret, n_in = num_wanted, None
    else:
        n_in, _ = window.split(theta)
        n_ret = min(len(theta), n_in + 1)
    wanted = np.argsort(theta)[:n_ret]
    return theta[wanted], block_combine(Q[:, 0:dim], S[:, wanted], 0.0), n_in


def _extract_invariant_subspace(alphas, betas_off, widths, Q_basis, num_wanted, comm, window, m_actual, m, q_m):
    """Terminal early exit for the initial sweep when the block-Krylov space is **closed**.

    Closed means the sweep returned fewer blocks than asked (``m_actual < m``) or stored no trailing
    residual block (``q_m is None``). Its Ritz pairs are then exact, ``T`` is block-tridiagonal and
    the banded solver suffices. **Block deflation is not closure**: a rank-deficient residual
    narrows the block but leaves a live residual block, and the restart loop is what converges the
    pairs then. Returns the result, or ``None`` when the restart loop should run.
    """
    if not (m_actual < m or q_m is None):
        return None
    theta, S = eigh_block_tridiagonal(alphas, betas_off, block_widths=widths)
    if comm is not None:
        theta = comm.bcast(theta, root=0)
        S = comm.bcast(S, root=0)
    if window is None:
        n_ret, n_in = num_wanted, None
    else:
        n_in, _ = window.split(theta)
        n_ret = min(len(theta), n_in + 1)
    wanted = np.argsort(theta)[:n_ret]
    return TRLMResult(theta[wanted], block_combine(Q_basis, S[:, wanted], 0.0), "invariant_subspace_sweep", n_in)


def _deflate(wp, locked, mpi, comm):
    """Project ``wp`` out of ``span(locked)`` (two CGS passes); the identity when ``locked`` is None.

    Runs *last*, after the passes against the stored basis and right before normalization: the
    stored columns carry an O(eps) locked component that those passes put back into ``wp``, and
    normalization amplifies it. Collective on ``comm`` whenever ``locked`` is given, which is a
    rank-invariant argument.
    """
    if locked is not None:
        for _ in range(2):
            wp, _ = block_orthogonalize(wp, locked, mpi=mpi, comm=comm)
    return wp


def trlm(
    psi0,
    h_op,
    comm,
    *,
    num_wanted,
    max_subspace_blocks,
    tol=DEFAULT_EIGEN_TOL,
    max_restarts=100,
    verbose=False,
    num_converge=None,
    locked=None,
    window=None,
    max_keep=None,
):
    """Thick-restart block Lanczos for the lowest eigenpairs of ``h_op``.

    Runs one block-Lanczos sweep of ``m`` blocks, diagonalizes the block-tridiagonal ``T``, keeps
    the lowest ``nkeep = k_blocks * p`` Ritz pairs as one retained super-block, hangs the residual
    block on it as a thick-restart spike, and continues the recurrence from block ``k_blocks``
    until the stopping rule is met or ``max_restarts`` is exhausted. Every new vector is fully
    reorthogonalized against the stored basis (two CGS passes), so the textbook thick-restart
    coefficients ``diag(theta_keep)`` and ``beta_res @ Y_last`` are exact up to roundoff.

    Parameters
    ----------
    psi0 : ndarray (n_local, p)
        Start block; projected out of ``locked`` and orthonormalized here.
    h_op
        The operator, ``DistributedSparseMatrix`` or coercible.
    comm
        ``mpi4py`` communicator or ``None``.
    num_wanted : int
        Count mode: pairs returned. Window mode: the initial number of retained Ritz vectors
        (``nkeep`` grows from it as the window demands).
    max_subspace_blocks : int
        Krylov depth ``m`` in blocks; must exceed ``ceil(num_wanted / p)``.
    num_converge : int, optional
        Count mode: how many of the lowest pairs must reach ``tol`` (default all).
    locked : ndarray (n_local, L), optional
        Orthonormal columns; the solve is restricted to their orthogonal complement.
    window : Window, optional
        Window mode, see the module docstring.
    max_keep : int, optional
        Window mode: ceiling on retained vectors. Reaching it ends the solve with status
        ``"ceiling"`` -- the window is then *not* certified whole.

    Returns
    -------
    TRLMResult
        May hold fewer pairs than requested when the space closes on an invariant subspace.
        Callers must use ``len(eigvals)``.
    """
    from impurityModel.ed.BlockLanczosCore import block_lanczos

    mpi = comm is not None and getattr(comm, "size", 1) > 1
    rank0 = (not mpi) or comm.rank == 0
    h_op = DistributedSparseMatrix.coerce(h_op, comm)

    psi0 = np.ascontiguousarray(psi0 if psi0.ndim == 2 else np.reshape(psi0, (-1, 1)), dtype=complex)
    if locked is not None:
        locked = np.ascontiguousarray(locked, dtype=complex)
        psi0 = _deflate(psi0, locked, mpi, comm)
    psi0, _ = block_normalize(psi0, mpi, comm, 0.0)

    p = psi0.shape[1]
    k_blocks = int(np.ceil(num_wanted / p))
    m = max_subspace_blocks
    if m <= k_blocks:
        raise ValueError("max_subspace_blocks must be strictly greater than ceil(num_wanted / p).")
    continuation = m - k_blocks
    n_converge = num_wanted if num_converge is None else max(1, min(int(num_converge), num_wanted))

    # --- Initial sweep ---------------------------------------------------
    alphas, betas, Q_basis, widths, _status = block_lanczos(
        psi0=psi0,
        h_op=h_op,
        converged=lambda a, b, **kw: False,
        max_iter=m,
        verbose=verbose,
        reort=True,
        return_widths=True,
        return_status=True,
        comm=comm,
        locked=locked,
    )
    m_actual = len(alphas)
    if m_actual == 0:
        raise RuntimeError("Block Lanczos produced zero iterations.")
    # The sweep can shrink blocks (rank-deficient beta -> deflation), so the subspace dimension is
    # sum(widths), not m_actual * p.
    total = width_synced_total(Q_basis, widths, m_actual, p, "TRLM initial sweep")
    if verbose and rank0 and total < m_actual * p:
        print(f"[TRLM] Sweep deflated to dim {total} (< {m_actual * p}); continuing to the restart loop.")
    q_m = Q_basis[:, total:].copy() if block_cols(Q_basis) > total else None
    Q_basis = Q_basis[:, 0:total]
    betas_off = trim_trailing_beta(betas, m_actual)

    closed = _extract_invariant_subspace(alphas, betas_off, widths, Q_basis, num_wanted, comm, window, m_actual, m, q_m)
    if closed is not None:
        return closed

    # The thick restart builds an arrowhead T (a spike couples the retained block to the residual),
    # which is not banded, so this path keeps the dense T_full.
    T_full = _build_full_T(alphas, betas_off, block_widths=widths)
    nkeep = k_blocks * p
    p_resid = block_cols(q_m)
    # betas[-1] is the trailing coupling, padded to (p, p) by the kernel; its real shape is
    # (width of the block it couples to, width of the block it couples from).
    beta_res = betas[len(betas) - 1][:p_resid, : widths[len(widths) - 1]]
    cur_widths = list(widths)

    monitor = StagnationMonitor()
    stagnated = False
    done = False
    status = "restart_cap"
    n_in = None
    store = KrylovColumnStore()
    row_hint = psi0.shape[0]

    for restart in range(max_restarts):
        check_width_sync(Q_basis, cur_widths, f"TRLM restart {restart}", exact=True)
        D = int(sum(cur_widths))
        p_last = cur_widths[len(cur_widths) - 1]
        theta, S = sp.eigh(T_full[:D, :D])
        if comm is not None:
            theta = comm.bcast(theta, root=0)
            S = comm.bcast(S, root=0)
        res = np.linalg.norm(beta_res @ S[D - p_last : D, :], axis=0)
        metric, _n_ret, n_in = _assess(theta, res, tol, window, n_converge)

        # Stop threshold for a residual block: ||beta|| < tol (every Ritz residual is then under tol)
        # or numerically zero against the operator scale (a genuine invariant subspace).
        t_scale = float(np.max(np.abs(theta))) if theta.size else 1.0
        stop_beta = max(float(tol), BREAKDOWN_TOL * t_scale)

        if verbose and rank0:
            print(f"[TRLM] Restart {restart:3d} | MinEigval={theta[0]:.6f} | Metric={metric:.2e} | n_in={n_in}")

        done = metric < tol
        stagnated = monitor.update(restart, metric)
        if mpi:
            # One collective for both decisions on every rank: a rank-local disagreement would send
            # two ranks out of a collective loop body through different breaks.
            done, stagnated = comm.bcast((done, stagnated), root=0)
        if done:
            status = "converged"
            break
        if stagnated:
            status = "stagnated"
            break

        if window is not None:
            # The window holds more states than are being retained: retain more. T_full and the
            # store are reallocated every restart, so growing costs nothing already built.
            want = n_in + 1 + p
            if want > nkeep:
                new_blocks = int(np.ceil(want / p))
                if max_keep is not None and new_blocks * p > max_keep:
                    status = "ceiling"
                    break
                k_blocks, nkeep = new_blocks, new_blocks * p
                m = max(m, k_blocks + continuation)
                monitor = StagnationMonitor()

        keep = np.argsort(theta)[:nkeep]
        Y_k = S[:, keep]
        Y_last = Y_k[D - p_last : D, :]
        # The retained block X = Q Y inherits Q's orthonormality; normalizing it only removes the
        # O(eps) drift. It cannot lose rank (Q is orthonormal, Y has orthonormal columns).
        Q_ret, _ = block_normalize(block_combine(Q_basis[:, 0:D], Y_k, 0.0), mpi, comm, 0.0)
        k_ret = block_cols(Q_ret)
        if k_ret != len(keep):
            raise RuntimeError(f"TRLM: retained Ritz block lost rank ({k_ret} of {len(keep)}) at restart {restart}.")
        T_lead = np.asarray(np.diag(theta[keep]), dtype=complex)
        cross = beta_res @ Y_last  # (p_resid, k_ret)
        # The carried-over residual block q_m is already the whole residual: (I - P) H Q_ret =
        # q_m beta_res Y_last has rank <= p_resid.

        dim = k_ret + (m - k_blocks) * p_resid
        _trace_note(
            "trlm_restart", restart=int(restart), k_ret=int(k_ret), p_resid=int(p_resid), dim=int(dim), n_in=n_in
        )
        T_full = np.zeros((dim, dim), dtype=complex)
        T_full[:k_ret, :k_ret] = T_lead

        # The old basis is dead (Q_ret is a fresh array) and may be a view of the store's buffer,
        # which reset overwrites: drop it first so a growing reallocation does not hold two buffers.
        Q_basis = None
        store.reset(Q_ret, dim, row_hint=row_hint)
        store.append(Q_ret)
        store.append(q_m)
        Q_basis = store.columns
        T_full[k_ret : k_ret + p_resid, :k_ret] = cross
        T_full[:k_ret, k_ret : k_ret + p_resid] = np.conj(cross.T)

        cur_widths = [k_ret, p_resid]
        off = k_ret
        w1 = p_resid
        q1 = q_m
        q_m = None

        for i in range(k_blocks, m):
            with _trace_timed("block_apply", site="continuation", w=block_cols(q1)):
                wp = block_apply(h_op, q1, None, mpi, 0.0)
            # CGS2 against the stored basis. The first pass's overlaps, taken before it projects,
            # are Q^H H q1, so their last w1 rows are alpha_i.
            wp, overlaps = store.project(wp, mpi, comm)
            T_full[off : off + w1, off : off + w1] = overlaps[overlaps.shape[0] - w1 :, :]
            wp, _ = store.project(wp, mpi, comm)
            wp = _deflate(wp, locked, mpi, comm)

            try:
                q_next, beta_i = block_normalize(wp, mpi, comm, 0.0)
            except (sp.LinAlgError, ValueError):
                q_next = None

            if q_next is None or np.linalg.norm(beta_i, ord=2) <= stop_beta:
                if verbose and rank0:
                    print(f"[TRLM] Invariant subspace found during restart at block {i}.")
                vals, vecs, n_in = _extract(T_full, Q_basis, off + w1, num_wanted, comm, window)
                return TRLMResult(vals, vecs, "invariant_subspace_continuation", n_in)

            D_now = off + w1
            if i < m - 1 and D_now > nkeep and (i - k_blocks + 1) % CONTINUATION_CHECK_EVERY == 0:
                theta_now, s_now = sp.eigh(T_full[:D_now, :D_now])
                res_now = np.linalg.norm(beta_i @ s_now[D_now - w1 : D_now, :], axis=0)
                metric_now, _, _ = _assess(theta_now, res_now, tol, window, n_converge)
                done_now = metric_now < tol
                # `eigh` is a rank-local LAPACK call on a replicated T_full and a threaded BLAS can
                # make its last bits differ between ranks: decide on rank 0 and broadcast.
                if mpi:
                    done_now = comm.bcast(done_now, root=0)
                if done_now:
                    vals, vecs, n_in = _extract(T_full, Q_basis, D_now, num_wanted, comm, window)
                    return TRLMResult(vals, vecs, "continuation_converged", n_in)

            w_next = block_cols(q_next)
            if i < m - 1:
                T_full[off + w1 : off + w1 + w_next, off : off + w1] = beta_i
                T_full[off : off + w1, off + w1 : off + w1 + w_next] = np.conj(beta_i.T)
                store.append(q_next)
                Q_basis = store.columns
                cur_widths.append(w_next)
                off += w1
                w1 = w_next
                q1 = q_next
            else:
                beta_res = beta_i
                q_m = q_next
                p_resid = w_next

    vals, vecs, n_in = _extract(T_full, Q_basis, int(sum(cur_widths)), num_wanted, comm, window)
    if verbose and rank0:
        print(f"[TRLM] Final eigvals:\n{vals}")
    return TRLMResult(vals, vecs, status, n_in)


def thick_restart_block_lanczos(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = DEFAULT_EIGEN_TOL,
    max_restarts: int = 100,
    verbose: bool = False,
    slaterWeightMin: float = 0,
    num_converge=None,
    locked=None,
):
    """Count-mode :func:`trlm` returning ``(eigvals, eigvecs)``; ``comm`` is read off ``basis``."""
    comm = getattr(basis, "comm", None)
    res = trlm(
        psi0,
        h_op,
        comm,
        num_wanted=num_wanted,
        max_subspace_blocks=max_subspace_blocks,
        tol=tol,
        max_restarts=max_restarts,
        verbose=verbose,
        num_converge=num_converge,
        locked=locked,
    )
    return res.eigvals, res.eigvecs


def _thick_restart_block_lanczos_array(
    psi0, h_op, basis, num_wanted, max_subspace_blocks, tol, max_restarts, verbose, comm, num_converge=None, locked=None
):
    """Count-mode :func:`trlm` with an explicit ``comm``, returning ``(eigvals, eigvecs)``."""
    res = trlm(
        psi0,
        h_op,
        comm,
        num_wanted=num_wanted,
        max_subspace_blocks=max_subspace_blocks,
        tol=tol,
        max_restarts=max_restarts,
        verbose=verbose,
        num_converge=num_converge,
        locked=locked,
    )
    return res.eigvals, res.eigvecs
