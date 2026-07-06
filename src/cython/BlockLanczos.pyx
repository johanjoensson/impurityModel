# distutils: language = c++
# cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, freethreading_compatible=True, cdivision=True, cpow=True
"""
BlockLanczos.pyx
================
Parallel Block Lanczos eigensolver implemented in Cython.

This module provides:

* ``block_lanczos_cy`` – core iteration loop producing the block-tridiagonal
  Lanczos representation :math:`T = Q^\\dagger H Q` of the Hamiltonian.
* ``thick_restart_block_lanczos_cy`` – thick-restart (TRLM) wrapper that
  restarts the Krylov subspace while retaining the best Ritz pairs.
* ``implicitly_restarted_block_lanczos_cy`` – implicitly-restarted (IRLM)
  wrapper that applies :math:`(m-k)` implicit QR shifts to compress the
  subspace back to :math:`k` blocks before continuing.

All distributed inner products use ``mpi4py``'s Python API
(``comm.Allreduce``) over small :math:`p \\times p` matrices.  Heavy
matvec work is delegated to ``ManyBodyOperator.apply_multi`` which
releases the GIL internally.

Notes
-----
SlaterDeterminant distribution:
    Each rank owns the SDs with ``hash(sd) % mpi_size == rank``.
    This is maintained by ``basis.redistribute_psis()`` after every
    ``apply_multi`` call.

Pre-allocation:
    ``alphas`` and ``betas`` arrays are pre-allocated before the loop
    and sliced at return time; no numpy allocation occurs inside the
    Lanczos iteration body.

Reorthogonalization modes (``Reort`` enum from ``lanczos.py``):

* ``NONE``      – no reorthogonalization.
* ``PARTIAL``   – Paige-Simon W-matrix estimator; reorthogonalize only
  when the estimated overlap exceeds :math:`\\sqrt{\\varepsilon}`.
* ``SELECTIVE`` – as PARTIAL but additionally projects against converged
  Ritz vectors.
* ``FULL``      – full Gram-Schmidt against all previous blocks (2
  passes).
* ``PERIODIC``  – full reorthogonalization every ``reort_period`` steps.
"""

import math

import numpy as np
import scipy.linalg as sp
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    add_scaled_multi,
    inner_multi,
    SparseKrylovDense,
)
from mpi4py import MPI

cimport numpy as np

from impurityModel.ed.BlockLanczosArray import estimate_orthonormality, _build_full_T, _cholesky_or_deflate, _cholesky_qr2, eigh_block_tridiagonal
from impurityModel.ed.BlockLanczosArray import (
    apply_reort,
    divergence_guard,
    resolve_reort,
    selective_orthogonalize,
    is_array,
    block_apply,
    block_combine,
    block_inner,
    block_orthogonalize,
    block_normalize,
    Reort,
    EPS,
    REORT_TOL,
    BAD_BLOCK_TOL,
)

# --- Optional per-step profiling (env-gated, ~zero cost when off) -------------------
# Set BLOCKLANCZOS_PROFILE=1 to accumulate wall time per sub-operation of the sparse
# block-Lanczos step (matvec / recurrence-LA / W-estimator / triggered reort /
# CholeskyQR2 / convergence monitor). Read with get_block_lanczos_profile().
import os as _os
import time as _time
_PROF = {}
_PROF_ON = _os.environ.get("BLOCKLANCZOS_PROFILE") == "1"


def get_block_lanczos_profile():
    """Return a copy of the accumulated per-operation timings (seconds) and call counts."""
    return dict(_PROF)


def reset_block_lanczos_profile():
    _PROF.clear()


cdef inline void _prof_acc(str key, double t0):
    if _PROF_ON:
        _PROF[key] = _PROF.get(key, 0.0) + (_time.perf_counter() - t0)
        _PROF[key + "#n"] = _PROF.get(key + "#n", 0.0) + 1.0


cpdef list block_combine_sparse(list Q, np.ndarray Y, double slaterWeightMin=0.0):
    cdef int n_out = Y.shape[1]
    cdef list out = [ManyBodyState() for _ in range(n_out)]
    add_scaled_multi(out, Q, np.ascontiguousarray(Y, dtype=complex))
    if slaterWeightMin > 0:
        for st in out:
            st.prune(slaterWeightMin)
    return out


cpdef tuple block_orthogonalize_sparse(list wp, list Q, object overlaps=None, object comm=None):
    if overlaps is None:
        overlaps = inner_multi(Q, wp)
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
    add_scaled_multi(wp, Q, -overlaps)
    return wp, overlaps


cpdef tuple block_normalize_sparse(list wp, bint mpi=False, object comm=None, double slaterWeightMin=0.0):
    cdef np.ndarray M = inner_multi(wp, wp)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
    cdef int n = M.shape[0]
    cdef np.ndarray beta_j, beta_inv
    cdef int active_k
    beta_j, beta_inv, active_k = _cholesky_or_deflate(M, n)
    if active_k == 0:
        raise ValueError("Block collapsed to zero rank")
    cdef list q_next = [ManyBodyState() for _ in range(active_k)]
    add_scaled_multi(q_next, wp, beta_inv)
    if slaterWeightMin > 0:
        for st in q_next:
            st.prune(slaterWeightMin)
    return q_next, beta_j

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _block_inner_mpi(states_a, states_b, mpi: bool, comm):
    """Compute the block inner product :math:`G = \\langle A | B \\rangle` with MPI reduction.

    Computes the :math:`p \\times p` Gram matrix

    .. math::

        G_{ij} = \\langle a_i \\mid b_j \\rangle, \\quad i, j = 0, \\dots, p-1

    using a *zero-copy* local pass via ``inner_multi`` (which sums contributions from
    the Slater determinants owned by this rank), then performs a single ``MPI_Allreduce``
    (``MPI.SUM``) so that every rank holds the identical, globally reduced :math:`G`.

    Note:
        **Collective operation** – all MPI ranks must call this function simultaneously
        whenever ``mpi=True``.  The inner computation itself (``inner_multi``) is local
        (no communication); only the subsequent ``Allreduce`` is collective.

    Args:
        states_a: List of ``ManyBodyState`` of length ``p`` representing the bra block
            :math:`A = [a_0, \\dots, a_{p-1}]`.
        states_b: List of ``ManyBodyState`` of length ``p`` representing the ket block
            :math:`B = [b_0, \\dots, b_{p-1}]`.
        mpi: ``True`` when running under MPI with more than one rank.  When ``False`` no
            communication occurs and the local result is returned directly.
        comm: Active ``mpi4py.MPI.Comm`` communicator, or ``None`` when running serially.

    Returns:
        numpy.ndarray: Complex array of shape ``(p, p)`` holding the globally reduced
        Gram matrix :math:`G`.
    """
    G = inner_multi(states_a, states_b)
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, G, op=MPI.SUM)
    return G


# _cholesky_or_deflate is shared: imported from BlockLanczosArray (identical logic,
# single source of the EA16 shrinking-block deflation policy).


def block_lanczos_step_cy(
    h_op,
    q_prev,
    q_curr,
    Q_basis,
    alphas,
    betas,
    it: int,
    reort_mode,
    W,
    mpi: bool,
    comm,
    basis,
    slaterWeightMin: float = 0.0,
    truncation_threshold: float = 0.0,
    reort_period: int = 5,
    start_it: int = 0,
    block_widths=None,
    locked=None,
    locked_reort="full",
    krylov=None,
):
    """Perform one step of the distributed block Lanczos iteration.

    Executes the three-term block recurrence

    .. math::

        W_p = H Q_i - Q_i \\alpha_i - Q_{i-1} \\beta_{i-1}^\\dagger

    where :math:`\\alpha_i = Q_i^\\dagger H Q_i` is the diagonal block and
    :math:`\\beta_{i-1}` is the off-diagonal block from the previous step.
    The residual block :math:`W_p` is then QR-factorised via the Cholesky
    decomposition of :math:`M = W_p^\\dagger W_p`:

    .. math::

        M = L L^\\dagger, \\quad
        \\beta_i = L^\\dagger, \\quad
        Q_{i+1} = W_p \\beta_i^{-1}.

    **MPI collective operations** (all ranks must call simultaneously):

    1. ``MPI_Allreduce`` (``MPI.SUM``) on :math:`\\alpha_i` – :math:`p \\times p`
       complex matrix; result is replicated on every rank.
    2. ``MPI_Allreduce`` (``MPI.SUM``) on :math:`M = W_p^\\dagger W_p` – :math:`p \\times p`
       complex matrix; used for Cholesky QR and breakdown detection.
    3. For ``FULL`` / ``PERIODIC`` modes, an additional ``MPI_Allreduce`` per
       reorthogonalization pass on the :math:`(it \\cdot p) \\times p` overlap matrix.
    4. For ``PARTIAL``, standard Paige-Simon tracking operates against all Lanczos vectors,
       triggering ``MPI_Allreduce`` calls over potentially large swaths of the Krylov
       basis when orthogonality loss exceeds :math:`\\sqrt{\\varepsilon_{\\text{mach}}}`.
    5. For ``SELECTIVE`` (EA16), tracking operates exclusively against converged Ritz vectors.
       The Ritz overlap is calculated as $w_{ritz, k} = s_k^\\dagger W[:, \\text{curr}]$, and
       the W-matrix geometric projection update is
       $W[:, \\text{curr}] \\leftarrow W[:, \\text{curr}] - (W \\cdot s_k) \\langle y_k \\mid w_p \\rangle$.
       Because ``SELECTIVE`` exclusively projects against a small subset of Ritz vectors rather
       than the entire basis, the ``MPI_Allreduce`` footprint is drastically lower.

    Note:
        ``h_op.apply_multi`` releases the GIL internally; all other operations in
        this function hold the GIL.

    Args:
        h_op: ``ManyBodyOperator`` Hamiltonian; must implement
            ``apply_multi(psis, cutoff)``.
        q_prev: List of ``p`` ``ManyBodyState`` objects from iteration ``i-1``
            (pass an empty list or zero states at ``it=0``).
        q_curr: List of ``p`` ``ManyBodyState`` objects from iteration ``i``.
        Q_basis: Accumulated Krylov basis as a flat list of ``ManyBodyState``
            (length ``p * (it + 1)`` on entry, grows by ``p`` each step).
        alphas: Pre-allocated numpy array of shape ``(max_iter, p, p)`` that
            receives the diagonal block :math:`\\alpha_i` at index ``it``.
        betas: Pre-allocated numpy array of shape ``(max_iter, p, p)`` that
            receives the off-diagonal block :math:`\\beta_i` at index ``it``.
        it: Current iteration index into ``alphas`` / ``betas`` buffers
            (0-based, reset to 0 at the start of each fresh run).
        reort_mode: ``Reort`` enum value controlling reorthogonalization strategy.
            ``Reort.NONE`` skips all reorthogonalization entirely.
        W: Paige-Simon W-matrix estimator state array of shape
            ``(2, it+1, p, p)``, or ``None`` when not used.  Required for
            ``PARTIAL`` / ``SELECTIVE`` modes; ignored otherwise.
        mpi: ``True`` when running under MPI with more than one rank.
        comm: Active ``mpi4py.MPI.Comm`` communicator, or ``None`` when running
            serially.
        slaterWeightMin: Amplitude cutoff passed to ``ManyBodyState.prune``;
            SD coefficients below this value are dropped.  Default ``0.0``
            (no pruning).
        reort_period: Number of steps between full reorthogonalization sweeps
            for ``Reort.PERIODIC`` mode.  Full reorthogonalization is applied at
            step ``it`` when ``it > 0`` and ``it % reort_period == 0``.
            Default ``5``.

    Returns:
        tuple: A 5-tuple ``(q_next, alpha_i, beta_i, W_updated, breakdown)``:

        * ``q_next`` – List of ``p`` ``ManyBodyState`` objects forming the next
          Krylov block :math:`Q_{i+1}`, or ``None`` if breakdown occurred.
        * ``alpha_i`` – numpy complex array of shape ``(p, p)`` for the current
          diagonal block.
        * ``beta_i`` – numpy complex array of shape ``(p, p)`` for the current
          off-diagonal block, or ``None`` if breakdown occurred.
        * ``W_updated`` – updated Paige-Simon estimator array, or ``None`` if
          not applicable.
        * ``breakdown`` – ``True`` if an invariant subspace or an ill-conditioned
          block was detected (``NaN``/``Inf`` in :math:`M`, or condition number
          exceeding :math:`100 / \\varepsilon_{\\text{mach}}`).
    """
    p = len(q_curr)

    # --- 1. Block matvec: wp = H q_curr ---------------------------------
    _t0 = _time.perf_counter()
    wp = h_op.apply_multi(q_curr, slaterWeightMin)
    if mpi and comm is not None and basis is not None:
        wp = basis.redistribute_psis(wp)
    _prof_acc("matvec", _t0)

    # --- 2. alpha_i = <q_curr | wp> -------------------------------------
    _t0 = _time.perf_counter()
    alpha_i = inner_multi(q_curr, wp)
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, alpha_i, op=MPI.SUM)
    alphas[it, :p, :p] = alpha_i

    # --- 3. Subtract: wp = wp - q_curr * alpha_i - q_prev * beta_{i-1}^† -
    add_scaled_multi(wp, q_curr, -alpha_i)
    if it > 0:
        n_prev = len(q_prev)
        beta_prev_dag = np.conj(betas[it - 1, :p, :n_prev].T)
        add_scaled_multi(wp, q_prev, -beta_prev_dag)

    # --- 3b. EA16 §2.6.2 locking deflation ------------------------------
    # Keep the residual block orthogonal to the already-converged ("locked") Ritz
    # vectors *before* forming M and beta, so the stored beta and the resulting
    # q_next are consistent with the deflated vector (matching the array kernel).
    # The locked Ritz vectors are only approximate eigenvectors, so H q_curr leaks
    # a small component along them that is otherwise amplified across iterations,
    # reintroducing locked eigenvalues (and their 2*theta harmonics) as spurious
    # Ritz values below the true spectral minimum on restarted sweeps. Twice for
    # numerical robustness. Skipped in the "partial" mode, where the estimate-driven
    # EA16 §2.6.2 reorth is applied to q_next in block_lanczos_cy instead.
    if locked and locked_reort != "partial":
        for _ in range(2):
            wp, _ = block_orthogonalize_sparse(wp, list(locked), None, comm if mpi else None)

    # --- 4. Full / Periodic reorthogonalization -------------------------
    # The PERIODIC cadence gate stays in the caller; the reort action itself goes
    # through the shared apply_reort (single FULL implementation for both kernels;
    # bit-for-bit equal to the old 2x inner_multi/add_scaled_multi loop because
    # block_orthogonalize_sparse does exactly that with comm=comm-if-mpi).
    if reort_mode == Reort.FULL or (reort_mode == Reort.PERIODIC and it > 0 and it % reort_period == 0):
        wp, _, _ = apply_reort(wp, Q_basis, None, Reort.FULL, mpi, comm, block_widths or [], krylov)

    # --- 5. M = <wp|wp>, check breakdown --------------------------------
    M = inner_multi(wp, wp)
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)

    if np.any(np.isnan(M)) or np.any(np.isinf(M)):
        # Non-finite Gram matrix => the recurrence is *corrupted*, not a genuine invariant
        # subspace. Signal it with active_k = -1 so the caller can report "diverged" (and
        # warn) instead of treating the truncated result as exact. active_k == 0 below is a
        # real rank-deficient residual (the Krylov space is closed) => invariant subspace.
        return None, alpha_i, None, W, -1, True

    # --- 6. Deflation / Cholesky QR -------------------------------------
    beta_i, beta_inv, active_k = _cholesky_or_deflate(M, p)
    if active_k == 0:
        return None, alpha_i, None, W, 0, True

    q_next = [ManyBodyState() for _ in range(active_k)]
    add_scaled_multi(q_next, wp, beta_inv)
    _prof_acc("recurrence", _t0)

    # --- 6a. Forceful / Amplitude Truncation ----------------------------
    cdef bint did_truncate = False
    if slaterWeightMin > 0.0:
        for st in q_next:
            st.prune(slaterWeightMin)
        did_truncate = True
    if truncation_threshold > 0:
        from impurityModel.ed.ManyBodyUtils import apply_global_truncation
        for st in q_next:
            apply_global_truncation(st, truncation_threshold, comm if mpi else None)
        did_truncate = True

    # --- 6b. CholeskyQR2 (conditional): re-orthonormalize using the actual vectors -----
    _t0 = _time.perf_counter()
    # A single Cholesky-QR leaves q_next non-orthonormal by O(cond(M)*EPS); when cond(M) is
    # large this amplifies the Krylov vectors and diverges the recurrence, so a second pass is
    # required. But when cond(M) < EPS^(-1/3) the first pass is already orthonormal to
    # < EPS^(2/3) (~4e-11), far below sqrt(EPS), so the second pass (an extra Gram inner product
    # + MPI Allreduce) is skipped. The same cond(M) gates both kernels, so they stay in lock-step.
    # If the basis was forcefully truncated, orthogonality is broken and CholeskyQR2 MUST run.
    if did_truncate or np.linalg.cond(M) >= EPS ** (-1.0 / 3.0):
        M2 = inner_multi(q_next, q_next)
        if mpi and comm is not None:
            comm.Allreduce(MPI.IN_PLACE, M2, op=MPI.SUM)
        M2 = 0.5 * (M2 + np.conj(M2.T))
        beta2_inv, beta_i, active_k = _cholesky_qr2(M2, beta_i, active_k)
        if active_k == 0:
            return None, alpha_i, None, W, 0, True
        q_next2 = [ManyBodyState() for _ in range(active_k)]
        add_scaled_multi(q_next2, q_next, beta2_inv)
        q_next = q_next2
    _prof_acc("choleskyqr2_cond", _t0)

    betas[it, :active_k, :p] = beta_i

    # --- 7. EA16 Selective Orthogonalization / Partial Reortho ---------
    if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
        _t0 = _time.perf_counter()
        if W is None:
            if start_it > 0:
                W = np.zeros((2, start_it + 1, alphas.shape[1], alphas.shape[1]), dtype=complex)
                for j in range(start_it):
                    w_j = block_widths[j]
                    Q_j = Q_basis[sum(block_widths[:j]) : sum(block_widths[:j+1])]
                    W[1, j, :w_j, :p] = inner_multi(Q_j, wp)
                    if j < start_it - 1:
                        W[0, j, :w_j, :n_prev] = inner_multi(Q_j, q_prev)
                if mpi and comm is not None:
                    comm.Allreduce(MPI.IN_PLACE, W, op=MPI.SUM)
                W[1, start_it, :p, :p] = np.eye(p)
                W[0, start_it - 1, :n_prev, :n_prev] = np.eye(n_prev)
            else:
                W = np.zeros((2, 1, p, p), dtype=complex)
                W[1, 0] = np.eye(p)

        W = estimate_orthonormality(
            W,
            alphas[: it + 1],
            betas[: it + 1],
            block_widths=block_widths + [p, active_k],
            eps=EPS,
        )
        _prof_acc("w_estimate", _t0)
        _t0 = _time.perf_counter()

        reort_eps = REORT_TOL

        if reort_mode == Reort.SELECTIVE:
            # EA16 §2.6.2 selective orthogonalization (shared with block_lanczos_array_cy).
            # beta_i's 2-norm is the Ritz residual scale; the driver has not computed it yet at
            # this point, so pass it explicitly.
            q_next = selective_orthogonalize(
                q_next, Q_basis, alphas, betas, W, block_widths,
                it, p, np.linalg.norm(beta_i, ord=2), reort_eps, reort_period, mpi, comm,
            )

        if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
            # Bad-block partial reorthogonalization via the shared apply_reort (single
            # implementation for both kernels). Pass block_widths + [p] so the current
            # block (index it) is included in the width table apply_reort indexes.
            q_next, W, _reort_acted = apply_reort(q_next, Q_basis, W, reort_mode, mpi, comm, block_widths + [p], krylov)
            if _PROF_ON:
                _PROF["reort_total#n"] = _PROF.get("reort_total#n", 0.0) + 1.0
                if _reort_acted:
                    _PROF["reort_acted#n"] = _PROF.get("reort_acted#n", 0.0) + 1.0
            # Only when a bad block was actually projected does q_next need re-orthonormalizing;
            # otherwise it is unchanged and this would be an exact no-op (M2 == I), so skip it and
            # save the Gram inner product + MPI Allreduce. Mirrors block_lanczos_array_cy.
            if _reort_acted:
                M2 = inner_multi(q_next, q_next)
                if mpi and comm is not None:
                    comm.Allreduce(MPI.IN_PLACE, M2, op=MPI.SUM)
                M2 = 0.5 * (M2 + np.conj(M2.T))
                # Absolutely tiny residual after projection => block contained in the existing span
                # (invariant subspace); renormalizing it would amplify rounding. Treat as breakdown.
                if float(np.max(np.real(np.diag(M2)))) < EPS:
                    return None, alpha_i, None, W, 0, True
                beta2_inv, beta_i, active_k = _cholesky_qr2(M2, beta_i, active_k)
                if active_k == 0:
                    return None, alpha_i, None, W, 0, True
                q_next2 = [ManyBodyState() for _ in range(active_k)]
                add_scaled_multi(q_next2, q_next, beta2_inv)
                q_next = q_next2
                betas[it, :active_k, :p] = beta_i
        _prof_acc("reort", _t0)

    return q_next, alpha_i, beta_i, W, active_k, False


def block_lanczos_cy(
    psi0,
    h_op,
    basis,
    converged_fn,
    verbose: bool = False,
    reort="full",
    max_iter=None,
    slaterWeightMin: float = 0.0,
    truncation_threshold: int = 0,
    comm=None,
    reort_period: int = 5,
    alphas_init=None,
    betas_init=None,
    Q_init=None,
    W_init=None,
    return_widths=False,
    return_status=False,
    block_widths_init=None,
    locked=None,
    locked_evals=None,
    locked_res=0.0,
    locked_reort="full",
    store_krylov=True,
):
    """Run the distributed block Lanczos iteration with ``ManyBodyState``.

    Implements the block Lanczos three-term recurrence

    .. math::

        H Q_i = Q_i \\alpha_i + Q_{i+1} \\beta_i + Q_{i-1} \\beta_{i-1}^\\dagger

    building the block-tridiagonal matrix

    .. math::

        T_m = \\begin{pmatrix}
            \\alpha_0 & \\beta_0^\\dagger & & \\\\
            \\beta_0  & \\alpha_1 & \\ddots & \\\\
                     & \\ddots & \\ddots & \\beta_{m-2}^\\dagger \\\\
                     & & \\beta_{m-2} & \\alpha_{m-1}
        \\end{pmatrix}.

    **Pre-allocation**: ``alphas_buf`` and ``betas_buf`` of shape
    ``(max_iter, p, p)`` are allocated *once* before the main loop so that no
    NumPy allocation occurs inside the Lanczos body.  The convergence check and the
    returned arrays are *views* into these buffers (``alphas_buf[:it_abs+1]``) — no
    per-iteration list rebuild or ``np.array()`` copy.

    **Warm-start / resume protocol**: if ``alphas_init``, ``betas_init``, and
    ``Q_init`` are all provided the iteration resumes from block
    ``len(alphas_init)``.  The warm-start Q blocks are extracted from
    ``Q_init`` and the existing W-estimator (``W_init``) is reused.  If
    ``W_init`` is ``None`` but ``reort`` is ``'partial'`` or ``'selective'``,
    the Paige-Simon estimator array ``W`` is exactly initialized by computing the
    exact overlaps of the starting blocks against all prior blocks (Exact
    Overlap Restart - EOR).  Pass ``None`` for all three to start a fresh run.

    **MPI collective operations** (called every iteration, all ranks must participate):

    * One ``MPI_Allreduce`` (``MPI.SUM``) for :math:`\\alpha_i` – shape
      ``(p, p)``; result is replicated on all ranks.
    * One ``MPI_Allreduce`` (``MPI.SUM``) for :math:`M = W_p^\\dagger W_p` –
      shape ``(p, p)``; used for Cholesky QR.
    * Additional ``MPI_Allreduce`` calls inside reorthogonalization (see
      ``block_lanczos_step_cy`` for details).

    Note:
        ``converged_fn`` is called on **all ranks** with *replicated* data
        (``alphas``, ``betas`` are identical on every rank).  If the callable
        performs rank-dependent logic the caller is responsible for broadcasting
        the result.  When ``reort == Reort.NONE`` no reorthogonalization of any
        kind is performed; orthogonality degrades at the rate of floating-point
        rounding errors.

    Args:
        psi0: Initial block of ``p`` ``ManyBodyState`` objects, or ``None``
            when resuming from ``Q_init`` (warm-start mode).
        h_op: ``ManyBodyOperator`` that implements ``apply_multi(psis, cutoff)``.
        basis: ``Basis`` object providing ``redistribute_psis`` and ``basis.comm``.
        converged_fn: Callable with signature
            ``converged_fn(alphas, betas, verbose=bool) -> bool`` that returns
            ``True`` once the desired accuracy is reached.  Called after every
            accepted step with the *full* ``alphas``/``betas`` arrays built so far.
        verbose: Print per-iteration diagnostics including ``|beta|`` norms and
            diagonal alpha values.  Default ``False``.
        reort: Reorthogonalization mode.  Accepts ``Reort`` enum members or the
            equivalent strings ``'none'``, ``'partial'``, ``'selective'``,
            ``'full'``, ``'periodic'``.  ``Reort.NONE`` disables all
            reorthogonalization.  Default ``'full'``.
        max_iter: Maximum number of Lanczos steps before returning.  Defaults to
            ``basis.size // p`` when ``basis.size`` is available, otherwise
            ``10 * p``.
        slaterWeightMin: Amplitude cutoff passed to ``ManyBodyState.prune``.
            Slater determinants with :math:`|c_k| < \\text{slaterWeightMin}` are
            dropped after each matvec.  Default ``0.0`` (no pruning).
        comm: ``mpi4py`` communicator.  If ``None``, falls back to ``basis.comm``
            and then to serial mode (``mpi=False``).
        reort_period: Period for ``Reort.PERIODIC`` mode.  Full
            reorthogonalization is applied at steps that satisfy
            ``it > 0 and it % reort_period == 0``.  Default ``5``.
        alphas_init: Warm-start diagonal block array of shape ``(k0, p, p)``.
            If provided together with ``betas_init`` and ``Q_init``, iteration
            resumes from block ``k0``.  Default ``None``.
        betas_init: Warm-start off-diagonal block array of shape ``(k0, p, p)``.
            Default ``None``.
        Q_init: Warm-start Krylov basis as a flat list of ``ManyBodyState``
            (length ``>= k0 * p``).  Default ``None``.
        W_init: Warm-start Paige-Simon W-estimator array.  Passed through to
            the step function; ``None`` causes exact initialisation via EOR
            when resuming with partial/selective reorthogonalization.  Default
            ``None``.
        store_krylov: When ``False`` (requires ``reort == 'none'`` and no
            ``locked`` set) the accumulated Krylov basis is *not* retained;
            the returned ``Q_basis`` holds only the last two blocks (previous
            block + residual) — exactly what the warm-start protocol reads.
            The alphas/betas are bit-identical to ``store_krylov=True``; only
            the O(N_det * p * k) dead retention is dropped.  On resume,
            ``Q_init`` is interpreted as that two-block tail (split by
            ``block_widths_init[-1]``).  Default ``True``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, list, numpy.ndarray]: A 4-tuple
        ``(alphas, betas, Q_basis, W)`` where

        * ``alphas`` – complex array of shape ``(k, p, p)`` holding the
          diagonal blocks :math:`\\alpha_0, \\dots, \\alpha_{k-1}`.
        * ``betas`` – complex array of shape ``(k, p, p)`` holding the
          off-diagonal blocks :math:`\\beta_0, \\dots, \\beta_{k-1}`.
        * ``Q_basis`` – flat list of ``ManyBodyState`` of length
          ``(k + 1) * p``; the last ``p`` entries form the residual

        ``return_widths=True`` appends ``block_widths`` (the per-block true widths)
        and ``return_status=True`` appends a ``termination`` string -- one of
        ``"converged"`` (``converged_fn`` satisfied), ``"invariant_subspace"`` (the
        block-Krylov space is closed under H within the active restrictions, so the
        result is *exact*), ``"diverged"`` (the divergence guard truncated a corrupted
        tail; NOT exact) or ``"max_iter"`` (the ``max_iter`` budget was exhausted before
        the recurrence terminated naturally). Both flags are independent and opt-in, so
        existing call sites that pass neither keep the 4-tuple return.
    """
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    if isinstance(h_op, dict):
        h_op = ManyBodyOperator(h_op)

    # --- Resolve communicator / MPI flag --------------------------------
    if comm is None:
        comm = getattr(basis, "comm", None)
    mpi = comm is not None and comm.Get_size() > 1

    # --- Resolve reort mode ---------------------------------------------
    reort_mode = resolve_reort(reort)

    # Tail-only mode: with reort == NONE (and no locked set) nothing in the recurrence
    # ever projects against the accumulated Krylov basis, and the warm-start protocol
    # reads only the last two blocks -- so retaining the full O(N_det * p * k) basis is
    # pure dead memory (the dominant avoidable allocation in a Green's-function run).
    if not store_krylov and (reort_mode != Reort.NONE or locked):
        raise ValueError("store_krylov=False requires reort='none' and no locked vectors")

    # --- Resume or start fresh? -----------------------------------------
    resuming = alphas_init is not None and betas_init is not None and Q_init is not None

    cdef list block_widths = list(block_widths_init) if block_widths_init is not None else []
    if resuming:
        start_it = len(alphas_init)
        p = alphas_init[0].shape[0] if len(alphas_init) > 0 else len(Q_init[0] if Q_init else psi0)
        if len(block_widths) == 0:
            block_widths = [p] * start_it
        W = W_init

        if not store_krylov:
            # Q_init is the two-block tail [prev block, residual block] returned by the
            # previous tail-only run; split it by the true width of the last accepted
            # block (deflation-safe: block_widths is authoritative, the tail may be
            # narrower than p).
            Q_basis = []
            if start_it == 0:
                q_prev = [ManyBodyState() for _ in range(p)]
                q_curr = list(Q_init)
            else:
                q_prev_len = block_widths[start_it - 1]
                q_prev = list(Q_init[:q_prev_len])
                q_curr = list(Q_init[q_prev_len:])
        elif start_it == 0 or len(Q_init) < p:
            Q_basis = list(Q_init)
            q_prev = [ManyBodyState() for _ in range(p)]
            q_curr = [Q_basis[i] for i in range(p)]
        else:
            Q_basis = list(Q_init)
            q_prev_start = sum(block_widths[:start_it - 1])
            q_prev_len = block_widths[start_it - 1]
            q_curr_start = sum(block_widths[:start_it])
            q_curr_len = len(Q_basis) - q_curr_start
            q_prev = [Q_basis[i] for i in range(q_prev_start, q_prev_start + q_prev_len)]
            q_curr = [Q_basis[i] for i in range(q_curr_start, q_curr_start + q_curr_len)]
    else:
        start_it = 0
        p = len(psi0)
        W = None

        # Redistribute initial states across ranks
        q_curr = basis.redistribute_psis(list(psi0))
        q_prev = [ManyBodyState() for _ in range(p)]
        Q_basis = [st.copy() for st in q_curr] if store_krylov else []

    # Maintain a dense copy of the (rank-local) Krylov basis so the block reort slices
    # columns instead of re-materializing Q from flat_maps every step (see SparseKrylovDense).
    # Only kept for FULL/PERIODIC, which project against *all* columns every (periodic) step:
    # there the per-step re-materialization would dominate. PARTIAL/SELECTIVE project only
    # against flagged bad blocks on rare trigger steps, so they use the transient
    # reorth_cgs2_dense fallback in apply_reort instead of holding a second full copy of the
    # Krylov basis (the mirror doubles the dominant memory of every CIPSI/ground-state solve).
    krylov = None
    if reort_mode in (Reort.FULL, Reort.PERIODIC):
        krylov = SparseKrylovDense()
        krylov.append(Q_basis)

    # --- Determine max_iter ---------------------------------------------
    if max_iter is None:
        max_iter = getattr(basis, "size", 10 * p) // p

    # Pre-allocate buffers to avoid allocation inside loop
    _buf_size = max(int(max_iter), 1)
    if start_it > 0:
        alphas_buf = np.zeros((start_it + _buf_size, p, p), dtype=complex)
        betas_buf = np.zeros((start_it + _buf_size, p, p), dtype=complex)
        alphas_buf[:start_it] = alphas_init
        betas_buf[:start_it] = betas_init
    else:
        alphas_buf = np.zeros((_buf_size, p, p), dtype=complex)
        betas_buf = np.zeros((_buf_size, p, p), dtype=complex)
    # Seed W-estimator state
    if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE) and W is None:
        if start_it > 0:
            W = np.zeros((2, start_it + 1, p, p), dtype=complex)
            for j in range(start_it):
                w_j = block_widths[j]
                Q_j = Q_basis[sum(block_widths[:j]) : sum(block_widths[:j+1])]

                ov_curr = inner_multi(Q_j, q_curr)
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, ov_curr, op=MPI.SUM)
                W[1, j, :w_j, :len(q_curr)] = ov_curr

                if j < start_it - 1:
                    ov_prev = inner_multi(Q_j, q_prev)
                    if mpi:
                        comm.Allreduce(MPI.IN_PLACE, ov_prev, op=MPI.SUM)
                    W[0, j, :w_j, :len(q_prev)] = ov_prev

            W[1, start_it, :len(q_curr), :len(q_curr)] = np.eye(len(q_curr))
            W[0, start_it - 1, :len(q_prev), :len(q_prev)] = np.eye(len(q_prev))
        else:
            W = np.zeros((2, 1, p, p), dtype=complex)
            W[1, 0] = np.eye(p)
    elif reort_mode in (Reort.PARTIAL, Reort.SELECTIVE) and W is not None:
        pass  # W expands dynamically in estimate_orthonormality

    # Estimate-driven locking reorthogonalization (EA16 §2.6.2) state. Only active when a
    # locked set is supplied and locked_reort == "partial"; otherwise the step does the
    # unconditional "full" projection. See ea16.locked_overlap_step for the recurrence.
    partial_locked = bool(locked) and locked_reort == "partial"
    nlock = len(locked) if locked else 0
    if partial_locked:
        from impurityModel.ed import ea16 as _ea16
        locked_evals_arr = np.ascontiguousarray(np.real(np.asarray(locked_evals)), dtype=float)
        _Ntot = getattr(basis, "size", 0) or 1
        omega_min_l = EPS * p * math.sqrt(_Ntot)
        xi_l = np.full(nlock, omega_min_l)
        xi_prev_l = np.full(nlock, omega_min_l)
        locked_rho = float(locked_res)

    # --- Main Lanczos loop ----------------------------------------------
    it = 0
    breakdown = False
    # Why the recurrence stops (returned when return_status=True). Default assumes the loop
    # runs out the max_iter budget without terminating naturally; the branches below overwrite
    # it. "invariant_subspace" => the block-Krylov space is closed under H (within the active
    # restrictions) and the result is *exact*; "diverged" => the divergence guard truncated a
    # corrupted tail (NOT exact); "converged" => converged_fn was satisfied.
    termination = "max_iter"
    # Largest healthy block-norm seen so far (proxy for ||H|| on the subspace); used to
    # detect a runaway recurrence that the deflation/QR safeguards did not catch.
    t_norm_max = 0.0
    # Spectral-scale estimate (seeded from beta_0 ~ ||H||, grown only by ||alpha_i||, which is
    # bounded by ||H|| and never runs away) — catches gradual beta growth the relative check misses.
    h_norm_est = 0.0

    while it < _buf_size:
        it_abs = start_it + it
        n_curr = len(q_curr)
        q_next, alpha_i, beta_i, W, active_k, breakdown = block_lanczos_step_cy(
            h_op=h_op,
            q_prev=q_prev,
            q_curr=q_curr,
            Q_basis=Q_basis,
            alphas=alphas_buf,
            betas=betas_buf,
            it=it_abs,
            reort_mode=reort_mode,
            W=W,
            mpi=mpi,
            comm=comm,
            basis=basis,
            slaterWeightMin=slaterWeightMin,
            truncation_threshold=truncation_threshold,
            reort_period=reort_period,
            start_it=start_it,
            block_widths=block_widths,
            locked=locked,
            locked_reort=locked_reort,
            krylov=krylov,
        )

        if breakdown:
            # active_k < 0 marks a non-finite (corrupted) Gram matrix -> truncated, NOT exact;
            # active_k == 0 is a genuine rank-deficient residual -> the block-Krylov space is
            # closed under H -> the continued fraction is exact on it.
            termination = "diverged" if active_k < 0 else "invariant_subspace"
            if verbose:
                if comm is None or comm.Get_rank() == 0:
                    print(f"[BlockLanczos] {termination} detected at iteration {it_abs}.")
            block_widths.append(n_curr)
            it += 1
            break

        # --- Divergence safeguard ---------------------------------------
        # For a Hermitian H every block norm is bounded by ||H||, so ||alpha_i||/||beta_i||
        # can never exceed the largest healthy block norm by more than rounding. A jump of
        # several orders of magnitude means the recurrence has been corrupted (lost
        # orthogonality the QR/deflation did not repair). Truncate *before* the offending
        # block — exclude alpha_i/beta_i at it_abs — so the returned T-matrix is the last
        # numerically trustworthy subspace and the diverged tail never reaches the Green's
        # function. The trailing beta then plays the role of the (ignored) residual coupling.
        # One SVD of beta_i per step: its largest singular value is the 2-norm (guard/verbose)
        # and its smallest gives ||beta_i^-1|| for the locked-reort estimate below.
        _svb = np.linalg.svd(beta_i, compute_uv=False)
        beta_norm = float(_svb[0])
        alpha_norm = np.linalg.norm(alpha_i, ord=2)
        diverged, t_norm_max, h_norm_est = divergence_guard(
            beta_norm, alpha_norm, it == 0, t_norm_max, h_norm_est
        )
        if diverged:
            if verbose and (comm is None or comm.Get_rank() == 0):
                print(
                    f"[BlockLanczos] Divergence detected at iteration {it_abs}: "
                    f"|beta|={beta_norm:.3e}, |alpha|={alpha_norm:.3e} >> spectral scale "
                    f"{h_norm_est:.3e}. Truncating to the last trustworthy block."
                )
            termination = "diverged"
            breakdown = True
            break

        # EA16 §2.6.2 estimate-driven locking reorth: propagate the per-pair overlap
        # estimate (cheap, no O(N) work) and reorthogonalize q_next only against the
        # locked vectors whose estimate now exceeds omega_TOL.
        if partial_locked:
            bj_inv_norm = 1.0 / max(float(_svb[len(_svb) - 1]), EPS)  # reuse the step's beta_i SVD
            bjm1_norm = float(np.linalg.norm(betas_buf[it_abs - 1], 2)) if it_abs > 0 else 0.0
            xi_new_l, xi_trigger, xi_mask = _ea16.locked_overlap_step(
                xi_l, xi_prev_l, locked_evals_arr, alpha_i,
                bj_inv_norm, bjm1_norm, locked_rho, REORT_TOL, BAD_BLOCK_TOL, EPS,
            )
            if xi_trigger:
                idx_l = np.nonzero(xi_mask)[0]
                Lm = [locked[int(t)] for t in idx_l]
                for _ in range(2):
                    q_next, _ = block_orthogonalize_sparse(q_next, Lm, None, comm if mpi else None)
                xi_new_l[idx_l] = omega_min_l
                xi_l[idx_l] = omega_min_l
            xi_prev_l = xi_l
            xi_l = xi_new_l

        # Append q_next to the basis (last block = residual direction). The locking
        # deflation is applied inside block_lanczos_step_cy (before M/beta) in "full"
        # mode, or just above in "partial" mode, so q_next is orthogonal to the locked set.
        # Tail-only mode keeps nothing here: q_prev/q_curr roll below and the two-block
        # tail is assembled at return.
        if store_krylov:
            Q_basis.extend([st.copy() for st in q_next])
        if krylov is not None:
            krylov.append(q_next)

        if verbose:
            print(
                f"[BlockLanczos] it={it_abs:4d}  " f"|beta|={beta_norm:.3e}  " f"alpha_diag={np.real(np.diag(alpha_i))}"
            )

        # Convergence check
        alphas_np = alphas_buf[:it_abs+1]
        betas_np = betas_buf[:it_abs+1]
        _t0m = _time.perf_counter()
        _converged = converged_fn(alphas_np, betas_np, verbose=verbose, block_widths=block_widths + [n_curr])
        _prof_acc("monitor", _t0m)
        if _converged:
            termination = "converged"
            block_widths.append(n_curr)
            it += 1
            break

        q_prev = q_curr
        q_curr = q_next
        block_widths.append(n_curr)
        it += 1

    alphas_out = alphas_buf[:start_it + it]
    betas_out = betas_buf[:start_it + it]
    if not store_krylov:
        # Return the two-block tail [last accepted block, residual block], mirroring the
        # last two blocks of the full Q_basis: on a "converged" break the q_prev/q_curr
        # roll has not run yet (tail = q_curr + q_next); on breakdown/divergence q_next
        # was never appended, and on budget exhaustion the roll has run (both: tail =
        # q_prev + q_curr).
        if termination == "converged":
            Q_basis = list(q_curr) + list(q_next)
        else:
            Q_basis = list(q_prev) + list(q_curr)
    if return_widths and return_status:
        return alphas_out, betas_out, Q_basis, W, block_widths, termination
    if return_widths:
        return alphas_out, betas_out, Q_basis, W, block_widths
    if return_status:
        return alphas_out, betas_out, Q_basis, W, termination
    return alphas_out, betas_out, Q_basis, W


# ---------------------------------------------------------------------------
# Thick-Restart Block Lanczos (TRLM)
# ---------------------------------------------------------------------------


def _trlm_extract(T_full, Q, dim, num_wanted, comm, slater):
    """Diagonalize the leading ``dim`` x ``dim`` block of the (possibly arrowhead)
    ``T_full`` and form the ``num_wanted`` lowest Ritz vectors as combinations of
    ``Q[:dim]``. Path-agnostic (array ndarray or ManyBodyState list) via ``block_combine``.
    Shared by the TRLM early-exit / breakdown / final-extraction paths so they all honor
    the true (possibly deflated) subspace dimension ``dim`` instead of a padded
    ``m_actual * p``."""
    eigvals_T, eigvecs_T = sp.eigh(T_full[:dim, :dim])
    if comm is not None:
        eigvals_T = comm.bcast(eigvals_T, root=0)
        eigvecs_T = comm.bcast(eigvecs_T, root=0)
    wanted = np.argsort(eigvals_T)[:num_wanted]
    return eigvals_T[wanted], block_combine(_q_slice(Q, 0, dim), eigvecs_T[:, wanted], slater)


def _trlm_core(
    psi0,
    h_op,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol,
    max_restarts,
    verbose,
    reort_mode,
    slater,
    comm,
    sweep,
):
    """Path-agnostic thick-restart block Lanczos (TRLM).

    Implements the thick-restart strategy of Knyazev (2001) and Wu & Simon (2000) to find
    the ``num_wanted`` algebraically smallest eigenvalues (and Ritz vectors) of ``H``.
    Drives both the dense-array and ``ManyBodyState`` paths through the path-agnostic
    ``block_*`` helpers (``block_apply`` / ``block_combine`` / ``block_inner`` /
    ``block_orthogonalize`` / ``block_normalize``) and a path-specific ``sweep`` callable
    (``block_lanczos_array`` for arrays, ``block_lanczos_cy`` for ``ManyBodyState``).

    Algorithm: run one block-Lanczos sweep of ``m`` blocks; diagonalise the
    block-tridiagonal ``T``; keep the ``k = ceil(n_w/p)`` lowest Ritz pairs as one retained
    super-block; reattach the residual block as a thick-restart spike; and continue the
    recurrence from block ``k`` until the maximum wanted residual ``||beta_res s_i||`` drops
    below ``tol`` or ``max_restarts`` is exhausted. A genuine invariant subspace (fewer
    blocks than requested) or a block deflation (rank-deficient residual) terminates early
    with a direct banded extraction.

    The loop is width-aware: blocks can shrink mid-restart (rank-deficient residual ->
    rectangular beta + narrower ``q_next``), so it tracks each block's actual width
    (``cur_widths``) and addresses ``T_full`` / ``Q_basis`` by cumulative offsets instead of
    a constant block width ``p``. ``nkeep = k_blocks * p`` is a *count* of retained Ritz
    vectors (one diagonal super-block coupled to the residual by the thick-restart spike),
    not a block width.

    MPI: the dense ``sp.eigh`` results (replicated across ranks because ``T_full`` is built
    from Allreduced coefficients) are broadcast from rank 0 so every rank uses identical Ritz
    vectors and the restart bases stay in lock-step.

    Returns:
        tuple[numpy.ndarray, list | numpy.ndarray]: ``(eigvals, eigvecs)`` — the
        ``num_wanted`` smallest eigenvalues (ascending) and matching Ritz vectors in the
        path's basis representation.
    """
    mpi = comm is not None and getattr(comm, "size", 1) > 1
    rank0 = (not mpi) or comm.rank == 0

    is_arr = is_array(psi0)
    p = (psi0.shape[1] if (is_arr and psi0.ndim == 2) else 1) if is_arr else len(psi0)
    k_blocks = int(np.ceil(num_wanted / p))
    m = max_subspace_blocks
    if m <= k_blocks:
        raise ValueError("max_subspace_blocks must be strictly greater than ceil(num_wanted / p).")

    # --- Initial Lanczos run --------------------------------------------
    alphas, betas, Q_basis, widths = sweep(psi0, m)
    m_actual = len(alphas)
    if m_actual == 0:
        raise RuntimeError("Block Lanczos produced zero iterations.")

    # The sweep can shrink blocks (rank-deficient beta -> deflation), so the true subspace
    # dimension is sum(widths), not the padded m_actual * p. All slicing / T construction
    # must use the real widths or they desynchronize from Q_basis.
    total = int(sum(widths)) if widths is not None else m_actual * p
    deflated = total < m_actual * p

    # Split the trailing residual block off the basis.
    q_m = _copy_block(_q_slice(Q_basis, total, _q_cols(Q_basis))) if _q_cols(Q_basis) > total else None
    Q_basis = _q_slice(Q_basis, 0, total)

    _betas_off = betas[: m_actual - 1] if len(betas) == m_actual else betas

    # Early termination: a genuine invariant subspace (fewer blocks than asked) or block
    # deflation (rank-deficient residual). In both cases the spanned block-Krylov space is
    # (near-)invariant, so its Ritz pairs are accurate eigenpairs and we extract directly.
    # T here is a pure block-tridiagonal, so the banded solver suffices (no dense T); this
    # also avoids the uniform-width restart loop, whose arrowhead bookkeeping assumes a
    # constant block width p and is invalid once blocks have shrunk.
    if m_actual < m or deflated:
        if verbose and rank0:
            reason = "Invariant subspace" if m_actual < m else "Block deflation"
            print(f"[TRLM] {reason} (dim {total}). Extracting directly.")
        eigvals_T, eigvecs_T = eigh_block_tridiagonal(alphas, _betas_off, block_widths=widths)
        if comm is not None:
            eigvals_T = comm.bcast(eigvals_T, root=0)
            eigvecs_T = comm.bcast(eigvecs_T, root=0)
        wanted = np.argsort(eigvals_T)[:num_wanted]
        return eigvals_T[wanted], block_combine(Q_basis, eigvecs_T[:, wanted], slater)

    # The thick restart below builds an *arrowhead* T (a spike couples the retained Ritz
    # block to the residual), which is not banded; this path keeps the dense T_full.
    T_full = _build_full_T(alphas, _betas_off, block_widths=widths)

    # --- Width-aware thick restart -------------------------------------------------
    nkeep = k_blocks * p
    p_resid = _q_cols(q_m) if q_m is not None else p
    # betas[-1] is the trailing coupling, padded to (p, p) by the kernel. The residual block
    # can deflate (rank p_resid < p) even when the diagonal blocks do not, leaving total ==
    # m_actual*p (so we still reach here); slice off the phantom padded rows.
    beta_res = betas[len(betas) - 1][:p_resid, :]
    cur_widths = list(widths)

    for restart in range(max_restarts):
        D = int(sum(cur_widths))
        p_last = cur_widths[len(cur_widths) - 1]
        eigvals_T, eigvecs_T = sp.eigh(T_full[:D, :D])
        if comm is not None:
            eigvals_T = comm.bcast(eigvals_T, root=0)
            eigvecs_T = comm.bcast(eigvecs_T, root=0)
        res_norms = np.linalg.norm(beta_res @ eigvecs_T[D - p_last : D, :], axis=0)
        wanted = np.argsort(eigvals_T)[:num_wanted]
        max_res = float(np.max(res_norms[wanted]))

        if verbose and rank0:
            print(f"[TRLM] Restart {restart:3d} | MinEigval={eigvals_T[0]:.6f} | MaxWantedRes={max_res:.2e}")

        done = max_res < tol
        if mpi:
            done = comm.bcast(done, root=0)
        if done:
            if verbose and rank0:
                print("[TRLM] Converged!")
            break

        keep = np.argsort(eigvals_T)[:nkeep]
        Y_k = eigvecs_T[:, keep]
        Y_last = Y_k[D - p_last : D, :]  # last-block rows -> thick-restart spike
        T_k = np.diag(eigvals_T[keep])

        Q_ret = block_combine(_q_slice(Q_basis, 0, D), Y_k, 0.0)
        Q_ret, _ = block_normalize(Q_ret, mpi, comm, 0.0)

        # Worst case the continuation adds (m - k_blocks) full-width-p blocks.
        T_full = np.zeros((nkeep + (m - k_blocks) * p, nkeep + (m - k_blocks) * p), dtype=complex)
        T_full[:nkeep, :nkeep] = T_k

        # Residual seed block (carried over as q_m, or recomputed if absent).
        if q_m is None:
            q_seed = _q_slice(Q_ret, max(0, nkeep - p), nkeep)
            wp = block_apply(h_op, q_seed, basis, mpi, slater)
            # Thick-restart always full-reorthogonalizes the residual seed against the whole
            # retained basis (all modes): the arrowhead T_full requires it, and the PRO
            # W-recurrence is not maintained across restart.
            for _ in range(2):
                wp, _ = block_orthogonalize(wp, Q_ret, mpi=mpi, comm=comm)
            try:
                q_m, beta_res = block_normalize(wp, mpi, comm, 0.0)
            except (sp.LinAlgError, ValueError):
                q_m = None
            if q_m is None or np.linalg.norm(beta_res, ord=2) < 1e-5:
                if verbose and rank0:
                    print(f"[TRLM] Invariant subspace found at restart {restart}. Stopping early.")
                return _trlm_extract(T_full, Q_ret, nkeep, num_wanted, comm, slater)
            p_resid = _q_cols(q_m)

        Q_basis = _q_concat(Q_ret, _copy_block(q_m))
        cross = beta_res @ Y_last  # (p_resid, nkeep)
        T_full[nkeep : nkeep + p_resid, :nkeep] = cross
        T_full[:nkeep, nkeep : nkeep + p_resid] = np.conj(cross.T)

        cur_widths = [nkeep, p_resid]
        off = nkeep  # column start of the current block q1
        w1 = p_resid
        q1 = q_m
        q_m = None  # consumed; the new trailing residual is set at the last inner step

        for i in range(k_blocks, m):
            wp = block_apply(h_op, q1, basis, mpi, slater)

            overlaps = block_inner(Q_basis, wp, mpi, comm)
            alpha_i = overlaps[overlaps.shape[0] - w1 :, :]  # q1^H H q1  (w1, w1)
            T_full[off : off + w1, off : off + w1] = alpha_i

            # First pass reuses the overlaps already formed for alpha_i (wp is unchanged), so
            # this is the same projection the per-pass recompute would give; the second pass
            # recomputes against the now-cleaned wp.
            wp, _ = block_orthogonalize(wp, Q_basis, overlaps=overlaps, mpi=mpi, comm=comm)
            wp, _ = block_orthogonalize(wp, Q_basis, mpi=mpi, comm=comm)

            try:
                q_next, beta_i = block_normalize(wp, mpi, comm, 0.0)
            except (sp.LinAlgError, ValueError):
                q_next = None

            # Full collapse, or a near-invariant subspace: extract from what we have.
            if q_next is None or np.linalg.norm(beta_i, ord=2) < 1e-5:
                if verbose and rank0:
                    print(f"[TRLM] Invariant subspace found during restart at block {i}. Stopping early.")
                return _trlm_extract(T_full, Q_basis, off + w1, num_wanted, comm, slater)

            # Partial deflation shrinks the block: beta_i is (w_next, w1), q_next has
            # w_next <= w1 columns; place the arrowhead with those widths.
            w_next = _q_cols(q_next)
            if i < m - 1:
                T_full[off + w1 : off + w1 + w_next, off : off + w1] = beta_i
                T_full[off : off + w1, off + w1 : off + w1 + w_next] = np.conj(beta_i.T)
                Q_basis = _q_concat(Q_basis, _copy_block(q_next))
                cur_widths.append(w_next)
                off += w1
                w1 = w_next
                q1 = q_next
            else:
                beta_res = beta_i
                q_m = q_next
                p_resid = w_next

    # --- Final extraction -----------------------------------------------
    final_eigvals, final_eigvecs = _trlm_extract(T_full, Q_basis, int(sum(cur_widths)), num_wanted, comm, slater)
    if verbose and rank0:
        print(f"[TRLM] Final eigvals:\n{final_eigvals}")
    return final_eigvals, final_eigvecs


def _thick_restart_block_lanczos_array(
    psi0, h_op, basis, num_wanted, max_subspace_blocks, tol, max_restarts, verbose, reort_mode, comm
):
    """Array-path entry point: prepares the ``(N, p)`` start block and the
    ``block_lanczos_array`` sweep, then delegates to the shared :func:`_trlm_core`."""
    from impurityModel.ed.BlockLanczosArray import block_lanczos_array

    mpi = comm is not None and getattr(comm, "size", 1) > 1
    # block_lanczos_array assumes an orthonormal start block (it does not normalize
    # internally); normalize here so the betas do not grow geometrically and overflow T.
    psi0 = np.ascontiguousarray(psi0 if psi0.ndim == 2 else np.reshape(psi0, (-1, 1)), dtype=complex)
    psi0, _ = block_normalize(psi0, mpi, comm, 0.0)

    def sweep(v0, max_iter):
        res = block_lanczos_array(
            psi0=v0,
            h_op=h_op,
            converged=lambda a, b, **kw: False,
            max_iter=max_iter,
            verbose=verbose,
            reort=reort_mode,
            return_W=False,
            return_widths=True,
            comm=comm,
        )
        # (alphas, betas, Q, block_widths)
        return res[0], res[1], res[2], res[3]

    return _trlm_core(
        psi0, h_op, basis, num_wanted, max_subspace_blocks, tol, max_restarts,
        verbose, reort_mode, 0.0, comm, sweep,
    )


def thick_restart_block_lanczos_cy(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0.0,
    truncation_threshold: int = 0,
    reort="partial",
    comm=None,
):
    """Thick-restart block Lanczos (TRLM) for the ManyBodyState path.

    The MBS-only entry point: prepares the length-``p`` state block and the
    ``block_lanczos_cy`` sweep, then runs the shared path-agnostic :func:`_trlm_core`.

    Args:
        psi0: Starting block of ``p`` ``ManyBodyState`` objects.
        h_op: ``ManyBodyOperator`` Hamiltonian implementing ``apply_multi(psis, cutoff)``.
        basis: ``Basis`` providing ``redistribute_psis`` and ``basis.comm``.
        num_wanted: Number of wanted lowest eigenvalues.
        max_subspace_blocks: Maximum Krylov subspace size in blocks
            (``> ceil(num_wanted / p)``).
        tol: Convergence tolerance on the maximum wanted residual. Default ``1e-8``.
        max_restarts: Maximum number of thick restarts. Default ``100``.
        verbose: Print restart diagnostics. Default ``True``.
        slaterWeightMin: Amplitude cutoff for ``ManyBodyState.prune``. Default ``0.0``.
        reort: Reorthogonalization mode (``Reort`` enum or string). Default ``'partial'``.
        comm: ``mpi4py`` communicator. Falls back to ``basis.comm`` or serial.

    Returns:
        tuple[numpy.ndarray, list]: ``(eigvals, eigvecs)`` — the ``num_wanted`` smallest
        eigenvalues (ascending) and matching ``ManyBodyState`` Ritz vectors.
    """
    if comm is None:
        comm = getattr(basis, "comm", None)
    mpi = comm is not None and comm.Get_size() > 1
    reort_mode = resolve_reort(reort)

    psi0 = list(psi0) if isinstance(psi0, (list, tuple)) else psi0
    psi0, _ = block_normalize(psi0, mpi, comm, slaterWeightMin)

    def sweep(v0, max_iter):
        r = reort_mode.name.lower() if hasattr(reort_mode, "name") else reort_mode
        res = block_lanczos_cy(
            psi0=v0,
            h_op=h_op,
            basis=basis,
            converged_fn=lambda a, b, **kw: False,
            verbose=verbose,
            reort=r,
            max_iter=max_iter,
            slaterWeightMin=slaterWeightMin,
            truncation_threshold=truncation_threshold,
            comm=comm,
            return_widths=True,
        )
        # (alphas, betas, Q_basis, W, block_widths) -> drop W
        return res[0], res[1], res[2], res[4]

    return _trlm_core(
        psi0, h_op, basis, num_wanted, max_subspace_blocks, tol, max_restarts,
        verbose, reort_mode, slaterWeightMin, comm, sweep,
    )


def thick_restart_block_lanczos(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0,
    reort=Reort.PARTIAL,
):
    """Thick-restart block Lanczos (TRLM), dispatching on the operator type.

    Routes to the array path (dense/sparse ``h_op``) or the ManyBodyState path
    (``ManyBodyOperator``); both share the path-agnostic :func:`_trlm_core`.

    Returns:
        tuple[numpy.ndarray, list | numpy.ndarray]: ``(eigvals, eigvecs)``.
    """
    mpi = basis is not None and getattr(basis, "comm", None) is not None
    comm = basis.comm if mpi else None
    reort_mode = resolve_reort(reort)

    if not is_array(h_op):
        return thick_restart_block_lanczos_cy(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=tol,
            max_restarts=max_restarts,
            verbose=verbose,
            slaterWeightMin=slaterWeightMin,
            reort=reort_mode,
            comm=comm,
        )

    return _thick_restart_block_lanczos_array(
        psi0, h_op, basis, num_wanted, max_subspace_blocks, tol, max_restarts, verbose, reort_mode, comm
    )


# ===========================================================================
# Implicitly Restarted Block Lanczos (IRLM) \u2014 EA16 (Meerbergen & Scott,
# RAL-TR-2000-011). The whole business logic lives here (Cython); the Python
# module impurityModel.ed.irlm is a thin re-export. The core is path-agnostic:
# both the dense-array and ManyBodyState paths run through _irlm_core via the
# is_array-dispatching block_* primitives.
# ===========================================================================

# --- Path-agnostic basis helpers. The array path represents a basis as an
# ``(N, k)`` ndarray (column blocks); the ManyBodyState path as a length-``k``
# list of states. ---------------------------------------------------------
def _q_cols(Q):
    return Q.shape[1] if is_array(Q) else len(Q)


def _q_slice(Q, a, b):
    return Q[:, a:b] if is_array(Q) else Q[a:b]


def _q_concat(A, B):
    return np.concatenate([A, B], axis=1) if is_array(A) else (list(A) + list(B))


def _copy_block(V):
    return V.copy() if is_array(V) else [s.copy() for s in V]


def _implicitly_restarted_block_lanczos_array(
    psi0,
    h_op,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol,
    max_restarts,
    verbose,
    reort_mode,
    comm,
    cntl2=None,
    cntl3=0.0,
    locked_reort="full",
):
    """Array-path entry point: prepares the ``(N, p)`` start block and the sweep
    callable, then delegates to the shared :func:`_irlm_core`. See that function for
    the EA16 algorithm description."""
    from impurityModel.ed.BlockLanczosArray import block_lanczos_array

    mpi = comm is not None and comm.size > 1
    psi0 = np.ascontiguousarray(psi0 if psi0.ndim == 2 else psi0.reshape(-1, 1), dtype=complex)
    psi0, _ = block_normalize(psi0, mpi, comm, 0.0)

    def sweep(v0, max_iter, alphas=None, betas=None, Q=None, W=None, reort=None,
              locked=None, locked_evals=None, locked_res=0.0):
        res = block_lanczos_array(
            psi0=v0,
            h_op=h_op,
            converged=lambda a, b, **kw: False,
            max_iter=max_iter,
            verbose=verbose,
            reort=reort if reort is not None else reort_mode,
            return_W=True,
            return_widths=True,
            comm=comm,
            alphas=alphas,
            betas=betas,
            Q=Q,
            W=W,
            locked=locked,
            locked_evals=locked_evals,
            locked_res=locked_res,
            locked_reort=locked_reort,
        )
        # (alphas, betas, Q, W, block_widths)
        return res[0], res[1], res[2], res[3], res[4]

    return _irlm_core(
        psi0,
        basis,
        num_wanted,
        max_subspace_blocks,
        tol,
        max_restarts,
        verbose,
        reort_mode,
        comm,
        sweep,
        slater=0.0,
        cntl2=cntl2,
        cntl3=cntl3,
        tag="IRLM-array",
    )


def _implicitly_restarted_block_lanczos_manybody(
    psi0,
    h_op,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol,
    max_restarts,
    verbose,
    reort_mode,
    comm,
    slaterWeightMin=0.0,
    truncation_threshold=0,
    cntl2=None,
    cntl3=0.0,
    locked_reort="full",
):
    """ManyBodyState-path entry point: prepares the length-``p`` state block and the
    sweep callable (the Cython ``block_lanczos_cy`` kernel), then delegates to the
    shared :func:`_irlm_core`. Bit-for-bit consistent with the array path."""
    mpi = comm is not None and comm.size > 1
    psi0 = list(psi0) if isinstance(psi0, (list, tuple)) else psi0
    psi0, _ = block_normalize(psi0, mpi, comm, slaterWeightMin)

    def sweep(v0, max_iter, alphas=None, betas=None, Q=None, W=None, reort=None,
              locked=None, locked_evals=None, locked_res=0.0):
        r = reort if reort is not None else reort_mode
        r = r.name.lower() if hasattr(r, "name") else r
        res = block_lanczos_cy(
            psi0=v0,
            h_op=h_op,
            basis=basis,
            converged_fn=lambda a, b, **kw: False,
            verbose=verbose,
            reort=r,
            max_iter=max_iter,
            slaterWeightMin=slaterWeightMin,
            truncation_threshold=truncation_threshold,
            comm=comm,
            alphas_init=alphas,
            betas_init=betas,
            Q_init=Q,
            W_init=W,
            return_widths=True,
            locked=locked,
            locked_evals=locked_evals,
            locked_res=locked_res,
            locked_reort=locked_reort,
        )
        # (alphas, betas, Q, W, block_widths)
        return res[0], res[1], res[2], res[3], res[4]

    return _irlm_core(
        psi0,
        basis,
        num_wanted,
        max_subspace_blocks,
        tol,
        max_restarts,
        verbose,
        reort_mode,
        comm,
        sweep,
        slater=slaterWeightMin,
        cntl2=cntl2,
        cntl3=cntl3,
        tag="IRLM-mbs",
    )


def _irlm_core(
    psi0,
    basis,
    num_wanted,
    max_subspace_blocks,
    tol,
    max_restarts,
    verbose,
    reort_mode,
    comm,
    sweep,
    slater,
    cntl2,
    cntl3,
    tag,
):
    """Path-agnostic EA16 implicitly restarted block Lanczos (locking + purging).

    Faithful to Meerbergen & Scott, RAL-TR-2000-011 (EA16) for the standard
    real-symmetric/Hermitian problem in regular mode. Drives both the array and
    ManyBodyState paths through the path-agnostic ``block_*`` helpers and a path-specific
    ``sweep`` callable. Combines the block Lanczos recurrence (``sweep``, with
    resumption); **locking** of converged Ritz pairs (\u00a72.2.2); **explicit purging**
    (\u00a72.2.1, eq. 6) as the restart compression (``ea16.purge_restart``); and the EA16
    eq. (15) acceptance test (\u00a73.2.4),
    ``res <= u*||T_k|| + |CNTL(2)| + |CNTL(3)|*|theta|``.

    Returns:
        tuple: ``(eigvals, eigvecs)`` \u2014 sorted-ascending eigenvalues (length
        ``num_wanted``) and the matching Ritz vectors in the path's basis representation.
    """
    from impurityModel.ed import ea16

    mpi = comm is not None and comm.size > 1
    rank0 = (not mpi) or comm.rank == 0
    u = ea16.EPS
    if cntl2 is None:
        cntl2 = tol

    is_arr = is_array(psi0)
    p = (psi0.shape[1] if psi0.ndim == 2 else 1) if is_arr else len(psi0)
    m = max_subspace_blocks
    k0 = int(np.ceil(num_wanted / p))
    if m <= k0:
        raise ValueError("max_subspace_blocks must be strictly greater than ceil(num_wanted / p).")

    Xl = np.zeros((psi0.shape[0], 0), dtype=complex) if is_arr else []
    theta_l = []

    def _nlock():
        return Xl.shape[1] if is_arr else len(Xl)

    def _orth_against_locked(V):
        if _nlock() == 0:
            return V
        for _ in range(2):
            V, _ = block_orthogonalize(V, Xl, mpi=mpi, comm=comm)
        return V

    def _lock_block(X, vals):
        """Lock the columns of ``X`` (Ritz vectors) one at a time, reorthogonalizing each
        against the running locked set (\u00a72.6.2) and skipping any column that collapses \u2014
        i.e. is already represented in ``Xl``. Stops at ``num_wanted``."""
        nonlocal Xl
        n_locked_now = 0
        for j in range(_q_cols(X)):
            if len(theta_l) >= num_wanted:
                break
            col = _orth_against_locked(_q_slice(X, j, j + 1))
            g = block_inner(col, col, mpi, comm)
            if float(np.abs(g[0, 0])) < 1e-16:
                continue
            try:
                col, _ = block_normalize(col, mpi, comm, slater)
            except ValueError:
                # The column collapsed under block_normalize's stricter deflation floor
                # (DEFLATE_TOL ~ sqrt(eps)): it is already represented in the locked set.
                # block_normalize reduces M with a collective Allreduce, so every rank
                # raises together and skipping is MPI-collective-safe.
                continue
            Xl = np.concatenate([Xl, col], axis=1) if is_arr else (list(Xl) + list(col))
            theta_l.append(float(vals[j]))
            n_locked_now += 1
        return n_locked_now

    def _locked_kwargs():
        # Locked Ritz pairs to deflate the inner sweep against (EA16 \u00a72.6.2). Empty locked
        # set -> locked=None so the kernel skips it. ``locked_evals``/``locked_res`` feed
        # the \u00a72.6.2 overlap estimate (used only by the "partial" locked-reort mode).
        if _nlock() == 0:
            return {"locked": None}
        return {
            "locked": Xl,
            "locked_evals": np.asarray(theta_l, dtype=float),
            "locked_res": float(cntl2),
        }

    # --- Initial Lanczos run --------------------------------------------
    alphas, betas, Q_basis, _W, widths = sweep(psi0, m, **_locked_kwargs())

    for restart in range(max_restarts):
        n_need = num_wanted - len(theta_l)
        if n_need <= 0:
            break
        m_act = len(alphas)
        if m_act < 1:
            break

        # The sweep may shrink blocks (rank-deficient beta -> deflation), so the true
        # subspace dimension is sum(block_widths), not m_act * p. Build T against the
        # real widths so T (and its eigenvectors Z) line up with the stored Q_basis.
        total = int(sum(widths)) if widths is not None else m_act * p
        _betas_off = betas[: m_act - 1] if len(betas) == m_act else betas
        # Banded eigensolve straight from the block coefficients (no dense T); the IRLM
        # implicit-QR restart keeps T block-tridiagonal, so the band is valid.
        evals, Z = eigh_block_tridiagonal(alphas, _betas_off, block_widths=widths)
        if mpi:
            evals = comm.bcast(evals, root=0)
            Z = comm.bcast(Z, root=0)

        order = np.argsort(evals.real)
        if total < m_act * p:
            # Sweep deflated: the block Krylov recurrence broke down into a
            # (near-)invariant subspace, so the uniform-width purge/restart below does
            # not apply. Stop and let _assemble_results pull the lowest wanted Ritz
            # pairs from this width-consistent factorization.
            if verbose and rank0:
                print(f"[{tag}] Restart {restart:3d} | sweep deflated to dim {total} (<{m_act * p}); extracting & stopping.")
            break

        beta_last = betas[m_act - 1]
        res = ea16.ritz_residual_norms(beta_last, Z, p)
        tnorm = ea16.operator_norm_estimate(evals, theta_l)

        wanted = order[:n_need]
        if rank0:
            locked_local = [int(i) for i in wanted if res[i] <= ea16.acceptance_tol(evals[i], tnorm, cntl2, cntl3, u)]
        else:
            locked_local = None
        if mpi:
            locked_local = comm.bcast(locked_local, root=0)

        if verbose and rank0:
            print(
                f"[{tag}] Restart {restart:3d} | locked={len(theta_l)} | "
                f"MinEig={evals[order[0]].real:.6f} | "
                f"MaxWantedRes={float(np.max(res[wanted])):.2e} | newly_locked={len(locked_local)}"
            )

        # --- Lock converged wanted pairs (\u00a72.2.2) ----------------------
        if locked_local:
            X_new = block_combine(_q_slice(Q_basis, 0, total), Z[:, locked_local], slater)
            _lock_block(X_new, [evals[i].real for i in locked_local])
            n_need = num_wanted - len(theta_l)
            if n_need <= 0:
                break

        k_blocks = int(np.ceil(n_need / p))
        n_keep = k_blocks * p
        if total - n_keep < p:
            v0 = _orth_against_locked(_copy_block(psi0))
            try:
                v0, _ = block_normalize(v0, mpi, comm, slater)
            except ValueError:
                # The start block, projected orthogonal to the locked Ritz vectors, has
                # collapsed: psi0's Krylov space lies entirely within the already-locked
                # subspace, so no further wanted pairs are reachable. Collective-safe break.
                break
            alphas, betas, Q_basis, _W, widths = sweep(v0, m, **_locked_kwargs())
            continue

        # --- Purge + restart in the Ritz basis (EA16 \u00a72.2.1, eq. 6) ----
        kept_idx, _ = ea16.select_restart_indices(evals, n_keep, locked_local, which="smallest")
        C, beta_new, alphas_new, betas_new = ea16.purge_restart(evals, Z, beta_last, p, kept_idx)
        Q_used = _q_slice(Q_basis, 0, total)
        Q_new = block_combine(Q_used, C, slater)
        if _nlock() > 0:
            Q_new = _orth_against_locked(Q_new)

        # The trailing residual block can itself be rank-deficient when the sweep reached
        # a (near-)invariant subspace without shrinking any *diagonal* block, so the
        # alpha-width guard above (total < m_act * p) did not fire. Its stored width is
        # then < p and the Sorensen residual rotation qres @ beta_new is undefined. Lock
        # the lowest wanted Ritz pairs and stop.
        res_width = _q_cols(Q_basis) - total
        if res_width < p:
            X_rem = block_combine(_q_slice(Q_basis, 0, total), Z[:, order], slater)
            _lock_block(X_rem, [evals[i].real for i in order])
            if verbose and rank0:
                print(f"[{tag}] Restart {restart:3d} | trailing residual block deflated (width {res_width}<{p}). Locking remaining & stopping.")
            break

        # The trailing normalized residual block is always present after a sweep (the
        # recurrence stores m_act+1 blocks), so the Sorensen residual reduces to rotating
        # it by the re-banding coupling beta_new (EA16 \u00a72.2.1).
        qres = _q_slice(Q_basis, total, total + p)
        f_plus = block_combine(qres, beta_new, slater)
        f_plus = _orth_against_locked(f_plus)

        M = block_inner(f_plus, f_plus, mpi, comm)
        if np.any(np.isnan(M)) or np.any(np.isinf(M)):
            if verbose and rank0:
                print(f"[{tag}] Breakdown at restart -- returning current Ritz pairs.")
            break

        beta_k, beta_k_inv, active_k = _cholesky_or_deflate(M, p)
        if active_k < p:
            # Trailing block deflated => near-invariant subspace. Lock the lowest wanted
            # Ritz pairs (ascending; collapses against Xl are skipped) and stop.
            X_rem = block_combine(_q_slice(Q_basis, 0, total), Z[:, order], slater)
            _lock_block(X_rem, [evals[i].real for i in order])
            if verbose and rank0:
                print(f"[{tag}] Restart-block deflation (active_k={active_k}). Locking remaining & stopping.")
            break

        q_k_next = block_combine(f_plus, beta_k_inv, slater)
        Q_basis_new = _q_concat(Q_new, q_k_next)

        betas_pass_list = list(betas_new) if len(betas_new) > 0 else []
        if len(betas_pass_list) < k_blocks:
            betas_pass_list.append(beta_k)
        else:
            betas_pass_list[len(betas_pass_list) - 1] = beta_k
        betas_pass = np.array(betas_pass_list) if betas_pass_list else np.empty((0, p, p), dtype=complex)
        alphas_pass = np.array(alphas_new)

        # Restart-PRO continuation: continue in PARTIAL/SELECTIVE, seeding the Paige-Simon
        # estimator W at REORT_TOL (EA16 \u00a72.6.3). NONE/PERIODIC/FULL restart with FULL reort.
        if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
            W_init = np.zeros((2, k_blocks + 1, p, p), dtype=complex)
            W_init[1, :k_blocks] = REORT_TOL
            if k_blocks >= 2:
                W_init[0, : k_blocks - 1] = REORT_TOL
            W_init[1, k_blocks] = np.eye(p)
            W_init[0, k_blocks - 1] = np.eye(p)
            reort_continuation = reort_mode
        else:
            W_init = None
            reort_continuation = Reort.FULL

        alphas, betas, Q_basis, _W, widths = sweep(
            None,
            m - k_blocks,
            alphas=alphas_pass,
            betas=betas_pass,
            Q=Q_basis_new,
            W=W_init,
            reort=reort_continuation,
            **_locked_kwargs(),
        )

    # --- Final extraction -----------------------------------------------
    return _assemble_results(
        Xl, theta_l, alphas, betas, Q_basis, num_wanted, p, mpi, comm, slater, is_arr, widths
    )


def _assemble_results(
    Xl, theta_l, alphas, betas, Q_basis, num_wanted, p, mpi, comm, slater, is_arr, widths=None
):
    """Combine locked pairs with the best remaining active Ritz pairs, sorted ascending.

    Active Ritz candidates are deflated against the locked set (and each other) and any
    that collapse are skipped, so the result never contains duplicate eigenpairs. May
    return **fewer than ``num_wanted``** pairs when the reachable invariant subspace is
    smaller than ``num_wanted``; callers must use ``len(eigvals)``.
    """
    eigvals_list = list(theta_l)
    nlock = Xl.shape[1] if is_arr else len(Xl)
    eigvecs_cols = [_q_slice(Xl, j, j + 1) for j in range(nlock)]

    n_need = num_wanted - len(eigvals_list)
    if n_need > 0 and len(alphas) > 0:
        m_act = len(alphas)
        total = int(sum(widths)) if widths is not None else m_act * p
        _betas_off = betas[: m_act - 1] if len(betas) == m_act else betas
        # Banded eigensolve straight from the block coefficients (no dense T).
        evals, Z = eigh_block_tridiagonal(alphas, _betas_off, block_widths=widths)
        if mpi:
            evals = comm.bcast(evals, root=0)
            Z = comm.bcast(Z, root=0)
        # Deflate active Ritz candidates against the locked set (and against each other)
        # before accepting them, so near-copies of locked Ritz vectors are not accepted as
        # spurious duplicate eigenpairs.
        accepted = Xl
        Q_used = _q_slice(Q_basis, 0, total)
        for k in np.argsort(evals.real):
            if len(eigvals_list) >= num_wanted:
                break
            col = block_combine(Q_used, Z[:, k : k + 1], slater)
            n_acc = accepted.shape[1] if is_arr else len(accepted)
            if n_acc > 0:
                for _ in range(2):
                    col, _ = block_orthogonalize(col, accepted, mpi=mpi, comm=comm)
            g = block_inner(col, col, mpi, comm)
            if float(np.abs(g[0, 0])) < 1e-12:
                continue
            try:
                col, _ = block_normalize(col, mpi, comm, slater)
            except ValueError:
                continue
            eigvals_list.append(float(evals[k].real))
            eigvecs_cols.append(col)
            accepted = np.concatenate([accepted, col], axis=1) if is_arr else (list(accepted) + list(col))

    eigvals = np.array(eigvals_list[:num_wanted])
    order = np.argsort(eigvals)
    cols = eigvecs_cols[:num_wanted]
    if is_arr:
        eigvecs = np.concatenate(cols, axis=1) if cols else np.zeros((0, 0))
        eigvecs = eigvecs[:, order]
    else:
        flat = [c[0] if isinstance(c, list) else c for c in cols]
        eigvecs = [flat[i] for i in order]
    return eigvals[order], eigvecs


def implicitly_restarted_block_lanczos(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0.0,
    reort=None,
    comm=None,
    locked_reort: str = "full",
):
    """Implicitly-restarted block Lanczos (IRLM), dispatching on the operator type.

    Finds the ``num_wanted`` algebraically smallest eigenvalues of ``h_op`` via the EA16
    block Lanczos algorithm with locking, explicit purging, partial reorthogonalization
    against locked Ritz vectors, and the eq. (15) stopping criterion. Routes to the array
    path (dense/sparse ``h_op``) or the ManyBodyState path (``ManyBodyOperator``); both
    share the path-agnostic :func:`_irlm_core`.

    Returns:
        tuple[numpy.ndarray, list | numpy.ndarray]: ``(eigvals, eigvecs)``.
    """
    if reort is None:
        reort_mode = Reort.PARTIAL
    elif isinstance(reort, Reort):
        reort_mode = reort
    else:
        reort_mode = Reort[str(reort).upper()]

    if comm is None:
        comm = getattr(basis, "comm", None)

    if is_array(h_op):
        return _implicitly_restarted_block_lanczos_array(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            num_wanted=num_wanted,
            max_subspace_blocks=max_subspace_blocks,
            tol=tol,
            max_restarts=max_restarts,
            verbose=verbose,
            reort_mode=reort_mode,
            comm=comm,
            locked_reort=locked_reort,
        )

    return _implicitly_restarted_block_lanczos_manybody(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=num_wanted,
        max_subspace_blocks=max_subspace_blocks,
        tol=tol,
        max_restarts=max_restarts,
        verbose=verbose,
        reort_mode=reort_mode,
        comm=comm,
        slaterWeightMin=slaterWeightMin,
        locked_reort=locked_reort,
    )


def implicitly_restarted_block_lanczos_cy(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0.0,
    truncation_threshold: int = 0,
    reort="partial",
    comm=None,
):
    """Implicitly-restarted block Lanczos (IRLM) for the ManyBodyState path (EA16).

    The MBS-only entry point. Resolves ``reort`` and the communicator, then runs the
    shared path-agnostic :func:`_irlm_core` via the ManyBodyState sweep kernel
    ``block_lanczos_cy``.

    Returns:
        tuple[numpy.ndarray, list]: ``(eigvals, eigvecs)`` \u2014 ``num_wanted`` smallest
        eigenvalues (ascending) and matching ``ManyBodyState`` Ritz vectors.
    """
    if comm is None:
        comm = getattr(basis, "comm", None)
    if isinstance(reort, str):
        reort_mode = Reort[reort.upper()]
    else:
        reort_mode = reort

    return _implicitly_restarted_block_lanczos_manybody(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=num_wanted,
        max_subspace_blocks=max_subspace_blocks,
        tol=tol,
        max_restarts=max_restarts,
        verbose=verbose,
        reort_mode=reort_mode,
        comm=comm,
        slaterWeightMin=slaterWeightMin,
        truncation_threshold=truncation_threshold,
    )
