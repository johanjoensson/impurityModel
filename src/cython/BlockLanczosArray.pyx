# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True, freethreading_compatible=True

"""Block Lanczos kernel for numpy arrays / scipy sparse operators (MPI row-block).

This module is the numerical core shared with the ManyBodyState kernel in
``BlockLanczos.pyx`` (which imports the constants, ``estimate_orthonormality``,
``_build_full_T`` and ``apply_reort`` from here). Both build a block-tridiagonal
``T`` (diagonal blocks ``alpha``, off-diagonal blocks ``beta``) whose eigenpairs
approximate those of the operator; the TRLM/IRLM drivers restart it.

Reorthogonalization modes (``Reort``), what each guarantees:

* ``NONE``      — no reorthogonalization. Cheapest; orthogonality is allowed to
  decay and ghost (spurious duplicate) eigenvalues may appear. Eigenvalues still
  converge for well-separated spectra. The restart machinery in TRLM/IRLM
  re-orthogonalizes regardless, so NONE only relaxes the *inner* recurrence.
* ``FULL``      — double-pass Gram–Schmidt against the whole basis every step.
  Most robust; ``‖QᴴQ − I‖`` stays at ~machine precision.
* ``PERIODIC``  — FULL, but only every ``REORT_PERIOD`` iterations.
* ``PARTIAL``   — partial reorthogonalization (PRO, Simon): the W-recurrence
  ``estimate_orthonormality`` cheaply *estimates* loss of orthogonality; a full
  reorth against the offending blocks fires only when an estimate exceeds
  ``REORT_TOL``. Matches FULL accuracy at lower cost; the default.
* ``SELECTIVE`` — PARTIAL plus periodic locking of converged Ritz vectors
  (gated to a ``REORT_PERIOD`` cadence; see ``block_lanczos_array_cy``).

Every mode retains all ``p * m`` Krylov columns. That is the dominant allocation of a
long Green's-function run, and it cannot be compressed away: reorthogonalization needs
the *converged Ritz vectors*, and a Ritz vector converging at step ``m`` has weight on
every block, including any an earlier compression would have discarded. Bounding the
retained basis therefore needs secondary storage, not a smaller basis — see the
"Ritz compression is unsound" section of ``doc/plans/blocklanczos_reort_memory.md``.

Module thresholds (all derived from machine ``eps``; see definitions below):
``REORT_TOL`` (loss-of-orthogonality trigger), ``BAD_BLOCK_TOL`` (which blocks to
reorth against), ``DEFLATE_TOL`` (relative rank floor on the block's singular values),
``BREAKDOWN_TOL`` (invariant-subspace detection), ``REORT_PERIOD`` (cadence). The
deflation floors are defined in the ``TSQR`` module that applies them.

Orthonormalization goes through ``TSQR`` (``tsqr`` here, ``block_tsqr`` in ``_reort.pxi``
for the other block representations): the residual block's triangular factor is computed
from the block itself, never from the Gram matrix ``Wp^H Wp``, which is what makes it
stable at any conditioning and its singular values trustworthy.

Deflation policy: when a Lanczos block is rank-deficient (a singular value below the
``DEFLATE_TOL`` floor), the block size shrinks (EA16 shrinking-block, Meerbergen & Scott,
RAL-TR-2000-011) rather than zero-padding or terminating, so ``beta`` becomes rectangular
and ``T`` carries variable-size blocks; the recurrence keeps converging.
"""

cimport cython
import numpy as np
cimport numpy as np
import scipy.linalg as la
import scipy.sparse as sps

from scipy.linalg.cython_blas cimport zgemm

from mpi4py import MPI
from enum import Enum


class Reort(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2
    PERIODIC = 3
    SELECTIVE = 4


# The machine constant and the deflation floors are defined once, in the TSQR leaf that
# applies them; a second literal here would be a second source of truth. ``DEFLATE_TOL``
# itself is only read by callers, who import it straight from ``TSQR`` (cython-lint has no
# noqa escape for a pure re-export).
from impurityModel.ed.TSQR import (
    EPS,
    DEFLATE_EVAL_TOL,
    BREAKDOWN_TOL,
    tsqr,
)

cdef double EPS_VAL = EPS
REORT_TOL = np.sqrt(EPS_VAL)        # ~1.49e-8  : trigger — reorth when max|W| exceeds this
BAD_BLOCK_TOL = EPS_VAL ** 0.75        # ~1.83e-12 : selection — reorth against blocks above this
# Ritz vectors projected out per pass in selective_orthogonalize. Each pass costs one sweep
# of the Krylov store, so batching turns k sweeps into ceil(k/RITZ_BATCH) BLAS-3 sweeps; the
# batch bounds the transient Ritz block at (n_rows x RITZ_BATCH) instead of (n_rows x k),
# which matters because k grows with the number of converged Ritz pairs.
RITZ_BATCH = 8
# DEFLATE_TOL (~6.06e-6, the rank floor on the block's singular values), its squared
# counterpart DEFLATE_EVAL_TOL for tests on a Gram matrix, and BREAKDOWN_TOL are imported
# from TSQR above. The floor bounds the *retained* block's condition number by ~EPS**(-1/3)
# (~1.7e5); it was originally tightened to that from sqrt(EPS) because a marginally
# conditioned block let the O(cond) amplification of the per-step Allreduce's rank-order
# rounding accumulate, so the reort=NONE recurrence diverged under MPI but not serially.
BETA_BLOWUP_FACTOR = 1e3           # ||beta_i|| above this * max(||beta||, ||alpha||) ⇒ divergence
REORT_PERIOD = 5                   # PERIODIC cadence, and SELECTIVE Ritz-check cadence
# Semi-orthogonality threshold (Simon's classical sqrt(EPS) criterion — the very level the
# PARTIAL estimator maintains, hence the shared constant). TRLM's thick-restart coefficient
# shortcuts (diag(theta) for the retained block, beta_res @ Y_last for the spike) are both
# derived from Q^H Q = I; above this level that premise is too inaccurate to use them and the
# restart is rebuilt from actual matvecs instead. See _trlm_core.
RESTART_ORTH_TOL = REORT_TOL       # ~1.49e-8

# --- Optional reort instrumentation (env-gated; ~zero cost when off) ----------------
import os as _os_bla
_REORT_PROF_ON = _os_bla.environ.get("BLOCKLANCZOS_PROFILE") == "1"
_REORT_PROF = {}


def get_reort_profile():
    """Accumulated apply_reort stats: total calls, calls that acted, summed bad-block and
    bad-column counts (so the average fan-out of the selective reort can be inspected)."""
    return dict(_REORT_PROF)


def reset_reort_profile():
    _REORT_PROF.clear()


def enable_reort_profile(on=True):
    """Toggle the apply_reort instrumentation at runtime (equivalent to setting
    BLOCKLANCZOS_PROFILE=1 in the environment before import)."""
    global _REORT_PROF_ON
    _REORT_PROF_ON = bool(on)


def _cholesky_or_deflate(M, p_in, double scale=1.0):
    r"""QR-factor the residual block via its Gram matrix ``M = Wp^H Wp``.

    **Superseded** by :func:`impurityModel.ed.TSQR.tsqr`, which factors the block itself and
    is therefore stable at conditioning this cannot survive; no production path calls this
    any more. It is kept as the reference implementation the CholeskyQR2-era regression tests
    (``test_block_lanczos_blowup``, ``test_deflation_scale_invariance``,
    ``test_warm_restart_refines``) are written against — that record is the reason the
    deflation contract below reads the way it does, and TSQR inherited it unchanged.

    Returns ``(beta_j, beta_inv, k)`` with ``beta_j`` (``k x p_in``) the upper-triangular
    QR factor (off-diagonal block), ``beta_inv`` (``p_in x k``) such that
    ``Q = Wp @ beta_inv`` and ``Wp = Q @ beta_j``, and ``k`` the retained rank.

    Two *different* questions, deliberately kept apart.

    **Breakdown** — is the block numerically zero, i.e. is the block-Krylov space closed?
    That is an absolute statement, and it needs a reference: a block is zero relative to
    *something*.  ``scale`` supplies it, and ``k = 0`` is returned when
    :math:`\lVert\beta\rVert_2 = \sqrt{\lambda_{\max}(M)} \le` ``BREAKDOWN_TOL * scale``.
    Callers that normalize near-unit-norm vectors (:func:`block_normalize`, the CholeskyQR2
    second pass, ``block_bicgstab``'s pre-normalized Gram) leave ``scale = 1``.  The Lanczos
    sweeps pass the operator scale ``~||H||``, because a *residual* block is zero when it is
    negligible against ``H``, not against 1.

    **Rank deficiency** — are some column *directions* linearly dependent?  That is a
    statement about :math:`\lambda_{\min}/\lambda_{\max}` and must be judged *relative* to the
    block's own largest singular value: a direction is deflated when
    :math:`\sigma_k < \texttt{EPS}^{1/3}\sigma_{\max}`, i.e.
    :math:`\lambda_k < \texttt{EPS}^{2/3}\lambda_{\max}`.  This bounds the retained block's
    condition number to :math:`\lesssim \texttt{EPS}^{-1/3}` (~1.7e5), comfortably inside the
    regime where a second pass (CholeskyQR2, see :func:`_cholesky_qr2`) restores orthonormality
    to machine precision (which needs :math:`\kappa \lesssim \texttt{EPS}^{-1/2}`), and it is
    what stops the :math:`O(\kappa)` amplification of the per-step ``Allreduce`` rank-order
    rounding from accumulating into a divergence of the ``reort=NONE`` recurrence.

    The two used to be fused into ``evals > DEFLATE_EVAL_TOL * max(evals[-1], 1.0)``, whose
    ``1.0`` clamp turned the *rank* test absolute for any block below
    ``DEFLATE_TOL = EPS**(1/3)`` (~6.06e-6): a small but perfectly well-conditioned block was
    declared rank 0.  Every warm-started Krylov solve was then handed back its own input and
    reported success — ``block_bicgstab`` returned ``x0`` unrefined once ``||R0|| < 6e-6``, and
    TRLM warm-started from ``CIPSISolver.expand``'s eigenvectors returned them unimproved at
    ``||r|| = 2.2e-9``, capping the ground-state accuracy whatever ``tol`` asked for.
    ``BREAKDOWN_TOL`` existed but was referenced nowhere; invariant-subspace detection rode
    entirely on that clamp.

    Both the fast (Cholesky) and the fallback (``eigh``) path apply the *same* two tests —
    historically the fast path tested the Cholesky diagonal against ``sqrt(EPS)`` (an effective
    ``EPS`` eigenvalue floor) while the fallback tested eigenvalues against ``sqrt(EPS)`` (an
    ``EPS**0.25`` floor on singular values); that inconsistency let a near-singular ``M``
    through the fast path and produced a ``beta_inv`` with norm up to ``1/EPS``, amplifying the
    Krylov block and diverging the recurrence.

    Args:
        M: Hermitian positive-semidefinite Gram matrix ``Wp^H Wp`` (``p_in x p_in``).
        p_in: Column count of ``Wp``.
        scale: Reference for the breakdown test — the norm ``||beta||_2`` is compared against
            ``BREAKDOWN_TOL * scale``. Pass ``~||H||`` for a Lanczos residual block; leave at
            ``1.0`` when the columns are already O(1).
    """
    cdef double breakdown_lam = (BREAKDOWN_TOL * scale) ** 2
    # Try Cholesky first (fast path). diag(L)**2 are the (real, positive) Cholesky
    # pivots, an O(EPS)-faithful surrogate for the eigenvalues of the Hermitian-PD M.
    try:
        L = la.cholesky(M, lower=True)
        d2 = np.square(np.diag(L).real)
        d2_max = float(np.max(d2)) if d2.size else 0.0
        if d2_max <= breakdown_lam:                        # ||beta||_2 <= BREAKDOWN_TOL * scale
            return None, None, 0
        if np.any(d2 < DEFLATE_EVAL_TOL * d2_max):
            raise la.LinAlgError("Numeric singularity in Cholesky diagonal")
        beta_j = np.conj(L.T)
        beta_inv = la.inv(beta_j)
        p_next = p_in
        return beta_j, beta_inv, p_next
    except (la.LinAlgError, ValueError):
        # Fall back to eigh, using the same two tests as the fast path.
        evals, evecs = la.eigh(M)              # ascending
        lam_max = float(evals[evals.size - 1]) if evals.size else 0.0
        if lam_max <= breakdown_lam:                      # the whole block is numerically zero
            return None, None, 0
        keep = evals > DEFLATE_EVAL_TOL * lam_max         # boolean mask over p_in
        p_next = int(keep.sum())
        if p_next == 0:                                   # unreachable: lam_max always survives
            return None, None, 0
        V = evecs[:, keep]                                # (p_in, p_next)
        s = np.sqrt(evals[keep])                          # (p_next,)
        beta_j = (s[:, None] * np.conj(V.T))             # (p_next, p_in)   off-diag block
        beta_inv = V / s[None, :]                         # (p_in,  p_next)
        return beta_j, beta_inv, p_next


def _cholesky_qr2(M2, beta_j, active_k):
    r"""Second pass of CholeskyQR2 on the *recomputed* Gram of the once-QR'd block.

    **Superseded** together with :func:`_cholesky_or_deflate`, and for the same reason: the
    repetition below exists only to repair the first pass, and TSQR's first pass needs no
    repair (it repeats itself at all only above ``kappa ~ EPS**(-1/4)``, and can then never
    be *rescuing* a failed factorization). Kept for the regression tests.

    Cholesky-QR (``_cholesky_or_deflate``) of an ill-conditioned block leaves
    ``Q1 = Wp @ beta_inv`` with an orthonormality error of order
    :math:`\kappa(M)\,\texttt{EPS}`, which can be :math:`O(1)`.  Re-orthonormalizing once
    more — using the Gram ``M2 = Q1^H Q1`` recomputed from the *actual* (high-dimensional)
    vectors, not from ``M`` — drives the error back to :math:`O(\texttt{EPS})` as long as the
    deflated block satisfies :math:`\kappa \lesssim 1/\sqrt{\texttt{EPS}}` (guaranteed by the
    ``_cholesky_or_deflate`` floor).

    Given ``M2`` and the first-pass factor ``beta_j`` (so ``Wp = Q1 @ beta_j``), returns
    ``(beta2_inv, beta_j_new, k2)`` such that ``Q2 = Q1 @ beta2_inv`` is orthonormal and
    ``Wp = Q2 @ beta_j_new``.  Returns ``(None, None, 0)`` on full collapse.
    """
    beta2, beta2_inv, k2 = _cholesky_or_deflate(M2, active_k)
    if k2 == 0:
        return None, None, 0
    return beta2_inv, beta2 @ beta_j, k2


def resolve_reort(reort):
    """Resolve a ``reort`` argument (``Reort`` member or string) to a ``Reort`` enum.

    Shared by both kernels so the accepted spellings stay in sync. A non-string is
    returned unchanged. Raises ``ValueError`` on an unknown string.
    """
    if not isinstance(reort, str):
        return reort
    _map = {
        "none": Reort.NONE,
        "partial": Reort.PARTIAL,
        "selective": Reort.SELECTIVE,
        "full": Reort.FULL,
        "periodic": Reort.PERIODIC,
    }
    resolved = _map.get(reort.lower())
    if resolved is None:
        raise ValueError(f"Unknown reort string '{reort}'. Must be one of {list(_map.keys())}.")
    return resolved


def divergence_guard(double beta_norm, double alpha_norm, bint first_step,
                     double t_norm_max, double h_norm_est):
    r"""Spectral-scale divergence safeguard for the block-Lanczos recurrence.

    Shared by both kernels (``block_lanczos_array_cy`` and the ``block_lanczos_cy``
    driver) so the safeguard logic lives in exactly one place.

    For a Hermitian ``H`` every block norm is bounded by ``||H||``; a jump of several
    orders of magnitude over the largest healthy block norm means the recurrence has
    been corrupted (orthogonality the QR/deflation did not repair). ``h_norm_est`` is a
    spectral-scale estimate seeded on the first step (where ``beta ~ ||H||``) and grown
    only by ``||alpha_i||`` (also bounded by ``||H||``, and which never runs away with
    beta) — so it catches *gradual* runaway growth that the relative ``t_norm_max``
    jump-check misses. ``first_step`` (the scale-establishing step of this run) never
    self-triggers.

    Parameters
    ----------
    beta_norm, alpha_norm : float
        2-norms of the current ``beta_i`` / ``alpha_i`` blocks.
    first_step : bool
        ``True`` on the first step of this run (``t_norm_max == 0``).
    t_norm_max, h_norm_est : float
        Running trackers (carried across steps by the caller).

    Returns
    -------
    diverged : bool
        ``True`` if the recurrence has diverged and must be truncated *before* this block.
    t_norm_max, h_norm_est : float
        Updated trackers (``t_norm_max`` unchanged when ``diverged`` — the caller breaks).
    """
    if first_step:
        h_norm_est = max(beta_norm, alpha_norm)
    else:
        h_norm_est = max(h_norm_est, alpha_norm)
    diverged = (not first_step) and (
        max(beta_norm, alpha_norm) > BETA_BLOWUP_FACTOR * max(t_norm_max, 1.0)
        or beta_norm > BETA_BLOWUP_FACTOR * max(h_norm_est, 1.0)
    )
    if not diverged:
        t_norm_max = max(t_norm_max, beta_norm, alpha_norm)
    return diverged, t_norm_max, h_norm_est


def calculate_thermal_gs(h, block_size, e_max, v0=None, reort=Reort.FULL, comm=None):
    mpi = comm is not None
    rank = comm.rank if mpi else 0
    size = comm.size if mpi else 1

    def converged(alphas, betas, *args, **kwargs):
        e, s = eigsh(alphas, betas, de=e_max, select="m")
        sorted_indices = np.argsort(e)
        e = e[sorted_indices]
        s = s[:, sorted_indices]
        mask = e - np.min(e) <= e_max
        return np.linalg.norm(betas[-1] @ s[-block_size:, mask], ord=2) < 1e-6

    if v0 is None:
        v0 = np.random.rand(h.shape[1], block_size) + 1j * np.random.rand(h.shape[1], block_size)
        if mpi:
            counts = np.empty((size), dtype=int)
            comm.Allgather(np.array([v0.size], dtype=int), counts)
            offsets = np.array([np.sum(counts[:r]) for r in range(size)], dtype=int)
            v0_full = np.empty((h.shape[0], block_size), dtype=complex, order="C") if rank == 0 else None
            comm.Gatherv(v0, (v0_full, counts, offsets, MPI.C_DOUBLE_COMPLEX), root=0)
            if rank == 0:
                v0_full, _ = la.qr(v0_full, mode="economic", overwrite_a=True, check_finite=False)
            comm.Scatterv((v0_full, counts, offsets, MPI.C_DOUBLE_COMPLEX), v0, root=0)
        else:
            v0, _ = la.qr(v0, mode="economic", overwrite_a=True, check_finite=False)
    elif v0.shape[1] < block_size:
        new_v0 = np.random.rand(v0.shape[0], block_size - v0.shape[1]) + 1j * np.random.rand(
            v0.shape[0], block_size - v0.shape[1]
        )
        new_v0 -= v0 @ np.conj(v0.T) @ new_v0
        new_v0, _ = la.qr(new_v0, mode="economic", overwrite_a=True, check_finite=False)
        v0 = np.append(v0, new_v0, axis=1)
    elif v0.shape[1] > block_size:
        v0 = v0[:, :block_size]

    alphas, betas, Q, *_ = block_lanczos_array_cy(
        psi0=v0,
        h_op=h,
        converged=converged,
        verbose=True,
        reort=reort,
        comm=comm,
    )
    if rank == 0:
        eigvals, eigvecs = eigsh(alphas, betas, de=e_max, Q=Q[:, : alphas.shape[0] * alphas.shape[1]], select="m")
    else:
        eigvals = None
        eigvecs = None
    if mpi:
        eigvals = comm.bcast(eigvals, root=0)
        eigvecs = comm.bcast(eigvecs, root=0)
    return eigvals, eigvecs


cpdef np.ndarray estimate_orthonormality(
    np.ndarray[double complex, ndim=4] W,
    np.ndarray[double complex, ndim=3] alphas,
    np.ndarray[double complex, ndim=3] betas,
    object block_widths=None,
    double eps=0.0,
    double N=1.0,
    object out=None,
    object beta_norms=None,
):
    cdef int i = alphas.shape[0] - 1
    cdef int n = alphas.shape[1]
    if eps == 0.0:
        eps = np.finfo(float).eps

    # Rounding-accumulation scale: a matvec/orthogonalization over an N-dimensional
    # state accumulates ~N rounding errors whose sum grows like sqrt(N) (random-walk;
    # the same convention as the locked-reort floor eps*p*sqrt(N) in the drivers).
    # Callers pass N = the global problem dimension; the historical default N=1
    # under-scaled the floor by sqrt(N) (measured ~10x at N=670).
    cdef double n_scale = np.sqrt(N) if N > 1.0 else 1.0

    cdef list widths
    if block_widths is None:
        widths = [n] * (i + 2)
    else:
        widths = list(block_widths)

    cdef int w_curr = widths[i]
    cdef int w_i = w_curr
    cdef int w_next = widths[i+1]
    cdef int w_0 = widths[0]

    # Bounded-W: the caller may provide a persistent buffer (`out`, shape
    # (2, >=i+2, n, n)); the estimate is built into its zeroed leading view instead of a
    # fresh allocation every step. The caller must ping-pong two buffers (the previous
    # estimate `W` is read while the new one is written, so they must not alias).
    cdef np.ndarray[double complex, ndim=4] W_out
    if out is None:
        W_out = np.zeros((2, i + 2, n, n), dtype=complex)
    else:
        W_out = out[:, : i + 2]
        W_out[...] = 0
    # Build the new estimate directly into W_out[1] (a zero-initialized view) instead of a
    # separate w_bar buffer that is then copied — saves one (i+2, n, n) allocation + copy/step.
    cdef np.ndarray[double complex, ndim=3] w_bar = W_out[1]

    w_bar[i + 1, :w_next, :w_next] = np.identity(w_next)

    cdef np.ndarray beta_i_dag_inv = np.conj(la.pinv(betas[i, :w_next, :w_curr]).T)  # shape (w_next, w_curr)

    # omega_{i+1,i}: forming q_{i+1} = wp @ beta_i^-1 injects rounding of size eps*sqrt(N)*||H||
    # (the scale of the matvec that produced wp) and the normalization amplifies it by
    # ||beta_i^-1||. Simon (1984) writes this as eps*||A||/beta_i.
    #
    # The scale factor must bound ||H||, NOT ||beta_0||. Those coincide for a *cold* start --
    # a random q_0 has ||H q_0|| ~ ||H|| -- and the old code exploited that, writing the term as
    # ``eps * n_scale * beta_i^-H @ betas[0]``. Warm-started they do not: from converged
    # eigenvectors ||beta_0|| is the eigenpair residual (measured 2.2e-9 on the NiO ground state
    # against ||H|| ~ 1.1e2), and at i = 0 the expression collapses to ``beta_0^-H @ beta_0 ~ I``
    # -- an estimate of eps, where the true overlap is eps*||H||/||beta_0|| ~ 1e-5. The trigger
    # never fired, PARTIAL silently did no reorthogonalization at all, and the recurrence
    # diverged ~30 steps later (measured ||Q^H Q - I|| = 11.3, ||beta|| 8x FULL's).
    #
    # So take the operator scale directly: max(||alpha_0||, ||beta_0||), the same seed
    # divergence_guard uses for h_norm_est. alpha_0 is a Rayleigh quotient, hence bounded by
    # ||H|| and O(||H||) for a warm start; beta_0 covers a cold start whose spectrum straddles
    # zero. On a cold start the two forms agree in magnitude and the reort trigger rate is
    # unchanged (measured identical: 38/120, 8/60, 9/60 blocks acted).
    cdef double anorm = max(
        float(np.linalg.norm(alphas[0, :w_0, :w_0], ord=2)),
        float(np.linalg.norm(betas[0, :widths[1], :w_0], ord=2)),
    )
    # A magnitude, like the noise floor below and unlike the signed three-term propagation:
    # this term models a rounding *injection*, which has no sign structure to cancel.
    w_bar[i, :w_next, :w_curr] = eps * n_scale * anorm * np.abs(beta_i_dag_inv)

    if i == 0:
        W_out[0, : i + 1] = W[1]  # w_bar is already W_out[1] (built in place)
        return W_out

    # j = 0
    cdef int w_j = widths[0]
    cdef int w_j_next = widths[1]
    cdef int w_i_prev = widths[i-1]
    # Propagate the estimate through the SIGNED three-term recurrence (Paige/Simon; EA16
    # eq. 14). The signs matter: the O(||beta||) structural terms (e.g. W[1,j+1] beta_j vs
    # beta_{i-1} W[0,j] around the identity entries) cancel to O(eps ||beta||) exactly as
    # the true overlaps do — that cancellation IS the physics of the recurrence. A
    # magnitude version (sum of |terms|, tried 2026-06 as an "upper bound") destroys it:
    # the estimate jumps to O(||beta||/sigma_min) ~ O(1) after a single step and then
    # compounds exponentially (measured 1e15–1e62 x over-prediction on the NiO ground
    # state), so every block is flagged on every iteration and PARTIAL silently does FULL
    # work. The rounding injection the magnitudes were meant to capture is modeled
    # explicitly by the sqrt(N)-scaled noise floor below instead.
    cdef np.ndarray term1 = W[1, 1, :w_i, :w_j_next] @ betas[0, :w_j_next, :w_j]
    cdef np.ndarray term2 = W[1, 0, :w_i, :w_j] @ alphas[0, :w_j, :w_j]
    cdef np.ndarray term3 = alphas[i, :w_i, :w_i] @ W[1, 0, :w_i, :w_j]
    cdef np.ndarray term5 = betas[i-1, :w_i, :w_i_prev] @ W[0, 0, :w_i_prev, :w_j]
    cdef np.ndarray RHS_0 = term1 + term2 - term3 - term5
    w_bar[0, :w_next, :w_j] = beta_i_dag_inv @ RHS_0

    cdef int j, w_j_prev
    cdef np.ndarray term4
    cdef np.ndarray RHS
    for j in range(1, i):
        w_j = widths[j]
        w_j_prev = widths[j-1]
        w_j_next = widths[j+1]

        term1 = W[1, j+1, :w_i, :w_j_next] @ betas[j, :w_j_next, :w_j]
        term2 = W[1, j, :w_i, :w_j] @ alphas[j, :w_j, :w_j]
        term3 = alphas[i, :w_i, :w_i] @ W[1, j, :w_i, :w_j]
        term4 = W[1, j-1, :w_i, :w_j_prev] @ np.conj(betas[j-1, :w_j, :w_j_prev].T)
        term5 = betas[i-1, :w_i, :w_i_prev] @ W[0, j, :w_i_prev, :w_j]

        RHS = term1 + term2 - term3 + term4 - term5
        w_bar[j, :w_next, :w_j] = beta_i_dag_inv @ RHS

    # Local-rounding noise floor (Simon 1984 / Larsen PROPACK). Forming q_{i+1} = w_p beta_i^{-1}
    # injects rounding ~eps*sqrt(N)*(||beta_i||+||beta_j||) that the normalization amplifies by
    # ||beta_i^{-1}|| = 1/sigma_min(beta_i): when beta_i is small (a near-invariant block) the new
    # vector is rounding-dominated and orthogonality is lost fastest, so the floor must *grow* as
    # beta_i shrinks. An earlier version `eps*(beta_i + beta_j)` omitted the 1/sigma_min factor (and
    # shrank with beta_i), so the estimate vanished exactly when the true loss was worst -> the
    # bad-block trigger never fired and PARTIAL degenerated to no reorthogonalization. The floor is
    # added as a positive magnitude (no sign cancellation): it, not the signed propagation above,
    # carries the per-step rounding injection, and with the sqrt(N) scale it upper-bounds the
    # measured true loss by a stable ~2-10x on the NiO ground-state workload.
    # One SVD of the current beta gives both the 2-norm (largest singular value) and
    # sigma_min — np.linalg.norm(ord=2) computes the same SVD internally, so this is
    # bit-identical and drops a redundant factorization per step.
    _sv_bi = la.svd(betas[i, :w_next, :w_curr], compute_uv=False)
    _sig_min_bi = float(_sv_bi[len(_sv_bi) - 1])
    _binv_norm = 1.0 / max(_sig_min_bi, eps)
    _bnorm_i = float(_sv_bi[0])
    for j in range(i):
        # The past ||beta_j||_2 never change; the caller may pass its running history
        # (`beta_norms`, one entry per completed block) instead of re-factorizing every
        # previous beta on every step (O(k^2) SVDs over a run without it).
        if beta_norms is not None and j < len(beta_norms) and beta_norms[j] is not None:
            _bnorm_j = float(beta_norms[j])
        else:
            _bnorm_j = float(np.linalg.norm(betas[j, :widths[j+1], :widths[j]], ord=2))
        w_bar[j, :w_next, :widths[j]] += eps * n_scale * (_bnorm_i + _bnorm_j) * _binv_norm

    W_out[0, : i + 1] = W[1]  # w_bar is already W_out[1] (built in place)

    return W_out


cpdef np.ndarray _build_full_T(np.ndarray[double complex, ndim=3] alphas, np.ndarray[double complex, ndim=3] betas, object block_widths=None, object comm=None):
    cdef int m = alphas.shape[0]
    if m == 0:
        return np.zeros((0, 0), dtype=complex)

    cdef list widths
    if block_widths is None:
        widths = [alphas.shape[1]] * m
    else:
        widths = list(block_widths)

    cdef int total_dim = sum(widths)
    cdef np.ndarray[double complex, ndim=2] T = np.zeros((total_dim, total_dim), dtype=complex)

    cdef list offsets = [0]
    cdef int off = 0
    cdef object w_val
    for w_val in widths:
        off += int(w_val)
        offsets.append(off)

    cdef int i, w_i, w_next, o_i, o_next
    for i in range(m):
        w_i = int(widths[i])
        o_i = offsets[i]
        T[o_i : o_i + w_i, o_i : o_i + w_i] = alphas[i, :w_i, :w_i]
        if i < m - 1:
            w_next = int(widths[i+1])
            o_next = offsets[i+1]
            T[o_next : o_next + w_next, o_i : o_i + w_i] = betas[i, :w_next, :w_i]
            T[o_i : o_i + w_i, o_next : o_next + w_next] = np.conj(betas[i, :w_next, :w_i].T)
    return T


cpdef tuple _extract_blocks(np.ndarray[double complex, ndim=2] T, int m, int n):
    cdef np.ndarray[double complex, ndim=3] alphas = np.zeros((m, n, n), dtype=complex)
    cdef np.ndarray[double complex, ndim=3] betas = np.zeros((m - 1, n, n), dtype=complex)
    cdef int i
    for i in range(m):
        alphas[i] = T[i * n : (i + 1) * n, i * n : (i + 1) * n]
        if i < m - 1:
            betas[i] = T[(i + 1) * n : (i + 2) * n, i * n : (i + 1) * n]
    return alphas, betas


def _build_banded_lower(alphas, betas, widths):
    r"""LAPACK lower-banded storage of the *variable-width* block-tridiagonal T, assembled
    directly from the block coefficients — no dense matrix is ever formed.

    The full T (dimension :math:`\sum_i w_i`) is banded with lower bandwidth
    :math:`\max_i(w_i + w_{i+1} - 1)`: every nonzero is inside a diagonal block
    :math:`\alpha_i` (``w_i x w_i``) or an off-diagonal block :math:`\beta_i`
    (``w_{i+1} x w_i``, coupling block ``i+1`` to block ``i``, hence in the lower triangle).
    Returns ``a_band`` of shape ``(bw + 1, total)`` with ``a_band[d, j] == T[j + d, j]`` (the
    format ``scipy.linalg.eig_banded(..., lower=True)`` expects) and the bandwidth ``bw``.
    """
    widths = [int(w) for w in widths]
    m = len(widths)
    offsets = [0]
    for w in widths:
        offsets.append(offsets[-1] + w)
    total = offsets[-1]
    bw = 0
    for i in range(m):
        bw = max(bw, widths[i] - 1)
        if i < m - 1:
            bw = max(bw, widths[i] + widths[i + 1] - 1)
    a_band = np.zeros((bw + 1, total), dtype=complex)
    for i in range(m):
        wi = widths[i]
        oi = offsets[i]
        ai = np.asarray(alphas[i])[:wi, :wi]
        for d in range(wi):  # lower diagonals of the diagonal block alpha_i
            a_band[d, oi : oi + wi - d] = np.diagonal(ai, -d)
        if i < m - 1:
            wn = widths[i + 1]
            bi = np.asarray(betas[i])[:wn, :wi]
            # T[oi+wi+r, oi+c] = beta_i[r, c]  ->  band index (wi + r - c) at column (oi + c).
            # Vectorized scatter (no Python element loop over the block).
            rr, cc = np.indices((wn, wi))
            a_band[(wi + rr - cc).ravel(), (oi + cc).ravel()] = bi.ravel()
    return a_band, total


def eigh_block_tridiagonal(alphas, betas, block_widths=None, eigvals_only=False):
    r"""Eigen-decomposition of a (variable-width) block-tridiagonal T via the **banded** solver.

    Builds the lower-banded storage straight from the ``alphas``/``betas`` blocks (no dense T,
    see :func:`_build_banded_lower`) and calls ``scipy.linalg.eig_banded``. Use this instead of
    ``_build_full_T(...) + scipy.linalg.eigh(...)`` whenever T is a genuine block-tridiagonal
    (the Lanczos recurrence, an implicit-QR/IRLM restart) — i.e. *not* a thick-restart arrowhead,
    which is not banded and must stay dense.

    Args:
        alphas: Diagonal blocks ``(m, p, p)`` (or sequence of 2D blocks).
        betas: Sub-diagonal blocks; only the first ``m-1`` are used (the trailing residual is
            ignored, matching ``_build_full_T``).
        block_widths: True per-block widths; ``None`` => uniform ``p``.
        eigvals_only: If True, skip eigenvectors.

    Returns:
        tuple ``(evals, Z)`` with ascending real ``evals`` and eigenvectors ``Z`` of dimension
        ``sum(block_widths)`` (``Z`` is ``None`` when ``eigvals_only``).
    """
    cdef int m = (alphas.shape[0] if hasattr(alphas, "shape") else len(alphas))
    cdef int p = (alphas.shape[1] if hasattr(alphas, "shape") else np.asarray(alphas[0]).shape[0])
    widths = list(block_widths) if block_widths is not None else [p] * m
    a_band, _total = _build_banded_lower(alphas, betas, widths)
    if eigvals_only:
        return la.eig_banded(a_band, lower=True, eigvals_only=True, overwrite_a_band=True, check_finite=False), None
    evals, Z = la.eig_banded(a_band, lower=True, eigvals_only=False, overwrite_a_band=True, check_finite=False)
    return evals, Z


cpdef tuple eigsh(
    np.ndarray[double complex, ndim=3] alphas,
    np.ndarray[double complex, ndim=3] betas,
    object de=None,
    np.ndarray Q=None,
    bint eigvals_only=False,
    str select="a",
    object select_range=None,
    int max_ev=0,
    object comm=None,
    object block_widths=None,
):
    cdef bint within_gs = False
    if select == "m":
        assert de is not None
        select = "a"
        within_gs = True

    # One band builder for both uniform and shrinking-block-deflated T: assembled straight from
    # the block coefficients (no dense T) and honoring the true per-block widths, so deflated
    # blocks neither inject spurious zero eigenvalues nor break the Ritz reconstruction.
    cdef int _p = alphas.shape[1]
    cdef list _widths = list(block_widths) if block_widths is not None else [_p] * alphas.shape[0]
    a_band, total = _build_banded_lower(alphas, betas, _widths)
    cdef np.ndarray eigvals
    cdef np.ndarray eigvecs

    if eigvals_only:
        eigvals = np.sort(
            la.eig_banded(
                a_band,
                lower=True,
                eigvals_only=True,
                overwrite_a_band=True,
                check_finite=False,
                select=select,
                select_range=select_range,
                max_ev=max_ev,
            )
        )
        if within_gs:
            return (eigvals[eigvals - eigvals[0] <= de], None)
        return (eigvals, None)

    eigvals, eigvecs = la.eig_banded(
        a_band,
        lower=True,
        eigvals_only=False,
        overwrite_a_band=True,
        check_finite=False,
        select=select,
        select_range=select_range,
        max_ev=max_ev,
    )

    cdef np.ndarray mask
    if within_gs:
        mask = eigvals - np.min(eigvals) <= de
    else:
        mask = np.ones(len(eigvals), dtype=bool)

    cdef np.ndarray mask_indices = np.where(mask)[0]
    cdef np.ndarray sort_indices = np.argsort(eigvals[mask_indices])
    cdef np.ndarray final_indices = mask_indices[sort_indices]

    eigvals = eigvals[final_indices]
    eigvecs = eigvecs[:, final_indices]

    if Q is not None:
        # total == sum(widths): equals Q's columns when uniform, fewer when deflated.
        eigvecs = Q[:, :total] @ eigvecs

    return eigvals, eigvecs


cpdef np.ndarray block_combine_array(np.ndarray Q, np.ndarray Y):
    return Q @ np.ascontiguousarray(Y, dtype=complex)


cpdef tuple block_orthogonalize_array(np.ndarray wp, np.ndarray Q, object overlaps=None, object comm=None):
    if overlaps is None:
        overlaps = np.conj(Q.T) @ wp
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
    wp -= Q @ overlaps
    return wp, overlaps


cpdef tuple block_normalize_array(np.ndarray wp, object comm=None):
    """Orthonormalize a dense row-distributed block: ``wp = q_next @ beta_j``.

    The columns here are already O(1) (a starting block, a restart block), so the breakdown
    reference is 1 rather than an operator norm; a block that collapses raises instead of
    returning a rank, because its callers have no shrinking-block path to fall back on.
    """
    q_next, beta_j, active_k, _ = tsqr(wp, comm, 1.0)
    if active_k <= 0:
        raise ValueError("Block collapsed to zero rank")
    return q_next, beta_j

cdef extern from "complex.h":
    double complex conj(double complex z) nogil

# from scipy.linalg.cython_blas cimport zgemm, zgemv


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void matmul_nogil(
    int m, int n, int k,
    double complex alpha,
    double complex[:, ::1] A, char transA,
    double complex[:, ::1] B, char transB,
    double complex beta,
    double complex[:, ::1] C
) noexcept nogil:
    # C (m x n, row-major) = beta*C + alpha * opA(A) (m x k) @ opB(B) (k x n),
    # where op = conjugate-transpose when the trans flag is b'C'.
    # Implemented with BLAS-3 zgemm. BLAS is column-major; our arrays are C-contiguous
    # (row-major), so the row-major product C = opA(A) @ opB(B) is obtained by asking
    # BLAS for the column-major Cᵀ = opB(B)ᵀ @ opA(A)ᵀ: swap the operands (B then A),
    # swap (m, n) -> (n, m), keep the same trans flags, and use the physical row
    # lengths (A.shape[1], B.shape[1], n) as the leading dimensions. Validated against
    # numpy @ for all four trans combinations in test_zgemm_matmul (1e-13).
    cdef int i, j
    cdef int lda, ldb, ldc
    if m <= 0 or n <= 0:
        return
    if k <= 0:
        # No contraction: C = beta * C (zgemm with K=0 would still need valid A/B ptrs).
        if beta == 0.0:
            for i in range(m):
                for j in range(n):
                    C[i, j] = 0.0
        elif beta != 1.0:
            for i in range(m):
                for j in range(n):
                    C[i, j] = C[i, j] * beta
        return
    lda = A.shape[1]
    ldb = B.shape[1]
    ldc = n
    zgemm(&transB, &transA, &n, &m, &k, &alpha, &B[0, 0], &ldb, &A[0, 0], &lda, &beta, &C[0, 0], &ldc)


def _matmul_nogil_test(A, int transA, B, int transB, alpha, beta, C, int m, int n, int k):
    """Test-only Python wrapper around matmul_nogil (pass transA/transB as ord('N'/'C'))."""
    cdef double complex[:, ::1] Av = np.ascontiguousarray(A, dtype=complex)
    cdef double complex[:, ::1] Bv = np.ascontiguousarray(B, dtype=complex)
    cdef double complex[:, ::1] Cv = np.ascontiguousarray(C, dtype=complex)
    cdef char ta = transA
    cdef char tb = transB
    cdef double complex al = alpha
    cdef double complex be = beta
    matmul_nogil(m, n, k, al, Av, ta, Bv, tb, be, Cv)
    return np.asarray(Cv)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply_sparse_csr_nogil(
    int num_rows,
    int num_cols,
    int p,
    double complex[:] data,
    long[:] indices,
    long[:] indptr,
    double complex[:, ::1] X,
    double complex[:, ::1] Y
) noexcept nogil:
    cdef int i, j, k
    cdef long row_start, row_end
    cdef double complex val
    for i in range(num_rows):
        row_start = indptr[i]
        row_end = indptr[i+1]
        for k in range(p):
            Y[i, k] = 0.0
        for j in range(row_start, row_end):
            val = data[j]
            for k in range(p):
                Y[i, k] = Y[i, k] + val * X[indices[j], k]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply_dense_nogil(
    int M, int K, int p,
    double complex[:, ::1] H,
    double complex[:, ::1] X,
    double complex[:, ::1] Y
) noexcept nogil:
    matmul_nogil(M, p, K, 1.0, H, b'N', X, b'N', 0.0, Y)


def block_lanczos_array_cy(
    np.ndarray psi0,
    h_op,
    converged,
    verbose=False,
    reort=Reort.NONE,
    alphas=None,
    betas=None,
    Q=None,
    W=None,
    return_W=False,
    comm=None,
    return_widths=False,
    return_status=False,
    build_krylov_basis=None,
    **kwargs
):
    # Resolve a string reort (e.g. "full") to the Reort enum, mirroring
    # block_lanczos_cy; otherwise the `reort == Reort.FULL` comparisons below never
    # match a string and the build silently runs with no reorthogonalization.
    reort = resolve_reort(reort)

    # Whether to retain the accumulated Krylov basis. The historical default (True) stands:
    # restart machinery (TRLM/IRLM) and direct callers consume the returned Q regardless of
    # the reort mode. Callers that only need the final residual block (the GF continued
    # fraction with reort NONE) opt out explicitly, dropping the O(N * n * k) growth; any
    # reort mode projects against the basis and the warm-start protocol slices its blocks,
    # so opting out requires reort='none' and a fresh start.
    cdef bint keep_krylov
    if build_krylov_basis is None:
        keep_krylov = True
    else:
        keep_krylov = bool(build_krylov_basis)
        if not keep_krylov and (reort != Reort.NONE or Q is not None):
            raise ValueError("build_krylov_basis=False requires reort='none' and no warm-start Q")
    cdef int N = psi0.shape[0] if psi0 is not None else Q.shape[0]
    cdef int n = (psi0.shape[1] if psi0.ndim == 2 else 1) if psi0 is not None else alphas[0].shape[0]

    cdef int start_it = 0
    cdef np.ndarray alphas_arr, betas_arr
    cdef list Q_list
    # Krylov storage: Q_buf is an over-allocated column buffer grown geometrically
    # (amortized O(1) copies instead of one full np.concatenate reallocation per step);
    # Q_list[0] is always the filled Q_buf[:, :q_cols] view the reort machinery reads.
    cdef np.ndarray Q_buf = None
    cdef int q_cols = 0

    if alphas is not None and betas is not None and Q is not None:
        start_it = len(alphas)
        alphas_arr = alphas
        betas_arr = betas

        q0_fallback = psi0.copy() if psi0 is not None else Q[:, :n]
        q = [Q[:, (start_it - 1) * n : start_it * n] if start_it > 0 else q0_fallback]
        q.append(Q[:, start_it * n : (start_it + 1) * n] if start_it > 0 else q0_fallback)
        # Zero spare capacity: the first append reallocates, so the caller's Q is never
        # written to in place.
        Q_buf = Q
        q_cols = Q.shape[1]
        Q_list = [Q]
    else:
        start_it = 0
        alphas_arr = np.empty((0, n, n), dtype=complex)
        betas_arr = np.empty((0, n, n), dtype=complex)

        q = [np.zeros((N, n), dtype=complex, order='C')]
        q.append(np.ascontiguousarray(psi0 if psi0.ndim == 2 else psi0.reshape(-1, 1)))
        if keep_krylov:
            Q_buf = q[1].copy()
            q_cols = Q_buf.shape[1]
            Q_list = [Q_buf]
        else:
            Q_list = None

    cdef int period = kwargs.get("reort_period", 5)
    cdef int max_iter = kwargs.get("max_iter", int(np.ceil(h_op.shape[0] / n if sps.issparse(h_op) or isinstance(h_op, np.ndarray) else N / n)))
    cdef int _buf_size = start_it + max_iter
    cdef np.ndarray[double complex, ndim=3] alphas_buf = np.zeros((_buf_size, n, n), dtype=complex)
    cdef np.ndarray[double complex, ndim=3] betas_buf = np.zeros((_buf_size, n, n), dtype=complex)
    if start_it > 0:
        alphas_buf[:start_it] = alphas_arr
        betas_buf[:start_it] = betas_arr

    # Bounded-W ping-pong buffers + beta-norm history (Phase 1), mirroring
    # block_lanczos_cy: the estimator writes into the spare persistent buffer and
    # reuses the completed blocks' ||beta_j||_2 instead of re-factorizing them.
    _w_bufs = None
    beta_norm_hist = None
    if reort in (Reort.PARTIAL, Reort.SELECTIVE):
        _w_bufs = (
            np.empty((2, _buf_size + 2, n, n), dtype=complex),
            np.empty((2, _buf_size + 2, n, n), dtype=complex),
        )
        beta_norm_hist = [None] * start_it

    cdef bint is_sparse = sps.issparse(h_op)
    cdef bint is_dense = isinstance(h_op, np.ndarray)

    cdef int it = start_it
    cdef double t_norm_max = 0.0
    cdef double h_norm_est = 0.0
    cdef double _h_scale = 0.0
    cdef double beta_norm, alpha_norm
    cdef double reort_eps = np.sqrt(np.finfo(float).eps)

    cdef bint mpi = comm is not None
    cdef int rank = comm.rank if mpi else 0
    cdef int size = comm.size if mpi else 1
    cdef np.ndarray counts, offsets
    cdef int global_N = N

    if mpi:
        counts = np.empty((size), dtype=int)
        comm.Allgather(np.array([N], dtype=int), counts)
        offsets = np.array([np.sum(counts[:r]) for r in range(size)], dtype=int)
        global_N = np.sum(counts)

    cdef double complex[:] h_data
    cdef long[:] h_indices
    cdef long[:] h_indptr
    cdef double complex[:, ::1] h_dense

    if is_sparse:
        h_op = h_op.tocsr()
        h_data = h_op.data
        h_indices = np.ascontiguousarray(h_op.indices, dtype=np.int64)
        h_indptr = np.ascontiguousarray(h_op.indptr, dtype=np.int64)
    elif is_dense:
        h_dense = np.ascontiguousarray(h_op)

    cdef np.ndarray wp_arr = np.empty((N, n), dtype=complex, order='C')
    cdef double complex[:, ::1] wp = wp_arr
    cdef double complex[:, ::1] q1, q0, beta_prev_dag_mv
    cdef double complex[:, ::1] alpha_i

    # GUARDRAIL: the MPI matvec forms the full (global_N, n) partial product on *every*
    # rank (column-distributed H -> Allreduce -> slice local rows), so per-rank memory
    # here scales with global_N, not local_N. This is intentional: the array kernel is
    # for small/dense sectors. For a large global_N use the sparse hash-distributed
    # kernel (BlockLanczos.pyx / block_lanczos_cy), which never forms a dense global
    # vector. We deliberately do NOT halo-exchange this (see blocklanczos_blas_
    # acceleration.md §3, "WON'T FIX"): the dense path already OOMs on the global_N^2
    # matrix first, and the memory-bound CIPSI/GF case runs on the sparse kernel anyway.
    cdef np.ndarray wp_global = np.empty((global_N, n), dtype=complex, order='C') if mpi else None
    cdef double complex[:, ::1] wp_g = wp_global

    cdef list block_widths = list(kwargs.get("block_widths_init", [n] * start_it))
    cdef int n_curr, n_prev, active_k

    # EA16 §2.6.2 locking deflation: keep every Lanczos vector orthogonal to the
    # already-converged ("locked") Ritz vectors. Without this the matvec keeps
    # amplifying the dominant locked directions back into the active subspace,
    # which reintroduces locked eigenvalues (and their 2*theta harmonics) as
    # spurious Ritz values *below* the true spectral minimum on restarted sweeps.
    # `locked` is column-distributed exactly like the Krylov vectors (local rows).
    cdef np.ndarray locked_arr = kwargs.get("locked", None)
    cdef np.ndarray locked_ovl
    cdef bint have_locked = locked_arr is not None and locked_arr.shape[1] > 0
    if have_locked:
        locked_arr = np.ascontiguousarray(locked_arr, dtype=complex)

    # Locking-deflation mode: "full" (default) unconditionally projects every Lanczos
    # vector against the locked set each step; "partial" implements the estimate-driven
    # EA16 §2.6.2 scheme — a cheap per-pair overlap recurrence (no O(N) inner products)
    # that reorthogonalizes only the locked vectors whose estimated overlap exceeds
    # omega_TOL. The default preserves the previous behaviour exactly.
    cdef str locked_reort = kwargs.get("locked_reort", "full")
    cdef bint partial_locked = have_locked and locked_reort == "partial"
    cdef np.ndarray locked_evals_arr = kwargs.get("locked_evals", None)
    cdef double locked_rho = float(kwargs.get("locked_res", REORT_TOL))
    cdef double omega_min_l = EPS * n * np.sqrt(global_N)
    cdef np.ndarray xi_l, xi_prev_l, xi_mask, q_next_ovl
    cdef double bj_inv_norm, bjm1_norm
    cdef int nlock = locked_arr.shape[1] if have_locked else 0
    if partial_locked:
        if locked_evals_arr is None:
            raise ValueError("locked_reort='partial' requires 'locked_evals'")
        locked_evals_arr = np.ascontiguousarray(np.real(locked_evals_arr), dtype=float)
        xi_l = np.full(nlock, omega_min_l)
        xi_prev_l = np.full(nlock, omega_min_l)
    from impurityModel.ed import ea16 as _ea16

    # Why the recurrence stops (returned when return_status=True), mirroring block_lanczos_cy:
    # "converged" (converged() satisfied), "invariant_subspace" (rank-deficient residual => the
    # block-Krylov space is closed under H => result is *exact*), "diverged" (non-finite Gram or
    # the divergence guard truncated a corrupted tail => NOT exact) or "max_iter" (budget spent
    # before the recurrence terminated naturally).
    termination = "max_iter"

    # Two-consecutive-steps reort rule (Simon/PROPACK): when a step acts, the next step
    # bypasses the REORT_TOL gate (see apply_reort force=) so q_curr's remaining overlap
    # cannot silently re-contaminate the recurrence through the three-term coupling.
    _force_reort = False

    while it < _buf_size:
        q1 = np.ascontiguousarray(q[1])
        n_curr = q1.shape[1]

        # Re-allocate wp buffers to match current active width for contiguous alignment
        if wp_arr.shape[1] != n_curr:
            wp_arr = np.empty((N, n_curr), dtype=complex, order='C')
            wp = wp_arr
            if mpi:
                wp_global = np.empty((global_N, n_curr), dtype=complex, order='C')
                wp_g = wp_global

        if is_sparse:
            with nogil:
                apply_sparse_csr_nogil(global_N, N, n_curr, h_data, h_indices, h_indptr, q1, wp_g if mpi else wp)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, wp_global, op=MPI.SUM)
                wp_arr[:] = wp_global[offsets[rank] : offsets[rank] + N, :]
        elif is_dense:
            with nogil:
                apply_dense_nogil(global_N, N, n_curr, h_dense, q1, wp_g if mpi else wp)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, wp_global, op=MPI.SUM)
                wp_arr[:] = wp_global[offsets[rank] : offsets[rank] + N, :]
        else:
            if hasattr(h_op, "dot"):
                wp_arr[:] = h_op.dot(q1)
            elif hasattr(h_op, "__matmul__"):
                wp_arr[:] = h_op @ q1
            else:
                wp_arr[:] = h_op(q1)

        # Allocate alpha_i as a contiguous array
        alpha_i_arr = np.empty((n_curr, n_curr), dtype=complex, order='C')
        alpha_i = alpha_i_arr
        with nogil:
            matmul_nogil(n_curr, n_curr, N, 1.0, q1, b'C', wp, b'N', 0.0, alpha_i)

        if mpi:
            comm.Allreduce(MPI.IN_PLACE, alpha_i_arr, op=MPI.SUM)

        alphas_buf[it, :n_curr, :n_curr] = alpha_i_arr

        with nogil:
            matmul_nogil(N, n_curr, n_curr, -1.0, q1, b'N', alpha_i, b'N', 1.0, wp)

        if it > 0:
            n_prev = q[0].shape[1]
            beta_prev_dag_arr = np.conj(betas_buf[it - 1, :n_curr, :n_prev].T).copy()
            beta_prev_dag_arr_c = np.ascontiguousarray(beta_prev_dag_arr)
            beta_prev_dag_mv = beta_prev_dag_arr_c
            q0 = np.ascontiguousarray(q[0])
            with nogil:
                matmul_nogil(N, n_curr, n_prev, -1.0, q0, b'N', beta_prev_dag_mv, b'N', 1.0, wp)

        # "full" locking deflation: unconditionally project against the locked set every
        # step (twice for robustness), for every reort mode. The "partial" mode skips this
        # and instead does the estimate-driven EA16 §2.6.2 reorth on q_next below.
        if have_locked and not partial_locked:
            for _ in range(2):
                locked_ovl = np.ascontiguousarray(locked_arr.conj().T @ wp_arr)
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, locked_ovl, op=MPI.SUM)
                wp_arr -= locked_arr @ locked_ovl
            wp = wp_arr

        if reort == Reort.FULL or (reort == Reort.PERIODIC and it > 0 and it % period == 0):
            if not keep_krylov:
                raise RuntimeError("Krylov basis must be built for reorthogonalization")
            wp_arr, _, _ = apply_reort(wp_arr, Q_list, None, Reort.FULL, mpi, comm, block_widths)

        # TSQR of the residual block: the triangular factor is built from the block itself
        # (panel Householder + Givens merges + one Allgather), never from wp^H wp, so it is
        # backward stable at any conditioning and the rank decision below is made on the
        # block's true singular values. A *residual* block is negligible when it is small
        # against H, not against 1, so the breakdown reference is the divergence guard's
        # running spectral-scale estimate (max of the block norms seen so far, all bounded by
        # ||H||), widened by this step's alpha since the guard has not run yet. On the first
        # step of a cold start h_norm_est is still 0 and alpha_0 carries the scale.
        _h_scale = max(h_norm_est, t_norm_max, float(np.linalg.norm(alpha_i_arr, ord=2)))
        q_next, beta_i, active_k, sv_i = tsqr(wp_arr, comm if mpi else None, _h_scale)
        if active_k < 0:
            # Non-finite factor => corrupted recurrence, truncated (NOT exact).
            termination = "diverged"
            block_widths.append(n_curr)
            it += 1
            break
        if active_k == 0:
            # Rank-deficient residual => the block-Krylov space is closed => exact.
            termination = "invariant_subspace"
            block_widths.append(n_curr)
            it += 1
            break

        betas_buf[it, :active_k, :n_curr] = beta_i

        # --- Divergence safeguard ---------------------------------------
        # For a Hermitian H every block norm is bounded by ||H||; a jump of several orders
        # of magnitude over the largest healthy block norm means the recurrence has been
        # corrupted past repair. Truncate *before* this block (exclude alpha_i/beta_i at
        # index it) so the diverged tail never reaches the Green's function. Mirrors the
        # guard in block_lanczos_cy.
        # ||beta_i||_2 comes out of the factorization: beta_i and the block share their
        # singular values, so sv_i[0] is the 2-norm (reused by the SELECTIVE Ritz-error check
        # below) with no SVD of its own.
        beta_norm = float(sv_i[0])
        _reort_acted = False
        alpha_norm = np.linalg.norm(alpha_i_arr, ord=2)
        diverged, t_norm_max, h_norm_est = divergence_guard(
            beta_norm, alpha_norm, it == start_it, t_norm_max, h_norm_est
        )
        if diverged:
            if verbose and (not mpi or comm.rank == 0):
                print(
                    f"[BlockLanczos] Divergence detected at iteration {it}: "
                    f"|beta|={beta_norm:.3e}, |alpha|={alpha_norm:.3e} >> spectral scale "
                    f"{h_norm_est:.3e}. Truncating to the last trustworthy block.",
                    flush=True,
                )
            termination = "diverged"
            break

        # EA16 §2.6.2 estimate-driven locking reorthogonalization. Propagate the cheap
        # per-pair overlap estimate (no O(N) work) and reorthogonalize q_next only against
        # the locked vectors whose estimate now exceeds omega_TOL, resetting those to
        # omega_min. ||B_j^{-1}|| = 1/sigma_min of the retained block, ||B_{j-1}|| =
        # ||betas_buf[it-1]||_2.
        if partial_locked:
            bj_inv_norm = 1.0 / float(sv_i[active_k - 1])
            bjm1_norm = float(np.linalg.norm(betas_buf[it - 1], 2)) if it > 0 else 0.0
            xi_new_l, xi_trigger, xi_mask = _ea16.locked_overlap_step(
                xi_l, xi_prev_l, locked_evals_arr, alpha_i_arr,
                bj_inv_norm, bjm1_norm, locked_rho, REORT_TOL, BAD_BLOCK_TOL, EPS,
            )
            if xi_trigger:
                idx_l = np.nonzero(xi_mask)[0]
                Lm = np.ascontiguousarray(locked_arr[:, idx_l])
                for _ in range(2):
                    q_next_ovl = np.ascontiguousarray(Lm.conj().T @ q_next)
                    if mpi:
                        comm.Allreduce(MPI.IN_PLACE, q_next_ovl, op=MPI.SUM)
                    q_next = q_next - Lm @ q_next_ovl
                xi_new_l[idx_l] = omega_min_l
                xi_l[idx_l] = omega_min_l
            xi_prev_l = xi_l
            xi_l = xi_new_l

        if reort in (Reort.PARTIAL, Reort.SELECTIVE):
            if not keep_krylov:
                raise RuntimeError("Krylov basis must be built for reorthogonalization")
            if W is None:
                if start_it > 0:
                    W = np.zeros((2, start_it + 1, n, n), dtype=complex)
                    for j in range(start_it):
                        w_j = block_widths[j]
                        Q_j = Q_list[0][:, sum(block_widths[:j]) : sum(block_widths[:j+1])]
                        W[1, j, :w_j, :n_curr] = Q_j.conj().T @ wp_arr
                        if j < start_it - 1:
                            W[0, j, :w_j, :n_prev] = Q_j.conj().T @ q[0]
                    if mpi:
                        comm.Allreduce(MPI.IN_PLACE, W, op=MPI.SUM)
                    W[1, start_it, :n_curr, :n_curr] = np.eye(n_curr)
                    W[0, start_it - 1, :n_prev, :n_prev] = np.eye(n_prev)
                else:
                    W = np.zeros((2, 1, n, n), dtype=complex)
                    W[1, 0, :n_curr, :n_curr] = np.eye(n_curr)
            elif W.shape[1] < it + 1:
                W_new = np.zeros((2, it + 1, n, n), dtype=complex)
                W_new[:, : W.shape[1]] = W
                W = W_new

            block_widths.append(n_curr)
            block_widths.append(active_k)
            W = estimate_orthonormality(
                W, alphas_buf[: it + 1], betas_buf[: it + 1], block_widths=block_widths, eps=EPS, N=global_N,
                out=_w_bufs[it % 2] if _w_bufs is not None else None, beta_norms=beta_norm_hist,
            )
            block_widths.pop()
            block_widths.pop()

            reort_eps = REORT_TOL

            if reort == Reort.SELECTIVE:
                # EA16 §2.6.2 selective orthogonalization (shared with block_lanczos_step_cy).
                # Q_list[0] holds all it+1 completed blocks (the current block q1 is already its
                # last block), spanning the full subspace the Ritz vectors index. beta_norm is the
                # already-computed ||beta_i||_2.
                q_next = selective_orthogonalize(
                    q_next, Q_list[0], alphas_buf, betas_buf, W, block_widths,
                    it, n_curr, beta_norm, reort_eps, REORT_PERIOD, mpi, comm,
                )

            if reort in (Reort.PARTIAL, Reort.SELECTIVE):
                # Pass block_widths + [n_curr] so the current block (index it, already
                # appended to Q_list) is in the width table apply_reort indexes — matching
                # block_lanczos_cy. Without it the W-recurrence's bad-block columns are
                # mis-mapped, which silently loses orthogonality when resuming on a
                # restarted (e.g. IRLM-compressed) basis.
                q_next, W, _reort_acted = apply_reort(
                    q_next, Q_list, W, reort, mpi, comm, block_widths + [n_curr], force=_force_reort
                )
                _force_reort = _reort_acted
                # Only when the bad-block projection actually changed q_next does it need
                # re-orthonormalizing; when no block was projected q_next is unchanged and this
                # would be an exact no-op (M2 == I), so skip it and save the Gram + MPI Allreduce.
                if _reort_acted:
                    # The projection acts on the post-QR block, so it leaves q_next non-orthonormal
                    # and beta_i inconsistent; on a near-redundant block (tight clusters) it removes
                    # almost all of q_next, and that residual must be renormalized (or legitimately
                    # deflated) before it feeds the next iteration — else the recurrence amplifies
                    # the tiny leftover and the basis blows up. Re-factoring folds the correction
                    # into beta_i: wp = (q_next @ R2) @ beta_i.
                    q_next_2, R2, active_k, sv2 = tsqr(q_next, comm if mpi else None, 1.0)
                    # An *absolutely* tiny residual => the new block is fully contained in the
                    # existing Krylov span (invariant subspace, e.g. a tight cluster whose
                    # effective rank is exhausted). Renormalizing that noise would amplify rounding
                    # into the recurrence (beta blow-up), so stop here. sqrt(EPS) is the largest
                    # column norm the old max(diag(q_next^H q_next)) < EPS test admitted.
                    if active_k <= 0 or float(sv2[0]) < np.sqrt(EPS):
                        termination = "invariant_subspace"
                        block_widths.append(n_curr)
                        it += 1
                        break
                    beta_i = R2 @ beta_i
                    q_next = q_next_2
                    # The renormalization rescales q_next's true overlaps with every
                    # Krylov column by up to ||R2^{-1}||_2 = 1/sigma_min; propagate the same
                    # bound into the just-written honest post-reort W estimates (a no-op when
                    # the projection removed little: R2 ~ I).
                    W[1, : W.shape[1] - 1] *= 1.0 / float(sv2[active_k - 1])
                    betas_buf[it, :active_k, :n_curr] = beta_i

        # Record ||beta_it||_2 of the FINAL stored beta (the acted-renormalize above may
        # have replaced beta_i) for the estimator's noise floor at later iterations.
        if beta_norm_hist is not None:
            beta_norm_hist.append(
                float(np.linalg.svd(beta_i, compute_uv=False)[0]) if _reort_acted else beta_norm
            )

        if converged(alphas_buf[: it + 1], betas_buf[: it + 1], verbose=verbose, block_widths=block_widths + [n_curr]):
            termination = "converged"
            block_widths.append(n_curr)
            it += 1
            break

        q[0] = q[1]
        q[1] = q_next
        if keep_krylov:
            n_new = q_next.shape[1]
            if q_cols + n_new > Q_buf.shape[1]:
                Q_grown = np.empty((Q_buf.shape[0], max(2 * Q_buf.shape[1], q_cols + n_new)), dtype=complex, order='C')
                Q_grown[:, :q_cols] = Q_buf[:, :q_cols]
                Q_buf = Q_grown
            Q_buf[:, q_cols : q_cols + n_new] = q_next
            q_cols += n_new
            Q_list[0] = Q_buf[:, :q_cols]
        block_widths.append(n_curr)
        it += 1

    if verbose:
        print(f"Converged at iteration {it}")

    # Tail-only mode: q[1] is the last block ever appended in stored mode (the roll
    # q[0]=q[1]; q[1]=q_next runs before the append, and every break happens before the
    # append), so callers slicing the final residual columns see identical data.
    if keep_krylov:
        # Trim: returning a view of the over-allocated growth buffer would pin its spare
        # capacity for as long as the caller (e.g. TRLM restart) holds Q.
        res_Q = Q_buf if Q_buf.shape[1] == q_cols else np.ascontiguousarray(Q_buf[:, :q_cols])
    else:
        res_Q = q[1]
    res_alphas = alphas_buf[:it]
    res_betas = betas_buf[:it]
    # return_status appends `termination` as the final element, independent of the other
    # flags, so existing call sites that pass neither keep their original return arity.
    if return_widths:
        if return_W:
            if return_status:
                return res_alphas, res_betas, res_Q, W, block_widths, termination
            return res_alphas, res_betas, res_Q, W, block_widths
        if return_status:
            return res_alphas, res_betas, res_Q, block_widths, termination
        return res_alphas, res_betas, res_Q, block_widths
    else:
        if return_W:
            if return_status:
                return res_alphas, res_betas, res_Q, W, termination
            return res_alphas, res_betas, res_Q, W
        if return_status:
            return res_alphas, res_betas, res_Q, termination
        return res_alphas, res_betas, res_Q


include "_reort.pxi"
