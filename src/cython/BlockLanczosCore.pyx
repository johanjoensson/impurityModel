# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=True, initializedcheck=False, cdivision=True, freethreading_compatible=True

"""BlockLanczosCore.pyx
=====================
Shared numerical layer for the array (``BlockLanczosArray.pyx``) and ManyBodyState
(``BlockLanczos.pyx``) block-Lanczos kernels: the ``Reort`` modes, every tolerance
constant, the Paige-Simon partial-reorthogonalization estimator
(``estimate_orthonormality``), the block-tridiagonal eigensolvers (``eigh_block_tridiagonal``,
``eigsh``), and the representation-dispatching block primitives (``block_inner``/``apply``/
``add_scaled``/``combine``/``orthogonalize``/``normalize``/``tsqr``, ``apply_reort``,
``selective_orthogonalize``) both kernels drive through.

Both kernels import from here, never from each other -- this module exists so that is
true. It used to compile inside ``BlockLanczosArray.pyx``, which forced
``BlockLanczos.pyx`` to import from it and forced the representation-dispatch
primitives (``_reort.pxi``) to close the resulting cycle with two **per-call** Python
imports on the hot path (inside ``block_normalize`` and ``apply_reort``). Neither is
true any more: both kernels import downward from here, and the two dispatch functions
reference these module-level names directly.
"""

import numpy as np
cimport numpy as np
import scipy.linalg as la
import scipy.sparse as sps

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
    robust_svd,
    spectral_norm,
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
# DEFLATE_TOL (~3.67e-11, the rank floor on the block's singular values), its squared
# counterpart DEFLATE_EVAL_TOL for tests on a Gram matrix, and BREAKDOWN_TOL are imported
# from TSQR above, which is where the reasoning for their values lives. Short version: the
# floor used to be EPS**(1/3) to keep the retained block inside CholeskyQR2's recovery
# regime, and TSQR needs no such protection -- at EPS**(1/3) the floor sat *above* physical
# splittings (it deflated away the near-copies of a 1e-9-split degeneracy and produced the
# spurious Ritz values test_no_ghost_bands is named after).
BETA_BLOWUP_FACTOR = 1e3           # ||beta_i|| above this * max(||beta||, ||alpha||) ⇒ divergence
REORT_PERIOD = 5                   # PERIODIC cadence, and SELECTIVE Ritz-check cadence
# R8c fallback cadence (audit_bad_blocks in _block_ops.pxi): every AUDIT_PERIOD steps,
# PARTIAL/SELECTIVE measure the TRUE overlap against every historical block instead of
# trusting the PRO estimate for that one step. Sized from a period sweep on the R8
# clustered/gf_style probes (doc/lanczos_invariants.md, 320-iteration horizon): period 3
# holds ||Q^HQ-I|| <= 1.1e-9 on clustered_near_deg (>10x below REORT_TOL) at every horizon
# in {80,160,240,320}, vs. period 4 already failing the 320 horizon (8.9e-8, above
# REORT_TOL) -- the margin between "holds" and "fails" is a narrow few steps, not a wide
# plateau, so this is deliberately not tuned to the cheapest passing value. Cost is
# real (~80% trigger rate on clustered_near_deg at this period, vs. FULL's 100%) but
# stays essentially free on the healthy probe regardless of period (13-15/320 acted at
# every period tested 2-20, matching the 13/320 no-audit baseline) -- the audit measures a
# true overlap that simply never exceeds BAD_BLOCK_TOL there.
AUDIT_PERIOD = 3
# Semi-orthogonality threshold (Simon's classical sqrt(EPS) criterion — the very level the
# PARTIAL estimator maintains, hence the shared constant). TRLM's thick-restart coefficient
# shortcuts (diag(theta) for the retained block, beta_res @ Y_last for the spike) are both
# derived from Q^H Q = I; above this level that premise is too inaccurate to use them and the
# restart is rebuilt from actual matvecs instead. See _trlm_core.
RESTART_ORTH_TOL = REORT_TOL       # ~1.49e-8
# Default convergence tolerance for TRLM/IRLM's public entry points (the max wanted
# Ritz residual at which a restart loop stops); single-sourced so every dispatcher's
# `tol` default moves together.
DEFAULT_EIGEN_TOL = 1e-8
# calculate_thermal_gs's own (looser) Ritz-residual stopping tolerance -- distinct value
# from DEFAULT_EIGEN_TOL, so it gets its own name rather than sharing one.
THERMAL_GS_RESIDUAL_TOL = 1e-6

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


# --- R8a: per-step, per-block reort trace (env-gated; ~zero cost when off) ----------
# Distinct from _REORT_PROF above: that accumulates run-wide aggregates (calls/acted/
# bad_blocks), cheap enough to leave on by the probe script by default. This records one
# entry per step -- predicted vs *measured* overlap per block, which costs a real O(total
# columns) inner product every traced step -- so it is opt-in via its own env var, never
# implied by BLOCKLANCZOS_PROFILE.
_REORT_TRACE_ON = _os_bla.environ.get("BLOCKLANCZOS_REORT_TRACE") == "1"
_REORT_TRACE = []


def get_reort_trace():
    """The accumulated per-step trace records (see the array kernel's recording site for
    the field list); a plain list, one dict per traced step."""
    return list(_REORT_TRACE)


def reset_reort_trace():
    _REORT_TRACE.clear()


def enable_reort_trace(on=True):
    """Toggle the per-step reort trace at runtime (equivalent to setting
    BLOCKLANCZOS_REORT_TRACE=1 in the environment before import)."""
    global _REORT_TRACE_ON
    _REORT_TRACE_ON = bool(on)


def record_reort_trace(entry):
    """Append one per-step trace record (a plain dict; see the array kernel's recording
    site for the field list). Called from the driver, not from ``apply_reort`` itself --
    the driver is the only place that has both the pre-reort candidate and the
    ``estimate_orthonormality`` decomposition in scope at once."""
    _REORT_TRACE.append(entry)


def reort_trace_enabled():
    """Live read of the trace toggle -- a plain ``from ... import _REORT_TRACE_ON`` in a
    caller module would snapshot the value at import time and go stale across a runtime
    ``enable_reort_trace()`` call, the same reason ``_REORT_PROF_ON`` is never imported
    directly (its checks all live in the same compilation unit via the ``_block_ops.pxi``
    include)."""
    return _REORT_TRACE_ON


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


cpdef np.ndarray estimate_orthonormality(
    np.ndarray[double complex, ndim=4] W,
    np.ndarray[double complex, ndim=3] alphas,
    np.ndarray[double complex, ndim=3] betas,
    object block_widths=None,
    double eps=0.0,
    double N=1.0,
    object out=None,
    object beta_norms=None,
    object parts_out=None,
):
    """... (see module docstring for the algorithm). ``parts_out``, if given a dict, is
    populated (R8a instrumentation only -- None by default, zero extra cost when unused)
    with the three additive contributions kept separate instead of summed into ``w_bar``:
    ``"rounding_injection"`` (nonzero only at row ``i``, the diagonal/self entry),
    ``"signed_propagation"`` (the three-term recurrence result BEFORE the noise floor is
    added, rows ``0..i-1``), and ``"noise_floor"`` (the magnitude added on top of it, rows
    ``0..i-1``) -- each shape ``(i+2, n, n)``, matching ``w_bar``, so
    ``rounding_injection + signed_propagation + noise_floor == w_bar`` row-wise (row
    ``i+1``, the seeded identity, is not decomposed -- it carries no estimate)."""
    cdef int i = alphas.shape[0] - 1
    cdef int n = alphas.shape[1]
    if eps == 0.0:
        eps = EPS

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

    # R8a instrumentation only: three fresh (i+2, n, n) buffers, allocated only when the
    # caller asks. Each additive term below writes into its own buffer in addition to (not
    # instead of) w_bar, so the summed estimate is unchanged bit-for-bit.
    if parts_out is not None:
        parts_out["rounding_injection"] = np.zeros((i + 2, n, n), dtype=complex)
        parts_out["signed_propagation"] = np.zeros((i + 2, n, n), dtype=complex)
        parts_out["noise_floor"] = np.zeros((i + 2, n, n), dtype=complex)

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
    # spectral_norm, not np.linalg.norm(ord=2): both are an SVD of a tiny block, but numpy's
    # goes straight to LAPACK gesdd with no fallback, and a core dump from CI run 33817368575
    # put the intermittent exit-139 inside exactly this statement -- anorm still unassigned
    # (reading as a subnormal), on a block deflated from width 2 to 1. gesdd is why robust_svd
    # exists twenty lines below; these two calls were the ones that never used it.
    cdef double anorm = max(
        spectral_norm(alphas[0, :w_0, :w_0]),
        spectral_norm(betas[0, :widths[1], :w_0]),
    )
    # A magnitude, like the noise floor below and unlike the signed three-term propagation:
    # this term models a rounding *injection*, which has no sign structure to cancel.
    w_bar[i, :w_next, :w_curr] = eps * n_scale * anorm * np.abs(beta_i_dag_inv)
    if parts_out is not None:
        parts_out["rounding_injection"][i, :w_next, :w_curr] = w_bar[i, :w_next, :w_curr]

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
    if parts_out is not None:
        parts_out["signed_propagation"][0, :w_next, :w_j] = w_bar[0, :w_next, :w_j]

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
        if parts_out is not None:
            parts_out["signed_propagation"][j, :w_next, :w_j] = w_bar[j, :w_next, :w_j]

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
    # sigma_min, dropping a redundant factorization per step. (This comment used to claim the
    # result was bit-identical to np.linalg.norm(ord=2). Measured over 2000 random small
    # blocks, 251 differ -- by at most 8.5e-16 relative, i.e. last-bit rounding between two
    # LAPACK front ends. Harmless for a heuristic scale, but not bit-identity.)
    _sv_bi = robust_svd(betas[i, :w_next, :w_curr], compute_uv=False)
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
            _bnorm_j = spectral_norm(betas[j, :widths[j+1], :widths[j]])
        _floor_j = eps * n_scale * (_bnorm_i + _bnorm_j) * _binv_norm
        w_bar[j, :w_next, :widths[j]] += _floor_j
        if parts_out is not None:
            parts_out["noise_floor"][j, :w_next, :widths[j]] = _floor_j

    W_out[0, : i + 1] = W[1]  # w_bar is already W_out[1] (built in place)

    return W_out


cpdef np.ndarray _build_full_T(np.ndarray[double complex, ndim=3] alphas, np.ndarray[double complex, ndim=3] betas, object block_widths=None):
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


include "_block_ops.pxi"
