# distutils: language = c++
# Compiler directives are centralized in setup.py (_DIRECTIVES), so a build mode can
# turn Cython's runtime checks on: IMPURITYMODEL_BUILD=debug. A `# cython:` header here
# would override that and silently defeat it.

"""BlockLanczosCore.pyx
=====================
Shared numerical layer for the array (``BlockLanczosArray.pyx``) and ManyBodyState
(``BlockLanczos.pyx``) block-Lanczos kernels: the ``Reort`` modes, every tolerance
constant, the block-tridiagonal eigensolvers (``eigh_block_tridiagonal``,
``eigsh``), and the representation-dispatching block primitives (``block_inner``/``apply``/
``add_scaled``/``combine``/``orthogonalize``/``normalize``/``tsqr``, ``apply_full_reort``) both kernels drive through.

Both kernels import from here, never from each other -- this module exists so that is
true. It used to compile inside ``BlockLanczosArray.pyx``, which forced
``BlockLanczos.pyx`` to import from it and forced the representation-dispatch
primitives (``_reort.pxi``) to close the resulting cycle with two **per-call** Python
imports on the hot path (inside ``block_normalize`` and ``apply_full_reort``). Neither is
true any more: both kernels import downward from here, and the two dispatch functions
reference these module-level names directly.
"""

import time as _time
import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport zgemm
import scipy.linalg as la
import scipy.sparse as sps

from mpi4py import MPI
from enum import Enum


class Reort(Enum):
    NONE = 0
    FULL = 2


# The machine constant and the deflation floors are defined once, in the TSQR leaf that
# applies them; a second literal here would be a second source of truth. ``DEFLATE_TOL``
# itself is only read by callers, who import it straight from ``TSQR`` (cython-lint has no
# noqa escape for a pure re-export).
from impurityModel.ed import config
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.manybody_basis import collective_top_k_mask
from impurityModel.ed.TSQR import (
    EPS,
    DEFLATE_EVAL_TOL,
    BREAKDOWN_TOL,
    spectral_norm,
    tsqr,
)

cdef double EPS_VAL = EPS
# DEFLATE_TOL (~3.67e-11, the rank floor on the block's singular values), its squared
# counterpart DEFLATE_EVAL_TOL for tests on a Gram matrix, and BREAKDOWN_TOL are imported
# from TSQR above, which is where the reasoning for their values lives.
BETA_BLOWUP_FACTOR = 1e3           # ||beta_i|| above this * max(||beta||, ||alpha||) ⇒ divergence
# Semi-orthogonality threshold (Simon's classical sqrt(EPS) criterion — the very level the
# old PRO estimator maintained). TRLM's thick-restart coefficient
# shortcuts (diag(theta) for the retained block, beta_res @ Y_last for the spike) are both
# derived from Q^H Q = I; above this level that premise is too inaccurate to use them and the
# restart is rebuilt from actual matvecs instead. See _trlm_core.
RESTART_ORTH_TOL = np.sqrt(EPS_VAL)       # ~1.49e-8
# Default convergence tolerance for TRLM's public entry points (the max wanted
# Ritz residual at which a restart loop stops); single-sourced so every dispatcher's
# `tol` default moves together.
DEFAULT_EIGEN_TOL = 1e-8
# calculate_thermal_gs's own (looser) Ritz-residual stopping tolerance -- distinct value
# from DEFAULT_EIGEN_TOL, so it gets its own name rather than sharing one.
THERMAL_GS_RESIDUAL_TOL = 1e-6

def resolve_reort(reort):
    """Resolve a ``reort`` argument (``Reort`` member or string) to a ``Reort`` enum.

    Shared by both kernels so the accepted spellings stay in sync. A non-string is
    returned unchanged. Raises ``ValueError`` on an unknown string.
    """
    if not isinstance(reort, str):
        return reort
    _map = {
        "none": Reort.NONE,
        "full": Reort.FULL,
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
        # `offsets[len(offsets) - 1]`, never `offsets[-1]`: this module now compiles with
        # wraparound=False like every other kernel, and under that directive Cython does not
        # add the length to a negative index -- even on a plain list, where it lowers to
        # PyList_GET_ITEM and reads out of bounds. Measured: a one-line extension doing
        # `xs[-1]` on a 3-element list segfaults (exit 139) when built wraparound=False.
        offsets.append(offsets[len(offsets) - 1] + w)
    total = offsets[len(offsets) - 1]
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


cdef void _adjoint_gemm_nogil(double complex[:, ::1] V, double complex[:, ::1] W, double complex[:, ::1] out) noexcept nogil:
    matmul_nogil(V.shape[1], W.shape[1], V.shape[0], 1.0, V, b'C', W, b'N', 0.0, out)


def _adjoint_gemm(double complex[:, ::1] V, double complex[:, ::1] W, double complex[:, ::1] out):
    with nogil:
        _adjoint_gemm_nogil(V, W, out)


cpdef np.ndarray adjoint_product(np.ndarray V, np.ndarray W):
    """``V^H W`` as a C-contiguous array, conjugating whichever operand has fewer columns.

    ``np.conj(V.T) @ W`` materializes a conjugated copy of all of ``V`` before the GEMM. With
    ``V`` a stored Krylov basis (N x D) and ``W`` a block (N x p), that copy is the same
    O(N D) as the product itself and measured at ~70% of it (N=50000, D=200, p=4: 37 of
    51 ms), paid on every inner product. ``conj(W^H V)^T`` conjugates only ``W`` and the
    small (p x D) result instead -- the form ``SparseKrylovDense.reort`` already uses. The
    two agree to rounding, not bitwise: the GEMM runs with the operands swapped.
    """
    if V.ndim == 2 and W.ndim == 2 and V.shape[1] > W.shape[1]:
        return np.ascontiguousarray(np.conj(np.conj(W).T @ V).T)
    if (
        V.ndim == 2 and W.ndim == 2 and V.dtype == np.complex128 and W.dtype == np.complex128
        and V.flags.c_contiguous and W.flags.c_contiguous and V.shape[0] == W.shape[0] and V.shape[0] > 0
    ):
        # The Lanczos overlaps (a block against the product): one zgemm with the conjugate-transpose
        # flag, no N x p conjugated copy.
        out = np.empty((V.shape[1], W.shape[1]), dtype=complex)
        _adjoint_gemm(V, W, out)
        return out
    return np.ascontiguousarray(np.conj(V.T) @ W)


cpdef tuple block_orthogonalize_array(np.ndarray wp, np.ndarray Q, object overlaps=None, object comm=None):
    if overlaps is None:
        overlaps = adjoint_product(Q, wp)
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
    wp -= Q @ overlaps
    return wp, overlaps


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


include "_block_ops.pxi"
include "_lanczos_step.pxi"
