# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, freethreading_compatible=True

r"""Tall-skinny QR (TSQR) for row-distributed blocks — the orthonormalization primitive.

Every block-Krylov recurrence in this package has to factor a tall, skinny block
:math:`A` (``n_local x p`` rows on this rank, the rows partitioned disjointly over the
communicator) into :math:`A = Q\beta` with :math:`Q^\dagger Q = I`. The historical
implementation went through the Gram matrix (``M = A^H A``, one ``Allreduce``, Cholesky),
which squares the condition number: it breaks down entirely once
:math:`\kappa(A) \gtrsim \varepsilon^{-1/2}`, leaves :math:`\lVert Q^\dagger Q - I\rVert =
O(\kappa^2\varepsilon)` (hence the CholeskyQR2 second pass), and can only resolve a
singular value to :math:`\varepsilon\sigma_{\max}^2/\sigma`.

TSQR never forms :math:`A^\dagger A`. It is two passes over the tall data:

**Pass 1 — the triangular factor.** :math:`A_{\text{local}}` is walked in panels of
``panel_rows`` rows; each panel is factored with LAPACK ``zgeqrf`` and only its ``p x p``
triangular factor is kept, merged into a running triangle by a Givens sweep
(:func:`merge_packed_r`) that exploits the triangularity of *both* operands and never
materializes the stacked ``2p x p`` matrix. The reflectors are discarded. This costs
:math:`O(n p^2)` flops and only :math:`O(\texttt{panel\_rows}\cdot p + p^2)` scratch, and
it never writes to ``A``. The rank-local triangles are then combined with a single
``Allgather`` of the packed triangles followed by the same Givens merge replayed in rank
order on every rank — so the global :math:`R` is *bitwise identical* everywhere by
construction, which the shrinking-block deflation policy requires (every rank must pick
the same block width). Ranks owning no rows contribute :math:`R = 0`, which the merge
skips entry by entry.

**Pass 2 — the orthonormal factor.** :math:`R` is canonicalized to a real positive
diagonal (making the factorization unique and matching the Cholesky convention the stored
``beta``s have always had), then its ``p x p`` SVD gives the block's *true* singular
values, from which breakdown and rank deficiency are decided. At full rank
:math:`Q = A R^{-1}` is obtained by back substitution (one ``ztrsm`` — no inverse is ever
formed); under deflation the retained directions are :math:`Q = A V_k \Sigma_k^{-1}`.

Because :math:`R` is computed by orthogonal transformations only, it is backward stable
for any :math:`\kappa` up to :math:`\varepsilon^{-1}` — the first pass can never fail. The
triangular solve does inherit :math:`\lVert Q^\dagger Q - I\rVert = O(\kappa\varepsilon)`,
so :func:`tsqr` repeats itself once (folding the correction into ``beta``) when
:math:`\kappa` exceeds ``REFINE_TOL``; below that the first pass is already orthonormal to
better than :math:`\varepsilon^{3/4}`, far under the ``sqrt(EPS)`` semi-orthogonality level
the reorthogonalization machinery maintains.

This module is a leaf: it imports nothing from the solver layer, and it owns the machine
constants and deflation floors that ``BlockLanczosArray`` (and through it every other
Krylov module) re-exports.
"""

import numpy as np
cimport numpy as np
import scipy.linalg as la

from libc.math cimport hypot, sqrt
from scipy.linalg.cython_lapack cimport zgeqrf
from scipy.linalg.cython_blas cimport ztrsm

# --- Tolerances (single definition site for the whole package) ----------------------
cdef double EPS_VAL = np.finfo(float).eps
EPS = EPS_VAL                              # ~2.22e-16
# Rank floor on the singular values of the block, relative to sigma_max: a direction is
# deflated when sigma_k <= DEFLATE_TOL * sigma_max.
#
# This was EPS**(1/3) (~6.06e-6) for as long as the factorization went through the Gram
# matrix, and *that* was why: the floor had to keep the retained block inside CholeskyQR2's
# kappa <~ EPS**(-1/2) recovery regime, so blocks were being deflated to protect the
# arithmetic rather than because their directions were dependent. TSQR needs no such
# protection -- it is backward stable to kappa ~ EPS**(-1) -- so the floor can say what it
# is supposed to say: which directions are numerically independent.
#
# EPS**(2/3) is where that lands. TSQR resolves a singular value of R to ~EPS*sigma_max, so
# 3.67e-11 is five orders above the noise it could possibly be measuring, while sitting
# below any physical scale a calculation is likely to care about. The old floor did not:
# measured on the near-degenerate spectrum of test_no_ghost_bands, whose eigenvalues split
# by 1e-9 relative, EPS**(1/3) sat three orders *above* the splitting and deflated away the
# near-copies, leaving a partially-filled T and spurious Ritz values. Ten cells of that test
# (five serial, five MPI) were marked xfail for it and now pass. The trade is downstream
# conditioning: the retained block's kappa is now bounded by 1/DEFLATE_TOL ~ 2.7e10 instead
# of 1.7e5, which propagates into ||beta^+|| in the W-estimator and into T.
DEFLATE_TOL = EPS_VAL ** (2.0 / 3.0)       # ~3.67e-11
DEFLATE_EVAL_TOL = DEFLATE_TOL * DEFLATE_TOL  # ~1.35e-21 : the same floor on a squared norm
# Rank floor for blocks whose deficiency is *structural* rather than emergent, i.e. seed
# blocks built by applying a set of transition operators (the Cartesian polarization
# components of XAS/NIXS/RIXS) to an eigenstate. Symmetry makes some of those components
# linearly dependent, and the solvers rely on the factorization to find that out -- the
# group-rule dedup that would otherwise do it was refuted for rank-4 tensors, so automatic
# rank deflation *is* the mechanism (doc/plans/rixs_r2_performance.md).
#
# Such a dependent direction is zero only up to the rounding accumulated while building the
# seed, and that grows with the basis. Measured on the RIXS tensor benchmark the discarded
# sigma/sigma_max reach 3.5e-11 -- already within 6% of the default DEFLATE_TOL, on a small
# problem. Above the floor they would be *retained*, injecting a pure-noise direction with
# sigma_min ~ 1e-11 into the block: ||beta^+|| ~ 1e11 in the estimator and a near-null
# direction in the resolvent.
#
# So these callers ask a different question and get a different floor. There is no tension
# with the default's job (resolving near-degeneracies down to ~1e-9): a genuinely distinct
# polarization component has a singular value O(1) relative to the block, five orders above
# this floor, so nothing physical is at risk of being deflated by it.
DEFLATE_TOL_SEEDS = EPS_VAL ** (1.0 / 3.0)  # ~6.06e-6
BREAKDOWN_TOL = 1e-12                      # absolute: ||beta||_2 <= this * scale => invariant subspace
# Condition number above which the A R^{-1} back substitution is repeated. One pass leaves
# ||Q^H Q - I|| ~ kappa * EPS, so this bounds it by EPS**(3/4) ~ 1e-12.
REFINE_TOL = EPS_VAL ** (-0.25)            # ~1.5e4
PANEL_ROWS = 512                           # default local panel height


# ===========================================================================
# Packed upper-triangular storage
# ===========================================================================
# Row-wise packed upper triangle of a p x p matrix: element (i, j), j >= i, lives at
# index i*p - i*(i-1)/2 + (j - i), so row i from column i onward is CONTIGUOUS — exactly
# the span a Givens rotation touches. Length p*(p+1)/2.

cdef inline Py_ssize_t _pidx(Py_ssize_t i, Py_ssize_t j, Py_ssize_t p) noexcept nogil:
    return i * p - (i * (i - 1)) // 2 + (j - i)


def packed_len(Py_ssize_t p):
    """Number of stored entries of a ``p x p`` packed upper triangle."""
    return p * (p + 1) // 2


def pack_upper(M):
    """Pack the upper triangle of a square matrix into the flat row-wise layout."""
    M = np.ascontiguousarray(M, dtype=complex)
    cdef Py_ssize_t p = M.shape[0]
    if M.ndim != 2 or M.shape[1] != p:
        raise ValueError("pack_upper expects a square matrix")
    out = np.empty(p * (p + 1) // 2, dtype=complex)
    cdef Py_ssize_t i
    cdef Py_ssize_t off = 0
    for i in range(p):
        out[off : off + p - i] = M[i, i:]
        off += p - i
    return out


def unpack_upper(v, Py_ssize_t p):
    """Inverse of :func:`pack_upper`: the dense ``p x p`` upper-triangular matrix."""
    v = np.ascontiguousarray(v, dtype=complex)
    if v.size != p * (p + 1) // 2:
        raise ValueError(f"packed length {v.size} != p*(p+1)/2 for p={p}")
    out = np.zeros((p, p), dtype=complex)
    cdef Py_ssize_t i
    cdef Py_ssize_t off = 0
    for i in range(p):
        out[i, i:] = v[off : off + p - i]
        off += p - i
    return out


# ===========================================================================
# Complex Givens rotation and the triangular-triangular merge
# ===========================================================================

cdef inline void _givens(
    double complex a, double complex b, double *c, double complex *s, double complex *r
) noexcept nogil:
    r"""Complex Givens rotation annihilating ``b`` against ``a``.

    Produces real ``c`` and complex ``s`` with :math:`|c|^2 + |s|^2 = 1` such that

    .. math::
        \begin{pmatrix} c & s \\ -\bar{s} & c \end{pmatrix}
        \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} r \\ 0 \end{pmatrix},

    with ``r`` carrying the phase of ``a`` and :math:`|r| = \sqrt{|a|^2+|b|^2}` (the LAPACK
    ``zlartg`` convention). ``hypot`` supplies the scaling that keeps the magnitudes from
    overflowing or flushing to zero; hand-rolled rather than calling ``zlartg`` because at
    these block sizes the sweep is thousands of scalar rotations.
    """
    cdef double ha, hb, d
    hb = hypot(b.real, b.imag)
    if hb == 0.0:
        c[0] = 1.0
        s[0] = 0.0
        r[0] = a
        return
    ha = hypot(a.real, a.imag)
    if ha == 0.0:
        c[0] = 0.0
        s[0] = b.conjugate() / hb
        r[0] = hb
        return
    d = hypot(ha, hb)
    c[0] = ha / d
    # a / ha is the unit phase of a; it multiplies both s and r.
    s[0] = (a / ha) * b.conjugate() / d
    r[0] = (a / ha) * d


cdef void _merge_packed(double complex *R, double complex *S, Py_ssize_t p) noexcept nogil:
    r"""Merge the packed upper triangle ``S`` into ``R`` in place (``S`` is destroyed).

    Computes the triangular factor of the stacked :math:`\begin{pmatrix} R \\ S
    \end{pmatrix}` without ever forming it. Column ``j`` is cleared by rotating row ``j``
    of ``R`` against rows ``0..j`` of ``S`` — the only rows of ``S`` that can have a
    nonzero there. Rows ``S_i`` with ``i > j`` are untouched until column ``i`` is reached,
    so they are still upper triangular when their turn comes, and columns left of ``j``
    stay zero because every rotation acts only on columns ``j..p-1``.
    """
    cdef Py_ssize_t i, j, jj, rj, si
    cdef double c
    cdef double complex s, r, x, y
    for j in range(p):
        rj = _pidx(j, j, p)
        for i in range(j + 1):
            si = _pidx(i, j, p)
            if S[si].real == 0.0 and S[si].imag == 0.0:
                continue
            _givens(R[rj], S[si], &c, &s, &r)
            R[rj] = r
            S[si] = 0.0
            for jj in range(1, p - j):
                x = R[rj + jj]
                y = S[si + jj]
                R[rj + jj] = c * x + s * y
                S[si + jj] = -s.conjugate() * x + c * y


cdef void _canonicalize(double complex *R, Py_ssize_t p) noexcept nogil:
    """Rotate each row's phase so the diagonal of ``R`` is real and non-negative.

    A unitary diagonal applied on the left, so it leaves ``R^H R`` (and hence the
    factorization) intact while making it unique — the convention the Cholesky path
    produced, and what keeps a rank's factor reproducible run to run.
    """
    cdef Py_ssize_t i, jj, ri
    cdef double hd
    cdef double complex ph
    for i in range(p):
        ri = _pidx(i, i, p)
        hd = hypot(R[ri].real, R[ri].imag)
        if hd == 0.0:
            continue
        ph = R[ri].conjugate() / hd
        R[ri] = hd
        for jj in range(1, p - i):
            R[ri + jj] = ph * R[ri + jj]


cdef Py_ssize_t _p_from_packed(Py_ssize_t length) except -1:
    """Recover ``p`` from a packed length ``p*(p+1)/2``."""
    cdef Py_ssize_t p = <Py_ssize_t>((sqrt(8.0 * length + 1.0) - 1.0) / 2.0 + 0.5)
    if p * (p + 1) // 2 != length:
        raise ValueError(f"{length} is not a valid packed triangle length")
    return p


def merge_packed_r(R, S):
    """Python entry point for the Givens merge: returns the packed triangle of ``[R; S]``.

    Both inputs are packed upper triangles of the same ``p``; neither is modified.
    """
    R_out = np.array(R, dtype=complex, order="C")
    S_work = np.array(S, dtype=complex, order="C")
    if R_out.shape != S_work.shape or R_out.ndim != 1:
        raise ValueError("merge_packed_r expects two packed triangles of equal length")
    cdef Py_ssize_t p = _p_from_packed(R_out.size)
    cdef double complex[::1] rv = R_out
    cdef double complex[::1] sv = S_work
    if p > 0:
        with nogil:
            _merge_packed(&rv[0], &sv[0], p)
    return R_out


# ===========================================================================
# Pass 1: the local (panel-blocked) triangular factor
# ===========================================================================

cdef int _panel_r(
    double complex[:, :] A,
    Py_ssize_t r0,
    Py_ssize_t m,
    double complex[::1] work,      # (panel x p) Fortran-ordered scratch, ldw = panel
    Py_ssize_t ldw,
    double complex[::1] tau,
    double complex[::1] lwork_buf,
    double complex[::1] S,         # packed output triangle
    Py_ssize_t p,
) noexcept nogil:
    """Factor rows ``[r0, r0+m)`` of ``A`` and write their packed triangular factor to ``S``.

    ``A`` is read-only (the panel is copied into the Fortran-ordered LAPACK scratch), and
    only the triangular factor is retained — the reflectors in ``work`` are discarded.
    Panels shorter than ``p`` rows produce a trapezoidal factor, zero-padded into ``S``.
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t rows = m if m < p else p
    cdef int mi = <int>m, pi = <int>p, ldwi = <int>ldw, lworki = <int>lwork_buf.shape[0]
    cdef int info = 0
    for j in range(p):
        for i in range(m):
            work[i + j * ldw] = A[r0 + i, j]
    zgeqrf(&mi, &pi, &work[0], &ldwi, &tau[0], &lwork_buf[0], &lworki, &info)
    if info != 0:
        return info
    for i in range(p * (p + 1) // 2):
        S[i] = 0.0
    for i in range(rows):
        for j in range(i, p):
            S[_pidx(i, j, p)] = work[i + j * ldw]
    return 0


def local_r(A, Py_ssize_t panel_rows=PANEL_ROWS):
    """Packed triangular factor of this rank's rows only (pass 1, no communication).

    Parameters
    ----------
    A : array_like
        ``(n_local, p)`` complex block; read-only, any strides.
    panel_rows : int
        Height of the LAPACK panels. Only affects rounding, never the exactness of the
        factorization; the default keeps the scratch inside cache.

    Returns
    -------
    numpy.ndarray
        Packed upper triangle (``p*(p+1)/2`` complex) with a real non-negative diagonal.
    """
    cdef double complex[:, :] Av = np.asarray(A, dtype=complex)
    cdef Py_ssize_t n = Av.shape[0]
    cdef Py_ssize_t p = Av.shape[1]
    cdef Py_ssize_t L = p * (p + 1) // 2
    R_np = np.zeros(L, dtype=complex)
    if p == 0 or n == 0:
        return R_np

    cdef Py_ssize_t panel = max(panel_rows, p)
    if panel > n:
        panel = n
    # Workspace query, once, for the largest panel this call will use.
    cdef int mi = <int>panel, pi = <int>p, ldwi = <int>panel, info = 0, lworki = -1
    cdef double complex probe = 0.0
    work_np = np.zeros(panel * p, dtype=complex)
    tau_np = np.zeros(p, dtype=complex)
    cdef double complex[::1] work = work_np
    cdef double complex[::1] tau = tau_np
    zgeqrf(&mi, &pi, &work[0], &ldwi, &tau[0], &probe, &lworki, &info)
    cdef Py_ssize_t lwork = max(<Py_ssize_t>probe.real, p)
    lwork_np = np.zeros(lwork, dtype=complex)
    S_np = np.zeros(L, dtype=complex)
    cdef double complex[::1] lwork_buf = lwork_np
    cdef double complex[::1] S = S_np
    cdef double complex[::1] R = R_np

    cdef Py_ssize_t r0 = 0, m, t
    cdef bint first = True
    cdef int rc = 0
    with nogil:
        while r0 < n:
            m = panel if r0 + panel <= n else n - r0
            rc = _panel_r(Av, r0, m, work, panel, tau, lwork_buf, S, p)
            if rc != 0:
                break
            if first:
                for t in range(L):
                    R[t] = S[t]
                first = False
            else:
                _merge_packed(&R[0], &S[0], p)
            r0 += panel
        if rc == 0:
            _canonicalize(&R[0], p)
    if rc != 0:
        raise np.linalg.LinAlgError(f"zgeqrf failed with info={rc}")
    return R_np


def reduce_r(R_local, comm=None):
    """Combine the rank-local packed triangles into the global one (one ``Allgather``).

    **Collective**: every rank must call it. The gathered triangles are merged in rank
    order with the same Givens sweep on every rank, so the returned triangle is bitwise
    identical everywhere — which is what lets each rank decide the same deflated block
    width without a further broadcast.
    """
    R_local = np.ascontiguousarray(R_local, dtype=complex)
    if comm is None or comm.size == 1:
        return R_local
    cdef Py_ssize_t L = R_local.size
    cdef Py_ssize_t p = _p_from_packed(L)
    if p == 0:
        return R_local
    buf = np.empty(comm.size * L, dtype=complex)
    comm.Allgather(R_local, buf)
    out = np.ascontiguousarray(buf[:L])
    cdef double complex[::1] ov = out
    cdef double complex[::1] sv
    cdef Py_ssize_t r
    for r in range(1, comm.size):
        sv = buf[r * L : (r + 1) * L]
        with nogil:
            _merge_packed(&ov[0], &sv[0], p)
    with nogil:
        _canonicalize(&ov[0], p)
    return out


def tsqr_r(A, comm=None, Py_ssize_t panel_rows=PANEL_ROWS):
    """Pass 1 alone: the global packed triangular factor of the distributed block.

    **Collective** when ``comm`` is not ``None``. See :func:`local_r` and :func:`reduce_r`.
    """
    return reduce_r(local_r(A, panel_rows), comm)


# ===========================================================================
# Pass 2: Q by back substitution, with deflation
# ===========================================================================

# Above this block width the solve goes to BLAS; at or below it, the hand-rolled loop wins.
#
# The BLAS call for a tall-skinny right-solve is `m = p, n = n_rows`, and OpenBLAS threads
# that shape by columns -- the wrong axis. At p = 2, n ~ 8000 the thread dispatch alone
# measured ~1.2 ms per call against ~30 us of actual work: 88% of the factorization time of
# a whole ground-state solve, and invisible to a single-threaded microbenchmark (the same
# run is 3x faster end-to-end under OPENBLAS_NUM_THREADS=1). The hand-rolled substitution
# below is one pass over the rows with the triangle in registers -- no dispatch, no
# threading, perfect locality.
#
# It is O(n p^2) scalar work though, so it loses to BLAS-3 blocking once the block is wide.
# Measured against the Cholesky path it replaced: at p <= 8 the loop wins outright, by
# p = 16 it is ~3x a single Cholesky-QR, and by p = 40 the scalar cost would dominate
# outright. Every block this package builds (Lanczos p ~ 2, GF seeds 1, RIXS polarization
# tensors <= 9, CIPSI reference blocks ~ 10) sits in the region where the loop wins.
DEF TRSM_BLAS_MIN_WIDTH = 16


cdef void _trsm_right_inv(double complex[:, ::1] Q, double complex[:, ::1] R) noexcept nogil:
    r"""``Q <- Q R^{-1}`` by back substitution, in place, for row-major ``Q``.

    Each row :math:`a` of ``Q`` is replaced by the solution :math:`x` of :math:`x R = a`.
    With ``R`` upper triangular that is forward substitution in ``j``:

    .. math:: x_j = \Bigl(a_j - \sum_{l<j} x_l R_{lj}\Bigr) / R_{jj}.

    For a wide block this hands over to ``ztrsm``: BLAS reads the C-contiguous ``(n, p)``
    buffer as the column-major ``p x n`` matrix ``Q^T`` and the C-contiguous ``R`` as ``R^T``
    (lower triangular), so the row-major right-solve is the column-major left-solve
    ``R^T X = Q^T``.
    """
    cdef int p = <int>Q.shape[1]
    cdef int n = <int>Q.shape[0]
    cdef double complex one = 1.0
    cdef char side = b'L', uplo = b'L', transa = b'N', diag = b'N'
    cdef Py_ssize_t i, j, l
    cdef double complex s
    cdef double complex rdiag_inv[TRSM_BLAS_MIN_WIDTH]
    if n == 0 or p == 0:
        return
    if p > TRSM_BLAS_MIN_WIDTH:
        ztrsm(&side, &uplo, &transa, &diag, &p, &n, &one, &R[0, 0], &p, &Q[0, 0], &p)
        return
    # Reciprocals once, not once per row: n*p divisions become p.
    for j in range(p):
        rdiag_inv[j] = 1.0 / R[j, j]
    for i in range(n):
        for j in range(p):
            s = Q[i, j]
            for l in range(j):
                s = s - Q[i, l] * R[l, j]
            Q[i, j] = s * rdiag_inv[j]


def tsqr(A, comm=None, double scale=1.0, Py_ssize_t panel_rows=PANEL_ROWS, bint refine=True,
         double deflate_tol=-1.0):
    r"""Orthonormalize a row-distributed tall-skinny block: ``A = Q @ beta``.

    **Collective** when ``comm`` is not ``None`` (one ``Allgather`` per pass; a second pass
    fires only for ill-conditioned blocks, see ``REFINE_TOL``). Every rank returns the same
    ``beta``, ``k`` and ``sv``; ``Q`` holds this rank's rows.

    Parameters
    ----------
    A : array_like
        ``(n_local, p)`` complex block. Never modified. The rows must be partitioned
        *disjointly* over ``comm`` — the same premise the Gram-matrix ``Allreduce`` this
        replaces has always relied on.
    comm : mpi4py.MPI.Comm or None
        Communicator over which the rows are distributed. ``None`` for a serial block.
    scale : float
        Reference for the *breakdown* test: the block is declared numerically zero when
        :math:`\sigma_{\max} \le` ``BREAKDOWN_TOL * scale``. Pass ``~||H||`` for a Lanczos
        residual block (a residual is zero when it is negligible against the operator, not
        against 1); leave at ``1.0`` for a block whose columns are already O(1).
    panel_rows : int
        Local panel height (see :func:`local_r`).
    refine : bool
        Allow the conditional second pass. ``False`` inside the recursion.
    deflate_tol : float
        Rank floor: a direction is deflated when ``sigma_k <= deflate_tol * sigma_max``.
        Negative (the default) means :data:`DEFLATE_TOL`, which is set to resolve directions
        down to the factorization's own noise. Callers whose blocks are *structurally*
        rank-deficient -- the transition-operator seed blocks of the spectroscopies, whose
        symmetry-dependent components are zero only up to their construction rounding --
        pass :data:`DEFLATE_TOL_SEEDS` instead. Like ``scale``, this is a property of the
        block the caller is handing over, not of the factorization.

    Returns
    -------
    Q : numpy.ndarray or None
        ``(n_local, k)`` C-contiguous orthonormal block, ``None`` on breakdown.
    beta : numpy.ndarray or None
        ``(k, p)`` factor with ``A = Q @ beta``, ``None`` on breakdown. Upper triangular
        with a real non-negative diagonal at full rank; ``Sigma_k V_k^H`` under deflation.
    k : int
        Retained rank. ``0`` marks breakdown (numerically zero block / closed Krylov
        space); ``-1`` marks a non-finite factor, i.e. a corrupted recurrence rather than
        a genuine invariant subspace.
    sv : numpy.ndarray or None
        The ``p`` singular values of the *global* block, descending. ``sv[0]`` is
        ``||beta||_2`` and ``1/sv[k-1]`` is ``||beta^+||_2``, so callers need no SVD of
        their own.
    """
    if deflate_tol < 0.0:
        deflate_tol = DEFLATE_TOL
    A_np = np.asarray(A, dtype=complex)
    if A_np.ndim != 2:
        raise ValueError("tsqr expects a 2-D (n_local, p) block")
    cdef Py_ssize_t p = A_np.shape[1]
    if p == 0:
        return A_np[:, :0].copy(), np.zeros((0, 0), dtype=complex), 0, np.zeros(0)

    R_packed = tsqr_r(A_np, comm, panel_rows)
    if not np.all(np.isfinite(R_packed.view(float))):
        # A non-finite factor means the recurrence feeding this block is corrupted; that is
        # a different statement from "the block is zero" and the callers act differently.
        return None, None, -1, None
    R = unpack_upper(R_packed, p)

    _, sv, Vh = la.svd(R)
    cdef double s_max = float(sv[0]) if sv.size else 0.0
    if s_max <= BREAKDOWN_TOL * scale:
        return None, None, 0, sv
    cdef Py_ssize_t k = int(np.count_nonzero(sv > deflate_tol * s_max))
    if k == 0:                       # unreachable: sv[0] > 0 always survives its own floor
        return None, None, 0, sv

    cdef double kappa = s_max / float(sv[k - 1])
    Q = None
    if k == p:
        # Full rank: Q = A R^{-1} by back substitution. No inverse is formed, and the
        # triangular solve is backward stable row by row.
        Q = A_np.copy(order="C")
        _trsm_right_inv(Q, np.ascontiguousarray(R))
        beta = R
    else:
        # Deflated: keep the k leading right-singular directions. Q = A V_k Sigma_k^{-1}
        # equals (A R^{-1}) U_k, without needing R to be invertible; the retained
        # conditioning is bounded by 1/deflate_tol, so the scaling is safe.
        Y = Vh[:k].conj().T / sv[:k][np.newaxis, :]
        Q = np.ascontiguousarray(A_np @ Y)
        beta = sv[:k][:, np.newaxis] * Vh[:k]

    if refine and kappa > REFINE_TOL:
        # The back substitution leaves ||Q^H Q - I|| ~ kappa * EPS. One repetition on the
        # (now well-conditioned) Q drives that to O(EPS) and folds the correction into
        # beta. Unlike CholeskyQR2 this is never a rescue: pass 1 cannot fail.
        Q2, R2, k2, _ = tsqr(Q, comm, 1.0, panel_rows, False, deflate_tol)
        if k2 <= 0:
            return None, None, k2, sv
        return Q2, R2 @ beta, k2, sv
    return Q, beta, k, sv
