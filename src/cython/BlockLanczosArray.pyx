# distutils: language = c++
# Compiler directives are centralized in setup.py (_DIRECTIVES), so a build mode can
# turn Cython's runtime checks on: IMPURITYMODEL_BUILD=debug. A `# cython:` header here
# would override that and silently defeat it.
# Permissive on purpose, unlike BlockLanczos.pyx -- see "Compiler directives" in
# doc/lanczos_invariants.md.

"""Block Lanczos kernel for numpy arrays / scipy sparse operators (MPI row-block).

The shared numerical layer (``Reort``, every tolerance constant, the Paige-Simon
``estimate_orthonormality`` estimator, the block-tridiagonal eigensolvers, and the
representation-dispatching block primitives / ``apply_reort``) lives in
``BlockLanczosCore.pyx``, imported here as a re-export block so every existing
``from impurityModel.ed.BlockLanczosArray import ...`` site keeps resolving. The
ManyBodyState kernel in ``BlockLanczos.pyx`` imports the same names directly from
``BlockLanczosCore``, not from here -- both kernels import downward from the shared
layer, never from each other. Both build a block-tridiagonal ``T`` (diagonal blocks
``alpha``, off-diagonal blocks ``beta``) whose eigenpairs approximate those of the
operator; the TRLM drivers restart it.

Reorthogonalization modes (``Reort``), what each guarantees:

* ``NONE``      — no reorthogonalization. Cheapest; orthogonality is allowed to
  decay and ghost (spurious duplicate) eigenvalues may appear. Eigenvalues still
  converge for well-separated spectra. The restart machinery in TRLM
  re-orthogonalizes regardless, so NONE only relaxes the *inner* recurrence.
* ``FULL``      — double-pass Gram–Schmidt against the whole basis every step.
  Most robust; ``‖QᴴQ − I‖`` stays at ~machine precision.

Every mode retains all ``p * m`` Krylov columns. That is the dominant allocation of a
long Green's-function run, and it cannot be compressed away: reorthogonalization needs
the *converged Ritz vectors*, and a Ritz vector converging at step ``m`` has weight on
every block, including any an earlier compression would have discarded. Bounding the
retained basis therefore needs secondary storage, not a smaller basis — see the
"Ritz compression is unsound" section of ``doc/plans/blocklanczos_reort_memory.md``.

Module thresholds (all derived from machine ``eps``; see definitions below):
``DEFLATE_TOL`` (relative rank floor on the block's singular values),
``BREAKDOWN_TOL`` (invariant-subspace detection). The
deflation floors are defined in the ``TSQR`` module that applies them.

Orthonormalization goes through ``TSQR`` via ``block_tsqr`` (``BlockLanczosCore.pyx``'s
``_block_ops.pxi``, dispatch-generic across block representations), reached here through
the shared ``factor_residual`` step-policy helper: the residual block's
triangular factor is computed from the block itself, never from the Gram matrix
``Wp^H Wp``, which is what makes it stable at any conditioning and its singular values
trustworthy.

Deflation policy: when a Lanczos block is rank-deficient (a singular value below the
``DEFLATE_TOL`` floor), the block size shrinks (EA16 shrinking-block, Meerbergen & Scott,
RAL-TR-2000-011) rather than zero-padding or terminating, so ``beta`` becomes rectangular
and ``T`` carries variable-size blocks; the recurrence keeps converging.
"""

import numpy as np
cimport numpy as np
import scipy.linalg as la
import scipy.sparse as sps

from scipy.linalg.cython_blas cimport zgemm

from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.mpi_comm import (
    MatvecExchangePlan,
    dest_flags_all,
    dest_flags_from_csr_indptr,
    matvec_exchange_mode,
)
from impurityModel.ed.solver_trace import note as _trace_note

# Shared numerical layer: Reort, every tolerance constant, the Paige-Simon estimator,
# the block-tridiagonal eigensolvers, and the representation-dispatching block
# primitives all now live in BlockLanczosCore.pyx -- re-exported here so every existing
# `from impurityModel.ed.BlockLanczosArray import ...` site (production and test) keeps
# resolving unchanged. Put new imports needed only by THIS module's own code above this
# block; put anything that belongs to the shared layer in BlockLanczosCore.pyx instead.
from impurityModel.ed.BlockLanczosCore import (  # noqa: F401
    Reort,
    BETA_BLOWUP_FACTOR,
    RESTART_ORTH_TOL,
    DEFAULT_EIGEN_TOL,
    THERMAL_GS_RESIDUAL_TOL,
    resolve_reort,
    divergence_guard,
    _build_full_T,
    _build_banded_lower,
    eigh_block_tridiagonal,
    eigsh,
    is_array,
    block_inner,
    block_apply,
    block_add_scaled,
    block_combine,
    block_orthogonalize,
    block_normalize,
    BlockBreakdown,
    block_tsqr,
    apply_full_reort,
    block_combine_array,
    block_orthogonalize_array,
    check_divergence,
    factor_residual,
    pack_lanczos_result,
)

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
)


def calculate_thermal_gs(h, block_size, e_max, v0=None, reort=Reort.FULL, comm=None, verbose=False):
    mpi = comm is not None
    rank = comm.rank if mpi else 0
    size = comm.size if mpi else 1

    def converged(alphas, betas, *args, **kwargs):
        e, s = eigsh(alphas, betas, de=e_max, select="m")
        sorted_indices = np.argsort(e)
        e = e[sorted_indices]
        s = s[:, sorted_indices]
        mask = e - np.min(e) <= e_max
        # Positive indices only: this module is compiled with `wraparound=False`, under which
        # `betas[-1]` reaches numpy as `-1 - len(betas)` and raises IndexError.
        beta_last = betas[len(betas) - 1]
        s_last = s[s.shape[0] - block_size:, mask]
        return spectral_norm(beta_last @ s_last) < THERMAL_GS_RESIDUAL_TOL

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
        verbose=verbose,
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


cdef extern from "complex.h":
    double complex conj(double complex z) nogil


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


cdef void apply_sparse_csr_nogil(
    int row_begin,
    int row_count,
    int num_cols,
    int p,
    int col_begin,
    double complex[:] data,
    long[:] indices,
    long[:] indptr,
    double complex[:, ::1] X,
    double complex[:, ::1] Y
) noexcept nogil:
    """``Y[:row_count, :p] = H[row_begin:row_begin+row_count, :] @ X[:, col_begin:col_begin+p]``,
    output row-indexed from 0.

    ``row_begin=0, row_count=global_N, col_begin=0`` (the whole matrix, every column) recovers
    the original single-shot apply; the MPI matvec loops below instead call this once per
    destination-rank row chunk (see the chunked-reduce comment there), so ``Y`` never has to
    hold more than one rank's own row range at a time. ``col_begin`` lets the graph exchange
    process the block in column rounds without copying ``X``'s column slice to a contiguous
    buffer each round.
    """
    cdef int i, j, k, ii
    cdef long row_start, row_stop
    cdef double complex val
    for ii in range(row_count):
        i = row_begin + ii
        row_start = indptr[i]
        row_stop = indptr[i+1]
        for k in range(p):
            Y[ii, k] = 0.0
        for j in range(row_start, row_stop):
            val = data[j]
            for k in range(p):
                Y[ii, k] = Y[ii, k] + val * X[indices[j], col_begin + k]


ctypedef fused row_index_t:
    int
    long


cdef void scatter_columns_nogil(
    int col_begin,
    long[:] indptr,
    row_index_t[:] slots,
    double complex[:] data,
    int p,
    int c0,
    double complex[:, ::1] X,
    long send_rows,
    double complex[:, ::1] S,
    double complex[:, ::1] Y
) noexcept nogil:
    """Scatter one column block of a bound ``LocalColumnOperator`` (basis_transcription) into the
    matvec's buffers: for local column ``col_begin + jj``, every stored ``(slot, value)`` adds
    ``value * X[col, c0:c0+p]`` to row ``slot`` of the send buffer ``S`` when ``slot < send_rows``,
    else to row ``slot - send_rows`` of the self buffer ``Y``. Both must be zeroed first.

    Columns are walked in ascending order, so each output row is summed in the order a CSR row
    loop (``apply_sparse_csr_nogil``) sums it -- the two agree to the last bit under strict IEEE.
    """
    cdef Py_ssize_t n_cols = indptr.shape[0] - 1
    cdef Py_ssize_t jj, e, col
    cdef long slot
    cdef int k
    cdef double complex val
    for jj in range(n_cols):
        col = col_begin + jj
        for e in range(indptr[jj], indptr[jj + 1]):
            slot = slots[e]
            val = data[e]
            if slot < send_rows:
                for k in range(p):
                    S[slot, k] = S[slot, k] + val * X[col, c0 + k]
            else:
                slot = slot - send_rows
                for k in range(p):
                    Y[slot, k] = Y[slot, k] + val * X[col, c0 + k]


cdef void _scatter_segments(segments, int p, int c0, double complex[:, ::1] X, long send_rows,
                            double complex[:, ::1] S, double complex[:, ::1] Y):
    """Every column block of a bound ``LocalColumnOperator`` through :func:`scatter_columns_nogil`."""
    cdef long[:] indptr
    cdef int[:] slots32
    cdef long[:] slots64
    cdef double complex[:] data
    cdef int col_begin
    for col_begin_obj, indptr_obj, rows_obj, data_obj in segments:
        col_begin = col_begin_obj
        indptr = indptr_obj
        data = data_obj
        if rows_obj.dtype == np.int32:
            slots32 = rows_obj
            with nogil:
                scatter_columns_nogil(col_begin, indptr, slots32, data, p, c0, X, send_rows, S, Y)
        else:
            slots64 = rows_obj
            with nogil:
                scatter_columns_nogil(col_begin, indptr, slots64, data, p, c0, X, send_rows, S, Y)


def scatter_column_operator(segments, int p, int c0, double complex[:, ::1] X, long send_rows,
                            double complex[:, ::1] S, double complex[:, ::1] Y):
    """Python entry to :func:`_scatter_segments`, for a matvec outside this kernel (the frozen-basis
    BiCGSTAB operator, ``gf_primitives.FrozenColumnMatvec``). ``S`` and ``Y`` must be zeroed first."""
    _scatter_segments(segments, p, c0, X, send_rows, S, Y)


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
    return_W=False,
    comm=None,
    return_widths=False,
    return_status=False,
    build_krylov_basis=None,
    deflate_tol=-1.0,
    max_iter=None,
    block_widths_init=None,
    locked=None,
    tail=None,
    return_tail=False,
):
    # tail=(q_prev, q_curr): resume a reort-'none' recurrence from its last two blocks -- the
    # local rows of the block before the current one and of the current one, each at its own
    # width -- with `alphas`/`betas` the coefficients so far (padded to the seed width) and
    # `block_widths_init` their widths. It is how a sparse recurrence whose support froze
    # carries on as the CSR recurrence of P H P (gf_solvers.block_Green_sparse), and how a
    # tail-only run continues after its budget ran out. `return_tail=True` appends the
    # (q_prev, q_curr) a "max_iter" stop can be resumed from (None after any other stop).
    # Resolve a string reort (e.g. "full") to the Reort enum, mirroring
    # block_lanczos_cy; otherwise the `reort == Reort.FULL` comparisons below never
    # match a string and the build silently runs with no reorthogonalization.
    reort = resolve_reort(reort)

    # Whether to retain the accumulated Krylov basis. The historical default (True) stands:
    # restart machinery (TRLM) and direct callers consume the returned Q regardless of
    # the reort mode. Callers that only need the final residual block (the GF continued
    # fraction with reort NONE) opt out explicitly, dropping the O(N * n * k) growth; any
    # reort mode projects against the basis and the warm-start protocol slices its blocks,
    # so opting out requires reort='none' and a fresh start.
    cdef bint keep_krylov
    if tail is not None:
        if reort != Reort.NONE or Q is not None or psi0 is not None or build_krylov_basis:
            raise ValueError("a tail resume needs reort='none', no psi0, no Q and no Krylov basis")
        if alphas is None or betas is None or np.ndim(alphas) != 3:
            raise ValueError("a tail resume needs the (n_blocks, p, p) alphas and betas so far")
        if len(alphas) > 0 and (block_widths_init is None or len(block_widths_init) != len(alphas)):
            raise ValueError("a tail resume needs block_widths_init, one width per block so far")
        build_krylov_basis = False
    if build_krylov_basis is None:
        keep_krylov = True
    else:
        keep_krylov = bool(build_krylov_basis)
        if not keep_krylov and (reort != Reort.NONE or Q is not None):
            raise ValueError("build_krylov_basis=False requires reort='none' and no warm-start Q")
    cdef int N
    cdef int n
    if tail is not None:
        N = tail[1].shape[0]
        n = alphas.shape[1]
    else:
        N = psi0.shape[0] if psi0 is not None else Q.shape[0]
        n = (psi0.shape[1] if psi0.ndim == 2 else 1) if psi0 is not None else alphas[0].shape[0]

    cdef int start_it = 0
    cdef np.ndarray alphas_arr, betas_arr
    cdef list Q_list
    # Krylov storage: Q_buf is an over-allocated column buffer grown geometrically
    # (amortized O(1) copies instead of one full np.concatenate reallocation per step);
    # Q_list[0] is always the filled Q_buf[:, :q_cols] view the reort machinery reads.
    cdef np.ndarray Q_buf = None
    cdef int q_cols = 0

    if tail is not None:
        start_it = len(alphas)
        alphas_arr = alphas
        betas_arr = betas
        q = [np.ascontiguousarray(tail[0], dtype=complex), np.ascontiguousarray(tail[1], dtype=complex)]
        Q_list = None
    elif alphas is not None and betas is not None and Q is not None:
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
    # Tail-only mode keeps, besides the final block, every block whose successor came out
    # narrower: the last vectors of the columns that deflated there. A basis-growing caller
    # (gf_solvers.block_Green) must probe from them too -- a column whose chain closes only
    # because the still-incomplete basis truncates H deflates, and probing from the final block
    # alone never reaches the determinants that chain was missing (review ledger C12).
    cdef list deflated_tail = []

    if max_iter is None:
        max_iter = int(
            np.ceil(
                h_op.shape[0] / n
                if sps.issparse(h_op) or isinstance(h_op, np.ndarray) or hasattr(h_op, "bind_slots")
                else N / n
            )
        )
    cdef int _buf_size = start_it + int(max_iter)
    cdef np.ndarray[double complex, ndim=3] alphas_buf = np.zeros((_buf_size, n, n), dtype=complex)
    cdef np.ndarray[double complex, ndim=3] betas_buf = np.zeros((_buf_size, n, n), dtype=complex)
    if start_it > 0:
        alphas_buf[:start_it] = alphas_arr
        betas_buf[:start_it] = betas_arr

    W = None  # the retired PRO estimator's slot in the return tuple

    cdef bint is_sparse = sps.issparse(h_op)
    # basis_transcription.LocalColumnOperator: the frozen-basis GF's (N, N_local) columns, kept as
    # the build's column blocks (~20 B/element instead of a CSR's ~50 at its peak, and no N + 1
    # row pointer). Its matvec scatters columns into the exchange buffers; see the matvec below.
    cdef bint is_columns = hasattr(h_op, "bind_slots")
    cdef bint is_dense = isinstance(h_op, np.ndarray)

    cdef int it = start_it
    cdef int dest, dest_off, dest_count  # row-chunk bounds for the MPI reduce-scatter matvec
    cdef int c0, c1, wc, nb_i  # column round and neighbour index of the graph exchange
    cdef double complex[:, ::1] slot_view, q1_cols  # graph exchange: a send/self slot, a column slice
    cdef double t_norm_max = 0.0
    cdef double h_norm_est = 0.0
    # A tail resume replays the divergence guard over the coefficients it was handed, so it keeps the
    # scale (and the TSQR breakdown reference built from it) the earlier steps established, exactly as
    # the sparse kernel's resume does; otherwise its first step would re-seed the scale unguarded.
    cdef bint guard_seeded = False
    cdef int _k
    if tail is not None:
        for _k in range(start_it):
            _, t_norm_max, h_norm_est = divergence_guard(
                spectral_norm(betas[_k]), spectral_norm(alphas[_k]), _k == 0, t_norm_max, h_norm_est
            )
            guard_seeded = True
    cdef double _h_scale = 0.0
    cdef double beta_norm, alpha_norm

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
        h_indptr_arr = np.ascontiguousarray(h_op.indptr, dtype=np.int64)
        h_indptr = h_indptr_arr
    elif is_dense:
        h_dense = np.ascontiguousarray(h_op)

    # GS_MATVEC_EXCHANGE=graph (the default): the matvec's reduce-scatter sends only the
    # structurally nonzero row blocks -- the destinations read off the CSR row pointers (a
    # complete graph for a dense H) -- over one Neighbor_alltoallv per column round on a cached
    # dist-graph, and sums in fixed source order (mpi_comm.MatvecExchangePlan). Built once per
    # call: three tiny collectives, and `n` is the widest block this call will see (`n_curr`
    # only ever shrinks, so every round fits the plan's buffers). `reduce` keeps the per-root
    # Reduce loop below. The generic operator branch (`h_op.dot`) has no MPI matvec either way.
    if is_columns and mpi and matvec_exchange_mode() != "graph":
        # The `reduce` exchange Reduces one destination's rows at a time, which a column scatter
        # cannot produce without a global-length buffer: fall back to the CSR (and its memory).
        h_op = h_op.tocsc().tocsr()
        is_columns = False
        is_sparse = True
        h_data = h_op.data
        h_indices = np.ascontiguousarray(h_op.indices, dtype=np.int64)
        h_indptr_arr = np.ascontiguousarray(h_op.indptr, dtype=np.int64)
        h_indptr = h_indptr_arr
    cdef bint use_graph = mpi and (is_sparse or is_dense or is_columns) and matvec_exchange_mode() == "graph"
    plan = None
    if use_graph:
        if is_sparse:
            dest_flags = dest_flags_from_csr_indptr(h_indptr_arr, counts, offsets)
        elif is_columns:
            dest_flags = h_op.owner_flags(counts, offsets)
        else:
            dest_flags = dest_flags_all(counts)
        plan = MatvecExchangePlan(comm, counts, offsets, dest_flags, n, config.GS_MATVEC_EXCHANGE_BYTES.get())
        _trace_note("matvec_exchange", site="block_lanczos_array", **plan.describe())
        if is_columns:
            h_op.bind_slots(rank, offsets, plan.destinations, plan.send_row_offsets, plan.total_send_rows)
    elif is_columns:
        # Serial: every row is this rank's own, so slot == row.
        h_op.bind_slots(0, np.zeros(1, dtype=np.int64), [], np.zeros(1, dtype=np.int64), 0)
    cdef np.ndarray no_send = np.empty((0, n), dtype=complex)

    cdef np.ndarray wp_arr = np.empty((N, n), dtype=complex, order='C')
    cdef double complex[:, ::1] wp = wp_arr
    cdef double complex[:, ::1] q1, q0, beta_prev_dag_mv
    cdef double complex[:, ::1] alpha_i

    # The `reduce`-mode MPI matvec below is a row-chunked reduce-scatter (one chunk per
    # destination rank's own row range, `Reduce`d straight to it) -- it never materializes a
    # (global_N, n_curr) product on any rank. (The `graph` mode keeps its own slots inside
    # `plan` and does not use `chunk_buf`.) This used to allocate exactly that buffer
    # here and Allreduce it before discarding every row but this rank's own share; see
    # doc/plans/dc_smo_performance.md for the measured cost (several GiB/rank on a
    # production-size SrMnO3 solve). `chunk_buf`, sized to the largest chunk this rank
    # will ever compute, is reused across both the chunk loop below and the destination
    # slots -- ranks are not equal-sized in general (deflation aside, hash distribution
    # only balances approximately), so this is `max(counts)`, not `N`.
    # Allocated unconditionally (mirroring wp_arr/wp above), not just on a width change: the
    # loop's reallocation guard (`wp_arr.shape[1] != n_curr`) is false on iteration 0, since
    # wp_arr was already built at the initial width -- an allocation gated the same way here
    # would leave chunk_view unassigned (and a slice into it a segfault) on every run whose
    # first sweep never deflates.
    cdef np.ndarray chunk_buf = np.empty((int(np.max(counts)), n), dtype=complex, order='C') if (mpi and not use_graph) else None
    cdef double complex[:, ::1] chunk_view = chunk_buf

    cdef list block_widths = list(block_widths_init) if block_widths_init is not None else [n] * start_it
    cdef int n_curr, n_prev, active_k

    # EA16 §2.6.2 locking deflation: keep every Lanczos vector orthogonal to the
    # already-converged ("locked") Ritz vectors. Without this the matvec keeps
    # amplifying the dominant locked directions back into the active subspace,
    # which reintroduces locked eigenvalues (and their 2*theta harmonics) as
    # spurious Ritz values *below* the true spectral minimum on restarted sweeps.
    # `locked` is column-distributed exactly like the Krylov vectors (local rows).
    cdef np.ndarray locked_arr = locked
    cdef np.ndarray locked_ovl
    cdef bint have_locked = locked_arr is not None and locked_arr.shape[1] > 0
    if have_locked:
        locked_arr = np.ascontiguousarray(locked_arr, dtype=complex)

    # Why the recurrence stops (returned when return_status=True), mirroring block_lanczos_cy:
    # "converged" (converged() satisfied), "invariant_subspace" (rank-deficient residual => the
    # block-Krylov space is closed under H => result is *exact*), "diverged" (non-finite Gram or
    # the divergence guard truncated a corrupted tail => NOT exact) or "max_iter" (budget spent
    # before the recurrence terminated naturally).
    termination = "max_iter"

    while it < _buf_size:
        q1_arr = np.ascontiguousarray(q[1])
        q1 = q1_arr
        n_curr = q1.shape[1]

        # The block matvec below Reduces `dest_count * n_curr` elements per destination rank.
        # `dest_count` comes from an Allgather so it agrees by construction, but `n_curr` is
        # read from this rank's own block: if the ranks ever disagree about the width, they
        # post the same Reduce with different counts and MPICH memcpys past the receive
        # buffer -- a SIGSEGV inside MPI_Reduce, with no Python traceback, no core dump (the
        # MPI runtime's own SIGSEGV handler exits 1 instead of re-raising) and both ranks
        # dying mute. Unconditional and collective on every rank, so it cannot itself
        # desynchronize; one 2-element Allreduce against a matvec is not measurable.
        if mpi:
            _w_probe = np.array([n_curr, -n_curr], dtype=np.int64)
            comm.Allreduce(MPI.IN_PLACE, _w_probe, op=MPI.MAX)
            if _w_probe[0] != -_w_probe[1]:
                raise RuntimeError(
                    f"block width disagrees across ranks at iteration {it}: this rank has "
                    f"n_curr={n_curr}, the communicator spans "
                    f"[{-_w_probe[1]}, {_w_probe[0]}]. The width is set by block_tsqr's "
                    "retained rank, which is computed from a replicated R and must therefore "
                    "be rank-invariant; a disagreement means that invariant has broken."
                )

        # --- 1. Block matvec: wp = H q_curr (+ MPI row-chunked reduce-scatter) ------------
        # Re-allocate wp/chunk buffers to match current active width for contiguous alignment
        if wp_arr.shape[1] != n_curr:
            wp_arr = np.empty((N, n_curr), dtype=complex, order='C')
            wp = wp_arr
            if mpi and not use_graph:
                chunk_buf = np.empty((int(np.max(counts)), n_curr), dtype=complex, order='C')
                chunk_view = chunk_buf

        if is_sparse:
            if use_graph:
                for c0, c1 in plan.column_rounds(n_curr):
                    wc = c1 - c0
                    for nb_i, dest in enumerate(plan.destinations):
                        dest_off = offsets[dest]
                        dest_count = counts[dest]
                        slot_view = plan.send_slot(nb_i, wc)
                        with nogil:
                            apply_sparse_csr_nogil(
                                dest_off, dest_count, N, wc, c0,
                                h_data, h_indices, h_indptr, q1, slot_view,
                            )
                    # This rank's own rows from its own columns never cross MPI.
                    dest_off = offsets[rank]
                    slot_view = plan.self_slot(wc)
                    with nogil:
                        apply_sparse_csr_nogil(dest_off, N, N, wc, c0, h_data, h_indices, h_indptr, q1, slot_view)
                    plan.exchange(wc)
                    plan.accumulate(wp_arr[:, c0:c1], wc)
            elif mpi:
                # One destination-rank row range at a time, Reduced straight to its owner --
                # never a (global_N, n_curr) buffer. See the comment on `chunk_buf`'s
                # declaration above; the same pattern (and its bit-identity caveat) lives in
                # _block_ops.pxi's block_apply. `dest_off`/`dest_count` are plain C ints,
                # read from `offsets`/`counts` before the nogil block -- indexing a numpy
                # array is a Python-level operation and cannot happen without the GIL.
                for dest in range(size):
                    dest_off = offsets[dest]
                    dest_count = counts[dest]
                    with nogil:
                        apply_sparse_csr_nogil(
                            dest_off, dest_count, N, n_curr, 0,
                            h_data, h_indices, h_indptr, q1, chunk_view[:dest_count, :],
                        )
                    if rank == dest:
                        # MPICH 4.0 (ubuntu-24.04's) segfaults inside MPI_Reduce when sendbuf is
                        # MPI.IN_PLACE, the root is NOT rank 0, and the message is over the
                        # 2048-byte eager threshold -- measured on a bare mpi4py grid: root 0
                        # passes at every size, root 1 passes to 2048 B and dies from 2064 B up,
                        # and the non-IN_PLACE form passes everywhere. Reducing straight into
                        # the destination buffer avoids it and drops a copy: on the destination
                        # rank dest_count == counts[rank] == N, so the shapes already match.
                        comm.Reduce(chunk_buf[:dest_count, :], wp_arr, op=MPI.SUM, root=dest)
                    else:
                        comm.Reduce(chunk_buf[:dest_count, :], None, op=MPI.SUM, root=dest)
            else:
                with nogil:
                    apply_sparse_csr_nogil(0, global_N, N, n_curr, 0, h_data, h_indices, h_indptr, q1, wp)
        elif is_columns:
            if use_graph:
                for c0, c1 in plan.column_rounds(n_curr):
                    wc = c1 - c0
                    send_view = plan.send_buffer(wc)
                    self_view = plan.self_slot(wc)
                    send_view[...] = 0.0
                    self_view[...] = 0.0
                    _scatter_segments(h_op.segments, wc, c0, q1, plan.total_send_rows, send_view, self_view)
                    plan.exchange(wc)
                    plan.accumulate(wp_arr[:, c0:c1], wc)
            else:
                wp_arr[...] = 0.0
                _scatter_segments(h_op.segments, n_curr, 0, q1, 0, no_send[:, :n_curr], wp)
        elif is_dense:
            if use_graph:
                for c0, c1 in plan.column_rounds(n_curr):
                    wc = c1 - c0
                    q1_cols = np.ascontiguousarray(q1_arr[:, c0:c1])
                    for nb_i, dest in enumerate(plan.destinations):
                        dest_off = offsets[dest]
                        dest_count = counts[dest]
                        slot_view = plan.send_slot(nb_i, wc)
                        with nogil:
                            apply_dense_nogil(
                                dest_count, N, wc, h_dense[dest_off : dest_off + dest_count, :], q1_cols, slot_view,
                            )
                    dest_off = offsets[rank]
                    slot_view = plan.self_slot(wc)
                    with nogil:
                        apply_dense_nogil(N, N, wc, h_dense[dest_off : dest_off + N, :], q1_cols, slot_view)
                    plan.exchange(wc)
                    plan.accumulate(wp_arr[:, c0:c1], wc)
            elif mpi:
                for dest in range(size):
                    dest_off = offsets[dest]
                    dest_count = counts[dest]
                    with nogil:
                        apply_dense_nogil(
                            dest_count, N, n_curr, h_dense[dest_off : dest_off + dest_count, :], q1,
                            chunk_view[:dest_count, :],
                        )
                    if rank == dest:
                        # MPICH 4.0 (ubuntu-24.04's) segfaults inside MPI_Reduce when sendbuf is
                        # MPI.IN_PLACE, the root is NOT rank 0, and the message is over the
                        # 2048-byte eager threshold -- measured on a bare mpi4py grid: root 0
                        # passes at every size, root 1 passes to 2048 B and dies from 2064 B up,
                        # and the non-IN_PLACE form passes everywhere. Reducing straight into
                        # the destination buffer avoids it and drops a copy: on the destination
                        # rank dest_count == counts[rank] == N, so the shapes already match.
                        comm.Reduce(chunk_buf[:dest_count, :], wp_arr, op=MPI.SUM, root=dest)
                    else:
                        comm.Reduce(chunk_buf[:dest_count, :], None, op=MPI.SUM, root=dest)
            else:
                with nogil:
                    apply_dense_nogil(global_N, N, n_curr, h_dense, q1, wp)
        else:
            if hasattr(h_op, "dot"):
                wp_arr[:] = h_op.dot(q1)
            elif hasattr(h_op, "__matmul__"):
                wp_arr[:] = h_op @ q1
            else:
                wp_arr[:] = h_op(q1)

        # --- 2. alpha_i = <q_curr | wp> ---------------------------------
        # Allocate alpha_i as a contiguous array
        alpha_i_arr = np.empty((n_curr, n_curr), dtype=complex, order='C')
        alpha_i = alpha_i_arr
        with nogil:
            matmul_nogil(n_curr, n_curr, N, 1.0, q1, b'C', wp, b'N', 0.0, alpha_i)

        if mpi:
            comm.Allreduce(MPI.IN_PLACE, alpha_i_arr, op=MPI.SUM)

        alphas_buf[it, :n_curr, :n_curr] = alpha_i_arr

        # --- 3. Subtract: wp = wp - q_curr * alpha_i - q_prev * beta_{i-1}^† -
        with nogil:
            matmul_nogil(N, n_curr, n_curr, -1.0, q1, b'N', alpha_i, b'N', 1.0, wp)

        # Bound unconditionally: q[0] is always a valid width-n block (a zero placeholder
        # on the very first step of a fresh start), so this is well-defined at every call.
        # See the matching note at _lanczos_step.pxi's n_prev binding / doc/lanczos_invariants.md.
        n_prev = q[0].shape[1]
        if it > 0:
            beta_prev_dag_arr = np.conj(betas_buf[it - 1, :n_curr, :n_prev].T).copy()
            beta_prev_dag_arr_c = np.ascontiguousarray(beta_prev_dag_arr)
            beta_prev_dag_mv = beta_prev_dag_arr_c
            q0 = np.ascontiguousarray(q[0])
            with nogil:
                matmul_nogil(N, n_curr, n_prev, -1.0, q0, b'N', beta_prev_dag_mv, b'N', 1.0, wp)

        # --- 3b. EA16 §2.6.2 locking deflation -------------------------
        # Locking deflation: unconditionally project against the locked set every step
        # (twice for robustness), for every reort mode.
        if have_locked:
            for _ in range(2):
                locked_ovl = np.ascontiguousarray(locked_arr.conj().T @ wp_arr)
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, locked_ovl, op=MPI.SUM)
                wp_arr -= locked_arr @ locked_ovl
            wp = wp_arr

        # --- 4. Full reorthogonalization --------------------------------
        if reort == Reort.FULL:
            if not keep_krylov:
                raise RuntimeError("Krylov basis must be built for reorthogonalization")
            wp_arr = apply_full_reort(wp_arr, Q_list, mpi, comm)

        # --- 5. TSQR of the residual block -------------------------------
        # TSQR of the residual block: the triangular factor is built from the block itself
        # (panel Householder + Givens merges + one Allgather), never from wp^H wp, so it is
        # backward stable at any conditioning and the rank decision below is made on the
        # block's true singular values. A *residual* block is negligible when it is small
        # against H, not against 1, so the breakdown reference is the divergence guard's
        # running spectral-scale estimate (max of the block norms seen so far, all bounded by
        # ||H||), widened by this step's alpha since the guard has not run yet. On the first
        # step of a cold start h_norm_est is still 0 and alpha_0 carries the scale.
        _h_scale = max(h_norm_est, t_norm_max, spectral_norm(alpha_i_arr))
        q_next, beta_i, active_k, sv_i = factor_residual(
            wp_arr, mpi, comm, _h_scale, deflate_tol=deflate_tol
        )
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

        # No "5a. Truncation" phase here: amplitude/forceful truncation prunes individual
        # determinant weights below slaterWeightMin, a ManyBodyState-only operation. The
        # array kernel's block is a dense/CSR column block with no per-determinant
        # structure to prune, so this phase is genuinely absent, not unmirrored.

        # --- Divergence safeguard ---------------------------------------
        # For a Hermitian H every block norm is bounded by ||H||; a jump of several orders
        # of magnitude over the largest healthy block norm means the recurrence has been
        # corrupted past repair. Truncate *before* this block (exclude alpha_i/beta_i at
        # index it) so the diverged tail never reaches the Green's function.
        # ||beta_i||_2 comes out of the factorization: beta_i and the block share their
        # singular values, so sv_i[0] is the 2-norm (reused by the Ritz-error check
        # below) with no SVD of its own.
        beta_norm = float(sv_i[0])
        alpha_norm = spectral_norm(alpha_i_arr)
        # first_step = it == start_it: this kernel's own absolute loop counter, always
        # start_it on the first step executed by THIS call (including a resumed call)
        # -- see check_divergence's docstring for why this must not be "it == 0".
        diverged, t_norm_max, h_norm_est = check_divergence(
            beta_norm, alpha_norm, it == start_it and not guard_seeded, t_norm_max, h_norm_est, it, comm
        )
        if diverged:
            termination = "diverged"
            break

        _converged = converged(alphas_buf[: it + 1], betas_buf[: it + 1], verbose=verbose, block_widths=block_widths + [n_curr])
        if mpi:
            # Rank-local verdict on an Allreduce'd alpha, which MPI does not promise is bitwise
            # identical on every rank: root decides for everyone, or a verdict flipping on one
            # rank near the threshold deadlocks the next step's collectives (review ledger M2).
            _converged = comm.bcast(bool(_converged), root=0)
        if _converged:
            termination = "converged"
            block_widths.append(n_curr)
            it += 1
            break

        if not keep_krylov and q_next.shape[1] < q[1].shape[1]:
            deflated_tail.append(q[1].copy())
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

    if verbose and (not mpi or rank == 0):
        print(f"Block Lanczos terminated ({termination}) at iteration {it}")

    # Tail-only mode: q[1] is the last block ever appended in stored mode (the roll
    # q[0]=q[1]; q[1]=q_next runs before the append, and every break happens before the
    # append), so callers slicing the final residual columns see identical data. The blocks
    # recorded before each narrowing (deflated_tail) come first, so those trailing columns
    # are unchanged.
    if keep_krylov:
        # Trim: returning a view of the over-allocated growth buffer would pin its spare
        # capacity for as long as the caller (e.g. TRLM restart) holds Q.
        res_Q = Q_buf if Q_buf.shape[1] == q_cols else np.ascontiguousarray(Q_buf[:, :q_cols])
    elif deflated_tail:
        res_Q = np.ascontiguousarray(np.hstack(deflated_tail + [q[1]]))
    else:
        res_Q = q[1]
    res_alphas = alphas_buf[:it]
    res_betas = betas_buf[:it]
    res = pack_lanczos_result(res_alphas, res_betas, res_Q, W, block_widths, termination,
                              return_widths, return_status, return_W)
    if return_tail:
        # Only a budget stop has rolled q onto the next step's (previous, current) pair; every other stop
        # ends the recurrence (or, "converged", breaks before the roll).
        res = tuple(res) + (((q[0], q[1]) if termination == "max_iter" else None),)
    return res


def block_lanczos_array(*args, **kwargs):
    return block_lanczos_array_cy(*args, **kwargs)
