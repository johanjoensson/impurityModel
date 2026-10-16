# distutils: language = c++
# Compiler directives are centralized in setup.py (_DIRECTIVES); see CLAUDE.md.

"""DistributedSparseMatrix.pyx
===========================
The one sparse matrix type of the solver: this rank's columns of an operator ``H`` on a
hash-distributed determinant basis, ``(N, N_local)``, stored as the column blocks the build made.

Rows are global (``N`` of them) and the *row* partition equals the *column* partition: rank ``r``
owns rows ``offsets[r] .. offsets[r] + counts[r]`` of every vector, and the columns that go with
them. A matvec ``Y = H X`` of a distributed block ``X`` (``(N_local, p)``) is therefore a
reduce-scatter: every rank computes the contribution of its columns to all ``N`` rows and sends
each destination rank's rows to it. ``routing_hash`` is linear in the occupied orbitals, so an
operator term shifts the owner by a constant and most of those contributions are structurally
zero (measured 11% dense at 256 ranks on SrMnO3); :class:`mpi_comm.MatvecExchangePlan` sends only
the nonzero ones, over one ``Neighbor_alltoallv`` per column round, and sums what arrives in
fixed source order, which is deterministic and independent of MPI's reduction trees.

Storage is ~20 B per stored element (16 for the value, 4 for an int32 row index) plus 8 B per
local column -- a scipy CSR costs ~50 B at its peak and a row pointer of ``N + 1`` entries on
every rank. The matvec scatters each column into the rows it reaches (``Y[row] += v X[col]``),
walking the columns in ascending order, so every output row is summed in the order a CSR row loop
would sum it (the two agree to the last bit under strict IEEE).

``comm`` is ``None`` for a matrix that lives on one rank (a replicated small sector); a
communicator of size 1 behaves the same.
"""

import numpy as np
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.solver_trace import note as _trace_note


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
    """Scatter one column block of a bound matrix into the matvec's buffers: for local column
    ``col_begin + jj``, every stored ``(slot, value)`` adds ``value * X[col, c0:c0+p]`` to row
    ``slot`` of the send buffer ``S`` when ``slot < send_rows``, else to row ``slot - send_rows`` of
    the self buffer ``Y``. Both must be zeroed first.

    Columns are walked in ascending order, so each output row is summed in the order a CSR row
    loop sums it.
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
    """Every column block of a bound matrix through :func:`scatter_columns_nogil`."""
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


cdef class DistributedSparseMatrix:
    """This rank's ``(N, N_local)`` columns of an operator; see the module docstring.

    Parameters
    ----------
    shape : (int, int)
        ``(N, N_local)``: the global dimension and the number of columns this rank owns.
    segments : list of (col_begin, indptr, rows, data)
        Local columns ``col_begin`` to ``col_begin + len(indptr) - 1`` in compressed-column form
        (``indptr`` int64, ``rows`` int32 when ``N < 2**31`` else int64, ``data`` complex128).
        Rows within a column ascend. Taken by reference: :meth:`bind_slots` rewrites ``rows`` in
        place.
    comm : mpi4py communicator, optional
        The communicator the matrix is distributed over; ``None`` for a matrix held on one rank.
    col_offset : int
        Global index of the first local column (``basis.offset``); only :meth:`toarray` and
        :meth:`to_scipy` use it.
    """

    cdef readonly tuple shape
    cdef readonly list segments
    cdef readonly object comm
    cdef readonly Py_ssize_t col_offset
    cdef object _binding
    cdef object _flags

    def __init__(self, shape, segments, comm=None, col_offset=0):
        self.shape = (int(shape[0]), int(shape[1]))
        self.segments = list(segments)
        self.comm = comm
        self.col_offset = int(col_offset)
        self._binding = None
        self._flags = None

    # -- constructors from other representations -------------------------------------------------

    @staticmethod
    def from_scipy(mat, comm=None, col_offset=0):
        """A matrix from a scipy sparse ``(N, N_local)`` (the columns this rank owns)."""
        csc = sps.csc_array(mat)
        csc.sort_indices()
        n_rows, n_cols = csc.shape
        row_dtype = np.int32 if n_rows < 2**31 else np.int64
        segments = []
        if n_cols:
            segments.append(
                (
                    0,
                    np.ascontiguousarray(csc.indptr, dtype=np.int64),
                    np.array(csc.indices, dtype=row_dtype),
                    np.array(csc.data, dtype=complex),
                )
            )
        return DistributedSparseMatrix((n_rows, n_cols), segments, comm, col_offset)

    @staticmethod
    def from_dense(a, comm=None, col_offset=0):
        """A matrix from a dense ``(N, N_local)`` array. For small, replicated sectors."""
        return DistributedSparseMatrix.from_scipy(sps.csc_array(np.asarray(a)), comm, col_offset)

    @staticmethod
    def coerce(h, comm=None):
        """``h`` as a matrix: a :class:`DistributedSparseMatrix` is returned as is; a scipy sparse
        ``(N, N_local)`` or a dense ndarray ``(N, N_local)`` (the columns this rank owns) is converted
        once. For callers that hold a small local array (tests, scripts); the solver builds the
        distributed type directly."""
        if isinstance(h, DistributedSparseMatrix):
            return h
        comm = comm if comm is not None and comm.size > 1 else None
        if sps.issparse(h):
            return DistributedSparseMatrix.from_scipy(h, comm)
        if isinstance(h, np.ndarray):
            return DistributedSparseMatrix.from_dense(h, comm)
        raise TypeError(f"cannot use a {type(h).__name__} as a sparse matrix")

    # -- size ------------------------------------------------------------------------------------

    @property
    def dtype(self):
        return np.dtype(complex)

    @property
    def nnz(self):
        return int(sum(len(rows) for _, _, rows, _ in self.segments))

    @property
    def nbytes(self):
        return int(sum(indptr.nbytes + rows.nbytes + data.nbytes for _, indptr, rows, data in self.segments))

    # -- exchange graph --------------------------------------------------------------------------

    def owner_flags(self, counts, offsets):
        """Bool array over ranks: does any stored row belong to rank ``d`` (rows ``offsets[d]`` to
        ``offsets[d] + counts[d]``)? Rank-local; used to build the matvec's exchange graph.

        Cached per layout, so it can still be answered after :meth:`bind_slots` has replaced the
        global rows (a matrix is applied again with a wider block, which rebuilds the plan)."""
        counts = np.asarray(counts, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        key = (tuple(counts.tolist()), tuple(offsets.tolist()))
        if self._flags is not None and self._flags[0] == key:
            return self._flags[1].copy()
        if self._binding is not None:
            raise RuntimeError("owner_flags needs global row indices; this matrix is already bound to slots")
        present = np.zeros(len(counts), dtype=bool)
        for _, _, rows, _ in self.segments:
            if len(rows):
                present[np.unique(np.searchsorted(offsets, rows, side="right") - 1)] = True
        flags = present & (counts > 0)
        self._flags = (key, flags)
        return flags.copy()

    def bind_slots(self, rank, offsets, destinations, send_row_offsets, total_send_rows):
        """Rewrite every row index, in place, as its row in the matvec's buffers.

        A row owned by ``destinations[i]`` goes to ``send_row_offsets[i] + (row - offsets[dest])``
        of the send buffer; a row this rank owns goes to ``total_send_rows + (row - offsets[rank])``
        (the self buffer, offset by the send rows). These are the layout of
        ``mpi_comm.MatvecExchangePlan``. Binding again with the same layout is a no-op; a different
        layout raises.
        """
        offsets = np.asarray(offsets, dtype=np.int64)
        key = (int(rank), tuple(offsets.tolist()), tuple(int(d) for d in destinations), int(total_send_rows))
        if self._binding is not None:
            if self._binding != key:
                raise RuntimeError("DistributedSparseMatrix is already bound to a different exchange layout")
            return
        base = np.full(len(offsets), np.iinfo(np.int64).min, dtype=np.int64)
        for i, dest in enumerate(destinations):
            base[dest] = int(send_row_offsets[i]) - offsets[dest]
        base[rank] = int(total_send_rows) - offsets[rank]
        for _, _, rows, _ in self.segments:
            if not len(rows):
                continue
            shift = base[np.searchsorted(offsets, rows, side="right") - 1]
            if (shift == np.iinfo(np.int64).min).any():
                raise RuntimeError("a stored row belongs to a rank that is not a matvec destination")
            rows += shift.astype(rows.dtype, copy=False)
        self._binding = key

    @property
    def is_distributed(self):
        return self.comm is not None and self.comm.size > 1

    def exchange_plan(self, width, site="exchange_plan"):
        """The :class:`mpi_comm.MatvecExchangePlan` for blocks up to ``width`` columns wide, with this
        matrix bound to its buffer layout; ``None`` on one rank.

        **Collective** on a distributed matrix (three tiny collectives), and not for reuse across
        long gaps: the plan keeps a distributed-graph communicator that the topology cache's FIFO
        eviction can free behind it on a later redistribution. Build one per kernel call. ``site``
        labels the ``matvec_exchange`` trace note.
        """
        if not self.is_distributed:
            if self._binding is None:
                self.bind_slots(0, np.zeros(1, dtype=np.int64), [], np.zeros(1, dtype=np.int64), 0)
            return None
        from impurityModel.ed.mpi_comm import MatvecExchangePlan

        comm = self.comm
        counts = np.empty(comm.size, dtype=np.int64)
        comm.Allgather(np.array([self.shape[1]], dtype=np.int64), counts)
        offsets = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
        plan = MatvecExchangePlan(
            comm, counts, offsets, self.owner_flags(counts, offsets), int(width), config.GS_MATVEC_EXCHANGE_BYTES.get()
        )
        self.bind_slots(comm.rank, offsets, plan.destinations, plan.send_row_offsets, plan.total_send_rows)
        _trace_note("matvec_exchange", site=site, **plan.describe())
        return plan

    # -- the matvec ------------------------------------------------------------------------------

    def apply(self, X, plan=None, out=None):
        """``Y = H X`` for a distributed block ``X`` of shape ``(N_local, p)``; returns ``(N_local, p)``.

        ``plan`` is the :meth:`exchange_plan` of a width at least ``p`` (built here when omitted).
        ``out``, if given, is a C-contiguous complex ``(N_local, p)`` array that receives the result.
        **Collective** when the matrix is distributed.
        """
        cdef double complex[:, ::1] Xv = np.ascontiguousarray(X, dtype=complex)
        cdef int p = Xv.shape[1]
        cdef int wc, c0, c1
        if Xv.shape[0] != self.shape[1]:
            raise ValueError(f"block has {Xv.shape[0]} local rows, the matrix owns {self.shape[1]} columns")
        if out is None:
            out = np.empty((self.shape[1], p), dtype=complex)
        cdef double complex[:, ::1] Yv = out
        cdef double complex[:, ::1] send_view, self_view
        if plan is None:
            plan = self.exchange_plan(p, "block_apply")
        if plan is None:
            Yv[...] = 0.0
            _scatter_segments(self.segments, p, 0, Xv, 0, np.empty((0, p), dtype=complex), Yv)
            return out
        for c0, c1 in plan.column_rounds(p):
            wc = c1 - c0
            send_view = plan.send_buffer(wc)
            self_view = plan.self_slot(wc)
            send_view[...] = 0.0
            self_view[...] = 0.0
            _scatter_segments(self.segments, wc, c0, Xv, plan.total_send_rows, send_view, self_view)
            plan.exchange(wc)
            plan.accumulate(out[:, c0:c1], wc)
        return out

    # -- conversions (tests, the small dense eigensolve) -----------------------------------------

    def to_scipy(self):
        """The same ``(N, N_local)`` matrix as a scipy CSC. Only before the matrix is bound to a
        plan (the row indices are global until then)."""
        if self._binding is not None:
            raise RuntimeError("to_scipy needs global row indices; this matrix is already bound to slots")
        n_cols = self.shape[1]
        counts = np.zeros(n_cols, dtype=np.int64)
        for col_begin, indptr, _, _ in self.segments:
            counts[col_begin : col_begin + len(indptr) - 1] = np.diff(indptr)
        indptr = np.zeros(n_cols + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        rows = np.concatenate([r for _, _, r, _ in self.segments]) if self.segments else np.empty(0, np.int64)
        data = np.concatenate([d for _, _, _, d in self.segments]) if self.segments else np.empty(0, complex)
        return sps.csc_array((data, rows.astype(np.int64, copy=False), indptr), shape=self.shape)

    def toarray(self):
        """Dense ``(N, N)`` with this rank's columns filled and every other column zero: summed over
        the ranks it is the whole operator. Not collective."""
        local = self.to_scipy().toarray()
        out = np.zeros((self.shape[0], self.shape[0]), dtype=complex)
        out[:, self.col_offset : self.col_offset + self.shape[1]] = local
        return out

    def __repr__(self):
        return f"DistributedSparseMatrix(shape={self.shape}, nnz={self.nnz}, distributed={self.is_distributed})"
