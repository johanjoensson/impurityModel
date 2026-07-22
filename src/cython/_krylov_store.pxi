# ===========================================================================
# Sparse Krylov retention store
# ===========================================================================
# SparseKrylovDense: the chunked columnar store that retains the Krylov basis for the
# reorthogonalizing Lanczos modes (zero-copy growth; _scatter_block_rows fills it). Since the
# MBS kernel's retention IS this store, it backs every reort mode's Q.

cdef void _scatter_block_rows(
    krylov_t[:, :] Qv,
    ManyBodyBlockState_cpp* b,
    const int* rows,
    Py_ssize_t nrow,
    Py_ssize_t ncol,
    Py_ssize_t base,
) noexcept:
    """Copy a shared-support block's coefficients into ``Qv[:, base:base+ncol]``.

    ``rows[r]`` is the store row that block row ``r`` was registered to. Compiled once per
    ``krylov_t`` so a complex64 store scatters without a complex128 staging buffer; the
    narrowing conversion happens element-wise here and nowhere else.
    """
    cdef Py_ssize_t r, ci
    cdef ManyBodyBlockState_cpp.Row src
    for r in range(nrow):
        src = dereference(b).row(r)
        for ci in range(ncol):
            Qv[rows[r], base + ci].real = src[ci].real()
            Qv[rows[r], base + ci].imag = src[ci].imag()


cdef class SparseKrylovDense:
    r"""Incrementally-maintained dense copy of the (rank-local) sparse block-Krylov basis.

    Holds the Krylov vectors as COLUMN-CHUNKED dense complex buffers over a growing
    determinant->row support map, so block reorthogonalization can gather columns
    (``Q[:, cols]``) instead of re-materializing them from the ``flat_map`` states on
    every step. ``append``/``append_block`` add one block of columns at a time; ``reort``
    runs ``n_passes`` of classical Gram-Schmidt of ``wp`` against the selected columns via
    BLAS ``zgemm``. The buffer is rank-local (over the rank's owned determinants); the
    only collective is the small ``(n_cols x p)`` overlap ``Allreduce`` each pass.

    **Chunked growth (Phase 4)**: instead of one geometrically-doubled buffer whose every
    growth reallocates AND copies the whole basis (a ~3x transient at each doubling —
    old + new buffer live simultaneously), new columns go into a fresh chunk sized
    geometrically. Existing chunks are never copied. Rows registered after a chunk was
    created are implicitly zero in that chunk's columns — exactly right, because older
    Krylov vectors have no weight on determinants discovered later. ``reserve_rows``
    (e.g. the local basis size, which bounds the support) sizes future chunks so row
    growth never forces a new chunk.
    """
    cdef cpp_map[SlaterDeterminant_cpp[uint64_t], int] support
    cdef vector[SlaterDeterminant_cpp[uint64_t]] row_det
    cdef list _chunks       # dense (rows_c x cols_c) complex buffers
    cdef list _used         # used columns per chunk
    cdef int n_rows
    cdef int n_cols
    cdef int _rows_hint
    cdef object _dtype      # chunk dtype: complex128 (default) or complex64
    cdef bint _single       # True iff _dtype is complex64

    def __cinit__(self, dtype=None):
        self.n_rows = 0
        self.n_cols = 0
        self._rows_hint = 0
        self._chunks = []
        self._used = []
        self._dtype = np.dtype(complex if dtype is None else dtype)
        if self._dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError(f"SparseKrylovDense dtype must be complex64 or complex128, got {self._dtype}")
        self._single = self._dtype == np.dtype(np.complex64)

    @property
    def dtype(self):
        """Storage dtype of the Krylov coefficients (``complex128`` or ``complex64``).

        ``complex64`` halves the store at the cost of representing each basis vector to
        ~6e-8 relative accuracy. That is the same order as the semi-orthogonality target
        ``REORT_TOL = sqrt(EPS) ~ 1.5e-8`` the reorthogonalization is aiming for, so the
        projection quality is essentially unchanged; the overlaps and the residual are
        still accumulated in complex128. Do not use it where the Krylov basis is read to
        *reconstruct* a vector (eigenvectors on the ground-state path) — only to project
        against.
        """
        return self._dtype

    def reserve_rows(self, Py_ssize_t n):
        """Row-capacity hint for future chunks (e.g. the local basis size, an upper
        bound on the support after redistribution). Never shrinks."""
        if <int>n > self._rows_hint:
            self._rows_hint = <int>n

    cdef int _register(self, SlaterDeterminant_cpp[uint64_t] det):
        """Row index for ``det``, allocating a new (logical) zero row if absent."""
        cdef cpp_map[SlaterDeterminant_cpp[uint64_t], int].iterator f = self.support.find(det)
        if f != self.support.end():
            return dereference(f).second
        cdef int row = self.n_rows
        self.support[det] = row
        self.row_det.push_back(det)
        self.n_rows += 1
        return row

    cdef object _chunk_for_append(self, int ncol):
        """The chunk new columns go into: the last one if it has room (columns AND
        rows), else a fresh chunk sized geometrically (cols ~ current total capacity)
        and to the row hint / current rows. One append never straddles chunks."""
        # NOTE: this file compiles with wraparound=False, so cdef-list indexing must
        # use explicit positive indices — a bare [-1] is a raw out-of-bounds access.
        cdef Py_ssize_t li = len(self._chunks) - 1
        cdef object last
        if li >= 0:
            last = self._chunks[li]
            if (<int>last.shape[1]) - <int>self._used[li] >= ncol and <int>last.shape[0] >= self.n_rows:
                return last
            # Retire the last chunk: row growth (not column exhaustion) usually ends a
            # chunk's life, leaving most of its reserved columns unwritten. On a growing
            # support that dead reservation was measured at 95% of all chunk slack (~38%
            # of the buffer). Trim it to the columns it actually received; the copy is
            # one chunk wide and happens once per chunk.
            if <int>self._used[li] < <int>last.shape[1]:
                self._chunks[li] = np.ascontiguousarray(last[:, : self._used[li]])
        cdef int rows = max(self._rows_hint, 256)
        if self.n_rows > rows:
            # The support outgrew the hint (e.g. serial runs discovering excited-sector
            # determinants beyond the built basis): allocate with a 25% growth margin
            # and adopt it as the new hint, so once the support saturates no further
            # row-overflow chunks are spawned.
            rows = self.n_rows + self.n_rows // 4 + 256
            self._rows_hint = rows
        # Fixed-width column chunks: geometric growth only amortized COPIES, and the
        # chunked design has none — so fixed chunks minimize capacity slack (~one
        # chunk's worth) where geometric chunk sums doubled the steady-state footprint.
        cdef int cols = max(ncol, 32)
        arr = np.zeros((rows, cols), dtype=self._dtype)
        self._chunks.append(arr)
        self._used.append(0)
        return arr

    def append(self, list cols):
        """Append the columns of ``cols`` (a list of ManyBodyState) as new Krylov vectors.

        Routed through :meth:`append_block` over the columns' union support: a determinant
        absent from one column is an exact zero there, and the chunk entry it lands on is
        already zero, so writing it changes nothing. This is the cold path (warm-start
        ingestion of a legacy Krylov list); the recurrence appends blocks directly.
        """
        if len(cols) == 0:
            return
        self.append_block(ManyBodyBlockState.from_states(cols))

    def append_block(self, ManyBodyBlockState block):
        """Append the columns of a shared-support block as new Krylov vectors —
        the block analogue of ``append`` (reads rows directly, no per-state
        materialization)."""
        cdef Py_ssize_t ncol = <Py_ssize_t>block.b.width()
        if ncol == 0:
            return
        cdef Py_ssize_t nrow = <Py_ssize_t>block.b.rows()
        # Cache the row index handed back by _register instead of looking each key up a
        # second time in the support map during the scatter.
        cdef vector[int] rows_v
        rows_v.reserve(nrow)
        cdef Py_ssize_t r
        for r in range(nrow):
            rows_v.push_back(self._register(block.b.key(r)))
        chunk = self._chunk_for_append(<int>ncol)
        cdef Py_ssize_t li = len(self._used) - 1
        cdef int base = self._used[li]
        cdef float complex[:, :] Qv32
        cdef double complex[:, :] Qv64
        if self._single:
            Qv32 = chunk
            _scatter_block_rows(Qv32, &block.b, rows_v.data(), nrow, ncol, base)
        else:
            Qv64 = chunk
            _scatter_block_rows(Qv64, &block.b, rows_v.data(), nrow, ncol, base)
        self._used[li] = base + <int>ncol
        self.n_cols += <int>ncol

    def _plan_selection(self, object cols):
        """Per-chunk plan for reading the selected global columns, in ``cols`` order.

        Returns a list of ``(chunk_view, dest, rows_c)``: ``chunk_view`` is a
        ``(rows_c x n_k)`` slice of one chunk holding that chunk's share of the
        selection, ``dest`` indexes the rows of the overlap matrix those columns own.
        A contiguous ascending run inside a chunk yields a zero-copy *view*; otherwise
        the fancy index copies at most one chunk's worth of columns. Nothing of size
        ``n_rows x n_cols`` is ever materialized.
        """
        sel = np.arange(self.n_cols, dtype=np.intp) if cols is None else np.asarray(cols, dtype=np.intp)
        cdef list plan = []
        cdef Py_ssize_t off = 0
        cdef Py_ssize_t k, used, rows_c
        for k in range(len(self._chunks)):
            chunk = self._chunks[k]
            used = <Py_ssize_t>self._used[k]
            mask = (sel >= off) & (sel < off + used)
            if np.any(mask):
                dest = np.where(mask)[0]
                local = sel[mask] - off
                rows_c = min(<Py_ssize_t>chunk.shape[0], <Py_ssize_t>self.n_rows)
                # A contiguous ascending run slices as a view; anything else copies
                # (bounded by this chunk's column count, not by the whole store).
                # The step-1 test must be elementwise: `last - first == size - 1` also
                # holds for permutations that merely start at the min and end at the max
                # (e.g. [0, 2, 1, 3]), which must NOT be sliced as a view.
                if local.size == 1 or (local.size > 1 and np.all(np.diff(local) == 1)):
                    view = chunk[:rows_c, local[0] : local[0] + local.size]
                else:
                    view = chunk[:rows_c][:, local]
                plan.append((view, dest, rows_c))
            off += used
        return plan

    def reort(self, ManyBodyBlockState wp, object cols, int n_passes, object comm):
        """``n_passes`` of CGS2: ``O = Q[:,cols]^H wp`` (Allreduced); ``wp -= Q[:,cols] O``.

        ``cols`` is a list of column indices (the flagged bad blocks) or ``None`` for all
        columns. Returns ``(out, O_last)``: the cleaned ``wp`` as a ``ManyBodyBlockState``
        and the FINAL pass's measured (Allreduced) overlap matrix ``(len(cols) x p)`` — an
        upper bound on the residual overlap left after the projection, which the caller
        uses as the honest post-reorthogonalization W-estimate (``None`` when nothing was
        done).

        Both gemms stream over the column chunks (:meth:`_plan_selection`) rather than
        gathering ``Q[:, cols]`` into one dense buffer: the old ``_gather_columns`` +
        ``np.conj(Qsel.T)`` pair held *two* ``(n_rows x len(cols))`` copies at once —
        measured at 1.85x the whole store for a FULL sweep, which is the dominant
        avoidable peak on the Green's-function path. The transient here is bounded by
        ``n_rows * p`` (the residual ``Wd``) plus one chunk's columns.

        ``O`` is accumulated as ``conj(Q_c^T conj(Wd))`` so the large operand enters the
        gemm as a plain (possibly strided) view: ``Q_c^H`` would materialize a conjugated
        copy of the chunk, while ``conj(Wd)`` is only ``n_rows x p``.

        .. warning:: **Collective on** ``comm``: the per-pass ``Allreduce(O)`` runs
           unconditionally. ``p``, ``n_cols`` and ``n_passes`` are rank-identical (``cols``
           is broadcast by the caller), but ``n_rows`` is not — a rank owning zero
           determinants contributes zero-row gemms and must still join every ``Allreduce``.
        """
        cdef int p = <int>wp.b.width()
        if p == 0 or self.n_cols == 0 or n_passes <= 0:
            return wp, None
        cdef Py_ssize_t nrow_wp = <Py_ssize_t>wp.b.rows()
        cdef Py_ssize_t ri
        cdef int ci
        # Register wp's determinants so its components outside the current Q support get rows
        # (Q is zero there -> untouched by the projection, preserved by the scatter).
        cdef vector[int] wp_rows
        wp_rows.reserve(nrow_wp)
        for ri in range(nrow_wp):
            wp_rows.push_back(self._register(wp.b.key(ri)))
        cdef Py_ssize_t ns = self.n_rows
        Wd = np.zeros((ns, p), dtype=complex)
        cdef complex[:, :] Wv = Wd
        cdef int row
        cdef ManyBodyBlockState_cpp.Row wprow
        for ri in range(nrow_wp):
            row = wp_rows[ri]
            wprow = wp.b.row(ri)
            for ci in range(p):
                Wv[row, ci].real = wprow[ci].real()
                Wv[row, ci].imag = wprow[ci].imag()
        cdef list plan = self._plan_selection(cols)
        cdef Py_ssize_t n_sel = self.n_cols if cols is None else len(cols)
        if comm is not None:
            from mpi4py import MPI
        O = np.zeros((n_sel, p), dtype=complex)
        cdef Py_ssize_t k_plan
        cdef Py_ssize_t n_plan = len(plan)
        for _ in range(n_passes):
            O[...] = 0
            Wd_conj = np.conj(Wd)
            for k_plan in range(n_plan):
                entry = plan[k_plan]
                O[entry[1]] = np.conj(entry[0].T @ Wd_conj[: entry[2]])
            if comm is not None:
                comm.Allreduce(MPI.IN_PLACE, O, op=MPI.SUM)
            for k_plan in range(n_plan):
                entry = plan[k_plan]
                Wd[: entry[2]] -= entry[0] @ O[entry[1]]
        cdef complex[:, :] Wout = Wd
        # Build the cleaned block directly over the union of rows that are nonzero in ANY
        # column -- the same support a from_states() build over p independently-pruned
        # ManyBodyStates would produce -- instead of materializing p sparse states first.
        cdef vector[SlaterDeterminant_cpp[uint64_t]] out_keys
        cdef vector[ManyBodyBlockState_cpp.Value] out_amps
        out_keys.reserve(ns)
        out_amps.reserve(<Py_ssize_t>(ns * p))
        cdef double complex z
        cdef bint any_nonzero
        for row in range(ns):
            any_nonzero = False
            for ci in range(p):
                if Wout[row, ci].real != 0 or Wout[row, ci].imag != 0:
                    any_nonzero = True
                    break
            if not any_nonzero:
                continue
            out_keys.push_back(self.row_det[row])
            for ci in range(p):
                out_amps.push_back(_amp_to_cpp(Wout[row, ci]))
        cdef ManyBodyBlockState out = ManyBodyBlockState()
        out.b = ManyBodyBlockState_cpp.from_unsorted(out_keys, out_amps, <size_t>p)
        return out, O

    def __len__(self):
        """Number of stored Krylov columns."""
        return self.n_cols

    def __iter__(self):
        for ci in range(self.n_cols):
            yield self._materialize(ci)

    def __getitem__(self, idx):
        """Materialize column(s) back to ``ManyBodyState`` (exact scatter: every stored
        nonzero coefficient round-trips bit-identically). An integer index yields one
        state; a slice yields a list — so the store is a drop-in sequence wherever a
        Krylov list of ``ManyBodyState`` is indexed, sliced or iterated."""
        cdef Py_ssize_t ci, start, stop, step
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.n_cols)
            return [self._materialize(ci) for ci in range(start, stop, step)]
        ci = idx
        if ci < 0:
            ci += self.n_cols
        if ci < 0 or ci >= self.n_cols:
            raise IndexError(f"column {idx} out of range for {self.n_cols} stored columns")
        return self._materialize(ci)

    cdef ManyBodyState _materialize(self, Py_ssize_t col):
        cdef Py_ssize_t k = 0
        cdef Py_ssize_t off = 0
        while col - off >= <Py_ssize_t>self._used[k]:
            off += <Py_ssize_t>self._used[k]
            k += 1
        chunk = self._chunks[k]
        cdef Py_ssize_t local = col - off
        cdef Py_ssize_t rows_c = min(<Py_ssize_t>chunk.shape[0], <Py_ssize_t>self.n_rows)
        # One column, widened to complex128 (a no-op copy for a complex128 store).
        cdef double complex[::1] Qv = np.ascontiguousarray(chunk[:rows_c, local], dtype=complex)
        cdef vector[ManyBodyState_cpp.key_type] keys
        cdef vector[ManyBodyState_cpp.mapped_type] vals
        cdef Py_ssize_t row
        cdef double complex z
        for row in range(rows_c):
            z = Qv[row]
            if z.real != 0 or z.imag != 0:
                keys.push_back(self.row_det[row])
                vals.push_back(ManyBodyState_cpp.mapped_type(z.real, z.imag))
        cdef ManyBodyState ms = ManyBodyState()
        ms.v = ManyBodyState_cpp(keys, vals)
        return ms

    def _combine_dense(self, object Y, Py_ssize_t a=0, object b=None):
        """Dense ``(n_rows x Y.shape[1])`` complex128 result of ``Q[:, a:b] @ Y``.

        One gemm per chunk, accumulating partial sums (the chunk partition splits the
        column sum, so the floating-point accumulation order differs from a single-buffer
        gemm — tolerance-equivalent). Accumulation is complex128 even for a complex64
        store.
        """
        cdef Py_ssize_t bb = self.n_cols if b is None else <Py_ssize_t>b
        Ya = np.ascontiguousarray(Y, dtype=complex)
        if Ya.ndim == 1:
            Ya = Ya[:, np.newaxis]
        if Ya.shape[0] != bb - a:
            raise ValueError(f"Y rows {Ya.shape[0]} != selected columns {bb - a}")
        C = np.zeros((self.n_rows, Ya.shape[1]), dtype=complex)
        cdef Py_ssize_t off = 0
        cdef Py_ssize_t k
        cdef Py_ssize_t lo, hi, rows_c, used
        for k in range(len(self._chunks)):
            chunk = self._chunks[k]
            used = <Py_ssize_t>self._used[k]
            lo = max(a, off)
            hi = min(bb, off + used)
            if hi > lo:
                rows_c = min(<Py_ssize_t>chunk.shape[0], <Py_ssize_t>self.n_rows)
                C[:rows_c] += chunk[:rows_c, lo - off : hi - off] @ Ya[lo - a : hi - a]
            off += used
        return C

    def combine(self, object Y, Py_ssize_t a=0, object b=None, double slaterWeightMin=0.0):
        """Linear combinations ``out_k = sum_j Q[:, a:b][:, j] * Y[j, k]`` as ManyBodyStates.

        ``slaterWeightMin > 0`` prunes the outputs exactly like ``block_combine_sparse``.
        """
        C = self._combine_dense(Y, a, b)
        cdef Py_ssize_t n_out = C.shape[1]
        cdef complex[:, :] Cv = C
        cdef vector[ManyBodyState_cpp.key_type] keys
        cdef vector[ManyBodyState_cpp.mapped_type] vals
        cdef list out = []
        cdef ManyBodyState ms
        cdef Py_ssize_t row, kk
        cdef double complex z
        for kk in range(n_out):
            keys.clear()
            vals.clear()
            for row in range(self.n_rows):
                z = Cv[row, kk]
                if z.real != 0 or z.imag != 0:
                    keys.push_back(self.row_det[row])
                    vals.push_back(ManyBodyState_cpp.mapped_type(z.real, z.imag))
            ms = ManyBodyState()
            ms.v = ManyBodyState_cpp(keys, vals)
            if slaterWeightMin > 0:
                ms.prune(slaterWeightMin)
            out.append(ms)
        return out

    def combine_block(self, object Y, Py_ssize_t a=0, object b=None, double slaterWeightMin=0.0):
        """Same linear combinations as :meth:`combine`, but returned as ONE
        ``ManyBodyBlockState`` over the union of rows nonzero in any output column,
        instead of ``Y.shape[1]`` independently-pruned ``ManyBodyState`` columns.

        Built directly off the dense result via ``from_unsorted`` -- the same
        union-support construction :meth:`reort` uses -- rather than scattering into
        many sparse states first.

        ``slaterWeightMin > 0`` prunes by ROW (:meth:`ManyBodyBlockState.prune_rows`):
        a row survives if ANY column exceeds the cutoff, so a row can keep a
        sub-cutoff amplitude in one column when another column is above cutoff. This
        is NOT the same as :meth:`combine`'s per-state ``prune``, which drops each
        column's sub-cutoff entries independently -- the two only agree when no row
        is simultaneously above cutoff in one column and below it in another.
        """
        C = self._combine_dense(Y, a, b)
        cdef Py_ssize_t n_out = C.shape[1]
        cdef complex[:, :] Cv = C
        cdef Py_ssize_t ns = self.n_rows
        cdef vector[SlaterDeterminant_cpp[uint64_t]] out_keys
        cdef vector[ManyBodyBlockState_cpp.Value] out_amps
        out_keys.reserve(ns)
        out_amps.reserve(<Py_ssize_t>(ns * n_out))
        cdef Py_ssize_t row, kk
        cdef bint any_nonzero
        for row in range(ns):
            any_nonzero = False
            for kk in range(n_out):
                if Cv[row, kk].real != 0 or Cv[row, kk].imag != 0:
                    any_nonzero = True
                    break
            if not any_nonzero:
                continue
            out_keys.push_back(self.row_det[row])
            for kk in range(n_out):
                out_amps.push_back(_amp_to_cpp(Cv[row, kk]))
        cdef ManyBodyBlockState out = ManyBodyBlockState()
        out.b = ManyBodyBlockState_cpp.from_unsorted(out_keys, out_amps, <size_t>n_out)
        if slaterWeightMin > 0:
            out.prune_rows(slaterWeightMin)
        return out

    def slice_block(self, Py_ssize_t a, object b=None):
        """``ManyBodyBlockState`` of columns ``[a:b)``, built directly off the store's
        chunks -- the union-of-nonzero-rows construction :meth:`combine_block`/:meth:`reort`
        already use, but a plain column COPY per chunk instead of a ``Y``-matrix gemm
        (there is no linear combination to perform: the requested columns are lifted out
        verbatim). Replaces ``ManyBodyBlockState.from_states(self[a:b])`` at the two sites
        that used to materialize the slice as a list of independently-pruned
        ``ManyBodyState`` (:meth:`__getitem__`) and then re-merge it into one block
        (``from_states``'s union-support scatter) -- same result, one fewer round trip.

        ``b`` beyond the stored column count is clamped to ``n_cols`` (Python slice
        semantics, matching what ``self[a:b]`` returned before this method existed) --
        NOT zero-padded out to ``b``, which would silently change the output width.
        """
        cdef Py_ssize_t bb = self.n_cols if b is None else min(<Py_ssize_t>b, self.n_cols)
        cdef Py_ssize_t ncol = bb - a
        if ncol < 0:
            raise ValueError(f"slice_block: empty range [{a}:{bb})")
        cdef Py_ssize_t ns = self.n_rows
        C = np.zeros((ns, ncol), dtype=complex)
        cdef Py_ssize_t off = 0
        cdef Py_ssize_t k, lo, hi, rows_c, used
        for k in range(len(self._chunks)):
            chunk = self._chunks[k]
            used = <Py_ssize_t>self._used[k]
            lo = max(a, off)
            hi = min(bb, off + used)
            if hi > lo:
                rows_c = min(<Py_ssize_t>chunk.shape[0], <Py_ssize_t>self.n_rows)
                C[:rows_c, lo - a : hi - a] = chunk[:rows_c, lo - off : hi - off]
            off += used
        cdef complex[:, :] Cv = C
        cdef vector[SlaterDeterminant_cpp[uint64_t]] out_keys
        cdef vector[ManyBodyBlockState_cpp.Value] out_amps
        out_keys.reserve(ns)
        out_amps.reserve(<Py_ssize_t>(ns * ncol))
        cdef Py_ssize_t row, kk
        cdef bint any_nonzero
        for row in range(ns):
            any_nonzero = False
            for kk in range(ncol):
                if Cv[row, kk].real != 0 or Cv[row, kk].imag != 0:
                    any_nonzero = True
                    break
            if not any_nonzero:
                continue
            out_keys.push_back(self.row_det[row])
            for kk in range(ncol):
                out_amps.push_back(_amp_to_cpp(Cv[row, kk]))
        cdef ManyBodyBlockState out = ManyBodyBlockState()
        out.b = ManyBodyBlockState_cpp.from_unsorted(out_keys, out_amps, <size_t>ncol)
        return out

    def memory_bytes(self):
        """Estimated heap bytes: the dense chunk buffers (capacity) plus the support
        map / row_det key storage (one 32-B-class heap block per determinant key
        vector, ~72 B map-node overhead per entry)."""
        cdef size_t n_chunks = self.row_det[0].size() if self.n_rows > 0 else 1
        cdef size_t key_heap = (8 * n_chunks + 8 + 15) & (~<size_t>15)
        if key_heap < 32:
            key_heap = 32
        cdef size_t buf_bytes = 0
        for chunk in self._chunks:
            buf_bytes += chunk.nbytes
        return buf_bytes + <size_t>self.n_rows * (2 * key_heap + 72)

    def stats(self):
        """Storage breakdown of the column store, for memory profiling and regressions.

        Returns a dict with ``rows`` / ``cols`` (the logical shape), ``n_chunks``,
        ``buffer_bytes`` (allocated chunk capacity), ``payload_bytes`` (the coefficients
        actually addressed, ``sum_c rows_c * used_c * itemsize``), ``slack_bytes``
        (capacity minus payload — the price of the chunked staircase), ``support_bytes``
        (determinant keys + map nodes) and ``total_bytes`` (== :meth:`memory_bytes`).

        ``payload_bytes`` is the quantity the sizing model in
        ``impurityModel.ed.memory_estimate`` predicts; comparing it against
        ``buffer_bytes`` calibrates the chunk row/column growth policy. The slack splits
        into ``unused_col_bytes`` (columns a chunk reserved but never received, because
        row growth retired it early) and ``unused_row_bytes`` (rows reserved above the
        support that ever registered).

        ``chunks`` lists ``(rows, cols, used)`` per chunk. Every chunk but the last is
        retired and therefore trimmed, i.e. ``used == cols``.
        """
        cdef size_t nchunks_key = self.row_det[0].size() if self.n_rows > 0 else 1
        cdef size_t key_heap = (8 * nchunks_key + 8 + 15) & (~<size_t>15)
        if key_heap < 32:
            key_heap = 32
        cdef size_t buf_bytes = 0
        cdef size_t payload = 0
        cdef size_t unused_cols = 0
        cdef size_t unused_rows = 0
        cdef Py_ssize_t k, rows_c, item
        cdef list chunk_shapes = []
        for k in range(len(self._chunks)):
            chunk = self._chunks[k]
            item = <Py_ssize_t>chunk.itemsize
            buf_bytes += chunk.nbytes
            rows_c = min(<Py_ssize_t>chunk.shape[0], <Py_ssize_t>self.n_rows)
            payload += <size_t>rows_c * <size_t>self._used[k] * <size_t>item
            unused_cols += <size_t>chunk.shape[0] * <size_t>(chunk.shape[1] - self._used[k]) * <size_t>item
            unused_rows += <size_t>(chunk.shape[0] - rows_c) * <size_t>self._used[k] * <size_t>item
            chunk_shapes.append((int(chunk.shape[0]), int(chunk.shape[1]), int(self._used[k])))
        cdef size_t support_bytes = <size_t>self.n_rows * (2 * key_heap + 72)
        return {
            "rows": int(self.n_rows),
            "cols": int(self.n_cols),
            "n_chunks": len(self._chunks),
            "chunks": chunk_shapes,
            "buffer_bytes": int(buf_bytes),
            "payload_bytes": int(payload),
            "slack_bytes": int(buf_bytes - payload),
            "unused_col_bytes": int(unused_cols),
            "unused_row_bytes": int(unused_rows),
            "support_bytes": int(support_bytes),
            "total_bytes": int(buf_bytes + support_bytes),
        }
