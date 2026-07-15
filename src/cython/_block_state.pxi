cdef class ManyBodyBlockState:
    r"""A block of ``p`` many-body vectors over ONE shared Slater-determinant support.

    The hot-loop counterpart of ``ManyBodyState`` (which stays the single-vector
    boundary type): the union support is stored once as a sorted key vector, the
    coefficients as a row-major ``(rows x width)`` dense array. This is the container
    the block-Lanczos loop operates on so ``ManyBodyOperator::apply`` amortizes the
    term/sign/accumulator work over the block width and the block linear algebra runs
    as dense row-block BLAS (see the Phase-2 plan in
    ``doc/plans/blocklanczos_partial_perf_memory.md``).

    Supports the buffer protocol: ``np.asarray(block)`` is a zero-copy writable
    ``(rows, width)`` complex view (the array keeps this object alive). Mutating the
    row structure (``prune_rows``) while such a view is exported raises, because the
    reallocation would dangle the view.
    """
    cdef ManyBodyBlockState_cpp b
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _strides[2]
    cdef int _n_exports

    @staticmethod
    def from_states(list states):
        """Build the block from a list of ``ManyBodyState`` over their union support.

        Missing determinants of a column hold exact zeros; every stored coefficient
        round-trips bit-identically through ``to_states``.
        """
        cdef Py_ssize_t p = len(states)
        cdef ManyBodyBlockState out = ManyBodyBlockState()
        if p == 0:
            return out
        cdef vector[ManyBodyState_cpp*] ptrs
        cdef ManyBodyState ms
        for obj in states:
            ms = <ManyBodyState?>obj
            ptrs.push_back(&ms.v)

        cdef vector[SlaterDeterminant_cpp[uint64_t]] support
        cdef ManyBodyState_cpp.iterator it
        cdef Py_ssize_t ci
        for ci in range(p):
            it = ptrs[ci].begin()
            while it != ptrs[ci].end():
                support.push_back(dereference(it).first)
                preincrement(it)
        sort(support.begin(), support.end())
        support.erase(unique(support.begin(), support.end()), support.end())
        cdef Py_ssize_t ns = <Py_ssize_t>support.size()

        cdef vector[ManyBodyBlockState_cpp.Value] amps
        amps.resize(ns * p)  # value-initialized: exact zeros
        cdef Py_ssize_t row
        for ci in range(p):
            it = ptrs[ci].begin()
            while it != ptrs[ci].end():
                row = lower_bound(support.begin(), support.end(), dereference(it).first) - support.begin()
                amps[row * p + ci] = dereference(it).second
                preincrement(it)
        out.b = ManyBodyBlockState_cpp(move(support), move(amps), <size_t>p)
        return out

    def to_states(self):
        """Materialize the columns back to a list of ``ManyBodyState`` (exact-zero
        entries are skipped, so a ``from_states`` round-trip is bit-identical)."""
        cdef Py_ssize_t p = <Py_ssize_t>self.b.width()
        cdef Py_ssize_t ns = <Py_ssize_t>self.b.rows()
        cdef list out = []
        cdef vector[ManyBodyState_cpp.key_type] keys
        cdef vector[ManyBodyState_cpp.mapped_type] vals
        cdef ManyBodyState new_ms
        cdef Py_ssize_t ci, row
        cdef ManyBodyBlockState_cpp.Value z
        for ci in range(p):
            keys.clear()
            vals.clear()
            for row in range(ns):
                z = self.b.data()[row * p + ci]
                if z.real() != 0 or z.imag() != 0:
                    keys.push_back(self.b.key(row))
                    vals.push_back(z)
            new_ms = ManyBodyState()
            new_ms.v = ManyBodyState_cpp(keys, vals)
            out.append(new_ms)
        return out

    @property
    def width(self):
        """Number of block vectors (columns)."""
        return self.b.width()

    def __len__(self):
        """Number of shared-support determinants (rows)."""
        return self.b.rows()

    def prune_rows(self, double cutoff):
        """Drop rows where ALL columns satisfy the ``ManyBodyState.prune`` test
        (``|amp|^2 <= cutoff^2``); a row survives if ANY column survives. This keeps
        the support shared across the block — the deliberate semantic difference vs
        pruning p independent states."""
        if self._n_exports > 0:
            raise RuntimeError("cannot prune_rows while a buffer view is exported (np.asarray view alive)")
        self.b.prune_rows(cutoff)

    def keep_rows(self, ManyBodyBlockState mask):
        """Keep only rows whose determinant appears in ``mask``'s support (the
        set-intersection complement of ``prune_rows``): a linear merge over the two
        sorted key vectors, no Python-object traffic. ``mask`` is any block over the
        retained determinant set (its amplitudes are ignored); build it once, e.g. via
        ``from_states([ManyBodyState(dict.fromkeys(keys, 1.0))])``, and reuse it every
        step. Raises while a buffer view is exported, like ``prune_rows``."""
        if self._n_exports > 0:
            raise RuntimeError("cannot keep_rows while a buffer view is exported (np.asarray view alive)")
        with nogil:
            self.b.keep_rows(mask.b.keys())

    def row_max_norms2(self):
        """Per-row max column ``|amp|^2`` and the row keys, as ``(keys, norms2)``.

        ``keys`` is the full support in row order (including exact-zero rows, unlike
        ``support_keys(0.0)`` which drops them), so ``keys[i]`` is aligned with
        ``norms2[i]``. This is the importance measure used to rank boundary
        determinants when a capped Green's-function basis overflows."""
        cdef Py_ssize_t n = <Py_ssize_t>self.b.rows()
        res = np.zeros(n, dtype=float)
        cdef double[:] rv = res
        if n > 0:
            with nogil:
                self.b.row_max_norm2(&rv[0])
        cdef list keys = []
        cdef SlaterDeterminant sd
        cdef Py_ssize_t r
        for r in range(n):
            sd = SlaterDeterminant()
            sd.s = self.b.key(r)
            keys.append(sd)
        return keys, res

    def count_rows_in(self, ManyBodyBlockState mask):
        """Number of rows whose determinant appears in ``mask``'s support (linear
        merge over the two sorted key vectors; no Python-object traffic)."""
        cdef size_t n
        with nogil:
            n = self.b.count_rows_in(mask.b.keys())
        return n

    def new_row_max_norms2(self, ManyBodyBlockState mask):
        """Max column ``|amp|^2`` of every row NOT in ``mask``, as a float array in
        row order — the candidate-importance array for the capped recurrence's
        overflow-step amplitude bisection."""
        cdef vector[double] out
        with nogil:
            self.b.new_row_max_norm2(mask.b.keys(), out)
        res = np.empty(out.size(), dtype=float)
        cdef double[:] rv = res
        cdef Py_ssize_t i
        for i in range(<Py_ssize_t>out.size()):
            rv[i] = out[i]
        return res

    def keys_new_above(self, ManyBodyBlockState mask, double cutoff2):
        """Width-0 key-only block of the rows NOT in ``mask`` whose max column
        ``|amp|^2`` exceeds ``cutoff2`` (the admitted boundary determinants once the
        overflow bisection has fixed the cutoff)."""
        cdef ManyBodyBlockState res = ManyBodyBlockState()
        with nogil:
            res.b = self.b.keys_new_above(mask.b.keys(), cutoff2)
        return res

    def key_union(self, ManyBodyBlockState other):
        """Width-0 key-only block holding the sorted union of both supports
        (amplitudes ignored) — the retained-set mask accumulator of the capped
        recurrence."""
        cdef ManyBodyBlockState res = ManyBodyBlockState()
        with nogil:
            res.b = self.b.key_union(other.b)
        return res

    def merge_keys(self, ManyBodyBlockState other):
        """In-place ``key_union`` for width-0 mask blocks: only the genuinely new
        keys are copied, the existing ones are moved — the per-step retained-mask
        accumulate of the capped recurrence. Requires ``width == 0`` and no exported
        buffer view."""
        if self.b.width() != 0:
            raise ValueError("merge_keys requires a width-0 (key-only) mask block")
        if self._n_exports > 0:
            raise RuntimeError("cannot merge_keys while a buffer view is exported (np.asarray view alive)")
        with nogil:
            self.b.merge_keys(other.b)

    def col_norm2(self):
        """Per-column squared L2 norms as a float array of length ``width``."""
        res = np.zeros(self.b.width(), dtype=float)
        cdef double[:] rv = res
        if self.b.width() > 0:
            self.b.col_norm2(&rv[0])
        return res

    def memory_bytes(self):
        """Estimated heap bytes: dense amplitude array + one heap block per key vector."""
        cdef size_t n_chunks = self.b.key(0).size() if self.b.rows() > 0 else 1
        cdef size_t key_heap = (8 * n_chunks + 8 + 15) & (~<size_t>15)
        if key_heap < 32:
            key_heap = 32
        return self.b.rows() * (16 * self.b.width() + key_heap + sizeof(SlaterDeterminant_cpp[uint64_t]))

    def support_keys(self, double min_amp=0.0):
        """The shared-support determinants (``SlaterDeterminant`` wrappers) of rows
        where ANY column has ``|amp| > min_amp`` — the block analogue of iterating the
        union of the columns' flat_map keys with a per-entry amplitude filter (the
        union of per-column tests equals the row-max test)."""
        cdef double cutoff2 = min_amp * min_amp
        cdef Py_ssize_t p = <Py_ssize_t>self.b.width()
        cdef Py_ssize_t r, c
        cdef const ManyBodyBlockState_cpp.Value* row
        cdef list out = []
        cdef SlaterDeterminant sd
        for r in range(<Py_ssize_t>self.b.rows()):
            row = self.b.row(r)
            for c in range(p):
                if row[c].real() * row[c].real() + row[c].imag() * row[c].imag() > cutoff2:
                    sd = SlaterDeterminant()
                    sd.s = self.b.key(r)
                    out.append(sd)
                    break
        return out

    def keys(self):
        """Every shared-support determinant in row order, amplitudes ignored.

        Works at ``width == 0``, which ``support_keys`` cannot: that filters on the row's
        column amplitudes, of which a key-only mask block has none. This is the accessor
        for materializing the output of ``keys_new_above`` / ``key_union`` / ``merge_keys``
        back into Python ``SlaterDeterminant`` objects."""
        cdef Py_ssize_t n = <Py_ssize_t>self.b.rows()
        cdef list out = []
        cdef SlaterDeterminant sd
        cdef Py_ssize_t r
        for r in range(n):
            sd = SlaterDeterminant()
            sd.s = self.b.key(r)
            out.append(sd)
        return out

    def combine_columns(self, Y):
        """New block ``OUT = self @ Y`` on the same support: ``out[det, k] =
        sum_j self[det, j] * Y[j, k]``. The j-ascending accumulation matches
        ``block_combine_sparse`` (add_scaled_multi) bit-for-bit. Used for the
        Cholesky-QR normalization ``q_next = wp @ beta_inv`` (the output width may
        shrink under deflation)."""
        Ya = np.ascontiguousarray(Y, dtype=complex)
        if Ya.ndim == 1:
            Ya = Ya[:, np.newaxis]
        if Ya.shape[0] != self.b.width():
            raise ValueError(f"Y rows {Ya.shape[0]} != block width {self.b.width()}")
        cdef double complex[:, ::1] yv = Ya
        cdef ManyBodyBlockState out = ManyBodyBlockState()
        cdef ManyBodyBlockState_cpp.Value* yptr = NULL
        if Ya.size > 0:
            yptr = <ManyBodyBlockState_cpp.Value*>&yv[0, 0]
        cdef size_t wout = Ya.shape[1]
        with nogil:
            out.b = c_block_combine_cols(self.b, yptr, wout)
        return out

    def copy(self):
        """Deep copy (independent key and amplitude storage)."""
        cdef ManyBodyBlockState res = ManyBodyBlockState()
        res.b = self.b
        return res

    def __eq__(self, other):
        if not isinstance(other, ManyBodyBlockState):
            return NotImplemented
        return self.b == (<ManyBodyBlockState>other).b

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self._shape[0] = <Py_ssize_t>self.b.rows()
        self._shape[1] = <Py_ssize_t>self.b.width()
        self._strides[0] = self._shape[1] * <Py_ssize_t>sizeof(double complex)
        self._strides[1] = <Py_ssize_t>sizeof(double complex)
        buffer.buf = <void*>self.b.data()
        buffer.format = "Zd"
        buffer.internal = NULL
        buffer.itemsize = <Py_ssize_t>sizeof(double complex)
        buffer.len = self._shape[0] * self._shape[1] * <Py_ssize_t>sizeof(double complex)
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self._shape
        buffer.strides = self._strides
        buffer.suboffsets = NULL
        self._n_exports += 1

    def __releasebuffer__(self, Py_buffer *buffer):
        self._n_exports -= 1


def pack_block_fused_cy(ManyBodyBlockState block, int comm_size, size_t chunks_per_state):
    """Pack a block state into a single rank-ordered byte buffer for a one-shot
    ``Neighbor_alltoallv(MPI.BYTE)`` redistribute. One entry per shared-support row:
    ``[det | width x complex amp]`` — no per-entry psi index (the column position
    identifies the vector), so the wire cost per determinant is ``state_bytes +
    16*width`` instead of ``width * (state_bytes + 20)``. Ownership uses the same
    ``routing_hash() % comm_size`` as ``pack_psis_fused_cy``."""
    cdef vector[int64_t] send_counts
    cdef vector[int] owners

    with nogil:
        c_pack_block_count(block.b, comm_size, send_counts, owners)

    # Serialize straight into the numpy buffer handed to MPI (Phase 4): no
    # intermediate std::vector wire buffer and no full copy-out.
    cdef size_t total = 0
    cdef int rk
    for rk in range(comm_size):
        total += <size_t>send_counts[rk]
    cdef size_t bpe = chunks_per_state * 8 + block.b.width() * 16
    send_counts_np = np.zeros(comm_size, dtype=np.int64)
    send_buf_np = np.empty(total * bpe, dtype=np.uint8)

    cdef int64_t[:] send_counts_view = send_counts_np
    cdef uint8_t[:] send_buf_view = send_buf_np
    if comm_size > 0:
        memcpy(&send_counts_view[0], <void*>send_counts.data(), comm_size * sizeof(int64_t))
    if total > 0:
        with nogil:
            c_pack_block_fill(block.b, comm_size, chunks_per_state, send_counts, owners, <char*>&send_buf_view[0])

    return send_counts_np, send_buf_np


def unpack_block_fused_cy(int comm_size, size_t width, int64_t[:] recv_counts, uint8_t[:] recv_buf, size_t chunks_per_state):
    """Rebuild a ``ManyBodyBlockState`` from the received fused byte buffer. Rows for
    the same determinant arriving from different ranks are summed in arrival order,
    matching ``unpack_psis_fused_cy``'s accumulate semantics bit-for-bit per column."""
    cdef vector[int64_t] c_recv_counts

    if comm_size > 0:
        c_recv_counts.assign(<int64_t*> &recv_counts[0], <int64_t*> &recv_counts[0] + comm_size)

    # Parse straight out of the numpy receive buffer (no std::vector copy).
    cdef const char* buf_ptr = NULL
    if recv_buf.shape[0] > 0:
        buf_ptr = <const char*> &recv_buf[0]

    cdef ManyBodyBlockState res = ManyBodyBlockState()
    with nogil:
        res.b = c_unpack_block_fused(comm_size, width, c_recv_counts, buf_ptr, chunks_per_state)
    return res


def block_inner_cy(ManyBodyBlockState A, ManyBodyBlockState B):
    """Block Gram matrix ``C[i, j] = <A_i | B_j>`` over the merged supports.

    Merge-join over the two sorted key vectors; the determinant summation order
    equals the sorted flat_map order of ``inner_multi`` over lists, so the result is
    bit-for-bit identical to the scalar path. Rank-local (no MPI)."""
    res = np.zeros((A.b.width(), B.b.width()), dtype=complex)
    cdef double complex[:, ::1] rv = res
    if A.b.width() > 0 and B.b.width() > 0:
        with nogil:
            c_block_inner(A.b, B.b, <ManyBodyBlockState_cpp.Value*>&rv[0, 0])
    return res


def block_add_scaled_cy(ManyBodyBlockState A, ManyBodyBlockState B, C):
    """New block ``OUT = A + B @ C`` over the union support: ``out[det, j] =
    A[det, j] + sum_i B[det, i] * C[i, j]`` — the block analogue of
    ``add_scaled_multi(target=A, source=B, coeffs=C)``, with the same i-ascending
    accumulation order (bit-for-bit) but returning a NEW block instead of mutating
    (the union support can outgrow A's storage)."""
    Ca = np.ascontiguousarray(C, dtype=complex)
    if Ca.ndim != 2 or Ca.shape[0] != B.b.width() or Ca.shape[1] != A.b.width():
        raise ValueError(f"coeffs shape {Ca.shape} != (B.width={B.b.width()}, A.width={A.b.width()})")
    cdef double complex[:, ::1] cv = Ca
    cdef ManyBodyBlockState out = ManyBodyBlockState()
    cdef ManyBodyBlockState_cpp.Value* cptr = NULL
    if Ca.size > 0:
        cptr = <ManyBodyBlockState_cpp.Value*>&cv[0, 0]
    with nogil:
        out.b = c_block_add_scaled(A.b, B.b, cptr)
    return out


def parallel_apply_build():
    """True when the extension was compiled with the opt-in threaded apply
    (``IMPURITYMODEL_PARALLEL=1`` at install time). The threaded merge changes the
    duplicate-accumulation order, so bit-for-bit reproducibility of apply results is
    a serial-build property — tests use this to pick exact vs tolerance assertions."""
    return bool(c_apply_parallel_build())
