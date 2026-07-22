# ===========================================================================
# Many-body block state
# ===========================================================================
# ManyBodyBlockState: a block (column set) of many-body states with width-0 key-only masks
# and nogil merge primitives, plus block inner/add and the MPI block pack/unpack.

cdef inline double complex _amp_to_py(ManyBodyBlockState_cpp.Value v) noexcept nogil:
    """std::complex<double> -> C double complex (no implicit conversion exists)."""
    cdef double complex z
    z.real = v.real()
    z.imag = v.imag()
    return z


cdef inline ManyBodyBlockState_cpp.Value _amp_to_cpp(double complex z) noexcept nogil:
    """C double complex -> std::complex<double>."""
    return ManyBodyBlockState_cpp.Value(z.real, z.imag)


cdef class Row:
    """A zero-copy view of ONE determinant's ``width`` amplitudes.

    What the mapping API of a state yields: ``state[det]``, and the second element of
    every ``items()`` pair. Indexable, iterable, and exportable as a ``(width,)`` complex
    array through the buffer protocol (``np.asarray(row)``); writes through it land in the
    block's storage.

    The view is non-owning and re-derives its pointer from the owning state on every
    access (never caching one) rather than blocking the state against mutation, so a
    ``Row`` sitting unused in a reference cycle can never wedge the state: it records the
    owner's generation counter at creation and compares on every access, raising
    ``RuntimeError`` if the state was structurally modified since (the row count or order
    changed -- see the invalidation list on ``ManyBodyBlockState::row``) rather than
    silently reading stale or freed memory. In-place scaling (``*=``, ``/=``) does not
    change the generation, since it does not move anything.

    ``np.asarray(row)`` is the one case where a raw pointer genuinely escapes to another
    object; for that (and only that) duration the owning state's export guard applies, the
    same as ``np.asarray(state)``.
    """
    cdef ManyBodyBlockState _owner
    cdef Py_ssize_t _row_idx
    cdef unsigned long long _generation
    cdef Py_ssize_t _size
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]

    def __cinit__(self):
        self._owner = None
        self._row_idx = 0
        self._generation = 0
        self._size = 0

    def __len__(self):
        return self._size

    cdef ManyBodyBlockState_cpp.Value* _ptr(self) except? NULL:
        """The row's current storage, freshly looked up, or a clear error if the owning
        state moved under us since this row was taken."""
        if self._owner._generation != self._generation:
            raise RuntimeError(
                "stale Row: the state was structurally modified (row count or order "
                "changed) since this row was taken"
            )
        return self._owner.b.row(<size_t>self._row_idx).data()

    def __getitem__(self, index):
        cdef ManyBodyBlockState_cpp.Value* data
        cdef Py_ssize_t i
        if isinstance(index, slice):
            data = self._ptr()
            return [_amp_to_py(data[i]) for i in range(*index.indices(self._size))]
        i = index
        if i < 0:
            i += self._size
        if i < 0 or i >= self._size:
            raise IndexError(f"column {index} out of range for width {self._size}")
        data = self._ptr()
        return _amp_to_py(data[i])

    def __setitem__(self, index, value):
        cdef ManyBodyBlockState_cpp.Value* data
        cdef Py_ssize_t i, k
        if isinstance(index, slice):
            idxs = range(*index.indices(self._size))
            data = self._ptr()
            for k, i in enumerate(idxs):
                data[i] = _amp_to_cpp(<double complex>value[k])
            return
        i = index
        if i < 0:
            i += self._size
        if i < 0 or i >= self._size:
            raise IndexError(f"column {index} out of range for width {self._size}")
        data = self._ptr()
        data[i] = _amp_to_cpp(<double complex>value)

    def __iter__(self):
        cdef ManyBodyBlockState_cpp.Value* data = self._ptr()
        cdef Py_ssize_t i
        for i in range(self._size):
            yield _amp_to_py(data[i])

    def __repr__(self):
        return "Row(" + ", ".join(repr(v) for v in self) + ")"

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef ManyBodyBlockState_cpp.Value* data = self._ptr()
        self._shape[0] = self._size
        self._strides[0] = <Py_ssize_t>sizeof(double complex)
        buffer.buf = <void*>data
        buffer.format = "Zd"
        buffer.internal = NULL
        buffer.itemsize = <Py_ssize_t>sizeof(double complex)
        buffer.len = self._size * <Py_ssize_t>sizeof(double complex)
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self._shape
        buffer.strides = self._strides
        buffer.suboffsets = NULL
        self._owner._n_exports += 1

    def __releasebuffer__(self, Py_buffer *buffer):
        self._owner._n_exports -= 1


cdef Row _make_row(ManyBodyBlockState owner, Py_ssize_t row_idx, Py_ssize_t size):
    """A view of row ``row_idx``, stamped with the owner's current generation."""
    cdef Row res = Row.__new__(Row)
    res._owner = owner
    res._row_idx = row_idx
    res._generation = owner._generation
    res._size = size
    return res


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

    ``_generation`` counts every structural mutation (a change in row count or order);
    a live ``Row`` view stamps its creation generation and compares on each access
    instead of pinning the state against mutation for as long as the ``Row`` object
    happens to be reachable -- see ``Row`` for why.
    """
    cdef ManyBodyBlockState_cpp b
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _strides[2]
    cdef int _n_exports
    cdef unsigned long long _generation

    cdef inline void _bump_generation(self) noexcept:
        self._generation += 1

    def __cinit__(self, mapping=None, width=None):
        """Build a state from a determinant mapping, or an empty one of a given width.

        ``ManyBodyBlockState()`` is the polymorphic zero: width 0 and no rows, which
        adopts a width on the first ``+=`` (or ``__setitem__``). ``ManyBodyBlockState({})``
        is the same polymorphic zero -- an empty mapping carries no width information.
        ``ManyBodyBlockState(width=p)`` is the zero block of an explicit width ``p``
        (widths are then checked, not adopted). ``mapping`` maps determinants to a scalar
        amplitude (width 1) or to a sequence of ``width`` amplitudes.
        """
        cdef vector[SlaterDeterminant_cpp[uint64_t]] keys
        cdef vector[ManyBodyBlockState_cpp.Value] amps
        cdef SlaterDeterminant sd
        cdef Py_ssize_t w
        if not mapping:
            self.b = ManyBodyBlockState_cpp(<size_t>(0 if width is None else width))
            return
        first = next(iter(mapping.values()))
        w = 1 if (np.isscalar(first) or isinstance(first, complex)) else len(first)
        if width is not None and <Py_ssize_t>width != w:
            raise ValueError(f"width {width} does not match the {w} amplitudes given per determinant")
        keys.reserve(len(mapping))
        amps.reserve(len(mapping) * w)
        for key, value in mapping.items():
            sd = <SlaterDeterminant?>key
            keys.push_back(sd.s)
            if w == 1 and (np.isscalar(value) or isinstance(value, complex)):
                amps.push_back(_amp_to_cpp(<double complex>value))
            else:
                if len(value) != w:
                    raise ValueError(f"expected {w} amplitudes per determinant, got {len(value)}")
                for c in value:
                    amps.push_back(_amp_to_cpp(<double complex>c))
        # from_unsorted sorts (stable) and dedups in C++, so no Python-level sort of
        # mapping.items() is needed -- dict keys are unique anyway, so "first wins on
        # duplicates" never actually triggers here; it matters for from_unsorted's other
        # (non-dict) callers.
        self.b = ManyBodyBlockState_cpp.from_unsorted(keys, amps, <size_t>w)

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

    @staticmethod
    def from_keys_and_amps(ManyBodyBlockState src, amps):
        """Build a block over ``src``'s determinant support with new amplitudes.

        The write-back counterpart of the buffer protocol (``np.asarray(block)`` exports the
        ``(rows x width)`` coefficients for reading): a dense array computed from that view —
        the orthonormal factor of a QR, say — becomes a block again without a detour through
        ``ManyBodyState`` objects. ``amps`` must have exactly ``len(src)`` rows; its column
        count is the new width, which may differ from ``src``'s (a deflated QR returns fewer
        columns than it was given).
        """
        arr = np.ascontiguousarray(amps, dtype=complex)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.ndim != 2 or arr.shape[0] != <Py_ssize_t>src.b.rows():
            raise ValueError(f"amps rows {arr.shape[0]} != block rows {src.b.rows()}")
        cdef Py_ssize_t ns = arr.shape[0]
        cdef Py_ssize_t w = arr.shape[1]
        cdef double complex[:, ::1] av = arr
        cdef vector[ManyBodyBlockState_cpp.Value] vals
        vals.resize(ns * w)
        if ns > 0 and w > 0:
            memcpy(vals.data(), &av[0, 0], ns * w * sizeof(ManyBodyBlockState_cpp.Value))
        cdef vector[SlaterDeterminant_cpp[uint64_t]] keys = src.b.keys()
        cdef ManyBodyBlockState out = ManyBodyBlockState()
        out.b = ManyBodyBlockState_cpp(move(keys), move(vals), <size_t>w)
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

    # --- mapping surface: determinant -> Row --------------------------------
    # Lookup is a binary search over the sorted support. Inserting a NEW
    # determinant shifts the tail of both arrays, so building a state one key at
    # a time is O(n^2) -- build from a dict or from_states instead.

    def __contains__(self, SlaterDeterminant key):
        return self.b.contains(key.s)

    def __getitem__(self, SlaterDeterminant key):
        """The determinant's row of amplitudes, or ``KeyError`` if it is absent."""
        cdef size_t r = self.b.find_row(key.s)
        if r == self.b.rows():
            raise KeyError(repr(key))
        return _make_row(self, <Py_ssize_t>r, <Py_ssize_t>self.b.width())

    def __setitem__(self, SlaterDeterminant key, value):
        """Write a determinant's row, inserting it when absent.

        ``value`` is either a scalar (broadcast across the row) or a sequence of
        ``width`` amplitudes. On a width-0, row-less state (the polymorphic zero) the
        first assignment adopts ``value``'s width; on a width-0 state that already has
        rows (a key-only mask), assigning amplitudes is rejected -- use ``merge_keys``
        to add keys to a mask instead.
        """
        cdef ManyBodyBlockState_cpp.Row span
        cdef Py_ssize_t w, c
        cdef bint is_scalar = np.isscalar(value) or isinstance(value, complex)
        cdef bint existed
        if self.b.width() == 0:
            if self.b.rows() > 0:
                raise ValueError(
                    "cannot assign amplitudes into a width-0 (key-only) mask block; "
                    "use merge_keys to add keys instead"
                )
            if self._n_exports > 0:
                raise RuntimeError("cannot assign a row while a buffer view is exported")
            self.b = ManyBodyBlockState_cpp(<size_t>(1 if is_scalar else len(value)))
            self._bump_generation()
        w = <Py_ssize_t>self.b.width()
        if not is_scalar and len(value) != w:
            raise ValueError(f"expected {w} amplitudes, got {len(value)}")
        if self._n_exports > 0:
            raise RuntimeError("cannot assign a row while a buffer view is exported")
        existed = self.b.contains(key.s)
        span = self.b[key.s]  # inserts a zero row when absent
        if not existed:
            self._bump_generation()
        if is_scalar:
            for c in range(w):
                span[c] = _amp_to_cpp(<double complex>value)
            return
        for c in range(w):
            span[c] = _amp_to_cpp(<double complex>value[c])

    def get(self, SlaterDeterminant key, default=None):
        """The determinant's row, or ``default`` when it is absent (no insertion)."""
        cdef size_t r = self.b.find_row(key.s)
        if r == self.b.rows():
            return default
        return _make_row(self, <Py_ssize_t>r, <Py_ssize_t>self.b.width())

    def __iter__(self):
        """Iterate the support determinants, in row (sorted) order."""
        return iter(self.keys())

    def values(self):
        """The rows, in row order."""
        cdef Py_ssize_t r
        cdef Py_ssize_t w = <Py_ssize_t>self.b.width()
        for r in range(<Py_ssize_t>self.b.rows()):
            yield _make_row(self, r, w)

    def items(self):
        """``(determinant, row)`` pairs, in row order."""
        cdef SlaterDeterminant sd
        cdef Py_ssize_t r
        cdef Py_ssize_t w = <Py_ssize_t>self.b.width()
        for r in range(<Py_ssize_t>self.b.rows()):
            sd = SlaterDeterminant.__new__(SlaterDeterminant)
            sd.s = self.b.key(r)
            yield sd, _make_row(self, r, w)

    def to_dict(self):
        """``{determinant: (width,) complex array}`` -- a detached copy of the block."""
        cdef dict res = {}
        cdef SlaterDeterminant sd
        cdef Py_ssize_t r
        cdef Py_ssize_t w = <Py_ssize_t>self.b.width()
        for r in range(<Py_ssize_t>self.b.rows()):
            sd = SlaterDeterminant.__new__(SlaterDeterminant)
            sd.s = self.b.key(r)
            res[sd] = np.array(_make_row(self, r, w), dtype=complex)
        return res

    def erase(self, SlaterDeterminant key):
        """Drop a determinant's row. Returns True if it was present."""
        if self._n_exports > 0:
            raise RuntimeError("cannot erase a row while a buffer view is exported")
        cdef bint removed = self.b.erase(key.s) > 0
        if removed:
            self._bump_generation()
        return removed

    def clear(self):
        """Drop every row (the width is kept)."""
        if self._n_exports > 0:
            raise RuntimeError("cannot clear while a buffer view is exported")
        self.b.clear()
        self._bump_generation()

    def is_empty(self):
        """True when the state holds no determinants."""
        return self.b.empty()

    def insert_rows(self, keys, amps):
        """Bulk-insert (or overwrite) ``(determinant, row)`` pairs in ``O(n log n)``
        total, instead of the ``O(rows)``-per-insertion of repeated ``state[det] = ...``
        once the state already holds rows (``O(n^2)`` over a build loop of ``n`` new
        keys). Duplicate keys are last-write-wins (``dict.update`` semantics): the LAST
        matching entry in ``keys`` wins, whether the match is against another entry in
        this same call or an existing row.

        ``keys`` is any sequence of ``SlaterDeterminant``; ``amps`` is ``(len(keys),
        width)`` (or ``(len(keys),)`` at width 1). On the polymorphic zero (width 0, no
        rows) the width is adopted from ``amps``, exactly like ``__setitem__`` -- unless
        ``keys`` is empty, which is always a true no-op (nothing to adopt a width from).
        """
        if self._n_exports > 0:
            raise RuntimeError("cannot insert_rows while a buffer view is exported")
        keys = list(keys)
        cdef Py_ssize_t n_new = len(keys)
        if n_new == 0:
            return
        cdef Py_ssize_t w = <Py_ssize_t>self.b.width()
        amps_arr = np.ascontiguousarray(amps, dtype=complex)
        if amps_arr.ndim == 1:
            amps_arr = amps_arr[:, np.newaxis]
        if w == 0 and self.b.rows() == 0:
            w = amps_arr.shape[1]
        if amps_arr.shape != (n_new, w):
            raise ValueError(f"amps shape {tuple(amps_arr.shape)} != ({n_new}, {w})")
        cdef vector[SlaterDeterminant_cpp[uint64_t]] all_keys
        cdef vector[ManyBodyBlockState_cpp.Value] all_amps
        cdef Py_ssize_t n_old = <Py_ssize_t>self.b.rows()
        all_keys.reserve(n_new + n_old)
        all_amps.reserve((n_new + n_old) * w)
        cdef SlaterDeterminant sd
        cdef double complex[:, ::1] av = amps_arr
        cdef Py_ssize_t i, c, r
        cdef ManyBodyBlockState_cpp.Row old_row
        # `from_unsorted` stable-sorts by key and keeps the FIRST of each run of equal
        # keys, so whichever row we push first among a set of duplicates is the one that
        # survives. New rows must win over old ones (this call is meant to replace/extend
        # them), so they go first; and within the new batch itself, a LATER entry of
        # `keys` must win over an earlier one (dict.update semantics), so the new rows are
        # pushed in REVERSE order -- the true-last entry ends up first, and therefore wins.
        for i in range(n_new - 1, -1, -1):
            sd = <SlaterDeterminant?>keys[i]
            all_keys.push_back(sd.s)
            for c in range(w):
                all_amps.push_back(_amp_to_cpp(av[i, c]))
        for r in range(n_old):
            all_keys.push_back(self.b.key(r))
            old_row = self.b.row(r)
            for c in range(w):
                all_amps.push_back(old_row[c])
        self.b = ManyBodyBlockState_cpp.from_unsorted(all_keys, all_amps, <size_t>w)
        self._bump_generation()

    # --- vector space -------------------------------------------------------

    def norm2(self):
        """Frobenius norm squared: the sum over every stored amplitude."""
        return self.b.norm2()

    def norm(self):
        """Frobenius norm."""
        return self.b.norm()

    def max_norm2(self):
        """Largest ``|amplitude|^2`` anywhere in the block."""
        return self.b.max_norm2()

    def count_above(self, double cutoff2):
        """Rows whose largest column ``|amplitude|^2`` exceeds ``cutoff2``."""
        return self.b.count_above(cutoff2)

    def truncate(self, size_t max_rows):
        """Keep the ``max_rows`` rows with the largest row-max ``|amplitude|^2``.

        Ties at the cutoff are all kept, so the result can exceed ``max_rows``.
        """
        if self.b.width() == 0 and self.b.rows() > 0:
            # A width-0 mask has no amplitudes, so every row-max is 0 and this would
            # silently keep everything regardless of max_rows -- reject rather than
            # let that pass for "it worked".
            raise ValueError("cannot truncate a width-0 (key-only) mask block")
        if self._n_exports > 0:
            raise RuntimeError("cannot truncate while a buffer view is exported")
        with nogil:
            self.b.truncate(max_rows)
        self._bump_generation()

    def add_scaled(self, ManyBodyBlockState other, ManyBodyBlockState_cpp.Value scalar):
        """In place: ``self += scalar * other``, over the union support.

        Widths must match, except that a width-0 row-less state (a default
        construction) is the polymorphic zero and adopts ``other``'s width.
        """
        if self._n_exports > 0:
            raise RuntimeError("cannot add while a buffer view is exported")
        with nogil:
            self.b.add_scaled(other.b, scalar)
        self._bump_generation()
        return self

    def __add__(self, ManyBodyBlockState other):
        cdef ManyBodyBlockState res = ManyBodyBlockState()
        with nogil:
            res.b = self.b + other.b
        return res

    def __iadd__(self, ManyBodyBlockState other):
        """``self += other``: true in-place, via ``add_scaled`` (bumps the generation,
        since the union support can outgrow this state's storage -- unlike ``*=``/``/=``,
        a live ``Row`` does not survive it)."""
        return self.add_scaled(other, 1.0 + 0j)

    def __sub__(self, ManyBodyBlockState other):
        cdef ManyBodyBlockState res = ManyBodyBlockState()
        with nogil:
            res.b = self.b - other.b
        return res

    def __isub__(self, ManyBodyBlockState other):
        return self.add_scaled(other, -1.0 + 0j)

    def __neg__(self):
        cdef ManyBodyBlockState res = ManyBodyBlockState()
        with nogil:
            res.b = -self.b
        return res

    def __mul__(self, ManyBodyBlockState_cpp.Value s):
        cdef ManyBodyBlockState res = ManyBodyBlockState()
        with nogil:
            res.b = self.b * s
        return res

    def __rmul__(self, ManyBodyBlockState_cpp.Value s):
        return self.__mul__(s)

    def __imul__(self, ManyBodyBlockState_cpp.Value s):
        """``self *= s``: true in-place scaling of every stored amplitude, without
        touching the row layout -- a live ``Row`` survives this."""
        with nogil:
            self.b.scale(s)
        return self

    def __truediv__(self, ManyBodyBlockState_cpp.Value s):
        cdef ManyBodyBlockState res = ManyBodyBlockState()
        with nogil:
            res.b = self.b / s
        return res

    def __itruediv__(self, ManyBodyBlockState_cpp.Value s):
        with nogil:
            self.b.scale_inv(s)
        return self

    def prune_rows(self, double cutoff):
        """Drop rows where ALL columns satisfy the ``ManyBodyState.prune`` test
        (``|amp|^2 <= cutoff^2``); a row survives if ANY column survives. This keeps
        the support shared across the block — the deliberate semantic difference vs
        pruning p independent states."""
        if self._n_exports > 0:
            raise RuntimeError("cannot prune_rows while a buffer view is exported (np.asarray view alive)")
        self.b.prune_rows(cutoff)
        self._bump_generation()

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
        self._bump_generation()

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
        self._bump_generation()

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
        cdef ManyBodyBlockState_cpp.Row row
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

    def select(self, cols):
        """A new block of the given columns (any order, repeats allowed), same support.

        Expressed as ``self @ Y`` for a 0/1 selection matrix ``Y`` -- reuses
        ``combine_columns`` exactly, rather than a second hand-rolled gather.
        """
        cdef Py_ssize_t w = <Py_ssize_t>self.b.width()
        cols = list(cols)
        cdef Py_ssize_t n = len(cols)
        Y = np.zeros((w, n), dtype=complex)
        cdef Py_ssize_t k, c
        for k in range(n):
            c = cols[k]
            if c < 0:
                c += w
            if c < 0 or c >= w:
                raise IndexError(f"column {cols[k]} out of range for width {w}")
            Y[c, k] = 1.0
        return self.combine_columns(Y)

    def column(self, Py_ssize_t i):
        """The ``i``-th column, as a width-1 block over the same support."""
        return self.select([i])

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


def block_inner_scalar(ManyBodyBlockState a, ManyBodyBlockState b):
    """Scalar inner product ``<a|b>`` for width-1 states -- the block counterpart of
    ``ManyBodyState``'s free ``inner()``, for callers with nothing but width-1 blocks.
    ``block_inner_cy`` is the general (any width) Gram matrix this reduces to.

    Deliberately not named ``block_inner``: ``_reort.pxi`` (compiled into
    ``BlockLanczosArray``) already exports an array/state-dispatching ``block_inner``
    that several solver modules import by that exact name (``BiCGSTAB.pyx``,
    ``GMRES.pyx``, ``BlockLanczos.pyx``) -- reusing the name here would not collide
    today (different module), but a caller pulling both into one namespace later would
    silently shadow one with the other.
    """
    if a.b.width() != 1 or b.b.width() != 1:
        raise ValueError(f"block_inner_scalar requires width-1 states, got widths {a.b.width()} and {b.b.width()}")
    return block_inner_cy(a, b)[0, 0]


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
