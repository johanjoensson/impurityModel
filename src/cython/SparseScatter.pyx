# distutils: language = c++
# Compiler directives are centralized in setup.py (_DIRECTIVES); see CLAUDE.md.

"""SparseScatter.pyx
================
The scatter kernels of :class:`impurityModel.ed.DistributedSparseMatrix.DistributedSparseMatrix`: the compiled
inner loops of its matvec. The class itself is Python (it owns the key store, the growth and the exchange layout);
only the per-entry loop, which must not hold the GIL, is compiled here.
"""

import numpy as np


ctypedef fused row_index_t:
    int
    long


cdef void scatter_discovered_nogil(
    long col_begin,
    long[:] indptr,
    row_index_t[:] packed,
    double complex[:] data,
    char[:] alive,
    int p,
    int c0,
    double complex[:, ::1] X,
    long[:] base,
    long[:] goff,
    long[:] ptab,
    bint use_ptab,
    long send_rows,
    long shift,
    double complex[:, ::1] S,
    double complex[:, ::1] Y
) noexcept nogil:
    """Scatter one column block of a discovered-key matrix: entry ``packed = owner << shift | slot`` of
    local column ``col_begin + jj`` adds ``value * X[col, c0:c0+p]`` to row ``base[owner] + slot`` of
    the exchange buffers (send buffer ``S`` below ``send_rows``, self buffer ``Y`` from there on).
    With ``use_ptab`` the slot is first translated through ``ptab[goff[owner] + slot]`` (the K row's
    P slot, negative for a row outside P, which is skipped). Dead columns are skipped."""
    cdef Py_ssize_t n_cols = indptr.shape[0] - 1
    cdef Py_ssize_t jj, e, col
    cdef long owner, slot, t, entry
    cdef long mask = (<long>1 << shift) - 1
    cdef int k
    cdef double complex val
    # The two loops differ only by the translation: it is a property of the call, not of the entry, so it is
    # hoisted rather than tested per element (a P-only matrix whose slots are in order never translates).
    cdef long base0 = base[0]
    if not use_ptab and base.shape[0] == 1:
        # One owner (one rank): the row is the slot plus a constant.
        for jj in range(n_cols):
            col = col_begin + jj
            if not alive[col]:
                continue
            for e in range(indptr[jj], indptr[jj + 1]):
                t = base0 + (packed[e] & mask)
                val = data[e]
                if t < send_rows:
                    for k in range(p):
                        S[t, k] = S[t, k] + val * X[col, c0 + k]
                else:
                    t = t - send_rows
                    for k in range(p):
                        Y[t, k] = Y[t, k] + val * X[col, c0 + k]
        return
    if not use_ptab:
        for jj in range(n_cols):
            col = col_begin + jj
            if not alive[col]:
                continue
            for e in range(indptr[jj], indptr[jj + 1]):
                entry = packed[e]
                t = base[entry >> shift] + (entry & mask)
                val = data[e]
                if t < send_rows:
                    for k in range(p):
                        S[t, k] = S[t, k] + val * X[col, c0 + k]
                else:
                    t = t - send_rows
                    for k in range(p):
                        Y[t, k] = Y[t, k] + val * X[col, c0 + k]
        return
    for jj in range(n_cols):
        col = col_begin + jj
        if not alive[col]:
            continue
        for e in range(indptr[jj], indptr[jj + 1]):
            entry = packed[e]
            owner = entry >> shift
            slot = ptab[goff[owner] + (entry & mask)]
            if slot < 0:
                continue
            t = base[owner] + slot
            val = data[e]
            if t < send_rows:
                for k in range(p):
                    S[t, k] = S[t, k] + val * X[col, c0 + k]
            else:
                t = t - send_rows
                for k in range(p):
                    Y[t, k] = Y[t, k] + val * X[col, c0 + k]


def scatter_discovered(segments, char[:] alive, double complex[:, ::1] X, int c0, int p, long[:] base,
                       long[:] goff, long[:] ptab, bint use_ptab, long send_rows,
                       double complex[:, ::1] S, double complex[:, ::1] Y, long slot_bits=31):
    """Every column block of a discovered-key matrix through :func:`scatter_discovered_nogil`.
    ``S`` and ``Y`` must be zeroed by the caller. Segments are ``(col_begin, indptr, packed, data)``
    with int64 ``indptr`` and complex128 ``data``; ``packed`` is int64 (``owner << 32 | slot``) or int32
    (``owner << slot_bits | slot``)."""
    cdef long[:] indptr
    cdef int[:] packed32
    cdef long[:] packed64
    cdef double complex[:] data
    cdef long col_begin
    for col_begin_obj, indptr_obj, packed_obj, data_obj in segments:
        col_begin = col_begin_obj
        indptr = indptr_obj
        data = data_obj
        if packed_obj.dtype == np.int32:
            packed32 = packed_obj
            with nogil:
                scatter_discovered_nogil(col_begin, indptr, packed32, data, alive, p, c0, X, base, goff, ptab,
                                         use_ptab, send_rows, slot_bits, S, Y)
        else:
            packed64 = packed_obj
            with nogil:
                scatter_discovered_nogil(col_begin, indptr, packed64, data, alive, p, c0, X, base, goff, ptab,
                                         use_ptab, send_rows, 32, S, Y)
