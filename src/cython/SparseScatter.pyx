# distutils: language = c++
# Compiler directives are centralized in setup.py (_DIRECTIVES); see CLAUDE.md.

"""SparseScatter.pyx
================
The scatter kernels of :class:`impurityModel.ed.DistributedSparseMatrix.DistributedSparseMatrix`: the compiled
inner loops of its matvec. The class itself is Python (it owns the key store, the growth and the exchange layout);
only the per-entry loop, which must not hold the GIL, is compiled here. Also the fused per-row update of
:class:`impurityModel.ed.gf_shift_recycling.ShiftedGalerkin`, which is the same kind of memory-bound sweep.
"""

import numpy as np
from libc.stdlib cimport malloc, free


ctypedef fused row_index_t:
    int
    long

ctypedef float complex c64_t
ctypedef double complex c128_t

# The storage of a ShiftedGalerkin accumulator: the arithmetic is always double, a single-precision accumulator only
# rounds what it stores.
ctypedef fused acc_t:
    c64_t
    c128_t

ctypedef fused acc_real_t:
    float
    double


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


cdef inline void _fold_scalar(acc_real_t* p, acc_real_t* x, double vr, double vi, double br, double bi, bint first,
                              const double* d, const double* z, Py_ssize_t nz) noexcept nogil:
    """One width-one block on the ``nz`` shifts of one row: ``p = (v + p b) d``, ``x += p z``, on interleaved
    (re, im) reals (the storage's precision; the arithmetic is double). Spelled out in real arithmetic: the compiler's complex product is several times slower."""
    cdef Py_ssize_t i
    cdef double pr, pi, tr, ti
    for i in range(nz):
        if first:
            tr = vr
            ti = vi
        else:
            pr = p[2 * i]
            pi = p[2 * i + 1]
            tr = vr + pr * br - pi * bi
            ti = vi + pr * bi + pi * br
        pr = tr * d[2 * i] - ti * d[2 * i + 1]
        pi = tr * d[2 * i + 1] + ti * d[2 * i]
        p[2 * i] = <acc_real_t>pr
        p[2 * i + 1] = <acc_real_t>pi
        x[2 * i] = <acc_real_t>(x[2 * i] + pr * z[2 * i] - pi * z[2 * i + 1])
        x[2 * i + 1] = <acc_real_t>(x[2 * i + 1] + pr * z[2 * i + 1] + pi * z[2 * i])


def shifted_update(const acc_t[:, :, ::1] P_prev not None, acc_t[:, :, ::1] P not None,
                   acc_t[:, :, ::1] X not None, const double complex[:, ::1] V not None,
                   const double complex[:, ::1] bH not None, const double complex[:, :, ::1] d_inv not None,
                   const double complex[:, :, ::1] zeta not None, bint first):
    """One block of :class:`~impurityModel.ed.gf_shift_recycling.ShiftedGalerkin` for every shift in one sweep.

    Per row ``r`` and shift ``i``: ``t = V[r] + P_prev[r, i] @ bH`` (``V[r]`` alone when ``first``),
    ``P[r, i] = t @ d_inv[i]`` and ``X[r, i] += P[r, i] @ zeta[i]``. ``V`` is read once for all shifts and the
    shifts of a row are contiguous, so the sweep touches each of ``P_prev``, ``P`` and ``X`` once. ``P`` may be
    ``P_prev`` (the update is row-local and reads the old row before writing the new one); ``X`` is updated in
    place. Shapes: ``V (n_v, w)``, ``bH (w_prev, w)``, ``P (n, nz, w)``, ``P_prev (n, nz, w_prev)``,
    ``X (n, nz, p)``, ``d_inv (nz, w, w)``, ``zeta (nz, w, p)``. ``V`` may hold fewer rows than the accumulator
    (``n_v <= n``): the rows it lacks are zero, so no padded copy of the block is ever made.

    Width one throughout (a single seed, the common case of a large unit) runs a scalar loop over the shifts of a
    row, which the compiler vectorises: about 4x the general path's throughput, and the sweep is then bound by
    memory bandwidth."""
    cdef Py_ssize_t n = P.shape[0]
    cdef Py_ssize_t nv = V.shape[0]
    cdef Py_ssize_t nz = P.shape[1]
    cdef Py_ssize_t w = P.shape[2]
    cdef Py_ssize_t wp = P_prev.shape[2]
    cdef Py_ssize_t p = X.shape[2]
    cdef Py_ssize_t r, i, c, m, k, q
    cdef double complex s, v0, b0
    if V.shape[1] != w or nv > n or X.shape[0] != n or X.shape[1] != nz or P_prev.shape[0] != n:
        raise ValueError("shifted_update: inconsistent block shapes")
    if P_prev.shape[1] != nz or d_inv.shape[0] != nz or d_inv.shape[1] != w or d_inv.shape[2] != w:
        raise ValueError("shifted_update: d_inv must have shape (nz, w, w) and P_prev (n, nz, w_prev)")
    if zeta.shape[0] != nz or zeta.shape[1] != w or zeta.shape[2] != p:
        raise ValueError("shifted_update: zeta must have shape (nz, w, p)")
    if not first and (bH.shape[0] != wp or bH.shape[1] != w):
        raise ValueError("shifted_update: bH must have shape (w_prev, w)")
    if n == 0 or nz == 0 or w == 0:
        return
    cdef const double complex* pv = &V[0, 0] if nv > 0 else NULL
    cdef const acc_t* pprev = &P_prev[0, 0, 0] if wp > 0 else NULL
    cdef acc_t* pp = &P[0, 0, 0]
    cdef acc_t* px = &X[0, 0, 0] if p > 0 else NULL
    cdef const double complex* pd = &d_inv[0, 0, 0]
    cdef const double complex* pzt = &zeta[0, 0, 0] if p > 0 else NULL
    cdef const double complex* pb = &bH[0, 0] if not first else NULL
    cdef const acc_t* row_prev
    cdef acc_t* row_p
    cdef acc_t* row_x
    cdef double complex* t
    if w == 1 and p == 1 and (first or wp == 1):
        b0 = 0 if first else pb[0]
        with nogil:
            for r in range(n):
                v0 = pv[r] if r < nv else 0
                if acc_t is c128_t:
                    _fold_scalar(<double*>(pp + r * nz), <double*>(px + r * nz), v0.real, v0.imag, b0.real,
                                 b0.imag, first, <const double*>pd, <const double*>pzt, nz)
                else:
                    _fold_scalar(<float*>(pp + r * nz), <float*>(px + r * nz), v0.real, v0.imag, b0.real,
                                 b0.imag, first, <const double*>pd, <const double*>pzt, nz)
        return
    t = <double complex*>malloc(w * sizeof(double complex))
    if t == NULL:
        raise MemoryError()
    try:
        with nogil:
            for r in range(n):
                for i in range(nz):
                    for c in range(w):
                        s = pv[r * w + c] if r < nv else 0
                        if not first:
                            row_prev = pprev + (r * nz + i) * wp
                            for k in range(wp):
                                s = s + <double complex>row_prev[k] * pb[k * w + c]
                        t[c] = s
                    row_p = pp + (r * nz + i) * w
                    for c in range(w):
                        s = 0
                        for m in range(w):
                            s = s + t[m] * pd[(i * w + m) * w + c]
                        row_p[c] = <acc_t>s
                    row_x = px + (r * nz + i) * p
                    for q in range(p):
                        s = 0
                        for c in range(w):
                            s = s + <double complex>row_p[c] * pzt[(i * w + c) * p + q]
                        row_x[q] = <acc_t>(<double complex>row_x[q] + s)
    finally:
        free(t)


def shifted_update_many(acc_t[:, :, ::1] P not None, acc_t[:, :, ::1] X not None, list V not None,
                        const double complex[:, :, ::1] bH not None,
                        const double complex[:, :, :, ::1] d_inv not None,
                        const double complex[:, :, :, ::1] zeta not None):
    """``k`` consecutive blocks of :func:`shifted_update` in **one** sweep over the accumulator, for blocks of one
    width ``w`` that are not the first of the recurrence.

    The sweep is bound by memory bandwidth (every rank reads and writes its whole accumulator), so folding ``k``
    blocks while a row is in cache costs one pass over ``P`` and ``X`` instead of ``k``. Per row and shift the
    arithmetic is :func:`shifted_update`'s, block after block. ``V`` is a list of ``k`` C-contiguous ``(n_j, w)``
    blocks (``n_j <= n``, zero on the rows a block lacks), ``bH (k, w, w)``, ``d_inv (k, nz, w, w)``,
    ``zeta (k, nz, w, p)``, ``P (n, nz, w)`` and ``X (n, nz, p)``, both updated in place."""
    cdef Py_ssize_t n = P.shape[0]
    cdef Py_ssize_t nz = P.shape[1]
    cdef Py_ssize_t w = P.shape[2]
    cdef Py_ssize_t p = X.shape[2]
    cdef Py_ssize_t nb = len(V)
    cdef Py_ssize_t r, i, j, c, m, q
    cdef double complex s, v0, b0
    cdef const double complex[:, ::1] vj
    if X.shape[0] != n or X.shape[1] != nz:
        raise ValueError("shifted_update_many: P and X must share their rows and shifts")
    if bH.shape[0] != nb or bH.shape[1] != w or bH.shape[2] != w:
        raise ValueError("shifted_update_many: bH must have shape (k, w, w)")
    if d_inv.shape[0] != nb or d_inv.shape[1] != nz or d_inv.shape[2] != w or d_inv.shape[3] != w:
        raise ValueError("shifted_update_many: d_inv must have shape (k, nz, w, w)")
    if zeta.shape[0] != nb or zeta.shape[1] != nz or zeta.shape[2] != w or zeta.shape[3] != p:
        raise ValueError("shifted_update_many: zeta must have shape (k, nz, w, p)")
    if nb == 0 or n == 0 or nz == 0 or w == 0 or p == 0:
        return
    cdef const double complex** pv = <const double complex**>malloc(nb * sizeof(void*))
    cdef Py_ssize_t* nv = <Py_ssize_t*>malloc(nb * sizeof(Py_ssize_t))
    cdef double complex* t = <double complex*>malloc(w * sizeof(double complex))
    cdef acc_t* pp = &P[0, 0, 0]
    cdef acc_t* px = &X[0, 0, 0]
    cdef const double complex* pb = &bH[0, 0, 0]
    cdef const double complex* pd = &d_inv[0, 0, 0, 0]
    cdef const double complex* pzt = &zeta[0, 0, 0, 0]
    cdef const double complex* dj
    cdef const double complex* zj
    cdef const double complex* bj
    cdef acc_t* row_p
    cdef acc_t* row_x
    if pv == NULL or nv == NULL or t == NULL:
        free(pv)
        free(nv)
        free(t)
        raise MemoryError()
    try:
        for j in range(nb):
            vj = V[j]
            if vj.shape[1] != w or vj.shape[0] > n:
                raise ValueError("shifted_update_many: every block must be (n_j <= n, w)")
            nv[j] = vj.shape[0]
            pv[j] = &vj[0, 0] if vj.shape[0] > 0 else NULL
        with nogil:
            if w == 1 and p == 1:
                for r in range(n):
                    row_p = pp + r * nz
                    row_x = px + r * nz
                    for j in range(nb):
                        v0 = pv[j][r] if r < nv[j] else 0
                        b0 = pb[j]
                        if acc_t is c128_t:
                            _fold_scalar(<double*>row_p, <double*>row_x, v0.real, v0.imag, b0.real, b0.imag, False,
                                         <const double*>(pd + j * nz), <const double*>(pzt + j * nz), nz)
                        else:
                            _fold_scalar(<float*>row_p, <float*>row_x, v0.real, v0.imag, b0.real, b0.imag, False,
                                         <const double*>(pd + j * nz), <const double*>(pzt + j * nz), nz)
            else:
                for r in range(n):
                    for i in range(nz):
                        row_p = pp + (r * nz + i) * w
                        row_x = px + (r * nz + i) * p
                        for j in range(nb):
                            bj = pb + j * w * w
                            dj = pd + (j * nz + i) * w * w
                            zj = pzt + (j * nz + i) * w * p
                            for c in range(w):
                                s = pv[j][r * w + c] if r < nv[j] else 0
                                for m in range(w):
                                    s = s + <double complex>row_p[m] * bj[m * w + c]
                                t[c] = s
                            for c in range(w):
                                s = 0
                                for m in range(w):
                                    s = s + t[m] * dj[m * w + c]
                                row_p[c] = <acc_t>s
                            for q in range(p):
                                s = 0
                                for c in range(w):
                                    s = s + <double complex>row_p[c] * zj[c * p + q]
                                row_x[q] = <acc_t>(<double complex>row_x[q] + s)
    finally:
        free(pv)
        free(nv)
        free(t)


def pt2_scores(const acc_t[:, :, ::1] X not None, long offset, const long[::1] slots not None,
               const double[::1] diag not None, const double complex[::1] zs not None,
               const double[::1] gnorm not None, double[::1] s not None, double[::1] mass not None):
    """The second-order growth scores (:mod:`impurityModel.ed.gf_growth_score`) of the frontier rows, in one pass
    over the accumulator and with no array the size of the frontier but the result.

    ``X`` is one row chunk ``(n, nz, p)`` of the accumulator, holding the K slots ``offset .. offset + n``. For row
    ``j`` (K slot ``slots[j]``, diagonal ``diag[j]``) in that range and shift ``i``:
    ``v = sum_q |X[slots[j] - offset, i, q]|^2 / max(|zs[i] - diag[j]|, 1e-12) / gnorm[i]``; ``s[j]`` is set to the
    maximum over the shifts and ``mass[i]`` is incremented by the sum over the rows. A slot outside the chunk is left
    alone: the caller zeroes ``s`` and ``mass`` and calls once per chunk, so a slot past the accumulator's rows
    (reached after the last fold) scores zero."""
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t nz = X.shape[1]
    cdef Py_ssize_t p = X.shape[2]
    cdef Py_ssize_t m = slots.shape[0]
    cdef Py_ssize_t j, i, q
    cdef long r
    cdef double num, best, v, d, dr, di, xr, xi
    cdef const acc_t* row
    cdef const acc_t* px
    if diag.shape[0] != m or s.shape[0] != m:
        raise ValueError("pt2_scores: slots, diag and s must have one entry per frontier row")
    if zs.shape[0] != nz or gnorm.shape[0] != nz or mass.shape[0] != nz:
        raise ValueError("pt2_scores: zs, gnorm and mass must have one entry per shift")
    if m == 0 or n == 0 or nz == 0 or p == 0:
        return
    px = &X[0, 0, 0]
    with nogil:
        for j in range(m):
            r = slots[j] - offset
            if 0 <= r < n:
                best = 0.0
                row = px + r * nz * p
                d = diag[j]
                for i in range(nz):
                    num = 0.0
                    for q in range(p):
                        xr = row[i * p + q].real
                        xi = row[i * p + q].imag
                        num = num + xr * xr + xi * xi
                    dr = zs[i].real - d
                    di = zs[i].imag
                    v = dr * dr + di * di
                    v = num / (gnorm[i] * (v ** 0.5 if v > 1e-24 else 1e-12))
                    mass[i] = mass[i] + v
                    if v > best:
                        best = v
                s[j] = best
