"""
Transcription between the distributed determinant basis and dense/sparse
linear-algebra objects: wavefunction vectors, operator matrices, and
single-particle density matrices.

All functions take the basis as their first argument. The ones documented as
reducing over ranks (`build_vector` without root, `build_dense_matrix`,
`build_density_matrices`) contain MPI collectives and must be called by all
ranks of ``basis.comm``.
"""

from typing import Any, Optional, Union

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, Row, applyOp, inner_multi
from impurityModel.ed.work_queue import queue_progress


def build_vector(
    basis, psis: list[ManyBodyState], root: Optional[int] = None, slaterWeightMin: float = 0
) -> np.ndarray:
    """Build a dense matrix representation of wavefunctions in the basis.

    Parameters
    ----------
    psis : list of ManyBodyState
        The wavefunctions to represent.
    root : int, optional
        MPI rank to reduce the vector to. If None, it is reduced to all ranks.
    slaterWeightMin : float, default 0
        Minimum amplitude threshold below which coefficients are ignored.

    Returns
    -------
    np.ndarray
        The 2D dense matrix representation of the wavefunctions.
    """
    v = np.zeros((len(psis), basis.size), dtype=complex, order="C")
    # psis = basis.redistribute_psis(psis)
    # row_states_in_basis: list[bytes] = []
    # row_dict = {state: basis._index_dict[state] for state in basis.local_basis}
    # col_dict = dict(zip(basis.local_basis, range(basis.local_indices.start, basis.local_indices.stop)))
    _size = basis.size
    for row, psi in enumerate(psis):
        for state, val in psi.items():
            idx = basis._local_index(state)
            # psi may be a ManyBodyState (val is a width-1 Row) or a plain dict of
            # scalars (a documented duck-typed input): only a Row needs unwrapping.
            amp = val[0] if isinstance(val, Row) else val
            if idx == _size or abs(amp) < slaterWeightMin:
                continue
            v[row, idx] = amp

    if basis.is_distributed and root is None:
        basis.comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
    elif basis.is_distributed and root == 0:
        basis.comm.Reduce(MPI.IN_PLACE if basis.comm.rank == root else v, v, op=MPI.SUM, root=root)
    elif basis.is_distributed:
        # Every caller passes root=0 or None today, so this branch is unreached -- but an
        # IN_PLACE Reduce at a non-zero root segfaults on MPICH 4.0 above the 2048-byte
        # eager threshold (measured; see BlockLanczosArray.pyx's matvec), and `v` here is
        # (n_psis x n_basis), so a future caller passing root=1 would hit it at any real
        # size. Pay for a temporary on that path rather than leave the trap armed.
        out = np.zeros_like(v) if basis.comm.rank == root else None
        basis.comm.Reduce(v, out, op=MPI.SUM, root=root)
        if basis.comm.rank == root:
            v[...] = out
    return v


def build_distributed_vector(basis, psis: list[ManyBodyState], dtype: Any = complex) -> np.ndarray:
    """Build the MPI-local portion of a wavefunction vector.

    Parameters
    ----------
    psis : list of ManyBodyState
        The wavefunctions to represent.
    dtype : Any, default complex
        The data type of the returned array.

    Returns
    -------
    np.ndarray
        The 2D array containing the local amplitudes.
    """
    psis = basis.redistribute_psis(*psis)
    v = np.empty((len(psis), len(basis.local_basis)), dtype=dtype, order="C")
    # Nested loops, not `itertools.product`: product materializes each argument to a tuple
    # when it is constructed, so `enumerate(basis.local_basis)` realized the whole local
    # basis as `SlaterDeterminant` objects -- plus one index tuple each -- before the first
    # amplitude was read, and held it there for the whole build. Measured on a
    # 400k-determinant basis: 200.5 B/det at the peak for an output that is 16 B/det.
    #
    # The basis is the OUTER loop, and that is the whole point. `local_basis` streams now
    # (see `_LocalBasisView.__iter__`), so it is free to walk but not free to walk twice:
    # nesting the other way re-runs the walk once per column, allocating `len(psis) *
    # n_local` determinant wrappers instead of `n_local`. Measured on a 60k basis:
    #
    #     columns     1        4       16
    #     psis-outer  0.017 s  0.067   0.268
    #     basis-outer 0.025 s  0.061   0.171
    #
    # so this is a deliberate trade, not a free win: one column gets ~1.5x slower (an extra
    # inner-loop frame per determinant for a single psi) and everything above two gets
    # faster, up to 1.57x at 16. Production calls it with one column per Green's-function
    # seed (`gf_primitives.py:159`), typically 10-30, which is the case worth having.
    #
    # Every cell is written exactly once from a pure function of `(psi, state)`, so only
    # the write ORDER changes against the product form, not any value.
    for col, state in enumerate(basis.local_basis):
        for row, psi in enumerate(psis):
            row_amp = psi.get(state)
            v[row, col] = 0 if row_amp is None else row_amp[0]
    return v


def build_state(basis, vs: Union[list[np.ndarray], np.ndarray], slaterWeightMin: float = 0) -> ManyBodyState:
    """Convert dense vectors to a single ``ManyBodyState`` block of ``vs.shape[0]`` columns.

    Parameters
    ----------
    vs : list of np.ndarray or np.ndarray
        The dense vector representations, one row per column of the returned block.
    slaterWeightMin : float, default 0
        Minimum amplitude threshold to keep.

    Returns
    -------
    ManyBodyState
        One block of width ``vs.shape[0]`` over the union support: a determinant is a row
        when *any* column has ``|amplitude| > slaterWeightMin`` there, and the entries that
        fall below the cutoff are stored as exact zeros.

    Notes
    -----
    **Returns one wide block, not a list of width-1 blocks.** The list form cost one key
    array per column -- at the SrMnO3 production geometry (128 ranks, ``p`` = 314, ~31k
    determinants per rank) that is 314 copies of one key set, measured at 447 MiB per rank
    against round 9 (``doc/plans/memory_footprint_audit.md``, Phase 6). The block holds the
    support once.

    It is also what every consumer already wanted: ``CIPSISolver.truncate`` and
    ``_apply_block_and_redistribute`` both opened with ``ManyBodyState.from_states(...)`` on
    this function's output, and ``block_lanczos_cy`` has taken a block seed since the
    block-native restart work. Those conversions are now gone rather than moved.

    Exact by construction (tier 1), because ``from_states`` built precisely this union block
    from the list form: same rows, same stored coefficients, same order. ``e0`` is
    bit-identical at fixed rank count.

    **``len()`` on the result is its ROW count, and iterating it yields determinant keys** --
    neither is the column count. Use ``.width`` for the number of columns (it is
    rank-invariant, including on a rank that owns no rows) and ``.to_states()`` or
    ``.select(...)`` to reach individual columns. This is the same footgun
    ``_lanczos_step.pxi``'s seed handling calls out.
    """
    if isinstance(vs, np.matrix):
        vs = vs.A
    if isinstance(vs, np.ndarray) and len(vs.shape) == 1:
        vs = vs.reshape((1, vs.shape[0]))
    if isinstance(vs, list):
        vs = np.array(vs)
    if vs.shape[1] == basis.size:
        local = np.asarray(vs[:, basis.local_indices])
    elif vs.shape[1] == len(basis.local_basis):
        local = np.asarray(vs)
    else:
        raise RuntimeError(
            f"The dimensions of the input dense vector does not match a distributed, or full vector.\n"
            f"{vs.shape} != ({vs.shape[0]}, {basis.size}) || ({vs.shape[0]}, {len(basis.local_basis)})"
        )
    width = local.shape[0]
    keep = np.abs(local) > slaterWeightMin
    rows = np.flatnonzero(keep.any(axis=0)) if width else np.zeros(0, dtype=np.intp)
    # Pre-allocated (rows, width) so the only (width, rows) temporary is the fancy-index
    # gather; `.T` and the mask are views, and `from_keys_and_amps` finds it already
    # contiguous and complex, so it does not copy again.
    amps = np.zeros((rows.size, width), dtype=complex)
    if rows.size and width:
        np.copyto(amps, local[:, rows].T, where=keep[:, rows].T)
    support = ManyBodyState.from_keys([basis.local_basis[i] for i in rows])
    return ManyBodyState.from_keys_and_amps(support, amps)


def iter_local_operator_images(basis, op, slaterWeightMin, indices=None):
    """Yield ``op`` applied to each (MPI local) basis state, in basis order.

    The streaming counterpart of :func:`build_local_operator_list`. A caller that consumes
    one image at a time should use this: the list form keeps every image alive at once, so
    the whole ``H|D>`` image of the local basis -- ``O(N_local * fan-out)`` rows -- is
    resident before the first one is read, and stays resident until the consumer finishes.

    ``indices`` (local positions) restricts the walk to those states, in the order given; a
    caller sampling the image size passes a stride, because ``local_basis`` is sorted and its
    first states share their leading-orbital occupation.
    """
    unit_state = ManyBodyState()
    local_basis = basis.local_basis
    for state in local_basis if indices is None else (local_basis[int(i)] for i in indices):
        unit_state[state] = 1.0
        yield applyOp(op, unit_state, cutoff=slaterWeightMin)
        unit_state.erase(state)


def build_local_operator_list(basis, op, slaterWeightMin):
    """
    Apply the operator to all (MPI local) basis states, in order.
    Return the results in a list.

    Materializes every image at once; prefer :func:`iter_local_operator_images` unless the
    whole list is genuinely needed.
    """
    return list(iter_local_operator_images(basis, op, slaterWeightMin))


def build_dense_matrix(basis, op, distribute=True):
    """
    Get the operator as a dense matrix in the current basis.
    by default the dense matrix is distributed to all ranks.
    """
    h_local = build_sparse_matrix(basis, op)
    if basis.is_distributed:
        h = np.empty(h_local.shape, dtype=h_local.dtype)
        basis.comm.Allreduce(h_local.todense(), h, op=MPI.SUM)
    else:
        h = h_local.todense(order="C")
    return h


#: Matrix elements per collective index lookup in the distributed :func:`build_sparse_matrix`.
#: Bounds the build's transient (one key object, column and value per element, ~104-160 B) to a
#: batch instead of the whole local image of ``op``; the lookup's own per-call cost is a few
#: collectives. 2**19 (80 MiB budgeted) rather than 2**21 (320 MiB): the batch is part of the
#: frozen-basis matrix's memory need, and it decided several of the CrI3 units the 2026-10 runs
#: declined; an SrMnO3 auto-cap unit (~87M raw elements per rank) then does ~166 lookups, not ~41.
_SPARSE_BUILD_BATCH = 1 << 19


def _concat_consuming(chunks, dtype):
    """``np.concatenate(chunks)`` that releases each chunk as soon as it is copied.

    ``np.concatenate`` holds the list and the result at once (2x the data). Copying chunk by chunk
    and dropping each reference as it goes keeps the peak at the result plus the largest chunk.
    ``chunks`` is emptied.
    """
    out = np.empty(sum(len(c) for c in chunks), dtype=dtype)
    pos = 0
    for k, chunk in enumerate(chunks):
        out[pos : pos + len(chunk)] = chunk
        pos += len(chunk)
        chunks[k] = None
    chunks.clear()
    return out


#: Kets between two answers to the GF unit queue during the build (work_queue.queue_progress).
#: A frozen-CSR build is one stretch of 4-12 minutes per capped SrMnO3 unit with no Lanczos
#: block in it, and the per-block poke is the only other time rank 0 answers the other colors'
#: fetches. On every other rank the poke is one list check.
_QUEUE_POKE_KETS = 4096


def _column_batches(basis, op: ManyBodyOperator):
    """The stored entries of ``op`` in ``basis``'s local columns, in batches of whole columns.

    Yields ``(col_begin, counts, rows, values)``: the batch covers local columns ``col_begin`` to
    ``col_begin + len(counts)``, ``counts[c]`` entries each, and ``rows`` (int64 global row
    indices) and ``values`` hold them column after column, each column's rows in the order the
    operator produced them (unsorted). A batch holds about :data:`_SPARSE_BUILD_BATCH` matrix
    elements before the out-of-basis rows are dropped. Collective on a distributed basis: every
    rank runs the same number of batches, an exhausted rank with empty ones.
    """
    _offset = basis.offset
    _size = basis.size
    # Images are streamed, not listed: the list form holds the whole `H|D>` image of the
    # local basis (`O(N_local * fan-out)` rows) resident for the entire build. Amplitudes are
    # read through the buffer protocol rather than `.items()`, which allocated a
    # `SlaterDeterminant` and a `Row` object per matrix element -- the same cost the CIPSI
    # selection round removed and measured at 33-43% of a round (`cipsi_solver.py`'s
    # `_candidate_overlaps_and_energies`). `keys()` is in row order, so the buffer's rows and
    # the key list line up.
    # `ket` walks `local_basis` in order and `local_indices` is
    # `range(offset, offset + len(local_basis))`, so the column index is the loop position --
    # the old `_index_dict[ket]` was looking up an answer the enumerate already had.
    if not basis.is_distributed:
        # Filter as we go. Only a minority of the image survives -- every bra outside the
        # basis is dropped -- so accumulating the whole image first and masking at the end
        # (which the distributed branch below is forced into by its single collective) costs
        # several times the final matrix. Measured: doing that here took the serial peak from
        # 291 to 520 MiB on a 393k-determinant basis, because ~76% of image rows are discarded.
        keys_block = basis._keys
        n_local_keys = len(keys_block)
        col_begin = 0
        counts, rows, values = [], [], []
        pending = 0
        for i, ket_state in enumerate(iter_local_operator_images(basis, op, 0)):
            if i % _QUEUE_POKE_KETS == 0:
                queue_progress()  # see _QUEUE_POKE_KETS
            n_keep = 0
            ks = ket_state.keys()
            if ks:
                local_rows = keys_block.find_rows(ks)
                keep = local_rows != n_local_keys
                n_keep = int(keep.sum())
                if n_keep:
                    rows.append(local_rows[keep].astype(np.int64, copy=False) + _offset)
                    values.append(np.asarray(ket_state)[:, 0][keep])
            counts.append(n_keep)
            pending += len(ks)
            if pending >= _SPARSE_BUILD_BATCH:
                yield col_begin, np.array(counts, dtype=np.int64), _concat_consuming(
                    rows, np.int64
                ), _concat_consuming(values, complex)
                col_begin = i + 1
                counts, pending = [], 0
        if counts:
            yield col_begin, np.array(counts, dtype=np.int64), _concat_consuming(rows, np.int64), _concat_consuming(
                values, complex
            )
        return
    # The routed lookup is collective, so the bras cannot be resolved image by image. They
    # are accumulated and resolved in batches of about `_SPARSE_BUILD_BATCH` matrix elements:
    # resolving the whole local image at once held one key object per element of it -- the
    # full `O(N_local * fan-out)` image, most of which is then masked away -- which at a
    # multi-million determinant basis is what limits the build, not the matrix. Every rank
    # runs the same number of lookups: the loop continues while ANY rank has images left,
    # and a rank that has run out enters each further lookup with an empty batch.
    images = iter_local_operator_images(basis, op, 0)
    position = 0
    exhausted = False
    while True:
        col_begin = position
        bras = []
        batch_cols = []
        batch_vals = []
        while not exhausted and len(bras) < _SPARSE_BUILD_BATCH:
            ket_state = next(images, None)
            if ket_state is None:
                exhausted = True
                break
            column = position
            position += 1
            if column % _QUEUE_POKE_KETS == 0:
                queue_progress()  # see _QUEUE_POKE_KETS
            ks = ket_state.keys()
            if not ks:
                continue
            bras.extend(ks)
            batch_cols.append(np.full(len(ks), column - col_begin, dtype=np.int64))
            batch_vals.append(np.asarray(ket_state)[:, 0].copy())
        columns = np.concatenate(batch_cols) if batch_cols else np.empty(0, dtype=np.int64)
        values = np.concatenate(batch_vals) if batch_vals else np.empty(0, dtype=complex)
        del batch_cols, batch_vals
        global_rows = np.fromiter(basis._index_sequence(bras), dtype=np.int64, count=len(bras))
        del bras
        keep = global_rows != _size
        counts = np.bincount(columns[keep], minlength=position - col_begin).astype(np.int64, copy=False)
        rows = global_rows[keep]
        del global_rows, columns
        values = values[keep]
        del keep
        yield col_begin, counts, rows, values
        del rows, values
        if not basis.comm.allreduce(not exhausted, op=MPI.LOR):
            break


class LocalColumnOperator:
    """This rank's columns of an operator, ``(N, N_local)``, as the column blocks the build made.

    The frozen-basis Green's function runs the array Lanczos kernel on ``H[:, local columns]``.
    As a scipy CSR -- the form the kernel takes for every other caller -- that matrix costs, at
    its peak, the concatenation of the build's per-batch pieces (~40 B per stored element), the
    CSC-to-CSR conversion on top (~50 B), and a row pointer of ``N + 1`` int64 entries on every
    rank (149 MB per rank at N = 18.6M, more than the local matrix itself on SrMnO3). This keeps
    the pieces as they come: ``segments`` is a list of ``(col_begin, indptr, rows, data)``, local
    columns ``col_begin`` to ``col_begin + len(indptr) - 1`` in compressed-column form, about 20 B
    per element (16 for the value, 4 for an int32 row index) plus 8 B per local column.

    The kernel multiplies by scattering each column into the rows it reaches (``Y[row] += v X[col]``),
    walking the columns in ascending order, so every output row is summed in the same order as a
    CSR row loop. It first calls :meth:`bind_slots`, which replaces each global row index, in
    place, by the row it lands in in the matvec's exchange buffers -- no owner search per element.

    Rows are int32 when ``N < 2**31``, decided from the replicated ``N``, so every rank agrees.
    """

    def __init__(self, shape, segments):
        self.shape = (int(shape[0]), int(shape[1]))
        self.segments = segments
        self._binding = None
        self._flags = None

    @property
    def nnz(self):
        return int(sum(len(rows) for _, _, rows, _ in self.segments))

    @property
    def nbytes(self):
        return int(sum(indptr.nbytes + rows.nbytes + data.nbytes for _, indptr, rows, data in self.segments))

    def owner_flags(self, counts, offsets):
        """Bool array over ranks: does any stored row belong to rank ``d`` (rows ``offsets[d]`` to
        ``offsets[d] + counts[d]``)? Rank-local; used to build the matvec's exchange graph.

        Cached per layout, so it can still be answered after :meth:`bind_slots` has replaced the
        global rows (the kernel is re-run on the same operator when it doubles its block budget)."""
        counts = np.asarray(counts, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        key = (tuple(counts.tolist()), tuple(offsets.tolist()))
        if self._flags is not None and self._flags[0] == key:
            return self._flags[1].copy()
        if self._binding is not None:
            raise RuntimeError("owner_flags needs global row indices; this operator is already bound to slots")
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
        ``mpi_comm.MatvecExchangePlan``. Binding again with the same layout is a no-op (the kernel
        is re-run on the same operator when it doubles its block budget); a different layout raises.
        """
        offsets = np.asarray(offsets, dtype=np.int64)
        key = (int(rank), tuple(offsets.tolist()), tuple(int(d) for d in destinations), int(total_send_rows))
        if self._binding is not None:
            if self._binding != key:
                raise RuntimeError("LocalColumnOperator is already bound to a different exchange layout")
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

    def tocsc(self):
        """The same matrix as a scipy CSC (``build_sparse_matrix(..., local_columns=True)``, unsorted
        columns). Only before :meth:`bind_slots`."""
        if self._binding is not None:
            raise RuntimeError("tocsc needs global row indices; this operator is already bound to slots")
        n_cols = self.shape[1]
        counts = np.zeros(n_cols, dtype=np.int64)
        for col_begin, indptr, _, _ in self.segments:
            counts[col_begin : col_begin + len(indptr) - 1] = np.diff(indptr)
        indptr = np.zeros(n_cols + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        rows = np.concatenate([r for _, _, r, _ in self.segments]) if self.segments else np.empty(0, np.int64)
        data = np.concatenate([d for _, _, _, d in self.segments]) if self.segments else np.empty(0, complex)
        return sp.sparse.csc_array((data, rows.astype(np.int64, copy=False), indptr), shape=self.shape)


def build_local_column_operator(basis, op: ManyBodyOperator):
    """``op``'s columns for this rank's determinants as a :class:`LocalColumnOperator`.

    The same entries as ``build_sparse_matrix(basis, op, local_columns=True)``, kept in the build's
    per-batch column blocks: the peak is the stored matrix (~20 B per element) plus one batch, with
    no concatenation, no CSR conversion and no global-length row pointer. Collective on a
    distributed basis, like :func:`build_sparse_matrix`.
    """
    if isinstance(op, dict):
        op = ManyBodyOperator(op)
    n = len(basis)
    row_dtype = np.int32 if n < 2**31 else np.int64  # replicated n: the same choice on every rank
    segments = []
    for col_begin, counts, rows, values in _column_batches(basis, op):
        if not len(counts):
            continue  # an exhausted rank's empty batch
        indptr = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        segments.append((col_begin, indptr, rows.astype(row_dtype), values))
        del rows
    return LocalColumnOperator((n, len(basis.local_basis)), segments)


def build_sparse_matrix(basis, op: ManyBodyOperator, local_columns=False):
    """
    Get the operator as a sparse matrix in the current basis.
    The sparse matrix is distributed over all ranks.

    Returns a canonical CSC array whose stored columns are the ones this rank owns. By default it
    has shape ``(N, N)`` with every other column empty; ``local_columns=True`` returns just the
    ``(N, N_local)`` slice, which is what ``H[:, basis.local_indices]`` would give, without ever
    forming the ``(N, N)`` column pointer or the slice's second copy.

    The CSC is assembled directly: images arrive one ket at a time, so the entries are already
    grouped by column and only the per-column counts are needed -- no COO triplet, and no
    coordinate-to-compressed conversion holding both. Peak memory is the stored entries (index +
    value) plus the chunks not yet consumed, instead of ~4x that.
    """
    if isinstance(op, dict):
        op = ManyBodyOperator(op)

    _offset = basis.offset
    n_local = len(basis.local_basis)

    row_chunks = []
    val_chunks = []
    col_counts = np.zeros(n_local, dtype=np.int64)  # stored entries per local column
    for col_begin, batch_counts, rows, values in _column_batches(basis, op):
        row_chunks.append(rows)
        val_chunks.append(values)
        col_counts[col_begin : col_begin + len(batch_counts)] += batch_counts

    n = len(basis)
    nnz = int(col_counts.sum())
    indices = _concat_consuming(row_chunks, np.int64)
    data = _concat_consuming(val_chunks, complex)
    if local_columns:
        indptr = np.zeros(n_local + 1, dtype=np.int64)
        np.cumsum(col_counts, out=indptr[1:])
        shape = (n, n_local)
    else:
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(col_counts, out=indptr[_offset + 1 : _offset + n_local + 1])
        indptr[_offset + n_local + 1 :] = nnz
        shape = (n, n)
    res = sp.sparse.csc_array((data, indices, indptr), shape=shape)
    # The images were ordered by the operator, not by row: sort each column in place and mark the
    # result canonical (no duplicates -- a ManyBodyState holds each determinant once per image).
    res.sort_indices()
    return res


def build_density_matrices(basis, psis, orbital_indices_left=None, orbital_indices_right=None):
    r"""Compute single-particle density matrices for a list of many-body states.

    ``rho[n, i, j] = <psi_n| c_{orb_j}^dagger c_{orb_i} |psi_n>``

    For the square case (``orbital_indices_left == orbital_indices_right``) the
    identity ``rho[i, j] = <phi_j | phi_i>`` (where ``|phi_k> = c_{orb_k}|psi>``)
    is exploited, cutting operator applications from O(n^2) to O(n) and
    halving inner products via Hermitian symmetry.

    For the general rectangular case we also exploit this decomposition
    ``rho[i, j] = <chi_j | phi_i>`` (where ``|chi_k> = c_{orb_k}|psi>`` and
    ``|phi_k> = c_{orb_k}|psi>``), reducing operator applications to O(n).

    ``psis`` may also be a ``ManyBodyState`` (Phase 6a of the state-unification
    refactor); it is unpacked to a list once on entry and the body below is otherwise
    unchanged. Rewriting the per-state loop as one block-apply per orbital was
    considered and rejected, re-evaluated after ``ManyBodyState.select``/``column``
    got a direct O(rows) gather (no longer routed through a selection-matrix matvec) and
    still rejected: reading each orbital's per-state value back out no longer needs a
    full ``width x width`` Gram matrix (cheap columns make a per-state diagonal
    extraction possible), but that only matches this loop's existing per-state
    inner-product cost -- it doesn't beat it. The apply side is the one place
    ``apply_block`` could still win (near-flat cost in block width vs one term/sign/
    restriction/hash pass per state), and that gain is genuinely small here: every
    operator applied is a *trivial* single-term annihilator, so there is little shared
    per-determinant work left to amortize across states in the first place. See
    doc/plans/manybodystate_block_unification.md, Phase 6a, item 3b.
    """
    if isinstance(psis, ManyBodyState):
        psis = psis.to_states()
    if orbital_indices_left is None:
        orbital_indices_left = list(range(basis.num_spin_orbitals))
    if orbital_indices_right is None:
        orbital_indices_right = list(range(basis.num_spin_orbitals))
    n_left, n_right = len(orbital_indices_left), len(orbital_indices_right)
    rhos = np.zeros((len(psis), n_left, n_right), dtype=complex)

    square = orbital_indices_left == orbital_indices_right

    # Built once, not once per state: each annihilator is a pure function of orb, not
    # of psi_n, so constructing it inside the loop below paid len(psis)x redundant
    # ManyBodyOperator construction for no reason.
    left_ops = [ManyBodyOperator({((orb, "a"),): 1.0}) for orb in orbital_indices_left]
    right_ops = left_ops if square else [ManyBodyOperator({((orb, "a"),): 1.0}) for orb in orbital_indices_right]

    for n, psi_n in enumerate(psis):
        phi = [op(psi_n, 0) for op in left_ops]
        chi = phi if square else [op(psi_n, 0) for op in right_ops]

        if basis.is_distributed:
            phi = basis.redistribute_psis(*phi)
            chi = phi if square else basis.redistribute_psis(*chi)

        rhos[n] = inner_multi(chi, phi).T

    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, rhos, op=MPI.SUM)

    return rhos
