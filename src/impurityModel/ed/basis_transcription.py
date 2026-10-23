"""
Transcription between the distributed determinant basis and dense/sparse
linear-algebra objects: wavefunction vectors and operator matrices (single-particle
density matrices live in :mod:`impurityModel.ed.density_matrices`).

All functions take the basis as their first argument. The ones documented as
reducing over ranks (`build_vector` without root, `build_dense_matrix`) contain MPI collectives and
must be called by all ranks of ``basis.comm``.
"""

from typing import Any, Optional, Union

import numpy as np
from mpi4py import MPI

from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.ManyBodyUtils import KeyFilter, ManyBodyOperator, ManyBodyState, Row, applyOp
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
    this function's output, and ``block_lanczos`` has taken a block seed since the
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


def comm_support_filter(states, comm=None):
    """A :class:`KeyFilter` of the union of ``states``' supports over every rank of ``comm`` (``None``:
    only this rank; collective otherwise): the filled filters are OR-reduced, so each rank can test
    determinants that other ranks own. Sized from the sum of the local union sizes, the same on every
    rank: exact when the supports are owner-disjoint (after ``redistribute_psis``), an upper bound when
    they overlap across ranks (un-redistributed applies), which only lowers the false-positive rate."""
    union = ManyBodyState()
    for state in states:
        union = union.key_union(state)
    n_keys = len(union)
    if comm is not None:
        n_keys = comm.allreduce(n_keys, op=MPI.SUM)
    keep = KeyFilter(n_keys)
    keep.add(union)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, keep.words(), op=MPI.BOR)
    return keep


def support_filter(states, basis):
    """:func:`comm_support_filter` over ``basis``'s communicator (collective on a distributed basis)."""
    return comm_support_filter(states, basis.comm if basis.is_distributed else None)


def basis_filter(basis):
    """A :class:`KeyFilter` of every determinant of ``basis``, on every rank (collective on a
    distributed basis). It answers "is this determinant in the basis?" with no false negatives, so
    an operator applied with ``keep=`` generates only rows the exact lookup can still accept.
    Replicated: about 1-2 B per global determinant on each rank."""
    keep = KeyFilter(basis.size)
    keep.add(basis._keys)
    if basis.is_distributed:
        basis.comm.Allreduce(MPI.IN_PLACE, keep.words(), op=MPI.BOR)
    return keep


def iter_local_operator_images(basis, op, slaterWeightMin, indices=None, keep=None):
    """Yield ``op`` applied to each (MPI local) basis state, in basis order.

    The streaming counterpart of :func:`build_local_operator_list`. A caller that consumes
    one image at a time should use this: the list form keeps every image alive at once, so
    the whole ``H|D>`` image of the local basis -- ``O(N_local * fan-out)`` rows -- is
    resident before the first one is read, and stays resident until the consumer finishes.

    ``indices`` (local positions) restricts the walk to those states, in the order given; a
    caller sampling the image size passes a stride, because ``local_basis`` is sorted and its
    first states share their leading-orbital occupation.

    ``keep`` (a :class:`~impurityModel.ed.ManyBodyUtils.KeyFilter`, e.g. :func:`basis_filter`) drops
    every image row it rules out as the row is generated, so a caller that reads the image only on a
    known set of determinants never sees the rest. Rows that are kept are bit-identical to the
    unfiltered image's; a caller that reads the whole image (a size estimate) must not pass it.
    """
    unit_state = ManyBodyState()
    local_basis = basis.local_basis
    for state in local_basis if indices is None else (local_basis[int(i)] for i in indices):
        unit_state[state] = 1.0
        yield (
            applyOp(op, unit_state, cutoff=slaterWeightMin)
            if keep is None
            else op.apply_block(unit_state, slaterWeightMin, keep)
        )
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
    h_local = build_sparse_matrix(basis, op).toarray()
    if basis.is_distributed:
        h = np.empty(h_local.shape, dtype=h_local.dtype)
        basis.comm.Allreduce(h_local, h, op=MPI.SUM)
        return h
    return h_local


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


def _column_batches(basis, op: ManyBodyOperator, filtered: bool = True):
    """The stored entries of ``op`` in ``basis``'s local columns, in batches of whole columns.

    Yields ``(col_begin, counts, rows, values)``: the batch covers local columns ``col_begin`` to
    ``col_begin + len(counts)``, ``counts[c]`` entries each, and ``rows`` (int64 global row
    indices) and ``values`` hold them column after column, each column's rows in the order the
    operator produced them (unsorted). A batch holds about :data:`_SPARSE_BUILD_BATCH` matrix
    elements as generated through :func:`basis_filter` (most out-of-basis rows are never generated;
    the exact lookup then drops the false positives). Collective on a distributed basis: every
    rank runs the same number of batches, an exhausted rank with empty ones.

    Only the rows inside the basis are kept, so the operator is applied through
    :func:`basis_filter` (``filtered``): an image row the filter rules out is never generated, and
    on a distributed basis never packed and routed through the collective lookup. The exact
    ``find_rows`` / ``_index_sequence`` check still runs, so a false positive is dropped there.
    The filter is built here, on every rank, before the first batch.
    """
    bloom = basis_filter(basis) if filtered else None
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
        for i, ket_state in enumerate(iter_local_operator_images(basis, op, 0, keep=bloom)):
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
                yield col_begin, np.array(counts, dtype=np.int64), _concat_consuming(rows, np.int64), _concat_consuming(
                    values, complex
                )
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
    images = iter_local_operator_images(basis, op, 0, keep=bloom)
    position = 0
    exhausted = False
    while True:
        col_begin = position
        bras: list = []
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


def build_sparse_matrix(basis, op: ManyBodyOperator):
    """``op`` as this rank's columns of the sparse operator on ``basis``: a
    :class:`~impurityModel.ed.DistributedSparseMatrix.DistributedSparseMatrix` of shape
    ``(N, N_local)``, kept in the build's per-batch column blocks.

    Images arrive one ket at a time, so the entries are already grouped by column and only the
    per-column counts are needed: the peak is the stored matrix (~20 B per element) plus one
    batch -- no COO triplet, no concatenation, no CSR conversion and no global-length row pointer.
    Collective on a distributed basis.
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
    comm = basis.comm if basis.is_distributed else None
    return DistributedSparseMatrix((n, len(basis.local_basis)), segments, comm, basis.offset)


class MatrixSnapshot:
    """A matrix built on a basis, kept so that :func:`extend_sparse_matrix` can grow it.

    ``key`` names what the matrix is *of* (the caller's operator and window); ``keys`` is a copy
    of the rank-local determinant store the matrix was built on, and ``offsets`` every rank's
    global offset then (replicated). Collective to construct on a distributed basis.
    """

    def __init__(self, basis, mat, key):
        self.mat = mat
        self.key = key
        self.keys = basis._keys.copy()
        counts = basis.comm.allgather(len(basis._keys)) if basis.is_distributed else [len(basis._keys)]
        self.offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)


def _concat_or_empty(chunks, dtype):
    return np.concatenate(chunks).astype(dtype, copy=False) if chunks else np.empty(0, dtype=dtype)


def _columns_of(basis, op, local_cols):
    """The stored entries of ``op`` in the local columns ``local_cols``, rows inside ``basis``:
    ``(cols, rows, values)`` with local columns and global rows. Collective on a distributed basis
    (the routed row lookup runs in rank-symmetric batches, as in :func:`_column_batches`)."""
    bloom = basis_filter(basis)
    images = zip(local_cols, iter_local_operator_images(basis, op, 0, indices=local_cols, keep=bloom))
    cols_out, rows_out, vals_out = [], [], []
    if not basis.is_distributed:
        keys_block = basis._keys
        n_local = len(keys_block)
        for col, ket in images:
            ks = ket.keys()
            if not ks:
                continue
            rows = keys_block.find_rows(ks)
            keep = rows != n_local
            rows_out.append(rows[keep])
            vals_out.append(np.asarray(ket)[:, 0][keep])
            cols_out.append(np.full(int(keep.sum()), col, dtype=np.int64))
    else:
        exhausted = False
        while True:
            bras, batch_cols, batch_vals = [], [], []
            while not exhausted and len(bras) < _SPARSE_BUILD_BATCH:
                item = next(images, None)
                if item is None:
                    exhausted = True
                    break
                col, ket = item
                ks = ket.keys()
                if not ks:
                    continue
                bras.extend(ks)
                batch_cols.append(np.full(len(ks), col, dtype=np.int64))
                batch_vals.append(np.asarray(ket)[:, 0].copy())
            rows = np.fromiter(basis._index_sequence(bras), dtype=np.int64, count=len(bras))
            if bras:
                keep = rows != basis.size
                rows_out.append(rows[keep])
                cols_out.append(np.concatenate(batch_cols)[keep])
                vals_out.append(np.concatenate(batch_vals)[keep])
            if not basis.comm.allreduce(not exhausted, op=MPI.LOR):
                break
    return (
        _concat_or_empty(cols_out, np.int64),
        _concat_or_empty(rows_out, np.int64),
        _concat_or_empty(vals_out, complex),
    )


def _segment_of(cols, rows, values, row_dtype):
    """One compressed-column segment over the span of the local columns ``cols`` (any order)."""
    if not len(cols):
        return None
    order = np.lexsort((rows, cols))
    cols, rows, values = cols[order], rows[order], values[order]
    first = int(cols[0])
    counts = np.bincount(cols - first, minlength=int(cols[-1]) - first + 1)
    indptr = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return (first, indptr, rows.astype(row_dtype), values)


def extend_sparse_matrix(snapshot, basis, op):
    """``op`` on ``basis``, grown from ``snapshot`` -- a matrix of the same Hermitian ``op`` on a
    subset of ``basis`` -- instead of rebuilt; ``None`` when ``basis`` does not contain every
    determinant the snapshot was built on (decided collectively, so every rank returns ``None``
    together).

    A CIPSI expansion rebuilt its whole matrix every cycle by applying ``op`` to every determinant,
    so the build cost summed the basis size over the cycles: on a NiO 10-bath sector class (18 cycles
    to 6.5k determinants) it was 23 of 34 s. Here ``op`` is applied to the *new* determinants only:

    * the snapshot's entries keep their values; their rows are renumbered (an old global row ``g``
      moves up by the number of new determinants inserted at or before it, which every rank derives
      from one replicated list) and their columns are spread to the columns' new local positions;
    * the new columns come from applying ``op`` to the new determinants, rows anywhere in ``basis``;
    * the new rows of the old columns are the Hermitian mirror of the new columns' old rows, sent
      to the rank owning each old column.

    Determinant owners do not change when a basis grows (``routing_hash``), which is what makes the
    renumbering local. The result stores the three parts as separate segments, some with
    overlapping column ranges; the matvec sums them. Collective on a distributed basis.
    """
    comm = basis.comm if basis.is_distributed else None
    store = basis._keys
    n_local = len(store)
    old_keys = snapshot.keys.keys()
    position = store.find_rows(old_keys) if len(old_keys) else np.empty(0, dtype=np.int64)
    superset = bool(np.all(position != n_local))
    if comm is not None:
        superset = comm.allreduce(superset, op=MPI.LAND)
    if not superset:
        return None
    snapshot.mat.unbind_slots()
    rank = comm.rank if comm is not None else 0
    is_new = np.ones(n_local, dtype=bool)
    is_new[position] = False
    new_cols = np.flatnonzero(is_new)
    # Global old index each new determinant is inserted before; owner-major, hence sorted.
    inserted = snapshot.offsets[rank] + (new_cols - np.arange(len(new_cols)))
    if comm is not None:
        inserted = np.concatenate(comm.allgather(inserted))
    n = basis.size
    row_dtype = np.int32 if n < 2**31 else np.int64

    segments = []
    for col_begin, indptr, rows, values in snapshot.mat.segments:
        width = len(indptr) - 1
        if not width:
            continue
        old_rows = rows.astype(np.int64)
        moved = (old_rows + np.searchsorted(inserted, old_rows, side="right")).astype(row_dtype)
        cols = position[col_begin : col_begin + width]
        first = int(cols[0])
        counts = np.zeros(int(cols[-1]) - first + 1, dtype=np.int64)
        counts[cols - first] = np.diff(indptr)
        spread = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=spread[1:])
        segments.append((first, spread, moved, values))

    cols, rows, values = _columns_of(basis, op, new_cols)
    segment = _segment_of(cols, rows, values, row_dtype)
    if segment is not None:
        segments.append(segment)

    offset = int(basis.offset)
    if comm is not None:
        offsets = np.concatenate(([0], np.cumsum(comm.allgather(n_local)))).astype(np.int64)
        new_global = np.concatenate(comm.allgather(new_cols + offset))
    else:
        offsets = np.array([0, n_local], dtype=np.int64)
        new_global = new_cols
    mirrored = ~np.isin(rows, new_global)
    m_cols = rows[mirrored]  # old determinants: columns on their owners
    m_rows = cols[mirrored] + offset  # new determinants: global rows
    m_vals = np.conj(values[mirrored])
    if comm is not None:
        owner = np.searchsorted(offsets, m_cols, side="right") - 1
        order = np.argsort(owner, kind="stable")
        m_cols, m_rows, m_vals = m_cols[order], m_rows[order], m_vals[order]
        send = np.bincount(owner, minlength=comm.size).astype(np.int64)
        recv = np.empty_like(send)
        comm.Alltoall(send, recv)
        total = int(recv.sum())
        got_cols = np.empty(total, dtype=np.int64)
        got_rows = np.empty(total, dtype=np.int64)
        got_vals = np.empty(total, dtype=complex)
        comm.Alltoallv([m_cols, send], [got_cols, recv])
        comm.Alltoallv([m_rows, send], [got_rows, recv])
        comm.Alltoallv([m_vals, send], [got_vals, recv])
        m_cols, m_rows, m_vals = got_cols, got_rows, got_vals
    segment = _segment_of(m_cols - offset, m_rows, m_vals, row_dtype)
    if segment is not None:
        segments.append(segment)
    return DistributedSparseMatrix((n, n_local), segments, comm, offset)
