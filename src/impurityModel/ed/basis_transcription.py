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

#: Determinant keys materialised at once while the build admits the basis (~100 B each as Python objects).
_ADMIT_CHUNK = 1 << 16


def build_sparse_matrix(basis, op: ManyBodyOperator):
    """``op`` on ``basis`` as this rank's columns of the sparse operator: a
    :class:`~impurityModel.ed.DistributedSparseMatrix.DistributedSparseMatrix` of shape ``(N, N_local)`` (``P H P``:
    only rows inside the basis are stored).

    The columns are applied in batches of :data:`_SPARSE_BUILD_BATCH` elements, through :func:`basis_filter` so that
    most rows outside the basis are never generated, and the P slots are the basis's local order -- so the global row
    numbering is the basis's own, which needs the basis to be owned by ``routing_hash() % size``, as every
    distributed basis here is. Collective on a distributed basis.
    """
    if isinstance(op, dict):
        op = ManyBodyOperator(op)
    comm = basis.comm if basis.is_distributed else None
    matrix = DistributedSparseMatrix(
        op,
        (basis.n_bytes + 7) // 8,
        comm,
        batch=_SPARSE_BUILD_BATCH,
        keep_frontier=False,
        keep=basis_filter(basis),
    )
    local = basis.local_basis
    for start in range(0, len(local), _ADMIT_CHUNK):  # a chunk of key objects at a time, not every key at once
        matrix.admit_keys(list(local[start : start + _ADMIT_CHUNK]))
    matrix.build_columns()
    return matrix
