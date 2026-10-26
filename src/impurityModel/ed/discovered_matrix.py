"""The discovered-key matrix: ``H`` restricted to everything a unit has discovered, built incrementally.

``K`` is every determinant that the H-image of an admitted determinant has reached (hash-owned:
``routing_hash() % size``, owners never move); ``P`` is the admitted subset. This rank holds the
**columns of its admitted determinants** (``P_local``, the *P slots*) over **rows for all of K**
(the *K slots*, append-only per rank). A column is the complete restricted image ``H|D>`` at
``cutoff = 0``, stored once when ``D`` is admitted (:meth:`grow`). So ``M V`` for ``V`` on ``P`` is
exactly ``H V`` inside the unit's restrictions: rows in ``P`` are the Lanczos step, rows in
``F = K \\ P`` are the frontier amplitudes an admission decision reads.

**Row partition differs from column partition.** Entries store their row as
``owner << 32 | slot`` (int64). The matvec scatters through a per-call ``base[owner]`` table
(``mpi_comm.MatvecExchangePlan``'s send/self buffer offsets), so no row index is ever rewritten and
nothing is renumbered when ``K`` grows. ``rows="all"`` returns the ``K`` rows; ``rows="P"`` returns
only the rows that are in ``P`` (the frozen recurrence's ``P H P``), translated through a per-epoch
table of every rank's ``K -> P slot`` map.

**Collective contract.** :meth:`grow`, :meth:`drop_columns`, :meth:`compact_p_slots`,
:meth:`exchange_plan`, :meth:`matvec` and :meth:`to_scipy` are called on every rank, in the same
order, each with its own (possibly empty) arguments. The first three advance ``epoch``, which
invalidates the cached cross-rank tables; only a matrix whose ranks all made the same calls has
matching epochs.

``comm`` is ``None`` (or of size 1) for a matrix on one rank.
"""

import numpy as np
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.DistributedSparseMatrix import scatter_discovered
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    SlaterDeterminant,
    applyOp,
    pack_determinants_ordered_cy,
)
from impurityModel.ed.mpi_comm import MatvecExchangePlan, routed_index_lookup

#: Matrix elements per collective key lookup in :meth:`DiscoveredMatrix.grow` (~80 MiB of transient).
_GROW_BATCH = 1 << 19

_SLOT_MASK = (1 << 32) - 1


class _Layout:
    """What one matvec needs: the exchange plan (``None`` on one rank) and the row-index tables."""

    __slots__ = ("base", "epoch", "goff", "n_rows", "plan", "ptab", "rows", "send_rows", "use_ptab")

    def __init__(self, rows, plan, base, goff, ptab, n_rows, epoch):
        self.rows = rows
        self.plan = plan
        self.base = base
        self.goff = goff
        self.ptab = ptab
        self.use_ptab = rows == "P"
        self.n_rows = n_rows
        self.send_rows = 0 if plan is None else plan.total_send_rows
        self.epoch = epoch


class DiscoveredMatrix:
    """See the module docstring.

    Parameters
    ----------
    hop : ManyBodyOperator
        The operator whose images are the columns. Its restrictions (and weighted restrictions)
        are applied at emission, so the stored image is the restricted one.
    chunks_per_state : int
        ``uint64`` words per determinant key.
    comm : mpi4py communicator, optional
        ``None`` (or size 1) for one rank.
    batch : int
        Matrix elements per collective key lookup in :meth:`grow`.
    keep_frontier : bool
        ``False`` stores only the rows that are in ``K`` when a column is built and never adds the rows an
        image reaches to ``K``: ``K = P`` and the matrix is ``P H P`` (build it in one :meth:`grow` /
        :meth:`build_columns`; a row admitted later is missing from the columns built before it).
    """

    def __init__(self, hop, chunks_per_state, comm=None, batch=_GROW_BATCH, keep_frontier=True):
        self.hop = hop
        self.cps = int(chunks_per_state)
        self.comm = comm if comm is not None and comm.size > 1 else None
        self.rank = 0 if self.comm is None else self.comm.rank
        self.size = 1 if self.comm is None else self.comm.size
        self._batch = int(batch)
        self.keep_frontier = bool(keep_frontier)
        # K store: sorted unique keys; slots are append-only, positions are the sorted order.
        self._kstore = ManyBodyState.from_keys([])
        self._slot_of_pos = np.zeros(0, dtype=np.int64)
        self._pos_of_slot = np.zeros(0, dtype=np.int64)
        self._k_pslot = np.zeros(0, dtype=np.int64)  # K slot -> P slot, -1 outside P
        # P columns: segments (col_begin, indptr, packed rows, data), one per grow batch.
        self._segments = []
        self._alive = np.zeros(16, dtype=np.int8)
        self._p_kslot = np.zeros(0, dtype=np.int64)  # P slot -> K slot of the column's own key
        self.epoch = 0
        self.n_applied = 0  # hop applications, one per admitted key (tests count it)
        self._tables = None

    # -- sizes ----------------------------------------------------------------------------------

    @property
    def n_k_local(self):
        return int(self._slot_of_pos.shape[0])

    @property
    def n_p_local(self):
        """P slots, dead ones included until :meth:`compact_p_slots`."""
        return int(self._p_kslot.shape[0])

    @property
    def n_alive_local(self):
        return int(self._alive[: self.n_p_local].sum())

    @property
    def nnz(self):
        return int(sum(len(packed) for _, _, packed, _ in self._segments))

    @property
    def nbytes(self):
        seg = sum(ip.nbytes + pk.nbytes + d.nbytes for _, ip, pk, d in self._segments)
        return int(
            seg
            + self._kstore.memory_bytes()
            + self._slot_of_pos.nbytes
            + self._pos_of_slot.nbytes
            + self._k_pslot.nbytes
            + self._p_kslot.nbytes
            + self._alive.nbytes
        )

    @property
    def dtype(self):
        return np.dtype(complex)

    def alive(self):
        """Boolean view over the P slots: is the column in use."""
        return self._alive[: self.n_p_local].astype(bool)

    # -- the K store ------------------------------------------------------------------------------

    def _append_keys(self, new):
        """Append the keys of the sorted, unique, absent block ``new`` as fresh K slots (in key order)."""
        m = len(new)
        if m == 0:
            return
        n_old = self.n_k_local
        self._kstore.merge_keys(new)
        pos_new = self._kstore.find_rows(new.keys())
        slot_of_pos = np.empty(n_old + m, dtype=np.int64)
        is_old = np.ones(n_old + m, dtype=bool)
        is_old[pos_new] = False
        slot_of_pos[is_old] = self._slot_of_pos
        slot_of_pos[pos_new] = n_old + np.arange(m, dtype=np.int64)
        pos_of_slot = np.empty(n_old + m, dtype=np.int64)
        pos_of_slot[slot_of_pos] = np.arange(n_old + m, dtype=np.int64)
        self._slot_of_pos = slot_of_pos
        self._pos_of_slot = pos_of_slot
        self._k_pslot = np.concatenate((self._k_pslot, np.full(m, -1, dtype=np.int64)))

    def _slots_of_packed(self, buf, append=True):
        """K slot of every determinant packed in ``buf`` (``uint64`` chunks), appending the absent ones
        (``append=False``: ``-1`` for an absent one)."""
        buf = np.ascontiguousarray(buf, dtype=np.uint64)
        cps = self.cps
        rows = self._kstore.find_rows_packed(buf, cps)
        miss = rows == len(self._kstore)
        if not append:
            out = np.full(len(rows), -1, dtype=np.int64)
            out[~miss] = self._slot_of_pos[rows[~miss]]
            return out
        if miss.any():
            chunks = buf.reshape(-1, cps)[miss]
            self._append_keys(
                ManyBodyState.from_keys([SlaterDeterminant(tuple(int(c) for c in row)) for row in chunks])
            )
            rows = self._kstore.find_rows_packed(buf, cps)
        return self._slot_of_pos[rows]

    def _lookup(self, keys, append=True):
        """Packed ``owner << 32 | slot`` of every key, appending the new ones to their owner's K store
        (``append=False``: ``-1`` for a key that is not in ``K``). Collective."""

        def resolve(buf):
            slots = self._slots_of_packed(buf, append)
            return np.where(slots >= 0, (np.int64(self.rank) << 32) | np.maximum(slots, 0), -1)

        if self.comm is None:
            order = np.arange(len(keys), dtype=np.int64)
            return resolve(pack_determinants_ordered_cy(keys, order, self.cps))
        return routed_index_lookup(keys, self.cps, resolve, self.comm)

    def key_slots(self, keys):
        """K slot of each local ``key`` (this rank owns them all), ``-1`` for one not in K. Not collective."""
        pos = self._kstore.find_rows(keys)
        out = np.full(len(keys), -1, dtype=np.int64)
        hit = pos < len(self._kstore)
        out[hit] = self._slot_of_pos[pos[hit]]
        return out

    def keys_of(self, kslots):
        """The determinants at K slots ``kslots``."""
        return [self._kstore.key_at(int(self._pos_of_slot[s])) for s in np.asarray(kslots, dtype=np.int64)]

    def p_keys(self):
        """The determinant of every P slot, in slot order (dead slots included)."""
        return self.keys_of(self._p_kslot)

    def k_to_p_slot(self):
        """Copy of the K slot -> P slot map (``-1`` outside P)."""
        return self._k_pslot.copy()

    def p_to_k_slot(self):
        """Copy of the P slot -> K slot map."""
        return self._p_kslot.copy()

    # -- growth ---------------------------------------------------------------------------------

    def _new_keys(self, keys):
        """The determinants of ``keys`` that are not in ``P``, in ascending key order, registered in ``K``."""
        keys = list(keys)
        for key in keys:
            if self.size > 1 and key.routing_hash() % self.size != self.rank:
                raise ValueError("this takes determinants the rank owns")
        todo = ManyBodyState.from_keys(keys).keys() if keys else []
        if todo:
            slots = self.key_slots(todo)
            fresh = slots < 0
            if fresh.any():
                self._append_keys(ManyBodyState.from_keys([k for k, f in zip(todo, fresh) if f]))
                slots = self.key_slots(todo)
            keep = self._k_pslot[slots] < 0
            todo = [k for k, kp in zip(todo, keep) if kp]
        return todo

    def _batches(self, todo):
        """Apply the operator to each of ``todo`` and look the images up, a batch of about ``batch`` matrix
        elements at a time. Yields ``(batch_keys, counts, packed, vals)``. **Collective**: every rank runs
        the same number of batches (``allreduce(not exhausted, LOR)``), a rank with nothing left enters each
        one empty."""
        i = 0
        unit = ManyBodyState()
        while True:
            batch_keys, counts, out_keys, vals = [], [], [], []
            elements = 0
            while i < len(todo) and elements < self._batch:
                unit[todo[i]] = 1.0
                image = applyOp(self.hop, unit, cutoff=0)
                unit.erase(todo[i])
                self.n_applied += 1
                ks = image.keys()
                counts.append(len(ks))
                if ks:
                    out_keys.extend(ks)
                    vals.append(np.asarray(image)[:, 0].copy())
                elements += len(ks)
                batch_keys.append(todo[i])
                i += 1
            packed = self._lookup(out_keys, append=self.keep_frontier)
            data = np.concatenate(vals) if vals else np.zeros(0, dtype=complex)
            if not self.keep_frontier and len(packed):
                hit = packed >= 0
                if not hit.all():
                    col = np.repeat(np.arange(len(counts)), counts)
                    counts = np.bincount(col[hit], minlength=len(counts))
                    packed, data = packed[hit], data[hit]
            yield batch_keys, counts, packed, data
            if self.size == 1:
                if i >= len(todo):
                    break
            elif not self.comm.allreduce(i < len(todo), op=MPI.LOR):
                break

    def grow(self, keys):
        """Admit ``keys`` (determinants this rank owns) to ``P``: store each one's complete image as a
        column and (``keep_frontier``) add every key the images reach to ``K``. A key already in ``P`` is
        skipped, one that was dropped is re-admitted as a new column. Returns the P slots of the keys
        admitted, in the order of their columns (ascending key order). **Collective**."""
        todo = self._new_keys(keys)
        admitted = []
        for batch_keys, counts, packed, data in self._batches(todo):
            admitted.extend(self._append_columns(batch_keys, counts, packed, data))
        self.epoch += 1
        self._tables = None
        return np.asarray(admitted, dtype=np.int64)

    def admit_keys(self, keys):
        """Admit ``keys`` to ``P`` **without** applying the operator: the columns are empty until
        :meth:`build_columns`. The matrix-free modes use the matrix as their slot manager (P slots, the key
        store, the live mask) and build it once at the freeze. Returns the new P slots. Not collective in
        itself, but advances the epoch: every rank calls it, each with its own keys."""
        todo = self._new_keys(keys)
        out = self._append_columns(todo, [0] * len(todo), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=complex))
        self.epoch += 1
        self._tables = None
        return np.asarray(out, dtype=np.int64)

    def build_columns(self):
        """Apply the operator to every live P column (in slot order) and store the images, replacing the
        columns :meth:`admit_keys` left empty. Rows outside ``K`` are dropped when ``keep_frontier`` is off
        (``K = P``: the ``P H P`` matrix). **Collective.**"""
        alive = self.alive()
        keys = [k for k, a in zip(self.p_keys(), alive) if a]
        if not alive.all():
            raise ValueError("build_columns needs every P slot live (compact_p_slots first)")
        self._segments = []
        col_begin = 0
        for batch_keys, counts, packed, data in self._batches(keys):
            n = len(batch_keys)
            if n:
                indptr = np.zeros(n + 1, dtype=np.int64)
                np.cumsum(np.asarray(counts, dtype=np.int64), out=indptr[1:])
                self._segments.append((col_begin, indptr, np.ascontiguousarray(packed, dtype=np.int64), data))
                col_begin += n
        self.epoch += 1
        self._tables = None

    def _append_columns(self, batch_keys, counts, packed, data):
        n = len(batch_keys)
        if n == 0:
            return []
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.asarray(counts, dtype=np.int64), out=indptr[1:])
        col_begin = self.n_p_local
        kslots = self.key_slots(batch_keys)
        self._segments.append((col_begin, indptr, np.ascontiguousarray(packed, dtype=np.int64), data))
        if col_begin + n > self._alive.shape[0]:
            grown = np.zeros(max(2 * self._alive.shape[0], col_begin + n), dtype=np.int8)
            grown[:col_begin] = self._alive[:col_begin]
            self._alive = grown
        self._alive[col_begin : col_begin + n] = 1
        self._p_kslot = np.concatenate((self._p_kslot, kslots))
        pslots = col_begin + np.arange(n, dtype=np.int64)
        self._k_pslot[kslots] = pslots
        return pslots.tolist()

    def drop_columns(self, pslots):
        """Remove the columns at ``pslots`` from ``P``: the column is dead (skipped by the matvec) and
        its determinant stays in ``K``. The P slot stays allocated until :meth:`compact_p_slots`.
        **Collective** in the sense of the epoch: every rank calls it, with its own (possibly empty)
        slots."""
        pslots = np.asarray(pslots, dtype=np.int64)
        if pslots.size:
            live = self._alive[pslots].astype(bool)
            pslots = pslots[live]
            self._alive[pslots] = 0
            self._k_pslot[self._p_kslot[pslots]] = -1
        self.epoch += 1
        self._tables = None

    def compact_p_slots(self):
        """Drop the dead P slots for good and renumber the live ones in order. Returns the old P slots
        that were kept (``q = q[keep]`` carries a dense block over). **Collective** (epoch)."""
        alive = self.alive()
        keep = np.flatnonzero(alive)
        segments = []
        col_begin = 0
        for c0, indptr, packed, data in self._segments:
            n = len(indptr) - 1
            seg_alive = alive[c0 : c0 + n]
            n_keep = int(seg_alive.sum())
            if n_keep == 0:
                continue
            counts = np.diff(indptr)
            if n_keep == n:
                new = (col_begin, indptr, packed, data)
            else:
                entry = np.repeat(seg_alive, counts)
                new_ptr = np.zeros(n_keep + 1, dtype=np.int64)
                np.cumsum(counts[seg_alive], out=new_ptr[1:])
                new = (col_begin, new_ptr, packed[entry], data[entry])
            segments.append(new)
            col_begin += n_keep
        self._segments = segments
        self._p_kslot = self._p_kslot[keep]
        self._k_pslot[:] = -1
        self._k_pslot[self._p_kslot] = np.arange(len(keep), dtype=np.int64)
        self._alive = np.ones(max(16, len(keep)), dtype=np.int8)
        self.epoch += 1
        self._tables = None
        return keep

    # -- the exchange layout ------------------------------------------------------------------------

    def _gather_tables(self, rows):
        """``(row_counts, col_counts, goff, ptab)`` for ``rows``; cached per ``(rows, epoch)``. Collective
        on a cache miss, and a miss is rank-invariant because the epoch is."""
        if rows not in ("all", "P"):
            raise ValueError(f"rows must be 'all' or 'P', got {rows!r}")
        if self._tables is not None and self._tables[0] == (rows, self.epoch):
            return self._tables[1]
        n_k, n_p = self.n_k_local, self.n_p_local
        if self.comm is None:
            tables = (
                np.array([n_k if rows == "all" else n_p], dtype=np.int64),
                np.array([n_p], dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                self._k_pslot if rows == "P" else np.zeros(0, dtype=np.int64),
            )
        else:
            comm = self.comm
            counts = np.empty((comm.size, 2), dtype=np.int64)
            comm.Allgather(np.array([n_k, n_p], dtype=np.int64), counts)
            col_counts = counts[:, 1].copy()
            row_counts = (counts[:, 0] if rows == "all" else counts[:, 1]).copy()
            goff = np.concatenate(([0], np.cumsum(counts[:, 0])[:-1])).astype(np.int64)
            ptab = np.zeros(0, dtype=np.int64)
            if rows == "P":
                ptab = np.empty(int(counts[:, 0].sum()), dtype=np.int64)
                comm.Allgatherv(self._k_pslot, [ptab, counts[:, 0].copy(), goff, MPI.INT64_T])
            tables = (row_counts, col_counts, goff, ptab)
        self._tables = ((rows, self.epoch), tables)
        return tables

    def _owner_flags(self, goff, ptab, use_ptab):
        present = np.zeros(self.size, dtype=bool)
        for _, _, packed, _ in self._segments:
            if not len(packed):
                continue
            owners = packed >> 32
            if use_ptab:
                owners = owners[ptab[goff[owners] + (packed & _SLOT_MASK)] >= 0]
            present[np.unique(owners)] = True
        return present

    def exchange_plan(self, width, rows="all", site="discovered_matrix"):
        """The layout for blocks up to ``width`` columns wide returning ``rows`` (``"all"``: every K
        row; ``"P"``: the P rows). **Collective** (three tiny collectives, plus the table gather when
        the epoch changed); build one per kernel call."""
        row_counts, col_counts, goff, ptab = self._gather_tables(rows)
        if self.comm is None:
            return _Layout(rows, None, np.zeros(1, dtype=np.int64), goff, ptab, int(row_counts[0]), self.epoch)
        flags = self._owner_flags(goff, ptab, rows == "P")
        plan = MatvecExchangePlan(
            self.comm,
            col_counts,
            None,
            flags,
            int(width),
            config.GS_MATVEC_EXCHANGE_BYTES.get(),
            row_counts=row_counts,
        )
        base = np.full(self.size, -1, dtype=np.int64)
        for i, dest in enumerate(plan.destinations):
            base[dest] = int(plan.send_row_offsets[i])
        base[self.rank] = plan.total_send_rows
        return _Layout(rows, plan, base, goff, ptab, int(row_counts[self.rank]), self.epoch)

    # -- the matvec -------------------------------------------------------------------------------

    def matvec(self, X, rows="all", plan=None, out=None):
        """``Y = M X`` for a block ``X`` on the P slots, shape ``(n_p_local, p)``. Returns this rank's
        rows: ``(n_k_local, p)`` for ``rows="all"``, ``(n_p_local, p)`` for ``rows="P"``. ``plan`` is
        :meth:`exchange_plan`'s layout of a width at least ``p`` and the same ``rows``. **Collective**."""
        X = np.ascontiguousarray(X, dtype=complex)
        if X.ndim != 2 or X.shape[0] != self.n_p_local:
            raise ValueError(f"block has {X.shape} rows, the matrix has {self.n_p_local} P slots")
        p = X.shape[1]
        if plan is None:
            plan = self.exchange_plan(p, rows)
        if plan.rows != rows or plan.epoch != self.epoch:
            raise ValueError("the layout is stale or was built for other rows")
        if out is None:
            out = np.empty((plan.n_rows, p), dtype=complex)
        alive = self._alive[: self.n_p_local]
        if plan.plan is None:
            out[...] = 0.0
            scatter_discovered(
                self._segments,
                alive,
                X,
                0,
                p,
                plan.base,
                plan.goff,
                plan.ptab,
                plan.use_ptab,
                0,
                np.empty((0, p), dtype=complex),
                out,
            )
            return out
        ex = plan.plan
        for c0, c1 in ex.column_rounds(p):
            wc = c1 - c0
            send_view = ex.send_buffer(wc)
            self_view = ex.self_slot(wc)
            send_view[...] = 0.0
            self_view[...] = 0.0
            scatter_discovered(
                self._segments,
                alive,
                X,
                c0,
                wc,
                plan.base,
                plan.goff,
                plan.ptab,
                plan.use_ptab,
                plan.send_rows,
                send_view,
                self_view,
            )
            ex.exchange(wc)
            ex.accumulate(out[:, c0:c1], wc)
        return out

    # -- conversions (tests, small dense eigensolves) -----------------------------------------------

    def to_scipy(self, rows="all"):
        """This rank's live columns as a scipy CSC of shape ``(total rows, n_p_local)``; global row
        ``r`` is the rows of rank 0, then rank 1, ... (K slots or P slots in order). Dead columns are
        empty. **Collective** on a distributed matrix."""
        row_counts, _cc, goff, ptab = self._gather_tables(rows)
        roff = np.concatenate(([0], np.cumsum(row_counts)))
        n_rows = int(roff[-1])
        alive = self.alive()
        cols, rws, vals = [], [], []
        for c0, indptr, packed, data in self._segments:
            n = len(indptr) - 1
            col = np.repeat(np.arange(c0, c0 + n, dtype=np.int64), np.diff(indptr))
            owner = packed >> 32
            slot = packed & _SLOT_MASK
            if rows == "P":
                slot = ptab[goff[owner] + slot]
            ok = alive[col] & (slot >= 0)
            cols.append(col[ok])
            rws.append(roff[owner[ok]] + slot[ok])
            vals.append(data[ok])
        if not cols:
            return sps.csc_array((n_rows, self.n_p_local), dtype=complex)
        out = sps.coo_array(
            (np.concatenate(vals), (np.concatenate(rws), np.concatenate(cols))), shape=(n_rows, self.n_p_local)
        ).tocsc()
        out.sum_duplicates()
        return out

    def row_offset(self, rows="all"):
        """Global index of this rank's first row in :meth:`to_scipy`'s numbering. **Collective** on a
        cache miss."""
        row_counts = self._gather_tables(rows)[0]
        return int(row_counts[: self.rank].sum())

    def __repr__(self):
        return (
            f"DiscoveredMatrix(K_local={self.n_k_local}, P_local={self.n_p_local}, nnz={self.nnz}, "
            f"distributed={self.comm is not None})"
        )
