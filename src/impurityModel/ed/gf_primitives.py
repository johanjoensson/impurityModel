"""Bottom-layer Green's-function primitives: QR/state-vector plumbing, the block-tridiagonal
continued fraction, and the ``truncation_threshold``-capping basis proxy.

Split out of :mod:`greens_function` (which re-exports everything here for backwards
compatibility) so the solver-policy code in that module and in :mod:`gf_convergence` /
:mod:`gf_shift_recycling` sits on a small, dependency-free layer: nothing here imports from
either of those two, or from :mod:`greens_function` itself.
"""

from contextlib import contextmanager

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed.basis_transcription import (
    build_distributed_vector,
    build_vector,
    comm_support_filter,
)
from impurityModel.ed.BlockLanczosCore import BETA_BLOWUP_FACTOR
from impurityModel.ed.manybody_basis import collective_first_keys, collective_top_k_bounds
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.memory_estimate import emit_memory_warning, format_bytes, gf_resident_bytes, gf_resident_note


def build_qr(psi):
    """
    Perform an economic QR decomposition of a state matrix.

    Parameters
    ----------
    psi : ndarray
        The input state matrix.

    Returns
    -------
    psi_orthogonal : ndarray
        The orthogonalized matrix Q.
    r : ndarray
        The upper triangular matrix R.
    """
    # Do a QR decomposition of the starting block.
    # Later on, use r to restore the psi block
    psi, r = sp.linalg.qr(psi.copy(), mode="economic", overwrite_a=True, check_finite=False, pivoting=False)
    return np.ascontiguousarray(psi), r


def _scatter_qr_columns(comm, psi_dense, r, local_size):
    """Scatter the row-distributed QR factor ``Q`` (held on rank 0) across MPI ranks.

    Rank 0 holds the full ``(N, n)`` ``Q`` and the ``(n, n)`` ``R`` after :func:`build_qr`.
    Broadcast ``R`` and the column count, then ``Scatterv`` ``Q``'s rows onto each rank's
    local partition (``local_size`` rows). Shared by ``block_green_impl`` (sparse branch)
    and ``block_Green_sparse``.

    Returns
    -------
    psi_dense_local : ndarray
        This rank's ``(local_size, n)`` slice of ``Q``.
    r : ndarray
        The ``(n, n)`` ``R`` factor (replicated on every rank).
    """
    rank = comm.rank
    r = comm.bcast(r if rank == 0 else None, root=0)
    columns = comm.bcast(psi_dense.shape[1] if rank == 0 else None, root=0)
    psi_dense_local = np.empty((local_size, columns), dtype=complex, order="C")
    send_counts = np.empty((comm.size), dtype=int) if rank == 0 else None
    comm.Gather(np.array([psi_dense_local.size]), send_counts, root=0)
    offsets = np.array([np.sum(send_counts[:rr]) for rr in range(comm.size)], dtype=int) if rank == 0 else None
    comm.Scatterv(
        [psi_dense, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None,
        psi_dense_local,
        root=0,
    )
    return psi_dense_local, r


def _gather_qr_rows(comm, psi_local):
    """Gather each rank's local ``(local_size, n)`` row-block onto rank 0 as ``(N, n)``.

    The ``Gatherv`` counterpart of :func:`_scatter_qr_columns`. Every rank contributes only
    its own local block; only rank 0 ever holds the full ``(N, n)`` array -- unavoidable
    since the seed QR must run on the whole block, but a real bound compared to
    :func:`~impurityModel.ed.basis_transcription.build_vector`'s ``root=0`` path, which
    allocates the full ``(n, N)`` array on *every* rank before reducing into rank 0's copy
    (see that function's ``Reduce`` branch and CLAUDE.md's "No full state-vector gathers").

    Returns
    -------
    ndarray or None
        The ``(N, n)`` matrix on rank 0; ``None`` on every other rank.
    """
    rank = comm.rank
    n = psi_local.shape[1]
    recv_counts = np.empty(comm.size, dtype=int) if rank == 0 else None
    comm.Gather(np.array([psi_local.size]), recv_counts, root=0)
    if rank == 0:
        offsets = np.array([np.sum(recv_counts[:rr]) for rr in range(comm.size)], dtype=int)
        # `// n` only when there is a column to divide by: a width-0 seed block gives every
        # rank a (local_size, 0) slice and a zero element count, and `block_green_impl` calls
        # this *before* its own `shape[1] == 0` early return (gf_solvers.py), so this must stay
        # total the way the `build_vector` path it replaced was. A (0, 0) result is what the
        # QR and scatter below both expect on that path.
        rows = int(np.sum(recv_counts)) // n if n else 0
        psi_dense = np.empty((rows, n), dtype=complex, order="C")
    else:
        offsets = None
        psi_dense = None
    comm.Gatherv(
        np.ascontiguousarray(psi_local),
        [psi_dense, recv_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None,
        root=0,
    )
    return psi_dense


def _distributed_seed_qr(basis, psi_arr, slaterWeightMin=0):
    """Row-distributed orthonormal seed block + its ``R`` factor.

    Shared preamble of every resolvent solver that needs an orthonormal seed block
    distributed by ``basis``'s ownership (:func:`block_Green_sparse`, the sparse branch
    of :func:`block_green_impl`, :class:`KrylovShiftedResolvent`): build the dense seed
    matrix, QR it on rank 0 (:func:`build_qr`), then scatter ``Q``'s rows onto each rank's
    local partition (:func:`_scatter_qr_columns`) so every rank ends up with only its own
    slice. Serial (``basis.comm is None``) just runs ``build_qr`` directly.

    The MPI branch builds the seed matrix via
    :func:`~impurityModel.ed.basis_transcription.build_distributed_vector` (local-shaped,
    ``(n, len(local_basis))`` per rank) and :func:`_gather_qr_rows`, not
    ``build_vector(..., root=0)``: the latter allocates the full ``(n, basis.size)`` array
    on every rank before its ``Reduce`` -- unbounded in the basis size and invisible to
    :func:`~impurityModel.ed.memory_estimate.estimate_gf_peak_bytes` (see
    ``doc/plans/dc_smo_memory.md``, "GF unit memory", item 5). Only rank 0 now holds a
    global-shaped array, which is unavoidable since the QR itself runs there.

    Returns
    -------
    psi_dense_local : ndarray
        This rank's ``(local_size, n)`` slice of the orthonormal seed block ``Q``.
    r : ndarray
        The ``(n, n)`` ``R`` factor (replicated on every rank).
    """
    comm = basis.comm
    mpi = comm is not None
    rank = comm.rank if mpi else 0
    if mpi:
        psi_local = build_distributed_vector(basis, psi_arr).T
        if slaterWeightMin > 0:
            # Reproduce build_vector's amplitude cutoff (entries below slaterWeightMin are
            # left at their initialized zero there); applied post-hoc since
            # build_distributed_vector has no cutoff parameter of its own.
            psi_local[np.abs(psi_local) < slaterWeightMin] = 0
        psi_dense = _gather_qr_rows(comm, psi_local)
    else:
        psi_dense = build_vector(basis, psi_arr, slaterWeightMin=slaterWeightMin).T
    r = None
    if rank == 0:
        psi_dense, r = build_qr(psi_dense)
    if mpi:
        psi_dense_local, r = _scatter_qr_columns(
            comm, psi_dense if rank == 0 else None, r if rank == 0 else None, len(basis.local_basis)
        )
    else:
        psi_dense_local = psi_dense
    return psi_dense_local, r


def calc_thermally_averaged_G(alphas, betas, r, mesh, es, e0, tau, delta):
    """
    Calculate the thermally averaged Green's function over multiple initial states.

    Parameters
    ----------
    alphas : list of list of ndarray
    betas : list of list of ndarray
    r : list of ndarray
    mesh : ndarray
    es : list of float
    e0 : float
    tau : float
    delta : float

    Returns
    -------
    G_avg : ndarray
    """
    n_ops = r[0].shape[-1]
    G_avg = np.zeros((len(mesh), n_ops, n_ops), dtype=complex)

    for e, alphas_e, betas_e, r_e in zip(es, alphas, betas, r):
        G_avg += calc_G(alphas_e, betas_e, r_e, mesh, e, delta) * np.exp(-(e - e0) / tau)

    return G_avg


class SupportDecisions:
    """The block-agnostic decisions of a capped support: counts, the cap, the memory guard, the boundary
    ranking, and the log phrases.

    Everything here works on scalars, arrays and key bytes -- never on a ``ManyBodyState`` -- so the same
    decisions serve :class:`SupportPolicy` (which applies them to the matvec's block states) and the
    recurrence on a discovered-key matrix (``gf_recurrence``, which applies them to dense frontier rows).
    :class:`SupportPolicy` holds one and forwards the attributes below under their old names.

    ``global_count`` is the replicated number of admitted determinants; ``frozen`` is whether the retained set
    is final (replicated: derived only from allreduced data).
    """

    def __init__(self, comm, cap, global_count, memory_budget=None, memory_policy="tighten"):
        self.comm = comm
        self.cap = int(cap)
        self.global_count = int(global_count)
        self.frozen = self.global_count >= self.cap
        self.cap_hit = self.frozen
        self.memory_budget = memory_budget
        self.memory_policy = memory_policy
        self.memory_frozen = False
        self.memory_warned = False
        #: The support is closed under ``H``: every row the matvec reaches is already retained, so nothing can ever be
        #: admitted and the support is final without being cap-bound (``cap_hit`` stays as it was).
        self.closed = False

    def allreduce_sum(self, value):
        if self.comm is None or self.comm.size == 1:
            return value
        return self.comm.allreduce(value, op=MPI.SUM)

    def over_memory_budget(self):
        """Collective (on the color's comm) when the guard is on: is the color's MAX RSS at budget?

        Called only on the pre-freeze path, right beside that path's own admission-count
        allreduce, so every rank of the color reaches it equally often. The answer is replicated.
        """
        if self.memory_budget is None or self.memory_warned:
            return False
        rss = gf_resident_bytes()
        if self.comm is not None and self.comm.size > 1:
            rss = self.comm.allreduce(rss, op=MPI.MAX)
        if rss < self.memory_budget:
            return False
        root = self.comm is None or self.comm.rank == 0
        if self.memory_policy == "tighten":
            cap_text = f"GF cap {self.cap:,}" if self.cap < 2**62 else "no GF cap"
            emit_memory_warning(
                f"WARNING determinant cap: a Green's-function unit stopped growing at {self.global_count:,} "
                f"determinants ({cap_text}): measured {format_bytes(rss)}/rank ({gf_resident_note()}) reached the "
                f"{format_bytes(self.memory_budget)} memory budget.",
                root=root,
                kind="gf-memory",
            )
            return True
        self.memory_warned = True
        emit_memory_warning(
            f"WARNING determinant cap: a Green's-function unit at {self.global_count:,} determinants uses "
            f"{format_bytes(rss)}/rank >= the {format_bytes(self.memory_budget)} memory budget. Your "
            f"truncation_threshold={self.cap:,} is kept as set; if the job is killed, lower it, use 'auto', or "
            "run fewer ranks per node.",
            root=root,
            kind="gf-memory",
        )
        return False

    def freeze_on_memory(self):
        """The memory guard tripped: freeze where the support stands. Nothing new is admitted from here on,
        exactly as a cap hit at the current size (the recurrence continues as the exact PHP on what is
        retained)."""
        self.frozen = True
        self.cap_hit = True
        self.memory_frozen = True

    def rank_boundary(self, norms2):
        """Rank the candidates' max column ``|amp|^2`` for the ``cap - retained`` free slots: whole near-tie
        groups above the boundary, then the boundary group. Returns
        :func:`~impurityModel.ed.manybody_basis.collective_top_k_bounds`'s ``(above, boundary, n_fill)``.
        Collective."""
        return collective_top_k_bounds(norms2, self.cap - self.global_count, self.comm)

    def first_keys(self, key_bytes, n_fill):
        """The first ``n_fill`` boundary-group keys in determinant-key order, collectively
        (:func:`~impurityModel.ed.manybody_basis.collective_first_keys`)."""
        return collective_first_keys(key_bytes, n_fill, self.comm)

    def freeze_closed(self):
        """The support is closed under ``H`` (a replicated verdict: no rank has a row outside it): it is final. Freezes
        without marking the cap or the memory guard, which did not stop it."""
        self.frozen = True
        self.closed = True

    def freeze_at_cap(self, n_admitted_local):
        """Count what the overflow step admitted (summed over the ranks) and freeze at the cap."""
        self.global_count += self.allreduce_sum(n_admitted_local)
        self.frozen = True
        self.cap_hit = True

    def stop_phrase(self):
        """How the support stopped growing, as the opening of a log line: held or frozen at the cap."""
        if self.memory_frozen:
            return "GF basis held at"
        if self.closed and not self.cap_hit:
            return "GF basis closed under H at"
        return "GF basis frozen at the cap at"

    def freeze_message(self):
        """One-line description of the cap state (rank-0 logging)."""
        if self.memory_frozen:
            return (
                f"GF basis held at {self.global_count:,} determinants by the measured-memory guard; the Green's "
                "function is exact on the retained subspace."
            )
        if self.closed and not self.cap_hit:
            return (
                f"GF basis closed under H at {self.global_count:,} determinants: every determinant H reaches is "
                "retained, so the Green's function is that of the whole reachable space."
            )
        return (
            f"GF basis frozen at {self.global_count:,} determinants by the cap of {self.cap:,}; the Green's "
            "function is exact on the retained subspace."
        )


class SupportPolicy:
    """Enforce ``truncation_threshold`` on the sparse-kernel GF recurrence.

    ``block_lanczos``'s matvec discovers new Slater determinants every step, so the
    live block-state support (and, at reort != none, the Krylov store) grows without
    bound — the excited ``Basis`` itself stays frozen and never sees them. This proxy
    wraps that basis and caps the growth at the point where every residual row sits on
    its hash-owner rank: the ``redistribute_block`` call. At ``APPLY_ROW_CHUNKS`` > 1
    (the default, 4; ``_lanczos_step.pxi``'s row-chunked matvec) that is ``n_chunks``
    calls per step, one per chunk of ``q_curr``'s rows. The block-Lanczos step brackets
    them with :meth:`begin_step` / :meth:`finish_step`, so the chunks pass through and the
    freeze/admit decision below runs once, on the step's whole summed output: a chunk
    holds partial amplitudes, and deciding per chunk made the frozen set depend on the
    colour layout. The cap-crossing step therefore holds its whole unprojected output
    (``retained`` plus that step's new rows) until :meth:`finish_step` trims it -- one
    step per unit, not priced by ``memory_estimate.estimate_gf_peak_bytes``; the memory
    guard still checks every chunk.

    Policy (freeze-growth + importance-ranked boundary admission):

    * while ``retained + n_new <= cap``: admit every newly discovered determinant;
    * on the single overflow step: rank that step's candidate rows by max column
      ``|amp|^2`` of the residual and admit exactly the top ``cap - retained``
      (``collective_top_k_bounds``: whole near-tie groups, then the boundary group in
      determinant-key order -- collective and layout-invariant), then freeze;
    * after the freeze: drop non-retained rows of every residual (rank-local
      ``keep_rows`` merge; ownership routing makes membership checks local).

    Why this is safe: every previously accepted Krylov block has support inside the
    retained set, so the diagonal projector ``P`` is invisible to inner products
    against them (``<Q_j, P wp> = <Q_j, wp>``) — orthogonality is untouched. From the
    freeze on, the recurrence is an *exact* block Lanczos of the Hermitian projected
    operator ``P H P``: the continued fraction stays causal, moments up to the freeze
    are exact w.r.t. ``H``, and the recurrence terminates as ``invariant_subspace``
    (already treated as exact-on-subspace). All reort modes remain valid, and the
    Krylov store's row set is bounded by the retained set (it never needs removal).

    Importance pruning no longer lives here: it runs on the discovered-key matrix
    (:mod:`impurityModel.ed.gf_recurrence`), where a row the cut defers is not banned.

    MPI: one scalar allreduce per pre-freeze step; the freeze decision and the
    bisection derive only from allreduce'd data, so ranks cannot disagree, and every
    collective runs unconditionally (a rank may retain zero rows).

    The RIXS BiCGSTAB solves reuse this proxy
    unchanged in spirit: ``block_bicgstab``'s matvec routes through
    ``redistribute_block`` (also in serial runs, keyed on ``caps_growth``), so the same
    freeze-growth policy bounds a linear solve's live support, and post-freeze the solve
    is an exact BiCGSTAB of the projected operator ``P H P`` -- the same
    exact-on-retained-subspace contract as the capped Lanczos recurrence. The extra
    forwarders below (``add_states``, ``contains_local``, the restriction properties)
    are the attributes ``block_bicgstab`` reads off its basis.
    """

    caps_growth = True
    #: Capability flags, read instead of the class (``doc/gf_support_pipeline.md``): this basis counts the
    #: support (``retained_size``, ``cap_hit``), and whether it admits by importance and bans the rest.
    tracks_support = True
    #: Ask ``block_lanczos`` to stop with status ``"frozen"`` at the end of the step that
    #: froze the support (see :func:`gf_solvers.block_Green_sparse`'s frozen-basis CSR
    #: fallback). Off by default: every other user of the proxy runs on through the freeze.
    stop_on_freeze = False
    #: A :class:`KeyFilter` of the frozen retained set (built by :meth:`begin_step`), or None; the matvec
    #: applies through it so rows the projection would drop are never generated.
    apply_filter = None

    def __init__(self, basis, cap, memory_budget=None, memory_policy="tighten"):
        """``memory_budget`` (absolute per-rank bytes, ``None`` = off, the default) adds a measured
        guard: before admitting a step's new rows, the color's MAX resident set is compared with
        it, and at or over it the support freezes where it stands (``memory_policy="tighten"``,
        for an auto cap; ``memory_frozen`` records it) or a warning is printed once
        (``"warn"``, for a cap the user set, which is final). Must be replicated across the
        communicator; it is computed once per color by ``gf_units.run_units_distributed``,
        never probed here."""
        self._basis = basis
        self.comm = basis.comm
        # The block-agnostic state and decisions (cap, counts, memory guard, log phrases).
        self.decisions = SupportDecisions(
            basis.comm, cap, int(basis.size), memory_budget=memory_budget, memory_policy=memory_policy
        )
        # Width-0 key-only mask of the retained determinants on this rank; grown by
        # in-place C++ sorted merges only (no per-row Python objects in the hot path).
        # Straight to the width-0 key mask. The previous spelling went via a
        # `dict.fromkeys` and an intermediate width-1 block, which cost a throwaway
        # Python dict and a full block of amplitudes that were never read -- and needed
        # an explicit width=1 so that a rank owning zero determinants did not fall into
        # the width-0 polymorphic zero and raise in `from_states`. `from_keys` returns a
        # width-0 block for an empty input, which is what the mask is anyway, so the
        # empty-rank case stops being a special case.
        self._mask = ManyBodyState.from_keys(basis.local_basis)
        self._verbose_freeze_logged = False
        self._deferred = False

    # --- the decisions' state under the names the callers use
    @property
    def cap(self):
        return self.decisions.cap

    @cap.setter
    def cap(self, value):
        self.decisions.cap = value

    @property
    def memory_budget(self):
        return self.decisions.memory_budget

    @memory_budget.setter
    def memory_budget(self, value):
        self.decisions.memory_budget = value

    @property
    def memory_policy(self):
        return self.decisions.memory_policy

    @memory_policy.setter
    def memory_policy(self, value):
        self.decisions.memory_policy = value

    @property
    def memory_frozen(self):
        return self.decisions.memory_frozen

    @memory_frozen.setter
    def memory_frozen(self, value):
        self.decisions.memory_frozen = value

    @property
    def cap_hit(self):
        return self.decisions.cap_hit

    @cap_hit.setter
    def cap_hit(self, value):
        self.decisions.cap_hit = value

    @property
    def _frozen(self):
        return self.decisions.frozen

    @_frozen.setter
    def _frozen(self, value):
        self.decisions.frozen = value

    @property
    def _global_count(self):
        return self.decisions.global_count

    @_global_count.setter
    def _global_count(self, value):
        self.decisions.global_count = value

    @property
    def _memory_warned(self):
        return self.decisions.memory_warned

    @_memory_warned.setter
    def _memory_warned(self, value):
        self.decisions.memory_warned = value

    # --- attributes block_lanczos_cy reads off its basis ---------------------
    @property
    def local_basis(self):
        return self._basis.local_basis

    @property
    def size(self):
        return self._basis.size

    @property
    def n_bytes(self):
        return self._basis.n_bytes

    @property
    def is_distributed(self):
        return self._basis.is_distributed

    @property
    def restrictions(self):
        return self._basis.restrictions

    @property
    def weighted_restrictions(self):
        return self._basis.weighted_restrictions

    def redistribute_psis(self, *blocks):
        return self._basis.redistribute_psis(*blocks)

    def add_states(self, new_states, unique_sorted=False):
        # Growth bookkeeping only: every determinant block_bicgstab offers here came off a
        # redistribute_block-capped block, so it is already inside the retained mask and
        # counted by _global_count -- the wrapped basis can never outgrow the cap through
        # this path.
        return self._basis.add_states(new_states, unique_sorted=unique_sorted)

    def contains_local(self, state):
        return self._basis.contains_local(state)

    @property
    def frozen(self):
        """Whether the retained set is final. Derived only from allreduced data, so replicated."""
        return self._frozen

    @property
    def retained_size(self):
        """Global number of determinants currently admitted to the recurrence."""
        return self._global_count

    @property
    def retained_mask(self):
        """Width-0 key-only block of this rank's retained determinants (read-only by contract)."""
        return self._mask

    def retained_keys(self):
        """Rank-local retained determinants as ``SlaterDeterminant`` wrappers (sorted).

        Builds one Python object per retained determinant — diagnostics/tests only,
        never the hot path."""
        keys, _ = self._mask.row_max_norms2()
        return keys

    def _allreduce_sum(self, value):
        return self.decisions.allreduce_sum(value)

    def _over_memory_budget(self):
        """Collective (on the color's comm) when the guard is on: is the color's MAX RSS at budget?
        See :meth:`SupportDecisions.over_memory_budget`."""
        return self.decisions.over_memory_budget()

    def redistribute_block(self, block):
        return self._admit(self._basis.redistribute_block(block))

    # --- the policy's three entry points (doc/gf_support_pipeline.md) ---------------------------------
    def admit_seed(self, block):
        """A seed block, outside any step: redistribute and admit it (no cut, and it is not a step of the
        support-freeze window). Collective."""
        return self.redistribute_block(block)

    def transport(self, raw):
        """One chunk of a step's matvec output: redistribute it, with only the work that commutes with the
        sum over chunks (a frozen support's projection, the memory guard's check). Between :meth:`begin_step`
        and :meth:`decide` the admission itself is deferred. Collective."""
        return self.redistribute_block(raw)

    def decide(self, wp, cutoff):
        """The step's whole matvec output, summed over ranks and chunks: cut ``cutoff`` on the summed rows,
        then admit (or project, if frozen). The one
        decision point of a recurrence step. Collective."""
        wp.prune_rows(cutoff)
        return self.finish_step(wp)

    def begin_step(self):
        """A recurrence step starts: its matvec arrives in chunks, admission waits for the whole.

        With ``APPLY_ROW_CHUNKS`` > 1 the step's output reaches :meth:`redistribute_block` one
        chunk at a time, and each chunk holds *partial* amplitudes of rows that other chunks also
        reach. Deciding admission per chunk ranked rows on those partials, in an order set by how
        the rows were split over ranks -- so a capped unit froze on a different set in every colour
        layout (CrI3, cap 20,000: same size, different determinants, G apart by 5e-3 against a 1e-4
        tolerance; one chunk: identical sets, 6e-5). Between ``begin_step`` and :meth:`finish_step`
        the chunks pass through unprojected and the decision is taken once, on the summed output.
        Called by the block-Lanczos step on every rank of the colour (replicated)."""
        self._deferred = not self._frozen
        if self._frozen and self.apply_filter is None:
            # The retained set no longer changes (every admission above is pre-freeze), so the filter of
            # it is built once. `_frozen` is replicated, so every rank enters this collective together.
            self.apply_filter = comm_support_filter(
                [self._mask], self._basis.comm if self._basis.is_distributed else None
            )

    def finish_step(self, wp):
        """The step's whole matvec output: admit on full amplitudes (or project, if frozen). Collective."""
        if not self._deferred:
            return wp
        self._deferred = False
        if self._frozen:
            # The memory guard froze mid-step: what earlier chunks brought in was never admitted.
            wp.keep_rows(self._mask)
            return wp
        return self._admit_counted(wp)

    def _admit(self, block):
        """Project one redistributed matvec output onto the retained set, growing it while it may.

        Split out of :meth:`redistribute_block` so a subclass can change *which* new rows are
        admitted without re-running the (collective) redistribution. Inside a step
        (:meth:`begin_step`) a chunk is only checked against the memory guard and passed on."""
        if self._frozen:
            block.keep_rows(self._mask)
            return block
        if self._over_memory_budget():
            return self._freeze_on_memory(block)
        if self._deferred:
            return block
        return self._admit_counted(block)

    def _freeze_on_memory(self, block):
        """The memory guard tripped: freeze where the support stands. Nothing new is admitted from here on,
        exactly as a cap hit at the current size (the recurrence continues as the exact PHP on what is
        retained)."""
        self.decisions.freeze_on_memory()
        block.keep_rows(self._mask)
        return block

    def _admit_counted(self, block):
        """Admit ``block``'s new rows if they fit under the cap, else its top rows, and freeze. Collective."""
        n_new = self._allreduce_sum(len(block) - block.count_rows_in(self._mask))
        if self._global_count + n_new <= self.cap:
            self._mask.merge_keys(block)
            self._global_count += n_new
            return block
        self._admit_top_and_freeze(block)
        block.keep_rows(self._mask)
        return block

    def _admit_top_and_freeze(self, block):
        """Admit the ``cap - retained`` most important candidate rows, then freeze.

        Ranked on the rows' max column ``|amp|^2`` by
        :func:`~impurityModel.ed.manybody_basis.collective_top_k_bounds`: whole near-tie groups
        above the boundary, then the boundary group filled in determinant-key order
        (:func:`~impurityModel.ed.manybody_basis.collective_first_keys`). The cap is filled exactly
        and the retained set does not depend on the rank count, colour width or rounding.
        """
        norms2 = block.new_row_max_norms2(self._mask)
        above, boundary, n_fill = self.decisions.rank_boundary(norms2)
        admitted = block.keys_new_above(self._mask, above)
        if n_fill > 0:
            # The boundary group: new rows above `boundary` that are not already admitted (a key-only
            # block has max |amp|^2 = 0 per row, so `> -1` keeps every row not in the mask).
            group = block.keys_new_above(self._mask, boundary).keys_new_above(admitted, -1.0)
            group_keys, _ = group.row_max_norms2()
            chosen = self.decisions.first_keys([bytes(k.to_bytearray()) for k in group_keys], n_fill)
            picked = [k for k in group_keys if bytes(k.to_bytearray()) in chosen]
            admitted = admitted.key_union(ManyBodyState.from_keys(picked))
        self._mask.merge_keys(admitted)
        self.decisions.freeze_at_cap(len(admitted))

    def stop_phrase(self):
        """How the support stopped growing, as the opening of a log line: held or frozen at the cap."""
        return self.decisions.stop_phrase()

    def freeze_message(self):
        """One-line description of the cap state (rank-0 logging)."""
        return self.decisions.freeze_message()


def _allreduced_col_norms2(block, n_cols, comm):
    """Per-column ``|.|^2`` of a distributed block, summed over ``comm``, always length ``n_cols``.

    A rank owning none of the block's rows can hold the width-0 polymorphic zero, whose
    ``col_norm2`` is empty; sizing the buffer by the known column count keeps the Allreduce
    symmetric (the width-0 deadlock class)."""
    out = np.zeros(n_cols, dtype=float)
    local = np.asarray(block.col_norm2(), dtype=float)
    if local.size == n_cols:
        out[:] = local
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, out, op=MPI.SUM)
    return out


#: The pre-merge name; every capped or pruned GF support is one :class:`SupportPolicy`.
_CappedBasisProxy = SupportPolicy


def guarded_proxy(basis, cap):
    """``basis`` capped at ``cap`` and memory-guarded as the GF stage configured it.

    The guard's budget and policy are the ones ``gf_units.run_units_distributed`` puts on the
    split basis (``gf_memory_budget``/``gf_memory_policy``), which every clone carries. Returns a
    :class:`_CappedBasisProxy` when there is a finite cap *or* a budget (an ``unlimited`` GF unit
    keeps its guard, with an effectively infinite count cap), else ``basis`` itself.
    """
    budget = getattr(basis, "gf_memory_budget", None)
    if np.isfinite(cap) or budget is not None:
        return _CappedBasisProxy(
            basis,
            cap if np.isfinite(cap) else 2**62,
            memory_budget=budget,
            memory_policy=getattr(basis, "gf_memory_policy", None) or "tighten",
        )
    return basis


def _trim_blocks(alphas, betas, block_widths):
    r"""Strip the zero padding from block-Lanczos coefficients (shrinking blocks).

    The Lanczos kernels store every block into a fixed ``(P, P)`` pre-allocated
    buffer, zero-padding the inactive rows/columns whenever a block deflates
    (``block_widths[i] < P``).  This returns the true variable-dimension blocks:
    ``alphas[i] -> (w_i, w_i)`` and ``betas[i] -> (w_{i+1}, w_i)`` where
    ``w_i = block_widths[i]`` (the trailing ``betas[-1]`` residual block keeps its
    stored row count — it is the coupling beyond the subspace and is unused by the
    continued fraction).

    Args:
        alphas: Diagonal blocks, ``(k, P, P)`` ndarray (or length-``k`` sequence).
        betas: Off-diagonal blocks, same outer length.
        block_widths: True width ``w_i`` of every block.

    Returns:
        tuple[list, list]: ragged ``(alphas, betas)`` lists of 2D arrays.

    Raises:
        ValueError: if the width table and the coefficient arrays disagree in length.
            The kernels append a width for every stored block, so a mismatch means a
            caller trimmed one and not the other -- which would silently shorten the
            continued fraction (``k = len(widths)``) instead of failing.
    """
    widths = [int(w) for w in block_widths]
    k = len(widths)
    if k != len(alphas) or k != len(betas):
        raise ValueError(
            f"block_widths has {k} entries but alphas/betas have {len(alphas)}/{len(betas)}; "
            "the continued fraction would silently use only the first "
            f"{min(k, len(alphas))} block(s)."
        )
    a = [np.asarray(alphas[i])[: widths[i], : widths[i]] for i in range(k)]
    b = []
    for i in range(k):
        rows = widths[i + 1] if i + 1 < k else np.asarray(betas[i]).shape[0]
        b.append(np.asarray(betas[i])[:rows, : widths[i]])
    return a, b


def _sanitize_continued_fraction(alphas, betas, rank=0):
    r"""Drop a corrupted trailing tail from the block-Lanczos coefficients.

    Defense-in-depth before the continued fraction / self-energy: the Lanczos kernels now
    truncate a diverging recurrence at the source (CholeskyQR2 + the ``BETA_BLOWUP_FACTOR``
    guard), but should a non-finite or runaway block ever reach here it must *not* be fed
    silently into :func:`calc_G` and ``sig_static``.  Scans the (trimmed) blocks and keeps
    only the leading run whose norms stay bounded relative to the healthy part; the trailing
    ``beta`` of the kept run is the (ignored) residual coupling, so dropping the tail is
    consistent with the continued fraction's own convention.

    Returns the (possibly shortened) ``(alphas, betas)`` and warns when a tail is dropped.
    """
    norm_max = 0.0
    keep = len(alphas)
    for i in range(len(alphas)):
        a = np.asarray(alphas[i])
        b = np.asarray(betas[i])
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            keep = i
            break
        a_norm = float(np.linalg.norm(a, 2)) if a.size else 0.0
        b_norm = float(np.linalg.norm(b, 2)) if b.size else 0.0
        if i > 0 and max(a_norm, b_norm) > BETA_BLOWUP_FACTOR * max(norm_max, 1.0):
            keep = i
            break
        norm_max = max(norm_max, a_norm, b_norm)
    if keep < len(alphas):
        if rank == 0:
            print(
                f"warning: discarding {len(alphas) - keep} corrupted block(s) from the "
                f"Green's-function continued fraction before computing the self-energy.",
                flush=True,
            )
        return alphas[:keep], betas[:keep]
    return alphas, betas


def _block_cf_inverse(alphas, betas, omegaP):
    r"""Level-0 inverse resolvent of a block-tridiagonal :math:`T` by continued fraction.

    Builds, for every frequency ``omegaP`` (already shifted, i.e.
    :math:`\omega + i\delta + e`),

    .. math::

        G^{-1}_0(\omega) = \omega I - \alpha_0
            - \beta_0^\dagger \big(\omega I - \alpha_1 - \cdots\big)^{-1} \beta_0 ,

    where ``alphas[i]`` is the ``(n_i, n_i)`` diagonal block and ``betas[i]`` the
    ``(n_{i+1}, n_i)`` sub-diagonal block coupling block ``i`` to ``i+1``.  Block
    dimensions may vary from level to level (shrinking-block deflation) and the
    ``betas`` may be rectangular; the identity at each level is sized from that
    level's diagonal block, so no fixed block dimension is assumed.  The trailing
    ``betas[-1]`` (residual coupling beyond the retained subspace) is ignored.

    Args:
        alphas: Length-``k`` sequence of square diagonal blocks.
        betas: Length-``k`` sequence of sub-diagonal blocks.
        omegaP: ``(n_w,)`` complex frequency mesh (shift already applied).

    Returns:
        numpy.ndarray: ``(n_w, n_0, n_0)`` inverse resolvent at the first block.
    """
    nw = omegaP.shape[0]
    if all(np.shape(alpha) == (1, 1) for alpha in alphas):
        return _scalar_cf_inverse(alphas, betas, omegaP)

    def wI(n):
        return omegaP[:, np.newaxis, np.newaxis] * np.identity(n, dtype=complex)[np.newaxis]

    a_last = np.asarray(alphas[-1])
    G_inv = wI(a_last.shape[0]) - a_last[np.newaxis]
    for alpha_raw, beta_raw in zip(alphas[-2::-1], betas[-2::-1]):
        alpha = np.asarray(alpha_raw)
        beta = np.asarray(beta_raw)
        n_i = alpha.shape[0]
        beta_b = np.broadcast_to(beta, (nw,) + beta.shape)
        G_inv = wI(n_i) - alpha[np.newaxis] - np.conj(beta.T)[np.newaxis] @ np.linalg.solve(G_inv, beta_b)
    return G_inv


def _scalar_cf_inverse(alphas, betas, omegaP):
    """:func:`_block_cf_inverse` when every block is 1 x 1: the same recursion on scalars.

    A width-1 recurrence (every spectra unit, and most rotated self-energy blocks) otherwise
    pays a batched ``(n_w, 1, 1)`` LAPACK solve per level, whose call overhead dwarfs the
    arithmetic: measured 8x (64-point monitor mesh) to 24x (3001-point output mesh) faster at 400
    levels. It evaluates ``w - a - conj(b) * (b / g)`` level by level, as the block form does,
    but agrees with it only to the last ulp, not bitwise: LAPACK's complex division rounds
    differently.
    """
    a = np.fromiter((alpha[0][0] for alpha in alphas), dtype=complex, count=len(alphas))
    b = np.fromiter((beta[0][0] for beta in betas), dtype=complex, count=len(betas))
    g = omegaP - a[-1]
    for a_i, b_i in zip(a[-2::-1], b[-2::-1]):
        g = omegaP - a_i - np.conj(b_i) * (b_i / g)
    return g[:, np.newaxis, np.newaxis]


class _IncrementalBlockCF:
    r"""Seed-block resolvent :math:`G_{00}(z)` of a growing block-tridiagonal matrix, one level per block.

    :func:`_block_cf_inverse` eliminates from the *bottom*, so every new block changes the
    whole recursion and the convergence monitor that called it once per check paid O(k) levels
    per call, O(k^2) per run. This eliminates from the *top* instead: with ``D_0 = z - a_0`` and

    .. math::

        D_j = z - a_j - b_{j-1} D_{j-1}^{-1} b_{j-1}^\dagger, \qquad
        L_j = L_{j-1} b_{j-1}^\dagger D_j^{-1}, \qquad
        R_j = D_j^{-1} b_{j-1} R_{j-1},

    (``L_0 = R_0 = D_0^{-1}``) block ``j`` adds exactly
    :math:`dG_j = L_{j-1} b_{j-1}^\dagger D_j^{-1} b_{j-1} R_{j-1}` to :math:`G_{00}`. It is the same
    arithmetic as :class:`~impurityModel.ed.gf_shift_recycling.ShiftedGalerkin`, and equals
    ``inv(_block_cf_inverse(alphas[:j + 1], betas[:j + 1], z))`` to rounding. ``Im D_j >= |Im z|``
    for a Hermitian ``T``, so every factor is bounded by ``|b| / |Im z|`` and the recursion is stable.

    ``advance`` consumes only the blocks it has not seen, so a monitor that calls it every few
    blocks does O(1) work per block. Blocks may be ragged (``block_widths`` trimming) and the
    ``betas`` rectangular, as in :func:`_block_cf_inverse`; the trailing ``betas[-1]`` is never read.

    Parameters
    ----------
    omegaP : ndarray
        ``(n_w,)`` complex frequencies, shift already applied.
    """

    _DEAD = 1e-120  # |L| or |R| below this contributes nothing; zeroed so it never goes denormal

    def __init__(self, omegaP):
        self.z = np.asarray(omegaP, dtype=complex)
        self.consumed = 0
        self.G = None  # (n_w, n0, n0)
        self.dG = None  # last block's contribution
        self._D_inv = self._L = self._R = None
        self._last_alpha = self._last_beta = None  # copies, to detect a history that was replaced

    def _consistent(self, alphas, betas, widths):
        """Whether the blocks already consumed are still the ones the caller holds."""
        k = self.consumed
        if len(alphas) < k:
            return False
        w = int(widths[k - 1]) if widths is not None else None
        a = np.asarray(alphas[k - 1])
        if w is not None:
            a = a[:w, :w]
        if not np.array_equal(a, self._last_alpha):
            return False
        if k >= 2:
            b = np.asarray(betas[k - 2])
            if widths is not None:
                b = b[: int(widths[k - 1]), : int(widths[k - 2])]
            if not np.array_equal(b, self._last_beta):
                return False
        return True

    def advance(self, alphas, betas, block_widths=None):
        """Consume blocks ``alphas[consumed:]`` and return the running ``G_00`` ``(n_w, n0, n0)``.

        A history that is shorter than, or differs from, the one already consumed restarts the
        recursion from block 0.
        """
        k = len(alphas)
        use_w = block_widths is not None and len(block_widths) == k
        widths = block_widths if use_w else None
        if self.consumed and not self._consistent(alphas, betas, widths):
            self.__init__(self.z)
        for j in range(self.consumed, k):
            a = np.asarray(alphas[j])
            if use_w:
                w = int(widths[j])
                a = a[:w, :w]
            n = a.shape[0]
            if j == 0:
                D = self.z[:, None, None] * np.identity(n, dtype=complex)[None] - a[None]
                D_inv = 1.0 / D if n == 1 else np.linalg.inv(D)
                self._L = self._R = D_inv
                self.G = D_inv.copy()
                self.dG = self.G
                self._D_inv = D_inv
            else:
                b = np.asarray(betas[j - 1])
                if use_w:
                    b = b[:n, : int(widths[j - 1])]
                bH = np.conj(b.T)
                n_prev = b.shape[1]
                if n == 1 and n_prev == 1:
                    D = self.z[:, None, None] - a[None] - (b[0, 0] * bH[0, 0]) * self._D_inv
                    D_inv = 1.0 / D
                else:
                    D = (
                        self.z[:, None, None] * np.identity(n, dtype=complex)[None]
                        - a[None]
                        - b[None] @ self._D_inv @ bH[None]
                    )
                    D_inv = np.linalg.inv(D)
                LbH = self._L @ bH[None]
                bR = b[None] @ self._R
                self.dG = LbH @ D_inv @ bR
                self.G = self.G + self.dG
                self._L = LbH @ D_inv
                self._R = D_inv @ bR
                self._D_inv = D_inv
                dead = (np.abs(self._L).max(axis=(1, 2)) < self._DEAD) | (np.abs(self._R).max(axis=(1, 2)) < self._DEAD)
                if dead.any():
                    self._L[dead] = 0.0
                    self._R[dead] = 0.0
                self._last_beta = np.array(b)
            self._last_alpha = np.array(a)
            self.consumed = j + 1
        return self.G


def calc_G(alphas, betas, r, omega, e, delta):
    r"""Green's function from block-Lanczos continued-fraction coefficients.

    Computes :math:`G(\omega) = r^\dagger (\omega + i\delta + e - T)^{-1} r` where
    ``T`` is the block-tridiagonal matrix with diagonal blocks ``alphas`` and
    sub-diagonal blocks ``betas``.  ``alphas`` / ``betas`` may be either a uniform
    ``(k, p, p)`` ndarray (no deflation) or ragged sequences of variable-dimension
    2D blocks (after :func:`_trim_blocks`); rectangular ``betas`` from shrinking-block
    deflation are handled — no fixed block dimension is assumed.

    Parameters
    ----------
    alphas : ndarray or sequence of ndarray
        Diagonal continued-fraction blocks.
    betas : ndarray or sequence of ndarray
        Off-diagonal continued-fraction blocks (``betas[i]`` couples block ``i`` to
        ``i+1`` with shape ``(n_{i+1}, n_i)``).
    r : ndarray
        ``(n_0, n_ops)`` projection of the seed block onto the first Lanczos block.
    omega : ndarray
        Frequency mesh.
    e : float
        Energy offset.
    delta : float
        Broadening factor.

    Returns
    -------
    G : ndarray
        ``(len(omega), n_ops, n_ops)`` Green's function.
    """
    r = np.asarray(r)
    if len(alphas) == 0 or not np.any(r):
        # A zero seed projection gives G = r^H (...) r = 0 identically; skip the solve,
        # whose tridiagonal may be singular on the mesh (e.g. an all-elastic
        # susceptibility seed projected to zero, evaluated at nu = 0 with delta = 0).
        n_ops = r.shape[-1]
        return np.zeros((len(omega), n_ops, n_ops), dtype=complex)
    omegaP = np.asarray(omega) + 1j * delta + e
    G_inv = _block_cf_inverse(alphas, betas, omegaP)
    r_b = np.broadcast_to(r, (omegaP.shape[0],) + r.shape)
    return np.conj(r.T)[np.newaxis] @ np.linalg.solve(G_inv, r_b)


class FrozenColumnMatvec:
    """``Y = H X`` on a frozen basis's local rows, through the frozen ``P H P`` the array Lanczos kernel
    uses (a :class:`~impurityModel.ed.DistributedSparseMatrix.DistributedSparseMatrix`) and the same
    graph exchange.

    For solvers that are not the Lanczos kernel (the frozen-basis BiCGSTAB of
    ``gf_solvers._FrozenShiftedSystem``). The exchange plan is held only inside :meth:`session` -- one
    solve -- never across calls: its graph communicator lives in ``mpi_comm``'s FIFO cache, which other
    redistributions on the same communicator (the solve's leak check) can evict and ``Free()``
    (``MatvecExchangePlan``'s own rule). Re-planning is three small collectives; the matrix's slots are
    bound once, as the plan's layout is the same every time.
    """

    def __init__(self, op, comm, width):
        self.op = op
        self.n_local = int(op.shape[1])
        self.comm = comm
        self.width = int(width)
        self.plan = None
        if not op.is_distributed:
            op.exchange_plan(self.width)  # serial: binds the identity slots

    @contextmanager
    def session(self):
        """Collective: hold an exchange plan for the matvecs of one solve."""
        if not self.op.is_distributed:
            yield self
            return
        self.plan = self.op.exchange_plan(self.width)
        try:
            yield self
        finally:
            self.plan = None

    def apply(self, X):
        """``H X`` for this rank's ``(n_local, w)`` rows of ``X``; collective (inside :meth:`session`)."""
        if self.op.is_distributed and self.plan is None:
            raise RuntimeError("FrozenColumnMatvec.apply outside a session(): no exchange plan")
        return self.op.apply(X, self.plan)


def block_to_local_dense(basis, block, width):
    """This rank's ``(n_local, width)`` rows of ``block`` in ``basis``'s local order; rows outside the
    basis are dropped. Key merges in C++ (no Python object per determinant): the zero block over the
    basis's sorted keys plus ``block`` restricted to them has exactly the basis's rows, in its order.
    ``block`` must be distributed by ``basis``'s ownership. Rank-local."""
    keys = basis._keys
    n_local = len(keys)
    dense = ManyBodyState.from_keys_and_amps(keys, np.zeros((n_local, width), dtype=complex))
    if block.width == width and len(block):
        inside = block.copy()
        inside.keep_rows(keys)
        dense += inside
    out = np.array(np.asarray(dense), dtype=complex, order="C").reshape(n_local, width)
    return out


def local_dense_to_block(basis, dense):
    """The inverse of :func:`block_to_local_dense`: a ``ManyBodyState`` over ``basis``'s local keys."""
    return ManyBodyState.from_keys_and_amps(basis._keys, np.ascontiguousarray(dense, dtype=complex))


def boundary_residual(A_op, X, basis, n_cols, slaterWeightMin, comm):
    r"""Column norms of ``(1 - P) A X`` above ``slaterWeightMin``: what a frozen solve left out that the
    sparse solver, which admits every row at or above the cutoff, would have taken in.

    ``X`` lives on ``basis`` (= ``P``); one full sparse matvec, summed on the owners before the cut,
    the order every other redistributed matvec cuts in. Collective over ``comm``.

    The norms are broadcast from rank 0, so they are bitwise identical on every rank: callers compare them
    with a tolerance and pick different collectives on the answer (the frozen solve's accept or the sparse
    fallback), and MPI does not promise an Allreduce returns the same bits everywhere."""
    AX = basis.redistribute_block(A_op.apply_block(X, 0.0))
    outside = AX.copy()
    outside.keep_rows(AX.keys_new_above(basis._keys, 0.0))
    if slaterWeightMin > 0:
        outside.prune_rows(slaterWeightMin)
    norms = np.sqrt(_allreduced_col_norms2(outside, n_cols, comm))
    if comm is not None and comm.size > 1:
        comm.Bcast(norms, root=0)
    return norms
