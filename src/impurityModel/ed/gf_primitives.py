"""Bottom-layer Green's-function primitives: QR/state-vector plumbing, the block-tridiagonal
continued fraction, and the ``truncation_threshold``-capping basis proxy.

Split out of :mod:`greens_function` (which re-exports everything here for backwards
compatibility) so the solver-policy code in that module and in :mod:`gf_convergence` /
:mod:`gf_shift_recycling` sits on a small, dependency-free layer: nothing here imports from
either of those two, or from :mod:`greens_function` itself.
"""

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_distributed_vector, build_vector
from impurityModel.ed.BlockLanczosArray import BETA_BLOWUP_FACTOR
from impurityModel.ed.manybody_basis import collective_amplitude_cutoff
from impurityModel.ed.ManyBodyUtils import ManyBodyState, block_add_scaled_cy
from impurityModel.ed.memory_estimate import current_rss_bytes, emit_memory_warning, format_bytes


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


class _CappedBasisProxy:
    """Enforce ``truncation_threshold`` on the sparse-kernel GF recurrence.

    ``block_lanczos_cy``'s matvec discovers new Slater determinants every step, so the
    live block-state support (and, at reort != none, the Krylov store) grows without
    bound — the excited ``Basis`` itself stays frozen and never sees them. This proxy
    wraps that basis and caps the growth at the point where every residual row sits on
    its hash-owner rank: the ``redistribute_block`` call. At ``GF_APPLY_ROW_CHUNKS`` > 1
    (the default, 4; ``_lanczos_step.pxi``'s row-chunked matvec) that is ``n_chunks``
    calls per step, one per chunk of ``q_curr``'s rows, instead of one call on the
    whole matvec residual at once (``GF_APPLY_ROW_CHUNKS=1``) -- each chunk runs the
    freeze/admit decision below on its own candidate rows rather than once on the
    whole step's new rows. The cap itself is unaffected (a chunked step still ends at
    ``retained <= cap``, exactly as an unchunked one does -- see
    ``test_gf_apply_row_chunking.py``), but which specific rows land on the admitted
    side of a freeze that happens to fall mid-step can differ: the importance ranking
    below is collective over one chunk's candidates, not the whole step's, so the
    boundary tie-break is finer-grained than the unchunked path's.

    Policy (freeze-growth + importance-ranked boundary admission):

    * while ``retained + n_new <= cap``: admit every newly discovered determinant;
    * on the single overflow step: rank that step's candidate rows by max column
      ``|amp|^2`` of the residual and admit the top ``cap - retained`` via a
      fixed-iteration distributed amplitude bisection (allreduce'd counts, so the
      cutoff is collective and deterministic), then freeze;
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

    MPI: one scalar allreduce per pre-freeze step; the freeze decision and the
    bisection derive only from allreduce'd data, so ranks cannot disagree, and every
    collective runs unconditionally (a rank may retain zero rows).

    The per-frequency BiCGSTAB driver (:func:`block_Green_bicgstab`) reuses this proxy
    unchanged in spirit: ``block_bicgstab``'s matvec routes through
    ``redistribute_block`` (also in serial runs, keyed on ``caps_growth``), so the same
    freeze-growth policy bounds a linear solve's live support, and post-freeze the solve
    is an exact BiCGSTAB of the projected operator ``P H P`` -- the same
    exact-on-retained-subspace contract as the capped Lanczos recurrence. The extra
    forwarders below (``add_states``, ``contains_local``, the restriction properties)
    are the attributes ``block_bicgstab`` reads off its basis.
    """

    caps_growth = True

    def __init__(self, basis, cap, memory_budget=None, memory_policy="tighten"):
        """``memory_budget`` (absolute per-rank bytes, ``None`` = off, the default) adds a measured
        guard: before admitting a step's new rows, the color's MAX resident set is compared with
        it, and at or over it the support freezes where it stands (``memory_policy="tighten"``,
        for an auto cap; ``memory_frozen`` records it) or a warning is printed once
        (``"warn"``, for a cap the user set, which is final). Must be replicated across the
        communicator; it is computed once per color by ``gf_units.run_units_distributed``,
        never probed here."""
        self._basis = basis
        self.cap = int(cap)
        self.memory_budget = memory_budget
        self.memory_policy = memory_policy
        self.memory_frozen = False
        self._memory_warned = False
        self.comm = basis.comm
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
        self._global_count = int(basis.size)
        self._frozen = self._global_count >= self.cap
        self.cap_hit = self._frozen
        self._verbose_freeze_logged = False

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
        if self.comm is None or self.comm.size == 1:
            return value
        return self.comm.allreduce(value, op=MPI.SUM)

    def _over_memory_budget(self):
        """Collective (on the color's comm) when the guard is on: is the color's MAX RSS at budget?

        Called only on the pre-freeze path, right beside that path's own admission-count
        allreduce, so every rank of the color reaches it equally often. The answer is replicated.
        """
        if self.memory_budget is None or self._memory_warned:
            return False
        rss = current_rss_bytes()
        if self.comm is not None and self.comm.size > 1:
            rss = self.comm.allreduce(rss, op=MPI.MAX)
        if rss < self.memory_budget:
            return False
        root = self.comm is None or self.comm.rank == 0
        if self.memory_policy == "tighten":
            cap_text = f"GF cap {self.cap:,}" if self.cap < 2**62 else "no GF cap"
            emit_memory_warning(
                f"WARNING determinant cap: a Green's-function unit stopped growing at {self._global_count:,} "
                f"determinants ({cap_text}): measured {format_bytes(rss)}/rank reached the "
                f"{format_bytes(self.memory_budget)} memory budget.",
                root=root,
                kind="gf-memory",
            )
            return True
        self._memory_warned = True
        emit_memory_warning(
            f"WARNING determinant cap: a Green's-function unit at {self._global_count:,} determinants uses "
            f"{format_bytes(rss)}/rank >= the {format_bytes(self.memory_budget)} memory budget. Your "
            f"truncation_threshold={self.cap:,} is kept as set; if the job is killed, lower it, use 'auto', or "
            "run fewer ranks per node.",
            root=root,
            kind="gf-memory",
        )
        return False

    def redistribute_block(self, block):
        return self._admit(self._basis.redistribute_block(block))

    def _admit(self, block):
        """Project one redistributed matvec output onto the retained set, growing it while it may.

        Split out of :meth:`redistribute_block` so a subclass can change *which* new rows are
        admitted without re-running the (collective) redistribution."""
        if self._frozen:
            block.keep_rows(self._mask)
            return block
        if self._over_memory_budget():
            # Freeze where it stands: nothing new is admitted from here on, exactly as a cap hit
            # at the current size (the recurrence continues as the exact PHP on what is retained).
            self._frozen = True
            self.cap_hit = True
            self.memory_frozen = True
            block.keep_rows(self._mask)
            return block
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

        The amplitude-cutoff bisection runs a fixed iteration count on allreduce'd
        counts, so all ranks compute the identical cutoff. Ties at the cutoff are
        under-admitted (the cap is never exceeded); near-tie retained sets may differ
        across rank counts through summation-order rounding, like the CIPSI basis
        trajectory.
        """
        slots = self.cap - self._global_count
        norms2 = block.new_row_max_norms2(self._mask)
        cutoff2 = collective_amplitude_cutoff(norms2, slots, self.comm)
        admitted = block.keys_new_above(self._mask, cutoff2)
        self._global_count += self._allreduce_sum(len(admitted))
        self._mask.merge_keys(admitted)
        self._frozen = True
        self.cap_hit = True

    def freeze_message(self):
        """One-line description of the cap state (rank-0 logging)."""
        why = "the measured-memory guard" if self.memory_frozen else f"the cap of {self.cap:,}"
        return (
            f"GF basis frozen at {self._global_count:,} determinants by {why}; the Green's "
            "function is exact on the retained subspace."
        )


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


def residual_blocks(A_op, X, Y, basis, mask, n_cols):
    r"""The true residual ``R = Y - A X`` of a projected solve, as two blocks split at ``P``.

    ``X`` solved ``P A P X = Y`` (``supp Y`` inside ``P``) on a basis that may have frozen, so
    the solver's own residual measures only the in-``P`` part. One full matvec at cutoff 0
    recovers both parts:

    * ``inside`` -- the rows of ``R`` inside ``P``: the solver residual, recomputed rather than
      trusted (``block_bicgstab``'s is a recursively-updated estimate);
    * ``outside`` -- the rows outside ``P``, which equal ``(1-P) H X`` because ``Y`` and ``z X``
      both live inside ``P``: the *boundary residual*, zero exactly when nothing was truncated.

    ``A_op`` must be the operator the solve ran with, restrictions included (the solver set
    them from the basis), so that ``outside`` holds only determinants the restricted model can
    reach. ``basis`` must be the **raw** ``Basis``: its ``redistribute_block`` sums every
    rank's contribution to a determinant onto its owner, which a capped proxy would follow by
    ``keep_rows``-ing the boundary away. ``mask`` is the width-0 block of this rank's retained
    determinants. Both blocks are distributed per ``basis``.
    """
    AX = basis.redistribute_block(A_op.apply_block(X, 0.0))
    R = block_add_scaled_cy(Y, AX, -np.eye(n_cols, dtype=complex))
    outside = R.copy()
    outside.keep_rows(R.keys_new_above(mask, 0.0))
    inside = R
    inside.keep_rows(mask)
    return inside, outside


def residual_split(A_op, X, Y, basis, mask, n_cols, comm):
    r"""Column norms ``(||r_P,j||, ||b_j||)`` of :func:`residual_blocks`, summed over ``comm``.

    Collective over ``comm``; always length-``n_cols`` arrays.
    """
    inside, outside = residual_blocks(A_op, X, Y, basis, mask, n_cols)
    return (
        np.sqrt(_allreduced_col_norms2(inside, n_cols, comm)),
        np.sqrt(_allreduced_col_norms2(outside, n_cols, comm)),
    )


def real_up_to_phase(block, n_cols, comm, rtol=1e-12):
    r"""Per column: is it a real vector times one global phase?

    ``|sum_D s_D^2| <= sum_D |s_D|^2`` with equality exactly when every amplitude shares one
    phase up to sign, so the test is two allreduced column sums -- no gather, no pivot row.
    """
    sums = np.zeros((2, n_cols), dtype=complex)
    amps = np.asarray(block)
    if amps.ndim == 2 and amps.shape[1] == n_cols and amps.shape[0] > 0:
        sums[0] = np.sum(amps * amps, axis=0)
        sums[1] = np.sum(np.abs(amps) ** 2, axis=0)
    del amps  # release the buffer view before anything mutates the block
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, sums, op=MPI.SUM)
    n2 = np.real(sums[1])
    return np.abs(sums[0]) >= (1.0 - rtol) * n2


def resolvent_error_bound(s_norm, r_p, b, imag_z, symmetric):
    r"""Elementwise bound on ``|G_exact - G|`` for ``G_ij = <s_i|X_j>``.

    With ``A = z - H`` (``H`` Hermitian), the full residual ``R_j = s_j - A X_j = r_P,j + b_j``
    (``r_P`` inside the solve basis ``P``, ``b`` outside) and ``||A^{-1}|| <= 1/|Im z|``:

    * always, first order: ``dG_ij = <s_i|A^{-1} R_j>``, so
      ``|dG_ij| <= ||s_i|| ||R_j|| / |Im z|``, with ``||R_j||^2 = ||r_P,j||^2 + ||b_j||^2``;
    * second order, for a column ``i`` whose adjoint solve on ``P`` is known. Let
      ``Y_i = (P A^dagger P)^{-1} s_i`` and ``b~_i = (1-P) A^dagger Y_i``; then
      ``dG_ij = <Y_i|r_P,j> - <b~_i|A^{-1} R_j>``, so
      ``|dG_ij| <= ||s_i|| ||r_P,j|| / |Im z| + ||b~_i|| ||R_j|| / |Im z|``. When ``H`` is real
      and ``s_i`` is real up to a phase (``symmetric[i]``), ``Y_i`` is the conjugate of the
      forward solution on ``P`` and ``||b~_i|| = ||b_i||``, which is what is used. That equality
      is exact for the exact ``P`` solve; with the solver's own ``r_P`` it carries a correction of
      order ``||(1-P) H P|| ||r_P|| / |Im z|``, negligible for a converged solve and the reason
      this bound is stated for converged solves.

    Returns the elementwise minimum of the applicable bounds. ``symmetric`` is a per-column
    boolean array (only the bra column ``i`` matters).
    """
    R = np.sqrt(r_p**2 + b**2)
    first = np.outer(s_norm, R) / imag_z
    second = (np.outer(s_norm, r_p) + np.outer(b, R)) / imag_z
    return np.where(np.asarray(symmetric, dtype=bool)[:, None], np.minimum(first, second), first)


class _PrunedBasisProxy(_CappedBasisProxy):
    r"""Importance-pruned growth for the Lanczos recurrence: the comparator of ``outer`` admission.

    Every step's new rows are admitted only if their largest column amplitude exceeds ``eta``;
    the rest are **banned for good**. The ban is what keeps the recurrence exact: a row that is
    outside ``P_k`` when ``H q_k`` is formed and is not admitted then must never enter later, or
    ``q_{k+1}`` was built with ``P_{k+1} H q_k`` while the final retained set ``P_m`` would
    contain the row -- and the recurrence would be the Lanczos of no single operator. With the
    ban, ``P_m H q_k = P_{k+1} H q_k`` for every ``k``, so the recurrence is the exact Lanczos of
    ``P_m H P_m`` under every reorthogonalization mode (the same argument as the freeze, applied
    row by row rather than all at once).

    Three conditions make that argument hold, and the first two are checked by the caller
    (``block_Green_sparse``) because they are properties of the apply, not of the proxy:

    * the apply runs at cutoff 0 -- a row dropped inside the apply is invisible here and cannot
      be banned;
    * the matvec is not row-chunked -- chunks carry *partial* amplitudes, so a row banned on one
      chunk's partial sum could clear the threshold on the full sum;
    * the first matvec that reaches new determinants admits all of them (``eta`` is not applied
      to ``H q_0``): the seeds' first H-shell is what keeps the moments of G through ``H^2``,
      hence the ``Sigma`` tail, exact. ``first_shell_tol`` > 0 relaxes this to an amplitude cut of
      its own, for models whose tiny couplings put a whole hole-space in the first shell. (Not
      simply "the first call": the kernel also routes the seed block itself through here, which
      reaches nothing new.)

    The ban mask grows like the frontier; :attr:`ban_bytes` reports what it holds.
    """

    def __init__(self, basis, cap, eta, first_shell_tol=0.0, **kwargs):
        super().__init__(basis, cap, **kwargs)
        self._eta2 = float(eta) ** 2
        self._shell_tol2 = float(first_shell_tol) ** 2
        self._ban = ManyBodyState.from_keys([])
        self._shell_admitted = False

    @property
    def ban_bytes(self):
        """Rank-local bytes held by the ban mask."""
        return int(self._ban.memory_bytes())

    def _admit(self, block):
        if self._frozen or self._over_memory_budget():
            return super()._admit(block)
        new = block.keys_new_above(self._mask, 0.0)
        # Global, so every rank agrees on whether this is the first-shell step.
        first_shell = not self._shell_admitted and self._allreduce_sum(len(new)) > 0
        self._shell_admitted = self._shell_admitted or first_shell
        candidates = block.keys_new_above(self._mask, self._shell_tol2 if first_shell else self._eta2)
        allowed = [key for key in candidates.keys() if key not in self._ban]
        self._ban.merge_keys(new)  # rows admitted below are in the mask, which takes priority
        # One collective count per call on every rank, whatever this rank's own candidates.
        n_new = self._allreduce_sum(len(allowed))
        allowed_block = ManyBodyState.from_keys(allowed)
        if self._global_count + n_new <= self.cap:
            self._mask.merge_keys(allowed_block)
            self._global_count += n_new
            block.keep_rows(self._mask)
            return block
        # Over the cap: the surviving candidates compete for the remaining slots, then freeze.
        restricted = block.copy()
        restricted.keep_rows(allowed_block)
        self._admit_top_and_freeze(restricted)
        block.keep_rows(self._mask)
        return block


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
