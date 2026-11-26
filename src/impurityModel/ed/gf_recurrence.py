"""The Green's-function block-Lanczos recurrence on a discovered-key matrix.

One step is the textbook block recurrence over tall-skinny blocks (:func:`lanczos_step`)::

    w = A x;  alpha = <x|w>;  w -= x alpha + x_prev beta_prev^H;  x_next, beta = TSQR(w)

with ``A`` a callable, so the same step serves a fixed matrix and one that grows while it is applied.
:func:`block_lanczos_discovered` wraps it for
:class:`~impurityModel.ed.DistributedSparseMatrix.DistributedSparseMatrix`:
the matvec returns the step's rows in ``P`` (the Lanczos step) and in ``F = K \\ P`` (the frontier), the
admission decision reads ``F``, :meth:`DistributedSparseMatrix.grow` admits the chosen rows (one operator
application per admitted determinant), and the admitted rows' amplitude enters the same step. After the
support freezes (cap, memory guard) ``F`` is discarded and the recurrence runs on ``P H P`` from
the same blocks -- the flag is the whole freeze.

**Cut semantics are the array kernel's**: no ``slaterWeightMin`` cut on the ``P`` rows of ``w`` and none on
``q_next``; the cut acts on the ``F`` rows (the rows admission may take). This is what
the frozen restart of the ManyBodyState kernel (the GF's growth path until 2026-10) did; it differed from that
kernel's pre-freeze steps by the cut amounts at ``slaterWeightMin > 0``.

The admission rules (cap with importance-ranked boundary, importance pruning ``eta`` without a ban, the
measured-memory guard) are the :class:`~impurityModel.ed.gf_primitives.SupportDecisions`
applied to dense frontier rows.
"""

import time

import numpy as np
from mpi4py import MPI
from scipy.linalg.blas import zgemm as _zgemm

from impurityModel.ed import config
from impurityModel.ed.basis_transcription import comm_support_filter
from impurityModel.ed.BlockLanczosCore import apply_and_redistribute, divergence_guard
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix, stage
from impurityModel.ed.gf_convergence import (
    _gf_monitor_tol,
    _make_gf_convergence_monitor,
    full_reort_fits,
    krylov_exhausted,
)
from impurityModel.ed.gf_growth_score import PT2_ACC_DTYPE, GrowthScorer, pt2_mesh
from impurityModel.ed.gf_primitives import (
    SupportDecisions,
    _distributed_seed_qr,
    _sanitize_continued_fraction,
    _trim_blocks,
    calc_G,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.memory_estimate import format_bytes, resident_set_bytes
from impurityModel.ed.TSQR import DEFLATE_TOL_SEEDS, spectral_norm, tsqr

#: Fields of :class:`DiscoveredSupport` that live on its :class:`SupportDecisions` (the names callers read).
_FORWARDED = {
    "cap": "cap",
    "memory_budget": "memory_budget",
    "memory_policy": "memory_policy",
    "memory_frozen": "memory_frozen",
    "cap_hit": "cap_hit",
    "frozen": "frozen",
    "closed": "closed",
}


class DiscoveredSupport:
    """The support policy of a run on a discovered-key matrix, under the names the GF callers read.

    Holds the :class:`SupportDecisions` and the matrix; ``retained_*`` are the matrix's live ``P`` columns.
    """

    caps_growth = True
    tracks_support = True
    apply_filter = None
    #: The matvec mode that actually ran, when it is not the one chosen (a retained unit whose ``P H P`` matrix no
    #: longer fit at the freeze continues matrix-free).
    mode_ran = None

    def __init__(self, matrix, decisions, eta=None, first_shell_tol=0.0, pt2=False):
        self.matrix = matrix
        self.decisions = decisions
        self.comm = matrix.comm
        #: The second-order growth rule (``gf_pt2_tol``) decides what is admitted after the first shell: it prunes, and
        #: it replaces the amplitude cut ``eta``. Its :class:`~impurityModel.ed.gf_growth_score.GrowthScorer` lives
        #: here for the growth pass only (``None`` before it and after it).
        self.pt2 = bool(pt2)
        self.scorer = None
        self.prunes = eta is not None or self.pt2
        self._eta = 0.0 if eta is None else float(eta)
        self._eta2 = self._eta**2
        self._shell_tol = float(first_shell_tol)
        self._shell_tol2 = self._shell_tol**2
        self._shell_admitted = False
        self._shell_pending = False
        #: Determinants admitted per step (replicated), the first being the seeds' first H-shell.
        self.step_counts = []
        #: Key-only block of this rank's first-shell rows the cut kept (``None`` before the first shell).
        self.shell_keys = None

    def depth_histogram(self):
        """Global number of determinants admitted at each H-depth from the seeds (step ``k`` reaches depth ``k``);
        a deferred row that a later step admits counts at the step that admitted it."""
        return list(self.step_counts)

    @property
    def shell_mask(self):
        """Width-0 key-only block of this rank's first-shell rows that the cut kept (the guarded shell)."""
        return ManyBodyState.from_keys(self.shell_keys or [])

    @property
    def retained_size(self):
        """Global number of determinants admitted to the recurrence."""
        return self.decisions.global_count

    def retained_keys(self):
        """This rank's admitted determinants (sorted by P slot); diagnostics and tests only."""
        alive = self.matrix.alive()
        return [k for k, a in zip(self.matrix.p_keys(), alive) if a]

    @property
    def retained_mask(self):
        """Width-0 key-only block of this rank's admitted determinants."""
        return ManyBodyState.from_keys(self.retained_keys())

    def stop_phrase(self):
        return self.decisions.stop_phrase()

    def freeze_message(self):
        return self.decisions.freeze_message()


def _forwarded(attr):
    def get(self):
        return getattr(self.decisions, attr)

    def put(self, value):
        setattr(self.decisions, attr, value)

    return property(get, put)


for _name, _attr in _FORWARDED.items():
    setattr(DiscoveredSupport, _name, _forwarded(_attr))


#: Fraction of a rank's whole memory share the discovered-key matrix may reach (a constant, not a user knob). The
#: matrix is resident from the first step and holds the rows outside ``P`` too, so it is priced against a tighter
#: share than the frozen CSR's ``GF_CSR_MEMORY_SAFETY``.
MATRIX_MEMORY_SAFETY = 0.75
#: ``|K_local|`` as a multiple of ``|P_local|`` before the unit has measured it (the NiO estimate: 2.3-2.5).
K_OVER_P_ESTIMATE = 2.5
#: How many times its starting basis a unit's ``P`` is priced at when the cap is larger. Measured |P| / seeds: SMO
#: units 1 and 4 2.2 and 3.3 (cap 4e6, |P| 161k and 409k), SMO cubic heavy units 1.7 (|P| 3.6-4.4M of a 21M cap, which
#: the factor 10 priced at the cap: every one ran matrix-free and refused the second-order rule); a unit that
#: outgrows the price freezes on the memory guard, reported as such.
GROWTH_PRICE_FACTOR = 4
#: Bytes per stored element (a 16 B value and an 8 B packed row) and per K row (key, two slot arrays, the
#: ``K -> P`` map and slack).
_ELEMENT_BYTES = 24
_K_ROW_BYTES = 36


def matrix_need_bytes(n_p_local, n_k_local, fanout, n_global_k, n_ranks, width):
    """Upper bound on the bytes a rank needs for the discovered-key matrix and the recurrence on it: the
    columns of ``P_local`` over the rows of ``K``, the ``K`` store, the dense blocks (``q``, ``q_prev``, the
    next block over ``P`` and the product over ``K``), the exchange buffers, the replicated K -> P table (multi-rank)
    and one ``grow`` batch."""
    from impurityModel.ed.DistributedSparseMatrix import _GROW_BATCH
    from impurityModel.ed.mpi_comm import matvec_column_chunk_width

    elements = n_p_local * fanout
    matrix = elements * _ELEMENT_BYTES + n_k_local * _K_ROW_BYTES
    working = (3 * n_p_local + n_k_local) * width * 16
    if n_ranks > 1:
        send_rows = max(n_global_k - n_k_local, 0)
        recv_rows = n_k_local * (n_ranks - 1)
        w_c = matvec_column_chunk_width(max(send_rows, recv_rows), width, config.GS_MATVEC_EXCHANGE_BYTES.get())
        working += 16 * w_c * (send_rows + recv_rows + n_k_local)
    # The K -> P table of every ``rows="P"`` plan (the frozen phase, a refinement's scoring runs) is gathered whole
    # on every rank (int64 per K row of every rank).
    ptab = 8 * n_global_k if n_ranks > 1 else 0
    return matrix + working + ptab + min(_GROW_BATCH, elements) * 160


def _sample_fanout(basis, hop):
    """Mean full ``H``-image size of a strided sample of this rank's determinants (0 for an empty rank)."""
    from impurityModel.ed.basis_transcription import iter_local_operator_images
    from impurityModel.ed.gf_solvers import _CSR_FANOUT_SAMPLE

    n_basis_local = len(basis.local_basis)
    if not n_basis_local:
        return 0.0
    positions = np.unique(np.linspace(0, n_basis_local - 1, min(_CSR_FANOUT_SAMPLE, n_basis_local)).astype(np.int64))
    sampled = elements = 0
    for image in iter_local_operator_images(basis, hop, 0, indices=positions):
        sampled += 1
        elements += len(image)
    return elements / sampled if sampled else 0.0


#: The routing hash is deliberately skewed: the worst rank holds this multiple of the mean share of ``P``.
REFINE_SKEW = 1.25


def choose_matrix_mode(
    basis, hop, cap, width, memory_budget, report=None, want_discovered=True, refine_shifts=0, pt2=False
):
    """The matvec mode of a unit, decided once and collectively before its first step.

    ``"discovered"`` when ``want_discovered`` (a consumer of the frontier rows: the importance probe, swap
    refinement) and the discovered-key matrix fits ``MATRIX_MEMORY_SAFETY`` of the rank's share on every
    rank, else ``"retained"``: never ``"none"`` from this estimate, which prices at the cap -- the freeze re-prices the
    ``P H P`` matrix against the real ``P`` (``_ApplyEngine._fits_now``) and falls back to matrix-free there. No memory
    budget (the guard is off) always fits. ``P`` is priced at ``min(cap, GROWTH_PRICE_FACTOR * |starting basis|)`` determinants spread over the
    ranks (the starting basis when the cap is unbounded), ``K`` at :data:`K_OVER_P_ESTIMATE` times that. ``report`` (a dict)
    receives the numbers the decision used. ``refine_shifts`` (the number of scoring shifts a refinement will
    accumulate, conjugates included) adds the accumulator's ``2 * shifts * |P_local| * width`` complex entries, times
    :data:`REFINE_SKEW` for the routing hash's imbalance, to the price of the discovered mode: a unit whose matrix
    fits but matrix plus accumulator does not runs the retained mode, without a refinement. ``pt2`` (the second-order
    growth rule asked for; replicated): a matrix that does not fit at that price is priced again at the starting basis
    alone, and runs ``"discovered"`` if it fits there -- the rule needs the frontier rows, and the measured guard then
    freezes the growth where the memory runs out (fewer determinants, the rule kept: ``priced_from`` says so). The
    report's ``rss`` and needs are the worst rank's. **Collective.**
    """
    from impurityModel.ed.gf_solvers import _frozen_matrix_need
    from impurityModel.ed.memory_estimate import resident_set_bytes

    comm = basis.comm if basis.is_distributed else None
    if memory_budget is None:
        return "discovered" if want_discovered else "retained"
    n_ranks = comm.size if comm is not None else 1
    n_basis_local = len(basis.local_basis)
    fanout = _sample_fanout(basis, hop)
    # A cap far above anything the unit will reach says nothing about its size (the harness's 1e9 priced the matrix at
    # 4.7e12 B and silently ran `retained`): price at the cap or at GROWTH_PRICE_FACTOR times the starting basis,
    # whichever is smaller. The growth itself is guarded by the measured memory (``_grow_fits``, the freeze guard).
    priced = min(float(cap), GROWTH_PRICE_FACTOR * float(basis.size)) if np.isfinite(cap) else None
    priced_from = (
        "none" if priced is None else ("the cap" if priced >= float(cap) else f"{GROWTH_PRICE_FACTOR} x the seeds")
    )
    n_p = n_basis_local if priced is None else max(n_basis_local, int(priced / n_ranks))
    n_k = int(K_OVER_P_ESTIMATE * n_p)
    need_e = matrix_need_bytes(n_p, n_k, fanout, n_k * n_ranks, n_ranks, width)
    need_acc = int(REFINE_SKEW * 2 * refine_shifts * n_p * width * 16)
    need_e += need_acc
    need_csr = _frozen_matrix_need(n_p, fanout, n_p * n_ranks, n_ranks, width)
    csr_budget = getattr(basis, "gf_csr_memory_budget", None)
    csr_budget = memory_budget if csr_budget is None else max(memory_budget, csr_budget)
    e_budget = csr_budget * MATRIX_MEMORY_SAFETY / config.GF_CSR_MEMORY_SAFETY.get()
    rss = resident_set_bytes()
    over = np.array([rss + need_e > e_budget, rss + need_csr > csr_budget, rss, need_e, need_csr], dtype=np.int64)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, over, op=MPI.MAX)
    rss, need_e, need_csr = int(over[2]), int(over[3]), int(over[4])  # the worst rank's, for the report
    if pt2 and want_discovered and over[0] and priced is not None and n_p > n_basis_local:
        # The second-order rule: price the matrix at the seeds (``n_basis_local``) before giving the rule up. ``pt2``,
        # ``over`` and ``priced`` are replicated, so every rank joins this reduction or none does.
        n_p, n_k = n_basis_local, int(K_OVER_P_ESTIMATE * n_basis_local)
        need_seed = matrix_need_bytes(n_p, n_k, fanout, n_k * n_ranks, n_ranks, width) + int(
            REFINE_SKEW * 2 * refine_shifts * n_p * width * 16
        )
        again = np.array([resident_set_bytes() + need_seed > e_budget], dtype=np.int64)
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, again, op=MPI.MAX)
        if not again[0]:
            over[0] = 0
            priced_from = "the seeds (the second-order rule; the memory guard bounds the growth)"
            need_e = need_seed
    if report is not None:
        report.update(
            n_p_local=n_p,
            n_p_priced=n_p * n_ranks,
            priced_from=priced_from,
            n_k_local=n_k,
            fanout=fanout,
            need_discovered=need_e,
            need_accumulator=need_acc,
            need_retained=need_csr,
            rss=rss,
            budget_discovered=e_budget,
            budget_retained=csr_budget,
        )
    if want_discovered and not over[0]:
        return "discovered"
    # Never pre-commit to "none" from the cap-priced estimate: "retained" and "none" are the same until the
    # support freezes, and _ApplyEngine._fits_now re-prices the matrix at the freeze against the real P and
    # falls back to "none" there. Priced at the cap, a 30k-determinant restart ran matrix-free (every block
    # re-applied H) because the cap, not P, was over budget. The report keeps ``over`` for the log line.
    if report is not None:
        report["retained_fits_at_cap"] = not bool(over[1])
    return "retained"


def _allreduce(comm, array, op=MPI.SUM):
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, array, op=op)
    return array


def _pad_rows(x, n_rows):
    """``x`` with zero rows appended up to ``n_rows`` (newly admitted P slots come last)."""
    if x.shape[0] == n_rows:
        return x
    out = np.zeros((n_rows, x.shape[1]), dtype=complex)
    out[: x.shape[0]] = x
    return out


def _overlap(x, w):
    """``x^H w`` for C-ordered blocks, ``x`` having at most as many rows as ``w`` (the rest of ``x`` is zero):
    one ``zgemm`` on the transposed views, no conjugated copy of ``x``."""
    p, q = x.shape[1], w.shape[1]
    r = x.shape[0]
    if p == 0 or q == 0 or r == 0:
        return np.zeros((p, q), dtype=complex)
    # With A = x^T and B = w^T (Fortran-ordered views), A B^H = conj(x^H w).
    return np.ascontiguousarray(_zgemm(1.0, x.T, w[:r].T, trans_b=2).conj())


def _subtract_product(w, x, coef):
    """``w[:rows of x] -= x @ coef`` in place (``w`` C-ordered, ``x`` a C-ordered block with at most ``w``'s rows,
    the rows beyond are zero): one ``zgemm`` accumulating into the transposed view of ``w``."""
    r = x.shape[0]
    if r == 0 or x.shape[1] == 0 or w.shape[1] == 0:
        return
    view = w[:r].T  # Fortran-ordered (q, r) view onto w's rows
    out = _zgemm(-1.0, np.asfortranarray(coef.T), x.T, beta=1.0, c=view, overwrite_c=True)
    if not np.shares_memory(out, w):  # not accumulated in place (should not happen for these layouts)
        w[:r] = out.T


def lanczos_step(apply, x, x_prev, beta_prev, *, comm=None, scale=0.0, deflate_tol=-1.0, store=None, diag=None):
    """One block-Lanczos step; collective when ``comm`` is given.

    ``apply(x)`` returns ``w = A x``; it may return **more rows** than ``x`` has (a matrix that admitted
    determinants while being applied appends their slots at the end), in which case ``x`` and ``x_prev`` are
    zero there. ``x_prev`` and ``beta_prev`` are ``None`` on the first step. ``scale`` is the operator-scale
    estimate the TSQR breakdown test is measured against (``max(scale, ||alpha||)`` is used).

    Returns ``(alpha, x_next, beta, k)``: ``alpha`` is ``(p, p)``, ``x_next`` the ``(rows, k)`` orthonormal
    next block and ``beta`` the ``(k, p)`` factor with ``w - ... = x_next @ beta``; ``k`` is the retained
    rank (``0`` breakdown: the Krylov space is closed, ``-1`` a non-finite factor; ``x_next`` and ``beta``
    are then ``None``). ``store`` (a list of the earlier blocks, each with at most ``w``'s rows) switches on
    full reorthogonalization: two classical Gram-Schmidt passes against every stored block. ``diag`` (a dict)
    receives ``"dropped"``, the singular values TSQR deflated at this step (empty when none).
    """
    with stage("apply"):
        w = np.asarray(apply(x), dtype=complex)
        if not w.flags.c_contiguous or not w.flags.writeable:
            w = np.array(w, dtype=complex, order="C")
    with stage("step_dense"):
        alpha = _allreduce(comm, _overlap(x, w))
        _subtract_product(w, x, alpha)
        if x_prev is not None:
            _subtract_product(w, x_prev, beta_prev.conj().T)
    if store is not None:
        for _ in range(2):
            for block in store:
                rows = block.shape[0]
                overlap = _allreduce(comm, np.ascontiguousarray(block.conj().T @ w[:rows]))
                w[:rows] -= block @ overlap
    with stage("tsqr"):
        x_next, beta, k, sv = tsqr(w, comm, max(float(scale), spectral_norm(alpha)), deflate_tol=deflate_tol)
    if diag is not None:
        diag["dropped"] = np.zeros(0) if sv is None or k <= 0 else np.asarray(sv, dtype=float)[k:]
    return alpha, x_next, beta, k


class _Admission:
    """The admission decision of one step on dense frontier rows. Collective; replicated verdicts."""

    def __init__(self, support, slater_weight_min):
        self.support = support
        self.d = support.decisions
        self.swm = float(slater_weight_min)

    def decide(self, f_ids, wf, keys_of):
        """:meth:`_decide`, plus the step's admitted count and (once) the first shell's kept keys."""
        before = self.d.global_count
        admitted = self._decide(f_ids, wf, keys_of)
        sup = self.support
        sup.step_counts.append(int(self.d.global_count - before))
        if sup._shell_pending:
            sup._shell_pending = False
            sup.shell_keys = list(keys_of(admitted))
        return admitted

    def _decide(self, f_ids, wf, keys_of):
        """The frontier rows to admit from the step's matvec output.

        ``f_ids`` identify this rank's frontier rows (K slots, or row numbers of the applied block) and
        ``wf`` their amplitudes ``(n_f, p)`` and ``keys_of(ids)`` the determinants of some ids. Cuts the
        frontier at ``slaterWeightMin``, applies the memory guard, then the counted or pruned rule; at the cap it
        ranks the candidates and freezes. Returns the ids admitted."""
        d = self.d
        sup = self.support
        f_slots = np.asarray(f_ids, dtype=np.int64)
        amp2 = (np.abs(wf) ** 2).max(axis=1) if wf.shape[1] else np.zeros(len(f_slots))
        keep = amp2 > self.swm**2
        slots = f_slots[keep]
        norms2 = amp2[keep]
        if d.over_memory_budget():
            d.freeze_on_memory()
            return slots[:0]
        if sup.prunes:
            return self._pruned(keys_of, slots, norms2)
        n_new = int(d.allreduce_sum(len(slots)))
        if d.global_count + n_new <= d.cap:
            d.global_count += n_new
            return slots
        return self._top_and_freeze(keys_of, slots, norms2)

    def _pruned(self, keys_of, slots, norms2):
        sup, d = self.support, self.d
        first_shell = not sup._shell_admitted and int(d.allreduce_sum(len(slots))) > 0
        sup._shell_admitted = sup._shell_admitted or first_shell
        sup._shell_pending = first_shell
        if sup.pt2 and not first_shell:
            return self._second_order(keys_of, slots)
        sel = norms2 > (sup._shell_tol2 if first_shell else sup._eta2)
        slots, norms2 = slots[sel], norms2[sel]
        n_new = int(d.allreduce_sum(len(slots)))
        if d.global_count + n_new <= d.cap:
            d.global_count += n_new
            return slots
        return self._top_and_freeze(keys_of, slots, norms2)

    def _second_order(self, keys_of, slots):
        """The second-order rule (:mod:`~impurityModel.ed.gf_growth_score`): after the first shell, the frontier rows
        whose score clears one global tail cut of mass ``tau``; nothing on a block that is not on the scoring cadence.
        At the cap the candidates are ranked by score. Every collective is unconditional on the replicated cadence
        (a rank with no frontier rows takes part with empty arrays)."""
        sup, d = self.support, self.d
        sc = sup.scorer
        if not sc.due():
            return slots[:0]
        s, mass = sc.scores(slots, sup.matrix.k_diagonal().real[slots])
        sel = (s > sc.cutoff(s)) & (s > 0.0)
        sc.note_mass(mass)
        slots, s = slots[sel], s[sel]
        n_new = int(d.allreduce_sum(len(slots)))
        if d.global_count + n_new <= d.cap:
            d.global_count += n_new
            sc.note_decision(n_new)
            return slots
        admitted = self._top_and_freeze(keys_of, slots, s)
        sc.note_decision(int(d.allreduce_sum(len(admitted))))  # what the cap let in, not the candidates
        return admitted

    def _top_and_freeze(self, keys_of, slots, norms2):
        """Admit the ``cap - retained`` most important candidates (whole near-tie groups, the boundary group
        in determinant-key order), then freeze at the cap."""
        d = self.d
        above, boundary, n_fill = d.rank_boundary(norms2)
        take = norms2 > above
        if n_fill > 0:
            group = np.flatnonzero(~take & (norms2 > boundary))
            keys = keys_of(slots[group])
            chosen = d.first_keys([bytes(k.to_bytearray()) for k in keys], n_fill)
            for g, key in zip(group, keys):
                if bytes(key.to_bytearray()) in chosen:
                    take[g] = True
        admitted = slots[take]
        d.freeze_at_cap(len(admitted))
        return admitted


class _OutputBuffer:
    """One ``(rows, width)`` complex buffer reused by the frozen matvecs (the step overwrites it in place, so a
    block's rows are never allocated and page-faulted again)."""

    def __init__(self):
        self._buf = None

    def get(self, rows, width):
        if self._buf is None or self._buf.shape != (rows, width):
            self._buf = np.empty((rows, width), dtype=complex)
        return self._buf


class _MatrixEngine:
    """Matvec mode ``"discovered"``: ``w = M q`` over every K row, ``grow`` for what admission takes."""

    def __init__(self, matrix, admission):
        self.matrix = matrix
        self.admission = admission
        self._out = _OutputBuffer()

    def apply(self, block):
        matrix, admission = self.matrix, self.admission
        d = admission.d
        if d.frozen:
            return matrix.matvec(block, rows="P", out=self._out.get(matrix.n_p_local, block.shape[1]))
        w_k = matrix.matvec(block, rows="all")
        if admission.support.scorer is not None:
            admission.support.scorer.stash(w_k)
        f_slots = np.flatnonzero(matrix.k_to_p_slot() < 0)
        if not int(d.allreduce_sum(len(f_slots))):
            # K holds every row H reaches from P (deferred rows stay in it), so no row outside P on any rank is
            # H P inside P: the support is final. Replicated count, so every rank freezes on this step.
            d.freeze_closed()
            w_p = w_k[matrix.p_to_k_slot()]
            w_p[~matrix.alive()] = 0.0
            return w_p
        slots = admission.decide(f_slots, w_k[f_slots], matrix.keys_of)
        # The verdict (how many rows were admitted) is replicated, so every rank takes the same branch.
        if int(d.allreduce_sum(len(slots))):
            matrix.grow(matrix.keys_of(slots))
        w_p = w_k[matrix.p_to_k_slot()]
        w_p[~matrix.alive()] = 0.0
        return w_p


class _FrozenFilter:
    """The slice of the support-policy protocol ``apply_and_redistribute`` reads, for a frozen ``P``: the apply
    is filtered to ``P``, the chunks are redistributed, and nothing is decided (the engine projects)."""

    frozen = True

    def __init__(self, basis, keep):
        self._basis = basis
        self.apply_filter = keep

    def __getattr__(self, name):  # the basis's own redistribute methods (a test basis has only the scalar one)
        return getattr(self._basis, name)

    def begin_step(self):
        pass

    def transport(self, raw):
        return self._basis.redistribute_block(raw)

    def decide(self, w, cutoff):
        return w


class _ApplyEngine:
    """Matvec modes ``"retained"`` and ``"none"``: ``w`` is the operator applied to the block and routed to
    the row owners (the ManyBodyState kernel's matvec), the matrix is only the slot manager of ``P``
    (``keep_frontier=False``, ``admit_keys``: ``K = P``). ``"retained"`` builds the ``P H P`` columns once,
    when the support freezes, and runs the frozen recurrence on them; ``"none"`` applies the operator to
    every block, frozen or not, and projects onto ``P``. The frontier is the applied rows outside ``P``;
    their ids are row numbers of the applied block."""

    def __init__(self, matrix, admission, hop, basis, retained):
        self.matrix = matrix
        self.admission = admission
        self.hop = hop
        self.basis = basis
        self.retained = retained
        self._built = False
        self._width = 1  # raised to the block width by apply(); array_operator may run first (restart)
        self._out = _OutputBuffer()
        self._filter = None
        self._n_chunks = int(config.APPLY_ROW_CHUNKS.get() or 1)

    def _applied(self, x):
        """``H x`` as a ManyBodyState routed to the owners (no cut: the frontier is cut at admission).

        Once the support is frozen the output is read on ``P`` only, so the operator is applied through a filter
        of ``P`` and never generates the other rows (as the ManyBodyState kernel's frozen support does)."""
        matrix = self.matrix
        k_to_p = matrix.k_to_p_slot()
        by_pos = np.zeros((matrix.n_k_local, x.shape[1]), dtype=complex)
        live = k_to_p[matrix._slot_of_pos] >= 0
        by_pos[live] = x[k_to_p[matrix._slot_of_pos[live]]]
        block = ManyBodyState.from_keys_and_amps(matrix._kstore, by_pos)
        redistribute = matrix.comm is not None
        policy = self.basis
        if self.admission.d.frozen and redistribute:
            if self._filter is None:
                # Collective, entered on the same step by every rank: ``frozen`` is replicated.
                self._filter = _FrozenFilter(self.basis, comm_support_filter([matrix._kstore], matrix.comm))
            policy = self._filter
        return apply_and_redistribute(self.hop, block, policy, 0.0, redistribute, self._n_chunks if redistribute else 1)

    def _fits_now(self):
        """Whether the ``P H P`` matrix about to be built fits the budget the mode was priced against, re-measured
        at the freeze: the worst rank's actual ``P``, and the resident set now (a memory-guard freeze arrives with it
        at the guard's budget). Replicated; **collective** on every rank whenever a budget is set."""
        from impurityModel.ed.gf_solvers import _frozen_matrix_need

        matrix, d = self.matrix, self.admission.d
        budget = d.memory_budget
        if budget is None:
            return True
        csr_budget = getattr(self.basis, "gf_csr_memory_budget", None)
        csr_budget = budget if csr_budget is None else max(budget, csr_budget)
        comm = matrix.comm if matrix.comm is not None and matrix.comm.size > 1 else None
        worst = np.array([matrix.n_p_local, _sample_fanout(self.basis, self.hop)], dtype=np.float64)
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, worst, op=MPI.MAX)
        need = _frozen_matrix_need(
            int(worst[0]), float(worst[1]), int(d.global_count), comm.size if comm is not None else 1, self._width
        )
        over = np.array([resident_set_bytes() + need > csr_budget], dtype=np.int64)
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, over, op=MPI.MAX)
        return not over[0]

    def array_operator(self, width=None):
        """``(operator, order)``: the frozen ``P H P`` for the array kernel with its P slots renumbered in key order
        (``order[i]`` is the old slot now numbered ``i``); ``None`` unless ``"retained"``. Builds the columns.
        ``width`` is the block width the matrix will be applied to (``_fits_now`` prices the work arrays at it)."""
        if not self.retained:
            return None
        if width is not None:
            # A restart reaches the freeze at block 0, before any apply() has set the width: price the matrix at
            # the block width it will actually run, not at 1.
            self._width = int(width)
        if not self._fits_now():
            self.retained = False
            self.admission.support.mode_ran = "none"
            return None
        order = self.matrix.sort_p_slots()
        if not self._built:
            self.matrix.build_columns()
            self._built = True
        return self.matrix, order

    def _closed_under_h(self):
        """Whether ``H P`` lies inside ``P``. This mode keeps no frontier (``K = P``), and a block's image says only
        where *its* support goes, so the question is put to ``H`` applied to every row of ``P`` at once, with
        random phases (an exact cancellation is then a measure-zero event). Asked only on a step whose own image
        had no row outside ``P`` on any rank. **Collective**; replicated verdict."""
        matrix = self.matrix
        rng = np.random.default_rng(0x5EED + (matrix.comm.rank if matrix.comm is not None else 0))
        probe = np.exp(2j * np.pi * rng.random((matrix.n_p_local, 1)))
        probe[~matrix.alive()] = 0.0
        applied = self._applied(probe)
        outside = int(np.count_nonzero(applied.find_rows_in(matrix._kstore) >= matrix.n_k_local))
        return not int(self.admission.d.allreduce_sum(outside))

    def apply(self, x):
        matrix, admission = self.matrix, self.admission
        d = admission.d
        self._width = x.shape[1]
        if d.frozen and self.retained and not self._built and not self._fits_now():
            # The reort-on path never reaches array_operator, so the freeze re-pricing happens here (collective:
            # every rank gets here on the same step, ``frozen`` being replicated).
            self.retained = False
            admission.support.mode_ran = "none"
        if d.frozen and self.retained:
            if not self._built:
                matrix.build_columns()
                self._built = True
            return matrix.matvec(x, rows="P", out=self._out.get(matrix.n_p_local, x.shape[1]))
        with stage("grow_apply"):
            applied = self._applied(x)
        with stage("grow_keys"):
            amps = np.asarray(applied)
            pos = applied.find_rows_in(matrix._kstore)
        in_p = pos < matrix.n_k_local
        k_to_p = matrix.k_to_p_slot()
        w_p = np.zeros((matrix.n_p_local, x.shape[1]), dtype=complex)
        in_rows = np.flatnonzero(in_p)
        p_slot = k_to_p[matrix._slot_of_pos[pos[in_rows]]]
        live = p_slot >= 0
        w_p[p_slot[live]] = amps[in_rows[live]]
        if d.frozen:
            return w_p
        f_ids = np.flatnonzero(~in_p)
        if not int(d.allreduce_sum(len(f_ids))) and self._closed_under_h():
            d.freeze_closed()
            return w_p
        with stage("decide"):
            ids = admission.decide(f_ids, amps[f_ids], lambda sel: [applied.key_at(int(i)) for i in sel])
        if int(d.allreduce_sum(len(ids))):
            new = matrix.admit_keys([applied.key_at(int(i)) for i in ids])
            w_new = np.zeros((matrix.n_p_local, x.shape[1]), dtype=complex)
            w_new[: w_p.shape[0]] = w_p
            new_keys = matrix.keys_of(matrix.p_to_k_slot()[new])
            rows = applied.find_rows(new_keys)
            w_new[new] = amps[rows]
            w_p = w_new
        return w_p


def _pt2_accumulator_fits(support, matrix, width):
    """How the second-order accumulator over the **measured** ``K`` fits the memory budget: ``(verdict, need_bytes,
    rss_bytes, budget_bytes)``, ``verdict`` one of ``"full"`` (the whole mesh), ``"thin"`` (every other point,
    :meth:`GrowthScorer.thin`) or ``"decline"`` (the amplitude rule takes over). **Collective** whenever a budget is
    set; replicated verdict, and the bytes are the worst rank's. Called once, at the scorer's first fold, where the
    first shell is in and ``matrix.n_k_local`` is real: the accumulator holds ``(|K|, n_z, p)`` and ``(|K|, n_z, w)``
    arrays of ``PT2_ACC_DTYPE`` (``width`` is ``p + w``; ``REFINE_SKEW`` is the buffers' own geometric over-allocation). It grows
    with ``K`` after this, which the measured guard (``SupportDecisions.over_memory_budget``) holds.

    The rule gives way in this order, so it runs whenever it can: the whole mesh inside ``MATRIX_MEMORY_SAFETY`` of the
    guard's budget (room left for the growth), else the halved mesh inside it, else the halved mesh inside the whole
    budget (the guard then freezes the growth sooner: fewer determinants, the rule kept), else no rule."""
    budget = support.decisions.memory_budget
    if budget is None:
        return "full", 0, 0, 0
    scorer = support.scorer
    n_full = len(scorer.zs)
    n_thin = len(scorer.kept[::2])
    per_point = REFINE_SKEW * int(width) * matrix.n_k_local * np.dtype(PT2_ACC_DTYPE).itemsize
    need_full, need_thin = int(per_point * n_full), int(per_point * n_thin)
    rss = resident_set_bytes()
    room = budget * MATRIX_MEMORY_SAFETY
    worst = np.array(
        [rss + need_full > room, rss + need_thin > room, rss + need_thin > budget, rss, need_full], dtype=np.int64
    )
    comm = matrix.comm
    if comm is not None and comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, worst, op=MPI.MAX)
    rss, need = int(worst[3]), int(worst[4])
    if not worst[0]:
        return "full", need, rss, int(budget)
    if not worst[2]:
        return "thin", need, rss, int(budget)
    return "decline", need, rss, int(budget)


def _decline_second_order(support, need, rss, budget):
    """The second-order rule does not fit: drop its scorer and carry on in the same matrix mode by the amplitude rule at
    its default (what a unit without the rule runs). Replicated; the caller made the verdict collectively. The reason
    stays on the dropped scorer for the unit's report."""
    from impurityModel.ed.options import DEFAULT_GF_PRUNE_TOL

    why = (
        f"the scoring accumulator over the measured K needs {format_bytes(need)}/rank (half of it on a halved mesh) on "
        f"top of {format_bytes(rss)} resident, budget {format_bytes(budget)}; worst rank"
    )
    support.scorer.declined = why
    support.scorer.release()
    support.scorer = None
    support.pt2 = False
    support._eta = float(DEFAULT_GF_PRUNE_TOL)
    support._eta2 = support._eta**2


def _arm_plateau(converged):
    """Let the convergence monitor ``converged`` stop on a plateau (:data:`gf_convergence._GF_PLATEAU_FACTOR`): only a
    frozen, unreorthogonalized recurrence calls this, the one that can sit on a rounding floor. A stand-in monitor
    without the hook is left alone."""
    arm = getattr(converged, "arm_plateau", None)
    if arm is not None:
        arm()


def _continue_on_array_kernel(operator, comm, converged, alphas, betas, widths, blocks, deflate_tol, stage_name):
    """The recurrence carried on by the array kernel (``BlockLanczosCore.block_lanczos``) over the frozen
    ``P H P`` ``operator``, from the blocks ``[x_prev, x]`` (``x_prev`` ``None`` on the first step; the list is
    emptied) the growth loop stopped at: what the frozen continuation of the ManyBodyState kernel has always
    run. Returns what :func:`block_lanczos_discovered` does."""
    from impurityModel.ed.BlockLanczosCore import block_lanczos
    from impurityModel.ed.gf_solvers import _csr_continue_budget

    start = len(alphas)
    n_dim = operator.shape[0]
    x_prev, x = blocks
    blocks.clear()
    tail = (np.zeros(x.shape, dtype=complex) if x_prev is None else x_prev, x)
    x_prev = x = None
    alphas, betas, widths = np.array(alphas), np.array(betas), list(widths)
    n_before = sum(widths)  # vectors built before the hand-off, on a support that was still growing: not counted

    def blocks_left():
        # The budget is in *vectors* of the space left, spent at the width the recurrence has now: blocks of
        # width p reach n_dim after n_dim / p of them, not n_dim (which ran 5x |P| vectors at width 5).
        rem = max(n_dim - (sum(widths) - n_before), 1)
        return max(1, -(-rem // max(int(widths[len(widths) - 1]) if widths else 1, 1)))

    budget = min(_csr_continue_budget(start), blocks_left())
    while True:
        with stage(stage_name):
            alphas, betas, last_q, widths, status, tail = block_lanczos(
                psi0=None,
                h_op=operator,
                converged=converged,
                reort=False,
                store_krylov=False,
                verbose=False,
                comm=comm,
                alphas_init=alphas,
                betas_init=betas,
                tail=tail,
                block_widths_init=widths,
                max_iter=budget,
                return_widths=True,
                return_status=True,
                return_tail=True,
                deflate_tol=deflate_tol,
            )
        if status != "max_iter":
            break
        if krylov_exhausted(sum(widths) - n_before, n_dim):
            # |P| vectors without the kernel finding an invariant subspace or the monitor's tolerance: at reort
            # off the recurrence sits on its rounding floor and the space is used up. NOT "invariant_subspace":
            # that would claim the continued fraction is exact, which an unreorthogonalized run cannot. ``widths``
            # and ``n_dim`` are replicated, so every rank reaches the same verdict.
            status = "exhausted"
            break
        budget = min(2 * budget, blocks_left())
    return alphas, betas, widths, status, last_q


def block_lanczos_discovered(
    matrix,
    support,
    q0,
    converged,
    *,
    engine=None,
    slater_weight_min=0.0,
    verbose=False,
    deflate_tol=-1.0,
    max_blocks=10**7,
    reort=False,
    shifted=None,
    shifted_rtol=None,
    score_only=False,
    small_exact=False,
    stop_at_freeze=None,
):
    """Block Lanczos of ``H`` on the discovered-key matrix ``matrix`` from the orthonormal seed block ``q0``
    (``(n_p_local, n)``, rows in P-slot order), growing ``P`` through ``support``'s decisions.

    Returns ``(alphas, betas, widths, status, q_last)``: the padded ``(m, n, n)`` coefficient arrays, the
    per-block widths, why it stopped (``"converged"``, ``"quiet"``, ``"closed"``, ``"stalled"``,
    ``"invariant_subspace"``, ``"diverged"``, ``"exhausted"`` or ``"max_iter"``; ``"exhausted"`` = a frozen reort-off
    recurrence has used as many vectors as the support has determinants, so it sits on its rounding floor and cannot
    reach the tolerance; ``"closed"`` = a pruned support that grew closed under ``H``, and ``"stalled"`` = a growing
    support admitted nothing for as many vectors as it has determinants, both leaving for the restart on the final
    ``P``) and this rank's rows of the last accepted block (P-slot order of the final matrix).
    **Collective.** ``converged`` is the GF convergence monitor, called every block on every rank.
    ``shifted`` (a :class:`~impurityModel.ed.gf_shift_recycling.ShiftedGalerkin`) is stepped with every block the
    recurrence accepts (a diverged step is not folded in; an invariant subspace is, as an exact solution) and told
    of every deflation; its blocks grow with the support. With ``shifted_rtol`` the recurrence also runs until
    every shift's residual (the Galerkin residual or the deflation floor, whichever is larger) is below it, so
    "converged" then means the Green's function *and* the scoring solutions are. ``score_only`` stops on that residual
    alone (the Green's-function monitor is still called every block) and reports ``"scored"``: the coefficients are
    then good for the scores, not for ``G``. ``small_exact`` leaves with ``"frozen_small"`` as soon as an
    unreorthogonalized support is frozen at a size :func:`~impurityModel.ed.gf_convergence.full_reort_fits` accepts,
    for the caller's reorthogonalized restart on it.
    ``stop_at_freeze`` (``None``: off; else a replicated, rank-invariant callable) ends the run with the status
    ``"frozen"`` on the first block after the support froze (the cap or the memory guard), provided it returns true:
    the caller restarts from the seeds on the frozen ``P`` anyway, so the frozen continuation of the growth would be
    discarded. A run that starts frozen (a restart, a rerun) never stops this way. The caller **must** run that restart.
    """
    comm = matrix.comm
    admission = _Admission(support, slater_weight_min)
    engine = _MatrixEngine(matrix, admission) if engine is None else engine(matrix, admission)
    d = support.decisions
    n = q0.shape[1]
    alphas = np.zeros((16, n, n), dtype=complex)
    betas = np.zeros((16, n, n), dtype=complex)
    widths = []
    x = np.ascontiguousarray(q0, dtype=complex)
    x_prev = None
    beta_prev = None
    t_norm_max = 0.0
    h_norm_est = 0.0
    status = "max_iter"
    store = [x] if reort else None
    n_vectors = 0  # Krylov vectors produced since the support froze (replicated)
    n_seed = support.retained_size  # the support this recurrence started on (replicated)
    # Krylov vectors since the support last grew, and its size then (replicated: global_count is an allreduced sum).
    n_since_growth, size_at_growth = 0, d.global_count
    frozen_on_entry = bool(d.frozen)

    it = 0
    while it < max_blocks:
        n_curr = x.shape[1]
        if comm is not None:
            probe = np.array([n_curr, -n_curr], dtype=np.int64)
            comm.Allreduce(MPI.IN_PLACE, probe, op=MPI.MAX)
            if probe[0] != -probe[1]:
                raise RuntimeError(f"block width disagrees across ranks at block {it}: [{-probe[1]}, {probe[0]}]")
        if small_exact and d.frozen and not reort and shifted is None and full_reort_fits(d.global_count):
            # Frozen and small: the exact recurrence (full reorthogonalization, from the seeds) is cheaper than an
            # unreorthogonalized one that can only end on its rounding floor. Replicated (frozen, the count).
            status = "frozen_small"
            break
        if stop_at_freeze is not None and d.frozen and not frozen_on_entry and stop_at_freeze():
            # ``d.frozen`` is replicated, so every rank leaves on the same block, before the frozen continuation (and
            # its array-kernel handoff) that the restart would throw away: on LMO unit 5 that continuation was 2,066
            # blocks after a freeze at block 3, repeated by a 2,080-block restart.
            status = "frozen"
            break
        if d.frozen and not reort and hasattr(engine, "array_operator"):
            handed = engine.array_operator(n_curr)
            if handed is not None:
                operator, order = handed
                n_rows = x.shape[0]
                # The blocks travel in a list the callee empties, and this frame lets go of its own: the
                # kernel runs with one copy of each, not two.
                blocks = [None if x_prev is None else _pad_rows(x_prev, n_rows)[order], x[order]]
                x = x_prev = None
                if shifted is None:
                    _arm_plateau(converged)
                return _continue_on_array_kernel(
                    operator, comm, converged, alphas[:it], betas[:it], widths, blocks, deflate_tol, "frozen_kernel"
                )
        diag = {}
        alpha, x_next, beta, k = lanczos_step(
            engine.apply,
            x,
            x_prev,
            beta_prev,
            comm=comm,
            scale=max(h_norm_est, t_norm_max),
            deflate_tol=deflate_tol,
            store=store,
            diag=diag,
        )
        if it >= alphas.shape[0]:
            alphas = np.concatenate((alphas, np.zeros_like(alphas)))
            betas = np.concatenate((betas, np.zeros_like(betas)))
        alphas[it, :n_curr, :n_curr] = alpha
        if k <= 0:
            status = "diverged" if k < 0 else "invariant_subspace"
            widths.append(n_curr)
            it += 1
            if shifted is not None and k == 0:
                shifted.step(x, alpha, beta_prev)
            break
        betas[it, :k, :n_curr] = beta
        if support.scorer is not None:
            if d.frozen:  # the support froze in this step (the cap, the memory guard): nothing is scored any more
                support.scorer.release()
                support.scorer = None
            else:
                if support.scorer.acc is None and support.scorer.has_stash():
                    # First fold: the accumulator is about to be allocated over the measured K. Replicated cadence, so
                    # every rank joins the collective on this block, empty or not.
                    verdict, need, rss, budget_b = _pt2_accumulator_fits(support, matrix, n_curr + q0.shape[1])
                    if verdict == "thin":
                        support.scorer.thin()
                    elif verdict == "decline":
                        _decline_second_order(support, need, rss, budget_b)
                if support.scorer is not None:
                    support.scorer.fold(alpha, beta)
        diverged, t_norm_max, h_norm_est = divergence_guard(
            spectral_norm(beta), spectral_norm(alpha), it == 0, t_norm_max, h_norm_est
        )
        if diverged:
            status = "diverged"
            break
        if shifted is not None:
            shifted.step(x, alpha, beta_prev)
            if diag["dropped"].size:
                shifted.note_deflation(diag["dropped"])
        if d.frozen and not reort and shifted is None:
            _arm_plateau(converged)
        if d.closed and shifted is None and support.prunes and support.retained_size > n_seed:
            # Closed after a pruned growth: nothing can be admitted any more, and the recurrence so far ran on a
            # support that grew under it. Leave for the restart on the final P (as after ``quiet``). Replicated.
            status = "closed"
            widths.append(n_curr)
            it += 1
            break
        if support.scorer is not None and support.scorer.quiet(support.retained_size):
            # The second-order rule admitted nothing for as long as it took to reach the last admission (and at
            # least PT2_QUIET_BLOCKS): the growth is over. Replicated state, so every rank leaves on this block; the
            # restart on the final P is the answer, as after any pruned growth.
            status = "quiet"
            widths.append(n_curr)
            it += 1
            break
        done = converged(alphas[: it + 1], betas[: it + 1], verbose=verbose, block_widths=widths + [n_curr])
        if comm is not None:
            done = comm.bcast(bool(done), root=0)
        if shifted is not None and shifted_rtol is not None and (score_only or (done and support.cap_hit)):
            # The residual is a function of the coefficients, which ranks may disagree on in the last bits (mixed
            # BLAS kernels): decide on the worst rank's value, so every rank leaves on the same block. A deflation
            # floor above the tolerance can never be met: it is reported (`floor_exceeded`) and the stop rule falls
            # back to the Galerkin residual alone, the scores being then as good as the recurrence made them.
            res = float(shifted.residuals(beta).max(initial=0.0))
            floor = float(shifted.floor.max(initial=0.0))
            if floor >= shifted_rtol:
                shifted.floor_exceeded = True
                floor = 0.0
            worst = np.array([max(res, floor)])
            if comm is not None:
                comm.Allreduce(MPI.IN_PLACE, worst, op=MPI.MAX)
            done = bool(worst[0] <= shifted_rtol)
        if done:
            status = "scored" if score_only else "converged"
            widths.append(n_curr)
            it += 1
            break
        if d.frozen:  # counted from the freeze: before it the support was still growing
            n_vectors += n_curr
        elif d.global_count != size_at_growth:
            n_since_growth, size_at_growth = 0, d.global_count
        else:
            n_since_growth += n_curr
            if shifted is None and krylov_exhausted(n_since_growth, d.global_count):
                # A growing support that admitted nothing for |P| vectors: the recurrence has spent the space it has,
                # and a deferred row that has not crossed the cut by now is held back by a change on its rounding
                # floor. The growth is over; the caller restarts on the final P (or, if P never grew, this run was
                # already on it). Replicated: the count and the support size are.
                status = "stalled"
                widths.append(n_curr)
                it += 1
                break
        # One block of lag: the block that fills the space is followed by the genuine breakdown (an invariant subspace,
        # detected at the top of the next iteration), which is the better verdict.
        # A scoring run (``shifted``) stops here too: past |P| vectors an unreorthogonalized recurrence only re-finds
        # ghost copies of what it has, and its Galerkin solutions, like G, are on their rounding floor. Without it a
        # scoring run whose residual could not reach ``shifted_rtol`` ran until the block budget or the divergence
        # guard (the shifted ``exhausted`` flag reports it).
        if d.frozen and not reort and krylov_exhausted(n_vectors - n_curr, d.global_count):
            # Frozen P, no reorthogonalization, |P| vectors spent and the monitor still not satisfied: the run is
            # on its rounding floor. ``n_vectors`` and ``d.global_count`` are replicated (see _continue_on_array_kernel).
            status = "exhausted"
            if shifted is not None:
                shifted.exhausted = True
            widths.append(n_curr)
            it += 1
            break
        x_prev = x
        beta_prev = beta
        x = x_next
        if store is not None:
            store.append(x)
        widths.append(n_curr)
        it += 1
    return alphas[:it], betas[:it], widths, status, x


def start_discovered(
    hop,
    basis,
    psi_dense_local,
    cap,
    *,
    memory_budget=None,
    memory_policy="tighten",
    eta=None,
    first_shell_tol=0.0,
    mode="discovered",
    pt2=False,
):
    """The matrix, the support and the seed block on ``P0 = basis``.

    Admits every determinant of ``basis`` (a ``grow`` of its local keys; ``admit_keys``, with no operator
    application, in the ``"retained"`` and ``"none"`` modes) and returns
    ``(matrix, support, q0)`` with ``q0`` the seed block ``psi_dense_local`` (rows in ``basis.local_basis``
    order) permuted to the matrix's P-slot order. **Collective.**
    """
    comm = basis.comm if basis.is_distributed else None
    cps = (basis.n_bytes + 7) // 8
    matrix = DistributedSparseMatrix(hop, cps, comm, keep_frontier=(mode == "discovered"))
    decisions = SupportDecisions(comm, cap, int(basis.size), memory_budget=memory_budget, memory_policy=memory_policy)
    support = DiscoveredSupport(matrix, decisions, eta=eta, first_shell_tol=first_shell_tol, pt2=pt2)
    local = list(basis.local_basis)
    if mode == "discovered":
        matrix.grow(local)
    else:
        matrix.admit_keys(local)
    p_keys = matrix.p_keys()
    where = basis._keys.find_rows(p_keys)  # local positions of the P-slot keys in basis order
    q0 = np.ascontiguousarray(np.asarray(psi_dense_local, dtype=complex)[where])
    return matrix, support, q0


def _seed_block_on_final_slots(matrix, seed_keys, q0):
    """The seed block ``q0`` (rows in the order of ``seed_keys``) on the matrix's final P slots, zero on the rows the
    growth added. The slots only grew when nothing renumbered them (the keys of the first rows are still the
    seeds'), else the rows follow their determinants."""
    final = matrix.p_keys()
    if final[: len(seed_keys)] == seed_keys:
        return _pad_rows(q0, len(final))
    where = {bytes(k.to_bytearray()): i for i, k in enumerate(seed_keys)}
    out = np.zeros((len(final), q0.shape[1]), dtype=complex)
    for slot, key in enumerate(final):
        i = where.get(bytes(key.to_bytearray()))
        if i is not None:
            out[slot] = q0[i]
    return out


def _refine_line(rec):
    """The one ``-v`` line of a swap refinement (a decline, a fall back, or the rounds it took)."""
    if rec.get("declined"):
        return f"GF refinement declined: {rec['declined']}"
    head = f"GF refinement: {rec['rounds']} round(s), admitted/swapped {rec['swapped']}, remaining gain {rec['gain']}"
    if rec.get("blocks"):
        head += f", blocks per recurrence {rec['blocks']}, deflation floor {max(rec['floor']):.1e}"
    if rec.get("fell_back"):
        return head + " -- final estimate worse than round 0, kept round 0's support"
    tail = "settled" if rec.get("settled") else "not settled"
    if rec.get("rejected") is not None:
        before, after = rec["sigma_error"][-2], rec["sigma_error"][-1]
        tail = (
            f"round {rec['rejected']} rejected, estimated Sigma error per axis "
            f"{[float(f'{e:.2e}') for e in before]} -> {[float(f'{e:.2e}') for e in after]}"
        )
    if "dG_scoring" in rec:
        tail += f", max|dG|/max|G| {rec['dG_scoring']:.2e} on the scoring shifts"
        if "dG_holdout" in rec:
            tail += f", {rec['dG_holdout']:.2e} held out"
    return f"{head} ({tail}) in {rec['seconds']:.1f} s"


def _needs_restart(support, n_seed):
    """Whether a pruned growth ended on a support that is larger than the one the first steps ran on (``n_seed``,
    the seeds'). A test seam: the negative control of ``test_gf_lanczos_pruning`` switches the restart off."""
    return support.prunes and support.retained_size > n_seed


#: Test seam, not a user knob: the block budget of the growth recurrence and of its restart. Every stop the
#: recurrence can take must come from its own rules (the monitor, an invariant subspace, ``quiet``, ``closed``,
#: ``stalled``, ``exhausted``); a test lowers this so a missing rule fails as ``"max_iter"`` instead of hanging.
_GROWTH_MAX_BLOCKS = 10**7


#: Fraction of the final recurrence's blocks the peak-shift check keeps for its "before" value, and the fewest blocks
#: at which dropping a tail says anything (below it the last block is not a tail).
PEAK_SHIFT_TAIL_KEEP = 0.9
PEAK_SHIFT_MIN_BLOCKS = 20


def peak_shift_check(alphas, betas, r, zs, complete=False):
    """How far G still moves, at the Sigma-sensitive shifts, when the end of the recurrence is left out.

    ``zs`` are the unit's shifts from :func:`gf_refinement.g0_zero_shifts` (peaks of ``|G0^-1|``, the Fermi level and
    two Matsubara points), in the frame the coefficients live in. Returns ``{"n_shifts", "n_dropped", "tail_change",
    "complete"}``: ``tail_change`` is the largest *per-shift* relative change ``|dG(z)| / |G(z)|`` when the last
    ``n_dropped`` blocks are left out (per shift, not against the largest ``|G|``: a Matsubara point can be 30x larger
    than the real axis and would hide its motion). ``complete`` marks a Krylov space that is used up, where G is exact
    on the kept determinants and a tail change means nothing. NaN below ``PEAK_SHIFT_MIN_BLOCKS``.

    This is a statement about the recurrence only. It repeats what the convergence monitor checks on its meshes and
    says nothing about whether the determinant set is complete. Never raises: a diagnostic must not fail a unit.
    ``alphas``/``betas`` are the trimmed, sanitised coefficients; rank-local and pure.
    """
    zs = np.asarray(zs, dtype=complex)
    n = len(alphas)
    out = {"n_shifts": int(zs.size), "n_dropped": 0, "tail_change": float("nan"), "complete": bool(complete)}
    if complete or n < PEAK_SHIFT_MIN_BLOCKS or zs.size == 0:
        return out
    keep = max(1, int(PEAK_SHIFT_TAIL_KEEP * n))
    try:
        with np.errstate(all="ignore"):  # a corrupted tail gives NaN here, reported as such, not a warning
            g_all = np.asarray(calc_G(alphas, betas, r, zs, 0.0, 0.0)).reshape(zs.size, -1)
            g_cut = np.asarray(calc_G(alphas[:keep], betas[:keep], r, zs, 0.0, 0.0)).reshape(zs.size, -1)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return out
    scale = np.abs(g_all).max(axis=1)
    change = np.abs(g_all - g_cut).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, change / scale, np.nan)
    out["n_dropped"] = n - keep
    out["tail_change"] = float(np.max(rel)) if np.all(np.isfinite(rel)) else float("nan")
    return out


def block_Green_discovered(
    hOp,
    psi_arr,
    basis,
    delta,
    reort=None,
    slaterWeightMin=0,
    verbose=True,
    cap_info=None,
    eval_meshes=None,
    info=None,
    memory_budget=None,
    memory_policy="tighten",
    matrix_mode="auto",
    prune_tol=None,
    first_shell_tol=None,
    pt2_tol=None,
    refine=None,
):
    """One block of the Green's function from the recurrence on a discovered-key matrix.

    The counterpart of ``gf_solvers.block_Green_sparse`` with the same arguments and return value
    ``(alphas, betas, r)``; ``cap_info`` additionally gets ``engine`` (the matvec mode that ran:
    ``"discovered"``, ``"retained"`` or ``"none"``) and ``matrix`` (the :class:`DistributedSparseMatrix` in the
    ``"discovered"`` mode, else ``None``; never put it in ``cap_stats``). A swap
    refinement (``refine``) reads the discovered frontier, which is what ``"auto"`` needs the
    ``"discovered"`` mode for: without one it takes the cheaper ``"retained"`` mode (the frozen phase then runs on
    the array kernel over ``P H P``). ``matrix_mode`` picks the matvec: the discovered-key matrix, the operator
    applied each step with the ``P H P`` matrix built at the freeze (``"retained"``), or the operator applied
    every block (``"none"``). ``prune_tol`` / ``first_shell_tol`` (``None``: off / whole) are the pruned growth's
    cut and its first-shell cut. ``refine`` (a :class:`~impurityModel.ed.gf_refinement.RefineSpec` with ``rounds > 0``)
    swaps the least important retained determinants of a unit the cap froze for the most important discovered ones
    (``"discovered"`` mode only; the record lands in ``cap_info["refine"]``). The determinant cap is
    ``basis.truncation_threshold``; ``prune_tol > 0`` turns on importance pruning; ``pt2_tol`` (``None``: off) is the second-order rule that
    replaces it (``doc/plans/pt2_growth_rule.md``).
    A frontier row is admitted when one block's amplitude squared clears ``eta^2``.
    """
    pt2_tol = 0.0 if pt2_tol is None else float(pt2_tol)
    if pt2_tol > 0.0:  # arguments only: replicated, and refused before any collective
        if prune_tol is not None and float(prune_tol) > 0.0:
            raise ValueError(f"pt2_tol replaces the amplitude rule prune_tol; got both ({pt2_tol!r}, {prune_tol!r})")
    comm = basis.comm
    rank = comm.rank if comm is not None else 0
    n = len(psi_arr)
    if info is not None:
        info["converged"] = True
        info["d_g"] = float("nan")
        info["n_blocks"] = 0
        info["tol"] = _gf_monitor_tol(slaterWeightMin, eval_meshes)
    if cap_info is not None:
        cap_info["cap_hit"] = False
        cap_info["retained_size"] = None
        cap_info["proxy"] = None
        cap_info["engine"] = "none"
        cap_info["csr_fallback"] = False
        cap_info["csr_seconds"] = 0.0
        cap_info["passes"] = None
        cap_info["matrix"] = None
    if len(basis) == 0 or n == 0:
        if cap_info is not None:
            cap_info["retained_size"] = 0
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), np.zeros((n, n), dtype=complex)
    psi_dense_local, r = _distributed_seed_qr(basis, psi_arr, slaterWeightMin)
    if psi_dense_local.shape[1] == 0:
        if cap_info is not None:
            cap_info["retained_size"] = 0
        return np.empty((0, n, n), dtype=complex), np.empty((0, n, n), dtype=complex), r

    converged, converged_flag, delta_min, last_dg = _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes)
    plateau_cell = [
        getattr(converged, "plateau_hit", [False])
    ]  # the evaluation monitor whose verdict is reported (a restart/rerun replaces it)
    cap = getattr(basis, "truncation_threshold", np.inf)
    cap = cap if np.isfinite(cap) else 2**62
    # Unspecified (``None``) is no pruning and a whole first shell.
    prune_tol = 0.0 if prune_tol is None else float(prune_tol)
    if first_shell_tol is None:
        first_shell_tol = 0.0
    if memory_budget is None:
        memory_budget = getattr(basis, "gf_memory_budget", None)
        memory_policy = getattr(basis, "gf_memory_policy", None) or "tighten"
    resolved_reort = False if reort is None else bool(reort)
    if matrix_mode not in ("auto", "discovered", "retained", "none"):
        raise ValueError(f"matrix_mode must be 'auto', 'discovered', 'retained' or 'none', got {matrix_mode!r}")
    # The second-order rule needs a real-frequency window to score over and the discovered matrix (its frontier rows,
    # ``H_ff`` and ``H X`` over every K row). A unit that cannot have them grows by the amplitude rule at its
    # default, said in one line. Every input is replicated, so every rank takes the same branch.
    pt2_asked = pt2_tol > 0.0
    pt2_zs, pt2_why = (None, None)

    def _fall_back(why):
        nonlocal pt2_zs, pt2_why, prune_tol
        from impurityModel.ed.options import DEFAULT_GF_PRUNE_TOL

        pt2_zs, pt2_why, prune_tol = None, why, DEFAULT_GF_PRUNE_TOL

    if pt2_asked:
        pt2_zs, why = pt2_mesh(eval_meshes)
        if pt2_zs is None:
            _fall_back(why)
        elif matrix_mode in ("retained", "none"):
            _fall_back(f"matrix mode {matrix_mode} was forced")

    def _choose(pt2_on):
        report: dict = {}
        refine_on = refine is not None and refine.rounds > 0
        mode = choose_matrix_mode(
            basis,
            hOp,
            cap,
            len(psi_arr),
            memory_budget,
            report,
            # A pruned unit grows on the discovered-key matrix whether or not it refines: stored growth blocks cost
            # a fraction of matrix-free ones (NiO unit 3 at 6 ranks: 40 s against 287 s of growth).
            pt2_on or prune_tol > 0.0 or (refine is not None and refine.rounds > 0 and bool(refine.shifts)),
            # Upper bound of len(ScoringPlan.accumulator_shifts): need_y is decided later, from the seeds.
            (2 * (len(refine.shifts) + (len(refine.checks) if refine.verify else 0)) if refine_on else 0),
            pt2=pt2_on,
            # The second-order accumulator is not priced here: its size is |K| x the mesh, and |K| is measured after the
            # first shell (``_pt2_accumulator_fits``); a guessed |K| at the cap refused every SMO unit.
        )
        return mode, report

    if matrix_mode == "auto":
        matrix_mode, report = _choose(pt2_zs is not None)
        if pt2_zs is not None and matrix_mode != "discovered":
            _fall_back("the discovered matrix did not fit")
            matrix_mode, report = _choose(False)
        if verbose and rank == 0 and report:
            print(
                f"GF matvec mode {matrix_mode}: {report['n_p_local']:,} local determinants "
                f"(priced at {report['priced_from']}) x fan-out "
                f"{report['fanout']:.1f}, need {format_bytes(report['need_discovered'])} (discovered) / "
                f"{format_bytes(report['need_retained'])} (retained) on top of RSS {format_bytes(report['rss'])}; "
                f"budgets {format_bytes(report['budget_discovered'])} / {format_bytes(report['budget_retained'])}.",
                flush=True,
            )
    if pt2_asked and pt2_zs is None and cap_info is not None:
        cap_info["pt2"] = {"declined": pt2_why}
    if pt2_asked and pt2_zs is None and verbose and rank == 0:
        print(
            f"GF second-order growth rule unavailable ({pt2_why}): this unit grows by the amplitude rule "
            f"gf_prune_tol = {prune_tol:g}",
            flush=True,
        )
    matrix, support, q0 = start_discovered(
        hOp,
        basis,
        psi_dense_local,
        cap,
        memory_budget=memory_budget,
        memory_policy=memory_policy,
        eta=prune_tol if prune_tol > 0.0 else None,
        first_shell_tol=first_shell_tol if prune_tol > 0.0 else 0.0,
        mode=matrix_mode,
        pt2=pt2_zs is not None,
    )
    pt2_scorer = None
    if pt2_zs is not None:
        pt2_scorer = support.scorer = GrowthScorer(pt2_zs, pt2_tol, comm)
        pt2_scorer.n_start = support.retained_size
    engine = (
        None if matrix_mode == "discovered" else lambda m, a: _ApplyEngine(m, a, hOp, basis, matrix_mode == "retained")
    )
    t0 = time.perf_counter()
    n_seed = support.retained_size
    # The seeds' P slots, for a restart on the final P: a pruned unit restarts, and so may a small one (the exact
    # reorthogonalized run), which a seed set above GF_FULL_REORT_MAX can never become.
    seed_keys = matrix.p_keys() if support.prunes or full_reort_fits(n_seed) else None
    refining = refine is not None and refine.rounds > 0 and matrix_mode == "discovered"
    seeds_initial = (matrix.p_keys(), q0.copy()) if refining else None
    operator_real = all(np.imag(amp) == 0 for _term, amp in hOp.items())  # replicated: every rank holds hOp
    plan, acc = None, None

    def _swappable_gate():
        from impurityModel.ed.gf_refinement import anything_swappable

        return anything_swappable(matrix, seeds_initial[0], _seed_block_on_final_slots(matrix, *seeds_initial))

    def _decline_nothing_swappable():
        # Every retained row is a seed or a heavy first-shell row: no score can change P (the frozen cap is final).
        why = "nothing is swappable: the seeds and the first shell fill the cap"
        if cap_info is not None:
            cap_info["refine"] = {"declined": why, "rounds": 0, "swapped": [], "gain": []}
        if verbose and rank == 0:
            print(f"GF refinement declined: {why}", flush=True)

    alphas, betas, widths, status, _q_last = block_lanczos_discovered(
        matrix,
        support,
        q0,
        converged,
        slater_weight_min=slaterWeightMin,
        deflate_tol=DEFLATE_TOL_SEEDS,
        reort=resolved_reort,
        engine=engine,
        max_blocks=_GROWTH_MAX_BLOCKS,
        small_exact=not resolved_reort,
        # The restart below is certain once a pruned growth froze on a support larger than the seeds'
        # (``_needs_restart``): end the growth at the freeze instead of running a frozen continuation the restart
        # discards. G is the restart's either way, bitwise.
        stop_at_freeze=lambda: _needs_restart(support, n_seed),
    )
    growth_status = status
    if status == "stalled" and not _needs_restart(support, n_seed) and not full_reort_fits(support.retained_size):
        # No restart follows: the support has not changed for |P| vectors, so the recurrence has exhausted the final
        # P (a counted unit that grew earlier admitted its rows whole, so its run is still the Lanczos of P H P).
        status = "exhausted"
    if pt2_scorer is not None:
        pt2_scorer.release()  # the accumulator is GBs on a large unit: the restart and the refinement never read it
    support.scorer = None
    # The unit's wall by pass: the first run (growth, and the frozen continuation of a unit the cap
    # froze) and the restart on the final P. Rank-local clocks; the caller reports rank 0's.
    passes = {"growth_blocks": len(widths), "growth_seconds": time.perf_counter() - t0, "growth_stop": growth_status}
    passes.update(restart_blocks=0, restart_seconds=0.0)
    if (
        refine is not None
        and refine.rounds > 0
        and (
            (support.cap_hit and not (refining and not support.memory_frozen))
            # a pruned unit below its cap that runs on another matrix mode: its refinement (grow mode) cannot start
            or (support.prunes and not support.cap_hit and not refining)
        )
    ):
        why = (
            f"matrix mode {matrix_mode} (the discovered matrix and the scoring accumulator did not fit, or the mode "
            "was forced)"
            if matrix_mode != "discovered"
            else (
                "the memory guard froze the unit below the cap"
                if support.memory_frozen
                else "the growth ended below the cap"
            )
        )
        if cap_info is not None:
            cap_info["refine"] = {"declined": f"not attempted: {why}", "rounds": 0, "swapped": [], "gain": []}
        if verbose and rank == 0:
            print(f"GF refinement not attempted: {why}", flush=True)
    # A pruned unit the cap did not freeze has room: its refinement only admits (grow mode); a capped one swaps.
    grow_room = None
    grow_ratio = K_OVER_P_ESTIMATE  # |K|/|P| at the entry to grow mode: global, so the same on every rank
    if refining and support.prunes and not support.pt2 and not support.cap_hit:
        # Discovered rows outside P exist only when |K| > |P|; both sizes are global (one Allreduce, every rank).
        sizes = np.array([matrix.n_k_local, matrix.n_alive_local], dtype=np.int64)
        if comm is not None and comm.size > 1:
            comm.Allreduce(MPI.IN_PLACE, sizes, op=MPI.SUM)
        room = max(cap - int(support.retained_size), 0) if cap < 2**62 else 2**62
        grow_room = int(room) if sizes[0] > sizes[1] and room > 0 else None
        grow_ratio = float(sizes[0]) / max(float(sizes[1]), 1.0)

    if refining and support.pt2 and not support.cap_hit:
        why = "second-order growth rule: refinement acts on cap-frozen units only"
        if cap_info is not None:
            cap_info["refine"] = {"declined": why, "rounds": 0, "swapped": [], "gain": []}
        if verbose and rank == 0:
            print(f"GF refinement declined: {why}", flush=True)

    def _grow_fits(n_add):
        """Whether admitting ``n_add`` more determinants (global) keeps the discovered matrix and the scoring
        accumulator inside the memory budget. Replicated; **collective** whenever a budget is set."""
        budget = support.decisions.memory_budget
        if budget is None:
            return True
        n_p, n_k = matrix.n_alive_local, matrix.n_k_local
        n_ranks = comm.size if comm is not None else 1
        add = int(np.ceil(REFINE_SKEW * n_add / n_ranks))
        # The K rows an admitted determinant brings: the larger of the estimate and this unit's own measured ratio
        # (SMO units 4/5/7: 4.0-4.4 against the 2.5 estimate), measured at the entry to grow mode.
        ratio = max(K_OVER_P_ESTIMATE, grow_ratio)
        fan = _sample_fanout(basis, hOp)
        before = matrix_need_bytes(n_p, n_k, fan, n_k * n_ranks, n_ranks, len(psi_arr))
        after = matrix_need_bytes(
            n_p + add,
            n_k + int(ratio * add),
            fan,
            (n_k + int(ratio * add)) * n_ranks,
            n_ranks,
            len(psi_arr),
        )
        n_shifts = 2 * len(refine.shifts)  # grow mode runs without the check shifts
        # Only the accumulator's growth: the resident one is in the RSS already (it is freed after the admission).
        acc_bytes = int(REFINE_SKEW * n_shifts * add * len(psi_arr) * 16)
        over = np.array(
            [resident_set_bytes() + (after - before) + acc_bytes > budget * MATRIX_MEMORY_SAFETY], dtype=np.int64
        )
        if comm is not None and comm.size > 1:
            comm.Allreduce(MPI.IN_PLACE, over, op=MPI.MAX)
        return not over[0]

    if refining and (support.cap_hit or grow_room is not None) and not support.memory_frozen:
        from impurityModel.ed.gf_refinement import scoring_plan
        from impurityModel.ed.gf_shift_recycling import ShiftedGalerkin

        plan, why = scoring_plan(
            refine,
            _seed_block_on_final_slots(matrix, *seeds_initial),
            matrix,
            operator_real,
            verify=grow_room is None,
        )
        if plan is None:
            if cap_info is not None:
                cap_info["refine"] = {"declined": why, "rounds": 0, "swapped": [], "gain": []}
            if verbose and rank == 0:
                print(f"GF refinement declined: {why}", flush=True)
        elif grow_room is None and not _swappable_gate():
            plan = None
            _decline_nothing_swappable()
        else:
            acc = ShiftedGalerkin(plan.accumulator_shifts, matrix.n_p_local)
    # A small frozen support is solved exactly: a reorthogonalized restart from the seeds, which ends on a genuine
    # invariant subspace within |P| vectors, unless the growth already did (replicated: statuses and counts are). A
    # refining unit's restart (``acc``) too: a scoring run has no ``exhausted`` stop, and unreorthogonalized on a small
    # stiff space it ran to the divergence guard (NiO 2p+3d: 34,350 blocks on 72 determinants).
    exact = (
        not resolved_reort
        and full_reort_fits(support.retained_size)
        and (
            growth_status not in ("converged", "invariant_subspace")
            or _needs_restart(support, n_seed)
            or acc is not None
        )
    )
    if _needs_restart(support, n_seed) or acc is not None or exact:
        # A row the cut deferred can enter a later step, so the recurrence above ran on a support that grew under
        # it. Restart from the seeds on the final P, which is then the exact resolvent of P H P (what the banned
        # rule of the ManyBodyState kernel got by never admitting a row twice). A unit the cap froze has had its P
        # slots renumbered in key order for the handoff, so the seed block follows its keys. A refining unit
        # restarts too: the scoring solutions are accumulated on that run, which is on the final P.
        q0 = _seed_block_on_final_slots(matrix, *(seeds_initial if seeds_initial else (seed_keys, q0)))
        support.decisions.frozen = True
        t_restart = time.perf_counter()
        converged, converged_flag, delta_min, last_dg = _make_gf_convergence_monitor(
            delta, slaterWeightMin, eval_meshes
        )
        plateau_cell[0] = getattr(converged, "plateau_hit", [False])
        alphas, betas, widths, status, _q_last = block_lanczos_discovered(
            matrix,
            support,
            q0,
            converged,
            slater_weight_min=slaterWeightMin,
            deflate_tol=DEFLATE_TOL_SEEDS,
            reort=resolved_reort or exact,
            engine=engine,
            shifted=acc,
            shifted_rtol=None if plan is None else plan.stop_rtol,
            # Reorthogonalized, |P| vectors span the space (width >= 1): two more blocks see the breakdown.
            max_blocks=min(_GROWTH_MAX_BLOCKS, support.retained_size + 2) if exact else _GROWTH_MAX_BLOCKS,
        )
        passes.update(restart_blocks=len(widths), restart_seconds=time.perf_counter() - t_restart)
    if acc is not None:
        from impurityModel.ed.gf_refinement import refine_frozen_unit

        n_first = len(widths)

        def rerun(q0_new, shifted, shifted_rtol, score_only=False):
            nonlocal converged_flag, delta_min, last_dg
            support.decisions.frozen = True
            conv, converged_flag, delta_min, last_dg = _make_gf_convergence_monitor(delta, slaterWeightMin, eval_meshes)
            plateau_cell[0] = getattr(conv, "plateau_hit", [False])
            # A small support is reorthogonalized here too: see ``exact``. Its size is the live count (a grow round
            # admits rows before ``global_count`` is updated, after the refinement); collective, every rank reruns.
            n_live = np.array([matrix.n_alive_local], dtype=np.int64)
            if comm is not None and comm.size > 1:
                comm.Allreduce(MPI.IN_PLACE, n_live, op=MPI.SUM)
            # The run's |P| (its exhausted stop counts against it) is the support it runs on, not the pre-round count.
            support.decisions.global_count = int(n_live[0])
            return block_lanczos_discovered(
                matrix,
                support,
                q0_new,
                conv,
                slater_weight_min=slaterWeightMin,
                deflate_tol=DEFLATE_TOL_SEEDS,
                reort=resolved_reort or full_reort_fits(int(n_live[0])),
                engine=engine,
                shifted=shifted,
                shifted_rtol=shifted_rtol,
                score_only=score_only,
                # A scoring run that cannot reach its residual must not outlive the Green's function by much.
                max_blocks=4 * n_first + 100 if score_only else _GROWTH_MAX_BLOCKS,
            )

        def g_cf(result, zs, framed=True):
            # ``framed=False``: the orthonormal seeds' frame (r -> 1), the one the scores are taken in.
            a_t, b_t = _trim_blocks(result[0], result[1], result[2])
            return calc_G(
                a_t, b_t, r if framed else np.eye(r.shape[0], dtype=r.dtype), np.asarray(zs, dtype=complex), 0.0, 0.0
            )

        monitor_state = (
            converged_flag,
            delta_min,
            last_dg,
            plateau_cell[0],
        )  # round 0's, for a refinement that keeps round 0
        holder = [acc]
        acc = None  # the refinement owns it now (and frees it before it swaps)
        outcome = refine_frozen_unit(
            matrix,
            refine,
            plan,
            seeds_initial[0],
            seeds_initial[1],
            (alphas, betas, widths, status, _q_last),
            holder,
            rerun,
            g_cf,
            eval_meshes,
            _seed_block_on_final_slots,
            room=grow_room,
            fits=_grow_fits,
        )
        if cap_info is not None:
            cap_info["refine"] = outcome.record
        if grow_room is not None:
            # The admitted rows count: the size the unit reports (and a cap it now fills) is the final P's.
            n_final = np.array([matrix.n_alive_local], dtype=np.int64)
            if comm is not None and comm.size > 1:
                comm.Allreduce(MPI.IN_PLACE, n_final, op=MPI.SUM)
            support.decisions.global_count = int(n_final[0])
            # Grow mode that ran out of room or memory is cap- or memory-bound exactly like a growth that froze there
            # (replicated: the record's reason and the count are). The report, the re-solve and the advice read these.
            if outcome.record.get("stopped") == "the next admission would exceed the memory budget":
                support.decisions.cap_hit = support.decisions.memory_frozen = True
            elif outcome.record.get("stopped") == "no room under the cap" or int(n_final[0]) >= cap:
                support.decisions.cap_hit = True
        if outcome.result is not None:
            alphas, betas, widths, status, _q_last = outcome.result
        else:
            converged_flag, delta_min, last_dg, plateau_cell[0] = monitor_state
        if verbose and rank == 0:
            print(_refine_line(outcome.record), flush=True)
    seconds = time.perf_counter() - t0
    if status in ("converged", "invariant_subspace"):
        converged_flag[0] = True
    if (support.cap_hit or support.closed) and verbose and rank == 0:
        print(support.freeze_message(), flush=True)
    k_over_p = float("nan")
    # The retained and matrix-free modes prune too (the cut is in the policy), but they hold no K store to size.
    if (support.prunes or support.mode_ran == "none") and matrix_mode != "discovered" and verbose and rank == 0:
        # The fall-back to matrix-free applies is a silent slowdown otherwise: say it, pruned or not.
        print(
            f"GF {'pruned growth' if support.prunes else 'growth'}: |P| = {support.retained_size:,}, "
            f"matrix mode {support.mode_ran or matrix_mode}, admitted per depth {support.depth_histogram()}",
            flush=True,
        )
    if pt2_scorer is not None and pt2_scorer.declined is not None:
        if cap_info is not None:
            cap_info["pt2"] = {"declined": pt2_scorer.declined}
        if verbose and rank == 0:
            print(
                f"GF second-order growth rule unavailable ({pt2_scorer.declined}): the growth went on by the amplitude "
                f"rule gf_prune_tol = {support._eta:g}",
                flush=True,
            )
    elif pt2_scorer is not None:
        sc = pt2_scorer
        n_rule = int(sc.admitted)
        n_first = max(int(support.retained_size) - n_seed - n_rule, 0)
        why_stop = (
            ("the memory guard" if support.memory_frozen else "the cap")
            if support.cap_hit
            else (
                ("twice its last admission" if sc.last_admission is not None else "its quiet floor")
                if growth_status == "quiet"
                else growth_status
            )
        )
        if cap_info is not None:
            cap_info["pt2"] = {
                "first_shell_rows": n_first,
                "rule_rows": n_rule,
                "mesh_points": int(sc.zs0.size),
                "mesh_points_kept": int(sc.kept.size),
                "mesh_thinned": bool(sc.thinned),
                "blocks": int(sc.blocks),
                "last_admission": sc.last_admission,
                "stop": why_stop,
            }
        if verbose and rank == 0:
            last = "never" if sc.last_admission is None else f"block {sc.last_admission}"
            print(
                f"GF second-order growth: {n_first:,} rows from the first shell, {n_rule:,} from the rule, "
                f"{sc.kept.size} of {sc.zs0.size} mesh points kept"
                f"{' (halved to fit the memory)' if sc.thinned else ''}, last admission {last} of {sc.blocks}, "
                f"growth ended by {'closure under H' if why_stop == 'closed' else why_stop}",
                flush=True,
            )
    if matrix_mode == "discovered" and support.prunes:
        # A deferred row stays in K, so under pruning |K|/|P| is unbounded in principle: report it. The sums are
        # collective and the branch is replicated (mode and pruning are), never gated on the verbose rank.
        sizes = np.array([matrix.n_k_local, matrix.n_alive_local], dtype=np.int64)
        if comm is not None and comm.size > 1:
            comm.Allreduce(MPI.IN_PLACE, sizes, op=MPI.SUM)
        k_over_p = float(sizes[0]) / max(float(sizes[1]), 1.0)
        if verbose and rank == 0:
            print(
                f"GF pruned growth: |K| = {int(sizes[0]):,}, |P| = {int(sizes[1]):,}, "
                f"|K|/|P| = {k_over_p:.2f}, admitted per depth {support.depth_histogram()}",
                flush=True,
            )
    if cap_info is not None:
        cap_info["cap_hit"] = support.cap_hit
        cap_info["retained_size"] = support.retained_size
        cap_info["memory_frozen"] = support.memory_frozen
        cap_info["closed"] = support.closed  # the support is the whole space H reaches (final, not cap-bound)
        cap_info["proxy"] = support
        cap_info["engine"] = support.mode_ran or matrix_mode
        cap_info["status"] = "plateau" if status == "converged" and plateau_cell[0][0] else status
        cap_info["n_vectors"] = int(sum(widths))  # Krylov vectors the final recurrence produced
        cap_info["matrix"] = matrix if matrix_mode == "discovered" else None
        # Whether the frozen recurrence ran on a stored matrix (``csr_seconds`` is then the unit's recurrence wall).
        cap_info["csr_fallback"] = bool(support.decisions.frozen and cap_info["engine"] != "none")
        cap_info["csr_seconds"] = seconds
        cap_info["passes"] = passes
        if pt2_asked:
            # Plain words for the per-unit summary, the one line every unit gets on rank 0's output (the unit's own
            # lines above are printed by its colour's root only).
            from impurityModel.ed.options import DEFAULT_GF_PRUNE_TOL

            declined = (cap_info.get("pt2") or {}).get("declined")
            thinned = bool((cap_info.get("pt2") or {}).get("mesh_thinned"))
            passes["growth_rule"] = (
                f"second-order rule (gf_pt2_tol {pt2_tol:g}{', mesh halved to fit the memory' if thinned else ''})"
                if declined is None
                else f"amplitude rule {DEFAULT_GF_PRUNE_TOL:g}, not gf_pt2_tol {pt2_tol:g}: {declined}"
            )
        cap_info["k_over_p"] = k_over_p
    n_blocks = len(alphas)
    if not converged_flag[0] and rank == 0:
        why = (
            "its divergence guard truncated a corrupted tail"
            if status == "diverged"
            else (
                f"the Krylov space is used up without reorthogonalization, so the recurrence is on its rounding "
                f"floor (last relative change {'n/a' if last_dg[0] is None else format(last_dg[0], '.1e')})"
                if status == "exhausted"
                else "the block budget ran out before the monitor's tolerance was met"
            )
        )
        print(
            f"warning: block Green's function did not reach the convergence tolerance {delta_min:.1e}; the "
            f"block-Lanczos recurrence stopped after {n_blocks} block(s) ({why}). The continued "
            "fraction uses the subspace built so far.",
            flush=True,
        )
    if info is not None:
        info["converged"] = converged_flag[0]
        info["d_g"] = last_dg[0]
        info["n_blocks"] = n_blocks
        info["tol"] = delta_min
        if status == "converged" and plateau_cell[0][0]:
            info["status"] = "plateau"
    alphas, betas = _trim_blocks(alphas, betas, widths)
    alphas, betas = _sanitize_continued_fraction(alphas, betas, rank=rank)
    if cap_info is not None and refine is not None and refine.shifts:
        # After the sanitising, so it describes the continued fraction that is returned. Local arithmetic on replicated
        # coefficients, never raising: no rank can leave before the collectives above or below it.
        passes["peak_shifts"] = peak_shift_check(
            alphas, betas, r, [z for z, _w in refine.shifts], complete=status in ("invariant_subspace", "exhausted")
        )
    elif cap_info is not None and refine is not None and refine.note:
        passes["peak_shifts_note"] = refine.note
    return alphas, betas, r
