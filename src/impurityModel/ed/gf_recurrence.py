"""The Green's-function block-Lanczos recurrence on a discovered-key matrix.

One step is the textbook block recurrence over tall-skinny blocks (:func:`lanczos_step`)::

    w = A x;  alpha = <x|w>;  w -= x alpha + x_prev beta_prev^H;  x_next, beta = TSQR(w)

with ``A`` a callable, so the same step serves a fixed matrix and one that grows while it is applied.
:func:`block_lanczos_discovered` wraps it for :class:`~impurityModel.ed.discovered_matrix.DiscoveredMatrix`:
the matvec returns the step's rows in ``P`` (the Lanczos step) and in ``F = K \\ P`` (the frontier), the
admission decision reads ``F``, :meth:`DiscoveredMatrix.grow` admits the chosen rows (one operator
application per admitted determinant), and the admitted rows' amplitude enters the same step. After the
support freezes (cap, memory guard, settle) ``F`` is discarded and the recurrence runs on ``P H P`` from
the same blocks -- the flag is the whole freeze.

**Cut semantics are the array kernel's**: no ``slaterWeightMin`` cut on the ``P`` rows of ``w`` and none on
``q_next``; the cut acts on the ``F`` rows (the rows admission may take). This is what
``gf_solvers._frozen_csr_green``'s restart does; it differs from the ManyBodyState kernel's pre-freeze steps
by the cut amounts at ``slaterWeightMin > 0``.

The admission rules (cap with importance-ranked boundary, importance pruning ``eta`` without a ban, the
measured-memory guard, the support settle) are the :class:`~impurityModel.ed.gf_primitives.SupportDecisions`
applied to dense frontier rows.
"""

import math
import time

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.BlockLanczosCore import apply_and_redistribute, divergence_guard
from impurityModel.ed.discovered_matrix import DiscoveredMatrix
from impurityModel.ed.gf_convergence import _gf_monitor_tol, _make_gf_convergence_monitor
from impurityModel.ed.gf_primitives import (
    SETTLE_WINDOW,
    SupportDecisions,
    _distributed_seed_qr,
    _sanitize_continued_fraction,
    _trim_blocks,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.memory_estimate import format_bytes
from impurityModel.ed.TSQR import DEFLATE_TOL_SEEDS, spectral_norm, tsqr

#: Fields of :class:`DiscoveredSupport` that live on its :class:`SupportDecisions` (the names callers read).
_FORWARDED = {
    "cap": "cap",
    "memory_budget": "memory_budget",
    "memory_policy": "memory_policy",
    "memory_frozen": "memory_frozen",
    "cap_hit": "cap_hit",
    "support_freeze": "support_freeze",
    "support_frozen": "support_frozen",
    "support_trace": "support_trace",
    "monitor_dg": "monitor_dg",
    "frozen": "frozen",
}


class DiscoveredSupport:
    """The support policy of a run on a discovered-key matrix, under the names the GF callers read.

    Holds the :class:`SupportDecisions` and the matrix; ``retained_*`` are the matrix's live ``P`` columns.
    """

    caps_growth = True
    tracks_support = True
    stop_on_freeze = False
    apply_filter = None

    def __init__(self, matrix, decisions, eta=None, first_shell_tol=0.0):
        self.matrix = matrix
        self.decisions = decisions
        self.comm = matrix.comm
        self.prunes = eta is not None
        self._eta2 = float(eta) ** 2 if self.prunes else 0.0
        self._shell_tol2 = float(first_shell_tol) ** 2
        self._shell_admitted = False

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

    ban_bytes = 0

    def enable_support_freeze(self, window, ratio, slater_weight_min, log=False, trace=False):
        if self.prunes:
            raise ValueError("the support freeze and importance pruning (GF_LANCZOS_ADMIT_TOL) are exclusive")
        self.decisions.enable_support_freeze(window, ratio, slater_weight_min, log=log, trace=trace)

    def settle_verdict(self):
        return self.decisions.settle_verdict()

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
#: Bytes per stored element (a 16 B value and an 8 B packed row) and per K row (key, two slot arrays, the
#: ``K -> P`` map and slack).
_ELEMENT_BYTES = 24
_K_ROW_BYTES = 36


def matrix_need_bytes(n_p_local, n_k_local, fanout, n_global_k, n_ranks, width):
    """Upper bound on the bytes a rank needs for the discovered-key matrix and the recurrence on it: the
    columns of ``P_local`` over the rows of ``K``, the ``K`` store, the dense blocks (``q``, ``q_prev``, the
    next block over ``P`` and the product over ``K``), the exchange buffers and one ``grow`` batch."""
    from impurityModel.ed.discovered_matrix import _GROW_BATCH
    from impurityModel.ed.mpi_comm import matvec_column_chunk_width

    elements = n_p_local * fanout
    matrix = elements * _ELEMENT_BYTES + n_k_local * _K_ROW_BYTES
    working = (3 * n_p_local + n_k_local) * width * 16
    if n_ranks > 1:
        send_rows = max(n_global_k - n_k_local, 0)
        recv_rows = n_k_local * (n_ranks - 1)
        w_c = matvec_column_chunk_width(max(send_rows, recv_rows), width, config.GS_MATVEC_EXCHANGE_BYTES.get())
        working += 16 * w_c * (send_rows + recv_rows + n_k_local)
    return matrix + working + min(_GROW_BATCH, elements) * 160


def choose_matrix_mode(basis, hop, cap, width, memory_budget, report=None):
    """The matvec mode of a unit, decided once and collectively before its first step.

    ``"discovered"`` when the discovered-key matrix fits ``MATRIX_MEMORY_SAFETY`` of the rank's share on every
    rank, else ``"retained"`` when the ``P H P`` matrix built at the freeze fits the GF stage's CSR budget
    (``gf_solvers._frozen_matrix_need``), else ``"none"`` (the operator is applied to every block). No memory
    budget (the guard is off) always fits. ``P`` is priced at ``cap`` determinants spread over the ranks (the
    seed basis when the cap is unbounded), ``K`` at :data:`K_OVER_P_ESTIMATE` times that. ``report`` (a dict)
    receives the numbers the decision used. **Collective.**
    """
    from impurityModel.ed.basis_transcription import iter_local_operator_images
    from impurityModel.ed.gf_solvers import _CSR_FANOUT_SAMPLE, _frozen_matrix_need
    from impurityModel.ed.memory_estimate import current_rss_bytes

    comm = basis.comm if basis.is_distributed else None
    if memory_budget is None:
        return "discovered"
    n_ranks = comm.size if comm is not None else 1
    n_basis_local = len(basis.local_basis)
    positions = ()
    if n_basis_local:
        positions = np.unique(
            np.linspace(0, n_basis_local - 1, min(_CSR_FANOUT_SAMPLE, n_basis_local)).astype(np.int64)
        )
    sampled = elements = 0
    for image in iter_local_operator_images(basis, hop, 0, indices=positions):
        sampled += 1
        elements += len(image)
    fanout = elements / sampled if sampled else 0.0
    n_p = max(n_basis_local, int(min(cap, 2**40) / n_ranks)) if np.isfinite(cap) and cap < 2**40 else n_basis_local
    n_k = int(K_OVER_P_ESTIMATE * n_p)
    need_e = matrix_need_bytes(n_p, n_k, fanout, n_k * n_ranks, n_ranks, width)
    need_csr = _frozen_matrix_need(n_p, fanout, n_p * n_ranks, n_ranks, width)
    csr_budget = getattr(basis, "gf_csr_memory_budget", None)
    csr_budget = memory_budget if csr_budget is None else max(memory_budget, csr_budget)
    e_budget = csr_budget * MATRIX_MEMORY_SAFETY / config.GF_CSR_MEMORY_SAFETY.get()
    rss = current_rss_bytes()
    over = np.array([rss + need_e > e_budget, rss + need_csr > csr_budget], dtype=np.int64)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, over, op=MPI.MAX)
    if report is not None:
        report.update(
            n_p_local=n_p,
            n_k_local=n_k,
            fanout=fanout,
            need_discovered=need_e,
            need_retained=need_csr,
            rss=rss,
            budget_discovered=e_budget,
            budget_retained=csr_budget,
        )
    if not over[0]:
        return "discovered"
    return "retained" if not over[1] else "none"


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


def lanczos_step(apply, x, x_prev, beta_prev, *, comm=None, scale=0.0, deflate_tol=-1.0, store=None):
    """One block-Lanczos step; collective when ``comm`` is given.

    ``apply(x)`` returns ``w = A x``; it may return **more rows** than ``x`` has (a matrix that admitted
    determinants while being applied appends their slots at the end), in which case ``x`` and ``x_prev`` are
    zero there. ``x_prev`` and ``beta_prev`` are ``None`` on the first step. ``scale`` is the operator-scale
    estimate the TSQR breakdown test is measured against (``max(scale, ||alpha||)`` is used).

    Returns ``(alpha, x_next, beta, k)``: ``alpha`` is ``(p, p)``, ``x_next`` the ``(rows, k)`` orthonormal
    next block and ``beta`` the ``(k, p)`` factor with ``w - ... = x_next @ beta``; ``k`` is the retained
    rank (``0`` breakdown: the Krylov space is closed, ``-1`` a non-finite factor; ``x_next`` and ``beta``
    are then ``None``). ``store`` (a list of the earlier blocks, each with at most ``w``'s rows) switches on
    full reorthogonalization: two classical Gram-Schmidt passes against every stored block.
    """
    w = np.array(apply(x), dtype=complex, order="C")  # a private copy: the residual is formed in place
    n_rows = w.shape[0]
    x = _pad_rows(x, n_rows)
    alpha = _allreduce(comm, np.ascontiguousarray(x.conj().T @ w))
    w -= x @ alpha
    if x_prev is not None:
        w -= _pad_rows(x_prev, n_rows) @ beta_prev.conj().T
    if store is not None:
        for _ in range(2):
            for block in store:
                rows = block.shape[0]
                overlap = _allreduce(comm, np.ascontiguousarray(block.conj().T @ w[:rows]))
                w[:rows] -= block @ overlap
    x_next, beta, k, _sv = tsqr(w, comm, max(float(scale), spectral_norm(alpha)), deflate_tol=deflate_tol)
    return alpha, x_next, beta, k


class _Admission:
    """The admission decision of one step on dense frontier rows. Collective; replicated verdicts."""

    def __init__(self, support, slater_weight_min):
        self.support = support
        self.d = support.decisions
        self.swm = float(slater_weight_min)
        self.pending = None  # the settle record of the step, completed by note_next_block
        self.pending_hq2 = 0.0

    def decide(self, f_ids, wf, hq2, keys_of):
        """The frontier rows to admit from the step's matvec output.

        ``f_ids`` identify this rank's frontier rows (K slots, or row numbers of the applied block) and
        ``wf`` their amplitudes ``(n_f, p)``; ``hq2`` is this rank's ``||w||^2`` over all rows (the settle
        trace) and ``keys_of(ids)`` the determinants of some ids. Cuts the frontier at ``slaterWeightMin``
        (the weight it drops is reported for the settle), applies the memory guard, then the counted or
        pruned rule; at the cap it ranks the candidates and freezes. Returns the ids admitted."""
        d = self.d
        sup = self.support
        self.pending = None
        f_slots = np.asarray(f_ids, dtype=np.int64)
        amp2 = (np.abs(wf) ** 2).max(axis=1) if wf.shape[1] else np.zeros(len(f_slots))
        if d.support_freeze:
            self.pending_hq2 = float(hq2)
        keep = amp2 > self.swm**2
        if d.support_freeze:
            d.note_step_cut(float(np.sum(np.abs(wf[~keep]) ** 2)))
        slots = f_slots[keep]
        norms2 = amp2[keep]
        if d.over_memory_budget():
            d.freeze_on_memory()
            return slots[:0]
        if sup.prunes:
            return self._pruned(keys_of, slots, norms2)
        if d.support_freeze:
            return self._counted_support(keys_of, slots, norms2, wf[keep])
        n_new = int(d.allreduce_sum(len(slots)))
        if d.global_count + n_new <= d.cap:
            d.global_count += n_new
            return slots
        return self._top_and_freeze(keys_of, slots, norms2)

    def _pruned(self, keys_of, slots, norms2):
        sup, d = self.support, self.d
        first_shell = not sup._shell_admitted and int(d.allreduce_sum(len(slots))) > 0
        sup._shell_admitted = sup._shell_admitted or first_shell
        sel = norms2 > (sup._shell_tol2 if first_shell else sup._eta2)
        slots, norms2 = slots[sel], norms2[sel]
        n_new = int(d.allreduce_sum(len(slots)))
        if d.global_count + n_new <= d.cap:
            d.global_count += n_new
            return slots
        return self._top_and_freeze(keys_of, slots, norms2)

    def _counted_support(self, keys_of, slots, norms2, rows):
        d = self.d
        n_new = len(slots)
        leak = float(np.sum(np.abs(rows) ** 2))
        max2 = float(norms2.max()) if len(norms2) else 0.0
        local = np.array([n_new, leak, d.pending_cut, self.pending_hq2, max2], dtype=np.float64)
        d.pending_cut = 0.0
        hq2_local = self.pending_hq2
        self.pending_hq2 = 0.0
        if d.comm is not None and d.comm.size > 1:
            gathered = np.empty((d.comm.size, 5), dtype=np.float64)
            d.comm.Allgather(local, gathered)
        else:
            gathered = local[None, :]
        n_new_g = round(math.fsum(gathered[:, 0]))
        leak_g = math.fsum(gathered[:, 1])
        cut_g = math.fsum(gathered[:, 2])
        hq2_g = math.fsum(gathered[:, 3]) if d.comm is not None else hq2_local
        max2_g = float(np.max(gathered[:, 4]))
        if d.global_count + n_new_g > d.cap:
            return self._top_and_freeze(keys_of, slots, norms2)
        retained_before = d.global_count
        d.global_count += n_new_g
        self.pending = (slots, retained_before, n_new_g, leak_g, cut_g, hq2_g, max2_g)
        return slots

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

    def attach(self, pslots):
        """The engine tells the pending settle record which P slots the step's admitted rows took."""
        if self.pending is not None:
            self.pending = (np.asarray(pslots, dtype=np.int64),) + self.pending[1:]

    def note_next_block(self, q_next):
        """Complete the step's settle record with the next block's amplitude on the rows it admitted, and take
        the verdict (one fixed-shape MAX on every rank)."""
        pending, self.pending = self.pending, None
        d = self.d
        if pending is None:
            return
        pslots, retained_before, n_new, leak, cut, hq2, max2_matvec = pending
        buf = np.zeros(1)
        if len(pslots) and q_next.shape[0] > int(pslots.max()):
            buf[0] = float(np.max(np.sum(np.abs(q_next[pslots]) ** 2, axis=1)))
        _allreduce(d.comm if d.comm is not None and d.comm.size > 1 else None, buf, MPI.MAX)
        d.settle_step(float(buf[0]), retained_before, n_new, leak, cut, hq2, max2_matvec)


class _MatrixEngine:
    """Matvec mode ``"discovered"``: ``w = M q`` over every K row, ``grow`` for what admission takes."""

    def __init__(self, matrix, admission):
        self.matrix = matrix
        self.admission = admission

    def apply(self, block):
        matrix, admission = self.matrix, self.admission
        d = admission.d
        if d.frozen:
            return matrix.matvec(block, rows="P")
        w_k = matrix.matvec(block, rows="all")
        f_slots = np.flatnonzero(matrix.k_to_p_slot() < 0)
        hq2 = math.fsum(np.sum(np.abs(w_k) ** 2, axis=0)) if d.support_freeze else 0.0
        slots = admission.decide(f_slots, w_k[f_slots], hq2, matrix.keys_of)
        # The verdict (how many rows were admitted) is replicated, so every rank takes the same branch.
        if int(d.allreduce_sum(len(slots))):
            admitted = matrix.grow(matrix.keys_of(slots))
            admission.attach(admitted)
        w_p = w_k[matrix.p_to_k_slot()]
        w_p[~matrix.alive()] = 0.0
        return w_p


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
        self._n_chunks = int(config.APPLY_ROW_CHUNKS.get() or 1)

    def _applied(self, x):
        """``H x`` as a ManyBodyState routed to the owners (no cut: the frontier is cut at admission)."""
        matrix = self.matrix
        k_to_p = matrix.k_to_p_slot()
        by_pos = np.zeros((matrix.n_k_local, x.shape[1]), dtype=complex)
        live = k_to_p[matrix._slot_of_pos] >= 0
        by_pos[live] = x[k_to_p[matrix._slot_of_pos[live]]]
        block = ManyBodyState.from_keys_and_amps(matrix._kstore, by_pos)
        redistribute = matrix.comm is not None
        return apply_and_redistribute(
            self.hop, block, self.basis, 0.0, redistribute, self._n_chunks if redistribute else 1
        )

    def apply(self, x):
        matrix, admission = self.matrix, self.admission
        d = admission.d
        if d.frozen and self.retained:
            if not self._built:
                matrix.build_columns()
                self._built = True
            return matrix.matvec(x, rows="P")
        applied = self._applied(x)
        keys = applied.keys()
        amps = np.asarray(applied)
        pos = matrix._kstore.find_rows(keys) if keys else np.zeros(0, dtype=np.int64)
        in_p = pos < matrix.n_k_local
        k_to_p = matrix.k_to_p_slot()
        w_p = np.zeros((matrix.n_p_local, x.shape[1]), dtype=complex)
        p_slot = k_to_p[matrix._slot_of_pos[pos[in_p]]]
        w_p[p_slot[p_slot >= 0]] = amps[in_p][p_slot >= 0]
        if d.frozen:
            return w_p
        f_ids = np.flatnonzero(~in_p)
        hq2 = float(np.sum(np.abs(amps) ** 2)) if d.support_freeze else 0.0
        ids = admission.decide(f_ids, amps[f_ids], hq2, lambda sel: [keys[i] for i in sel])
        if int(d.allreduce_sum(len(ids))):
            new = matrix.admit_keys([keys[i] for i in ids])
            admission.attach(new)
            w_new = np.zeros((matrix.n_p_local, x.shape[1]), dtype=complex)
            w_new[: w_p.shape[0]] = w_p
            new_keys = matrix.keys_of(matrix.p_to_k_slot()[new])
            rows = applied.find_rows(new_keys)
            w_new[new] = amps[rows]
            w_p = w_new
        return w_p


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
):
    """Block Lanczos of ``H`` on the discovered-key matrix ``matrix`` from the orthonormal seed block ``q0``
    (``(n_p_local, n)``, rows in P-slot order), growing ``P`` through ``support``'s decisions.

    Returns ``(alphas, betas, widths, status, q_last)``: the padded ``(m, n, n)`` coefficient arrays, the
    per-block widths, why it stopped (``"converged"``, ``"invariant_subspace"``, ``"diverged"`` or
    ``"max_iter"``) and this rank's rows of the last accepted block (P-slot order of the final matrix).
    **Collective.** ``converged`` is the GF convergence monitor, called every block on every rank.
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
    freeze_step = None

    it = 0
    while it < max_blocks:
        n_curr = x.shape[1]
        if comm is not None:
            probe = np.array([n_curr, -n_curr], dtype=np.int64)
            comm.Allreduce(MPI.IN_PLACE, probe, op=MPI.MAX)
            if probe[0] != -probe[1]:
                raise RuntimeError(f"block width disagrees across ranks at block {it}: [{-probe[1]}, {probe[0]}]")
        alpha, x_next, beta, k = lanczos_step(
            engine.apply,
            x,
            x_prev,
            beta_prev,
            comm=comm,
            scale=max(h_norm_est, t_norm_max),
            deflate_tol=deflate_tol,
            store=store,
        )
        if it >= alphas.shape[0]:
            alphas = np.concatenate((alphas, np.zeros_like(alphas)))
            betas = np.concatenate((betas, np.zeros_like(betas)))
        alphas[it, :n_curr, :n_curr] = alpha
        if k <= 0:
            status = "diverged" if k < 0 else "invariant_subspace"
            widths.append(n_curr)
            it += 1
            break
        betas[it, :k, :n_curr] = beta
        diverged, t_norm_max, h_norm_est = divergence_guard(
            spectral_norm(beta), spectral_norm(alpha), it == 0, t_norm_max, h_norm_est
        )
        if diverged:
            status = "diverged"
            break
        if d.support_freeze and not d.frozen:
            admission.note_next_block(x_next)
            if d.frozen:
                freeze_step = it
        done = converged(alphas[: it + 1], betas[: it + 1], verbose=verbose, block_widths=widths + [n_curr])
        if comm is not None:
            done = comm.bcast(bool(done), root=0)
        if done:
            status = "converged"
            widths.append(n_curr)
            it += 1
            break
        x_prev = _pad_rows(x, x_next.shape[0])
        beta_prev = beta
        x = x_next
        if store is not None:
            store.append(x)
        widths.append(n_curr)
        it += 1
    if status != "max_iter" and freeze_step is not None and freeze_step == it - 1 and d.support_frozen:
        # The support settled on the step the recurrence ended on anyway: nothing was dropped.
        d.unfreeze_support()
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
):
    """The matrix, the support and the seed block on ``P0 = basis``.

    Admits every determinant of ``basis`` (a ``grow`` of its local keys; ``admit_keys``, with no operator
    application, in the ``"retained"`` and ``"none"`` modes) and returns
    ``(matrix, support, q0)`` with ``q0`` the seed block ``psi_dense_local`` (rows in ``basis.local_basis``
    order) permuted to the matrix's P-slot order. **Collective.**
    """
    comm = basis.comm if basis.is_distributed else None
    cps = (basis.n_bytes + 7) // 8
    matrix = DiscoveredMatrix(hop, cps, comm, keep_frontier=(mode == "discovered"))
    decisions = SupportDecisions(comm, cap, int(basis.size), memory_budget=memory_budget, memory_policy=memory_policy)
    support = DiscoveredSupport(matrix, decisions, eta=eta, first_shell_tol=first_shell_tol)
    local = list(basis.local_basis)
    if mode == "discovered":
        matrix.grow(local)
    else:
        matrix.admit_keys(local)
    p_keys = matrix.p_keys()
    where = basis._keys.find_rows(p_keys)  # local positions of the P-slot keys in basis order
    q0 = np.ascontiguousarray(np.asarray(psi_dense_local, dtype=complex)[where])
    return matrix, support, q0


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
    support_settle=True,
    settle_ratio=None,
    matrix_mode="auto",
):
    """One block of the Green's function from the recurrence on a discovered-key matrix.

    The counterpart of ``gf_solvers.block_Green_sparse`` with the same arguments and return value
    ``(alphas, betas, r)``; ``cap_info`` additionally gets ``engine`` (the matvec mode that ran:
    ``"discovered"``, ``"retained"`` or ``"none"``) and ``matrix`` (the :class:`DiscoveredMatrix` in the
    ``"discovered"`` mode, else ``None``; never put it in ``cap_stats``). ``matrix_mode`` picks the matvec: the
    discovered-key matrix, the operator applied each step with the ``P H P`` matrix built at the freeze
    (``"retained"``), or the operator applied every block (``"none"``). The determinant cap is
    ``basis.truncation_threshold``; ``GF_LANCZOS_ADMIT_TOL > 0`` turns on importance pruning.
    """
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
        cap_info["matrix"] = None
        cap_info["support_frozen"] = False
        cap_info["settle_verdict"] = None
        cap_info["support_trace"] = []
        for key in ("leak_settle", "frontier_settle_matvec", "frontier_end", "frontier_end_matvec", "leak_end"):
            cap_info[key] = float("nan")
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
    cap = getattr(basis, "truncation_threshold", np.inf)
    cap = cap if np.isfinite(cap) else 2**62
    prune_tol = config.GF_LANCZOS_ADMIT_TOL.get()
    if memory_budget is None:
        memory_budget = getattr(basis, "gf_memory_budget", None)
        memory_policy = getattr(basis, "gf_memory_policy", None) or "tighten"
    resolved_reort = False if reort is None else bool(reort)
    if matrix_mode not in ("auto", "discovered", "retained", "none"):
        raise ValueError(f"matrix_mode must be 'auto', 'discovered', 'retained' or 'none', got {matrix_mode!r}")
    if matrix_mode == "auto":
        report: dict = {}
        matrix_mode = choose_matrix_mode(basis, hOp, cap, len(psi_arr), memory_budget, report)
        if verbose and rank == 0 and report:
            print(
                f"GF matvec mode {matrix_mode}: {report['n_p_local']:,} local determinants x fan-out "
                f"{report['fanout']:.1f}, need {format_bytes(report['need_discovered'])} (discovered) / "
                f"{format_bytes(report['need_retained'])} (retained) on top of RSS {format_bytes(report['rss'])}; "
                f"budgets {format_bytes(report['budget_discovered'])} / {format_bytes(report['budget_retained'])}.",
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
        first_shell_tol=config.GF_ADMIT_FIRST_SHELL_TOL.get() if prune_tol > 0.0 else 0.0,
        mode=matrix_mode,
    )
    if support_settle and not resolved_reort and not support.prunes:
        support.enable_support_freeze(
            SETTLE_WINDOW,
            config.GF_SETTLE_RATIO.get() if settle_ratio is None else settle_ratio,
            slaterWeightMin,
            log=verbose and rank == 0,
            trace=config.GF_SUPPORT_TRACE.get() and rank == 0,
        )
        support.monitor_dg = last_dg
    t0 = time.perf_counter()
    alphas, betas, widths, status, _q_last = block_lanczos_discovered(
        matrix,
        support,
        q0,
        converged,
        slater_weight_min=slaterWeightMin,
        deflate_tol=DEFLATE_TOL_SEEDS,
        reort=resolved_reort,
        engine=(
            None
            if matrix_mode == "discovered"
            else lambda m, a: _ApplyEngine(m, a, hOp, basis, matrix_mode == "retained")
        ),
    )
    seconds = time.perf_counter() - t0
    if status in ("converged", "invariant_subspace"):
        converged_flag[0] = True
    if (support.cap_hit or support.support_frozen) and verbose and rank == 0:
        print(support.freeze_message(), flush=True)
    if cap_info is not None:
        cap_info["cap_hit"] = support.cap_hit
        cap_info["retained_size"] = support.retained_size
        cap_info["memory_frozen"] = support.memory_frozen
        cap_info["proxy"] = support
        cap_info["engine"] = matrix_mode
        cap_info["matrix"] = matrix if matrix_mode == "discovered" else None
        cap_info["csr_fallback"] = True
        cap_info["csr_seconds"] = seconds
        cap_info["support_frozen"] = support.support_frozen
        cap_info["settle_verdict"] = support.settle_verdict()
        cap_info["support_trace"] = list(support.support_trace)
    n_blocks = len(alphas)
    if not converged_flag[0] and rank == 0:
        print(
            f"warning: block Green's function did not reach the convergence tolerance {delta_min:.1e}; the "
            f"block-Lanczos recurrence was truncated after {n_blocks} block(s) (divergent tail). The continued "
            "fraction uses the subspace built so far.",
            flush=True,
        )
    if info is not None:
        info["converged"] = converged_flag[0]
        info["d_g"] = last_dg[0]
        info["n_blocks"] = n_blocks
        info["tol"] = delta_min
    alphas, betas = _trim_blocks(alphas, betas, widths)
    alphas, betas = _sanitize_continued_fraction(alphas, betas, rank=rank)
    return alphas, betas, r
