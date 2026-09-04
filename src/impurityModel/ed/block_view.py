"""block_view.py
================
Per-restart block-view helpers for the TRLM/IRLM restart layer (``ed/trlm.py``,
``ed/irlm.py``). Path-agnostic: a block is an ``(N, k)`` ndarray, a single
``ManyBodyState``, a ``list[ManyBodyState]``, or (``slice_cols`` only) a
``SparseKrylovDense`` store.

``is_array`` and ``block_cols`` are the per-*step* dispatch primitives (called from the
Cython kernels' hot loop too) and live in ``BlockLanczosCore``'s ``_block_ops.pxi``;
re-exported here so callers of this module have one import site for both the per-step
and per-restart helpers.
"""

import numpy as np

from impurityModel.ed.BlockLanczosCore import block_cols, is_array  # noqa: F401
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SparseKrylovDense


def check_width_sync(Q, widths, where, exact=False):
    """Assert the block-width table and the stored Krylov basis agree.

    Every restarted kernel indexes ``T`` (sized from ``block_widths``) and ``Q_basis``
    (a column store) by the *same* cumulative offsets, so ``sum(widths)`` must never
    exceed ``Q``'s column count. A sweep ends either with a trailing residual block
    appended but not counted (``sum(widths) == cols - w_last``, the ``max_iter`` and
    ``diverged`` exits) or with the last block counted and no residual stored
    (``sum(widths) == cols``, the breakdown exits) -- never with a counted block whose
    vectors were never stored.

    Desynchronization used to surface as an opaque ``matmul`` shape error several
    restarts later (or, where the shapes happened to line up, as Ritz values silently
    paired with the wrong vectors), so check it where the two meet. Pass
    ``exact=True`` where the residual block has already been split off.

    Raises:
        RuntimeError: if the invariant is violated.
    """
    cols = block_cols(Q)
    total = int(sum(widths)) if widths is not None else 0
    if total > cols:
        raise RuntimeError(
            f"{where}: block widths sum to {total} but the Krylov basis holds only {cols} "
            f"columns ({widths!r}). The recurrence counted a block whose vectors were "
            "never stored."
        )
    if exact and total != cols:
        raise RuntimeError(
            f"{where}: block widths sum to {total} but the Krylov basis holds {cols} "
            f"columns ({widths!r}). The residual block has already been split off here, so "
            "the stored columns must be exactly the ones T is expressed in."
        )


def _representation(V):
    """Classify a block's storage representation: ``"array"``, ``"manybody"``, or
    ``"list"`` (plain ``list[ManyBodyState]``, or -- for callers that check further --
    a ``SparseKrylovDense`` store).

    Check array *before* ``ManyBodyState``, always: a ``ManyBodyState``'s own
    ``len()`` is its ROW count (dict-like, matching ``len(dict)``), not its column
    count, so routing on ``len()`` or checking it first silently misclassifies it as
    a bare list. This is the one place that ordering rule is stated -- every
    dispatching helper below routes through it instead of re-deriving it.
    """
    if is_array(V):
        return "array"
    if isinstance(V, ManyBodyState):
        return "manybody"
    return "list"


def slice_cols(Q, a, b):
    rep = _representation(Q)
    if rep == "array":
        return Q[:, a:b]
    if rep == "manybody":
        return Q.select(range(a, b))
    if isinstance(Q, SparseKrylovDense):
        # The store is never itself sliced again below this point (the assigned
        # result -- not Q -- is what every caller keeps rebinding to): this is the one
        # place a TRLM/IRLM run converts its raw Krylov store into the persistent block
        # representation the rest of the restart bookkeeping stays in. slice_block reads
        # the requested columns directly off the store's dense chunks; no more
        # per-column ManyBodyState materialization (__getitem__) followed by a
        # from_states union re-merge.
        return Q.slice_block(a, b)
    return Q[a:b]


def concat_cols(A, B):
    """Concatenate two column blocks. For ``ManyBodyState`` operands, ``A`` and
    ``B`` need not share support: the result is built via ``from_states``, the same
    union-support merge every other block boundary in this module already pays for
    (``block_combine``, ``SparseKrylovDense.combine_block``) -- there is no cheaper
    zero-copy hstack across two independently-supported blocks."""
    rep = _representation(A)
    if rep == "array":
        return np.concatenate([A, B], axis=1)
    if rep == "manybody":
        return ManyBodyState.from_states(A.to_states() + B.to_states())
    return list(A) + list(B)


def copy_block(V):
    rep = _representation(V)
    if rep in ("array", "manybody"):
        return V.copy()
    return [s.copy() for s in V]


def width_synced_total(Q_basis, widths, m_act, p, where, exact=False):
    """Assert width/basis sync (``check_width_sync``) and return the true, possibly
    deflation-shrunk subspace dimension ``sum(widths)`` -- falling back to the padded
    ``m_act * p`` when no width table is tracked. Repeated verbatim at every TRLM/IRLM
    site that builds ``T`` (or slices ``Q_basis``) off the block widths."""
    check_width_sync(Q_basis, widths, where, exact=exact)
    return int(sum(widths)) if widths is not None else m_act * p


def trim_trailing_beta(betas, m_act):
    """Drop the trailing residual-block coupling from ``betas``: a running sweep's
    ``betas`` has one entry per Lanczos step including the not-yet-accepted residual
    (``len(betas) == m_act``), while a sweep that already split the residual off
    returns ``betas`` unchanged."""
    return betas[: m_act - 1] if len(betas) == m_act else betas


def as_state_list(V):
    """Boundary conversion for TRLM/IRLM's returned Ritz vectors: the documented
    ``eigvecs: list[ManyBodyState]`` contract (relied on by every downstream caller,
    e.g. ``cipsi_solver.py``/``groundstate.py``) predates the block-native restart
    bookkeeping, so a ``ManyBodyState`` result is materialized once here, at the
    actual return boundary -- never inside the restart loop itself."""
    return V.to_states() if isinstance(V, ManyBodyState) else V


#: Restarts of no real progress before a restart loop gives up. The residual is *not*
#: monotone across restarts -- measured on an SrMnO3 ground state it swung between 3.9e-5 and
#: 7.7e-4 while the eigenvalue stayed bit-identical -- so progress is judged on the best value
#: seen so far, never on the latest one.
STAGNATION_PATIENCE = 20
#: Factor the best-so-far residual must improve by within one patience window to count as
#: progress. Calibrated on the 69 TRLM solves of one real SrMnO3 gap-DC run: every solve that
#: reached ``tol`` improved its best-so-far by 34x-30000x after restart 10, while the one that
#: never converged managed 17.6x over 89 restarts and plateaued three orders of magnitude
#: above ``tol``. 10x sits inside that gap with room on both sides.
STAGNATION_FACTOR = 10.0


class StagnationMonitor:
    """Windowed no-progress detector for the TRLM/IRLM restart loops.

    Both loops could previously run to ``max_restarts`` on a tolerance they were never going
    to reach. Measured on a real SrMnO3 gap-DC run: one TRLM solve spent all 100 restarts
    moving its max residual from 6.9e-5 to 6.4e-5 while the eigenvalue stayed bit-identical
    for the last 61 and the Krylov basis stayed perfectly orthogonal -- about 10% of every
    Lanczos restart in that run, for nothing.

    Two things make this more than "did the residual go down":

    * the residual is **not monotone** across restarts, so the test is on the best value seen
      so far, compared across a window of ``patience`` restarts;
    * a loop can make real progress **without** the residual moving. IRLM locks converged
      pairs one at a time and then re-aims at the next unconverged ones, so its
      max-over-wanted residual legitimately *rises* on a productive restart. ``progress``
      is any monotone counter of that (locked pairs); an increase resets the window.

    Stopping early is not worse than the status quo -- exhausting ``max_restarts`` already
    returns under-converged Ritz pairs -- it reaches the same outcome sooner.
    """

    def __init__(self, patience=None, factor=None):
        # Read at construction, not as default arguments, so a test can monkeypatch the
        # module-level calibration (default arguments bind at def time and would not see it).
        self.patience = STAGNATION_PATIENCE if patience is None else int(patience)
        self.factor = STAGNATION_FACTOR if factor is None else float(factor)
        self.best = float("inf")
        self._window_ref = float("inf")
        self._window_progress = None
        self._window_start = 0

    def update(self, restart, residual, progress=0):
        """Record one restart; return True when the loop has stopped making progress.

        ``residual`` is the quantity the loop is driving to ``tol`` (its max over the states
        it must converge). ``progress`` is an optional monotone counter of work completed
        that the residual does not reflect.
        """
        residual = float(residual)
        self.best = min(self.best, residual)
        if restart == 0 or self._window_progress is None:
            # Seed the window from the first restart rather than skipping it.
            self._window_ref = residual
            self._window_progress = progress
            self._window_start = restart
            return False
        if restart - self._window_start < self.patience:
            return False
        advanced = progress > self._window_progress or self.best <= self._window_ref / self.factor
        self._window_ref = self.best
        self._window_progress = progress
        self._window_start = restart
        return not advanced
