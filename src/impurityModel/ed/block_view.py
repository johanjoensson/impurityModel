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
