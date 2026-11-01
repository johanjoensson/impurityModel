"""Which eigenvalues form one degenerate manifold: the one place the question is answered.

A leaf module (numpy only), so the eigensolvers, the CIPSI selection, the observables and the self-energy all
reach it without importing each other. The tolerance is always the caller's: how close counts as degenerate
depends on what achieved the energies (see ``cipsi_solver._degeneracy_tol``), not on this module.

Two rules exist, and they differ only for a run of levels that drifts:

``anchor="first"``
    A level joins the manifold when it lies within ``tol`` of the manifold's *first* level. A drifting run is
    cut where it leaves ``tol`` of its start.
``anchor="chain"``
    A level joins when it lies within ``tol`` of the *previous* level. A run of near-equal levels is never split
    by where its anchor happens to sit, at the price of a manifold wider than ``tol`` when levels creep.
"""

import numpy as np


def degenerate_groups(e, tol, *, anchor="first"):
    """Partition ascending energies ``e`` into manifolds: a list of index lists (empty for no energies).

    See the module docstring for ``anchor``. The comparison is ``> tol`` splits, so a gap of exactly ``tol``
    stays within the manifold.
    """
    if anchor not in ("first", "chain"):
        raise ValueError(f"anchor must be 'first' or 'chain', got {anchor!r}")
    e = np.asarray(e).real
    groups, start = [], 0
    for i in range(1, len(e) + 1):
        reference = e[start] if anchor == "first" else e[i - 1]
        if i == len(e) or abs(e[i] - reference) > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


def absorb_degenerate_tail(e, n_keep, tol, *, anchor="last_kept"):
    """The cut position ``n_keep`` of ascending ``e``, moved up so it does not fall inside a manifold.

    ``anchor="last_kept"`` absorbs every level within ``tol`` of ``e[n_keep - 1]``, the last level the cut kept
    (fixed: a creeping run is cut once it leaves ``tol`` of it). ``anchor="chain"`` absorbs while each next level
    is within ``tol`` of the one before it. A ``n_keep`` outside ``(0, len(e))`` is returned as is.
    """
    if anchor not in ("last_kept", "chain"):
        raise ValueError(f"anchor must be 'last_kept' or 'chain', got {anchor!r}")
    if not 0 < n_keep < len(e):
        return n_keep
    last = e[n_keep - 1]
    while n_keep < len(e) and abs(e[n_keep] - (last if anchor == "last_kept" else e[n_keep - 1])) <= tol:
        n_keep += 1
    return n_keep
