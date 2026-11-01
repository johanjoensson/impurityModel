"""Which eigenvalues form one degenerate manifold: the one place the question is answered.

A leaf module (numpy only), so the eigensolvers, the CIPSI selection, the observables and the self-energy all
reach it without importing each other. The tolerance is always the caller's: how close counts as degenerate
depends on what achieved the energies (see ``cipsi_solver._degeneracy_tol``), not on this module.

The rule is a *chain*: a level joins the manifold when it lies within ``tol`` of the previous level, so a run
of near-equal levels is never split by where an anchor happens to sit. The price is a manifold wider than
``tol`` when levels creep by just under ``tol`` each; with ``tol`` an order above the eigensolver's residual
that is a run of levels the solver cannot tell apart anyway. (Before this module the rule was anchored at the
manifold's first level in four places and chained in one.)
"""

import numpy as np


def degenerate_groups(e, tol):
    """Partition ascending energies ``e`` into manifolds: a list of index lists (empty for no energies).

    A gap ``> tol`` between neighbours splits; a gap of exactly ``tol`` stays within the manifold.
    """
    e = np.asarray(e).real
    groups, start = [], 0
    for i in range(1, len(e) + 1):
        if i == len(e) or abs(e[i] - e[i - 1]) > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


def absorb_degenerate_tail(e, n_keep, tol):
    """The cut position ``n_keep`` of ascending ``e``, moved up so it does not fall inside a manifold.

    Absorbs levels while each is within ``tol`` of the one before it (the chain rule of
    :func:`degenerate_groups`). A ``n_keep`` outside ``(0, len(e))`` is returned as is.
    """
    if not 0 < n_keep < len(e):
        return n_keep
    while n_keep < len(e) and abs(e[n_keep] - e[n_keep - 1]) <= tol:
        n_keep += 1
    return n_keep
