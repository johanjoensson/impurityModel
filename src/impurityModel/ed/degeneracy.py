"""Which eigenvalues form one degenerate manifold: the one place the question is answered.

A leaf module (numpy only), so the eigensolvers, the CIPSI selection, the observables and the self-energy all
reach it without importing each other. The tolerance is always the caller's: how close counts as degenerate
depends on what achieved the energies (see ``cipsi_solver._degeneracy_tol``), not on this module.

The rule is a *chain*: a level joins the manifold when it lies within ``tol`` of the previous level, so a run
of near-equal levels is never split by where an anchor happens to sit. The price is a manifold wider than
``tol`` when levels creep by just under ``tol`` each; with ``tol`` an order above the eigensolver's residual
that is a run of levels the solver cannot tell apart anyway. (Before this module the rule was anchored at the
manifold's first level in four places and chained in one.)

Two callers keep the first-level anchor (``anchor="first"``: a level joins when within ``tol`` of the manifold's
*first* level), because the chain moves their answer in the unsafe direction: ``observables.is_j_sharp`` (a wider
ground block suppresses the term symbol, g_J and mu_eff because a different-J level sits a chain step away) and
``sector_solver.split_multiplets`` (a level interleaved in a split multiplet chains its parts into one cluster that
looks whole). Both ask whether levels *differ*, so they want the narrowest cluster. (`manifold_observable_values`
keeps it too, since it shares `is_j_sharp`'s tolerance.)
"""

import numpy as np

#: Energy tolerance (absolute) below which the *report* treats eigenstates as one degenerate manifold: the single
#: literal behind ``observables.DEGENERACY_TOL`` and ``gs_report.manifold_tol``. The report never groups tighter
#: than the solve did (``gs_report.manifold_tol`` raises this to the solver's own tolerance when that is looser).
MANIFOLD_TOL = 1e-6


def degenerate_groups(e, tol, *, anchor="chain"):
    """Partition ascending energies ``e`` into manifolds: a list of index lists (empty for no energies).

    ``anchor="chain"`` compares each level with the previous one, ``anchor="first"`` with the manifold's first (see
    the module docstring for who uses which). A gap ``> tol`` splits; a gap of exactly ``tol`` stays within.
    """
    if anchor not in ("chain", "first"):
        raise ValueError(f"anchor must be 'chain' or 'first', got {anchor!r}")
    e = np.asarray(e).real
    groups, start = [], 0
    for i in range(1, len(e) + 1):
        if i == len(e) or abs(e[i] - (e[i - 1] if anchor == "chain" else e[start])) > tol:
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
