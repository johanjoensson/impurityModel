"""Set-valued AAA rational approximation and greedy adaptive sampling.

Tools for approximating a *vector-valued* function ``F : R -> C^K`` whose components share
their poles -- e.g. the RIXS map's incoming-frequency dependence, where every polarization
pair and every energy-loss point sees the same intermediate (core-hole) resolvent poles and
only the numerators differ. The set-valued AAA variant (Lietaert et al., "Automatic rational
approximation and linearization of nonlinear eigenvalue problems") fits ONE set of support
points and barycentric weights for all components jointly, so the number of expensive
evaluations (resolvent solves) is governed by the shared pole structure, not by the number
of components.

Everything here is plain numpy on data already gathered to one rank; MPI orchestration
(which rank evaluates what, broadcasting the greedy selection) belongs to the caller.
"""

import numpy as np


def set_valued_aaa(x, F, rtol=1e-13, mmax=None):
    r"""Set-valued AAA fit of ``F`` on the nodes ``x`` with shared support points/weights.

    Greedy AAA in barycentric form: at each step the node with the largest residual (max
    over components) joins the support set, and the weights are the smallest right singular
    vector of the component-stacked Loewner matrix.

    Parameters
    ----------
    x : (n,) array_like
        Sample nodes (real or complex, distinct).
    F : (n, K) array_like
        Function values; each column is one component sharing the pole structure.
    rtol : float
        Stop when ``max_{nodes, components} |F - R| <= rtol * max|F|``.
    mmax : int, optional
        Maximum number of support points. Capped at ``n - 1`` (at least one non-support
        node is needed to define the Loewner residual rows).

    Returns
    -------
    support : list of int
        Indices into ``x`` of the chosen support points, in selection order.
    weights : (m,) ndarray
        Barycentric weights (unit 2-norm), aligned with ``support``.
    """
    x = np.asarray(x)
    F = np.asarray(F, dtype=complex)
    if F.ndim == 1:
        F = F[:, None]
    n = len(x)
    if n < 2:
        raise ValueError("set_valued_aaa needs at least 2 nodes")
    mmax = n - 1 if mmax is None else min(mmax, n - 1)
    scale = np.max(np.abs(F))
    if scale == 0.0:
        return [0], np.ones(1)

    support: list[int] = []
    weights = np.ones(1)
    # Rank-0 initial approximant: the component-wise mean.
    R = np.tile(np.mean(F, axis=0), (n, 1))
    for _ in range(mmax):
        err = np.max(np.abs(F - R), axis=1)
        if support:
            err[support] = 0.0
        j = int(np.argmax(err))
        if support and err[j] <= rtol * scale:
            break
        support.append(j)
        mask = np.ones(n, dtype=bool)
        mask[support] = False
        cauchy = 1.0 / (x[mask, None] - x[support][None, :])  # (n-m, m)
        # Loewner rows stacked over components: L[(k, i), j] = C[i, j] * (F[i, k] - F[support_j, k]).
        loewner = cauchy[:, :, None] * (F[mask, None, :] - F[support][None, :, :])
        loewner = np.moveaxis(loewner, 2, 0).reshape(-1, len(support))
        _, _, vh = np.linalg.svd(loewner, full_matrices=False)
        weights = vh[-1].conj()
        denom = cauchy @ weights
        numer = cauchy @ (weights[:, None] * F[support])
        R = F.copy()
        R[mask] = numer / denom[:, None]
    return support, weights


def barycentric_eval(x_eval, x_support, weights, F_support):
    r"""Evaluate the barycentric rational interpolant at ``x_eval``.

    Nodes that coincide with a support point take the support value exactly (the
    barycentric form interpolates).

    Parameters
    ----------
    x_eval : (p,) array_like
    x_support : (m,) array_like
    weights : (m,) array_like
    F_support : (m, K) array_like
        Function values at the support points. Any number of components ``K`` may be
        used -- in particular *more* components than the fit that produced ``weights``
        saw, since components sharing the pole structure share the weights.

    Returns
    -------
    (p, K) ndarray
    """
    x_eval = np.asarray(x_eval)
    x_support = np.asarray(x_support)
    weights = np.asarray(weights, dtype=complex)
    F_support = np.asarray(F_support, dtype=complex)
    if F_support.ndim == 1:
        F_support = F_support[:, None]
    diff = x_eval[:, None] - x_support[None, :]
    exact_row, exact_col = np.nonzero(diff == 0)
    # Exact support hits are overwritten below; give them a finite placeholder so the
    # barycentric sums stay warning-free.
    diff[exact_row, exact_col] = 1.0
    cauchy = 1.0 / diff
    out = (cauchy @ (weights[:, None] * F_support)) / (cauchy @ weights)[:, None]
    out[exact_row] = F_support[exact_col]
    return out


def greedy_next_samples(x, solved, surrogate_err, n_pick):
    r"""Pick the next ``n_pick`` sample nodes for adaptive rational sampling.

    Primary criterion: the largest entries of ``surrogate_err`` over the unsolved nodes
    (the caller's error proxy -- typically the disagreement between two consecutive AAA
    iterates). Nodes where the surrogate is degenerate (zero / non-finite, e.g. on the
    very first round or when the fit did not change) are filled space-fillingly: the
    unsolved node farthest from every solved node.

    Parameters
    ----------
    x : (n,) array_like
        All candidate nodes.
    solved : sequence of int
        Indices already evaluated.
    surrogate_err : (n,) array_like or None
        Error proxy per node; ``None`` means no proxy is available yet.
    n_pick : int
        How many new indices to return (fewer if not enough unsolved nodes remain).

    Returns
    -------
    list of int
    """
    x = np.asarray(x)
    n = len(x)
    unsolved = [i for i in range(n) if i not in set(solved)]
    if not unsolved:
        return []
    n_pick = min(n_pick, len(unsolved))
    picks: list[int] = []
    if surrogate_err is not None:
        err = np.asarray(surrogate_err, dtype=float).copy()
        err[~np.isfinite(err)] = 0.0
        order = sorted(unsolved, key=lambda i: -err[i])
        picks = [i for i in order[:n_pick] if err[i] > 0.0]
    chosen = set(solved) | set(picks)
    while len(picks) < n_pick:
        # Space-filling fallback: unsolved node farthest from everything chosen so far.
        best, best_dist = None, -1.0
        for i in unsolved:
            if i in chosen:
                continue
            dist = min(abs(x[i] - x[j]) for j in chosen)
            if dist > best_dist:
                best, best_dist = i, dist
        if best is None:
            break
        picks.append(best)
        chosen.add(best)
    return picks
