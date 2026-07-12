# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True, freethreading_compatible=True

"""
ChebyshevFilter.pyx
===================
Chebyshev window filters (Jackson-damped partition of unity) for the many-body layer.

The filter stage of the spectrum-slicing Green's function
(``doc/plans/spectrum_slicing.md``): given spectral bounds of ``H`` on the excited
sector, build window polynomials :math:`p_s(H)` that tile the interval and sum to one
*identically* (the coefficients of tiling indicator windows telescope, Jackson damping
included), and apply the whole filter bank to a seed block in a **single** three-term
Chebyshev recurrence — three live blocks plus one accumulator per window, cap-aware
through ``caps_growth`` bases (``_CappedBasisProxy``).

Phase-0 calibration (measured, FCC Ni): the filtered seeds' *dominant* amplitudes are
strongly energy-local, their sub-1e-6 amplitude tails are not — a slice basis truncated
at amplitude ``eps`` represents the slice seed to roughly ``sqrt(n_eps) * eps``, which is
what the sliced driver's memory model prices.

``impurityModel.ed.chebyshev_filter`` re-exports the public entry points, mirroring the
``cg.py`` / ``gmres.py`` thin-wrapper arrangement.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import Reort, resolve_reort
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyBlockState,
    ManyBodyState,
    block_add_scaled_cy,
)


def spectral_bounds(hOp, basis, n_iter=40, pad_rel=0.05, seed=1):
    """Padded extremal Ritz values of ``H`` on (a capped view of) ``basis``'s sector.

    A short ``reort=none`` Lanczos run from a random vector; both edges come from one
    tridiagonal. The Ritz values approach the true edges from inside, so the interval is
    padded by ``pad_rel`` of the estimated width — the KPM-standard guard that keeps the
    rescaled spectrum inside ``[-1, 1]`` (a filter evaluated outside the Chebyshev
    interval diverges) [Weisse et al., RMP 78, 275 (2006)].

    Collective on ``basis``'s communicator (the Lanczos matvecs are). The random seed is
    fixed by default so every rank builds the identical start vector from its local rows.

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian; its restrictions are set from ``basis``.
    basis : Basis or _CappedBasisProxy
        Sector definition (and optional growth cap) for the bounds estimate.
    n_iter : int
        Lanczos steps; ~40 gives edge estimates far better than the padding absorbs.
    pad_rel : float
        Relative padding of the interval on each side.
    seed : int
        RNG seed for the start vector.

    Returns
    -------
    tuple of float
        ``(lower, upper)`` padded spectral bounds.
    """
    rng = np.random.default_rng(seed)
    v = ManyBodyState({d: complex(*rng.standard_normal(2)) for d in basis.local_basis})
    norm2 = v.norm2()
    if basis.comm is not None and getattr(basis, "is_distributed", False):
        norm2 = basis.comm.allreduce(norm2, op=MPI.SUM)
    v = v * (1.0 / np.sqrt(norm2))
    hOp.set_restrictions(basis.restrictions)
    hOp.set_weighted_restrictions(basis.weighted_restrictions)
    alphas, betas, *_ = block_lanczos_cy(
        [v],
        hOp,
        basis,
        lambda *a, **k: False,
        verbose=False,
        reort=resolve_reort(Reort.NONE),
        slaterWeightMin=0.0,
        max_iter=n_iter,
        store_krylov=False,
    )
    cdef Py_ssize_t k = len(alphas)
    T = np.zeros((k, k), dtype=complex)
    for i in range(k):
        T[i, i] = np.asarray(alphas[i])[0, 0]
        if i + 1 < k:
            b = np.asarray(betas[i])[0, 0]
            T[i, i + 1] = np.conj(b)
            T[i + 1, i] = b
    ev = np.linalg.eigvalsh(T)
    pad = pad_rel * (ev[-1] - ev[0])
    return float(ev[0] - pad), float(ev[-1] + pad)


def partition_of_unity(bounds, slice_edges, degree):
    """Jackson-damped Chebyshev coefficients of windows tiling the spectral interval.

    The windows are the indicator functions of ``[edge_i, edge_{i+1}]`` (the caller's
    ``slice_edges`` clipped into ``bounds``, with the two rest-windows down to the lower
    and up to the upper bound added when the edges do not reach them). Their exact
    Chebyshev coefficients are analytic,

        c_0 = (theta_a - theta_b) / pi,
        c_n = (2/pi) (sin(n theta_a) - sin(n theta_b)) / n,   theta = arccos(x),

    damped by the Jackson kernel g_n (positivity-preserving, leakage width ~ pi/degree).
    Because the exact window coefficients of a tiling *telescope* to those of the
    constant 1 and the damping is applied uniformly, ``sum_s p_s(H) = 1`` holds to
    machine precision **by construction** — the partition error of the sliced Green's
    function is exactly zero; only per-window *leakage* (weight from outside the window,
    kernel-broadened over ~ (bounds width)·pi/degree around each edge) remains.

    Parameters
    ----------
    bounds : tuple of float
        ``(lower, upper)`` from :func:`spectral_bounds`.
    slice_edges : sequence of float
        Monotone slice boundaries in energy units (typically spanning the caller's
        evaluation band, shifted by the thermal energy).
    degree : int
        Polynomial degree of every window.

    Returns
    -------
    tuple
        ``(coefficient_sets, window_edges, edge_width)`` — one ``(degree+1,)`` float
        array per window, the actual window intervals (rest-windows included), and the
        kernel edge-broadening estimate (energy units) for the diagnostics.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    c_mid = 0.5 * (hi + lo)
    e_half = 0.5 * (hi - lo)
    edges = sorted({lo, hi, *(min(max(float(e), lo), hi) for e in slice_edges)})
    x_edges = np.clip((np.asarray(edges) - c_mid) / e_half, -1.0, 1.0)
    thetas = np.arccos(x_edges)  # decreasing in x

    n = np.arange(1, degree + 1)
    jackson = (
        (degree + 1 - n) * np.cos(np.pi * n / (degree + 1))
        + np.sin(np.pi * n / (degree + 1)) / np.tan(np.pi / (degree + 1))
    ) / (degree + 1)

    coefficient_sets = []
    for i in range(len(edges) - 1):
        th_a, th_b = thetas[i], thetas[i + 1]  # th_a > th_b
        c = np.empty(degree + 1)
        c[0] = (th_a - th_b) / np.pi
        c[1:] = (2.0 / np.pi) * (np.sin(n * th_a) - np.sin(n * th_b)) / n * jackson
        coefficient_sets.append(c)
    window_edges = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    edge_width = float(2.0 * e_half * np.pi / (degree + 1))
    return coefficient_sets, window_edges, edge_width


def chebyshev_apply(hOp, basis, seeds, coefficient_sets, double slaterWeightMin, bounds, verbose=False):
    """Apply a bank of Chebyshev filters to a seed block in one three-term recurrence.

    Computes ``v^s = p_s(H) v`` for every coefficient set simultaneously:

        t_0 = v,   t_1 = H~ t_0,   t_{n+1} = 2 H~ t_n - t_{n-1},
        v^s = sum_n c^{(s)}_n t_n,          H~ = (H - c_mid) / e_half,

    with the rescaling done in the vector algebra (no shifted operator is built). Live
    blocks: ``t_{n-1}, t_n, H t_n`` plus one accumulator per window. The matvec output is
    routed through ``basis.redistribute_block`` when the basis is distributed **or** caps
    growth (``caps_growth`` — a ``_CappedBasisProxy`` bounds the recurrence support, and
    the filtered seeds are then those of the projected ``P H P``, consistent with every
    other capped solve in this code).

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian; restrictions are set from ``basis``.
    basis : Basis or _CappedBasisProxy
        Hosts (and optionally caps) the recurrence support.
    seeds : list of ManyBodyState
        The seed block columns (width ``p``).
    coefficient_sets : sequence of ndarray
        From :func:`partition_of_unity` (or any Chebyshev coefficient vectors).
    slaterWeightMin : float
        Amplitude cutoff pruned from the recurrence blocks each step.
    bounds : tuple of float
        The ``(lower, upper)`` interval the coefficients were built for.
    verbose : bool
        Progress prints every 200 steps (rank-local; no collectives).

    Returns
    -------
    list of list of ManyBodyState
        ``filtered[s]`` = the ``p`` columns of ``p_s(H) v``.
    """
    cdef Py_ssize_t degree = max(len(c) for c in coefficient_sets) - 1
    cdef Py_ssize_t p = len(seeds)
    cdef Py_ssize_t it, s
    cdef double c_mid = 0.5 * (bounds[1] + bounds[0])
    cdef double e_half = 0.5 * (bounds[1] - bounds[0])
    cdef bint mpi = basis is not None and getattr(basis, "is_distributed", False)
    caps = getattr(basis, "caps_growth", False)

    hOp.set_restrictions(basis.restrictions)
    hOp.set_weighted_restrictions(basis.weighted_restrictions)

    eye_p = np.eye(p, dtype=complex)
    t_prev = None
    t_cur = ManyBodyBlockState.from_states(list(seeds))
    accs = [
        block_add_scaled_cy(ManyBodyBlockState.from_states([ManyBodyState() for _ in range(p)]), t_cur, c[0] * eye_p)
        for c in coefficient_sets
    ]
    for it in range(1, degree + 1):
        w = hOp.apply_block(t_cur, slaterWeightMin)
        if mpi or caps:
            w = basis.redistribute_block(w)
        w = block_add_scaled_cy(w, t_cur, (-c_mid) * eye_p)  # (H - c_mid) t_n
        if t_prev is not None:
            t_next = block_add_scaled_cy(t_prev.combine_columns(-eye_p), w, (2.0 / e_half) * eye_p)
        else:
            t_next = w.combine_columns((1.0 / e_half) * eye_p)  # t_1 = H~ t_0
        if slaterWeightMin > 0:
            t_next.prune_rows(slaterWeightMin)
        for s in range(len(coefficient_sets)):
            c = coefficient_sets[s]
            if it < len(c) and abs(c[it]) > 1e-15:
                accs[s] = block_add_scaled_cy(accs[s], t_next, c[it] * eye_p)
        t_prev, t_cur = t_cur, t_next
        if verbose and it % 200 == 0:
            print(f"[chebyshev_apply] step {it}/{degree}, live support {len(t_cur)}", flush=True)

    return [acc.to_states() for acc in accs]
