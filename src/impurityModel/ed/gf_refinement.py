r"""The importance probe of a cap-limited Green's-function unit: which retained determinants carry the least, which
discovered ones would carry the most.

Layer 5: reads a :class:`DistributedSparseMatrix` (``P`` admitted, ``F = K \ P`` discovered), never grows it. For a
shift ``z`` solve ``(z - M_P) X = s`` on ``P`` and score, in units of the Green's function ``G = s^H X``:

* a retained determinant ``D``: removing it changes ``G_ij`` by exactly ``-X_i(D) X_j(D) / R_DD`` with
  ``R = (z - H_P)^-1`` (every cross term cancels; real ``H``, complex-symmetric ``R``), so
  ``s_ret(D) = max_j |X_j(D)|^2 |z - H_DD - Sigma_D|`` with ``R_DD ~ 1 / (z - H_DD - Sigma_D)``;
* a discovered determinant (``X_D = 0``): the Epstein-Nesbet first-order gain
  ``s_cand(D) = max_j |(M_F X)_j(D)|^2 / |z - H_DD - Sigma_D|``.

``Sigma_D = sum_{D' in P, D' != D} |M_DD'|^2 / (z - H_D'D')`` is one entrywise-``|M|^2`` matvec against
``1 / (z - H_D'D')`` over the ``P`` columns, so the two scores share one denominator and are commensurable.
The maximum over the shifts is kept. Exact for real ``H``: with spin-orbit coupling ``R`` is not
complex-symmetric and a second solve at ``z*`` would be needed (declined by the caller).
"""

from dataclasses import dataclass, field

import numpy as np
from mpi4py import MPI


@dataclass
class ImportanceScores:
    """Per-rank scores. ``s_ret`` is indexed by P slot (``-inf`` on a dead slot), ``s_cand`` by K slot (``-inf``
    on a determinant that is in ``P``). ``solutions`` are the converged ``X`` per shift (a warm start for the next
    round); ``residuals`` the worst relative residual over the columns of each shift."""

    s_ret: np.ndarray
    s_cand: np.ndarray
    solutions: list = field(default_factory=list)
    residuals: list = field(default_factory=list)
    iterations: list = field(default_factory=list)


def _dot(comm, a, b):
    """``a^H b`` summed over the ranks (a fixed-shape ``Allreduce``, never the lowercase one on an array)."""
    value = np.array([np.vdot(a, b)], dtype=complex)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, value, op=MPI.SUM)
    return value[0]


def solve_shifted(apply, z, b, comm=None, x0=None, rtol=1e-6, maxit=500):
    """BiCGSTAB for ``(z - A) x = b``, ``apply(v)`` giving ``A v`` for this rank's rows (``v`` of shape ``(n,)``).

    Returns ``(x, iterations, relative residual)``. The loop count and the breakdown test are decided on
    allreduced scalars, so every rank leaves on the same iteration. **Collective** on a distributed ``b``.
    """

    def op(v):
        return z * v - apply(v)

    b = np.asarray(b, dtype=complex)
    bnorm = np.sqrt(_dot(comm, b, b).real)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=complex)
    if bnorm == 0.0:
        return np.zeros_like(b), 0, 0.0
    r = b - op(x) if x0 is not None else b.copy()
    r_hat = r.copy()
    rho = alpha = omega = 1.0 + 0j
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    rel = np.sqrt(_dot(comm, r, r).real) / bnorm
    it = 0
    while rel > rtol and it < maxit:
        rho_new = _dot(comm, r_hat, r)
        if abs(rho_new) < 1e-300 or abs(omega) < 1e-300:
            break
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = op(p)
        denom = _dot(comm, r_hat, v)
        if abs(denom) < 1e-300:
            break
        alpha = rho_new / denom
        s = r - alpha * v
        if np.sqrt(_dot(comm, s, s).real) / bnorm <= rtol:
            x = x + alpha * p
            r = s
            rel = np.sqrt(_dot(comm, r, r).real) / bnorm
            it += 1
            break
        t = op(s)
        tt = _dot(comm, t, t)
        if abs(tt) < 1e-300:
            break
        omega = _dot(comm, t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
        rel = np.sqrt(_dot(comm, r, r).real) / bnorm
        it += 1
    return x, it, float(rel)


def importance_scores(matrix, seeds, zs, rtol=1e-6, maxit=500, x0s=None, first_order=True):
    r"""``s_ret`` and ``s_cand`` (see the module docstring) of ``matrix`` for the seed block ``seeds``
    (``(n_p_local, m)`` on the P slots) at the shifts ``zs``, maximised over the shifts and the columns.

    ``x0s`` (optional, one ``(n_p_local, m)`` block per shift) warm-starts the solves. ``first_order=False``
    drops ``Sigma_D`` (the bare ``|z - H_DD|``), kept only so a test can show the self-energy matters.
    **Collective.**
    """
    seeds = np.ascontiguousarray(seeds, dtype=complex)
    n_p, n_k = matrix.n_p_local, matrix.n_k_local
    if seeds.shape[0] != n_p:
        raise ValueError(f"seed block has {seeds.shape[0]} rows, the matrix has {n_p} P slots")
    alive = matrix.alive().astype(bool)
    p_kslot = matrix.p_to_k_slot()
    diag_p = matrix.diagonal()
    diag_k = matrix.k_diagonal()
    in_p = np.zeros(n_k, dtype=bool)
    in_p[p_kslot[alive]] = True
    comm = matrix.comm
    plan_p = matrix.exchange_plan(1, "importance", rows="P")
    plan_all = matrix.exchange_plan(1, "importance", rows="all")
    plan_all_m = matrix.exchange_plan(max(seeds.shape[1], 1), "importance", rows="all")
    s_ret = np.full(n_p, -np.inf)
    s_cand = np.full(n_k, -np.inf)
    out = ImportanceScores(s_ret, s_cand)
    for iz, shift in enumerate(zs):
        z = complex(shift)
        if first_order:
            inv_p = np.where(alive, 1.0 / (z - diag_p), 0.0)
            sigma = matrix.matvec(inv_p[:, None], rows="all", plan=plan_all, squared=True)[:, 0]
            # A retained determinant's own column is not among its neighbours.
            sigma[p_kslot[alive]] -= (np.abs(diag_p[alive]) ** 2) * inv_p[alive]
        else:
            sigma = np.zeros(n_k, dtype=complex)
        denom = np.abs(z - diag_k - sigma)
        denom = np.maximum(denom, 1e-300)
        X = np.zeros_like(seeds)
        worst, iters = 0.0, 0
        for j in range(seeds.shape[1]):
            x0 = None if x0s is None else x0s[iz][:, j]
            x, it, rel = solve_shifted(
                lambda v: matrix.matvec(v[:, None], rows="P", plan=plan_p)[:, 0],
                z,
                np.where(alive, seeds[:, j], 0.0),
                comm,
                x0=x0,
                rtol=rtol,
                maxit=maxit,
            )
            X[:, j] = np.where(alive, x, 0.0)
            worst, iters = max(worst, rel), max(iters, it)
        flow = matrix.matvec(X, rows="all", plan=plan_all_m)
        s_cand = np.maximum(s_cand, np.where(in_p, -np.inf, (np.abs(flow) ** 2).max(axis=1) / denom))
        ret = (np.abs(X) ** 2).max(axis=1) * denom[p_kslot]
        s_ret = np.maximum(s_ret, np.where(alive, ret, -np.inf))
        out.solutions.append(X)
        out.residuals.append(worst)
        out.iterations.append(iters)
    out.s_ret, out.s_cand = s_ret, s_cand
    return out


def seed_block_on_p_slots(matrix, states):
    """The seed states (this rank's determinants, as a list of ``ManyBodyState`` columns) as the ``(n_p_local, m)``
    block on the matrix's P slots; every seed key must be in ``P``. Rank-local."""
    block = np.zeros((matrix.n_p_local, len(states)), dtype=complex)
    k_to_p = matrix.k_to_p_slot()
    for j, state in enumerate(states):
        keys = list(state.keys())
        if not keys:
            continue
        pslots = k_to_p[matrix.key_slots(keys)]
        if (pslots < 0).any():
            raise ValueError("a seed determinant is not in the matrix's P")
        block[pslots, j] = np.asarray(state)[:, 0]
    return block


def probe_unit(matrix, support, seeds, hop, basis, info, zs=(), rtol=1e-6):
    """The ``unit_probe`` of the GF engine: one importance pass on a capped unit's discovered matrix.

    Returns a small replicated record (nothing proportional to ``|K|``): the shifts, the iteration counts, the
    worst residual and the top of each ranking. **Collective.**
    """
    comm = matrix.comm
    scores = importance_scores(matrix, seed_block_on_p_slots(matrix, seeds), zs, rtol=rtol)
    top_cand = float(scores.s_cand.max()) if scores.s_cand.size else -np.inf
    low_ret = float(scores.s_ret[np.isfinite(scores.s_ret)].min()) if np.isfinite(scores.s_ret).any() else np.inf
    if comm is not None:
        top_cand = comm.allreduce(top_cand, op=MPI.MAX)
        low_ret = comm.allreduce(low_ret, op=MPI.MIN)
    return {
        "zs": [complex(z) for z in zs],
        "iterations": list(scores.iterations),
        "residual": max(scores.residuals) if scores.residuals else 0.0,
        "top_s_cand": top_cand,
        "low_s_ret": low_ret,
    }
