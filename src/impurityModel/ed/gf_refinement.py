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

import time
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


def _cdots(comm, a, b):
    """Column-wise ``a_j^H b_j`` summed over the ranks: one fixed-shape ``Allreduce`` of ``m`` numbers."""
    value = np.einsum("ij,ij->j", a.conj(), b)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, value, op=MPI.SUM)
    return value


#: Columns solved together: the work blocks are ``8 m`` dense columns.
SOLVE_BLOCK_COLUMNS = 6


def solve_shifted_block(apply, z, b, comm=None, x0=None, rtol=1e-6, maxit=500):
    """BiCGSTAB for ``(z - A) X = B`` with the columns of ``B`` solved together: one width-``m`` ``apply`` per
    half-step (a wide matvec costs little more than a width-1 one) and per-column scalars, so a column that has
    converged or broken down stops updating while the others go on. ``apply(V)`` gives ``A V`` for this rank's rows
    (``V`` of shape ``(n, m)``). Returns ``(X, iterations, relative residuals per column)``. Every rank leaves on the
    same iteration (the scalars are allreduced). **Collective.**"""
    b = np.asarray(b, dtype=complex)
    m = b.shape[1]

    def op(v):
        return z * v - apply(v)

    bnorm = np.sqrt(_cdots(comm, b, b).real)
    done = bnorm == 0.0
    safe = np.where(done, 1.0, bnorm)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=complex)
    r = b - op(x) if x0 is not None else b.copy()
    r_hat = r.copy()
    rho = np.ones(m, dtype=complex)
    alpha = np.ones(m, dtype=complex)
    omega = np.ones(m, dtype=complex)
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    rel = np.sqrt(_cdots(comm, r, r).real) / safe
    rel[done] = 0.0
    x[:, done] = 0.0
    active = ~done & (rel > rtol)
    it = 0
    while active.any() and it < maxit:
        rho_new = _cdots(comm, r_hat, r)
        active &= ~((np.abs(rho_new) < 1e-300) | (np.abs(omega) < 1e-300))
        if not active.any():
            break
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = op(p)
        denom = _cdots(comm, r_hat, v)
        active &= ~(np.abs(denom) < 1e-300)
        alpha = rho_new / np.where(np.abs(denom) < 1e-300, 1.0, denom)
        s = r - alpha * v
        half = active & (np.sqrt(_cdots(comm, s, s).real) / safe <= rtol)
        t = op(s)
        tt = _cdots(comm, t, t)
        broke = np.abs(tt) < 1e-300
        omega_new = _cdots(comm, t, s) / np.where(broke, 1.0, tt)
        full = active & ~half & ~broke
        x = np.where(half, x + alpha * p, np.where(full, x + alpha * p + omega_new * s, x))
        r = np.where(half, s, np.where(full, s - omega_new * t, r))
        rho = np.where(full, rho_new, rho)
        omega = np.where(full, omega_new, omega)
        active &= ~broke
        it += 1
        rel = np.sqrt(_cdots(comm, r, r).real) / safe
        rel[done] = 0.0
        active &= rel > rtol
    return x, it, rel


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
    block = min(max(seeds.shape[1], 1), SOLVE_BLOCK_COLUMNS)
    plan_p = matrix.exchange_plan(block, "importance", rows="P")
    plan_p_tail = (
        matrix.exchange_plan(seeds.shape[1] % block, "importance", rows="P") if seeds.shape[1] % block else None
    )
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
        for j0 in range(0, seeds.shape[1], block):
            cols = slice(j0, min(j0 + block, seeds.shape[1]))
            plan = plan_p if cols.stop - cols.start == block else plan_p_tail
            x, it, rel = solve_shifted_block(
                lambda v, plan=plan: matrix.matvec(np.ascontiguousarray(v), rows="P", plan=plan),
                z,
                np.where(alive[:, None], seeds[:, cols], 0.0),
                comm,
                x0=None if x0s is None else x0s[iz][:, cols],
                rtol=rtol,
                maxit=maxit,
            )
            X[:, cols] = np.where(alive[:, None], x, 0.0)
            worst, iters = max(worst, float(rel.max())), max(iters, it)
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


# --- swap refinement of a capped unit ------------------------------------------------------------------------

#: Defaults of the refinement (beside ``GF_METHODS``: not environment knobs).
REFINE_TOL = 1e-3
REFINE_SWAP = 0.10
#: The mass of the first H-shell that may be left swappable: the heaviest rows covering ``1 - this`` of ``|H s|^2``
#: are exempt from removal (they carry the moments of G and with them the self-energy tail).
FIRST_SHELL_EXEMPT_TAIL = 1e-6


@dataclass(frozen=True)
class RefineSpec:
    """``rounds`` swap rounds at most; stop when the estimated remaining first-order gain (the summed ``s_cand`` of the
    ``swap`` fraction of the cap that would be swapped in) is below ``tol`` times the largest ``|G|`` on the shifts;
    each round swaps at most ``swap`` of the retained determinants."""

    rounds: int = 0
    tol: float = REFINE_TOL
    swap: float = REFINE_SWAP


@dataclass
class RefineOutcome:
    """What a refinement left: ``result`` is the new recurrence ``(alphas, betas, widths, status, q_last)`` or ``None``
    when the unit keeps round 0's (declined, nothing swapped, or it fell back); ``record`` a replicated summary."""

    result: object
    record: dict


class _LazyKeys:
    """``keys[i]`` the determinant at ``kslots[i]``, made on demand (the tie-break asks for a boundary group only)."""

    def __init__(self, matrix, kslots):
        self._matrix, self._kslots = matrix, kslots

    def __len__(self):
        return len(self._kslots)

    def __getitem__(self, i):
        return self._matrix.keys_of([self._kslots[i]])[0]


def _allreduce(comm, value, op):
    value = np.array([value], dtype=float)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, value, op=op)
    return float(value[0])


def matrix_is_real(matrix):
    """Whether every stored element is real (the score identities need real ``H``). **Collective.**"""
    worst = 0.0
    for _c0, _indptr, _packed, data in matrix._segments:
        if data.size:
            worst = max(worst, float(np.abs(np.asarray(data).imag).max()))
    return _allreduce(matrix.comm, worst, MPI.MAX) <= 1e-12


def pick_shifts(g_of, eval_meshes, n_peaks=3, n_matsubara=2):
    """``(Z, Z_holdout)``: the shifts the scores are maximised over and a disjoint set the change is judged on.

    ``g_of(zs)`` gives the round-0 continued-fraction ``G`` of shape ``(len(zs), n, n)``. A mesh whose imaginary part
    varies more than its real part is the Matsubara axis (``Z`` takes its ``n_matsubara`` points nearest the real
    axis: the floor for weight-carrying rows away from a pole); on a real axis ``Z`` takes the ``n_peaks`` largest
    local maxima of ``sum_i |Im G_ii|`` and the point of steepest change, the holdout the next peaks. Replicated: a
    function of replicated data only."""
    z_pick, z_hold = [], []
    for axis in eval_meshes or ():
        mesh = np.asarray(axis, dtype=complex)
        if mesh.size == 0:
            continue
        if np.ptp(mesh.imag) > np.ptp(mesh.real):
            order = np.argsort(np.abs(mesh.imag), kind="stable")
            z_pick += [complex(z) for z in mesh[order[:n_matsubara]]]
            continue
        order = np.argsort(mesh.real, kind="stable")
        zs = mesh[order]
        g = g_of(zs)
        weight = np.abs(np.einsum("kii->k", g.imag))
        local_max = [
            i
            for i in range(len(zs))
            if (i == 0 or weight[i] >= weight[i - 1]) and (i == len(zs) - 1 or weight[i] > weight[i + 1])
        ]
        ranked = sorted(local_max, key=lambda i: -weight[i])
        steep = int(np.argmax(np.abs(np.gradient(weight)))) if len(zs) > 1 else 0
        chosen = list(dict.fromkeys(ranked[:n_peaks] + [steep]))
        rest = [i for i in ranked[n_peaks:] + sorted(range(len(zs)), key=lambda i: -weight[i]) if i not in chosen]
        z_pick += [complex(zs[i]) for i in chosen]
        z_hold += [complex(zs[i]) for i in list(dict.fromkeys(rest))[:n_peaks]]
    return list(dict.fromkeys(z_pick)), list(dict.fromkeys(z for z in z_hold if z not in z_pick))


def _exempt_pslots(matrix, seed_keys, seed_block, extra_keys):
    """Boolean over the P slots: the seeds, the heaviest first-shell rows (covering all but
    :data:`FIRST_SHELL_EXEMPT_TAIL` of ``|H s|^2``) and ``extra_keys`` (last round's admissions). **Collective.**"""
    from impurityModel.ed.manybody_basis import collective_mass_cutoff

    n_p = matrix.n_p_local
    exempt = np.zeros(n_p, dtype=bool)
    k_to_p = matrix.k_to_p_slot()
    for keys in (seed_keys, extra_keys):
        if len(keys):
            pslots = k_to_p[matrix.key_slots(keys)]
            exempt[pslots[pslots >= 0]] = True
    flow = matrix.matvec(seed_block, rows="P", plan=matrix.exchange_plan(seed_block.shape[1], "refine_shell", rows="P"))
    weight = (np.abs(flow) ** 2).max(axis=1)
    weight[exempt & (np.abs(seed_block).max(axis=1) > 0)] = 0.0  # a seed row's own amplitude is not first-shell weight
    total = _allreduce(matrix.comm, float(weight.sum()), MPI.SUM)
    if total > 0.0:
        cutoff = collective_mass_cutoff(weight, FIRST_SHELL_EXEMPT_TAIL * total, matrix.comm)
        exempt |= weight > cutoff
    return exempt


def _select_swap(matrix, scores, exempt, k_max):
    """``(retire_pslots, admit_keys)``: the ``k <= k_max`` lowest-``s_ret`` retained non-exempt determinants and the
    ``k`` highest-``s_cand`` discovered ones, ``k`` the largest (halving from ``k_max``) for which the ``k``-th best
    candidate still beats the ``k``-th worst retained. Rank-count-invariant selections
    (:func:`~impurityModel.ed.manybody_basis.collective_top_k_mask`). Also returns the summed ``s_cand`` of the first
    ``k_max`` candidates. **Collective.**"""
    from impurityModel.ed.manybody_basis import collective_top_k_mask

    comm = matrix.comm
    alive = matrix.alive().astype(bool)
    p_kslot = matrix.p_to_k_slot()
    in_p = np.zeros(matrix.n_k_local, dtype=bool)
    in_p[p_kslot[alive]] = True
    f_slots = np.flatnonzero(~in_p)
    cand = np.maximum(scores.s_cand[f_slots], 0.0)
    cand = np.where(np.isfinite(cand), cand, 0.0)
    swappable = alive & ~exempt
    r_slots = np.flatnonzero(swappable)
    s_ret = np.where(np.isfinite(scores.s_ret[r_slots]), scores.s_ret[r_slots], 0.0)
    smax = _allreduce(comm, float(s_ret.max()) if s_ret.size else 0.0, MPI.MAX)
    low = 1.0 / (1.0 + s_ret / smax) if smax > 0.0 else np.ones_like(s_ret)
    cand_keys, low_keys = _LazyKeys(matrix, f_slots), _LazyKeys(matrix, p_kslot[r_slots])
    top_cand = collective_top_k_mask(cand, cand_keys, k_max, comm)
    gain = _allreduce(comm, float(cand[top_cand].sum()), MPI.SUM)
    k = k_max
    while k >= 1:
        m_c = collective_top_k_mask(cand, cand_keys, k, comm)
        m_r = collective_top_k_mask(low, low_keys, k, comm)
        n_c = int(_allreduce(comm, float(m_c.sum()), MPI.SUM))
        n_r = int(_allreduce(comm, float(m_r.sum()), MPI.SUM))
        kk = min(n_c, n_r)
        if kk == 0:
            break
        if kk < k:
            m_c = collective_top_k_mask(cand, cand_keys, kk, comm)
            m_r = collective_top_k_mask(low, low_keys, kk, comm)
        a_k = _allreduce(comm, float(cand[m_c].min()) if m_c.any() else np.inf, MPI.MIN)
        r_k = _allreduce(comm, float(s_ret[m_r].max()) if m_r.any() else -np.inf, MPI.MAX)
        if a_k >= r_k:
            return r_slots[m_r], matrix.keys_of(f_slots[m_c]), gain
        k = kk // 2
    return np.zeros(0, dtype=np.int64), [], gain


def _g_of_solutions(comm, seeds, solutions):
    """``seeds^H X`` per shift, ``(len(zs), m, m)``, summed over the ranks."""
    out = np.zeros((len(solutions), seeds.shape[1], seeds.shape[1]), dtype=complex)
    for i, x in enumerate(solutions):
        out[i] = seeds.conj().T @ x
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, out, op=MPI.SUM)
    return out


def refine_frozen_unit(matrix, spec, seed_keys, seed_q0, first, rerun, g_cf, eval_meshes, seed_block, rtol=1e-6):
    """Swap refinement of a unit the cap froze, on its discovered matrix. **Collective** (every rank, same arguments).

    ``first`` is round 0's ``(alphas, betas, widths, status, q_last)``; ``rerun(q0)`` the frozen recurrence from a
    seed block on the matrix's current P slots; ``g_cf(alphas_widths_result, zs)`` the continued-fraction ``G`` of a
    result on the shifts ``zs``; ``seed_block(matrix, seed_keys, q0)`` the seed block on the current P slots.

    Round ``n`` scores the retained and discovered determinants (:func:`importance_scores`, warm-started from the
    previous round's solutions), then swaps the least important retained against the most important discovered ones
    (:meth:`DistributedSparseMatrix.set_p_keys`: the cap is unchanged). It stops when the remaining estimated gain is
    below ``spec.tol`` times the largest ``|G|`` on the shifts, when no swap gains, or after ``spec.rounds`` swaps; the
    recurrence is then run once on the final P. A refinement whose final estimated gain is larger than round 0's falls
    back to round 0's P and result. The record's ``dG_*`` are ``max|G_final - G_0| / max|G_0|`` of the continued
    fractions on the scoring shifts and on the held-out ones."""
    comm = matrix.comm
    t0 = time.perf_counter()
    record = {"declined": None, "rounds": 0, "swapped": [], "gain": [], "tol": spec.tol, "swap": spec.swap}
    if not matrix_is_real(matrix):
        record["declined"] = "complex H"
        return RefineOutcome(None, record)
    zs, z_hold = pick_shifts(lambda z: g_cf(first, z), eval_meshes)
    if not zs:
        record["declined"] = "no evaluation mesh"
        return RefineOutcome(None, record)
    g0 = g_cf(first, zs)
    scale = float(np.abs(g0).max())
    g0_hold = g_cf(first, z_hold) if z_hold else None
    keys0 = [key for key, a in zip(matrix.p_keys(), matrix.alive()) if a]
    n_alive = int(_allreduce(comm, float(matrix.n_alive_local), MPI.SUM))
    k_max = max(1, int(spec.swap * n_alive))
    warm, extra, first_gain = None, [], None
    swapped_total = 0
    for round_ in range(spec.rounds + 1):
        t_round = time.perf_counter()
        seeds = seed_block(matrix, seed_keys, seed_q0)
        scores = importance_scores(matrix, seeds, zs, rtol=rtol, x0s=warm)
        exempt = _exempt_pslots(matrix, seed_keys, seeds, extra)
        retire, admit, gain = _select_swap(matrix, scores, exempt, k_max)
        record["gain"].append(gain)
        record.setdefault("pass_seconds", []).append(time.perf_counter() - t_round)
        record.setdefault("iterations", []).append(max(scores.iterations) if scores.iterations else 0)
        if first_gain is None:
            first_gain = gain
        n_swap = int(_allreduce(comm, float(len(retire)), MPI.SUM))
        record["swapped"].append(n_swap)
        if gain <= spec.tol * scale or n_swap == 0 or round_ == spec.rounds:
            break
        old_k = matrix.p_to_k_slot()[matrix.alive().astype(bool)]
        old_x = [x[matrix.alive().astype(bool)] for x in scores.solutions]
        keep = np.ones(matrix.n_p_local, dtype=bool)
        keep[retire] = False
        keep &= matrix.alive().astype(bool)
        keys = [key for key, k_ in zip(matrix.p_keys(), keep) if k_] + list(admit)
        matrix.set_p_keys(keys)
        new_p_k = matrix.p_to_k_slot()
        warm = []
        for x in old_x:
            xk = np.zeros((matrix.n_k_local, x.shape[1]), dtype=complex)
            xk[old_k] = x
            warm.append(xk[new_p_k])
        extra = list(admit)
        swapped_total += n_swap
        record["rounds"] = round_ + 1
    record["settled"] = bool(gain <= spec.tol * scale)
    if swapped_total == 0:
        record["seconds"] = time.perf_counter() - t0
        return RefineOutcome(None, record)
    if record["gain"][-1] > first_gain:
        matrix.set_p_keys(keys0)
        record["fell_back"] = True
        record["seconds"] = time.perf_counter() - t0
        return RefineOutcome(None, record)
    result = rerun(seed_block(matrix, seed_keys, seed_q0))
    dg = np.abs(g_cf(result, zs) - g0).max() / scale if scale > 0 else 0.0
    record["dG_scoring"] = float(dg)
    if g0_hold is not None:
        record["dG_holdout"] = float(np.abs(g_cf(result, z_hold) - g0_hold).max() / max(np.abs(g0_hold).max(), 1e-300))
    record["seconds"] = time.perf_counter() - t0
    return RefineOutcome(result, record)
