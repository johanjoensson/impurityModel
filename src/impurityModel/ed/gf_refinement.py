r"""The importance probe of a cap-limited Green's-function unit: which retained determinants carry the least, which
discovered ones would carry the most.

Layer 5: reads a :class:`DistributedSparseMatrix` (``P`` admitted, ``F = K \ P`` discovered), never grows it. For a
shift ``z`` take ``X = (z - M_P)^-1 s`` on ``P`` (the unit's own block-Lanczos recurrence accumulates it:
:class:`~impurityModel.ed.gf_shift_recycling.ShiftedGalerkin`) and score, in units of the Green's function
``G = s^H X``:

* a retained determinant ``D``: removing it changes ``G_ij`` by exactly ``-X_i(D) X_j(D) / R_DD`` with
  ``R = (z - H_P)^-1`` (every cross term cancels; Hermitian ``H``), so
  ``s_ret(D) = max_j |X_j(D)|^2 |z - H_DD - Sigma_D|`` with ``R_DD ~ 1 / (z - H_DD - Sigma_D)``;
* a discovered determinant (``X_D = 0``): the Epstein-Nesbet first-order gain
  ``s_cand(D) = max_j |(M_F X)_j(D)|^2 / |z - H_DD - Sigma_D|``.

``Sigma_D = sum_{D' in P, D' != D} |M_DD'|^2 / (z - H_D'D')`` is one entrywise-``|M|^2`` matvec against
``1 / (z - H_D'D')`` over the ``P`` columns, so the two scores share one denominator and are commensurable.
The maximum over the shifts is kept. The removal identity is ``-conj(Y_i(D)) X_j(D) / R_DD`` with ``X = R(z) s``
and ``Y = R(z*) s``: when ``H`` is real and the seed columns real up to a phase ``|Y| = |X|`` and one set of solutions
serves; otherwise the conjugate shifts ride along in the same recurrence. Only the Hermitian structure of ``H`` is
used, so complex ``H`` (spin-orbit) is scored like real.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from mpi4py import MPI


@dataclass
class ImportanceScores:
    """Per-rank scores. ``s_ret`` is indexed by P slot (``-inf`` on a dead slot), ``s_cand`` by K slot (``-inf``
    on a determinant that is in ``P``)."""

    s_ret: np.ndarray
    s_cand: np.ndarray


def importance_scores(matrix, seeds, zs, xs, ys=None, first_order=True, weights=None):
    r"""``s_ret`` and ``s_cand`` (see the module docstring) of ``matrix`` for the seed block ``seeds``
    (``(n_p_local, m)`` on the P slots) at the shifts ``zs``, maximised over the shifts and the columns.

    ``xs[i]`` is ``X = R(z_i) s`` on the P slots (the unit's own recurrence accumulates it:
    :class:`~impurityModel.ed.gf_shift_recycling.ShiftedGalerkin`), ``ys[i]`` is ``Y = R(z_i^*) s`` and is needed
    exactly when ``H`` is complex or a seed column is not real up to a phase: removing ``D`` changes
    ``G_ij = s_i^H R s_j`` by ``-conj(Y_i(D)) X_j(D) / R_DD`` (``R(z)^H = R(z*)`` for Hermitian ``H``), the scores read
    ``max_i |Y_i| max_j |X_j|``, and without ``ys`` ``Y = X`` is taken. ``first_order=False`` drops ``Sigma_D``
    (the bare ``|z - H_DD|``), kept only so a test can show the self-energy matters.
    ``weights`` (one per shift, replicated) multiply each shift's two scores before the maximum: the self-energy moves by ``G^-1 dG G^-1``, so a shift where
    ``G^-1`` is large counts for more. **Collective.**"""
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
    # A plan holds a cached dist-graph communicator that a later `exchange_plan` may evict (and free): build each
    # one right before the matvecs that use it, never ahead of other plans' construction.

    def flow_of(V):
        plan_all_m = matrix.exchange_plan(max(seeds.shape[1], 1), "importance", rows="all")
        return np.abs(matrix.matvec(V, rows="all", plan=plan_all_m)).max(axis=1)

    s_ret = np.full(n_p, -np.inf)
    s_cand = np.full(n_k, -np.inf)
    out = ImportanceScores(s_ret, s_cand)
    for iz, shift in enumerate(zs):
        z = complex(shift)
        if first_order:
            inv_p = np.where(alive, 1.0 / (z - diag_p), 0.0)
            plan_all = matrix.exchange_plan(1, "importance", rows="all")
            sigma = matrix.matvec(inv_p[:, None], rows="all", plan=plan_all, squared=True)[:, 0]
            # A retained determinant's own column is not among its neighbours.
            sigma[p_kslot[alive]] -= (np.abs(diag_p[alive]) ** 2) * inv_p[alive]
        else:
            sigma = np.zeros(n_k, dtype=complex)
        denom = np.abs(z - diag_k - sigma)
        denom = np.maximum(denom, 1e-300)
        X = np.where(alive[:, None], xs[iz], 0.0)
        ax, flow_x = np.abs(X).max(axis=1), flow_of(X)
        ay, flow_y = ax, flow_x
        if ys is not None:
            Y = np.where(alive[:, None], ys[iz], 0.0)
            ay, flow_y = np.abs(Y).max(axis=1), flow_of(Y)
        wz = 1.0 if weights is None else float(weights[iz])
        s_cand = np.maximum(s_cand, np.where(in_p, -np.inf, wz * flow_x * flow_y / denom))
        s_ret = np.maximum(s_ret, np.where(alive, wz * ax * ay * denom[p_kslot], -np.inf))
    out.s_ret, out.s_cand = s_ret, s_cand
    return out


def _pslots_of(matrix, keys):
    """The P slot of each key, ``-1`` for a key that is not in K or not in P (a ``-1`` K slot would wrap to the
    last entry of ``k_to_p_slot``)."""
    slots = np.asarray(matrix.key_slots(keys))
    k_to_p = matrix.k_to_p_slot()
    return np.where(slots >= 0, k_to_p[np.maximum(slots, 0)], -1)


def seed_block_on_p_slots(matrix, states):
    """The seed states (this rank's determinants, as a list of ``ManyBodyState`` columns) as the ``(n_p_local, m)``
    block on the matrix's P slots; every seed key must be in ``P``. Rank-local."""
    block = np.zeros((matrix.n_p_local, len(states)), dtype=complex)
    for j, state in enumerate(states):
        keys = list(state.keys())
        if not keys:
            continue
        pslots = _pslots_of(matrix, keys)
        if (pslots < 0).any():
            raise ValueError("a seed determinant is not in the matrix's P")
        block[pslots, j] = np.asarray(state)[:, 0]
    return block


# --- swap refinement of a capped unit ------------------------------------------------------------------------

#: Defaults of the refinement (beside ``GF_METHODS``: not environment knobs).
REFINE_TOL = 1e-3
REFINE_SWAP = 0.10
#: The residual the recurrence-fed scoring solutions must reach, and whether the shifts are weighted by ``|G_0^-1|^2``.
#: Harness switches (``freeze_replay`` sets them identically on every rank), not user options, until measured: a
#: ``RefineSpec`` field left ``None`` reads the constant when the unit is refined.
REFINE_STOP_RTOL = 1e-4
REFINE_WEIGHTED = False


#: The mass of the first H-shell that may be left swappable: the heaviest rows covering ``1 - this`` of ``|H s|^2``
#: are exempt from removal (they carry the moments of G and with them the self-energy tail).
FIRST_SHELL_EXEMPT_TAIL = 1e-6


@dataclass(frozen=True)
class RefineSpec:
    """``rounds`` swap rounds at most; stop when the estimated remaining first-order gain (the summed ``s_cand`` of the
    ``swap`` fraction of the cap that would be swapped in) is below ``tol`` times the largest ``|G|`` on the shifts;
    each round swaps at most ``swap`` of the retained determinants. The scoring solutions come out of the unit's own
    recurrence at ``shifts`` (``((z, weight), ...)`` in the frame the recurrence's coefficients live in; filled per
    unit from :func:`g0_zero_shifts`) and the recurrence runs until their residual is below ``stop_rtol`` (``None``:
    ``REFINE_STOP_RTOL``). ``weighted`` multiplies each shift's scores by its weight (``None``:
    ``REFINE_WEIGHTED``). ``tables`` holds one
    :func:`g0_zero_shifts` result per Green's-function block, ``((z, weight), ...)`` in the unsigned frame the caller
    evaluates ``G`` in; :func:`unit_refine_spec` turns it into the ``shifts`` of one unit."""

    rounds: int = 0
    tol: float = REFINE_TOL
    swap: float = REFINE_SWAP
    stop_rtol: Optional[float] = None
    weighted: Optional[bool] = None
    shifts: tuple = ()
    tables: tuple = ()


def unit_refine_spec(spec, block_i, side_i, es):
    """``spec`` with ``shifts`` the unit's: block ``block_i``'s table signed for the spectral side (``side_i`` 0:
    addition, else removal: the sign multiplies the real part *and* the broadening, as
    :func:`~impurityModel.ed.gf_convergence._gf_signed_axes` has it) and offset by each thermal energy the unit
    stacks, the frame the recurrence's coefficients are in. A spec without a table for the block keeps no shifts
    (the unit declines with its reason)."""
    from dataclasses import replace

    if not spec.tables or block_i >= len(spec.tables):
        return replace(spec, shifts=())
    sign = 1.0 if side_i == 0 else -1.0
    shifts = tuple((sign * z + e, w) for e in es for z, w in spec.tables[block_i])
    return replace(spec, shifts=shifts)


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


def seeds_real_up_to_a_phase(matrix, block, tol=1e-10):
    """Whether every column of the seed block on the P slots is real up to one global phase (``|s^T s| = s^H s``).
    The removal identity reads ``X_i(D) = s_i^H R e_D`` off the solution of column ``i`` only then. **Collective.**"""
    gram = np.stack([(block * block).sum(axis=0), (np.abs(block) ** 2).sum(axis=0)]).astype(complex)
    if matrix.comm is not None and matrix.comm.size > 1:
        matrix.comm.Allreduce(MPI.IN_PLACE, gram, op=MPI.SUM)
    return bool(np.all(np.abs(gram[0]) >= (1.0 - tol) * gram[1].real))


def pick_shifts(g_of, eval_meshes, n_peaks=3, n_matsubara=2):
    """``(Z, Z_holdout)``: the shifts the scores are maximised over and a disjoint set the change is judged on.

    ``g_of(zs)`` gives the round-0 continued-fraction ``G`` of shape ``(len(zs), n, n)``. A mesh whose imaginary part
    varies more than its real part is the Matsubara axis (``Z`` takes its ``n_matsubara`` points nearest the real
    axis: the floor for weight-carrying rows away from a pole); on a real axis ``Z`` takes the ``n_peaks`` largest
    local maxima of ``sum_i |Im G_ii|`` and the point of steepest change, the holdout the next peaks. Replicated: a
    function of replicated data only.
    """
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


REFINE_MAX_SHIFTS = 8


def g0_zero_shifts(g0_inv_of, omega_mesh, matsubara_mesh, delta, max_shifts=None, n_matsubara=2):
    r"""The scoring shifts of one Green's-function block, chosen before its recurrence runs, with their weights.

    ``g0_inv_of(zs)`` gives ``G_0^-1(z) = z - h_corr - Delta(z)`` of shape ``(len(zs), n, n)``. A change ``dG`` moves
    the self-energy by ``G^-1 dG G^-1``, which is largest where ``G^-1`` is: on the real mesh (at ``omega + i delta``)
    the local maxima of ``||G_0^-1||_2`` (the zeros of ``G_0``, i.e. the bath poles the self-energy is fragile at; at
    most ``max_shifts - 1 - n_matsubara`` of them, heaviest first), the point nearest ``omega = 0`` (the
    quasi-particle point of a metal), and the ``n_matsubara`` Matsubara points nearest the real axis (the floor for
    weight-carrying rows away from every pole). Replicated if ``g0_inv_of`` and the meshes are.

    Returns ``(zs, weights)``: unsigned complex shifts (``omega + i delta``, ``i omega_n``; ``matsubara_mesh`` holds
    the ``i omega_n``) and
    ``w(z) = ||G_0^-1(z)||^2``, normalised to a maximum of 1. The caller signs them and adds each thermal energy
    (the frame :func:`~impurityModel.ed.gf_convergence._gf_signed_axes` fixes)."""
    max_shifts = REFINE_MAX_SHIFTS if max_shifts is None else max_shifts
    zs, norms = [], []

    def norm_at(points):
        g = np.asarray(g0_inv_of(np.asarray(points, dtype=complex)))
        return np.array([np.linalg.norm(m, 2) for m in g])

    if matsubara_mesh is not None and len(matsubara_mesh):
        pts = np.asarray(matsubara_mesh, dtype=complex)  # the mesh is the imaginary frequencies i omega_n themselves
        order = np.argsort(np.abs(pts.imag), kind="stable")[:n_matsubara]
        zs += [complex(pts[i]) for i in order]
        norms += list(norm_at(pts[order]))
    if omega_mesh is not None and len(omega_mesh):
        w = np.asarray(omega_mesh, dtype=float)
        order = np.argsort(w, kind="stable")
        pts = w[order] + 1j * delta
        nrm = norm_at(pts)
        peaks = [i for i in range(1, len(pts) - 1) if nrm[i] >= nrm[i - 1] and nrm[i] > nrm[i + 1]]
        peaks = sorted(peaks, key=lambda i: -nrm[i])[: max(max_shifts - len(zs) - 1, 0)]
        centre = int(np.argmin(np.abs(pts.real)))
        for i in dict.fromkeys(peaks + [centre]):
            zs.append(complex(pts[i]))
            norms.append(float(nrm[i]))
    if not zs:
        return [], []
    weight = np.asarray(norms) ** 2
    weight = weight / max(float(weight.max()), 1e-300)
    return zs, [float(x) for x in weight]


def _exempt_pslots(matrix, seed_keys, seed_block, extra_keys):
    """Boolean over the P slots: the seeds, the heaviest first-shell rows (covering all but
    :data:`FIRST_SHELL_EXEMPT_TAIL` of ``|H s|^2``) and ``extra_keys`` (last round's admissions). **Collective.**"""
    from impurityModel.ed.manybody_basis import collective_mass_cutoff

    n_p = matrix.n_p_local
    exempt = np.zeros(n_p, dtype=bool)
    for keys in (seed_keys, extra_keys):
        if len(keys):
            pslots = _pslots_of(matrix, keys)
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
    # Larger = less important, with the relative resolution of a float (the selections bin scores to 32 mantissa
    # bits: `1 / (1 + s/smax)` would put every row below 2e-10 smax into one bin, broken by key order).
    low = 1.0 / np.maximum(s_ret, 1e-300)
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


@dataclass(frozen=True)
class ScoringPlan:
    """What the recurrence must accumulate for the scores, decided before it runs (replicated): the scoring shifts
    and their weights, the shifts the accumulator is given (the conjugates appended when ``need_y``) and the
    residual the run stops at."""

    zs: tuple
    weights: tuple
    need_y: bool
    stop_rtol: float

    @property
    def accumulator_shifts(self):
        return list(self.zs) + ([complex(z).conjugate() for z in self.zs] if self.need_y else [])

    def split(self, solutions):
        """``(xs, ys)`` from the accumulator's solution list."""
        n = len(self.zs)
        return list(solutions[:n]), (list(solutions[n:]) if self.need_y else None)


def scoring_plan(matrix, spec, seed_block):
    """The :class:`ScoringPlan` of a refining unit, or ``(None, reason)`` when it cannot be scored. ``seed_block`` is
    the seed block on the P slots. **Collective** (``H`` real and the seeds real up to a phase are global tests)."""
    if not spec.shifts:
        return None, "no scoring shifts"
    need_y = not matrix_is_real(matrix) or not seeds_real_up_to_a_phase(matrix, seed_block)
    weighted = REFINE_WEIGHTED if spec.weighted is None else spec.weighted
    zs = tuple(complex(z) for z, _w in spec.shifts)
    weights = tuple(float(w) if weighted else 1.0 for _z, w in spec.shifts)
    rtol = REFINE_STOP_RTOL if spec.stop_rtol is None else spec.stop_rtol
    return ScoringPlan(zs, weights, need_y, rtol), None


def refine_frozen_unit(matrix, spec, plan, seed_keys, seed_q0, first, first_acc, rerun, g_cf, eval_meshes, seed_block):
    """Swap refinement of a unit the cap froze, on its discovered matrix. **Collective** (every rank, same arguments).

    ``first`` is round 0's ``(alphas, betas, widths, status, q_last)`` and ``first_acc`` the
    :class:`~impurityModel.ed.gf_shift_recycling.ShiftedGalerkin` that run stepped (a recurrence on the final P,
    ``plan.accumulator_shifts``); ``rerun(q0, shifted, shifted_rtol)`` is the frozen recurrence from a seed block on
    the matrix's current P slots, stepping the accumulator it is given; ``g_cf(result, zs)`` the continued-fraction
    ``G`` of a result on the shifts ``zs``; ``seed_block(matrix, seed_keys, q0)`` the seed block on the current P slots.

    Round ``n`` scores the retained and discovered determinants from the accumulator's solutions
    (:func:`importance_scores`), then swaps the least important retained against the most important discovered ones
    (:meth:`DistributedSparseMatrix.set_p_keys`: the cap is unchanged) and runs the recurrence again on the new P. It
    stops when the remaining estimated gain is below ``spec.tol`` times the largest ``|G|`` on the shifts, when no swap
    gains, or after ``spec.rounds`` swaps; the last recurrence is the unit's result. A refinement whose final estimated
    gain is larger than round 0's falls back to round 0's P and result. The record's ``dG_*`` are
    ``max|G_final - G_0| / max|G_0|`` of the continued fractions on the scoring shifts and on held-out ones."""
    from impurityModel.ed.gf_shift_recycling import ShiftedGalerkin

    comm = matrix.comm
    t0 = time.perf_counter()
    record = {"declined": None, "rounds": 0, "swapped": [], "gain": [], "tol": spec.tol, "swap": spec.swap}
    zs = list(plan.zs)
    _unused, z_hold = pick_shifts(lambda z: g_cf(first, z), eval_meshes)
    z_hold = [z for z in z_hold if z not in zs]
    g0 = g_cf(first, zs)
    scale = float(np.abs(g0).max())
    g0_hold = g_cf(first, z_hold) if z_hold else None
    keys0 = [key for key, a in zip(matrix.p_keys(), matrix.alive()) if a]
    n_alive = int(_allreduce(comm, float(matrix.n_alive_local), MPI.SUM))
    k_max = max(1, int(spec.swap * n_alive))
    extra, first_gain = [], None
    swapped_total = 0
    result, acc = first, first_acc
    for round_ in range(spec.rounds + 1):
        t_round = time.perf_counter()
        seeds = seed_block(matrix, seed_keys, seed_q0)
        xs, ys = plan.split(acc.solutions())
        scores = importance_scores(matrix, seeds, zs, xs=xs, ys=ys, weights=plan.weights)
        exempt = _exempt_pslots(matrix, seed_keys, seeds, extra)
        retire, admit, gain = _select_swap(matrix, scores, exempt, k_max)
        record["gain"].append(gain)
        record.setdefault("pass_seconds", []).append(time.perf_counter() - t_round)
        record.setdefault("blocks", []).append(len(result[2]))
        record.setdefault("floor", []).append(float(acc.floor.max(initial=0.0)))
        if first_gain is None:
            first_gain = gain
        n_swap = int(_allreduce(comm, float(len(retire)), MPI.SUM))
        if gain <= spec.tol * scale or n_swap == 0 or round_ == spec.rounds:
            break
        record["swapped"].append(n_swap)  # the swaps made, not the one the last pass would have made
        del scores, seeds, xs, ys, acc  # the solutions and the accumulator go before the (memory-hungry) set_p_keys
        keep = np.ones(matrix.n_p_local, dtype=bool)
        keep[retire] = False
        keep &= matrix.alive().astype(bool)
        keys = [key for key, k_ in zip(matrix.p_keys(), keep) if k_] + list(admit)
        matrix.set_p_keys(keys)
        extra = list(admit)
        swapped_total += n_swap
        record["rounds"] = round_ + 1
        acc = ShiftedGalerkin(plan.accumulator_shifts, matrix.n_p_local)
        result = rerun(seed_block(matrix, seed_keys, seed_q0), acc, plan.stop_rtol)
    record["settled"] = bool(gain <= spec.tol * scale)
    if swapped_total == 0:
        record["seconds"] = time.perf_counter() - t0
        return RefineOutcome(None, record)
    if record["gain"][-1] > first_gain:
        matrix.set_p_keys(keys0)
        record["fell_back"] = True
        record["seconds"] = time.perf_counter() - t0
        return RefineOutcome(None, record)
    dg = np.abs(g_cf(result, zs) - g0).max() / scale if scale > 0 else 0.0
    record["dG_scoring"] = float(dg)
    if g0_hold is not None:
        record["dG_holdout"] = float(np.abs(g_cf(result, z_hold) - g0_hold).max() / max(np.abs(g0_hold).max(), 1e-300))
    record["seconds"] = time.perf_counter() - t0
    return RefineOutcome(result, record)
