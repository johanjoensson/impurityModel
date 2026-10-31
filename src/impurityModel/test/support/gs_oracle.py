"""A dense ground-state oracle: exact E(N) for every electron count of a small random impurity model.

The model, the exact spectra and the checks that ``solve_ground_state`` and the occupation walk are
held to. Ported from ``debug/gs_review_2026-10/oracle.py`` (the review's sweep) so the defects it found
(P7, S1 of ``doc/plans/gs_review_2026-10.md``) stay found.

:func:`check_model` wraps ``groundstate.solve_sector`` -- the one seam every sector solve goes through,
walk and production alike -- and compares *every* sector the run solves with the dense answer for its
electron count, not only the winner.
"""

import itertools

import numpy as np

from impurityModel.ed import atomic_physics
from impurityModel.ed import groundstate
from impurityModel.ed.average import energy_cut
from impurityModel.ed.interaction_models import kanamori_u4
from impurityModel.ed.lie_algebra import tensors_to_operator
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp

def det(occ):
    chunk = 0
    for o in occ:
        chunk |= 1 << (63 - o)
    return SlaterDeterminant((chunk,))


def popcount(d):
    return bin(int.from_bytes(bytes(d.to_bytearray())[:8], "big")).count("1")


def exact_spectra(h_op, n_orb):
    h_op.set_restrictions(None)
    h_op.set_weighted_restrictions(None)
    out = {}
    for n in range(n_orb + 1):
        dets = [det(o) for o in itertools.combinations(range(n_orb), n)]
        index = {d: i for i, d in enumerate(dets)}
        h = np.zeros((len(dets), len(dets)), dtype=complex)
        for j, d in enumerate(dets):
            for key, row in applyOp(h_op, ManyBodyState({d: 1.0}), cutoff=0).items():
                h[index[key], j] += row[0]
        out[n] = np.linalg.eigvalsh(h)
    return out


def random_model(rng, complex_h=False):
    n_spatial = 2
    n_imp = 2 * n_spatial
    n_val = int(rng.integers(2, 5))  # valence bath spin-orbitals
    n_val -= n_val % 2
    n_cond = int(rng.integers(0, 3)) * 2
    n_orb = n_imp + n_val + n_cond
    h = np.zeros((n_orb, n_orb), dtype=complex)
    eps = rng.uniform(-3, 1, n_spatial)
    for s in range(2):
        for a in range(n_spatial):
            i = a + n_spatial * s
            h[i, i] = eps[a]
    val = list(range(n_imp, n_imp + n_val))
    cond = list(range(n_imp + n_val, n_orb))
    for blk, lo, hi in ((val, -3, -0.5), (cond, 0.5, 3)):
        for k in range(len(blk) // 2):
            e = rng.uniform(lo, hi)
            a = int(rng.integers(n_spatial))
            for s in range(2):
                b = blk[2 * k + s]
                i = a + n_spatial * s
                h[b, b] = e
                v = rng.uniform(0.1, 1.0)
                if complex_h:
                    v = v * np.exp(1j * rng.uniform(0, 2 * np.pi))
                h[i, b] = v
                h[b, i] = np.conj(v)
    # orbital mixing
    t = rng.uniform(-0.5, 0.5) * (np.exp(1j * rng.uniform(0, 6.3)) if complex_h else 1)
    for s in range(2):
        h[n_spatial * s, n_spatial * s + 1] = t
        h[n_spatial * s + 1, n_spatial * s] = np.conj(t)
    if complex_h:  # spin-mixing (SOC-like)
        so = 0.2j
        h[0, n_spatial + 1] = so
        h[n_spatial + 1, 0] = np.conj(so)
    U = rng.uniform(1, 8)
    J = rng.uniform(0, 0.3) * U
    u_op = ManyBodyOperator(atomic_physics.getUop_from_rspt_u4(kanamori_u4(n_spatial, U, J)))
    h_op = tensors_to_operator(h) + u_op
    imp = {0: [list(range(n_imp))]}
    bath = ({0: [val]}, {0: [cond]})
    N0 = {0: int(rng.integers(0, n_imp + 1))}
    return h_op, imp, bath, N0, n_orb, n_val, dict(U=U, J=J, eps=eps.tolist(), n_val=n_val, n_cond=n_cond)


def pinned_model(seed, index, complex_h=False):
    """The ``index``-th model drawn from ``default_rng(seed)``, the way the review's sweep draws them."""
    rng = np.random.default_rng(seed)
    for _ in range(index + 1):
        model = random_model(rng, complex_h=complex_h)
    return model


def _gather_counts(basis, comm):
    local = {popcount(d) for d in basis.local_basis}
    return set().union(*comm.allgather(local)) if comm is not None else local


def check_model(h_op, imp, bath, N0, n_orb, *, comm=None, tau=0.05, spectra=None, **options):
    """Run ``solve_ground_state`` and compare it with the dense answer: a list of failure messages.

    Every sector the run solves (``groundstate.solve_sector``) is held to the exact lowest level of its
    electron count, to 1e-6 -- the count is read off the solved basis by popcount, not derived from
    ``N0``. The winner must sit at the global minimum, and its thermal window must be the exact one
    (a degenerate group straddling the cut may come back whole). A winner whose basis mixes electron
    counts is a failure, not a skipped check. ``options`` go to ``solve_ground_state``; the cap is off.
    """
    spectra = exact_spectra(h_op, n_orb) if spectra is None else spectra
    e_by_n = {n: float(e[0]) for n, e in spectra.items() if len(e)}
    e_star = min(e_by_n.values())
    msgs = []
    real_solve = groundstate.solve_sector

    def recording(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        if len(result.basis) == 0:
            return result
        counts = _gather_counts(result.basis, comm)
        if len(counts) != 1:
            msgs.append(f"a sector solve mixes electron counts {sorted(counts)}")
            return result
        (n,) = counts
        e = float(np.min(np.real(result.es)))
        if abs(e - e_by_n[n]) > 1e-6:
            msgs.append(f"sector N={n}: E={e:.6f} vs exact {e_by_n[n]:.6f}")
        return result

    kw = dict(
        tau=tau,
        slaterWeightMin=1e-12,
        comm=comm,
        rank=comm.rank if comm is not None else 0,
        verbose=0,
        truncation_threshold=np.inf,
        dense_cutoff=1000,
    )
    kw.update(options)
    groundstate.solve_sector = recording
    try:
        gs = groundstate.solve_ground_state(h_op, imp, bath, dict(N0), **kw)
    finally:
        groundstate.solve_sector = real_solve
    es = np.sort(np.real(np.asarray(gs.es)))
    counts = _gather_counts(gs.basis, comm)
    if len(counts) != 1:
        msgs.append(f"the winner's basis mixes electron counts {sorted(counts)}")
        return msgs
    (n_found,) = counts
    if abs(float(es[0]) - e_star) > 1e-6:
        msgs.append(f"E0 {es[0]:.6f} vs exact {e_star:.6f} (found N={n_found}, exact levels {e_by_n})")
        return msgs
    ex = spectra[n_found]
    win = ex[ex <= ex[0] + energy_cut(tau) + 1e-6]
    whole = ex[ex <= win[-1] + 1e-6]  # the straddling degenerate group kept whole
    if not any(len(c) == len(es) and np.allclose(c, es, atol=1e-6, rtol=0) for c in (win, whole)):
        msgs.append(f"thermal window: exact {np.round(win - ex[0], 4)} vs got {np.round(es - es[0], 4)}")
    return msgs
