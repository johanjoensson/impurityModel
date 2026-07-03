"""Benchmark: full rank-4 RIXS tensor (:func:`spectra.getRIXSmap_tensor`) vs the
per-polarization Kramers-Heisenberg map (:func:`spectra.getRIXSmap_new`).

Both produce the identical ``[n_in_pol, n_out_pol, wIn, wLoss]`` intensity map. The tensor path
runs one ``block_bicgstab`` (R2) over the K Cartesian in-components and one ``block_Green`` (R3)
over the K*K seed block per (wIn, eigenstate) -- cost independent of the number of requested
polarizations -- then contracts. The per-polarization path solves an N-in block and N
block-Greens of size N, so it grows with N.

Marked ``benchmark`` (skipped by default; run with ``pytest -m benchmark``) with a second
``RUN_RIXS_BENCH=1`` env-var gate, mirroring ``test_selfenergy_perf.py``. Use ``-s`` to see the
printed tables::

    RUN_RIXS_BENCH=1 pytest -s -m benchmark src/impurityModel/test/test_rixs_tensor_perf.py

Findings this harness pins (vs a dense Kramers-Heisenberg ground truth):

* **Speed:** the tensor is flat in the number of polarizations while per-pol grows -- ~2x for a
  single polarization up to ~7x for 16, the R4 payoff (compute once, contract for any
  polarizations).
* **Accuracy:** both paths are accurate to the ``block_bicgstab`` R1 tolerance (~1e-7..1e-8) and
  independent of the number of seeds/polarizations. (Historically a floor-division bug in the
  block-Lanczos iteration cap -- ``max_iter = N // block_width`` in ``greens_function`` -- left
  up to ``block_width - 1`` dimensions of a closed final-state sector unresolved, producing a
  spurious ~1e-4 "floor" that grew with the block width; fixed to ``ceil`` so the final,
  deflating block is always taken.)
"""

import os
import time
from itertools import combinations

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import spectra
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, applyOp, inner

RUN = os.environ.get("RUN_RIXS_BENCH") == "1"
# The RUN_RIXS_BENCH gate is a second guard so even `pytest -m benchmark` skips this unless
# explicitly opted in (it prints timing tables and is not a correctness gate for CI).
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not RUN, reason="Set RUN_RIXS_BENCH=1 to run the RIXS tensor benchmark."),
]

# --- model geometry (block key 2 = valence/impurity, block key 1 = core, as in the RIXS tests) ---
NV = 6  # valence orbitals (block 2), indices 0..NV-1
NC = 2  # core orbitals (block 1), indices NV..NV+NC-1
NE_VAL = 3  # valence electrons in the ground sector (core full)
K = 3  # number of Cartesian components (dipole target valence orbitals)
N_ORB = NV + NC
CORE = list(range(NV, NV + NC))
NE_TOT = NE_VAL + NC  # total electrons; conserved by H and by the dipole (create + annihilate)

TAU = 0.02
D1, D2 = 0.4, 0.2
WIN = np.array([-7.0, -6.5])
WLOSS = np.linspace(-1.0, 3.0, 11)
N_GS = 2  # thermally-averaged ground states
N_POLS = [1, 2, 4, 8, 16]


def _model(seed=1):
    rng = np.random.default_rng(seed)
    d = {}
    eps = rng.uniform(-1.0, 1.0, NV)
    for i in range(NV):
        d[((i, "c"), (i, "a"))] = eps[i]
    for i in range(NV):
        for j in range(i + 1, NV):
            t = rng.uniform(-0.3, 0.3)
            d[((i, "c"), (j, "a"))] = t
            d[((j, "c"), (i, "a"))] = t
    for i in range(NV):
        for j in range(i + 1, NV):
            d[((i, "c"), (j, "c"), (j, "a"), (i, "a"))] = rng.uniform(0.5, 2.0)
    for c in CORE:
        d[((c, "c"), (c, "a"))] = -8.0
    return ManyBodyOperator(d)


def _bytes(occ):
    b = bytearray((N_ORB + 7) // 8)
    for o in occ:
        b[o // 8] |= 1 << (7 - (o % 8))
    return bytes(b)


def _ground_dets():
    return [_bytes(list(val) + CORE) for val in combinations(range(NV), NE_VAL)]


def _all_dets():
    # every NE_TOT-electron determinant: spans both the core-full (ground) and core-hole
    # (RIXS-intermediate) sectors, which the dense-reference resolvents need.
    return [_bytes(occ) for occ in combinations(range(N_ORB), NE_TOT)]


def _states(dets):
    return [ManyBodyState({SlaterDeterminant.from_bytes(d): 1.0}) for d in dets]


def _matrix(op, states):
    n = len(states)
    m = np.zeros((n, n), dtype=complex)
    for j, sj in enumerate(states):
        col = applyOp(op, sj)
        for i, si in enumerate(states):
            m[i, j] = inner(si, col)
    return m


def _thermal_states(op):
    dets = _ground_dets()
    states = _states(dets)
    ev, vec = np.linalg.eigh(_matrix(op, states))
    psis = [
        ManyBodyState(
            {SlaterDeterminant.from_bytes(dets[i]): vec[i, k] for i in range(len(dets)) if abs(vec[i, k]) > 1e-14}
        )
        for k in range(N_GS)
    ]
    return psis, list(ev[:N_GS]), dets


def _basis(dets):
    return Basis(
        impurity_orbitals={2: [list(range(NV))], 1: [CORE]},
        bath_states=({2: [[]], 1: [[]]}, {2: [[]], 1: [[]]}),
        initial_basis=list(dets),
        verbose=False,
        comm=MPI.COMM_SELF,
    )


def _components():
    in_comp = [ManyBodyOperator({((k, "c"), (CORE[0], "a")): 1.0}) for k in range(K)]  # core-hole excitation
    out_comp = [ManyBodyOperator({((CORE[0], "c"), (k, "a")): 1.0}) for k in range(K)]  # fill the core hole
    return in_comp, out_comp


def _dense_states(op):
    dets = _all_dets()
    states = _states(dets)
    ev, vec = np.linalg.eigh(_matrix(op, states))
    return ev, vec, states


def _dense_ref(op, in_comp, out_comp, epsIn, epsOut, es_all, vec_all, states):
    """Independent dense Kramers-Heisenberg map contracted with arbitrary in/out polarizations.

    T_in(e)  = sum_a e_a       Tin_a   ; T_out(e) = sum_b e_b^* Tout_b (daggered dipole).
    """
    n = len(states)
    eye = np.eye(n, dtype=complex)
    H = _matrix(op, states)
    Tin = [_matrix(t, states) for t in in_comp]
    Tout = [_matrix(t, states) for t in out_comp]
    e0 = min(es_all[:N_GS])
    Z = float(np.sum(np.exp(-(np.asarray(es_all[:N_GS]) - e0) / TAU)))
    out = np.zeros((len(epsIn), len(epsOut), len(WIN), len(WLOSS)), dtype=complex)
    for g in range(N_GS):
        gvec = vec_all[:, g]
        wg = np.exp(-(es_all[g] - e0) / TAU)
        for ki, win in enumerate(WIN):
            psi2 = [np.linalg.solve((win + 1j * D1 + es_all[g]) * eye - H, Tin[a] @ gvec) for a in range(K)]
            for pin, e_in in enumerate(epsIn):
                psi2_in = sum(e_in[a] * psi2[a] for a in range(K))
                for pout, e_out in enumerate(epsOut):
                    psi3 = sum(np.conj(e_out[b]) * (Tout[b] @ psi2_in) for b in range(K))
                    for kl, wl in enumerate(WLOSS):
                        r2 = np.linalg.solve((wl + 1j * D2 + es_all[g]) * eye - H, psi3)
                        out[pin, pout, ki, kl] += wg * (psi3.conj() @ r2)
    return out / Z


def _run_perpol(op, psis, es, dets, in_comp, out_comp, epsIn, epsOut):
    t_ops_in = [spectra._combine_component_ops(in_comp, e) for e in epsIn]
    t_ops_out = [spectra._combine_component_ops(out_comp, np.conj(e)) for e in epsOut]
    return spectra.getRIXSmap_new(
        op,
        t_ops_in,
        t_ops_out,
        psis,
        es,
        tau=TAU,
        wIns=WIN,
        wLoss=WLOSS,
        delta1=D1,
        delta2=D2,
        basis=_basis(dets),
        verbose=False,
        slaterWeightMin=1e-12,
    )


def _run_tensor(op, psis, es, dets, in_comp, out_comp, epsIn, epsOut):
    return spectra.getRIXSmap_tensor(
        op,
        in_comp,
        out_comp,
        epsIn,
        epsOut,
        psis,
        es,
        tau=TAU,
        wIns=WIN,
        wLoss=WLOSS,
        delta1=D1,
        delta2=D2,
        basis=_basis(dets),
        verbose=False,
        slaterWeightMin=1e-12,
    )


def test_rixs_tensor_vs_perpol_scaling():
    """Wall-time and accuracy-vs-dense of the tensor and per-pol paths as the number of
    requested in/out polarizations grows. The tensor should stay flat and win at scale."""
    op = _model()
    psis, es, dets = _thermal_states(op)
    in_comp, out_comp = _components()
    ev, vec, states = _dense_states(op)

    print(f"\nmodel: {NV} valence + {NC} core orbitals, {NE_VAL} valence e-, K={K} components")
    print(
        f"ground dim = {len(_ground_dets())}, full {NE_TOT}e- space = {len(_all_dets())}, "
        f"wIn={len(WIN)}, wLoss={len(WLOSS)}, n_gs={N_GS}"
    )
    hdr = (
        f"{'n_pol':>6} | {'per-pol (s)':>12} | {'tensor (s)':>12} | {'speedup':>8} | "
        f"{'err_perpol':>11} | {'err_tensor':>11}"
    )
    print("\n" + hdr + "\n" + "-" * len(hdr))

    results = {}
    for n in N_POLS:
        rng = np.random.default_rng(100 + n)
        epsIn = rng.normal(size=(n, K)) + 1j * rng.normal(size=(n, K))
        epsOut = rng.normal(size=(n, K)) + 1j * rng.normal(size=(n, K))
        ref = _dense_ref(op, in_comp, out_comp, epsIn, epsOut, ev, vec, states)

        t0 = time.perf_counter()
        old = _run_perpol(op, psis, es, dets, in_comp, out_comp, epsIn, epsOut)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = _run_tensor(op, psis, es, dets, in_comp, out_comp, epsIn, epsOut)
        t_new = time.perf_counter() - t0

        err_old = float(np.max(np.abs(old - ref)))
        err_new = float(np.max(np.abs(new - ref)))
        results[n] = (t_old, t_new, err_old, err_new)
        print(
            f"{n:>6} | {t_old:>12.3f} | {t_new:>12.3f} | {t_old / t_new:>7.2f}x | "
            f"{err_old:>11.2e} | {err_new:>11.2e}"
        )

    # Both paths are now accurate at the block_bicgstab R1 tolerance, flat in the number of
    # polarizations. Timings are noisy, so only assert the qualitative win at the largest sweep.
    n_max = N_POLS[-1]
    t_old, t_new, err_old, err_new = results[n_max]
    assert t_new < t_old, f"tensor ({t_new:.3f}s) not faster than per-pol ({t_old:.3f}s) at n_pol={n_max}"
    assert err_new < 1e-4, f"tensor error {err_new:.2e} unexpectedly large (block-Lanczos should span the sector)"
    assert err_old < 1e-4, f"per-pol error {err_old:.2e} unexpectedly large at n_pol={n_max}"


def test_rixs_blockgreen_accuracy_independent_of_seed_count():
    """Accuracy is independent of the block_Green seed count: 1-seed, K-seed (per-pol) and
    K*K-seed (tensor) blocks are all accurate to the block_bicgstab R1 tolerance.

    This is a regression guard for the floor-division bug in the block-Lanczos iteration cap
    (``max_iter = N // block_width``), which truncated the final deflating block and left up to
    ``block_width - 1`` dimensions of the closed final-state sector unresolved -- a spurious error
    that GREW with the block width (~4e-9 at 1 seed, ~4e-5 at 3, ~4e-4 at 9). With ``ceil`` the
    recurrence spans the sector and all three sit near machine precision (bounded by R1)."""
    op = _model()
    psis, es, dets = _thermal_states(op)
    in_comp, out_comp = _components()
    ev, vec, states = _dense_states(op)
    I = np.eye(K, dtype=complex)
    dense = _dense_ref(op, in_comp, out_comp, I, I, ev, vec, states)

    tens = _run_tensor(op, psis, es, dets, in_comp, out_comp, I, I)  # one 9-seed block_Green
    perpol = _run_perpol(op, psis, es, dets, in_comp, out_comp, I, I)  # K block_Greens of size K
    one = [[1.0]]  # single-component polarization
    single = np.zeros_like(dense)  # one in- and one out-component at a time (single-seed)
    for a in range(K):
        for b in range(K):
            single[a, b] = _run_perpol(op, psis, es, dets, [in_comp[a]], [out_comp[b]], one, one)[0, 0]

    err_single = float(np.max(np.abs(single - dense)))
    err_perpol = float(np.max(np.abs(perpol - dense)))
    err_tensor = float(np.max(np.abs(tens - dense)))
    print(
        f"\nseed-count accuracy vs dense: 1 seed = {err_single:.2e}, "
        f"{K} seeds (per-pol) = {err_perpol:.2e}, {K * K} seeds (tensor) = {err_tensor:.2e}"
    )

    # All near machine precision (bounded by the block_bicgstab R1 tolerance), and -- the point --
    # the wider blocks are NOT worse than the single seed (the old bug made them ~10^5x worse).
    assert err_single < 1e-6
    assert err_perpol < 1e-6
    assert err_tensor < 1e-6
