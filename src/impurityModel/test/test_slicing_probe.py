"""Opt-in probe: energy-filtered seed support vs the union support (spectrum slicing, Phase 0).

The calibration measurement behind ``doc/plans/spectrum_slicing.md``: intercept
``calc_selfenergy`` at its ``get_Greens_function`` call (so the ground state, the excited
restrictions and the seeds are exactly the production driver's), estimate the spectral bounds
of ``H`` on the (capped) excited sector with a short Lanczos run, apply a Jackson-damped
Chebyshev partition-of-unity filter bank to one transition seed in a single three-term
recurrence, and tabulate each filtered seed's eps-support against the unfiltered recurrence's
(capped) union support.

Usage::

    RUN_SLICING_PROBE=1 WORKLOAD_H5=path/to/impurityModel_data.h5 \
    CAP=400000 DEGREE=1500 N_SLICES=8 SIDE=a BLOCK=0 \
    python -m pytest src/impurityModel/test/test_slicing_probe.py -m benchmark -s

Measured verdict on FCC Ni (2026-07-12): dominant amplitudes localize (up to ~260x below the
union at the 1e-4 level), the sub-1e-6 tail does not -- see the plan doc's table.
"""

import os
import time

import numpy as np
import pytest

import impurityModel.ed.greens_function as gf
import impurityModel.ed.selfenergy as se
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import Reort, resolve_reort
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, block_add_scaled_cy
from impurityModel.test.support.real_workload import load_workload, run_selfenergy

RUN = os.environ.get("RUN_SLICING_PROBE", "0") not in ("0", "", "false", "False")

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not RUN, reason="Set RUN_SLICING_PROBE=1 (and WORKLOAD_H5) to run."),
]


class _Captured(Exception):
    pass


def capture_driver_inputs(h5_path, cap):
    """Run ``calc_selfenergy`` up to (and excluding) the GF phase; return its GF inputs.

    The interception keeps the probe faithful: the ground state, the excited-sector
    restrictions and the transition seeds are exactly what the production driver would use.
    """
    captured = {}
    orig = se.get_Greens_function

    def spy(**kwargs):
        captured.update(kwargs)
        raise _Captured

    se.get_Greens_function = spy
    try:
        wl = load_workload(h5_path)
        try:
            run_selfenergy(wl, comm=None, truncation_threshold=cap, n_iw=4, n_w=4, verbosity=0)
        except _Captured:
            pass
    finally:
        se.get_Greens_function = orig
    return captured


def spectral_bounds(hOp, basis, n_iter=40, pad_rel=0.05, seed=1):
    """Extremal Ritz values of ``H`` on (a capped view of) the excited sector, padded."""
    rng = np.random.default_rng(seed)
    v = ManyBodyState({d: complex(*rng.standard_normal(2)) for d in basis.local_basis})
    v = v * (1.0 / np.sqrt(v.norm2()))
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
    k = len(alphas)
    T = np.zeros((k, k), dtype=complex)
    for i in range(k):
        T[i, i] = np.asarray(alphas[i])[0, 0]
        if i + 1 < k:
            b = np.asarray(betas[i])[0, 0]
            T[i, i + 1] = np.conj(b)
            T[i + 1, i] = b
    ev = np.linalg.eigvalsh(T)
    pad = pad_rel * (ev[-1] - ev[0])
    return ev[0] - pad, ev[-1] + pad


def window_coefficients(degree, theta_pairs):
    """Jackson-damped Chebyshev coefficients of indicator windows on ``[-1, 1]``.

    ``theta_pairs``: ``(theta_hi, theta_lo)`` with ``theta = arccos(x)`` (decreasing in x).
    Windows tiling the interval telescope to the constant 1 exactly, Jackson damping
    included (it is applied uniformly), so the partition of unity is exact by construction.
    """
    n = np.arange(1, degree + 1)
    g = (
        (degree + 1 - n) * np.cos(np.pi * n / (degree + 1))
        + np.sin(np.pi * n / (degree + 1)) / np.tan(np.pi / (degree + 1))
    ) / (degree + 1)
    sets = []
    for th_a, th_b in theta_pairs:
        c = np.empty(degree + 1)
        c[0] = (th_a - th_b) / np.pi
        c[1:] = (2.0 / np.pi) * (np.sin(n * th_a) - np.sin(n * th_b)) / n * g
        sets.append(c)
    return sets


def eps_support(state, cutoff):
    """Rows of a ``ManyBodyState`` with ``|amplitude| >= cutoff``."""
    return sum(1 for _k, a in state.items() if abs(a[0]) >= cutoff)


@pytest.mark.mpi_skip
def test_slicing_support_probe():
    h5_path = os.environ.get("WORKLOAD_H5")
    assert h5_path, "WORKLOAD_H5 must point to an impurityModel_data.h5 archive"
    cap = int(os.environ.get("CAP", "400000"))
    degree = int(os.environ.get("DEGREE", "1500"))
    n_slices = int(os.environ.get("N_SLICES", "8"))
    side = os.environ.get("SIDE", "a")
    block_i = int(os.environ.get("BLOCK", "0"))

    t0 = time.perf_counter()
    cap_in = capture_driver_inputs(h5_path, cap)
    hOp, basis, psis, es = cap_in["hOp"], cap_in["basis"], cap_in["psis"], cap_in["es"]
    blocks, w_mesh = cap_in["blocks"], cap_in["omega_mesh"]
    slaterWeightMin = cap_in["slaterWeightMin"]
    print(f"[probe] captured driver inputs in {time.perf_counter() - t0:.0f} s; GS basis {basis.size}")

    exc_restr, exc_wrestr = gf._build_excited_restrictions(basis, hOp, psis, es, cap_in["dN"], cap_in["occ_cutoff"])
    block_v = gf._apply_transition_ops(
        [ManyBodyOperator({((orb, side),): 1}) for orb in blocks[block_i]],
        [psis[int(np.argmin(es))]],
        exc_restr,
        exc_wrestr,
        slaterWeightMin,
    )
    seed = block_v[0][0]
    e0 = float(np.min(es))
    print(f"[probe] seed support {len(seed)} (block {blocks[block_i]}, side {side!r})")

    def _clone():
        return basis.clone(
            initial_basis=sorted(seed.keys()), restrictions=exc_restr, weighted_restrictions=exc_wrestr, verbose=False
        )

    hOp.set_restrictions(exc_restr)
    hOp.set_weighted_restrictions(exc_wrestr)
    lo, hi = spectral_bounds(hOp, gf._CappedBasisProxy(_clone(), cap))
    c_mid, e_half = (hi + lo) / 2.0, (hi - lo) / 2.0
    print(f"[probe] spectral bounds (capped sector): [{lo:.3f}, {hi:.3f}]")

    sign = 1.0 if side == "c" else -1.0
    ends = sorted((e0 + sign * float(np.min(w_mesh)), e0 + sign * float(np.max(w_mesh))))
    edges = sorted({lo, *np.linspace(max(lo, ends[0]), min(hi, ends[1]), n_slices + 1), hi})
    x_edges = np.clip((np.array(edges) - c_mid) / e_half, -1.0, 1.0)
    thetas = np.arccos(x_edges)
    coeff_sets = window_coefficients(degree, [(thetas[i], thetas[i + 1]) for i in range(len(x_edges) - 1)])
    labels = [f"[{edges[i]:.2f},{edges[i + 1]:.2f}]" for i in range(len(edges) - 1)]

    proxy = gf._CappedBasisProxy(_clone(), cap)
    one = np.eye(1, dtype=complex)
    t_prev = None
    t_cur = ManyBodyState.from_states([seed])
    accs = [block_add_scaled_cy(ManyBodyState(width=1), t_cur, c[0] * one) for c in coeff_sets]
    for it in range(1, degree + 1):
        w = hOp.apply_block(t_cur, slaterWeightMin)
        w = proxy.redistribute_block(w)
        w = block_add_scaled_cy(w, t_cur, (-c_mid) * one)
        if t_prev is not None:
            t_next = block_add_scaled_cy(t_prev.combine_columns(-one), w, (2.0 / e_half) * one)
        else:
            t_next = w.combine_columns((1.0 / e_half) * one)
        if slaterWeightMin > 0:
            t_next.prune_rows(slaterWeightMin)
        for s, c in enumerate(coeff_sets):
            if abs(c[it]) > 1e-15:
                accs[s] = block_add_scaled_cy(accs[s], t_next, c[it] * one)
        t_prev, t_cur = t_cur, t_next
        if it % 200 == 0:
            print(f"[probe] cheb it {it}/{degree}, live support {len(t_cur)}, retained {proxy.retained_size}")

    cutoffs = (1e-4, 1e-5, 1e-6, slaterWeightMin)
    print(f"\n[probe] union (recurrence) support: {proxy.retained_size:,} of cap {cap:,} (cap_hit={proxy.cap_hit})")
    print(f"[probe] {'window':>22} {'|v_s|^2':>10} " + " ".join(f"eps={c:.0e}" for c in cutoffs))
    for label, acc in zip(labels, accs):
        v_s = acc.to_states()[0]
        sups = " ".join(f"{eps_support(v_s, c):>9,}" for c in cutoffs)
        print(f"[probe] {label:>22} {v_s.norm2():10.3e} {sups}")
