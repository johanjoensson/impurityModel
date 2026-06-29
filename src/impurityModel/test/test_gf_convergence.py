r"""Green's-function convergence reliability (block Lanczos, WS4).

With partial reorthogonalization fixed (see ``test_reort_oracle.py``) the interacting
Green's function must converge *reliably*:

* the frozen-mesh relative monitor stops a resolvable spectrum early — before the basis can
  degrade — so the pathological "run far past the effective rank" regime is never reached;
* on a genuinely dense spectrum that needs (nearly) the full Krylov space, ``PARTIAL`` keeps
  the basis orthonormal and the resulting ``G`` matches ``FULL`` and the dense Lehmann
  reference (``NONE`` does not — lost orthogonality injects ghost weight);
* an absolute spectral-scale safeguard bounds gradual beta growth even when convergence is
  disabled, so a corrupted tail can never reach the continued fraction.
"""

import numpy as np
import scipy.linalg as la

from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array
from impurityModel.ed.greens_function import (
    _make_gf_convergence_monitor,
    _sanitize_continued_fraction,
    _trim_blocks,
    build_qr,
    calc_G,
)

DELTA = 0.1


def _H_and_seed(d, seed):
    rng = np.random.default_rng(seed)
    n = d.size
    U = la.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    H = (U * d) @ U.conj().T
    H = 0.5 * (H + H.conj().T)
    S = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
    Q0, r = build_qr(S)
    return H, Q0, r, U, S


def _gf_monitor():
    """The *production* GF convergence monitor (:func:`_make_gf_convergence_monitor`), wrapped
    only to record the iteration at which it first declares convergence. Using the real monitor
    (rather than a hand-copy) keeps this test honest: it exercises the actual adaptive-freeze +
    consecutive-step logic and breaks if that logic regresses."""
    converged_fn, _flag, _tol = _make_gf_convergence_monitor(DELTA, slaterWeightMin=0.0)
    stop_it = [None]

    def converged(alphas, betas, verbose=False, block_widths=None, **kw):
        ok = converged_fn(alphas, betas, verbose=verbose, block_widths=block_widths, **kw)
        if ok and stop_it[0] is None:
            stop_it[0] = len(alphas)
        return ok

    return converged, stop_it


def test_monitor_stops_resolvable_spectrum_early():
    """Two tight clusters => 2 effective poles => the monitor must converge in a handful of
    blocks, while the basis is still orthonormal and beta is bounded (no degradation)."""
    rng = np.random.default_rng(3)
    d = np.concatenate([rng.normal(0.0, 1e-3, 150), rng.normal(10.0, 1e-3, 150)])
    H, Q0, _, _, _ = _H_and_seed(d, seed=3)

    converged, stop_it = _gf_monitor()
    a, b, Q, _ = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H.astype(complex),
        converged=converged,
        reort=Reort.PARTIAL,
        return_widths=True,
        max_iter=120,
    )
    assert stop_it[0] is not None and stop_it[0] <= 15  # stops early, not 100+
    assert len(a) == stop_it[0]
    assert np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1])) < 1e-8  # basis still clean
    assert max(np.linalg.norm(x, 2) for x in b) < 10 * np.linalg.norm(H, 2)


def test_partial_matches_full_gf_on_dense_spectrum():
    """A genuinely dense spectrum needs ~full space; PARTIAL must keep the basis clean so its
    G matches FULL and the dense Lehmann reference (where NONE would be corrupted)."""
    rng = np.random.default_rng(3)
    d = np.sort(rng.uniform(-30.0, 10.0, 300))
    H, Q0, r, U, S = _H_and_seed(d, seed=3)
    omega = np.linspace(-32.0, 12.0, 120)
    Sd = U.conj().T @ S
    G_dense = np.einsum("wi,ia,ib->wab", 1.0 / (omega[:, None] + 1j * DELTA - d[None, :]), np.conj(Sd), Sd)

    def gf(mode):
        a, b, _, w = block_lanczos_array(
            psi0=Q0.copy(),
            h_op=H.astype(complex),
            converged=_gf_monitor()[0],
            reort=mode,
            return_widths=True,
            max_iter=300,
        )
        at, bt = _sanitize_continued_fraction(*_trim_blocks(a, b, w))
        return calc_G(at, bt, r, omega, 0.0, DELTA)

    err_partial = np.max(np.abs(gf(Reort.PARTIAL) - G_dense))
    err_full = np.max(np.abs(gf(Reort.FULL) - G_dense))
    err_none = np.max(np.abs(gf(Reort.NONE) - G_dense))

    assert err_partial < 1e-8  # PARTIAL is accurate
    assert err_partial < 5 * err_full + 1e-12  # ... matching FULL
    assert err_none > 20 * err_partial  # ... and far better than NONE (lost orthogonality)


def test_absolute_safeguard_bounds_gradual_beta_growth():
    """Forcing the rank-~2 spectrum far past its effective rank (convergence disabled) makes
    beta grow gradually; the absolute spectral-scale safeguard must truncate it rather than
    let beta run away by ~10 orders of magnitude."""
    rng = np.random.default_rng(3)
    d = np.concatenate([rng.normal(0.0, 1e-3, 150), rng.normal(10.0, 1e-3, 150)])
    H, Q0, _, _, _ = _H_and_seed(d, seed=3)
    a, b, _, _ = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H.astype(complex),
        converged=lambda *x, **k: False,
        reort=Reort.PARTIAL,
        return_widths=True,
        max_iter=120,
    )
    assert len(a) < 120  # truncated before the cap
    assert max(np.linalg.norm(x, 2) for x in b) < 1e7  # bounded (was ~2.5e10 unguarded)


def test_safeguard_does_not_false_trigger_on_large_norm_H():
    """A well-behaved Hermitian H with ||H|| >> BETA_BLOWUP threshold must run to the cap
    (the spectral scale is seeded from beta_0 ~ ||H||, so it never self-triggers)."""
    d = np.linspace(-5000.0, 5000.0, 200)
    H, Q0, _, _, _ = _H_and_seed(d, seed=3)
    a, b, _, _ = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H.astype(complex),
        converged=lambda *x, **k: False,
        reort=Reort.PARTIAL,
        return_widths=True,
        max_iter=60,
    )
    assert len(a) == 60  # no false truncation
    assert max(np.linalg.norm(x, 2) for x in b) < 2 * np.linalg.norm(H, 2)
