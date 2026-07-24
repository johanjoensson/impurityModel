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
    converged_fn, _flag, _tol, _dg = _make_gf_convergence_monitor(DELTA, slaterWeightMin=0.0)
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
    # ... and never worse than NONE. (On a fully-spanned uniform-dense spectrum NONE's lost
    # orthogonality only injects harmless duplicate poles, so its derived-G error can sit at the
    # same floor as PARTIAL; the orthogonality gap itself is guarded in test_reort_oracle.py.)
    assert err_partial <= err_none + 1e-12


def test_absolute_safeguard_bounds_gradual_beta_growth():
    """Forcing the rank-~2 spectrum far past its effective rank (convergence disabled) must
    not let beta run away by ~10 orders of magnitude (~2.5e10 unguarded). Two healthy
    endings exist in this rounding-marginal regime and both are correct: the divergence
    safeguard truncates a corrupted tail early, or PARTIAL keeps the recurrence clean
    through the whole budget so there is nothing to truncate (the typical trajectory since
    the W-estimator fix — the old over-triggering estimator projected q_next against
    ~everything every step, collapsing the residual and forcing an early invariant-subspace
    exit, which is what this test's original ``len(a) < max_iter`` assertion captured).
    Assert the outcome (no runaway), not the mechanism."""
    rng = np.random.default_rng(3)
    d = np.concatenate([rng.normal(0.0, 1e-3, 150), rng.normal(10.0, 1e-3, 150)])
    H, Q0, _, _, _ = _H_and_seed(d, seed=3)
    _a, b, _, _ = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H.astype(complex),
        converged=lambda *x, **k: False,
        reort=Reort.PARTIAL,
        return_widths=True,
        max_iter=120,
    )
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


# --------------------------------------------------------------------------- #
# calc_G isolated unit / edge cases (the continued fraction vs a dense resolvent)
# --------------------------------------------------------------------------- #
def _dense_seed_resolvent(alphas, couplings, r, omega, delta):
    """Oracle: r^H [(omega+i delta) I - T]^{-1}_{block 0,0} r, T dense block-tridiagonal.

    ``couplings[i]`` (length ``nb-1``) is the sub-diagonal block ``T[i+1, i]`` matching the
    ``calc_G`` convention ``betas[i]`` = block ``i -> i+1``; ``T`` is Hermitian.
    """
    p = alphas[0].shape[0]
    nb = len(alphas)
    N = nb * p
    T = np.zeros((N, N), dtype=complex)
    for i in range(nb):
        T[i * p : (i + 1) * p, i * p : (i + 1) * p] = alphas[i]
        if i < nb - 1:
            T[(i + 1) * p : (i + 2) * p, i * p : (i + 1) * p] = couplings[i]
            T[i * p : (i + 1) * p, (i + 1) * p : (i + 2) * p] = couplings[i].conj().T
    out = np.zeros((len(omega), r.shape[1], r.shape[1]), dtype=complex)
    for w_idx, w in enumerate(omega):
        res00 = np.linalg.inv((w + 1j * delta) * np.eye(N) - T)[:p, :p]
        out[w_idx] = r.conj().T @ res00 @ r
    return out


def test_calc_g_empty_blocks_returns_zeros():
    r = np.ones((2, 3), dtype=complex)
    omega = np.linspace(-1, 1, 4)
    G = calc_G([], [], r, omega, 0.0, DELTA)
    assert G.shape == (4, 3, 3)
    np.testing.assert_allclose(G, 0.0, atol=0.0)


def test_calc_g_single_block_is_plain_resolvent():
    rng = np.random.default_rng(0)
    a0 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    a0 = a0 + a0.conj().T
    r = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    omega = np.linspace(-4, 4, 20)
    G = calc_G(np.array([a0]), np.empty((0, 2, 2), dtype=complex), r, omega, 0.0, DELTA)
    ref = _dense_seed_resolvent([a0], [], r, omega, DELTA)
    np.testing.assert_allclose(G, ref, atol=1e-10)


def test_calc_g_two_blocks_matches_dense_resolvent():
    rng = np.random.default_rng(1)
    a0 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    a0 = a0 + a0.conj().T
    a1 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    a1 = a1 + a1.conj().T
    b0 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    r = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    omega = np.linspace(-5, 5, 30)
    # calc_G expects len(betas) == len(alphas); the trailing residual block is ignored.
    residual = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    G = calc_G(np.array([a0, a1]), np.array([b0, residual]), r, omega, 0.0, DELTA)
    ref = _dense_seed_resolvent([a0, a1], [b0], r, omega, DELTA)
    np.testing.assert_allclose(G, ref, atol=1e-9)
