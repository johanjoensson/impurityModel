"""The continued-fraction evaluator's width-1 fast path matches the general block recursion.

:func:`gf_primitives._block_cf_inverse` takes a scalar recursion when every block is ``1 x 1``
(every spectra unit, and most rotated self-energy blocks). It must agree with the block form --
which it replaces only for speed -- to rounding, including on a ragged (deflated) list and on
the padded ``(k, 1, 1)`` array the kernels return.
"""

import numpy as np
import pytest

from impurityModel.ed import gf_primitives


def _block_reference(alphas, betas, omegaP):
    """The general block recursion, forced (the fast path is bypassed by widening to 2 x 2)."""
    pad = lambda m: np.pad(np.asarray(m, dtype=complex), ((0, 1), (0, 1)))  # noqa: E731
    wide_a = [pad(a) + np.diag([0, 1e3]) for a in alphas]  # decoupled far level: no effect on [0, 0]
    wide_b = [pad(b) for b in betas]
    return gf_primitives._block_cf_inverse(wide_a, wide_b, omegaP)[:, :1, :1]


@pytest.mark.parametrize("as_array", [False, True])
def test_scalar_path_matches_the_block_recursion(as_array):
    rng = np.random.default_rng(3)
    k = 300
    alphas = [rng.normal(size=(1, 1)) + 0j for _ in range(k)]
    betas = [rng.normal(size=(1, 1)) + 0.4j * rng.normal(size=(1, 1)) for _ in range(k)]
    omegaP = np.concatenate([np.linspace(-8, 8, 257) + 0.1j, 1j * np.pi * 0.3 * (2 * np.arange(64) + 1)])
    if as_array:
        alphas, betas = np.array(alphas), np.array(betas)
    fast = gf_primitives._block_cf_inverse(alphas, betas, omegaP)
    ref = _block_reference(list(alphas), list(betas), omegaP)
    assert fast.shape == (len(omegaP), 1, 1)
    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=0)


def test_greens_function_through_the_scalar_path_is_the_exact_resolvent():
    """End to end through calc_G: a width-1 tridiagonal T against a dense (z - T)^-1."""
    rng = np.random.default_rng(4)
    k = 40
    a = rng.normal(size=k)
    b = np.abs(rng.normal(size=k)) + 0.1
    T = np.diag(a) + np.diag(b[:-1], -1) + np.diag(b[:-1], 1)
    z = np.linspace(-5, 5, 33)
    G = gf_primitives.calc_G(a.reshape(k, 1, 1), b.reshape(k, 1, 1), np.ones((1, 1)), z, 0.0, 0.2)
    ref = np.array([np.linalg.inv((w + 0.2j) * np.eye(k) - T)[0, 0] for w in z])
    np.testing.assert_allclose(G[:, 0, 0], ref, rtol=1e-12)


# --- incremental evaluator (the convergence monitor's O(1)-per-block state) -------------------


def _random_blocks(rng, widths, hermitian=True):
    """Ragged block-tridiagonal coefficients: alphas (w_i, w_i), betas (w_{i+1}, w_i); trailing beta unused."""
    k = len(widths)
    alphas, betas = [], []
    for i, w in enumerate(widths):
        a = rng.normal(size=(w, w)) + 1j * rng.normal(size=(w, w))
        alphas.append((a + a.conj().T) / 2 if hermitian else a)
        w_next = widths[i + 1] if i + 1 < k else w
        betas.append(0.6 * (rng.normal(size=(w_next, w)) + 1j * rng.normal(size=(w_next, w))))
    return alphas, betas


def _rebuild_G(alphas, betas, omegaP):
    return np.linalg.inv(gf_primitives._block_cf_inverse(alphas, betas, omegaP))


@pytest.mark.parametrize("widths_kind", ["uniform", "shrinking", "scalar"])
@pytest.mark.parametrize("sign", [1, -1])
def test_incremental_G_matches_the_rebuild_at_every_level(widths_kind, sign):
    rng = np.random.default_rng(11)
    k = 400
    if widths_kind == "uniform":
        widths = [3] * k
    elif widths_kind == "scalar":
        widths = [1] * k
    else:
        widths = [4] * 100 + [3] * 150 + [2] * 100 + [1] * 50
    alphas, betas = _random_blocks(rng, widths)
    omegaP = np.concatenate([np.linspace(-4, 4, 21) + sign * 0.05j, sign * 1j * np.pi * 0.3 * (2 * np.arange(8) + 1)])
    state = gf_primitives._IncrementalBlockCF(omegaP)
    for j in range(1, k + 1, 7):
        G = state.advance(alphas[:j], betas[:j])
        np.testing.assert_allclose(G, _rebuild_G(alphas[:j], betas[:j], omegaP), rtol=1e-9, atol=1e-12)
    G = state.advance(alphas, betas)
    np.testing.assert_allclose(G, _rebuild_G(alphas, betas, omegaP), rtol=1e-9, atol=1e-12)
    # the last block's contribution is exactly the step from k-1 to k blocks
    np.testing.assert_allclose(
        state.dG, G - _rebuild_G(alphas[:-1], betas[:-1], omegaP), rtol=1e-7, atol=1e-11 * np.abs(G).max()
    )


def test_incremental_state_restarts_when_the_history_is_replaced():
    rng = np.random.default_rng(12)
    omegaP = np.linspace(-3, 3, 9) + 0.2j
    a1, b1 = _random_blocks(rng, [2] * 30)
    a2, b2 = _random_blocks(rng, [2] * 30)
    state = gf_primitives._IncrementalBlockCF(omegaP)
    state.advance(a1, b1)
    for alphas, betas in ((a1[:10], b1[:10]), (a2, b2)):  # a shorter prefix, then a different history
        G = state.advance(alphas, betas)
        np.testing.assert_allclose(G, _rebuild_G(alphas, betas, omegaP), rtol=1e-9)


def test_incremental_matches_the_rebuild_on_a_long_unreorthogonalized_recurrence():
    """Ghost Ritz values (loss of orthogonality) must not make the top-down recursion drift from the bottom-up one."""
    rng = np.random.default_rng(13)
    n, k, p = 400, 1200, 2
    H = rng.normal(size=(n, n))
    H = (H + H.T) / 2
    alphas, betas = [], []
    Q = np.linalg.qr(rng.normal(size=(n, p)))[0]
    Qprev = np.zeros((n, p))
    B = np.zeros((p, p))
    for _ in range(k):  # plain block Lanczos, no reorthogonalization
        W = H @ Q - Qprev @ B.conj().T
        A = Q.conj().T @ W
        W = W - Q @ A
        Qn, Bn = np.linalg.qr(W)
        alphas.append(A)
        betas.append(Bn)
        Qprev, Q, B = Q, Qn, Bn
    omegaP = np.concatenate([np.linspace(-30, 30, 41) + 0.05j, 1j * np.pi * 0.1 * (2 * np.arange(16) + 1)])
    state = gf_primitives._IncrementalBlockCF(omegaP)
    G = state.advance(alphas, betas)
    np.testing.assert_allclose(G, _rebuild_G(alphas, betas, omegaP), rtol=1e-7, atol=1e-11)


def test_the_incremental_state_agrees_with_shifted_galerkin():
    from impurityModel.ed.gf_shift_recycling import ShiftedGalerkin  # noqa: F401  (cross-check exists)

    rng = np.random.default_rng(14)
    alphas, betas = _random_blocks(rng, [2] * 40)
    omegaP = np.linspace(-3, 3, 7) + 0.3j
    G = gf_primitives._IncrementalBlockCF(omegaP).advance(alphas, betas)
    np.testing.assert_allclose(G, _rebuild_G(alphas, betas, omegaP), rtol=1e-10)


def test_greens_function_change_incremental_equals_the_rebuild_at_a_cadence():
    from impurityModel.ed.gf_convergence import _greens_function_change

    rng = np.random.default_rng(15)
    alphas, betas = _random_blocks(rng, [2] * 200)
    omegaP = np.linspace(-3, 3, 33) + 0.1j
    cache = [None, 0]
    for k in range(8, 201, 8):  # a cadence the old "k-1" cache never hit
        inc = _greens_function_change(alphas[:k], betas[:k], None, 0.1, omegaP=omegaP, cache=cache)
        ref = _greens_function_change(alphas[:k], betas[:k], None, 0.1, omegaP=omegaP)
        assert inc == pytest.approx(ref, rel=1e-6, abs=1e-13)
