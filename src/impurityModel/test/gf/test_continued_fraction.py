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
