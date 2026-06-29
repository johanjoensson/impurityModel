r"""Brute-force orthogonality oracle for block-Lanczos partial reorthogonalization.

These tests pin the fix for the partial-reort estimator (``estimate_orthonormality``), which
previously under-predicted the orthogonality loss by orders of magnitude — its noise floor
``eps*(beta_i+beta_j)`` *shrank* with ``beta_i`` instead of growing as ``eps*||A||*||beta_i^-1||``,
and the signed three-term recurrence cancelled away the amplification — so on clustered /
near-degenerate spectra the bad-block trigger never fired and ``PARTIAL`` degenerated to (or
worse than) no reorthogonalization. The fix propagates the estimate in magnitude (a guaranteed
upper bound), restores the amplified noise floor, and renormalizes the block after the bad-block
projection.

The oracle runs the *real* array kernel and measures the true accumulated orthogonality
``||Q^H Q - I||`` against the semi-orthogonality target ``sqrt(eps)``, and checks eigenvalues
against a dense reference.
"""

import numpy as np
import pytest
import scipy.linalg as la

from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array, eigsh

SQRT_EPS = np.sqrt(np.finfo(float).eps)


def _hermitian_from_spectrum(d, seed):
    """Random Hermitian matrix with the given eigenvalues."""
    rng = np.random.default_rng(seed)
    n = d.size
    U = la.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    H = (U * d) @ U.conj().T
    return 0.5 * (H + H.conj().T)


def _seed_block(n, p, seed):
    rng = np.random.default_rng(seed + 1)
    return la.qr(rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p)), mode="economic")[0]


def _run(H, Q0, mode, max_iter):
    a, b, Q, w = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H.astype(complex),
        converged=lambda *x, **k: False,
        reort=mode,
        return_widths=True,
        max_iter=max_iter,
    )
    ortho = float(np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1])))
    return a, b, Q, w, ortho


# Realistic spectra where partial reort must hold semi-orthogonality. (The pathological
# "two tight clusters" — a rank-~2 problem — is covered separately below: PARTIAL recovers the
# eigenvalues with bounded beta at realistic iteration counts.)
_SPECTRA = {
    "clustered_near_deg": (lambda r: np.concatenate([r.normal(c, 0.02, 50) for c in range(8)]), 80),
    "uniform_dense": (lambda r: np.sort(r.uniform(-30, 10, 300)), 90),
    "degenerate_pairs": (lambda r: np.repeat(r.uniform(-5, 5, 150), 2), 80),
    "well_separated": (lambda r: np.linspace(-20.0, 20.0, 200), 70),
}


@pytest.mark.parametrize("name", list(_SPECTRA))
def test_partial_reort_holds_semiorthogonality(name):
    dgen, max_iter = _SPECTRA[name]
    d = dgen(np.random.default_rng(3))
    H = _hermitian_from_spectrum(d, seed=3)
    Q0 = _seed_block(d.size, 2, seed=3)

    _, _, _, _, ortho_none = _run(H, Q0, Reort.NONE, max_iter)
    a_p, b_p, _, _, ortho_partial = _run(H, Q0, Reort.PARTIAL, max_iter)
    _, _, _, _, ortho_full = _run(H, Q0, Reort.FULL, max_iter)

    # PARTIAL must hold semi-orthogonality (Paige-Simon guarantee) and never be worse than NONE.
    assert ortho_partial < 1e3 * SQRT_EPS, f"{name}: PARTIAL ||Q^HQ-I||={ortho_partial:.2e}"
    assert ortho_partial <= max(ortho_none, 1e3 * SQRT_EPS)
    assert ortho_full < 1e3 * SQRT_EPS  # sanity on the reference

    # Lowest Ritz values from the PARTIAL run match the dense reference.
    evals = np.asarray(eigsh(np.asarray(a_p), np.asarray(b_p), eigvals_only=True)[0]).real
    np.testing.assert_allclose(np.sort(evals)[:4], np.sort(d)[:4], atol=1e-6)


def test_partial_reort_recovers_tight_clusters():
    """Two tight clusters (effectively rank ~2). At a realistic iteration count — where the
    Green's-function convergence monitor stops once the two poles are resolved — PARTIAL must
    recover both cluster eigenvalues with bounded beta and good orthogonality. (Before the
    estimator fix this case corrupted to ||Q^HQ-I|| ~ 7e3; forcing it tens of iterations past
    its effective rank-2 still breaks down — only an absolute spectral-scale breakdown, a
    separate convergence-criterion item, bounds that, and real use never reaches it.)"""
    rng = np.random.default_rng(3)
    d = np.concatenate([rng.normal(0.0, 1e-3, 150), rng.normal(10.0, 1e-3, 150)])
    H = _hermitian_from_spectrum(d, seed=3)
    Q0 = _seed_block(d.size, 2, seed=3)

    a_p, b_p, _, _, ortho_partial = _run(H, Q0, Reort.PARTIAL, 30)
    max_beta = max(np.linalg.norm(x, 2) for x in b_p)

    assert max_beta < 10 * np.linalg.norm(H, 2)  # beta stays physical (~||H||)
    assert ortho_partial < 1e-1  # not the pre-fix 7e3 catastrophe
    evals = np.asarray(eigsh(np.asarray(a_p), np.asarray(b_p), eigvals_only=True)[0]).real
    lo, hi = float(np.min(evals)), float(np.max(evals))
    assert abs(lo - 0.0) < 0.05 and abs(hi - 10.0) < 0.05  # both clusters recovered
