r"""Brute-force orthogonality oracle for block-Lanczos partial reorthogonalization.

These tests pin the fixes for the partial-reort estimator (``estimate_orthonormality``), which
has twice under-predicted the orthogonality loss by orders of magnitude — so the bad-block trigger
never fired and ``PARTIAL`` degenerated to (or worse than) no reorthogonalization:

* its noise floor ``eps*(beta_i+beta_j)`` *shrank* with ``beta_i`` instead of growing as
  ``eps*||A||*||beta_i^-1||``, which broke clustered / near-degenerate spectra. Fixed by
  restoring the amplified floor (added in magnitude, so no sign cancellation — the three-term
  propagation itself stays *signed*, see the comment in ``estimate_orthonormality``) and
  renormalizing the block after the bad-block projection.
* the ``omega_{i+1,i}`` seed used ``beta_0`` as a stand-in for ``||A||``, which holds only for a
  *cold* start. Warm-started from converged eigenvectors it cancelled itself and estimated ``eps``
  where the truth is ``eps*||A||/||beta_0||``. Fixed by using the operator scale directly.

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


def test_partial_reort_survives_a_warm_start_from_converged_eigenvectors():
    """``omega_{i+1,i} ~ eps*||A||/beta_i``, not ``eps * beta_i^-H @ beta_0``.

    Those two agree for a *cold* start, where ``||beta_0|| ~ ||A||``. Warm-started from (nearly)
    converged eigenvectors ``beta_0`` is the eigenpair residual, and at ``i = 0`` the old
    expression collapsed to ``beta_0^-H @ beta_0 ~ I`` — an estimate of ``eps`` at the one step
    where the true overlap is largest, ``eps*||A||/||beta_0||``. The trigger never fired and the
    recurrence lost orthogonality completely.

    ``||A|| ~ 3e5`` is what makes this reachable without any other change: it keeps
    ``||beta_0|| = 9.2e-6`` just above ``_cholesky_or_deflate``'s absolute ``EPS**(1/3) = 6.06e-6``
    floor, so the residual block is not deflated away and the recurrence actually runs. Pre-fix
    this reached ``||Q^H Q - I|| = 0.73`` with ``max|beta| = 1.24e6`` against FULL's ``1.54e5``.
    """
    rng = np.random.default_rng(0)
    n, p = 400, 3
    d = np.sort(rng.standard_normal(n)) * 1e5
    U = la.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    H = 0.5 * (((U * d) @ U.conj().T) + ((U * d) @ U.conj().T).conj().T)

    # Warm start: the p lowest eigenvectors, nudged so the residual is small but nonzero.
    exact = U[:, np.argsort(d)[:p]]
    Q0 = la.qr(exact + 1e-12 * (rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))), mode="economic")[0]

    a_p, b_p, _, _, ortho_partial = _run(H, Q0, Reort.PARTIAL, 30)
    _, b_f, _, _, ortho_full = _run(H, Q0, Reort.FULL, 30)

    beta0 = np.linalg.norm(b_p[0], 2)
    assert beta0 > 6.06e-6, f"beta_0={beta0:.2e} deflated away; the test no longer bites"

    assert ortho_partial < 1e3 * SQRT_EPS, f"PARTIAL ||Q^HQ-I||={ortho_partial:.2e}"
    assert ortho_full < 1e3 * SQRT_EPS

    # beta must not run away past the FULL reference (pre-fix: 8x).
    max_beta_p = max(np.linalg.norm(x, 2) for x in b_p)
    max_beta_f = max(np.linalg.norm(x, 2) for x in b_f)
    assert max_beta_p < 2 * max_beta_f, f"PARTIAL max|beta|={max_beta_p:.2e} vs FULL {max_beta_f:.2e}"

    evals = np.asarray(eigsh(np.asarray(a_p), np.asarray(b_p), eigvals_only=True)[0]).real
    np.testing.assert_allclose(np.sort(evals)[:p], np.sort(d)[:p], rtol=1e-8)
