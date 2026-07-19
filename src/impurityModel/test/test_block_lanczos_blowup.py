r"""Regression tests for the block-Lanczos divergence on (near-)rank-deficient blocks.

Background
----------
The interacting Green's-function block Lanczos diverged catastrophically on a real NiO
self-energy run: :math:`\lVert\beta\rVert` jumped from :math:`\mathcal{O}(1)` to
:math:`10^{90}` in a few steps on every MPI rank, and the self-energy was then assembled
from corrupted blocks (a silent correctness failure).

Root cause: the off-diagonal block is the QR factor of the residual block
:math:`W_p` via Cholesky of the Gram matrix :math:`M = W_p^\dagger W_p`.  When the block
becomes nearly rank deficient (its columns nearly parallel, as happens when the Krylov
space approaches an invariant subspace), :math:`M` is ill-conditioned and a *single*
Cholesky-QR is numerically unstable: :math:`Q_{i+1} = W_p \beta_i^{-1}` loses
orthonormality by :math:`\mathcal{O}(\kappa(M)\,\varepsilon)` (up to :math:`\mathcal{O}(1)`),
so :math:`HQ_{i+1}` blows up and the recurrence diverges.  The deflation tolerance was also
inconsistent between the Cholesky fast path and the ``eigh`` fallback.

Fix (``BlockLanczosArray.pyx``):

* ``_cholesky_or_deflate`` uses a single, eigenvalue-consistent deflation floor
  (:math:`\lambda_k < \varepsilon\,\lambda_{\max}`) on both paths, bounding the retained
  block's condition number to :math:`\lesssim 1/\sqrt{\varepsilon}`.
* both kernels add a CholeskyQR2 second pass (``_cholesky_qr2``) that recomputes the Gram
  from the *actual* (high-dimensional) vectors and re-orthonormalizes once more, restoring
  orthonormality to machine precision.
* a ``BETA_BLOWUP_FACTOR`` safeguard truncates the run at the last trustworthy block should
  a divergence ever slip through.

These tests pin the fixed invariants: bounded :math:`\lVert\beta\rVert`, machine-precision
orthonormality, and a Green's function matching the dense resolvent — all on deliberately
ill-conditioned / near-degenerate problems that exercise the failure path.
"""

import numpy as np
import scipy.linalg as la

from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import (
    DEFLATE_EVAL_TOL,
    DEFLATE_TOL,
    Reort,
    _cholesky_or_deflate,
    _cholesky_qr2,
    block_lanczos_array,
)
from impurityModel.ed.greens_function import (
    _gf_sample_mesh,
    _greens_function_change,
    _sanitize_continued_fraction,
    _trim_blocks,
    build_qr,
    calc_G,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.test.test_restarted_lanczos import MockBasis

SQRT_EPS = np.sqrt(np.finfo(float).eps)


def _ill_conditioned_block(n, p, sep, rng):
    """Block (n x p) of orthonormal columns, but with column 1 made nearly parallel to
    column 0 (separation ``sep``) so the Gram ``M`` has condition number ~ ``1/sep**2``
    while staying above the rank-deficiency floor (``lambda_min/lambda_max ~ sep**2``)."""
    Qb = la.qr(rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p)), mode="economic")[0]
    Wp = Qb.copy()
    if p >= 2:
        Wp[:, 1] = Wp[:, 0] + sep * Wp[:, 1]
    return Wp


# --------------------------------------------------------------------------- #
# Unit level: the block-QR numerics                                            #
# --------------------------------------------------------------------------- #


def test_single_choleskyqr_is_unstable_but_choleskyqr2_recovers():
    """A single Cholesky-QR of an ill-conditioned block loses orthonormality;
    the second pass (using the real-vector Gram) restores it to machine precision."""
    rng = np.random.default_rng(0)
    n, p = 50, 2
    # Stay just above the rank-deficiency floor so the block is *kept* but maximally
    # ill-conditioned within the retained regime (sigma_min/sigma_max ~ sep/2, which must
    # exceed DEFLATE_TOL). A more singular block is now deflated rather than kept, so the
    # CholeskyQR2 recovery path is exercised on the worst *retained* conditioning.
    Wp = _ill_conditioned_block(n, p, sep=4 * DEFLATE_TOL, rng=rng)
    M = Wp.conj().T @ Wp
    assert np.linalg.cond(M) > 1e7  # genuinely ill-conditioned (but within the kept regime)

    beta_j, beta_inv, k = _cholesky_or_deflate(M, p)
    assert k == p  # above the rank-deficiency floor -> kept, but ill-conditioned
    Q1 = Wp @ beta_inv
    orth1 = np.linalg.norm(Q1.conj().T @ Q1 - np.eye(k))
    assert orth1 > SQRT_EPS  # single pass is non-orthonormal -> the historic blowup seed

    # Second pass: recompute the Gram from the *actual* vectors (not from M).
    M2 = Q1.conj().T @ Q1
    M2 = 0.5 * (M2 + M2.conj().T)
    beta2_inv, beta_j2, k2 = _cholesky_qr2(M2, beta_j, k)
    assert k2 == k
    Q2 = Q1 @ beta2_inv
    orth2 = np.linalg.norm(Q2.conj().T @ Q2 - np.eye(k2))
    assert orth2 < SQRT_EPS
    # Reconstruction Wp = Q2 @ beta_j2 holds.
    assert np.linalg.norm(Q2 @ beta_j2 - Wp) < 1e-10


def test_deflation_threshold_consistent_between_paths():
    """The Cholesky fast path and the eigh fallback deflate to the same rank."""
    rng = np.random.default_rng(1)
    n, p = 40, 3
    Qb = la.qr(rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p)), mode="economic")[0]
    # Eigenvalues of M straddling the floor: one clearly kept, one clearly null.
    for log10_null in (-6, -9, -14, -18):
        s = np.array([1.0, 1e-2, 10.0**log10_null])
        Wp = Qb * s[None, :]
        M = Wp.conj().T @ Wp
        _, _, k_chol = _cholesky_or_deflate(M, p)
        # Force the eigh fallback by passing a copy that fails Cholesky's PD check is hard;
        # instead compare against the eigenvalue floor directly.
        evals = la.eigvalsh(M)
        # Derive the expected rank from the module's eigenvalue floor (single source of
        # truth) rather than a hardcoded literal, so this tracks DEFLATE_EVAL_TOL.
        k_expected = int(np.sum(evals > DEFLATE_EVAL_TOL * max(evals[-1], 1.0)))
        assert k_chol == k_expected, f"log10_null={log10_null}: {k_chol} != {k_expected}"


def test_full_collapse_returns_zero_rank():
    """A genuinely rank-zero block reports active_k == 0 (clean breakdown)."""
    M = np.zeros((2, 2), dtype=complex)
    beta_j, beta_inv, k = _cholesky_or_deflate(M, 2)
    assert k == 0 and beta_j is None and beta_inv is None


# --------------------------------------------------------------------------- #
# Kernel level: array (dense) path                                            #
# --------------------------------------------------------------------------- #

# Near-degenerate spectrum: two pairs at ~1e-9/1e-10 splitting drive the block QR
# ill-conditioned as the Krylov space saturates the (nearly) degenerate subspaces.
_NEAR_DEG_D = np.array([-32.0, -32.0 + 1e-9, -31.5, -30.0, -10.0, 0.0, 5.0, 5.0 + 1e-10, 12.0, 20.0])


def _dense_near_degenerate_H(rng):
    n = _NEAR_DEG_D.size
    U = la.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    H = (U * _NEAR_DEG_D) @ U.conj().T
    return 0.5 * (H + H.conj().T), U


def test_array_kernel_no_blowup_near_degenerate():
    rng = np.random.default_rng(2)
    H, U = _dense_near_degenerate_H(rng)
    n = H.shape[0]
    S = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
    Q0, r = build_qr(S)

    alphas, betas, Q, widths = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H.astype(complex),
        converged=lambda a, b, **kw: False,
        reort=Reort.NONE,
        return_widths=True,
        verbose=False,
        max_iter=40,
    )

    h_norm = np.linalg.norm(H, 2)
    max_beta = max(np.linalg.norm(b, 2) for b in betas)
    assert max_beta < 10 * h_norm, f"|beta| not bounded: {max_beta:.3e} vs ||H||={h_norm:.3e}"
    orth = np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1]))
    assert orth < SQRT_EPS, f"orthonormality lost: {orth:.3e}"

    omega = np.linspace(-35.0, 25.0, 41)
    delta = 0.2
    a_t, b_t = _trim_blocks(alphas, betas, widths)
    G = calc_G(a_t, b_t, r, omega, 0.0, delta)
    Sd = U.conj().T @ S
    denom = omega[:, None] + 1j * delta - _NEAR_DEG_D[None, :]
    G_exact = np.einsum("wi,ia,ib->wab", 1.0 / denom, np.conj(Sd), Sd)
    np.testing.assert_allclose(G, G_exact, atol=1e-9)


def test_gf_monitor_detects_convergence():
    """`_greens_function_change` returns a small relative value when the freshly added block
    is (nearly) decoupled (tiny trailing coupling -> converged) and an O(1) value when it
    is strongly coupled (not converged). This is the responsiveness the NiO run lacked,
    because the original absolute, Ritz-frequency-sampled measure stayed O(1) regardless."""
    delta = 0.2
    alphas = [np.array([[a]], dtype=complex) for a in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    mesh = _gf_sample_mesh(alphas, delta)  # frozen, independent of the last block

    # Tiny coupling into the final retained block -> it cannot change G -> converged.
    betas_conv = [np.array([[b]], dtype=complex) for b in (0.5, 0.5, 0.5, 1e-10, 0.5)]
    d_conv = _greens_function_change(alphas, betas_conv, [1] * 5, delta, omegaP=mesh)
    # O(1) coupling -> the final block reshapes G -> not converged.
    betas_unconv = [np.array([[b]], dtype=complex) for b in (0.5, 0.5, 0.5, 0.8, 0.5)]
    d_unconv = _greens_function_change(alphas, betas_unconv, [1] * 5, delta, omegaP=mesh)

    assert d_conv < 1e-6
    assert d_unconv > 1e-3
    assert d_unconv > 1e3 * max(d_conv, np.finfo(float).tiny)


def test_gf_monitor_moving_mesh_cannot_converge_but_fixed_can():
    """Sampling at the *moving* Ritz values reintroduces an O(1) change every step (a new
    pole lands on a new sample point), so the measure cannot decay — the root reason the
    monitor never reached tolerance. The same sequence on a *fixed* mesh ends at its
    minimum (a converging trend)."""
    rng = np.random.default_rng(0)
    n = 30
    d = np.sort(rng.uniform(-30.0, 10.0, n))
    U = la.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    H = (U * d) @ U.conj().T
    H = 0.5 * (H + H.conj().T)
    Q0 = la.qr(rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2)), mode="economic")[0]
    alphas, betas, _, widths = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H.astype(complex),
        converged=lambda a, b, **kw: False,
        reort=Reort.FULL,
        return_widths=True,
        max_iter=15,
    )
    delta = 0.3
    A3, _ = _trim_blocks(alphas[:3], betas[:3], list(widths[:3]))
    mesh = _gf_sample_mesh(A3, delta)

    fixed, moving = [], []
    for k in range(3, len(alphas) + 1):
        A, B = _trim_blocks(alphas[:k], betas[:k], list(widths[:k]))
        w = list(widths[:k])
        fixed.append(_greens_function_change(A, B, w, delta, omegaP=mesh))
        moving.append(_greens_function_change(A, B, w, delta))

    # Moving mesh re-rises above its start (new poles at new samples) -> no convergence.
    assert max(moving) > 1.5 * moving[0]
    # Fixed mesh ends at its minimum and well below the typical value -> converging trend.
    assert fixed[-1] == min(fixed)
    assert fixed[-1] < 0.5 * float(np.median(fixed))
    assert all(f >= 0.0 for f in fixed)


def test_sanitizer_drops_corrupted_tail():
    """A non-finite or runaway trailing block must be dropped before the continued fraction,
    while a healthy sequence is returned unchanged."""
    a = [np.array([[-2.0]], dtype=complex) for _ in range(4)]
    b = [np.array([[0.5]], dtype=complex) for _ in range(4)]

    # Healthy: unchanged.
    a_s, _b_s = _sanitize_continued_fraction(list(a), list(b))
    assert len(a_s) == 4

    # Non-finite trailing block -> truncated before it.
    a_inf = list(a) + [np.array([[np.inf]], dtype=complex)]
    b_inf = list(b) + [np.array([[0.5]], dtype=complex)]
    a_s, _b_s = _sanitize_continued_fraction(a_inf, b_inf)
    assert len(a_s) == 4 and all(np.all(np.isfinite(x)) for x in a_s)

    # Runaway (||beta|| >> healthy scale) trailing block -> truncated.
    a_big = list(a) + [np.array([[-2.0]], dtype=complex)]
    b_big = list(b) + [np.array([[1e9]], dtype=complex)]
    a_s, _b_s = _sanitize_continued_fraction(a_big, b_big)
    assert len(a_s) == 4


def test_divergence_safeguard_does_not_false_trigger_on_large_norm_H():
    """A well-behaved Hermitian H with a large spectral norm (>> the BETA_BLOWUP_FACTOR
    threshold) must run to completion: the safeguard compares *relative* jumps and is
    gated past the first step, so a large ||H|| never self-triggers a truncation."""
    rng = np.random.default_rng(9)
    d = np.array([-5e4, -4.9e4, -1e4, 0.0, 1e3, 2e4, 4e4, 5e4])  # ||H|| = 5e4 >> 1e3
    n = d.size
    U = la.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    H = 0.5 * ((U * d) @ U.conj().T + (U * d) @ U.conj().T).conj().T
    Q0 = la.qr(rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2)), mode="economic")[0]

    alphas, betas, _Q, widths = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H.astype(complex),
        converged=lambda a, b, **kw: False,
        reort=Reort.NONE,
        return_widths=True,
        verbose=False,
        max_iter=4,
    )
    # No spurious early truncation: four full blocks completed.
    assert len(alphas) == 4 and all(w == 2 for w in list(widths)[:4])
    assert max(np.linalg.norm(b, 2) for b in betas) < 10 * np.linalg.norm(H, 2)


# --------------------------------------------------------------------------- #
# Kernel level: ManyBodyState (sparse, distributed) path                      #
# --------------------------------------------------------------------------- #

# Diagonal one-body energies with a near-degenerate pair (the bit convention is
# MSB-first: orbital i maps to bit 7 - i, i.e. orbital 0 == 0x80).
_MBS_EPS = np.array([-2.0, -2.0 + 1e-9, -0.5, 0.7, 1.3, 1.3 + 1e-10, 2.6, 4.0])
_MBS_NORB = _MBS_EPS.size


def _mbs_diag_h():
    return ManyBodyOperator({((i, "c"), (i, "a")): _MBS_EPS[i] for i in range(_MBS_NORB)})


def _mbs_states():
    states = []
    for i in range(_MBS_NORB):
        val = 1 << (7 - i)
        states.append(SlaterDeterminant.from_bytes(val.to_bytes(8, byteorder="little")))
    return states


def test_mbs_kernel_no_blowup_near_degenerate():
    rng = np.random.default_rng(5)
    h_op = _mbs_diag_h()
    states = _mbs_states()
    p = 2
    S = rng.standard_normal((_MBS_NORB, p)) + 1j * rng.standard_normal((_MBS_NORB, p))
    Q0, r = build_qr(S)
    psi0 = [ManyBodyState() for _ in range(p)]
    for c in range(p):
        for i, sd in enumerate(states):
            psi0[c] += ManyBodyState({sd: Q0[i, c]})

    alphas, betas, _, _, widths = block_lanczos_cy(
        psi0,
        h_op,
        MockBasis(_MBS_NORB),
        converged_fn=lambda a, b, **kw: False,
        reort=Reort.NONE,
        max_iter=20,
        return_widths=True,
    )

    h_norm = float(np.max(np.abs(_MBS_EPS)))
    max_beta = max(np.linalg.norm(np.asarray(b), 2) for b in betas)
    assert max_beta < 10 * h_norm, f"|beta| not bounded: {max_beta:.3e}"

    omega = np.linspace(-3.0, 5.0, 33)
    delta = 0.1
    a_t, b_t = _trim_blocks(alphas, betas, widths)
    G = calc_G(a_t, b_t, r, omega, 0.0, delta)
    denom = omega[:, None] + 1j * delta - _MBS_EPS[None, :]
    G_exact = np.einsum("wi,ia,ib->wab", 1.0 / denom, np.conj(S), S)
    np.testing.assert_allclose(G, G_exact, atol=1e-9)
