"""The block factorization answers two questions, and they have different scales.

* **Rank deficiency** is a statement about column *directions*: judged relative to the block's
  own largest singular value, so it is scale invariant.
* **Breakdown** — is the block numerically zero? — is absolute, and therefore needs a reference.
  A Lanczos residual block is zero when it is negligible against ``||H||``, not against ``1``.

The two used to be fused into ``evals > DEFLATE_EVAL_TOL * max(evals[-1], 1.0)``. That ``1.0``
clamp made the *rank* test absolute for any block below the then-floor ``EPS**(1/3) ~ 6.06e-6``: a
small but perfectly well-conditioned block was declared rank 0. Every warm-started Krylov solve
was then handed back its own input and reported success — ``block_bicgstab`` returned ``x0``
unrefined, and TRLM warm-started from converged eigenvectors returned them unimproved at
``||r|| = 2.2e-9``, capping the ground-state accuracy whatever ``tol`` asked for. Meanwhile
``BREAKDOWN_TOL`` was declared, documented, and referenced nowhere.

These tests pin both halves against the **live** factorization (:func:`TSQR.tsqr`, which every
block-Krylov call site now goes through): a small well-conditioned block survives, a
rank-deficient one deflates by exactly its null dimension, and an invariant subspace is still
detected — at ``||H||`` spanning twelve orders of magnitude. The historic ``_cholesky_or_deflate``
is exercised only where the point *is* the historic contrast (``test_block_lanczos_blowup``).
"""

import numpy as np
import pytest

from impurityModel.ed.BlockLanczosArray import (
    BREAKDOWN_TOL,
    Reort,
    block_lanczos_array,
    eigsh,
)
from impurityModel.ed.TSQR import tsqr


def _block(n, p, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p)))[0]
    return q * scale


# --------------------------------------------------------------------------------------
# The rank test is relative
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("norm", [1e2, 1.0, 1e-4, 1e-8, 1e-11])
def test_a_small_well_conditioned_block_keeps_its_rank(norm):
    """Pre-fix: any block with ``||W|| < 6.06e-6`` came back as rank 0, however well conditioned.

    The range stops at ``1e-11`` because ``scale`` defaults to 1: below ``BREAKDOWN_TOL = 1e-12``
    the block *is* zero by the caller's own reference, and rank 0 is then the right answer.
    """
    assert norm > BREAKDOWN_TOL  # otherwise breakdown, correctly, wins
    w = _block(40, 3, seed=1, scale=norm)
    q, beta, k, _ = tsqr(w)

    assert k == 3, f"||W||={norm:.0e} deflated to rank {k}"
    # The factorization is exact: W = Q @ beta with Q orthonormal.
    np.testing.assert_allclose(np.conj(q.T) @ q, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(q @ beta, w, rtol=1e-8, atol=1e-16 * norm)


@pytest.mark.parametrize("norm", [1e2, 1.0, 1e-8])
def test_rank_deficiency_is_detected_at_every_scale(norm):
    """A genuinely dependent column deflates whatever the block's overall size."""
    w = _block(40, 3, seed=2, scale=norm)
    w[:, 2] = w[:, 0] + 1e-14 * w[:, 1]  # third column is (numerically) the first
    k = tsqr(w)[2]
    assert k == 2, f"||W||={norm:.0e}: expected rank 2, got {k}"


def test_a_tiny_block_is_not_rank_deficient_just_because_it_is_tiny():
    """The boundary case the old clamp got wrong: a block tiny in *absolute* terms, cond ~ 1.

    Anchored to ``BREAKDOWN_TOL``, not to the rank floor: "tiny" here is an absolute
    statement and the rank floor is a relative one, so ``BREAKDOWN_TOL`` is the only constant
    that can place this scenario. (It used to be sized as ``DEFLATE_TOL / 100``, which worked
    only while the rank test still carried an absolute clamp.)
    """
    w = _block(40, 2, seed=3, scale=1e4 * BREAKDOWN_TOL)
    cond = np.linalg.cond(np.conj(w.T) @ w)
    assert cond < 10.0, f"test block is not well conditioned (cond={cond:.1e})"
    assert tsqr(w)[2] == 2


# --------------------------------------------------------------------------------------
# The breakdown test is absolute, against `scale`
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_breakdown_fires_relative_to_the_supplied_scale(scale):
    zero = _block(40, 2, seed=4, scale=0.1 * BREAKDOWN_TOL * scale)
    alive = _block(40, 2, seed=4, scale=10.0 * BREAKDOWN_TOL * scale)

    assert tsqr(zero, None, scale)[2] == 0
    assert tsqr(alive, None, scale)[2] == 2

    # ... and with the default scale=1 the same block is judged against 1, not against `scale`.
    assert tsqr(alive)[2] == (0 if scale < 1.0 else 2)


def test_an_exactly_zero_block_always_breaks_down():
    w = np.zeros((40, 2), dtype=complex)
    for scale in (0.0, 1.0, 1e12):
        assert tsqr(w, None, scale)[2] == 0


# --------------------------------------------------------------------------------------
# End to end, through the array sweep
# --------------------------------------------------------------------------------------


def _hermitian(n, spread, seed=7):
    rng = np.random.default_rng(seed)
    d = np.sort(rng.standard_normal(n)) * spread
    u = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h = (u * d) @ np.conj(u.T)
    return 0.5 * (h + np.conj(h.T)), d, u


def _sweep(h, psi0, max_iter, reort=Reort.PARTIAL):
    return block_lanczos_array(
        psi0=psi0.copy(),
        h_op=h.astype(complex),
        converged=lambda *a, **k: False,
        max_iter=max_iter,
        verbose=False,
        reort=reort,
        return_widths=True,
        return_status=True,
    )


@pytest.mark.parametrize("spread", [1e-6, 1.0, 1e6])
def test_an_invariant_start_block_breaks_down_at_every_operator_scale(spread):
    """Breakdown detection must not depend on how H is scaled."""
    h, d, u = _hermitian(60, spread)
    psi0 = u[:, np.argsort(d)[:2]]  # an exact invariant subspace
    alphas, _, _, _, status = _sweep(h, psi0, 20)
    assert status == "invariant_subspace"
    assert len(alphas) == 1


@pytest.mark.parametrize("spread", [1e-6, 1.0, 1e6])
def test_a_cold_start_never_breaks_down_early(spread):
    h, d, _ = _hermitian(60, spread)
    rng = np.random.default_rng(11)
    psi0 = np.linalg.qr(rng.standard_normal((60, 2)) + 0j)[0]
    alphas, betas, _, _, status = _sweep(h, psi0, 20)
    assert status == "max_iter" and len(alphas) == 20
    got = np.sort(np.asarray(eigsh(np.asarray(alphas), np.asarray(betas), eigvals_only=True)[0]).real)
    np.testing.assert_allclose(got[:2] / spread, np.sort(d)[:2] / spread, atol=1e-12)


def _nearly_invariant_start(n, p, noise, seed=0, spread=50.0):
    """A start block that is an eigenspace to within ``noise`` — so ``beta_0`` *is* its residual."""
    h, d, u = _hermitian(n, spread, seed=seed)
    rng = np.random.default_rng(seed)
    exact = u[:, np.argsort(d)[:p]]
    psi0 = np.linalg.qr(exact + noise * (rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p))))[0]
    return h, d, psi0


@pytest.mark.parametrize("noise", [1e-9, 1e-12])
def test_a_warm_start_above_the_breakdown_floor_is_refined(noise):
    """The live boundary, approached from above.

    ``beta_0`` of a nearly-converged start block *is* its eigenpair residual. The only
    threshold that may stop the sweep is the absolute one, ``BREAKDOWN_TOL * ||H||``: the rank
    test is relative, so however small ``beta_0`` gets, a well-conditioned block keeps its
    rank. (Historically the rank test carried an absolute clamp at ``EPS**(1/3) = 6.06e-6``,
    and every warm start below it was deflated to rank 0 and handed straight back --
    ``block_bicgstab`` returned ``x0`` unrefined and TRLM returned converged eigenvectors
    unimproved, whatever ``tol`` asked for.)
    """
    n, p = 400, 3
    h, d, psi0 = _nearly_invariant_start(n, p, noise)
    beta0 = np.linalg.norm(_sweep(h, psi0, 30)[1][0], 2)
    h_norm = np.linalg.norm(h, 2)
    assert beta0 > BREAKDOWN_TOL * h_norm, (
        f"beta_0={beta0:.2e} is below the live breakdown floor {BREAKDOWN_TOL * h_norm:.2e}; "
        "this scenario no longer tests refinement"
    )

    alphas, betas, _, _, status = _sweep(h, psi0, 30)
    assert status == "max_iter", f"warm start stopped early: {status}"
    assert len(alphas) == 30

    # And the refinement actually buys accuracy: the Ritz values beat the start block's own
    # residual by orders of magnitude.
    got = np.sort(np.asarray(eigsh(np.asarray(alphas), np.asarray(betas), eigvals_only=True)[0]).real)
    np.testing.assert_allclose(got[:p], np.sort(d)[:p], atol=1e-10)


def test_a_warm_start_below_the_breakdown_floor_is_declared_invariant():
    """The same boundary from below: once ``beta_0`` is negligible *against* ``||H||`` there is
    nothing left to refine, and stopping is the correct answer rather than a regression."""
    n, p = 400, 3
    h, _, psi0 = _nearly_invariant_start(n, p, noise=0.0)
    h_norm = np.linalg.norm(h, 2)

    betas = _sweep(h, psi0, 30)[1]
    beta0 = np.linalg.norm(betas[0], 2) if len(betas) else 0.0
    assert beta0 <= BREAKDOWN_TOL * h_norm

    alphas, _, _, _, status = _sweep(h, psi0, 30)
    assert status == "invariant_subspace"
    assert len(alphas) == 1
