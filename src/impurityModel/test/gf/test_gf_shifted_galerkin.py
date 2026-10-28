"""``ShiftedGalerkin``: the per-shift Galerkin solutions of the GF recurrence, against dense linear algebra.

The blocks come from the real ``gf_recurrence.lanczos_step`` (TSQR deflation included), so the conventions of
``alpha`` and ``beta`` are the recurrence's, not a hand-built T's. The oracle is ``V_j (z - T_j)^{-1} E_1`` from a
dense solve and the existing ``_shifted_tridiag_solutions`` (same conventions, banded).
"""

import numpy as np
import pytest

from impurityModel.ed.gf_recurrence import lanczos_step
from impurityModel.ed.gf_shift_recycling import ShiftedGalerkin, _shifted_tridiag_solutions
from impurityModel.ed.TSQR import DEFLATE_TOL_SEEDS

N = 48
ZS = [0.3 + 0.2j, -0.4 + 0.05j, 0.1 - 0.3j, -1.2 - 0.02j]  # both half planes (the removal side has Im z < 0)


def _hermitian(complex_valued, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((N, N)) + (1j * rng.standard_normal((N, N)) if complex_valued else 0)
    return (a + a.conj().T) / 2


def _seeds(h, deflating):
    rng = np.random.default_rng(11)
    first = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    if not deflating:
        second = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    else:  # an exact eigenvector: the block Krylov space loses a column after one step
        second = np.linalg.eigh(h)[1][:, 3]
        first = first - second * (second.conj() @ first)
    q = np.column_stack([first, second])
    return np.linalg.qr(q)[0] if not deflating else q / np.linalg.norm(q, axis=0)


def _run(h, q0, steps, galerkin):
    x, x_prev, beta_prev = q0, None, None
    alphas, betas, widths, blocks = [], [], [], []
    for _ in range(steps):
        alpha, x_next, beta, k = lanczos_step(
            lambda v: h @ v, x, x_prev, beta_prev, scale=np.linalg.norm(h, 2), deflate_tol=DEFLATE_TOL_SEEDS
        )
        galerkin.step(x, alpha, beta_prev)
        alphas.append(alpha)
        widths.append(x.shape[1])
        blocks.append(x)
        if k <= 0:
            return alphas, betas, widths, blocks, None
        betas.append(beta)
        res = galerkin.residuals(beta)
        yield alphas, betas, widths, blocks, x_next, res
        x_prev, beta_prev, x = x, beta, x_next


def _dense_t(alphas, betas, widths):
    starts = np.concatenate([[0], np.cumsum(widths)]).astype(int)
    t = np.zeros((starts[-1], starts[-1]), dtype=complex)
    for i, a in enumerate(alphas):
        t[starts[i] : starts[i + 1], starts[i] : starts[i + 1]] = a
        if i < len(betas) and i + 1 < len(widths):
            b = betas[i]
            t[starts[i + 1] : starts[i + 2], starts[i] : starts[i + 1]] = b
            t[starts[i] : starts[i + 1], starts[i + 1] : starts[i + 2]] = b.conj().T
    return t


@pytest.mark.parametrize("complex_h", [False, True])
@pytest.mark.parametrize("deflating", [False, True])
def test_the_solutions_equal_the_dense_galerkin_solves_at_every_step(complex_h, deflating):
    h = _hermitian(complex_h)
    q0 = _seeds(h, deflating)
    g = ShiftedGalerkin(ZS, N)
    seen_deflation = False
    for alphas, betas, widths, blocks, _x_next, res in _run(h, q0, 6, g):
        if min(widths) < 2:
            seen_deflation = True
        v = np.concatenate(blocks, axis=1)
        t = _dense_t(alphas, betas[:-1], widths)
        e1 = np.zeros((t.shape[0], 2), dtype=complex)
        e1[:2] = np.eye(2)
        for z, x in zip(ZS, g.solutions()):
            want = v @ np.linalg.solve(z * np.eye(t.shape[0]) - t, e1)
            np.testing.assert_allclose(x, want, atol=1e-10)
        # The scalar residual is the true one: (z - H) x - q0 = -v_{j+1} b d^-1 zeta, so its norm per column.
        for i, z in enumerate(ZS):
            true = np.linalg.norm((z * np.eye(N) - h) @ g.solutions()[i] - q0, axis=0)
            np.testing.assert_allclose(res[i], true, rtol=1e-6, atol=1e-9)
    assert seen_deflation == deflating


def test_the_residual_agrees_with_the_banded_batch_solver():
    h = _hermitian(True)
    q0 = _seeds(h, False)
    g = ShiftedGalerkin(ZS, N)
    for alphas, betas, widths, _blocks, _x, res in _run(h, q0, 5, g):  # noqa: B007
        pass
    a = np.zeros((len(alphas), 2, 2), dtype=complex)
    b = np.zeros_like(a)
    for i, al in enumerate(alphas):
        a[i] = al
        b[i] = betas[i]
    _y, rn = _shifted_tridiag_solutions(a, b, widths, np.eye(2, dtype=complex), ZS)
    np.testing.assert_allclose(np.linalg.norm(res, axis=1), rn, rtol=1e-6)


def test_rows_admitted_late_are_zero_padded():
    h = _hermitian(False)
    q0 = _seeds(h, False)
    g = ShiftedGalerkin(ZS, 10)
    g.step(q0[:10], np.eye(2), None)
    g.pad_rows(N)
    assert all(x.shape[0] == N and not x[10:].any() for x in g.solutions())
