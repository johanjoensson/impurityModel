"""Unit tests for the small dense EA16 numerics in ``impurityModel.ed.ea16``.

These helpers underpin the IRLM/TRLM restart logic; they operate only on the small
block-tridiagonal ``T`` and its eigenpairs, so they are cheap to pin down against
direct linear-algebra oracles.
"""

import numpy as np
import pytest

from impurityModel.ed.ea16 import (
    EPS,
    acceptance_tol,
    locked_overlap_step,
    operator_norm_estimate,
    purge_restart,
    ritz_residual_norms,
    select_restart_indices,
)


def _hermitian_block_tridiagonal(m, p, seed=0):
    rng = np.random.default_rng(seed)
    N = m * p
    T = np.zeros((N, N), dtype=complex)
    for i in range(m):
        a = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        T[i * p : (i + 1) * p, i * p : (i + 1) * p] = a + a.conj().T
        if i < m - 1:
            b = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
            T[i * p : (i + 1) * p, (i + 1) * p : (i + 2) * p] = b
            T[(i + 1) * p : (i + 2) * p, i * p : (i + 1) * p] = b.conj().T
    return T


# --------------------------------------------------------------------------- #
# ritz_residual_norms
# --------------------------------------------------------------------------- #
def test_ritz_residual_norms_matches_direct_formula():
    rng = np.random.default_rng(1)
    m, p = 4, 2
    beta_last = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
    Z = rng.standard_normal((m * p, m * p)) + 1j * rng.standard_normal((m * p, m * p))
    got = ritz_residual_norms(beta_last, Z, p)
    expected = np.linalg.norm(beta_last @ Z[-p:, :], axis=0)
    assert got.shape == (m * p,)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_ritz_residual_norms_zero_coupling_gives_zero():
    Z = np.random.default_rng(2).standard_normal((6, 6))
    got = ritz_residual_norms(np.zeros((2, 2)), Z, 2)
    np.testing.assert_allclose(got, 0.0, atol=1e-14)


# --------------------------------------------------------------------------- #
# operator_norm_estimate
# --------------------------------------------------------------------------- #
def test_operator_norm_estimate_largest_magnitude():
    assert operator_norm_estimate(np.array([-3.0, 1.0, 2.0])) == 3.0


def test_operator_norm_estimate_includes_locked():
    assert operator_norm_estimate(np.array([1.0, 2.0]), np.array([5.0])) == 5.0


def test_operator_norm_estimate_empty_active():
    assert operator_norm_estimate(np.array([])) == 0.0
    assert operator_norm_estimate(np.array([]), np.array([4.0])) == 4.0


# --------------------------------------------------------------------------- #
# acceptance_tol
# --------------------------------------------------------------------------- #
def test_acceptance_tol_formula():
    got = acceptance_tol(theta_i=-2.0, tnorm=10.0, cntl2=1e-3, cntl3=1e-2, u=1e-15)
    expected = 1e-15 * 10.0 + 1e-3 + 1e-2 * 2.0
    assert got == pytest.approx(expected)


def test_acceptance_tol_uses_absolute_values():
    # Negative CNTL knobs must not cancel the tolerance.
    assert acceptance_tol(1.0, 0.0, -0.5, -0.5, u=0.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# select_restart_indices
# --------------------------------------------------------------------------- #
def test_select_smallest_keeps_lowest_and_shifts_rest():
    theta = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    kept, shifts = select_restart_indices(theta, n_keep=2, locked_local=[])
    assert set(kept.tolist()) == {1, 3}  # values 1.0 and 2.0
    assert sorted(shifts.tolist()) == [3.0, 4.0, 5.0]


def test_select_largest_keeps_highest():
    theta = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    kept, _ = select_restart_indices(theta, n_keep=2, locked_local=[], which="largest")
    assert set(kept.tolist()) == {0, 4}  # values 5.0 and 4.0


def test_select_excludes_locked_local():
    theta = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    kept, _ = select_restart_indices(theta, n_keep=2, locked_local=[1])
    # idx 1 (value 1.0) is locked away, so the two smallest remaining are 2.0, 3.0.
    assert set(kept.tolist()) == {2, 3}


def test_select_ghost_filter_pushes_out_near_locked_value():
    theta = np.array([1.0, 2.0, 3.0])
    kept, _ = select_restart_indices(theta, n_keep=2, locked_local=[], locked_evals=np.array([2.0]), ghost_tol=1e-6)
    # idx 1 (== a locked eigenvalue) is treated as a ghost and skipped.
    assert set(kept.tolist()) == {0, 2}


def test_select_ghost_filter_disabled_by_default():
    theta = np.array([1.0, 2.0, 3.0])
    kept, _ = select_restart_indices(theta, n_keep=2, locked_local=[], locked_evals=np.array([2.0]))
    assert set(kept.tolist()) == {0, 1}


# --------------------------------------------------------------------------- #
# purge_restart
# --------------------------------------------------------------------------- #
def _assemble_band(alphas, betas, p):
    nb = alphas.shape[0]
    N = nb * p
    T = np.zeros((N, N), dtype=complex)
    for i in range(nb):
        T[i * p : (i + 1) * p, i * p : (i + 1) * p] = alphas[i]
        if i < nb - 1:
            T[(i + 1) * p : (i + 2) * p, i * p : (i + 1) * p] = betas[i]
            T[i * p : (i + 1) * p, (i + 1) * p : (i + 2) * p] = betas[i].conj().T
    return T


def test_purge_restart_preserves_kept_eigenvalues_and_orthonormal_C():
    m, p = 3, 2
    T = _hermitian_block_tridiagonal(m, p, seed=5)
    evals, Z = np.linalg.eigh(T)
    beta_last = np.random.default_rng(9).standard_normal((p, p)) + 0j

    # Keep the 4 algebraically smallest Ritz values (nb = 2 retained blocks).
    kept_idx, _ = select_restart_indices(evals, n_keep=2 * p, locked_local=[])

    C, beta_new, alphas_new, betas_new = purge_restart(evals, Z, beta_last, p, kept_idx)

    # The re-banded T^+ is a unitary similarity of diag(kept evals): same spectrum.
    Tplus = _assemble_band(alphas_new, betas_new, p)
    np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(Tplus)), np.sort(evals[kept_idx].real), atol=1e-10)
    # C = V_m -> V_new combination has orthonormal columns.
    np.testing.assert_allclose(C.conj().T @ C, np.eye(2 * p), atol=1e-10)
    assert beta_new.shape == (p, p)


# --------------------------------------------------------------------------- #
# locked_overlap_step
# --------------------------------------------------------------------------- #
def test_locked_overlap_step_recurrence_and_trigger():
    xi = np.array([0.0])
    xi_prev = np.array([0.0])
    locked_evals = np.array([0.0])
    alpha_j = np.array([[2.0 + 0j]])
    # xi_new = binv*(xi*|lam-mu| + xi_prev*bprev + rho) + eps
    xi_new, trigger, mask = locked_overlap_step(
        xi,
        xi_prev,
        locked_evals,
        alpha_j,
        beta_j_inv_norm=1.0,
        beta_jm1_norm=0.0,
        rho=1.0,
        omega_tol=0.5,
        bad_tol=0.5,
        eps=0.0,
    )
    np.testing.assert_allclose(xi_new, [1.0], atol=1e-14)
    assert trigger is True
    assert mask.tolist() == [True]


def test_locked_overlap_step_stays_below_threshold():
    xi_new, trigger, mask = locked_overlap_step(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([[2.0 + 0j]]),
        beta_j_inv_norm=1.0,
        beta_jm1_norm=0.0,
        rho=0.0,
        omega_tol=1e-8,
        bad_tol=1e-8,
        eps=EPS,
    )
    assert trigger is False
    assert mask.tolist() == [False]
    np.testing.assert_allclose(xi_new, [EPS], atol=1e-18)
