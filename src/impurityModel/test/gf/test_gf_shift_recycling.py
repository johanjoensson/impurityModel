"""Direct unit test for _shifted_tridiag_solutions (gf_shift_recycling.py:246).

The higher-level KrylovShiftedResolvent is already extensively tested against dense
per-shift solves in the full physical space (test_rixs_tensor.py:
test_krylov_shifted_resolvent_matches_dense_solve/_long_recurrence, plus an MPI-distributed
variant), which can only pass if the reconstruction x = Q @ y is correct -- but that
conflates two things: the small-tridiagonal-system solve itself, and the Lanczos
recurrence/reconstruction around it. This isolates the former: hand-built block-Lanczos
coefficients (varying block widths, so the banded assembly's block bookkeeping is actually
exercised), independently assembled into a dense block-tridiagonal T, solved per shift with
plain np.linalg.solve, and checked against the function's own banded solve -- plus the
claimed exact-residual formula ``res = ||tail @ y[last_block]||``, which the higher-level
tests never check directly (they only check the final reconstructed x, not the internal
residual bookkeeping the resume/convergence loop actually relies on).
"""

import numpy as np

from impurityModel.ed.gf_shift_recycling import _shifted_tridiag_solutions


def _hermitian_2x2(diag_a, diag_b, off):
    return np.array([[diag_a, off], [np.conj(off), diag_b]], dtype=complex)


def _dense_reference(widths, a_trim, b_trim, b0, zs):
    """Independently assemble the dense block-tridiagonal T and solve per shift."""
    k = len(widths)
    starts = np.concatenate([[0], np.cumsum(widths)])
    n_t = int(starts[-1])
    T = np.zeros((n_t, n_t), dtype=complex)
    for i in range(k):
        s = slice(starts[i], starts[i + 1])
        T[s, s] = a_trim[i]
        if i + 1 < k:
            s_next = slice(starts[i + 1], starts[i + 2])
            T[s_next, s] = b_trim[i]
            T[s, s_next] = b_trim[i].conj().T
    rhs = np.zeros((n_t, b0.shape[1]), dtype=complex)
    rhs[: widths[0]] = b0
    eye = np.eye(n_t, dtype=complex)
    Y_ref = np.array([np.linalg.solve(z * eye - T, rhs) for z in zs])

    tail = b_trim[-1]
    last = slice(starts[k - 1], n_t)
    res_ref = np.array([np.linalg.norm(tail @ y[last]) for y in Y_ref])
    return Y_ref, res_ref


def test_shifted_tridiag_solutions_matches_dense_per_shift_solve():
    # Three blocks of widths [2, 1, 2]: non-uniform widths exercise the banded assembly's
    # per-block bookkeeping (a uniform width would leave off-by-one row/column slicing bugs
    # undetected).
    widths = [2, 1, 2]
    P = 2  # padded storage width >= max(widths)

    alphas = np.zeros((3, P, P), dtype=complex)
    alphas[0] = _hermitian_2x2(1.0, -0.5, 0.3 + 0.1j)
    alphas[1][0, 0] = 0.7  # only the top-left 1x1 is meaningful (w_1 = 1)
    alphas[2] = _hermitian_2x2(-0.2, 0.9, 0.15 - 0.05j)

    betas = np.zeros((3, P, P), dtype=complex)
    betas[0][:1, :2] = [[0.4, -0.2j]]  # block0 (w=2) -> block1 (w=1)
    betas[1][:2, :1] = [[0.25], [0.1j]]  # block1 (w=1) -> block2 (w=2)
    betas[2] = [[0.1, 0.05], [0.02, 0.3]]  # tail: residual coupling beyond the subspace

    rng = np.random.default_rng(0)
    b0 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    zs = np.array([0.3 + 0.2j, -0.5 + 0.15j, 1.1 + 0.3j])

    Y, res = _shifted_tridiag_solutions(alphas, betas, widths, b0, zs)

    a_trim = [alphas[0], alphas[1][:1, :1], alphas[2]]
    b_trim = [betas[0][:1, :2], betas[1][:2, :1], betas[2]]
    Y_ref, res_ref = _dense_reference(widths, a_trim, b_trim, b0, zs)

    assert Y.shape == (len(zs), sum(widths), b0.shape[1])
    np.testing.assert_allclose(Y, Y_ref, atol=1e-10)
    np.testing.assert_allclose(res, res_ref, atol=1e-10)


def test_shifted_tridiag_solutions_would_catch_a_wrong_coupling_bug():
    """Sanity check that the comparison above is actually sensitive to the off-diagonal
    coupling blocks -- guards against a vacuously-passing test (e.g. if betas happened to
    be small enough that T was effectively diagonal). Perturbing betas[1] must change Y."""
    widths = [2, 1, 2]
    P = 2
    alphas = np.zeros((3, P, P), dtype=complex)
    alphas[0] = _hermitian_2x2(1.0, -0.5, 0.3 + 0.1j)
    alphas[1][0, 0] = 0.7
    alphas[2] = _hermitian_2x2(-0.2, 0.9, 0.15 - 0.05j)

    betas = np.zeros((3, P, P), dtype=complex)
    betas[0][:1, :2] = [[0.4, -0.2j]]
    betas[1][:2, :1] = [[0.25], [0.1j]]
    betas[2] = [[0.1, 0.05], [0.02, 0.3]]

    rng = np.random.default_rng(0)
    b0 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    zs = np.array([0.3 + 0.2j])

    Y, _ = _shifted_tridiag_solutions(alphas, betas, widths, b0, zs)

    betas_perturbed = betas.copy()
    betas_perturbed[1][:2, :1] = [[2.5], [1.0j]]  # a genuinely different coupling
    Y_perturbed, _ = _shifted_tridiag_solutions(alphas, betas_perturbed, widths, b0, zs)

    assert not np.allclose(Y, Y_perturbed, atol=1e-6)
