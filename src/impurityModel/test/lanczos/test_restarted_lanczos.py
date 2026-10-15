import warnings

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyState, inner_multi
from impurityModel.test.support.lanczos_fixtures import MockBasis, get_test_system

warnings.filterwarnings("ignore")


def test_trlm_invariant_subspace_breakdown():
    from impurityModel.ed.trlm import thick_restart_block_lanczos

    # Small 4x4 matrix
    H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    np.random.seed(42)
    psi0 = np.random.randn(4, 1).astype(complex)
    psi0, _ = np.linalg.qr(psi0)

    # Ask for 2 eigenvalues, but max_subspace_blocks = 4.
    # Since total dimension is 4, it will exhaust the Hilbert space and trigger breakdown.
    eigvals, _eigvecs = thick_restart_block_lanczos(
        psi0=psi0, h_op=H, basis=None, num_wanted=2, max_subspace_blocks=4, tol=1e-8, max_restarts=50, verbose=True
    )

    np.testing.assert_allclose(eigvals, [1.0, 2.0], atol=1e-10)


def test_trlm_reort_partial():
    h_op, N, eigvals_exact, basis_states = get_test_system()
    basis = MockBasis(N)

    # Starting block
    np.random.seed(42)
    n_blocks = 2
    psi0 = []
    for _ in range(n_blocks):
        state = ManyBodyState()
        for b in basis_states:
            state += b * (np.random.rand() + 1j * np.random.rand())
        psi0.append(state)

    import scipy.linalg as sp

    M = inner_multi(psi0, psi0)
    L = sp.cholesky(M, lower=True)
    beta_inv = sp.inv(np.conj(L.T))
    psi0 = ManyBodyState.from_states(psi0).combine_columns(beta_inv).to_states()

    from impurityModel.ed.trlm import thick_restart_block_lanczos

    eigvals, eigvecs = thick_restart_block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=50,
        verbose=True,
        reort=True,
    )

    np.testing.assert_allclose(eigvals, eigvals_exact[:4], atol=1e-6)

    # Assert Ritz vectors are orthogonal
    overlaps = inner_multi(eigvecs, eigvecs)
    np.testing.assert_allclose(np.abs(overlaps), np.eye(4), atol=1e-5)
