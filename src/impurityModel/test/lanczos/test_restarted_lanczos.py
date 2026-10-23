import warnings

import numpy as np

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
