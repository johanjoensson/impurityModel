import numpy as np
import scipy.linalg as sp
from impurityModel.test.test_irlm import get_test_system, MockBasis
from impurityModel.ed.ManyBodyUtils import ManyBodyState, inner_multi, add_scaled_multi
from impurityModel.ed.trlm import thick_restarted_block_lanczos

def test_run():
    h_op, N, eigvals_exact, basis_states = get_test_system()
    basis = MockBasis(N)
    
    np.random.seed(42)
    n_blocks = 2
    psi0 = []
    for _ in range(n_blocks):
        state = ManyBodyState()
        for b in basis_states:
            state += b * (np.random.rand() + 1j * np.random.rand())
        psi0.append(state)
        
    M = inner_multi(psi0, psi0)
    L = sp.cholesky(M, lower=True)
    beta_inv = sp.inv(np.conj(L.T))
    psi0_orth = [ManyBodyState() for _ in range(n_blocks)]
    add_scaled_multi(psi0_orth, psi0, beta_inv)
    psi0 = psi0_orth
    
    eigvals, eigvecs = thick_restarted_block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=50,
        verbose=True
    )
    
    print("Exact:", eigvals_exact[:4])
    print("Lanczos:", eigvals)
    np.testing.assert_allclose(eigvals, eigvals_exact[:4], atol=1e-6)
    print("Test passed!")

if __name__ == '__main__':
    run_test()
