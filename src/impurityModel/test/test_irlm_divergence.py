import numpy as np
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import Reort
from impurityModel.test.test_restarted_lanczos import get_test_system, MockBasis
from impurityModel.ed.ManyBodyUtils import inner_multi, add_scaled_multi, ManyBodyState
import scipy.linalg as sp


def test_compare():
    np.random.seed(42)
    h_op, N, eigvals_exact, basis_states = get_test_system()
    basis = MockBasis(N)

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

    # Increase number of restarts dramatically to force breakdown/loss of orthogonality
    num_wanted = 4
    max_subspace_blocks = 6

    evals_cy, evecs_cy = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=num_wanted,
        max_subspace_blocks=max_subspace_blocks,
        tol=1e-8,
        max_restarts=50,
        verbose=True,
        reort=Reort.PARTIAL,
    )

    print("Cython vals:", evals_cy)
