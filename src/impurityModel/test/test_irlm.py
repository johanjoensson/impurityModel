import itertools
import warnings

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

warnings.filterwarnings("ignore")
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos


class MockBasis:
    def __init__(self, size):
        self.size = size
        self.comm = None
        self._index_dict = {}

    def redistribute_psis(self, psis):
        return psis

    def add_states(self, states):
        pass


def get_test_system():
    n_sites = 8
    n_particles = 4

    # Generate all basis states
    combinations = list(itertools.combinations(range(n_sites), n_particles))
    states = []
    for c in combinations:
        val = sum(1 << i for i in c)
        states.append(SlaterDeterminant.from_bytes(val.to_bytes(8, byteorder="little")))

    # 1D Tight-binding hopping
    op_dict = {}
    for i in range(n_sites - 1):
        op_dict[((i, "c"), (i + 1, "a"))] = -1.0
        op_dict[((i + 1, "c"), (i, "a"))] = -1.0

    h_op = ManyBodyOperator(op_dict)

    # Build dense matrix for exact diagonalization
    N = len(states)
    H_dense = np.zeros((N, N), dtype=complex)

    basis_states = [ManyBodyState({sd: 1.0}) for sd in states]
    H_basis_states = h_op.apply_multi(basis_states)

    for j in range(N):
        for i, sd_i in enumerate(states):
            H_dense[i, j] = H_basis_states[j].get(sd_i, 0.0)

    eigvals_exact, _ = np.linalg.eigh(H_dense)

    return h_op, N, eigvals_exact, basis_states


def test_irlm_thick_restart():
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

    from impurityModel.ed.ManyBodyUtils import add_scaled_multi, inner_multi

    M = inner_multi(psi0, psi0)
    L = sp.cholesky(M, lower=True)
    beta_inv = sp.inv(np.conj(L.T))
    psi0_orth = [ManyBodyState() for _ in range(n_blocks)]
    add_scaled_multi(psi0_orth, psi0, beta_inv)
    psi0 = psi0_orth

    from impurityModel.ed.trlm import thick_restarted_block_lanczos

    eigvals, eigvecs = thick_restarted_block_lanczos(
        psi0=psi0, h_op=h_op, basis=basis, num_wanted=4, max_subspace_blocks=6, tol=1e-8, max_restarts=50, verbose=False
    )

    np.testing.assert_allclose(eigvals, eigvals_exact[:4], atol=1e-6)


def test_irlm_qr_restart():
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

    from impurityModel.ed.ManyBodyUtils import add_scaled_multi, inner_multi

    M = inner_multi(psi0, psi0)
    L = sp.cholesky(M, lower=True)
    beta_inv = sp.inv(np.conj(L.T))
    psi0_orth = [ManyBodyState() for _ in range(n_blocks)]
    add_scaled_multi(psi0_orth, psi0, beta_inv)
    psi0 = psi0_orth

    from impurityModel.ed.lanczos import Reort

    # Run IRLM QR Restart
    eigvals, eigvecs = implicitly_restarted_block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=50,
        verbose=True,
        reort=Reort.FULL,
    )

    np.testing.assert_allclose(eigvals, eigvals_exact[:4], atol=1.0)
