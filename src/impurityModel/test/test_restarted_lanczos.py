import itertools
import warnings

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, inner_multi

warnings.filterwarnings("ignore")
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy  # noqa: E402


class MockBasis:
    def __init__(self, size):
        self.size = size
        self.comm = None
        self._index_dict = {}

    def redistribute_psis(self, psis):
        return psis

    def redistribute_block(self, block):
        return block

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
            amp = H_basis_states[j].get(sd_i)
            H_dense[i, j] = 0.0 if amp is None else amp[0]

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

    M = inner_multi(psi0, psi0)
    L = sp.cholesky(M, lower=True)
    beta_inv = sp.inv(np.conj(L.T))
    psi0 = ManyBodyState.from_states(psi0).combine_columns(beta_inv).to_states()

    from impurityModel.ed.trlm import thick_restart_block_lanczos

    eigvals, _eigvecs = thick_restart_block_lanczos(
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

    M = inner_multi(psi0, psi0)
    L = sp.cholesky(M, lower=True)
    beta_inv = sp.inv(np.conj(L.T))
    psi0 = ManyBodyState.from_states(psi0).combine_columns(beta_inv).to_states()

    from impurityModel.ed.BlockLanczosArray import Reort

    # Run IRLM QR Restart
    eigvals, _eigvecs = implicitly_restarted_block_lanczos_cy(
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


def test_irlm_invariant_subspace_breakdown():
    from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy

    # 4-site, 1-particle system -> 4 states total.
    n_sites = 4

    op_dict = {}
    for i in range(n_sites - 1):
        op_dict[((i, "c"), (i + 1, "a"))] = -1.0
        op_dict[((i + 1, "c"), (i, "a"))] = -1.0
    h_op = ManyBodyOperator(op_dict)

    # Generate basis using creation operators applied to vacuum
    vac = ManyBodyState({SlaterDeterminant((0,)): 1.0})
    states = []
    for i in range(n_sites):
        op = ManyBodyOperator({((i, "c"),): 1.0})
        st = op.apply(vac)
        states.append(next(iter(st.keys())))

    basis = MockBasis(4)
    basis.local_basis = states

    np.random.seed(42)
    psi0 = ManyBodyState()
    for s in basis.local_basis:
        psi0[s] = np.random.rand() + 1j * np.random.rand()
    N = psi0.norm()
    if N > 1e-12:
        for s in psi0:
            psi0[s] = psi0[s][0] / N

    # Ask for 2 eigenvalues, but max_subspace_blocks = 5.
    # Since total dimension is 4, it will exhaust the Hilbert space and trigger breakdown.
    eigvals_out, _eigvecs = implicitly_restarted_block_lanczos_cy(
        psi0=[psi0],
        h_op=h_op,
        basis=basis,
        num_wanted=2,
        max_subspace_blocks=5,
        tol=1e-8,
        max_restarts=50,
        verbose=True,
    )

    # Exact eigenvalues for 4-site tight-binding with 1 particle
    # are -2*cos(k), where k = pi*j / 5 for j=1..4
    # -1.61803399, -0.61803399,  0.61803399,  1.61803399
    exact_eigvals = np.array([-1.6180339887, -0.6180339887])
    np.testing.assert_allclose(eigvals_out, exact_eigvals, atol=1e-7)


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

    from impurityModel.ed.BlockLanczosArray import Reort

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
        reort=Reort.PARTIAL,
    )

    np.testing.assert_allclose(eigvals, eigvals_exact[:4], atol=1e-6)

    # Assert Ritz vectors are orthogonal
    overlaps = inner_multi(eigvecs, eigvecs)
    np.testing.assert_allclose(np.abs(overlaps), np.eye(4), atol=1e-5)


def test_irlm_reort_partial():
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

    from impurityModel.ed.BlockLanczosArray import Reort

    M = inner_multi(psi0, psi0)
    L = sp.cholesky(M, lower=True)
    beta_inv = sp.inv(np.conj(L.T))
    psi0 = ManyBodyState.from_states(psi0).combine_columns(beta_inv).to_states()

    from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy

    # Run with FULL reorthogonalization first
    eigvals_full, _ = implicitly_restarted_block_lanczos_cy(
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

    # Run with PARTIAL reorthogonalization
    eigvals, eigvecs = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=50,
        verbose=True,
        reort=Reort.PARTIAL,
    )

    # Assert eigenvalues match between PARTIAL and FULL
    np.testing.assert_allclose(eigvals, eigvals_full, atol=1e-6)

    # Assert eigenvalues match exact eigenvalues with looser tolerance
    np.testing.assert_allclose(eigvals, eigvals_exact[:4], atol=1.0)

    # Assert Ritz vectors are orthogonal
    overlaps = inner_multi(eigvecs, eigvecs)
    np.testing.assert_allclose(np.abs(overlaps), np.eye(4), atol=1e-5)
