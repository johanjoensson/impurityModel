import itertools
import warnings
import numpy as np
import pytest
from mpi4py import MPI
import scipy.linalg as sp

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, inner_multi
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.BlockLanczos import (
    block_lanczos_cy,
    thick_restart_block_lanczos_cy,
    implicitly_restarted_block_lanczos_cy,
)
from impurityModel.ed.BlockLanczosArray import eigsh, Reort

warnings.filterwarnings("ignore")


def get_diagonal_system_mpi(size=6):
    states = [SlaterDeterminant.from_bytes((1 << (63 - i)).to_bytes(8, byteorder="big")) for i in range(size)]
    hop = {((i, "c"), (i, "a")): float(i) for i in range(size)}
    basis = Basis(
        impurity_orbitals={0: [list(range(size))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    return hop, basis, states


def get_tight_binding_system_mpi():
    n_sites = 6
    n_particles = 3
    combinations = list(itertools.combinations(range(n_sites), n_particles))
    states = [
        SlaterDeterminant.from_bytes(sum(1 << (63 - i) for i in c).to_bytes(8, byteorder="big")) for c in combinations
    ]

    op_dict = {}
    for i in range(n_sites - 1):
        op_dict[((i, "c"), (i + 1, "a"))] = -1.0
        op_dict[((i + 1, "c"), (i, "a"))] = -1.0

    basis = Basis(
        impurity_orbitals={0: [list(range(n_sites))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    # Exact eigenvalues
    H_dense = basis.build_dense_matrix(op_dict)
    eigvals_exact, _ = np.linalg.eigh(H_dense)
    return op_dict, basis, states, eigvals_exact


@pytest.mark.mpi
def test_mpi_block_lanczos_cy_orthogonality():
    hop, basis, states = get_diagonal_system_mpi(6)
    h_op = ManyBodyOperator(hop)

    block_size = 2
    psi0 = []
    for i in range(block_size):
        state = ManyBodyState()
        if states[i] in basis.local_basis:
            state += ManyBodyState({states[i]: 1.0})
        psi0.append(state)

    def converged(alphas, betas, **kw):
        return alphas.shape[0] >= 3

    alphas, betas, Q_basis, W = block_lanczos_cy(
        psi0, h_op, basis, converged, reort=Reort.FULL, max_iter=5, comm=MPI.COMM_WORLD
    )

    k = len(alphas) * block_size
    G = inner_multi(Q_basis[:k], Q_basis[:k])
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, G, op=MPI.SUM)

    np.testing.assert_allclose(np.abs(G - np.eye(k)), 0, atol=1e-10)


@pytest.mark.mpi
def test_mpi_trlm_cy_correctness():
    hop, basis, states, exact_ev = get_tight_binding_system_mpi()
    h_op = ManyBodyOperator(hop)

    np.random.seed(42)
    psi0 = []
    for _ in range(2):
        state = ManyBodyState()
        for s in basis.local_basis:
            state += ManyBodyState({s: np.random.rand() + 1j * np.random.rand()})
        psi0.append(state)

    M = inner_multi(psi0, psi0)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
    L = sp.cholesky(M, lower=True)
    beta_inv = sp.inv(np.conj(L.T))
    psi0_orth = [ManyBodyState() for _ in range(2)]
    from impurityModel.ed.ManyBodyUtils import add_scaled_multi

    add_scaled_multi(psi0_orth, psi0, beta_inv)

    eigvals, eigvecs = thick_restart_block_lanczos_cy(
        psi0=psi0_orth,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=10,
        tol=1e-8,
        max_restarts=50,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    np.testing.assert_allclose(eigvals, exact_ev[:4], atol=1e-6)


@pytest.mark.mpi
def test_mpi_irlm_cy_correctness():
    hop, basis, states, exact_ev = get_tight_binding_system_mpi()
    h_op = ManyBodyOperator(hop)

    np.random.seed(42)
    psi0 = []
    for _ in range(2):
        state = ManyBodyState()
        for s in basis.local_basis:
            state += ManyBodyState({s: np.random.rand() + 1j * np.random.rand()})
        psi0.append(state)

    M = inner_multi(psi0, psi0)
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
    L = sp.cholesky(M, lower=True)
    beta_inv = sp.inv(np.conj(L.T))
    psi0_orth = [ManyBodyState() for _ in range(2)]
    from impurityModel.ed.ManyBodyUtils import add_scaled_multi

    add_scaled_multi(psi0_orth, psi0, beta_inv)

    # We use max_subspace_blocks=10 to avoid restarting since block implicit QR is unstable
    eigvals, eigvecs = implicitly_restarted_block_lanczos_cy(
        psi0=psi0_orth,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=10,
        tol=1e-8,
        max_restarts=50,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    np.testing.assert_allclose(eigvals, exact_ev[:4], atol=1e-6)
