import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczos import (
    block_lanczos_cy,
    implicitly_restarted_block_lanczos_cy,
    thick_restart_block_lanczos_cy,
)
from impurityModel.ed.BlockLanczosArray import block_normalize
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, inner_multi


def create_diagonal_h_and_basis_mpi(n_states, comm):
    eigvals = np.arange(n_states, dtype=float)
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"][:n_states]
    hop = {((i, "c"), (i, "a")): float(val) for i, val in enumerate(eigvals)}

    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5][:n_states]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
        comm=comm,
    )
    h_op = ManyBodyOperator(hop)
    return h_op, basis, states, eigvals


@pytest.mark.mpi
def test_block_lanczos_cy_mpi_orthogonality_full():
    comm = MPI.COMM_WORLD
    h_op, basis, states, _ = create_diagonal_h_and_basis_mpi(6, comm)

    # Block size 1
    np.random.seed(42 + comm.Get_rank())
    psi0 = []
    for _i in range(1):
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn()
        psi0.append(st)

    psi0 = basis.redistribute_psis(psi0)
    psi0, _ = block_normalize(psi0, mpi=True, comm=comm)

    alphas, _betas, Q_basis, _W = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: len(a) >= 3,
        reort="full",
        max_iter=3,
        verbose=False,
        comm=comm,
    )

    assert len(alphas) == 3
    # Check orthogonality
    ov = inner_multi(Q_basis, Q_basis)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, ov, op=MPI.SUM)
    err = np.linalg.norm(ov - np.eye(len(Q_basis)))
    assert err < 1e-10


@pytest.mark.mpi
def test_block_lanczos_cy_mpi_choleskyqr2_near_degenerate():
    """The CholeskyQR2 second pass (and its M2 Allreduce) keeps the recurrence orthonormal
    and bounded under MPI even with reort=none and a near-degenerate spectrum — the regime
    that previously diverged. The block-QR runs on the replicated p x p Gram, so the result
    must match the serial numerics."""
    comm = MPI.COMM_WORLD
    h_op, basis, states, _ = create_diagonal_h_and_basis_mpi(6, comm)
    # Inject a near-degenerate pair into the (otherwise distinct) diagonal spectrum.
    eigvals = np.array([0.0, 1e-9, 1.0, 2.0, 3.0, 4.0])
    h_op = ManyBodyOperator({((i, "c"), (i, "a")): float(v) for i, v in enumerate(eigvals)})

    np.random.seed(7 + comm.Get_rank())
    psi0 = []
    for _ in range(2):  # block size 2 -> exercises the block QR / CholeskyQR2 path
        st = ManyBodyState()
        for j in range(6):
            st[basis.type.from_bytes(states[j])] = np.random.randn() + 1j * np.random.randn()
        psi0.append(st)
    psi0 = basis.redistribute_psis(psi0)
    psi0, _ = block_normalize(psi0, mpi=True, comm=comm)  # orthonormal start (as the GF path provides)

    _alphas, betas, Q_basis, _ = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: False,
        reort="none",
        max_iter=3,
        verbose=False,
        comm=comm,
    )

    max_beta = max(np.linalg.norm(np.asarray(b), 2) for b in betas)
    assert max_beta < 10 * float(np.max(np.abs(eigvals)) + 1.0)
    ov = inner_multi(Q_basis, Q_basis)
    comm.Allreduce(MPI.IN_PLACE, ov, op=MPI.SUM)
    assert np.linalg.norm(ov - np.eye(len(Q_basis))) < 1e-9


@pytest.mark.mpi
def test_trlm_cy_diagonal_mpi():
    comm = MPI.COMM_WORLD
    h_op, basis, states, eigvals = create_diagonal_h_and_basis_mpi(6, comm)

    st = ManyBodyState()
    for i in range(6):
        st[basis.type.from_bytes(states[i])] = 1.0 / np.sqrt(6)
    psi0 = basis.redistribute_psis([st])
    psi0, _ = block_normalize(psi0, mpi=True, comm=comm)

    eigs, _evecs = thick_restart_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=2,
        max_subspace_blocks=4,
        tol=1e-8,
        max_restarts=10,
        verbose=False,
        comm=comm,
    )

    assert len(eigs) == 2
    np.testing.assert_allclose(eigs, eigvals[:2], atol=1e-8)


def create_tight_binding_h_and_basis_mpi(comm):
    import itertools

    from impurityModel.ed.manybody_basis import Basis

    n_sites = 8
    n_particles = 4

    indices = list(range(n_sites))
    states = []
    for c in itertools.combinations(indices, n_particles):
        s = sum(1 << i for i in c)
        states.append(s.to_bytes(8, "little"))

    basis = Basis(
        impurity_orbitals={0: [indices]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
        comm=comm,
    )

    hop = {}
    for i in range(n_sites - 1):
        hop[((i, "c"), (i + 1, "a"))] = -1.0
        hop[((i + 1, "c"), (i, "a"))] = -1.0
    h_op = ManyBodyOperator(hop)

    return h_op, basis


@pytest.mark.mpi
def test_trlm_cy_tight_binding_mpi():
    import numpy as np
    from mpi4py import MPI

    from impurityModel.ed.BlockLanczos import thick_restart_block_lanczos_cy
    from impurityModel.ed.BlockLanczosArray import block_normalize

    comm = MPI.COMM_WORLD
    h_op, basis = create_tight_binding_h_and_basis_mpi(comm)

    np.random.seed(42 + comm.Get_rank())
    psi0 = []
    for _ in range(2):
        st = ManyBodyState()
        for b in basis.local_basis:
            st[b] = np.random.randn()
        psi0.append(st)

    psi0 = basis.redistribute_psis(psi0)
    psi0, _ = block_normalize(psi0, True, comm)

    eigs, _evecs = thick_restart_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=10,
        tol=1e-6,
        max_restarts=20,
        verbose=False,
        comm=comm,
    )

    assert len(eigs) == 4


@pytest.mark.mpi
def test_irlm_cy_diagonal_mpi():
    comm = MPI.COMM_WORLD
    h_op, basis, states, eigvals = create_diagonal_h_and_basis_mpi(6, comm)

    st = ManyBodyState()
    for i in range(6):
        st[basis.type.from_bytes(states[i])] = 1.0 / np.sqrt(6)
    psi0 = basis.redistribute_psis([st])
    psi0, _ = block_normalize(psi0, mpi=True, comm=comm)

    eigs, _evecs = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=2,
        max_subspace_blocks=4,
        tol=1e-8,
        max_restarts=10,
        verbose=False,
        comm=comm,
    )

    assert len(eigs) == 2
    np.testing.assert_allclose(eigs, eigvals[:2], atol=1e-8)


@pytest.mark.mpi
def test_irlm_cy_tight_binding_mpi():
    from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy

    comm = MPI.COMM_WORLD
    h_op, basis = create_tight_binding_h_and_basis_mpi(comm)

    np.random.seed(42 + comm.Get_rank())
    psi0 = []
    for _ in range(2):
        st = ManyBodyState()
        for b in basis.local_basis:
            st[b] = np.random.randn()
        psi0.append(st)

    psi0 = basis.redistribute_psis(psi0)
    psi0, _ = block_normalize(psi0, True, comm)

    eigs, _evecs = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=4,
        max_subspace_blocks=10,
        tol=1e-6,
        max_restarts=20,
        verbose=False,
        comm=comm,
    )

    assert len(eigs) == 4


@pytest.mark.mpi
def test_irlm_cy_selective_reort_orthogonality_mpi():
    comm = MPI.COMM_WORLD
    n_states = 12
    states = [(1 << i).to_bytes(8, "little") for i in range(n_states)]
    np.random.seed(42)
    H_mat = np.random.randn(n_states, n_states)
    H_mat = H_mat + H_mat.T
    hop = {}
    for i in range(n_states):
        for j in range(n_states):
            if abs(H_mat[i, j]) > 1e-10:
                hop[((i, "c"), (j, "a"))] = float(H_mat[i, j])
    basis = Basis(
        impurity_orbitals={0: [list(range(n_states))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
        comm=comm,
    )
    h_op = ManyBodyOperator(hop)

    psi0 = [ManyBodyState({b: np.random.randn() for b in basis.local_basis}, width=1)]
    psi0 = basis.redistribute_psis(psi0)
    psi0, _ = block_normalize(psi0, True, comm)

    _eigvals, eigvecs = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=6,
        max_subspace_blocks=10,
        tol=1e-8,
        max_restarts=15,
        verbose=False,
        reort="selective",
        comm=comm,
    )

    # Assert eigenvectors are orthogonal
    ov = inner_multi(eigvecs, eigvecs)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, ov, op=MPI.SUM)
    err = np.linalg.norm(ov - np.eye(len(eigvecs)))
    assert err < 1e-10


@pytest.mark.mpi
def test_trlm_cy_selective_reort_orthogonality_mpi():
    comm = MPI.COMM_WORLD
    n_states = 12
    states = [(1 << i).to_bytes(8, "little") for i in range(n_states)]
    np.random.seed(42)
    H_mat = np.random.randn(n_states, n_states)
    H_mat = H_mat + H_mat.T
    hop = {}
    for i in range(n_states):
        for j in range(n_states):
            if abs(H_mat[i, j]) > 1e-10:
                hop[((i, "c"), (j, "a"))] = float(H_mat[i, j])
    basis = Basis(
        impurity_orbitals={0: [list(range(n_states))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
        comm=comm,
    )
    h_op = ManyBodyOperator(hop)

    psi0 = [ManyBodyState({b: np.random.randn() for b in basis.local_basis}, width=1)]
    psi0 = basis.redistribute_psis(psi0)
    psi0, _ = block_normalize(psi0, True, comm)

    _eigvals, eigvecs = thick_restart_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=6,
        max_subspace_blocks=10,
        tol=1e-8,
        max_restarts=15,
        verbose=False,
        reort="selective",
        comm=comm,
    )

    # Assert eigenvectors are orthogonal
    ov = inner_multi(eigvecs, eigvecs)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, ov, op=MPI.SUM)
    err = np.linalg.norm(ov - np.eye(len(eigvecs)))
    assert err < 1e-10
