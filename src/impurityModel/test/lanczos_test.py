import pytest
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.finite import eigensystem_new as eigensystem, applyOp_new as applyOp, norm2
from impurityModel.ed.lanczos import block_lanczos, block_lanczos_sparse, eigsh, Reort
import numpy as np
from mpi4py import MPI


def test_lancos():
    Delta = 1
    Hop = {
        ((0, "c"), (0, "a")): 1 - Delta,
        ((1, "c"), (1, "a")): 1 - Delta,
        ((2, "c"), (2, "a")): 1 - Delta,
        ((3, "c"), (3, "a")): 1 + Delta,
        ((4, "c"), (4, "a")): 1 + Delta,
    }
    basis = Basis(
        impurity_orbitals={2: [[0, 1, 2, 3, 4]]},
        bath_states=({2: [[]]}, {2: [[]]}),
        initial_basis=[b"\xF0", b"\xE8", b"\xD8", b"\xB8", b"\x78"],
        verbose=True,
    )
    h_mat = basis.build_dense_matrix(Hop)
    gs_es, gs_psis = eigensystem(h_mat, 0)
    electron_addition_ops = {
        "t2g": [{((0, "c"),): 1}, {((1, "c"),): 1}, {((2, "c"),): 1}],
        "eg": [{((3, "c"),): 1}, {((4, "c"),): 1}],
    }
    electron_removal_ops = {
        "t2g": [{((0, "a"),): 1}, {((1, "a"),): 1}, {((2, "a"),): 1}],
        "eg": [{((3, "a"),): 1}, {((4, "a"),): 1}],
    }
    alphas = {"t2g": [], "eg": []}
    betas = {"t2g": [], "eg": []}

    def converged(alphas, betas, *args, **kwargs):
        return alphas.shape[0] * alphas.shape[1] >= len(basis)

    for irrep in electron_removal_ops:
        for op in electron_removal_ops[irrep]:
            gs_i = basis.build_state(gs_psis.T)
            psi = applyOp(5, op, gs_i[0])
            N = np.sqrt(norm2(psi))
            psi = {state: amp / N for state, amp in psi.items()}
            excited_basis = Basis(
                impurity_orbitals={2: [[0, 1, 2, 3, 4]]},
                bath_states=({2: [[]]}, {2: [[]]}),
                initial_basis=list(psi.keys()),
                verbose=True,
            )
            alpha, beta, _ = block_lanczos([psi], Hop, excited_basis, converged)
            alphas[irrep].append(alpha)
            betas[irrep].append(beta)
    print(f"{alphas=}\n{betas=}")


@pytest.mark.mpi
def test_lancos_mpi():
    Delta = 1
    Hop = {
        ((0, "c"), (0, "a")): 1 - Delta,
        ((1, "c"), (1, "a")): 1 - Delta,
        ((2, "c"), (2, "a")): 1 - Delta,
        ((3, "c"), (3, "a")): 1 + Delta,
        ((4, "c"), (4, "a")): 1 + Delta,
    }
    basis = Basis(
        impurity_orbitals={2: [[0, 1, 2, 3, 4]]},
        bath_states=({2: [[]]}, {2: [[]]}),
        initial_basis=[b"\xF0", b"\xE8", b"\xD8", b"\xB8", b"\x78"],
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    h_mat = basis.build_dense_matrix(Hop)
    gs_es, gs_psis = eigensystem(h_mat, 0)
    electron_addition_ops = {
        "t2g": [{((0, "c"),): 1}, {((1, "c"),): 1}, {((2, "c"),): 1}],
        "eg": [{((3, "c"),): 1}, {((4, "c"),): 1}],
    }
    electron_removal_ops = {
        "t2g": [{((0, "a"),): 1}, {((1, "a"),): 1}, {((2, "a"),): 1}],
        "eg": [{((3, "a"),): 1}, {((4, "a"),): 1}],
    }
    alphas = {"t2g": [], "eg": []}
    betas = {"t2g": [], "eg": []}

    def converged(alphas, betas, *args, **kwargs):
        return alphas.shape[0] * alphas.shape[1] >= len(basis)

    for irrep in electron_removal_ops:
        for op in electron_removal_ops[irrep]:
            gs_i = basis.build_state(gs_psis.T)
            psi = applyOp(5, op, gs_i[0])
            N2 = norm2(psi)
            MPI.COMM_WORLD.allreduce(N2, op=MPI.SUM)
            psi = {state: amp / np.sqrt(N2) for state, amp in psi.items()}
            excited_basis = Basis(
                impurity_orbitals={2: [[0, 1, 2, 3, 4]]},
                bath_states=({2: [[]]}, {2: [[]]}),
                initial_basis=list(psi.keys()),
                verbose=True,
                comm=MPI.COMM_WORLD,
            )
            alpha, beta, _ = block_lanczos([psi], Hop, excited_basis, converged)
            alphas[irrep].append(alpha)
            betas[irrep].append(beta)
    print(f"{alphas=}\n{betas=}")


def test_eigsh():
    eigvals = np.array(np.arange(6))
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"]
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    def converged(alphas, betas, *args, **kwargs):
        print(f"{alphas.shape=}")
        return alphas.shape[0] > 5

    psi0 = [{state: 1 / np.sqrt(len(states)) for state in states}]
    alphas, betas, _ = block_lanczos(psi0, hop, basis, converged, reort=Reort.FULL)
    ev = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


@pytest.mark.mpi
def test_eigsh_mpi():
    eigvals = np.array(np.arange(6))
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"]
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    def converged(alphas, betas, *args, **kwargs):
        print(f"{alphas.shape=}")
        return alphas.shape[0] > 5

    psi0 = [{state: 1 / np.sqrt(len(states)) for state in basis.local_basis}]
    alphas, betas, _ = block_lanczos(psi0, hop, basis, converged, reort=Reort.PARTIAL)
    ev = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


def test_block_eigsh():
    eigvals = np.array(np.arange(6))
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"]
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    def converged(alphas, betas, *args, **kwargs):
        print(f"{alphas.shape=}")
        return alphas.shape[0] > 2

    psi0 = [
        {state: 1 / np.sqrt(len(states) / 2) for state in states[:3]},
        {state: 1 / np.sqrt(len(states) / 2) for state in states[3:]},
    ]
    alphas, betas, _ = block_lanczos(psi0, hop, basis, converged)
    ev = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


@pytest.mark.mpi
def test_block_eigsh_mpi():
    eigvals = np.array(np.arange(6))
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"]
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    def converged(alphas, betas, *args, **kwargs):
        print(f"{alphas.shape=}")
        return alphas.shape[0] > 2

    if basis.comm.rank == 0:
        psi0 = [
            {state: 1 / np.sqrt(len(states) / 2) for state in states[:3]},
            {state: 1 / np.sqrt(len(states) / 2) for state in states[3:]},
        ]
    else:
        psi0 = [{}, {}]
    psi0 = list(basis.redistribute_psis(psi0))
    print(f"RANK {basis.comm.rank} psi0: {psi0}", flush=True)
    alphas, betas, _ = block_lanczos(psi0, hop, basis, converged)
    ev = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


def test_block_lanczos_sparse():
    eigvals = np.array(np.arange(6))
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"]
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    def converged(alphas, betas, *args, **kwargs):
        return alphas.shape[0] > 5

    psi0 = [{state: 1 / np.sqrt(len(states)) for state in states}]
    alphas, betas, _ = block_lanczos_sparse(psi0, hop, basis, converged)
    ev = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


@pytest.mark.mpi
def test_block_lanczos_sparse_mpi():
    eigvals = np.array(np.arange(6))
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"]
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    def converged(alphas, betas, *args, **kwargs):
        return alphas.shape[0] > 5

    psi0 = [{state: 1 / np.sqrt(len(states)) for state in basis.local_basis}]
    alphas, betas, _ = block_lanczos_sparse(psi0, hop, basis, converged)
    ev = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


def test_get_block_Lanczos_matrices_and_GS():
    from impurityModel.ed.lanczos import get_block_Lanczos_matrices, calculate_thermal_gs, get_Lanczos_vectors
    eigvals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"]
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )
    H_mat = basis.build_dense_matrix(hop)
    
    psi0 = np.zeros((6, 1), dtype=complex)
    psi0[:, 0] = 1 / np.sqrt(6)
    
    def converged(alphas, betas, *args, **kwargs):
        return alphas.shape[0] > 4

    alphas, betas, Q = get_block_Lanczos_matrices(psi0, H_mat[:, basis.local_indices], converged)
    ev = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])

    # test calculate_thermal_gs
    ev_gs, _ = calculate_thermal_gs(H_mat[:, basis.local_indices], block_size=1, e_max=0.1, comm=MPI.COMM_SELF)
    np.testing.assert_allclose(np.min(ev_gs), 0.5, atol=1e-5)


@pytest.mark.mpi
def test_get_block_Lanczos_matrices_and_GS_mpi():
    from impurityModel.ed.lanczos import get_block_Lanczos_matrices, calculate_thermal_gs, get_Lanczos_vectors
    eigvals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    states = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08", b"\x04"]
    hop = {((i, "c"), (i, "a")): val for i, val in enumerate(eigvals)}
    basis = Basis(
        impurity_orbitals={0: [[0, 1, 2, 3, 4, 5]]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    H_mat = basis.build_dense_matrix(hop)
    
    psi0 = np.zeros((len(basis.local_basis), 1), dtype=complex)
    psi0[:, 0] = 1 / np.sqrt(6)  # Distributed vector representation

    def converged(alphas, betas, *args, **kwargs):
        return alphas.shape[0] > 4

    alphas, betas, Q = get_block_Lanczos_matrices(psi0, H_mat[:, basis.local_indices], converged, comm=MPI.COMM_WORLD)
    ev = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])

    # test calculate_thermal_gs
    ev_gs, _ = calculate_thermal_gs(H_mat[:, basis.local_indices], block_size=1, e_max=0.1, comm=MPI.COMM_WORLD)
    np.testing.assert_allclose(np.min(ev_gs), 0.5, atol=1e-5)
