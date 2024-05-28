import pytest
from manybody_basis import Basis
from finite import eigensystem_new as eigensystem, applyOp_new as applyOp, norm2
from lanczos import block_lanczos, eigsh, Reort
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
        impurity_orbitals={2: 5},
        valence_baths={2: 0},
        conduction_baths={2: 0},
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

    def converged(alphas, betas):
        return alphas.shape[0] * alphas.shape[1] >= len(basis)

    for irrep in electron_removal_ops:
        for op in electron_removal_ops[irrep]:
            gs_i = basis.build_state(gs_psis.T)
            psi = applyOp(5, op, gs_i[0])
            N = np.sqrt(norm2(psi))
            psi = {state: amp / N for state, amp in psi.items()}
            excited_basis = Basis(
                impurity_orbitals={2: 5},
                valence_baths={2: 0},
                conduction_baths={2: 0},
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
        impurity_orbitals={2: 5},
        valence_baths={2: 0},
        conduction_baths={2: 0},
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

    def converged(alphas, betas):
        return alphas.shape[0] * alphas.shape[1] >= len(basis)

    for irrep in electron_removal_ops:
        for op in electron_removal_ops[irrep]:
            gs_i = basis.build_state(gs_psis.T)
            psi = applyOp(5, op, gs_i[0])
            N2 = norm2(psi)
            MPI.COMM_WORLD.allreduce(N2, op=MPI.SUM)
            psi = {state: amp / np.sqrt(N2) for state, amp in psi.items()}
            excited_basis = Basis(
                impurity_orbitals={2: 5},
                valence_baths={2: 0},
                conduction_baths={2: 0},
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
        impurity_orbitals={0: 6},
        valence_baths={0: 0},
        conduction_baths={0: 0},
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    def converged(alphas, betas):
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
        impurity_orbitals={0: 6},
        valence_baths={0: 0},
        conduction_baths={0: 0},
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    def converged(alphas, betas):
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
        impurity_orbitals={0: 6},
        valence_baths={0: 0},
        conduction_baths={0: 0},
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    def converged(alphas, betas):
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
        impurity_orbitals={0: 6},
        valence_baths={0: 0},
        conduction_baths={0: 0},
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    def converged(alphas, betas):
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
    alphas, betas, _ = block_lanczos(psi0, hop, basis, converged)
    ev = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])
