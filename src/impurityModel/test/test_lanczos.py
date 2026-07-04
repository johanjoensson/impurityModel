import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.eigensolvers import eigensystem
from impurityModel.ed.BlockLanczosArray import Reort, eigsh
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, applyOp
from impurityModel.ed.basis_transcription import build_dense_matrix, build_state


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
        initial_basis=[b"\xf0", b"\xe8", b"\xd8", b"\xb8", b"\x78"],
        verbose=True,
    )
    h_mat = build_dense_matrix(basis, Hop)
    gs_es, gs_psis = eigensystem(h_mat, 0)
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
            op_mbo = ManyBodyOperator(op)
            gs_i = build_state(basis, gs_psis.T)
            psi = applyOp(op_mbo, gs_i[0])
            N = psi.norm()
            if N > 1e-12:
                psi = ManyBodyState({state: amp / N for state, amp in psi.items()})
                excited_basis = Basis(
                    impurity_orbitals={2: [[0, 1, 2, 3, 4]]},
                    bath_states=({2: [[]]}, {2: [[]]}),
                    initial_basis=list(psi.keys()),
                    verbose=True,
                )
                alpha, beta, _, _ = block_lanczos_cy([psi], ManyBodyOperator(Hop), excited_basis, converged, max_iter=1)
                alphas[irrep].append(alpha)
                betas[irrep].append(beta)

                # Assert that the eigenvalues of the Lanczos tridiagonal matrix
                # match the direct eigenvalues of Hop on the excited basis
                ev, _ = eigsh(alpha, beta, eigvals_only=True, de=10)
                h_excited = build_dense_matrix(excited_basis, Hop)
                es_direct, _ = eigensystem(h_excited, 0)
                np.testing.assert_allclose(ev, es_direct, atol=1e-12)
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
        initial_basis=[b"\xf0", b"\xe8", b"\xd8", b"\xb8", b"\x78"],
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    h_mat = build_dense_matrix(basis, Hop)
    gs_es, gs_psis = eigensystem(h_mat, 0)
    electron_removal_ops = {
        "t2g": [{((0, "a"),): 1}, {((1, "a"),): 1}, {((2, "a"),): 1}],
        "eg": [{((3, "a"),): 1}, {((4, "a"),): 1}],
    }
    alphas = {"t2g": [], "eg": []}
    betas = {"t2g": [], "eg": []}

    def converged(alphas, betas, *args, **kwargs):
        # We only want to test the first iteration, because block_lanczos_cy
        # will otherwise expand the subspace beyond the initial basis,
        # changing the exact eigenvalues from the 1x1 projection.
        return alphas.shape[0] >= 1

    for irrep in electron_removal_ops:
        for op in electron_removal_ops[irrep]:
            op_mbo = ManyBodyOperator(op)
            print(
                f"Rank {MPI.COMM_WORLD.rank if MPI.COMM_WORLD else 'no MPI'} before build_state for op {op}", flush=True
            )
            gs_i = build_state(basis, gs_psis.T)
            print(f"Rank {MPI.COMM_WORLD.rank if MPI.COMM_WORLD else 'no MPI'} before applyOp for op {op}", flush=True)
            psi = applyOp(op_mbo, gs_i[0])
            print(f"Rank {MPI.COMM_WORLD.rank if MPI.COMM_WORLD else 'no MPI'} before norm2 for op {op}", flush=True)
            N2 = psi.norm2()
            print(f"Rank {MPI.COMM_WORLD.rank if MPI.COMM_WORLD else 'no MPI'} at allreduce for N2", flush=True)
            if MPI.COMM_WORLD is not None:
                N2 = MPI.COMM_WORLD.allreduce(N2, op=MPI.SUM)
            print(f"Rank {MPI.COMM_WORLD.rank if MPI.COMM_WORLD else 'no MPI'} passed allreduce for N2", flush=True)
            N = np.sqrt(N2)
            if N > 1e-12:
                psi = ManyBodyState({state: amp / N for state, amp in psi.items()})
                excited_basis = Basis(
                    impurity_orbitals={2: [[0, 1, 2, 3, 4]]},
                    bath_states=({2: [[]]}, {2: [[]]}),
                    initial_basis=list(psi.keys()),
                    verbose=True,
                    comm=MPI.COMM_WORLD,
                )
                alpha, beta, _, _ = block_lanczos_cy([psi], ManyBodyOperator(Hop), excited_basis, converged, max_iter=1)
                alphas[irrep].append(alpha)
                betas[irrep].append(beta)

                # Assert that the eigenvalues of the Lanczos tridiagonal matrix
                # match the direct eigenvalues of Hop on the excited basis
                h_excited = build_dense_matrix(excited_basis, Hop)
                es_direct, _ = eigensystem(h_excited, 0)
                if MPI.COMM_WORLD.rank == 0:
                    try:
                        ev, _ = eigsh(alpha, beta, eigvals_only=True, de=10)
                        np.testing.assert_allclose(ev, es_direct, atol=1e-12)
                    except Exception as e:
                        print(f"Rank 0 Exception: {e}", flush=True)
                        MPI.COMM_WORLD.Abort(1)
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

    psi0 = [ManyBodyState({state: 1 / np.sqrt(len(basis.local_basis)) for state in basis.local_basis})]
    alphas, betas, _, _ = block_lanczos_cy(psi0, ManyBodyOperator(hop), basis, converged, reort=Reort.FULL)
    ev, _ = eigsh(alphas, betas, eigvals_only=True, de=10)
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

    psi0 = [ManyBodyState({state: 1 / np.sqrt(len(states)) for state in basis.local_basis})]
    alphas, betas, _, _ = block_lanczos_cy(psi0, ManyBodyOperator(hop), basis, converged, reort=Reort.PARTIAL)
    ev, _ = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


@pytest.mark.parametrize("reort_mode", [Reort.NONE, Reort.FULL, Reort.PERIODIC, Reort.PARTIAL, Reort.SELECTIVE])
def test_block_eigsh(reort_mode):
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

    lb = list(basis.local_basis)
    psi0 = [
        ManyBodyState({state: 1 / np.sqrt(len(lb) / 2) for state in lb[:3]}),
        ManyBodyState({state: 1 / np.sqrt(len(lb) / 2) for state in lb[3:]}),
    ]
    alphas, betas, _, _ = block_lanczos_cy(psi0, ManyBodyOperator(hop), basis, converged, reort=reort_mode)
    ev, _ = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


@pytest.mark.mpi
@pytest.mark.parametrize("reort_mode", [Reort.NONE, Reort.FULL, Reort.PERIODIC, Reort.PARTIAL, Reort.SELECTIVE])
def test_block_eigsh_mpi(reort_mode):
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
        # Construct globally valid states on rank 0
        from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

        global_states = [SlaterDeterminant.from_bytes(s) for s in states]
        psi0 = [
            ManyBodyState({s: 1 / np.sqrt(3) for s in global_states[:3]}),
            ManyBodyState({s: 1 / np.sqrt(3) for s in global_states[3:]}),
        ]
    else:
        psi0 = [ManyBodyState(), ManyBodyState()]
    psi0 = list(basis.redistribute_psis(psi0))
    print(f"RANK {basis.comm.rank} psi0: {psi0}", flush=True)
    alphas, betas, _, _ = block_lanczos_cy(psi0, ManyBodyOperator(hop), basis, converged, reort=reort_mode)
    ev, _ = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


@pytest.mark.parametrize("reort_mode", [Reort.NONE, Reort.FULL, Reort.PERIODIC, Reort.PARTIAL, Reort.SELECTIVE])
def test_block_lanczos_cy(reort_mode):
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

    psi0 = [ManyBodyState({state: 1 / np.sqrt(len(basis.local_basis)) for state in basis.local_basis})]
    alphas, betas, _, _ = block_lanczos_cy(psi0, ManyBodyOperator(hop), basis, converged, reort=reort_mode)
    ev, _ = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


@pytest.mark.mpi
@pytest.mark.parametrize("reort_mode", [Reort.NONE, Reort.FULL, Reort.PERIODIC, Reort.PARTIAL, Reort.SELECTIVE])
def test_block_lanczos_cy_mpi(reort_mode):
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

    psi0 = [ManyBodyState({state: 1 / np.sqrt(len(states)) for state in basis.local_basis})]
    alphas, betas, _, _ = block_lanczos_cy(psi0, ManyBodyOperator(hop), basis, converged, reort=reort_mode)
    ev, _ = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])


@pytest.mark.parametrize("reort_mode", [Reort.NONE, Reort.FULL, Reort.PERIODIC, Reort.PARTIAL, Reort.SELECTIVE])
def test_get_block_Lanczos_matrices_and_GS(reort_mode):
    from impurityModel.ed.BlockLanczosArray import calculate_thermal_gs, block_lanczos_array

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
    H_mat = build_dense_matrix(basis, hop)

    psi0 = np.zeros((6, 1), dtype=complex)
    psi0[:, 0] = 1 / np.sqrt(6)

    def converged(alphas, betas, *args, **kwargs):
        return alphas.shape[0] > 5

    alphas, betas, Q = block_lanczos_array(psi0, H_mat[:, basis.local_indices], converged, reort=reort_mode)[:3]
    ev, _ = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])

    # test calculate_thermal_gs
    ev_gs, _ = calculate_thermal_gs(H_mat[:, basis.local_indices], block_size=1, e_max=0.1, comm=MPI.COMM_SELF)
    np.testing.assert_allclose(np.min(ev_gs), 0.5, atol=1e-5)


@pytest.mark.mpi
@pytest.mark.parametrize("reort_mode", [Reort.NONE, Reort.FULL, Reort.PERIODIC, Reort.PARTIAL, Reort.SELECTIVE])
def test_get_block_Lanczos_matrices_and_GS_mpi(reort_mode):
    from impurityModel.ed.BlockLanczosArray import calculate_thermal_gs, block_lanczos_array

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
    H_mat = build_dense_matrix(basis, hop)

    psi0 = np.zeros((len(basis.local_basis), 1), dtype=complex)
    psi0[:, 0] = 1 / np.sqrt(6)  # Distributed vector representation

    def converged(alphas, betas, *args, **kwargs):
        return alphas.shape[0] > 5

    alphas, betas, Q = block_lanczos_array(
        psi0, H_mat[:, basis.local_indices], converged, comm=MPI.COMM_WORLD, reort=reort_mode
    )[:3]
    ev, _ = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])

    # test calculate_thermal_gs
    ev_gs, _ = calculate_thermal_gs(H_mat[:, basis.local_indices], block_size=1, e_max=0.1, comm=MPI.COMM_WORLD)
    np.testing.assert_allclose(np.min(ev_gs), 0.5, atol=1e-5)


@pytest.mark.parametrize("reort_mode", [Reort.NONE, Reort.FULL, Reort.PERIODIC, Reort.PARTIAL, Reort.SELECTIVE])
def test_get_block_Lanczos_matrices_dense(reort_mode):
    from impurityModel.ed.BlockLanczosArray import block_lanczos_array

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
    H_mat = build_dense_matrix(basis, hop)

    psi0 = np.zeros((6, 1), dtype=complex)
    psi0[:, 0] = 1 / np.sqrt(6)

    def converged(alphas, betas, *args, **kwargs):
        return alphas.shape[0] > 5

    alphas, betas, Q = block_lanczos_array(psi0, H_mat, converged, reort=reort_mode)[:3]
    ev, _ = eigsh(alphas, betas, eigvals_only=True, de=10)
    assert np.allclose(ev, eigvals[: len(ev)])
