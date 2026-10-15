import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczos import (
    block_lanczos,
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

    psi0 = basis.redistribute_psis(*psi0)
    psi0, _ = block_normalize(psi0, mpi=True, comm=comm)

    alphas, _betas, Q_basis = block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged=lambda a, b, **kw: len(a) >= 3,
        reort=True,
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
    psi0 = basis.redistribute_psis(*psi0)
    psi0, _ = block_normalize(psi0, mpi=True, comm=comm)  # orthonormal start (as the GF path provides)

    _alphas, betas, Q_basis = block_lanczos(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged=lambda a, b, **kw: False,
        reort=False,
        max_iter=3,
        verbose=False,
        comm=comm,
    )

    max_beta = max(np.linalg.norm(np.asarray(b), 2) for b in betas)
    assert max_beta < 10 * float(np.max(np.abs(eigvals)) + 1.0)
    ov = inner_multi(Q_basis, Q_basis)
    comm.Allreduce(MPI.IN_PLACE, ov, op=MPI.SUM)
    assert np.linalg.norm(ov - np.eye(len(Q_basis))) < 1e-9


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
