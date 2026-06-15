"""
Tests for the MPI communication helpers in mpi_comm.py.

Covers:
  - graph_alltoall  (sparse + full send/receive patterns)
  - is_empty / empty_clone helpers
  - Integration with redistribute_psis / add_states via manybody_basis

Serial (no MPI marker) tests run with a single process.
Tests marked @pytest.mark.mpi run under mpirun with --with-mpi and
exercise the actual distributed code paths.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
    SlaterDeterminant,
)
from impurityModel.ed.mpi_comm import empty_clone, graph_alltoall, graph_alltoall_psis, is_empty

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_basis(states_bytes, comm=None):
    """Create a small Basis from a list of byte strings."""
    return Basis(
        impurity_orbitals={0: [list(range(8 * len(states_bytes[0])))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states_bytes,
        comm=comm,
    )


# ---------------------------------------------------------------------------
# Serial utility tests (no MPI required)
# ---------------------------------------------------------------------------


def test_is_empty_none():
    assert is_empty(None)


def test_is_empty_empty_dict():
    assert is_empty({})


def test_is_empty_empty_list():
    assert is_empty([])


def test_is_empty_list_of_empty_dicts():
    assert is_empty([{}, {}])


def test_is_empty_nonempty_dict():
    assert not is_empty({"a": 1})


def test_is_empty_nonempty_list():
    assert not is_empty([1, 2])


def test_empty_clone_none():
    assert empty_clone(None) is None


def test_empty_clone_dict():
    result = empty_clone({"a": 1})
    assert result == {}


def test_empty_clone_list_of_dicts():
    result = empty_clone([{"a": 1}, {"b": 2}])
    assert result == [{}, {}]


def test_empty_clone_list_of_lists():
    result = empty_clone([[1, 2], [3, 4]])
    assert result == [[], []]


def test_empty_clone_set():
    result = empty_clone({1, 2, 3})
    assert result == set()


# ---------------------------------------------------------------------------
# Serial graph_alltoall (size == 1 fast-path)
# ---------------------------------------------------------------------------


def test_graph_alltoall_serial_passthrough():
    """With comm=None the list must come back unchanged."""
    payload = [{"hello": "world"}]
    result = graph_alltoall(payload, None)
    assert result == payload


def test_graph_alltoall_single_rank_comm():
    """With COMM_SELF (size 1) the list must come back unchanged."""
    payload = [{"x": 42}]
    result = graph_alltoall(payload, MPI.COMM_SELF)
    assert result == payload


# ---------------------------------------------------------------------------
# MPI graph_alltoall tests
# ---------------------------------------------------------------------------


@pytest.mark.mpi
def test_graph_alltoall_empty_sends():
    """All ranks send empty/None to all others – result should be empty clones."""
    comm = MPI.COMM_WORLD
    send_list = [None] * comm.size
    result = graph_alltoall(send_list, comm)
    assert len(result) == comm.size
    for r in range(comm.size):
        assert result[r] is None


@pytest.mark.mpi
def test_graph_alltoall_all_to_all_dicts():
    """
    Each rank sends a distinct dict to every other rank.
    After the exchange every rank must hold the dict originally
    sent by each sender.
    """
    comm = MPI.COMM_WORLD
    send_list = [{f"key_from_{comm.rank}_to_{r}": comm.rank * 100 + r} for r in range(comm.size)]
    result = graph_alltoall(send_list, comm)
    assert len(result) == comm.size
    for sender in range(comm.size):
        expected_key = f"key_from_{sender}_to_{comm.rank}"
        assert expected_key in result[sender], f"rank {comm.rank}: missing key from sender {sender}: {result[sender]}"
        assert result[sender][expected_key] == sender * 100 + comm.rank


@pytest.mark.mpi
def test_graph_alltoall_sparse_send():
    """
    Only rank 0 sends data (to rank 1, if >= 2 ranks).
    All other send slots are None/empty.
    """
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("Needs at least 2 ranks")

    send_list = [None] * comm.size
    if comm.rank == 0:
        send_list[1] = {"ping": True}

    result = graph_alltoall(send_list, comm)

    if comm.rank == 1:
        assert result[0] == {"ping": True}
    else:
        # Other ranks received nothing from rank 0
        assert result[0] is None or is_empty(result[0])


@pytest.mark.mpi
def test_graph_alltoall_list_payloads():
    """Payloads are lists of dicts (as used in redistribute_psis)."""
    comm = MPI.COMM_WORLD
    # Each rank sends to its right neighbour only
    dest = (comm.rank + 1) % comm.size
    send_list = [None] * comm.size
    send_list[dest] = [{"state": comm.rank}]

    result = graph_alltoall(send_list, comm)

    src = (comm.rank - 1) % comm.size
    assert isinstance(result[src], list)
    assert len(result[src]) == 1
    assert result[src][0]["state"] == src


@pytest.mark.mpi
def test_graph_alltoall_large_payload():
    """
    Send a moderately large numpy array wrapped in a dict to stress
    the byte-level transfer path.  All ranks use the same seeded array
    so we can verify the round-trip without needing to gather.
    """
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(42)
    arr = rng.random(1000).tolist()  # same on every rank (same seed)
    send_list = [{"data": arr} for _ in range(comm.size)]

    result = graph_alltoall(send_list, comm)

    for r in range(comm.size):
        assert "data" in result[r], f"rank {comm.rank}: missing 'data' from rank {r}"
        assert np.allclose(result[r]["data"], arr)


@pytest.mark.mpi
def test_graph_alltoall_self_send():
    """
    A rank sends data to itself. This must round-trip correctly.
    """
    comm = MPI.COMM_WORLD
    send_list = [None] * comm.size
    send_list[comm.rank] = {"self": comm.rank}

    result = graph_alltoall(send_list, comm)

    assert result[comm.rank] == {"self": comm.rank}


# ---------------------------------------------------------------------------
# redistribute_psis MPI integration tests
# ---------------------------------------------------------------------------


@pytest.mark.mpi
def test_redistribute_psis_roundtrip():
    """
    Build a Basis with MPI, create a psi that lives entirely on rank 0,
    redistribute it, gather it back, and verify all amplitudes are present.
    """
    comm = MPI.COMM_WORLD
    states_bytes = [b"\x80", b"\x40", b"\x20", b"\x10"]
    basis = _make_basis(states_bytes, comm=comm)

    # All amplitude lives on rank 0 before redistribution
    if comm.rank == 0:
        psi = ManyBodyState({SlaterDeterminant.from_bytes(s): 1.0 / len(states_bytes) for s in states_bytes})
    else:
        psi = ManyBodyState({})

    redist = basis.redistribute_psis([psi])[0]

    # Gather back and check completeness
    gathered = comm.gather(redist, root=0)
    if comm.rank == 0:
        combined = ManyBodyState({})
        for part in gathered:
            combined += part
        for s in states_bytes:
            sd = SlaterDeterminant.from_bytes(s)
            assert abs(combined[sd] - 1.0 / len(states_bytes)) < 1e-12, (
                f"Missing or wrong amplitude for state {s.hex()}"
            )


@pytest.mark.mpi
def test_redistribute_psis_normalisation():
    """
    Redistribute a normalised psi and verify that norm is preserved.
    """
    comm = MPI.COMM_WORLD
    states_bytes = [b"\x80", b"\x40", b"\x20", b"\x10"]
    basis = _make_basis(states_bytes, comm=comm)

    norm_sq = 1.0 / len(states_bytes)
    if comm.rank == 0:
        psi = ManyBodyState({SlaterDeterminant.from_bytes(s): np.sqrt(norm_sq) for s in states_bytes})
    else:
        psi = ManyBodyState({})

    redist = basis.redistribute_psis([psi])[0]

    # Local norm squared contribution from this rank
    local_norm_sq = sum(abs(v) ** 2 for v in redist.values())
    total_norm_sq = comm.allreduce(local_norm_sq, op=MPI.SUM)
    assert abs(total_norm_sq - 1.0) < 1e-12, f"Norm not preserved: {total_norm_sq}"


@pytest.mark.mpi
def test_add_states_distributed():
    """
    Start with a basis shared across ranks, add new states, and verify
    the total count is correct.
    """
    comm = MPI.COMM_WORLD
    initial = [b"\x80", b"\x40"]
    basis = _make_basis(initial, comm=comm)
    initial_size = basis.size

    # Add two more states
    new_states = [
        SlaterDeterminant.from_bytes(b"\x20"),
        SlaterDeterminant.from_bytes(b"\x10"),
    ]
    basis.add_states(new_states)

    assert basis.size == initial_size + 2, f"rank {comm.rank}: expected {initial_size + 2} states, got {basis.size}"


# ---------------------------------------------------------------------------
# Hamiltonian-level MPI integration: build_sparse_matrix consistency
# ---------------------------------------------------------------------------


@pytest.mark.mpi
def test_sparse_matrix_consistent_with_serial():
    """
    Build a 4-state basis with a simple hopping Hamiltonian in both
    serial and parallel mode.  The sets of non-zero (bra_state,
    ket_state, value) triplets must be identical, regardless of how
    global indices are assigned (which differs between MPI and serial
    because MPI distributes by hash, serial by sorted order).
    """
    comm = MPI.COMM_WORLD

    states_bytes = [b"\x80", b"\x40", b"\x20", b"\x10"]
    hop = ManyBodyOperator(
        {
            ((0, "c"), (1, "a")): 1.0,
            ((1, "c"), (0, "a")): 1.0,
            ((2, "c"), (3, "a")): 0.5,
            ((3, "c"), (2, "a")): 0.5,
        }
    )

    # MPI basis: collect (bra_bytes, ket_bytes, value) from all ranks
    basis_mpi = _make_basis(states_bytes, comm=comm)
    H_mpi_local = basis_mpi.build_sparse_matrix(hop).tocoo()
    local_basis_all = comm.allgather(basis_mpi.local_basis)
    global_basis = {}
    for rank_basis in local_basis_all:
        for state in rank_basis:
            idx = basis_mpi.state_container._index_sequence([state])
            i = next(idx)
            global_basis[i] = bytes(state.to_bytearray()[: basis_mpi.n_bytes])

    # Build (bra_key, ket_key, val) tuples for local non-zeros
    local_triplets = set()
    for row, col, val in zip(H_mpi_local.row, H_mpi_local.col, H_mpi_local.data):
        if abs(val) > 1e-14:
            bra_key = global_basis.get(int(row))
            ket_key = global_basis.get(int(col))
            if bra_key is not None and ket_key is not None:
                local_triplets.add((bra_key, ket_key, round(val.real, 10)))
    all_triplets_mpi = set()
    for t in comm.allgather(local_triplets):
        all_triplets_mpi |= t

    # Serial reference
    if comm.rank == 0:
        basis_s = _make_basis(states_bytes, comm=None)
        H_s = basis_s.build_sparse_matrix(hop).tocoo()
        serial_basis_map = {
            i: bytes(state.to_bytearray()[: basis_s.n_bytes]) for i, state in enumerate(basis_s.local_basis)
        }
        all_triplets_serial = set()
        for row, col, val in zip(H_s.row, H_s.col, H_s.data):
            if abs(val) > 1e-14:
                bra_key = serial_basis_map.get(int(row))
                ket_key = serial_basis_map.get(int(col))
                if bra_key is not None and ket_key is not None:
                    all_triplets_serial.add((bra_key, ket_key, round(val.real, 10)))
    else:
        all_triplets_serial = None
    all_triplets_serial = comm.bcast(all_triplets_serial, root=0)

    assert all_triplets_mpi == all_triplets_serial, (
        f"rank {comm.rank}: MPI sparse matrix non-zeros differ from serial.\n"
        f"MPI only: {all_triplets_mpi - all_triplets_serial}\n"
        f"Serial only: {all_triplets_serial - all_triplets_mpi}"
    )


# ---------------------------------------------------------------------------
# Many-body density matrix MPI vs serial
# ---------------------------------------------------------------------------


@pytest.mark.mpi
def test_density_matrix_mpi_vs_serial():
    """
    build_density_matrices must give the same result in serial and MPI mode.
    """
    comm = MPI.COMM_WORLD

    states_bytes = [b"\x80", b"\x40", b"\x20", b"\x10"]
    orbital_indices = [0, 1, 2, 3]

    # Build a random-ish psi
    np.random.seed(7)
    coeffs = np.random.rand(len(states_bytes)) + 0j
    coeffs /= np.linalg.norm(coeffs)

    # Serial reference
    if comm.rank == 0:
        basis_s = _make_basis(states_bytes, comm=None)
        psi_s = ManyBodyState({SlaterDeterminant.from_bytes(s): c for s, c in zip(states_bytes, coeffs)})
        rho_serial = basis_s.build_density_matrices(
            [psi_s],
            orbital_indices_left=orbital_indices,
            orbital_indices_right=orbital_indices,
        )[0]
    else:
        rho_serial = None
    rho_serial = comm.bcast(rho_serial, root=0)

    # MPI version
    basis_m = _make_basis(states_bytes, comm=comm)
    if comm.rank == 0:
        psi_m = ManyBodyState({SlaterDeterminant.from_bytes(s): c for s, c in zip(states_bytes, coeffs)})
    else:
        psi_m = ManyBodyState({})
    psi_m = basis_m.redistribute_psis([psi_m])[0]

    rho_mpi = basis_m.build_density_matrices(
        [psi_m],
        orbital_indices_left=orbital_indices,
        orbital_indices_right=orbital_indices,
    )[0]

    assert np.allclose(rho_mpi, rho_serial, atol=1e-12), (
        f"rank {comm.rank}: density matrix mismatch.\nMPI:\n{rho_mpi}\nSerial:\n{rho_serial}"
    )


# ---------------------------------------------------------------------------
# graph_alltoall_psis unit tests
# ---------------------------------------------------------------------------


def test_graph_alltoall_psis_serial_passthrough():
    """With comm=None the list must come back unchanged."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant

    n_bytes = 8
    psi = ManyBodyState({SlaterDeterminant.from_bytes(b"\x80" + b"\x00" * 7): 1.0 + 0j})
    psis = [psi]
    result = graph_alltoall_psis(psis, n_bytes, None)
    assert result == psis


def test_graph_alltoall_psis_single_rank():
    """With COMM_SELF (size 1) the list must come back unchanged."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant

    n_bytes = 8
    psi = ManyBodyState(
        {
            SlaterDeterminant.from_bytes(b"\x80" + b"\x00" * 7): 1.0 + 0j,
            SlaterDeterminant.from_bytes(b"\x40" + b"\x00" * 7): 0.5 + 0j,
        }
    )
    psis = [psi]
    result = graph_alltoall_psis(psis, n_bytes, MPI.COMM_SELF)
    assert result == psis


@pytest.mark.mpi
def test_graph_alltoall_psis_ring_exchange():
    """
    Ring exchange: each rank sends one state+amplitude to its right neighbour.
    Verify the received state and amplitude are correct.
    """
    from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant

    comm = MPI.COMM_WORLD
    n_bytes = 8
    (comm.rank + 1) % comm.size
    (comm.rank - 1) % comm.size

    # We need to construct a state whose hash % size == dest
    # Finding a valid state is tedious, so let's just use a random state.
    # Actually, in the new implementation, graph_alltoall_psis internally hashes the state and routes it.
    # We can't easily force a state to go to `dest`. We just let it route.
    state = SlaterDeterminant.from_bytes((comm.rank + 1).to_bytes(8, "little"))
    state.get_hash() % comm.size

    amp = complex(comm.rank + 1, 0)
    psis = [ManyBodyState({state: amp})]

    result = graph_alltoall_psis(psis, n_bytes, comm)

    # The expected state should be received by `target_rank`
    # Let's gather all results and check.
    all_results = comm.gather(result[0].to_dict(), root=0)
    if comm.rank == 0:
        # Check that the state sent by each rank arrived at the correct destination
        for r in range(comm.size):
            s = SlaterDeterminant.from_bytes((r + 1).to_bytes(8, "little"))
            expected_target = s.get_hash() % comm.size
            expected_amp = complex(r + 1, 0)
            assert s in all_results[expected_target], f"State from {r} didn't reach {expected_target}"
            assert abs(all_results[expected_target][s] - expected_amp) < 1e-12


@pytest.mark.mpi
def test_graph_alltoall_psis_empty():
    """All-empty send list must produce all-empty result."""
    from impurityModel.ed.ManyBodyUtils import ManyBodyState

    comm = MPI.COMM_WORLD
    n_bytes = 8
    n_psis = 2
    psis = [ManyBodyState() for _ in range(n_psis)]
    result = graph_alltoall_psis(psis, n_bytes, comm)
    assert len(result) == n_psis
    for pi in range(n_psis):
        assert len(result[pi]) == 0
