import pytest
import pickle
from mpi4py import MPI
import numpy as np
from impurityModel.ed.manybody_basis import Basis, CIPSI_Basis
from math import ceil


def test_Basis_states():
    # 10000000  01000000 00100000 00010000 00001000
    exact = [b"\x80", b"\x40", b"\x20", b"\x10", b"\x08"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
            {0: [[]]},
        ),
        delta_valence_occ={0: 0},
        delta_conduction_occ={0: 0},
        delta_impurity_occ={0: 0},
        nominal_impurity_occ={0: 1},
        verbose=True,
    )
    assert all(state in basis for state in exact), basis.local_basis
    assert all(state in exact for state in basis)


def test_Basis_states_val():
    # 1000 0111 1100 0000  0100 0111 1100 0000  0010 0111 1100 0000  0001 0111 1100 0000  0000 1111 1100 0000
    exact = [b"\x87\xc0", b"\x47\xc0", b"\x27\xc0", b"\x17\xc0", b"\x0f\xc0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [list(range(5, 10))]},
            {0: [[]]},
            {0: [[]]},
        ),
        delta_valence_occ={0: 0},
        delta_conduction_occ={0: 0},
        delta_impurity_occ={0: 0},
        nominal_impurity_occ={0: 1},
        verbose=True,
    )
    assert all(state in basis for state in exact), basis.local_basis
    assert all(state in exact for state in basis)


def test_Basis_states_zero():
    # 1000 0011 1100 0000  0100 0101 1100 0000  0010 0110 1100 0000  0001 0111 0100 0000  0000 1111 1000 0000
    exact = [b"\x83\xc0", b"\x45\xc0", b"\x26\xc0", b"\x17\x40", b"\x0f\x80"]
    basis = Basis(
        impurity_orbitals={0: [[i] for i in range(5)]},
        bath_states=(
            {0: [[]]},
            {0: [[i] for i in range(5, 10)]},
            {0: [[]]},
        ),
        delta_valence_occ={0: 0},
        delta_conduction_occ={0: 0},
        delta_impurity_occ={0: 0},
        nominal_impurity_occ={0: 1},
        verbose=True,
    )
    assert all(state in basis for state in exact), f"{basis.local_basis=}"
    assert all(state in exact for state in basis)


@pytest.mark.parametrize(
    "valence_baths, conduction_baths, delta_valence_occ, delta_conduction_occ, delta_impurity_occ, nominal_impurity_occ, expected",
    [
        (0, 0, 0, 0, 0, 9, 10),
        (10, 0, 2, 0, 3, 8, 190),
        (10, 10, 2, 1, 3, 8, 10390),
    ],
)
def test_Basis_len(
    valence_baths,
    conduction_baths,
    delta_valence_occ,
    delta_conduction_occ,
    delta_impurity_occ,
    nominal_impurity_occ,
    expected,
):
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=(
            {0: [list(range(10, 10 + valence_baths))]},
            {0: [[]]},
            {0: [list(range(10 + valence_baths, 10 + valence_baths + conduction_baths))]},
        ),
        delta_valence_occ={0: delta_valence_occ},
        delta_conduction_occ={0: delta_conduction_occ},
        delta_impurity_occ={0: delta_impurity_occ},
        nominal_impurity_occ={0: nominal_impurity_occ},
        verbose=True,
    )
    print(f"{basis.num_spin_orbitals=} {basis.n_bytes=}")
    assert len(basis) == expected


@pytest.mark.mpi
@pytest.mark.parametrize(
    "valence_baths, conduction_baths, delta_valence_occ, delta_conduction_occ, delta_impurity_occ, nominal_impurity_occ, expected",
    [
        (0, 0, 0, 0, 0, 9, 10),
        (10, 0, 2, 0, 3, 8, 190),
        (10, 10, 2, 1, 3, 8, 10390),
    ],
)
def test_Basis_len_mpi(
    valence_baths,
    conduction_baths,
    delta_valence_occ,
    delta_conduction_occ,
    delta_impurity_occ,
    nominal_impurity_occ,
    expected,
):
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=(
            {0: [list(range(10, 10 + valence_baths))]},
            {0: [[]]},
            {0: [list(range(10 + valence_baths, 10 + valence_baths + conduction_baths))]},
        ),
        delta_valence_occ={0: delta_valence_occ},
        delta_conduction_occ={0: delta_conduction_occ},
        delta_impurity_occ={0: delta_impurity_occ},
        nominal_impurity_occ={0: nominal_impurity_occ},
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    assert len(basis) == expected


@pytest.mark.parametrize(
    "valence_baths, conduction_baths, delta_valence_occ, delta_conduction_occ, delta_impurity_occ, nominal_impurity_occ",
    [
        (0, 0, 0, 0, 0, 9),
        (10, 0, 2, 0, 3, 8),
        (10, 10, 2, 1, 3, 8),
    ],
)
def test_Basis_in(
    valence_baths,
    conduction_baths,
    delta_valence_occ,
    delta_conduction_occ,
    delta_impurity_occ,
    nominal_impurity_occ,
):
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=(
            {0: [list(range(10, 10 + valence_baths))]},
            {0: [[]]},
            {0: [list(range(10 + valence_baths, 10 + valence_baths + conduction_baths))]},
        ),
        delta_valence_occ={0: delta_valence_occ},
        delta_conduction_occ={0: delta_conduction_occ},
        delta_impurity_occ={0: delta_impurity_occ},
        nominal_impurity_occ={0: nominal_impurity_occ},
        verbose=True,
    )
    for state in basis:
        assert state in basis


@pytest.mark.mpi
@pytest.mark.parametrize(
    "valence_baths, conduction_baths, delta_valence_occ, delta_conduction_occ, delta_impurity_occ, nominal_impurity_occ",
    [
        (0, 0, 0, 0, 0, 9),
        (10, 0, 2, 0, 3, 8),
        (10, 10, 2, 1, 3, 8),
    ],
)
def test_Basis_in_mpi(
    valence_baths,
    conduction_baths,
    delta_valence_occ,
    delta_conduction_occ,
    delta_impurity_occ,
    nominal_impurity_occ,
):
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=(
            {0: [list(range(10, 10 + valence_baths))]},
            {0: [[]]},
            {0: [list(range(10 + valence_baths, 10 + valence_baths + conduction_baths))]},
        ),
        delta_valence_occ={0: delta_valence_occ},
        delta_conduction_occ={0: delta_conduction_occ},
        delta_impurity_occ={0: delta_impurity_occ},
        nominal_impurity_occ={0: nominal_impurity_occ},
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    for state in basis:
        assert state in basis


@pytest.mark.parametrize(
    "valence_baths, conduction_baths, delta_valence_occ, delta_conduction_occ, delta_impurity_occ, nominal_impurity_occ",
    [
        (0, 0, 0, 0, 0, 9),
        (10, 0, 2, 0, 3, 8),
        (10, 10, 2, 1, 3, 8),
    ],
)
def test_Basis_list(
    valence_baths,
    conduction_baths,
    delta_valence_occ,
    delta_conduction_occ,
    delta_impurity_occ,
    nominal_impurity_occ,
):
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=(
            {0: [list(range(10, 10 + valence_baths))]},
            {0: [[]]},
            {0: [list(range(10 + valence_baths, 10 + valence_baths + conduction_baths))]},
        ),
        delta_valence_occ={0: delta_valence_occ},
        delta_conduction_occ={0: delta_conduction_occ},
        delta_impurity_occ={0: delta_impurity_occ},
        nominal_impurity_occ={0: nominal_impurity_occ},
        verbose=True,
    )
    assert len(basis) == len(list(basis))


@pytest.mark.mpi
@pytest.mark.parametrize(
    "valence_baths, conduction_baths, delta_valence_occ, delta_conduction_occ, delta_impurity_occ, nominal_impurity_occ",
    [
        (0, 0, 0, 0, 0, 9),
        (10, 0, 2, 0, 3, 8),
        (10, 10, 2, 1, 3, 8),
    ],
)
def test_Basis_list_mpi(
    valence_baths,
    conduction_baths,
    delta_valence_occ,
    delta_conduction_occ,
    delta_impurity_occ,
    nominal_impurity_occ,
):
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=(
            {0: [list(range(10, 10 + valence_baths))]},
            {0: [[]]},
            {0: [list(range(10 + valence_baths, 10 + valence_baths + conduction_baths))]},
        ),
        delta_valence_occ={0: delta_valence_occ},
        delta_conduction_occ={0: delta_conduction_occ},
        delta_impurity_occ={0: delta_impurity_occ},
        nominal_impurity_occ={0: nominal_impurity_occ},
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    assert len(basis) == len(list(basis))


@pytest.mark.parametrize(
    "valence_baths, conduction_baths, delta_valence_occ, delta_conduction_occ, delta_impurity_occ, nominal_impurity_occ, expected",
    [
        (0, 0, 0, 0, 0, 9, 10),
        (10, 0, 2, 0, 3, 8, 190),
        (10, 10, 2, 1, 3, 8, 10390),
    ],
)
def test_CIPSI_Basis_len(
    valence_baths,
    conduction_baths,
    delta_valence_occ,
    delta_conduction_occ,
    delta_impurity_occ,
    nominal_impurity_occ,
    expected,
):
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=(
            {0: [list(range(10, 10 + valence_baths))]},
            {0: [[]]},
            {0: [list(range(10 + valence_baths, 10 + valence_baths + conduction_baths))]},
        ),
        delta_valence_occ={0: delta_valence_occ},
        delta_conduction_occ={0: delta_conduction_occ},
        delta_impurity_occ={0: delta_impurity_occ},
        nominal_impurity_occ={0: nominal_impurity_occ},
        verbose=True,
    )
    assert len(basis) == expected


@pytest.mark.mpi
@pytest.mark.parametrize(
    "valence_baths, conduction_baths, delta_valence_occ, delta_conduction_occ, delta_impurity_occ, nominal_impurity_occ, expected",
    [
        (0, 0, 0, 0, 0, 9, 10),
        (10, 0, 2, 0, 3, 8, 190),
        (10, 10, 2, 1, 3, 8, 10390),
    ],
)
def test_CIPSI_Basis_len_mpi(
    valence_baths,
    conduction_baths,
    delta_valence_occ,
    delta_conduction_occ,
    delta_impurity_occ,
    nominal_impurity_occ,
    expected,
):
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=(
            {0: [list(range(10, 10 + valence_baths))]},
            {0: [[]]},
            {0: [list(range(10 + valence_baths, 10 + valence_baths + conduction_baths))]},
        ),
        delta_valence_occ={0: delta_valence_occ},
        delta_conduction_occ={0: delta_conduction_occ},
        delta_impurity_occ={0: delta_impurity_occ},
        nominal_impurity_occ={0: nominal_impurity_occ},
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    assert len(basis) == expected


# @pytest.mark.mpi
# def test_Basis_alltoall_states():
#     basis = Basis(
#         valence_baths={2: 10},
#         conduction_baths={2: 0},
#         delta_valence_occ={2: 1},
#         delta_conduction_occ={2: 0},
#         delta_impurity_occ={2: 2},
#         nominal_impurity_occ={2: 1},
#         verbose=True,
#         comm=MPI.COMM_WORLD,
#     )
#     states = list(basis)
#     offset = 0
#     send_list = []
#     ranks = MPI.COMM_WORLD.size
#     for rank in range(ranks):
#         n_states = len(states) // ranks + (1 if rank < len(states)%ranks else 0)
#         send_list.append(states[offset:offset + n_states])
#         offset += n_states
#     print(f"rank {MPI.COMM_WORLD.rank}: {send_list=}", flush=True)
#     received_states = basis.alltoall_states(send_list)
#     n_states = len(states) // ranks + (1 if MPI.COMM_WORLD.rank < len(states)%ranks else 0)
#     offset = MPI.COMM_WORLD.scan(n_states, op=MPI.SUM)
#     assert np.all(states[offset:offset + n_states] == received_states)


# @pytest.mark.mpi
# def test_Basis_dense_matrix():
#     basis = Basis(
# ls=[2],
#         valence_baths={2: 10},
#         conduction_baths={2: 0},
#         delta_valence_occ={2: 1},
#         delta_conduction_occ={2: 0},
#         delta_impurity_occ={2: 2},
#         nominal_impurity_occ={2: 8},
#         verbose=True,
#         comm=MPI.COMM_WORLD,
#     )
#     print(f"{len(basis)=}", flush=True)
#     with open("Ni_hamiltonian.pickle", "rb") as f:
#         h_op = pickle.load(f)

#     h_dict = basis.expand(h_op, {})
#     print(f"h_dict contains {len(h_dict)} elements.", flush=True)
#     h_matrix = basis.build_dense_matrix(h_op, h_dict)
#     print(f"{h_matrix.shape=}", flush=True)
#     eigvals = np.linalg.eigvalsh(h_matrix, UPLO="L")
#     # eigvals = np.sort(eigvals)

#     print(f"{eigvals - eigvals[0]=}")
#     assert eigvals[0] != 0
#     assert np.all(np.abs(eigvals[:9] - eigvals[0]) <= 1e-12)


@pytest.mark.mpi
def test_contains_2():
    states = [b"\x00\x1a\x2b", b"\xff\x00\x1a"]
    basis = Basis(
        impurity_orbitals={0: [list(range(24))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    assert basis.contains(bytes(b"\x00\x1a\x2b"))
    assert not basis.contains(bytes(b"\xff\x1a\x2b"))
    assert all(basis.contains(states))
    assert basis.index(states[0]) == 0
    assert basis.index(states[1]) == 1


@pytest.mark.mpi
@pytest.mark.parametrize(
    "n_bytes, n_states",
    [
        (3, 2),
        (3, 5),
        (3, 1000),
        (10, 5),
        (10, 1000),
    ],
)
def test_contains_random(n_bytes, n_states):
    state_bytes = np.random.randint(0, high=255, size=n_states * n_bytes, dtype=np.ubyte)
    MPI.COMM_WORLD.Bcast(state_bytes)
    states = [i.tobytes() for i in np.split(state_bytes, n_states)]
    basis = Basis(
        impurity_orbitals={0: [list(range(8 * n_bytes))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    sorted_states = sorted(set(states))
    sorted_indices = basis.index(sorted_states)
    too_large_state = np.array(np.frombuffer(sorted_states[-1], dtype=np.ubyte, count=n_bytes))
    too_large_state[-1] += 1
    assert all(basis.contains(states))
    assert too_large_state.tobytes() not in basis
    assert not list(basis.contains(states + [too_large_state.tobytes()]))[-1]
    assert all(si == i for si, i in enumerate(sorted_indices))
    for i in range(len(sorted_states)):
        assert basis.index(sorted_states[i]) == i


@pytest.mark.mpi
@pytest.mark.parametrize(
    "n_bytes, n_states",
    [
        (3, 2),
        (3, 5),
        (10, 5),
        (3, 100),
        (10, 100),
    ],
)
def test_contains_random_distributed(n_bytes, n_states):
    state_bytes = np.random.randint(0, high=255, size=n_states * n_bytes, dtype=np.ubyte)
    states = [i.tobytes() for i in np.split(state_bytes, n_states)]
    basis = Basis(
        impurity_orbitals={0: [list(range(8 * n_bytes))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    all_states = MPI.COMM_WORLD.allgather(states)
    all_states = [state for states in all_states for state in states]
    sorted_states = sorted(set(all_states))
    too_large_state = np.array(np.frombuffer(sorted_states[-1], dtype=np.ubyte, count=n_bytes))
    too_large_state[-1] += 1
    assert all(basis.contains(states))
    assert too_large_state.tobytes() not in basis
    assert not list(basis.contains(states + [too_large_state.tobytes()]))[-1]
    sorted_indices = basis.index(sorted_states)
    assert all(si == i for i, si in enumerate(sorted_indices))
    for i in range(len(sorted_states)):
        assert basis.index(sorted_states[i]) == i


@pytest.mark.mpi
@pytest.mark.parametrize(
    "n_bytes, n_states, n_sample_states",
    [
        (3, 2, 100),
        (3, 5, 100),
        (3, 1000, 100),
        (10, 5, 100),
        (10, 1000, 100),
        (10, 1000, 100),
    ],
)
def test_contains_random_distributed_random(n_bytes, n_states, n_sample_states):
    state_bytes = np.random.randint(0, high=255, size=n_states * n_bytes, dtype=np.ubyte)
    states = [i.tobytes() for i in np.split(state_bytes, n_states)]
    basis = Basis(
        impurity_orbitals={0: [list(range(8 * n_bytes))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    all_states = MPI.COMM_WORLD.allgather(states)
    all_states = {state for states in all_states for state in states}
    sample_bytes = np.random.randint(0, high=255, size=n_sample_states * n_bytes, dtype=np.ubyte)
    sample_states = [i.tobytes() for i in np.split(sample_bytes, n_sample_states)]
    correct_contains = [state in all_states for state in sample_states]
    basis_contains = basis.contains(sample_states)
    assert all(correct == sample for correct, sample in zip(correct_contains, basis_contains))


@pytest.mark.mpi
@pytest.mark.parametrize(
    "n_bytes, n_states, n_sample_states",
    [
        (3, 2, 100),
        (3, 5, 1000),
        (3, 100, 1000),
        (10, 5, 100),
        (10, 100, 100),
        (10, 100, 1000),
    ],
)
def test_index_random_distributed_random(n_bytes, n_states, n_sample_states):
    state_bytes = np.random.randint(0, high=255, size=n_states * n_bytes, dtype=np.ubyte)
    states = [i.tobytes() for i in np.split(state_bytes, n_states)]
    basis = Basis(
        impurity_orbitals={0: [list(range(8 * n_bytes))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    all_states = MPI.COMM_WORLD.allgather(states)
    all_states = sorted({state for states in all_states for state in states})
    sample_bytes = np.random.randint(0, high=255, size=n_sample_states * n_bytes, dtype=np.ubyte)
    sample_states = [i.tobytes() for i in np.split(sample_bytes, n_sample_states)]
    correct_indices = [all_states.index(state) for state in sample_states if state in all_states]
    basis_mask = list(basis.contains(sample_states))
    samples_in_basis = [sample_states[i] for i in range(len(sample_states)) if basis_mask[i]]
    basis_indices = basis.index(samples_in_basis)
    assert all(bi == ci for bi, ci in zip(basis_indices, correct_indices))


def test_operator_dict_simple():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
    }
    states = [b"\x78"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    op_dict = basis.build_operator_dict(operator)
    correct = {b"\x78": {b"\xF0": -1 / 2, b"\x78": 9 / 2}}
    assert all(fk in correct for fk in op_dict.keys())
    for key in states:
        assert all(fk in correct[key] for fk in op_dict[key].keys())
        for row in correct[key].keys():
            assert correct[key][row] == op_dict[key][row]


@pytest.mark.mpi
def test_operator_dict_simple_mpi():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
    }
    states = [b"\x78"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    op_dict = basis.build_operator_dict(operator)
    correct = {b"\x78": {b"\xF0": -1 / 2, b"\x78": 9 / 2}}
    all_dicts = MPI.COMM_WORLD.allgather(op_dict)
    full_dict = {}
    for d in all_dicts:
        for key in d:
            if key not in full_dict:
                full_dict[key] = {}
            for state in d[key]:
                full_dict[key][state] = d[key][state] + full_dict[key].get(state, 0)
    assert all(fk in correct for fk in full_dict.keys())
    for key in states:
        assert all(fk in correct[key] for fk in full_dict[key].keys())
        for row in correct[key].keys():
            assert correct[key][row] == full_dict[key][row]


def test_operator_dict_simple_with_extra_states():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
        ((0, "c"),): 500,
    }
    states = [b"\x78"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    op_dict = basis.build_operator_dict(operator)
    correct = {b"\x78": {b"\xF0": -1 / 2, b"\x78": 9 / 2, b"\xF8": 500}}
    assert all(fk in correct for fk in op_dict.keys())
    for key in states:
        assert all(fk in correct[key] for fk in op_dict[key].keys())
        for row in correct[key].keys():
            assert correct[key][row] == op_dict[key][row]


@pytest.mark.mpi
def test_operator_dict_simple_with_extra_states_mpi():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
        ((0, "c"),): 500,
    }
    states = [b"\x78"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    op_dict = basis.build_operator_dict(operator)
    correct = {b"\x78": {b"\xF0": -1 / 2, b"\x78": 9 / 2, b"\xF8": 500}}
    all_dicts = MPI.COMM_WORLD.allgather(op_dict)
    full_dict = {}
    for d in all_dicts:
        for key in d:
            if key not in full_dict:
                full_dict[key] = {}
            for state in d[key]:
                full_dict[key][state] = d[key][state] + full_dict[key].get(state, 0)
    assert all(fk in correct for fk in full_dict.keys())
    for key in states:
        assert all(fk in correct[key] for fk in full_dict[key].keys())
        for row in correct[key].keys():
            assert correct[key][row] == full_dict[key][row]


def test_operator_dict_eg_t2g():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
    }
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    op_dict = basis.build_operator_dict(operator)
    correct = {
        b"\x78": {b"\xF0": -1 / 2, b"\x78": 9 / 2},
        b"\xB8": {b"\xB8": 9 / 2},
        b"\xD8": {b"\xD8": 4},
        b"\xE8": {b"\xE8": 9 / 2},
        b"\xF0": {b"\x78": -1 / 2, b"\xF0": 9 / 2},
    }

    assert all(fk in correct for fk in op_dict.keys())
    for key in states:
        assert all(fk in correct[key] for fk in op_dict[key].keys())
        for row in correct[key].keys():
            assert correct[key][row] == op_dict[key][row], f"{key=} {row=}"


@pytest.mark.mpi
def test_operator_dict_eg_t2g_mpi():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
    }
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    op_dict = basis.build_operator_dict(operator)
    correct = {
        b"\x78": {b"\xF0": -1 / 2, b"\x78": 9 / 2},
        b"\xB8": {b"\xB8": 9 / 2},
        b"\xD8": {b"\xD8": 4},
        b"\xE8": {b"\xE8": 9 / 2},
        b"\xF0": {b"\x78": -1 / 2, b"\xF0": 9 / 2},
    }
    all_dicts = MPI.COMM_WORLD.allgather(op_dict)
    full_dict = {}
    for d in all_dicts:
        for key in d:
            if key not in full_dict:
                full_dict[key] = {}
            for state in d[key]:
                full_dict[key][state] = d[key][state] + full_dict[key].get(state, 0)

    assert all(fk in correct for fk in full_dict.keys())
    for key in states:
        assert all(fk in correct[key] for fk in full_dict[key].keys())
        for row in correct[key].keys():
            assert correct[key][row] == full_dict[key][row], f"{key=} {row=}"


def test_operator_dict_eg_t2g_with_extra_states():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
        ((0, "c"),): 500,
    }
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    op_dict = basis.build_operator_dict(operator)
    correct = {
        b"\x78": {b"\xF0": -1 / 2, b"\x78": 9 / 2, b"\xF8": 500},
        b"\xB8": {b"\xB8": 9 / 2},
        b"\xD8": {b"\xD8": 4},
        b"\xE8": {b"\xE8": 9 / 2},
        b"\xF0": {b"\x78": -1 / 2, b"\xF0": 9 / 2},
    }

    assert all(fk in correct for fk in op_dict.keys())
    for key in states:
        assert all(
            fk in correct[key] for fk in op_dict[key].keys()
        ), f"{list(op_dict[key].keys())=} {list(correct.keys())=} "
        for row in correct[key].keys():
            assert correct[key][row] == op_dict[key][row], f"{key=} {row=}"


@pytest.mark.mpi
def test_operator_dict_eg_t2g_with_extra_states_mpi():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
        ((0, "c"),): 500,
    }
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    op_dict = basis.build_operator_dict(operator)
    correct = {
        b"\x78": {b"\xF0": -1 / 2, b"\x78": 9 / 2, b"\xF8": 500},
        b"\xB8": {b"\xB8": 9 / 2},
        b"\xD8": {b"\xD8": 4},
        b"\xE8": {b"\xE8": 9 / 2},
        b"\xF0": {b"\x78": -1 / 2, b"\xF0": 9 / 2},
    }
    all_dicts = MPI.COMM_WORLD.allgather(op_dict)
    full_dict = {}
    for d in all_dicts:
        for key in d:
            if key not in full_dict:
                full_dict[key] = {}
            for state in d[key]:
                full_dict[key][state] = d[key][state] + full_dict[key].get(state, 0)

    assert all(fk in correct for fk in full_dict.keys())
    for key in states:
        assert all(
            fk in correct[key] for fk in full_dict[key].keys()
        ), f"{list(full_dict[key].keys())=} {list(correct.keys())=} "
        for row in correct[key].keys():
            assert correct[key][row] == full_dict[key][row], f"{key=} {row=}"


def test_simple_dense_matrix():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
    }
    states = [b"\x78"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    op_dict = basis.build_operator_dict(operator)
    dense_mat = basis.build_dense_matrix(operator, op_dict)
    assert dense_mat.shape == (1, 1)
    assert dense_mat[0, 0] == 9 / 2


@pytest.mark.mpi
def test_simple_dense_matrix_mpi():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
    }
    states = [b"\x78"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    op_dict = basis.build_operator_dict(operator)
    dense_mat = basis.build_dense_matrix(operator, op_dict)
    assert dense_mat.shape == (1, 1)
    assert dense_mat[0, 0] == 9 / 2


def test_eg_t2g_dense_matrix():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
    }
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    op_dict = basis.build_operator_dict(operator)
    dense_mat = basis.build_dense_matrix(operator, op_dict)
    assert dense_mat.shape == (5, 5)
    assert np.allclose(
        dense_mat,
        np.array(
            [
                [9 / 2, 0, 0, 0, -1 / 2],
                [0, 9 / 2, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 9 / 2, 0],
                [-1 / 2, 0, 0, 0, 9 / 2],
            ],
            dtype=float,
        ),
    ), f"{dense_mat=}"


@pytest.mark.mpi
def test_eg_t2g_dense_matrix_mpi():
    operator = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
    }
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    op_dict = basis.build_operator_dict(operator)
    dense_mat = basis.build_dense_matrix(operator, op_dict)
    assert dense_mat.shape == (5, 5)
    assert np.allclose(
        dense_mat,
        np.array(
            [
                [9 / 2, 0, 0, 0, -1 / 2],
                [0, 9 / 2, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 9 / 2, 0],
                [-1 / 2, 0, 0, 0, 9 / 2],
            ],
            dtype=float,
        ),
    ), f"{dense_mat=}"


def test_simple_vector():
    states = [b"\x00\x1a\x2b", b"\xff\x00\x1a"]
    basis = Basis(
        impurity_orbitals={0: [list(range(24))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )
    state = {}
    if states[0] in basis._index_dict:
        state[states[0]] = 0.25 + 0.2j
    if states[1] in basis._index_dict:
        state[states[1]] = 0.33 + 0.15j

    v = basis.build_vector([state])[0]
    v_exact = np.array([0.25 + 0.2j, 0.33 + 0.15j], dtype=complex)

    assert v.shape == (len(basis),)
    assert v.shape == v_exact.shape
    assert np.allclose(v, v_exact)


@pytest.mark.mpi
def test_simple_vector_mpi():
    states = [b"\x00\x1a\x2b", b"\xff\x00\x1a"]
    basis = Basis(
        impurity_orbitals={0: [list(range(24))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    state = {}
    if states[0] in basis._index_dict:
        state[states[0]] = 0.25 + 0.2j
    if states[1] in basis._index_dict:
        state[states[1]] = 0.33 + 0.15j

    v = basis.build_vector([state])[0]
    v_exact = np.array([0.25 + 0.2j, 0.33 + 0.15j], dtype=complex)

    assert v.shape == (len(basis),)
    assert v.shape == v_exact.shape
    assert np.allclose(v, v_exact)


def test_vector():
    comm = None
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states[:-1],
        verbose=True,
        comm=None,
    )
    state = {states[-1]: 1 + 1j}
    state[states[0]] = 0.25 + 0.2j
    if states[1] in basis._index_dict:
        state[states[1]] = 0.33 + 0.15j
    v = basis.build_vector([state])[0]
    v_exact = np.array([0.25 + 0.2j, 0.33 + 0.15j, 0, 0], dtype=complex)

    assert v.shape == (len(basis),)
    assert v.shape == v_exact.shape
    assert np.allclose(v, v_exact)


@pytest.mark.mpi
def test_vector_mpi():
    comm = MPI.COMM_WORLD
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states[:-1],
        verbose=True,
        comm=comm,
    )
    state = {states[-1]: 1 + 1j}
    state[states[0]] = 0.25 + 0.2j
    if states[1] in basis._index_dict:
        state[states[1]] = 0.33 + 0.15j
    v = basis.build_vector([state])[0]
    v_exact = np.array([0.25 * comm.size + 0.2j * comm.size, 0.33 + 0.15j, 0, 0], dtype=complex)

    assert v.shape == (len(basis),)
    assert v.shape == v_exact.shape
    assert np.allclose(v, v_exact)


def test_simple_state():
    states = [b"\x00\x1a\x2b", b"\xff\x00\x1a"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    v = np.array([[1.0, -2.5]])
    s = basis.build_state(v)
    s_exact = [{states[0]: v[0, 0], states[1]: v[0, 1]}]

    for i in range(len(s)):
        assert all(s[i][state] == s_exact[i][state] for state in s[i])


@pytest.mark.mpi
def test_simple_state_mpi():
    states = [b"\x00\x1a\x2b", b"\xff\x00\x1a"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    v = np.array([[1.0, -2.5]])
    s = basis.build_state(v)
    s_exact = [{states[0]: v[0, 0], states[1]: v[0, 1]}]

    for i in range(len(s)):
        assert all(s[i][state] == s_exact[i][state] for state in s[i])


def test_state_mpi():
    comm = None
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=comm,
    )
    v = np.array([[1.0, -2.5, 0, 0, 1.2], [0, 3, 1, 0, 0]])
    s = basis.build_state(v)
    s_exact = [
        {states[0]: v[0, 0], states[1]: v[0, 1], states[4]: v[0, 4]},
        {states[1]: v[1, 1], states[2]: v[1, 2]},
    ]

    for i in range(len(s)):
        assert all(s[i][state] == s_exact[i][state] for state in s[i])


@pytest.mark.mpi
def test_state_mpi():
    comm = MPI.COMM_WORLD
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=comm,
    )
    v = np.array([[1.0, -2.5, 0, 0, 1.2], [0, 3, 1, 0, 0]])
    s = basis.build_state(v)
    s_exact = [
        {states[0]: v[0, 0], states[1]: v[0, 1], states[4]: v[0, 4]},
        {states[1]: v[1, 1], states[2]: v[1, 2]},
    ]

    for i in range(len(s)):
        assert all(s[i][state] == s_exact[i][state] for state in s[i])


# def test_spin_flip():
#     comm = None
#     # dn 11001 -> n_dn = 3
#     # up 00101 -> n_up = 2
#     states = [b"\xC9\x40"]
#     basis = Basis(
#         impurity_orbitals={0: [list(range(10))]},
#         bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
#         initial_basis=[],
#         verbose=True,
#         comm=comm,
#     )
#     spin_flipped = basis._generate_spin_flipped_determinants(states)
#     # flips:
#     # dn 11001  01001  10001  11101  00001  10101  01101  00101
#     # up 00101  10101  01101  00001  11101  01001  10001  11001
#     spin_flipped_check = [
#         b"\xC9\x40",
#         b"\x4D\x40",
#         b"\x8B\x40",
#         # b"\xE8\x40",
#         # b"\x0F\x40",
#         b"\xAA\x40",
#         b"\x6C\x40",
#         b"\x2E\x40",
#     ]
#     assert all(state in spin_flipped for state in spin_flipped_check), f"{spin_flipped_check=} {spin_flipped=}"
#     assert all(state in spin_flipped_check for state in spin_flipped), f"{spin_flipped_check=} {spin_flipped=}"


# @pytest.mark.mpi
# def test_spin_flip_mpi():
#     comm = MPI.COMM_WORLD
#     # dn 11001 -> n_dn = 3
#     # up 00101 -> n_up = 2
#     states = [b"\xC9\x40"]
#     basis = Basis(
#         impurity_orbitals={0: [list(range(10))]},
#         bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
#         initial_basis=[],
#         verbose=True,
#         comm=comm,
#     )
#     spin_flipped = basis._generate_spin_flipped_determinants(states)
#     # flips:
#     # dn 11001  01001  10001  11101  00001  10101  01101  00101
#     # up 00101  10101  01101  00001  11101  01001  10001  11001
#     spin_flipped_check = [
#         b"\xC9\x40",
#         b"\x4D\x40",
#         b"\x8B\x40",
#         # b"\xE8\x40",
#         # b"\x0F\x40",
#         b"\xAA\x40",
#         b"\x6C\x40",
#         b"\x2E\x40",
#     ]
#     assert all(state in spin_flipped for state in spin_flipped_check), f"{spin_flipped_check=} {spin_flipped=}"
#     assert all(state in spin_flipped_check for state in spin_flipped), f"{spin_flipped_check=} {spin_flipped=}"


@pytest.mark.mpi
def test_alltoall_states_mpi():
    comm = MPI.COMM_WORLD
    num_spin_orbitals = comm.size
    bytes_per_state = int(ceil(num_spin_orbitals / 8))
    send_states = [[r.to_bytes(bytes_per_state, "big")] for r in range(comm.size)]
    basis = Basis(
        impurity_orbitals={0: [list(range(num_spin_orbitals))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=[],
        verbose=True,
        comm=comm,
    )
    received_states = basis.alltoall_states(send_states)
    assert all(
        state == comm.rank.to_bytes(bytes_per_state, "big") for rs in received_states for state in rs
    ), f"{comm.rank=} {received_states=} {basis.local_basis=}"


@pytest.mark.mpi
def test_alltoall_states_with_empty_mpi():
    comm = MPI.COMM_WORLD
    num_spin_orbitals = comm.size
    bytes_per_state = ceil(num_spin_orbitals / 8)
    send_states = [[r.to_bytes(bytes_per_state, "big")] if r < comm.rank else [] for r in range(comm.size)]
    basis = Basis(
        impurity_orbitals={0: [list(range(num_spin_orbitals))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=[],
        verbose=True,
        comm=comm,
    )
    basis.add_states([comm.rank.to_bytes(bytes_per_state, "big")])
    received_states = basis.alltoall_states(send_states)
    assert all(
        state == comm.rank.to_bytes(bytes_per_state, "big") for rs in received_states for state in list(rs)
    ), f"{comm.rank=} {received_states=} {basis.local_basis=}"


def test_eg_t2g_basis_expand():
    Hop = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
        ((5, "c"), (5, "a")): 1,
        ((5, "c"), (9, "a")): 1 / 2,
        ((9, "c"), (5, "a")): 1 / 2,
        ((9, "c"), (9, "a")): 1,
        ((7, "c"), (7, "a")): 3 / 2,
        ((6, "c"), (6, "a")): 1,
        ((8, "c"), (8, "a")): 1,
    }
    # Start with 10000  01000
    #            00000  00000
    states = [b"\x80\x00", b"\x40\x00"]
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    basis.expand(Hop)
    # expect 10000  01000  00001  00000  00000  00000
    #        00000  00000  00000  10000  01000  00001

    expected = [b"\x80\x00", b"\x40\x00", b"\x08\x00"]  # , b"\x04\x00", b"\x02\x00", b"\x00\x40"]
    assert all(state in expected for state in basis), f"{expected=} {list(basis)=}"
    assert all(state in basis for state in expected), f"{expected=} {list(basis)=}"


@pytest.mark.mpi
def test_eg_t2g_basis_expand_mpi():
    Hop = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
        ((5, "c"), (5, "a")): 1,
        ((5, "c"), (9, "a")): 1 / 2,
        ((9, "c"), (5, "a")): 1 / 2,
        ((9, "c"), (9, "a")): 1,
        ((7, "c"), (7, "a")): 3 / 2,
        ((6, "c"), (6, "a")): 1,
        ((8, "c"), (8, "a")): 1,
    }
    # Start with 10000  01000
    #            00000  00000
    states = [b"\x80\x00", b"\x40\x00"]
    basis = Basis(
        impurity_orbitals={2: [list(range(10))]},
        bath_states=({2: [[]]}, {2: [[]]}, {2: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    basis.expand(Hop)
    # expect 10000  01000  00001  00000  00000  00000
    #        00000  00000  00000  10000  01000  00001

    expected = [b"\x80\x00", b"\x40\x00", b"\x08\x00"]  # , b"\x04\x00", b"\x02\x00", b"\x00\x40"]
    assert all(state in expected for state in basis), f"{expected=} {list(basis)=}"
    assert all(state in basis for state in expected), f"{expected=} {list(basis)=}"


def test_eg_t2g_CIPSI_basis_expand():
    Hop = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
        ((5, "c"), (5, "a")): 1,
        ((5, "c"), (9, "a")): 1 / 2,
        ((9, "c"), (5, "a")): 1 / 2,
        ((9, "c"), (9, "a")): 1,
        ((7, "c"), (7, "a")): 3 / 2,
        ((6, "c"), (6, "a")): 1,
        ((8, "c"), (8, "a")): 1,
    }
    # Start with 10000
    #            00000
    states = [b"\x80\x00"]
    basis = CIPSI_Basis(
        impurity_orbitals={2: [list(range(10))]},
        bath_states=({2: [[]]}, {2: [[]]}, {2: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    basis.expand(Hop)

    expected = [b"\x80\x00", b"\x08\x00"]
    assert all(state in expected for state in basis), f"{expected=} {list(basis)=}"
    assert all(state in basis for state in expected), f"{expected=} {list(basis)=}"


@pytest.mark.mpi
def test_eg_t2g_CIPSI_basis_expand_mpi():
    Hop = {
        ((0, "c"), (0, "a")): 1,
        ((0, "c"), (4, "a")): 1 / 2,
        ((4, "c"), (0, "a")): 1 / 2,
        ((4, "c"), (4, "a")): 1,
        ((2, "c"), (2, "a")): 3 / 2,
        ((1, "c"), (1, "a")): 1,
        ((3, "c"), (3, "a")): 1,
        ((5, "c"), (5, "a")): 1,
        ((5, "c"), (9, "a")): 1 / 2,
        ((9, "c"), (5, "a")): 1 / 2,
        ((9, "c"), (9, "a")): 1,
        ((7, "c"), (7, "a")): 3 / 2,
        ((6, "c"), (6, "a")): 1,
        ((8, "c"), (8, "a")): 1,
    }
    # Start with 10000
    #            00000
    states = [b"\x80\x00"]
    basis = CIPSI_Basis(
        impurity_orbitals={2: [list(range(10))]},
        bath_states=({2: [[]]}, {2: [[]]}, {2: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    basis.expand(Hop)
    # expect 10000  00001  00000  00000
    #        00000  00000  10000  00001

    expected = [b"\x80\x00", b"\x08\x00"]
    assert all(state in expected for state in basis), f"{expected=} {list(basis)=}"
    assert all(state in basis for state in expected), f"{expected=} {list(basis)=}"


@pytest.mark.mpi
def test_distributed_simple_vector():
    states = (b"\x00\x1a\x2b", b"\xff\x00\x1a")
    basis = Basis(
        impurity_orbitals={2: [list(range(24))]},
        bath_states=({2: [[]]}, {2: [[]]}, {2: [[]]}),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    state = {}
    if states[0] in basis._index_dict:
        state[states[0]] = 0.25 + 0.2j
    if states[1] in basis._index_dict:
        state[states[1]] = 0.33 + 0.15j

    v = basis.build_distributed_vector([state])[0]
    v_exact = np.zeros((len(basis.local_basis),), dtype=complex)
    if states[0] in basis._index_dict:
        v_exact[0] = 0.25 + 0.2j
    if states[1] in basis._index_dict:
        v_exact[-1] = 0.33 + 0.15j

    assert v.shape == (len(basis.local_basis),)
    assert v.shape == v_exact.shape
    assert np.allclose(v, v_exact)


@pytest.mark.mpi
def test_distributed_vector_mpi():
    comm = MPI.COMM_WORLD
    states = [b"\x78", b"\xB8", b"\xD8", b"\xE8", b"\xF0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}, {0: [[]]}),
        initial_basis=states[:-1],
        verbose=True,
        comm=comm,
    )
    state = {states[-1]: 1 + 1j}
    state[states[0]] = 0.25 + 0.2j
    if states[1] in basis._index_dict:
        state[states[1]] = 0.33 + 0.15j
    v = basis.build_distributed_vector([state])[0]
    v_exact = np.array([0.25 * comm.size + 0.2j * comm.size, 0.33 + 0.15j, 0, 0], dtype=complex)

    n = len(basis.local_basis)
    assert v.shape == (len(basis.local_basis),)
    if n > 0:
        assert np.allclose(v, v_exact[basis.index_bounds[comm.rank] - n : basis.index_bounds[comm.rank]])
