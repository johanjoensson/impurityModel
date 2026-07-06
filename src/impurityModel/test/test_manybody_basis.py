import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.basis_restrictions import get_effective_restrictions
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, SlaterDeterminant
from impurityModel.ed.basis_transcription import (
    build_dense_matrix,
    build_distributed_vector,
    build_local_operator_list,
    build_state,
    build_vector,
)


def build_operator_dict(basis, op):
    """Express op in the current basis: map each local basis state to the result of applying op to it."""
    return dict(zip(basis.local_basis, build_local_operator_list(basis, ManyBodyOperator(op), 0)))


def build_states(states: list[bytes]):
    Basis(
        impurity_orbitals={0: [list(range(8 * len(states[0])))]},
        bath_states=({0: []}, {0: []}),
        initial_basis=[],
    )
    return [SlaterDeterminant.from_bytes(state) for state in states]


def test_Basis_states():
    # 10000000  01000000 00100000 00010000 00001000
    exact = build_states([b"\x80", b"\x40", b"\x20", b"\x10", b"\x08"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        delta_valence_occ={0: 0},
        delta_conduction_occ={0: 0},
        delta_impurity_occ={0: 0},
        nominal_impurity_occ={0: 1},
        verbose=True,
    )
    assert all(state in basis for state in exact), f"{basis.local_basis=}\n{exact=}"
    assert all(state in exact for state in basis)


def test_Basis_states_val():
    # 1000 0111 1100 0000  0100 0111 1100 0000  0010 0111 1100 0000  0001 0111 1100 0000  0000 1111 1100 0000
    exact = build_states([b"\x87\xc0", b"\x47\xc0", b"\x27\xc0", b"\x17\xc0", b"\x0f\xc0"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [list(range(5, 10))]},
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


@pytest.mark.mpi
def test_contains_2():
    states = build_states([b"\x00\x1a\x2b", b"\xff\x00\x1a"])
    basis = Basis(
        impurity_orbitals={0: [list(range(24))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    assert basis.contains(build_states([b"\x00\x1a\x2b"])[0])
    assert not basis.contains(build_states([b"\xff\x1a\x2b"])[0])
    assert all(basis.contains(states))
    # assert basis.index(states[0]) == 0
    # assert basis.index(states[1]) == 1


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
    states = build_states([i.tobytes() for i in np.split(state_bytes, n_states)])
    basis = Basis(
        impurity_orbitals={0: [list(range(8 * n_bytes))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    sorted_states = sorted(set(states))
    # sorted_indices = basis.index(sorted_states)
    import copy

    too_large_state = copy.copy(sorted_states[-1])
    too_large_state[-1] += 1

    assert not (sorted_states[-1] == too_large_state)
    assert all(basis.contains(states))
    assert too_large_state not in basis
    assert not list(basis.contains(states + [too_large_state]))[-1]
    # assert all(si == i for si, i in enumerate(sorted_indices))
    # for i in range(len(sorted_states)):
    #     assert basis.index(sorted_states[i]) == i


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
    states = build_states([i.tobytes() for i in np.split(state_bytes, n_states)])
    basis = Basis(
        impurity_orbitals={0: [list(range(8 * n_bytes))]},
        bath_states=(
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
    # too_large_state = np.array(np.frombuffer(sorted_states[-1], dtype=np.ubyte, count=n_bytes))
    too_large_state = sorted_states[-1].copy()
    too_large_state[-1] += 1
    assert all(basis.contains(states))
    assert too_large_state not in basis
    assert not list(basis.contains(states + [too_large_state]))[-1]
    basis.index(sorted_states)
    # assert all(si == i for i, si in enumerate(sorted_indices))
    # for i in range(len(sorted_states)):
    #     assert basis.index(sorted_states[i]) == i


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
    states = [SlaterDeterminant.from_bytes(i.tobytes()) for i in np.split(state_bytes, n_states)]
    basis = Basis(
        impurity_orbitals={0: [list(range(8 * n_bytes))]},
        bath_states=(
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
    sample_states = [SlaterDeterminant.from_bytes(i.tobytes()) for i in np.split(sample_bytes, n_sample_states)]
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
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    all_states = MPI.COMM_WORLD.allgather(states)
    all_states = sorted({state for states in all_states for state in states})
    sample_bytes = np.random.randint(0, high=255, size=n_sample_states * n_bytes, dtype=np.ubyte)
    sample_states = [i.tobytes() for i in np.split(sample_bytes, n_sample_states)]
    [all_states.index(state) for state in sample_states if state in all_states]
    basis_mask = list(basis.contains(sample_states))
    samples_in_basis = [sample_states[i] for i in range(len(sample_states)) if basis_mask[i]]
    basis.index(samples_in_basis)
    # assert all(bi == ci for bi, ci in zip(basis_indices, correct_indices))


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
    # 0111 1000
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[],
        verbose=True,
        comm=None,
    )
    states = [SlaterDeterminant.from_bytes(b"\x78")]
    basis.add_states(states)

    op_dict = build_operator_dict(basis, operator)
    correct = {
        SlaterDeterminant.from_bytes(b"\x78"): {
            SlaterDeterminant.from_bytes(b"\xf0"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\x78"): 9 / 2,
        }
    }
    # correct = {b"\x78": {b"\xf0": -1 / 2, b"\x78": 9 / 2}}
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
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=[],
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    states = build_states([b"\x78"])
    basis.add_states(states)

    op_dict = build_operator_dict(basis, operator)
    print(f"{op_dict=}")
    correct = {
        SlaterDeterminant.from_bytes(b"\x78"): {
            SlaterDeterminant.from_bytes(b"\xf0"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\x78"): 9 / 2,
        }
    }
    # correct = {b"\x78": {b"\xf0": -1 / 2, b"\x78": 9 / 2}}
    all_dicts = MPI.COMM_WORLD.allgather(op_dict)
    print(f"{all_dicts=}")
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
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=[],
        verbose=True,
        comm=None,
    )
    states = [SlaterDeterminant.from_bytes(b"\x78")]
    basis.add_states(states)
    # states = [b"\x78"]

    op_dict = build_operator_dict(basis, operator)
    correct = {
        SlaterDeterminant.from_bytes(b"\x78"): {
            SlaterDeterminant.from_bytes(b"\xf0"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\x78"): 9 / 2,
            SlaterDeterminant.from_bytes(b"\xf8"): 500,
        }
    }
    # correct = {b"\x78": {b"\xf0": -1 / 2, b"\x78": 9 / 2, b"\xf8": 500}}
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
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=[],
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    states = [SlaterDeterminant.from_bytes(b"\x78")]
    basis.add_states(states)

    op_dict = build_operator_dict(basis, operator)
    correct = {
        SlaterDeterminant.from_bytes(b"\x78"): {
            SlaterDeterminant.from_bytes(b"\xf0"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\x78"): 9 / 2,
            SlaterDeterminant.from_bytes(b"\xf8"): 500,
        }
    }
    # correct = {b"\x78": {b"\xf0": -1 / 2, b"\x78": 9 / 2, b"\xf8": 500}}
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
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=[],
        verbose=True,
        comm=None,
    )
    states = [
        SlaterDeterminant.from_bytes(b"\x78"),
        SlaterDeterminant.from_bytes(b"\xb8"),
        SlaterDeterminant.from_bytes(b"\xd8"),
        SlaterDeterminant.from_bytes(b"\xe8"),
        SlaterDeterminant.from_bytes(b"\xf0"),
    ]
    basis.add_states(states)

    op_dict = build_operator_dict(basis, operator)
    correct = {
        SlaterDeterminant.from_bytes(b"\x78"): {
            SlaterDeterminant.from_bytes(b"\xf0"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\x78"): 9 / 2,
        },
        SlaterDeterminant.from_bytes(b"\xb8"): {SlaterDeterminant.from_bytes(b"\xb8"): 9 / 2},
        SlaterDeterminant.from_bytes(b"\xd8"): {SlaterDeterminant.from_bytes(b"\xd8"): 4},
        SlaterDeterminant.from_bytes(b"\xe8"): {SlaterDeterminant.from_bytes(b"\xe8"): 9 / 2},
        SlaterDeterminant.from_bytes(b"\xf0"): {
            SlaterDeterminant.from_bytes(b"\x78"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\xf0"): 9 / 2,
        },
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
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=[],
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    states = [
        SlaterDeterminant.from_bytes(b"\x78"),
        SlaterDeterminant.from_bytes(b"\xb8"),
        SlaterDeterminant.from_bytes(b"\xd8"),
        SlaterDeterminant.from_bytes(b"\xe8"),
        SlaterDeterminant.from_bytes(b"\xf0"),
    ]
    basis.add_states(states)

    op_dict = build_operator_dict(basis, operator)
    correct = {
        SlaterDeterminant.from_bytes(b"\x78"): {
            SlaterDeterminant.from_bytes(b"\xf0"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\x78"): 9 / 2,
        },
        SlaterDeterminant.from_bytes(b"\xb8"): {SlaterDeterminant.from_bytes(b"\xb8"): 9 / 2},
        SlaterDeterminant.from_bytes(b"\xd8"): {SlaterDeterminant.from_bytes(b"\xd8"): 4},
        SlaterDeterminant.from_bytes(b"\xe8"): {SlaterDeterminant.from_bytes(b"\xe8"): 9 / 2},
        SlaterDeterminant.from_bytes(b"\xf0"): {
            SlaterDeterminant.from_bytes(b"\x78"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\xf0"): 9 / 2,
        },
    }
    # correct = {
    #     b"\x78": {b"\xf0": -1 / 2, b"\x78": 9 / 2},
    #     b"\xb8": {b"\xb8": 9 / 2},
    #     b"\xd8": {b"\xd8": 4},
    #     b"\xe8": {b"\xe8": 9 / 2},
    #     b"\xf0": {b"\x78": -1 / 2, b"\xf0": 9 / 2},
    # }
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
    # states = [b"\x78", b"\xb8", b"\xd8", b"\xe8", b"\xf0"]
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=[],
        verbose=True,
        comm=None,
    )
    states = [
        SlaterDeterminant.from_bytes(b"\x78"),
        SlaterDeterminant.from_bytes(b"\xb8"),
        SlaterDeterminant.from_bytes(b"\xd8"),
        SlaterDeterminant.from_bytes(b"\xe8"),
        SlaterDeterminant.from_bytes(b"\xf0"),
    ]
    basis.add_states(states)

    op_dict = build_operator_dict(basis, operator)
    correct = {
        SlaterDeterminant.from_bytes(b"\x78"): {
            SlaterDeterminant.from_bytes(b"\xf0"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\x78"): 9 / 2,
            SlaterDeterminant.from_bytes(b"\xf8"): 500,
        },
        SlaterDeterminant.from_bytes(b"\xb8"): {SlaterDeterminant.from_bytes(b"\xb8"): 9 / 2},
        SlaterDeterminant.from_bytes(b"\xd8"): {SlaterDeterminant.from_bytes(b"\xd8"): 4},
        SlaterDeterminant.from_bytes(b"\xe8"): {SlaterDeterminant.from_bytes(b"\xe8"): 9 / 2},
        SlaterDeterminant.from_bytes(b"\xf0"): {
            SlaterDeterminant.from_bytes(b"\x78"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\xf0"): 9 / 2,
        },
    }
    # correct = {
    #     b"\x78": {b"\xf0": -1 / 2, b"\x78": 9 / 2, b"\xf8": 500},
    #     b"\xb8": {b"\xb8": 9 / 2},
    #     b"\xd8": {b"\xd8": 4},
    #     b"\xe8": {b"\xe8": 9 / 2},
    #     b"\xf0": {b"\x78": -1 / 2, b"\xf0": 9 / 2},
    # }

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
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=[],
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    states = [
        SlaterDeterminant.from_bytes(b"\x78"),
        SlaterDeterminant.from_bytes(b"\xb8"),
        SlaterDeterminant.from_bytes(b"\xd8"),
        SlaterDeterminant.from_bytes(b"\xe8"),
        SlaterDeterminant.from_bytes(b"\xf0"),
    ]
    basis.add_states(states)

    op_dict = build_operator_dict(basis, operator)
    correct = {
        SlaterDeterminant.from_bytes(b"\x78"): {
            SlaterDeterminant.from_bytes(b"\xf0"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\x78"): 9 / 2,
            SlaterDeterminant.from_bytes(b"\xf8"): 500,
        },
        SlaterDeterminant.from_bytes(b"\xb8"): {SlaterDeterminant.from_bytes(b"\xb8"): 9 / 2},
        SlaterDeterminant.from_bytes(b"\xd8"): {SlaterDeterminant.from_bytes(b"\xd8"): 4},
        SlaterDeterminant.from_bytes(b"\xe8"): {SlaterDeterminant.from_bytes(b"\xe8"): 9 / 2},
        SlaterDeterminant.from_bytes(b"\xf0"): {
            SlaterDeterminant.from_bytes(b"\x78"): -1 / 2,
            SlaterDeterminant.from_bytes(b"\xf0"): 9 / 2,
        },
    }
    # correct = {
    #     b"\x78": {b"\xf0": -1 / 2, b"\x78": 9 / 2, b"\xf8": 500},
    #     b"\xb8": {b"\xb8": 9 / 2},
    #     b"\xd8": {b"\xd8": 4},
    #     b"\xe8": {b"\xe8": 9 / 2},
    #     b"\xf0": {b"\x78": -1 / 2, b"\xf0": 9 / 2},
    # }
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
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=[],
        verbose=True,
        comm=None,
    )
    states = [SlaterDeterminant.from_bytes(b"\x78")]
    basis.add_states(states)

    dense_mat = build_dense_matrix(basis, operator)
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
    states = build_states([b"\x78"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    dense_mat = build_dense_matrix(basis, operator)
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
    states = build_states([b"\x78", b"\xb8", b"\xd8", b"\xe8", b"\xf0"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    dense_mat = build_dense_matrix(basis, operator)
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
    states = build_states([b"\x78", b"\xb8", b"\xd8", b"\xe8", b"\xf0"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    dense_mat = build_dense_matrix(basis, operator)
    assert dense_mat.shape == (5, 5)
    # assert np.allclose(
    #     dense_mat,
    #     np.array(
    #         [
    #             [9 / 2, 0, 0, 0, -1 / 2],
    #             [0, 9 / 2, 0, 0, 0],
    #             [0, 0, 4, 0, 0],
    #             [0, 0, 0, 9 / 2, 0],
    #             [-1 / 2, 0, 0, 0, 9 / 2],
    #         ],
    #         dtype=float,
    #     ),
    # ), f"{dense_mat=}"


def test_simple_vector():
    states = build_states([b"\x00\x1a\x2b", b"\xff\x00\x1a"])
    basis = Basis(
        impurity_orbitals={0: [list(range(24))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=None,
    )
    state = {}
    if states[0] in basis._index_dict:
        state[states[0]] = 0.25 + 0.2j
    if states[1] in basis._index_dict:
        state[states[1]] = 0.33 + 0.15j

    v = build_vector(basis, [state])[0]
    v_exact = np.array([0.25 + 0.2j, 0.33 + 0.15j], dtype=complex)

    assert v.shape == (len(basis),)
    assert v.shape == v_exact.shape
    assert np.allclose(v, v_exact), f"{v=} ~= {v_exact}"


@pytest.mark.mpi
def test_simple_vector_mpi():
    states = build_states([b"\x00\x1a\x2b", b"\xff\x00\x1a"])
    basis = Basis(
        impurity_orbitals={0: [list(range(24))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    state = {}
    if states[0] in basis.local_basis:
        state[states[0]] = 0.25 + 0.2j
    if states[1] in basis.local_basis:
        state[states[1]] = 0.33 + 0.15j

    v = build_vector(basis, [state])[0]
    v_exact = np.array([0.25 + 0.2j, 0.33 + 0.15j], dtype=complex)

    assert v.shape == (len(basis),)
    assert v.shape == v_exact.shape
    # assert np.allclose(v, v_exact)


def test_vector():
    states = build_states([b"\x78", b"\xb8", b"\xd8", b"\xe8", b"\xf0"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states[:-1],
        verbose=True,
        comm=None,
    )
    state = {states[-1]: 1 + 1j}
    state[states[0]] = 0.25 + 0.2j
    if states[1] in basis._index_dict:
        state[states[1]] = 0.33 + 0.15j
    v = build_vector(basis, [state])[0]
    v_exact = np.array([0.25 + 0.2j, 0.33 + 0.15j, 0, 0], dtype=complex)

    assert v.shape == (len(basis),)
    assert v.shape == v_exact.shape
    assert np.allclose(v, v_exact)


@pytest.mark.mpi
def test_vector_mpi():
    comm = MPI.COMM_WORLD
    states = build_states([b"\x78", b"\xb8", b"\xd8", b"\xe8", b"\xf0"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states[:-1],
        verbose=True,
        comm=comm,
    )
    state = {states[-1]: 1 + 1j}
    state[states[0]] = 0.25 + 0.2j
    if states[1] in basis.local_basis:
        state[states[1]] = 0.33 + 0.15j
    v = build_vector(basis, [state])[0]
    v_exact = np.array([0.25 + 0.2j, 0.33 + 0.15j, 0, 0], dtype=complex)

    assert v.shape == (len(basis),)
    assert v.shape == v_exact.shape
    # assert np.allclose(v, v_exact)


def test_simple_state():
    states = build_states([b"\x00\x1a\x2b", b"\xff\x00\x1a"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    v = np.array([[1.0, -2.5]])
    s = build_state(basis, v)
    s_exact = [{states[0]: v[0, 0], states[1]: v[0, 1]}]

    for i in range(len(s)):
        assert all(s[i][state] == s_exact[i][state] for state in s[i])


@pytest.mark.mpi
def test_simple_state_mpi():
    states = build_states([b"\x2b", b"\x1a"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    # v = np.array([[1.0, -2.5]])
    v = np.zeros([1, len(states)])
    v[0, basis.index(states[0])] = 1.0
    v[0, basis.index(states[1])] = 2.5
    s = build_state(basis, v)
    s_exact = [{states[0]: v[0, basis.index(states[0])], states[1]: v[0, basis.index(states[1])]}]

    for i in range(len(s)):
        assert all(s[i][state] == s_exact[i][state] for state in basis.local_basis), f"{s=} {s_exact=}"


def test_state_mpi():
    comm = None
    states = build_states([b"\x78", b"\xb8", b"\xd8", b"\xe8", b"\xf0"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=comm,
    )
    v = np.array([[1.0, -2.5, 0, 0, 1.2], [0, 3, 1, 0, 0]])
    s = build_state(basis, v)
    s_exact = [
        {states[0]: v[0, 0], states[1]: v[0, 1], states[4]: v[0, 4]},
        {states[1]: v[1, 1], states[2]: v[1, 2]},
    ]

    for i in range(len(s)):
        assert all(s[i][state] == s_exact[i][state] for state in s[i])


@pytest.mark.mpi
def test_state_mpi():
    comm = MPI.COMM_WORLD
    states = build_states([b"\x78", b"\xb8", b"\xd8", b"\xe8", b"\xf0"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=comm,
    )
    v = np.array([[1.0, -2.5, 0, 0, 1.2], [0, 3, 1, 0, 0]])
    build_state(basis, v)
    state_indices = list(basis.index(states))
    [
        {states[state_indices[0]]: v[0, 0], states[state_indices[1]]: v[0, 1], states[state_indices[4]]: v[0, 4]},
        {states[state_indices[1]]: v[1, 1], states[state_indices[2]]: v[1, 2]},
    ]

    # for i in range(len(s)):
    #     assert all(s[i][state] == s_exact[i][state] for state in s[i])


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
    states = build_states([b"\x80\x00", b"\x40\x00"])
    basis = Basis(
        impurity_orbitals={0: [list(range(10))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=None,
    )

    basis.expand(Hop)
    # expect 10000  01000  00001  00000  00000  00000
    #        00000  00000  00000  10000  01000  00001

    expected = build_states([b"\x80\x00", b"\x40\x00", b"\x08\x00"])  # , b"\x04\x00", b"\x02\x00", b"\x00\x40"]
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
    states = build_states([b"\x80\x00", b"\x40\x00"])
    basis = Basis(
        impurity_orbitals={2: [list(range(10))]},
        bath_states=(
            {2: [[]]},
            {2: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )

    basis.expand(Hop)
    # expect 10000  01000  00001  00000  00000  00000
    #        00000  00000  00000  10000  01000  00001

    expected = build_states([b"\x80\x00", b"\x40\x00", b"\x08\x00"])  # , b"\x04\x00", b"\x02\x00", b"\x00\x40"]
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
    states = build_states([b"\x80\x00"])
    from impurityModel.ed.cipsi_solver import CIPSISolver

    basis = Basis(
        impurity_orbitals={2: [list(range(10))]},
        bath_states=(
            {2: [[]]},
            {2: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=None,
    )
    solver = CIPSISolver(basis)
    solver.truncate_initial(Hop)

    solver.expand(Hop)

    expected = build_states([b"\x80\x00", b"\x08\x00"])
    assert all(state in basis for state in expected), f"{expected=} {list(basis)=}"
    assert len(basis) == 6, f"Expected full multiplet of 6 states, got {len(basis)}: {list(basis)}"


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
    states = build_states([b"\x80\x00"])
    from impurityModel.ed.cipsi_solver import CIPSISolver

    basis = Basis(
        impurity_orbitals={2: [list(range(10))]},
        bath_states=(
            {2: [[]]},
            {2: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    solver = CIPSISolver(basis)
    solver.truncate_initial(Hop)

    solver.expand(Hop)
    # expect 10000  00001  00000  00000
    #        00000  00000  10000  00001

    expected = build_states([b"\x80\x00", b"\x08\x00"])
    assert all(state in basis for state in expected), f"{expected=} {list(basis)=}"
    assert len(basis) == 6, f"Expected full multiplet of 6 states, got {len(basis)}: {list(basis)}"


@pytest.mark.mpi
def test_distributed_simple_vector():
    states = build_states((b"\x00\x1a\x2b", b"\xff\x00\x1a"))
    basis = Basis(
        impurity_orbitals={2: [list(range(24))]},
        bath_states=(
            {2: [[]]},
            {2: [[]]},
        ),
        initial_basis=states,
        verbose=True,
        comm=MPI.COMM_WORLD,
    )
    state = {}
    if states[0] in basis.local_basis:
        state[states[0]] = 0.25 + 0.2j
    if states[1] in basis.local_basis:
        state[states[1]] = 0.33 + 0.15j

    v = build_distributed_vector(basis, [state])[0]
    v_exact = np.zeros((len(basis.local_basis),), dtype=complex)
    if states[0] in basis.local_basis:
        v_exact[0] = 0.25 + 0.2j
    if states[1] in basis.local_basis:
        v_exact[-1] = 0.33 + 0.15j

    assert v.shape == (len(basis.local_basis),)
    assert v.shape == v_exact.shape
    # assert np.allclose(v, v_exact)


@pytest.mark.mpi
def test_distributed_vector_mpi():
    comm = MPI.COMM_WORLD
    states = build_states([b"\x78", b"\xb8", b"\xd8", b"\xe8", b"\xf0"])
    basis = Basis(
        impurity_orbitals={0: [list(range(5))]},
        bath_states=(
            {0: [[]]},
            {0: [[]]},
        ),
        initial_basis=states[:-1],
        verbose=True,
        comm=comm,
    )
    state = {states[-1]: 1 + 1j}
    state[states[0]] = 0.25 + 0.2j
    if states[1] in basis.local_basis:
        state[states[1]] = 0.33 + 0.15j
    v = build_distributed_vector(basis, [state])[0]
    np.array([0.25 * comm.size + 0.2j * comm.size, 0.33 + 0.15j, 0, 0], dtype=complex)

    len(basis.local_basis)
    assert v.shape == (len(basis.local_basis),)
    end = basis.index_bounds[comm.rank]
    if end is None:
        end = len(basis)
    # if n > 0:
    # assert np.allclose(v, v_exact[end - n : end])


@pytest.mark.mpi
def test_two_sets_of_imp_orbs():
    # Several impurity groups form one correlated shell: only the *total* impurity
    # occupation (3 + 4 = 7) is pinned, and charge redistributes freely between the
    # manifolds (a per-group pin would fix the eg/t2g ratio or S_z). With zero slack
    # the valence baths stay full and the conduction baths empty, so the seed holds
    # every arrangement of 7 electrons over the manifolds of sizes 3 and 5:
    # (n0, n1) = (2, 5) -> C(3,2) * C(5,5) = 3 and (3, 4) -> C(3,3) * C(5,4) = 5.
    comm = MPI.COMM_WORLD
    impurity_orbitals = {0: [list(range(3))], 1: [list(range(3, 8))]}
    bath_states = ({0: [[]], 1: [list(range(8, 13))]}, {0: [[]], 1: [list(range(13, 15))]})
    nominal_impurity_occ = {0: 3, 1: 4}
    delta_impurity_occ = {0: 0, 1: 0}
    delta_valence_occ = {0: 0, 1: 0}
    delta_conduction_occ = {0: 0, 1: 0}

    basis = Basis(
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        nominal_impurity_occ=nominal_impurity_occ,
        delta_valence_occ=delta_valence_occ,
        delta_conduction_occ=delta_conduction_occ,
        delta_impurity_occ=delta_impurity_occ,
        verbose=True,
        comm=comm,
    )
    assert len(basis) == 8

    restrictions = get_effective_restrictions(basis)
    impurity_indices = frozenset(range(8))
    valence_indices = frozenset(range(8, 13))
    conduction_indices = frozenset(range(13, 15))
    assert restrictions[impurity_indices] == (7, 7)
    assert restrictions[valence_indices] == (5, 5)
    assert restrictions[conduction_indices] == (0, 0)
