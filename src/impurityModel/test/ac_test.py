import pytest
from impurityModel.ed import create
from impurityModel.ed import remove
from impurityModel.ed import ac
import impurityModel.ed.product_state_representation as psr


@pytest.mark.parametrize(
    "s, i, expected_state, expected_sign",
    [
        ("11000100", 0, "11000100", 0),
        ("11000100", 2, "11100100", 1),
        ("1100010001100", 8, "1100010011100", -1),
        ("1100010001100", 11, "1100010001110", -1),
    ],
)
def test_create(s, i, expected_state, expected_sign):
    expected_state = psr.str2bytes(expected_state)
    n_orb = len(s)
    b = psr.str2bytes(s)

    old = create.ubytes(n_orb, i, b)
    new = ac.create(n_orb, i, b)
    print(flush=True)
    assert old == new
    assert old == (expected_state, expected_sign)
    assert new == (expected_state, expected_sign)


@pytest.mark.parametrize(
    "s, i, expected_state, expected_sign",
    [
        ("11000100", 0, "01000100", 1),
        ("11000100", 2, "11000100", 0),
        ("1100010001100", 9, "1100010000100", -1),
        ("1100010001100", 10, "1100010001000", 1),
    ],
)
def test_annihilate(s, i, expected_state, expected_sign):
    expected_state = psr.str2bytes(expected_state)
    n_orb = len(s)
    b = psr.str2bytes(s)

    old = remove.ubytes(n_orb, i, b)
    new = ac.annihilate(n_orb, i, b)
    print(flush=True)
    assert old == new
    assert old == (expected_state, expected_sign)
    assert new == (expected_state, expected_sign)


@pytest.mark.parametrize(
    "s, op, expected",
    [
        ("11000100", {((2, "c"),): 1.0}, {psr.str2bytes("11100100"): 1.0}),
        ("11000100", {((2, "c"), (2, "a")): 1.0}, dict()),
        ("01111001", {((2, "c"), (2, "a")): 1}, {psr.str2bytes("01111001"): 1}),
        ("1110010001", {((9, "c"), (2, "c"), (2, "a"), (9, "a")): 0.5}, {psr.str2bytes("1110010001"): 0.5}),
        ("1110010001", {((8, "c"), (2, "c"), (2, "a"), (9, "a")): 0.5}, {psr.str2bytes("1110010010"): 0.5}),
        ("1110010001", {((3, "c"), (2, "c"), (2, "a"), (9, "a")): 0.5}, {psr.str2bytes("1111010000"): -0.5}),
        (
            "1110010001",
            {((3, "c"),): 0.5, ((9, "a"),): 0.5},
            {psr.str2bytes("1111010001"): -0.5, psr.str2bytes("1110010000"): 0.5},
        ),
    ],
)
def test_apply_to_state(s, op, expected):
    n_orb = len(s)
    state = psr.str2bytes(s)
    test = ac.apply_to_state(n_orb, op, state)
    print(f"{test=}")
    assert test.keys() == expected.keys()
    for state in test:
        assert test[state] == expected[state]


@pytest.mark.parametrize(
    "psi_in, op, expected",
    [
        ({psr.str2bytes("01011001"): 1}, {((2, "c"),): 1}, {psr.str2bytes("01111001"): -1}),
        ({psr.str2bytes("01111001"): 1}, {((2, "c"),): 1}, dict()),
        ({psr.str2bytes("01111001"): 1}, {((2, "c"), (2, "a")): 1}, {psr.str2bytes("01111001"): 1}),
        (
            {psr.str2bytes("01111001"): 1, psr.str2bytes("01011001"): 1},
            {((2, "c"), (2, "a")): 1},
            {psr.str2bytes("01111001"): 1},
        ),
        (
            {psr.str2bytes("01111001"): 0.5, psr.str2bytes("01011011"): 0.5},
            {((2, "c"), (2, "a")): 1, ((2, "c"), (6, "a")): 1},
            {psr.str2bytes("01111001"): 1},
        ),
    ],
)
def test_applyOp(psi_in, op, expected):
    n_orb = len(next(iter(psi_in.keys())))
    psi_new = ac.applyOp(n_orb, op, psi_in)
    print(f"{psi_new=}", flush=True)
    assert psi_new.keys() == expected.keys()
    for key in psi_new:
        assert psi_new[key] == expected[key]
