import numpy as np
import pytest

from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
    SlaterDeterminant,
)


def all_isclose(dict1, dict2, **kwargs):
    return all(pytest.approx(dict1[key][0]) == dict2[key][0] for key in dict1.keys()) and all(
        pytest.approx(dict1[key][0]) == dict2[key][0] for key in dict2.keys()
    )


def test_ManyBodyOperator():
    d = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    op = ManyBodyOperator(d)

    for process, value in d.items():
        assert pytest.approx(value) == op[process]
    for process in op:
        assert pytest.approx(d[process]) == op[process]


def test_ManyBodyOperator_2():
    d = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    op = ManyBodyOperator()

    for process, amp in d.items():
        op[process] = amp

    for process, value in d.items():
        assert pytest.approx(value) == op[process]
    for process in op:
        assert pytest.approx(d[process]) == op[process]


def test_ManyBodyOperator_arithmetic():
    add = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    a = ManyBodyOperator({((1, "c"),): 1.0})
    b = ManyBodyOperator({((0, "a"),): 1j})

    op = a + b
    for state, value in add.items():
        assert pytest.approx(value) == op[state]
    for state in op:
        assert pytest.approx(add[state]) == op[state]

    sub = {((1, "c"),): 1.0, ((0, "a"),): -1j}
    op = a - b
    for state, value in sub.items():
        assert pytest.approx(value) == op[state]
    for state in op:
        assert pytest.approx(sub[state]) == op[state]

    scale = {((1, "c"),): 2.5, ((0, "a"),): 2.5j}
    op = (a + b) * 2.5
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]

    op = 2.5 * (a + b)
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]

    op = (a + b) / 0.4
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]
    for state, value in scale.items():
        assert pytest.approx(value) == op[state]


def test_ManyBodyOperator_apply():
    #                      1010          1011
    psi = ManyBodyState({SlaterDeterminant.from_bytes(b"\xa0\x00"): 1.0, SlaterDeterminant.from_bytes(b"\xbf"): 1.0j})

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      0011          1010           1011
    res = ManyBodyState(
        {
            SlaterDeterminant.from_bytes(b"\x30"): 1.0,
            SlaterDeterminant.from_bytes(b"\xa0"): 1.0j,
            SlaterDeterminant.from_bytes(b"\xbf"): -1.0,
        }
    )

    assert all_isclose(res, op(psi, 0))


def test_ManyBodyOperator_apply2():
    #                      1010          1011           1110
    psi = ManyBodyState(
        {
            SlaterDeterminant.from_bytes(b"\xa0"): 1.0,
            SlaterDeterminant.from_bytes(b"\xbf"): 1.0j,
            SlaterDeterminant.from_bytes(b"\xe0"): 1e-13,
        }
    )

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      0011
    res = ManyBodyState(
        {
            SlaterDeterminant.from_bytes(b"\x30"): 1.0,
            SlaterDeterminant.from_bytes(b"\xa0"): 1.0j,
            SlaterDeterminant.from_bytes(b"\xbf"): -1.0,
        }
    )

    assert all_isclose(res, op(psi, 1e-12))


def test_ManyBodyOperator_apply3():
    #                      1010          1011
    psi = ManyBodyState({SlaterDeterminant.from_bytes(b"\xa0"): 1.0, SlaterDeterminant.from_bytes(b"\xbf"): 1.0j})

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      1010
    res = ManyBodyState({SlaterDeterminant.from_bytes(b"\xa0"): 1.0j})

    op.set_restrictions({frozenset([2, 3]): (1, 1)})
    assert all_isclose(res, op(psi, 0))


def test_ManyBodyOperator_pickle():
    import pickle

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    pickled_op = pickle.dumps(op)
    new_op = pickle.loads(pickled_op)

    assert op == new_op


def test_SlaterDeterminant_operators():
    sd1 = SlaterDeterminant.from_bytes(b"\x01")
    sd2 = SlaterDeterminant.from_bytes(b"\x02")
    sd1_copy = SlaterDeterminant.from_bytes(b"\x01")

    assert sd1 < sd2
    assert sd2 > sd1
    assert sd1 == sd1_copy
    assert sd1 != sd2
    assert hash(sd1) == hash(sd1_copy)
    assert len(sd1) > 0
    assert repr(sd1).startswith("SlaterDeterminant")


def test_SlaterDeterminant_extra():
    sd = SlaterDeterminant.from_bytes(b"\x01\x02")
    assert len(sd) > 0
    # test __getitem__ and __setitem__
    val = sd[0]
    sd[0] = val + 1
    assert sd[0] == val + 1
    # test __iter__
    chunks = list(sd)
    assert len(chunks) == len(sd)
    # test __copy__ and __deepcopy__
    import copy

    sd_copy = copy.copy(sd)
    sd_deepcopy = copy.deepcopy(sd)
    assert sd == sd_copy
    assert sd == sd_deepcopy
    # test to_bytearray
    ba = sd.to_bytearray()
    assert isinstance(ba, bytearray)
    assert ba[0] == 1
    assert ba[1] == 2
    assert ba[7] == 1
    assert all(x == 0 for x in ba[2:7])
    assert all(x == 0 for x in ba[8:])


def test_ManyBodyOperator_extra():
    key1 = ((1, "c"),)
    key2 = ((0, "a"),)
    d = {key1: 1.0, key2: 2.0j}
    op = ManyBodyOperator(d)

    # test __contains__
    assert key1 in op
    assert ((2, "a"),) not in op

    # test keys(), values(), items(), to_dict()
    assert set(op.keys()) == {key1, key2}
    assert set(op.values()) == {1.0, 2.0j}
    assert dict(op.items()) == d
    assert op.to_dict() == d

    # test operator *= and /=
    op1 = ManyBodyOperator(d)
    op1 *= 2.0
    assert op1[key1] == 2.0
    assert op1[key2] == 4.0j

    op1 /= 2.0
    assert op1[key1] == 1.0
    assert op1[key2] == 2.0j

    # test unary -
    op_neg = -op
    assert op_neg[key1] == -1.0
    assert op_neg[key2] == -2.0j

    # test __eq__ and __ne__
    op2 = ManyBodyOperator(d)
    assert op == op2
    assert op != op_neg

    # test size() and len()
    assert op.size() == 2
    assert len(op) == 2

    # test erase
    op.erase(key1)
    assert key1 not in op
    assert op.size() == 1


def get_random_state(num_terms):
    s = ManyBodyState()
    for _ in range(num_terms):
        key = tuple(sorted(np.random.randint(0, 100, size=np.random.randint(1, 5))))
        val = np.random.rand() + 1j * np.random.rand()
        s.add_scaled(ManyBodyState({SlaterDeterminant(key): 1.0}), val)
    return s


def test_apply_multi():
    n_states = 3
    num_terms_op = 10
    num_terms_state = 10

    # Build random many-body operator
    op_dict = {}
    for _ in range(num_terms_op):
        num_c = np.random.randint(1, 3)
        num_a = np.random.randint(1, 3)
        k_c = tuple((int(np.random.randint(0, 50)), "c") for _ in range(num_c))
        k_a = tuple((int(np.random.randint(0, 50)), "a") for _ in range(num_a))
        op_dict[k_c + k_a] = np.random.rand() + 1j * np.random.rand()

    op = ManyBodyOperator(op_dict)

    # Create random states
    states = [get_random_state(num_terms_state) for _ in range(n_states)]

    # 1. Apply multi
    results_multi = op.apply_multi(states)

    # 2. Apply in loop
    results_loop = [op(s) for s in states]

    # Assert equality
    for r_multi, r_loop in zip(results_multi, results_loop):
        assert len(r_multi) == len(r_loop)
        for key in r_multi:
            assert key in r_loop
            np.testing.assert_allclose(r_multi[key][0], r_loop[key][0], atol=1e-12)
