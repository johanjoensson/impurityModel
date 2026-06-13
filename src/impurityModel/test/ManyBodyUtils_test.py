import pytest
import numpy as np
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    ManyBodyOperator,
    SlaterDeterminant,
    inner,
)
from math import isclose


def all_isclose(dict1, dict2, **kwargs):
    return all(pytest.approx(dict1[key]) == dict2[key] for key in dict1.keys()) and all(
        pytest.approx(dict1[key]) == dict2[key] for key in dict2.keys()
    )


def test_ManyBodyState():
    d = {
        SlaterDeterminant.from_bytes(b"\xff\xac"): 1.0,
        SlaterDeterminant.from_bytes(b"\xac\xff"): 1j,
    }
    for sd in d.keys():
        for chunk in sd:
            print(f"{chunk.to_bytes(8)}", end=" ")
        print()
    psi = ManyBodyState(d)
    for state in d:
        assert d[state] == psi[state]
    # for state in psi:
    #     assert d[state] == psi[state]

    assert pytest.approx(psi.norm2()) == 2
    assert pytest.approx(psi.norm()) == np.sqrt(2)


def test_ManyBodyState_2():
    d = {
        SlaterDeterminant.from_bytes(b"\xff\xac"): 1.0,
        SlaterDeterminant.from_bytes(b"\xac\xff"): 1j,
    }
    psi = ManyBodyState()
    for state, amp in d.items():
        psi[state] = amp
    for state in d:
        assert d[state] == psi[state]
    # for state in psi:
    #     assert d[state] == psi[state]

    assert pytest.approx(psi.norm2()) == 2
    assert pytest.approx(psi.norm()) == np.sqrt(2)


def test_ManyBodyState_prune():
    d = {SlaterDeterminant.from_bytes(b"\xff\xac"): 1.0, SlaterDeterminant.from_bytes(b"\xac\xff"): 2j}
    psi = ManyBodyState(d)
    for state in d:
        assert d[state] == psi[state]
    # for state in psi:
    #     assert d[state] == psi[state]

    print(f"{psi=}")
    psi.prune(1)
    assert SlaterDeterminant.from_bytes(b"\xff\xac") not in psi
    assert psi[SlaterDeterminant.from_bytes(b"\xac\xff")] == 2j

    assert pytest.approx(psi.norm2()) == 4
    assert pytest.approx(psi.norm()) == np.sqrt(4)


def test_ManyBodyState_arithmetic():
    add = ManyBodyState({SlaterDeterminant.from_bytes(b"\xff\xac"): 1.0, SlaterDeterminant.from_bytes(b"\xac\xff"): 2j})
    a = ManyBodyState({SlaterDeterminant.from_bytes(b"\xff\xac"): 1.0})
    b = ManyBodyState({SlaterDeterminant.from_bytes(b"\xac\xff"): 2j})

    psi = a + b
    for state in add:
        assert pytest.approx(add[state]) == psi[state]
    # for state in psi:
    #     assert add[state] == psi[state]

    sub = ManyBodyState(
        {SlaterDeterminant.from_bytes(b"\xff\xac"): 1.0, SlaterDeterminant.from_bytes(b"\xac\xff"): -2j}
    )
    psi = a - b
    for state in sub:
        assert pytest.approx(sub[state]) == psi[state], f"{sub=}, {psi=}"
    # for state in psi:
    #     assert sub[state] == psi[state]

    scale = ManyBodyState(
        {SlaterDeterminant.from_bytes(b"\xff\xac"): 2.5, SlaterDeterminant.from_bytes(b"\xac\xff"): 5.0j}
    )
    psi = (a + b) * 2.5
    for state in scale:
        assert pytest.approx(scale[state]) == psi[state]
    for state in scale:
        assert pytest.approx(scale[state]) == psi[state]

    psi = 2.5 * (a + b)
    for state in scale:
        assert pytest.approx(scale[state]) == psi[state]
    for state in scale:
        assert pytest.approx(scale[state]) == psi[state]

    psi = (a + b) / 0.4
    for state in scale:
        assert pytest.approx(scale[state]) == psi[state]
    for state in scale:
        assert pytest.approx(scale[state]) == psi[state]


def test_ManyBodyState_pickle():
    import pickle

    psi = ManyBodyState({SlaterDeterminant.from_bytes(b"\xa0"): 1.0, SlaterDeterminant.from_bytes(b"\xbf"): 1.0j})
    pickled_psi = pickle.dumps(psi)
    new_psi = pickle.loads(pickled_psi)
    assert psi == new_psi


def test_ManyBodyOperator():
    d = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    op = ManyBodyOperator(d)

    for process in d:
        assert pytest.approx(d[process]) == op[process]
    for process in op:
        assert pytest.approx(d[process]) == op[process]


def test_ManyBodyOperator_2():
    d = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    op = ManyBodyOperator()

    for process, amp in d.items():
        op[process] = amp

    for process in d:
        assert pytest.approx(d[process]) == op[process]
    for process in op:
        assert pytest.approx(d[process]) == op[process]


def test_ManyBodyOperator_arithmetic():
    add = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    a = ManyBodyOperator({((1, "c"),): 1.0})
    b = ManyBodyOperator({((0, "a"),): 1j})

    op = a + b
    for state in add:
        assert pytest.approx(add[state]) == op[state]
    for state in op:
        assert pytest.approx(add[state]) == op[state]

    sub = {((1, "c"),): 1.0, ((0, "a"),): -1j}
    op = a - b
    for state in sub:
        assert pytest.approx(sub[state]) == op[state]
    for state in op:
        assert pytest.approx(sub[state]) == op[state]

    scale = {((1, "c"),): 2.5, ((0, "a"),): 2.5j}
    op = (a + b) * 2.5
    for state in scale:
        assert pytest.approx(scale[state]) == op[state]
    for state in scale:
        assert pytest.approx(scale[state]) == op[state]

    op = 2.5 * (a + b)
    for state in scale:
        assert pytest.approx(scale[state]) == op[state]
    for state in scale:
        assert pytest.approx(scale[state]) == op[state]

    op = (a + b) / 0.4
    for state in scale:
        assert pytest.approx(scale[state]) == op[state]
    for state in scale:
        assert pytest.approx(scale[state]) == op[state]


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


def test_ManyBodyState_inplace_operators():
    psi1 = ManyBodyState({SlaterDeterminant.from_bytes(b"\x01"): 1.0})
    psi2 = ManyBodyState({SlaterDeterminant.from_bytes(b"\x02"): 2j})
    
    psi1 += psi2
    assert psi1[SlaterDeterminant.from_bytes(b"\x01")] == 1.0
    assert psi1[SlaterDeterminant.from_bytes(b"\x02")] == 2j
    
    psi1 -= psi2
    assert psi1[SlaterDeterminant.from_bytes(b"\x01")] == 1.0
    assert SlaterDeterminant.from_bytes(b"\x02") not in psi1 or psi1[SlaterDeterminant.from_bytes(b"\x02")] == 0.0


def test_ManyBodyState_erase():
    psi = ManyBodyState({SlaterDeterminant.from_bytes(b"\x01"): 1.0, SlaterDeterminant.from_bytes(b"\x02"): 2.0})
    assert len(psi) == 2
    psi.erase(SlaterDeterminant.from_bytes(b"\x01"))
    assert len(psi) == 1
    assert SlaterDeterminant.from_bytes(b"\x01") not in psi


def test_inner_product():
    psi1 = ManyBodyState({SlaterDeterminant.from_bytes(b"\x01"): 1.0, SlaterDeterminant.from_bytes(b"\x02"): 2.0j})
    psi2 = ManyBodyState({SlaterDeterminant.from_bytes(b"\x01"): 2.0, SlaterDeterminant.from_bytes(b"\x02"): 1.0j})
    # Inner product: conj(1.0)*2.0 + conj(2.0j)*1.0j = 2.0 + (-2j)*j = 2.0 + 2.0 = 4.0
    assert pytest.approx(inner(psi1, psi2)) == 4.0


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




def test_ManyBodyState_extra():
    sd1 = SlaterDeterminant.from_bytes(b"\x01")
    sd2 = SlaterDeterminant.from_bytes(b"\x02")
    d = {sd1: 1.0, sd2: 2.0j}
    psi = ManyBodyState(d)

    # test __contains__
    assert sd1 in psi
    assert SlaterDeterminant.from_bytes(b"\x03") not in psi

    # test keys(), values(), items(), to_dict()
    assert set(psi.keys()) == {sd1, sd2}
    assert set(psi.values()) == {1.0, 2.0j}
    assert dict(psi.items()) == d
    assert psi.to_dict() == d

    # test copy()
    psi_copy = psi.copy()
    assert psi == psi_copy

    # test get()
    assert psi.get(sd1) == 1.0
    assert psi.get(SlaterDeterminant.from_bytes(b"\x03"), 4.0) == 4.0

    # test size() and max_size()
    assert psi.size() == 2
    assert psi.max_size() >= 2


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

