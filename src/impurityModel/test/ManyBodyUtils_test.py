import pytest
import numpy as np
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator
from math import isclose


def test_ManyBodyState():
    d = {b"\xff\xac": 1.0, b"\xac\xff": 1j}
    psi = ManyBodyState(d)
    for state in d:
        assert d[state] == psi[state]
    for state in psi:
        assert d[state] == psi[state]

    assert isclose(psi.norm2(), 2)
    assert isclose(psi.norm(), np.sqrt(2))


def test_ManyBodyState_2():
    d = {b"\xff\xac": 1.0, b"\xac\xff": 1j}
    psi = ManyBodyState()
    for state, amp in d.items():
        psi[state] = amp
    for state in d:
        assert d[state] == psi[state]
    for state in psi:
        assert d[state] == psi[state]

    assert isclose(psi.norm2(), 2)
    assert isclose(psi.norm(), np.sqrt(2))


def test_ManyBodyState_prune():
    d = {b"\xff\xac": 1.0, b"\xac\xff": 2j}
    psi = ManyBodyState(d)
    for state in d:
        assert d[state] == psi[state]
    for state in psi:
        assert d[state] == psi[state]

    psi.prune(1)
    assert b"\xff\xac" not in psi
    assert psi[b"\xac\xff"] == 2j

    assert isclose(psi.norm2(), 4)
    assert isclose(psi.norm(), np.sqrt(4))


def test_ManyBodyState_arithmetic():
    add = {b"\xff\xac": 1.0, b"\xac\xff": 2j}
    a = ManyBodyState({b"\xff\xac": 1.0})
    b = ManyBodyState({b"\xac\xff": 2j})

    psi = a + b
    for state in add:
        assert add[state] == psi[state]
    for state in psi:
        assert add[state] == psi[state]

    sub = {b"\xff\xac": 1.0, b"\xac\xff": -2j}
    psi = a - b
    for state in sub:
        assert sub[state] == psi[state]
    for state in psi:
        assert sub[state] == psi[state]

    scale = {b"\xff\xac": 2.5, b"\xac\xff": 5.0j}
    psi = (a + b) * 2.5
    for state in scale:
        assert scale[state] == psi[state]
    for state in scale:
        assert scale[state] == psi[state]

    psi = 2.5 * (a + b)
    for state in scale:
        assert scale[state] == psi[state]
    for state in scale:
        assert scale[state] == psi[state]

    psi = (a + b) / 0.4
    for state in scale:
        assert scale[state] == psi[state]
    for state in scale:
        assert scale[state] == psi[state]

def test_ManyBodyState_pickle():
    import pickle
    psi = ManyBodyState({b"\xa0": 1.0, b"\xbf": 1.0j})
    pickled_psi = pickle.dumps(psi)
    new_psi = pickle.loads(pickled_psi)
    assert psi == new_psi


def test_ManyBodyOperator():
    d = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    op = ManyBodyOperator(d)

    for process in d:
        assert d[process] == op[process]
    for process in op:
        assert d[process] == op[process]


def test_ManyBodyOperator_2():
    d = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    op = ManyBodyOperator()

    for process, amp in d.items():
        op[process] = amp

    for process in d:
        assert d[process] == op[process]
    for process in op:
        assert d[process] == op[process]


def test_ManyBodyOperator_arithmetic():
    add = {((1, "c"),): 1.0, ((0, "a"),): 1j}
    a = ManyBodyOperator({((1, "c"),): 1.0})
    b = ManyBodyOperator({((0, "a"),): 1j})

    op = a + b
    for state in add:
        assert add[state] == op[state]
    for state in op:
        assert add[state] == op[state]

    sub = {((1, "c"),): 1.0, ((0, "a"),): -1j}
    op = a - b
    for state in sub:
        assert sub[state] == op[state]
    for state in op:
        assert sub[state] == op[state]

    scale = {((1, "c"),): 2.5, ((0, "a"),): 2.5j}
    op = (a + b) * 2.5
    for state in scale:
        assert scale[state] == op[state]
    for state in scale:
        assert scale[state] == op[state]

    op = 2.5 * (a + b)
    for state in scale:
        assert scale[state] == op[state]
    for state in scale:
        assert scale[state] == op[state]

    op = (a + b) / 0.4
    for state in scale:
        assert scale[state] == op[state]
    for state in scale:
        assert scale[state] == op[state]


def test_ManyBodyOperator_apply():
    #                      1010          1011
    psi = ManyBodyState({b"\xa0": 1.0, b"\xbf": 1.0j})

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      0011
    res = ManyBodyState({b"\x30": 1.0, b"\xa0": 1.0j, b"\xbf": -1.0})

    assert res == op(psi, 0)


def test_ManyBodyOperator_pickle():
    import pickle

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    pickled_op = pickle.dumps(op)
    new_op = pickle.loads(pickled_op)

    assert op == new_op
