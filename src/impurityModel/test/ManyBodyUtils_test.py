import pytest
import numpy as np
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator
from math import isclose


def test_ManyBodyState():
    d = {
        b"\xff\xac": 1.0,
        b"\xac\xff": 1j,
    }
    psi = ManyBodyState(d)
    # print(f"{psi}")
    for state in d:
        assert d[state] == psi[state]
    # for state in psi:
    #     assert d[state] == psi[state]

    assert isclose(psi.norm2(), 2)
    assert isclose(psi.norm(), np.sqrt(2))


def test_ManyBodyState_2():
    d = {b"\xff\xac": 1.0, b"\xac\xff": 1j}
    psi = ManyBodyState()
    print(f"{psi}", flush=True)
    for state, amp in d.items():
        psi[state] = amp
    print(f"{psi}", flush=True)
    for state in d:
        assert d[state] == psi[state]
    # for state in psi:
    #     assert d[state] == psi[state]

    assert isclose(psi.norm2(), 2)
    assert isclose(psi.norm(), np.sqrt(2))


def test_ManyBodyState_prune():
    d = {b"\xff\xac": 1.0, b"\xac\xff": 2j}
    psi = ManyBodyState(d)
    for state in d:
        assert d[state] == psi[state]
    # for state in psi:
    #     assert d[state] == psi[state]

    psi.prune(1)
    assert b"\xff\xac" not in psi
    assert psi[b"\xac\xff"] == 2j

    assert isclose(psi.norm2(), 4)
    assert isclose(psi.norm(), np.sqrt(4))


def test_ManyBodyState_arithmetic():
    add = ManyBodyState({b"\xff\xac": 1.0, b"\xac\xff": 2j})
    a = ManyBodyState({b"\xff\xac": 1.0})
    b = ManyBodyState({b"\xac\xff": 2j})

    psi = a + b
    for state in add:
        assert add[state] == psi[state]
    # for state in psi:
    #     assert add[state] == psi[state]

    sub = ManyBodyState({b"\xff\xac": 1.0, b"\xac\xff": -2j})
    psi = a - b
    for state in sub:
        assert sub[state] == psi[state], f"{sub=}, {psi=}"
    # for state in psi:
    #     assert sub[state] == psi[state]

    scale = ManyBodyState({b"\xff\xac": 2.5, b"\xac\xff": 5.0j})
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
    psi = ManyBodyState({b"\xa0\x00": 1.0, b"\xbf": 1.0j})

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      0011          1010           1011
    res = ManyBodyState({b"\x30": 1.0, b"\xa0": 1.0j, b"\xbf": -1.0})

    print(f"{psi=}", flush=True)
    print(f"{op(psi, 0)=}", flush=True)
    assert res == op(psi, 0)


def test_ManyBodyOperator_apply2():
    #                      1010          1011           1110
    psi = ManyBodyState({b"\xa0": 1.0, b"\xbf": 1.0j, b"\xe0": 1e-13})

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      0011
    res = ManyBodyState({b"\x30": 1.0, b"\xa0": 1.0j, b"\xbf": -1.0})

    print(f"{op(psi)}", flush=True)
    assert res == op(psi, 1e-12)


def test_ManyBodyOperator_apply3():
    #                      1010          1011
    psi = ManyBodyState({b"\xa0": 1.0, b"\xbf": 1.0j})

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    #                      1010
    res = ManyBodyState({b"\xa0": 1.0j})

    print(f"{op(psi, 0, {frozenset([2, 3]): (1, 1)})}", flush=True)
    assert res == op(psi, 0, {frozenset([2, 3]): (1, 1)})


def test_ManyBodyOperator_pickle():
    import pickle

    op = ManyBodyOperator({((0, "a"), (3, "c")): 1.0, ((1, "a"), (1, "c")): 1.0j})
    pickled_op = pickle.dumps(op)
    new_op = pickle.loads(pickled_op)

    assert op == new_op


# def test_applyOp_create_one_electron_from_vacuum():
#     vacuum_as_string = "000000"
#     n_spin_orbitals = len(vacuum_as_string)
#     # Multi-configurational state is a single product state
#     psi = {psr.str2bytes(vacuum_as_string): 7}
#     for i in range(n_spin_orbitals):
#         op = {((i, "c"),): 9}
#         psi_new = applyOp(n_spin_orbitals, op, psi)
#         assert psi_new == {psr.str2bytes(vacuum_as_string[:i] + "1" + vacuum_as_string[i + 1 :]): 7 * 9}


# def test_applyOp_create_one_electron_from_one_electron_state():
#     product_state = "001000"
#     n_spin_orbitals = len(product_state)
#     # Multi-configurational state is a single product state
#     psi = {psr.str2bytes(product_state): 7}
#     for i in range(n_spin_orbitals):
#         op = {((i, "c"),): 9}
#         psi_new = applyOp(n_spin_orbitals, op, psi)
#         index_of_spin_orbital_with_one_electron_already = 2
#         if i == index_of_spin_orbital_with_one_electron_already:
#             # Can not put two electrons in the same spin orbital
#             assert psi_new == {}
#         else:
#             amp = 7 * 9 * (2 * (i < index_of_spin_orbital_with_one_electron_already) - 1)
#             assert psi_new == {psr.str2bytes(product_state[:i] + "1" + product_state[i + 1 :]): amp}


# def test_applyOp_create_two_electrons():
#     vacuum_as_string = "000000"
#     n_spin_orbitals = len(vacuum_as_string)
#     # Multi-configurational state is a single product state
#     psi = {psr.str2bytes(vacuum_as_string): 7}
#     for i in range(n_spin_orbitals):
#         for j in range(n_spin_orbitals):
#             op = {((i, "c"), (j, "c")): 9}
#             psi_new = applyOp(n_spin_orbitals, op, psi)
#             if i == j:
#                 # Can not put two electrons in the same spin orbital
#                 assert psi_new == {}
#             else:
#                 a, b = min(i, j), max(i, j)
#                 product_state = psr.str2bytes(
#                     vacuum_as_string[:a] + "1" + vacuum_as_string[a + 1 : b] + "1" + vacuum_as_string[b + 1 :]
#                 )
#                 amp = 7 * 9 * (2 * (i < j) - 1)
#                 assert psi_new == {product_state: amp}


# def test_applyOp_two_creation_processes():
#     vacuum_as_string = "000000"
#     n_spin_orbitals = len(vacuum_as_string)
#     # Multi-configurational state is a single product state
#     psi = {psr.str2bytes(vacuum_as_string): 7}
#     for i in range(n_spin_orbitals):
#         for j in range(n_spin_orbitals):
#             if i == j:
#                 continue
#             op = {((i, "c"),): 9, ((j, "c"),): 11}
#             psi_new = applyOp(n_spin_orbitals, op, psi)
#             assert psi_new == {
#                 psr.str2bytes(vacuum_as_string[:i] + "1" + vacuum_as_string[i + 1 :]): 7 * 9,
#                 psr.str2bytes(vacuum_as_string[:j] + "1" + vacuum_as_string[j + 1 :]): 7 * 11,
#             }


# def test_applyOp_remove_one_electron():
#     vacuum_as_string = "000000"
#     n_spin_orbitals = len(vacuum_as_string)
#     # Multi-configurational state is a single product state
#     psi = {psr.str2bytes(vacuum_as_string): 7}
#     for i in range(n_spin_orbitals):
#         op = {((i, "a"),): 9}
#         psi_new = applyOp(n_spin_orbitals, op, psi)
#         # Can't remove an electron from un-occupied spin-orbital
#         assert psi_new == {}


# def test_applyOp_two_processes_but_same_final_state():
#     product_state_as_string = "011000"
#     n_spin_orbitals = len(product_state_as_string)
#     # Multi-configurational state is a single product state
#     psi = {psr.str2bytes(product_state_as_string): 7}
#     op = {((1, "c"), (1, "a")): 9, ((2, "c"), (2, "a")): 11}
#     psi_new = applyOp(n_spin_orbitals, op, psi)
#     assert psi_new == {psr.str2bytes(product_state_as_string): 7 * (9 + 11)}


# def test_applyOp_opResult():
#     vacuum_as_string = "000000"
#     n_spin_orbitals = len(vacuum_as_string)
#     # Multi-configurational state is a single product state
#     psi = {psr.str2bytes(vacuum_as_string): 7}
#     for i in range(n_spin_orbitals):
#         for j in range(n_spin_orbitals):
#             if i == j:
#                 continue
#             op = {((i, "c"),): 9, ((j, "c"),): 11}
#             opResult = {}
#             psi_new = applyOp(n_spin_orbitals, op, psi, opResult=opResult)
#             psi_new_expected = {
#                 psr.str2bytes(vacuum_as_string[:i] + "1" + vacuum_as_string[i + 1 :]): 7 * 9,
#                 psr.str2bytes(vacuum_as_string[:j] + "1" + vacuum_as_string[j + 1 :]): 7 * 11,
#             }
#             assert psi_new == psi_new_expected
#             # opResult stores how the operator acts on individual product states.
#             assert opResult == {
#                 psr.str2bytes(vacuum_as_string): {
#                     psr.str2bytes(vacuum_as_string[:i] + "1" + vacuum_as_string[i + 1 :]): 9,
#                     psr.str2bytes(vacuum_as_string[:j] + "1" + vacuum_as_string[j + 1 :]): 11,
#                 }
#             }
#             # Now opResult is available and is used to (quickly) look-up the result
#             psi_new = applyOp(n_spin_orbitals, op, psi, opResult=opResult)
#             assert psi_new == psi_new_expected

#             # Store a wrong result in opResult and see that it's used instead of op
#             opResult = {psr.str2bytes(vacuum_as_string): {psr.str2bytes(n_spin_orbitals * "1"): 2}}
#             psi_new = applyOp(n_spin_orbitals, op, psi, opResult=opResult)
#             assert psi_new == {psr.str2bytes(n_spin_orbitals * "1"): 7 * 2}


# def test_applyOp_restrictions():
#     # Specify one restriction of the occupation (for each product state)
#     spin_orbital_indices = frozenset([1, 4, 5])
#     occupation_lower_and_upper_limits = (0, 1)
#     restrictions = {spin_orbital_indices: occupation_lower_and_upper_limits}

#     vacuum_as_string = "000000"
#     n_spin_orbitals = len(vacuum_as_string)
#     # Multi-configurational state is a single product state
#     psi = {psr.str2bytes(vacuum_as_string): 7}
#     for i in range(n_spin_orbitals):
#         for j in range(n_spin_orbitals):
#             op = {((i, "c"), (j, "c")): 9}
#             psi_new = applyOp(n_spin_orbitals, op, psi, restrictions=restrictions)
#             # Sanity check psi_new
#             if i == j:
#                 # Never can not put two electrons in the same spin orbital
#                 assert psi_new == {}
#             elif i in spin_orbital_indices and j in spin_orbital_indices:
#                 # The specified restrictions allow max one electron in spin-orbitals
#                 # with indices specified by the variable indices.
#                 assert psi_new == {}
#             else:
#                 a, b = min(i, j), max(i, j)
#                 product_state = psr.str2bytes(
#                     vacuum_as_string[:a] + "1" + vacuum_as_string[a + 1 : b] + "1" + vacuum_as_string[b + 1 :]
#                 )
#                 amp = 7 * 9 * (2 * (i < j) - 1)
#                 assert psi_new == {product_state: amp}
