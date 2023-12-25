cimport cython
cimport numpy as np
from libc.stdio cimport printf
from impurityModel.ed import product_state_representation as psr

cdef int[256] bits_set = [
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
        3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
        3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
        2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
        3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
        5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
        2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
        4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
        4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
        5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
        5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
 ]

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bytes2tuple(const unsigned char[:] state, Py_ssize_t n_spin_orbitals):
    cdef Py_ssize_t byte_i
    cdef Py_ssize_t byte_offset
    cdef unsigned char byte
    res = []
    for byte_i, byte in enumerate(state):
        for byte_offset in range(8):
            if byte_i*8 + byte_offset >= n_spin_orbitals:
                break
            if byte & (1 << byte_offset) == 1:
                res.append(byte_i*8 + byte_offset)
    return tuple(res)


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef add(state_1: dict, state_2: dict, v: complex = 1):
    for s2, a in state_2.items():
        state_1[s2] = a*v + state_1.get(s2, 0)
    return state_1

def applyOp(n_spin_orbitals, op, psi, slaterWeightMin=0, restrictions=None, opResult=None):
    result_psi = {}
    if opResult is None:
        opResult = dict()
    if restrictions is None:
        restrictions = dict()
    for state, amp in psi.items():
        if state in opResult:
            result_psi = add(result_psi, opResult[state], amp)
            continue
        opResult[state] = {}
        new_partial_psi = apply_to_state(n_spin_orbitals, op, state)
        for new_state, new_amp in new_partial_psi.items():
            new_state_tuple = psr.bytes2tuple(new_state, n_spin_orbitals)
            for restriction, occupations in restrictions.items():
                n = len(restriction.intersection(new_state_tuple))
                if n < occupations[0] or occupations[1] < n:
                    break
            else:
                # Occupations ok, so add contributions
                result_psi[new_state] = new_amp*amp + result_psi.get(new_state, 0)
                opResult[state][new_state] = new_amp
    for state, amp in list(result_psi.items()):
        if abs(amp) ** 2 < slaterWeightMin:
            result_psi.pop(state)
    return result_psi


def apply_to_state(n_spin_orbitals, op, state):
    result = dict()
    new_states, amps = apply_to_state_cy(n_spin_orbitals, list(op.keys()), list(op.values()), state)
    for state, amp in zip(new_states, amps):
        result[state] = amp + result.get(state, 0)
    return result


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef apply_to_state_cy(Py_ssize_t n_spin_orbitals, op_processes, op_amps, const unsigned char[:] state):
    cdef Py_ssize_t i, p_i, op_i
    cdef str action
    cdef unsigned char[:] new_state
    cdef complex signTot, sign, amp
    cdef tuple process
    # cdef list[bytes] states = [b'']
    # cdef list[float] amplitudes = [0]
    states = []
    amplitudes = []
    for op_i in range(len(op_processes)):
        process = op_processes[op_i]
        amp = op_amps[op_i]
        signTot = 1
        # for i, action in process[-1::-1]:
        # for action, i in process:
        new_state = state.copy()
        for p_i in range(len(process) - 1, -1, -1):
            i, action = process[p_i]
            if action == "a":
                # sign = remove.ubitarray(i, state_new)
                new_state, sign = annihilate_cy(n_spin_orbitals, i, new_state)
            elif action == "c":
                # sign = create.ubitarray(i, state_new)
                new_state, sign  = create_cy(n_spin_orbitals, i, new_state)
            elif action == "i":
                sign = 1

            signTot *= sign
            if sign == 0:
                break
        if sign == 0:
            continue
        states.append(bytes(new_state))
        amplitudes.append(amp*signTot)
    return states, amplitudes

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned char reverse_byte(const unsigned char byte):
    return ((byte * 0x0802LU & 0x22110LU) | (byte * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Py_ssize_t count_set_bits(const unsigned char[:] state, Py_ssize_t i):
    cdef Py_ssize_t byte_i = i / (8*sizeof(char))
    cdef Py_ssize_t byte_offset = i % (8*sizeof(char))
    cdef Py_ssize_t res = 0, index, j
    cdef unsigned char byte
    for j in range(byte_i):
        byte = state[j]
        index = byte
        res += bits_set[index]
    byte = state[byte_i]
    byte = byte >> (8*sizeof(char) - byte_offset)
    index = byte
    res += bits_set[index]
    return res

def create(Py_ssize_t n_spin_orbitals, Py_ssize_t i, const unsigned char[:] state):
    new_state, amp = create_cy(n_spin_orbitals, i, state)
    return bytes(new_state), amp

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef create_cy(Py_ssize_t n_spin_orbitals, Py_ssize_t i, const unsigned char[:] state):
    """
    Add electron at orbital i in state.

    Parameters
    ----------
    n_spin_orbitals : int (>=0)
        Total number of spin-orbitals in the system.
    i : int (>=0)
        Spin-orbital index
    state : bytes
        Product state.

    Returns
    -------
    state_new : bytes
        Product state.
    amp : int
        Amplitude. 0, -1 or 1.

    """
    cdef Py_ssize_t byte_i = i / (8*sizeof(char))
    cdef Py_ssize_t byte_offset = i % (8*sizeof(char))

    cdef unsigned char byte = state[byte_i]
    cdef unsigned char mask = (0x01 << 8*sizeof(char) - 1 - byte_offset)
    if byte & mask:
        return state, 0
    byte |= mask
    cdef unsigned char[:] new_state = state.copy()
    new_state[byte_i] = byte

    return new_state, 1 if count_set_bits(state, i) % 2 == 0 else -1


def annihilate(Py_ssize_t n_spin_orbitals, Py_ssize_t i, const unsigned char[:] state):
    new_state, amp = annihilate_cy(n_spin_orbitals, i, state)
    return bytes(new_state), amp

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef annihilate_cy(Py_ssize_t n_spin_orbitals, Py_ssize_t i, const unsigned char[:] state):
    """
    Remove electron at orbital i in state.

    Parameters
    ----------
    n_spin_orbitals : int (>=0)
        Total number of spin-orbitals in the system.
    i : int (>=0)
        Spin-orbital index
    state : bytes
        Product state.

    Returns
    -------
    state_new : bytes
        Product state.
    amp : int
        Amplitude. 0, -1 or 1.

    """
    cdef Py_ssize_t byte_i = i / (8*sizeof(char))
    cdef Py_ssize_t byte_offset = i % (8*sizeof(char))

    cdef unsigned char byte = state[byte_i]
    cdef unsigned char mask = (0x01 << 8*sizeof(char) - 1 - byte_offset)
    if not (byte & mask):
        return state, 0
    byte &= ~ mask
    cdef unsigned char[:] new_state = state.copy()
    new_state[byte_i] = byte

    return new_state, 1 if count_set_bits(state, i) % 2 == 0 else -1
