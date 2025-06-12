# distutils: language = c++
# cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, freethreading_compatible=True, cdivision=True, cpow=True 

from ManyBodyState cimport ManyBodyState as ManyBodyState_cpp, inner as inner_cpp
from ManyBodyOperator cimport ManyBodyOperator as ManyBodyOperator_cpp
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libc.stdint cimport uint8_t, int64_t

cdef bytes key_to_bytes(vector[uint8_t]& key):
    cdef bytearray res = bytearray(key.size())
    cdef size_t i
    for i in range(key.size()):
        res[i] = key[i]

    return bytes(res)

cdef vector[uint8_t] bytes_to_key(bytes b):
    cdef vector[uint8_t] key 
    cdef uint8_t byte
    key.reserve(len(b))
    for byte in b:
        key.push_back(byte)
    return key

cdef class ManyBodyState:

    cdef ManyBodyState_cpp v

    def __init__(self, dict[bytes, complex] psi={}):
        cdef vector[vector[uint8_t]] keys
        cdef vector[double complex] amplitudes
        cdef vector[uint8_t] key
        cdef double complex val
        cdef bytes b
        for b, val in psi.items():
            keys.push_back(bytes_to_key(b))
            amplitudes.push_back(val)
        self.v = ManyBodyState_cpp(keys, amplitudes)

    def __repr__(self):
        cdef pair[vector[uint8_t], double complex] p
        return "ManyBodyState({ " + ", ".join([f"{key_to_bytes(p.first)}: {p.second}" for p in self.v]) + "})"

    def __eq__(self, ManyBodyState other):
        return self.v == other.v

    def __add__(self, ManyBodyState other):
        res = ManyBodyState()
        res.v = self.v + other.v
        return res

    def __sub__(self, ManyBodyState other):
        res = ManyBodyState()
        res.v = self.v - other.v
        return res

    def __mul__(self, double complex s):
        res = ManyBodyState()
        res.v = self.v*  s
        return res

    def __rmul__(self, double complex s):
        res = ManyBodyState()
        res.v = self.v*s
        return res

    def __truediv__(self, double complex s):
        res = ManyBodyState()
        res.v = self.v /  s
        return res

    def __getitem__(self, bytes key):
        return self.v[bytes_to_key(key)]

    def __setitem__(self, bytes key, double complex value):
        self.v[bytes_to_key(key)] = value

    def norm2(self):
        return self.v.norm2()

    def norm(self):
        return self.v.norm()

    def __len__(self):
        return self.v.size()

    def size(self):
        return len(self)

    def max_size(self):
        return self.v.max_size()

    def erase(self, bytes key):
        self.v.erase(bytes_to_key(key))

    def __contains__(self, bytes key):
        return self.v.find(bytes_to_key(key)) != self.v.end()

    def __iter__(self):
        for p in self.v:
            yield key_to_bytes(p.first)

    def keys(self):
        return (key_to_bytes(p.first) for p in self.v)

    def values(self):
        return (p.second for p in self.v)

    def items(self):
        return ((key_to_bytes(p.first), p.second) for p in self.v)

    def prune(self, double cutoff):
        self.v.prune(cutoff)

    def to_dict(self):
        return dict((key_to_bytes(p.first), p.second) for p in self.v)

def inner(ManyBodyState a, ManyBodyState b):
    return inner_cpp(a.v, b.v)

cdef vector[int64_t] processes_to_ints(tuple[tuple[int, str]] processes):
    cdef tuple[int, str] process
    cdef vector[int64_t] ints
    ints.reserve(len(processes))
    for process in processes[::-1]:
        if process[1] == 'a':
            ints.push_back(-process[0] -1)
        else:
            ints.push_back(process[0])
    return ints

cdef tuple[tuple[int, str]] ints_to_processes(vector[int64_t]& ints):
    cdef  list[tuple[int, str]] processes = []
    cdef int64_t i
    for i in ints:
        if i < 0:
            processes.append((-i-1, 'a'))
        else:
            processes.append((i, 'c'))
    return  tuple(processes)

cdef class ManyBodyOperator:
    cdef ManyBodyOperator_cpp o

    def __init__(self, dict[tuple[tuple[int, str]], complex] op={}):
        cdef double complex amp 
        cdef tuple[int64_t, str] processes
        cdef str action
        cdef int64_t i
        cdef vector[pair[vector[int64_t], doublecomplex]] new_ops
        for processes, amp in op.items():
            new_ops.emplace_back(processes_to_ints(processes), amp)


        self.o = ManyBodyOperator_cpp(new_ops)

    def __repr__(self):
        cdef pair[vector[int64_t], double complex] p
        return "ManyBodyOperator({" + ", ".join([f"{ints_to_processes(p.first)}: {p.second}" for p in self.o]) + "})"

    def __eq__(self, ManyBodyOperator other):
        return self.o == other.o

    def __getitem__(self, tuple[tuple[int, str]] key):
        return self.o[processes_to_ints(key)]

    def __setitem__(self, tuple[tuple[int, str]]key, double complex value):
        self.o[processes_to_ints(key)] = value
        self.o.clear_memory()

    def __add__(self, ManyBodyOperator other) ->ManyBodyOperator:
        res = ManyBodyOperator()
        res.o = self.o + other.o
        return res

    def __sub__(self, ManyBodyOperator other) -> ManyBodyOperator:
        res = ManyBodyOperator()
        res.o = self.o - other.o
        return res

    def __mul__(self, complex s) ->ManyBodyOperator:
        res = ManyBodyOperator()
        res.o = self.o*s
        return res

    def __rmul__(self, complex s) -> ManyBodyOperator:
        return self*s

    def __truediv__(self, complex s) -> ManyBodyOperator:
        res = ManyBodyOperator()
        res.o = self.o/s
        return res

    def __len__(self):
        return self.o.size()

    def size(self):
        return len(self)

    def __call__(self, ManyBodyState psi, double cutoff=0, dict[frozenset[int], tuple[int, int]] restrictions=None) -> ManyBodyState:
        res = ManyBodyState()
        if restrictions is None:
            restrictions = {}
        cdef map[vector[size_t], pair[size_t, size_t], ManyBodyOperator_cpp.Comparer[size_t]] rest
        cdef frozenset[int] indices
        cdef tuple[int, int] limits
        for indices, limits in restrictions.items():
            if len(indices) == 0:
                continue
            rest.insert((sorted(indices),limits))
        res.v = self.o(psi.v, cutoff, rest)
        return res

    def erase(self, tuple[tuple[int, str]]key):
        self.op.erase(processes_to_ints(key))
        self.o.clear_memory()

    def __contains__(self, tuple[tuple[int, str]]key):
        return self.o.find(processes_to_ints(key)) != self.o.end()

    def __iter__(self):
        for p in self.o:
            yield ints_to_processes(p.first)

    def keys(self):
        return (ints_to_processes(p.first) for p in self.o)

    def values(self):
        return (p.second for p in self.o)

    def items(self):
        return ((ints_to_processes(p.first), p.second) for p in self.o)

    def memory(self):
        res = dict((key_to_bytes(p.first), dict((key_to_bytes(s.first), s.second) for s in p.second)) for p in self.o.memory())
        return res

    def to_dict(self):
        return dict((ints_to_processes(p.first), p.second) for p in self.o)


def applyOp(ManyBodyOperator op, ManyBodyState psi, double cutoff=0, dict[vector[size_t], pair[size_t, size_t]] restrictions=None):
    return op(psi, cutoff, restrictions)
