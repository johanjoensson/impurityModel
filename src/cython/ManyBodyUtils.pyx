# distutils: language = c++
# cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, freethreading_compatible=True, cdivision=True, cpow=True 

from ManyBodyState cimport ManyBodyState as ManyBodyState_cpp, inner as inner_cpp
from ManyBodyOperator cimport ManyBodyOperator as ManyBodyOperator_cpp
import struct
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string

cdef bytes key_to_bytes(const ManyBodyState_cpp.key_type& key):
    cdef n_bytes = sizeof(ManyBodyState_cpp.key_type.value_type)
    cdef bytearray res = bytearray(n_bytes*key.size())
    cdef size_t i
    for i in range(key.size()):
        res[i*n_bytes: (i+1)*n_bytes] = key[i].to_bytes(n_bytes)
        # res[i*n_bytes: (i+1)*n_bytes] = key[i].to_bytes(n_bytes, byteorder='little')

    return bytes(res)

cdef ManyBodyState_cpp.key_type bytes_to_key(bytes b):
    cdef n_bytes = sizeof(ManyBodyState_cpp.key_type.value_type)
    cdef ManyBodyState_cpp.key_type key 
    cdef size_t i = 0
    key.reserve(len(b)//n_bytes)
    for i in range(0, len(b)//n_bytes, n_bytes):
        key.push_back(int.from_bytes(b[i*n_bytes:(i+1)*n_bytes]))
    if len(b) % n_bytes:

        # key.push_back(int.from_bytes(b[i*n_bytes:]))
        key.push_back(int.from_bytes(b[i*n_bytes:len(b)] + b'\x00'*(n_bytes - (len(b)%n_bytes))))
    return key

cdef class ManyBodyState:

    cdef ManyBodyState_cpp v

    def __cinit__(self, dict[bytes, complex] psi={}):
        cdef vector[ManyBodyState_cpp.key_type] keys
        cdef vector[double complex] amplitudes
        cdef ManyBodyState_cpp.key_type key
        cdef double complex val
        cdef bytes b
        for b, val in psi.items():
            keys.push_back(bytes_to_key(b))
            amplitudes.push_back(val)
        with nogil:
            self.v = ManyBodyState_cpp(keys, amplitudes)

    def __reduce__(self):
        return (self.__class__, (self.to_dict(), ))

    def __repr__(self):
        return "ManyBodyState({ " + ", ".join([f"{key_to_bytes(key)}: {amp}" for (key, amp) in self.v]) + "})"

    def __eq__(self, ManyBodyState other):
        return self.v == other.v

    def __add__(self, ManyBodyState other):
        res = ManyBodyState()
        with nogil:
            res.v = self.v + other.v
        return res

    def __iadd__(self, ManyBodyState other) -> ManyBodyState:
        with nogil:
            self.v = self.v + other.v
        return self


    def __sub__(self, ManyBodyState other):
        res = ManyBodyState()
        with nogil:
            res.v = self.v - other.v
        return res

    def __isub__(self, ManyBodyState other):
        with nogil:
            self.v = self.v - other.v
        return self

    def __mul__(self, double complex s):
        res = ManyBodyState()
        with nogil:
            res.v = self.v*  s
        return res

    def __imul__(self, double complex s):
        with nogil:
            self.v = self.v*  s
        return self

    def __rmul__(self, double complex s):
        res = ManyBodyState()
        with nogil:
            res.v = self.v*s
        return res

    def __truediv__(self, double complex s):
        res = ManyBodyState()
        with nogil:
            res.v = self.v /  s
        return res

    def __itruediv__(self, double complex s):
        with nogil:
            self.v = self.v /  s
        return self

    def __getitem__(self, bytes key):
        res= self.v[bytes_to_key(key)]
        return res

    def __setitem__(self, bytes key, double complex value):
        self.v[bytes_to_key(key)] = value

    def get(self, bytes key, double complex default = 0):
        if key in self:
            return self[key]
        return default



    def norm2(self):
        with nogil:
            res = self.v.norm2()
        return res

    def norm(self):
        with nogil:
            res= self.v.norm()
        return res

    def __len__(self):
        with nogil:
            res = self.v.size()
        return res

    def size(self):
        return len(self)

    def max_size(self):
        with nogil:
            res = self.v.max_size()
        return res

    def erase(self, bytes key):
        self.v.erase(bytes_to_key(key))

    def __contains__(self, bytes key):
        res = self.v.find(bytes_to_key(key)) != self.v.end()
        return res

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
        with nogil:
            self.v.prune(cutoff)

    def to_dict(self):
        return dict((key_to_bytes(p.first), p.second) for p in self.v)

    def copy(self):
        d = self.to_dict()
        # return ManyBodyState(list(d.keys()),list(d.values()) )
        return ManyBodyState(self.to_dict())

def inner(ManyBodyState a, ManyBodyState b):
    with nogil:
        res = inner_cpp(a.v, b.v)
    return res

cdef ManyBodyOperator_cpp.value_type.first_type processes_to_ints(tuple[tuple[int, str]] processes):
    cdef tuple[int, str] process
    cdef ManyBodyOperator_cpp.value_type.first_type ints
    ints.reserve(len(processes))
    for process in processes[::-1]:
        if process[1] == 'a':
            ints.push_back(-process[0] -1)
        else:
            ints.push_back(process[0])
    return ints

cdef tuple[tuple[int, str]] ints_to_processes(ManyBodyOperator_cpp.value_type.first_type& ints):
    cdef  list[tuple[int, str]] processes = []
    cdef ManyBodyOperator_cpp.value_type.first_type.value_type i
    for i in ints:
        if i < 0:
            processes.append((-i-1, 'a'))
        else:
            processes.append((i, 'c'))
    return  tuple(processes[::-1])

cdef class ManyBodyOperator:
    cdef ManyBodyOperator_cpp o

    def __cinit__(self, dict[tuple[tuple[int, str]], complex] op={}):
        cdef double complex amp 
        cdef tuple[int, str] processes
        cdef str action
        cdef ManyBodyOperator_cpp.value_type.first_type.value_type  i
        cdef vector[ManyBodyOperator_cpp.value_type] new_ops
        for processes, amp in op.items():
            new_ops.emplace_back(processes_to_ints(processes), amp)


        with nogil:
            self.o = ManyBodyOperator_cpp(new_ops)

    def __reduce__(self):
        return (self.__class__, (self.to_dict(), ))

    def __repr__(self):
        cdef ManyBodyOperator_cpp.value_type p
        return "ManyBodyOperator({" + ", ".join([f"{ints_to_processes(p.first)}: {p.second}" for p in self.o]) + "})"

    def __eq__(self, ManyBodyOperator other):
        return self.o == other.o

    def __getitem__(self, tuple[tuple[int, str]] key):
        return self.o[processes_to_ints(key)]

    def __setitem__(self, tuple[tuple[int, str]]key, double complex value):
        self.o[processes_to_ints(key)] = value

    def __add__(self, ManyBodyOperator other) ->ManyBodyOperator:
        res = ManyBodyOperator()
        with nogil:
            res.o = self.o + other.o
        return res

    def __iadd__(self, ManyBodyOperator other) ->ManyBodyOperator:
        with nogil:
            self.o = self.o + other.o
        return self

    def __sub__(self, ManyBodyOperator other) -> ManyBodyOperator:
        res = ManyBodyOperator()
        with nogil:
            res.o = self.o - other.o
        return res

    def __isub__(self, ManyBodyOperator other) -> ManyBodyOperator:
        with nogil:
            self.o = self.o - other.o
        return self

    def __mul__(self, complex s) ->ManyBodyOperator:
        res = ManyBodyOperator()
        with nogil:
            res.o = self.o*s
        return res

    def __imul__(self, complex s) ->ManyBodyOperator:
        with nogil:
            self.o = self.o*s
        return self

    def __rmul__(self, complex s) -> ManyBodyOperator:
        return self*s

    def __truediv__(self, complex s) -> ManyBodyOperator:
        res = ManyBodyOperator()
        with nogil:
            res.o = self.o/s
        return res

    def __itruediv__(self, complex s) -> ManyBodyOperator:
        with nogil:
            self.o = self.o/s
        return self

    def __len__(self):
        with nogil:
            res = self.o.size()
        return res

    def size(self):
        return len(self)

    def  __call__(self, ManyBodyState psi, double cutoff = 0, dict[frozenset[int], pair[int, int]] restrictions=None) -> ManyBodyState:
        res = ManyBodyState()
        if restrictions is None:
            restrictions = {}
        # For some reason using ManyBodyOpeartor_cpp.restrictions does not work
        cdef frozenset[int] indices
        cdef pair[size_t, size_t] limits
        cdef vector[pair[vector[size_t], pair[size_t, size_t]]] rest
        cdef vector[ManyBodyState_cpp] v

        rest.reserve(len(restrictions))
        for indices, limits in restrictions.items():
            if len(indices) == 0:
                continue
            rest.push_back(pair[vector[size_t], pair[size_t, size_t]](sorted(indices),pair[size_t, size_t](limits.first, limits.second)))
        with nogil:
            res.v = self.o(psi.v, cutoff, rest)
        return res

    def erase(self, tuple[tuple[int, str]]key):
        self.op.erase(processes_to_ints(key))

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

    
    def to_dict(self):
        return dict((ints_to_processes(p.first), p.second) for p in self.o)


def applyOp(ManyBodyOperator op, ManyBodyState psi, double cutoff=0, dict[vector[size_t], pair[size_t, size_t]] restrictions=None) ->ManyBodyState :
    return op(psi, cutoff, restrictions)
