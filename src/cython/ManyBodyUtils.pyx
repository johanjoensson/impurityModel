# distutils: language = c++
# cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, freethreading_compatible=True, cdivision=True, cpow=True


from ManyBodyState cimport ManyBodyState as ManyBodyState_cpp, inner as inner_cpp
from ManyBodyOperator cimport ManyBodyOperator as ManyBodyOperator_cpp
from SlaterDeterminant cimport SlaterDeterminant as SlaterDeterminant_cpp
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement
from libc.stdint cimport uint8_t, uint16_t, uint64_t
from libcpp.complex cimport complex

from copy import copy, deepcopy
from bitarray import bitarray

cdef extern from "<utility>" namespace "std" nogil:
    ManyBodyState_cpp& move(ManyBodyState_cpp)
    SlaterDeterminant_cpp& move(SlaterDeterminant_cpp)


cdef class SlaterDeterminant:
    cdef SlaterDeterminant_cpp[uint64_t] s

    def __cinit__(self, chunks: tuple[uint64_t] = None):
        if chunks is None:
            return
        cdef uint64_t chunk
        self.s.reserve(len(chunks))
        for chunk in chunks:
            self.s.push_back(chunk)

    @classmethod
    def from_bytes(cls, bytes b):
        cdef SlaterDeterminant_cpp[uint64_t] s
        cdef size_t n_bytes = sizeof(SlaterDeterminant_cpp[uint64_t].value_type)
        cdef size_t n_chunks = len(b) // n_bytes
        if len(b) % n_bytes:
            n_chunks += 1
        cdef bytes padded_b = int.from_bytes(b, byteorder='little').to_bytes(n_bytes*n_chunks)[::-1]

        return cls(tuple(int.from_bytes(padded_b[i:i+n_bytes]) for i in range(0, len(padded_b), n_bytes)))

    def to_bytearray(self):
        cdef size_t n_bytes = sizeof(SlaterDeterminant_cpp[uint64_t].value_type)
        res = bytearray(8*n_bytes*len(self))
        for i in range(len(self)):
            res[i*n_bytes:(i+1)*n_bytes] = self[i].to_bytes(n_bytes)
        return res


    def __getitem__(self, index: int):
        if index < 0:
            index = self.s.size() + index
        return self.s[index]

    def __setitem__(self, index: int, val: uint64_t):
        if index < 0:
            index = self.s.size() + index
        self.s[index] = val

    def __reduce__(self):
        return (self.__class__, (tuple(chunk for chunk in self.s),))


    def __len__(self):
        return self.s.size()

    def __repr__(self):
        return self.s.to_string().decode('utf-8')

    def __eq__(self, SlaterDeterminant other):
        return self.s == other.s

    def __ne__(self, SlaterDeterminant other):
        return self.s != other.s

    def __lt__(self, SlaterDeterminant other):
        return self.s < other.s

    def __le__(self, SlaterDeterminant other):
        return self.s <= other.s

    def __gt__(self, SlaterDeterminant other):
        return self.s > other.s

    def __ge__(self, SlaterDeterminant other):
        return self.s >= other.s

    def __iter__(self):
        it = self.s.begin()
        while it != self.s.end():
            yield dereference(it)
            preincrement(it)

    def __copy__(self):
        cls = self.__class__
        cdef SlaterDeterminant result = cls.__new__(cls)
        result.s = self.s
        return result

    def __deepcopy__(self, memo=None):
        cls = self.__class__
        cdef uint64_t val
        cdef SlaterDeterminant result = SlaterDeterminant(tuple(val for val in self.s))
        return result

    def __hash__(self):
        return self.s.hash()

    def copy(self, deep: bool = False):
        if deep:
            return deepcopy(self)
        return copy(self)



cdef class ManyBodyState:

    cdef ManyBodyState_cpp v

    def __cinit__(self, dict[SlaterDeterminant, ManyBodyState_cpp.mapped_type] psi=None):
        if psi is None:
            return
        cdef vector[ManyBodyState_cpp.key_type] keys
        cdef vector[ManyBodyState_cpp.mapped_type] amplitudes
        cdef SlaterDeterminant b
        cdef ManyBodyState_cpp.mapped_type val
        for b, val in psi.items():
            keys.push_back(b.s)
            amplitudes.push_back(val)
        self.v = ManyBodyState_cpp(keys, amplitudes)


    def __reduce__(self):
        return (self.__class__, (self.to_dict(), ))

    def __repr__(self):
        cdef pair[SlaterDeterminant_cpp[uint64_t], complex[double]] p
        return "ManyBodyState({ " + ", ".join([f"{p.first.to_string().decode('utf-8')}: {p.second}" for p in self.v]) + "})"

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

    def __neg__(self):
        res = ManyBodyState()
        with nogil:
            res.v = -self.v
        return res

    def __mul__(self, ManyBodyState_cpp.mapped_type s):
        res = ManyBodyState()
        with nogil:
            res.v = self.v * s
        return res

    def __imul__(self, ManyBodyState_cpp.mapped_type  s):
        with nogil:
            self.v = self.v * s
        return self

    def __rmul__(self, ManyBodyState_cpp.mapped_type  s):
        res = ManyBodyState()
        with nogil:
            res.v = self.v * s
        return res

    def __truediv__(self, ManyBodyState_cpp.mapped_type  s):
        res = ManyBodyState()
        with nogil:
            res.v = self.v / s
        return res

    def __itruediv__(self, ManyBodyState_cpp.mapped_type s):
        with nogil:
            self.v = self.v / s
        return self

    def __getitem__(self, SlaterDeterminant key):
        res= self.v[key.s]
        return res

    def __setitem__(self, SlaterDeterminant key, double complex value):
        self.v[key.s] = value

    def get(self, SlaterDeterminant key, double complex default = 0):
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

    def erase(self, SlaterDeterminant key):
        self.v.erase(key.s)

    def __contains__(self, SlaterDeterminant key):
        return self.v.find(key.s) != self.v.end()

    def __iter__(self):
        cdef SlaterDeterminant res
        it = self.v.begin()
        while it != self.v.end():
            res = SlaterDeterminant()
            res.s = dereference(it).first
            yield res
            preincrement(it)

    def keys(self):
        return (SlaterDeterminant(tuple(chunk for chunk in p.first)) for p in self.v)

    def values(self):
        return (p.second for p in self.v)

    def items(self):
        return ((SlaterDeterminant(tuple(chunk for chunk in p.first)), p.second) for p in self.v)

    def prune(self, double cutoff):
        with nogil:
            self.v.prune(cutoff)

    def to_dict(self):
        return dict((SlaterDeterminant(tuple(chunk for chunk in p.first)), p.second) for p in self.v)

    def copy(self):
        return ManyBodyState(self.to_dict())

def inner(ManyBodyState a, ManyBodyState b):
    with nogil:
        res = inner_cpp(a.v, b.v)
    return res


cdef ManyBodyOperator_cpp.value_type.first_type processes_to_ints(tuple[tuple[int, str]] processes):
    cdef tuple[int, str] process
    cdef ManyBodyOperator_cpp.key_type ints
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

    def __cinit__(self, dict[tuple[tuple[int, str]], complex] op=None):
        if op is None:
            op = {}
        cdef double complex amp
        cdef tuple[int, str] processes
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

    def __neg__(self) -> ManyBodyOperator:
        res = ManyBodyOperator()
        with nogil:
            res.o = -self.o
        return res

    def __mul__(self, ManyBodyOperator_cpp.mapped_type s) ->ManyBodyOperator:
        res = ManyBodyOperator()
        with nogil:
            res.o = self.o*s
        return res

    def __imul__(self, ManyBodyOperator_cpp.mapped_type s) ->ManyBodyOperator:
        with nogil:
            self.o = self.o*s
        return self

    def __rmul__(self, ManyBodyOperator_cpp.mapped_type s) -> ManyBodyOperator:
        return self*s

    def __truediv__(self, ManyBodyOperator_cpp.mapped_type s) -> ManyBodyOperator:
        res = ManyBodyOperator()
        with nogil:
            res.o = self.o / s
        return res

    def __itruediv__(self, ManyBodyOperator_cpp.mapped_type s) -> ManyBodyOperator:
        with nogil:
            self.o = self.o / s
        return self

    def __len__(self):
        with nogil:
            res = self.o.size()
        return res

    def size(self):
        return len(self)

    def set_restrictions(self,  dict[frozenset[int], pair[int, int]] restrictions=None):
        cdef frozenset[int] indices
        cdef pair[size_t, size_t] limits
        cdef vector[pair[vector[size_t], pair[size_t, size_t]]] rest

        if restrictions is None:
            restrictions = dict()
        rest.reserve(len(restrictions))
        for indices, limits in restrictions.items():
            if len(indices) == 0:
                continue
            rest.push_back(pair[vector[size_t], pair[size_t, size_t]](sorted(indices), pair[size_t, size_t](limits.first, limits.second)))
        with nogil:
            self.o.build_restriction_mask(rest)

    def  __call__(self, ManyBodyState psi, double cutoff = 0) -> ManyBodyState:
        cdef ManyBodyState res
        res = ManyBodyState()
        with nogil:
            res.v = self.o(psi.v, cutoff)
        return res

    def erase(self, tuple[tuple[int, str]]key):
        self.o.erase(processes_to_ints(key))

    def __contains__(self, tuple[tuple[int, str]]key):
        cdef ManyBodyOperator_cpp.key_type k = processes_to_ints(key)
        cdef ManyBodyOperator_cpp.iterator it = self.o.find(k)
        return it != self.o.end() and dereference(it).first == k

    def __iter__(self):
        it = self.o.begin()
        while it != self.o.end():
            yield ints_to_processes(dereference(it).first)
            preincrement(it)

    def keys(self):
        return (ints_to_processes(p.first) for p in self.o)

    def values(self):
        return (p.second for p in self.o)

    def items(self):
        return ((ints_to_processes(p.first), p.second) for p in self.o)

    def to_dict(self):
        return dict((ints_to_processes(p.first), p.second) for p in self.o)


def applyOp(ManyBodyOperator op, ManyBodyState psi, double cutoff=0) ->ManyBodyState :
    return op(psi, cutoff)
