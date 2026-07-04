# distutils: language = c++
# cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, freethreading_compatible=True, cdivision=True, cpow=True


from ManyBodyState cimport ManyBodyState as ManyBodyState_cpp, inner as inner_cpp
from ManyBodyOperator cimport ManyBodyOperator as ManyBodyOperator_cpp
from SlaterDeterminant cimport SlaterDeterminant as SlaterDeterminant_cpp
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement
from libcpp.algorithm cimport sort, unique, lower_bound
from libcpp.map cimport map as cpp_map
from libc.stdint cimport uint8_t, uint64_t, int64_t, int32_t
from libc.string cimport memcpy
from libcpp.complex cimport complex as complex_cpp

from copy import copy, deepcopy

import numpy as np

cdef extern from "<utility>" namespace "std" nogil:
    ManyBodyState_cpp& move(ManyBodyState_cpp)
    SlaterDeterminant_cpp& move(SlaterDeterminant_cpp)


cdef class SlaterDeterminant:
    """
    Cython wrapper class for C++ SlaterDeterminant.

    Represents a many-body Slater determinant using 64-bit integer chunks
    to track spin-orbital occupations.
    """
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
        """
        Create a SlaterDeterminant from bytes representation.
        """
        cdef SlaterDeterminant_cpp[uint64_t] _s
        cdef size_t n_bytes = sizeof(SlaterDeterminant_cpp[uint64_t].value_type)
        cdef size_t n_chunks = len(b) // n_bytes
        if len(b) % n_bytes:
            n_chunks += 1
        cdef bytes padded_b = int.from_bytes(b, byteorder='little').to_bytes(n_bytes*n_chunks)[::-1]

        return cls(tuple(int.from_bytes(padded_b[i:i+n_bytes]) for i in range(0, len(padded_b), n_bytes)))

    def to_bytearray(self):
        """
        Convert SlaterDeterminant to bytearray.
        """
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
        cdef uint64_t val
        cdef SlaterDeterminant result = SlaterDeterminant(tuple(val for val in self.s))
        return result

    def __hash__(self):
        return self.s.hash()

    def get_hash(self):
        return self.s.hash()

    def copy(self, deep: bool = False):
        """
        Return a copy of the Slater determinant.
        """
        if deep:
            return deepcopy(self)
        return copy(self)


cdef class ManyBodyState:
    """
    Cython wrapper class for C++ ManyBodyState.

    Represents a quantum many-body state as a map from SlaterDeterminants
    to complex amplitudes.
    """
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
        cdef pair[SlaterDeterminant_cpp[uint64_t], complex_cpp[double]] p
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

    def add_scaled(self, ManyBodyState other, ManyBodyState_cpp.mapped_type scalar):
        """
        In-place addition of another ManyBodyState scaled by a complex scalar.
        Equivalent to: self += scalar * other, but avoids creating intermediate objects.
        """
        with nogil:
            self.v.add_scaled(other.v, scalar)
        return self

    def __getitem__(self, SlaterDeterminant key):
        res= self.v[key.s]
        return res

    def __setitem__(self, SlaterDeterminant key, ManyBodyState_cpp.mapped_type value):
        self.v[key.s] = value

    def get(self, SlaterDeterminant key, ManyBodyState_cpp.mapped_type default=ManyBodyState_cpp.mapped_type(0., 0.)):
        """
        Get the amplitude of a SlaterDeterminant key, or return the default.
        """
        if key in self:
            return self[key]
        return default

    def norm2(self):
        """
        Compute squared L2 norm of the state.
        """
        with nogil:
            res = self.v.norm2()
        return res

    def norm(self):
        """
        Compute L2 norm of the state.
        """
        with nogil:
            res= self.v.norm()
        return res

    def __len__(self):
        with nogil:
            res = self.v.size()
        return res

    def size(self):
        """
        Return the number of configurations in the state.
        """
        return len(self)

    def max_size(self):
        """
        Return the maximum potential size of the state.
        """
        with nogil:
            res = self.v.max_size()
        return res

    def erase(self, SlaterDeterminant key):
        """
        Remove a SlaterDeterminant configuration from the state.
        """
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
        """
        Yield all SlaterDeterminant configurations in the state.
        """
        cdef SlaterDeterminant result
        for p in self.v:
            result = SlaterDeterminant.__new__(SlaterDeterminant)
            result.s = p.first
            yield result

    def values(self):
        """
        Yield all amplitudes in the state.
        """
        return (p.second for p in self.v)

    def items(self):
        """
        Yield pairs of (SlaterDeterminant, amplitude).
        """
        cdef SlaterDeterminant result
        for p in self.v:
            result = SlaterDeterminant.__new__(SlaterDeterminant)
            result.s = p.first
            yield (result, p.second)

    def prune(self, double cutoff):
        """
        Prune amplitudes with norm below cutoff.
        """
        with nogil:
            self.v.prune(cutoff)

    def to_dict(self):
        """
        Convert the ManyBodyState to a python dict.
        """
        cdef dict res = {}
        cdef SlaterDeterminant result
        for p in self.v:
            result = SlaterDeterminant.__new__(SlaterDeterminant)
            result.s = p.first
            res[result] = p.second
        return res

    def copy(self):
        """
        Return a copy of the many-body state.
        """
        return ManyBodyState(self.to_dict())

    def clear(self):
        """
        Clear the manybody state.
        """
        self.v.clear()

    def is_empty(self):
        """
        Returns true if the ManyBodyState is empty (i.e. contains no SlaterDeterminants)
        """
        return self.v.empty()


def inner(ManyBodyState a, ManyBodyState b):
    """
    Compute the inner product of many-body states: <a|b>.
    """
    cdef ManyBodyState_cpp.mapped_type res
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
    """
    Cython wrapper class for C++ ManyBodyOperator.

    Represents a quantum many-body operator composed of creation
    and annihilation process sequences with corresponding amplitudes.
    """
    cdef ManyBodyOperator_cpp o

    def __cinit__(self, dict[tuple[tuple[int, str]], complex_cpp] op=None):
        if op is None:
            op = {}
        cdef ManyBodyState_cpp.mapped_type amp
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

    def __setitem__(self, tuple[tuple[int, str]]key, ManyBodyState_cpp.mapped_type value):
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
        """
        Return the number of operator terms.
        """
        return len(self)

    def set_restrictions(self,  dict[frozenset[int], pair[int, int]] restrictions=None):
        """
        Configure orbital restrictions for operator application.
        """
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

    def set_weighted_restrictions(self, list restrictions=None):
        """
        Configure weighted-sum occupation constraints, e.g. S_z = sum +-1 n_i or
        L_z = sum m_l n_i (after scaling weights to integers).

        Each element of ``restrictions`` is ``(weights, (q_min, q_max))`` where
        ``weights`` maps an orbital index to its (integer) weight; a Slater
        determinant passes iff ``q_min <= sum_i weights[i] * n_i <= q_max``.
        """
        cdef vector[pair[vector[pair[long, vector[size_t]]], pair[long, long]]] rest
        cdef vector[pair[long, vector[size_t]]] groups
        cdef long w

        if restrictions is None:
            restrictions = []
        rest.reserve(len(restrictions))
        for weights, bounds in restrictions:
            by_weight = {}
            for orb, weight in weights.items():
                by_weight.setdefault(int(weight), []).append(int(orb))
            groups.clear()
            for w, orbs in by_weight.items():
                if w == 0:
                    continue
                groups.push_back(pair[long, vector[size_t]](w, sorted(orbs)))
            rest.push_back(
                pair[vector[pair[long, vector[size_t]]], pair[long, long]](
                    groups, pair[long, long](<long>bounds[0], <long>bounds[1])
                )
            )
        with nogil:
            self.o.build_weighted_restriction_mask(rest)

    def  __call__(self, ManyBodyState psi, double cutoff = 0) -> ManyBodyState:
        cdef ManyBodyState res
        res = ManyBodyState()
        with nogil:
            res.v = self.o(psi.v, cutoff)
        return res

    def apply(self, ManyBodyState psi, double cutoff=0) -> ManyBodyState:
        cdef ManyBodyState res
        res = ManyBodyState()

        with nogil:
            res.v = self.o.apply(psi.v, cutoff)
        return res

    def apply_multi(self, list psis, double cutoff = 0) -> list:
        """
        Apply operator to a list of ManyBodyStates without python looping.
        """
        cdef int n = len(psis)
        cdef vector[const ManyBodyState_cpp*] p_ptrs
        p_ptrs.reserve(n)

        cdef ManyBodyState state
        for obj in psis:
            state = <ManyBodyState?>obj
            p_ptrs.push_back(&state.v)

        cdef vector[ManyBodyState_cpp] res_vec

        with nogil:
            res_vec = self.o.apply(p_ptrs, cutoff)

        cdef list res_list = []
        cdef ManyBodyState res_state
        for i in range(n):
            res_state = ManyBodyState()
            res_state.v = move(res_vec[i])
            res_list.append(res_state)

        return res_list

    def erase(self, tuple[tuple[int, str]]key):
        """
        Remove a term from the operator.
        """
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
        """
        Yield all operator terms.
        """
        return (ints_to_processes(p.first) for p in self.o)

    def values(self):
        """
        Yield all operator amplitudes.
        """
        return (p.second for p in self.o)

    def items(self):
        """
        Yield pairs of (operator_term, amplitude).
        """
        return ((ints_to_processes(p.first), p.second) for p in self.o)

    def to_dict(self):
        """
        Convert operator terms to a python dict.
        """
        return dict((ints_to_processes(p.first), p.second) for p in self.o)

    def is_empty(self):
        """
        Returns True if the ManyBodyOperator is empty (contains no processes).
        """
        return self.o.empty()

    def clear(self):
        """
        Clears the operator (removes all processes)
        """
        self.o.clear()

    def set_normal_ordering(self, bint enable):
        """
        Enable/disable build-time normal ordering of the apply() representation.

        Representation change only; the operator's action is unchanged. Default enabled.
        """
        self.o.set_normal_ordering(enable)

    def normal_ordering(self) -> bool:
        """Return whether build-time normal ordering is enabled."""
        return self.o.normal_ordering()

    def num_flat_terms(self) -> int:
        """Number of terms in the (possibly normal-ordered) apply() representation."""
        return self.o.num_flat_terms()


def applyOp(ManyBodyOperator op, ManyBodyState psi, double cutoff=0) ->ManyBodyState :
    """
    Apply a ManyBodyOperator to a ManyBodyState.
    """
    return op(psi, cutoff)


from MpiUtils cimport pack_determinants as c_pack_determinants, unpack_determinants as c_unpack_determinants, pack_psis as c_pack_psis, unpack_psis as c_unpack_psis, pack_psis_fused as c_pack_psis_fused, unpack_psis_fused as c_unpack_psis_fused
import numpy as np


def pack_determinants_cy(list dets, int comm_size):
    cdef vector[SlaterDeterminant_cpp[uint64_t]] c_dets
    c_dets.reserve(len(dets))
    cdef SlaterDeterminant det
    for det in dets:
        c_dets.push_back(det.s)

    cdef vector[int64_t] send_counts
    cdef vector[uint64_t] state_buf

    with nogil:
        c_pack_determinants(c_dets, comm_size, send_counts, state_buf)

    cdef size_t total_states = state_buf.size()

    send_counts_np = np.zeros(comm_size, dtype=np.int64)
    state_buf_np = np.zeros(total_states, dtype=np.uint64)

    cdef int64_t[:] send_counts_view = send_counts_np
    cdef uint64_t[:] state_buf_view = state_buf_np

    if comm_size > 0:
        memcpy(&send_counts_view[0], <void*>send_counts.data(), comm_size * sizeof(int64_t))

    if total_states > 0:
        memcpy(&state_buf_view[0], <void*>state_buf.data(), total_states * sizeof(uint64_t))

    return send_counts_np, state_buf_np


def unpack_determinants_cy(int comm_size, int64_t[:] recv_counts, uint64_t[:] state_buf, size_t chunks_per_state):
    cdef vector[int64_t] c_recv_counts
    cdef vector[uint64_t] c_state_buf

    if comm_size > 0:
        c_recv_counts.assign(<int64_t*> &recv_counts[0], <int64_t*> &recv_counts[0] + comm_size)

    if state_buf.shape[0] > 0:
        c_state_buf.assign(<uint64_t*> &state_buf[0], <uint64_t*> &state_buf[0] + state_buf.shape[0])

    cdef vector[vector[SlaterDeterminant_cpp[uint64_t]]] c_res
    with nogil:
        c_res = c_unpack_determinants(comm_size, c_recv_counts, c_state_buf, chunks_per_state)

    cdef list res = []
    cdef SlaterDeterminant py_det
    for i in range(comm_size):
        rank_dets = []
        for j in range(c_res[i].size()):
            py_det = SlaterDeterminant()
            py_det.s = c_res[i][j]
            rank_dets.append(py_det)
        res.append(rank_dets)
    return res


def pack_psis_cy(list psis, int comm_size):
    cdef vector[const ManyBodyState_cpp*] c_psis
    cdef ManyBodyState psi
    for psi in psis:
        c_psis.push_back(&(psi.v))

    cdef vector[int64_t] send_counts
    cdef vector[uint64_t] state_buf
    cdef vector[double] amp_buf_reim
    cdef vector[int32_t] psi_buf

    with nogil:
        c_pack_psis(c_psis, comm_size, send_counts, state_buf, amp_buf_reim, psi_buf)

    cdef size_t total_states = state_buf.size()
    cdef size_t n_entries = psi_buf.size()

    send_counts_np = np.zeros(comm_size, dtype=np.int64)
    state_buf_np = np.zeros(total_states, dtype=np.uint64)
    amp_buf_np = np.zeros(n_entries, dtype=np.complex128)
    psi_buf_np = np.zeros(n_entries, dtype=np.int32)

    cdef int64_t[:] send_counts_view = send_counts_np
    cdef uint64_t[:] state_buf_view = state_buf_np
    cdef complex[:] amp_buf_view = amp_buf_np
    cdef int32_t[:] psi_buf_view = psi_buf_np

    if comm_size > 0:
        memcpy(&send_counts_view[0], <void*>send_counts.data(), comm_size * sizeof(int64_t))

    if total_states > 0:
        memcpy(&state_buf_view[0], <void*>state_buf.data(), total_states * sizeof(uint64_t))

    if n_entries > 0:
        memcpy(&amp_buf_view[0], <void*>amp_buf_reim.data(), n_entries * 2 * sizeof(double))
        memcpy(&psi_buf_view[0], <void*>psi_buf.data(), n_entries * sizeof(int32_t))

    return send_counts_np, state_buf_np, amp_buf_np, psi_buf_np


def unpack_psis_cy(list psis, int comm_size, int64_t[:] recv_counts, uint64_t[:] state_buf, complex[:] amp_buf, int32_t[:] psi_buf, size_t chunks_per_state):
    """
    Unpacks vectors natively into ManyBodyState objects.
    """
    cdef vector[ManyBodyState_cpp*] c_psis
    cdef ManyBodyState psi
    for psi in psis:
        c_psis.push_back(&(psi.v))

    cdef vector[int64_t] c_recv_counts
    cdef vector[uint64_t] c_state_buf
    cdef vector[double] c_amp_buf_reim
    cdef vector[int32_t] c_psi_buf

    if comm_size > 0:
        c_recv_counts.assign(<int64_t*> &recv_counts[0], <int64_t*> &recv_counts[0] + comm_size)

    if state_buf.shape[0] > 0:
        c_state_buf.assign(<uint64_t*> &state_buf[0], <uint64_t*> &state_buf[0] + state_buf.shape[0])

    if amp_buf.shape[0] > 0:
        c_amp_buf_reim.assign(<double*> &amp_buf[0], <double*> &amp_buf[0] + amp_buf.shape[0] * 2)

    if psi_buf.shape[0] > 0:
        c_psi_buf.assign(<int32_t*> &psi_buf[0], <int32_t*> &psi_buf[0] + psi_buf.shape[0])

    with nogil:
        c_unpack_psis(c_psis, comm_size, c_recv_counts, c_state_buf, c_amp_buf_reim, c_psi_buf, chunks_per_state)


def pack_psis_fused_cy(list psis, int comm_size, size_t chunks_per_state):
    """Pack psis into a single interleaved byte buffer (state||amp||psi_idx per entry),
    rank-ordered, for a one-shot Neighbor_alltoallv(MPI.BYTE) redistribute."""
    cdef vector[const ManyBodyState_cpp*] c_psis
    cdef ManyBodyState psi
    for psi in psis:
        c_psis.push_back(&(psi.v))

    cdef vector[int64_t] send_counts
    cdef vector[char] send_buf

    with nogil:
        c_pack_psis_fused(c_psis, comm_size, chunks_per_state, send_counts, send_buf)

    cdef size_t nbytes = send_buf.size()
    send_counts_np = np.zeros(comm_size, dtype=np.int64)
    send_buf_np = np.zeros(nbytes, dtype=np.uint8)

    cdef int64_t[:] send_counts_view = send_counts_np
    cdef uint8_t[:] send_buf_view = send_buf_np
    if comm_size > 0:
        memcpy(&send_counts_view[0], <void*>send_counts.data(), comm_size * sizeof(int64_t))
    if nbytes > 0:
        memcpy(&send_buf_view[0], <void*>send_buf.data(), nbytes)

    return send_counts_np, send_buf_np


def unpack_psis_fused_cy(list psis, int comm_size, int64_t[:] recv_counts, uint8_t[:] recv_buf, size_t chunks_per_state):
    """Unpack the interleaved byte buffer produced by pack_psis_fused_cy."""
    cdef vector[ManyBodyState_cpp*] c_psis
    cdef ManyBodyState psi
    for psi in psis:
        c_psis.push_back(&(psi.v))

    cdef vector[int64_t] c_recv_counts
    cdef vector[char] c_recv_buf

    if comm_size > 0:
        c_recv_counts.assign(<int64_t*> &recv_counts[0], <int64_t*> &recv_counts[0] + comm_size)
    if recv_buf.shape[0] > 0:
        c_recv_buf.assign(<char*> &recv_buf[0], <char*> &recv_buf[0] + recv_buf.shape[0])

    with nogil:
        c_unpack_psis_fused(c_psis, comm_size, c_recv_counts, c_recv_buf, chunks_per_state)


def extract_new_states(list states, object existing_dict):
    """
    Extracts all unique SlaterDeterminant keys from a list of ManyBodyStates
    that are not already present in existing_dict.
    """
    cdef set new_states = set()
    cdef ManyBodyState s
    cdef ManyBodyState_cpp.const_iterator it
    cdef SlaterDeterminant key
    for obj in states:
        s = <ManyBodyState?>obj
        it = s.v.cbegin()
        while it != s.v.cend():
            key = SlaterDeterminant(tuple(chunk for chunk in dereference(it).first))
            if key not in existing_dict and key not in new_states:
                new_states.add(key)
            preincrement(it)
    return new_states


def inner_multi(list states_a, list states_b):
    """
    Compute inner products between two lists of ManyBodyStates.
    Returns a complex numpy array of shape (len(states_a), len(states_b)).
    """
    cdef int na = len(states_a)
    cdef int nb = len(states_b)
    cdef res = np.zeros((na, nb), dtype=complex)
    cdef complex[:, :] res_view = res
    cdef ManyBodyState_cpp.mapped_type val
    cdef int i, j

    cdef vector[ManyBodyState_cpp*] a_ptrs
    cdef vector[ManyBodyState_cpp*] b_ptrs
    cdef ManyBodyState state

    a_ptrs.reserve(na)
    for obj in states_a:
        state = <ManyBodyState?>obj
        a_ptrs.push_back(&state.v)

    b_ptrs.reserve(nb)
    for obj in states_b:
        state = <ManyBodyState?>obj
        b_ptrs.push_back(&state.v)

    with nogil:
        for i in range(na):
            for j in range(nb):
                val = inner_cpp(dereference(a_ptrs[i]), dereference(b_ptrs[j]))
                res_view[i, j].real = val.real()
                res_view[i, j].imag = val.imag()

    return res


def add_scaled_multi(list states_target, list states_source, complex[:, :] coeffs):
    """
    Add a scaled sum of states_source to each state in states_target::

        for each j in range(len(states_target)):
            for i in range(len(states_source)):
                states_target[j] += coeffs[i, j] * states_source[i]
    """
    cdef int n_target = len(states_target)
    cdef int n_source = len(states_source)

    cdef vector[ManyBodyState_cpp*] t_ptrs
    cdef vector[ManyBodyState_cpp*] s_ptrs
    cdef ManyBodyState state

    t_ptrs.reserve(n_target)
    for obj in states_target:
        state = <ManyBodyState?>obj
        t_ptrs.push_back(&state.v)

    s_ptrs.reserve(n_source)
    for obj in states_source:
        state = <ManyBodyState?>obj
        s_ptrs.push_back(&state.v)

    cdef int i, j
    cdef double complex coeff
    with nogil:
        for j in range(n_target):
            for i in range(n_source):
                coeff = coeffs[i, j]
                if coeff.real != 0 or coeff.imag != 0:
                    dereference(t_ptrs[j]).add_scaled(dereference(s_ptrs[i]), ManyBodyState_cpp.mapped_type(coeff.real, coeff.imag))


def reorth_cgs2_dense(list wp, list Q, int n_passes, object comm):
    r"""Dense (BLAS) block reorthogonalization of ``wp`` against ``Q`` for the sparse
    (``ManyBodyState``) path.

    Performs ``n_passes`` of classical block Gram-Schmidt

    .. math:: O = Q^\dagger\, wp \quad(\text{Allreduced over ranks});\qquad wp \leftarrow wp - Q\,O,

    but materializes ``wp`` and ``Q`` onto their merged determinant support and runs the two
    projections as ``zgemm`` instead of per-pair ``flat_map`` inner products / merges.
    Mathematically equivalent (to floating point) to repeating ``block_orthogonalize_sparse``
    ``n_passes`` times -- the W-estimator / bad-block selection is unchanged; only the projection
    is accelerated. Returns a new list of ``wp`` ManyBodyStates.
    """
    cdef int p = len(wp)
    cdef int nq = len(Q)
    if p == 0 or nq == 0 or n_passes <= 0:
        return wp

    cdef vector[ManyBodyState_cpp*] wp_ptrs
    cdef vector[ManyBodyState_cpp*] q_ptrs
    cdef ManyBodyState ms
    cdef int ci
    for obj in wp:
        ms = <ManyBodyState?>obj
        wp_ptrs.push_back(&ms.v)
    for obj in Q:
        ms = <ManyBodyState?>obj
        q_ptrs.push_back(&ms.v)

    # --- merged local support: sorted, unique determinant keys over wp ∪ Q ---
    cdef vector[SlaterDeterminant_cpp[uint64_t]] support
    cdef ManyBodyState_cpp.iterator it
    for ci in range(p):
        it = wp_ptrs[ci].begin()
        while it != wp_ptrs[ci].end():
            support.push_back(dereference(it).first)
            preincrement(it)
    for ci in range(nq):
        it = q_ptrs[ci].begin()
        while it != q_ptrs[ci].end():
            support.push_back(dereference(it).first)
            preincrement(it)
    # NOTE: an empty local support must NOT return early under MPI -- p, nq and
    # n_passes are rank-identical (bad_cols is bcast), so every rank must join the
    # n_passes Allreduce(O) collectives below; a rank-local early return mispairs
    # the other ranks' Allreduce with this rank's next collective (wrong numerics,
    # then deadlock -- seen at 4 ranks where a rank owns no determinants). With
    # ns == 0 the GEMMs are zero-row no-ops and O contributes zeros, as intended.
    if support.size() == 0 and comm is None:
        return wp
    sort(support.begin(), support.end())
    support.erase(unique(support.begin(), support.end()), support.end())
    cdef Py_ssize_t ns = <Py_ssize_t>support.size()

    # --- materialize dense (|S| x p) and (|S| x nq) over the local support ---
    Wd = np.zeros((ns, p), dtype=complex)
    Qd = np.zeros((ns, nq), dtype=complex)
    cdef complex[:, :] Wv = Wd
    cdef complex[:, :] Qv = Qd
    cdef Py_ssize_t row
    cdef ManyBodyState_cpp.mapped_type cval
    for ci in range(p):
        it = wp_ptrs[ci].begin()
        while it != wp_ptrs[ci].end():
            row = lower_bound(support.begin(), support.end(), dereference(it).first) - support.begin()
            cval = dereference(it).second
            Wv[row, ci].real = cval.real()
            Wv[row, ci].imag = cval.imag()
            preincrement(it)
    for ci in range(nq):
        it = q_ptrs[ci].begin()
        while it != q_ptrs[ci].end():
            row = lower_bound(support.begin(), support.end(), dereference(it).first) - support.begin()
            cval = dereference(it).second
            Qv[row, ci].real = cval.real()
            Qv[row, ci].imag = cval.imag()
            preincrement(it)

    # --- CGS2 passes via BLAS; the small (nq x p) overlap is Allreduced each pass ---
    Qh = np.conj(Qd.T)
    if comm is not None:
        from mpi4py import MPI
    for _ in range(n_passes):
        O = Qh @ Wd
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, O, op=MPI.SUM)
        Wd = Wd - Qd @ O

    # --- scatter back to ManyBodyStates (keep the nonzero rows of each column) ---
    cdef complex[:, :] Wout = Wd
    cdef list out = []
    cdef vector[ManyBodyState_cpp.key_type] keys
    cdef vector[ManyBodyState_cpp.mapped_type] vals
    cdef ManyBodyState new_ms
    cdef double complex z
    for ci in range(p):
        keys.clear()
        vals.clear()
        for row in range(ns):
            z = Wout[row, ci]
            if z.real != 0 or z.imag != 0:
                keys.push_back(support[row])
                vals.push_back(ManyBodyState_cpp.mapped_type(z.real, z.imag))
        new_ms = ManyBodyState()
        new_ms.v = ManyBodyState_cpp(keys, vals)
        out.append(new_ms)
    return out


cdef class SparseKrylovDense:
    r"""Incrementally-maintained dense copy of the (rank-local) sparse block-Krylov basis.

    Holds the Krylov vectors as a dense ``(n_rows x n_cols)`` complex buffer over a growing
    determinant->row support map, so block reorthogonalization can *slice* columns
    (``Q[:, cols]``) instead of re-materializing them from the ``flat_map`` states on every
    step. ``append`` mirrors the driver's ``Q_basis.extend`` (one block at a time); ``reort``
    runs ``n_passes`` of classical Gram-Schmidt of ``wp`` against the selected columns via BLAS
    ``zgemm``, identical (to floating point) to repeating ``block_orthogonalize_sparse``. The
    buffer is rank-local (over the rank's owned determinants); the only collective is the small
    ``(n_cols x p)`` overlap ``Allreduce`` each pass, exactly as in the map-based path.
    """
    cdef cpp_map[SlaterDeterminant_cpp[uint64_t], int] support
    cdef vector[SlaterDeterminant_cpp[uint64_t]] row_det
    cdef object Qbuf
    cdef int n_rows
    cdef int n_cols
    cdef int cap_rows
    cdef int cap_cols

    def __cinit__(self):
        self.n_rows = 0
        self.n_cols = 0
        self.cap_rows = 256
        self.cap_cols = 32
        self.Qbuf = np.zeros((self.cap_rows, self.cap_cols), dtype=complex)

    cdef void _realloc(self, int new_cap_rows, int new_cap_cols):
        # Copy the old buffer's full extent (it holds all written data plus zeros); n_rows may
        # already have been grown past the old capacity by _register, and the new rows carry no
        # data yet, so copy by the *old buffer* shape, not by n_rows/n_cols.
        newbuf = np.zeros((new_cap_rows, new_cap_cols), dtype=complex)
        cdef int r = min(<int>self.Qbuf.shape[0], new_cap_rows)
        cdef int c = min(<int>self.Qbuf.shape[1], new_cap_cols)
        newbuf[:r, :c] = self.Qbuf[:r, :c]
        self.Qbuf = newbuf
        self.cap_rows = new_cap_rows
        self.cap_cols = new_cap_cols

    cdef int _register(self, SlaterDeterminant_cpp[uint64_t] det):
        """Row index for ``det``, allocating a new (logical) zero row if absent."""
        cdef cpp_map[SlaterDeterminant_cpp[uint64_t], int].iterator f = self.support.find(det)
        if f != self.support.end():
            return dereference(f).second
        cdef int row = self.n_rows
        self.support[det] = row
        self.row_det.push_back(det)
        self.n_rows += 1
        return row

    cdef void _ensure(self, int need_rows, int need_cols):
        cdef int nr = self.cap_rows
        cdef int nc = self.cap_cols
        while nr < need_rows:
            nr *= 2
        while nc < need_cols:
            nc *= 2
        if nr != self.cap_rows or nc != self.cap_cols:
            self._realloc(nr, nc)

    def append(self, list cols):
        """Append the columns of ``cols`` (a list of ManyBodyState) as new Krylov vectors."""
        cdef int ncol = len(cols)
        if ncol == 0:
            return
        cdef vector[ManyBodyState_cpp*] ptrs
        cdef ManyBodyState ms
        cdef int ci
        for obj in cols:
            ms = <ManyBodyState?>obj
            ptrs.push_back(&ms.v)
        cdef ManyBodyState_cpp.iterator it
        for ci in range(ncol):
            it = ptrs[ci].begin()
            while it != ptrs[ci].end():
                self._register(dereference(it).first)
                preincrement(it)
        self._ensure(self.n_rows, self.n_cols + ncol)
        cdef complex[:, :] Qv = self.Qbuf
        cdef int base = self.n_cols
        cdef int row
        cdef ManyBodyState_cpp.mapped_type cval
        for ci in range(ncol):
            it = ptrs[ci].begin()
            while it != ptrs[ci].end():
                row = dereference(self.support.find(dereference(it).first)).second
                cval = dereference(it).second
                Qv[row, base + ci].real = cval.real()
                Qv[row, base + ci].imag = cval.imag()
                preincrement(it)
        self.n_cols += ncol

    def reort(self, list wp, object cols, int n_passes, object comm):
        """``n_passes`` of CGS2: ``O = Q[:,cols]^H wp`` (Allreduced); ``wp -= Q[:,cols] O``.

        ``cols`` is a list of column indices (the flagged bad blocks) or ``None`` for all
        columns. Returns a new list of ``wp`` ManyBodyStates.
        """
        cdef int p = len(wp)
        if p == 0 or self.n_cols == 0 or n_passes <= 0:
            return wp
        cdef vector[ManyBodyState_cpp*] wptrs
        cdef ManyBodyState ms
        cdef int ci
        for obj in wp:
            ms = <ManyBodyState?>obj
            wptrs.push_back(&ms.v)
        # Register wp's determinants so its components outside the current Q support get rows
        # (Q is zero there -> untouched by the projection, preserved by the scatter).
        cdef ManyBodyState_cpp.iterator it
        for ci in range(p):
            it = wptrs[ci].begin()
            while it != wptrs[ci].end():
                self._register(dereference(it).first)
                preincrement(it)
        self._ensure(self.n_rows, self.n_cols)
        cdef int ns = self.n_rows
        Wd = np.zeros((ns, p), dtype=complex)
        cdef complex[:, :] Wv = Wd
        cdef int row
        cdef ManyBodyState_cpp.mapped_type cval
        for ci in range(p):
            it = wptrs[ci].begin()
            while it != wptrs[ci].end():
                row = dereference(self.support.find(dereference(it).first)).second
                cval = dereference(it).second
                Wv[row, ci].real = cval.real()
                Wv[row, ci].imag = cval.imag()
                preincrement(it)
        if cols is None:
            Qsel = self.Qbuf[:ns, :self.n_cols]
        else:
            Qsel = np.ascontiguousarray(self.Qbuf[:ns][:, cols])
        Qh = np.conj(Qsel.T)
        if comm is not None:
            from mpi4py import MPI
        for _ in range(n_passes):
            O = Qh @ Wd
            if comm is not None:
                comm.Allreduce(MPI.IN_PLACE, O, op=MPI.SUM)
            Wd = Wd - Qsel @ O
        cdef complex[:, :] Wout = Wd
        cdef list out = []
        cdef vector[ManyBodyState_cpp.key_type] keys
        cdef vector[ManyBodyState_cpp.mapped_type] vals
        cdef ManyBodyState new_ms
        cdef double complex z
        for ci in range(p):
            keys.clear()
            vals.clear()
            for row in range(ns):
                z = Wout[row, ci]
                if z.real != 0 or z.imag != 0:
                    keys.push_back(self.row_det[row])
                    vals.push_back(ManyBodyState_cpp.mapped_type(z.real, z.imag))
            new_ms = ManyBodyState()
            new_ms.v = ManyBodyState_cpp(keys, vals)
            out.append(new_ms)
        return out
