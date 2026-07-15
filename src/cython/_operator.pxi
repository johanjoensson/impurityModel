# ===========================================================================
# Many-body operator
# ===========================================================================
# ManyBodyOperator wraps the C++ operator (ManyBodyOperator.pxd), with the tuple<->int
# process conversions (processes_to_ints / ints_to_processes) and applyOp. The matvec hot
# path (ManyBodyOperator::apply, optionally threaded via -DPARALLEL) lives in the C++ layer.

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

    def apply_block(self, ManyBodyBlockState block, double cutoff = 0):
        """Apply the operator to a shared-support block of p vectors (Phase 2.2).

        The term loop, fermion sign, restriction checks and the accumulator hash
        operation run once per (determinant, term); the p amplitudes are emitted with
        p multiply-adds — near-flat cost in p vs the linear scaling of
        ``apply_multi`` (see the p-scaling baseline in the campaign doc). Per-column
        arithmetic is identical to p independent ``apply`` calls (bit-for-bit at
        ``cutoff=0``). The cutoff keeps whole rows (any column above threshold), so
        the output block retains its shared support.
        """
        cdef ManyBodyBlockState res = ManyBodyBlockState()
        with nogil:
            res.b = self.o.apply(block.b, cutoff)
        return res

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
