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
    and annihilation process sequences with corresponding amplitudes. A term is keyed
    by its process tuple in product (left-to-right) order, e.g.
    ``((i, 'c'), (j, 'a'))`` for :math:`c^\\dagger_i c_j`; the empty tuple ``()`` keys
    the constant (identity) term.

    Stored terms are kept in **canonical normal order**: creations before
    annihilations, each group ascending in orbital, Pauli-vanishing terms dropped and
    terms equal up to ordering merged. Constructors and all algebra maintain this, so
    ``to_dict()`` reports the canonical form rather than the terms as they were
    written -- e.g. ``{((0, 'a'), (0, 'c')): 1}`` reads back as
    ``{(): 1, ((0, 'c'), (0, 'a')): -1}``, the same operator. Only ``__setitem__`` can
    break the invariant; see :meth:`canonicalize`.

    Occupation restrictions (:meth:`set_restrictions` /
    :meth:`set_weighted_restrictions`) are a property of the operator *object*, not of
    the operator algebra: they are NOT propagated through ``+``, ``-``, ``*`` or any
    other operation, so a result must have its restrictions set explicitly (this is
    what e.g. ``gf_solvers`` and ``manybody_basis.Basis`` do).
    """
    # Opt out of numpy's ufunc dispatch. Without this a numpy scalar on the left of a
    # mixed expression (`np.complex128(z) - hOp`, which is what a frequency taken from a
    # mesh actually is) tries to broadcast the operator as an array element and raises
    # _UFuncNoLoopError instead of deferring; setting it to None is numpy's documented
    # way to make it return NotImplemented so __rsub__/__radd__/__rmul__ run.
    __array_ufunc__ = None

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

    @staticmethod
    def identity(scale=1.0) -> ManyBodyOperator:
        """
        Return ``scale`` times the identity operator.

        The identity is the single empty-string term ``{(): scale}``; note that a
        default-constructed ``ManyBodyOperator()`` is the *zero* operator, not this.
        """
        cdef ManyBodyOperator res = ManyBodyOperator()
        cdef ManyBodyOperator_cpp.mapped_type s = scale
        with nogil:
            res.o.set_constant(s)
        return res

    @property
    def constant(self) -> complex:
        """
        Coefficient of the identity: the amplitude of the empty term ``()``.

        Writable. Setting it to zero removes the term.
        """
        return self.o.constant()

    @constant.setter
    def constant(self, ManyBodyOperator_cpp.mapped_type value):
        with nogil:
            self.o.set_constant(value)

    def __add__(self, other) -> ManyBodyOperator:
        cdef ManyBodyOperator res = ManyBodyOperator()
        cdef ManyBodyOperator rhs
        cdef ManyBodyOperator_cpp.mapped_type s
        if isinstance(other, ManyBodyOperator):
            rhs = <ManyBodyOperator>other
            with nogil:
                res.o = self.o + rhs.o
        else:
            s = other
            with nogil:
                res.o = self.o + s
        return res

    def __radd__(self, other) -> ManyBodyOperator:
        return self.__add__(other)

    def __iadd__(self, other) -> ManyBodyOperator:
        cdef ManyBodyOperator rhs
        cdef ManyBodyOperator_cpp.mapped_type s
        if isinstance(other, ManyBodyOperator):
            rhs = <ManyBodyOperator>other
            with nogil:
                self.o = self.o + rhs.o
        else:
            s = other
            with nogil:
                self.o = self.o + s
        return self

    def __sub__(self, other) -> ManyBodyOperator:
        cdef ManyBodyOperator res = ManyBodyOperator()
        cdef ManyBodyOperator rhs
        cdef ManyBodyOperator_cpp.mapped_type s
        if isinstance(other, ManyBodyOperator):
            rhs = <ManyBodyOperator>other
            with nogil:
                res.o = self.o - rhs.o
        else:
            s = other
            with nogil:
                res.o = self.o - s
        return res

    def __rsub__(self, other) -> ManyBodyOperator:
        """``scalar - op``, i.e. ``scalar*I - op`` (used for resolvents ``z - H``)."""
        cdef ManyBodyOperator res = ManyBodyOperator()
        cdef ManyBodyOperator_cpp.mapped_type s = other
        with nogil:
            res.o = s - self.o
        return res

    def __isub__(self, other) -> ManyBodyOperator:
        cdef ManyBodyOperator rhs
        cdef ManyBodyOperator_cpp.mapped_type s
        if isinstance(other, ManyBodyOperator):
            rhs = <ManyBodyOperator>other
            with nogil:
                self.o = self.o - rhs.o
        else:
            s = other
            with nogil:
                self.o = self.o - s
        return self

    def __neg__(self) -> ManyBodyOperator:
        res = ManyBodyOperator()
        with nogil:
            res.o = -self.o
        return res

    def __mul__(self, other) -> ManyBodyOperator:
        """
        Scale by a scalar, or compose with another operator.

        ``A * B`` is composition: ``(A * B)(psi) == A(B(psi))``. The result is
        canonicalized, so terms that cancel do cancel. Cost is ``len(A) * len(B)`` term
        pairs *before* that cancellation, so compose small operators -- squaring a full
        Hamiltonian is not tractable. ``A @ B`` is a synonym.
        """
        cdef ManyBodyOperator res = ManyBodyOperator()
        cdef ManyBodyOperator rhs
        cdef ManyBodyOperator_cpp.mapped_type s
        if isinstance(other, ManyBodyOperator):
            rhs = <ManyBodyOperator>other
            with nogil:
                res.o = self.o * rhs.o
        else:
            s = other
            with nogil:
                res.o = self.o * s
        return res

    def __matmul__(self, other) -> ManyBodyOperator:
        """Operator composition; a synonym for ``*`` between two operators."""
        return self.__mul__(other)

    def __imul__(self, other) -> ManyBodyOperator:
        cdef ManyBodyOperator rhs
        cdef ManyBodyOperator_cpp.mapped_type s
        if isinstance(other, ManyBodyOperator):
            rhs = <ManyBodyOperator>other
            with nogil:
                self.o = self.o * rhs.o
        else:
            s = other
            with nogil:
                self.o = self.o * s
        return self

    def __rmul__(self, ManyBodyOperator_cpp.mapped_type s) -> ManyBodyOperator:
        return self * s

    def __pow__(self, int n, mod=None) -> ManyBodyOperator:
        """``op ** n``: the n-fold composition. ``n == 0`` gives the identity."""
        if mod is not None:
            raise TypeError("ManyBodyOperator does not support modular exponentiation")
        if n < 0:
            raise ValueError("ManyBodyOperator cannot be raised to a negative power (no inverse)")
        cdef ManyBodyOperator res = ManyBodyOperator()
        cdef unsigned int k = n
        with nogil:
            res.o = self.o.power(k)
        return res

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

    def commutator(self, ManyBodyOperator other) -> ManyBodyOperator:
        """
        Return the commutator ``[self, other] = self*other - other*self``.

        Term pairs acting on disjoint orbitals are skipped without being formed --
        exactly, not within a tolerance: two strings on disjoint orbitals satisfy
        ``a b = (-1)^(len_a * len_b) b a``, so the pair cancels unless both strings have
        odd length. That is what makes ``H.commutator(c_i)`` cost a pass over the terms
        touching orbital ``i`` rather than ``len(H)`` products.
        """
        cdef ManyBodyOperator res = ManyBodyOperator()
        with nogil:
            res.o = commutator_cpp(self.o, other.o)
        return res

    def anticommutator(self, ManyBodyOperator other) -> ManyBodyOperator:
        """
        Return the anticommutator ``{self, other} = self*other + other*self``.

        Same disjoint-support skip as :meth:`commutator`, with the opposite parity rule:
        a disjoint pair cancels exactly when both strings have odd length (which is why
        ``{c_i, c_j} == 0`` for ``i != j`` costs nothing to discover).
        """
        cdef ManyBodyOperator res = ManyBodyOperator()
        with nogil:
            res.o = anticommutator_cpp(self.o, other.o)
        return res

    def adjoint(self) -> ManyBodyOperator:
        """
        Return the Hermitian adjoint ``self^dagger``.

        Each operator string is reversed and each factor exchanged between creation and
        annihilation on the same orbital, with the coefficient conjugated. The
        operator-level counterpart of :func:`operator_algebra.daggerOp`, which does the
        same thing on a plain term dict for the Hamiltonian-construction path.
        """
        cdef ManyBodyOperator res = ManyBodyOperator()
        with nogil:
            res.o = self.o.adjoint()
        return res

    def dagger(self) -> ManyBodyOperator:
        """Alias for :meth:`adjoint`."""
        return self.adjoint()

    def is_hermitian(self, double tol=1e-12) -> bool:
        """Whether the operator equals its adjoint to within ``tol``."""
        return self.o.is_hermitian(tol)

    def hermitian_part(self) -> ManyBodyOperator:
        """Return ``(self + self.adjoint()) / 2``."""
        cdef ManyBodyOperator res = ManyBodyOperator()
        with nogil:
            res.o = self.o.hermitian_part()
        return res

    def orbitals(self) -> tuple:
        """Sorted spin-orbital indices this operator acts on."""
        return tuple(self.o.orbitals())

    def body_rank(self) -> int:
        """
        Highest n-body rank present: 0 for a pure constant, 1 for a one-body operator,
        2 for a Coulomb term. Computed as ``ceil(len / 2)`` over the term strings, so a
        bare ``c_i`` counts as one-body.
        """
        return self.o.body_rank()

    def approx_equal(self, ManyBodyOperator other, double tol=1e-12) -> bool:
        """
        Whether every coefficient agrees with ``other``'s to within ``tol``.

        ``==`` compares complex coefficients exactly, which is rarely what you want
        after a chain of algebra.
        """
        return self.o.approx_equal(other.o, tol)

    def prune(self, double tol):
        """
        Drop every term with ``abs(coefficient) <= tol``, in place.

        Products and commutators canonicalize their result, which already removes terms
        that cancel exactly. ``+`` and ``-`` do not: they merge coefficients but keep the
        key, so ``A - A`` retains its terms with zero amplitude. Prune when that matters.
        """
        with nogil:
            self.o.prune(tol)

    def canonicalize(self):
        """
        Rewrite the stored terms into canonical normal order, in place.

        Only ever needed after mutating the operator through ``__setitem__``, which is
        the one route that can introduce a non-normal-ordered term; every constructor
        and every algebraic operation leaves the operator canonical already.

        Representation change only -- the operator's action on any state is unchanged.
        """
        with nogil:
            self.o.canonicalize()

    def is_canonical(self) -> bool:
        """Whether the stored terms are known to be in canonical normal order."""
        return self.o.is_canonical()

    def set_normal_ordering(self, bint enable):
        """
        Enable/disable build-time normal ordering of the apply() representation.

        Only observable on an operator made non-canonical through ``__setitem__``;
        see :meth:`canonicalize`. Representation change only; the operator's action
        is unchanged. Default enabled.
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


def commutator(ManyBodyOperator A, ManyBodyOperator B) -> ManyBodyOperator:
    """
    Return ``[A, B] = A*B - B*A``. See :meth:`ManyBodyOperator.commutator`.
    """
    return A.commutator(B)


def anticommutator(ManyBodyOperator A, ManyBodyOperator B) -> ManyBodyOperator:
    """
    Return ``{A, B} = A*B + B*A``. See :meth:`ManyBodyOperator.anticommutator`.
    """
    return A.anticommutator(B)
