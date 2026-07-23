# ===========================================================================
# Slater determinant + many-body state wrappers
# ===========================================================================
# SlaterDeterminant and ManyBodyState wrap the C++ types (ManyBodyState.pxd). The
# orbital->bit convention is MSB-first: orbital i is bit (7 - i) of each byte, so orbital 0
# is 0x80.

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

    def routing_hash(self):
        return self.s.routing_hash()

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

    def memory_bytes(self):
        """Estimated heap bytes held by this state's determinant->amplitude storage.

        Counts the contiguous ``flat_map`` entry array (``sizeof(pair<SlaterDeterminant,
        complex>)`` per entry) plus one heap block per determinant key: the key is a
        ``std::vector<uint64_t>``, so every entry owns a separate allocation of
        ``8 * n_chunks`` payload rounded up to the 16-byte glibc granularity plus an
        8-byte header (minimum chunk 32 bytes). Capacity slack of the entry vector is
        not visible through the map interface and is ignored, so after a growth phase
        the true footprint can be up to ~2x the entry-array part of this estimate.
        """
        cdef size_t n = self.v.size()
        if n == 0:
            return 0
        cdef size_t n_chunks = dereference(self.v.begin()).first.size()
        cdef size_t key_heap = (8 * n_chunks + 8 + 15) & (~<size_t>15)
        if key_heap < 32:
            key_heap = 32
        return n * (sizeof(ManyBodyState_cpp.value_type) + key_heap)

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

    def truncate(self, size_t max_size):
        """
        Truncate the state to keep only the max_size elements with the largest amplitudes.
        """
        with nogil:
            self.v.truncate(max_size)

    def max_norm2(self):
        """
        Return the maximum squared amplitude in the state.
        """
        return self.v.max_norm2()

    def count_above(self, double cutoff2):
        """
        Count how many elements have a squared amplitude strictly greater than cutoff2.
        """
        return self.v.count_above(cutoff2)

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


def apply_global_truncation(ManyBodyState st, int max_size, object comm):
    """
    Truncate the ManyBodyState across all MPI ranks such that the global total
    number of Slater Determinants is at most max_size.
    Uses a distributed binary search to find the exact amplitude threshold.
    """
    import math
    from mpi4py import MPI

    if comm is None or comm.Get_size() == 1 or max_size <= 0:
        if max_size > 0:
            st.truncate(max_size)
        return

    cdef int local_size = len(st)
    cdef int global_size = comm.allreduce(local_size, op=MPI.SUM)
    if global_size <= max_size:
        return

    # Distributed binary search for the threshold cutoff2
    cdef double high = comm.allreduce(st.max_norm2(), op=MPI.MAX)
    cdef double low = 0.0
    cdef double mid
    cdef int global_count

    # Keep iterating until the range of possible values is small enough
    while abs(high - low) > 1e-8:
        mid = (low + high) * 0.5
        global_count = comm.allreduce(st.count_above(mid), op=MPI.SUM)
        # Keep too many, mid is lower than the required value
        if global_count > max_size:
            low = mid
        # Keep too few, mid is higher than the required value
        # If you keep just the right amount, keep increasing the lower bound until the range converges.
        else:
            high = mid

    st.prune(math.sqrt(high))


def inner(a, b):
    """
    Compute the inner product of many-body states: <a|b>.

    ``a``/``b`` may also be width-1 ``ManyBodyBlockState``s (``block_inner_scalar``,
    the block counterpart of this function) -- the boundary is representation-
    transparent, matching ``ManyBodyOperator.__call__``'s dispatch.
    """
    if isinstance(a, ManyBodyBlockState) or isinstance(b, ManyBodyBlockState):
        if not (isinstance(a, ManyBodyBlockState) and isinstance(b, ManyBodyBlockState)):
            raise TypeError(f"inner: mixed operand types {type(a)!r} and {type(b)!r}")
        return block_inner_scalar(<ManyBodyBlockState>a, <ManyBodyBlockState>b)
    cdef ManyBodyState sa = <ManyBodyState?>a
    cdef ManyBodyState sb = <ManyBodyState?>b
    cdef ManyBodyState_cpp.mapped_type res
    with nogil:
        res = inner_cpp(sa.v, sb.v)
    return res
