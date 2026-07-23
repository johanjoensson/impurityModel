# ===========================================================================
# Slater determinant
# ===========================================================================
# SlaterDeterminant wraps the C++ type (SlaterDeterminant.pxd). The orbital->bit
# convention is MSB-first: orbital i is bit (7 - i) of each byte, so orbital 0 is 0x80.
# ManyBodyState (the block class, p == 1 an ordinary block) lives in _block_state.pxi.

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

    Both operands must be width-1 blocks (``block_inner_scalar`` raises otherwise).
    """
    return block_inner_scalar(a, b)
