# distutils: language = c++

from SlaterDeterminant cimport SlaterDeterminant
from libcpp.vector cimport vector
from libcpp.complex cimport complex
from libc.stdint cimport uint64_t

cdef extern from "ManyBodyBlockState.h" nogil:
    # One determinant's amplitudes: std::span where the standard library has it,
    # the header's stand-in otherwise. Declared as a class so Cython can name the
    # return type of row(); the generated C++ is the same either way.
    cdef cppclass RowSpan[T]:
        RowSpan()
        RowSpan(T*, size_t)
        T* data()
        size_t size()
        bint empty()
        T& operator[](size_t)

    cdef cppclass ManyBodyBlockState:
        ctypedef SlaterDeterminant[uint64_t] Key
        ctypedef complex[double] Value
        ctypedef RowSpan[complex[double]] Row

        ManyBodyBlockState()
        ManyBodyBlockState(const ManyBodyBlockState&)
        ManyBodyBlockState(size_t)
        ManyBodyBlockState(vector[Key], vector[Value], size_t)

        @staticmethod
        ManyBodyBlockState from_unsorted(const vector[Key]&, const vector[Value]&, size_t) except +

        size_t width()
        size_t rows()
        size_t size()
        size_t max_size()
        bint empty()

        const Key& key(size_t)
        const vector[Key]& keys()
        Value* data()
        Row row(size_t)
        size_t find_row(const Key&)

        bint contains(const Key&)
        Row at(const Key&) except +
        Row operator[](const Key&) except +
        size_t erase(const Key&) except +
        void erase_row(size_t)
        void clear()
        void reserve(size_t) except +
        void swap(ManyBodyBlockState&)

        double norm2()
        double norm()
        double max_norm2()
        size_t count_above(double)
        void truncate(size_t) except +
        ManyBodyBlockState& add_scaled(const ManyBodyBlockState&, Value) except +

        # True in-place scaling: values change, the row layout does not, so (unlike
        # add_scaled/+=/-=, which rebuild storage over the union support) a live Row
        # survives these. Plain-named (not operator*=/operator/=) because Cython's
        # cppclass declarations do not support compound-assignment operators.
        void scale(Value)
        void scale_inv(Value)

        # Every one of these builds a temporary via the copy constructor and/or
        # reallocates through add_scaled, either of which can raise on a width
        # mismatch or bad_alloc; without `except +` such a C++ exception would
        # unwind past Cython's generated code with no handler and abort the
        # process (see the commit fixing this).
        ManyBodyBlockState operator-() except +
        ManyBodyBlockState operator+(const ManyBodyBlockState&, const ManyBodyBlockState&) except +
        ManyBodyBlockState operator-(const ManyBodyBlockState&, const ManyBodyBlockState&) except +
        ManyBodyBlockState operator*(const ManyBodyBlockState&, Value) except +
        ManyBodyBlockState operator*(Value, const ManyBodyBlockState&) except +
        ManyBodyBlockState operator/(const ManyBodyBlockState&, Value) except +

        void prune_rows(double)
        void keep_rows(const vector[Key]&) except +
        void row_max_norm2(double*)
        size_t count_rows_in(const vector[Key]&)
        void new_row_max_norm2(const vector[Key]&, vector[double]&) except +
        ManyBodyBlockState keys_new_above(const vector[Key]&, double) except +
        ManyBodyBlockState key_union(const ManyBodyBlockState&) except +
        void merge_keys(const ManyBodyBlockState&) except +
        void col_norm2(double*)
        ManyBodyBlockState select_cols(const vector[size_t]&) except +

        bint operator==(const ManyBodyBlockState&)
        bint operator!=(const ManyBodyBlockState&)

    void block_inner(const ManyBodyBlockState&, const ManyBodyBlockState&, ManyBodyBlockState.Value*)
    ManyBodyBlockState block_add_scaled(const ManyBodyBlockState&, const ManyBodyBlockState&, const ManyBodyBlockState.Value*) except +
    ManyBodyBlockState block_combine_cols(const ManyBodyBlockState&, const ManyBodyBlockState.Value*, size_t) except +
