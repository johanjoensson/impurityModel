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
        ManyBodyBlockState(const vector[Key]&, const vector[Value]&)

        size_t width()
        size_t rows()
        size_t size()
        bint empty()

        const Key& key(size_t)
        const vector[Key]& keys()
        Value* data()
        Row row(size_t)
        size_t find_row(const Key&)

        bint contains(const Key&)
        Row at(const Key&) except +
        Row operator[](const Key&)
        size_t erase(const Key&)
        void erase_row(size_t)
        void clear()
        void reserve(size_t)
        void swap(ManyBodyBlockState&)

        double norm2()
        double norm()
        double max_norm2()
        size_t count_above(double)
        void truncate(size_t)
        ManyBodyBlockState& add_scaled(const ManyBodyBlockState&, Value) except +

        ManyBodyBlockState operator-()
        ManyBodyBlockState operator+(const ManyBodyBlockState&, const ManyBodyBlockState&)
        ManyBodyBlockState operator-(const ManyBodyBlockState&, const ManyBodyBlockState&)
        ManyBodyBlockState operator*(const ManyBodyBlockState&, Value)
        ManyBodyBlockState operator*(Value, const ManyBodyBlockState&)
        ManyBodyBlockState operator/(const ManyBodyBlockState&, Value)

        void prune_rows(double)
        void keep_rows(const vector[Key]&)
        void row_max_norm2(double*)
        size_t count_rows_in(const vector[Key]&)
        void new_row_max_norm2(const vector[Key]&, vector[double]&)
        ManyBodyBlockState keys_new_above(const vector[Key]&, double)
        ManyBodyBlockState key_union(const ManyBodyBlockState&)
        void merge_keys(const ManyBodyBlockState&)
        void col_norm2(double*)

        bint operator==(const ManyBodyBlockState&)
        bint operator!=(const ManyBodyBlockState&)

    void block_inner(const ManyBodyBlockState&, const ManyBodyBlockState&, ManyBodyBlockState.Value*)
    ManyBodyBlockState block_add_scaled(const ManyBodyBlockState&, const ManyBodyBlockState&, const ManyBodyBlockState.Value*)
    ManyBodyBlockState block_combine_cols(const ManyBodyBlockState&, const ManyBodyBlockState.Value*, size_t)
