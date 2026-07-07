# distutils: language = c++

from SlaterDeterminant cimport SlaterDeterminant
from libcpp.vector cimport vector
from libcpp.complex cimport complex
from libc.stdint cimport uint64_t

cdef extern from "ManyBodyBlockState.h" nogil:
    cdef cppclass ManyBodyBlockState:
        ctypedef SlaterDeterminant[uint64_t] Key
        ctypedef complex[double] Value

        ManyBodyBlockState()
        ManyBodyBlockState(const ManyBodyBlockState&)
        ManyBodyBlockState(vector[Key], vector[Value], size_t)

        size_t width()
        size_t rows()
        bint empty()

        const Key& key(size_t)
        const vector[Key]& keys()
        Value* data()
        Value* row(size_t)
        size_t find_row(const Key&)

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
