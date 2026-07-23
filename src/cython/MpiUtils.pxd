# distutils: language = c++

from SlaterDeterminant cimport SlaterDeterminant as SlaterDeterminant_cpp
from ManyBodyBlockState cimport ManyBodyBlockState as ManyBodyBlockState_cpp
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, int64_t, int32_t


cdef extern from "MpiUtils.h" namespace "mpi_utils" nogil:
    void pack_determinants(
        const vector[SlaterDeterminant_cpp[uint64_t]]& dets,
        int comm_size,
        vector[int64_t]& send_counts,
        vector[uint64_t]& state_buf)

    vector[vector[SlaterDeterminant_cpp[uint64_t]]] unpack_determinants(
        int comm_size,
        const vector[int64_t]& recv_counts,
        const vector[uint64_t]& state_buf,
        size_t chunks_per_state)

    void pack_block_count(
        const ManyBodyBlockState_cpp& block,
        int comm_size,
        vector[int64_t]& send_counts,
        vector[int]& owners)

    void pack_block_fill(
        const ManyBodyBlockState_cpp& block,
        int comm_size,
        size_t chunks_per_state,
        const vector[int64_t]& send_counts,
        const vector[int]& owners,
        char* send_buf)

    ManyBodyBlockState_cpp unpack_block_fused(
        int comm_size,
        size_t width,
        const vector[int64_t]& recv_counts,
        const char* recv_buf,
        size_t chunks_per_state)
