# distutils: language = c++

from SlaterDeterminant cimport SlaterDeterminant as SlaterDeterminant_cpp
from ManyBodyState cimport ManyBodyState as ManyBodyState_cpp
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

    void pack_psis(
        const vector[const ManyBodyState_cpp*]& psis,
        int comm_size,
        vector[int64_t]& send_counts,
        vector[uint64_t]& state_buf,
        vector[double]& amp_buf_reim,
        vector[int32_t]& psi_buf)

    void unpack_psis(
        vector[ManyBodyState_cpp*]& psis,
        int comm_size,
        const vector[int64_t]& recv_counts,
        const vector[uint64_t]& state_buf,
        const vector[double]& amp_buf_reim,
        const vector[int32_t]& psi_buf,
        size_t chunks_per_state)

    void pack_psis_fused(
        const vector[const ManyBodyState_cpp*]& psis,
        int comm_size,
        size_t chunks_per_state,
        vector[int64_t]& send_counts,
        vector[char]& send_buf)

    void unpack_psis_fused(
        vector[ManyBodyState_cpp*]& psis,
        int comm_size,
        const vector[int64_t]& recv_counts,
        const vector[char]& recv_buf,
        size_t chunks_per_state)
