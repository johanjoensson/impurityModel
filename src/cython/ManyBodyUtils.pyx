# distutils: language = c++
# cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, freethreading_compatible=True, cdivision=True, cpow=True


from ManyBodyBlockState cimport (
    ManyBodyBlockState as ManyBodyBlockState_cpp,
    block_inner as c_block_inner,
    block_add_scaled as c_block_add_scaled,
    block_combine_cols as c_block_combine_cols,
)
from ManyBodyOperator cimport (
    ManyBodyOperator as ManyBodyOperator_cpp,
    apply_parallel_build as c_apply_parallel_build,
    commutator as commutator_cpp,
    anticommutator as anticommutator_cpp,
)
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


# Krylov-store element type. The store may hold its basis in complex64 (see
# SparseKrylovDense.dtype); the scatter loop is compiled for both widths rather than
# duplicated, and every *arithmetic* consumer (combine / reort) promotes to complex128
# through numpy, so only the storage narrows.
ctypedef fused krylov_t:
    float complex
    double complex

# --- Optional transient-allocation profiling (env-gated, ~zero cost when off) -------
# Shares the BLOCKLANCZOS_PROFILE=1 switch with BlockLanczos.pyx: accumulates the peak
# transient dense-materialization footprint of reorth_cgs2_dense (the PARTIAL bad-block
# projection path). Read with get_manybody_profile().
import os as _os
_MBU_PROF = {}
_MBU_PROF_ON = _os.environ.get("BLOCKLANCZOS_PROFILE") == "1"


def get_manybody_profile():
    """Return a copy of the accumulated transient-allocation counters (bytes / counts)."""
    return dict(_MBU_PROF)


def reset_manybody_profile():
    _MBU_PROF.clear()


def enable_manybody_profile(on=True):
    """Toggle the transient-allocation counters at runtime (equivalent to setting
    BLOCKLANCZOS_PROFILE=1 in the environment before import)."""
    global _MBU_PROF_ON
    _MBU_PROF_ON = bool(on)


cdef extern from "<utility>" namespace "std" nogil:
    SlaterDeterminant_cpp& move(SlaterDeterminant_cpp)
    vector[SlaterDeterminant_cpp[uint64_t]]& move(vector[SlaterDeterminant_cpp[uint64_t]])
    vector[complex_cpp[double]]& move(vector[complex_cpp[double]])


include "_slater_state.pxi"
include "_operator.pxi"
include "_mpi_pack.pxi"
include "_krylov_store.pxi"
include "_block_state.pxi"
