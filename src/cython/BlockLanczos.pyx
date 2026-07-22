# distutils: language = c++
# cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, freethreading_compatible=True, cdivision=True, cpow=True
"""
BlockLanczos.pyx
================
Parallel Block Lanczos eigensolver implemented in Cython.

This module provides:

* ``block_lanczos_cy`` – core iteration loop producing the block-tridiagonal
  Lanczos representation :math:`T = Q^\\dagger H Q` of the Hamiltonian.
* ``thick_restart_block_lanczos_cy`` – thick-restart (TRLM) wrapper that
  restarts the Krylov subspace while retaining the best Ritz pairs.
* ``implicitly_restarted_block_lanczos_cy`` – implicitly-restarted (IRLM)
  wrapper that applies :math:`(m-k)` implicit QR shifts to compress the
  subspace back to :math:`k` blocks before continuing.

All distributed inner products use ``mpi4py``'s Python API
(``comm.Allreduce``) over small :math:`p \\times p` matrices.  Heavy
matvec work is delegated to ``ManyBodyOperator.apply_multi`` which
releases the GIL internally.

Notes
-----
SlaterDeterminant distribution:
    Each rank owns the SDs with ``hash(sd) % mpi_size == rank``.
    This is maintained by ``basis.redistribute_psis()`` after every
    ``apply_multi`` call.

Pre-allocation:
    ``alphas`` and ``betas`` arrays are pre-allocated before the loop
    and sliced at return time; no numpy allocation occurs inside the
    Lanczos iteration body.

Reorthogonalization modes (``Reort`` enum from ``lanczos.py``):

* ``NONE``      – no reorthogonalization.
* ``PARTIAL``   – Paige-Simon W-matrix estimator; reorthogonalize only
  when the estimated overlap exceeds :math:`\\sqrt{\\varepsilon}`.
* ``SELECTIVE`` – as PARTIAL but additionally projects against converged
  Ritz vectors.
* ``FULL``      – full Gram-Schmidt against all previous blocks (2
  passes).
* ``PERIODIC``  – full reorthogonalization every ``reort_period`` steps.
"""

import math

import numpy as np
import scipy.linalg as sp
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    ManyBodyBlockState,
    add_scaled_multi,
    block_add_scaled_cy,
    block_inner_cy,
    inner_multi,
    SparseKrylovDense,
)
from mpi4py import MPI

cimport numpy as np

from impurityModel.ed.BlockLanczosArray import estimate_orthonormality, _build_full_T, eigh_block_tridiagonal
from impurityModel.ed.BlockLanczosArray import (
    apply_reort,
    divergence_guard,
    resolve_reort,
    selective_orthogonalize,
    is_array,
    block_apply,
    block_combine,
    block_inner,
    block_orthogonalize,
    block_normalize,
    block_tsqr,
    Reort,
    EPS,
    REORT_TOL,
    BAD_BLOCK_TOL,
    RESTART_ORTH_TOL,
    BREAKDOWN_TOL,
    DEFLATE_EVAL_TOL,
)

# --- Optional per-step profiling (env-gated, ~zero cost when off) -------------------
# Set BLOCKLANCZOS_PROFILE=1 to accumulate wall time per sub-operation of the sparse
# block-Lanczos step (matvec / recurrence-LA / W-estimator / triggered reort / TSQR /
# convergence monitor). Read with get_block_lanczos_profile().
import os as _os
import time as _time
_PROF = {}
_PROF_ON = _os.environ.get("BLOCKLANCZOS_PROFILE") == "1"


def get_block_lanczos_profile():
    """Return a copy of the accumulated per-operation timings (seconds) and call counts."""
    return dict(_PROF)


def reset_block_lanczos_profile():
    _PROF.clear()


def enable_block_lanczos_profile(on=True):
    """Toggle the per-step profiling accumulators at runtime (equivalent to setting
    BLOCKLANCZOS_PROFILE=1 in the environment before import)."""
    global _PROF_ON
    _PROF_ON = bool(on)


cdef inline void _prof_acc(str key, double t0):
    if _PROF_ON:
        _PROF[key] = _PROF.get(key, 0.0) + (_time.perf_counter() - t0)
        _PROF[key + "#n"] = _PROF.get(key + "#n", 0.0) + 1.0


cpdef list block_combine_sparse(list Q, np.ndarray Y, double slaterWeightMin=0.0):
    """No production caller reaches this since Phase 5 step 6 (block_combine's dispatcher
    now raises on a bare list[ManyBodyState] instead of falling through here) -- kept as
    test_block_state.py's / test_krylov_store.py's bit-for-bit oracle for
    ManyBodyBlockState.combine_columns."""
    cdef int n_out = Y.shape[1]
    cdef list out = [ManyBodyState() for _ in range(n_out)]
    add_scaled_multi(out, Q, np.ascontiguousarray(Y, dtype=complex))
    if slaterWeightMin > 0:
        for st in out:
            st.prune(slaterWeightMin)
    return out


cpdef tuple block_orthogonalize_sparse(list wp, list Q, object overlaps=None, object comm=None):
    """No production caller reaches this since Phase 5 step 6 (block_orthogonalize's
    dispatcher now raises on a bare list[ManyBodyState] instead of falling through here) --
    kept as test_block_state.py's oracle for the ManyBodyBlockState arm."""
    if overlaps is None:
        overlaps = inner_multi(Q, wp)
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
    add_scaled_multi(wp, Q, -overlaps)
    return wp, overlaps


cpdef tuple block_normalize_sparse(object wp, bint mpi=False, object comm=None, double slaterWeightMin=0.0):
    """``block_normalize_array``'s counterpart for a list of ``ManyBodyState`` or a
    ``ManyBodyBlockState`` (see it for the breakdown convention). Untyped (not ``list``):
    ``block_tsqr`` itself dispatches on ``ManyBodyBlockState`` vs. list, so this is a pure
    passthrough for either representation, not list-only."""
    q_next, beta_j, active_k, _ = block_tsqr(wp, mpi, comm, 1.0, slaterWeightMin)
    if active_k <= 0:
        raise ValueError("Block collapsed to zero rank")
    return q_next, beta_j

# Orthonormalization is shared: block_tsqr (imported from BlockLanczosArray, which includes
# _reort.pxi) dispatches on the block representation, so this kernel and the array kernel run
# the same factorization and the same EA16 shrinking-block deflation policy.


include "_lanczos_step.pxi"
include "_trlm.pxi"
include "_irlm.pxi"
