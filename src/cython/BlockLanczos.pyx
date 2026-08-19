# distutils: language = c++
# cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, freethreading_compatible=True, cdivision=True, cpow=True
# Strict on purpose, unlike BlockLanczosArray.pyx -- see "Compiler directives" in
# doc/lanczos_invariants.md. wraparound=False means _lanczos_step.pxi's cdef-typed
# lists must use lst[len(lst) - 1], never lst[-1].
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

import numpy as np
import scipy.linalg as sp
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    apply_global_truncation,
    block_add_scaled_cy,
    block_inner_cy,
    SparseKrylovDense,
)
from mpi4py import MPI

cimport numpy as np

# The shared numerical layer (Reort, tolerances, estimate_orthonormality, the
# block-tridiagonal eigensolvers, and the representation-dispatching block
# primitives / apply_reort) lives in BlockLanczosCore.pyx -- imported directly here,
# not transitively through BlockLanczosArray, so both kernels import downward from
# the shared layer and neither depends on the other.
from impurityModel.ed.BlockLanczosCore import (
    estimate_orthonormality,
    _build_full_T,
    eigh_block_tridiagonal,
    apply_reort,
    divergence_guard,
    resolve_reort,
    selective_orthogonalize,
    is_array,
    block_cols,
    block_apply,
    block_combine,
    block_inner,
    block_orthogonalize,
    block_normalize,
    block_tsqr,
    check_divergence,
    factor_residual,
    finish_reort,
    seed_w_estimator,
    omega_floor,
    locked_reort_step,
    pack_lanczos_result,
    Reort,
    EPS,
    REORT_TOL,
    REORT_PERIOD,
    BAD_BLOCK_TOL,
    RESTART_ORTH_TOL,
    BREAKDOWN_TOL,
    DEFLATE_EVAL_TOL,
    DEFAULT_EIGEN_TOL,
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


# block_normalize_sparse used to live here as a thin block_tsqr wrapper for the
# ManyBodyState/list representations; it and its array-kernel counterpart
# (block_normalize_array) were the same 5-line body twice, now absorbed into Core's
# single block_normalize (imported above), which dispatches on representation itself.


include "_lanczos_step.pxi"

# TRLM/IRLM's restart-layer business logic (the path-agnostic `_trlm_core` /
# `_irlm_core`, locking, purging, and result assembly) is plain Python in
# ed/trlm.py and ed/irlm.py -- it has no Cython constructs and profiling showed
# ~97.7% of MBS IRLM runtime inside `block_lanczos_cy` above, ~2.3% in this restart
# glue. Both modules import `block_lanczos_cy` back from here, function-locally
# (inside their MBS entry point, not at module level): the two-way name dependency
# below would otherwise cycle at the first `pip install -e .` depending on which
# module happens to import first. Re-exported here so the existing
# `from impurityModel.ed.BlockLanczos import ...` import paths keep working.
from impurityModel.ed.trlm import (  # noqa: F401
    _thick_restart_block_lanczos_array,
    _trlm_core,
    _trlm_extract,
    thick_restart_block_lanczos,
    thick_restart_block_lanczos_cy,
)
from impurityModel.ed.irlm import (  # noqa: F401
    _assemble_results,
    _implicitly_restarted_block_lanczos_array,
    _implicitly_restarted_block_lanczos_manybody,
    _implicitly_restarted_block_lanczos_cy_manybody as implicitly_restarted_block_lanczos_cy,
    _irlm_core,
    implicitly_restarted_block_lanczos,
)
