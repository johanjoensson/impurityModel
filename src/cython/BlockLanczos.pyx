# distutils: language = c++
# Compiler directives are centralized in setup.py (_DIRECTIVES), so a build mode can
# turn Cython's runtime checks on: IMPURITYMODEL_BUILD=debug. A `# cython:` header here
# would override that and silently defeat it.
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
* ``FULL``      – full Gram-Schmidt against all previous blocks (2
  passes).
"""

import numpy as np
from impurityModel.ed import config
from impurityModel.ed.manybody_basis import collective_top_k_mask
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    block_add_scaled_cy,
    block_inner_cy,
    SparseKrylovDense,
)
from mpi4py import MPI

cimport numpy as np

# The shared numerical layer (Reort, tolerances, the
# block-tridiagonal eigensolvers, and the representation-dispatching block
# primitives / apply_full_reort) lives in BlockLanczosCore.pyx -- imported directly here,
# not transitively through BlockLanczosArray, so both kernels import downward from
# the shared layer and neither depends on the other.
from impurityModel.ed.BlockLanczosCore import (
    apply_full_reort,
    resolve_reort,
    block_cols,
    block_combine,
    block_tsqr,
    check_divergence,
    divergence_guard,
    factor_residual,
    apply_and_redistribute,
    matvec_cut_after_sum,
    pack_lanczos_result,
    Reort,
    EPS,
)
from impurityModel.ed.TSQR import robust_svd, spectral_norm

# block_normalize_sparse used to live here as a thin block_tsqr wrapper for the
# ManyBodyState/list representations; it and its array-kernel counterpart
# (block_normalize_array) were the same 5-line body twice, now absorbed into Core's
# single block_normalize (imported above), which dispatches on representation itself.


include "_lanczos_step.pxi"

# TRLM's restart-layer business logic (the path-agnostic `_trlm_core`, locking and
# result assembly) is plain Python in ed/trlm.py -- it has no Cython constructs and
# profiling showed ~97.7% of runtime inside `block_lanczos_cy` above, ~2.3% in this
# restart glue. The module imports `block_lanczos_cy` back from here, function-locally
# (inside its MBS entry point, not at module level): the two-way name dependency
# below would otherwise cycle at the first `pip install -e .`. Re-exported here so the existing
# `from impurityModel.ed.BlockLanczos import ...` import paths keep working.
from impurityModel.ed.trlm import (  # noqa: F401
    _thick_restart_block_lanczos_array,
    _trlm_core,
    _trlm_extract,
    thick_restart_block_lanczos,
    thick_restart_block_lanczos_cy,
)
