# distutils: language = c++
# Compiler directives are centralized in setup.py (_DIRECTIVES), so a build mode can
# turn Cython's runtime checks on: IMPURITYMODEL_BUILD=debug. A `# cython:` header here
# would override that and silently defeat it.
# Permissive on purpose, unlike BlockLanczos.pyx -- see "Compiler directives" in
# doc/lanczos_invariants.md.

"""Block Lanczos kernel for numpy arrays / scipy sparse operators (MPI row-block).

The shared numerical layer (every tolerance constant, the Paige-Simon
``estimate_orthonormality`` estimator, the block-tridiagonal eigensolvers, and the
representation-dispatching block primitives / ``apply_reort``) lives in
``BlockLanczosCore.pyx``, imported here as a re-export block so every existing
``from impurityModel.ed.BlockLanczosArray import ...`` site keeps resolving. The
ManyBodyState kernel in ``BlockLanczos.pyx`` imports the same names directly from
``BlockLanczosCore``, not from here -- both kernels import downward from the shared
layer, never from each other. Both build a block-tridiagonal ``T`` (diagonal blocks
``alpha``, off-diagonal blocks ``beta``) whose eigenpairs approximate those of the
operator; the TRLM drivers restart it.

Reorthogonalization (the boolean ``reort``), what each setting guarantees:

* ``False``     — no reorthogonalization. Cheapest; orthogonality is allowed to
  decay and ghost (spurious duplicate) eigenvalues may appear. Eigenvalues still
  converge for well-separated spectra. The restart machinery in TRLM
  re-orthogonalizes regardless, so NONE only relaxes the *inner* recurrence.
* ``True``      — double-pass Gram–Schmidt against the whole basis every step.
  Most robust; ``‖QᴴQ − I‖`` stays at ~machine precision.

Every mode retains all ``p * m`` Krylov columns. That is the dominant allocation of a
long Green's-function run, and it cannot be compressed away: reorthogonalization needs
the *converged Ritz vectors*, and a Ritz vector converging at step ``m`` has weight on
every block, including any an earlier compression would have discarded. Bounding the
retained basis therefore needs secondary storage, not a smaller basis — see the
"Ritz compression is unsound" section of ``doc/plans/blocklanczos_reort_memory.md``.

Module thresholds (all derived from machine ``eps``; see definitions below):
``DEFLATE_TOL`` (relative rank floor on the block's singular values),
``BREAKDOWN_TOL`` (invariant-subspace detection). The
deflation floors are defined in the ``TSQR`` module that applies them.

Orthonormalization goes through ``TSQR`` via ``block_tsqr`` (``BlockLanczosCore.pyx``'s
``_block_ops.pxi``, dispatch-generic across block representations), reached here through
the shared ``factor_residual`` step-policy helper: the residual block's
triangular factor is computed from the block itself, never from the Gram matrix
``Wp^H Wp``, which is what makes it stable at any conditioning and its singular values
trustworthy.

Deflation policy: when a Lanczos block is rank-deficient (a singular value below the
``DEFLATE_TOL`` floor), the block size shrinks (EA16 shrinking-block, Meerbergen & Scott,
RAL-TR-2000-011) rather than zero-padding or terminating, so ``beta`` becomes rectangular
and ``T`` carries variable-size blocks; the recurrence keeps converging.
"""

import numpy as np
cimport numpy as np
import scipy.linalg as la
import scipy.sparse as sps


from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.solver_trace import note as _trace_note

# Shared numerical layer: every tolerance constant, the Paige-Simon estimator,
# the block-tridiagonal eigensolvers, and the representation-dispatching block
# primitives all now live in BlockLanczosCore.pyx -- re-exported here so every existing
# `from impurityModel.ed.BlockLanczosArray import ...` site (production and test) keeps
# resolving unchanged. Put new imports needed only by THIS module's own code above this
# block; put anything that belongs to the shared layer in BlockLanczosCore.pyx instead.
from impurityModel.ed.BlockLanczosCore import (  # noqa: F401
    BETA_BLOWUP_FACTOR,
    DEFAULT_EIGEN_TOL,
    divergence_guard,
    _build_full_T,
    _build_banded_lower,
    eigh_block_tridiagonal,
    eigsh,
    is_array,
    block_inner,
    block_apply,
    block_add_scaled,
    block_combine,
    block_orthogonalize,
    block_normalize,
    BlockBreakdown,
    block_tsqr,
    apply_full_reort,
    block_lanczos,
    _matmul_nogil_test,
    block_combine_array,
    block_orthogonalize_array,
    check_divergence,
    factor_residual,
    pack_lanczos_result,
)

# The machine constant and the deflation floors are defined once, in the TSQR leaf that
# applies them; a second literal here would be a second source of truth. ``DEFLATE_TOL``
# itself is only read by callers, who import it straight from ``TSQR`` (cython-lint has no
# noqa escape for a pure re-export).
from impurityModel.ed.TSQR import (
    EPS,
    DEFLATE_EVAL_TOL,
    BREAKDOWN_TOL,
    robust_svd,
    spectral_norm,
)

