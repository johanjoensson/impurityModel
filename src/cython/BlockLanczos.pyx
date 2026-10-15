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

* ``block_lanczos`` – core iteration loop producing the block-tridiagonal
  Lanczos representation :math:`T = Q^\\dagger H Q` of the Hamiltonian.

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

Reorthogonalization (the boolean ``reort``):

* ``False``     – no reorthogonalization.
* ``True``      – full Gram-Schmidt against all previous blocks (2
  passes).
"""

# The recurrence itself -- ``lanczos_step`` and the ManyBodyState driver ``block_lanczos`` -- lives in
# BlockLanczosCore.pyx next to the representation-dispatching primitives it is written against; this
# module re-exports it so the existing `from impurityModel.ed.BlockLanczos import ...` paths keep working.
from impurityModel.ed.BlockLanczosCore import (  # noqa: F401
    block_lanczos,
    lanczos_step,
)
