"""Shared, derived numerical tolerances for tests.

Machine-relative, not hard-coded literals: a test asserting agreement tighter than the
arithmetic actually delivers passes on one BLAS/compiler/MPI-reduction-order and fails
on another. Each helper below states the error model it derives from; use them instead
of a fresh ``atol=1e-12``-style literal whenever the compared quantity comes from a
sum/inner-product/eigensolve rather than an exact algebraic identity on a tiny dense
array (those may legitimately use a literal machine-epsilon-scale tolerance directly).
"""

import numpy as np

EPS = np.finfo(float).eps


def sum_rtol(n, dtype=np.complex128, c=8.0):
    """Relative tolerance for an n-term positive reduction (a norm, a sum of squares).

    Summing n nonnegative terms in floating point accumulates rounding error that
    grows like ``n * eps`` in the worst case; ``c`` is a small safety factor above
    that worst-case bound (reduction order -- BLAS vs a Python loop, tree vs
    sequential -- changes the realized error within this envelope but not its scale).
    """
    eps = np.finfo(dtype).eps if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating) else EPS
    return c * n * eps


def inner_atol(k, scale, dtype=np.complex128, c=8.0):
    """Absolute tolerance for a k-term inner product of vectors with norm ``scale``.

    A k-term dot product of unit-scale operands accumulates rounding error of order
    ``k * eps``; scaling by the operands' actual norm (``scale``) turns that into an
    absolute tolerance. Use this for comparisons across different summation orders
    (e.g. a Cython BLAS-backed reduction vs a Python-loop reference).
    """
    eps = np.finfo(dtype).eps if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating) else EPS
    return c * k * eps * max(scale, 1.0)


def eig_atol(h_norm, n=1, dtype=np.complex128, c=1e3):
    """Absolute tolerance for eigenvalues obtained via a backward-stable path.

    A backward-stable eigensolver (dense ``eigh``/``eigvalsh``, or a Lanczos
    recurrence with adequate reorthogonalization) guarantees eigenvalues accurate to
    ``O(n * eps * ||H||)`` relative to the operator norm; ``c`` is a safety factor
    covering the extra rounding in building the operator/tridiagonal matrix itself
    (matvecs, block reductions) on top of the solve.
    """
    eps = np.finfo(dtype).eps if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating) else EPS
    return c * n * eps * max(abs(h_norm), 1.0)
