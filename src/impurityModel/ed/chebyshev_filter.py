"""Chebyshev window filters — thin Python wrapper.

All filter logic (spectral-bounds estimation, the Jackson-damped partition-of-unity
window coefficients, and the single-pass filter-bank application) lives in Cython in
:mod:`impurityModel.ed.ChebyshevFilter`. This module only re-exports the public entry
points so the import surface matches the ``cg.py`` / ``gmres.py`` thin-wrapper
arrangement.
"""

from impurityModel.ed.ChebyshevFilter import chebyshev_apply, partition_of_unity, spectral_bounds

__all__ = ["chebyshev_apply", "partition_of_unity", "spectral_bounds"]
