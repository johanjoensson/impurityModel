"""Stable external surface of impurityModel.

External consumers (e.g. the RSPt wrapper impurityModel_interface) should
import from this module only. Everything under ``impurityModel.ed.*`` is
internal and may change without notice.

The supported surface is deliberately small: build the impurity problem
(:class:`ImpurityModel` plus the option groups :class:`Meshes`,
:class:`BasisOptions`, :class:`SolverOptions`), convert a one-particle
Hamiltonian matrix to the operator format (:func:`matrixToIOp`), solve for
the self-energy (:func:`calc_selfenergy`), determine the double counting
(:func:`fixed_peak_dc`, :func:`fixed_occupation_dc`) and write Green's
functions in RSPt's .dat format (:func:`save_Greens_function`).
"""

from impurityModel.ed.greens_function import save_Greens_function
from impurityModel.ed.model import (
    BasisOptions,
    ImpurityModel,
    Meshes,
    SolverOptions,
    SpectraOptions,
    atomic_u4,
    load_selfenergy_archive,
)
from impurityModel.ed.operator_algebra import matrixToIOp
from impurityModel.ed.selfenergy import calc_selfenergy, fixed_occupation_dc, fixed_peak_dc
from impurityModel.ed.susceptibility import calc_susceptibility_workflow

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("impurityModel")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "__version__",
    "BasisOptions",
    "ImpurityModel",
    "Meshes",
    "SolverOptions",
    "SpectraOptions",
    "atomic_u4",
    "calc_selfenergy",
    "calc_susceptibility_workflow",
    "fixed_occupation_dc",
    "fixed_peak_dc",
    "load_selfenergy_archive",
    "matrixToIOp",
    "save_Greens_function",
]
