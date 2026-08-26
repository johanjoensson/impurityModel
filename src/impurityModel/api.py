"""Stable external surface of impurityModel.

External consumers (e.g. the RSPt wrapper impurityModel_interface) should
import from this module only. Everything under ``impurityModel.ed.*`` is
internal and may change without notice.

The supported surface is deliberately small: build the impurity problem
(:class:`ImpurityModel` plus the option groups :class:`Meshes`,
:class:`BasisOptions`, :class:`SolverOptions`), convert a one-particle
Hamiltonian matrix to the operator format (:func:`matrixToIOp`), solve for
the self-energy (:func:`calc_selfenergy`), determine the double counting
(:func:`fixed_peak_dc`, :func:`fixed_gap_dc`, :func:`fixed_occupation_dc`,
:func:`fll_dc`, :func:`amf_dc`, :func:`sigma_inf_dc`, :func:`nominal_dc` --
the first three raising :class:`DoubleCountingUnreachable` when the
requested target has no solution; :func:`fixed_gap_dc` is the one to reach
for on a charge-transfer insulator, where the occupation criterion has no
unique solution), diagnose the fixed-occupation criterion's DFT
reference against the continuum hybridization
(:func:`report_continuum_reference`) and write Green's functions in RSPt's
.dat format (:func:`save_Greens_function`).

Tuning knobs can also be taken from an input file rather than the environment
(:func:`find_environment_file`, :func:`load_environment`, :func:`apply_environment`), so a
run driven from outside this package -- RSPt through ``impurityModel_interface``, say -- can
still be described by one file instead of a shell's worth of exported variables.
"""

from impurityModel.ed.dc_record import dc_levels, dc_spread
from impurityModel.ed.dc_record import emit as emit_dc_record
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
from impurityModel.ed.selfenergy import (
    DoubleCountingUnreachable,
    amf_dc,
    calc_selfenergy,
    discretized_impurity_occupation,
    fixed_gap_dc,
    fixed_occupation_dc,
    fixed_peak_dc,
    fll_dc,
    nominal_dc,
    report_continuum_reference,
    sigma_inf_dc,
)
from impurityModel.ed.susceptibility import calc_susceptibility_workflow
from impurityModel.inputformat.reader import (
    apply_environment,
    find_environment_file,
    load_environment,
)

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("impurityModel")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "apply_environment",
    "find_environment_file",
    "load_environment",
    "BasisOptions",
    "DoubleCountingUnreachable",
    "ImpurityModel",
    "Meshes",
    "SolverOptions",
    "SpectraOptions",
    "__version__",
    "amf_dc",
    "atomic_u4",
    "calc_selfenergy",
    "calc_susceptibility_workflow",
    "dc_levels",
    "dc_spread",
    "discretized_impurity_occupation",
    "emit_dc_record",
    "fixed_gap_dc",
    "fixed_occupation_dc",
    "fixed_peak_dc",
    "fll_dc",
    "load_selfenergy_archive",
    "matrixToIOp",
    "nominal_dc",
    "report_continuum_reference",
    "save_Greens_function",
    "sigma_inf_dc",
]
