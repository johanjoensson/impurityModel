r"""Double-counting determination for the self-energy workflow -- re-export shim.

The double counting is a first-class part of the model (``model.dc``), always subtracted from
the non-interacting Hamiltonian: the local Hamiltonian is ``h0 - dc + U(u4)`` throughout this
package and in :func:`solver_basis.prepare_solver_basis` / :func:`susceptibility` (never added).
Fixing it is the job of five schemes split across four modules:

- :mod:`impurityModel.ed.dc_reference` -- the DFT reference filling (the raw-``h0`` Fermi
  occupation/density matrix) and the saturation check every scheme that defaults to it routes
  through.
- :mod:`impurityModel.ed.dc_static` -- the three closed-form schemes: :func:`fll_dc` (Fully
  Localized Limit), :func:`amf_dc` (Around Mean Field), :func:`sigma_inf_dc` (K. Held's
  :math:`\Sigma(\infty)`). No ED solve, no MPI collective.
- :mod:`impurityModel.ed.dc_search` -- :func:`_solve_dc_shift`, the observable-agnostic root find
  over the uniform shift ``dc(mu) = dc_guess + mu * identity``, plus its cost accounting.
- :mod:`impurityModel.ed.dc_criteria` -- the two ED-based criteria built on it,
  :func:`fixed_peak_dc` and :func:`fixed_occupation_dc`.

This module re-exports the public names so existing callers -- and :mod:`selfenergy`, which
re-exports them again -- are unchanged. **A re-exported name is a new binding**: patching
``double_counting.suggest_truncation_threshold`` does not affect the call site inside
:mod:`dc_criteria`, which reads its own. Tests that stub a dependency must patch the module that
looks the name up at call time.

The self-energy extraction proper lives in :mod:`sigma`; the orchestration and CLI in
:mod:`selfenergy`.
"""

from impurityModel.ed.dc_criteria import (  # noqa: F401
    fixed_occupation_dc,
    fixed_peak_dc,
)
from impurityModel.ed.dc_reference import (  # noqa: F401
    _noninteracting_impurity_occupation,
    _noninteracting_impurity_rho,
)
from impurityModel.ed.dc_search import _solve_dc_shift  # noqa: F401
from impurityModel.ed.dc_static import (  # noqa: F401
    _model_u4_dense,
    _model_uj,
    amf_dc,
    fll_dc,
    sigma_inf_dc,
)

__all__ = [
    "amf_dc",
    "fixed_occupation_dc",
    "fixed_peak_dc",
    "fll_dc",
    "sigma_inf_dc",
]
