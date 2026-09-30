r"""Self-energy estimators: which operator family the Green's-function engine resolves, and how
the self-energy is read off it.

Layer: above :mod:`greens_function` (whose ``operator_families`` seam it feeds) and
:mod:`sigma`, below :mod:`selfenergy`, which picks one by ``SolverOptions.sigma_method``.

An estimator answers three questions:

* :meth:`~SelfEnergyEstimator.operator_families` -- which operators ``X`` to resolve for a
  block. The engine returns the Green's function of that family,
  ``G_ab(z) = <X_a (z - H + E)^-1 X_b^dag> + <X_b^dag (z + H - E)^-1 X_a>``, from addition seeds
  ``X^dag |psi>`` and removal seeds ``X |psi>``. By contract the family's leading
  ``len(block)`` operators are the block's own ``c`` in block order.
* :meth:`~SelfEnergyEstimator.impurity_gf` -- the impurity Green's function inside that
  family's G (the leading block).
* :meth:`~SelfEnergyEstimator.sigma` -- the self-energy from the family's G.

:class:`DysonEstimator` is the production estimator: the family is just ``c``, and
:math:`\Sigma = G_0^{-1} - G^{-1}`. The symmetric improved estimator resolves ``X = [c, q]``
with ``q = [c, H_int]`` (so the removal side needs width ``2n``) and reads :math:`\Sigma` off the
blocks of that G without inverting it; this module is where it plugs in. The interaction it
needs is :attr:`SolverBasis.h_int <impurityModel.ed.solver_basis.SolverBasis.h_int>`, defined
against the same ``h0_solve`` the Dyson estimator subtracts, so the two estimators measure the
same :math:`\Sigma`.
"""

from dataclasses import dataclass
from typing import ClassVar, Protocol

from impurityModel.ed import config
from impurityModel.ed.greens_function import impurity_operator_family
from impurityModel.ed.sigma import get_sigma


class SelfEnergyEstimator(Protocol):
    """What :func:`selfenergy.calc_selfenergy` needs from a self-energy estimator."""

    name: ClassVar[str]

    def operator_families(self, block, solver_basis):
        """``(addition_ops, removal_ops)`` for ``block``: ``X^dag`` and ``X``, leading ``c`` first."""

    def impurity_gf(self, g_family, block):
        """The ``len(block)``-wide impurity Green's function inside the family's G."""

    def sigma(self, mesh, g_families, *, delta, solver_basis, blocks, cluster_label, return_components):
        """Per-block self-energies on ``mesh`` (and, with ``return_components``, the Dyson terms)."""


@dataclass(frozen=True)
class DysonEstimator:
    r"""The Dyson-equation estimator, :math:`\Sigma = G_0^{-1} - G^{-1}` (:func:`sigma.get_sigma`)."""

    name: ClassVar[str] = "dyson"

    def operator_families(self, block, solver_basis=None):
        return impurity_operator_family(block)

    def impurity_gf(self, g_family, block):
        return g_family

    def sigma(self, mesh, g_families, *, delta, solver_basis, blocks, cluster_label="", return_components=False):
        return get_sigma(
            omega_mesh=mesh,
            impurity_orbitals=solver_basis.total_impurity_orbitals,
            nBaths=solver_basis.sum_bath_states,
            gs=g_families,
            h0op=solver_basis.h0_solve,
            delta=delta,
            blocks=blocks,
            clustername=cluster_label,
            return_components=return_components,
        )


#: ``SolverOptions.sigma_method`` -> estimator class. Its keys are :data:`config.SIGMA_METHODS`.
ESTIMATORS = {DysonEstimator.name: DysonEstimator}


def make_estimator(sigma_method: str) -> SelfEnergyEstimator:
    """The estimator ``sigma_method`` names; ``ValueError`` for any other value."""
    if sigma_method not in config.SIGMA_METHODS:
        raise ValueError(
            f"Unknown sigma_method {sigma_method!r}; expected one of {', '.join(map(repr, config.SIGMA_METHODS))}"
        )
    return ESTIMATORS[sigma_method]()
