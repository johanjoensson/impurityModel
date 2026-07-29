r"""Sweeping the double-counting shift over a *frozen* determinant space.

The double counting is a uniform impurity shift, so the whole family of trial Hamiltonians is

.. math:: H(\mu) = H(0) - \mu\,\hat N_{\rm imp},

and :math:`\hat N_{\rm imp}` is **diagonal** in the determinant basis. On a fixed space that makes
the family one matrix build plus a diagonal shift: the expensive half of a trial evaluation --
the CIPSI expansion, measured at 96.5 % of a NiO double-counting search -- is skipped entirely,
and what is left is one small eigensolve. Measured on the NiO-20 workload at cap 400: a true
re-expanding evaluation 3.41 s, one frozen-space solve **0.003 s**, the two matrix builds 0.00 s
-- **0.08 %**, about 1240x cheaper per :math:`\mu`.

That is what makes the derivative affordable. The answer a double-counting search returns is
``dc``, not ``n``, and the error in ``dc`` is :math:`\delta\mu = \delta n / \chi` with
:math:`\chi = dn/d\mu`: on a plateau :math:`\chi \to 0` and an occupation converged to
``occ_tol`` still leaves ``dc`` unbounded. Measuring :math:`\chi` used to cost two full
evaluations; here it costs two eigensolves.

**What this is not.** The frozen space is not the answer, it is a cheap model of it. Freezing
fixes the variational space at whatever the seeding solve reached, so a :math:`\mu` far from the
seed is being scored on a space selected for a different Hamiltonian, and the reachable
occupation is bounded by the basis's own charge window rather than by the double counting
(:meth:`FrozenSpaceSweep.occupation_is_interior` reports that). The intended use is to locate the
root and read :math:`\chi` cheaply, then confirm with the true, re-expanding observable.

**What it buys in exchange for that.** On a frozen space the theory is exact:
:math:`E_0(\mu) = \min_{\|\psi\|=1} \langle\psi|H(0)|\psi\rangle - \mu\langle\psi|\hat N|\psi\rangle`
is a minimum of affine functions of :math:`\mu`, hence **concave**, with
:math:`-dE_0/d\mu = n(\mu)`, hence :math:`n(\mu)` **non-decreasing**. Neither holds for the
re-expanding observable, which is why the production search assumes no monotonicity. Both are
pinned as tests, along with the sharper statement that on a *pure* sector the shift is exactly
affine to machine precision.
"""

import numpy as np

from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_transcription import build_density_matrices, build_sparse_matrix
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.dc_criteria import _dc_operator

#: Finite-difference step for ``chi``, as a fraction of the shift scale. Central differences on a
#: quantity that is itself an eigensolve are second-order accurate, so the step only has to sit
#: well above the eigenvalue tolerance and well below the scale on which ``n(mu)`` bends -- the
#: sector spacing, which is O(1) in the units of ``h0``.
CHI_STEP = 1e-3


class FrozenSpaceSweep:
    """``energy(mu)`` / ``occupation(mu)`` / ``chi(mu)`` on one fixed determinant space.

    Parameters
    ----------
    basis : Basis
        The frozen space. Its restrictions must already be set -- the matrices are built once,
        here, and cannot be retro-fitted afterwards.
    h_op : ManyBodyOperator
        The Hamiltonian at ``mu = 0``, i.e. already carrying ``dc_guess`` (``sb.h``). Only the
        incremental shift is applied below, or the guess is double-counted.
    impurity_indices : sequence of int
        Impurity spin-orbitals, in the solver basis.
    tau : float
        Temperature for the thermal average.
    energy_cut : float, optional
        Thermal cut for the eigenstate manifold; defaults to the ``-tau*log(1e-4)`` the rest of
        the stack uses.

    Notes
    -----
    Every solve goes through :meth:`CIPSISolver.get_eigenvectors` with a prebuilt matrix rather
    than around it, so the warm-start cold-retry guard still applies: a warm block spans a
    near-invariant subspace and silently returns an excited state as "lowest" once the ground
    state has moved charge sector, which is exactly what a double-counting sweep makes it do.
    """

    def __init__(
        self,
        basis,
        h_op,
        impurity_indices,
        tau,
        *,
        energy_cut=None,
        dense_cutoff=1000,
        slater_weight_min=0,
        num_wanted=10,
        solver="irlm",
    ):
        self.basis = basis
        self.impurity_indices = list(impurity_indices)
        self.tau = tau
        self.energy_cut = -tau * np.log(1e-4) if energy_cut is None else energy_cut
        self.dense_cutoff = dense_cutoff
        self.slater_weight_min = slater_weight_min
        self.num_wanted = num_wanted
        self.solver_method = solver

        self._solver = CIPSISolver(basis)
        self._h_op = h_op
        # The two builds that make the sweep cheap. `h0` carries dc_guess already; `n` is the
        # impurity number operator, which _dc_operator(identity) is exactly -- diagonal in the
        # determinant basis, which is what makes H(mu) a diagonal shift rather than a rebuild.
        self._h0_matrix = build_sparse_matrix(basis, h_op)
        self._n_matrix = build_sparse_matrix(basis, _dc_operator(np.identity(len(self.impurity_indices))))
        self._solved = {}

    def hamiltonian(self, mu):
        """``H(mu) = H(0) - mu * N_imp`` as a sparse matrix, without rebuilding anything."""
        return self._h0_matrix - mu * self._n_matrix

    def _solve(self, mu):
        """Eigenstates of ``H(mu)`` on the frozen space, memoized (a solve is not free, just cheap)."""
        if mu not in self._solved:
            energies, psis = self._solver.get_eigenvectors(
                self._h_op,
                num_wanted=self.num_wanted,
                max_energy=self.energy_cut,
                dense_cutoff=self.dense_cutoff,
                slaterWeightMin=self.slater_weight_min,
                solver=self.solver_method,
                h_matrix=self.hamiltonian(mu),
            )
            self._solved[mu] = (energies, psis)
        return self._solved[mu]

    def energy(self, mu):
        """Lowest eigenvalue of ``H(mu)`` on the frozen space."""
        energies, _psis = self._solve(mu)
        return float(np.min(energies))

    def occupation(self, mu):
        """Thermally averaged impurity occupation ``Tr rho_imp`` at ``mu``.

        Routed through ``build_density_matrices``, i.e. the apply-local -> redistribute -> local
        inner product -> Allreduce path the rest of the stack uses, so it is correct on a
        distributed basis rather than only on a replicated one.
        """
        energies, psis = self._solve(mu)
        rhos = build_density_matrices(self.basis, psis, self.impurity_indices, self.impurity_indices)
        rho = thermal_average_scale_indep(energies, rhos, self.tau)
        return float(np.real(np.trace(rho)))

    def chi(self, mu, step=CHI_STEP):
        r"""``dn/dmu`` by central difference -- the slope that turns ``delta_n`` into ``delta_mu``.

        Two extra eigensolves on the frozen space, so of order 1 % of a single true evaluation.
        A one-sided difference would be first-order accurate and, on a staircase-like ``n(mu)``,
        systematically biased toward whichever side of the step it sampled.
        """
        return (self.occupation(mu + step) - self.occupation(mu - step)) / (2 * step)

    def shift_error(self, residual, mu, step=CHI_STEP):
        r"""``delta_mu = delta_n / chi``: what an occupation residual is worth in ``dc``.

        ``None`` when ``chi`` vanishes -- on a plateau the occupation carries no information
        about the shift at all, and reporting a finite error bar there would be a fiction. The
        caller should report the plateau's width instead.
        """
        slope = self.chi(mu, step)
        if slope == 0:
            return None
        return residual / slope

    def occupation_is_interior(self, mu, margin=0.5):
        """Is the achieved occupation strictly inside what this frozen space can represent?

        The space was generated for a charge window, so the occupation it can reach is bounded by
        that window whatever the double counting does. When the answer sits against the boundary
        the search has not determined ``mu`` -- the basis has -- and saying so is the difference
        between a result and an artefact.
        """
        occupations = np.real(np.diag(self._n_matrix.toarray())) if self._n_matrix.shape[0] else np.array([0.0])
        return bool(occupations.min() + margin <= self.occupation(mu) <= occupations.max() - margin)
