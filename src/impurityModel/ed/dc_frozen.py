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

#: Finite-difference step for ``chi``.
#:
#: .. warning::
#:    **A single fixed step cannot tell a slope from a straddle, so `chi` no longer trusts one on
#:    its own.** An earlier version of this comment claimed 1e-3 sits "well below the scale on
#:    which ``n(mu)`` bends -- the sector spacing, which is O(1)". That is false: ``n(mu)`` bends
#:    on the *thermal crossover* scale ``~tau/dn``, i.e. 1e-3 to 1e-2 -- the same size as the step.
#:    Measured on the fixture in ``test_dc_frozen.py`` at ``mu = -0.5225`` (``mixed_valence=1``,
#:    right at a sector boundary), ``chi(h)`` reads 126.30 / 252.26 / 468.23 / 617.31 / 636.47 as
#:    ``h`` halves from 8e-3: the ``1/h`` signature of ``dn / 2h``, a *straddle* of a sector step
#:    rather than a slope. A true derivative is independent of ``h`` -- measured at ``mu = 0.3`` on
#:    the same fixture, the same sweep from 8e-3 to 5e-4 changes the 4th significant figure only.
#:
#:    :meth:`FrozenSpaceSweep.chi` now runs exactly that comparison -- the estimate at ``step``
#:    against the estimate at ``2 * step`` -- and returns ``None`` rather than a number when they
#:    disagree by more than its ``rel_tol``. A search that trusted an unchecked ``chi`` returned an
#:    answer wrong by fifty times ``occ_tol`` while its guard certified it, and was reverted; this
#:    is what "certified" now actually requires.
CHI_STEP = 1e-3


def build_union_space(
    h_op,
    impurity_orbitals,
    bath_states,
    nominal_occ,
    *,
    mu_samples=(0.0,),
    sector_radius=1,
    tau=0.002,
    chain_restrict=False,
    spin_flip_dj=False,
    truncation_threshold=np.inf,
    comm=None,
    verbose=False,
    frozen_occupations=None,
    weighted_restrictions=None,
    slater_weight_min=0,
    de2_min=1e-8,
    dense_cutoff=1000,
):
    """A frozen space spanning the *candidate* charge sectors, grown across a ``mu`` bracket.

    A space seeded from one solve spans whatever sectors its expansion happened to reach --
    measured on NiO, ``n_imp = 8, 9, 10`` against a nominal 8: three sectors, but only upward, so
    a search that has to cross downward is scored on a space that cannot represent the answer and
    reports the nearest sector it can as though converged. This builds the span deliberately
    instead.

    No new generation code is needed for that. ``generate_initial_basis`` already filters the
    cross-group combinations on the *total* impurity charge window
    ``[total_nominal ± total_slack]``, and ``total_slack`` is exactly
    ``max(|mixed_valence| + |delta_impurity_occ|)`` over the unfrozen groups -- so widening
    ``mixed_valence`` to ``sector_radius`` widens the window symmetrically. Verified on a toy:
    radius 0 gives span ``(2,)``, radius 1 gives ``(1, 2, 3)``, radius 2 gives ``(0..4)``.

    **Growth-only, and state-averaged over the bracket.** The space is expanded once at each
    ``mu`` in ``mu_samples`` -- the ends of the search bracket, not just its centre -- and
    ``expand`` only *adds* determinants, so the result accumulates: ``S_{k+1} = S_k ∪ new``. That
    is what makes the construction terminate rather than limit-cycle, which re-freezing on drift
    does not (selection under a cap is discontinuous in ``mu``). The one exception is a binding
    ``truncation_threshold``: fixed-budget CIPSI prunes to stay under the cap, so growth-only
    holds up to the cap and no further. Pass ``truncation_threshold=np.inf`` when the guarantee
    matters more than the memory.

    Returns
    -------
    (FrozenSpaceSweep, Basis)
    """
    from impurityModel.ed.groundstate import build_basis_and_solver

    impurity_indices = [orb for blocks in impurity_orbitals.values() for block in blocks for orb in block]
    # The sector radius enters as mixed_valence: total_slack is the max over unfrozen groups, so
    # one entry per group at `sector_radius` opens the total window to nominal +/- sector_radius.
    mixed_valence = dict.fromkeys(nominal_occ, sector_radius)
    basis, solver = build_basis_and_solver(
        h_op,
        impurity_orbitals,
        bath_states,
        nominal_occ,
        mixed_valence,
        tau,
        chain_restrict,
        spin_flip_dj,
        truncation_threshold,
        comm,
        verbose,
        frozen_occupations,
        weighted_restrictions,
        slater_weight_min,
        None,
    )
    identity = np.identity(len(impurity_indices))
    for mu in mu_samples:
        # Expand at this end of the bracket. `expand` reads solver.psi_refs, so successive calls
        # are warm-started from the previous end's converged manifold -- which is the
        # state-averaging: the space ends up adequate across the sweep rather than at one point.
        solver.expand(
            h_op - _dc_operator(mu * identity),
            dense_cutoff=dense_cutoff,
            de2_min=de2_min,
            slaterWeightMin=slater_weight_min,
        )
    sweep = FrozenSpaceSweep(
        basis,
        h_op,
        impurity_indices,
        tau,
        dense_cutoff=dense_cutoff,
        slater_weight_min=slater_weight_min,
    )
    return sweep, basis


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
    than around it. Note the warm-start cold-retry guard inside that method -- originally given as
    the reason for putting the seam there -- is **inert here**: :meth:`_solve` passes no
    ``psi_refs``, so every solve is a cold start and the guard never engages. Cold starts are the
    safe choice for a sweep that crosses charge sectors; the seam's justification is that it keeps
    one code path, not that the guard is doing work.
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
        # Apply the basis's restrictions to the operator BEFORE the matrices are built. They
        # cannot be retro-fitted: `get_eigenvectors` calls `set_restrictions` on the *operator*,
        # which does nothing for a matrix that already exists, so a matrix built first is the
        # unrestricted PHP. The docstring above states this as a precondition; doing it here is
        # what makes the precondition true rather than merely asserted.
        if basis.restrictions is not None:
            h_op.set_restrictions(basis.restrictions)
        if basis.weighted_restrictions is not None:
            h_op.set_weighted_restrictions(basis.weighted_restrictions)
        # The two builds that make the sweep cheap. `h0` carries dc_guess already; `n` is the
        # impurity number operator, which _dc_operator(identity) is exactly -- diagonal in the
        # determinant basis, which is what makes H(mu) a diagonal shift rather than a rebuild.
        self._h0_matrix = build_sparse_matrix(basis, h_op)
        self._n_matrix = build_sparse_matrix(basis, _dc_operator(np.identity(len(self.impurity_indices))))
        # Impurity occupation per determinant, reduced across ranks once. `build_sparse_matrix`
        # returns a *globally shaped* matrix in which only this rank's columns are populated, so
        # reading the diagonal locally sees the owned determinants plus a spurious 0 everywhere
        # else -- measured at -n 3: spans (0,2), (0,3), (0,1,4) on the three ranks of one
        # 18-determinant space. Any decision taken on that is rank-local, and `sector_span` gates
        # a choice between different sequences of collectives. Reduced here, once, unconditionally
        # (CLAUDE.md: never gate a collective on rank-local state). `.diagonal()` rather than
        # `.toarray()`: densifying n x n to read n numbers is 64 MB at the benchmarked cap and
        # terabytes at production caps.
        local = np.real(self._n_matrix.diagonal())
        owned = local[np.asarray(basis.local_indices)] if basis.is_distributed else local
        occupations = sorted({int(round(n)) for n in owned})
        if basis.is_distributed:
            occupations = sorted({n for part in basis.comm.allgather(occupations) for n in part})
        self._sector_span = tuple(occupations)
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

    def _central_difference(self, mu, step):
        """Central difference of :meth:`occupation` at ``mu``, one fixed ``step``."""
        return (self.occupation(mu + step) - self.occupation(mu - step)) / (2 * step)

    def chi(self, mu, step=CHI_STEP, rel_tol=0.1):
        r"""``dn/dmu`` by central difference, straddle-checked by step-doubling.

        Four eigensolves on the frozen space (the estimate at ``step`` and again at ``2 * step``),
        so still of order 1 % of a single true evaluation. A *slope* is nearly independent of the
        step size; a *straddle* -- the central difference spanning a sector-boundary jump rather
        than sampling a smooth ``n(mu)`` -- scales as ``1/step`` instead, so doubling the step
        roughly halves it. Comparing the two catches exactly that (see ``CHI_STEP``'s warning for
        the measured numbers) without needing to know where any boundary is in advance.

        Returns
        -------
        float or None
            The central difference at ``step``, or ``None`` when it and the estimate at
            ``2 * step`` disagree by more than ``rel_tol`` (relative to the larger of the two,
            floored to avoid a spurious rejection when both are exactly zero) -- i.e. when the
            estimate cannot be trusted as a slope. Also ``None`` on an exact plateau is not
            possible: both estimates are then identically zero, which trivially agrees.
        """
        estimate = self._central_difference(mu, step)
        coarse = self._central_difference(mu, 2 * step)
        denom = max(abs(estimate), abs(coarse), 1e-9)
        if abs(estimate - coarse) / denom > rel_tol:
            return None
        return estimate

    def shift_error(self, residual, mu, step=CHI_STEP):
        r"""``delta_mu = delta_n / chi``: what an occupation residual is worth in ``dc``.

        ``None`` when ``chi`` is zero (a plateau, where the occupation carries no information
        about the shift at all) or unresolvable (a straddle, :meth:`chi`) -- either way, reporting
        a finite error bar would be a fiction. The caller should report the plateau's width, or
        refine the step, instead.
        """
        slope = self.chi(mu, step)
        if not slope:
            return None
        return residual / slope

    def sector_span(self):
        """The impurity occupations the frozen space can actually represent, as a sorted tuple.

        A frozen space is seeded from one solve, so which charge sectors it spans is *incidental*
        -- a consequence of how far the CIPSI expansion happened to reach with an unconstrained
        impurity window -- rather than chosen. Measured: on a real NiO workload the seeded space
        spans ``n_imp = 8, 9, 10`` against a nominal 8, i.e. three sectors but **only upward**;
        on a small toy it collapsed to a single sector and then reported that sector's occupation
        for *every* ``mu``, missing the true observable by a full electron across a charge-sector
        boundary.

        That asymmetry is the argument for building the union space deliberately instead of
        accepting whatever the seed reached: a sweep is only meaningful where the space can
        represent the answer, and this is the check that says where that is.
        """
        return self._sector_span

    def spans_sector(self, n_imp):
        """Whether the frozen space can represent impurity occupation ``n_imp`` at all.

        A driver taking a Newton step must not trust a shift whose answer lies outside this: the
        sweep does not report "I cannot say", it reports the nearest sector it *can* represent,
        which reads as a converged value.
        """
        span = self.sector_span()
        return bool(span) and span[0] <= n_imp <= span[-1]

    def occupation_is_interior(self, mu, margin=0.5):
        """Is the achieved occupation strictly inside what this frozen space can represent?

        The space was generated for a charge window, so the occupation it can reach is bounded by
        that window whatever the double counting does. When the answer sits against the boundary
        the search has not determined ``mu`` -- the basis has -- and saying so is the difference
        between a result and an artefact.
        """
        span = self.sector_span() or (0,)
        return bool(span[0] + margin <= self.occupation(mu) <= span[-1] - margin)
