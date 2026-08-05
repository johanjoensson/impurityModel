r"""The ED-based double-counting criteria: pin a peak, pin the gap centre, or pin the occupation.

All three are special cases of :func:`dc_search._solve_dc_shift` over the uniform shift
``dc(mu) = dc_guess + mu * identity``. :func:`fixed_peak_dc` pins a chosen spectral peak --
the sector-energy difference ``E[N+1] - E[N]`` (or ``E[N] - E[N-1]`` for a removal peak);
:func:`fixed_gap_dc` pins the *midpoint* of the two, ``(E[N+1] - E[N-1])/2``, which is Karolak
et al.'s prescription for a charge-transfer insulator (arXiv:1004.4569 -- see its docstring for
which criterion suits which regime); :func:`fixed_occupation_dc` pins the impurity occupation,
and with no explicit target pins it to the DFT reference occupation from :mod:`dc_reference`,
which is the natural criterion for CSC DFT+DMFT of wide-window p-d models.

The peak and gap criteria share :class:`_SectorContext`: the setup, the centre-sector walk and
the fixed-sector solves have one definition between them, so they cannot drift apart on the
ground-state conventions the way the two open-coded copies below once did.

Each of the three ends by emitting one :mod:`dc_record` block on rank 0 -- unconditionally, at any
verbosity, and from a ``finally`` so a search that raises still says how far it got. That block
replaced three differently-shaped ``if verbose and rank == 0`` summaries and two full matrix dumps
per call; the dumps now live behind ``DC_DIAGNOSTICS`` (:func:`_dump_dc_matrices`), where they
answer the one question the record's per-orbital level cannot: whether the matrix is uniform.

At every trial shift all three determine the ground-state sector through the same *function*
:func:`groundstate.calc_gs` uses -- :func:`groundstate.find_ground_state_basis`'s
HF-seed-then-walk search, not a search pinned at the input occupation. That matters because a
``dc`` measured on a different sector than the one ``calc_selfenergy`` later finds does not
approximate the requested physics, it misdirects it: the downstream calculation is locked onto the
wrong charge state, which is worse than not fixing anything at all.

.. warning::
   Until this fix, it was the same function but **not the same call** -- an earlier version of
   this docstring claimed otherwise. Three differences, all found by review, are now closed:

   * ``calc_gs`` divides ``tau`` by 100 before the walk (``groundstate.py``, ``calc_gs``); both
     criteria now do the same for their own ``find_ground_state_basis`` call, restoring the full
     ``tau`` for everything computed afterward at the found sector (the thermal average, the
     peak's sector-energy solves).
   * ``calc_gs`` runs the occupation walk at ``sqrt(slaterWeightMin)`` and tightens to the full
     cutoff only for the refinement; both criteria now do the same. Running the walk four orders
     of magnitude tighter than the one ``calc_selfenergy`` will run is a different search, and can
     settle on a different sector.
   * Neither criterion threads ``symmetry_generators`` down any more, and neither does
     ``calc_gs``. :meth:`cipsi_solver.CIPSISolver.expand` re-derives them from the ``H`` it is
     actually expanding whenever the argument is ``None``, which is the only operator they can
     correctly come from; a set computed once at ``mu = 0`` and passed everywhere is how the two
     paths drifted apart. ``expand`` additionally re-gates every generator on ``[H, g] = 0``
     against the *full* Hamiltonian, two-body terms included.
   * The refinement runs on a basis whose ``tau`` has been restored to the full value, matching
     ``calc_gs``; leaving the walk's ``tau/100`` in place made ``expand``'s
     ``de0_max = energy_cut(basis.tau)`` a hundred times narrower than the window ``calc_gs``
     expands into. The thermal solve uses ``groundstate.GS_CIPSI_SOLVER_METHOD``, not a
     hardcoded second kernel.
   * ``find_ground_state_basis`` accepts ``frozen_occupations``, and the CLI populates it for
     bath-less core shells (``get_spectra.py``: ``{i for i in nBaths if nBaths[i] == 0}``); both
     criteria now derive the same set from :attr:`solver_basis.SolverBasis.sum_bath_states` and
     pass it down, so a core shell can no longer drain in the double-counting search only.
   * The determinant budget is derived at the **full** memory-safety fraction, as ``calc_gs``
     derives its own. The peak and gap criteria used to halve it, to pay for two ``N +- 1`` bases
     believed to be live alongside the centre search's -- they are not, the three solves are
     sequential -- so the double counting was measured on half the space the self-energy run would
     use at that same ``dc``.

   * The PT2 threshold matches. Every DC sector energy now runs at
     :data:`groundstate.GS_DE2_MIN` (``1e-8``), the value :func:`groundstate.solve_ground_state`
     converges the production ground state to, rather than the ``1e-6`` that used to be a literal
     inside :func:`groundstate.calc_energy`. The looser value is still the default there and is
     still right for the *sector walk*, where only the ordering of charge sectors is decided and
     a common error largely cancels; it is wrong for the fixed-sector energies the peak and gap
     criteria **difference**, where it does not cancel and is then amplified by ``1 / |chi|`` --
     a factor of 9 on the workload this was measured on. ``calc_energy`` and ``solve_sector`` take
     ``de2_min`` per call so the walk and the sector solves can differ; see
     :data:`groundstate.SECTOR_WALK_DE2_MIN`.

   No convention is now knowingly unshared between the two paths.

Like :func:`selfenergy.calc_selfenergy` and :func:`groundstate.find_ground_state_basis`, both
derive their determinant budget from available per-rank memory
(:func:`impurityModel.ed.memory_estimate.suggest_truncation_threshold`) when
``BasisOptions.truncation_threshold`` is left at ``None``, and honor
``BasisOptions.excitation_budget``/``chain_restrict`` through the same
:func:`impurityModel.ed.basis_restrictions.build_weighted_restrictions` the other ED drivers use
-- the double counting is otherwise found on a different variational space than the solve that
will use it.
"""

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config, dc_record, solver_trace
from impurityModel.ed.average import energy_cut as boltzmann_energy_cut
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.basis_restrictions import build_weighted_restrictions
from impurityModel.ed.basis_transcription import build_density_matrices
from impurityModel.ed.dc_frozen import FrozenSpaceSweep
from impurityModel.ed.dc_reference import (
    _SATURATION_ADVICE,
    _noninteracting_impurity_occupation,
    _warn_if_not_fermi_referenced,
    _warn_if_reference_far_from_nominal,
    _warn_if_reference_saturated,
)
from impurityModel.ed.dc_search import _dc_chi, _dc_search_trace, _solve_dc_shift, bracket_width_tol
from impurityModel.ed.lie_algebra import extract_tensors, tensors_to_operator
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.memory_estimate import DEFAULT_MEMORY_SAFETY, log_memory_budget, suggest_truncation_threshold
from impurityModel.ed.solver_basis import _per_group_occupation, get_symmetry_generators, prepare_solver_basis
from impurityModel.ed.utils import matrix_print


def _dump_dc_matrices(dc_guess, dc, rank):
    """The full ``dc`` matrices, under ``DC_DIAGNOSTICS`` -- what the record's ``dc_level`` averages.

    These used to print on any ``verbosity > 0`` run, two ``n_imp x n_imp`` complex matrices per
    double-counting call, which is most of a screen for a d shell and says nothing a converged
    uniform ``dc`` does not already say through its level. Behind the knob they answer the one
    question the record's scalar cannot: whether the matrix is uniform at all. That matters for
    :func:`dc_static.sigma_inf_dc` (Hartree-Fock at an anisotropic density matrix) and for a
    ``dc_guess`` arriving from RSPt, never for the two searches' own ``dc_guess + mu * I``.

    Rank-local by design and safe to be so: this branch gates *printing only*. Nothing collective
    may move inside it -- that is what separates it from :func:`dc_search._dc_search_trace`, whose
    knob has to be broadcast because it gates an ``allreduce``.
    """
    if rank != 0 or not config.DC_DIAGNOSTICS.get():
        return
    matrix_print(dc_guess, label="DC guess:")
    matrix_print(dc, label="DC found:")


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

    The span is requested explicitly, through ``generate_initial_basis``'s
    ``total_charge_slack``: it widens the *total charge* window to ``nominal ± sector_radius``.
    Verified on a toy: radius 0 gives span ``(2,)``, radius 1 gives ``(1, 2, 3)``, radius 2
    gives ``(0..4)``. This is the one caller that wants a multi-sector space; every energy
    comparison needs the opposite (``H`` conserves ``N``, so a multi-sector basis makes
    ``calc_energy(N0)`` report the minimum over the window), which is why it is a separate knob
    from ``mixed_valence`` -- the latter fluctuates the impurity charge against the bath at
    *fixed* total.

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
    mixed_valence = dict.fromkeys(nominal_occ, 0)
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
        total_charge_slack=sector_radius,
    )
    identity = np.identity(len(impurity_indices))
    for mu in mu_samples:
        # Expand at this end of the bracket. `expand` reads solver.psi_refs, so successive calls
        # are warm-started from the previous end's converged manifold -- which is the
        # state-averaging: the space ends up adequate across the sweep rather than at one point.
        solver.expand(
            h_op - tensors_to_operator(np.asarray(mu * identity, dtype=complex)),
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


@dataclass(frozen=True)
class _SectorSolution:
    """What one charge-sector solve yields, once the eigenvectors have been reduced to numbers.

    Immutable and picklable, because it is broadcast: every field crosses
    :meth:`_SectorContext.sector_solve`'s boundary as a rank-replicated fact.

    ``occupation`` and ``occupation_ground`` are the same quantity under the two conventions this
    module has to reconcile. ``energy`` is ``min(es)``, a ``T = 0`` quantity, and Hellmann-Feynman
    relates ``d(energy)/dmu`` to the impurity occupation *of that one state* --
    ``occupation_ground``. ``occupation`` is the Boltzmann average over the retained manifold,
    which is what :func:`fixed_occupation_dc` and :func:`groundstate.calc_gs` report. They differ
    only when the manifold's states do not share ``N_imp``, which ``occupation_spread`` measures:
    a degenerate multiplet split only by symmetry has spread zero and the distinction is empty.
    """

    energy: float
    occupation: float
    occupation_ground: float
    n_states: int
    occupation_spread: float


@dataclass
class _SectorContext:
    """The mu-independent setup a *sector-energy* criterion needs.

    Shared by :func:`fixed_peak_dc` and :func:`fixed_gap_dc`, which differ only in *which*
    combination of ``E[N-1]``, ``E[N]``, ``E[N+1]`` they drive onto their target -- one
    difference for the peak, the symmetric one for the gap centre. Everything up to and
    including "solve one charge sector at this shift" is identical, and it is precisely the part
    where the DC/ground-state parity conventions live (the walk's ``tau / 100`` and
    ``sqrt(slaterWeightMin)``, the frozen bath-less groups, the excitation-budget restriction
    list, the memory-derived cap). Those had already drifted once between two open-coded copies;
    a second criterion sharing them by construction is the point of this class.

    ``n_center_at`` is filled as a side effect of :meth:`centre_sector`, keyed by ``mu``. It is
    what the search reads through its ``sector_of`` callback, and it is why the centre charge
    state at the returned shift is recoverable afterwards without another solve.

    **This class is the boundary where solver outputs become rank-replicated facts.** Everything
    it returns -- :meth:`sector_energy`, :meth:`centre_sector` -- is broadcast before it leaves,
    so every quantity derived from them downstream (the gap width, the energy tolerance sized from
    it, the gap centre, ``omega_+-``, the caller's ``report``) is rank-identical by construction
    rather than by luck. That invariant is load-bearing, not tidiness: ``fixed_gap_dc`` sizes its
    convergence tolerance from a gap width, and a rank-local ``tol`` gates every branch deciding
    whether the next collective solve happens -- ranks disagreeing in the last bits do not return
    different answers, they hang. A perturbation test pins it
    (``test_a_rank_dependent_sector_energy_cannot_reach_the_gap_tolerance``).
    """

    h_op_i: object
    dc_guess: np.ndarray
    identity: np.ndarray
    impurity_orbitals: dict
    bath_states: dict
    N0: dict
    mixed_valence: dict
    h_solver_matrix: np.ndarray
    max_occ_total: int
    frozen_occupations: set
    weighted_restrictions: object
    truncation_threshold: object
    tau: float
    slater_weight_min: float
    chain_restrict: bool
    spin_flip_dj: bool
    dense_cutoff: int
    bandwidth: float
    rank: int
    verbose: bool
    n_center_at: dict = None
    #: ``(mu, n_trial) -> _SectorSolution | None``. Filled by :meth:`sector_solve`, which never
    #: solves the same sector at the same shift twice. Two floats per entry, not eigenvectors --
    #: see that method for why retaining ``psis`` does not scale.
    sector_at: dict = None

    def __post_init__(self):
        if self.n_center_at is None:
            self.n_center_at = {}
        if self.sector_at is None:
            self.sector_at = {}

    @property
    def nominal_total(self):
        """The nominal *whole-impurity* occupation. Never one group's: the many-body basis is
        filtered on the total charge alone (:func:`basis_generation.generate_initial_basis`), and
        ``prepare_solver_basis`` re-derives the grouping from the block structure, so a group key
        taken from the input occupation indexes the wrong layout."""
        return sum(self.N0.values())

    def shifted_h(self, mu):
        """``H`` at trial shift ``mu``.

        ``h_op_i`` (``sb.h``, from ``prepare_solver_basis``) already has ``dc_guess`` subtracted
        (``H = h0 - DC + U`` with ``DC = model.dc = dc_guess``); only the incremental ``mu`` shift
        is subtracted here, or ``dc_guess`` would be double-counted.
        """
        return self.h_op_i - tensors_to_operator(np.asarray(mu * self.identity, dtype=complex))

    def dc(self, mu):
        """The double-counting matrix at ``mu``: ``dc_guess + mu * identity``."""
        return self.dc_guess + mu * self.identity

    def centre_sector(self, h_op, mu):
        """The centre charge state at ``mu``, from the same walk ``calc_selfenergy`` will run.

        :func:`groundstate.walk_to_ground_state_sector` owns the ``tau / 100`` and
        ``sqrt(slaterWeightMin)`` conventions. Only the walk: the refinement and thermal manifold
        that ``solve_ground_state`` adds do not change ``ground_state_occupation``, and the sector
        is all a sector-energy criterion needs before solving ``N +- 1``.
        """
        from impurityModel.ed.groundstate import walk_to_ground_state_sector

        basis_center = walk_to_ground_state_sector(
            h_op,
            self.impurity_orbitals,
            self.bath_states,
            self.N0,
            tau=self.tau,
            slaterWeightMin=self.slater_weight_min,
            frozen_occupations=self.frozen_occupations,
            mixed_valence=self.mixed_valence,
            chain_restrict=self.chain_restrict,
            dense_cutoff=self.dense_cutoff,
            spin_flip_dj=self.spin_flip_dj,
            comm=MPI.COMM_WORLD,
            verbose=self.verbose,
            truncation_threshold=self.truncation_threshold,
            weighted_restrictions=self.weighted_restrictions,
        )
        # Read from the search that chose it. NOT from a determinant: the returned basis is the
        # eigenvector support of an expansion whose impurity occupation window was widened
        # (build_excited_restrictions with imp_change=None is unconstrained), so it spans several
        # impurity occupations -- {1, 2, 3} on a split-block toy whose winning sector is 2, where
        # the first determinant reports 1. Centring on whichever determinant happened to come
        # first measured E[2] - E[1] where E[3] - E[2] was meant. fixed_occupation_dc never had
        # this problem: it reads Tr rho_imp over the whole basis, which averages correctly.
        # Broadcast for the same reason sector_energy is: `ground_state_occupation` is a rank-local
        # read off the winning basis, and this integer decides *which sectors each rank solves
        # next* (n_center +- 1) and whether the criterion raises on a non-nominal charge state. It
        # is currently rank-identical by inheritance -- find_ground_state_basis broadcasts every
        # energy its sector election compares -- but nothing here enforces that, and an asymmetric
        # raise strands the ranks that did not raise.
        n_center = MPI.COMM_WORLD.bcast(sum(basis_center.ground_state_occupation.values()), root=0)
        self.n_center_at[mu] = n_center
        return n_center

    def sector_solve(self, h_op, mu, n_trial):
        """One charge sector, solved once: :class:`_SectorSolution`, or ``None`` where undefined.

        ``None``, never a large finite number. A sector outside ``[0, max_occ_total]`` means the
        criterion is *undefined* at this ``mu``, not that its residual is big: a finite penalty on
        one side of a sector boundary and a real energy on the other flips the residual's sign and
        manufactures a bracket that is not a root. Measured on a real fixture: 9 of 15 evaluations
        and 54 of 93 sector solves (58%) spent refining one such fake bracket, after which the
        failure message blamed "a charge-sector boundary" -- the search's own artefact, reported
        as physics.

        **The energy and the impurity occupation come out of the same expansion, and that is the
        point.** They used to be two methods running two solves: ``sector_energy`` through
        ``calc_energy``, which discards its eigenvectors, and ``sector_occupation`` through
        ``solve_sector``, which re-solved the same sector at the same shift from scratch. The gap
        criterion's edge-character measurement therefore cost three extra sector solves at the
        converged shift -- a third of the whole search on a workload that converges in one
        evaluation -- and the occupations it reported were from a *re-run*, matching the energies
        they are the slope of only by assuming the expansion is deterministic. Now there is one
        solve, the pair is cached per ``(mu, n_trial)``, and ``_measure_edge_character`` is a
        dictionary lookup.

        Only the two floats are kept, never the eigenvectors. Retaining ``psis`` across a search
        was the obvious alternative and does not scale: ~20 states over a sector basis is ~2 GB
        across a search at the 400k-determinant caps this stack runs at, where a float pair is
        nothing. The added cost is one :func:`build_density_matrices` per solve -- a handful of
        single-term annihilator applies per state, against a whole CIPSI expansion.

        Everything returned is broadcast; see the class docstring for why that boundary exists.
        """
        from impurityModel.ed.groundstate import GS_DE2_MIN, solve_sector

        if not 0 <= n_trial <= self.max_occ_total:
            return None
        if (mu, n_trial) in self.sector_at:
            return self.sector_at[(mu, n_trial)]
        occ_trial = _per_group_occupation(n_trial, self.impurity_orbitals, self.h_solver_matrix)
        es, psis, sector_basis = solve_sector(
            h_op,
            self.impurity_orbitals,
            self.bath_states,
            occ_trial,
            self.mixed_valence,
            self.tau,
            self.chain_restrict,
            self.spin_flip_dj,
            self.dense_cutoff,
            comm=MPI.COMM_WORLD,
            verbose=self.verbose,
            truncation_threshold=self.truncation_threshold,
            slaterWeightMin=self.slater_weight_min,
            weighted_restrictions=self.weighted_restrictions,
            frozen_occupations=self.frozen_occupations,
            # The accuracy the answer is quoted at. `solve_sector` defaults to the walk's looser
            # `SECTOR_WALK_DE2_MIN`, which is right for deciding which charge sector wins and
            # wrong here: this energy is about to be *differenced* against another sector's, and
            # the residual truncation error does not cancel in a difference. It is then amplified
            # by 1 / |chi| on its way into `dc` -- a factor of 9 on the workload this was measured
            # on -- so the DC would be determined on a looser variational space than the
            # self-energy run that consumes it. Last unshared convention on the parity list in
            # this module's docstring.
            de2_min=GS_DE2_MIN,
        )
        # Broadcast the verdict *before* branching on it, not the answer afterwards. `es` and
        # `psis` come back rank-local from `solve_sector` (its docstring says so), and the next
        # line runs `build_density_matrices`, which ends in an `Allreduce`. A rank-local decision
        # in front of a collective is the deadlock this module has already shipped once.
        #
        # `solve_sector` returns `([inf], [])` for an empty basis -- a sector the restrictions
        # admit no determinants for -- undefined for the same reason as the bounds check above.
        usable = MPI.COMM_WORLD.bcast(bool(len(psis)) and bool(np.all(np.isfinite(es))), root=0)
        if not usable:
            self.sector_at[(mu, n_trial)] = None
            return None
        # The manifold *size* is broadcast too, and the manifold truncated to it.
        # `build_density_matrices` allocates its reduction buffer as `(len(psis), n_orb, n_orb)`,
        # so ranks disagreeing on the number of retained states do not merely disagree about a
        # number -- they enter the same `Allreduce` with different buffer shapes.
        #
        # This is belt-and-braces rather than a live hazard: the cut inside `get_eigenvectors` is
        # a pure function of `e_ref`, which is bit-identical across ranks (see the note there, and
        # the measurement behind it). Kept because it costs one bcast of an int and this class's
        # whole job is to make solver output a rank-replicated fact rather than a hope.
        n_states = MPI.COMM_WORLD.bcast(len(psis), root=0)
        if len(psis) < n_states:
            # Not a `raise`: a rank-local exception here strands the others inside the reduction
            # below, turning a diagnosable crash into a hang. Abort the job with the reason.
            print(
                f"rank {MPI.COMM_WORLD.rank} kept {len(psis)} eigenstates where rank 0 kept "
                f"{n_states}; the thermal manifold has to be rank-identical before its density "
                "matrices are reduced.",
                flush=True,
            )
            MPI.COMM_WORLD.Abort(1)
        es, psis = es[:n_states], psis[:n_states]
        impurity_indices = [orb for blocks in self.impurity_orbitals.values() for block in blocks for orb in block]
        rhos = build_density_matrices(sector_basis, psis, impurity_indices, impurity_indices)
        # Per-state impurity occupations, then both conventions off the same array. `occupation`
        # is the thermal average `fixed_occupation_dc` and `calc_gs` use; `occupation_ground` is
        # the single state attaining `energy`, which is the one Hellmann-Feynman actually relates
        # to `d(energy)/dmu`. `spread` says whether the distinction can matter on this model --
        # a manifold whose states share `N_imp` by symmetry makes the two identical.
        occupations = np.real(np.einsum("nii->n", rhos))
        solution = _SectorSolution(
            energy=float(np.min(es)),
            occupation=float(np.real(np.trace(thermal_average_scale_indep(es, rhos, self.tau)))),
            occupation_ground=float(occupations[int(np.argmin(es))]),
            n_states=int(n_states),
            occupation_spread=float(np.max(occupations) - np.min(occupations)),
        )
        solution = MPI.COMM_WORLD.bcast(solution, root=0)
        self.sector_at[(mu, n_trial)] = solution
        return solution

    def sector_energy(self, h_op, mu, n_trial):
        """``E[n_trial]`` at this shift, or ``None`` where the criterion is undefined."""
        solution = self.sector_solve(h_op, mu, n_trial)
        return None if solution is None else solution.energy

    def sector_occupation(self, h_op, mu, n_trial):
        """``Tr rho_imp`` of sector ``n_trial`` at this shift, or ``None`` where it is undefined.

        The impurity character of a charge excitation, measured rather than inferred. The gap
        criterion's ``delta_sum`` has always come from ``chi``, the secant of the gap centre
        against ``mu``, via Hellmann-Feynman -- which is exact for the exact eigenstate of
        ``H(mu)`` and *not* exact for a CIPSI expansion that re-selects its space at every
        ``mu``, and which needs at least two evaluations inside one sector to exist at all.

        Both of those bit on the first production workload it met. The NiO 5-bath archive already
        carries a converged double counting, so the search satisfied ``|g0| <= tol`` at ``mu = 0``
        and returned after **one** evaluation: no secant, ``chi = None``, ``delta_sum = None``.
        The diagnostic built to say whether the answer is trustworthy was silent on exactly the
        run that accepted one. This measures the same quantity from the eigenvectors of the solve
        the criterion already ran, so it always has a value and carries no finite-difference or
        basis-reselection error.
        """
        solution = self.sector_solve(h_op, mu, n_trial)
        return None if solution is None else solution.occupation


def _prepare_sector_context(model, basis, solver, *, comm=None, verbosity=0, memory_label):
    """Build the setup :class:`_SectorContext` holds, from the grouped option objects.

    Verbatim extraction of the preamble :func:`fixed_peak_dc` used to open with, so that
    :func:`fixed_gap_dc` cannot acquire a second, subtly different copy of it.
    """
    n_imp = len(model.impurity_indices)
    dc_guess = extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0]
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0

    # Derive the same solver-basis layout calc_selfenergy uses (root cause 3 of the original
    # DC<->GS mismatch: the fit valence/conduction split model.bath_states carries is not
    # necessarily the split calc_selfenergy re-derives from the sign of the bath on-site energy).
    sb = prepare_solver_basis(
        model.h0,
        model.dc,
        model.u4,
        model.impurity_orbitals,
        basis.nominal_occ,
        basis.mixed_valence,
        model.rot_to_spherical,
        verbosity,
    )
    impurity_orbitals = sb.impurity_orbitals
    bath_states = sb.bath_states

    max_occ_total = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)
    # One-body matrix in the *solver* basis (h0_solve, i.e. after any symmetry rotation), so the
    # energetic filling _per_group_occupation performs sees the same on-site energies that defined
    # the derived groups.
    h_solver_matrix = extract_tensors(sb.h0_solve, n_orb=sb.n_spin_orbitals, two_body=False)[0]

    truncation_threshold = basis.truncation_threshold
    if truncation_threshold is None:
        # The full safety fraction, as every other driver uses. This used to be halved, on the
        # grounds that "two fixed-sector solves (N +- 1) are built alongside the centre search's
        # own basis" -- which is not what happens. `centre_sector` reads the winning occupation off
        # its basis and drops it; `sector_energy` discards its basis on the way out
        # (`e_trial, _ = calc_energy(...)`). The three solves are strictly sequential and exactly
        # one is live at a time, so the peak is the centre walk's own SectorCache -- the same peak
        # `calc_gs` has, since it runs the same walk.
        #
        # Halving was not merely unnecessary, it was a DC<->GS parity break of the kind this
        # module exists to close: it measured the double counting on a determinant budget half the
        # size of the one `calc_selfenergy` will use at that dc, and truncation error in the
        # sector energies is the dominant error in both the gap centre and its width.
        # `fixed_occupation_dc` never halved, so this also makes the three criteria agree.
        truncation_threshold = suggest_truncation_threshold(
            model.n_spin_orbitals, comm=MPI.COMM_WORLD, safety=DEFAULT_MEMORY_SAFETY
        )
        log_memory_budget(
            truncation_threshold, model.n_spin_orbitals, comm=MPI.COMM_WORLD, verbose=verbose, label=memory_label
        )

    # The spread of the one-body h0 eigenvalues: the scale a sector-energy difference can move
    # over, used to size the search range. Nothing is derived from a penalty value any more -- an
    # unreachable sector is reported as *undefined*, not as a large number (see sector_energy).
    h1_for_scale = extract_tensors(ManyBodyOperator(model.h0), n_orb=model.n_spin_orbitals, two_body=False)[0]

    return _SectorContext(
        h_op_i=sb.h,
        dc_guess=dc_guess,
        identity=np.identity(dc_guess.shape[0]),
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        N0=sb.nominal_occ,
        mixed_valence=sb.mixed_valence,
        h_solver_matrix=h_solver_matrix,
        max_occ_total=max_occ_total,
        # A bath-less group pinned exactly as get_spectra.py pins it -- one of the three inputs
        # that make this the same call calc_gs makes (see the module warning).
        frozen_occupations={i for i in impurity_orbitals if sb.sum_bath_states[i] == 0},
        # Bath-only restriction (the reference "valence filled, conduction empty" occupation never
        # references N0), so the identical restriction list is valid for every sector -- the same
        # restrictions find_ground_state_basis applies to its own occupation-scan trials.
        weighted_restrictions=build_weighted_restrictions(bath_states, basis.excitation_budget),
        truncation_threshold=truncation_threshold,
        tau=basis.tau,
        slater_weight_min=basis.slater_weight_min,
        chain_restrict=basis.chain_restrict,
        spin_flip_dj=basis.spin_flip_dj,
        dense_cutoff=solver.dense_cutoff,
        bandwidth=max(float(np.ptp(np.linalg.eigvalsh(h1_for_scale))), 1.0),
        rank=rank,
        verbose=verbose,
    )


def fixed_peak_dc(
    model,
    basis,
    solver,
    *,
    peak_position,
    comm=None,
    verbosity=0,
    allow_charge_state_change=False,
    return_sector=False,
    report=None,
):
    r"""
    Calculate the double counting correction using a fixed peak position criterion.

    Choose the double counting so that a peak in the impurity spectral function
    lands at the requested energy,

    .. math::
        E[N+1] - E[N] &= \omega_{peak},\quad \omega_{peak} \geq 0,\\
        E[N] - E[N-1] &= \omega_{peak},\quad \omega_{peak} < 0,

    where :math:`E[M]` is the lowest energy with M electrons on the impurity.
    A positive peak position places an electron-addition peak, a negative one an
    electron-removal peak.

    The double counting is parametrized as a uniform shift of the guess,
    ``dc(mu) = dc_guess + mu * identity``. The shift couples to the impurity
    occupation as :math:`-\mu \hat N_{imp}`, so the peak position responds
    monotonically, :math:`d(E_{upper} - E_{lower})/d\mu = -(\langle N
    \rangle_{upper} - \langle N \rangle_{lower}) \approx -1`, and ``mu`` is
    found by the shared secant/bisection search :func:`_solve_dc_shift`.

    At every trial ``mu`` the *center* sector ``N`` is not the input ``model.impurity_orbitals``
    occupation: it is found by :func:`groundstate.find_ground_state_basis`, the identical
    HF-seed-then-walk search :func:`groundstate.calc_gs` uses for the selfenergy/spectra solve.
    A peak measured relative to a different center than the one calc_selfenergy later finds would
    not approximate the requested peak position, it would misplace it -- the same reasoning that
    drives :func:`fixed_occupation_dc`'s sector determination. The ``N +- 1`` sectors are then
    each a single, fixed-occupation solve (:func:`groundstate.calc_energy`) relative to that
    found center, not a further search. ``N`` here is the *whole-impurity* occupation, never one
    orbital-symmetry group's: :func:`solver_basis.prepare_solver_basis` re-derives the grouping
    from the block structure (a cubic d-shell input arrives as one group and comes back as
    ``eg``/``t2g``), and the many-body basis is filtered on the total impurity charge alone, so
    the total is the only well-defined handle. Energies carry no fixed unit, they follow the inputs
    (e.g. Ry when called from RSPt); the convergence tolerance is ``max(tau, 1e-4)`` in those
    units. All three sectors honor ``BasisOptions.excitation_budget``/``chain_restrict``
    identically -- the bath-only excitation-budget restriction never references the occupation,
    so the same restriction list applies regardless of which sector it is evaluated on.

    Parameters
    ----------
    model : impurityModel.ed.model.ImpurityModel
        The impurity problem: ``h0`` (non-interacting Hamiltonian), ``u4`` (Coulomb tensor),
        ``dc`` (double counting correction, used as the search's starting guess),
        ``impurity_orbitals`` and ``bath_states`` -- the ``(valence, conduction)`` bath split is
        required here (build the model with it, e.g. ``from_blocks(..., bath_valence_conduction=...)``).
    basis : impurityModel.ed.model.BasisOptions
        Nominal occupation (``{group: N}``; any number of groups -- the criterion moves one
        electron on/off the impurity *as a whole*, so no group is singled out), mixed valence,
        spin-flip determinants,
        temperature and the determinant budget. ``truncation_threshold=None`` (the default)
        derives the cap from available per-rank memory (collective on ``MPI.COMM_WORLD``,
        :func:`impurityModel.ed.memory_estimate.suggest_truncation_threshold`) at the same safety
        fraction ``calc_gs`` uses -- the three sector solves are sequential, so the live peak is
        the centre walk's own ``SectorCache``, which is the peak ``calc_gs`` has too; ``numpy.inf``
        disables capping.
    solver : impurityModel.ed.model.SolverOptions
        Provides the dense-eigensolver cutoff.
    peak_position : float
        Requested peak position; the sign selects addition/removal, see above.
        The magnitude is kept above ``4 * tau`` (thermal broadening).
    comm : MPI.Comm or None
        MPI communicator (used for rank-0 logging; the basis build uses ``MPI.COMM_WORLD``).
    verbosity : int
        Verbosity level.
    allow_charge_state_change : bool
        This criterion is **multivalued**: ``n_center`` is re-found at every trial ``mu``
        (never fixed at the input nominal occupation), so the same requested peak position can
        be attained at several different centre charge states, each several eV apart in ``dc``
        (measured on NiO: ~6 distinct roots, ~4 eV apart), with the bidirectional search
        returning whichever sits closest to ``mu = 0``. By default (``False``) a converged ``mu``
        whose centre charge state differs from the nominal total
        (``sum(basis.nominal_occ.values())``) is rejected with ``RuntimeError`` rather than
        silently returned -- placing a peak in the wrong charge state is not a tolerable
        approximation of the request, any more than a mis-sectored occupation search is (see the
        module docstring). Set ``True`` to accept whichever charge state the search lands on.

    Returns
    -------
    dc : ndarray
        The double counting matrix, ``dc_guess + mu * identity``.

    Raises
    ------
    dc_search.DoubleCountingUnreachable
        If the requested peak cannot be bracketed within the reachable range, e.g. because the
        criterion is ill conditioned (the upper and lower sectors have the same impurity
        occupation, so a uniform shift cannot move the peak).
    RuntimeError
        If the converged ``mu`` centres the peak on a non-nominal charge state and
        ``allow_charge_state_change`` is ``False`` -- a different failure mode (the target *was*
        reached, on the wrong sector), raised directly here rather than by the shared search.
    """
    with dc_record.recording(
        "peak", rank=comm.rank if comm is not None else MPI.COMM_WORLD.rank, report=report
    ) as dc_rec:
        ctx = _prepare_sector_context(
            model, basis, solver, comm=comm, verbosity=verbosity, memory_label="fixed-peak dc"
        )
        rank, verbose, tau = ctx.rank, ctx.verbose, ctx.tau

        # The residual *is* a sector energy difference, so it inherits the E_F = 0 convention every
        # sector comparison silently depends on -- the same reason fixed_occupation_dc checks it.
        _warn_if_not_fermi_referenced(model.h0, model.n_spin_orbitals, rank=rank)

        # Keep the requested peak outside the thermal broadening, preserving the sign: a negative
        # peak position places a removal peak at E[N] - E[N-1].
        if peak_position >= 0:
            peak_position = max(peak_position, 4 * tau)
            addition = True
        else:
            peak_position = min(peak_position, -4 * tau)
            addition = False

        # Keyed by mu (never re-evaluated for the same mu -- _solve_dc_shift's own residual cache),
        # so the centre charge state at the mu the search finally returns can be recovered afterward
        # without a further solve.
        n_center_at = ctx.n_center_at

        # mu -> the achieved peak position, for the record's chi (delta_mu = delta_peak / chi). Free:
        # every entry is a point the search already paid a pair of sector solves for.
        peak_at = {}

        def peak_observable(mu):
            with solver_trace.labelled(mu=mu), solver_trace.timed("dc_evaluation") as evaluation_fields:
                gap = _peak_gap_at_mu(mu)
                evaluation_fields["gap"] = gap
                if gap is not None:
                    peak_at[mu] = gap
                return gap

        def _peak_gap_at_mu(mu):
            h_op = ctx.shifted_h(mu)
            n_center = ctx.centre_sector(h_op, mu)

            n_upper = n_center + 1 if addition else n_center
            n_lower = n_center if addition else n_center - 1

            # Both unconditionally, and only then tested for None: each is a collective solve, so an
            # early return between them would move calc_energy off some ranks (CLAUDE.md's MPI rule).
            e_upper = ctx.sector_energy(h_op, mu, n_upper)
            e_lower = ctx.sector_energy(h_op, mu, n_lower)
            if e_upper is None or e_lower is None:
                if verbose and rank == 0:
                    missing = "N+1" if e_upper is None else "N-1"
                    print(f"mu={mu:.6f} n_center={n_center} {missing} sector unreachable: peak undefined here")
                return None
            if verbose and rank == 0:
                print(f"mu={mu:.6f} n_center={n_center} E_upper - E_lower={e_upper - e_lower:.6f}")
            return e_upper - e_lower

        # `ctx.bandwidth` is the one-body spectrum's own spread -- not a penalty value divided by the
        # constant that scaled it, which made the search range move whenever the penalty was tuned.
        # The peak position responds to the shift with slope ~ -1, so this comfortably covers the
        # reachable range. An observable that does not move with mu (the upper and lower sectors hold
        # the same impurity occupation -- the old delta_n ~ 0 ill-conditioning) never brackets and
        # surfaces as the unreachable error.
        tol = bracket_width_tol(tau)
        unreachable = (
            "The fixed-peak double counting could not place the peak at {target}: E_upper - E_lower "
            "reached {value:.4f} at mu = {mu:.3f}. The upper and lower sectors may hold equal impurity "
            "occupation (a uniform shift cannot move the peak), or the target lies beyond the "
            "reachable range."
        )
        # Held by the record so the emit in its `finally` can read the search's own verdict even on
        # the path where the raise skips past every line that would have projected it.
        search_report = dc_rec.search = {}
        dc_rec["target"] = peak_position
        dc_rec["tol"] = tol
        with _dc_search_trace("fixed-peak", MPI.COMM_WORLD, rank, width_tol=tol, report=search_report):
            mu = _solve_dc_shift(
                peak_observable,
                peak_position,
                tol=tol,
                width_tol=tol,
                initial_step=max(10 * tau, abs(peak_position)),
                max_shift=max(ctx.bandwidth, 10 * abs(peak_position), 1.0),
                plateau_ok=False,
                unreachable_message=unreachable,
                rank=rank,
                # Sector-first: bracket the nominal charge state, then solve the criterion inside it.
                # This criterion is multivalued -- the same requested peak position is attained at
                # several charge states, several eV apart in dc -- so "which sector" has to be decided
                # before "which root", not read off afterwards and complained about.
                sector_of=n_center_at.get,
                nominal_sector=ctx.nominal_total if not allow_charge_state_change else None,
                # d(E_upper - E_lower)/dmu = -(n_upper - n_lower): raising mu lowers the impurity
                # level, so the peak moves down. The two sectors differ by one *total* electron, and
                # the fraction of it landing on the impurity is < 1 in the charge-transfer regime, so
                # -1 is the upper bound on |slope| rather than its value. Used to size the first step
                # only; overestimating it makes that step short, which the walk then doubles out of.
                slope_sign=-1,
                slope=-1.0,
                report=search_report,
                verbose=verbose,
                # The observable's own collectives run on COMM_WORLD (the basis builds below
                # hardcode it), not on the caller's `comm`, so the residual must be broadcast
                # there too or the branch decisions can diverge between ranks.
                comm=MPI.COMM_WORLD,
            )

        # mu is always a point _solve_dc_shift actually evaluated (the mu=0 fast path, a direct scan
        # hit, or a refined bracket point), so peak_observable(mu) ran and n_center_at holds it.
        n_center = n_center_at[mu]
        nominal_total = ctx.nominal_total
        dc = ctx.dc(mu)
        dc_rec["peak"] = peak_at.get(mu)
        dc_rec["chi"], dc_rec["chi_span"] = _dc_chi(
            peak_at, mu, width_tol=tol, in_sector=lambda trial: n_center_at.get(trial) == n_center_at.get(mu)
        )
        dc_rec["dc_trace"], dc_rec["dc_level"] = dc_record.dc_levels(dc)
        dc_rec["dc_spread"] = dc_record.dc_spread(dc)
        _dump_dc_matrices(ctx.dc_guess, dc, rank)
        if n_center != nominal_total and not allow_charge_state_change:
            raise RuntimeError(
                f"fixed_peak_dc found mu = {mu:.6f} (dc = dc_guess + mu), but the centre charge state "
                f"there is n_center = {n_center}, not the nominal total {nominal_total}. This "
                "criterion is multivalued: the same requested peak position can be attained at "
                "several different charge states, several eV apart in dc, and the bidirectional "
                "search returns whichever sits closest to mu = 0 -- not necessarily the one at the "
                "nominal charge state. Pass allow_charge_state_change=True to accept this result "
                "anyway, or adjust dc_guess/peak_position to land nearer the nominal sector."
            )

        return (dc, n_center) if return_sector else dc


#: Energy tolerance for the gap criterion, as a fraction of the *measured* gap width.
#:
#: The gap width is the scale on which "the centre of the gap" is a meaningful statement, so it is
#: the scale the residual must be converged on. Converging much tighter than a percent of it is
#: chasing the truncation noise of two independently expanded CIPSI sectors, at a full
#: ground-state solve per step. This is what separates the *energy* tolerance from the *mu*
#: resolution, which stays :func:`dc_search.bracket_width_tol`; :func:`fixed_peak_dc` still passes
#: one number for both, because a requested peak position carries no width to scale against.
GAP_ENERGY_TOLERANCE_FRACTION = 0.01

#: Ceiling on the gap criterion's energy tolerance, as a fraction of the one-body bandwidth.
#:
#: :data:`GAP_ENERGY_TOLERANCE_FRACTION` scales the tolerance to a *measured* gap width, and that
#: width is a second difference of three independently truncated CIPSI energies -- the one quantity
#: in this criterion with no error cancellation at all. When truncation inflates it, nothing else
#: bounds the tolerance, and an inflated tolerance is not merely imprecise: ``abs(g0) <= tol`` is
#: the first thing :func:`dc_search._solve_dc_shift` tests, so a large enough tolerance makes the
#: search return ``mu = 0`` after a single evaluation and report ``converged``. A no-op that says
#: it succeeded is the worst failure mode available to a criterion.
#:
#: The bandwidth is the scale over which a sector energy difference can move at all, so a tolerance
#: approaching it cannot localize ``mu`` inside the search's own range. A tenth of it is loose
#: enough never to bind on a sane model (on the analytic toy the tolerance is 0.03 against a cap of
#: 0.3) and tight enough that a runaway width is caught and said out loud rather than absorbed.
GAP_ENERGY_TOLERANCE_MAX_FRACTION = 0.1


#: Below this, *one* gap edge is more bath than impurity and the answer is warned about.
#:
#: Applied to ``min(delta_+, delta_-)``, not to their sum, and that is the whole point. Each delta
#: is the fraction of the transferred electron landing on the impurity at one edge, in ``[0, 1]``.
#: **The sum hides exactly the case this warning exists for**, because the two edges fail
#: independently: measured directly (see :meth:`_SectorContext.sector_occupation`), the
#: charge-transfer fixture is ``delta_+ = 0.075`` against ``delta_- = 0.951`` and NiO 5-bath is
#: ``delta_+ = 0.561`` against ``delta_- = 0.032``. Their sums, ``1.03`` and ``0.59``, look
#: unremarkable next to a near-atomic ``2.00`` -- a sum-based floor tuned to catch either of them
#: has to sit so high it fires on healthy models. The minimum separates them by more than a factor
#: of four, in both directions, on every case measured.
#:
#: **Calibrated against the measurement, not before it.** The predecessor of this constant was a
#: sum-based ``0.2``, chosen from an error-propagation argument alone; the workload that motivated
#: it then measured ``0.218`` and the warning stayed silent on its own example. Raising it to
#: ``0.5`` fixed that case and broke the general one -- a sum of ``0.59`` on NiO, whose removal edge
#: is *pure ligand*, is above the floor. ``0.3`` on the minimum is what the three measured models
#: actually separate at: ``0.032`` and ``0.075`` fire, ``0.56``, ``0.95`` and ``1.00`` do not.
GAP_EDGE_IMPURITY_FLOOR = 0.3


def _warn_if_a_gap_edge_is_not_impurity_like(delta_plus, delta_minus, rank):
    """Say when one of the two edges being centred belongs to the bath rather than the impurity.

    Printing only, no collective, so the rank gate is safe. Unconditional on rank 0 rather than
    verbosity-gated, like the other reference-quality warnings in this module: a double counting
    fixed against a ligand band is not wrong in a way the user can see from the returned matrix.

    The text deliberately does **not** say the answer is invalid. For a charge-transfer insulator
    a ligand-like removal edge is what the material *is* -- NiO's valence-band top is O-2p, and a
    criterion centring the cluster charge gap is then centring the p-to-d gap, which is arguably
    the quantity one wants. What the user has to know is *which* gap was centred, because it is
    not the gap of the impurity spectral function ``A_imp`` that Karolak et al. write down, and
    the two are several eV apart.
    """
    measured = [
        (name, value) for name, value in (("delta_+", delta_plus), ("delta_-", delta_minus)) if value is not None
    ]
    if rank != 0 or not measured:
        return
    name, weakest = min(measured, key=lambda pair: pair[1])
    if weakest >= GAP_EDGE_IMPURITY_FLOOR:
        return
    edge = "addition (N+1)" if name == "delta_+" else "removal (N-1)"
    print(
        f"WARNING: the {edge} edge this double counting was fixed against is only {weakest:.3f} "
        f"impurity-like ({name}, of a maximum of 1; below {GAP_EDGE_IMPURITY_FLOOR} is flagged). "
        "The N +- 1 sector energies carry no impurity projection, so what was centred on the Fermi "
        "level is the CLUSTER charge gap, and this edge is a state of the bath. Two very different "
        "situations produce that. (1) A charge-transfer insulator: the valence-band top really is "
        "ligand, the cluster gap really is the p-to-d gap, and the answer is meaningful -- but it "
        "is not the gap of the impurity spectral function A_imp, which is what the fixed-gap "
        "prescription is usually quoted for, and the two can sit several eV apart. (2) A "
        "discretization artefact: an isolated fitted bath level inside the true gap, which does "
        "not move with the double-counting shift at all. Distinguish them with "
        "test/support/dc_weights.py -- a band shows many nearly degenerate states carrying weight, "
        "an artefact shows one isolated level carrying none -- and cross-check against "
        "fixed_occupation_dc before using this dc.",
        flush=True,
    )


def _measure_edge_character(ctx, mu, n_center):
    """``(delta_+, delta_-)``: how much of each gap edge is impurity, at shift ``mu``.

    ``delta_+ = <N_imp>_{N+1} - <N_imp>_{N}`` and ``delta_- = <N_imp>_{N} - <N_imp>_{N-1}``.
    Near 1 per edge for an impurity-like excitation and near 0 for a bath-like one, but **not
    bounded by 1**: only the total change across impurity *and* bath is one electron, and the
    bath's share can be negative. At a charge-transfer level crossing -- ``N`` ground state
    ``d8``, ``N+1`` ground state ``d10 L`` -- ``delta_+`` is 2, and ``delta_-`` can come out
    negative. That regime is precisely the one this criterion is aimed at, so the numbers are
    reported as measured rather than clipped to a range they do not obey.

    **Free.** Every sector this needs was already solved at this ``mu`` by the search itself, and
    :meth:`_SectorContext.sector_solve` kept the occupation alongside the energy, so this is three
    dictionary lookups. It used to be three fresh solves at the converged shift, which on a
    workload converging in one evaluation was a third of the total work. Being free is also what
    lets it run at *every* evaluated ``mu`` rather than only the converged one -- including the
    last one before :class:`DoubleCountingUnreachable`, where the edges were previously absent
    from the record and where a bath-like edge is a leading *cause* of the failure.

    Every value read here is already broadcast, so the ``None`` tests are rank-identical verdicts
    rather than rank-local ones.
    """
    h_op = ctx.shifted_h(mu)
    n_add = ctx.sector_occupation(h_op, mu, n_center + 1)
    n_mid = ctx.sector_occupation(h_op, mu, n_center)
    n_rem = ctx.sector_occupation(h_op, mu, n_center - 1)
    delta_plus = None if n_add is None or n_mid is None else n_add - n_mid
    delta_minus = None if n_mid is None or n_rem is None else n_mid - n_rem
    return delta_plus, delta_minus


def _size_gap_tolerance(width, mu_tol, bandwidth, rank=0):
    """The gap criterion's energy tolerance, and the one word that explains where it came from.

    Split out of :func:`fixed_gap_dc` so the three degenerate cases can be tested directly, without
    standing up a model and running a search to reach each of them.

    The ``abs()`` this replaced was the problem. A **negative** width means ``E[N]`` is not the
    lowest of the three sectors -- the centre walk (at ``tau/100`` and ``sqrt(slaterWeightMin)``)
    and the sector solves (at ``tau`` and ``slaterWeightMin``) have disagreed about which charge
    state is the ground state. That is a real and reportable diagnostic; ``abs()`` turned it into a
    slightly different tolerance and said nothing, while the *signed* width still went on to the
    record, where it inverts the documented ``omega_+ >= 0 >= omega_-`` ordering with no warning.

    Returns
    -------
    (energy_tol, basis) : tuple of (float, str)
        ``basis`` is one of ``measured_gap``, ``width_undefined``, ``width_negative``,
        ``width_capped`` -- recorded so the log says which of the four produced the number.
    """
    if width is None:
        return mu_tol, "width_undefined"
    if width < 0.0:
        if rank == 0:
            print(
                f"WARNING: the measured gap width is negative ({width:.6f}). E[N] is not the lowest "
                "of the three sectors, so the centre walk and the fixed-sector solves disagree "
                "about which charge state is the ground state -- they run at different truncation "
                "conventions (tau/100 and sqrt(slaterWeightMin) against tau and slaterWeightMin), "
                "and a sector comparison that close is not resolved by either. The energy "
                "tolerance falls back to the mu resolution; treat the reported gap width and "
                "omega_+- as unreliable.",
                flush=True,
            )
        return mu_tol, "width_negative"
    scaled = GAP_ENERGY_TOLERANCE_FRACTION * width
    tol = max(scaled, mu_tol)
    cap = max(GAP_ENERGY_TOLERANCE_MAX_FRACTION * bandwidth, mu_tol)
    if tol > cap:
        if rank == 0:
            print(
                f"WARNING: the measured gap width {width:.6f} would set an energy tolerance of "
                f"{tol:.4e}, more than {GAP_ENERGY_TOLERANCE_MAX_FRACTION:g} of the one-body "
                f"bandwidth {bandwidth:.4f}. A tolerance that large cannot localize mu inside the "
                "search's own range -- the criterion would accept the double-counting guess "
                f"unchanged and report success. Capping at {cap:.4e}. The width is a second "
                "difference of three independently truncated sector energies; a value this large "
                "usually means the determinant cap is too tight for one of them.",
                flush=True,
            )
        return cap, "width_capped"
    # Which of the two terms the `max` actually returned. Reporting `measured_gap` whenever the
    # width was merely *defined* was wrong on both real workloads measured: NiO 15-bath sized
    # 0.01 * 0.0654 = 6.5e-4 against a mu resolution of 2.5e-3, and NiO 5-bath 0.01 * 0.119 =
    # 1.2e-3 against the same 2.5e-3 -- so in both cases the tolerance came from `bracket_width_tol`
    # and the record said it came from the gap. A field whose whole job is to say where a number
    # came from has to distinguish the two, or the one case it is read in is the one it misreports.
    return tol, "measured_gap" if scaled >= mu_tol else "mu_resolution"


def fixed_gap_dc(
    model,
    basis,
    solver,
    *,
    offset=0.0,
    comm=None,
    verbosity=0,
    allow_charge_state_change=False,
    return_sector=False,
    report=None,
):
    r"""Double counting from the **centre of the charge gap** -- after Karolak's insulator
    prescription.

    Choose the double counting so that the midpoint between the electron-removal and
    electron-addition excitations sits at ``offset`` (zero, i.e. the Fermi level, by default):

    .. math::
        \omega_+ &= E[N+1] - E[N] \geq 0 \quad\text{(addition)}\\
        \omega_- &= E[N] - E[N-1] \leq 0 \quad\text{(removal)}\\
        \text{residual}(\mu) &= \frac{\omega_+ + \omega_-}{2}
                             = \frac{E[N+1] - E[N-1]}{2} - \text{offset}

    **Why this criterion exists.** Karolak, Ulm, Wehling, Mazurenko, Poteryaev and Lichtenstein,
    *Double counting in LDA+DMFT -- the example of NiO* (J. Electron Spectrosc. Relat. Phenom.
    **181**, 11 (2010), arXiv:1004.4569) show that the fixed-occupation condition -- their Eq. (2),
    :func:`fixed_occupation_dc` here -- "essentially breaks down" for a charge-transfer insulator.
    Inside a gap the impurity occupation is *flat* in the chemical potential, so a whole interval
    of ``mu`` satisfies the occupation criterion and none of them is picked out by it; which point
    of that interval a search returns is an artefact of the search, not physics. Their
    recommendation is to place :math:`\mu_{dc}` at the middle of the gap of the impurity spectral
    function instead. For NiO they obtain 25.3 eV that way, against 20.4 eV from (SC)AMF and
    21.3 eV from the self-energy-based scheme -- and they show 21 eV gives a qualitatively wrong,
    Mott-like answer. That is a 4-5 eV discrimination between schemes on a real material, which is
    why this is a separate criterion and not a tolerance setting.

    .. warning:: **What this implementation actually measures is the gap of the whole cluster,
       not of the impurity.** The paper's prescription is the middle of the gap *of the impurity
       spectral function*; ``E[N +- 1]`` here are the ground-state energies of the whole
       impurity-plus-bath cluster one electron above and below ``N``, obtained from
       :meth:`_SectorContext.sector_energy` with no impurity projection of any kind. Two
       consequences, both real:

       * **The added electron need not land on the impurity.** In a bath-discretized model the
         lowest empty level near ``E_F`` is generically a *bath* level -- that is what a
         hybridization fit puts there -- and bath levels do not move with ``mu``, which couples
         only to :math:`\hat N_{imp}`. What gets centred on ``E_F`` is then largely the bath's
         finite-size level spacing: a discretization artefact.
       * **No spectral weight is checked.** Nothing evaluates
         :math:`|\langle N \pm 1| c_d |N\rangle|^2`, so an edge can be a pole of the *cluster* with
         no weight in the impurity Green's function at all -- by orbital character, or exactly zero
         by spin/irrep selection if the sector's ground state is not reachable from
         :math:`|N\rangle` with one impurity operator. :func:`fixed_peak_dc` risks this for one
         pole; averaging two edges risks it for either.

       The criterion measures its own exposure and reports it, per edge: ``delta_plus`` and
       ``delta_minus`` in the record are :math:`\delta_\pm`, the fraction of the added/removed
       electron that lands on the impurity, measured from the eigenvectors at the converged shift
       (:meth:`_SectorContext.sector_occupation`). **Read them before trusting the result**, and
       read them *separately*: the two edges fail independently, and their sum hides it.

    .. note:: **Measured, on real models.** The picture above is not hypothetical, and it is not
       uniformly bad either.

       *NiO 5-bath, iteration 1* (58 spin-orbitals, the archived production run): the criterion
       converges at ``mu = 0`` with a cluster gap of 1.62 eV symmetric about ``E_F``, and
       ``delta_+ = 0.561`` against ``delta_- = 0.032``. Enumerating the ``N-1`` spectrum
       (``test/support/dc_weights.py``) shows the four lowest removal states carrying **zero**
       impurity weight and lying within 4 meV of each other -- a *band*, not an isolated level.
       That is the O-2p valence band: NiO's valence-band top is ligand, which is what makes it a
       charge-transfer insulator. So the criterion is centring the p-to-d gap. That is a
       defensible quantity -- arguably the one wanted -- but it is **not** the gap of
       :math:`A_{imp}`, whose removal weight on this model is centred 5.8 eV below ``E_F`` and
       whose addition weight is centred at +1.75 eV (exact first moments, same module).
       Cap 2000 to 8000 moves the answer by 6 meV, so none of this is truncation.

       *The charge-transfer toy* (``charge_transfer_kwargs``) fails on the other edge:
       ``delta_+ = 0.075`` against ``delta_- = 0.951``, with the lowest addition state carrying
       15% of the addition sum rule and the state carrying 80% sitting 1.45 above it. There the
       low-lying state *is* an isolated fitted level -- the artefact case.

       Distinguishing the two is what ``dc_weights.py`` is for: a band shows many nearly
       degenerate states, an artefact shows one isolated level. **A weight-thresholded edge is not
       a usable replacement criterion**, at least not here -- on NiO no removal state reaches 5%
       of the sum rule, so a weighted gap centre has no root at all.

    **Which criterion for which regime.**

    ``gap``
        Mott insulators, and any case where **both** ``delta_+`` and ``delta_-`` show impurity
        character. This is the criterion the paper recommends for insulators, and it is well
        defined precisely where ``occ`` is not. In the *charge-transfer* regime the paper
        addresses, be explicit about which gap you want: this implementation centres the cluster
        charge gap, whose removal edge is ligand by construction there. If the p-to-d gap is what
        you mean, that is the right answer; if you mean the gap of :math:`A_{imp}`, this is not
        it, and no reweighting of these sector energies makes it so.
    ``occ``
        Metals, and wide-window p-d models in charge self-consistency. This is Karolak's Eq. (2)
        and it is exact where the occupation actually responds to the shift.
    ``<peak_position>`` (:func:`fixed_peak_dc`)
        When a specific measured excitation is being reproduced, e.g. a photoemission satellite
        at a known binding energy. It pins *one* pole; ``gap`` pins the midpoint of two.
    ``fll`` / ``amf`` / ``sigma_inf`` / ``nominal`` (:mod:`dc_static`)
        Closed-form schemes needing no ED solve at all -- the natural ``dc_guess``, and the
        reference to sanity-check a converged ED answer against.

    **Why the residual is the well-conditioned combination.** The ``E[N]`` terms cancel in
    :math:`(\omega_+ + \omega_-)/2`, leaving a *first* difference of two sector energies. The gap
    *width* :math:`\omega_+ - \omega_- = E[N+1] + E[N-1] - 2E[N]` does not: it is a second
    difference of three independently expanded, independently truncated CIPSI energies, so the
    truncation errors add instead of cancelling. The width is therefore **reported and never
    root-found** -- it is used only to size the energy tolerance
    (:data:`GAP_ENERGY_TOLERANCE_FRACTION`), where being off by a factor of two costs a step.

    Slope: :math:`d(\text{residual})/d\mu = -(\delta_+ + \delta_-)/2`, where
    :math:`\delta_\pm` is the fraction of the added/removed electron that lands on the impurity.
    Both are in ``(0, 1]``, so the slope is in ``[-1, -0.5]`` *when the gap edges are impurity
    states*, and there the criterion is better conditioned than :func:`fixed_peak_dc`, whose
    single-pole slope :math:`-\delta_\pm` is half as large.

    .. note:: That range is the impurity-like limit, and it is **not** what a bath-discretized
       charge-transfer insulator gives. Measured on the archived NiO 15-bath workload at a
       determinant cap of 1500: :math:`d(\text{centre})/d\mu = -0.109`, consistently over five
       evaluations all in the same charge sector, i.e. :math:`\delta_+ + \delta_- = 0.22`.

       By Hellmann-Feynman that slope *is* :math:`-(\langle N_{imp}\rangle_{N+1} -
       \langle N_{imp}\rangle_{N-1})/2`, so it is not a conditioning nuisance to divide out -- it
       is the measurement of how much of the transferred charge reached the impurity, and 0.22
       says most of it did not. Two separate costs follow. The benign one: the ``slope=-1`` passed
       below is only a Newton seed, and overestimating its magnitude makes the first step
       *short*, which the walk doubles out of (5 evaluations on that workload). The one that
       matters: the acceptance band in ``mu`` is ``tol/|slope|``, so a small slope multiplies the
       energy tolerance by ~9 on its way into the answer. The cap there is far below production --
       read the number as "this can happen and here is how to see it", not as NiO's converged
       value.

    ``offset`` is read the same way as :func:`fixed_peak_dc`'s ``peak_position``: an energy
    relative to the Fermi level, which this package fixes at zero. That convention is what makes a
    target of zero meaningful at all, and it is checked here (as in the other ED criteria) by
    :func:`dc_reference._warn_if_not_fermi_referenced`. Energies carry no fixed unit; they follow
    the inputs (Ry when called from RSPt).

    Parameters
    ----------
    model, basis, solver
        As :func:`fixed_peak_dc` -- the identical setup, built once by
        :func:`_prepare_sector_context` and shared with it, so the two criteria cannot drift apart
        on the ground-state conventions.
    offset : float
        Where to put the gap centre, relative to the Fermi level. ``0.0`` (the default) centres
        the gap on E_F, which is what the paper prescribes.
    allow_charge_state_change : bool
        As :func:`fixed_peak_dc`, and for the same reason: the centre sector ``N`` is re-found at
        every trial ``mu``, so this criterion is multivalued and a converged ``mu`` on a
        non-nominal charge state is rejected by default rather than silently returned.
    report : dict, optional
        Filled with **exactly the record this criterion emits** (:mod:`dc_record`) -- same keys,
        same normalized ``status``, plus ``walltime``. It is written from ``recording``'s
        ``finally``, so a caller that catches :class:`dc_search.DoubleCountingUnreachable` still
        learns how far the search got. All three criteria accept it and agree on its vocabulary;
        this one used to have a private dialect of its own (``gap_centre``, ``energy_tol``, the
        search's raw internal status), which is how a caller could assert on a field that had
        quietly stopped being written.

    Returns
    -------
    dc : ndarray
        The double counting matrix, ``dc_guess + mu * identity``.

    Raises
    ------
    dc_search.DoubleCountingUnreachable
        If the gap centre cannot be brought to ``offset`` within the reachable range -- including
        the case where one of the two off-centre sectors does not exist (a full or empty
        impurity), where the criterion is *undefined* rather than merely far off.
    RuntimeError
        If the converged ``mu`` centres the gap on a non-nominal charge state and
        ``allow_charge_state_change`` is ``False``.
    """
    with dc_record.recording(
        "gap", rank=comm.rank if comm is not None else MPI.COMM_WORLD.rank, report=report
    ) as dc_rec:
        ctx = _prepare_sector_context(model, basis, solver, comm=comm, verbosity=verbosity, memory_label="fixed-gap dc")
        rank, verbose, tau = ctx.rank, ctx.verbose, ctx.tau

        # The residual *is* a sector energy difference, so it inherits the E_F = 0 convention every
        # sector comparison silently depends on -- and here the target of zero means nothing without
        # it, since "the Fermi level" is where the gap is being centred.
        _warn_if_not_fermi_referenced(model.h0, model.n_spin_orbitals, rank=rank)

        n_center_at = ctx.n_center_at
        # mu -> (n_center, E[N+1], E[N-1]). _solve_dc_shift never re-evaluates the same mu, but the
        # tolerance-sizing evaluation below happens *before* the search starts, so the seed point
        # would otherwise be paid for twice. Keyed on the broadcast mu, so it is rank-identical.
        sectors_at = {}
        # Likewise for the width's extra E[N] solve: the tolerance is sized at mu = 0, and the search
        # returns mu = 0 whenever the guess already satisfies the criterion -- the common case in a
        # converged self-consistency loop, where paying for the same solve twice is pure waste.
        width_at = {}

        def _gap_sectors_at(mu):
            if mu not in sectors_at:
                h_op = ctx.shifted_h(mu)
                n_center = ctx.centre_sector(h_op, mu)
                # Both unconditionally, and only then tested for None: each is a collective solve, so
                # an early return between them would move calc_energy off some ranks (CLAUDE.md's MPI
                # rule). The gap needs *both* off-centre sectors, so unlike the peak it is undefined
                # at either shell edge.
                e_add = ctx.sector_energy(h_op, mu, n_center + 1)
                e_rem = ctx.sector_energy(h_op, mu, n_center - 1)
                sectors_at[mu] = (n_center, e_add, e_rem)
            return sectors_at[mu]

        def _gap_centre_at_mu(mu):
            n_center, e_add, e_rem = _gap_sectors_at(mu)
            if e_add is None or e_rem is None:
                if verbose and rank == 0:
                    missing = "N+1" if e_add is None else "N-1"
                    print(f"mu={mu:.6f} n_center={n_center} {missing} sector unreachable: gap undefined here")
                return None
            centre = 0.5 * (e_add - e_rem)
            if verbose and rank == 0:
                print(f"mu={mu:.6f} n_center={n_center} gap centre=(E[N+1] - E[N-1])/2={centre:.6f}")
            return centre

        def _gap_width_at(mu):
            """``omega_+ - omega_- = E[N+1] + E[N-1] - 2 E[N]``: reported, never root-found.

            The one place ``E[N]`` is needed, and it is deliberately not taken from the centre walk:
            the walk runs at ``tau / 100`` and ``sqrt(slaterWeightMin)`` while the sector solves run
            at ``tau`` and ``slaterWeightMin``, so an ``E[N]`` from the walk mixed with ``E[N +- 1]``
            from ``calc_energy`` would make this second difference straddle two different truncation
            conventions -- strictly worse than the conditioning problem it is already exposed to.
            Costs one extra solve, paid exactly twice per search (once to size the tolerance, once at
            the converged shift for the record), not once per evaluation.
            """
            if mu not in width_at:
                n_center, e_add, e_rem = _gap_sectors_at(mu)
                if e_add is None or e_rem is None:
                    # The width is undefined whatever E[N] turns out to be, so the third solve is
                    # skipped. Skipping a *collective* on a condition is only safe because
                    # `sector_energy` broadcasts its result: the `is None` verdict is rank-
                    # identical, so no rank is left waiting inside a solve the others skipped.
                    # Before that broadcast this early return would have been the deadlock the
                    # class docstring warns about, which is why it is written this way now and
                    # was not before.
                    width_at[mu] = None
                else:
                    e_centre = ctx.sector_energy(ctx.shifted_h(mu), mu, n_center)
                    width_at[mu] = None if e_centre is None else (e_add - e_centre) - (e_centre - e_rem)
            return width_at[mu]

        def gap_observable(mu):
            with solver_trace.labelled(mu=mu), solver_trace.timed("dc_evaluation") as evaluation_fields:
                centre = _gap_centre_at_mu(mu)
                evaluation_fields["gap_centre"] = centre
                return centre

        mu_tol = bracket_width_tol(tau)
        unreachable = (
            "The fixed-gap double counting could not centre the gap at {target}: (E[N+1] - E[N-1])/2 "
            "reached {value:.4f} at mu = {mu:.3f}. The impurity may be full or empty over the "
            "reachable range (one of the N +- 1 sectors then does not exist and the gap centre is "
            "undefined), or the requested offset lies beyond it."
        )
        # Held by the record so the emit in its `finally` can read the search's own verdict even on
        # the path where the raise skips past every line that would have projected it.
        search_report = dc_rec.search = {}
        dc_rec["target"] = offset
        with _dc_search_trace("fixed-gap", MPI.COMM_WORLD, rank, width_tol=mu_tol, report=search_report):
            # Inside the trace block, not before it. The tolerance-sizing evaluation costs three
            # full sector solves, and `_report_dc_trace`'s cross-rank witness -- which allreduces
            # the `sector_solve` count precisely to catch ranks walking different collective paths
            # -- can only count what a trace is open for. Sizing the tolerance outside meant the
            # one evaluation whose result gates every later branch was invisible to the check
            # designed to notice it diverging.
            #
            # Size the energy tolerance against the gap the model actually has. Falls back to the
            # mu resolution when the width is undefined at the guess (a shell edge, or a sector
            # the restrictions admit no determinants for) -- the search's own undefined-start
            # handling then takes over.
            width_at_guess = _gap_width_at(0.0)
            energy_tol, dc_rec["tol_basis"] = _size_gap_tolerance(width_at_guess, mu_tol, ctx.bandwidth, rank=rank)
            dc_rec["tol"] = energy_tol

            mu = None
            try:
                mu = _solve_dc_shift(
                    gap_observable,
                    offset,
                    # The two tolerances are genuinely different quantities here: how well the gap
                    # centre is placed (an energy, scaled by the gap it sits in) and how finely mu
                    # is resolved.
                    tol=energy_tol,
                    width_tol=mu_tol,
                    initial_step=max(10 * tau, abs(offset)),
                    max_shift=max(ctx.bandwidth, 10 * abs(offset), 1.0),
                    # A gap centre that does not move with mu is an anomaly to surface, not a
                    # plateau to take the midpoint of: it means both sectors shift together, i.e.
                    # the impurity is decoupled and no uniform shift can place the gap.
                    plateau_ok=False,
                    unreachable_message=unreachable,
                    rank=rank,
                    # Sector-first, for the same reason as the peak: this criterion is
                    # multivalued, so "which charge state" is decided before "which root".
                    sector_of=n_center_at.get,
                    nominal_sector=ctx.nominal_total if not allow_charge_state_change else None,
                    # d(centre)/dmu = -(delta_+ + delta_-)/2, in [-1, -0.5]. The larger magnitude
                    # is the safe end: it makes the first step short, which the walk doubles out
                    # of, where an overshoot has to be walked back. It matters more here than for
                    # the peak because offset = 0 collapses initial_step to 10*tau, so the seed is
                    # doing all the work.
                    slope_sign=-1,
                    slope=-1.0,
                    report=search_report,
                    verbose=verbose,
                    # The observable's collectives run on COMM_WORLD (the basis builds hardcode
                    # it), not on the caller's `comm`, so the residual must be broadcast there too.
                    comm=MPI.COMM_WORLD,
                )
            finally:
                # In a `finally`, and inside the trace, for two separate reasons.
                #
                # *Inside the trace*: `_report_dc_trace`'s cross-rank witness allreduces the
                # sector-solve count precisely to catch ranks walking different collective paths,
                # and it can only count what a trace is open for. The edge measurement is normally
                # free -- every sector it reads was already solved at this `mu` -- but the centre
                # sector is solved lazily (see `_gap_width_at`), so it *can* miss the cache and
                # solve. A collective outside the witness is exactly what moving the
                # tolerance-sizing evaluation inside it was meant to stop.
                #
                # *In a `finally`*: `DoubleCountingUnreachable` skips every line below, and a
                # bath-like edge that does not respond to `mu` is a leading cause of
                # unreachability. The diagnostic was absent from the record on precisely the runs
                # that most needed it. The raise is rank-identical (the residual is broadcast), so
                # every rank reaches this together and a collective here is safe.
                #
                # `sectors_at` is insertion-ordered by evaluation, so its last key is the last
                # shift actually evaluated -- the best available stand-in for `mu` when the search
                # never returned one.
                mu_seen = mu if mu is not None else (next(reversed(sectors_at)) if sectors_at else None)
                if mu_seen is not None:
                    delta_plus, delta_minus = _measure_edge_character(ctx, mu_seen, n_center_at[mu_seen])
                    dc_rec["delta_plus"], dc_rec["delta_minus"] = delta_plus, delta_minus
                    _warn_if_a_gap_edge_is_not_impurity_like(delta_plus, delta_minus, rank)

        # mu is always a point _solve_dc_shift actually evaluated, so sectors_at holds it.
        n_center, e_add, e_rem = _gap_sectors_at(mu)
        nominal_total = ctx.nominal_total
        gap_width = _gap_width_at(mu)
        dc = ctx.dc(mu)
        # The evaluated gap centres, from the solves the search already paid for. `sectors_at` is
        # the cache those two sector energies live in, so no third dict has to be kept in step
        # with it.
        centre_at = {
            trial: 0.5 * (add - rem)
            for trial, (_n, add, rem) in sectors_at.items()
            if add is not None and rem is not None
        }
        chi, chi_span = _dc_chi(
            centre_at, mu, width_tol=mu_tol, in_sector=lambda trial: n_center_at.get(trial) == n_center_at.get(mu)
        )
        dc_rec.update(
            gap_center=centre_at.get(mu),
            gap_width=gap_width,
            omega_plus=None if gap_width is None else centre_at[mu] + 0.5 * gap_width,
            omega_minus=None if gap_width is None else centre_at[mu] - 0.5 * gap_width,
            chi=chi,
            chi_span=chi_span,
        )
        # `delta_plus` / `delta_minus` were measured and recorded inside the search block above,
        # from the eigenvectors of the solves the search had already paid for. All that is left
        # here is the sum, which needs `chi` for its fallback and so cannot be done up there.
        delta_plus, delta_minus = dc_rec.get("delta_plus"), dc_rec.get("delta_minus")
        if delta_plus is not None and delta_minus is not None:
            dc_rec["delta_sum"] = delta_plus + delta_minus
            # Two estimators of the same quantity, with opposite error structure: `delta_sum` is
            # built from occupations, first-order in the wavefunction error, while `-2 chi` is a
            # secant of energies, second-order in it but carrying a finite-difference and a
            # basis-reselection term instead. Agreement is real evidence that the sector solves
            # are converged; disagreement localizes which. Recorded rather than silently resolved
            # in favour of one -- the criterion previously kept both and compared neither.
            if chi is not None:
                dc_rec["delta_sum_vs_chi"] = (delta_plus + delta_minus) - (-2.0 * chi)
        elif chi is not None:
            # Fall back to the secant only where the direct measurement is undefined (a sector at
            # a shell edge), so the field is never silently absent when something can be said.
            dc_rec["delta_sum"] = -2.0 * chi
        dc_rec["dc_trace"], dc_rec["dc_level"] = dc_record.dc_levels(dc)
        dc_rec["dc_spread"] = dc_record.dc_spread(dc)
        _dump_dc_matrices(ctx.dc_guess, dc, rank)
        if n_center != nominal_total and not allow_charge_state_change:
            raise RuntimeError(
                f"fixed_gap_dc found mu = {mu:.6f} (dc = dc_guess + mu), but the centre charge state "
                f"there is n_center = {n_center}, not the nominal total {nominal_total}. Like the "
                "fixed-peak criterion this one is multivalued -- the gap of a different charge state "
                "can be centred on the same offset, several eV away in dc. Pass "
                "allow_charge_state_change=True to accept this result anyway, or adjust "
                "dc_guess/offset to land nearer the nominal sector."
            )

        return (dc, n_center) if return_sector else dc


@dataclass
class _OccupationContext:
    """The mu-independent setup a fixed-occupation evaluation needs, shared between
    :func:`fixed_occupation_dc`'s search and the single-mu diagnostic
    :func:`occupation_and_energy_at_mu` (B2's convergence sweep,
    ``test/support/dc_diagnostics.py``): the solver-basis layout, frozen shells, symmetry
    generators, the derived determinant cap, and the weighted restrictions. Building this once
    and evaluating many ``mu`` against it (the search) or one ``mu`` (the diagnostic) keeps both
    on the identical code path -- the diagnostic is not a reimplementation of the search's setup.
    """

    h0_op: object
    h_op_i: object
    impurity_orbitals: dict
    bath_states: dict
    N0: dict
    frozen_occupations: set
    mixed_valence: dict
    tau: float
    chain_restrict: bool
    dense_cutoff: int
    spin_flip_dj: bool
    slaterWeightMin: float
    truncation_threshold: int
    weighted_restrictions: object
    symmetry_generators: object
    impurity_indices: list
    energy_cut: float
    identity: np.ndarray
    total_impurity_orbitals: int
    n0: float
    # Decides *which* ground state is found, so it has to carry the same value calc_gs will use
    # -- the DC is only meaningful if calc_selfenergy lands on the same state. Populated from
    # groundstate.GS_CIPSI_SOLVER_METHOD (imported lazily; groundstate imports back into this
    # module), never spelled out as a literal here.
    cipsi_solver_method: str


def _prepare_occupation_context(model, basis, solver, comm=None, verbosity=0):
    """Build the :class:`_OccupationContext` for a fixed-occupation evaluation.

    Verbatim extraction of the setup :func:`fixed_occupation_dc` used to build inline: the same
    solver-basis derivation, frozen-shell/symmetry-generator computation, and determinant-cap
    derivation :func:`groundstate.calc_gs` uses, so a fixed-mu diagnostic built on this context
    measures on exactly the variational space the production search would have used at that
    ``mu`` -- not an approximation of it.
    """
    h0_op = model.h0
    u = model.u4
    n_imp = len(model.impurity_indices)
    N0 = basis.nominal_occ
    mixed_valence = basis.mixed_valence
    spin_flip_dj = basis.spin_flip_dj
    tau = basis.tau
    excitation_budget = basis.excitation_budget
    chain_restrict = basis.chain_restrict
    slaterWeightMin = basis.slater_weight_min
    dense_cutoff = solver.dense_cutoff
    verbose = verbosity > 0

    # Derive the same solver-basis layout calc_selfenergy uses (the DC search must measure its
    # observable on the same sector/basis the downstream selfenergy solve will use, or "fixed"
    # does not mean what it says). The mu shift is a uniform impurity shift (-mu * identity),
    # which commutes with the impurity block and hence with any rotation prepare_solver_basis
    # applies, so the layout derived once here at the guess dc is valid for every trial mu.
    sb = prepare_solver_basis(
        h0_op, model.dc, u, model.impurity_orbitals, N0, mixed_valence, model.rot_to_spherical, verbosity
    )
    impurity_orbitals = sb.impurity_orbitals
    bath_states = sb.bath_states
    N0 = sb.nominal_occ
    mixed_valence = sb.mixed_valence
    h_op_i = sb.h

    total_impurity_orbitals = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)
    impurity_indices = [orb for orb_blocks in impurity_orbitals.values() for block in orb_blocks for orb in block]

    # The three inputs that make this the same call calc_gs makes (see the module warning):
    # a bath-less group pinned exactly as get_spectra.py pins it, the same one-body symmetry
    # generators (valid at every trial mu -- a uniform impurity shift commutes with them), and
    # the walk's tau divided by 100, restored to the full tau for everything solved at the sector
    # the walk finds.
    frozen_occupations = {i for i in impurity_orbitals if sb.sum_bath_states[i] == 0}
    symmetry_generators = get_symmetry_generators(h_op_i, impurity_orbitals, bath_states)

    truncation_threshold = basis.truncation_threshold
    if truncation_threshold is None:
        truncation_threshold = suggest_truncation_threshold(model.n_spin_orbitals, comm=MPI.COMM_WORLD)
        log_memory_budget(
            truncation_threshold,
            model.n_spin_orbitals,
            comm=MPI.COMM_WORLD,
            verbose=verbose,
            label="fixed-occupation dc",
        )

    # DFT reference occupation: Fermi filling of the raw h0 (the KS Hamiltonian of the
    # h0 - dc + U contract; no double counting subtracted before filling), independent of
    # dc_guess and of the trial shift, and of the solver-basis rotation (h0's raw filling is
    # evaluated in the model's input basis).
    n0 = _noninteracting_impurity_occupation(h0_op, impurity_indices, model.n_spin_orbitals, tau)

    # Imported here, not at module scope: groundstate imports back into this module.
    from impurityModel.ed.groundstate import GS_CIPSI_SOLVER_METHOD

    return _OccupationContext(
        h0_op=h0_op,
        h_op_i=h_op_i,
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        N0=N0,
        frozen_occupations=frozen_occupations,
        mixed_valence=mixed_valence,
        tau=tau,
        chain_restrict=chain_restrict,
        dense_cutoff=dense_cutoff,
        spin_flip_dj=spin_flip_dj,
        slaterWeightMin=slaterWeightMin,
        truncation_threshold=truncation_threshold,
        weighted_restrictions=build_weighted_restrictions(bath_states, excitation_budget),
        symmetry_generators=symmetry_generators,
        impurity_indices=impurity_indices,
        energy_cut=boltzmann_energy_cut(tau),
        identity=np.identity(n_imp),
        total_impurity_orbitals=total_impurity_orbitals,
        n0=n0,
        cipsi_solver_method=GS_CIPSI_SOLVER_METHOD,
    )


def _evaluate_occupation_and_energy_at_mu(ctx, mu, verbose, rank):
    """Evaluate the true, re-expanding occupation observable at one ``mu``: the identical
    HF-seed-then-walk :func:`groundstate.find_ground_state_basis` search, refinement, and thermal
    average :func:`fixed_occupation_dc` performs per trial shift -- not a cheaper approximation.
    Returns ``(n, E0, sector)``: the achieved thermal impurity occupation, the lowest eigenvalue
    found, and the *walk's* winning sector total (``sum(mb_basis.ground_state_occupation.values())``
    -- the integer charge state the search settled on, which need not equal ``round(n)`` when the
    thermal average mixes competing sectors near a crossing).

    .. note::
       The observable's own collectives run on ``MPI.COMM_WORLD``, matching
       :func:`fixed_occupation_dc`'s pre-existing behaviour (not the caller's ``comm`` -- a latent,
       out-of-scope concern recorded in the campaign plan, not introduced here).
    """
    from impurityModel.ed.groundstate import solve_ground_state

    h_op = ctx.h_op_i - tensors_to_operator(np.asarray(mu * ctx.identity, dtype=complex))

    # Determine the ground state the SAME way calc_selfenergy will -- literally the same
    # function, `groundstate.solve_ground_state`, which owns the walk/refine/thermal sequence and
    # every knob that decides which state comes out. A dc measured on a different state than the
    # one calc_selfenergy later finds does not just add noise: it locks the downstream
    # calculation onto the wrong charge state, which is worse than not fixing anything at all.
    #
    # N0 stays the *original* nominal occupation on every call (never a carried-forward sector
    # from the previous mu): feeding the walk its own last answer would make the result
    # path-dependent on the bracket's evaluation order, which breaks parity by another route.
    with solver_trace.timed("expand", stage="dc_refine"):
        mb_basis, _mb_solver, es, psis = solve_ground_state(
            h_op,
            ctx.impurity_orbitals,
            ctx.bath_states,
            ctx.N0,
            tau=ctx.tau,
            slaterWeightMin=ctx.slaterWeightMin,
            frozen_occupations=ctx.frozen_occupations,
            mixed_valence=ctx.mixed_valence,
            chain_restrict=ctx.chain_restrict,
            dense_cutoff=ctx.dense_cutoff,
            spin_flip_dj=ctx.spin_flip_dj,
            comm=MPI.COMM_WORLD,
            verbose=verbose,
            truncation_threshold=ctx.truncation_threshold,
            weighted_restrictions=ctx.weighted_restrictions,
            cipsi_solver_method=ctx.cipsi_solver_method,
        )

    rhos = build_density_matrices(mb_basis, psis, ctx.impurity_indices, ctx.impurity_indices)
    rho = thermal_average_scale_indep(es, rhos, ctx.tau)
    n_out = float(np.real(np.trace(rho)))
    e0 = float(es[0])
    sector = sum(mb_basis.ground_state_occupation.values())
    return n_out, e0, sector


def occupation_and_energy_at_mu(model, basis, solver, mu, *, comm=None, verbosity=0):
    """Evaluate the true occupation observable at **one fixed** ``mu`` -- no search.

    This is the per-``mu`` core of :func:`fixed_occupation_dc`, exposed standalone for B2's
    convergence diagnostic (``test/support/dc_diagnostics.py``'s ``occupation_convergence``
    mode): does the achieved occupation and ground-state energy drift with
    ``BasisOptions.truncation_threshold``/``excitation_budget``/``chain_restrict`` at a fixed
    ``mu``? If ``n`` is not converged with respect to the ED's own variational truncation, the
    search absorbs that drift into ``dc`` rather than into physics -- a fixed-``mu`` sweep across
    the cap ladder is the cheap way to check, since it needs no search at all (Haule 2015's
    truncation-as-fitting-parameter pathology).

    Parameters
    ----------
    mu : float
        The fixed trial shift; ``H(mu) = H(0) - mu * N_imp``.

    Returns
    -------
    (n, E0, sector) : tuple
        Achieved thermal impurity occupation, the lowest eigenvalue, and the walk's winning
        sector total at this ``mu`` (see :func:`_evaluate_occupation_and_energy_at_mu`).
    """
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0
    ctx = _prepare_occupation_context(model, basis, solver, comm=comm, verbosity=verbosity)
    return _evaluate_occupation_and_energy_at_mu(ctx, mu, verbose, rank)


def fixed_occupation_dc(
    model,
    basis,
    solver,
    *,
    occupation=None,
    comm=None,
    verbosity=0,
    occ_tol=1e-2,
    initial_step=0.25,
    max_shift=20.0,
    return_sector=False,
    report=None,
):
    r"""
    Calculate the double counting correction using a fixed (or self-consistent) impurity
    occupation criterion.

    With an explicit ``occupation``, choose the double counting so that the interacting thermal
    impurity occupation equals the requested value, :math:`\mathrm{Tr}\,\rho_{imp} = N_{target}`.

    With ``occupation=None``, pin it to the DFT reference occupation instead: :math:`N(\mu) =
    N_0`, where :math:`N_0` is the Fermi filling of the *raw* non-interacting ``h0`` at
    :math:`\mu_{chem} = 0` (:func:`_noninteracting_impurity_occupation`), computed once before
    the search -- the DFT impurity occupation the CSC DFT+DMFT self-consistency loop targets.
    ``h0`` is the KS/DFT Hamiltonian of the ``h0 - dc + U`` contract, so no double counting is
    subtracted before filling: :math:`N_0` is independent of both ``dc_guess`` and the trial
    shift. Note that :math:`N_0` is a property of the *discretized* bath: a coarse valence-only
    fit may place no impurity spectral weight across the Fermi level at all, saturating
    :math:`N_0` at the full (or empty) shell -- a threshold effect of the discretization, not a
    smooth error. A warning is printed in that case; supply an explicit ``occupation`` target
    instead.

    The double counting is parametrized as a uniform shift of the guess, ``dc(mu) = dc_guess +
    mu * identity``, coupling to the impurity occupation as :math:`-\mu \hat N_{imp}`, so the
    residual :math:`N(\mu) - N_{target}` is monotone in ``mu`` for either target and is driven
    to zero by the search :func:`_solve_dc_shift`. At low temperature and weak
    hybridization the occupation approaches a staircase in ``mu``; if the requested (fractional)
    occupation falls on a plateau -- the observable steps across the target at a charge-sector
    boundary rather than crossing it -- the search reports the jump, the boundary and the
    distance to it on rank 0, then raises :class:`dc_search.DoubleCountingUnreachable` (B3: a
    target with no solution must be signalled as such to the caller, not merely narrated and
    silently approximated).

    Unlike :func:`fixed_peak_dc`, the sector this search measures the occupation on is **not**
    fixed at the input ``model.impurity_orbitals``/``model.bath_states``: at every trial ``mu``
    it derives the same solver-basis layout :func:`calc_selfenergy` uses
    (:func:`solver_basis.prepare_solver_basis` -- the impurity/bath grouping, the bath valence/
    conduction split from the sign of the bath on-site energy rather than the hybridization fit,
    and any symmetry-adapted rotation), then determines the ground-state sector by calling
    :func:`groundstate.find_ground_state_basis` itself -- the identical HF-seed-then-walk search
    :func:`groundstate.calc_gs` uses for the selfenergy/spectra solve, not a cheaper
    approximation of it. A dc measured on a sector different from the one calc_selfenergy later
    finds would not just be imprecise: it would lock the downstream calculation onto the wrong
    charge state, so this search pays the full per-trial cost of walking rather than caching a
    single Hartree-Fock seed across mu. This is what makes the "fixed" occupation the same
    quantity :func:`groundstate.find_ground_state_basis` finds for the selfenergy/spectra solve
    at the returned ``dc`` -- the point of this whole search. The search follows wherever that
    walk leads, including across a change in total electron number (matching
    ``find_ground_state_basis``'s own grand-canonical-style sector search); the reachable
    occupations are limited by the bath, and a tight
    ``BasisOptions.excitation_budget``/``chain_restrict`` can itself make a requested occupation
    unreachable, or reachable only via a sector the un-walked search could never have found.

    Parameters other than the following match :func:`fixed_peak_dc` (``model``, ``basis``,
    ``solver``, ``comm``, ``verbosity``); neither function requires ``model.bath_states`` (the
    valence/conduction split used by both is the one derived at solve time, not the hybridization
    fit). Unlike :func:`fixed_peak_dc`, ``basis.truncation_threshold=None`` here derives the cap
    without halving the safety fraction -- this search does not hold an upper and a lower sector
    basis simultaneously.

    Parameters
    ----------
    occupation : float or None
        Requested impurity occupation (may be fractional). ``None`` targets the DFT reference
        occupation :math:`N_0` instead (see above).
    occ_tol : float
        Convergence tolerance on the occupation.
    initial_step : float
        First bracketing step for ``mu``, in the energy units of the
        Hamiltonian (energies here carry no fixed unit, they follow the
        inputs -- e.g. Ry when called from RSPt). A small fraction of the
        bandwidth is a good choice.
    max_shift : float
        Bracketing gives up if ``|mu|`` exceeds this, in the energy units of
        the Hamiltonian (the requested occupation is then unreachable).

    Returns
    -------
    dc : ndarray
        The double counting matrix, ``dc_guess + mu * identity``.

    Raises
    ------
    dc_search.DoubleCountingUnreachable
        If the target cannot be bracketed within ``max_shift``, or the requested occupation falls
        on a charge-sector plateau the observable steps across rather than crosses (B3: this is a
        modelling verdict -- "no solution here" -- not a tolerable miss to approximate silently).
    """
    with dc_record.recording(
        "occupation", rank=comm.rank if comm is not None else MPI.COMM_WORLD.rank, report=report
    ) as dc_rec:
        n_imp = len(model.impurity_indices)
        dc_guess = extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0]
        rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
        verbose = verbosity > 0

        ctx = _prepare_occupation_context(model, basis, solver, comm=comm, verbosity=verbosity)

        self_consistent = occupation is None
        if self_consistent:
            occupation = ctx.n0
            # Only when N0 is the *target*: with an explicit target a saturated reference is merely
            # logged, not acted on.
            # The E_F = 0 convention underpins both the reference filling below and every sector
            # comparison the occupation walk makes; it is asserted throughout and was never checked.
            _warn_if_not_fermi_referenced(ctx.h0_op, model.n_spin_orbitals, rank=rank)
            # Saturation is the more specific diagnosis, so it wins when both would fire; the
            # nominal-gap check catches the reference that is grossly wrong *without* being pinned at
            # a shell edge, which the saturation test alone lets through (a runaway CSC iterate
            # reporting n0 = 1.54 against a nominal 8 converged silently to a 43 eV shift).
            if not _warn_if_reference_saturated(
                ctx.n0, ctx.total_impurity_orbitals, _SATURATION_ADVICE["search"], occ_tol=occ_tol, rank=rank
            ):
                _warn_if_reference_far_from_nominal(
                    ctx.n0, sum(ctx.N0.values()), _SATURATION_ADVICE["search"], rank=rank
                )
        # Tolerance absorbs the roundoff of a target derived elsewhere the same way
        # _noninteracting_impurity_occupation is (a sum of Fermi occupations, each in [0, 1]).
        if not -1e-9 <= occupation <= ctx.total_impurity_orbitals + 1e-9:
            raise ValueError(f"Requested impurity occupation {occupation} outside [0, {ctx.total_impurity_orbitals}].")

        occupation_at = {}
        # The walk's winning sector at each evaluated mu (M1): the sector at the RETURNED mu is the
        # number this whole branch exists to establish -- DC <-> GS parity is a statement about which
        # charge state the downstream calc_selfenergy will find. find_ground_state_basis already
        # returns it (.ground_state_occupation), previously discarded here.
        sector_at = {}

        def occupation_observable(mu):
            with solver_trace.labelled(mu=mu), solver_trace.timed("dc_evaluation") as evaluation_fields:
                n_out, _e0, sector = _evaluate_occupation_and_energy_at_mu(ctx, mu, verbose, rank)
                occupation_at[mu] = n_out
                sector_at[mu] = sector
                if verbose and rank == 0:
                    # The sector itself is reported by find_ground_state_basis's own verbose output
                    # above ("Ground state occupation"), evaluated fresh at this mu.
                    print(f"mu={mu:.6f} n={n_out:.6f} n0={ctx.n0:.6f}")
                evaluation_fields["n"] = n_out
                return n_out

        # The same bracket-collapse resolution _solve_dc_shift uses below (width_tol); computed once
        # so the trace's chi report filters against the identical value, not a second copy of it.
        occ_width_tol = bracket_width_tol(ctx.tau)

        target = occupation
        if self_consistent:
            unreachable = (
                f"Could not reach the DFT reference occupation N0 = {ctx.n0:.4f} with |mu| <= "
                f"{max_shift}: " + "the occupation reached {value:.4f} at mu = {mu:.3f}. "
                "The target may be unreachable with the available bath states."
            )
        else:
            unreachable = (
                "Could not bracket the requested impurity occupation {target} with "
                f"|mu| <= {max_shift}: " + "the occupation reached {value:.4f} at mu = {mu:.3f}. "
                "The target may be unreachable with the available bath states."
            )
        # Held by the record so the emit in its `finally` can read the search's own verdict even on
        # the path where the raise skips past every line that would have projected it.
        search_report = dc_rec.search = {}
        dc_rec["target"] = target
        dc_rec["tol"] = occ_tol
        dc_rec["n_ref"] = ctx.n0
        with _dc_search_trace("fixed-occupation", MPI.COMM_WORLD, rank, width_tol=occ_width_tol, report=search_report):
            mu = _solve_dc_shift(
                occupation_observable,
                target,
                tol=occ_tol,
                width_tol=occ_width_tol,
                initial_step=max(10 * ctx.tau, initial_step),
                max_shift=max_shift,
                plateau_ok=True,
                unreachable_message=unreachable,
                rank=rank,
                # Sector-first, and the nominal occupation is what "prefer the user's nominal" means:
                # the criterion is solved inside the mu-interval where the walk lands on that charge
                # state, so a root at another sector cannot win by sitting nearer mu = 0.
                sector_of=sector_at.get,
                nominal_sector=sum(ctx.N0.values()),
                # dn/dmu >= 0: a positive mu lowers the impurity level and raises its occupation.
                # No analytic magnitude here -- chi is precisely what the plateau/gap question is
                # about, and assuming a value for it is how a search talks itself past a flat region.
                # Direction only; the walk finds the scale.
                slope_sign=1,
                report=search_report,
                verbose=verbose,
                # The observable's own collectives run on COMM_WORLD (the basis builds below
                # hardcode it), not on the caller's `comm`, so the residual must be broadcast
                # there too or the branch decisions can diverge between ranks.
                comm=MPI.COMM_WORLD,
            )

        # mu is always a point _solve_dc_shift actually evaluated (the mu=0 fast path, a direct scan
        # hit, or a refined bracket point), so occupation_observable(mu) ran and cached it here.
        n = occupation_at[mu]
        sector = sector_at[mu]
        dc = dc_guess + mu * ctx.identity

        # M1: the sector at the RETURNED mu is the number this whole branch exists to establish -- DC
        # <-> GS parity is a statement about which charge state calc_selfenergy will find at this same
        # dc. Warn, not raise: unlike fixed_peak_dc, following the walk across sectors here is
        # deliberate (the module docstring), and the thermally averaged N need not equal the walk's
        # winning sector total near a crossing -- that disagreement is itself worth surfacing, not an
        # error. Unconditional on rank 0, matching the other reference-quality warnings in this module.
        nominal_total = sum(ctx.N0.values())
        if rank == 0 and sector != nominal_total:
            print(
                f"WARNING: fixed_occupation_dc found mu = {mu:.6f} (dc = dc_guess + mu), but the "
                f"walk's winning sector there totals {sector}, not the nominal occupation "
                f"{nominal_total}. calc_selfenergy will find this same sector at the returned dc.",
                flush=True,
            )

        # chi is recorded unconditionally, not only on a miss: a *converged* occupation with a small
        # chi still leaves dc essentially undetermined (delta_mu = delta_residual / chi), which is
        # exactly the case a miss-only report would hide (review correction 5). occupation_at holds
        # every mu the TRUE, re-expanding observable actually visited, so this is free -- no frozen
        # space, no extra solve -- and it is strictly better evidence than a frozen-space surrogate
        # would be. width_tol matches _solve_dc_shift's own bracket resolution and the sector
        # predicate keeps the difference inside one charge state, so neither a collapsed bracket at a
        # sector boundary nor a step across one is read as a slope (see _dc_chi).
        occ_chi, occ_chi_span = _dc_chi(
            occupation_at,
            mu,
            width_tol=occ_width_tol,
            in_sector=lambda trial: sector_at.get(trial) == sector_at.get(mu),
        )
        dc_rec.update(
            n_gs=n,
            chi=occ_chi,
            chi_span=occ_chi_span,
        )
        dc_rec["dc_trace"], dc_rec["dc_level"] = dc_record.dc_levels(dc)
        dc_rec["dc_spread"] = dc_record.dc_spread(dc)
        # No "achieved occupation misses the target" line (B3): _solve_dc_shift only returns via a
        # path where |n - target| <= occ_tol -- a miss beyond that raises DoubleCountingUnreachable
        # instead, so the record's n_gs never has a genuine miss hidden in it.
        _dump_dc_matrices(dc_guess, dc, rank)
        return (dc, sector) if return_sector else dc
