r"""Shared setup of the ED-based double-counting criteria (peak, gap, occupation).

The criteria pin a peak, the gap centre, or the occupation.

All three are special cases of :func:`dc_search._solve_dc_shift` over the uniform shift
``dc(mu) = dc_guess + mu * P`` (``P`` = the identity on one shell). :func:`fixed_peak_dc` pins a chosen spectral peak --
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
:func:`groundstate.calc_gs` uses -- :func:`groundstate.find_ground_state_sector`'s
HF-seed-then-walk search, not a search pinned at the input occupation. That matters because a
``dc`` measured on a different sector than the one ``calc_selfenergy`` later finds does not
approximate the requested physics, it misdirects it: the downstream calculation is locked onto the
wrong charge state, which is worse than not fixing anything at all.

.. warning::
   Until this fix, it was the same function but **not the same call** -- an earlier version of
   this docstring claimed otherwise. The differences below, all found by review, are now closed:

   * ``calc_gs`` divides ``tau`` by 100 before the walk (``groundstate.py``, ``calc_gs``); both
     criteria now do the same for their own ``find_ground_state_sector`` call, restoring the full
     ``tau`` for everything computed afterward at the found sector (the thermal average, the
     peak's sector-energy solves).
   * The occupation walk runs at the full accuracy (``slaterWeightMin``, ``de2_min``,
     ``e_pt2_tol``) everywhere -- ``calc_gs`` and both criteria. It used to run at
     ``sqrt(slaterWeightMin)`` and a per-determinant ``de2`` floor, which let a sector stall in the
     wrong symmetry class and reorder the sectors (NiO 10-bath: d6 0.17 high). The criteria's walk
     now *is* their own cached sector solves, so at one shift the walk and ``N +- 1`` never
     differ.
   * The refinement runs on a basis whose ``tau`` has been restored to the full value, matching
     ``calc_gs``; leaving the walk's ``tau/100`` in place made ``expand``'s
     ``de0_max = energy_cut(basis.tau)`` a hundred times narrower than the window ``calc_gs``
     expands into.
   * Nothing is frozen (review J1/P1a). Both criteria used to freeze every bath-less group,
     "exactly as get_spectra pins it" -- but get_spectra pins the *declared* X-ray core, and
     ``calc_selfenergy`` freezes nothing, so with a bath-less shell the double counting was
     measured on a different state than the self-energy then solved (E0 -8.97 frozen vs -9.34
     unfrozen on the review's toy). Freezing exists for a shell with no orbital energies (the
     X-ray core); a proper local Hamiltonian, which every DC input is, needs none.
   * The determinant budget is derived at the **full** memory-safety fraction, as ``calc_gs``
     derives its own. The peak and gap criteria used to halve it, to pay for two ``N +- 1`` bases
     believed to be live alongside the centre search's -- they are not, the three solves are
     sequential -- so the double counting was measured on half the space the self-energy run would
     use at that same ``dc``.

   * The PT2 threshold matches. Every DC sector energy now runs at
     :data:`groundstate.GS_DE2_MIN` (``1e-8``), the value :func:`groundstate.solve_ground_state`
     converges the production ground state to, rather than the ``1e-6`` that used to be a literal
     inside :func:`groundstate.calc_energy`. The looser value was wrong for the fixed-sector
     energies the peak and gap criteria **difference**, where it does not cancel and is then
     amplified by ``1 / |chi|`` -- a factor of 9 on the workload this was measured on -- and it
     is wrong for the walk too (see the walk bullet above).

   No convention is now knowingly unshared between the two paths.

Like :func:`selfenergy.calc_selfenergy` and :func:`groundstate.find_ground_state_sector`, both
derive their determinant budget from available per-rank memory
(:func:`impurityModel.ed.memory_estimate.resolve_cap_policy`) when
``BasisOptions.truncation_threshold`` is left at ``None``, and honor
``BasisOptions.chain_restrict`` through the same
:func:`impurityModel.ed.basis_restrictions.build_weighted_restrictions` the other ED drivers use
-- the double counting is otherwise found on a different variational space than the solve that
will use it.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.average import energy_cut
from impurityModel.ed.basis_restrictions import build_weighted_restrictions
from impurityModel.ed.dc_search import (
    DoubleCountingUnreachable,
    resolve_cap_at_max,
)
from impurityModel.ed.density_matrices import occupations as state_occupations
from impurityModel.ed.gs_connectivity import solver_basis_connectivity
from impurityModel.ed.lie_algebra import extract_tensors, tensors_to_operator
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.memory_estimate import CapPolicy, resolve_cap_policy
from impurityModel.ed.observables import shell_index_of_orbitals
from impurityModel.ed.sector_ensemble import resolve_mixed_valence
from impurityModel.ed.sector_solver import SectorPlanStore
from impurityModel.ed.solver_basis import _per_group_occupation, prepare_solver_basis
from impurityModel.ed.truncation_flags import cap_ever_hit
from impurityModel.ed.utils import matrix_print

#: A bath-less shell the DC shift skips must lie at least this far (in the units of ``h0``) from the Fermi level.
#: The shift acts on the hybridising shells (:attr:`solver_basis.SolverBasis.active_impurity_indices`), the rule
#: the fitter and the RSPt interface use; RSPt sends no core/valence flag. A skipped shell with a level inside
#: this window is a valence shell the hybridisation cut dropped, not a core: the search raises on every rank,
#: before any collective, and the way out is to declare its role (not yet available) or give it a bath.
DC_INACTIVE_SHELL_EF_WINDOW = 1.0


def _check_active_shells(sb, rank, verbosity, comm=None):
    """Say which shells the DC shift acts on and refuse an excluded shell that is not a core (MS T4.7).

    Single shell (``sb.active_impurity_indices is None``): nothing to decide. The decision reads the
    replicated ``h0``, and rank 0's verdict is broadcast on ``comm`` (a boundary level on heterogeneous
    nodes must not split the ranks), so every rank raises or none does, before any other collective.
    """
    if sb.active_impurity_indices is None:
        return
    active = set(sb.active_impurity_indices)
    inactive = sorted(o for o in sb.impurity_shell_index if o not in active)
    names = {a: f"l={sb.shell_ls[a]}" for a in sb.shell_has_bath}
    hybridising = [names[a] for a, has in sb.shell_has_bath.items() if has]
    excluded = [names[a] for a, has in sb.shell_has_bath.items() if not has]
    refusal = None
    if not active:
        refusal = (
            "the DC shift has no shell to act on: no shell of the impurity has bath orbitals, so nothing the "
            "criterion measures responds to it. Give the correlated shell a bath."
        )
    elif inactive:
        block = extract_tensors(sb.h0_solve, n_orb=sb.n_spin_orbitals, two_body=False)[0][np.ix_(inactive, inactive)]
        closest = float(np.min(np.abs(np.linalg.eigvalsh(block))))
        if closest < DC_INACTIVE_SHELL_EF_WINDOW:
            refusal = (
                f"the DC shift skips the shell(s) {', '.join(excluded)} (no bath orbitals), but a level of it lies "
                f"{closest:.3g} from the Fermi level (< {DC_INACTIVE_SHELL_EF_WINDOW}): that is a valence shell the "
                "hybridization cut dropped, not a core. Give it bath orbitals, or declare its role explicitly "
                "(not yet supported)."
            )
    if comm is not None and comm.size > 1:
        refusal = comm.bcast(refusal, root=0)
    if refusal is not None:
        raise ValueError(refusal)
    if verbosity > 0 and rank == 0:
        print(
            f"DC shift acts on shell(s) {', '.join(hybridising)} (hybridizing); excluded: "
            f"{', '.join(excluded) or 'none'} (no bath)",
            flush=True,
        )


def _active_nominal(sb):
    """Nominal electron count of the groups the shift acts on (the whole impurity on a single shell)."""
    if sb.active_impurity_indices is None:
        return sum(sb.nominal_occ.values())
    active = set(sb.active_impurity_indices)
    return sum(n for g, n in sb.nominal_occ.items() if sb.impurity_orbitals[g][0][0] in active)


def _active_shift(model, sb, n_imp):
    """The matrices the DC shift ``mu`` multiplies, and the orbitals whose occupation it moves.

    Returns ``(shift_input, shift_solver, active_indices)``.

    Single shell (``sb.active_impurity_indices is None``, i.e. no declared ``shell_ls``): exactly
    ``np.identity(n_imp)`` twice and ``None``, the objects every search used before multi-shell, never
    a product ``u^dagger I u`` (which differs from the identity at rounding level).

    Multi-shell: the projector onto the *active* shells, those that hybridise (``sb.shell_has_bath``,
    the rule the fitter and the interface use). A deep bath-less shell has an occupation its energy
    pins, so a shift on it moves nothing the criterion measures. ``shift_input`` is diagonal in the
    caller's input basis (the DC matrix lives there), ``shift_solver`` in the solver basis (``sb.h`` lives
    there). The solver works in the caller's basis, so the two are the same matrix.
    """
    if sb.active_impurity_indices is None:
        identity = np.identity(n_imp)
        return identity, identity, None
    shell_of_input = shell_index_of_orbitals(model.rot_to_spherical, model.shell_ls)
    shift_input = np.diag([1.0 if sb.shell_has_bath[int(a)] else 0.0 for a in shell_of_input])
    position = {orb: k for k, orb in enumerate(sorted(model.impurity_indices))}
    shift_solver = np.zeros((n_imp, n_imp))
    for orb in sb.active_impurity_indices:
        shift_solver[position[orb], position[orb]] = 1.0
    return shift_input, shift_solver, list(sb.active_impurity_indices)


def _ctx_spec(ctx):
    # `mixed_valence` is passed explicitly to `solve_charge_sector_ensemble`, never through this spec.
    """What decides which state a DC context's solves find, as a :class:`~groundstate.GroundStateSpec`.

    Built at each solve (like :func:`_ctx_cap_policy`, which it carries) because the cap ladder moves
    ``ctx.truncation_threshold``. The weighted restrictions are the context's built list, passed beside it.
    """
    from impurityModel.ed.gs_calc import GroundStateSpec

    return GroundStateSpec(
        tau=ctx.tau,
        chain_restrict=ctx.chain_restrict,
        dense_cutoff=ctx.dense_cutoff,
        e_pt2_tol=ctx.e_pt2_tol,
        de2_min=ctx.de2_min,
        truncation_threshold=_ctx_cap_policy(ctx),
    )


def _ctx_cap_policy(ctx):
    """The :class:`CapPolicy` a DC context's sector solves run under.

    ``ctx.truncation_threshold`` is a plain number the search itself may move (the cap ladder),
    so the policy is built at each solve rather than stored: memory may hold the basis below it
    unless the caller set a finite cap, i.e. exactly when ``ctx.cap_from_memory`` or the cap is
    unlimited.
    """
    cap = ctx.truncation_threshold
    unlimited = cap is None or not cap < np.inf
    return CapPolicy(gs=cap, gf=cap, from_memory=bool(ctx.cap_from_memory or unlimited))


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


def _any_cap_bound(ctx):
    """Did the determinant cap bind anything solved so far? ``None`` when it cannot be told.

    Both criterion contexts answer this, from the two different shapes their solves take:

    * :class:`_SectorContext` (gap, peak) reads ``cap_bound`` off each cached
      :class:`_SectorSolution` -- one per ``(mu, n_trial)`` charge sector.
    * :class:`_OccupationContext` reads ``cap_bound_at``, one entry per evaluated ``mu``, which
      :func:`_evaluate_occupation_and_energy_at_mu` fills from the ground-state determination's
      own truncation report.

    ``None`` is not "no": it is "nothing has been solved yet, so nothing can be said". The cap
    strategy treats it exactly as ``True`` (:func:`dc_search.resolve_cap_at_max`), which costs one
    extra rung rather than reporting a convergence the run never established.

    Both readings are already rank-replicated where they are stored -- ``_SectorSolution`` is
    broadcast whole, ``cap_bound_at`` is broadcast on write -- so this is a pure local reduction
    over replicated facts and adds no collective of its own.
    """
    verdicts = []
    sector_at = getattr(ctx, "sector_at", None)
    if sector_at:
        verdicts += [solution.cap_bound for solution in sector_at.values() if solution is not None]
    cap_bound_at = getattr(ctx, "cap_bound_at", None)
    if cap_bound_at:
        verdicts += list(cap_bound_at.values())
    if not verdicts:
        return None
    return any(verdicts)


def _any_memory_bound(ctx):
    """Was any sector behind this answer stopped by the **memory guard**? ``None`` when untold.

    The same two shapes :func:`_any_cap_bound` reads, but the stronger field: ``cap_bound`` is
    true whenever the fixed-budget branch engaged at all, which a perfectly healthy capped run
    does on purpose. ``memory_bound`` is true only when the rank ran out of room, so the basis is
    not a budget that was chosen but whatever the expansion had reached when it was stopped.

    The union over the search, deliberately, not just the converged ``mu``: every evaluation
    steers the bisection, so one memory-bound energy anywhere moves the bracket and therefore the
    answer. There is no such thing as a discarded evaluation in a bisection.
    """
    verdicts = []
    sector_at = getattr(ctx, "sector_at", None)
    if sector_at:
        verdicts += [sol.memory_bound for sol in sector_at.values() if sol is not None]
    memory_bound_at = getattr(ctx, "memory_bound_at", None)
    if memory_bound_at:
        verdicts += list(memory_bound_at.values())
    if not verdicts:
        return None
    return any(verdicts)


def _reject_if_memory_bound(ctx, dc_rec, *, rank, allow_memory_bound=False):
    """Refuse to hand back a double counting whose sectors ran out of memory.

    The SrMnO3 cubic gap-DC run reported ``status = converged`` and shipped
    ``dc_level = 0.580041`` into the DMFT loop from ground states of 10 to 252 determinants
    (``doc/plans/dc_smo_memory.md``, round 9). Every signal needed to catch it was already in the
    record -- ``dc_cap_bound = yes``, per-sector truncation reports -- and none was wired to the
    verdict.

    Why this and not ``dc_cap_drift`` against ``tol``: drift is the *cap-calibration ladder's*
    spread, i.e. how far the controlled quantity moved between the rungs that certified the cap.
    It is silent in exactly this failure -- a run pinned at its seed basis returns the same frozen
    answer on every rung, so the rungs agree and drift goes to zero -- and it is not populated at
    all when the cap is resolved at the ceiling. It stays in the record as
    context; it is not the gate.

    Why not the raw ``discarded_de2_mass``: ``_admit_top`` computes it as an unnormalized
    ``sum(scores[discarded])``, so it grows with how *many* candidates were dropped rather than
    how badly. A healthy run at a large cap discarding millions of marginal candidates would trip
    it. It bounds the error from above: below tolerance proves the answer is fine, above it
    proves nothing.

    ``allow_memory_bound`` (the criteria's argument; ``allow_memory_bound`` on the RSPt double-counting line)
    downgrades the refusal to a warning.
    """
    bound = _any_memory_bound(ctx)
    dc_rec["dc_memory_bound"] = "unknown" if bound is None else ("yes" if bound else "no")
    if not bound:
        return
    message = (
        "the double-counting search ran out of memory: at least one charge-sector ground state "
        "was stopped by the memory guard rather than by convergence, so the energies the search "
        "bisected on are those of whatever basis each expansion had reached, not of a basis "
        "anyone chose. Raise the per-rank memory (fewer ranks per node, e.g. a larger '-c'), "
        "lower the determinant cap so the budget is spent deliberately, or set "
        "allow_memory_bound to accept the answer anyway. The per-sector 'GS basis cap hit' "
        "warnings above say how much PT2 importance each retained subspace left behind."
    )
    if allow_memory_bound:
        if rank == 0:
            print(f"WARNING: {message}", flush=True)
        return
    raise DoubleCountingUnreachable(message)


def _calibrate_cap(ctx, evaluate_at_guess, clear_caches, tol, dc_rec, *, verbose, rank):
    """Run :func:`dc_search.resolve_cap_at_max` against ``ctx``, and record the
    result. Shared by :func:`fixed_gap_dc` and :func:`fixed_occupation_dc`, which differ only in
    *what* they evaluate at ``mu = 0`` and in what a cache-reuse guard needs to clear -- the
    ladder call itself, the ``dc_cap``/``dc_cap_drift``/``dc_cap_parity`` bookkeeping, and the
    ``cached_cap`` tracking a cache-reuse guard verifies against were duplicated near-verbatim
    between the two before this (review of ``6176c4f``, finding 4). Note what this does **not**
    cover: the ``mu_tol_effective`` truthiness bug that same review found (finding 1/2) was in a
    *different* duplicated block, ~150 lines further down in each function, where each criterion
    decides whether its own zero-slope case is a measured answer or a missing one -- unrelated to
    the cap-calibration bookkeeping this function consolidates. See :func:`_mu_tol_effective` for
    that duplication instead.

    Only called when ``ctx.cap_from_memory``; an explicit ``truncation_threshold`` is the caller's
    instruction, never a candidate for recalibration -- checked by the caller, not here, so that a
    caller skipping this entirely never even imports the closures it would need.

    Parameters
    ----------
    evaluate_at_guess : callable
        ``() -> float or None``, the criterion's own controlled quantity at ``mu = 0`` under
        whatever ``ctx.truncation_threshold`` is current when it is called -- this function sets
        that before every rung, the callable only has to read it (indirectly, through ``ctx`` and
        whatever local caches it closes over).
    clear_caches : callable
        ``() -> None``. Every cache the criterion keyed by ``mu`` (not by cap), cleared before
        every rung -- a hit from the previous rung would otherwise answer with another cap's
        energy. Called by this function on every rung; the caller applies it once more afterward
        only if it wants an unconditional clear (:func:`fixed_occupation_dc`) rather than the
        verified keep-or-clear :func:`fixed_gap_dc` applies via this function's return value.

    Returns
    -------
    (cap, cached_cap) : tuple
        ``cap`` is the accepted cap (already applied to ``ctx.truncation_threshold``).
        ``cached_cap`` is the cap the caches described when the ladder last called
        ``evaluate_at_guess`` -- compare it to ``cap`` before deciding whether they may be kept
        (``cached_cap != cap`` means they must be cleared; verified, not assumed, per 91109b8's
        history of a bug of exactly the opposite shape).
    """
    cached_cap = None

    def quantity(cap):
        nonlocal cached_cap
        ctx.truncation_threshold = cap
        clear_caches()
        cached_cap = cap
        return evaluate_at_guess()

    # The cap is chosen by running at the ceiling and asking the expansion whether the cap bound it
    # (`dc_search.resolve_cap_at_max` for why an unbound expansion needs no ladder).
    cap, cap_drift, _rungs, cap_status = resolve_cap_at_max(
        quantity,
        tol,
        memory_cap=ctx.memory_cap,
        bound_probe=lambda: _any_cap_bound(ctx),
        verbose=verbose,
        rank=rank,
        comm=MPI.COMM_WORLD,
    )
    ctx.truncation_threshold = cap
    if ctx.cap_ledger is not None:
        ctx.cap_ledger.resolved = cap
    dc_rec["dc_cap"], dc_rec["dc_cap_drift"] = cap, cap_drift
    dc_rec["dc_cap_parity"] = ctx.memory_cap
    # Which exit produced `cap` (unbound / cap_bound / memory_retreat). Taken from the search rather than
    # re-derived from `cap_drift` against `tol` here, so the record can tell a converged answer from a
    # truncation-limited one.
    dc_rec["dc_cap_status"] = cap_status
    # Recorded next to the cap it qualifies. "no" is the strong statement -- no sector was stopped
    # by the cap, so no larger cap can move this answer -- and it is the one a reader acts on.
    bound = _any_cap_bound(ctx)
    dc_rec["dc_cap_bound"] = "unknown" if bound is None else ("yes" if bound else "no")
    return cap, cached_cap


def _mu_tol_effective(tol, per_mu):
    """``|tol / per_mu|``, the search-tolerance error bar in ``mu`` -- shared by
    :func:`fixed_gap_dc` (``per_mu = 0.5 * delta_sum``) and :func:`fixed_occupation_dc`
    (``per_mu = occ_chi``), which previously computed this independently and picked up the same
    truthiness bug in both places (review of ``6176c4f``, finding 1/2).

    ``per_mu == 0.0`` is a *measured* answer -- a real charge-transfer level crossing
    (``delta_minus``'s own docstring: it can go negative, so the sum can land on exactly zero) or
    a genuine occupation plateau -- not a missing one, so this returns ``float("inf")`` rather
    than raising ``ZeroDivisionError`` or the caller silently skipping the write. Callers still
    check ``per_mu is not None`` themselves before calling this: that case (no slope measured at
    all) has nothing to report, which ``inf`` would misstate as "measured but undetermined".

    ``dc_record``'s own annotation of ``mu_tol_effective`` recomputes ``per_mu`` independently
    from the record's ``delta_sum``/``chi`` fields (rather than reading it from here) and prints
    "not a meaningful bound" beside the ``inf`` -- a deliberate second copy of the same zero-slope
    test, in the display layer rather than the write layer, so that a record built by hand (as
    ``dc_record``'s own tests do) is still annotated correctly without going through this
    function at all.
    """
    return abs(tol / per_mu) if per_mu else float("inf")


@dataclass(frozen=True)
class _SectorSolution:
    """What one charge-sector solve yields, once the eigenvectors have been reduced to numbers.

    Immutable and picklable, because it is broadcast: every field crosses
    :meth:`_SectorContext.sector_solve`'s boundary as a rank-replicated fact.

    ``energy`` is the lowest eigenvalue of the sector and ``occupation`` the impurity occupation
    ``Tr rho_imp`` *of the one state that attains it*: the ``T = 0`` pairing Hellmann-Feynman
    relates ``d(energy)/dmu`` to. A sector is solved for its lowest state alone, so there is no
    thermal manifold to average over and no spread to report.
    """

    energy: float
    occupation: float
    #: Did ``truncation_threshold`` bind this sector's CIPSI expansion? ``True`` when the
    #: fixed-budget branch engaged (``CIPSISolver.truncation_report is not None``), i.e. the
    #: expansion was stopped by the cap rather than by running out of candidates above
    #: ``de2_min``. This is the exact, per-sector, single-run test for "is the cap the limiting
    #: factor here", and it is what makes a cap ladder unnecessary in the common case: a sector
    #: that did **not** bind has already found the basis any larger cap would give it, so its
    #: energy cannot move by raising the cap. Measured on SrMnO3: the N+1 sector self-limits at
    #: 88,164 determinants and returns a bit-identical energy at caps 128k, 256k and 512k, while
    #: N-1 binds at every one of them. Note what it does *not* say: an unbound sector is
    #: converged **in the cap**, not exact -- whatever error is left is the ``de2_min`` /
    #: ``slaterWeightMin`` truncation, a different knob.
    cap_bound: bool = False
    #: Did the **memory guard** -- not the caller's cap -- stop this sector's expansion?
    #: ``truncation_report["memory_bound"]``. Strictly stronger than ``cap_bound``, and the
    #: distinction is the whole point: a cap that binds is a deliberate budget being spent, while
    #: a memory-bound stop means the rank ran out of room and the basis is whatever it had
    #: reached. On the SrMnO3 cubic gap-DC run that was the seed basis -- 10 to 252 determinants
    #: -- and the search still reported ``status = converged``.
    memory_bound: bool = False
    #: The expansion's residual PT2 energy, the error bar of ``energy``, and whether it was certified converged
    #: (``CIPSISolver.convergence_report``); ``None`` where the solve does not know. The walk's margin to its
    #: neighbours is measured against these (review P11).
    residual_pt2: float | None = None
    certified: bool | None = None


@dataclass
class _SectorContext:
    """The mu-independent setup a *sector-energy* criterion needs.

    Shared by :func:`fixed_peak_dc` and :func:`fixed_gap_dc`, which differ only in *which*
    combination of ``E[N-1]``, ``E[N]``, ``E[N+1]`` they drive onto their target -- one
    difference for the peak, the symmetric one for the gap centre. Everything up to and
    including "solve one charge sector at this shift" is identical, and it is precisely the part
    where the DC/ground-state parity conventions live (the walk's ``tau / 100`` and
    ``sqrt(slaterWeightMin)``, the weighted-restriction
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
    shift_input: np.ndarray
    shift_solver: np.ndarray
    #: Orbitals whose occupation ``mu`` moves (the active shells); ``None`` = every impurity orbital.
    active_indices: Optional[list]
    impurity_orbitals: dict
    bath_states: dict
    N0: dict
    h_solver_matrix: np.ndarray
    max_occ_total: int
    weighted_restrictions: object
    truncation_threshold: object
    tau: float
    slater_weight_min: float
    chain_restrict: bool
    dense_cutoff: int
    bandwidth: float
    rank: int
    verbose: bool
    #: The cap the memory budget would have allowed, and whether that is where
    #: :attr:`truncation_threshold` came from. Only a memory-derived cap is a candidate for
    #: recalibration (:func:`dc_search.resolve_cap_at_max`, wired into
    #: :func:`fixed_gap_dc`); a cap the caller passed explicitly is an instruction, not a default
    #: to second-guess.
    memory_cap: object = None
    cap_from_memory: bool = False
    #: The calculation's :class:`~impurityModel.ed.memory_estimate.CapLedger`; the cap ladder records the cap it
    #: settled on, which is what a later call may not exceed (P10).
    cap_ledger: object = None
    n_center_at: dict | None = None
    #: ``(mu, n_trial) -> _SectorSolution | None``. Filled by :meth:`sector_solve`, which never
    #: solves the same sector at the same shift twice. Two floats per entry, not eigenvectors --
    #: see that method for why retaining ``psis`` does not scale.
    sector_at: dict | None = None
    #: Epstein-Nesbet PT2 admission floor for :meth:`sector_solve`'s CIPSI expansions. Resolved by
    #: :func:`_prepare_sector_context` from, in order, the criterion's own ``de2_min`` argument
    #: (the RSPt double-counting line or the TOML criterion table), ``basis.de2_min``, and finally
    #: :data:`groundstate.GS_DE2_MIN` (0: no floor) -- so by default the
    #: search runs at the floor the self-energy run's ground state does. It is not the
    #: accuracy control (``e_pt2_tol`` is); a floor only loosens a solve.
    #: ``None`` resolves in :meth:`__post_init__`; `groundstate` is imported lazily throughout
    #: this module (circular import), so the fallback cannot be a module-level default here.
    de2_min: float | None = None
    #: Residual Epstein-Nesbet PT2 energy the sector solves converge to -- the accuracy control,
    #: where ``de2_min`` is only an optional floor. Resolved from ``basis.e_pt2_tol``, else
    #: :data:`groundstate.GS_E_PT2_TOL`, the tolerance of the production ground state.
    e_pt2_tol: float | None = None
    #: Each sector's certification (:class:`sector_solver.SectorPlanStore`): certified once per
    #: search, then only the classes that can still be lowest are expanded at a new shift.
    plan_store: object = None
    #: The (dn, up) pairing of the solver basis (:attr:`solver_basis.SolverBasis.spin_layout`): the
    #: ``-mu`` shift commutes with ``S_z`` and ``S_+``, so one layout serves every shift.
    spin_layout: object = None
    #: A bath-less shell's seed window (:attr:`solver_basis.SolverBasis.seed_window`), ``None`` on a single shell.
    #: Passed as ``frozen_occupations`` to every solve, so the deep shell is seeded as a window, never spanned.
    seed_window: object = None
    #: ``mu -> {window group: filling}``: where the walk at ``mu`` holds the window (Hartree-Fock's filling, which
    #: need not be the nominal one). Read by :meth:`trial_occupation`; replicated, like the walk's trials.
    window_filling_at: dict | None = None

    def __post_init__(self):
        if self.window_filling_at is None:
            self.window_filling_at = {}
        if self.de2_min is None:
            self.de2_min = _resolve_dc_de2_min()
        if self.e_pt2_tol is None:
            self.e_pt2_tol = _resolve_dc_e_pt2_tol(None, None)
        if self.n_center_at is None:
            self.n_center_at = {}
        if self.sector_at is None:
            self.sector_at = {}
        if self.plan_store is None:
            self.plan_store = SectorPlanStore()

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
        return self.h_op_i - tensors_to_operator(np.asarray(mu * self.shift_solver, dtype=complex))

    def dc(self, mu):
        """The double-counting matrix at ``mu``: ``dc_guess + mu * shift_input``."""
        return self.dc_guess + mu * self.shift_input

    def centre_sector(self, h_op, mu):
        """The centre charge state at ``mu``, from the same walk ``calc_selfenergy`` will run.

        Only the walk: the thermal solve that ``solve_ground_state`` adds does not change the
        winning occupation, and the sector is all a sector-energy criterion needs before solving
        ``N +- 1``. Its trials are :meth:`sector_solve`, so they are cached with, and are the same
        solves as, the criterion's own ``N +- 1``.
        """
        from impurityModel.ed.groundstate import find_ground_state_sector

        search = find_ground_state_sector(
            h_op,
            self.impurity_orbitals,
            self.bath_states,
            self.N0,
            slaterWeightMin=self.slater_weight_min,
            comm=MPI.COMM_WORLD,
            verbose=self.verbose,
            weighted_restrictions=self.weighted_restrictions,
            **_ctx_spec(self).solve_kwargs(
                "chain_restrict", "dense_cutoff", "truncation_threshold", "de2_min", "e_pt2_tol", "use_hf_seed"
            ),
            # The walk's trials are this search's own cached sector solves: the N +- 1 solves at this
            # shift are the same solves, at the same accuracy, and are never repeated.
            sector_energy=lambda trial: self._walk_energy(h_op, mu, trial),
            spin_layout=self.spin_layout,
            frozen_occupations=self.seed_window,
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
        # is currently rank-identical by inheritance -- find_ground_state_sector broadcasts every
        # energy its sector election compares -- but nothing here enforces that, and an asymmetric
        # raise strands the ranks that did not raise.
        n_center = MPI.COMM_WORLD.bcast(sum(search.occupation.values()), root=0)
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
        from impurityModel.ed.eigensolvers import Lowest
        from impurityModel.ed.groundstate import solve_sector

        if not 0 <= n_trial <= self.max_occ_total:
            return None
        if (mu, n_trial) in self.sector_at:
            return self.sector_at[(mu, n_trial)]
        occ_trial = self.trial_occupation(n_trial, mu)
        if occ_trial is None:
            self.sector_at[(mu, n_trial)] = None
            return None
        es, psis, sector_basis, sector_solver = solve_sector(
            h_op,
            self.impurity_orbitals,
            self.bath_states,
            occ_trial,
            manifold=Lowest(),
            comm=MPI.COMM_WORLD,
            verbose=self.verbose,
            slaterWeightMin=self.slater_weight_min,
            weighted_restrictions=self.weighted_restrictions,
            # The accuracy the answer is quoted at: this energy is about to be *differenced*
            # against another sector's, and the residual truncation error does not cancel in a
            # difference. It is then amplified by 1 / |chi| on its way into `dc` -- a factor of 9 on
            # the workload this was measured on.
            # Resolved once in `_prepare_sector_context` (RSPt line > basis.de2_min > GS_DE2_MIN,
            # RSPt line > basis.e_pt2_tol > GS_E_PT2_TOL), so every sector of a search shares one convergence
            # criterion and the record can report it.
            plan_store=self.plan_store,
            shift=mu,
            **_ctx_spec(self).solve_kwargs(
                "tau", "chain_restrict", "dense_cutoff", "truncation_threshold", "de2_min", "e_pt2_tol"
            ),
            spin_layout=self.spin_layout,
            frozen_occupations=self.seed_window,
        )
        # Broadcast the verdict *before* branching on it, not the answer afterwards. `es` and
        # `psis` come back rank-local from `solve_sector` (its docstring says so), and the next
        # line runs `build_density_matrices`, which ends in an `Allreduce`. A rank-local decision
        # in front of a collective is the deadlock this module has already shipped once.
        #
        # `solve_sector` returns `([inf], [])` for an empty basis -- a sector the restrictions
        # admit no determinants for -- undefined for the same reason as the bounds check above.
        # `psis` is the block `get_eigenvectors` returns. `len()` on it is the rank-local ROW
        # count, so `bool(len(psis))` would read "this rank owns no determinants" as "the sector
        # is unusable" -- and since only rank 0's value is broadcast, an empty rank 0 (which
        # appears from -n 3 up) would silently declare every sector unusable and let the DC
        # search proceed on a wrong answer. `.width` is the manifold size and is rank-invariant.
        n_kept = psis.width if isinstance(psis, ManyBodyState) else len(psis)
        usable = MPI.COMM_WORLD.bcast(bool(n_kept) and bool(np.all(np.isfinite(es))), root=0)
        if not usable:
            self.sector_at[(mu, n_trial)] = None
            return None
        impurity_indices = self.active_indices
        if impurity_indices is None:
            impurity_indices = [orb for blocks in self.impurity_orbitals.values() for block in blocks for orb in block]
        # Only the diagonal of the impurity density matrix is read (its trace, thermally averaged),
        # and <n_i> needs no operator applications and no redistribution of c_i|psi>.
        # <n_i> needs no operator application: bit counts over the one state's support. `.width` is
        # replicated (`build_state` takes it from the dense array), so every rank enters the reduction
        # with the same buffer shape.
        occupations = state_occupations(sector_basis, psis, impurity_indices).sum(axis=1)
        solution = _SectorSolution(
            energy=float(es[0]),
            occupation=float(occupations[0]),
            # `_solve_sector_core` hangs its solver's `truncation_report` on the basis it returns
            # (`occupation_search_truncation`) precisely so a caller that never sees the solver can
            # still tell a cap-limited expansion from a converged one. Read as a bool here: the
            # report's contents describe *how* the cap bound, which the cap strategy does not need
            # -- only whether it did. Rank-local until the broadcast below, like everything else
            # in this constructor.
            cap_bound=cap_ever_hit(sector_solver.truncation_report),
            memory_bound=bool((sector_solver.truncation_report or {}).get("memory_bound", False)),
            residual_pt2=_float_or_none((getattr(sector_solver, "convergence_report", None) or {}).get("residual_pt2")),
            certified=_bool_or_none((getattr(sector_solver, "convergence_report", None) or {}).get("converged")),
        )
        solution = MPI.COMM_WORLD.bcast(solution, root=0)
        self.sector_at[(mu, n_trial)] = solution
        return solution

    def trial_occupation(self, n_trial, mu=None):
        """The per-group occupation of the walk's sector ``n_trial`` (energetic filling over every impurity
        orbital), or ``None`` if it has none. Nothing is frozen on the DC path (review J1), so the split is a
        basis-generation parameter only: the sector is its total. A bath-less shell beside a correlated one is
        held out of the walk and seeded as a window (:attr:`seed_window`), not frozen.

        The window's groups keep the filling the walk at ``mu`` holds them at (:attr:`window_filling_at`, the
        Hartree-Fock seed's; the nominal one before any walk ran at ``mu``), and only the other groups share the
        rest of ``n_trial``, as on the self-energy path. Filling every orbital energetically put a deep shell's
        electrons wherever the unshifted levels said: a d-shell double counting of 65 left the p window empty, so
        the DC solved sectors the self-energy never seeds (review H-MS4). ``None`` when the rest does not fit the
        other groups.
        """
        n_trial = int(n_trial)
        if not 0 <= n_trial <= sum(len(block) for blocks in self.impurity_orbitals.values() for block in blocks):
            return None
        window = set(self.seed_window or ())
        if not window:
            return _per_group_occupation(n_trial, self.impurity_orbitals, self.h_solver_matrix)
        pinned = self.window_filling_at.get(mu) or {g: int(self.N0[g]) for g in window}
        free = {g: blocks for g, blocks in self.impurity_orbitals.items() if g not in window}
        rest = n_trial - sum(pinned.values())
        if not 0 <= rest <= sum(len(block) for blocks in free.values() for block in blocks):
            return None
        split = {**pinned, **_per_group_occupation(rest, free, self.h_solver_matrix)}
        return {g: split[g] for g in self.impurity_orbitals}

    def _walk_energy(self, h_op, mu, trial):
        """The occupation walk's view of its trial ``trial``: a :class:`groundstate.SectorEnergy` built from
        :meth:`sector_solve`'s, ``inf`` where undefined. The truncation report stays ``None`` (the solution keeps
        only the ``cap_bound`` flag), the residual and the verdict are the cached solve's own."""
        from impurityModel.ed.groundstate import SectorEnergy

        if self.seed_window:
            # The walk holds the window where Hartree-Fock put it (review of H-MS4); the sector is solved there.
            self.window_filling_at[mu] = {g: int(trial[g]) for g in self.seed_window}
        solution = self.sector_solve(h_op, mu, sum(trial.values()))
        if solution is None:
            return SectorEnergy(np.inf, None)
        return SectorEnergy(solution.energy, None, solution.residual_pt2, solution.certified, solution.cap_bound)

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
        if solution is None:
            return None
        return solution.occupation


def _float_or_none(x):
    return None if x is None else float(x)


def _bool_or_none(x):
    return None if x is None else bool(x)


def _resolve_dc_de2_min(de2_min=None, basis_de2_min=None):
    """The PT2 admission floor a double-counting sector solve should use.

    Precedence: the criterion's own argument (which the RSPt double-counting line and the TOML
    criterion table supply) beats the solver's own ``BasisOptions.de2_min`` (the floor the
    production ground state runs at), which beats :data:`groundstate.GS_DE2_MIN`. One precedence,
    one function (P5); ``fixed_occupation_dc`` passes no argument, so it finds the state the
    self-energy finds.
    """
    from impurityModel.ed.groundstate import GS_DE2_MIN

    if de2_min is not None:
        return float(de2_min)
    return GS_DE2_MIN if basis_de2_min is None else float(basis_de2_min)


def _resolve_dc_e_pt2_tol(e_pt2_tol, basis_e_pt2_tol):
    """The residual PT2 energy a double-counting ground-state solve converges to.

    Precedence: the criterion's own argument (the RSPt double-counting line's ``e_pt2 X``), then
    the solver's own ``BasisOptions.e_pt2_tol`` (the solver line's ``e_pt2 X``) -- so by default
    the double counting is measured on a space converged as far as the self-energy run's ground
    state -- and finally :data:`groundstate.GS_E_PT2_TOL`.
    """
    from impurityModel.ed.groundstate import GS_E_PT2_TOL

    for value in (e_pt2_tol, basis_e_pt2_tol):
        if value is not None:
            return float(value)
    return GS_E_PT2_TOL


def _prepare_sector_context(
    model,
    basis,
    solver,
    *,
    comm=None,
    verbosity=0,
    memory_label,
    de2_min=None,
    e_pt2_tol=None,
    cap_ledger=None,
):
    """Build the setup :class:`_SectorContext` holds, from the grouped option objects.

    Verbatim extraction of the preamble :func:`fixed_peak_dc` used to open with, so that
    :func:`fixed_gap_dc` cannot acquire a second, subtly different copy of it.
    """
    n_imp = len(model.impurity_indices)
    dc_guess = extract_tensors(model.dc or {}, n_orb=n_imp, two_body=False)[0]
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    verbose = verbosity > 0

    # Peak/gap measure the winning sector alone under every mode (auto, on and off): the switch is resolved here,
    # so a malformed value raises on every rank before any collective, but it no longer reaches the basis layer.
    resolve_mixed_valence(basis.mixed_valence, driver_default="on")

    # Derive the same solver-basis layout calc_selfenergy uses. Both pass model.bath_states, so the
    # fit valence/conduction split decides here and there (root cause 3 of the original DC<->GS
    # mismatch was the self-energy path re-deriving it from the sign of the bath on-site energy).
    sb = prepare_solver_basis(
        model.h0,
        model.dc,
        model.u4,
        model.impurity_orbitals,
        basis.nominal_occ,
        rot_to_spherical=model.rot_to_spherical,
        verbosity=verbosity,
        rank=rank,
        comm=comm if comm is not None else MPI.COMM_WORLD,
        shell_ls=model.shell_ls,
        interaction_merges_blocks=basis.interaction_merges_blocks,
        bath_split=model.bath_states,
    )
    _check_active_shells(sb, rank, verbosity, comm if comm is not None else MPI.COMM_WORLD)
    impurity_orbitals = sb.impurity_orbitals
    bath_states = sb.bath_states
    shift_input, shift_solver, active_indices = _active_shift(model, sb, dc_guess.shape[0])

    max_occ_total = sum(len(block) for blocks in impurity_orbitals.values() for block in blocks)
    # One-body matrix in the *solver* basis (h0_solve), so the
    # energetic filling _per_group_occupation performs sees the same on-site energies that defined
    # the derived groups.
    h_solver_matrix = extract_tensors(sb.h0_solve, n_orb=sb.n_spin_orbitals, two_body=False)[0]

    truncation_threshold = basis.truncation_threshold
    # Provenance, not just value: only a cap that *defaulted* is a candidate for recalibration
    # against the search's tolerance (:func:`dc_search.resolve_cap_at_max`, wired into
    # :func:`fixed_gap_dc` below). A cap the caller passed explicitly is an instruction.
    #
    # Broadcast, though it is derived from a replicated input (``basis.truncation_threshold`` is
    # the same object on every rank): this flag gates the whole cap ladder -- a dozen collective
    # solves -- and CLAUDE.md's rule against gating a collective on rank-local state is categorical
    # for a reason. One bool against a CIPSI expansion is a cheap place to honour it.
    # Recalibration (the cap ladder) is only for a cap the caller left to memory; `unlimited` and
    # a set number are instructions. Broadcast, though derived from a replicated input.
    cap_from_memory = MPI.COMM_WORLD.bcast(CapPolicy.coerce(truncation_threshold).source == "auto", root=0)
    # The one resolver every driver uses: the same ground-state-only sizing, safety and
    # GS_NUM_WANTED as calc_selfenergy, so the dc is found at the determinant budget the
    # self-energy then solves with; it prints the `determinant cap:` line and starts a new
    # calculation's memory-warning count. Collective on COMM_WORLD, unconditional.
    policy, _ = resolve_cap_policy(
        truncation_threshold,
        model.n_spin_orbitals,
        comm=MPI.COMM_WORLD,
        verbose=verbose,
        label=memory_label,
        log="derived",
        connectivity=solver_basis_connectivity(sb, MPI.COMM_WORLD),
        ledger=cap_ledger,
    )
    truncation_threshold = policy.gs
    memory_cap = policy.gs

    # The spread of the one-body h0 eigenvalues: the scale a sector-energy difference can move
    # over, used to size the search range. Nothing is derived from a penalty value any more -- an
    # unreachable sector is reported as *undefined*, not as a large number (see sector_energy).
    h1_for_scale = extract_tensors(ManyBodyOperator(model.h0), n_orb=model.n_spin_orbitals, two_body=False)[0]

    return _SectorContext(
        h_op_i=sb.h,
        dc_guess=dc_guess,
        shift_input=shift_input,
        shift_solver=shift_solver,
        active_indices=active_indices,
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        N0=sb.nominal_occ,
        h_solver_matrix=h_solver_matrix,
        max_occ_total=max_occ_total,
        # Bath-only restriction (the reference "valence filled, conduction empty" occupation never
        # references N0), so the identical restriction list is valid for every sector -- the same
        # restrictions find_ground_state_sector applies to its own occupation-scan trials.
        weighted_restrictions=build_weighted_restrictions(
            bath_states,
            cost_budget=basis.cost_budget,
            h_op=sb.h,
            impurity_orbitals=impurity_orbitals,
            slater_weight_min=basis.slater_weight_min,
        ),
        truncation_threshold=truncation_threshold,
        memory_cap=memory_cap,
        cap_from_memory=cap_from_memory,
        cap_ledger=cap_ledger,
        spin_layout=sb.spin_layout,
        plan_store=SectorPlanStore(active_orbitals=sb.active_impurity_indices),
        seed_window=sb.seed_window,
        tau=basis.tau,
        slater_weight_min=basis.slater_weight_min,
        chain_restrict=basis.chain_restrict,
        dense_cutoff=solver.dense_cutoff,
        bandwidth=max(float(np.ptp(np.linalg.eigvalsh(h1_for_scale))), 1.0),
        rank=rank,
        verbose=verbose,
        de2_min=_resolve_dc_de2_min(de2_min, basis.de2_min),
        e_pt2_tol=_resolve_dc_e_pt2_tol(e_pt2_tol, basis.e_pt2_tol),
    )


def _warn_if_a_neighbour_sector_is_thermal(excess, tau, rank, criterion):
    """Warn when the nearest N+-1 sector lies inside the thermal window of the sector the criterion measures.

    Peak/gap measure the winning sector alone, under every ``mixed_valence`` mode. When a neighbour sits within
    ``energy_cut(tau)`` of it, ``calc_gs`` under ``auto`` forms an ensemble and the Sigma step then sees a different
    state than the one this criterion placed on the Fermi level. ``excess`` is that neighbour's energy above the
    measured sector (``None`` when it is unreachable: nothing to warn about).
    """
    if excess is not None and excess < float(energy_cut(tau)) and rank == 0:
        print(
            f"WARNING: {criterion} double counting measures one charge sector, but the nearest N+-1 sector lies "
            f"{excess:.4g} above it, inside the thermal window ({float(energy_cut(tau)):.4g}); the criterion is "
            "ill-posed here and the self-energy ensemble will differ from the state it measured.",
            flush=True,
        )
