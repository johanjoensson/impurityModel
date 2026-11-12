"""The peak criterion: :func:`fixed_peak_dc` pins a chosen spectral peak, ``E[N+1] - E[N]`` or ``E[N] - E[N-1]``.

Shared setup and conventions are in :mod:`~impurityModel.ed.dc_sector_context`.
"""

from mpi4py import MPI

from impurityModel.ed import config, dc_record, solver_trace
from impurityModel.ed.dc_reference import (
    _warn_if_not_fermi_referenced,
)
from impurityModel.ed.dc_search import (
    _dc_chi,
    _dc_search_trace,
    _solve_dc_shift,
    bracket_width_tol,
)
from impurityModel.ed.dc_sector_context import (
    _dump_dc_matrices,
    _prepare_sector_context,
    _reject_if_memory_bound,
    _warn_if_a_neighbour_sector_is_thermal,
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
    de2_min=None,
    e_pt2_tol=None,
    allow_memory_bound=False,
    cap_ledger=None,
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
    ``dc(mu) = dc_guess + mu * P`` (``P`` = the identity on one shell). The shift couples to the impurity
    occupation as :math:`-\mu \hat N_{imp}`, so the peak position responds
    monotonically, :math:`d(E_{upper} - E_{lower})/d\mu = -(\langle N
    \rangle_{upper} - \langle N \rangle_{lower}) \approx -1`, and ``mu`` is
    found by the shared secant/bisection search :func:`_solve_dc_shift`.

    At every trial ``mu`` the *center* sector ``N`` is not the input ``model.impurity_orbitals``
    occupation: it is found by :func:`groundstate.find_ground_state_sector`, the identical
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
    units. All three sectors honor ``BasisOptions.chain_restrict``
    identically -- the bath-only weighted restrictions (the observed excitation bounds are attached per solve) never
    reference the occupation, so the same restriction list applies regardless of which sector it is evaluated on.

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
        :func:`impurityModel.ed.memory_estimate.resolve_cap_policy`) at the same safety
        fraction ``calc_gs`` uses -- the three sector solves are sequential, so the live peak is
        the centre walk's own retained sector, which is the peak ``calc_gs`` has too; ``numpy.inf``
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
    de2_min : float, optional
        Per-determinant Epstein-Nesbet PT2 floor for the charge-sector CIPSI expansions. ``None``
        (the default) falls back to
        ``basis.de2_min``, then to :data:`groundstate.GS_DE2_MIN` (0: no floor). Supplied from the
        RSPt double-counting line as ``de2_min X``. The solves are *converged* to the
        ``e_pt2_tol`` residual PT2 energy (below); a floor above that loosens them, and the
        expansion then warns with the residual it left -- a bounded approximation, unlike
        lowering the determinant cap, which truncates the basis itself.
    e_pt2_tol : float, optional
        Residual PT2 energy the charge-sector solves converge to. ``None`` (the default) falls
        back to ``basis.e_pt2_tol``, then to
        :data:`groundstate.GS_E_PT2_TOL`. Supplied from the RSPt double-counting line as
        ``e_pt2 X``; loosen it (e.g. ``1e-5``) when the search, not its accuracy, is the cost.
    Returns
    -------
    dc : ndarray
        The double counting matrix, ``dc_guess + mu * P``.

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
    config.assert_uniform(MPI.COMM_WORLD)  # collective: a rank-divergent env knob is an error, not a deadlock
    with dc_record.recording(
        "peak", rank=comm.rank if comm is not None else MPI.COMM_WORLD.rank, report=report
    ) as dc_rec:
        ctx = _prepare_sector_context(
            model,
            basis,
            solver,
            comm=comm,
            verbosity=verbosity,
            memory_label="fixed-peak dc",
            de2_min=de2_min,
            e_pt2_tol=e_pt2_tol,
            cap_ledger=cap_ledger,
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
        dc_rec.certification = ctx.plan_store
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
        dc_rec["de2_min"] = ctx.de2_min
        dc_rec["e_pt2_tol"] = ctx.e_pt2_tol
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

        _warn_if_a_neighbour_sector_is_thermal(
            None if dc_rec.get("peak") is None else abs(dc_rec["peak"]), tau, rank, "fixed-peak"
        )
        _reject_if_memory_bound(ctx, dc_rec, rank=rank, allow_memory_bound=allow_memory_bound)
        return (dc, n_center) if return_sector else dc
