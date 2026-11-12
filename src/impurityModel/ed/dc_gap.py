"""The gap criterion: :func:`fixed_gap_dc` pins the gap midpoint ``(E[N+1] - E[N-1]) / 2`` (Karolak et al.).

Shared setup and conventions are in :mod:`~impurityModel.ed.dc_sector_context`.
"""

from mpi4py import MPI

from impurityModel.ed import config, dc_record, solver_trace
from impurityModel.ed.dc_reference import (
    _warn_if_not_fermi_referenced,
)
from impurityModel.ed.dc_search import (
    DoubleCountingUnreachable,
    _dc_chi,
    _dc_search_trace,
    _solve_dc_shift,
    bracket_width_tol,
)
from impurityModel.ed.dc_sector_context import (
    _calibrate_cap,
    _dump_dc_matrices,
    _mu_tol_effective,
    _prepare_sector_context,
    _reject_if_memory_bound,
    _warn_if_a_neighbour_sector_is_thermal,
)

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
#: is how much of the transferred electron lands on the impurity at one edge -- near 1 for an
#: impurity-like excitation, near 0 for a bath-like one, and **not** confined to ``[0, 1]``: only
#: the change summed over impurity *and* bath is one electron, so at a charge-transfer level
#: crossing (``d8`` to ``d10 L``) it is 2, and the removal partner can be negative.
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

    **What this can and cannot see.** ``delta_+-`` is a Hellmann-Feynman slope: how far an edge
    moves per unit ``mu``. That is exactly the right quantity for the *conditioning* of the search
    -- the acceptance band in ``mu`` is ``tol / |chi|`` and ``chi`` is built from these -- and it
    is a good proxy for orbital character, which is why it is worth warning on. It is **not** the
    Lehmann weight :math:`w_n = \\sum_d |\\langle n|c_d^\\dagger|0\\rangle|^2`, and the two
    decouple *exactly*: a state differing from :math:`c_d^\\dagger|0\\rangle` in total spin or
    irrep has ``w_n = 0`` identically while its ``<N_imp>`` differs from the ``N`` sector's by a
    full electron. In that case the criterion is centring a pole that does not exist in ``A_imp``,
    the search is well conditioned, ``delta`` is near 1, and **this warning stays silent**. A high
    ``delta`` is therefore necessary for spectral weight and not sufficient for it. Nothing in the
    production path measures the weight itself; ``test/support/dc_weights.py`` does, serially.
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
        f"WARNING: the {edge} edge this double counting was fixed against moves only {weakest:.3f} "
        f"of an electron onto the impurity ({name}; below {GAP_EDGE_IMPURITY_FLOOR} is flagged). "
        "The N +- 1 sector energies carry no impurity projection, so what was centred on the Fermi "
        "level is the CLUSTER charge gap, and this edge is largely a state of the bath. Two very "
        "different situations produce that. (1) A charge-transfer insulator: the valence-band top "
        "really is ligand, the cluster gap really is the p-to-d gap, and the answer is meaningful "
        "-- but it is not the gap of the impurity spectral function A_imp, which is what the "
        "fixed-gap prescription is usually quoted for, and the two can sit several eV apart. "
        "(2) A discretization artefact: an isolated fitted bath level inside the true gap, which "
        "does not move with the double-counting shift at all. Distinguish them with the opt-in "
        "weight diagnostic, `python -m impurityModel.test.support.dc_weights <archive> <cap> <mu>` "
        "(serial) -- a band shows many nearly degenerate states carrying weight, an artefact shows "
        "one isolated level carrying none -- and cross-check against fixed_occupation_dc, and "
        "against fixed_peak_dc if the OTHER edge is the responsive one. Note what this number is: "
        "a mu-response, not a spectral weight. A state orthogonal to c_d^dagger|0> by spin or "
        "irrep carries zero weight in A_imp while still moving fully with mu, and no value of "
        "delta detects that -- only the weight diagnostic does.",
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


def _refuse_an_ill_conditioned_gap(ctx, offset, energy_tol, centre_at, dc_rec, search_report):
    """Raise :class:`DoubleCountingUnreachable` when the gap centre barely responds to the shift.

    The centre moves by ``-(delta_+ + delta_-) / 2`` per unit ``mu``: the electrons the two edges
    move onto the impurity. When that lever is below ``DC_GAP_LEVER_MIN`` both edges are mostly
    bath, the centre is a property of the bath fit, and the shift that would place it is far
    outside what the solves at the guess say anything about -- the search then walks sectors
    for many evaluations without converging (22 on the NiO 10-bath archive, lever 0.02). Measured
    at the guess from the three sectors the tolerance sizing has already solved, so it costs
    nothing; a guess that already satisfies the criterion is never refused. The lever is recorded
    either way. Every value read is broadcast, so the raise is rank-identical.
    """
    centre = centre_at(0.0)
    lever = None
    if centre is not None:
        delta_plus, delta_minus = _measure_edge_character(ctx, 0.0, ctx.n_center_at[0.0])
        if delta_plus is not None and delta_minus is not None:
            lever = 0.5 * (delta_plus + delta_minus)
    dc_rec["lever_at_guess"] = lever
    lever_min = config.DC_GAP_LEVER_MIN.get()
    if lever is None or abs(centre - offset) <= energy_tol or lever >= lever_min:
        return
    search_report["status"] = "ill_conditioned"
    search_report["evaluations"] = 1
    if lever > 0.0:
        estimate = f"a shift of about {(centre - offset) / lever:+.2f} on a first-order estimate"
    else:
        estimate = "a shift in a direction the centre does not even move in"
    raise DoubleCountingUnreachable(
        f"The fixed-gap double counting is ill-conditioned on this model: the gap centre "
        f"(E[N+1] - E[N-1])/2 = {centre:.5f} at the guess (N = {ctx.n_center_at[0.0]}) moves only "
        f"{lever:.3f} eV per eV of shift (delta_+ = {delta_plus:.3f}, delta_- = {delta_minus:.3f}; "
        f"DC_GAP_LEVER_MIN = {lever_min}). Both gap edges are mostly states of the bath, so the "
        "centre is a property of the bath fit rather than of the impurity, and placing it at the "
        f"offset would take {estimate} -- far beyond what the solves at the guess describe. Use the occupation "
        "criterion (`occ <target>`), which needs only the impurity occupation to respond to the "
        "shift, or set DC_GAP_LEVER_MIN=0 to search anyway."
    )


def _size_gap_tolerance(width, mu_tol, bandwidth, rank=0):
    """The gap criterion's energy tolerance, and the one word that explains where it came from.

    Split out of :func:`fixed_gap_dc` so the three degenerate cases can be tested directly, without
    standing up a model and running a search to reach each of them.

    The ``abs()`` this replaced was the problem. A **negative** width means ``E[N]`` is not the
    lowest of the three sectors: the centre walk has settled on a sector that is not the lowest of
    its neighbours (the walk and the sector solves share their solves, so they cannot disagree on
    an energy). That is a real and reportable diagnostic; ``abs()`` turned it into a
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
                "of the three sectors, so the centre walk settled on a charge state that is not "
                "the lowest of its neighbours. The energy "
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
    de2_min=None,
    e_pt2_tol=None,
    allow_memory_bound=False,
    cap_ledger=None,
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

        **When one edge is unresponsive, this collapses into the peak criterion.** Since
        :math:`d(\text{centre})/d\mu = -(\delta_+ + \delta_-)/2`, an edge with
        :math:`\delta \approx 0` contributes nothing to the derivative, and root-finding the
        centre reduces to root-finding the *other* edge against a :math:`\mu`-independent
        reference. On NiO 5-bath (:math:`\delta_- = 0.032`) ``gap`` and :func:`fixed_peak_dc` are
        therefore solving very nearly the same equation, and the averaging buys only a factor 2 in
        the residual. That makes ``peak`` the informative cross-check there, alongside ``occ``:
        two criteria that agree because they are the same equation are not two pieces of evidence.
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
        If the gap centre cannot be brought to ``offset`` within the reachable range -- including
        the case where one of the two off-centre sectors does not exist (a full or empty
        impurity), where the criterion is *undefined* rather than merely far off.
    RuntimeError
        If the converged ``mu`` centres the gap on a non-nominal charge state and
        ``allow_charge_state_change`` is ``False``.
    """
    config.assert_uniform(MPI.COMM_WORLD)  # collective: a rank-divergent env knob is an error, not a deadlock
    with dc_record.recording(
        "gap", rank=comm.rank if comm is not None else MPI.COMM_WORLD.rank, report=report
    ) as dc_rec:
        ctx = _prepare_sector_context(
            model,
            basis,
            solver,
            comm=comm,
            verbosity=verbosity,
            memory_label="fixed-gap dc",
            de2_min=de2_min,
            e_pt2_tol=e_pt2_tol,
            cap_ledger=cap_ledger,
        )
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

            The one place ``E[N]`` is needed. It comes from :meth:`_SectorContext.sector_solve`, the
            cache the centre walk fills, so at a shift the walk has visited it costs nothing.
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
        dc_rec.certification = ctx.plan_store
        dc_rec["target"] = offset
        with _dc_search_trace("fixed-gap", MPI.COMM_WORLD, rank, width_tol=mu_tol, report=search_report):
            # Inside the trace block, not before it. The tolerance-sizing evaluation costs three
            # full sector solves, and `_report_dc_trace`'s cross-rank witness -- which allreduces
            # the `sector_solve` count precisely to catch ranks walking different collective paths
            # -- can only count what a trace is open for. Sizing the tolerance outside meant the
            # one evaluation whose result gates every later branch was invisible to the check
            # designed to notice it diverging.
            #
            # Calibrate the determinant cap against the tolerance BEFORE sizing that tolerance,
            # because the cap decides every number the sizing reads. Only when the cap defaulted:
            # an explicit `truncation_threshold` is the caller's instruction, not a candidate for
            # recalibration.
            #
            # Targets `mu_tol`, not the gap-scaled tolerance -- that one needs a width, which
            # needs a cap. `mu_tol` is `_size_gap_tolerance`'s own floor and, on both real
            # workloads this was measured against (`doc/plans/dc_smo_performance.md`), is what
            # the tolerance resolves to anyway (`tol_basis = "mu_resolution"`). Targeting the
            # floor errs toward a *larger* cap, the safe direction.
            if ctx.cap_from_memory:

                def _clear_gap_caches():
                    ctx.sector_at.clear()
                    ctx.plan_store.clear()
                    ctx.n_center_at.clear()
                    sectors_at.clear()
                    width_at.clear()

                cap, cached_cap = _calibrate_cap(
                    ctx,
                    lambda: _gap_centre_at_mu(0.0),
                    _clear_gap_caches,
                    mu_tol,
                    dc_rec,
                    verbose=verbose,
                    rank=rank,
                )
                # Keep the ladder's last rung when it *is* the accepted cap (the normal case:
                # `resolve_cap_at_max` returns the rung it last evaluated on every
                # exit path) rather than re-solving mu = 0 for values already in hand. Verified,
                # not assumed: keeping caches that describe a *different* cap would run the rest
                # of the search on another cap's energies while the record named the accepted one.
                if cached_cap != cap:
                    _clear_gap_caches()

            # Size the energy tolerance against the gap the model actually has. Falls back to the
            # mu resolution when the width is undefined at the guess (a shell edge, or a sector
            # the restrictions admit no determinants for) -- the search's own undefined-start
            # handling then takes over.
            width_at_guess = _gap_width_at(0.0)
            energy_tol, dc_rec["tol_basis"] = _size_gap_tolerance(width_at_guess, mu_tol, ctx.bandwidth, rank=rank)
            dc_rec["tol"] = energy_tol

            mu = None
            try:
                _refuse_an_ill_conditioned_gap(ctx, offset, energy_tol, _gap_centre_at_mu, dc_rec, search_report)
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
                # Unconditional, unlike the measurements below: a property of how the search *ran*, not of
                # whether a diagnostic could be taken at some `mu`. A record key written on some paths only
                # inherits stale values (`dc_record.recording` does `report.update(record)` and never clears).
                dc_rec["de2_min"] = ctx.de2_min
                dc_rec["e_pt2_tol"] = ctx.e_pt2_tol
                if mu_seen is not None:
                    delta_plus, delta_minus = _measure_edge_character(ctx, mu_seen, n_center_at[mu_seen])
                    dc_rec["delta_plus"], dc_rec["delta_minus"] = delta_plus, delta_minus
                    _warn_if_a_gap_edge_is_not_impurity_like(delta_plus, delta_minus, rank)

        # mu is always a point _solve_dc_shift actually evaluated, so sectors_at holds it.
        n_center, _e_add, _e_rem = _gap_sectors_at(mu)
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
        if gap_width is not None:
            _warn_if_a_neighbour_sector_is_thermal(
                min(centre_at[mu] + 0.5 * gap_width, 0.5 * gap_width - centre_at[mu]), ctx.tau, rank, "fixed-gap"
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
        # What the answer is actually pinned to, in the units of the answer. `chi` already carries
        # this conversion in its own annotation, but `chi` is a *secant* and is `None` on any
        # search that converged in one evaluation -- the common case in a converged CSC loop, and
        # exactly where a reader is most likely to take `mu` at face value. `delta_sum` is measured
        # from eigenvectors and is therefore available on every run that measured it, so the error
        # bar can be too.
        #
        # `is not None`, not truthiness: `delta_sum` is a real charge-transfer level crossing away
        # from landing on exactly `0.0` (`delta_minus`'s own docstring: it can go negative), and
        # that is a measured zero slope, not a missing measurement -- see `_mu_tol_effective`.
        delta_sum = dc_rec.get("delta_sum")
        if delta_sum is not None:
            dc_rec["mu_tol_effective"] = _mu_tol_effective(energy_tol, 0.5 * delta_sum)
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

        _reject_if_memory_bound(ctx, dc_rec, rank=rank, allow_memory_bound=allow_memory_bound)
        return (dc, n_center) if return_sector else dc
