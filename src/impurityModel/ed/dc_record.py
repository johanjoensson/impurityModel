"""The one block of output a double-counting calculation leaves behind.

Every criterion -- the three searches in :mod:`dc_criteria` and the four closed forms in
:mod:`dc_static` -- ends by emitting a single delimited ``key = value`` record on rank 0. It is
unconditional: not gated on ``verbosity``, because the double counting is the *answer* of the call
and a run whose log does not say what it decided cannot be audited afterwards, and not gated on
``DC_DIAGNOSTICS`` either, which turns on the per-``mu`` cost table and the full ``dc`` matrix dumps
instead (:func:`dc_search._report_dc_trace`, the ``matrix_print`` calls in :mod:`dc_criteria`).

What it replaced was a handful of ``if verbose and rank == 0`` summary lines, each written in a
different order with a different set of quantities, plus two matrix dumps per call. The record is
the same fields for every criterion, in the same order, so a grep over a charge-self-consistency log
returns a table rather than a reading exercise.

**The keys are a closed vocabulary** (:data:`_FIELDS`). The searches fill their record from
:func:`dc_search._solve_dc_shift`'s internal ``report`` dict, which carries working state
(``nominal_sector_seed``, ``interval``) that is nobody's business outside the search and would
otherwise appear in, and then silently change, the public block. Projection is explicit and
one-directional; ``test_dc_record.py`` asserts that every key the criteria write is one this module
knows how to print, so a typo is a test failure rather than a field that quietly vanishes.

Nothing downstream parses this. RSPt reads only the integer return code
(``green_solver.F90:1438``), so the format is free to be human-first -- but it is kept
machine-readable anyway (one ``key = value`` per line, annotations after the value) because the
person reading a hundred CSC iterations will want to ``awk`` it.
"""

import time
from contextlib import contextmanager

import numpy as np
from mpi4py import MPI

RECORD_HEADER = "=== impurityModel double-counting ==="
RECORD_FOOTER = "====================================="

#: The record's vocabulary and print order: ``(key, format)``. A key whose value is missing or
#: ``None`` is skipped entirely rather than printed as ``None`` -- "this criterion has no gap width"
#: and "the gap width could not be measured" are both better said by absence than by a null.
#: ``sector`` and ``chi`` carry an annotation built by :func:`_annotate` instead of a bare format.
#:
#: Most keys are therefore criterion-specific, and one is deliberately narrower than it looks:
#: ``n_gs``, the thermal impurity occupation, is filled by ``fixed_occupation_dc`` alone. The peak
#: and gap criteria work on :class:`dc_criteria._SectorContext`, which yields the *integer* charge
#: sector and nothing else, so reporting a thermal occupation beside them would cost an extra
#: ground-state solve for a cosmetic field. ``sector`` is what they have, and it is the number the
#: DC<->GS parity claim is actually about.
_FIELDS = (
    ("criterion", "{}"),
    ("status", "{}"),
    ("sector_verdict", "{}"),
    ("target", "{:.6g}"),
    ("mu", "{:.6f}"),
    ("dc_trace", "{:.6f}"),
    ("dc_level", "{:.6f}"),
    ("dc_spread", "{:.6f}"),
    ("sector", "{}"),
    ("n_gs", "{:.4f}"),
    ("n_ref", "{:.4f}"),
    ("u", "{:.4f}"),
    ("j", "{:.4f}"),
    ("gap_center", "{:.6f}"),
    ("gap_width", "{:.6f}"),
    ("omega_plus", "{:.6f}"),
    ("omega_minus", "{:.6f}"),
    ("delta_plus", "{:.3f}"),
    ("delta_minus", "{:.3f}"),
    ("delta_sum", "{:.3f}"),
    ("delta_sum_vs_chi", "{:.3f}"),
    ("manifold_spread", "{:.3f}"),
    ("peak", "{:.6f}"),
    ("chi", "{:.4g}"),
    ("chi_span", "{:.4g}"),
    ("tol", "{:.2e}"),
    ("tol_basis", "{}"),
    ("alpha", "{:.3g}"),
    ("evaluations", "{}"),
    ("walltime", "{:.1f} s"),
)

#: Keys that are part of the vocabulary but never get a line of their own: they annotate one that
#: does. ``nominal_sector`` follows ``sector`` on its line, and ``n_ref_kind`` says which of the
#: two very different fillings ``n_ref`` is -- a DFT reference read off the bath fit, or the
#: nominal integer ``nominal_dc`` was handed. Same key, same units, incomparable provenance.
#: ``manifold_states`` follows ``manifold_spread`` because a spread is unreadable without the
#: number of states it is a spread over: zero across one state is silence, not agreement.
_ANNOTATIONS = frozenset({"nominal_sector", "n_ref_kind", "manifold_states"})

#: Everything a criterion may write into a record. Enforced by a test, not at runtime: a key this
#: module does not know is dropped silently at print time, which is exactly the failure mode the
#: closed vocabulary exists to prevent, so it has to be caught before it ships.
KEYS = frozenset(key for key, _ in _FIELDS) | _ANNOTATIONS

#: How :func:`dc_search._solve_dc_shift`'s internal status names map onto the four the record
#: reports. The internal names distinguish *how* the answer was found (at the guess, inside the
#: nominal sector, with no sector constraint at all); the record answers the only question a log
#: reader has, which is whether to trust the number.
_STATUS = {
    "converged_at_guess": "converged",
    "nominal_sector": "converged",
    "unconstrained": "converged",
    "outside_nominal_sector": "sector_unattainable",
    "plateau": "plateau",
    "unreachable": "unreachable",
    "plateau_rejected": "plateau_rejected",
}


def dc_levels(dc):
    r"""``(trace, level)`` of a double-counting matrix: :math:`\mathrm{Tr}\,dc` and its per-orbital
    mean.

    The literature quotes double counting as a single potential in eV -- Karolak et al.'s 25.3 eV
    for NiO, say -- and that number is the *mean diagonal*, not the trace: a 10-spin-orbital d shell
    at 24.9 eV has a trace of 249. Both are reported because they answer different questions, and
    reporting only one of them mislabelled is how a factor of ``n_imp`` gets into a comparison
    against a published value.

    ``level`` alone is the whole story only for a **uniform** ``dc``, which is what the two searches
    return by construction (``dc_guess + mu * I``) but is *not* what :func:`dc_static.sigma_inf_dc`
    returns -- that one is the Hartree-Fock potential at an anisotropic density matrix, and its
    orbital dependence is visible only in the matrix itself. Set ``DC_DIAGNOSTICS`` for that; the
    record says the average.
    """
    matrix = np.asarray(dc)
    # `np.real` on the trace is only defensible for a Hermitian dc, which every scheme here
    # produces and RSPt requires; check it rather than quietly discarding an imaginary part that
    # would mean something had gone wrong upstream. Off-diagonal weight is separately not
    # summarised by either number -- see `dc_spread` for the diagonal's own caveat.
    #
    # Scaled and deliberately loose. This is a *reporting* helper: turning a working static scheme
    # into a hard failure over accumulated roundoff would be a far worse regression than the
    # mislabelled number it guards against, and `sigma_inf_dc` reaches here through a chain of
    # tensor contractions. The archived RSPt double countings measure 1e-47 and 0 against their own
    # adjoint, so 1e-8 relative is orders of magnitude away from anything real.
    scale = max(1.0, float(np.max(np.abs(matrix))) if matrix.size else 1.0)
    if not np.allclose(matrix, matrix.conj().T, atol=1e-8 * scale, rtol=0.0):
        raise ValueError(f"the double counting is not Hermitian; its trace is not a real level:\n{matrix}")
    trace = float(np.real(np.trace(matrix)))
    return trace, trace / matrix.shape[0]


def dc_spread(dc):
    """Peak-to-peak spread of the diagonal, or ``None`` when it is uniform to roundoff.

    What :func:`dc_levels`' per-orbital mean cannot say. The two searches return
    ``dc_guess + mu * I``, so their spread is whatever the *guess* had; :func:`dc_static.sigma_inf_dc`
    is Hartree-Fock at an anisotropic density matrix and its orbital dependence -- the ``eg``/``t2g``
    splitting that is the entire difference between it and ``amf`` -- is exactly what averaging
    destroys. A record that prints one number for both cases with no marker cannot be read: the
    spread is that marker, present only when there is something to mark.
    """
    diagonal = np.real(np.diag(np.asarray(dc)))
    spread = float(np.ptp(diagonal)) if diagonal.size else 0.0
    return None if spread <= 1e-10 else spread


def _annotate(record, key, text):
    """The parenthetical that follows a value: what makes ``sector`` and ``chi`` readable."""
    if key == "sector":
        nominal = record.get("nominal_sector")
        if nominal is None:
            return text
        verdict = "OK" if record["sector"] == nominal else "MISMATCH"
        return f"{text}   (nominal {nominal})  {verdict}"
    if key == "alpha":
        return f"{text}   (damping toward the previous iteration's answer; 1 = undamped)"
    if key == "dc_spread":
        return f"{text}   (peak-to-peak over the diagonal: dc_level is an AVERAGE, not the value)"
    if key == "n_ref":
        return f"{text}   ({record.get('n_ref_kind', 'DFT reference filling')})"
    if key == "chi_span":
        return f"{text}   (mu-distance between the two points chi was measured from)"
    if key == "delta_plus":
        # "~1 impurity-like", not "of max 1". Only the change summed over impurity AND bath is one
        # electron; the bath's share can be negative, so this is not bounded above by 1. At a
        # charge-transfer level crossing -- N ground state d8, N+1 ground state d10 L -- it is 2.
        # That regime is the one the gap criterion is aimed at, so a range the number does not
        # obey is worse than no range at all.
        return f"{text}   (how much of an ADDED electron lands on the impurity; ~1 impurity, ~0 bath)"
    if key == "delta_minus":
        # Split out from delta_sum because the sum hides the case that matters. On NiO the two
        # are 0.561 and 0.032: the addition edge is a genuine impurity state and the removal edge
        # is the ligand valence band, which is what a charge-transfer insulator *is*. A symmetric
        # pair summing to the same 0.59 would mean something entirely different, and the record
        # could not tell them apart. Can come out negative, for the reason above.
        return f"{text}   (how much of a REMOVED electron came off the impurity; ~1 impurity, ~0 bath)"
    if key == "delta_sum":
        # Kept for continuity with older logs, and demoted: `delta_plus` and `delta_minus` above
        # are what should be read. The sum hides the case the pair exists for -- NiO's (0.561,
        # 0.032) sums to 0.593, which looks unremarkable next to a healthy 2.0 while one edge is
        # pure ligand. Say which of the two ways it was obtained, because they are not equally
        # trustworthy: the edges are measured from eigenvectors at the converged shift, while the
        # fallback is a secant of the gap centre over `chi_span` and is an average across that
        # interval rather than the value at the answer.
        source = (
            "measured at the converged shift"
            if record.get("delta_plus") is not None and record.get("delta_minus") is not None
            else "from the chi secant, averaged over chi_span"
        )
        return f"{text}   (delta_+ + delta_-, of max 2; {source} -- read the two edges separately)"
    if key == "delta_sum_vs_chi":
        # The two routes to the same quantity, differenced. They fail differently: `delta_sum` is
        # built from occupations, which are first-order in the wavefunction error, while `-2 chi`
        # is a secant of energies, second-order in it but carrying a finite-difference term and a
        # basis-reselection term instead. Near zero means the sector solves are converged enough
        # that neither error shows; large means one of them is not, and `chi_span` says whether to
        # suspect the secant's interval.
        return f"{text}   (delta_sum minus -2*chi; two estimators with opposite error structure)"
    if key == "manifold_spread":
        # The one number that says whether `delta_+-` being thermal while `omega_+-` are T=0
        # matters on this model. Hellmann-Feynman relates them state by state, so the pairing is
        # exact only when the retained manifold's states share N_imp -- which is what a zero spread
        # says. Sizes come along because a zero spread over a single retained state is silence.
        sizes = record.get("manifold_states")
        over = "" if sizes is None else f" over {sizes} states (N+1/N/N-1)"
        return f"{text}   (max N_imp spread within a retained manifold{over}; 0 = thermal and T=0 agree)"
    if key == "chi":
        # The residual is converged to `tol`; what the *answer* is determined to is tol / |chi|.
        # That conversion is the entire reason chi is reported (review correction 5): a criterion
        # can converge its observable perfectly and still leave `dc` undetermined by several eV,
        # and the plateau is exactly the case where it does.
        tol = record.get("tol")
        chi = record["chi"]
        if not chi:
            return f"{text}   (dc undetermined: the observable does not respond to mu)"
        if tol is None:
            # A measured slope with no tolerance beside it is still worth reporting; it just does
            # not convert into an error bar, and claiming one from a missing number would be worse
            # than the bare slope.
            return text
        # "search tolerance only" is not hedging, it is the scope of the claim: tol/|chi| converts
        # the residual the search converged into an error on the answer, and says nothing about
        # the truncation error in the observable itself -- which for the gap centre is the larger
        # of the two. Printing a bare +- would overstate what has been established.
        return f"{text}   (dc determined to +- {abs(tol / chi):.2e} by the search tolerance alone)"
    return text


def format_record(record):
    """The record's lines, in :data:`_FIELDS` order. Unknown and ``None`` valued keys are dropped."""
    lines = [RECORD_HEADER]
    for key, fmt in _FIELDS:
        value = record.get(key)
        if value is None:
            if key == "chi" and "tol" in record:
                # Distinguish "this criterion does not measure a slope" from "it tried and the
                # evaluated points do not support one" -- the second is a statement about how well
                # `dc` is pinned and must not read as the first. `tol` is the discriminator because
                # it is exactly what a criterion that *searches* has and a closed form does not; a
                # separate flag beside it would be one more thing for a new criterion to forget.
                lines.append(f"{'chi':<11} = not resolvable")
            continue
        lines.append(f"{key:<11} = {_annotate(record, key, fmt.format(value))}")
    lines.append(RECORD_FOOTER)
    return lines


def emit(record, rank=None):
    """Print the record on rank 0. Pure output -- no collective, nothing to gate.

    ``rank`` defaults to ``MPI.COMM_WORLD``'s. A criterion running on a sub-communicator passes its
    own, so the record follows the same rank convention as the warnings around it.
    """
    if rank is None:
        rank = MPI.COMM_WORLD.rank
    if rank != 0:
        return
    print("\n".join(format_record(record)), flush=True)


class Record(dict):
    """The record being built, plus the search report it is projected from.

    A plain dict everywhere it is written to, with one attribute: ``search``, the dict a criterion
    hands :func:`dc_search._solve_dc_shift`. It is held by *reference* and read at emit time, which
    is what lets the failure path report the search's own verdict -- the criteria learn ``mu`` and
    the status from local variables the raise skips past, but the search filled its report before
    raising, and that dict is still reachable from here.
    """

    search = None


@contextmanager
def recording(criterion, rank=None, report=None):
    """Collect a double-counting record and emit it, whatever happens to the calculation.

    Yields the (mutable) :class:`Record`; ``criterion``, ``walltime``, and everything in
    :attr:`Record.search` are filled in here. The emit is in a ``finally``, for the same reason
    :func:`dc_search._dc_search_trace`'s report is: a search that raises
    :class:`dc_search.DoubleCountingUnreachable` is precisely the one whose record -- how far it got,
    in which sector, over how many evaluations -- is worth having, and a traceback alone does not
    carry it.

    The clock starts here, which is why callers open this *before* building the solver basis rather
    than around the search: the memory probe and the basis build are a large share of a short run,
    and a walltime that excludes them would understate the criterion's cost by more than the search
    itself takes.
    """
    # "incomplete" is what a criterion that never got as far as an answer reports -- a raise inside
    # the basis build, before any search or formula ran. The static schemes overwrite it with
    # "closed_form"; a search's own verdict arrives with the projection below.
    record = Record(criterion=criterion, status="incomplete")
    started = time.monotonic()
    try:
        yield record
    finally:
        # A diagnostic must never be able to hide the failure it exists to document. This runs in a
        # `finally`, so anything raised here *replaces* an in-flight exception -- a
        # DoubleCountingUnreachable would survive only as `__context__` while the traceback blamed
        # a format string. Reachable, not hypothetical: `"{:.6f}".format` raises TypeError on a
        # numpy array, and a criterion writing an array into a float-formatted field is one typo
        # away. Degrade to a warning and let the real exception through.
        try:
            if record.search is not None:
                project_search_report(record, record.search)
            record["walltime"] = time.monotonic() - started
            if report is not None:
                # The caller's out-parameter is the *same* dict the block below prints, filled
                # here rather than at the criterion's return statement so that it carries the
                # projected search verdict and the walltime -- and so that a criterion which
                # raises still hands back how far it got, which a fill at the return could not.
                report.update(record)
            emit(record, rank=rank)
        except Exception as exc:
            print(f"WARNING: could not format the double-counting record ({exc!r}); the calculation is unaffected.")


def project_search_report(record, search_report):
    """Copy the public half of :func:`dc_search._solve_dc_shift`'s report into ``record``.

    One-directional and explicit: the search's dict is internal working state (which seed found the
    nominal sector, whether a plateau was mapped) and the record's is a published format. Anything
    the search adds later appears in the record only when it is named here.
    """
    status = search_report.get("status")
    if status is not None:
        record["status"] = _STATUS.get(status, status)
    for key in ("mu", "sector", "nominal_sector", "evaluations", "sector_verdict"):
        if search_report.get(key) is not None:
            record[key] = search_report[key]
