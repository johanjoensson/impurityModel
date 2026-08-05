"""The one block of output every double-counting criterion leaves behind (Phase 5).

The record replaced a handful of ``if verbose and rank == 0`` summary lines that each printed a
different set of quantities in a different order. What has to hold for that to be an improvement
rather than a reshuffle: the block appears for *every* criterion, on the failure path as well as the
success path, without anyone having to ask for verbosity; the keys are a closed vocabulary rather
than whatever the search happened to write into its internal report; and the quantities that used to
be printed (the achieved value, the sector, ``chi``) are still there.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import dc_record, dc_search
from impurityModel.ed.dc_static import amf_dc, fll_dc, nominal_dc, sigma_inf_dc
from impurityModel.ed.selfenergy import fixed_gap_dc, fixed_occupation_dc, fixed_peak_dc

from ..support.dc_records import parse_record, records_in
from .test_fixed_dc import common_kwargs
from .test_static_dc import build_dimer_model, build_split_dimer_model


@pytest.fixture
def captured(monkeypatch):
    """Every record dict a criterion emits, captured before formatting.

    Intercepting :func:`dc_record.emit` rather than reading stdout is what makes the closed-
    vocabulary test possible at all: a key this module cannot print is *dropped* at format time,
    which is precisely the failure the vocabulary exists to prevent, so it has to be caught on the
    dict and not on the text.
    """
    seen = []

    def capture(record, rank=None):
        # The vocabulary check lives here, not in one test, so that *every* test in this module
        # which runs a criterion checks it -- including the failure-path ones, whose records carry
        # keys (`sector_verdict`, a `None` sector) that no success path writes. As a single
        # assertion in a single success-path test it could only ever have covered the converged
        # branch of seven criteria.
        unknown = set(record) - dc_record.KEYS
        assert not unknown, f"{record.get('criterion')} wrote keys the record cannot print: {sorted(unknown)}"
        seen.append(dict(record))

    monkeypatch.setattr(dc_record, "emit", capture)
    return seen


CRITERIA = [
    ("occupation", lambda: fixed_occupation_dc(occupation=1.0, **common_kwargs(v=0.3, tau=1e-2)[0])),
    ("peak", lambda: fixed_peak_dc(peak_position=1.2, **common_kwargs(v=0.4, tau=1e-3)[0])),
    ("gap", lambda: fixed_gap_dc(**common_kwargs(v=0.01, tau=1e-3)[0])),
    ("fll", lambda: fll_dc(build_dimer_model(v=0.01), tau=1e-3)),
    ("amf", lambda: amf_dc(build_dimer_model(v=0.01), tau=1e-3)),
    ("sigma_inf", lambda: sigma_inf_dc(build_dimer_model(v=0.01), tau=1e-3)),
    ("nominal", lambda: nominal_dc(build_dimer_model(v=0.01), 2)),
]


@pytest.mark.parametrize("criterion, call", CRITERIA, ids=[name for name, _ in CRITERIA])
def test_every_criterion_emits_exactly_one_record_with_a_known_vocabulary(captured, criterion, call):
    """Both halves matter. *One* record: :func:`nominal_dc` is FLL evaluated at the nominal
    occupation and calls the same formula, so a naive implementation emits two, the first of them
    naming the wrong criterion. *Known keys*: anything else is silently dropped when printed.
    """
    call()
    assert len(captured) == 1, [record.get("criterion") for record in captured]
    record = captured[0]
    assert record["criterion"] == criterion
    # The vocabulary itself is asserted by the `captured` fixture, on every record any test in this
    # module produces. What is left to check here is the one-record-per-call property.
    #
    # The level is the trace over the *model's* impurity orbital count, read off the record rather
    # than hardcoded: every fixture here happens to have two, and a literal 2 would silently stop
    # testing anything the moment one did not.
    n_imp = record["dc_trace"] / record["dc_level"]
    assert n_imp == pytest.approx(2), n_imp


@pytest.mark.parametrize("criterion, call", CRITERIA, ids=[name for name, _ in CRITERIA])
def test_the_record_needs_no_verbosity(capsys, criterion, call):
    """Unconditional by design: the double counting *is* the answer of the call, and a default-
    verbosity run whose log does not say what it decided cannot be audited afterwards. Every call
    above passes no ``verbosity``.
    """
    call()
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank != 0:
        # Rank-0 only, asserted rather than skipped, so the gating itself is under test.
        assert records_in(out) == 0, out
        return
    record = parse_record(out)
    assert record["criterion"] == criterion
    assert record["status"] in ("converged", "closed_form")
    assert "walltime" in record


def test_the_search_reports_its_own_sector_against_the_nominal_one(capsys):
    """The sector line is the DC<->GS parity claim in one field: which charge state the double
    counting was measured on, beside the one the user asked for."""
    kwargs, _ = common_kwargs(v=0.4, tau=1e-3)
    fixed_peak_dc(peak_position=1.2, **kwargs)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank != 0:
        return
    sector = parse_record(out)["sector"]
    assert "(nominal " in sector and sector.endswith("OK"), sector


def test_a_failed_search_still_leaves_a_record(capsys):
    """The record is emitted from a ``finally``, for the same reason the cost accounting is: the
    search that could not reach its target is the one whose ``mu``, sector and evaluation count are
    worth having, and a traceback carries none of them."""
    kwargs, _ = common_kwargs(v=0.4, tau=1e-3)
    with pytest.raises(dc_search.DoubleCountingUnreachable):
        # 1.05 sits on a charge-sector plateau the observable steps across (B3).
        fixed_occupation_dc(occupation=1.05, **kwargs)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank != 0:
        return
    record = parse_record(out)
    assert record["status"] == "unreachable", record
    # And it says how far it got, which is the whole point of recording a failure.
    assert "mu" in record and int(record["evaluations"]) > 1, record


@pytest.mark.parametrize(
    "call",
    [
        lambda report: fixed_occupation_dc(occupation=1.0, report=report, **common_kwargs(v=0.3, tau=1e-2)[0]),
        lambda report: fixed_peak_dc(peak_position=1.2, report=report, **common_kwargs(v=0.4, tau=1e-3)[0]),
        lambda report: fixed_gap_dc(report=report, **common_kwargs(v=0.01, tau=1e-3)[0]),
    ],
    ids=["occupation", "peak", "gap"],
)
def test_the_report_out_parameter_is_the_record(captured, call):
    """One result, one vocabulary.

    ``report=`` existed on ``fixed_gap_dc`` alone and spoke a different dialect from the emitted
    record -- ``gap_centre`` against ``gap_center``, ``energy_tol`` against ``tol``, and the
    search's raw internal status against the normalized one. Two names for one number is how a
    caller ends up asserting on a field that quietly stopped being written; all three criteria now
    hand back exactly what they printed.
    """
    report = {}
    call(report)
    assert report == captured[0], sorted(set(report) ^ set(captured[0]))
    # Including the pieces a fill at the return statement could not have carried: the projected
    # search verdict and the walltime, both written after the criterion computes its answer.
    assert "status" in report and "walltime" in report


def test_a_failed_search_still_fills_the_report(captured):
    """A caller that passes ``report=`` and catches the raise still learns where the search got to
    -- the same reason the record itself is emitted from a ``finally``."""
    report = {}
    with pytest.raises(dc_search.DoubleCountingUnreachable):
        fixed_occupation_dc(occupation=1.05, report=report, **common_kwargs(v=0.4, tau=1e-3)[0])
    assert report["status"] == "unreachable"
    assert "mu" in report and report["evaluations"] > 1


def test_a_broken_record_cannot_mask_the_exception_it_is_documenting(capsys, monkeypatch):
    """The record is emitted from a ``finally``, so anything it raises *replaces* the exception in
    flight -- the solver's failure would survive only as ``__context__`` while the traceback blamed
    a format string. Reachable rather than hypothetical: ``"{:.6f}".format`` raises ``TypeError``
    on a numpy array, and a criterion writing an array into a float field is one typo away.
    """

    def exploding_emit(record, rank=None):
        raise TypeError("unsupported format string passed to numpy.ndarray.__format__")

    monkeypatch.setattr(dc_record, "emit", exploding_emit)

    with pytest.raises(ValueError, match="the real failure"), dc_record.recording("gap"):
        raise ValueError("the real failure")
    assert "could not format the double-counting record" in capsys.readouterr().out

    # And on the success path a broken record must not turn a good answer into an exception.
    with dc_record.recording("fll") as record:
        record["dc_level"] = 1.0
    assert "could not format the double-counting record" in capsys.readouterr().out


def test_the_dc_matrices_are_dumped_only_under_the_diagnostics_knob(capsys, monkeypatch):
    """What the record's scalar ``dc_level`` cannot say is whether the matrix is uniform at all.
    That question is real only for a non-uniform scheme, so the full dump moved behind
    ``DC_DIAGNOSTICS`` instead of printing two complex matrices on every verbose call."""
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    fixed_occupation_dc(occupation=1.0, verbosity=1, **kwargs)
    assert "DC found:" not in capsys.readouterr().out

    monkeypatch.setenv("DC_DIAGNOSTICS", "1")
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    fixed_occupation_dc(occupation=1.0, verbosity=1, **kwargs)
    out = capsys.readouterr().out
    if MPI.COMM_WORLD.rank == 0:
        assert "DC guess:" in out and "DC found:" in out


def test_dc_level_is_the_per_orbital_potential_not_the_trace(captured):
    """The literature quotes double counting as a single potential in eV -- Karolak et al.'s
    25.3 eV for NiO -- and that is the *mean diagonal*. Reporting a trace under that name is a
    factor of ``n_imp`` in any comparison against a published number.

    The split dimer is the case that separates them: ``sigma_inf_dc`` at an anisotropic density
    matrix is the one scheme here whose ``dc`` is not a uniform shift, so its level is a genuine
    average over two different potentials rather than the value of both.
    """
    # One impurity level deep below E_F and one well above it, so the density matrix is diag(0, 1)
    # rather than the identity every uniform-fill fixture produces (test_static_dc.py:93).
    model = build_split_dimer_model(v=0.001, eps1=-1.0, eps2=5.0, eps_b=-100.0)
    dc = sigma_inf_dc(model, tau=1e-4)
    record = captured[0]
    assert not np.allclose(dc[0, 0], dc[1, 1])
    assert record["dc_trace"] == pytest.approx(np.real(np.trace(dc)))
    assert record["dc_level"] == pytest.approx(np.real(np.trace(dc)) / 2)


def test_the_searchs_internal_report_does_not_leak_into_the_record():
    """The record is a published format; :func:`dc_search._solve_dc_shift`'s report is working
    state. Projection is explicit and one-directional, so a key the search adds later cannot
    silently appear in -- and then silently change -- the block."""
    record = {"criterion": "occupation"}
    dc_record.project_search_report(
        record,
        {
            "status": "outside_nominal_sector",
            "mu": 1.5,
            "sector": 9,
            "nominal_sector": 8,
            "evaluations": 4,
            # Internal: which seed found the sector, and whether a plateau was mapped.
            "nominal_sector_seed": 0.75,
            "plateau": False,
            "interval": None,
        },
    )
    assert set(record) <= dc_record.KEYS
    assert "nominal_sector_seed" not in record and "interval" not in record
    # And the internal status names collapse onto the four the record reports.
    assert record["status"] == "sector_unattainable"


@pytest.mark.parametrize(
    "status, reported",
    [
        ("converged_at_guess", "converged"),
        ("nominal_sector", "converged"),
        ("unconstrained", "converged"),
        ("outside_nominal_sector", "sector_unattainable"),
        ("plateau", "plateau"),
        ("unreachable", "unreachable"),
    ],
)
def test_every_internal_status_maps_onto_a_reported_one(status, reported):
    record = {}
    dc_record.project_search_report(record, {"status": status})
    assert record["status"] == reported


def test_chi_carries_the_uncertainty_it_implies():
    """``chi`` is reported so a converged residual can be converted into an error on the *answer*:
    ``delta_mu = tol / chi``. A criterion can converge its observable perfectly and still leave
    ``dc`` undetermined by several eV -- the plateau -- and that is invisible without the
    conversion beside it."""
    lines = dc_record.format_record({"criterion": "occupation", "chi": 0.0071, "tol": 1e-2})
    assert "dc determined to +- 1.41e+00" in "\n".join(lines)
    # A flat observable determines nothing, and dividing by it would say so with an infinity.
    flat = dc_record.format_record({"criterion": "occupation", "chi": 0.0, "tol": 1e-2})
    assert "dc undetermined" in "\n".join(flat)
    # Measured-and-absent is not the same statement as never-measured. `tol` is the discriminator:
    # a criterion that searched has one, a closed form does not.
    assert "chi" not in "\n".join(dc_record.format_record({"criterion": "fll"}))
    assert "chi         = not resolvable" in "\n".join(
        dc_record.format_record({"criterion": "occupation", "chi": None, "tol": 1e-2})
    )


def test_the_record_prints_on_rank_zero_of_the_communicator_it_was_given(capsys):
    """The rank convention is the criteria's own: they warn and report on ``comm.rank == 0``, not
    on the world root, so a per-cluster solve on a sub-communicator leaves one record per cluster
    rather than one for whichever cluster happens to own world rank 0."""
    record = {"criterion": "fll", "status": "closed_form", "dc_level": 1.0}
    dc_record.emit(record, rank=1)
    assert capsys.readouterr().out == ""
    dc_record.emit(record, rank=0)
    assert "criterion   = fll" in capsys.readouterr().out
