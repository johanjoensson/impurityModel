"""``mixed_valence`` no longer widens the seed; an explicit ``on`` raises where no driver forms the ensemble yet.

The option is the charge-sector ensemble switch (``doc/plans/mixed_valence_sector_ensemble.md``). Until a driver is
migrated, honouring ``on`` is impossible and ignoring it would hand a single-sector answer to a caller who asked
for a mixed-valence one, so it raises -- from replicated input, before any collective, so every rank raises together
instead of one rank leaving the others in a collective. ``auto`` and ``off`` run the winning sector.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import groundstate
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.sector_ensemble import reject_unmigrated_ensemble, resolve_mixed_valence
from impurityModel.test.support.comms import COMMS

IMPURITY = {0: [[0, 1]]}
BATHS = ({0: [[2, 3]]}, {0: [[4, 5]]})


def _model():
    terms = {}
    for s in (0, 1):
        for orbital, energy in ((s, 0.9), (2 + s, -1.2), (4 + s, 1.0)):
            terms[((orbital, "c"), (orbital, "a"))] = energy
    return ManyBodyOperator(terms)


@pytest.mark.parametrize("off", [None, False, 0, "off", "OFF", "false", {0: 0}, {0: False}, {0: None}, np.int64(0)])
def test_the_off_spellings_pass(off):
    assert reject_unmigrated_ensemble(off, driver="x") == "off"


def test_auto_runs_the_winning_sector_and_never_raises():
    assert reject_unmigrated_ensemble("auto", driver="x") == "auto"
    assert reject_unmigrated_ensemble(None, driver="x", driver_default="auto") == "auto"


@pytest.mark.parametrize("on", [1, True, 2, "on", "True", b"on", {0: 1}, {0: 0, 1: 2}, np.int64(3)])
def test_an_explicit_on_raises(on):
    with pytest.raises(ValueError, match="charge-sector ensemble"):
        reject_unmigrated_ensemble(on, driver="x")


@pytest.mark.parametrize(
    "value, default, mode, noted",
    [
        (None, "auto", "auto", False),
        (None, "off", "off", False),
        ("auto", "off", "auto", False),
        (True, "off", "on", False),
        ("on", "off", "on", False),
        (False, "auto", "off", False),
        (0, "auto", "off", False),
        (2, "off", "on", True),
        ({0: 0}, "auto", "off", False),
        ({0: 1}, "auto", "on", True),
    ],
)
def test_the_resolver_mapping_table(value, default, mode, noted):
    got, note = resolve_mixed_valence(value, driver_default=default)
    assert got == mode and (note is not None) == noted


@pytest.mark.parametrize("bad", [-1, "maybe", 1.5, {0: "auto"}, [1]])
def test_the_resolver_refuses_nonsense(bad):
    with pytest.raises(ValueError):
        resolve_mixed_valence(bad, driver_default="off")


@pytest.mark.mpi
@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("entry", ["find_ground_state_sector", "solve_ground_state"])
def test_every_rank_raises_at_the_public_entries(comm, entry):
    """No rank gets past the check, so none is left in a collective waiting for the others."""
    common = dict(slaterWeightMin=1e-12, comm=comm, truncation_threshold=np.inf, verbose=False)
    args = (_model(), IMPURITY, BATHS, {0: 1})
    raised = False
    try:
        if entry == "find_ground_state_sector":
            groundstate.find_ground_state_sector(*args, mixed_valence={0: 1}, **common)
        else:
            groundstate.solve_ground_state(*args, tau=0.1, mixed_valence={0: 1}, **common)
    except ValueError as err:
        raised = "charge-sector ensemble" in str(err)
    assert all((comm or MPI.COMM_SELF).allgather(raised))


@pytest.mark.mpi
def test_prepare_solver_basis_raises_before_its_collective():
    from impurityModel.ed.solver_basis import prepare_solver_basis

    raised = False
    try:
        prepare_solver_basis(
            {}, {}, {}, IMPURITY, {0: 1}, mixed_valence=1, rot_to_spherical=None, verbosity=0, comm=MPI.COMM_WORLD
        )
    except ValueError as err:
        raised = "charge-sector ensemble" in str(err)
    assert all(MPI.COMM_WORLD.allgather(raised))


@pytest.mark.mpi
def test_the_convergence_report_carries_the_sector_diagnostics():
    """MV 3: how firmly the winner beat its neighbours reaches ``convergence_report``, which the JSON reads."""
    gs = groundstate.solve_ground_state(
        _model(), IMPURITY, BATHS, {0: 1}, tau=0.1, slaterWeightMin=1e-12, truncation_threshold=np.inf, verbose=False
    )
    d = gs.solver.convergence_report["sector_diagnostics"]
    assert set(d) == {
        "winner_total",
        "neighbour_totals",
        "margin",
        "second_difference",
        "undecided",
        "neighbour_gaps_tau",
    }
    assert d["undecided"] is False and isinstance(d["neighbour_totals"], list)


@pytest.mark.mpi
def test_a_neighbour_inside_the_thermal_window_is_warned_about(capsys):
    """P9: the toy's N_imp = 0 winner has an N_imp = 1 neighbour 0.9 above; a tau of 1 puts it in the window."""
    kw = dict(slaterWeightMin=1e-12, truncation_threshold=np.inf, verbose=False)
    capsys.readouterr()
    gs = groundstate.solve_ground_state(_model(), IMPURITY, BATHS, {0: 1}, tau=1.0, **kw)
    out = capsys.readouterr().out
    gaps = gs.solver.convergence_report["sector_diagnostics"]["neighbour_gaps_tau"]
    assert gaps and min(gaps.values()) < groundstate.NEIGHBOUR_GAP_WARN
    assert "lies within the thermal window" in out  # comm=None: each rank is its own solve
    capsys.readouterr()
    groundstate.solve_ground_state(_model(), IMPURITY, BATHS, {0: 1}, tau=1e-3, **kw)
    assert "lies within the thermal window" not in capsys.readouterr().out


@pytest.mark.mpi
def test_the_warning_can_be_silenced_and_the_report_is_strict_json(capsys):
    """The DC search turns the P9 warning off; an infinite margin is stored as null so the file parses strictly."""
    import json

    kw = dict(slaterWeightMin=1e-12, truncation_threshold=np.inf, verbose=False)
    capsys.readouterr()
    gs = groundstate.solve_ground_state(_model(), IMPURITY, BATHS, {0: 1}, tau=1.0, warn_neighbours=False, **kw)
    assert "lies within the thermal window" not in capsys.readouterr().out
    d = gs.solver.convergence_report["sector_diagnostics"]
    json.dumps(d, allow_nan=False)
