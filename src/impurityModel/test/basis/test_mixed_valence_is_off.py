"""``mixed_valence`` no longer widens the seed; a non-zero value raises, identically on every rank.

The option is reserved for the charge-sector ensemble (``doc/plans/mixed_valence_sector_ensemble.md``).
Until that exists, honouring a non-zero value is impossible and ignoring it would hand a single-sector answer to
a caller who asked for a mixed-valence one, so it raises -- from replicated input, before any collective, so every
rank raises together instead of one rank leaving the others in a collective.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import groundstate
from impurityModel.ed.basis_generation import check_mixed_valence
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.test.support.comms import COMMS

IMPURITY = {0: [[0, 1]]}
BATHS = ({0: [[2, 3]]}, {0: [[4, 5]]})


def _model():
    terms = {}
    for s in (0, 1):
        for orbital, energy in ((s, 0.9), (2 + s, -1.2), (4 + s, 1.0)):
            terms[((orbital, "c"), (orbital, "a"))] = energy
    return ManyBodyOperator(terms)


@pytest.mark.parametrize("off", [None, False, 0, {0: 0}, {0: False}, {0: None}])
def test_the_off_spellings_pass(off):
    check_mixed_valence(off)


@pytest.mark.parametrize("on", [1, True, 2, {0: 1}, {0: 0, 1: 2}, -1])
def test_a_non_zero_value_raises(on):
    with pytest.raises(ValueError, match="charge-sector ensemble"):
        check_mixed_valence(on)


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
    assert set(d) == {"winner_total", "neighbour_totals", "margin", "second_difference", "undecided", "neighbour_gaps_tau"}
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
