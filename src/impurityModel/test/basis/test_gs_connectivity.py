"""``gs_connectivity``: the ground-state cap's model-dependent inputs are measured, not borrowed."""

from pathlib import Path

import pytest
from mpi4py import MPI

from impurityModel.ed import memory_estimate as me
from impurityModel.ed.gs_connectivity import GsConnectivity, measure_gs_connectivity
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.test.support.comms import COMMS
from impurityModel.test.symmetry.test_seed_sectors import BATH_STATES, IMPURITY_ORBITALS, _hamiltonian


def _measure(h_op, comm=None):
    return measure_gs_connectivity(h_op, IMPURITY_ORBITALS, BATH_STATES, {0: 2}, comm)


@pytest.mark.parametrize("comm", COMMS)
def test_the_connectivity_is_measured_and_bounded_by_what_an_operator_can_reach(comm):
    got = _measure(_hamiltonian(), comm)
    assert got.measured and got.probe_dets > got.seed_dets > 0
    assert 1.0 <= got.selection_fanout <= got.nnz_per_state <= got.probe_dets
    # Far below the SrMnO3 constants this replaces: a two-orbital toy connects a handful of rows.
    assert got.nnz_per_state < me._UNMEASURED_NNZ_PER_STATE
    assert got.selection_fanout < me._UNMEASURED_SELECTION_FANOUT


def test_a_chemical_potential_shift_does_not_move_the_connectivity():
    """The double-counting search applies ``H - mu N_imp``; its cap must be the self-energy's."""
    base = _measure(_hamiltonian())
    terms = dict(_hamiltonian())
    for orbital in (o for blocks in IMPURITY_ORBITALS.values() for block in blocks for o in block):
        key = ((orbital, "c"), (orbital, "a"))
        terms[key] = terms.get(key, 0) - 0.37
    shifted = _measure(ManyBodyOperator(terms))
    assert (shifted.nnz_per_state, shifted.selection_fanout, shifted.seed_dets) == (
        base.nnz_per_state,
        base.selection_fanout,
        base.seed_dets,
    )


@pytest.mark.mpi
def test_every_rank_gets_the_same_numbers():
    got = _measure(_hamiltonian(), MPI.COMM_WORLD)
    gathered = MPI.COMM_WORLD.allgather((got.nnz_per_state, got.selection_fanout, got.probe_dets))
    assert len(set(gathered)) == 1


def test_the_measured_connectivity_sizes_the_cap_and_the_log_says_so(monkeypatch, capsys):
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda comm: 5 * 2**30)
    monkeypatch.delenv("GS_MAX_BLOCK_WIDTH", raising=False)
    measured = GsConnectivity(nnz_per_state=12.0, selection_fanout=4.0, probe_dets=15_000, seed_dets=100, seconds=0.5)
    called = []

    def lazy():
        called.append(1)
        return measured

    policy, _ = me.resolve_cap_policy(None, 100, verbose=True, connectivity=lazy)
    unmeasured, _ = me.resolve_cap_policy(None, 100, verbose=True)
    out = capsys.readouterr().out
    assert called == [1]
    assert policy.gs > unmeasured.gs
    assert "measured on this model" in out and "UNMEASURED" in out


def test_a_cap_the_user_set_is_not_measured(monkeypatch):
    """The probe is only worth its seconds when a cap is sized or reported from it."""
    monkeypatch.setattr(me, "available_bytes_per_rank", lambda comm: 5 * 2**30)
    called = []
    me.resolve_cap_policy(50_000, 100, verbose=False, log="derived", connectivity=lambda: called.append(1))
    assert called == []


def test_every_driver_hands_its_solver_basis_connectivity_to_the_cap_resolver():
    """The double-counting search and the self-energy must size one cap from one set of numbers."""
    from impurityModel.ed import dc_criteria, groundstate, selfenergy, susceptibility

    for module, count in ((selfenergy, 1), (susceptibility, 1), (dc_criteria, 2), (groundstate, 1)):
        src = Path(module.__file__).read_text()
        assert src.count("connectivity=") >= count and "resolve_cap_policy(" in src, module.__name__


def test_the_window_width_is_the_lowest_levels_degeneracy_of_the_seed():
    got = _measure(_hamiltonian())
    assert got.window_states >= 1
    assert "states in the lowest level" in got.describe()
