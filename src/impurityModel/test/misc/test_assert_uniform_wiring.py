"""Every public solver entry checks that the env knobs agree across ranks, on the comm it collects on (M1)."""

import inspect

import pytest
from mpi4py import MPI

from impurityModel.ed import (
    config,
    dc_criteria,
    get_spectra,
    groundstate,
    gs_ensemble,
    rixs,
    selfenergy,
    susceptibility,
)

# (function, the comm expression its collectives run on)
ENTRIES = [
    (groundstate.solve_ground_state, "comm"),
    (gs_ensemble.solve_charge_sector_ensemble, "comm"),
    (groundstate.find_ground_state_sector, "comm"),
    (groundstate.solve_sector, "comm"),
    (selfenergy.calc_selfenergy, "comm"),
    (get_spectra.run_spectra, "comm"),
    (susceptibility.calc_susceptibility, "basis.comm"),
    (susceptibility.calc_susceptibility_workflow, "comm"),
    (rixs.calc_map, "basis.comm"),
    (rixs.calc_tensor_map, "basis.comm"),
    # their collectives hard-code COMM_WORLD whatever `comm` says
    (dc_criteria.fixed_peak_dc, "MPI.COMM_WORLD"),
    (dc_criteria.fixed_gap_dc, "MPI.COMM_WORLD"),
    (dc_criteria.occupation_and_energy_at_mu, "MPI.COMM_WORLD"),
    (dc_criteria.fixed_occupation_dc, "MPI.COMM_WORLD"),
]


@pytest.mark.parametrize("func, comm_expr", ENTRIES, ids=[f.__name__ for f, _ in ENTRIES])
def test_entry_asserts_uniform_knobs_on_its_own_comm(func, comm_expr):
    assert f"config.assert_uniform({comm_expr})" in inspect.getsource(func)


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs two ranks to disagree")
def test_solve_ground_state_raises_on_every_rank_when_a_knob_differs(monkeypatch):
    """Before this check, GS_CERTIFY_MAX_SEED=1 on one rank deadlocked the solve."""
    from impurityModel.test.support import gs_oracle

    comm = MPI.COMM_WORLD
    if comm.rank == 1:
        monkeypatch.setenv("GS_CERTIFY_MAX_SEED", "1")
    h_op, imp, bath, N0, _n_orb, _n_val, _params = gs_oracle.pinned_model(1, 3, False)
    with pytest.raises(config.NonUniformKnobError, match="GS_CERTIFY_MAX_SEED"):
        groundstate.solve_ground_state(h_op, imp, bath, dict(N0), tau=0.05, slaterWeightMin=1e-12, comm=comm, verbose=0)


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="needs two ranks to disagree")
def test_a_bare_sector_solve_raises_on_every_rank_when_a_knob_differs(monkeypatch):
    """H-GS3: solve_sector is public (calc_energy and the DC search's sector solves call it directly)."""
    from impurityModel.ed.eigensolvers import Lowest
    from impurityModel.test.support import gs_oracle

    comm = MPI.COMM_WORLD
    if comm.rank == 1:
        monkeypatch.setenv("GS_CERTIFY_MAX_SEED", "1")
    h_op, imp, bath, N0, _n_orb, _n_val, _params = gs_oracle.pinned_model(1, 3, False)
    with pytest.raises(config.NonUniformKnobError, match="GS_CERTIFY_MAX_SEED"):
        groundstate.solve_sector(
            h_op, imp, bath, dict(N0), tau=0.05, chain_restrict=False, dense_cutoff=1000, manifold=Lowest(),
            slaterWeightMin=1e-12, comm=comm, verbose=False,
        )  # fmt: skip


def test_solve_ground_state_takes_its_rank_from_the_communicator():
    """``rank=`` was a second source of truth for what ``comm.rank`` says: a caller passing the wrong one printed
    the warnings of rank 0 from every rank (or from none). There is one source now."""
    import inspect

    assert "rank" not in inspect.signature(groundstate.solve_ground_state).parameters
