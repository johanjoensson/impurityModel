"""A failure inside a rank-0 block that precedes a broadcast is raised on EVERY rank, not left as a hang.

Each site below computes a result on rank 0 and broadcasts it. An exception on rank 0 used to unwind rank 0 alone
while the other ranks waited in the broadcast forever (probe: mpiexec exit 124 after the timeout).
"""

import pytest
from mpi4py import MPI

from impurityModel.ed import groundstate, gs_connectivity, orbits, sector_solver, seed_sectors
from impurityModel.ed.eigensolvers import Lowest
from impurityModel.ed.groundstate import build_basis_and_solver
from impurityModel.test.support.cubic_d_shell import cubic_d_shell

pytestmark = pytest.mark.mpi


def _boom(*_a, **_k):
    # Raised only on rank 0: the other ranks never reach a patched callee that sits inside a rank-0 block.
    raise RuntimeError("injected on rank 0")


def _seed_basis(comm):
    _h0, H, imp, bath = cubic_d_shell(n_bath_sets=1)
    basis, _ = build_basis_and_solver(
        H,
        imp,
        bath,
        {0: 8},
        tau=0.0,
        chain_restrict=False,
        truncation_threshold=None,
        comm=comm,
        verbose=False,
        frozen_occupations=None,
        weighted_restrictions=None,
        slaterWeightMin=0,
        manifold=Lowest(),
    )
    return H, imp, bath, basis


@pytest.mark.parametrize("site", ["connectivity", "seed_plan", "certification_plan", "orbits", "widen"])
def test_a_rank0_failure_is_raised_on_every_rank(monkeypatch, site):
    comm = MPI.COMM_WORLD
    H, imp, bath, basis = _seed_basis(comm)
    if site == "connectivity":
        monkeypatch.setattr(gs_connectivity, "_probe", _boom)
        call = lambda: gs_connectivity.measure_gs_connectivity(H, imp, bath, {0: 8}, comm)  # noqa: E731
    elif site == "seed_plan":
        monkeypatch.setattr(seed_sectors, "_build_classes", _boom)
        call = lambda: seed_sectors.seed_class_plan(H, basis, max_seed=10**6)  # noqa: E731
    elif site == "certification_plan":
        monkeypatch.setattr(sector_solver, "conserves_sz", _boom)
        call = lambda: sector_solver.certification_plan(H, basis, max_seed=10**6)  # noqa: E731
    elif site == "orbits":
        monkeypatch.setattr(orbits, "resolve_spin_layout", _boom)
        call = lambda: orbits.build_orbits(H, basis)  # noqa: E731
    else:
        monkeypatch.setattr(groundstate, "tensors_to_operator", _boom)
        call = lambda: groundstate.widen_by_single_excitations(basis, max_seed=10**6)  # noqa: E731
    with pytest.raises(RuntimeError):
        call()
    # Every rank got here: nobody is left in a broadcast that rank 0 abandoned.
    assert comm.allreduce(1, op=MPI.SUM) == comm.size
