"""``GS_MAX_BLOCK_WIDTH`` (Phase 4, ``doc/plans/dc_smo_performance.md``): capping the warm block
``CIPSISolver.get_eigenvectors`` feeds into the Lanczos solve, independent of how many states
were actually requested (``num_wanted``) or of how wide the manifold a caller warm-starts from
happens to be.

Uses a real (non-interacting, diagonal) Hamiltonian on a basis large enough (252 determinants)
that ``cipsi_solver._size_subspace``'s own basis-size clamp (``cap // width - 1``) never binds
at the small widths this test exercises -- a smaller basis (first attempt: 15 determinants)
made that clamp itself shrink the returned manifold at small widths, an artifact of the toy
fixture rather than of ``GS_MAX_BLOCK_WIDTH``; at production scale (basis sizes of 1e5-1e6) the
same clamp does not bind at any width this knob would realistically be set to either. Forced
through the Krylov/TRLM branch via ``dense_cutoff=1`` rather than a fake kernel, since the
property under test is that truncating the warm block still finds the true ground state (the
cold full-support vector's reachability guarantee), which a fake kernel would not exercise.
"""

import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed import solver_trace
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3, 4, 5]]}, {0: [[6, 7, 8, 9]]})
N_SPIN_ORBITALS = 10
N_OCCUPIED = 5
# Distinct one-body energies: C(10, 5) = 252 five-particle determinants. The ground state
# (the 5 lowest orbitals) sits 0.3 below the second-lowest sum -- checked below -- so it is
# unambiguous even though the full spectrum has incidental ties higher up (irrelevant here).
ORBITAL_ENERGIES = [0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.4, 1.7, 2.05, 2.4]


def _det(occupied):
    """SlaterDeterminant with the given orbitals occupied (MSB-first bit convention)."""
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _make_solver(comm):
    basis = Basis(IMPURITY_ORBITALS, BATH_STATES, nominal_impurity_occ={0: 1}, comm=comm, verbose=False)
    basis.add_states([_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), N_OCCUPIED)])
    return CIPSISolver(basis)


def _diagonal_hop():
    return {((i, "c"), (i, "a")): ORBITAL_ENERGIES[i] for i in range(N_SPIN_ORBITALS)}


def test_ground_state_is_well_separated_from_the_second_lowest():
    """Premise check: the fixture's ground state has a real gap to the next sum, so a test
    asserting the ground energy is found is not vulnerable to a near-degenerate runner-up."""
    sums = sorted(sum(c) for c in itertools.combinations(ORBITAL_ENERGIES, N_OCCUPIED))
    assert sums[1] - sums[0] > 0.1, sums[:2]


def test_gs_max_block_width_caps_the_warm_block_without_losing_the_ground_state(monkeypatch):
    comm = MPI.COMM_WORLD
    solver = _make_solver(comm)
    hop = ManyBodyOperator(_diagonal_hop())
    ground_energy = sum(sorted(ORBITAL_ENERGIES)[:N_OCCUPIED])

    # Cold solve: wide enough that the returned manifold makes truncation meaningful.
    e_ref_cold, psi_refs = solver.get_eigenvectors(hop, 8, dense_cutoff=1, slaterWeightMin=0)
    assert len(psi_refs) >= 4, f"need a wide warm block to truncate, got {len(psi_refs)}"
    np.testing.assert_allclose(min(e_ref_cold), ground_energy, atol=1e-8)

    # Reference: warm-started from that manifold, uncapped (today's default, knob unset).
    e_ref_uncapped, _ = solver.get_eigenvectors(hop, 8, dense_cutoff=1, slaterWeightMin=0, psi_refs=psi_refs)
    np.testing.assert_allclose(min(e_ref_uncapped), ground_energy, atol=1e-8)

    # Capped: same warm-started call, GS_MAX_BLOCK_WIDTH=2 -- narrower than len(psi_refs).
    monkeypatch.setenv("GS_MAX_BLOCK_WIDTH", "2")
    with solver_trace.tracing() as trace:
        e_ref_capped, _ = solver.get_eigenvectors(hop, 8, dense_cutoff=1, slaterWeightMin=0, psi_refs=psi_refs)
    widths = [event["p"] for event in trace.of_kind("eigensolve_block_width")]
    assert widths, "no eigensolve_block_width events traced"
    # 2 warm columns (the cap) + 1 cold full-support column = 3, never len(psi_refs) + 1.
    assert max(widths) <= 3, widths

    # Correctness: the cold-start reachability guard still finds the true ground state despite
    # the narrower warm block -- this is the property GS_MAX_BLOCK_WIDTH's docstring calls a
    # "correctness guard, not an optimization".
    np.testing.assert_allclose(min(e_ref_capped), ground_energy, atol=1e-8)


def test_gs_max_block_width_leaves_the_returned_manifold_unchanged(monkeypatch):
    """The manifold *returned* must not shrink -- only the block the solver runs with does.

    Otherwise a narrower cap would look, to ``expand``'s "exhausted" check, exactly like a
    request that never got the states it certified: a mid-search behavior change this knob's
    docstring explicitly says must not happen. Checked at the most aggressive cap (1: just the
    ground state plus the cold vector) -- if the manifold survives that unchanged, it survives
    every milder cap too.
    """
    comm = MPI.COMM_WORLD
    solver = _make_solver(comm)
    hop = ManyBodyOperator(_diagonal_hop())

    e_ref_cold, psi_refs = solver.get_eigenvectors(hop, 8, dense_cutoff=1, slaterWeightMin=0)

    monkeypatch.setenv("GS_MAX_BLOCK_WIDTH", "1")
    e_ref_capped, psi_refs_capped = solver.get_eigenvectors(
        hop, 8, dense_cutoff=1, slaterWeightMin=0, psi_refs=psi_refs
    )
    assert len(psi_refs_capped) == len(psi_refs), (len(psi_refs_capped), len(psi_refs))
    np.testing.assert_allclose(sorted(e_ref_capped), sorted(e_ref_cold), atol=1e-8)
