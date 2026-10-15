"""The occupation walk hands back the winning sector it already solved.

``find_ground_state_basis`` used to end by clearing its caches and calling ``calc_energy`` at the
winning occupation again -- the whole expansion and eigensolve a second time, from a cold start.
It now returns the best trial's ``(basis, solver)``: no sector is solved twice, and the solver
(attached as ``basis.walk_solver``) holds an eigensolve that is valid for the returned basis, so the
refinement continues warm.
"""

import numpy as np

from impurityModel.ed import groundstate
from impurityModel.ed.groundstate import find_ground_state_basis, sector_key
from impurityModel.ed.lie_algebra import tensors_to_operator

N_ORB = 12


def _model():
    """A 2-group impurity with one valence and one conduction bath pair per group."""
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    impurity_orbitals = {0: [[0, 1]], 1: [[2, 3]]}
    valence = {0: [[4, 5]], 1: [[6, 7]]}
    conduction = {0: [[8, 9]], 1: [[10, 11]]}
    for eps, imp in ((-0.6, (0, 1)), (0.4, (2, 3))):
        for orb in imp:
            h[orb, orb] = eps
    for orb in (4, 5, 6, 7):
        h[orb, orb] = -2.0
    for orb in (8, 9, 10, 11):
        h[orb, orb] = 2.0
    for imp, val, con in ((0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)):
        for other in (val, con):
            h[imp, other] = h[other, imp] = 0.35
    return tensors_to_operator(h), impurity_orbitals, (valence, conduction)


def test_no_sector_is_solved_twice_and_the_winner_comes_with_a_valid_solve(monkeypatch):
    solved = []
    real = groundstate.calc_energy

    def recording(h_op, impurity_orbitals, bath_states, N0, *args, **kwargs):
        solved.append(sector_key(N0))
        return real(h_op, impurity_orbitals, bath_states, N0, *args, **kwargs)

    monkeypatch.setattr(groundstate, "calc_energy", recording)
    h_op, impurity_orbitals, bath_states = _model()
    basis = find_ground_state_basis(
        h_op,
        impurity_orbitals,
        bath_states,
        {0: 1, 1: 1},
        tau=1e-3,
        dense_cutoff=1,
        comm=None,
        verbose=False,
        truncation_threshold=np.inf,
        slaterWeightMin=1e-12,
    )

    assert solved and len(solved) == len(set(solved)), f"a sector was solved twice: {solved}"
    solver = basis.walk_solver
    assert solver is not None and solver.basis is basis
    assert solver.eig is not None and solver.eig.generation == basis.generation
    assert solver.eig.certified
