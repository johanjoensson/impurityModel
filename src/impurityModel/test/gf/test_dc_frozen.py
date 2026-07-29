"""The frozen-space double-counting sweep: exact where the re-expanding search cannot be.

``H(mu) = H(0) - mu*N_imp`` with ``N_imp`` diagonal, so on a *fixed* determinant space
``E_0(mu)`` is a minimum of affine functions of ``mu`` -- concave, with ``-dE_0/dmu = n(mu)``,
hence ``n(mu)`` non-decreasing. Neither holds for the production observable, which re-selects
both the CIPSI space and the charge sector at every ``mu``; that is exactly why
``_solve_dc_shift`` assumes no monotonicity. Freezing buys those two properties back, and buys
``chi = dn/dmu`` for two cheap eigensolves instead of two full evaluations.

The sharpest of the three is the pure-sector check: when every determinant carries the same
impurity occupation ``N``, the shift is *exactly* ``-mu*N`` and must reproduce to machine
precision. A wrong number operator, a wrong basis ordering or a stale matrix all fail it
immediately, where the softer inequalities would still pass.
"""

import numpy as np
import pytest

from impurityModel.ed.dc_frozen import FrozenSpaceSweep
from impurityModel.ed.groundstate import build_basis_and_solver
from impurityModel.ed.lie_algebra import tensors_to_operator

N_ORB = 8
N_IMP = 4
IMPURITY = {0: [[0, 1]], 1: [[2, 3]]}
BATHS = ({0: [[4, 5]], 1: [[6, 7]]}, {0: [], 1: []})


def _h_op(eps_imp=-0.5, eps_bath=-2.0, v=0.3):
    h = np.zeros((N_ORB, N_ORB), dtype=complex)
    for orb in range(N_IMP):
        h[orb, orb] = eps_imp
    for orb in range(N_IMP, N_ORB):
        h[orb, orb] = eps_bath
    for orb in range(N_IMP):
        partner = N_IMP + (orb % (N_ORB - N_IMP))
        h[orb, partner] += v
        h[partner, orb] += v
    return tensors_to_operator(h)


def _frozen(mixed_valence, *, v=0.3, expand=True, tau=1e-3):
    """A frozen space and its sweep. `mixed_valence=0` keeps the seed a single pure sector."""
    h_op = _h_op(v=v)
    nominal = {0: 1, 1: 1}
    basis, solver = build_basis_and_solver(
        h_op,
        IMPURITY,
        BATHS,
        nominal,
        dict.fromkeys(nominal, mixed_valence),
        tau,
        False,
        False,
        np.inf,
        None,
        False,
        None,
        None,
        1e-12,
        None,
    )
    if expand:
        solver.expand(h_op, dense_cutoff=1000, de2_min=1e-10, slaterWeightMin=1e-12)
    sweep = FrozenSpaceSweep(basis, h_op, list(range(N_IMP)), tau=tau, dense_cutoff=1000, slater_weight_min=1e-12)
    return basis, sweep


def test_the_number_operator_is_diagonal_and_counts_impurity_electrons():
    """The construction everything else rests on: N_imp must be diagonal in this basis."""
    _basis, sweep = _frozen(mixed_valence=1)
    n_dense = sweep._n_matrix.toarray()
    assert np.abs(n_dense - np.diag(np.diag(n_dense))).max() == 0.0
    counts = np.real(np.diag(n_dense))
    assert np.allclose(counts, np.round(counts)), counts
    assert counts.min() >= 0 and counts.max() <= N_IMP


def test_a_pure_sector_shifts_exactly_affinely():
    """The machine-precision oracle. Every determinant carries the same impurity occupation, so
    ``H(mu)`` differs from ``H(0)`` by a constant ``-mu*N`` and the spectrum moves rigidly."""
    _basis, sweep = _frozen(mixed_valence=0, expand=False)
    occupations = np.unique(np.round(np.real(np.diag(sweep._n_matrix.toarray()))))
    assert len(occupations) == 1, f"fixture is not a pure sector: {occupations}"
    n = occupations[0]

    e0 = sweep.energy(0.0)
    for mu in (0.7, -0.4, 2.0):
        assert abs((sweep.energy(mu) - e0) - (-mu * n)) < 1e-12, mu
    # And the occupation cannot move at all, so chi is exactly zero -- the plateau case where
    # delta_mu = delta_n / chi is undefined and must be reported as such rather than as a number.
    assert sweep.occupation(0.0) == pytest.approx(n)
    assert sweep.chi(0.0) == pytest.approx(0.0, abs=1e-12)
    assert sweep.shift_error(0.01, 0.0) is None


def test_energy_is_concave_and_occupation_is_non_decreasing():
    """F3 on a genuinely mixed space -- a min of affine functions is concave, and its slope
    ``-dE/dmu = n`` therefore only ever increases."""
    _basis, sweep = _frozen(mixed_valence=1)
    mus = np.linspace(-1.0, 1.0, 9)
    energies = np.array([sweep.energy(mu) for mu in mus])
    occupations = np.array([sweep.occupation(mu) for mu in mus])

    second_difference = energies[2:] - 2 * energies[1:-1] + energies[:-2]
    assert second_difference.max() <= 1e-12, second_difference
    assert np.diff(occupations).min() >= -1e-12, occupations
    # The space must actually span more than one impurity occupation, or the test is vacuous.
    assert occupations.max() - occupations.min() > 0.5, occupations


def test_hamiltonian_is_a_diagonal_shift_not_a_rebuild():
    """The cheapness claim, structurally: H(mu) - H(0) is -mu*N and nothing else."""
    _basis, sweep = _frozen(mixed_valence=1)
    difference = (sweep.hamiltonian(0.75) - sweep.hamiltonian(0.0)).toarray()
    assert np.abs(difference + 0.75 * sweep._n_matrix.toarray()).max() < 1e-14


def test_chi_matches_the_slope_it_is_supposed_to_measure():
    _basis, sweep = _frozen(mixed_valence=1)
    mu, step = 0.3, 1e-3
    expected = (sweep.occupation(mu + step) - sweep.occupation(mu - step)) / (2 * step)
    assert sweep.chi(mu, step) == pytest.approx(expected)
    if sweep.chi(mu, step) != 0:
        assert sweep.shift_error(0.01, mu, step) == pytest.approx(0.01 / sweep.chi(mu, step))


def test_a_matrix_that_outlived_its_basis_is_rejected():
    """A stale prebuilt matrix yields eigenvalues of the wrong operator rather than an error,
    so get_eigenvectors checks the shape it was handed."""
    from impurityModel.ed.cipsi_solver import CIPSISolver

    basis, sweep = _frozen(mixed_valence=1)
    solver = CIPSISolver(basis)
    too_small = sweep.hamiltonian(0.0)[:-1, :-1]
    with pytest.raises(ValueError, match="outlived its basis|built for a basis"):
        solver.get_eigenvectors(sweep._h_op, num_wanted=2, dense_cutoff=1000, slaterWeightMin=1e-12, h_matrix=too_small)


def test_the_prebuilt_matrix_path_agrees_with_the_operator_path():
    """Passing the matrix must not change the answer -- it is the same operator, built once."""
    from impurityModel.ed.cipsi_solver import CIPSISolver

    basis, sweep = _frozen(mixed_valence=1)
    solver = CIPSISolver(basis)
    common = dict(num_wanted=6, max_energy=None, dense_cutoff=1000, slaterWeightMin=1e-12, solver="irlm")
    from_operator, _ = solver.get_eigenvectors(sweep._h_op, **common)
    from_matrix, _ = solver.get_eigenvectors(sweep._h_op, h_matrix=sweep.hamiltonian(0.0), **common)
    assert np.allclose(np.sort(from_operator), np.sort(from_matrix), atol=1e-10)


def test_occupation_interiority_is_reported():
    """The frozen space bounds the reachable occupation whatever the double counting does; a
    result sitting on that boundary was pinned by the basis window, not by the search."""
    _basis, sweep = _frozen(mixed_valence=1)
    assert isinstance(sweep.occupation_is_interior(0.0), bool)
    # Driven hard enough, the occupation must saturate against the window and stop being interior.
    assert not sweep.occupation_is_interior(1e3)
