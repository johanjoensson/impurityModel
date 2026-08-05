"""The frozen-space double-counting sweep: exact where the re-expanding search cannot be.

``H(mu) = H(0) - mu*N_imp`` with ``N_imp`` diagonal, so on a *fixed* determinant space
``E_0(mu)`` is a minimum of affine functions of ``mu`` -- concave, with ``-dE_0/dmu = n(mu)``,
hence ``n(mu)`` non-decreasing. Neither holds for the production observable, which re-selects
both the CIPSI space and the charge sector at every ``mu`` -- which is why ``_solve_dc_shift``
brackets on the integer *sector* (monotone even there, because the winning sector is the argmin of
a family of affine functions) and only assumes a smooth residual inside one. Freezing buys the
two properties back outright, and buys ``chi = dn/dmu`` for two cheap eigensolves instead of two
full evaluations.

The sharpest of the three is the pure-sector check: when every determinant carries the same
impurity occupation ``N``, the shift is *exactly* ``-mu*N`` and must reproduce to machine
precision. A wrong number operator, a wrong basis ordering or a stale matrix all fail it
immediately, where the softer inequalities would still pass.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.dc_criteria import build_union_space
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


def _frozen(sector_radius, *, v=0.3, expand=True, tau=1e-3):
    """A frozen space and its sweep. `sector_radius=0` keeps the seed a single pure sector.

    The sweep exists to vary the charge -- ``E(mu)`` concave, ``n(mu) = -dE/dmu``
    non-decreasing -- so it needs a basis spanning several *total* charge sectors. That is
    ``total_charge_slack``, not ``mixed_valence``: the latter fluctuates the impurity charge
    against the bath at fixed total, which leaves ``n`` pinned and the sweep vacuous.
    """
    h_op = _h_op(v=v)
    nominal = {0: 1, 1: 1}
    basis, solver = build_basis_and_solver(
        h_op,
        IMPURITY,
        BATHS,
        nominal,
        dict.fromkeys(nominal, 0),
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
        total_charge_slack=sector_radius,
    )
    if expand:
        solver.expand(h_op, dense_cutoff=1000, de2_min=1e-10, slaterWeightMin=1e-12)
    sweep = FrozenSpaceSweep(basis, h_op, list(range(N_IMP)), tau=tau, dense_cutoff=1000, slater_weight_min=1e-12)
    return basis, sweep


def test_the_number_operator_is_diagonal_and_counts_impurity_electrons():
    """The construction everything else rests on: N_imp must be diagonal in this basis."""
    _basis, sweep = _frozen(sector_radius=1)
    n_dense = sweep._n_matrix.toarray()
    assert np.abs(n_dense - np.diag(np.diag(n_dense))).max() == 0.0
    counts = np.real(np.diag(n_dense))
    assert np.allclose(counts, np.round(counts)), counts
    assert counts.min() >= 0 and counts.max() <= N_IMP


def test_a_pure_sector_shifts_exactly_affinely():
    """The machine-precision oracle. Every determinant carries the same impurity occupation, so
    ``H(mu)`` differs from ``H(0)`` by a constant ``-mu*N`` and the spectrum moves rigidly."""
    _basis, sweep = _frozen(sector_radius=0, expand=False)
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
    _basis, sweep = _frozen(sector_radius=1)
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
    _basis, sweep = _frozen(sector_radius=1)
    difference = (sweep.hamiltonian(0.75) - sweep.hamiltonian(0.0)).toarray()
    assert np.abs(difference + 0.75 * sweep._n_matrix.toarray()).max() < 1e-14


def test_chi_matches_the_slope_it_is_supposed_to_measure():
    _basis, sweep = _frozen(sector_radius=1)
    mu, step = 0.3, 1e-3
    expected = (sweep.occupation(mu + step) - sweep.occupation(mu - step)) / (2 * step)
    assert sweep.chi(mu, step) == pytest.approx(expected)
    if sweep.chi(mu, step) != 0:
        assert sweep.shift_error(0.01, mu, step) == pytest.approx(0.01 / sweep.chi(mu, step))


def test_chi_rejects_a_straddle_near_a_sector_boundary():
    """R3: the CHI_STEP warning's own measured numbers, reproduced as a test.

    ``mu = -0.5225`` on this exact fixture (``sector_radius=1``) sits right at the sector
    boundary where occupation jumps from 1.0 to ~3.0. The plain central difference there reads
    126.30 / 252.26 / 468.23 / 617.31 / 636.47 as the step halves from 8e-3 -- the ``1/h``
    signature of a straddle, not a slope -- while the same sweep at ``mu = 0.3`` is step-size
    independent to four significant figures. ``chi`` must reject the former and accept the
    latter.
    """
    _basis, sweep = _frozen(sector_radius=1)

    straddle_mu, smooth_mu = -0.5225, 0.3
    # Reproduce the module docstring's own numbers first, so a fixture change surfaces as a
    # readable mismatch here rather than a mystery `None` below.
    raw = [sweep._central_difference(straddle_mu, h) for h in (8e-3, 4e-3, 2e-3, 1e-3)]
    np.testing.assert_allclose(raw, [126.30, 252.26, 468.23, 617.31], rtol=2e-3)

    assert sweep.chi(straddle_mu, step=8e-3) is None
    assert sweep.shift_error(0.01, straddle_mu, step=8e-3) is None
    assert sweep.chi(smooth_mu, step=8e-3) == pytest.approx(sweep.chi(smooth_mu, step=1e-3), rel=1e-3)


def test_a_matrix_that_outlived_its_basis_is_rejected():
    """A stale prebuilt matrix yields eigenvalues of the wrong operator rather than an error,
    so get_eigenvectors checks the shape it was handed."""
    from impurityModel.ed.cipsi_solver import CIPSISolver

    basis, sweep = _frozen(sector_radius=1)
    solver = CIPSISolver(basis)
    too_small = sweep.hamiltonian(0.0)[:-1, :-1]
    with pytest.raises(ValueError, match="outlived its basis|built for a basis"):
        solver.get_eigenvectors(sweep._h_op, num_wanted=2, dense_cutoff=1000, slaterWeightMin=1e-12, h_matrix=too_small)


def test_the_prebuilt_matrix_path_agrees_with_the_operator_path():
    """Passing the matrix must not change the answer -- it is the same operator, built once."""
    from impurityModel.ed.cipsi_solver import CIPSISolver

    basis, sweep = _frozen(sector_radius=1)
    solver = CIPSISolver(basis)
    common = dict(num_wanted=6, max_energy=None, dense_cutoff=1000, slaterWeightMin=1e-12, solver="irlm")
    from_operator, _ = solver.get_eigenvectors(sweep._h_op, **common)
    from_matrix, _ = solver.get_eigenvectors(sweep._h_op, h_matrix=sweep.hamiltonian(0.0), **common)
    assert np.allclose(np.sort(from_operator), np.sort(from_matrix), atol=1e-10)


def test_occupation_interiority_is_reported():
    """The frozen space bounds the reachable occupation whatever the double counting does; a
    result sitting on that boundary was pinned by the basis window, not by the search."""
    _basis, sweep = _frozen(sector_radius=1)
    assert isinstance(sweep.occupation_is_interior(0.0), bool)
    # Driven hard enough, the occupation must saturate against the window and stop being interior.
    assert not sweep.occupation_is_interior(1e3)


def test_the_frozen_space_reports_which_sectors_it_can_represent():
    """A sweep is only meaningful where its space can represent the answer.

    Which sectors a seeded space spans is incidental -- it is however far the CIPSI expansion
    happened to reach with an unconstrained impurity window -- so it has to be reported rather
    than assumed. Measured on a real NiO workload the seeded space spans n_imp = 8, 9, 10 against
    a nominal 8: three sectors, but only upward.
    """
    _basis, sweep = _frozen(sector_radius=1)
    span = sweep.sector_span()
    assert span == tuple(sorted(span)) and len(span) >= 2, span
    assert all(isinstance(n, int) for n in span), span
    assert sweep.spans_sector(span[0]) and sweep.spans_sector(span[-1])
    assert not sweep.spans_sector(span[-1] + 1)
    assert not sweep.spans_sector(span[0] - 1)


def test_a_single_sector_space_reports_that_it_cannot_cross():
    """The failure mode, made visible instead of silent.

    A space confined to one sector answers every mu with that sector's occupation -- on a toy it
    missed the true observable by a full electron across a charge-sector boundary, while looking
    perfectly converged. spans_sector is what a driver checks before trusting a Newton step.
    """
    _basis, sweep = _frozen(sector_radius=0, expand=False)
    span = sweep.sector_span()
    assert len(span) == 1, span
    assert not sweep.spans_sector(span[0] + 1)
    assert not sweep.spans_sector(span[0] - 1)


CONDUCTION_BATHS = ({0: [[4, 5]], 1: [[6]]}, {0: [[7]], 1: []})


def test_the_sector_radius_widens_the_span_symmetrically():
    """The union space needs no new generation code.

    generate_initial_basis already filters cross-group combinations on the *total* impurity
    charge, and total_slack is the max |mixed_valence| over unfrozen groups -- so the sector
    radius is just mixed_valence, and it opens the window on both sides.
    """
    spans = {}
    for radius in (0, 1, 2):
        sweep, _basis = build_union_space(
            _h_op(),
            IMPURITY,
            BATHS,
            {0: 1, 1: 1},
            sector_radius=radius,
            tau=1e-3,
            truncation_threshold=np.inf,
            slater_weight_min=1e-12,
            dense_cutoff=1000,
        )
        spans[radius] = sweep.sector_span()
    # Each radius contains the previous one, and the low edge moves down with the radius --
    # which is the half the seeded space cannot reach on its own (see the test below).
    assert set(spans[0]) < set(spans[1]) < set(spans[2]), spans
    assert spans[1][0] < spans[0][0] and spans[2][0] < spans[1][0], spans


def test_a_valence_only_bath_makes_the_seeded_span_upward_only():
    """The mechanism behind the asymmetry measured on NiO, isolated.

    With no conduction orbitals an impurity electron has nowhere to go, so CIPSI excitations can
    only *add* impurity charge: the expansion widens the span upward and never downward. That is
    why a seeded frozen space on a valence-only fit -- the typical coarse fit, and what the NiO
    archive has -- is structurally blind to lower charge states, and why only the generation
    window can open them.
    """
    common = dict(
        sector_radius=0,
        mu_samples=(0.0,),
        tau=1e-3,
        truncation_threshold=np.inf,
        slater_weight_min=1e-12,
        de2_min=1e-10,
        dense_cutoff=1000,
    )
    valence_only, _b1 = build_union_space(_h_op(), IMPURITY, BATHS, {0: 1, 1: 1}, **common)
    with_conduction, _b2 = build_union_space(_h_op(), IMPURITY, CONDUCTION_BATHS, {0: 1, 1: 1}, **common)
    nominal_total = 2
    assert min(valence_only.sector_span()) == nominal_total, valence_only.sector_span()
    assert min(with_conduction.sector_span()) < nominal_total, with_conduction.sector_span()


def test_the_union_answers_where_the_seeded_space_is_blind():
    """The point of building the span deliberately, in the quantity the search controls."""
    common = dict(tau=1e-3, truncation_threshold=np.inf, slater_weight_min=1e-12, de2_min=1e-10, dense_cutoff=1000)
    seed, _b1 = build_union_space(_h_op(), IMPURITY, BATHS, {0: 1, 1: 1}, sector_radius=0, mu_samples=(0.0,), **common)
    union, _b2 = build_union_space(
        _h_op(), IMPURITY, BATHS, {0: 1, 1: 1}, sector_radius=1, mu_samples=(-2.0, 0.0, 2.0), **common
    )
    assert min(union.sector_span()) < min(seed.sector_span())
    # Deep on the side the seed cannot represent, the two disagree by about a full electron --
    # the seed reports its lowest reachable sector as though it were converged.
    assert abs(union.occupation(-2.0) - seed.occupation(-2.0)) > 0.5


def test_expanding_at_more_shifts_only_grows_the_space():
    """Growth-only: S_{k+1} = S_k union new, which is what makes the construction terminate.

    Re-freezing on drift is an uncontracted fixed-point iteration and can limit-cycle, because
    selection under a cap is discontinuous in mu. Accumulating cannot: the space is finite and
    only ever grows (up to a binding truncation_threshold, which prunes -- hence np.inf here).
    """
    common = dict(
        sector_radius=1,
        tau=1e-3,
        truncation_threshold=np.inf,
        slater_weight_min=1e-12,
        de2_min=1e-10,
        dense_cutoff=1000,
    )
    one, _b1 = build_union_space(_h_op(), IMPURITY, BATHS, {0: 1, 1: 1}, mu_samples=(0.0,), **common)
    three, _b3 = build_union_space(_h_op(), IMPURITY, BATHS, {0: 1, 1: 1}, mu_samples=(0.0, -2.0, 2.0), **common)
    assert len(three.basis) >= len(one.basis)
    assert set(three.sector_span()) >= set(one.sector_span())


@pytest.mark.mpi
def test_sector_span_is_identical_on_every_rank():
    """The gap that let a reproduced deadlock through, closed.

    `build_sparse_matrix` returns a *globally shaped* matrix in which only this rank's columns are
    populated, so reading the diagonal locally sees the owned determinants plus a spurious 0
    everywhere else. Measured before the fix, at -n 3 on one 18-determinant space: spans (0, 2),
    (0, 3) and (0, 1, 4) on the three ranks, and `spans_sector(4)` returning False/False/True --
    a rank-local boolean that a caller then used to choose between two different sequences of
    collectives. The whole `-n 2`/`-n 3` gate passed throughout, because no test built this class
    with a live communicator: every fixture here passes `comm=None`, so under `mpiexec` each rank
    merely repeated the serial computation.
    """
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("needs at least 2 ranks")
    h_op = _h_op()
    nominal = {0: 1, 1: 1}
    basis, solver = build_basis_and_solver(
        h_op,
        IMPURITY,
        BATHS,
        nominal,
        dict.fromkeys(nominal, 1),
        1e-3,
        False,
        False,
        np.inf,
        comm,
        False,
        None,
        None,
        1e-12,
        None,
    )
    solver.expand(h_op, dense_cutoff=1000, de2_min=1e-10, slaterWeightMin=1e-12)
    sweep = FrozenSpaceSweep(basis, h_op, list(range(N_IMP)), tau=1e-3, dense_cutoff=1000, slater_weight_min=1e-12)

    assert basis.is_distributed, "fixture must be distributed or this proves nothing"
    spans = comm.allgather(sweep.sector_span())
    assert len(set(spans)) == 1, spans
    # The spurious 0 is what made the downward guard dead; it must not be in the span unless a
    # determinant really is empty on the impurity.
    assert 0 not in sweep.sector_span() or min(spans[0]) == 0
    # And every derived decision must agree, since callers branch on it.
    for n in range(N_IMP + 1):
        verdicts = comm.allgather(sweep.spans_sector(n))
        assert len(set(verdicts)) == 1, (n, verdicts)
    occupations = comm.allgather(sweep.occupation(0.0))
    assert max(occupations) - min(occupations) < 1e-9, occupations
