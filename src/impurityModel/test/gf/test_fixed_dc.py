"""
Tests for the fixed-peak and fixed-occupation double counting criteria.

Analytically solvable model: two impurity spin-orbitals at energy eps with a
Hubbard interaction U, weakly coupled (hopping v) to two valence bath
spin-orbitals at energy eps_b. With the double counting dc entering the
Hamiltonian as -dc * n_imp:

    E[N_imp = 0] = 2 eps_b
    E[N_imp = 1] = (eps - dc) + 2 eps_b          + O(v^2)
    E[N_imp = 2] = 2 (eps - dc) + U + 2 eps_b    + O(v^2)

so the electron-addition peak sits at E[2] - E[1] = eps + U - dc and the
electron-removal peak at E[1] - E[0] = eps - dc. The total electron number is
conserved (N_imp + N_bath = 3 with N0 = 1), and the impurity occupation
switches from 1 to 2 through charge transfer when eps - dc + U < eps_b, i.e.
for dc > eps + U - eps_b = 6.
"""

from dataclasses import replace

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.model import BasisOptions, ImpurityModel, SolverOptions
from impurityModel.ed.selfenergy import fixed_occupation_dc, fixed_peak_dc

EPS = -1.0
U = 3.0
EPS_B = -4.0


def build_model(v, dc_scale):
    h0 = np.zeros((4, 4), dtype=complex)
    for s in range(2):
        imp, bath = s, 2 + s
        h0[imp, imp] = EPS
        h0[bath, bath] = EPS_B
        h0[imp, bath] = v
        h0[bath, imp] = v
    dc = np.identity(2, dtype=complex) * dc_scale
    u4 = np.zeros((4, 4, 4, 4), dtype=complex)
    # RSPt convention <ij|V|kl> with pairs (i,k),(j,l): the density-density
    # element U n_0 n_1 sits at u4[0,1,0,1] (and its exchange-symmetric partner).
    u4[0, 1, 0, 1] = U
    u4[1, 0, 1, 0] = U
    return h0, dc, u4


def common_kwargs(v, tau, dc_scale=0.5):
    """``fixed_peak_dc``/``fixed_occupation_dc`` kwargs plus the dense dc guess used to build them.

    Returns ``(kwargs, dc_guess)``: ``kwargs`` is exactly the ``model``/``basis``/``solver``/``comm``
    the search functions accept (no extra keys), ``dc_guess`` is the dense ``model.dc`` matrix for
    the tests' own assertions (:func:`assert_uniform_shift`).
    """
    h0, dc, u4 = build_model(v, dc_scale)
    model = ImpurityModel.from_solver_matrix(
        h0,
        dc.shape[0],
        dc=dc,
        u4=u4,
        rot_to_spherical=np.eye(2, dtype=complex),
        bath_valence_conduction=([2, 3], []),
    )
    basis = BasisOptions(
        nominal_occ={0: 1},
        mixed_valence=None,
        spin_flip_dj=False,
        tau=tau,
        slater_weight_min=np.sqrt(np.finfo(float).eps),
        truncation_threshold=int(1e8),
    )
    solver = SolverOptions(dense_cutoff=1000)
    return dict(model=model, basis=basis, solver=solver, comm=MPI.COMM_WORLD), dc


def assert_uniform_shift(dc, dc_guess):
    """The result must be dc_guess plus a real uniform shift."""
    shift = dc - dc_guess
    assert np.allclose(shift, shift[0, 0] * np.identity(2)), shift
    assert abs(shift[0, 0].imag) < 1e-12


def test_fixed_peak_dc_addition_peak():
    target = 1.2
    kwargs, dc_guess = common_kwargs(v=0.01, tau=1e-3)
    dc = fixed_peak_dc(peak_position=target, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    # E[2] - E[1] = eps + U - dc = target
    expected = EPS + U - target
    assert np.allclose(np.diag(dc).real, expected, atol=5e-3), dc


def test_fixed_peak_dc_removal_peak():
    # A negative peak position must exercise the removal branch, E[1] - E[0]
    target = -1.5
    kwargs, dc_guess = common_kwargs(v=0.01, tau=1e-3, dc_scale=0.2)
    dc = fixed_peak_dc(peak_position=target, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    # E[1] - E[0] = eps - dc = target
    expected = EPS - target
    assert np.allclose(np.diag(dc).real, expected, atol=5e-3), dc


def test_fixed_peak_dc_multiple_groups_raises():
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3, dc_scale=1.0)
    kwargs["basis"] = replace(kwargs["basis"], nominal_occ={0: 1, 1: 1})
    with pytest.raises(ValueError, match="single impurity group"):
        fixed_peak_dc(peak_position=1.0, **kwargs)


def test_fixed_occupation_dc_already_converged():
    # At the guess (dc=0.5, well below the dc=6 charge-transfer point) the impurity already
    # holds one electron; requesting occupation 1 must return the guess unchanged (mu=0).
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)
    np.testing.assert_allclose(dc, dc_guess, atol=1e-8)


def test_fixed_occupation_dc_increases_occupation():
    # Requesting two electrons on the impurity requires pushing the doubly
    # occupied impurity below the bath: dc > eps + U - eps_b = 6.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    dc = fixed_occupation_dc(occupation=2.0, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    assert dc[0, 0].real > 6.0, dc


def test_fixed_occupation_dc_decreases_occupation():
    # A guess of 7 puts two electrons on the impurity; requesting one electron
    # must bring the double counting back below the charge-transfer point.
    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2, dc_scale=7.0)
    dc = fixed_occupation_dc(occupation=1.0, **kwargs)
    assert_uniform_shift(dc, dc_guess)
    assert dc[0, 0].real < 6.0, dc


def test_fixed_occupation_dc_unreachable_raises():
    # Only two bath spin-orbitals and three electrons in total: the impurity
    # occupation cannot drop below one.
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2, dc_scale=0.5)
    with pytest.raises(RuntimeError, match="Could not bracket"):
        fixed_occupation_dc(occupation=0.2, **kwargs)


def test_noninteracting_impurity_occupation_matches_fermi_fill():
    # The h_loc-derived target is the Fermi-filled (mu=0) occupation of the full
    # non-interacting h0. Build a 1-impurity / 1-bath cluster with an impurity
    # level poking above the Fermi level so the answer is genuinely fractional,
    # and compare against an independent per-eigenvector Fermi sum.
    from impurityModel.ed.double_counting import _noninteracting_impurity_occupation

    e_imp, e_bath, v, tau = 0.5, -2.0, 0.5, 0.1
    h0 = {
        ((0, "c"), (0, "a")): e_imp,
        ((1, "c"), (1, "a")): e_bath,
        ((0, "c"), (1, "a")): v,
        ((1, "c"), (0, "a")): v,
    }
    n = _noninteracting_impurity_occupation(h0, None, impurity_indices=[0], n_spin_orbitals=2, tau=tau)

    h = np.array([[e_imp, v], [v, e_bath]], dtype=complex)
    energies, vecs = np.linalg.eigh(h)
    f = 1.0 / (1.0 + np.exp(energies / tau))
    expected = float(np.sum(f * np.abs(vecs[0, :]) ** 2))  # <imp| sum_n f_n |v_n><v_n| |imp>
    assert 0.0 < expected < 1.0  # genuinely fractional, not a plateau boundary
    assert np.isclose(n, expected, atol=1e-12), (n, expected)


def test_fixed_occupation_dc_self_consistent_pins_fermi_level():
    # Omitting `occupation` now solves the self-consistent N(mu) = N0(mu) criterion (not a
    # target derived once from the guess). For this dimer, both the interacting N_imp=1->0
    # charge-transfer threshold (E[1]<E[0] iff dc>eps, i.e. dc=eps=-1) and the non-interacting
    # h_loc's resonance (the bare impurity level eps-dc crosses the Fermi level at dc=eps too)
    # sit at the same leading-order point -- so the nearest self-consistent root to the mu=0
    # guess is expected close to dc=eps=-1, not at the trivial guess (dc=0.5) or the far
    # dc>6 charge-transfer family (which the geometric scan, growing outward from mu=0, never
    # even reaches once the near root is found and refined).
    from impurityModel.ed.double_counting import (
        _dc_operator,
        _lowest_energy_and_thermal_rho,
        _noninteracting_impurity_occupation,
        _normalize_dc_orbitals,
        _prepare_dc_solver,
    )
    from impurityModel.ed.lie_algebra import tensors_to_operator
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    kwargs, dc_guess = common_kwargs(v=0.3, tau=1e-2)
    model, basis, solver = kwargs["model"], kwargs["basis"], kwargs["solver"]
    dc_auto = fixed_occupation_dc(**kwargs)
    assert_uniform_shift(dc_auto, dc_guess)
    assert -2.0 < dc_auto[0, 0].real < 0.0, dc_auto

    # Independently verify self-consistency AT the returned dc -- not by re-invoking the search
    # (n(mu) is nearly flat here, so an explicit-occupation cross-check can't discriminate mu
    # precisely) -- by directly recomputing N(dc_auto) [interacting] and N0(dc_auto)
    # [non-interacting] with the same low-level primitives fixed_occupation_dc itself uses, and
    # checking they agree within occ_tol (the search's default convergence tolerance).
    n0 = _noninteracting_impurity_occupation(model.h0, tensors_to_operator(dc_auto).to_dict(), [0, 1], 4, basis.tau)

    impurity_orbitals, bath_states = _normalize_dc_orbitals(model.impurity_orbitals, model.bath_states)
    h_op_i = ManyBodyOperator(model.h0) + ManyBodyOperator(model.u4)
    mb_basis, mb_solver = _prepare_dc_solver(
        h_op_i,
        impurity_orbitals,
        bath_states,
        basis.nominal_occ,
        basis.mixed_valence,
        basis.truncation_threshold,
        basis.spin_flip_dj,
        basis.tau,
        False,
    )
    h_op = h_op_i - _dc_operator(dc_auto)
    mb_solver.expand(h_op, dense_cutoff=solver.dense_cutoff, de2_min=1e-5, slaterWeightMin=basis.slater_weight_min)
    energy_cut = -basis.tau * np.log(1e-4)
    _, rho = _lowest_energy_and_thermal_rho(
        mb_basis, mb_solver, h_op, [0, 1], energy_cut, solver.dense_cutoff, basis.slater_weight_min
    )
    n = float(np.real(np.trace(rho)))
    assert np.isclose(n, n0, atol=1e-2), (n, n0)  # default occ_tol


@pytest.mark.mpi
def test_fixed_peak_dc_ranks_agree():
    # The Newton loop in fixed_peak_dc branches on Lanczos energies, which are
    # only replicated to roundoff across ranks. Every rank must nevertheless run
    # the same iterations and return an identical dc; a per-rank divergence
    # would deadlock on the next collective solve instead of returning here.
    comm = MPI.COMM_WORLD
    kwargs, _ = common_kwargs(v=0.01, tau=1e-3)
    dc = fixed_peak_dc(peak_position=1.2, **kwargs)
    gathered = comm.gather(dc, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert np.array_equal(dc, other), (dc, other)


@pytest.mark.mpi
def test_fixed_occupation_dc_ranks_agree():
    # Occupation control keys off the Allreduced density matrix, so agreement is
    # by construction; guard it against regressions all the same.
    comm = MPI.COMM_WORLD
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    dc = fixed_occupation_dc(occupation=2.0, **kwargs)
    gathered = comm.gather(dc, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert np.array_equal(dc, other), (dc, other)


@pytest.mark.mpi
def test_fixed_occupation_dc_self_consistent_ranks_agree():
    # The self-consistent search evaluates two observables (interacting N, non-interacting N0)
    # per trial mu; both must stay rank-invariant for the same reason as the explicit path.
    comm = MPI.COMM_WORLD
    kwargs, _ = common_kwargs(v=0.3, tau=1e-2)
    dc = fixed_occupation_dc(**kwargs)
    gathered = comm.gather(dc, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert np.array_equal(dc, other), (dc, other)
