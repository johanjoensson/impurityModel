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


def build_model(v):
    h0 = {}
    for s in range(2):
        imp, bath = s, 2 + s
        h0[((imp, "c"), (imp, "a"))] = EPS
        h0[((bath, "c"), (bath, "a"))] = EPS_B
        h0[((imp, "c"), (bath, "a"))] = v
        h0[((bath, "c"), (imp, "a"))] = v
    u4 = np.zeros((4, 4, 4, 4), dtype=complex)
    # RSPt convention <ij|V|kl> with pairs (i,k),(j,l): the density-density
    # element U n_0 n_1 sits at u4[0,1,0,1] (and its exchange-symmetric partner).
    u4[0, 1, 0, 1] = U
    u4[1, 0, 1, 0] = U
    return h0, u4


def common_kwargs(v, tau):
    h0, u4 = build_model(v)
    model = ImpurityModel(
        h0=h0,
        u4=u4,
        impurity_orbitals={0: [0, 1]},
        rot_to_spherical=np.eye(2, dtype=complex),
        bath_states=({0: [2, 3]}, {0: []}),
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
    return dict(model=model, basis=basis, solver=solver, comm=MPI.COMM_WORLD)


def assert_uniform_shift(dc, dc_guess):
    """The result must be dc_guess plus a real uniform shift."""
    shift = dc - dc_guess
    assert np.allclose(shift, shift[0, 0] * np.identity(2)), shift
    assert abs(shift[0, 0].imag) < 1e-12


def test_fixed_peak_dc_addition_peak():
    target = 1.2
    dc_guess = 0.5 * np.identity(2, dtype=complex)
    dc = fixed_peak_dc(peak_position=target, dc_guess=dc_guess, **common_kwargs(v=0.01, tau=1e-3))
    assert_uniform_shift(dc, dc_guess)
    # E[2] - E[1] = eps + U - dc = target
    expected = EPS + U - target
    assert np.allclose(np.diag(dc).real, expected, atol=5e-3), dc


def test_fixed_peak_dc_removal_peak():
    # A negative peak position must exercise the removal branch, E[1] - E[0]
    target = -1.5
    dc_guess = 0.2 * np.identity(2, dtype=complex)
    dc = fixed_peak_dc(peak_position=target, dc_guess=dc_guess, **common_kwargs(v=0.01, tau=1e-3))
    assert_uniform_shift(dc, dc_guess)
    # E[1] - E[0] = eps - dc = target
    expected = EPS - target
    assert np.allclose(np.diag(dc).real, expected, atol=5e-3), dc


def test_fixed_peak_dc_multiple_groups_raises():
    kwargs = common_kwargs(v=0.01, tau=1e-3)
    kwargs["basis"] = replace(kwargs["basis"], nominal_occ={0: 1, 1: 1})
    with pytest.raises(ValueError, match="single impurity group"):
        fixed_peak_dc(peak_position=1.0, dc_guess=np.identity(2, dtype=complex), **kwargs)


def test_fixed_occupation_dc_already_converged():
    # At the guess the impurity holds one electron; requesting occupation 1
    # must return the guess unchanged.
    dc_guess = 0.5 * np.identity(2, dtype=complex)
    dc = fixed_occupation_dc(occupation=1.0, dc_guess=dc_guess, **common_kwargs(v=0.3, tau=1e-2))
    assert np.allclose(dc, dc_guess)


def test_fixed_occupation_dc_increases_occupation():
    # Requesting two electrons on the impurity requires pushing the doubly
    # occupied impurity below the bath: dc > eps + U - eps_b = 6.
    dc_guess = 0.5 * np.identity(2, dtype=complex)
    dc = fixed_occupation_dc(occupation=2.0, dc_guess=dc_guess, **common_kwargs(v=0.3, tau=1e-2))
    assert_uniform_shift(dc, dc_guess)
    assert dc[0, 0].real > 6.0, dc


def test_fixed_occupation_dc_decreases_occupation():
    # A guess of 7 puts two electrons on the impurity; requesting one electron
    # must bring the double counting back below the charge-transfer point.
    dc_guess = 7.0 * np.identity(2, dtype=complex)
    dc = fixed_occupation_dc(occupation=1.0, dc_guess=dc_guess, **common_kwargs(v=0.3, tau=1e-2))
    assert_uniform_shift(dc, dc_guess)
    assert dc[0, 0].real < 6.0, dc


def test_fixed_occupation_dc_unreachable_raises():
    # Only two bath spin-orbitals and three electrons in total: the impurity
    # occupation cannot drop below one.
    dc_guess = 0.5 * np.identity(2, dtype=complex)
    with pytest.raises(RuntimeError, match="Could not bracket"):
        fixed_occupation_dc(occupation=0.2, dc_guess=dc_guess, **common_kwargs(v=0.3, tau=1e-2))


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
    n = _noninteracting_impurity_occupation(h0, impurity_indices=[0], n_spin_orbitals=2, tau=tau)

    h = np.array([[e_imp, v], [v, e_bath]], dtype=complex)
    energies, vecs = np.linalg.eigh(h)
    f = 1.0 / (1.0 + np.exp(energies / tau))
    expected = float(np.sum(f * np.abs(vecs[0, :]) ** 2))  # <imp| sum_n f_n |v_n><v_n| |imp>
    assert 0.0 < expected < 1.0  # genuinely fractional, not a plateau boundary
    assert np.isclose(n, expected, atol=1e-12), (n, expected)


def test_fixed_occupation_dc_derives_target_from_hloc():
    # Omitting `occupation` must pin the h_loc-derived target. For this cluster
    # both non-interacting levels sit below the Fermi level, so the derived
    # occupation is 2; the resulting DC must match an explicit occupation=2 call
    # (and, as in test_fixed_occupation_dc_increases_occupation, push dc > 6).
    from impurityModel.ed.double_counting import _noninteracting_impurity_occupation

    kwargs = common_kwargs(v=0.3, tau=1e-2)
    derived = _noninteracting_impurity_occupation(
        kwargs["model"].h0, impurity_indices=[0, 1], n_spin_orbitals=4, tau=kwargs["basis"].tau
    )
    assert np.isclose(derived, 2.0, atol=1e-6)

    dc_guess = 0.5 * np.identity(2, dtype=complex)
    dc_auto = fixed_occupation_dc(dc_guess=dc_guess, **kwargs)
    dc_explicit = fixed_occupation_dc(occupation=2.0, dc_guess=dc_guess, **common_kwargs(v=0.3, tau=1e-2))
    assert_uniform_shift(dc_auto, dc_guess)
    assert dc_auto[0, 0].real > 6.0, dc_auto
    assert np.allclose(dc_auto, dc_explicit), (dc_auto, dc_explicit)


@pytest.mark.mpi
def test_fixed_peak_dc_ranks_agree():
    # The Newton loop in fixed_peak_dc branches on Lanczos energies, which are
    # only replicated to roundoff across ranks. Every rank must nevertheless run
    # the same iterations and return an identical dc; a per-rank divergence
    # would deadlock on the next collective solve instead of returning here.
    comm = MPI.COMM_WORLD
    dc_guess = 0.5 * np.identity(2, dtype=complex)
    dc = fixed_peak_dc(peak_position=1.2, dc_guess=dc_guess, **common_kwargs(v=0.01, tau=1e-3))
    gathered = comm.gather(dc, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert np.array_equal(dc, other), (dc, other)


@pytest.mark.mpi
def test_fixed_occupation_dc_ranks_agree():
    # Occupation control keys off the Allreduced density matrix, so agreement is
    # by construction; guard it against regressions all the same.
    comm = MPI.COMM_WORLD
    dc_guess = 0.5 * np.identity(2, dtype=complex)
    dc = fixed_occupation_dc(occupation=2.0, dc_guess=dc_guess, **common_kwargs(v=0.3, tau=1e-2))
    gathered = comm.gather(dc, root=0)
    if comm.rank == 0:
        for other in gathered[1:]:
            assert np.array_equal(dc, other), (dc, other)
