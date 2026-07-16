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

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.selfenergy import fixed_peak_dc, fixed_occupation_dc

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
    return dict(
        h0_op=h0,
        N0={0: 1},
        mixed_valence=None,
        impurity_orbitals={0: [0, 1]},
        bath_states=({0: [2, 3]}, {0: []}),
        u4=u4,
        spin_flip_dj=False,
        tau=tau,
        rank=MPI.COMM_WORLD.rank,
        verbose=False,
        dense_cutoff=1000,
        slaterWeightMin=np.sqrt(np.finfo(float).eps),
        truncation_threshold=int(1e8),
    )


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
    kwargs["N0"] = {0: 1, 1: 1}
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
