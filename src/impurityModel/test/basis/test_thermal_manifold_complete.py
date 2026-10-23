"""The production thermal ground state is the dense-ED thermal manifold; the walk never claims one.

Two manifold kinds, one per question. The occupation walk asks "which charge sector is lowest" and
solves each trial for its lowest energy alone (``Lowest``): one converged pair, no completeness
claim, no window mode. The production ground state asks "which states carry thermal weight" and
solves the winning sector from the seed basis for ``Within.thermal(tau)``, certified complete.

The model is a one-site impurity with a filled valence level and an empty conduction level (6
spin-orbitals, spin-degenerate). Its winning sector (two electrons) has a non-degenerate singlet
ground state, a spin triplet and a further singlet inside the thermal window -- levels that are
unrelated to the ground state by symmetry -- and the exact answer is the dense spectrum of that sector.

An uncapped solve spans the whole 15-determinant sector, so a ground-level-only walk basis finds the
same states here: this pins the contract (energies and multiplicities against the oracle), not the
failure of the previous design, which needs a basis that cannot saturate.
"""

import itertools

import numpy as np
import pytest

from impurityModel.ed import eigensolvers, groundstate
from impurityModel.ed.average import energy_cut
from impurityModel.ed.basis_transcription import build_dense_matrix
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.test.support.comms import COMMS

N_ORB = 6
TAU = 0.27  # energy_cut = 2.49: the singlet, the triplet and the second singlet are inside
IMPURITY = {0: [[0, 1]]}
BATHS = ({0: [[2, 3]]}, {0: [[4, 5]]})
EPS, EV, EC, V, U = 0.9, -1.2, 1.0, 0.2, 2.0


def _model():
    terms = {}
    for s in (0, 1):
        for orbital, energy in ((s, EPS), (2 + s, EV), (4 + s, EC)):
            terms[((orbital, "c"), (orbital, "a"))] = energy
        for bath in (2 + s, 4 + s):
            terms[((s, "c"), (bath, "a"))] = V
            terms[((bath, "c"), (s, "a"))] = V
    terms[((0, "c"), (1, "c"), (1, "a"), (0, "a"))] = U
    return ManyBodyOperator(terms)


def _sector_spectrum(h, n_electrons, comm):
    """The exact spectrum of the ``n_electrons`` sector: every determinant, densely diagonalized."""
    dets = [bytes([sum(1 << (7 - o) for o in occ)]) for occ in itertools.combinations(range(N_ORB), n_electrons)]
    basis = Basis(IMPURITY, BATHS, initial_basis=dets, verbose=False, comm=comm)
    return np.sort(np.linalg.eigvalsh(np.asarray(build_dense_matrix(basis, h))))


def _ground_state(h, comm, dense_cutoff):
    return groundstate.solve_ground_state(
        h,
        IMPURITY,
        BATHS,
        {0: 1},
        tau=TAU,
        slaterWeightMin=1e-12,
        mixed_valence={0: 1},
        dense_cutoff=dense_cutoff,
        comm=comm,
        truncation_threshold=np.inf,
        verbose=False,
    )


@pytest.mark.parametrize("dense_cutoff", [1, 1000], ids=["krylov", "dense"])
@pytest.mark.parametrize("comm", COMMS)
def test_the_thermal_manifold_is_every_state_inside_the_window(comm, dense_cutoff):
    h = _model()
    exact = _sector_spectrum(h, 2, comm)
    expected = exact[exact - exact[0] <= energy_cut(TAU)]
    # Premise: the window holds a non-degenerate ground state and a degenerate triplet.
    assert len(expected) >= 5 and np.sum(np.abs(expected - expected[1]) < 1e-9) == 3

    gs = _ground_state(h, comm, dense_cutoff)

    np.testing.assert_allclose(np.sort(np.real(gs.es)), expected, atol=1e-8)
    assert gs.psis.width == len(expected)
    assert sum(gs.occupation.values()) == 0  # the walk's sector: an empty impurity, the valence level filled


@pytest.mark.parametrize("comm", COMMS)
def test_the_walk_asks_for_the_lowest_state_only(comm, monkeypatch):
    """Every trial sector is a ``Lowest`` solve: window mode (and its certificate) never runs in the walk."""
    from impurityModel.ed import cipsi_solver

    def forbidden(*args, **kwargs):
        raise AssertionError("the occupation walk ran a window solve")

    monkeypatch.setattr(cipsi_solver, "solve_window", forbidden)
    search = groundstate.find_ground_state_sector(
        _model(),
        IMPURITY,
        BATHS,
        {0: 1},
        mixed_valence={0: 1},
        dense_cutoff=1,
        comm=comm,
        verbose=False,
        truncation_threshold=np.inf,
    )
    exact = _sector_spectrum(_model(), 2, comm)
    assert search.energy == pytest.approx(exact[0], abs=1e-6)  # the walk is loose by design


@pytest.mark.parametrize("comm", COMMS)
def test_the_production_solve_does_not_continue_the_walks_basis(comm, monkeypatch):
    """The winner is solved from its seed basis: exactly one thermal build, and nothing is retained by the walk."""
    built = []
    real = groundstate.build_basis_and_solver

    def recording(*args, **kwargs):
        built.append(kwargs["manifold"])
        return real(*args, **kwargs)

    monkeypatch.setattr(groundstate, "build_basis_and_solver", recording)
    gs = _ground_state(_model(), comm, 1)

    assert any(isinstance(m, eigensolvers.Lowest) for m in built)
    assert [m for m in built if isinstance(m, eigensolvers.Within)] == [eigensolvers.Within.thermal(TAU)]
    assert not hasattr(gs.basis, "walk_solver")
