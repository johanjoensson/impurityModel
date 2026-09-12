"""``CIPSISolver.expand``'s eigenstate request: ask for the manifold that is there, not twice it.

``expand`` used to request ``2 * len(psi_refs)`` eigenstates per CIPSI cycle. The request -- not
the kept count -- is what sizes the eigensolver: ``_size_subspace`` turns it into roughly
``4 * num_wanted`` retained Krylov columns, and ``get_eigenvectors`` holds that many residuals to
``tol``. The doubling existed to keep the ``need_more`` re-solve loop from firing, and it is a
permanent cost paid against an occasional one.

On the SrMnO3 double-counting reproduction the doubling is worse than a flat 2x: cycle 0 runs the
*dense* branch on the 120-determinant seed basis, where 86 of 120 states sit inside the 0.23 eV
thermal window -- a property of a basis too small to resolve the window rather than of the
physics -- and that 86 then sized cycle 1's solve on a basis 35x larger whose converged thermal
manifold is 10-78. Cycle 1 is the peak of the whole expansion.

What must not change is the manifold itself. These tests pin the request arithmetic directly, and
then check end-to-end that ``expand`` reaches the *same kept manifold* under the new rule as under
the old doubling -- on a fixture whose manifold actually grows across cycles, since a fixture whose
manifold is constant would pass under any rule.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import cipsi_solver
from impurityModel.ed.cipsi_solver import _EIGENSTATE_PAD, CIPSISolver, _manifold_request
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3, 4, 5]]}, {0: [[6, 7, 8, 9]]})
N_SPIN_ORBITALS = 10
N_OCCUPIED = 5
# Evenly spaced one-body energies, so the five-particle spectrum is dense and a thermal window
# admits a manifold that genuinely grows as the basis does -- the premise the end-to-end test
# below needs, and which it checks rather than assumes.
ORBITAL_ENERGIES = [0.10 * (i + 1) for i in range(N_SPIN_ORBITALS)]


def _det(occupied):
    """SlaterDeterminant with the given orbitals occupied (MSB-first bit convention)."""
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _diagonal_hop():
    return {((i, "c"), (i, "a")): ORBITAL_ENERGIES[i] for i in range(N_SPIN_ORBITALS)}


# ---------------------------------------------------------------------------------------
# The request arithmetic
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_kept", [1, 2, 10, 44, 78, 86, 500])
@pytest.mark.parametrize("prev_kept", [None, 0, 1, 10, 86, 1000])
def test_the_request_always_covers_the_manifold_plus_the_pad(n_kept, prev_kept):
    """The floor is non-negotiable: a request below the kept count would drop states that are
    inside the cut, which is the one thing the thermal manifold may not do."""
    assert _manifold_request(n_kept, prev_kept) >= n_kept + _EIGENSTATE_PAD


@pytest.mark.parametrize("n_kept", [10, 44, 78, 86, 500])
def test_no_history_means_the_flat_pad_and_not_the_whole_manifold_as_growth(n_kept):
    """``prev_kept is None`` must not be read as ``0``.

    Cycle 0 returns its whole manifold from the dense branch (which ignores ``num_wanted``
    entirely when a cut is given), so scoring that count as "growth from zero" would extrapolate
    a request of ``3 * n_kept`` -- larger than the doubling this rule replaces. ``None`` is
    therefore "no history", distinct from "grew from nothing".
    """
    assert _manifold_request(n_kept, None) == n_kept + _EIGENSTATE_PAD
    assert _manifold_request(n_kept, None) < 2 * n_kept + _EIGENSTATE_PAD


@pytest.mark.parametrize(("prev_kept", "n_kept"), [(86, 78), (78, 10), (44, 44), (1000, 1)])
def test_a_flat_or_shrinking_manifold_gets_the_flat_pad(prev_kept, n_kept):
    assert _manifold_request(n_kept, prev_kept) == n_kept + _EIGENSTATE_PAD


@pytest.mark.parametrize(("prev_kept", "n_kept", "expected_margin"), [(10, 30, 40), (40, 58, 36), (2, 4, 10)])
def test_growth_is_extrapolated_at_twice_the_last_rate(prev_kept, n_kept, expected_margin):
    """A growing manifold buys headroom of twice the last increment (floored at the pad), so a
    run that is still widening does not pay a re-solve on every cycle."""
    assert _manifold_request(n_kept, prev_kept) == n_kept + expected_margin


def test_the_request_beats_the_old_doubling_on_a_settled_manifold():
    """The point of the change, stated as an inequality rather than a measurement: once the
    manifold stops growing fast, the new request is strictly smaller than ``2 * n_kept``."""
    for n_kept in (44, 78, 86, 182):
        for prev_kept in (None, n_kept, n_kept - 1):
            assert _manifold_request(n_kept, prev_kept) < 2 * n_kept


# ---------------------------------------------------------------------------------------
# End to end: the kept manifold is unchanged
# ---------------------------------------------------------------------------------------


def _run_expand(comm, request_rule, tau):
    """One full ``expand`` under a given request rule; return (kept energies, basis size)."""
    basis = Basis(
        IMPURITY_ORBITALS,
        BATH_STATES,
        nominal_impurity_occ={0: 1},
        tau=tau,
        comm=comm,
        verbose=False,
    )
    # Seed with a handful of determinants so CIPSI has somewhere to expand from.
    seeds = list(itertools.combinations(range(N_SPIN_ORBITALS), N_OCCUPIED))[:6]
    basis.add_states([_det(occ) for occ in seeds])
    solver = CIPSISolver(basis)
    saved = cipsi_solver._manifold_request
    cipsi_solver._manifold_request = request_rule
    try:
        # dense_cutoff=1 forces the Krylov/TRLM branch, the only one the request rule reaches.
        solver.expand(_diagonal_hop(), de2_min=1e-12, dense_cutoff=1, slaterWeightMin=0)
    finally:
        cipsi_solver._manifold_request = saved
    es, _psis = solver.get_eigenvectors(
        _diagonal_hop(),
        num_wanted=len(solver.psi_refs),
        max_energy=cipsi_solver.energy_cut(tau),
        dense_cutoff=1,
        slaterWeightMin=0,
        psi_refs=solver.psi_refs,
    )
    return np.sort(np.real(es)), basis.size


def _old_doubling(n_kept, prev_kept):
    """``expand``'s pre-change rule, verbatim: twice the kept manifold."""
    return 2 * n_kept


@pytest.mark.parametrize("tau", [0.05, 0.2])
def test_expand_keeps_the_same_manifold_as_the_old_doubling(tau):
    comm = MPI.COMM_WORLD
    new_es, new_size = _run_expand(comm, _manifold_request, tau)
    old_es, old_size = _run_expand(comm, _old_doubling, tau)

    # Premise: the fixture must actually put several states inside the cut, or this compares
    # two one-state manifolds and proves nothing.
    assert len(old_es) >= 3, f"fixture admits only {len(old_es)} states at tau={tau}"
    assert new_size == old_size
    assert len(new_es) == len(old_es)
    np.testing.assert_allclose(new_es, old_es, rtol=0, atol=1e-9)


@pytest.mark.mpi
@pytest.mark.parametrize("tau", [0.2])
def test_the_kept_manifold_is_rank_independent(tau):
    """The request is derived from ``len(psi_refs)``, which is rank-replicated -- so every rank
    must enter the collective eigensolve asking for the same number. A rank-local request would
    be the MPI-rule violation, not merely a different answer."""
    comm = MPI.COMM_WORLD
    es, size = _run_expand(comm, _manifold_request, tau)
    gathered = comm.allgather((len(es), size))
    assert len(set(gathered)) == 1, gathered
