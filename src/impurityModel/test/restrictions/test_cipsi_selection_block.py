"""Exactness of ``CIPSISolver._apply_block_and_redistribute``'s in-place per-column prune.

``_apply_block_and_redistribute`` used to prune each column of ``apply_block``'s output by
splitting the block into ``p`` separate width-1 ``ManyBodyState`` objects
(:meth:`ManyBodyState.to_states`), pruning each independently, then reassembling
(:meth:`ManyBodyState.from_states`) -- a round trip that copies the full determinant key for
every (row, column) pair twice, measured at ~104 B/pair and the dominant term in the CIPSI
selection round's memory peak (see ``doc/plans/dc_smo_memory.md``, the crashed SrMnO3
double-counting search this was written to fix). The replacement zeros pruned entries in place
via the block's own buffer-protocol view, never materializing the per-column objects.

This file locks the replacement to the *exact* semantics of the round trip it replaces --
``_old_apply_block_and_redistribute`` below is that round trip, rebuilt verbatim from primitives
``to_states``/``from_states``/``prune_rows`` still ship (nothing was removed), so this is a direct
comparison against the documented old behaviour, not a description of it. Comparison is bit-for-
bit (`np.array_equal`, not `np.allclose`): the only thing that could differ between the two paths
is which entries the cutoff test zeros, and a boundary mismatch there is exactly the class of bug
`test_cipsi_determinism.py` exists to catch elsewhere in this module (a 2% cross-rank score drift
once flipped a candidate across ``de2_min`` and changed a produced basis by 5%). Test amplitudes
are chosen well clear of every cutoff tested, so no case here is a coin flip on a cutoff-boundary
rounding artifact -- see ``test_cutoff_lands_between_two_real_magnitudes`` for the check that this
premise holds.
"""

import itertools

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3]]}, {0: [[4, 5]]})
N_SPIN_ORBITALS = 6
N_ELECTRONS = 3
# Hopping magnitude chosen so the couplings this Hamiltonian actually produces sit far from
# every cutoff tested below (see test_cutoff_lands_between_two_real_magnitudes).
_HOP = 0.15 + 0.05j


def _det(occupied):
    """SlaterDeterminant with the given orbitals occupied (MSB-first bit convention)."""
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _make_solver(comm):
    """A small basis (4 of the 20 three-electron determinants in 6 spin-orbitals) plus a real,
    off-diagonal Hamiltonian -- real enough that `apply_block` both stays in-basis (diagonal
    on-site terms) and leaves it (the hoppings), which is what exercises the candidate-block
    machinery under test."""
    basis = Basis(IMPURITY_ORBITALS, BATH_STATES, nominal_impurity_occ={0: 1}, comm=comm, verbose=False)
    all_dets = [_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), N_ELECTRONS)]
    basis.add_states(all_dets[:4])
    return CIPSISolver(basis), all_dets[:4]


def _hamiltonian():
    terms = {((i, "c"), (i, "a")): 0.3 * (i + 1) for i in range(N_SPIN_ORBITALS)}
    for a, b in ((0, 2), (1, 3), (2, 4), (3, 5), (0, 4)):
        terms[((a, "c"), (b, "a"))] = _HOP
        terms[((b, "c"), (a, "a"))] = _HOP.conjugate()
    return ManyBodyOperator(terms)


def _psi_ref_block(basis_dets, widths=(1, 2, 3)):
    """A few reference columns of increasing, overlapping support -- width p = len(widths)."""
    return [ManyBodyState(dict.fromkeys(basis_dets[: w + 1], 1.0 + 0.3j)) for w in widths]


def _old_apply_block_and_redistribute(solver, H, psi_ref, cutoff):
    """The pre-rewrite algorithm, rebuilt verbatim from primitives still shipped today."""
    cols = H.apply_block(ManyBodyState.from_states(psi_ref), cutoff).to_states()
    for s in cols:
        s.prune_rows(cutoff)
    merged = solver.basis.redistribute_block(ManyBodyState.from_states(cols))
    merged.prune_rows(0.0)
    return merged


def test_cutoff_lands_between_two_real_magnitudes():
    """Premise check for every comparison below: no cutoff tested sits within many ULPs of an
    actual computed |amp|, so a pass here cannot be explained by a lucky boundary rounding."""
    _solver, basis_dets = _make_solver(None)
    H = _hamiltonian()
    raw = H.apply_block(ManyBodyState.from_states(_psi_ref_block(basis_dets)), 0.0)
    mags = np.sort(np.unique(np.abs(np.asarray(raw))))
    mags = mags[mags > 0]
    assert mags[0] > 1e-3, "premise: the smallest nonzero coupling is well clear of cutoff=0.0"
    for cutoff in (0.001, 0.1):
        # every tested cutoff must fall strictly between two produced magnitudes (or below all
        # of them), never within 1e-9 of one
        assert np.all(np.abs(mags - cutoff) > 1e-9), (cutoff, mags)


@pytest.mark.parametrize("cutoff", [0.0, 0.001, 0.1])
def test_matches_the_old_to_states_from_states_round_trip_serial(cutoff):
    solver, basis_dets = _make_solver(None)
    H = _hamiltonian()
    psi_ref = _psi_ref_block(basis_dets)

    new_block = solver._apply_block_and_redistribute(H, psi_ref, cutoff)
    old_block = _old_apply_block_and_redistribute(solver, H, psi_ref, cutoff)

    assert new_block.keys() == old_block.keys()
    assert len(new_block.keys()) > 0, "premise: this Hamiltonian must actually produce rows"
    np.testing.assert_array_equal(np.asarray(new_block), np.asarray(old_block))


@pytest.mark.mpi
@pytest.mark.parametrize("cutoff", [0.0, 0.001, 0.1])
def test_matches_the_old_to_states_from_states_round_trip_mpi(cutoff):
    """Same comparison, distributed -- run at -n 2 and -n 3 (CLAUDE.md: an empty local
    partition only shows up at -n 3+, and this Hamiltonian's tiny 4-determinant basis is
    exactly the kind of small hash-distributed fixture that can produce one)."""
    comm = MPI.COMM_WORLD
    solver, basis_dets = _make_solver(comm)
    H = _hamiltonian()
    psi_ref = _psi_ref_block(basis_dets)

    new_block = solver._apply_block_and_redistribute(H, psi_ref, cutoff)
    old_block = _old_apply_block_and_redistribute(solver, H, psi_ref, cutoff)

    # Each rank's *local* share of the redistributed block (its hash-owned slice of the shared
    # support, not the whole thing -- ranks legitimately disagree on row count here) must match
    # between the two algorithms exactly.
    assert new_block.keys() == old_block.keys()
    np.testing.assert_array_equal(np.asarray(new_block), np.asarray(old_block))


def test_buffer_export_is_released_before_the_redistribute():
    """The in-place prune's numpy view must not still be exported when structural methods
    (redistribute, prune_rows) run on the same block afterward -- both raise RuntimeError while
    a view is alive (see ManyBodyState.prune_rows's own guard). A regression here would fail
    every call, not just this one, but this pins the exact mechanism.
    """
    solver, basis_dets = _make_solver(None)
    H = _hamiltonian()
    psi_ref = _psi_ref_block(basis_dets)
    # Must simply not raise.
    block = solver._apply_block_and_redistribute(H, psi_ref, 0.0)
    # And the result must still be usable as a normal block afterward (buffer genuinely freed).
    block.prune_rows(0.0)
