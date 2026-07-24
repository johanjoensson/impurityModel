"""Regression tests for TRLM/IRLM on the ManyBodyState path with an empty MPI rank.

When more MPI ranks exist than the basis distributes onto (determinants are owned by
``routing_hash() % comm.size``), some rank can own *zero* local determinants. The
locking-deflation branches in ``_lanczos_step.pxi`` used to gate their ``Allreduce``
calls on ``if locked``/``bool(locked)``, which for a ``ManyBodyState`` (no ``__bool__``
defined) falls back to ``len()`` -- its LOCAL ROW count, not its width. An empty rank
has width > 0 but rows == 0, so it read the gate as False while every other rank read
it as True, executing the collective on some ranks but not others -> MPI deadlock.
Confirmed via py-spy at ``mpiexec -n 3`` for ``test_irlm_cy_diagonal_mpi``
(``test_block_lanczos_cy_mpi.py``): the routing hash of that test's fixed 6 states
happens to leave rank 0 empty only when ``comm.size == 3``, so the standing ``-n 2``
gate never caught it.

These tests force the empty-rank condition *deterministically*, independent of
``comm.size`` and of any particular test's hash luck: they scan single-occupied
determinants across a wide orbital range and keep only the ones whose owner is not
the last rank, so the last rank is guaranteed empty for the whole solve.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczos import (
    _implicitly_restarted_block_lanczos_manybody,
    implicitly_restarted_block_lanczos_cy,
    thick_restart_block_lanczos_cy,
)
from impurityModel.ed.BlockLanczosArray import block_normalize
from impurityModel.ed.BlockLanczosArray import Reort as ArrayReort
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState

_N_ORBITALS = 64  # one uint64 chunk; wide enough to find 6 owners avoiding one rank
_N_STATES = 6


def _singlet_bytes(orbital, n_orbitals=_N_ORBITALS):
    """MSB-first single-occupation bytes for ``orbital`` (orbital i = bit 7-i of byte i//8)."""
    b = bytearray((n_orbitals + 7) // 8)
    b[orbital // 8] |= 1 << (7 - (orbital % 8))
    return bytes(b)


def _diagonal_h_and_basis_with_empty_rank(comm):
    """Build a diagonal H and Basis whose last rank is guaranteed to own zero states.

    Returns ``(h_op, basis, eigvals)``. ``eigvals`` are the (ascending) diagonal
    entries, i.e. the exact spectrum, for ``_N_STATES`` single-occupation
    determinants scattered across ``_N_ORBITALS`` orbitals. Diagonal H never
    generates new determinants, so the basis never grows past these states and the
    empty rank stays empty for the whole solve.
    """
    from impurityModel.ed.ManyBodyUtils import SlaterDeterminant

    empty_rank = comm.size - 1
    chosen_orbitals = []
    for orbital in range(_N_ORBITALS):
        sd = SlaterDeterminant.from_bytes(_singlet_bytes(orbital))
        if sd.routing_hash() % comm.size != empty_rank:
            chosen_orbitals.append(orbital)
        if len(chosen_orbitals) == _N_STATES:
            break
    assert len(chosen_orbitals) == _N_STATES, (
        f"could not find {_N_STATES} orbitals avoiding rank {empty_rank} at comm.size="
        f"{comm.size} within the first {_N_ORBITALS} candidates"
    )

    eigvals = np.arange(_N_STATES, dtype=float)
    hop = {((orb, "c"), (orb, "a")): float(val) for orb, val in zip(chosen_orbitals, eigvals)}
    h_op = ManyBodyOperator(hop)

    states = [_singlet_bytes(orb) for orb in chosen_orbitals]
    basis = Basis(
        impurity_orbitals={0: [list(range(_N_ORBITALS))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
        comm=comm,
    )

    local_counts = comm.allgather(len(basis.local_basis))
    assert local_counts[empty_rank] == 0, f"test setup failed to produce an empty rank: local counts {local_counts}"

    return h_op, basis, states, eigvals


def _uniform_psi0(basis, states, comm):
    st = ManyBodyState()
    for s in states:
        st[basis.type.from_bytes(s)] = 1.0 / np.sqrt(len(states))
    psi0 = basis.redistribute_psis(st)
    psi0, _ = block_normalize(psi0, mpi=True, comm=comm)
    return psi0


@pytest.mark.mpi
def test_irlm_mbs_empty_rank_full_locked_reort():
    """IRLM, default locked_reort='full' -- the site that deadlocked (_lanczos_step.pxi
    ~line 172-180): the locking-deflation Allreduce must run identically on every rank,
    including the empty one, once a Ritz pair locks."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("empty rank needs comm.size >= 2")

    h_op, basis, states, eigvals = _diagonal_h_and_basis_with_empty_rank(comm)
    psi0 = _uniform_psi0(basis, states, comm)

    eigs, _evecs = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=2,
        max_subspace_blocks=4,
        tol=1e-8,
        max_restarts=10,
        verbose=False,
        comm=comm,
    )

    assert len(eigs) == 2
    np.testing.assert_allclose(eigs, eigvals[:2], atol=1e-8)


@pytest.mark.mpi
def test_trlm_mbs_empty_rank():
    """TRLM under the same empty-rank distribution (general width-sync/redistribution
    robustness; TRLM has no locked-deflation branch of its own)."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("empty rank needs comm.size >= 2")

    h_op, basis, states, eigvals = _diagonal_h_and_basis_with_empty_rank(comm)
    psi0 = _uniform_psi0(basis, states, comm)

    eigs, _evecs = thick_restart_block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=2,
        max_subspace_blocks=4,
        tol=1e-8,
        max_restarts=10,
        verbose=False,
        comm=comm,
    )

    assert len(eigs) == 2
    np.testing.assert_allclose(eigs, eigvals[:2], atol=1e-8)


@pytest.mark.mpi
def test_irlm_mbs_empty_rank_partial_locked_reort():
    """IRLM, locked_reort='partial' -- the sibling defect (_lanczos_step.pxi ~line 712,
    gating the estimate-driven ``_lovl`` Allreduce at ~line 824). Drives the ManyBodyState
    core directly since the dispatching entry points don't expose ``locked_reort``."""
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("empty rank needs comm.size >= 2")

    h_op, basis, states, eigvals = _diagonal_h_and_basis_with_empty_rank(comm)
    psi0 = _uniform_psi0(basis, states, comm)

    eigs, _evecs = _implicitly_restarted_block_lanczos_manybody(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        num_wanted=2,
        max_subspace_blocks=4,
        tol=1e-8,
        max_restarts=10,
        verbose=False,
        reort_mode=ArrayReort.PARTIAL,
        comm=comm,
        locked_reort="partial",
    )

    assert len(eigs) == 2
    np.testing.assert_allclose(eigs, eigvals[:2], atol=1e-8)
