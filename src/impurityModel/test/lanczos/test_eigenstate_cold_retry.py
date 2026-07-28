"""``CIPSISolver.get_eigenvectors`` warm-start sector-blindness guards.

A block warm-started from converged eigenvectors spans a (near-)invariant subspace. When the
ground state has moved to another near-decoupled charge sector between solves (the
fixed-occupation DC search jumps across a charge-transfer crossing between trial shifts),
Lanczos restarted from that block converges the wrong sector's states only -- either returning
short with everything inside the thermal cut (an uncertified, path-dependent manifold), or,
worse, returning a full-looking certified set whose "lowest" state is an excited state. Both
showed up as a noisy, non-monotone occupation in the NiO DC search. Two guards:

1. the warm block is always augmented with the cold rank-independent full-support hash vector,
   so no sector is ever unreachable;
2. if a warm-started solve nevertheless exhausts inside the thermal cut, it is retried ONCE
   from the cold vector alone before the uncertified manifold is accepted.

The Lanczos kernel is replaced by a deterministic fake (same on every rank, so all collective
decisions stay rank-invariant); the real physics is covered by the NiO end-to-end reproduction.
"""

import itertools

import numpy as np
from mpi4py import MPI

import impurityModel.ed.cipsi_solver as cipsi_module
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant

IMPURITY_ORBITALS = {0: [[0, 1]]}
BATH_STATES = ({0: [[2, 3]]}, {0: [[4, 5]]})
N_SPIN_ORBITALS = 6
CUT = 1e-3  # thermal energy cut handed to get_eigenvectors as max_energy


def _det(occupied):
    """SlaterDeterminant with the given orbitals occupied (MSB-first bit convention)."""
    chunk = 0
    for orb in occupied:
        chunk |= 1 << (63 - orb)
    return SlaterDeterminant((chunk,))


def _make_solver():
    basis = Basis(
        IMPURITY_ORBITALS,
        BATH_STATES,
        nominal_impurity_occ={0: 2},
        comm=MPI.COMM_WORLD,
        verbose=False,
    )
    basis.add_states([_det(occ) for occ in itertools.combinations(range(N_SPIN_ORBITALS), 4)])
    assert len(basis) >= 5  # num_wanted below must exceed the fake's 3 exhausted states
    return basis, CIPSISolver(basis)


def _warm_refs(basis):
    """A single-determinant reference block, fed on rank 0 only (redistribute_psis SUMS replicas)."""
    local = list(basis.local_basis)
    parts = basis.comm.allgather(local) if basis.is_distributed else [local]
    det0 = sorted(det for part in parts for det in part)[0]
    rank = basis.comm.rank if basis.is_distributed else 0
    seed = ManyBodyState({det0: 1.0} if rank == 0 else {}, width=1)
    (out,) = basis.redistribute_psis(seed)[0].to_states()
    return [out]


def _h_op():
    return ManyBodyOperator({((i, "c"), (i, "a")): float(i + 1) for i in range(N_SPIN_ORBITALS)})


def _fake_lanczos(exhaust_first_calls, calls):
    """Fake restarted_lanczos: the first ``exhaust_first_calls`` calls return 3 states inside the
    cut (a short, exhausted solve); later calls return the full request with the last state far
    beyond the cut (a certified boundary manifold)."""

    def fake(psi0, h_op, basis, num_wanted, max_subspace_blocks, tol, max_restarts, verbose, slaterWeightMin, reort):
        calls.append(np.array(psi0, copy=True))
        if len(calls) <= exhaust_first_calls:
            e = np.array([0.0, 1e-4, 2e-4])
        else:
            e = np.concatenate([np.linspace(0.0, 2e-4, num_wanted - 1), [10.0]])
        vecs = np.zeros((psi0.shape[0], len(e)), dtype=complex)
        for j in range(min(len(e), psi0.shape[0])):
            vecs[j, j] = 1.0
        return e, vecs

    return fake


def test_warm_start_exhaustion_triggers_one_cold_retry(monkeypatch, capsys):
    basis, solver = _make_solver()
    calls = []
    monkeypatch.setitem(cipsi_module.SOLVERS, "irlm", _fake_lanczos(1, calls))

    e_ref, psi_refs = solver.get_eigenvectors(
        _h_op(), num_wanted=1, max_energy=CUT, dense_cutoff=1, solver="irlm", psi_refs=_warm_refs(basis)
    )

    # First call: the warm block augmented with the cold full-support column (2 columns).
    # Exactly one retry, from the cold block alone (1 column, dense on every rank).
    assert len(calls) == 2
    assert calls[0].shape[1] == 2
    assert calls[1].shape[1] == 1
    if calls[1].shape[0] > 0:
        assert np.all(calls[1] != 0)
    # The certified manifold: the beyond-cut state is trimmed, the rest is returned.
    assert len(e_ref) == len(psi_refs) > 3
    assert np.max(e_ref) < CUT
    assert "cannot be shown to be complete" not in capsys.readouterr().out


def test_certified_first_solve_does_not_retry(monkeypatch):
    basis, solver = _make_solver()
    calls = []
    monkeypatch.setitem(cipsi_module.SOLVERS, "irlm", _fake_lanczos(0, calls))

    solver.get_eigenvectors(
        _h_op(), num_wanted=1, max_energy=CUT, dense_cutoff=1, solver="irlm", psi_refs=_warm_refs(basis)
    )

    assert len(calls) == 1  # no extra solves on the common (certified) path
    assert calls[0].shape[1] == 2  # but the warm block still carries the cold guard column


def test_cold_start_exhaustion_does_not_loop(monkeypatch, capsys):
    basis, solver = _make_solver()
    # psi_refs=None (the default): cold from the start, nothing fresh to retry with
    calls = []
    monkeypatch.setitem(cipsi_module.SOLVERS, "irlm", _fake_lanczos(10**9, calls))

    e_ref, _ = solver.get_eigenvectors(_h_op(), num_wanted=1, max_energy=CUT, dense_cutoff=1, solver="irlm")

    assert len(calls) == 1
    assert len(e_ref) == 3
    rank = basis.comm.rank if basis.is_distributed else 0
    if rank == 0:
        assert "cannot be shown to be complete" in capsys.readouterr().out
