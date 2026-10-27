r"""``DiscoveredMatrix.set_p_keys``: the matrix CIPSI keeps across its cycles equals one rebuilt on the basis.

A CIPSI expansion grows its basis every cycle (and truncates it now and then). The kept matrix stores each
column's complete image, so a determinant admitted later finds its rows in the older columns, and
``set_p_keys`` applies the operator to the new determinants only. Pinned against a full
:func:`build_sparse_matrix` on the same basis:

* every stage of a growing basis, chained, with a matvec in between (the P slots are renumbered into key order, and
  the built columns with them);
* a truncation (columns dropped, their rows kept) followed by a regrowth;
* at every rank count, including ranks that gain no determinant in a stage;
* through ``CIPSISolver.expand``: the expansion reuses its matrix and lands on the energy of one that rebuilds.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import cipsi_solver
from impurityModel.ed.basis_transcription import build_sparse_matrix
from impurityModel.ed.discovered_matrix import DiscoveredMatrix
from impurityModel.ed.eigensolvers import Lowest
from impurityModel.test.basis.test_sparse_matrix_build import _all_dets, _basis, _operator
from impurityModel.test.support.comms import COMMS


def _gather_dense(mat, comm):
    local = mat.toarray()
    if comm is not None and comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)
    return local


def _stages():
    dets = _all_dets()
    order = np.random.default_rng(5).permutation(len(dets))
    # 40 -> 300 -> 310 -> 900: the 10-determinant stage leaves ranks with nothing new from -n 3 up.
    cuts = [40, 300, 310, 900]
    return [[dets[i] for i in order[a:b]] for a, b in zip([0] + cuts[:-1], cuts)]


def _held(basis, op, comm):
    return DiscoveredMatrix(op, (basis.n_bytes + 7) // 8, comm if comm is not None and comm.size > 1 else None)


@pytest.mark.parametrize("comm", COMMS)
def test_a_chained_growth_equals_a_rebuild_at_every_stage(comm):
    op = _operator()
    stages = _stages()
    basis = _basis(stages[0] if comm is None or comm.rank == 0 else [], comm)
    mat = _held(basis, op, comm)
    mat.set_p_keys(list(basis.local_basis))
    for added in stages[1:]:
        mat.apply(np.ones((len(basis.local_basis), 1), dtype=complex))
        basis.add_states(added if comm is None or comm.rank == 0 else [])
        n_new, n_gone = mat.set_p_keys(list(basis.local_basis))
        assert n_gone == 0
        expected = build_sparse_matrix(basis, op)
        np.testing.assert_allclose(_gather_dense(mat, comm), _gather_dense(expected, comm), atol=1e-13)
        x = np.random.default_rng(1).standard_normal((len(basis.local_basis), 2)) + 0j
        np.testing.assert_allclose(mat.apply(x), expected.apply(x), atol=1e-12)


@pytest.mark.parametrize("comm", COMMS)
def test_a_truncation_then_a_regrowth_equals_a_rebuild(comm):
    op = _operator()
    dets = _stages()[1]
    mine = dets if comm is None or comm.rank == 0 else []
    basis = _basis(mine, comm)
    mat = _held(basis, op, comm)
    mat.set_p_keys(list(basis.local_basis))
    smaller = _basis(dets[::2] if comm is None or comm.rank == 0 else [], comm)
    n_new, n_gone = mat.set_p_keys(list(smaller.local_basis))
    total = len(smaller.local_basis) if comm is None else comm.allreduce(n_gone)
    assert n_new == 0
    assert (n_gone > 0) or len(basis.local_basis) == len(smaller.local_basis)
    expected = build_sparse_matrix(smaller, op)
    np.testing.assert_allclose(_gather_dense(mat, comm), _gather_dense(expected, comm), atol=1e-13)
    mat.set_p_keys(list(basis.local_basis))  # back to the larger basis: the dropped columns are rebuilt
    np.testing.assert_allclose(_gather_dense(mat, comm), _gather_dense(build_sparse_matrix(basis, op), comm), atol=1e-13)
    assert total >= 0


@pytest.mark.parametrize("comm", COMMS)
def test_the_expansion_reuses_its_matrix_and_keeps_the_rebuilt_energy(comm, monkeypatch):
    op = _operator()
    seed = _stages()[0][:12]

    def solve(**kw):
        basis = _basis(seed if comm is None or comm.rank == 0 else [], comm)
        solver = cipsi_solver.CIPSISolver(basis)
        solver.expand(op, Lowest(), dense_cutoff=1e3, slaterWeightMin=1e-12, de2_min=1e-12, **kw)
        e, _psi = solver.get_eigenvectors(op, Lowest(), dense_cutoff=1e3, slaterWeightMin=1e-12, reuse_last=True)
        return float(e[0]), len(basis)

    calls = []
    real = DiscoveredMatrix.set_p_keys

    def counting(self, keys):
        calls.append(hasattr(self, "_counted"))  # True: a matrix an earlier cycle left behind
        self._counted = True
        return real(self, keys)

    monkeypatch.setattr(DiscoveredMatrix, "set_p_keys", counting)
    e_kept, n_kept = solve()
    assert any(calls), "no cycle reused its matrix: every cycle built a new one"

    calls.clear()
    monkeypatch.setattr(cipsi_solver, "_SNAPSHOT_BUDGET_FRACTION", 0.0)  # the matrix never fits: dropped each cycle
    e_full, n_full = solve(memory_budget_bytes=10**12)
    assert not any(calls)
    assert n_kept == n_full
    assert abs(e_kept - e_full) < 1e-10
