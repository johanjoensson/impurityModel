r"""``basis_transcription.extend_sparse_matrix``: a matrix grown to a larger basis equals one rebuilt on it.

A CIPSI expansion grows its basis every cycle and used to rebuild the whole Hamiltonian matrix each
time. The extension keeps the old entries (rows renumbered, columns spread), applies the operator to
the new determinants only, and mirrors their old-row entries into the old columns. Pinned here against
a full :func:`build_sparse_matrix` on the same basis:

* every stage of a growing basis, the extension chained from the previous extension;
* after the matrix was used in a matvec, which rewrites its row indices in place (``bind_slots``);
* at every rank count, including ranks that gain no determinant in a stage;
* ``None`` when the basis lost a determinant, so the caller rebuilds.

And through ``CIPSISolver.expand``: the expansion takes the extension path, and lands on the energy
of an expansion that rebuilds every cycle.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed import cipsi_solver
from impurityModel.ed.basis_transcription import MatrixSnapshot, build_sparse_matrix, extend_sparse_matrix
from impurityModel.ed.eigensolvers import Lowest
from impurityModel.test.basis.test_sparse_matrix_build import _all_dets, _basis, _operator
from impurityModel.test.support.comms import COMMS


def _gather_dense(mat, comm):
    """The whole operator as a dense array on every rank (the columns summed over the ranks)."""
    local = mat.toarray()
    if comm is not None and comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)
    return local


def _grow(basis, dets, comm):
    """Add ``dets`` from rank 0 (``add_states`` routes them to their owners; collective)."""
    basis.add_states(dets if comm is None or comm.rank == 0 else [])


def _stages():
    dets = _all_dets()
    order = np.random.default_rng(5).permutation(len(dets))
    # 40 -> 300 -> 310 -> 900: the 10-determinant stage leaves ranks with nothing new from -n 3 up.
    cuts = [40, 300, 310, 900]
    return [[dets[i] for i in order[a:b]] for a, b in zip([0] + cuts[:-1], cuts)]


@pytest.mark.parametrize("comm", COMMS)
def test_a_chained_extension_equals_a_rebuild_at_every_stage(comm):
    op = _operator()
    stages = _stages()
    basis = _basis(stages[0] if comm is None or comm.rank == 0 else [], comm)
    mat = build_sparse_matrix(basis, op)
    for added in stages[1:]:
        # Use the matrix first: the matvec binds it, rewriting its row indices in place.
        mat.apply(np.ones((len(basis.local_basis), 1), dtype=complex))
        snapshot = MatrixSnapshot(basis, mat, key="H")
        _grow(basis, added, comm)
        mat = extend_sparse_matrix(snapshot, basis, op)
        assert mat is not None
        expected = build_sparse_matrix(basis, op)
        np.testing.assert_allclose(_gather_dense(mat, comm), _gather_dense(expected, comm), atol=1e-13)
        assert mat.nnz == expected.nnz


@pytest.mark.parametrize("comm", COMMS)
def test_a_basis_that_lost_a_determinant_is_not_extended(comm):
    op = _operator()
    dets = _stages()[1]
    basis = _basis(dets if comm is None or comm.rank == 0 else [], comm)
    snapshot = MatrixSnapshot(basis, build_sparse_matrix(basis, op), key="H")
    smaller = _basis(dets[1:] if comm is None or comm.rank == 0 else [], comm)
    assert extend_sparse_matrix(snapshot, smaller, op) is None


@pytest.mark.parametrize("comm", COMMS)
def test_the_expansion_extends_its_matrix_and_keeps_the_rebuilt_energy(comm, monkeypatch):
    op = _operator()
    seed = _stages()[0][:12]

    def solve():
        basis = _basis(seed if comm is None or comm.rank == 0 else [], comm)
        solver = cipsi_solver.CIPSISolver(basis)
        solver.expand(op, Lowest(), dense_cutoff=1e3, slaterWeightMin=1e-12, de2_min=1e-12)
        e, _psi = solver.get_eigenvectors(op, Lowest(), dense_cutoff=1e3, slaterWeightMin=1e-12, reuse_last=True)
        return float(e[0]), len(basis)

    calls = []
    real = cipsi_solver.extend_sparse_matrix

    def counting(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(cipsi_solver, "extend_sparse_matrix", counting)
    e_ext, n_ext = solve()
    assert any(calls), "no cycle grew its matrix: every cycle rebuilt"

    monkeypatch.setattr(cipsi_solver, "extend_sparse_matrix", lambda *a, **k: None)
    e_full, n_full = solve()
    assert n_ext == n_full
    assert abs(e_ext - e_full) < 1e-10
