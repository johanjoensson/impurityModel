r"""``DistributedSparseMatrix`` honours the contract of the sparse operator the eigensolvers and the array kernels read.

The oracle is scipy on the same matrix: ``shape``, ``nnz``, ``apply`` through ``exchange_plan``, ``to_scipy``,
``toarray``, ``diagonal`` and ``col_offset``, for a matrix built
from scipy/dense data (keyless) and for one built on a basis from an operator (``K = P``, the ``P H P`` matrix, with
its P slots in the basis' key order). Both at every rank count, an empty rank included.
"""

import numpy as np
import pytest
import scipy.sparse as sps

from impurityModel.ed.basis_transcription import build_sparse_matrix
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.test.basis.test_sparse_matrix_build import _all_dets, _basis, _operator, _oracle_columns
from impurityModel.test.support.comms import COMMS

N = 17
WIDTH = 3


def _local_columns(comm, seed=0, density=0.3):
    """This rank's ``(N, n_local)`` columns of one random square matrix (rank-contiguous split, an empty rank
    when ``N`` is small against the rank count)."""
    rng = np.random.default_rng(seed)
    full = sps.random(N, N, density=density, random_state=rng, dtype=float).toarray()
    full = full + 1j * rng.standard_normal((N, N)) * (full != 0)
    full += np.diag(rng.standard_normal(N))
    size, rank = (1, 0) if comm is None else (comm.size, comm.rank)
    bounds = np.linspace(0, N, size + 1).astype(int)
    if size > 2:
        bounds = np.array([0, 0] + [N * i // (size - 1) for i in range(1, size)])[: size + 1]
        bounds[-1] = N
    lo, hi = int(bounds[rank]), int(bounds[rank + 1])
    return full, sps.csc_array(full[:, lo:hi]), lo


def _matrix(comm, seed=0):
    full, local, lo = _local_columns(comm, seed)
    return full, local, lo, DistributedSparseMatrix.from_scipy(local, comm, lo)


@pytest.mark.parametrize("comm", COMMS)
def test_a_keyless_matrix_reports_its_size_and_conversions(comm):
    full, local, lo, mat = _matrix(comm)
    assert mat.shape == (N, local.shape[1])
    assert mat.nnz == local.nnz and mat.dtype == np.dtype(complex)
    assert mat.is_distributed == (comm is not None and comm.size > 1)
    assert mat.col_offset == lo
    np.testing.assert_array_equal(mat.to_scipy().toarray(), local.toarray())
    dense = np.zeros((N, N), dtype=complex)
    dense[:, lo : lo + local.shape[1]] = local.toarray()
    np.testing.assert_array_equal(mat.toarray(), dense)
    np.testing.assert_allclose(mat.diagonal(), np.diag(full)[lo : lo + local.shape[1]])


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("width", [1, WIDTH])
def test_apply_equals_the_dense_product_through_an_exchange_plan(comm, width):
    full, local, lo, mat = _matrix(comm, seed=1)
    rng = np.random.default_rng(5)
    x_full = rng.standard_normal((N, width)) + 1j * rng.standard_normal((N, width))
    x = np.ascontiguousarray(x_full[lo : lo + local.shape[1]])
    got = mat.apply(x, mat.exchange_plan(width, "contract"))
    np.testing.assert_allclose(got, (full @ x_full)[lo : lo + local.shape[1]], atol=1e-12)
    out = np.empty_like(got)
    assert mat.apply(x, None, out) is out
    np.testing.assert_allclose(out, got, atol=1e-13)


@pytest.mark.parametrize("comm", COMMS)
def test_coerce_returns_a_matrix_unchanged_and_converts_arrays_without_a_collective(comm):
    _full, _local, _lo, mat = _matrix(comm)
    assert DistributedSparseMatrix.coerce(mat, comm) is mat
    dense = DistributedSparseMatrix.coerce(
        np.arange(9.0).reshape(3, 3), None
    )  # rank-local: no communicator, no collective
    assert dense.shape == (3, 3) and dense.is_distributed is False
    np.testing.assert_array_equal(dense.toarray(), np.arange(9.0).reshape(3, 3))
    with pytest.raises(TypeError):
        DistributedSparseMatrix.coerce([[1.0]], None)


def test_a_keyless_matrix_has_no_keys_to_grow_or_name():
    m = DistributedSparseMatrix.from_dense(np.eye(3), None)
    for call in (lambda: m.grow([]), lambda: m.keys_of([0]), m.p_keys, lambda: m.admit_keys([])):
        with pytest.raises(TypeError, match="determinant keys"):
            call()


def test_a_matrix_whose_rows_do_not_add_up_is_refused():
    with pytest.raises(ValueError, match="rows"):
        DistributedSparseMatrix.from_scipy(sps.csc_array(np.ones((4, 3))), None)


def _whole(comm, local_columns):
    """The whole operator from every rank's columns."""
    if comm is None or comm.size == 1:
        return local_columns
    return np.concatenate(comm.allgather(local_columns), axis=1)


@pytest.mark.parametrize("comm", COMMS)
def test_the_matrix_built_on_a_basis_is_the_operator_in_the_bases_order(comm):
    dets = _all_dets()
    order = np.random.default_rng(3).permutation(len(dets))
    basis = _basis([dets[i] for i in order[:40]], comm)
    op = _operator()
    new = build_sparse_matrix(basis, op)
    assert new.p_keys() == list(basis.local_basis), "P slots follow the basis' key order"
    assert new.shape == (basis.size, len(basis.local_basis)) and new.col_offset == basis.offset
    oracle = _oracle_columns(basis, op, comm)
    np.testing.assert_allclose(new.to_scipy().toarray(), oracle, atol=1e-13)
    n_local = len(basis.local_basis)
    rows = basis.offset + np.arange(n_local)
    np.testing.assert_allclose(new.diagonal(), oracle[rows, np.arange(n_local)], atol=1e-13)
    x = np.random.default_rng(7).standard_normal((n_local, 2)) + 0j
    full_x = np.concatenate(comm.allgather(x)) if comm is not None and comm.size > 1 else x
    got = new.apply(x, new.exchange_plan(2, "contract"))
    np.testing.assert_allclose(got, (_whole(comm, oracle) @ full_x)[basis.offset : basis.offset + n_local], atol=1e-12)
