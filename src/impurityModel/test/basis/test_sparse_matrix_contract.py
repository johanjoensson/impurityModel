r"""``DiscoveredMatrix`` honours the contract of the sparse operator the eigensolvers and the array kernels read.

The oracle is the class it is to replace (``DistributedSparseMatrix``), on the same matrix: ``shape``, ``nnz``,
``apply`` through ``exchange_plan``, ``to_scipy``, ``toarray``, ``diagonal`` and ``col_offset``, for a matrix built
from scipy/dense data (keyless) and for one built on a basis from an operator (``K = P``, the ``P H P`` matrix, with
its P slots in the basis' key order). Both at every rank count, an empty rank included.
"""

import numpy as np
import pytest
import scipy.sparse as sps

from impurityModel.ed.basis_transcription import build_sparse_matrix
from impurityModel.ed.discovered_matrix import DiscoveredMatrix
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.test.basis.test_sparse_matrix_build import _all_dets, _basis, _operator
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


def _both(comm, seed=0):
    full, local, lo = _local_columns(comm, seed)
    new = DiscoveredMatrix.from_scipy(local, comm, lo)
    old = DistributedSparseMatrix.from_scipy(local, comm, lo)
    return full, local, lo, new, old


@pytest.mark.parametrize("comm", COMMS)
def test_a_keyless_matrix_has_the_old_matrixs_size_and_conversions(comm):
    full, local, lo, new, old = _both(comm)
    assert new.shape == old.shape == (N, local.shape[1])
    assert new.nnz == old.nnz and new.dtype == old.dtype
    assert new.is_distributed == old.is_distributed
    assert new.col_offset == lo
    np.testing.assert_array_equal(new.to_scipy().toarray(), old.to_scipy().toarray())
    np.testing.assert_array_equal(new.toarray(), old.toarray())
    np.testing.assert_allclose(new.diagonal(), np.diag(full)[lo : lo + local.shape[1]])


@pytest.mark.parametrize("comm", COMMS)
@pytest.mark.parametrize("width", [1, WIDTH])
def test_apply_equals_the_old_matrixs_through_an_exchange_plan(comm, width):
    full, local, lo, new, old = _both(comm, seed=1)
    rng = np.random.default_rng(5)
    x_full = rng.standard_normal((N, width)) + 1j * rng.standard_normal((N, width))
    x = np.ascontiguousarray(x_full[lo : lo + local.shape[1]])
    got = new.apply(x, new.exchange_plan(width, "contract"))
    want = old.apply(x, old.exchange_plan(width, "contract"))
    np.testing.assert_allclose(got, want, atol=1e-13)
    np.testing.assert_allclose(got, (full @ x_full)[lo : lo + local.shape[1]], atol=1e-12)
    out = np.empty_like(got)
    assert new.apply(x, None, out) is out
    np.testing.assert_allclose(out, want, atol=1e-13)


@pytest.mark.parametrize("comm", COMMS)
def test_coerce_returns_a_matrix_unchanged_and_converts_arrays_without_a_collective(comm):
    _full, _local, _lo, new, _old = _both(comm)
    assert DiscoveredMatrix.coerce(new, comm) is new
    dense = DiscoveredMatrix.coerce(np.arange(9.0).reshape(3, 3), None)  # rank-local: no communicator, no collective
    assert dense.shape == (3, 3) and dense.is_distributed is False
    np.testing.assert_array_equal(dense.toarray(), np.arange(9.0).reshape(3, 3))
    with pytest.raises(TypeError):
        DiscoveredMatrix.coerce([[1.0]], None)


def test_a_keyless_matrix_has_no_keys_to_grow_or_name():
    m = DiscoveredMatrix.from_dense(np.eye(3), None)
    for call in (lambda: m.grow([]), lambda: m.keys_of([0]), m.p_keys, lambda: m.admit_keys([])):
        with pytest.raises(TypeError, match="determinant keys"):
            call()


def test_a_matrix_whose_rows_do_not_add_up_is_refused():
    with pytest.raises(ValueError, match="rows"):
        DiscoveredMatrix.from_scipy(sps.csc_array(np.ones((4, 3))), None)


def _on_basis(comm, n=40, seed=3):
    dets = _all_dets()
    order = np.random.default_rng(seed).permutation(len(dets))
    basis = _basis([dets[i] for i in order[:n]], comm)
    op = _operator()
    new = DiscoveredMatrix(
        op, (basis.n_bytes + 7) // 8, comm if comm is not None and comm.size > 1 else None, keep_frontier=False
    )
    new.admit_keys(list(basis.local_basis))
    new.build_columns()
    return basis, op, new, build_sparse_matrix(basis, op)


@pytest.mark.parametrize("comm", COMMS)
def test_the_matrix_built_on_a_basis_is_the_old_builders_in_the_bases_order(comm):
    basis, _op, new, old = _on_basis(comm)
    assert new.p_keys() == list(basis.local_basis), "P slots follow the basis' key order"
    assert new.shape == old.shape and new.col_offset == basis.offset and new.nnz == old.nnz
    np.testing.assert_allclose(new.to_scipy().toarray(), old.to_scipy().toarray(), atol=1e-13)
    n_local = len(basis.local_basis)
    rows = basis.offset + np.arange(n_local)
    np.testing.assert_allclose(new.diagonal(), old.to_scipy().toarray()[rows, np.arange(n_local)], atol=1e-13)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((len(basis.local_basis), 2)) + 0j
    np.testing.assert_allclose(
        new.apply(x, new.exchange_plan(2, "contract")), old.apply(x, old.exchange_plan(2, "contract")), atol=1e-13
    )
