import numpy as np
import pytest
import scipy.sparse


def test_eigensystem_none_e_max_keeps_all_states():
    """e_max=None means no energy cutoff; it must not raise (max(None, ...) TypeError).

    get_eigenvectors calls eigensystem with e_max=max_energy=None on the dense path
    (basis < dense_cutoff). Regression: that used to crash in max(e_max, ...).
    """
    from impurityModel.ed.eigensolvers import eigensystem

    N = 8
    np.random.seed(0)
    H = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    H = H + H.conj().T
    op = scipy.sparse.csr_matrix(H)

    es, vs = eigensystem(op, e_max=None)
    # No cutoff -> the whole computed spectrum comes back, sorted, matching dense eigh.
    ref = np.linalg.eigvalsh(H)
    assert len(es) == N
    assert np.allclose(es, ref)
    assert vs.shape == (N, N)


@pytest.mark.mpi
def test_dense_lowest_returns_a_lowest_level_wider_than_its_first_subset_whole():
    """``dense_lowest`` computes only the lowest eigenpairs, doubling the count while the lowest level
    fills them: a 20-fold lowest level (more than the first 16) comes back whole, with its vectors,
    on every rank -- rank 0 decides the count and broadcasts it. The matrix is split across ranks
    (summed on rank 0), as the CIPSI dense branch hands it in."""
    from mpi4py import MPI

    from impurityModel.ed.eigensolvers import dense_lowest

    comm = MPI.COMM_WORLD
    n, degenerate = 60, 20
    rng = np.random.default_rng(7)
    levels = np.concatenate([np.full(degenerate, -2.0), np.linspace(-1.0, 3.0, n - degenerate)])
    unitary, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    h = unitary @ np.diag(levels) @ unitary.conj().T
    h = 0.5 * (h + h.conj().T)
    local = h if comm.rank == 0 else np.zeros_like(h)

    def level_size(es):
        return int(np.count_nonzero(es - es[0] < 1e-8))

    es, vecs = dense_lowest(local, level_size, comm=comm)
    assert len(es) > degenerate and vecs.shape == (n, len(es))
    np.testing.assert_allclose(es[:degenerate], -2.0, atol=1e-10)
    assert es[degenerate] > -1.5
    np.testing.assert_allclose(h @ vecs[:, :degenerate], -2.0 * vecs[:, :degenerate], atol=1e-9)
    assert len(set(comm.allgather(len(es)))) == 1


def _split_columns(h, comm):
    """This rank's columns of ``h`` as a ``DistributedSparseMatrix``: rank-contiguous blocks, rank 1
    owning none at three ranks or more, and ``col_offset`` left at 0 -- the gather must not need it."""
    import scipy.sparse as sps

    from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix

    n = h.shape[0]
    weights = np.array([0 if (r == 1 and comm.size >= 3) else 1 for r in range(comm.size)], dtype=float)
    counts = np.floor(n * weights / weights.sum()).astype(int)
    counts[0] += n - counts.sum()
    start = int(counts[: comm.rank].sum())
    cols = slice(start, start + int(counts[comm.rank]))
    return DistributedSparseMatrix.from_scipy(sps.csc_array(h[:, cols]), comm), cols


def _hermitian(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    a[np.abs(a) < 1.2] = 0.0  # sparse-ish
    return 0.1 * (a + a.conj().T) + np.diag(np.linspace(0.0, 5.0, n))


@pytest.mark.mpi
def test_a_distributed_matrix_is_gathered_onto_rank_0_whole_and_solved_there():
    """``dense_eigensystem`` / ``dense_lowest`` send each rank's sparse columns to rank 0 (``O(nnz)``,
    not an ``n x n`` reduction), placed by the ranks' local sizes; an empty rank takes part."""
    from mpi4py import MPI

    from impurityModel.ed.eigensolvers import dense_eigensystem, dense_lowest

    comm = MPI.COMM_WORLD
    h = _hermitian(40, 3)
    exact = np.linalg.eigvalsh(h)
    h_dist, _cols = _split_columns(h, comm)
    es, vecs = dense_eigensystem(h_dist, True, comm)
    np.testing.assert_allclose(es, exact, atol=1e-10)
    np.testing.assert_allclose(h @ vecs, vecs * es, atol=1e-9)
    es, vecs = dense_lowest(h_dist, lambda e: 1, comm=comm)
    np.testing.assert_allclose(es[0], exact[0], atol=1e-10)
    np.testing.assert_allclose(h @ vecs[:, 0], es[0] * vecs[:, 0], atol=1e-9)


@pytest.mark.mpi
@pytest.mark.parametrize("solver", ["davidson", "trlm"])
def test_a_serial_lowest_solve_on_rank_0_returns_each_rank_its_rows(solver, monkeypatch):
    """``solve_lowest(serial=True)`` gathers the operator, hashes and warm column to rank 0, solves there
    and scatters the rows back: the same exact pair at every rank count, an empty rank included."""
    from mpi4py import MPI

    from impurityModel.ed.eigensolvers import solve_lowest

    monkeypatch.setenv("GS_LOWEST_SOLVER", solver)
    comm = MPI.COMM_WORLD
    n = 40
    h = _hermitian(n, 5)
    exact = np.linalg.eigvalsh(h)[0]
    h_dist, cols = _split_columns(h, comm)
    hashes = np.arange(n, dtype=np.uint64)[cols] * np.uint64(2654435761)
    eig = solve_lowest(h_dist, comm, hashes, warm=None, tol=1e-10, n_global=n, serial=True)
    assert eig.certified
    np.testing.assert_allclose(eig.e[0], exact, atol=1e-9)
    x = np.concatenate(comm.allgather(eig.vecs[:, 0]))
    np.testing.assert_allclose(h @ x, eig.e[0] * x, atol=1e-8)
    assert len(set(comm.allgather(float(eig.e[0])))) == 1


@pytest.mark.mpi
def test_a_failure_on_rank_0_is_raised_on_every_rank():
    """A root-only computation that raises does not leave the other ranks waiting for its result."""
    from mpi4py import MPI

    from impurityModel.ed.eigensolvers import _on_root

    comm = MPI.COMM_WORLD

    def fails():
        raise np.linalg.LinAlgError("eigh did not converge")

    with pytest.raises((np.linalg.LinAlgError, RuntimeError), match="did not converge"):
        _on_root(comm, fails)
    assert _on_root(comm, lambda: 7) == (7 if comm.rank == 0 else None)
