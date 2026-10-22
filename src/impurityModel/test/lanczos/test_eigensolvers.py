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
