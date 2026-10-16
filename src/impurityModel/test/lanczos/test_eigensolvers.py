import numpy as np
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
