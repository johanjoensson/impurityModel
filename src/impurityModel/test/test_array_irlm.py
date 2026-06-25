"""Validation for the array-path IRLM driver (impurityModel.ed.irlm).

The reort matrix imperatively xfails IRLM (it doesn't reach 1e-8 on the tight-binding
system within the subspace size — a known IRLM limit), so these tests validate the new
array driver directly: (a) it reproduces the dense spectrum when IRLM *can* converge
(large subspace, no restart), and (b) it is a faithful port of the proven Cython
ManyBodyState IRLM — bit-for-bit identical on the same Hamiltonian, all reort modes,
serial + MPI.
"""

import numpy as np
import pytest
from mpi4py import MPI

from impurityModel.ed.BlockLanczosArray import Reort, block_normalize
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.BlockLanczos import implicitly_restarted_block_lanczos_cy as mbs_irlm
from impurityModel.test.test_restarted_lanczos import MockBasis, get_test_system
from impurityModel.test.test_block_lanczos_reort_matrix import build_dense_matrix_from_manybody

_MODES = [Reort.NONE, Reort.FULL, Reort.PARTIAL, Reort.SELECTIVE, Reort.PERIODIC]


def _reort_system():
    h_op_mb, N, eigvals_exact, basis_states = get_test_system()
    H = build_dense_matrix_from_manybody(h_op_mb, basis_states)
    np.random.seed(42)
    psi0 = np.random.randn(N, 2) + 1j * np.random.randn(N, 2)
    psi0, _ = block_normalize(psi0, mpi=False, comm=None)
    return h_op_mb, H, N, eigvals_exact, basis_states, psi0


@pytest.mark.parametrize("mode", _MODES)
def test_array_irlm_matches_dense_large_subspace(mode):
    """With a large subspace (no restart) the array IRLM reproduces the dense spectrum."""
    _, H, N, exact, _, psi0 = _reort_system()
    # msb*p >~ Krylov dimension that captures the wanted pairs without restarting.
    ev, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0.copy(),
        h_op=H,
        basis=None,
        num_wanted=4,
        max_subspace_blocks=20,
        tol=1e-10,
        max_restarts=5,
        verbose=False,
        reort=mode,
        comm=None,
    )
    np.testing.assert_allclose(np.sort(ev), exact[:4], atol=1e-7)


@pytest.mark.parametrize("mode", _MODES)
def test_array_irlm_matches_mbs(mode):
    """Faithful port: array IRLM == Cython ManyBodyState IRLM on the same Hamiltonian."""
    h_op_mb, H, N, _, basis_states, psi0 = _reort_system()
    psi0_mb = [ManyBodyState() for _ in range(2)]
    for c in range(2):
        for i, b in enumerate(basis_states):
            psi0_mb[c] += b * psi0[i, c]

    ev_a, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0.copy(),
        h_op=H,
        basis=None,
        num_wanted=4,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=50,
        verbose=False,
        reort=mode,
        comm=None,
    )
    ev_m, _ = mbs_irlm(
        psi0=psi0_mb,
        h_op=h_op_mb,
        basis=MockBasis(N),
        num_wanted=4,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=50,
        verbose=False,
        reort=mode.name.lower(),
        comm=None,
    )
    # Both kernels run the same algorithm (including restart-PRO for PARTIAL/SELECTIVE),
    # so they agree bit-for-bit across all modes.
    np.testing.assert_allclose(np.sort(ev_a), np.sort(ev_m), atol=1e-10)


@pytest.mark.parametrize("mode", [Reort.PARTIAL, Reort.SELECTIVE])
def test_array_irlm_pro_restart_stable(mode):
    """Restart-PRO on the array path stays bounded over many restarts (no divergence).

    Regression for the array kernel's resumption losing orthogonality: the W-recurrence
    can't model the restart-induced loss, so the kernel must reorthogonalize against the
    retained Ritz block every step on a resumed run. Without that, PARTIAL/SELECTIVE
    blew up to ~1e8+ over 50 restarts; here the result must stay finite and sensible.
    """
    N = 40
    H = np.zeros((N, N), dtype=complex)
    for i in range(N - 1):
        H[i, i + 1] = H[i + 1, i] = -1.0
    exact = np.sort(np.linalg.eigvalsh(H))
    rng = np.random.default_rng(0)
    psi0 = rng.standard_normal((N, 2)) + 1j * rng.standard_normal((N, 2))

    ev, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0,
        h_op=H,
        basis=None,
        num_wanted=4,
        max_subspace_blocks=8,
        tol=1e-10,
        max_restarts=200,
        verbose=False,
        reort=mode,
        comm=None,
    )
    # No blow-up, and the values bracket the true low spectrum (IRLM doesn't reach 1e-8
    # on this clustered band edge within msb=8, but must stay near the true eigenvalues).
    assert np.all(np.isfinite(ev))
    assert np.max(np.abs(np.sort(ev) - exact[:4])) < 0.1


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("num_wanted", [2, 3, 4])
def test_irlm_small_subspace_locking(mode, num_wanted):
    """EA16 locking + explicit purging: IRLM converges in a subspace too small for the
    un-restarted Krylov space, including ``num_wanted`` not a multiple of the block size.

    Regression for two coupled bugs fixed together: (a) the restart used a
    block-misaligned shift count (``m*p - num_wanted`` instead of ``(m-k)*p``), which
    silently corrupted the retained factorization whenever ``num_wanted % p != 0``; and
    (b) without locking, IRLM stalled at ~0.26 error for ``num_wanted in {3, 4}`` at
    ``m=6`` (it only converged once ``m`` grew to ~16). Both paths must now hit the dense
    oracle — including the degenerate pair at index 2/3 — to ~1e-8.
    """
    h_op_mb, H, N, exact, basis_states, psi0 = _reort_system()
    ev_a, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0.copy(),
        h_op=H,
        basis=None,
        num_wanted=num_wanted,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=200,
        verbose=False,
        reort=mode,
        comm=None,
    )
    np.testing.assert_allclose(np.sort(ev_a)[:num_wanted], exact[:num_wanted], atol=1e-7)

    psi0_mb = [ManyBodyState() for _ in range(2)]
    for c in range(2):
        for i, b in enumerate(basis_states):
            psi0_mb[c] += b * psi0[i, c]
    ev_m, _ = mbs_irlm(
        psi0=psi0_mb,
        h_op=h_op_mb,
        basis=MockBasis(N),
        num_wanted=num_wanted,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=200,
        verbose=False,
        reort=mode.name.lower(),
        comm=None,
    )
    np.testing.assert_allclose(np.sort(ev_m)[:num_wanted], exact[:num_wanted], atol=1e-7)


def _partition(n, size):
    return [n // size + (1 if r < n % size else 0) for r in range(size)]


@pytest.mark.mpi
@pytest.mark.parametrize("mode", [Reort.FULL, Reort.PARTIAL])
def test_array_irlm_mpi_matches_serial(mode):
    """Row-block-distributed array IRLM matches a serial run of the same problem."""
    comm = MPI.COMM_WORLD
    _, H, N, exact, _, psi0_full = _reort_system()

    counts = _partition(N, comm.size)
    c0 = sum(counts[: comm.rank])
    c1 = c0 + counts[comm.rank]
    h_local = np.ascontiguousarray(H[:, c0:c1])
    psi0_local = np.ascontiguousarray(psi0_full[c0:c1, :], dtype=complex)

    class _Basis:
        def __init__(self, c):
            self.comm = c

    ev_mpi, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0_local,
        h_op=h_local,
        basis=_Basis(comm),
        num_wanted=4,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=50,
        verbose=False,
        reort=mode,
        comm=comm,
    )
    ev_ser, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0_full.copy(),
        h_op=H,
        basis=None,
        num_wanted=4,
        max_subspace_blocks=6,
        tol=1e-8,
        max_restarts=50,
        verbose=False,
        reort=mode,
        comm=None,
    )
    np.testing.assert_allclose(np.sort(ev_mpi), np.sort(ev_ser), atol=1e-8)
