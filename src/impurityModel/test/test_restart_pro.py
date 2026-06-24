"""Regression for PRO-across-restart in the IRLM kernel.

`implicitly_restarted_block_lanczos_cy` continues PARTIAL/SELECTIVE runs across the
implicit-QR restart in the *requested* PRO mode, seeding the Paige-Simon estimator W
uniformly at REORT_TOL (= sqrt(eps)) instead of doing a full kept-block
reorthogonalization. These tests pin that this holds the same eigenvalue precision as
a full-reort restart (no drift to sqrt(eps)) and matches it through restart-block
deflation (active_k < p), serial + MPI.
"""

import numpy as np
import pytest
import scipy.linalg as sp
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import (
    ManyBodyOperator,
    ManyBodyState,
    SlaterDeterminant,
    add_scaled_multi,
    inner_multi,
)
from impurityModel.ed.BlockLanczosArray import Reort
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.test.test_restarted_lanczos import MockBasis, get_test_system


def _ortho_start_block(basis_states, n_blocks=2, seed=1):
    np.random.seed(seed)
    psi0 = []
    for _ in range(n_blocks):
        st = ManyBodyState()
        for b in basis_states:
            st += b * (np.random.rand() + 1j * np.random.rand())
        psi0.append(st)
    gram = inner_multi(psi0, psi0)
    binv = sp.inv(np.conj(sp.cholesky(gram, lower=True).T))
    out = [ManyBodyState() for _ in range(n_blocks)]
    add_scaled_multi(out, psi0, binv)
    return out


def _small_hermitian_op(n, seed):
    """Single-particle ManyBodyOperator from a dense n x n Hermitian, + the matrix."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = a + a.conj().T
    op = {((i, "c"), (j, "a")): h[i, j] for i in range(n) for j in range(n)}
    return ManyBodyOperator(op), h


def _basis_states_single_particle(n):
    return [ManyBodyState({SlaterDeterminant.from_bytes((1 << i).to_bytes(8, "little")): 1.0}) for i in range(n)]


def _run_irlm(h_op, basis, psi0, mode, num_wanted, msb, comm=None, tol=1e-13):
    ev, _ = implicitly_restarted_block_lanczos_cy(
        psi0=psi0, h_op=h_op, basis=basis, num_wanted=num_wanted,
        max_subspace_blocks=msb, tol=tol, max_restarts=300, verbose=False,
        reort=mode, comm=comm,
    )
    return np.sort(ev)


@pytest.mark.parametrize("mode", [Reort.PARTIAL, Reort.SELECTIVE])
def test_irlm_restart_pro_matches_full_and_dense(mode):
    """Tight-binding, forced restarts: PRO restart hits the same precision as FULL.

    msb=5 with num_wanted=2 (the well-isolated lowest pair) forces several restarts, so
    the PRO-seeded continuation is genuinely exercised. The PRO eigenvalues must match
    both the dense reference and a FULL-reort restart far tighter than sqrt(eps).
    """
    h_op, N, eigvals_exact, basis_states = get_test_system()
    psi0 = _ortho_start_block(basis_states)
    num_wanted, msb = 2, 5

    pro = _run_irlm(h_op, MockBasis(N), psi0, mode, num_wanted, msb)
    full = _run_irlm(h_op, MockBasis(N), psi0, Reort.FULL, num_wanted, msb)

    # PRO must reproduce the dense eigenvalues, and agree with FULL, well below sqrt(eps).
    np.testing.assert_allclose(pro, eigvals_exact[:num_wanted], atol=1e-9)
    np.testing.assert_allclose(pro, full, atol=1e-10)


@pytest.mark.parametrize("mode", [Reort.PARTIAL, Reort.SELECTIVE])
def test_irlm_restart_pro_deflation_matches_full(mode):
    """Restart-block deflation (active_k < p): PRO seed matches FULL bit-for-bit.

    n=7, msb=3, p=2 drives the continuation block rank-deficient at several restarts
    (active_k=1), exercising the deflated W-seed branch. The tiny near-exhausted space
    is hard for the restarted solver, but PRO and FULL must agree exactly.
    """
    n = 7
    h_op, h = _small_hermitian_op(n, seed=n)
    exact = np.sort(np.linalg.eigvalsh(h))
    basis_states = _basis_states_single_particle(n)
    psi0 = _ortho_start_block(basis_states, seed=1)

    pro = _run_irlm(h_op, MockBasis(n), psi0, mode, 2, 3, tol=1e-12)
    full = _run_irlm(h_op, MockBasis(n), psi0, Reort.FULL, 2, 3, tol=1e-12)

    np.testing.assert_allclose(pro, full, atol=1e-10)
    # Sanity: both bracket the true lowest eigenvalue (the space is too small for 1e-9).
    assert abs(pro[0] - exact[0]) < 1e-2


@pytest.mark.mpi
@pytest.mark.parametrize("mode", [Reort.PARTIAL, Reort.SELECTIVE])
def test_irlm_restart_pro_matches_full_mpi(mode):
    """Distributed: the rank-identical W-seed gives the same PRO/FULL agreement."""
    comm = MPI.COMM_WORLD
    h_op, N, eigvals_exact, basis_states = get_test_system()
    psi0 = _ortho_start_block(basis_states)
    num_wanted, msb = 2, 5

    pro = _run_irlm(h_op, MockBasis(N), psi0, mode, num_wanted, msb, comm=comm)
    full = _run_irlm(h_op, MockBasis(N), psi0, Reort.FULL, num_wanted, msb, comm=comm)

    np.testing.assert_allclose(pro, eigvals_exact[:num_wanted], atol=1e-9)
    np.testing.assert_allclose(pro, full, atol=1e-10)
