r"""Shrinking-block deflation consistency across all ``block_widths`` consumers (WS3).

When the block Krylov space saturates an invariant subspace whose dimension is not a multiple
of the block size, an interior block deflates and every downstream consumer of the coefficients
must interpret the *true* per-block widths identically:

* ``_build_full_T`` / ``eigsh`` — the eigenvalue path. ``eigsh`` previously used a uniform
  banded build that padded deflated blocks with zeros, injecting a **spurious zero eigenvalue**
  (and breaking the Ritz reconstruction ``Q @ eigvecs``); it now honors ``block_widths``.
* ``_trim_blocks`` / ``_block_cf_inverse`` (``calc_G``) — the Green's-function continued
  fraction (already covered by ``test_greens_function_deflation``; re-checked here per reort
  mode).

These run a dim-5 problem with block size 2 (forcing widths ``[2, 2, 1]``) through both the
array and the ManyBodyState kernels, under every reorthogonalization mode, and check the
eigenvalues and ``G`` against the exact diagonal reference.
"""

import numpy as np
import pytest
import scipy.linalg as la

from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import (
    Reort,
    _build_banded_lower,
    _build_full_T,
    block_lanczos_array,
    eigsh,
)
from impurityModel.ed.greens_function import _trim_blocks, build_qr, calc_G
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.test.test_restarted_lanczos import MockBasis

EPS = np.array([0.3, 0.9, 1.5, 2.2, 3.1])  # 5 distinct energies; block size 2 -> interior deflation
N_ORB = len(EPS)
P = 2
_MODES = [Reort.NONE, Reort.PARTIAL, Reort.FULL]


def _exact_G(S, omega, delta):
    denom = omega[:, None] + 1j * delta - EPS[None, :]
    return np.einsum("wi,ia,ib->wab", 1.0 / denom, np.conj(S), S)


@pytest.mark.parametrize("mode", _MODES)
def test_eigsh_with_widths_no_spurious_zero(mode):
    """Array kernel + eigsh: interior deflation must not produce a spurious zero eigenvalue,
    and the Ritz vectors must be reconstructed at the right dimension."""
    H = np.diag(EPS).astype(complex)
    rng = np.random.default_rng(1)
    Q0 = la.qr(rng.standard_normal((N_ORB, P)) + 1j * rng.standard_normal((N_ORB, P)), mode="economic")[0]
    a, b, Q, widths = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H,
        converged=lambda *x, **k: False,
        reort=mode,
        return_widths=True,
        max_iter=8,
    )
    assert min(widths) < P, f"expected interior deflation, widths={list(widths)}"

    evals = np.sort(
        np.asarray(eigsh(np.asarray(a), np.asarray(b), eigvals_only=True, block_widths=list(widths))[0]).real
    )
    np.testing.assert_allclose(evals, EPS, atol=1e-8)
    assert not np.any(np.abs(evals) < 1e-9)  # no spurious zero (0 is not in EPS)

    evals_v, evecs = eigsh(np.asarray(a), np.asarray(b), Q=Q, block_widths=list(widths))
    assert evecs.shape == (N_ORB, len(evals_v))
    res = np.linalg.norm(H @ evecs[:, :1] - evals_v[0] * evecs[:, :1])
    assert res < 1e-10


@pytest.mark.parametrize("mode", _MODES)
def test_array_gf_under_deflation(mode):
    """Array kernel: the continued-fraction G from deflated (trimmed) blocks matches exact G."""
    H = np.diag(EPS).astype(complex)
    rng = np.random.default_rng(7)
    S = rng.standard_normal((N_ORB, P)) + 1j * rng.standard_normal((N_ORB, P))
    Q0, r = build_qr(S)
    a, b, _, widths = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H,
        converged=lambda *x, **k: False,
        reort=mode,
        return_widths=True,
        max_iter=8,
    )
    assert min(widths) < P
    omega = np.linspace(-1.0, 4.0, 21)
    delta = 0.1
    at, bt = _trim_blocks(a, b, widths)
    np.testing.assert_allclose(calc_G(at, bt, r, omega, 0.0, delta), _exact_G(S, omega, delta), atol=1e-10)


def _mbs_states():
    return [SlaterDeterminant.from_bytes((1 << (7 - i)).to_bytes(8, "little")) for i in range(N_ORB)]


@pytest.mark.parametrize("mode", ["none", "partial", "full"])
def test_mbs_gf_under_deflation(mode):
    """ManyBodyState kernel: interior deflation -> trimmed calc_G matches exact G, every mode."""
    h_op = ManyBodyOperator({((i, "c"), (i, "a")): EPS[i] for i in range(N_ORB)})
    states = _mbs_states()
    rng = np.random.default_rng(7)
    S = rng.standard_normal((N_ORB, P)) + 1j * rng.standard_normal((N_ORB, P))
    Q0, r = build_qr(S)
    psi0 = [ManyBodyState() for _ in range(P)]
    for c in range(P):
        for i, sd in enumerate(states):
            psi0[c] += ManyBodyState({sd: Q0[i, c]})

    a, b, _, _, widths = block_lanczos_cy(
        psi0,
        h_op,
        MockBasis(N_ORB),
        converged_fn=lambda a, b, **kw: False,
        reort=mode,
        max_iter=8,
        return_widths=True,
    )
    assert min(widths) < P, f"expected deflation, widths={list(widths)}"
    omega = np.linspace(-1.0, 4.0, 21)
    delta = 0.1
    at, bt = _trim_blocks(a, b, widths)
    np.testing.assert_allclose(calc_G(at, bt, r, omega, 0.0, delta), _exact_G(S, omega, delta), atol=1e-10)


def _band_to_dense(a_band, total):
    """Reconstruct the Hermitian matrix from its lower-banded storage (for the assertion)."""
    T = np.zeros((total, total), dtype=complex)
    for j in range(total):
        for d in range(a_band.shape[0]):
            if j + d < total:
                T[j + d, j] = a_band[d, j]
                T[j, j + d] = np.conj(a_band[d, j])
    return T


def test_banded_build_matches_dense_T_and_no_dense_path():
    """The deflated eigensolve must use the *banded* T built straight from alphas/betas (no
    dense matrix). Pin that the band reconstructs the dense T exactly and yields the dense
    eigenvalues, for block size 3 (lower bandwidth up to 2*p-1) with interior deflation."""
    eps3 = np.array([0.2, 0.6, 1.1, 1.7, 2.4, 3.0, 3.9])  # dim 7, p=3 -> deflation
    H = np.diag(eps3).astype(complex)
    rng = np.random.default_rng(2)
    Q0 = la.qr(rng.standard_normal((7, 3)) + 1j * rng.standard_normal((7, 3)), mode="economic")[0]
    a, b, _, widths = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H,
        converged=lambda *x, **k: False,
        reort=Reort.FULL,
        return_widths=True,
        max_iter=8,
    )
    assert min(widths) < 3  # deflation occurred

    a_band, total = _build_banded_lower(np.asarray(a), np.asarray(b), list(widths))
    T_dense = _build_full_T(np.asarray(a), np.asarray(b), block_widths=list(widths))
    assert total == T_dense.shape[0]
    np.testing.assert_allclose(_band_to_dense(a_band, total), T_dense, atol=1e-13)

    evals = np.sort(
        np.asarray(eigsh(np.asarray(a), np.asarray(b), eigvals_only=True, block_widths=list(widths))[0]).real
    )
    np.testing.assert_allclose(evals, eps3, atol=1e-8)
