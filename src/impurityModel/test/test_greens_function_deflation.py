"""Green's function under interior block deflation (shrinking-block Lanczos).

When the block Krylov space saturates an H-invariant subspace whose dimension is
not a multiple of the block size ``p``, an *interior* Lanczos block deflates: the
off-diagonal coupling ``beta_i`` becomes rectangular ``(p_{i+1}, p_i)`` with
``p_{i+1} < p_i`` and the following diagonal block ``alpha_{i+1}`` shrinks to
``(p_{i+1}, p_{i+1})``.

The kernels store these into the fixed ``(P, P)`` pre-allocated buffers, zero-padding
the inactive rows/columns. If the continued-fraction Green's function consumes those
padded blocks as if they were full ``(P, P)`` couplings it produces a *wrong* ``G``
(a spurious zero-energy pole appears in the padded diagonal slot). The fix trims each
block back to its true dimension (via the returned ``block_widths``) and builds ``G``
without assuming fixed block dimensions.

These tests build a tiny diagonal Hamiltonian whose Hilbert space has dimension 5 with
a block size of 2, forcing the block sizes ``[2, 2, 1]`` (interior deflation), and
check that the resulting ``calc_G`` reproduces the exact diagonal resolvent
``G_{ab}(w) = sum_i conj(S_{ia}) S_{ib} / (w + i*delta - eps_i)``.
"""

import numpy as np

from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import Reort, block_lanczos_array
from impurityModel.ed.greens_function import _trim_blocks, build_qr, calc_G
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant
from impurityModel.test.test_restarted_lanczos import MockBasis

# Diagonal one-body energies on 5 orbitals; single particle -> 5 determinants,
# 5-dimensional Hilbert space, all distinct energies (no accidental degeneracy).
EPS = np.array([0.3, 0.9, 1.5, 2.2, 3.1])
N_ORB = len(EPS)
P = 2  # block size: 5 is not a multiple of 2 -> interior deflation


def _diagonal_h():
    return ManyBodyOperator({((i, "c"), (i, "a")): EPS[i] for i in range(N_ORB)})


def _basis_states():
    # Orbital i maps to bit (7 - i) (MSB-first within the byte, the codebase
    # convention where orbital 0 == 0x80), so n_i = c_i^dag a_i acts as eps_i on det_i.
    states = []
    for i in range(N_ORB):
        val = 1 << (7 - i)
        states.append(SlaterDeterminant.from_bytes(val.to_bytes(8, byteorder="little")))
    return states


def _assert_h_is_diagonal(h_op, states):
    """Sanity check: H restricted to these determinants is exactly diag(EPS)."""
    cols = h_op.apply_multi([ManyBodyState({sd: 1.0}) for sd in states])

    def _amp(j, i):
        val = cols[j].get(states[i])
        return 0.0 if val is None else val[0]

    H = np.array([[_amp(j, i) for j in range(N_ORB)] for i in range(N_ORB)])
    np.testing.assert_allclose(H, np.diag(EPS), atol=1e-12)


def _seed_block(rng):
    """Generic full-rank seed block S (N_ORB x P) and its QR factor r."""
    S = rng.standard_normal((N_ORB, P)) + 1j * rng.standard_normal((N_ORB, P))
    Q0, r = build_qr(S)  # S = Q0 @ r, Q0 orthonormal columns
    return S, Q0, r


def _exact_G(S, omega, delta):
    """Exact diagonal resolvent G_{ab}(w) = S^H (w + i*delta - diag(eps))^{-1} S."""
    denom = omega[:, None] + 1j * delta - EPS[None, :]  # (n_w, N_ORB)
    # G[w] = sum_i (1/denom[w,i]) * outer(conj(S[i,:]), S[i,:])
    inv = 1.0 / denom  # (n_w, N_ORB)
    return np.einsum("wi,ia,ib->wab", inv, np.conj(S), S)


def test_interior_deflation_mbs_path():
    """ManyBodyState kernel: interior deflation -> trimmed calc_G matches exact G."""
    rng = np.random.default_rng(7)
    h_op = _diagonal_h()
    states = _basis_states()
    _assert_h_is_diagonal(h_op, states)
    S, Q0, r = _seed_block(rng)

    psi0 = [ManyBodyState() for _ in range(P)]
    for c in range(P):
        for i, sd in enumerate(states):
            psi0[c] += ManyBodyState({sd: Q0[i, c]})

    alphas, betas, _, _, widths = block_lanczos_cy(
        psi0,
        h_op,
        MockBasis(N_ORB),
        converged_fn=lambda a, b, **kw: False,
        reort=Reort.NONE,
        max_iter=8,
        return_widths=True,
    )

    # Interior deflation must actually have happened for this to be a real test.
    assert min(widths) < P, f"expected a deflated block, got widths={widths}"
    assert list(widths)[:3] == [2, 2, 1], f"unexpected block structure widths={widths}"

    omega = np.linspace(-1.0, 4.0, 11)
    delta = 0.1
    a_t, b_t = _trim_blocks(alphas, betas, widths)
    G = calc_G(a_t, b_t, r, omega, 0.0, delta)
    np.testing.assert_allclose(G, _exact_G(S, omega, delta), atol=1e-10)


def test_interior_deflation_array_path():
    """Array kernel: interior deflation -> trimmed calc_G matches exact G."""
    rng = np.random.default_rng(11)
    H = np.diag(EPS).astype(complex)
    S, Q0, r = _seed_block(rng)

    alphas, betas, _, widths = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H,
        converged=lambda a, b, **kw: False,
        reort=Reort.NONE,
        return_widths=True,
    )

    assert min(widths) < P, f"expected a deflated block, got widths={widths}"

    omega = np.linspace(-1.0, 4.0, 11)
    delta = 0.1
    a_t, b_t = _trim_blocks(alphas, betas, widths)
    G = calc_G(a_t, b_t, r, omega, 0.0, delta)
    np.testing.assert_allclose(G, _exact_G(S, omega, delta), atol=1e-10)


def test_calc_G_uniform_still_works():
    """No deflation: calc_G on a plain (k, p, p) ndarray is unchanged (regression)."""
    rng = np.random.default_rng(3)
    H = np.diag(np.array([0.3, 0.9, 1.5, 2.2])).astype(complex)  # dim 4, p=2 -> no deflation
    S = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    Q0, r = build_qr(S)

    alphas, betas, _, widths = block_lanczos_array(
        psi0=Q0.copy(),
        h_op=H,
        converged=lambda a, b, **kw: False,
        reort=Reort.NONE,
        return_widths=True,
    )
    assert min(widths) == P  # genuinely no deflation

    omega = np.linspace(-1.0, 3.0, 9)
    delta = 0.1
    # Pass the raw (uniform) ndarray blocks straight through.
    G = calc_G(alphas, betas, r, omega, 0.0, delta)
    denom = omega[:, None] + 1j * delta - np.diag(H).real[None, :]
    G_exact = np.einsum("wi,ia,ib->wab", 1.0 / denom, np.conj(S), S)
    np.testing.assert_allclose(G, G_exact, atol=1e-10)
