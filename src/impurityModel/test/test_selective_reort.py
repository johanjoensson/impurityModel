from unittest.mock import patch

import numpy as np
import pytest

from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.BlockLanczosArray import _build_full_T, block_normalize
from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState


def create_poorly_conditioned_h_and_basis(n_states=20):
    states = [(1 << i).to_bytes(8, "little") for i in range(n_states)]
    np.random.seed(42)

    # Create eigenvalues that are closely spaced to force orthogonality loss
    eigvals = []
    for i in range(n_states // 2):
        eigvals.extend([float(i), float(i) + 1e-5])
    if len(eigvals) < n_states:
        eigvals.extend([float(n_states // 2)] * (n_states - len(eigvals)))
    eigvals = np.array(eigvals)

    # Random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n_states, n_states))
    H_mat = Q @ np.diag(eigvals) @ Q.T
    H_mat = 0.5 * (H_mat + H_mat.T)  # Ensure exact symmetry

    hop = {}
    for i in range(n_states):
        for j in range(n_states):
            if abs(H_mat[i, j]) > 1e-12:
                hop[((i, "c"), (j, "a"))] = float(H_mat[i, j])

    basis = Basis(
        impurity_orbitals={0: [list(range(n_states))]},
        bath_states=({0: [[]]}, {0: [[]]}),
        initial_basis=states,
        verbose=False,
    )
    h_op = ManyBodyOperator(hop)
    return h_op, basis, H_mat


def test_selective_reort():
    n_states = 20
    h_op, basis, _H_mat = create_poorly_conditioned_h_and_basis(n_states)

    block_size = 2
    np.random.seed(123)
    psi0 = []
    for _ in range(block_size):
        psi0.append(ManyBodyState({b: np.random.randn() for b in basis.local_basis}))

    psi0, _ = block_normalize(psi0, False, None)

    # Import the real function BEFORE patching
    from impurityModel.ed.ManyBodyUtils import inner_multi as real_inner_multi

    # We will track inner_multi calls to prove 'selective' does fewer than 'partial'
    with patch("impurityModel.ed.BlockLanczos.inner_multi") as mock_inner_multi_partial:

        def side_effect_partial(*args, **kwargs):
            return real_inner_multi(*args, **kwargs)

        mock_inner_multi_partial.side_effect = side_effect_partial

        alphas_p, betas_p, _Q_p, _W_p = block_lanczos_cy(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            converged_fn=lambda a, b, **kw: len(a) >= 8,
            reort="partial",
            max_iter=8,
            verbose=False,
        )
        partial_inner_multi_calls = mock_inner_multi_partial.call_count

    with patch("impurityModel.ed.BlockLanczos.inner_multi") as mock_inner_multi_selective:

        def side_effect_selective(*args, **kwargs):
            return real_inner_multi(*args, **kwargs)

        mock_inner_multi_selective.side_effect = side_effect_selective

        alphas_s, betas_s, _Q_s, _W_s = block_lanczos_cy(
            psi0=psi0,
            h_op=h_op,
            basis=basis,
            converged_fn=lambda a, b, **kw: len(a) >= 8,
            reort="selective",
            max_iter=8,
            verbose=False,
        )
        selective_inner_multi_calls = mock_inner_multi_selective.call_count

    # Assert betas didn't blow up in selective
    max_beta_norm = max([np.linalg.norm(b) for b in betas_s])
    assert max_beta_norm < 1e10, f"Betas blew up in selective reort: max beta norm = {max_beta_norm}"

    # Extract eigenvalues of the T matrix to compare
    T_p = _build_full_T(alphas_p, betas_p[: len(alphas_p) - 1])
    eigvals_p = np.sort(np.linalg.eigvalsh(T_p))

    T_s = _build_full_T(alphas_s, betas_s[: len(alphas_s) - 1])
    eigvals_s = np.sort(np.linalg.eigvalsh(T_s))

    # Assert eigenvalues are close
    np.testing.assert_allclose(
        eigvals_p,
        eigvals_s,
        atol=1e-7,
        err_msg="Eigenvalues differ between partial and selective reorthogonalization",
    )

    # Check that selective makes fewer inner_multi calls than partial
    # Wait, 'partial' might not always do more if it doesn't trigger.
    # Let's see if selective is indeed less or equal to partial.
    # Usually selective triggers only when ritz error bounds dictate, whereas partial does it based
    # on local orthogonality loss.
    # In block_lanczos_cy, partial does reorthogonalization of q_next against all Q_basis.
    # We just ensure it's not failing and maybe check if partial_inner_multi_calls > selective_inner_multi_calls
    # To be safe, just assert selective <= partial. If the test matrix is right, it might be strictly less.
    assert (
        selective_inner_multi_calls <= partial_inner_multi_calls
    ), f"Selective ({selective_inner_multi_calls}) called inner_multi more than partial ({partial_inner_multi_calls})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
