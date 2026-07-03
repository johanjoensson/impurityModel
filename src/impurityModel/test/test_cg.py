import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from impurityModel.ed.cg import block_bicgstab


def _fake_block_combine(Q, Y, slaterWeightMin=0.0):
    # Stand-in for the Cython block_combine in the mock-based dict tests: the real one
    # requires ManyBodyState blocks and segfaults on the plain dicts used here.
    return [dict(Q[i % len(Q)]) for i in range(Y.shape[1])]


def test_block_bicgstab_array_single():
    np.random.seed(42)
    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    A = A @ A.conj().T + np.eye(10)

    x_exact = np.random.rand(10, 1) + 1j * np.random.rand(10, 1)
    y = A @ x_exact
    x0 = np.zeros((10, 1), dtype=complex)

    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-5, atol=1e-5)


def test_block_bicgstab_array_block():
    np.random.seed(42)
    A = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    A = A @ A.conj().T + np.eye(10)

    x_exact = np.random.rand(10, 3) + 1j * np.random.rand(10, 3)
    y = A @ x_exact
    x0 = np.zeros((10, 3), dtype=complex)

    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-5, atol=1e-5)


def test_block_bicgstab_zero_rhs():
    A = np.eye(5)
    x0 = np.zeros((5, 2), dtype=complex)
    y = np.zeros((5, 2), dtype=complex)
    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, np.zeros((5, 2), dtype=complex))


def test_block_bicgstab_exact_guess():
    A = np.eye(5)
    x_exact = np.ones((5, 2), dtype=complex)
    x0 = x_exact.copy()
    y = A @ x_exact
    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact)


def test_block_bicgstab_max_iter():
    A = np.eye(5)
    A[0, 1] = 0.5
    x_exact = np.ones((5, 2), dtype=complex)
    y = A @ x_exact
    x0 = np.zeros((5, 2), dtype=complex)
    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0, max_iter=0)
    assert not np.allclose(x_sol, x_exact)


@patch("impurityModel.ed.cg.block_combine", side_effect=_fake_block_combine)
@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_dict_mpi(
    mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled, mock_block_combine
):
    # Setting up inputs
    A = MagicMock()
    A.set_restrictions = MagicMock()

    x0 = [{"state1": 0.0 + 0j}]
    y = [{"state1": 1.0 + 0j}]

    basis = MagicMock()
    basis.is_distributed = True
    basis.comm = MagicMock()
    basis.local_basis = set(["state1"])
    basis.add_states = MagicMock()
    basis.restrictions = {}

    # We want block_bicgstab to do exactly one iteration and finish.
    # To do that, the active_mask should be false after one iteration, or cond > eps,
    # or the residual norms become small.
    # We can control the loop through mock_block_inner and mock_inner

    # Let's say max_iter=1
    # Matmat returns something
    mock_block_apply.return_value = [{"state1": 0.5 + 0j, "state2": 0.5 + 0j, "state3": 0.5 + 0j}]

    # inner returns something big for r0_norm
    mock_inner.return_value = 1.0

    # block_inner returns a 1x1 matrix
    mock_block_inner.return_value = np.array([[1.0 + 0j]])

    # After one iteration, it calculates residual norm. We can just set max_iter=1.
    x_sol = block_bicgstab(A, x0, y, basis=basis, slaterWeightMin=0.0, max_iter=1, atol=1e-8, rtol=1e-12)

    assert len(x_sol) == 1
    basis.add_states.assert_called()
    A.set_restrictions.assert_called_with(basis.restrictions)
    basis.comm.Allreduce.assert_called()


@patch("impurityModel.ed.cg.block_combine", side_effect=_fake_block_combine)
@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_dict_no_mpi(
    mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled, mock_block_combine
):
    A = MagicMock()
    x0 = [{"state1": 0.0 + 0j}, {"state2": 0.0 + 0j}]
    y = [{"state1": 1.0 + 0j}, {"state2": 1.0 + 0j}]

    basis = MagicMock()
    basis.is_distributed = False
    basis.local_basis = set()
    basis.add_states = MagicMock()

    mock_block_apply.return_value = [{"state1": 0.5 + 0j}, {"state2": 0.5 + 0j}]
    mock_inner.return_value = 1.0
    mock_block_inner.return_value = np.eye(2, dtype=complex)

    # Let it run for 2 iterations, then finish.
    # To make it finish, we need active_mask to be False, which means r_norms < atol
    # We will use side_effect on mock_inner to return 1.0 initially, then 0.0 to break.
    # inner is called in block_norm (len=2), then inside loop for r_norms2 (len=2), then for s_norms2 (len=2), then ts/tt (len=2, len=2).
    # Since side_effect is sequential, we just provide enough 1.0s and then 0.0s.
    # Actually, if s_norms2 is 0, it breaks early.
    mock_inner.side_effect = [
        1.0,
        1.0,  # block_norm
        1.0,
        1.0,  # r_norms2
        0.0,
        0.0,  # s_norms2 -> breaks early
    ]

    x_sol = block_bicgstab(A, x0, y, basis=basis, slaterWeightMin=0.0, max_iter=2, atol=1e-8, rtol=1e-12)

    assert len(x_sol) == 2


@patch("impurityModel.ed.cg.block_combine", side_effect=_fake_block_combine)
@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_cond_break(
    mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled, mock_block_combine
):
    # Test line 138: np.linalg.cond(R0_V) > 1 / np.finfo(float).eps
    A = MagicMock()
    x0 = [{"state1": 0.0 + 0j}]
    y = [{"state1": 1.0 + 0j}]
    basis = MagicMock()
    basis.is_distributed = False

    mock_block_apply.return_value = [{"state1": 0.5 + 0j, "state2": 0.5 + 0j, "state3": 0.5 + 0j}]
    mock_inner.return_value = 1.0
    # Make R0_V perfectly singular
    mock_block_inner.return_value = np.zeros((1, 1), dtype=complex)

    # We provide enough side effects for inner if needed
    mock_inner.side_effect = [1.0, 1.0]  # block_norm, r_norms2

    x_sol = block_bicgstab(A, x0, y, basis=basis, slaterWeightMin=0.0, max_iter=2)
    assert len(x_sol) == 1


@patch("impurityModel.ed.cg.block_combine", side_effect=_fake_block_combine)
@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_tt_zero(
    mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled, mock_block_combine
):
    # Test line 184: abs(tt) < eps
    A = MagicMock()
    x0 = [{"state1": 0.0 + 0j}]
    y = [{"state1": 1.0 + 0j}]
    basis = MagicMock()
    basis.is_distributed = False

    mock_block_apply.return_value = [{"state1": 0.5 + 0j, "state2": 0.5 + 0j, "state3": 0.5 + 0j}]
    mock_block_inner.return_value = np.eye(1, dtype=complex)

    # inner calls:
    # 1. block_norm
    # Loop 1:
    # 2. r_norms2
    # 3. s_norms2
    # 4. ts (inner(ti, si))
    # 5. tt (inner(ti, ti))
    # Loop 2:
    # 6. r_norms2 -> make it 0.0 to break
    mock_inner.side_effect = [
        1.0,  # block_norm
        1.0,  # r_norms2
        1.0,  # s_norms2
        0.0,  # ts
        0.0,  # tt -> triggers abs(tt) < eps
        0.0,  # r_norms2 (loop 2) -> break
    ]

    x_sol = block_bicgstab(A, x0, y, basis=basis, slaterWeightMin=0.0, max_iter=2)
    assert len(x_sol) == 1


def test_block_bicgstab_break_active_mask():
    # Test line 110: break if not np.any(active_mask)
    # We want a real array test where r0_norm > eps, but after 1 iteration, r_norms drops below atol
    np.random.seed(42)
    A = np.eye(2)
    x_exact = np.ones((2, 1), dtype=complex)
    y = A @ x_exact
    x0 = np.zeros((2, 1), dtype=complex)

    # We can force a break after 1 iteration because A is identity
    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0)
    np.testing.assert_allclose(x_sol, x_exact)


@patch("impurityModel.ed.cg.block_combine", side_effect=_fake_block_combine)
@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_active_mask_break(
    mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled, mock_block_combine
):
    A = MagicMock()
    x0 = [{"state1": 0.0 + 0j}]
    y = [{"state1": 1.0 + 0j}]
    basis = MagicMock()
    basis.is_distributed = False

    mock_block_apply.return_value = [{"state1": 0.5 + 0j, "state2": 0.5 + 0j, "state3": 0.5 + 0j}]
    mock_block_inner.return_value = np.eye(1, dtype=complex)

    mock_inner.side_effect = [
        1.0,  # block_norm
        1.0,  # r_norms2 loop 1
        1.0,  # s_norms2 loop 1
        1.0,  # ts loop 1
        1.0,  # tt loop 1
        0.0,  # r_norms2 loop 2 -> breaks at line 110
    ]

    x_sol = block_bicgstab(A, x0, y, basis=basis, slaterWeightMin=0.0, max_iter=2)
    assert len(x_sol) == 1


# --- rank-deficient block RHS (initial-block deflation + reconstruction) ---


def test_block_bicgstab_array_rank_deficient():
    """A block RHS with linearly dependent columns is solved exactly (A invertible), matching
    the per-column solves -- the case that previously stalled at the zero guess."""
    np.random.seed(1)
    A = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
    A = A @ A.conj().T + np.eye(8)
    c0 = np.random.rand(8, 1) + 1j * np.random.rand(8, 1)
    x_exact = np.hstack([c0, 2.0 * c0])  # rank-1 block: column 1 == 2 * column 0
    y = A @ x_exact
    x0 = np.zeros((8, 2), dtype=complex)

    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(A @ x_sol, y, rtol=1e-6, atol=1e-6)

    per_col = np.hstack(
        [
            block_bicgstab(
                A, np.zeros((8, 1), complex), y[:, k : k + 1], basis=None, slaterWeightMin=0.0, atol=1e-10, rtol=1e-12
            )
            for k in range(2)
        ]
    )
    np.testing.assert_allclose(x_sol, per_col, rtol=1e-6, atol=1e-6)


def test_block_bicgstab_array_partial_rank():
    """A 3-column block of rank 2 (column 2 = column 0 + column 1) is solved exactly."""
    np.random.seed(2)
    A = np.random.rand(9, 9) + 1j * np.random.rand(9, 9)
    A = A @ A.conj().T + np.eye(9)
    c0 = np.random.rand(9, 1) + 1j * np.random.rand(9, 1)
    c1 = np.random.rand(9, 1) + 1j * np.random.rand(9, 1)
    x_exact = np.hstack([c0, c1, c0 + c1])
    y = A @ x_exact
    x0 = np.zeros((9, 3), dtype=complex)

    x_sol = block_bicgstab(A, x0, y, basis=None, slaterWeightMin=0.0, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(x_sol, x_exact, rtol=1e-6, atol=1e-6)


def test_block_bicgstab_sparse_rank_deficient():
    """Sparse (ManyBodyState) path: two proportional RHS states solved as a block match the
    per-column solves to machine precision."""
    from mpi4py import MPI
    from impurityModel.ed.manybody_basis import Basis
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, SlaterDeterminant, inner

    det0 = SlaterDeterminant.from_bytes(b"\x80")  # orbital 0 occupied
    z = 3.0 + 0.5j
    # A = z*I - H, with H a 0<->1 hopping (the (c,a)+(a,c) term on orbital 0 is the identity).
    A = ManyBodyOperator(
        {
            ((0, "c"), (0, "a")): z,
            ((0, "a"), (0, "c")): z,
            ((0, "c"), (1, "a")): -0.7,
            ((1, "c"), (0, "a")): -0.7,
        }
    )

    def _basis():
        b = Basis(
            impurity_orbitals={0: [[0, 1]]},
            bath_states=({0: [[]]}, {0: [[]]}),
            initial_basis=[b"\x80", b"\x40"],
            verbose=False,
            comm=MPI.COMM_SELF,
        )
        b.add_states([b"\x80", b"\x40"])
        return b

    y_block = [ManyBodyState({det0: 1.0}), ManyBodyState({det0: 2.0})]  # proportional -> rank 1

    x_block = block_bicgstab(
        A, [ManyBodyState(), ManyBodyState()], y_block, basis=_basis(), slaterWeightMin=0.0, atol=1e-10, rtol=1e-12
    )

    ref = [
        block_bicgstab(A, [ManyBodyState()], [col], basis=_basis(), slaterWeightMin=0.0, atol=1e-10, rtol=1e-12)[0]
        for col in y_block
    ]
    for k in range(2):
        diff = x_block[k] - ref[k]
        assert inner(diff, diff).real < 1e-18
    # Column 1 is exactly twice column 0 (linearity preserved through deflation).
    diff10 = x_block[1] - (x_block[0] + x_block[0])
    assert inner(diff10, diff10).real < 1e-18
