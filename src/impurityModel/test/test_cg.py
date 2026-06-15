import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from impurityModel.ed.cg import block_bicgstab

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

@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_dict_mpi(mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled):
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

@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_dict_no_mpi(mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled):
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
        1.0, 1.0, # block_norm
        1.0, 1.0, # r_norms2
        0.0, 0.0, # s_norms2 -> breaks early
    ]
    
    x_sol = block_bicgstab(A, x0, y, basis=basis, slaterWeightMin=0.0, max_iter=2, atol=1e-8, rtol=1e-12)
    
    assert len(x_sol) == 2

@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_cond_break(mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled):
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
    mock_inner.side_effect = [1.0, 1.0] # block_norm, r_norms2
    
    x_sol = block_bicgstab(A, x0, y, basis=basis, slaterWeightMin=0.0, max_iter=2)
    assert len(x_sol) == 1

@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_tt_zero(mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled):
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
        1.0, # block_norm
        1.0, # r_norms2
        1.0, # s_norms2
        0.0, # ts
        0.0, # tt -> triggers abs(tt) < eps
        0.0, # r_norms2 (loop 2) -> break
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

@patch("impurityModel.ed.cg.block_add_scaled")
@patch("impurityModel.ed.cg.block_apply")
@patch("impurityModel.ed.cg.block_inner")
@patch("impurityModel.ed.cg.inner")
def test_block_bicgstab_active_mask_break(mock_inner, mock_block_inner, mock_block_apply, mock_block_add_scaled):
    A = MagicMock()
    x0 = [{"state1": 0.0 + 0j}]
    y = [{"state1": 1.0 + 0j}]
    basis = MagicMock()
    basis.is_distributed = False
    
    mock_block_apply.return_value = [{"state1": 0.5 + 0j, "state2": 0.5 + 0j, "state3": 0.5 + 0j}]
    mock_block_inner.return_value = np.eye(1, dtype=complex)
    
    mock_inner.side_effect = [
        1.0, # block_norm
        1.0, # r_norms2 loop 1
        1.0, # s_norms2 loop 1
        1.0, # ts loop 1
        1.0, # tt loop 1
        0.0, # r_norms2 loop 2 -> breaks at line 110
    ]
    
    x_sol = block_bicgstab(A, x0, y, basis=basis, slaterWeightMin=0.0, max_iter=2)
    assert len(x_sol) == 1
