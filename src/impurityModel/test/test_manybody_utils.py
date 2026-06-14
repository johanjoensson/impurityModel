import pytest
import numpy as np
from impurityModel.ed.ManyBodyUtils import ManyBodyState, ManyBodyOperator, inner_multi, add_scaled_multi, inner, SlaterDeterminant

def get_random_state(num_terms):
    s = ManyBodyState()
    for _ in range(num_terms):
        key = tuple(sorted(np.random.randint(0, 100, size=np.random.randint(1, 5))))
        val = np.random.rand() + 1j * np.random.rand()
        s.add_scaled(ManyBodyState({SlaterDeterminant(key): 1.0}), val)
    return s

def test_inner_multi():
    n_states_left = 5
    n_states_right = 6
    num_terms = 20

    left_states = [get_random_state(num_terms) for _ in range(n_states_left)]
    right_states = [get_random_state(num_terms) for _ in range(n_states_right)]

    # Calculate using optimized multi function
    M_multi = inner_multi(left_states, right_states)

    # Calculate using simple nested loops
    M_loop = np.zeros((n_states_left, n_states_right), dtype=complex)
    for i, s_l in enumerate(left_states):
        for j, s_r in enumerate(right_states):
            M_loop[i, j] = inner(s_l, s_r)

    np.testing.assert_allclose(M_multi, M_loop, atol=1e-12)

def test_add_scaled_multi():
    n_states_base = 4
    n_states_add = 5
    num_terms = 15

    base_states_1 = [get_random_state(num_terms) for _ in range(n_states_base)]
    base_states_2 = [s.copy() for s in base_states_1]

    add_states = [get_random_state(num_terms) for _ in range(n_states_add)]

    # Random scale matrix
    scale_matrix = np.random.rand(n_states_add, n_states_base) + 1j * np.random.rand(n_states_add, n_states_base)

    # 1. Update base_states_1 with add_scaled_multi
    add_scaled_multi(base_states_1, add_states, scale_matrix)

    # 2. Update base_states_2 with python loops
    for j in range(n_states_base):
        for k in range(n_states_add):
            if scale_matrix[k, j] != 0:
                base_states_2[j].add_scaled(add_states[k], scale_matrix[k, j])

    # Assert equality
    for s1, s2 in zip(base_states_1, base_states_2):
        assert len(s1) == len(s2)
        for key in s1:
            assert key in s2
            np.testing.assert_allclose(s1[key], s2[key], atol=1e-12)

def test_apply_multi():
    n_states = 3
    num_terms_op = 10
    num_terms_state = 10

    # Build random many-body operator
    op_dict = {}
    for _ in range(num_terms_op):
        num_c = np.random.randint(1, 3)
        num_a = np.random.randint(1, 3)
        k_c = tuple((int(np.random.randint(0, 50)), 'c') for _ in range(num_c))
        k_a = tuple((int(np.random.randint(0, 50)), 'a') for _ in range(num_a))
        op_dict[k_c + k_a] = np.random.rand() + 1j * np.random.rand()

    op = ManyBodyOperator(op_dict)
    
    # Create random states
    states = [get_random_state(num_terms_state) for _ in range(n_states)]

    # 1. Apply multi
    results_multi = op.apply_multi(states)

    # 2. Apply in loop
    results_loop = [op(s) for s in states]

    # Assert equality
    for r_multi, r_loop in zip(results_multi, results_loop):
        assert len(r_multi) == len(r_loop)
        for key in r_multi:
            assert key in r_loop
            np.testing.assert_allclose(r_multi[key], r_loop[key], atol=1e-12)
