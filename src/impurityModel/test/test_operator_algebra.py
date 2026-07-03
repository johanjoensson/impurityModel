import pytest

from impurityModel.ed.operator_algebra import assert_hermitian, daggerOp


def test_daggerOp():
    op = {((0, "c"), (1, "a")): 1.0 + 0.5j}
    dag = daggerOp(op)
    assert len(dag) == 1
    # Note: dagger reverses the tuple and swaps 'c' and 'a', conjugates value
    assert ((1, "c"), (0, "a")) in dag
    assert dag[((1, "c"), (0, "a"))] == 1.0 - 0.5j


def test_assert_hermitian():
    op_hermitian = {((0, "c"), (1, "a")): 1.0j, ((1, "c"), (0, "a")): -1.0j}
    assert_hermitian(op_hermitian)

    op_nonhermitian = {((0, "c"), (1, "a")): 1.0j}
    with pytest.raises(AssertionError):
        assert_hermitian(op_nonhermitian)
