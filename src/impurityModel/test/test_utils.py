"""Unit tests for the pure formatting/utility helpers in ``impurityModel.ed.utils``.

These functions have no numerical algorithm behind them, so the tests pin down exact
formatted strings on small fixed inputs plus the column-alignment invariants the
pretty-printers promise.
"""

import numpy as np
import pytest

from impurityModel.ed.utils import (
    _float_field_width,
    matrix_connectivity_print,
    matrix_print,
    matrix_to_string,
    partition,
    vector_to_string,
)


# --------------------------------------------------------------------------- #
# _float_field_width
# --------------------------------------------------------------------------- #
def test_field_width_positive_single_digit():
    # "1.50" -> int_digits(1) + point(1) + n_prec(2), no sign column.
    assert _float_field_width(np.array([1.5]), n_prec=2) == 4


def test_field_width_reserves_sign_column_for_negatives():
    # A negative entry adds one leading sign column: "-1.50".
    assert _float_field_width(np.array([-1.5]), n_prec=2) == 5


def test_field_width_force_sign_adds_column_even_when_positive():
    assert _float_field_width(np.array([1.5]), n_prec=2, force_sign=True) == 5


def test_field_width_counts_integer_digits_of_largest_magnitude():
    # max_abs = 123.0 -> 3 integer digits, + point + 1 decimal = 5.
    assert _float_field_width(np.array([1.0, 123.0]), n_prec=1) == 5


def test_field_width_empty_defaults_to_single_integer_digit():
    assert _float_field_width(np.array([]), n_prec=3) == 1 + 1 + 3


# --------------------------------------------------------------------------- #
# vector_to_string
# --------------------------------------------------------------------------- #
def test_vector_to_string_real_exact():
    v = np.array([1.0, 2.0])
    # width 4 => "1.00", "2.00" joined by a single space.
    assert vector_to_string(v, n_prec=2) == "1.00 2.00"


def test_vector_to_string_auto_detects_real_when_imag_negligible():
    v = np.array([1.0 + 0j, 2.0 + 0j])
    assert vector_to_string(v, n_prec=2) == "1.00 2.00"


def test_vector_to_string_complex_uses_signed_imag_and_j_suffix():
    v = np.array([1.0 + 2.0j])
    out = vector_to_string(v, n_prec=1)
    assert out.endswith("j")
    assert "+2.0j" in out
    assert "1.0" in out


def test_vector_to_string_rejects_non_1d():
    with pytest.raises(AssertionError):
        vector_to_string(np.zeros((2, 2)))


# --------------------------------------------------------------------------- #
# matrix_to_string
# --------------------------------------------------------------------------- #
def test_matrix_to_string_one_line_per_row_and_aligned():
    m = np.array([[1.0, -20.0], [300.0, 4.0]])
    s = matrix_to_string(m, n_prec=1)
    lines = s.split("\n")
    assert len(lines) == 2
    # Every line has identical length => columns line up.
    assert len({len(line) for line in lines}) == 1


def test_matrix_to_string_offset_indents_every_row():
    m = np.eye(2)
    s = matrix_to_string(m, n_prec=1, offset=3)
    assert all(line.startswith("   ") for line in s.split("\n"))


# --------------------------------------------------------------------------- #
# matrix_print / matrix_connectivity_print (stdout side effects)
# --------------------------------------------------------------------------- #
def test_matrix_print_vector_branch(capsys):
    matrix_print(np.array([1.0, 2.0]), n_prec=1)
    assert capsys.readouterr().out.strip() == "1.0 2.0"


def test_matrix_print_label_is_printed(capsys):
    matrix_print(np.eye(2), label="M", n_prec=1)
    out = capsys.readouterr().out
    assert out.startswith("M\n")


def test_connectivity_diagonal_only_O(capsys):
    matrix_connectivity_print(np.eye(3))
    out = capsys.readouterr().out
    assert "O" in out
    assert "X" not in out


def test_connectivity_offdiagonal_gets_X(capsys):
    m = np.array([[1.0, 1.0], [1.0, 1.0]])
    matrix_connectivity_print(m)
    out = capsys.readouterr().out
    assert "O" in out and "X" in out


def test_connectivity_block_size_folds_offdiagonals_into_O(capsys):
    # A fully dense 2x2 read as a single 2-block is entirely "diagonal".
    m = np.ones((2, 2))
    matrix_connectivity_print(m, block_size=2)
    out = capsys.readouterr().out
    assert "X" not in out and "O" in out


# --------------------------------------------------------------------------- #
# partition
# --------------------------------------------------------------------------- #
def test_partition_default_truthiness():
    passed, failed = partition([0, 1, 2, 0, 3])
    assert passed == [1, 2, 3]
    assert failed == [0, 0]


def test_partition_custom_predicate():
    passed, failed = partition(range(6), predicate=lambda x: x % 2 == 0)
    assert passed == [0, 2, 4]
    assert failed == [1, 3, 5]


def test_partition_empty():
    assert partition([]) == ([], [])


def test_partition_all_pass_and_all_fail():
    assert partition([1, 2], predicate=lambda _: True) == ([1, 2], [])
    assert partition([1, 2], predicate=lambda _: False) == ([], [1, 2])
