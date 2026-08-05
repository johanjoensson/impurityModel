import os
import tempfile

import numpy as np
import pytest

from impurityModel.ed import op_parser
from impurityModel.ed.average import thermal_average, thermal_average_scale_indep
from impurityModel.ed.utils import rotate_matrix


def test_average():
    # Test thermal averaging
    energies = np.array([0.0, 1.0, 2.0])
    observables = np.array([10.0, 5.0, 1.0])

    # Large temperature or scale should average them closer
    avg_scale = thermal_average_scale_indep(energies, observables, 1.0)
    w_scale = np.exp(-energies)
    expected_scale = np.sum(w_scale * observables) / np.sum(w_scale)
    np.testing.assert_allclose(avg_scale, expected_scale, atol=1e-12)

    avg_temp = thermal_average(energies, observables, T=300)
    import scipy as sp

    k_B = sp.constants.physical_constants["Boltzmann constant in eV/K"][0]
    expected_temp = thermal_average_scale_indep(energies, observables, k_B * 300)
    np.testing.assert_allclose(avg_temp, expected_temp, atol=1e-12)


def test_op_parser():
    # Test helper text parsing functions
    assert op_parser.skip_whitespaces(" \t\nhello") == "hello"

    rem, val = op_parser.read_state_tuple("1, 2, 3) remaining")
    assert rem == " remaining"
    assert val == (1, 2, 3)

    rem, real_val = op_parser.read_real("  -3.14e-2 remaining")
    assert rem == " remaining"
    assert np.allclose(real_val, -0.0314)

    complex_val = op_parser.read_real_imag(" 1.5 -2.5")
    assert complex_val == (1.5 - 2.5j)

    op_key, amp = op_parser.extract_operator("(1, 2) (3, 4) 2.0 1.0")
    assert op_key == (((1, 2), "c"), ((3, 4), "a"))
    assert amp == (2.0 + 1.0j)

    assert op_parser.read_name("OpName") == "OpName"
    assert op_parser.read_name("(1, 2)") == ""

    # Test file writing and reading using the expected format that the parser reads
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_ops.dat")
        with open(filepath, "w") as f:
            f.write("# This is a comment\n")
            f.write("OpName\n")
            f.write("(1, 2) (3, 4) 1.5 0.5\n")
            f.write("(5, 6) (7, 8) -0.5 0.0\n")

        # Parse it back
        parsed_ops = op_parser.parse_file(filepath)
        assert len(parsed_ops) == 1
        assert "OpName" in parsed_ops
        op = parsed_ops["OpName"]
        assert (((1, 2), "c"), ((3, 4), "a")) in op
        assert op[(((1, 2), "c"), ((3, 4), "a"))] == 1.5 + 0.5j
        assert op[(((5, 6), "c"), ((7, 8), "a"))] == -0.5 + 0j


def test_op_parser_singleton_state_tuple():
    """The singleton form "(2,)" -- the parser's own documented example -- must parse.

    The trailing comma used to leave an empty final segment that went to int(), so the
    docstring example raised ValueError.
    """
    assert op_parser.read_state_tuple("2,) residue") == (" residue", (2,))
    assert op_parser.extract_operator("(2,) (1,) 1.5 0.0") == ((((2,), "c"), ((1,), "a")), 1.5 + 0j)


def test_op_parser_consumes_a_trailing_number():
    """read_real must not leave the number's last character in the remainder."""
    assert op_parser.read_real("1.5") == ("", 1.5)


def test_op_parser_degenerate_input_raises_valueerror():
    """Empty / all-whitespace / unterminated input raises ValueError, never UnboundLocalError.

    These used to raise UnboundLocalError (an unbound loop variable) or, worse,
    return silently: read_state_tuple on a paren-free line returned ('', ()), a
    valid-looking empty state tuple.
    """
    assert op_parser.skip_whitespaces("   ") == ""
    assert op_parser.skip_whitespaces("") == ""
    assert op_parser.read_name("") == ""

    for bad in ("", "   ", "0 0 -4.12 0.0"):
        with pytest.raises(ValueError):
            op_parser.read_state_tuple(bad)
    with pytest.raises(ValueError):
        op_parser.read_real("   ")


def test_op_parser_rejects_flat_index_format_with_context():
    """The rspt2spectra flat-index line must raise ValueError naming the file and line.

    This is the regression for the reported bug: op_parser used to die with a bare
    UnboundLocalError from skip_whitespaces, against a line number nobody could locate.
    """
    with pytest.raises(ValueError):
        op_parser.extract_operator("  0   0 -4.125093021114729  0.000000000000000")

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "flat.dat")
        with open(filepath, "w") as f:
            f.write("  0   0 -4.125093021114729  0.000000000000000\n")
            f.write("  0   1  0.130836789023578  0.092828753963260\n")

        with pytest.raises(ValueError, match=r"flat\.dat:2:"):
            op_parser.parse_file(filepath)


def test_rotate_matrix():
    M = np.array([[1.0, 0], [0, 2.0]])
    T = np.array([[0, 1], [1, 0]])
    M_rot = rotate_matrix(M, T)
    expected = np.array([[2.0, 0], [0, 1.0]])
    assert np.allclose(M_rot, expected)

    T_dict = {0: np.array([[0, 1], [1, 0]])}
    M_rot_dict = rotate_matrix(M, T_dict)
    assert np.allclose(M_rot_dict, expected)
