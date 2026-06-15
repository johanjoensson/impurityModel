import os
import tempfile

import numpy as np

from impurityModel.ed import op_parser
from impurityModel.ed.average import thermal_average, thermal_average_scale_indep


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
