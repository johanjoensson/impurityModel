"""Tests for the legacy bare-integer h0 reader.

The synthetic cases always run. The real-workload cases need the sibling ``impmod_tests``
checkout, which is not part of this repository, and skip cleanly without it.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from impurityModel.ed import h0_format
from impurityModel.ed.h0_format import H0FormatError

IMPMOD_TESTS = os.environ.get("IMPMOD_TESTS_DIR", "/home/johan/Programming/impmod_tests")
NIO_LEGACY = os.path.join(IMPMOD_TESTS, "NiO", "base", "Ni_h0_op.dict")


def _write_star_model(path, n_imp=4, n_bath=4):
    """A dense impurity block, every bath coupled to it, and a diagonal (star) bath block."""
    n = n_imp + n_bath
    h = np.zeros((n, n), dtype=complex)
    h[:n_imp, :n_imp] = 1.0 + np.eye(n_imp)
    for b in range(n_imp, n):
        h[b, b] = -1.0 - 0.1 * b
        for i in range(n_imp):
            h[b, i] = h[i, b] = 0.3

    lines = [
        f"{i} {j} {complex(h[i, j]).real!r} {complex(h[i, j]).imag!r}"
        for i in range(n)
        for j in range(n)
        if h[i, j] != 0
    ]
    path.write_text("\n".join(lines) + "\n")
    return h, n_imp


def test_legacy_reader_synthetic(tmp_path):
    path = tmp_path / "synthetic_h0_op.dict"
    h, n_imp = _write_star_model(path)

    with pytest.deprecated_call():
        parsed = h0_format.read_legacy_flat_h0(path, n_imp)

    assert parsed.n_orb == h.shape[0]
    assert parsed.impurity_orbitals == {0: (0, 1, 2, 3)}
    assert parsed.unit == "unknown"
    np.testing.assert_array_equal(parsed.to_matrix(), h)


def test_legacy_reader_symmetrizes_a_dropped_conjugate_partner(tmp_path):
    """One shipped file has an element whose conjugate partner was filtered away.

    The producer applied its abs(x) > 0 filter elementwise to a matrix that is not bitwise
    Hermitian, so a term can survive while its partner does not. Symmetrize rather than
    reject -- rejecting would make a real stored workload unreadable.
    """
    path = tmp_path / "asym_h0_op.dict"
    path.write_text("0 0 1.0 0.0\n1 0 2.0 0.0\n1 1 -1.0 0.0\n")

    with pytest.deprecated_call():
        h = h0_format.read_legacy_flat_h0(path, 1, verify_layout=False).to_matrix()

    assert h[1, 0] == 1.0 and h[0, 1] == 1.0
    np.testing.assert_array_equal(h, h.conj().T)


def test_wrong_n_imp_raises_and_suggests_the_right_one(tmp_path):
    path = tmp_path / "synthetic_h0_op.dict"
    _write_star_model(path, n_imp=4, n_bath=4)

    # The layout check runs before the deprecation warning, so only the raise is expected.
    with pytest.raises(H0FormatError, match=r"n_imp = 4"):
        h0_format.read_legacy_flat_h0(path, 6)


def test_layout_check_can_be_disabled(tmp_path):
    path = tmp_path / "synthetic_h0_op.dict"
    _write_star_model(path, n_imp=4, n_bath=4)

    with pytest.deprecated_call():
        parsed = h0_format.read_legacy_flat_h0(path, 6, verify_layout=False)
    assert parsed.n_imp == 6


def test_out_of_range_n_imp_raises(tmp_path):
    path = tmp_path / "synthetic_h0_op.dict"
    _write_star_model(path)
    for bad in (0, 99):
        with pytest.raises(H0FormatError, match="is not in 1"):
            h0_format.read_legacy_flat_h0(path, bad)


@pytest.mark.skipif(not os.path.exists(NIO_LEGACY), reason="impmod_tests checkout not available")
def test_legacy_reader_reads_the_stored_nio_workload():
    with pytest.deprecated_call():
        parsed = h0_format.read_legacy_flat_h0(NIO_LEGACY, 10)

    assert parsed.n_orb == 176
    assert parsed.n_imp == 10
    h = parsed.to_matrix()
    np.testing.assert_array_equal(h, h.conj().T)
    # The whole point of the format work: the impurity block is *not* on the bath's energy
    # zero in the stored files (see doc/h0_file_format.md). Pin that so a later fix to
    # build_h0 shows up here rather than silently changing results.
    assert h.diagonal()[:10].real.min() > h.diagonal()[10:].real.max()


@pytest.mark.skipif(not os.path.exists(NIO_LEGACY), reason="impmod_tests checkout not available")
def test_sparsity_check_recovers_n_imp_on_the_stored_workload():
    # Path.read_text, not a bare open(): pytest.ini turns warnings into errors and
    # conftest forces a gc.collect(), so a dangling file object fails the session.
    terms = h0_format.parse_flat_terms(Path(NIO_LEGACY).read_text().splitlines(), path=NIO_LEGACY)
    assert h0_format.infer_n_impurity_orbitals(terms, 176) == 10
