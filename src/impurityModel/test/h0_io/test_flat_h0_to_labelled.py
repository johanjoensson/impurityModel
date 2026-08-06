"""``hamiltonian_io.flat_h0_to_labelled``: relabelling a flat ``.h0`` into ``(l, s, m)``/``(l, b)``.

This is the reader the spectra path uses (via :func:`hamiltonian_io.read_h0_operator`) so a
``.h0`` file can flow through the same labelled Hamiltonian assembly
(:func:`hamiltonian_io.get_hamiltonian_operator`) as the pickled/``.dat``/``.json`` formats.
"""

from collections import OrderedDict

import numpy as np
import pytest

from impurityModel.ed import h0_format
from impurityModel.ed.hamiltonian_io import flat_h0_to_labelled, read_h0_operator
from impurityModel.ed.operator_algebra import c2i, i2c


def _d_shell_file(tmp_path, *, l=2, n_bath=4, name="d_shell.h0", **header_extra):
    """A dense d-shell impurity block coupled to ``n_bath`` bath orbitals."""
    n_imp = 2 * (2 * l + 1)
    n = n_imp + n_bath
    rng = np.random.default_rng(0)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)

    out = tmp_path / name
    header_extra.setdefault("basis", "spherical")
    header_extra.setdefault("spin_ordering", "down_first")
    header_extra.setdefault("energy_reference", "fermi")
    header_extra.setdefault("impurity_l", l)
    h0_format.write_h0_file(out, h, impurity_orbitals={0: list(range(n_imp))}, **header_extra)
    return out, h


def test_flat_h0_to_labelled_round_trips_through_c2i(tmp_path):
    """The relabelling is exactly the inverse of c2i on a single-shell nBaths dict."""
    l, n_bath = 2, 4
    path, h = _d_shell_file(tmp_path, l=l, n_bath=n_bath)
    parsed = h0_format.read_h0_file(path)

    labelled = flat_h0_to_labelled(parsed, OrderedDict({l: n_bath}))

    n = h.shape[0]
    reconstructed = np.zeros((n, n), dtype=complex)
    for (op_c, op_a), value in labelled.items():
        reconstructed[c2i({l: n_bath}, op_c[0]), c2i({l: n_bath}, op_a[0])] = value
    np.testing.assert_allclose(reconstructed, h)


def test_flat_h0_to_labelled_via_read_h0_operator(tmp_path):
    """``read_h0_operator`` dispatches .h0 files to the same relabelling."""
    l, n_bath = 2, 4
    path, h = _d_shell_file(tmp_path, l=l, n_bath=n_bath)
    nBaths = OrderedDict({1: 0, l: n_bath})

    operator = read_h0_operator(str(path), nBaths)

    assert all(op_c[0][0] == l or op_a[0][0] == l for (op_c, op_a) in operator)
    n = h.shape[0]
    reconstructed = np.zeros((n, n), dtype=complex)
    for (op_c, op_a), value in operator.items():
        reconstructed[c2i({l: n_bath}, op_c[0]), c2i({l: n_bath}, op_a[0])] = value
    np.testing.assert_allclose(reconstructed, h)


def test_flat_h0_to_labelled_rejects_non_spherical_basis(tmp_path):
    path, _ = _d_shell_file(tmp_path, basis="cubic")
    parsed = h0_format.read_h0_file(path)
    with pytest.raises(RuntimeError, match="basis"):
        flat_h0_to_labelled(parsed, OrderedDict({2: 4}))


def test_flat_h0_to_labelled_rejects_wrong_spin_ordering(tmp_path):
    path, _ = _d_shell_file(tmp_path, spin_ordering="up_first")
    parsed = h0_format.read_h0_file(path)
    with pytest.raises(RuntimeError, match="spin_ordering"):
        flat_h0_to_labelled(parsed, OrderedDict({2: 4}))


def test_flat_h0_to_labelled_rejects_absolute_energy_reference(tmp_path):
    path, _ = _d_shell_file(tmp_path, energy_reference="absolute")
    parsed = h0_format.read_h0_file(path)
    with pytest.raises(RuntimeError, match="energy_reference"):
        flat_h0_to_labelled(parsed, OrderedDict({2: 4}))


def test_flat_h0_to_labelled_rejects_unknown_impurity_l(tmp_path):
    path, _ = _d_shell_file(tmp_path)
    parsed = h0_format.read_h0_file(path)
    with pytest.raises(RuntimeError, match="impurity_l"):
        flat_h0_to_labelled(parsed, OrderedDict({1: 0}))


def test_flat_h0_to_labelled_rejects_bath_count_mismatch(tmp_path):
    path, _ = _d_shell_file(tmp_path, n_bath=4)
    parsed = h0_format.read_h0_file(path)
    with pytest.raises(RuntimeError, match="bath orbitals"):
        flat_h0_to_labelled(parsed, OrderedDict({2: 10}))


def test_flat_h0_to_labelled_rejects_multi_shell_files(tmp_path):
    path, _ = _d_shell_file(tmp_path, n_bath=0)
    # Hand-author a multi-group header on top of the single-shell body.
    text = path.read_text().splitlines()
    import json

    header = json.loads(text[1])
    header["impurity_orbitals"] = {"0": list(range(5)), "1": list(range(5, 10))}
    path.write_text("\n".join([text[0], json.dumps(header), *text[2:]]) + "\n")

    parsed = h0_format.read_h0_file(path)
    with pytest.raises(RuntimeError, match="more than one"):
        flat_h0_to_labelled(parsed, OrderedDict({2: 0}))


def test_read_h0_operator_rejects_legacy_headerless_files(tmp_path):
    legacy = tmp_path / "legacy.h0"
    legacy.write_text("0 0 1.0 0.0\n")
    with pytest.raises(RuntimeError, match="legacy"):
        read_h0_operator(str(legacy), OrderedDict({2: 0}))


def test_i2c_is_the_exact_inverse_used_by_the_relabelling():
    """Anchors the claim the plan and the docstring both rely on."""
    l, n_bath = 2, 3
    shell = OrderedDict({l: n_bath})
    for i in range(2 * (2 * l + 1) + n_bath):
        assert c2i(shell, i2c(shell, i)) == i
