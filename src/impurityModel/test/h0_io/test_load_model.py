"""``load_model``: the single place that answers which h0 formats impurityModel reads."""

import os
from pathlib import Path

import numpy as np
import pytest

from impurityModel.ed import h0_format, hamiltonian_io
from impurityModel.ed.model import load_model

REPO_ROOT = Path(__file__).resolve().parents[4]
NIO_PICKLE = REPO_ROOT / "h0" / "h0_NiO_10bath.pickle"

D_SHELL_SLATER = [7.5, 0.0, 9.9, 0.0, 6.6]


def _flat_matrix(n_imp=10, n_bath=4):
    n = n_imp + n_bath
    h = np.zeros((n, n), dtype=complex)
    h[:n_imp, :n_imp] = np.diag(np.linspace(-1.5, -1.0, n_imp)).astype(complex)
    for b in range(n_imp, n):
        h[b, b] = -3.0 + 0.5 * (b - n_imp)
        for i in range(n_imp):
            h[b, i] = h[i, b] = 0.4 - 0.01 * i
    return h


def _write_legacy(path, h):
    n = h.shape[0]
    path.write_text(
        "\n".join(
            f"{i} {j} {complex(h[i, j]).real!r} {complex(h[i, j]).imag!r}"
            for i in range(n)
            for j in range(n)
            if h[i, j] != 0
        )
        + "\n"
    )


def test_dispatches_the_h0_format(tmp_path):
    h = _flat_matrix()
    path = tmp_path / "model.h0"
    h0_format.write_h0_file(path, h, impurity_orbitals={2: list(range(10))})

    model = load_model(path, l=2, slater=D_SHELL_SLATER)
    assert model.n_spin_orbitals == 14
    assert model.u4 is not None


@pytest.mark.parametrize("suffix", [".dict", ".dat"])
def test_sniffs_a_legacy_flat_file_whatever_it_is_called(tmp_path, suffix):
    """A .dict renamed to .dat was the standard workaround, so the wild is full of them."""
    h = _flat_matrix()
    path = tmp_path / f"legacy{suffix}"
    _write_legacy(path, h)

    with pytest.deprecated_call():
        model = load_model(path, l=2, slater=D_SHELL_SLATER, n_impurity_orbitals=10)
    assert model.n_spin_orbitals == 14


@pytest.mark.skipif(not NIO_PICKLE.exists(), reason="NiO h0 pickle fixture not available")
def test_labelled_pickle_still_routes_to_the_labelled_reader():
    """The legacy labelled path must be untouched; it is what `spectra` and the shipped
    workloads use."""
    model = load_model(NIO_PICKLE, l=2, n_baths=10, slater=D_SHELL_SLATER)
    assert model.impurity_orbitals == {2: list(range(10))}
    assert model.n_spin_orbitals == 20


def test_default_h_field_is_per_format(tmp_path):
    """The default nudge must reach the labelled readers and not the flat one, which cannot
    place a field without the header's basis/spin_ordering guarantees. An explicit field on a
    flat file that does not declare them still raises."""
    h = _flat_matrix()
    path = tmp_path / "model.h0"
    h0_format.write_h0_file(path, h, impurity_orbitals={2: list(range(10))})

    assert load_model(path, l=2, slater=D_SHELL_SLATER).n_spin_orbitals == 14
    with pytest.raises(ValueError, match="basis"):
        load_model(path, l=2, slater=D_SHELL_SLATER, h_field=(0.0, 0.0, 1e-4))


def test_unrecognised_extension_names_the_supported_formats(tmp_path):
    path = tmp_path / "model.xyz"
    path.write_text("nonsense\n")
    with pytest.raises(ValueError, match=r"\.h0.*\.pickle"):
        load_model(path, l=2, slater=D_SHELL_SLATER)


def test_labelled_format_without_its_required_arguments_raises(tmp_path):
    path = tmp_path / "model.pickle"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="l, n_baths and slater"):
        load_model(path, l=2)


def test_read_h0_operator_points_a_flat_file_at_the_right_subcommand(tmp_path):
    """`spectra` goes through read_h0_operator, which cannot interpret a flat file."""
    path = tmp_path / "model.h0"
    h0_format.write_h0_file(path, _flat_matrix(), impurity_orbitals={2: list(range(10))})

    with pytest.raises(RuntimeError, match="selfenergy.*susceptibility"):
        hamiltonian_io.read_h0_operator(str(path), {2: 4})


def test_sniffing_does_not_misroute_a_labelled_dat(tmp_path):
    """A labelled .dat must still reach the labelled reader, not the flat one."""
    path = tmp_path / "labelled.dat"
    path.write_text("# a comment\nOpName\n(2, 0, -2) (2, 0, -2) -1.5 0.0\n")
    assert not h0_format.looks_like_flat_terms(path)
    assert os.path.splitext(path)[1] == ".dat"
