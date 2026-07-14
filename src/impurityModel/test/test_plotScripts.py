import argparse
from unittest.mock import patch

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from impurityModel.scripts.plot_RIXS import main as plot_rixs_main
from impurityModel.scripts.plot_RIXS import run as plot_rixs_run
from impurityModel.scripts.plot_spectra import main as plot_spectra_main
from impurityModel.scripts.plot_spectra import plot_spectra_in_file


@pytest.fixture(autouse=True)
def _close_figures():
    # Each plot test opens several figures and never closes them (plt.show is mocked out);
    # across the whole module that trips matplotlib's "too many open figures" warning, which
    # pytest.ini promotes to a hard error.
    yield
    plt.close("all")


def _hermitian_tensor(rng, n_w, m):
    A = rng.standard_normal((n_w, m, m)) + 1j * rng.standard_normal((n_w, m, m))
    return A + np.conj(np.swapaxes(A, 1, 2))


def _physical_rixs_tensor(rng, n_in, n_out, n_wIn, n_wLoss):
    """A rank-4 RIXS tensor whose flattened (in,out) block is -i*A A^H per (wIn, wLoss) point,
    matching a resolvent's negative-semidefinite imaginary part (so intensity = -Im is
    non-negative and the plot CLIs have something to show)."""
    n_pair = n_in * n_out
    flat = np.empty((n_wIn, n_wLoss, n_pair, n_pair), dtype=complex)
    for i in range(n_wIn):
        A = rng.standard_normal((n_wLoss, n_pair, n_pair)) + 1j * rng.standard_normal((n_wLoss, n_pair, n_pair))
        flat[i] = -1j * np.einsum("wij,wkj->wik", A, A.conj())
    reshaped = flat.reshape(n_wIn, n_wLoss, n_in, n_out, n_in, n_out)
    return np.transpose(reshaped, (2, 3, 4, 5, 0, 1))


@pytest.fixture
def mock_h5_file(tmp_path):
    rng = np.random.default_rng(0)
    n_w, n_orb, n_q = 12, 4, 2
    n_wIn, n_wLoss = 5, 9
    filepath = tmp_path / "mock_spectra.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("E", data=np.array([0.0, 0.1]))
        f.create_dataset("w", data=np.linspace(-10, 10, n_w))
        f.create_dataset("wIn", data=np.linspace(0, 10, n_wIn))
        f.create_dataset("wLoss", data=np.linspace(-2, 8, n_wLoss))
        f.create_dataset("qsNIXS", data=np.random.rand(n_q, 3))
        f.create_dataset("r", data=np.linspace(0, 5, 20))
        f.create_dataset("RiNIXS", data=np.random.rand(20))
        f.create_dataset("RjNIXS", data=np.random.rand(20))
        f.create_dataset("PS/spectra", data=np.random.rand(n_w, n_orb) + 1j * np.random.rand(n_w, n_orb))
        f.create_dataset("XPS/spectra", data=np.random.rand(n_w, n_orb) + 1j * np.random.rand(n_w, n_orb))
        f.create_dataset("NIXS/spectra", data=np.random.rand(n_wLoss, n_q) + 1j * np.random.rand(n_wLoss, n_q))
        f.create_dataset("XAS/tensor", data=_hermitian_tensor(rng, n_w, 3))
        f.create_dataset("RIXS/tensor", data=_physical_rixs_tensor(rng, 3, 3, n_wIn, n_wLoss).astype(np.complex64))
    return str(filepath)


@pytest.fixture
def mock_h5_file_projected(tmp_path):
    """A file produced by the XAS_projectors/RIXS_projectors code paths: already
    polarization-resolved, no Cartesian tensor."""
    n_w, n_proj = 10, 2
    n_wIn, n_wLoss = 4, 6
    filepath = tmp_path / "mock_spectra_projected.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("w", data=np.linspace(-5, 5, n_w))
        f.create_dataset("wIn", data=np.linspace(0, 5, n_wIn))
        f.create_dataset("wLoss", data=np.linspace(-1, 4, n_wLoss))
        f.create_dataset("XAS/projected", data=-1j * np.random.rand(n_w, n_proj))
        f.create_dataset("RIXS/projected", data=-1j * np.random.rand(n_proj, n_proj, n_wIn, n_wLoss))
    return str(filepath)


def _spectra_args(**overrides):
    defaults = dict(pol=None, xmcd=False, xld=False, tensor_components=False, orbitals=None, export=None, output=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _rixs_args(**overrides):
    defaults = dict(
        filename="unused",
        cutoff=-1e12,  # accept everything; the synthetic tensors are physically signed but tiny
        pol_in=None,
        pol_out=None,
        mcd=False,
        fy=False,
        emission=False,
        cuts=None,
        export=None,
        output=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@patch("matplotlib.pyplot.show")
def test_plot_spectra_default(mock_show, mock_h5_file):
    plot_spectra_in_file(mock_h5_file)
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_spectra_dichroism_and_tensor_components(mock_show, mock_h5_file):
    args = _spectra_args(xmcd=True, xld=True, tensor_components=True, pol=["x", "z"])
    plot_spectra_in_file(mock_h5_file, args)
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_spectra_orbital_grouping(mock_show, mock_h5_file):
    args = _spectra_args(orbitals="0-1,2+3")
    plot_spectra_in_file(mock_h5_file, args)
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_spectra_export(mock_show, mock_h5_file, tmp_path):
    prefix = str(tmp_path / "export")
    args = _spectra_args(export=prefix)
    plot_spectra_in_file(mock_h5_file, args)
    assert (tmp_path / "export-PS.dat").exists()
    assert (tmp_path / "export-XAS.dat").exists()


@patch("matplotlib.pyplot.show")
def test_plot_spectra_projected_xas(mock_show, mock_h5_file_projected):
    plot_spectra_in_file(mock_h5_file_projected)
    assert mock_show.called


@patch("sys.argv", ["plot_spectra"])
@patch("matplotlib.pyplot.show")
def test_plot_spectra_main_entry_point(mock_show, mock_h5_file, monkeypatch):
    monkeypatch.chdir("/".join(mock_h5_file.split("/")[:-1]))
    with patch("sys.argv", ["plot_spectra", "--filename", mock_h5_file]):
        plot_spectra_main()
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_rixs_tensor_default(mock_show, mock_h5_file):
    args = _rixs_args(filename=mock_h5_file)
    plot_rixs_run(args)
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_rixs_tensor_polarization_grid_and_mcd(mock_show, mock_h5_file):
    args = _rixs_args(filename=mock_h5_file, pol_in=["x", "y"], pol_out=["x", "y"], mcd=True)
    plot_rixs_run(args)
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_rixs_tensor_fy_and_cuts(mock_show, mock_h5_file):
    args = _rixs_args(filename=mock_h5_file, fy=True, cuts=[0.5, 5.0], emission=True)
    plot_rixs_run(args)
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_rixs_projected_ignores_polarization_flags(mock_show, mock_h5_file_projected):
    args = _rixs_args(filename=mock_h5_file_projected, pol_in=["x"], pol_out=["x"])
    plot_rixs_run(args)  # warns but does not crash
    assert mock_show.called


def test_plot_rixs_mcd_requires_tensor(mock_h5_file_projected):
    args = _rixs_args(filename=mock_h5_file_projected, mcd=True)
    with pytest.raises(SystemExit):
        plot_rixs_run(args)


def test_plot_rixs_missing_data(tmp_path):
    filepath = tmp_path / "empty.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("w", data=np.linspace(-1, 1, 3))
    args = _rixs_args(filename=str(filepath))
    with pytest.raises(SystemExit):
        plot_rixs_run(args)


@patch("matplotlib.pyplot.show")
def test_plot_rixs_export(mock_show, mock_h5_file, tmp_path):
    prefix = str(tmp_path / "export")
    args = _rixs_args(filename=mock_h5_file, fy=True, export=prefix)
    plot_rixs_run(args)
    assert (tmp_path / "export-RIXS-wLoss-sum.dat").exists()
    assert (tmp_path / "export-RIXS-FY.dat").exists()


@patch("matplotlib.pyplot.show")
def test_plot_rixs_main_entry_point(mock_show, mock_h5_file):
    with patch("sys.argv", ["plot_RIXS", "--filename", mock_h5_file]):
        plot_rixs_main()
    assert mock_show.called


def test_plot_rixs_missing_file_raises():
    with patch("sys.argv", ["plot_RIXS", "--filename", "does-not-exist.h5"]), pytest.raises(SystemExit):
        plot_rixs_main()
