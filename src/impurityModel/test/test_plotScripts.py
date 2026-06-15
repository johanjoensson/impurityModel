import pytest
import h5py
import numpy as np
import os
from unittest.mock import patch
from impurityModel.plotScripts.plotSpectra import plot_spectra_in_file
from impurityModel.plotScripts.plotRIXS import main as plot_rixs_main


@pytest.fixture
def mock_h5_file(tmp_path):
    filepath = tmp_path / "mock_spectra.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("w", data=np.linspace(-10, 10, 10))
        f.create_dataset("wIn", data=np.linspace(0, 10, 5))
        f.create_dataset("wLoss", data=np.linspace(-2, 8, 8))
        f.create_dataset("qsNIXS", data=np.random.rand(3, 3))
        f.create_dataset("r", data=np.linspace(0, 5, 20))
        f.create_dataset("RiNIXS", data=np.random.rand(20))
        f.create_dataset("RjNIXS", data=np.random.rand(20))
        f.create_dataset("PSthermal", data=np.random.rand(5, 10))
        f.create_dataset("XPSthermal", data=np.random.rand(5, 10))
        f.create_dataset("XASthermal", data=np.random.rand(3, 10))
        f.create_dataset("RIXSthermal", data=np.random.rand(3, 3, 5, 8))
        f.create_dataset("NIXSthermal", data=np.random.rand(3, 8))
    return str(filepath)


@pytest.fixture
def mock_bin_file(tmp_path):
    filepath = tmp_path / "mock_rixs.bin"
    # Create a small binary file with float32 mimicking plotRIXS expectations
    # x[0] = ncols = 5 (wIn length)
    # total elements = (ncols+1) * (nrows+1) = 6 * 4 = 24
    data = np.random.rand(24).astype(np.float32)
    data[0] = 5
    data.tofile(filepath)
    return str(filepath)


@patch("matplotlib.pyplot.show")
def test_plotSpectra(mock_show, mock_h5_file):
    plot_spectra_in_file(mock_h5_file)
    assert mock_show.called


@patch("sys.argv", ["plotRIXS.py", "mock_rixs.bin"])
@patch("matplotlib.pyplot.show")
def test_plotRIXS(mock_show, mock_bin_file):
    with patch("sys.argv", ["plotRIXS.py", mock_bin_file]):
        plot_rixs_main()
    assert mock_show.called
