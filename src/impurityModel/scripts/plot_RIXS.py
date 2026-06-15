import itertools as it
from argparse import ArgumentParser

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np


def run(prefix: str, *kwargs) -> None:
    """Read RIXS data from an HDF5 file and generate plots.

    Parameters
    ----------
    prefix : str
        Directory containing the spectra.h5 file.
    *kwargs : tuple
        Additional arguments.
    """
    RIXS = None
    wIn = None
    wLoss = None
    with h5.File(f"{prefix}/spectra.h5", "r") as ar:
        RIXS = np.array(ar["RIXSthermal"])
        wIn = np.array(ar["wIn"])
        wLoss = np.array(ar["wLoss"])
    print(f"{wIn.shape=} {wLoss.shape=} {RIXS.shape=}")
    print(f"{wIn[0]=} {wLoss[0]=}")
    nrows, ncols, _, _ = RIXS.shape
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    for i, j in it.product(range(nrows), range(ncols)):
        data = RIXS[i, j]
        min_val, max_val = np.min(data), np.max(data)

        ax[i, j].matshow(
            (data.T - min_val) / (max_val - min_val),
            extent=(wIn[0], wIn[-1], wLoss[0], wLoss[-1]),
            aspect="auto",
            norm="log",
        )
    # fig.subplots_adjust(right=0.8)

    # fig.colorbar(im, cbar_ax)
    # plt.tight_layout()
    plt.show()

    data = np.sum(RIXS, axis=(0, 1))
    min_val, max_val = np.min(data), np.max(data)
    plt.matshow(
        (data.T - min_val) / (max_val - min_val),
        extent=(wIn[0], wIn[-1], wLoss[0], wLoss[-1]),
        aspect="auto",
        norm="log",
    )
    plt.colorbar()
    plt.show()


def main() -> None:
    """Parse command line arguments and execute the plot generator.

    Reads incident photon energies and energy loss spectrum to plot the RIXS data.
    """
    parser = ArgumentParser(
        prog="Plot RIXS",
        description="Plot RIXS spectra calculated using impurityModel.",
    )
    parser.add_argument("-d", "--directory", type=str, default=".", dest="prefix")
    args = parser.parse_args()
    run(**args)
