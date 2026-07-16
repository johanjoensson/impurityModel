"""``impurityModel spectra`` -- calculate PS/XPS/NIXS/XAS/RIXS spectra from an ``h0`` file.

Builds the full interacting model with :func:`impurityModel.ed.get_spectra.build_spectra_model`
and runs it through :func:`impurityModel.ed.get_spectra.run_spectra`, which writes ``spectra.h5``
in the current directory. Contract the stored polarization tensors at plot time with
``impurityModel plot-spectra`` / ``plot-rixs``.
"""

from collections import OrderedDict

import numpy as np
from mpi4py import MPI

from impurityModel.ed.average import k_B
from impurityModel.ed.get_spectra import build_spectra_model, run_spectra
from impurityModel.ed.model import BasisOptions, SpectraOptions


def add_arguments(parser):
    """Register the ``spectra`` sub-command arguments on ``parser``."""
    parser.add_argument("h0_filename", type=str, help="Filename of non-interacting Hamiltonian, in pickle-format.")
    parser.add_argument(
        "radial_filename",
        type=str,
        nargs="?",
        default=None,
        help="Filename of the radial part of the correlated orbitals (needed for NIXS; NIXS is skipped if omitted).",
    )
    parser.add_argument("--ls", type=int, nargs="+", default=[1, 2], help="Angular momenta of correlated orbitals.")
    parser.add_argument(
        "--nBaths", type=int, nargs="+", default=[0, 10], help="Number of bath states, for each angular momentum."
    )
    parser.add_argument(
        "--nValBaths",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Number of valence bath states, for each angular momentum.",
    )
    parser.add_argument(
        "--n0imps", type=int, nargs="+", default=[6, 8], help="Initial impurity occupation, for each angular momentum."
    )
    parser.add_argument(
        "--dnTols",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Accepted for backwards compatibility; does not affect the spectra basis.",
    )
    parser.add_argument(
        "--dnValBaths",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Accepted for backwards compatibility; does not affect the spectra basis.",
    )
    parser.add_argument(
        "--dnConBaths",
        type=int,
        nargs="+",
        default=[0, 0],
        help="Accepted for backwards compatibility; does not affect the spectra basis.",
    )
    parser.add_argument(
        "--Fdd", type=float, nargs="+", default=[7.5, 0, 9.9, 0, 6.6], help="Slater-Condon parameters Fdd (d-orbitals)."
    )
    parser.add_argument(
        "--Fpp", type=float, nargs="+", default=[0.0, 0.0, 0.0], help="Slater-Condon parameters Fpp (p-orbitals)."
    )
    parser.add_argument(
        "--Fpd", type=float, nargs="+", default=[8.9, 0, 6.8], help="Slater-Condon parameters Fpd (p- and d-orbitals)."
    )
    parser.add_argument(
        "--Gpd",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 0, 2.8],
        help="Slater-Condon parameters Gpd (p- and d-orbitals).",
    )
    parser.add_argument("--xi_2p", type=float, default=11.629, help="SOC value for p-orbitals.")
    parser.add_argument("--xi_3d", type=float, default=0.096, help="SOC value for d-orbitals.")
    parser.add_argument("--chargeTransferCorrection", type=float, default=1.5, help="Double counting parameter.")
    parser.add_argument(
        "--hField", type=float, nargs="+", default=[0, 0, 0.0001], help="Magnetic field. (h_x, h_y, h_z)"
    )
    parser.add_argument("--nPsiMax", type=int, default=5, help="Maximum number of eigenstates to consider.")
    parser.add_argument("--T", type=float, default=300, help="Temperature (Kelvin).")
    parser.add_argument(
        "--energy_cut", type=float, default=10, help="How many k_B*T above lowest eigenenergy to consider."
    )
    parser.add_argument("--delta", type=float, default=0.2, help="Smearing HWHM. Due to short core-hole lifetime (eV).")
    parser.add_argument(
        "--deltaRIXS",
        type=float,
        default=0.050,
        help="Smearing HWHM for RIXS (eV); <= 0 disables the RIXS calculation.",
    )
    parser.add_argument("--deltaNIXS", type=float, default=0.100, help="Smearing HWHM for NIXS (eV).")
    parser.add_argument(
        "--truncation_threshold",
        type=int,
        default=None,
        help=(
            "Maximum number of Slater determinants in any many-body basis. "
            "Default: as many as fit in RAM (see memory_estimate)."
        ),
    )
    parser.add_argument(
        "--no-auto-block-structure",
        dest="auto_block_structure",
        action="store_false",
        help="Keep the hand-coded 2p/3d block structure instead of deriving it from the hybridization.",
    )
    parser.set_defaults(auto_block_structure=True)


def _validate(args):
    assert len(args.ls) == len(args.nBaths)
    assert len(args.ls) == len(args.nValBaths)
    for nBath, nValBath in zip(args.nBaths, args.nValBaths):
        assert nBath >= nValBath
    for ang, n0imp in zip(args.ls, args.n0imps):
        assert 0 <= n0imp <= 2 * (2 * ang + 1)
    assert len(args.Fdd) == 5
    assert len(args.Fpp) == 3
    assert len(args.Fpd) == 3
    assert len(args.Gpd) == 4
    assert len(args.hField) == 3


def run(args):
    """Build the model and option groups from ``args`` and run the spectra calculation."""
    _validate(args)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    verbosity = 2 if rank == 0 else 0

    model = build_spectra_model(
        args.h0_filename,
        tuple(args.ls),
        tuple(args.nBaths),
        tuple(args.nValBaths),
        tuple(args.n0imps),
        tuple(args.Fdd),
        tuple(args.Fpp),
        tuple(args.Fpd),
        tuple(args.Gpd),
        args.xi_2p,
        args.xi_3d,
        args.chargeTransferCorrection,
        tuple(args.hField),
        rank=rank,
        verbose=verbosity > 0,
    )

    # The radial part of the correlated orbitals is only needed for NIXS; skip NIXS when absent.
    if args.radial_filename:
        radial_mesh, radial_i = np.loadtxt(args.radial_filename).T
        radial = (radial_mesh, radial_i, np.copy(radial_i))
    else:
        radial = None

    spectra_options = SpectraOptions(
        delta=args.delta,
        deltaRIXS=args.deltaRIXS,
        deltaNIXS=args.deltaNIXS,
        radial=radial,
        energy_cut=args.energy_cut,
        nPsiMax=args.nPsiMax,
        auto_block_structure=args.auto_block_structure,
    )
    basis = BasisOptions(
        nominal_occ=OrderedDict(zip(args.ls, args.n0imps)),
        dN=2,
        truncation_threshold=args.truncation_threshold,
        occ_cutoff=1e-6,
        tau=k_B * args.T,
    )
    run_spectra(model, spectra_options, basis, comm, verbosity=verbosity)


def main():
    """Stand-alone entry point (``python -m impurityModel.scripts.spectra``)."""
    import argparse  # noqa: PLC0415 -- only needed on the stand-alone path

    parser = argparse.ArgumentParser(description="Spectroscopy simulations (PS/XPS/NIXS/XAS/RIXS).")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
