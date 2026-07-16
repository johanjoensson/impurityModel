"""``impurityModel susceptibility`` -- dynamical impurity susceptibilities chi(w) / chi(i nu).

Builds an :class:`impurityModel.ed.model.ImpurityModel` from a non-interacting ``h0`` file and
runs :func:`impurityModel.ed.susceptibility.calc_susceptibility_workflow`, which writes a
``chi.h5`` file (one group per operator) and prints the static / screening-scale summary.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.model import BasisOptions, ImpurityModel, Meshes, SolverOptions, load_selfenergy_archive
from impurityModel.ed.susceptibility import calc_susceptibility_workflow


def add_arguments(parser):
    """Register the ``susceptibility`` sub-command arguments on ``parser``."""
    parser.add_argument(
        "h0_filename",
        type=str,
        nargs="?",
        default=None,
        help="Filename of non-interacting Hamiltonian (omit when using --from-archive).",
    )
    parser.add_argument(
        "--from-archive",
        type=str,
        default=None,
        metavar="PATH",
        help="Take the impurity model (and nominal occupation/tau) from an impurityModel_data.h5 archive.",
    )
    parser.add_argument("--cluster", type=str, default=None, help="Archive cluster label (with --from-archive).")
    parser.add_argument(
        "--iteration", type=int, default=None, help="Archive DMFT iteration (with --from-archive; default: last)."
    )
    parser.add_argument("--clustername", type=str, default="cluster", help="Label of the cluster.")
    parser.add_argument("--ls", type=int, default=2, help="Angular momentum of the correlated orbitals.")
    parser.add_argument("--nBaths", type=int, default=10, help="Total number of bath states.")
    parser.add_argument("--n0imps", type=int, default=8, help="Nominal impurity occupation.")
    parser.add_argument(
        "--Fdd", type=float, nargs="+", default=[7.5, 0, 9.9, 0, 6.6], help="Slater-Condon parameters Fdd."
    )
    parser.add_argument("--xi", type=float, default=0, help="SOC value for the correlated orbitals.")
    parser.add_argument("--hField", type=float, nargs="+", default=[0, 0, 0.0001], help="Magnetic field (x, y, z).")
    parser.add_argument("--nPsiMax", type=int, default=5, help="Maximum number of eigenstates to consider.")
    parser.add_argument("--tau", type=float, default=0.002, help="Fundamental temperature (kb*T).")
    parser.add_argument("--w_min", type=float, default=-5.0, help="Lower edge of the real frequency mesh (eV).")
    parser.add_argument("--w_max", type=float, default=5.0, help="Upper edge of the real frequency mesh (eV).")
    parser.add_argument("--w_n", type=int, default=501, help="Number of real mesh points.")
    parser.add_argument("--delta", type=float, default=0.01, help="Broadening above the real axis (eV).")
    parser.add_argument("--n_matsubara", type=int, default=64, help="Number of bosonic Matsubara points (0 disables).")
    parser.add_argument("--output", type=str, default="chi.h5", help="Output HDF5 filename.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Set verbose output.")


def run(args):
    """Build the model and option groups from ``args`` and run the susceptibility workflow."""
    comm = MPI.COMM_WORLD

    if args.from_archive:
        # The archive supplies the model plus its nominal occupation / mixed valence / tau; the
        # susceptibility-specific real mesh and Matsubara count still come from the CLI flags.
        model, _meshes, basis, _solver, cluster_label = load_selfenergy_archive(
            args.from_archive, cluster=args.cluster, iteration=args.iteration
        )
    else:
        if not args.h0_filename:
            raise SystemExit("Provide an h0 file (positional) or --from-archive PATH.")
        assert 0 <= args.n0imps <= 2 * (2 * args.ls + 1)
        assert len(args.hField) == 3
        ls = args.ls
        model = ImpurityModel.from_h0_file(
            args.h0_filename,
            l=ls,
            n_baths=args.nBaths,
            slater=args.Fdd,
            xi=args.xi,
            h_field=tuple(args.hField),
            rank=comm.rank,
            verbose=args.verbose,
        )
        basis = BasisOptions(nominal_occ={ls: args.n0imps}, mixed_valence={ls: 0}, tau=args.tau)
        cluster_label = args.clustername

    meshes = Meshes(w=np.linspace(args.w_min, args.w_max, args.w_n), delta=args.delta)
    solver = SolverOptions()

    calc_susceptibility_workflow(
        model,
        meshes,
        basis,
        solver,
        comm=comm,
        verbosity=2 if args.verbose else 0,
        cluster_label=cluster_label,
        num_wanted=args.nPsiMax,
        n_matsubara=args.n_matsubara,
        output_filename=args.output,
    )


def main():
    """Stand-alone entry point (``python -m impurityModel.scripts.susceptibility``)."""
    import argparse  # noqa: PLC0415 -- only needed on the stand-alone path

    parser = argparse.ArgumentParser(description="Calculate dynamical impurity susceptibilities chi(w) / chi(i nu).")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
