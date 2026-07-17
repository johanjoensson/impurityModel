"""``impurityModel selfenergy`` -- impurity self-energy Sigma(w) / Sigma(i nu) and G_imp.

Builds an :class:`impurityModel.ed.model.ImpurityModel` from a non-interacting ``h0`` file and
runs :func:`impurityModel.ed.selfenergy.calc_selfenergy`. On rank 0 the results are written to
disk: the frequency-dependent self-energy and impurity Green's function in RSPt ``.dat`` format
(:func:`impurityModel.ed.greens_function.save_Greens_function`), the static self-energy to a
``.dat`` file, and everything into a per-cluster HDF5 archive.
"""

from dataclasses import replace

import numpy as np
from mpi4py import MPI

from impurityModel.ed.model import BasisOptions, ImpurityModel, Meshes, SolverOptions, load_selfenergy_archive
from impurityModel.ed.selfenergy import calc_selfenergy


def add_arguments(parser):
    """Register the ``selfenergy`` sub-command arguments on ``parser``."""
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
        help="Reconstruct the whole run (model, meshes, options) from an impurityModel_data.h5 archive.",
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
    parser.add_argument("--tau", type=float, default=0.002, help="Fundamental temperature (kb*T).")

    # Real-frequency mesh.
    parser.add_argument("--w_min", type=float, default=-10.0, help="Lower edge of the real frequency mesh (eV).")
    parser.add_argument("--w_max", type=float, default=10.0, help="Upper edge of the real frequency mesh (eV).")
    parser.add_argument("--w_n", type=int, default=2001, help="Number of real mesh points.")
    parser.add_argument("--delta", type=float, default=0.1, help="Broadening above the real axis (eV).")
    parser.add_argument("--no-realaxis", dest="realaxis", action="store_false", help="Skip the real-frequency output.")
    # Matsubara mesh (fermionic; i*nu_n with nu_n = (2n+1)*pi*tau).
    parser.add_argument(
        "--n_matsubara", type=int, default=0, help="Number of fermionic Matsubara points (0 disables Matsubara)."
    )

    # Basis / solver knobs.
    parser.add_argument("--dN", type=int, default=None, help="Impurity occupation window (+-dN) for the excited bases.")
    parser.add_argument(
        "--truncation_threshold", type=int, default=None, help="Determinant budget (default: as many as fit in RAM)."
    )
    parser.add_argument(
        "--excitation_budget",
        type=int,
        default=None,
        help=(
            "Maximum total bath excitations per determinant (holes in filled-valence + "
            "electrons in empty-conduction orbitals). Default: unset. A memory lever; validate "
            "accuracy on the target system (see doc/plans/restrictions_redux.md)."
        ),
    )
    parser.add_argument(
        "--no-chain-restrict",
        dest="chain_restrict",
        action="store_false",
        help="Disable chain occupation restrictions.",
    )
    parser.add_argument("--reort", type=str, default=None, help="Block-Lanczos reorthogonalization mode.")
    parser.add_argument(
        "--gf-method",
        type=str,
        default="lanczos",
        choices=["lanczos", "bicgstab", "sliced", "cipsi"],
        help="Green's-function kernel.",
    )
    parser.add_argument("--dense-cutoff", type=int, default=500, help="Use a dense eigensolver below this size.")
    parser.add_argument(
        "--no-sparse-green", dest="sparse_green", action="store_false", help="Disable the sparse block-Lanczos path."
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Per-cluster HDF5 output (default: selfenergy-<cluster>.h5)."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Set verbose output.")
    parser.set_defaults(realaxis=True, chain_restrict=True, sparse_green=True)


def _fermionic_matsubara(tau, n_points):
    """Complex fermionic Matsubara mesh ``i*nu_n``, ``nu_n = (2n+1)*pi*tau``, ``n = 0..n_points-1``."""
    return 1j * (2 * np.arange(n_points) + 1) * np.pi * tau


def _save_static(sigma_static, cluster_label):
    """Write the static (Hartree-Fock) self-energy matrix to ``sigma_static-<cluster>.dat`` (real, imag)."""
    filename = f"sigma_static-{cluster_label}.dat"
    flat = np.asarray(sigma_static).reshape(-1)
    np.savetxt(
        filename,
        np.column_stack([np.real(flat), np.imag(flat)]),
        header=f"Static self-energy ({sigma_static.shape[0]}x{sigma_static.shape[1]}), row-major: Re Im",
    )
    print(f"Wrote static self-energy to {filename}")


def _save_results(result, meshes, cluster_label, output):
    """Rank-0 saving: RSPt ``.dat`` files for Sigma/G, static Sigma, and a per-cluster HDF5 archive."""
    import h5py  # noqa: PLC0415 -- only the save path needs it

    from impurityModel.ed.greens_function import save_Greens_function  # noqa: PLC0415

    if meshes.iw is not None and result["sigma"] is not None:
        save_Greens_function(result["sigma"], meshes.iw, "Sigma", cluster_label)
        save_Greens_function(result["gs_matsubara"], meshes.iw, "Gimp", cluster_label)
    if meshes.w is not None and result["sigma_real"] is not None:
        save_Greens_function(result["sigma_real"], meshes.w, "Sigma", cluster_label)
        save_Greens_function(result["gs_realaxis"], meshes.w, "Gimp", cluster_label)
    _save_static(result["sigma_static"], cluster_label)

    output = output or f"selfenergy-{cluster_label}.h5"
    with h5py.File(output, "w") as f:
        g = f.create_group(cluster_label)
        g.create_dataset("Sigma Static", data=result["sigma_static"])
        g.create_dataset("thermal_rho", data=result["thermal_rho"])
        g.create_dataset("gs_energies", data=result["gs_energies"])
        if meshes.iw is not None and result["sigma"] is not None:
            g.create_dataset("iw", data=meshes.iw)
            g.create_dataset("Sigma Matsubara", data=result["sigma"])
            g.create_dataset("Gimp Matsubara", data=result["gs_matsubara"])
        if meshes.w is not None and result["sigma_real"] is not None:
            g.create_dataset("w", data=meshes.w)
            g.create_dataset("Sigma real", data=result["sigma_real"])
            g.create_dataset("Gimp real", data=result["gs_realaxis"])
    print(f"Wrote self-energy archive to {output} (cluster '{cluster_label}').")


def run(args):
    """Build the model and option groups from ``args``, solve, and save the results on rank 0."""
    comm = MPI.COMM_WORLD
    verbosity = 2 if args.verbose else 0

    if args.from_archive:
        # The archive carries the model, both meshes and every recorded basis/solver option;
        # --gf-method still overrides the kernel (it is not part of the recorded run).
        model, meshes, basis, solver, cluster_label = load_selfenergy_archive(
            args.from_archive, cluster=args.cluster, iteration=args.iteration
        )
        solver = replace(solver, gf_method=args.gf_method)
    else:
        if not args.h0_filename:
            raise SystemExit("Provide an h0 file (positional) or --from-archive PATH.")
        assert 0 <= args.n0imps <= 2 * (2 * args.ls + 1)
        assert len(args.hField) == 3
        if not args.realaxis and args.n_matsubara <= 0:
            raise SystemExit("Nothing to compute: enable the real axis or request Matsubara points (--n_matsubara).")
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
        w = np.linspace(args.w_min, args.w_max, args.w_n) if args.realaxis else None
        iw = _fermionic_matsubara(args.tau, args.n_matsubara) if args.n_matsubara > 0 else None
        meshes = Meshes(iw=iw, w=w, delta=args.delta)
        basis = BasisOptions(
            nominal_occ={ls: args.n0imps},
            mixed_valence={ls: 0},
            dN=args.dN,
            truncation_threshold=args.truncation_threshold,
            chain_restrict=args.chain_restrict,
            tau=args.tau,
            excitation_budget=args.excitation_budget,
        )
        solver = SolverOptions(
            reort=args.reort,
            dense_cutoff=args.dense_cutoff,
            sparse_green=args.sparse_green,
            gf_method=args.gf_method,
        )
        cluster_label = args.clustername

    result = calc_selfenergy(model, meshes, basis, solver, comm=comm, verbosity=verbosity, cluster_label=cluster_label)
    if comm.rank == 0 and result is not None:
        _save_results(result, meshes, cluster_label, args.output)


def main():
    """Stand-alone entry point (``python -m impurityModel.scripts.selfenergy``)."""
    import argparse  # noqa: PLC0415 -- only needed on the stand-alone path

    parser = argparse.ArgumentParser(description="Calculate the impurity self-energy Sigma(w) / Sigma(i nu).")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
