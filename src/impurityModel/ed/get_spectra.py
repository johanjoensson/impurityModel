"""
Script for calculating various spectra.

"""

import argparse
import time
from collections import OrderedDict

import h5py
import numpy as np
from mpi4py import MPI

# Local stuff
from impurityModel.ed import spectra
from impurityModel.ed.average import k_B
from impurityModel.ed.block_structure import BlockStructure
from impurityModel.ed.hamiltonian_io import get_hamiltonian_operator
from impurityModel.ed.symmetries import (
    extract_tensors,
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)
from impurityModel.ed.groundstate import calc_gs
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator


def main(
    h0_filename,
    radial_filename,
    ls,
    nBaths,
    nValBaths,
    n0imps,
    dnTols,
    dnValBaths,
    dnConBaths,
    Fdd,
    Fpp,
    Fpd,
    Gpd,
    xi_2p,
    xi_3d,
    chargeTransferCorrection,
    hField,
    nPsiMax,
    nPrintSlaterWeights,
    tolPrintOccupation,
    T,
    energy_cut,
    delta,
    deltaRIXS,
    deltaNIXS,
    XAS_projectors_filename,
    RIXS_projectors_filename,
    auto_block_structure=True,
):
    """
    First find the lowest eigenstates and then use them to calculate various spectra.

    Parameters
    ----------
    h0_filename : str
        Filename of the non-relativistic non-interacting Hamiltonian operator, in pickle-format.
    radial_filename : str
        File name of file containing radial mesh and radial part of final
        and initial orbitals in the NIXS excitation process.
    ls : tuple
        Angular momenta of correlated orbitals.
    nBaths : tuple
        Number of bath states,
        for each angular momentum.
    nValBaths : tuple
        Number of valence bath states,
        for each angular momentum.
    n0imps : tuple
        Initial impurity occupation.
    dnTols : tuple
        Max devation from initial impurity occupation,
        for each angular momentum.
    dnValBaths : tuple
        Max number of electrons to leave valence bath orbitals,
        for each angular momentum.
    dnConBaths : tuple
        Max number of electrons to enter conduction bath orbitals,
        for each angular momentum.
    Fdd : tuple
        Slater-Condon parameters Fdd. This assumes d-orbitals.
    Fpp : tuple
        Slater-Condon parameters Fpp. This assumes p-orbitals.
    Fpd : tuple
        Slater-Condon parameters Fpd. This assumes p- and d-orbitals.
    Gpd : tuple
        Slater-Condon parameters Gpd. This assumes p- and d-orbitals.
    xi_2p : float
        SOC value for p-orbitals. This assumes p-orbitals.
    xi_3d : float
        SOC value for d-orbitals. This assumes d-orbitals.
    chargeTransferCorrection : float
        Double counting parameter
    hField : tuple
        Magnetic field.
    nPsiMax : int
        Maximum number of eigenstates to consider.
    nPrintSlaterWeights : int
        Printing parameter.
    tolPrintOccupation : float
        Printing parameter.
    T : float
        Temperature (Kelvin)
    energy_cut : float
        How many k_B*T above lowest eigenenergy to consider.
    delta : float
        Smearing, half width half maximum (HWHM). Due to short core-hole lifetime.
    deltaRIXS : float
        Smearing, half width half maximum (HWHM).
        Due to finite lifetime of excited states.
    deltaNIXS : float
        Smearing, half width half maximum (HWHM).
        Due to finite lifetime of excited states.
    RIXS_projectors_filename : string
        File containing the RIXS projectors, separated by an empty line.

    """

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank

    verbosity = 2 if rank == 0 else 0
    500
    rot_to_spherical = {1: np.eye(6, dtype=complex), 2: np.eye(10, dtype=complex)}
    block_structure = BlockStructure(
        blocks=[list(range(6)), list(range(6, 16))],
        identical_blocks=[[i] for i in range(2)],
        transposed_blocks=[[] for _ in range(2)],
        particle_hole_blocks=[[] for _ in range(2)],
        particle_hole_transposed_blocks=[[] for _ in range(2)],
        inequivalent_blocks=list(range(2)),
    )

    if rank == 0:
        time.perf_counter()

    # -- System information --
    nBaths = OrderedDict(zip(ls, nBaths))
    nValBaths = OrderedDict(zip(ls, nValBaths))
    impurity_orbitals = {}
    valence_baths = {}
    conduction_baths = {}
    offset = 0
    for l in ls:
        impurity_orbitals[l] = [[offset + i for i in range(2 * (2 * l + 1))]]
        offset += 2 * (2 * l + 1)
        valence_baths[l] = [[offset + i for i in range(nValBaths[l])]]
        offset += nValBaths[l]
        conduction_baths[l] = [[offset + i for i in range(nBaths[l] - nValBaths[l])]]
        offset += nBaths[l] - nValBaths[l]

    if rank == 0:
        print("Orbital layout (spin-orbital indices):")
        for l in ls:
            print(
                f"  l = {l}: impurity {impurity_orbitals[l]}, "
                f"valence bath {valence_baths[l]}, conduction bath {conduction_baths[l]}"
            )
    # -- Basis occupation information --
    n0imps = OrderedDict(zip(ls, n0imps))
    dnTols = OrderedDict(zip(ls, dnTols))
    dnValBaths = OrderedDict(zip(ls, dnValBaths))
    dnConBaths = OrderedDict(zip(ls, dnConBaths))

    # -- Spectra information --
    # Energy cut in eV.
    energy_cut *= k_B * T
    # XAS parameters
    # Energy-mesh
    # w = np.linspace(-35, 35, 4000)
    w = np.linspace(-25, 25, 3001)
    # Each element is a XAS polarization vector.
    epsilons = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # [[0,0,1]]
    # epsilons = [[ 1./np.sqrt(2), 1.j/np.sqrt(2), 0.], [ 1/np.sqrt(2), -1j/np.sqrt(2), 0.]] # [[0,0,1]]
    # epsilons = [[0.,  -1./np.sqrt(2), -1.j/np.sqrt(2)], [0., 1/np.sqrt(2), -1j/np.sqrt(2)]] # [[0,0,1]]
    # RIXS parameters
    # Polarization vectors, of in and outgoing photon.
    # epsilonsRIXSin = [[ 0., -1./np.sqrt(2), -1.j/np.sqrt(2)], [ 0., 1./np.sqrt(2), -1.j/np.sqrt(2) ]] # [[0,0,1]]
    # epsilonsRIXSin = [[ -1./np.sqrt(2), -1.j/np.sqrt(2), 0.], [1/np.sqrt(2), -1j/np.sqrt(2), 0.]] # [[0,0,1]]
    epsilonsRIXSin = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # x, y, z, cl, cr
    # epsilonsRIXSout = [[ -1./np.sqrt(2), -1.j/np.sqrt(2), 0], [  1./np.sqrt(2), -1.j/np.sqrt(2), 0], [0, 0, 1]] # [[0,0,1]]
    # epsilonsRIXSout = [[1./np.sqrt(2), 1.j/np.sqrt(2), 0], [1./np.sqrt(2), -1.j/np.sqrt(2), 0], [0, 0, 1]] # [[0,0,1]]
    epsilonsRIXSout = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # x, y, z, cl, cr
    if deltaRIXS > 0:
        wIn = np.linspace(-10, 20, 50)
        # wIn = np.linspace(-1, 2, 500)
    else:
        wIn = []
    wLoss = np.linspace(-2.0, 12.0, 4000)
    # wLoss = np.linspace(-1.0, 2.5, 3000)

    # Read XAS and/or RIXS projectors from file
    XAS_projectors = None
    RIXS_projectors = None
    # if XAS_projectors_filename:
    #     XAS_projectors = get_XAS_projectors(XAS_projectors_filename)
    #     if rank == 0:
    #         print("XAS projectors")
    #         print(XAS_projectors)
    # if RIXS_projectors_filename:
    #     RIXS_projectors = get_XAS_projectors(RIXS_projectors_filename)
    #     if rank == 0:
    #         print("RIXS projectors")
    #         print(RIXS_projectors)

    # NIXS parameters
    qsNIXS = [2 * np.array([1, 1, 1]) / np.sqrt(3), 7 * np.array([1, 1, 1]) / np.sqrt(3)]
    # Angular momentum of final and initial orbitals in the NIXS excitation process.
    liNIXS, ljNIXS = 2, 2

    # -- Occupation restrictions for excited states --
    l = 2

    # Read the radial part of correlated orbitals
    radialMesh, RiNIXS = np.loadtxt(radial_filename).T
    RjNIXS = np.copy(RiNIXS)

    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    if rank == 0:
        print(f"Number of spin-orbitals: {n_spin_orbitals}")

    # Hamiltonian
    if rank == 0:
        print("Constructing the Hamiltonian operator ...")
    hOp = get_hamiltonian_operator(
        nBaths,
        nValBaths,
        [Fdd, Fpp, Fpd, Gpd],
        [xi_2p, xi_3d],
        [n0imps, chargeTransferCorrection],
        hField,
        h0_filename,
        rank,
    )
    hOp = ManyBodyOperator(hOp)
    # Default: derive the block structure from the hybridization-dressed impurity matrix
    # (impurity_block_structure) rather than the hand-coded one. It matches or strictly refines
    # the manual structure (e.g. SOC / crystal field splits each shell into sub-blocks) and
    # fixes bath-mediated coupling. Pass auto_block_structure=False to keep the hand-coded one.
    #
    # Adaptive symmetry-adapted solver basis: rotate the correlated 3d shell into the basis that
    # diagonalises its one-body block, IF that keeps the Coulomb term roughly as sparse (the
    # fill-ratio gate; a d-shell with SOC densifies ~8x and stays spherical). The scalar XAS /
    # PES / NIXS / RIXS spectra are basis-invariant, so simulate_spectra just rotates the one-body
    # transition operators to match and deduplicates the now-degenerate PES/IPS operators (B2a).
    rotation = None
    correlated_block_structure = None
    correlated_l = 2
    if auto_block_structure:
        impurity_indices = sorted(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)
        h_matrix = extract_tensors(hOp, n_orb=n_spin_orbitals, two_body=False)[0]
        block_structure = impurity_block_structure(hOp, impurity_indices, h0_matrix=h_matrix)
        if rank == 0:
            print(f"Auto-derived block structure: {len(block_structure.blocks)} blocks")

        if correlated_l in impurity_orbitals:
            d_indices = sorted(orb for block in impurity_orbitals[correlated_l] for orb in block)
            W, u_imp = impurity_symmetry_rotation(hOp, d_indices, n_orb=n_spin_orbitals, h0_matrix=h_matrix)
            h_rotated = rotate_hamiltonian(hOp, W, tol=spectra._ROTATION_TRIM_TOL)
            fill_ratio = len(h_rotated) / max(1, len(hOp))
            if fill_ratio <= spectra._MAX_ROTATION_FILL:
                rotation = W
                hOp = h_rotated
                h_matrix = extract_tensors(hOp, n_orb=n_spin_orbitals, two_body=False)[0]
                block_structure = impurity_block_structure(hOp, impurity_indices, h0_matrix=h_matrix)
                correlated_block_structure = impurity_block_structure(hOp, d_indices, h0_matrix=h_matrix)
                # rot_to_spherical maps the (rotated) computational basis back to spherical harmonics
                # for the L/S/J Casimir reporting in calc_gs; identity on the un-rotated core p shell.
                rot_to_spherical = dict(rot_to_spherical)
                rot_to_spherical[correlated_l] = u_imp.conj().T
                if rank == 0:
                    n_classes = len(correlated_block_structure.inequivalent_blocks)
                    print(
                        f"Rotated 3d shell into symmetry-adapted basis (fill {fill_ratio:.2f}x); "
                        f"{n_classes} inequivalent PES/IPS classes."
                    )
            else:
                if rank == 0:
                    print(
                        f"Kept spherical basis (rotation would densify {fill_ratio:.2f}x > {spectra._MAX_ROTATION_FILL})."
                    )
                correlated_block_structure = impurity_block_structure(hOp, d_indices, h0_matrix=h_matrix)
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0:
        print(f"Hamiltonian contains {len(hOp)} terms.")
    # Many body basis for the ground state
    if rank == 0:
        print("Creating the many-body basis ...")
    tau = k_B * T
    basis_setup = {
        "impurity_orbital": impurity_orbitals,
        "bath_states": (valence_baths, conduction_baths),
        "nominal_impurity_occ": n0imps,
        "frozen_occupations": set(i for i in nBaths if nBaths[i] == 0),
        # None = "as many determinants as fit in RAM", resolved against the per-rank
        # available memory inside find_ground_state_basis (see memory_estimate).
        "truncation_threshold": None,
        "tau": tau,
        "comm": comm,
    }
    psis, es, basis, rho, _ = calc_gs(
        hOp,
        basis_setup,
        block_structure,
        rot_to_spherical,
        verbosity > 0,
    )

    # Save some information to disk
    h5f = None
    if rank == 0:
        # Most of the input parameters. Dictonaries can be stored in this file format.
        np.savez_compressed(
            "data",
            ls=ls,
            nBaths=nBaths,
            nValBaths=nValBaths,
            n0imps=n0imps,
            dnTols=dnTols,
            dnValBaths=dnValBaths,
            dnConBaths=dnConBaths,
            Fdd=Fdd,
            Fpp=Fpp,
            Fpd=Fpd,
            Gpd=Gpd,
            xi_2p=xi_2p,
            xi_3d=xi_3d,
            chargeTransferCorrection=chargeTransferCorrection,
            hField=hField,
            h0_filename=h0_filename,
            nPsiMax=nPsiMax,
            T=T,
            energy_cut=energy_cut,
            delta=delta,
            restrictions=basis.restrictions,
            epsilons=epsilons,
            epsilonsRIXSin=epsilonsRIXSin,
            epsilonsRIXSout=epsilonsRIXSout,
            deltaRIXS=deltaRIXS,
            deltaNIXS=deltaNIXS,
            n_spin_orbitals=n_spin_orbitals,
            hOp=hOp.to_dict(),
        )
    if rank == 0:
        # Save some of the arrays.
        # HDF5-format does not directly support dictonaries.
        h5f = h5py.File("spectra.h5", "w")
        h5f.create_dataset("E", data=es)
        h5f.create_dataset("w", data=w)
        h5f.create_dataset("wIn", data=wIn)
        h5f.create_dataset("wLoss", data=wLoss)
        h5f.create_dataset("qsNIXS", data=qsNIXS)
        h5f.create_dataset("r", data=radialMesh)
        h5f.create_dataset("RiNIXS", data=RiNIXS)
        h5f.create_dataset("RjNIXS", data=RjNIXS)

    if rank == 0:
        print("\n" + "=" * 80)
        print("  Spectra")
        print("=" * 80)
        print(f"Considering {len(es)} eigenstate(s) for the spectra.")
        print("Calculating spectra ...", flush=verbosity >= 2)
    spectra.simulate_spectra(
        es,
        psis,
        hOp,
        k_B * T,
        w,
        delta,
        epsilons,
        wLoss,
        deltaNIXS,
        qsNIXS,
        liNIXS,
        ljNIXS,
        RiNIXS,
        RjNIXS,
        radialMesh,
        wIn,
        deltaRIXS,
        epsilonsRIXSin,
        epsilonsRIXSout,
        basis.restrictions,
        h5f,
        nBaths,
        XAS_projectors,
        RIXS_projectors,
        basis,
        1e-6,
        2,
        np.sqrt(np.finfo(float).eps),
        verbosity >= 1,
        rotation=rotation,
        correlated_l=correlated_l,
        correlated_block_structure=correlated_block_structure,
    )

    if comm is not None:
        comm.Barrier()
    if rank == 0:
        print("\nDone.")


if __name__ == "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser(description="Spectroscopy simulations")
    parser.add_argument(
        "h0_filename",
        type=str,
        help="Filename of non-interacting Hamiltonian, in pickle-format.",
    )
    parser.add_argument(
        "radial_filename",
        type=str,
        help="Filename of radial part of correlated orbitals.",
    )
    parser.add_argument(
        "--ls",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Angular momenta of correlated orbitals.",
    )
    parser.add_argument(
        "--nBaths",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Number of bath states, for each angular momentum.",
    )
    parser.add_argument(
        "--nValBaths",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Number of valence bath states, for each angular momentum.",
    )
    parser.add_argument(
        "--n0imps",
        type=int,
        nargs="+",
        default=[6, 8],
        help="Initial impurity occupation, for each angular momentum.",
    )
    parser.add_argument(
        "--dnTols",
        type=int,
        nargs="+",
        default=[0, 2],
        help=("Max devation from initial impurity occupation, for each angular momentum."),
    )
    parser.add_argument(
        "--dnValBaths",
        type=int,
        nargs="+",
        default=[0, 2],
        help=("Max number of electrons to leave valence bath orbitals, for each angular momentum."),
    )
    parser.add_argument(
        "--dnConBaths",
        type=int,
        nargs="+",
        default=[0, 0],
        help=("Max number of electrons to enter conduction bath orbitals, for each angular momentum."),
    )
    parser.add_argument(
        "--Fdd",
        type=float,
        nargs="+",
        default=[7.5, 0, 9.9, 0, 6.6],
        help="Slater-Condon parameters Fdd. d-orbitals are assumed.",
    )
    parser.add_argument(
        "--Fpp",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0],
        help="Slater-Condon parameters Fpp. p-orbitals are assumed.",
    )
    parser.add_argument(
        "--Fpd",
        type=float,
        nargs="+",
        default=[8.9, 0, 6.8],
        help="Slater-Condon parameters Fpd. p- and d-orbitals are assumed.",
    )
    parser.add_argument(
        "--Gpd",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 0, 2.8],
        help="Slater-Condon parameters Gpd. p- and d-orbitals are assumed.",
    )
    parser.add_argument(
        "--xi_2p",
        type=float,
        default=11.629,
        help="SOC value for p-orbitals. p-orbitals are assumed.",
    )
    parser.add_argument(
        "--xi_3d",
        type=float,
        default=0.096,
        help="SOC value for d-orbitals. d-orbitals are assumed.",
    )
    parser.add_argument(
        "--chargeTransferCorrection",
        type=float,
        default=1.5,
        help="Double counting parameter.",
    )
    parser.add_argument(
        "--hField",
        type=float,
        nargs="+",
        default=[0, 0, 0.0001],
        help="Magnetic field. (h_x, h_y, h_z)",
    )
    parser.add_argument(
        "--nPsiMax",
        type=int,
        default=5,
        help="Maximum number of eigenstates to consider.",
    )
    parser.add_argument("--nPrintSlaterWeights", type=int, default=3, help="Printing parameter.")
    parser.add_argument("--tolPrintOccupation", type=float, default=0.5, help="Printing parameter.")
    parser.add_argument("--T", type=float, default=300, help="Temperature (Kelvin).")
    parser.add_argument(
        "--energy_cut",
        type=float,
        default=10,
        help="How many k_B*T above lowest eigenenergy to consider.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help=("Smearing, half width half maximum (HWHM). Due to short core-hole lifetime."),
    )
    parser.add_argument(
        "--deltaRIXS",
        type=float,
        default=0.050,
        help=("Smearing, half width half maximum (HWHM). Due to finite lifetime of excited states."),
    )
    parser.add_argument(
        "--deltaNIXS",
        type=float,
        default=0.100,
        help=("Smearing, half width half maximum (HWHM). Due to finite lifetime of excited states."),
    )
    parser.add_argument(
        "--XAS_projectors_filename",
        type=str,
        default=None,
        help=("File containing the XAS projectors. Separated by newlines."),
    )
    parser.add_argument(
        "--RIXS_projectors_filename",
        type=str,
        default=None,
        help=("File containing the RIXS projectors. Separated by newlines."),
    )

    args = parser.parse_args()

    # Sanity checks
    assert len(args.ls) == len(args.nBaths)
    assert len(args.ls) == len(args.nValBaths)
    for nBath, nValBath in zip(args.nBaths, args.nValBaths):
        assert nBath >= nValBath
    for ang, n0imp in zip(args.ls, args.n0imps):
        assert n0imp <= 2 * (2 * ang + 1)  # Full occupation
        assert n0imp >= 0
    assert len(args.Fdd) == 5
    assert len(args.Fpp) == 3
    assert len(args.Fpd) == 3
    assert len(args.Gpd) == 4
    assert len(args.hField) == 3
    # print("n0imps: ",args.n0imps)

    main(
        h0_filename=args.h0_filename,
        radial_filename=args.radial_filename,
        ls=tuple(args.ls),
        nBaths=tuple(args.nBaths),
        nValBaths=tuple(args.nValBaths),
        n0imps=tuple(args.n0imps),
        dnTols=tuple(args.dnTols),
        dnValBaths=tuple(args.dnValBaths),
        dnConBaths=tuple(args.dnConBaths),
        Fdd=tuple(args.Fdd),
        Fpp=tuple(args.Fpp),
        Fpd=tuple(args.Fpd),
        Gpd=tuple(args.Gpd),
        xi_2p=args.xi_2p,
        xi_3d=args.xi_3d,
        chargeTransferCorrection=args.chargeTransferCorrection,
        hField=tuple(args.hField),
        nPsiMax=args.nPsiMax,
        nPrintSlaterWeights=args.nPrintSlaterWeights,
        tolPrintOccupation=args.tolPrintOccupation,
        T=args.T,
        energy_cut=args.energy_cut,
        delta=args.delta,
        deltaRIXS=args.deltaRIXS,
        deltaNIXS=args.deltaNIXS,
        XAS_projectors_filename=args.XAS_projectors_filename,
        RIXS_projectors_filename=args.RIXS_projectors_filename,
    )
