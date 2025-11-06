"""
Script for calculating various spectra.

"""

import itertools
import numpy as np
import scipy.sparse.linalg
from collections import OrderedDict
import sys, os
from mpi4py import MPI
import argparse
import pickle
import time
from collections import OrderedDict

import h5py
import numpy as np
from mpi4py import MPI

# Local stuff
from impurityModel.ed import finite, spectra
from impurityModel.ed.finite import assert_hermitian, c2i
from impurityModel.ed.average import k_B, thermal_average
from impurityModel.ed import op_parser
from impurityModel.ed.manybody_basis import CIPSI_Basis, Basis
from impurityModel.ed.block_structure import BlockStructure, print_block_structure
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState


def matrix_print(matrix: np.ndarray, label: str = None) -> None:
    """
    Pretty print the matrix, with optional label.
    """
    ms = "\n".join([" ".join([f"{np.real(val): .4f}{np.imag(val):+.4f}j" for val in row]) for row in matrix])
    if label is not None:
        print(label)
    print(ms)


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
    dense_cutoff = int(500)
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
        t0 = time.perf_counter()

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
        print(f"{impurity_orbitals=}")
        print(f"{valence_baths=}")
        print(f"{conduction_baths=}")
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
    if XAS_projectors_filename:
        XAS_projectors = get_RIXS_projectors(XAS_projectors_filename)
        if rank == 0:
            print("XAS projectors")
            print(XAS_projectors)
    if RIXS_projectors_filename:
        RIXS_projectors = get_RIXS_projectors(RIXS_projectors_filename)
        if rank == 0:
            print("RIXS projectors")
            print(RIXS_projectors)

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
        print("#spin-orbitals:", n_spin_orbitals)

    # Hamiltonian
    if rank == 0:
        print("Construct the Hamiltonian operator...")
    hOp = get_hamiltonian_operator_new(
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
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0:
        print("{:d} processes in the Hamiltonian.".format(len(hOp)))
    # Many body basis for the ground state
    if rank == 0:
        print("Create basis...")
    tau = k_B * T
    basis = CIPSI_Basis(
        H=hOp,
        impurity_orbitals=impurity_orbitals,
        bath_states=(valence_baths, conduction_baths),
        nominal_impurity_occ=n0imps,
        truncation_threshold=1e9,
        tau=tau,
        comm=comm,
    )
    basis.expand(hOp, de2_min=1e-4)
    restrictions = basis.restrictions

    if restrictions is not None and verbosity >= 2:
        print("Restrictions GS on occupation")
        for indices, limits in restrictions.items():
            print(f"---> {sorted(indices)} : {limits}", flush=True)

    energy_cut = -tau * np.log(1e-4)

    block_roots, block_basis, _ = basis.split_into_block_basis_and_redistribute_psi(hOp, None)
    h_gs = block_basis.build_sparse_matrix(hOp)
    block_es, block_psis_dense = finite.eigensystem_new(
        h_gs,
        e_max=energy_cut,
        k=2 * sum(len(imp_orbs) for imp_orbs in impurity_orbitals.values()), 
        eigenValueTol=np.sqrt(np.finfo(float).eps),
        comm=block_basis.comm,
        dense=block_basis.size < dense_cutoff,
    )
    psis = []
    es = np.array([], dtype=float)
    proc_cutoff = np.array(block_roots[1:] + [basis.comm.size])
    block_color = np.argmax(basis.comm.rank < proc_cutoff)
    for c, c_root in enumerate(block_roots):
        es_c =basis.comm.bcast(block_es, root=c_root) 
        es = np.append(es, es_c)
        if c != block_color:
            psi_c = basis.redistribute_psis([ManyBodyState() for _ in es_c])
        else:
            psi_c = basis.redistribute_psis(block_basis.build_state(block_psis_dense.T))
        psis.extend(psi_c)


    sort_idx = np.argsort(es)
    es = es[sort_idx]
    mask = es <= (es[0] + energy_cut)
    es = es[mask]
    psis = [psis[idx] for idx in itertools.compress(sort_idx, mask)]
    effective_restrictions = basis.get_effective_restrictions()
    if verbosity >= 1:
        print("Effective GS restrictions:")
        for indices, occupations in effective_restrictions.items():
            print(f"---> {sorted(indices)} : {occupations}")
        print("=" * 80)
    if verbosity >= 1:
        print(f"{len(basis)} Slater determinants in the basis.")
    gs_stats = basis.get_state_statistics(psis)
    rho_imps, rho_baths = basis.build_density_matrices(psis)
    n_orb = {i: sum(len(block) for block in basis.impurity_orbitals[i]) for i in basis.impurity_orbitals}
    full_rho_imps = {i: np.zeros((len(psis), n_orb[i], n_orb[i]), dtype=complex) for i in basis.impurity_orbitals}
    for i, i_blocks in basis.impurity_orbitals.items():
        orb_offset = min(orb for block in i_blocks for orb in block)
        for k in range(len(psis)):
            for j, block_orbs in enumerate(i_blocks):
                idx = np.ix_([k], [orb - orb_offset for orb in block_orbs], [orb - orb_offset for orb in block_orbs])
                full_rho_imps[i][idx] = rho_imps[i][j][k]
    thermal_rho_imps = {
        i: [finite.thermal_average_scale_indep(es, block_rhos, tau) for block_rhos in rho_imps[i]]
        for i in basis.impurity_orbitals.keys()
    }
    thermal_rho_baths = {
        i: [finite.thermal_average_scale_indep(es, block_rhos, tau) for block_rhos in rho_baths[i]]
        for i in basis.impurity_orbitals.keys()
    }
    if verbosity >= 1:
        print("Block structure")
        print_block_structure(block_structure)
        for i, blocks in basis.impurity_orbitals.items():
            print(f"Impurity orbital set {i}")
            subset_block_structuce = BlockStructure(
                blocks=blocks,
                identical_blocks=[[i] for i in range(len(blocks))],
                transposed_blocks=[[] for _ in range(len(blocks))],
                particle_hole_blocks=[[] for _ in range(len(blocks))],
                particle_hole_transposed_blocks=[[] for _ in range(len(blocks))],
                inequivalent_blocks=[j for j in range(len(blocks))],
            )
            finite.printThermalExpValues_new(full_rho_imps[i], es, tau, rot_to_spherical[i], subset_block_structuce)
            finite.printExpValues(full_rho_imps[i], es, rot_to_spherical[i], subset_block_structuce)
        print("Occupation statistics for each eigenstate in the thermal ground state")
        print("Impurity, Valence, Conduction: Weight (|amp|^2)")
        for i, psi_stats in enumerate(gs_stats):
            print(f"{i}:")
            for imp_occ, val_occ, con_occ in sorted(psi_stats.keys()):
                print(f"{imp_occ:^8d},{val_occ:^8d},{con_occ:^11d}: {psi_stats[(imp_occ, val_occ, con_occ)]}")
            print("=" * 80)
            print()
        print("Ground state bath occupation statistics:")
        for i in basis.impurity_orbitals.keys():
            print(f"orbital set {i}:")
            for block_i, (imp_rho, bath_rho) in enumerate(zip(thermal_rho_imps[i], thermal_rho_baths[i])):
                print(f"Block {block_i} (impurity orbitals {basis.impurity_orbitals[i][block_i]})")
                matrix_print(imp_rho, "Impurity density matrix:")
                matrix_print(bath_rho, "Bath density matrix:")
                print("=" * 80)
            print("", flush=verbosity >= 2)

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
            restrictions=restrictions,
            epsilons=epsilons,
            epsilonsRIXSin=epsilonsRIXSin,
            epsilonsRIXSout=epsilonsRIXSout,
            deltaRIXS=deltaRIXS,
            deltaNIXS=deltaNIXS,
            n_spin_orbitals=n_spin_orbitals,
            hOp=hOp.to_dict(),
        )
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
        print()
        print(f"Consider {len(es):d} eigenstates for the spectra \n")
        print("Calculate Interacting Green's function...", flush=verbosity >= 2)
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
    )

    print("Script finished for rank:", rank)


def get_restrictions(l, n0imps, nBaths, nValBaths, dnTols, dnValBaths, dnConBaths):
    restrictions = {}
    # Restriction on impurity orbitals
    indices = frozenset(c2i(nBaths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))
    # restrictions[indices] = (n0imps[l] - 1, n0imps[l] + dnTols[l] + 1)
    restrictions[indices] = (n0imps[l] - dnTols[l], n0imps[l] + dnTols[l] + 1)
    # Restriction on valence bath orbitals
    indices = []
    for b in range(nValBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (nValBaths[l] - dnValBaths[l], nValBaths[l] + 1)
    # Restriction on conduction bath orbitals
    indices = []
    for b in range(nValBaths[l], nBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (0, dnConBaths[l] + 1)

    return restrictions


def get_noninteracting_hamiltonian_operator(nBaths, SOCs, hField, h0_filename, rank, verbose=True):
    # Divide up input parameters to more concrete variables
    xi_2p, xi_3d = SOCs
    hx, hy, hz = hField
    # Add SOC, in spherical harmonics basis.
    SOC2pOperator = finite.getSOCop(xi_2p, l=1)
    SOC3dOperator = finite.getSOCop(xi_3d, l=2)

    # Magnetic field
    hHfieldOperator = {}
    l = 2
    for m in range(-l, l + 1):
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] = hx * 1 / 2.0
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] = hx * 1 / 2.0
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] += -hy * 1 / 2.0 * 1j
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] += hy * 1 / 2.0 * 1j
        for s in range(2):
            hHfieldOperator[(((l, s, m), "c"), ((l, s, m), "a"))] = hz * 1 / 2 if s == 1 else -hz * 1 / 2

    # Read the non-relativistic non-interacting Hamiltonian operator from file.
    h0_operator = read_h0_operator(h0_filename, nBaths)
    # h0_operator = read_h0_operator(h0_filename, nBaths)

    if rank == 0 and verbose:
        print("Non-interacting, non-relativistic Hamiltonian (h0):")
        print(h0_operator)
    hOperator = finite.addOps([hHfieldOperator, SOC2pOperator, SOC3dOperator, h0_operator])
    return hOperator


def read_h0_operator(filename, nBaths):
    _, ext = os.path.splitext(filename)
    if ext.lower() == ".pickle":
        return read_pickled_file(filename)
    if ext.lower() == ".dat":
        return read_h0_dict(filename)
    raise RuntimeError(f"Unknown file h0 file extension {ext}")


def gethHfieldop(hx, hy, hz, l=2):
    """
    Return magnetic field operator for one l-shell.

    Returns
    -------
    hHfieldOperator : dict
        Elements of the form:
        ((sorb1,'c'), (sorb2,'a') : h_value
        where sorb1 is a superindex of (l, s, m).

    """
    hHfieldOperator = {}
    for m in range(-l, l + 1):
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] = hx / 2
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] = hx / 2
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] += -hy * 1j / 2
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] += hy * 1j / 2
        for s in range(2):
            hHfieldOperator[(((l, s, m), "c"), ((l, s, m), "a"))] = hz / 2 if s == 1 else -hz / 2
    return hHfieldOperator


def get_hamiltonian_operator(nBaths, nValBaths, slaterCondon, SOCs, DCinfo, hField, h0_filename):
    """
    Return the Hamiltonian, in operator form.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
        Number of valence bath states for each angular momentum.
    slaterCondon : list
        List of Slater-Condon parameters.
    SOCs : list
        List of SOC parameters.
    DCinfo : list
        Contains information needed for the double counting energy.
    hField : list
        External magnetic field.
        Elements hx,hy,hz
    h0_filename : str
        Filename of non-interacting, non-relativistic operator.

    Returns
    -------
    hOp : dict
        The Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form: (i,'c') or (i,'a'),
        where i is a spin-orbital index.

    """
    # Divide up input parameters to more concrete variables
    Fdd, Fpp, Fpd, Gpd = slaterCondon
    xi_2p, xi_3d = SOCs
    n0imps, chargeTransferCorrection = DCinfo
    hx, hy, hz = hField

    # Calculate the U operator, in spherical harmonics basis.
    uOperator = finite.get2p3dSlaterCondonUop(Fdd=Fdd, Fpp=Fpp, Fpd=Fpd, Gpd=Gpd)
    # Add SOC, in spherical harmonics basis.
    SOC2pOperator = finite.getSOCop(xi_2p, l=1)
    SOC3dOperator = finite.getSOCop(xi_3d, l=2)

    # Double counting (DC) correction values.
    # MLFT DC
    dc = finite.dc_MLFT(
        n3d_i=n0imps[2],
        c=chargeTransferCorrection,
        Fdd=Fdd,
        n2p_i=n0imps[1],
        Fpd=Fpd,
        Gpd=Gpd,
    )
    eDCOperator = {}
    for il, l in enumerate([2, 1]):
        for s in range(2):
            for m in range(-l, l + 1):
                eDCOperator[(((l, s, m), "c"), ((l, s, m), "a"))] = -dc[l]

    # Magnetic field
    hHfieldOperator = finite.gethHfieldop(hx, hy, hz, l=2)

    # Read the non-relativistic non-interacting Hamiltonian operator from file.
    h0_operator = read_pickled_file(h0_filename)

    # Add Hamiltonian terms to one operator.
    hOperator = finite.addOps(
        [
            uOperator,
            hHfieldOperator,
            SOC2pOperator,
            SOC3dOperator,
            eDCOperator,
            h0_operator,
        ]
    )
    if MPI.COMM_WORLD.rank == 0:
        finite.printOp(nBaths, hOperator, "Local Hamiltonian")
    # Convert spin-orbital and bath state indices to a single index notation.
    hOp = {}
    for process, value in hOperator.items():
        hOp[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value

    assert_hermitian(hOp)

    return hOp


def get_hamiltonian_operator_new(
    nBaths, nValBaths, slaterCondon, SOCs, DCinfo, hField, h0_filename, rank, verbose=True
):
    """
    Return the Hamiltonian, in operator form.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
        Number of valence bath states for each angular momentum.
    slaterCondon : list
        List of Slater-Condon parameters.
    SOCs : list
        List of SOC parameters.
    DCinfo : list
        Contains information needed for the double counting energy.
    hField : list
        External magnetic field.
        Elements hx,hy,hz
    h0_filename : str
        Filename of non-interacting, non-relativistic operator.

    Returns
    -------
    hOp : dict
        The Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form: (i,'c') or (i,'a'),
        where i is a spin-orbital index.

    """
    # Divide up input parameters to more concrete variables
    Fdd, Fpp, Fpd, Gpd = slaterCondon
    n0imps, chargeTransferCorrection = DCinfo

    h_non_interacting = get_noninteracting_hamiltonian_operator(nBaths, SOCs, hField, h0_filename, rank, verbose)
    # Calculate the U operator, in spherical harmonics basis.
    uOperator = finite.get2p3dSlaterCondonUop(Fdd=Fdd, Fpp=Fpp, Fpd=Fpd, Gpd=Gpd)
    dc = finite.dc_MLFT(n3d_i=n0imps[2], c=chargeTransferCorrection, Fdd=Fdd, n2p_i=n0imps[1], Fpd=Fpd, Gpd=Gpd)
    eDCOperator = {}
    for l in [2, 1]:
        # for il, l in enumerate([2, 1]):
        for s in range(2):
            for m in range(-l, l + 1):
                eDCOperator[(((l, s, m), "c"), ((l, s, m), "a"))] = -dc[l]

    # Add Hamiltonian terms to one operator.
    hOperator = finite.addOps([uOperator, eDCOperator, h_non_interacting])
    if rank == 0 and verbose:
        finite.printOp(nBaths, hOperator, "Local Hamiltonian: ")

    # Convert spin-orbital and bath state indices to a single index notation.
    hOp = {}
    for process, value in hOperator.items():
        hOp[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value

    assert_hermitian(hOp)
    return hOp


def read_pickled_file(filename: str):
    with open(filename, "rb") as handle:
        content = pickle.load(handle)
    return content


def read_h0_dict(h0_filename):
    r"""
    Reads the non-interacting Hamiltoninan from file.
    Parameters
    ----------
        h0_filename : String
        File containing the non-interacting Hamiltonian.
    """
    h0_dict = {}
    for _, op in op_parser.parse_file(h0_filename).items():
        for key, val in op.items():
            if key in h0_dict:
                h0_dict[key] += val
            else:
                h0_dict[key] = val
    return h0_dict


def read_RIXS_projectors(filename):
    r"""
    Reads projectors for the RIXS calculations from file.
    Parameters
    ----------
        filename : String
        File containing the projectors. Projectors are separated by new lines.
    """
    return op_parser.parse_file(filename)


def read_key_val(line):
    r"""
    Read key and value pair from a string.
    Returns a tuple, (key, value).
    Parameters
    ----------
        line : String
        String of the form "key:value"
    """
    parts = line.split(":")
    if len(parts) != 2:
        print("Error reading key, value pair from file!")
        print(line)
        return {}
    val = complex(parts[1])
    keys = "".join(parts[0].split())
    key = read_tuple(keys)
    return (key, val)


def read_tuple(line):
    r"""
    Read arbitratily nested tuples from string.
    Returns the tuple contained in string.
    """
    store = []
    tmp_tup = []
    line = line.replace("(", "(,")
    line = line.replace(")", ",)")
    tmp = line.split(",")
    for cs in tmp:
        cs = cs.strip()
        if cs == "(":
            store.append(tmp_tup)
            tmp_tup = []
        elif cs == ")":
            t = tuple(tmp_tup)
            tmp_tup = []
            if store:
                s = store.pop()
                if s:
                    tmp_tup = [s[0]]
            tmp_tup.append(t)
        elif cs not in ["("]:
            if cs in ["a", "c"]:
                tmp_tup.append(cs)
            else:
                tmp_tup.append(int(cs))
    return tuple(tmp_tup[0])


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
        help=("Max devation from initial impurity occupation, " "for each angular momentum."),
    )
    parser.add_argument(
        "--dnValBaths",
        type=int,
        nargs="+",
        default=[0, 2],
        help=("Max number of electrons to leave valence bath orbitals, " "for each angular momentum."),
    )
    parser.add_argument(
        "--dnConBaths",
        type=int,
        nargs="+",
        default=[0, 0],
        help=("Max number of electrons to enter conduction bath orbitals, " "for each angular momentum."),
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
        help=("Smearing, half width half maximum (HWHM). " "Due to short core-hole lifetime."),
    )
    parser.add_argument(
        "--deltaRIXS",
        type=float,
        default=0.050,
        help=("Smearing, half width half maximum (HWHM). " "Due to finite lifetime of excited states."),
    )
    parser.add_argument(
        "--deltaNIXS",
        type=float,
        default=0.100,
        help=("Smearing, half width half maximum (HWHM). " "Due to finite lifetime of excited states."),
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
