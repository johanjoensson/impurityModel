"""
Script for calculating various spectra.

"""

import argparse
import json
import time
from collections import OrderedDict

import h5py
import numpy as np
from mpi4py import MPI

# Local stuff
from impurityModel.ed import finite, spectra
from impurityModel.ed.average import k_B, thermal_average
from impurityModel.ed.finite import assert_hermitian, c2i
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
    h0_CF_filename,
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
):
    """
    First find the lowest eigenstates and then use them to calculate various spectra.

    Parameters
    ----------
    h0_CF_filename : str
        Filename of the non-relativistic non-interacting CF Hamiltonian operator, in json-format.
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
        t0 = time.time()

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
    w = np.linspace(-25, 25, 3000)
    # Each element is a XAS polarization vector.
    epsilons = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # [[0,0,1]]
    # RIXS parameters
    # Polarization vectors, of in and outgoing photon.
    epsilonsRIXSin = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # [[0,0,1]]
    epsilonsRIXSout = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # [[0,0,1]]
    wIn = np.linspace(-10, 20, 50)
    wLoss = np.linspace(-2, 12, 4000)
    XAS_projectors = None
    RIXS_projectors = None
    # if XAS_projectors_filename:
    #     XAS_projectors = get_RIXS_projectors(XAS_projectors_filename)
    #     if rank == 0:
    #         print("XAS projectors")
    #     if rank == 0:
    #         print(XAS_projectors)
    # if RIXS_projectors_filename:
    #     RIXS_projectors = get_RIXS_projectors(RIXS_projectors_filename)
    #     if rank == 0:
    #         print("RIXS projectors")
    #     if rank == 0:
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
        print("#spin-orbitals:", n_spin_orbitals)

    # Hamiltonian
    if rank == 0:
        print("Construct the Hamiltonian operator...")
    hOp = get_hamiltonian_operator_using_CF(
        nBaths,
        nValBaths,
        [Fdd, Fpp, Fpd, Gpd],
        [xi_2p, xi_3d],
        [n0imps, chargeTransferCorrection],
        hField,
        h0_CF_filename,
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
        hOp,
        impurity_orbitals,
        (valence_baths, conduction_baths),
        nominal_impurity_occ=n0imps,
        truncation_threshold=1e9,
        tau=tau,
        comm=comm,
        verbose=verbosity >= 2,
    )
    if verbosity >= 2:
        print(f"Initial basis contains {len(basis)} determinants")
    basis.expand(hOp, de2_min=1e-3)
    restrictions = basis.restrictions

    if restrictions is not None and verbosity >= 2:
        print("Restrictions GS on occupation")
        for indices, limits in restrictions.items():
            print(f"---> {sorted(indices)} : {limits}", flush=True)

    energy_cut = -tau * np.log(1e-4)

    h_gs = basis.build_sparse_matrix(hOp)
    es, psis_dense = finite.eigensystem_new(
        h_gs,
        e_max=energy_cut,
        k=5,
        eigenValueTol=0,
        comm=basis.comm,
        dense=basis.size < dense_cutoff,
    )
    psis = basis.build_state(psis_dense.T)
    effective_restrictions = basis.get_effective_restrictions()
    if verbosity >= 1:
        print("Effective GS restrictions:")
        for indices, occupations in effective_restrictions.items():
            print(f"---> {sorted(indices)} : {occupations}")
        print("=" * 80)
        print()
        print(f"Consider {len(es):d} eigenstates for the spectra \n")
        print("Calculate Interacting Green's function...", flush=verbosity >= 2)
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
            h0_CF_filename=h0_CF_filename,
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
    else:
        h5f = None

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
        restrictions,
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


def get_hamiltonian_operator_using_CF(
    nBaths, nValBaths, slaterCondon, SOCs, DCinfo, hField, h0_CF_filename, bath_state_basis="spherical"
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
    h0_CF_filename : str
        Filename of the non-relativistic non-interacting CF Hamiltonian operator, in json-format.
    bath_state_basis : str
        'spherical' or 'cubic'.
        Which basis to use for the bath states.

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
    dc = finite.dc_MLFT(n3d_i=n0imps[2], c=chargeTransferCorrection, Fdd=Fdd, n2p_i=n0imps[1], Fpd=Fpd, Gpd=Gpd)
    eDCOperator = {}
    for il, l in enumerate([2, 1]):
        for s in range(2):
            for m in range(-l, l + 1):
                eDCOperator[(((l, s, m), "c"), ((l, s, m), "a"))] = -dc[l]

    # Magnetic field
    hHfieldOperator = {}
    l = 2
    for m in range(-l, l + 1):
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] = hx * 1 / 2.0
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] = hx * 1 / 2.0
        hHfieldOperator[(((l, 1, m), "c"), ((l, 0, m), "a"))] = -hy * 1 / 2.0 * 1j
        hHfieldOperator[(((l, 0, m), "c"), ((l, 1, m), "a"))] = hy * 1 / 2.0 * 1j
        for s in range(2):
            hHfieldOperator[(((l, s, m), "c"), ((l, s, m), "a"))] = hz * 1 / 2 if s == 1 else -hz * 1 / 2

    # Construct non-relativistic and non-interacting Hamiltonian, from CF parameters.
    h0_operator = get_CF_hamiltonian(nBaths, nValBaths, h0_CF_filename, bath_state_basis)

    # Add Hamiltonian terms to one operator.
    hOperator = finite.addOps([uOperator, hHfieldOperator, SOC2pOperator, SOC3dOperator, eDCOperator, h0_operator])
    # Convert spin-orbital and bath state indices to a single index notation.
    hOp = {}
    for process, value in hOperator.items():
        hOp[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value

    assert_hermitian(hOp)
    return hOp


def get_CF_hamiltonian(nBaths, nValBaths, h0_CF_filename, bath_state_basis="spherical"):
    """
    Construct non-relativistic and non-interacting Hamiltonian, from CF parameters.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
        Number of valence bath states for each angular momentum.
    h0_CF_filename : str
        Filename of the non-relativistic non-interacting CF Hamiltonian operator, in json-format.
    bath_state_basis : str
        'spherical' or 'cubic'.
        Which basis to use for the bath states.

    Returns
    -------
    h0_operator : dict
        The non-relativistic non-interacting Hamiltonian in operator form.
        Hamiltonian describes 3d orbitals and bath orbitals.
        tuple : complex,
        where each tuple describes a process of two steps (annihilation and then creation).
        Each step is described by a tuple of the form:
        (spin_orb, 'c') or (spin_orb, 'a'),
        where spin_orb is a tuple of the form (l, s, m) or (l, b) or ((l_a, l_b), b).

    """
    (
        e_imp,
        e_deltaO_imp,
        e_val_eg,
        e_val_t2g,
        e_con_eg,
        e_con_t2g,
        v_val_eg,
        v_val_t2g,
        v_con_eg,
        v_con_t2g,
    ) = read_h0_CF_file(h0_CF_filename)

    # Calculate impurity 3d Hamiltonian.
    # First formulate in cubic harmonics basis and then rotate to
    # the spherical harmonics basis.
    l = 2
    e_imp_eg = e_imp + 3 / 5 * e_deltaO_imp
    e_imp_t2g = e_imp - 2 / 5 * e_deltaO_imp
    h_imp_3d = np.zeros((2 * l + 1, 2 * l + 1))
    np.fill_diagonal(h_imp_3d, (e_imp_eg, e_imp_eg, e_imp_t2g, e_imp_t2g, e_imp_t2g))
    # Convert to spherical harmonics basis
    u = finite.get_spherical_2_cubic_matrix(spinpol=False, l=l)
    h_imp_3d = np.dot(u, np.dot(h_imp_3d, np.conj(u.T)))
    # Convert from matrix to operator form.
    # Also add spin.
    h_imp_3d_operator = {}
    for i, mi in enumerate(range(-l, l + 1)):
        for j, mj in enumerate(range(-l, l + 1)):
            if h_imp_3d[i, j] != 0:
                for s in range(2):
                    h_imp_3d_operator[(((l, s, mi), "c"), ((l, s, mj), "a"))] = h_imp_3d[i, j]

    # Bath (3d) on-site energies and hoppings.
    # Calculate hopping terms between bath and impurity.
    # First formulate the terms in the cubic harmonics basis.
    l = 2
    vVal3d = np.zeros((2 * l + 1, 2 * l + 1))
    vCon3d = np.zeros((2 * l + 1, 2 * l + 1))
    eBathVal3d = np.zeros((2 * l + 1, 2 * l + 1))
    eBathCon3d = np.zeros((2 * l + 1, 2 * l + 1))
    np.fill_diagonal(vVal3d, (v_val_eg, v_val_eg, v_val_t2g, v_val_t2g, v_val_t2g))
    np.fill_diagonal(vCon3d, (v_con_eg, v_con_eg, v_con_t2g, v_con_t2g, v_con_t2g))
    np.fill_diagonal(eBathVal3d, (e_val_eg, e_val_eg, e_val_t2g, e_val_t2g, e_val_t2g))
    np.fill_diagonal(eBathCon3d, (e_con_eg, e_con_eg, e_con_t2g, e_con_t2g, e_con_t2g))
    # For the bath states, we can rotate to any basis.
    # Which bath state basis to use is determined selected here.
    if bath_state_basis == "spherical":
        # One example is to use spherical harmonics basis for the bath states.
        # This implies the following rotation matrix:
        u_bath = u
    elif bath_state_basis == "cubic":
        # One example is to keep the cubic harmonics basis for the bath states.
        # This implies the following rotation matrix:
        u_bath = np.eye(np.shape(u)[0])
    else:
        raise Exception("Design of this basis is not (yet) implemented.")
    # Rotate the bath energies and the hopping parameters
    vVal3d = np.dot(u_bath, np.dot(vVal3d, np.conj(u.T)))
    vCon3d = np.dot(u_bath, np.dot(vCon3d, np.conj(u.T)))
    eBathVal3d = np.dot(u_bath, np.dot(eBathVal3d, np.conj(u_bath.T)))
    eBathCon3d = np.dot(u_bath, np.dot(eBathCon3d, np.conj(u_bath.T)))
    # Convert from matrix to operator form.
    # Also introduce spin.
    h_hopp_operator = {}
    e_bath_3d_operator = {}
    # Loop over spin
    for s in range(2):
        # Loop over impurity orbitals
        for i, _mi in enumerate(range(-l, l + 1)):
            # Bath state index for valence bath states.
            bi_val = s * (2 * l + 1) + i
            # Bath state index for conduction bath states.
            bi_con = 2 * (2 * l + 1) + bi_val
            # Loop over impurity orbitals
            for j, mj in enumerate(range(-l, l + 1)):
                # Bath state index for valence bath states.
                bj_val = s * (2 * l + 1) + j
                # Bath state index for conduction bath states.
                bj_con = 2 * (2 * l + 1) + bj_val
                # Hamiltonian values related to valence bath states.
                vHopp = vVal3d[i, j]
                eBath = eBathVal3d[i, j]
                if vHopp != 0:
                    h_hopp_operator[(((l, bi_val), "c"), ((l, s, mj), "a"))] = vHopp
                    h_hopp_operator[(((l, s, mj), "c"), ((l, bi_val), "a"))] = vHopp.conjugate()
                if eBath != 0:
                    e_bath_3d_operator[(((l, bi_val), "c"), ((l, bj_val), "a"))] = eBath
                # Only add the processes related to the conduction bath states if they are
                # in the basis.
                if nBaths[l] - nValBaths[l] == 10:
                    # Hamiltonian values related to conduction bath states.
                    vHopp = vCon3d[i, j]
                    eBath = eBathCon3d[i, j]
                    if vHopp != 0:
                        h_hopp_operator[(((l, bi_con), "c"), ((l, s, mj), "a"))] = vHopp
                        h_hopp_operator[(((l, s, mj), "c"), ((l, bi_con), "a"))] = vHopp.conjugate()
                    if eBath != 0:
                        e_bath_3d_operator[(((l, bi_con), "c"), ((l, bj_con), "a"))] = eBath

    # Add Hamiltonian terms to one operator.
    h0_operator = finite.addOps([h_imp_3d_operator, h_hopp_operator, e_bath_3d_operator])
    return h0_operator


def read_h0_CF_file(h0_CF_filename):
    """
    Reads CF Hamiltonian from json-file.

    Parameters
    ----------
    h0_CF_filename : str
        Filename of the non-relativistic non-interacting CF Hamiltonian operator, in json-format.

    Returns
    -------
    e_imp : float
        Average 3d onsite energy.
    e_deltaO_imp : float
        Energy split of 3d orbitals into eg and t2g orbitals.
    e_val_eg : float
        Energy position of valence bath states, coupled to eg orbitals.
    e_val_t2g : float
        Energy position of valence bath states, coupled to t2g orbitals.
    e_con_eg : float
        Energy position of conduction bath states, coupled to eg orbitals.
    e_con_t2g : float
        Energy position of conduction bath states, coupled to t2g orbitals.
    v_val_eg : float
        Hybridization/hopping strength of valence bath states with eg orbitals.
    v_val_t2g : float
        Hybridization/hopping strength of valence bath states with t2g orbitals.
    v_con_eg : float
        Hybridization/hopping strength of conduction bath states with eg orbitals.
    v_con_t2g : float
        Hybridization/hopping strength of conduction bath states with t2g orbitals.

    Note
    ----
    If a parameter is not specified in the json-file, a default value will used.

    """
    with open(h0_CF_filename, "r") as file_handle:
        parameters = json.loads(file_handle.read())
    # Default values are for Ni in NiO.
    e_imp = parameters.get("e_imp", -1.31796)
    e_deltaO_imp = parameters.get("e_deltaO_imp", 0.60422)
    e_val_eg = parameters.get("e_val_eg", -4.4)
    e_val_t2g = parameters.get("e_val_t2g", -6.5)
    e_con_eg = parameters.get("e_con_eg", 3)
    e_con_t2g = parameters.get("e_con_t2g", 2)
    v_val_eg = parameters.get("v_val_eg", 1.883)
    v_val_t2g = parameters.get("v_val_t2g", 1.395)
    v_con_eg = parameters.get("v_con_eg", 0.6)
    v_con_t2g = parameters.get("v_con_t2g", 0.4)
    return (
        e_imp,
        e_deltaO_imp,
        e_val_eg,
        e_val_t2g,
        e_con_eg,
        e_con_t2g,
        v_val_eg,
        v_val_t2g,
        v_con_eg,
        v_con_t2g,
    )


if __name__ == "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser(description="Spectroscopy simulations using crystal-field Hamiltonian")
    parser.add_argument(
        "h0_CF_filename",
        type=str,
        help="Filename of non-interacting crystal-field Hamiltonian, in json-format.",
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

    assert args.ls[0] == 1
    assert args.ls[1] == 2
    assert args.nBaths[1] in (10, 20)
    assert args.nValBaths[1] == 10

    main(
        h0_CF_filename=args.h0_CF_filename,
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
    )
