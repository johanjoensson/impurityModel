"""
This module contains functions for calculating various spectra.
"""

import time
from math import sqrt

import numpy as np
import scipy.integrate as si
from mpi4py import MPI
from scipy.special import sph_harm_y, spherical_jn


def sph_harm(m, n, theta, phi):
    """
    Compute the spherical harmonics.

    This function wraps scipy's `sph_harm_y` to compute the spherical harmonic of
    degree `n` and order `m` at polar angle `theta` and azimuthal angle `phi`.

    Parameters
    ----------
    m : int
        Order of the harmonic (often denoted `m`).
    n : int
        Degree of the harmonic (often denoted `l` or `n`).
    theta : float
        Polar (colatitudinal) coordinate in radians.
    phi : float
        Azimuthal (longitudinal) coordinate in radians.

    Returns
    -------
    complex
        The value of the spherical harmonic.
    """
    return sph_harm_y(n, m, phi, theta)


# Local imports
import impurityModel.ed.greens_function as gf
from impurityModel.ed.finite import (
    arrayOp2Dict,
    c2i,
    combineOp,
    daggerOp,
    gauntC,
)
from impurityModel.ed.lanczos import Reort
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
from impurityModel.ed.mpi_comm import gather_distributed_results

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


def simulate_spectra(
    es,
    psis,
    hOp,
    tau,
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
    occ_cutoff,
    dN,
    slaterWeightMin,
    verbose,
):
    """
    Simulate various spectra.

    Parameters
    ----------
    es : tuple
        Eigen-energy (in eV).
    psis : tuple
        Many-body eigen-states.
    hOp : dict
        The Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form: (i,'c') or (i,'a'),
        where i is a spin-orbital index.
    T : float
        Temperature (in Kelvin).
    w : ndarray
        Real-energy mesh (in eV).
    delta : float
        Distance above the real axis (in eV).
        Gives smearing to spectra.
    epsilons : list
        Each element is a XAS polarization vector.
    wLoss : ndarray
        Real-energy mesh (in eV).
        Incoming minus outgoing photon energy.
    deltaNIXS : float
        Distance above the real axis (in eV).
        Gives smearing to NIXS spectra.
    qsNIXS : list
        Various momenta used in NIXS.
    liNIXS : int
        Angular momentum of final orbitals in the NIXS excitation process.
    ljNIXS : int
        Angular momentum of initial orbitals in the NIXS excitation process.
    RiNIXS : ndarray
        Radial part of final correlated orbitals.
    RjNIXS : ndarray
        Radial part of initial correlated orbitals.
    radialMesh : ndarray
        Radial mesh, using in NIXS.
    wIn : ndarray
        Incoming photon energies in RIXS.
    deltaRIXS : float
        Distance above the real axis (in eV).
        Gives smearing to RIXS spectra.
    epsilonsRIXSin : list
        Polarization vectors of in-going photon.
    epsilonsRIXSout : list
        Polarization vectors of out-going photon.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    h5f : h5py file-handle
        Will be used to write data to disk.
    nBaths : OrderedDict
        Angular momentum : number of bath states.
    RIXS_projectors : dict
        dict of dicts representing the projections to apply for the calculation of the RIXS spectra

    """
    if rank == 0:
        t0 = time.perf_counter()

    if rank == 0:
        print("Create 3d inverse photoemission and photoemission spectra...")
    # Transition operators
    tOpsIPS = getInversePhotoEmissionOperators(nBaths, l=2)
    tOpsPS = getPhotoEmissionOperators(nBaths, l=2)
    if rank == 0:
        print("Inverse photoemission Green's function..")
    assert isinstance(hOp, ManyBodyOperator)
    gsIPS = getSpectra_new(
        hOp,
        [ManyBodyOperator(t) for t in tOpsIPS],
        psis,
        es,
        tau,
        w,
        basis,
        delta,
        slaterWeightMin,
        verbose,
        occ_cutoff,
        dN_imp={1: (0, 0), 2: (0, 1)},
        dN_val={1: (0, 0), 2: (1, 0)},
        dN_con={1: (0, 0), 2: (0, 1)},
    )
    if rank == 0:
        print("Photoemission Green's function..")
    gsPS = getSpectra_new(
        hOp,
        [ManyBodyOperator(t) for t in tOpsPS],
        psis,
        es,
        tau,
        -w,
        basis,
        -delta,
        slaterWeightMin,
        verbose,
        occ_cutoff,
        dN_imp={1: (0, 0), 2: (1, 0)},
        dN_val={1: (0, 0), 2: (1, 0)},
        dN_con={1: (0, 0), 2: (0, 1)},
    )
    gsPS *= -1
    gs = gsPS + gsIPS
    if rank == 0:
        # print("#eigenstates = {:d}".format(np.shape(gs)[0]))
        print("#spin orbitals = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[0]))
    # Thermal average
    # a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0 and h5f:
        # h5f.create_dataset("PS", data=-gs.imag)
        h5f.create_dataset("PSthermal", data=-gs.imag)
    # Sum over transition operators
    aSum = np.sum(-gs.imag, axis=1)
    # Save spectra to disk
    if rank == 0:
        tmp = [w, aSum]
        # Each transition operator seperatly
        for i in range(np.shape(gs)[1]):
            tmp.append(-gs.imag[:, i])
        print("Save spectra to disk...\n")
        np.savetxt("PS.dat", np.array(tmp).T, fmt="%8.4f", header="E  sum  T1  T2  T3 ...")
    if rank == 0:
        print("time(PS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    if rank == 0:
        print("Create core 2p x-ray photoemission spectra (XPS) ...")
    # Transition operators
    tOpsPS = getPhotoEmissionOperators(nBaths, l=1)
    # Photoemission Green's function
    gs = getSpectra_new(
        hOp,
        [ManyBodyOperator(t) for t in tOpsPS],
        psis,
        es,
        tau,
        -w,
        basis,
        -delta,
        slaterWeightMin,
        verbose,
        occ_cutoff,
        dN_imp={1: (1, 0), 2: (1, 1)},
        dN_val={1: (0, 0), 2: (1, 0)},
        dN_con={1: (0, 0), 2: (0, 1)},
    )
    gs *= -1
    if rank == 0:
        # print("#eigenstates = {:d}".format(np.shape(gs)[0]))
        print("#spin orbitals = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[0]))
    # Thermal average
    # a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0 and h5f:
        # h5f.create_dataset("XPS", data=-gs.imag)
        h5f.create_dataset("XPSthermal", data=-gs.imag)
    # Sum over transition operators
    aSum = np.sum(-gs.imag, axis=1)
    # Save spectra to disk
    if rank == 0:
        tmp = [w, aSum]
        # Each transition operator seperatly
        for i in range(np.shape(gs)[1]):
            tmp.append(-gs.imag[:, i])
        print("Save spectra to disk...\n")
        np.savetxt("XPS.dat", np.array(tmp).T, fmt="%8.4f", header="E  sum  T1  T2  T3 ...")
    if rank == 0:
        print("time(XPS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    if rank == 0:
        print("Create NIXS spectra...")
    # Transition operator: exp(iq*r)
    tOps = getNIXSOperators(nBaths, qsNIXS, liNIXS, ljNIXS, RiNIXS, RjNIXS, radialMesh)
    # Green's function
    gs = getSpectra_new(
        hOp,
        [ManyBodyOperator(t) for t in tOps],
        psis,
        es,
        tau,
        wLoss,
        basis,
        deltaNIXS,
        slaterWeightMin,
        verbose,
        occ_cutoff,
        dN_imp={liNIXS: (1, 1), ljNIXS: (1, 1)},
        dN_val={liNIXS: (1, 0), ljNIXS: (1, 0)},
        dN_con={liNIXS: (0, 1), ljNIXS: (0, 1)},
    )
    # gs = getSpectra(n_spin_orbitals, hOp, tOps, psis, es, wLoss, deltaNIXS, restrictions)
    if rank == 0:
        # print("#eigenstates = {:d}".format(np.shape(gs)[0]))
        print("#q-points = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[0]))
    # Thermal average
    # a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0 and h5f:
        # h5f.create_dataset("NIXS", data=-gs.imag)
        h5f.create_dataset("NIXSthermal", data=-gs.imag)
    # Sum over q-points
    aSum = np.sum(-gs.imag, axis=1)
    # Save spectra to disk
    if rank == 0:
        tmp = [wLoss, aSum]
        # Each q-point seperatly
        for i in range(np.shape(gs)[1]):
            tmp.append(-gs.imag[:, i])
        print("Save spectra to disk...\n")
        np.savetxt("NIXS.dat", np.array(tmp).T, fmt="%8.4f", header="E  sum  T1  T2  T3 ...")

    if rank == 0:
        print("time(NIXS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    if rank == 0:
        print("Create XAS spectra...")
    # Dipole transition operators
    tOps = getDipoleOperators(nBaths, epsilons)
    if XAS_projectors:
        iBasisProjectors = arrayOp2Dict(nBaths, XAS_projectors.values())
        projectedTOps = []
        for proj in iBasisProjectors:
            for op in tOps:
                projectedTOps.append(combineOp(nBaths, proj, op))
        tOps = projectedTOps

    # Green's function
    gs = getSpectra_new(
        hOp,
        [ManyBodyOperator(t) for t in tOps],
        psis,
        es,
        tau,
        w,
        basis,
        delta,
        slaterWeightMin,
        verbose,
        occ_cutoff,
        dN_imp={1: (1, 0), 2: (0, 1)},
        dN_val={1: (0, 0), 2: (1, 0)},
        dN_con={1: (0, 0), 2: (0, 1)},
    )
    # gs = getSpectra(n_spin_orbitals, hOp, tOps, psis, es, w, delta, restrictions)
    if rank == 0:
        # print("#eigenstates = {:d}".format(np.shape(gs)[0]))
        print("#polarizations = {:d}".format(np.shape(gs)[1]))
        print("#mesh points = {:d}".format(np.shape(gs)[0]))
    # Thermal average
    # a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
    if rank == 0 and h5f:
        # h5f.create_dataset("XAS", data=-gs.imag)
        h5f.create_dataset("XASthermal", data=-gs.imag)
    # Sum over transition operators
    aSum = np.sum(-gs.imag, axis=1)
    # Save spectra to disk
    if rank == 0:
        tmp = [w, aSum]
        # Each transition operator seperatly
        for i in range(np.shape(gs)[1]):
            tmp.append(-gs.imag[:, i])
        print("Save spectra to disk...\n")
        np.savetxt("XAS.dat", np.array(tmp).T, fmt="%8.4f", header="E  sum  T1  T2  T3 ...")
    if rank == 0:
        print("time(XAS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
        t0 = time.perf_counter()

    if len(wIn) > 0:
        if rank == 0:
            print("Create RIXS spectra...")
        # Dipole 2p -> 3d transition operators
        tOpsIn = getDipoleOperators(nBaths, epsilonsRIXSin)
        # Dipole 3d -> 2p transition operators
        tOpsOut = getDaggeredDipoleOperators(nBaths, epsilonsRIXSout)

        if RIXS_projectors:
            iBasisProjectors = arrayOp2Dict(nBaths, RIXS_projectors.values())
            projectedTOpsIn = []
            projectedTOpsOut = []
            for proj in iBasisProjectors:
                for opIn in tOpsIn:
                    projectedTOpsIn.append(combineOp(nBaths, proj, opIn))
                for opOut in tOpsOut:
                    projectedTOpsOut.append(combineOp(nBaths, opOut, proj))
            tOpsIn = projectedTOpsIn
            tOpsOut = projectedTOpsOut

        gs = getRIXSmap_new(
            hOp,
            [ManyBodyOperator(t) for t in tOpsIn],
            [ManyBodyOperator(t) for t in tOpsOut],
            psis,
            es,
            tau,
            wIn,
            wLoss,
            delta,
            deltaRIXS,
            basis,
            verbose,
            slaterWeightMin=slaterWeightMin,
        )

        if rank == 0:
            # print("#eigenstates = {:d}".format(np.shape(gs)[0]))
            if RIXS_projectors:
                print("RIXS projectors = {}".format(RIXS_projectors.keys()))
            print(f"shape(gs) = {np.shape(gs)}")
            print("#in-polarizations = {:d}".format(np.shape(gs)[0]))
            print("#out-polarizations = {:d}".format(np.shape(gs)[1]))
            print("#mesh points of input energy = {:d}".format(np.shape(gs)[2]))
            print("#mesh points of energy loss = {:d}".format(np.shape(gs)[3]))
        # Thermal average
        # a = thermal_average(es[: np.shape(gs)[0]], -gs.imag, T=T)
        if rank == 0 and h5f:
            # h5f.create_dataset("RIXS", data=-gs.imag)
            h5f.create_dataset("RIXSthermal", data=-gs.imag)
            if RIXS_projectors:
                g = h5f.create_group("RIXSprojectors")
                for key, proj in RIXS_projectors:
                    g.create_dataset(key, data=str(proj))
        # Sum over transition operators
        aSum = np.sum(-gs.imag, axis=(0, 1))
        # Save spectra to disk
        if rank == 0:
            print("Save spectra to disk...\n")
            # I[wLoss,wIn], with wLoss on first column and wIn on first row.
            tmp = np.empty((len(wLoss) + 1, len(wIn) + 1), dtype=np.float32)
            tmp[0, 0] = len(wIn)
            tmp[0, 1:] = wIn
            tmp[1:, 0] = wLoss
            tmp[1:, 1:] = aSum.T
            tmp.tofile("RIXS.bin")
        if rank == 0:
            print("time(RIXS) = {:.2f} seconds \n".format(time.perf_counter() - t0))
            t0 = time.perf_counter()

    if rank == 0 and h5f:
        h5f.close()


def getDipoleOperators(nBaths, ns):
    r"""
    Return dipole transition operators.

    Transitions between states of different angular momentum,
    defined by the keys in the nBaths dictionary.

    Parameters
    ----------
    nBaths : dict
        angular momentum: number of bath states.
    ns : list
        Each element contains a polarization vector n = [nx,ny,nz]

    """
    tOps = []
    for n in ns:
        tOps.append(getDipoleOperator(nBaths, n))
    return tOps


def getDaggeredDipoleOperators(nBaths, ns):
    """
    Return daggered dipole transition operators.

    Parameters
    ----------
    nBaths : dict
        angular momentum: number of bath states.
    ns : list
        Each element contains a polarization vector n = [nx,ny,nz]

    """
    tDaggerOps = []
    for n in ns:
        tDaggerOps.append(daggerOp(getDipoleOperator(nBaths, n)))
    return tDaggerOps


def getDipoleOperator(nBaths, n):
    r"""
    Return dipole transition operator :math:`\hat{T}`.

    Transition between states of different angular momentum,
    defined by the keys in the nBaths dictionary.

    Parameters
    ----------
    nBaths : Ordered dict
        int : int,
        where the keys are angular momenta and values are number of bath states.
    n : list
        polarization vector n = [nx,ny,nz]

    """
    tOp = {}
    nDict = {-1: (n[0] + 1j * n[1]) / sqrt(2), 0: n[2], 1: (-n[0] + 1j * n[1]) / sqrt(2)}
    # Angular momentum
    l1, l2 = nBaths.keys()
    for m in range(-l2, l2 + 1):
        for mp in range(-l1, l1 + 1):
            for s in range(2):
                if abs(m - mp) <= 1:
                    # See Robert Eder's lecture notes:
                    # "Multiplets in Transition Metal Ions"
                    # in Julich school.
                    # tij = d*n*c1(l=2,m;l=1,mp),
                    # d - radial integral
                    # n - polarization vector
                    # c - Gaunt coefficient
                    tij = gauntC(k=1, l=l2, m=m, lp=l1, mp=mp, prec=16)
                    tij *= nDict[m - mp]
                    if tij != 0:
                        i = c2i(nBaths, (l2, s, m))
                        j = c2i(nBaths, (l1, s, mp))
                        tOp[((i, "c"), (j, "a"))] = tij
    return tOp


def getNIXSOperators(nBaths, qs, li, lj, Ri, Rj, r, kmin=1):
    r"""
    Return non-resonant inelastic x-ray scattering transition operators.

    :math:`\hat{T} = \sum_{i,j,\sigma} T_{i,j}
    \hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma}`,

    where
    :math:`T_{i,j} = \langle i | e^{i\mathbf{q}\cdot \mathbf{r}} | j \rangle`.
    The plane-wave is expanded in spherical harmonics.
    See PRL 99 257401 (2007) for more information.

    Parameters
    ----------
    nBaths : Ordered dict
        angular momentum: number of bath states.
    qs : list
        Each element contain a photon scattering vector q = [qx,qy,qz].
    li : int
        Angular momentum of the orbitals to excite into.
    lj : int
        Angular momentum of the orbitals to excite from.
    Ri : list
        Radial part of the orbital to excite into.
        Normalized such that the integral of Ri^2(r) * r^2
        should be equal to one.
    Rj : list
        Radial part of the orbital to excite from.
        Normalized such that the integral of Ri^2(r) * r^2
        should be equal to one.
    r : list
        Radial mesh points.
    kmin : int
        The lowest integer in the plane-wave expansion.
        By default kmin = 1, which means that the monopole contribution
        is not included.
        To include also the monopole scattering, set kmin = 0.

    """
    if rank == 0:
        if kmin == 0:
            print("Monopole contribution included in the expansion")
        elif kmin > 0:
            print("Monopole contribution not included in the expansion")
    tOps = []
    for q in qs:
        if rank == 0:
            print("q =", q)
        tOps.append(getNIXSOperator(nBaths, q, li, lj, Ri, Rj, r, kmin))
    return tOps


def getNIXSOperator(nBaths, q, li, lj, Ri, Rj, r, kmin=1):
    r"""
    Return non-resonant inelastic x-ray scattering transition
    operator :math:`\hat{T}`.

    :math:`\hat{T} = \sum_{i,j,\sigma} T_{i,j}
    \hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma}`,

    where
    :math:`T_{i,j} = \langle i | e^{i\mathbf{q}\cdot \mathbf{r}} | j \rangle`.
    The plane-wave is expanded in spherical harmonics.
    See PRL 99 257401 (2007) for more information.

    Parameters
    ----------
    nBaths : Ordered dict
        angular momentum: number of bath states.
    q : list
        Photon scattering vector q = [qx,qy,qz]
        The change in photon momentum.
    li : int
        Angular momentum of the orbitals to excite into.
    lj : int
        Angular momentum of the orbitals to excite from.
    Ri : list
        Radial part of the orbital to excite into.
        Normalized such that the integral of Ri^2(r) * r^2
        should be equal to one.
    Rj : list
        Radial part of the orbital to excite from.
        Normalized such that the integral of Ri^2(r) * r^2
        should be equal to one.
    r : list
        Radial mesh points.
    kmin : int
        The lowest integer in the plane-wave expansion.
        By default kmin = 1, which means that the monopole contribution
        is not included.
        To include also the monopole scattering, set kmin = 0.

    """
    # Convert scattering list to numpy array
    q = np.array(q)
    qNorm = np.linalg.norm(q)
    # Polar (colatitudinal) coordinate
    theta = np.arccos(q[2] / qNorm)
    # Azimuthal (longitudinal) coordinate
    phi = np.arccos(q[0] / (qNorm * np.sin(theta)))
    tOp = {}
    for k in range(kmin, abs(li + lj) + 1):
        if (li + lj + k) % 2 == 0:
            Rintegral = si.simpson(np.conj(Ri) * spherical_jn(k, qNorm * r) * Rj * r**2, r)
            if rank == 0:
                print("Rintegral(k=", k, ") =", Rintegral)
            for mi in range(-li, li + 1):
                for mj in range(-lj, lj + 1):
                    m = mi - mj
                    if abs(m) <= k:
                        tij = Rintegral
                        tij *= 1j ** (k) * sqrt(2 * k + 1)
                        tij *= np.conj(sph_harm(m, k, phi, theta))
                        tij *= gauntC(k, li, mi, lj, mj, prec=16)
                        if tij != 0:
                            for s in range(2):
                                i = c2i(nBaths, (li, s, mi))
                                j = c2i(nBaths, (lj, s, mj))
                                process = ((i, "c"), (j, "a"))
                                if process in tOp:
                                    tOp[((i, "c"), (j, "a"))] += tij
                                else:
                                    tOp[((i, "c"), (j, "a"))] = tij
    return tOp


def getInversePhotoEmissionOperators(nBaths, l=2):
    r"""
    Return inverse photo emission operators :math:`\{ c_i^\dagger \}`.

    Parameters
    ----------
    nBaths : OrderedDict
        Angular momentum: number of bath states.
    l : int
        Angular momentum.

    """
    # Transition operators
    tOpsIPS = []
    for s in range(2):
        for m in range(-l, l + 1):
            tOpsIPS.append({((c2i(nBaths, (l, s, m)), "c"),): 1})
    return tOpsIPS


def getPhotoEmissionOperators(nBaths, l=2):
    r"""
    Return photo emission operators :math:`\{ c_i \}`.

    Parameters
    ----------
    nBaths : OrderedDict
        Angular momentum: number of bath states.
    l : int
        Angular momentum.

    """
    # Transition operators
    tOpsPS = []
    for s in range(2):
        for m in range(-l, l + 1):
            tOpsPS.append({((c2i(nBaths, (l, s, m)), "a"),): 1})
    return tOpsPS


def getSpectra_new(
    hOp,
    tOps,
    psis,
    es,
    tau,
    w,
    basis,
    delta,
    slaterWeightMin,
    verbose,
    occ_cutoff,
    dN_imp,
    dN_val,
    dN_con,
):
    """
    Calculate the Green's function spectra for a list of transition operators.

    Supports both single-process and distributed parallel calculations over MPI.

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian operator.
    tOps : list of ManyBodyOperator
        List of transition operators.
    psis : list of ManyBodyState
        List of many-body eigenstates.
    es : list of float
        Total energies of the eigenstates.
    tau : float
        Temperature parameter for Boltzmann averaging.
    w : ndarray
        Real energy mesh points.
    basis : Basis
        The basis container.
    delta : float
        Broadening/resolution parameter (distance from the real axis).
    slaterWeightMin : float
        Minimum weight of Slater determinants to retain in basis expansion.
    verbose : bool
        If True, prints progress and diagnostic messages.
    occ_cutoff : float
        Occupation cutoff for state pruning.
    dN_imp : dict
        Restrictions on particle number change in the impurity shell.
    dN_val : dict
        Restrictions on particle number change in the valence shell.
    dN_con : dict
        Restrictions on particle number change in the conduction shell.

    Returns
    -------
    ndarray
        A 2D array of shape `(len(w), len(tOps))` containing the complex-valued
        spectra on the real axis. Only returned on root process rank 0; other ranks
        return an empty array.
    """
    comm = basis.comm
    if comm is None or comm.size <= 1:
        gs_realaxis = np.empty((len(w), len(tOps)), dtype=complex)
        for i, tOp in enumerate(tOps):
            alphas, betas, r = gf.calc_Greens_function_with_offdiag(
                hOp,
                [tOp],
                psis,
                es,
                basis,
                delta,
                occ_cutoff=occ_cutoff,
                slaterWeightMin=slaterWeightMin,
                verbose=verbose,
                sparse=True,
                dN_imp=dN_imp,
                dN_val=dN_val,
                dN_con=dN_con,
            )
            e0 = np.min(es)
            Z = np.sum(np.exp(-(es - e0) / tau))
            G_tOp = gf.calc_thermally_averaged_G(alphas, betas, r, w, es, e0, tau, delta)
            G_tOp /= Z
            gs_realaxis[:, i] = G_tOp[:, 0, 0]
        return gs_realaxis

    (
        tOps_indices,
        tOps_roots,
        color,
        tOps_per_color,
        tOp_basis,
        psis,
        _,
    ) = basis.split_basis_and_redistribute_psi([1] * len(tOps), psis)
    indices_for_colors = gather_distributed_results(
        basis.comm,
        tOp_basis.comm.rank if tOp_basis.comm is not None else 0,
        tOps_roots,
        tOps_per_color,
        np.array(tOps_indices),
        is_array=True,
    )

    gs_realaxis_local = np.empty((len(w), len(tOps_indices)), dtype=complex)
    for local_idx, tOp_idx in enumerate(tOps_indices):
        tOp = tOps[tOp_idx]
        assert isinstance(hOp, ManyBodyOperator)
        alphas, betas, r = gf.calc_Greens_function_with_offdiag(
            hOp,
            [tOp],
            psis,
            es,
            tOp_basis,
            delta,
            occ_cutoff=occ_cutoff,
            slaterWeightMin=slaterWeightMin,
            verbose=verbose,
            sparse=True,
            dN_imp=dN_imp,
            dN_val=dN_val,
            dN_con=dN_con,
        )
        if tOp_basis.comm.rank == 0:
            e0 = np.min(es)
            Z = np.sum(np.exp(-(es - e0) / tau))
            G_tOp = gf.calc_thermally_averaged_G(alphas, betas, r, w, es, e0, tau, delta)
            G_tOp /= Z
            gs_realaxis_local[:, local_idx] = G_tOp[:, 0, 0]

    gathered_gs = gather_distributed_results(
        basis.comm,
        tOp_basis.comm.rank if tOp_basis.comm is not None else 0,
        tOps_roots,
        tOps_per_color,
        np.swapaxes(gs_realaxis_local, 0, 1).copy(),
        is_array=True,
    )
    if basis.comm.rank == 0:
        gs_realaxis = np.empty((len(w), len(tOps)), dtype=complex)
        for i, tOps_idx in enumerate(indices_for_colors):
            gs_realaxis[:, tOps_idx] = gathered_gs[i]
    if tOp_basis is not None and tOp_basis.comm != basis.comm:
        tOp_basis.free_comm()
    return gs_realaxis if basis.comm.rank == 0 else np.empty((0, 0), dtype=complex)


def getRIXSmap_new(
    hOp,
    tOpsIn,
    tOpsOut,
    psis,
    Es,
    tau,
    wIns,
    wLoss,
    delta1,
    delta2,
    basis,
    verbose,
    slaterWeightMin,
):
    r"""
    Return RIXS Green's function for states.

    For states :math:`|psi \rangle`, calculate:

    :math:`g(w+1j*delta)
    = \langle psi| ROp^\dagger ((wLoss+1j*delta2+e)*\hat{1} - hOp)^{-1} ROp
    |psi \rangle`,

    where :math:`e = \langle psi| hOp |psi \rangle`, and

    :math:`Rop = tOpOut ((wIns+1j*delta1+e)*\hat{1} - hOp)^{-1} tOpIn`.

    Calculations are performed according to:

    1) Calculate state `|psi1> = tOpIn |psi>`.
    2) Calculate state `|psi2> = ((wIns+1j*delta1+e)*\hat{1} - hOp)^{-1}|psi1>`
        This is done by introducing operator:
        `A = (wIns+1j*delta1+e)*\hat{1} - hOp`.
        By applying A from the left on `|psi2> = A^{-1}|psi1>` gives
        the inverse problem: `A|psi2> = |psi1>`.
        This equation can be solved by guessing `|psi2>` and iteratively
        improving it.
    3) Calculate state `|psi3> = tOpOut |psi2>`
    4) Calculate `normalization = sqrt(<psi3|psi3>)`
    5) Normalize psi3 according to: `psi3 /= normalization`
    6) Now the Green's function is given by:
        :math:`g(wLoss+1j*delta2) = normalization^2
        * \langle psi3| ((wLoss+1j*delta2+e)*\hat{1} - hOp)^{-1} |psi3 \rangle`,
        which can efficiently be evaluation using Lanczos.

    Parameters
    ----------
    n_spin_orbitals : int
        Total number of spin-orbitals in the system.
    hOp : dict
        Operator
    tOpsIn : list
        List of dict operators, describing core-hole excitation.
    tOpsOut : list
        List of dict operators, describing filling of the core-hole.
    psis : list
        List of Multi state dictionaries
    es : list
        Total energies
    wIns : list
        Real axis energy mesh for incoming photon energy
    wLoss : list
        Real axis energy mesh for photon energy loss, i.e.
        wLoss = wIns - wOut
    delta1 : float
        Deviation from real axis.
        Broadening/resolution parameter.
    delta2 : float
        Deviation from real axis.
        Broadening/resolution parameter.
    restrictions : dict
        Restriction the occupation of generated
        product states.
    krylovSize : int
        Size of the Krylov space
    slaterWeightMin : float
        Restrict the number of product states by
        looking at `|amplitudes|^2`.
    h_dict_ground : dict
        Stores the result of the (Hamiltonian) operator hOp acting
        on individual product states. Information is stored according to:
        `|product state> : H|product state>`, where
        each product state is represented by an integer, and the result is
        a dictionary (of the format int : complex).
        Only product states without a core hole are stored in this variable.
        If present, it may also be updated by this function.
    parallelization_mode : str
        "serial", "H_build", "wIn" or "H_build_wIn"

    """
    excited_restrictions = basis.build_excited_restrictions(
        hOp,
        psis,
        Es,
        imp_change={1: (1, 0), 2: (1, 1)},
        val_change={1: (0, 0), 2: (1, 0)},
        con_change={1: (0, 0), 2: (0, 1)},
    )
    relaxed_restrictions = basis.build_excited_restrictions(
        hOp,
        psis,
        Es,
        imp_change={1: (0, 0), 2: (1, 1)},
        val_change={1: (0, 0), 2: (1, 0)},
        con_change={1: (0, 0), 2: (0, 1)},
    )

    E0 = min(Es)
    Z = np.sum(np.exp(-(Es - E0) / tau))
    (
        eigen_indices,
        eigen_roots,
        color,
        eigen_per_color,
        eigen_basis,
        psis,
        _,
    ) = basis.split_basis_and_redistribute_psi([1 for _ in Es], psis)
    eigen_basis.restrictions = relaxed_restrictions
    if eigen_basis.comm.rank == 0:
        gs = np.zeros((len(tOpsIn), len(wIns), len(tOpsOut), len(wLoss)), dtype=complex)
    for e, psi_e, E_e in zip(eigen_indices, (psis[ei] for ei in eigen_indices), (Es[ei] for ei in eigen_indices)):
        for i, tin in enumerate(tOpsIn):
            psi1 = applyOp_test(tin, psi_e)

            basis_final = eigen_basis.clone(
                initial_basis=[],
                restrictions=excited_restrictions,
                verbose=False,
                comm=eigen_basis.comm.Clone() if eigen_basis.comm is not None else None,
            )
            (
                wIn_indices,
                wIn_roots,
                _,
                wIn_per_color,
                wIn_basis,
                psi1_arr,
                _,
            ) = basis_final.split_basis_and_redistribute_psi([1 for _ in wIns], [psi1])
            indices_for_colors = gather_distributed_results(
                eigen_basis.comm,
                wIn_basis.comm.rank if wIn_basis.comm is not None else 0,
                wIn_roots,
                wIn_per_color,
                np.array(wIn_indices),
                is_array=True,
            )
            if eigen_basis.comm.rank != 0:
                gs = np.zeros((len(tOpsIn), len(wIn_indices), len(tOpsOut), len(wLoss)), dtype=complex)
            basis_tmp = eigen_basis.clone(
                initial_basis=[],
                restrictions=excited_restrictions,
                verbose=False,
                comm=wIn_basis.comm.Clone() if wIn_basis.comm is not None else None,
            )
            psi1 = psi1_arr[0]
            psi2 = ManyBodyState()

            from impurityModel.ed.cg import block_bicgstab

            for k, win in enumerate(wIns[wIn_indices]):
                psi2.prune(slaterWeightMin)
                basis_tmp.clear()
                basis_tmp.add_states(sorted(set(state for p in (psi1, psi2) for state in p.keys())))

                A_op = (
                    ManyBodyOperator(
                        {((0, "c"), (0, "a")): win + delta1 * 1j + E_e, ((0, "a"), (0, "c")): win + delta1 * 1j + E_e}
                    )
                    - hOp
                )

                psi2_list = block_bicgstab(
                    A=A_op,
                    x0=[psi2],
                    y=[psi1],
                    basis=basis_tmp,
                    slaterWeightMin=slaterWeightMin,
                    atol=1e-5,
                    rtol=1e-7,
                )
                psi2 = psi2_list[0]
                for j, tout in enumerate(tOpsOut):
                    psi3 = applyOp_test(tout, psi2)
                    wIn_basis.add_states(psi3.keys())
                    psi3 = wIn_basis.redistribute_psis([psi3])[0]
                    alphas, betas, r = gf.block_Green(
                        hOp,
                        [psi3],
                        wIn_basis,
                        delta2,
                        Reort.NONE,
                        slaterWeightMin=slaterWeightMin,
                        verbose=verbose,
                    )
                    if eigen_basis.comm.rank == 0:
                        gs[i, wIn_indices[k], j, :, None, None] += gf.calc_G(
                            alphas, betas, r, wLoss, E_e, delta2
                        ) * np.exp(-(E_e - E0) / tau)
                    else:
                        gs[i, k, j, :, None, None] += gf.calc_G(alphas, betas, r, wLoss, E_e, delta2) * np.exp(
                            -(E_e - E0) / tau
                        )
            local_gs = (
                gs[i, :, :, :]
                if eigen_basis.comm.rank != 0
                else np.zeros((len(wIn_indices), len(tOpsOut), len(wLoss)), dtype=complex)
            )
            gathered_gs = gather_distributed_results(
                eigen_basis.comm,
                wIn_basis.comm.rank if wIn_basis.comm is not None else 0,
                wIn_roots,
                wIn_per_color,
                local_gs,
                is_array=True,
            )
            if eigen_basis.comm.rank == 0:
                for idx_local, idx_global in enumerate(indices_for_colors):
                    gs[i, idx_global, :, :] += gathered_gs[idx_local, :, :]

            # Free loop-local split/cloned communicators collectively
            if basis_tmp is not None and basis_tmp.comm != eigen_basis.comm:
                basis_tmp.free_comm()
            if wIn_basis is not None and wIn_basis.comm != basis_final.comm:
                wIn_basis.free_comm()
            if basis_final is not None and basis_final.comm != eigen_basis.comm:
                basis_final.free_comm()

    if eigen_basis is not None and eigen_basis.comm != basis.comm:
        eigen_basis.free_comm()

    if basis.comm.rank == 0:
        basis.comm.Reduce(MPI.IN_PLACE, gs, op=MPI.SUM, root=0)
    elif basis.comm.rank in eigen_roots:
        basis.comm.Reduce(gs, None, op=MPI.SUM, root=0)
    else:
        basis.comm.Reduce(
            np.zeros((len(tOpsIn), len(wIns), len(tOpsOut), len(wLoss)), dtype=complex), None, op=MPI.SUM, root=0
        )
    return np.transpose(gs, (0, 2, 1, 3)).copy() / Z
