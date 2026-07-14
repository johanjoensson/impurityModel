"""
This module contains functions for calculating various spectra.
"""

import os
import time
from math import ceil, sqrt

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
from impurityModel.ed.atomic_physics import gauntC
from impurityModel.ed.basis_restrictions import build_excited_restrictions
from impurityModel.ed.cg import block_bicgstab
from impurityModel.ed.gmres import block_gmres
from impurityModel.ed.operator_algebra import arrayOp2Dict, c2i, combineOp, daggerOp
from impurityModel.ed.rational_sampling import barycentric_eval, greedy_next_samples, set_valued_aaa
from impurityModel.ed.BlockLanczosArray import Reort
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState, inner
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
from impurityModel.ed.symmetries import (
    ComponentReduction,
    component_symmetry_reduction,
    conserved_subset_charges,
    extract_tensors,
    measure_conserved_charges,
    transition_sector_restrictions,
)

from impurityModel.ed.symmetries import rotate_hamiltonian

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size

# Adaptive-rotation gate, mirroring selfenergy: rotate the correlated shell into its
# symmetry-adapted basis only when doing so keeps the operator roughly as sparse (a d-shell
# with SOC densifies the Coulomb term ~8x and must stay in spherical harmonics). See the
# "symmetry-rotation-densifies-coulomb" note.
_ROTATION_TRIM_TOL = 1e-8
_MAX_ROTATION_FILL = 2.0


def _rotate_op_dict(tOp_dict, rotation):
    """Rotate a one-body transition-operator dict into the symmetry-adapted basis (U† T U)."""
    return rotate_hamiltonian(ManyBodyOperator(tOp_dict), rotation, tol=_ROTATION_TRIM_TOL).to_dict()


def _shell_orbitals(nBaths, l):
    """Sorted global spin-orbital indices of impurity shell ``l`` (spin-up then spin-down m's)."""
    return sorted(c2i(nBaths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))


def _pes_ips_equivalence_groups(nBaths, l, block_structure):
    """Symmetry-equivalence label per PES/IPS operator of shell ``l`` (B2a dedup).

    ``getPhotoEmissionOperators`` / ``getInversePhotoEmissionOperators`` emit one operator per
    ``(s, m)`` in the order ``for s in 0,1: for m in -l..l``. ``block_structure`` is the impurity
    block structure of shell ``l`` in the symmetry-adapted basis (local indices into the sorted
    shell). Operators whose orbital lands in the same ``identical_blocks`` class get the same
    label, so :func:`getSpectra_new` computes one representative per class.
    """
    shell = _shell_orbitals(nBaths, l)
    local_of_global = {orb: k for k, orb in enumerate(shell)}
    class_of_block = {}
    for cls_id, cls in enumerate(block_structure.identical_blocks):
        for b in cls:
            class_of_block[b] = cls_id
    label_of_local = {}
    for b, block in enumerate(block_structure.blocks):
        for local in block:
            label_of_local[local] = class_of_block.get(b, b)
    return [label_of_local[local_of_global[c2i(nBaths, (l, s, m))]] for s in range(2) for m in range(-l, l + 1)]


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
    rotation=None,
    correlated_l=2,
    correlated_block_structure=None,
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
    rotation : np.ndarray, optional
        Full-space single-particle unitary rotating the correlated shell into its symmetry-adapted
        basis. When given, ``hOp`` and ``psis`` are assumed already expressed in that basis, so the
        one-body transition operators (dipole/NIXS/RIXS) are rotated to match; the scalar spectra
        are basis-invariant and need no un-rotation. ``None`` keeps the spherical-harmonics basis.
    correlated_l : int, optional
        Angular momentum of the rotated correlated shell (default 2, the 3d shell).
    correlated_block_structure : BlockStructure, optional
        Impurity block structure of the correlated shell in the symmetry-adapted basis. When given
        (with ``rotation``), degenerate PES/IPS operators of that shell are deduplicated (B2a).

    """

    # One-body transition operators must be rotated into the same basis as hOp/psis; PES/IPS are
    # bare ladder operators whose integer indices already refer to the rotated orbitals, so they
    # are left as-is and instead deduplicated via the shell's symmetry-equivalence classes.
    def _prep_one_body(tOp_dicts):
        if rotation is None:
            return [ManyBodyOperator(t) for t in tOp_dicts]
        return [ManyBodyOperator(_rotate_op_dict(t, rotation)) for t in tOp_dicts]

    if rotation is not None and correlated_block_structure is not None:
        correlated_groups = _pes_ips_equivalence_groups(nBaths, correlated_l, correlated_block_structure)
    else:
        correlated_groups = None

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
        equivalence_groups=correlated_groups,
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
        equivalence_groups=correlated_groups,
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
        _prep_one_body(tOps),
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
    dN_XAS = dict(
        dN_imp={1: (1, 0), 2: (0, 1)},
        dN_val={1: (0, 0), 2: (1, 0)},
        dN_con={1: (0, 0), 2: (0, 1)},
    )
    if XAS_projectors:
        # Projected operators are not a plain Cartesian linear combination -> keep the
        # per-operator path.
        tOps = getDipoleOperators(nBaths, epsilons)
        iBasisProjectors = arrayOp2Dict(nBaths, XAS_projectors.values())
        projectedTOps = []
        for proj in iBasisProjectors:
            for op in tOps:
                projectedTOps.append(combineOp(nBaths, proj, op))
        gs = getSpectra_new(
            hOp,
            _prep_one_body(projectedTOps),
            psis,
            es,
            tau,
            w,
            basis,
            delta,
            slaterWeightMin,
            verbose,
            occ_cutoff,
            **dN_XAS,
        )
    else:
        # Dipole is linear in the polarization: compute the spectral tensor over the 3 Cartesian
        # components once (symmetry-reduced) and contract with every requested polarization (B2b).
        cartesian_ops = _prep_one_body(getDipoleOperators(nBaths, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        n_orb = basis.num_spin_orbitals
        h_onebody = extract_tensors(hOp, n_orb=n_orb, two_body=False)[0]
        reduction = component_symmetry_reduction(cartesian_ops, h_onebody, n_orb=n_orb)
        gs = getSpectra_tensor(
            hOp,
            cartesian_ops,
            epsilons,
            psis,
            es,
            tau,
            w,
            basis,
            delta,
            slaterWeightMin,
            verbose,
            occ_cutoff,
            reduction=reduction,
            **dN_XAS,
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

        if RIXS_projectors:
            # Projected operators are not a plain Cartesian linear combination -> keep the
            # per-operator Kramers-Heisenberg path.
            tOpsIn = getDipoleOperators(nBaths, epsilonsRIXSin)
            tOpsOut = getDaggeredDipoleOperators(nBaths, epsilonsRIXSout)
            iBasisProjectors = arrayOp2Dict(nBaths, RIXS_projectors.values())
            projectedTOpsIn = []
            projectedTOpsOut = []
            for proj in iBasisProjectors:
                for opIn in tOpsIn:
                    projectedTOpsIn.append(combineOp(nBaths, proj, opIn))
                for opOut in tOpsOut:
                    projectedTOpsOut.append(combineOp(nBaths, opOut, proj))
            gs = getRIXSmap_new(
                hOp,
                _prep_one_body(projectedTOpsIn),
                _prep_one_body(projectedTOpsOut),
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
        else:
            # Dipole is linear in the polarization: compute the full rank-4 Kramers-Heisenberg
            # tensor over the 3 Cartesian in/out components once and contract with every
            # requested (in, out) polarization pair (R4 -- the RIXS analogue of B2b).
            in_component_ops = _prep_one_body(getDipoleOperators(nBaths, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            out_component_ops = _prep_one_body(getDaggeredDipoleOperators(nBaths, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            gs = getRIXSmap_tensor(
                hOp,
                in_component_ops,
                out_component_ops,
                epsilonsRIXSin,
                epsilonsRIXSout,
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
        # Sum over transition operators and save to disk. gs is None on non-root ranks
        # of a distributed run (getRIXSmap_* gathers to global rank 0 only); computing
        # aSum unguarded crashed rank != 0 with AttributeError, leaving rank 0 hung in
        # the post-spectra Barrier.
        if rank == 0:
            aSum = np.sum(-gs.imag, axis=(0, 1))
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


def _sector_restrictions_per_top(hOp, tOps, psis, basis):
    r"""Conserved-charge sector restrictions for each transition operator (or ``None``).

    Each Lanczos seed ``tOp|\psi\rangle`` lives in a definite conserved-charge sector
    (``q_ψ`` shifted by the operator's charge change). Confining the excited basis to that
    sector prunes determinants the per-shell occupation window would otherwise admit -- the
    spectra analogue of :func:`symmetries.gf_sector_restrictions` used by the self-energy.

    Returns a list aligned with ``tOps``; an entry is ``None`` when the operator has no
    definite sector (its terms disagree) so the caller falls back to the occupation window.
    Returns ``None`` (whole list) when the ground states do not share a single charge
    signature -- then no per-operator sector is well defined.
    """
    n_orb = basis.num_spin_orbitals
    comm = basis.comm
    charges = conserved_subset_charges(hOp, n_orb=n_orb)
    gs_occ = None
    for psi in psis:
        occ = measure_conserved_charges(psi, charges, n_orb, comm=comm)
        if gs_occ is None:
            gs_occ = occ
        elif occ != gs_occ:
            # Thermally-populated states span different sectors: no shared confinement.
            return None
    if gs_occ is None:
        return None
    return [transition_sector_restrictions(charges, gs_occ, tOp) for tOp in tOps]


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
    equivalence_groups=None,
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
    equivalence_groups : list, optional
        One hashable label per transition operator. Operators sharing a label are guaranteed
        (by symmetry) to yield identical spectra, so the Lanczos is run for one representative
        per label and the result is broadcast to every member -- the B2a degeneracy dedup.
        ``None`` (default) computes every operator independently.

    Returns
    -------
    ndarray
        A 2D array of shape `(len(w), len(tOps))` containing the complex-valued
        spectra on the real axis. Only returned on root process rank 0; other ranks
        return an empty array.
    """
    if equivalence_groups is not None:
        # Compute one representative per equivalence class, then broadcast columns. Every rank
        # takes the same reduced path (the recursion is collective), so MPI stays in lock-step.
        first_index = {}
        rep_order = []
        for i, label in enumerate(equivalence_groups):
            if label not in first_index:
                first_index[label] = i
                rep_order.append(label)
        reduced = getSpectra_new(
            hOp,
            [tOps[first_index[label]] for label in rep_order],
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
            equivalence_groups=None,
        )
        if reduced.size == 0:  # non-root ranks return an empty array
            return reduced
        pos = {label: k for k, label in enumerate(rep_order)}
        full = np.empty((reduced.shape[0], len(equivalence_groups)), dtype=complex)
        for i, label in enumerate(equivalence_groups):
            full[:, i] = reduced[:, pos[label]]
        return full

    comm = basis.comm
    # Conserved-charge sector confinement, one restriction per transition operator (computed
    # on the full communicator before any basis split, since it measures the ground states).
    sector_restrictions = _sector_restrictions_per_top(hOp, tOps, psis, basis)

    # Shared excited-sector occupation window (identical for every operator); the per-operator
    # window intersects it with that operator's charge sector -- it can only tighten, never
    # loosen. Built on the full basis before the split, so every rank holds the identical list.
    base_restrictions, weighted_restrictions = gf._build_excited_restrictions(
        basis, hOp, psis, es, None, occ_cutoff, dN_imp=dN_imp, dN_val=dN_val, dN_con=dN_con
    )
    if sector_restrictions is None:
        group_restrictions = [base_restrictions] * len(tOps)
    else:
        group_restrictions = [
            base_restrictions if sec is None else gf._intersect_restrictions(base_restrictions, sec)
            for sec in sector_restrictions
        ]

    # Flat work units = (transition operator x eigenstate chunk), distributed in ONE split with
    # the shared cost-model weights -- the same scheme as the self-energy path
    # (gf.get_Greens_function); the engine handles the serial path internally.
    op_groups = [([tOp], delta) for tOp in tOps]
    units, unit_seeds, unit_restrictions = gf.enumerate_gf_units(
        op_groups, psis, group_restrictions, weighted_restrictions, slaterWeightMin
    )
    unit_weights = gf.unit_cost_weights(unit_seeds, comm)

    def kernel(split_basis, u, seeds):
        unit = units[u]
        alphas, betas, r, n_basis, _cap_stats = gf._block_green_group(
            split_basis,
            hOp,
            seeds,
            None,
            unit.delta,
            slaterWeightMin,
            True,
            verbose,
            unit_restrictions[u],
            weighted_restrictions,
        )
        if verbose:
            print(f"Expanded excited state basis contains {n_basis} elements.")
        return alphas, betas, [r[:, p * unit.n_ops : (p + 1) * unit.n_ops] for p in range(len(unit.chunk))]

    results = gf.run_units_distributed(basis, unit_seeds, unit_weights, kernel, verbose=verbose)
    if results is None:  # non-root rank of a distributed run
        return np.empty((0, 0), dtype=complex)

    # Reassemble per-(tOp, eigenstate) coefficients (unit.group_i indexes tOps; at width 1 the
    # operator-split mode emits only diagonal units, so the same reassembly covers both modes),
    # then evaluate the thermal average on the frequency mesh -- on the root rank only, matching
    # the self-energy path and shrinking the gather payload to the Lanczos coefficients.
    acc_alphas = [[None] * len(psis) for _ in tOps]
    acc_betas = [[None] * len(psis) for _ in tOps]
    acc_r = [[None] * len(psis) for _ in tOps]
    for unit, (alphas, betas, r_slices) in zip(units, results):
        for p, ei in enumerate(unit.chunk):
            acc_alphas[unit.group_i][ei] = alphas
            acc_betas[unit.group_i][ei] = betas
            acc_r[unit.group_i][ei] = r_slices[p]

    e0 = np.min(es)
    Z = np.sum(np.exp(-(es - e0) / tau))
    gs_realaxis = np.empty((len(w), len(tOps)), dtype=complex)
    for i in range(len(tOps)):
        G_tOp = gf.calc_thermally_averaged_G(acc_alphas[i], acc_betas[i], acc_r[i], w, es, e0, tau, delta)
        gs_realaxis[:, i] = G_tOp[:, 0, 0] / Z
    return gs_realaxis


def _combine_component_ops(component_ops, coeffs):
    r"""Linear combination :math:`\sum_\alpha c_\alpha T_\alpha` of one-body component operators."""
    combined = {}
    for coeff, op in zip(coeffs, component_ops):
        if abs(coeff) == 0:
            continue
        for factors, amp in (op.to_dict() if hasattr(op, "to_dict") else op).items():
            combined[factors] = combined.get(factors, 0) + coeff * amp
    return ManyBodyOperator(combined)


def _component_seed_moments(hOp, comp_ops, psis, es, e0, tau, basis, slaterWeightMin):
    r"""Thermal seed norms ``<seed|seed>`` and energies ``<seed|H|seed>`` per component.

    Used as the point-group-dedup safety net: symmetry predicts equal moments within a
    dedup group, so a mismatch flags an *incomplete* symmetry multiplet (an ensemble that is
    not actually symmetric) and the caller falls back to the full tensor. States are applied
    locally then redistributed onto a shared working basis so the inner products are complete
    under MPI (the apply-local -> redistribute -> local-inner -> Allreduce pattern).
    """
    m = len(comp_ops)
    comm = basis.comm
    weights = np.exp(-(np.asarray(es) - e0) / tau)
    m0 = np.zeros(m)
    m1 = np.zeros(m)
    work = basis.clone(initial_basis=[], verbose=False, comm=comm)
    for psi, wgt in zip(psis, weights):
        seeds = [applyOp_test(op, psi) for op in comp_ops]
        hseeds = [applyOp_test(hOp, s) for s in seeds]
        work.clear()
        for s in seeds + hseeds:
            work.add_states(s.keys())
        red = work.redistribute_psis(seeds + hseeds)
        for a in range(m):
            m0[a] += wgt * np.real(inner(red[a], red[a]))
            m1[a] += wgt * np.real(inner(red[a], red[m + a]))
    if comm is not None and comm.size > 1:
        comm.Allreduce(MPI.IN_PLACE, m0, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, m1, op=MPI.SUM)
    return m0, m1


def _moments_consistent(m0, m1, group_of_column, tol=1e-6):
    r"""Whether seed moments are equal within every dedup group (symmetry actually holds)."""
    groups = {}
    for a, g in enumerate(group_of_column):
        groups.setdefault(g, []).append(a)
    for members in groups.values():
        if len(members) == 1:
            continue
        ref = members[0]
        scale0 = max(abs(m0[ref]), 1.0)
        for a in members[1:]:
            if abs(m0[a] - m0[ref]) > tol * scale0:
                return False
            # Compare mean energies m1/m0 where the seed is non-trivial.
            if m0[ref] > tol and m0[a] > tol:
                if abs(m1[a] / m0[a] - m1[ref] / m0[ref]) > tol * max(abs(m1[ref] / m0[ref]), 1.0):
                    return False
    return True


def getSpectra_tensor(
    hOp,
    component_ops,
    polarizations,
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
    reduction=None,
):
    r"""Polarization-resolved spectra via the one-body spectral tensor (B2b).

    A dipole (or NIXS) transition operator is *linear* in the polarization,
    :math:`T_\varepsilon = \sum_\alpha \varepsilon_\alpha T_\alpha`, so every polarization's
    spectrum is a contraction of a single Hermitian spectral tensor

    .. math:: \chi_{\alpha\beta}(\omega) = \langle g| T_\alpha^\dagger (\omega - H)^{-1}
              T_\beta |g\rangle, \qquad I_\varepsilon(\omega)
              = \sum_{\alpha\beta} \varepsilon_\alpha^* \chi_{\alpha\beta}(\omega)
              \varepsilon_\beta .

    The tensor is computed with **one** block-Lanczos recurrence over the (symmetry-reduced)
    component operators -- decoupling the number of Lanczos runs from the number of requested
    polarizations, giving arbitrary/circular polarization for free -- and confined to the
    conserved-charge sector of the seeds (B1). ``reduction`` (from
    :func:`symmetries.component_symmetry_reduction`) optionally collapses symmetry-equivalent
    components so the block shrinks further; a seed-moment spot-check guards against an
    incomplete symmetry multiplet and falls back to the full tensor when it fails.

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian.
    component_ops : list of ManyBodyOperator
        The Cartesian component transition operators (e.g. the 3 dipole components).
    polarizations : sequence of array_like
        Each element is a length-``len(component_ops)`` (complex) polarization vector.
    reduction : ComponentReduction, optional
        Point-group reduction of the components. ``None`` computes the full tensor.
    **kwargs
        The remaining parameters match :func:`getSpectra_new`.

    Returns
    -------
    ndarray
        ``(len(w), len(polarizations))`` complex spectra on rank 0; empty array elsewhere.
    """
    m = len(component_ops)
    if reduction is None:
        reduction = ComponentReduction(np.eye(m, dtype=complex), list(range(m)), list(range(m)), m <= 1)
    Q = np.asarray(reduction.Q, dtype=complex)
    diagonalizable = reduction.diagonalizable

    # Representative component operators to actually run the block-Lanczos over.
    rep_ops = [_combine_component_ops(component_ops, Q[:, c]) for c in reduction.representatives]

    # Safety net: if the dedup would drop columns, verify the seed moments really match within
    # each group (the ensemble is a complete symmetry multiplet). Otherwise fall back to full.
    if diagonalizable and len(rep_ops) < m:
        all_rot_ops = [_combine_component_ops(component_ops, Q[:, a]) for a in range(m)]
        e0 = np.min(es)
        m0, m1 = _component_seed_moments(hOp, all_rot_ops, psis, es, e0, tau, basis, slaterWeightMin)
        if not _moments_consistent(m0, m1, reduction.group_of_column):
            diagonalizable = False
            Q = np.eye(m, dtype=complex)
            rep_ops = list(component_ops)
            reduction = reduction._replace(
                Q=Q, representatives=list(range(m)), group_of_column=list(range(m)), diagonalizable=False
            )

    # One conserved-charge sector for the whole block (all components share the charge shift).
    sector = _sector_restrictions_per_top(hOp, [rep_ops[0]], psis, basis)
    extra = None if sector is None else sector[0]

    comm = basis.comm
    e0 = np.min(es)
    Z = np.sum(np.exp(-(es - e0) / tau))
    alphas, betas, r = gf.calc_Greens_function_with_offdiag(
        hOp,
        rep_ops,
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
        extra_restrictions=extra,
    )
    if comm is not None and comm.rank != 0:
        return np.empty((0, 0), dtype=complex)

    chi_red = gf.calc_thermally_averaged_G(alphas, betas, r, w, es, e0, tau, delta) / Z  # (n_w, r, r)

    if diagonalizable:
        rep_diag = np.diagonal(chi_red, axis1=1, axis2=2)  # (n_w, r)
        chi_diag = rep_diag[:, reduction.group_of_column]  # (n_w, m)
        chi_full = np.einsum("wa,pa,qa->wpq", chi_diag, Q, Q.conj(), optimize=True)
    else:
        chi_full = chi_red  # full m x m tensor in the Cartesian basis (Q = I)

    eps = np.array([np.asarray(p, dtype=complex) for p in polarizations])  # (n_pol, m)
    spectra_out = np.einsum("pa,wab,pb->wp", eps.conj(), chi_full, eps, optimize=True)
    return spectra_out


def _rixs_win_chunk(n_eigen: int, n_win: int, comm_size: int) -> int:
    r"""Number of contiguous incoming-photon frequencies stacked into one RIXS work unit.

    A unit is (eigenstate x contiguous wIn-chunk); contiguity preserves the bicgstab
    warm-start locality (consecutive wIn points reuse the previous resolvent solution as the
    initial guess) inside a unit -- a unit is atomic, the engine never reorders within one.
    The default targets ~3 units per rank so the LPT packing has slack to balance, without
    fragmenting the warm-start chains more than needed. Serial runs get one unit per
    eigenstate (maximal warm-start locality). Override with ``GF_RIXS_WIN_CHUNK``.
    """
    env = os.environ.get("GF_RIXS_WIN_CHUNK")
    if env is not None:
        return max(1, int(env))
    if comm_size <= 1:
        return max(1, n_win)
    return max(1, min(n_win, ceil(n_eigen * n_win / (3 * comm_size))))


def _rixs_adaptive_tol():
    """Adaptive-wIn stop tolerance from ``GF_RIXS_ADAPTIVE_TOL``; unset/empty disables."""
    env = os.environ.get("GF_RIXS_ADAPTIVE_TOL")
    return float(env) if env else None


def _rixs_adaptive_batch():
    """New wIn solves per adaptive round (``GF_RIXS_ADAPTIVE_BATCH``, default 1)."""
    return max(1, int(os.environ.get("GF_RIXS_ADAPTIVE_BATCH", 1)))


# Below this many requested wIn points the adaptive sampler cannot beat the dense sweep
# (its initial space-filling sample plus the two-quiet-rounds stop already cost that much).
_RIXS_ADAPTIVE_MIN_GRID = 12
# Fit-component subsample bound: the set-valued AAA weights are determined from at most this
# many components (strided across polarization pairs x wLoss); the final reconstruction uses
# the full component set with the shared weights, which is exact for shared-pole functions.
_RIXS_ADAPTIVE_MAX_FIT_COMPONENTS = 4096
# A reconstructed magnitude exceeding the solved data's envelope by this factor at an
# unsolved node is treated as a spurious barycentric pole (Froissart artifact) and forces a
# solve there; legitimate inter-sample peaks are excluded by the grid being finer than the
# physical broadening.
_RIXS_ADAPTIVE_BLOWUP_FACTOR = 2.0
# Relative residual target of the RIXS intermediate (R1) resolvent solves -- shared by
# every solver on that path (shift-recycled Krylov, BiCGSTAB, its GMRES escalation), so
# a point rescued by a different solver meets the same accuracy. It was 1e-5 while
# BiCGSTAB applied the tolerance to the *warm-start* residual instead, which on this
# sweep is ~10x smaller -- so 1e-6 is what the RIXS map was actually getting, and keeps
# the measured accuracy-vs-dense at 2.6e-7 (test_rixs_tensor_perf). Tighten to 1e-7 for
# ~6e-9 if a map ever needs it.
_RIXS_R1_ATOL = 1e-6


def _rixs_map_adaptive(map_fn, wIns, comm, tol, verbose):
    r"""Greedy adaptive-wIn evaluation of a RIXS map via set-valued AAA.

    Every component of the map (polarization pairs x energy-loss points) shares its
    ``wIn`` poles -- the intermediate core-hole resolvent's -- so the whole map is a
    vector-valued rational function of ``wIn`` and can be reconstructed from solves at a
    few support points (measured on NiO L3: 20 of 121 points at 1e-3, ``doc`` Gate B).

    Strategy: solve a small space-filling initial sample, fit a set-valued AAA
    approximant (one shared support/weight set for a strided component subsample),
    and iterate greedily -- each round solves the wIn point(s) where two consecutive
    approximants disagree most, until they agree within ``tol * max|map|`` on every
    unsolved point for two consecutive rounds (the standard guard against the
    lookahead's failure mode: two iterates agreeing prematurely). Solved points enter
    the returned map exactly; unsolved points are barycentric-evaluated with the full
    component set. Falls back to the dense sweep (all points solved) if convergence
    never sets in.

    MPI: the greedy selection runs on global rank 0 and is broadcast each round;
    ``map_fn`` (collective) is called by every rank with the identical wIn subset, so
    all collectives stay in lock-step. Returns the assembled map on rank 0, ``None``
    elsewhere (the :func:`_rixs_map_flat` contract).

    Trade-off: within one round the warm-start chain only spans that round's batch, so
    each adaptive solve pays more bicgstab iterations than a dense-sweep point; the
    win is the ~5-10x reduction in the number of solves.
    """
    wIns = np.asarray(wIns)
    n = len(wIns)
    root = comm is None or comm.rank == 0
    batch_size = _rixs_adaptive_batch()

    solved: list[int] = []
    cols = {}  # wIn index -> (n_i, n_o, n_l) map column; root only
    fit_idx = None  # strided component subsample, fixed on first solve
    prev_R = None
    quiet_rounds = 0
    min_solves = min(n, 8)
    next_batch = sorted(set(int(i) for i in np.round(np.linspace(0, n - 1, min(5, n)))))
    n_rounds = 0
    last_fit = None  # (support indices into `solved`, weights) of the last guarded fit

    while True:
        if comm is not None:
            next_batch = comm.bcast(next_batch, root=0)
        if not next_batch:
            break
        n_rounds += 1
        sub = map_fn(wIns[np.asarray(next_batch)])  # collective
        if root:
            for k, idx in enumerate(next_batch):
                cols[idx] = sub[:, :, k, :]
            solved.extend(next_batch)
            x_solved = wIns[np.asarray(solved)]
            F_full = np.array([cols[i].reshape(-1) for i in solved])  # (n_solved, K)
            if fit_idx is None:
                stride = max(1, ceil(F_full.shape[1] / _RIXS_ADAPTIVE_MAX_FIT_COMPONENTS))
                fit_idx = np.arange(0, F_full.shape[1], stride)
            F_fit = F_full[:, fit_idx]
            scale = np.max(np.abs(F_fit))
            support, weights = set_valued_aaa(x_solved, F_fit, rtol=0.1 * tol)
            last_fit = (support, weights)
            R = barycentric_eval(wIns, x_solved[support], weights, F_fit[support])
            if prev_R is None or scale == 0.0:
                surrogate = None
            else:
                surrogate = np.max(np.abs(R - prev_R), axis=1) / scale
            prev_R = R
            unsolved = [i for i in range(n) if i not in set(solved)]
            # Spurious-pole (Froissart) guard: a barycentric denominator zero between
            # support points makes the reconstruction blow up at nodes no data pins
            # down -- and BOTH consecutive iterates can share the artifact, so the
            # surrogate alone would happily converge on it (that exact failure produced
            # a 3x-of-scale spike on the NiO L3 map). With the grid finer than the
            # physical broadening, no true inter-sample feature can exceed the sampled
            # envelope by this factor: treat any such node as must-solve.
            recon_mag = np.max(np.abs(R), axis=1)
            blown = [
                i
                for i in unsolved
                if not np.isfinite(recon_mag[i]) or recon_mag[i] > _RIXS_ADAPTIVE_BLOWUP_FACTOR * scale
            ]
            if blown:
                if surrogate is None:
                    surrogate = np.zeros(n)
                finite_max = float(np.max(surrogate[np.isfinite(surrogate)], initial=tol))
                surrogate[blown] = 10.0 * finite_max
            converged = not unsolved or (
                not blown and len(solved) >= min_solves and surrogate is not None and np.max(surrogate[unsolved]) <= tol
            )
            quiet_rounds = quiet_rounds + 1 if converged else 0
            if not unsolved or quiet_rounds >= 2:
                next_batch = []
            else:
                next_batch = greedy_next_samples(wIns, solved, surrogate, batch_size)
        else:
            next_batch = None

    if not root:
        return None
    n_i, n_o, n_l = cols[solved[0]].shape
    gs = np.empty((n_i, n_o, n, n_l), dtype=complex)
    for idx in solved:
        gs[:, :, idx, :] = cols[idx]
    unsolved = [i for i in range(n) if i not in set(solved)]
    if unsolved:
        # Reuse the loop's LAST fit -- it is the one the blow-up guard vetted; a fresh
        # refit here could reintroduce an unchecked artifact.
        support, weights = last_fit
        x_solved = wIns[np.asarray(solved)]
        F_full = np.array([cols[i].reshape(-1) for i in solved])
        recon = barycentric_eval(wIns[np.asarray(unsolved)], x_solved[support], weights, F_full[support])
        for k, idx in enumerate(unsolved):
            gs[:, :, idx, :] = recon[k].reshape(n_i, n_o, n_l)
    if verbose:
        print(
            f"Adaptive RIXS wIn sampling: solved {len(solved)}/{n} points in {n_rounds} rounds "
            f"({len(last_fit[0]) if unsolved else 0} support points; tol {tol:g})."
        )
        print(f"  solved wIn: {np.array2string(np.sort(wIns[np.asarray(solved)]), precision=4, max_line_width=100)}")
    return gs


def _rixs_map_flat(
    hOp,
    in_ops,
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
    n_i,
    n_o,
    eval_out,
    r1_caches=None,
):
    r"""Shared flat-unit RIXS driver behind :func:`getRIXSmap_new` and :func:`getRIXSmap_tensor`.

    Work units = (eigenstate x contiguous wIn-chunk), distributed in ONE weighted split through
    the shared engine (:func:`greens_function.run_units_distributed`) -- the same scheme as the
    self-energy and spectra paths. Per-eigenstate metadata (in-component seeds
    ``Tin_a |psi_e>``, conserved-charge sector windows) is computed on the full communicator
    before the split, so every rank holds the identical unit list.

    Each unit's kernel walks its wIn chunk in order: warm-started :func:`cg.block_bicgstab`
    for the intermediate resolvent (R1 sector confinement, R2 in-component block), then
    ``eval_out(green_basis, psi2_all, E_e) -> (n_i, n_o, len(wLoss))`` evaluates the
    out-transition Green's functions (per-pair diagonal or full tensor contraction).

    ``r1_caches`` (dict, eigenstate index -> :class:`greens_function.SectorResolventCache`,
    owned by the caller so it survives repeated invocations, e.g. adaptive-sampling rounds)
    solves the intermediate resolvent spectrally on cacheable sectors -- exact and immune to
    the near-pole BiCGSTAB stagnation that silently poisoned solved columns (measured: a
    cold-started solve at the NiO L3 window edge returned relative residual 7.2 while
    targeting 1e-6). Sectors the cache declines (distributed bases, oversized sectors) go to
    :class:`greens_function.KrylovShiftedResolvent`: one distributed block-Lanczos recurrence
    per chunk serves every remaining wIn shift (the right-hand side is wIn-independent).
    Should that decline too (memory bound), the per-point ``block_bicgstab`` fallback runs,
    restarted while progressing and escalated to ``block_gmres`` when stagnated -- every tier
    targets the same ``_RIXS_R1_ATOL``.

    Returns ``gs[i, o, wIn, wLoss] / Z`` (thermally averaged) on global rank 0 and in the
    serial path; ``None`` on other ranks.
    """
    excited_restrictions = build_excited_restrictions(
        basis,
        hOp,
        psis,
        Es,
        imp_change={1: (1, 0), 2: (1, 1)},
        val_change={1: (0, 0), 2: (1, 0)},
        con_change={1: (0, 0), 2: (0, 1)},
    )

    E0 = min(Es)
    Z = np.sum(np.exp(-(Es - E0) / tau))
    comm = basis.comm
    n_win = len(wIns)

    # Conserved-charge sector of the core-excited intermediate state (all in-components share
    # the same charge shift): confines the resolvent solve (R1). Computed per eigenstate on the
    # full communicator (collective, lock-step) before the split.
    charges = conserved_subset_charges(hOp, n_orb=basis.num_spin_orbitals)
    psi1_per_e = [[applyOp_test(tin, psi_e) for tin in in_ops] for psi_e in psis]
    tmp_restrictions_per_e = []
    for psi_e in psis:
        tmp = excited_restrictions
        if charges:
            gs_occ = measure_conserved_charges(psi_e, charges, basis.num_spin_orbitals, comm=comm)
            sector_in = transition_sector_restrictions(charges, gs_occ, in_ops[0])
            if sector_in:
                tmp = gf._intersect_restrictions(excited_restrictions, sector_in)
        tmp_restrictions_per_e.append(tmp)

    # Flat work units. Unit seeds are the eigenstate's in-component excitations (duplicated
    # across its wIn chunks -- core-excited seeds are small); the unit weight is the shared
    # cost model scaled by the chunk's wIn count (resolvent cost is linear in wIn points).
    chunk_size = _rixs_win_chunk(len(Es), n_win, 1 if comm is None else comm.size)
    unit_infos = []  # (eigenstate index, contiguous wIn indices) per unit
    unit_seeds = []
    for e in range(len(Es)):
        for start in range(0, n_win, chunk_size):
            unit_infos.append((e, list(range(start, min(start + chunk_size, n_win)))))
            unit_seeds.append(psi1_per_e[e])
    unit_weights = gf.unit_cost_weights(unit_seeds, comm) * np.array(
        [len(chunk) for _, chunk in unit_infos], dtype=float
    )

    # green_basis depends only on excited_restrictions (identical for every unit), so each
    # color creates it once on its first unit (lazily -- all ranks of a color run the same
    # unit list, so the collective Clone stays in lock-step) and clears it between units
    # instead of cloning a fresh sub-communicator + basis per unit. Freed collectively after
    # run_units_distributed returns on all ranks.
    green_basis_cache = {}

    def kernel(split_basis, u, seeds):
        e, w_chunk = unit_infos[u]
        E_e = Es[e]
        thermal_weight = np.exp(-(E_e - E0) / tau)
        sub_comm = split_basis.comm
        # green_basis hosts the out-transition block-Green solves and accumulates states over
        # the chunk; tmp_basis hosts the intermediate resolvent and is rebuilt per wIn point.
        green_basis = green_basis_cache.get(id(split_basis))
        if green_basis is None:
            green_basis = split_basis.clone(
                initial_basis=[],
                restrictions=excited_restrictions,
                verbose=False,
                comm=sub_comm.Clone() if sub_comm is not None else None,
            )
            green_basis_cache[id(split_basis)] = green_basis
        else:
            green_basis.clear()
        tmp_basis = split_basis.clone(
            initial_basis=[],
            restrictions=tmp_restrictions_per_e[e],
            verbose=False,
            comm=sub_comm.Clone() if sub_comm is not None else None,
        )
        psi1_all = list(seeds)
        psi2_all = [ManyBodyState() for _ in in_ops]
        r1_cache = r1_caches.setdefault(e, gf.SectorResolventCache()) if r1_caches is not None else None
        r1_recycled = None  # chunk index -> shift-recycled solutions (filled on first dense decline)
        r1_recycle_declined = r1_cache is None
        out = np.zeros((len(w_chunk), n_i, n_o, len(wLoss)), dtype=complex)
        for k, win in enumerate(wIns[w_chunk]):
            if r1_cache is not None:
                psi2_spectral = r1_cache.try_solve(
                    tmp_basis,
                    hOp,
                    psi1_all,
                    win + delta1 * 1j + E_e,
                    slaterWeightMin=slaterWeightMin,
                    verbose=verbose,
                )
                if psi2_spectral is not None:
                    out[k] = eval_out(green_basis, psi2_spectral, E_e) * thermal_weight
                    continue
                if r1_recycled is None and not r1_recycle_declined:
                    # The dense sector cache declined (distributed basis or oversized
                    # sector): recycle ONE block-Lanczos recurrence across every remaining
                    # shift of the chunk -- the right-hand-side block is wIn-independent,
                    # so all shifts share the same Krylov space.
                    sols = gf.KrylovShiftedResolvent().solve(
                        tmp_basis,
                        hOp,
                        psi1_all,
                        wIns[w_chunk][k:] + delta1 * 1j + E_e,
                        slaterWeightMin=slaterWeightMin,
                        atol=_RIXS_R1_ATOL,
                        verbose=verbose,
                    )
                    if sols is None:
                        r1_recycle_declined = True
                    else:
                        r1_recycled = dict(zip(range(k, len(w_chunk)), sols))
                if r1_recycled is not None:
                    out[k] = eval_out(green_basis, r1_recycled.pop(k), E_e) * thermal_weight
                    continue
            for psi2 in psi2_all:
                psi2.prune(slaterWeightMin)
            tmp_basis.clear()
            tmp_basis.add_states(sorted(set(state for p in psi1_all + psi2_all for state in p.keys())))
            # Align seeds and warm starts to tmp_basis's ownership layout -- the solver assumes
            # its states are distributed per `basis`, and the layout of the freshly rebuilt
            # tmp_basis need not match where the amplitudes currently live.
            redistributed = tmp_basis.redistribute_psis(psi1_all + psi2_all)
            psi1_all = list(redistributed[: len(psi1_all)])
            psi2_all = list(redistributed[len(psi1_all) :])
            A_op = (
                ManyBodyOperator(
                    {((0, "c"), (0, "a")): win + delta1 * 1j + E_e, ((0, "a"), (0, "c")): win + delta1 * 1j + E_e}
                )
                - hOp
            )
            # Warm-started resolvent solved as one block over all in-components, sharing a
            # single Krylov space / iteration (block_bicgstab deflates a rank-deficient block).
            # atol is relative to ||psi1_all|| (see _RIXS_R1_ATOL); the extra iterations are
            # cheap now that a warm start shortens the solve rather than silently tightening
            # its target. Restarted while unconverged and still making progress (each call
            # re-deflates the residual and picks a fresh shadow residual).
            solve_info = {}
            prev_residual = np.inf
            for _attempt in range(1 + gf._GF_BICGSTAB_RESTARTS):
                psi2_all = block_bicgstab(
                    A=A_op,
                    x0=psi2_all,
                    y=psi1_all,
                    basis=tmp_basis,
                    slaterWeightMin=slaterWeightMin,
                    atol=_RIXS_R1_ATOL,
                    rtol=1e-7,
                    info=solve_info,
                )
                if (
                    solve_info["converged"]
                    or solve_info["rel_residual"] > gf._GF_BICGSTAB_RESTART_PROGRESS * prev_residual
                ):
                    break
                prev_residual = solve_info["rel_residual"]
            # GMRES escalation, warm-started from BiCGSTAB's partial iterate: near-pole
            # points are exactly where BiCGSTAB stagnates (measured: a cold-started solve
            # at the NiO L3 window edge silently returned relative residual 7.2), and a
            # stagnated solve caps the map's accuracy at its residual level -- so a wrong
            # column must be rescued, and failing that, loud.
            if not solve_info["converged"]:
                psi2_all = block_gmres(
                    A_op,
                    psi2_all,
                    psi1_all,
                    tmp_basis,
                    slaterWeightMin,
                    atol=_RIXS_R1_ATOL,
                    restart=gf._GF_GMRES_RESTART,
                    max_restarts=gf._GF_GMRES_MAX_RESTARTS,
                    info=solve_info,
                )
            if not solve_info["converged"] and (sub_comm is None or sub_comm.rank == 0):
                print(
                    f"warning: RIXS intermediate resolvent at wIn = {win:.6g} (eigenstate {e}) "
                    f"stopped unconverged at relative residual "
                    f"{solve_info.get('rel_residual', float('nan')):.2e} (after GMRES escalation).",
                    flush=True,
                )
            out[k] = eval_out(green_basis, psi2_all, E_e) * thermal_weight
        # Free the per-unit cloned sub-communicator collectively -- every rank of this color
        # runs the same unit list in the same order. green_basis's clone outlives the unit
        # (per-color cache) and is freed after run_units_distributed.
        if sub_comm is not None:
            tmp_basis.free_comm()
        return out

    # Accumulate each unit's contribution into the preallocated output as it arrives, so
    # rank 0 never holds all unit results plus the assembled tensor simultaneously.
    gs = np.zeros((n_i, n_win, n_o, len(wLoss)), dtype=complex)

    def accumulate(u, res):
        _e, w_chunk = unit_infos[u]
        for k, w_global in enumerate(w_chunk):
            gs[:, w_global, :, :] += res[k]

    got = gf.run_units_distributed(basis, unit_seeds, unit_weights, kernel, verbose=verbose, reduce_fn=accumulate)
    # Collective on each color's clone: every rank of a color created (at most) one cached
    # green_basis and reaches this point after its unit loop.
    for cached in green_basis_cache.values():
        if cached.comm is not None:
            cached.free_comm()
    green_basis_cache.clear()
    if got is None:  # non-root rank of a distributed run
        return None
    return np.transpose(gs, (0, 2, 1, 3)).copy() / Z


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

    The map ``gs[in, out, wIn, wLoss]`` is the Kramers-Heisenberg intensity for every
    (in-operator, out-operator) pair. Two efficiency levers are applied on top of the
    straightforward per-pair evaluation (numerically identical results):

    * **Conserved-charge sector confinement (R1):** the intermediate resolvent
      ``(wIn + i delta1 + E - H)^{-1} Tin|psi>`` stays in the core-excited charge sector of
      ``Tin|psi>``, so the resolvent basis is pinned to that sector
      (:func:`symmetries.transition_sector_restrictions`) on top of the occupation window.
    * **In-component block resolvent (R2):** for each ``wIn`` all in-operators' resolvents are
      solved as one block (:func:`cg.block_bicgstab`), sharing a single Krylov space /
      iteration; the block solver deflates a rank-deficient in-component right-hand side.
    * **Out-component block Green (R3):** for a fixed in-operator all out-operators are run
      through a single block-Lanczos (:func:`greens_function.block_Green`); the diagonal
      ``(j, j)`` of the resulting block reproduces the per-out-operator Green's function.

    (A full polarization tensor with arbitrary in/out polarizations would require the rank-4
    cross tensor and is not computed; the map is over the supplied operator pairs.)

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
    hOp : ManyBodyOperator
        The Hamiltonian.
    tOpsIn : list of ManyBodyOperator
        Transition operators describing the core-hole excitation.
    tOpsOut : list of ManyBodyOperator
        Transition operators describing the filling of the core-hole.
    psis : list of ManyBodyState
        Thermal eigenstates.
    Es : list of float
        Total energies of the eigenstates.
    wIns : ndarray
        Real axis energy mesh for the incoming photon energy.
    wLoss : ndarray
        Real axis energy mesh for the photon energy loss, i.e. ``wLoss = wIns - wOut``.
    delta1 : float
        Deviation from the real axis for the intermediate (core-excited) resolvent.
    delta2 : float
        Deviation from the real axis for the final (energy-loss) resolvent.
    basis : Basis
        The basis container (carries the communicator).
    slaterWeightMin : float
        Restrict the number of product states by looking at ``|amplitudes|^2``.

    Returns
    -------
    ndarray or None
        ``gs[in, out, wIn, wLoss]`` (thermally averaged) on global rank 0 and in the serial
        path; ``None`` on other ranks.
    """
    n_in = len(tOpsIn)
    n_out = len(tOpsOut)

    def eval_out(green_basis, psi2_all, E_e):
        out = np.zeros((n_in, n_out, len(wLoss)), dtype=complex)
        for i in range(n_in):
            # R3: build the final states for every out-component and run one block-Green over
            # them; the diagonal (out-component j vs itself) reproduces the per-operator result.
            psi3_all = [applyOp_test(tout, psi2_all[i]) for tout in tOpsOut]
            for psi3 in psi3_all:
                green_basis.add_states(psi3.keys())
            psi3_all = green_basis.redistribute_psis(psi3_all)
            alphas, betas, r = gf.block_Green(
                hOp,
                psi3_all,
                green_basis,
                delta2,
                Reort.NONE,
                slaterWeightMin=slaterWeightMin,
                verbose=verbose,
            )
            g_tensor = gf.calc_G(alphas, betas, r, wLoss, E_e, delta2)
            for j in range(n_out):
                out[i, j, :] = g_tensor[:, j, j]
        return out

    return _rixs_map_flat(
        hOp,
        tOpsIn,
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
        n_i=n_in,
        n_o=n_out,
        eval_out=eval_out,
    )


def getRIXSmap_tensor(
    hOp,
    in_component_ops,
    out_component_ops,
    epsilonsIn,
    epsilonsOut,
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
    adaptive_wIn_tol=None,
):
    r"""RIXS map for arbitrary in/out polarizations via the full rank-4 Kramers-Heisenberg tensor.

    A dipole (or NIXS) transition operator is *linear* in the polarization,
    :math:`T_\varepsilon = \sum_\alpha \varepsilon_\alpha T_\alpha`, so the Kramers-Heisenberg
    amplitude for every pair of in/out polarizations is a contraction of a single
    Cartesian-component tensor

    .. math:: C_{\alpha\alpha'\beta\beta'}(\omega_\text{in}, \omega_\text{loss}) =
              \langle \psi^{(2)}_\alpha | T^{\text{out}\dagger}_\beta R_2 T^\text{out}_{\beta'}
              | \psi^{(2)}_{\alpha'} \rangle, \qquad
              \psi^{(2)}_\alpha = R_1 T^\text{in}_\alpha |g\rangle,

    with :math:`R_1 = (\omega_\text{in} + i\delta_1 + E - H)^{-1}` and
    :math:`R_2 = (\omega_\text{loss} + i\delta_2 + E - H)^{-1}`. Since
    :math:`C_{\alpha\alpha'\beta\beta'} = \langle s_{\alpha\beta} | R_2 | s_{\alpha'\beta'}
    \rangle` with the seeds :math:`s_{\alpha\beta} = T^\text{out}_\beta \psi^{(2)}_\alpha`, the
    tensor is exactly the resolvent matrix over the flattened seed block -- one block-Lanczos
    (:func:`greens_function.block_Green`) yields every polarization cross term at once, and the
    number of solves is decoupled from the number of requested polarizations (arbitrary /
    circular polarization for free). This is the RIXS analogue of :func:`getSpectra_tensor`.

    The same efficiency levers as :func:`getRIXSmap_new` apply -- R1 conserved-charge sector
    confinement of the intermediate resolvent and the R2 block resolvent over the in-components
    (:func:`cg.block_bicgstab`, which deflates the frequently rank-deficient Cartesian
    in-component right-hand side). The redundancy of symmetry-equivalent seeds is handled
    exactly and automatically by the rank deflation inside ``block_bicgstab`` and
    ``block_Green`` (a symmetry-based seed drop cannot reduce below the linear rank of the seed
    span, and the XAS-style group rule does not generalize to this rank-4 tensor).

    Parameters
    ----------
    hOp : ManyBodyOperator
        The Hamiltonian.
    in_component_ops : list of ManyBodyOperator
        Cartesian in-transition (core-hole excitation) component operators, e.g. the three
        dipole components ``getDipoleOperators(nBaths, [[1,0,0],[0,1,0],[0,0,1]])``.
    out_component_ops : list of ManyBodyOperator
        Cartesian out-transition (core-hole filling) component operators, e.g. the daggered
        dipole components ``getDaggeredDipoleOperators(nBaths, [[1,0,0],[0,1,0],[0,0,1]])``.
    epsilonsIn, epsilonsOut : sequence of array_like
        In/out polarization vectors, each of length ``len(in_component_ops)`` /
        ``len(out_component_ops)`` (real or complex).
    adaptive_wIn_tol : float, optional
        Enable greedy adaptive sampling of the ``wIns`` grid (:func:`_rixs_map_adaptive`):
        only the AAA-selected support frequencies are actually solved, the rest are
        rational-reconstructed to this relative tolerance. ``None`` (default) reads
        ``GF_RIXS_ADAPTIVE_TOL`` from the environment; unset there too means dense.
        Grids shorter than ``_RIXS_ADAPTIVE_MIN_GRID`` are always solved densely.
    **kwargs
        The remaining parameters match :func:`getRIXSmap_new`.

    Returns
    -------
    ndarray
        ``gs[p_in, p_out, wIn, wLoss]`` on rank 0, thermally averaged.
    """
    epsIn = np.array([np.asarray(p, dtype=complex) for p in epsilonsIn])  # (n_pin, n_in_comp)
    epsOut = np.array([np.asarray(p, dtype=complex) for p in epsilonsOut])  # (n_pout, n_out_comp)
    n_pin = epsIn.shape[0]
    n_pout = epsOut.shape[0]
    n_in = len(in_component_ops)
    n_out = len(out_component_ops)

    # The R1 (per eigenstate) and R2 sectors are the same for every wIn point and
    # adaptive round (the shift enters only at evaluation time), so each sector's
    # eigendecomposition is computed once: every point's intermediate solve and
    # resolvent matrix become dense contractions. Held in this closure so they outlive
    # the per-round _rixs_map_flat calls of the adaptive sampler.
    r1_caches = {}
    r2_cache = gf.SectorResolventCache()

    def eval_out(green_basis, psi2_all, E_e):
        # Flattened out-seed block s_{a,b} = Tout_b psi2_a; index kf = a * n_out + b. One
        # block-Green over all seeds gives the full resolvent matrix (every cross term).
        seeds = [applyOp_test(out_component_ops[b], psi2_all[a]) for a in range(n_in) for b in range(n_out)]
        g_flat = r2_cache.try_eval(
            green_basis, hOp, seeds, wLoss + 1j * delta2 + E_e, slaterWeightMin=slaterWeightMin, verbose=verbose
        )
        if g_flat is None:  # distributed or over the dense-size bound: per-seed block-Lanczos
            for s in seeds:
                green_basis.add_states(s.keys())
            seeds = green_basis.redistribute_psis(seeds)
            alphas, betas, r = gf.block_Green(
                hOp,
                seeds,
                green_basis,
                delta2,
                Reort.NONE,
                slaterWeightMin=slaterWeightMin,
                verbose=verbose,
            )
            g_flat = gf.calc_G(alphas, betas, r, wLoss, E_e, delta2)
        # C[w, alpha, beta, alpha', beta'] = <s_{alpha,beta}| R2 |s_{alpha',beta'}>.
        C5 = g_flat.reshape(len(wLoss), n_in, n_out, n_in, n_out)
        # Contract with polarizations. Out operators are daggered (getDaggeredDipole..),
        # T_out(eps) = sum_b eps_b^* Tout_b, so eps_out is unconjugated on the R2-ket seed
        # index (beta) and conjugated on the bra index (beta'); in operators carry no dagger.
        return np.einsum(
            "pa,qb,pc,qd,wabcd->pqw",
            epsIn.conj(),
            epsOut,
            epsIn,
            epsOut.conj(),
            C5,
            optimize=True,
        )  # (n_pin, n_pout, n_wLoss)

    def map_fn(wIn_subset):
        return _rixs_map_flat(
            hOp,
            in_component_ops,
            psis,
            Es,
            tau,
            wIn_subset,
            wLoss,
            delta1,
            delta2,
            basis,
            verbose,
            slaterWeightMin,
            n_i=n_pin,
            n_o=n_pout,
            eval_out=eval_out,
            r1_caches=r1_caches,
        )

    tol = adaptive_wIn_tol if adaptive_wIn_tol is not None else _rixs_adaptive_tol()
    if tol is not None and len(wIns) >= _RIXS_ADAPTIVE_MIN_GRID:
        return _rixs_map_adaptive(map_fn, wIns, basis.comm, tol, verbose)
    return map_fn(np.asarray(wIns))
