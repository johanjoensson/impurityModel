"""Second-quantized transition operators for the spectroscopy drivers.

Pure operator builders (dipole for XAS/RIXS, the plane-wave NIXS operator, and the bare
photo-emission / inverse-photo-emission ladder operators) plus the ``sph_harm`` helper the
NIXS expansion uses. Each returns an operator as a ``{process: amplitude}`` dict in the
convention the rest of the package consumes.

This is a physics layer: it depends only on the single-shell atomic physics (Gaunt
coefficients), the ``(l, s, m)`` <-> flat-index conversion, and the operator algebra --
never on a solver or on :mod:`spectra`. :mod:`spectra` builds its transition operators
through these functions.
"""

from math import sqrt

import numpy as np
import scipy.integrate as si
from mpi4py import MPI
from scipy.special import sph_harm_y, spherical_jn

from impurityModel.ed.atomic_physics import gauntC
from impurityModel.ed.operator_algebra import c2i, daggerOp

comm = MPI.COMM_WORLD
rank = comm.rank


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


def dipole_operators(nBaths, ns):
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
        tOps.append(dipole_operator(nBaths, n))
    return tOps


def daggered_dipole_operators(nBaths, ns):
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
        tDaggerOps.append(daggerOp(dipole_operator(nBaths, n)))
    return tDaggerOps


def dipole_operator(nBaths, n):
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


def nixs_operators(nBaths, qs, li, lj, Ri, Rj, r, kmin=1):
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
        tOps.append(nixs_operator(nBaths, q, li, lj, Ri, Rj, r, kmin))
    return tOps


def nixs_operator(nBaths, q, li, lj, Ri, Rj, r, kmin=1):
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


def inverse_photoemission_operators(nBaths, l=2):
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


def photoemission_operators(nBaths, l=2):
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
