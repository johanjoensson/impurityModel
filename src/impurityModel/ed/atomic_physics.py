"""
Single-shell atomic physics: Slater-Condon Coulomb integrals, spin-orbit
coupling, Zeeman field, spherical<->cubic harmonics transforms, and the MLFT
double-counting correction.
"""

import itertools
from math import pi, sqrt
from typing import Optional

import numpy as np
from sympy.physics.wigner import gaunt, wigner_3j
from sympy import Rational


from impurityModel.ed.operator_algebra import addOps


def dc_MLFT(n3d_i, c, Fdd, n2p_i=None, Fpd=None, Gpd=None):
    r"""
    Return double counting (DC) in multiplet ligand field theory.

    Parameters
    ----------
    n3d_i : int
        Nominal (integer) 3d occupation.
    c : float
        Many-body correction to the charge transfer energy.
    n2p_i : int
        Nominal (integer) 2p occupation.
    Fdd : list
        Slater integrals {F_{dd}^k}, k \in [0,1,2,3,4]
    Fpd : list
        Slater integrals {F_{pd}^k}, k \in [0,1,2]
    Gpd : list
        Slater integrals {G_{pd}^k}, k \in [0,1,2,3]

    Notes
    -----
    The `c` parameter is related to the charge-transfer
    energy :math:`\Delta_{CT}` by:

    .. math:: \Delta_{CT} = (e_d-e_b) + c.

    """
    if not int(n3d_i) == n3d_i:
        raise ValueError("3d occupation should be an integer")
    if n2p_i is not None and int(n2p_i) != n2p_i:
        raise ValueError("2p occupation should be an integer")

    # Average repulsion energy defines Udd and Upd
    Udd = Fdd[0] - 14.0 / 441 * (Fdd[2] + Fdd[4])
    if n2p_i is None and Fpd is None and Gpd is None:
        return {2: Udd * n3d_i - c}
    if n2p_i == 6 and Fpd is not None and Gpd is not None:
        Upd = Fpd[0] - (1 / 15.0) * Gpd[1] - (3 / 70.0) * Gpd[3]
        return {2: Udd * n3d_i + Upd * n2p_i - c, 1: Upd * (n3d_i + 1) - c}
    else:
        raise ValueError("double counting input wrong.")


def uj_from_u4(u4, tol=1e-8):
    r"""Average Coulomb repulsion :math:`\bar U` and average exchange :math:`\bar J` from a
    spherical, spin-collinear Coulomb tensor.

    ``u4`` must be in the RSPt convention (``u4[i,j,k,l] = <ij|V|kl>``, see
    :func:`getUop_from_rspt_u4`), in the spherical-harmonics basis, with the spin-major
    ordering :func:`impurityModel.ed.operator_algebra.c2i` produces: the first half of the
    spin-orbital indices is one spin (all orbitals), the second half the other.

    This is the Anisimov / LDA+U spherical average that pairs with :math:`\bar U = F^0` in the
    FLL and AMF double-counting functionals -- not the bare average exchange integral. It is
    the direct average, corrected by the *off-diagonal* part of the direct average and replaced
    by the off-diagonal exchange average:

    .. math::
        \bar U &= \frac{1}{n_{orb}^2} \sum_{m,m'} \langle m\uparrow, m'\downarrow|V|
            m\uparrow, m'\downarrow\rangle, \\
        \bar J &= \bar U - \frac{1}{n_{orb}(n_{orb}-1)} \sum_{m \neq m'} \langle m\uparrow,
            m'\downarrow|V|m\uparrow, m'\downarrow\rangle + \frac{1}{n_{orb}(n_{orb}-1)}
            \sum_{m \neq m'} \langle m\uparrow, m'\uparrow|V|m'\uparrow, m\uparrow\rangle.

    For a single-``l``-shell atomic Slater-Condon tensor (:func:`impurityModel.ed.model.atomic_u4`)
    this reproduces :math:`\bar U = F^0` exactly (a basis-independent trace, verified against
    :func:`getUop_from_rspt_u4`'s convention) and the standard :math:`\bar J = \frac{F^2+F^4}{14}`
    for a d-shell, :math:`\bar J = \frac{F^2}{5}` for a p-shell -- contrast :func:`dc_MLFT`'s
    ``Udd``, a different combination purpose-built for the MLFT charge-transfer model, not this
    plain average.

    Parameters
    ----------
    u4 : numpy.ndarray, shape (n, n, n, n)
        Coulomb tensor, spherical spin-collinear basis (see above). ``n`` must be even.
    tol : float, optional
        The cross-spin exchange elements (``<m up, m' down|V|m' down, m up>``) must vanish for
        a spin-collinear basis; if any exceeds this magnitude, the basis is not collinear
        (e.g. a spin-orbit-coupled :math:`(j, m_j)` basis) and a :class:`ValueError` is raised.

    Returns
    -------
    (U, J) : tuple of float
        The average Coulomb repulsion and average exchange.

    Raises
    ------
    ValueError
        If ``u4``'s spin-orbital dimension is odd, or the cross-spin exchange is non-negligible.
    """
    n = u4.shape[0]
    if n % 2 != 0:
        raise ValueError(f"uj_from_u4 requires an even spin-orbital dimension, got {n}.")
    n_orb = n // 2
    down = np.arange(n_orb)
    up = np.arange(n_orb, n)

    cross_exchange = u4[down[:, None], up[None, :], up[None, :], down[:, None]]
    max_cross = float(np.max(np.abs(cross_exchange)))
    if max_cross > tol:
        raise ValueError(
            f"u4 has non-negligible cross-spin exchange (max |element| = {max_cross:.3e} > "
            f"tol={tol}); the basis is not spin-collinear (e.g. spin-orbit-coupled). Pass "
            "explicit u=/j= instead of deriving them from u4."
        )

    direct = u4[down[:, None], up[None, :], down[:, None], up[None, :]]
    U = float(np.real(np.mean(direct)))

    if n_orb == 1:
        J = 0.0
    else:
        off_diag = ~np.eye(n_orb, dtype=bool)
        exchange = u4[down[:, None], down[None, :], down[None, :], down[:, None]]
        J = float(np.real(U - np.mean(direct[off_diag]) + np.mean(exchange[off_diag])))
    return U, J


def get_spherical_2_cubic_matrix(spinpol=False, l=2):
    r"""
    Return unitary ndarray for transforming from spherical to cubic harmonics.

    Parameters
    ----------
    spinpol : boolean
        If transformation involves spin.
    l : integer
        Angular momentum number. p: l=1, d: l=2.

    Returns
    -------
    u : (M,M) ndarray
        The unitary matrix from spherical to cubic harmonics.

    Notes
    -----
    Element :math:`u_{i,j}` represents the contribution of spherical
    harmonics :math:`i` to the cubic harmonic :math:`j`:

    .. math:: \lvert l_j \rangle  = \sum_{i=0}^4 u_{d,(i,j)}
        \lvert Y_{d,i} \rangle.

    """
    if l == 1:
        # u = np.zeros((3,3),dtype=complex)
        u = np.zeros((3, 3), dtype=complex)
        u[0, 0] = 1j / np.sqrt(2)
        u[2, 0] = 1j / np.sqrt(2)
        u[0, 1] = 1 / np.sqrt(2)
        u[2, 1] = -1 / np.sqrt(2)
        u[1, 2] = 1
    elif l == 2:
        # u = np.zeros((5,5),dtype=complex)
        u = np.zeros((5, 5), dtype=complex)
        u[2, 0] = 1
        u[[0, -1], 1] = 1 / np.sqrt(2)
        u[1, 2] = -1j / np.sqrt(2)
        u[-2, 2] = -1j / np.sqrt(2)
        u[1, 3] = 1 / np.sqrt(2)
        u[-2, 3] = -1 / np.sqrt(2)
        u[0, 4] = 1j / np.sqrt(2)
        u[-1, 4] = -1j / np.sqrt(2)
    if spinpol:
        n, m = np.shape(u)
        # U = np.zeros((2*n,2*m),dtype=complex)
        U = np.zeros((2 * n, 2 * m), dtype=complex)
        U[0:n, 0:m] = u
        U[n:, m:] = u
        u = U
    return u


def gauntC(k, l, m, lp, mp, prec=16):
    """
    return "nonvanishing" Gaunt coefficients of
    Coulomb interaction expansion.
    """
    c = sqrt(4 * pi / (2 * k + 1)) * (-1) ** m * gaunt(l, k, lp, -m, m - mp, mp, prec=prec)
    return float(c)


def getU(l1, m1, l2, m2, l3, m3, l4, m4, R):
    r"""
    Return Hubbard U term for four spherical harmonics functions.

    Scattering process:

    :math:`u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    * c_{l_1,m_1}^\dagger c_{l_2,m_2}^\dagger c_{l_3,m_3} c_{l_4,m_4}`.

    Parameters
    ----------
    l1 : int
        angular momentum of orbital 1
    m1 : int
        z projected angular momentum of orbital 1
    l2 : int
        angular momentum of orbital 2
    m2 : int
        z projected angular momentum of orbital 2
    l3 : int
        angular momentum of orbital 3
    m3 : int
        z projected angular momentum of orbital 3
    l4 : int
        angular momentum of orbital 4
    m4 : int
        z projected angular momentum of orbital 4
    R : list
        Slater-Condon parameters.
        Elements R[k] fullfill
        :math:`0<=k<=\textrm{min}(|l_1+l_4|,|l_2+l_3|)`.
        Note, U is nonzero if :math:`k+l_1+l_4` is an even integer
        and :math:`k+l_3+l_2` is an even integer.
        For example: if :math:`l_1=l_2=l_3=l_4=2`,
        R = [R0,R1,R2,R3,R4] and only R0,R2 and R4 will
        give nonzero contribution.

    Returns
    -------
    u - float
        Hubbard U term.
    """
    # Check if angular momentum is conserved
    if m1 + m2 == m3 + m4:
        u = 0
        for k, Rk in enumerate(R):
            u += Rk * gauntC(k, l1, m1, l4, m4) * gauntC(k, l3, m3, l2, m2)
    else:
        u = 0
    return u


def getUop_from_rspt_u4(u4: np.ndarray) -> dict:
    r"""Convert a 4-index U matrix in RSPt's convention to an operator dictionary.

    RSPt stores the Coulomb tensor in physicists' notation,
    :math:`u4[i,j,k,l] = \langle ij|V|kl \rangle` with bra/ket pairs (i,k) and
    (j,l), corresponding to the operator

    .. math:: \hat U = \frac{1}{2} \sum_{ijkl} u4[i,j,k,l]\,
        c^\dagger_i c^\dagger_j c_l c_k .

    Parameters
    ----------
    u4 : np.ndarray
        The 4D Coulomb interaction tensor, in RSPt's index order.

    Returns
    -------
    uDict : dict
        The converted operator dictionary.
    """
    uDict = {}
    for i, j, k, l in itertools.product(range(u4.shape[0]), range(u4.shape[1]), range(u4.shape[2]), range(u4.shape[3])):
        u = u4[i, j, k, l]
        if abs(u) > 1e-10:
            proccess = (
                (i, "c"),
                (j, "c"),
                (l, "a"),
                (k, "a"),
            )
            uDict[proccess] = u / 2
    return uDict


def getUop(l1, l2, l3, l4, R):
    r"""
    Return U operator.

    Scattering processes:
    :math:`1/2 \sum_{m_1,m_2,m_3,m_4} u_{l_1,m_1,l_2,m_2,l_3,m_3,l_4,m_4}
    * \sum_{s,sp} c_{l_1, s, m_1}^\dagger c_{l_2, sp, m_2}^\dagger
    c_{l_3, sp, m_3} c_{l_4, s, m_4}`.

    Spin polarization is considered, thus basis: (l, s, m),
    where :math:`s \in \{0, 1 \}` and these indices respectively
    corresponds to the physical values
    :math:`\{-\frac{1}{2},\frac{1}{2} \}`.

    Returns
    -------
    uDict : dict
        Elements of the form:
        ((sorb1,'c'),(sorb2,'c'),(sorb3,'a'),(sorb4,'a')) : u/2
        where sorb1 is a superindex of (l, s, m).

    """
    uDict = {}
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            for m3 in range(-l3, l3 + 1):
                for m4 in range(-l4, l4 + 1):
                    u = getU(l1, m1, l2, m2, l3, m3, l4, m4, R)
                    if u != 0:
                        for s in range(2):
                            for sp in range(2):
                                proccess = (
                                    ((l1, s, m1), "c"),
                                    ((l2, sp, m2), "c"),
                                    ((l3, sp, m3), "a"),
                                    ((l4, s, m4), "a"),
                                )
                                # Pauli exclusion principle
                                if not (s == sp and ((l1, m1) == (l2, m2) or (l3, m3) == (l4, m4))):
                                    uDict[proccess] = u / 2.0
    return uDict


def intra_orbital_coefficients(l: int):
    """
    Calculates the same shell spherical average coefficients for
    the F^k terms, k > 0. Returns only the even numbered coefficients.
    """
    coefficients = []
    for k in range(2, 2 * l + 1, 2):
        wj = wigner_3j(l, k, l, 0, 0, 0)
        coefficients.append(Rational((2 * l + 1), (4 * l + 1)) * (wj**2))
    return tuple(coefficients)


def inter_orbital_coefficients(lv: int, lc: int):
    """
    Calculates the inter shell spherical average coefficients for
    the G^k terms, k > 0. Returns only the even numbered coefficients.
    """
    start_k = abs(lv - lc)
    stop_k = lc + lv
    coefficients = []
    for k in range(start_k, stop_k + 1):
        if (lc + lv + k) % 2 == 0:
            wj = wigner_3j(lc, k, lv, 0, 0, 0)
            coefficients.append(Rational(1, 2) * wj**2)
    return tuple(coefficients)


def slater_condon_Uop(Fvv: tuple, Fcc: Optional[tuple], Fcv: Optional[tuple], Gcv: Optional[tuple]):
    assert len(Fvv) % 2 == 1, "Fvv must have 2lv+1 components"
    if Fcc is not None:
        assert len(Fcc) % 2 == 1, "Fpp must have 2lc+1 components"
    if Fcv is not None:
        assert len(Fcv) == len(Fcc), "Fpp and Fpd must have the same number of components"
    if Gcv is not None:
        assert len(Gcv) == len(Fcc) + 1, "Gpd must have 2lc+2 components"
    lv = (len(Fvv) - 1) // 2
    lc = (len(Fcc) - 1) // 2
    U = getUop(l1=lv, l2=lv, l3=lv, l4=lv, R=Fvv)
    if Fcc is not None:
        U = addOps([U, getUop(l1=lc, l2=lc, l3=lc, l4=lc, R=Fcc)])
    if Fcv is not None:
        U = addOps([U, getUop(l1=lc, l2=lv, l3=lv, l4=lc, R=Fcv)])
        U = addOps([U, getUop(l1=lv, l2=lc, l3=lc, l4=lv, R=Fcv)])
    if Gcv is not None:
        U = addOps([U, getUop(l1=lc, l2=lv, l3=lc, l4=lv, R=Gcv)])
        U = addOps([U, getUop(l1=lv, l2=lc, l3=lv, l4=lc, R=Gcv)])
    return U


def get2p3dSlaterCondonUop(Fdd=(9, 0, 8, 0, 6), Fpp=(20, 0, 8), Fpd=(10, 0, 8), Gpd=(0, 3, 0, 2)):
    """
    Return a 2p-3d U operator containing a sum of
    different Slater-Condon proccesses.

    Parameters
    ----------
    Fdd : tuple
    Fpp : tuple
    Fpd : tuple
    Gpd : tuple

    """
    # Calculate F_dd^{0,2,4}
    FddOp = {}
    if Fdd is not None:
        FddOp = getUop(l1=2, l2=2, l3=2, l4=2, R=Fdd)
    # Calculate F_pp^{0,2}
    FppOp = {}
    if Fpp is not None:
        FppOp = getUop(l1=1, l2=1, l3=1, l4=1, R=Fpp)
    # Calculate F_pd^{0,2}
    FpdOp = {}
    if Fpd is not None:
        FpdOp1 = getUop(l1=1, l2=2, l3=2, l4=1, R=Fpd)
        FpdOp2 = getUop(l1=2, l2=1, l3=1, l4=2, R=Fpd)
        FpdOp = addOps([FpdOp1, FpdOp2])
    # Calculate G_pd^{1,3}
    GpdOp = {}
    if Gpd is not None:
        GpdOp1 = getUop(l1=1, l2=2, l3=1, l4=2, R=Gpd)
        GpdOp2 = getUop(l1=2, l2=1, l3=2, l4=1, R=Gpd)
        GpdOp = addOps([GpdOp1, GpdOp2])
    # Add operators
    uOp = addOps([FddOp, FppOp, FpdOp, GpdOp])
    return uOp


def getSOCop(xi, l=2):
    """
    Return SOC operator for one l-shell.

    Parameters
    ----------
    xi : float
        Spin-orbit coupling constant.
    l : int, default 2
        Angular momentum quantum number.

    Returns
    -------
    uDict : dict
        Elements of the form:
        (((l, s1, m1),'c'), ((l, s2, m2),'a')) : h_value
        where (l, s, m) is the state.
    """
    opDict = {}
    for m in range(-l, l + 1):
        for s in range(2):
            value = xi * m * (1 / 2.0 if s == 1 else -1 / 2.0)
            opDict[(((l, s, m), "c"), ((l, s, m), "a"))] = value
    for m in range(-l, l):
        value = xi / 2.0 * sqrt((l - m) * (l + m + 1))
        opDict[(((l, 1, m), "c"), ((l, 0, m + 1), "a"))] = value
        opDict[(((l, 0, m + 1), "c"), ((l, 1, m), "a"))] = value
    return opDict


def gethHfieldop(hx, hy, hz, l=2):
    """
    Return magnetic field operator for one l-shell.

    Parameters
    ----------
    hx : float
        Magnetic field x-component.
    hy : float
        Magnetic field y-component.
    hz : float
        Magnetic field z-component.
    l : int, default 2
        Angular momentum quantum number.

    Returns
    -------
    hHfieldOperator : dict
        Elements of the form:
        (((l, s1, m1),'c'), ((l, s2, m2),'a')) : h_value
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
