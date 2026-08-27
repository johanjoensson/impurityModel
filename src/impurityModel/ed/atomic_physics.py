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


def dc_MLFT(lv, n_val_i, c, Fvv, lc=None, n_core_i=None, Fcv=None, Gcv=None):
    r"""
    Return double counting (DC) in multiplet ligand field theory.

    Shell-agnostic: the spherical-average combinations that define the average valence-valence
    and core-valence repulsions are taken from :func:`intra_orbital_coefficients` and
    :func:`inter_orbital_coefficients` rather than written out for the 3d/2p case. For
    ``lv=2, lc=1`` those coefficients are exactly ``14/441 == 2/63`` and ``(1/15, 3/70)``, so
    this reproduces the historical ``Udd``/``Upd`` expressions identically.

    Parameters
    ----------
    lv : int
        Angular momentum of the valence shell.
    n_val_i : int
        Nominal (integer) valence occupation.
    c : float
        Many-body correction to the charge transfer energy.
    Fvv : sequence of float
        Slater integrals :math:`F^k_{vv}`, ``2*lv + 1`` components.
    lc : int, optional
        Angular momentum of the core shell. ``None`` for a valence-only double counting.
    n_core_i : int, optional
        Nominal (integer) core occupation. Must be the *filled* shell,
        ``2*(2*lc + 1)`` -- the MLFT core-hole prescription is written for a full core.
    Fcv : sequence of float, optional
        Slater integrals :math:`F^k_{cv}`, ``2*lc + 1`` components.
    Gcv : sequence of float, optional
        Slater integrals :math:`G^k_{cv}`, ``2*lc + 2`` components.

    Returns
    -------
    dict
        ``{lv: dc_valence}``, or ``{lv: dc_valence, lc: dc_core}`` when a core shell is given.

    Notes
    -----
    The `c` parameter is related to the charge-transfer
    energy :math:`\Delta_{CT}` by:

    .. math:: \Delta_{CT} = (e_d-e_b) + c.

    """
    if int(n_val_i) != n_val_i:
        raise ValueError(f"valence (l={lv}) occupation should be an integer, got {n_val_i}")
    if n_core_i is not None and int(n_core_i) != n_core_i:
        raise ValueError(f"core (l={lc}) occupation should be an integer, got {n_core_i}")
    if len(Fvv) != 2 * lv + 1:
        raise ValueError(f"Fvv has {len(Fvv)} components, but l_valence={lv} requires {2 * lv + 1}.")

    # Average valence-valence repulsion: F^0 minus the spherical average of the higher F^k.
    Uvv = Fvv[0] - sum(
        float(coeff) * Fvv[k] for k, coeff in zip(intra_orbital_k_values(lv), intra_orbital_coefficients(lv))
    )

    core_given = (lc, n_core_i, Fcv, Gcv)
    if all(x is None for x in core_given):
        return {lv: Uvv * n_val_i - c}
    if any(x is None for x in core_given):
        raise ValueError(
            "double counting input wrong: a core shell needs all of lc, n_core_i, Fcv and Gcv, "
            f"got lc={lc}, n_core_i={n_core_i}, "
            f"Fcv={'given' if Fcv is not None else None}, Gcv={'given' if Gcv is not None else None}."
        )
    if n_core_i != 2 * (2 * lc + 1):
        raise ValueError(
            f"double counting input wrong: the MLFT core-hole prescription assumes a filled "
            f"core shell, i.e. n_core_i = {2 * (2 * lc + 1)} for l={lc}, got {n_core_i}."
        )
    if len(Fcv) != direct_array_length(lv, lc):
        raise ValueError(
            f"Fcv has {len(Fcv)} components, but l_valence={lv} / l_core={lc} "
            f"requires {direct_array_length(lv, lc)}."
        )
    if len(Gcv) != exchange_array_length(lv, lc):
        raise ValueError(
            f"Gcv has {len(Gcv)} components, but l_valence={lv} / l_core={lc} "
            f"requires {exchange_array_length(lv, lc)}."
        )

    # Average core-valence repulsion: F^0_cv minus the spherical average of the exchange G^k.
    Ucv = Fcv[0] - sum(
        float(coeff) * Gcv[k] for k, coeff in zip(inter_orbital_k_values(lv, lc), inter_orbital_coefficients(lv, lc))
    )
    return {
        lv: Uvv * n_val_i + Ucv * n_core_i - c,
        lc: Ucv * (n_val_i + 1) - c,
    }


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


#: Octahedral (O_h) level structure of a single ``l`` shell.
#:
#: Each entry is ``(irrep, degeneracy, weights)`` in the **column order** of
#: :func:`get_spherical_2_cubic_matrix`, so a caller can fill a diagonal matrix from it
#: without re-hardcoding that order. ``weights`` holds one traceless weight per independent
#: octahedral invariant (rank 4, then rank 6), normalised so that the highest and lowest
#: level of that invariant differ by exactly one -- which makes the rank-4 weight of a d
#: shell ``(3/5, -2/5)``, i.e. the historical ``e_deltaO_imp`` = 10Dq.
#:
#: The table is derived, not asserted: ``test_atomic_physics`` rebuilds every entry by
#: diagonalising the octahedral point-charge crystal field built from ``gauntC``.
#:
#: s and p have a single level (a1g, t1u), so no splitting parameter exists for them. l=3
#: has two independent invariants, which is why a single ``e_deltaO_imp`` cannot describe an
#: f shell. From l=5 up, irreps repeat (t1u twice) and the point group alone no longer fixes
#: the basis, so no such table can exist.
OCTAHEDRAL_LEVELS = {
    0: (("a1g", 1, ()),),
    1: (("t1u", 3, ()),),
    2: (("eg", 2, (3 / 5,)), ("t2g", 3, (-2 / 5,))),
    3: (("t1u", 3, (1 / 3, 5 / 21)), ("t2u", 3, (-1 / 9, -3 / 7)), ("a2u", 1, (-2 / 3, 4 / 7))),
}

#: Angular momenta :func:`get_spherical_2_cubic_matrix` and :data:`OCTAHEDRAL_LEVELS` cover.
CUBIC_HARMONIC_SHELLS = tuple(sorted(OCTAHEDRAL_LEVELS))


def octahedral_level_structure(l):
    """Return the O_h level structure of an ``l`` shell, or raise saying why there is none.

    Parameters
    ----------
    l : int
        Angular momentum of the shell.

    Returns
    -------
    tuple
        ``(irrep, degeneracy, weights)`` per level, in the column order of
        :func:`get_spherical_2_cubic_matrix`. See :data:`OCTAHEDRAL_LEVELS`.

    Raises
    ------
    ValueError
        If ``l`` has no tabulated octahedral level structure.
    """
    try:
        return OCTAHEDRAL_LEVELS[l]
    except KeyError:
        raise ValueError(
            f"No octahedral level structure for l={l}. Implemented: "
            f"{', '.join(f'l={k}' for k in CUBIC_HARMONIC_SHELLS)} (s, p, d, f). From l=5 up "
            "an irrep appears more than once in the O_h decomposition, so the point group "
            "alone does not fix the symmetry-adapted basis and no cubic parametrisation is "
            "well defined; supply a .h0 file instead."
        ) from None


def n_octahedral_splittings(l):
    """Number of independent octahedral splitting parameters an ``l`` shell has.

    One fewer than its number of distinct levels: 0 for s and p, 1 for d (10Dq), 2 for f.
    """
    return len(octahedral_level_structure(l)[0][2])


def get_spherical_2_cubic_matrix(spinpol=False, l=2):
    r"""
    Return unitary ndarray for transforming from spherical to cubic harmonics.

    The columns are the octahedral symmetry-adapted (cubic harmonic) orbitals, **grouped and
    ordered by O_h level** to match :func:`octahedral_level_structure`: for a d shell
    ``(e_g, e_g, t_2g, t_2g, t_2g)``, for an f shell ``(t_1u x3, t_2u x3, a_2u)``.

    Parameters
    ----------
    spinpol : boolean
        If transformation involves spin.
    l : integer
        Angular momentum number. s: l=0, p: l=1, d: l=2, f: l=3.

    Returns
    -------
    u : (M,M) ndarray
        The unitary matrix from spherical to cubic harmonics.

    Raises
    ------
    ValueError
        If ``l`` is outside :data:`CUBIC_HARMONIC_SHELLS`. It used to fall off the end of the
        ``if``/``elif`` chain and raise ``UnboundLocalError`` on an undefined ``u``.

    Notes
    -----
    Element :math:`u_{i,j}` represents the contribution of spherical
    harmonics :math:`i` to the cubic harmonic :math:`j`:

    .. math:: \lvert l_j \rangle  = \sum_{i=0}^4 u_{d,(i,j)}
        \lvert Y_{d,i} \rangle.

    The f-shell columns are the O_h-adapted set, which is *not* the plain real-harmonic set:
    :math:`f_{x^3}` mixes :math:`f_{xz^2}` and :math:`f_{x(x^2-3y^2)}` with weights
    :math:`\sqrt{3/8}` and :math:`-\sqrt{5/8}`. Using the unmixed real harmonics would leave
    the rank-4 crystal field non-diagonal, silently coupling t_1u to t_2u.

    """
    if l == 0:
        u = np.ones((1, 1), dtype=complex)
    elif l == 1:
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
    elif l == 3:
        # Built from the real harmonics rather than written out as 28 literals: the two
        # mixing angles are the whole content of the f case, and they are visible here.
        def cos_m(m):
            v = np.zeros(7, dtype=complex)
            v[3 - m] = 1 / np.sqrt(2)
            v[3 + m] = (-1) ** m / np.sqrt(2)
            return v

        def sin_m(m):
            v = np.zeros(7, dtype=complex)
            v[3 - m] = 1j / np.sqrt(2)
            v[3 + m] = -1j * (-1) ** m / np.sqrt(2)
            return v

        f_z3 = np.zeros(7, dtype=complex)
        f_z3[3] = 1
        a, b = np.sqrt(3 / 8), np.sqrt(5 / 8)
        u = np.array(
            [
                f_z3,  # t1u  f_z^3
                a * cos_m(1) - b * cos_m(3),  # t1u  f_x^3
                a * sin_m(1) + b * sin_m(3),  # t1u  f_y^3
                cos_m(2),  # t2u  f_z(x^2-y^2)
                b * cos_m(1) + a * cos_m(3),  # t2u  f_x(y^2-z^2)
                b * sin_m(1) - a * sin_m(3),  # t2u  f_y(z^2-x^2)
                sin_m(2),  # a2u  f_xyz
            ]
        ).T
    else:
        raise ValueError(
            f"get_spherical_2_cubic_matrix has no cubic harmonics for l={l}. Implemented: "
            f"{', '.join(f'l={k}' for k in CUBIC_HARMONIC_SHELLS)} (s, p, d, f)."
        )
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


def intra_orbital_k_values(l: int):
    """Return the ``k`` for which a same-shell :math:`F^k` contributes, excluding ``k = 0``."""
    return tuple(range(2, 2 * l + 1, 2))


def inter_orbital_k_values(lv: int, lc: int):
    """Return the ``k`` for which an inter-shell :math:`G^k` contributes."""
    return tuple(k for k in range(abs(lv - lc), lc + lv + 1) if (lc + lv + k) % 2 == 0)


def intra_orbital_coefficients(l: int):
    """
    Calculates the same shell spherical average coefficients for
    the F^k terms, k > 0. Returns one coefficient per entry of
    :func:`intra_orbital_k_values`, in the same order.
    """
    return tuple(
        Rational((2 * l + 1), (4 * l + 1)) * (wigner_3j(l, k, l, 0, 0, 0) ** 2) for k in intra_orbital_k_values(l)
    )


def inter_orbital_coefficients(lv: int, lc: int):
    """
    Calculates the inter shell spherical average coefficients for
    the G^k terms, k > 0. Returns one coefficient per entry of
    :func:`inter_orbital_k_values`, in the same order.
    """
    return tuple(Rational(1, 2) * wigner_3j(lc, k, lv, 0, 0, 0) ** 2 for k in inter_orbital_k_values(lv, lc))


def exchange_array_length(lv, lc):
    """Number of ``G^k_cv`` components a core/valence pair needs.

    The exchange integrals run over ``|lv - lc| <= k <= lv + lc``, and the arrays are indexed
    by ``k`` itself, so the array must reach ``k = lv + lc``.

    The historical spelling was ``2*lc + 2``, which is the same number **only** when
    ``lv = lc + 1`` -- true of every dipole-allowed edge, and so of every input this package
    had ever seen. It is wrong the moment the two shells are not one apart: a 1s core under a
    3d valence shell needs ``G^2``, which ``2*lc + 2 = 2`` does not reach, and the general
    assembler raised ``IndexError`` on it. Shell roles are meaningful for any pair (they say
    which shell carries the core SOC and which way the core-valence Coulomb runs); only a
    transition *operator* is bound by |lv - lc| = 1.
    """
    return lv + lc + 1


def direct_array_length(lv, lc):
    """Number of ``F^k_cv`` components a core/valence pair needs.

    The direct integrals run over even ``k`` up to ``2*min(lv, lc)``, so for a core shell
    below the valence shell this is the historical ``2*lc + 1`` at every pair, not just the
    dipole-allowed ones.
    """
    return 2 * min(lv, lc) + 1


def _check_slater_length(name, values, expected, lv, lc):
    """Cross-check one Slater-Condon array against the length its shells imply."""
    if values is not None and len(values) != expected:
        raise ValueError(
            f"{name} has {len(values)} components, but l_valence={lv} / l_core={lc} " f"requires {expected}."
        )


def slater_condon_Uop(lv, lc, Fvv, Fcc=None, Fcv=None, Gcv=None):
    r"""Return the Coulomb operator for a valence shell plus an optional core shell.

    Shell-agnostic replacement for :func:`get2p3dSlaterCondonUop`: the angular momenta are
    passed in explicitly rather than inferred from the array lengths. Inference would be
    ambiguous exactly where it matters -- an omitted (``None``) array carries no length, and
    a zero-filled placeholder carries the *wrong* one -- so ``lv``/``lc`` are the single
    source of truth and the lengths are only cross-checked against them.

    Parameters
    ----------
    lv : int
        Angular momentum of the valence shell.
    lc : int or None
        Angular momentum of the core shell, or ``None`` when there is no core shell (in
        which case ``Fcc``, ``Fcv`` and ``Gcv`` must all be ``None``).
    Fvv : sequence of float
        Valence-valence direct integrals :math:`F^k_{vv}`, ``2*lv + 1`` components.
    Fcc : sequence of float, optional
        Core-core direct integrals :math:`F^k_{cc}`, ``2*lc + 1`` components.
    Fcv : sequence of float, optional
        Core-valence direct integrals :math:`F^k_{cv}`, ``2*lc + 1`` components.
    Gcv : sequence of float, optional
        Core-valence exchange integrals :math:`G^k_{cv}`, ``2*lc + 2`` components.

    Returns
    -------
    dict
        The Coulomb operator, in ``(l, s, m)`` label form.

    Raises
    ------
    ValueError
        If an array's length disagrees with the angular momenta, or if a core-shell array
        is given without an ``lc``.
    """
    _check_slater_length("Fvv", Fvv, 2 * lv + 1, lv, lc)
    if lc is None:
        for name, values in (("Fcc", Fcc), ("Fcv", Fcv), ("Gcv", Gcv)):
            if values is not None:
                raise ValueError(f"{name} was given, but lc is None: there is no core shell to put it on.")
    else:
        _check_slater_length("Fcc", Fcc, 2 * lc + 1, lv, lc)
        _check_slater_length("Fcv", Fcv, direct_array_length(lv, lc), lv, lc)
        _check_slater_length("Gcv", Gcv, exchange_array_length(lv, lc), lv, lc)

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
