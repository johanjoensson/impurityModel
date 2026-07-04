"""
Ground-state and thermal observables: occupations and angular-momentum
expectation values from single-particle density matrices in the spherical
basis, many-body spin/orbital/Casimir operators, and (thermally averaged)
expectation-value reporting for degenerate manifolds.
"""

from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.block_structure import get_equivalent_blocks
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, inner
from impurityModel.ed.utils import rotate_matrix


def print_expectation_values(
    rhos, es, rot_to_spherical, block_structure, s_values=None, l_values=None, j_values=None, sisb_values=None
):
    """
    print several expectation values, e.g. E, N, L^2.

    If ``s_values`` / ``l_values`` / ``j_values`` are given (one impurity ``S`` / ``L``
    / ``J`` quantum number per eigenstate, e.g. from :func:`manifold_observable_values`
    with :func:`make_impurity_casimir_operators` + :func:`casimir_to_quantum_number`),
    the corresponding columns are appended. When all are ``None`` the output is
    identical to before (used when the eigenstates are not available, e.g. on non-root
    ranks).
    """
    orb_offset = min(orb for block in block_structure.blocks for orb in block)
    equivalent_blocks = get_equivalent_blocks(block_structure)
    print(f"E0 = {es[0]:9.6f}")
    block_N_string = [f"N({','.join(f'{b}' for b in blocks)})" for blocks in equivalent_blocks]
    # Each block-occupation column is right-aligned to a width that fits both its header
    # and the 7-8 char ``.5f`` value below it, so header and numbers line up.
    block_N_widths = [max(len(Ns), 8) for Ns in block_N_string]
    block_N_string_formatted = [f"{Ns:>{w}s}" for Ns, w in zip(block_N_string, block_N_widths)]
    extra = [
        (name, vals)
        for name, vals in (("S", s_values), ("L", l_values), ("J", j_values), ("Si.Sb", sisb_values))
        if vals is not None
    ]
    # Lz/Sz/L.S and the S/L/J/Si.Sb columns are printed with the space-flag format
    # ``{x: 8.6f}``, which is 9 characters wide (the sign column sits on top of the 8),
    # so their headers must be 9 wide to line up with the numbers below.
    extra_header = "".join(f"  {name:>9s}" for name, _ in extra)
    print(
        f"{'i':>3s}  {'E-E0':>11s}  {'N':>8s}  {'N(Dn)':>8s}  {'N(Up)':>8s}  {'  '.join(block_N_string_formatted)}  {'Lz':>9s}  {'Sz':>9s}  {'L.S':>9s}{extra_header}"
    )
    for i, (e, rho) in enumerate(zip(es - es[0], rhos)):
        block_occs = [
            np.sum(np.diag(rho)[list(orb - orb_offset for block in blocks for orb in block_structure.blocks[block])])
            for blocks in equivalent_blocks
        ]
        block_occ_string_formatted = ["" for _ in block_occs]
        for ib, b_occ in enumerate(block_occs):
            block_occ_string_formatted[ib] = f"{np.real(b_occ):>{block_N_widths[ib]}.5f}"
        rho_spherical = rotate_matrix(rho, rot_to_spherical)
        N, Ndn, Nup = get_occupations_from_rho_spherical(rho_spherical)
        Lz = get_Lz_from_rho_spherical(rho_spherical)
        Sz = get_Sz_from_rho_spherical(rho_spherical)
        LS = get_LS_from_rho_spherical(rho_spherical)
        extra_fields = "".join(f"  {vals[i]: 8.6f}" for _, vals in extra)
        print(
            f"{i:>3d}  {e:11.8f}  {N:8.5f}  {Ndn:8.5f}  {Nup:8.5f}  {'  '.join(block_occ_string_formatted)}  {Lz: 8.6f}  {Sz: 8.6f}  {LS: 8.6f}{extra_fields}"
        )
    print("\n")


def get_occupations_from_rho_spherical(rho):
    """
    Calculate the (spin polarized) occupation from the density matrix.
    """
    n_orbs = rho.shape[0]
    return (
        np.real(np.trace(rho)),
        np.real(np.trace(rho[: n_orbs // 2, : n_orbs // 2])),
        np.real(np.trace(rho[n_orbs // 2 :, n_orbs // 2 :])),
    )


def get_Lz_from_rho_spherical(rho: np.ndarray, l: Optional[int] = None) -> float:
    """Calculate the expectation value of L_z from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int, optional
        The orbital angular momentum quantum number. If None, it is calculated from rho's shape.

    Returns
    -------
    float
        The expectation value <L_z>.
    """
    if l is None:
        l = (rho.shape[0] // 2 - 1) // 2
    return np.real(
        sum(ml * (rho[i, i] + rho[i + (2 * l + 1), i + (2 * l + 1)]) for i, ml in enumerate(range(-l, l + 1)))
    )


def get_Lplus_from_rho_spherical(rho: np.ndarray, l: int) -> complex:
    """Calculate the expectation value of L_+ from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int
        The orbital angular momentum quantum number.

    Returns
    -------
    complex
        The expectation value <L_+>.
    """
    # L+ |l, ml> = sqrt(l*(l+1) - ml*(ml+1))|l, ml+1>
    llp1 = l * (l + 1)
    #   L+    |2, -2>,  |2, -1>, |2,  0>, |2,  1>, |2,  2>
    # <2, -2|    0         0        0        0        0
    # <2, -1| sqrt(8)      0        0        0        0
    # <2,  0|    0      sqrt(6)     0        0        0
    # <2,  1|    0         0     sqrt(6)     0        0
    # <2,  2|    0         0        0     sqrt(8)     0
    Lplus = np.diag([np.sqrt(llp1 - ml * (ml + 1)) for ml in range(-l, l)], k=-1)
    return np.trace(
        rho @ np.block([[Lplus, np.zeros((2 * l + 1, 2 * l + 1))], [np.zeros((2 * l + 1, 2 * l + 1)), Lplus]])
    )


def get_Sminus_from_rho_spherical(rho: np.ndarray, l: int, s: float = 0.5) -> complex:
    """Calculate the expectation value of ``S_-`` from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int
        The orbital angular momentum quantum number.
    s : float, default 0.5
        The spin quantum number.

    Returns
    -------
    complex
        The expectation value ``<S_->``.
    """
    # S+ |s, ms> = sqrt(s*(s+1) - ms*(ms+1))|s, ms+1>
    ssp1 = s * (s + 1)
    ms = +1 / 2
    #   S-      |1/2,-1/2>,  |1/2, 1/2>
    # <1/2,-1/2|    0            1
    # <1/2, 1/2|    0            0
    # S- = [[0   S-],
    #        0   0 ]]
    Sminus = np.diag(np.repeat(np.sqrt(ssp1 - ms * (ms - 1)), 2 * l), k=1)
    return np.trace(
        rho
        @ np.block(
            [
                [np.zeros((2 * l + 1, 2 * l + 1)), Sminus],
                [np.zeros((2 * l + 1, 2 * l + 1)), np.zeros((2 * l + 1, 2 * l + 1))],
            ]
        )
    )


def get_Lminus_from_rho_spherical(rho: np.ndarray, l: int) -> complex:
    """Calculate the expectation value of ``L_-`` from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int
        The orbital angular momentum quantum number.

    Returns
    -------
    complex
        The expectation value ``<L_->``.
    """
    # L- |l, ml> = sqrt(l*(l+1) - ml*(ml-1))|l, ml-1>
    llp1 = l * (l + 1)
    #   L+    |2, -2>,  |2, -1>, |2,  0>, |2,  1>, |2,  2>
    # <2, -2|    0      sqrt(4)     0        0        0
    # <2, -1|    0         0     sqrt(6)     0        0
    # <2,  0|    0         0        0     sqrt(6)     0
    # <2,  1|    0         0        0        0     sqrt(4)
    # <2,  2|    0         0        0        0        0
    Lminus = np.diag([np.sqrt(llp1 - ml * (ml - 1)) for ml in range(-l + 1, l + 1)], k=1)
    return np.trace(
        rho @ np.block([[Lminus, np.zeros((2 * l + 1, 2 * l + 1))], [np.zeros((2 * l + 1, 2 * l + 1)), Lminus]])
    )


def get_Splus_from_rho_spherical(rho: np.ndarray, l: int, s: float = 0.5) -> complex:
    """Calculate the expectation value of S_+ from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int
        The orbital angular momentum quantum number.
    s : float, default 0.5
        The spin quantum number.

    Returns
    -------
    complex
        The expectation value <S_+>.
    """
    # S+ |s, ms> = sqrt(s*(s+1) - ms*(ms+1))|s, ms+1>
    ssp1 = s * (s + 1)
    ms = -1 / 2
    #   S+      |1/2,-1/2>,  |1/2, 1/2>
    # <1/2,-1/2|    0            0
    # <1/2, 1/2|    1            0
    # S+ = [[0   0],
    #        S+  0]]
    Splus = np.diag(np.repeat(np.sqrt(ssp1 - ms * (ms + 1)), 2 * l), k=-1)
    return np.trace(
        rho
        @ np.block(
            [
                [np.zeros((2 * l + 1, 2 * l + 1)), np.zeros((2 * l + 1, 2 * l + 1))],
                [Splus, np.zeros((2 * l + 1, 2 * l + 1))],
            ]
        )
    )


def get_Sz_from_rho_spherical(rho: np.ndarray, l: Optional[int] = None) -> float:
    """Calculate the expectation value of S_z from the density matrix in spherical basis.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix.
    l : int, optional
        The orbital angular momentum quantum number. If None, it is calculated from rho's shape.

    Returns
    -------
    float
        The expectation value <S_z>.
    """
    if l is None:
        l = (rho.shape[0] // 2 - 1) // 2
    return 1 / 2 * np.real(sum(-rho[i, i] + rho[i + (2 * l + 1), i + (2 * l + 1)] for i in range(2 * l + 1)))


def _single_particle_lsj_matrices(l):
    r"""Single-particle ``L`` and ``S`` operator matrices in the spherical basis.

    Layout: a ``2*(2l+1)`` space whose first ``2l+1`` orbitals are spin-down
    (:math:`m_s=-1/2`, ``ml=-l..l``) and the next ``2l+1`` are spin-up — matching the
    ``get_*_from_rho_spherical`` helpers.

    Returns
    -------
    (lz, lplus, lminus, sz, splus, sminus) : tuple of np.ndarray
        Each of shape ``(2*(2l+1), 2*(2l+1))``.
    """
    n = 2 * l + 1
    mls = np.arange(-l, l + 1)
    llp1 = l * (l + 1)
    zeros = np.zeros((n, n))
    eye = np.eye(n)
    lz = np.diag(np.concatenate((mls, mls)).astype(float)).astype(complex)
    sz = np.diag(np.concatenate((-0.5 * np.ones(n), 0.5 * np.ones(n)))).astype(complex)
    lplus_block = np.diag([np.sqrt(llp1 - ml * (ml + 1)) for ml in mls[:-1]], k=-1)
    lplus = np.block([[lplus_block, zeros], [zeros, lplus_block]]).astype(complex)
    lminus = lplus.conj().T
    splus = np.block([[zeros, zeros], [eye, zeros]]).astype(complex)
    sminus = splus.conj().T
    return lz, lplus, lminus, sz, splus, sminus


def get_LS_from_rho_spherical(rho: np.ndarray, l: Optional[int] = None) -> float:
    r"""Calculate the expectation value of the one-body spin-orbit coupling
    :math:`\langle \mathbf{L}\cdot\mathbf{S}\rangle` from the density matrix in the
    spherical basis.

    :math:`\mathbf{L}\cdot\mathbf{S}` is a one-body operator in the single-particle
    space, so its expectation value is the contraction
    :math:`\langle \mathbf{L}\cdot\mathbf{S}\rangle = \mathrm{Tr}(\rho\, (l\cdot s))`,
    where :math:`l\cdot s = l_z s_z + \tfrac{1}{2}(l_+ s_- + l_- s_+)` is the
    single-particle spin-orbit matrix. No many-body solver is required.

    The spherical basis layout matches the other ``get_*_from_rho_spherical``
    helpers: a ``2*(2l+1)`` matrix whose first ``2l+1`` rows/columns are the
    spin-down (:math:`m_s=-1/2`) orbitals ``ml = -l..l`` and whose last ``2l+1``
    are the spin-up (:math:`m_s=+1/2`) orbitals.

    Parameters
    ----------
    rho : np.ndarray
        The density matrix in the spherical basis.
    l : int, optional
        The orbital angular momentum quantum number. If None, it is calculated
        from rho's shape.

    Returns
    -------
    float
        The expectation value :math:`\langle \mathbf{L}\cdot\mathbf{S}\rangle`.
    """
    if l is None:
        l = (rho.shape[0] // 2 - 1) // 2
    n = 2 * l + 1
    # Contract against the leading 2*(2l+1) sub-block, matching the index-based
    # get_Lz/get_Sz helpers (robust when rho is not exactly spin-doubled, e.g. an
    # odd-sized block).
    rho = rho[: 2 * n, : 2 * n]
    lz, lplus, lminus, sz, splus, sminus = _single_particle_lsj_matrices(l)
    ls = lz @ sz + 0.5 * (lplus @ sminus + lminus @ splus)
    return np.real(np.trace(rho @ ls))


def make_spin_operators(spin_pairs):
    r"""Build the one-body spin ladder/Cartan operators for a set of spatial orbitals.

    Each spatial orbital contributes a spin doublet; ``spin_pairs`` lists its
    ``(dn_index, up_index)`` spin-orbital indices. The returned operators are

    .. math::
        \hat S_+ = \sum_a c^\dagger_{a\uparrow} c_{a\downarrow}, \quad
        \hat S_- = \sum_a c^\dagger_{a\downarrow} c_{a\uparrow}, \quad
        \hat S_z = \tfrac12 \sum_a (n_{a\uparrow} - n_{a\downarrow}).

    Parameters
    ----------
    spin_pairs : iterable of (int, int)
        ``(dn_index, up_index)`` spin-orbital index pairs, one per spatial orbital.

    Returns
    -------
    (ManyBodyOperator, ManyBodyOperator, ManyBodyOperator)
        The operators :math:`(\hat S_+, \hat S_-, \hat S_z)`.
    """
    s_plus, s_minus, s_z = {}, {}, {}
    for dn, up in spin_pairs:
        s_plus[((up, "c"), (dn, "a"))] = 1.0
        s_minus[((dn, "c"), (up, "a"))] = 1.0
        s_z[((up, "c"), (up, "a"))] = 0.5
        s_z[((dn, "c"), (dn, "a"))] = -0.5
    return ManyBodyOperator(s_plus), ManyBodyOperator(s_minus), ManyBodyOperator(s_z)


def make_orbital_angular_momentum_operators(channels):
    r"""Build the one-body orbital angular-momentum operators for a set of shells.

    Each *channel* is one ``(l, spin)`` block: an ordered list of the orbital
    indices for ``ml = -l, -l+1, ..., l`` at fixed spin. The returned operators are

    .. math::
        \hat L_+ = \sum c^\dagger_{m_l+1} c_{m_l}\,\sqrt{l(l+1)-m_l(m_l+1)}, \quad
        \hat L_z = \sum m_l\, n_{m_l},

    summed over every channel (both spins), and :math:`\hat L_- = \hat L_+^\dagger`.

    Parameters
    ----------
    channels : iterable of sequence[int]
        Each element lists the ``2l+1`` orbital indices ordered by ``ml`` from
        ``-l`` to ``+l`` for one spin of one ``l``-shell. ``l`` is inferred from
        the length.

    Returns
    -------
    (ManyBodyOperator, ManyBodyOperator, ManyBodyOperator)
        The operators :math:`(\hat L_+, \hat L_-, \hat L_z)`.
    """
    l_plus, l_minus, l_z = {}, {}, {}
    for indices in channels:
        indices = list(indices)
        l = (len(indices) - 1) // 2
        llp1 = l * (l + 1)
        for a, ml in enumerate(range(-l, l + 1)):
            l_z[((indices[a], "c"), (indices[a], "a"))] = float(ml)
            if ml < l:
                coeff = np.sqrt(llp1 - ml * (ml + 1))
                l_plus[((indices[a + 1], "c"), (indices[a], "a"))] = coeff
                l_minus[((indices[a], "c"), (indices[a + 1], "a"))] = coeff
    return ManyBodyOperator(l_plus), ManyBodyOperator(l_minus), ManyBodyOperator(l_z)


def make_impurity_casimir_operators(impurity_orbitals, rot_to_spherical):
    r"""Build the total impurity ``(L, S, J)`` ladder/Cartan operators in the
    **computational** basis.

    For each impurity ``l``-shell the single-particle ``L``/``S`` matrices are built in
    the spherical basis (:func:`_single_particle_lsj_matrices`, where the ``ml``/spin
    structure is explicit) and rotated to the computational basis via
    ``rot_to_spherical`` (``O_comp = R\,O_sph\,R^\dagger`` with ``R`` the
    spherical→computational rotation, matching :func:`rotate_matrix`), then summed over
    shells. This makes ``L²``/``J²``/``S²`` evaluable on states stored in the
    computational basis — the ``ml`` dependence of ``L`` is carried by the rotation, and
    the construction is robust to whatever spin ordering the computational basis uses
    (Phase 5 unblocks this for the deferred ``L²``/``J²`` reporting).

    Parameters
    ----------
    impurity_orbitals : dict
        ``Basis.impurity_orbitals`` (``partition -> list of orbital-index blocks``).
        The shell's ``l`` is inferred from the orbital count ``2*(2l+1)``.
    rot_to_spherical : np.ndarray or dict
        The spherical→computational rotation: a single ``2(2l+1)`` matrix, or a dict
        ``{partition: matrix}`` (as in ``get_spectra``).

    Returns
    -------
    (L, S, J) : tuple
        Each is ``(plus, minus, z)`` as ``ManyBodyOperator``s, ready for
        :func:`apply_casimir` / :func:`expect_casimir`. ``J = L + S``.
    """
    l_plus, l_minus, l_z = {}, {}, {}
    s_plus, s_minus, s_z = {}, {}, {}
    for partition, blocks in impurity_orbitals.items():
        orbs = [orb for block in blocks for orb in block]
        n_so = len(orbs)
        shell_l = (n_so // 2 - 1) // 2
        if 2 * (2 * shell_l + 1) != n_so:
            raise ValueError(
                f"Impurity partition {partition} has {n_so} spin-orbitals, which is not a "
                f"spin-doubled l-shell (2*(2l+1)); cannot build L/S/J operators for it."
            )
        lz_m, lp_m, lm_m, sz_m, sp_m, sm_m = _single_particle_lsj_matrices(shell_l)
        rot = rot_to_spherical[partition] if isinstance(rot_to_spherical, dict) else rot_to_spherical
        rot = np.asarray(rot, dtype=complex)
        for target, matrix in (
            (l_z, lz_m),
            (l_plus, lp_m),
            (l_minus, lm_m),
            (s_z, sz_m),
            (s_plus, sp_m),
            (s_minus, sm_m),
        ):
            computational = rot @ matrix @ rot.conj().T
            for i in range(n_so):
                for j in range(n_so):
                    if abs(computational[i, j]) > 1e-12:
                        key = ((orbs[i], "c"), (orbs[j], "a"))
                        target[key] = target.get(key, 0.0) + computational[i, j]
    l_ops = (ManyBodyOperator(l_plus), ManyBodyOperator(l_minus), ManyBodyOperator(l_z))
    s_ops = (ManyBodyOperator(s_plus), ManyBodyOperator(s_minus), ManyBodyOperator(s_z))
    j_ops = (l_ops[0] + s_ops[0], l_ops[1] + s_ops[1], l_ops[2] + s_ops[2])
    return l_ops, s_ops, j_ops


def apply_casimir(psi, j_plus, j_minus, j_z):
    r"""Apply a su(2) Casimir operator to ``psi`` and return the resulting state.

    Uses the ladder identity :math:`\hat J^2 = \hat J_- \hat J_+ + \hat J_z^2 +
    \hat J_z` (with :math:`\hat J_- = \hat J_+^\dagger`), so only the one-body
    ladder/Cartan operators are needed — no explicit two-body operator product is
    constructed. Each factor is applied sequentially to the state.

    Parameters
    ----------
    psi : ManyBodyState
        The state to act on.
    j_plus, j_minus, j_z : ManyBodyOperator
        The raising, lowering, and Cartan operators of the su(2) algebra.

    Returns
    -------
    ManyBodyState
        :math:`\hat J^2 |\psi\rangle`.
    """
    jz_psi = j_z(psi, 0)
    result = j_minus(j_plus(psi, 0), 0)
    result += j_z(jz_psi, 0)
    result += jz_psi
    return result


def apply_spin_correlation(psi, ops_a, ops_b):
    r"""Apply the spin-correlation operator :math:`\hat{\mathbf S}_A\cdot\hat{\mathbf S}_B`.

    For two **disjoint** orbital sets A and B the spin operators commute, so

    .. math::
        \hat{\mathbf S}_A\cdot\hat{\mathbf S}_B
            = \hat S^A_z \hat S^B_z
            + \tfrac12\left(\hat S^A_+ \hat S^B_- + \hat S^A_- \hat S^B_+\right),

    with no normal-ordering correction. Each factor is applied sequentially.

    Parameters
    ----------
    psi : ManyBodyState
        The state to act on.
    ops_a, ops_b : (ManyBodyOperator, ManyBodyOperator, ManyBodyOperator)
        The ``(S_+, S_-, S_z)`` operators for set A and set B (see
        :func:`make_spin_operators`). A and B must address disjoint orbitals.

    Returns
    -------
    ManyBodyState
        :math:`\hat{\mathbf S}_A\cdot\hat{\mathbf S}_B\,|\psi\rangle`.
    """
    a_plus, a_minus, a_z = ops_a
    b_plus, b_minus, b_z = ops_b
    result = a_z(b_z(psi, 0), 0)
    result += 0.5 * a_plus(b_minus(psi, 0), 0)
    result += 0.5 * a_minus(b_plus(psi, 0), 0)
    return result


def expect_spin_correlation(psi, ops_a, ops_b, comm=None):
    r"""Return :math:`\langle\psi|\hat{\mathbf S}_A\cdot\hat{\mathbf S}_B|\psi\rangle`.

    A negative value signals impurity-bath singlet (Kondo) screening. See
    :func:`apply_spin_correlation` for the operator and disjointness requirement.
    """
    val = inner(psi, apply_spin_correlation(psi, ops_a, ops_b))
    if comm is not None:
        val = comm.allreduce(val)
    return np.real(val)


def expect_casimir(psi, j_plus, j_minus, j_z, comm=None):
    r"""Return :math:`\langle\psi|\hat J^2|\psi\rangle` for a (possibly distributed) state.

    Parameters
    ----------
    psi : ManyBodyState
        The state. Assumed normalised (``inner(psi, psi) == 1``).
    j_plus, j_minus, j_z : ManyBodyOperator
        The su(2) ladder/Cartan operators (see :func:`make_spin_operators`).
    comm : MPI.Comm, optional
        If given, the local inner products are summed across ranks. The state must
        be hash-distributed so that every basis determinant reachable by the
        operators is owned by exactly one rank.

    Returns
    -------
    float
        The expectation value :math:`\langle \hat J^2\rangle`.
    """
    val = inner(psi, apply_casimir(psi, j_plus, j_minus, j_z))
    if comm is not None:
        val = comm.allreduce(val)
    return np.real(val)


def _group_degenerate(energies, tol):
    """Group indices of (ascending-sorted) ``energies`` into near-degenerate blocks.

    Returns a list of lists of indices into ``energies``; consecutive energies
    within ``tol`` of the block's first energy share a block.
    """
    groups = []
    current = [0]
    for i in range(1, len(energies)):
        if abs(energies[i] - energies[current[0]]) <= tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def manifold_observable_values(eigenstates, energies, apply_op, degeneracy_tol=1e-6, comm=None, redistribute=None):
    r"""Per-state physical values of an observable on a low-energy manifold.

    Block Lanczos returns a *block* spanning a (near-)degenerate eigenspace; any
    single returned vector is an arbitrary combination within a degenerate
    manifold, so :math:`\langle\psi|\hat O|\psi\rangle` on it is not well defined
    when :math:`[\hat O, H]\neq 0` inside the manifold. For each degenerate
    subspace this builds the small matrix :math:`O_{mn}=\langle m|\hat O|n\rangle`
    and diagonalises it; the eigenvalues are the physical observable values.

    This may be evaluated distributed: pass the rank-local states together with
    ``comm`` and a ``redistribute`` callback (e.g. ``Basis.redistribute_psis``). Then
    :math:`\hat O|n\rangle` is formed on each rank's local partition, redistributed to
    align with the bra partition, the local contributions to :math:`O_{mn}` are summed,
    and the small matrix is ``Allreduce``-d. The result is identical on every rank, so no
    state-vector gather is needed. With ``comm=None`` the ``eigenstates`` are treated as
    full (single-rank) states, as before.

    Parameters
    ----------
    eigenstates : sequence of ManyBodyState
        Orthonormal manifold basis (e.g. a Lanczos block); rank-local when ``comm`` is
        given, otherwise full.
    energies : array_like of shape (N,)
        Energy of each eigenstate (used only to group degenerate subspaces).
    apply_op : callable
        ``apply_op(psi)`` returns :math:`\hat O|\psi\rangle` as a ManyBodyState.
    degeneracy_tol : float, default 1e-6
        Energies within this tolerance are treated as degenerate.
    comm : MPI.Comm, optional
        Communicator over which the states are distributed. If ``None`` the computation
        is purely local.
    redistribute : callable, optional
        ``redistribute(op_states)`` returns the operator-applied states reshuffled onto
        the same determinant partition as ``eigenstates`` (e.g. ``Basis.redistribute_psis``).
        Required when the states are distributed.

    Returns
    -------
    np.ndarray of shape (N,)
        Real observable values aligned with ``eigenstates``. States in the same
        degenerate subspace receive that subspace's eigenvalues (their assignment
        within the subspace is arbitrary, the values being physically interchangeable).
    """
    n = len(eigenstates)
    energies = np.asarray(energies, dtype=float)
    op_states = [apply_op(psi) for psi in eigenstates]
    if redistribute is not None:
        op_states = redistribute(op_states)
    o_matrix = np.array(
        [[inner(eigenstates[i], op_states[j]) for j in range(n)] for i in range(n)],
        dtype=complex,
    )
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, o_matrix, op=MPI.SUM)
    # Hermitise to kill rounding asymmetry before diagonalising.
    o_matrix = 0.5 * (o_matrix + o_matrix.conj().T)

    order = np.argsort(energies, kind="stable")
    values = np.empty(n)
    for block in _group_degenerate(energies[order], degeneracy_tol):
        idx = order[block]
        sub = o_matrix[np.ix_(idx, idx)]
        evals = np.linalg.eigvalsh(sub)
        for k, state_index in enumerate(idx):
            values[state_index] = evals[k]
    return values


def thermal_observable_value(values, energies, tau):
    r"""Boltzmann-weighted average of per-state observable ``values``.

    :math:`\langle\hat O\rangle = \sum_n e^{-\beta E_n} o_n / Z`, evaluated with
    the energy-scale convention of :func:`average.thermal_average_scale_indep`.
    ``values`` should be the physical per-state values from
    :func:`manifold_observable_values`, so degenerate manifolds contribute correctly.
    """
    return thermal_average_scale_indep(np.asarray(energies, dtype=float), np.asarray(values), tau)


def casimir_to_quantum_number(jj_plus_1):
    r"""Invert :math:`j(j+1)` to recover the angular-momentum quantum number ``j``.

    Parameters
    ----------
    jj_plus_1 : float
        The Casimir eigenvalue :math:`j(j+1)`.

    Returns
    -------
    float
        ``j = (-1 + sqrt(1 + 4 j(j+1))) / 2`` (clamped at 0 for tiny negatives).
    """
    return 0.5 * (-1.0 + np.sqrt(max(1.0 + 4.0 * np.real(jj_plus_1), 0.0)))


def print_thermal_expectation_values(
    rho_thermal,
    e_thermal,
    rot_to_spherical,
    block_structure,
    s_thermal=None,
    l_thermal=None,
    j_thermal=None,
    sisb_thermal=None,
):
    """
    print several thermal expectation values, e.g. E, N, Sz, Lz.

    If ``s_thermal`` / ``l_thermal`` / ``j_thermal`` are given (the thermally-averaged
    impurity ``S(S+1)`` / ``L(L+1)`` / ``J(J+1)``), the corresponding ``<S^2>`` /
    ``<L^2>`` / ``<J^2>`` lines (with the quantum number) are appended. When all are
    ``None`` the output is identical to before.
    """
    orb_offset = min(orb for block in block_structure.blocks for orb in block)
    equivalent_blocks = get_equivalent_blocks(block_structure)
    rho_thermal_spherical = rotate_matrix(rho_thermal, rot_to_spherical)
    N, Ndn, Nup = get_occupations_from_rho_spherical(rho_thermal_spherical)

    # Collect (label, value, suffix) rows, then print with the '=' signs aligned and the
    # numbers right-aligned (sign-padded), so the column reads as a tidy table.
    rows = [
        ("<E-E0>", e_thermal, ""),
        ("<N>", N, ""),
        ("<N(Dn)>", Ndn, ""),
        ("<N(Up)>", Nup, ""),
    ]
    for blocks in equivalent_blocks:
        occ = np.sum(
            np.diag(rho_thermal)[list(orb - orb_offset for block in blocks for orb in block_structure.blocks[block])]
        ).real
        rows.append((f"<N({','.join(str(orb) for orb in blocks)})>", occ, ""))
    rows.append(("<Lz>", get_Lz_from_rho_spherical(rho_thermal_spherical), ""))
    rows.append(("<Sz>", get_Sz_from_rho_spherical(rho_thermal_spherical), ""))
    rows.append(("<L.S>", get_LS_from_rho_spherical(rho_thermal_spherical), ""))
    for label, value in (("S", s_thermal), ("L", l_thermal), ("J", j_thermal)):
        if value is not None:
            rows.append((f"<{label}^2>", np.real(value), f"({label} = {casimir_to_quantum_number(value): 6.4f})"))
    if sisb_thermal is not None:
        rows.append(("<S_imp.S_bath>", np.real(sisb_thermal), ""))

    label_width = max(len(label) for label, _, _ in rows)
    for label, value, suffix in rows:
        line = f"{label:<{label_width}} = {value: 12.7f}"
        if suffix:
            line += f"  {suffix}"
        print(line)
