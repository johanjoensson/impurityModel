"""
Ground-state and thermal observables: occupations and angular-momentum
expectation values from single-particle density matrices in the spherical
basis, many-body spin/orbital/Casimir operators, and (thermally averaged)
expectation-value reporting for degenerate manifolds.
"""

from functools import lru_cache
from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed.atomic_physics import gauntC
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.block_structure import get_equivalent_blocks
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, block_inner_cy, inner
from impurityModel.ed.utils import rotate_matrix


def _letter(idx):
    """Spreadsheet-column-style label: 0->'a', ..., 25->'z', 26->'aa', 27->'ab', ..."""
    letters = []
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters.append(chr(ord("a") + rem))
    return "".join(reversed(letters))


def _block_shell_tag(orbs, impurity_orbitals):
    """Classify a block's global orbitals by impurity_orbitals partition/shell membership.

    Returns ``None`` when no ``impurity_orbitals`` was given (no shell information at
    all -- distinct from ``"mixed"``, which means shell information *is* available but
    ``orbs`` doesn't fall cleanly into one shell), ``"mixed"`` when ``orbs`` spans more
    than one partition or its partition isn't a full spin-doubled shell, or ``"l=<l>"``.
    """
    if not impurity_orbitals:
        return None
    orb_to_partition = {
        orb: partition for partition, blocks in impurity_orbitals.items() for blk in blocks for orb in blk
    }
    partitions = {orb_to_partition.get(orb) for orb in orbs}
    if len(partitions) != 1 or None in partitions:
        return "mixed"
    partition = next(iter(partitions))
    n = sum(len(blk) for blk in impurity_orbitals[partition])
    l = (n // 2 - 1) // 2
    if 2 * (2 * l + 1) != n:
        return "mixed"
    return f"l={l}"


def shell_qualified_block_labels(equivalent_blocks, block_structure, impurity_orbitals=None):
    r"""Assign compact, shell-qualified labels to :func:`block_structure.get_equivalent_blocks`'
    groups, instead of joining their raw *block indices* (meaningless to a reader -- those
    are positions in ``block_structure.blocks``, not orbital indices, and e.g.
    ``N(4,6,9,11)`` neither identifies orbitals nor spans the impurity).

    Each equivalence-class group gets a label like ``l=2,a`` (its shell, plus a per-shell
    letter distinguishing multiple equivalence classes within one shell, e.g. eg vs t2g),
    ``mixed,a`` if the group's orbitals don't fall cleanly into one shell, or plain ``a``
    when no ``impurity_orbitals`` was given (no shell information available at all).

    Parameters
    ----------
    equivalent_blocks : list of list of int
        From :func:`block_structure.get_equivalent_blocks`.
    block_structure : BlockStructure
        Its ``.blocks`` are global spin-orbital indices per block.
    impurity_orbitals : dict, optional
        ``Basis.impurity_orbitals``.

    Returns
    -------
    (labels, legend) : (list of str, list of (str, list of int))
        ``labels[i]`` is the label for ``equivalent_blocks[i]``; ``legend`` pairs each
        label with its sorted global orbital indices, for a legend line.
    """
    counters = {}
    labels = []
    legend = []
    for blocks in equivalent_blocks:
        orbs = sorted(orb for b in blocks for orb in block_structure.blocks[b])
        tag = _block_shell_tag(orbs, impurity_orbitals)
        counter_key = tag if tag is not None else ""
        idx = counters.get(counter_key, 0)
        counters[counter_key] = idx + 1
        letter = _letter(idx)
        label = f"{tag},{letter}" if tag is not None else letter
        labels.append(label)
        legend.append((label, orbs))
    return labels, legend


def print_impurity_orbital_groups(equivalent_blocks, block_structure, impurity_orbitals=None, file=None):
    """Print the legend mapping each ``N(...)`` column label to its global orbital indices.

    A no-op when there is only one group (nothing to disambiguate).
    """
    _, legend = shell_qualified_block_labels(equivalent_blocks, block_structure, impurity_orbitals)
    if len(legend) <= 1:
        return
    entries = [f"{label}: {orbs}" for label, orbs in legend]
    print(f"Impurity orbital groups:  {'  '.join(entries)}", file=file)


def print_expectation_values(
    rhos,
    es,
    rot_to_spherical,
    block_structure,
    s_values=None,
    l_values=None,
    j_values=None,
    sisb_values=None,
    sisb_z_values=None,
    impurity_orbitals=None,
):
    """
    print several expectation values, e.g. E, N, L^2.

    If ``s_values`` / ``l_values`` / ``j_values`` are given (one impurity ``S`` / ``L``
    / ``J`` quantum number per eigenstate, e.g. from :func:`manifold_observable_values`
    with :func:`make_impurity_casimir_operators` + :func:`casimir_to_quantum_number`),
    the corresponding columns are appended; ``sisb_z_values`` (the longitudinal
    ``<Sz_imp Sz_bath>``, reported for a collinear spin-polarized bath) adds a
    ``Szi.Szb`` column. When all are ``None`` the output is identical to before (used
    when the eigenstates are not available, e.g. on non-root ranks).

    ``impurity_orbitals`` (``Basis.impurity_orbitals``, optional) makes the ``N``/``Lz``/
    ``Sz``/``L.S`` columns shell-aware via :func:`compute_shell_observables`: for a
    multi-shell impurity (e.g. a 2p+3d core-hole model) each shell's own contribution is
    evaluated separately and summed, instead of misinferring a single ``l`` from the
    concatenated block. ``None`` (the default) reproduces today's single-shell behaviour
    exactly.
    """
    orb_offset = min(orb for block in block_structure.blocks for orb in block)
    equivalent_blocks = get_equivalent_blocks(block_structure)
    print(f"E0 = {es[0]:9.6f}")
    block_labels, _ = shell_qualified_block_labels(equivalent_blocks, block_structure, impurity_orbitals)
    block_N_string = [f"N({label})" for label in block_labels]
    # Each block-occupation column is right-aligned to a width that fits both its header
    # and the 7-8 char ``.5f`` value below it, so header and numbers line up.
    block_N_widths = [max(len(Ns), 8) for Ns in block_N_string]
    block_N_string_formatted = [f"{Ns:>{w}s}" for Ns, w in zip(block_N_string, block_N_widths)]
    extra = [
        (name, vals)
        for name, vals in (
            ("S", s_values),
            ("L", l_values),
            ("J", j_values),
            ("Si.Sb", sisb_values),
            ("Szi.Szb", sisb_z_values),
        )
        if vals is not None
    ]
    # Lz/Sz/L.S and the S/L/J/Si.Sb columns are printed with the space-flag format
    # ``{x: 8.6f}``, which is 9 characters wide (the sign column sits on top of the 8),
    # so their headers must be 9 wide to line up with the numbers below.
    extra_header = "".join(f"  {name:>9s}" for name, _ in extra)
    print(
        f"{'i':>3s}  {'E-E0':>11s}  {'N':>8s}  {'N(Dn)':>8s}  {'N(Up)':>8s}  "
        f"{'  '.join(block_N_string_formatted)}  {'Lz':>9s}  {'Sz':>9s}  {'L.S':>9s}{extra_header}"
    )
    for i, (e, rho) in enumerate(zip(es - es[0], rhos)):
        block_occs = [
            np.sum(np.diag(rho)[[orb - orb_offset for block in blocks for orb in block_structure.blocks[block]]])
            for blocks in equivalent_blocks
        ]
        block_occ_string_formatted = ["" for _ in block_occs]
        for ib, b_occ in enumerate(block_occs):
            block_occ_string_formatted[ib] = f"{np.real(b_occ):>{block_N_widths[ib]}.5f}"
        totals = compute_shell_observables(rho, rot_to_spherical, impurity_orbitals)["total"]
        N, Ndn, Nup = totals["n"], totals["n_dn"], totals["n_up"]
        Lz, Sz, LS = totals["lz"], totals["sz"], totals["l_dot_s"]
        extra_fields = "".join(f"  {vals[i]: 8.6f}" for _, vals in extra)
        print(
            f"{i:>3d}  {e:11.8f}  {N:8.5f}  {Ndn:8.5f}  {Nup:8.5f}  "
            f"{'  '.join(block_occ_string_formatted)}  {Lz: 8.6f}  {Sz: 8.6f}  {LS: 8.6f}{extra_fields}"
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


def get_moments_from_rho_spherical(rho: np.ndarray, l: Optional[int] = None) -> tuple[float, float]:
    r"""Magnetic moments :math:`\langle L_z + 2S_z\rangle` and :math:`\langle J_z\rangle`.

    :math:`\langle L_z + 2S_z\rangle` is the saturation magnetic moment along ``z`` in
    units of :math:`\mu_B` (the magnetic-moment operator is
    :math:`\mu_z = -\mu_B (L_z + 2S_z)`; the sign convention is left to the reader, the
    printed value is the bare :math:`\langle L_z + 2S_z\rangle`).

    Parameters
    ----------
    rho : np.ndarray
        The density matrix in the spherical basis (down-then-up layout).
    l : int, optional
        The orbital angular momentum quantum number. If None, inferred from rho's shape.

    Returns
    -------
    (m_z, j_z) : tuple of float
        :math:`(\langle L_z + 2S_z\rangle, \langle L_z + S_z\rangle)`.
    """
    lz = get_Lz_from_rho_spherical(rho, l)
    sz = get_Sz_from_rho_spherical(rho, l)
    return float(lz + 2.0 * sz), float(lz + sz)


@lru_cache(maxsize=None)
def _single_particle_tz_matrix(l):
    r"""Single-particle magnetic-dipole operator :math:`T_z` in the spherical basis.

    :math:`T_z = \sum_i [s_z - 3\hat r_z(\hat{\mathbf r}\cdot\mathbf s)]_i` — the
    intra-atomic magnetic-dipole term of the XMCD spin sum rule. Expanding the
    direction-cosine products in Racah tensors :math:`C^2_q` gives the one-body form

    .. math::
        T_z = -2 C^2_0\, s_z - \sqrt{3/2}\,(C^2_{-1}\, s_+ - C^2_{+1}\, s_-),

    whose orbital matrix elements :math:`\langle l m'|C^2_q|l m\rangle` are Gaunt
    coefficients (:func:`atomic_physics.gauntC`). Layout matches
    :func:`_single_particle_lsj_matrices`: first ``2l+1`` orbitals spin-down
    (``ml=-l..l``), then spin-up.

    For a pure :math:`|m_l, m_s\rangle` state this reproduces the closed form
    :math:`\langle T_z\rangle = m_s[1 - 3(2l^2+2l-1-2m_l^2)/((2l-1)(2l+3))]`.
    """
    n = 2 * l + 1
    tz = np.zeros((2 * n, 2 * n), dtype=complex)
    pref = np.sqrt(1.5)
    for a, m in enumerate(range(-l, l + 1)):
        c20 = gauntC(2, l, m, l, m)
        tz[a, a] += c20  # -2 * (-1/2) * C^2_0  (spin-down block)
        tz[n + a, n + a] -= c20  # -2 * (+1/2) * C^2_0  (spin-up block)
        if m - 1 >= -l:
            # -sqrt(3/2) C^2_{-1} s_+ : |dn, m> -> |up, m-1>
            tz[n + a - 1, a] -= pref * gauntC(2, l, m - 1, l, m)
        if m + 1 <= l:
            # +sqrt(3/2) C^2_{+1} s_- : |up, m> -> |dn, m+1>
            tz[a + 1, n + a] += pref * gauntC(2, l, m + 1, l, m)
    return tz


def get_Tz_from_rho_spherical(rho: np.ndarray, l: Optional[int] = None) -> float:
    r"""Expectation of the magnetic-dipole term :math:`\langle T_z\rangle` (XMCD spin sum rule).

    :math:`T_z` is a one-body operator, so
    :math:`\langle T_z\rangle = \mathrm{Tr}(\rho\, t_z)` with the single-particle matrix
    from :func:`_single_particle_tz_matrix`. Together with :math:`\langle S_z\rangle`
    and :math:`\langle L_z\rangle` this is the ground-state side of the XMCD spin and
    orbital sum rules (the "effective spin moment" is
    :math:`\langle S_z\rangle + \tfrac{7}{2}\langle T_z\rangle` for a d shell).

    Parameters
    ----------
    rho : np.ndarray
        The density matrix in the spherical basis (down-then-up layout).
    l : int, optional
        The orbital angular momentum quantum number. If None, inferred from rho's shape.

    Returns
    -------
    float
    """
    if l is None:
        l = (rho.shape[0] // 2 - 1) // 2
    n = 2 * l + 1
    rho = rho[: 2 * n, : 2 * n]
    return float(np.real(np.trace(rho @ _single_particle_tz_matrix(l))))


def impurity_shell_rhos(rho_imp, rot_to_spherical, impurity_orbitals=None):
    r"""Split the impurity density matrix into one spherical-basis block per l-shell.

    The ``get_*_from_rho_spherical`` helpers above are single-shell primitives: each
    assumes ``rho`` is exactly one spin-doubled ``l``-shell laid out
    ``[down(2l+1), up(2l+1)]``. Handing them a multi-shell impurity block (e.g. a 2p+3d
    core-hole impurity) silently misinfers ``l`` from the concatenated orbital count and
    produces physically wrong Lz/Sz/L.S/etc — this function is the fix: it slices
    ``rho_imp`` back into one rho per physical shell before any of those helpers run.

    ``rot_to_spherical``'s type is the signal for whether ``impurity_orbitals`` partitions
    physical shells at all -- **not** each partition's orbital count, which is ambiguous
    (a cubic shell's t2g manifold has 6 spin-orbitals, the same count as a genuine l=1
    shell). By construction of every current caller:

    * a **dict** (:func:`get_spectra`'s multi-shell path, ``dict(model.rot_to_spherical)``)
      is always keyed by the shell's own ``l``, one full spin-doubled shell per key;
    * a plain **ndarray** (the single-shell path, and every selfenergy run --
      ``prepare_solver_basis`` collapses it to one matrix even when it internally regroups
      the impurity into eg/t2g symmetry manifolds) covers the *whole* impurity block as one
      aggregate shell, exactly like today's un-split behaviour.

    Parameters
    ----------
    rho_imp : np.ndarray
        Impurity block of the density matrix, in the computational basis, indexed by
        position in ``sorted(impurity_indices)`` (the convention used throughout
        ``groundstate.calc_gs``, e.g. ``rhos[full_impurity_ix]``).
    rot_to_spherical : np.ndarray or dict
        Spherical -> computational rotation. A dict (keyed by shell ``l``) splits
        ``rho_imp`` into one shell per key; a plain array treats the whole block as one
        shell (today's behaviour).
    impurity_orbitals : dict, optional
        ``Basis.impurity_orbitals`` (partition -> list of orbital-index blocks), needed to
        map each dict key to its global orbital indices. Required when ``rot_to_spherical``
        is a dict; ignored otherwise. ``None`` (or a non-dict rotation) always yields a
        single aggregate shell, exactly today's behaviour.

    Yields
    ------
    (l, partition, rho_spherical) : tuple of (int or None, hashable or None, np.ndarray)
        ``l`` is the shell's angular momentum, or ``None`` for the single aggregate-shell
        case (matching the ``l=None`` "infer from shape" convention of the
        ``get_*_from_rho_spherical`` helpers). ``partition`` is the ``impurity_orbitals``
        dict key this shell came from (``None`` in the aggregate case) -- callers that
        need to join this shell against another per-partition structure (e.g. the
        per-shell Casimir dict from :func:`make_impurity_casimir_operators`) must use
        ``partition``, not ``l``: two distinct shells can share the same inferred ``l``,
        and ``l`` alone would then merge them. Shells are yielded in ascending order of
        their lowest global orbital index -- the same ordering ``groundstate.calc_gs``
        uses to build ``rho_imp`` itself (``sorted(impurity_indices)``) -- *not* ascending
        ``l``; nothing requires the lower-``l`` shell to hold the lower orbital indices.
    """
    if not isinstance(rot_to_spherical, dict) or impurity_orbitals is None:
        yield None, None, rotate_matrix(rho_imp, rot_to_spherical)
        return

    impurity_indices = sorted(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)
    position = {orb: i for i, orb in enumerate(impurity_indices)}

    shells = []
    for key, blocks in impurity_orbitals.items():
        if key not in rot_to_spherical:
            raise KeyError(
                f"impurity_orbitals partition {key!r} has no matching entry in the "
                "rot_to_spherical dict; every dict-keyed partition must be a physical "
                "l-shell with its own rotation."
            )
        orbs = sorted(orb for block in blocks for orb in block)
        n = len(orbs)
        l = (n // 2 - 1) // 2
        if 2 * (2 * l + 1) != n:
            raise ValueError(
                f"impurity_orbitals partition {key!r} has {n} spin-orbitals, which is not "
                "a spin-doubled l-shell (2*(2l+1)). rot_to_spherical is a dict, so every "
                "partition is expected to be one full physical shell keyed by its own l "
                "(the get_spectra.py contract); this partition breaks that contract."
            )
        # Carry the dict key alongside the inferred l: the two coincide for every current
        # caller, but the rotation lookup below must use the *key* that indexed
        # rot_to_spherical above, not the independently re-derived l, so a future caller
        # whose keys aren't literally l can't desync validation from the actual lookup.
        shells.append((key, l, orbs))
    shells.sort(key=lambda item: item[2][0])  # ascending lowest global orbital index

    for key, l, orbs in shells:
        positions = [position[orb] for orb in orbs]
        rho_shell_computational = rho_imp[np.ix_(positions, positions)]
        yield l, key, rotate_matrix(rho_shell_computational, rot_to_spherical[key])


def compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals=None):
    r"""Per-shell (and total) one-body impurity observables, from the density matrix.

    Splits ``rho_imp`` into per-shell spherical-basis blocks with
    :func:`impurity_shell_rhos` and evaluates ``N``/``N(Dn)``/``N(Up)``/``Lz``/``Sz``/
    ``Lz+2Sz``/``Jz``/``L.S``/``T_z`` on each shell independently, then sums them into a
    ``"total"`` entry. Every one of these is an additive one-body observable, so the sum
    over shells is exact -- this ``"total"`` should always be used in place of evaluating
    the same helpers on the concatenated multi-shell rho (the bug this function replaces).

    Parameters
    ----------
    rho_imp : np.ndarray
        Impurity block of the density matrix, computational basis (see
        :func:`impurity_shell_rhos`).
    rot_to_spherical : np.ndarray or dict
        Spherical -> computational rotation.
    impurity_orbitals : dict, optional
        ``Basis.impurity_orbitals``; see :func:`impurity_shell_rhos`.

    Returns
    -------
    dict
        ``{"shells": [...], "total": {...}}``. Each shell dict has keys ``"l"`` (the
        shell's angular momentum, or ``None`` for a single aggregate shell),
        ``"partition"`` (the ``impurity_orbitals`` dict key this shell came from, or
        ``None`` in the aggregate case -- the key to join against
        :func:`make_impurity_casimir_operators`'s ``per_shell_ops``; two distinct shells
        can share the same inferred ``l``, so ``l`` alone is not a safe join key),
        ``"n"`` (total occupation, ``n_dn + n_up``), ``"n_dn"``, ``"n_up"``, ``"lz"``,
        ``"sz"``, ``"m_z"`` (:math:`\langle L_z+2S_z\rangle`), ``"j_z"``, ``"l_dot_s"``,
        ``"t_z"``. ``"total"`` has every key except ``"l"``/``"partition"`` (a sum over
        shells is not itself a shell), each the plain sum of that key over ``"shells"``.
    """
    shells = []
    for l, partition, rho_sph in impurity_shell_rhos(rho_imp, rot_to_spherical, impurity_orbitals):
        n, n_dn, n_up = get_occupations_from_rho_spherical(rho_sph)
        lz = get_Lz_from_rho_spherical(rho_sph, l)
        sz = get_Sz_from_rho_spherical(rho_sph, l)
        m_z, j_z = get_moments_from_rho_spherical(rho_sph, l)
        l_dot_s = get_LS_from_rho_spherical(rho_sph, l)
        t_z = get_Tz_from_rho_spherical(rho_sph, l)
        shells.append(
            {
                "l": l,
                "partition": partition,
                "n": n,
                "n_dn": n_dn,
                "n_up": n_up,
                "lz": lz,
                "sz": sz,
                "m_z": m_z,
                "j_z": j_z,
                "l_dot_s": l_dot_s,
                "t_z": t_z,
            }
        )
    total = {
        key: sum(shell[key] for shell in shells)
        for key in ("n", "n_dn", "n_up", "lz", "sz", "m_z", "j_z", "l_dot_s", "t_z")
    }
    return {"shells": shells, "total": total}


_TERM_LETTERS = "SPDFGHIKLMNOQRTUV"

#: Tolerance for treating a quantum number (S, J as half-integers; L as an integer) as
#: "clean" rather than a mixed-valence/strongly-hybridized non-integer. Single source of
#: truth for both `term_symbol`'s own "~" decision and `is_j_sharp`'s manifold-spread
#: gate (Stage 5 of the multi-shell observable fix): reusing one named tolerance for both
#: "is this single value clean" and "do these values agree with each other" avoids a
#: second, independently-tunable literal for what is conceptually the same judgment.
TERM_SYMBOL_TOL = 5e-2

#: Energy tolerance for grouping eigenstates into a degenerate manifold -- the same scale
#: `manifold_observable_values` uses internally (`_group_degenerate`) to decide which
#: states share a physical eigenspace of a Casimir operator. `is_j_sharp` reuses this
#: exact scale to identify the ground-state manifold, rather than a fresh literal.
DEGENERACY_TOL = 1e-6


def term_symbol(s, l, j=None, tol=TERM_SYMBOL_TOL):
    r"""Spectroscopic term symbol :math:`^{2S+1}L_J` (or :math:`^{2S+1}L` without ``j``)
    from quantum numbers.

    Parameters
    ----------
    s, l : float
        The spin / orbital angular-momentum quantum numbers (e.g. from
        :func:`casimir_to_quantum_number` of the impurity Casimirs).
    j : float, optional
        The total angular-momentum quantum number. ``None`` omits the ``J`` subscript
        entirely (e.g. ``"3F"`` rather than ``"3F4"``) -- for use when ``J`` is not a
        sharp quantum number across the manifold (see :func:`is_j_sharp`), where
        reporting *some* ``J`` subscript would misleadingly imply one exists.
    tol : float, optional
        Tolerance for treating ``s``/``j`` as half-integers and ``l`` as an integer.

    Returns
    -------
    str
        E.g. ``"3F4"`` or ``"2F7/2"`` (``"3F"`` when ``j`` is ``None``). When the values
        are not clean (half-)integers — a mixed-valence or strongly hybridized state —
        the nearest term is prefixed with ``~`` (e.g. ``"~3F4"``) to mark it approximate.
    """
    s, l = float(s), float(l)
    l_r = round(l)
    mult = round(2 * s + 1)
    letter = _TERM_LETTERS[l_r] if l_r < len(_TERM_LETTERS) else f"(L={l_r})"
    if j is None:
        clean = abs(2 * s - round(2 * s)) <= 2 * tol and abs(l - round(l)) <= tol
        term = f"{mult}{letter}"
        return term if clean else "~" + term
    j = float(j)
    clean = abs(2 * s - round(2 * s)) <= 2 * tol and abs(l - round(l)) <= tol and abs(2 * j - round(2 * j)) <= 2 * tol
    j2_r = round(2 * j)
    j_str = str(j2_r // 2) if j2_r % 2 == 0 else f"{j2_r}/2"
    term = f"{mult}{letter}{j_str}"
    return term if clean else "~" + term


def lande_g_and_moments(s2, l2, j2, j2_tol=1e-3):
    r"""Landé :math:`g_J` and effective moments from Casimir expectation values.

    .. math::
        g_J = \tfrac32 + \frac{\langle S^2\rangle - \langle L^2\rangle}
                              {2\langle J^2\rangle},
        \qquad
        \mu_\mathrm{eff} = g_J \sqrt{\langle J^2\rangle},
        \qquad
        \mu_\mathrm{spin} = 2\sqrt{\langle S^2\rangle},

    in units of :math:`\mu_B`. Evaluated directly on the (thermal) expectation values
    :math:`\langle S^2\rangle = S(S+1)` etc., so mixed-valence states give the
    correspondingly interpolated moments.

    Parameters
    ----------
    s2, l2, j2 : float
        Expectation values of the :math:`S^2`, :math:`L^2`, :math:`J^2` Casimirs.
    j2_tol : float, optional
        Below this :math:`\langle J^2\rangle`, ``g_J``/``mu_eff`` are returned as
        ``None`` (a ``J=0`` state has no Landé factor).

    Returns
    -------
    (g_j, mu_eff, mu_spin) : tuple
        ``g_j`` and ``mu_eff`` are ``None`` when :math:`\langle J^2\rangle \le` tol.
    """
    s2 = float(np.real(s2))
    l2 = float(np.real(l2))
    j2 = float(np.real(j2))
    mu_spin = 2.0 * np.sqrt(max(s2, 0.0))
    if j2 <= j2_tol:
        return None, None, mu_spin
    g_j = 1.5 + (s2 - l2) / (2.0 * j2)
    return g_j, g_j * np.sqrt(j2), mu_spin


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
    for indices_raw in channels:
        indices = list(indices_raw)
        l = (len(indices) - 1) // 2
        llp1 = l * (l + 1)
        for a, ml in enumerate(range(-l, l + 1)):
            l_z[((indices[a], "c"), (indices[a], "a"))] = float(ml)
            if ml < l:
                coeff = np.sqrt(llp1 - ml * (ml + 1))
                l_plus[((indices[a + 1], "c"), (indices[a], "a"))] = coeff
                l_minus[((indices[a], "c"), (indices[a + 1], "a"))] = coeff
    return ManyBodyOperator(l_plus), ManyBodyOperator(l_minus), ManyBodyOperator(l_z)


def make_impurity_casimir_operators(impurity_orbitals, rot_to_spherical, per_shell=False):
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
    per_shell : bool, default False
        When ``True``, also return the un-summed per-shell operators (see below). Every
        partition must be a full spin-doubled shell either way -- this does not change
        which inputs raise, it only additionally exposes what was already computed per
        shell before being summed.

    Returns
    -------
    (L, S, J) : tuple
        Each is ``(plus, minus, z)`` as ``ManyBodyOperator``s, ready for
        :func:`apply_casimir` / :func:`expect_casimir`. ``J = L + S``.
    per_shell_ops : dict, only when ``per_shell=True``
        ``{partition: (l, l_ops, s_ops, j_ops)}``, one entry per impurity partition, in
        the same ``(plus, minus, z)`` layout as the totals above -- e.g. ``L`` and ``S``
        for one shell, summed, reproduce that shell's contribution to the totals exactly
        (the total is a plain sum over shells, since they address disjoint orbitals).
    """
    # One operator per component, accumulated over the shells with the operator algebra
    # (the shells address disjoint orbitals, so this is a plain direct sum).
    l_plus, l_minus, l_z = ManyBodyOperator(), ManyBodyOperator(), ManyBodyOperator()
    s_plus, s_minus, s_z = ManyBodyOperator(), ManyBodyOperator(), ManyBodyOperator()
    per_shell_ops = {}
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
        shell = []
        for matrix in (lz_m, lp_m, lm_m, sz_m, sp_m, sm_m):
            computational = rot @ matrix @ rot.conj().T
            shell.append(
                ManyBodyOperator(
                    {
                        ((orbs[i], "c"), (orbs[j], "a")): computational[i, j]
                        for i in range(n_so)
                        for j in range(n_so)
                        if abs(computational[i, j]) > 1e-12
                    }
                )
            )
        l_z += shell[0]
        l_plus += shell[1]
        l_minus += shell[2]
        s_z += shell[3]
        s_plus += shell[4]
        s_minus += shell[5]
        if per_shell:
            shell_l_ops = (shell[1], shell[2], shell[0])
            shell_s_ops = (shell[4], shell[5], shell[3])
            shell_j_ops = (
                shell_l_ops[0] + shell_s_ops[0],
                shell_l_ops[1] + shell_s_ops[1],
                shell_l_ops[2] + shell_s_ops[2],
            )
            per_shell_ops[partition] = (shell_l, shell_l_ops, shell_s_ops, shell_j_ops)
    l_ops = (l_plus, l_minus, l_z)
    s_ops = (s_plus, s_minus, s_z)
    j_ops = (l_ops[0] + s_ops[0], l_ops[1] + s_ops[1], l_ops[2] + s_ops[2])
    if per_shell:
        return l_ops, s_ops, j_ops, per_shell_ops
    return l_ops, s_ops, j_ops


def casimir_operator(j_plus, j_minus, j_z):
    r"""The su(2) Casimir :math:`\hat J^2` as an explicit two-body operator.

    Built from the ladder identity :math:`\hat J^2 = \hat J_- \hat J_+ + \hat J_z^2 +
    \hat J_z` (with :math:`\hat J_- = \hat J_+^\dagger`), so only the one-body
    ladder/Cartan operators are needed as input. Build this once and reuse it across
    states rather than calling :func:`apply_casimir` per state, which rebuilds it.

    Parameters
    ----------
    j_plus, j_minus, j_z : ManyBodyOperator
        The raising, lowering, and Cartan operators of the su(2) algebra.

    Returns
    -------
    ManyBodyOperator
        :math:`\hat J^2`.
    """
    return j_minus * j_plus + j_z * j_z + j_z


def apply_casimir(psi, j_plus, j_minus, j_z):
    r"""Apply a su(2) Casimir operator to ``psi`` and return the resulting state.

    Convenience wrapper over :func:`casimir_operator`; hoist that out of any loop over
    states instead of calling this repeatedly.

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
    return casimir_operator(j_plus, j_minus, j_z)(psi, 0)


def spin_correlation_operator(ops_a, ops_b, z_only=False):
    r"""The spin-correlation operator :math:`\hat{\mathbf S}_A\cdot\hat{\mathbf S}_B`.

    For two **disjoint** orbital sets A and B the spin operators commute, so

    .. math::
        \hat{\mathbf S}_A\cdot\hat{\mathbf S}_B
            = \hat S^A_z \hat S^B_z
            + \tfrac12\left(\hat S^A_+ \hat S^B_- + \hat S^A_- \hat S^B_+\right).

    ``z_only`` keeps just the Ising term :math:`\hat S^A_z \hat S^B_z`. That part needs
    only the spin *labels* (up vs down), not the down↔up pairing, so it stays exact for a
    collinear spin-polarized bath where the transverse pairing is a modelling choice
    (see :func:`spin_pairs.collinear_spin_pairs_consistent_with_h`), and A and B need not
    be disjoint there — passing the same set twice yields :math:`\hat S_z^2`.

    Parameters
    ----------
    ops_a, ops_b : (ManyBodyOperator, ManyBodyOperator, ManyBodyOperator)
        The ``(S_+, S_-, S_z)`` operators for set A and set B (see
        :func:`make_spin_operators`). A and B must address disjoint orbitals unless
        ``z_only``.
    z_only : bool, optional
        Drop the transverse terms.

    Returns
    -------
    ManyBodyOperator
    """
    a_plus, a_minus, a_z = ops_a
    b_plus, b_minus, b_z = ops_b
    op = a_z * b_z
    if not z_only:
        op += 0.5 * (a_plus * b_minus) + 0.5 * (a_minus * b_plus)
    return op


def apply_spin_correlation(psi, ops_a, ops_b):
    r"""Apply :math:`\hat{\mathbf S}_A\cdot\hat{\mathbf S}_B` to ``psi``.

    Convenience wrapper over :func:`spin_correlation_operator`; hoist that out of any
    loop over states instead of calling this repeatedly.
    """
    return spin_correlation_operator(ops_a, ops_b)(psi, 0)


def apply_spin_z_correlation(psi, ops_a, ops_b):
    r"""Apply the longitudinal :math:`\hat S^A_z \hat S^B_z` to ``psi``.

    Convenience wrapper over ``spin_correlation_operator(..., z_only=True)``.
    """
    return spin_correlation_operator(ops_a, ops_b, z_only=True)(psi, 0)


def get_Sz_from_rho_pairs(rho, spin_pairs):
    r"""``<S_z> = 1/2 sum_a (n_{a up} - n_{a dn})`` from a density matrix and a pairing.

    Works directly in the computational basis: ``spin_pairs`` lists the ``(dn, up)``
    spin-orbital index pairs (into ``rho``) of the orbital set, so no spherical
    rotation is needed. Only the spin labels matter (each pair contributes through its
    two diagonal entries), so this is exact whenever ``[h, S_z] = 0`` validates the
    labelling — including for a collinear spin-polarized bath.

    Parameters
    ----------
    rho : np.ndarray
        Single-particle density matrix in the computational basis.
    spin_pairs : sequence of (int, int)
        ``(dn, up)`` index pairs of the orbital set.

    Returns
    -------
    float
    """
    return float(np.real(sum(0.5 * (rho[up, up] - rho[dn, dn]) for dn, up in spin_pairs)))


def expect_spin_correlation(psi, ops_a, ops_b, comm=None):
    r"""Return :math:`\langle\psi|\hat{\mathbf S}_A\cdot\hat{\mathbf S}_B|\psi\rangle`.

    A negative value signals impurity-bath singlet (Kondo) screening. See
    :func:`spin_correlation_operator` for the operator and disjointness requirement.
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


def compute_correlation_diagnostics(psis, es, tau, thermal_rho, imp_pairs, comm=None, redistribute=None):
    r"""Correlation-strength diagnostics of the impurity (Mott/Hund physics).

    Per impurity spatial orbital ``a`` (one ``(dn, up)`` pair each): the double occupancy
    :math:`d_a = \langle n_{a\uparrow} n_{a\downarrow}\rangle` and the local moment
    :math:`\langle m_z^2\rangle_a = (\langle n_\uparrow\rangle + \langle n_\downarrow\rangle
    - 2 d_a)/4` (using :math:`n^2 = n`). For the whole impurity: :math:`\langle S_z^2\rangle`,
    the static longitudinal susceptibility
    :math:`\chi_{zz} = (\langle S_z^2\rangle - \langle S_z\rangle^2)/\tau`, and the
    inter-orbital Hund matrix :math:`\langle \mathbf S_a\cdot\mathbf S_b\rangle` (diagonal
    :math:`\langle S_a^2\rangle = 3\langle m_z^2\rangle_a`).

    All two-body values are evaluated per state with :func:`manifold_observable_values`
    (manifold-safe) and thermally averaged. Collective when ``comm`` is given.

    Parameters
    ----------
    psis : ManyBodyState
        Eigenstates as one shared-support block (rank-local when distributed), one
        column per state.
    es, tau
        Energies, thermal energy scale.
    thermal_rho : np.ndarray
        Full thermally-averaged one-particle density matrix (replicated).
    imp_pairs : list of (int, int)
        Validated impurity ``(dn, up)`` spin-orbital pairs.
    comm, redistribute
        As in :func:`manifold_observable_values`.

    Returns
    -------
    dict
    """
    n_sp = len(imp_pairs)
    docc_values = np.empty((n_sp, psis.width))
    for a, (dn, up) in enumerate(imp_pairs):
        op = ManyBodyOperator({((up, "c"), (dn, "c"), (dn, "a"), (up, "a")): 1.0})
        docc_values[a] = manifold_observable_values(
            psis, es, lambda blk, _op=op: _op.apply_block(blk, 0), comm=comm, redistribute=redistribute
        )
    docc_thermal = np.array([np.real(thermal_observable_value(docc_values[a], es, tau)) for a in range(n_sp)])
    n_dn = np.array([np.real(thermal_rho[dn, dn]) for dn, _ in imp_pairs])
    n_up = np.array([np.real(thermal_rho[up, up]) for _, up in imp_pairs])
    mz2 = (n_dn + n_up - 2.0 * docc_thermal) / 4.0

    ops_imp = make_spin_operators(imp_pairs)
    sz2_op = spin_correlation_operator(ops_imp, ops_imp, z_only=True)
    sz2_values = manifold_observable_values(
        psis,
        es,
        lambda blk: sz2_op.apply_block(blk, 0),
        comm=comm,
        redistribute=redistribute,
    )
    sz2_thermal = float(np.real(thermal_observable_value(sz2_values, es, tau)))
    sz_thermal = get_Sz_from_rho_pairs(thermal_rho, imp_pairs)
    chi_zz = (sz2_thermal - sz_thermal**2) / tau

    hund = np.diag(3.0 * mz2)
    orbital_ops = [make_spin_operators([pair]) for pair in imp_pairs]
    for i in range(n_sp):
        for j in range(i + 1, n_sp):
            corr_op = spin_correlation_operator(orbital_ops[i], orbital_ops[j])
            vals = manifold_observable_values(
                psis,
                es,
                lambda blk, _op=corr_op: _op.apply_block(blk, 0),
                comm=comm,
                redistribute=redistribute,
            )
            hund[i, j] = hund[j, i] = float(np.real(thermal_observable_value(vals, es, tau)))

    return {
        "pairs": [(int(dn), int(up)) for dn, up in imp_pairs],
        "n_dn": n_dn,
        "n_up": n_up,
        "docc": docc_thermal,
        "docc_values": np.real(docc_values),
        "docc_total": float(np.sum(docc_thermal)),
        "local_moment_z2": mz2,
        "sz2_values": np.real(sz2_values),
        "sz2_thermal": sz2_thermal,
        "sz_thermal": sz_thermal,
        "chi_zz": float(chi_zz),
        "tau": float(tau),
        "hund": hund,
    }


def compute_static_susceptibilities(
    psis, es, tau, impurity_indices, s_z_op=None, l_z_op=None, comm=None, redistribute=None
):
    r"""Static (Curie) susceptibilities of the impurity from the retained thermal manifold.

    Fluctuation-dissipation form :math:`\chi_O = (\langle O^2\rangle_\mathrm{th} -
    \langle O\rangle_\mathrm{th}^2)/\tau` for the impurity charge :math:`O = N`, and —
    when the many-body operators are available — the longitudinal spin
    :math:`O = S_z`, orbital :math:`O = L_z` and the spin-orbital cross term
    :math:`\chi_{SL} = (\langle S_z L_z\rangle - \langle S_z\rangle\langle
    L_z\rangle)/\tau`. Comparing them is the static Hund's-metal fingerprint: a large,
    slowly screened :math:`\chi_\mathrm{spin}` next to quenched
    :math:`\chi_\mathrm{orb}` and suppressed :math:`\chi_\mathrm{charge}`.

    These are *Curie* terms only — fluctuations within the retained low-energy states;
    Van Vleck contributions from states above the energy cut are not included (see the
    dynamical susceptibility CLI for the full response).

    All expectation values are evaluated per state with
    :func:`manifold_observable_values` (manifold-safe) and thermally averaged.
    Collective when ``comm`` is given.

    Parameters
    ----------
    psis : ManyBodyState
        Eigenstates as one shared-support block (rank-local when distributed), one
        column per state.
    es, tau
        Energies, thermal energy scale.
    impurity_indices : sequence of int
        Flat impurity spin-orbital indices (for the charge operator).
    s_z_op, l_z_op : ManyBodyOperator, optional
        Many-body impurity :math:`S_z` / :math:`L_z` (e.g. from
        :func:`make_impurity_casimir_operators`); ``None`` skips the respective rows.
    comm, redistribute
        As in :func:`manifold_observable_values`.

    Returns
    -------
    dict
    """
    n_op = ManyBodyOperator({((int(i), "c"), (int(i), "a")): 1.0 for i in impurity_indices})

    def thermal_pair(op_a, op_b):
        """Thermal <A B> (A, B commuting) and, for A == B, the pieces for the variance."""
        product = op_a * op_b
        vals = manifold_observable_values(
            psis,
            es,
            lambda blk: product.apply_block(blk, 0),
            comm=comm,
            redistribute=redistribute,
        )
        return float(np.real(thermal_observable_value(vals, es, tau)))

    def thermal_single(op):
        vals = manifold_observable_values(
            psis, es, lambda blk: op.apply_block(blk, 0), comm=comm, redistribute=redistribute
        )
        return float(np.real(thermal_observable_value(vals, es, tau)))

    n_th = thermal_single(n_op)
    n2_th = thermal_pair(n_op, n_op)
    result = {
        "tau": float(tau),
        "n_thermal": n_th,
        "n2_thermal": n2_th,
        "chi_charge": (n2_th - n_th**2) / tau,
        "chi_spin_zz": None,
        "chi_orb_zz": None,
        "chi_spin_orb": None,
    }
    if s_z_op is not None:
        sz_th = thermal_single(s_z_op)
        sz2_th = thermal_pair(s_z_op, s_z_op)
        result["sz_thermal"] = sz_th
        result["sz2_thermal"] = sz2_th
        result["chi_spin_zz"] = (sz2_th - sz_th**2) / tau
    if l_z_op is not None:
        lz_th = thermal_single(l_z_op)
        lz2_th = thermal_pair(l_z_op, l_z_op)
        result["lz_thermal"] = lz_th
        result["lz2_thermal"] = lz2_th
        result["chi_orb_zz"] = (lz2_th - lz_th**2) / tau
    if s_z_op is not None and l_z_op is not None:
        szlz_th = thermal_pair(s_z_op, l_z_op)
        result["szlz_thermal"] = szlz_th
        result["chi_spin_orb"] = (szlz_th - result["sz_thermal"] * result["lz_thermal"]) / tau
    return result


def static_susceptibility_rows(chi):
    """``(label, value, suffix)`` rows for the *Static susceptibilities* report group."""
    rows = []
    if chi.get("chi_spin_zz") is not None:
        rows.append(("chi_spin_zz", chi["chi_spin_zz"], "((<Sz^2> - <Sz>^2)/tau)"))
    if chi.get("chi_orb_zz") is not None:
        rows.append(("chi_orb_zz", chi["chi_orb_zz"], "((<Lz^2> - <Lz>^2)/tau)"))
    if chi.get("chi_spin_orb") is not None:
        rows.append(("chi_spin_orb", chi["chi_spin_orb"], "((<Sz.Lz> - <Sz><Lz>)/tau)"))
    rows.append(("chi_charge", chi["chi_charge"], "((<N^2> - <N>^2)/tau)"))
    rows.append(
        (
            "note",
            None,
            "Curie terms of the retained manifold only (no Van Vleck part); tau = " f"{chi['tau']:.4g}",
        )
    )
    return rows


def print_correlation_diagnostics(corr, file=None):
    """Pretty-print the dict from :func:`compute_correlation_diagnostics`."""
    print("Impurity correlation diagnostics (thermal):", file=file)
    print(f"  {'orbital(dn,up)':>16s} {'n_dn':>8s} {'n_up':>8s} {'<n_up*n_dn>':>12s} {'<m_z^2>':>9s}", file=file)
    for a, (dn, up) in enumerate(corr["pairs"]):
        pair_label = f"({dn},{up})"
        print(
            f"  {pair_label:>16s} {corr['n_dn'][a]:>8.5f} {corr['n_up'][a]:>8.5f} "
            f"{corr['docc'][a]:>12.6f} {corr['local_moment_z2'][a]:>9.5f}",
            file=file,
        )
    print(f"  total double occupancy D = {corr['docc_total']:.6f}", file=file)
    print(
        f"  <Sz_imp^2> = {corr['sz2_thermal']:.6f}   "
        f"chi_zz = (<Sz^2> - <Sz>^2)/tau = {corr['chi_zz']:.4f}  (tau = {corr['tau']:.4g})",
        file=file,
    )
    print("  Inter-orbital <S_i.S_j> (i,j = impurity spatial orbitals, labelled by their (dn,up) pairs):", file=file)
    for i, row in enumerate(corr["hund"]):
        label = f"({corr['pairs'][i][0]},{corr['pairs'][i][1]})"
        print(f"  {label:>10s} " + " ".join(f"{x: 9.5f}" for x in np.real(row)), file=file)


def compute_screening_diagnostics(
    psis,
    es,
    tau,
    thermal_rho,
    imp_pairs,
    bath_pairs,
    h1,
    imp_groups=None,
    z_only=False,
    comm=None,
    redistribute=None,
    max_bath_correlation_levels=200,
):
    r"""Channel- and bath-level-resolved impurity-bath spin correlations (Kondo screening).

    Two resolutions of :math:`\langle\mathbf S_\mathrm{imp}\cdot\mathbf S_\mathrm{bath}\rangle`:

    - **per impurity channel**: ``imp_groups`` maps a label (e.g. the equivalent-block
      group of the ``N(...)`` columns) to a subset of impurity pairs; one correlation per
      group tells *which* orbital channel is screened;
    - **per bath level**: one row per bath ``(dn, up)`` pair with its on-site energies,
      thermal occupations, hybridization strengths and (when the level count does not
      exceed ``max_bath_correlation_levels``) its :math:`\langle\mathbf S_\mathrm{imp}
      \cdot\mathbf S_b\rangle` — the screening cloud resolved over the bath spectrum.

    With ``z_only=True`` (collinear spin-polarized bath) the longitudinal part
    :math:`\langle S^z_\mathrm{imp} S^z_b\rangle` is evaluated instead — exact under the
    label-only validation, consistent with the flagged full ``<S_imp.S_bath>``.

    Parameters
    ----------
    h1 : np.ndarray
        One-body Hamiltonian matrix (``extract_tensors(..., two_body=False)``).
    imp_groups : dict or None
        ``{label: [pair, ...]}``; ``None`` skips the channel resolution.

    Returns
    -------
    dict
    """

    def corr_op(ops_left, ops_right):
        return spin_correlation_operator(ops_left, ops_right, z_only=z_only)

    imp_orbs = sorted(orb for pair in imp_pairs for orb in pair)

    channels = []
    if imp_groups:
        ops_bath = make_spin_operators(bath_pairs)
        for label, pairs in imp_groups.items():
            if not pairs:
                continue
            op_g = corr_op(make_spin_operators(pairs), ops_bath)
            vals = manifold_observable_values(
                psis,
                es,
                lambda blk, _op=op_g: _op.apply_block(blk, 0),
                comm=comm,
                redistribute=redistribute,
            )
            channels.append((label, float(np.real(thermal_observable_value(vals, es, tau)))))

    with_correlation = len(bath_pairs) <= max_bath_correlation_levels
    ops_imp = make_spin_operators(imp_pairs)
    levels = []
    for dn, up in bath_pairs:
        row = {
            "pair": (int(dn), int(up)),
            "eps_dn": float(np.real(h1[dn, dn])),
            "eps_up": float(np.real(h1[up, up])),
            "n_dn": float(np.real(thermal_rho[dn, dn])),
            "n_up": float(np.real(thermal_rho[up, up])),
            "v_dn": float(np.linalg.norm(h1[imp_orbs, dn])),
            "v_up": float(np.linalg.norm(h1[imp_orbs, up])),
        }
        if with_correlation:
            op_b = corr_op(ops_imp, make_spin_operators([(dn, up)]))
            vals = manifold_observable_values(
                psis,
                es,
                lambda blk, _op=op_b: _op.apply_block(blk, 0),
                comm=comm,
                redistribute=redistribute,
            )
            row["sisb"] = float(np.real(thermal_observable_value(vals, es, tau)))
        levels.append(row)
    levels.sort(key=lambda r: 0.5 * (r["eps_dn"] + r["eps_up"]))

    return {"z_only": bool(z_only), "channels": channels, "levels": levels, "with_correlation": with_correlation}


def print_screening_diagnostics(scr, file=None):
    """Pretty-print the dict from :func:`compute_screening_diagnostics`."""
    corr_label = "Sz_imp.Sz_b" if scr["z_only"] else "S_imp.S_b"
    print("Screening channels (thermal):", file=file)
    if scr["z_only"]:
        print("  (spin-polarized bath: longitudinal z-parts only, exact under the label check)", file=file)
    for label, value in scr["channels"]:
        name = f"<Sz_imp({label}).Sz_bath>" if scr["z_only"] else f"<S_imp({label}).S_bath>"
        print(f"  {name} = {value: .6f}", file=file)
    header = (
        f"  {'bath(dn,up)':>14s} {'eps_dn':>9s} {'eps_up':>9s} {'n_dn':>8s} {'n_up':>8s} {'|V|_dn':>8s} {'|V|_up':>8s}"
    )
    if scr["with_correlation"]:
        header += f" {f'<{corr_label}>':>13s}"
    print("  Bath levels (sorted by energy):", file=file)
    print(header, file=file)
    for row in scr["levels"]:
        pair_label = "({},{})".format(*row["pair"])
        line = (
            f"  {pair_label:>14s} {row['eps_dn']:>9.4f} {row['eps_up']:>9.4f} "
            f"{row['n_dn']:>8.5f} {row['n_up']:>8.5f} {row['v_dn']:>8.4f} {row['v_up']:>8.4f}"
        )
        if scr["with_correlation"]:
            line += f" {row['sisb']:>13.6f}"
        print(line, file=file)


def compute_magnetic_summary(
    rho_imp, rot_to_spherical, s2=None, l2=None, j2=None, impurity_orbitals=None, j_sharp=True
):
    r"""JSON-able summary of the magnetism/multiplet observables (A-bundle).

    Same underlying helpers as :func:`print_thermal_expectation_values`; used to persist
    the values into the ground-state statistics JSON.

    Parameters
    ----------
    rho_imp : np.ndarray
        Impurity block of the (thermal) density matrix, computational basis.
    rot_to_spherical : np.ndarray
        Spherical -> computational rotation for the impurity block.
    s2, l2, j2 : float, optional
        Thermal Casimir expectation values; when all are given the term symbol and
        Landé/effective moments are included.
    impurity_orbitals : dict, optional
        ``Basis.impurity_orbitals``; makes ``lz``/``sz``/etc shell-aware via
        :func:`compute_shell_observables` -- see :func:`print_expectation_values`.
    j_sharp : bool, default True
        From :func:`is_j_sharp`; gates ``term``/``g_j``/``mu_eff`` via
        :func:`gated_term_symbol`, matching the printed report -- see
        :func:`print_thermal_expectation_values`. The persisted JSON must agree with what
        is printed, so this defaults to ``True`` (ungated) only for callers that have not
        computed ``j_sharp``, not as a "usually fine" assumption.

    Returns
    -------
    dict
    """
    totals = compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals)["total"]
    out = {
        "lz": float(totals["lz"]),
        "sz": float(totals["sz"]),
        "m_z": float(totals["m_z"]),
        "j_z": float(totals["j_z"]),
        "t_z": float(totals["t_z"]),
        "l_dot_s": float(totals["l_dot_s"]),
    }
    if s2 is not None and l2 is not None and j2 is not None:
        _, _, mu_spin = lande_g_and_moments(s2, l2, j2)
        term, g_j, mu_eff = gated_term_symbol(s2, l2, j2, j_sharp)
        out.update(
            {
                "s2": float(np.real(s2)),
                "l2": float(np.real(l2)),
                "j2": float(np.real(j2)),
                "term": term,
                "g_j": g_j,
                "mu_eff": mu_eff,
                "mu_spin_only": mu_spin,
            }
        )
    return out


def compute_state_summary(
    rhos, es, rot_to_spherical, s_values=None, l_values=None, j_values=None, entanglement=None, impurity_orbitals=None
):
    r"""Compact per-eigenstate summary rows (term symbol, moments, entanglement).

    One dict per state with ``energy_rel``, ``m_z`` (:math:`\langle L_z+2S_z\rangle`),
    ``j_z``, and — when available — ``term`` (from the manifold-resolved S/L/J values)
    and ``s_ent`` (impurity-bath entanglement entropy). Shared by the printed table and
    the statistics JSON.

    Note the usual degenerate-manifold caveat: ``m_z``/``j_z`` are single-particle
    values on one arbitrary member of a degenerate manifold; ``term`` and ``s_ent``
    are manifold-safe.

    Parameters
    ----------
    rhos : np.ndarray
        Per-state impurity density matrices (computational basis), shape (n, d, d).
    es : array_like
        Eigen-energies.
    rot_to_spherical : np.ndarray
        Spherical -> computational rotation for the impurity block.
    s_values, l_values, j_values : array_like or None
        Per-state quantum numbers (from the Casimirs); all three needed for ``term``.
    entanglement : dict or None
        The dict from :func:`gs_statistics.compute_entanglement_entropy`.
    impurity_orbitals : dict, optional
        ``Basis.impurity_orbitals``; makes ``m_z``/``j_z`` shell-aware (summed over
        shells) via :func:`compute_shell_observables` -- see
        :func:`print_expectation_values`.

    Returns
    -------
    list of dict
    """
    es = np.asarray(es, dtype=float)
    e0 = float(np.min(es)) if es.size else 0.0
    have_term = s_values is not None and l_values is not None and j_values is not None
    ent_values = entanglement["per_state_entropy"] if entanglement is not None else None
    rows = []
    for i, (e, rho) in enumerate(zip(es, rhos)):
        totals = compute_shell_observables(rho, rot_to_spherical, impurity_orbitals)["total"]
        m_z, j_z = totals["m_z"], totals["j_z"]
        row = {"state": i, "energy_rel": float(e - e0), "m_z": m_z, "j_z": j_z}
        if have_term:
            row["term"] = term_symbol(s_values[i], l_values[i], j_values[i])
        if ent_values is not None and i < len(ent_values):
            row["s_ent"] = float(ent_values[i])
        rows.append(row)
    return rows


def print_state_summary(summary, file=None):
    """Pretty-print the rows from :func:`compute_state_summary`."""
    have_term = any("term" in row for row in summary)
    have_ent = any("s_ent" in row for row in summary)
    header = f"{'i':>3s}  {'E-E0':>11s}  {'Lz+2Sz':>9s}  {'Jz':>9s}"
    if have_term:
        header += f"  {'term':>8s}"
    if have_ent:
        header += f"  {'S_ent':>8s}"
    print("Per-state summary (m_z/Jz are arbitrary within a degenerate manifold; term/S_ent are not):", file=file)
    print(header, file=file)
    for row in summary:
        line = f"{row['state']:>3d}  {row['energy_rel']:11.8f}  {row['m_z']: 8.5f}  {row['j_z']: 8.5f}"
        if have_term:
            line += f"  {row.get('term', ''):>8s}"
        if have_ent:
            line += f"  {row.get('s_ent', float('nan')):>8.4f}"
        print(line, file=file)


def compute_energy_decomposition(thermal_rho, h1, impurity_indices, e_thermal):
    r"""Decompose the thermal energy into one-body blocks and the Coulomb remainder.

    With the density-matrix convention :math:`\rho_{ij} = \langle c_j^\dagger c_i\rangle`
    the one-body energy is :math:`\langle H_{1b}\rangle = \mathrm{Tr}(h_1\rho)`, split into
    the impurity block, the bath block and the impurity-bath (hybridization) cross terms;
    the interaction energy follows as
    :math:`\langle H_U\rangle = \langle E\rangle - \langle H_{1b}\rangle`.

    Returns
    -------
    dict with keys ``e_imp_1b``, ``e_bath``, ``e_hyb``, ``e_one_body``, ``e_coulomb``,
    ``e_total``.
    """
    n_orb = thermal_rho.shape[0]
    imp = sorted(impurity_indices)
    bath = sorted(set(range(n_orb)) - set(imp))
    e_one_body = float(np.real(np.trace(h1 @ thermal_rho)))
    e_imp = float(np.real(np.trace(h1[np.ix_(imp, imp)] @ thermal_rho[np.ix_(imp, imp)])))
    e_bath = float(np.real(np.trace(h1[np.ix_(bath, bath)] @ thermal_rho[np.ix_(bath, bath)])))
    e_hyb = e_one_body - e_imp - e_bath
    return {
        "e_imp_1b": e_imp,
        "e_bath": e_bath,
        "e_hyb": e_hyb,
        "e_one_body": e_one_body,
        "e_coulomb": float(np.real(e_thermal) - e_one_body),
        "e_total": float(np.real(e_thermal)),
    }


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


def manifold_observable_values(
    eigenstates, energies, apply_op, degeneracy_tol=DEGENERACY_TOL, comm=None, redistribute=None
):
    r"""Per-state physical values of an observable on a low-energy manifold.

    Block Lanczos returns a *block* spanning a (near-)degenerate eigenspace; any
    single returned vector is an arbitrary combination within a degenerate
    manifold, so :math:`\langle\psi|\hat O|\psi\rangle` on it is not well defined
    when :math:`[\hat O, H]\neq 0` inside the manifold. For each degenerate
    subspace this builds the small matrix :math:`O_{mn}=\langle m|\hat O|n\rangle`
    and diagonalises it; the eigenvalues are the physical observable values.

    This may be evaluated distributed: pass the rank-local block together with
    ``comm`` and a ``redistribute`` callback (e.g. ``Basis.redistribute_block``). Then
    :math:`\hat O|n\rangle` is formed on each rank's local partition, redistributed to
    align with the bra partition, the local contributions to :math:`O_{mn}` are summed,
    and the small matrix is ``Allreduce``-d. The result is identical on every rank, so no
    state-vector gather is needed. With ``comm=None`` the ``eigenstates`` block is treated
    as full (single-rank), as before.

    Parameters
    ----------
    eigenstates : ManyBodyState
        Orthonormal manifold basis (e.g. a Lanczos block) as one shared-support block,
        one column per state; rank-local when ``comm`` is given, otherwise full.
    energies : array_like of shape (N,)
        Energy of each eigenstate (used only to group degenerate subspaces).
    apply_op : callable
        ``apply_op(eigenstates)`` returns :math:`\hat O` applied to every column at once,
        as a ``ManyBodyState`` of the same width (e.g. ``lambda blk: op.apply_block(blk,
        0)``) -- one term/sign pass per determinant shared across all N states, not N
        independent applies.
    degeneracy_tol : float, default 1e-6
        Energies within this tolerance are treated as degenerate.
    comm : MPI.Comm, optional
        Communicator over which the states are distributed. If ``None`` the computation
        is purely local.
    redistribute : callable, optional
        ``redistribute(op_states)`` returns the operator-applied block reshuffled onto
        the same determinant partition as ``eigenstates`` (e.g. ``Basis.redistribute_block``).
        Required when the states are distributed.

    Returns
    -------
    np.ndarray of shape (N,)
        Real observable values aligned with ``eigenstates``. States in the same
        degenerate subspace receive that subspace's eigenvalues (their assignment
        within the subspace is arbitrary, the values being physically interchangeable).
    """
    n = eigenstates.width
    energies = np.asarray(energies, dtype=float)
    op_states = apply_op(eigenstates)
    if redistribute is not None:
        op_states = redistribute(op_states)
    o_matrix = block_inner_cy(eigenstates, op_states)
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


def is_j_sharp(j_values, es, degeneracy_tol=DEGENERACY_TOL, term_tol=TERM_SYMBOL_TOL):
    r"""Whether ``J`` is a sharp (well-defined) quantum number across the ground-state
    manifold.

    Without spin-orbit coupling, ``J`` is generically **not** a good quantum number: the
    crystal-field/Coulomb multiplet splits states by ``S`` and ``L`` but leaves ``J``
    free to vary within one energy-degenerate manifold (this is real physics, not
    numerical noise -- see e.g. a cubic ``d^8`` ground manifold, where ``J`` ranges over
    several values at the same energy). Reporting a single ``term``/``g_J``/``mu_eff``
    for such a manifold would misleadingly imply a definite ``J``.

    Groups ``es`` into near-degenerate manifolds with :func:`_group_degenerate`, using
    the *same* ``degeneracy_tol`` that :func:`manifold_observable_values` already uses to
    decide two states are degenerate (not a fresh energy scale), then checks whether the
    ground manifold's per-state ``J`` quantum numbers (``j_values``) all agree to within
    ``term_symbol``'s own half-integer cleanliness tolerance (``term_tol`` -- reused
    rather than inventing a second "how much spread is too much" literal).

    ``_group_degenerate`` compares each candidate state against the *first* energy in the
    growing block, not transitively against its immediate neighbour, so a manifold spread
    over many states with consecutive gaps just under ``degeneracy_tol`` can be grouped
    narrower than a transitive-closure reading of "all mutually within tolerance" would
    give. That only ever shrinks the ground block this function inspects, which is the
    conservative direction: a narrower block has less room to show ``J``-spread, so this
    can only bias the gate towards *not* suppressing, never towards a false suppression.

    Parameters
    ----------
    j_values : array_like
        Per-state ``J`` quantum numbers (e.g. ``casimir_to_quantum_number`` applied to
        :func:`manifold_observable_values`'s per-state ``J^2`` eigenvalues), one per
        retained eigenstate, in the same order as ``es``.
    es : array_like
        The corresponding eigen-energies (need not be pre-sorted).
    degeneracy_tol : float, default :data:`DEGENERACY_TOL`
        Energy tolerance for the ground-state manifold grouping.
    term_tol : float, default :data:`TERM_SYMBOL_TOL`
        ``J``-spread tolerance within the ground manifold.

    Returns
    -------
    bool
    """
    es = np.asarray(es, dtype=float)
    j_values = np.asarray(j_values, dtype=float)
    order = np.argsort(es, kind="stable")
    ground_block = _group_degenerate(es[order], degeneracy_tol)[0]
    ground_j = j_values[order[ground_block]]
    return bool(np.max(ground_j) - np.min(ground_j) <= 2 * term_tol)


def gated_term_symbol(s2, l2, j2, j_sharp):
    r"""Term symbol / Landé ``g_J`` / ``mu_eff`` from thermal Casimir values, gated on
    ``J`` being a sharp quantum number across the manifold (see :func:`is_j_sharp`).

    When ``j_sharp`` is ``False`` the term symbol is built from ``S``/``L`` alone (no
    ``J`` subscript, e.g. ``"~3F"`` rather than ``"~3F4"``) and ``g_J``/``mu_eff`` --
    which need a well-defined ``J`` -- are suppressed (``None``). ``mu_spin_only`` is
    unaffected by this gate (it only needs ``S``) and is not returned here; compute it
    separately with :func:`lande_g_and_moments` as before.

    Both call sites that print a term symbol (the scalar magnetism block in
    :func:`print_thermal_expectation_values` and the per-shell table in
    :func:`_print_shell_observable_table`) must go through this one function, sharing
    the same ``j_sharp`` decision -- otherwise they could print contradictory terms
    (e.g. a gated ``~3F`` next to an ungated ``~3F4``) for the same manifold.

    Parameters
    ----------
    s2, l2, j2 : float
        Thermally-averaged impurity ``S(S+1)`` / ``L(L+1)`` / ``J(J+1)``.
    j_sharp : bool
        From :func:`is_j_sharp`.

    Returns
    -------
    (term, g_j, mu_eff) : (str, float or None, float or None)
    """
    s_qn = casimir_to_quantum_number(s2)
    l_qn = casimir_to_quantum_number(l2)
    if not j_sharp:
        return term_symbol(s_qn, l_qn, None), None, None
    j_qn = casimir_to_quantum_number(j2)
    term = term_symbol(s_qn, l_qn, j_qn)
    g_j, mu_eff, _ = lande_g_and_moments(s2, l2, j2)
    return term, g_j, mu_eff


def _print_shell_observable_table(shells, total, shell_casimir, j_sharp=True):
    r"""Print the *Per-shell impurity observables* table (one row per shell + a total row).

    ``shells`` is ``compute_shell_observables(...)["shells"]`` (one dict per shell, in
    ascending global-orbital order); ``total`` is that same call's ``["total"]`` -- passed
    in rather than re-summed here, so the table's total row is guaranteed to equal the
    scalar charge/magnetism rows printed above it, not an independently computed number
    that happens to agree. ``shell_casimir`` is ``{partition: {"l", "s2_thermal",
    "l2_thermal", "j2_thermal"}}`` from ``calc_gs``'s per-shell Casimir evaluation, joined
    here by ``shell["partition"]`` -- **not** ``shell["l"]``, since two distinct shells can
    share the same inferred ``l`` (e.g. two independent p-shells). A shell absent from
    ``shell_casimir`` (Casimirs unavailable/skipped) prints its N/Lz/Sz/L.S/T_z columns
    with the ``<S^2>``/``<L^2>``/``<J^2>``/``term`` columns left blank.

    The total row deliberately excludes the Casimir columns, since ``S^2``/``L^2``/``J^2``
    do not simply add across shells (a many-body Casimir of a combined system is not the
    sum of its parts' Casimirs).
    """
    header = (
        f"  {'shell':>5s}  {'N':>8s}  {'N(Dn)':>8s}  {'N(Up)':>8s}  {'Lz':>9s}  {'Sz':>9s}  "
        f"{'L.S':>9s}  {'T_z':>9s}  {'<S^2>':>7s}  {'<L^2>':>7s}  {'<J^2>':>7s}  {'term':>6s}"
    )
    print("Per-shell impurity observables:")
    print(header)
    for shell in shells:
        shell_label = f"l={shell['l']}" if shell["l"] is not None else "?"
        casimir = shell_casimir.get(shell["partition"], {}) if shell_casimir else {}
        line = (
            f"  {shell_label:>5s}  {shell['n']:8.5f}  {shell['n_dn']:8.5f}  {shell['n_up']:8.5f}  "
            f"{shell['lz']: 9.5f}  {shell['sz']: 9.5f}  {shell['l_dot_s']: 9.5f}  {shell['t_z']: 9.5f}"
        )
        if all(k in casimir for k in ("s2_thermal", "l2_thermal", "j2_thermal")):
            s2, l2, j2 = (np.real(casimir[k]) for k in ("s2_thermal", "l2_thermal", "j2_thermal"))
            term, _, _ = gated_term_symbol(s2, l2, j2, j_sharp)
            line += f"  {s2:7.4f}  {l2:7.4f}  {j2:7.4f}  {term:>6s}"
        print(line)
    total_line = (
        f"  {'total':>5s}  {total['n']:8.5f}  {total['n_dn']:8.5f}  {total['n_up']:8.5f}  "
        f"{total['lz']: 9.5f}  {total['sz']: 9.5f}  {total['l_dot_s']: 9.5f}  {total['t_z']: 9.5f}"
    )
    print(total_line)


def print_thermal_expectation_values(
    rho_thermal,
    e_thermal,
    rot_to_spherical,
    block_structure,
    s_thermal=None,
    l_thermal=None,
    j_thermal=None,
    sisb_thermal=None,
    sisb_z_thermal=None,
    sisb_z_connected=None,
    sisb_pairing_approx=False,
    extra_groups=None,
    impurity_orbitals=None,
    shell_casimir=None,
    j_sharp=True,
):
    """
    print several thermal expectation values, e.g. E, N, Sz, Lz.

    The table is printed in titled sub-blocks (*Charge & occupations*, *Magnetism &
    multiplets*, *Spin correlations*) with the ``=`` signs aligned across all blocks.
    ``extra_groups`` (optional) is a list of ``(title, rows)`` sub-blocks appended after
    the built-in ones (e.g. the energy decomposition), where each row is a
    ``(label, value, suffix)`` tuple; a ``None`` value prints the suffix as text.

    If ``s_thermal`` / ``l_thermal`` / ``j_thermal`` are given (the thermally-averaged
    impurity ``S(S+1)`` / ``L(L+1)`` / ``J(J+1)``), the corresponding ``<S^2>`` /
    ``<L^2>`` / ``<J^2>`` lines (with the quantum number) are appended. When all are
    ``None`` the output is identical to before.

    For a collinear spin-polarized bath the longitudinal correlation and its connected
    (covariance) form can be passed as ``sisb_z_thermal`` / ``sisb_z_connected`` — they
    print as ``<Sz_imp.Sz_bath>`` / ``cov(Sz_imp,Sz_bath)`` — and
    ``sisb_pairing_approx=True`` marks the full ``<S_imp.S_bath>`` line as depending on
    the (index-convention) bath down↔up pairing.

    ``impurity_orbitals`` (``Basis.impurity_orbitals``, optional) makes the charge/
    magnetism rows shell-aware via :func:`compute_shell_observables` -- see
    :func:`print_expectation_values`. ``None`` reproduces today's single-shell behaviour.
    When it resolves to more than one shell, a *Per-shell impurity observables* table is
    additionally printed (see :func:`_print_shell_observable_table`); ``shell_casimir``
    (from ``calc_gs``'s per-shell Casimir evaluation) supplies that table's ``<S^2>``/
    ``<L^2>``/``<J^2>``/``term`` columns when given.

    ``j_sharp`` (default ``True``, from :func:`is_j_sharp`) gates the scalar ``term``/
    ``g_J``/``mu_eff`` rows *and* the per-shell table's ``term`` column through the same
    :func:`gated_term_symbol` decision -- without spin-orbit coupling ``J`` is generically
    not a good quantum number, so reporting one value for it is misleading; when
    ``j_sharp=False`` the term omits the ``J`` subscript and ``g_J``/``mu_eff`` are
    suppressed (``mu_spin_only`` is unaffected). The default ``True`` reproduces today's
    (ungated) behaviour for callers that have not computed ``j_sharp``.
    """
    orb_offset = min(orb for block in block_structure.blocks for orb in block)
    equivalent_blocks = get_equivalent_blocks(block_structure)
    block_labels, _ = shell_qualified_block_labels(equivalent_blocks, block_structure, impurity_orbitals)
    shell_observables = compute_shell_observables(rho_thermal, rot_to_spherical, impurity_orbitals)
    totals = shell_observables["total"]
    N, Ndn, Nup = totals["n"], totals["n_dn"], totals["n_up"]

    # Collect (label, value, suffix) rows into titled sub-blocks, then print with the '='
    # signs aligned across all blocks and the numbers right-aligned (sign-padded), so the
    # whole section reads as one tidy table.
    charge_rows = [
        ("<E>", e_thermal, ""),
        ("<N>", N, ""),
        ("<N(Dn)>", Ndn, ""),
        ("<N(Up)>", Nup, ""),
    ]
    for blocks, label in zip(equivalent_blocks, block_labels):
        occ = np.sum(
            np.diag(rho_thermal)[[orb - orb_offset for block in blocks for orb in block_structure.blocks[block]]]
        ).real
        charge_rows.append((f"<N({label})>", occ, ""))
    magnetism_rows = [
        ("<Lz>", totals["lz"], ""),
        ("<Sz>", totals["sz"], ""),
    ]
    magnetism_rows.append(("<Lz+2Sz>", totals["m_z"], "(saturation moment m_z, in mu_B)"))
    magnetism_rows.append(("<Jz>", totals["j_z"], ""))
    magnetism_rows.append(("<L.S>", totals["l_dot_s"], ""))
    magnetism_rows.append(("<T_z>", totals["t_z"], "(magnetic dipole, XMCD spin sum rule)"))
    for label, value in (("S", s_thermal), ("L", l_thermal), ("J", j_thermal)):
        if value is not None:
            magnetism_rows.append(
                (f"<{label}^2>", np.real(value), f"({label} = {casimir_to_quantum_number(value): 6.4f})")
            )
    if s_thermal is not None and l_thermal is not None and j_thermal is not None:
        _, _, mu_spin = lande_g_and_moments(s_thermal, l_thermal, j_thermal)
        term, g_j, mu_eff = gated_term_symbol(s_thermal, l_thermal, j_thermal, j_sharp)
        magnetism_rows.append(("term", None, term))
        if g_j is not None:
            magnetism_rows.append(("g_J", g_j, ""))
            magnetism_rows.append(("mu_eff", mu_eff, "(g_J sqrt(<J^2>), in mu_B)"))
        magnetism_rows.append(("mu_spin_only", mu_spin, "(2 sqrt(<S^2>), in mu_B)"))
    correlation_rows = []
    if sisb_thermal is not None:
        suffix = "(transverse part depends on the bath dn/up pairing)" if sisb_pairing_approx else ""
        correlation_rows.append(("<S_imp.S_bath>", np.real(sisb_thermal), suffix))
    if sisb_z_thermal is not None:
        correlation_rows.append(("<Sz_imp.Sz_bath>", np.real(sisb_z_thermal), ""))
    if sisb_z_connected is not None:
        correlation_rows.append(("cov(Sz_imp,Sz_bath)", np.real(sisb_z_connected), ""))

    groups = [
        ("Charge & occupations", charge_rows),
        ("Magnetism & multiplets", magnetism_rows),
        ("Spin correlations", correlation_rows),
    ]
    if extra_groups:
        groups.extend(extra_groups)
    groups = [(title, rows) for title, rows in groups if rows]

    label_width = max(len(label) for _, rows in groups for label, _, _ in rows)
    for gi, (title, rows) in enumerate(groups):
        if gi > 0:
            print()
        print(f"{title}:")
        for label, value, suffix in rows:
            # Text-only rows (e.g. the term symbol) carry their content in the suffix.
            line = (
                f"  {label:<{label_width}} = {suffix}"
                if value is None
                else f"  {label:<{label_width}} = {value: 12.7f}"
            )
            if suffix and value is not None:
                line += f"  {suffix}"
            print(line)

    if len(shell_observables["shells"]) > 1:
        print()
        _print_shell_observable_table(shell_observables["shells"], totals, shell_casimir, j_sharp)
