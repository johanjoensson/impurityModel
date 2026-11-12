"""
Ground-state and thermal observables: occupations and angular-momentum
expectation values from single-particle density matrices in the spherical
basis, many-body spin/orbital/Casimir operators, and (thermally averaged)
expectation-value reporting for degenerate manifolds.
"""

from collections import Counter
from functools import lru_cache
from typing import Optional

import numpy as np
from mpi4py import MPI

from impurityModel.ed.atomic_physics import gauntC
from impurityModel.ed.average import thermal_average_scale_indep
from impurityModel.ed.degeneracy import MANIFOLD_TOL, degenerate_groups
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


def block_group_labels(equivalent_blocks, block_structure, impurity_orbitals=None):
    r"""Name each block equivalence class after the orbital group it belongs to.

    The classes from :func:`block_structure.get_equivalent_blocks` need a reader-facing name:
    their raw *block indices* are positions in ``block_structure.blocks``, so a column headed
    ``N(4,6,9,11)`` neither identifies orbitals nor spans the impurity. The name used here is
    the ``impurity_orbitals`` group key -- exactly what the occupation search, the
    basis-generation windows and the frozen-shell notes call the same set of orbitals.

    Naming is by containment, never by inference:

    * one class per group -> ``group 1``;
    * a group the block structure splits further (e.g. one ``3d`` shell auto-split into
      ``eg`` and ``t2g``) -> ``group 1.a``, ``group 1.b``;
    * a class spanning several groups -> ``groups 1,2``;
    * no ``impurity_orbitals``, or a class holding an orbital outside every group -> a plain
      letter, since there is then no group to name it after.

    Labels used to carry a guessed shell instead (``l=2,a``). The guess inverted
    ``n = 2(2l+1)`` on the *orbital count* of the group, which is only meaningful when a group
    is a whole spin-doubled shell. Groups derived by :func:`symmetries.group_orbitals_by_blocks`
    -- the selfenergy and double-counting paths -- are single symmetry manifolds by
    construction, so the inversion labelled a 6-orbital ``t2g`` manifold ``l=1`` (a p shell,
    which has the same orbital count) and degraded ``eg``'s 4 orbitals to ``mixed``.

    Parameters
    ----------
    equivalent_blocks : list of list of int
        From :func:`block_structure.get_equivalent_blocks`.
    block_structure : BlockStructure
        Its ``.blocks`` hold the orbital indices of each block, in the same convention as
        ``impurity_orbitals`` (both are the impurity spin-orbital indices of the basis being
        reported on).
    impurity_orbitals : dict, optional
        ``Basis.impurity_orbitals``.

    Returns
    -------
    (labels, legend) : (list of str, list of (str, list of int))
        ``labels[i]`` is the label for ``equivalent_blocks[i]``; ``legend`` pairs each
        label with its sorted orbital indices, for a legend line.
    """
    orb_to_group = {orb: group for group, blocks in (impurity_orbitals or {}).items() for blk in blocks for orb in blk}
    # The blocks hold positions into the sorted impurity list; the groups hold spin-orbital indices.
    imp_sorted = sorted(orb_to_group)

    def _global(pos):
        return imp_sorted[pos] if 0 <= pos < len(imp_sorted) else pos

    class_orbs = [
        sorted(_global(orb) for b in blocks for orb in block_structure.blocks[b]) for blocks in equivalent_blocks
    ]
    # Which groups each class draws on; None as soon as one of its orbitals belongs to no
    # group at all, which is the one case there is no group name to give.
    signatures = []
    for orbs in class_orbs:
        groups = {orb_to_group.get(orb) for orb in orbs}
        signatures.append(None if None in groups else tuple(sorted(groups)))
    split_groups = Counter(sig for sig in signatures if sig is not None and len(sig) == 1)

    counters: dict = {}

    def _next(key):
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        return idx

    labels = []
    legend = []
    for orbs, sig in zip(class_orbs, signatures):
        if sig is None:
            label = _letter(_next(None))
        elif len(sig) > 1:
            label = "groups " + ",".join(str(group) for group in sig)
        elif split_groups[sig] > 1:
            label = f"group {sig[0]}.{_letter(_next(sig))}"
        else:
            label = f"group {sig[0]}"
        labels.append(label)
        legend.append((label, orbs))
    return labels, legend


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


def resolve_shell_ls(rot_to_spherical, shell_ls=None):
    """The ``l`` of each consecutive spherical column block of ``rot_to_spherical``, validated.

    ``shell_ls=None`` infers one shell from the column count (``2(2l+1)`` columns), the single-shell
    behaviour every caller had before shells existed. A given tuple is checked against the column count
    (``sum(2(2l+1)) == n_cols``); repeated values are legitimate (two sets of one ``l`` are never merged).
    """
    n_cols = int(np.asarray(rot_to_spherical).shape[1])
    if shell_ls is None:
        l = (n_cols // 2 - 1) // 2
        if l < 0 or 2 * (2 * l + 1) != n_cols:
            raise ValueError(
                f"rot_to_spherical has {n_cols} columns, which is not a spin-doubled l-shell (2*(2l+1)); "
                "pass shell_ls to describe a multi-shell rotation."
            )
        return (l,)
    shell_ls = tuple(int(l) for l in shell_ls)
    if any(l < 0 for l in shell_ls) or sum(2 * (2 * l + 1) for l in shell_ls) != n_cols:
        raise ValueError(
            f"shell_ls={shell_ls} spans {sum(2 * (2 * l + 1) for l in shell_ls)} columns, but rot_to_spherical "
            f"has {n_cols}."
        )
    return shell_ls


def shell_column_slices(shell_ls):
    """``[slice, ...]``: the column block of each shell, in order (shell ``a`` is ``2(2 l_a + 1)`` wide)."""
    out, start = [], 0
    for l in shell_ls:
        out.append(slice(start, start + 2 * (2 * l + 1)))
        start += 2 * (2 * l + 1)
    return out


def shell_index_of_orbitals(rot_to_spherical, shell_ls, tol=1e-6):
    """For every impurity orbital, the shell holding its weight: ``argmax_a sum_{c in cols_a} |R[o, c]|^2``.

    Raises ``ValueError`` if an orbital's largest shell weight is below ``1 - tol`` (it straddles shells).
    """
    R = np.asarray(rot_to_spherical)
    weights = np.stack([np.sum(np.abs(R[:, cols]) ** 2, axis=1) for cols in shell_column_slices(shell_ls)], axis=1)
    index = np.argmax(weights, axis=1)
    best = weights[np.arange(len(index)), index]
    total = np.sum(np.abs(R) ** 2, axis=1)
    bad = np.flatnonzero(best < total * (1.0 - tol))
    if bad.size:
        raise ValueError(
            f"impurity orbitals {bad.tolist()} straddle shells: their weight on the best shell is below "
            f"1 - {tol:g} of their norm (shell_ls={tuple(shell_ls)})."
        )
    return index


def shell_is_complete(R_a, tol=1e-8):
    """``True`` when the shell block ``R_a`` is square and unitary (a whole spin-doubled shell)."""
    R_a = np.asarray(R_a)
    return R_a.shape[0] == R_a.shape[1] and np.allclose(R_a.conj().T @ R_a, np.eye(R_a.shape[1]), atol=tol)


def impurity_shell_rhos(rho_imp, rot_to_spherical, impurity_orbitals=None, shell_ls=None):
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
    if not isinstance(rot_to_spherical, dict) and shell_ls is not None and len(shell_ls) > 1:
        R = np.asarray(rot_to_spherical)
        shell_ls = resolve_shell_ls(R, shell_ls)
        for a, (l, cols) in enumerate(zip(shell_ls, shell_column_slices(shell_ls))):
            yield l, a, rotate_matrix(rho_imp, R[:, cols])
        return
    if not isinstance(rot_to_spherical, dict) or impurity_orbitals is None:
        if not isinstance(rot_to_spherical, dict):
            # The aggregate shell must be one complete spin-doubled l-shell (or described by shell_ls): the
            # get_*_from_rho_spherical helpers infer l from the size and would read a sub-shell wrongly.
            resolve_shell_ls(rot_to_spherical, shell_ls)
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


def compute_shell_observables(rho_imp, rot_to_spherical, impurity_orbitals=None, shell_ls=None):
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
    for l, partition, rho_sph in impurity_shell_rhos(rho_imp, rot_to_spherical, impurity_orbitals, shell_ls):
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
#: `manifold_observable_values` uses internally (`degenerate_groups`) to decide which
#: states share a physical eigenspace of a Casimir operator. `is_j_sharp` reuses this
#: exact scale to identify the ground-state manifold, rather than a fresh literal.
DEGENERACY_TOL = MANIFOLD_TOL


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


def _multishell_casimir_operators(impurity_orbitals, rot_to_spherical, shell_ls, per_shell):
    """``make_impurity_casimir_operators`` for an ndarray rotation holding several shells (or one partial shell).

    Every operator lives on all impurity orbitals; shell ``a`` contributes ``R_a M R_a^dagger`` with
    ``R_a = R[:, cols_a]``. The totals exist only if every shell is complete (isometric ``R_a``);
    a partial shell's ``L``/``S`` is not an operator of the kept space, so the totals are then ``None`` and only the
    complete shells' operators (keyed by shell index; a partial shell is absent) are returned, with
    ``per_shell=True``; otherwise ``ValueError``.
    """
    # Rows of the rotation follow the sorted impurity indices (as rho_imp does), not the group-dict order.
    orbs = sorted(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)
    rot = np.asarray(rot_to_spherical, dtype=complex)
    if rot.shape[0] != len(orbs):
        raise ValueError(f"rotation has {rot.shape[0]} rows for {len(orbs)} impurity orbitals")
    per_shell_ops, complete, totals = {}, True, [ManyBodyOperator() for _ in range(6)]
    for a, (l_a, cols) in enumerate(zip(shell_ls, shell_column_slices(shell_ls))):
        R_a = rot[:, cols]
        shell_complete = np.allclose(R_a.conj().T @ R_a, np.eye(R_a.shape[1]), atol=1e-8)
        complete = complete and shell_complete
        shell = []
        for matrix in _single_particle_lsj_matrices(l_a):
            comp = R_a @ matrix @ R_a.conj().T
            shell.append(
                ManyBodyOperator(
                    {
                        ((orbs[i], "c"), (orbs[j], "a")): comp[i, j]
                        for i in range(len(orbs))
                        for j in range(len(orbs))
                        if abs(comp[i, j]) > 1e-12
                    }
                )
            )
        if per_shell and shell_complete:  # a partial shell's L/S is not an operator of the kept space
            l_ops, s_ops = (shell[1], shell[2], shell[0]), (shell[4], shell[5], shell[3])
            j_ops = tuple(x + y for x, y in zip(l_ops, s_ops))
            per_shell_ops[a] = (l_a, l_ops, s_ops, j_ops)
        for k in range(6):
            totals[k] += shell[k]
    if not complete:
        if not per_shell:
            raise ValueError("a partial shell has no L/S/J operator on the kept space; pass per_shell=True")
        return None, None, None, per_shell_ops
    l_ops = (totals[1], totals[2], totals[0])
    s_ops = (totals[4], totals[5], totals[3])
    j_ops = tuple(x + y for x, y in zip(l_ops, s_ops))
    return (l_ops, s_ops, j_ops, per_shell_ops) if per_shell else (l_ops, s_ops, j_ops)


def make_impurity_casimir_operators(impurity_orbitals, rot_to_spherical, per_shell=False, shell_ls=None):
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
    shell_ls : tuple of int, optional
        ``l`` of each consecutive column block of an ndarray ``rot_to_spherical`` (multi-shell impurity block,
        or a single partial shell). ``None`` keeps the single-shell behaviour. Per-shell operators are then
        keyed by shell index; the totals are ``None`` if any shell is partial.
    """
    # One operator per component, accumulated over the shells with the operator algebra
    # (the shells address disjoint orbitals, so this is a plain direct sum).
    if shell_ls is not None and not isinstance(rot_to_spherical, dict):
        shells = resolve_shell_ls(rot_to_spherical, shell_ls)
        if len(shells) > 1 or not shell_is_complete(np.asarray(rot_to_spherical)):
            return _multishell_casimir_operators(impurity_orbitals, rot_to_spherical, shells, per_shell)
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
    (see :class:`spin_pairs.SpinLayout`, grade ``collinear``), and A and B need not
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


def manifold_square_weights(groups, blocks, n):
    """``col2[n] = sum_m |<m|O|n>|^2`` over the members ``m`` of ``n``'s manifold.

    This is the in-manifold part of ``<n|O^2|n>``: transitions out of the retained manifolds are not in it, which is
    what makes a Curie susceptibility built on it a statement about the retained states alone.
    """
    col2 = np.zeros(n)
    for idx, sub in zip(groups, blocks):
        col2[list(idx)] = np.sum(np.abs(sub) ** 2, axis=0)
    return col2


def spin_curie_susceptibility(es, tau, sz_diag, sz_col2):
    r"""Curie term of the longitudinal impurity spin susceptibility from the retained manifolds.

    .. math:: \chi_{S_z} = \frac{1}{\tau}\Big[\sum_n w_n \sum_{m\in M(n)} |\langle m|S_z|n\rangle|^2
        - \Big(\sum_n w_n \langle n|S_z|n\rangle\Big)^2\Big]

    ``S_z`` is the *impurity* spin. ``sz_diag`` and ``sz_col2`` are the per-state ``<n|S_z|n>`` and
    :func:`manifold_square_weights` of ``S_z``: the second sums ``|<m|S_z|n>|^2`` over the manifold partners
    ``m`` only, so the in-manifold projected term is what enters (transitions out of the manifold are excluded).
    Only meaningful where ``[S_z, H] = 0`` (a verified spin layout): then ``S_z`` does not leave a manifold and this
    is the Curie law. The orbital ``L_z`` and charge analogues are not this: ``L_z`` connects a manifold to states
    outside the retained window (Van Vleck), and the charge variance is already the ``<N^2> - <N>^2`` of the
    marginals.
    """
    sz = float(np.real(thermal_observable_value(np.asarray(sz_diag, dtype=float), es, tau)))
    sz2 = float(np.real(thermal_observable_value(np.asarray(sz_col2, dtype=float), es, tau)))
    return {"tau": float(tau), "sz_thermal": sz, "sz2_curie": sz2, "chi_spin_zz": (sz2 - sz**2) / tau}


def manifold_observable_matrices(
    eigenstates, energies, apply_op, degeneracy_tol=DEGENERACY_TOL, comm=None, redistribute=None
):
    r"""The Hermitised matrix :math:`O_{mn}=\langle m|\hat O|n\rangle` of an observable inside each degenerate manifold.

    One block apply for all states, one ``Allreduce`` when distributed (see :func:`manifold_observable_values` for
    the distribution contract and the parameters). Returns ``(groups, blocks)``: ``groups`` is the manifolds in
    ascending energy as arrays of indices into ``eigenstates`` (the ``anchor="first"`` grouping at
    ``degeneracy_tol``), ``blocks[k]`` the ``len(groups[k])`` square matrix of manifold ``k`` in the order of
    ``groups[k]``. The matrices are identical on every rank.
    """
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
    groups = [order[block] for block in degenerate_groups(energies[order], degeneracy_tol, anchor="first")]
    return groups, [o_matrix[np.ix_(idx, idx)] for idx in groups]


def manifold_values_from_matrices(n, groups, blocks):
    """Per-state values from :func:`manifold_observable_matrices`: k-th sorted eigenvalue of each block to slot k."""
    values = np.empty(n)
    for idx, block in zip(groups, blocks):
        sub = np.asarray(block)
        diag = np.real(np.diag(sub))
        off = np.linalg.norm(sub - np.diag(np.diag(sub)))
        # Already diagonal (a sharpened basis): row k *is* state k, and sorted eigenvalues would pair a value with
        # some other state of the manifold.
        sharp = off <= 1e-8 * (1.0 + np.abs(diag).max(initial=0.0))
        evals = diag if sharp else np.linalg.eigvalsh(sub)
        for k, state_index in enumerate(idx):
            values[state_index] = evals[k]
    return values


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
    groups, blocks = manifold_observable_matrices(
        eigenstates, energies, apply_op, degeneracy_tol, comm=comm, redistribute=redistribute
    )
    return manifold_values_from_matrices(eigenstates.width, groups, blocks)


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

    Groups ``es`` into near-degenerate manifolds with :func:`~impurityModel.ed.degeneracy.degenerate_groups`, using
    the *same* ``degeneracy_tol`` that :func:`manifold_observable_values` already uses to
    decide two states are degenerate (not a fresh energy scale), then checks whether the
    ground manifold's per-state ``J`` quantum numbers (``j_values``) all agree to within
    ``term_symbol``'s own half-integer cleanliness tolerance (``term_tol`` -- reused
    rather than inventing a second "how much spread is too much" literal).

    ``degenerate_groups`` compares each candidate state against the *first* energy in the
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
    ground_block = degenerate_groups(es[order], degeneracy_tol, anchor="first")[0]
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
