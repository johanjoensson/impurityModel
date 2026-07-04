"""
Derivation of impurity/bath spin pairs (the ``(down, up)`` spin-orbital
pairings) consistent with a given one-body Hamiltonian.
"""

from collections import deque

import numpy as np

# Local imports
from impurityModel.ed.observables import _single_particle_lsj_matrices
from impurityModel.ed.symmetries import extract_tensors


def impurity_spin_pairs(impurity_orbitals):
    r"""Return the ``(dn_index, up_index)`` impurity spin-orbital pairs.

    Within each angular-momentum partition of ``impurity_orbitals`` the first half
    of the spin-orbitals are spin-down and the second half spin-up (the basis
    layout, matching ``basis_generation.spin_flipped_determinants``), so orbital
    ``k`` pairs with orbital ``k + n//2``.

    Parameters
    ----------
    impurity_orbitals : dict
        Mapping ``partition -> list of orbital-index blocks`` (``Basis.impurity_orbitals``).

    Returns
    -------
    list of (int, int)
        ``(dn, up)`` global spin-orbital index pairs, suitable for
        :func:`make_spin_operators`.
    """
    pairs = []
    for orb_blocks in impurity_orbitals.values():
        orbs = [orb for block in orb_blocks for orb in block]
        n = len(orbs)
        for k in range(n // 2):
            pairs.append((orbs[k], orbs[k + n // 2]))
    return pairs


def bath_spin_pairs(bath_states):
    r"""Return the ``(dn_index, up_index)`` bath spin-orbital pairs.

    Same down-then-up convention as :func:`impurity_spin_pairs`, applied to each
    valence and conduction bath block independently (the ``get_CF_hamiltonian`` /
    ``c2i`` layout, where a bath block is ``[down(2l+1), up(2l+1)]``). Odd-sized blocks
    are skipped. The result is only *trusted* after
    :func:`spin_pairs_consistent_with_h` confirms the induced spin operators commute
    with the one-body Hamiltonian.

    Parameters
    ----------
    bath_states : tuple of dict
        ``(valence_baths, conduction_baths)`` (``Basis.bath_states``).

    Returns
    -------
    list of (int, int)
    """
    pairs = []
    for baths in bath_states:
        for blocks in baths.values():
            for block in blocks:
                n = len(block)
                if n % 2 != 0:
                    continue
                for k in range(n // 2):
                    pairs.append((block[k], block[k + n // 2]))
    return pairs


def spin_pairs_consistent_with_h(h_op, spin_pairs, n_orb, tol=1e-6):
    r"""Whether the spin operators from ``spin_pairs`` commute with the one-body ``h``.

    Builds the single-particle ``S_z`` and ``S_+`` matrices implied by the ``(dn, up)``
    pairing and checks ``[h, S_z] = [h, S_+] = 0``. If both hold, the spin labelling
    **and** the down↔up pairing are consistent with the Hamiltonian's spin symmetry, so
    the spin operators are physically correct. If either fails (spin-orbit coupling, a
    non-standard orbital ordering, …), the pairing is **not** trustworthy and spin-spin
    observables built from it should be skipped rather than reported wrong.

    Parameters
    ----------
    h_op : ManyBodyOperator or dict
        The Hamiltonian (its one-body part is used).
    spin_pairs : sequence of (int, int)
        ``(dn, up)`` global spin-orbital pairs (impurity + bath).
    n_orb : int
        Total number of spin-orbitals.
    tol : float, optional
        Commutator norm tolerance.

    Returns
    -------
    bool
    """
    h, _, _ = extract_tensors(h_op, n_orb=n_orb, two_body=False)
    sz = np.zeros((n_orb, n_orb), dtype=complex)
    splus = np.zeros((n_orb, n_orb), dtype=complex)
    for dn, up in spin_pairs:
        sz[up, up] += 0.5
        sz[dn, dn] -= 0.5
        splus[up, dn] += 1.0  # S_+ = c†_up c_dn
    return bool(np.linalg.norm(h @ sz - sz @ h) <= tol and np.linalg.norm(h @ splus - splus @ h) <= tol)


def _pairs_from_rotated_splus(orbs, rot, tol=1e-6):
    r"""``(dn, up)`` global pairs from the spherical ``S_+`` rotated to the computational basis.

    ``orbs`` are the global spin-orbital indices of a complete spin-doubled l-shell, listed in
    the SAME order as the rows/columns of ``rot`` (the spherical -> computational rotation).
    The single non-zero in each column of the rotated ``S_+`` marks a ``(down, up)`` partner.

    Returns ``None`` when ``orbs`` is not a complete l-shell, ``rot`` is not sized to it, or the
    rotated ``S_+`` is not a clean spin-diagonal permutation (e.g. spin-orbit coupling mixes the
    spins) -- so the caller can fall back or skip rather than report a wrong pairing.
    """
    n_so = len(orbs)
    shell_l = (n_so // 2 - 1) // 2
    if 2 * (2 * shell_l + 1) != n_so:
        return None
    rot = np.asarray(rot, dtype=complex)
    if rot.shape[0] != n_so:
        return None
    _, _, _, _, sp_m, _ = _single_particle_lsj_matrices(shell_l)
    sp_comp = rot @ sp_m @ rot.conj().T  # S_+ in the computational basis
    downs = {}  # local down index -> local up index
    for j in range(n_so):
        rows = [i for i in range(n_so) if abs(sp_comp[i, j]) > 0.5]
        if len(rows) > 1 or (len(rows) == 1 and abs(abs(sp_comp[rows[0], j]) - 1.0) > 1e-3):
            return None  # not a clean spin-eigen pairing (e.g. SOC)
        if len(rows) == 1:
            downs[j] = rows[0]
    ups = set(downs.values())
    if len(downs) != n_so // 2 or len(ups) != n_so // 2 or (set(downs) & ups):
        return None
    return [(orbs[j], orbs[i]) for j, i in downs.items()]


def _impurity_pairs_per_partition(impurity_orbitals, rot_to_spherical, tol=1e-6):
    r"""Impurity ``(dn, up)`` pairs when each partition is a complete spin-doubled l-shell.

    Each partition of ``impurity_orbitals`` is paired independently from its own rotation
    (``rot_to_spherical[partition]`` when a dict, else the shared matrix). Returns ``None`` if
    any partition is not a spin-doubled l-shell matching its rotation -- e.g. crystal-field
    sub-manifolds (eg / t2g) under a single whole-impurity rotation, which
    :func:`_impurity_pairs_whole_shell` handles instead.
    """
    imp_pairs = []
    for partition, blocks in impurity_orbitals.items():
        orbs = [orb for block in blocks for orb in block]
        rot = rot_to_spherical[partition] if isinstance(rot_to_spherical, dict) else rot_to_spherical
        pairs = _pairs_from_rotated_splus(orbs, rot, tol)
        if pairs is None:
            return None
        imp_pairs.extend(pairs)
    return imp_pairs


def _impurity_pairs_whole_shell(impurity_orbitals, rot_to_spherical, tol=1e-6):
    r"""Impurity ``(dn, up)`` pairs when the impurity is a single l-shell split into manifolds.

    The partitions (crystal-field sub-manifolds such as eg / t2g) are not individually l-shells,
    but their union is one complete l-shell described by a single whole-impurity rotation whose
    local indices run over the *sorted* impurity orbitals (the convention of
    :func:`impurity_block_structure` / ``rot_to_spherical``). The full-shell ``S_+`` then yields
    the pairing across all manifolds at once. Returns ``None`` for a per-partition (dict)
    rotation or when the union is not a clean single l-shell.
    """
    if isinstance(rot_to_spherical, dict):
        return None
    orbs = sorted(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)
    return _pairs_from_rotated_splus(orbs, rot_to_spherical, tol)


def derive_spin_pairs(h_op, impurity_orbitals, rot_to_spherical, n_orb, tol=1e-6):
    r"""Derive the global ``(down, up)`` spin-orbital pairing from the one-body Hamiltonian.

    Geometry-agnostic alternative to the down-then-up index convention of
    :func:`impurity_spin_pairs` / :func:`bath_spin_pairs`, which is only valid in the
    spherical-harmonics representation. It is needed for bath geometries (e.g. the linked
    double-chain / Haverkort bath) where the computational orbital order is not
    down-then-up.

    Two ingredients:

    1. **Impurity** — the pairing is read from the spherical spin-raising operator
       :math:`S_+` rotated to the computational basis via ``rot_to_spherical``: the single
       non-zero in each column marks a ``(down, up)`` partner. This carries no index-order
       assumption. It is tried first per-partition (each partition a complete l-shell,
       :func:`_impurity_pairs_per_partition`) and, failing that, over the whole impurity as
       one l-shell split into crystal-field manifolds (eg / t2g,
       :func:`_impurity_pairs_whole_shell`). If the rotated :math:`S_+` is not a clean
       permutation (spin-orbit coupling mixes the impurity spins) the derivation gives up.
    2. **Bath** — propagated outward along the Hamiltonian's hopping graph. Because the
       one-body ``h`` is spin-blind for a collinear model (``h_up == h_dn``), the spin-down
       and spin-up sectors form identical connectivity blocks. Starting from each impurity
       ``(down, up)`` pair, a simultaneous breadth-first search matches each spin-down bath
       orbital with the structurally identical (same hopping magnitude and on-site energy)
       spin-up bath orbital.

    The result is a *candidate* that must still be confirmed with
    :func:`spin_pairs_consistent_with_h`. Returns ``None`` when the pairing cannot be
    determined unambiguously (spin-orbit coupling, a bath orbital disconnected from the
    impurity, or a structurally ambiguous match).

    Parameters
    ----------
    h_op : ManyBodyOperator or dict
        The Hamiltonian (its one-body part is used).
    impurity_orbitals : dict
        ``Basis.impurity_orbitals`` (``partition -> list of orbital-index blocks``).
    rot_to_spherical : np.ndarray or dict
        The spherical->computational rotation (single matrix or ``{partition: matrix}``).
    n_orb : int
        Total number of spin-orbitals.
    tol : float, optional
        Magnitude tolerance for graph edges and structural matching.

    Returns
    -------
    (imp_pairs, bath_pairs) : tuple of list of (int, int), or None
    """
    h, _, _ = extract_tensors(h_op, n_orb=n_orb, two_body=False)
    impurity_orbs = set(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)

    # --- impurity (dn, up) pairs from the rotated spherical S_+ ---
    # Two derivations, tried in order:
    #  1. Per-partition -- each partition of ``impurity_orbitals`` is itself a complete
    #     spin-doubled l-shell (spherical / single-shell layout), paired from its own
    #     (per-partition or whole) rotation.
    #  2. Whole-shell -- the impurity is grouped into crystal-field sub-manifolds
    #     (eg / t2g) that are NOT individually l-shells, but their union is one complete
    #     l-shell and a single whole-impurity rotation (in sorted-orbital order) is given.
    #     Build S_+ for the full shell and read the pairing across all manifolds at once.
    imp_pairs = _impurity_pairs_per_partition(impurity_orbitals, rot_to_spherical, tol)
    if imp_pairs is None:
        imp_pairs = _impurity_pairs_whole_shell(impurity_orbitals, rot_to_spherical, tol)
    if imp_pairs is None:
        return None

    # --- bath pairs by simultaneous BFS over the hopping graph, seeded by imp_pairs ---
    h_abs = np.abs(h)
    matched = {}
    for dn, up in imp_pairs:
        matched[dn] = up
        matched[up] = dn

    def bath_neighbors(x):
        return [y for y in range(n_orb) if y != x and y not in impurity_orbs and h_abs[x, y] > tol]

    bath_pairs = []
    queue = deque(imp_pairs)
    while queue:
        dn, up = queue.popleft()
        up_candidates = [y for y in bath_neighbors(up) if y not in matched]
        for ndn in bath_neighbors(dn):
            if ndn in matched:
                continue
            cands = [
                nup
                for nup in up_candidates
                if nup not in matched
                and abs(h_abs[dn, ndn] - h_abs[up, nup]) <= tol
                and abs(h[ndn, ndn] - h[nup, nup]) <= tol
            ]
            if len(cands) != 1:
                return None  # no match or ambiguous
            nup = cands[0]
            matched[ndn] = nup
            matched[nup] = ndn
            bath_pairs.append((ndn, nup))
            queue.append((ndn, nup))

    bath_orbs = set(range(n_orb)) - impurity_orbs
    if any(orb not in matched for orb in bath_orbs):
        return None  # some bath orbital disconnected from the impurity
    return imp_pairs, bath_pairs
