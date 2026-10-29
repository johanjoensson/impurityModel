"""
Derivation of impurity/bath spin pairs (the ``(down, up)`` spin-orbital
pairings) consistent with a given one-body Hamiltonian.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

# Local imports
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.observables import _single_particle_lsj_matrices, make_spin_operators
from impurityModel.ed.symmetries import PARITY_TERM_TOL, bath_chains, conserves_sz, extract_tensors

#: Largest ``max |[H, S_+]|`` coefficient, relative to the largest coefficient of ``H``, for which
#: total spin counts as a symmetry of ``H``. The commutator is exact operator algebra, so a symmetry
#: cancels to roundoff and a broken one at the size of the breaking term.
SPIN_SYMMETRY_TOL = 1e-10


def impurity_spin_pairs(impurity_orbitals):
    r"""Return the ``(dn_index, up_index)`` impurity spin-orbital pairs.

    Within each angular-momentum partition of ``impurity_orbitals`` the first half
    of the spin-orbitals are spin-down and the second half spin-up (the basis
    layout), so orbital ``k`` pairs with orbital ``k + n//2``.

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


def _sz_splus_matrices(spin_pairs, n_orb):
    """Single-particle ``S_z`` and ``S_+`` matrices implied by a ``(dn, up)`` pairing."""
    sz = np.zeros((n_orb, n_orb), dtype=complex)
    splus = np.zeros((n_orb, n_orb), dtype=complex)
    for dn, up in spin_pairs:
        sz[up, up] += 0.5
        sz[dn, dn] -= 0.5
        splus[up, dn] += 1.0  # S_+ = c†_up c_dn
    return sz, splus


def spin_pair_consistency(h_op, spin_pairs, n_orb, tol=1e-6):
    r"""The two consistency checks of a ``(dn, up)`` pairing against the one-body ``h``.

    Builds the single-particle ``S_z`` and ``S_+`` matrices implied by the pairing and
    tests the two commutators separately, because they validate two different things:

    - ``[h, S_z] = 0`` validates the spin **labelling** (which orbitals are up vs down,
      consistently between impurity and bath). It holds for any collinear model,
      including a spin-*polarized* one.
    - ``[h, S_+] = 0`` additionally validates the down↔up **pairing** as an SU(2)
      symmetry of ``h``. It fails when spin rotation symmetry is broken — by a wrong
      pairing, by spin-orbit coupling, or *by design* when the spin polarization lives
      in the hybridization function (spin-split bath energies/hoppings).

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
    (sz_ok, splus_ok) : tuple of bool
    """
    h, _, _ = extract_tensors(h_op, n_orb=n_orb, two_body=False)
    sz, splus = _sz_splus_matrices(spin_pairs, n_orb)
    sz_ok = bool(np.linalg.norm(h @ sz - sz @ h) <= tol)
    splus_ok = bool(np.linalg.norm(h @ splus - splus @ h) <= tol)
    return sz_ok, splus_ok


def spin_pairs_consistent_with_h(h_op, spin_pairs, n_orb, tol=1e-6):
    r"""Whether the spin operators from ``spin_pairs`` commute with the one-body ``h``.

    ``True`` iff both checks of :func:`spin_pair_consistency` hold, i.e. the spin
    labelling **and** the down↔up pairing are consistent with the Hamiltonian's full
    SU(2) spin symmetry, so the spin operators are physically correct. If either fails
    (spin-orbit coupling, a non-standard orbital ordering, a spin-polarized bath, …),
    the pairing is **not** trustworthy at this level; see
    :func:`collinear_spin_pairs_consistent_with_h` for the weaker collinear guarantee.

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
    sz_ok, splus_ok = spin_pair_consistency(h_op, spin_pairs, n_orb, tol)
    return sz_ok and splus_ok


def collinear_spin_pairs_consistent_with_h(h_op, imp_pairs, bath_pairs, n_orb, tol=1e-6):
    r"""Whether the pairing is trustworthy for a *collinear spin-polarized bath*.

    Targets the common RSPt setup where all spin polarization lives in the
    hybridization function: the impurity one-body block is spin degenerate while the
    bath energies/hoppings are spin split. Full SU(2) consistency
    (:func:`spin_pairs_consistent_with_h`) necessarily fails there, but two weaker
    statements can still be verified, and together they make the spin *labels* and the
    *impurity* pairing physically correct:

    1. ``[h, S_z] = 0`` for the global labelling — ``h`` never mixes up- and
       down-labelled orbitals, which (because every bath orbital hybridizes with the
       impurity, directly or through a chain) also pins the relative impurity/bath
       labelling.
    2. ``[h_imp, S^imp_z] = [h_imp, S^imp_+] = 0`` for the pairing restricted to the
       *impurity-projected* one-body block — the impurity block is spin degenerate in
       the paired basis, so the impurity down↔up pairing is a genuine symmetry there.
       (The global ``[h, S^imp_+]`` cannot be used: it picks up the spin-split
       hybridization terms even for a correct impurity pairing.)

    What remains unverifiable is the *bath* down↔up pairing: spin-split bath levels
    are different spatial states, so no symmetry fixes their pairing and any choice
    (here: the index convention, i.e. same fit slot) is a modelling approximation for
    the transverse spin operators. Observables built from ``S_z`` alone are exact under
    this check; observables involving the bath ``S_±`` are pairing-dependent and should
    be reported flagged.

    Parameters
    ----------
    h_op : ManyBodyOperator or dict
        The Hamiltonian (its one-body part is used).
    imp_pairs, bath_pairs : sequence of (int, int)
        ``(dn, up)`` global spin-orbital pairs of the impurity and the bath.
    n_orb : int
        Total number of spin-orbitals.
    tol : float, optional
        Commutator norm tolerance.

    Returns
    -------
    bool
    """
    h, _, _ = extract_tensors(h_op, n_orb=n_orb, two_body=False)
    sz, _ = _sz_splus_matrices(list(imp_pairs) + list(bath_pairs), n_orb)
    if np.linalg.norm(h @ sz - sz @ h) > tol:
        return False
    imp_orbs = sorted({orb for pair in imp_pairs for orb in pair})
    pos = {orb: i for i, orb in enumerate(imp_orbs)}
    h_imp = h[np.ix_(imp_orbs, imp_orbs)]
    local_pairs = [(pos[dn], pos[up]) for dn, up in imp_pairs]
    sz_imp, splus_imp = _sz_splus_matrices(local_pairs, len(imp_orbs))
    return bool(
        np.linalg.norm(h_imp @ sz_imp - sz_imp @ h_imp) <= tol
        and np.linalg.norm(h_imp @ splus_imp - splus_imp @ h_imp) <= tol
    )


def _pairs_from_rotated_splus(orbs, rot, tol=1e-6):
    r"""``(dn, up)`` global pairs from the spherical ``S_+`` rotated to the computational basis.

    ``orbs`` are the global spin-orbital indices of a complete spin-doubled l-shell, listed in
    the SAME order as the rows/columns of ``rot`` (the spherical -> computational rotation).
    The single non-zero in each column of the rotated ``S_+`` marks a ``(down, up)`` partner.

    Returns ``None`` when ``orbs`` is not a complete l-shell, ``rot`` is not sized to it, or the
    rotated ``S_+`` is not a clean spin-diagonal permutation (e.g. spin-orbit coupling mixes the
    spins) -- so the caller can fall back or skip rather than report a wrong pairing.
    """
    if rot is None:
        return None
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
        rot = rot_to_spherical.get(partition) if isinstance(rot_to_spherical, dict) else rot_to_spherical
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


def _bath_pairs_structural(h, imp_pairs, impurity_orbs, n_orb, tol=1e-6):
    r"""Bath ``(dn, up)`` pairs matched by equal hopping and on-site energy, or ``None``.

    Propagated outward along the hopping graph of the one-body ``h`` by a simultaneous breadth-first
    search from each impurity ``(dn, up)`` pair: a spin-down bath orbital is matched with the unique
    spin-up bath orbital of the same hopping magnitude and on-site energy. ``None`` when a match is
    missing or ambiguous (a spin-polarized bath has no equal-energy partner) or a bath orbital is
    disconnected from the impurity.
    """
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

    if any(orb not in matched for orb in set(range(n_orb)) - impurity_orbs):
        return None  # some bath orbital disconnected from the impurity
    return bath_pairs


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
    impurity_orbs = {orb for blocks in impurity_orbitals.values() for block in blocks for orb in block}

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

    bath_pairs = _bath_pairs_structural(h, imp_pairs, impurity_orbs, n_orb, tol)
    if bath_pairs is None:
        return None
    return imp_pairs, bath_pairs


def _spin_symmetric(h_op, pairs):
    """Whether ``[h_op, S_+] = 0`` for the spin ladder operator built from ``pairs``."""
    if not pairs:
        return False
    s_plus, _s_minus, _s_z = make_spin_operators(pairs)
    scale = max((abs(v) for v in h_op.values()), default=0.0)
    comm = h_op.commutator(s_plus)
    return max((abs(v) for v in comm.values()), default=0.0) <= SPIN_SYMMETRY_TOL * max(scale, 1.0)


_GRADE_TEXT = {"su2": "SU(2)", "collinear": "collinear", "sz_only": "S_z only", "none": "none"}
_SOURCE_TEXT = {
    "index": "index",
    "rotation": "rotation",
    "matched_energies": "matched energies",
    "chain_position": "chain position",
}


@dataclass(frozen=True)
class SpinLayout:
    r"""The ``(dn, up)`` spin-orbital pairing of the impurity and the bath, and what is verified of it.

    ``grade`` is judged on the full operator, from weakest to strongest:

    - ``"none"``: no pairing makes ``H`` conserve ``2 S_z`` (spin-orbit coupling); the pairs are empty.
    - ``"sz_only"``: ``2 S_z`` is conserved. Nothing else is verified.
    - ``"collinear"``: also the impurity pairing is the physical spin up to a spin-dependent level
      shift (the impurity block's dn and up halves have the same eigenvectors, and the impurity
      interaction commutes with the impurity ``S_+``); ``impurity_split`` is the size of that shift.
      The bath ``S_+`` is a modelling choice.
    - ``"su2"``: also ``[H, S_+] = 0``.

    ``imp_source`` / ``bath_source`` say where the pairs come from: ``"index"`` the dn-then-up
    convention, ``"rotation"`` the rotated spherical ``S_+``, ``"matched_energies"`` equal-energy
    structural matching, ``"chain_position"`` the k-th site of the dn chain with the k-th of the up
    chain. ``rejected`` lists, verbatim, why the other candidates lost.
    """

    imp_pairs: tuple
    bath_pairs: tuple
    grade: str
    imp_source: str
    bath_source: str
    impurity_split: float
    rejected: tuple

    def __post_init__(self):
        for name in ("imp_pairs", "bath_pairs"):
            object.__setattr__(self, name, tuple((int(dn), int(up)) for dn, up in getattr(self, name)))
        object.__setattr__(self, "impurity_split", float(self.impurity_split))
        object.__setattr__(self, "rejected", tuple(str(r) for r in self.rejected))

    @property
    def su2(self):
        return self.grade == "su2"

    @property
    def labelled(self):
        """Whether ``2 S_z`` is a symmetry of ``H`` under this pairing."""
        return self.grade != "none"

    @property
    def pairs(self):
        return self.imp_pairs + self.bath_pairs

    def pairs_source(self):
        """``"impurity pairs: rotation, bath pairs: chain position"`` -- only the halves that are not ``index``."""
        parts = []
        if self.imp_source != "index":
            parts.append(f"impurity pairs: {_SOURCE_TEXT[self.imp_source]}")
        if self.bath_source != "index":
            parts.append(f"bath pairs: {_SOURCE_TEXT[self.bath_source]}")
        return ", ".join(parts)

    def rejected_note(self):
        return "rejected: " + "; ".join(self.rejected) if self.rejected else None

    def provenance_note(self):
        """The ``calc_gs`` log note, or ``None`` when the layout is the plain SU(2) index one."""
        if self.grade == "none":
            head = (
                "spin layout: none -- no dn/up pairing of impurity and bath conserves S_z; <S_imp.S_bath>, "
                "screening and the transverse susceptibility are not reported"
            )
        elif self.grade == "su2" and self.imp_source == "index" and self.bath_source == "index":
            return None
        else:
            imp = f"impurity pairs: {_SOURCE_TEXT[self.imp_source]}"
            if self.impurity_split > 0:
                imp += f" (impurity spin split {self.impurity_split:.4g})"
            bath = f"bath pairs: {_SOURCE_TEXT[self.bath_source]}"
            if self.grade != "su2" and self.bath_pairs:
                bath += " (not verified: transverse bath quantities are flagged)"
            exact = "SU(2) exact" if self.grade == "su2" else "S_z exact"
            head = f"spin layout: {_GRADE_TEXT[self.grade]} -- {exact}; {imp}; {bath}"
        rejected = self.rejected_note()
        return head if rejected is None else head + "\n  " + rejected

    def sisb_note(self):
        """Suffix of the ``<S_imp.S_bath>`` row: which part of it depends on an unverified pairing, or ``None``."""
        if self.grade == "collinear":
            return f"(transverse part depends on the bath dn/up pairing, by {_SOURCE_TEXT[self.bath_source]})"
        if self.grade == "sz_only":
            return (
                "(transverse part depends on the dn/up pairing: "
                f"impurity by {_SOURCE_TEXT[self.imp_source]}, bath by {_SOURCE_TEXT[self.bath_source]})"
            )
        return None


def _invalid_pairs(pairs, orbitals, cover):
    """Why ``pairs`` is not a pairing of ``orbitals`` (``cover``: every orbital must be paired), or ``None``."""
    seen = [o for pair in pairs for o in pair]
    if any(dn == up for dn, up in pairs) or len(set(seen)) != len(seen):
        return "an orbital is paired twice"
    if not set(seen) <= orbitals:
        return "pairs outside the orbital set"
    if cover and set(seen) != orbitals:
        missing = ", ".join(str(o) for o in sorted(orbitals - set(seen)))
        return f"impurity orbitals {missing} unpaired"
    return None


def _impurity_candidates(impurity_orbitals, rot_to_spherical, tol):
    """``[(pairs, source)]`` in order of trust, and the reasons candidates were dropped."""
    imp_orbs = {orb for blocks in impurity_orbitals.values() for block in blocks for orb in block}
    index = impurity_spin_pairs(impurity_orbitals)
    rotation = _impurity_pairs_per_partition(impurity_orbitals, rot_to_spherical, tol)
    if rotation is None:
        rotation = _impurity_pairs_whole_shell(impurity_orbitals, rot_to_spherical, tol)
    rejected, candidates = [], []
    index_bad = _invalid_pairs(index, imp_orbs, True)
    if rotation is not None:
        rotation = sorted((int(dn), int(up)) for dn, up in rotation)
        rot_bad = _invalid_pairs(rotation, imp_orbs, True)
        if rot_bad is not None:
            rejected.append(f"rotation: {rot_bad}")
            rotation = None
    if rotation is not None:
        same = sorted(index) == rotation
        if index_bad is not None:
            rejected.append(f"index: {index_bad}")
        elif not same:
            rejected.append("index: contradicts the rotation pairing")
        candidates.append((rotation, "index" if same else "rotation"))
    if index_bad is None and (rotation is None or sorted(index) != rotation):
        candidates.append((index, "index"))
    elif index_bad is not None and rotation is None:
        rejected.append(f"index: {index_bad}")
    return candidates, rejected


def _bath_pairs_positional(h, imp_pairs, impurity, n_orb, cut):
    """Bath pairs from the k-th site of each dn chain with the k-th of its up chain, or ``(None, reason)``."""
    chains = bath_chains(h, impurity, n_orb, cut)
    up_of, dn_of = {}, {}
    for dn, up in imp_pairs:
        if len(chains[dn]) != len(chains[up]):
            return None, f"chains of {dn} and {up} differ in length ({len(chains[dn])} vs {len(chains[up])})"
        for x, y in zip(chains[dn], chains[up]):
            if up_of.setdefault(x, y) != y or dn_of.setdefault(y, x) != x:
                return None, f"overlapping chains pair bath orbital {x} inconsistently"
    return sorted(up_of.items()), None


def spin_layout(h_op, impurity_orbitals, bath_states, rot_to_spherical, n_orb, tol=1e-6, *, comm=None):
    r"""Resolve the ``(dn, up)`` pairing of impurity and bath and grade how much of it ``H`` verifies.

    Total: always returns a :class:`SpinLayout`. With a ``comm`` the layout is resolved on rank 0
    and broadcast (one collective, unconditional), so every rank holds the same copy whatever
    roundoff its own ``H`` carries; without one it is rank-local and deterministic.

    The impurity pairs come from the rotated spherical ``S_+`` (``rot_to_spherical``) whenever it
    yields a clean pairing, and from the dn-then-up index convention otherwise: a pairing read off
    ``H`` cannot tell which dn orbital belongs with which up orbital once ``h`` is diagonal in the
    solver basis and ``U`` is density-density, but the rotation can. The bath pairs, seeded by the
    impurity pairs, are the first of the index convention, equal-energy structural matching and
    chain position (:func:`~impurityModel.ed.symmetries.bath_chains`) under which ``H`` conserves
    ``2 S_z``. The grade is then judged on the full operator, see :class:`SpinLayout`.

    Parameters
    ----------
    h_op : ManyBodyOperator or dict
        The Hamiltonian.
    impurity_orbitals : dict
        ``partition -> list of orbital-index blocks`` (``Basis.impurity_orbitals``).
    bath_states : tuple
        ``(valence_baths, conduction_baths)`` (``Basis.bath_states``).
    rot_to_spherical : np.ndarray or dict or None
        The spherical -> computational rotation; ``None`` leaves the index convention.
    n_orb : int
        Total number of spin-orbitals.
    tol : float, optional
        Tolerance of the structural matching and of the impurity commutation test.
    comm : MPI communicator, optional
        Resolve on its rank 0 and broadcast.

    Returns
    -------
    SpinLayout
    """
    layout = (
        _spin_layout(h_op, impurity_orbitals, bath_states, rot_to_spherical, n_orb, tol)
        if comm is None or comm.rank == 0
        else None
    )
    return layout if comm is None else comm.bcast(layout, root=0)


def _spin_layout(h_op, impurity_orbitals, bath_states, rot_to_spherical, n_orb, tol):
    if not hasattr(h_op, "commutator"):
        h_op = ManyBodyOperator(dict(h_op))
    h = extract_tensors(h_op, n_orb=n_orb, two_body=False)[0]
    impurity_orbs = {orb for blocks in impurity_orbitals.values() for block in blocks for orb in block}
    impurity = sorted(impurity_orbs)
    scale = max((abs(v) for v in h_op.values()), default=0.0)
    cut = PARITY_TERM_TOL * scale
    candidates, rejected = _impurity_candidates(impurity_orbitals, rot_to_spherical, tol)
    for imp_pairs, imp_source in candidates:
        if not imp_pairs:
            continue
        bath_options = [("index", bath_spin_pairs(bath_states))]
        structural = _bath_pairs_structural(h, imp_pairs, impurity_orbs, n_orb, tol)
        if structural is None:
            rejected.append("matched_energies: no equal-energy partner (spin-polarized bath)")
        else:
            bath_options.append(("matched_energies", structural))
        for bath_source, candidate in bath_options + [("chain_position", None)]:
            bath_pairs = candidate
            if bath_pairs is None:
                bath_pairs, why = _bath_pairs_positional(h, imp_pairs, impurity, n_orb, cut)
                if bath_pairs is None:
                    rejected.append(f"chain_position: {why}")
                    continue
            bad = _invalid_pairs(bath_pairs, set(range(n_orb)) - impurity_orbs, False)
            if bad is not None:
                rejected.append(f"{bath_source}: {bad}")
                continue
            if not conserves_sz(h_op, list(imp_pairs) + list(bath_pairs), PARITY_TERM_TOL):
                rejected.append(f"{bath_source}: does not conserve S_z")
                continue
            grade, split = _grade(h_op, h, imp_pairs, bath_pairs, tol)
            return SpinLayout(imp_pairs, bath_pairs, grade, imp_source, bath_source, split, rejected)
    return SpinLayout((), (), "none", "index", "index", 0.0, rejected)


def _grade(h_op, h, imp_pairs, bath_pairs, tol):
    """``("su2" | "collinear" | "sz_only", impurity_split)`` of a pairing under which ``H`` conserves ``2 S_z``."""
    dn = [d for d, _ in imp_pairs]
    up = [u for _, u in imp_pairs]
    h_dn, h_up = h[np.ix_(dn, dn)], h[np.ix_(up, up)]
    if np.max(np.abs(h[np.ix_(dn, up)]), initial=0.0) > tol:
        return "sz_only", 0.0
    if np.max(np.abs(h_up @ h_dn - h_dn @ h_up), initial=0.0) > tol:
        return "sz_only", 0.0
    imp = set(dn) | set(up)
    interaction = ManyBodyOperator({t: a for t, a in h_op.items() if len(t) == 4 and all(o in imp for o, _ in t)})
    s_plus = make_spin_operators(imp_pairs)[0]
    scale = max((abs(v) for v in h_op.values()), default=0.0)
    comm = interaction.commutator(s_plus)
    if max((abs(v) for v in comm.values()), default=0.0) > SPIN_SYMMETRY_TOL * max(scale, 1.0):
        return "sz_only", 0.0
    if _spin_symmetric(h_op, list(imp_pairs) + list(bath_pairs)):
        return "su2", 0.0
    return "collinear", float(np.max(np.abs(np.linalg.eigvalsh(h_up - h_dn)), initial=0.0))


def resolve_spin_pairs(h_op, impurity_orbitals, bath_states, rot_to_spherical, n_orb, tol=1e-6):
    """Compatibility shim over :func:`spin_layout`: ``(imp_pairs, bath_pairs, pairing_approx)`` or ``None``."""
    layout = spin_layout(h_op, impurity_orbitals, bath_states, rot_to_spherical, n_orb, tol)
    if not layout.labelled:
        return None
    return list(layout.imp_pairs), list(layout.bath_pairs), not layout.su2
