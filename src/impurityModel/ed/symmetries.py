r"""Conserved charges, occupation restrictions, and Green's-function block structure.

This module turns the discovered one-body symmetry algebra (see
:mod:`impurityModel.ed.lie_algebra`) into the objects the solvers consume: conserved
subset charges, occupation-window restrictions (including the ``S_z``-weighted and
frozen-shell flavors), impurity/bath occupation classification, and the impurity and
Green's-function block structures used to deduplicate and sectorize the GF/RIXS solves.

The algebraic primitives (tensor extraction/rotation, symmetry discovery, Cartan
reduction, Casimirs) live in :mod:`impurityModel.ed.lie_algebra` and are re-exported
here for backward compatibility.
"""

from collections import namedtuple

import numpy as np

import impurityModel.ed.product_state_representation as psr
from impurityModel.ed.block_structure import build_block_structure, get_equivalent_orbs
from impurityModel.ed.lie_algebra import (  # noqa: F401  -- most names re-exported for backward compat
    ComponentReduction,
    SymmetryRotationCache,
    apply_reconstructed_casimir,
    cartan_subalgebra,
    component_symmetry_reduction,
    discover_one_body_symmetries,
    discover_rotation,
    expect_reconstructed_casimir,
    extract_tensors,
    hermitian_algebra_basis,
    in_span,
    is_abelian,
    joint_diagonalize,
    rotate_hamiltonian,
    rotate_one_body,
    rotate_two_body,
    structure_constants,
    symmetry_adapted_transformation,
    tensors_to_operator,
    weights_are_01,
)


def _one_body_matrix(op, n_orb=None, h0_matrix=None, must_span=None):
    """One-body matrix of ``op``, guaranteed large enough to index by ``must_span``.

    :func:`lie_algebra.extract_tensors` sizes ``h`` from the operator's own extent
    (``1 + max index``) when ``n_orb`` is ``None``. Every caller below then slices that
    matrix with a *caller-supplied* orbital list, so an operator that simply does not
    touch its highest-numbered orbital — a genuinely decoupled orbital at zero on-site
    energy, which carries no term at all — yields a matrix too small to index. Widen the
    inferred extent to cover the orbitals the caller intends to address.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The operator to extract from (ignored when ``h0_matrix`` is given).
    n_orb : int, optional
        Explicit spin-orbital count; inferred (and widened) when ``None``.
    h0_matrix : ndarray, optional
        Pre-extracted one-body matrix; returned as-is.
    must_span : sequence of int, optional
        Orbital indices the returned matrix must be indexable by.

    Returns
    -------
    ndarray
    """
    if h0_matrix is not None:
        return h0_matrix
    if n_orb is None and must_span is not None:
        span = list(must_span)
        if span:
            terms = op.to_dict() if hasattr(op, "to_dict") else dict(op)
            op_extent = 1 + max((idx for factors in terms for (idx, _) in factors), default=-1)
            n_orb = max(op_extent, 1 + max(span))
    return extract_tensors(op, n_orb=n_orb, two_body=False)[0]


def conserved_subset_charges(op, n_orb=None, tol=1e-9):
    r"""Find the orbital subsets whose total occupation is conserved by the **full** ``op``.

    A subset charge :math:`N_S = \sum_{i\in S} n_i` commutes with ``H`` iff every term
    is *block-balanced* in ``S`` — i.e. each term creates as many electrons in ``S`` as
    it annihilates. This returns the **finest** partition of the orbitals for which
    every block is conserved, found by union-find:

    - a one-body term ``c†_i c_j`` (``i≠j``) forces ``i`` and ``j`` into one block
      (otherwise it moves one electron across the boundary);
    - a two-body term is scanned for per-block imbalance; any blocks it imbalances are
      merged, iterated to a fixed point.

    Unlike the one-body commutant (Phase 2, which sees only ``h``), this accounts for
    the two-body interaction, so the returned charges are conserved by the interacting
    Hamiltonian. Each is a ``{0,1}``-weight (subset-occupation) charge directly mappable
    to a ``Basis`` restriction (Phase 3).

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The Hamiltonian (1- and 2-body, number-conserving).
    n_orb : int, optional
        Number of spin-orbitals (inferred from ``op`` if ``None``).
    tol : float, optional
        Terms with ``|amp| <= tol`` are ignored.

    Returns
    -------
    list of frozenset of int
        The conserved orbital subsets (a partition of ``range(n_orb)``), sorted by
        smallest orbital.
    """
    from collections import Counter

    terms = op.to_dict() if hasattr(op, "to_dict") else dict(op)
    if n_orb is None:
        n_orb = 1 + max((idx for factors in terms for (idx, _) in factors), default=-1)

    parent = list(range(n_orb))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            return True
        return False

    parsed = []
    for factors, amp in terms.items():
        if abs(amp) <= tol:
            continue
        ladder = tuple(c for (_, c) in factors)
        idx = [i for (i, _) in factors]
        if ladder == ("c", "a") and idx[0] != idx[1]:
            union(idx[0], idx[1])
        elif ladder == ("c", "c", "a", "a"):
            parsed.append((idx[:2], idx[2:]))  # (creators, annihilators)

    changed = True
    while changed:
        changed = False
        for creators, annihilators in parsed:
            balance = Counter()
            for orb in creators:
                balance[find(orb)] += 1
            for orb in annihilators:
                balance[find(orb)] -= 1
            imbalanced = [block for block, net in balance.items() if net != 0]
            for block in imbalanced[1:]:
                if union(block, imbalanced[0]):
                    changed = True

    components = {}
    for orb in range(n_orb):
        components.setdefault(find(orb), []).append(orb)
    return sorted((frozenset(orbs) for orbs in components.values()), key=min)


def restrictions_from_charges(charges, occupations, slack=0):
    r"""Map conserved subset charges + target occupations to a ``Basis`` restriction dict.

    Parameters
    ----------
    charges : sequence of frozenset of int
        Conserved orbital subsets (e.g. from :func:`conserved_subset_charges`).
    occupations : sequence of int
        Target electron count in each subset (e.g. counted on the ground state from the
        Phase 3.0 pre-scan).
    slack : int, optional
        Allow ``occ ± slack`` electrons (default 0 = a strict sector). Use ``slack=1``
        to also admit the immediate neighbour sectors (``N ± 1`` etc.).

    Returns
    -------
    dict of frozenset of int to (int, int)
        ``Basis.restrictions``-format mapping ``subset -> (min, max)``. Subsets of size
        1 whose occupation is fully pinned still produce a valid bound.
    """
    restrictions = {}
    for subset, occ in zip(charges, occupations):
        lo = max(0, occ - slack)
        hi = min(len(subset), occ + slack)
        restrictions[frozenset(subset)] = (lo, hi)
    return restrictions


def subset_occupations(charges, occupied_orbitals):
    """Count electrons of a Slater determinant (set of occupied orbitals) in each subset."""
    occupied = set(occupied_orbitals)
    return [len(subset & occupied) for subset in charges]


def group_orbitals_by_charges(op, impurity_orbitals, valence_orbitals, conduction_orbitals, n_orb=None):
    r"""Group flat impurity / valence / conduction orbital lists into conserved-charge blocks.

    The many-body ``Basis`` is built from ``{group: [orbital-block, ...]}`` dictionaries whose
    keys pair each impurity block with its valence and conduction baths. Given **flat** orbital
    lists, this reconstructs that grouping from the **full-Hamiltonian** conserved charges
    (:func:`conserved_subset_charges`): each conserved subset that contains impurity orbitals
    becomes one group, holding the impurity, valence and conduction orbitals that fall in it.
    Because a bath orbital is fused into the conserved charge of the impurity orbital it
    hybridises with, each bath lands in the group of the impurity it couples to.

    Groups are keyed by a 0-based integer sorted by their smallest impurity orbital, giving a
    deterministic keying that ``nominal_occ`` / ``mixed_valence`` can match.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The full Hamiltonian (1- and 2-body).
    impurity_orbitals, valence_orbitals, conduction_orbitals : sequence of int
        Flat spin-orbital index lists.
    n_orb : int, optional
        Number of spin-orbitals (inferred if ``None``).

    Returns
    -------
    impurity_orbitals : dict[int, list[list[int]]]
        ``{group: [impurity orbital block]}``.
    bath_states : tuple(dict, dict)
        ``(valence_baths, conduction_baths)``, each ``{group: [bath orbital block]}`` with the
        same keys as ``impurity_orbitals``.
    """
    imp_set = set(impurity_orbitals)
    val_set = set(valence_orbitals)
    con_set = set(conduction_orbitals)

    charges = conserved_subset_charges(op, n_orb=n_orb)
    groups = sorted((c for c in charges if c & imp_set), key=lambda c: min(c & imp_set))

    impurity_dict, valence_dict, conduction_dict = {}, {}, {}
    for g, subset in enumerate(groups):
        impurity_dict[g] = [sorted(subset & imp_set)]
        valence_dict[g] = [sorted(subset & val_set)]
        conduction_dict[g] = [sorted(subset & con_set)]
    return impurity_dict, (valence_dict, conduction_dict)


def group_orbitals_by_blocks(
    op, impurity_orbitals, valence_orbitals, conduction_orbitals, block_structure, n_orb=None, h0_matrix=None
):
    r"""Group flat impurity / valence / conduction orbital lists into orbital-symmetry blocks.

    The many-body ``Basis`` is built from ``{group: [orbital-block, ...]}`` dictionaries that
    pair each impurity block with its valence and conduction baths. This variant derives the
    grouping from the **impurity block structure** (:func:`impurity_block_structure`) rather
    than the conserved charges of the full Hamiltonian: each inequivalent block (with all its
    equivalent partners, e.g. both spins of an ``eg`` / ``t2g`` manifold) becomes one group.

    Crucially, because :func:`block_structure.get_equivalent_orbs` folds the spin-degenerate
    (``identical``) partners into one block, **each group holds both spins of its manifold**.
    Grouping this way therefore does *not* pin ``S_z`` (unlike
    :func:`group_orbitals_by_charges`, which splits the impurity into per-spin conserved
    charges and, with a fixed per-group occupation, confines the basis to one ``S_z`` sector
    and destroys spin-multiplet degeneracy). The occupation window that ties the groups
    together is applied to the impurity *as a whole* by the restriction machinery, not per
    group, so the manifolds ``eg``/``t2g`` are free to redistribute charge without leaking it.

    Each bath orbital is placed in the group of the impurity orbital it couples to most
    strongly (largest ``|h[b, o]| + |h[o, b]|``).

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The full Hamiltonian (1- and 2-body); only its one-body part is used here.
    impurity_orbitals, valence_orbitals, conduction_orbitals : sequence of int
        Flat spin-orbital index lists.
    block_structure : BlockStructure
        The impurity block structure (blocks in the **local** ``0 .. n_imp-1`` convention over
        the sorted ``impurity_orbitals``), e.g. from :func:`impurity_block_structure`.
    n_orb : int, optional
        Number of spin-orbitals (inferred if ``None``).
    h0_matrix : ndarray, optional
        Pre-extracted one-body matrix of ``op`` (``extract_tensors(op, two_body=False)[0]``);
        pass it to avoid re-walking the operator and re-allocating the dense matrix.

    Returns
    -------
    impurity_orbitals : dict[int, list[list[int]]]
        ``{group: [impurity orbital block]}``.
    bath_states : tuple(dict, dict)
        ``(valence_baths, conduction_baths)``, each ``{group: [bath orbital block]}`` with the
        same keys as ``impurity_orbitals``.
    """
    imp = sorted(impurity_orbitals)
    val_set = set(valence_orbitals)
    con_set = set(conduction_orbitals)
    h = _one_body_matrix(op, n_orb=n_orb, h0_matrix=h0_matrix, must_span=imp)

    # get_equivalent_orbs returns local (0..n_imp-1) indices per inequivalent block; map back
    # to global spin-orbital indices via the sorted impurity list.
    manifolds = [sorted(imp[o] for o in local_orbs) for local_orbs in get_equivalent_orbs(block_structure)]
    manifolds = sorted(manifolds, key=min)
    imp_to_group = {orb: g for g, orbs in enumerate(manifolds) for orb in orbs}

    impurity_dict = {g: [list(orbs)] for g, orbs in enumerate(manifolds)}
    valence_dict = {g: [] for g in range(len(manifolds))}
    conduction_dict = {g: [] for g in range(len(manifolds))}
    for b in sorted(val_set | con_set):
        couplings = [abs(h[b, o]) + abs(h[o, b]) for o in imp]
        g = imp_to_group[imp[int(np.argmax(couplings))]]
        (valence_dict if b in val_set else conduction_dict)[g].append(b)
    valence_dict = {g: [sorted(v)] for g, v in valence_dict.items()}
    conduction_dict = {g: [sorted(c)] for g, c in conduction_dict.items()}
    return impurity_dict, (valence_dict, conduction_dict)


def classify_bath_occupation(op, impurity_orbitals, n_orb=None, h0_matrix=None):
    r"""Split the bath orbitals into initially-occupied (valence) and empty (conduction) sets.

    The bath orbitals are all spin-orbitals of ``op`` that are **not** impurity orbitals; each is
    classified by its one-body on-site energy ``h[o, o]``: baths below the Fermi level
    (``h[o, o] < 0``) sit filled in the nominal configuration (**valence**, initially occupied),
    the rest are empty (**conduction**, initially empty). This is the same Fermi-level-zero
    convention used by the tooling that assembled h0, so such a Hamiltonian
    reproduces its valence/conduction split without the caller passing it.

    The on-site energies are basis-independent for the bath here: the symmetry-adapted rotation
    (:func:`impurity_symmetry_rotation`) acts only on the impurity block, leaving the bath diagonal
    unchanged, so this may be called on either the input or the solver-basis Hamiltonian.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The Hamiltonian (only its one-body diagonal on the bath orbitals is used).
    impurity_orbitals : sequence of int
        The impurity spin-orbital indices; everything else in ``op`` is treated as bath.
    n_orb : int, optional
        Number of spin-orbitals (inferred if ``None``).
    h0_matrix : ndarray, optional
        Pre-extracted one-body matrix of ``op``; pass it to avoid re-walking the operator.

    Returns
    -------
    valence_orbitals : list[int]
        Initially-occupied bath orbital indices (``h[o, o] < 0``), sorted.
    conduction_orbitals : list[int]
        Initially-empty bath orbital indices (``h[o, o] >= 0``), sorted.
    """
    h = _one_body_matrix(op, n_orb=n_orb, h0_matrix=h0_matrix, must_span=impurity_orbitals)
    imp_set = set(impurity_orbitals)
    bath = [o for o in range(h.shape[0]) if o not in imp_set]
    valence = [o for o in bath if h[o, o].real < 0]
    conduction = [o for o in bath if h[o, o].real >= 0]
    return valence, conduction


def weighted_restriction(weights, target, slack=0):
    r"""Build one weighted-sum restriction ``({orbital: int weight}, (q_min, q_max))``.

    This is the format consumed by
    ``ManyBodyOperator.set_weighted_restrictions``: a determinant passes iff
    ``q_min <= sum_i weights[i] * n_i <= q_max``. Weights and ``target`` must be
    integers (scale a fractional charge — e.g. ``S_z`` — to integers first; see
    :func:`sz_weighted_restriction`). This is the Phase-6 counterpart of
    :func:`restrictions_from_charges` for charges that are **not** ``{0,1}`` subset
    occupations (``S_z``, ``L_z``, general ``Σ wᵢ nᵢ``).

    Parameters
    ----------
    weights : dict
        ``{orbital_index: integer weight}``.
    target : int
        The conserved weighted-sum value.
    slack : int, optional
        Allow ``target ± slack`` (default 0 = a strict sector).

    Returns
    -------
    tuple
        ``(weights, (target - slack, target + slack))``.
    """
    return (dict(weights), (int(target) - slack, int(target) + slack))


def widen_weighted_restrictions(weighted_restrictions, extra=None):
    r"""Widen weighted-restriction bounds so a single addition/removal stays in-bounds.

    A Green's-function transition operator ``c_j^\dagger`` / ``c_j`` shifts a weighted
    charge by ``±w_j``, so the excited sector is ``q_ψ ± w_j`` — the ground-state (tight)
    weighted restriction would wrongly filter it out. This widens each restriction's
    ``[q_min, q_max]`` by the maximum single-orbital weight magnitude (or ``extra``), so
    every single-orbital excitation is admitted while still confining the basis.

    Parameters
    ----------
    weighted_restrictions : list of (dict, (int, int)) or None
        Weighted restrictions (see :func:`weighted_restriction`).
    extra : int, optional
        Amount to widen by. Default: ``max |weight|`` of each restriction.

    Returns
    -------
    list of (dict, (int, int)) or None
        Widened restrictions (or the input unchanged if ``None``/empty).
    """
    if not weighted_restrictions:
        return weighted_restrictions
    widened = []
    for weights, (q_min, q_max) in weighted_restrictions:
        by = extra if extra is not None else max((abs(w) for w in weights.values()), default=0)
        widened.append((dict(weights), (q_min - by, q_max + by)))
    return widened


def sz_weighted_restriction(spin_pairs, two_sz_target, slack=0):
    r"""Auto-generate the ``S_z`` weighted restriction from ``(dn, up)`` spin pairs.

    Uses the integer-weight form ``2 S_z = Σ (n_up - n_dn)`` (weight ``+1`` on each up
    orbital, ``-1`` on each down orbital), so ``two_sz_target = 2 * S_z`` is an integer.
    The spin pairs should come from
    ``spin_pairs.impurity_spin_pairs`` / ``spin_pairs.bath_spin_pairs`` after
    ``spin_pairs.spin_pairs_consistent_with_h`` validates them.

    Parameters
    ----------
    spin_pairs : sequence of (int, int)
        ``(dn, up)`` spin-orbital index pairs.
    two_sz_target : int
        Target value of ``2 S_z`` (``N_up - N_dn``).
    slack : int, optional
        Allow ``two_sz_target ± slack``.

    Returns
    -------
    tuple
        A weighted restriction (see :func:`weighted_restriction`).
    """
    weights = {}
    for dn, up in spin_pairs:
        weights[up] = 1
        weights[dn] = -1
    return weighted_restriction(weights, two_sz_target, slack)


def auto_block_structure(op, n_orb=None, orbitals=None):
    r"""Auto-derive a full ``BlockStructure`` from the one-body Hamiltonian.

    Replaces a hand-coded ``block_structure``: extracts the one-body tensor and runs the
    existing :func:`block_structure.build_block_structure` on it, which returns both the
    orbital blocks (connectivity / symmetry sectors) **and** the equivalences
    (``identical_blocks``, ``transposed_blocks``, ``particle_hole_blocks``, …) used to
    skip redundant transition-operator evaluations in the spectra loop (Phase 4.2).

    .. warning::

       This inspects **only the one-body sub-block** ``h[orbitals, orbitals]``. When
       ``orbitals`` is the impurity set, the resulting partition can be **strictly finer**
       than the interacting Green's-function block structure: two impurity orbitals with
       ``h_ij = 0`` still have a nonzero ``G_ij`` if they couple through a shared bath
       orbital (bath-mediated hybridization) or a two-body term. Use
       :func:`impurity_block_structure` (which folds in the full-Hamiltonian conserved
       charges) for the GF / self-energy paths; keep this only where the one-body
       connectivity is known to be complete.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The Hamiltonian (only its one-body part is used).
    n_orb : int, optional
        Number of spin-orbitals (inferred if ``None``).
    orbitals : sequence of int, optional
        Restrict to this sub-block (e.g. the impurity orbitals) before analysing.

    Returns
    -------
    BlockStructure
    """
    h = _one_body_matrix(op, n_orb=n_orb, must_span=orbitals)
    if orbitals is not None:
        h = h[np.ix_(list(orbitals), list(orbitals))]
    return build_block_structure(None, mat=h)


def impurity_block_structure(op, impurity_orbitals, n_orb=None, h0_matrix=None):
    r"""Impurity ``BlockStructure`` consistent with the interacting Green's function.

    The GF / self-energy paths partition the impurity orbitals into blocks and compute one
    block-Lanczos Green's function per inequivalent block. For that partition to be
    *correct*, ``G_ij`` must be zero for every pair ``(i, j)`` in different blocks, i.e. ``i``
    and ``j`` must transform under inequivalent irreps of the symmetry group of the **full**
    Hamiltonian.

    The partition is derived from the **hybridization-dressed** impurity one-body matrix

    .. math:: M = h_{\mathrm{imp}} + V^\dagger V,

    where ``h_imp = h[imp, imp]`` is the impurity crystal field and ``V = h[bath, imp]`` the
    impurity-bath hopping, so ``V†V`` is the (first-moment) bath-mediated coupling. Running
    :func:`block_structure.build_block_structure` on ``M`` gives both the orbital blocks and
    the value-based equivalences (``identical`` / ``transposed`` / ``particle_hole``) used to
    skip redundant Green's functions. Using ``M`` rather than ``h[imp, imp]`` alone is what
    makes the partition correct:

    * ``V†V`` **merges bath-mediated pairs**: two impurity orbitals with ``h_ij = 0`` but a
      common bath neighbour acquire ``(V†V)_{ij} \neq 0`` and are grouped together — the case
      :func:`auto_block_structure` gets wrong (strictly too fine).
    * ``V†V`` carries the **symmetry cancellation**, so it does *not* over-merge: for a bath
      that respects the impurity symmetry, ``(V†V)`` between different irreps
      (e.g. ``eg``/``t2g``) vanishes even though individual ``V`` elements do not. This avoids
      the collapse that abelian conserved charges of the two-body ``U`` would cause (which
      would fuse a correlated shell into one block per spin).

    This is correct for impurity models because the local Coulomb interaction is rotationally
    invariant — it carries higher symmetry than the one-body part, so ``U`` never couples
    orbitals that the one-body + hybridization symmetry keeps separate.

    In a symmetry-adapted single-particle basis (impurity one-body block diagonalised; see
    :func:`impurity_symmetry_rotation`) ``M`` is diagonal and this yields the maximally-split
    structure — for a cubic d-shell, ten ``1x1`` blocks with the ``eg`` and ``t2g`` orbitals
    detected as *identical* (one inequivalent Green's function per irrep and spin).

    Blocks are returned in the **local** impurity index convention (``0 .. n_imp-1`` over the
    sorted ``impurity_orbitals``), matching what
    :func:`greens_function.build_full_greens_function` and ``selfenergy.get_sigma`` expect.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The full Hamiltonian (only its one-body part — impurity, bath, and hybridization — is
        used).
    impurity_orbitals : sequence of int
        The impurity spin-orbital indices.
    n_orb : int, optional
        Number of spin-orbitals (inferred if ``None``).
    h0_matrix : ndarray, optional
        Pre-extracted one-body matrix of ``op``; pass it to avoid re-walking the operator.

    Returns
    -------
    BlockStructure
    """
    imp = sorted(impurity_orbitals)
    imp_set = set(imp)
    h = _one_body_matrix(op, n_orb=n_orb, h0_matrix=h0_matrix, must_span=imp)
    n = h.shape[0]
    bath = [o for o in range(n) if o not in imp_set]

    m = h[np.ix_(imp, imp)]
    if bath:
        v = h[np.ix_(bath, imp)]  # (n_bath, n_imp) impurity-bath hopping
        m = m + v.conj().T @ v  # add the bath-mediated (first-moment) coupling
    return build_block_structure(None, mat=m)


def impurity_symmetry_rotation(op, impurity_orbitals, n_orb=None, h0_matrix=None):
    r"""Full-space rotation that diagonalises the impurity one-body block (crystal field / SOC).

    Returns the unitary that takes the impurity single-particle Hamiltonian ``h[imp, imp]`` to
    its eigenbasis, obtained by a plain Hermitian eigendecomposition. This is the physical
    single-particle basis: without spin-orbit coupling it is the crystal-field basis (e.g.
    cubic ``eg`` / ``t2g``), and with SOC it is the ``j, m_j`` basis. Degenerate eigenvalues
    stay degenerate (adjacent, equal), so the impurity Green's-function block structure
    (:func:`impurity_block_structure`) collapses to its finest form with the degenerate
    orbitals detected as identical.

    .. note::

       Use a straight eigendecomposition, **not** ``discover_one_body_symmetries`` +
       ``joint_diagonalize``: a degenerate ``h_imp`` (e.g. six-fold ``t2g``xspin, four-fold
       ``eg``xspin) has a huge ``U(6)xU(4)`` one-body commutant, and jointly diagonalising a
       *random* element of it picks an arbitrary spin/orbital-scrambling basis that densifies
       the (spherical-harmonic-sparse) Coulomb interaction ~7x. Diagonalising the physical
       ``h_imp`` keeps that basis (and hence ``U``) as sparse as the crystal field allows.

    .. warning::

       With SOC the ``j, m_j`` eigenbasis genuinely densifies the Coulomb tensor (the Gaunt
       sparsity holds only in the spherical spin-orbital basis), so rotating into it is not
       free — weigh the finer block structure against the denser matvec before enabling it.

    The rotation acts **only within the impurity block** (identity on the bath), matching the
    existing ``rot_to_spherical`` convention (``selfenergy`` uses ``eye(n_imp)``): the bath
    layout and the impurity-bath hybridization ``get_hcorr_v_hbath`` consumes are preserved,
    the hybridization is simply expressed in the rotated impurity basis.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The Hamiltonian (only its one-body impurity block is used to build the rotation).
    impurity_orbitals : sequence of int
        The impurity spin-orbital indices.
    n_orb : int, optional
        Number of spin-orbitals (inferred if ``None``).
    h0_matrix : ndarray, optional
        Pre-extracted one-body matrix of ``op``; pass it to avoid re-walking the operator.

    Returns
    -------
    W : ndarray, shape (n_orb, n_orb)
        The full-space unitary (``u_imp`` embedded on the impurity orbitals, identity on the
        bath). Rotate the Hamiltonian with :func:`rotate_hamiltonian`.
    u_imp : ndarray, shape (n_imp, n_imp)
        The impurity-block rotation, in the sorted-``impurity_orbitals`` order (columns are the
        eigenvectors of ``h[imp, imp]``).
    """
    imp = sorted(impurity_orbitals)
    h = _one_body_matrix(op, n_orb=n_orb, h0_matrix=h0_matrix, must_span=imp)
    n = h.shape[0]
    h_imp = h[np.ix_(imp, imp)]
    h_imp = 0.5 * (h_imp + h_imp.conj().T)  # symmetrise away round-off before eigh

    # Block-diagonalize h_imp first to prevent eigh from unnecessarily scrambling
    # uncoupled degenerate orbitals (which densifies the Coulomb tensor).
    import scipy.sparse as sp

    mask = np.abs(h_imp) > 1e-10
    n_components, labels = sp.csgraph.connected_components(mask, directed=False)

    u_imp = np.zeros_like(h_imp, dtype=complex)
    for comp in range(n_components):
        idx = np.where(labels == comp)[0]
        sub_h = h_imp[np.ix_(idx, idx)]
        _, sub_u = np.linalg.eigh(sub_h)
        u_imp[np.ix_(idx, idx)] = sub_u

    W = np.eye(n, dtype=complex)
    W[np.ix_(imp, imp)] = u_imp
    return W, u_imp


def gf_sector_restrictions(charges, gs_occupations, orbital, kind, slack=0):
    r"""Conserved-charge restrictions for an addition/removal Green's-function run.

    The GF :math:`G_{jj}(\omega)` has an **addition** part (``c_j^\dagger|\psi\rangle``,
    sector ``q_ψ + w_j``) and a **removal** part (``c_j|\psi\rangle``, sector
    ``q_ψ - w_j``). They live in *different* sectors, so each Lanczos run can be confined
    to its own sector before it starts (`tOp.set_restrictions`), preventing
    Hilbert-space explosion — Phase 4.3. The conserved charge containing ``orbital``
    shifts by ``+1`` (addition) or ``-1`` (removal); all other charges stay at the
    ground-state value.

    Parameters
    ----------
    charges : sequence of frozenset of int
        Conserved subset charges (from :func:`conserved_subset_charges`).
    gs_occupations : sequence of int
        Ground-state occupation of each charge (from :func:`measure_conserved_charges`).
    orbital : int
        The Green's-function orbital ``j``.
    kind : {"addition", "removal"}
        Whether ``c_j^\dagger`` (addition) or ``c_j`` (removal) acts.
    slack : int, optional
        Neighbour-sector slack passed to :func:`restrictions_from_charges`.

    Returns
    -------
    dict of frozenset to (int, int)
        ``Basis.restrictions`` confining the run to the target sector.
    """
    if kind not in ("addition", "removal"):
        raise ValueError(f"kind must be 'addition' or 'removal', got {kind!r}")
    delta = 1 if kind == "addition" else -1
    occupations = list(gs_occupations)
    for k, subset in enumerate(charges):
        if orbital in subset:
            occupations[k] += delta
            break
    else:
        raise ValueError(f"orbital {orbital} is not in any conserved charge subset")
    return restrictions_from_charges(charges, occupations, slack=slack)


def operator_charge_shift(op, charges, tol=1e-9):
    r"""Net occupation change each conserved subset undergoes under a transition operator.

    A transition operator :math:`\hat{T} = \sum T_{...} \hat{c}^\dagger\cdots\hat{c}\cdots`
    maps a state of definite conserved-charge signature into a *definite* target sector
    **only if every term shifts every conserved subset by the same amount** (e.g. a dipole
    ``c_i^\dagger c_j`` with ``i`` in the d-shell subset and ``j`` in the core-p subset shifts
    ``(d:+1, p:-1)`` for every term). When that holds, ``T|\psi\rangle`` lives in one sector and
    the ensuing Lanczos can be confined to it (see :func:`transition_sector_restrictions`).

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The transition operator.
    charges : sequence of frozenset of int
        Conserved subset charges (from :func:`conserved_subset_charges` on the **full**
        Hamiltonian -- the charges that are actually preserved during the Lanczos).
    tol : float, optional
        Terms with ``|amp| <= tol`` are ignored.

    Returns
    -------
    list of int or None
        The per-subset occupation shift (aligned with ``charges``) if every non-negligible
        term induces the *same* shift; ``None`` if the operator does not have a definite
        sector (its terms disagree), i.e. it cannot be sector-confined.
    """
    terms = op.to_dict() if hasattr(op, "to_dict") else dict(op)
    subset_of = {}
    for k, subset in enumerate(charges):
        for orb in subset:
            subset_of[orb] = k

    reference = None
    for factors, amp in terms.items():
        if abs(amp) <= tol:
            continue
        shift = [0] * len(charges)
        for idx, action in factors:
            k = subset_of.get(idx)
            if k is None:
                # Orbital outside every conserved subset -> cannot reason about the sector.
                return None
            shift[k] += 1 if action == "c" else -1
        if reference is None:
            reference = shift
        elif shift != reference:
            return None
    return reference


def transition_sector_restrictions(charges, gs_occupations, op, slack=0, tol=1e-9):
    r"""Conserved-charge restrictions confining ``op|\psi\rangle`` to its target sector.

    Generalises :func:`gf_sector_restrictions` from a single ladder operator to an arbitrary
    (many-term) transition operator. The seed ``T|\psi\rangle`` occupies the sector
    ``q_ψ + Δq`` where ``Δq`` is the common per-subset shift of ``T``
    (:func:`operator_charge_shift`). Because the Hamiltonian preserves every conserved subset,
    the whole Krylov space stays in that sector, so the excited basis can be pinned to it --
    pruning determinants that per-shell occupation windows would otherwise admit.

    Parameters
    ----------
    charges : sequence of frozenset of int
        Conserved subset charges of the **full** Hamiltonian.
    gs_occupations : sequence of int
        Ground-state occupation of each charge (from :func:`measure_conserved_charges`).
    op : ManyBodyOperator or dict
        The transition operator.
    slack : int, optional
        Neighbour-sector slack passed to :func:`restrictions_from_charges`.
    tol : float, optional
        Amplitude cutoff forwarded to :func:`operator_charge_shift`.

    Returns
    -------
    dict of frozenset to (int, int) or None
        ``Basis.restrictions`` confining the run to the target sector, or ``None`` when the
        operator has no definite sector (fall back to the occupation-only restrictions).
    """
    shift = operator_charge_shift(op, charges, tol=tol)
    if shift is None:
        return None
    occupations = [g + s for g, s in zip(gs_occupations, shift)]
    if any(o < 0 for o in occupations):
        return None
    return restrictions_from_charges(charges, occupations, slack=slack)


def green_function_block_structure(op, n_orb=None):
    r"""Orbital blocks of the one-body Green's function implied by the symmetry of ``op``.

    The retarded GF :math:`G_{ij}(\omega) = \langle\psi| c_i (\omega - H + E)^{-1}
    c_j^\dagger |\psi\rangle + (\text{removal})` is **symmetry-forbidden** (zero for all
    :math:`\omega`) unless orbitals ``i`` and ``j`` carry the same conserved-charge
    signature: ``c_j^\dagger|\psi\rangle`` lives in the sector ``q_ψ + w_j`` and
    ``c_i^\dagger|\psi\rangle`` in ``q_ψ + w_i``, and since :math:`H` preserves every
    conserved charge, *every* moment ``<ψ|c_i H^n c_j†|ψ>`` vanishes when the sectors
    differ. With subset charges that means ``i`` and ``j`` must lie in the same conserved
    subset, so the GF block structure is exactly :func:`conserved_subset_charges`.

    This replaces a hand-coded ``block_structure``: pairs across different returned blocks
    need not be computed (Phase 4.1).

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The Hamiltonian (1- and 2-body).
    n_orb : int, optional
        Number of spin-orbitals (inferred if ``None``).

    Returns
    -------
    list of frozenset of int
        The GF orbital blocks; ``G_ij`` can be nonzero only within a block.
    """
    return conserved_subset_charges(op, n_orb=n_orb)


def green_function_allowed_mask(op, n_orb):
    """Boolean ``(n_orb, n_orb)`` mask: ``True`` where ``G_ij`` is symmetry-allowed.

    ``mask[i, j]`` is ``True`` iff ``i`` and ``j`` are in the same
    :func:`green_function_block_structure` block.
    """
    blocks = green_function_block_structure(op, n_orb=n_orb)
    label = np.full(n_orb, -1, dtype=int)
    for b, block in enumerate(blocks):
        for orb in block:
            label[orb] = b
    return label[:, None] == label[None, :]


ImpurityBlockConsistency = namedtuple(
    "ImpurityBlockConsistency",
    ["impurity_only_blocks", "gf_blocks", "consistent", "missing_pairs"],
)


def impurity_gf_block_consistency(op, impurity_orbitals, n_orb=None):
    r"""Compare the impurity-only block structure against the true Green's-function blocks.

    The spectra / self-energy paths historically derived their Green's-function block
    structure from :func:`auto_block_structure` with ``orbitals=impurity_orbitals``, which
    inspects **only** the impurity sub-block ``h[imp, imp]`` of the one-body Hamiltonian.
    That can be *strictly too fine*: two impurity orbitals with ``h_ij = 0`` still have a
    nonzero ``G_ij`` when they are coupled through a **shared bath orbital** (bath-mediated
    hybridization ``Δ_ij(ω) ≠ 0``). The correct GF block structure is that of the
    hybridization-dressed impurity matrix ``h[imp, imp] + V†V`` (``V = h[bath, imp]``) — see
    :func:`impurity_block_structure` — whose ``V†V`` term supplies the bath-mediated coupling
    (with its symmetry cancellation) that ``h[imp, imp]`` alone misses.

    This diagnostic computes both partitions (as sets of *global* orbital indices) and reports
    whether the impurity-only structure reproduces the dressed one, listing every impurity
    pair ``(i, j)`` that the impurity-only structure wrongly separates (same dressed block,
    different impurity-only block).

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The full Hamiltonian (1- and 2-body).
    impurity_orbitals : sequence of int
        The impurity spin-orbital indices.
    n_orb : int, optional
        Number of spin-orbitals (inferred from ``op`` if ``None``).

    Returns
    -------
    ImpurityBlockConsistency
        ``impurity_only_blocks`` and ``gf_blocks`` are lists of ``frozenset`` of global
        orbital indices; ``consistent`` is ``True`` iff they are the same partition;
        ``missing_pairs`` lists the impurity pairs the impurity-only structure drops.
    """
    imp = sorted(impurity_orbitals)

    # Impurity-only partition: connected components of h[imp, imp] (auto_block_structure with
    # orbitals=imp), mapped from local matrix indices back to global orbital indices.
    imp_set = set(imp)
    h = _one_body_matrix(op, n_orb=n_orb, must_span=imp)
    n = h.shape[0]
    bath = [o for o in range(n) if o not in imp_set]
    h_imp = h[np.ix_(imp, imp)]
    imp_only_local = build_block_structure(None, mat=h_imp).blocks
    impurity_only_blocks = [frozenset(imp[k] for k in block) for block in imp_only_local]

    # Dressed partition: connectivity of h[imp, imp] + V†V (bath-mediated coupling),
    # mapped from local matrix indices back to global orbital indices.
    m = h_imp + (h[np.ix_(bath, imp)].conj().T @ h[np.ix_(bath, imp)] if bath else 0.0)
    gf_blocks = [frozenset(imp[k] for k in block) for block in build_block_structure(None, mat=m).blocks]

    def _sorted_partition(blocks):
        return sorted((frozenset(b) for b in blocks), key=min)

    impurity_only_blocks = _sorted_partition(impurity_only_blocks)
    gf_blocks = _sorted_partition(gf_blocks)
    consistent = impurity_only_blocks == gf_blocks

    # Which impurity orbital is in which impurity-only block.
    imp_only_label = {}
    for b, block in enumerate(impurity_only_blocks):
        for orb in block:
            imp_only_label[orb] = b
    missing_pairs = []
    for block in gf_blocks:
        orbs = sorted(block)
        for a in range(len(orbs)):
            for b in range(a + 1, len(orbs)):
                if imp_only_label[orbs[a]] != imp_only_label[orbs[b]]:
                    missing_pairs.append((orbs[a], orbs[b]))

    return ImpurityBlockConsistency(impurity_only_blocks, gf_blocks, consistent, missing_pairs)


def discovered_orbital_blocks(op, n_orb=None):
    r"""Orbital block decomposition implied by the one-body symmetry of ``op``.

    Two orbitals lie in the same block iff the one-body Hamiltonian connects them
    (the connected components of ``h``). Equivalently, this is the orbital partition
    induced by the **diagonal part of the discovered Cartan**: orbitals in different
    blocks carry distinct conserved one-body quantum numbers, so their Green's-function
    cross terms are symmetry-forbidden. (A diagonal operator commutes with ``h`` iff it
    is constant on each connected component, so the finest diagonal conserved charge is
    exactly the component indicator.)

    Implemented by extracting the one-body tensor and reusing
    :func:`conserved_subset_charges` on the one-body operator.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The Hamiltonian (only its one-body part is used).
    n_orb : int, optional
        Number of spin-orbitals (inferred if ``None``).

    Returns
    -------
    list of frozenset of int
        The orbital blocks (a partition of ``range(n_orb)``).
    """
    h, _, _ = extract_tensors(op, n_orb=n_orb, two_body=False)
    one_body = tensors_to_operator(h)
    return conserved_subset_charges(one_body, n_orb=h.shape[0])


def blocks_refine_or_match(discovered, reference):
    r"""Whether ``discovered`` blocks **match or refine** the ``reference`` blocks.

    True iff every discovered block is a subset of some reference block (the discovered
    partition is at least as fine). This is the cross-phase acceptance condition: the
    auto-discovered orbital structure must reproduce or strictly refine a hand-coded
    ``BlockStructure`` — never coarsen it (which would merge orbitals the manual
    structure keeps separate).

    Parameters
    ----------
    discovered, reference : sequence of iterable of int
        Orbital block partitions (e.g. ``frozenset``s, or ``BlockStructure.blocks``).

    Returns
    -------
    bool
    """
    reference_sets = [set(block) for block in reference]
    return all(any(set(block) <= ref for ref in reference_sets) for block in discovered)


def measure_conserved_charges(psi, charges, n_orb, comm=None, round_to_int=True):
    r"""Measure the conserved subset-charge occupations of a state, ``<psi|N_S|psi>``.

    Because each ``N_S = Σ_{i∈S} n_i`` is **diagonal** in the determinant basis, this is
    a weighted sum of determinant occupations — no operator application, so it is safe on
    a hash-distributed ``ManyBodyState`` (sum locally, then ``Allreduce``). For a genuine
    eigenstate each charge has a definite (integer) value; ``round_to_int`` rounds the
    weighted average to that value.

    Parameters
    ----------
    psi : ManyBodyState
        The state. Its local determinants are summed; pass ``comm`` to combine ranks.
    charges : sequence of frozenset of int
        Conserved orbital subsets (e.g. from :func:`conserved_subset_charges`).
    n_orb : int
        Number of spin-orbitals (for decoding determinant bit patterns).
    comm : MPI.Comm, optional
        If given, the local sums are reduced across ranks.
    round_to_int : bool, optional
        Round each charge to the nearest integer (default True).

    Returns
    -------
    list
        ``<N_S>`` for each charge (ints if ``round_to_int``, else floats).
    """
    totals = np.zeros(len(charges))
    norm2 = 0.0
    for det, amp in psi.items():
        weight = abs(amp[0]) ** 2
        norm2 += weight
        occupied = {k for k, bit in enumerate(psr.bytes2bitarray(bytes(det.to_bytearray()), n_orb)) if bit}
        for i, subset in enumerate(charges):
            totals[i] += weight * len(subset & occupied)
    if comm is not None:
        totals = comm.allreduce(totals)
        norm2 = comm.allreduce(norm2)
    if norm2 == 0:
        return [0 for _ in charges]
    averages = totals / norm2
    if round_to_int:
        return [int(round(x)) for x in averages]  # noqa: RUF046  (np.float64 round() returns float, cast is needed)
    return list(averages)
