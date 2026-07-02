r"""Automated symmetry discovery for second-quantized Hamiltonians.

This module discovers the one-body symmetry algebra of a ``ManyBodyOperator``
Hamiltonian by solving the single-particle commutant condition :math:`[h, O] = 0`
and extracting its null space. The abelian (Cartan) part of the discovered algebra
labels conserved sectors (``N``, ``S_z``, ...); the full algebra (including the
non-abelian generators such as ``S_x``, ``S_y``) feeds the multiplet / Casimir
reconstruction in ``nonabelian_symmetry_casimir.md``.

**Scope and limitations.**

- Only **one-body** symmetry generators are found here. ``S^2`` and the other
  Casimirs are two-body and do not appear in the one-body null space; they are
  constructed separately (see ``finite.make_spin_operators`` / ``apply_casimir``).
- The method finds only **unitary** symmetries. **Anti-unitary symmetries (time
  reversal / Kramers degeneracy) are not detectable** by ``[H, O] = 0`` over the
  complex field. ``H`` is genuinely complex in this code (spin-orbit coupling), so
  this is a real gap, not a theoretical aside — report it wherever discovered
  symmetries are surfaced.

Implements symmetry-plan Phase 2 (``doc/plans/symmetry_implementation_plan.md``).
"""

from collections import namedtuple

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator


def extract_tensors(op, n_orb=None):
    r"""Extract the one- and two-body coefficient tensors of a ``ManyBodyOperator``.

    The operator is assumed purely 0-, 1- and 2-body and number-conserving, with
    terms in normal order. With the codebase convention
    (``finite.get_2_body_operator``) a stored term is

    - constant: ``()``  ->  ``const``
    - one-body: ``((i,'c'),(j,'a'))``  ->  :math:`h_{ij}` (coeff of
      :math:`c^\dagger_i c_j`)
    - two-body: ``((i,'c'),(j,'c'),(l,'a'),(k,'a'))``  ->  :math:`V_{ijkl}` (coeff of
      :math:`c^\dagger_i c^\dagger_j c_l c_k`)

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The operator (or its term dict).
    n_orb : int, optional
        Number of spin-orbitals. If ``None``, inferred as ``max index + 1``.

    Returns
    -------
    h : np.ndarray, shape (n_orb, n_orb)
        One-body coefficient matrix.
    V : np.ndarray, shape (n_orb, n_orb, n_orb, n_orb)
        Two-body coefficient tensor, :math:`V_{ijkl}` = coeff of
        :math:`c^\dagger_i c^\dagger_j c_l c_k`.
    const : complex
        Coefficient of the identity (energy shift).

    Raises
    ------
    ValueError
        If any term is not 0/1/2-body number-conserving (e.g. a 3-body term, or a
        non-number-conserving term such as a bare ``c``/``a``).
    """
    terms = op.to_dict() if hasattr(op, "to_dict") else dict(op)
    if n_orb is None:
        n_orb = 1 + max((idx for factors in terms for (idx, _) in factors), default=-1)

    h = np.zeros((n_orb, n_orb), dtype=complex)
    V = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)
    const = 0.0 + 0.0j

    for factors, amp in terms.items():
        ladder = tuple(c for (_, c) in factors)
        idx = [i for (i, _) in factors]
        if ladder == ():
            const += amp
        elif ladder == ("c", "a"):
            h[idx[0], idx[1]] += amp
        elif ladder == ("c", "c", "a", "a"):
            i, j, l, k = idx  # term is c†_i c†_j c_l c_k
            V[i, j, k, l] += amp
        else:
            raise ValueError(
                f"Operator term {factors} is not a 0/1/2-body number-conserving term "
                f"(ladder pattern {ladder}); symmetry discovery supports only 1- and "
                f"2-body operators."
            )
    return h, V, const


def tensors_to_operator(h, V=None, const=0.0, tol=0.0):
    r"""Inverse of :func:`extract_tensors`: build a ``ManyBodyOperator`` from tensors.

    Parameters
    ----------
    h : np.ndarray, shape (n_orb, n_orb)
        One-body coefficients :math:`h_{ij}` (coeff of :math:`c^\dagger_i c_j`).
    V : np.ndarray, shape (n_orb,)*4, optional
        Two-body coefficients :math:`V_{ijkl}` (coeff of
        :math:`c^\dagger_i c^\dagger_j c_l c_k`).
    const : complex, optional
        Identity coefficient.
    tol : float, optional
        Drop coefficients with magnitude ``<= tol`` (default 0: keep exact nonzeros).

    Returns
    -------
    ManyBodyOperator
    """
    d = {}
    n_orb = h.shape[0]
    for i in range(n_orb):
        for j in range(n_orb):
            if abs(h[i, j]) > tol:
                d[((i, "c"), (j, "a"))] = h[i, j]
    if V is not None:
        for i in range(n_orb):
            for j in range(n_orb):
                for k in range(n_orb):
                    for l in range(n_orb):
                        if abs(V[i, j, k, l]) > tol:
                            d[((i, "c"), (j, "c"), (l, "a"), (k, "a"))] = V[i, j, k, l]
    if abs(const) > tol:
        d[()] = const
    return ManyBodyOperator(d)


def rotate_one_body(h, u):
    r"""Rotate the one-body tensor to a new single-particle basis: ``h' = U† h U``.

    ``U``'s columns are the new basis vectors expressed in the old basis, i.e. the
    new operators are ``c'_a = Σ_i U*_{ia} c_i``.

    Parameters
    ----------
    h : np.ndarray, shape (n, n)
    u : np.ndarray, shape (n, n)
        Unitary single-particle transformation.

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    u = np.asarray(u, dtype=complex)
    return np.einsum("mi,mn,nj->ij", u.conj(), np.asarray(h, dtype=complex), u, optimize=True)


def rotate_two_body(v_tensor, u):
    r"""Rotate the two-body tensor to a new single-particle basis.

    With the :func:`extract_tensors` convention (:math:`V_{ijkl}` = coeff of
    :math:`c^\dagger_i c^\dagger_j c_l c_k`) and ``c'_a = Σ_i U*_{ia} c_i``,

    .. math:: V'_{ijkl} = \sum_{mnpq} U^*_{mi} U^*_{nj} V_{mnpq} U_{pk} U_{ql}.

    Parameters
    ----------
    v_tensor : np.ndarray, shape (n, n, n, n)
    u : np.ndarray, shape (n, n)

    Returns
    -------
    np.ndarray, shape (n, n, n, n)
    """
    u = np.asarray(u, dtype=complex)
    return np.einsum(
        "mi,nj,mnpq,pk,ql->ijkl",
        u.conj(),
        u.conj(),
        np.asarray(v_tensor, dtype=complex),
        u,
        u,
        optimize=True,
    )


def rotate_hamiltonian(op, u, tol=1e-12):
    r"""Express a ``ManyBodyOperator`` in a rotated single-particle basis (``H' = U† H U``).

    Extracts the 1-/2-body tensors (:func:`extract_tensors`), rotates them
    (:func:`rotate_one_body`, :func:`rotate_two_body`) and rebuilds the operator.
    This is the one-time setup cost of symmetry-plan Phase 5 — a pure ``einsum``
    contraction, no Lanczos hot loop involved.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The 1-/2-body operator to rotate.
    u : np.ndarray, shape (n, n)
        Unitary single-particle transformation (e.g. from :func:`joint_diagonalize`).
    tol : float, optional
        Drop rotated coefficients with magnitude ``<= tol`` (default 1e-12) to prune
        numerical noise from the dense contraction.

    Returns
    -------
    ManyBodyOperator
    """
    u = np.asarray(u, dtype=complex)
    h, v_tensor, const = extract_tensors(op, n_orb=u.shape[0])
    h_rot = rotate_one_body(h, u)
    v_rot = rotate_two_body(v_tensor, u)
    return tensors_to_operator(h_rot, v_rot, const, tol=tol)


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


def group_orbitals_by_blocks(op, impurity_orbitals, valence_orbitals, conduction_orbitals, block_structure, n_orb=None):
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

    Returns
    -------
    impurity_orbitals : dict[int, list[list[int]]]
        ``{group: [impurity orbital block]}``.
    bath_states : tuple(dict, dict)
        ``(valence_baths, conduction_baths)``, each ``{group: [bath orbital block]}`` with the
        same keys as ``impurity_orbitals``.
    """
    from impurityModel.ed.block_structure import get_equivalent_orbs

    imp = sorted(impurity_orbitals)
    val_set = set(valence_orbitals)
    con_set = set(conduction_orbitals)
    h, _, _ = extract_tensors(op, n_orb=n_orb)

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


def classify_bath_occupation(op, impurity_orbitals, n_orb=None):
    r"""Split the bath orbitals into initially-occupied (valence) and empty (conduction) sets.

    The bath orbitals are all spin-orbitals of ``op`` that are **not** impurity orbitals; each is
    classified by its one-body on-site energy ``h[o, o]``: baths below the Fermi level
    (``h[o, o] < 0``) sit filled in the nominal configuration (**valence**, initially occupied),
    the rest are empty (**conduction**, initially empty). This is the same Fermi-level-zero
    convention as :func:`edchain.build_imp_bath_blocks` / the ``build_h0`` bath partitioning, so a
    Hamiltonian built there reproduces its valence/conduction split without the caller passing it.

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

    Returns
    -------
    valence_orbitals : list[int]
        Initially-occupied bath orbital indices (``h[o, o] < 0``), sorted.
    conduction_orbitals : list[int]
        Initially-empty bath orbital indices (``h[o, o] >= 0``), sorted.
    """
    h, _, _ = extract_tensors(op, n_orb=n_orb)
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

    Uses the integer-weight form ``2 S_z = Σ (n_up − n_dn)`` (weight ``+1`` on each up
    orbital, ``−1`` on each down orbital), so ``two_sz_target = 2 * S_z`` is an integer.
    The spin pairs should come from
    ``finite.impurity_spin_pairs`` / ``finite.bath_spin_pairs`` after
    ``finite.spin_pairs_consistent_with_h`` validates them.

    Parameters
    ----------
    spin_pairs : sequence of (int, int)
        ``(dn, up)`` spin-orbital index pairs.
    two_sz_target : int
        Target value of ``2 S_z`` (``N_up − N_dn``).
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
    from impurityModel.ed.block_structure import build_block_structure

    h, _, _ = extract_tensors(op, n_orb=n_orb)
    if orbitals is not None:
        h = h[np.ix_(list(orbitals), list(orbitals))]
    return build_block_structure(None, mat=h)


def impurity_block_structure(op, impurity_orbitals, n_orb=None):
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

    Returns
    -------
    BlockStructure
    """
    from impurityModel.ed.block_structure import build_block_structure

    imp = sorted(impurity_orbitals)
    imp_set = set(imp)
    h, _, _ = extract_tensors(op, n_orb=n_orb)
    n = h.shape[0]
    bath = [o for o in range(n) if o not in imp_set]

    m = h[np.ix_(imp, imp)]
    if bath:
        v = h[np.ix_(bath, imp)]  # (n_bath, n_imp) impurity-bath hopping
        m = m + v.conj().T @ v  # add the bath-mediated (first-moment) coupling
    return build_block_structure(None, mat=m)


def impurity_symmetry_rotation(op, impurity_orbitals, n_orb=None):
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
       ``joint_diagonalize``: a degenerate ``h_imp`` (e.g. six-fold ``t2g``×spin, four-fold
       ``eg``×spin) has a huge ``U(6)×U(4)`` one-body commutant, and jointly diagonalising a
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

    Returns
    -------
    W : ndarray, shape (n_orb, n_orb)
        The full-space unitary (``u_imp`` embedded on the impurity orbitals, identity on the
        bath). Rotate the Hamiltonian with :func:`rotate_hamiltonian`.
    u_imp : ndarray, shape (n_imp, n_imp)
        The impurity-block rotation, in the sorted-``impurity_orbitals`` order (columns are the
        eigenvectors of ``h[imp, imp]``).
    """
    h, _, _ = extract_tensors(op, n_orb=n_orb)
    n = h.shape[0]
    imp = sorted(impurity_orbitals)
    h_imp = h[np.ix_(imp, imp)]
    h_imp = 0.5 * (h_imp + h_imp.conj().T)  # symmetrise away round-off before eigh
    _eigvals, u_imp = np.linalg.eigh(h_imp)
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
    from impurityModel.ed.block_structure import build_block_structure

    imp = sorted(impurity_orbitals)

    # Impurity-only partition: connected components of h[imp, imp] (auto_block_structure with
    # orbitals=imp), mapped from local matrix indices back to global orbital indices.
    imp_set = set(imp)
    h, _, _ = extract_tensors(op, n_orb=n_orb)
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
    h, _, _ = extract_tensors(op, n_orb=n_orb)
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
    import impurityModel.ed.product_state_representation as psr

    totals = np.zeros(len(charges))
    norm2 = 0.0
    for det, amp in psi.items():
        weight = abs(amp) ** 2
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
        return [int(round(x)) for x in averages]
    return list(averages)


def discover_rotation(op, n_orb=None, seed=0):
    r"""Discover the symmetry and return ``(U, cartan)``: the symmetry-adapting rotation
    and the (un-rotated) Cartan generators that commute with ``h``.

    Unlike :func:`symmetry_adapted_transformation` (which returns the *rotated* operator
    and rotated generators), this returns the Cartan in the **original** basis, so the
    generators can be cheaply re-tested against a later Hamiltonian of the same symmetry
    (see :class:`SymmetryRotationCache`).
    """
    h, _, _ = extract_tensors(op, n_orb=n_orb)
    generators = discover_one_body_symmetries(h)
    cartan = cartan_subalgebra(generators, seed=seed)
    u, _ = joint_diagonalize(cartan, seed=seed)
    return u, cartan


class SymmetryRotationCache:
    r"""Cache the symmetry-adapting rotation ``U`` across e.g. DMFT iterations.

    In a self-consistency loop ``H`` changes every iteration (bath fitting), but the
    symmetry *structure* (point group + spin) is fixed by the problem — only coefficients
    move. Re-running the full discovery (SVD null space + Cartan + joint diagonalisation)
    each iteration is wasteful. This caches ``U`` and re-discovers **only** when the
    cached Cartan generators no longer commute with the new ``h`` (a cheap O(n³) check),
    i.e. when the symmetry structure actually changed.

    Attributes
    ----------
    discovery_count : int
        How many times the full discovery actually ran (the metric a cache-hit test
        checks).
    """

    def __init__(self, seed=0, tol=1e-9):
        self.seed = seed
        self.tol = tol
        self._u = None
        self._cartan = None
        self.discovery_count = 0

    def _symmetry_preserved(self, h):
        return self._cartan is not None and all(np.linalg.norm(h @ g - g @ h) <= self.tol for g in self._cartan)

    def get_rotation(self, op, n_orb=None):
        """Return ``U`` for ``op``, reusing the cached rotation if the symmetry is unchanged."""
        h, _, _ = extract_tensors(op, n_orb=n_orb)
        if self._symmetry_preserved(h):
            return self._u
        self._u, self._cartan = discover_rotation(op, n_orb=n_orb, seed=self.seed)
        self.discovery_count += 1
        return self._u


def symmetry_adapted_transformation(op, n_orb=None, seed=0):
    r"""Discover the symmetry and rotate the Hamiltonian into its symmetry-adapted basis.

    Ties the Phase-2 / Phase-5 pieces together: extract the one-body tensor, discover
    the commutant, pick a Cartan subalgebra, build the single-particle ``U`` that
    simultaneously diagonalises it, and return the rotated operator ``H' = U† H U``. In
    this basis every Cartan generator is **diagonal** — those with ``{0,1}`` weights are
    occupation numbers that map directly to subset-occupation restrictions (Phase 3),
    the rest (e.g. ``S_z``) need the weighted-sum machinery (Phase 6). This is the
    bridge that lets Phase 3 sectorize in a basis where the conserved charges are
    manifest.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The Hamiltonian.
    n_orb : int, optional
        Number of spin-orbitals (inferred from ``op`` if ``None``).
    seed : int, optional
        Seed for the random regular element / joint-diagonalisation combination.

    Returns
    -------
    u : np.ndarray
        Single-particle transformation to the symmetry-adapted basis.
    rotated_op : ManyBodyOperator
        ``H'`` expressed in that basis.
    cartan : list of np.ndarray
        The Cartan generators, each **diagonal** in the new basis (``U† G U``). These
        are the raw material Phase 3 maps to restrictions (after rescaling to integer
        orbital weights and the ``{0,1}`` test); the mapping itself is left to Phase 3.
    """
    h, _, _ = extract_tensors(op, n_orb=n_orb)
    generators = discover_one_body_symmetries(h)
    cartan = cartan_subalgebra(generators, seed=seed)
    u, _ = joint_diagonalize(cartan, seed=seed)
    rotated_op = rotate_hamiltonian(op, u)
    cartan_rotated = [rotate_one_body(g, u) for g in cartan]
    return u, rotated_op, cartan_rotated


def _commutator_superoperator(h):
    r"""Matrix ``A`` of the linear map ``O -> [h, O]`` acting on ``vec(O)``.

    Uses **column-stacking** ``vec`` (``O.reshape(-1, order='F')``). With that
    convention ``vec(h O) = (I ⊗ h) vec(O)`` and ``vec(O h) = (hᵀ ⊗ I) vec(O)``, so

    .. math:: \mathrm{vec}([h, O]) = (I\otimes h - h^{T}\otimes I)\,\mathrm{vec}(O).

    Reshape any null vector back with the **same** ``order='F'``.
    """
    n = h.shape[0]
    eye = np.eye(n)
    return np.kron(eye, h) - np.kron(h.T, eye)


def discover_one_body_symmetries(h, sigma_cut=None):
    r"""Discover the one-body symmetry algebra ``{O : [h, O] = 0}`` of ``h``.

    Solves the single-particle commutant condition by taking the null space of the
    commutator super-operator (via SVD). Each returned generator is an ``n x n``
    matrix; together they form an orthonormal (Frobenius) basis of the commutant.

    For an SU(2)-symmetric ``h`` the algebra contains **all three** one-body spin
    generators ``S_x, S_y, S_z`` (and ``N``), not just ``S_z``. ``S^2`` is two-body
    and by construction cannot appear here.

    Parameters
    ----------
    h : np.ndarray, shape (n, n)
        The one-body Hamiltonian matrix (e.g. from :func:`extract_tensors`).
    sigma_cut : float, optional
        Singular values ``<= sigma_cut`` are treated as zero (in the null space).
        Default ``||h-superoperator||_2 * n * eps`` — scaled to the problem so it is
        robust to the floating-point null space being only approximate.

    Returns
    -------
    list of np.ndarray
        The generator matrices spanning the commutant.

    Notes
    -----
    Unitary symmetries only: ``[H, O] = 0`` over the complex field does not detect
    anti-unitary (time-reversal / Kramers) symmetries. See the module docstring.
    """
    h = np.asarray(h, dtype=complex)
    n = h.shape[0]
    a_matrix = _commutator_superoperator(h)
    _, s, vh = np.linalg.svd(a_matrix)
    norm_a = s[0] if s.size else 0.0
    if sigma_cut is None:
        sigma_cut = max(norm_a, 1.0) * n * np.finfo(float).eps
    null_mask = s <= sigma_cut
    null_vecs = vh.conj().T[:, null_mask]  # columns = vec(O), orthonormal
    return [null_vecs[:, a].reshape(n, n, order="F") for a in range(null_vecs.shape[1])]


def is_abelian(generators, tol=1e-9):
    r"""Whether the one-body symmetry algebra is abelian (all commutators vanish).

    A non-abelian algebra (some ``[O_a, O_b] != 0``) is the signature of a non-abelian
    symmetry — e.g. SU(2) spin, whose three one-body generators ``S_x, S_y, S_z`` are all
    in the commutant but do not mutually commute (companion plan
    ``nonabelian_symmetry_casimir.md``, Phase A.1).
    """
    gens = [np.asarray(g, dtype=complex) for g in generators]
    for a in range(len(gens)):
        for b in range(a + 1, len(gens)):
            if np.linalg.norm(gens[a] @ gens[b] - gens[b] @ gens[a]) > tol:
                return False
    return True


def structure_constants(generators):
    r"""Lie-algebra structure constants ``f_{abc}`` with ``[O_a, O_b] = Σ_c f_{abc} O_c``.

    Assumes the ``generators`` are **mutually orthogonal** under the Frobenius inner
    product ``<A, B> = Tr(A† B)`` (true for the SVD null-space basis from
    :func:`discover_one_body_symmetries`, and for ``{S_x, S_y, S_z}``). For an su(2)
    triplet these reproduce ``[S_a, S_b] = i ε_{abc} S_c`` (up to the generator
    normalisation).

    Parameters
    ----------
    generators : sequence of np.ndarray
        Single-particle generator matrices (mutually Frobenius-orthogonal).

    Returns
    -------
    np.ndarray, shape (n, n, n)
        ``f[a, b, c]``.
    """
    gens = [np.asarray(g, dtype=complex) for g in generators]
    n = len(gens)
    norms = [np.vdot(g, g) for g in gens]
    f = np.zeros((n, n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            commutator = gens[a] @ gens[b] - gens[b] @ gens[a]
            for c in range(n):
                if abs(norms[c]) > 1e-15:
                    f[a, b, c] = np.vdot(gens[c], commutator) / norms[c]
    return f


def apply_reconstructed_casimir(psi, generators):
    r"""Apply a reconstructed Casimir ``Ĉ = Σ_a Ô_a²`` to a state.

    Given a (sub)set of one-body symmetry generators ``{O_a}`` (single-particle
    matrices, e.g. an su(2) spin triplet ``{S_x, S_y, S_z}``), each is promoted to a
    one-body ``ManyBodyOperator`` ``Ô_a = Σ_ij (O_a)_ij c†_i c_j`` and the Casimir is
    ``Ĉ = Σ_a Ô_a Ô_a``, applied by sequential one-body application (no explicit
    two-body product is formed). For the spin triplet this is exactly ``Ŝ²`` (companion
    plan Phase A.2).

    Parameters
    ----------
    psi : ManyBodyState
    generators : sequence of np.ndarray
        Single-particle generator matrices spanning the sub-algebra.

    Returns
    -------
    ManyBodyState
        ``Ĉ |psi>``.
    """
    ops = [tensors_to_operator(np.asarray(g, dtype=complex)) for g in generators]
    result = None
    for op in ops:
        term = op(op(psi, 0), 0)
        result = term if result is None else result + term
    return result


def expect_reconstructed_casimir(psi, generators, comm=None):
    r"""Return ``<psi|Ĉ|psi>`` for the reconstructed Casimir ``Ĉ = Σ_a Ô_a²``."""
    from impurityModel.ed.ManyBodyUtils import inner

    val = inner(psi, apply_reconstructed_casimir(psi, generators))
    if comm is not None:
        val = comm.allreduce(val)
    return np.real(val)


def hermitian_algebra_basis(generators, tol=1e-9):
    r"""Return an orthonormal **Hermitian** basis of the algebra spanned by ``generators``.

    The commutant of a Hermitian ``h`` is closed under conjugate-transpose, so it is
    the complexification of a real algebra of Hermitian operators. For each generator
    ``O`` both ``(O+O†)/2`` and ``(O-O†)/(2i)`` are Hermitian and lie in the algebra;
    these are orthonormalised (real Gram-Schmidt under the Frobenius inner product
    ``Re Tr(A†B)``) so every returned matrix stays Hermitian.
    """
    candidates = []
    for gen in generators:
        gen = np.asarray(gen, dtype=complex)
        candidates.append(0.5 * (gen + gen.conj().T))
        candidates.append(0.5j * (gen.conj().T - gen))
    basis = []
    for mat in candidates:
        residual = mat.copy()
        for b in basis:
            residual = residual - np.real(np.vdot(b, residual)) * b
        norm = np.sqrt(np.real(np.vdot(residual, residual)))
        if norm > tol:
            basis.append(residual / norm)
    return basis


def cartan_subalgebra(generators, seed=0, tol=None):
    r"""Extract a maximal mutually-commuting (Cartan) subalgebra of the commutant.

    Uses the standard regular-element method: form a generic Hermitian element
    ``X = Σ rₖ Hₖ`` of the algebra (random real ``rₖ``); its centralizer within the
    algebra is, for generic ``X``, a maximal abelian (Cartan) subalgebra. The
    centralizer is found as the null space of ``O -> [X, O]`` restricted to the
    algebra. This is far more robust than diagonalising generators sequentially.

    Parameters
    ----------
    generators : sequence of np.ndarray
        A basis of the commutant (e.g. from :func:`discover_one_body_symmetries`).
    seed : int, optional
        Seed for the random regular element (reproducibility).
    tol : float, optional
        Null-space cutoff for the centralizer system. Default scales with the
        problem size and machine epsilon.

    Returns
    -------
    list of np.ndarray
        Mutually-commuting Hermitian matrices spanning the Cartan subalgebra.
    """
    herm = hermitian_algebra_basis(generators)
    m = len(herm)
    if m == 0:
        return []
    n = herm[0].shape[0]
    rng = np.random.default_rng(seed)
    x = sum(r * h for r, h in zip(rng.standard_normal(m), herm))
    # Solve sum_k c_k [X, H_k] = 0 for real c.
    cols = np.array([(x @ h - h @ x).reshape(-1) for h in herm]).T  # (n^2, m), complex
    real_sys = np.vstack([cols.real, cols.imag])  # (2 n^2, m), real -> enforces real c
    _, s, vt = np.linalg.svd(real_sys)
    if tol is None:
        tol = max(s[0] if s.size else 0.0, 1.0) * max(n, m) * np.finfo(float).eps
    # Right singular vectors (rows of vt) with singular value <= tol span the null space.
    null_coeffs = [vt[i] for i in range(m) if (i >= len(s) or s[i] <= tol)]
    cartan = [sum(c[k] * herm[k] for k in range(m)) for c in null_coeffs]
    return cartan


def joint_diagonalize(commuting_ops, seed=0):
    r"""Simultaneously diagonalise a set of mutually-commuting Hermitian operators.

    Forms a random real combination ``M = Σ rₖ Oₖ``; for generic ``rₖ`` its spectrum
    is non-degenerate, so its eigenvectors diagonalise **all** the ``Oₖ`` at once —
    robust even when an individual ``Oₖ`` has a degenerate spectrum (where the naive
    "diagonalise ``O₁`` then ``O₂`` in its eigenspaces" approach is ambiguous).

    Parameters
    ----------
    commuting_ops : sequence of np.ndarray
        Mutually-commuting Hermitian matrices.
    seed : int, optional
        Seed for the random combination.

    Returns
    -------
    U : np.ndarray
        Unitary whose columns are the common eigenvectors (the single-particle
        transformation).
    diagonals : list of np.ndarray
        For each operator, its real eigenvalues ordered along ``U`` (``diag(U† Oₖ U)``).
    """
    ops = [np.asarray(o, dtype=complex) for o in commuting_ops]
    rng = np.random.default_rng(seed)
    combo = sum(rng.standard_normal() * o for o in ops)
    combo = 0.5 * (combo + combo.conj().T)
    _, u = np.linalg.eigh(combo)
    diagonals = [np.real(np.diag(u.conj().T @ o @ u)) for o in ops]
    return u, diagonals


def weights_are_01(diag, tol=1e-6):
    r"""Whether a generator's single-particle eigenvalues (weights) are all in {0, 1}.

    A generator can be mapped to a subset-occupation restriction (symmetry-plan
    Phase 3) only if its orbital weights are ``{0, 1}`` (e.g. ``N``, a spin-up count).
    Generators with other weights (e.g. ``S_z`` with ``±1/2``) require the extended
    weighted-sum restriction machinery (Phase 6). ``diag`` is the eigenvalue list
    from :func:`joint_diagonalize`.
    """
    diag = np.asarray(diag)
    return bool(np.all(np.isclose(diag, 0.0, atol=tol) | np.isclose(diag, 1.0, atol=tol)))


def in_span(generators, matrix, tol=1e-9):
    r"""Whether ``matrix`` lies in the Frobenius span of the ``generators``.

    Parameters
    ----------
    generators : sequence of np.ndarray
        Generator matrices (e.g. from :func:`discover_one_body_symmetries`).
    matrix : np.ndarray
        The matrix to test for membership.
    tol : float, default 1e-9
        Relative residual tolerance.

    Returns
    -------
    bool
    """
    vec = np.asarray(matrix, dtype=complex).reshape(-1, order="F")
    if len(generators) == 0:
        return np.linalg.norm(vec) <= tol
    basis = np.array([np.asarray(g, dtype=complex).reshape(-1, order="F") for g in generators]).T
    q, _ = np.linalg.qr(basis)
    residual = vec - q @ (q.conj().T @ vec)
    return np.linalg.norm(residual) <= tol * max(np.linalg.norm(vec), 1.0)


# Result of :func:`component_symmetry_reduction`.
#
# ``Q`` : (m, m) unitary; column ``a`` is a symmetry-adapted component ``T'_a = sum_alpha
#         Q[alpha, a] T_alpha`` expressed in the original Cartesian component basis.
# ``representatives`` : column indices of ``Q`` that must actually be computed (one per
#         symmetry-equivalence group).
# ``group_of_column`` : length ``m``; ``group_of_column[a]`` is the index *into*
#         ``representatives`` of the representative for column ``a``.
# ``diagonalizable`` : if True, ``chi' = Q^dagger chi Q`` is diagonal with equal entries
#         within each group, so the full spectral tensor is rebuilt from the representative
#         diagonals via ``Q``. If False (no closing continuous symmetry, or a non-abelian
#         commutant), the caller falls back to the full ``m x m`` tensor (``Q`` is the
#         identity, every column is its own representative) -- always correct, just no dedup.
ComponentReduction = namedtuple("ComponentReduction", ["Q", "representatives", "group_of_column", "diagonalizable"])


def _matrix_commutant(mats, tol=None):
    r"""Orthonormal (Frobenius) basis of ``{X : [X, M] = 0 for all M in mats}``.

    Uses column-stacking ``vec``: ``vec(X M) = (M^T (x) I) vec(X)`` and
    ``vec(M X) = (I (x) M) vec(X)``, so ``vec([X, M]) = (M^T (x) I - I (x) M) vec(X)``.
    The joint null space over all ``M`` is the commutant. Returns the null-space matrices
    (generally not Hermitian; Hermitianise separately if needed).
    """
    mats = [np.asarray(m, dtype=complex) for m in mats]
    if not mats:
        return []
    n = mats[0].shape[0]
    eye = np.eye(n)
    rows = [np.kron(m.T, eye) - np.kron(eye, m) for m in mats]
    system = np.vstack(rows)
    _, s, vh = np.linalg.svd(system)
    norm = s[0] if s.size else 0.0
    if tol is None:
        tol = max(norm, 1.0) * n * np.finfo(float).eps
    null = vh.conj().T[:, [i for i in range(vh.shape[0]) if i >= len(s) or s[i] <= tol]]
    return [null[:, a].reshape(n, n, order="F") for a in range(null.shape[1])]


def component_symmetry_reduction(component_ops, h_onebody, n_orb=None, tol=1e-8):
    r"""Point-group dedup of a set of one-body *component* transition operators.

    A dipole (or other one-body) transition tensor
    :math:`\chi_{\alpha\beta}(\omega) = \langle g| T_\alpha^\dagger (\omega - H)^{-1}
    T_\beta |g\rangle` is invariant under any continuous single-particle symmetry ``G`` of
    ``h`` (``[h, G] = 0``): if ``G`` maps the component span onto itself,
    :math:`[G, T_\alpha] = \sum_\beta M_{\beta\alpha} T_\beta`, then :math:`\chi` commutes
    with the representation ``{M}`` on component space. When that representation is
    multiplicity-free (commutant abelian) a common eigenbasis ``Q`` of the commutant makes
    :math:`\chi' = Q^\dagger \chi Q` diagonal with **equal** entries within each
    symmetry-equivalence group -- so only one representative component per group needs a
    Lanczos run and the whole tensor is rebuilt from ``Q``.

    Only **continuous** (Lie-algebra) symmetries are detected (via
    :func:`discover_one_body_symmetries`): a fully rotational ``h`` collapses the three
    Cartesian dipoles to a single representative. Lower continuous symmetry (e.g. axial
    ``L_z``) still block-diagonalises the tensor -- ``chi`` comes back diagonal in ``Q`` --
    but does not by itself equate the two in-plane components (``chi_{+} = chi_{-}`` needs a
    reflection / time reversal, which ``[h, G] = 0`` cannot see), so no columns are dropped.
    A purely **discrete** point group (e.g. a cubic crystal field) or no symmetry falls back
    to the identity reduction and the full ``m x m`` tensor -- always correct, just no dedup.

    Parameters
    ----------
    component_ops : sequence of ManyBodyOperator or dict
        The one-body component operators (e.g. the three Cartesian dipole operators).
    h_onebody : np.ndarray, shape (n_orb, n_orb)
        One-body Hamiltonian matrix whose symmetries are used (from :func:`extract_tensors`).
    n_orb : int, optional
        Number of spin-orbitals; inferred from the operators/matrix if ``None``.
    tol : float, optional
        Closure / commutant residual tolerance.

    Returns
    -------
    ComponentReduction
    """
    if n_orb is None:
        n_orb = np.asarray(h_onebody).shape[0]
    m = len(component_ops)
    Ts = [extract_tensors(op, n_orb=n_orb)[0] for op in component_ops]
    identity = ComponentReduction(np.eye(m, dtype=complex), list(range(m)), list(range(m)), m <= 1)
    if m <= 1:
        return identity

    # Component matrices, vectorised, as the least-squares span for the closure test.
    tvec = np.array([T.reshape(-1) for T in Ts]).T  # (n^2, m)
    q_t, _ = np.linalg.qr(tvec)  # orthonormal basis of span(T); projector P = q_t q_t^dagger
    generators = discover_one_body_symmetries(np.asarray(h_onebody, dtype=complex))
    if not generators:
        return identity

    # The commutant basis from discovery is arbitrary: the generators that *close* on the
    # component span (``[G, T_alpha] in span(T)``) are linear combinations ``G = sum_k x_k C_k``.
    # Solve for that closing subalgebra as the null space of the "leaves-the-span" residual,
    # then represent each closing generator on the component span as an m x m matrix M.
    gens = [np.asarray(g, dtype=complex) for g in generators]
    columns = []
    for c in gens:
        res_c = []
        for T in Ts:
            r = (c @ T - T @ c).reshape(-1)
            r = r - q_t @ (q_t.conj().T @ r)  # component of [C, T_alpha] orthogonal to span(T)
            res_c.append(r)
        columns.append(np.concatenate(res_c))
    b_matrix = np.array(columns).T  # (m * n^2, n_gen)
    _, s_b, vh_b = np.linalg.svd(b_matrix)
    scale = s_b[0] if s_b.size else 0.0
    cut = max(scale, 1.0) * b_matrix.shape[0] * np.finfo(float).eps
    null_coeffs = [vh_b[i].conj() for i in range(vh_b.shape[0]) if i >= len(s_b) or s_b[i] <= cut]

    reps = []
    for x in null_coeffs:
        g = sum(x[k] * gens[k] for k in range(len(gens)))
        M = np.zeros((m, m), dtype=complex)
        for a, T in enumerate(Ts):
            coeff, *_ = np.linalg.lstsq(tvec, (g @ T - T @ g).reshape(-1), rcond=None)
            M[:, a] = coeff
        if np.linalg.norm(M) > tol:
            reps.append(M)

    if not reps:
        return identity  # no continuous symmetry acts non-trivially -> full tensor

    # chi commutes with the representation; work with the commutant of {M, M^dagger} so it is
    # closed under adjoint, then take its Hermitian basis.
    commutant = _matrix_commutant([r for M in reps for r in (M, M.conj().T)], tol=tol)
    herm = hermitian_algebra_basis(commutant, tol=tol)
    if not herm or not is_abelian(herm, tol=tol):
        return identity  # multiplicity > 1 (non-abelian commutant): keep the full tensor

    # Common eigenbasis of the abelian commutant; group columns by identical eigenvalue
    # signature (same "character" -> chi forced equal on them).
    Q, diagonals = joint_diagonalize(herm)
    signatures = np.round(np.array(diagonals).T / max(tol, 1e-12)).astype(np.int64)  # (m, len(herm))
    group_of_column = [0] * m
    representatives = []
    seen = {}
    for a in range(m):
        key = tuple(signatures[a])
        if key not in seen:
            seen[key] = len(representatives)
            representatives.append(a)
        group_of_column[a] = seen[key]
    return ComponentReduction(Q, representatives, group_of_column, True)
