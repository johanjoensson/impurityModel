r"""Signed orbital permutations that are exact symmetries of a Hamiltonian, and their action on labels.

A certified sector solve (:mod:`impurityModel.ed.sector_solver`) expands every hard label
``(2 S_z, parity pattern)`` of its seed separately. When an orbital permutation ``P`` -- with a sign
per orbital, ``c_i -> s_i c_{P(i)}`` -- leaves ``H`` *and* the basis restrictions unchanged, it maps
the problem of one label onto the problem of another unitarily: the two labels have identical
spectra, bath included, and one expansion answers both. On the NiO linked-chain iteration 1 a swap of
two t2g orbitals (with their bath chains, and one sign) and the spin flip are such symmetries; the
swap alone made two of four labels in two sectors bit-identical repeats.

This is not the "equal seed spectra" merge :mod:`~impurityModel.ed.sector_solver` rules out: that
compares what the seed sees and misses the bath energies, while a symmetry here is verified on the
whole operator, term by term after normal ordering.

Only transpositions of impurity spatial orbitals (both spins together, each carrying its bath
chain) and the spin flip are generated. A symmetry that is not a signed permutation -- a ``C_3``
mixing the e_g orbitals -- is not found; that is safe, it only goes undeduplicated.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from impurityModel.ed.lie_algebra import extract_tensors
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import impurity_two_body_tensor

#: Largest coefficient of ``P H P^-1 - H``, relative to the largest coefficient of ``H``, for which
#: ``P`` counts as a symmetry. Exact operator algebra: a symmetry cancels to roundoff, a broken one at
#: the size of the breaking term (a bath level fitted 1e-3 apart is caught by six orders).
SYMMETRY_TOL = 1e-10


@dataclass(frozen=True)
class SignedPermutation:
    """``c_i -> signs[i] * c_{perm[i]}`` on every spin-orbital; ``flips_spin`` when it swaps up and down."""

    perm: tuple
    signs: tuple
    flips_spin: bool = False

    def compose(self, other):
        """``self`` after ``other``."""
        perm = tuple(self.perm[other.perm[i]] for i in range(len(self.perm)))
        signs = tuple(other.signs[i] * self.signs[other.perm[i]] for i in range(len(self.perm)))
        return SignedPermutation(perm, signs, self.flips_spin != other.flips_spin)

    def map_bits(self, bits):
        """Occupations ``(n, n_orb)`` of the image determinants: orbital ``i`` moves to ``perm[i]``."""
        out = np.zeros_like(bits)
        out[:, list(self.perm)] = bits
        return out


def _transformed_terms(op, sp):
    out = {}
    for term, amp in op.items():
        sign = 1
        for orb, _action in term:
            sign *= sp.signs[orb]
        image = tuple((sp.perm[orb], action) for orb, action in term)
        out[image] = out.get(image, 0) + sign * amp
    return out


def is_symmetry(op, sp, tol=SYMMETRY_TOL):
    """Whether ``P op P^-1 == op``, compared after normal ordering (``ManyBodyOperator``)."""
    items = list(op.items())
    scale = max((abs(amp) for _term, amp in items), default=0.0)
    diff = ManyBodyOperator(_transformed_terms(op, sp)) - ManyBodyOperator(dict(items))
    return max((abs(v) for v in diff.values()), default=0.0) <= tol * max(scale, 1.0)


def preserves_restrictions(sp, restrictions, weighted_restrictions):
    """Whether ``sp`` maps the occupation and weighted restrictions onto themselves.

    A symmetry of ``H`` alone is not enough: the label's problem is ``H`` on the restricted space, and
    a window that singles out one orbital's chain makes the swapped label a different problem.
    """
    if restrictions:
        own = {frozenset(indices): tuple(limits) for indices, limits in restrictions.items()}
        for indices, limits in own.items():
            if own.get(frozenset(sp.perm[i] for i in indices)) != limits:
                return False
    if weighted_restrictions:
        own = [(dict(weights), tuple(limits)) for weights, limits in weighted_restrictions]
        for weights, limits in own:
            image = {sp.perm[i]: w for i, w in weights.items()}
            if (image, limits) not in own:
                return False
    return True


def _bath_chains(h1, impurity, n_orb):
    """Each impurity spin-orbital's bath orbitals in breadth-first order through bath-only hopping.

    Ties within one depth are ordered by on-site energy, then by index, so equivalent chains list
    their sites in corresponding order. A mis-ordered pairing is never trusted: the full-operator
    check rejects it.
    """
    imp = set(impurity)
    adjacency = [np.flatnonzero(np.abs(h1[i]) > 0) for i in range(n_orb)]
    chains = {}
    for i in impurity:
        seen, frontier, order = {i}, [i], []
        while frontier:
            depth = sorted(
                {int(j) for o in frontier for j in adjacency[o] if j not in seen and j not in imp},
                key=lambda j: (round(float(np.real(h1[j, j])), 9), j),
            )
            seen.update(depth)
            order.extend(depth)
            frontier = depth
        chains[i] = order
    return chains


def _transposition(n_orb, impurity_pairs, chains, a, b):
    """The orbital permutation swapping impurity spatial orbitals ``a`` and ``b`` with their chains."""
    perm = list(range(n_orb))
    for (dn_a, up_a), (dn_b, up_b) in [(impurity_pairs[a], impurity_pairs[b])]:
        for x, y in ((dn_a, dn_b), (up_a, up_b)):
            if len(chains[x]) != len(chains[y]):
                return None
            for p, q in zip([x] + chains[x], [y] + chains[y]):
                perm[p], perm[q] = q, p
    return tuple(perm) if sorted(perm) == list(range(n_orb)) else None


def _sign_vectors(perm, impurity_pairs, tensor, h_imp, imp_index):
    """The per-spatial-orbital sign vectors under which ``perm`` leaves the impurity tensors unchanged."""
    local = np.array([imp_index[perm[o]] for o in sorted(imp_index)])
    for spatial_signs in product((1, -1), repeat=len(impurity_pairs)):
        s = np.ones(len(imp_index))
        for (dn, up), sign in zip(impurity_pairs, spatial_signs):
            s[imp_index[dn]] = s[imp_index[up]] = sign
        # c_a -> s_a c_{P(a)}: the image coefficient at P-indices is s_a s_b s_c s_d A[a, b, c, d].
        image = np.zeros_like(tensor)
        image[np.ix_(local, local, local, local)] = np.einsum("a,b,c,d,abcd->abcd", s, s, s, s, tensor)
        one = np.zeros_like(h_imp)
        one[np.ix_(local, local)] = np.outer(s, s) * h_imp
        if np.allclose(image, tensor, atol=1e-10) and np.allclose(one, h_imp, atol=1e-10):
            yield spatial_signs


def symmetry_generators(op, impurity_pairs, bath_pairs, n_orb, restrictions=None, weighted_restrictions=None):
    """Verified signed-permutation symmetries of ``op`` that generate the label orbits.

    Candidates: the spin flip (every ``(dn, up)`` pair swapped, impurity and bath) and, for every two
    impurity spatial orbitals with equal on-site energy and equally long bath chains, their
    transposition (both spins, chains carried along) under each sign assignment the impurity one- and
    two-body tensors allow. Each candidate is kept only if :func:`is_symmetry` and
    :func:`preserves_restrictions` both hold. Rank-local and deterministic; call it on one rank and
    broadcast.
    """
    impurity = sorted(o for pair in impurity_pairs for o in pair)
    imp_index = {o: k for k, o in enumerate(impurity)}
    h1 = extract_tensors(op, n_orb=n_orb, two_body=False)[0]
    tensor = impurity_two_body_tensor(op, impurity)
    h_imp = h1[np.ix_(impurity, impurity)]
    chains = _bath_chains(h1, impurity, n_orb)
    unit = (1,) * n_orb
    found = []

    def keep(sp):
        if is_symmetry(op, sp) and preserves_restrictions(sp, restrictions, weighted_restrictions):
            found.append(sp)

    if impurity_pairs or bath_pairs:
        perm = list(range(n_orb))
        for dn, up in list(impurity_pairs) + list(bath_pairs):
            perm[dn], perm[up] = up, dn
        keep(SignedPermutation(tuple(perm), unit, flips_spin=True))
    for a in range(len(impurity_pairs)):
        for b in range(a + 1, len(impurity_pairs)):
            dn_a, dn_b = impurity_pairs[a][0], impurity_pairs[b][0]
            if not np.isclose(h1[dn_a, dn_a], h1[dn_b, dn_b], atol=1e-10):
                continue
            perm = _transposition(n_orb, impurity_pairs, chains, a, b)
            if perm is None:
                continue
            for spatial_signs in _sign_vectors(perm, impurity_pairs, tensor, h_imp, imp_index):
                signs = [1] * n_orb
                for (dn, up), sign in zip(impurity_pairs, spatial_signs):
                    for o in [dn, up] + chains[dn] + chains[up]:
                        signs[o] = sign
                before = len(found)
                keep(SignedPermutation(perm, tuple(signs)))
                if len(found) > before:
                    break
    return found


def _gf2_solve(basis, target):
    """Coefficients (a bitmask over ``basis``) with ``XOR basis[k] == target``, or ``None``."""
    pivots = []  # (mask, combination, pivot bit), fully reduced
    for k, subset in enumerate(basis):
        mask, combo = subset, 1 << k
        for p_mask, p_combo, bit in pivots:
            if mask >> bit & 1:
                mask, combo = mask ^ p_mask, combo ^ p_combo
        if mask:
            bit = mask.bit_length() - 1
            pivots = [
                (p_mask ^ mask, p_combo ^ combo, b) if p_mask >> bit & 1 else (p_mask, p_combo, b)
                for p_mask, p_combo, b in pivots
            ]
            pivots.append((mask, combo, bit))
    combo = 0
    for p_mask, p_combo, bit in pivots:
        if target >> bit & 1:
            target, combo = target ^ p_mask, combo ^ p_combo
    return None if target else combo


def label_map(sp, parities):
    """The label map of ``sp``: ``(two_sz, parity) -> (two_sz', parity')``, or ``None``.

    A determinant ``D`` with parity ``p_k`` on subset ``A_k`` maps to ``P(D)``, whose parity on ``A_k``
    is ``D``'s on ``P^-1(A_k)``: an XOR of ``D``'s own parities whenever that preimage lies in the span
    of the conserved subsets. ``None`` when it does not (``P`` does not act on these labels). The spin
    flip negates ``2 S_z``.
    """
    inverse = [0] * len(sp.perm)
    for i, j in enumerate(sp.perm):
        inverse[j] = i
    combos = []
    for mask in parities:
        preimage = 0
        for bit in range(mask.bit_length()):
            if mask >> bit & 1:
                preimage |= 1 << inverse[bit]
        combo = _gf2_solve(parities, preimage)
        if combo is None:
            return None
        combos.append(combo)

    def apply(label):
        two_sz, parity = label
        image = tuple(sum(parity[k] for k in range(len(parity)) if combo >> k & 1) % 2 for combo in combos)
        return (None if two_sz is None else (-two_sz if sp.flips_spin else two_sz), image)

    return apply


def label_orbits(labels, generators, parities):
    """``{label: (representative, P)}`` over ``labels``, ``P`` taking the representative's problem to it.

    Breadth-first from each label in order, applying every generator whose label map exists; a label
    reached from an earlier one belongs to that orbit. Labels outside ``labels`` (a spin-flipped
    ``2 S_z`` that SU(2) left out) are passed through but never become representatives.
    """
    maps = [(sp, label_map(sp, parities)) for sp in generators]
    maps = [(sp, m) for sp, m in maps if m is not None]
    wanted = set(labels)
    n_orb = len(generators[0].perm) if generators else 0
    identity = SignedPermutation(tuple(range(n_orb)), (1,) * n_orb)
    orbit_of = {}
    for start in labels:
        if start in orbit_of:
            continue
        orbit_of[start] = (start, identity)
        frontier, seen = [(start, identity)], {start}
        while frontier:
            nxt = []
            for label, to_label in frontier:
                for sp, m in maps:
                    image = m(label)
                    if image in seen:
                        continue
                    seen.add(image)
                    composed = sp.compose(to_label)
                    if image in wanted and image not in orbit_of:
                        orbit_of[image] = (start, composed)
                    nxt.append((image, composed))
            frontier = nxt
    return orbit_of
