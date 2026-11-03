"""An independent, pure-Python reference for ``ManyBodyOperator.apply_block``.

``apply``, ``__call__`` and ``apply_multi`` all forward to the one C++ ``apply``, so comparing any of them
with ``apply_block`` compares the code with itself. This module is the oracle that does not share a line
with it: a determinant is a ``frozenset`` of occupied orbitals, a term is applied operator by operator
(rightmost first), the fermion sign is the parity of the occupied orbitals below the touched one, and the
restrictions are evaluated on the output determinant.

Orbital ``idx`` lives in chunk ``idx // 64``, bit ``63 - idx % 64`` (MSB-first); a determinant of ``n_chunks``
words holds orbitals ``0 .. 64 * n_chunks - 1`` and creating one beyond that gives zero, as in C++.
"""

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant
from impurityModel.ed.symmetries import weighted_parts


def sd_from_orbitals(orbitals, n_chunks):
    """Pack occupied orbital indices into a ``SlaterDeterminant`` of ``n_chunks`` words."""
    chunks = [0] * n_chunks
    for idx in orbitals:
        chunks[idx // 64] |= 1 << (63 - idx % 64)
    return SlaterDeterminant(tuple(chunks))


def orbitals_from_chunks(chunks):
    return frozenset(64 * c + (63 - b) for c, w in enumerate(chunks) for b in range(64) if (int(w) >> b) & 1)


def block_from_orbital_sets(states, n_chunks):
    """``{frozenset(orbitals): amplitude row}`` -> a ``ManyBodyState`` block of the common width."""
    return ManyBodyState({sd_from_orbitals(o, n_chunks): list(a) for o, a in states.items()})


def _apply_string(string, occ, n_chunks):
    """Apply one term's operator string (stored apply-order-last-first, as written) to ``occ``; ``(sign, occ)``."""
    sign = 1
    occ = set(occ)
    for orb, kind in reversed(string):  # the rightmost operator acts first
        below = sum(1 for o in occ if o < orb)
        if kind == "c":
            if orb in occ or orb >= 64 * n_chunks:
                return 0, None
            occ.add(orb)
        else:
            if orb not in occ:
                return 0, None
            occ.discard(orb)
        if below % 2:
            sign = -sign
    return sign, frozenset(occ)


def _within(occ, restrictions, weighted):
    for orbs, (lo, hi) in (restrictions or {}).items():
        if not lo <= len(occ & orbs) <= hi:
            return False
    for entry in weighted or []:
        weights, (lo, hi), _tail = weighted_parts(entry)
        if not lo <= sum(weights.get(o, 0) for o in occ) <= hi:
            return False
    return True


def reference_apply(terms, states, n_chunks, restrictions=None, weighted=None):
    """``{frozenset(occ): ndarray(p)}`` image of ``states`` under ``terms`` (a ``{string: coeff}`` dict).

    Exactly-zero rows are dropped, as ``apply(..., cutoff=0)`` does.
    """
    out = {}
    for occ, raw_amp in states.items():
        amp = np.asarray(raw_amp, dtype=complex)
        for string, coeff in terms.items():
            sign, new = _apply_string(string, occ, n_chunks)
            if sign == 0 or not _within(new, restrictions, weighted):
                continue
            out.setdefault(new, np.zeros_like(amp))
            out[new] = out[new] + coeff * sign * amp
    return {k: v for k, v in out.items() if np.any(v != 0)}


def block_to_orbital_sets(block):
    """The inverse of :func:`block_from_orbital_sets` for an applied block."""
    return {
        orbitals_from_chunks(tuple(k)): np.asarray(v, dtype=complex).reshape(-1) for k, v in block.to_dict().items()
    }


def assert_matches(got, ref, tol=1e-12):
    """``got`` and ``ref`` are ``{occ: ndarray}``; a determinant missing on one side counts as zero."""
    for k in got.keys() | ref.keys():
        g = got.get(k)
        r = ref.get(k)
        if g is None:
            g = np.zeros_like(r)
        if r is None:
            r = np.zeros_like(g)
        scale = max(1.0, float(np.max(np.abs(r))))
        assert np.max(np.abs(g - r)) <= tol * scale, f"mismatch at {sorted(k)}: {g} vs {r}"


def state_to_orbital_sets(psi):
    """A width-1 (or any-width) ``ManyBodyState`` as ``{frozenset(occ): ndarray}``."""
    return block_to_orbital_sets(psi)
