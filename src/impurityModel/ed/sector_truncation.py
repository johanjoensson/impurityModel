"""Cap enforcement for the thermal union: importance scores, whole-unit and whole-orbit cuts.

Split from :mod:`impurityModel.ed.sector_solver` (see its docstring for the method). It does not
import ``sector_solver``.
"""

import numpy as np
from mpi4py import MPI

from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant


def _group_size(group, comm):
    """The summed length of ``group`` over ``comm``: an upper bound on what adding it grows a basis by."""
    n = len(group)
    return n if comm is None or comm.size == 1 else int(comm.allreduce(n, op=MPI.SUM))


def _importance(psis):
    """This rank's ``{determinant: max |amplitude|^2 over the columns of psis}``."""
    keys = list(psis.keys())
    if not keys:
        return {}
    weight = (np.abs(np.array(psis)) ** 2).reshape(len(keys), -1).max(axis=1)
    return dict(zip(keys, weight.tolist()))


#: Resolution of :func:`_score_cut`'s histogram over scores in ``[0, 1]``.
_SCORE_BINS = 1024


def _scores(dets, importance=None):
    """A rank-independent score in ``[0, 1]`` per determinant; the lowest scores are kept first.

    From ``importance`` (a determinant-to-weight map) the score is ``-log10`` of the weight, scaled to
    ``[0, 1]`` over 30 decades; without it, the determinant's hash as a fraction of the hash range. Both
    are functions of the determinant alone, so a cut made on them is the same set on any number of ranks.
    """
    if importance is None:
        return [det.get_hash() / 2.0**64 for det in dets]
    return [min(1.0, -np.log10(max(importance.get(det, 0.0), 1e-30)) / 30.0) for det in dets]


def _aligned_weights(keys, psis):
    """float32 ``|amplitude|^2`` (max over the columns of ``psis``) of each determinant of ``keys``, in key order."""
    weight = _importance(psis)
    return np.fromiter((weight.get(d, 0.0) for d in keys.keys()), dtype=np.float32, count=len(keys))


def _class_importance(kept, i, fallback, comm):
    """The weight map the cut ranks class ``i`` by: its own, else the winner's if that describes it, else ``None``."""
    own = kept.importance[i]
    if own is not None:
        return dict(zip(kept[i].keys(), own.tolist()))
    return _covered_importance(kept[i], fallback, comm)


def _covered_importance(keys, importance, comm):
    """``importance`` when it describes the class ``keys`` (at least half its determinants carry weight), else ``None``.

    The importance map is one class's amplitudes. Another class (another block of the per-label window solve) has
    none of those determinants, so every one of its scores would tie at the maximum and no share of it would
    ever fit; it is ranked by hash instead. Decided on the global counts, so every rank decides alike. Collective.
    """
    if importance is None:
        return None
    counts = np.array([sum(1 for d in keys.keys() if d in importance), len(keys)], dtype=np.int64)
    if comm is not None and comm.size > 1:
        total = np.empty_like(counts)
        comm.Allreduce(counts, total, op=MPI.SUM)
        counts = total
    return importance if 2 * counts[0] >= counts[1] and counts[1] > 0 else None


def _score_cut(scores, share, comm):
    """The score below which a ``share`` of all determinants (summed over ranks) lies: a histogram cut. Collective."""
    hist = np.histogram(np.asarray(scores, dtype=float), bins=_SCORE_BINS, range=(0.0, 1.0))[0].astype(np.int64)
    if comm is not None and comm.size > 1:
        total_hist = np.empty_like(hist)
        comm.Allreduce(hist, total_hist, op=MPI.SUM)
        hist = total_hist
    total = int(hist.sum())
    if share >= 1.0 or total == 0:
        return 1.0
    reached = int(np.searchsorted(np.cumsum(hist), share * total))
    return (reached + 1) / _SCORE_BINS


def _unit_groups(dets, rows_of, comm):
    """``[class determinants, *partner rows]`` of ``dets``, the row count replicated over ``comm``.

    A rank holding none of the class builds no rows; every rank must still add the same number of
    groups, so the list is padded to the longest.
    """
    rows = rows_of(dets)
    n_rows = len(rows) if comm is None or comm.size == 1 else int(comm.allreduce(len(rows), op=MPI.MAX))
    return [list(dets.keys()), *rows, *([] for _ in range(n_rows - len(rows)))]


def _grow_with(merged, groups):
    """A copy of ``merged`` holding ``groups`` as well (``merged`` itself when the copy is not needed)."""
    trial = merged.clone(initial_basis=list(merged.local_basis))
    for group in groups:
        trial.add_states(group)
    return trial


def _unit_importance(importance, i, keys, comm):
    """Unit ``i``'s weight map: ``importance[i]`` if a list (one per unit), else the winner's map for unit 0 only."""
    if isinstance(importance, list):
        return importance[i]
    return _covered_importance(keys, importance, comm) if i == 0 else None


def _add_units_within_cap(merged, units, rows_of, cap, *, truncate, importance=None, derived=None):
    """Add classes and their partner rows to ``merged`` within ``cap``: ``(merged, whole)``. Collective.

    ``units`` are class key sets added together. Decided on the exact, replicated size of the union,
    never on a summed estimate. Units that fit are added whole. If they do not, they are left out
    (``truncate`` False), or with ``truncate`` every unit is cut to the same share of its *class*, the
    largest that fits together with the partner rows of the cut classes -- partners are regenerated from
    the cut class, so a multiplet keeps its shape, and no degenerate class is starved to feed another.
    The cut keeps a class's determinants by ``importance`` (a determinant-to-weight map, e.g. the
    winner's amplitudes; the first unit only) and otherwise by hash. ``derived[i]`` is ``None`` or
    ``(j, image)``: unit ``i`` is then ``image`` of whatever unit ``j`` keeps, not cut on its own.
    """
    derived = derived or [None] * len(units)
    comm = merged.comm
    groups = [group for keys in units for group in _unit_groups(keys, rows_of, comm)]
    if cap is None:
        for group in groups:
            merged.add_states(group)
        return merged, True
    if merged.size + sum(_group_size(group, comm) for group in groups) <= cap:  # fits even with no overlap
        for group in groups:
            merged.add_states(group)
        return merged, True
    trial = _grow_with(merged, groups)
    if trial.size <= cap:
        return trial, True
    if not truncate:
        return merged, False
    scores = [
        (_scores(keys.keys(), _unit_importance(importance, i, keys, comm)) if derived[i] is None else None)
        for i, keys in enumerate(units)
    ]
    best, lo, hi = merged, 0.0, 1.0
    for _ in range(_FIT_BISECTIONS):
        mid = 0.5 * (lo + hi)
        groups, shares = [], {}
        for i, (keys, score) in enumerate(zip(units, scores)):
            if derived[i] is not None:
                source, image = derived[i]
                shares[i] = image(shares[source])
            else:
                cut = _score_cut(score, mid, comm)
                shares[i] = ManyBodyState.from_keys([det for det, sc in zip(keys.keys(), score) if sc <= cut])
            groups += _unit_groups(shares[i], rows_of, comm)
        trial = _grow_with(merged, groups)
        if trial.size <= cap:
            best, lo = trial, mid
        else:
            hi = mid
    return best, False


def _orbit_reps(orbits, basis, keys, importance):
    """``(reps, scores)`` of the orbits the class ``keys`` touch that this rank owns. Collective.

    ``scores`` in ``[0, 1]`` as :func:`_scores`: from ``importance`` (summed over an orbit's members) where given,
    else the hash of the orbit's representative -- a function of the orbit alone, so a cut made on it is the same on
    any number of ranks and every member of an orbit is in or out together.
    """
    dets = list(keys.keys())
    weights = np.array([importance.get(d, 0.0) for d in dets]) if importance is not None else np.ones(len(dets))
    reps, sums = orbits._owned_orbits(basis, dets, [weights])
    reps, _images = orbits.canonical(reps)
    if importance is not None:
        scores = [min(1.0, -np.log10(max(float(w), 1e-30)) / 30.0) for w in sums[:, 0]]
    else:
        scores = [d.get_hash() / 2.0**64 for d in orbits.representative_det(reps)]
    return reps, np.array(scores, dtype=float)


def _orbit_members(orbits, reps):
    """The determinants (restriction-filtered) of the orbits with patterns ``reps``, generated by this rank."""
    bits = orbits.members(reps) if len(reps) else np.zeros((0, orbits.n_orb), dtype=np.uint8)
    return _keys_of_bits(bits, orbits.n_bytes)


def _keys_of_bits(bits, n_bytes):
    if not len(bits):
        return []
    packed = np.packbits(bits, axis=1)
    packed = np.pad(packed, ((0, 0), (0, n_bytes - packed.shape[1])))
    return [SlaterDeterminant.from_bytes(bytes(row)) for row in packed]


def _add_orbits_within_cap(merged, units, orbits, cap, *, truncate):
    """:func:`_add_units_within_cap` on whole symmetry orbits: ``(merged, whole)``. Collective.

    ``units`` are ``(reps, scores)`` per class (:func:`_orbit_reps`). A unit goes in as every member of every one of
    its orbits; a mirror class is the same orbits (the mirror's symmetry is in the group), so it needs no unit of
    its own. If the union does not fit, with ``truncate`` every unit keeps the same share of *its orbits*, by score
    -- never part of an orbit -- the largest share that fits; none is starved to feed another unless even the
    smallest share does not fit.
    """
    comm = merged.comm

    def groups_of(reps_per_unit):
        return [_orbit_members(orbits, reps) for reps in reps_per_unit]

    groups = groups_of([reps for reps, _scores in units])
    if cap is None or merged.size + sum(_group_size(g, comm) for g in groups) <= cap:
        for group in groups:
            merged.add_states(group)
        return merged, True
    trial = _grow_with(merged, groups)
    if trial.size <= cap:
        return trial, True
    if not truncate:
        return merged, False
    best, lo, hi = merged, 0.0, 1.0
    for _ in range(_FIT_BISECTIONS):
        mid = 0.5 * (lo + hi)
        shares = []
        for reps, scores in units:
            cut = _score_cut(scores, mid, comm)
            shares.append(reps[scores <= cut])
        trial = _grow_with(merged, groups_of(shares))
        if trial.size <= cap:
            best, lo = trial, mid
        else:
            hi = mid
    return best, False


#: Bisection steps of :func:`_add_units_within_cap`'s truncation: the share kept is exact to ``2**-8``.
_FIT_BISECTIONS = 8
