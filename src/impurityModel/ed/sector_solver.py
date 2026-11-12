r"""Certified fixed-N sector solves: every symmetry class that can hold the ground state is expanded.

A CIPSI expansion admits only determinants in the support of ``H|psi_ref>``, and its reference is,
from the first cycle on, the lowest state of its own truncated basis: an expansion therefore never
leaves the symmetry class that state is in (a class *diagonal on determinants*), or grows only
slowly towards another one (a class that is not). The seed's lowest state -- the atomic multiplet,
before hybridization -- is in the wrong class whenever hybridization reorders the multiplets. On the
NiO 10-bath gap-DC archive every sector from d5 to d9 converged on an exact excited level of another
symmetry, up to 0.17 above the ground state (hole-space exact diagonalization, d6-d9).

Nothing about the symmetries is assumed; they are detected from ``H``, in two kinds:

* **Hard labels**, diagonal on determinants: ``2 S_z`` when ``H`` conserves it, and the occupation
  parities ``(-1)^{N_A}`` it conserves (:func:`symmetries.conserved_parities`). An expansion seeded
  inside one label never leaves it, exactly. When total spin is also a symmetry (``[H, S_+] = 0``)
  only the labels at the smallest ``|2 S_z|`` are needed: every multiplet has a member there.
* **Soft classes**, every other symmetry ``H`` has, known or not: inside each hard label the seed is
  partitioned by ``P H P`` and ``P H Q H P`` (:func:`seed_sectors.seed_class_plan`, with no
  spin assumption), and each class is expanded from its own seed states, held there by the class
  penalty.

Total spin is deliberately *not* used as a soft label: it need not be a good quantum number, and
where it is, a ``lambda S^2`` penalty on a truncated basis -- not closed under ``S_+-`` -- steered the
NiO d7 sector into a spin-mixed state 0.053 high at three of four shifts while being necessary for
d6.

Each class reports its unpenalized lowest eigenvalue -- variational whatever the penalty did -- and
the lowest class wins. Nothing is deduplicated: two labels or classes with equal *seed* spectra are
not equivalent (the bath energies enter only through ``Q H Q``; review measured a label merged that
way dropping the ground state by 0.04-0.12 on a toy with differently fitted bath levels). What keeps
the cost near a single expansion's is that a class whose state has left its seed span and
re-converged on an energy another class already found stops early (:data:`LEAK_ENERGY_TOL`).
"""

from dataclasses import dataclass

import numpy as np

from impurityModel.ed.basis_transcription import comm_support_filter
from impurityModel.ed.cipsi_solver import (
    DEGENERACY_TOL,
    DEGENERACY_TOL_MARGIN,
    CIPSISolver,
    _degeneracy_tol,
    _energy_cut_indices,
)
from impurityModel.ed.degeneracy import degenerate_groups
from impurityModel.ed.eigensolvers import Lowest, cold_vector
from impurityModel.ed.ManyBodyUtils import ManyBodyState, applyOp
from impurityModel.ed.memory_estimate import emit_memory_warning, measured_phase
from impurityModel.ed.observables import (
    casimir_operator,
    casimir_to_quantum_number,
    make_spin_operators,
    manifold_observable_values,
)
from impurityModel.ed.sector_certify import (
    CertificationPlan,
    KeptClasses,
    _bits,
    _energy_error,
    _label_window,
    certify_sector,
    effective_ranking_cap,
)
from impurityModel.ed.sector_truncation import (
    _add_orbits_within_cap,
    _add_units_within_cap,
    _class_importance,
    _importance,
    _orbit_reps,
)
from impurityModel.ed.symmetries import (
    conserved_parities,
    parity_labels,
)
from impurityModel.ed.truncation_flags import is_cap_bound


def _spin_partner_keys(keys, pairs, max_two_sz, n_orb):
    """This rank's share of the spin partners' supports of the states on ``keys``: ``S_+-^k``, ``k >= 1``.

    All rows of :func:`_spin_partner_rows`, flat.
    """
    return [det for row in _spin_partner_rows(keys, pairs, max_two_sz, n_orb) for det in row]


def _spin_partner_rows(keys, pairs, max_two_sz, n_orb):
    """This rank's share of each partner row of the states on ``keys``: ``rows[k - 1]`` holds ``S_+^k`` and ``S_-^k``.

    A class expansion holds one ``S_z`` row of each multiplet it found (its hard label's, the
    smallest ``|S_z|``); the other rows of every multiplet inside the thermal window must be in the
    window solve's basis at the same quality, or they sit at seed quality outside the window and
    the manifold comes back short (measured: 5 of 8 states of a spin-and-orbitally degenerate
    level). The ladders are applied to a state with generic amplitudes on ``keys`` -- every
    determinant of the class -- so no image cancels. Each rank steps its own rows; by linearity the
    union over the ranks covers every partner, so this needs no communication -- ``add_states``
    routes the keys afterwards.

    Stepped only as far as ``|2 S_z| <= max_two_sz``, the largest seed ``2 S_z``, which bounds the
    ``2 S`` of every multiplet the window holds. The ladders were once stepped ``2 max_two_sz``
    times each way: on the NiO linked-chain d8 sector (``max_two_sz = 2``) a quarter of the union
    was then rows at ``2 S_z = +-4, +-6``, which no state of the window has, and the union is the
    ground state's memory peak. The rows are by distance from the class's own ``S_z`` so that a
    binding cap keeps the nearest, ``+-`` together.
    """
    dets = keys.keys()
    if not len(dets):
        return []
    s_plus, s_minus, _s_z = make_spin_operators(pairs)
    sign = np.zeros(n_orb, dtype=np.int64)
    for dn, up in pairs:
        sign[up], sign[dn] = 1, -1
    # Every determinant of a class carries its label's 2 S_z (a mirror image too: its own label's).
    two_sz = int(_bits([next(iter(dets))], n_orb).astype(np.int64)[0] @ sign)
    hashes = np.fromiter((det.get_hash() for det in dets), dtype=np.uint64, count=len(dets))
    # The eigensolvers' hash-seeded generic vector: deterministic, and no two images cancel.
    state = ManyBodyState(dict(zip(dets, cold_vector(hashes)[:, 0])))
    ups, downs = [], []
    for ladder, steps, found in (
        (s_plus, (max_two_sz - two_sz) // 2, ups),
        (s_minus, (max_two_sz + two_sz) // 2, downs),
    ):
        ladder.set_restrictions(None)
        ladder.set_weighted_restrictions(None)
        image = state
        for _ in range(max(0, int(steps))):
            image = applyOp(ladder, image, cutoff=0)
            if not len(image):
                break
            found.append(list(image.keys()))
    return [
        (ups[k] if k < len(ups) else []) + (downs[k] if k < len(downs) else [])
        for k in range(max(len(ups), len(downs)))
    ]


def manifold_deficit(plan, verdicts, es, psis, basis):
    """``(found, expected)``: the states a ``Within`` solve holds in its lowest level, and a lower bound.

    The bound comes from what certification measured, with no symmetry assumed beyond the detected
    ones. Every hard label with a class at the winning energy (within both error bars) holds a row
    of the lowest level of its own -- labels are exact, so those rows are orthogonal -- and when total
    spin is a symmetry each such row is one ``S_z`` member of a ``2S + 1`` multiplet, ``S`` read
    off ``<S^2>`` on the level. A short manifold misweights every thermal average without changing
    its energy. Collective: every rank must call it (the ``<S^2>`` evaluation reduces).
    """
    comm = basis.comm if basis.is_distributed else None
    finished = [v for v in verdicts if not v.stopped_early] or list(verdicts)
    lowest = min(finished, key=lambda v: v.energy)
    labels = {
        (v.job.two_sz, v.job.parity)
        for v in finished
        if v.energy - lowest.energy <= _energy_error(v) + _energy_error(lowest)
    }
    es = np.real(np.asarray(es, dtype=complex))
    if comm is not None:
        es = comm.bcast(es, root=0)
    tol = _energy_error(lowest)
    level = np.flatnonzero(es - np.min(es) <= tol)
    multiplicity = 1
    if plan.su2 and plan.pairs:
        s2 = _total_spin_casimir(plan)
        # The level's columns only, and S^2 applied onto their own support: the matrix of S^2 on the
        # level is all that is read, and the rest of S^2 |n> is a transient as large as the window.
        block = basis.redistribute_block(psis.select([int(i) for i in level]))
        keep = comm_support_filter([block], comm)
        values = manifold_observable_values(
            block,
            es[level],
            lambda blk: s2.apply_block(blk, 0, keep),
            degeneracy_tol=tol,
            comm=comm,
            redistribute=basis.redistribute_block if comm is not None else None,
        )
        # The smallest S on the level: a lower bound stays one if the level is accidentally mixed.
        spin = casimir_to_quantum_number(float(np.min(np.real(values))))
        multiplicity = round(2 * spin + 1)
        if comm is not None:
            multiplicity = comm.bcast(multiplicity, root=0)
    return len(level), len(labels) * multiplicity


def _total_spin_casimir(plan):
    """``S^2`` over the plan's spin pairs, unrestricted (it is applied to states, not used to build a basis)."""
    ladders = make_spin_operators(plan.pairs)
    for op in ladders:
        op.set_restrictions(None)
        op.set_weighted_restrictions(None)
    s2 = casimir_operator(*ladders)
    s2.set_restrictions(None)
    s2.set_weighted_restrictions(None)
    return s2


def roundoff_tol(es, *, dense, slater_weight_min):
    """How far apart two exactly degenerate eigenvalues of one solve can come back: its own accuracy.

    A dense ``eigh`` is exact to rounding, ``DEGENERACY_TOL_MARGIN * eps * max|E|`` (floored at
    :data:`~impurityModel.ed.cipsi_solver.DEGENERACY_TOL`); an iterative window solve to its residual,
    :func:`~impurityModel.ed.cipsi_solver._degeneracy_tol`. Pure: the caller passes replicated ``es``.
    """
    if not dense:
        return _degeneracy_tol(es, slater_weight_min)
    e = np.real(np.asarray(es, dtype=complex))
    scale = float(np.max(np.abs(e))) if e.size else 0.0
    return max(DEGENERACY_TOL, DEGENERACY_TOL_MARGIN * float(np.finfo(float).eps) * scale)


def split_multiplets(energies, two_s, roundoff):
    """The decision of :func:`multiplet_split` on per-state energies and ``2S``: ``None`` or the split record. Pure.

    Clusters at ``roundoff``; for each ``2S`` the *partial* clusters are those holding a count of that ``S`` that is
    not a multiple of ``2S + 1``. One partial cluster is an incomplete level, two or more a split multiplet; the
    spread is taken across the partial clusters of that ``S`` only.
    """
    energies = np.asarray(energies, dtype=float)
    two_s = np.asarray(two_s, dtype=int)
    order = np.argsort(energies, kind="stable")
    clusters = [order[c] for c in degenerate_groups(energies[order], roundoff, anchor="first")]
    bad, spread = [], 0.0
    for ts in sorted(set(two_s.tolist())):
        partial = [c for c in clusters if int(np.count_nonzero(two_s[c] == ts)) % (ts + 1)]
        if len(partial) > 1:
            bad.append(int(ts))
            members = np.concatenate([c[two_s[c] == ts] for c in partial])
            spread = max(spread, float(np.ptp(energies[members])))
    return {"two_s": bad, "spread": spread, "roundoff": float(roundoff)} if bad else None


def multiplet_split(plan, verdicts, es, psis, basis, roundoff, *, certified=True):
    """The spin multiplets of the lowest level that the solve split, or ``None`` (review S2, detection).

    Under requirement SYM the members of one ``2S + 1`` multiplet are degenerate to the solve's own accuracy
    ``roundoff``, whatever the cap did: the selection and the cuts act on whole symmetry orbits. So group the
    lowest level (the states within the winning class's error bar, as :func:`manifold_deficit`) into clusters
    at ``roundoff``, read ``S`` of every state from ``S^2`` diagonalised on its cluster, and count: every cluster
    holds a multiple of ``2S + 1`` states of each ``S``. One cluster that does not is an *incomplete* level (copies
    missing: :func:`manifold_deficit`'s business, not a split); two or more that do not hold parts of a multiplet
    that came back at different energies -- a split, which under SYM can only come from a symmetry-breaking basis.
    An uncertified window solve (``certified=False``: stagnated above its tolerance, so ``roundoff`` overstates
    its accuracy) is not judged. ``certified`` is read on rank 0 only; the ``S^2`` evaluation runs regardless.

    Only when total spin is a symmetry (``plan.su2``); without it (spin-orbit coupling, CrI3) nothing is
    checked and ``None`` is returned. Returns ``{"two_s": [...], "spread": float, "roundoff": float}``:
    the split ``2S`` values and the largest energy spread across the partial clusters of one ``S``. Replicated:
    decided on rank 0 and broadcast. Collective when ``plan.su2`` (the ``S^2`` evaluation reduces).
    """
    if not (plan.su2 and plan.pairs):
        return None
    comm = basis.comm if basis.is_distributed else None
    finished = [v for v in verdicts if not v.stopped_early] or list(verdicts)
    lowest = min(finished, key=lambda v: v.energy)
    es = np.real(np.asarray(es, dtype=complex))
    if comm is not None:
        es = comm.bcast(es, root=0)
    level = np.flatnonzero(es - np.min(es) <= _energy_error(lowest))
    s2 = _total_spin_casimir(plan)
    block = basis.redistribute_block(psis.select([int(i) for i in level]))
    keep = comm_support_filter([block], comm)
    values = manifold_observable_values(
        block,
        es[level],
        lambda blk: s2.apply_block(blk, 0, keep),
        degeneracy_tol=roundoff,
        comm=comm,
        redistribute=basis.redistribute_block if comm is not None else None,
    )
    split = None
    if (comm is None or comm.rank == 0) and certified:
        two_s = np.array([round(2 * casimir_to_quantum_number(float(np.real(x)))) for x in values])
        split = split_multiplets(es[level], two_s, roundoff)
    if comm is not None:
        split = comm.bcast(split, root=0)
    return split


def _record_union_cut(solver, union_cut, cap, retained):
    """Say in the solver's reports what the cap did to the thermal union (review S3).

    A union that left out or cut a class is a truncated basis: ``truncation_report`` gets ``union_cut``, and the
    convergence report is no longer ``converged``. The ``manifold_check`` rides on the convergence report, which
    is what reaches the ground-state JSON.
    """
    conv = solver.convergence_report
    if conv is None:
        conv = solver.convergence_report = {}
    check = getattr(solver, "manifold_check", None)
    if check is not None:
        conv["manifold_check"] = dict(check)
        # A state that is known to be incomplete, split or uncertified is not a converged one, whatever the PT2
        # residual says (review GS8): the flag is what the ensemble plan, the DC record and the report read.
        why = _manifold_degradation(check)
        if why is not None and union_cut is None and conv.get("converged") is not False:
            conv["converged"] = False
            conv["limited_by"] = conv.get("limited_by") or why
    if union_cut is None:
        return
    report = dict(
        solver.truncation_report
        or {
            "cycles": 0,
            "memory_bound": False,
            "discarded_de2_mass": 0.0,
            "discarded_de2_mass_total": 0.0,
            "n_candidates_last": 0,
            "residual_pt2": float(conv.get("residual_pt2") or 0.0),
        }
    )
    if "per_label" in union_cut:
        # Solved label by label (`_solve_window_per_label`): the labels' own flags decide, not the union's size.
        report.update(cap_hit=True, union_cut=dict(union_cut), retained=int(retained))
        report.setdefault("cap_bound", False)
        report.setdefault("threshold", int(cap) if cap is not None and np.isfinite(cap) else int(retained))
        solver.truncation_report = report
        # The blocks converged one by one; missing or split states of the merged level still degrade the state.
        why = _manifold_degradation(check) if check is not None else None
        if why is not None and conv.get("converged") is not False:
            conv["converged"] = False
            conv["limited_by"] = conv.get("limited_by") or why
        return
    report.update(cap_hit=True, cap_bound=True, union_cut=dict(union_cut), retained=int(retained))
    report.setdefault("threshold", int(cap) if cap is not None and np.isfinite(cap) else int(retained))
    solver.truncation_report = report
    conv["converged"] = False
    conv["limited_by"] = conv.get("limited_by") or "cap"


def _manifold_degradation(check):
    """Why a thermal manifold's ``manifold_check`` says the state is degraded (``None``: it does not)."""
    if check.get("multiplet_split") is not None:
        return "multiplet_split"
    if check.get("found", 0) < check.get("expected", 0) or check.get("window_complete") is False:
        return "incomplete_manifold"
    if check.get("certified") is False:
        return "uncertified"
    return None


def _raise_cap_advice(verdicts, *, widen):
    """What to do about a window the cap or the memory guard cut short.

    A class stopped by the measured-memory guard (``memory_bound``) was held below its cap by the RAM the rank
    actually had, so raising ``truncation_threshold`` cannot help: the advice then names the guard.
    """
    if any(v.memory_bound for v in verdicts):
        return (
            "The measured-memory guard held a class below its determinant cap, so raising truncation_threshold "
            f"will not help: give each rank more memory (fewer ranks per node, or more nodes), or {widen}."
        )
    return f"Raise the determinant cap (truncation_threshold), or {widen}."


def window_deficit(verdicts, es, within, lowest_expected):
    """``(found, expected)`` for the whole thermal window, not only its lowest level (review C7).

    ``expected`` is a lower bound: the lowest level's bound ``lowest_expected`` plus one state per hard label
    in every further cluster of distinct class energies inside the window. Classes that leaked into another
    class's state are left out (their energy is that class's), and a class is in the window by its *own*
    error bar. Labels are exact, so the states of different labels at one energy are different states; a
    class's multiplicity above the lowest level is not read (it is one at least). ``found`` counts the
    eigenvalues within the window. Pure and replicated: no collective.
    """
    finished = [v for v in verdicts if not v.stopped_early] or list(verdicts)
    lowest = min(finished, key=lambda v: v.energy)
    inside = sorted(
        (v for v in finished if v.energy - lowest.energy <= within + _energy_error(v)), key=lambda v: v.energy
    )
    clusters = []  # [last energy, last error, {labels}]
    for v in inside:
        label = (v.job.two_sz, v.job.parity)
        if clusters and v.energy - clusters[-1][0] <= _energy_error(v) + clusters[-1][1]:
            clusters[-1][2].add(label)
            clusters[-1][0], clusters[-1][1] = v.energy, _energy_error(v)
        else:
            clusters.append([v.energy, _energy_error(v), {label}])
    expected = int(lowest_expected) + sum(len(c[2]) for c in clusters[1:])
    # A class that converged its window holds every state of its label there: labels are exact, so the sum over
    # labels of the most any one class of the label found is a bound of its own (the classes of a label may
    # share states, hence the max and not the sum).
    per_label = {}
    for v in inside:
        n = max(1, sum(1 for e in v.levels if e - lowest.energy <= within + _energy_error(v)))
        label = (v.job.two_sz, v.job.parity)
        per_label[label] = max(per_label.get(label, 0), n)
    expected = max(expected, sum(per_label.values()))
    es = np.sort(np.real(np.asarray(es, dtype=complex)))
    top = lowest.energy + within + max((_energy_error(v) for v in inside), default=0.0)
    # The same absolute edge `expected` counts to: `es[0]` can sit below `lowest.energy` once the union is solved.
    found = int(np.count_nonzero(es <= top)) if len(es) else 0
    return found, expected


def _thermal_union(merged, plan, kept, importance, orbits, cap, max_two_sz, n_orb, verbose):
    """Grow ``merged`` (the seed) by the classes of ``kept`` within ``cap``: ``(merged, union_cut, cut)``. Collective.

    The classes at the winning level go in together and are cut together; every later class goes in whole or stops
    the union. ``union_cut`` is the report dict, ``cut`` the list of messages (empty when everything fit).
    """
    cut = []

    def rows_of(dets):
        return _spin_partner_rows(dets, plan.pairs, max_two_sz, n_orb) if plan.su2 else []

    # The classes at the winning energy make the level the window exists to hold: they go in
    # together and are cut together. Every later class goes in whole or stops the union.
    level = [i for i, own in enumerate(kept.in_window) if own or kept.at_winning_level[i]]
    local = {i: n for n, i in enumerate(level)}
    if orbits is not None:
        # Whole symmetry orbits instead of classes plus ladder rows: the seed is closed first, so the union
        # (and any cut of it) is made of whole orbits. A mirror is the same orbits as its source.
        seed_reps = _orbit_reps(orbits, merged, ManyBodyState.from_keys(list(merged.local_basis)), None)
        merged, _ = _add_orbits_within_cap(merged, [seed_reps], orbits, None, truncate=False)
        level_sources = [i for i in level if kept.derived[i] is None]
        units = [
            _orbit_reps(
                orbits, merged, kept[i], _class_importance(kept, i, importance if n == 0 else None, merged.comm)
            )
            for n, i in enumerate(level_sources)
        ]
        # A mirror is the same orbits only if its symmetry is in the group (a generator that did not map
        # spin pairs onto pairs, or a group over the cap, leaves it out): its orbits not already there join.
        held = {bytes(r) for reps, _s in units for r in reps}
        for i in (j for j in level if kept.derived[j] is not None):
            reps, scores = _orbit_reps(orbits, merged, kept[i], None)
            fresh = np.array([bytes(r) not in held for r in reps], dtype=bool)
            units.append((reps[fresh], scores[fresh]))
        merged, whole = _add_orbits_within_cap(merged, units, orbits, cap, truncate=True)
    else:
        merged, whole = _add_units_within_cap(
            merged,
            [kept[i] for n, i in enumerate(level)],
            rows_of,
            cap,
            truncate=True,
            importance=[
                _class_importance(kept, i, importance if n == 0 else None, merged.comm) for n, i in enumerate(level)
            ],
            derived=[
                None if kept.derived[i] is None else (local[kept.derived[i][0]], kept.derived[i][1]) for i in level
            ],
        )
    later = [i for i in range(len(kept)) if i not in local]
    union_cut = {
        "level_cut": not whole,
        "level_classes": len(level),
        "later_left_out": 0,
        "later_classes": len(later),
    }
    if not whole:
        cut.append(f"the {len(level)} class(es) inside the window were cut to a common share")
        union_cut["later_left_out"] = len(later)
        if later:
            cut.append(f"{len(later)} class(es) admitted by the margin alone not added: the window did not fit")
    else:
        # A source goes in with its mirrors or not at all: an asymmetric union breaks SYM.
        sources = [i for i in later if kept.derived[i] is None]
        for pos, i in enumerate(sources):
            family = [i] + [j for j in later if kept.derived[j] is not None and kept.derived[j][0] == i]
            if orbits is not None:
                merged, whole = _add_orbits_within_cap(
                    merged, [_orbit_reps(orbits, merged, kept[i], None)], orbits, cap, truncate=False
                )
            else:
                merged, whole = _add_units_within_cap(merged, [kept[j] for j in family], rows_of, cap, truncate=False)
            if not whole:
                left = sum(
                    1 + sum(1 for j in later if kept.derived[j] and kept.derived[j][0] == k) for k in sources[pos:]
                )
                union_cut["later_left_out"] = left
                cut.append(f"{left} later class(es) left out")
                break
    if cut and (merged.comm is None or merged.comm.rank == 0) and verbose:
        print(
            f"  thermal window: held at the cap ({merged.size:,} of {int(cap):,} determinants): " + "; ".join(cut),
            flush=True,
        )
    return merged, union_cut, cut


def _label_blocks(plan, kept, basis, orbits, parities, n_orb):
    """``(node_of, blocks)``: the blocks the per-label window solve runs in. Collective.

    A node is ``(2 S_z or None, parity)`` (``2 S_z`` is part of the node only when the labels carry
    one). The orbit group may map a node's determinants onto another node's (a spin flip changes both labels),
    and a block must hold
    whole orbits, so the nodes one orbit touches share a block. ``node_of(bits)`` is the node of every row;
    ``blocks`` maps each in-window node to the frozenset of nodes of its block.
    """
    sign = np.zeros(n_orb, dtype=np.int64)
    for dn, up in plan.pairs:
        sign[up], sign[dn] = 1, -1

    def node_of(bits):
        par = parity_labels(bits, parities) if parities else np.zeros((len(bits), 0), dtype=np.int64)
        sz = np.asarray(bits, dtype=np.int64) @ sign if has_sz else np.zeros(len(bits), dtype=np.int64)
        return [(int(z) if has_sz else None, tuple(int(c) for c in row)) for z, row in zip(sz, par)]

    # 2 S_z is part of the node exactly when the labels carry one (not with total spin, and not with spin-orbit
    # coupling, where S_z is not conserved): the node of a determinant must be comparable with a label's.
    has_sz = any(lab[0] is not None for lab in kept.labels)
    nodes = {(lab[0] if has_sz else None, lab[1]) for lab, w in zip(kept.labels, kept.in_window) if w}
    joined = []
    if orbits is not None and not orbits.trivial:
        probe = []
        for i, w in enumerate(kept.in_window):
            if w:
                probe.extend(list(kept[i].keys())[:64])
        probe.extend(list(basis.local_basis)[:64])
        if probe:
            patterns = np.unique(orbits.pattern(_bits(probe, n_orb)), axis=0)
            saved, orbits.det_filter = orbits.det_filter, None
            try:
                for pattern in patterns:
                    members = orbits.members(pattern[None, :])
                    if len(members):
                        joined.append(sorted(set(node_of(members)), key=repr))
            finally:
                orbits.det_filter = saved
    if basis.comm is not None:
        joined = [group for part in basis.comm.allgather(joined) for group in part]
    block = {node: frozenset([node]) for node in nodes}
    for group in joined:
        merged = set(group)
        for node in group:
            merged |= block.get(node, frozenset([node]))
        merged = frozenset(merged)
        for node in merged:
            block[node] = merged
        for node in nodes:
            if block[node] & merged:
                block[node] = merged
    return node_of, {node: block[node] for node in nodes}


def _block_key(block):
    """Sort key of a block of nodes that does not depend on set iteration order (replicated across ranks)."""
    return sorted(repr(node) for node in block)


def _solve_window_per_label(
    h_op,
    basis,
    plan,
    kept,
    importance,
    orbits,
    cap,
    manifold,
    node_of,
    block_of,
    *,
    expand_kwargs,
    dense_cutoff,
    slater_weight_min,
    max_two_sz,
    n_orb,
    verbose,
    group_tol=None,
):
    """The thermal window solved one hard label at a time: ``(merged, solver, es, psis, info)``. Collective.

    Used when the in-window union of every label does not fit the cap. Labels are exact symmetries, so a label's
    window problem is unitarily equivalent to its symmetry partner's whether or not the code detected the
    symmetry: solving each label to convergence under the full cap makes a level whose members live in different
    labels degenerate to the solves' accuracy, where one common cut of the union split it (FeO: 5 of 15 states).

    One label's solver and basis are alive at a time. ``merged`` holds the window's states only (it is the union of
    the label bases' keys, enough to hold and measure them, not to expand); it may exceed ``cap``. The states are
    those within ``manifold.max_energy`` of the lowest over all labels. ``info`` carries ``labels``,
    ``certified`` and the per-label ``sizes``.
    """
    import copy

    comm = basis.comm
    window = float(manifold.max_energy)
    has_sz = any(lab[0] is not None for lab in kept.labels)

    def node(label):
        return (label[0] if has_sz else None, label[1])

    groups = sorted({block_of[node(lab)] for lab, w in zip(kept.labels, kept.in_window) if w}, key=_block_key)
    energies, states, key_sets, reports, converged, certified, sizes, references = [], [], [], [], [], [], [], []
    for lab in groups:
        idx = [i for i, other in enumerate(kept.labels) if node(other) in lab]
        pos = {i: n for n, i in enumerate(idx)}
        sub = KeptClasses(
            [kept[i] for i in idx],
            [kept.at_winning_level[i] for i in idx],
            [kept.in_window[i] for i in idx],
            [
                (
                    None
                    if kept.derived[i] is None or kept.derived[i][0] not in pos
                    else (pos[kept.derived[i][0]], kept.derived[i][1])
                )
                for i in idx
            ],
            [kept.labels[i] for i in idx],
            [kept.importance[i] for i in idx],
        )
        restrictions = list(basis.weighted_restrictions or [])
        two_szs = {n[0] for n in lab}
        if len(two_szs) == 1 and None not in two_szs:
            restrictions = _label_window(basis, plan.pairs, next(iter(two_szs)))
        seed = basis.clone(
            initial_basis=[
                d for d, nd in zip(basis.local_basis, node_of(_bits(basis.local_basis, n_orb))) if nd in lab
            ],
            weighted_restrictions=restrictions,
        )
        orbits_l = None
        if orbits is not None:
            orbits_l = copy.copy(orbits)
            orbits_l.weighted_restrictions = list(restrictions)
            orbits_l.det_filter = lambda bits, lab=lab: np.array([nd in lab for nd in node_of(bits)], dtype=bool)
        union_l, _cut_dict, _cut = _thermal_union(seed, plan, sub, importance, orbits_l, cap, max_two_sz, n_orb, False)
        solver_l = CIPSISolver(union_l)
        solver_l.orbits = orbits_l
        solver_l.group_tol = group_tol
        solver_l.expand(h_op, manifold, **expand_kwargs)
        es_l, psis_l = solver_l.get_eigenvectors(
            h_op,
            manifold,
            dense_cutoff=dense_cutoff,
            slaterWeightMin=slater_weight_min,
            psi_refs=solver_l.psi_refs,
            reuse_last=True,
        )
        es_l = np.real(np.asarray(es_l, dtype=complex))
        if comm is not None:
            es_l = comm.bcast(es_l, root=0)
        eig = getattr(solver_l, "eig", None)
        certified.append(union_l.size < dense_cutoff or eig is None or bool(eig.certified))
        conv = solver_l.convergence_report or {}
        converged.append(conv.get("converged") is not False)
        reports.append(solver_l.truncation_report)
        sizes.append(int(union_l.size))
        references.append(int((solver_l.last_selection or {}).get("n_references", 0)))
        energies.append(es_l)
        states.extend(psis_l.to_states())
        key_sets.append(list(union_l.local_basis))
        if verbose and (comm is None or comm.rank == 0):
            print(f"  thermal window, block {lab}: {union_l.size:,} determinants, {len(es_l)} state(s)", flush=True)
        del solver_l, union_l, psis_l, seed, sub
    es_all = np.concatenate(energies) if energies else np.zeros(0)
    order = np.argsort(es_all, kind="stable")
    keep, _more = _energy_cut_indices(es_all, window, max(_degeneracy_tol(es_all, slater_weight_min), group_tol or 0.0))
    keep = sorted(keep, key=lambda j: (es_all[j], j))
    merged = basis.clone(initial_basis=list(basis.local_basis))
    for keys in key_sets:
        merged.add_states(keys)
    del order, key_sets
    psis = merged.redistribute_block(ManyBodyState.from_states([states[j] for j in keep]))
    psis.prune_rows(0.0)
    es = es_all[keep]
    solver = CIPSISolver(merged)
    solver.orbits = orbits
    solver.psi_refs = psis
    # The references the blocks' last selection rounds scored, together: the merged window is not wider than them.
    solver.last_selection = {"n_references": int(sum(references))}
    hit = [r for r in reports if r is not None]
    solver.truncation_report = (
        dict(
            hit[0],
            cap_bound=any(is_cap_bound(r) for r in hit),
            memory_bound=any(bool(r.get("memory_bound")) for r in hit),
            retained=int(merged.size),
        )
        if hit
        else None
    )
    solver.convergence_report = {
        "converged": all(converged),
        "residual_pt2": max([float((r or {}).get("residual_pt2", 0.0)) for r in reports] + [0.0]),
    }
    # The window the states were cut from: its width and the lowest level *outside* it (None when every solved
    # level was kept), so a reader can tell "nothing else within the window" from "the window was full".
    kept = set(keep)
    above = [float(es_all[j]) for j in range(len(es_all)) if j not in kept]
    solver.convergence_report["window"] = {
        "width": float(window),
        "n_states": len(keep),
        "next_level_above": (min(above) - float(es_all[keep[0]])) if above and keep else None,
    }
    if not all(converged):
        solver.convergence_report["limited_by"] = "cap" if any(is_cap_bound(r) for r in hit) else "de2_min"
    info = {"labels": len(groups), "certified": all(certified), "sizes": sizes}
    return merged, solver, es, psis, info


def solve_certified(
    h_op,
    basis,
    plan,
    manifold,
    *,
    expand_kwargs,
    dense_cutoff,
    slater_weight_min,
    verbose=False,
    only=None,
    orbits=None,
    occupation_orbitals=None,
):
    """A certified sector solve for ``manifold``: ``(es, psis, basis, solver, verdicts)``.

    ``Lowest`` is :func:`certify_sector`'s winner. ``Within`` needs every state of the window, and
    a class expansion holds one ``S_z`` row of one hard label only, so the window is solved on the
    union of the seed and, for every class whose energy lies within the window (plus the largest
    residual PT2, the expansions' error bar), its determinants and their spin partners
    (:func:`_spin_partner_rows`). Classes are added winner first, then by energy, each only if the
    union stays within the basis's cap: the window solve costs more per determinant than a class
    expansion, so the cap has to bound it too. A class at the winning energy that does not fit is cut
    to the room left (its nearest partner rows first), any other stops the union. The union is then
    expanded for the window. Collective.
    """
    comm = basis.comm
    verbose_memory = bool(verbose) and (comm is None or comm.rank == 0)
    ranking_cap = effective_ranking_cap(basis.truncation_threshold)
    if isinstance(manifold, Lowest):
        with measured_phase("class expansions", comm, verbose=verbose_memory) as phase:
            result = certify_sector(
                h_op,
                basis,
                plan,
                expand_kwargs=expand_kwargs,
                dense_cutoff=dense_cutoff,
                slater_weight_min=slater_weight_min,
                verbose=verbose,
                only=only,
                occupation_orbitals=occupation_orbitals,
                ranking_cap=ranking_cap,
            )
            phase.dets = result[2].size
        return result
    with measured_phase("class expansions", comm, verbose=verbose_memory) as phase:
        es, psis, winner_basis, _solver, verdicts, kept = certify_sector(
            h_op,
            basis,
            plan,
            expand_kwargs=expand_kwargs,
            dense_cutoff=dense_cutoff,
            slater_weight_min=slater_weight_min,
            verbose=verbose,
            within=float(manifold.max_energy),
            only=only,
            occupation_orbitals=occupation_orbitals,
            ranking_cap=ranking_cap,
        )
        phase.dets = winner_basis.size
    del es, _solver, winner_basis
    # The largest seed 2 S_z bounds 2 S_max: the partners' rows stop there.
    max_two_sz = max(abs(label[0] or 0) for label in plan.labels.values())
    n_orb = basis.num_spin_orbitals
    cap = basis.truncation_threshold
    # The certification's own error bar: a level split by more than roundoff but less than this is one level.
    group_tol = _energy_error(
        min((v for v in verdicts if not v.stopped_early), key=lambda v: v.energy, default=verdicts[0])
    )
    with measured_phase("thermal window", comm, verbose=verbose_memory) as phase:
        importance = _importance(psis)
        merged, union_cut, cut = _thermal_union(
            basis.clone(initial_basis=list(basis.local_basis)),
            plan,
            kept,
            importance,
            orbits,
            cap,
            max_two_sz,
            n_orb,
            verbose,
        )
        per_label = None
        blocks = None
        if cut and len({lab for lab, w in zip(kept.labels, kept.in_window) if w}) > 1:
            blocks = _label_blocks(plan, kept, basis, orbits, conserved_parities(h_op), n_orb)
            if len(set(blocks[1].values())) < 2:
                blocks = None
        if blocks is not None:
            # The in-window union does not fit and holds several hard labels: solve the window label by label
            # (review S2). Labels are exact symmetries, so each is its own problem under the full cap, and a
            # level whose members live in different labels is degenerate to each solve's convergence.
            del merged
            merged, solver, es, psis, per_label = _solve_window_per_label(
                h_op,
                basis,
                plan,
                kept,
                importance,
                orbits,
                cap,
                manifold,
                *blocks,
                expand_kwargs=expand_kwargs,
                dense_cutoff=dense_cutoff,
                slater_weight_min=slater_weight_min,
                max_two_sz=max_two_sz,
                n_orb=n_orb,
                verbose=verbose,
                group_tol=group_tol,
            )
            union_cut = dict(union_cut, per_label=per_label["labels"])
            comm_root = basis.comm is None or basis.comm.rank == 0
            emit_memory_warning(
                f"WARNING thermal window: the in-window union of the sector's symmetry classes does not fit the "
                f"determinant cap of {int(cap):,}, so the window was solved separately in "
                f"{per_label['labels']} symmetry block(s) "
                f"(largest {max(per_label['sizes']):,} determinants), each under the full cap, and the states "
                "merged; a level whose members sit in different blocks stays degenerate to each solve's "
                "convergence. Raise truncation_threshold above the union size (TOML [many_body_basis] "
                "truncation_threshold, RSPt `truncation_threshold N`, CLI --truncation_threshold N) to solve it "
                "in one basis.",
                root=comm_root,
                kind="union-split",
            )
        kept.clear()
        del importance
        if per_label is None:
            del psis
            solver = CIPSISolver(merged)
            solver.orbits = orbits
            solver.group_tol = group_tol
            solver.expand(h_op, manifold, **expand_kwargs)
            es, psis = solver.get_eigenvectors(
                h_op,
                manifold,
                dense_cutoff=dense_cutoff,
                slaterWeightMin=slater_weight_min,
                psi_refs=solver.psi_refs,
                reuse_last=True,
            )
        phase.dets = merged.size
    found, expected = manifold_deficit(plan, verdicts, es, psis, merged)
    solver.manifold_check = {"found": found, "expected": expected}
    es_bcast = es if basis.comm is None else basis.comm.bcast(es, root=0)
    roundoff = roundoff_tol(es_bcast, dense=merged.size < dense_cutoff, slater_weight_min=slater_weight_min)
    eig = getattr(solver, "eig", None)
    certified = merged.size < dense_cutoff or eig is None or bool(eig.certified)
    if per_label is not None:
        certified = per_label["certified"]
    split = multiplet_split(plan, verdicts, es, psis, merged, roundoff, certified=certified)
    solver.manifold_check["multiplet_split"] = split
    solver.manifold_check["certified"] = bool(certified)
    if split is not None and (basis.comm is None or basis.comm.rank == 0):
        spins = ", ".join(f"S = {ts / 2:g}" for ts in split["two_s"])
        cause = (
            "the selection or a truncation broke it (a bug: orbit closure is on)"
            if orbits is not None
            else "the selection or a truncation broke it (orbit-closed selection is off here: "
            "groundstate._ORBIT_CLOSURE is False, or no symmetry orbits were built)"
        )
        print(
            f"ERROR: the thermal manifold splits the {spins} multiplet(s) of its lowest level by "
            f"{split['spread']:.2e}, above the solve's own accuracy {split['roundoff']:.1e}. The Hamiltonian has "
            f"total-spin symmetry, so its eigenstates cannot split this way: {cause}. Thermal averages and the "
            "Green's functions built on these states can carry a spurious spin polarization.",
            flush=True,
        )
    es_all = es_bcast
    in_window, in_window_expected = window_deficit(verdicts, es_all, float(manifold.max_energy), expected)
    solver.manifold_check.update(
        window_found=in_window, window_expected=in_window_expected, window_complete=in_window >= in_window_expected
    )
    _record_union_cut(solver, union_cut if cut else None, cap, merged.size)
    if found < expected and (basis.comm is None or basis.comm.rank == 0):
        print(
            f"WARNING: the thermal manifold holds {found} states of its lowest level, but certification "
            f"found that level in rows that make at least {expected}: thermal averages over it are "
            f"misweighted. {_raise_cap_advice(verdicts, widen='widen the thermal window')}",
            flush=True,
        )
    if in_window < in_window_expected and (basis.comm is None or basis.comm.rank == 0):
        print(
            f"WARNING: the thermal window holds {in_window} states, but certification found classes in it that "
            f"make at least {in_window_expected}: states of the higher levels are missing. "
            f"{_raise_cap_advice(verdicts, widen='narrow the thermal window')}",
            flush=True,
        )
    return es, psis, merged, solver, verdicts


#: Added to the measured spread of the classes' ``<N_imp>`` when bounding how far two class energies
#: can move relative to each other under a shift (:meth:`SectorPlanStore.jobs_at`). The spread is a
#: property of the classes' states at the certified shift, not a bound at another one; this is the
#: margin for that, in electrons.
N_IMP_SPREAD_SAFETY = 0.5


@dataclass(frozen=True)
class SectorRecord:
    """One sector's certification: the plan, the shift it ran at and what every class found."""

    plan: CertificationPlan
    mu: float
    verdicts: tuple


class SectorPlanStore:
    """Certify each sector of a search once; at other shifts expand only the classes that can win.

    A search over a shift ``mu`` (``H(mu) = H - mu N_imp``) solves the same sectors again and again.
    The certification plan of a sector does not depend on the shift: it is the symmetry labels of ``H``
    and the partition of the seed into classes, and the shift ``-mu P`` is diagonal in the orbitals and
    the same on every orbital of a shell, so it commutes with every label that respects the shells. (It
    is not a constant on the seed: a :class:`~basis_generation.SeedWindow` seed holds determinants with
    different active-shell counts, and a single-shell seed too once the sector's charge is not fixed.
    The slope below is therefore measured per class, never assumed.) A class's lowest energy moves
    with the shift by ``-<N_active>`` of its state (Hellmann-Feynman), so from the certified energies the classes that
    can still be lowest at another shift are known: those whose predicted energy is within
    ``|mu - mu_c| * (spread of <N_imp> + N_IMP_SPREAD_SAFETY)``, plus the classes' error bar, of the
    leader's. The margin widens with the distance from the certified shift, so a far shift expands
    every class again. Every class is expanded from the seed, never from another shift's basis, so a
    solve does not depend on which shifts came before. Records hold replicated scalars only.
    """

    def __init__(self, active_orbitals=None):
        self.records = {}
        #: The orbitals the shift acts on, whose occupation is the energy's slope in it (``None``: the whole
        #: impurity). Fixed at construction: it belongs to the shift operator, so one store never mixes two.
        self.active_orbitals = None if active_orbitals is None else tuple(int(o) for o in active_orbitals)
        #: Sectors solved by a single expansion because their seed was too large to certify.
        self.uncertified = set()
        #: Sectors whose uncertified thermal ground state was already announced (one WARNING per sector, B10).
        self.announced_uncertified = set()
        #: Sectors certified again because a solve contradicted the record (:meth:`violated`).
        self.recertified = set()
        #: The largest :func:`window_overtake` ratio of any certification.
        self.overtake = None

    def plan(self, key):
        """The certification plan of sector ``key``, or ``None`` if it was never certified."""
        record = self.records.get(key)
        return None if record is None else record.plan

    def record(self, key, plan, mu, verdicts, replace=False):
        """Keep the first, full certification of sector ``key``; later partial ones add nothing.

        ``replace`` is for a recertification, which supersedes the record its tangent violation invalidated.
        """
        if replace or key not in self.records:
            self.records[key] = SectorRecord(plan, float(mu), tuple(verdicts))

    def clear(self):
        self.records.clear()
        self.uncertified.clear()
        self.announced_uncertified.clear()
        self.recertified.clear()
        self.overtake = None

    def summary(self):
        """Counts for a result record: sectors certified, solved uncertified, and re-certified.

        Keyed by sector, not by setting: a sector certified at two caps of a ladder counts once.
        """

        def sectors(keys):
            return len({key[0] if isinstance(key, tuple) else key for key in keys})

        return {
            "certified_sectors": sectors(self.records),
            "uncertified_sectors": sectors(self.uncertified),
            "recertified_sectors": sectors(self.recertified),
            "window_overtake": self.overtake,
        }

    @staticmethod
    def _error(verdict):
        return _energy_error(verdict)

    def jobs_at(self, key, mu, window=0.0):
        """Indices of the jobs worth expanding at ``mu``, or ``None`` (no usable record: expand all).

        ``window`` widens the margin by a thermal window, for a ``Within`` solve. A class that was
        stopped early (it leaked into another class's state) has an energy that bounds nothing: it
        is expanded whenever the class it leaked into is.
        """
        record = self.records.get(key)
        if record is None:
            return None
        verdicts = record.verdicts
        if not verdicts or not all(np.isfinite(v.energy) and np.isfinite(v.n_imp) for v in verdicts):
            return None
        shift = float(mu) - record.mu
        finished = [v for v in verdicts if not v.stopped_early]
        if not finished:
            return None
        predicted = {id(v): v.energy - shift * v.n_imp for v in finished}
        lead = min(finished, key=lambda v: predicted[id(v)])
        spread = max(v.n_imp for v in verdicts) - min(v.n_imp for v in verdicts)
        reach = abs(shift) * (spread + N_IMP_SPREAD_SAFETY) + float(window)
        keep = [
            v for v in finished if predicted[id(v)] - predicted[id(lead)] <= reach + self._error(v) + self._error(lead)
        ]
        jobs = {record.plan.jobs.index(v.job) for v in keep}
        for v in verdicts:
            if v.stopped_early:
                host = min(finished, key=lambda f: abs(f.energy - v.energy))
                if any(host is k for k in keep):
                    jobs.add(record.plan.jobs.index(v.job))
        return jobs

    def violated(self, key, mu, verdicts):
        """Whether a solve at ``mu`` contradicts the record: some class came out above its tangent.

        The lowest energy of ``H - mu N_imp`` on a fixed class is concave in ``mu``, so its tangent at
        the certified shift, ``E_c - (mu - mu_c) <N_imp>_c``, bounds it from above. A solve landing
        higher than that (beyond both error bars) has missed what the certified one found -- a trap
        or a truncation -- and the classes this shift did not expand can no longer be trusted.
        """
        record = self.records.get(key)
        if record is None:
            return False
        shift = float(mu) - record.mu
        certified = {v.job: v for v in record.verdicts if not v.stopped_early}
        for v in verdicts:
            ref = certified.get(v.job)
            if ref is None or v.stopped_early:
                continue
            if v.energy > ref.energy - shift * ref.n_imp + self._error(ref) + self._error(v):
                return True
        return False
