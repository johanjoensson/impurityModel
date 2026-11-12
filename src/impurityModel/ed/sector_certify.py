"""Certification of a fixed-N sector: the plan of symmetry classes, their expansions and verdicts.

Split from :mod:`impurityModel.ed.sector_solver` (see its docstring for the method); this module
holds the plan, ``certify_sector`` and the error bars of a verdict. It does not import ``sector_solver``.
"""

import time
from dataclasses import dataclass, field, replace

import numpy as np

from impurityModel.ed import config, label_symmetries, seed_sectors, solver_trace
from impurityModel.ed.cipsi_solver import (
    CIPSISolver,
)
from impurityModel.ed.density_matrices import occupations as state_occupations
from impurityModel.ed.eigensolvers import Lowest, Within
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant
from impurityModel.ed.mpi_root import on_root
from impurityModel.ed.sector_truncation import _aligned_weights
from impurityModel.ed.spin_pairs import _spin_symmetric, bath_spin_pairs, impurity_spin_pairs
from impurityModel.ed.spin_pairs import spin_layout as resolve_spin_layout
from impurityModel.ed.symmetries import (
    PARITY_TERM_TOL,
    conserved_parities,
    conserves_sz,
    parity_labels,
    sz_weighted_restriction,
)
from impurityModel.ed.truncation_flags import is_cap_bound

#: A class expansion whose state has left its seed span (seed weight below
#: :data:`seed_sectors.DRIFT_TOL`) stops once its energy has settled (:data:`LEAK_PLATEAU_TOL`) this
#: close above an energy another class already converged to: it has leaked into that class's state
#: and can only re-derive it. On the NiO 10-bath d6 sector six of twelve class expansions stopped
#: this way (459 s to 113 s, the same ground state).
LEAK_ENERGY_TOL = 1e-3


#: ...and only once its energy moved by less than this over the last cycle. Without it a state still
#: descending towards a *new*, lower level would be stopped as it passed through the band above an
#: existing one (review: the band is as wide as the sector spacings that matter).
LEAK_PLATEAU_TOL = 1e-5


#: Pass 1 of a certification stops a class once its last :data:`RANK_PATIENCE` per-cycle energy
#: drops are all below :data:`RANK_DROP_TOL` and shrinking: its energy is then known well enough
#: to rank it. The rest of its convergence -- most of its cycles, on its largest bases -- is spent
#: only on the classes that can still win (pass 2). Replayed on the recorded NiO 10-bath d5 and d6
#: trajectories this keeps every sector's winner and cuts the expansion work to 0.45 and 0.34. A
#: single small drop is not enough: d6's winner dropped 0.16, 0.021, then 0.042 and 0.064 as it
#: crossed into the ground state.
RANK_PATIENCE = 3


RANK_DROP_TOL = 1e-3


#: The remaining drop of a ranked class is bounded by a geometric tail of its largest recent drop,
#: at the largest recent drop ratio but at least this one (a slower decay than any measured).
RANK_RATIO_CAP = 0.9


#: Added to that tail: what a ranked class's energy may still fall by, beyond the extrapolation.
RANK_FLOOR = 1e-3


#: A class pass 1 stops at the ranking cap has not settled: what it may still fall by is this multiple of its
#: PT2 estimate (residual plus the mass the cap left out), at least :data:`RANK_FLOOR`, at least the geometric
#: tail of its last drops, or -- when its trajectory is not monotone or too short to extrapolate -- as much as
#: it has already fallen. A calibrated margin, not a bound.
RANK_KAPPA = 2.0


effective_ranking_cap = config.effective_ranking_cap


def _cap_tail(energies, selection):
    """How far a class stopped at the ranking cap may still fall (:data:`RANK_KAPPA`; see there)."""
    selection = selection or {}
    pt2 = abs(float(selection.get("residual_pt2") or 0.0)) + abs(float(selection.get("discarded_de2_mass") or 0.0))
    tail = max(RANK_KAPPA * pt2, RANK_FLOOR)
    drops = [energies[i] - energies[i + 1] for i in range(len(energies) - 1)]
    recent = drops[len(drops) - RANK_PATIENCE :]
    if len(recent) == RANK_PATIENCE and min(recent) >= -1e-9:
        ratios = [recent[i] / recent[i - 1] for i in range(1, len(recent)) if recent[i - 1] > 1e-12]
        if not any(r > 1.0 for r in ratios):
            rho = min(max([RANK_RATIO_CAP] + ratios), 1.0 - 1e-6)
            return max(tail, RANK_KAPPA * max(recent) * rho / (1.0 - rho))
    # Not monotone, or too short to extrapolate: it may fall as far again as it already has.
    return max(tail, float(energies[0] - energies[len(energies) - 1]) if energies else 0.0)


def _ranking_remaining(energies):
    """How far a class's energy may still fall, once its trajectory has settled; ``None`` before.

    ``energies`` is the class's variational energy after each cycle. Settled: the last
    :data:`RANK_PATIENCE` drops are non-negative, below :data:`RANK_DROP_TOL`, and none larger than
    the one before it. The bound is the geometric tail of the largest of them at the largest ratio
    between them, capped below by :data:`RANK_RATIO_CAP`, plus :data:`RANK_FLOOR`.
    """
    if len(energies) < RANK_PATIENCE + 1:
        return None
    tail = energies[len(energies) - RANK_PATIENCE - 1 :]
    drops = [tail[i] - tail[i + 1] for i in range(RANK_PATIENCE)]
    if min(drops) < -1e-9 or max(drops) >= RANK_DROP_TOL:
        return None
    ratios = [drops[i] / drops[i - 1] for i in range(1, len(drops)) if drops[i - 1] > 1e-12]
    if any(r > 1.0 for r in ratios):
        return None
    rho = max([RANK_RATIO_CAP] + ratios)
    rho = min(rho, 1.0 - 1e-6)
    return max(drops) * rho / (1.0 - rho) + RANK_FLOOR


@dataclass(frozen=True)
class CertificationJob:
    """One class expansion: the seed states of ``seed_class`` inside the hard label ``(two_sz, parity)``."""

    two_sz: int | None  # None when S_z is not conserved
    parity: tuple
    seed_class: int | None  # index into the label's class plan; None: the label's whole seed
    n_seed: int
    #: Index of the job this one is the symmetry image of (:mod:`label_symmetries`): it is not
    #: expanded, and its verdict is that job's. ``None`` for a job that is expanded.
    image_of: int | None = None


@dataclass(frozen=True)
class ClassVerdict:
    """What one class expansion found. Replicated (every field is broadcast)."""

    job: CertificationJob
    energy: float
    n_dets: int
    residual_pt2: float
    cap_bound: bool
    memory_bound: bool
    #: ``<N_imp>`` of the class's lowest state: ``-dE/dmu`` for a shift ``-mu N_imp``.
    n_imp: float = float("nan")
    #: Its state left its class -- the leak test stopped it, or its final state kept less than
    #: :data:`seed_sectors.DRIFT_TOL` of its seed weight in the class. Its energy is then another
    #: class's state's, reached first by whichever class happened to run first, and bounds nothing
    #: at another shift.
    stopped_early: bool = False
    #: The expansion's ``truncation_report`` (replicated scalars), when a cap or the memory guard
    #: limited it: a class can *lose* because it was held high, which the winner's own report misses.
    truncation_report: dict | None = None
    #: Pass 1 stopped this class as settled (:func:`_ranking_remaining`): ``energy`` is an upper bound,
    #: ``lower`` what it may still fall to, and ``residual_pt2`` includes the difference.
    ranked: bool = False
    lower: float = float("nan")
    #: The class basis's energies inside the thermal window above its lowest (:func:`_class_window_levels`), when
    #: it was asked for them (pass 2 only: a class that converges unranked in pass 1 keeps it empty and counts as one
    #: state, a weaker but sound bound); empty otherwise. They are the class's penalised Ritz values, measured from
    #: its own penalised lowest, not from ``energy``.
    levels: tuple = ()
    #: Pass 1 stopped this class when its basis reached this size (:func:`effective_ranking_cap`), and pass 2 did
    #: not continue it: ``energy`` is an upper bound at that size, ``lower`` what it may still fall to. ``None``
    #: for a class that converged, settled, or was continued at the full cap.
    ranking_cap: int | None = None


@dataclass(frozen=True)
class CertificationPlan:
    """The jobs of one seed, replicated on every rank."""

    jobs: list
    labels: dict  # seed determinant (chunk tuple) -> (two_sz, parity)
    pairs: list
    su2: bool
    class_plans: dict  # (two_sz, parity) -> seed_sectors.SeedClassPlan or None
    #: Mirror job index -> the :class:`label_symmetries.SignedPermutation` taking the problem of the
    #: job it is the image of onto its own (maps that job's determinants onto this label's).
    images: dict = field(default_factory=dict)
    #: The :class:`~impurityModel.ed.spin_pairs.SpinLayout` the spin labels were judged under.
    layout: object = None


def _bits(dets, n_orb):
    """``(n, n_orb)`` 0/1 occupations of ``dets`` (``SlaterDeterminant``; MSB-first bytes)."""
    if not dets:
        return np.zeros((0, n_orb), dtype=np.uint8)
    raw = np.frombuffer(b"".join(bytes(d.to_bytearray()) for d in dets), dtype=np.uint8).reshape(len(dets), -1)
    return np.unpackbits(raw, axis=1)[:, :n_orb]


def _respects_sz(sp, sign):
    """Whether the permutation maps the ``2 S_z`` sign vector onto itself (or, for a spin flip, its negative).

    The label of an image determinant is the mirrored label only if the permutation carries ``2 S_z``
    along; a generator built from other pairs than the ones that label would map the wrong labels.
    """
    image = np.zeros_like(sign)
    image[list(sp.perm)] = sign
    return bool(np.array_equal(image, -sign if sp.flips_spin else sign))


def _label_window(basis, pairs, two_sz):
    window = list(basis.weighted_restrictions or [])
    if two_sz is not None:
        window.append(sz_weighted_restriction(pairs, two_sz))
    return window


def _label_keys(labels, label, basis):
    """This rank's seed determinants with the hard label ``label``."""
    return [d for d in basis.local_basis if labels.get(tuple(int(c) for c in d)) == label]


def certification_plan(h_op, basis, *, max_seed=1000, spin_layout=None):
    """The class expansions a sector solve on the seed ``basis`` needs (collective), or ``None``.

    Rank 0 gathers the seed, detects the hard labels and decides which are needed; then every needed
    label's seed is partitioned into its soft classes. ``h_op``'s
    restrictions are left set to ``basis``'s on every rank. ``None`` when the seed is empty or larger
    than ``max_seed`` (the plan is built from dense seed matrices): the caller then runs a single,
    uncertified expansion.

    ``spin_layout`` is the :class:`~impurityModel.ed.spin_pairs.SpinLayout` the caller resolved from the
    solver basis (``SolverBasis.spin_layout``). ``None``: resolved here, on rank 0, from the index
    convention and the full ``H``, which is right for a dn-then-up layout and misses the pairing of a
    per-spin grouped impurity.
    """
    comm = basis.comm if basis.is_distributed else None
    h_op.set_restrictions(basis.restrictions)
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    if len(basis) == 0 or len(basis) > max_seed:
        # Dense seed matrices are what the plan is built from; above max_seed the sector is solved
        # by a single expansion and reported uncertified (``None``), never refused.
        return None
    seed = seed_sectors.gather_seed(basis)

    def _plan_head():
        n_orb = max(basis.num_spin_orbitals, max((o for term in h_op.keys() for o, _ in term), default=-1) + 1)
        index_imp = impurity_spin_pairs(basis.impurity_orbitals)
        index_bath = bath_spin_pairs(basis.bath_states)
        layout = spin_layout
        if layout is None:
            layout = resolve_spin_layout(h_op, basis.impurity_orbitals, basis.bath_states, None, n_orb)
        # The layout was judged on the solver basis' H; `h_op` here may differ (a chemical-potential
        # shift, say), so the two spin gates are re-run on it -- cheap, exact, and on rank 0 only.
        sz_ok = layout.labelled and conserves_sz(h_op, layout.pairs, PARITY_TERM_TOL)
        su2 = sz_ok and layout.su2 and _spin_symmetric(h_op, layout.pairs)
        # Only pairs that make H conserve S_z may label; at grade "none" the index pairs remain as the
        # spatial-orbital candidates of the label symmetries, which is_symmetry then verifies.
        pairs = list(layout.pairs) if sz_ok else index_imp + index_bath
        parities = conserved_parities(h_op)
        bits = _bits([SlaterDeterminant(d) for d in seed], n_orb).astype(np.int64)
        par = parity_labels(bits, parities) if parities else np.zeros((len(seed), 0), dtype=np.int64)
        if sz_ok:
            sign = np.zeros(n_orb, dtype=np.int64)
            for dn, up in pairs:
                sign[up], sign[dn] = 1, -1
            two_sz = bits @ sign
        labels = {
            det: ((int(two_sz[i]) if sz_ok else None), tuple(int(x) for x in par[i])) for i, det in enumerate(seed)
        }
        needed = sorted(set(labels.values()), key=lambda lab: ((lab[0] or 0), lab[1]))
        if su2:
            two_min = min(abs(lab[0]) for lab in needed)
            needed = [lab for lab in needed if lab[0] == two_min]
        orbit_of = {}
        if label_symmetries.CERTIFY_LABEL_SYMMETRY and len(needed) > 1:
            generators = label_symmetries.symmetry_generators(
                h_op,
                list(layout.imp_pairs) if sz_ok else index_imp,
                list(layout.bath_pairs) if sz_ok else index_bath,
                n_orb,
                restrictions=basis.restrictions,
                weighted_restrictions=basis.weighted_restrictions,
            )
            if sz_ok:
                generators = [sp for sp in generators if _respects_sz(sp, sign)]
            if generators:
                orbit_of = label_symmetries.label_orbits(needed, generators, parities)
        return (labels, needed, pairs, su2, orbit_of, layout)

    # A rank-0 failure is raised on every rank before the broadcast of the result.
    head = on_root(comm, _plan_head)
    if comm is not None:
        head = comm.bcast(head, root=0)
    # Decided on rank 0 and broadcast: which labels are expanded must be identical on every rank.
    labels, needed, pairs, su2, orbit_of, layout = head
    mirrors = [lab for lab in needed if lab in orbit_of and orbit_of[lab][0] != lab]
    class_plans, jobs = {}, []
    for lab in needed:
        if lab in mirrors:
            continue
        # Collective: every rank builds the label's seed and its class plan, whatever its share.
        sub = basis.clone(
            initial_basis=_label_keys(labels, lab, basis), weighted_restrictions=_label_window(basis, pairs, lab[0])
        )
        plan = seed_sectors.seed_class_plan(h_op, sub, max_seed=max_seed)
        class_plans[lab] = plan
        n_label = sum(1 for v in labels.values() if v == lab)
        # Every class inside the window, "partner" classes too: a partner is judged on seed spectra,
        # which do not see the bath energies, so skipping it can skip the ground state.
        inside = [] if plan is None else [k for k, cls in enumerate(plan.classes) if cls.status != "window"]
        if len(inside) <= 1:
            jobs.append(CertificationJob(lab[0], lab[1], None, n_label))
        else:
            for k in inside:
                span = plan.classes[k].span
                n = int(np.count_nonzero(np.max(np.abs(span), axis=1) > seed_sectors.COUPLING_TOL))
                jobs.append(CertificationJob(lab[0], lab[1], k, n))
    # A label that a verified symmetry maps from an expanded one is a unitary copy of its problem:
    # one mirror job per job of its representative, never expanded (see certify_sector).
    images = {}
    for lab in mirrors:
        rep, to_lab = orbit_of[lab]
        for k, job in [(k, job) for k, job in enumerate(jobs) if (job.two_sz, job.parity) == rep]:
            images[len(jobs)] = to_lab
            jobs.append(CertificationJob(lab[0], lab[1], job.seed_class, job.n_seed, image_of=k))
    h_op.set_restrictions(basis.restrictions)
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    return CertificationPlan(jobs, labels, pairs, su2, class_plans, images, layout)


def describe_plan(plan):
    """One line for a log: the hard labels, the jobs, and the classes the seed window left out.

    The window is the certifier's one heuristic (:func:`seed_sectors._window`): a class outside it
    is never expanded. Its edge -- the lowest seed energy left out, above the label's lowest -- is
    what a reader needs to judge whether it was wide enough.
    """
    outside, edge, windows = 0, None, []
    for class_plan in plan.class_plans.values():
        if class_plan is None:
            continue
        windows.append(class_plan.window)
        e_min = class_plan.classes[0].energy
        for cls in class_plan.classes:
            if cls.status == "window":
                outside += 1
                gap = cls.energy - e_min
                edge = gap if edge is None else min(edge, gap)
    spin = "SU(2)" if plan.su2 else ("S_z" if any(job.two_sz is not None for job in plan.jobs) else "none")
    if plan.layout is not None and plan.layout.labelled and plan.layout.pairs_source():
        spin += ", " + plan.layout.pairs_source()
    expanded = [j for j in plan.jobs if j.image_of is None]
    text = (
        f"certifying: {len({(j.two_sz, j.parity) for j in plan.jobs})} hard labels (spin: {spin}), "
        f"{len(expanded)} class expansions"
    )
    mirrored = len(plan.jobs) - len(expanded)
    if mirrored:
        n_labels = len({(j.two_sz, j.parity) for j in plan.jobs if j.image_of is not None})
        text += f" ({mirrored} more mirrored from them by symmetry, {n_labels} label(s))"
    if windows:
        text += f", window {max(windows):.4g}"
    if outside:
        text += f", {outside} classes outside it (nearest {edge:.4g} above its label's lowest)"
    return text


#: Fraction of the seed window, in units of the classes' hybridization spread, that a measured
#: overtake may reach before the window is flagged as too narrow (:func:`window_overtake`).
WINDOW_OVERTAKE_WARN = 2.0 / 3.0


def window_overtake(plan, verdicts):
    """How close the certification came to needing a wider seed window: a ratio, or ``None``.

    A class enters the expansion only if its seed energy lies within the window above its label's
    lowest class, ``window = factor * sigma_max``. Hybridization lowers a class by an amount set by
    its own spread ``sigma``; one that ends up *below* a class that started lower has overtaken it.
    For every expanded class this measures its overtake of the label's seed-lowest class,
    ``((s_c - s_0) - (E_c - E_0)) / sigma_c`` (seed energies ``s``, converged ``E``; differences
    within one label, so no shift enters), and returns the largest, divided by the window's own
    ``window / sigma_max``. A class outside the window can only win by overtaking by more than that
    window, so a ratio near 1 says the window had little to spare. On the NiO 10-bath d6 sector the
    winner overtook by 2.05 sigma against a 3 sigma window (0.68). ``None`` when no label had two
    expanded classes to compare.
    """
    by_job = {v.job: v for v in verdicts if not v.stopped_early}
    worst = None
    for label, class_plan in plan.class_plans.items():
        if class_plan is None or len(class_plan.classes) < 2:
            continue
        jobs = {job.seed_class: job for job in plan.jobs if (job.two_sz, job.parity) == label}
        base = by_job.get(jobs.get(0))
        sigma_max = max(cls.sigma for cls in class_plan.classes)
        if base is None or sigma_max <= 0.0 or class_plan.window <= 0.0:
            continue
        reach = class_plan.window / sigma_max
        s_0 = class_plan.classes[0].energy
        for k, job in jobs.items():
            verdict = by_job.get(job)
            cls = class_plan.classes[k] if k is not None else None
            if verdict is None or cls is None or k == 0 or cls.sigma <= 0.0:
                continue
            overtake = ((cls.energy - s_0) - (verdict.energy - base.energy)) / cls.sigma
            ratio = overtake / reach
            worst = ratio if worst is None else max(worst, ratio)
    return worst


def _has_leaked(weight, energy, previous_energy, finished):
    """Whether a class expansion has leaked into a state another class already converged to.

    Its seed weight left the class (``weight`` below :data:`seed_sectors.DRIFT_TOL`), its energy has
    settled (moved less than :data:`LEAK_PLATEAU_TOL` since ``previous_energy``), and it sits within
    :data:`LEAK_ENERGY_TOL` above one of the ``finished`` energies.
    """
    settled = abs(energy - previous_energy) < LEAK_PLATEAU_TOL
    near = any(0.0 <= energy - e < LEAK_ENERGY_TOL for e in finished)
    return weight < seed_sectors.DRIFT_TOL and settled and near


def _job_keys(plan, job, basis):
    """This rank's seed determinants of ``job``: its class's support, or its whole hard label."""
    label = (job.two_sz, job.parity)
    if job.seed_class is None:
        return _label_keys(plan.labels, label, basis)
    class_plan = plan.class_plans[label]
    return seed_sectors.class_seed_keys(class_plan, class_plan.classes[job.seed_class], basis)


def _expand_job(
    h_op,
    basis,
    plan,
    job,
    finished,
    expand_kwargs,
    dense_cutoff,
    slater_weight_min,
    comm,
    rank=False,
    keys=None,
    window=None,
    occupation_orbitals=None,
    ranking_cap=None,
):
    """One class expansion: ``(verdict, es, psis, class_basis, solver)`` (collective).

    ``rank`` stops it once settled (pass 1, :func:`_ranking_remaining`); ``keys`` continues it from
    those determinants instead of its seed (pass 2). With ``ranking_cap`` a ranking class is also stopped
    when its basis reaches that size (nothing is truncated), bounded by :func:`_cap_tail`; the basis
    carries the cap only for this expansion, so a pass 2 clone from the seed basis runs at the full one.

    ``window`` (a thermal window width) makes a finished class report the energies of its basis within it of
    its lowest state (:func:`_class_window_levels`, review C7), which the verdict counts.

    ``occupation_orbitals`` are the orbitals whose occupation is the verdict's ``n_imp`` (the slope of the energy
    in the shift): the shift's own, the active shells'. ``None`` is the whole impurity.
    """
    class_basis = basis.clone(
        initial_basis=_job_keys(plan, job, basis) if keys is None else list(keys.keys()),
        weighted_restrictions=_label_window(basis, plan.pairs, job.two_sz),
    )
    # A seed already at the ranking cap is ranked like any other: the cap would truncate it at once.
    capped = bool(rank) and ranking_cap is not None and class_basis.size < ranking_cap
    if capped:
        class_basis.truncation_threshold = ranking_cap
    solver = CIPSISolver(class_basis)
    solver.stop_at_cap = capped
    at_cap = [False]
    # The penalty holds the expansion in its class, so the faster solver that stays in its start's
    # symmetry sector is the right one there (``eigensolvers._LOWEST_SOLVER = "auto"``). A whole-label job has no
    # penalty: it is a plain expansion inside the hard label, where spin and orbital traps remain
    # and only Lanczos growing the generic start vector escapes them.
    solver.davidson = job.seed_class is not None
    class_plan = cls = None
    if job.seed_class is not None:
        class_plan = plan.class_plans[(job.two_sz, job.parity)]
        cls = class_plan.classes[job.seed_class]
        solver.matrix_hook = seed_sectors.ClassPenalty(class_plan, cls)
    energies = []
    remaining = [None]
    weight = [1.0]

    def hook(sv, e_ref, psi_refs):
        e0 = float(np.min(e_ref))
        stop = False
        if cls is not None:
            weight[0] = seed_sectors.class_weight(class_plan, cls, sv.basis, psi_refs)  # collective
            # Only this label's: H conserves the label, so a state cannot leak into another label's.
            previous = energies[len(energies) - 1] if energies else np.inf
            stop = _has_leaked(weight[0], e0, previous, finished.get((job.two_sz, job.parity), ()))
        energies.append(e0)
        if not stop and rank:
            remaining[0] = _ranking_remaining(energies)
            stop = remaining[0] is not None
            if not stop and capped and sv.basis.size >= ranking_cap:
                # The cap first binds: rank on what the basis gives (the last selection round is a standard
                # PT2 estimate of it), continue only if it can still win.
                remaining[0] = _cap_tail(energies, sv.last_selection)
                at_cap[0] = True
                stop = True
        return stop

    if cls is not None or rank:
        solver.cycle_hook = hook
    with solver_trace.timed("class_expand", label=(job.two_sz, job.parity), seed_class=job.seed_class) as fields:
        solver.expand(h_op, Lowest(), **expand_kwargs)
        fields["n_dets"] = len(class_basis)
        fields["stopped_early"] = bool(solver.stopped_early)
        fields["ranking_cap"] = int(ranking_cap) if capped else None
    solver.cycle_hook = None
    solver.stop_at_cap = False
    levels = ()
    if window is not None and not rank and not solver.stopped_early:
        levels = _class_window_levels(solver, h_op, window, dense_cutoff, slater_weight_min)
    solver.matrix_hook = None
    solver.eig = None
    # The verdict is the unpenalized lowest state of the whole class basis, whatever sector it is in:
    # Lanczos, whose generic start vector reaches every sector, on rank 0 for a small basis.
    solver.davidson = False
    solver.sparse_lowest = True
    with solver_trace.timed("class_eigensolve", seed_class=job.seed_class):
        es, psis = solver.get_eigenvectors(
            h_op, Lowest(), dense_cutoff=dense_cutoff, slaterWeightMin=slater_weight_min, psi_refs=solver.psi_refs
        )
    energy = float(np.min(es)) if len(es) else np.inf
    report = solver.truncation_report or {}
    residual = float((solver.convergence_report or {}).get("residual_pt2") or 0.0)
    if occupation_orbitals is None:
        impurity = [orb for blocks in basis.impurity_orbitals.values() for block in blocks for orb in block]
    else:
        impurity = list(occupation_orbitals)
    # Collective (one Allreduce), entered on every rank: `psis.width` is replicated.
    n_imp = float(state_occupations(class_basis, psis, impurity).sum(axis=1)[0]) if psis.width else float("nan")
    # The hook's last verdict decided why the expansion stopped (rank 0's, broadcast by `expand`).
    ranked = bool(solver.stopped_early) and remaining[0] is not None
    tail = remaining[0] if ranked else 0.0
    # Left its class, whether or not the leak test stopped it: its energy is another class's state's.
    # Only the first class to reach a state leaks into it unstopped -- nothing had finished there yet
    # -- and which one that is depends on the order, so its energy bounds nothing at another shift.
    leaked = (bool(solver.stopped_early) and not ranked) or (cls is not None and weight[0] < seed_sectors.DRIFT_TOL)
    fields = (
        energy,
        residual + tail,
        is_cap_bound(solver.truncation_report),
        bool(report.get("memory_bound")),
        n_imp,
        leaked,
        None if solver.truncation_report is None else dict(solver.truncation_report),
        ranked,
        energy - tail if ranked else float("nan"),
        levels,
        int(ranking_cap) if ranked and at_cap[0] else None,
    )
    if comm is not None:
        fields = comm.bcast(fields, root=0)
    return ClassVerdict(job, fields[0], len(class_basis), *fields[1:]), es, psis, class_basis, solver


def _class_window_levels(solver, h_op, within, dense_cutoff, slater_weight_min):
    """The class basis's energies within ``within`` of its lowest, ascending (collective, replicated result).

    No expansion: these are the Ritz values of the basis the class already converged for its lowest state.
    By interlacing, ``k`` of them inside a window mean at least ``k`` true states of the class inside it, so
    counting them keeps :func:`window_deficit`'s ``expected`` a lower bound. A state the basis has no overlap
    with is not among them -- the limit of what a class expansion can see (review C7, oracle seed 1 m8).
    """
    es, _psis = solver.get_eigenvectors(
        h_op, Within(float(within)), dense_cutoff=dense_cutoff, slaterWeightMin=slater_weight_min,
        psi_refs=solver.psi_refs,
    )  # fmt: skip
    es = np.sort(np.real(np.asarray(es, dtype=complex)))
    return tuple(float(e) for e in es if e - es[0] <= within + 1e-12)


def certify_sector(
    h_op,
    basis,
    plan,
    *,
    expand_kwargs,
    dense_cutoff,
    slater_weight_min,
    verbose=False,
    within=None,
    only=None,
    occupation_orbitals=None,
    ranking_cap=None,
):
    """Expand every job of ``plan`` from ``basis`` (the seed) and return the lowest.

    Returns ``(es, psis, class_basis, class_solver, verdicts)``: the winner's unpenalized lowest
    eigenpair on its own basis, that basis (with the sector's window put back) and solver, and one
    :class:`ClassVerdict` per job expanded. Every branch reads a broadcast value. Collective.

    Two passes. Pass 1 expands every job until its energy has settled enough to rank it
    (:func:`_ranking_remaining`), which bounds what it may still fall to. Pass 2 continues, from
    its own determinants and under its own penalty, every class whose bound still reaches the
    lowest energy found (plus ``within``): the classes that can win, or lie in the thermal window.
    Only their keys are held between the passes, and only the best solve so far is kept alive.

    ``within`` (an energy window) also returns, as a sixth element, the determinant keys of every
    class inside the window above the winner -- the winner's first. ``only`` (job indices)
    restricts the expansions to those jobs (:meth:`SectorPlanStore.jobs_at`).

    ``ranking_cap`` (:func:`effective_ranking_cap`) ends a pass-1 expansion when its basis reaches that size,
    bounded by :func:`_cap_tail`. The contenders for pass 2 among those classes are fixed once, on the pass-1
    energies, and closed under "tied within both error bars" -- the members of a degenerate level are
    continued together or not at all -- because the lowest energy falls during pass 2 and would otherwise
    cut a class on a margin that one member's head start made. Pass 2 runs at the basis's full cap, from the
    pass-1 keys, and its solve replaces the pass-1 one of the same job even when it is not lower.
    """
    comm = basis.comm if basis.is_distributed else None
    rank = comm.rank if comm is not None else 0
    best = None
    frozen = set()
    verdict_of = {}
    finished = {}  # hard label -> energies its classes reached
    keys = {}
    #: class -> float32 weights aligned with ``list(keys[class].keys())`` (the class's own lowest state), kept
    #: only for a window solve: what the thermal union's cut ranks the class's determinants by.
    weights = {}
    reach = 0.0 if within is None else float(within)

    def e_min():
        return min(v.energy for v in verdict_of.values())

    timings = {}

    def run(k, ranking, start=None):
        nonlocal best
        job = plan.jobs[k]
        t_class = time.perf_counter()
        verdict, es, psis, class_basis, solver = _expand_job(
            h_op, basis, plan, job, finished, expand_kwargs, dense_cutoff, slater_weight_min, comm,
            rank=ranking, keys=start, window=within, occupation_orbitals=occupation_orbitals,
            ranking_cap=ranking_cap if ranking else None,
        )  # fmt: skip
        verdict_of[k] = verdict
        timings.setdefault(k, []).append(time.perf_counter() - t_class)
        finished.setdefault((job.two_sz, job.parity), []).append(verdict.energy)
        if verdict.ranked or within is not None:
            keys[k] = class_basis._keys.copy()
            if within is not None:
                weights[k] = _aligned_weights(keys[k], psis)
        # Drop the keys of every class that can no longer reach the lowest energy (plus the window):
        # the lowest energy only falls, so a class dropped here stays out. A class stopped at the ranking cap
        # is decided once, between the passes (:func:`_ranking_contenders`): a class tied with a contender is
        # continued with it even though its own bound does not reach the lowest energy.
        floor = e_min()
        for q in [
            q
            for q in keys
            if verdict_of[q].ranking_cap is None and _bound(verdict_of[q]) - floor > reach + _bound_error(verdict_of[q])
        ]:
            del keys[q]
            weights.pop(q, None)
        if verbose and rank == 0:
            label = f"2Sz={job.two_sz} parity={''.join(map(str, job.parity))}"
            cls = "" if job.seed_class is None else f" class {job.seed_class}"
            note = " (leaked)" if verdict.stopped_early else (" (ranked)" if verdict.ranked else "")
            bound = ", memory-bound" if verdict.memory_bound else (", cap-bound" if verdict.cap_bound else "")
            print(
                f"  [pass {1 if ranking else 2}] {label}{cls}: E = {verdict.energy:.8f} ({verdict.n_dets:,} "
                f"determinants{bound}, PT2 {verdict.residual_pt2:.1e}, {timings[k][-1]:.0f} s){note}",
                flush=True,
            )
        # A pass-2 solve of the job whose pass-1 solve is `best` replaces it: the continued basis is the one at
        # the full cap, and the variational energy cannot have risen beyond noise.
        if best is None or verdict.energy < best[0] or (not ranking and best[1] == k):
            best = (verdict.energy, k, es, psis, class_basis, solver)

    # A mirror job is never expanded: asking for it asks for the job it is the image of.
    wanted = None if only is None else {k if plan.jobs[k].image_of is None else plan.jobs[k].image_of for k in only}
    for k in range(len(plan.jobs)):
        if plan.jobs[k].image_of is None and (wanted is None or k in wanted):
            run(k, ranking=True)
    # Pass 2, most promising first. The lowest energy only falls, so re-test before each.
    candidates = sorted(
        (k for k, v in verdict_of.items() if v.ranked and not v.stopped_early), key=lambda k: verdict_of[k].lower
    )
    frozen.update(_ranking_contenders(verdict_of, candidates, reach))
    for q in [q for q in keys if verdict_of[q].ranking_cap is not None and q not in frozen]:
        del keys[q]
        weights.pop(q, None)

    def reaches(k):
        # A class stopped at the ranking cap is a contender by the pass-1 verdict (frozen); any other by the
        # lowest energy found so far.
        if verdict_of[k].ranking_cap is not None:
            return k in frozen
        return verdict_of[k].lower - e_min() <= reach + _bound_error(verdict_of[k])

    if verbose and rank == 0 and candidates:
        n_in = sum(1 for k in candidates if k in keys and reaches(k))
        print(
            f"  pass 2: {n_in} of {len(candidates)} ranked class(es) reach E_min + {reach:.4g} (+ their error bars); "
            f"{len(candidates) - n_in} cut",
            flush=True,
        )
    for k in candidates:
        if k in keys and reaches(k):
            start = keys.pop(k)
            frozen.discard(k)
            weights.pop(k, None)
            run(k, ranking=False, start=start)
    # A mirror's problem is its representative's under a verified symmetry: same verdict, its own job.
    mirrored = {k: job.image_of for k, job in enumerate(plan.jobs) if job.image_of in verdict_of}
    for k, source in mirrored.items():
        verdict_of[k] = replace(verdict_of[source], job=plan.jobs[k])
    verdicts = [verdict_of[k] for k in sorted(verdict_of)]
    if verbose and rank == 0 and verdicts:
        floor_e = min(v.energy for v in verdicts)
        print("  class table: label, class, E, E-E_min [mRy], dets, PT2, bound, passes, seconds", flush=True)
        for k in sorted(verdict_of):
            v, job = verdict_of[k], plan.jobs[k]
            tag = f"2Sz={job.two_sz} parity={''.join(map(str, job.parity))} class {job.seed_class}"
            bound = "memory" if v.memory_bound else ("cap" if v.cap_bound else "-")
            spent = timings.get(job.image_of if k not in timings else k, [])
            print(
                f"    {tag}: {v.energy:.8f}  {1e3 * (v.energy - floor_e):9.3f}  {v.n_dets:>10,}  {v.residual_pt2:.1e}  "
                f"{bound}  {len(spent)}  {sum(spent):.0f}",
                flush=True,
            )
    kept = None
    if within is not None:
        floor = e_min()
        inside = _classes_in_window(verdict_of, keys, floor, within)
        winner = best[1]
        # The mirror labels' rows of the window's states are the images of their sources' rows: without
        # them the window solve would hold the representative's half of a degenerate level only. Each
        # image follows its source, where the mirror's own (degenerate) class would have sorted, so the
        # union's cap drops the same classes with and without the symmetry.
        n_orb = basis.num_spin_orbitals
        order = _thermal_union_order(winner, inside, mirrored)
        kept = [
            keys[source] if mirror is None else _image_keys(keys[source], plan.images[mirror], n_orb, basis.n_bytes)
            for source, mirror in order
        ]
        win_v = verdict_of[winner]
        position = {source: i for i, (source, mirror) in enumerate(order) if mirror is None}

        def image_of_source(sp):
            return lambda share: _image_keys(share, sp, n_orb, basis.n_bytes)

        kept = KeptClasses(
            kept,
            [
                abs(verdict_of[source].energy - win_v.energy)
                <= _energy_error(verdict_of[source]) + _energy_error(win_v)
                for source, _mirror in order
            ],
            # In the window by the class's *own* error bar, not through the margin that the largest residual
            # of any class inflates: these are never dropped, only cut to a common share.
            [verdict_of[source].energy - floor <= within + _energy_error(verdict_of[source]) for source, _m in order],
            [
                None if mirror is None else (position[source], image_of_source(plan.images[mirror]))
                for source, mirror in order
            ],
            [
                (
                    plan.jobs[mirror if mirror is not None else source].two_sz,
                    plan.jobs[mirror if mirror is not None else source].parity,
                )
                for source, mirror in order
            ],
            [weights.get(source) if mirror is None else None for source, mirror in order],
        )
        keys.clear()
        weights.clear()
    _energy, _k, es, psis, class_basis, solver = best
    # The winner carries the full cap whatever pass solved it: later users of its basis (the thermal window,
    # the sector's state) grow it under the cap the caller asked for.
    class_basis.truncation_threshold = basis.truncation_threshold
    class_basis.weighted_restrictions = basis.weighted_restrictions
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    if within is not None:
        return es, psis, class_basis, solver, verdicts, kept
    return es, psis, class_basis, solver, verdicts


def _ranking_contenders(verdict_of, candidates, reach):
    """The jobs among ``candidates`` stopped at the ranking cap that pass 2 must continue, fixed on pass-1 energies.

    A class is a contender when its lower bound reaches the lowest pass-1 energy (plus ``reach`` and its error
    bar); any capped class tied with a contender within both error bars joins it, to a fixed point.
    """
    capped = [k for k in candidates if verdict_of[k].ranking_cap is not None]
    if not capped:
        return set()
    floor = min(v.energy for v in verdict_of.values())
    contenders = {k for k in capped if verdict_of[k].lower - floor <= reach + _bound_error(verdict_of[k])}
    # A class that was not stopped at the ranking cap is final: it anchors the tie closure when it reaches the
    # floor, so a capped degenerate partner of the (converged) lowest class is continued with it.
    anchors = contenders | {
        k for k, v in verdict_of.items() if v.ranking_cap is None and v.energy - floor <= reach + _energy_error(v)
    }
    grown = True
    while grown:
        grown = False
        for k in capped:
            if k in contenders:
                continue
            vk = verdict_of[k]
            if any(
                abs(vk.energy - verdict_of[c].energy) <= _energy_error(vk) + _energy_error(verdict_of[c])
                for c in anchors
            ):
                contenders.add(k)
                anchors.add(k)
                grown = True
    return contenders


def _classes_in_window(verdict_of, keys, floor, within):
    """The job indices, lowest energy first, of the classes inside the thermal window that still hold keys.

    A class is inside by its *own* error bar (``within`` plus :func:`_energy_error`): the largest residual of
    any other class does not widen it (review B2), as that admitted classes the window does not contain.
    """
    return sorted(
        (k for k, v in verdict_of.items() if v.energy - floor <= within + _energy_error(v) and k in keys),
        key=lambda k: verdict_of[k].energy,
    )


class KeptClasses(list):
    """The key sets of the classes inside the thermal window, in the order the union adds them.

    ``at_winning_level[i]`` is whether class ``i`` lies at the winning energy (within both error bars):
    the classes whose states the window's lowest level is made of, which a binding cap must truncate
    rather than drop.
    """

    def __init__(self, classes=(), at_winning_level=(), in_window=(), derived=(), labels=(), importance=()):
        super().__init__(classes)
        #: ``None`` or float32 weights aligned with ``list(self[i].keys())``: the class's own lowest state's
        #: ``|amplitude|^2`` per determinant (a mirror has none: it is the image of its source's share).
        self.importance = list(importance) or [None] * len(self)
        #: The hard label ``(2 S_z, parity)`` the rows of class ``i`` live in (a mirror's is its own, not its
        #: source's): what the per-label window solve groups by. ``None`` when not recorded.
        self.labels = list(labels) or [None] * len(self)
        self.at_winning_level = list(at_winning_level)
        #: Whether class ``i`` is inside the window by its own error bar (a class admitted only through the
        #: window's inflated margin is not).
        self.in_window = list(in_window) or [True] * len(self)
        #: ``None`` for a class expanded itself; for a mirror, ``(position of its source, share -> image)``: a
        #: mirror is the image of whatever share of its source is kept, never cut on its own.
        self.derived = list(derived) or [None] * len(self)


def _thermal_union_order(winner, inside, mirrored):
    """``(source job, mirror job or None)`` in the order the thermal union adds them.

    The winner first, then the other in-window classes by energy (``inside`` is sorted), each followed
    at once by the images of its mirrors -- where those degenerate classes would have sorted had they
    been expanded -- so a binding cap drops the same classes with the symmetry as without it.
    """
    order = []
    for source in [winner] + [k for k in inside if k != winner]:
        order.append((source, None))
        order.extend((source, k) for k, s in mirrored.items() if s == source)
    return order


def _image_keys(keys, sp, n_orb, n_bytes):
    """This rank's share of the determinants the signed permutation ``sp`` maps ``keys`` onto.

    A width-0 block, like ``Basis._keys``, of determinants at the basis's canonical width ``n_bytes``
    (``to_bytearray`` is eight times wider, so its length is not the width to rebuild at). Rank-local,
    as :func:`_spin_partner_keys` is: ``add_states`` routes the images to their owners afterwards.
    """
    dets = list(keys.keys())
    if not dets:
        return ManyBodyState.from_keys([])
    packed = np.packbits(sp.map_bits(_bits(dets, n_orb)), axis=1)
    packed = np.pad(packed, ((0, 0), (0, n_bytes - packed.shape[1])))
    return ManyBodyState.from_keys([SlaterDeterminant.from_bytes(bytes(row)) for row in packed])


def _bound(verdict):
    """The lowest a class's energy may still reach: its ranked bound, else its energy."""
    return verdict.lower if verdict.ranked else verdict.energy


def _bound_error(verdict):
    """The error bar of :func:`_bound`: :func:`_energy_error` less the ranking tail ``_bound`` already took.

    A ranked class's ``residual_pt2`` includes that tail (``energy - lower``), so counting it again
    against ``lower`` let pass 2 continue every class within twice its tail of the lowest energy.
    """
    tail = verdict.energy - verdict.lower if verdict.ranked else 0.0
    return _energy_error(verdict) - tail


def _energy_error(verdict):
    """The error bar of a class energy: its residual PT2 plus :data:`EXPANSION_ENERGY_FLOOR`."""
    return verdict.residual_pt2 + EXPANSION_ENERGY_FLOOR


#: Floor on a class energy's error bar, added to the largest residual PT2: an eigensolve that
#: stagnated rather than converged is good to its residual norm, measured up to 5e-6 on the NiO
#: 10-bath d6 classes.
EXPANSION_ENERGY_FLOOR = 1e-5
