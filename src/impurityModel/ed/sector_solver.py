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
from impurityModel.ed import seed_sectors, solver_trace
from impurityModel.ed.basis_transcription import comm_support_filter
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.density_matrices import occupations as state_occupations
from impurityModel.ed.eigensolvers import Lowest, cold_vector
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant, applyOp
from impurityModel.ed.observables import (
    casimir_operator,
    casimir_to_quantum_number,
    make_spin_operators,
    manifold_observable_values,
)
from impurityModel.ed.spin_pairs import bath_spin_pairs, impurity_spin_pairs
from impurityModel.ed.symmetries import (
    PARITY_TERM_TOL,
    conserved_parities,
    conserves_sz,
    parity_labels,
    sz_weighted_restriction,
)

#: Largest ``max |[H, S_+]|`` coefficient, relative to the largest coefficient of ``H``, for which
#: total spin counts as a symmetry of ``H``. The commutator is exact operator algebra, so a symmetry
#: cancels to roundoff and a broken one at the size of the breaking term.
SPIN_SYMMETRY_TOL = 1e-10

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


@dataclass(frozen=True)
class CertificationPlan:
    """The jobs of one seed, replicated on every rank."""

    jobs: list
    labels: dict  # seed determinant (chunk tuple) -> (two_sz, parity)
    pairs: list
    su2: bool
    class_plans: dict  # (two_sz, parity) -> seed_sectors.SeedClassPlan or None


def _bits(dets, n_orb):
    """``(n, n_orb)`` 0/1 occupations of ``dets`` (``SlaterDeterminant``; MSB-first bytes)."""
    if not dets:
        return np.zeros((0, n_orb), dtype=np.uint8)
    raw = np.frombuffer(b"".join(bytes(d.to_bytearray()) for d in dets), dtype=np.uint8).reshape(len(dets), -1)
    return np.unpackbits(raw, axis=1)[:, :n_orb]


def _spin_symmetric(h_op, pairs):
    if not pairs:
        return False
    s_plus, _s_minus, _s_z = make_spin_operators(pairs)
    scale = max((abs(v) for v in h_op.values()), default=0.0)
    comm = h_op.commutator(s_plus)
    return max((abs(v) for v in comm.values()), default=0.0) <= SPIN_SYMMETRY_TOL * max(scale, 1.0)


def _label_window(basis, pairs, two_sz):
    window = list(basis.weighted_restrictions or [])
    if two_sz is not None:
        window.append(sz_weighted_restriction(pairs, two_sz))
    return window


def _label_keys(labels, label, basis):
    """This rank's seed determinants with the hard label ``label``."""
    return [d for d in basis.local_basis if labels.get(tuple(int(c) for c in d)) == label]


def certification_plan(h_op, basis, *, max_seed=1000):
    """The class expansions a sector solve on the seed ``basis`` needs (collective), or ``None``.

    Rank 0 gathers the seed, detects the hard labels and decides which are needed; then every needed
    label's seed is partitioned into its soft classes. ``h_op``'s
    restrictions are left set to ``basis``'s on every rank. ``None`` when the seed is empty or larger
    than ``max_seed`` (the plan is built from dense seed matrices): the caller then runs a single,
    uncertified expansion.
    """
    comm = basis.comm if basis.is_distributed else None
    rank = comm.rank if comm is not None else 0
    h_op.set_restrictions(basis.restrictions)
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    if len(basis) == 0 or len(basis) > max_seed:
        # Dense seed matrices are what the plan is built from; above max_seed the sector is solved
        # by a single expansion and reported uncertified (``None``), never refused.
        return None
    seed = seed_sectors.gather_seed(basis)
    head = None
    if rank == 0:
        n_orb = max(basis.num_spin_orbitals, max((o for term in h_op.keys() for o, _ in term), default=-1) + 1)
        pairs = impurity_spin_pairs(basis.impurity_orbitals) + bath_spin_pairs(basis.bath_states)
        sz_ok = bool(pairs) and conserves_sz(h_op, pairs, PARITY_TERM_TOL)
        su2 = sz_ok and _spin_symmetric(h_op, pairs)
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
        head = (labels, needed, pairs, su2)
    if comm is not None:
        head = comm.bcast(head, root=0)
    labels, needed, pairs, su2 = head
    class_plans, jobs = {}, []
    for lab in needed:
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
    h_op.set_restrictions(basis.restrictions)
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    return CertificationPlan(jobs, labels, pairs, su2, class_plans)


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
    text = (
        f"certifying: {len({(j.two_sz, j.parity) for j in plan.jobs})} hard labels (spin: {spin}), "
        f"{len(plan.jobs)} class expansions"
    )
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
    h_op, basis, plan, job, finished, expand_kwargs, dense_cutoff, slater_weight_min, comm, rank=False, keys=None
):
    """One class expansion: ``(verdict, es, psis, class_basis, solver)`` (collective).

    ``rank`` stops it once settled (pass 1, :func:`_ranking_remaining`); ``keys`` continues it from
    those determinants instead of its seed (pass 2).
    """
    class_basis = basis.clone(
        initial_basis=_job_keys(plan, job, basis) if keys is None else list(keys.keys()),
        weighted_restrictions=_label_window(basis, plan.pairs, job.two_sz),
    )
    solver = CIPSISolver(class_basis)
    # The penalty holds the expansion in its class, so the faster solver that stays in its start's
    # symmetry sector is the right one here (GS_LOWEST_SOLVER=auto).
    solver.davidson = True
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
        return stop

    if cls is not None or rank:
        solver.cycle_hook = hook
    with solver_trace.timed("class_expand", label=(job.two_sz, job.parity), seed_class=job.seed_class) as fields:
        solver.expand(h_op, Lowest(), **expand_kwargs)
        fields["n_dets"] = len(class_basis)
        fields["stopped_early"] = bool(solver.stopped_early)
    solver.matrix_hook = None
    solver.cycle_hook = None
    solver.eig = None
    # The verdict is the unpenalized lowest state of the whole class basis, whatever sector it is in.
    solver.davidson = False
    with solver_trace.timed("class_eigensolve", seed_class=job.seed_class):
        es, psis = solver.get_eigenvectors(
            h_op, Lowest(), dense_cutoff=dense_cutoff, slaterWeightMin=slater_weight_min, psi_refs=solver.psi_refs
        )
    energy = float(np.min(es)) if len(es) else np.inf
    report = solver.truncation_report or {}
    residual = float((solver.convergence_report or {}).get("residual_pt2") or 0.0)
    impurity = [orb for blocks in basis.impurity_orbitals.values() for block in blocks for orb in block]
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
        bool(solver.truncation_report is not None),
        bool(report.get("memory_bound")),
        n_imp,
        leaked,
        None if solver.truncation_report is None else dict(solver.truncation_report),
        ranked,
        energy - tail if ranked else float("nan"),
    )
    if comm is not None:
        fields = comm.bcast(fields, root=0)
    return ClassVerdict(job, fields[0], len(class_basis), *fields[1:]), es, psis, class_basis, solver


def certify_sector(
    h_op, basis, plan, *, expand_kwargs, dense_cutoff, slater_weight_min, verbose=False, within=None, only=None
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
    """
    comm = basis.comm if basis.is_distributed else None
    rank = comm.rank if comm is not None else 0
    best = None
    verdict_of = {}
    finished = {}  # hard label -> energies its classes reached
    keys = {}
    reach = 0.0 if within is None else float(within)

    def e_min():
        return min(v.energy for v in verdict_of.values())

    def run(k, ranking, start=None):
        nonlocal best
        job = plan.jobs[k]
        verdict, es, psis, class_basis, solver = _expand_job(
            h_op, basis, plan, job, finished, expand_kwargs, dense_cutoff, slater_weight_min, comm,
            rank=ranking, keys=start,
        )  # fmt: skip
        verdict_of[k] = verdict
        finished.setdefault((job.two_sz, job.parity), []).append(verdict.energy)
        if verdict.ranked or within is not None:
            keys[k] = class_basis._keys.copy()
        # Drop the keys of every class that can no longer reach the lowest energy (plus the window):
        # the lowest energy only falls, so a class dropped here stays out.
        floor = e_min()
        for q in [q for q in keys if _bound(verdict_of[q]) - floor > reach + _energy_error(verdict_of[q])]:
            del keys[q]
        if verbose and rank == 0:
            label = f"2Sz={job.two_sz} parity={''.join(map(str, job.parity))}"
            cls = "" if job.seed_class is None else f" class {job.seed_class}"
            note = " (leaked)" if verdict.stopped_early else (" (ranked)" if verdict.ranked else "")
            print(f"  {label}{cls}: E = {verdict.energy:.8f} ({verdict.n_dets:,} determinants){note}", flush=True)
        if best is None or verdict.energy < best[0]:
            best = (verdict.energy, k, es, psis, class_basis, solver)

    for k in range(len(plan.jobs)):
        if only is None or k in only:
            run(k, ranking=True)
    # Pass 2, most promising first. The lowest energy only falls, so re-test before each.
    for k in sorted(
        (k for k, v in verdict_of.items() if v.ranked and not v.stopped_early), key=lambda k: verdict_of[k].lower
    ):
        if k in keys and verdict_of[k].lower - e_min() <= reach + _energy_error(verdict_of[k]):
            start = keys.pop(k)
            run(k, ranking=False, start=start)
    verdicts = [verdict_of[k] for k in sorted(verdict_of)]
    kept = None
    if within is not None:
        floor = e_min()
        margin = within + max(v.residual_pt2 for v in verdicts)
        inside = sorted(
            (k for k, v in verdict_of.items() if v.energy - floor <= margin and k in keys),
            key=lambda k: verdict_of[k].energy,
        )
        winner = best[1]
        kept = [keys[winner]] + [keys[k] for k in inside if k != winner]
        keys.clear()
    _energy, _k, es, psis, class_basis, solver = best
    class_basis.weighted_restrictions = basis.weighted_restrictions
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    if within is not None:
        return es, psis, class_basis, solver, verdicts, kept
    return es, psis, class_basis, solver, verdicts


def _bound(verdict):
    """The lowest a class's energy may still reach: its ranked bound, else its energy."""
    return verdict.lower if verdict.ranked else verdict.energy


def _spin_partner_keys(keys, pairs, max_steps):
    """This rank's share of the spin partners' supports of the states on ``keys``: ``S_+-^k``, ``k >= 1``.

    A class expansion holds one ``S_z`` row of each multiplet it found (its hard label's, the
    smallest ``|S_z|``); the other rows of every multiplet inside the thermal window must be in the
    window solve's basis at the same quality, or they sit at seed quality outside the window and
    the manifold comes back short (measured: 5 of 8 states of a spin-and-orbitally degenerate
    level). The ladders are applied to a state with generic amplitudes on ``keys`` -- every
    determinant of the class -- so no image cancels, and stepped both ways until the image
    vanishes. Each rank steps its own rows; by linearity the union over the ranks covers every
    partner, so this needs no communication -- ``add_states`` routes the keys afterwards.
    """
    s_plus, s_minus, _s_z = make_spin_operators(pairs)
    dets = keys.keys()
    hashes = np.fromiter((det.get_hash() for det in dets), dtype=np.uint64, count=len(dets))
    # The eigensolvers' hash-seeded generic vector: deterministic, and no two images cancel.
    state = ManyBodyState(dict(zip(dets, cold_vector(hashes)[:, 0])))
    found = []
    for ladder in (s_plus, s_minus):
        ladder.set_restrictions(None)
        ladder.set_weighted_restrictions(None)
        image = state
        for _ in range(int(max_steps)):
            image = applyOp(ladder, image, cutoff=0)
            if not len(image):
                break
            found.extend(image.keys())
    return found


def _energy_error(verdict):
    """The error bar of a class energy: its residual PT2 plus :data:`EXPANSION_ENERGY_FLOOR`."""
    return verdict.residual_pt2 + EXPANSION_ENERGY_FLOOR


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
        ladders = make_spin_operators(plan.pairs)
        for op in ladders:
            op.set_restrictions(None)
            op.set_weighted_restrictions(None)
        s2 = casimir_operator(*ladders)
        s2.set_restrictions(None)
        s2.set_weighted_restrictions(None)
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


def solve_certified(
    h_op, basis, plan, manifold, *, expand_kwargs, dense_cutoff, slater_weight_min, verbose=False, only=None
):
    """A certified sector solve for ``manifold``: ``(es, psis, basis, solver, verdicts)``.

    ``Lowest`` is :func:`certify_sector`'s winner. ``Within`` needs every state of the window, and
    a class expansion holds one ``S_z`` row of one hard label only, so the window is solved on the
    union of the seed and, for every class whose energy lies within the window (plus the largest
    residual PT2, the expansions' error bar), its determinants and their spin partners
    (:func:`_spin_partner_keys`). Classes are added winner first, then by energy, and adding stops
    once the union passes the basis's cap -- by at most the last class added. The union is then
    expanded for the window. Collective.
    """
    if isinstance(manifold, Lowest):
        return certify_sector(
            h_op,
            basis,
            plan,
            expand_kwargs=expand_kwargs,
            dense_cutoff=dense_cutoff,
            slater_weight_min=slater_weight_min,
            verbose=verbose,
            only=only,
        )
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
    )
    del es, psis, _solver, winner_basis
    # 2 S_max steps reach every row from any row; the largest seed 2 S_z bounds 2 S_max.
    max_steps = 2 * max(abs(label[0] or 0) for label in plan.labels.values())
    cap = basis.truncation_threshold
    merged = basis.clone(initial_basis=list(basis.local_basis))
    for n_added, keys in enumerate(kept):
        merged.add_states(keys.keys() + (_spin_partner_keys(keys, plan.pairs, max_steps) if plan.su2 else []))
        # `merged.size` is replicated, so every rank stops at the same class.
        if cap is not None and merged.size > cap and n_added + 1 < len(kept):
            if (basis.comm is None or basis.comm.rank == 0) and verbose:
                print(
                    f"  thermal window: {len(kept) - n_added - 1} of {len(kept)} classes left out at the cap "
                    f"({merged.size:,} > {int(cap):,} determinants)",
                    flush=True,
                )
            break
    kept.clear()
    solver = CIPSISolver(merged)
    solver.expand(h_op, manifold, **expand_kwargs)
    es, psis = solver.get_eigenvectors(
        h_op,
        manifold,
        dense_cutoff=dense_cutoff,
        slaterWeightMin=slater_weight_min,
        psi_refs=solver.psi_refs,
        reuse_last=True,
    )
    found, expected = manifold_deficit(plan, verdicts, es, psis, merged)
    solver.manifold_check = {"found": found, "expected": expected}
    if found < expected and (basis.comm is None or basis.comm.rank == 0):
        print(
            f"WARNING: the thermal manifold holds {found} states of its lowest level, but certification "
            f"found that level in rows that make at least {expected}: thermal averages over it are "
            "misweighted. Raise the determinant cap, or widen the thermal window.",
            flush=True,
        )
    return es, psis, merged, solver, verdicts


#: Added to the measured spread of the classes' ``<N_imp>`` when bounding how far two class energies
#: can move relative to each other under a shift (:meth:`SectorPlanStore.jobs_at`). The spread is a
#: property of the classes' states at the certified shift, not a bound at another one; this is the
#: margin for that, in electrons.
N_IMP_SPREAD_SAFETY = 0.5

#: Floor on a class energy's error bar, added to the largest residual PT2: an eigensolve that
#: stagnated rather than converged is good to its residual norm, measured up to 5e-6 on the NiO
#: 10-bath d6 classes.
EXPANSION_ENERGY_FLOOR = 1e-5


@dataclass(frozen=True)
class SectorRecord:
    """One sector's certification: the plan, the shift it ran at and what every class found."""

    plan: CertificationPlan
    mu: float
    verdicts: tuple


class SectorPlanStore:
    """Certify each sector of a search once; at other shifts expand only the classes that can win.

    A search over a shift ``mu`` (``H(mu) = H - mu N_imp``) solves the same sectors again and again.
    The certification plan of a sector does not depend on the shift: the seed holds one impurity
    occupation, so the shift is a constant on it. And a class's lowest energy moves with the shift
    by ``-<N_imp>`` of its state (Hellmann-Feynman), so from the certified energies the classes that
    can still be lowest at another shift are known: those whose predicted energy is within
    ``|mu - mu_c| * (spread of <N_imp> + N_IMP_SPREAD_SAFETY)``, plus the classes' error bar, of the
    leader's. The margin widens with the distance from the certified shift, so a far shift expands
    every class again. Every class is expanded from the seed, never from another shift's basis, so a
    solve does not depend on which shifts came before. Records hold replicated scalars only.
    """

    def __init__(self):
        self.records = {}
        #: Sectors solved by a single expansion because their seed was too large to certify.
        self.uncertified = set()
        #: Sectors certified again because a solve contradicted the record (:meth:`violated`).
        self.recertified = set()
        #: The largest :func:`window_overtake` ratio of any certification.
        self.overtake = None

    def plan(self, key):
        """The certification plan of sector ``key``, or ``None`` if it was never certified."""
        record = self.records.get(key)
        return None if record is None else record.plan

    def record(self, key, plan, mu, verdicts):
        """Keep the first, full certification of sector ``key``; later partial ones add nothing."""
        if key not in self.records:
            self.records[key] = SectorRecord(plan, float(mu), tuple(verdicts))

    def clear(self):
        self.records.clear()
        self.uncertified.clear()
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
