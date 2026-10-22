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
from impurityModel.ed import seed_sectors
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.density_matrices import occupations as state_occupations
from impurityModel.ed.eigensolvers import Lowest, cold_vector
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant, applyOp
from impurityModel.ed.observables import make_spin_operators
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
    stopped_early: bool = False
    #: The expansion's ``truncation_report`` (replicated scalars), when a cap or the memory guard
    #: limited it: a class can *lose* because it was held high, which the winner's own report misses.
    truncation_report: dict | None = None


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


def _expand_job(h_op, basis, plan, job, finished, expand_kwargs, dense_cutoff, slater_weight_min, comm):
    """One class expansion: ``(verdict, es, psis, class_basis, solver)`` (collective)."""
    class_basis = basis.clone(
        initial_basis=_job_keys(plan, job, basis), weighted_restrictions=_label_window(basis, plan.pairs, job.two_sz)
    )
    solver = CIPSISolver(class_basis)
    if job.seed_class is not None:
        class_plan = plan.class_plans[(job.two_sz, job.parity)]
        cls = class_plan.classes[job.seed_class]
        solver.matrix_hook = seed_sectors.ClassPenalty(class_plan, cls)

        previous = [np.inf]

        def leaked(sv, e_ref, psi_refs):
            weight = seed_sectors.class_weight(class_plan, cls, sv.basis, psi_refs)  # collective
            e0 = float(np.min(e_ref))
            # Only this label's: H conserves the label, so a state cannot leak into another label's.
            stop = _has_leaked(weight, e0, previous[0], finished.get((job.two_sz, job.parity), ()))
            previous[0] = e0
            return stop

        solver.cycle_hook = leaked
    solver.expand(h_op, Lowest(), **expand_kwargs)
    solver.matrix_hook = None
    solver.cycle_hook = None
    solver.eig = None
    es, psis = solver.get_eigenvectors(
        h_op, Lowest(), dense_cutoff=dense_cutoff, slaterWeightMin=slater_weight_min, psi_refs=solver.psi_refs
    )
    energy = float(np.min(es)) if len(es) else np.inf
    report = solver.truncation_report or {}
    residual = float((solver.convergence_report or {}).get("residual_pt2") or 0.0)
    impurity = [orb for blocks in basis.impurity_orbitals.values() for block in blocks for orb in block]
    # Collective (one Allreduce), entered on every rank: `psis.width` is replicated.
    n_imp = float(state_occupations(class_basis, psis, impurity).sum(axis=1)[0]) if psis.width else float("nan")
    fields = (
        energy,
        residual,
        bool(solver.truncation_report is not None),
        bool(report.get("memory_bound")),
        n_imp,
        bool(solver.stopped_early),
        None if solver.truncation_report is None else dict(solver.truncation_report),
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
    :class:`ClassVerdict` per job expanded. Jobs run one after another; only the best so far is kept
    alive while the next one expands. Every branch reads a broadcast value. Collective.

    ``within`` (an energy window) also returns, as a sixth element, the determinant keys of every
    class inside the window above the winner -- the winner's first -- plus the largest residual PT2;
    Only the keys are kept, never a class's basis or solver.

    ``only`` (job indices) restricts the expansions to those jobs (:meth:`SectorPlanStore.jobs_at`).
    """
    comm = basis.comm if basis.is_distributed else None
    rank = comm.rank if comm is not None else 0
    best = None
    verdicts = []
    finished = {}  # hard label -> energies its classes converged to
    keys = {}
    energy_of_job = {}

    def run(k):
        nonlocal best
        verdict, es, psis, class_basis, solver = _expand_job(
            h_op, basis, plan, plan.jobs[k], finished, expand_kwargs, dense_cutoff, slater_weight_min, comm
        )
        energy_of_job[k] = verdict.energy
        verdicts.append(verdict)
        job = plan.jobs[k]
        finished.setdefault((job.two_sz, job.parity), []).append(verdict.energy)
        if within is not None:
            keys[k] = class_basis._keys.copy()
            # Drop, as soon as they are known to be outside it, the keys of classes above the window:
            # holding every class's keys to the end costs the sum of all class sizes.
            e_min = min(v.energy for v in verdicts)
            margin = within + max(v.residual_pt2 for v in verdicts)
            # The lowest energy only falls and the margin only grows, so a class dropped here stays out.
            for q in [q for q in keys if energy_of_job[q] - e_min > margin]:
                del keys[q]
        if verbose and rank == 0:
            job = plan.jobs[k]
            label = f"2Sz={job.two_sz} parity={''.join(map(str, job.parity))}"
            cls = "" if job.seed_class is None else f" class {job.seed_class}"
            leak = " (leaked; stopped)" if verdict.stopped_early else ""
            print(f"  {label}{cls}: E = {verdict.energy:.8f} ({verdict.n_dets:,} determinants){leak}", flush=True)
        if best is None or verdict.energy < best[0]:
            best = (verdict.energy, k, es, psis, class_basis, solver)

    for k, job in enumerate(plan.jobs):
        if only is None or k in only:
            run(k)
    kept = None
    if within is not None:
        e_min = min(v.energy for v in verdicts)
        margin = within + max(v.residual_pt2 for v in verdicts)
        energy_of = {plan.jobs.index(v.job): v.energy for v in verdicts}
        inside = sorted((k for k, e in energy_of.items() if e - e_min <= margin), key=lambda k: energy_of[k])
        winner = best[1]
        kept = [keys[winner]] + [keys[k] for k in inside if k != winner]
        keys.clear()
    _energy, _k, es, psis, class_basis, solver = best
    class_basis.weighted_restrictions = basis.weighted_restrictions
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    if within is not None:
        return es, psis, class_basis, solver, verdicts, kept
    return es, psis, class_basis, solver, verdicts


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

    @staticmethod
    def _error(verdict):
        return verdict.residual_pt2 + EXPANSION_ENERGY_FLOOR

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
