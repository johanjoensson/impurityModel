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
  partitioned by ``P H P`` and ``P H Q H P`` (:func:`seed_sectors.build_seed_class_plan`, with no
  spin assumption), and each class is expanded from its own seed states, held there by the class
  penalty.

Total spin is deliberately *not* used as a soft label: it need not be a good quantum number, and
where it is, a ``lambda S^2`` penalty on a truncated basis -- not closed under ``S_+-`` -- steered the
NiO d7 sector into a spin-mixed state 0.053 high at three of four shifts while being necessary for
d6.

Each class reports its unpenalized lowest eigenvalue -- variational whatever the penalty did -- and
the lowest class wins. Two savings keep the cost near a single expansion's: hard labels that are
equivalent (the same seed spectra of ``P H P``, ``P H Q H P`` and a generic mix) expand once, and a
class whose state has left its seed span and re-converged on an energy another class already found
stops early (:data:`LEAK_ENERGY_TOL`): it can only re-derive that class's state.
"""

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from impurityModel.ed import seed_sectors
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.density_matrices import occupations as state_occupations
from impurityModel.ed.eigensolvers import Lowest
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant, applyOp
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
#: :data:`seed_sectors.DRIFT_TOL`) stops once its energy is this close above an energy another class
#: already converged to: it has leaked into that class's state and can only re-derive it. On the
#: NiO 10-bath d6 sector six of twelve class expansions stopped this way (459 s to 113 s, the
#: same ground state). A leaked state still descending through that window towards a *new*, lower
#: state would be stopped too; it would have to sit within this of an existing level to do so.
LEAK_ENERGY_TOL = 1e-3


@dataclass(frozen=True)
class CertificationJob:
    """One class expansion: the seed states of ``seed_class`` inside the hard label ``(two_sz, parity)``."""

    two_sz: int | None  # None when S_z is not conserved
    parity: tuple
    seed_class: int | None  # index into the label's class plan; None: the label's whole seed
    n_seed: int
    duplicate_of: int | None = None  # a job of the equivalent label expanded instead


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


def _equivalent(a, m, idx_a, idx_b, scale):
    """Do two hard labels' seed blocks have the same spectra (P H P, P H Q H P, a generic mix)?"""
    if len(idx_a) != len(idx_b):
        return False
    tol = seed_sectors.PARTNER_TOL * scale
    for t in (0.0, None, 0.6180339887498949):
        pick = (lambda x: x[np.ix_(idx_a, idx_a)], lambda x: x[np.ix_(idx_b, idx_b)])
        if t is None:
            xa, xb = pick[0](m), pick[1](m)
        else:
            xa, xb = pick[0](a) + t * pick[0](m), pick[1](a) + t * pick[1](m)
        if np.max(np.abs(np.linalg.eigvalsh(xa) - np.linalg.eigvalsh(xb))) > tol:
            return False
    return True


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

    Rank 0 gathers the seed, detects the hard labels and decides which are needed and which are
    equivalent; then every needed label's seed is partitioned into its soft classes. ``h_op``'s
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
        a, m, _b, _out = seed_sectors.seed_matrices(h_op, seed)
        scale = max(float(np.max(np.abs(np.linalg.eigvalsh(a)))), 1e-300)
        index = {det: i for i, det in enumerate(seed)}
        members = {lab: [index[d] for d, v in labels.items() if v == lab] for lab in needed}
        source = {}
        for k, lab in enumerate(needed):
            for ref in needed[:k]:
                if ref not in source and ref[0] == lab[0] and _equivalent(a, m, members[ref], members[lab], scale):
                    source[lab] = ref
                    break
        head = (labels, needed, source, pairs, su2)
    if comm is not None:
        head = comm.bcast(head, root=0)
    labels, needed, source, pairs, su2 = head
    class_plans, jobs, first_job = {}, [], {}
    for lab in needed:
        # Collective: every rank builds the label's seed and its class plan, whatever its share.
        sub = basis.clone(
            initial_basis=_label_keys(labels, lab, basis), weighted_restrictions=_label_window(basis, pairs, lab[0])
        )
        plan = seed_sectors.build_seed_class_plan(h_op, sub, max_seed=max_seed, spin_tier=False)
        class_plans[lab] = plan
        n_label = sum(1 for v in labels.values() if v == lab)
        dup = first_job.get(source[lab]) if lab in source else None
        first_job[lab] = len(jobs)
        if plan is None or len(plan.expanded) <= 1:
            jobs.append(CertificationJob(lab[0], lab[1], None, n_label, dup))
        else:
            for k, cls in enumerate(plan.classes):
                if cls.status == "expand":
                    n = int(np.count_nonzero(np.max(np.abs(cls.span), axis=1) > seed_sectors.COUPLING_TOL))
                    jobs.append(CertificationJob(lab[0], lab[1], k, n, dup))
    h_op.set_restrictions(basis.restrictions)
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    return CertificationPlan(jobs, labels, pairs, su2, class_plans)


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

        def leaked(sv, e_ref, psi_refs):
            weight = seed_sectors.class_weight(class_plan, cls, sv.basis, psi_refs)  # collective
            e0 = float(np.min(e_ref))
            return weight < seed_sectors.DRIFT_TOL and any(0.0 <= e0 - e < LEAK_ENERGY_TOL for e in finished)

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
    and it expands the classes of the labels equivalent to those classes' labels, which hold the
    orbital partners of a degenerate multiplet. Only the keys are kept, never a class's basis or solver.

    ``only`` (job indices) restricts the expansions to those jobs (:meth:`SectorPlanStore.jobs_at`).
    """
    comm = basis.comm if basis.is_distributed else None
    rank = comm.rank if comm is not None else 0
    best = None
    verdicts = []
    finished = []
    keys = {}

    def run(k):
        nonlocal best
        verdict, es, psis, class_basis, solver = _expand_job(
            h_op, basis, plan, plan.jobs[k], finished, expand_kwargs, dense_cutoff, slater_weight_min, comm
        )
        verdicts.append(verdict)
        finished.append(verdict.energy)
        if within is not None:
            keys[k] = class_basis._keys.copy()
        if verbose and rank == 0:
            job = plan.jobs[k]
            label = f"2Sz={job.two_sz} parity={''.join(map(str, job.parity))}"
            cls = "" if job.seed_class is None else f" class {job.seed_class}"
            leak = " (leaked; stopped)" if verdict.stopped_early else ""
            print(f"  {label}{cls}: E = {verdict.energy:.8f} ({verdict.n_dets:,} determinants){leak}", flush=True)
        if best is None or verdict.energy < best[0]:
            best = (verdict.energy, k, es, psis, class_basis, solver)

    for k, job in enumerate(plan.jobs):
        if job.duplicate_of is None and (only is None or k in only):
            run(k)
    kept = None
    if within is not None:
        e_min = min(v.energy for v in verdicts)
        margin = within + max(v.residual_pt2 for v in verdicts)
        energy_of = {plan.jobs.index(v.job): v.energy for v in verdicts}
        inside = [k for k, e in energy_of.items() if e - e_min <= margin]
        inside_labels = {(plan.jobs[k].two_sz, plan.jobs[k].parity) for k in inside}
        for k, job in enumerate(plan.jobs):
            if job.duplicate_of is not None:
                src = plan.jobs[job.duplicate_of]
                if (src.two_sz, src.parity) in inside_labels:
                    run(k)
                    inside.append(k)
        winner = best[1]
        kept = [keys[winner]] + [keys[k] for k in inside if k != winner]
        keys.clear()
    _energy, _k, es, psis, class_basis, solver = best
    class_basis.weighted_restrictions = basis.weighted_restrictions
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    if within is not None:
        return es, psis, class_basis, solver, verdicts, kept
    return es, psis, class_basis, solver, verdicts


def _spin_partner_keys(psis, pairs, max_steps):
    """This rank's share of the supports of ``S_+^k psi`` and ``S_-^k psi``, ``k >= 1``.

    ``psis`` is the winning class's lowest state, of total spin ``S`` and of whichever ``S_z`` its
    hard label held (the smallest ``|S_z|``, where every multiplet has a member). Stepping both
    ways until the image vanishes reaches every row. Each rank steps its own
    rows only; by linearity the union over the ranks covers every partner's support, so this needs
    no communication -- ``add_states`` routes the keys afterwards.
    """
    s_plus, s_minus, _s_z = make_spin_operators(pairs)
    (state,) = psis.select([0]).to_states() if psis.width > 1 else psis.to_states()
    keys = []
    for ladder in (s_plus, s_minus):
        ladder.set_restrictions(None)
        ladder.set_weighted_restrictions(None)
        image = state
        for _ in range(int(max_steps)):
            image = applyOp(ladder, image, cutoff=0)
            if not len(image):
                break
            keys.extend(image.keys())
    return keys


def solve_certified(
    h_op, basis, plan, manifold, *, expand_kwargs, dense_cutoff, slater_weight_min, verbose=False, only=None
):
    """A certified sector solve for ``manifold``: ``(es, psis, basis, solver, verdicts)``.

    ``Lowest`` is :func:`certify_sector`'s winner. ``Within`` needs every state of the window, and
    a class expansion holds one ``S_z`` row of one hard label only, so the window is solved on the
    union of: the seed; every class whose energy lies within the window (plus the largest residual
    PT2, the expansions' error bar) -- symmetry-equivalent labels included, since they hold the
    orbital partners of a multiplet; and the winner's spin partners ``S_+-^k psi``. The union is
    expanded for the window. It is never larger than the basis's cap would allow a single
    expansion to reach: past the cap only the winner's own classes and spin partners are kept.
    Collective.
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
    )
    partners = []
    if plan.su2:
        # 2 S_max steps reach every row from any row; the largest seed 2 S_z bounds 2 S_max.
        max_steps = 2 * max(abs(label[0]) for label in plan.labels.values())
        partners = _spin_partner_keys(psis, plan.pairs, max_steps)
    del es, psis, _solver
    merged = basis.clone(initial_basis=list(basis.local_basis))
    merged.add_states(list(winner_basis.local_basis) + partners)
    del winner_basis
    cap = basis.truncation_threshold
    for keys in kept[1:]:
        merged.add_states(keys.keys())
    if cap is not None and merged.size > cap:
        merged = basis.clone(initial_basis=list(basis.local_basis))
        merged.add_states(kept[0].keys() + partners)
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

    def jobs_at(self, key, mu):
        """Indices of the jobs worth expanding at ``mu``, or ``None`` (no usable record: expand all)."""
        record = self.records.get(key)
        if record is None:
            return None
        verdicts = record.verdicts
        if not verdicts or not all(np.isfinite(v.energy) and np.isfinite(v.n_imp) for v in verdicts):
            return None
        shift = float(mu) - record.mu
        predicted = [v.energy - shift * v.n_imp for v in verdicts]
        lead = min(predicted)
        spread = max(v.n_imp for v in verdicts) - min(v.n_imp for v in verdicts)
        error = max(v.residual_pt2 for v in verdicts) + EXPANSION_ENERGY_FLOOR
        margin = abs(shift) * (spread + N_IMP_SPREAD_SAFETY) + error
        return {record.plan.jobs.index(v.job) for v, e in zip(verdicts, predicted) if e - lead <= margin}
