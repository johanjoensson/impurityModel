r"""Certified fixed-N sector solves: every symmetry class that can hold the ground state is expanded.

A CIPSI expansion admits only determinants in the support of ``H|psi_ref>``, so it never leaves the
symmetry class of its reference, and the seed's lowest state -- the atomic multiplet, before
hybridization -- is in the wrong class whenever hybridization reorders the multiplets. On the NiO
10-bath gap-DC archive every sector from d5 to d9 converged on an exact excited level of another
symmetry, up to 0.17 above the ground state (hole-space exact diagonalization, d6-d9).

The classes come in two kinds, and are handled accordingly:

* **Hard labels**, diagonal on determinants: ``2 S_z`` and the occupation parities ``(-1)^{N_A}``
  ``H`` conserves (:func:`symmetries.conserved_parities`). A class expansion starts from the seed
  determinants of one label only; ``H`` then never leaves it, so no penalty and no restriction is
  needed beyond the existing ``S_z`` window, and different labels' expansions are disjoint.
* **Total spin**, which is soft (``S^2`` is not diagonal): when ``[H, S_+] = 0``, one expansion per
  ``S`` runs at ``2 S_z = 2S`` with the penalty ``lambda S^2`` in its eigensolves
  (:attr:`CIPSISolver.matrix_penalty`). At ``S_z = S`` every state has spin ``>= S``, so the penalty
  lifts exactly the higher spins and keeps the reference on spin ``S``.

The reported energy of a class is the *unpenalized* lowest eigenvalue of its final basis --
variational whatever the penalty steered -- and the lowest class wins. Hard labels that are
equivalent (the same seed spectra of ``P H P``, ``P H Q H P`` and a generic combination) are
expanded once.

Measured on NiO (d5-d9): the exact ground state in every sector; soft orbital (C3/C4) traps
*inside* a hard label exist but did not hit a ground state there.
"""

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from impurityModel.ed import seed_sectors
from impurityModel.ed.cipsi_solver import CIPSISolver
from impurityModel.ed.eigensolvers import Lowest
from impurityModel.ed.ManyBodyUtils import SlaterDeterminant, applyOp
from impurityModel.ed.observables import casimir_operator, make_spin_operators
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


@dataclass(frozen=True)
class CertificationJob:
    """One class expansion: the seed determinants with this hard label, at this penalty target."""

    two_sz: int | None  # None when S_z is not conserved
    parity: tuple
    spin: float | None  # S of the lambda S^2 tier; None: no penalty (the top spin, or no SU(2))
    n_seed: int
    duplicate_of: int | None = None  # index of an equivalent job expanded instead


@dataclass(frozen=True)
class ClassVerdict:
    """What one class expansion found. Replicated (every field is broadcast)."""

    job: CertificationJob
    energy: float
    n_dets: int
    residual_pt2: float
    cap_bound: bool
    memory_bound: bool


@dataclass(frozen=True)
class CertificationPlan:
    """The jobs of one seed, decided on rank 0 and broadcast."""

    jobs: list
    labels: dict  # seed determinant (chunk tuple) -> (two_sz, parity)
    pairs: list
    su2: bool
    penalty: float


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


def certification_plan(h_op, basis, *, max_seed=1000):
    """The class expansions a sector solve on the seed ``basis`` needs (collective), or ``None``.

    Rank 0 gathers the seed, labels every determinant and decides the jobs; the plan is broadcast.
    ``h_op`` must carry ``basis``'s restrictions (they are set here on every rank). ``None`` when the
    seed is empty or larger than ``max_seed`` (the plan is built from dense seed matrices): the
    caller then runs a single, uncertified expansion.
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
    plan = None
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
        distinct = sorted(set(labels.values()), key=lambda lab: ((lab[0] or 0), lab[1]))
        jobs = []
        if su2:
            two_s_values = sorted({abs(lab[0]) for lab in distinct})
            top = two_s_values[-1]
            for two_s in range(two_s_values[0], top + 1, 2):
                for lab in distinct:
                    if lab[0] == two_s:
                        n = sum(1 for v in labels.values() if v == lab)
                        jobs.append(CertificationJob(two_s, lab[1], two_s / 2 if two_s < top else None, n))
        else:
            for lab in distinct:
                n = sum(1 for v in labels.values() if v == lab)
                jobs.append(CertificationJob(lab[0], lab[1], None, n))
        a, m, _b, _out = seed_sectors.seed_matrices(h_op, seed)
        evals = np.linalg.eigvalsh(a)
        scale = max(float(np.max(np.abs(evals))), 1e-300)
        # Far above any gap that decides which spin wins: the seed's spread, or its hybridization
        # scale when the seed alone is (nearly) degenerate -- a d9 seed spreads over 2e-4 only.
        hybridization = float(np.sqrt(max(np.linalg.eigvalsh(m)[-1], 0.0))) if len(seed) else 0.0
        penalty = seed_sectors.PENALTY_FACTOR * max(float(evals[-1] - evals[0]), hybridization, 1e-6 * scale)
        index = {det: i for i, det in enumerate(seed)}
        members = [[index[d] for d, v in labels.items() if v == (j.two_sz, j.parity)] for j in jobs]
        for k, job in enumerate(jobs):
            for q in range(k):
                if jobs[q].duplicate_of is None and jobs[q].spin == job.spin and jobs[q].two_sz == job.two_sz:
                    if _equivalent(a, m, members[q], members[k], scale):
                        jobs[k] = CertificationJob(job.two_sz, job.parity, job.spin, job.n_seed, duplicate_of=q)
                        break
        plan = CertificationPlan(jobs, labels, pairs, su2, penalty)
    if comm is not None:
        plan = comm.bcast(plan, root=0)
    return plan


def _job_keys(plan, job, basis):
    """This rank's seed determinants with ``job``'s hard label."""
    want = (job.two_sz, job.parity)
    return [d for d in basis.local_basis if plan.labels.get(tuple(int(c) for c in d)) == want]


def _expand_job(h_op, basis, plan, job, s2_op, expand_kwargs, dense_cutoff, slater_weight_min, comm):
    """One class expansion: ``(verdict, es, psis, class_basis, solver)`` (collective)."""
    window = list(basis.weighted_restrictions or [])
    if job.two_sz is not None:
        window.append(sz_weighted_restriction(plan.pairs, job.two_sz))
    class_basis = basis.clone(initial_basis=_job_keys(plan, job, basis), weighted_restrictions=window)
    solver = CIPSISolver(class_basis)
    if job.spin is not None:
        solver.matrix_penalty = plan.penalty * s2_op
    solver.expand(h_op, Lowest(), **expand_kwargs)
    solver.matrix_penalty = None
    solver.eig = None
    es, psis = solver.get_eigenvectors(
        h_op, Lowest(), dense_cutoff=dense_cutoff, slaterWeightMin=slater_weight_min, psi_refs=solver.psi_refs
    )
    energy = float(np.min(es)) if len(es) else np.inf
    report = solver.truncation_report or {}
    residual = float((solver.convergence_report or {}).get("residual_pt2") or 0.0)
    fields = (energy, residual, bool(solver.truncation_report is not None), bool(report.get("memory_bound")))
    if comm is not None:
        fields = comm.bcast(fields, root=0)
    return ClassVerdict(job, fields[0], len(class_basis), *fields[1:]), es, psis, class_basis, solver


def certify_sector(h_op, basis, plan, *, expand_kwargs, dense_cutoff, slater_weight_min, verbose=False, within=None):
    """Expand every job of ``plan`` from ``basis`` (the seed) and return the lowest.

    Returns ``(es, psis, class_basis, class_solver, verdicts)``: the winner's unpenalized lowest
    eigenpair on its own basis, that basis (with the sector's window put back) and solver, and one
    :class:`ClassVerdict` per job expanded. Jobs run one after another; only the best so far is kept
    alive while the next one expands. Every branch reads a broadcast value. Collective.

    ``within`` (an energy window) also returns, as a sixth element, the determinant keys of every
    class inside the window above the winner -- the winner's first -- plus the largest residual PT2;
    and it expands the symmetry-equivalent duplicates of those classes, which hold the orbital
    partners of a degenerate multiplet. Only the keys are kept, never a class's basis or solver.
    """
    comm = basis.comm if basis.is_distributed else None
    rank = comm.rank if comm is not None else 0
    s2_op = None
    if plan.su2:
        s_plus, s_minus, s_z = make_spin_operators(plan.pairs)
        s2_op = casimir_operator(s_plus, s_minus, s_z)
    best = None
    verdicts = []
    keys = {}

    def run(k):
        nonlocal best
        verdict, es, psis, class_basis, solver = _expand_job(
            h_op, basis, plan, plan.jobs[k], s2_op, expand_kwargs, dense_cutoff, slater_weight_min, comm
        )
        verdicts.append(verdict)
        if within is not None:
            keys[k] = class_basis._keys.copy()
        if verbose and rank == 0:
            job = plan.jobs[k]
            label = f"2Sz={job.two_sz} parity={''.join(map(str, job.parity))}"
            tier = "" if job.spin is None else f" S={job.spin:g}"
            print(f"  class {label}{tier}: E = {verdict.energy:.8f} ({verdict.n_dets:,} determinants)", flush=True)
        if best is None or verdict.energy < best[0]:
            best = (verdict.energy, k, es, psis, class_basis, solver)

    for k, job in enumerate(plan.jobs):
        if job.duplicate_of is None:
            run(k)
    kept = None
    if within is not None:
        e_min = min(v.energy for v in verdicts)
        margin = within + max(v.residual_pt2 for v in verdicts)
        energy_of = {plan.jobs.index(v.job): v.energy for v in verdicts}
        inside = [k for k, e in energy_of.items() if e - e_min <= margin]
        for k, job in enumerate(plan.jobs):
            if job.duplicate_of in inside:
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

    ``psis`` is the winning class's lowest state, of total spin ``S`` but of whichever ``S_z`` the
    class held: a level degenerate across classes (a triplet's ``S_z = 0`` row sits, at the same
    energy, in the ``S = 0`` tier's class) can be won by any of its rows, and roundoff decides
    which. Stepping both ways until the image vanishes reaches every row. Each rank steps its own
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


def solve_certified(h_op, basis, plan, manifold, *, expand_kwargs, dense_cutoff, slater_weight_min, verbose=False):
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
        max_steps = max(abs(job.two_sz) for job in plan.jobs) * 2
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
