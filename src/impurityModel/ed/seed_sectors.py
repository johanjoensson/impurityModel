r"""The symmetry classes of a charge sector's seed basis, and how each one is kept.

Why this exists
---------------
A CIPSI expansion only ever admits determinants in the support of ``H |psi_ref>``. ``H`` conserves
its symmetries, so an expansion can never leave the symmetry sectors its reference has weight in.
The reference of the first cycle is the ground state of the *seed* -- every impurity
configuration of the sector's charge with the bath full, no hybridization inside it -- whose
lowest level is the atomic Hund's multiplet in the crystal field. Hybridization can reorder the
multiplets (covalency favours eg holes, hence low spin), and then the true ground state lies in a
sector the seed's ground level has no component in. On the NiO 10-bath gap-DC search this trapped
expansions 0.15-0.17 above the ground state, which was larger than every energy difference the
search compared.

What it does
------------
1. **Classes, found without knowing the symmetry group.** Every symmetry of ``H`` leaves the seed
   space invariant and commutes with both ``A = P H P`` and ``M = P H Q H P`` (``P`` the seed
   projector, ``Q = 1 - P``). Diagonalizing a generic combination ``X = A + eps M + ...`` and
   joining eigenvectors that ``A``, ``M`` or any added operator couples gives a partition of the
   seed eigenstates that is never coarser than the true symmetry sectors -- continuous, discrete
   (point group), spin-orbit reduced or anti-unitary alike. The error direction is the safe one:
   a symmetry-unrelated pair that happens not to couple at this order is a *false split*, which
   costs one extra expansion; a *false merge* (two sectors joined) would silently drop one, and
   is guarded against by redrawing the combination whenever a degenerate level of ``X`` is not a
   partner multiplet.
2. **One expansion per class**, kept in its class by ``lambda Pi_other``, the projector onto the
   other classes' seed vectors, added to the matrix of every eigensolve (:class:`ClassPenalty`).
   It is approximate -- a state of another class with little seed weight is barely lifted -- so
   a class whose state leaves its seed span is detected (:func:`class_weight`) and the caller
   decides what that means (:mod:`sector_solver` stops it once it has re-converged on another
   class's energy). The reported energy is the unpenalized lowest eigenvalue of each class's
   final basis, so the answer stays variational whatever the penalty did.

No symmetry is assumed or labelled here -- not even total spin, which need not be a good quantum
number. :mod:`sector_solver` splits a sector by the labels that are exact on determinants
(``S_z``, occupation parities) first, and runs this partition inside each.

The classes are cut by an energy window on their seed energy (``GS_CERTIFY_WINDOW``, default
``GS_CERTIFY_WINDOW_FACTOR`` times the largest hybridization spread ``sqrt(<v|M|v>)``). Classes
with identical seed spectra are marked partners, but :mod:`sector_solver` expands them too: seed
spectra do not see the bath energies.
"""

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sps
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.DistributedSparseMatrix import DistributedSparseMatrix
from impurityModel.ed.ManyBodyUtils import ManyBodyState, SlaterDeterminant, applyOp

#: Relative size of an off-diagonal element, in the eigenbasis of ``X``, that joins two seed
#: eigenvectors into one class. Exact symmetry zeros sit at roundoff (~1e-15 relative), genuine
#: couplings at the scale of the hybridization; a tight cut errs toward false splits, the safe side.
COUPLING_TOL = 1e-9
#: Two ``X`` eigenvalues closer than this (relative) are treated as one degenerate level.
_DEGENERACY_TOL = 1e-10
#: Two classes are partners when their seed spectra agree to this (relative).
PARTNER_TOL = 1e-8
#: Generic, mutually irrational weights of the operators in ``X`` (golden ratio, sqrt 2, sqrt 3,
#: sqrt 5 fractional parts). Deterministic, so the classes are the same on every rank and run.
_DRAWS = (
    (0.6180339887498949, 0.4142135623730951, 0.7320508075688772),
    (0.2360679774997898, 0.8284271247461903, 0.4641016151377544),
    (0.3819660112501051, 0.6457513110645906, 0.1622776601683795),
)
#: The penalties are this multiple of the seed spectrum's width: far above any gap that decides
#: which class wins, small enough not to wreck the Lanczos conditioning.
PENALTY_FACTOR = 2.0
#: A class's final state must keep at least this fraction of its seed weight inside the class.
DRIFT_TOL = 0.5


@dataclass(frozen=True)
class SeedClass:
    """One symmetry class of the seed: its span, lowest seed state, labels and seed spectrum."""

    span: np.ndarray  # (n0, k) orthonormal columns, canonical seed coordinates
    energy: float  # lowest eigenvalue of P H P inside the class
    spectrum: np.ndarray  # every eigenvalue of P H P inside the class, ascending
    m_spectrum: np.ndarray  # every eigenvalue of P H Q H P inside the class, ascending
    sigma: float  # sqrt(<v|P H Q H P|v>) of the lowest state: its hybridization spread
    status: str = "expand"  # "expand", "partner" (of an earlier class) or "window"


@dataclass(frozen=True)
class SeedClassPlan:
    """The classes of one seed, replicated on every rank (decided on rank 0 and broadcast)."""

    seed: list  # canonical (sorted) seed determinants, as chunk tuples
    classes: list  # every class, ascending in energy, with its status
    window: float
    penalty: float
    coupled_ok: bool  # False when no draw of X avoided a non-partner degeneracy

    @property
    def expanded(self):
        return [c for c in self.classes if c.status == "expand"]


def _comm_of(basis):
    return basis.comm if basis.is_distributed else None


def gather_seed(basis):
    """Every seed determinant as a chunk tuple, sorted: one canonical order on every rank count."""
    local = [tuple(int(chunk) for chunk in det) for det in basis.local_basis]
    comm = _comm_of(basis)
    parts = comm.allgather(local) if comm is not None else [local]
    return sorted(det for part in parts for det in part)


def _images(op, det):
    """``op |det>`` as ``{chunk tuple: amplitude}``, with whatever restrictions ``op`` carries."""
    out = applyOp(op, ManyBodyState({SlaterDeterminant(det): 1.0}), cutoff=0)
    return {tuple(int(chunk) for chunk in key): complex(row[0]) for key, row in out.items()}


def seed_matrices(h_op, seed):
    """``A = P H P`` and ``M = P H Q H P`` in canonical seed coordinates, plus ``H`` on the first shell."""
    index = {det: i for i, det in enumerate(seed)}
    n0 = len(seed)
    a = np.zeros((n0, n0), dtype=complex)
    outside = {}
    rows, cols, vals = [], [], []
    for j, det in enumerate(seed):
        for key, amp in _images(h_op, det).items():
            i = index.get(key)
            if i is not None:
                a[i, j] += amp
            else:
                rows.append(outside.setdefault(key, len(outside)))
                cols.append(j)
                vals.append(amp)
    b = sps.csc_array((vals, (rows, cols)), shape=(len(outside), n0))
    m = (b.conj().T @ b).toarray()
    return 0.5 * (a + a.conj().T), 0.5 * (m + m.conj().T), b, outside


def _components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=min)


def _is_scalar(mat, tol):
    if mat.shape[0] < 2:
        return True
    shifted = mat - np.trace(mat) / mat.shape[0] * np.eye(mat.shape[0])
    return np.linalg.norm(shifted) <= tol


def _partition(ops):
    """Classes of the seed eigenvectors under the Hermitian ``ops``: ``(groups, W, ok)``.

    ``X`` is a generic combination of ``ops``. A degenerate level of ``X`` is accepted only when
    every operator is a scalar on it -- the signature of partner rows of one multiplet, which a
    symmetry maps onto each other. A level that fails it is an *accidental* degeneracy between
    different classes, where ``eigh`` may mix them into a false merge, so ``X`` is redrawn.
    """
    scales = [max(np.linalg.norm(op, 2), 1e-300) for op in ops]
    normalized = [op / s for op, s in zip(ops, scales)]
    n = ops[0].shape[0]
    for draw in _DRAWS:
        weights = (1.0,) + tuple(draw[k % len(draw)] for k in range(len(ops) - 1))
        x = sum(w * op for w, op in zip(weights, normalized))
        evals, w_mat = np.linalg.eigh(0.5 * (x + x.conj().T))
        rotated = [w_mat.conj().T @ op @ w_mat for op in normalized]
        levels = _components(n, [(i, i + 1) for i in range(n - 1) if evals[i + 1] - evals[i] <= _DEGENERACY_TOL])
        ok = all(_is_scalar(r[np.ix_(lvl, lvl)], COUPLING_TOL * np.sqrt(len(lvl))) for lvl in levels for r in rotated)
        edges = [(lvl[0], k) for lvl in levels for k in lvl[1:]]
        # `eigh` mixes two eigenvectors of gap g by ~eps ||X|| / g, which leaks (r_aa - r_bb) times
        # that into r_ab: an apparent coupling between close levels that is pure roundoff, and an
        # edge built on it would be a false merge. Only couplings above that floor count.
        gaps = np.abs(evals[:, None] - evals[None, :])
        mixing = 100.0 * np.finfo(float).eps * max(np.max(np.abs(evals)), 1.0) / np.maximum(gaps, 1e-300)
        for r in rotated:
            d = np.real(np.diag(r))
            floor = COUPLING_TOL + mixing * np.abs(d[:, None] - d[None, :])
            ii, jj = np.nonzero(np.triu(np.abs(r) > floor, 1))
            edges.extend(zip(ii.tolist(), jj.tolist()))
        if ok:
            return _components(n, edges), w_mat, True
    return _components(n, edges), w_mat, False


def _build_classes(a, m):
    groups, w_mat, ok = _partition([a, m])
    classes = []
    for group in groups:
        span = w_mat[:, group]
        evals, vecs = np.linalg.eigh(span.conj().T @ a @ span)
        lowest = span @ vecs[:, 0]
        sigma = float(np.sqrt(max(np.real(np.vdot(lowest, m @ lowest)), 0.0)))
        m_evals = np.linalg.eigvalsh(span.conj().T @ m @ span)
        classes.append(SeedClass(span, float(evals[0]), evals, m_evals, sigma))
    # Degenerate classes (partners) differ in energy by roundoff only, so they are ordered on a grid
    # of the energy, and within one grid cell by their place in the eigenbasis (a stable sort): the
    # same order, and the same representative, at every shift and on every machine.
    scale = max(float(np.max(np.abs(np.linalg.eigvalsh(a)))), 1e-300)
    return sorted(classes, key=lambda c: int(np.rint(c.energy / (scale * PARTNER_TOL)))), ok


def _same_spectrum(x, y, scale, m_scale):
    """Whether ``x`` and ``y`` can be partners: a symmetry maps one onto the other, so *every*
    operator that commutes with it has the same spectrum on both. ``P H P`` alone is not enough: two
    orbitals degenerate in the crystal field but hybridized differently share it, and deduplicating
    them drops the one that wins."""
    return (
        len(x.spectrum) == len(y.spectrum)
        and np.max(np.abs(x.spectrum - y.spectrum)) <= PARTNER_TOL * scale
        and np.max(np.abs(x.m_spectrum - y.m_spectrum)) <= PARTNER_TOL * m_scale
    )


def _select(classes, window, scale, m_scale):
    """Mark partners and out-of-window classes; of a set of partners the lowest is kept."""
    e_min = classes[0].energy
    out = []
    for c in classes:
        status = "window" if c.energy - e_min > window else "expand"
        out.append(replace(c, status=status))

    groups = []
    for i, c in enumerate(out):
        for group in groups:
            if _same_spectrum(c, out[group[0]], scale, m_scale):
                group.append(i)
                break
        else:
            groups.append([i])
    for group in groups:
        keep = group[0]
        for i in group:
            if i != keep and out[i].status == "expand":
                c = out[i]
                out[i] = replace(c, status="partner")
    return out


def _window(classes):
    explicit = config.GS_CERTIFY_WINDOW.get()
    if explicit is not None:
        return float(explicit)
    return float(config.GS_CERTIFY_WINDOW_FACTOR.get()) * max(c.sigma for c in classes)


def seed_class_plan(h_op, basis, *, max_seed=1000, verbose=False):
    """The symmetry classes of ``basis`` (a freshly built seed) under ``h_op``, or ``None``.

    Partitioned by ``P H P`` and ``P H Q H P`` alone: every symmetry ``H`` has is found, none is
    assumed or labelled. ``None`` when the seed is empty or larger than ``max_seed`` (the seed
    matrices are dense). Collective; the plan is computed on rank 0 and broadcast, so every rank
    holds the same classes and makes the same decisions on them. ``h_op`` must carry ``basis``'s
    restrictions, as the expansion will apply it.
    """
    comm = _comm_of(basis)
    n0 = len(basis)
    if n0 == 0 or n0 > max_seed:
        return None
    seed = gather_seed(basis)
    rank = comm.rank if comm is not None else 0
    plan = None
    # On every rank, not only on the one that builds the plan: whatever applies ``h_op`` next must
    # see the same operator everywhere.
    h_op.set_restrictions(basis.restrictions)
    h_op.set_weighted_restrictions(basis.weighted_restrictions)
    if rank == 0:
        a, m, _b, _outside = seed_matrices(h_op, seed)
        classes, ok = _build_classes(a, m)
        evals_a = np.linalg.eigvalsh(a)
        width = float(evals_a[-1] - evals_a[0]) if n0 > 1 else 0.0
        scale = max(float(np.max(np.abs(evals_a))), 1e-300)
        window = _window(classes)
        classes = _select(classes, window, scale, max(float(np.linalg.norm(m, 2)), 1e-300))
        penalty = PENALTY_FACTOR * max(width, window, 1e-6 * scale)
        plan = SeedClassPlan(seed, classes, window, penalty, ok)
    if comm is not None:
        plan = comm.bcast(plan, root=0)
    if verbose and rank == 0:
        print_plan(plan)
    return plan


def print_plan(plan):
    n_exp = len(plan.expanded)
    print(
        f"Seed symmetry classes: {len(plan.classes)} found over {len(plan.seed)} seed determinants, "
        f"{n_exp} outside a partner set (window {plan.window:.4g})",
        flush=True,
    )
    if not plan.coupled_ok:
        print(
            "warning: every draw of the class-finding combination hit an accidental degeneracy; "
            "two symmetry classes may have been merged, and one of them is then not expanded.",
            flush=True,
        )
    e0 = plan.classes[0].energy
    for k, c in enumerate(plan.classes):
        print(
            f"  class {k:3d}: dim={c.span.shape[1]:4d} E_seed-E_min={c.energy - e0:.6f} "
            f"sigma={c.sigma:.4g} {c.status}",
            flush=True,
        )


def class_seed_keys(plan, cls, basis):
    """The seed determinants (``SlaterDeterminant``) of ``cls``'s support this rank owns."""
    support = np.max(np.abs(cls.span), axis=1) > COUPLING_TOL
    keep = {plan.seed[i] for i in np.flatnonzero(support)}
    return [det for det in basis.local_basis if tuple(int(chunk) for chunk in det) in keep]


class ClassPenalty:
    """``H_mat -> H_mat + lambda Pi_other`` on the current basis.

    Installed as :attr:`cipsi_solver.CIPSISolver.matrix_hook`. Collective, like the matrix build it
    follows: every rank calls :meth:`Basis.contains` and :meth:`Basis.index` with the same list.
    """

    def __init__(self, plan, cls):
        self.lam = plan.penalty
        self.seed = [SlaterDeterminant(det) for det in plan.seed]
        # Pi_other over the whole seed: the complement of the class's span, so a seed determinant
        # of another class that a later selection round re-admits is lifted too.
        self.span = cls.span
        self.pi = np.eye(len(plan.seed), dtype=complex) - cls.span @ cls.span.conj().T

    def _pi_on(self, idx):
        """``Pi_other`` on the seed determinants ``idx`` still in the basis.

        The principal submatrix of ``1 - span span^H`` is not a projector once a cap has dropped
        some of the class's own seed determinants: it penalizes the class's own truncated state by
        ``|P s|^2 (1 - |P s|^2)``. The span is re-orthonormalized on the surviving rows instead.
        """
        if len(idx) == len(self.seed):
            return self.pi
        rows = self.span[idx]
        u, sv, _vh = np.linalg.svd(rows, full_matrices=False)
        kept = u[:, sv > COUPLING_TOL * max(float(sv[0]) if sv.size else 0.0, 1e-300)]
        return np.eye(len(idx), dtype=complex) - kept @ kept.conj().T

    def __call__(self, basis, h_mat):
        n = len(basis)
        offset = getattr(basis, "offset", 0) or 0
        n_local = len(basis.local_basis)
        local = h_mat.to_scipy().tocsc()
        present = np.fromiter(basis.contains(self.seed), dtype=bool, count=len(self.seed))
        dets = [det for det, p in zip(self.seed, present) if p]
        idx = np.flatnonzero(present)
        if dets:
            rows = np.fromiter(basis.index(dets), dtype=np.int64, count=len(dets))
            mine = (rows >= offset) & (rows < offset + n_local)
            block = self.lam * self._pi_on(idx)[:, mine]
            rr, cc = np.nonzero(block)
            pen = sps.csc_array(
                (block[rr, cc], (rows[rr], rows[mine][cc] - offset)),
                shape=(n, n_local),
            )
            local = local + pen
        return DistributedSparseMatrix.from_scipy(local, h_mat.comm, offset)


def class_weight(plan, cls, basis, psis):
    """Fraction of the state's seed weight that lies inside ``cls``'s span (collective)."""
    (state,) = psis.select([0]).to_states() if psis.width > 1 else psis.to_states()
    local = {tuple(int(c) for c in det): complex(row[0]) for det, row in state.items()}
    vec = np.array([local.get(det, 0.0) for det in plan.seed], dtype=complex)
    comm = _comm_of(basis)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, vec, op=MPI.SUM)
    total = float(np.vdot(vec, vec).real)
    if total <= 0.0:
        return 0.0
    inside = cls.span.conj().T @ vec
    return float(np.vdot(inside, inside).real) / total
