import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.basis_transcription import (
    MatrixSnapshot,
    build_distributed_vector,
    build_sparse_matrix,
    build_state,
    extend_sparse_matrix,
)
from impurityModel.ed.cipsi_budget import CapRefiner, MemoryGuard
from impurityModel.ed.eigensolvers import (
    EigenResult,
    Lowest,
    Within,
    cold_vector,
    eigensystem,
    solve_lowest,
    solve_window,
)
from impurityModel.ed.manybody_basis import (
    Basis,
    collective_mass_cutoff,
    collective_top_k_mask,
)
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.memory_estimate import (
    current_rss_bytes,
    emit_memory_warning,
    format_bytes,
    peak_rss_bytes,
    reset_peak_rss,
    rss_breakdown,
)
from impurityModel.ed.solver_trace import note as _trace_note
from impurityModel.ed.solver_trace import timed as _trace_timed

#: Headroom, in states, on the number of eigenstates one CIPSI cycle asks the eigensolver to retain
#: initially (:func:`_manifold_request`). The window solve grows its retention past it on demand.
_EIGENSTATE_PAD = 10

#: Largest basis a ``Lowest`` solve diagonalizes densely, whatever ``dense_cutoff`` allows. The dense
#: branch computes *every* eigenpair on rank 0 and broadcasts all ``n**2`` amplitudes, every CIPSI
#: cycle; a ``Lowest`` request needs one. Measured on the NiO 10-bath d7 sector (an 866-determinant
#: class, 18 cycles under ``dense_cutoff=1000``): 16.5 s of a 17.8 s expansion was ``eigh``, and the
#: same expansion through the count-mode solve took 2.4 s to the same energy. ``Within`` keeps the
#: dense branch up to ``dense_cutoff``: there it is the complete-by-construction certificate.
_DENSE_LOWEST_MAX = 64

#: Largest share of the expansion's per-rank memory budget the kept Hamiltonian matrix may take.
#: :meth:`CIPSISolver._build_h_matrix` keeps the last matrix through the selection round so the next
#: cycle can grow it (:func:`basis_transcription.extend_sparse_matrix`) instead of rebuilding it;
#: the selection round is the expansion's memory peak, so the kept matrix adds to it. Above this
#: share the matrix is dropped and the next cycle rebuilds -- the old behaviour, at the scale where
#: the build was measured to be a few percent of a cycle.
_SNAPSHOT_BUDGET_FRACTION = 0.1

#: Generic start columns per eigensolve (``R``); raised to ``max_multiplicity + 1`` after a solve that found more.
_MIN_RANDOM_COLUMNS = 4


#: *Floor* on the tolerance below which reference eigenvalues are treated as one degenerate
#: manifold. A restarted Lanczos converges the invariant *subspace*, not a basis within it --
#: every rotation of a degenerate block has the same residual -- so any per-eigenvector score
#: must be summed over the block to be well defined. Far below any physical splitting of
#: interest (crystal field, exchange and spin-orbit are all >= 1e-2 eV here).
#:
#: This is a floor, not the tolerance: use :func:`_degeneracy_tol`, which raises it to track the
#: residual the eigensolver was actually asked to achieve. It used to be the tolerance, and the
#: claim that it sits "well above the eigensolver's attainable eigenvalue accuracy" was false on
#: every path that does not pass a tight ``slaterWeightMin`` -- see :func:`_degeneracy_tol`.
DEGENERACY_TOL = 1e-9

#: Residual Epstein-Nesbet PT2 energy at which :meth:`CIPSISolver.expand` calls the reference
#: states converged. A bound on the *sum* over every candidate left out, not on each one: the
#: per-candidate ``de2_min`` it replaced as the accuracy control stopped the SrMnO3 production
#: ground state with 5.1e-4 of PT2 energy left below a 1e-8 floor (doc/plans/dc_smo_memory.md).
DEFAULT_E_PT2_TOL = 1e-8

#: How far above the eigensolver's residual tolerance the degeneracy grouping tolerance must sit.
#: A Ritz pair converged to ``||r||`` has ``|theta - lambda| <= ||r||``, and *within* a cluster the
#: returned Ritz values are generically split by ``O(||r||)``, so grouping at anything below the
#: residual splits manifolds that are exactly degenerate. One order of margin: large enough that
#: an ``O(||r||)`` splitting is absorbed, small enough to stay orders below any physical scale.
DEGENERACY_TOL_MARGIN = 10.0

#: Loosest eigenvector residual we ever accept -- the historical hard-coded default. Deriving the
#: tolerance from ``slaterWeightMin`` must never make the eigensolver *lazier* than this.
_EIGEN_TOL_MAX = 1e-8

#: Tightest residual worth asking for: below ``eps * ||H||`` (``||H||`` of order 1e2 here) the
#: residual is pure roundoff and the restart loop would never terminate.
_EIGEN_TOL_FLOOR = 1e-13


def _eigen_tol(slaterWeightMin):
    """Eigenvector residual tolerance implied by the amplitude cutoff ``slaterWeightMin``.

    An eigenvector converged to residual ``||r||`` carries spurious amplitudes of order ``||r||``.
    If ``||r||`` exceeds the cutoff those amplitudes survive the ``slaterWeightMin`` prune, and
    *which* of them survive is decided by rounding -- so the state's support, the excited basis
    seeded from it and the Green's function all become rank-count dependent. Measured on the NiO
    ground state: at ``||r|| = 2.2e-9`` and ``slaterWeightMin = 1e-12`` the support varied between
    4205 and 5099 determinants; converged to ``1.8e-11`` the smallest genuine amplitude is 4.5e-10,
    three orders above the cutoff, and the support is identical at every rank count.

    So converge until the residual sits at or below the cutoff. Clamped so this can only ever
    tighten the historical ``1e-8``, and never chase roundoff. ``slaterWeightMin <= 0`` means no
    pruning happens at all, so the noise is harmless and the loose default is kept.
    """
    if not slaterWeightMin or slaterWeightMin <= 0:
        return _EIGEN_TOL_MAX
    return max(min(float(slaterWeightMin), _EIGEN_TOL_MAX), _EIGEN_TOL_FLOOR)


def _degeneracy_tol(e_ref, slaterWeightMin):
    """The tolerance at which ``e_ref`` may be partitioned into degenerate manifolds.

    ``DEGENERACY_TOL`` alone is **not** safe as a fixed absolute tolerance, because it is
    unrelated to how hard the eigensolver was asked to work. A Ritz pair at residual ``||r||``
    satisfies ``|theta - lambda| <= ||r||``, and the members of a genuinely degenerate cluster
    come back split by ``O(||r||)``. So whenever ``_eigen_tol(slaterWeightMin)`` exceeds
    ``DEGENERACY_TOL / DEGENERACY_TOL_MARGIN`` the grouping is *tighter than the solver's own
    accuracy*, and an exactly degenerate manifold is split into singletons -- which is precisely
    the failure the manifold-summed CIPSI score and :func:`_energy_cut_indices` exist to prevent.

    That was the case in production, not in principle: ``_eigen_tol`` returns ``_EIGEN_TOL_MAX``
    (1e-8, ten times *looser* than ``DEGENERACY_TOL``) whenever ``slaterWeightMin`` is 0 or above
    1e-8, and two manifold-consuming paths run exactly there -- ``dc_criteria``
    defaults ``slater_weight_min=0``, and the occupation walk ran at
    ``sqrt(slaterWeightMin)`` (1e-6 for the usual 1e-12). The production ground-state path
    (``slaterWeightMin = 1e-12``) is unaffected: 10 * 1e-12 is far under the 1e-9 floor.

    The third term keeps the tolerance meaningful when the spectrum itself is large. Eigenvalues
    of magnitude ``E`` carry ``eps * E`` of rounding noise, so a fixed 1e-9 stops being "well
    above roundoff" once ``E`` reaches ~1e5 (this code has run charge-transfer poles at 1.5e4).

    Rank-invariant: a pure function of ``e_ref``, which the callers either broadcast or rely on
    being bit-identical (see the comment in :meth:`CIPSISolver.get_eigenvectors`).

    Parameters
    ----------
    e_ref : array_like
        The reference eigenvalues about to be grouped. Only their magnitude is used.
    slaterWeightMin : float or None
        The amplitude cutoff the eigensolver was run at, i.e. the argument
        :func:`_eigen_tol` turned into the residual tolerance.

    Returns
    -------
    float
        ``max(DEGENERACY_TOL, MARGIN * _eigen_tol(cutoff), MARGIN * eps * max|e_ref|)``.
    """
    e = np.asarray(e_ref).real
    scale = float(np.max(np.abs(e))) if e.size else 0.0
    return max(
        DEGENERACY_TOL,
        DEGENERACY_TOL_MARGIN * _eigen_tol(slaterWeightMin),
        DEGENERACY_TOL_MARGIN * float(np.finfo(float).eps) * scale,
    )


def _degenerate_groups(e_ref, tol=DEGENERACY_TOL):
    """Partition reference-state indices into runs of (near-)degenerate eigenvalues.

    ``e_ref`` comes from the eigensolver in ascending order, so a single forward scan suffices.
    Returns a list of index lists, one per manifold.

    ``tol`` defaults to the :data:`DEGENERACY_TOL` *floor* so this stays a pure, directly
    testable function of its arguments. **Production callers must pass**
    ``tol=_degeneracy_tol(e_ref, slaterWeightMin)`` instead: the floor alone is tighter than the
    eigensolver's own accuracy whenever ``slaterWeightMin`` is loose, and then splits manifolds
    that are exactly degenerate.
    """
    e = np.asarray(e_ref).real
    groups, start = [], 0
    for i in range(1, len(e) + 1):
        if i == len(e) or abs(e[i] - e[start]) > tol:
            groups.append(list(range(start, i)))
            start = i
    return groups


def _chunk_groups(groups, chunk_size):
    """Batch ``groups`` (a list of index lists) into runs of at least ``chunk_size`` reference
    rows each, never splitting a group across a batch. ``chunk_size=None`` yields one batch
    holding every group -- today's unchunked behaviour exactly."""
    if chunk_size is None or not groups:
        return [groups] if groups else []
    batches = []
    pending: list = []
    pending_width = 0
    for g in groups:
        pending.append(g)
        pending_width += len(g)
        if pending_width >= chunk_size:
            batches.append(pending)
            pending, pending_width = [], 0
    if pending:
        batches.append(pending)
    return batches


def _score_candidates(overlaps, e_ref, e_Dj, groups, chunk_size=None):
    """Manifold-summed Epstein-Nesbet importance ``max over degenerate groups of
    (group-summed de2)``, computed in group-aligned batches over the reference (``p``) axis
    instead of materializing the whole ``(p, n_Dj)`` de2/mask stack at once.

    Exactly equal to the unchunked computation (``np.max(np.stack([...for g in groups]))``):
    each group's contribution ``sum_{i in g} |<Dj|H|psi_i>|^2 / |E_i - E_Dj|`` is independent
    of every other group, so accumulating an elementwise running max over already-processed
    groups is identical to stacking every group's sum and maxing once at the end -- the same
    "a degenerate manifold has no preferred basis, so only a quantity summed over the whole
    manifold is well defined" argument :data:`DEGENERACY_TOL` and :func:`_degenerate_groups`
    already rest on, one level up: a batch boundary is only ever placed *between* manifolds.

    Written to size the CIPSI selection round's memory peak
    (``doc/plans/dc_smo_memory.md``): the temporaries this function needs live only for one
    batch's rows (``de``, ``de2``, ``mask``, all ``(batch_width, n_Dj)``) rather than for the
    whole ``p``, at the cost of processing ``overlaps`` in ``ceil(p / chunk_size)`` passes
    instead of one. ``overlaps`` and ``e_Dj`` themselves are **not** chunked here -- they come
    from :meth:`CIPSISolver._candidate_overlaps_and_energies`'s single, unchanged diagonal-probe
    pass, which is unsound to chunk (the probe needs the full candidate support to see every
    candidate-candidate coupling).

    Parameters
    ----------
    overlaps : ndarray, complex, shape (p, n_Dj)
        ``overlaps[i, j] = <Dj_j | H | psi_i>``.
    e_ref : ndarray, shape (p,)
        Reference-state energies (ascending), the axis ``groups`` partitions.
    e_Dj : ndarray, shape (n_Dj,)
        Diagonal-probe candidate energies.
    groups : list of list of int
        Degenerate-manifold partition of ``range(p)`` (:func:`_degenerate_groups`).
    chunk_size : int, optional
        Target number of reference rows per batch (:func:`_chunk_groups`); ``None``
        processes every group in one batch -- the unchunked computation, bit-for-bit.

    Returns
    -------
    ndarray, shape (n_Dj,)
        Real, non-negative candidate importance scores.
    """
    n_Dj = overlaps.shape[1]
    scores = np.zeros(n_Dj, dtype=float)
    for batch in _chunk_groups(groups, chunk_size):
        idx = np.fromiter(itertools.chain.from_iterable(batch), dtype=np.intp)
        de = np.maximum(np.abs(e_ref[idx, None] - e_Dj[None, :]), 1e-12)
        ov = overlaps[idx]
        de2 = np.zeros(ov.shape, dtype=float)
        mask = np.abs(ov) > 1e-12
        de2[mask] = np.square(np.abs(ov[mask])) / de[mask]
        offset = 0
        for g in batch:
            width = len(g)
            np.maximum(scores, de2[offset : offset + width].sum(axis=0), out=scores)
            offset += width
    return scores


def _psi_ref_width(psi_refs):
    """Number of reference columns, whether ``psi_refs`` is a block or a legacy list.

    ``len()`` on a :class:`ManyBodyState` is its **rank-local row count**, not its width, so
    every ``len(psi_refs)`` that meant "how many reference states" has to come through here.
    Two of them size MPI collectives (``_manifold_request`` -> ``num_wanted``, and the memory
    look-ahead -> ``max_new`` -> ``_admit_top``'s rank-dependent branch), and a rank-local value
    there is the deadlock class this repo has hit twice. ``.width`` is rank-invariant by
    construction: ``build_state`` takes it from the dense array's row count, which is replicated,
    and it survives a rank owning zero determinants.
    """
    return psi_refs.width if isinstance(psi_refs, ManyBodyState) else len(psi_refs)


def _manifold_request(n_kept, prev_kept):
    """How many eigenstates one CIPSI cycle should ask the eigensolver to retain initially.

    The *request*, not the kept count, sizes the eigensolver (:func:`~impurityModel.ed.memory_estimate.gs_krylov_blocks`
    turns it into roughly ``4 * num_wanted`` Krylov columns), so it is what the last cycle kept plus room to
    grow at twice the rate it last grew at, floored at :data:`_EIGENSTATE_PAD`. An undershoot costs nothing
    but a restart: the window solve grows its retention on demand.

    ``prev_kept is None`` means *no history* and is not the same as ``0``: a first cycle has no predecessor
    to extrapolate from. The growth term is ``max(0, ...)``: a shrinking manifold gets the flat pad, never a
    negative margin.

    Parameters
    ----------
    n_kept : int
        States the previous cycle kept inside the thermal cut.
    prev_kept : int or None
        What the cycle before *that* kept, or ``None`` when there was none.

    Returns
    -------
    int
        The request: ``n_kept`` plus the larger of the flat pad and twice the last growth.
    """
    growth = 0 if prev_kept is None else max(0, n_kept - prev_kept)
    return n_kept + max(_EIGENSTATE_PAD, 2 * growth)


def _memory_growth_bound(budget_bytes, basis_size, p_now, p_next):
    """The look-ahead half of ``expand``'s memory guard, as a callable for ``determine_new_Dj``.

    A CIPSI selection round's transient memory scales with the candidate space it enumerates,
    ``Hpsi_rows x p`` -- proportional to the basis it runs on and to the reference-block width.
    Given this round's measured transient ``T`` on a basis of ``basis_size`` determinants, the
    next round on ``b_next`` determinants with ``p_next`` references is predicted to peak at
    ``rss + T * (b_next / basis_size) * (p_next / p_now)``, and the largest ``b_next`` that keeps
    that under ``budget_bytes`` bounds how many determinants this round may admit.

    Linear in the basis is conservative: the fan-out ``Hpsi_rows / basis`` falls as the basis
    grows (10.4 -> 7.4 -> 5.8 across the crashed SrMnO3 sector's last cycles). ``p_next`` is the
    request the next eigensolve will make (:func:`_manifold_request`), an upper bound on what it
    keeps. No byte model anywhere: only the round's own measured peak, so a skewed partition or
    an under-counted term in :mod:`memory_estimate` cannot fool it.

    ``budget_bytes`` is an **absolute** RSS ceiling -- the rank's whole share of node RAM, from
    :func:`memory_estimate.absolute_rss_budget` -- because ``rss_bytes`` is an absolute reading.
    Handing it the *free*-memory fraction instead is what pinned the SrMnO3 gap-DC search at its
    seed basis in every sector after the first (``doc/plans/dc_smo_memory.md``, round 9).

    Returns ``callable(transient_bytes, rss_bytes) -> (int | None, str)``: the affordable
    admission and *why* it came out that way --

    ``"unmeasured"``
        the transient could not be measured; the cap is ``None`` ("no opinion").
    ``"budget"``
        the process already holds its whole share, so headroom is non-positive and nothing is
        affordable *whatever* this round cost.
    ``"transient"``
        headroom is positive and this round's own measured cost is what bounds the next one.

    The caller needs that distinction: only a ``"transient"`` zero is a measurement of *this*
    round. A ``"budget"`` zero is one sample of a ceiling that moves with the other ranks on the
    node, so acting on it immediately lets a momentary dip in ``MemAvailable`` pin a sector's
    basis for the rest of the run -- the shape of the very bug this function was fixed for.
    """
    p_ratio = max(1.0, float(p_next) / max(1, int(p_now)))
    budget_bytes = float(budget_bytes)
    basis_size = int(basis_size)

    def affordable(transient_bytes, rss_bytes):
        if transient_bytes <= 0:
            return None, "unmeasured"
        headroom = budget_bytes - float(rss_bytes)
        if headroom <= 0.0:
            return 0, "budget"
        next_max = basis_size * headroom / (float(transient_bytes) * p_ratio)
        return max(0, int(next_max) - basis_size), "transient"

    return affordable


def _energy_cut_indices(e_ref, max_energy, tol=DEGENERACY_TOL):
    """Indices of the eigenstates within ``max_energy`` of the lowest, never bisecting a manifold.

    A degenerate manifold has no preferred basis: the eigensolver returns an arbitrary rotation of
    it. Keeping only *some* of its members therefore makes every downstream quantity built from
    that set -- the CIPSI candidate scores, the thermal average, the Green's-function seed support
    -- depend on which rotation the solver happened to land on, and hence on the MPI rank count.
    Whenever the cut would fall inside a manifold, extend it to include the whole manifold.

    Returns ``(indices, need_more_states)``. ``need_more_states`` is True when *every* computed
    eigenstate was kept: the solver never produced a state beyond the cut, so there is no evidence
    that the boundary manifold is complete (or even that the cut was reached). Only a computed
    state lying strictly outside -- and not degenerate with the last kept one -- certifies that.

    As for :func:`_degenerate_groups`, ``tol`` defaults to the :data:`DEGENERACY_TOL` floor for
    testability and production callers must pass :func:`_degeneracy_tol`; grouping below the
    achieved residual bisects the very manifolds this function exists to keep whole.
    """
    e = np.asarray(e_ref).real
    if len(e) == 0:
        return [], False
    order = np.argsort(e, kind="stable")
    e_sorted = e[order]

    n_keep = max(int(np.count_nonzero(e_sorted - e_sorted[0] <= max_energy)), 1)
    # Never split a manifold: absorb any state degenerate with the last kept one.
    original_n_keep = n_keep
    while n_keep < len(e_sorted) and abs(e_sorted[n_keep] - e_sorted[original_n_keep - 1]) <= tol:
        n_keep += 1
    return [int(i) for i in order[:n_keep]], n_keep == len(e_sorted)


def _covers(have, want):
    """Whether a solve that answered ``have`` (``Lowest``, ``Within`` or None) also answers ``want``."""
    if isinstance(have, Within):
        return isinstance(want, Lowest) or (isinstance(want, Within) and have.max_energy >= want.max_energy)
    return isinstance(have, Lowest) and isinstance(want, Lowest)


#: Rows per slice of :func:`_zero_entries_at_most`'s in-place pass: bounds its temporaries to
#: ``_PRUNE_SLICE_ROWS * p`` entries instead of three full ``(rows, p)`` float arrays.
_PRUNE_SLICE_ROWS = 65536


def _zero_entries_at_most(view, cutoff):
    """Zero, in place, every entry of the ``(rows, p)`` complex ``view`` with ``|v| <= cutoff``.

    ``std::norm(v) <= cutoff**2`` (no sqrt), matching ``ManyBodyBlockState::prune_rows``' C++
    criterion exactly -- not ``np.abs(view) <= cutoff``, which takes a sqrt first and so is not
    guaranteed bit-identical at the boundary. Slice by slice with reused buffers: a whole-block
    ``view.real**2 + view.imag**2`` allocated ~24 B per entry on top of the 16 B block, and on the
    summed selection block that is not bounded by
    ``APPLY_ROW_CHUNKS``. The caller must drop its own reference to ``view`` before
    ``prune_rows``/redistribute (they refuse to run while a buffer export is alive); this function
    keeps none.
    """
    rows = view.shape[0]
    if rows == 0:
        return
    step = min(rows, _PRUNE_SLICE_ROWS)
    norm2 = np.empty((step,) + view.shape[1:], dtype=float)
    imag2 = np.empty_like(norm2)
    c2 = cutoff * cutoff
    for lo in range(0, rows, step):
        part = view[lo : lo + step]
        n = len(part)
        np.multiply(part.real, part.real, out=norm2[:n])
        np.multiply(part.imag, part.imag, out=imag2[:n])
        norm2[:n] += imag2[:n]
        part[norm2[:n] <= c2] = 0.0


class CIPSISolver:
    def __init__(self, basis: Basis):
        self.basis = basis
        self.psi_refs = None
        #: Latest :class:`~impurityModel.ed.eigensolvers.EigenResult`, and the random-column width the next solve uses
        #: (adapted to the largest multiplicity seen).
        self.eig: EigenResult | None = None
        self.n_random = _MIN_RANDOM_COLUMNS
        #: The width of the manifold the previous window solve was warm-started from (``None``: no history),
        #: for the growth margin of the next request (:func:`_manifold_request`). ``expand`` resets it.
        self._prev_kept: int | None = None
        # Diagnostics of the latest candidate selection / basis truncation (see
        # determine_new_Dj / truncate); None until the corresponding event happens.
        self.last_selection = None
        self.last_truncation: dict | None = None
        self.truncation_report = None
        #: ``(basis, H_mat) -> H_mat``, applied to every matrix the eigensolves build (collective,
        #: like the build). ``seed_sectors.ClassPenalty`` installs one to keep an expansion in a
        #: symmetry class; ``None`` (the default) diagonalizes ``H`` itself. Selection never sees it.
        self.matrix_hook = None
        #: ``(solver, e_ref, psi_refs) -> bool``, called by :meth:`expand` after every eigensolve
        #: (collective; every rank calls it). ``True`` ends the expansion there, as converged would;
        #: rank 0's answer is broadcast, so the ranks cannot disagree. :attr:`stopped_early` records it.
        self.cycle_hook = None
        self.stopped_early = False
        #: The last Hamiltonian matrix and what it was built on (:class:`MatrixSnapshot`), kept
        #: during :meth:`expand` only; see :data:`_SNAPSHOT_BUDGET_FRACTION`.
        self._h_snapshot = None
        self._snapshot_budget = False

    def _allreduce_sum(self, value):
        if self.basis.is_distributed:
            return self.basis.comm.allreduce(value, op=MPI.SUM)
        return value

    def _allreduce_max(self, value):
        if self.basis.is_distributed:
            return self.basis.comm.allreduce(value, op=MPI.MAX)
        return value

    def truncate_initial(self, H: ManyBodyOperator, manifold, dense_cutoff=1e3) -> None:
        """Perform an initial truncation if the basis exceeds the truncation threshold.

        ``manifold`` (:class:`~impurityModel.ed.eigensolvers.Lowest` or
        :class:`~impurityModel.ed.eigensolvers.Within`) is the set of states whose support the
        truncation keeps.

        Routed through :meth:`get_eigenvectors` (TRLM above ``dense_cutoff``, dense below)
        rather than calling ``eigensystem`` directly: TRLM's collective count
        depends only on `basis`/`H_mat`, which are already rank-uniform, so it cannot desync
        (the retired multi-rank ARPACK fallback could, under multithreaded BLAS: measured -n3 hang).
        """
        if self.basis.size > self.basis.truncation_threshold and H is not None:
            if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
                print("Truncating basis!")
            # Rank over the requested manifold. A `Within` request keeps the whole near-degenerate
            # ground manifold, so the truncation does not bias the retained determinants toward one
            # member of a multiplet; a `Lowest` request keeps the support of one state, which is all
            # its caller needs. Cold start (psi_refs=None): truncate_initial runs before any converged block
            # exists, so get_eigenvectors falls back to its rank-independent hash start vector.
            _e_ref, psi_ref = self.get_eigenvectors(H, manifold, dense_cutoff=dense_cutoff, psi_refs=None)
            self.truncate(psi_ref, _e_ref)

    def truncate(self, psis, e_ref=None, target=None, slaterWeightMin=0) -> ManyBodyState:
        """Keep the globally top-``target`` determinants by eigenvector amplitude.

        Importance is the max ``|amplitude|^2`` over ``psis`` (each determinant counted
        once, on its hash-owner rank -- ``psis`` must be redistributed). The cutoff comes
        from the collective amplitude bisection, so every rank retains the identical set
        and the global count never exceeds ``target`` (ties at the cutoff are
        under-admitted). Collective on ``basis.comm``.

        ``slaterWeightMin`` is the cutoff the eigenvectors were converged at. It does not
        affect the amplitude bisection, only :func:`_degeneracy_tol` -- the importance is
        summed over degenerate manifolds (see below), so how those manifolds are delimited
        has to track the eigensolver's accuracy rather than a fixed constant.
        """
        if target is None:
            target = self.basis.truncation_threshold
        # `psis` is normally the block `get_eigenvectors` returns; a list of width-1 states is
        # still accepted, the same way `_candidate_overlaps_and_energies` takes either.
        #
        # `.copy()`, NOT the caller's object: `keep_rows` and `prune_rows` below mutate IN
        # PLACE, and `redistribute_block` returns its argument unchanged on a non-distributed
        # basis -- so binding `blk` to the argument truncates the caller's block too. `expand`
        # aliases exactly that (`best_psis = psi_refs`, then `psi_refs = self.truncate(psi_refs,
        # ...)` in the same cycle), and the best-basis snapshot silently lost every row a later
        # cycle dropped: measured -13.2657157 against -13.2657322 on the capped cubic d-shell,
        # with a 240-determinant basis where 400 was asked for. No exception, just a worse
        # answer. The list form was safe only because `from_states` always built a fresh block.
        blk = psis.copy() if isinstance(psis, ManyBodyState) else ManyBodyState.from_states(list(psis))
        if e_ref is not None and len(e_ref) == blk.width:
            # e_ref is only replicated to roundoff (it comes from a Lanczos solve run
            # independently on every rank); a splitting sitting near DEGENERACY_TOL could group
            # differently per rank and desync which determinants survive. Broadcast rank 0's copy
            # so every rank groups identically, matching the rule this file already applies to
            # `restarted_lanczos`'s own near-cut decisions above.
            if self.basis.is_distributed:
                e_ref = self.basis.comm.bcast(e_ref, root=0)
            keys = list(blk.keys())
            # A copy, not a view: np.asarray(blk) would hold a live buffer export on ``blk``
            # (ManyBodyState.__getbuffer__ increments its export count), and the keep_rows call
            # below refuses to run while any buffer view of the block is still alive.
            amps = np.array(blk)
            norms2 = np.zeros(len(keys), dtype=float)
            if amps.size > 0:
                for group in _degenerate_groups(e_ref, tol=_degeneracy_tol(e_ref, slaterWeightMin)):
                    group_amps = amps[:, group]
                    group_norms2 = np.real(group_amps * np.conj(group_amps)).sum(axis=1)
                    norms2 = np.maximum(norms2, group_norms2)
        else:
            keys, norms2 = blk.row_max_norms2()
        keep_mask = self._top_k_mask(norms2, keys, int(target))
        retained = set(itertools.compress(keys, keep_mask))
        kept_weight = self._allreduce_sum(float(norms2[keep_mask].sum()))
        total_weight = self._allreduce_sum(float(norms2.sum()))
        self.last_truncation = {
            "target": int(target),
            "retained": self._allreduce_sum(len(retained)),
            "discarded_weight": 1.0 - kept_weight / total_weight if total_weight > 0.0 else 0.0,
        }
        # Use the full container reset, not list.clear(): clearing only the local_basis
        # *list* leaves self.size and the _index_dict stale, so the subsequent
        # add_states() dedupes the (subset) trimmed states against the still-populated
        # index and repopulates nothing -- leaving size > 0 with an empty local_basis.
        # That desync later crashes build_state (IndexError) and collapses the Lanczos
        # seed block ("Block collapsed to zero rank").
        self.basis.clear()
        self.basis.add_states(retained)
        # Reuse the block already built for the norms above: dropping the non-retained
        # rows in place (one linear merge over the sorted key vectors, `keep_rows`)
        # replaces a second per-state dict-comprehension pass over every psi's own
        # flat_map, and `redistribute_block` replaces `len(psis)` separate
        # `redistribute_psis` transfers with one shared-support wire pass. This site
        # operates on the clean reference manifold (not the wider H-applied candidate
        # space `determine_new_Dj` builds), so it isn't subject to that function's
        # MIXED fill-gate verdict (Phase 6b of the state-unification refactor; see
        # doc/plans/manybodystate_block_unification.md).
        # A rank may legitimately retain zero determinants (small target, most weight
        # elsewhere): dict.fromkeys(retained, ...) is then {}, which must stay an
        # explicit width-1 empty block, not the width-0 polymorphic zero -- the latter
        # makes from_states raise on this rank only, an asymmetric exception against
        # the other ranks' populated masks that deadlocks redistribute_block below.
        mask = ManyBodyState.from_states([ManyBodyState(dict.fromkeys(retained, 1.0 + 0j), width=1)])
        blk.keep_rows(mask)
        blk = self.basis.redistribute_block(blk)
        # The list form this used to return went out through `to_states()`, which drops a
        # column's exact zeros, so a row that was zero in *every* column disappeared and the
        # caller's `from_states` union never saw it. Returning the block keeps such a row, so
        # prune them here: same support as before, and hence a bit-identical `e0`. This is the
        # same `prune_rows(0.0)` `_apply_block_and_redistribute` applies after its own
        # redistribute, for the same reason (exact cross-rank cancellation on every column).
        blk.prune_rows(0.0)
        return blk

    def _apply_block_and_redistribute(self, H, psi_ref, cutoff):
        """Apply ``H`` to the reference states as one shared-support block, then redistribute.

        Amortizes the term walk / sign / restriction-mask / hash work over
        ``len(psi_ref)`` columns in a single :meth:`ManyBodyOperator.apply_block` call,
        instead of ``len(psi_ref)`` separate per-state applies. ``apply_block`` keeps a
        row if ANY column exceeds ``cutoff`` -- a safe superset per column, but not the
        same as pruning each column to its own threshold. By default
        nothing is cut before the cross-rank sum: every
        partial is sent, and each column of the summed block is cut to ``cutoff``
        (:meth:`_prune_summed`), so the candidate set is the same whatever the rank count and
        ``APPLY_ROW_CHUNKS`` (CrI3 at a binding cap: a different basis at 1, 2 and 3 ranks
        before, identical after). ``before_sum`` restores the old order -- every column of each
        rank's partial pruned to ``cutoff`` before redistributing, bit-for-bit the old
        per-state ``applyOp(H, psi_i, cutoff)`` -- which the paragraphs below describe.

        The per-column prune runs as one vectorized pass over the block's own buffer-
        protocol view (zero-copy, in place) rather than -- as it did before -- splitting
        the block into ``p`` separate width-1 ``ManyBodyState`` objects
        (:meth:`ManyBodyState.to_states`) and reassembling them
        (:meth:`ManyBodyState.from_states`). That round trip copies the full determinant
        key *for every (row, column) pair twice* (once out, once back in): measured at
        ~104 B/pair, the dominant term in the CIPSI selection round's memory peak (see
        ``doc/plans/dc_smo_memory.md`` -- it is what the crashed SrMnO3 DC search actually
        ran out of memory in). Zeroing entries in place changes nothing ``to_states()``
        would not also have produced: a dropped entry in a per-column state and an
        explicit-zero entry in the shared block read identically to every downstream
        consumer, and ``apply_block``'s own row-survival test already guarantees a row
        that makes it this far has at least one column above ``cutoff`` -- so no row can
        come out of this all-zero (the corner case where that guarantee is looser than it
        looks costs nothing beyond an all-zero row briefly crossing the network, since
        the identical ``prune_rows(0.0)`` after the redistribute below drops it either
        way).

        Returns the merged **block** (not a list): ``_candidate_overlaps_and_energies``
        reads it via the buffer protocol (``np.asarray``) instead of iterating
        ``.items()``. ``prune_rows(0.0)`` after the redistribute is what turns an exact
        cross-rank cancellation on every column of a row into that row's removal -- a
        real selection-rule cancellation, not a truncation artifact.
        """
        block = psi_ref if isinstance(psi_ref, ManyBodyState) else ManyBodyState.from_states(psi_ref)
        n_chunks = max(config.APPLY_ROW_CHUNKS.get() or 1, 1)
        # Send every partial and cut the summed amplitude, so the candidate set does not depend on how
        # the reference rows are split over ranks and chunks.
        # Row-chunked: the round's peak is the raw apply output, its packed send buffer, the
        # receive buffer and the merged block all alive at once (~6x the owned block, measured;
        # see doc/plans/dc_smo_memory.md round 6). Applying one chunk of the reference rows at
        # a time bounds the first three to chunk size; only the accumulating merged block stays.
        # Exact up to summation order (a candidate reached from rows in different chunks has its
        # partial sums added chunk by chunk), which is the same class of difference a change of rank
        # count makes.
        #
        # The chunk COUNT is the knob, replicated on every rank, so every rank makes exactly
        # `n_chunks` collective `redistribute_block` calls whatever its row count -- a rank with
        # fewer rows than chunks sends empty chunks (an explicit width-p block with no rows, which
        # `apply_block` and the packer both accept; never the width-0 polymorphic zero).
        n_rows = len(block)
        bounds = np.linspace(0, n_rows, int(n_chunks) + 1).astype(int)
        merged = None
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            # `row_slice` allocates exactly this chunk's rows. The earlier spelling
            # (`block.copy()` then `keep_rows(mask)`) allocated a FULL-SIZE duplicate of the
            # reference block per chunk -- `keep_rows` shrinks the logical length but not the
            # vector's capacity -- so every chunk's "bounded" apply ran alongside a full-size
            # copy, and the mask cost one Python key object per row of the whole block.
            part = block.row_slice(int(lo), int(hi))
            raw = H.apply_block(part, 0.0)
            del part
            piece = self.basis.redistribute_block(raw)
            del raw
            if merged is None:
                merged = piece
            elif len(piece):
                # Rank-local decision on a rank-local quantity (no collective inside), so an
                # empty piece on one rank cannot desynchronize the others.
                merged += piece
            del piece
        return self._prune_summed(merged, cutoff)

    @staticmethod
    def _prune_summed(merged, cutoff):
        """Zero every summed entry with ``|v| <= cutoff`` in place, then drop the all-zero rows.

        ``cutoff = 0`` is the bare ``prune_rows(0.0)``: a row that cancels exactly across ranks on
        every column is a real selection-rule cancellation, not a truncation artifact."""
        if cutoff > 0.0 and len(merged):
            _zero_entries_at_most(np.asarray(merged), cutoff)
        merged.prune_rows(0.0)
        return merged

    def _candidate_overlaps_and_energies(self, H, Hpsi_ref, slaterWeightMin: float = 0):
        """Enumerate the out-of-basis candidates of ``Hpsi_ref`` with couplings and energies.

        The candidate machinery under :func:`_score_candidates`'s Epstein-Nesbet score (the
        importance denominator is applied on top): this rank's hash-owned candidate
        determinants ``local_Djs`` (sorted, so rank-independent given ``Hpsi_ref`` is
        redistributed), the coupling matrix ``overlaps[i, j] = <Dj | H | psi_i>`` read off
        ``Hpsi_ref``, and the candidate diagonal energies ``e_Dj[j] = <Dj|H|Dj>``.

        ``Hpsi_ref`` is normally the block :meth:`_apply_block_and_redistribute` returns
        (already redistributed and pruned to its own support); a bare list of width-1
        states is also accepted (wrapped via ``from_states``) for direct callers. Reading
        the block via the buffer protocol (``np.asarray``) instead of ``.items()`` avoids
        allocating a ``SlaterDeterminant`` and a ``Row`` object per row per column -- the
        dominant Python-level cost of a selection round (measured: 33-43% of the round
        across two problem sizes, `doc/plans/manybodystate_block_unification.md`'s
        Phase 9 write-up).

        Rank-local: nothing here communicates. It used to, because ``e_Dj`` was estimated
        by applying ``H`` to one random-phase superposition of every candidate and reading
        back the diagonal, which made each rank's estimate depend on the partition and so
        needed a ``redistribute_psis`` to restore rank-independence. That probe was also an
        *approximation* -- the superposition contributes ``Re(sum_{k != j} conj(p_j) p_k
        <Dj|H|Dk>)`` on top of the wanted element, which the random phases scatter the sign
        of but do not remove. Measured on the SrMnO3 workload it reached 63% of the
        Epstein-Nesbet denominator, and only 67% of the top-1% admitted set agreed with the
        exact ranking (``doc/plans/dc_smo_memory.md`` round 9). Epstein-Nesbet needs the
        diagonal and nothing else, so :meth:`ManyBodyOperator.diagonal` now computes it
        exactly, for a tenth of the memory and no collective.
        """
        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        _contains = self.basis.contains_local
        blk = Hpsi_ref if isinstance(Hpsi_ref, ManyBodyState) else ManyBodyState.from_states(Hpsi_ref)

        # `keys()` returns the shared support in row (sorted) order -- the same order the
        # old `sorted({state for hp in Hpsi_ref for state in hp ...})` produced over the
        # per-state dicts, so no re-sort is needed here (verified by an explicit
        # `keys() == sorted(...)` assertion in the block/list equivalence test rather than
        # taken on faith -- SlaterDeterminant's `__lt__` and the C++ key-vector ordering
        # are the same comparator, but that equivalence is exactly the kind of thing this
        # campaign has been bitten by before).
        keys = blk.keys()
        amps = np.asarray(blk)  # (rows, p) zero-copy buffer-protocol view
        new_mask = np.fromiter((not _contains(k) for k in keys), dtype=bool, count=len(keys))
        local_Djs = list(itertools.compress(keys, new_mask))
        overlaps = np.ascontiguousarray(amps[new_mask].T)  # (p, n_Dj); boolean indexing copies
        del amps  # release the buffer export before any later mutation of Hpsi_ref

        # The candidate diagonals <Dj|H|Dj>, evaluated directly. `diagonal` reads only the
        # block's keys, so this needs no state built for the purpose: it runs over the whole
        # `Hpsi_ref` support (a few percent more determinants than there are candidates) and
        # the mask selects the candidates out. Purely rank-local -- a determinant's diagonal
        # element depends on nothing but the determinant.
        e_Dj = np.real(H.diagonal(blk))[new_mask]
        return local_Djs, overlaps, e_Dj

    def _top_k_mask(self, scores, keys, k):
        """Local mask of the global top ``k`` of ``scores`` (``keys``: this rank's determinants, aligned).

        :func:`~impurityModel.ed.manybody_basis.collective_top_k_mask` on the basis's communicator:
        exactly ``k`` admitted (fewer only when fewer scores are positive), the same set whatever the
        rank count. This replaced a bisection that left near-tie groups out (under-filling the cap)
        with an all-tied fallback that picked group members by exact raw equality, i.e. by rounding.
        Collective on ``basis.comm``."""
        return collective_top_k_mask(scores, keys, k, self.basis.comm if self.basis.is_distributed else None)

    def _admit_top(self, scores, mask, max_new, keys):
        """Cap an importance-masked candidate set at the globally top ``max_new`` scores.

        ``mask`` is the rank-local boolean pre-selection (e.g. ``scores >= de2_min``);
        ``max_new=None`` admits it unchanged. Otherwise the global top ``max_new`` are admitted
        by :meth:`_top_k_mask` (``keys``: the candidates, aligned with ``scores``), so every rank
        admits its share of the identical set. Collective on ``basis.comm``; returns
        ``(admitted_mask, stats)`` with the ``last_selection``-shaped stats dict.

        ``stats["residual_pt2"]`` is the importance of every candidate *not* admitted, for any
        reason; see :meth:`expand`'s ``e_pt2_tol``.

        ``stats["subthreshold_de2_mass"]`` is the PT2 importance of candidates that never
        passed ``mask`` (typically ``scores >= de2_min``) at all -- unconditional, unlike
        ``discarded_de2_mass`` below (which is only populated once a cap actually binds).
        It is the error bound a *de2_min* choice spends, as opposed to the error bound a
        *cap* spends: the two are compared side by side when calibrating ``de2_min``
        against the DC search's own answer (see ``doc/plans/dc_smo_memory.md``).
        """
        n_candidates = self._allreduce_sum(int(np.count_nonzero(mask)))
        subthreshold_de2_mass = self._allreduce_sum(float(scores[~mask].sum()))
        discarded_de2_mass = 0.0
        if max_new is not None and n_candidates > max_new:
            if max_new <= 0:
                # An exhausted budget: admit nothing (callers still want the selection
                # stats, e.g. the boundary residual). The tie fallback below must not
                # run -- it exists to avoid *under*-admission, not to override a zero cap.
                discarded_de2_mass = self._allreduce_sum(float(scores[mask].sum()))
                mask = np.zeros_like(mask)
            else:
                admitted = np.zeros_like(mask)
                admitted[mask] = self._top_k_mask(scores[mask], list(itertools.compress(keys, mask)), int(max_new))
                discarded_de2_mass = self._allreduce_sum(float(scores[mask & ~admitted].sum()))
                mask = admitted
        stats = {
            "n_candidates": n_candidates,
            "n_admitted": self._allreduce_sum(int(np.count_nonzero(mask))),
            "discarded_de2_mass": discarded_de2_mass,
            "subthreshold_de2_mass": subthreshold_de2_mass,
            # Everything this round leaves out, whichever limit left it out: the pre-selection
            # (`de2_min` floor, PT2 tail cutoff) and the cap alike. Once a round admits nothing it
            # is the whole Epstein-Nesbet PT2 correction of the reference states -- the estimate of
            # how far their energies are from converged.
            "residual_pt2": self._allreduce_sum(float(scores[~mask].sum())),
        }
        return mask, stats

    def determine_new_Dj(
        self,
        e_ref,
        psi_ref,
        H,
        de2_min,
        slater_cutoff=0,
        max_new=None,
        affordable_growth=None,
        e_pt2_tol=None,
        clamp_admission=True,
    ):
        """Select the candidate determinants to add to the basis.

        Candidates connected to ``psi_ref`` through ``H`` are kept when their de2
        importance (max over the reference states) reaches ``de2_min`` and, with
        ``e_pt2_tol`` set, lies above the tail cutoff: the candidates left out then sum to at
        most ``e_pt2_tol`` (:func:`collective_mass_cutoff`). ``de2_min`` bounds each discarded
        term, ``e_pt2_tol`` their sum -- which is what bounds the energy error. ``max_new``
        optionally caps the *global* number of selected candidates: the top ``max_new``
        by de2 importance are kept (collective bisection cutoff, ties under-admitted)
        and ``self.last_selection`` records
        ``{"n_candidates", "n_admitted", "discarded_de2_mass", "subthreshold_de2_mass",
        "residual_pt2", "n_references", "hpsi_rows"}``. Collective on ``basis.comm``.

        ``affordable_growth``, optional
            ``callable(transient_bytes, rss_bytes) -> int | None``: a second, *memory-derived*
            cap on the admitted count, evaluated after the candidate space has been built and
            scored but before anything is admitted. ``transient_bytes`` is this round's own
            memory peak above the RSS it started from and ``rss_bytes`` the RSS it ends at, both
            MAX over ranks; the callable returns how many new determinants the *next* round can
            afford (``None`` for "no opinion", e.g. the peak could not be measured). The result
            is combined with ``max_new`` by ``min``. Every rank must pass the same callable (or
            ``None``): the measurement is collective. The selection stats then also carry
            ``"round_transient_bytes"``, ``"round_rss_bytes"`` and ``"memory_admit_cap"``.

            This is the look-ahead half of ``expand``'s memory guard: the round that would not
            fit is the *next* one, on the basis this admission creates, and its cost scales with
            that basis. A guard that only compares the high-water mark *after* a round can be
            overrun in one step by an expansion growing 5-10x per cycle -- which is what killed
            the SrMnO3 double-counting search at 3.6M determinants, one cycle after reading
            2.4 GiB against a 2.5 GiB budget (``doc/plans/dc_smo_memory.md``, round 6).

        ``clamp_admission``
            ``False`` measures the round and records the memory stats exactly as above, but does
            not let the result cap the admission: ``expand``'s warn-only mode, for a cap the user
            set. Must be replicated across ranks, like ``affordable_growth``.
        """
        measure = affordable_growth is not None
        if measure:
            rss_start = current_rss_bytes()
            peak_is_own = reset_peak_rss()
        Hpsi_ref = self._apply_block_and_redistribute(H, psi_ref, slater_cutoff)
        # Global row count of the shared support H|psi_ref> spans (before the not-in-basis
        # filter): the quantity `_apply_block_and_redistribute`'s per-round
        # transient scale with (see doc/plans/dc_smo_memory.md) -- recorded here, once, rather
        # than re-derived by every caller that wants the per-cycle memory/timing picture.
        hpsi_rows = self._allreduce_sum(len(Hpsi_ref))
        local_Djs, overlaps, e_Dj = self._candidate_overlaps_and_energies(H, Hpsi_ref, 0)
        # Importance = max over reference *manifolds* of the manifold-summed de2, not max over
        # individual reference states. Within a degenerate manifold the eigensolver returns an
        # arbitrary basis (all rotations share the same residual), and `max_i |<Dj|H|psi_i>|^2`
        # moves with that rotation -- measured: 2% between a serial and a 2-rank run, enough to
        # flip a candidate across `de2_min` and, through the resulting cascade, change the basis
        # by 5% and the Green's function by 1e-7. The manifold sum
        # `Sum_{i in block} |<Dj|H|psi_i>|^2` is the squared norm of the projection of H|Dj> onto
        # that eigenspace and is rotation invariant; the shared `|e_i - e_Dj|` denominator makes
        # summing de2 directly equivalent. Reduces to the old max when the spectrum is
        # non-degenerate.
        if len(local_Djs):
            groups = _degenerate_groups(e_ref, tol=_degeneracy_tol(e_ref, slater_cutoff))
            scores = _score_candidates(overlaps, e_ref, e_Dj, groups, chunk_size=config.GS_SELECTION_CHUNK.get())
        else:
            scores = np.zeros(0)
        memory_stats = {}
        if measure:
            # The apply, the redistribution and the overlaps above are where this round peaks
            # (measured: ~6x the owned candidate block, doc/plans/dc_smo_memory.md round 6);
            # `_admit_top` below adds only per-candidate scalars. Sampled per rank, then reduced
            # unconditionally -- the callable is replicated, so every rank takes this branch.
            transient = (peak_rss_bytes() - rss_start) if peak_is_own else -1
            transient = self._allreduce_max(int(transient))
            # The baseline is the RSS the round *started* from, not the RSS now: at this point
            # the candidate space (`Hpsi_ref`, `local_Djs`, `overlaps`, `scores`) is still
            # resident, so "now" already contains most of the transient and would count it
            # twice in the prediction `baseline + transient * growth`. Measured on SrMnO3 at
            # 2 ranks: 953 MiB "now" against a ~600 MiB start, under a 1.2 GiB transient.
            rss_base = self._allreduce_max(int(rss_start))
            cap, cap_reason = affordable_growth(transient, rss_base) if transient >= 0 else (None, "unmeasured")
            if cap is not None:
                cap = int(cap)
                if clamp_admission:
                    max_new = cap if max_new is None else min(int(max_new), cap)
            memory_stats = {
                "round_transient_bytes": int(transient),
                "round_rss_bytes": int(rss_base),
                "memory_admit_cap": cap,
                "memory_cap_reason": cap_reason,
            }
        # `scores > 0` as well: a zero-importance candidate (coupling pruned below 1e-12) never
        # carries energy, and with the floor at 0 `>=` alone would admit every one of them --
        # measured: an expansion that no longer terminates on a 2,598-determinant SIAM.
        preselect = (scores >= de2_min) & (scores > 0.0)
        if e_pt2_tol is not None:
            # Collective, and reached on every rank: `e_pt2_tol` is a replicated argument.
            comm = self.basis.comm if self.basis.is_distributed else None
            preselect &= scores > collective_mass_cutoff(scores, float(e_pt2_tol), comm)
        de2_mask, selection_stats = self._admit_top(scores, preselect, max_new, local_Djs)
        selection_stats["hpsi_rows"] = hpsi_rows
        # The reference states this round scored, i.e. the states `residual_pt2` covers. Not
        # `psi_refs` after `expand` returns: a capped run exits at the loop head, after an
        # eigensolve that may have widened the block past what was scored.
        selection_stats["n_references"] = _psi_ref_width(psi_ref)
        selection_stats.update(memory_stats)
        self.last_selection = selection_stats
        new_Dj = set(itertools.compress(local_Djs, de2_mask))

        return new_Dj

    def expand(
        self,
        H,
        manifold,
        de2_min=0.0,
        dense_cutoff=1e3,
        slaterWeightMin=0,
        cap_e_tol=1e-8,
        max_cap_cycles=10,
        memory_budget_bytes=None,
        e_pt2_tol=DEFAULT_E_PT2_TOL,
        memory_policy="tighten",
    ):
        """Expand the basis variationally (CIPSI) until it stops growing.

        ``manifold`` is what the expansion refines the basis for: the reference states of every
        cycle are :class:`~impurityModel.ed.eigensolvers.Lowest` (one state) or
        :class:`~impurityModel.ed.eigensolvers.Within` (every state inside the window, whose
        low-lying members then get their own determinants selected). The window is the request's,
        never read from ``basis.tau``.

        ``e_pt2_tol`` is the convergence criterion: each selection round admits candidates until
        the ones it leaves out carry at most ``e_pt2_tol`` of Epstein-Nesbet PT2 energy
        (:func:`collective_mass_cutoff`), so the expansion stops -- a round admits nothing -- only
        once the whole residual PT2 correction of the reference states is within ``e_pt2_tol``.
        ``de2_min`` is an optional per-determinant floor on top of it; it bounds each discarded
        term, not their sum, and is *not* an energy tolerance: at ``de2_min=1e-8`` alone a SIAM
        with a few hundred determinants stops 4.3e-7 above its exact ground state (the residual
        PT2 predicted it to within 2%). ``e_pt2_tol=None`` restores the floor-only selection.
        ``convergence_report`` records ``{"residual_pt2", "e_pt2_tol", "converged",
        "limited_by"}``, and an unconverged expansion (a cap, the memory guard or the floor
        stopped it first) prints a warning with the residual as its error bar.

        The residual is summed over the reference *manifolds* (``_score_candidates``), so it
        covers every reference state the expansion carries -- conservatively, by up to a
        manifold's degeneracy. It is a second-order *estimate* of the energy error, not a
        rigorous bound (measured to within 2% on the SIAM above), and states found only after
        the expansion (e.g. a later ``num_wanted`` widening) were never references.

        With a finite ``basis.truncation_threshold`` the expansion becomes a
        **fixed-budget CIPSI** once the cap binds: each cycle prunes the currently
        least important determinants (by eigenvector amplitude, collective top-K),
        admits the best de2-ranked candidates into the freed room, and
        re-diagonalizes; cycles stop when the ground-state energy changes by less
        than ``cap_e_tol`` or after ``max_cap_cycles`` cycles. ``truncation_report``
        records whether (and how) the cap bound the expansion.

        ``memory_budget_bytes``, optional
            Per-rank byte budget (e.g. ``memory_estimate.available_bytes_per_rank(comm) *
            memory_estimate.DEFAULT_MEMORY_SAFETY``). ``None`` (the default) disables both
            guards below entirely and the expansion never samples its memory.

            **Look-ahead bound** (:func:`_memory_growth_bound`, evaluated inside
            :meth:`determine_new_Dj`): every selection round measures its own transient -- the
            process high-water mark is reset before the round (``memory_estimate.reset_peak_rss``)
            -- and this cycle's admission is capped so that the *next* round, on the basis it
            creates and with the reference-block width the next eigensolve will request, is
            predicted to fit the budget. When that bound decides the admission, the affordable
            size becomes the fixed budget (``truncation_threshold``, tightened only) and the
            fixed-budget machinery above takes over; ``truncation_report["memory_bound"]``
            records it. This is what an expansion that admits everything needs: its basis grows
            5-10x per cycle and the selection round's memory follows ``Hpsi_rows x p``, so a guard
            that only looks at the mark *after* a round is overrun in a single step -- the SrMnO3
            double-counting search read 2.4 GiB against a 2.5 GiB budget and was OOM-killed at
            5.8 GiB one cycle later (``doc/plans/dc_smo_memory.md``, round 6).

            **After-the-fact trip-wire** (the backstop): the first cycle whose measured peak RSS
            (MAX over ranks, already sampled for the diagnostic log) reaches the budget adopts a
            fixed-budget cap at the *current* basis size and warns. It fires whatever the cap --
            a finite cap far beyond what memory allows is exactly the configuration that crashed
            -- and only ever tightens one.

            Both are empirical: they react to what the process actually allocated, so a skewed
            hash partition or an under-counted term in :mod:`memory_estimate` cannot fool them.
            Wired at both production call sites through ``groundstate.expand_memory_budget``.

        ``memory_policy``, ``"tighten"`` (default) or ``"warn"``
            What the two guards above may do. ``"tighten"`` is the behaviour described there, for
            a cap derived from memory or no cap at all. ``"warn"`` is for a cap the **user** set,
            which is final: every round is still measured, but neither guard limits an admission
            or lowers the cap; the first time the budget is exceeded a warning is printed on rank
            0 (stdout and stderr, whatever the verbosity) and ``memory_warning`` records it. Must
            be replicated across ranks.
        """
        if memory_policy not in ("tighten", "warn"):
            raise ValueError(f"memory_policy must be 'tighten' or 'warn', got {memory_policy!r}")
        if isinstance(H, dict):
            H = ManyBodyOperator(H)
        # Unconditional (None clears): a mask left on H by an earlier Green's-function stage must
        # not survive into this solve (review ledger C3).
        H.set_restrictions(self.basis.restrictions)
        H.set_weighted_restrictions(self.basis.weighted_restrictions)
        psi_refs = getattr(self, "psi_refs", None)

        guard = MemoryGuard(self.basis, memory_budget_bytes, tighten=memory_policy == "tighten")
        self.stopped_early = False
        # Matrix snapshots are kept only while this call runs (see `_build_h_matrix`).
        self._snapshot_budget = memory_budget_bytes
        self._h_snapshot = None
        refiner = CapRefiner(cap_e_tol, max_cap_cycles)
        measuring = memory_budget_bytes is not None
        cap_cycles = 0
        e0 = np.inf
        self.truncation_report = None
        self.convergence_report = None
        self.memory_warning = None
        # Per-call: a `last_selection` left over from an earlier `expand` must not be read as this call's.
        self.last_selection = None
        # Whether `last_selection["residual_pt2"]` describes the basis as it now stands: true only after a
        # round that neither grew the basis nor was followed by a truncation (a capped cycle scores
        # candidates, then `truncate` drops determinants, so its residual belongs to a basis that is gone).
        residual_current = False
        cycle = 0
        # The kept manifold one cycle back, for the request's growth margin; `None` (no history) is not 0.
        self._prev_kept = None
        lowest = isinstance(manifold, Lowest)
        while True:
            # The eigensolve belongs to the cycle the guard measures: reset the high-water mark here and
            # sample right after, since the selection round resets it again for its own transient.
            if measuring:
                reset_peak_rss()
            e_ref, psi_refs = self.get_eigenvectors(
                H,
                manifold,
                dense_cutoff=dense_cutoff,
                slaterWeightMin=slaterWeightMin,
                psi_refs=psi_refs,
            )
            eigensolve_peak = peak_rss_bytes() if measuring else 0
            if len(e_ref) == 0:
                break
            e0 = float(np.min(e_ref))
            if cap_cycles > 0 and refiner.observe(self.basis, e0, psi_refs, e_ref, cap_cycles):
                break
            if self.cycle_hook is not None:
                stop = bool(self.cycle_hook(self, e_ref, psi_refs))
                if self.basis.is_distributed:
                    stop = self.basis.comm.bcast(stop, root=0)
                if stop:
                    self.stopped_early = True
                    break

            admit_target = guard.admission_target()
            old_size = self.basis.size
            # Look-ahead: bound this cycle's admission by what the *next* selection round can afford,
            # predicted from this round's own measured transient.
            affordable_growth = None
            if measuring:
                p_now = _psi_ref_width(psi_refs)
                affordable_growth = _memory_growth_bound(
                    memory_budget_bytes, old_size, p_now, 1 if lowest else _manifold_request(p_now, self._prev_kept)
                )
            with _trace_timed("cipsi_selection", cycle=cycle) as _sel_event:
                new_Dj = self.determine_new_Dj(
                    e_ref,
                    psi_refs,
                    H,
                    de2_min,
                    slater_cutoff=slaterWeightMin,
                    max_new=admit_target,
                    affordable_growth=affordable_growth,
                    e_pt2_tol=e_pt2_tol,
                    clamp_admission=guard.tighten,
                )
                n_new = self._allreduce_sum(len(new_Dj))
                sel = self.last_selection or {}
                guard.after_selection(sel, old_size, admit_target)
                _sel_event.update(
                    basis_size=int(old_size),
                    p=_psi_ref_width(psi_refs),
                    hpsi_rows=int(sel.get("hpsi_rows", 0)),
                    n_candidates=int(sel.get("n_candidates", 0)),
                    n_admitted=int(sel.get("n_admitted", 0)),
                    n_new=int(n_new),
                    subthreshold_de2_mass=float(sel.get("subthreshold_de2_mass", 0.0)),
                    residual_pt2=float(sel.get("residual_pt2", 0.0)),
                )
            peak_rss, anon_rss, shmem_rss, local_min, local_max = self._cycle_memory(eigensolve_peak, cycle)
            if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
                print(
                    f"  cycle {cycle}: basis={old_size:,} p={_psi_ref_width(psi_refs):,} "
                    f"Hpsi_rows={sel.get('hpsi_rows', 0):,} candidates={sel.get('n_candidates', 0):,} "
                    f"admitted={sel.get('n_admitted', 0):,} new={n_new:,} "
                    f"subthreshold_de2_mass={sel.get('subthreshold_de2_mass', 0.0):.3e} "
                    f"residual_pt2={sel.get('residual_pt2', 0.0):.3e} "
                    f"local[min,max]=[{local_min:,},{local_max:,}] VmHWM={format_bytes(peak_rss)} "
                    f"(anon={format_bytes(anon_rss)} shm={format_bytes(shmem_rss)})",
                    flush=True,
                )
            cycle += 1
            guard.after_cycle(peak_rss, anon_rss, shmem_rss)

            truncated = False
            if guard.capped and self.basis.size + n_new > guard.threshold:
                # Fixed-budget cycle: make room by dropping the currently least important determinants
                # (by eigenvector amplitude), then admit the de2-ranked candidates; the loop head
                # re-diagonalizes and the cycle repeats until the energy stabilizes.
                cap_cycles += 1
                truncated = True
                keep = max(int(guard.threshold) - n_new, int(guard.threshold) // 2, 1)
                psi_refs = self.truncate(psi_refs, e_ref, target=keep, slaterWeightMin=slaterWeightMin)
                if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
                    print(
                        f"------> Basis truncated! (cycle {cap_cycles}: kept "
                        f"{self.last_truncation['retained']:,} determinants, admitting {n_new:,} candidates)"
                    )
            self.basis.add_states(new_Dj)
            # `redistribute_block`, not `redistribute_psis(*psi_refs)`: unpacking a block with `*` iterates
            # its determinant KEYS (an error at >1 rank, silently wrong at 1). `prune_rows(0.0)` drops a row
            # that cancelled exactly across ranks on every column.
            psi_refs = self.basis.redistribute_block(psi_refs)
            psi_refs.prune_rows(0.0)
            residual_current = not truncated and self.basis.size == old_size
            if cap_cycles == 0 and self.basis.size == old_size:
                break
            e0 = np.inf  # the basis was mutated; e0 no longer describes it

        restored = refiner.restore_if_worse(self.basis, e0)
        if restored is not None:
            psi_refs, e_ref = restored
            residual_current = False
        self.psi_refs = psi_refs
        self.memory_warning = guard.warning
        if guard.capped and self.basis.size > guard.threshold and self.psi_refs is not None:
            # Enforce the hard threshold on exit. `e_ref` must describe the *current* `psi_refs`: after a
            # restore that is the best cycle's, not the last one's.
            self.psi_refs = self.truncate(self.psi_refs, e_ref, slaterWeightMin=slaterWeightMin)
            residual_current = False
        if cap_cycles > 0 or guard.limited:
            self._record_truncation(cap_cycles, guard)

        self._report_convergence(e_pt2_tol, de2_min, cap_cycles > 0, guard.limited, residual_current)
        self._h_snapshot = None
        self._snapshot_budget = False

        if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
            print(f"After expansion, the basis contains {self.basis.size} elements.", flush=True)

    def _cycle_memory(self, eigensolve_peak, cycle):
        """``(peak_rss, anon_rss, shmem_rss, local_min, local_max)`` for the cycle just run, and its trace note.

        Unconditional collectives (never gated on rank-local state such as ``verbose``): single-scalar
        reductions against a selection round that costs seconds to minutes. ``VmHWM`` alone cannot be
        attributed -- under an MPI that uses shared memory the ``shmem`` part is a node-wide pool every rank
        counts -- so ``anon``, the part a code change can move, is reported beside it.
        """
        local_count = len(self.basis.local_basis)
        peak_rss = max(peak_rss_bytes(), eigensolve_peak)
        rss_parts = rss_breakdown()
        anon_rss, shmem_rss = rss_parts["anon"], rss_parts["shmem"]
        if self.basis.is_distributed:
            comm = self.basis.comm
            local_max = comm.allreduce(local_count, op=MPI.MAX)
            local_min = comm.allreduce(local_count, op=MPI.MIN)
            peak_rss = comm.allreduce(peak_rss, op=MPI.MAX)
            anon_rss = comm.allreduce(anon_rss, op=MPI.MAX)
            shmem_rss = comm.allreduce(shmem_rss, op=MPI.MAX)
        else:
            local_max = local_min = local_count
        _trace_note(
            "cipsi_cycle",
            cycle=cycle,
            local_max=int(local_max),
            local_min=int(local_min),
            vm_hwm_bytes=int(peak_rss),
            anon_rss_bytes=int(anon_rss),
            shmem_rss_bytes=int(shmem_rss),
        )
        return peak_rss, anon_rss, shmem_rss, local_min, local_max

    def _record_truncation(self, cap_cycles, guard):
        """Write ``truncation_report`` (and warn): a cap or the memory guard limited this expansion.

        Written when a refinement cycle ran or memory limited the expansion even without one, so a
        memory-limited expansion never reads as converged (``dc_search.resolve_cap_at_max`` takes
        ``truncation_report is None`` to mean "unbound").
        """
        sel = self.last_selection or {}
        threshold = guard.threshold
        self.truncation_report = {
            "cap_hit": True,
            "cycles": cap_cycles,
            "retained": int(self.basis.size),
            # Still `inf` when only an observed round limited the expansion; report where it stopped.
            "threshold": int(threshold) if np.isfinite(threshold) else int(self.basis.size),
            "discarded_de2_mass": float(sel.get("discarded_de2_mass", 0.0)),
            "n_candidates_last": int(sel.get("n_candidates", 0)),
            "memory_bound": bool(guard.limited),
            "residual_pt2": float(sel.get("residual_pt2", 0.0)),
        }
        rank = self.basis.comm.rank if self.basis.is_distributed else 0
        rep = self.truncation_report
        # discarded_de2_mass bounds the ground-state error, so this is said at any verbosity: once per
        # calculation (a double-counting search runs dozens of expansions), every time at -vv.
        why = "the memory guard" if rep["memory_bound"] else f"the cap of {rep['threshold']:,}"
        emit_memory_warning(
            f"WARNING determinant cap: GS basis stopped at {rep['retained']:,} determinants "
            f"({why}); discarded candidates carry {rep['discarded_de2_mass']:.3e} of PT2 importance.",
            root=rank == 0,
            kind="gs-cap",
            force=self.basis.verbose,
        )
        if rank == 0 and self.basis.verbose:
            print(
                f"  ({rep['cycles']} refinement cycle(s); {rep['n_candidates_last']:,} "
                "candidates considered in the last cycle)",
                flush=True,
            )

    def _report_convergence(self, e_pt2_tol, de2_min, cap_bound, memory_bound, residual_current):
        """Record (and warn about) how far the expansion is from PT2 convergence.

        ``residual_pt2`` is the last selection round's. When that round left the basis as it is
        (``residual_current``) it is exactly the PT2 correction of the final references. Otherwise
        -- a capped cycle truncated after scoring, or the best capped basis was restored -- it
        describes an earlier basis, ``residual_is_current`` is False, and the expansion is not
        reported converged whatever the number says. ``None`` when no round ran at all.
        Every input is replicated (allreduced selection stats, replicated arguments and flags),
        so every rank records the same report; only the print is rank-0.
        """
        selection = self.last_selection
        residual = None if selection is None else float(selection.get("residual_pt2", 0.0))
        converged = None if e_pt2_tol is None or residual is None else residual_current and residual <= e_pt2_tol
        limited_by = None
        if converged is False:
            if memory_bound:
                limited_by = "memory"
            elif cap_bound:
                limited_by = "cap"
            elif de2_min > 0:
                limited_by = "de2_min"
        self.convergence_report = {
            "residual_pt2": residual,
            "residual_is_current": bool(residual_current),
            "e_pt2_tol": e_pt2_tol,
            "converged": converged,
            "limited_by": limited_by,
        }
        rank = self.basis.comm.rank if self.basis.is_distributed else 0
        if converged is False and rank == 0:
            reason = {
                "memory": "the memory guard held the basis",
                "cap": "the truncation_threshold cap held the basis",
                "de2_min": f"the de2_min={de2_min:.1e} floor refused the remaining candidates",
            }.get(limited_by, "the expansion stopped")
            when = "" if residual_current else " (measured before the last truncation; the kept basis may carry more)"
            print(
                f"WARNING: CIPSI ground state not converged: {reason} with a residual PT2 energy of "
                f"{residual:.3e}{when} against e_pt2_tol={e_pt2_tol:.1e}. The reference energies are "
                f"variational upper bounds; the residual estimates how far above the converged "
                f"energies they lie.",
                flush=True,
            )

    def _build_h_matrix(self, H):
        """``H`` as a ``DistributedSparseMatrix`` on the current basis (collective).

        Inside :meth:`expand` the previous cycle's matrix is grown to the current basis rather than
        rebuilt, when it was of the same operator under the same window and the basis only gained
        determinants since (:func:`basis_transcription.extend_sparse_matrix`). The operator object
        itself is held by the snapshot, so an identity check cannot be fooled by a recycled ``id``.
        """
        key = (H, repr(self.basis.restrictions), repr(self.basis.weighted_restrictions))
        snapshot, self._h_snapshot = self._h_snapshot, None
        H_mat = None
        reuse = snapshot is not None and snapshot.key[0] is H and snapshot.key[1:] == key[1:]
        if self.basis.is_distributed:
            reuse = self.basis.comm.allreduce(reuse, op=MPI.LAND)
        if reuse:
            H_mat = extend_sparse_matrix(snapshot, self.basis, H)
        del snapshot
        if H_mat is None:
            H_mat = build_sparse_matrix(self.basis, H)
        if self._snapshot_budget is not False:
            keep = self._snapshot_budget is None or H_mat.nbytes <= _SNAPSHOT_BUDGET_FRACTION * self._snapshot_budget
            if self.basis.is_distributed:
                keep = self.basis.comm.allreduce(keep, op=MPI.LAND)
            if keep:
                self._h_snapshot = MatrixSnapshot(self.basis, H_mat, key)
        if self.matrix_hook is not None:
            H_mat = self.matrix_hook(self.basis, H_mat)
        _trace_note("h_matrix_nnz", nnz=int(H_mat.nnz), n=int(H_mat.shape[0]))
        return H_mat

    def _warm_columns(self, psi_refs, width=None):
        """The lowest ``GS_MAX_BLOCK_WIDTH`` (or ``width``) columns of ``psi_refs``, distributed ``(n_local, w)``.

        ``psi_refs`` is energy-ordered ascending, so the lowest states are kept: the ones a restart is
        most likely to need again. Empty (zero columns) when there is no warm block.
        """
        if psi_refs is None:
            return np.zeros((len(self.basis.local_basis), 0), dtype=complex)
        cap = config.GS_MAX_BLOCK_WIDTH.get() if width is None else width
        if isinstance(psi_refs, ManyBodyState):
            block = psi_refs if cap is None or psi_refs.width <= cap else psi_refs.select(list(range(cap)))
            states = block.to_states()
        else:
            states = list(psi_refs)[:cap] if cap is not None else list(psi_refs)
        if not states:
            return np.zeros((len(self.basis.local_basis), 0), dtype=complex)
        return np.ascontiguousarray(build_distributed_vector(self.basis, states).T)

    def _eig_scale(self):
        """Energies whose magnitude sets the degeneracy tolerance's roundoff term (the last result's)."""
        return self.eig.e if self.eig is not None and len(self.eig.e) else np.zeros(1)

    def get_eigenvectors(
        self,
        H,
        manifold,
        dense_cutoff=1e3,
        slaterWeightMin=0,
        psi_refs=None,
        h_matrix=None,
        reuse_last=False,
    ):
        """The requested manifold of the lowest spectrum: ``(e_ref, psi_refs)``.

        ``manifold`` is :class:`~impurityModel.ed.eigensolvers.Lowest` (one state, the ground
        level's energy; no claim about its degeneracy) or
        :class:`~impurityModel.ed.eigensolvers.Within` (every state within ``max_energy`` of the
        lowest, certified complete).

        Above ``dense_cutoff``: one count-mode solve for ``Lowest``
        (:func:`~impurityModel.ed.eigensolvers.solve_lowest`), one window solve for ``Within``
        (:func:`~impurityModel.ed.eigensolvers.solve_window`). Below it a dense diagonalization --
        for ``Lowest`` only up to :data:`_DENSE_LOWEST_MAX` determinants, above which the count-mode
        solve is used whatever ``dense_cutoff`` says. Its cold start column converges, on a
        degenerate ground level, to the cold vector's projection onto the level: the same generic
        reference :meth:`_generic_lowest_level_vector` builds on the dense branch.
        Both branches cut with :func:`_energy_cut_indices`, which never bisects a degenerate
        manifold, so a group straddling the window edge is kept whole whichever branch ran. The
        window solve's initial retention is derived from the warm block and the
        previous call's (see :func:`_manifold_request`); it grows as the window demands.

        ``psi_refs``, if given, warm-starts the solve (its lowest ``GS_MAX_BLOCK_WIDTH`` columns for
        ``Within``, its lowest column for ``Lowest``); generic hash-seeded columns are always added,
        so every charge sector stays reachable from the start block. This method never reads or
        writes ``self.psi_refs``.

        ``reuse_last=True`` returns the states of the previous solve instead of solving again when
        that solve was certified, ran on the current determinant set (``Basis.generation``) and
        covered this request (a ``Within`` result covers a narrower window and ``Lowest``; a
        ``Lowest`` result covers only ``Lowest``). The caller asserts it is the same operator:
        ``expand``'s last solve followed by a read of its result is the use case.

        ``h_matrix``, if given, replaces ``build_sparse_matrix(self.basis, H)``: a caller sweeping a
        family ``H(mu) = H(0) - mu N_imp`` over one frozen determinant space builds once and shifts
        the diagonal. It must have been built from this basis, at this size, after the basis's
        restrictions were set; only the shape is checked.
        """
        if not isinstance(manifold, (Lowest, Within)):
            raise TypeError(f"manifold must be Lowest or Within, got {manifold!r}")
        if isinstance(H, dict):
            H = ManyBodyOperator(H)
        # Unconditional (None clears): a mask left on H by an earlier Green's-function stage must
        # not survive into this solve (review ledger C3).
        H.set_restrictions(self.basis.restrictions)
        H.set_weighted_restrictions(self.basis.weighted_restrictions)
        if h_matrix is not None and h_matrix.shape[0] != len(self.basis):
            raise ValueError(
                f"h_matrix was built for a basis of {h_matrix.shape[0]} determinants but this one "
                f"holds {len(self.basis)}. A matrix that outlived its basis produces eigenvalues "
                "of the wrong operator rather than an error, so this is checked."
            )
        lowest = isinstance(manifold, Lowest)
        max_energy = 0.0 if lowest else float(manifold.max_energy)
        if lowest:
            num_wanted = 1
        elif psi_refs is None:
            num_wanted = 10
            self._prev_kept = None
        else:
            # `.width`, not `len()`: len() on a block is its rank-local ROW count, which would make
            # num_wanted -- and the eigensolver collectives it sizes -- diverge across ranks.
            n_kept = _psi_ref_width(psi_refs)
            num_wanted = _manifold_request(n_kept, self._prev_kept)
            self._prev_kept = n_kept

        if self.basis.size >= (min(dense_cutoff, _DENSE_LOWEST_MAX) if lowest else dense_cutoff):
            reusable = (
                reuse_last
                and h_matrix is None
                and self.eig is not None
                and self.eig.certified
                and self.eig.generation == self.basis.generation
                and _covers(self.eig.manifold, manifold)
            )
            if reusable:
                eig = self.eig
            elif lowest:
                eig = self._solve_lowest(H, h_matrix, slaterWeightMin, psi_refs)
            else:
                eig = self._solve_window(H, h_matrix, manifold, num_wanted, slaterWeightMin, psi_refs)
            assert eig is not None
            e_ref, psi_refs_arr = eig.e, eig.vecs
            valid_idx = None
            if len(e_ref) > 0:
                # Rank-locally, deliberately: this cut sets ``len(e_ref)``, which callers feeding ``psi_refs``
                # into ``build_density_matrices`` turn into an ``Allreduce`` buffer shape. It is safe because
                # ``e_ref`` is bit-identical across ranks (TSQR returns a bitwise-identical R everywhere and
                # the Ritz values are a rank-local ``eigh`` on identical bytes).
                if lowest:
                    valid_idx = [0]
                elif reusable and eig.manifold != manifold:
                    # A reused wider result is cut to the narrower request ...
                    valid_idx, _ = _energy_cut_indices(e_ref, max_energy, tol=_degeneracy_tol(e_ref, slaterWeightMin))
                else:
                    # ... a fresh window solve keeps its own in-window set (it absorbed a straddling group whole).
                    valid_idx = list(range(eig.n_in))
                if not eig.certified and (self.basis.comm is None or self.basis.comm.rank == 0):
                    what = (
                        "the lowest eigenstate"
                        if lowest
                        else f"the {len(valid_idx)} eigenstates within {max_energy:.3g} of the lowest"
                    )
                    print(
                        f"warning: {what} could not be certified (solver status {eig.status!r})."
                        + ("" if lowest else " A degenerate manifold may be missing copies."),
                        flush=True,
                    )
                _trace_note(
                    "thermal_manifold",
                    n_computed=len(e_ref),
                    n_kept=len(valid_idx),
                    num_wanted=int(num_wanted),
                    n_dets=len(self.basis),
                    certified=bool(eig.certified),
                    max_multiplicity=int(eig.max_multiplicity),
                    reused=bool(reusable),
                )
                e_ref = e_ref[valid_idx]
            if len(e_ref) > 0:
                cols = psi_refs_arr[:, valid_idx]
                psi_refs = build_state(self.basis, cols.T, slaterWeightMin=slaterWeightMin)
        else:
            H_mat = h_matrix if h_matrix is not None else self._build_h_matrix(H)
            e_all, psi_dense = eigensystem(H_mat, e_max=None, comm=self.basis.comm, return_eigvecs=True)
            if lowest:
                idx = [0]
                psi_dense = self._generic_lowest_level_vector(e_all, psi_dense, slaterWeightMin)
            else:
                idx, _ = _energy_cut_indices(e_all, max_energy, tol=_degeneracy_tol(e_all, slaterWeightMin))
            e_ref = e_all[idx]
            psi_refs = build_state(self.basis, psi_dense[:, idx].T, slaterWeightMin=slaterWeightMin)

        return e_ref, psi_refs

    def _generic_lowest_level_vector(self, e_all, psi_dense, slaterWeightMin):
        """``psi_dense`` with column 0 replaced by a generic vector of the lowest level, when it is degenerate.

        ``eigh`` returns *some* orthonormal basis of a degenerate level, and which one is decided by
        roundoff. On the NiO 10-bath gap-DC search the 210-determinant d6 seed is the same matrix at
        every trial shift up to a constant diagonal, yet column 0 came back with three different
        supports (cycle-0 ``Hpsi_rows`` 18, 72 and 180). Every expansion started from the 18-row
        member converged 0.15-0.17 above the others: it sits in a symmetry sector the expansion never
        leaves, and the resulting noise exceeded every energy difference the search compares.

        The projection of the hash-seeded :func:`cold_vector` onto the level does not depend on which
        basis ``eigh`` chose for it, so it is the same state at every shift, and it overlaps every
        symmetry sector the level contains. A non-degenerate level is returned untouched.

        The degeneracy verdict is read from ``e_all``, which ``eigensystem`` broadcasts, so every rank
        enters the ``allreduce`` below or none does.
        """
        group = _degenerate_groups(e_all, tol=_degeneracy_tol(e_all, slaterWeightMin))[0]
        if len(group) < 2:
            return psi_dense
        hashes = np.fromiter((state.get_hash() for state in self.basis.local_basis), dtype=np.uint64)
        level = psi_dense[:, group]
        overlaps = level[np.asarray(self.basis.local_indices, dtype=np.intp)].conj().T @ cold_vector(hashes)[:, 0]
        if self.basis.is_distributed:
            self.basis.comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
        generic = level @ overlaps
        # `astype` copies, and widens a real `eigh` result: the hash vector is complex.
        psi_dense = psi_dense.astype(complex)
        psi_dense[:, 0] = generic / np.linalg.norm(generic)
        return psi_dense

    def _solve_lowest(self, H, h_matrix, slaterWeightMin, psi_refs) -> EigenResult:
        """One count-mode solve for the lowest pair on the current basis; records it as ``self.eig``."""
        H_mat = h_matrix if h_matrix is not None else self._build_h_matrix(H)
        hashes = np.fromiter((state.get_hash() for state in self.basis.local_basis), dtype=np.uint64)
        warm = self._warm_columns(psi_refs, width=1)
        _trace_note("eigensolve_block_width", p=int(warm.shape[1] + 1), num_wanted=1)
        with _trace_timed("eigensolve", n=len(self.basis), window=0.0):
            eig = solve_lowest(
                H_mat,
                self.basis.comm if self.basis.is_distributed else None,
                hashes,
                warm=warm,
                tol=_eigen_tol(slaterWeightMin),
                n_global=len(self.basis),
                max_restarts=100,
                verbose=self.basis.verbose,
            )
        eig.generation = self.basis.generation
        eig.manifold = Lowest()
        self.eig = eig
        return eig

    def _solve_window(self, H, h_matrix, manifold, num_wanted, slaterWeightMin, psi_refs) -> EigenResult:
        """One certified window solve on the current basis; records it as ``self.eig``."""
        max_energy = float(manifold.max_energy)
        H_mat = h_matrix if h_matrix is not None else self._build_h_matrix(H)
        hashes = np.fromiter((state.get_hash() for state in self.basis.local_basis), dtype=np.uint64)
        warm = self._warm_columns(psi_refs)
        tol = _eigen_tol(slaterWeightMin)
        # The degeneracy tolerance is a function of the eigenvalues, which the solve is about to find:
        # use the previous result's scale when there is one, else the floor, and re-derive from the
        # converged energies below.
        deg_tol = _degeneracy_tol(self._eig_scale(), slaterWeightMin)
        _trace_note("eigensolve_block_width", p=int(warm.shape[1] + self.n_random), num_wanted=int(num_wanted))
        with _trace_timed("eigensolve", n=int(len(self.basis)), window=float(max_energy)):
            eig = solve_window(
                H_mat,
                self.basis.comm if self.basis.is_distributed else None,
                hashes,
                warm=warm,
                max_energy=max_energy,
                deg_tol=deg_tol,
                tol=tol,
                num_wanted=num_wanted,
                n_global=len(self.basis),
                n_random=self.n_random,
                max_restarts=100,
                verbose=self.basis.verbose,
            )
        eig.generation = self.basis.generation
        eig.manifold = Within(max_energy)
        self.eig = eig
        self.n_random = max(_MIN_RANDOM_COLUMNS, eig.max_multiplicity + 1)
        return eig
