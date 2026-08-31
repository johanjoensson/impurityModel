import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed.average import energy_cut
from impurityModel.ed.basis_transcription import (
    build_distributed_vector,
    build_sparse_matrix,
    build_state,
)
from impurityModel.ed.BlockLanczosArray import Reort, block_normalize
from impurityModel.ed.eigensolvers import eigensystem
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.manybody_basis import Basis, collective_amplitude_cutoff
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
from impurityModel.ed.solver_basis import get_symmetry_generators
from impurityModel.ed.trlm import thick_restart_block_lanczos

SOLVERS = {
    "trlm": thick_restart_block_lanczos,
    "irlm": implicitly_restarted_block_lanczos_cy,
}

_U64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """One splitmix64 round: a well-mixed 64-bit hash of a 64-bit integer."""
    x = (x + 0x9E3779B97F4A7C15) & _U64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64
    return x ^ (x >> 31)


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

#: How far above the eigensolver's residual tolerance the degeneracy grouping tolerance must sit.
#: A Ritz pair converged to ``||r||`` has ``|theta - lambda| <= ||r||``, and *within* a cluster the
#: returned Ritz values are generically split by ``O(||r||)``, so grouping at anything below the
#: residual splits manifolds that are exactly degenerate. One order of margin: large enough that
#: an ``O(||r||)`` splitting is absorbed, small enough to stay orders below any physical scale.
DEGENERACY_TOL_MARGIN = 10.0

#: How many times ``get_eigenvectors`` may double ``num_wanted`` looking for an eigenstate beyond
#: the thermal energy cut. Each doubling is a full re-solve, so this bounds the worst case; in
#: practice the first solve already overshoots the cut.
_MAX_EIGENSTATE_DOUBLINGS = 24

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
    1e-8, and three manifold-consuming paths run exactly there -- ``dc_frozen.FrozenSpaceSweep``
    and ``dc_criteria`` both default ``slater_weight_min=0``, and the occupation walk runs at
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

    if max_energy is None:
        return [int(i) for i in order], False

    n_keep = max(int(np.count_nonzero(e_sorted - e_sorted[0] <= max_energy)), 1)
    # Never split a manifold: absorb any state degenerate with the last kept one.
    original_n_keep = n_keep
    while n_keep < len(e_sorted) and abs(e_sorted[n_keep] - e_sorted[original_n_keep - 1]) <= tol:
        n_keep += 1
    return [int(i) for i in order[:n_keep]], n_keep == len(e_sorted)


def _amplitude_from_hash(det_hash: int) -> complex:
    """A deterministic pseudo-random start amplitude for one determinant.

    Depends only on the determinant (through its C++ splitmix64 hash), never on the MPI rank
    that happens to own it or on ``PYTHONHASHSEED``. That makes the Lanczos start vector -- and
    therefore the CIPSI basis grown from it -- identical at any rank count. The two components
    come from independently mixed streams so the real and imaginary parts are uncorrelated.
    """
    re = _splitmix64(det_hash) / 2.0**64
    im = _splitmix64(det_hash ^ 0xD1B54A32D192ED03) / 2.0**64
    return complex(re, im)


#: Whether ``expand`` completes symmetry orbits when the caller passes no generators.
#:
#: **Off**, on measurement rather than principle. With correct generators (the total spin ladder
#: operators, gated on ``[H, g] = 0`` against the full Hamiltonian) the closure does what it
#: promises -- on a cubic d-shell with a Slater interaction it drives the ground triplet's
#: ``|S^2 - 2|`` from 1.3e-11 to 5.1e-15 -- but only in the *uncapped* limit, and at 3589
#: determinants against 1771. Under a binding ``truncation_threshold`` it is a net loss: at a cap
#: of 400 the closure run returned ``E0 = -13.26573223`` against ``-13.26573295`` without it,
#: because symmetry partners are not generally the highest-de2 candidates and displace ones that
#: are. Two legitimate goals -- variational energy and symmetry purity -- that conflict under a
#: budget, so the caller has to choose. Pass ``symmetry_generators`` explicitly to opt in.
SYMMETRY_CLOSURE_DEFAULT = False


def _commutes_with(h_op, op, tol: float = 1e-10) -> bool:
    """Is ``op`` a symmetry of ``h_op``, i.e. ``[h_op, op] = 0``?

    Checked against the operator as a whole, two-body terms included -- the point being that
    the one-body block's commutant is generally much larger than the full Hamiltonian's, so a
    generator discovered from ``h`` alone is not a symmetry of ``h + U``.

    Pure operator algebra: no basis, no communication, and
    :meth:`ManyBodyOperator.commutator` skips term pairs on disjoint orbitals without forming
    them, so this costs a pass over the terms the generator actually touches.
    """
    residual = h_op.commutator(op)
    return not residual or max((abs(v) for v in residual.values()), default=0.0) <= tol


def _scalar_amp(row, default: complex = 0.0) -> complex:
    """Read back a width-1 block ``Row`` (or ``None``) as a plain scalar amplitude."""
    return default if row is None else row[0]


class CIPSISolver:
    def __init__(self, basis: Basis):
        self.basis = basis
        self.psi_refs = None
        # Diagnostics of the latest candidate selection / basis truncation (see
        # determine_new_Dj / truncate); None until the corresponding event happens.
        self.last_selection = None
        self.last_truncation: dict | None = None
        self.truncation_report = None

    def _allreduce_sum(self, value):
        if self.basis.is_distributed:
            return self.basis.comm.allreduce(value, op=MPI.SUM)
        return value

    def _allreduce_max(self, value):
        if self.basis.is_distributed:
            return self.basis.comm.allreduce(value, op=MPI.MAX)
        return value

    def truncate_initial(self, H: ManyBodyOperator, dense_cutoff=1e3) -> None:
        """Perform an initial truncation if the basis exceeds the truncation threshold.

        Routed through :meth:`get_eigenvectors` (TRLM above ``dense_cutoff``, dense below)
        rather than calling ``eigensystem``/ARPACK directly: multi-rank ARPACK
        (``scipy_eigensystem``) lets each rank's own ARPACK convergence bookkeeping decide how
        many collectives it posts, which under multithreaded BLAS diverges between ranks and
        deadlocks (measured -n3 hang, see the memory note
        n3-arpack-truncate-initial-deadlock). TRLM's collective count depends only on
        `basis`/`H_mat`, which are already rank-uniform, so it cannot desync this way.
        """
        if self.basis.size > self.basis.truncation_threshold and H is not None:
            if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
                print("Truncating basis!")
            # Rank over the low-energy manifold (up to 10 states within energy_cut, though
            # get_eigenvectors' TRLM branch pads num_wanted by +10, so this can return up to 20
            # states), not a single eigenvector: the downstream expansion / get_eigenvectors
            # keep the whole near-degenerate ground manifold, so truncating to one state's
            # support would bias the retained determinants toward one member of the multiplet.
            # Cold start (psi_refs=None): truncate_initial runs before any converged block
            # exists, so get_eigenvectors falls back to its rank-independent hash start vector.
            _e_ref, psi_ref = self.get_eigenvectors(
                H,
                num_wanted=10,
                max_energy=energy_cut(self.basis.tau),
                dense_cutoff=dense_cutoff,
                psi_refs=None,
            )
            self.truncate(psi_ref, _e_ref)

    def truncate(self, psis: list[ManyBodyState], e_ref=None, target=None, slaterWeightMin=0) -> list[ManyBodyState]:
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
        blk = ManyBodyState.from_states(list(psis))
        if e_ref is not None and len(e_ref) == len(psis):
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
        cutoff2 = collective_amplitude_cutoff(norms2, int(target), self.basis.comm)
        keep_mask = norms2 > cutoff2
        if self._allreduce_sum(int(np.count_nonzero(keep_mask))) == 0:
            # The bisection under-admits ties: if every candidate ties at the maximum
            # score the strict cutoff retains nothing. Keep the max-score tie class
            # (possibly exceeding target) rather than emptying the basis.
            global_max = self._allreduce_max(float(norms2.max()) if norms2.size else 0.0)
            if global_max > 0.0:
                keep_mask = norms2 >= global_max
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
        return blk.to_states()

    def _apply_block_and_redistribute(self, H, psi_ref, cutoff):
        """Apply ``H`` to the reference states as one shared-support block, then redistribute.

        Amortizes the term walk / sign / restriction-mask / hash work over
        ``len(psi_ref)`` columns in a single :meth:`ManyBodyOperator.apply_block` call,
        instead of ``len(psi_ref)`` separate per-state applies. ``apply_block`` keeps a
        row if ANY column exceeds ``cutoff`` -- a safe superset per column, but not the
        same as pruning each column to its own threshold -- so each split-out column is
        re-pruned to ``cutoff`` before the cross-rank sum. This reproduces the old
        per-state ``applyOp(H, psi_i, cutoff)`` bit-for-bit, including pruning locally
        *before* redistributing (the same order every other probe in this module uses,
        e.g. ``psi_all_Dj`` above), rather than pruning the already-summed total.

        Returns the merged **block** (not a list): ``_candidate_overlaps_and_energies``
        reads it via the buffer protocol (``np.asarray``) instead of iterating
        ``.items()``. ``prune_rows(0.0)`` after the redistribute reproduces the old
        ``to_states()``-based union exactly: a determinant survives iff at least one
        column is genuinely nonzero, so an exact cross-rank cancellation on every
        column of a given row is the only way to drop it -- a real selection-rule
        cancellation, not a truncation artifact.
        """
        cols = H.apply_block(ManyBodyState.from_states(psi_ref), cutoff).to_states()
        for s in cols:
            s.prune_rows(cutoff)
        merged = self.basis.redistribute_block(ManyBodyState.from_states(cols))
        merged.prune_rows(0.0)
        return merged

    def _candidate_overlaps_and_energies(self, H, Hpsi_ref, slaterWeightMin: float = 0):
        """Enumerate the out-of-basis candidates of ``Hpsi_ref`` with couplings and energies.

        The machinery every CIPSI-style selection shares (the ground-state
        :meth:`_calc_de2` and the resolvent-targeted :meth:`select_at` differ only in the
        importance denominator applied on top): this rank's hash-owned candidate
        determinants ``local_Djs`` (sorted, so rank-independent given ``Hpsi_ref`` is
        redistributed), the coupling matrix ``overlaps[i, j] = <Dj | H | psi_i>`` read off
        ``Hpsi_ref``, and the diagonal-probe energies ``e_Dj[j] ~ <Dj|H|Dj>``.

        ``Hpsi_ref`` is normally the block :meth:`_apply_block_and_redistribute` returns
        (already redistributed and pruned to its own support); a bare list of width-1
        states is also accepted (wrapped via ``from_states``) for direct callers. Reading
        the block via the buffer protocol (``np.asarray``) instead of ``.items()`` avoids
        allocating a ``SlaterDeterminant`` and a ``Row`` object per row per column -- the
        dominant Python-level cost of a selection round (measured: 33-43% of the round
        across two problem sizes, `doc/plans/manybodystate_block_unification.md`'s
        Phase 9 write-up).

        Collective on ``basis.comm`` (the probe redistribution), so it must run on every
        rank -- including one that owns no candidates, whose arrays come back empty.
        """
        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        _index_dict = self.basis._index_dict
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
        new_mask = np.fromiter((k not in _index_dict for k in keys), dtype=bool, count=len(keys))
        local_Djs = list(itertools.compress(keys, new_mask))
        overlaps = np.ascontiguousarray(amps[new_mask].T)  # (p, n_Dj); boolean indexing copies
        del amps  # release the buffer export before any later mutation of Hpsi_ref

        # Diagonal probe <Dj|H|Dj> from a single H application to one superposition of
        # all candidates. Unit-modulus pseudo-random phases (derived from the
        # determinant hash, so deterministic and rank-independent) make the
        # candidate-candidate couplings enter with quasi-random phases instead of the
        # systematic offset an all-ones probe would add to the diagonal estimate.
        #
        # `local_Djs` is this rank's share of the hash-routed candidates (the *union* over
        # ranks is partition independent, the per-rank slice is not). So `H psi_all_Dj`
        # computed locally is a partial sum: it misses every candidate-candidate coupling
        # <Dj|H|Dk> whose Dk is owned by another rank -- and *which* ones are missing depends
        # on `comm.size`. Redistributing accumulates each determinant's contributions on its
        # hash owner, reconstructing the exact global probe, so `e_Dj` (and hence the whole
        # CIPSI selection, and every basis derived from it) is the same at any rank count.
        #
        # `redistribute_psis` is COLLECTIVE, so it must run on every rank -- including a rank
        # that happens to own no candidates at all, whose probe is simply empty. Returning
        # early on `not local_Djs` before it deadlocks the ranks that do have candidates.
        phases = np.exp(2j * np.pi * np.array([(hash(Dj) & 0xFFFF) / 65536.0 for Dj in local_Djs]))
        # width=1 even when local_Djs is empty on this rank: a bare empty dict would
        # construct the width-0 polymorphic zero, which would make this rank's total
        # flattened width in redistribute_psis' combined block disagree with every
        # other rank's -- an asymmetric wire shape in the shared collective, the same
        # deadlock class fixed in Phase 7 step 2b (there caught as a clean per-rank
        # raise; here it would surface as a silently mismatched pack instead).
        psi_all_Dj = ManyBodyState({Dj: phases[j] for j, Dj in enumerate(local_Djs)}, width=1)
        H_psi_all = applyOp_test(H, psi_all_Dj, cutoff=slaterWeightMin)
        if self.basis.is_distributed:
            H_psi_all = self.basis.redistribute_psis(H_psi_all)[0]

        if not local_Djs:
            return local_Djs, overlaps, np.zeros(0, dtype=float)

        e_Dj = np.array(
            [np.real(np.conj(phases[j]) * _scalar_amp(H_psi_all.get(Dj))) for j, Dj in enumerate(local_Djs)],
            dtype=float,
        )
        return local_Djs, overlaps, e_Dj

    def _calc_de2(self, H, Hpsi_ref, e_ref: np.ndarray, slaterWeightMin: float = 0):
        local_Djs, overlaps, e_Dj = self._candidate_overlaps_and_energies(H, Hpsi_ref, slaterWeightMin)
        if not local_Djs:
            return local_Djs, overlaps

        # Epstein-Nesbet importance |<Dj|H|psi>|^2 / |E_ref - E_Dj| (the magnitude of
        # the second-order energy contribution). The denominator must not be a signed
        # clamp: candidates sit *above* E_ref for a ground-state search, so
        # max(E_ref - E_Dj, eps) would collapse to eps and turn the selection into a
        # bare coupling filter.
        de = np.maximum(np.abs(e_ref[:, None] - e_Dj[None, :]), 1e-12)
        de2 = np.zeros_like(overlaps)
        mask = np.abs(overlaps) > 1e-12
        de2[mask] = np.square(np.abs(overlaps[mask])) / de[mask]
        return local_Djs, de2

    def _admit_top(self, scores, mask, max_new):
        """Cap an importance-masked candidate set at the globally top ``max_new`` scores.

        ``mask`` is the rank-local boolean pre-selection (e.g. ``scores >= de2_min``);
        ``max_new=None`` admits it unchanged. Otherwise the cutoff comes from the
        collective amplitude bisection so every rank admits the identical set (ties at
        the cutoff under-admitted, with the all-tied fallback admitting the max-score
        tie class rather than nothing). Collective on ``basis.comm``; returns
        ``(admitted_mask, stats)`` with the ``last_selection``-shaped stats dict.
        """
        n_candidates = self._allreduce_sum(int(np.count_nonzero(mask)))
        discarded_de2_mass = 0.0
        if max_new is not None and n_candidates > max_new:
            if max_new <= 0:
                # An exhausted budget: admit nothing (callers still want the selection
                # stats, e.g. the boundary residual). The tie fallback below must not
                # run -- it exists to avoid *under*-admission, not to override a zero cap.
                discarded_de2_mass = self._allreduce_sum(float(scores[mask].sum()))
                mask = np.zeros_like(mask)
            else:
                comm = self.basis.comm if self.basis.is_distributed else None
                cutoff = collective_amplitude_cutoff(scores[mask], int(max_new), comm)
                admitted = mask & (scores > cutoff)
                if self._allreduce_sum(int(np.count_nonzero(admitted))) == 0:
                    # All candidates tie at the maximum importance (the bisection
                    # under-admits ties): admit the max-score tie class instead of nothing.
                    global_max = self._allreduce_max(float(scores[mask].max()) if np.any(mask) else 0.0)
                    if global_max > 0.0:
                        admitted = mask & (scores >= global_max)
                discarded_de2_mass = self._allreduce_sum(float(scores[mask & ~admitted].sum()))
                mask = admitted
        stats = {
            "n_candidates": n_candidates,
            "n_admitted": self._allreduce_sum(int(np.count_nonzero(mask))),
            "discarded_de2_mass": discarded_de2_mass,
        }
        return mask, stats

    def select_at(self, z, psi_ref, H, de2_min=0.0, max_new=None, scorer="de2", slater_cutoff=0):
        r"""One resolvent-targeted selection round around the complex frequency ``z``.

        The Green's-function analogue of :meth:`determine_new_Dj` (the revived
        ``expand_at``): for the linear system :math:`(z - H) X = s` solved on the current
        basis :math:`P`, the residual at an out-of-basis determinant :math:`D_j` is exactly
        :math:`-\langle D_j | H | X \rangle` (the seed lives inside :math:`P`), so with
        ``psi_ref`` = the current iterate block the selection is residual-driven greedy
        expansion, and the leading-order weight of :math:`D_j` in the exact solution is

        .. math:: w_j = \sum_i |\langle D_j|H|X_i\rangle|^2 \, / \, |z - E_{D_j}|^2 .

        The column sum makes the score invariant under rotations within the reference
        block (the manifold-sum rationale of :meth:`determine_new_Dj`); the complex ``z``
        (carrying :math:`i\delta` or the Matsubara distance) regularizes near-resonant
        candidates, so no clamp is needed -- near-resonant *is* important here, that is
        the point of frequency targeting.

        Parameters
        ----------
        z : complex
            The resolvent frequency, already shifted by the eigenstate energy
            (``omega + i*delta + E_e`` in the Green's-function drivers).
        psi_ref : list of ManyBodyState
            Reference block, distributed per ``self.basis`` -- the current solution
            iterate (or the seeds, for a cold start).
        H : ManyBodyOperator or dict
            The (unshifted) Hamiltonian.
        de2_min : float, optional
            Importance floor on :math:`w_j`; 0 keeps every coupled candidate and leaves
            the capping entirely to ``max_new``.
        max_new : int, optional
            Global cap on the number of admitted candidates (collective bisection, ties
            under-admitted -- see :meth:`_admit_top`).
        scorer : {"de2", "amplitude"}, optional
            ``"de2"`` is the resolvent importance above; ``"amplitude"`` drops the energy
            denominator (:math:`w_j = \sum_i |\langle D_j|H|X_i\rangle|^2`), the
            bare-coupling baseline the frequency targeting must beat.
        slater_cutoff : float, optional
            Amplitude cutoff forwarded to the ``H`` applications.

        Returns
        -------
        new_Dj : set
            This rank's admitted candidates (hash-owned).
        stats : dict
            ``boundary_norms2``: global per-reference-column boundary residual norms
            :math:`\sum_{D \notin P} |\langle D|H|X_i\rangle|^2` (the true-residual
            contribution outside the basis -- the outer loop's convergence measure),
            the :meth:`_admit_top` selection counts, and the rank-local PT2 ingredients
            ``overlaps`` (couplings, reference x candidate), ``e_Dj`` (diagonal-probe
            energies) and ``admitted`` (mask) for the downfolding correction of the
            discarded boundary.

        Collective on ``basis.comm`` (every branch, including empty candidate sets).
        """
        if isinstance(H, dict):
            H = ManyBodyOperator(H)
        Hpsi_ref = self._apply_block_and_redistribute(H, psi_ref, slater_cutoff)
        local_Djs, overlaps, e_Dj = self._candidate_overlaps_and_energies(H, Hpsi_ref, slater_cutoff)

        coupling2 = np.square(np.abs(overlaps))
        boundary_norms2 = np.ascontiguousarray(coupling2.sum(axis=1), dtype=float)
        if self.basis.is_distributed:
            self.basis.comm.Allreduce(MPI.IN_PLACE, boundary_norms2, op=MPI.SUM)

        if scorer == "de2":
            scores = coupling2.sum(axis=0) / np.square(np.abs(complex(z) - e_Dj))
        elif scorer == "amplitude":
            scores = coupling2.sum(axis=0)
        else:
            raise ValueError(f"Unknown scorer {scorer!r}; expected 'de2' or 'amplitude'")

        admitted, selection_stats = self._admit_top(scores, scores >= de2_min, max_new)
        new_Dj = set(itertools.compress(local_Djs, admitted))
        stats = {
            "boundary_norms2": boundary_norms2,
            "overlaps": overlaps,
            "e_Dj": e_Dj,
            "admitted": admitted,
            **selection_stats,
        }
        return new_Dj, stats

    def determine_new_Dj(
        self, e_ref, psi_ref, H, de2_min, slater_cutoff=0, return_Hpsi_ref=False, gen_ops=None, max_new=None
    ):
        """Select the candidate determinants to add to the basis.

        Candidates connected to ``psi_ref`` through ``H`` are kept when their de2
        importance (max over the reference states) reaches ``de2_min``. ``max_new``
        optionally caps the *global* number of selected candidates: the top ``max_new``
        by de2 importance are kept (collective bisection cutoff, ties under-admitted)
        before the symmetry closure, and ``self.last_selection`` records
        ``{"n_candidates", "n_admitted", "discarded_de2_mass"}``. Collective on
        ``basis.comm``.
        """
        Hpsi_ref = self._apply_block_and_redistribute(H, psi_ref, slater_cutoff)
        local_Djs, de2 = self._calc_de2(H, Hpsi_ref, e_ref)
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
            de2_abs = np.abs(de2)
            groups = _degenerate_groups(e_ref, tol=_degeneracy_tol(e_ref, slater_cutoff))
            scores = np.max(np.stack([de2_abs[g, :].sum(axis=0) for g in groups]), axis=0)
        else:
            scores = np.zeros(0)
        de2_mask, selection_stats = self._admit_top(scores, scores >= de2_min, max_new)
        self.last_selection = selection_stats
        new_Dj = set(itertools.compress(local_Djs, de2_mask))

        if gen_ops:
            unexplored_list = sorted(new_Dj)
            chunk_size = 1000

            while unexplored_list:
                chunk = unexplored_list[:chunk_size]
                unexplored_list = unexplored_list[chunk_size:]

                # Pseudo-random superpositions (derived from each determinant's hash, not
                # Python's global `random` stream) avoid destructive interference the same
                # way the diagonal probe above does, but deterministically: an unseeded
                # `random.random()` here made which determinants this closure discovers --
                # and hence the basis grown from them -- depend on run-to-run RNG state and
                # on `new_Dj`'s (rank-local, insertion-order-dependent) set iteration order,
                # the same reproducibility failure `_amplitude_from_hash` was introduced to
                # close off elsewhere in this class.
                # No MPI collective anywhere in this closure (each rank explores its own
                # local_Djs independently), so ManyBodyState's width-0 polymorphic
                # zero on an empty next_amps below is just a local falsy value, not the
                # cross-rank deadlock hazard it is at a collective boundary.
                chunk_state = ManyBodyState({state: _amplitude_from_hash(state.get_hash()) for state in chunk})

                while chunk_state:
                    # Collect into a plain dict and build the next ManyBodyState in
                    # one bulk range-insert (its dict constructor's flat_map insert(begin,
                    # end)) instead of `p` repeated single-key inserts: `operator[]` on a
                    # missing key is a sorted-vector insert, so accumulating one
                    # determinant at a time here was O(n^2) in the size of the closure
                    # wave. Every determinant can only be discovered once across the
                    # whole pass (the `state not in new_Dj` guard below), so no key here
                    # is ever written twice -- batching changes nothing about which
                    # (state, amp) pairs end up in the next wave, only how they're
                    # assembled into it.
                    next_amps = {}
                    for op in gen_ops:
                        # Apply generator (cutoff=1e-12 to prune float noise)
                        psi_op = applyOp_test(op, chunk_state, cutoff=1e-12)

                        for state, _row in psi_op.items():
                            # Skip determinants already IN THE BASIS as well as ones already
                            # discovered in this pass. `_candidate_overlaps_and_energies` only
                            # ever returns out-of-basis candidates, and `Basis.add_states` dedupes
                            # against the same index, so an in-basis image admits nothing -- but
                            # it still inflated `n_new` in `expand`, which is compared against the
                            # `truncation_threshold`. That spuriously triggered a fixed-budget
                            # cycle and made room by pruning genuinely important determinants for
                            # candidates that were already there. Measured on a 12-orbital toy at
                            # cap 120: E0 worse by 30x, and the final basis 94 determinants -- a
                            # cap the run never actually reached.
                            #
                            # `contains_local`, never `in self.basis`: the latter runs a routed
                            # global index query when the basis is distributed, and this closure
                            # is rank-local (each rank walks its own `local_Djs`, so ranks reach
                            # here a different number of times). A collective in here is the
                            # deadlock CLAUDE.md's MPI rules describe. Distributed runs therefore
                            # still over-count images owned by another rank -- an upper bound on
                            # `n_new`, never an under-count, so the cap stays conservative.
                            if state not in new_Dj and not self.basis.contains_local(state):
                                new_Dj.add(state)
                                next_amps[state] = _amplitude_from_hash(state.get_hash())

                    chunk_state = ManyBodyState(next_amps)

        if return_Hpsi_ref:
            return new_Dj, Hpsi_ref
        return new_Dj

    def expand(
        self,
        H,
        de2_min=1e-10,
        dense_cutoff=1e3,
        slaterWeightMin=0,
        solver="trlm",
        reort=Reort.PARTIAL,
        symmetry_generators=None,
        cap_e_tol=1e-8,
        max_cap_cycles=10,
    ):
        """Expand the basis variationally (CIPSI) until it stops growing.

        With a finite ``basis.truncation_threshold`` the expansion becomes a
        **fixed-budget CIPSI** once the cap binds: each cycle prunes the currently
        least important determinants (by eigenvector amplitude, collective top-K),
        admits the best de2-ranked candidates into the freed room, and
        re-diagonalizes; cycles stop when the ground-state energy changes by less
        than ``cap_e_tol`` or after ``max_cap_cycles`` cycles. ``truncation_report``
        records whether (and how) the cap bound the expansion.
        """
        if self.basis.restrictions is not None:
            H.set_restrictions(self.basis.restrictions)
        if self.basis.weighted_restrictions is not None:
            H.set_weighted_restrictions(self.basis.weighted_restrictions)
        de0_max = energy_cut(self.basis.tau)
        psi_refs = getattr(self, "psi_refs", None)

        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        if symmetry_generators is None:
            symmetry_generators = (
                get_symmetry_generators(H, self.basis.impurity_orbitals, self.basis.bath_states)
                if SYMMETRY_CLOSURE_DEFAULT
                else []
            )

        from impurityModel.ed.symmetries import tensors_to_operator

        # Keep only generators that actually commute with the FULL H, two-body included.
        # `get_symmetry_generators` discovers them from the one-body block alone
        # (`extract_tensors(h_op, two_body=False)`) and gates them against that same one-body
        # matrix, so a degenerate impurity block hands over the whole commutant of `h_imp` --
        # far more than the Coulomb tensor preserves. Measured on a degenerate 4-orbital block
        # with an orbital-dependent U: 8 of 16 discovered generators have `[H, g] != 0`.
        #
        # A non-symmetry does not corrupt the scoring (the closure only enlarges the candidate
        # set, never the de2 ranking), and with no cap it merely wastes determinants. Under a
        # binding `truncation_threshold` it is not benign: the junk images compete for the same
        # budget and displace determinants that carry real weight, so the "symmetry-adapted"
        # run comes back with a *higher* variational energy than the plain one.
        gen_ops = []
        for g in symmetry_generators:
            # `get_symmetry_generators` returns operators (the total spin ladder operators);
            # callers may still hand over plain one-body matrices.
            op = g if isinstance(g, ManyBodyOperator) else tensors_to_operator(g, tol=1e-12)
            if not _commutes_with(H, op):
                continue
            if self.basis.restrictions is not None:
                op.set_restrictions(self.basis.restrictions)
            if self.basis.weighted_restrictions is not None:
                op.set_weighted_restrictions(self.basis.weighted_restrictions)
            gen_ops.append(op)

        threshold = self.basis.truncation_threshold
        capped = np.isfinite(threshold)
        cap_cycles = 0
        no_improve = 0
        e0 = np.inf
        best_e0 = None
        best_basis = None
        best_psis = None
        best_e_ref = None
        self.truncation_report = None
        while True:
            e_ref, psi_refs = self.get_eigenvectors(
                H,
                num_wanted=min(2 * len(psi_refs) if psi_refs is not None else 10, len(self.basis)),
                max_energy=de0_max,
                dense_cutoff=dense_cutoff,
                slaterWeightMin=slaterWeightMin,
                solver=solver,
                reort=reort,
                psi_refs=psi_refs,
            )

            if len(e_ref) == 0:
                break
            e0 = float(np.min(e_ref))
            if cap_cycles > 0:
                # Fixed-budget refinement: keep the best capped basis seen so far and
                # stop once cycles stop lowering the (variational) ground-state energy.
                improved = best_e0 is None or e0 < best_e0 - cap_e_tol
                if best_e0 is None or e0 < best_e0:
                    best_e0 = e0
                    best_basis = list(self.basis.local_basis)
                    best_psis = psi_refs
                    best_e_ref = e_ref
                no_improve = 0 if improved else no_improve + 1
                if no_improve >= 2 or cap_cycles >= max_cap_cycles:
                    break

            admit_target = None
            if capped:
                # Cap the selection so one cycle turns over at most ~10% of the basis
                # (or fills the remaining budget, whichever is larger).
                budget = int(threshold) - self.basis.size
                admit_target = max(budget, -(-int(threshold) // 10))
            new_Dj = self.determine_new_Dj(
                e_ref, psi_refs, H, de2_min, slater_cutoff=slaterWeightMin, gen_ops=gen_ops, max_new=admit_target
            )
            old_size = self.basis.size
            n_new = self._allreduce_sum(len(new_Dj))
            if capped and self.basis.size + n_new > threshold:
                # Fixed-budget CIPSI cycle: make room by dropping the currently least
                # important determinants (by eigenvector amplitude), then admit the
                # de2-ranked candidates; the loop head re-diagonalizes and the cycle
                # repeats until the energy stabilizes.
                cap_cycles += 1
                keep = max(int(threshold) - n_new, int(threshold) // 2, 1)
                psi_refs = self.truncate(psi_refs, e_ref, target=keep, slaterWeightMin=slaterWeightMin)
                if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
                    print(
                        f"------> Basis truncated! (cycle {cap_cycles}: kept "
                        f"{self.last_truncation['retained']:,} determinants, admitting {n_new:,} candidates)"
                    )
            self.basis.add_states(new_Dj)
            psi_refs = self.basis.redistribute_psis(*psi_refs)
            if cap_cycles == 0 and self.basis.size == old_size:
                break
            e0 = np.inf  # the basis was mutated; e0 no longer describes it
        if best_basis is not None and e0 > best_e0:
            # The last refinement cycle left a worse basis (e.g. score/amplitude
            # ping-pong): restore the best capped basis seen during the cycles.
            self.basis.clear()
            self.basis.add_states(best_basis)
            psi_refs = self.basis.redistribute_psis(*best_psis)
            e_ref = best_e_ref
        self.psi_refs = psi_refs
        if capped and self.basis.size > threshold and self.psi_refs is not None:
            # The symmetry closure can push an admission slightly past the cap; enforce
            # the hard threshold on exit (downstream get_eigenvectors re-solves). e_ref
            # must describe the *current* psi_refs -- after a best_basis restore above,
            # that is best_e_ref, not the last cycle's (possibly length-mismatched) e_ref.
            self.psi_refs = self.truncate(self.psi_refs, e_ref, slaterWeightMin=slaterWeightMin)
        if cap_cycles > 0:
            sel = self.last_selection or {}
            self.truncation_report = {
                "cap_hit": True,
                "cycles": cap_cycles,
                "retained": int(self.basis.size),
                "threshold": int(threshold),
                "discarded_de2_mass": float(sel.get("discarded_de2_mass", 0.0)),
                "n_candidates_last": int(sel.get("n_candidates", 0)),
            }
            rank = self.basis.comm.rank if self.basis.is_distributed else 0
            if rank == 0:
                rep = self.truncation_report
                # discarded_de2_mass is an error bound on the ground state (PT2 importance
                # left out of the retained subspace), so this fires regardless of verbose;
                # the refinement-cycle detail is cosmetic and stays behind -v.
                print(
                    f"WARNING: GS basis cap hit: fixed-budget CIPSI held the basis at "
                    f"{rep['retained']:,} determinants (truncation_threshold={rep['threshold']:,}); "
                    f"discarded candidates carry {rep['discarded_de2_mass']:.3e} of PT2 importance. "
                    f"The ground state is exact on the retained subspace.",
                    flush=True,
                )
                if self.basis.verbose:
                    print(
                        f"  ({rep['cycles']} refinement cycle(s); {rep['n_candidates_last']:,} "
                        "candidates considered in the last cycle)",
                        flush=True,
                    )

        if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
            print(f"After expansion, the basis contains {self.basis.size} elements.", flush=True)

    def get_eigenvectors(
        self,
        H,
        num_wanted: int,
        max_energy=None,
        dense_cutoff=1e3,
        slaterWeightMin=0,
        solver="trlm",
        reort=Reort.PARTIAL,
        psi_refs=None,
        h_matrix=None,
    ):
        """Solve for the ``num_wanted`` lowest eigenstates below ``max_energy``.

        ``psi_refs``, if given, warm-starts the Krylov solve from a previously converged
        eigenvector block (e.g. the caller's own ``solver.psi_refs`` from a prior ``expand``/
        ``get_eigenvectors`` call); ``None`` is a cold start from the rank-independent hash
        vector. Callers own this choice explicitly -- this method never reads or writes
        ``self.psi_refs``.

        ``h_matrix``, if given, is used in place of ``build_sparse_matrix(self.basis, H)``. It
        exists for a caller sweeping a *family* of Hamiltonians over one frozen determinant
        space -- the double-counting search's ``H(mu) = H(0) - mu * N_imp``, where ``N_imp`` is
        diagonal, so the whole family is one build plus a diagonal shift. Keeping the seam here
        rather than solving outside this method is deliberate: the warm-start cold-retry guard
        below (a warm block spans a near-invariant subspace and silently returns an excited state
        as "lowest" when the ground state has moved charge sector, which is exactly what a DC
        search does) has no equivalent outside, and its own comment records that the miss is
        undetectable downstream.

        The caller owns the matrix's consistency. ``H.set_restrictions`` below still runs on the
        *operator* and cannot retro-fit a matrix, so ``h_matrix`` must have been built after the
        basis's restrictions were set, from this basis, at this size -- the shape assertion
        catches a stale one, nothing catches stale restrictions.
        """
        if self.basis.restrictions is not None:
            H.set_restrictions(self.basis.restrictions)
        if self.basis.weighted_restrictions is not None:
            H.set_weighted_restrictions(self.basis.weighted_restrictions)
        if h_matrix is not None and h_matrix.shape[0] != len(self.basis):
            raise ValueError(
                f"h_matrix was built for a basis of {h_matrix.shape[0]} determinants but this one "
                f"holds {len(self.basis)}. A matrix that outlived its basis produces eigenvalues "
                "of the wrong operator rather than an error, so this is checked."
            )

        if solver in SOLVERS and self.basis.size >= dense_cutoff:
            restarted_lanczos = SOLVERS[solver]

            # Same rank-independent start vector as `expand` (see `_amplitude_from_hash`): a
            # `random.seed(42 + rank)` stream made the start vector depend on the rank count
            # *and* on the local_basis iteration order, so the Lanczos trajectory -- and the
            # eigenvector whose support seeds the next CIPSI selection -- differed between
            # serial and MPI runs of the same Hamiltonian.
            local_states = list(self.basis.local_basis)

            def cold_start_block():
                # width=1 even when local_states is empty on this rank: a bare {}
                # construction would be the width-0 polymorphic zero, which
                # block_normalize's from_states round trip below rejects (raising inside
                # the try/except, silently skipping this rank out of the collective
                # block_tsqr call other ranks still enter -- an MPI deadlock).
                return [
                    ManyBodyState({state: _amplitude_from_hash(state.get_hash()) for state in local_states}, width=1)
                ]

            warm_started = psi_refs is not None
            # A warm block spans the previously converged (near-)invariant subspace. When the
            # ground state has moved to another near-decoupled charge sector since (e.g. the
            # fixed-occupation DC search jumping across a charge-transfer crossing between
            # trials), Lanczos restarted from it converges that subspace's states only and
            # silently returns an excited state as "lowest" -- nothing is short, the boundary
            # lies beyond the thermal cut, and the miss is undetectable downstream. Appending
            # the cold full-support start vector keeps every sector reachable while the warm
            # columns retain their fast convergence.
            psi0 = list(psi_refs) + cold_start_block() if warm_started else cold_start_block()

            num_wanted = min(num_wanted + 10, len(self.basis))

            max_subspace = min(max(2 * num_wanted, num_wanted + 10), len(self.basis))
            if len(psi0) > 0:
                psi0, _ = block_normalize(psi0, self.basis.is_distributed, self.basis.comm, slaterWeightMin)

            max_subspace_blocks = 2 * int(np.ceil(max_subspace / max(1, len(psi0)))) + 20
            if len(psi0) > 0:
                max_blocks = max(2, len(self.basis) // len(psi0) - 1)
                max_subspace_blocks = min(max_subspace_blocks, max_blocks)

            num_wanted = min(num_wanted, (max_subspace_blocks - 1) * len(psi0))

            H_mat = build_sparse_matrix(self.basis, H) if h_matrix is None else h_matrix
            if self.basis.is_distributed:
                H_mat = H_mat[:, self.basis.local_indices]

            psi0_arr = (
                build_distributed_vector(self.basis, psi0).T
                if len(psi0) > 0
                else np.zeros((len(self.basis.local_basis), 1), dtype=complex)
            )

            # Solve for more and more eigenstates until at least one lands *outside* the thermal
            # cut. Only then is the boundary manifold provably complete: a degenerate manifold has
            # no preferred basis, so keeping part of one leaves the CIPSI selection, the thermal
            # average and the Green's-function seeds at the mercy of whichever rotation the
            # eigensolver returned -- which depends on the MPI rank count. Cheap in practice: the
            # first solve almost always overshoots the cut already.
            cap = len(self.basis)
            cold_retry_available = warm_started
            for _ in range(_MAX_EIGENSTATE_DOUBLINGS):
                e_ref, psi_refs_arr = restarted_lanczos(
                    psi0=psi0_arr,
                    h_op=H_mat,
                    basis=self.basis,
                    num_wanted=num_wanted,
                    max_subspace_blocks=max_subspace_blocks,
                    tol=_eigen_tol(slaterWeightMin),
                    max_restarts=100,
                    # The TRLM/IRLM kernels already rank-gate every print internally
                    # (rank0 = (not mpi) or comm.rank == 0, see _trlm.pxi/_irlm.pxi); no need
                    # to pre-AND rank 0 into the bool here too.
                    verbose=self.basis.verbose,
                    slaterWeightMin=slaterWeightMin,
                    reort=reort,
                )
                if max_energy is None or len(e_ref) == 0:
                    break
                _, need_more = _energy_cut_indices(e_ref, max_energy, tol=_degeneracy_tol(e_ref, slaterWeightMin))
                # A short return means the eigensolver exhausted the Krylov space reachable from
                # `psi0` -- it hit an invariant subspace, which a block warm-started from converged
                # eigenvectors does immediately. Doubling `num_wanted` then re-solves the *same*
                # subspace and returns the same states, at the cost of a full re-solve each time.
                exhausted = len(e_ref) < num_wanted
                # `restarted_lanczos` is collective. `e_ref` is replicated but only to roundoff,
                # so a state sitting on the cut could make ranks disagree about re-solving and
                # deadlock. Decide on rank 0 and broadcast, per the MPI rule in CLAUDE.md.
                if self.basis.is_distributed:
                    need_more, exhausted = self.basis.comm.bcast((need_more, exhausted), root=0)
                if not need_more or (exhausted and not cold_retry_available) or (not exhausted and num_wanted >= cap):
                    break
                if exhausted:
                    # The warm block spans a (near-)invariant subspace, so the missing thermal
                    # states -- typically the other charge sector of a near-degenerate crossing,
                    # e.g. the fixed-occupation DC search walking over a charge-transfer point --
                    # are unreachable from it at any num_wanted. Retry once from the cold
                    # rank-independent start vector, whose support covers the whole basis.
                    cold_retry_available = False
                    psi0 = cold_start_block()
                    psi0, _ = block_normalize(psi0, self.basis.is_distributed, self.basis.comm, slaterWeightMin)
                    psi0_arr = build_distributed_vector(self.basis, psi0).T
                else:
                    num_wanted = min(2 * num_wanted, cap)
                max_subspace = min(max(2 * num_wanted, num_wanted + 10), cap)
                max_subspace_blocks = min(
                    2 * int(np.ceil(max_subspace / max(1, len(psi0)))) + 20,
                    max(2, cap // max(1, len(psi0)) - 1),
                )
                num_wanted = min(num_wanted, (max_subspace_blocks - 1) * max(1, len(psi0)))

            if len(e_ref) > 0:
                psi_refs = build_state(self.basis, psi_refs_arr.T, slaterWeightMin=slaterWeightMin)
            if max_energy is not None and len(e_ref) > 0:
                # Rank-locally, unlike the `need_more`/`exhausted` decision twenty lines up, and
                # deliberately: this cut sets `len(e_ref)`, which callers feeding `psi_refs` into
                # `build_density_matrices` turn into an `Allreduce` buffer shape of
                # `(len(psis), n_orb, n_orb)`. Ranks disagreeing here would not return different
                # answers, they would enter one reduction with different shapes.
                #
                # It is safe because `e_ref` is bit-identical across ranks, not merely close:
                # `block_tsqr` returns a bitwise-identical R everywhere (see TSQR.pyx), so the
                # alphas/betas are identical (_lanczos_step.pxi), and the Ritz values are a
                # rank-local `eigh` on identical bytes. `_energy_cut_indices` is then a pure
                # function of identical input. Measured, not assumed: the full suite at -n 2 and
                # -n 3 allgathered `e_ref.tobytes()` from every call -- 7036 and 7040 checked, 375
                # and 379 of them through this Krylov branch -- with zero divergence in either the
                # length or the bytes. `_degeneracy_tol` is part of that pure function: it reads
                # only `max|e_ref|` and `slaterWeightMin` (a replicated argument), so it inherits
                # the same bit-identity rather than adding a new way for ranks to disagree.
                #
                # Two things that measurement does not cover, so do not widen the claim: the
                # suite's Krylov cases are small (no production-cap solve with partial
                # reorthogonalization), and it ran at OPENBLAS_NUM_THREADS=1 throughout. A
                # multithreaded LAPACK whose thread count differed between ranks could break the
                # determinism this relies on.
                valid_idx, need_more = _energy_cut_indices(
                    e_ref, max_energy, tol=_degeneracy_tol(e_ref, slaterWeightMin)
                )
                if need_more and (self.basis.comm is None or self.basis.comm.rank == 0):
                    print(
                        f"warning: every one of the {len(e_ref)} computed eigenstates falls inside "
                        f"the thermal energy cut, so the boundary manifold cannot be shown to be "
                        f"complete. A partially-kept degenerate manifold has no rotation-invariant "
                        f"basis and makes the results depend on the MPI rank count.",
                        flush=True,
                    )
                e_ref = e_ref[valid_idx]
                psi_refs = [psi_refs[i] for i in valid_idx]

        else:
            H_mat = build_sparse_matrix(self.basis, H) if h_matrix is None else h_matrix
            e_ref, psi_ref_dense = eigensystem(
                H_mat,
                e_max=max_energy,
                k=num_wanted,
                e0=None,
                v0=None,
                eigenValueTol=0,
                comm=self.basis.comm,
                dense=self.basis.size < dense_cutoff,
                return_eigvecs=True,
            )
            psi_refs = build_state(self.basis, psi_ref_dense.T, slaterWeightMin=slaterWeightMin)

        return e_ref, psi_refs
