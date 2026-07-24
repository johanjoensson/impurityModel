import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed.basis_transcription import (
    build_distributed_vector,
    build_sparse_matrix,
    build_state,
    build_vector,
)
from impurityModel.ed.BlockLanczosArray import Reort, block_normalize
from impurityModel.ed.eigensolvers import eigensystem
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.manybody_basis import Basis, collective_amplitude_cutoff
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
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


#: Reference eigenvalues closer than this are treated as one degenerate manifold when scoring
#: CIPSI candidates. A restarted Lanczos converges the invariant *subspace*, not a basis within
#: it -- every rotation of a degenerate block has the same residual -- so any per-eigenvector
#: score must be summed over the block to be well defined. Chosen well above the eigensolver's
#: attainable eigenvalue accuracy and far below any physical splitting of interest.
DEGENERACY_TOL = 1e-9

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


def _degenerate_groups(e_ref, tol=DEGENERACY_TOL):
    """Partition reference-state indices into runs of (near-)degenerate eigenvalues.

    ``e_ref`` comes from the eigensolver in ascending order, so a single forward scan suffices.
    Returns a list of index lists, one per manifold.
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
    while n_keep < len(e_sorted) and abs(e_sorted[n_keep] - e_sorted[n_keep - 1]) <= tol:
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

    def truncate_initial(self, H: ManyBodyOperator) -> None:
        """Perform an initial truncation if the basis exceeds the truncation threshold."""
        if self.basis.size > self.basis.truncation_threshold and H is not None:
            if self.basis.verbose:
                print("Truncating basis!")
            H_sparse = build_sparse_matrix(self.basis, H)
            # Rank over the low-energy manifold (up to 10 states within energy_cut), not a
            # single eigenvector: the downstream expansion / get_eigenvectors keep the whole
            # near-degenerate ground manifold, so truncating to one state's support would
            # bias the retained determinants toward one member of the multiplet.
            _e_ref, psi_ref = eigensystem(
                H_sparse,
                e_max=-self.basis.tau * np.log(1e-4),
                k=min(10, self.basis.size),
                eigenValueTol=0,
                comm=self.basis.comm,
                dense=False,
            )
            # eigensystem returns eigenvectors as columns (N, k); build_state wants one
            # row per state vector, so transpose (matches CIPSISolver.expand's usage).
            self.truncate(build_state(self.basis, psi_ref.T))

    def truncate(self, psis: list[ManyBodyState], target=None) -> list[ManyBodyState]:
        """Keep the globally top-``target`` determinants by eigenvector amplitude.

        Importance is the max ``|amplitude|^2`` over ``psis`` (each determinant counted
        once, on its hash-owner rank -- ``psis`` must be redistributed). The cutoff comes
        from the collective amplitude bisection, so every rank retains the identical set
        and the global count never exceeds ``target`` (ties at the cutoff are
        under-admitted). Collective on ``basis.comm``.
        """
        if target is None:
            target = self.basis.truncation_threshold
        blk = ManyBodyState.from_states(list(psis))
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

        One observable difference from the old ``redistribute_psis``-based path: the
        final ``to_states()`` drops a determinant from column ``i`` only when its
        *own* post-sum amplitude is exact zero, never when some *other* reference
        column merely happens to carry the nonzero coupling (the old union-support
        ``column(i)`` padded every output to the full merged support, so an exact
        cross-rank cancellation on column ``i`` alone still left the entry visible via
        iteration). A determinant is therefore only ever dropped from consideration in
        ``_candidate_overlaps_and_energies`` when it has zero net coupling to *every*
        reference column -- a genuine selection-rule cancellation, which is correctly a
        non-candidate rather than a zero-scored one.
        """
        cols = H.apply_block(ManyBodyState.from_states(psi_ref), cutoff).to_states()
        for s in cols:
            s.prune_rows(cutoff)
        merged = self.basis.redistribute_block(ManyBodyState.from_states(cols))
        return merged.to_states()

    def _candidate_overlaps_and_energies(self, H, Hpsi_ref, slaterWeightMin: float = 0):
        """Enumerate the out-of-basis candidates of ``Hpsi_ref`` with couplings and energies.

        The machinery every CIPSI-style selection shares (the ground-state
        :meth:`_calc_de2` and the resolvent-targeted :meth:`select_at` differ only in the
        importance denominator applied on top): this rank's hash-owned candidate
        determinants ``local_Djs`` (sorted, so rank-independent given ``Hpsi_ref`` is
        redistributed), the coupling matrix ``overlaps[i, j] = <Dj | H | psi_i>`` read off
        ``Hpsi_ref``, and the diagonal-probe energies ``e_Dj[j] ~ <Dj|H|Dj>``.

        Collective on ``basis.comm`` (the probe redistribution), so it must run on every
        rank -- including one that owns no candidates, whose arrays come back empty.
        """
        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        _index_dict = self.basis._index_dict
        local_Djs = sorted({state for hp in Hpsi_ref for state in hp if state not in _index_dict})

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
        # construct the width-0 polymorphic zero, which redistribute_psis's width guard
        # rejects on this rank only while a rank with real candidates sails into the
        # matching collective -- the exact asymmetric-exception deadlock fixed in Phase
        # 7 step 2b.
        psi_all_Dj = ManyBodyState({Dj: phases[j] for j, Dj in enumerate(local_Djs)}, width=1)
        H_psi_all = applyOp_test(H, psi_all_Dj, cutoff=slaterWeightMin)
        if self.basis.is_distributed:
            H_psi_all = self.basis.redistribute_psis([H_psi_all])[0]

        if not local_Djs:
            return local_Djs, np.zeros((len(Hpsi_ref), 0), dtype=complex), np.zeros(0, dtype=float)

        Dj_index = {Dj: j for j, Dj in enumerate(local_Djs)}
        overlaps = np.zeros((len(Hpsi_ref), len(local_Djs)), dtype=complex)
        for i, Hpsi_i in enumerate(Hpsi_ref):
            for state, amp in Hpsi_i.items():
                j = Dj_index.get(state)
                if j is not None:
                    overlaps[i, j] = amp[0]
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
            scores = np.max(np.stack([de2_abs[g, :].sum(axis=0) for g in _degenerate_groups(e_ref)]), axis=0)
        else:
            scores = np.zeros(0)
        de2_mask, selection_stats = self._admit_top(scores, scores >= de2_min, max_new)
        self.last_selection = selection_stats
        new_Dj = set(itertools.compress(local_Djs, de2_mask))

        if gen_ops:
            unexplored_list = list(new_Dj)
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

                        for state, row in psi_op.items():
                            if state not in new_Dj:
                                new_Dj.add(state)
                                next_amps[state] = row[0]

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
        de0_max = -self.basis.tau * np.log(1e-4)
        psi_refs = None

        if isinstance(H, dict):
            H = ManyBodyOperator(H)

        from impurityModel.ed.symmetries import (
            discover_one_body_symmetries,
            extract_tensors,
            tensors_to_operator,
        )

        if symmetry_generators is None:
            imp_orbs = []
            if getattr(self.basis, "impurity_orbitals", None):
                for orbs in self.basis.impurity_orbitals.values():
                    for o in orbs:
                        imp_orbs.extend(o)

            h, _, _ = extract_tensors(H, two_body=False)

            if imp_orbs:
                imp = sorted(set(imp_orbs))
                h_imp = h[np.ix_(imp, imp)]
                imp_generators = discover_one_body_symmetries(h_imp)

                imp_map = {}
                for group, imp_blocks in self.basis.impurity_orbitals.items():
                    imp_blk = imp_blocks[0]
                    for idx_in_grp, o in enumerate(imp_blk):
                        imp_map[o] = (group, idx_in_grp)

                generators = []
                for g_imp in imp_generators:
                    g_full = np.zeros_like(h)
                    for i, oi in enumerate(imp):
                        for j, oj in enumerate(imp):
                            g_full[oi, oj] = g_imp[i, j]

                    for bath_dict in self.basis.bath_states:
                        if not bath_dict:
                            continue
                        n_bath = max([len(blks) for blks in bath_dict.values()], default=0)

                        for k in range(n_bath):
                            site_k_map = {}
                            for i, o_imp in enumerate(imp):
                                group, idx_in_grp = imp_map[o_imp]
                                if (
                                    group in bath_dict
                                    and k < len(bath_dict[group])
                                    and idx_in_grp < len(bath_dict[group][k])
                                ):
                                    site_k_map[i] = bath_dict[group][k][idx_in_grp]

                            for i in range(len(imp)):
                                if i not in site_k_map:
                                    continue
                                oi = site_k_map[i]
                                for j in range(len(imp)):
                                    if j not in site_k_map:
                                        continue
                                    oj = site_k_map[j]
                                    g_full[oi, oj] = g_imp[i, j]

                    if np.linalg.norm(h @ g_full - g_full @ h) < 1e-9:
                        generators.append(g_full)
            else:
                generators = discover_one_body_symmetries(h)
        else:
            generators = symmetry_generators

        gen_ops = []
        for g in generators:
            op = tensors_to_operator(g, tol=1e-12)
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
        self.truncation_report = None
        while True:
            if solver in SOLVERS and self.basis.size >= dense_cutoff:
                restarted_lanczos = SOLVERS[solver]

                if psi_refs is None:
                    # Seed each determinant's amplitude from its *own* hash rather than from a
                    # per-rank RNG stream. A `random.seed(42 + rank)` start vector depends on
                    # how the determinants happen to be partitioned, so the adaptively-selected
                    # CIPSI basis -- and every excited basis and Green's function derived from
                    # it -- differed with the rank count (measured: 532/566/526 determinants at
                    # 1/2/3 ranks, and G shifting by 1e-7). The energy was reproducible, which
                    # is why the tests never caught it. Deriving the amplitude from the
                    # determinant makes the global start vector identical at any rank count,
                    # with no scatter needed.
                    local_states = list(self.basis.local_basis)
                    psi0_dict = {state: _amplitude_from_hash(state.get_hash()) for state in local_states}
                    # An empty-local-basis rank must not reach from_states with a bare
                    # ManyBodyState({}) element -- that is the width-0 polymorphic zero,
                    # which from_states rejects (every element must already be width 1).
                    # Passing width=1 explicitly works whether psi0_dict is empty or not,
                    # giving every rank the same representation into redistribute_psis'
                    # collective instead of an asymmetric width-0-vs-width-1 split.
                    psi0_blk = ManyBodyState.from_states([ManyBodyState(psi0_dict, width=1)])
                    psi0 = self.basis.redistribute_psis([psi0_blk])[0].to_states()

                    N2s = np.array([psi.norm2() for psi in psi0], dtype=float)
                    if self.basis.is_distributed:
                        self.basis.comm.Allreduce(MPI.IN_PLACE, N2s, op=MPI.SUM)
                    psi0 = [psi / np.sqrt(N2s[i]) if N2s[i] > 0 else psi for i, psi in enumerate(psi0)]
                else:
                    psi0 = psi_refs

                num_wanted = min(2 * len(psi_refs) if psi_refs is not None else 10, len(self.basis))

                max_subspace = min(max(2 * num_wanted, num_wanted + 10), len(self.basis))
                if len(psi0) > 0:
                    try:
                        psi0, _ = block_normalize(psi0, self.basis.is_distributed, self.basis.comm, slaterWeightMin)
                    except Exception:
                        pass

                max_subspace_blocks = 2 * int(np.ceil(max_subspace / max(1, len(psi0)))) + 20
                if len(psi0) > 0:
                    max_blocks = max(2, len(self.basis) // len(psi0) - 1)
                    max_subspace_blocks = min(max_subspace_blocks, max_blocks)

                num_wanted = min(num_wanted, (max_subspace_blocks - 1) * len(psi0))

                # Iterative CIPSI uses the *array* IRLM (sparse H_mat, column-distributed
                # via local_indices). Memory note: the array MPI matvec forms the full
                # (global_N, n) dense partial product on every rank before the Allreduce
                # (see the guardrail in BlockLanczosArray.pyx), so per-rank memory scales
                # with the *global* sector dimension, not the local partition. This path is
                # for sectors small enough to also hold the sparse H_mat; very large sectors
                # should use the hash-distributed ManyBodyState kernel instead.
                H_mat = build_sparse_matrix(self.basis, H)
                if self.basis.is_distributed:
                    H_mat = H_mat[:, self.basis.local_indices]
                psi0_arr = (
                    build_distributed_vector(self.basis, psi0).T
                    if len(psi0) > 0
                    else np.zeros((len(self.basis.local_basis), 1), dtype=complex)
                )

                e_ref, psi_refs_arr = restarted_lanczos(
                    psi0=psi0_arr,
                    h_op=H_mat,
                    basis=self.basis,
                    num_wanted=num_wanted,
                    max_subspace_blocks=max_subspace_blocks,
                    tol=de2_min / 10,
                    max_restarts=10,
                    verbose=False,
                    slaterWeightMin=slaterWeightMin,
                    reort=reort,
                )

                if len(e_ref) > 0:
                    psi_refs = build_state(self.basis, psi_refs_arr.T, slaterWeightMin=slaterWeightMin)
                    e_min = np.min(e_ref)
                    valid_idx = [i for i, e in enumerate(e_ref) if e - e_min <= de0_max]
                else:
                    valid_idx = []
                if not valid_idx:
                    if self.basis.verbose:
                        print("No eigenvalues below energy threshold found.")
                    break
                e_ref = e_ref[valid_idx]
                psi_refs = [psi_refs[i] for i in valid_idx]
            else:
                H_mat = build_sparse_matrix(self.basis, H)
                v0 = (
                    build_vector(self.basis, psi_refs).T
                    if psi_refs is not None and self.basis.size >= dense_cutoff
                    else None
                )
                e_ref, psi_ref_dense = eigensystem(
                    H_mat,
                    e_max=de0_max,
                    k=2 * len(psi_refs) if psi_refs is not None else 10,
                    e0=None,
                    v0=v0,
                    eigenValueTol=0,
                    comm=self.basis.comm,
                    dense=self.basis.size < dense_cutoff,
                )
                psi_refs = build_state(self.basis, psi_ref_dense.T)

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
                psi_refs = self.truncate(psi_refs, target=keep)
                if self.basis.verbose:
                    print(
                        f"------> Basis truncated! (cycle {cap_cycles}: kept "
                        f"{self.last_truncation['retained']:,} determinants, admitting {n_new:,} candidates)"
                    )
            self.basis.add_states(new_Dj)
            psi_refs = self.basis.redistribute_psis(psi_refs)
            if cap_cycles == 0 and self.basis.size == old_size:
                break
            e0 = np.inf  # the basis was mutated; e0 no longer describes it
        if best_basis is not None and e0 > best_e0:
            # The last refinement cycle left a worse basis (e.g. score/amplitude
            # ping-pong): restore the best capped basis seen during the cycles.
            self.basis.clear()
            self.basis.add_states(best_basis)
            psi_refs = self.basis.redistribute_psis(best_psis)
        self.psi_refs = psi_refs
        if capped and self.basis.size > threshold and self.psi_refs is not None:
            # The symmetry closure can push an admission slightly past the cap; enforce
            # the hard threshold on exit (downstream get_eigenvectors re-solves).
            self.psi_refs = self.truncate(self.psi_refs)
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
            if self.basis.verbose and rank == 0:
                rep = self.truncation_report
                print(
                    f"GS basis cap hit: fixed-budget CIPSI held the basis at "
                    f"{rep['retained']:,} determinants (truncation_threshold={rep['threshold']:,}) "
                    f"over {rep['cycles']} refinement cycle(s); the last cycle discarded "
                    f"candidates carrying {rep['discarded_de2_mass']:.3e} of PT2 importance. "
                    f"The ground state is exact on the retained subspace.",
                    flush=True,
                )

        if self.basis.verbose:
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
    ):
        if self.basis.restrictions is not None:
            H.set_restrictions(self.basis.restrictions)
        if self.basis.weighted_restrictions is not None:
            H.set_weighted_restrictions(self.basis.weighted_restrictions)

        if solver in SOLVERS and self.basis.size >= dense_cutoff:
            restarted_lanczos = SOLVERS[solver]

            # Same rank-independent start vector as `expand` (see `_amplitude_from_hash`): a
            # `random.seed(42 + rank)` stream made the start vector depend on the rank count
            # *and* on the local_basis iteration order, so the Lanczos trajectory -- and the
            # eigenvector whose support seeds the next CIPSI selection -- differed between
            # serial and MPI runs of the same Hamiltonian.
            local_states = list(self.basis.local_basis)
            if hasattr(self, "psi_refs") and self.psi_refs is not None:
                psi0 = self.psi_refs
            else:
                # width=1 even when local_states is empty on this rank: a bare {}
                # construction would be the width-0 polymorphic zero, which
                # block_normalize's from_states round trip below rejects (raising inside
                # the try/except, silently skipping this rank out of the collective
                # block_tsqr call other ranks still enter -- an MPI deadlock).
                psi0 = [
                    ManyBodyState({state: _amplitude_from_hash(state.get_hash()) for state in local_states}, width=1)
                ]

            num_wanted = min(num_wanted + 10, len(self.basis))

            max_subspace = min(max(2 * num_wanted, num_wanted + 10), len(self.basis))
            if len(psi0) > 0:
                try:
                    psi0, _ = block_normalize(psi0, self.basis.is_distributed, self.basis.comm, slaterWeightMin)
                except Exception:
                    pass

            max_subspace_blocks = 2 * int(np.ceil(max_subspace / max(1, len(psi0)))) + 20
            if len(psi0) > 0:
                max_blocks = max(2, len(self.basis) // len(psi0) - 1)
                max_subspace_blocks = min(max_subspace_blocks, max_blocks)

            num_wanted = min(num_wanted, (max_subspace_blocks - 1) * len(psi0))

            H_mat = build_sparse_matrix(self.basis, H)
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
            for _ in range(_MAX_EIGENSTATE_DOUBLINGS):
                e_ref, psi_refs_arr = restarted_lanczos(
                    psi0=psi0_arr,
                    h_op=H_mat,
                    basis=self.basis,
                    num_wanted=num_wanted,
                    max_subspace_blocks=max_subspace_blocks,
                    tol=_eigen_tol(slaterWeightMin),
                    max_restarts=100,
                    verbose=self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0),
                    slaterWeightMin=slaterWeightMin,
                    reort=reort,
                )
                if max_energy is None or len(e_ref) == 0:
                    break
                _, need_more = _energy_cut_indices(e_ref, max_energy)
                # A short return means the eigensolver exhausted the Krylov space reachable from
                # `psi0` -- it hit an invariant subspace, which a block warm-started from converged
                # eigenvectors does immediately. Doubling `num_wanted` then re-solves the *same*
                # subspace and returns the same states, at the cost of a full re-solve each time.
                # Stop and let the warning below report the uncertified manifold.
                exhausted = len(e_ref) < num_wanted
                # `restarted_lanczos` is collective. `e_ref` is replicated but only to roundoff,
                # so a state sitting on the cut could make ranks disagree about re-solving and
                # deadlock. Decide on rank 0 and broadcast, per the MPI rule in CLAUDE.md.
                if self.basis.is_distributed:
                    need_more, exhausted = self.basis.comm.bcast((need_more, exhausted), root=0)
                if not need_more or exhausted or num_wanted >= cap:
                    break
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
                valid_idx, need_more = _energy_cut_indices(e_ref, max_energy)
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
            H_mat = build_sparse_matrix(self.basis, H)
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
