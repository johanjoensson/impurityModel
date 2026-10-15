"""Resource policy of :meth:`CIPSISolver.expand`: the memory guard and the fixed-budget refinement.

* :class:`MemoryGuard` -- the determinant cap (a user cap, one derived from memory, or none) and the two
  measured-RSS mechanisms that may tighten it: a look-ahead bound on each selection round's admission, and
  a trip-wire on the process high-water mark.
* :class:`CapRefiner` -- under a binding cap, keeps the best basis seen across refinement cycles and
  decides when to stop.

Both are driven by replicated inputs only (global sizes, ``MAX``-reduced RSS), so every rank takes the same
decision without a collective of its own.
"""

import numpy as np

from impurityModel.ed.manybody_basis import Basis
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.memory_estimate import emit_memory_warning, format_bytes


class MemoryGuard:
    """The determinant cap of one expansion and the memory measurements that may tighten it.

    Parameters
    ----------
    basis : Basis
        The basis being expanded; ``basis.truncation_threshold`` is written back whenever the cap
        tightens, so a caller inspecting it afterwards sees the cap that actually governed the run.
    budget_bytes : int or None
        Per-rank RSS budget. ``None`` disables both mechanisms.
    tighten : bool
        ``True``: the mechanisms may lower the cap (a cap derived from memory, or none). ``False`` (policy
        ``"warn"``): the user's cap is final; the first time the budget is exceeded a warning is recorded in
        :attr:`warning` and printed, nothing is limited.

    Attributes
    ----------
    threshold, capped
        The cap in force (``inf`` when none) and whether it is finite.
    bound
        A look-ahead bound became a *permanent* cap (two consecutive binding rounds, or zero affordable
        growth).
    observed
        Any memory limit was observed -- a binding round, or the trip-wire. This is what the truncation
        report keys off, so a memory-limited expansion is never reported as converged.
    """

    def __init__(self, basis: Basis, budget_bytes, tighten: bool):
        self.basis = basis
        self.budget_bytes = budget_bytes
        self.tighten = tighten
        self.threshold = basis.truncation_threshold
        self.capped = bool(np.isfinite(self.threshold))
        self.bound = False
        self.observed = False
        self.warning = None
        self._tripped = False
        self._streak = 0

    @property
    def _root(self):
        return self.basis.comm is None or self.basis.comm.rank == 0

    def _lower(self, to):
        """Tighten the cap to ``to`` (never loosen an existing one); returns the previous cap."""
        previous = self.threshold
        self.threshold = min(float(self.threshold), float(to))
        self.capped = True
        self.basis.truncation_threshold = self.threshold
        return previous

    @staticmethod
    def _was(previous):
        return "no cap (unlimited)" if not np.isfinite(previous) else f"the auto cap of {int(previous):,}"

    def admission_target(self):
        """Cap one cycle's selection so it turns over at most ~10% of the basis, or fills the remaining budget."""
        if not self.capped:
            return None
        return max(int(self.threshold) - self.basis.size, -(-int(self.threshold) // 10))

    def after_selection(self, sel, old_size, admit_target):
        """Account for a selection round whose look-ahead cap was ``sel["memory_admit_cap"]``.

        The memory bound decided the admission only if it was the tightest of the limits in force
        (candidates above ``de2_min``, ``admit_target``, itself). One binding round already limited *that*
        round's admission; a permanent tightening needs two in a row, so one noisy transient cannot freeze
        the basis. A cap of zero affordable growth locks in at once: that round admitted nothing, so the
        expansion ends before a second could confirm it.
        """
        memory_cap = sel.get("memory_admit_cap")
        round_bound = (
            self.tighten
            and memory_cap is not None
            and memory_cap < sel.get("n_candidates", 0)
            and (admit_target is None or memory_cap < admit_target)
        )
        self._streak = self._streak + 1 if round_bound else 0
        self.observed = self.observed or round_bound
        if self.bound or not round_bound or not (memory_cap == 0 or self._streak >= 2):
            return
        self.bound = True
        previous = self._lower(old_size + memory_cap)
        rss_now = sel.get("round_rss_bytes", 0)
        budget = self.budget_bytes
        if sel.get("memory_cap_reason") == "budget":
            why = (
                f"the resident set alone is over the {format_bytes(budget)} budget by "
                f"{format_bytes(rss_now - budget)}, so the round's own cost did not enter into it"
            )
        else:
            why = (
                f"that leaves {format_bytes(budget - rss_now)} under the {format_bytes(budget)} budget, "
                "which this round's cost does not fit a larger basis into"
            )
        streak_note = "no growth at all is affordable" if memory_cap == 0 else "for the second round running"
        emit_memory_warning(
            f"WARNING determinant cap: the selection round on {old_size:,} determinants peaked "
            f"{format_bytes(sel.get('round_transient_bytes', 0))} above its {format_bytes(rss_now)} resident "
            f"set ({streak_note}); {why}. The next round can afford {memory_cap:,} of the "
            f"{sel.get('n_candidates', 0):,} candidates. Tightening {self._was(previous)} to "
            f"{int(self.threshold):,}.",
            root=self._root,
            kind="gs-memory",
            force=self.basis.verbose,
        )

    def after_cycle(self, peak_rss, anon_rss, shmem_rss):
        """The after-the-fact trip-wire: the first cycle whose measured peak reaches the budget.

        Arguments are the ``MAX`` over ranks, so every rank evaluates the same condition. In ``"warn"`` mode
        the user's cap is kept and a warning is recorded; otherwise the cap tightens to the current basis
        size (never loosening an existing one) and the expansion is reported memory-bound.
        """
        budget = self.budget_bytes
        if budget is None or self._tripped or peak_rss < budget:
            return
        self._tripped = True
        if not self.tighten:
            self.warning = {
                "peak_rss_bytes": int(peak_rss),
                "anon_rss_bytes": int(anon_rss),
                "shmem_rss_bytes": int(shmem_rss),
                "budget_bytes": int(budget),
                "basis_size": int(self.basis.size),
                "cap": float(self.threshold),
            }
            cap_text = f"{int(self.threshold):,}" if np.isfinite(self.threshold) else "unlimited"
            emit_memory_warning(
                f"WARNING determinant cap: GS basis at {self.basis.size:,} determinants uses "
                f"{format_bytes(peak_rss)}/rank (anon {format_bytes(anon_rss)} + shm {format_bytes(shmem_rss)}, "
                f"max over ranks) >= the {format_bytes(budget)} memory budget. Your "
                f"truncation_threshold={cap_text} is kept as set; if the job is killed, set "
                f"truncation_threshold <= {self.basis.size:,}, use 'auto', or run fewer ranks per node. "
                "(Shared memory may be over-counted under MPI.)",
                root=self._root,
                kind="gs-memory",
            )
            return
        # Only `observed` (a reporting flag), never `bound`: whether a look-ahead may still adopt a tighter
        # cap later is unaffected by this backstop having fired.
        self.observed = True
        previous = self._lower(self.basis.size)
        emit_memory_warning(
            f"WARNING determinant cap: measured per-rank RSS {format_bytes(peak_rss)} reached the "
            f"{format_bytes(budget)} memory budget mid-expansion; tightening {self._was(previous)} to a "
            f"fixed-budget cap at the current basis ({self.basis.size:,} determinants) rather than risk an "
            "uncatchable OOM kill.",
            root=self._root,
            kind="gs-memory",
            force=self.basis.verbose,
        )

    @property
    def limited(self):
        """Whether memory limited the expansion (permanently or only observed)."""
        return self.bound or self.observed


class CapRefiner:
    """Fixed-budget refinement: the best capped basis seen so far, and when to stop cycling.

    Under a binding cap each cycle drops the least important determinants and admits the best candidates;
    the energy can ping-pong, so the best basis (and its eigenvectors) is snapshotted and restored when the
    last cycle left a worse one. The snapshot is the packed key array (9.8 B/determinant, against 80 for a
    list of determinants), since it is retained across every later cycle alongside the live basis.
    """

    def __init__(self, e_tol, max_cycles):
        self.e_tol = e_tol
        self.max_cycles = max_cycles
        self.best_e0 = None
        self._keys = None
        self._psis = None
        self._e_ref = None
        self._no_improve = 0

    def observe(self, basis: Basis, e0, psi_refs, e_ref, cycles):
        """Record this cycle's lowest energy; True when refinement should stop (no gain twice, or out of cycles)."""
        improved = self.best_e0 is None or e0 < self.best_e0 - self.e_tol
        if self.best_e0 is None or e0 < self.best_e0:
            self.best_e0 = e0
            keys = np.empty((len(basis.local_basis), basis.n_bytes), dtype=np.uint8)
            for row, state in enumerate(basis.local_basis):
                keys[row] = np.frombuffer(bytes(state.to_bytearray()[: basis.n_bytes]), dtype=np.uint8)
            self._keys, self._psis, self._e_ref = keys, psi_refs, e_ref
        self._no_improve = 0 if improved else self._no_improve + 1
        return self._no_improve >= 2 or cycles >= self.max_cycles

    def restore_if_worse(self, basis: Basis, e0):
        """Put the best basis back when the last cycle left a worse one: ``(psi_refs, e_ref)``, else ``None``."""
        if self._keys is None or not e0 > self.best_e0:
            return None
        basis.clear()
        basis.add_states(self._keys[row].tobytes() for row in range(self._keys.shape[0]))
        psis = self._psis
        psi_refs = basis.redistribute_block(
            psis if isinstance(psis, ManyBodyState) else ManyBodyState.from_states(list(psis))
        )
        psi_refs.prune_rows(0.0)
        return psi_refs, self._e_ref
