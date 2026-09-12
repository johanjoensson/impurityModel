import itertools

import numpy as np
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.average import energy_cut
from impurityModel.ed.basis_transcription import (
    build_distributed_vector,
    build_sparse_matrix,
    build_state,
)
from impurityModel.ed.BlockLanczosArray import BlockBreakdown, Reort, block_normalize
from impurityModel.ed.eigensolvers import eigensystem
from impurityModel.ed.irlm import implicitly_restarted_block_lanczos_cy
from impurityModel.ed.manybody_basis import Basis, collective_amplitude_cutoff
from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, ManyBodyState
from impurityModel.ed.ManyBodyUtils import applyOp as applyOp_test
from impurityModel.ed.memory_estimate import current_rss_bytes, format_bytes, peak_rss_bytes, reset_peak_rss
from impurityModel.ed.solver_basis import get_symmetry_generators
from impurityModel.ed.solver_trace import note as _trace_note
from impurityModel.ed.solver_trace import timed as _trace_timed
from impurityModel.ed.trlm import thick_restart_block_lanczos

SOLVERS = {
    "trlm": thick_restart_block_lanczos,
    "irlm": implicitly_restarted_block_lanczos_cy,
}

#: Extra eigenstates requested beyond what the caller wants, so that a state landing outside the
#: thermal cut can certify the kept manifold is complete. Bought for their energies, not their
#: eigenvectors -- see `num_required` in `get_eigenvectors`.
_EIGENSTATE_PAD = 10

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


def _manifold_request(n_kept, prev_kept):
    """How many eigenstates one CIPSI cycle should ask for, given what the last one kept.

    ``expand`` used to ask for ``2 * n_kept``. The *request*, not the kept count, is what sizes
    the eigensolver -- :func:`_size_subspace` turns it into roughly ``4 * num_wanted`` retained
    Krylov columns, and :meth:`CIPSISolver.get_eigenvectors` holds that many residuals to ``tol``
    -- so the doubling was a permanent tax. What it bought was keeping
    ``get_eigenvectors``' ``need_more`` re-solve loop from ever firing, and that loop was measured
    dead on every real workload precisely *because* of the doubling: a permanent 2x against an
    occasional re-solve is the wrong side of the trade.

    So: ask for what the last cycle actually kept, plus room to grow at twice the rate it last
    grew at, floored at :data:`_EIGENSTATE_PAD`. An undershoot is not a wrong answer, it is one
    extra solve -- ``need_more`` fires and ``get_eigenvectors`` re-solves, the mechanism that has
    always been there for it.

    Two conventions that are load-bearing rather than cosmetic:

    * ``prev_kept is None`` means *no history*, and is **not** the same as ``0``. A first cycle has
      no predecessor to extrapolate from. On the SrMnO3 double-counting reproduction cycle 0 runs
      the *dense* branch, which ignores ``num_wanted`` entirely when given a cut and returns every
      state inside it -- 86 of the 120-determinant seed basis, a property of a basis too small to
      resolve the 0.23 eV window rather than of the physics. Scoring that as growth from zero
      would size the next cycle at 258, larger than the 182 this function exists to cut.
    * ``max(0, ...)`` on the growth: a *shrinking* manifold gets the flat pad, never a negative
      margin. Shrinking is the common case once the seed basis is left behind (86 -> 78 -> 10 on
      that same run), and it does mean a cycle following a large one still requests off the large
      count. Deliberate: the request has to cover the manifold that is actually there, and only
      the next cycle can know it shrank.

    Parameters
    ----------
    n_kept : int
        States the previous cycle kept inside the thermal cut.
    prev_kept : int or None
        What the cycle before *that* kept, or ``None`` when there was none.

    Returns
    -------
    int
        The unpadded request. ``get_eigenvectors`` adds its own ``_EIGENSTATE_PAD`` on top, for a
        different purpose (certifying the boundary manifold is whole, not covering growth).
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

    Returns ``callable(transient_bytes, rss_bytes) -> int | None``: the affordable admission, or
    ``None`` when the transient was not measurable (``<= 0``).
    """
    p_ratio = max(1.0, float(p_next) / max(1, int(p_now)))
    budget_bytes = float(budget_bytes)
    basis_size = int(basis_size)

    def affordable(transient_bytes, rss_bytes):
        if transient_bytes <= 0:
            return None
        headroom = budget_bytes - float(rss_bytes)
        if headroom <= 0.0:
            return 0
        next_max = basis_size * headroom / (float(transient_bytes) * p_ratio)
        return max(0, int(next_max) - basis_size)

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

    if max_energy is None:
        return [int(i) for i in order], False

    n_keep = max(int(np.count_nonzero(e_sorted - e_sorted[0] <= max_energy)), 1)
    # Never split a manifold: absorb any state degenerate with the last kept one.
    original_n_keep = n_keep
    while n_keep < len(e_sorted) and abs(e_sorted[n_keep] - e_sorted[original_n_keep - 1]) <= tol:
        n_keep += 1
    return [int(i) for i in order[:n_keep]], n_keep == len(e_sorted)


def _describe_block_health(psi0, comm):
    """Why a start block failed to orthonormalize: corrupted, or merely empty.

    ``tsqr``'s two failure codes answer this at the level of the whole block (non-finite factor
    vs numerically zero); this answers it *per column*, which is what says whether the warm
    eigenvectors arrived corrupted or the basis underneath them is gone. Both numbers are global
    -- a column is non-finite if any rank's slice of it is, and empty only if every rank's is --
    so this is **collective on** ``comm`` and every rank must call it.

    Only ever called on a failure path, so the full pass over the block costs nothing in a
    healthy run.

    Both reductions are sized ``len(psi0)`` -- the number of *states* -- while the measurement
    below reads ``amps.shape[1]``, the block's total *width*. Those agree, and the ``all-zero``
    count is arithmetic on the same length, only because ``ManyBodyState.from_states`` rejects
    anything but width-1 columns: a wider column would raise rather than silently widen the
    block. That raise lands in the ``except`` below, which leaves the buffers at their
    ``len(psi0)`` shape, so the length stays rank-invariant on every path through here -- which
    is what the reductions require.

    The local measurement is wrapped because **the three reductions below must be reached by
    every rank unconditionally**. This runs inside the recovery branch -- the least-exercised
    path in ``get_eigenvectors`` -- on a block already known to be malformed, and on ranks whose
    row partition may be empty. A rank that raised its way out of a diagnostic while the others
    waited in its ``allreduce`` would turn a report about a deadlock-free failure into a deadlock
    (the extracted-helper class CLAUDE.md's MPI rules name). A rank that cannot measure its own
    slice contributes nothing to the OR and nothing to the row count, which is the right neutral
    element for both, and the answer stays whatever the other ranks could see.
    """
    local_bad = np.zeros(len(psi0), np.int64)
    local_nonzero = np.zeros(len(psi0), np.int64)
    rows = np.int64(0)
    try:
        block = ManyBodyState.from_states(list(psi0))
        amps = np.asarray(block) if len(block) else np.zeros((0, len(psi0)), dtype=complex)
        rows = np.int64(len(block))
        if amps.size:
            # Per column, on this rank's rows: corrupted anywhere makes the column corrupted, so
            # these combine with OR (MAX over 0/1); nonzero anywhere makes it non-empty, likewise.
            local_bad = (~np.isfinite(amps)).any(axis=0).astype(np.int64)
            local_nonzero = (amps != 0).any(axis=0).astype(np.int64)
    except Exception:
        pass
    if comm is not None and comm.size > 1:
        # Buffer-based `Allreduce`, NOT the lowercase object form, and that is load-bearing:
        # mpi4py implements `comm.allreduce(obj, op)` as gather-to-root, apply `op` in PYTHON on
        # the root, broadcast. With a numpy array operand, `MPI.MAX` reduces to a Python `max()`
        # on arrays, which raises "truth value of an array is ambiguous" -- **on rank 0 only**,
        # while every other rank sits in the following broadcast. Measured, not reasoned about:
        # the `-n 2` gate hung here, py-spy showing rank 0 already in pytest teardown after a
        # failed test and rank 1 still inside this function. A diagnostic that deadlocks the run
        # it is diagnosing is worse than no diagnostic, and a *reduction* is the wrong place for
        # Python semantics: these are fixed-shape int64 buffers, so the elementwise MPI op
        # applies on every rank with no interpreter involved. Reduced into a destination buffer
        # rather than `MPI.IN_PLACE`, per CLAUDE.md's MPI rules.
        reduced = np.empty_like(local_bad)
        comm.Allreduce(local_bad, reduced, op=MPI.MAX)
        local_bad = reduced
        reduced = np.empty_like(local_nonzero)
        comm.Allreduce(local_nonzero, reduced, op=MPI.MAX)
        local_nonzero = reduced
        # A plain Python int: the object form is safe for a scalar, whose `op` is a real `max`.
        rows = comm.allreduce(int(rows), op=MPI.SUM)
    return (
        f"non-finite columns: {int(local_bad.sum())}, all-zero columns: "
        f"{int(len(psi0) - local_nonzero.sum())}, determinants: {int(rows)}"
    )


def _size_subspace(num_wanted, width, cap):
    """Krylov subspace depth for a ``num_wanted``-state request from a width-``width`` block.

    Returns ``(max_subspace_blocks, num_wanted)``. The request is padded to ``2 * num_wanted``
    (or ``num_wanted + 10``, whichever is larger), capped at the basis size, converted to a block
    count with a flat ``+20`` of headroom, and then bounded by what the basis itself can hold --
    ``cap // width - 1`` blocks, since the sweep needs room for a trailing residual block.
    ``num_wanted`` is finally clamped to ``(blocks - 1) * width``, the most those blocks can
    deliver.

    One function rather than two copies: ``get_eigenvectors`` sizes the subspace once up front
    and again whenever the request or the block width changes, and the two spellings had drifted
    apart cosmetically (an ``if width > 0`` guard against a ``max(1, width)``) while computing the
    same thing. ``width`` is in fact never 0 -- the start block always carries at least the cold
    full-support column, and :func:`block_normalize` raises rather than returning a width-0 block
    -- so the guard was dead either way.

    **The final clamp is not always a no-op, and when it binds it silently reduces the certified
    output.** It cannot bind on the padding term, which gives ``(blocks - 1) * width >=
    4 * num_wanted``. It binds when the basis-size bound wins: that leaves ``(blocks - 1) * width
    ~ cap - 2 * width``, so any request above ``cap - 2 * width`` is trimmed, and at
    ``cap < 3 * width`` the bound collapses to two blocks and the request is trimmed to ``width``.
    ``num_wanted`` is the *certified* output of the thermal-manifold search, so trimming it is
    the wrong trade -- shrinking ``width`` (dropping warm columns) or falling through to the dense
    branch would both preserve it. Left as-is here because this function is a verbatim lift;
    ``test_subspace_sizing.py`` pins the behaviour and marks the defect.

    Parameters
    ----------
    num_wanted : int
        States requested, before padding.
    width : int
        Lanczos block width, i.e. the number of columns in the start block.
    cap : int
        Basis size; nothing may exceed it.

    Returns
    -------
    tuple of (int, int)
        ``(max_subspace_blocks, num_wanted)``.
    """
    width = max(1, width)
    max_subspace = min(max(2 * num_wanted, num_wanted + 10), cap)
    blocks = min(2 * int(np.ceil(max_subspace / width)) + 20, max(2, cap // width - 1))
    return blocks, min(num_wanted, (blocks - 1) * width)


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
        same as pruning each column to its own threshold -- so every column still needs
        pruning to ``cutoff`` before the cross-rank sum. This reproduces the old
        per-state ``applyOp(H, psi_i, cutoff)`` bit-for-bit, including pruning locally
        *before* redistributing (the same order every other probe in this module uses,
        e.g. ``psi_all_Dj`` above), rather than pruning the already-summed total.

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
        block = ManyBodyState.from_states(psi_ref)
        n_chunks = config.GS_APPLY_ROW_CHUNKS.get()
        if n_chunks is None or n_chunks <= 1:
            raw = self._apply_and_prune_columns(H, block, cutoff)
            merged = self.basis.redistribute_block(raw)
            merged.prune_rows(0.0)
            return merged

        # Row-chunked: the round's peak is the raw apply output, its packed send buffer, the
        # receive buffer and the merged block all alive at once (~6x the owned block, measured;
        # see doc/plans/dc_smo_memory.md round 6). Applying one chunk of the reference rows at
        # a time bounds the first three to chunk size; only the accumulating merged block stays.
        # Exact up to summation order (a candidate reached from rows in different chunks has its
        # partial sums added chunk by chunk, and the per-column prune sees those partials), which
        # is the same class of difference a change of rank count makes.
        #
        # The chunk COUNT is the knob, replicated on every rank, so every rank makes exactly
        # `n_chunks` collective `redistribute_block` calls whatever its row count -- a rank with
        # fewer rows than chunks sends empty chunks (an explicit width-p block with no rows, which
        # `apply_block` and the packer both accept; never the width-0 polymorphic zero).
        keys = block.keys()
        n_rows = len(keys)
        bounds = np.linspace(0, n_rows, int(n_chunks) + 1).astype(int)
        merged = None
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            mask = ManyBodyState.from_states([ManyBodyState(dict.fromkeys(keys[lo:hi], 1.0 + 0j), width=1)])
            part = block.copy()
            part.keep_rows(mask)
            del mask
            raw = self._apply_and_prune_columns(H, part, cutoff)
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
        merged.prune_rows(0.0)
        return merged

    @staticmethod
    def _apply_and_prune_columns(H, block, cutoff):
        """``H`` applied to ``block`` with every column pruned to ``cutoff`` in place (the
        one-shot body of :meth:`_apply_block_and_redistribute`, shared with its chunked path)."""
        raw = H.apply_block(block, cutoff)
        view = np.asarray(raw)  # zero-copy (rows, p) view; buffer.readonly=0, so this writes through
        # `std::norm(v) <= cutoff**2` (no sqrt), matching ManyBodyBlockState::prune_rows'
        # C++ criterion exactly -- not `np.abs(view) <= cutoff`, which takes a sqrt first and so
        # is not guaranteed bit-identical to the C++ comparison at the cutoff boundary.
        norm2 = view.real**2 + view.imag**2
        view[norm2 <= cutoff * cutoff] = 0.0
        del view, norm2  # release the buffer export -- prune_rows/redistribute refuse to run while it's alive
        return raw

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
        # Real-valued by construction (a ratio of two non-negative magnitudes) -- `zeros_like`
        # used to inherit `overlaps`' complex128 dtype, doubling this array's footprint for no
        # reason (every value assigned into it is already real; found while sizing the CIPSI
        # selection round's memory peak, doc/plans/dc_smo_memory.md).
        de2 = np.zeros(overlaps.shape, dtype=float)
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
            "subthreshold_de2_mass": subthreshold_de2_mass,
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
        self,
        e_ref,
        psi_ref,
        H,
        de2_min,
        slater_cutoff=0,
        return_Hpsi_ref=False,
        gen_ops=None,
        max_new=None,
        affordable_growth=None,
    ):
        """Select the candidate determinants to add to the basis.

        Candidates connected to ``psi_ref`` through ``H`` are kept when their de2
        importance (max over the reference states) reaches ``de2_min``. ``max_new``
        optionally caps the *global* number of selected candidates: the top ``max_new``
        by de2 importance are kept (collective bisection cutoff, ties under-admitted)
        before the symmetry closure, and ``self.last_selection`` records
        ``{"n_candidates", "n_admitted", "discarded_de2_mass", "subthreshold_de2_mass",
        "hpsi_rows"}``. Collective on ``basis.comm``.

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
        """
        measure = affordable_growth is not None
        if measure:
            rss_start = current_rss_bytes()
            peak_is_own = reset_peak_rss()
        Hpsi_ref = self._apply_block_and_redistribute(H, psi_ref, slater_cutoff)
        # Global row count of the shared support H|psi_ref> spans (before the not-in-basis
        # filter): the quantity `_apply_block_and_redistribute`'s and `_calc_de2`'s per-round
        # transient scale with (see doc/plans/dc_smo_memory.md) -- recorded here, once, rather
        # than re-derived by every caller that wants the per-cycle memory/timing picture.
        hpsi_rows = self._allreduce_sum(len(Hpsi_ref))
        # `_candidate_overlaps_and_energies` directly, not `_calc_de2`: the latter materializes
        # the whole (p, n_Dj) de2 array in one shot, which is exactly the per-cycle memory peak
        # `_score_candidates` below exists to bound (doc/plans/dc_smo_memory.md). `0`, not
        # `slater_cutoff`, matches `_calc_de2`'s own default -- this call site never passed
        # `slater_cutoff` through `_calc_de2` either, so this preserves that behaviour exactly
        # rather than changing what the diagonal probe prunes to as a side effect of this
        # refactor.
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
            cap = affordable_growth(transient, rss_base) if transient >= 0 else None
            if cap is not None:
                cap = int(cap)
                max_new = cap if max_new is None else min(int(max_new), cap)
            memory_stats = {
                "round_transient_bytes": int(transient),
                "round_rss_bytes": int(rss_base),
                "memory_admit_cap": cap,
            }
        de2_mask, selection_stats = self._admit_top(scores, scores >= de2_min, max_new)
        selection_stats["hpsi_rows"] = hpsi_rows
        selection_stats.update(memory_stats)
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
        memory_budget_bytes=None,
    ):
        """Expand the basis variationally (CIPSI) until it stops growing.

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
        # The memory trip-wire's own latch. It used to ride on `not capped`, which silently
        # disabled the guard for exactly the configuration that crashed: a cap that is finite but
        # far beyond what memory allows. Production ran with `truncation_threshold=119,555,328`
        # and died at 949,834 determinants -- 0.79% of its own cap -- so the cap never bound, yet
        # its mere existence made `capped` true and skipped the check. A separate latch also stops
        # the guard from ratcheting the threshold down every cycle once it has fired, since
        # `peak_rss` is a high-water mark and stays above the budget forever after.
        budget_tripped = False
        memory_bound = False
        cap_cycles = 0
        no_improve = 0
        e0 = np.inf
        best_e0 = None
        best_basis = None
        best_psis = None
        best_e_ref = None
        self.truncation_report = None
        cycle = 0
        # The kept manifold two cycles back, for the growth margin below. `None` until there is
        # one, which is not the same as zero: a first cycle has no growth to extrapolate.
        prev_kept = None
        while True:
            if psi_refs is None:
                num_wanted = 10
                # Left at `None` deliberately; see `_manifold_request` for why that is not 0.
                prev_kept = None
            else:
                n_kept = len(psi_refs)
                num_wanted = _manifold_request(n_kept, prev_kept)
                prev_kept = n_kept
            e_ref, psi_refs = self.get_eigenvectors(
                H,
                num_wanted=min(num_wanted, len(self.basis)),
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
            old_size = self.basis.size
            # The look-ahead half of the memory guard (see `_memory_growth_bound`): bound this
            # cycle's admission by what the *next* selection round can afford, predicted from
            # this round's own measured transient. `prev_kept` already holds the previous
            # cycle's kept count here, so this is exactly the request the next loop head makes.
            affordable_growth = None
            if memory_budget_bytes is not None:
                p_now = len(psi_refs)
                affordable_growth = _memory_growth_bound(
                    memory_budget_bytes, old_size, p_now, _manifold_request(p_now, prev_kept)
                )
            with _trace_timed("cipsi_selection", cycle=cycle) as _sel_event:
                new_Dj = self.determine_new_Dj(
                    e_ref,
                    psi_refs,
                    H,
                    de2_min,
                    slater_cutoff=slaterWeightMin,
                    gen_ops=gen_ops,
                    max_new=admit_target,
                    affordable_growth=affordable_growth,
                )
                n_new = self._allreduce_sum(len(new_Dj))
                sel = self.last_selection or {}
                memory_cap = sel.get("memory_admit_cap")
                # The memory bound decided this admission only if it was the tightest of the
                # three limits (candidates above de2_min, an existing cap's `admit_target`, and
                # itself). A bound looser than the cap already in force is not a memory event
                # and must neither warn nor touch the threshold.
                if (
                    memory_cap is not None
                    and memory_cap < sel.get("n_candidates", 0)
                    and (admit_target is None or memory_cap < admit_target)
                ):
                    # The next round could not afford the full candidate set. Adopt the
                    # affordable size as the fixed budget from here on -- the same machinery a
                    # caller's cap uses -- tightening only (`min`), never loosening an existing cap.
                    memory_bound = True
                    previous = threshold
                    threshold = min(float(threshold), float(old_size + memory_cap))
                    capped = True
                    self.basis.truncation_threshold = threshold
                    if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
                        was = "uncapped" if not np.isfinite(previous) else f"a cap of {int(previous):,}"
                        print(
                            f"WARNING: the selection round on {old_size:,} determinants peaked "
                            f"{format_bytes(sel.get('round_transient_bytes', 0))} above its "
                            f"{format_bytes(sel.get('round_rss_bytes', 0))} resident set against a "
                            f"{format_bytes(memory_budget_bytes)} budget; the next round can afford "
                            f"{memory_cap:,} of the {sel.get('n_candidates', 0):,} candidates. "
                            f"Tightening {was} to {int(threshold):,}.",
                            flush=True,
                        )
                _sel_event.update(
                    basis_size=int(old_size),
                    p=len(psi_refs),
                    hpsi_rows=int(sel.get("hpsi_rows", 0)),
                    n_candidates=int(sel.get("n_candidates", 0)),
                    n_admitted=int(sel.get("n_admitted", 0)),
                    n_new=int(n_new),
                    subthreshold_de2_mass=float(sel.get("subthreshold_de2_mass", 0.0)),
                )
            # Unconditional collectives (CLAUDE.md: never gate a collective on rank-local
            # state, and `self.basis.verbose` may differ across ranks) -- only the print
            # below is gated. Cheap relative to the cycle they describe: single-scalar
            # reductions against a selection round that costs seconds to minutes.
            local_count = len(self.basis.local_basis)
            peak_rss = peak_rss_bytes()
            if self.basis.is_distributed:
                local_max = self.basis.comm.allreduce(local_count, op=MPI.MAX)
                local_min = self.basis.comm.allreduce(local_count, op=MPI.MIN)
                peak_rss = self.basis.comm.allreduce(peak_rss, op=MPI.MAX)
            else:
                local_max = local_min = local_count
            _trace_note(
                "cipsi_cycle",
                cycle=cycle,
                local_max=int(local_max),
                local_min=int(local_min),
                vm_hwm_bytes=int(peak_rss),
            )
            if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
                # Surfaced every cycle, not only once a cap has bound (see
                # doc/plans/dc_smo_memory.md) -- `last_selection`'s counts used to be
                # visible only behind the cap-hit WARNING below.
                print(
                    f"  cycle {cycle}: basis={old_size:,} p={len(psi_refs):,} "
                    f"Hpsi_rows={sel.get('hpsi_rows', 0):,} candidates={sel.get('n_candidates', 0):,} "
                    f"admitted={sel.get('n_admitted', 0):,} new={n_new:,} "
                    f"subthreshold_de2_mass={sel.get('subthreshold_de2_mass', 0.0):.3e} "
                    f"local[min,max]=[{local_min:,},{local_max:,}] VmHWM={format_bytes(peak_rss)}",
                    flush=True,
                )
            cycle += 1
            if memory_budget_bytes is not None and not budget_tripped and peak_rss >= memory_budget_bytes:
                # `peak_rss` and `self.basis.size` are both already rank-replicated at this point
                # (VmHWM was just MAX-allreduced above; `Basis.size` is the global count by
                # construction), so every rank evaluates this condition identically without a
                # separate collective for the decision itself.
                budget_tripped = True
                # `min`, never a bare assignment: an existing cap that already binds tighter than
                # the current basis is an instruction from the caller and must not be loosened.
                previous = threshold
                threshold = min(float(threshold), float(self.basis.size))
                capped = True
                # Written back, not just held locally: a caller inspecting
                # `basis.truncation_threshold` after `expand()` returns must see the cap that
                # actually governed the rest of this run, not the one it was constructed with.
                self.basis.truncation_threshold = threshold
                if self.basis.verbose and (self.basis.comm is None or self.basis.comm.rank == 0):
                    was = "uncapped" if not np.isfinite(previous) else f"a cap of {int(previous):,}"
                    print(
                        f"WARNING: measured per-rank RSS {format_bytes(peak_rss)} reached the "
                        f"{format_bytes(memory_budget_bytes)} memory budget mid-expansion; tightening "
                        f"{was} to a fixed-budget cap at the current basis "
                        f"({self.basis.size:,} determinants) rather than risk an uncatchable OOM kill.",
                        flush=True,
                    )
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
        if cap_cycles > 0 or memory_bound:
            # `memory_bound` alone (no refinement cycle ran) is the case where the memory guard
            # found even a same-size round unaffordable and the expansion stopped where it stood;
            # that must still be reported as a cap, not pass for a converged expansion.
            sel = self.last_selection or {}
            self.truncation_report = {
                "cap_hit": True,
                "cycles": cap_cycles,
                "retained": int(self.basis.size),
                "threshold": int(threshold),
                "discarded_de2_mass": float(sel.get("discarded_de2_mass", 0.0)),
                "n_candidates_last": int(sel.get("n_candidates", 0)),
                "memory_bound": bool(memory_bound),
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

    def _normalize_start_block(self, psi0, cold_start_block, warm_started, slaterWeightMin):
        """Orthonormalize the Lanczos start block, falling back to a cold start if it fails.

        A warm block is *inherited* state -- the previous CIPSI cycle's converged eigenvectors --
        so unlike the cold hash vector it can arrive unusable, and when it does the failure lands
        here, one function away from anything that could have caused it. That killed a production
        SrMnO3 double-counting search outright (rung 7 of the cap ladder, the ``N-1`` sector's
        second refinement cycle, a 261-column warm block): ``block_normalize`` raised, nothing
        caught it, and the whole search returned its unchanged input double counting after 20
        minutes of work.

        Nothing about that is unrecoverable. The cold full-support start vector is always
        available, always finite (``_amplitude_from_hash`` is a bounded integer ratio), and spans
        the whole basis -- it is what the *first* cycle of every expansion uses, and what the
        exhaustion guard below already falls back to. Losing the warm columns costs Krylov
        iterations, not correctness.

        The retry is rank-symmetric, which is what makes it safe to wrap a collective:
        ``block_tsqr`` returns the same rank code on every rank (TSQR's ``R`` is bitwise
        identical), so ``block_normalize`` raises :class:`BlockBreakdown` on all ranks or none,
        and every rank therefore takes the same branch into the same second collective. A cold
        block that fails too is a genuinely empty basis, and that re-raises.

        That argument is why the ``except`` names :class:`BlockBreakdown` and not ``ValueError``.
        It is narrower on purpose: ``block_normalize`` also raises *before* reaching the
        collective -- ``ManyBodyState.from_states`` rejects a width-0 block, which is a
        **rank-local** condition (a rank owning no determinants builds the polymorphic zero while
        its peers do not; see ``cold_start_block``'s own ``width=1`` note below). Recovering from
        that one would send this rank into ``_describe_block_health``'s ``Allreduce``s and a
        second ``block_normalize`` while every other rank was still inside the first, which is a
        deadlock rather than a recovery.

        The diagnostic is computed *only* on the failure path -- a full finiteness pass over the
        block, which says whether the warm columns were corrupted (non-finite) or merely
        degenerate, and is the measurement that pins where such a block came from. Free in the
        normal case, and the one case where the cost does not matter is the one where it runs.
        """
        try:
            psi0, _ = block_normalize(psi0, self.basis.is_distributed, self.basis.comm, slaterWeightMin)
            return psi0, warm_started
        except BlockBreakdown as exc:
            if not warm_started:
                raise
            # Collective (it allreduces over the row partition), so every rank calls it and
            # only the printing is rank-gated -- the ordering CLAUDE.md's MPI rule requires.
            health = _describe_block_health(psi0, self.basis.comm)
            # `psi0` is the warm columns PLUS the appended cold vector, so the warm count is one
            # less -- reporting `len(psi0)` overstated how many inherited columns were in play.
            n_warm = len(psi0) - 1
            # Free outside a `solver_trace.tracing()` block, so this costs nothing in a normal
            # run -- and buys nothing there either: `tracing()` only opens under `DC_DIAGNOSTICS`
            # (`dc_search._dc_search_trace`). It is for the run that is already being investigated;
            # the default-run signal is the rank-0 print below, which stays unconditional.
            _trace_note("warm_block_fallback", warm_columns=int(n_warm), health=health)
            if self.basis.comm is None or self.basis.comm.rank == 0:
                print(
                    f"warning: the warm-started Lanczos block ({n_warm} warm columns plus the "
                    f"cold vector) did not orthonormalize ({exc}); {health}. Restarting this "
                    "solve from the cold full-support vector instead -- slower to converge, same "
                    "subspace reachable.",
                    flush=True,
                )
            psi0, _ = block_normalize(cold_start_block(), self.basis.is_distributed, self.basis.comm, slaterWeightMin)
            return psi0, False

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

        ``max_energy=None`` means **no cut**: the lowest ``num_wanted`` states are returned as
        computed, with no trim and no widening. The widening loop below exists only to certify
        that a *kept manifold* is whole -- a degenerate manifold has no preferred basis, so
        keeping part of one makes the result depend on the rotation the solver returned -- and
        without a cut nothing is dropped, so there is nothing to certify. The dense branch's
        slice is by count alone for the same reason, which does mean the boundary can fall
        inside a degenerate group. Both are sound only for a caller whose answer does not depend
        on which basis a manifold came back in: :func:`groundstate.calc_energy`, which keeps
        ``min(es)``, is the only one, and no other caller should pass ``None``. The warm-start
        cold retry is a different question (reachability, not completeness) and still applies;
        see it below.

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
            warm_block = list(psi_refs) if warm_started else []
            max_block_width = config.GS_MAX_BLOCK_WIDTH.get()
            if max_block_width is not None and len(warm_block) > max_block_width:
                # Truncate the block the *solver* uses, not the manifold the caller asked for:
                # `psi_refs` is already energy-ordered ascending (the caller's own return path,
                # `_energy_cut_indices`'s `order = np.argsort(e_ref)`), so the lowest
                # `max_block_width` states are kept -- the ones a restart is most likely to need
                # again. `num_wanted` below is untouched: this narrows the block Lanczos runs
                # with, not how many states get returned, which is what caps `krylov_bytes`
                # (`memory_estimate._gs_krylov_columns`) without shrinking the certified manifold
                # `expand`'s "exhausted" check reads.
                warm_block = warm_block[:max_block_width]
            psi0 = warm_block + cold_start_block() if warm_started else cold_start_block()

            num_wanted = min(num_wanted + _EIGENSTATE_PAD, len(self.basis))
            psi0, warm_started = self._normalize_start_block(psi0, cold_start_block, warm_started, slaterWeightMin)
            max_subspace_blocks, num_wanted = _size_subspace(num_wanted, len(psi0), len(self.basis))
            _trace_note(
                "size_subspace",
                site="initial",
                blocks=int(max_subspace_blocks),
                width=int(len(psi0)),
                num_wanted=int(num_wanted),
            )
            # How many of those the eigensolver must actually converge to `tol`. The pad above is
            # bought for its *energies* -- a state landing outside the thermal cut is what certifies
            # the kept manifold is whole -- and `_energy_cut_indices` keeps a prefix of the sorted
            # spectrum, so the pad is exactly the top tail that gets discarded. Holding the solve to
            # `tol` on the tail's residuals is what let a single slow padding state burn all 100
            # restarts on an SrMnO3 ground state while the wanted states had long since converged.
            #
            # Safe against the case the pad is reached into (more states inside the cut than were
            # asked for): that is `need_more`, and the loop below re-solves with a larger request --
            # which raises this floor with it -- rather than keeping the under-converged tail.
            num_required = max(1, num_wanted - _EIGENSTATE_PAD)

            if h_matrix is None:
                H_mat = build_sparse_matrix(self.basis, H)
                # Phase 0 measurement (doc/plans/dc_smo_performance.md): calibrates
                # memory_estimate's nnz_per_state default against a real solve.
                _trace_note("h_matrix_nnz", nnz=int(H_mat.nnz), n=int(H_mat.shape[0]))
            else:
                H_mat = h_matrix
            if self.basis.is_distributed:
                H_mat = H_mat[:, self.basis.local_indices]

            psi0_arr = (
                build_distributed_vector(self.basis, psi0).T
                if len(psi0) > 0
                else np.zeros((len(self.basis.local_basis), 1), dtype=complex)
            )
            # Phase 0 measurement (doc/plans/dc_smo_performance.md): the array kernel's
            # per-rank matvec buffer scales with this width, not with a fixed default -- see
            # memory_estimate.estimate_gs_peak_bytes. Zero-cost when no solver_trace.tracing()
            # block is open.
            _trace_note("eigensolve_block_width", p=int(psi0_arr.shape[1]), num_wanted=int(num_wanted))

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
                    # The TRLM/IRLM drivers already rank-gate every print internally
                    # (`rank0 = (not mpi) or comm.rank == 0` in ed/trlm.py and ed/irlm.py);
                    # no need to pre-AND rank 0 into the bool here too.
                    verbose=self.basis.verbose,
                    slaterWeightMin=slaterWeightMin,
                    reort=reort,
                    num_converge=num_required,
                )
                if len(e_ref) == 0:
                    break
                # `need_more` asks a question only a thermal cut can pose -- "is the boundary
                # manifold provably complete?" -- so with `max_energy=None` there is nothing to
                # certify and it stays False. The *exhaustion* test below is a different
                # question and is asked either way; see the retry block.
                need_more = False
                if max_energy is not None:
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
                if exhausted:
                    # The warm block spans a (near-)invariant subspace, so the missing states --
                    # typically the other charge sector of a near-degenerate crossing, e.g. the
                    # fixed-occupation DC search walking over a charge-transfer point -- are
                    # unreachable from it at any num_wanted. Retry once from the cold
                    # rank-independent start vector, whose support covers the whole basis.
                    #
                    # Reachability, not completeness: this guard is about whether `psi0` could
                    # see the state at all, so it must fire on the no-cut path too. It used to
                    # sit behind an early `max_energy is None` break, which meant the caller
                    # that most needs it never got it -- `calc_energy`'s occupation walk, whose
                    # whole job is to step across charge sectors, and which keeps exactly the
                    # `min(es)` a sector-blind warm block silently gets wrong.
                    if cold_retry_available and (need_more or max_energy is None):
                        cold_retry_available = False
                        psi0 = cold_start_block()
                        psi0, _ = block_normalize(psi0, self.basis.is_distributed, self.basis.comm, slaterWeightMin)
                        psi0_arr = build_distributed_vector(self.basis, psi0).T
                        max_subspace_blocks, num_wanted = _size_subspace(num_wanted, len(psi0), cap)
                        _trace_note(
                            "size_subspace",
                            site="cold_retry",
                            blocks=int(max_subspace_blocks),
                            width=int(len(psi0)),
                            num_wanted=int(num_wanted),
                        )
                        num_required = max(1, num_wanted - _EIGENSTATE_PAD)
                        continue
                    break
                if max_energy is None or not need_more or num_wanted >= cap:
                    break
                num_wanted = min(2 * num_wanted, cap)
                max_subspace_blocks, num_wanted = _size_subspace(num_wanted, len(psi0), cap)
                _trace_note(
                    "size_subspace",
                    site="doubling",
                    blocks=int(max_subspace_blocks),
                    width=int(len(psi0)),
                    num_wanted=int(num_wanted),
                )
                num_required = max(1, num_wanted - _EIGENSTATE_PAD)

            valid_idx = None
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
                # Phase 0 measurement (doc/plans/dc_smo_memory.md): `n_computed` is what the
                # eigensolver was held to and what `build_state` above materialised; `n_kept` is
                # what survives the thermal cut and becomes the next cycle's `psi_refs`. The gap
                # between them is what `expand`'s `num_wanted = 2 * len(psi_refs)` buys, and the
                # only number that says whether the manifold grows with the basis.
                _trace_note(
                    "thermal_manifold",
                    n_computed=len(e_ref),
                    n_kept=len(valid_idx),
                    num_wanted=int(num_wanted),
                    n_dets=len(self.basis),
                    need_more=bool(need_more),
                )
                e_ref = e_ref[valid_idx]
            if len(e_ref) > 0:
                # Built *after* the cut, not before it: `build_state` materialises one
                # `ManyBodyState` per column, each carrying every determinant this rank owns, and
                # the cut above typically keeps well under half of them (measured on the SrMnO3
                # double-counting solve: 104 kept of 206 computed). Building all of them and then
                # dropping most is the same waste as the request itself, one layer down -- and it
                # peaks at the same moment the Krylov store is still live. Selecting the columns
                # first is otherwise a no-op: `valid_idx` is already the energy-ascending order
                # the old list comprehension applied, so the retained states and their order are
                # unchanged.
                cols = psi_refs_arr if valid_idx is None else psi_refs_arr[:, valid_idx]
                psi_refs = build_state(self.basis, cols.T, slaterWeightMin=slaterWeightMin)

        else:
            if h_matrix is None:
                H_mat = build_sparse_matrix(self.basis, H)
                # Phase 0 measurement (doc/plans/dc_smo_performance.md): calibrates
                # memory_estimate's nnz_per_state default against a real solve.
                _trace_note("h_matrix_nnz", nnz=int(H_mat.nnz), n=int(H_mat.shape[0]))
            else:
                H_mat = h_matrix
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
            if max_energy is None:
                # `eigensystem` reads `e_max=None` as "no cut" and returns the *whole* dense
                # spectrum, so without a bound here a cut-less caller would materialise one
                # ManyBodyState per determinant -- a cost set by the basis size rather than by
                # the request, and not what asking for `num_wanted` states means. With the cut
                # gone `num_wanted` is the only bound left, and it is the same one the Krylov
                # branch above obeys: that branch never returns more than it asked for either.
                # `es` comes back ascending, so this is the lowest `num_wanted`.
                e_ref = e_ref[:num_wanted]
                psi_ref_dense = psi_ref_dense[:, :num_wanted]
            psi_refs = build_state(self.basis, psi_ref_dense.T, slaterWeightMin=slaterWeightMin)

        return e_ref, psi_refs
