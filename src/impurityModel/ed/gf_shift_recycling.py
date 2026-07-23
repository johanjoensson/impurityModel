"""Sector-cache and shift-recycled Krylov resolvent solvers for the Green's-function chain.

Split out of :mod:`greens_function` (which re-exports everything here for backwards
compatibility): :class:`SectorResolventCache` (dense spectral cache over a closed H-sector)
and :class:`KrylovShiftedResolvent` (one distributed block-Lanczos recurrence serving every
shift of a fixed right-hand side) are the two tiers ahead of the per-point BiCGSTAB/GMRES
fallback in the RIXS R1 solver chain (see ``spectra._R1SolverChain``) and are used directly
by callers that need a resolvent over many shifts of the same seed block.

Depends on :mod:`gf_primitives` (``_CappedBasisProxy``, ``_distributed_seed_qr``,
``_trim_blocks``) but not on :mod:`greens_function` or :mod:`gf_convergence`.
"""

import hashlib
import os

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed import config
from impurityModel.ed.basis_transcription import build_sparse_matrix, build_state
from impurityModel.ed.BlockLanczos import block_lanczos_cy
from impurityModel.ed.TSQR import DEFLATE_TOL_SEEDS
from impurityModel.ed.BlockLanczosArray import resolve_reort
from impurityModel.ed.gf_primitives import _CappedBasisProxy, _distributed_seed_qr, _trim_blocks
from impurityModel.ed.ManyBodyUtils import ManyBodyState
from impurityModel.ed.memory_estimate import available_bytes_per_rank, format_bytes


def _sector_dense_max():
    """Largest sector size the spectral cache may densify (:data:`config.GF_SECTOR_DENSE_MAX`).

    The eigendecomposition holds ~3 dense ``(N, N)`` complex arrays (H, the eigenvector
    matrix and LAPACK workspace); unset, the cap is derived so that fits in a quarter of the
    available per-rank memory.
    """
    override = config.GF_SECTOR_DENSE_MAX.get()
    if override is not None:
        return override
    return int(np.sqrt(0.25 * available_bytes_per_rank() / (3 * 16)))


def _sector_cache_dir():
    """Directory for on-disk sector eigendecompositions (``GF_SECTOR_CACHE_DIR``).

    Unset/empty disables persistence -- the default, since the eigenvector file of a
    several-thousand-determinant sector is hundreds of MB and should not appear
    unasked. The sector is a fixed property of the physical workload, so across
    repeated runs on the same material the dominant one-time ``eigh`` cost (measured
    ~450 s at 5565 determinants: OpenBLAS's Hermitian eigensolvers are bound by their
    non-parallelizing reduction stage, and the measured alternatives -- banded
    eigensolvers, Lanczos tridiagonalization -- are no faster with eigenvectors) is
    then paid once per material instead of once per run.
    """
    return config.GF_SECTOR_CACHE_DIR.get()


def _sector_digest(states, hOp):
    """Content digest of a sector: its determinant list plus the Hamiltonian."""
    digest = hashlib.sha256()
    for state in states:
        digest.update(bytes(state.to_bytearray()))
    digest.update(repr(sorted(hOp.to_dict().items())).encode())
    return digest.hexdigest()[:24]


class SectorResolventCache:
    r"""Spectral cache for repeated resolvent Gram matrices over one closed sector.

    Serves callers that evaluate :math:`G[w, k, k'] = \langle s_k| (z_w - H)^{-1}
    |s_{k'}\rangle` for MANY different seed blocks living in the SAME H-closed sector --
    e.g. the RIXS final-state (R2) resolvent, whose seeds change with every incoming
    frequency while the sector does not. Instead of one block-Lanczos per seed block
    (:func:`block_Green` re-spans the sector every time), the sector Hamiltonian is
    densified and eigendecomposed ONCE; every subsequent evaluation is two small dense
    contractions.

    The cache snapshots the sector's determinant list at build time and is self-contained
    afterwards: callers may freely clear / rebuild the basis they hand in. A seed block
    whose support leaks outside the cached sector triggers a fresh expansion + rebuild
    (different sector). Sectors larger than :func:`_sector_dense_max` are declined
    (``try_eval`` returns ``None``) and the caller falls back to its per-seed solver.

    Only the serial / single-rank-color path is served (``basis.comm`` of size > 1
    returns ``None``): a distributed sector wants the distributed Lanczos, not a
    replicated dense eigenbasis.
    """

    def __init__(self):
        self._index = None  # determinant -> row
        self._evals = None
        self._evecs = None
        self._declined = False
        self._n_solves = 0
        self._n_builds = 0

    def _covers(self, seeds):
        return self._index is not None and all(state in self._index for psi in seeds for state in psi.keys())

    def _ensure(self, basis, hOp, seeds, slaterWeightMin, verbose):
        """Cover ``seeds``' sector, eigendecomposing it on first sight. False = declined.

        A decline is sticky: the sector is a fixed property of the caller's run, so
        once it proved too large (or distributed) there is no point re-paying the
        expansion probe on every subsequent evaluation.
        """
        if self._declined or (basis.comm is not None and basis.comm.size > 1):
            self._declined = True
            return False
        if self._covers(seeds):
            return True
        bound = _sector_dense_max()
        self._expand_to_closure(basis, hOp, seeds, slaterWeightMin, size_bound=bound)
        if len(basis) > bound:
            self._declined = True
            return False
        self._index = {state: i for i, state in enumerate(basis.local_basis)}
        how = self._load_from_disk(basis, hOp)
        if how is None:
            h = build_sparse_matrix(basis, hOp).toarray()
            h = 0.5 * (h + h.conj().T)
            self._evals, self._evecs = np.linalg.eigh(h)
            self._save_to_disk(basis, hOp)
            how = "eigendecomposed"
        self._n_builds += 1
        if verbose:
            print(
                f"Sector resolvent cache: {how} a {len(basis)}-determinant sector " f"(build {self._n_builds}).",
                flush=True,
            )
        return True

    def _disk_path(self, basis, hOp):
        cache_dir = _sector_cache_dir()
        if not cache_dir:
            return None
        return os.path.join(cache_dir, f"sector_{_sector_digest(basis.local_basis, hOp)}.npz")

    def _load_from_disk(self, basis, hOp):
        """Load a persisted eigendecomposition; None if disabled/missing/unusable."""
        path = self._disk_path(basis, hOp)
        if path is None or not os.path.exists(path):
            return None
        try:
            with np.load(path) as data:
                evals, evecs = data["evals"], data["evecs"]
        except Exception as exc:  # unreadable/corrupt (e.g. a killed writer): rebuild
            print(f"warning: ignoring unreadable sector cache file {path}: {exc}", flush=True)
            return None
        if evecs.shape != (len(basis), len(basis)):
            return None
        self._evals, self._evecs = evals, evecs
        return f"loaded (from {path})"

    def _save_to_disk(self, basis, hOp):
        path = self._disk_path(basis, hOp)
        if path is None:
            return
        # Write-then-rename so a killed run never leaves a truncated file behind.
        tmp = f"{path}.{os.getpid()}.tmp.npz"  # .npz suffix so np.savez keeps the name as-is
        np.savez(tmp, evals=self._evals, evecs=self._evecs)
        os.replace(tmp, path)

    def _expand_to_closure(self, basis, hOp, seeds, slaterWeightMin, size_bound):
        """Grow ``basis`` toward the H-connectivity closure of the seed support.

        The same reachability probe as :func:`block_Green`'s expansion loop (repeated
        ``apply_block`` on the accumulated support). Stops early once ``size_bound`` is
        exceeded: the caller will decline the sector anyway, and completing the closure
        of a massive sector would grow the basis all the way to its
        ``truncation_threshold`` for nothing.
        """
        basis.add_states(sorted({state for psi in seeds for state in psi.keys()}))
        probe = ManyBodyState.from_states(list(seeds))
        while True:
            old_size = basis.size
            probe = hOp.apply_block(probe, slaterWeightMin)
            basis.add_states(
                {state for state in probe.support_keys(0.0) if state not in basis.local_basis},
            )
            if basis.size == old_size or basis.size > min(size_bound, basis.truncation_threshold):
                break

    def try_eval(self, basis, hOp, seeds, zs, slaterWeightMin=0, verbose=False):
        """Resolvent Gram matrix ``G[w, k, k']`` over ``zs``, or ``None`` if not applicable.

        ``zs`` carries the full shift (e.g. ``wLoss + 1j * delta2 + E_e``): the cached
        eigenbasis is shift-independent, so one cache serves every eigenstate and
        frequency offset.
        """
        if not self._ensure(basis, hOp, seeds, slaterWeightMin, verbose):
            return None
        x = self._evecs.conj().T @ self._project(seeds)  # (N, n_seeds)
        self._n_solves += 1
        # G[w] = X^dagger diag(1 / (z_w - lambda)) X, batched over the z mesh.
        zs = np.asarray(zs)
        inv = 1.0 / (zs[:, None] - self._evals[None, :])  # (n_w, N)
        return np.einsum("nk,wn,nl->wkl", x.conj(), inv, x, optimize=True)

    def _project(self, seeds):
        s_dense = np.zeros((len(self._index), len(seeds)), dtype=complex)
        for k, psi in enumerate(seeds):
            for state, amp in psi.items():
                s_dense[self._index[state], k] = amp[0]
        return s_dense

    def try_solve(self, basis, hOp, rhs, z, slaterWeightMin=0, verbose=False):
        r"""Exact sector solutions ``x_k = (z - H)^{-1} rhs_k``, or ``None`` if not applicable.

        The spectral counterpart of an iterative shifted solve (e.g. the RIXS
        intermediate resolvent's ``block_bicgstab``): on a cached sector it is direct
        -- no iteration, no near-pole stagnation -- and exact on the sector.
        Amplitudes with ``|amp|^2 <= slaterWeightMin`` are pruned from the returned
        states, mirroring the iterative solvers' support pruning.
        """
        if not self._ensure(basis, hOp, rhs, slaterWeightMin, verbose):
            return None
        x = self._evecs.conj().T @ self._project(rhs)
        x = self._evecs @ (x / (z - self._evals)[:, None])  # (N, n_rhs)
        self._n_solves += 1
        states = list(self._index)
        out = []
        for k in range(x.shape[1]):
            keep = np.abs(x[:, k]) ** 2 > slaterWeightMin
            # width=1: a column whose projected solution is (numerically) exactly zero
            # everywhere kept must not become the width-0 polymorphic zero -- every
            # element of the returned list is expected width 1.
            out.append(ManyBodyState({states[i]: x[i, k] for i in np.nonzero(keep)[0]}, width=1))
        return out


def _gf_krylov_recycle_max_bytes():
    """Per-rank byte cap on a recycled Krylov store (:data:`config.GF_KRYLOV_RECYCLE_MAX_BYTES`).

    The retained Krylov basis is :class:`KrylovShiftedResolvent`'s dominant allocation;
    unset, it is capped at a quarter of the available per-rank memory, mirroring
    :func:`_sector_dense_max`'s budget for the dense spectral cache.
    """
    override = config.GF_KRYLOV_RECYCLE_MAX_BYTES.get()
    if override is not None:
        return override
    return available_bytes_per_rank() // 4


def _shifted_tridiag_solutions(alphas, betas, block_widths, b0, zs):
    r"""Shifted solves of the projected block-tridiagonal systems, plus exact residuals.

    For every shift ``z`` solves ``(z I - T) y = E_1 b_0`` where ``T`` is the block
    tridiagonal assembled from the (trimmed) block-Lanczos coefficients. By the
    recurrence ``H Q = Q T + q_{m+1} \beta_m E_m^H``, the residual of the full-space
    Galerkin solution ``x = Q y`` of ``(z - H) x = Q E_1 b_0`` is exactly
    ``q_{m+1} \beta_m y_m``, so its norm ``||\beta_m y_m||_F`` follows from the small
    quantities alone -- no matvec. The systems are solved banded (bandwidth below twice
    the maximum block width), so a residual check per resume round costs
    ``O(n_z * n_T * p^2)`` -- negligible next to one Lanczos step.

    Args:
        alphas, betas, block_widths: coefficients as returned by ``block_lanczos_cy``
            (padded; trimmed here). ``betas[-1]`` is the residual coupling used for
            the residual norms.
        b0: ``(w_0, n_rhs)`` projection of the right-hand-side block onto the first
            Lanczos block (the seed QR's R factor).
        zs: complex shifts.

    Returns:
        tuple: ``(Y, res)`` -- ``Y`` of shape ``(n_z, n_T, n_rhs)`` with
        ``n_T = sum(block_widths)``, and ``res`` of shape ``(n_z,)`` holding the
        residual Frobenius norms.
    """
    a, b = _trim_blocks(alphas, betas, block_widths)
    widths = [int(w) for w in block_widths]
    k = len(widths)
    starts = np.concatenate([[0], np.cumsum(widths)]).astype(int)
    n_t = int(starts[-1])
    zs = np.asarray(zs, dtype=complex)
    n_rhs = b0.shape[1]
    if n_t == 0:
        return np.zeros((len(zs), 0, n_rhs), dtype=complex), np.full(len(zs), float(np.linalg.norm(b0)))

    bw = max(w - 1 for w in widths)
    for i in range(k - 1):
        bw = max(bw, widths[i] + widths[i + 1] - 1)
    ab = np.zeros((2 * bw + 1, n_t), dtype=complex)

    def _put(M, r0, c0):
        # LAPACK banded storage: ab[bw + i - j, j] = T[i, j].
        for jl in range(M.shape[1]):
            j = c0 + jl
            ab[bw + r0 - j : bw + r0 - j + M.shape[0], j] = M[:, jl]

    for i in range(k):
        _put(a[i], starts[i], starts[i])
        if i + 1 < k:
            _put(b[i], starts[i + 1], starts[i])
            _put(np.conj(b[i].T), starts[i], starts[i + 1])

    rhs = np.zeros((n_t, n_rhs), dtype=complex)
    rhs[: widths[0]] = b0[: widths[0]]
    tail = b[-1]  # residual coupling beyond the retained subspace
    last = slice(int(starts[k - 1]), n_t)
    Y = np.empty((len(zs), n_t, n_rhs), dtype=complex)
    res = np.empty(len(zs))
    for wi, z in enumerate(zs):
        ab_z = -ab
        ab_z[bw] += z
        y = sp.linalg.solve_banded((bw, bw), ab_z, rhs, check_finite=False)
        Y[wi] = y
        res[wi] = float(np.linalg.norm(tail @ y[last]))
    return Y, res


def _no_convergence_check(alphas, betas, verbose=False, block_widths=None, **kwargs):
    """Convergence callback that never fires: convergence is judged by the caller instead.

    Matches ``block_lanczos_cy``'s actual convergence-callback contract -- it calls
    ``converged_fn(alphas, betas, verbose=verbose, block_widths=block_widths + [n_curr])``
    on *every* iteration (see ``BlockLanczos.pyx``), not just ``(alphas, betas, verbose)``.
    Used by :class:`KrylovShiftedResolvent`, whose ``solve`` judges convergence itself
    between resume rounds via the exact shifted Galerkin residuals
    (:func:`_shifted_tridiag_solutions`), never through the kernel's own check.

    A version of this that only accepted ``(a, b, verbose=False)`` crashed with a
    ``TypeError`` the first time a real workload's sector outlived the seed block's
    invariant-subspace closure (no unit test's tiny model reached the callback before
    then) -- see the regression test ``test_krylov_shifted_resolvent_long_recurrence``.
    """
    return False


class KrylovShiftedResolvent:
    r"""Shift-recycled block-Krylov resolvent: one recurrence serves every frequency.

    Solves ``(z - H) x_k = y_k`` for a FIXED right-hand-side block and MANY shifts
    ``z``. The Krylov space of ``z - H`` seeded with ``y`` is independent of ``z``, so a
    single distributed block-Lanczos recurrence serves all shifts:
    ``x(z) = Q (z I - T)^{-1} E_1 B_0``, with the small shifted systems solved banded
    and the per-shift Galerkin residual known exactly from the projected quantities
    (see :func:`_shifted_tridiag_solutions`). The recurrence is resumed through the
    kernel's warm-start protocol until every shift meets ``atol`` (or the recurrence
    closes: an invariant subspace makes the solutions exact).

    This serves the regime :class:`SectorResolventCache` declines -- distributed bases
    and sectors too large to densify -- replacing one iterative solve *per frequency*
    with one recurrence per right-hand-side block. On a ``truncation_threshold``-capped
    basis it is exact in the same sense as the other capped Lanczos paths: post-freeze,
    the converged solutions are those of the frozen ``P H P``.

    Memory: the solutions are RECONSTRUCTED from the retained Krylov store, so
    tail-only retention and complex64 storage are both off the table; the store is
    bounded by :func:`_gf_krylov_recycle_max_bytes` per rank instead, and a resume
    round that would exceed the bound before convergence declines the whole solve
    (``solve`` returns ``None``, the caller falls back to its per-point solver).

    ``reort`` defaults to ``"full"``: the reconstruction multiplies the shifted
    tridiagonal solution back into the retained Krylov store ``Q``, and the per-round
    convergence gate trusts the projected residual ``||beta_m y_m||`` -- both are only
    faithful while ``Q`` is orthonormal, i.e. while ``T`` really is ``Q^H H Q``. FULL
    reorthogonalizes each new block against the whole store, which this recycler keeps in
    memory anyway (``store_krylov=True``), so it costs no extra storage and keeps the
    residual estimate honest. PARTIAL's approximate ``sqrt(eps)`` estimator is mismatched
    with a full store: on a long single-seed recurrence it was observed to leave the
    reconstruction near ``1e-6`` while the projected residual still reported convergence
    at ``atol`` (worse than doing no reorthogonalization at all) -- see the regression test
    ``test_krylov_shifted_resolvent_long_recurrence``.
    """

    def __init__(self, reort="full"):
        self._reort = resolve_reort(reort)

    def solve(self, basis, hOp, rhs, zs, slaterWeightMin=0, atol=1e-6, verbose=False):
        """Solutions ``[x_k(z) for k] for z in zs`` as ``ManyBodyState`` lists, or ``None``.

        ``None`` means declined -- the memory bound would be exceeded before the shifted
        residuals reach ``atol * ||rhs||`` (or the recurrence diverged) -- and the caller
        should fall back to its per-point solver. MPI-collective on ``basis.comm``; every
        branch decision derives from replicated data, so all ranks agree. ``basis`` is
        cleared and regrown toward the seeds' H-closure (the same contract as the
        per-point iterative solvers' rebuild loop); the returned states are distributed
        by the grown basis's ownership.
        """
        comm = basis.comm
        mpi = comm is not None
        rank = comm.rank if mpi else 0
        n_rhs = len(rhs)
        zs = np.asarray(zs, dtype=complex)
        max_bytes = _gf_krylov_recycle_max_bytes()
        if max_bytes == 0:
            return None
        if n_rhs == 0 or len(zs) == 0:
            return [[ManyBodyState(width=1) for _ in range(n_rhs)] for _ in zs]

        basis.clear()
        basis.add_states(sorted({state for psi in rhs for state in psi.keys()}))
        rhs = basis.redistribute_psis(list(rhs))

        # Orthonormal seed block + projection B0 (the same preamble as block_green_impl).
        psi_dense_local, r = _distributed_seed_qr(basis, rhs, slaterWeightMin)
        psi_arr = build_state(basis, psi_dense_local.T, slaterWeightMin=0)
        b0 = r
        scale = float(np.linalg.norm(b0))
        if len(psi_arr) == 0 or scale == 0.0:
            return [[ManyBodyState(width=1) for _ in range(n_rhs)] for _ in zs]

        # Enforce the determinant cap on the recurrence (post-freeze: exact P H P).
        cap = getattr(basis, "truncation_threshold", np.inf)
        lanczos_basis = _CappedBasisProxy(basis, cap) if np.isfinite(cap) else basis

        # Resume in growing budget rounds (the block_Green_sparse pattern): convergence
        # is judged between rounds on the exact shifted residuals, not by the kernel.
        alphas = betas = Q = W = widths = None
        Y = None
        budget = max(int(getattr(basis, "size", 0)) // max(len(psi_arr), 1), 8)
        while True:
            alphas, betas, Q, W, widths, status = block_lanczos_cy(
                psi_arr,
                hOp,
                lanczos_basis,
                _no_convergence_check,
                verbose=verbose,
                reort=self._reort,
                slaterWeightMin=slaterWeightMin,
                max_iter=budget,
                return_widths=True,
                return_status=True,
                alphas_init=alphas,
                betas_init=betas,
                Q_init=Q,
                W_init=W,
                block_widths_init=widths,
                store_krylov=True,
                # The recycled right-hand side is the RIXS Cartesian polarization block;
                # its dependent components must deflate. See DEFLATE_TOL_SEEDS in TSQR.pyx.
                deflate_tol=DEFLATE_TOL_SEEDS,
            )
            Y, res = _shifted_tridiag_solutions(alphas, betas, widths, b0, zs)
            # An invariant subspace closes the recurrence: the coupling out of the
            # retained space is zero, so the solutions are exact there (the residual
            # check below is then satisfied by construction, up to roundoff).
            if np.max(res) <= atol * scale or status == "invariant_subspace":
                break
            if status == "diverged":
                if rank == 0:
                    print(
                        f"warning: shift-recycled Krylov resolvent diverged at relative residual "
                        f"{np.max(res) / scale:.2e} (target {atol:.1e}); declining to the per-point solver.",
                        flush=True,
                    )
                return None
            # Memory guard before growing: the next round roughly doubles the store.
            q_bytes = int(Q.memory_bytes())
            if mpi:
                q_bytes = comm.allreduce(q_bytes, op=MPI.MAX)
            if 2 * q_bytes > max_bytes:
                if rank == 0:
                    print(
                        f"Shift-recycled Krylov resolvent declined: store at {format_bytes(q_bytes)} "
                        f"per rank would exceed the {format_bytes(max_bytes)} bound "
                        f"(GF_KRYLOV_RECYCLE_MAX_BYTES) before reaching {atol:.1e} "
                        f"(residual {np.max(res) / scale:.2e}); falling back to the per-point solver.",
                        flush=True,
                    )
                return None
            budget *= 2

        if verbose and rank == 0:
            print(
                f"Shift-recycled Krylov resolvent: {len(zs)} shift(s) from one "
                f"{int(np.sum(widths))}-vector recurrence "
                f"(max relative residual {np.max(res) / scale:.2e}).",
                flush=True,
            )
        # The store's leading sum(widths) columns are the Lanczos blocks (laid out by
        # true, deflated widths); the trailing residual block is excluded.
        n_keep = int(np.sum(widths))
        return [Q.combine(Y[wi], 0, n_keep, slaterWeightMin) for wi in range(len(zs))]
