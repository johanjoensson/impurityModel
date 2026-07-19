"""Bottom-layer Green's-function primitives: QR/state-vector plumbing, the block-tridiagonal
continued fraction, and the ``truncation_threshold``-capping basis proxy.

Split out of :mod:`greens_function` (which re-exports everything here for backwards
compatibility) so the solver-policy code in that module and in :mod:`gf_convergence` /
:mod:`gf_shift_recycling` sits on a small, dependency-free layer: nothing here imports from
either of those two, or from :mod:`greens_function` itself.
"""

import numpy as np
import scipy as sp
from mpi4py import MPI

from impurityModel.ed.basis_transcription import build_vector
from impurityModel.ed.BlockLanczosArray import BETA_BLOWUP_FACTOR
from impurityModel.ed.manybody_basis import collective_amplitude_cutoff
from impurityModel.ed.ManyBodyUtils import ManyBodyBlockState, ManyBodyState


def build_qr(psi):
    """
    Perform an economic QR decomposition of a state matrix.

    Parameters
    ----------
    psi : ndarray
        The input state matrix.

    Returns
    -------
    psi_orthogonal : ndarray
        The orthogonalized matrix Q.
    r : ndarray
        The upper triangular matrix R.
    """
    # Do a QR decomposition of the starting block.
    # Later on, use r to restore the psi block
    psi, r = sp.linalg.qr(psi.copy(), mode="economic", overwrite_a=True, check_finite=False, pivoting=False)
    return np.ascontiguousarray(psi), r


def calc_continuants(diagonal, offdiagonal):
    """
    Calculate continued fraction continuants.

    """

    An = np.empty_like(diagonal)
    Bn = np.empty_like(An)
    An[-1] = np.eye(diagonal.shape[1])
    Bn[-1] = 0
    An[0] = diagonal[0]
    Bn[0] = 1
    for n in range(1, diagonal.shape[0]):
        An[n] = diagonal[n] * An[n - 1] - np.conj(offdiagonal[n]) * An[n - 2] * offdiagonal[n]
        Bn[n] = diagonal[n] * Bn[n - 1] - np.conj(offdiagonal[n]) * Bn[n - 2] * offdiagonal[n]
    return An, Bn


def _scatter_qr_columns(comm, psi_dense, r, local_size):
    """Scatter the row-distributed QR factor ``Q`` (held on rank 0) across MPI ranks.

    Rank 0 holds the full ``(N, n)`` ``Q`` and the ``(n, n)`` ``R`` after :func:`build_qr`.
    Broadcast ``R`` and the column count, then ``Scatterv`` ``Q``'s rows onto each rank's
    local partition (``local_size`` rows). Shared by ``block_green_impl`` (sparse branch)
    and ``block_Green_sparse``.

    Returns
    -------
    psi_dense_local : ndarray
        This rank's ``(local_size, n)`` slice of ``Q``.
    r : ndarray
        The ``(n, n)`` ``R`` factor (replicated on every rank).
    """
    rank = comm.rank
    r = comm.bcast(r if rank == 0 else None, root=0)
    columns = comm.bcast(psi_dense.shape[1] if rank == 0 else None, root=0)
    psi_dense_local = np.empty((local_size, columns), dtype=complex, order="C")
    send_counts = np.empty((comm.size), dtype=int) if rank == 0 else None
    comm.Gather(np.array([psi_dense_local.size]), send_counts, root=0)
    offsets = np.array([np.sum(send_counts[:rr]) for rr in range(comm.size)], dtype=int) if rank == 0 else None
    comm.Scatterv(
        [psi_dense, send_counts, offsets, MPI.C_DOUBLE_COMPLEX] if rank == 0 else None,
        psi_dense_local,
        root=0,
    )
    return psi_dense_local, r


def _distributed_seed_qr(basis, psi_arr, slaterWeightMin=0):
    """Row-distributed orthonormal seed block + its ``R`` factor.

    Shared preamble of every resolvent solver that needs an orthonormal seed block
    distributed by ``basis``'s ownership (:func:`block_Green_sparse`, the sparse branch
    of :func:`block_green_impl`, :class:`KrylovShiftedResolvent`): build the dense seed
    matrix on rank 0, QR it there (:func:`build_qr`), then scatter ``Q``'s rows onto each
    rank's local partition (:func:`_scatter_qr_columns`) so every rank ends up with only
    its own slice. Serial (``basis.comm is None``) just runs ``build_qr`` directly.

    Returns
    -------
    psi_dense_local : ndarray
        This rank's ``(local_size, n)`` slice of the orthonormal seed block ``Q``.
    r : ndarray
        The ``(n, n)`` ``R`` factor (replicated on every rank).
    """
    comm = basis.comm
    mpi = comm is not None
    rank = comm.rank if mpi else 0
    psi_dense = build_vector(basis, psi_arr, root=0, slaterWeightMin=slaterWeightMin).T
    r = None
    if rank == 0:
        psi_dense, r = build_qr(psi_dense)
    if mpi:
        psi_dense_local, r = _scatter_qr_columns(
            comm, psi_dense if rank == 0 else None, r if rank == 0 else None, len(basis.local_basis)
        )
    else:
        psi_dense_local = psi_dense
    return psi_dense_local, r


class PairwiseGF:
    r"""Per-eigenstate Green's-function block assembled from scalar (width-1) continued fractions.

    Holds the scalar block-Lanczos coefficients for the operator-split decomposition of one
    ``n x n`` block (one thermal state, one spectral side): the ``n`` diagonal seeds and, per
    off-diagonal pair, the two polarization seeds. :func:`calc_G_pairwise` evaluates these on a
    frequency mesh and reassembles the full matrix via the polarization identity.

    Attributes
    ----------
    n : int
        Block dimension (number of transition operators).
    diag : list[tuple]
        Length-``n`` list of ``(alphas, betas, r)`` scalar continued fractions for ``v_i``.
    pairs : dict[tuple[int, int], tuple[tuple, tuple]]
        ``{(i, j): (cf_sum, cf_imag)}`` for ``i < j`` -- the scalar continued fractions for the
        seeds ``v_i + v_j`` and ``v_i + i v_j``.
    """

    __slots__ = ("diag", "n", "pairs")

    def __init__(self, n, diag, pairs):
        self.n = n
        self.diag = diag
        self.pairs = pairs


def calc_G_pairwise(pgf: "PairwiseGF", mesh, e, delta):
    r"""Assemble an ``n x n`` Green's-function block from its scalar continued fractions.

    Each scalar seed ``w`` gives the resolvent
    ``S(w) = w^\dagger (\omega + i\delta + e - H)^{-1} w`` via the width-1 continued fraction
    (:func:`calc_G`). The diagonal elements are ``G_ii = S(v_i)``; each off-diagonal pair is
    recovered from the polarization identity

    .. math::

        S(v_i + v_j)   &= M_{ii} + M_{jj} + M_{ij} + M_{ji}, \\
        S(v_i + i v_j) &= M_{ii} + M_{jj} + i M_{ij} - i M_{ji},

    so ``M_ij = ½[S(v_i+v_j) - i S(v_i+i v_j) - (1-i)(M_ii+M_jj)]`` and ``M_ji`` is its mirror.
    Exact (no approximation) given converged scalar continued fractions.
    """
    n = pgf.n
    G = np.zeros((len(mesh), n, n), dtype=complex)

    def S(cf):
        alphas, betas, r = cf
        return calc_G(alphas, betas, r, mesh, e, delta)[:, 0, 0]

    diag_S = [S(cf) for cf in pgf.diag]
    for i in range(n):
        G[:, i, i] = diag_S[i]
    for (i, j), (cf_sum, cf_imag) in pgf.pairs.items():
        Mii, Mjj = diag_S[i], diag_S[j]
        S_sum, S_imag = S(cf_sum), S(cf_imag)
        G[:, i, j] = 0.5 * (S_sum - 1j * S_imag - (1 - 1j) * (Mii + Mjj))
        G[:, j, i] = 0.5 * (S_sum + 1j * S_imag - (1 + 1j) * (Mii + Mjj))
    return G


def calc_thermally_averaged_G(alphas, betas, r, mesh, es, e0, tau, delta):
    """
    Calculate the thermally averaged Green's function over multiple initial states.

    Parameters
    ----------
    alphas : list of list of ndarray
    betas : list of list of ndarray
    r : list of ndarray
    mesh : ndarray
    es : list of float
    e0 : float
    tau : float
    delta : float

    Returns
    -------
    G_avg : ndarray
    """
    # Operator-split (pairwise) path: r holds a per-eigenstate PairwiseGF; each carries its own
    # scalar continued fractions, so (alphas, betas) are unused and calc_G_pairwise assembles the
    # block from the polarization identity.
    if any(isinstance(r_e, PairwiseGF) for r_e in r):
        n_ops = next(r_e.n for r_e in r if isinstance(r_e, PairwiseGF))
        G_avg = np.zeros((len(mesh), n_ops, n_ops), dtype=complex)
        for e, r_e in zip(es, r):
            if r_e is None:
                continue
            G_avg += calc_G_pairwise(r_e, mesh, e, delta) * np.exp(-(e - e0) / tau)
        return G_avg

    if len(alphas) == 0:
        return np.zeros((len(mesh), 0, 0), dtype=complex)

    n_ops = r[0].shape[-1]
    G_avg = np.zeros((len(mesh), n_ops, n_ops), dtype=complex)

    for e, alphas_e, betas_e, r_e in zip(es, alphas, betas, r):
        G_avg += calc_G(alphas_e, betas_e, r_e, mesh, e, delta) * np.exp(-(e - e0) / tau)

    return G_avg


class _CappedBasisProxy:
    """Enforce ``truncation_threshold`` on the sparse-kernel GF recurrence.

    ``block_lanczos_cy``'s matvec discovers new Slater determinants every step, so the
    live block-state support (and, at reort != none, the Krylov store) grows without
    bound — the excited ``Basis`` itself stays frozen and never sees them. This proxy
    wraps that basis and caps the growth at the one point where every residual row
    sits on its hash-owner rank: the per-step ``redistribute_block`` call.

    Policy (freeze-growth + importance-ranked boundary admission):

    * while ``retained + n_new <= cap``: admit every newly discovered determinant;
    * on the single overflow step: rank that step's candidate rows by max column
      ``|amp|^2`` of the residual and admit the top ``cap - retained`` via a
      fixed-iteration distributed amplitude bisection (allreduce'd counts, so the
      cutoff is collective and deterministic), then freeze;
    * after the freeze: drop non-retained rows of every residual (rank-local
      ``keep_rows`` merge; ownership routing makes membership checks local).

    Why this is safe: every previously accepted Krylov block has support inside the
    retained set, so the diagonal projector ``P`` is invisible to inner products
    against them (``<Q_j, P wp> = <Q_j, wp>``) — orthogonality is untouched. From the
    freeze on, the recurrence is an *exact* block Lanczos of the Hermitian projected
    operator ``P H P``: the continued fraction stays causal, moments up to the freeze
    are exact w.r.t. ``H``, and the recurrence terminates as ``invariant_subspace``
    (already treated as exact-on-subspace). All reort modes remain valid, and the
    Krylov store's row set is bounded by the retained set (it never needs removal).

    MPI: one scalar allreduce per pre-freeze step; the freeze decision and the
    bisection derive only from allreduce'd data, so ranks cannot disagree, and every
    collective runs unconditionally (a rank may retain zero rows).

    The per-frequency BiCGSTAB driver (:func:`block_Green_bicgstab`) reuses this proxy
    unchanged in spirit: ``block_bicgstab``'s matvec routes through
    ``redistribute_block`` (also in serial runs, keyed on ``caps_growth``), so the same
    freeze-growth policy bounds a linear solve's live support, and post-freeze the solve
    is an exact BiCGSTAB of the projected operator ``P H P`` -- the same
    exact-on-retained-subspace contract as the capped Lanczos recurrence. The extra
    forwarders below (``add_states``, ``contains_local``, the restriction properties)
    are the attributes ``block_bicgstab`` reads off its basis.
    """

    caps_growth = True

    def __init__(self, basis, cap):
        self._basis = basis
        self.cap = int(cap)
        self.comm = basis.comm
        # Width-0 key-only mask of the retained determinants on this rank; grown by
        # in-place C++ sorted merges only (no per-row Python objects in the hot path).
        seed = ManyBodyState(dict.fromkeys(basis.local_basis, 1.0 + 0j))
        self._mask = ManyBodyBlockState.from_states([seed]).key_union(ManyBodyBlockState())
        self._global_count = int(basis.size)
        self._frozen = self._global_count >= self.cap
        self.cap_hit = self._frozen
        self._verbose_freeze_logged = False

    # --- attributes block_lanczos_cy reads off its basis ---------------------
    @property
    def local_basis(self):
        return self._basis.local_basis

    @property
    def size(self):
        return self._basis.size

    @property
    def n_bytes(self):
        return self._basis.n_bytes

    @property
    def is_distributed(self):
        return self._basis.is_distributed

    @property
    def restrictions(self):
        return self._basis.restrictions

    @property
    def weighted_restrictions(self):
        return self._basis.weighted_restrictions

    def redistribute_psis(self, psis):
        return self._basis.redistribute_psis(psis)

    def add_states(self, new_states, unique_sorted=False):
        # Growth bookkeeping only: every determinant block_bicgstab offers here came off a
        # redistribute_block-capped block, so it is already inside the retained mask and
        # counted by _global_count -- the wrapped basis can never outgrow the cap through
        # this path.
        return self._basis.add_states(new_states, unique_sorted=unique_sorted)

    def contains_local(self, state):
        return self._basis.contains_local(state)

    @property
    def retained_size(self):
        """Global number of determinants currently admitted to the recurrence."""
        return self._global_count

    def retained_keys(self):
        """Rank-local retained determinants as ``SlaterDeterminant`` wrappers (sorted).

        Builds one Python object per retained determinant — diagnostics/tests only,
        never the hot path."""
        keys, _ = self._mask.row_max_norms2()
        return keys

    def _allreduce_sum(self, value):
        if self.comm is None or self.comm.size == 1:
            return value
        return self.comm.allreduce(value, op=MPI.SUM)

    def redistribute_block(self, block):
        block = self._basis.redistribute_block(block)
        if self._frozen:
            block.keep_rows(self._mask)
            return block
        n_new = self._allreduce_sum(len(block) - block.count_rows_in(self._mask))
        if self._global_count + n_new <= self.cap:
            self._mask.merge_keys(block)
            self._global_count += n_new
            return block
        self._admit_top_and_freeze(block)
        block.keep_rows(self._mask)
        return block

    def _admit_top_and_freeze(self, block):
        """Admit the ``cap - retained`` most important candidate rows, then freeze.

        The amplitude-cutoff bisection runs a fixed iteration count on allreduce'd
        counts, so all ranks compute the identical cutoff. Ties at the cutoff are
        under-admitted (the cap is never exceeded); near-tie retained sets may differ
        across rank counts through summation-order rounding, like the CIPSI basis
        trajectory.
        """
        slots = self.cap - self._global_count
        norms2 = block.new_row_max_norms2(self._mask)
        cutoff2 = collective_amplitude_cutoff(norms2, slots, self.comm)
        admitted = block.keys_new_above(self._mask, cutoff2)
        self._global_count += self._allreduce_sum(len(admitted))
        self._mask.merge_keys(admitted)
        self._frozen = True
        self.cap_hit = True

    def freeze_message(self):
        """One-line description of the cap state (rank-0 logging)."""
        return (
            f"GF basis cap hit: froze the recurrence support at {self._global_count:,} "
            f"determinants (truncation_threshold={self.cap:,}); the Green's function is "
            f"exact on the retained subspace."
        )


def _trim_blocks(alphas, betas, block_widths):
    r"""Strip the zero padding from block-Lanczos coefficients (shrinking blocks).

    The Lanczos kernels store every block into a fixed ``(P, P)`` pre-allocated
    buffer, zero-padding the inactive rows/columns whenever a block deflates
    (``block_widths[i] < P``).  This returns the true variable-dimension blocks:
    ``alphas[i] -> (w_i, w_i)`` and ``betas[i] -> (w_{i+1}, w_i)`` where
    ``w_i = block_widths[i]`` (the trailing ``betas[-1]`` residual block keeps its
    stored row count — it is the coupling beyond the subspace and is unused by the
    continued fraction).

    Args:
        alphas: Diagonal blocks, ``(k, P, P)`` ndarray (or length-``k`` sequence).
        betas: Off-diagonal blocks, same outer length.
        block_widths: True width ``w_i`` of every block.

    Returns:
        tuple[list, list]: ragged ``(alphas, betas)`` lists of 2D arrays.

    Raises:
        ValueError: if the width table and the coefficient arrays disagree in length.
            The kernels append a width for every stored block, so a mismatch means a
            caller trimmed one and not the other -- which would silently shorten the
            continued fraction (``k = len(widths)``) instead of failing.
    """
    widths = [int(w) for w in block_widths]
    k = len(widths)
    if k != len(alphas) or k != len(betas):
        raise ValueError(
            f"block_widths has {k} entries but alphas/betas have {len(alphas)}/{len(betas)}; "
            "the continued fraction would silently use only the first "
            f"{min(k, len(alphas))} block(s)."
        )
    a = [np.asarray(alphas[i])[: widths[i], : widths[i]] for i in range(k)]
    b = []
    for i in range(k):
        rows = widths[i + 1] if i + 1 < k else np.asarray(betas[i]).shape[0]
        b.append(np.asarray(betas[i])[:rows, : widths[i]])
    return a, b


def _sanitize_continued_fraction(alphas, betas, verbose=False, rank=0):
    r"""Drop a corrupted trailing tail from the block-Lanczos coefficients.

    Defense-in-depth before the continued fraction / self-energy: the Lanczos kernels now
    truncate a diverging recurrence at the source (CholeskyQR2 + the ``BETA_BLOWUP_FACTOR``
    guard), but should a non-finite or runaway block ever reach here it must *not* be fed
    silently into :func:`calc_G` and ``sig_static``.  Scans the (trimmed) blocks and keeps
    only the leading run whose norms stay bounded relative to the healthy part; the trailing
    ``beta`` of the kept run is the (ignored) residual coupling, so dropping the tail is
    consistent with the continued fraction's own convention.

    Returns the (possibly shortened) ``(alphas, betas)`` and warns when a tail is dropped.
    """
    norm_max = 0.0
    keep = len(alphas)
    for i in range(len(alphas)):
        a = np.asarray(alphas[i])
        b = np.asarray(betas[i])
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            keep = i
            break
        a_norm = float(np.linalg.norm(a, 2)) if a.size else 0.0
        b_norm = float(np.linalg.norm(b, 2)) if b.size else 0.0
        if i > 0 and max(a_norm, b_norm) > BETA_BLOWUP_FACTOR * max(norm_max, 1.0):
            keep = i
            break
        norm_max = max(norm_max, a_norm, b_norm)
    if keep < len(alphas):
        if verbose and rank == 0:
            print(
                f"warning: discarding {len(alphas) - keep} corrupted block(s) from the "
                f"Green's-function continued fraction before computing the self-energy.",
                flush=True,
            )
        return alphas[:keep], betas[:keep]
    return alphas, betas


def _block_cf_inverse(alphas, betas, omegaP):
    r"""Level-0 inverse resolvent of a block-tridiagonal :math:`T` by continued fraction.

    Builds, for every frequency ``omegaP`` (already shifted, i.e.
    :math:`\omega + i\delta + e`),

    .. math::

        G^{-1}_0(\omega) = \omega I - \alpha_0
            - \beta_0^\dagger \big(\omega I - \alpha_1 - \cdots\big)^{-1} \beta_0 ,

    where ``alphas[i]`` is the ``(n_i, n_i)`` diagonal block and ``betas[i]`` the
    ``(n_{i+1}, n_i)`` sub-diagonal block coupling block ``i`` to ``i+1``.  Block
    dimensions may vary from level to level (shrinking-block deflation) and the
    ``betas`` may be rectangular; the identity at each level is sized from that
    level's diagonal block, so no fixed block dimension is assumed.  The trailing
    ``betas[-1]`` (residual coupling beyond the retained subspace) is ignored.

    Args:
        alphas: Length-``k`` sequence of square diagonal blocks.
        betas: Length-``k`` sequence of sub-diagonal blocks.
        omegaP: ``(n_w,)`` complex frequency mesh (shift already applied).

    Returns:
        numpy.ndarray: ``(n_w, n_0, n_0)`` inverse resolvent at the first block.
    """
    nw = omegaP.shape[0]

    def wI(n):
        return omegaP[:, np.newaxis, np.newaxis] * np.identity(n, dtype=complex)[np.newaxis]

    a_last = np.asarray(alphas[-1])
    G_inv = wI(a_last.shape[0]) - a_last[np.newaxis]
    for alpha_raw, beta_raw in zip(alphas[-2::-1], betas[-2::-1]):
        alpha = np.asarray(alpha_raw)
        beta = np.asarray(beta_raw)
        n_i = alpha.shape[0]
        beta_b = np.broadcast_to(beta, (nw,) + beta.shape)
        G_inv = wI(n_i) - alpha[np.newaxis] - np.conj(beta.T)[np.newaxis] @ np.linalg.solve(G_inv, beta_b)
    return G_inv


def calc_G(alphas, betas, r, omega, e, delta):
    r"""Green's function from block-Lanczos continued-fraction coefficients.

    Computes :math:`G(\omega) = r^\dagger (\omega + i\delta + e - T)^{-1} r` where
    ``T`` is the block-tridiagonal matrix with diagonal blocks ``alphas`` and
    sub-diagonal blocks ``betas``.  ``alphas`` / ``betas`` may be either a uniform
    ``(k, p, p)`` ndarray (no deflation) or ragged sequences of variable-dimension
    2D blocks (after :func:`_trim_blocks`); rectangular ``betas`` from shrinking-block
    deflation are handled — no fixed block dimension is assumed.

    Parameters
    ----------
    alphas : ndarray or sequence of ndarray
        Diagonal continued-fraction blocks.
    betas : ndarray or sequence of ndarray
        Off-diagonal continued-fraction blocks (``betas[i]`` couples block ``i`` to
        ``i+1`` with shape ``(n_{i+1}, n_i)``).
    r : ndarray
        ``(n_0, n_ops)`` projection of the seed block onto the first Lanczos block.
    omega : ndarray
        Frequency mesh.
    e : float
        Energy offset.
    delta : float
        Broadening factor.

    Returns
    -------
    G : ndarray
        ``(len(omega), n_ops, n_ops)`` Green's function.
    """
    r = np.asarray(r)
    if len(alphas) == 0 or not np.any(r):
        # A zero seed projection gives G = r^H (...) r = 0 identically; skip the solve,
        # whose tridiagonal may be singular on the mesh (e.g. an all-elastic
        # susceptibility seed projected to zero, evaluated at nu = 0 with delta = 0).
        n_ops = r.shape[-1]
        return np.zeros((len(omega), n_ops, n_ops), dtype=complex)
    omegaP = np.asarray(omega) + 1j * delta + e
    G_inv = _block_cf_inverse(alphas, betas, omegaP)
    r_b = np.broadcast_to(r, (omegaP.shape[0],) + r.shape)
    return np.conj(r.T)[np.newaxis] @ np.linalg.solve(G_inv, r_b)
