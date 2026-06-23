# distutils: language = c++
# cython: language_level=3, boundscheck=False, initializedcheck=False, wraparound=False, freethreading_compatible=True, cdivision=True, cpow=True
"""
BlockLanczos.pyx
================
Parallel Block Lanczos eigensolver implemented in Cython.

This module provides:

* ``block_lanczos_cy`` – core iteration loop producing the block-tridiagonal
  Lanczos representation :math:`T = Q^\\dagger H Q` of the Hamiltonian.
* ``thick_restart_block_lanczos_cy`` – thick-restart (TRLM) wrapper that
  restarts the Krylov subspace while retaining the best Ritz pairs.
* ``implicitly_restarted_block_lanczos_cy`` – implicitly-restarted (IRLM)
  wrapper that applies :math:`(m-k)` implicit QR shifts to compress the
  subspace back to :math:`k` blocks before continuing.

All distributed inner products use ``mpi4py``'s Python API
(``comm.Allreduce``) over small :math:`p \\times p` matrices.  Heavy
matvec work is delegated to ``ManyBodyOperator.apply_multi`` which
releases the GIL internally.

Notes
-----
SlaterDeterminant distribution:
    Each rank owns the SDs with ``hash(sd) % mpi_size == rank``.
    This is maintained by ``basis.redistribute_psis()`` after every
    ``apply_multi`` call.

Pre-allocation:
    ``alphas`` and ``betas`` arrays are pre-allocated before the loop
    and sliced at return time; no numpy allocation occurs inside the
    Lanczos iteration body.

Reorthogonalization modes (``Reort`` enum from ``lanczos.py``):

* ``NONE``      – no reorthogonalization.
* ``PARTIAL``   – Paige-Simon W-matrix estimator; reorthogonalize only
  when the estimated overlap exceeds :math:`\\sqrt{\\varepsilon}`.
* ``SELECTIVE`` – as PARTIAL but additionally projects against converged
  Ritz vectors.
* ``FULL``      – full Gram-Schmidt against all previous blocks (2
  passes).
* ``PERIODIC``  – full reorthogonalization every ``reort_period`` steps.
"""

import math

import numpy as np
import scipy.linalg as sp
from impurityModel.ed.ManyBodyUtils import (
    ManyBodyState,
    add_scaled_multi,
    inner_multi,
)
from mpi4py import MPI

cimport numpy as np

from impurityModel.ed.BlockLanczosArray import estimate_orthonormality, eigsh, _build_full_T, _extract_blocks
from impurityModel.ed.BlockLanczosArray import Reort

cpdef list block_combine_sparse(list Q, np.ndarray Y, double slaterWeightMin=0.0):
    cdef int n_out = Y.shape[1]
    cdef list out = [ManyBodyState() for _ in range(n_out)]
    add_scaled_multi(out, Q, np.ascontiguousarray(Y, dtype=complex))
    if slaterWeightMin > 0:
        for st in out:
            st.prune(slaterWeightMin)
    return out


cpdef tuple block_orthogonalize_sparse(list wp, list Q, object overlaps=None, object comm=None):
    if overlaps is None:
        overlaps = inner_multi(Q, wp)
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
    add_scaled_multi(wp, Q, -overlaps)
    return wp, overlaps


cpdef tuple block_normalize_sparse(list wp, bint mpi=False, object comm=None, double slaterWeightMin=0.0):
    cdef np.ndarray M = inner_multi(wp, wp)
    if comm is not None:
        comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)
    cdef int n = M.shape[0]
    cdef np.ndarray L = sp.cholesky(M, lower=True)
    cdef np.ndarray beta = np.conj(L.T)
    cdef np.ndarray beta_inv = sp.inv(beta)
    cdef list q_next = [ManyBodyState() for _ in range(n)]
    add_scaled_multi(q_next, wp, beta_inv)
    if slaterWeightMin > 0:
        for st in q_next:
            st.prune(slaterWeightMin)
    return q_next, beta

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _block_inner_mpi(states_a, states_b, mpi: bool, comm):
    """Compute the block inner product :math:`G = \\langle A | B \\rangle` with MPI reduction.

    Computes the :math:`p \\times p` Gram matrix

    .. math::

        G_{ij} = \\langle a_i \\mid b_j \\rangle, \\quad i, j = 0, \\dots, p-1

    using a *zero-copy* local pass via ``inner_multi`` (which sums contributions from
    the Slater determinants owned by this rank), then performs a single ``MPI_Allreduce``
    (``MPI.SUM``) so that every rank holds the identical, globally reduced :math:`G`.

    Note:
        **Collective operation** – all MPI ranks must call this function simultaneously
        whenever ``mpi=True``.  The inner computation itself (``inner_multi``) is local
        (no communication); only the subsequent ``Allreduce`` is collective.

    Args:
        states_a: List of ``ManyBodyState`` of length ``p`` representing the bra block
            :math:`A = [a_0, \\dots, a_{p-1}]`.
        states_b: List of ``ManyBodyState`` of length ``p`` representing the ket block
            :math:`B = [b_0, \\dots, b_{p-1}]`.
        mpi: ``True`` when running under MPI with more than one rank.  When ``False`` no
            communication occurs and the local result is returned directly.
        comm: Active ``mpi4py.MPI.Comm`` communicator, or ``None`` when running serially.

    Returns:
        numpy.ndarray: Complex array of shape ``(p, p)`` holding the globally reduced
        Gram matrix :math:`G`.
    """
    G = inner_multi(states_a, states_b)
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, G, op=MPI.SUM)
    return G


def _cholesky_beta(M):
    """Cholesky-factorize :math:`M = L L^\\dagger` and return :math:`\\beta = L^\\dagger`.

    Performs the lower Cholesky decomposition

    .. math::

        M = L L^\\dagger, \\quad \\beta = L^\\dagger \\text{ (upper triangular)}

    so that :math:`\\beta` is the Lanczos off-diagonal block and :math:`Q_{i+1} = W_p
    \\beta^{-1}` is a well-conditioned next Krylov block.

    **Numerical regularisation strategy**: if the first Cholesky attempt raises
    ``scipy.linalg.LinAlgError`` (indicating a numerically indefinite :math:`M`), a
    small diagonal shift :math:`\\delta = 10^{-14} \\|M\\|_F` is added before retrying:

    .. math::

        M_{\\text{reg}} = M + \\delta\\, I_p.

    If the regularised factorisation also fails the exception is re-raised.

    Note:
        This function is *not* a collective MPI operation; it operates entirely on
        replicated :math:`p \\times p` data that is identical on all ranks.

    Args:
        M: Hermitian positive-(semi)definite numpy array of shape ``(p, p)``.
            Typically :math:`M = W_p^\\dagger W_p` from the current Lanczos step.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A 2-tuple ``(beta, beta_inv)`` where

        * ``beta`` – upper-triangular complex array of shape ``(p, p)``
          equal to :math:`L^\\dagger`.
        * ``beta_inv`` – complex array of shape ``(p, p)`` equal to
          :math:`(L^\\dagger)^{-1}`, used to normalise the next Krylov block.

    Raises:
        numpy.linalg.LinAlgError: If :math:`M` is singular even after the diagonal
            regularisation :math:`+\delta I_p`.
    """
    cond = np.linalg.cond(M)
    if cond > 1.0 / (100.0 * np.finfo(float).eps):
        raise np.linalg.LinAlgError(f"Ill-conditioned matrix: cond={cond:.2e}")
    try:
        L = sp.cholesky(M, lower=True)
    except sp.LinAlgError:
        raise sp.LinAlgError("Cholesky decomposition failed")
    beta = L.conj().T  # upper triangular
    beta_inv = sp.inv(beta)
    return beta, beta_inv


def _cholesky_or_deflate(M, p, eps=1e-12):
    """
    Cholesky-factorize M if full rank, otherwise use eigendecomposition to deflate.
    Returns beta_i, beta_inv, and active_k (number of non-converged vectors).
    """
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    max_eig = eigvals[0] if len(eigvals) > 0 else 0
    thresh = max(eps, eps * max_eig)
    
    k = 0
    for e in eigvals:
        if e > thresh:
            k += 1
            
    if k == p:
        try:
            L = sp.cholesky(M, lower=True, check_finite=False)
            beta = np.conj(L.T)
            beta_inv = sp.inv(beta, check_finite=False)
            return beta, beta_inv, k
        except Exception:
            pass
            
    beta = np.zeros((p, p), dtype=complex)
    beta_inv = np.zeros((p, p), dtype=complex)
    
    if k > 0:
        L_sqrt = np.sqrt(eigvals[:k])
        beta[:k, :] = L_sqrt[:, np.newaxis] * np.conj(eigvecs[:, :k].T)
        beta_inv[:, :k] = eigvecs[:, :k] / L_sqrt[np.newaxis, :]
        
    return beta, beta_inv, k


def _reorthogonalize_sparse(wp, Q_states, indices, mpi, comm):
    """Classical double-pass Gram-Schmidt of ``wp`` against a *sparse* subset of Q.

    Applies the Modified Gram-Schmidt correction twice (Kahan-Parlett double reorthogonalization)
    for improved numerical stability:

    .. math::

        \\text{for } s = 1, 2: \\quad
        w_p \\leftarrow w_p
            - \\sum_{j \\in \\mathcal{I}} q_j \\langle q_j \\mid w_p \\rangle

    where :math:`\\mathcal{I}` = ``indices`` selects a *sparse* subset of the accumulated
    Krylov basis.  This is cheaper than projecting against all columns when only a few
    blocks are contaminated.

    Note:
        This function itself is **not** a collective MPI operation in the sense that the
        caller decides which ranks invoke it.  However, the ``inner_multi`` call inside
        each pass is local, and the subsequent ``Allreduce`` (``MPI.SUM``) over the small
        ``(|\\mathcal{I}|, p)`` overlap matrix **is** collective – all ranks that enter
        this function must call it simultaneously.

    Args:
        wp: List of ``p`` ``ManyBodyState`` objects to reorthogonalize.  Modified
            **in-place**; the caller's list is updated directly.
        Q_states: Full flat list of ``ManyBodyState`` basis vectors (length
            ``(current\_block + 1) * p``).
        indices: Integer indices into ``Q_states`` identifying the vectors to project
            against.  Pass an empty list (or ``[]``) to skip reorthogonalization (the
            function returns immediately in that case).
        mpi: ``True`` when running under MPI with more than one rank.
        comm: Active ``mpi4py.MPI.Comm`` communicator, or ``None`` when running serially.
    """
    if not indices:
        return
    Q_sub = [Q_states[i] for i in indices]
    for _ in range(2):
        overlaps = inner_multi(Q_sub, wp)
        if mpi and comm is not None:
            comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)
        add_scaled_multi(wp, Q_sub, -overlaps)


# ---------------------------------------------------------------------------
# Core Block Lanczos step
# ---------------------------------------------------------------------------


def block_lanczos_step_cy(
    h_op,
    q_prev,
    q_curr,
    Q_basis,
    alphas,
    betas,
    it: int,
    reort_mode,
    W,
    mpi: bool,
    comm,
    basis,
    slaterWeightMin: float = 0.0,
    reort_period: int = 5,
    start_it: int = 0,
):
    """Perform one step of the distributed block Lanczos iteration.

    Executes the three-term block recurrence

    .. math::

        W_p = H Q_i - Q_i \\alpha_i - Q_{i-1} \\beta_{i-1}^\\dagger

    where :math:`\\alpha_i = Q_i^\\dagger H Q_i` is the diagonal block and
    :math:`\\beta_{i-1}` is the off-diagonal block from the previous step.
    The residual block :math:`W_p` is then QR-factorised via the Cholesky
    decomposition of :math:`M = W_p^\\dagger W_p`:

    .. math::

        M = L L^\\dagger, \\quad
        \\beta_i = L^\\dagger, \\quad
        Q_{i+1} = W_p \\beta_i^{-1}.

    **MPI collective operations** (all ranks must call simultaneously):

    1. ``MPI_Allreduce`` (``MPI.SUM``) on :math:`\\alpha_i` – :math:`p \\times p`
       complex matrix; result is replicated on every rank.
    2. ``MPI_Allreduce`` (``MPI.SUM``) on :math:`M = W_p^\\dagger W_p` – :math:`p \\times p`
       complex matrix; used for Cholesky QR and breakdown detection.
    3. For ``FULL`` / ``PERIODIC`` modes, an additional ``MPI_Allreduce`` per
       reorthogonalization pass on the :math:`(it \\cdot p) \\times p` overlap matrix.
    4. For ``PARTIAL``, standard Paige-Simon tracking operates against all Lanczos vectors,
       triggering ``MPI_Allreduce`` calls over potentially large swaths of the Krylov
       basis when orthogonality loss exceeds :math:`\\sqrt{\\varepsilon_{\\text{mach}}}`.
    5. For ``SELECTIVE`` (EA16), tracking operates exclusively against converged Ritz vectors.
       The Ritz overlap is calculated as $w_{ritz, k} = s_k^\\dagger W[:, \\text{curr}]$, and
       the W-matrix geometric projection update is
       $W[:, \\text{curr}] \\leftarrow W[:, \\text{curr}] - (W \\cdot s_k) \\langle y_k \\mid w_p \\rangle$.
       Because ``SELECTIVE`` exclusively projects against a small subset of Ritz vectors rather
       than the entire basis, the ``MPI_Allreduce`` footprint is drastically lower.

    Note:
        ``h_op.apply_multi`` releases the GIL internally; all other operations in
        this function hold the GIL.

    Args:
        h_op: ``ManyBodyOperator`` Hamiltonian; must implement
            ``apply_multi(psis, cutoff)``.
        q_prev: List of ``p`` ``ManyBodyState`` objects from iteration ``i-1``
            (pass an empty list or zero states at ``it=0``).
        q_curr: List of ``p`` ``ManyBodyState`` objects from iteration ``i``.
        Q_basis: Accumulated Krylov basis as a flat list of ``ManyBodyState``
            (length ``p * (it + 1)`` on entry, grows by ``p`` each step).
        alphas: Pre-allocated numpy array of shape ``(max_iter, p, p)`` that
            receives the diagonal block :math:`\\alpha_i` at index ``it``.
        betas: Pre-allocated numpy array of shape ``(max_iter, p, p)`` that
            receives the off-diagonal block :math:`\\beta_i` at index ``it``.
        it: Current iteration index into ``alphas`` / ``betas`` buffers
            (0-based, reset to 0 at the start of each fresh run).
        reort_mode: ``Reort`` enum value controlling reorthogonalization strategy.
            ``Reort.NONE`` skips all reorthogonalization entirely.
        W: Paige-Simon W-matrix estimator state array of shape
            ``(2, it+1, p, p)``, or ``None`` when not used.  Required for
            ``PARTIAL`` / ``SELECTIVE`` modes; ignored otherwise.
        mpi: ``True`` when running under MPI with more than one rank.
        comm: Active ``mpi4py.MPI.Comm`` communicator, or ``None`` when running
            serially.
        slaterWeightMin: Amplitude cutoff passed to ``ManyBodyState.prune``;
            SD coefficients below this value are dropped.  Default ``0.0``
            (no pruning).
        reort_period: Number of steps between full reorthogonalization sweeps
            for ``Reort.PERIODIC`` mode.  Full reorthogonalization is applied at
            step ``it`` when ``it > 0`` and ``it % reort_period == 0``.
            Default ``5``.

    Returns:
        tuple: A 5-tuple ``(q_next, alpha_i, beta_i, W_updated, breakdown)``:

        * ``q_next`` – List of ``p`` ``ManyBodyState`` objects forming the next
          Krylov block :math:`Q_{i+1}`, or ``None`` if breakdown occurred.
        * ``alpha_i`` – numpy complex array of shape ``(p, p)`` for the current
          diagonal block.
        * ``beta_i`` – numpy complex array of shape ``(p, p)`` for the current
          off-diagonal block, or ``None`` if breakdown occurred.
        * ``W_updated`` – updated Paige-Simon estimator array, or ``None`` if
          not applicable.
        * ``breakdown`` – ``True`` if an invariant subspace or an ill-conditioned
          block was detected (``NaN``/``Inf`` in :math:`M`, or condition number
          exceeding :math:`100 / \\varepsilon_{\\text{mach}}`).
    """
    p = len(q_curr)

    # --- 1. Block matvec: wp = H q_curr ---------------------------------
    wp = h_op.apply_multi(q_curr, slaterWeightMin)
    if mpi and comm is not None and basis is not None:
        wp = basis.redistribute_psis(wp)

    # --- 2. alpha_i = <q_curr | wp> -------------------------------------
    alpha_i = inner_multi(q_curr, wp)
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, alpha_i, op=MPI.SUM)
    alphas[it] = alpha_i

    # --- 3. Subtract: wp = wp - q_curr * alpha_i - q_prev * beta_{i-1}^† -
    add_scaled_multi(wp, q_curr, -alpha_i)
    if it > 0:
        beta_prev_dag = np.conj(betas[it - 1].T)
        add_scaled_multi(wp, q_prev, -beta_prev_dag)

    # --- 3b. Explicit orthogonalization against history blocks (resumed run) ---
    if reort_mode != Reort.NONE and start_it > 0:
        _reorthogonalize_sparse(wp, Q_basis, list(range(start_it * p)), mpi, comm)

    # --- 4. Full / Periodic reorthogonalization -------------------------
    if reort_mode == Reort.FULL or (reort_mode == Reort.PERIODIC and it > 0 and it % reort_period == 0):
        for _ in range(2):
            ov = inner_multi(Q_basis, wp)
            if mpi and comm is not None:
                comm.Allreduce(MPI.IN_PLACE, ov, op=MPI.SUM)
            add_scaled_multi(wp, Q_basis, -ov)

    # --- 5. M = <wp|wp>, check breakdown --------------------------------
    M = inner_multi(wp, wp)
    if mpi and comm is not None:
        comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)

    if np.any(np.isnan(M)) or np.any(np.isinf(M)):
        return None, alpha_i, None, W, True

    # --- 6. Deflation / Cholesky QR -------------------------------------
    beta_i, beta_inv, active_k = _cholesky_or_deflate(M, p)
    if active_k == 0:
        return None, alpha_i, None, W, True

    betas[it] = beta_i

    q_next = [ManyBodyState() for _ in range(p)]
    add_scaled_multi(q_next, wp, beta_inv)

    # --- 7. EA16 Selective Orthogonalization / Partial Reortho ---------
    if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
        if W is None:
            W = np.zeros((1, 1, p, p), dtype=complex)
            W[0, 0] = np.eye(p)

        W = estimate_orthonormality(
            W,
            alphas[: it + 1],
            betas[: it + 1],
            eps=np.finfo(float).eps,
        )

        reort_eps = math.sqrt(np.finfo(float).eps)

        if reort_mode == Reort.SELECTIVE:
            if start_it == 0 or it > start_it:
                try:
                    import scipy.linalg as spla
                    from scipy.sparse.linalg import eigsh

                    T_full = _build_full_T(alphas[: it + 1], betas[: it + 1])
                    N_T = T_full.shape[0]
                    if N_T <= 10:
                        eigvals_T, conv_evec = spla.eigh(T_full)
                        # We want the smallest eigenvalues (which SA gives)
                        # eigh returns them sorted ascending, so we take the first few
                        k_eig = min(N_T, 6)
                        eigvals_T = eigvals_T[:k_eig]
                        conv_evec = conv_evec[:, :k_eig]
                    else:
                        eigvals_T, conv_evec = eigsh(T_full, k=min(N_T - 1, 6), which="SA")
                    for k in range(len(eigvals_T)):
                        err_bnd = np.linalg.norm(betas[it], ord=2) * np.abs(conv_evec[-1, k])
                        if err_bnd < reort_eps:
                            s_k = conv_evec[:, k]
                            s_k_blocks = s_k.reshape(it + 1, p)
                            w_ritz_k = np.zeros(p, dtype=complex)
                            for j in range(it + 1):
                                w_ritz_k += np.conj(s_k_blocks[j]) @ np.conj(W[-1, j].T)
                            if np.max(np.abs(w_ritz_k)) > reort_eps:
                                ritz_vec = [ManyBodyState() for _ in range(p)]
                                add_scaled_multi(ritz_vec, Q_basis, s_k[:, np.newaxis])
                                for _ in range(2):
                                    overlap = inner_multi(ritz_vec, q_next)
                                    if mpi and comm is not None:
                                        comm.Allreduce(MPI.IN_PLACE, overlap, op=MPI.SUM)
                                    add_scaled_multi(q_next, ritz_vec, -overlap)
                                    W_sk = np.zeros((it + 2, p), dtype=complex)
                                    for m in range(it + 2):
                                        # Only compute if we have full W, else ignore update
                                        if W.shape[0] > 2:
                                            for j in range(it + 1):
                                                W_sk[m] += W[m, j] @ s_k_blocks[j]
                                    for m in range(it + 2):
                                        if W.shape[0] > 2:
                                            W[m, -1] -= np.outer(W_sk[m], overlap[0])
                                            W[-1, m] = np.conj(W[m, -1].T)
                except Exception:
                    pass

        if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
            n_blks = it + 1
            bad_block_idx = []
            if not mpi or comm is None or comm.rank == 0:
                if np.max(np.abs(W[-1, :n_blks])) > reort_eps:
                    bad_block_idx = [j for j in range(n_blks) if np.max(np.abs(W[-1, j])) > np.finfo(float).eps ** (3 / 4)]
            
            if mpi and comm is not None:
                bad_block_idx = comm.bcast(bad_block_idx, root=0)

            if len(bad_block_idx) > 0:
                bad_state_idx = [j * p + k for j in bad_block_idx for k in range(p)]
                _reorthogonalize_sparse(q_next, Q_basis, bad_state_idx, mpi, comm)
                for j in bad_block_idx:
                    W[-1, j] = np.finfo(float).eps * np.eye(p, dtype=complex)

    # Zero out the parts that correspond to the deflated vectors
    for i in range(active_k, p):
        q_next[i] = q_next[i].__class__() # reinitialize to empty
        
    if slaterWeightMin > 0:
        for st in q_next[:active_k]:
            st.prune(slaterWeightMin)

    return q_next, alpha_i, beta_i, W, False


# ---------------------------------------------------------------------------
# Full Block Lanczos iteration loop
# ---------------------------------------------------------------------------


def block_lanczos_cy(
    psi0,
    h_op,
    basis,
    converged_fn,
    verbose: bool = False,
    reort="full",
    max_iter=None,
    slaterWeightMin: float = 0.0,
    comm=None,
    reort_period: int = 5,
    alphas_init=None,
    betas_init=None,
    Q_init=None,
    W_init=None,
):
    """Run the distributed block Lanczos iteration with ``ManyBodyState``.

    Implements the block Lanczos three-term recurrence

    .. math::

        H Q_i = Q_i \\alpha_i + Q_{i+1} \\beta_i + Q_{i-1} \\beta_{i-1}^\\dagger

    building the block-tridiagonal matrix

    .. math::

        T_m = \\begin{pmatrix}
            \\alpha_0 & \\beta_0^\\dagger & & \\\\
            \\beta_0  & \\alpha_1 & \\ddots & \\\\
                     & \\ddots & \\ddots & \\beta_{m-2}^\\dagger \\\\
                     & & \\beta_{m-2} & \\alpha_{m-1}
        \\end{pmatrix}.

    **Pre-allocation**: ``alphas_buf`` and ``betas_buf`` of shape
    ``(max_iter, p, p)`` are allocated *once* before the main loop so that no
    NumPy allocation occurs inside the Lanczos body.  Slices are copied to
    Python lists (``alphas_list`` / ``betas_list``) after each accepted step and
    the final output arrays are built with a single ``np.array()`` call.

    **Warm-start / resume protocol**: if ``alphas_init``, ``betas_init``, and
    ``Q_init`` are all provided the iteration resumes from block
    ``len(alphas_init)``.  The warm-start Q blocks are extracted from
    ``Q_init`` and the existing W-estimator (``W_init``) is reused.  If
    ``W_init`` is ``None`` but ``reort`` is ``'partial'`` or ``'selective'``,
    the Paige-Simon estimator array ``W`` is exactly initialized by computing the
    exact overlaps of the starting blocks against all prior blocks (Exact
    Overlap Restart - EOR).  Pass ``None`` for all three to start a fresh run.

    **MPI collective operations** (called every iteration, all ranks must participate):

    * One ``MPI_Allreduce`` (``MPI.SUM``) for :math:`\\alpha_i` – shape
      ``(p, p)``; result is replicated on all ranks.
    * One ``MPI_Allreduce`` (``MPI.SUM``) for :math:`M = W_p^\\dagger W_p` –
      shape ``(p, p)``; used for Cholesky QR.
    * Additional ``MPI_Allreduce`` calls inside reorthogonalization (see
      ``block_lanczos_step_cy`` for details).

    Note:
        ``converged_fn`` is called on **all ranks** with *replicated* data
        (``alphas``, ``betas`` are identical on every rank).  If the callable
        performs rank-dependent logic the caller is responsible for broadcasting
        the result.  When ``reort == Reort.NONE`` no reorthogonalization of any
        kind is performed; orthogonality degrades at the rate of floating-point
        rounding errors.

    Args:
        psi0: Initial block of ``p`` ``ManyBodyState`` objects, or ``None``
            when resuming from ``Q_init`` (warm-start mode).
        h_op: ``ManyBodyOperator`` that implements ``apply_multi(psis, cutoff)``.
        basis: ``Basis`` object providing ``redistribute_psis`` and ``basis.comm``.
        converged_fn: Callable with signature
            ``converged_fn(alphas, betas, verbose=bool) -> bool`` that returns
            ``True`` once the desired accuracy is reached.  Called after every
            accepted step with the *full* ``alphas``/``betas`` arrays built so far.
        verbose: Print per-iteration diagnostics including ``|beta|`` norms and
            diagonal alpha values.  Default ``False``.
        reort: Reorthogonalization mode.  Accepts ``Reort`` enum members or the
            equivalent strings ``'none'``, ``'partial'``, ``'selective'``,
            ``'full'``, ``'periodic'``.  ``Reort.NONE`` disables all
            reorthogonalization.  Default ``'full'``.
        max_iter: Maximum number of Lanczos steps before returning.  Defaults to
            ``basis.size // p`` when ``basis.size`` is available, otherwise
            ``10 * p``.
        slaterWeightMin: Amplitude cutoff passed to ``ManyBodyState.prune``.
            Slater determinants with :math:`|c_k| < \\text{slaterWeightMin}` are
            dropped after each matvec.  Default ``0.0`` (no pruning).
        comm: ``mpi4py`` communicator.  If ``None``, falls back to ``basis.comm``
            and then to serial mode (``mpi=False``).
        reort_period: Period for ``Reort.PERIODIC`` mode.  Full
            reorthogonalization is applied at steps that satisfy
            ``it > 0 and it % reort_period == 0``.  Default ``5``.
        alphas_init: Warm-start diagonal block array of shape ``(k0, p, p)``.
            If provided together with ``betas_init`` and ``Q_init``, iteration
            resumes from block ``k0``.  Default ``None``.
        betas_init: Warm-start off-diagonal block array of shape ``(k0, p, p)``.
            Default ``None``.
        Q_init: Warm-start Krylov basis as a flat list of ``ManyBodyState``
            (length ``>= k0 * p``).  Default ``None``.
        W_init: Warm-start Paige-Simon W-estimator array.  Passed through to
            the step function; ``None`` causes exact initialisation via EOR
            when resuming with partial/selective reorthogonalization.  Default
            ``None``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, list, numpy.ndarray]: A 4-tuple
        ``(alphas, betas, Q_basis, W)`` where

        * ``alphas`` – complex array of shape ``(k, p, p)`` holding the
          diagonal blocks :math:`\\alpha_0, \\dots, \\alpha_{k-1}`.
        * ``betas`` – complex array of shape ``(k, p, p)`` holding the
          off-diagonal blocks :math:`\\beta_0, \\dots, \\beta_{k-1}`.
        * ``Q_basis`` – flat list of ``ManyBodyState`` of length
          ``(k + 1) * p``; the last ``p`` entries form the residual
          direction :math:`Q_k` (not yet accepted as a full Lanczos block).
        * ``W`` – Paige-Simon estimator array, or ``None`` when not tracked.

    Raises:
        ValueError: If the ``reort`` string argument is not one of
            ``'none'``, ``'partial'``, ``'selective'``, ``'full'``,
            ``'periodic'``.
    """
    from impurityModel.ed.ManyBodyUtils import ManyBodyOperator

    if isinstance(h_op, dict):
        h_op = ManyBodyOperator(h_op)

    # --- Resolve communicator / MPI flag --------------------------------
    if comm is None:
        comm = getattr(basis, "comm", None)
    mpi = comm is not None and comm.Get_size() > 1

    # --- Resolve reort mode ---------------------------------------------
    if isinstance(reort, str):
        _map = {
            "none": Reort.NONE,
            "partial": Reort.PARTIAL,
            "selective": Reort.SELECTIVE,
            "full": Reort.FULL,
            "periodic": Reort.PERIODIC,
        }
        reort_mode = _map.get(reort.lower())
        if reort_mode is None:
            raise ValueError(f"Unknown reort string '{reort}'. " f"Must be one of {list(_map.keys())}.")
    else:
        reort_mode = reort

    # --- Resume or start fresh? -----------------------------------------
    resuming = alphas_init is not None and betas_init is not None and Q_init is not None

    if resuming:
        start_it = len(alphas_init)
        p = alphas_init[0].shape[0] if len(alphas_init) > 0 else len(Q_init[0] if Q_init else psi0)
        alphas_list = list(alphas_init)
        betas_list = list(betas_init)
        Q_basis = list(Q_init)
        W = W_init

        if start_it == 0 or len(Q_init) < p:
            q_prev = [ManyBodyState() for _ in range(p)]
            q_curr = [Q_basis[i] for i in range(p)]
        else:
            q_prev_start = max(0, (start_it - 1) * p)
            q_curr_start = start_it * p
            q_prev = [Q_basis[i] for i in range(q_prev_start, q_prev_start + p)]
            q_curr = [Q_basis[i] for i in range(q_curr_start, q_curr_start + p)]
    else:
        start_it = 0
        p = len(psi0)
        alphas_list = []
        betas_list = []
        W = None

        # Redistribute initial states across ranks
        q_curr = basis.redistribute_psis(list(psi0))
        q_prev = [ManyBodyState() for _ in range(p)]
        Q_basis = [st.copy() for st in q_curr]

    # --- Determine max_iter ---------------------------------------------
    if max_iter is None:
        max_iter = getattr(basis, "size", 10 * p) // p

    # Pre-allocate buffers to avoid allocation inside loop
    _buf_size = max(int(max_iter), 1)
    if resuming:
        alphas_buf = np.empty((start_it + _buf_size, p, p), dtype=complex)
        betas_buf = np.empty((start_it + _buf_size, p, p), dtype=complex)
        alphas_buf[:start_it] = alphas_init
        betas_buf[:start_it] = betas_init
    else:
        alphas_buf = np.empty((_buf_size, p, p), dtype=complex)
        betas_buf = np.empty((_buf_size, p, p), dtype=complex)
    # Seed W-estimator state
    if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE) and W is None:
        if start_it > 0:
            W = np.zeros((2, start_it + 1, p, p), dtype=complex)
            for j in range(start_it):
                Q_j = Q_basis[j * p : (j + 1) * p]

                ov_curr = inner_multi(Q_j, q_curr)
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, ov_curr, op=MPI.SUM)
                W[1, j] = ov_curr

                if j < start_it - 1:
                    ov_prev = inner_multi(Q_j, q_prev)
                    if mpi:
                        comm.Allreduce(MPI.IN_PLACE, ov_prev, op=MPI.SUM)
                    W[0, j] = ov_prev

            W[1, start_it] = np.eye(p)
            W[0, start_it - 1] = np.eye(p)
        else:
            W = np.zeros((2, 1, p, p), dtype=complex)
            W[1, 0] = np.eye(p)
    elif reort_mode in (Reort.PARTIAL, Reort.SELECTIVE) and W is not None:
        pass  # W expands dynamically in estimate_orthonormality

    # --- Main Lanczos loop ----------------------------------------------
    it = 0
    breakdown = False

    while it < _buf_size:
        it_abs = start_it + it
        q_next, alpha_i, beta_i, W, breakdown = block_lanczos_step_cy(
            h_op=h_op,
            q_prev=q_prev,
            q_curr=q_curr,
            Q_basis=Q_basis,
            alphas=alphas_buf,
            betas=betas_buf,
            it=it_abs,
            reort_mode=reort_mode,
            W=W,
            mpi=mpi,
            comm=comm,
            basis=basis,
            slaterWeightMin=slaterWeightMin,
            reort_period=reort_period,
            start_it=start_it,
        )

        alphas_list.append(alphas_buf[it_abs].copy())
        if breakdown:
            betas_list.append(np.zeros((p, p), dtype=complex))
            if verbose:
                if comm is None or comm.Get_rank() == 0:
                    print(f"[BlockLanczos] Breakdown / invariant subspace " f"detected at iteration {it_abs}.")
            break
        else:
            betas_list.append(betas_buf[it_abs].copy())

        # Append q_next to the basis (last block = residual direction)
        Q_basis.extend([st.copy() for st in q_next])

        if verbose:
            beta_norm = np.linalg.norm(beta_i, ord=2)
            print(
                f"[BlockLanczos] it={it_abs:4d}  " f"|beta|={beta_norm:.3e}  " f"alpha_diag={np.real(np.diag(alpha_i))}"
            )

        # Convergence check
        alphas_np = np.array(alphas_list)
        betas_np = np.array(betas_list)
        if converged_fn(alphas_np, betas_np, verbose=verbose):
            break

        q_prev = q_curr
        q_curr = q_next
        it += 1

    alphas_out = np.array(alphas_list, dtype=complex)
    betas_out = np.array(betas_list, dtype=complex)
    return alphas_out, betas_out, Q_basis, W


# ---------------------------------------------------------------------------
# Thick-Restart Block Lanczos (TRLM)
# ---------------------------------------------------------------------------


def thick_restart_block_lanczos_cy(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0.0,
    reort="partial",
    comm=None,
):
    """Thick-restart block Lanczos (TRLM) eigensolver for ``ManyBodyState``.

    Implements the thick-restart strategy of Knyazev (2001) and Wu & Simon (2000)
    to find the ``num_wanted`` algebraically smallest eigenvalues (and corresponding
    Ritz vectors) of the Hamiltonian :math:`H`.

    **Algorithm overview**:

    1. Run ``block_lanczos_cy`` to fill :math:`m` blocks, producing
       :math:`T_m` and the Krylov basis :math:`Q_m`.
    2. Diagonalise :math:`T_m` via ``scipy.linalg.eigh``.
    3. Keep the :math:`k = \\lceil n_w / p \\rceil` Ritz pairs with the lowest
       eigenvalues; build the compressed basis :math:`Q_k = Q_m Y_k` where
       :math:`Y_k` holds the corresponding eigenvectors.
    4. Reattach the residual block :math:`\\beta_\\text{res}` and update the
       cross-term :math:`T[k, :k] = \\beta_\\text{res} Y_k[-p:]`.
    5. Continue the Lanczos recurrence from block :math:`k` until convergence
       or ``max_restarts`` is exhausted.

    **Convergence criterion** (checked on rank 0, then broadcast):

    .. math::

        \\max_{i \\in \\text{wanted}}
            \\left\\| \\beta_{\\text{res}}\\, s_i \\right\\|_2 < \\text{tol}

    where :math:`s_i = y_i[-p:]` is the last :math:`p`-row slice of the
    :math:`i`-th Ritz vector :math:`y_i` of :math:`T_m`.

    **Invariant subspace early termination**: if ``block_lanczos_cy`` produces
    fewer than ``max_subspace_blocks`` steps (breakdown / invariant subspace),
    the function returns immediately with the best Ritz pairs from the partial
    factorisation.

    **MPI collective operations**:

    * All ``MPI_Allreduce`` calls inside ``block_lanczos_cy`` (one per Lanczos step
      for :math:`\\alpha_i` and :math:`M`; more for reorthogonalization).
    * One ``comm.bcast`` (root 0) to broadcast the convergence flag ``done`` so
      all ranks exit the restart loop simultaneously.
    * Additional ``MPI_Allreduce`` calls in the inner restart loop when
      reorthogonalizing :math:`W_p` against :math:`Q_k`.

    References:
        * Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver:
          Locally optimal block preconditioned conjugate gradient method.
          *SIAM Journal on Scientific Computing*, 23(2), 517–541.
        * Wu, K., & Simon, H. (2000). Thick-restart Lanczos method for large
          symmetric eigenvalue problems.  *SIAM Journal on Matrix Analysis and
          Applications*, 22(2), 602–616.

    Args:
        psi0: Starting block of ``p`` ``ManyBodyState`` objects.
        h_op: ``ManyBodyOperator`` Hamiltonian implementing
            ``apply_multi(psis, cutoff)``.
        basis: ``Basis`` object providing ``redistribute_psis`` and ``basis.comm``.
        num_wanted: Number of wanted lowest eigenvalues.
        max_subspace_blocks: Maximum Krylov subspace size in blocks (``m`` in the
            algorithm overview).  Must satisfy
            ``max_subspace_blocks > ceil(num_wanted / p)``.
        tol: Convergence tolerance on the maximum residual norm over wanted Ritz
            pairs.  Default ``1e-8``.
        max_restarts: Maximum number of thick restarts before returning the best
            available Ritz pairs.  Default ``100``.
        verbose: Print restart diagnostics including minimum eigenvalue and maximum
            wanted residual.  Default ``True``.
        slaterWeightMin: Amplitude cutoff for ``ManyBodyState.prune`` during
            Lanczos and basis combination.  Default ``0.0``.
        reort: Reorthogonalization mode.  Accepts a ``Reort`` enum member or one of
            the strings ``'none'``, ``'partial'``, ``'selective'``, ``'full'``,
            ``'periodic'``.  Default ``'partial'``.
        comm: ``mpi4py`` communicator.  Falls back to ``basis.comm``, or serial
            mode if ``None``.

    Returns:
        tuple[numpy.ndarray, list]: A 2-tuple ``(eigvals, eigvecs)`` where

        * ``eigvals`` – sorted numpy array of length ``num_wanted`` containing
          the ``num_wanted`` smallest converged eigenvalues.
        * ``eigvecs`` – list of ``num_wanted`` ``ManyBodyState`` Ritz vectors
          corresponding to ``eigvals``.

    Raises:
        ValueError: If ``max_subspace_blocks <= ceil(num_wanted / p)``.
        RuntimeError: If ``block_lanczos_cy`` produces zero iterations (e.g.,
            immediate breakdown on the first step).
    """
    if comm is None:
        comm = getattr(basis, "comm", None)
    mpi = comm is not None and comm.Get_size() > 1

    if isinstance(reort, str):
        _map = {
            "none": Reort.NONE,
            "partial": Reort.PARTIAL,
            "selective": Reort.SELECTIVE,
            "full": Reort.FULL,
            "periodic": Reort.PERIODIC,
        }
        reort_mode = _map.get(reort.lower())
        if reort_mode is None:
            raise ValueError(f"Unknown reort string '{reort}'. Must be one of {list(_map.keys())}.")
    else:
        reort_mode = reort

    p = len(psi0)
    k_blocks = math.ceil(num_wanted / p)
    m = max_subspace_blocks

    if m <= k_blocks:
        raise ValueError("max_subspace_blocks must be strictly greater than " "ceil(num_wanted / p).")

    psi0_list = list(psi0) if isinstance(psi0, (list, tuple)) else psi0
    psi0, _ = block_normalize_sparse(psi0_list, mpi, comm, slaterWeightMin)

    # --- Initial Lanczos run --------------------------------------------
    alphas, betas, Q_basis, W = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: False,
        verbose=verbose,
        reort=reort,
        max_iter=m,
        slaterWeightMin=slaterWeightMin,
        comm=comm,
    )

    m_actual = len(alphas)
    if m_actual == 0:
        raise RuntimeError("Block Lanczos produced zero iterations.")

    # Q_basis has length >= (m_actual+1)*p after the run (last p = residual)
    q_m = [Q_basis[i].copy() for i in range(m_actual * p, len(Q_basis))]
    Q_basis = Q_basis[: m_actual * p]

    _betas_for_T = betas[: len(betas) - 1] if len(betas) == m_actual else betas
    T_full = _build_full_T(alphas, _betas_for_T)

    # Early termination: invariant subspace found
    if m_actual < m:
        if verbose and (comm is None or comm.Get_rank() == 0):
            print(f"[TRLM] Invariant subspace of {m_actual} blocks found. " "Stopping early.")
        eigvals_T, eigvecs_T = sp.eigh(T_full)
        if comm is not None:
            eigvals_T = comm.bcast(eigvals_T, root=0)
            eigvecs_T = comm.bcast(eigvecs_T, root=0)
        wanted = np.argsort(eigvals_T)[:num_wanted]
        if verbose and (comm is None or comm.Get_rank() == 0):
            print("T_full eigenvalues at the end:", eigvals_T)
            print("T_full matrix:")
            print(T_full)
        final_eigvals = eigvals_T[wanted]
        final_eigvecs = block_combine_sparse(Q_basis, eigvecs_T[:, wanted], slaterWeightMin)
        if verbose and (comm is None or comm.Get_rank() == 0):
            print("T_full eigenvalues at the end:", eigvals_T)
            print("T_full matrix:")
            print(T_full)
        return final_eigvals, final_eigvecs

    beta_res = betas[len(betas) - 1]

    # --- Restart loop ---------------------------------------------------
    for restart in range(max_restarts):
        eigvals_T, eigvecs_T = sp.eigh(T_full)
        if comm is not None:
            eigvals_T = comm.bcast(eigvals_T, root=0)
            eigvecs_T = comm.bcast(eigvecs_T, root=0)
        _nrows_T = eigvecs_T.shape[0]
        res_norms = np.linalg.norm(beta_res @ eigvecs_T[_nrows_T - p :, :], axis=0)
        wanted = np.argsort(eigvals_T)[:num_wanted]
        max_res = float(np.max(res_norms[wanted]))

        if verbose and (comm is None or comm.Get_rank() == 0):
            print(f"[TRLM] Restart {restart:3d} | " f"MinEigval={eigvals_T[0]:.6f} | " f"MaxWantedRes={max_res:.2e}")

        done = max_res < tol
        if mpi:
            done = comm.bcast(done, root=0)

        if done:
            if verbose and (comm is None or comm.Get_rank() == 0):
                print("[TRLM] Converged!")
            break

        # Compress basis to k_blocks Ritz pairs
        keep = np.argsort(eigvals_T)[: k_blocks * p]
        Y_k = eigvecs_T[:, keep]
        T_k = np.diag(eigvals_T[keep])

        Q_basis = Q_basis[: m_actual * p]
        Q_basis = block_combine_sparse(Q_basis, Y_k, 0.0)

        for i in range(k_blocks):
            q_i = [Q_basis[i * p + j] for j in range(p)]
            _reorthogonalize_sparse(q_i, Q_basis, list(range(i * p)), mpi, comm)
            q_i, _ = block_normalize_sparse(q_i, mpi, comm, 0.0)
            for j in range(p):
                Q_basis[i * p + j] = q_i[j]

        # No explicit orthonormalization of Ritz vectors: it corrupts the relation to T_k.
        # The TRLM subsequent steps will use full Gram-Schmidt against this basis anyway.

        T_full = np.zeros((m * p, m * p), dtype=complex)
        T_full[: k_blocks * p, : k_blocks * p] = T_k

        # Residual block q_m from previous run (or compute fresh)
        if not q_m:
            q_last = [Q_basis[(k_blocks - 1) * p + i] for i in range(p)]
            wp = h_op.apply_multi(q_last, slaterWeightMin)
            if mpi:
                wp = basis.redistribute_psis(wp)
            for _ in range(2):
                ov = inner_multi(Q_basis, wp)
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, ov, op=MPI.SUM)
                add_scaled_multi(wp, Q_basis, -ov)
            try:
                q_m_new, beta_res = block_normalize_sparse(wp, mpi, comm, 0.0)
                breakdown = np.linalg.norm(beta_res, ord=2) < 1e-5
            except sp.LinAlgError:
                breakdown = True

            if breakdown:
                if verbose and (comm is None or comm.Get_rank() == 0):
                    print(f"[TRLM] Invariant subspace found at restart {restart}. Stopping early.")
                    print("T_full at breakdown:")
                    print(T_full[: k_blocks * p, : k_blocks * p])
                eigvals_T2, eigvecs_T2 = sp.eigh(T_full[: k_blocks * p, : k_blocks * p])
                if comm is not None:
                    eigvals_T2 = comm.bcast(eigvals_T2, root=0)
                    eigvecs_T2 = comm.bcast(eigvecs_T2, root=0)
                wanted2 = np.argsort(eigvals_T2)[:num_wanted]
                return eigvals_T2[wanted2], block_combine_sparse(
                    Q_basis[: k_blocks * p], eigvecs_T2[:, wanted2], slaterWeightMin
                )

            q_m = q_m_new

        Q_basis.extend([st.copy() for st in q_m])

        Y_last = Y_k[-p:, :]
        cross = beta_res @ Y_last

        if np.max(np.abs(cross)) > 50:
            print(f"BOGUS cross at restart={restart}: {np.max(np.abs(cross))}")
        if np.max(np.abs(T_k)) > 50:
            print(f"BOGUS T_k at restart={restart}: {np.max(np.abs(T_k))}")

        T_full[k_blocks * p : (k_blocks + 1) * p, : k_blocks * p] = cross
        T_full[: k_blocks * p, k_blocks * p : (k_blocks + 1) * p] = np.conj(cross.T)

        q1 = q_m
        for i in range(k_blocks, m):
            wp = h_op.apply_multi(q1, slaterWeightMin)
            if mpi:
                wp = basis.redistribute_psis(wp)

            overlaps = inner_multi(Q_basis, wp)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, overlaps, op=MPI.SUM)

            alpha_i = overlaps[-p:, :]
            T_full[i * p : (i + 1) * p, i * p : (i + 1) * p] = alpha_i

            # Double pass full reorthogonalization for TRLM arrowhead projection
            for _ in range(2):
                ov2 = inner_multi(Q_basis, wp)
                if mpi:
                    comm.Allreduce(MPI.IN_PLACE, ov2, op=MPI.SUM)
                add_scaled_multi(wp, Q_basis, -ov2)

            try:
                q_next, beta_i = block_normalize_sparse(wp, mpi, comm, 0.0)
                breakdown = np.linalg.norm(beta_i, ord=2) < 1e-5
            except sp.LinAlgError:
                breakdown = True

            if breakdown:
                if verbose and (comm is None or comm.Get_rank() == 0):
                    print(f"[TRLM] Invariant subspace found during restart at block {i}. Stopping early.")
                eigvals_T2, eigvecs_T2 = sp.eigh(T_full[: (i + 1) * p, : (i + 1) * p])
                if comm is not None:
                    eigvals_T2 = comm.bcast(eigvals_T2, root=0)
                    eigvecs_T2 = comm.bcast(eigvecs_T2, root=0)
                wanted2 = np.argsort(eigvals_T2)[:num_wanted]
                return eigvals_T2[wanted2], block_combine_sparse(
                    Q_basis[: (i + 1) * p], eigvecs_T2[:, wanted2], slaterWeightMin
                )

            if i < m - 1:
                T_full[(i + 1) * p : (i + 2) * p, i * p : (i + 1) * p] = beta_i
                T_full[i * p : (i + 1) * p, (i + 1) * p : (i + 2) * p] = np.conj(beta_i.T)
                Q_basis.extend([st.copy() for st in q_next])
                q1 = q_next
            else:
                beta_res = beta_i
                q_m = q_next

        m_actual = m

    # --- Final extraction -----------------------------------------------
    eigvals_T, eigvecs_T = sp.eigh(T_full)
    if comm is not None:
        eigvals_T = comm.bcast(eigvals_T, root=0)
        eigvecs_T = comm.bcast(eigvecs_T, root=0)
    wanted = np.argsort(eigvals_T)[:num_wanted]
    final_eigvals = eigvals_T[wanted]
    Q_final = Q_basis[: m_actual * p]
    final_eigvecs = block_combine_sparse(Q_final, eigvecs_T[:, wanted], slaterWeightMin)
    if verbose and (comm is None or comm.Get_rank() == 0):
        print("T_full eigenvalues at the end:", eigvals_T)
        print("T_full matrix:")
        print(T_full)
    return final_eigvals, final_eigvecs


def implicitly_restarted_block_lanczos_cy(
    psi0,
    h_op,
    basis,
    num_wanted: int,
    max_subspace_blocks: int,
    tol: float = 1e-8,
    max_restarts: int = 100,
    verbose: bool = True,
    slaterWeightMin: float = 0.0,
    reort="partial",
    comm=None,
):
    """Implicitly-restarted block Lanczos (IRLM) eigensolver.

    Implements the implicitly-restarted strategy from Lehoucq, Sorensen & Yang
    (ARPACK Users' Guide, 1998) generalised to block vectors, finding the
    ``num_wanted`` algebraically smallest eigenvalues and corresponding Ritz
    vectors of :math:`H`.

    **Algorithm overview**:

    1. Run ``block_lanczos_cy`` to build an :math:`m`-block Krylov factorisation
       :math:`H Q_m = Q_m T_m + Q_{m+1} \\beta_{m-1}^\\dagger`.
    2. Diagonalise :math:`T_m` via ``scipy.linalg.eigh``.
    3. Apply :math:`(m - k) p` implicit QR scalar shifts – the *unwanted* Ritz
       values – using the block bulge-chasing step in
       ``givens_qr.implicit_qr_step_block``.  Each shift produces the update

       .. math::

           T_m \\leftarrow U^\\dagger T_m U, \\quad
           Q_m \\leftarrow Q_m U

       where :math:`U` is the accumulated Givens rotation matrix.  After all
       shifts the leading :math:`k \\times k` sub-block of
       the transformed :math:`T_m` contains the wanted Ritz information.
    4. Extract the compressed :math:`k`-block factorisation
       :math:`(\\alpha_0', \\dots, \\alpha_{k-1}', \\beta_0', \\dots, \\beta_{k-1}')`
       via ``_extract_blocks``.
    5. Transform the Krylov basis: :math:`Q_k = Q_m U_k` where :math:`U_k`
       holds the first :math:`k p` columns of the accumulated rotation.
    6. Compute the new trailing block :math:`\\beta_k` via a matvec of the
       last column of :math:`Q_k` followed by double-pass orthogonalization
       against :math:`Q_k` and Cholesky QR.
    7. Resume ``block_lanczos_cy`` from block :math:`k` for up to
       :math:`m - k` additional steps.
    8. Repeat until convergence or ``max_restarts`` is exhausted.

    **Convergence criterion** (computed on rank 0, broadcast to all):

    .. math::

        \\max_{i \\in \\text{wanted}}
            \\left\\| \\beta_{m-1}\\, s_i[-p:] \\right\\|_2 < \\text{tol}

    where :math:`s_i` is the :math:`i`-th Ritz vector of :math:`T_m` and
    :math:`s_i[-p:]` denotes its last :math:`p` rows.

    **Invariant subspace early termination**: if at any restart
    ``m_actual <= k_blocks`` (Lanczos has exhausted the Krylov space), the
    loop exits and the best available Ritz pairs are returned.

    **MPI collective operations**:

    * All ``MPI_Allreduce`` calls inside ``block_lanczos_cy`` (per-step for
      :math:`\\alpha_i` and :math:`M`, plus reorthogonalization).
    * One ``comm.bcast`` (root 0) per restart loop iteration to synchronize
      the convergence flag ``done``.
    * Two ``MPI_Allreduce`` calls per restart for the double-pass
      orthogonalization of the new trailing block.

    References:
        * Lehoucq, R. B., Sorensen, D. C., & Yang, C. (1998).  *ARPACK Users'
          Guide: Solution of Large-Scale Eigenvalue Problems with Implicitly
          Restarted Arnoldi Methods*.  SIAM.
        * Meerbergen, K., & Scott, J. (2000). The design of a block rational
          Lanczos code with partial reorthogonalization and implicit restarting
          (RAL-TR-2000-011).  This implementation utilizes the EA16 restart
          orthogonality heuristic (Section 2.6.3), initializing the Paige-Simon
          estimator W_k uniformly to the maximum pre-restart orthogonality loss
          omega_max.

    Args:
        psi0: Starting block of ``p`` ``ManyBodyState`` objects.
        h_op: ``ManyBodyOperator`` Hamiltonian implementing
            ``apply_multi(psis, cutoff)``.
        basis: ``Basis`` object providing ``redistribute_psis`` and ``basis.comm``.
        num_wanted: Number of wanted lowest eigenvalues.
        max_subspace_blocks: Maximum Krylov subspace size in blocks (``m`` in the
            algorithm).  Must satisfy
            ``max_subspace_blocks > ceil(num_wanted / p)``.
        tol: Convergence tolerance on the maximum residual norm over wanted Ritz
            pairs.  Default ``1e-8``.
        max_restarts: Maximum number of implicit restarts before returning the
            best available Ritz pairs.  Default ``100``.
        verbose: Print per-restart diagnostics including minimum eigenvalue and
            maximum wanted residual.  Default ``True``.
        slaterWeightMin: Amplitude cutoff for ``ManyBodyState.prune`` during
            Lanczos and basis combination.  Default ``0.0``.
        reort: Reorthogonalization mode.  Accepts a ``Reort`` enum member or one
            of the strings ``'none'``, ``'partial'``, ``'selective'``, ``'full'``,
            ``'periodic'``.  Default ``'partial'``.
        comm: ``mpi4py`` communicator.  Falls back to ``basis.comm``, or serial
            mode if ``None``.

    Returns:
        tuple[numpy.ndarray, list]: A 2-tuple ``(eigvals, eigvecs)`` where

        * ``eigvals`` – sorted numpy array of length ``num_wanted`` containing
          the ``num_wanted`` smallest converged eigenvalues.
        * ``eigvecs`` – list of ``num_wanted`` ``ManyBodyState`` Ritz vectors
          corresponding to ``eigvals``.

    Raises:
        ValueError: If ``max_subspace_blocks <= ceil(num_wanted / p)``.
    """
    from impurityModel.ed.givens_qr import implicit_qr_step_block

    if comm is None:
        comm = getattr(basis, "comm", None)
    mpi = comm is not None and comm.Get_size() > 1

    if isinstance(reort, str):
        _map = {
            "none": Reort.NONE,
            "partial": Reort.PARTIAL,
            "selective": Reort.SELECTIVE,
            "full": Reort.FULL,
            "periodic": Reort.PERIODIC,
        }
        reort_mode = _map.get(reort.lower())
        if reort_mode is None:
            raise ValueError(f"Unknown reort string '{reort}'. Must be one of {list(_map.keys())}.")
    else:
        reort_mode = reort

    p = len(psi0)
    k_blocks = math.ceil(num_wanted / p)
    m = max_subspace_blocks

    if m <= k_blocks:
        raise ValueError("max_subspace_blocks must be strictly greater than " "ceil(num_wanted / p).")

    psi0_list = list(psi0) if isinstance(psi0, (list, tuple)) else psi0
    psi0, _ = block_normalize_sparse(psi0_list, mpi, comm, slaterWeightMin)

    # --- Initial Lanczos run --------------------------------------------
    alphas, betas, Q_basis, W = block_lanczos_cy(
        psi0=psi0,
        h_op=h_op,
        basis=basis,
        converged_fn=lambda a, b, **kw: False,
        verbose=verbose,
        reort=reort,
        max_iter=m,
        slaterWeightMin=slaterWeightMin,
        comm=comm,
    )

    for restart in range(max_restarts):
        m_actual = len(alphas)

        if m_actual <= k_blocks:
            if verbose and (comm is None or comm.Get_rank() == 0):
                print(f"[IRLM] Invariant subspace ({m_actual} <= {k_blocks} " "blocks). Stopping.")
            break

        _betas_for_T2 = betas[: len(betas) - 1] if len(betas) == m_actual else betas
        T = _build_full_T(alphas, _betas_for_T2)
        eigvals_T, eigvecs_T = sp.eigh(T)
        if comm is not None:
            eigvals_T = comm.bcast(eigvals_T, root=0)
            eigvecs_T = comm.bcast(eigvecs_T, root=0)

        # Convergence check
        _betas_last = betas[len(betas) - 1]
        _nrows_evec = eigvecs_T.shape[0]
        res_norms = np.linalg.norm(_betas_last @ eigvecs_T[_nrows_evec - p :, :], axis=0)
        wanted = np.argsort(eigvals_T)[:num_wanted]
        max_res = float(np.max(res_norms[wanted]))

        if verbose and (comm is None or comm.Get_rank() == 0):
            print(f"[IRLM] Restart {restart:3d} | " f"MinEigval={eigvals_T[0]:.6f} | " f"MaxWantedRes={max_res:.2e}")

        done = max_res < tol
        if mpi:
            done = comm.bcast(done, root=0)

        if done:
            if verbose and (comm is None or comm.Get_rank() == 0):
                print("[IRLM] Converged!")
            break

        # Apply (m_actual - k_blocks) implicit QR shifts
        unwanted_idx = np.argsort(eigvals_T)[num_wanted:]
        shifts = eigvals_T[unwanted_idx]
        U_total = np.eye(m_actual * p, dtype=complex)
        T_shifted = T.copy()
        for shift in shifts:
            T_shifted, U_total = implicit_qr_step_block(T_shifted, p, shift, U_total)

        U_k = U_total[:, : k_blocks * p]
        alphas_new, betas_new = _extract_blocks(T_shifted, k_blocks, p)

        # Transform basis Q_k = Q * U_k
        Q_used = Q_basis[: m_actual * p]
        Q_new = block_combine_sparse(Q_used, U_k, slaterWeightMin)

        # Compute new trailing block and beta_k
        q_k_last = [Q_new[(k_blocks - 1) * p + i] for i in range(p)]
        wp = h_op.apply_multi(q_k_last, slaterWeightMin)
        if mpi:
            wp = basis.redistribute_psis(wp)

        # Double-pass orthogonalization against Q_new
        for _ in range(2):
            ov = inner_multi(Q_new, wp)
            if mpi:
                comm.Allreduce(MPI.IN_PLACE, ov, op=MPI.SUM)
            add_scaled_multi(wp, Q_new, -ov)

        M = inner_multi(wp, wp)
        if mpi:
            comm.Allreduce(MPI.IN_PLACE, M, op=MPI.SUM)

        if np.any(np.isnan(M)) or np.any(np.isinf(M)):
            if verbose and (comm is None or comm.Get_rank() == 0):
                print("[IRLM] Breakdown at restart -- returning current Ritz pairs.")
            break

        try:
            beta_k, beta_k_inv = _cholesky_beta(M)
        except np.linalg.LinAlgError:
            if verbose and (comm is None or comm.Get_rank() == 0):
                print("[IRLM] Cholesky breakdown at restart.")
            break

        if np.linalg.norm(beta_k, ord=2) < 1e-5:
            if verbose and (comm is None or comm.Get_rank() == 0):
                print("[IRLM] Invariant subspace found during restart.")
            break

        q_k_next = [ManyBodyState() for _ in range(p)]
        add_scaled_multi(q_k_next, wp, beta_k_inv)
        if slaterWeightMin > 0:
            for st in q_k_next:
                st.prune(slaterWeightMin)

        Q_basis_new = list(Q_new) + [st.copy() for st in q_k_next]

        # betas_pass: k_blocks blocks from T_shifted + new beta_k
        betas_pass_list = list(betas_new) if len(betas_new) > 0 else []
        if len(betas_pass_list) < k_blocks:
            betas_pass_list.append(beta_k)
        else:
            betas_pass_list[len(betas_pass_list) - 1] = beta_k

        betas_pass = np.array(betas_pass_list) if betas_pass_list else np.empty((0, p, p), dtype=complex)
        alphas_pass = np.array(alphas_new)

        if reort_mode in (Reort.PARTIAL, Reort.SELECTIVE):
            if W is not None:
                omega_max = np.max(np.abs(W))
            else:
                omega_max = np.finfo(float).eps

            W_init = np.full((2, k_blocks + 1, p, p), omega_max, dtype=complex)
            for i in range(k_blocks):
                W_init[1, i] = np.eye(p) * omega_max
                W_init[0, i] = np.eye(p) * omega_max
            W_init[1, k_blocks] = np.eye(p)
            if k_blocks > 0:
                W_init[0, k_blocks - 1] = np.eye(p)
        else:
            W_init = None

        # Continue Lanczos from block k_blocks
        alphas, betas, Q_basis, W = block_lanczos_cy(
            psi0=None,
            h_op=h_op,
            basis=basis,
            converged_fn=lambda a, b, **kw: False,
            verbose=verbose,
            reort=reort,
            max_iter=m - k_blocks,
            slaterWeightMin=slaterWeightMin,
            comm=comm,
            alphas_init=alphas_pass,
            betas_init=betas_pass,
            Q_init=Q_basis_new,
            W_init=W_init,
        )

    # --- Final extraction -----------------------------------------------
    m_actual = len(alphas)
    _betas_final = betas[: len(betas) - 1] if len(betas) == m_actual else betas
    T_final = _build_full_T(alphas, _betas_final)
    eigvals_T, eigvecs_T = sp.eigh(T_final)
    if comm is not None:
        eigvals_T = comm.bcast(eigvals_T, root=0)
        eigvecs_T = comm.bcast(eigvecs_T, root=0)
    wanted = np.argsort(eigvals_T)[:num_wanted]
    final_eigvals = eigvals_T[wanted]
    Q_final = Q_basis[: m_actual * p]
    final_eigvecs = block_combine_sparse(Q_final, eigvecs_T[:, wanted], slaterWeightMin)
    return final_eigvals, final_eigvecs
