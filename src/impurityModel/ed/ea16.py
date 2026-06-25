"""Shared, path-agnostic numerics for the EA16 implicitly restarted block Lanczos.

Implements the small-matrix pieces of Meerbergen & Scott, *The design of a block
rational Lanczos code with partial reorthogonalization and implicit restarting*,
RAL-TR-2000-011 (EA16):

* per-Ritz residual norms ``||B_{k+1} E_k^T z||``                     (§2.1)
* the eq. (15) acceptance / convergence test                          (§3.2.4)
* locking-aware retain/purge selection of Ritz pairs                  (§2.2, §2.3.2)
* explicit purging in the Ritz basis with re-banding, eq. (6)         (§2.2.1)

These operate only on the small dense block-tridiagonal ``T`` and its eigenpairs
``(evals, Z)`` (all ``O(mb)`` sized) plus scalar control parameters, so they are
reused unchanged by both the array (``BlockLanczosArray``) and ManyBodyState
(``BlockLanczos``) IRLM drivers. The basis-carrying steps (forming Ritz vectors,
applying the returned combination matrix ``C``) stay in the drivers, expressed
through the path-agnostic ``block_*`` helpers.

The explicit purge of eq. (6) is mathematically the exact-shift implicitly-restarted
Lanczos of Theorem 2.1 / Algorithm 2.2 (Morgan 1996: an exact shift purges that Ritz
vector), used here in place of the block bulge-chasing QR step because the latter does
not preserve the staircase structure the Sorensen residual formula requires for more
than one retained block.
"""

import numpy as np

EPS = np.finfo(float).eps


def ritz_residual_norms(beta_last, Z, p):
    """Residual norm ``||beta_last @ z_i[-p:]||_2`` for every Ritz vector column.

    This is ``||B_{k+1} E_k^T z_i||`` of EA16 §2.1 — the exact residual of the Ritz
    pair ``(theta_i, V_k z_i)`` of the standard problem.

    Args:
        beta_last: Trailing ``(p, p)`` off-diagonal (residual-coupling) block.
        Z: ``(m*p, m*p)`` matrix of Ritz vectors (columns), from ``eigh(T)``.
        p: Block size ``b``.

    Returns:
        numpy.ndarray: Length ``m*p`` real array of residual norms.
    """
    nrows = Z.shape[0]
    return np.linalg.norm(beta_last @ Z[nrows - p :, :], axis=0)


def operator_norm_estimate(theta, theta_locked=None):
    """Estimate ``||T_k||`` as the largest-magnitude Ritz value (eq. 15 first term).

    Includes already-locked eigenvalues so the norm estimate does not shrink as the
    active subspace is deflated.
    """
    vals = [float(np.max(np.abs(theta)))] if len(theta) else [0.0]
    if theta_locked is not None and len(theta_locked):
        vals.append(float(np.max(np.abs(theta_locked))))
    return max(vals)


def acceptance_tol(theta_i, tnorm, cntl2, cntl3, u=EPS):
    """EA16 eq. (15) right-hand side for one Ritz value ``theta_i``.

    ``u*||T_k|| + |CNTL(2)| + |CNTL(3)|*|theta_i|`` — a blend of the minimum
    meaningful residual in finite precision (first term) with absolute (``CNTL(2)``)
    and relative (``CNTL(3)``) backward-error tolerances.
    """
    return u * tnorm + abs(cntl2) + abs(cntl3) * abs(theta_i)


def select_restart_indices(theta, n_keep, locked_local, which="smallest"):
    """Partition Ritz values into *kept* (retained) and *shifted* (purged) sets.

    Implements the locking-aware exact-shift selection of EA16 §2.2/§2.3.2: the
    ``n_keep`` lowest-priority-to-remove directions that are **not** just-locked are
    retained at the top of the restarted factorization; everything else — the
    just-locked converged Ritz values (purged into the locked set, cf. Morgan 1996:
    an exact shift purges that Ritz vector) and the unwanted extremal values — is
    used as implicit shifts.

    Args:
        theta: Length ``m*p`` real Ritz values of the active ``T``.
        n_keep: Number of Ritz directions to retain (``k_blocks * p``).
        locked_local: Indices (into ``theta``) of Ritz pairs converged/locked this
            restart; always shifted away.
        which: ``"smallest"`` (wanted = algebraically smallest) or ``"largest"``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(kept_idx, shift_vals)`` where
        ``kept_idx`` are the retained columns (length ``n_keep``) and ``shift_vals``
        are the real shift values (length ``m*p - n_keep``).
    """
    order = np.argsort(theta.real)
    if which == "largest":
        order = order[::-1]
    locked_set = set(int(i) for i in locked_local)
    ranked = [int(i) for i in order if int(i) not in locked_set]
    kept_idx = np.array(ranked[:n_keep], dtype=int)
    kept_set = set(kept_idx.tolist())
    shift_idx = np.array([int(i) for i in range(len(theta)) if int(i) not in kept_set], dtype=int)
    return kept_idx, theta[shift_idx].real


def _extract_blocks(T, nb, p):
    """Read the diagonal (``alpha``) and sub-diagonal (``beta``) blocks of band ``T``."""
    alphas = np.zeros((nb, p, p), dtype=complex)
    betas = np.zeros((max(nb - 1, 0), p, p), dtype=complex)
    for i in range(nb):
        alphas[i] = T[i * p : (i + 1) * p, i * p : (i + 1) * p]
        if i < nb - 1:
            betas[i] = T[(i + 1) * p : (i + 2) * p, i * p : (i + 1) * p]
    return alphas, betas


def _block_tridiagonalize(M, V0, p):
    """Block-Lanczos tridiagonalization of a small dense Hermitian ``M``.

    Builds an orthonormal basis ``Y`` (full reorthogonalization, since ``M`` is tiny)
    with ``Y[:, :p] = V0`` such that ``Y^H M Y`` is block-tridiagonal. Used by
    :func:`purge_restart` to re-band the purged Ritz factorization.

    Args:
        M: ``(n, n)`` Hermitian matrix (``n`` a multiple of ``p``).
        V0: ``(n, p)`` orthonormal starting block.
        p: Block size.

    Returns:
        numpy.ndarray: ``(n, n)`` orthonormal ``Y``.
    """
    n = M.shape[0]
    nb = n // p
    Q = [V0]
    Qfull = V0.copy()
    q_prev = None
    beta_prev = None
    for j in range(nb):
        w = M @ Q[-1]
        a = np.conj(Q[-1].T) @ w
        w = w - Q[-1] @ a
        if j > 0:
            w = w - q_prev @ np.conj(beta_prev.T)
        for _ in range(2):  # full reorthogonalization (cheap; n is small)
            w = w - Qfull @ (np.conj(Qfull.T) @ w)
        if j < nb - 1:
            qn, r = np.linalg.qr(w)
            q_prev, beta_prev = Q[-1], r
            Q.append(qn)
            Qfull = np.concatenate([Qfull, qn], axis=1)
    return Qfull


def purge_restart(evals, Z, beta_last, p, kept_idx):
    """Compress an order-``m`` factorization to the wanted Ritz pairs (EA16 §2.2.1).

    Explicit purging in the Ritz basis, eq. (6): keeps the ``kept_idx`` Ritz pairs
    ``(Lambda, X = V_m Z[:, kept])`` and re-bands the result so the resumed recurrence
    is a valid block-tridiagonal Lanczos factorization

    .. math::

        A\\,(V_m C) = (V_m C)\\,T^+ + (V_{m+1}\\,\\beta^+) E_k^T ,

    where the residual is confined to the trailing block by a reverse block-Lanczos
    re-banding of the purged arrowhead ``[[\\Lambda], [\\beta_{last} Z_{last,kept}]]``.
    This is mathematically the exact-shift implicit restart (Morgan 1996), but is
    numerically robust for any number of retained blocks (the block bulge-chase loses
    the staircase structure the Sorensen residual formula requires).

    Args:
        evals: Length ``m*p`` real Ritz values of the active ``T``.
        Z: ``(m*p, m*p)`` Ritz vectors (``eigh(T)`` eigenvectors).
        beta_last: Trailing ``(p, p)`` residual-coupling block.
        p: Block size ``b``.
        kept_idx: Indices of the Ritz pairs to retain (length ``n_keep = k*p``).

    Returns:
        tuple: ``(C, beta_new, alphas_new, betas_new)`` where ``C`` (``m*p x n_keep``)
        is the basis-combination matrix (``V_new = V_m C``), ``beta_new`` (``p x p``)
        the new residual coupling, and ``alphas_new``/``betas_new`` the band blocks of
        the restarted ``T^+``.
    """
    kept_idx = np.asarray(kept_idx, dtype=int)
    n_keep = kept_idx.size
    nb = n_keep // p
    m = Z.shape[0] // p

    W = Z[:, kept_idx]
    Lam = np.diag(evals[kept_idx].real).astype(complex)
    S = beta_last @ Z[(m - 1) * p :, kept_idx]  # (p, n_keep) residual->kept coupling

    P1, _ = np.linalg.qr(np.conj(S.T))  # (n_keep, p) orthonormal
    Y_fwd = _block_tridiagonalize(Lam, P1, p)
    # Reverse the block order so the residual coupling lands on the trailing block.
    Y = np.concatenate([Y_fwd[:, (nb - 1 - j) * p : (nb - j) * p] for j in range(nb)], axis=1)

    T_band = np.conj(Y.T) @ Lam @ Y
    beta_new = (S @ Y)[:, n_keep - p :]
    alphas_new, betas_new = _extract_blocks(T_band, nb, p)
    C = W @ Y
    return C, beta_new, alphas_new, betas_new
