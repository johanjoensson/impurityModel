import numpy as np
from numpy.linalg import eigh, svd
from scipy.linalg import block_diag

from impurityModel.ed.BlockLanczosArray import scalar_lanczos

# ---------------- helpers ----------------


# ---------------- helpers ----------------


def _psd_eigsqrt_and_pinv(G, tol=1e-12):
    """
    Compute the Hermitian positive-semidefinite (PSD) square root and its pseudo-inverse.

    Given a Hermitian matrix G ≽ 0 (typically a Gram matrix), this returns matrices:
        B       = G^{1/2}      (principal Hermitian square root)
        B_pinv  = G^{-1/2}     (Moore–Penrose pseudo-inverse of the square root)
        rank    = numerical rank of G

    Small or negative eigenvalues below a relative/absolute tolerance are clipped to zero,
    making this routine robust for nearly singular Gram matrices.

    Parameters
    ----------
    G : ndarray (n, n)
        Hermitian positive-semidefinite matrix.
    tol : float, optional
        Absolute lower bound for eigenvalue cutoff. Effective threshold is
        max(tol, max(spectrum(G)) * 1e-12).

    Returns
    -------
    B : ndarray (n, n)
        Hermitian square root of G (B @ B^H ≈ G).
    B_pinv : ndarray (n, n)
        Hermitian pseudo-inverse square root (B_pinv @ G @ B_pinv ≈ I on the support of G).
    rank : int
        Effective numerical rank of G after eigenvalue thresholding.

    Notes
    -----
    This is safer than a Cholesky decomposition for ill-conditioned or rank-deficient
    matrices and ensures B, B_pinv remain Hermitian PSD.
    """
    Gh = 0.5 * (G + G.conj().T)
    s, U = eigh(Gh)
    s = np.clip(s, 0.0, None)
    thr = max(tol, (s.max() if s.size else 0.0) * 1e-12)
    mask = s > thr
    r = int(mask.sum())
    if r == 0:
        z = np.zeros_like(Gh)
        return z, z, 0
    Ur = U[:, mask]
    sr = s[mask]
    B = Ur @ np.diag(np.sqrt(sr)) @ Ur.conj().T
    B_pinv = Ur @ np.diag(1.0 / np.sqrt(sr)) @ Ur.conj().T
    return B, B_pinv, r


def _axpy_last(A, C):
    """
    Perform a matrix multiplication along the last axis of A and the first axis of C.
    """
    # A shape (..., bA), C shape (bA, k) -> (..., k)
    return np.tensordot(A, C, axes=([-1], [0]))


def _zeros_like_last(X, b):
    """
    Create a zero array like X, but with the last axis dimension replaced by b.
    """
    shape = list(X.shape)
    shape[-1] = b
    return np.zeros(shape, dtype=X.dtype)


def _truncate_last(X, k):
    """
    Truncate the last axis of array X to size k.
    """
    return X[..., :k].copy()


def _orthonormalize_block(W, bases, gram, reorth=True, tol=1e-12):
    """
    Orthonormalize a block of vectors W against an existing set of orthonormal bases.

    This routine takes a block W (matrix or tensor whose last axis indexes columns),
    orthogonalizes it with respect to all blocks in `bases` using the inner product
    defined by the callable `gram(A, B)`, and normalizes it via the Hermitian square
    root of its local Gram matrix.

    Handles rank deficiency gracefully by projecting out linearly dependent directions.

    Parameters
    ----------
    W : ndarray
        Block of candidate vectors to be orthonormalized. Shape (..., b_W).
    bases : list of ndarrays
        List of previously orthonormalized blocks (same leading dimensions as W).
    gram : callable
        Function gram(A, B) -> ndarray (b_A, b_B) computing the Hermitian inner product.
        For the Euclidean case, this is A.conj().T @ B.
    reorth : bool, optional
        If True, perform a second reorthogonalization pass (recommended for stability).
    tol : float, optional
        Tolerance for numerical rank detection when normalizing.

    Returns
    -------
    Q : ndarray
        Orthonormalized block spanning the independent directions of W.
    B : ndarray
        Hermitian positive-definite square root of the Gram matrix of W (local overlap).
    rank : int
        Effective number of independent columns retained in Q.

    Notes
    -----
    The algorithm:
      1. Removes projections of W onto all previous bases (1 or 2 passes if reorth=True).
      2. Computes the local Gram matrix G = gram(W, W).
      3. Diagonalizes G to obtain its square root B = G^{1/2} and pseudo-inverse B^{-1}.
      4. Constructs Q = W @ B^{-1}, ensuring Q†Q = I on the support of G.
         Rank-deficient directions are automatically dropped.

    Used inside the block Lanczos iteration to keep each block orthonormal under
    an arbitrary inner product, even if near-linear dependencies appear.
    """
    # reorth against existing bases
    if bases:
        passes = 2 if reorth else 1
        for _ in range(passes):
            for P in bases:
                C = gram(P, W)  # shape (bP, bW)
                W = _axpy_last(P, -C) + W
    # local Gram and sqrt
    G = gram(W, W)
    B, B_pinv, r = _psd_eigsqrt_and_pinv(G, tol=tol)
    if r == 0:
        return None, None, 0
    Q = _axpy_last(W, B_pinv)  # W @ B_pinv
    # compress to rank r if needed
    if Q.shape[-1] != r:
        s, U = eigh(0.5 * (G + G.conj().T))
        thr = max(tol, (s.max() if s.size else 0.0) * 1e-12)
        Uc = U[:, s > thr]  # bW x r
        Q = _axpy_last(Q, Uc)  # keep r independent dirs
        B = Uc.conj().T @ B @ Uc
    return Q, B, Q.shape[-1]


# ---------------- definitive generic block Lanczos ----------------


def block_lanczos(apply_op, gram, Q0, K=None, reorth=True, tol=1e-12, cap_dim=None):
    """
    Generic block Lanczos for Hermitian problems with a custom inner product.

    Inputs
        apply_op: function(X) -> same shape as X, operator application
        gram    : function(A, B) -> A^H * B in the chosen inner product, shape (bA, bB)
        Q0      : initial block, last axis indexes block columns
        K       : max iterations, default very large
        reorth  : do second-pass reorthogonalization
        tol     : Gram tolerance
        cap_dim : optional cap on total accumulated columns

    Returns
        A_blocks, B_blocks, Q_blocks, R0
        A_k has shape (b_k, b_k)
        B_k has shape (b_{k+1}, b_k)
        Q_blocks is a list of blocks
        R0 is the initial block sqrt Gram
    """
    if K is None:
        K = 10**9

    Q_blocks = []
    Q0c = Q0.copy()
    Q0o, R0, r0 = _orthonormalize_block(Q0c, [], gram, reorth=reorth, tol=tol)
    if r0 == 0:
        return [], [], [], None
    Q_blocks.append(Q0o)

    A_blocks, B_blocks = [], []
    Qkm1 = _zeros_like_last(Q0o, 0)
    Bkm1 = np.zeros((0, Q0o.shape[-1]), dtype=np.complex128)
    total_cols = Q0o.shape[-1]

    for _ in range(K):
        Qk = Q_blocks[-1]
        HQk = apply_op(Qk)
        Ak = gram(Qk, HQk)
        Ak = 0.5 * (Ak + Ak.conj().T)
        A_blocks.append(Ak)

        # W = H Qk - Qk Ak - Q_{k-1} B_{k-1}^H
        W = HQk + _axpy_last(Qk, -Ak)
        if Bkm1.size:
            W = W + _axpy_last(Qkm1, -Bkm1.conj().T)

        Qnext, Bk, rk = _orthonormalize_block(W, Q_blocks, gram, reorth=reorth, tol=tol)
        if rk == 0:
            break

        if cap_dim is not None and total_cols + rk > cap_dim:
            keep = max(0, cap_dim - total_cols)
            if keep == 0:
                break
            Qnext = _truncate_last(Qnext, keep)
            Bk = Bk[:keep, :]
            rk = keep

        B_blocks.append(Bk)
        Qkm1, Bkm1 = Qk, Bk
        Q_blocks.append(Qnext)
        total_cols += rk

    return A_blocks, B_blocks, Q_blocks, R0


# ---------------- convenience wrappers that replace variants ----------------
def block_lanczos_matrix(H, r, max_steps=None, seed=None, reorth=True, tol=1e-12):
    """
    Dense or sparse Hermitian H in C^{n x n}.
    Returns A_blocks, B_blocks, Q with Q shape n x m.
    """
    n = H.shape[0]
    if r <= 0:
        return [], [], np.eye(n)[:, :0]
    if max_steps is None:
        max_steps = int(np.ceil(n / r))

    Q0 = np.eye(n, r, dtype=np.complex128) if seed is None else np.asarray(seed, dtype=np.complex128)

    def apply_op(X):
        """
        Apply the Hamiltonian matrix H to vectors X.
        """
        return H @ X

    def gram(A, B):
        """
        Compute the inner product matrix of two vector blocks.
        """
        return A.conj().T @ B

    A, B, Qblocks, R = block_lanczos(apply_op, gram, Q0, K=max_steps, reorth=reorth, tol=tol, cap_dim=n)
    Q = np.concatenate(Qblocks, axis=1) if Qblocks else np.zeros((n, 0), dtype=np.complex128)
    return A, B, Q, R


def block_lanczos_grid(x_vals, weight_mats, K, b, reorth=True, tol=1e-12):
    """
    Block Lanczos tridiagonalization for a weighted grid representation.

    Performs the block Lanczos algorithm where the operator acts as
    multiplication by x on a discrete grid, and the inner product is defined
    by position-dependent Hermitian weights μ_s at each grid point.

    This generates the block three-term recurrence for matrix-valued
    orthogonal polynomials Φ_k(x) satisfying:
        x Φ_k(x) = Φ_k(x) A_k + Φ_{k-1}(x) B_{k-1}^† + Φ_{k+1}(x) B_k

    Parameters
    ----------
    x_vals : ndarray (S,)
        Grid points representing the variable x (e.g. energy mesh).
    weight_mats : list[ndarray (M, M)]
        Hermitian positive-semidefinite weight matrices μ_s defining the
        inner product ⟨Φ, Ψ⟩ = Σ_s Φ(s)† μ_s Ψ(s).
    K : int
        Maximum number of Lanczos steps (recurrence depth).
    b : int
        Block size (number of initial orthogonal functions).
    reorth : bool, optional
        If True, perform a second reorthogonalization pass for numerical stability.
    tol : float, optional
        Tolerance for rank detection when orthonormalizing blocks.

    Returns
    -------
    A_blocks : list[ndarray (b_k, b_k)]
        Diagonal block matrices of the block-tridiagonal representation.
    B_blocks : list[ndarray (b_{k+1}, b_k)]
        Off-diagonal block matrices of the recurrence.
    Phi_blocks : list[ndarray (S, M, b_k)]
        List of orthonormal block functions Φ_k on the grid.
    M0 : ndarray (M, M)
        Total weight matrix M0 = Σ_s μ_s (symmetrized).

    Notes
    -----
    This routine is equivalent to constructing a block-tridiagonal representation
    of the multiplication operator x in a weighted inner-product space. The
    resulting (A, B) blocks can be used to approximate integrals or moments of
    spectral densities, or to build continued-fraction representations of
    Green's functions from grid data.
    """
    x = np.asarray(x_vals)
    mu = [np.asarray(W) for W in weight_mats]
    S = len(x)
    M = mu[0].shape[0]

    Phi0 = np.zeros((S, M, b), dtype=np.complex128)
    eyeMb = np.eye(M, b, dtype=np.complex128)
    for s in range(S):
        Phi0[s] = eyeMb

    mu_stack = np.stack(mu, axis=0)

    def apply_op(Phi):
        """
        Multiply the polynomial block Phi by grid coordinates x.
        """
        return x[:, None, None] * Phi

    def gram(Phi, Psi):
        """
        Compute the inner product of blocks Phi and Psi with the weight matrices.
        """
        acc = np.einsum("smb,smn,snk->bk", Phi.conj(), mu_stack, Psi, optimize=True)
        return 0.5 * (acc + acc.conj().T)

    A, B, Phi_blocks, _ = block_lanczos(apply_op, gram, Phi0, K=K, reorth=reorth, tol=tol, cap_dim=None)
    M0 = sum(mu).astype(np.complex128)
    M0 = 0.5 * (M0 + M0.conj().T)
    return A, B, Phi_blocks, M0


def get_double_chain_transform(h_spin, Nelec):
    """
    Performs the 5-step Natural Orbital transformation, yielding a Hamiltonian
    with a double-chain structure and the corresponding unitary transformation matrix C.

    Args:
        h_spin (np.ndarray): The initial single-particle Hamiltonian (M x M).
        u (float): The on-site interaction strength.
        Nelec (int): The number of electrons in the spin sector (Nelec_total / 2).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - h_final_matrix (np.ndarray): The final Hamiltonian in the double-chain basis.
            - C_total (np.ndarray): The total unitary transformation matrix C.
    """
    M = h_spin.shape[0]

    # --- (i) Mean-Field ---
    h_mf = h_spin.copy()
    # h_mf[0, 0] += u * 0.5

    toto = h_spin.copy()
    # print("original toto = ")
    # print(np.real(toto))
    # --- (ii) Natural Orbital Basis for the Bath ---
    e_mf, C_mf = eigh(h_mf)
    rho_mf = C_mf[:, :Nelec] @ C_mf[:, :Nelec].T
    rho_bath = rho_mf[1:, 1:]
    n_no, W = eigh(rho_bath)
    print(f"n_no = {n_no}")

    occupations_dist_from_integer = np.minimum(n_no, 1 - n_no)
    b_idx = np.argmax(occupations_dist_from_integer)

    other_indices = [i for i in range(M - 1) if i != b_idx]
    # Sorting ensures a deterministic basis ordering
    filled_indices = sorted([i for i in other_indices if n_no[i] > 0.5])
    empty_indices = sorted([i for i in other_indices if n_no[i] <= 0.5])

    ordered_bath_indices = [b_idx] + filled_indices + empty_indices
    W_ordered = W[:, ordered_bath_indices]
    C1 = block_diag(1, W_ordered)

    toto = C1.conj().T @ toto @ C1
    print("after bath no toto = ")
    print(np.real(toto))
    # --- (iii) Bonding/Anti-bonding Transformation ---
    rho_no_basis = C1.T @ rho_mf @ C1
    rho_ib = rho_no_basis[:2, :2]
    e_bond, U_bond = eigh(rho_ib)
    print(f"e_bond = {e_bond}")
    C2 = np.identity(M)
    C2[:2, :2] = U_bond

    C_upto_decoupled = C1 @ C2

    toto = C2.conj().T @ toto @ C2
    print("after bond/antibond toto = ")
    print(np.real(toto))

    # --- (iv) Lanczos Tridiagonalization (Chain Transformation) ---
    h_decoupled = C_upto_decoupled.T @ h_mf @ C_upto_decoupled

    num_filled = len(filled_indices)
    conduction_indices = [0] + list(range(2 + num_filled, M))
    valence_indices = [1] + list(range(2, 2 + num_filled))

    h_conduction = h_decoupled[np.ix_(conduction_indices, conduction_indices)]
    v0_c = np.zeros(h_conduction.shape[0])
    v0_c[0] = 1.0
    T_c, Q_c = scalar_lanczos(h_conduction, v0_c)

    h_valence = h_decoupled[np.ix_(valence_indices, valence_indices)]
    v0_v = np.zeros(h_valence.shape[0])
    v0_v[0] = 1.0
    T_v, Q_v = scalar_lanczos(h_valence, v0_v)

    C_lanczos = np.identity(M)
    if Q_c.shape[1] > 0:
        C_lanczos[np.ix_(conduction_indices, conduction_indices)] = Q_c
    if Q_v.shape[1] > 0:
        C_lanczos[np.ix_(valence_indices, valence_indices)] = Q_v

    # C_to_chains transforms from original basis to the basis of Lanczos vectors
    # (permuted).
    C_to_chains = C_upto_decoupled @ C_lanczos

    toto = C_lanczos.conj().T @ toto @ C_lanczos
    print("afterlanczos toto = ")
    print(np.real(toto))

    # --- (v) Construct Final Basis and Transformation Matrix C_total ---
    # The final basis is {|i>, |b>, |c1>, |c2>, ..., |v1>, |v2>, ...}
    # where |i> and |b> are the impurity and special NO in the C1 basis.
    # The other vectors are the Lanczos chain vectors (excluding chain heads).

    # Get the vector representations of the chain states in the original basis
    # These are the columns of the C_to_chains matrix
    len_c = Q_c.shape[1]
    len_v = Q_v.shape[1]

    # The vector for the head of the conduction chain, |c0> = |A>, in the original basis
    c0_vec = C_to_chains[:, conduction_indices[0]] if len_c > 0 else np.zeros(M)
    # The vector for the head of the valence chain, |v0> = |B>, in the original basis
    v0_vec = C_to_chains[:, valence_indices[0]] if len_v > 0 else np.zeros(M)

    C_total = np.zeros((M, M))

    # The first two columns of C_total are the impurity |i> and special NO |b>
    # We recover them by rotating |c0> and |v0> back with U_bond.T
    # |i> = U_bond[0,0]|c0> + U_bond[0,1]|v0>
    # |b> = U_bond[1,0]|c0> + U_bond[1,1]|v0>
    C_total[:, 0] = U_bond[0, 0] * c0_vec + U_bond[0, 1] * v0_vec
    C_total[:, 1] = U_bond[1, 0] * c0_vec + U_bond[1, 1] * v0_vec

    # The remaining columns are the rest of the Lanczos chain vectors
    c_chain_start_idx = 2
    if len_c > 1:
        c_rest_indices_in_decoupled_basis = [conduction_indices[i] for i in range(1, len_c)]
        C_total[:, c_chain_start_idx : c_chain_start_idx + len_c - 1] = C_to_chains[
            :, c_rest_indices_in_decoupled_basis
        ]

    v_chain_start_idx = c_chain_start_idx + (len_c - 1 if len_c > 0 else 0)
    if len_v > 1:
        v_rest_indices_in_decoupled_basis = [valence_indices[i] for i in range(1, len_v)]
        C_total[:, v_chain_start_idx : v_chain_start_idx + len_v - 1] = C_to_chains[
            :, v_rest_indices_in_decoupled_basis
        ]

    # --- Transform the Hamiltonian and correct for the mean-field shift ---
    h_mf_final_basis = C_total.T @ h_mf @ C_total

    # mean_field_correction_term = np.zeros_like(h_spin)
    # mean_field_correction_term[0, 0] = u * 0.5
    # transformed_correction = C_total.T @ mean_field_correction_term @ C_total

    h_final_matrix = h_mf_final_basis  # - transformed_correction

    return h_final_matrix, C_total


# ------------------------------------------------------------
# Multi-orbital
# ------------------------------------------------------------


def get_double_chain_transform_multi(h, Nimp, Nelec, tol_occ=1e-8):
    """
    Multiorbital double-chain transform.

    Steps
      1) MF density from h
      2) Rotate bath to NOs, split into active, filled, empty
      3) Schmidt pairing between impurity and active bath (SVD)
      4) Per-pair 2x2 rotation to get conduction and valence heads
      5) Block Lanczos on conduction and valence halves
      6) Restore the original impurity exactly

    Returns
      h_final, C_total, meta
    """

    M = h.shape[0]  # total one-body dimension
    assert h.shape == (M, M) and 1 <= Nimp < M  # sanity checks

    # ---- small utilities ----
    def classify_bath(rho_bb):
        """
        Eigen-decompose bath density matrix and classify orbitals by occupation.
        """
        occ, W = eigh(rho_bb)  # bath density eigenbasis (NOs)
        # split bath NOs by occupation: fractional = "active", ~1 = filled, ~0 = empty
        filled = [i for i, n in enumerate(occ) if n > 1 - tol_occ]
        empty = [i for i, n in enumerate(occ) if n < tol_occ]
        active = [i for i, n in enumerate(occ) if tol_occ <= n <= 1 - tol_occ]
        return occ, W, active, filled, empty

    def pad_with_identity(Q, n):
        """
        Pad an orthonormal basis matrix Q to dimension n with identity elements.
        """
        m = Q.shape[1]  # Q is n×m with orthonormal columns
        if m == n:  # already square/unitary in subspace
            return Q
        out = np.eye(n, dtype=Q.dtype)  # extend to a square unitary by appending basis vectors
        out[:, :m] = Q
        return out

    # At the starting point, h can be in general dense
    #   IMP   BATH
    #   himp  Vib
    #   .     hb

    # ---- (1) MF density ----
    es, vs = eigh(h)  # single-particle eigenpairs of h (Hermitian)
    rho = vs[:, :Nelec] @ vs[:, :Nelec].conj().T  # ρ = Σ_{occ} |ψ⟩⟨ψ| (projector onto lowest Nelec)

    # ---- (2) Bath NOs and deterministic order ----
    occ_b, W, active_idx, filled_idx, empty_idx = classify_bath(rho[Nimp:, Nimp:])
    na, nf, ne = len(active_idx), len(filled_idx), len(empty_idx)
    assert na + nf + ne == M - Nimp  # partition covers the bath

    order_bath = active_idx + filled_idx + empty_idx  # fix an ordering: active first (to pair), then filled, empty
    C1 = block_diag(np.eye(Nimp, dtype=h.dtype), W[:, order_bath])  # rotate only the bath by W (impurity untouched)
    rho1 = C1.conj().T @ rho @ C1  # transform density to bath-NO basis
    h1 = C1.conj().T @ h @ C1  # transform Hamiltonian similarly
    print("DEBUG: C1 = ")
    print(C1.real)
    # Here, h1 has the form
    #  IMP  ACTIVE  FILLED EMPTY
    #  himp hia     hif     hie
    #       ha      haf     hae
    #               hf      0
    #                       he
    # himp is untouched, all other are changed

    if na == 0:  # no fractional bath states ⇒ nothing to pair or chain
        return h1, C1, dict(r=0, na=0, nf=nf, ne=ne, order_bath=order_bath)

    # ---- (3) Schmidt pairing on active block ----
    a0 = Nimp  # start index of bath in current basis
    C = rho1[a0 : a0 + na, :Nimp]  # impurity–active-bath cross-block of ρ (na × Nimp)

    # SVD gives matched impurity/bath Schmidt directions: C = Wa Σ Vimp†
    Wa, svals, Vimp_dag = svd(C, full_matrices=False)  # thin SVD (min(na, Nimp) columns)
    Vimp = Vimp_dag.conj().T  # right singular vectors (impurity rotation)

    r_svd = int(min(np.sum(svals > 1e-12), min(na, Nimp)))  # numerical rank capped by available pairs

    # rotate impurity by Vimp and active bath by Wa; filled/empty bath unchanged
    C2 = block_diag(Vimp, block_diag(Wa, np.eye(nf + ne, dtype=h.dtype)))
    print("DEBUG: C2 = ")
    print(C2.real)
    rho2 = C2.conj().T @ rho1 @ C2
    h2 = C2.conj().T @ h1 @ C2

    # The goal here is to make the coupling between the impurity and the active space diagonal in the
    # density matrix. So, after the svd, rho2 has the block rho2_ia diagonal
    # This defines natural pairs of orbitals to further diagonalize the IMP-ACTIVE rho subspace by blocks of pairs
    # Now h2 has the form
    #  Imp-Act-mixture  FILLED EMPTY
    #  hiam            hiam-f  hiam-e
    #                   hf      0
    #                           he
    # The blocks filled and empty are untouched

    # ---- (4) Per-pair 2×2 bonding/antibonding on the r_svd heads ----
    Ub = np.eye(M, dtype=h.dtype)  # accumulate pairwise 2×2 rotations
    Ub_pairs = []  # keep the 2×2s for the later exact restore
    for alpha in range(r_svd):
        i_idx, a_idx = alpha, Nimp + alpha  # indices of the impurity/bath Schmidt pair in current basis
        rho2_pair = rho2[np.ix_([i_idx, a_idx], [i_idx, a_idx])]  # 2×2 density restricted to that pair
        evals, U2 = eigh(rho2_pair)  # diagonalize ⇒ more/less occupied directions
        U2 = U2[:, np.argsort(evals)[::-1]]  # order so col 0 is the more filled ("valence-like")
        Ub[np.ix_([i_idx, a_idx], [i_idx, a_idx])] = U2  # embed 2×2 into the big unitary
        Ub_pairs.append(U2)

    C3 = Ub  # per-pair rotation unitary
    rho3 = C3.conj().T @ rho2 @ C3  # apply to ρ
    h3 = C3.conj().T @ h2 @ C3  # and to h
    print("DEBUG: C3 = ")
    print(C3.real)
    # Now, h3 has the form
    #  IA-FILLED IA-EMPTY  FILLED   EMPTY
    #  hiaf         0       hiaf-f  0
    #               hiae    0       hiae-e
    #                       hf      0
    #                               he
    # hf and he still untouched, the hamiltonian is now fully block diagonal
    # with a filled and an empty block
    # The two blocks can be made tridiagonal separately

    # ---- (5) Build halves and run block Lanczos ----
    # remaining bath indices are bath NOs: first 'filled' then 'empty' per order_bath
    filled_block = list(range(Nimp + na, Nimp + na + nf))  # filled-bath tail of the valence half
    empty_block = list(range(Nimp + na + nf, M))  # empty-bath tail of the conduction half

    valence_seeds = [alpha for alpha in range(r_svd)]  # the "more filled" heads (col 0 of each pair)
    conduction_seeds = [Nimp + alpha for alpha in range(r_svd)]  # the "less filled" heads

    # leftover impurity directions (if Nimp > r_svd): send by diagonal occupation
    for alpha in range(r_svd, Nimp):
        if float(np.real(rho3[alpha, alpha])) > 0.5:
            valence_seeds.append(alpha)  # > 0.5 ⇒ valence half
        else:
            conduction_seeds.append(alpha)  # ≤ 0.5 ⇒ conduction half

    val_indices = valence_seeds + filled_block  # full index list of valence half
    cond_indices = conduction_seeds + empty_block  # full index list of conduction half

    r_v, r_c = len(valence_seeds), len(conduction_seeds)  # initial block sizes for block Lanczos

    Hv = h3[np.ix_(val_indices, val_indices)]  # project h to valence half
    Hc = h3[np.ix_(cond_indices, cond_indices)]  # project h to conduction half

    # block Lanczos: returns block-tridiagonal factors (A,B) and the basis Q (orthonormal columns)
    _, _, Qv, _ = block_lanczos_matrix(Hv, r_v)  # Qv: valence chain basis (tall)
    _, _, Qc, _ = block_lanczos_matrix(Hc, r_c)  # Qc: conduction chain basis (tall)

    Qv_embed = pad_with_identity(Qv, len(val_indices))  # embed each Q as a square unitary on its subspace
    Qc_embed = pad_with_identity(Qc, len(cond_indices))

    C4 = np.eye(M, dtype=h.dtype)  # assemble subspace unitaries
    C4[np.ix_(val_indices, val_indices)] = Qv_embed
    C4[np.ix_(cond_indices, cond_indices)] = Qc_embed
    print("DEBUG: C4 = ")
    print(C4.real)
    # compose all steps before the final restore
    C_total = C1 @ C2 @ C3 @ C4
    print("DEBUG: C_total = ")
    print(C_total.real)
    # ---- (6) Exact restore of original impurity ----
    C_total = C_total @ C3.conj().T
    print("DEBUG: C_total_restored = ")
    print(C_total.real)

    h_final = C_total.conj().T @ h @ C_total  # final Hamiltonian in double-chain basis (impurity restored)

    meta = dict(  # bookkeeping for diagnostics/restarts
        r=r_svd,
        na=na,
        nf=nf,
        ne=ne,
        order_bath=order_bath,
        conduction_seeds=conduction_seeds,
        valence_seeds=valence_seeds,
        cond_indices=cond_indices,
        val_indices=val_indices,
    )
    return h_final, C_total, meta


# ------------------------------------------------------------
# Block - Sym
# ------------------------------------------------------------


def double_chain_by_blocks(
    h: np.ndarray,
    rho: np.ndarray,
    Nimp: int,
    Nelec: int,
    analyze_symmetries_fn,
    transform_fn,  # e.g. get_double_chain_transform_multi
    tol_occ: float = 1e-8,
    verbose: bool = False,
):
    """
    Apply the multiorbital double-chain transform blockwise, preserving and mirroring
    exact block symmetries. For identical blocks, reuse the leader's transform.

    Inputs
      h:        M x M Hermitian one-body Hamiltonian whose Nimp first index will be treated as the impurity
      rho:      M x M 1-rdm for h with Nelec
      Nimp:     number of impurity orbitals at global indices [0..Nimp-1]
      Nelec:    electron count in this spin sector (global)
      analyze_symmetries_fn: callable(h) -> {"blocks": [...], "identical_groups": [...], ...}
      transform_fn: callable(h_block, Nimp_block, Nelec_block, tol_occ=...) -> (h_final_block, C_block, meta)
      tol_occ:  occupancy threshold forwarded to transform_fn
      verbose:  optional prints

    Returns
      h_final:  M x M Hamiltonian after blockwise double-chain transforms
      C_total:  M x M global unitary; h_final = C_total^† h C_total
      meta:     dict with per-block metadata
    """
    M = h.shape[0]
    assert h.shape == (M, M)
    assert 1 <= Nimp <= M

    e0, _ = np.linalg.eigh(h)

    # 0) Analyze block structure once
    sym = analyze_symmetries_fn(h, verbose=verbose)
    blocks = sym["blocks"]  # list[list[int]]
    identical_groups = sym["identical_groups"]  # list[list[block_index]]
    if verbose:
        print(f"[blocks] {len(blocks)} blocks; identical groups: {identical_groups}")

    # 2) Prepare outputs
    C_total = np.eye(M, dtype=h.dtype)  # global unitary, block-diagonal fill
    h_final = np.zeros_like(h)  # we will place each transformed block
    block_results = []  # per-block bookkeeping
    leader_unitaries = {}  # map leader block idx -> (C_block, h_block_final, meta)

    # Helper: extract submatrix by index list
    def submat(A, idx):
        """
        Extract submatrix from A at the specified indices.
        """
        return A[np.ix_(idx, idx)]

    # Helper: place a submatrix B into A at idx positions (Hermitian block)
    def place_block(A, idx, B):
        """
        Insert submatrix B into matrix A at index locations.
        """
        A[np.ix_(idx, idx)] = B

    # Helper: count impurity orbitals in a block
    imp_set = set(range(Nimp))

    def count_block_impurity(idx_list):
        """
        Count how many impurity orbitals are in the given index list.
        """
        return sum(1 for j in idx_list if j in imp_set)

    # Helper: compute integer electron count in a block from rho trace
    def electrons_in_block(idx_list):
        """
        Compute electron count in a block from the trace of density matrix.
        """
        tr = np.real_if_close(np.trace(submat(rho, idx_list)))
        # round to nearest integer for stability
        return int(np.rint(float(tr)))

    # electron counts per block from the provided rho
    electrons_per_block = np.array([electrons_in_block(idx) for idx in blocks], dtype=float)
    if verbose:
        print(
            f"[check] Tr rho = {np.trace(rho).real:.12f}, sum blocks = {electrons_per_block.sum():.12f}, Nelec = {Nelec}"
        )

    # 3) Iterate identical groups; compute once per leader
    for group in identical_groups:
        leader = group[0]  # choose first block in the group as leader
        leader_idx = blocks[leader]
        # Per-block impurity and electrons
        Nimp_b = count_block_impurity(leader_idx)
        Nelec_b = electrons_in_block(leader_idx)

        if verbose:
            print(f"[leader block {leader}] size={len(leader_idx)} Nimp_b={Nimp_b} Nelec_b={Nelec_b}")

        if Nimp_b == 0:
            # No impurity content in this block. Do nothing: C_block = identity, h_block_final = h_block.
            h_block = submat(h, leader_idx)
            C_block = np.eye(len(leader_idx), dtype=h.dtype)
            h_block_final = h_block.copy()
            meta = dict(note="no_impurity_in_block", size=len(leader_idx))
        else:
            # Run the double-chain transform on the leader block
            h_block = submat(h, leader_idx)
            h_block_final, C_block, meta = transform_fn(h_block, Nimp_b, Nelec_b, tol_occ=tol_occ)
            print("h_block_final:")
            print(np.real(h_block_final))
            print("C_block:")
            print(np.real(C_block))

        # Cache leader result
        leader_unitaries[leader] = (C_block, h_block_final, meta)

        # Install into global matrices for the leader
        place_block(C_total, leader_idx, C_block)
        place_block(h_final, leader_idx, h_block_final)

        block_results.append(
            dict(block_id=leader, leader=True, indices=leader_idx, Nimp_b=Nimp_b, Nelec_b=Nelec_b, meta=meta)
        )

        # 4) Mirror to the followers in the identical group
        for follower in group[1:]:
            idx = blocks[follower]
            # Safety: blocks in identical group must have same size and same Nimp_b
            assert len(idx) == len(leader_idx), "Identical blocks must have equal size"
            Nimp_f = count_block_impurity(idx)
            assert Nimp_f == Nimp_b, "Identical blocks must contain equal impurity size"
            Nelec_f = electrons_in_block(idx)
            # We reuse the exact same unitary and h_block_final to preserve symmetry
            C_f, h_f, meta_f = C_block, h_block_final, dict(meta)  # shallow copy ok

            place_block(C_total, idx, C_f)
            place_block(h_final, idx, h_f)

            block_results.append(
                dict(block_id=follower, leader=False, indices=idx, Nimp_b=Nimp_f, Nelec_b=Nelec_f, meta=meta_f)
            )

    # 5) Sanity: h_final should equal C_total^† h C_total to numerical precision
    if verbose:
        check = C_total.conj().T @ h @ C_total
        err = np.linalg.norm(h_final - check) / max(1.0, np.linalg.norm(h))
        print(f"[sanity] ||h_final - C^† h C|| / ||h|| = {err:.3e}")

    meta_global = dict(blocks=blocks, identical_groups=identical_groups, per_block=block_results)

    ef, _ = np.linalg.eigh(h_final)

    return h_final, C_total, meta_global
