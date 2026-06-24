r"""Automated symmetry discovery for second-quantized Hamiltonians.

This module discovers the one-body symmetry algebra of a ``ManyBodyOperator``
Hamiltonian by solving the single-particle commutant condition :math:`[h, O] = 0`
and extracting its null space. The abelian (Cartan) part of the discovered algebra
labels conserved sectors (``N``, ``S_z``, ...); the full algebra (including the
non-abelian generators such as ``S_x``, ``S_y``) feeds the multiplet / Casimir
reconstruction in ``nonabelian_symmetry_casimir.md``.

**Scope and limitations.**

- Only **one-body** symmetry generators are found here. ``S^2`` and the other
  Casimirs are two-body and do not appear in the one-body null space; they are
  constructed separately (see ``finite.make_spin_operators`` / ``apply_casimir``).
- The method finds only **unitary** symmetries. **Anti-unitary symmetries (time
  reversal / Kramers degeneracy) are not detectable** by ``[H, O] = 0`` over the
  complex field. ``H`` is genuinely complex in this code (spin-orbit coupling), so
  this is a real gap, not a theoretical aside — report it wherever discovered
  symmetries are surfaced.

Implements symmetry-plan Phase 2 (``doc/plans/symmetry_implementation_plan.md``).
"""

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator


def extract_tensors(op, n_orb=None):
    r"""Extract the one- and two-body coefficient tensors of a ``ManyBodyOperator``.

    The operator is assumed purely 0-, 1- and 2-body and number-conserving, with
    terms in normal order. With the codebase convention
    (``finite.get_2_body_operator``) a stored term is

    - constant: ``()``  ->  ``const``
    - one-body: ``((i,'c'),(j,'a'))``  ->  :math:`h_{ij}` (coeff of
      :math:`c^\dagger_i c_j`)
    - two-body: ``((i,'c'),(j,'c'),(l,'a'),(k,'a'))``  ->  :math:`V_{ijkl}` (coeff of
      :math:`c^\dagger_i c^\dagger_j c_l c_k`)

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The operator (or its term dict).
    n_orb : int, optional
        Number of spin-orbitals. If ``None``, inferred as ``max index + 1``.

    Returns
    -------
    h : np.ndarray, shape (n_orb, n_orb)
        One-body coefficient matrix.
    V : np.ndarray, shape (n_orb, n_orb, n_orb, n_orb)
        Two-body coefficient tensor, :math:`V_{ijkl}` = coeff of
        :math:`c^\dagger_i c^\dagger_j c_l c_k`.
    const : complex
        Coefficient of the identity (energy shift).

    Raises
    ------
    ValueError
        If any term is not 0/1/2-body number-conserving (e.g. a 3-body term, or a
        non-number-conserving term such as a bare ``c``/``a``).
    """
    terms = op.to_dict() if hasattr(op, "to_dict") else dict(op)
    if n_orb is None:
        n_orb = 1 + max((idx for factors in terms for (idx, _) in factors), default=-1)

    h = np.zeros((n_orb, n_orb), dtype=complex)
    V = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)
    const = 0.0 + 0.0j

    for factors, amp in terms.items():
        ladder = tuple(c for (_, c) in factors)
        idx = [i for (i, _) in factors]
        if ladder == ():
            const += amp
        elif ladder == ("c", "a"):
            h[idx[0], idx[1]] += amp
        elif ladder == ("c", "c", "a", "a"):
            i, j, l, k = idx  # term is c†_i c†_j c_l c_k
            V[i, j, k, l] += amp
        else:
            raise ValueError(
                f"Operator term {factors} is not a 0/1/2-body number-conserving term "
                f"(ladder pattern {ladder}); symmetry discovery supports only 1- and "
                f"2-body operators."
            )
    return h, V, const


def tensors_to_operator(h, V=None, const=0.0, tol=0.0):
    r"""Inverse of :func:`extract_tensors`: build a ``ManyBodyOperator`` from tensors.

    Parameters
    ----------
    h : np.ndarray, shape (n_orb, n_orb)
        One-body coefficients :math:`h_{ij}` (coeff of :math:`c^\dagger_i c_j`).
    V : np.ndarray, shape (n_orb,)*4, optional
        Two-body coefficients :math:`V_{ijkl}` (coeff of
        :math:`c^\dagger_i c^\dagger_j c_l c_k`).
    const : complex, optional
        Identity coefficient.
    tol : float, optional
        Drop coefficients with magnitude ``<= tol`` (default 0: keep exact nonzeros).

    Returns
    -------
    ManyBodyOperator
    """
    d = {}
    n_orb = h.shape[0]
    for i in range(n_orb):
        for j in range(n_orb):
            if abs(h[i, j]) > tol:
                d[((i, "c"), (j, "a"))] = h[i, j]
    if V is not None:
        for i in range(n_orb):
            for j in range(n_orb):
                for k in range(n_orb):
                    for l in range(n_orb):
                        if abs(V[i, j, k, l]) > tol:
                            d[((i, "c"), (j, "c"), (l, "a"), (k, "a"))] = V[i, j, k, l]
    if abs(const) > tol:
        d[()] = const
    return ManyBodyOperator(d)


def _commutator_superoperator(h):
    r"""Matrix ``A`` of the linear map ``O -> [h, O]`` acting on ``vec(O)``.

    Uses **column-stacking** ``vec`` (``O.reshape(-1, order='F')``). With that
    convention ``vec(h O) = (I ⊗ h) vec(O)`` and ``vec(O h) = (hᵀ ⊗ I) vec(O)``, so

    .. math:: \mathrm{vec}([h, O]) = (I\otimes h - h^{T}\otimes I)\,\mathrm{vec}(O).

    Reshape any null vector back with the **same** ``order='F'``.
    """
    n = h.shape[0]
    eye = np.eye(n)
    return np.kron(eye, h) - np.kron(h.T, eye)


def discover_one_body_symmetries(h, sigma_cut=None):
    r"""Discover the one-body symmetry algebra ``{O : [h, O] = 0}`` of ``h``.

    Solves the single-particle commutant condition by taking the null space of the
    commutator super-operator (via SVD). Each returned generator is an ``n x n``
    matrix; together they form an orthonormal (Frobenius) basis of the commutant.

    For an SU(2)-symmetric ``h`` the algebra contains **all three** one-body spin
    generators ``S_x, S_y, S_z`` (and ``N``), not just ``S_z``. ``S^2`` is two-body
    and by construction cannot appear here.

    Parameters
    ----------
    h : np.ndarray, shape (n, n)
        The one-body Hamiltonian matrix (e.g. from :func:`extract_tensors`).
    sigma_cut : float, optional
        Singular values ``<= sigma_cut`` are treated as zero (in the null space).
        Default ``||h-superoperator||_2 * n * eps`` — scaled to the problem so it is
        robust to the floating-point null space being only approximate.

    Returns
    -------
    list of np.ndarray
        The generator matrices spanning the commutant.

    Notes
    -----
    Unitary symmetries only: ``[H, O] = 0`` over the complex field does not detect
    anti-unitary (time-reversal / Kramers) symmetries. See the module docstring.
    """
    h = np.asarray(h, dtype=complex)
    n = h.shape[0]
    a_matrix = _commutator_superoperator(h)
    _, s, vh = np.linalg.svd(a_matrix)
    norm_a = s[0] if s.size else 0.0
    if sigma_cut is None:
        sigma_cut = max(norm_a, 1.0) * n * np.finfo(float).eps
    null_mask = s <= sigma_cut
    null_vecs = vh.conj().T[:, null_mask]  # columns = vec(O), orthonormal
    return [null_vecs[:, a].reshape(n, n, order="F") for a in range(null_vecs.shape[1])]


def hermitian_algebra_basis(generators, tol=1e-9):
    r"""Return an orthonormal **Hermitian** basis of the algebra spanned by ``generators``.

    The commutant of a Hermitian ``h`` is closed under conjugate-transpose, so it is
    the complexification of a real algebra of Hermitian operators. For each generator
    ``O`` both ``(O+O†)/2`` and ``(O-O†)/(2i)`` are Hermitian and lie in the algebra;
    these are orthonormalised (real Gram-Schmidt under the Frobenius inner product
    ``Re Tr(A†B)``) so every returned matrix stays Hermitian.
    """
    candidates = []
    for gen in generators:
        gen = np.asarray(gen, dtype=complex)
        candidates.append(0.5 * (gen + gen.conj().T))
        candidates.append(0.5j * (gen.conj().T - gen))
    basis = []
    for mat in candidates:
        residual = mat.copy()
        for b in basis:
            residual = residual - np.real(np.vdot(b, residual)) * b
        norm = np.sqrt(np.real(np.vdot(residual, residual)))
        if norm > tol:
            basis.append(residual / norm)
    return basis


def cartan_subalgebra(generators, seed=0, tol=None):
    r"""Extract a maximal mutually-commuting (Cartan) subalgebra of the commutant.

    Uses the standard regular-element method: form a generic Hermitian element
    ``X = Σ rₖ Hₖ`` of the algebra (random real ``rₖ``); its centralizer within the
    algebra is, for generic ``X``, a maximal abelian (Cartan) subalgebra. The
    centralizer is found as the null space of ``O -> [X, O]`` restricted to the
    algebra. This is far more robust than diagonalising generators sequentially.

    Parameters
    ----------
    generators : sequence of np.ndarray
        A basis of the commutant (e.g. from :func:`discover_one_body_symmetries`).
    seed : int, optional
        Seed for the random regular element (reproducibility).
    tol : float, optional
        Null-space cutoff for the centralizer system. Default scales with the
        problem size and machine epsilon.

    Returns
    -------
    list of np.ndarray
        Mutually-commuting Hermitian matrices spanning the Cartan subalgebra.
    """
    herm = hermitian_algebra_basis(generators)
    m = len(herm)
    if m == 0:
        return []
    n = herm[0].shape[0]
    rng = np.random.default_rng(seed)
    x = sum(r * h for r, h in zip(rng.standard_normal(m), herm))
    # Solve sum_k c_k [X, H_k] = 0 for real c.
    cols = np.array([(x @ h - h @ x).reshape(-1) for h in herm]).T  # (n^2, m), complex
    real_sys = np.vstack([cols.real, cols.imag])  # (2 n^2, m), real -> enforces real c
    _, s, vt = np.linalg.svd(real_sys)
    if tol is None:
        tol = max(s[0] if s.size else 0.0, 1.0) * max(n, m) * np.finfo(float).eps
    # Right singular vectors (rows of vt) with singular value <= tol span the null space.
    null_coeffs = [vt[i] for i in range(m) if (i >= len(s) or s[i] <= tol)]
    cartan = [sum(c[k] * herm[k] for k in range(m)) for c in null_coeffs]
    return cartan


def joint_diagonalize(commuting_ops, seed=0):
    r"""Simultaneously diagonalise a set of mutually-commuting Hermitian operators.

    Forms a random real combination ``M = Σ rₖ Oₖ``; for generic ``rₖ`` its spectrum
    is non-degenerate, so its eigenvectors diagonalise **all** the ``Oₖ`` at once —
    robust even when an individual ``Oₖ`` has a degenerate spectrum (where the naive
    "diagonalise ``O₁`` then ``O₂`` in its eigenspaces" approach is ambiguous).

    Parameters
    ----------
    commuting_ops : sequence of np.ndarray
        Mutually-commuting Hermitian matrices.
    seed : int, optional
        Seed for the random combination.

    Returns
    -------
    U : np.ndarray
        Unitary whose columns are the common eigenvectors (the single-particle
        transformation).
    diagonals : list of np.ndarray
        For each operator, its real eigenvalues ordered along ``U`` (``diag(U† Oₖ U)``).
    """
    ops = [np.asarray(o, dtype=complex) for o in commuting_ops]
    rng = np.random.default_rng(seed)
    combo = sum(rng.standard_normal() * o for o in ops)
    combo = 0.5 * (combo + combo.conj().T)
    _, u = np.linalg.eigh(combo)
    diagonals = [np.real(np.diag(u.conj().T @ o @ u)) for o in ops]
    return u, diagonals


def weights_are_01(diag, tol=1e-6):
    r"""Whether a generator's single-particle eigenvalues (weights) are all in {0, 1}.

    A generator can be mapped to a subset-occupation restriction (symmetry-plan
    Phase 3) only if its orbital weights are ``{0, 1}`` (e.g. ``N``, a spin-up count).
    Generators with other weights (e.g. ``S_z`` with ``±1/2``) require the extended
    weighted-sum restriction machinery (Phase 6). ``diag`` is the eigenvalue list
    from :func:`joint_diagonalize`.
    """
    diag = np.asarray(diag)
    return bool(np.all(np.isclose(diag, 0.0, atol=tol) | np.isclose(diag, 1.0, atol=tol)))


def in_span(generators, matrix, tol=1e-9):
    r"""Whether ``matrix`` lies in the Frobenius span of the ``generators``.

    Parameters
    ----------
    generators : sequence of np.ndarray
        Generator matrices (e.g. from :func:`discover_one_body_symmetries`).
    matrix : np.ndarray
        The matrix to test for membership.
    tol : float, default 1e-9
        Relative residual tolerance.

    Returns
    -------
    bool
    """
    vec = np.asarray(matrix, dtype=complex).reshape(-1, order="F")
    if len(generators) == 0:
        return np.linalg.norm(vec) <= tol
    basis = np.array([np.asarray(g, dtype=complex).reshape(-1, order="F") for g in generators]).T
    q, _ = np.linalg.qr(basis)
    residual = vec - q @ (q.conj().T @ vec)
    return np.linalg.norm(residual) <= tol * max(np.linalg.norm(vec), 1.0)
