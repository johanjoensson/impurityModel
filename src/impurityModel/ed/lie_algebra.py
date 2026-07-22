r"""One-body symmetry algebra: discovery, rotation, and Casimir reconstruction.

This module holds the *algebraic half* of the symmetry machinery: the tensor
extraction/rotation primitives (:func:`extract_tensors`, :func:`rotate_hamiltonian`),
the one-body symmetry discovery (solving the single-particle commutant
:math:`[h, O] = 0` and extracting its null space), the abelian (Cartan) reduction and
joint diagonalization, and the reconstructed-Casimir observables.

**Scope and limitations.**

- Only **one-body** symmetry generators are found here. ``S^2`` and the other
  Casimirs are two-body and do not appear in the one-body null space; they are
  constructed separately (see ``observables.make_spin_operators`` / ``apply_casimir``).
- The method finds only **unitary** symmetries. **Anti-unitary symmetries (time
  reversal / Kramers degeneracy) are not detectable** by ``[H, O] = 0`` over the
  complex field. ``H`` is genuinely complex in this code (spin-orbit coupling), so
  this is a real gap, not a theoretical aside -- report it wherever discovered
  symmetries are surfaced.

The conserved-charge, restriction, and block-structure consumers of this algebra
live in :mod:`impurityModel.ed.symmetries`.

Implements symmetry-plan Phase 2 (``doc/plans/symmetry_implementation_plan.md``).
"""

from collections import namedtuple

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator, inner


def extract_tensors(op, n_orb=None, two_body=True):
    r"""Extract the one- and two-body coefficient tensors of a ``ManyBodyOperator``.

    The operator is assumed purely 0-, 1- and 2-body and number-conserving, with
    terms in normal order. With the codebase convention
    (``symmetries.get_2_body_operator``) a stored term is

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
    two_body : bool, default True
        If ``False``, skip building the two-body tensor and return ``V=None`` (two-body terms are
        still validated but not stored). The dense ``V`` is ``O(n_orb^4)`` memory -- e.g. 2.7 GB
        for a 114-orbital chain -- so allocating it just to read ``h`` OOMs under MPI when every
        rank builds it. The many callers that need only ``h`` should pass ``two_body=False``.

    Returns
    -------
    h : np.ndarray, shape (n_orb, n_orb)
        One-body coefficient matrix.
    V : np.ndarray, shape (n_orb, n_orb, n_orb, n_orb), or None
        Two-body coefficient tensor, :math:`V_{ijkl}` = coeff of
        :math:`c^\dagger_i c^\dagger_j c_l c_k`; ``None`` when ``two_body=False``.
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
    V = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex) if two_body else None
    const = 0.0 + 0.0j

    for factors, amp in terms.items():
        ladder = tuple(c for (_, c) in factors)
        idx = [i for (i, _) in factors]
        if ladder == ():
            const += amp
        elif ladder == ("c", "a"):
            h[idx[0], idx[1]] += amp
        elif ladder == ("c", "c", "a", "a"):
            if two_body:
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


def rotate_one_body(h, u):
    r"""Rotate the one-body tensor to a new single-particle basis: ``h' = U† h U``.

    ``U``'s columns are the new basis vectors expressed in the old basis, i.e. the
    new operators are ``c'_a = Σ_i U*_{ia} c_i``.

    Parameters
    ----------
    h : np.ndarray, shape (n, n)
    u : np.ndarray, shape (n, n)
        Unitary single-particle transformation.

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    u = np.asarray(u, dtype=complex)
    return np.einsum("mi,mn,nj->ij", u.conj(), np.asarray(h, dtype=complex), u, optimize=True)


def rotate_two_body(v_tensor, u):
    r"""Rotate the two-body tensor to a new single-particle basis.

    With the :func:`extract_tensors` convention (:math:`V_{ijkl}` = coeff of
    :math:`c^\dagger_i c^\dagger_j c_l c_k`) and ``c'_a = Σ_i U*_{ia} c_i``,

    .. math:: V'_{ijkl} = \sum_{mnpq} U^*_{mi} U^*_{nj} V_{mnpq} U_{pk} U_{ql}.

    Parameters
    ----------
    v_tensor : np.ndarray, shape (n, n, n, n)
    u : np.ndarray, shape (n, n)

    Returns
    -------
    np.ndarray, shape (n, n, n, n)
    """
    u = np.asarray(u, dtype=complex)
    return np.einsum(
        "mi,nj,mnpq,pk,ql->ijkl",
        u.conj(),
        u.conj(),
        np.asarray(v_tensor, dtype=complex),
        u,
        u,
        optimize=True,
    )


def rotate_hamiltonian(op, u, tol=1e-12):
    r"""Express a ``ManyBodyOperator`` in a rotated single-particle basis (``H' = U† H U``).

    Extracts the 1-/2-body tensors (:func:`extract_tensors`), rotates them
    (:func:`rotate_one_body`, :func:`rotate_two_body`) and rebuilds the operator.
    This is the one-time setup cost of symmetry-plan Phase 5 — a pure ``einsum``
    contraction, no Lanczos hot loop involved.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The 1-/2-body operator to rotate.
    u : np.ndarray, shape (n, n)
        Unitary single-particle transformation (e.g. from :func:`joint_diagonalize`).
    tol : float, optional
        Drop rotated coefficients with magnitude ``<= tol`` (default 1e-12) to prune
        numerical noise from the dense contraction.

    Returns
    -------
    ManyBodyOperator
    """
    u = np.asarray(u, dtype=complex)
    n = u.shape[0]
    terms = op.to_dict() if hasattr(op, "to_dict") else dict(op)
    # One-body + const, skipping the O(n^4) two-body tensor (rebuilt on its small support below).
    h, _, const = extract_tensors(terms, n_orb=n, two_body=False)
    h_rot = rotate_one_body(h, u)

    # The two-body terms live on a small "interacting" subspace S (the impurity Coulomb block).
    # Build V only on S (|S|^4, tiny) and rotate that sub-block -- exact when u does not mix S with
    # its complement (impurity_symmetry_rotation is identity off the impurity). This avoids the
    # O(n^4) full tensor + einsum that OOMs for long-chain baths (n^4 ~ 2.7 GB at n=114) once every
    # MPI rank builds it. Falls back to the full rotation if u mixes S with the rest.
    two_body_terms = {f: a for f, a in terms.items() if len(f) == 4}
    S = sorted({idx for f in two_body_terms for (idx, _) in f})
    v_s_rot = None
    if S:
        s_set = set(S)
        comp = [o for o in range(n) if o not in s_set]
        if comp and (np.max(np.abs(u[np.ix_(S, comp)])) > 1e-9 or np.max(np.abs(u[np.ix_(comp, S)])) > 1e-9):
            _, v_full, _ = extract_tensors(terms, n_orb=n)  # u mixes S with the rest: full path
            return tensors_to_operator(h_rot, rotate_two_body(v_full, u), const, tol=tol)
        pos = {o: k for k, o in enumerate(S)}
        m = len(S)
        V_s = np.zeros((m, m, m, m), dtype=complex)
        for f, a in two_body_terms.items():
            (i, _), (j, _), (l, _), (k, _) = f  # ((i,c),(j,c),(l,a),(k,a)) -> V[i,j,k,l]
            V_s[pos[i], pos[j], pos[k], pos[l]] += a
        v_s_rot = rotate_two_body(V_s, u[np.ix_(S, S)])

    d = tensors_to_operator(h_rot, None, const, tol=tol).to_dict()  # one-body + const (n^2 loop)
    if v_s_rot is not None:
        for a in range(len(S)):
            for b in range(len(S)):
                for c in range(len(S)):
                    for e in range(len(S)):
                        val = v_s_rot[a, b, c, e]
                        if abs(val) > tol:
                            d[((S[a], "c"), (S[b], "c"), (S[e], "a"), (S[c], "a"))] = val
    return ManyBodyOperator(d)


def discover_rotation(op, n_orb=None, seed=0):
    r"""Discover the symmetry and return ``(U, cartan)``: the symmetry-adapting rotation
    and the (un-rotated) Cartan generators that commute with ``h``.

    Unlike :func:`symmetry_adapted_transformation` (which returns the *rotated* operator
    and rotated generators), this returns the Cartan in the **original** basis, so the
    generators can be cheaply re-tested against a later Hamiltonian of the same symmetry
    (see :class:`SymmetryRotationCache`).
    """
    h, _, _ = extract_tensors(op, n_orb=n_orb, two_body=False)
    generators = discover_one_body_symmetries(h)
    cartan = cartan_subalgebra(generators, seed=seed)
    u, _ = joint_diagonalize(cartan, seed=seed)
    return u, cartan


class SymmetryRotationCache:
    r"""Cache the symmetry-adapting rotation ``U`` across e.g. DMFT iterations.

    In a self-consistency loop ``H`` changes every iteration (bath fitting), but the
    symmetry *structure* (point group + spin) is fixed by the problem — only coefficients
    move. Re-running the full discovery (SVD null space + Cartan + joint diagonalisation)
    each iteration is wasteful. This caches ``U`` and re-discovers **only** when the
    cached Cartan generators no longer commute with the new ``h`` (a cheap O(n³) check),
    i.e. when the symmetry structure actually changed.

    Attributes
    ----------
    discovery_count : int
        How many times the full discovery actually ran (the metric a cache-hit test
        checks).
    """

    def __init__(self, seed=0, tol=1e-9):
        self.seed = seed
        self.tol = tol
        self._u = None
        self._cartan = None
        self.discovery_count = 0

    def _symmetry_preserved(self, h):
        return self._cartan is not None and all(np.linalg.norm(h @ g - g @ h) <= self.tol for g in self._cartan)

    def get_rotation(self, op, n_orb=None):
        """Return ``U`` for ``op``, reusing the cached rotation if the symmetry is unchanged."""
        h, _, _ = extract_tensors(op, n_orb=n_orb, two_body=False)
        if self._symmetry_preserved(h):
            return self._u
        self._u, self._cartan = discover_rotation(op, n_orb=n_orb, seed=self.seed)
        self.discovery_count += 1
        return self._u


def symmetry_adapted_transformation(op, n_orb=None, seed=0):
    r"""Discover the symmetry and rotate the Hamiltonian into its symmetry-adapted basis.

    Ties the Phase-2 / Phase-5 pieces together: extract the one-body tensor, discover
    the commutant, pick a Cartan subalgebra, build the single-particle ``U`` that
    simultaneously diagonalises it, and return the rotated operator ``H' = U† H U``. In
    this basis every Cartan generator is **diagonal** — those with ``{0,1}`` weights are
    occupation numbers that map directly to subset-occupation restrictions (Phase 3),
    the rest (e.g. ``S_z``) need the weighted-sum machinery (Phase 6). This is the
    bridge that lets Phase 3 sectorize in a basis where the conserved charges are
    manifest.

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The Hamiltonian.
    n_orb : int, optional
        Number of spin-orbitals (inferred from ``op`` if ``None``).
    seed : int, optional
        Seed for the random regular element / joint-diagonalisation combination.

    Returns
    -------
    u : np.ndarray
        Single-particle transformation to the symmetry-adapted basis.
    rotated_op : ManyBodyOperator
        ``H'`` expressed in that basis.
    cartan : list of np.ndarray
        The Cartan generators, each **diagonal** in the new basis (``U† G U``). These
        are the raw material Phase 3 maps to restrictions (after rescaling to integer
        orbital weights and the ``{0,1}`` test); the mapping itself is left to Phase 3.
    """
    h, _, _ = extract_tensors(op, n_orb=n_orb, two_body=False)
    generators = discover_one_body_symmetries(h)
    cartan = cartan_subalgebra(generators, seed=seed)
    u, _ = joint_diagonalize(cartan, seed=seed)
    rotated_op = rotate_hamiltonian(op, u)
    cartan_rotated = [rotate_one_body(g, u) for g in cartan]
    return u, rotated_op, cartan_rotated


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

    For Hermitian ``h`` the commutant is computed from the eigendecomposition:
    ``[h, O] = 0`` exactly when ``O`` is block diagonal in ``h``'s eigenbasis, and the
    superoperator's singular values are ``|e_i - e_j|``, so the null space at
    ``sigma_cut`` is spanned by the matrix units ``u_i u_j^\dagger`` of eigenvectors
    whose eigenvalues agree within ``sigma_cut``. This is ``O(n^3)`` time and
    ``O(n^2)`` working memory; the dense ``n^2 x n^2`` superoperator SVD it replaces
    is ``O(n^6)`` / ``O(n^4)`` and OOM-killed real workloads already at ``n = 112``
    (12544 x 12544 complex ~ 2.4 GiB before LAPACK workspace). Non-Hermitian input
    falls back to the dense superoperator path.
    """
    h = np.asarray(h, dtype=complex)
    n = h.shape[0]
    norm_h = np.linalg.norm(h)
    if np.linalg.norm(h - h.conj().T) > max(norm_h, 1.0) * n * np.finfo(float).eps:
        a_matrix = _commutator_superoperator(h)
        _, s, vh = np.linalg.svd(a_matrix)
        norm_a = s[0] if s.size else 0.0
        if sigma_cut is None:
            sigma_cut = max(norm_a, 1.0) * n * np.finfo(float).eps
        null_mask = s <= sigma_cut
        null_vecs = vh.conj().T[:, null_mask]  # columns = vec(O), orthonormal
        return [null_vecs[:, a].reshape(n, n, order="F") for a in range(null_vecs.shape[1])]

    es, u = np.linalg.eigh(h)
    if sigma_cut is None:
        # Same scale as the superoperator path: ||A||_2 = max |e_i - e_j| = the spectral spread.
        spread = float(es[-1] - es[0]) if n else 0.0
        sigma_cut = max(spread, 1.0) * n * np.finfo(float).eps
    generators = []
    start = 0
    for k in range(1, n + 1):
        if k == n or es[k] - es[start] > sigma_cut:
            for a in range(start, k):
                for b in range(start, k):
                    generators.append(np.outer(u[:, a], u[:, b].conj()))
            start = k
    return generators


def is_abelian(generators, tol=1e-9):
    r"""Whether the one-body symmetry algebra is abelian (all commutators vanish).

    A non-abelian algebra (some ``[O_a, O_b] != 0``) is the signature of a non-abelian
    symmetry — e.g. SU(2) spin, whose three one-body generators ``S_x, S_y, S_z`` are all
    in the commutant but do not mutually commute (companion plan
    ``nonabelian_symmetry_casimir.md``, Phase A.1).
    """
    gens = [np.asarray(g, dtype=complex) for g in generators]
    for a in range(len(gens)):
        for b in range(a + 1, len(gens)):
            if np.linalg.norm(gens[a] @ gens[b] - gens[b] @ gens[a]) > tol:
                return False
    return True


def structure_constants(generators):
    r"""Lie-algebra structure constants ``f_{abc}`` with ``[O_a, O_b] = Σ_c f_{abc} O_c``.

    Assumes the ``generators`` are **mutually orthogonal** under the Frobenius inner
    product ``<A, B> = Tr(A† B)`` (true for the SVD null-space basis from
    :func:`discover_one_body_symmetries`, and for ``{S_x, S_y, S_z}``). For an su(2)
    triplet these reproduce ``[S_a, S_b] = i ε_{abc} S_c`` (up to the generator
    normalisation).

    Parameters
    ----------
    generators : sequence of np.ndarray
        Single-particle generator matrices (mutually Frobenius-orthogonal).

    Returns
    -------
    np.ndarray, shape (n, n, n)
        ``f[a, b, c]``.
    """
    gens = [np.asarray(g, dtype=complex) for g in generators]
    n = len(gens)
    norms = [np.vdot(g, g) for g in gens]
    f = np.zeros((n, n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            commutator = gens[a] @ gens[b] - gens[b] @ gens[a]
            for c in range(n):
                if abs(norms[c]) > 1e-15:
                    f[a, b, c] = np.vdot(gens[c], commutator) / norms[c]
    return f


def reconstructed_casimir_operator(generators):
    r"""Build a reconstructed Casimir ``Ĉ = Σ_a Ô_a²``.

    Given a (sub)set of one-body symmetry generators ``{O_a}`` (single-particle
    matrices, e.g. an su(2) spin triplet ``{S_x, S_y, S_z}``), each is promoted to a
    one-body ``ManyBodyOperator`` ``Ô_a = Σ_ij (O_a)_ij c†_i c_j`` and squared. For the
    spin triplet this is exactly ``Ŝ²`` (companion plan Phase A.2). Build once and reuse
    across states rather than calling :func:`apply_reconstructed_casimir` per state.

    Parameters
    ----------
    generators : sequence of np.ndarray
        Single-particle generator matrices spanning the sub-algebra.

    Returns
    -------
    ManyBodyOperator
        ``Ĉ``.
    """
    return sum(
        (tensors_to_operator(np.asarray(g, dtype=complex)) ** 2 for g in generators),
        ManyBodyOperator(),
    )


def apply_reconstructed_casimir(psi, generators):
    r"""Apply a reconstructed Casimir ``Ĉ = Σ_a Ô_a²`` to a state.

    Convenience wrapper over :func:`reconstructed_casimir_operator`; hoist that out of
    any loop over states instead of calling this repeatedly.
    """
    return reconstructed_casimir_operator(generators)(psi, 0)


def expect_reconstructed_casimir(psi, generators, comm=None):
    r"""Return ``<psi|Ĉ|psi>`` for the reconstructed Casimir ``Ĉ = Σ_a Ô_a²``."""
    val = inner(psi, apply_reconstructed_casimir(psi, generators))
    if comm is not None:
        val = comm.allreduce(val)
    return np.real(val)


def hermitian_algebra_basis(generators, tol=1e-9):
    r"""Return an orthonormal **Hermitian** basis of the algebra spanned by ``generators``.

    The commutant of a Hermitian ``h`` is closed under conjugate-transpose, so it is
    the complexification of a real algebra of Hermitian operators. For each generator
    ``O`` both ``(O+O†)/2`` and ``(O-O†)/(2i)`` are Hermitian and lie in the algebra;
    these are orthonormalised (real Gram-Schmidt under the Frobenius inner product
    ``Re Tr(A†B)``) so every returned matrix stays Hermitian.
    """
    candidates = []
    for gen_raw in generators:
        gen = np.asarray(gen_raw, dtype=complex)
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


# Result of :func:`component_symmetry_reduction`.
#
# ``Q`` : (m, m) unitary; column ``a`` is a symmetry-adapted component ``T'_a = sum_alpha
#         Q[alpha, a] T_alpha`` expressed in the original Cartesian component basis.
# ``representatives`` : column indices of ``Q`` that must actually be computed (one per
#         symmetry-equivalence group).
# ``group_of_column`` : length ``m``; ``group_of_column[a]`` is the index *into*
#         ``representatives`` of the representative for column ``a``.
# ``diagonalizable`` : if True, ``chi' = Q^dagger chi Q`` is diagonal with equal entries
#         within each group, so the full spectral tensor is rebuilt from the representative
#         diagonals via ``Q``. If False (no closing continuous symmetry, or a non-abelian
#         commutant), the caller falls back to the full ``m x m`` tensor (``Q`` is the
#         identity, every column is its own representative) -- always correct, just no dedup.
ComponentReduction = namedtuple("ComponentReduction", ["Q", "representatives", "group_of_column", "diagonalizable"])


def _matrix_commutant(mats, tol=None):
    r"""Orthonormal (Frobenius) basis of ``{X : [X, M] = 0 for all M in mats}``.

    Uses column-stacking ``vec``: ``vec(X M) = (M^T (x) I) vec(X)`` and
    ``vec(M X) = (I (x) M) vec(X)``, so ``vec([X, M]) = (M^T (x) I - I (x) M) vec(X)``.
    The joint null space over all ``M`` is the commutant. Returns the null-space matrices
    (generally not Hermitian; Hermitianise separately if needed).
    """
    mats = [np.asarray(m, dtype=complex) for m in mats]
    if not mats:
        return []
    n = mats[0].shape[0]
    eye = np.eye(n)
    rows = [np.kron(m.T, eye) - np.kron(eye, m) for m in mats]
    system = np.vstack(rows)
    _, s, vh = np.linalg.svd(system)
    norm = s[0] if s.size else 0.0
    if tol is None:
        tol = max(norm, 1.0) * n * np.finfo(float).eps
    null = vh.conj().T[:, [i for i in range(vh.shape[0]) if i >= len(s) or s[i] <= tol]]
    return [null[:, a].reshape(n, n, order="F") for a in range(null.shape[1])]


def component_symmetry_reduction(component_ops, h_onebody, n_orb=None, tol=1e-8):
    r"""Point-group dedup of a set of one-body *component* transition operators.

    A dipole (or other one-body) transition tensor
    :math:`\chi_{\alpha\beta}(\omega) = \langle g| T_\alpha^\dagger (\omega - H)^{-1}
    T_\beta |g\rangle` is invariant under any continuous single-particle symmetry ``G`` of
    ``h`` (``[h, G] = 0``): if ``G`` maps the component span onto itself,
    :math:`[G, T_\alpha] = \sum_\beta M_{\beta\alpha} T_\beta`, then :math:`\chi` commutes
    with the representation ``{M}`` on component space. When that representation is
    multiplicity-free (commutant abelian) a common eigenbasis ``Q`` of the commutant makes
    :math:`\chi' = Q^\dagger \chi Q` diagonal with **equal** entries within each
    symmetry-equivalence group -- so only one representative component per group needs a
    Lanczos run and the whole tensor is rebuilt from ``Q``.

    Only **continuous** (Lie-algebra) symmetries are detected (via
    :func:`discover_one_body_symmetries`): a fully rotational ``h`` collapses the three
    Cartesian dipoles to a single representative. Lower continuous symmetry (e.g. axial
    ``L_z``) still block-diagonalises the tensor -- ``chi`` comes back diagonal in ``Q`` --
    but does not by itself equate the two in-plane components (``chi_{+} = chi_{-}`` needs a
    reflection / time reversal, which ``[h, G] = 0`` cannot see), so no columns are dropped.
    A purely **discrete** point group (e.g. a cubic crystal field) or no symmetry falls back
    to the identity reduction and the full ``m x m`` tensor -- always correct, just no dedup.

    Parameters
    ----------
    component_ops : sequence of ManyBodyOperator or dict
        The one-body component operators (e.g. the three Cartesian dipole operators).
    h_onebody : np.ndarray, shape (n_orb, n_orb)
        One-body Hamiltonian matrix whose symmetries are used (from :func:`extract_tensors`).
    n_orb : int, optional
        Number of spin-orbitals; inferred from the operators/matrix if ``None``.
    tol : float, optional
        Closure / commutant residual tolerance.

    Returns
    -------
    ComponentReduction
    """
    if n_orb is None:
        n_orb = np.asarray(h_onebody).shape[0]
    m = len(component_ops)
    Ts = [extract_tensors(op, n_orb=n_orb, two_body=False)[0] for op in component_ops]
    identity = ComponentReduction(np.eye(m, dtype=complex), list(range(m)), list(range(m)), m <= 1)
    if m <= 1:
        return identity

    # Component matrices, vectorised, as the least-squares span for the closure test.
    tvec = np.array([T.reshape(-1) for T in Ts]).T  # (n^2, m)
    q_t, _ = np.linalg.qr(tvec)  # orthonormal basis of span(T); projector P = q_t q_t^dagger
    generators = discover_one_body_symmetries(np.asarray(h_onebody, dtype=complex))
    if not generators:
        return identity

    # The commutant basis from discovery is arbitrary: the generators that *close* on the
    # component span (``[G, T_alpha] in span(T)``) are linear combinations ``G = sum_k x_k C_k``.
    # Solve for that closing subalgebra as the null space of the "leaves-the-span" residual,
    # then represent each closing generator on the component span as an m x m matrix M.
    gens = [np.asarray(g, dtype=complex) for g in generators]
    columns = []
    for c in gens:
        res_c = []
        for T in Ts:
            r = (c @ T - T @ c).reshape(-1)
            r = r - q_t @ (q_t.conj().T @ r)  # component of [C, T_alpha] orthogonal to span(T)
            res_c.append(r)
        columns.append(np.concatenate(res_c))
    b_matrix = np.array(columns).T  # (m * n^2, n_gen)
    # Economy SVD: with m * n^2 rows, full_matrices=True materialises a (m n^2)^2 U
    # (~21 GiB at n = 112) that is never read. Full right-singular vectors are only
    # needed when there are more generators than rows (then the extra rows of vh span
    # part of the null space, picked up by the ``i >= len(s_b)`` branch below).
    _, s_b, vh_b = np.linalg.svd(b_matrix, full_matrices=b_matrix.shape[1] > b_matrix.shape[0])
    scale = s_b[0] if s_b.size else 0.0
    cut = max(scale, 1.0) * b_matrix.shape[0] * np.finfo(float).eps
    null_coeffs = [vh_b[i].conj() for i in range(vh_b.shape[0]) if i >= len(s_b) or s_b[i] <= cut]

    reps = []
    for x in null_coeffs:
        g = sum(x[k] * gens[k] for k in range(len(gens)))
        M = np.zeros((m, m), dtype=complex)
        for a, T in enumerate(Ts):
            coeff, *_ = np.linalg.lstsq(tvec, (g @ T - T @ g).reshape(-1), rcond=None)
            M[:, a] = coeff
        if np.linalg.norm(M) > tol:
            reps.append(M)

    if not reps:
        return identity  # no continuous symmetry acts non-trivially -> full tensor

    # chi commutes with the representation; work with the commutant of {M, M^dagger} so it is
    # closed under adjoint, then take its Hermitian basis.
    commutant = _matrix_commutant([r for M in reps for r in (M, M.conj().T)], tol=tol)
    herm = hermitian_algebra_basis(commutant, tol=tol)
    if not herm or not is_abelian(herm, tol=tol):
        return identity  # multiplicity > 1 (non-abelian commutant): keep the full tensor

    # Common eigenbasis of the abelian commutant; group columns by identical eigenvalue
    # signature (same "character" -> chi forced equal on them).
    Q, diagonals = joint_diagonalize(herm)
    signatures = np.round(np.array(diagonals).T / max(tol, 1e-12)).astype(np.int64)  # (m, len(herm))
    group_of_column = [0] * m
    representatives = []
    seen = {}
    for a in range(m):
        key = tuple(signatures[a])
        if key not in seen:
            seen[key] = len(representatives)
            representatives.append(a)
        group_of_column[a] = seen[key]
    return ComponentReduction(Q, representatives, group_of_column, True)
