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
    h, v_tensor, const = extract_tensors(op, n_orb=u.shape[0])
    h_rot = rotate_one_body(h, u)
    v_rot = rotate_two_body(v_tensor, u)
    return tensors_to_operator(h_rot, v_rot, const, tol=tol)


def conserved_subset_charges(op, n_orb=None, tol=1e-9):
    r"""Find the orbital subsets whose total occupation is conserved by the **full** ``op``.

    A subset charge :math:`N_S = \sum_{i\in S} n_i` commutes with ``H`` iff every term
    is *block-balanced* in ``S`` — i.e. each term creates as many electrons in ``S`` as
    it annihilates. This returns the **finest** partition of the orbitals for which
    every block is conserved, found by union-find:

    - a one-body term ``c†_i c_j`` (``i≠j``) forces ``i`` and ``j`` into one block
      (otherwise it moves one electron across the boundary);
    - a two-body term is scanned for per-block imbalance; any blocks it imbalances are
      merged, iterated to a fixed point.

    Unlike the one-body commutant (Phase 2, which sees only ``h``), this accounts for
    the two-body interaction, so the returned charges are conserved by the interacting
    Hamiltonian. Each is a ``{0,1}``-weight (subset-occupation) charge directly mappable
    to a ``Basis`` restriction (Phase 3).

    Parameters
    ----------
    op : ManyBodyOperator or dict
        The Hamiltonian (1- and 2-body, number-conserving).
    n_orb : int, optional
        Number of spin-orbitals (inferred from ``op`` if ``None``).
    tol : float, optional
        Terms with ``|amp| <= tol`` are ignored.

    Returns
    -------
    list of frozenset of int
        The conserved orbital subsets (a partition of ``range(n_orb)``), sorted by
        smallest orbital.
    """
    from collections import Counter

    terms = op.to_dict() if hasattr(op, "to_dict") else dict(op)
    if n_orb is None:
        n_orb = 1 + max((idx for factors in terms for (idx, _) in factors), default=-1)

    parent = list(range(n_orb))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            return True
        return False

    parsed = []
    for factors, amp in terms.items():
        if abs(amp) <= tol:
            continue
        ladder = tuple(c for (_, c) in factors)
        idx = [i for (i, _) in factors]
        if ladder == ("c", "a") and idx[0] != idx[1]:
            union(idx[0], idx[1])
        elif ladder == ("c", "c", "a", "a"):
            parsed.append((idx[:2], idx[2:]))  # (creators, annihilators)

    changed = True
    while changed:
        changed = False
        for creators, annihilators in parsed:
            balance = Counter()
            for orb in creators:
                balance[find(orb)] += 1
            for orb in annihilators:
                balance[find(orb)] -= 1
            imbalanced = [block for block, net in balance.items() if net != 0]
            for block in imbalanced[1:]:
                if union(block, imbalanced[0]):
                    changed = True

    components = {}
    for orb in range(n_orb):
        components.setdefault(find(orb), []).append(orb)
    return sorted((frozenset(orbs) for orbs in components.values()), key=min)


def restrictions_from_charges(charges, occupations, slack=0):
    r"""Map conserved subset charges + target occupations to a ``Basis`` restriction dict.

    Parameters
    ----------
    charges : sequence of frozenset of int
        Conserved orbital subsets (e.g. from :func:`conserved_subset_charges`).
    occupations : sequence of int
        Target electron count in each subset (e.g. counted on the ground state from the
        Phase 3.0 pre-scan).
    slack : int, optional
        Allow ``occ ± slack`` electrons (default 0 = a strict sector). Use ``slack=1``
        to also admit the immediate neighbour sectors (``N ± 1`` etc.).

    Returns
    -------
    dict of frozenset of int to (int, int)
        ``Basis.restrictions``-format mapping ``subset -> (min, max)``. Subsets of size
        1 whose occupation is fully pinned still produce a valid bound.
    """
    restrictions = {}
    for subset, occ in zip(charges, occupations):
        lo = max(0, occ - slack)
        hi = min(len(subset), occ + slack)
        restrictions[frozenset(subset)] = (lo, hi)
    return restrictions


def subset_occupations(charges, occupied_orbitals):
    """Count electrons of a Slater determinant (set of occupied orbitals) in each subset."""
    occupied = set(occupied_orbitals)
    return [len(subset & occupied) for subset in charges]


def measure_conserved_charges(psi, charges, n_orb, comm=None, round_to_int=True):
    r"""Measure the conserved subset-charge occupations of a state, ``<psi|N_S|psi>``.

    Because each ``N_S = Σ_{i∈S} n_i`` is **diagonal** in the determinant basis, this is
    a weighted sum of determinant occupations — no operator application, so it is safe on
    a hash-distributed ``ManyBodyState`` (sum locally, then ``Allreduce``). For a genuine
    eigenstate each charge has a definite (integer) value; ``round_to_int`` rounds the
    weighted average to that value.

    Parameters
    ----------
    psi : ManyBodyState
        The state. Its local determinants are summed; pass ``comm`` to combine ranks.
    charges : sequence of frozenset of int
        Conserved orbital subsets (e.g. from :func:`conserved_subset_charges`).
    n_orb : int
        Number of spin-orbitals (for decoding determinant bit patterns).
    comm : MPI.Comm, optional
        If given, the local sums are reduced across ranks.
    round_to_int : bool, optional
        Round each charge to the nearest integer (default True).

    Returns
    -------
    list
        ``<N_S>`` for each charge (ints if ``round_to_int``, else floats).
    """
    import impurityModel.ed.product_state_representation as psr

    totals = np.zeros(len(charges))
    norm2 = 0.0
    for det, amp in psi.items():
        weight = abs(amp) ** 2
        norm2 += weight
        occupied = {k for k, bit in enumerate(psr.bytes2bitarray(bytes(det.to_bytearray()), n_orb)) if bit}
        for i, subset in enumerate(charges):
            totals[i] += weight * len(subset & occupied)
    if comm is not None:
        totals = comm.allreduce(totals)
        norm2 = comm.allreduce(norm2)
    if norm2 == 0:
        return [0 for _ in charges]
    averages = totals / norm2
    if round_to_int:
        return [int(round(x)) for x in averages]
    return list(averages)


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
    h, _, _ = extract_tensors(op, n_orb=n_orb)
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
