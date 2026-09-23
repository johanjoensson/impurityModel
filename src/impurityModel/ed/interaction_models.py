r"""Model Coulomb interactions, compiled to the one tensor form the solver consumes.

Every builder here returns a dense spin-orbital tensor in the RSPt convention that
:func:`impurityModel.ed.atomic_physics.getUop_from_rspt_u4` turns into an operator:

.. math:: \hat U = \frac{1}{2} \sum_{ijkl} u4[i,j,k,l]\, c^\dagger_i c^\dagger_j c_l c_k,
    \qquad u4[i,j,k,l] = \langle ij|V|kl \rangle,

on ``2n`` spin-orbitals laid out spin-major, spin down first: spin-orbital ``s*n + p`` is spatial
orbital ``p`` with spin ``s`` (0 = down). That is the layout :func:`impurityModel.ed.model.atomic_u4`
and ``c2i`` produce for a single shell, so a model interaction and a Slater-Condon one are
interchangeable wherever a ``u4`` is accepted.

The spatial forms (Hubbard-Kanamori, density-density, spatial term lists) are spin-rotation
invariant by construction: a spatial matrix element :math:`\langle pq|V|rs \rangle` becomes
``u4[p s, q s', r s, s s'] = <pq|V|rs>`` for every ``s, s'`` (spin is carried along each electron
line, ``p -> r`` and ``q -> s``). Entries whose operator vanishes identically (two creators or two
annihilators on the same spin-orbital) are zeroed, so equal operators give equal tensors.
"""

import numpy as np

from impurityModel.ed.atomic_physics import get_spherical_2_cubic_matrix, uj_from_u4
from impurityModel.ed.lie_algebra import rotate_two_body

#: Relative tolerance of the Hermiticity check and of the symmetry-completion conflict check.
U4_RELATIVE_TOLERANCE = 1e-10


def _drop_pauli_forbidden(u4):
    """Zero the entries whose operator is identically zero (``i == j`` or ``k == l``)."""
    n = u4.shape[0]
    diag = np.arange(n)
    u4[diag, diag, :, :] = 0
    u4[:, :, diag, diag] = 0
    return u4


def spatial_to_spin_u4(u_spatial):
    r"""Spin-rotation-invariant expansion of a spatial Coulomb tensor.

    Parameters
    ----------
    u_spatial : array_like, shape (n, n, n, n)
        Spatial matrix elements :math:`\langle pq|V|rs \rangle` (physicists' notation).

    Returns
    -------
    numpy.ndarray, shape (2n, 2n, 2n, 2n)
        ``u4[p s, q s', r s, s s'] = u_spatial[p, q, r, s]`` for all spins ``s, s'``, spin-major
        and spin down first, with identically vanishing entries zeroed.
    """
    u_spatial = np.asarray(u_spatial, dtype=complex)
    if u_spatial.ndim != 4 or len(set(u_spatial.shape)) != 1:
        raise ValueError(f"a spatial Coulomb tensor must have shape (n, n, n, n), got {u_spatial.shape}.")
    n = u_spatial.shape[0]
    u4 = np.zeros((2 * n,) * 4, dtype=complex)
    for s in range(2):
        for sp in range(2):
            u4[s * n : (s + 1) * n, sp * n : (sp + 1) * n, s * n : (s + 1) * n, sp * n : (sp + 1) * n] = u_spatial
    return _drop_pauli_forbidden(u4)


def kanamori_spatial(n, U, J=0.0, U_prime=None, J_pair=None):
    r"""Spatial Hubbard-Kanamori matrix elements.

    .. math::
        \langle aa|V|aa \rangle = U, \quad \langle ab|V|ab \rangle = U', \quad
        \langle ab|V|ba \rangle = J, \quad \langle aa|V|bb \rangle = J_p \qquad (a \neq b).

    After the spin expansion this is

    .. math::
        H = U \sum_a n_{a\uparrow} n_{a\downarrow}
          + U' \sum_{a \neq b} n_{a\uparrow} n_{b\downarrow}
          + (U' - J) \sum_{a < b, \sigma} n_{a\sigma} n_{b\sigma}
          - J \sum_{a \neq b} c^\dagger_{a\uparrow} c_{a\downarrow} c^\dagger_{b\downarrow} c_{b\uparrow}
          + J_p \sum_{a \neq b} c^\dagger_{a\uparrow} c^\dagger_{a\downarrow} c_{b\downarrow} c_{b\uparrow}.

    Parameters
    ----------
    n : int
        Number of spatial orbitals.
    U : float
        Intra-orbital repulsion.
    J : float, optional
        Hund's exchange (spin flip and the same-spin reduction).
    U_prime : float, optional
        Inter-orbital repulsion. Defaults to ``U - 2*J``, the value that makes the interaction
        rotationally invariant among real orbitals.
    J_pair : float, optional
        Pair hopping. Defaults to ``J``.

    Returns
    -------
    numpy.ndarray, shape (n, n, n, n)
    """
    if n < 1:
        raise ValueError(f"a Kanamori interaction needs at least one orbital, got n={n}.")
    U_prime = U - 2 * J if U_prime is None else U_prime
    J_pair = J if J_pair is None else J_pair
    u = np.zeros((n,) * 4, dtype=complex)
    for a in range(n):
        u[a, a, a, a] = U
        for b in range(n):
            if a == b:
                continue
            u[a, b, a, b] = U_prime
            u[a, b, b, a] = J
            u[a, a, b, b] = J_pair
    return u


def kanamori_u4(n, U, J=0.0, U_prime=None, J_pair=None):
    """Spin-orbital Hubbard-Kanamori tensor; see :func:`kanamori_spatial` for the parameters."""
    return spatial_to_spin_u4(kanamori_spatial(n, U, J=J, U_prime=U_prime, J_pair=J_pair))


def density_density_u4(U_opposite_spin, U_same_spin=None):
    r"""Density-density interaction.

    .. math::
        H = \sum_{a,b} U^{\uparrow\downarrow}_{ab}\, n_{a\uparrow} n_{b\downarrow}
          + \frac{1}{2} \sum_{\sigma} \sum_{a \neq b} U^{\sigma\sigma}_{ab}\, n_{a\sigma} n_{b\sigma}.

    Both matrices are real and symmetric, and each unordered same-spin pair is counted once:
    ``U_same_spin[a, b] = V`` adds ``V n_a n_b``, not ``2 V n_a n_b``.

    Parameters
    ----------
    U_opposite_spin : array_like, shape (n, n)
        Opposite-spin repulsion; the diagonal is the intra-orbital Hubbard U.
    U_same_spin : array_like, shape (n, n), optional
        Same-spin repulsion. The diagonal must be zero (the Pauli principle forbids the pair).
        Defaults to zero.

    Returns
    -------
    numpy.ndarray, shape (2n, 2n, 2n, 2n)
    """
    u_opp = np.asarray(U_opposite_spin, dtype=float)
    n = u_opp.shape[0]
    u_same = np.zeros((n, n)) if U_same_spin is None else np.asarray(U_same_spin, dtype=float)
    for name, mat in (("U_opposite_spin", u_opp), ("U_same_spin", u_same)):
        if mat.shape != (n, n):
            raise ValueError(f"{name} must be {n}x{n}, got shape {mat.shape}.")
        if not np.allclose(mat, mat.T, rtol=0, atol=U4_RELATIVE_TOLERANCE * max(1.0, np.max(np.abs(mat)))):
            raise ValueError(f"{name} must be symmetric: U[a, b] and U[b, a] describe the same pair.")
    if np.any(np.diag(u_same) != 0):
        raise ValueError(
            "U_same_spin must have a zero diagonal: two electrons of the same spin cannot share an orbital."
        )
    # n_i n_j = c+_i c+_j c_j c_i for i != j, i.e. u4[i, j, i, j] in the RSPt convention; the 1/2 of
    # the operator is compensated by the (i, j)/(j, i) pair both being stored.
    u4 = np.zeros((2 * n,) * 4, dtype=complex)
    for a in range(n):
        for b in range(n):
            dn_a, up_b = a, n + b
            u4[dn_a, up_b, dn_a, up_b] = u4[up_b, dn_a, up_b, dn_a] = u_opp[a, b]
            if a != b:
                for s in range(2):
                    u4[s * n + a, s * n + b, s * n + a, s * n + b] = u_same[a, b]
    return u4


def _parse_term(term, n_index):
    """``(p, q, r, s, re[, im])`` -> ``((p, q, r, s), complex)``, with index range checks."""
    if len(term) not in (5, 6):
        raise ValueError(f"an interaction term is [p, q, r, s, re] or [p, q, r, s, re, im], got {list(term)}.")
    idx = term[:4]
    for i in idx:
        if int(i) != i or not 0 <= int(i) < n_index:
            raise ValueError(f"interaction term {list(term)}: index {i} is not an integer in [0, {n_index}).")
    value = complex(term[4], term[5] if len(term) == 6 else 0.0)
    return tuple(int(i) for i in idx), value


def terms_to_tensor(n_index, terms, *, complete_symmetries=True):
    r"""Dense ``<pq|V|rs>`` tensor from a sparse list of matrix elements.

    With ``complete_symmetries`` each written element also sets its images under the two
    symmetries every Coulomb matrix element has,

    .. math:: \langle pq|V|rs \rangle = \langle qp|V|sr \rangle = \langle rs|V|pq \rangle^*,

    so a user writes each distinct element once. Completion *fills* the images and never adds to
    them, so writing an element and its image as well gives the same tensor as writing it once.
    Two writes of one element (directly or through images) that disagree are an error rather
    than being summed or averaged.

    Parameters
    ----------
    n_index : int
        Index range (spatial orbitals for a spatial list, spin-orbitals for a spin-orbital list).
    terms : iterable of sequence
        ``[p, q, r, s, re]`` or ``[p, q, r, s, re, im]``.
    complete_symmetries : bool, optional
        Fill in the symmetry images (default). ``False`` takes the list literally.

    Returns
    -------
    numpy.ndarray, shape (n_index,) * 4
    """
    values = {}
    scale = max([abs(_parse_term(t, n_index)[1]) for t in terms] + [1.0])
    tol = U4_RELATIVE_TOLERANCE * scale

    def assign(key, value, source):
        old = values.get(key)
        if old is not None and abs(old - value) > tol:
            raise ValueError(
                f"interaction term {source} sets <{key[0]} {key[1]}|V|{key[2]} {key[3]}> = {value}, but it "
                f"was already set to {old}" + (" (through a symmetry image)." if complete_symmetries else ".")
            )
        values[key] = value

    for term in terms:
        (p, q, r, s), value = _parse_term(term, n_index)
        source = list(term)
        assign((p, q, r, s), value, source)
        if complete_symmetries:
            assign((q, p, s, r), value, source)
            assign((r, s, p, q), np.conj(value), source)
            assign((s, r, q, p), np.conj(value), source)

    u = np.zeros((n_index,) * 4, dtype=complex)
    for key, value in values.items():
        u[key] = value
    return u


def terms_u4(n, spatial=(), spin_orbital=(), *, complete_symmetries=True):
    """Tensor from explicit term lists: the spin expansion of ``spatial`` plus ``spin_orbital``.

    Parameters
    ----------
    n : int
        Number of spatial orbitals (the tensor has ``2n`` spin-orbitals).
    spatial : iterable of sequence
        Spatial ``[p, q, r, s, re(, im)]`` elements, expanded over spin by :func:`spatial_to_spin_u4`.
    spin_orbital : iterable of sequence
        Spin-orbital ``[i, j, k, l, re(, im)]`` elements (index ``s*n + p``), taken literally apart
        from symmetry completion. Added to the expanded spatial part.
    complete_symmetries : bool, optional
        See :func:`terms_to_tensor`; applied to each list separately.

    Returns
    -------
    numpy.ndarray, shape (2n, 2n, 2n, 2n)
    """
    spatial, spin_orbital = list(spatial), list(spin_orbital)
    if not spatial and not spin_orbital:
        raise ValueError("an interaction term list needs at least one term (spatial or spin_orbital).")
    u4 = spatial_to_spin_u4(terms_to_tensor(n, spatial, complete_symmetries=complete_symmetries))
    if spin_orbital:
        u4 = u4 + _drop_pauli_forbidden(terms_to_tensor(2 * n, spin_orbital, complete_symmetries=complete_symmetries))
    return u4


def cubic_to_spherical_u4(u4_cubic, l):
    """Rotate a spin-orbital tensor written in the real (cubic-harmonic) basis of an ``l`` shell
    into the complex spherical-harmonic ``(l, s, m)`` basis.

    The cubic orbitals are the columns of :func:`get_spherical_2_cubic_matrix`, in its order
    (for a d shell ``e_g, e_g, t_2g, t_2g, t_2g``).
    """
    u = get_spherical_2_cubic_matrix(spinpol=False, l=l)
    # New (spherical) orbitals expressed in the old (cubic) basis are the columns of u^dagger.
    rot = np.kron(np.eye(2), u.conj().T)
    return rotate_two_body(np.asarray(u4_cubic, dtype=complex), rot)


def validate_u4(u4, n_imp):
    """Check a Coulomb tensor's shape and Hermiticity, returning it as a complex array.

    Hermiticity of the operator is ``u4[i,j,k,l] == conj(u4[k,l,i,j])``. A non-Hermitian tensor
    would make every Lanczos result meaningless without failing anywhere, so it is refused here.

    Raises
    ------
    ValueError
        On a wrong shape, or a Hermiticity violation above :data:`U4_RELATIVE_TOLERANCE` relative
        to the largest element; the message names the worst element.
    """
    u4 = np.asarray(u4, dtype=complex)
    if u4.shape != (n_imp,) * 4:
        raise ValueError(
            f"the Coulomb tensor must have shape {(n_imp,) * 4} for {n_imp} impurity spin-orbitals, got {u4.shape}."
        )
    scale = float(np.max(np.abs(u4), initial=0.0))
    if scale == 0.0:
        return u4
    defect = np.abs(u4 - u4.transpose(2, 3, 0, 1).conj())
    worst = np.unravel_index(np.argmax(defect), defect.shape)
    if defect[worst] > U4_RELATIVE_TOLERANCE * scale:
        i, j, k, l = (int(x) for x in worst)
        raise ValueError(
            f"the Coulomb tensor is not Hermitian: u4[{i},{j},{k},{l}] = {u4[worst]} but "
            f"conj(u4[{k},{l},{i},{j}]) = {np.conj(u4[k, l, i, j])}."
        )
    return u4


def mlft_uvv(u4):
    r"""The multiplet-average repulsion :math:`U_{vv}` of the MLFT double counting.

    The mean of :math:`\langle ij|V|ij \rangle - \langle ij|V|ji \rangle` over all pairs of distinct
    spin-orbitals, which in terms of :func:`uj_from_u4`'s Anisimov averages is

    .. math:: U_{vv} = \bar U - \frac{n - 1}{2n - 1} \bar J,

    ``n`` being the number of spatial orbitals. For a d shell this is the familiar
    :math:`F^0 - \frac{2}{63}(F^2 + F^4)`.
    """
    u4 = np.asarray(u4)
    n = u4.shape[0] // 2
    U, J = uj_from_u4(u4)
    return U - (n - 1) / (2 * n - 1) * J
