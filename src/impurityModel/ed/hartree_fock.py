"""Cheap unrestricted Hartree-Fock seed for the ground-state occupation.

The CIPSI ground-state search needs a *nominal* impurity occupation ``N0`` to seed the
accurate solve. Determining it with a rough many-body CIPSI over a broadened occupation
window is expensive and, for long bath chains, can build a massive basis and exhaust
memory. This module instead solves the problem at **mean-field (unrestricted Hartree-Fock)
level**: a single Slater determinant that variationally minimises the mean-field energy.
Its impurity occupation *is* "the occupation corresponding to the lowest energy" at
mean-field level, obtained from a handful of small ``(n_orb x n_orb)`` dense
diagonalisations -- quick, and hard-bounded in memory regardless of chain length.

Conventions (matching :func:`impurityModel.ed.symmetries.extract_tensors`): a
number-conserving 0/1/2-body ``ManyBodyOperator`` stores

- one-body ``((i,'c'),(j,'a'))``            -> ``h[i,j]`` (coeff of :math:`c^\\dagger_i c_j`)
- two-body ``((i,'c'),(j,'c'),(l,'a'),(k,'a'))`` -> ``V[i,j,k,l]`` (coeff of
  :math:`c^\\dagger_i c^\\dagger_j c_l c_k`).

The two-body Coulomb term acts only on the impurity (the bath is non-interacting), so ``V``
is stored on the impurity subspace only, avoiding an ``O(n_orb^4)`` allocation. With the
one-particle density matrix :math:`\\rho_{ij} = \\langle c^\\dagger_i c_j\\rangle` the
mean-field energy and Fock matrix are

.. math::
    E[\\rho] = \\sum_{ij} h_{ij}\\rho_{ij}
            + \\sum_{ijkl} V_{ijkl}(\\rho_{ik}\\rho_{jl} - \\rho_{il}\\rho_{jk}),
    \\qquad
    F_{ab} = h_{ab} + 2\\sum_{jl}(V_{ajbl} - V_{ajlb})\\rho_{jl},

(the second matches :math:`E_2 = \\tfrac12\\,\\mathrm{tr}(F_2\\rho)`; in the diagonal limit
:math:`E_2 = \\sum_{ij}(V_{ijij}-V_{ijji})n_i n_j`, the familiar Hartree-minus-exchange).
"""

import numpy as np

# SCF iteration cap. Raised from 200 after the constrained solve on the 112-orbital NiO L-edge
# was measured to need 283 iterations at the default mixing -- converging monotonically, not
# oscillating, so the old cap merely truncated it and reported "NOT converged". At ~3 ms per
# iteration (a few small dense diagonalisations) the whole solve is under a second, which is
# nothing against the dN occupation scan it exists to replace.
_SCF_MAX_ITER = 500


def extract_hf_tensors(h_op, impurity_indices):
    """Parse ``h_op`` into a full one-body matrix and an impurity-restricted two-body tensor.

    Parameters
    ----------
    h_op : ManyBodyOperator or dict
        Number-conserving 0/1/2-body Hamiltonian.
    impurity_indices : iterable of int
        Spin-orbital indices making up the (interacting) impurity.

    Returns
    -------
    h : np.ndarray, shape (n_orb, n_orb)
        One-body coefficient matrix.
    V : np.ndarray, shape (n_imp, n_imp, n_imp, n_imp)
        Two-body tensor on the impurity subspace (local impurity indexing).
    imp : list of int
        Sorted impurity spin-orbital indices; ``imp[k]`` is local index ``k``.
    n_orb : int
        Number of spin-orbitals (inferred as ``max index + 1``).
    const : complex
        Identity (constant energy) coefficient.

    Raises
    ------
    ValueError
        If a term is not 0/1/2-body number-conserving, or a two-body term touches a
        non-impurity orbital (HF here assumes a non-interacting bath).
    """
    terms = h_op.to_dict() if hasattr(h_op, "to_dict") else dict(h_op)
    n_orb = 1 + max((idx for factors in terms for (idx, _) in factors), default=-1)
    imp = sorted(impurity_indices)
    pos = {o: k for k, o in enumerate(imp)}
    n_imp = len(imp)

    h = np.zeros((n_orb, n_orb), dtype=complex)
    V = np.zeros((n_imp, n_imp, n_imp, n_imp), dtype=complex)
    const = 0.0 + 0.0j

    for factors, amp in terms.items():
        ladder = tuple(c for (_, c) in factors)
        idx = [i for (i, _) in factors]
        if ladder == ():
            const += amp
        elif ladder == ("c", "a"):
            h[idx[0], idx[1]] += amp
        elif ladder == ("c", "c", "a", "a"):
            i, j, l, k = idx  # term is c†_i c†_j c_l c_k  ->  V[i,j,k,l]
            if not all(o in pos for o in (i, j, k, l)):
                raise ValueError(
                    f"Two-body term {factors} touches a non-impurity orbital; the HF seed "
                    "assumes the interaction lives on the impurity (non-interacting bath)."
                )
            V[pos[i], pos[j], pos[k], pos[l]] += amp
        else:
            raise ValueError(
                f"Operator term {factors} is not a 0/1/2-body number-conserving term "
                f"(ladder {ladder}); the HF seed supports only 0/1/2-body operators."
            )
    return h, V, imp, n_orb, const


def _fock_matrix(h, V, imp, rho):
    """Build the Fock matrix ``F = h + F2(rho)`` (see module docstring)."""
    rho_imp = rho[np.ix_(imp, imp)]
    # F2_ab = 2 sum_jl (V_ajbl - V_ajlb) rho_jl, on the impurity block only.
    f2 = 2.0 * (np.einsum("ajbl,jl->ab", V, rho_imp) - np.einsum("ajlb,jl->ab", V, rho_imp))
    f = h.copy()
    f[np.ix_(imp, imp)] += f2
    # Symmetrise against round-off; the physical impurity block and rho are Hermitian.
    return 0.5 * (f + f.conj().T)


def mean_field_energy(h, V, imp, rho):
    """Return the (real) mean-field energy ``E[rho]`` for the one-particle density ``rho``."""
    rho_imp = rho[np.ix_(imp, imp)]
    e1 = np.einsum("ij,ij->", h, rho)
    e2 = np.einsum("ijkl,ik,jl->", V, rho_imp, rho_imp) - np.einsum("ijkl,il,jk->", V, rho_imp, rho_imp)
    return float(np.real(e1 + e2))


def _aufbau_density(f, n_tot):
    """One-particle density ``rho_ij = <c†_i c_j>`` from filling ``f``'s ``n_tot`` lowest states.

    For occupied eigenvectors ``U_occ`` of ``f``, ``<c†_i c_j> = sum_a U*_{ia} U_{ja}`` =
    ``(conj(U_occ) @ U_occ.T)_{ij}`` -- the transpose of the projector ``U_occ U_occ†``. The
    two coincide on the diagonal (occupations) but differ off-diagonal for complex ``f``; the
    energy and Fock einsums below assume the ``<c†_i c_j>`` ordering.
    """
    _, u = np.linalg.eigh(f)
    u_occ = u[:, :n_tot]
    return u_occ.conj() @ u_occ.T


def _resolve_groups(n_orb, n_tot, constraints):
    """``constraints`` -> a full partition of the orbitals into ``(indices, n_electrons)`` groups.

    The orbitals no constraint names form the free group, holding the electrons the constraints
    do not account for. ``constraints=None`` yields the single all-orbital group, i.e. exactly
    the unconstrained aufbau.
    """
    if not constraints:
        return [(np.arange(n_orb), n_tot)]
    groups = []
    claimed: set[int] = set()
    for indices, n_e in constraints:
        idx = np.asarray(sorted(indices), dtype=int)
        if idx.size and (idx.min() < 0 or idx.max() >= n_orb):
            raise ValueError(f"constrained orbital index out of range [0, {n_orb})")
        overlap = claimed.intersection(idx.tolist())
        if overlap:
            raise ValueError(f"constrained groups overlap on orbitals {sorted(overlap)}")
        if not 0 <= n_e <= idx.size:
            raise ValueError(f"constrained group of {idx.size} orbitals cannot hold {n_e} electrons")
        claimed.update(idx.tolist())
        groups.append((idx, int(n_e)))
    free = np.array([i for i in range(n_orb) if i not in claimed], dtype=int)
    n_free = n_tot - sum(n_e for _, n_e in groups)
    if not 0 <= n_free <= free.size:
        raise ValueError(
            f"{n_tot} electrons minus {n_tot - n_free} constrained leaves {n_free} for "
            f"{free.size} unconstrained orbitals"
        )
    groups.append((free, int(n_free)))
    return groups


def _grouped_aufbau_density(f, groups):
    """Aufbau density with each group's electron count held fixed.

    Each group is filled from the eigenvectors of ``f``'s *diagonal block* for that group, and the
    blocks are assembled without cross-group coherence. The groups still see each other: ``f`` is
    the **full** Fock matrix, so every group feels every other group's Hartree/exchange field --
    that mean field is exactly what keeps a frozen core's valence partner honest.

    Forbidding cross-group coherence is not an approximation whenever the groups do not couple in
    the one-body ``h`` -- which is the case that matters here: a *frozen* shell is by definition one
    with no bath states, so it cannot hybridize, and then a block-diagonal ``rho`` is already an
    exact fixed point of the unconstrained SCF (the only other channel, the exchange term in
    :func:`_fock_matrix`, feeds on the cross-group block of ``rho``, which stays zero). The
    constraint therefore *selects* a fixed point rather than approximating one.

    ``f`` is Hermitian on entry (both callers hand over a symmetrised matrix), and a diagonal
    sub-block of a Hermitian matrix is Hermitian, so the blocks go to ``eigh`` untouched -- which
    also keeps the single-group case bit-for-bit identical to :func:`_aufbau_density`.
    """
    if len(groups) == 1:
        idx, n_e = groups[0]
        if idx.size == f.shape[0]:
            return _aufbau_density(f, n_e)
    rho = np.zeros(f.shape, dtype=complex)
    for idx, n_e in groups:
        if idx.size == 0:
            continue
        rho[np.ix_(idx, idx)] = _aufbau_density(f[np.ix_(idx, idx)], n_e)
    return rho


def hartree_fock_density_matrix(h, V, imp, n_tot, constraints=None, max_iter=_SCF_MAX_ITER, tol=1e-8, mixing=0.5):
    """Self-consistent (unrestricted) Hartree-Fock density matrix at fixed particle number.

    The single-determinant, spin-orbital formulation is intrinsically unrestricted: a full
    ``(n_orb, n_orb)`` density (no enforced up/down equality) lets the mean field develop a
    magnetic moment, following any symmetry-breaking field already in ``h``.

    Parameters
    ----------
    h, V, imp
        As returned by :func:`extract_hf_tensors`.
    n_tot : int
        Total electron number (conserved); the ``n_tot`` lowest orbitals are filled.
    constraints : sequence of (indices, n_electrons), optional
        Orbital subsets held at a **fixed electron count**, filled group-wise (see
        :func:`_grouped_aufbau_density`); the orbitals no constraint names take the remainder.
        ``None`` (default) is the plain global aufbau, unchanged.

        This is what keeps a *frozen* shell frozen. Unconstrained HF is free to empty one: on the
        NiO L-edge model, where the MLFT double counting puts the bare 3d level (-106 eV) *below*
        the 2p core (-74.5 eV), it ionizes the core into the 3d -- returning ``2p^4 3d^10``, and
        not even converging (it oscillates across that crossing). Pinning the core makes it
        converge to ``2p^6 3d^8.2``, the true ground-state sector.

        **Expect a higher energy than the unconstrained solve** (measured: -1175 vs -1196 eV).
        That is the point, not a defect: the unconstrained minimum really is lower, it just gets
        there through a state the calculation has declared unphysical.
    max_iter, tol, mixing
        SCF controls: linear density mixing ``rho <- mixing*rho_new + (1-mixing)*rho``.

    Returns
    -------
    rho : np.ndarray
        Converged one-particle density matrix.
    converged : bool
    energy : float
        Mean-field energy of the converged density.
    """
    n_orb = h.shape[0]
    n_tot = int(max(0, min(n_tot, n_orb)))
    groups = _resolve_groups(n_orb, n_tot, constraints)
    # Non-interacting start: fill the lowest orbitals of the one-body part (carries any field).
    # The start must honour the constraints too, or the SCF sets off from the wrong sector.
    rho = _grouped_aufbau_density(0.5 * (h + h.conj().T), groups)
    converged = False
    for _ in range(max_iter):
        f = _fock_matrix(h, V, imp, rho)
        rho_new = _grouped_aufbau_density(f, groups)
        if np.linalg.norm(rho_new - rho) < tol:
            rho = rho_new
            converged = True
            break
        rho = mixing * rho_new + (1.0 - mixing) * rho
    return rho, converged, mean_field_energy(h, V, imp, rho)


def hartree_fock_occupation(
    h_op, impurity_orbitals, bath_states, N0, frozen_occupations=None, max_iter=_SCF_MAX_ITER, tol=1e-8, mixing=0.5
):
    """Nominal impurity occupation per orbital set from an unrestricted Hartree-Fock solve.

    The total electron number is fixed at the nominal sector: ``sum_i N0[i]`` impurity
    electrons plus the (nominally full) valence bath orbitals; conduction baths are
    nominally empty. HF then redistributes charge between impurity and bath self-
    consistently at that fixed total, and the impurity occupation is read off and rounded.

    Parameters
    ----------
    h_op : ManyBodyOperator
        The full Hamiltonian.
    impurity_orbitals : dict
        ``{set_index: [block, ...]}`` of impurity spin-orbital indices per orbital set.
    bath_states : tuple(dict, dict)
        ``(valence_baths, conduction_baths)``, each ``{set_index: [block, ...]}``.
    N0 : dict
        Nominal impurity occupations per orbital set (sets the total particle number).
    frozen_occupations : set, optional
        Orbital sets whose occupation is **held at** ``N0[i]`` (a core shell, say). Without this,
        HF is free to empty them wherever that lowers the mean-field energy -- which on a
        core-level workload it does, ionizing the core (see
        :func:`hartree_fock_density_matrix`). Frozen sets come back at exactly ``N0[i]``.
    max_iter, tol, mixing
        SCF controls, forwarded to :func:`hartree_fock_density_matrix`.

    Returns
    -------
    winning_N0 : dict
        Rounded HF impurity occupation per orbital set (clamped to ``[0, set size]``).
    energy : float
        Converged mean-field energy.
    converged : bool
    """
    valence_baths, _conduction_baths = bath_states

    all_impurity_indices = sorted(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)
    h, V, imp, _n_orb, _const = extract_hf_tensors(h_op, all_impurity_indices)

    n_valence = sum(len(block) for blocks in valence_baths.values() for block in blocks)
    n_tot = int(sum(int(N0[i]) for i in N0) + n_valence)

    constraints = [
        ([orb for block in impurity_orbitals[i] for orb in block], int(N0[i])) for i in sorted(frozen_occupations or ())
    ]

    rho, converged, energy = hartree_fock_density_matrix(
        h, V, imp, n_tot, constraints=constraints, max_iter=max_iter, tol=tol, mixing=mixing
    )

    occ = np.real(np.diag(rho))
    winning_N0 = {}
    for i in N0:
        set_indices = [orb for block in impurity_orbitals[i] for orb in block]
        n_i = round(float(np.sum(occ[set_indices])))
        winning_N0[i] = int(max(0, min(len(set_indices), n_i)))
    return winning_N0, energy, converged


def classify_orbitals(rho, eps=0.05):
    """Classify every spin-orbital as filled / empty / partial from an HF density matrix.

    An orbital is **partial** (part of the covalent *active space*) when its occupation is not
    within ``eps`` of an integer; **filled** when ``> 1-eps``; **empty** when ``< eps``. A
    *smaller* ``eps`` errs toward a *larger* active space -- the safe direction, because a
    covalent orbital wrongly frozen cannot be recovered by the (symmetry-sector-confined) CIPSI
    expansion, whereas an over-large active space only costs a slightly bigger seed.

    Parameters
    ----------
    rho : np.ndarray
        HF one-particle density matrix; its real diagonal is the per-orbital occupation.
    eps : float, default 0.05
        Closeness-to-integer tolerance.

    Returns
    -------
    filled_idx, empty_idx, partial_idx : list of int
        The three orbital classes (indices into ``0..n_orb-1``).
    active_electrons : int
        Electrons to distribute over the partial orbitals, ``n_tot - #filled``.
    n_tot : int
        Total electron number, ``round(sum of occupations)``.
    """
    occ = np.real(np.diag(rho))
    n_orb = occ.size
    filled_idx = [i for i in range(n_orb) if occ[i] > 1.0 - eps]
    empty_idx = [i for i in range(n_orb) if occ[i] < eps]
    partial_idx = [i for i in range(n_orb) if eps <= occ[i] <= 1.0 - eps]
    n_tot = round(float(np.sum(occ)))
    active_electrons = n_tot - len(filled_idx)
    return filled_idx, empty_idx, partial_idx, active_electrons, n_tot


def build_cas_seed(filled_idx, partial_idx, active_electrons, num_spin_orbitals):
    """Active-space (CAS) seed determinants for the CIPSI ground-state solve.

    The **filled** orbitals are always occupied; the ``active_electrons`` are distributed over the
    **partial** (active) orbitals in every ``C(len(partial), active_electrons)`` way; **empty**
    orbitals are unoccupied. This spans every symmetry sector representable within the active space
    (charge transfer between partial impurity/bath orbitals, and the spin/point-group arrangements
    of the active electrons), which the sector-confined CIPSI expansion cannot rebuild on its own.

    Bits are MSB-first within each byte (orbital ``i`` -> bit ``7 - i%8``), matching
    ``ManyBodyOperator``'s determinant convention so the seed is operator-compatible.

    Parameters
    ----------
    filled_idx, partial_idx : sequence of int
        Frozen-occupied and active orbital indices (from :func:`classify_orbitals`).
    active_electrons : int
        Number of electrons to place among the partial orbitals (``0 <= it <= len(partial)``).
    num_spin_orbitals : int
        Total spin-orbital count (determines the determinant byte width).

    Returns
    -------
    list of bytes
        One determinant per active-electron arrangement; length ``C(len(partial), active_electrons)``.
    """
    import itertools

    if active_electrons < 0 or active_electrons > len(partial_idx):
        raise ValueError(f"active_electrons={active_electrons} out of range for {len(partial_idx)} partial orbitals")
    n_bytes = (num_spin_orbitals + 7) // 8
    filled = list(filled_idx)
    seeds = []
    for combo in itertools.combinations(sorted(partial_idx), active_electrons):
        data = bytearray(n_bytes)
        for orb in filled:
            data[orb // 8] |= 1 << (7 - orb % 8)
        for orb in combo:
            data[orb // 8] |= 1 << (7 - orb % 8)
        seeds.append(bytes(data))
    return seeds


def hf_active_space(
    h_op,
    impurity_orbitals,
    bath_states,
    N0,
    frozen_occupations=None,
    eps=0.05,
    max_iter=_SCF_MAX_ITER,
    tol=1e-8,
    mixing=0.5,
):
    """Run unrestricted HF and classify every spin-orbital into the active space.

    Combines :func:`hartree_fock_density_matrix` with :func:`classify_orbitals`. The total
    electron number is the nominal sector (``sum_i N0[i]`` impurity electrons + the nominally
    full valence baths); HF redistributes at fixed total and the resulting density -- over
    impurity **and** bath -- is classified. The partial orbitals are exactly the near-Fermi
    covalent orbitals (ligand + impurity) whose occupation must be spanned by the seed.

    ``frozen_occupations`` pins those orbital sets at ``N0[i]``, exactly as in
    :func:`hartree_fock_occupation` -- the two must agree on which shells are frozen, or the CAS
    classification would be built on a density in which the core has quietly ionized.

    Returns
    -------
    filled_idx, empty_idx, partial_idx : list of int
    active_electrons, n_tot : int
    rho : np.ndarray
        The converged HF density (diagonal = occupations).
    converged : bool
    energy : float
    """
    valence_baths, _conduction_baths = bath_states
    all_impurity_indices = sorted(orb for blocks in impurity_orbitals.values() for block in blocks for orb in block)
    h, V, imp, _n_orb, _const = extract_hf_tensors(h_op, all_impurity_indices)
    n_valence = sum(len(block) for blocks in valence_baths.values() for block in blocks)
    n_tot = int(sum(int(N0[i]) for i in N0) + n_valence)
    constraints = [
        ([orb for block in impurity_orbitals[i] for orb in block], int(N0[i])) for i in sorted(frozen_occupations or ())
    ]
    rho, converged, energy = hartree_fock_density_matrix(
        h, V, imp, n_tot, constraints=constraints, max_iter=max_iter, tol=tol, mixing=mixing
    )
    filled_idx, empty_idx, partial_idx, active_electrons, n_tot = classify_orbitals(rho, eps=eps)
    return filled_idx, empty_idx, partial_idx, active_electrons, n_tot, rho, converged, energy
