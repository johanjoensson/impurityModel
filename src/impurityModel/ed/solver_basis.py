from dataclasses import dataclass

import numpy as np

from impurityModel.ed.ManyBodyUtils import ManyBodyOperator
from impurityModel.ed.symmetries import (
    classify_bath_occupation,
    discover_one_body_symmetries,
    extract_tensors,
    group_orbitals_by_blocks,
    impurity_block_structure,
    impurity_symmetry_rotation,
    rotate_hamiltonian,
)

# Adaptive symmetry-adapted-basis rotation (calc_selfenergy): drop rotated operator terms below
# this magnitude (eV; removes rotation round-off fill), and rotate into the symmetry-adapted
# basis only if it keeps the operator term count within this factor of the input basis.
_ROTATION_TRIM_TOL = 1e-8
_MAX_ROTATION_FILL = 2.0


@dataclass(frozen=True)
class SolverBasis:
    """Solver-basis Hamiltonian and derived orbital/block layout for a self-energy run.

    Produced by :func:`_prepare_solver_basis`: the (optionally symmetry-adapted) solver
    Hamiltonian ``h`` and the matching non-interacting operator ``h0_solve``, the impurity/bath
    orbital grouping and block structure derived from it, the per-group occupation windows, and
    the rotations (``rotation_full`` full-space, ``u_imp`` impurity block) that carry results
    back to the caller's input basis.
    """

    h: object
    h0_solve: ManyBodyOperator
    n_spin_orbitals: int
    block_structure: object
    impurity_orbitals: dict
    bath_states: tuple
    nominal_occ: dict
    mixed_valence: dict
    rotation_full: "np.ndarray"
    u_imp: "np.ndarray"
    rot_to_spherical: "np.ndarray"
    total_impurity_orbitals: dict
    sum_bath_states: dict


def _per_group_occupation(nominal_occ, impurity_orbitals, h=None):
    """Map ``nominal_occ`` onto the derived orbital-symmetry groups.

    Accepts a dict already keyed by the group indices (used as-is), or any other dict / a
    scalar interpreted as the *total* impurity occupation. When the one-body Hamiltonian ``h``
    is supplied, the total is distributed by **energetic filling** — the lowest on-site-energy
    impurity spin-orbitals (``h[o, o]``) are occupied first — so e.g. a cubic d-shell fills the
    lower ``t2g`` manifold before ``eg`` (giving ``t2g=6``, ``eg=2`` for ``d8``) and the split
    is spin-symmetric. Without ``h`` it falls back to a size-proportional split (remainder to
    the largest groups). The prescan refines the per-group split, so this only needs to be a
    sensible starting point.
    """
    keys = list(impurity_orbitals)
    if isinstance(nominal_occ, dict) and set(nominal_occ) == set(keys):
        return {k: int(nominal_occ[k]) for k in keys}
    total = int(sum(nominal_occ.values()) if isinstance(nominal_occ, dict) else nominal_occ)

    if h is not None:
        # Energetic filling: occupy the lowest on-site-energy impurity spin-orbitals first and
        # count how many land in each group. Ties broken by orbital index for determinism.
        orb_to_group = {orb: k for k in keys for block in impurity_orbitals[k] for orb in block}
        ordered = sorted(orb_to_group, key=lambda o: (np.real(h[o, o]), o))
        alloc = dict.fromkeys(keys, 0)
        for orb in ordered[: max(0, min(total, len(ordered)))]:
            alloc[orb_to_group[orb]] += 1
        return alloc

    sizes = {k: sum(len(block) for block in impurity_orbitals[k]) for k in keys}
    tot_size = sum(sizes.values()) or 1
    alloc = {k: int(total * sizes[k] // tot_size) for k in keys}
    remainder = total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: sizes[k], reverse=True):
        if remainder <= 0:
            break
        alloc[k] += 1
        remainder -= 1
    return alloc


def _per_group_scalar(value, impurity_orbitals, default=0):
    """Map a per-group scalar setting (e.g. ``mixed_valence``) onto the derived group keys.

    Unlike ``nominal_occ``, which is a *charge* and has to be redistributed (see
    :func:`_per_group_occupation`), this is a per-group *window width*: how far a group's
    occupation may deviate. The right image of one width under a group split is therefore the
    same width on each derived group -- ``basis_generation`` takes ``total_slack`` as the
    ``max`` over groups, so broadcasting reproduces the unsplit window exactly.

    A mismatched-key dict used to fall through to ``default`` and silently discard the setting.
    That is precisely the caller's case: the RSPt interface passes ``{0: mv}`` keyed on the
    *input* group, while the keys here are the groups derived from the block structure, so any
    impurity whose block structure splits (a cubic d-shell into ``eg``/``t2g`` -- NiO) lost its
    mixed valence without a word.
    """
    keys = list(impurity_orbitals)
    if isinstance(value, dict):
        if set(value) == set(keys):
            return dict(value)
        # Keys from a different grouping: keep the setting, widen every derived group by the
        # largest window asked for. Dropping it silently is never the safer reading -- it
        # collapses the charge-transfer window the caller explicitly requested.
        widest = max((abs(int(v)) for v in value.values()), default=default)
        return dict.fromkeys(keys, widest)
    if value is None:
        return dict.fromkeys(keys, default)
    # A bare scalar (including the bool that BasisOptions.mixed_valence often carries).
    return dict.fromkeys(keys, abs(int(value)))


def prepare_solver_basis(h0, dc, u, impurity_orbitals, nominal_occ, mixed_valence, rot_to_spherical, verbosity):
    """Build the solver-basis Hamiltonian and derive its orbital/block layout.

    Assembles the interacting Hamiltonian ``H = h0 - DC + U(u4)`` in the caller's input basis, then
    adaptively rotates into the impurity-diagonalising basis when that does not densify the
    Coulomb tensor (fill ratio ``<= _MAX_ROTATION_FILL``; the input basis is kept otherwise).
    Derives the bath valence/conduction split, the Green's-function block structure, and the
    per-group impurity/bath orbital grouping and occupation windows -- all in whichever basis is
    solved in. Returns a :class:`SolverBasis`.
    """
    # construct local, interacting, hamiltonian (in the caller's input/correlated basis B)
    h_input = ManyBodyOperator(h0) - ManyBodyOperator(dc) + ManyBodyOperator(u)
    # The *non-interacting part of the Hamiltonian actually diagonalised* -- h0 with the double
    # counting already removed, not the raw KS block. This is what the self-energy is extracted
    # against (`get_sigma`'s `h0op`, via `SolverBasis.h0_solve`), and the two must agree or the
    # returned Sigma carries a spurious `-dc`: RSPt hands us the raw block plus `sig_dc` and
    # expects the *pure interaction* self-energy back, applying the double counting to the
    # lattice itself exactly once in `double_counting_apply` (green_double_counting.F90:596).
    # The external CTHYB path shows the same contract from the other side -- RSPt subtracts
    # `sig_dc` from `hlda` before that solver is called (green_solver.F90:578-582) and still
    # subtracts it from `sig` afterwards.
    h0_input = ManyBodyOperator(h0) - ManyBodyOperator(dc)

    # Flatten the impurity orbital dict (dict[int, list[int]]) into a plain spin-orbital index
    # list; the total orbital count is inferred from the Hamiltonian (impurity + bath). The bath
    # orbitals and their valence/conduction split are derived below, not passed in.
    impurity_indices = sorted(o for orbs in impurity_orbitals.values() for o in orbs)
    h_input_matrix = extract_tensors(h_input, two_body=False)[0]
    n_spin_orbitals = h_input_matrix.shape[0]

    # Adaptive symmetry-adapted basis: diagonalising the impurity one-body block collapses the
    # Green's-function block structure to its finest form (e.g. 1x1 eg/t2g blocks) BUT can
    # express the Coulomb interaction more densely. h0 and u4 are in the caller's "correlated"
    # input basis (NOT assumed spherical); the fill test below is measured *relative to that
    # input basis*, so we rotate only when it does not densify the operator (fill <= threshold)
    # and keep the input basis otherwise (e.g. a j,m_j eigenbasis under spin-orbit coupling
    # densifies the Coulomb tensor). Every output is rotated back to the input basis B before
    # returning; nothing here presumes a spherical-harmonic input.
    rotation_full, u_imp = impurity_symmetry_rotation(
        h_input, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_input_matrix
    )
    h_rotated = rotate_hamiltonian(h_input, rotation_full, tol=_ROTATION_TRIM_TOL)
    n_terms_input = sum(1 for v in h_input.values() if abs(v) > _ROTATION_TRIM_TOL)
    fill_ratio = len(h_rotated) / max(n_terms_input, 1)

    rotate = fill_ratio <= _MAX_ROTATION_FILL
    if rotate:
        h = h_rotated
        h0_solve = rotate_hamiltonian(h0_input, rotation_full, tol=_ROTATION_TRIM_TOL)
        # Observable rotation for the solve (spherical -> S): compose the caller's input rotation
        # R_in (spherical -> B) with W^dag (B -> S). On the impurity block, R = u_imp^dag @ R_in.
        rot_to_spherical = u_imp.conj().T @ np.asarray(rot_to_spherical, dtype=complex)
    else:
        # Stay in the input basis; make the output rotation below a no-op.
        h = h_input
        h0_solve = h0_input
        rotation_full = np.eye(n_spin_orbitals, dtype=complex)
        u_imp = np.eye(len(impurity_indices), dtype=complex)

    # One-body matrix of the solver-basis Hamiltonian, extracted once and shared by the
    # classification/grouping helpers below (each would otherwise re-walk the full operator
    # and allocate its own dense n_orb x n_orb copy).
    h_matrix = extract_tensors(h, n_orb=n_spin_orbitals, two_body=False)[0] if rotate else h_input_matrix

    # Derive the bath orbitals (complement of the impurity set) and their initial occupation:
    # baths below the Fermi level (h[o, o] < 0) are valence (initially occupied), the rest are
    # conduction (initially empty). The bath one-body diagonal is unchanged by the impurity-only
    # rotation, so this is consistent whether measured in the input or solver basis.
    valence_flat, conduction_flat = classify_bath_occupation(
        h, impurity_indices, n_orb=n_spin_orbitals, h0_matrix=h_matrix
    )

    # GF block structure from the hybridization-dressed impurity matrix (h[imp,imp] + V^dag V),
    # in whichever basis we solve in (fixes bath-mediated coupling; 1x1 blocks when rotated).
    # Derived from h *after* any rotation, so the blocks label the sectors of the solver basis.
    block_structure = impurity_block_structure(h, impurity_indices, h0_matrix=h_matrix)

    # Group the flat orbital lists into orbital-symmetry manifolds (the inequivalent blocks and
    # their spin-degenerate partners, e.g. eg / t2g) **in the solver basis** h. Grouping by the
    # block structure keeps both spins of a manifold in one group, so the many-body basis spans
    # all S_z sectors (spin multiplets stay degenerate); the impurity occupation window is tied
    # across groups by the restriction machinery, not pinned per group.
    impurity_orbitals, bath_states = group_orbitals_by_blocks(
        h, impurity_indices, valence_flat, conduction_flat, block_structure, n_orb=n_spin_orbitals, h0_matrix=h_matrix
    )
    nominal_occ = _per_group_occupation(nominal_occ, impurity_orbitals, h_matrix)
    mixed_valence = _per_group_scalar(mixed_valence, impurity_orbitals, default=0)

    valence_baths, conduction_baths = bath_states
    total_impurity_orbitals = {i: sum(len(orbs) for orbs in impurity_orbitals[i]) for i in impurity_orbitals}
    sum_bath_states = {
        i: sum(len(orbs) for orbs in valence_baths[i]) + sum(len(orbs) for orbs in conduction_baths[i])
        for i in valence_baths
    }

    if verbosity > 0:
        basis_note = f"symmetry-adapted (fill {fill_ratio:.1f}x)" if rotate else f"input basis (fill {fill_ratio:.1f}x)"
        print(f"Block structure: {len(block_structure.blocks)} blocks, solving in {basis_note}")
    return SolverBasis(
        h=h,
        h0_solve=h0_solve,
        n_spin_orbitals=n_spin_orbitals,
        block_structure=block_structure,
        impurity_orbitals=impurity_orbitals,
        bath_states=bath_states,
        nominal_occ=nominal_occ,
        mixed_valence=mixed_valence,
        rotation_full=rotation_full,
        u_imp=u_imp,
        rot_to_spherical=rot_to_spherical,
        total_impurity_orbitals=total_impurity_orbitals,
        sum_bath_states=sum_bath_states,
    )


def _one_body_commutant_generators(h_op, impurity_orbitals, bath_states):
    """The continuous commutant of the *one-body* block, propagated along the bath chain.

    Correct exactly when ``H`` has no two-body part: then the commutant of ``h`` really is a
    symmetry algebra. With a Coulomb interaction it is far too large -- a degenerate manifold
    yields the whole of ``u(n)``, arbitrary orbital and spin rotations inside it, and a
    Slater-Condon ``U`` breaks essentially all of them (measured on a cubic d-shell: 36 of 36
    fail ``[H, g] = 0`` with residuals 0.86 to 2.57). Kept because it is the only source of
    *orbital* generators, and gated by the consumer against the full ``H``, which is what makes
    the over-production harmless.

    .. note:: Two known limitations, both of which make this return fewer generators than it
       should rather than wrong ones (the consumer's gate covers the rest):

       * the bath site map is built per bath index ``k`` while looping over the valence and
         conduction dicts in turn, so on a model with **both** the conduction entries overwrite
         the valence ones and every generator then fails the commutator test -- i.e. ``[]``;
       * ``imp_map`` keys on ``(group, index-within-block)``, so a group holding several blocks
         (``eg`` and ``t2g``: precisely a cubic d-shell) collides -- 10 orbitals collapse onto 6
         keys -- and the propagation pairs impurity orbitals with the wrong bath partners.
    """
    imp_orbs = []
    if impurity_orbitals:
        for orbs in impurity_orbitals.values():
            for o in orbs:
                imp_orbs.extend(o)

    h, _, _ = extract_tensors(h_op, two_body=False)

    if imp_orbs:
        imp = sorted(set(imp_orbs))
        h_imp = h[np.ix_(imp, imp)]
        imp_generators = discover_one_body_symmetries(h_imp)

        imp_map = {}
        for group, imp_blocks in impurity_orbitals.items():
            for imp_blk in imp_blocks:
                for idx_in_grp, o in enumerate(imp_blk):
                    imp_map[o] = (group, idx_in_grp)

        generators = []
        for g_imp in imp_generators:
            g_full = np.zeros_like(h, dtype=complex)
            for i, oi in enumerate(imp):
                for j, oj in enumerate(imp):
                    g_full[oi, oj] = g_imp[i, j]

            n_bath = 0
            for bath_dict in bath_states:
                if bath_dict:
                    n_bath = max(n_bath, max((len(blks) for blks in bath_dict.values()), default=0))

            site_maps = [{i: imp[i] for i in range(len(imp))}]

            for k in range(n_bath):
                site_k_map = {}
                for bath_dict in bath_states:
                    if not bath_dict:
                        continue
                    for i, o_imp in enumerate(imp):
                        if o_imp not in imp_map:
                            continue
                        group, idx_in_grp = imp_map[o_imp]
                        if group in bath_dict and k < len(bath_dict[group]) and idx_in_grp < len(bath_dict[group][k]):
                            site_k_map[i] = bath_dict[group][k][idx_in_grp]
                site_maps.append(site_k_map)

                for i in range(len(imp)):
                    if i not in site_k_map:
                        continue
                    oi = site_k_map[i]
                    for j in range(len(imp)):
                        if j not in site_k_map:
                            continue
                        oj = site_k_map[j]

                        propagated = False
                        for prev_map in reversed(site_maps[:-1]):
                            if i not in prev_map or j not in prev_map:
                                continue
                            prev_oi = prev_map[i]
                            prev_oj = prev_map[j]

                            t_j = h[oj, prev_oj]
                            t_i = h[oi, prev_oi]

                            # Chain hoppings decay geometrically along the fit, so an absolute gap
                            # test (e.g. < 1e-6) treats t_i=1e-9, t_j=1e-10 -- a 10x mismatch -- as
                            # equal; scale by the larger hopping.
                            if (
                                abs(t_j) > 1e-12
                                and abs(t_i) > 1e-12
                                and abs(abs(t_i) - abs(t_j)) < 1e-6 * max(abs(t_i), abs(t_j))
                            ):
                                g_full[oi, oj] = g_full[prev_oi, prev_oj] * (t_j / t_i)
                                propagated = True
                                break
                        if not propagated:
                            g_full[oi, oj] = 0.0

            # Absolute commutator tolerance is scale-dependent: a core level at -6000 eV vs
            # bath hoppings fit to ~1e-6 eV precision means the same 1e-9 cut either accepts
            # noise (large ||h||) or rejects true symmetries (small ||[h,g]|| relative to the
            # fit residual). Normalize by ||h||*||g|| so the cut is scale-free.
            norm_h = np.linalg.norm(h)
            norm_g = np.linalg.norm(g_full)
            if norm_h > 0 and norm_g > 0 and np.linalg.norm(h @ g_full - g_full @ h) / (norm_h * norm_g) < 1e-9:
                generators.append(g_full)
    else:
        generators = discover_one_body_symmetries(h)
    return generators


def get_symmetry_generators(h_op, impurity_orbitals, bath_states, n_orb=None, tol=1e-6):
    r"""Operators the CIPSI closure may use to complete a determinant's symmetry orbit.

    Returns the **total** spin ladder operators :math:`\hat S_+ = \sum_a c^\dagger_{a\uparrow}
    c_{a\downarrow}` and :math:`\hat S_-`, summed over impurity *and* bath, or ``[]`` when the
    spin labelling cannot be trusted.

    **Why these and not the commutant of the one-body block.** The previous implementation
    handed back ``discover_one_body_symmetries(h_imp)`` -- the continuous (Lie) commutant of the
    impurity one-body block, propagated to the bath. For a degenerate manifold that is the whole
    of ``u(n)``: arbitrary orbital *and* spin rotations inside it. A Slater-Condon ``U`` breaks
    essentially all of them. Measured on a cubic d-shell (``eg``/``t2g`` crystal field,
    ``F0,F2,F4 = 9,8,6``, bath propagation included): all 36 generators fail ``[H, g] = 0`` with
    residuals of 0.86 to 2.57 -- O(1), not roundoff. They are simply not symmetries.

    Nor could they be. The orbital symmetry of a cubic system is the octahedral point group
    ``O_h``, which is **discrete** and has no continuous generators, so it cannot be expressed as
    a Lie commutant at all. With spin-orbit coupling off, the continuous symmetries of a cubic
    shell with a Slater interaction are total charge and total spin -- and on that same model
    ``N_total`` and ``S_z_total`` both commute with the full ``H`` to exactly ``0``.

    Of those, only the ladder operators are *useful here*: the closure completes orbits by
    applying an operator to a determinant, and ``N`` and ``S_z`` are diagonal -- they return the
    determinant they were given and generate nothing. ``S_\pm`` flip one spin, so their orbit is
    the spin multiplet, which is what makes the retained space spin-adapted and the multiplet
    degeneracy exact rather than approximate.

    The pairing is validated before use (:func:`spin_pairs.spin_pairs_consistent_with_h` checks
    ``[h, S_z] = [h, S_+] = 0``), and that check sees only the one-body ``h``; the full ``H``
    including ``U`` and SOC is re-checked by the consumer
    (:meth:`cipsi_solver.CIPSISolver.expand`), which is the backstop for a spin-orbit-coupled
    model where ``S_\pm`` is genuinely not a symmetry.

    The one-body commutant is still offered alongside them, via
    :func:`_one_body_commutant_generators`: it is the only source of *orbital* generators, and
    it is exactly right when ``H`` has no two-body part. Over-production there is harmless
    precisely because the consumer gates on the full ``H``.

    Returns
    -------
    list
        ``[S_+, S_-]`` (``ManyBodyOperator``) followed by any one-body commutant generators
        (plain matrices). Empty when neither source yields anything trustworthy.
    """
    from impurityModel.ed.observables import make_spin_operators
    from impurityModel.ed.spin_pairs import bath_spin_pairs, impurity_spin_pairs, spin_pairs_consistent_with_h

    generators = []
    pairs = impurity_spin_pairs(impurity_orbitals) + bath_spin_pairs(bath_states)
    if pairs:
        if n_orb is None:
            n_orb = extract_tensors(h_op, two_body=False)[0].shape[0]
        if spin_pairs_consistent_with_h(h_op, pairs, n_orb, tol):
            s_plus, s_minus, _s_z = make_spin_operators(pairs)
            generators += [s_plus, s_minus]
        # Otherwise: spin-orbit coupling, a spin-polarized bath, or a non-standard orbital
        # ordering. The down/up pairing is then not fixed by the Hamiltonian, so S_+- built from
        # it would not be a symmetry -- better no closure than a wrong one.
    return generators + _one_body_commutant_generators(h_op, impurity_orbitals, bath_states)
